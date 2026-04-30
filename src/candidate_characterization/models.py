"""Probabilistic ML models with GPU-aware optional XGBoost paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from .features import add_engineered_features, build_xy, make_preprocessor, BASE_MODEL_FEATURES, model_features_from_feature_set
from .gpu import detect_accelerator, xgb_tree_params
from .labels import radius_class, RADIUS_CLASS_ORDER
from .utils import ensure_dir, write_json


class RandomForestQuantileRegressor(BaseEstimator, RegressorMixin):
    """Quantile regressor from the empirical distribution of tree predictions.

    This CPU fallback is stable and fast on small/medium tabular data. GPU acceleration
    is handled by the optional XGBoost path when CUDA is available.
    """

    def __init__(self, quantile: float, n_estimators: int = 96, random_state: int = 42):
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
            min_samples_leaf=3,
            max_features="sqrt",
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        self.model_ = self.model
        return self

    def predict(self, X):
        model = getattr(self, "model_", self.model)
        tree_preds = np.vstack([tree.predict(X) for tree in model.estimators_])
        return np.quantile(tree_preds, self.quantile, axis=0)


class QuantileModel:
    """Train one regressor per quantile with XGBoost GPU when available.

    If XGBoost quantile objective is unavailable or fails, this class falls back to
    scikit-learn GradientBoostingRegressor(loss='quantile').
    """

    def __init__(self, quantiles: Sequence[float], prefer_gpu: bool = True, random_state: int = 42):
        self.quantiles = list(quantiles)
        self.prefer_gpu = prefer_gpu
        self.random_state = random_state
        self.models: Dict[float, Pipeline] = {}
        self.backend_by_quantile: Dict[float, str] = {}
        self.features: List[str] = []
        self.target: str = ""

    def _make_xgb_regressor(self, q: float):
        from xgboost import XGBRegressor  # type: ignore

        params = xgb_tree_params(self.prefer_gpu, self.random_state)
        params.update(
            dict(
                n_estimators=160,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="reg:quantileerror",
                quantile_alpha=float(q),
            )
        )
        return XGBRegressor(**params)

    def _make_sklearn_regressor(self, q: float):
        return RandomForestQuantileRegressor(
            quantile=float(q),
            n_estimators=96,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, features: List[str], target: str) -> "QuantileModel":
        self.features = list(features)
        self.target = target
        for q in self.quantiles:
            fitted = False
            acc = detect_accelerator(self.prefer_gpu)
            if self.prefer_gpu and acc.torch_cuda_available and acc.xgboost_installed:
                try:
                    pipe = Pipeline([("pre", make_preprocessor()), ("model", self._make_xgb_regressor(q))])
                    pipe.fit(X[self.features], y)
                    self.models[q] = pipe
                    self.backend_by_quantile[q] = "xgboost_quantile_gpu_or_cpu"
                    fitted = True
                except Exception as exc:
                    warnings.warn(
                        f"XGBoost quantile fit failed for q={q}: {exc}. Falling back to sklearn quantile.",
                        RuntimeWarning,
                    )
            if not fitted:
                pipe = Pipeline([("pre", make_preprocessor()), ("model", self._make_sklearn_regressor(q))])
                pipe.fit(X[self.features], y)
                self.models[q] = pipe
                self.backend_by_quantile[q] = "sklearn_gradient_boosting_quantile_cpu"
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("QuantileModel has not been fitted.")
        Xp = X.copy()
        for f in self.features:
            if f not in Xp.columns:
                Xp[f] = np.nan
        preds = {}
        for q, model in self.models.items():
            preds[f"q{int(round(q * 100)):02d}"] = model.predict(Xp[self.features])
        pred_df = pd.DataFrame(preds, index=X.index)
        # Enforce monotone quantiles row-wise.
        ordered_cols = [f"q{int(round(q * 100)):02d}" for q in self.quantiles]
        arr = pred_df[ordered_cols].to_numpy(dtype=float)
        arr = np.sort(arr, axis=1)
        pred_df[ordered_cols] = arr
        return pred_df

    def metadata(self) -> Dict:
        return {
            "target": self.target,
            "features": self.features,
            "quantiles": self.quantiles,
            "backend_by_quantile": {str(k): v for k, v in self.backend_by_quantile.items()},
        }


class RadiusClassModel:
    def __init__(self, prefer_gpu: bool = True, random_state: int = 42, calibration_cv: int = 3):
        self.prefer_gpu = prefer_gpu
        self.random_state = random_state
        self.calibration_cv = calibration_cv
        self.features: List[str] = []
        self.label_encoder = LabelEncoder()
        self.model: Optional[Pipeline] = None
        self.backend = ""

    def _make_xgb_classifier(self, n_classes: int):
        from xgboost import XGBClassifier  # type: ignore

        params = xgb_tree_params(self.prefer_gpu, self.random_state)
        params.update(
            dict(
                n_estimators=160,
                max_depth=4,
                learning_rate=0.055,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=int(n_classes),
                eval_metric="mlogloss",
            )
        )
        return XGBClassifier(**params)

    def fit(self, X: pd.DataFrame, y_labels: pd.Series, features: List[str]) -> "RadiusClassModel":
        self.features = list(features)
        y_labels = y_labels.astype(str)
        valid = y_labels.notna() & (y_labels != "unknown")
        Xv = X.loc[valid, self.features].reset_index(drop=True)
        yv = y_labels.loc[valid].reset_index(drop=True)
        y_enc = self.label_encoder.fit_transform(yv)
        fitted = False
        if len(np.unique(y_enc)) < 2:
            raise ValueError("Need at least two radius classes to train classifier.")
        acc = detect_accelerator(self.prefer_gpu)
        if self.prefer_gpu and acc.torch_cuda_available and acc.xgboost_installed:
            try:
                base = self._make_xgb_classifier(n_classes=len(self.label_encoder.classes_))
                pipe = Pipeline([("pre", make_preprocessor()), ("model", base)])
                pipe.fit(Xv, y_enc)
                self.model = pipe
                self.backend = "xgboost_multiclass_gpu_or_cpu"
                fitted = True
            except Exception as exc:
                warnings.warn(f"XGBoost classifier fit failed: {exc}. Falling back to sklearn.", RuntimeWarning)
        if not fitted:
            clf = RandomForestClassifier(
                n_estimators=180,
                random_state=self.random_state,
                n_jobs=1,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                max_features="sqrt",
            )
            pipe = Pipeline([("pre", make_preprocessor()), ("model", clf)])
            pipe.fit(Xv, y_enc)
            self.backend = "sklearn_random_forest_cpu"
            self.model = pipe
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("RadiusClassModel has not been fitted.")
        Xp = X.copy()
        for f in self.features:
            if f not in Xp.columns:
                Xp[f] = np.nan
        proba = self.model.predict_proba(Xp[self.features])
        class_names = list(self.label_encoder.inverse_transform(np.arange(proba.shape[1])))
        out = pd.DataFrame(proba, columns=[f"prob_{c}" for c in class_names], index=X.index)
        # Add absent canonical classes as zeros for stable output schema.
        for c in RADIUS_CLASS_ORDER:
            col = f"prob_{c}"
            if col not in out.columns:
                out[col] = 0.0
        return out

    def metadata(self) -> Dict:
        return {"features": self.features, "classes": list(self.label_encoder.classes_), "backend": self.backend}


@dataclass
class TrainedCharacterizationModels:
    radius_model: QuantileModel
    mass_model: QuantileModel
    class_model: RadiusClassModel
    model_features: List[str]
    metadata: Dict

    def save(self, model_dir: Path) -> None:
        ensure_dir(model_dir)
        joblib.dump(self, model_dir / "candidate_characterization_models.joblib")
        write_json(model_dir / "model_metadata.json", self.metadata)

    @staticmethod
    def load(model_dir: Path) -> "TrainedCharacterizationModels":
        return joblib.load(model_dir / "candidate_characterization_models.joblib")


def train_models(
    catalog: pd.DataFrame,
    quantiles: Sequence[float],
    prefer_gpu: bool = True,
    random_state: int = 42,
    calibration_cv: int = 3,
    feature_set: str | None = None,
    registry_path: str = "configs/features/feature_registry.yaml",
    feature_sets_path: str = "configs/features/feature_sets.yaml",
    allow_audit_features: bool = False,
    allow_observed_diagnostic: bool = False,
) -> TrainedCharacterizationModels:
    df = add_engineered_features(catalog)
    radius_features = None
    mass_features = None
    leakage_warnings: list[str] = []
    if feature_set:
        radius_features, radius_warnings = model_features_from_feature_set(
            df,
            feature_set,
            target="log_pl_rade",
            registry_path=registry_path,
            feature_sets_path=feature_sets_path,
            allow_audit_features=allow_audit_features,
            allow_observed_diagnostic=allow_observed_diagnostic,
        )
        mass_features, mass_warnings = model_features_from_feature_set(
            df,
            feature_set,
            target="log_pl_bmasse",
            registry_path=registry_path,
            feature_sets_path=feature_sets_path,
            allow_audit_features=allow_audit_features,
            allow_observed_diagnostic=allow_observed_diagnostic,
        )
        leakage_warnings = radius_warnings + [warning for warning in mass_warnings if warning not in radius_warnings]
    X_r, y_r, features = build_xy(df, "log_pl_rade", include_topological_context=False, feature_names=radius_features)
    X_m, y_m, features_m = build_xy(df, "log_pl_bmasse", include_topological_context=False, feature_names=mass_features)
    features = [f for f in features if f in features_m]
    X_r = X_r[features]
    X_m = X_m[features]

    if len(X_r) < 30 or len(X_m) < 30:
        raise ValueError(f"Not enough training rows: radius={len(X_r)}, mass={len(X_m)}")

    radius_model = QuantileModel(quantiles=quantiles, prefer_gpu=prefer_gpu, random_state=random_state)
    radius_model.fit(X_r, y_r, features, "log_pl_rade")

    mass_model = QuantileModel(quantiles=quantiles, prefer_gpu=prefer_gpu, random_state=random_state)
    mass_model.fit(X_m, y_m, features, "log_pl_bmasse")

    # Train radius-class classifier using the same leakage-safe feature set.
    class_df = df.copy()
    labels_all = radius_class(class_df.get("pl_rade", pd.Series(np.nan, index=class_df.index)))
    feature_matrix = class_df[features].apply(pd.to_numeric, errors="coerce")
    class_features = [c for c in features if feature_matrix[c].notna().any()]
    feature_matrix = feature_matrix[class_features]
    valid_class = (labels_all != "unknown") & (feature_matrix.notna().sum(axis=1) >= 2)
    X_c = feature_matrix.loc[valid_class].reset_index(drop=True)
    labels_c = labels_all.loc[valid_class].reset_index(drop=True)
    class_model = RadiusClassModel(prefer_gpu=prefer_gpu, random_state=random_state, calibration_cv=calibration_cv)
    class_model.fit(X_c[class_features], labels_c, class_features)

    metadata = {
        "accelerator": detect_accelerator(prefer_gpu).to_dict(),
        "radius_model": radius_model.metadata(),
        "mass_model": mass_model.metadata(),
        "class_model": class_model.metadata(),
        "n_train_radius": int(len(X_r)),
        "n_train_mass": int(len(X_m)),
        "model_features": features,
        "feature_set": feature_set or "legacy_base_model_features",
        "leakage_warnings": leakage_warnings,
    }
    return TrainedCharacterizationModels(radius_model, mass_model, class_model, features, metadata)
