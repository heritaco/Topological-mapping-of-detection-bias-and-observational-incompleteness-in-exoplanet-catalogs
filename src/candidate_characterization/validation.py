"""Validation utilities for property characterization."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from .features import add_engineered_features, build_xy
from .labels import radius_class
from .models import train_models
from .predict import predict_candidates
from .utils import ensure_dir


def interval_coverage(y_true: pd.Series, q_low: pd.Series, q_high: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    lo = pd.to_numeric(q_low, errors="coerce")
    hi = pd.to_numeric(q_high, errors="coerce")
    mask = y.notna() & lo.notna() & hi.notna()
    if mask.sum() == 0:
        return float("nan")
    return float(((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])).mean())


def make_validation_split(catalog: pd.DataFrame, mode: str = "random", test_size: float = 0.2, random_state: int = 42):
    df = add_engineered_features(catalog).copy()
    valid = df["pl_rade"].notna() & df["pl_bmasse"].notna() & df["pl_orbper"].notna()
    df = df.loc[valid].reset_index(drop=True)
    if mode == "temporal" and "disc_year" in df.columns:
        years = pd.to_numeric(df["disc_year"], errors="coerce")
        cutoff = np.nanquantile(years, 1.0 - test_size)
        train = df.loc[years <= cutoff].reset_index(drop=True)
        test = df.loc[years > cutoff].reset_index(drop=True)
    elif mode == "multiplanet" and "hostname" in df.columns:
        counts = df.groupby("hostname")["hostname"].transform("size")
        multi = df.loc[counts >= 2].copy()
        if len(multi) >= 30:
            test = multi.groupby("hostname", group_keys=False).sample(n=1, random_state=random_state)
            train = df.drop(index=test.index).reset_index(drop=True)
            test = test.reset_index(drop=True)
        else:
            train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def validate_property_models(
    catalog: pd.DataFrame,
    quantiles: Sequence[float],
    prefer_gpu: bool = True,
    random_state: int = 42,
    mode: str = "random",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = make_validation_split(catalog, mode=mode, test_size=test_size, random_state=random_state)
    models = train_models(train, quantiles=quantiles, prefer_gpu=prefer_gpu, random_state=random_state)
    # Use observed planets as pseudo-candidates; target physical columns are ignored by model features.
    pred, _ = predict_candidates(test, train, models, quantiles=quantiles, analog_k=30)

    rows = []
    if "pl_rade_q50" in pred.columns:
        rows.append({
            "target": "pl_rade",
            "mode": mode,
            "n_test": int(len(test)),
            "mae_q50": float(mean_absolute_error(test["pl_rade"], pred["pl_rade_q50"])),
            "coverage_q05_q95": interval_coverage(test["pl_rade"], pred.get("pl_rade_q05"), pred.get("pl_rade_q95")),
        })
    if "pl_bmasse_q50" in pred.columns:
        rows.append({
            "target": "pl_bmasse",
            "mode": mode,
            "n_test": int(len(test)),
            "mae_q50": float(mean_absolute_error(test["pl_bmasse"], pred["pl_bmasse_q50"])),
            "coverage_q05_q95": interval_coverage(test["pl_bmasse"], pred.get("pl_bmasse_q05"), pred.get("pl_bmasse_q95")),
        })
    true_class = radius_class(test["pl_rade"]).reset_index(drop=True)
    if "predicted_radius_class" in pred.columns:
        rows.append({
            "target": "radius_class",
            "mode": mode,
            "n_test": int(len(test)),
            "accuracy": float(accuracy_score(true_class, pred["predicted_radius_class"])),
        })

    detail_cols = ["pl_name", "hostname", "pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse"]
    detail = pd.concat([test[[c for c in detail_cols if c in test.columns]].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    return pd.DataFrame(rows), detail


def write_validation_outputs(metrics: pd.DataFrame, details: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    table_dir = ensure_dir(output_dir / "tables")
    metrics_path = table_dir / "validation_metrics.csv"
    details_path = table_dir / "validation_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    details.to_csv(details_path, index=False)
    return {"validation_metrics": metrics_path, "validation_predictions": details_path}
