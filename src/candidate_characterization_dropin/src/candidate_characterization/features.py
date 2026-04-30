"""Feature engineering for candidate property characterization."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .utils import safe_log10, safe_numeric, unique_preserve_order
from .labels import radius_class, orbit_class, thermal_class

# Features that can plausibly be known for a hypothetical missing planet before
# predicting mass/radius. Do not include pl_rade, pl_bmasse or pl_dens here.
BASE_MODEL_FEATURES = [
    "log_pl_orbper",
    "log_pl_orbsmax",
    "log_pl_insol",
    "pl_eqt",
    "st_mass",
    "st_rad",
    "st_teff",
    "st_lum",
    "st_met",
    "st_logg",
    "sy_dist",
    "sy_pnum",
    "log_gap_ratio_inner",
    "log_gap_ratio_outer",
    "log_gap_width",
]

TOPOLOGICAL_CONTEXT_FEATURES = [
    "candidate_score",
    "topology_score",
    "gap_score",
    "toi",
    "ati",
    "shadow_score",
    "i_r3",
    "c_phys",
    "s_net",
]

TARGETS = {
    "log_pl_rade": "pl_rade",
    "log_pl_bmasse": "pl_bmasse",
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["pl_orbper", "pl_orbsmax", "pl_insol"]:
        if c in out.columns:
            out[f"log_{c}"] = safe_log10(out[c])
        else:
            out[f"log_{c}"] = np.nan
    for c in ["pl_rade", "pl_bmasse", "pl_dens"]:
        if c in out.columns:
            out[f"log_{c}"] = safe_log10(out[c])
    for c in ["pl_eqt", "st_mass", "st_rad", "st_teff", "st_lum", "st_met", "st_logg", "sy_dist", "sy_pnum"]:
        if c in out.columns:
            out[c] = safe_numeric(out[c])
        else:
            out[c] = np.nan

    # Ensure gap/topological columns exist so model matrices have stable schemas.
    for c in ["log_gap_ratio_inner", "log_gap_ratio_outer", "log_gap_width"] + TOPOLOGICAL_CONTEXT_FEATURES:
        if c not in out.columns:
            out[c] = np.nan
        else:
            out[c] = safe_numeric(out[c])

    if "radius_class" not in out.columns and "pl_rade" in out.columns:
        out["radius_class"] = radius_class(out["pl_rade"])
    if "orbit_class" not in out.columns and "pl_orbper" in out.columns:
        out["orbit_class"] = orbit_class(out["pl_orbper"])
    if "thermal_class" not in out.columns and "pl_eqt" in out.columns:
        out["thermal_class"] = thermal_class(out["pl_eqt"])
    return out


def available_model_features(df: pd.DataFrame, include_topological_context: bool = False) -> List[str]:
    candidates = BASE_MODEL_FEATURES + (TOPOLOGICAL_CONTEXT_FEATURES if include_topological_context else [])
    return [c for c in candidates if c in df.columns]


def build_xy(
    df: pd.DataFrame,
    target: str,
    include_topological_context: bool = False,
    min_finite_features: int = 2,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = add_engineered_features(df)
    features = available_model_features(df, include_topological_context=include_topological_context)
    if target not in df.columns:
        raise KeyError(f"Target {target!r} is not available. Existing columns: {list(df.columns)[:20]}...")
    y = pd.to_numeric(df[target], errors="coerce")
    X = df[features].apply(pd.to_numeric, errors="coerce")
    # Drop features that are entirely absent in the current table. This avoids
    # imputer warnings and keeps training/prediction schemas compact.
    features = [c for c in features if X[c].notna().any()]
    X = X[features]
    valid = y.notna() & np.isfinite(y)
    valid &= X.notna().sum(axis=1) >= min_finite_features
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), features


def make_preprocessor() -> Pipeline:
    try:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # older scikit-learn
        imputer = SimpleImputer(strategy="median")
    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", RobustScaler()),
        ]
    )


def candidate_identifier_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in ["candidate_id", "hostname", "node_id"] if c in df.columns]
