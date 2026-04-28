from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def shannon_entropy(counts: Iterable[float]) -> float:
    values = np.asarray(list(counts), dtype=float)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values[values > 0] / total
    return float(-(probs * np.log(probs)).sum())


def normalized_entropy(counts: Iterable[float], universe_k: int) -> float:
    if universe_k <= 1:
        return 0.0
    return float(shannon_entropy(counts) / np.log(float(universe_k)))


def method_l1_boundary(node_probs: np.ndarray, neighbor_probs: np.ndarray) -> float:
    return float(np.abs(node_probs - neighbor_probs).sum())


def method_js_boundary(node_probs: np.ndarray, neighbor_probs: np.ndarray) -> float:
    try:
        from scipy.spatial.distance import jensenshannon
    except ImportError as exc:
        raise RuntimeError("scipy no esta disponible para calcular Jensen-Shannon.") from exc
    return float(jensenshannon(node_probs, neighbor_probs, base=np.e))


def size_weight(n_members: float, max_n_members: float) -> float:
    if max_n_members <= 0 or n_members <= 0:
        return 0.0
    return float(np.log1p(n_members) / np.log1p(max_n_members))


def compute_shadow_scores(frame: pd.DataFrame, has_imputation: bool) -> pd.DataFrame:
    out = frame.copy()
    max_n = float(pd.to_numeric(out["n_members"], errors="coerce").max()) if not out.empty else 0.0
    out["size_weight"] = pd.to_numeric(out["n_members"], errors="coerce").fillna(0.0).apply(lambda value: size_weight(value, max_n))
    purity = pd.to_numeric(out["top_method_fraction"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    entropy_factor = 1.0 - pd.to_numeric(out["method_entropy_norm"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    boundary = pd.to_numeric(out["method_l1_boundary"], errors="coerce").fillna(0.0).clip(lower=0.0)
    raw = purity * entropy_factor * boundary
    out["shadow_score_raw"] = raw
    out["shadow_score_no_imputation"] = raw * out["size_weight"]
    if has_imputation:
        imputation_factor = 1.0 - pd.to_numeric(out["mean_imputation_fraction"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        out["shadow_score"] = raw * imputation_factor * out["size_weight"]
    else:
        out["shadow_score"] = out["shadow_score_no_imputation"]
    out.loc[out["no_neighbors"].fillna(False), ["shadow_score", "shadow_score_raw", "shadow_score_no_imputation"]] = np.nan
    return out


def classify_shadow_nodes(
    frame: pd.DataFrame,
    percentile: float,
    imputation_threshold: float,
    min_members: int,
) -> pd.DataFrame:
    out = frame.copy()
    valid = pd.to_numeric(out["shadow_score"], errors="coerce").dropna()
    threshold = float(np.nanpercentile(valid, percentile)) if not valid.empty else np.inf
    out["shadow_percentile_threshold"] = threshold
    high = pd.to_numeric(out["shadow_score"], errors="coerce") >= threshold
    low_imp = pd.to_numeric(out["mean_imputation_fraction"], errors="coerce").fillna(0.0) <= imputation_threshold
    enough_size = pd.to_numeric(out["n_members"], errors="coerce").fillna(0) >= min_members
    no_neighbors = out["no_neighbors"].fillna(False).astype(bool)
    classes = np.full(len(out), "mixed_or_low_shadow", dtype=object)
    classes[no_neighbors.to_numpy()] = "no_neighbor_information"
    classes[(high & low_imp & enough_size & ~no_neighbors).to_numpy()] = "high_confidence_shadow"
    classes[(high & ~enough_size & ~no_neighbors).to_numpy()] = "small_sample_shadow"
    classes[(high & ~low_imp & ~no_neighbors).to_numpy()] = "imputation_sensitive_shadow"
    out["shadow_class"] = classes
    return out

