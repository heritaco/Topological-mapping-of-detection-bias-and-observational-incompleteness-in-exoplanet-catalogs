from __future__ import annotations

import math

import numpy as np
import pandas as pd


def method_distribution(frame: pd.DataFrame) -> pd.Series:
    methods = frame.get("discoverymethod", pd.Series(dtype="string")).astype("string").fillna("Unknown")
    counts = methods.value_counts(dropna=False).sort_index()
    total = counts.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return counts.astype(float) / float(total)


def l1_boundary(a: pd.Series, b: pd.Series) -> float:
    methods = sorted(set(a.index.tolist()) | set(b.index.tolist()))
    return float(sum(abs(float(a.get(method, 0.0)) - float(b.get(method, 0.0))) for method in methods))


def entropy_norm(distribution: pd.Series) -> float:
    if distribution.empty:
        return 0.0
    values = distribution[distribution > 0].astype(float)
    if values.empty:
        return 0.0
    entropy = -float(np.sum(values * np.log(values)))
    denom = math.log(len(distribution)) if len(distribution) > 1 else 1.0
    return float(entropy / denom) if denom > 0 else 0.0


def jensen_shannon_distance(a: pd.Series, b: pd.Series) -> float | None:
    methods = sorted(set(a.index.tolist()) | set(b.index.tolist()))
    if not methods:
        return None
    pa = np.array([float(a.get(method, 0.0)) for method in methods], dtype=float)
    pb = np.array([float(b.get(method, 0.0)) for method in methods], dtype=float)
    try:
        from scipy.spatial.distance import jensenshannon  # type: ignore
    except Exception:
        return None
    return float(jensenshannon(pa, pb))


def composition_records(case_id: str, node_id: str, region_type: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    counts = frame.get("discoverymethod", pd.Series(dtype="string")).astype("string").fillna("Unknown").value_counts(dropna=False)
    total = counts.sum()
    rows: list[dict[str, object]] = []
    for method, count in counts.sort_index().items():
        rows.append(
            {
                "case_id": case_id,
                "node_id": node_id,
                "region_type": region_type,
                "method": str(method),
                "count": int(count),
                "fraction": float(count / total) if total else 0.0,
            }
        )
    return rows


def summarize_method_context(case_id: str, node_id: str, node_frame: pd.DataFrame, n1_frame: pd.DataFrame, n2_frame: pd.DataFrame, epsilon: float) -> tuple[dict[str, object], pd.DataFrame]:
    p_node = method_distribution(node_frame)
    p_n1 = method_distribution(n1_frame)
    p_n2 = method_distribution(n2_frame)
    top_method = str(p_node.idxmax()) if not p_node.empty else "Unknown"
    top_fraction = float(p_node.max()) if not p_node.empty else 0.0
    methods = sorted(set(p_node.index.tolist()) | set(p_n1.index.tolist()) | set(p_n2.index.tolist()))
    summary: dict[str, object] = {
        "top_method": top_method,
        "top_method_fraction": top_fraction,
        "method_entropy_norm": entropy_norm(p_node),
        "method_l1_boundary_N1": l1_boundary(p_node, p_n1),
        "method_l1_boundary_N2": l1_boundary(p_node, p_n2),
        "method_js_boundary_N1": jensen_shannon_distance(p_node, p_n1),
        "method_js_boundary_N2": jensen_shannon_distance(p_node, p_n2),
    }
    for method in methods:
        pv = float(p_node.get(method, 0.0))
        pn1 = float(p_n1.get(method, 0.0))
        summary[f"D_N1__{method}"] = pn1 - pv
        summary[f"R_N1__{method}"] = (pv + epsilon) / (pn1 + epsilon)
    records = []
    records.extend(composition_records(case_id, node_id, "node", node_frame))
    records.extend(composition_records(case_id, node_id, "N1", n1_frame))
    records.extend(composition_records(case_id, node_id, "N2", n2_frame))
    return summary, pd.DataFrame(records)

