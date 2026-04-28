from __future__ import annotations

import pandas as pd


def select_case_nodes(
    shadow_metrics: pd.DataFrame,
    top_candidates: pd.DataFrame,
    config_id: str,
    requested_node_ids: list[str],
    required_method: str,
    n_cases: int,
    min_members: int,
    warnings: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    metrics = shadow_metrics[shadow_metrics["config_id"].astype(str) == str(config_id)].copy()
    available_nodes = set(metrics["node_id"].astype(str).tolist())
    selected: list[str] = []
    replacements: list[dict[str, str]] = []
    for node_id in requested_node_ids:
        if node_id in available_nodes:
            selected.append(node_id)
        else:
            warnings.append(f"WARNING: no existe el nodo solicitado {node_id} en {config_id}; se buscara reemplazo.")

    if len(selected) >= n_cases:
        return selected[:n_cases], replacements

    candidates = top_candidates[top_candidates["config_id"].astype(str) == str(config_id)].copy()
    if "top_method" in candidates.columns:
        candidates = candidates[candidates["top_method"].astype(str) == required_method]
    if "n_members" in candidates.columns:
        candidates = candidates[pd.to_numeric(candidates["n_members"], errors="coerce").fillna(0) >= min_members]
    if "mean_imputation_fraction" in candidates.columns:
        candidates = candidates.sort_values(
            by=["shadow_score", "n_members", "mean_imputation_fraction"],
            ascending=[False, False, True],
        )
    else:
        candidates = candidates.sort_values(by=["shadow_score", "n_members"], ascending=[False, False])

    for _, row in candidates.iterrows():
        node_id = str(row["node_id"])
        if node_id in selected:
            continue
        replaced = requested_node_ids[len(selected)] if len(selected) < len(requested_node_ids) else "unfilled_slot"
        replacements.append(
            {
                "requested_node_id": replaced,
                "replacement_node_id": node_id,
                "reason": "requested node missing or insufficient; replaced from top_shadow_candidates filtered to RV and adequate size",
            }
        )
        selected.append(node_id)
        if len(selected) >= n_cases:
            break

    selected = selected[:n_cases]
    if len(selected) < n_cases:
        warnings.append(f"WARNING: solo fue posible seleccionar {len(selected)} casos para {config_id}.")
    return selected, replacements

