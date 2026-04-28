from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .graph_neighbors import component_lookup, neighbor_map
from .physical_gaps import add_physical_neighbor_gaps, available_physical_variables
from .shadow_metrics import method_js_boundary, method_l1_boundary, normalized_entropy, shannon_entropy


def _derive_trace_fraction(membership: pd.DataFrame, suffix: str) -> pd.Series:
    columns = [column for column in membership.columns if column.endswith(suffix)]
    if not columns:
        return pd.Series(dtype=float)
    values = membership[columns].apply(lambda col: col.astype("string").str.lower().isin(["true", "1", "yes"]))
    return values.mean(axis=1)


def _safe_row_number(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return float("nan")
    return float(pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0])


def _dominant_direction(method: str, row: pd.Series) -> tuple[str, str]:
    method_norm = str(method).strip().lower()
    parts: list[str] = []
    direction = "composicion observacional no homogenea; requiere revision manual"
    if method_norm == "transit":
        direction = "menor radio y periodos mas largos"
        parts.append("candidato a incompletitud observacional hacia planetas de menor radio y orbitas mas largas")
        if pd.notna(row.get("mean_pl_rade")) and pd.notna(row.get("neighbor_mean_pl_rade")) and row["mean_pl_rade"] > row["neighbor_mean_pl_rade"]:
            parts.append("el radio medio supera al vecindario, compatible con sesgo hacia radios detectables")
        if pd.notna(row.get("mean_pl_orbper")) and pd.notna(row.get("neighbor_mean_pl_orbper")) and row["mean_pl_orbper"] < row["neighbor_mean_pl_orbper"]:
            parts.append("el periodo medio es menor que el del vecindario, compatible con preferencia por orbitas cortas")
    elif method_norm == "radial velocity":
        direction = "masas menores y senales radiales mas debiles"
        parts.append("candidato a incompletitud observacional hacia masas menores y senales radiales debiles")
        if pd.notna(row.get("mean_pl_bmasse")) and pd.notna(row.get("neighbor_mean_pl_bmasse")) and row["mean_pl_bmasse"] > row["neighbor_mean_pl_bmasse"]:
            parts.append("la masa media supera al vecindario, compatible con sesgo hacia planetas masivos")
    elif method_norm == "imaging":
        direction = "objetos menos masivos, menos luminosos o menos jovenes en separaciones menos favorables"
        parts.append("candidato a incompletitud en objetos menos masivos, menos luminosos o menos jovenes")
        if pd.notna(row.get("mean_pl_orbsmax")) and pd.notna(row.get("neighbor_mean_pl_orbsmax")) and row["mean_pl_orbsmax"] > row["neighbor_mean_pl_orbsmax"]:
            parts.append("la separacion orbital media es amplia respecto al vecindario")
    elif method_norm == "microlensing":
        direction = "ventana geometrica de deteccion distinta"
        parts.append("posible frontera de seleccion ligada a la geometria de microlensing, no a una clase fisica directa")
    else:
        parts.append("la comunidad muestra composicion observacional no homogenea, pero requiere revision manual")
    text = (
        f"Nodo {row.get('node_id')} dominado por {method}; {parts[0]}. "
        f"Shadow={row.get('shadow_score', np.nan):.3f}, n={int(row.get('n_members', 0))}, "
        f"imputacion media={row.get('mean_imputation_fraction', np.nan):.3f}."
    )
    if len(parts) > 1:
        text += " " + " ".join(parts[1:]) + "."
    return direction, text


def add_interpretations(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    directions: list[str] = []
    texts: list[str] = []
    for _, row in out.iterrows():
        direction, text = _dominant_direction(str(row.get("top_method", "Unknown")), row)
        directions.append(direction)
        texts.append(text)
    out["expected_incompleteness_direction"] = directions
    out["interpretation_text"] = texts
    return out


def build_node_shadow_profiles(
    config_id: str,
    membership: pd.DataFrame,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    requested_physical_variables: list[str],
    peripheral_degree_threshold: int,
    peripheral_component_max_nodes: int,
    epsilon: float,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if membership.empty:
        raise RuntimeError(f"Falta membresia nodo-planeta para {config_id}.")
    if "node_id" not in membership.columns:
        raise RuntimeError(f"La membresia de {config_id} no contiene node_id.")
    if "discoverymethod" not in membership.columns:
        raise RuntimeError(f"La membresia de {config_id} no contiene discoverymethod.")

    nodes = node_table.copy()
    nodes["node_id"] = nodes["node_id"].astype(str)
    node_order = nodes["node_id"].astype(str).tolist()
    membership = membership.copy()
    membership["node_id"] = membership["node_id"].astype(str)
    membership["discoverymethod"] = membership["discoverymethod"].astype("string").fillna("Unknown")
    membership = membership[membership["node_id"].isin(node_order)].copy()
    if membership.empty:
        raise RuntimeError(f"La membresia de {config_id} no coincide con la tabla de nodos.")

    methods = sorted(membership["discoverymethod"].astype(str).unique().tolist())
    count_df = pd.crosstab(membership["node_id"], membership["discoverymethod"]).reindex(index=node_order, columns=methods, fill_value=0)
    frac_df = count_df.div(count_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    neighbors = neighbor_map(node_order, edge_table)
    comp_lookup, comp_sizes = component_lookup(node_order, edge_table)

    available_physical = available_physical_variables(membership, requested_physical_variables, warnings, config_id)
    physical_means = membership.groupby("node_id")[available_physical].mean(numeric_only=True) if available_physical else pd.DataFrame(index=count_df.index)
    imputation_fraction = _derive_trace_fraction(membership, "_was_imputed")
    derived_fraction = _derive_trace_fraction(membership, "_was_physically_derived")
    if not imputation_fraction.empty:
        membership["_row_imputation_fraction"] = imputation_fraction
        derived_by_node = membership.assign(_row_physically_derived_fraction=derived_fraction).groupby("node_id")["_row_physically_derived_fraction"].mean()
        imputed_by_node = membership.groupby("node_id")["_row_imputation_fraction"].mean()
    else:
        imputed_by_node = pd.Series(dtype=float)
        derived_by_node = pd.Series(dtype=float)

    node_lookup = nodes.set_index("node_id", drop=False)
    rows: list[dict[str, Any]] = []
    neighbor_rows: dict[str, dict[str, Any]] = {}
    js_available = True
    for node_id in node_order:
        counts = count_df.loc[node_id].to_numpy(dtype=float)
        fractions = frac_df.loc[node_id].to_numpy(dtype=float)
        top_method = str(frac_df.loc[node_id].idxmax()) if counts.sum() > 0 else "Unknown"
        top_fraction = float(frac_df.loc[node_id].max()) if counts.sum() > 0 else 0.0
        neighbor_ids = sorted(neighbors.get(node_id, set()))
        no_neighbors = len(neighbor_ids) == 0
        if no_neighbors:
            neighbor_counts = np.zeros(len(methods), dtype=float)
            neighbor_probs = np.zeros(len(methods), dtype=float)
        else:
            neighbor_counts = count_df.loc[neighbor_ids].sum(axis=0).to_numpy(dtype=float)
            total = neighbor_counts.sum()
            neighbor_probs = neighbor_counts / total if total > 0 else np.zeros(len(methods), dtype=float)
        l1 = np.nan if no_neighbors else method_l1_boundary(fractions, neighbor_probs)
        try:
            js = np.nan if no_neighbors else method_js_boundary(fractions, neighbor_probs)
        except RuntimeError:
            js_available = False
            js = np.nan
        base_row = node_lookup.loc[node_id]
        degree = int(len(neighbor_ids))
        component_id = base_row.get("component_id", comp_lookup.get(node_id, -1))
        component_size = comp_sizes.get(int(component_id), np.nan) if pd.notna(component_id) else np.nan
        row = {
            "config_id": config_id,
            "node_id": node_id,
            "n_members": int(counts.sum()),
            "top_method": top_method,
            "top_method_fraction": top_fraction,
            "method_entropy": shannon_entropy(counts),
            "method_entropy_norm": normalized_entropy(counts, len(methods)),
            "mean_imputation_fraction": _safe_row_number(base_row, "mean_imputation_fraction"),
            "mean_physically_derived_fraction": _safe_row_number(base_row, "physically_derived_fraction"),
            "degree": degree,
            "component_id": component_id,
            "component_n_nodes": component_size,
            "lens_1_mean": _safe_row_number(base_row, "lens_1_mean"),
            "lens_2_mean": _safe_row_number(base_row, "lens_2_mean"),
            "is_peripheral": bool(degree <= peripheral_degree_threshold or (pd.notna(component_size) and component_size <= peripheral_component_max_nodes)),
            "no_neighbors": bool(no_neighbors),
            "method_l1_boundary": l1,
            "method_js_boundary": js,
        }
        if pd.isna(row["mean_imputation_fraction"]):
            row["mean_imputation_fraction"] = float(imputed_by_node.get(node_id, np.nan))
        if pd.isna(row["mean_physically_derived_fraction"]):
            row["mean_physically_derived_fraction"] = float(derived_by_node.get(node_id, np.nan))
        for method, p_node, p_neigh in zip(methods, fractions, neighbor_probs):
            row[f"p_node__{method}"] = float(p_node)
            row[f"p_neighbors__{method}"] = float(p_neigh)
            row[f"D__{method}"] = float(p_neigh - p_node) if not no_neighbors else np.nan
            row[f"R__{method}"] = float((p_node + epsilon) / (p_neigh + epsilon)) if not no_neighbors else np.nan
        for variable in available_physical:
            row[f"mean_{variable}"] = float(physical_means.loc[node_id, variable]) if node_id in physical_means.index else np.nan
        rows.append(row)

        neigh_row: dict[str, Any] = {}
        for variable in available_physical:
            if no_neighbors:
                neigh_row[f"neighbor_mean_{variable}"] = np.nan
            else:
                values = physical_means.reindex(neighbor_ids)[variable]
                neigh_row[f"neighbor_mean_{variable}"] = float(pd.to_numeric(values, errors="coerce").mean())
        neighbor_rows[node_id] = neigh_row

    if not js_available:
        warnings.append("WARNING: scipy no esta disponible; method_js_boundary queda como NaN.")
    profiles = pd.DataFrame(rows).set_index("node_id", drop=False)
    neighbor_profile = pd.DataFrame.from_dict(neighbor_rows, orient="index")
    profiles = add_physical_neighbor_gaps(profiles, neighbor_profile, available_physical).reset_index(drop=True)
    metadata = {
        "method_universe": methods,
        "physical_variables_available": available_physical,
        "n_nodes": int(len(profiles)),
        "n_edges": int(len(edge_table)),
        "has_imputation": bool(profiles["mean_imputation_fraction"].notna().any()),
    }
    return profiles, metadata
