from __future__ import annotations

import numpy as np
import pandas as pd


def build_observational_priority_candidates(anchors: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    if anchors.empty:
        return pd.DataFrame()
    df = anchors.copy()
    for column in ["ATI_conservative", "TOI", "stable_deficit_score", "r3_imputation_score", "anchor_representativeness", "radius_sensitivity_score"]:
        if column not in df.columns:
            df[column] = np.nan
    df["observational_priority_score"] = (
        0.35 * _normalize(df["ATI_conservative"])
        + 0.25 * _normalize(df["TOI"])
        + 0.20 * _normalize(df["stable_deficit_score"])
        + 0.10 * (1 - pd.to_numeric(df["r3_imputation_score"], errors="coerce").fillna(0).clip(lower=0, upper=1))
        + 0.10 * _normalize(df["anchor_representativeness"])
        - 0.10 * _normalize(df["radius_sensitivity_score"])
    )
    df["reason_for_priority"] = df.apply(_priority_reason, axis=1)
    df["caution_text"] = df.apply(_priority_caution, axis=1)
    df = df.sort_values("observational_priority_score", ascending=False).head(top_n).reset_index(drop=True)
    keep = [
        "anchor_pl_name",
        "node_id",
        "observational_priority_score",
        "ATI_conservative",
        "TOI",
        "stable_deficit_score",
        "I_R3",
        "method",
        "expected_incompleteness_direction",
        "reason_for_priority",
        "caution_text",
    ]
    for column in keep:
        if column not in df.columns:
            df[column] = np.nan
    return df[keep]


def build_technical_audit_cases(anchors: pd.DataFrame, deficits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if anchors.empty:
        return pd.DataFrame(rows)
    df = anchors.copy()
    repeated = (
        df.groupby("anchor_pl_name", dropna=False)["node_id"]
        .nunique()
        .reset_index(name="n_nodes_as_anchor")
    )
    df = df.merge(repeated, on="anchor_pl_name", how="left")
    top_ati_cut = pd.to_numeric(df["ATI_original"], errors="coerce").quantile(0.90) if "ATI_original" in df.columns else np.nan
    for _, row in df.iterrows():
        if pd.notna(top_ati_cut) and float(row.get("ATI_original", 0) or 0) >= top_ati_cut and str(row.get("deficit_stability_class", "")) in {"radius_sensitive_deficit", "unstable_due_to_large_radius"}:
            rows.append(_issue_row(row, "high_ATI_but_radius_sensitive", "ATI alto con deficit sensible al radio."))
        if float(row.get("delta_rel_neighbors_best", 0) or 0) > 0.10 and float(row.get("delta_rel_mean", 0) or 0) <= 0.05:
            rows.append(_issue_row(row, "high_delta_best_low_delta_mean", "El deficit maximo supera al promedio y puede inflar la lectura."))
        if float(row.get("r3_imputation_score", 0) or 0) > 0.20:
            rows.append(_issue_row(row, "high_imputation", "La imputacion del ancla no es despreciable."))
        if float(row.get("n_members", 0) or 0) < 10 or float(row.get("S_net", 1) or 1) < 0.30:
            rows.append(_issue_row(row, "small_node_support", "El soporte de nodo o de red es pequeno."))
        if float(row.get("n_nodes_as_anchor", 1) or 1) >= 2:
            rows.append(_issue_row(row, "repeated_anchor_needs_context", "El ancla aparece en varios nodos y necesita contexto de solapamiento Mapper."))
    if not deficits.empty and "deficit_formula_check" in deficits.columns:
        mismatch = deficits[deficits["deficit_formula_check"].astype(str) != "ok"]
        for _, row in mismatch.iterrows():
            rows.append(
                {
                    "issue_type": "possible_units_issue",
                    "anchor_pl_name": row.get("anchor_pl_name"),
                    "node_id": row.get("node_id"),
                    "ATI_original": np.nan,
                    "ATI_conservative": np.nan,
                    "delta_rel_best": row.get("delta_rel_neighbors_recomputed"),
                    "delta_rel_mean": np.nan,
                    "r3_imputation_score": np.nan,
                    "n_members": np.nan,
                    "note": "La formula del deficit no coincide con las columnas originales y se recomputo.",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["issue_type", "anchor_pl_name", "node_id"]).reset_index(drop=True)


def build_final_future_work_cases(
    regions: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    candidate_n: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not regions.empty:
        best_region = regions.sort_values(["TOI_original", "TOI_rank_mean_sensitivity"], ascending=[False, True]).iloc[0]
        rows.append(_case_row("top_toi_region", best_region, why_selected="Region top por TOI original y con lectura regional fuerte.", how_to_present="Usar para explicar el indice regional y sus factores.", caution_text="Prioriza una zona Mapper; no equivale a confirmar objetos ausentes."))
    if not anchors.empty:
        sort_original = [column for column in ["ATI_original", "rank_ATI_original"] if column in anchors.columns]
        ascending_original = [False if column == "ATI_original" else True for column in sort_original]
        best_original = anchors.sort_values(sort_original, ascending=ascending_original).iloc[0]
        rows.append(_case_row("top_ati_original_anchor", best_original, why_selected="Ancla con ATI original mas alto.", how_to_present="Usar para mostrar como el ranking original prioriza un caso local.", caution_text="Puede ser sensible al radio y debe leerse junto al perfil por escalas."))
        sort_conservative = [column for column in ["ATI_conservative", "rank_ATI_conservative"] if column in anchors.columns]
        ascending_conservative = [False if column == "ATI_conservative" else True for column in sort_conservative]
        best_conservative = anchors.sort_values(sort_conservative, ascending=ascending_conservative).iloc[0]
        rows.append(_case_row("top_ati_conservative_anchor", best_conservative, why_selected="Ancla con ATI conservador mas alto tras penalizar inestabilidad.", how_to_present="Usar para mostrar la version mas prudente del ranking local.", caution_text="Sigue siendo una prioridad exploratoria, no una inferencia de objetos ausentes."))

        repeated = _repeated_anchor_case(anchors)
        if repeated is not None:
            rows.append(repeated)

        stable = anchors[anchors["deficit_stability_class"].astype(str).isin(["stable_positive_deficit", "small_but_stable_deficit"])]
        if not stable.empty:
            best_stable = stable.sort_values(["stable_deficit_score", "ATI_conservative"], ascending=[False, False]).iloc[0]
            rows.append(_case_row("stable_deficit_anchor", best_stable, why_selected="Ancla con deficit positivo mas estable en los tres radios.", how_to_present="Usar para contrastar un caso estable frente a uno sensible al radio.", caution_text="El deficit es pequeno o moderado y debe interpretarse como priorizacion topologica prudente."))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.head(candidate_n)


def _repeated_anchor_case(anchors: pd.DataFrame) -> dict[str, object] | None:
    repeated = (
        anchors.groupby("anchor_pl_name", dropna=False)
        .agg(
            n_nodes_as_anchor=("node_id", "nunique"),
            nodes_list=("node_id", lambda s: ", ".join(sorted({str(value) for value in s if pd.notna(value)}))),
            ATI_conservative=("ATI_conservative", "max"),
            ATI_original=("ATI_original", "max"),
        )
        .reset_index()
    )
    repeated = repeated[repeated["n_nodes_as_anchor"] >= 2].sort_values(["n_nodes_as_anchor", "ATI_conservative"], ascending=[False, False])
    if repeated.empty:
        return None
    chosen = repeated.iloc[0]
    row = anchors[anchors["anchor_pl_name"] == chosen["anchor_pl_name"]].sort_values(["ATI_conservative", "ATI_original"], ascending=[False, False]).iloc[0]
    return _case_row(
        "repeated_anchor_transition_case",
        row,
        why_selected="Ancla repetida en varios nodos Mapper; sirve como caso de transicion por solapamiento.",
        how_to_present="Usar para explicar por que una ancla puede vivir en la frontera entre vecindarios Mapper.",
        caution_text="La repeticion no implica duplicacion erronea; refleja solapamiento de cubiertas y necesita contexto topologico.",
        repeated_anchor_nodes=chosen["nodes_list"],
    )


def _case_row(
    case_type: str,
    row: pd.Series,
    *,
    why_selected: str,
    how_to_present: str,
    caution_text: str,
    repeated_anchor_nodes: str | None = None,
) -> dict[str, object]:
    return {
        "case_type": case_type,
        "anchor_pl_name": row.get("anchor_pl_name"),
        "node_id": row.get("node_id"),
        "TOI": row.get("TOI_original", row.get("TOI")),
        "ATI_original": row.get("ATI_original", row.get("ATI")),
        "ATI_conservative": row.get("ATI_conservative"),
        "delta_rel_mean": row.get("delta_rel_mean"),
        "delta_rel_best": row.get("delta_rel_neighbors_best", row.get("delta_rel_max")),
        "deficit_stability_class": row.get("deficit_stability_class"),
        "rank_shift": row.get("rank_shift"),
        "repeated_anchor_nodes": repeated_anchor_nodes or "",
        "why_selected": why_selected,
        "how_to_present": how_to_present,
        "caution_text": caution_text,
        "future_observation_direction": row.get("future_observation_direction", row.get("expected_incompleteness_direction")),
    }


def _issue_row(row: pd.Series, issue_type: str, note: str) -> dict[str, object]:
    return {
        "issue_type": issue_type,
        "anchor_pl_name": row.get("anchor_pl_name"),
        "node_id": row.get("node_id"),
        "ATI_original": row.get("ATI_original", row.get("ATI")),
        "ATI_conservative": row.get("ATI_conservative"),
        "delta_rel_best": row.get("delta_rel_neighbors_best"),
        "delta_rel_mean": row.get("delta_rel_mean"),
        "r3_imputation_score": row.get("r3_imputation_score"),
        "n_members": row.get("n_members"),
        "note": note,
    }


def _priority_reason(row: pd.Series) -> str:
    return "Combina TOI alto, ATI conservador, estabilidad razonable y una direccion fisica interpretable para inspeccion futura."


def _priority_caution(row: pd.Series) -> str:
    if str(row.get("deficit_stability_class", "")) in {"radius_sensitive_deficit", "unstable_due_to_large_radius"}:
        return "Caso sensible al radio; priorizacion observacional prudente y no concluyente."
    return "Caso util para inspeccion futura, sin afirmar objetos ausentes."


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return pd.Series(np.zeros(len(series)), index=series.index)
    min_value = float(values.min(skipna=True))
    max_value = float(values.max(skipna=True))
    if np.isclose(max_value, min_value):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (values - min_value) / (max_value - min_value)
