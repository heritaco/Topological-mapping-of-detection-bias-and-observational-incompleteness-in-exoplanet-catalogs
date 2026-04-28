from __future__ import annotations

import pandas as pd


def make_region_profile(top_regions: pd.DataFrame, toi_decomp: pd.DataFrame) -> pd.DataFrame:
    if top_regions.empty:
        return pd.DataFrame()
    cols = [
        "config_id", "node_id", "n_members", "top_method", "shadow_score", "I_R3",
        "C_phys", "S_net", "TOI", "TOI_rank", "region_class", "dominant_toi_driver",
        "physical_distance_v_to_N1", "method_l1_boundary_N1",
    ]
    merged = top_regions.merge(
        toi_decomp[[c for c in toi_decomp.columns if c in {"node_id", "dominant_toi_driver", "TOI_recomputed", "TOI_abs_error"}]],
        on="node_id",
        how="left",
    ) if "node_id" in top_regions.columns and "node_id" in toi_decomp.columns else top_regions.copy()
    return merged[[c for c in cols if c in merged.columns]]


def make_anchor_profile(top_anchors: pd.DataFrame, ati_decomp: pd.DataFrame) -> pd.DataFrame:
    if top_anchors.empty:
        return pd.DataFrame()
    join_cols = [c for c in ["node_id", "anchor_pl_name"] if c in top_anchors.columns and c in ati_decomp.columns]
    merged = top_anchors.merge(
        ati_decomp[[c for c in ati_decomp.columns if c in set(join_cols + ["dominant_ati_driver", "ATI_recomputed", "ATI_abs_error"])]],
        on=join_cols,
        how="left",
    ) if join_cols else top_anchors.copy()
    cols = [
        "config_id", "node_id", "anchor_pl_name", "discoverymethod", "TOI", "ATI",
        "delta_rel_neighbors_best", "delta_rel_neighbors_mean", "r3_imputation_score",
        "anchor_representativeness", "deficit_class", "expected_incompleteness_direction",
        "dominant_ati_driver",
    ]
    return merged[[c for c in cols if c in merged.columns]]


def build_interpretation_sentences(region_profile: pd.DataFrame, anchor_profile: pd.DataFrame) -> list[str]:
    sentences: list[str] = []
    if not region_profile.empty:
        best = region_profile.iloc[0]
        sentences.append(
            f"La region con mayor prioridad regional es {best.get('node_id', 'unknown')}, "
            f"con TOI={best.get('TOI', float('nan')):.4g}. La lectura debe descomponerse en sombra, "
            "imputacion R^3, continuidad fisica y soporte de red antes de convertirla en conclusion astrofisica."
        )
    if not anchor_profile.empty:
        best = anchor_profile.iloc[0]
        sentences.append(
            f"El ancla con mayor ATI es {best.get('anchor_pl_name', 'unknown')} / {best.get('node_id', 'unknown')}, "
            f"con ATI={best.get('ATI', float('nan')):.4g}. Este valor prioriza inspeccion, no prueba objetos ausentes."
        )
    sentences.append(
        "El criterio central del reporte es explicar por que gana cada caso: que factor multiplica el indice, que factor lo limita y que tan robusto es el deficit por radio."
    )
    return sentences
