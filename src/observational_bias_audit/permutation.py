from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from .metrics import (
    benjamini_hochberg,
    nominal_assortativity_from_codes,
    normalized_entropy_from_counts,
    purity_from_counts,
    shannon_entropy_from_counts,
)


@dataclass
class PermutationPreparation:
    node_ids: list[str]
    method_labels: list[str]
    row_node_codes: np.ndarray
    row_member_codes: np.ndarray
    original_method_codes_by_member: np.ndarray
    node_sizes: np.ndarray
    edge_pairs: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_methods(self) -> int:
        return len(self.method_labels)


def prepare_permutation_inputs(
    membership_with_metadata: pd.DataFrame,
    node_metrics: pd.DataFrame,
    edge_table: pd.DataFrame,
) -> PermutationPreparation:
    working = membership_with_metadata.copy()
    working["node_id"] = working["node_id"].astype(str)
    working["member_key"] = working["row_index"].astype(str)
    working["discoverymethod"] = working["discoverymethod"].astype("string").fillna("Unknown").astype(str)

    node_ids = node_metrics["node_id"].astype(str).tolist()
    node_lookup = {node_id: idx for idx, node_id in enumerate(node_ids)}
    method_labels = sorted(working["discoverymethod"].unique().tolist())
    method_lookup = {label: idx for idx, label in enumerate(method_labels)}

    member_keys = working["member_key"].drop_duplicates().tolist()
    member_lookup = {member_key: idx for idx, member_key in enumerate(member_keys)}
    original_method_codes_by_member = (
        working.drop_duplicates(subset=["member_key"])
        .set_index("member_key")["discoverymethod"]
        .reindex(member_keys)
        .map(method_lookup)
        .to_numpy(dtype=int)
    )

    row_node_codes = working["node_id"].map(node_lookup).to_numpy(dtype=int)
    row_member_codes = working["member_key"].map(member_lookup).to_numpy(dtype=int)
    node_sizes = node_metrics.set_index("node_id").reindex(node_ids)["n_members"].to_numpy(dtype=float)

    edge_pairs: list[tuple[int, int]] = []
    for row in edge_table.itertuples(index=False):
        source = str(getattr(row, "source"))
        target = str(getattr(row, "target"))
        if source in node_lookup and target in node_lookup:
            edge_pairs.append((node_lookup[source], node_lookup[target]))

    return PermutationPreparation(
        node_ids=node_ids,
        method_labels=method_labels,
        row_node_codes=row_node_codes,
        row_member_codes=row_member_codes,
        original_method_codes_by_member=original_method_codes_by_member,
        node_sizes=node_sizes,
        edge_pairs=np.asarray(edge_pairs, dtype=int) if edge_pairs else np.zeros((0, 2), dtype=int),
    )


def _count_matrix_from_member_codes(prepared: PermutationPreparation, member_method_codes: np.ndarray) -> np.ndarray:
    row_method_codes = member_method_codes[prepared.row_member_codes]
    flat_index = prepared.row_node_codes * prepared.n_methods + row_method_codes
    return np.bincount(flat_index, minlength=prepared.n_nodes * prepared.n_methods).reshape(prepared.n_nodes, prepared.n_methods)


def _global_metrics_from_counts(prepared: PermutationPreparation, count_matrix: np.ndarray, member_method_codes: np.ndarray) -> dict[str, float]:
    node_sizes = prepared.node_sizes
    purities = np.array([purity_from_counts(row) for row in count_matrix], dtype=float)
    entropies = np.array([shannon_entropy_from_counts(row) for row in count_matrix], dtype=float)
    weighted_mean_purity = float(np.average(purities, weights=node_sizes)) if node_sizes.sum() > 0 else np.nan
    weighted_mean_entropy = float(np.average(entropies, weights=node_sizes)) if node_sizes.sum() > 0 else np.nan

    nmi = np.nan
    if prepared.n_nodes > 1 and prepared.n_methods > 1:
        row_method_codes = member_method_codes[prepared.row_member_codes]
        nmi = float(normalized_mutual_info_score(prepared.row_node_codes, row_method_codes))

    assortativity = np.nan
    if prepared.edge_pairs.size > 0 and prepared.n_methods > 1:
        top_method_codes = np.argmax(count_matrix, axis=1)
        if len(np.unique(top_method_codes)) > 1:
            assortativity = nominal_assortativity_from_codes(prepared.edge_pairs, top_method_codes, prepared.n_methods)

    return {
        "weighted_mean_purity": weighted_mean_purity,
        "weighted_mean_entropy": weighted_mean_entropy,
        "node_method_nmi": nmi,
        "dominant_method_assortativity": assortativity,
    }


def run_permutation_audit(
    config_id: str,
    membership_with_metadata: pd.DataFrame,
    node_metrics: pd.DataFrame,
    edge_table: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if membership_with_metadata.empty or node_metrics.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    prepared = prepare_permutation_inputs(
        membership_with_metadata=membership_with_metadata,
        node_metrics=node_metrics,
        edge_table=edge_table,
    )
    observed_counts = _count_matrix_from_member_codes(prepared, prepared.original_method_codes_by_member)
    observed_global = _global_metrics_from_counts(prepared, observed_counts, prepared.original_method_codes_by_member)

    metric_directions = {
        "weighted_mean_purity": "high",
        "weighted_mean_entropy": "low",
        "node_method_nmi": "high",
        "dominant_method_assortativity": "high",
    }
    null_store = {metric: np.full(n_permutations, np.nan, dtype=float) for metric in metric_directions}
    observed_fractions = observed_counts / np.where(prepared.node_sizes[:, None] > 0, prepared.node_sizes[:, None], 1.0)
    frac_sum = np.zeros_like(observed_fractions, dtype=float)
    frac_sq_sum = np.zeros_like(observed_fractions, dtype=float)
    exceed_count = np.zeros_like(observed_counts, dtype=int)
    rng = np.random.default_rng(seed)

    for perm_index in range(n_permutations):
        permuted = rng.permutation(prepared.original_method_codes_by_member)
        perm_counts = _count_matrix_from_member_codes(prepared, permuted)
        perm_fracs = perm_counts / np.where(prepared.node_sizes[:, None] > 0, prepared.node_sizes[:, None], 1.0)
        frac_sum += perm_fracs
        frac_sq_sum += perm_fracs ** 2
        exceed_count += perm_fracs >= observed_fractions

        metrics = _global_metrics_from_counts(prepared, perm_counts, permuted)
        for metric_name, metric_value in metrics.items():
            null_store[metric_name][perm_index] = metric_value

    global_rows: list[dict[str, Any]] = []
    for metric_name, direction in metric_directions.items():
        observed = observed_global.get(metric_name, np.nan)
        null_values = null_store[metric_name]
        valid = null_values[np.isfinite(null_values)]
        if valid.size == 0 or not np.isfinite(observed):
            global_rows.append(
                {
                    "config_id": config_id,
                    "metric": metric_name,
                    "observed": observed,
                    "null_mean": np.nan,
                    "null_std": np.nan,
                    "z_score": np.nan,
                    "empirical_p_value": np.nan,
                    "n_permutations": n_permutations,
                    "seed": seed,
                }
            )
            continue
        tail_count = int((valid >= observed).sum()) if direction == "high" else int((valid <= observed).sum())
        null_mean = float(valid.mean())
        null_std = float(valid.std(ddof=0))
        z_score = float((observed - null_mean) / null_std) if null_std > 0 else np.nan
        global_rows.append(
            {
                "config_id": config_id,
                "metric": metric_name,
                "observed": observed,
                "null_mean": null_mean,
                "null_std": null_std,
                "z_score": z_score,
                "empirical_p_value": float((1 + tail_count) / (n_permutations + 1)),
                "n_permutations": n_permutations,
                "seed": seed,
            }
        )

    frac_mean = frac_sum / float(n_permutations)
    frac_var = np.maximum(frac_sq_sum / float(n_permutations) - frac_mean ** 2, 0.0)
    frac_std = np.sqrt(frac_var)
    global_method_fraction = observed_counts.sum(axis=0) / observed_counts.sum()

    enrichment_rows: list[dict[str, Any]] = []
    for node_idx, node_id in enumerate(prepared.node_ids):
        for method_idx, method_name in enumerate(prepared.method_labels):
            node_count = int(observed_counts[node_idx, method_idx])
            node_fraction = float(observed_fractions[node_idx, method_idx])
            global_fraction = float(global_method_fraction[method_idx])
            null_mean = float(frac_mean[node_idx, method_idx])
            null_std = float(frac_std[node_idx, method_idx])
            z_score = float((node_fraction - null_mean) / null_std) if null_std > 0 else np.nan
            enrichment_rows.append(
                {
                    "config_id": config_id,
                    "node_id": node_id,
                    "method": method_name,
                    "node_count": node_count,
                    "node_fraction": node_fraction,
                    "global_fraction": global_fraction,
                    "enrichment_ratio": float(node_fraction / global_fraction) if global_fraction > 0 else np.nan,
                    "z_score": z_score,
                    "empirical_p_value": float((1 + exceed_count[node_idx, method_idx]) / (n_permutations + 1)),
                }
            )

    enrichment_df = pd.DataFrame(enrichment_rows)
    enrichment_df["fdr_q_value"] = enrichment_df.groupby("config_id")["empirical_p_value"].transform(benjamini_hochberg)
    null_distribution = pd.DataFrame(
        {
            "config_id": config_id,
            "permutation_index": np.arange(n_permutations, dtype=int),
            **null_store,
        }
    )
    return (
        pd.DataFrame(global_rows),
        enrichment_df.sort_values(["empirical_p_value", "z_score"], ascending=[True, False]).reset_index(drop=True),
        null_distribution,
    )
