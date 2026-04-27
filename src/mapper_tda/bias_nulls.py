from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from .bias_audit import GraphBiasInput


@dataclass
class PreparedPermutationGraph:
    graph_input: GraphBiasInput
    labels: np.ndarray
    label_codes: np.ndarray
    label_names: list[str]
    member_ids: list[int]
    member_to_position: dict[int, int]
    node_positions: dict[str, np.ndarray]
    hard_assignment_positions: np.ndarray
    hard_assignment_labels: np.ndarray


def _valid_method_labels(graph_input: GraphBiasInput) -> tuple[list[int], np.ndarray]:
    if "discoverymethod" not in graph_input.metadata.columns:
        raise ValueError(f"`discoverymethod` is missing from metadata for {graph_input.config_id}")
    unique_members = sorted({member for members in graph_input.nodes.values() for member in members})
    indexed = graph_input.metadata.set_index("_mapper_row_index", drop=False)
    member_ids: list[int] = []
    labels: list[str] = []
    for member in unique_members:
        if member not in indexed.index:
            continue
        value = indexed.at[member, "discoverymethod"]
        if pd.isna(value) or str(value).strip() == "":
            continue
        member_ids.append(int(member))
        labels.append(str(value))
    if not member_ids:
        raise ValueError(f"No valid discoverymethod labels are available for {graph_input.config_id}")
    return member_ids, np.asarray(labels, dtype=object)


def _prepare_graph(graph_input: GraphBiasInput) -> PreparedPermutationGraph:
    member_ids, labels = _valid_method_labels(graph_input)
    member_to_position = {member: idx for idx, member in enumerate(member_ids)}
    codes, uniques = pd.factorize(labels, sort=True)
    node_positions: dict[str, np.ndarray] = {}
    for node_id, members in graph_input.nodes.items():
        positions = [member_to_position[member] for member in members if member in member_to_position]
        node_positions[node_id] = np.asarray(positions, dtype=int)

    best_assignment: dict[int, tuple[int, str]] = {}
    for node_id in sorted(graph_input.nodes):
        node_size = len(graph_input.nodes[node_id])
        for member in graph_input.nodes[node_id]:
            if member not in member_to_position:
                continue
            current = best_assignment.get(member)
            if current is None or node_size > current[0] or (node_size == current[0] and node_id < current[1]):
                best_assignment[member] = (node_size, node_id)

    hard_member_ids = sorted(best_assignment)
    hard_positions = np.asarray([member_to_position[member] for member in hard_member_ids], dtype=int)
    hard_labels = np.asarray([best_assignment[member][1] for member in hard_member_ids], dtype=object)
    if len(hard_positions) == 0:
        raise ValueError(f"Hard node assignment could not be built for {graph_input.config_id}")

    return PreparedPermutationGraph(
        graph_input=graph_input,
        labels=labels,
        label_codes=np.asarray(codes, dtype=int),
        label_names=[str(value) for value in uniques.tolist()],
        member_ids=member_ids,
        member_to_position=member_to_position,
        node_positions=node_positions,
        hard_assignment_positions=hard_positions,
        hard_assignment_labels=hard_labels,
    )


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return np.nan
    probs = counts[counts > 0].astype(float) / float(total)
    return float(-(probs * np.log2(probs)).sum())


def _purity_entropy_for_codes(label_codes: np.ndarray, node_positions: dict[str, np.ndarray], n_labels: int) -> tuple[float, float]:
    purity_sum = 0.0
    entropy_sum = 0.0
    weight_sum = 0.0
    for positions in node_positions.values():
        if len(positions) == 0:
            continue
        counts = np.bincount(label_codes[positions], minlength=n_labels)
        total = counts.sum()
        if total <= 0:
            continue
        weight = float(total)
        purity = float(counts.max() / total)
        entropy = _entropy_from_counts(counts)
        purity_sum += purity * weight
        entropy_sum += entropy * weight
        weight_sum += weight
    if weight_sum == 0:
        return np.nan, np.nan
    return purity_sum / weight_sum, entropy_sum / weight_sum


def _nmi_for_codes(label_codes: np.ndarray, hard_positions: np.ndarray, hard_labels: np.ndarray) -> float:
    method_labels = label_codes[hard_positions]
    if len(method_labels) == 0:
        return np.nan
    return float(normalized_mutual_info_score(hard_labels, method_labels))


def _z_score(observed: float, null_values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(null_values)) if len(null_values) else np.nan
    std = float(np.std(null_values, ddof=0)) if len(null_values) else np.nan
    z = float((observed - mean) / std) if np.isfinite(observed) and np.isfinite(std) and std > 0 else np.nan
    return mean, std, z


def _empirical_p_high(observed: float, null_values: np.ndarray) -> float:
    if not np.isfinite(observed) or len(null_values) == 0:
        return np.nan
    return float((np.sum(null_values >= observed) + 1) / (len(null_values) + 1))


def _empirical_p_low(observed: float, null_values: np.ndarray) -> float:
    if not np.isfinite(observed) or len(null_values) == 0:
        return np.nan
    return float((np.sum(null_values <= observed) + 1) / (len(null_values) + 1))


def _observed_node_targets(prepared: PreparedPermutationGraph) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node_id, positions in prepared.node_positions.items():
        if len(positions) == 0:
            continue
        counts = np.bincount(prepared.label_codes[positions], minlength=len(prepared.label_names))
        total = counts.sum()
        if total <= 0:
            continue
        dominant_code = int(np.argmax(counts))
        dominant = prepared.label_names[dominant_code]
        fraction = float(counts[dominant_code] / total)
        rows.append(
            {
                "config_id": prepared.graph_input.config_id,
                "feature_space": prepared.graph_input.feature_space,
                "lens": prepared.graph_input.lens,
                "node_id": node_id,
                "n_members_with_method": int(len(positions)),
                "dominant_discoverymethod": dominant,
                "dominant_method_code": dominant_code,
                "observed_dominant_method_fraction": float(fraction),
                "positions": positions,
            }
        )
    return rows


def run_discoverymethod_permutation_tests(
    graph_inputs: list[GraphBiasInput],
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_perm <= 0:
        raise ValueError("n_perm must be positive.")

    rng = np.random.default_rng(seed)
    graph_rows: list[dict[str, Any]] = []
    enrichment_rows: list[dict[str, Any]] = []

    for graph_input in graph_inputs:
        prepared = _prepare_graph(graph_input)
        n_labels = len(prepared.label_names)
        observed_purity, observed_entropy = _purity_entropy_for_codes(prepared.label_codes, prepared.node_positions, n_labels)
        observed_nmi = _nmi_for_codes(prepared.label_codes, prepared.hard_assignment_positions, prepared.hard_assignment_labels)
        purity_null = np.zeros(n_perm, dtype=float)
        entropy_null = np.zeros(n_perm, dtype=float)
        nmi_null = np.zeros(n_perm, dtype=float)

        node_targets = _observed_node_targets(prepared)
        node_sums = np.zeros(len(node_targets), dtype=float)
        node_sumsq = np.zeros(len(node_targets), dtype=float)
        node_ge = np.zeros(len(node_targets), dtype=int)

        for iteration in range(n_perm):
            permuted = rng.permutation(prepared.label_codes)
            purity_null[iteration], entropy_null[iteration] = _purity_entropy_for_codes(permuted, prepared.node_positions, n_labels)
            nmi_null[iteration] = _nmi_for_codes(permuted, prepared.hard_assignment_positions, prepared.hard_assignment_labels)

            for idx, target in enumerate(node_targets):
                positions = target["positions"]
                if len(positions) == 0:
                    fraction = np.nan
                else:
                    fraction = float(np.mean(permuted[positions] == target["dominant_method_code"]))
                if np.isfinite(fraction):
                    node_sums[idx] += fraction
                    node_sumsq[idx] += fraction * fraction
                    if fraction >= target["observed_dominant_method_fraction"]:
                        node_ge[idx] += 1

        purity_mean, purity_std, purity_z = _z_score(observed_purity, purity_null)
        entropy_mean, entropy_std, entropy_z = _z_score(observed_entropy, entropy_null)
        nmi_mean, nmi_std, nmi_z = _z_score(observed_nmi, nmi_null)
        graph_rows.append(
            {
                "config_id": graph_input.config_id,
                "feature_space": graph_input.feature_space,
                "lens": graph_input.lens,
                "n_perm": int(n_perm),
                "seed": int(seed),
                "observed_weighted_mean_method_purity": observed_purity,
                "null_mean_purity": purity_mean,
                "null_std_purity": purity_std,
                "purity_z": purity_z,
                "purity_empirical_p": _empirical_p_high(observed_purity, purity_null),
                "observed_weighted_mean_method_entropy": observed_entropy,
                "null_mean_entropy": entropy_mean,
                "null_std_entropy": entropy_std,
                "entropy_z": entropy_z,
                "entropy_empirical_p": _empirical_p_low(observed_entropy, entropy_null),
                "observed_nmi": observed_nmi,
                "null_mean_nmi": nmi_mean,
                "null_std_nmi": nmi_std,
                "nmi_z": nmi_z,
                "nmi_empirical_p": _empirical_p_high(observed_nmi, nmi_null),
            }
        )

        for idx, target in enumerate(node_targets):
            null_mean = float(node_sums[idx] / n_perm)
            variance = max(0.0, float(node_sumsq[idx] / n_perm) - null_mean * null_mean)
            null_std = float(np.sqrt(variance))
            observed_fraction = float(target["observed_dominant_method_fraction"])
            enrichment_z = float((observed_fraction - null_mean) / null_std) if null_std > 0 else np.nan
            enrichment_rows.append(
                {
                    "config_id": target["config_id"],
                    "feature_space": target["feature_space"],
                    "lens": target["lens"],
                    "node_id": target["node_id"],
                    "n_members_with_method": target["n_members_with_method"],
                    "dominant_discoverymethod": target["dominant_discoverymethod"],
                    "observed_dominant_method_fraction": observed_fraction,
                    "null_mean_fraction_for_that_method": null_mean,
                    "null_std_fraction_for_that_method": null_std,
                    "enrichment_z": enrichment_z,
                    "empirical_p_value": float((node_ge[idx] + 1) / (n_perm + 1)),
                    "n_perm": int(n_perm),
                    "seed": int(seed),
                }
            )

    return pd.DataFrame(graph_rows), pd.DataFrame(enrichment_rows)
