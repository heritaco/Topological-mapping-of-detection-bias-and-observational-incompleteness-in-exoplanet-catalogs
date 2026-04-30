from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import safe_log10_series


@dataclass
class GapStatistics:
    adjacency: pd.DataFrame
    global_median_log_gap: float
    global_log_gap_values: np.ndarray
    method_medians: dict[str, tuple[float, int]]
    group_medians: dict[tuple[str, str], tuple[float, int]]

    def typical_gap(self, method_group: str, mass_bin: str) -> tuple[float, str]:
        group_key = (method_group, mass_bin)
        if group_key in self.group_medians and self.group_medians[group_key][1] >= 20:
            return self.group_medians[group_key][0], "method_plus_mass_bin"
        if method_group in self.method_medians and self.method_medians[method_group][1] >= 20:
            return self.method_medians[method_group][0], "method_group"
        return self.global_median_log_gap, "global"

    def gap_percentile(self, log_gap: float) -> float:
        values = self.global_log_gap_values
        if values.size == 0 or not np.isfinite(log_gap):
            return 0.0
        return float(np.mean(values <= log_gap))


def build_gap_statistics(catalog: pd.DataFrame, system_metadata: pd.DataFrame) -> GapStatistics:
    metadata = system_metadata.set_index("hostname")
    rows: list[dict[str, object]] = []
    valid = catalog[pd.to_numeric(catalog["pl_orbper"], errors="coerce") > 0].copy()
    valid["pl_orbper"] = pd.to_numeric(valid["pl_orbper"], errors="coerce")
    valid["pl_orbsmax"] = pd.to_numeric(valid["pl_orbsmax"], errors="coerce")
    for hostname, group in valid.groupby("hostname", dropna=True):
        if hostname not in metadata.index:
            continue
        ordered = group.sort_values("pl_orbper").reset_index(drop=True)
        for idx in range(len(ordered) - 1):
            inner = ordered.iloc[idx]
            outer = ordered.iloc[idx + 1]
            p_in = float(inner["pl_orbper"])
            p_out = float(outer["pl_orbper"])
            if not np.isfinite(p_in) or not np.isfinite(p_out) or p_in <= 0 or p_out <= 0:
                continue
            rows.append(
                {
                    "hostname": hostname,
                    "inner_planet": inner["pl_name"],
                    "outer_planet": outer["pl_name"],
                    "P_in": p_in,
                    "P_out": p_out,
                    "a_in": inner.get("pl_orbsmax"),
                    "a_out": outer.get("pl_orbsmax"),
                    "gap_ratio": p_out / p_in,
                    "gap_log_width": math.log10(p_out) - math.log10(p_in),
                    "dominant_discoverymethod_system": metadata.loc[hostname, "dominant_discoverymethod_system"],
                    "dominant_method_group": metadata.loc[hostname, "dominant_method_group"],
                    "stellar_mass_bin": metadata.loc[hostname, "stellar_mass_bin"],
                }
            )
    adjacency = pd.DataFrame(rows)
    adjacency["gap_log_width"] = pd.to_numeric(adjacency.get("gap_log_width"), errors="coerce")
    gap_values = adjacency["gap_log_width"].dropna().to_numpy(dtype=float) if not adjacency.empty else np.array([], dtype=float)
    global_median = float(np.nanmedian(gap_values)) if gap_values.size else math.log10(2.0)
    method_medians = {
        str(method): (float(pd.to_numeric(group["gap_log_width"], errors="coerce").median()), int(len(group)))
        for method, group in adjacency.groupby("dominant_method_group")
    } if not adjacency.empty else {}
    group_medians = {
        (str(method), str(mass_bin)): (float(pd.to_numeric(group["gap_log_width"], errors="coerce").median()), int(len(group)))
        for (method, mass_bin), group in adjacency.groupby(["dominant_method_group", "stellar_mass_bin"])
    } if not adjacency.empty else {}
    return GapStatistics(
        adjacency=adjacency,
        global_median_log_gap=global_median,
        global_log_gap_values=gap_values,
        method_medians=method_medians,
        group_medians=group_medians,
    )


def expected_missing_counts(
    gap_ratio: float,
    gap_log_width: float,
    stats: GapStatistics,
    min_gap_ratio: float,
    max_candidates_per_gap: int,
    method_group: str,
    mass_bin: str,
) -> dict[str, object]:
    heuristic = 0
    if np.isfinite(gap_ratio) and gap_ratio >= min_gap_ratio:
        heuristic = int(math.floor(math.log(gap_ratio) / math.log(min_gap_ratio)))
        heuristic = int(np.clip(heuristic, 1, max_candidates_per_gap))
    typical_gap, source = stats.typical_gap(method_group, mass_bin)
    empirical = 0
    if np.isfinite(gap_log_width) and typical_gap > 0:
        empirical = int(np.round(gap_log_width / typical_gap) - 1)
        empirical = int(np.clip(empirical, 0, max_candidates_per_gap))
    expected = int(np.clip(max(heuristic, empirical), 0, max_candidates_per_gap))
    return {
        "expected_missing_count_heuristic": heuristic,
        "expected_missing_count_empirical": empirical,
        "expected_missing_count": expected,
        "typical_gap_log_width": typical_gap,
        "typical_gap_source": source,
    }


def find_candidate_gaps(
    catalog: pd.DataFrame,
    system_metadata: pd.DataFrame,
    stats: GapStatistics,
    min_gap_ratio: float,
    max_candidates_per_gap: int,
) -> pd.DataFrame:
    metadata = system_metadata.set_index("hostname")
    valid = catalog[pd.to_numeric(catalog["pl_orbper"], errors="coerce") > 0].copy()
    rows: list[dict[str, object]] = []
    for hostname, group in valid.groupby("hostname", dropna=True):
        if hostname not in metadata.index:
            continue
        ordered = group.sort_values("pl_orbper").reset_index(drop=True)
        for idx in range(len(ordered) - 1):
            inner = ordered.iloc[idx]
            outer = ordered.iloc[idx + 1]
            p_in = float(inner["pl_orbper"])
            p_out = float(outer["pl_orbper"])
            if not np.isfinite(p_in) or not np.isfinite(p_out) or p_in <= 0 or p_out <= 0:
                continue
            gap_ratio = p_out / p_in
            gap_log_width = math.log10(p_out) - math.log10(p_in)
            expected = expected_missing_counts(
                gap_ratio=gap_ratio,
                gap_log_width=gap_log_width,
                stats=stats,
                min_gap_ratio=min_gap_ratio,
                max_candidates_per_gap=max_candidates_per_gap,
                method_group=str(metadata.loc[hostname, "dominant_method_group"]),
                mass_bin=str(metadata.loc[hostname, "stellar_mass_bin"]),
            )
            rows.append(
                {
                    "hostname": hostname,
                    "inner_planet": inner["pl_name"],
                    "outer_planet": outer["pl_name"],
                    "P_in": p_in,
                    "P_out": p_out,
                    "a_in": pd.to_numeric(pd.Series([inner.get("pl_orbsmax")]), errors="coerce").iloc[0],
                    "a_out": pd.to_numeric(pd.Series([outer.get("pl_orbsmax")]), errors="coerce").iloc[0],
                    "gap_ratio": gap_ratio,
                    "gap_log_width": gap_log_width,
                    "dominant_discoverymethod_system": metadata.loc[hostname, "dominant_discoverymethod_system"],
                    "dominant_method_group": metadata.loc[hostname, "dominant_method_group"],
                    "stellar_mass_bin": metadata.loc[hostname, "stellar_mass_bin"],
                    "candidate_stellar_mass": metadata.loc[hostname, "system_st_mass_median"],
                    "candidate_stellar_radius": metadata.loc[hostname, "system_st_rad_median"],
                    "inner_discoverymethod": inner.get("discoverymethod", "Unknown"),
                    "outer_discoverymethod": outer.get("discoverymethod", "Unknown"),
                    **expected,
                }
            )
    gaps = pd.DataFrame(rows)
    if gaps.empty:
        return gaps
    gaps["gap_score"] = gaps["gap_log_width"].apply(stats.gap_percentile)
    return gaps[gaps["gap_ratio"] >= min_gap_ratio].reset_index(drop=True)


def expand_gap_candidates(gaps: pd.DataFrame) -> pd.DataFrame:
    if gaps.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for _, gap in gaps.iterrows():
        n_candidates = int(gap.get("expected_missing_count", 0))
        if n_candidates <= 0:
            continue
        logp_in = math.log10(float(gap["P_in"]))
        logp_out = math.log10(float(gap["P_out"]))
        loga_in = float("nan")
        loga_out = float("nan")
        if pd.notna(gap.get("a_in")) and pd.notna(gap.get("a_out")) and float(gap["a_in"]) > 0 and float(gap["a_out"]) > 0:
            loga_in = math.log10(float(gap["a_in"]))
            loga_out = math.log10(float(gap["a_out"]))
        for rank in range(1, n_candidates + 1):
            fraction = rank / (n_candidates + 1)
            logp_candidate = logp_in + fraction * (logp_out - logp_in)
            period = 10 ** logp_candidate
            stellar_mass = pd.to_numeric(pd.Series([gap.get("candidate_stellar_mass")]), errors="coerce").iloc[0]
            semimajor = np.nan
            method = "geometric_interpolation"
            if np.isfinite(stellar_mass) and stellar_mass > 0:
                semimajor = float(np.power(stellar_mass * np.power(period / 365.25, 2.0), 1.0 / 3.0))
                method = "kepler"
            elif np.isfinite(loga_in) and np.isfinite(loga_out):
                semimajor = float(10 ** (loga_in + fraction * (loga_out - loga_in)))
            candidate_id = f"{gap['hostname']}::{gap['inner_planet']}::{gap['outer_planet']}::{rank}"
            records.append(
                {
                    "hostname": gap["hostname"],
                    "candidate_id": candidate_id,
                    "inner_planet": gap["inner_planet"],
                    "outer_planet": gap["outer_planet"],
                    "candidate_rank_in_gap": rank,
                    "candidate_fraction": fraction,
                    "candidate_period_days": period,
                    "candidate_semimajor_au": semimajor,
                    "candidate_position_method": method,
                    "gap_period_ratio": gap["gap_ratio"],
                    "gap_log_width": gap["gap_log_width"],
                    "expected_missing_count": gap["expected_missing_count"],
                    "expected_missing_count_heuristic": gap["expected_missing_count_heuristic"],
                    "expected_missing_count_empirical": gap["expected_missing_count_empirical"],
                    "typical_gap_log_width": gap["typical_gap_log_width"],
                    "typical_gap_source": gap["typical_gap_source"],
                    "dominant_discoverymethod_system": gap["dominant_discoverymethod_system"],
                    "dominant_method_group": gap["dominant_method_group"],
                    "inner_discoverymethod": gap["inner_discoverymethod"],
                    "outer_discoverymethod": gap["outer_discoverymethod"],
                    "candidate_stellar_mass": gap["candidate_stellar_mass"],
                    "candidate_stellar_radius": gap["candidate_stellar_radius"],
                    "gap_score": gap["gap_score"],
                }
            )
    candidates = pd.DataFrame(records)
    if not candidates.empty:
        candidates["candidate_log_period"] = safe_log10_series(candidates["candidate_period_days"])
    return candidates
