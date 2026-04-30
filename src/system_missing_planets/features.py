from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import validate_prudent_text


METHOD_PRIORITY = ["Transit", "Radial Velocity"]


@dataclass
class AnalogPools:
    pools: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]]


def safe_log10_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & (numeric > 0)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[valid] = np.log10(numeric.loc[valid].astype(float))
    return out


def clip01(value: float | pd.Series) -> float | pd.Series:
    if isinstance(value, pd.Series):
        return value.clip(0.0, 1.0)
    return float(min(1.0, max(0.0, value)))


def normalize_score_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric[np.isfinite(numeric)]
    if valid.empty:
        return pd.Series(0.0, index=series.index, dtype=float)
    max_value = float(valid.max())
    min_value = float(valid.min())
    if math.isclose(max_value, min_value):
        return pd.Series(np.where(numeric.notna(), 1.0, 0.0), index=series.index, dtype=float)
    out = (numeric - min_value) / (max_value - min_value)
    out = out.fillna(0.0)
    return out.clip(0.0, 1.0)


def canonical_method(value: object) -> str:
    text = str(value or "Unknown").strip()
    lower = text.lower()
    if "transit" in lower:
        return "Transit"
    if "radial velocity" in lower:
        return "Radial Velocity"
    return text if text else "Unknown"


def classify_system_methods(methods: Iterable[object]) -> tuple[str, str, float]:
    canonical = pd.Series([canonical_method(value) for value in methods], dtype="string")
    if canonical.empty:
        return "Unknown", "Mixed", 0.0
    counts = canonical.value_counts(dropna=False)
    dominant = str(counts.index[0])
    fraction = float(counts.iloc[0] / max(len(canonical), 1))
    if dominant in METHOD_PRIORITY and fraction >= 0.60:
        return dominant, dominant, fraction
    if dominant not in METHOD_PRIORITY and fraction >= 0.75:
        return dominant, "Mixed", fraction
    return "Mixed", "Mixed", fraction


def stellar_mass_bin(st_mass: float | None) -> str:
    if st_mass is None or not np.isfinite(st_mass) or st_mass <= 0:
        return "unknown_mass_star"
    if st_mass < 0.8:
        return "low_mass_star"
    if st_mass <= 1.2:
        return "solar_like_star"
    return "high_mass_star"


def build_system_metadata(catalog: pd.DataFrame, min_planets_per_system: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid = catalog[pd.to_numeric(catalog["pl_orbper"], errors="coerce") > 0].copy()
    for hostname, group in valid.groupby("hostname", dropna=True):
        group_sorted = group.sort_values("pl_orbper").copy()
        n_planets = int(len(group_sorted))
        if n_planets < min_planets_per_system:
            continue
        dominant_method, method_group, dominant_fraction = classify_system_methods(group_sorted["discoverymethod"])
        st_mass = pd.to_numeric(group_sorted.get("st_mass"), errors="coerce")
        st_rad = pd.to_numeric(group_sorted.get("st_rad"), errors="coerce")
        rows.append(
            {
                "hostname": hostname,
                "n_observed_planets": n_planets,
                "dominant_discoverymethod_system": dominant_method,
                "dominant_method_group": method_group,
                "dominant_method_fraction": dominant_fraction,
                "system_st_mass_median": float(st_mass[st_mass > 0].median()) if not st_mass[st_mass > 0].empty else np.nan,
                "system_st_rad_median": float(st_rad[st_rad > 0].median()) if not st_rad[st_rad > 0].empty else np.nan,
                "stellar_mass_bin": stellar_mass_bin(float(st_mass[st_mass > 0].median())) if not st_mass[st_mass > 0].empty else "unknown_mass_star",
            }
        )
    return pd.DataFrame(rows)


def build_analog_pools(catalog: pd.DataFrame) -> AnalogPools:
    frame = catalog.copy()
    frame["candidate_logP"] = safe_log10_series(frame["pl_orbper"])
    frame["candidate_loga"] = safe_log10_series(frame["pl_orbsmax"])
    frame["candidate_logMstar"] = safe_log10_series(frame["st_mass"])
    pools: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for columns in [["candidate_logP", "candidate_loga"], ["candidate_logP", "candidate_loga", "candidate_logMstar"]]:
        subset = frame.dropna(subset=columns).copy()
        if subset.empty:
            continue
        matrix = subset[columns].to_numpy(dtype=float)
        scales = np.nanstd(matrix, axis=0)
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
        pools["|".join(columns)] = (subset.reset_index(drop=True), matrix, scales)
    return AnalogPools(pools=pools)


def _interpolate_log_quantity(inner: pd.Series, outer: pd.Series, column: str, fraction: float) -> float:
    inner_value = pd.to_numeric(pd.Series([inner.get(column)]), errors="coerce").iloc[0]
    outer_value = pd.to_numeric(pd.Series([outer.get(column)]), errors="coerce").iloc[0]
    if not np.isfinite(inner_value) or not np.isfinite(outer_value) or inner_value <= 0 or outer_value <= 0:
        return np.nan
    log_inner = math.log10(float(inner_value))
    log_outer = math.log10(float(outer_value))
    return float(log_inner + fraction * (log_outer - log_inner))


def estimate_candidate_properties(candidates: pd.DataFrame, catalog: pd.DataFrame, n_analogs: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    analogs = build_analog_pools(catalog)
    indexed_catalog = catalog.set_index("pl_name", drop=False)
    out = candidates.copy()
    out["candidate_logP"] = safe_log10_series(out["candidate_period_days"])
    out["candidate_loga"] = safe_log10_series(out["candidate_semimajor_au"])
    out["candidate_logMstar"] = safe_log10_series(out["candidate_stellar_mass"])

    records: list[dict[str, object]] = []
    for _, row in out.iterrows():
        inner = indexed_catalog.loc[row["inner_planet"]] if row["inner_planet"] in indexed_catalog.index else pd.Series(dtype=object)
        outer = indexed_catalog.loc[row["outer_planet"]] if row["outer_planet"] in indexed_catalog.index else pd.Series(dtype=object)
        fraction = float(row.get("candidate_fraction", np.nan))
        log_mass_interp = _interpolate_log_quantity(inner, outer, "pl_bmasse", fraction)
        log_radius_interp = _interpolate_log_quantity(inner, outer, "pl_rade", fraction)

        feature_columns = ["candidate_logP", "candidate_loga"]
        if np.isfinite(pd.to_numeric(pd.Series([row.get("candidate_logMstar")]), errors="coerce").iloc[0]):
            feature_columns.append("candidate_logMstar")
        key = "|".join(feature_columns)
        support_count = 0
        mass_median = np.nan
        mass_q16 = np.nan
        mass_q84 = np.nan
        radius_median = np.nan
        radius_q16 = np.nan
        radius_q84 = np.nan
        estimation_method = "analog_neighbors"

        if key in analogs.pools:
            pool, matrix, scales = analogs.pools[key]
            point = row[feature_columns].to_numpy(dtype=float)
            distances = np.linalg.norm((matrix - point) / scales, axis=1)
            n_select = min(n_analogs, len(pool))
            nearest_idx = np.argpartition(distances, n_select - 1)[:n_select] if n_select > 0 else np.array([], dtype=int)
            nearest = pool.iloc[nearest_idx].copy() if n_select > 0 else pd.DataFrame()
            support_count = int(len(nearest))
            masses = pd.to_numeric(nearest.get("pl_bmasse"), errors="coerce").dropna()
            radii = pd.to_numeric(nearest.get("pl_rade"), errors="coerce").dropna()
            if not masses.empty:
                mass_median = float(masses.median())
                mass_q16 = float(masses.quantile(0.16))
                mass_q84 = float(masses.quantile(0.84))
            if not radii.empty:
                radius_median = float(radii.median())
                radius_q16 = float(radii.quantile(0.16))
                radius_q84 = float(radii.quantile(0.84))

        if not np.isfinite(mass_median) and np.isfinite(log_mass_interp):
            mass_median = 10 ** float(log_mass_interp)
            mass_q16 = mass_median
            mass_q84 = mass_median
            estimation_method = "local_interpolation"
        if not np.isfinite(radius_median) and np.isfinite(log_radius_interp):
            radius_median = 10 ** float(log_radius_interp)
            radius_q16 = radius_median
            radius_q84 = radius_median
            estimation_method = "local_interpolation" if estimation_method == "analog_neighbors" else "hybrid"

        records.append(
            {
                "candidate_id": row["candidate_id"],
                "logM_candidate_interp": log_mass_interp,
                "logR_candidate_interp": log_radius_interp,
                "candidate_mass_median": mass_median,
                "candidate_mass_q16": mass_q16,
                "candidate_mass_q84": mass_q84,
                "candidate_radius_median": radius_median,
                "candidate_radius_q16": radius_q16,
                "candidate_radius_q84": radius_q84,
                "analog_support_count": support_count,
                "candidate_estimation_method": estimation_method,
            }
        )
    return out.merge(pd.DataFrame(records), on="candidate_id", how="left")


def estimate_analog_support_score(candidates: pd.DataFrame, n_analogs: int) -> pd.Series:
    count_frac = pd.to_numeric(candidates.get("analog_support_count"), errors="coerce").fillna(0.0) / max(float(n_analogs), 1.0)
    count_frac = count_frac.clip(0.0, 1.0)
    has_mass = pd.to_numeric(candidates.get("candidate_mass_median"), errors="coerce").notna().astype(float)
    has_radius = pd.to_numeric(candidates.get("candidate_radius_median"), errors="coerce").notna().astype(float)
    return clip01(0.6 * count_frac + 0.2 * has_mass + 0.2 * has_radius)


def quality_from_row(row: pd.Series, column: str) -> float:
    observed = row.get(f"{column}_was_observed")
    physically_derived = row.get(f"{column}_was_physically_derived")
    imputed = row.get(f"{column}_was_imputed")
    if bool(observed):
        return 1.0
    if bool(physically_derived):
        return 0.8
    if bool(imputed):
        return 0.35
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    if np.isfinite(value):
        return 0.7
    return 0.0


def build_data_quality_score(candidates: pd.DataFrame, catalog: pd.DataFrame) -> pd.Series:
    if candidates.empty:
        return pd.Series(dtype=float)
    indexed = catalog.set_index("pl_name", drop=False)
    scores: list[float] = []
    for _, row in candidates.iterrows():
        quality_terms: list[float] = []
        for planet_name in [row.get("inner_planet"), row.get("outer_planet")]:
            if planet_name in indexed.index:
                planet = indexed.loc[planet_name]
                quality_terms.extend(
                    [
                        quality_from_row(planet, "pl_orbper"),
                        quality_from_row(planet, "pl_orbsmax"),
                        quality_from_row(planet, "pl_bmasse"),
                        quality_from_row(planet, "pl_rade"),
                    ]
                )
        if row.get("candidate_position_method") == "kepler":
            quality_terms.append(0.85)
        elif row.get("candidate_position_method") == "geometric_interpolation":
            quality_terms.append(0.65)
        if np.isfinite(pd.to_numeric(pd.Series([row.get("candidate_stellar_mass")]), errors="coerce").iloc[0]):
            quality_terms.append(0.80)
        if np.isfinite(pd.to_numeric(pd.Series([row.get("candidate_stellar_radius")]), errors="coerce").iloc[0]):
            quality_terms.append(0.80)
        if pd.notna(row.get("candidate_mass_median")):
            quality_terms.append(0.75)
        if pd.notna(row.get("candidate_radius_median")):
            quality_terms.append(0.75)
        scores.append(float(np.mean(quality_terms)) if quality_terms else 0.0)
    return pd.Series(scores, index=candidates.index, dtype=float).clip(0.0, 1.0)


def compute_priority_scores(candidates: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = (
        weights["w_gap"] * pd.to_numeric(candidates["gap_score"], errors="coerce").fillna(0.0)
        + weights["w_topology"] * pd.to_numeric(candidates["topology_score"], errors="coerce").fillna(0.0)
        + weights["w_analog"] * pd.to_numeric(candidates["analog_support_score"], errors="coerce").fillna(0.0)
        + weights["w_detectability"] * pd.to_numeric(candidates["missing_detectability_score"], errors="coerce").fillna(0.0)
        + weights["w_data_quality"] * pd.to_numeric(candidates["data_quality_score"], errors="coerce").fillna(0.0)
    )
    return score.clip(0.0, 1.0)


def assign_priority_class(candidates: pd.DataFrame, high_gap_ratio: float) -> pd.Series:
    classes: list[str] = []
    for _, row in candidates.iterrows():
        score = float(pd.to_numeric(pd.Series([row.get("candidate_priority_score")]), errors="coerce").fillna(0.0).iloc[0])
        geometric = float(row.get("gap_period_ratio", 0.0)) >= high_gap_ratio or float(row.get("gap_score", 0.0)) >= 0.75
        topological = float(row.get("topology_score", 0.0)) >= 0.35
        observational = float(row.get("missing_detectability_score", 0.0)) >= 0.35 and str(row.get("likely_missing_due_to", "unknown")) != "unknown"
        if score >= 0.70 and geometric and topological and observational:
            classes.append("high")
        elif score >= 0.40:
            classes.append("medium")
        else:
            classes.append("low")
    return pd.Series(classes, index=candidates.index, dtype="string")


def build_candidate_interpretation(row: pd.Series) -> str:
    topology_text = "con soporte topologico previo" if float(row.get("topology_score", 0.0)) >= 0.35 else "sin soporte topologico fuerte"
    detectability_text = str(row.get("likely_missing_due_to", "unknown")).replace("_", " ")
    text = (
        f"Intervalo orbital prioritario entre {row.get('inner_planet')} y {row.get('outer_planet')} en {row.get('hostname')}; "
        f"gap_ratio={float(row.get('gap_period_ratio', np.nan)):.2f}, {topology_text}, "
        f"y candidato a submuestreo intra-sistema posiblemente consistente con {detectability_text}. "
        "La lectura es de priorizacion observacional y no implica un planeta confirmado."
    )
    validate_prudent_text(text)
    return text


def summarize_systems(system_metadata: pd.DataFrame, candidates: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped_candidates = candidates.groupby("hostname") if not candidates.empty else {}
    grouped_gaps = gaps.groupby("hostname") if not gaps.empty else {}
    for _, system in system_metadata.iterrows():
        hostname = system["hostname"]
        system_candidates = grouped_candidates.get_group(hostname) if hostname in getattr(grouped_candidates, "groups", {}) else pd.DataFrame()
        system_gaps = grouped_gaps.get_group(hostname) if hostname in getattr(grouped_gaps, "groups", {}) else pd.DataFrame()
        best = system_candidates.sort_values("candidate_priority_score", ascending=False).head(1) if not system_candidates.empty else pd.DataFrame()
        max_gap = float(pd.to_numeric(system_gaps.get("gap_ratio"), errors="coerce").max()) if not system_gaps.empty else np.nan
        max_score = float(pd.to_numeric(system_candidates.get("candidate_priority_score"), errors="coerce").max()) if not system_candidates.empty else np.nan
        mean_score = float(pd.to_numeric(system_candidates.get("candidate_priority_score"), errors="coerce").mean()) if not system_candidates.empty else np.nan
        system_class = "low"
        if not system_candidates.empty:
            best_class = str(best["candidate_priority_class"].iloc[0])
            system_class = best_class
        rows.append(
            {
                "hostname": hostname,
                "n_observed_planets": int(system["n_observed_planets"]),
                "n_candidate_missing_planets": int(len(system_candidates)),
                "max_gap_period_ratio": max_gap,
                "max_candidate_priority_score": max_score,
                "mean_candidate_priority_score": mean_score,
                "dominant_discoverymethod_system": system["dominant_discoverymethod_system"],
                "best_candidate_id": best["candidate_id"].iloc[0] if not best.empty else pd.NA,
                "best_candidate_period_days": best["candidate_period_days"].iloc[0] if not best.empty else np.nan,
                "best_candidate_semimajor_au": best["candidate_semimajor_au"].iloc[0] if not best.empty else np.nan,
                "system_priority_class": system_class,
            }
        )
    return pd.DataFrame(rows)
