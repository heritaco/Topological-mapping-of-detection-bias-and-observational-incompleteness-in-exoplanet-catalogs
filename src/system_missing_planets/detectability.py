from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SOLAR_RADIUS_AU, SOLAR_RADIUS_EARTH


def transit_probability_proxy(st_rad: float, semimajor_au: float) -> float:
    if not np.isfinite(st_rad) or st_rad <= 0 or not np.isfinite(semimajor_au) or semimajor_au <= 0:
        return np.nan
    return float((st_rad * SOLAR_RADIUS_AU) / semimajor_au)


def transit_depth_proxy(st_rad: float, radius_earth: float) -> float:
    if not np.isfinite(st_rad) or st_rad <= 0 or not np.isfinite(radius_earth) or radius_earth <= 0:
        return np.nan
    stellar_radius_earth = st_rad * SOLAR_RADIUS_EARTH
    return float((radius_earth / stellar_radius_earth) ** 2)


def rv_proxy(mass_earth: float, st_mass: float, period_days: float) -> float:
    if not np.isfinite(mass_earth) or mass_earth <= 0:
        return np.nan
    if not np.isfinite(st_mass) or st_mass <= 0 or not np.isfinite(period_days) or period_days <= 0:
        return np.nan
    return float(mass_earth / (np.power(st_mass, 2.0 / 3.0) * np.power(period_days, 1.0 / 3.0)))


def observed_system_proxies(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame(columns=["hostname"])
    frame = catalog.copy()
    frame["observed_transit_probability_proxy"] = [
        transit_probability_proxy(st_rad, semimajor)
        for st_rad, semimajor in zip(pd.to_numeric(frame.get("st_rad"), errors="coerce"), pd.to_numeric(frame.get("pl_orbsmax"), errors="coerce"))
    ]
    frame["observed_transit_depth_proxy"] = [
        transit_depth_proxy(st_rad, radius)
        for st_rad, radius in zip(pd.to_numeric(frame.get("st_rad"), errors="coerce"), pd.to_numeric(frame.get("pl_rade"), errors="coerce"))
    ]
    frame["observed_rv_proxy"] = [
        rv_proxy(mass, st_mass, period)
        for mass, st_mass, period in zip(
            pd.to_numeric(frame.get("pl_bmasse"), errors="coerce"),
            pd.to_numeric(frame.get("st_mass"), errors="coerce"),
            pd.to_numeric(frame.get("pl_orbper"), errors="coerce"),
        )
    ]
    summary = frame.groupby("hostname", dropna=True).agg(
        observed_period_median=("pl_orbper", "median"),
        observed_transit_probability_median=("observed_transit_probability_proxy", "median"),
        observed_transit_depth_median=("observed_transit_depth_proxy", "median"),
        observed_rv_proxy_median=("observed_rv_proxy", "median"),
    ).reset_index()
    return summary


def _difficulty_from_relative(relative: float) -> float:
    if not np.isfinite(relative):
        return 0.0
    if relative <= 0:
        return 1.0
    return float(np.clip(1.0 - min(relative, 1.0), 0.0, 1.0))


def _long_period_score(relative_period: float) -> float:
    if not np.isfinite(relative_period) or relative_period <= 1:
        return 0.0
    return float(np.clip(np.log10(relative_period) / np.log10(10.0), 0.0, 1.0))


def attach_detectability(candidates: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    system_summary = observed_system_proxies(catalog)
    out = candidates.merge(system_summary, on="hostname", how="left")
    out["transit_probability_proxy"] = [
        transit_probability_proxy(st_rad, semimajor)
        for st_rad, semimajor in zip(
            pd.to_numeric(out.get("candidate_stellar_radius"), errors="coerce"),
            pd.to_numeric(out.get("candidate_semimajor_au"), errors="coerce"),
        )
    ]
    out["transit_depth_proxy"] = [
        transit_depth_proxy(st_rad, radius)
        for st_rad, radius in zip(
            pd.to_numeric(out.get("candidate_stellar_radius"), errors="coerce"),
            pd.to_numeric(out.get("candidate_radius_median"), errors="coerce"),
        )
    ]
    out["rv_K_proxy"] = [
        rv_proxy(mass, st_mass, period)
        for mass, st_mass, period in zip(
            pd.to_numeric(out.get("candidate_mass_median"), errors="coerce"),
            pd.to_numeric(out.get("candidate_stellar_mass"), errors="coerce"),
            pd.to_numeric(out.get("candidate_period_days"), errors="coerce"),
        )
    ]
    out["candidate_transit_probability_relative_to_observed_median"] = pd.to_numeric(out["transit_probability_proxy"], errors="coerce") / pd.to_numeric(
        out["observed_transit_probability_median"], errors="coerce"
    )
    out["candidate_transit_depth_relative_to_observed_median"] = pd.to_numeric(out["transit_depth_proxy"], errors="coerce") / pd.to_numeric(
        out["observed_transit_depth_median"], errors="coerce"
    )
    out["candidate_RV_proxy_relative_to_observed_median"] = pd.to_numeric(out["rv_K_proxy"], errors="coerce") / pd.to_numeric(
        out["observed_rv_proxy_median"], errors="coerce"
    )
    out["candidate_period_relative_to_observed_median"] = pd.to_numeric(out["candidate_period_days"], errors="coerce") / pd.to_numeric(
        out["observed_period_median"], errors="coerce"
    )

    detectability_scores: list[float] = []
    reasons: list[str] = []
    for _, row in out.iterrows():
        rel_transit_probability = pd.to_numeric(pd.Series([row.get("candidate_transit_probability_relative_to_observed_median")]), errors="coerce").iloc[0]
        rel_transit_depth = pd.to_numeric(pd.Series([row.get("candidate_transit_depth_relative_to_observed_median")]), errors="coerce").iloc[0]
        rel_rv = pd.to_numeric(pd.Series([row.get("candidate_RV_proxy_relative_to_observed_median")]), errors="coerce").iloc[0]
        rel_period = pd.to_numeric(pd.Series([row.get("candidate_period_relative_to_observed_median")]), errors="coerce").iloc[0]
        transit_prob_score = _difficulty_from_relative(rel_transit_probability)
        transit_depth_score = _difficulty_from_relative(rel_transit_depth)
        rv_score = _difficulty_from_relative(rel_rv)
        long_period_score = _long_period_score(rel_period)
        method_group = str(row.get("dominant_method_group", "Mixed"))

        if method_group == "Transit":
            score = 0.45 * transit_prob_score + 0.35 * transit_depth_score + 0.20 * long_period_score
            if transit_depth_score >= max(transit_prob_score, long_period_score) and transit_depth_score >= 0.25:
                reason = "shallow_transit_depth"
            elif transit_prob_score >= max(transit_depth_score, long_period_score) and transit_prob_score >= 0.25:
                reason = "low_transit_probability"
            elif long_period_score >= 0.35:
                reason = "long_period"
            else:
                reason = "unknown"
        elif method_group == "Radial Velocity":
            score = 0.60 * rv_score + 0.40 * long_period_score
            if rv_score >= long_period_score and rv_score >= 0.25:
                reason = "weak_RV_signal"
            elif long_period_score >= 0.35:
                reason = "long_period"
            else:
                reason = "unknown"
        else:
            score = 0.30 * transit_prob_score + 0.20 * transit_depth_score + 0.30 * rv_score + 0.20 * long_period_score
            if max(transit_prob_score, transit_depth_score, rv_score, long_period_score) < 0.25:
                reason = "unknown"
            elif rv_score == max(transit_prob_score, transit_depth_score, rv_score, long_period_score):
                reason = "weak_RV_signal"
            elif transit_depth_score == max(transit_prob_score, transit_depth_score, rv_score, long_period_score):
                reason = "shallow_transit_depth"
            elif transit_prob_score == max(transit_prob_score, transit_depth_score, rv_score, long_period_score):
                reason = "low_transit_probability"
            else:
                reason = "long_period"
        detectability_scores.append(float(np.clip(score, 0.0, 1.0)))
        reasons.append(reason)
    out["missing_detectability_score"] = detectability_scores
    out["likely_missing_due_to"] = pd.Series(reasons, dtype="string")
    return out
