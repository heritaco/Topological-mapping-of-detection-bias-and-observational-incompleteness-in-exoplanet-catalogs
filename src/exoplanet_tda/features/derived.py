"""Safe derived feature creation for feature governance."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd


R_SUN_AU = 0.00465047


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _warn(logger: logging.Logger | None, message: str) -> None:
    if logger is not None:
        logger.warning(message)


def _assign_if_absent(out: pd.DataFrame, column: str, values: pd.Series | np.ndarray | float) -> None:
    if column not in out.columns:
        out[column] = values


def add_derived_features(
    df: pd.DataFrame,
    logger: logging.Logger | None = None,
    predicted_mass_column: str | None = None,
) -> pd.DataFrame:
    """Add candidate, system, topology, and proxy features without overwriting raw columns."""

    out = df.copy()
    n = len(out)

    period_col = _first_existing(out, ["candidate_pl_orbper", "pl_orbper", "period_days", "candidate_period_days"])
    semimajor_col = _first_existing(out, ["candidate_pl_orbsmax", "pl_orbsmax", "a_au", "candidate_semimajor_au"])
    if period_col:
        _assign_if_absent(out, "candidate_pl_orbper", _numeric(out, period_col))
    else:
        _warn(logger, "candidate orbital period is unavailable; candidate_pl_orbper set to NaN")
        _assign_if_absent(out, "candidate_pl_orbper", pd.Series(np.nan, index=out.index))
    if semimajor_col:
        _assign_if_absent(out, "candidate_pl_orbsmax", _numeric(out, semimajor_col))
    else:
        _warn(logger, "candidate semimajor axis is unavailable; candidate_pl_orbsmax set to NaN")
        _assign_if_absent(out, "candidate_pl_orbsmax", pd.Series(np.nan, index=out.index))

    a = _numeric(out, "candidate_pl_orbsmax")
    period = _numeric(out, "candidate_pl_orbper")
    st_rad = _numeric(out, "st_rad")
    st_mass = _numeric(out, "st_mass")
    st_lum = _numeric(out, "st_lum")
    st_teff = _numeric(out, "st_teff")

    transit_proxy = np.where((st_rad > 0) & (a > 0), (st_rad * R_SUN_AU) / a, np.nan)
    _assign_if_absent(out, "transit_probability_proxy", transit_proxy)
    _assign_if_absent(out, "proxy_transit_probability_proxy", transit_proxy)

    if predicted_mass_column and predicted_mass_column in out.columns:
        mass_source = _numeric(out, predicted_mass_column)
    else:
        mass_source_col = _first_existing(out, ["predicted_mass_or_neighbor_mass", "neighbor_mass", "inner_neighbor_mass", "outer_neighbor_mass"])
        mass_source = _numeric(out, mass_source_col) if mass_source_col else pd.Series(np.nan, index=out.index)
    rv_proxy = np.where((mass_source > 0) & (st_mass > 0) & (period > 0), mass_source / ((st_mass ** (2.0 / 3.0)) * (period ** (1.0 / 3.0))), np.nan)
    _assign_if_absent(out, "rv_amplitude_proxy", rv_proxy)
    _assign_if_absent(out, "proxy_rv_amplitude_proxy", rv_proxy)

    candidate_insol = np.where((st_lum > 0) & (a > 0), st_lum / (a**2), np.nan)
    _assign_if_absent(out, "candidate_insol", candidate_insol)
    _assign_if_absent(out, "proxy_candidate_insol", candidate_insol)

    missing_eqt = [col for col in ["st_rad", "st_teff", "candidate_pl_orbsmax"] if col not in out.columns]
    if missing_eqt:
        _warn(logger, "candidate_eqt requires stellar radius, stellar effective temperature, and candidate semimajor axis; set to NaN")
        candidate_eqt = pd.Series(np.nan, index=out.index)
    else:
        candidate_eqt = np.where((st_teff > 0) & (st_rad > 0) & (a > 0), st_teff * np.sqrt((st_rad * R_SUN_AU) / (2.0 * a)), np.nan)
    _assign_if_absent(out, "candidate_eqt", candidate_eqt)
    _assign_if_absent(out, "proxy_candidate_eqt", candidate_eqt)

    if "hostname" in out.columns:
        _assign_if_absent(out, "n_known_planets", out.groupby("hostname")["hostname"].transform("size"))
    else:
        _assign_if_absent(out, "n_known_planets", pd.Series(np.nan, index=out.index))

    inner_p = _numeric(out, _first_existing(out, ["inner_neighbor_period", "inner_period", "p_inner"]) or "")
    outer_p = _numeric(out, _first_existing(out, ["outer_neighbor_period", "outer_period", "p_outer"]) or "")
    _assign_if_absent(out, "inner_neighbor_period", inner_p)
    _assign_if_absent(out, "outer_neighbor_period", outer_p)
    _assign_if_absent(out, "period_ratio_inner", np.where((period > 0) & (inner_p > 0), period / inner_p, np.nan))
    _assign_if_absent(out, "period_ratio_outer", np.where((period > 0) & (outer_p > 0), outer_p / period, np.nan))
    log_gap_width = np.log10(outer_p.where(outer_p > 0)) - np.log10(inner_p.where(inner_p > 0))
    _assign_if_absent(out, "log_gap_width", log_gap_width)
    denom = np.log10(outer_p.where(outer_p > 0)) - np.log10(inner_p.where(inner_p > 0))
    normalized = (np.log10(period.where(period > 0)) - np.log10(inner_p.where(inner_p > 0))) / denom
    _assign_if_absent(out, "normalized_gap_position", normalized)

    for canonical, aliases in {
        "TOI": ["TOI", "toi", "gap_toi_score"],
        "ATI_region_context": ["ATI_region_context", "ati", "ATI", "gap_ati_score"],
        "shadow_score": ["shadow_score", "gap_shadow_score"],
    }.items():
        source = _first_existing(out, aliases)
        if source:
            _assign_if_absent(out, canonical, out[source])
            _assign_if_absent(out, "topo_" + canonical.lower(), out[source])
        else:
            _assign_if_absent(out, canonical, pd.Series(np.nan, index=out.index))

    for column in ["candidate_position_index", "inner_neighbor_radius", "outer_neighbor_radius", "inner_neighbor_mass", "outer_neighbor_mass"]:
        _assign_if_absent(out, column, pd.Series(np.nan, index=out.index))
    for column in ["node_degree", "node_size", "component_size", "distance_to_node_medoid", "local_method_entropy", "local_method_purity"]:
        _assign_if_absent(out, column, pd.Series(np.nan, index=out.index))
        _assign_if_absent(out, "topo_" + column, out[column])

    if n != len(out):
        raise RuntimeError("Derived feature creation changed row count unexpectedly.")
    return out
