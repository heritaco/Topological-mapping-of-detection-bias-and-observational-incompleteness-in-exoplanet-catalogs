"""Input/output adapters for the existing exoplanet Mapper repository."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .config import (
    ATI_ANCHOR_PATHS,
    CATALOG_CANDIDATE_PATHS,
    MISSING_PLANET_CANDIDATE_PATHS,
    TOI_REGION_PATHS,
    CharacterizationConfig,
    resolve_first_existing,
)
from .physics import density_from_mass_radius, semi_major_axis_from_period, insol_from_luminosity, equilibrium_temperature
from .utils import normalize_columns, first_existing_column, coalesce_columns, safe_numeric


CATALOG_ALIASES: Dict[str, list[str]] = {
    "pl_name": ["pl_name", "planet_name", "name"],
    "hostname": ["hostname", "host", "host_name", "star", "star_name"],
    "pl_rade": ["pl_rade", "radius_earth", "planet_radius", "rp", "pl_radius"],
    "pl_bmasse": ["pl_bmasse", "mass_earth", "planet_mass", "mp", "pl_mass", "pl_masse"],
    "pl_dens": ["pl_dens", "density", "planet_density", "rho"],
    "pl_orbper": ["pl_orbper", "period", "period_days", "orbital_period", "p", "p_days"],
    "pl_orbsmax": ["pl_orbsmax", "a", "a_au", "semi_major_axis", "semimajor_axis", "semi_major_axis_au"],
    "pl_insol": ["pl_insol", "insolation", "insol", "s_incident", "stellar_flux"],
    "pl_eqt": ["pl_eqt", "eqt", "teq", "t_eq", "equilibrium_temperature"],
    "st_mass": ["st_mass", "stellar_mass", "mstar", "host_mass"],
    "st_rad": ["st_rad", "stellar_radius", "rstar", "host_radius"],
    "st_teff": ["st_teff", "stellar_teff", "teff", "host_teff"],
    "st_lum": ["st_lum", "stellar_luminosity", "luminosity", "host_lum"],
    "st_met": ["st_met", "st_metfe", "stellar_metallicity", "metallicity", "feh"],
    "st_logg": ["st_logg", "stellar_logg", "logg"],
    "sy_dist": ["sy_dist", "distance", "distance_pc"],
    "sy_pnum": ["sy_pnum", "planet_count", "n_planets", "system_planet_count"],
    "discoverymethod": ["discoverymethod", "discovery_method", "method"],
    "disc_year": ["disc_year", "discovery_year", "year"],
    "disc_facility": ["disc_facility", "discovery_facility", "facility"],
}

CANDIDATE_ALIASES: Dict[str, list[str]] = {
    "candidate_id": ["candidate_id", "id", "anchor", "candidate", "name"],
    "hostname": ["hostname", "host", "host_name", "star", "star_name"],
    "node_id": ["node_id", "node", "mapper_node", "region", "region_id"],
    "pl_orbper": [
        "candidate_period_days",
        "p_star",
        "p_candidate",
        "p_candidate_days",
        "period_candidate",
        "period_days",
        "pl_orbper",
        "orbital_period",
    ],
    "pl_orbsmax": [
        "candidate_semimajor_au",
        "a_star",
        "a_candidate",
        "a_candidate_au",
        "semimajor_candidate",
        "a_au",
        "pl_orbsmax",
        "semi_major_axis",
    ],
    "inner_period": ["inner_period", "p_inner", "period_inner", "period_left", "left_period"],
    "outer_period": ["outer_period", "p_outer", "period_outer", "period_right", "right_period"],
    "candidate_score": ["candidate_priority_score", "candidate_score", "score", "priority_score"],
    "topology_score": ["topology_score", "mapper_score", "topological_score"],
    "gap_score": ["gap_score", "gap_period_ratio"],
    "toi": ["gap_toi_score", "toi", "TOI", "topological_observational_incompleteness"],
    "ati": ["gap_ati_score", "ati", "ATI", "anchor_topological_incompleteness"],
    "shadow_score": ["gap_shadow_score", "shadow_score", "shadow"],
}


def _standardize_known_columns(df: pd.DataFrame, aliases: Dict[str, list[str]]) -> pd.DataFrame:
    df = normalize_columns(df)
    out = df.copy()
    for canonical, alias_list in aliases.items():
        if canonical not in out.columns:
            src = first_existing_column(out, alias_list)
            if src is not None:
                out[canonical] = out[src]
    return out


def load_catalog(cfg: CharacterizationConfig) -> Tuple[pd.DataFrame, Path]:
    repo_root = Path(cfg.paths.repo_root).resolve()
    path = resolve_first_existing(repo_root, cfg.paths.catalog_csv, CATALOG_CANDIDATE_PATHS)
    if path is None or not path.exists():
        raise FileNotFoundError(
            "No catalog CSV found. Pass --catalog-csv or place an imputed catalog at one of: "
            + ", ".join(CATALOG_CANDIDATE_PATHS)
        )
    df = pd.read_csv(path)
    df = _standardize_known_columns(df, CATALOG_ALIASES)
    df = derive_catalog_columns(df, cfg)
    return df, path


def derive_catalog_columns(df: pd.DataFrame, cfg: CharacterizationConfig) -> pd.DataFrame:
    out = df.copy()
    for c in ["pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt", "st_mass", "st_rad", "st_teff", "st_lum"]:
        if c in out.columns:
            out[c] = safe_numeric(out[c])

    if "pl_dens" not in out.columns:
        out["pl_dens"] = np.nan
    if {"pl_bmasse", "pl_rade"}.issubset(out.columns):
        derived = density_from_mass_radius(out["pl_bmasse"], out["pl_rade"], cfg.physical.density_earth_g_cm3)
        out["pl_dens"] = out["pl_dens"].where(out["pl_dens"].notna(), derived)

    if "pl_orbsmax" not in out.columns:
        out["pl_orbsmax"] = np.nan
    if {"pl_orbper", "st_mass"}.issubset(out.columns):
        derived_a = semi_major_axis_from_period(out["pl_orbper"], out["st_mass"])
        out["pl_orbsmax"] = out["pl_orbsmax"].where(out["pl_orbsmax"].notna(), derived_a)

    if "pl_insol" not in out.columns:
        out["pl_insol"] = np.nan
    if {"st_lum", "pl_orbsmax"}.issubset(out.columns):
        derived_s = insol_from_luminosity(out["st_lum"], out["pl_orbsmax"])
        out["pl_insol"] = out["pl_insol"].where(out["pl_insol"].notna(), derived_s)

    if "pl_eqt" not in out.columns:
        out["pl_eqt"] = np.nan
    if {"st_teff", "st_rad", "pl_orbsmax"}.issubset(out.columns):
        derived_t = equilibrium_temperature(out["st_teff"], out["st_rad"], out["pl_orbsmax"], cfg.physical.default_bond_albedo)
        out["pl_eqt"] = out["pl_eqt"].where(out["pl_eqt"].notna(), derived_t)

    if "sy_pnum" not in out.columns and "hostname" in out.columns:
        out["sy_pnum"] = out.groupby("hostname")["hostname"].transform("size")
    return out


def load_candidates(cfg: CharacterizationConfig, catalog: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Path]:
    repo_root = Path(cfg.paths.repo_root).resolve()
    path = resolve_first_existing(repo_root, cfg.paths.candidates_csv, MISSING_PLANET_CANDIDATE_PATHS)
    if path is None or not path.exists():
        raise FileNotFoundError(
            "No candidate CSV found. Pass --candidates-csv. Expected columns include hostname and candidate period. "
            "Default searched paths: " + ", ".join(MISSING_PLANET_CANDIDATE_PATHS)
        )
    df = pd.read_csv(path)
    df = _standardize_known_columns(df, CANDIDATE_ALIASES)
    df = attach_topological_tables(df, cfg)
    if catalog is not None:
        df = enrich_candidates_from_catalog(df, catalog, cfg)
    return df, path


def attach_topological_tables(candidates: pd.DataFrame, cfg: CharacterizationConfig) -> pd.DataFrame:
    out = candidates.copy()
    repo_root = Path(cfg.paths.repo_root).resolve()
    toi_path = resolve_first_existing(repo_root, cfg.paths.toi_regions_csv, TOI_REGION_PATHS)
    if toi_path and toi_path.exists() and "node_id" in out.columns:
        toi = normalize_columns(pd.read_csv(toi_path))
        if "node_id" not in toi.columns:
            nid = first_existing_column(toi, ["node", "region", "region_id"])
            if nid:
                toi["node_id"] = toi[nid]
        if "node_id" in toi.columns:
            keep_cols = [c for c in ["node_id", "toi", "shadow_score", "i_r3", "c_phys", "s_net", "toi_conservative"] if c in toi.columns]
            if len(keep_cols) > 1:
                out = out.merge(toi[keep_cols].drop_duplicates("node_id"), on="node_id", how="left", suffixes=("", "_region"))
                for c in ["toi", "shadow_score"]:
                    rc = f"{c}_region"
                    if rc in out.columns:
                        out[c] = out[c].where(out[c].notna(), out[rc]) if c in out.columns else out[rc]
    ati_path = resolve_first_existing(repo_root, cfg.paths.ati_anchors_csv, ATI_ANCHOR_PATHS)
    if ati_path and ati_path.exists() and "node_id" in out.columns:
        ati = normalize_columns(pd.read_csv(ati_path))
        if "node_id" not in ati.columns:
            nid = first_existing_column(ati, ["node", "region", "region_id"])
            if nid:
                ati["node_id"] = ati[nid]
        if "node_id" in ati.columns:
            keep_cols = [c for c in ["node_id", "ati", "toi", "anchor_pl_name", "delta_rel_neighbors_best", "delta_rel_analog_best"] if c in ati.columns]
            if len(keep_cols) > 1:
                ati_agg = ati[keep_cols].copy()
                numeric_cols = [c for c in ati_agg.columns if c != "node_id" and pd.api.types.is_numeric_dtype(ati_agg[c])]
                text_cols = [c for c in ati_agg.columns if c not in {"node_id", *numeric_cols}]
                grouped = ati_agg.groupby("node_id", as_index=False)[numeric_cols].max() if numeric_cols else ati_agg[["node_id"]].drop_duplicates()
                for c in text_cols:
                    first_text = ati_agg.groupby("node_id", as_index=False)[c].first()
                    grouped = grouped.merge(first_text, on="node_id", how="left")
                out = out.merge(grouped.drop_duplicates("node_id"), on="node_id", how="left", suffixes=("", "_ati"))
                for c in ["ati", "toi"]:
                    rc = f"{c}_ati"
                    if rc in out.columns:
                        out[c] = out[c].where(out[c].notna(), out[rc]) if c in out.columns else out[rc]
    return out


def enrich_candidates_from_catalog(candidates: pd.DataFrame, catalog: pd.DataFrame, cfg: CharacterizationConfig) -> pd.DataFrame:
    out = candidates.copy()
    if "candidate_id" not in out.columns:
        if "hostname" in out.columns:
            out["candidate_id"] = out["hostname"].astype(str) + "_candidate_" + (out.index + 1).astype(str)
        else:
            out["candidate_id"] = "candidate_" + (out.index + 1).astype(str)

    # Attach host-level stellar properties by median over known planets in same host.
    host_cols = ["hostname", "st_mass", "st_rad", "st_teff", "st_lum", "st_met", "st_logg", "sy_dist", "sy_pnum"]
    available_host_cols = [c for c in host_cols if c in catalog.columns]
    if "hostname" in out.columns and "hostname" in catalog.columns and len(available_host_cols) > 1:
        numeric_cols = [c for c in available_host_cols if c != "hostname"]
        host_meta = catalog[available_host_cols].copy()
        for c in numeric_cols:
            host_meta[c] = safe_numeric(host_meta[c])
        host_meta = host_meta.groupby("hostname", as_index=False)[numeric_cols].median(numeric_only=True)
        out = out.merge(host_meta, on="hostname", how="left", suffixes=("", "_host"))
        for c in numeric_cols:
            hc = f"{c}_host"
            if hc in out.columns:
                out[c] = out[c].where(out[c].notna(), out[hc]) if c in out.columns else out[hc]

    # Derive a, insolation, and equilibrium temperature if absent.
    for c in ["pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt", "st_mass", "st_rad", "st_teff", "st_lum"]:
        if c in out.columns:
            out[c] = safe_numeric(out[c])

    if "pl_orbsmax" not in out.columns:
        out["pl_orbsmax"] = np.nan
    if {"pl_orbper", "st_mass"}.issubset(out.columns):
        out["pl_orbsmax"] = out["pl_orbsmax"].where(
            out["pl_orbsmax"].notna(), semi_major_axis_from_period(out["pl_orbper"], out["st_mass"])
        )

    if "pl_insol" not in out.columns:
        out["pl_insol"] = np.nan
    if {"st_lum", "pl_orbsmax"}.issubset(out.columns):
        out["pl_insol"] = out["pl_insol"].where(
            out["pl_insol"].notna(), insol_from_luminosity(out["st_lum"], out["pl_orbsmax"])
        )

    if "pl_eqt" not in out.columns:
        out["pl_eqt"] = np.nan
    if {"st_teff", "st_rad", "pl_orbsmax"}.issubset(out.columns):
        out["pl_eqt"] = out["pl_eqt"].where(
            out["pl_eqt"].notna(),
            equilibrium_temperature(out["st_teff"], out["st_rad"], out["pl_orbsmax"], cfg.physical.default_bond_albedo),
        )

    out = add_system_gap_features(out, catalog)
    return out


def add_system_gap_features(candidates: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    for c in ["inner_period", "outer_period"]:
        if c not in out.columns:
            out[c] = np.nan
    if "hostname" not in out.columns or "hostname" not in catalog.columns or "pl_orbper" not in catalog.columns:
        return _compute_gap_ratios(out)

    periods_by_host = {
        h: np.sort(pd.to_numeric(g["pl_orbper"], errors="coerce").dropna().to_numpy(dtype=float))
        for h, g in catalog.groupby("hostname")
    }
    inner_vals = []
    outer_vals = []
    for _, row in out.iterrows():
        p = row.get("pl_orbper", np.nan)
        host = row.get("hostname")
        periods = periods_by_host.get(host, np.array([]))
        if not np.isfinite(p) or periods.size == 0:
            inner_vals.append(np.nan)
            outer_vals.append(np.nan)
            continue
        inner = periods[periods < p]
        outer = periods[periods > p]
        inner_vals.append(float(inner[-1]) if inner.size else np.nan)
        outer_vals.append(float(outer[0]) if outer.size else np.nan)
    out["inner_period"] = out["inner_period"].where(out["inner_period"].notna(), inner_vals)
    out["outer_period"] = out["outer_period"].where(out["outer_period"].notna(), outer_vals)
    return _compute_gap_ratios(out)


def _compute_gap_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p = safe_numeric(out["pl_orbper"]) if "pl_orbper" in out.columns else pd.Series(np.nan, index=out.index)
    inner = safe_numeric(out["inner_period"]) if "inner_period" in out.columns else pd.Series(np.nan, index=out.index)
    outer = safe_numeric(out["outer_period"]) if "outer_period" in out.columns else pd.Series(np.nan, index=out.index)
    out["gap_ratio_inner"] = p / inner
    out["gap_ratio_outer"] = outer / p
    out["log_gap_ratio_inner"] = np.log10(out["gap_ratio_inner"].where(out["gap_ratio_inner"] > 0))
    out["log_gap_ratio_outer"] = np.log10(out["gap_ratio_outer"].where(out["gap_ratio_outer"] > 0))
    out["log_gap_width"] = np.log10(outer.where(outer > 0)) - np.log10(inner.where(inner > 0))
    return out
