from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import normalize_score_series
from .io import discover_optional_table, read_table, repo_relative


@dataclass
class TopologyResources:
    toi_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    ati_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    shadow_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    membership_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    paths: dict[str, str | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _normalize_node_column(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "node_id" in frame.columns:
        return frame
    for candidate in ["node", "cluster_id", "mapper_node", "id"]:
        if candidate in frame.columns:
            return frame.rename(columns={candidate: "node_id"})
    return frame


def _normalize_planet_column(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "pl_name" in frame.columns:
        return frame
    for candidate in ["anchor_pl_name", "planet_name", "member_id", "member", "planet", "name"]:
        if candidate in frame.columns:
            return frame.rename(columns={candidate: "pl_name"})
    return frame


def load_topology_resources(
    *,
    toi_table: str | None,
    ati_table: str | None,
    shadow_table: str | None,
    node_membership_table: str | None,
    logger: logging.Logger,
) -> TopologyResources:
    toi_path = discover_optional_table(
        toi_table,
        label="tabla TOI",
        patterns=["*regional_toi*.csv", "*toi*.csv", "*topological*incompleteness*.csv"],
        preferred_keywords=["regional_toi_scores", "topological_incompleteness_index", "toi"],
        logger=logger,
    )
    ati_path = discover_optional_table(
        ati_table,
        label="tabla ATI",
        patterns=["*anchor_ati*.csv", "*ati*.csv"],
        preferred_keywords=["anchor_ati_scores", "topological_incompleteness_index", "ati"],
        logger=logger,
    )
    shadow_path = discover_optional_table(
        shadow_table,
        label="tabla shadow",
        patterns=["*node_observational_shadow*.csv", "*shadow*.csv"],
        preferred_keywords=["node_observational_shadow_metrics", "shadow"],
        logger=logger,
    )
    membership_path = discover_optional_table(
        node_membership_table,
        label="tabla de membresia nodo-planeta",
        patterns=["*membership*shadow*inputs*.csv", "*membership*observational*metadata*.csv", "*membership*.csv"],
        preferred_keywords=["membership_with_shadow_inputs_orbital", "membership_with_observational_metadata_orbital", "membership"],
        logger=logger,
    )

    resources = TopologyResources(
        toi_table=_normalize_node_column(read_table(toi_path)) if toi_path else pd.DataFrame(),
        ati_table=_normalize_planet_column(read_table(ati_path)) if ati_path else pd.DataFrame(),
        shadow_table=_normalize_node_column(read_table(shadow_path)) if shadow_path else pd.DataFrame(),
        membership_table=_normalize_planet_column(_normalize_node_column(read_table(membership_path))) if membership_path else pd.DataFrame(),
        paths={
            "toi_table": repo_relative(toi_path),
            "ati_table": repo_relative(ati_path),
            "shadow_table": repo_relative(shadow_path),
            "node_membership_table": repo_relative(membership_path),
        },
        warnings=[],
    )
    if resources.toi_table.empty:
        resources.warnings.append("no TOI table found; topology prior will use zero fallback")
        logger.warning("no TOI table found; topology prior will use zero fallback")
    else:
        logger.info("loaded TOI table from %s", resources.paths["toi_table"])
    if resources.ati_table.empty:
        resources.warnings.append("no ATI table found; topology prior will use zero fallback")
        logger.warning("no ATI table found; topology prior will use zero fallback")
    else:
        logger.info("loaded ATI table from %s", resources.paths["ati_table"])
    if resources.shadow_table.empty:
        resources.warnings.append("no shadow table found; topology prior will use zero fallback")
        logger.warning("no shadow table found; topology prior will use zero fallback")
    else:
        logger.info("loaded shadow table from %s", resources.paths["shadow_table"])
    if resources.membership_table.empty:
        resources.warnings.append("no node membership table found; node-level TOI/shadow will be unavailable")
        logger.warning("no node membership table found; node-level TOI/shadow will be unavailable")
    else:
        logger.info("loaded node membership table from %s", resources.paths["node_membership_table"])
    return resources


def build_planet_topology_prior(resources: TopologyResources) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not resources.toi_table.empty and "TOI" in resources.toi_table.columns:
        toi = _normalize_node_column(resources.toi_table.copy())
        toi["planet_TOI_score"] = pd.to_numeric(toi["TOI"], errors="coerce")
        toi["planet_TOI_norm"] = normalize_score_series(toi["planet_TOI_score"])
    else:
        toi = pd.DataFrame(columns=["node_id", "planet_TOI_score", "planet_TOI_norm"])

    if not resources.shadow_table.empty and "shadow_score" in resources.shadow_table.columns:
        shadow = _normalize_node_column(resources.shadow_table.copy())
        shadow["planet_shadow_score"] = pd.to_numeric(shadow["shadow_score"], errors="coerce")
        shadow["planet_shadow_norm"] = normalize_score_series(shadow["planet_shadow_score"])
    elif not toi.empty and "shadow_score" in toi.columns:
        shadow = toi.rename(columns={"shadow_score": "planet_shadow_score"}).copy()
        shadow["planet_shadow_score"] = pd.to_numeric(shadow["planet_shadow_score"], errors="coerce")
        shadow["planet_shadow_norm"] = normalize_score_series(shadow["planet_shadow_score"])
    else:
        shadow = pd.DataFrame(columns=["node_id", "planet_shadow_score", "planet_shadow_norm"])

    node_scores = toi.merge(shadow[["node_id", "planet_shadow_score", "planet_shadow_norm"]], on="node_id", how="outer")
    node_scores = _normalize_node_column(node_scores)

    if not resources.membership_table.empty:
        membership = _normalize_planet_column(_normalize_node_column(resources.membership_table.copy()))
        membership = membership.merge(node_scores, on="node_id", how="left")
        grouped = membership.groupby("pl_name", dropna=True)
        node_based = grouped.agg(
            planet_TOI_score=("planet_TOI_score", "max"),
            planet_shadow_score=("planet_shadow_score", "max"),
            planet_TOI_norm=("planet_TOI_norm", "max"),
            planet_shadow_norm=("planet_shadow_norm", "max"),
            topology_node_count=("node_id", "nunique"),
        ).reset_index()
        frames.append(node_based)

    if not resources.ati_table.empty and "ATI" in resources.ati_table.columns:
        ati = _normalize_planet_column(resources.ati_table.copy())
        ati["planet_ATI_score"] = pd.to_numeric(ati["ATI"], errors="coerce")
        ati["planet_ATI_norm"] = normalize_score_series(ati["planet_ATI_score"])
        ati_grouped = ati.groupby("pl_name", dropna=True).agg(
            planet_ATI_score=("planet_ATI_score", "max"),
            planet_ATI_norm=("planet_ATI_norm", "max"),
        ).reset_index()
        frames.append(ati_grouped)

    if not frames:
        return pd.DataFrame(columns=["pl_name", "planet_TOI_score", "planet_shadow_score", "planet_ATI_score", "topology_prior_planet"])

    planet_prior = frames[0]
    for frame in frames[1:]:
        planet_prior = planet_prior.merge(frame, on="pl_name", how="outer")
    for column in [
        "planet_TOI_score",
        "planet_shadow_score",
        "planet_ATI_score",
        "planet_TOI_norm",
        "planet_shadow_norm",
        "planet_ATI_norm",
    ]:
        if column not in planet_prior.columns:
            planet_prior[column] = np.nan
    norm_columns = ["planet_TOI_norm", "planet_shadow_norm", "planet_ATI_norm"]
    planet_prior["topology_prior_planet"] = pd.concat(
        [pd.to_numeric(planet_prior[column], errors="coerce").fillna(0.0) for column in norm_columns],
        axis=1,
    ).max(axis=1)
    return planet_prior


def attach_topology_to_candidates(candidates: pd.DataFrame, resources: TopologyResources) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    planet_prior = build_planet_topology_prior(resources)
    if planet_prior.empty:
        out = candidates.copy()
        out["topology_prior_inner"] = 0.0
        out["topology_prior_outer"] = 0.0
        out["topology_prior_gap"] = 0.0
        out["gap_shadow_score"] = np.nan
        out["gap_TOI_score"] = np.nan
        out["gap_ATI_score"] = np.nan
        out["topology_score"] = 0.0
        return out

    inner = planet_prior.add_prefix("inner_").rename(columns={"inner_pl_name": "inner_planet"})
    outer = planet_prior.add_prefix("outer_").rename(columns={"outer_pl_name": "outer_planet"})
    out = candidates.merge(inner, on="inner_planet", how="left").merge(outer, on="outer_planet", how="left")
    out["topology_prior_inner"] = pd.to_numeric(out.get("inner_topology_prior_planet"), errors="coerce").fillna(0.0)
    out["topology_prior_outer"] = pd.to_numeric(out.get("outer_topology_prior_planet"), errors="coerce").fillna(0.0)
    out["topology_prior_gap"] = pd.concat([out["topology_prior_inner"], out["topology_prior_outer"]], axis=1).max(axis=1)
    out["gap_shadow_score"] = pd.concat(
        [
            pd.to_numeric(out.get("inner_planet_shadow_score"), errors="coerce"),
            pd.to_numeric(out.get("outer_planet_shadow_score"), errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    out["gap_TOI_score"] = pd.concat(
        [
            pd.to_numeric(out.get("inner_planet_TOI_score"), errors="coerce"),
            pd.to_numeric(out.get("outer_planet_TOI_score"), errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    out["gap_ATI_score"] = pd.concat(
        [
            pd.to_numeric(out.get("inner_planet_ATI_score"), errors="coerce"),
            pd.to_numeric(out.get("outer_planet_ATI_score"), errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    topology_norm = pd.concat(
        [
            pd.to_numeric(out.get("inner_planet_TOI_norm"), errors="coerce").fillna(0.0),
            pd.to_numeric(out.get("outer_planet_TOI_norm"), errors="coerce").fillna(0.0),
            pd.to_numeric(out.get("inner_planet_shadow_norm"), errors="coerce").fillna(0.0),
            pd.to_numeric(out.get("outer_planet_shadow_norm"), errors="coerce").fillna(0.0),
            pd.to_numeric(out.get("inner_planet_ATI_norm"), errors="coerce").fillna(0.0),
            pd.to_numeric(out.get("outer_planet_ATI_norm"), errors="coerce").fillna(0.0),
            pd.to_numeric(out["topology_prior_gap"], errors="coerce").fillna(0.0),
        ],
        axis=1,
    )
    out["topology_score"] = topology_norm.max(axis=1).clip(0.0, 1.0)
    return out
