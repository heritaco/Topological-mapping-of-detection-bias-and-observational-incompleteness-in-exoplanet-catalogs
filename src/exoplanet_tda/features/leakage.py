"""Leakage checks and filters for feature governance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .registry import FeatureRegistry, ResolvedFeatureSet


TARGET_ALIASES = {
    "pl_bmasse": {"pl_bmasse", "log_pl_bmasse"},
    "log_pl_bmasse": {"pl_bmasse", "log_pl_bmasse"},
    "pl_rade": {"pl_rade", "log_pl_rade"},
    "log_pl_rade": {"pl_rade", "log_pl_rade"},
    "radius_class": {"radius_class", "planet_class", "pl_rade", "log_pl_rade", "pl_bmasse", "log_pl_bmasse", "pl_dens"},
    "planet_class": {"radius_class", "planet_class", "pl_rade", "log_pl_rade", "pl_bmasse", "log_pl_bmasse", "pl_dens"},
}

AUDIT_ONLY_FEATURES = {"discoverymethod", "disc_year", "disc_facility"}
TARGET_COLUMNS = {"pl_rade", "log_pl_rade", "pl_bmasse", "log_pl_bmasse", "pl_dens", "radius_class", "mass_class", "planet_class"}


@dataclass
class LeakageReport:
    features: list[str]
    removed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def apply_leakage_rules(
    features: Iterable[str],
    target: str | None = None,
    registry: FeatureRegistry | None = None,
    resolved_set: ResolvedFeatureSet | None = None,
    allow_observed_diagnostic: bool = False,
    allow_audit_features: bool = False,
    hypothetical_candidates: bool = True,
) -> LeakageReport:
    """Remove target leakage and audit-only controls from model inputs."""

    feature_list = _unique(str(feature) for feature in features)
    remove: set[str] = set()
    warnings: list[str] = []

    if registry is not None and target:
        remove.update(registry.target_rule_exclusions(target))
    if target:
        remove.update(TARGET_ALIASES.get(target, {target}))
    if not allow_observed_diagnostic:
        if target in {"radius_class", "planet_class"}:
            remove.update(TARGET_ALIASES["radius_class"])
    if hypothetical_candidates:
        remove.update(TARGET_COLUMNS.intersection(feature_list))

    audit_groups = set(resolved_set.audit_only_groups if resolved_set else [])
    if resolved_set and audit_groups and not allow_audit_features:
        for spec in resolved_set.specs:
            if spec.group in audit_groups:
                remove.add(spec.name)
    if not allow_audit_features:
        remove.update(AUDIT_ONLY_FEATURES.intersection(feature_list))

    filtered = [feature for feature in feature_list if feature not in remove]
    removed = [feature for feature in feature_list if feature in remove]

    if removed:
        warnings.append("Removed leakage-prone or audit-only features: " + ", ".join(removed))
    if allow_audit_features:
        warnings.append("Audit-only variables are enabled for sensitivity analysis; do not treat them as physical predictors.")
    return LeakageReport(features=filtered, removed=removed, warnings=warnings)
