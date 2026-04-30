"""Feature registry and feature-set resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.exoplanet_tda.core.io import read_yaml


DEFAULT_REGISTRY_PATH = Path("configs/features/feature_registry.yaml")
DEFAULT_FEATURE_SETS_PATH = Path("configs/features/feature_sets.yaml")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    role: str
    leakage_risk: str
    recommended_action: str
    aliases: tuple[str, ...] = ()


@dataclass
class ResolvedFeatureSet:
    name: str
    features: list[str]
    groups: list[str]
    audit_only_groups: list[str]
    leakage_safe: bool
    missing_groups: list[str]
    specs: list[FeatureSpec]


def _as_path(path: str | Path | None, default: Path) -> Path:
    return Path(path) if path else default


def _feature_name_and_aliases(raw: Any) -> tuple[str, tuple[str, ...]]:
    if isinstance(raw, dict):
        aliases = tuple(str(item) for item in raw.get("aliases", []) or [])
        return str(raw["name"]), aliases
    return str(raw), ()


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class FeatureRegistry:
    """Load feature governance YAML and resolve named groups/sets."""

    def __init__(self, registry_path: str | Path | None = None, feature_sets_path: str | Path | None = None) -> None:
        self.registry_path = _as_path(registry_path, DEFAULT_REGISTRY_PATH)
        self.feature_sets_path = _as_path(feature_sets_path, DEFAULT_FEATURE_SETS_PATH)
        self.registry = read_yaml(self.registry_path)
        self.feature_sets_doc = read_yaml(self.feature_sets_path)
        self.groups: dict[str, dict[str, Any]] = self.registry.get("feature_groups", {}) or {}
        self.feature_sets: dict[str, dict[str, Any]] = self.feature_sets_doc.get("feature_sets", {}) or {}

    @property
    def default_candidate_feature_set(self) -> str:
        return str(self.feature_sets_doc.get("default_candidate_characterization", "candidate_characterization_minimal"))

    def group_specs(self, group_name: str) -> list[FeatureSpec]:
        group = self.groups.get(group_name)
        if not group:
            return []
        specs: list[FeatureSpec] = []
        for raw_feature in group.get("features", []) or []:
            name, aliases = _feature_name_and_aliases(raw_feature)
            specs.append(
                FeatureSpec(
                    name=name,
                    group=group_name,
                    role=str(group.get("role", "prediction")),
                    leakage_risk=str(group.get("leakage_risk", "low")),
                    recommended_action=str(group.get("recommended_action", "")),
                    aliases=aliases,
                )
            )
        return specs

    def resolve_group_names(self, item: str) -> list[str]:
        text = str(item).strip()
        if text in self.feature_sets:
            include = self.feature_sets[text].get("include", []) or []
            exclude = set(self.feature_sets[text].get("exclude", []) or [])
            return [group for group in include if group not in exclude]
        if "+" in text:
            groups: list[str] = []
            for part in text.split("+"):
                groups.extend(self.resolve_group_names(part.strip()))
            return _unique(groups)
        return [text]

    def resolve(self, name_or_expression: str, apply_set_excludes: bool = True) -> ResolvedFeatureSet:
        name = str(name_or_expression).strip()
        set_cfg = self.feature_sets.get(name, {})
        if set_cfg:
            group_names = []
            for item in set_cfg.get("include", []) or []:
                group_names.extend(self.resolve_group_names(str(item)))
            if apply_set_excludes:
                excluded = set(set_cfg.get("exclude", []) or [])
                group_names = [group for group in group_names if group not in excluded]
            audit_only_groups = [str(group) for group in set_cfg.get("audit_only", []) or []]
            leakage_safe = bool(set_cfg.get("leakage_safe", False))
        else:
            group_names = self.resolve_group_names(name)
            audit_only_groups = []
            leakage_safe = False

        group_names = _unique(group_names)
        missing_groups = [group for group in group_names if group not in self.groups]
        specs: list[FeatureSpec] = []
        for group in group_names:
            specs.extend(self.group_specs(group))
        return ResolvedFeatureSet(
            name=name,
            features=_unique(spec.name for spec in specs),
            groups=group_names,
            audit_only_groups=audit_only_groups,
            leakage_safe=leakage_safe,
            missing_groups=missing_groups,
            specs=specs,
        )

    def all_specs(self) -> list[FeatureSpec]:
        specs: list[FeatureSpec] = []
        for group in self.groups:
            specs.extend(self.group_specs(group))
        return specs

    def target_rule_exclusions(self, target: str) -> set[str]:
        rules = self.registry.get("target_rules", {}) or {}
        raw = rules.get(target, {}).get("exclude", []) if isinstance(rules.get(target, {}), dict) else []
        exclusions: set[str] = set()
        for item in raw:
            text = str(item)
            exclusions.add(text)
            if text in self.groups:
                exclusions.update(spec.name for spec in self.group_specs(text))
        return exclusions


def load_feature_registry(
    registry_path: str | Path | None = None,
    feature_sets_path: str | Path | None = None,
) -> FeatureRegistry:
    return FeatureRegistry(registry_path=registry_path, feature_sets_path=feature_sets_path)
