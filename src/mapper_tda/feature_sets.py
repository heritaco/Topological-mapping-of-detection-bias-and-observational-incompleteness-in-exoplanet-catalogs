from __future__ import annotations

from typing import Iterable


MAPPER_FEATURE_SPACES: dict[str, list[str]] = {
    "phys_min": ["pl_rade", "pl_bmasse"],
    "phys_density": ["pl_rade", "pl_bmasse", "pl_dens"],
    "orbital": ["pl_orbper", "pl_orbsmax"],
    "thermal": ["pl_insol", "pl_eqt"],
    "orb_thermal": ["pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt"],
    "joint_no_density": ["pl_rade", "pl_bmasse", "pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt"],
    "joint": ["pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt"],
}

LEGACY_SPACE_ALIASES: dict[str, str] = {
    "phys": "phys_density",
    "orb": "orb_thermal",
}

ALL_SPACE_KEYS = list(MAPPER_FEATURE_SPACES.keys())

SPACE_COMPARISON_ORDER = [
    "phys_min",
    "phys_density",
    "orbital",
    "thermal",
    "orb_thermal",
    "joint_no_density",
    "joint",
]


def canonical_space_name(space: str) -> str:
    normalized = str(space).strip().lower()
    if normalized == "all":
        return normalized
    canonical = LEGACY_SPACE_ALIASES.get(normalized, normalized)
    if canonical not in MAPPER_FEATURE_SPACES:
        raise ValueError(f"Espacio Mapper no soportado: {space}")
    return canonical


def expand_space_selection(space: str) -> list[str]:
    canonical = canonical_space_name(space)
    if canonical == "all":
        return list(ALL_SPACE_KEYS)
    return [canonical]


def features_for_space(space: str) -> list[str]:
    return list(MAPPER_FEATURE_SPACES[canonical_space_name(space)])


def has_density_feature(space: str | Iterable[str]) -> bool:
    if isinstance(space, str):
        return "pl_dens" in features_for_space(space)
    return "pl_dens" in list(space)
