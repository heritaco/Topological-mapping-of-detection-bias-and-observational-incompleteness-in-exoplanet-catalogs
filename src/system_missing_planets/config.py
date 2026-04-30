from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SOLAR_RADIUS_AU = 0.00465047
SOLAR_RADIUS_EARTH = 109.076

BANNED_PHRASES = [
    "planeta faltante confirmado",
    "descubrimos un planeta",
    "este sistema tiene x planetas reales faltantes",
    "encontramos planetas faltantes reales",
]


@dataclass
class ScoreWeights:
    w_gap: float = 0.30
    w_topology: float = 0.25
    w_analog: float = 0.20
    w_detectability: float = 0.15
    w_data_quality: float = 0.10

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class SystemMissingPlanetsConfig:
    catalog: str
    output_dir: str
    mode: str = "all"
    hostname: str | None = None
    min_planets_per_system: int = 2
    min_gap_ratio: float = 2.8
    high_gap_ratio: float = 5.0
    max_candidates_per_gap: int = 4
    n_analogs: int = 35
    random_state: int = 42
    toi_table: str | None = None
    ati_table: str | None = None
    shadow_table: str | None = None
    node_membership_table: str | None = None
    make_figures: bool = False
    make_latex_summary: bool = False
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    def validate(self) -> None:
        if self.mode not in {"all", "single"}:
            raise ValueError("mode debe ser 'all' o 'single'.")
        if self.mode == "single" and not self.hostname:
            raise ValueError("hostname es obligatorio cuando mode='single'.")
        if self.min_planets_per_system < 2:
            raise ValueError("min_planets_per_system debe ser >= 2.")
        if self.min_gap_ratio <= 1:
            raise ValueError("min_gap_ratio debe ser > 1.")
        if self.high_gap_ratio < self.min_gap_ratio:
            raise ValueError("high_gap_ratio debe ser >= min_gap_ratio.")
        if self.max_candidates_per_gap < 1:
            raise ValueError("max_candidates_per_gap debe ser >= 1.")
        if self.n_analogs < 1:
            raise ValueError("n_analogs debe ser >= 1.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["weights"] = self.weights.to_dict()
        return payload


def validate_prudent_text(text: str) -> None:
    lower = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lower:
            raise AssertionError(f"Frase no prudente detectada: {phrase}")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
