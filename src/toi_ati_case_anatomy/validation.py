from __future__ import annotations

from typing import Iterable

FORBIDDEN_CLAIMS = (
    "planetas faltantes confirmados",
    "descubrimos planetas",
    "faltan exactamente",
    "prediccion definitiva",
)


def contains_forbidden_claim(text: str, forbidden: Iterable[str] = FORBIDDEN_CLAIMS) -> list[str]:
    lower = text.lower()
    return [phrase for phrase in forbidden if phrase.lower() in lower]


def safe_ratio(numerator: float, denominator: float, epsilon: float = 1e-9) -> float:
    return float(numerator) / (float(denominator) + epsilon)
