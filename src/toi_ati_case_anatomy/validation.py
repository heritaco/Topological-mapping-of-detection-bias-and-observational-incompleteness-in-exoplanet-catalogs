from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

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


def compute_delta_n(n_expected: float, n_observed: float) -> float:
    if pd.isna(n_expected) or pd.isna(n_observed):
        return np.nan
    return float(n_expected) - float(n_observed)


def compute_delta_rel(n_expected: float, n_observed: float, epsilon: float = 1e-9) -> float:
    if pd.isna(n_expected) or pd.isna(n_observed):
        return np.nan
    return safe_ratio(compute_delta_n(n_expected, n_observed), n_expected, epsilon=epsilon)


def classify_deficit(delta_rel: float, n_expected: float | None = None) -> str:
    if pd.isna(delta_rel):
        return "undefined_reference"
    if n_expected is not None and (pd.isna(n_expected) or float(n_expected) <= 0):
        return "undefined_reference"
    if delta_rel < 0:
        return "overpopulated_reference"
    if delta_rel <= 0.10:
        return "no_deficit"
    if delta_rel <= 0.30:
        return "weak_deficit"
    if delta_rel <= 0.60:
        return "moderate_deficit"
    return "strong_deficit"


def deficit_stability_label(values: Iterable[float]) -> str:
    cleaned = [float(value) for value in values if not pd.isna(value)]
    if len(cleaned) < 2:
        return "undefined"
    positive = sum(value > 0 for value in cleaned)
    if positive == len(cleaned):
        return "consistent_positive_deficit"
    if positive >= 1:
        return "radius_sensitive_deficit"
    return "no_consistent_deficit"


def suspicious_delta_rel(delta_rel: float, delta_n: float, n_expected: float) -> bool:
    if pd.isna(delta_rel) or pd.isna(delta_n) or pd.isna(n_expected):
        return False
    if float(n_expected) <= 0:
        return False
    return float(delta_rel) > 1 and float(delta_n) <= float(n_expected) + 1e-9
