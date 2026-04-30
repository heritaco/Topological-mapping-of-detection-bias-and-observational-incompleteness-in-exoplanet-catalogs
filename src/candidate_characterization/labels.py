"""Heuristic labels for interpretive planet classes."""
from __future__ import annotations

import numpy as np
import pandas as pd

RADIUS_CLASS_ORDER = [
    "rocky_size",
    "super_earth_size",
    "sub_neptune_size",
    "neptune_sub_jovian_size",
    "jovian_size",
]


def radius_class(radius_earth) -> pd.Series:
    r = pd.to_numeric(pd.Series(radius_earth), errors="coerce")
    out = pd.Series("unknown", index=r.index, dtype="object")
    out[(r > 0) & (r <= 1.6)] = "rocky_size"
    out[(r > 1.6) & (r <= 2.5)] = "super_earth_size"
    out[(r > 2.5) & (r <= 4.0)] = "sub_neptune_size"
    out[(r > 4.0) & (r <= 8.0)] = "neptune_sub_jovian_size"
    out[(r > 8.0)] = "jovian_size"
    return out


def mass_class(mass_earth) -> pd.Series:
    m = pd.to_numeric(pd.Series(mass_earth), errors="coerce")
    out = pd.Series("unknown", index=m.index, dtype="object")
    out[(m > 0) & (m <= 2.0)] = "low_mass_terrestrial"
    out[(m > 2.0) & (m <= 10.0)] = "super_earth_mass"
    out[(m > 10.0) & (m <= 30.0)] = "sub_neptune_mass"
    out[(m > 30.0) & (m <= 150.0)] = "neptune_saturn_mass"
    out[(m > 150.0)] = "jovian_mass"
    return out


def thermal_class(eqt_k) -> pd.Series:
    t = pd.to_numeric(pd.Series(eqt_k), errors="coerce")
    out = pd.Series("unknown", index=t.index, dtype="object")
    out[(t > 0) & (t < 200)] = "cold"
    out[(t >= 200) & (t < 350)] = "temperate"
    out[(t >= 350) & (t < 700)] = "warm"
    out[(t >= 700) & (t < 1200)] = "hot"
    out[(t >= 1200)] = "ultra_hot"
    return out


def orbit_class(period_days) -> pd.Series:
    p = pd.to_numeric(pd.Series(period_days), errors="coerce")
    out = pd.Series("unknown", index=p.index, dtype="object")
    out[(p > 0) & (p < 10)] = "short_period"
    out[(p >= 10) & (p < 100)] = "intermediate_period"
    out[(p >= 100) & (p < 1000)] = "long_period"
    out[(p >= 1000)] = "very_long_period"
    return out


def most_probable_class(prob_row: dict) -> str:
    class_probs = {k.replace("prob_", ""): v for k, v in prob_row.items() if k.startswith("prob_")}
    if not class_probs:
        return "unknown"
    return max(class_probs.items(), key=lambda kv: kv[1])[0]
