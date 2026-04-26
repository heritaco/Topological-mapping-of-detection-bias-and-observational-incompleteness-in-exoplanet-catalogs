from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EARTH_DENSITY_G_CM3 = 5.514
DENSITY_FORMULA = "pl_dens = 5.514 * pl_bmasse / pl_rade**3"


@dataclass(frozen=True)
class DensityDerivationAudit:
    """Counts that make the density derivation traceable."""

    column_created: bool
    observed_before: int
    missing_before: int
    derived_count: int
    missing_after: int
    formula: str = DENSITY_FORMULA


def derive_planet_density(df: pd.DataFrame) -> tuple[pd.DataFrame, DensityDerivationAudit]:
    """Fill missing ``pl_dens`` values from mass and radius when possible.

    Existing observed density values are preserved. The returned dataframe always
    includes ``pl_dens`` and ``pl_dens_source``.
    """

    if not {"pl_bmasse", "pl_rade"}.issubset(df.columns):
        missing = sorted({"pl_bmasse", "pl_rade"} - set(df.columns))
        raise KeyError(f"No se puede derivar pl_dens; faltan columnas: {missing}")

    out = df.copy()
    column_created = "pl_dens" not in out.columns
    if column_created:
        out["pl_dens"] = np.nan

    density = pd.to_numeric(out["pl_dens"], errors="coerce")
    mass = pd.to_numeric(out["pl_bmasse"], errors="coerce")
    radius = pd.to_numeric(out["pl_rade"], errors="coerce")

    observed_before = int(density.notna().sum())
    missing_before = int(density.isna().sum())
    valid_physics = (mass > 0) & (radius > 0)
    derived_mask = density.isna() & valid_physics
    derived_values = EARTH_DENSITY_G_CM3 * mass / radius.pow(3)
    density = density.mask(derived_mask, derived_values)

    source = pd.Series(pd.NA, index=out.index, dtype="object")
    source.loc[density.notna()] = "observed"
    source.loc[derived_mask] = "derived_from_pl_bmasse_pl_rade"
    source.loc[density.isna()] = pd.NA

    out["pl_dens"] = density
    out["pl_dens_source"] = source
    out.attrs["density_derivation"] = {
        "formula": DENSITY_FORMULA,
        "derived_count": int(derived_mask.sum()),
        "column_created": column_created,
    }

    audit = DensityDerivationAudit(
        column_created=column_created,
        observed_before=observed_before,
        missing_before=missing_before,
        derived_count=int(derived_mask.sum()),
        missing_after=int(density.isna().sum()),
    )
    return out, audit

