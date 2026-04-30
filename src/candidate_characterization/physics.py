"""Physical derivations and observability proxies."""
from __future__ import annotations

import numpy as np
import pandas as pd

DENSITY_EARTH_G_CM3 = 5.514
R_SUN_AU = 0.00465047


def density_from_mass_radius(mass_earth, radius_earth, rho_earth: float = DENSITY_EARTH_G_CM3):
    m = pd.to_numeric(pd.Series(mass_earth), errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(pd.Series(radius_earth), errors="coerce").to_numpy(dtype=float)
    out = rho_earth * m / np.power(r, 3)
    out[~np.isfinite(out) | (m <= 0) | (r <= 0)] = np.nan
    return out


def semi_major_axis_from_period(period_days, st_mass_solar):
    """Kepler approximation: a_AU = [M_star (P/365.25)^2]^(1/3)."""
    p = pd.to_numeric(pd.Series(period_days), errors="coerce").to_numpy(dtype=float)
    m = pd.to_numeric(pd.Series(st_mass_solar), errors="coerce").to_numpy(dtype=float)
    out = np.power(m * np.power(p / 365.25, 2), 1.0 / 3.0)
    out[~np.isfinite(out) | (p <= 0) | (m <= 0)] = np.nan
    return out


def insol_from_luminosity(st_lum_solar, a_au):
    """Relative insolation in Earth units: S/S_earth ≈ L_star/L_sun / a^2."""
    L = pd.to_numeric(pd.Series(st_lum_solar), errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(pd.Series(a_au), errors="coerce").to_numpy(dtype=float)
    out = L / np.power(a, 2)
    out[~np.isfinite(out) | (L <= 0) | (a <= 0)] = np.nan
    return out


def equilibrium_temperature(teff_k, st_rad_solar, a_au, bond_albedo: float = 0.30):
    """Approximate equilibrium temperature.

    T_eq = T_eff * sqrt(R_star / (2a)) * (1 - A)^(1/4), with R_star in AU.
    """
    t = pd.to_numeric(pd.Series(teff_k), errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(pd.Series(st_rad_solar), errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(pd.Series(a_au), errors="coerce").to_numpy(dtype=float)
    out = t * np.sqrt((r * R_SUN_AU) / (2.0 * a)) * np.power(max(0.0, 1.0 - bond_albedo), 0.25)
    out[~np.isfinite(out) | (t <= 0) | (r <= 0) | (a <= 0)] = np.nan
    return out


def transit_probability(st_rad_solar, a_au, pl_rade_earth=None):
    """Geometric transit probability approximation.

    p_tr ≈ (R_star + R_p)/a. If R_p is absent, uses R_star/a only.
    """
    r_star = pd.to_numeric(pd.Series(st_rad_solar), errors="coerce").to_numpy(dtype=float) * R_SUN_AU
    a = pd.to_numeric(pd.Series(a_au), errors="coerce").to_numpy(dtype=float)
    if pl_rade_earth is None:
        r_planet_au = 0.0
    else:
        # Earth radius in AU ≈ 4.2635e-5
        r_planet_au = pd.to_numeric(pd.Series(pl_rade_earth), errors="coerce").to_numpy(dtype=float) * 4.2635e-5
    out = (r_star + r_planet_au) / a
    out = np.clip(out, 0, 1)
    out[~np.isfinite(out) | (r_star <= 0) | (a <= 0)] = np.nan
    return out


def rv_semiamplitude_proxy(mass_earth, period_days, st_mass_solar):
    """Relative radial-velocity signal proxy.

    K ∝ M_p / (M_star^(2/3) P^(1/3)). This ignores eccentricity and sin(i), so it is a
    ranking proxy, not a calibrated m/s prediction.
    """
    m_p = pd.to_numeric(pd.Series(mass_earth), errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pd.Series(period_days), errors="coerce").to_numpy(dtype=float)
    m_s = pd.to_numeric(pd.Series(st_mass_solar), errors="coerce").to_numpy(dtype=float)
    out = m_p / (np.power(m_s, 2.0 / 3.0) * np.power(p, 1.0 / 3.0))
    out[~np.isfinite(out) | (m_p <= 0) | (p <= 0) | (m_s <= 0)] = np.nan
    return out


def mutual_hill_separation(a1_au, a2_au, m1_earth, m2_earth, st_mass_solar):
    """Approximate orbital spacing in mutual Hill radii.

    Delta = |a2-a1| / R_H,m; R_H,m = ((m1+m2)/(3M_star))^(1/3) * (a1+a2)/2.
    """
    a1 = np.asarray(a1_au, dtype=float)
    a2 = np.asarray(a2_au, dtype=float)
    m1 = np.asarray(m1_earth, dtype=float)
    m2 = np.asarray(m2_earth, dtype=float)
    ms = np.asarray(st_mass_solar, dtype=float) * 332946.0
    rh = np.power((m1 + m2) / (3.0 * ms), 1.0 / 3.0) * (a1 + a2) / 2.0
    out = np.abs(a2 - a1) / rh
    out[~np.isfinite(out) | (rh <= 0)] = np.nan
    return out
