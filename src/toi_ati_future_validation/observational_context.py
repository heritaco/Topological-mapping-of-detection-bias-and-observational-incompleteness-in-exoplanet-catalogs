from __future__ import annotations

import numpy as np
import pandas as pd


def expected_direction_from_method(method: str | None) -> str:
    value = str(method or "Unknown").strip()
    mapping = {
        "Radial Velocity": "menor masa planetaria, menor proxy RV y senales dinamicas mas debiles a escala orbital comparable",
        "Transit": "menor radio, mayor periodo y menor probabilidad geometrica de transito",
        "Imaging": "menor luminosidad planetaria, menor masa o menor separacion angular observable",
        "Microlensing": "ventana geometrica especifica y sensibilidad dependiente del evento, no continuidad instrumental directa",
    }
    return mapping.get(value, "direccion fisica dependiente del metodo dominante y de la ventana observacional")


def add_observational_context(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    if "discoverymethod" in df.columns:
        method_series = df["discoverymethod"]
    elif "top_method" in df.columns:
        method_series = df["top_method"]
    else:
        method_series = pd.Series(["Unknown"] * len(df), index=df.index)
    df["method"] = method_series.astype(str)
    df["future_observation_direction"] = df["method"].map(expected_direction_from_method)

    if {"pl_bmasse", "pl_orbper"}.issubset(df.columns):
        mass = pd.to_numeric(df["pl_bmasse"], errors="coerce")
        period = pd.to_numeric(df["pl_orbper"], errors="coerce")
        valid = (mass > 0) & (period > 0)
        df["rv_proxy"] = np.nan
        df.loc[valid, "rv_proxy"] = np.log10(mass.loc[valid]) - (1.0 / 3.0) * np.log10(period.loc[valid])
        if "st_mass" in df.columns:
            st_mass = pd.to_numeric(df["st_mass"], errors="coerce")
            valid_star = valid & (st_mass > 0)
            df["rv_proxy_with_star"] = np.nan
            df.loc[valid_star, "rv_proxy_with_star"] = (
                np.log10(mass.loc[valid_star])
                - (1.0 / 3.0) * np.log10(period.loc[valid_star])
                - (2.0 / 3.0) * np.log10(st_mass.loc[valid_star])
            )
    return df

