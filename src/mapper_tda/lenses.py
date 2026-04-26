from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .feature_sets import canonical_space_name
from .preprocessing import safe_log10


EPS = 1e-8


def _pca_projection(Z: np.ndarray, n_components: int) -> tuple[np.ndarray, PCA, int]:
    if Z.ndim != 2 or Z.shape[0] == 0 or Z.shape[1] == 0:
        raise ValueError("La matriz Z para construir el lens esta vacia.")
    fitted_components = min(n_components, Z.shape[0], Z.shape[1])
    if fitted_components <= 0:
        raise ValueError("No hay suficientes observaciones para PCA.")
    pca = PCA(n_components=fitted_components)
    scores = pca.fit_transform(Z)
    return scores, pca, fitted_components


def make_lens_pca2(Z: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, dict[str, Any]]:
    del random_state
    scores, pca, fitted_components = _pca_projection(Z, n_components=2)
    if fitted_components == 1:
        scores = np.column_stack([scores[:, 0], np.zeros(Z.shape[0])])
    metadata = {
        "lens": "pca2",
        "explained_variance_ratio": [
            float(value) for value in np.pad(pca.explained_variance_ratio_, (0, max(0, 2 - fitted_components)))
        ],
        "n_components_fitted": int(fitted_components),
    }
    return np.asarray(scores[:, :2], dtype=float), metadata


def make_lens_density(Z: np.ndarray, k_density: int = 15, random_state: int = 42) -> tuple[np.ndarray, dict[str, Any]]:
    del random_state
    scores, pca, _ = _pca_projection(Z, n_components=1)
    pc1 = scores[:, 0]
    if Z.shape[0] <= 1:
        kth_distance = np.zeros(Z.shape[0], dtype=float)
        effective_k = 0
    else:
        effective_k = min(max(1, k_density), Z.shape[0] - 1)
        neighbors = NearestNeighbors(n_neighbors=effective_k + 1)
        distances, _ = neighbors.fit(Z).kneighbors(Z)
        kth_distance = distances[:, -1]
    density_score = np.log(kth_distance + EPS)
    return (
        np.column_stack([pc1, density_score]),
        {
            "lens": "density",
            "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0]),
            "k_density_requested": int(k_density),
            "k_density_effective": int(effective_k),
        },
    )


def _domain_coordinates(physical_df: pd.DataFrame, space: str) -> tuple[pd.Series, pd.Series, list[str]]:
    canonical = canonical_space_name(space)
    if canonical == "phys_min":
        return safe_log10(physical_df["pl_bmasse"]), safe_log10(physical_df["pl_rade"]), [
            "log10(pl_bmasse)",
            "log10(pl_rade)",
        ]
    if canonical == "phys_density":
        return safe_log10(physical_df["pl_dens"]), safe_log10(physical_df["pl_rade"]), [
            "log10(pl_dens)",
            "log10(pl_rade)",
        ]
    if canonical == "orbital":
        return safe_log10(physical_df["pl_orbper"]), safe_log10(physical_df["pl_orbsmax"]), [
            "log10(pl_orbper)",
            "log10(pl_orbsmax)",
        ]
    if canonical == "thermal":
        return safe_log10(physical_df["pl_insol"]), pd.to_numeric(physical_df["pl_eqt"], errors="coerce"), [
            "log10(pl_insol)",
            "pl_eqt",
        ]
    if canonical == "orb_thermal":
        return safe_log10(physical_df["pl_insol"]), pd.to_numeric(physical_df["pl_eqt"], errors="coerce"), [
            "log10(pl_insol)",
            "pl_eqt",
        ]
    if canonical in {"joint_no_density", "joint"}:
        values = pd.to_numeric(physical_df["pl_dens"], errors="coerce") if "pl_dens" in physical_df.columns else None
        if canonical == "joint" and values is not None and values.notna().any():
            return safe_log10(values), safe_log10(physical_df["pl_rade"]), ["log10(pl_dens)", "log10(pl_rade)"]
        scores, _, _ = _pca_projection(
            physical_df[
                [column for column in ["pl_rade", "pl_bmasse", "pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt"] if column in physical_df.columns]
            ]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(method="ffill")
            .fillna(method="bfill")
            .to_numpy(dtype=float),
            n_components=2,
        )
        return pd.Series(scores[:, 0], index=physical_df.index), pd.Series(scores[:, 1], index=physical_df.index), [
            "physical_pca1",
            "physical_pca2",
        ]
    raise ValueError(f"Espacio no soportado para lens domain: {space}")


def make_lens_domain(physical_df: pd.DataFrame, space: str) -> tuple[np.ndarray, dict[str, Any]]:
    x, y, labels = _domain_coordinates(physical_df, space)
    mask = x.notna() & y.notna()
    if not bool(mask.all()):
        raise ValueError(f"El lens domain requiere valores completos para el espacio {space}.")
    return np.column_stack([x.to_numpy(dtype=float), y.to_numpy(dtype=float)]), {
        "lens": "domain",
        "space": canonical_space_name(space),
        "columns": labels,
    }
