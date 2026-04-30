"""Weighted topological/orbital analog search for candidate planets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .features import add_engineered_features
from .labels import radius_class, RADIUS_CLASS_ORDER
from .utils import weighted_quantile, safe_numeric


@dataclass
class AnalogResult:
    summaries: pd.DataFrame
    neighbors: pd.DataFrame


class WeightedAnalogCharacterizer:
    """Find nearest observed planets in known feature space and summarize targets.

    The analog model is deliberately non-parametric. It is useful when the ML model
    is uncertain or when one wants an interpretable support set for a presentation.
    """

    def __init__(self, features: Sequence[str], k: int = 75, temperature: float = 1.0):
        self.features = list(features)
        self.k = int(k)
        self.temperature = float(temperature)
        self.preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
        self.nn: NearestNeighbors | None = None
        self.catalog_: pd.DataFrame | None = None
        self.X_scaled_: np.ndarray | None = None

    def fit(self, catalog: pd.DataFrame) -> "WeightedAnalogCharacterizer":
        df = add_engineered_features(catalog)
        available = [f for f in self.features if f in df.columns]
        self.features = available
        if not available:
            raise ValueError("No analog features available.")
        X = df[available].apply(pd.to_numeric, errors="coerce")
        valid = X.notna().sum(axis=1) >= 2
        if "pl_rade" in df.columns:
            valid &= pd.to_numeric(df["pl_rade"], errors="coerce").notna()
        if "pl_bmasse" in df.columns:
            valid &= pd.to_numeric(df["pl_bmasse"], errors="coerce").notna()
        df = df.loc[valid].reset_index(drop=True)
        X = X.loc[valid].reset_index(drop=True)
        self.X_scaled_ = self.preprocessor.fit_transform(X)
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(df)), metric="euclidean")
        self.nn.fit(self.X_scaled_)
        self.catalog_ = df
        return self

    def characterize(self, candidates: pd.DataFrame, quantiles: Sequence[float] = (0.05, 0.5, 0.95)) -> AnalogResult:
        if self.nn is None or self.catalog_ is None or self.X_scaled_ is None:
            raise RuntimeError("WeightedAnalogCharacterizer must be fitted first.")
        cand = add_engineered_features(candidates)
        for f in self.features:
            if f not in cand.columns:
                cand[f] = np.nan
        Xc = cand[self.features].apply(pd.to_numeric, errors="coerce")
        Xc_scaled = self.preprocessor.transform(Xc)
        distances, indices = self.nn.kneighbors(Xc_scaled, n_neighbors=min(self.k, len(self.catalog_)))

        summary_rows: List[Dict] = []
        neighbor_rows: List[Dict] = []
        for i, row in cand.reset_index(drop=True).iterrows():
            idx = indices[i]
            dist = distances[i]
            tau = max(self.temperature, 1.0e-6)
            weights = np.exp(-(dist ** 2) / (2 * tau ** 2))
            local = self.catalog_.iloc[idx].copy()
            mass = pd.to_numeric(local.get("pl_bmasse", pd.Series(np.nan, index=local.index)), errors="coerce").to_numpy(dtype=float)
            radius = pd.to_numeric(local.get("pl_rade", pd.Series(np.nan, index=local.index)), errors="coerce").to_numpy(dtype=float)
            dens = pd.to_numeric(local.get("pl_dens", pd.Series(np.nan, index=local.index)), errors="coerce").to_numpy(dtype=float)
            q_mass = weighted_quantile(mass, quantiles, weights)
            q_radius = weighted_quantile(radius, quantiles, weights)
            q_dens = weighted_quantile(dens, quantiles, weights)

            classes = radius_class(radius)
            class_probs: Dict[str, float] = {}
            denom = float(np.nansum(weights)) if np.isfinite(weights).any() else 0.0
            for cls in RADIUS_CLASS_ORDER:
                cls_weight = float(np.nansum(weights[classes.to_numpy() == cls])) if denom > 0 else 0.0
                class_probs[f"analog_prob_{cls}"] = cls_weight / denom if denom > 0 else 0.0

            ident = {
                "candidate_id": row.get("candidate_id", f"candidate_{i+1}"),
                "hostname": row.get("hostname", np.nan),
                "node_id": row.get("node_id", np.nan),
            }
            srow = dict(ident)
            srow.update({
                "analog_support_n": int(len(local)),
                "analog_distance_min": float(np.nanmin(dist)) if len(dist) else np.nan,
                "analog_distance_median": float(np.nanmedian(dist)) if len(dist) else np.nan,
            })
            for q, val in zip(quantiles, q_radius):
                srow[f"analog_pl_rade_q{int(round(q * 100)):02d}"] = float(val)
            for q, val in zip(quantiles, q_mass):
                srow[f"analog_pl_bmasse_q{int(round(q * 100)):02d}"] = float(val)
            for q, val in zip(quantiles, q_dens):
                srow[f"analog_pl_dens_q{int(round(q * 100)):02d}"] = float(val)
            srow.update(class_probs)
            summary_rows.append(srow)

            for rank, (j, d, w) in enumerate(zip(idx, dist, weights), start=1):
                planet = self.catalog_.iloc[j]
                neighbor_rows.append({
                    **ident,
                    "analog_rank": rank,
                    "analog_distance": float(d),
                    "analog_weight": float(w),
                    "analog_pl_name": planet.get("pl_name", np.nan),
                    "analog_hostname": planet.get("hostname", np.nan),
                    "analog_pl_orbper": planet.get("pl_orbper", np.nan),
                    "analog_pl_orbsmax": planet.get("pl_orbsmax", np.nan),
                    "analog_pl_rade": planet.get("pl_rade", np.nan),
                    "analog_pl_bmasse": planet.get("pl_bmasse", np.nan),
                    "analog_discoverymethod": planet.get("discoverymethod", np.nan),
                })
        return AnalogResult(pd.DataFrame(summary_rows), pd.DataFrame(neighbor_rows))
