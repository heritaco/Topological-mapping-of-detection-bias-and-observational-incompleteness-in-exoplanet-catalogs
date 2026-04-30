"""Prediction routines for candidate planets."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple
import numpy as np
import pandas as pd

from .analogs import WeightedAnalogCharacterizer
from .features import add_engineered_features
from .labels import most_probable_class, radius_class, thermal_class, orbit_class
from .models import TrainedCharacterizationModels
from .physics import density_from_mass_radius, rv_semiamplitude_proxy, transit_probability
from .utils import ensure_dir


def _unlog10_frame(df: pd.DataFrame, prefix: str, target_name: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        out[f"{target_name}_{col}"] = np.power(10.0, df[col].to_numpy(dtype=float))
    return out


def predict_candidates(
    candidates: pd.DataFrame,
    catalog: pd.DataFrame,
    models: TrainedCharacterizationModels,
    quantiles: Sequence[float],
    analog_k: int = 75,
    analog_temperature: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cand = candidates.copy()
    if "candidate_id" not in cand.columns:
        if "pl_name" in cand.columns:
            cand["candidate_id"] = cand["pl_name"].astype(str)
        elif "hostname" in cand.columns:
            cand["candidate_id"] = cand["hostname"].astype(str) + "_candidate_" + (cand.index + 1).astype(str)
        else:
            cand["candidate_id"] = "candidate_" + (cand.index + 1).astype(str)
    cand = add_engineered_features(cand)
    X = cand.copy()
    for f in models.model_features:
        if f not in X.columns:
            X[f] = np.nan
    radius_log_q = models.radius_model.predict(X[models.model_features])
    mass_log_q = models.mass_model.predict(X[models.model_features])
    radius_q = _unlog10_frame(radius_log_q, "log_pl_rade", "pl_rade")
    mass_q = _unlog10_frame(mass_log_q, "log_pl_bmasse", "pl_bmasse")
    class_prob = models.class_model.predict_proba(X)

    # Density distribution by aligned quantile approximation. Conservative enough for summaries;
    # detailed work should sample joint mass-radius posterior.
    density_q = pd.DataFrame(index=cand.index)
    for suffix in [c.replace("pl_rade_", "") for c in radius_q.columns]:
        r_col = f"pl_rade_{suffix}"
        m_col = f"pl_bmasse_{suffix}"
        if r_col in radius_q.columns and m_col in mass_q.columns:
            density_q[f"pl_dens_{suffix}"] = density_from_mass_radius(mass_q[m_col], radius_q[r_col])

    median_radius_col = "pl_rade_q50" if "pl_rade_q50" in radius_q.columns else radius_q.columns[len(radius_q.columns)//2]
    median_mass_col = "pl_bmasse_q50" if "pl_bmasse_q50" in mass_q.columns else mass_q.columns[len(mass_q.columns)//2]
    median_radius = radius_q[median_radius_col]
    median_mass = mass_q[median_mass_col]

    obs = pd.DataFrame(index=cand.index)
    obs["transit_probability_q50"] = transit_probability(cand.get("st_rad", pd.Series(np.nan, index=cand.index)), cand.get("pl_orbsmax", pd.Series(np.nan, index=cand.index)), median_radius)
    obs["rv_proxy_q50"] = rv_semiamplitude_proxy(median_mass, cand.get("pl_orbper", pd.Series(np.nan, index=cand.index)), cand.get("st_mass", pd.Series(np.nan, index=cand.index)))
    obs["orbit_class"] = orbit_class(cand.get("pl_orbper", pd.Series(np.nan, index=cand.index)))
    obs["thermal_class"] = thermal_class(cand.get("pl_eqt", pd.Series(np.nan, index=cand.index)))

    analog = WeightedAnalogCharacterizer(features=models.model_features, k=analog_k, temperature=analog_temperature)
    analog.fit(catalog)
    analog_result = analog.characterize(cand, quantiles=quantiles)

    id_cols = [c for c in ["candidate_id", "hostname", "node_id", "pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt", "toi", "ati", "shadow_score", "candidate_score", "topology_score", "gap_score"] if c in cand.columns]
    out = pd.concat([cand[id_cols].reset_index(drop=True), radius_q.reset_index(drop=True), mass_q.reset_index(drop=True), density_q.reset_index(drop=True), class_prob.reset_index(drop=True), obs.reset_index(drop=True)], axis=1)

    prob_cols = [c for c in out.columns if c.startswith("prob_")]
    if prob_cols:
        out["predicted_radius_class"] = out[prob_cols].apply(lambda r: most_probable_class(r.to_dict()), axis=1)
        out["predicted_radius_class_probability"] = out[prob_cols].max(axis=1)
    else:
        out["predicted_radius_class"] = radius_class(out[median_radius_col])
        out["predicted_radius_class_probability"] = np.nan

    out = out.merge(analog_result.summaries, on=[c for c in ["candidate_id", "hostname", "node_id"] if c in out.columns and c in analog_result.summaries.columns], how="left")
    out["interpretation_warning"] = "probabilistic characterization; not a confirmed planet or absolute missing-planet count"
    return out, analog_result.neighbors


def write_prediction_outputs(predictions: pd.DataFrame, analog_neighbors: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    table_dir = ensure_dir(output_dir / "tables")
    pred_path = table_dir / "candidate_property_predictions.csv"
    neigh_path = table_dir / "candidate_analog_neighbors.csv"
    predictions.to_csv(pred_path, index=False)
    analog_neighbors.to_csv(neigh_path, index=False)
    return {"predictions": pred_path, "analog_neighbors": neigh_path}
