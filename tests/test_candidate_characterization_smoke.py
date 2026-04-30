from pathlib import Path
import numpy as np
import pandas as pd

from src.candidate_characterization.config import CharacterizationConfig
from src.candidate_characterization.io import derive_catalog_columns, enrich_candidates_from_catalog
from src.candidate_characterization.models import train_models
from src.candidate_characterization.predict import predict_candidates


def make_catalog(n=160):
    rng = np.random.default_rng(42)
    hostnames = [f"H{i//3}" for i in range(n)]
    period = 10 ** rng.uniform(0, 3, n)
    st_mass = rng.uniform(0.7, 1.3, n)
    a = (st_mass * (period / 365.25) ** 2) ** (1 / 3)
    radius = 10 ** (0.15 + 0.15 * np.log10(period) + rng.normal(0, 0.18, n))
    mass = np.maximum(0.2, radius ** 2.2 * rng.lognormal(0, 0.35, n))
    return pd.DataFrame({
        "pl_name": [f"P{i}" for i in range(n)],
        "hostname": hostnames,
        "pl_orbper": period,
        "pl_orbsmax": a,
        "pl_rade": radius,
        "pl_bmasse": mass,
        "st_mass": st_mass,
        "st_rad": rng.uniform(0.7, 1.4, n),
        "st_teff": rng.uniform(4500, 6500, n),
        "st_lum": rng.uniform(0.3, 2.0, n),
        "disc_year": rng.integers(2000, 2026, n),
    })


def test_smoke_train_predict_cpu():
    cfg = CharacterizationConfig()
    cfg.model.prefer_gpu = False
    catalog = derive_catalog_columns(make_catalog(), cfg)
    models = train_models(catalog, quantiles=[0.05, 0.5, 0.95], prefer_gpu=False, random_state=42)
    cand = pd.DataFrame({
        "candidate_id": ["C1"],
        "hostname": ["H1"],
        "pl_orbper": [50.0],
        "node_id": ["cube12_cluster0"],
        "toi": [0.077],
        "shadow_score": [0.2],
    })
    cand = enrich_candidates_from_catalog(cand, catalog, cfg)
    pred, neighbors = predict_candidates(cand, catalog, models, quantiles=[0.05, 0.5, 0.95], analog_k=10)
    assert len(pred) == 1
    assert "pl_rade_q50" in pred.columns
    assert "pl_bmasse_q50" in pred.columns
    assert len(neighbors) > 0
