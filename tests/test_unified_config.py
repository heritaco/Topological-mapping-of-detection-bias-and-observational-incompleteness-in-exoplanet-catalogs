from pathlib import Path

from src.exoplanet_tda.core.config import load_pipeline_config


def test_unified_config_loads_default_and_resolves_paths():
    _, cfg = load_pipeline_config("configs/pipeline/default.yaml")
    for section in ("project", "paths", "run", "inputs", "stages"):
        assert section in cfg
    assert Path(cfg["paths"]["repo_root"]).is_absolute()
    assert "resolved_paths" in cfg
    assert "inputs.raw_catalog" in cfg["resolved_paths"]
