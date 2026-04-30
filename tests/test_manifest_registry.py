from src.exoplanet_tda.core.io import read_json
from src.exoplanet_tda.core.manifest import ArtifactRegistry


def test_manifest_registry_round_trips_artifacts(tmp_path):
    run_dir = tmp_path / "outputs" / "runs" / "manifest"
    artifact = run_dir / "tables" / "x.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("a\n1\n", encoding="utf-8")
    registry = ArtifactRegistry(tmp_path, run_dir, "manifest")
    registry.add_artifact("stage_a", artifact, "table", "Fake table", {"rows": 1})
    path = registry.save()
    loaded = read_json(path)
    assert loaded["artifacts"][0]["metadata"]["rows"] == 1
    assert loaded["artifacts"][0]["exists"] is True
