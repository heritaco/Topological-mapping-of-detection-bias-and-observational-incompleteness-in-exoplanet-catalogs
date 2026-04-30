from src.exoplanet_tda.pipeline.cli import main


def test_pipeline_dry_run_creates_config_and_manifest(tmp_path):
    run_id = "pytest_dry"
    output_template = (tmp_path / "runs" / "{run_id}").as_posix()
    exit_code = main(
        [
            "--config",
            "configs/pipeline/default.yaml",
            "--run-id",
            run_id,
            "--dry-run",
            "--stages",
            "data_audit",
        ]
    )
    assert exit_code == 0
    # CLI resolves repo default, so verify the conventional output location.
    from pathlib import Path

    run_dir = Path("outputs") / "runs" / run_id
    assert (run_dir / "config" / "config_effective.yaml").exists()
    assert (run_dir / "manifests" / "run_manifest.json").exists()
