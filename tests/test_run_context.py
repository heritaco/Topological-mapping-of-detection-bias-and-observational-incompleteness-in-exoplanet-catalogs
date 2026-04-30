from pathlib import Path

from src.exoplanet_tda.core.logging_utils import setup_run_logger
from src.exoplanet_tda.core.manifest import ArtifactRegistry
from src.exoplanet_tda.core.run_context import RunContext


def make_ctx(tmp_path: Path) -> RunContext:
    run_dir = tmp_path / "outputs" / "runs" / "ctx"
    registry = ArtifactRegistry(tmp_path, run_dir, "ctx")
    logger = setup_run_logger("test_run_context", run_dir / "logs" / "pipeline.log")
    return RunContext(tmp_path, "ctx", run_dir, {"stages": {}}, logger, registry)


def test_run_context_writes_files_and_logs_artifacts(tmp_path):
    ctx = make_ctx(tmp_path)
    json_path = ctx.write_json("manifests/example.json", {"ok": True})
    yaml_path = ctx.write_yaml("config/example.yaml", {"ok": True})
    class TinyCsv:
        def to_csv(self, path, index=False):
            Path(path).write_text("a\n1\n", encoding="utf-8")

    csv_path = ctx.write_csv("tables/data_audit/example.csv", TinyCsv())
    assert json_path.exists()
    assert yaml_path.exists()
    assert csv_path.exists()
    assert ctx.table_dir("mapper").exists()
    assert ctx.figure_dir("mapper").exists()
    assert ctx.report_dir("mapper").exists()
    artifact = ctx.log_artifact("test", csv_path, "table", "Example table")
    assert artifact["exists"] is True
    assert ctx.registry.manifest_path.exists()
