import time

from src.exoplanet_tda.pipeline.orchestrator import PipelineOrchestrator, create_context
from src.exoplanet_tda.pipeline.stages import PipelineStage, StageResult


class FakeStage(PipelineStage):
    name = "fake"

    def should_run(self, ctx):
        return True

    def run(self, ctx):
        start = time.perf_counter()
        path = ctx.table_dir(self.name) / "fake.csv"
        path.write_text("ok\n1\n", encoding="utf-8")
        output = ctx.log_artifact(self.name, path, "table", "Fake stage output")
        return StageResult(self.name, "success", [output], {"rows": 1}, elapsed_seconds=time.perf_counter() - start)


def test_fake_stage_runs_and_registers_artifact():
    ctx = create_context("configs/pipeline/default.yaml", run_id="pytest_fake", dry_run=False, overwrite=True)
    orchestrator = PipelineOrchestrator(stages={"fake": FakeStage()})
    results = orchestrator.run(ctx, stages="fake")
    assert results[0].status == "success"
    assert any(entry.stage == "fake" for entry in ctx.registry.artifacts)
