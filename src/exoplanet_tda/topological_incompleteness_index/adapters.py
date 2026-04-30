"""Thin adapter exports."""

from src.exoplanet_tda.pipeline.stages import LegacyOutputStage


def make_stage() -> LegacyOutputStage:
    return LegacyOutputStage("topological_incompleteness_index", module="src.topological_incompleteness_index.run_topological_incompleteness")
