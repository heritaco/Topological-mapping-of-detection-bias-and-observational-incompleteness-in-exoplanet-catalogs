"""Thin adapter exports."""

from src.exoplanet_tda.pipeline.stages import LegacyOutputStage


def make_stage() -> LegacyOutputStage:
    return LegacyOutputStage("local_shadow_case_studies", module="src.local_shadow_case_studies.run_local_shadow_cases")
