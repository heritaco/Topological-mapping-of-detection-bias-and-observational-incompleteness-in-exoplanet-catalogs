"""Thin adapter exports."""

from src.exoplanet_tda.pipeline.stages import LegacyOutputStage


def make_stage() -> LegacyOutputStage:
    return LegacyOutputStage("toi_ati_case_anatomy", module="src.toi_ati_case_anatomy.run_case_anatomy")
