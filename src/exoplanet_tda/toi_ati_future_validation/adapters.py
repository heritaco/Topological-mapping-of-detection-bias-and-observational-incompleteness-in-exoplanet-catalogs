"""Thin adapter exports."""

from src.exoplanet_tda.pipeline.stages import LegacyOutputStage


def make_stage() -> LegacyOutputStage:
    return LegacyOutputStage("toi_ati_future_validation", module="src.toi_ati_future_validation.run_future_validation")
