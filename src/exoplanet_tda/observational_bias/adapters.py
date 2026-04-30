"""Thin adapter exports."""

from src.exoplanet_tda.pipeline.stages import LegacyOutputStage


def make_stage() -> LegacyOutputStage:
    return LegacyOutputStage("observational_bias", module="src.observational_bias_audit.run_bias_audit")
