"""Stage registry for the unified pipeline."""

from __future__ import annotations

from .stages import (
    CandidateCharacterizationStage,
    DataAuditStage,
    FeatureAuditStage,
    ImputationStage,
    LegacyOutputStage,
    MapperStage,
    PipelineStage,
    ReportingStage,
    SystemMissingPlanetsStage,
)


def build_stage_registry() -> dict[str, PipelineStage]:
    return {
        "data_audit": DataAuditStage(),
        "feature_audit": FeatureAuditStage(),
        "imputation": ImputationStage(),
        "mapper": MapperStage(),
        "observational_bias": LegacyOutputStage("observational_bias", module="src.observational_bias_audit.run_bias_audit"),
        "observational_shadow": LegacyOutputStage("observational_shadow", module="src.observational_shadow.run_observational_shadow"),
        "local_shadow_case_studies": LegacyOutputStage("local_shadow_case_studies", module="src.local_shadow_case_studies.run_local_shadow_cases"),
        "topological_incompleteness_index": LegacyOutputStage(
            "topological_incompleteness_index",
            module="src.topological_incompleteness_index.run_topological_incompleteness",
        ),
        "toi_ati_case_anatomy": LegacyOutputStage("toi_ati_case_anatomy", module="src.toi_ati_case_anatomy.run_case_anatomy"),
        "toi_ati_future_validation": LegacyOutputStage("toi_ati_future_validation", module="src.toi_ati_future_validation.run_future_validation"),
        "system_missing_planets": SystemMissingPlanetsStage(),
        "candidate_characterization": CandidateCharacterizationStage(),
        "reporting": ReportingStage(),
    }
