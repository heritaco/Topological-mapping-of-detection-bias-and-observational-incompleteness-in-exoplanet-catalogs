"""Shared constants for the unified pipeline."""

from __future__ import annotations

STAGE_NAMES: tuple[str, ...] = (
    "data_audit",
    "imputation",
    "mapper",
    "observational_bias",
    "observational_shadow",
    "local_shadow_case_studies",
    "topological_incompleteness_index",
    "toi_ati_case_anatomy",
    "toi_ati_future_validation",
    "system_missing_planets",
    "candidate_characterization",
    "reporting",
)

ARTIFACT_KINDS: tuple[str, ...] = (
    "table",
    "figure",
    "model",
    "report",
    "config",
    "log",
    "manifest",
    "other",
)

STAGE_DIR_NAMES: dict[str, str] = {
    "data_audit": "data_audit",
    "imputation": "imputation",
    "mapper": "mapper",
    "observational_bias": "observational_bias",
    "observational_shadow": "observational_shadow",
    "local_shadow_case_studies": "local_shadow_case_studies",
    "topological_incompleteness_index": "topological_incompleteness_index",
    "toi_ati_case_anatomy": "toi_ati_case_anatomy",
    "toi_ati_future_validation": "toi_ati_future_validation",
    "system_missing_planets": "system_missing_planets",
    "candidate_characterization": "candidate_characterization",
}
