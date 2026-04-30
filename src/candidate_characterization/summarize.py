"""Generate a compact Markdown report from characterization outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import pandas as pd

from .utils import ensure_dir


def _frame_to_markdown_or_csv(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_csv(index=False) + "```"


def build_markdown_summary(
    predictions: pd.DataFrame,
    report_dir: Path,
    validation_metrics: Optional[pd.DataFrame] = None,
    notes: Optional[Sequence[str]] = None,
) -> Path:
    ensure_dir(report_dir)
    lines = []
    lines.append("# Candidate Characterization Report")
    lines.append("")
    lines.append("This report summarizes probabilistic characterizations of topologically prioritized candidate missing planets. The outputs are prioritization aids, not detections.")
    if notes:
        lines.append("")
        lines.append("## Run notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("The module combines orbital/stellar candidate features, weighted analogs in the observed catalog, quantile regression for mass and radius, radius-class probabilities, and physical post-processing for density and detectability proxies.")
    lines.append("")
    lines.append("## Top candidates")
    lines.append("")
    if predictions.empty:
        lines.append("No predictions were generated.")
    else:
        sort_cols = [c for c in ["toi", "ati", "candidate_score", "predicted_radius_class_probability"] if c in predictions.columns]
        top = predictions.copy()
        if sort_cols:
            top = top.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        keep = [c for c in [
            "candidate_id", "hostname", "node_id", "pl_orbper", "pl_orbsmax", "toi", "ati", "shadow_score",
            "predicted_radius_class", "predicted_radius_class_probability", "pl_rade_q50", "pl_bmasse_q50",
            "pl_dens_q50", "transit_probability_q50", "rv_proxy_q50", "analog_support_n"
        ] if c in top.columns]
        lines.append(_frame_to_markdown_or_csv(top[keep].head(20)))
    if validation_metrics is not None and not validation_metrics.empty:
        lines.append("")
        lines.append("## Validation")
        lines.append("")
        lines.append(_frame_to_markdown_or_csv(validation_metrics))
    lines.append("")
    lines.append("## Interpretation warning")
    lines.append("")
    lines.append("Predicted properties are conditional estimates under the observed catalog, feature engineering choices, imputation history, and discovery biases. They should be used to prioritize follow-up and stress-test candidates, not to claim confirmed planets.")
    path = report_dir / "candidate_characterization_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
