from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class R3Columns:
    mass: str
    period: str
    semimajor: str

    @property
    def raw(self) -> list[str]:
        return [self.mass, self.period, self.semimajor]

    @property
    def log(self) -> list[str]:
        return ["r3_log_mass", "r3_log_period", "r3_log_semimajor"]

    @property
    def z(self) -> list[str]:
        return ["r3_z_mass", "r3_z_period", "r3_z_semimajor"]


def safe_log10_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & (numeric > 0)
    output = pd.Series(np.nan, index=series.index, dtype=float)
    output.loc[valid] = np.log10(numeric.loc[valid].astype(float))
    return output, ~valid


def add_value_source_columns(frame: pd.DataFrame, columns: R3Columns) -> pd.DataFrame:
    out = frame.copy()
    for variable in columns.raw:
        imputed = out.get(f"{variable}_was_imputed", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        physical = out.get(f"{variable}_was_physically_derived", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        status = pd.Series("observed", index=out.index, dtype="string")
        status.loc[physical] = "physically_derived"
        status.loc[imputed] = "imputed"
        out[f"imputation_status_{variable}"] = status
    return out


def build_r3_frame(frame: pd.DataFrame, columns: R3Columns, warnings: list[str], skipped_items: list[dict[str, object]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = add_value_source_columns(frame, columns)
    invalid_any = pd.Series(False, index=out.index)
    invalid_counts: dict[str, int] = {}
    for raw_name, log_name in zip(columns.raw, columns.log):
        if raw_name not in out.columns:
            raise KeyError(f"Falta variable R3 requerida: {raw_name}")
        logged, invalid = safe_log10_series(out[raw_name])
        out[log_name] = logged
        invalid_any = invalid_any | invalid
        invalid_count = int(invalid.sum())
        invalid_counts[raw_name] = invalid_count
        if invalid_count > 0:
            warnings.append(f"WARNING: {raw_name} tiene {invalid_count} valores no positivos/faltantes para log10.")
            invalid_rows = out.loc[invalid, [column for column in ["node_id", "pl_name", raw_name] if column in out.columns]].copy()
            for record in invalid_rows.head(500).to_dict(orient="records"):
                skipped_items.append({"reason": "invalid_log_values", **record, "variable": raw_name})
    out["r3_valid"] = out[columns.log].notna().all(axis=1)

    valid = out[out["r3_valid"]].copy()
    stats: dict[str, dict[str, float]] = {}
    for raw_name, log_name, z_name in zip(columns.raw, columns.log, columns.z):
        mean = float(pd.to_numeric(valid[log_name], errors="coerce").mean()) if not valid.empty else 0.0
        std = float(pd.to_numeric(valid[log_name], errors="coerce").std(ddof=0)) if not valid.empty else 1.0
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        out[z_name] = (pd.to_numeric(out[log_name], errors="coerce") - mean) / std
        stats[raw_name] = {"mean": mean, "std": std}
    return out, {"r3_stats": stats, "invalid_counts": invalid_counts, "n_r3_valid": int(out["r3_valid"].sum())}


def z_matrix(frame: pd.DataFrame, z_columns: list[str]) -> np.ndarray:
    return frame[z_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def centroid(frame: pd.DataFrame, columns: list[str]) -> np.ndarray | None:
    valid = frame.dropna(subset=columns)
    if valid.empty:
        return None
    return np.nanmean(valid[columns].to_numpy(dtype=float), axis=0)


def mean_distance_to_center(frame: pd.DataFrame, center: np.ndarray | None, z_columns: list[str]) -> float | None:
    if center is None:
        return None
    valid = frame.dropna(subset=z_columns)
    if valid.empty:
        return None
    distances = np.linalg.norm(z_matrix(valid, z_columns) - center, axis=1)
    return float(np.mean(distances)) if len(distances) else None


def medoid_row(frame: pd.DataFrame, z_columns: list[str]) -> pd.Series | None:
    valid = frame.dropna(subset=z_columns).copy()
    if valid.empty:
        return None
    matrix = z_matrix(valid, z_columns)
    dsum = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2).sum(axis=1)
    return valid.iloc[int(np.argmin(dsum))]


def node_r3_imputation_summary(node_frame: pd.DataFrame, columns: R3Columns) -> dict[str, float | None]:
    valid = node_frame[node_frame["r3_valid"]].copy()
    summary: dict[str, float | None] = {
        "n_r3_valid": int(valid["r3_valid"].sum()) if "r3_valid" in valid.columns else int(len(valid)),
        "r3_valid_fraction": float(node_frame["r3_valid"].mean()) if len(node_frame) else 0.0,
    }
    if valid.empty:
        summary["I_R3"] = None
    else:
        imputed_total = 0
        for variable in columns.raw:
            imputed_total += int(valid.get(f"{variable}_was_imputed", pd.Series(False, index=valid.index)).fillna(False).astype(bool).sum())
        summary["I_R3"] = float(imputed_total / (len(valid) * len(columns.raw)))
    for variable in columns.raw:
        status = node_frame.get(f"imputation_status_{variable}", pd.Series(index=node_frame.index, dtype="string"))
        summary[f"imputed_fraction_{variable}"] = float((status == "imputed").mean()) if len(status) else 0.0
        summary[f"physically_derived_fraction_{variable}"] = float((status == "physically_derived").mean()) if len(status) else 0.0
        summary[f"observed_fraction_{variable}"] = float((status == "observed").mean()) if len(status) else 0.0
    return summary


def anchor_r3_imputation_score(row: pd.Series, columns: R3Columns) -> float:
    score = 0.0
    for variable in columns.raw:
        status = str(row.get(f"imputation_status_{variable}", "observed"))
        if status == "physically_derived":
            score += 0.5
        elif status == "imputed":
            score += 1.0
    return float(score / len(columns.raw))


def anchor_imputed_fraction(row: pd.Series, columns: R3Columns) -> float:
    total = 0
    for variable in columns.raw:
        if str(row.get(f"imputation_status_{variable}", "observed")) == "imputed":
            total += 1
    return float(total / len(columns.raw))


def centroid_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))
