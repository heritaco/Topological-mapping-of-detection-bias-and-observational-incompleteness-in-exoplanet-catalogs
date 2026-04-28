from __future__ import annotations
from dataclasses import dataclass
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

def add_r3_coordinates(catalog: pd.DataFrame, cols: R3Columns) -> tuple[pd.DataFrame, dict]:
    out = catalog.copy()
    valid_all = np.ones(len(out), dtype=bool)
    warnings = []
    for raw, logc in zip(cols.raw, cols.log):
        if raw not in out.columns:
            raise KeyError(f"Missing required R3 variable: {raw}")
        x = pd.to_numeric(out[raw], errors="coerce")
        valid = x.notna() & (x > 0)
        out[logc] = np.where(valid, np.log10(x), np.nan)
        valid_all &= valid.to_numpy()
        if (~valid).sum():
            warnings.append(f"{raw}: {(~valid).sum()} invalid/non-positive values for log10.")
    out["r3_valid"] = valid_all

    stats = {}
    for logc, zc in zip(cols.log, cols.z):
        mu = out.loc[out["r3_valid"], logc].mean()
        sd = out.loc[out["r3_valid"], logc].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        out[zc] = (out[logc] - mu) / sd
        stats[logc] = {"mean": float(mu), "std": float(sd)}
    return out, {"r3_stats": stats, "warnings": warnings}

def centroid(df: pd.DataFrame, z_cols: list[str]) -> np.ndarray:
    valid = df.dropna(subset=z_cols)
    if valid.empty:
        return np.full(len(z_cols), np.nan)
    return valid[z_cols].to_numpy(float).mean(axis=0)

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return float("nan")
    return float(np.linalg.norm(a - b))

def pairwise_distances_to_anchor(df: pd.DataFrame, anchor_row: pd.Series, z_cols: list[str]) -> pd.Series:
    valid = df.dropna(subset=z_cols)
    if valid.empty:
        return pd.Series(dtype=float)
    a = anchor_row[z_cols].to_numpy(float)
    x = valid[z_cols].to_numpy(float)
    return pd.Series(np.linalg.norm(x - a, axis=1), index=valid.index)

def choose_medoid(df: pd.DataFrame, z_cols: list[str]) -> pd.Series | None:
    valid = df.dropna(subset=z_cols)
    if valid.empty:
        return None
    x = valid[z_cols].to_numpy(float)
    dsum = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2).sum(axis=1)
    return valid.iloc[int(np.argmin(dsum))]
