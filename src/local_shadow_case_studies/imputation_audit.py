from __future__ import annotations

import pandas as pd


def variable_status(frame: pd.DataFrame, variable: str) -> pd.Series:
    imputed = frame.get(f"{variable}_was_imputed", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    physical = frame.get(f"{variable}_was_physically_derived", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    status = pd.Series("observed", index=frame.index, dtype="string")
    status.loc[physical] = "physically_derived"
    status.loc[imputed] = "imputed"
    return status


def add_variable_status_columns(frame: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for variable in variables:
        out[f"imputation_status_{variable}"] = variable_status(out, variable)
    return out


def anchor_imputation_score(row: pd.Series, variables: list[str]) -> int:
    score = 0
    for variable in variables:
        status = str(row.get(f"imputation_status_{variable}", "observed"))
        if status == "physically_derived":
            score += 1
        elif status == "imputed":
            score += 2
    return score


def summarize_r3_imputation(frame: pd.DataFrame, variables: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    valid = frame[frame["r3_valid"]].copy()
    if valid.empty:
        out["I_R3"] = None
    else:
        imputed_values = 0
        for variable in variables:
            imputed_values += int(valid.get(f"{variable}_was_imputed", pd.Series(False, index=valid.index)).fillna(False).astype(bool).sum())
        out["I_R3"] = float(imputed_values / (len(valid) * len(variables)))

    for variable in variables:
        status = variable_status(frame, variable)
        out[f"imputed_fraction_{variable}"] = float((status == "imputed").mean()) if len(status) else 0.0
        out[f"physically_derived_fraction_{variable}"] = float((status == "physically_derived").mean()) if len(status) else 0.0
        out[f"observed_fraction_{variable}"] = float((status == "observed").mean()) if len(status) else 0.0
    return out

