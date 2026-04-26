from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def validate_imputable_matrix(matrix: pd.DataFrame) -> None:
    if matrix.empty:
        raise ValueError("La matriz de imputacion esta vacia.")
    all_missing = [column for column in matrix.columns if matrix[column].isna().all()]
    if all_missing:
        raise ValueError(f"No se pueden imputar columnas completamente nulas: {all_missing}")

def robust_scale(matrix: pd.DataFrame) -> tuple[pd.DataFrame, RobustScaler]:
    validate_imputable_matrix(matrix)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(matrix)
    scaled_df = pd.DataFrame(scaled, index=matrix.index, columns=matrix.columns)
    return scaled_df.replace([np.inf, -np.inf], np.nan), scaler


def invert_robust_scale(matrix: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    restored = scaler.inverse_transform(matrix)
    return pd.DataFrame(restored, index=matrix.index, columns=matrix.columns)
