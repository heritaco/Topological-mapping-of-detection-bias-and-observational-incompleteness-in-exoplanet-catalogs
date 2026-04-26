from __future__ import annotations

import pandas as pd
from sklearn.impute import KNNImputer


def impute_with_knn(
    matrix: pd.DataFrame,
    n_neighbors: int = 7,
    weights: str = "distance",
) -> pd.DataFrame:
    effective_neighbors = max(1, min(n_neighbors, len(matrix)))
    imputer = KNNImputer(
        n_neighbors=effective_neighbors,
        weights=weights,
        keep_empty_features=True,
    )
    imputed = imputer.fit_transform(matrix.to_numpy(copy=True))
    return pd.DataFrame(imputed, index=matrix.index, columns=matrix.columns)
