from __future__ import annotations

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer


def impute_with_median(matrix: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(matrix.to_numpy(copy=True))
    return pd.DataFrame(imputed, index=matrix.index, columns=matrix.columns)


def impute_with_iterative(
    matrix: pd.DataFrame,
    random_state: int = 42,
    max_iter: int = 20,
) -> pd.DataFrame:
    imputer = IterativeImputer(
        random_state=random_state,
        max_iter=max_iter,
        initial_strategy="median",
        sample_posterior=False,
        keep_empty_features=True,
    )
    imputed = imputer.fit_transform(matrix.to_numpy(copy=True))
    return pd.DataFrame(imputed, index=matrix.index, columns=matrix.columns)
