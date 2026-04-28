from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def confidence_level(row: pd.Series) -> str:
    shadow = float(pd.to_numeric(pd.Series([row.get("shadow_score")]), errors="coerce").fillna(0.0).iloc[0])
    ir3 = float(pd.to_numeric(pd.Series([row.get("I_R3")]), errors="coerce").fillna(1.0).iloc[0])
    deficit = float(pd.to_numeric(pd.Series([row.get("delta_rel_neighbors_best")]), errors="coerce").fillna(0.0).iloc[0])
    if shadow >= 0.22 and ir3 <= 0.10 and deficit >= 0.30:
        return "higher"
    if shadow >= 0.15 and ir3 <= 0.20:
        return "medium"
    return "lower"


def final_interpretation(row: pd.Series) -> str:
    return (
        f"{row['node_id']} se interpreta como region fisico-orbital submuestreada bajo referencia local, "
        f"con deficit relativo {row.get('delta_rel_neighbors_best')} y lectura prudente como candidato a "
        "incompletitud observacional, sin equivaler a una confirmacion de objetos ausentes."
    )
