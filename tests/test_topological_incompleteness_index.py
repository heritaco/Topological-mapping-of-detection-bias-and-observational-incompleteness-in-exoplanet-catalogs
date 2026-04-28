import numpy as np
import pandas as pd
from src.topological_incompleteness_index.r3_geometry import R3Columns, add_r3_coordinates
from src.topological_incompleteness_index.regional_index import compute_regional_toi
from src.topological_incompleteness_index.anchor_index import best_positive_deficit

def test_r3_log_transform_marks_invalid_values():
    df = pd.DataFrame({
        "pl_name": ["a", "b"],
        "pl_bmasse": [10.0, -1.0],
        "pl_orbper": [100.0, 10.0],
        "pl_orbsmax": [1.0, 0.0],
    })
    out, _ = add_r3_coordinates(df, R3Columns("pl_bmasse", "pl_orbper", "pl_orbsmax"))
    assert out.loc[0, "r3_valid"]
    assert not out.loc[1, "r3_valid"]
    assert np.isclose(out.loc[0, "r3_log_mass"], 1.0)

def test_toi_penalizes_imputation():
    nodes = pd.DataFrame({
        "node_id": ["a", "b"],
        "shadow_score": [0.5, 0.5],
        "I_R3": [0.0, 1.0],
        "physical_distance_v_to_N1": [0.1, 0.1],
        "n_members": [10, 10],
        "degree": [2, 2],
    })
    out = compute_regional_toi(nodes, {"top_quantile": 0.5})
    scores = dict(zip(out["node_id"], out["toi_score"]))
    assert scores["a"] > scores["b"]
    assert scores["b"] == 0

def test_best_positive_deficit():
    df = pd.DataFrame({"delta_rel_neighbors": [-0.1, 0.2, 0.05]})
    assert best_positive_deficit(df) == 0.2
