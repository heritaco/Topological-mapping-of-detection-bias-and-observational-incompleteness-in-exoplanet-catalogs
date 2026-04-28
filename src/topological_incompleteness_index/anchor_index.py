from __future__ import annotations
import numpy as np
import pandas as pd
from .r3_geometry import choose_medoid, centroid, pairwise_distances_to_anchor

def select_anchor(node_planets: pd.DataFrame, z_cols: list[str], preferred_method: str) -> pd.Series | None:
    valid = node_planets.dropna(subset=z_cols).copy()
    if valid.empty:
        return None
    medoid = choose_medoid(valid, z_cols)
    if medoid is not None:
        m = medoid[z_cols].to_numpy(float)
        x = valid[z_cols].to_numpy(float)
        valid["distance_to_medoid"] = np.linalg.norm(x - m, axis=1)
    else:
        valid["distance_to_medoid"] = 0.0
    valid["preferred_method"] = (valid.get("discoverymethod", "") == preferred_method).astype(int)
    # If no explicit status columns exist, assume zero imputation penalty. The manifest will say this is a fallback.
    status_cols = [c for c in valid.columns if "status" in c.lower()]
    def score_status(row):
        if not status_cols:
            return 0.0
        vals = [str(row[c]).lower() for c in status_cols if c in row.index]
        if not vals:
            return 0.0
        return sum(1.0 if "imputed" in v else 0.5 if "derived" in v else 0.0 for v in vals) / len(vals)
    valid["anchor_r3_imputation_score"] = valid.apply(score_status, axis=1)
    valid = valid.sort_values(
        ["preferred_method", "anchor_r3_imputation_score", "distance_to_medoid"],
        ascending=[False, True, True]
    )
    return valid.iloc[0]

def compute_neighbor_deficits(
    anchor: pd.Series,
    node_planets: pd.DataFrame,
    n1_planets: pd.DataFrame,
    n2_planets: pd.DataFrame,
    z_cols: list[str],
    cfg: dict,
) -> pd.DataFrame:
    eps = float(cfg.get("eps", 1e-9))
    anchor_name = str(anchor.get("pl_name", "anchor"))
    universe = pd.concat([node_planets, n1_planets, n2_planets], ignore_index=True).dropna(subset=z_cols)

    node_valid = node_planets.dropna(subset=z_cols)
    d_node = pairwise_distances_to_anchor(node_valid, anchor, z_cols)
    if len(d_node):
        # Exclude the anchor itself if present.
        anchor_mask = node_valid.loc[d_node.index, "pl_name"].astype(str) != anchor_name
        d_node = d_node[anchor_mask]
    r_median = float(np.median(d_node)) if len(d_node) else 0.0
    r_q75 = float(np.quantile(d_node, 0.75)) if len(d_node) else r_median

    d_universe = pairwise_distances_to_anchor(universe, anchor, z_cols)
    if len(d_universe):
        anchor_mask = universe.loc[d_universe.index, "pl_name"].astype(str) != anchor_name
        d_universe = d_universe[anchor_mask]
    k = min(10, max(3, int(np.sqrt(max(len(universe), 1)))))
    r_knn = float(np.sort(d_universe.to_numpy())[min(k - 1, len(d_universe) - 1)]) if len(d_universe) else 0.0

    radii = {"r_node_median": r_median, "r_node_q75": r_q75, "r_kNN": r_knn}
    reference = n1_planets.dropna(subset=z_cols)
    if len(reference) < 3:
        reference = n2_planets.dropna(subset=z_cols)

    rows = []
    for name, r in radii.items():
        if r <= 0 or not np.isfinite(r):
            n_obs = 0
            n_exp = np.nan
        else:
            d_all = pairwise_distances_to_anchor(universe, anchor, z_cols)
            if len(d_all):
                mask_not_anchor = universe.loc[d_all.index, "pl_name"].astype(str) != anchor_name
                n_obs = int(((d_all <= r) & mask_not_anchor).sum())
            else:
                n_obs = 0
            d_ref = pairwise_distances_to_anchor(reference, anchor, z_cols)
            n_exp = float((d_ref <= r).sum()) if len(d_ref) else np.nan
        delta = n_exp - n_obs if np.isfinite(n_exp) else np.nan
        delta_rel = delta / (n_exp + eps) if np.isfinite(n_exp) else np.nan
        if not np.isfinite(delta_rel):
            cls = "not_available"
        elif delta_rel <= 0.1:
            cls = "no_deficit"
        elif delta_rel <= 0.3:
            cls = "weak_deficit"
        elif delta_rel <= 0.6:
            cls = "moderate_deficit"
        else:
            cls = "strong_deficit"
        rows.append({
            "radius_type": name,
            "radius_value": r,
            "N_obs": n_obs,
            "N_exp_neighbors": n_exp,
            "delta_N_neighbors": delta,
            "delta_rel_neighbors": delta_rel,
            "deficit_class_neighbors": cls,
        })
    return pd.DataFrame(rows)

def best_positive_deficit(deficits: pd.DataFrame) -> float:
    if deficits.empty or "delta_rel_neighbors" not in deficits:
        return 0.0
    vals = pd.to_numeric(deficits["delta_rel_neighbors"], errors="coerce").dropna()
    vals = vals[vals > 0]
    return float(vals.max()) if len(vals) else 0.0

def anchor_quality(anchor: pd.Series, node_planets: pd.DataFrame, z_cols: list[str]) -> float:
    valid = node_planets.dropna(subset=z_cols)
    if valid.empty:
        return 0.0
    c = centroid(valid, z_cols)
    x = anchor[z_cols].to_numpy(float)
    spread = np.linalg.norm(valid[z_cols].to_numpy(float) - c, axis=1).mean()
    if not np.isfinite(spread) or spread <= 0:
        spread = 1.0
    return float(np.exp(-(np.linalg.norm(x - c) ** 2) / (2 * spread**2)))
