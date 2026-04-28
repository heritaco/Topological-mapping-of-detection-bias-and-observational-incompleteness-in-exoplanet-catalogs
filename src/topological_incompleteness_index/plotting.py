from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def apply_project_style() -> None:
    try:
        from src.visual_style import apply_visual_style  # type: ignore
        apply_visual_style()
    except Exception:
        pass

def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_top_regions(regions: pd.DataFrame, figures_dir: Path, top_n: int = 25) -> None:
    if regions.empty:
        return
    apply_project_style()
    df = regions.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.28 * len(df))))
    plt.barh(df["node_id"].astype(str), df["toi_score"])
    plt.xlabel("TOI score")
    plt.ylabel("Mapper node")
    plt.title("Topological Observational Incompleteness by region")
    savefig(figures_dir / "top_regions_toi_score.pdf")

def plot_top_anchors(anchors: pd.DataFrame, figures_dir: Path, top_n: int = 25) -> None:
    if anchors.empty:
        return
    apply_project_style()
    df = anchors.sort_values("ati_score", ascending=False).head(top_n).iloc[::-1]
    labels = df["anchor_pl_name"].astype(str) + " / " + df["node_id"].astype(str)
    plt.figure(figsize=(8, max(4, 0.28 * len(df))))
    plt.barh(labels, df["ati_score"])
    plt.xlabel("ATI score")
    plt.ylabel("Anchor planet / node")
    plt.title("Anchor Topological Incompleteness ranking")
    savefig(figures_dir / "top_anchor_ati_score.pdf")

def plot_toi_vs_deficit(anchors: pd.DataFrame, figures_dir: Path) -> None:
    if anchors.empty:
        return
    apply_project_style()
    plt.figure(figsize=(6, 4))
    plt.scatter(anchors["toi_score"], anchors["delta_rel_neighbors_best"])
    for _, row in anchors.head(12).iterrows():
        plt.annotate(str(row["anchor_pl_name"]), (row["toi_score"], row["delta_rel_neighbors_best"]), fontsize=7)
    plt.xlabel("Regional TOI score")
    plt.ylabel("Best relative local deficit")
    plt.title("Regional incompleteness vs anchor deficit")
    savefig(figures_dir / "toi_vs_anchor_deficit.pdf")
