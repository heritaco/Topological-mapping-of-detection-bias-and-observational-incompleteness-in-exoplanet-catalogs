from __future__ import annotations

from typing import Any

PROJECT_COLOR_CYCLE = [
    "#2563eb",
    "#0f766e",
    "#f59e0b",
    "#dc2626",
    "#475569",
    "#0891b2",
    "#7c3aed",
]

SOURCE_PALETTE = {
    "observed": "#2563eb",
    "physically_derived": "#0f766e",
    "imputed": "#94a3b8",
}

LENS_MARKERS = {
    "pca2": "o",
    "density": "s",
    "domain": "^",
}


def configure_matplotlib(matplotlib: Any) -> None:
    from matplotlib import style as mpl_style

    matplotlib.use("Agg")
    mpl_style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "axes.edgecolor": "#d7dee7",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.labelcolor": "#1f2937",
            "axes.titlecolor": "#0f172a",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "grid.color": "#dbe4ee",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "axes.prop_cycle": matplotlib.cycler(color=PROJECT_COLOR_CYCLE),
            "image.cmap": "cividis",
        }
    )


def apply_axis_style(ax: Any, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if title:
        ax.set_title(title, loc="left", pad=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#dbe4ee", linewidth=0.8)
    ax.grid(False, axis="x")
    for side in ("left", "bottom"):
        if side in ax.spines:
            ax.spines[side].set_color("#d7dee7")
            ax.spines[side].set_linewidth(0.8)


def style_colorbar(colorbar: Any, label: str) -> None:
    colorbar.set_label(label, color="#334155")
    colorbar.outline.set_edgecolor("#d7dee7")
    colorbar.outline.set_linewidth(0.8)
