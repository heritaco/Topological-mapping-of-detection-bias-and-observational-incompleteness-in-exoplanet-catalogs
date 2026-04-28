from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ORDERED_RADII = ["r_kNN", "r_node_median", "r_node_q75"]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_top_regions(top_regions: pd.DataFrame, path: Path, *, value_col: str = "TOI") -> None:
    if top_regions.empty or value_col not in top_regions.columns:
        return
    df = top_regions.sort_values(value_col, ascending=True)
    labels = df.get("node_id", pd.Series(range(len(df)))).astype(str)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(df))))
    ax.barh(labels, df[value_col])
    ax.set_xlabel(value_col)
    ax.set_ylabel("Node ID")
    ax.set_title(f"Top regiones por {value_col}")
    _save(fig, path)


def plot_top_anchors(top_anchors: pd.DataFrame, path: Path, *, value_col: str = "ATI") -> None:
    if top_anchors.empty or value_col not in top_anchors.columns:
        return
    df = top_anchors.sort_values(value_col, ascending=True)
    labels = (
        df.get("anchor_pl_name", pd.Series(range(len(df)))).astype(str)
        + " / "
        + df.get("node_id", pd.Series([""] * len(df))).astype(str)
    )
    fig, ax = plt.subplots(figsize=(9, max(4, 0.30 * len(df))))
    ax.barh(labels, df[value_col])
    ax.set_xlabel(value_col)
    ax.set_ylabel("Anchor / node")
    ax.set_title(f"Top planetas ancla por {value_col}")
    _save(fig, path)


def plot_toi_decomposition(top_regions: pd.DataFrame, path: Path) -> None:
    cols = ["shadow_score", "one_minus_I_R3", "C_phys", "S_net"]
    available = [c for c in cols if c in top_regions.columns]
    if top_regions.empty or not available or "node_id" not in top_regions.columns:
        return
    df = top_regions.set_index("node_id")[available]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(df))))
    bottom = None
    for col in available:
        values = df[col]
        ax.barh(df.index.astype(str), values, left=bottom, label=col)
        bottom = values if bottom is None else bottom + values
    ax.set_xlabel("factor value; stacked only for visual audit")
    ax.set_title("Descomposición visual de factores TOI")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_ati_decomposition(top_anchors: pd.DataFrame, path: Path) -> None:
    cols = ["TOI", "positive_delta_rel_neighbors_best", "one_minus_anchor_I_R3", "anchor_representativeness"]
    available = [c for c in cols if c in top_anchors.columns]
    if top_anchors.empty or not available:
        return
    label = top_anchors.get("anchor_pl_name", pd.Series(range(len(top_anchors)))).astype(str) + " / " + top_anchors.get(
        "node_id", pd.Series([""] * len(top_anchors))
    ).astype(str)
    df = top_anchors.copy()
    df["label"] = label
    df = df.set_index("label")[available]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(df))))
    bottom = None
    for col in available:
        values = df[col]
        ax.barh(df.index.astype(str), values, left=bottom, label=col)
        bottom = values if bottom is None else bottom + values
    ax.set_xlabel("factor value; stacked only for visual audit")
    ax.set_title("Descomposición visual de factores ATI")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_deficit_relative_by_radius(deficit_detail: pd.DataFrame, path: Path) -> dict[str, object]:
    return _plot_deficit_distribution(
        deficit_detail,
        path,
        y_col="delta_rel_neighbors_recomputed",
        title="Déficit relativo por radio",
        ylabel=r"$\Delta_{\mathrm{rel}}$",
        add_threshold=True,
    )


def plot_deficit_absolute_by_radius(deficit_detail: pd.DataFrame, path: Path) -> dict[str, object]:
    return _plot_deficit_distribution(
        deficit_detail,
        path,
        y_col="delta_N_neighbors_recomputed",
        title="Déficit absoluto por radio",
        ylabel=r"$\Delta N = N_{\mathrm{exp}} - N_{\mathrm{obs}}$",
        add_threshold=False,
    )


def plot_deficit_by_radius(deficit_summary: pd.DataFrame, path: Path) -> dict[str, object]:
    """Legacy-compatible plot entrypoint used to audit the former Figure 5."""
    if deficit_summary.empty:
        return {"previous_y_column": None, "previous_y_max": None}
    radius_cols = [c for c in ORDERED_RADII if c in deficit_summary.columns]
    if not radius_cols:
        radius_cols = [
            c
            for c in deficit_summary.columns
            if c
            not in {"node_id", "anchor_pl_name", "delta_rel_neighbors_mean", "delta_rel_neighbors_median", "delta_rel_neighbors_best", "best_radius", "deficit_stability_label"}
            and pd.api.types.is_numeric_dtype(deficit_summary[c])
        ]
    if not radius_cols:
        return {"previous_y_column": None, "previous_y_max": None}
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [pd.to_numeric(deficit_summary[c], errors="coerce").dropna() for c in radius_cols]
    ax.boxplot(series, tick_labels=radius_cols)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Delta_rel_neighbors")
    ax.set_title("Déficit relativo por radio")
    _save(fig, path)
    ymax = max((float(values.max()) for values in series if not values.empty), default=0.0)
    return {"previous_y_column": "summary_radius_columns", "previous_y_max": ymax}


def _plot_deficit_distribution(
    deficit_detail: pd.DataFrame,
    path: Path,
    *,
    y_col: str,
    title: str,
    ylabel: str,
    add_threshold: bool,
) -> dict[str, object]:
    if deficit_detail.empty or y_col not in deficit_detail.columns or "radius_type" not in deficit_detail.columns:
        return {"y_col": y_col, "y_max": None}
    df = deficit_detail.copy()
    df = df[df["radius_type"].isin(ORDERED_RADII)]
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = [df.loc[df["radius_type"] == radius, y_col].dropna() for radius in ORDERED_RADII]
    grouped_nonempty = [values for values in grouped if not values.empty]
    if not grouped_nonempty:
        return {"y_col": y_col, "y_max": None}
    ax.boxplot(grouped, tick_labels=ORDERED_RADII)
    for idx, radius in enumerate(ORDERED_RADII, start=1):
        values = df.loc[df["radius_type"] == radius, y_col].dropna()
        if values.empty:
            continue
        ax.scatter([idx] * len(values), values, s=16, alpha=0.7, zorder=3)
    ax.axhline(0, linewidth=1, color="black")
    if add_threshold:
        ax.axhline(0.10, linewidth=1, color="gray", linestyle="--")
    ax.set_xlabel("Radio local")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, path)
    return {"y_col": y_col, "y_max": float(df[y_col].max(skipna=True))}


def plot_final_presentation_cases_summary(final_cases: pd.DataFrame, path: Path) -> None:
    if final_cases.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cases = {row["case_type"]: row for _, row in final_cases.iterrows()}

    _plot_case1_region(axes[0], cases.get("top_toi_region"))
    _plot_case2_anchor(axes[1], cases.get("top_ati_anchor"))
    _plot_case3_repeated(axes[2], cases.get("repeated_anchor_multi_node"))

    fig.suptitle("Tres casos finales para exposición")
    _save(fig, path)


def _plot_case1_region(ax: plt.Axes, row: pd.Series | dict | None) -> None:
    ax.set_title("A. Región top por TOI")
    if row is None:
        ax.axis("off")
        return
    values = [
        row.get("shadow_score", 0) or 0,
        1 - (row.get("I_R3", 0) or 0),
        row.get("C_phys", 0) or 0,
        row.get("S_net", 0) or 0,
    ]
    labels = ["shadow", "1-I_R3", "C_phys", "S_net"]
    ax.bar(labels, values)
    ax.set_ylim(0, max(1.05, max(values) * 1.15 if values else 1))
    ax.set_ylabel("factor")
    ax.text(0.02, 0.95, f"{row.get('node_id', '')}\nTOI={row.get('TOI', float('nan')):.3f}", transform=ax.transAxes, va="top")


def _plot_case2_anchor(ax: plt.Axes, row: pd.Series | dict | None) -> None:
    ax.set_title("B. Ancla top por ATI")
    if row is None:
        ax.axis("off")
        return
    values = [
        row.get("TOI", 0) or 0,
        max(0, row.get("Delta_rel_neighbors_best", 0) or 0),
        1 - (row.get("anchor_I_R3", row.get("I_R3", 0)) or 0),
        row.get("anchor_representativeness", 0) or row.get("A_p", 0) or 0,
    ]
    labels = ["TOI", "pos_Delta_rel", "1-I_anchor", "repr"]
    ax.bar(labels, values)
    ax.set_ylim(0, max(1.05, max(values) * 1.15 if values else 1))
    ax.text(
        0.02,
        0.95,
        f"{row.get('anchor_pl_name', '')}\n{row.get('node_id', '')}\nATI={row.get('ATI', float('nan')):.3f}",
        transform=ax.transAxes,
        va="top",
    )


def _plot_case3_repeated(ax: plt.Axes, row: pd.Series | dict | None) -> None:
    ax.set_title("C. Ancla repetida")
    ax.axis("off")
    if row is None:
        return
    text = "\n".join(
        [
            f"Ancla: {row.get('anchor_pl_name', '')}",
            f"Nodo mostrado: {row.get('node_id', '')}",
            f"n_nodes_as_anchor: {row.get('n_nodes_as_anchor', '')}",
            f"max ATI local: {row.get('ATI', float('nan')):.3f}" if pd.notna(row.get("ATI")) else "max ATI local: NA",
            f"Nodos: {row.get('nodes_list', '')}",
            f"Estabilidad: {row.get('deficit_stability_label', '')}",
        ]
    )
    ax.text(0.02, 0.98, text, va="top")
