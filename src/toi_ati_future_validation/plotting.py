from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RADIUS_ORDER = ["delta_rel_kNN", "delta_rel_node_median", "delta_rel_node_q75"]
RADIUS_LABELS = {
    "delta_rel_kNN": "r_kNN",
    "delta_rel_node_median": "r_node_median",
    "delta_rel_node_q75": "r_node_q75",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_stable_vs_sensitive_deficit(stability: pd.DataFrame, path: Path) -> None:
    if stability.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = sorted(stability["deficit_stability_class"].astype(str).dropna().unique().tolist())
    colors = {
        "stable_positive_deficit": "tab:blue",
        "small_but_stable_deficit": "tab:cyan",
        "radius_sensitive_deficit": "tab:orange",
        "unstable_due_to_large_radius": "tab:red",
        "no_deficit_or_overpopulated": "tab:gray",
    }
    for label in classes:
        subset = stability[stability["deficit_stability_class"].astype(str) == label]
        ax.scatter(
            pd.to_numeric(subset["delta_rel_mean"], errors="coerce"),
            pd.to_numeric(subset["radius_sensitivity_score"], errors="coerce"),
            label=label,
            alpha=0.8,
            color=colors.get(label, None),
        )
    top = stability.sort_values(["stable_deficit_score", "ATI"], ascending=[False, False]).head(5)
    for _, row in top.iterrows():
        ax.annotate(str(row.get("anchor_pl_name")), (row.get("delta_rel_mean"), row.get("radius_sensitivity_score")), fontsize=8)
    ax.axvline(0, linewidth=1, color="black")
    ax.set_xlabel("Delta_rel medio")
    ax.set_ylabel("Sensibilidad por radio")
    ax.set_title("Casos estables vs sensibles al radio")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_ati_vs_ati_conservative(anchors: pd.DataFrame, path: Path) -> None:
    if anchors.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    x = pd.to_numeric(anchors["ATI_original"], errors="coerce")
    y = pd.to_numeric(anchors["ATI_conservative"], errors="coerce")
    ax.scatter(x, y, alpha=0.75)
    names_to_label = {"HIP 97166 c", "HIP 90988 b", "HD 42012 b", "HD 4313 b"}
    subset = anchors[anchors["anchor_pl_name"].astype(str).isin(names_to_label)]
    for _, row in subset.iterrows():
        ax.annotate(f"{row.get('anchor_pl_name')} / {row.get('node_id')}", (row.get("ATI_original"), row.get("ATI_conservative")), fontsize=8)
    max_value = max(float(x.max(skipna=True)) if not x.dropna().empty else 0.0, float(y.max(skipna=True)) if not y.dropna().empty else 0.0)
    ax.plot([0, max_value], [0, max_value], linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("ATI original")
    ax.set_ylabel("ATI conservador")
    ax.set_title("ATI original vs ATI conservador")
    _save(fig, path)


def plot_rank_shift_after_stability_penalty(anchors: pd.DataFrame, path: Path, *, top_n: int = 10) -> None:
    if anchors.empty:
        return
    df = anchors.sort_values("rank_shift", ascending=False).head(top_n).copy()
    labels = (df["anchor_pl_name"].astype(str) + " / " + df["node_id"].astype(str)).tolist()
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 0.2 for i in x], df["rank_ATI_original"], width=0.4, label="rank original")
    ax.bar([i + 0.2 for i in x], df["rank_ATI_conservative"], width=0.4, label="rank conservador")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Rango")
    ax.set_title("Cambio de ranking tras penalizar estabilidad")
    ax.legend()
    _save(fig, path)


def plot_final_future_work_cases(cases: pd.DataFrame, path: Path) -> None:
    if cases.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4 + 0.5 * len(cases)))
    ax.axis("off")
    lines = ["Casos finales para trabajo futuro", ""]
    for _, row in cases.iterrows():
        label = f"{row.get('case_type')}: {row.get('anchor_pl_name') or row.get('node_id')}"
        stats = f"TOI={row.get('TOI', float('nan')):.3f} | ATI_cons={row.get('ATI_conservative', float('nan')):.3f} | clase={row.get('deficit_stability_class', '')}"
        lines.extend([label, stats, str(row.get("how_to_present", "")), ""])
    ax.text(0.01, 0.99, "\n".join(lines), va="top", family="monospace")
    _save(fig, path)


def plot_deficit_profiles_selected_anchors(stability: pd.DataFrame, path: Path) -> None:
    if stability.empty:
        return
    candidates = ["HIP 97166 c", "HIP 90988 b", "HD 42012 b", "HD 4313 b"]
    subset = stability[stability["anchor_pl_name"].astype(str).isin(candidates)].copy()
    if subset.empty:
        subset = stability.sort_values(["stable_deficit_score", "ATI"], ascending=[False, False]).head(4)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(RADIUS_ORDER)))
    for _, row in subset.iterrows():
        values = [row.get(column) for column in RADIUS_ORDER]
        ax.plot(x, values, marker="o", label=f"{row.get('anchor_pl_name')} / {row.get('node_id')}")
    ax.axhline(0, linewidth=1, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([RADIUS_LABELS[column] for column in RADIUS_ORDER])
    ax.set_ylabel("Delta_rel")
    ax.set_title("Perfiles de deficit por radio para anclas seleccionadas")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_observational_priority_ranking(candidates: pd.DataFrame, path: Path) -> None:
    if candidates.empty:
        return
    df = candidates.sort_values("observational_priority_score", ascending=True)
    labels = df["anchor_pl_name"].astype(str) + " / " + df["node_id"].astype(str)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df))))
    ax.barh(labels, df["observational_priority_score"])
    ax.set_xlabel("Puntaje de prioridad observacional")
    ax.set_ylabel("Ancla / nodo")
    ax.set_title("Top 10 candidatos para prioridad observacional")
    _save(fig, path)


def plot_technical_audit_cases_summary(issues: pd.DataFrame, path: Path) -> None:
    if issues.empty:
        return
    counts = issues["issue_type"].astype(str).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(counts.index.tolist(), counts.values.tolist())
    ax.set_xlabel("Numero de casos")
    ax.set_ylabel("Tipo de auditoria")
    ax.set_title("Resumen de casos para auditoria tecnica")
    _save(fig, path)

