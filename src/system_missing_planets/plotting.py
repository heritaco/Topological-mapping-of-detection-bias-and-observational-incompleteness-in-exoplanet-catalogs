from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import safe_hostname


METHOD_COLORS = {
    "Transit": "#1f77b4",
    "Radial Velocity": "#d95f02",
    "Mixed": "#6b7280",
    "Unknown": "#6b7280",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_top_high_priority_systems(system_summary: pd.DataFrame, path: Path, *, top_n: int = 15) -> None:
    if system_summary.empty:
        return
    frame = system_summary.sort_values(["max_candidate_priority_score", "n_candidate_missing_planets"], ascending=[False, False]).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(frame))))
    labels = frame["hostname"].astype(str)
    values = pd.to_numeric(frame["max_candidate_priority_score"], errors="coerce").fillna(0.0)
    ax.barh(labels, values, color="#2a9d8f")
    ax.set_xlabel("Max candidate priority score")
    ax.set_ylabel("System")
    ax.set_title("Top systems by candidate priority")
    ax.invert_yaxis()
    _save(fig, path)


def plot_candidate_priority_distribution(candidates: pd.DataFrame, path: Path) -> None:
    if candidates.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pd.to_numeric(candidates["candidate_priority_score"], errors="coerce").dropna(), bins=20, color="#264653", alpha=0.85)
    ax.set_xlabel("Candidate priority score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of candidate priority scores")
    _save(fig, path)


def plot_gap_ratio_vs_priority(candidates: pd.DataFrame, path: Path) -> None:
    if candidates.empty:
        return
    colors = candidates["candidate_priority_class"].astype(str).map({"high": "#c1121f", "medium": "#f4a261", "low": "#457b9d"}).fillna("#6b7280")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        pd.to_numeric(candidates["gap_period_ratio"], errors="coerce"),
        pd.to_numeric(candidates["candidate_priority_score"], errors="coerce"),
        c=colors.tolist(),
        alpha=0.75,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Gap period ratio")
    ax.set_ylabel("Candidate priority score")
    ax.set_title("Gap ratio vs candidate priority")
    _save(fig, path)


def plot_system_architecture(hostname: str, observed: pd.DataFrame, candidates: pd.DataFrame, path: Path) -> None:
    if observed.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 3.8))
    observed_periods = pd.to_numeric(observed["pl_orbper"], errors="coerce")
    observed_methods = observed.get("discoverymethod", pd.Series(["Unknown"] * len(observed), index=observed.index)).astype(str).map(
        lambda value: METHOD_COLORS.get("Transit" if "transit" in value.lower() else "Radial Velocity" if "radial velocity" in value.lower() else "Unknown")
    )
    ax.scatter(observed_periods, np.ones(len(observed)), c=observed_methods.tolist(), s=70, edgecolor="black", linewidth=0.6, zorder=3)
    for _, row in observed.iterrows():
        ax.annotate(str(row["pl_name"]), (row["pl_orbper"], 1.02), fontsize=8, rotation=25)

    if not candidates.empty:
        candidate_periods = pd.to_numeric(candidates["candidate_period_days"], errors="coerce")
        sizes = 60 + 180 * pd.to_numeric(candidates["candidate_priority_score"], errors="coerce").fillna(0.0)
        edge_colors = candidates["candidate_priority_class"].astype(str).map({"high": "#c1121f", "medium": "#f4a261", "low": "#457b9d"}).fillna("#6b7280")
        ax.scatter(candidate_periods, np.full(len(candidates), 1.10), facecolors="white", edgecolors=edge_colors.tolist(), s=sizes, linewidth=1.5, zorder=4)
        for _, row in candidates.iterrows():
            ax.annotate(f"C{int(row['candidate_rank_in_gap'])}", (row["candidate_period_days"], 1.12), fontsize=8)

    ax.set_xscale("log")
    ax.set_yticks([1.0, 1.10])
    ax.set_yticklabels(["Observed", "Candidates"])
    ax.set_xlabel("Orbital period [days]")
    ax.set_title(f"System architecture: {hostname}")
    ax.grid(True, axis="x", alpha=0.25)
    _save(fig, path)


def make_figures(system_summary: pd.DataFrame, candidates: pd.DataFrame, catalog: pd.DataFrame, figures_dir: Path, *, top_systems: int = 10) -> None:
    plot_top_high_priority_systems(system_summary, figures_dir / "top_high_priority_systems.pdf")
    plot_candidate_priority_distribution(candidates, figures_dir / "candidate_priority_distribution.pdf")
    plot_gap_ratio_vs_priority(candidates, figures_dir / "gap_ratio_vs_priority.pdf")
    if candidates.empty:
        return
    top_hosts = (
        system_summary.sort_values(["max_candidate_priority_score", "n_candidate_missing_planets"], ascending=[False, False])["hostname"]
        .astype(str)
        .head(top_systems)
        .tolist()
    )
    for hostname in top_hosts:
        observed = catalog[catalog["hostname"].astype(str) == hostname].sort_values("pl_orbper")
        system_candidates = candidates[candidates["hostname"].astype(str) == hostname].sort_values("candidate_period_days")
        if system_candidates.empty:
            continue
        plot_system_architecture(hostname, observed, system_candidates, figures_dir / f"system_architecture_{safe_hostname(hostname)}.pdf")
