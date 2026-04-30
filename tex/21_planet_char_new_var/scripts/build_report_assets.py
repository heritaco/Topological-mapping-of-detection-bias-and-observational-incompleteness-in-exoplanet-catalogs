"""Build figures and LaTeX tables for report 21.

Run from the repository root:
    python tex/21_planet_char_new_var/scripts/build_report_assets.py

The script intentionally does not compile LaTeX and does not require network,
GPU, Perl, or latexmk.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "tex" / "21_planet_char_new_var"
FIG_DIR = REPORT_DIR / "figures"
TAB_DIR = REPORT_DIR / "tables"

EXPECTED = {
    "candidate_property_predictions.csv": ROOT / "outputs/candidate_characterization/tables/candidate_property_predictions.csv",
    "candidate_analog_neighbors.csv": ROOT / "outputs/candidate_characterization/tables/candidate_analog_neighbors.csv",
    "validation_metrics.csv": ROOT / "outputs/candidate_characterization/tables/validation_metrics.csv",
    "validation_predictions.csv": ROOT / "outputs/candidate_characterization/tables/validation_predictions.csv",
    "candidate_characterization_summary.md": ROOT / "reports/candidate_characterization/candidate_characterization_summary.md",
    "feature_registry.yaml": ROOT / "configs/features/feature_registry.yaml",
    "feature_sets.yaml": ROOT / "configs/features/feature_sets.yaml",
}

WARNINGS: list[str] = []
FOUND_INPUTS: dict[str, bool] = {}
SUMMARY: dict[str, Any] = {}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"[build_report_assets] WARNING: {message}")


def tex_escape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return tex_escape(value)
    if not np.isfinite(x):
        return ""
    if digits == 0:
        return f"{x:.0f}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.{digits}f}"


def safe_yaml(path: Path) -> dict[str, Any]:
    FOUND_INPUTS[rel(path)] = path.exists()
    if not path.exists():
        warn(f"Missing YAML input: {rel(path)}")
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not parse YAML {rel(path)}: {exc}")
        return {}


def parse_validation_fallback(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    rows: list[dict[str, Any]] = []
    for target in ["pl_rade", "pl_bmasse", "radius_class"]:
        match = re.search(rf"{target}\s*,\s*([^,]+)\s*,\s*([0-9.]+)\s*,?\s*([^,]*)\s*,?\s*([^,]*)\s*,?\s*([^,\n]*)", text)
        if match:
            rows.append(
                {
                    "target": target,
                    "mode": match.group(1),
                    "n_test": pd.to_numeric(match.group(2), errors="coerce"),
                    "mae_q50": pd.to_numeric(match.group(3), errors="coerce"),
                    "coverage_q05_q95": pd.to_numeric(match.group(4), errors="coerce"),
                    "accuracy": pd.to_numeric(match.group(5), errors="coerce"),
                }
            )
    if rows:
        warn(f"Used fallback parser for malformed validation metrics: {rel(path)}")
    return pd.DataFrame(rows)


def read_csv_robust(path: Path, kind: str = "generic") -> pd.DataFrame:
    FOUND_INPUTS[rel(path)] = path.exists()
    if not path.exists():
        warn(f"Missing CSV input: {rel(path)}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if kind == "validation" and ("target" not in df.columns or len(df.columns) < 3):
            fallback = parse_validation_fallback(path)
            return fallback if not fallback.empty else df
        return df
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not read CSV {rel(path)} with pandas: {exc}")
        if kind == "validation":
            return parse_validation_fallback(path)
        return pd.DataFrame()


def newest(pattern: str) -> Path | None:
    paths = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return paths[0] if paths else None


def placeholder_figure(path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_barh(path: Path, labels: list[str], values: list[float], title: str, xlabel: str) -> None:
    if not labels:
        placeholder_figure(path, "No hay datos disponibles para esta figura.")
        return
    fig_h = max(4.5, 0.34 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#5277a3")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def shorten_label(value: Any, n: int = 42) -> str:
    text = str(value)
    return text if len(text) <= n else text[: n - 3] + "..."


def candidate_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    score_cols = ["candidate_score", "ati", "TOI", "toi", "topology_score", "gap_score", "shadow_score"]
    for col in score_cols:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            out = df.copy()
            out["_score"] = pd.to_numeric(out[col], errors="coerce")
            return out.sort_values("_score", ascending=False, na_position="last").drop(columns=["_score"])
    return df


def plot_validation_metrics(df: pd.DataFrame) -> None:
    path = FIG_DIR / "fig_validation_metrics.pdf"
    if df.empty:
        placeholder_figure(path, "No se encontro validation_metrics.csv.")
        return
    rows = []
    for _, row in df.iterrows():
        target = str(row.get("target", ""))
        for metric in ["mae_q50", "coverage_q05_q95", "accuracy"]:
            if metric in df.columns and pd.notna(row.get(metric)):
                rows.append((f"{target}\n{metric}", float(row[metric])))
    if not rows:
        placeholder_figure(path, "validation_metrics.csv existe, pero no contiene metricas numericas reconocibles.")
        return
    labels, values = zip(*rows)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(range(len(values)), values, color="#5f8f67")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Valor de la metrica")
    ax.set_title("Metricas de validacion por objetivo")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_candidate_intervals(df: pd.DataFrame) -> None:
    radius_path = FIG_DIR / "fig_candidate_radius_mass_intervals.pdf"
    mass_path = FIG_DIR / "fig_candidate_mass_intervals.pdf"
    if df.empty:
        placeholder_figure(radius_path, "Candidate prediction table exists but contains no rows." if EXPECTED["candidate_property_predictions.csv"].exists() else "No se encontro candidate_property_predictions.csv.")
        placeholder_figure(mass_path, "No hay datos de masa de candidatos disponibles.")
        return
    top = candidate_sort(df).head(12).copy()
    label_col = "candidate_id" if "candidate_id" in top.columns else "hostname" if "hostname" in top.columns else top.columns[0]
    labels = [shorten_label(x) for x in top[label_col].astype(str)]

    def one_interval(fig_path: Path, stem: str, title: str, xlabel: str) -> None:
        q05, q50, q95 = f"{stem}_q05", f"{stem}_q50", f"{stem}_q95"
        if q50 not in top.columns:
            placeholder_figure(fig_path, f"No hay columna {q50} para graficar intervalos.")
            return
        mid = pd.to_numeric(top[q50], errors="coerce")
        lo = pd.to_numeric(top[q05], errors="coerce") if q05 in top.columns else mid
        hi = pd.to_numeric(top[q95], errors="coerce") if q95 in top.columns else mid
        mask = mid.notna()
        if not mask.any():
            placeholder_figure(fig_path, f"La columna {q50} no contiene valores numericos.")
            return
        y = np.arange(mask.sum())
        fig, ax = plt.subplots(figsize=(9.0, max(4.8, 0.35 * mask.sum() + 1.4)))
        m = mid[mask].to_numpy(float)
        l = lo[mask].to_numpy(float)
        h = hi[mask].to_numpy(float)
        left = np.maximum(m - l, 0)
        right = np.maximum(h - m, 0)
        ax.errorbar(m, y, xerr=np.vstack([left, right]), fmt="o", color="#3f6f9f", ecolor="#9ab6d6", capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[i] for i, ok in enumerate(mask) if ok], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)

    one_interval(radius_path, "pl_rade", "Intervalos probabilisticos de radio para candidatos priorizados", "Radio planetario [R_\\oplus]")
    one_interval(mass_path, "pl_bmasse", "Intervalos probabilisticos de masa para candidatos priorizados", "Masa planetaria [M_\\oplus]")


def plot_class_probabilities(df: pd.DataFrame) -> None:
    path = FIG_DIR / "fig_radius_class_probabilities.pdf"
    note_path = TAB_DIR / "note_radius_class_probabilities.tex"
    if df.empty:
        placeholder_figure(path, "No hay filas de candidatos para probabilidades de clase.")
        note_path.write_text(r"\textit{No hay filas de candidatos para probabilidades de clase.}" + "\n", encoding="utf-8")
        return
    prob_cols = [c for c in df.columns if (c.startswith("prob_") or c.startswith("p_") or c.startswith("radius_class_")) and not c.startswith("analog_")]
    prob_cols = [c for c in prob_cols if pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not prob_cols:
        placeholder_figure(path, "No se encontraron columnas de probabilidad de clase.")
        note_path.write_text(r"\textit{No se encontraron columnas de probabilidad de clase en la tabla de predicciones.}" + "\n", encoding="utf-8")
        return
    top = candidate_sort(df).head(10).copy()
    label_col = "candidate_id" if "candidate_id" in top.columns else "hostname" if "hostname" in top.columns else top.columns[0]
    labels = [shorten_label(x, 34) for x in top[label_col].astype(str)]
    values = top[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9.0, max(4.8, 0.42 * len(top) + 1.5)))
    left = np.zeros(len(top))
    for col in prob_cols:
        vals = values[col].to_numpy(float)
        ax.barh(np.arange(len(top)), vals, left=left, label=col.replace("prob_", ""), height=0.75)
        left += vals
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Probabilidad")
    ax.set_title("Probabilidades de clase de radio")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    note_path.write_text(r"\textit{La figura muestra probabilidades de clase cuando las columnas prob\_* estan disponibles.}" + "\n", encoding="utf-8")


def plot_analog_support(df: pd.DataFrame) -> None:
    path = FIG_DIR / "fig_analog_support.pdf"
    if df.empty:
        placeholder_figure(path, "No se encontro tabla de analogos o no contiene filas.")
        return
    id_col = "candidate_id" if "candidate_id" in df.columns else "hostname" if "hostname" in df.columns else None
    if id_col is None:
        placeholder_figure(path, "La tabla de analogos no contiene candidate_id ni hostname.")
        return
    grouped = df.groupby(id_col).agg(n_analog=("analog_rank", "count")).reset_index()
    if "analog_weight" in df.columns:
        weights = df.assign(analog_weight=pd.to_numeric(df["analog_weight"], errors="coerce")).groupby(id_col)["analog_weight"].mean().reset_index(name="mean_weight")
        grouped = grouped.merge(weights, on=id_col, how="left")
        grouped = grouped.sort_values(["n_analog", "mean_weight"], ascending=False)
    else:
        grouped = grouped.sort_values("n_analog", ascending=False)
    top = grouped.head(15)
    save_barh(path, [shorten_label(x, 44) for x in top[id_col]], top["n_analog"].astype(float).tolist(), "Soporte empirico por analogos observados", "Numero de analogos")


def feature_count(raw_features: Any) -> int:
    return len(raw_features or []) if isinstance(raw_features, list) else 0


def plot_feature_group_summary(registry: dict[str, Any]) -> None:
    path = FIG_DIR / "fig_feature_group_summary.pdf"
    groups = registry.get("feature_groups", {}) or {}
    if not groups:
        placeholder_figure(path, "No se pudo leer feature_registry.yaml.")
        return
    labels = list(groups.keys())
    values = [feature_count(groups[g].get("features", [])) for g in labels]
    save_barh(path, labels, values, "Variables por grupo de feature governance", "Numero de variables")


def plot_feature_set_composition(feature_sets_doc: dict[str, Any]) -> None:
    path = FIG_DIR / "fig_feature_set_composition.pdf"
    sets = feature_sets_doc.get("feature_sets", {}) or {}
    if not sets:
        placeholder_figure(path, "No se pudo leer feature_sets.yaml.")
        return
    group_names = sorted({group for cfg in sets.values() for group in (cfg.get("include", []) or []) if "+" not in str(group)})
    set_names = list(sets.keys())
    matrix = np.zeros((len(set_names), len(group_names)))
    for i, set_name in enumerate(set_names):
        includes = set(str(x) for x in sets[set_name].get("include", []) or [])
        for j, group in enumerate(group_names):
            if group in includes:
                matrix[i, j] = 1.0
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(group_names) + 3.0), max(4.2, 0.45 * len(set_names) + 2.0)))
    ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(group_names)))
    ax.set_xticklabels(group_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(set_names)))
    ax.set_yticklabels(set_names, fontsize=8)
    ax.set_title("Composicion de feature sets")
    for i in range(len(set_names)):
        for j in range(len(group_names)):
            ax.text(j, i, "1" if matrix[i, j] else "", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_missingness() -> Path | None:
    path = FIG_DIR / "fig_feature_missingness.pdf"
    audit_path = newest("outputs/runs/*/tables/feature_audit/feature_missingness.csv") or newest("outputs/runs/*/tables/feature_audit/feature_availability.csv")
    SUMMARY["newest_feature_audit"] = rel(audit_path) if audit_path else None
    if audit_path is None:
        placeholder_figure(path, "No se encontraron salidas de feature_audit; queda pendiente una ejecucion completa.")
        return None
    df = read_csv_robust(audit_path)
    if df.empty or "missing_percentage" not in df.columns:
        placeholder_figure(path, "La auditoria de features existe, pero no contiene missing_percentage utilizable.")
        return audit_path
    df = df.copy()
    df["missing_percentage"] = pd.to_numeric(df["missing_percentage"], errors="coerce")
    finite = df[df["missing_percentage"].notna()].sort_values("missing_percentage", ascending=False).head(20)
    if finite.empty:
        placeholder_figure(path, "La auditoria disponible parece ser dry-run o no contiene porcentajes finitos de missingness.")
        return audit_path
    label_col = "feature_name" if "feature_name" in finite.columns else finite.columns[0]
    save_barh(path, finite[label_col].astype(str).tolist(), (100 * finite["missing_percentage"]).tolist(), "Top 20 features con mayor missingness", "Missingness [%]")
    return audit_path


def plot_ablation_metrics() -> Path | None:
    path = FIG_DIR / "fig_ablation_metrics.pdf"
    ablation_path = newest("outputs/runs/*/tables/feature_ablation/ablation_metrics.csv")
    SUMMARY["newest_feature_ablation"] = rel(ablation_path) if ablation_path else None
    if ablation_path is None:
        placeholder_figure(path, "La ablation aun no se ha ejecutado; es el siguiente paso de sensibilidad.")
        return None
    df = read_csv_robust(ablation_path)
    if df.empty:
        placeholder_figure(path, "La tabla de ablation existe, pero no contiene filas.")
        return ablation_path
    metric_cols = [c for c in ["mae_q50", "logMAE", "coverage_q05_q95", "accuracy", "f1"] if c in df.columns]
    if not metric_cols:
        placeholder_figure(path, "No se encontraron metricas reconocibles en ablation_metrics.csv.")
        return ablation_path
    feature_col = "feature_set" if "feature_set" in df.columns else "experiment" if "experiment" in df.columns else df.columns[0]
    metric = metric_cols[0]
    plot_df = df[[feature_col, metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna().head(20)
    save_barh(path, [shorten_label(x, 40) for x in plot_df[feature_col]], plot_df[metric].astype(float).tolist(), f"Ablation por feature set: {metric}", metric)
    return ablation_path


def write_validation_table(df: pd.DataFrame) -> None:
    path = TAB_DIR / "tab_validation_metrics.tex"
    cols = ["target", "mode", "n_test", "mae_q50", "coverage_q05_q95", "accuracy"]
    if df.empty:
        path.write_text(r"\begin{table}[H]\centering\caption{Metricas de validacion}\textit{No se encontro validation\_metrics.csv.}\end{table}" + "\n", encoding="utf-8")
        return
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Metricas de validacion por planetas retenidos}",
        r"\label{tab:validation-metrics}",
        r"\small",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Target & Modo & $n_{\mathrm{test}}$ & MAE q50 & Cobertura 5--95 & Accuracy \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        vals = [tex_escape(row.get("target", "")), tex_escape(row.get("mode", "")), fmt(row.get("n_test"), 0), fmt(row.get("mae_q50")), fmt(row.get("coverage_q05_q95")), fmt(row.get("accuracy"))]
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feature_groups_table(registry: dict[str, Any]) -> None:
    path = TAB_DIR / "tab_feature_groups.tex"
    groups = registry.get("feature_groups", {}) or {}
    lines = [
        r"\begin{longtable}{p{0.20\linewidth}p{0.12\linewidth}p{0.11\linewidth}rp{0.39\linewidth}}",
        r"\caption{Grupos de variables registrados para feature governance}\label{tab:feature-groups}\\",
        r"\toprule",
        r"Grupo & Rol & Riesgo & $n$ & Accion recomendada \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Grupo & Rol & Riesgo & $n$ & Accion recomendada \\",
        r"\midrule",
        r"\endhead",
    ]
    for group, cfg in groups.items():
        lines.append(
            f"{tex_escape(group)} & {tex_escape(cfg.get('role', ''))} & {tex_escape(cfg.get('leakage_risk', ''))} & {feature_count(cfg.get('features', []))} & {tex_escape(cfg.get('recommended_action', ''))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feature_sets_table(feature_sets_doc: dict[str, Any]) -> None:
    path = TAB_DIR / "tab_feature_sets.tex"
    sets = feature_sets_doc.get("feature_sets", {}) or {}
    lines = [
        r"\begin{longtable}{p{0.27\linewidth}p{0.36\linewidth}p{0.18\linewidth}p{0.10\linewidth}}",
        r"\caption{Feature sets configurados para prediccion, Mapper y sensibilidad}\label{tab:feature-sets}\\",
        r"\toprule",
        r"Feature set & Incluye & Excluye / audit-only & Leakage-safe \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Feature set & Incluye & Excluye / audit-only & Leakage-safe \\",
        r"\midrule",
        r"\endhead",
    ]
    for name, cfg in sets.items():
        include = ", ".join(str(x) for x in cfg.get("include", []) or [])
        special = ", ".join(str(x) for x in (cfg.get("exclude", []) or []) + (cfg.get("audit_only", []) or []))
        lines.append(f"{tex_escape(name)} & {tex_escape(include)} & {tex_escape(special)} & {tex_escape(cfg.get('leakage_safe', ''))} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_top_candidates_table(df: pd.DataFrame) -> None:
    path = TAB_DIR / "tab_top_candidates.tex"
    status_path = TAB_DIR / "candidate_prediction_status.tex"
    if df.empty:
        text = (
            "La infraestructura ya existe, pero esta corrida no produjo filas de candidatos caracterizados; "
            "esto indica que falta conectar o regenerar la tabla de candidatos de entrada o que el archivo esta vacio."
        )
        path.write_text(r"\begin{table}[H]\centering\caption{Candidatos caracterizados}\textit{" + tex_escape(text) + r"}\end{table}" + "\n", encoding="utf-8")
        status_path.write_text(tex_escape(text) + "\n", encoding="utf-8")
        SUMMARY["candidate_predictions_rows"] = 0
        return
    SUMMARY["candidate_predictions_rows"] = int(len(df))
    status_path.write_text(f"La tabla de predicciones contiene {len(df)} filas de candidatos caracterizados probabilisticamente.\n", encoding="utf-8")
    top = candidate_sort(df).head(10)
    wanted = [
        ("candidate_id", "Candidato"),
        ("hostname", "Host"),
        ("pl_orbper", "$P$ [d]"),
        ("pl_rade_q50", "$R_p$ q50"),
        ("pl_bmasse_q50", "$M_p$ q50"),
        ("predicted_radius_class", "Clase"),
        ("transit_probability_q50", "Transito"),
        ("rv_proxy_q50", "RV proxy"),
    ]
    cols = [(c, h) for c, h in wanted if c in top.columns]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Top 10 candidatos por score disponible}",
        r"\label{tab:top-candidates}",
        r"\scriptsize",
        r"\begin{tabular}{" + "l" * len(cols) + r"}",
        r"\toprule",
        " & ".join(h for _, h in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in top.iterrows():
        vals = []
        for col, _ in cols:
            value = row.get(col, "")
            vals.append(fmt(value) if col not in {"candidate_id", "hostname", "predicted_radius_class"} else tex_escape(shorten_label(value, 28)))
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_inventory_table(audit_path: Path | None, ablation_path: Path | None) -> None:
    path = TAB_DIR / "tab_outputs_inventory.tex"
    items = list(EXPECTED.items())
    items.append(("newest feature audit tables", audit_path))
    items.append(("newest feature ablation tables", ablation_path))
    lines = [
        r"\begin{longtable}{p{0.43\linewidth}p{0.38\linewidth}p{0.10\linewidth}}",
        r"\caption{Inventario de artefactos usados por el reporte}\label{tab:outputs-inventory}\\",
        r"\toprule",
        r"Artefacto esperado & Ruta & Existe \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Artefacto esperado & Ruta & Existe \\",
        r"\midrule",
        r"\endhead",
    ]
    for name, path_obj in items:
        exists = bool(path_obj and Path(path_obj).exists())
        lines.append(f"{tex_escape(name)} & {tex_escape(rel(Path(path_obj)) if path_obj else 'no disponible')} & {tex_escape('si' if exists else 'no')} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result_note(validation: pd.DataFrame) -> None:
    path = TAB_DIR / "validation_result_note.tex"
    if validation.empty:
        path.write_text("No se encontro una tabla de validacion utilizable.\n", encoding="utf-8")
        return
    parts = []
    for _, row in validation.iterrows():
        target = row.get("target")
        if target == "pl_rade":
            parts.append(f"radio MAE q50 = {fmt(row.get('mae_q50'))}, cobertura 5--95 = {fmt(row.get('coverage_q05_q95'))}")
        elif target == "pl_bmasse":
            parts.append(f"masa MAE q50 = {fmt(row.get('mae_q50'))}, cobertura 5--95 = {fmt(row.get('coverage_q05_q95'))}")
        elif target == "radius_class":
            parts.append(f"accuracy de clase de radio = {fmt(row.get('accuracy'))}")
    n_test = validation["n_test"].dropna().iloc[0] if "n_test" in validation.columns and validation["n_test"].notna().any() else ""
    text = "Resultado actual: " + "; ".join(parts)
    if n_test != "":
        text += f" con n_test = {fmt(n_test, 0)}."
    path.write_text(tex_escape(text) + "\n", encoding="utf-8")


def write_build_warnings() -> None:
    (TAB_DIR / "build_warnings.tex").write_text(
        "\n".join([r"\begin{itemize}[leftmargin=*]"] + [f"  \\item {tex_escape(w)}" for w in WARNINGS] + [r"\end{itemize}"]) if WARNINGS else r"\textit{No se registraron advertencias durante la generacion de assets.}" + "\n",
        encoding="utf-8",
    )
    log = REPORT_DIR / "scripts" / "build_report_assets.log"
    log.write_text("\n".join(WARNINGS) + ("\n" if WARNINGS else ""), encoding="utf-8")


def write_summary_json() -> None:
    SUMMARY["generated_at"] = datetime.now().isoformat(timespec="seconds")
    SUMMARY["warnings"] = WARNINGS
    SUMMARY["found_inputs"] = FOUND_INPUTS
    (REPORT_DIR / "asset_build_summary.json").write_text(json.dumps(SUMMARY, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    validation = read_csv_robust(EXPECTED["validation_metrics.csv"], kind="validation")
    predictions = read_csv_robust(EXPECTED["candidate_property_predictions.csv"])
    analogs = read_csv_robust(EXPECTED["candidate_analog_neighbors.csv"])
    for key in ["validation_predictions.csv", "candidate_characterization_summary.md"]:
        FOUND_INPUTS[rel(EXPECTED[key])] = EXPECTED[key].exists()
        if not EXPECTED[key].exists():
            warn(f"Missing expected artifact: {rel(EXPECTED[key])}")
    registry = safe_yaml(EXPECTED["feature_registry.yaml"])
    feature_sets = safe_yaml(EXPECTED["feature_sets.yaml"])

    SUMMARY["validation_metrics_rows"] = int(len(validation))
    SUMMARY["candidate_predictions_rows"] = int(len(predictions))
    SUMMARY["analog_neighbor_rows"] = int(len(analogs))

    plot_validation_metrics(validation)
    plot_candidate_intervals(predictions)
    plot_class_probabilities(predictions)
    plot_analog_support(analogs)
    plot_feature_group_summary(registry)
    plot_feature_set_composition(feature_sets)
    audit_path = plot_feature_missingness()
    ablation_path = plot_ablation_metrics()

    write_validation_table(validation)
    write_feature_groups_table(registry)
    write_feature_sets_table(feature_sets)
    write_top_candidates_table(predictions)
    write_inventory_table(audit_path, ablation_path)
    write_result_note(validation)
    write_build_warnings()
    write_summary_json()
    print(f"[build_report_assets] Generated assets in {rel(REPORT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
