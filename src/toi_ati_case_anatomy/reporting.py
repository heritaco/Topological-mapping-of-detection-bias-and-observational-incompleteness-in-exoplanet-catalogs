from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .validation import contains_forbidden_claim


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_manifest(
    path: Path,
    *,
    config_path: str | None,
    inputs_loaded: Dict[str, int],
    warnings: Dict[str, str],
    audit: Dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "inputs_loaded_rows": inputs_loaded,
        "warnings": warnings,
        "audit": audit or {},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_summary(
    path: Path,
    *,
    sentences: Iterable[str],
    top_regions: pd.DataFrame,
    top_anchors: pd.DataFrame,
    top_anchor_radius_summary: pd.DataFrame,
    final_cases: pd.DataFrame,
    deficit_audit: Dict[str, int] | None = None,
    figure5_audit: Dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# TOI/ATI case anatomy summary", ""]
    lines.extend(f"- {sentence}" for sentence in sentences)
    lines.append("")

    lines.append("## Top regions")
    if not top_regions.empty:
        columns = [c for c in ["node_id", "TOI", "shadow_score", "I_R3", "C_phys", "S_net", "top_method"] if c in top_regions.columns]
        lines.append(_markdown_table(top_regions[columns].head(10)))
    else:
        lines.append("No top regions available.")
    lines.append("")

    lines.append("## Top anchors")
    if not top_anchors.empty:
        columns = [c for c in ["anchor_pl_name", "node_id", "ATI", "TOI", "delta_rel_neighbors_best", "deficit_class"] if c in top_anchors.columns]
        lines.append(_markdown_table(top_anchors[columns].head(10)))
    else:
        lines.append("No top anchors available.")
    lines.append("")

    lines.append("## Top anchors with radius-by-radius deficit")
    lines.append("Estas tablas evitan sobreinterpretar `delta_rel_neighbors_best`: el valor maximo resume, pero no reemplaza, la lectura por los tres radios locales.")
    if not top_anchor_radius_summary.empty:
        columns = [
            c
            for c in [
                "anchor_pl_name",
                "node_id",
                "ATI",
                "best_radius_type",
                "Delta_rel_neighbors_best",
                "mean_Delta_rel_neighbors",
                "median_Delta_rel_neighbors",
                "deficit_stability_label",
                "interpretation_short",
            ]
            if c in top_anchor_radius_summary.columns
        ]
        lines.append(_markdown_table(top_anchor_radius_summary[columns]))
    else:
        lines.append("No hay tablas por radio disponibles.")
    lines.append("")

    lines.append("## Three final presentation cases")
    if not final_cases.empty:
        columns = [
            c
            for c in [
                "case_type",
                "anchor_pl_name",
                "node_id",
                "TOI",
                "ATI",
                "Delta_rel_neighbors_best",
                "deficit_stability_label",
                "how_to_present",
                "caution_text",
            ]
            if c in final_cases.columns
        ]
        lines.append(_markdown_table(final_cases[columns]))
    else:
        lines.append("No final presentation cases available.")
    lines.append("")

    lines.append("## Recommended presentation sequence")
    lines.append("1. Explicar primero la region top por TOI para abrir el indice regional.")
    lines.append("2. Explicar despues el ancla top por ATI para aterrizar la priorizacion local en R^3.")
    lines.append("3. Explicar al final el ancla repetida en varios nodos como caso de transicion Mapper por solapamiento de cubiertas.")
    lines.append("4. Cerrar con limitaciones: no hay completitud instrumental, no hay confirmacion de objetos ausentes y el deficit puede ser sensible al radio.")
    lines.append("")

    lines.append("## Deficit formula audit")
    if deficit_audit:
        raw_gt_one = int(deficit_audit.get("raw_delta_rel_gt_one_count", 0))
        recomputed_gt_one = int(deficit_audit.get("recomputed_delta_rel_gt_one_count", 0))
        mismatch_count = int(deficit_audit.get("mismatch_count", 0))
        lines.append(f"- Raw `delta_rel` > 1 count: {raw_gt_one}")
        lines.append(f"- Recomputed `Delta_rel` > 1 count: {recomputed_gt_one}")
        lines.append(f"- Formula mismatches recomputed: {mismatch_count}")
        if recomputed_gt_one > 0:
            lines.append("- Advertencia: algun `Delta_rel` recomputado sigue siendo mayor que 1; eso requiere auditoria manual antes de exposicion.")
        else:
            lines.append("- La auditoria no encontro `Delta_rel` recomputado mayor que 1. Si una figura previa parecia exceder 1, la lectura mas probable es que estaba mostrando `Delta_N` o una escala mal rotulada.")
    else:
        lines.append("No deficit audit available.")
    lines.append("")

    lines.append("## Figure 5 audit")
    if figure5_audit:
        lines.append(f"- Previous y column: {figure5_audit.get('previous_y_column')}")
        lines.append(f"- Previous y max: {figure5_audit.get('previous_y_max')}")
        lines.append(f"- Recomputed Delta_rel max: {figure5_audit.get('recomputed_delta_rel_max')}")
        lines.append(f"- Recomputed Delta_N max: {figure5_audit.get('recomputed_delta_N_max')}")
        lines.append(f"- Decision: {figure5_audit.get('decision')}")
        lines.append(f"- Reason: {figure5_audit.get('reason')}")
        lines.append("- Esta versión separa explícitamente déficit relativo y déficit absoluto. El primero sirve para comparar radios con escalas distintas; el segundo depende fuertemente del tamaño de la bola local.")
        lines.append("- Ninguno debe interpretarse como cantidad confirmada de planetas ausentes.")
    else:
        lines.append("No Figure 5 audit available.")
    lines.append("")

    lines.append("## Caution")
    lines.append("TOI/ATI no descubre planetas ausentes; prioriza donde buscar evidencia de incompletitud observacional.")
    text = "\n".join(lines)
    forbidden = contains_forbidden_claim(text)
    if forbidden:
        raise ValueError(f"Summary contains forbidden claims: {forbidden}")
    path.write_text(text, encoding="utf-8")


def write_top_anchor_deficit_tables_tex(
    path: Path,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if detail.empty or summary.empty:
        path.write_text("% No top-anchor deficit tables available.\n", encoding="utf-8")
        return

    lines = []
    for _, summary_row in summary.iterrows():
        anchor_name = summary_row.get("anchor_pl_name", "unknown")
        node_id = summary_row.get("node_id", "unknown")
        subset = detail[(detail["anchor_pl_name"] == anchor_name) & (detail["node_id"] == node_id)].copy()
        subset["radius_sort"] = subset["radius_type"].map({"r_kNN": 0, "r_node_median": 1, "r_node_q75": 2})
        subset = subset.sort_values("radius_sort")
        lines.append(
            f"\\subsection{{Deficit por radio para \\texttt{{{_escape_latex(anchor_name)}}} en \\texttt{{{_escape_latex(node_id)}}}}}"
        )
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append(
            f"\\caption{{Conteos observados, esperados y deficit local para el planeta ancla \\texttt{{{_escape_latex(anchor_name)}}}.}}"
        )
        lines.append("\\begin{tabular}{lrrrr}")
        lines.append("\\toprule")
        lines.append("Radio & $N_{\\mathrm{obs}}$ & $N_{\\mathrm{exp}}$ & $\\Delta N$ & $\\Delta_{\\mathrm{rel}}$ \\\\")
        lines.append("\\midrule")
        for _, row in subset.iterrows():
            lines.append(
                f"{_latex_radius(row.get('radius_type'))} & "
                f"{_fmt(row.get('N_obs'))} & "
                f"{_fmt(row.get('N_exp_neighbors'))} & "
                f"{_fmt(row.get('delta_N_neighbors_recomputed', row.get('Delta_N_neighbors')))} & "
                f"{_fmt(row.get('delta_rel_neighbors_recomputed', row.get('Delta_rel_neighbors')), digits=3)} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append(_stability_paragraph(summary_row.get("deficit_stability_label", "undefined")))
        lines.append("")
    text = "\n".join(lines) + "\n"
    forbidden = contains_forbidden_claim(text)
    if forbidden:
        raise ValueError(f"LaTeX partial contains forbidden claims: {forbidden}")
    path.write_text(text, encoding="utf-8")


def write_latex_report(
    path: Path,
    *,
    final_cases: pd.DataFrame,
    top_anchor_deficit_input_path: str = "top_anchor_deficit_tables.tex",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final_case_table = _latex_final_cases_table(final_cases)
    tex = r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-nodecimaldot]{babel}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,float,hyperref}
\newcommand{\figroot}{../../outputs/toi_ati_case_anatomy/figures_pdf}
\newcommand{\safeincludegraphics}[2][]{%
  \IfFileExists{#2}{\includegraphics[#1]{#2}}{\fbox{\begin{minipage}{0.86\textwidth}\centering Figura no encontrada: \texttt{#2}\end{minipage}}}%
}
\title{Anatomia de casos TOI/ATI: regiones Mapper y planetas ancla}
\author{Proyecto Mapper/TDA de exoplanetas}
\date{}
\begin{document}
\maketitle

\section{Introduccion}
Este reporte abre el ranking global TOI/ATI para explicar por que ciertas regiones Mapper y planetas ancla fueron priorizados. El objetivo no es afirmar descubrimientos de planetas, sino auditar la anatomia de los indices.

\section{Que problema resuelve la anatomia TOI/ATI}
La salida principal es una anatomia de los casos top: que factor empuja cada indice, que factor lo limita y que tan estable parece el deficit relativo al cambiar de radio.

\section{Datos usados}
El modulo consume tablas generadas por \texttt{topological\_incompleteness\_index} y, cuando existen, tablas contextuales de sombra observacional y estudios locales.

\section{Descomposicion de TOI}
\[
\mathrm{TOI}(v)=\mathrm{Shadow}(v)(1-I_{R^3}(v))C_{\mathrm{phys}}(v)S_{\mathrm{net}}(v).
\]
TOI debe leerse como una prioridad regional que combina frontera observacional, baja imputacion, continuidad fisica con vecinos y soporte de red.

\section{Descomposicion de ATI}
\[
\mathrm{ATI}(p^*)=\mathrm{TOI}(v)\Delta_{\mathrm{rel,best}}(p^*)(1-I_{R^3}(p^*))A(p^*).
\]
ATI debe leerse como una prioridad local de inspeccion. El termino $\Delta_{\mathrm{rel,best}}$ debe revisarse junto con los radios individuales.

\section{Top regiones}
\begin{figure}[H]\centering
\safeincludegraphics[width=.9\textwidth]{\figroot/top_regions_toi_case_anatomy.pdf}
\caption{Top regiones por TOI.}
\end{figure}

\begin{figure}[H]\centering
\safeincludegraphics[width=.9\textwidth]{\figroot/toi_factor_decomposition.pdf}
\caption{Descomposicion visual de factores TOI.}
\end{figure}

\section{Top planetas ancla}
\begin{figure}[H]\centering
\safeincludegraphics[width=.9\textwidth]{\figroot/top_anchors_ati_case_anatomy.pdf}
\caption{Top planetas ancla por ATI.}
\end{figure}

\begin{figure}[H]\centering
\safeincludegraphics[width=.9\textwidth]{\figroot/ati_factor_decomposition.pdf}
\caption{Descomposicion visual de factores ATI.}
\end{figure}

\section{Tablas de déficit por radio para las anclas principales}
No basta con usar $\Delta_{\mathrm{rel,best}}$ porque ese máximo puede inflar la lectura de un caso local. Por eso se reportan los tres radios: $r_{\mathrm{kNN}}$, $r_{\mathrm{node\_median}}$ y $r_{\mathrm{node\_q75}}$. Cada tabla muestra $N_{\mathrm{obs}}$, la referencia local $N_{\mathrm{exp}}$, el déficit absoluto $\Delta N$ y el déficit relativo $\Delta_{\mathrm{rel}}$.

La versión relativa usa $\Delta_{\mathrm{rel}}$ recomputado desde $N_{\mathrm{obs}}$ y $N_{\mathrm{exp}}$. Esto permite comparar radios con escalas de conteo distintas. En cambio, $\Delta N$ puede ser útil como conteo exploratorio, pero depende fuertemente del tamaño de la bola local. Un caso es más estable si $\Delta_{\mathrm{rel}}$ permanece positivo en los tres radios; es sensible al radio si solo aparece en una escala. Valores negativos indican más vecinos observados que los esperados bajo la referencia local. Ninguno debe interpretarse como número de planetas reales faltantes.

En una versión previa, la figura de déficit por radio podía confundirse porque el eje estaba rotulado como déficit relativo aunque la escala visual era compatible con déficit absoluto o con una columna no normalizada. En esta versión se separan explícitamente el déficit relativo y el déficit absoluto.

\input{__TOP_ANCHOR_DEFICIT_INPUT__}

\section{Déficit por radio}
\begin{figure}[H]\centering
\safeincludegraphics[width=.82\textwidth]{\figroot/deficit_relative_by_radius.pdf}
\caption{Déficit relativo por radio. Se grafica $\Delta_{\mathrm{rel}}=(N_{\mathrm{exp}}-N_{\mathrm{obs}})/(N_{\mathrm{exp}}+\epsilon)$ recomputado desde los conteos observados y esperados. Valores positivos indican menos vecinos observados que los esperados bajo la referencia local; valores negativos indican más vecinos observados que los esperados.}
\label{fig:deficit-relative-by-radius}
\end{figure}

\begin{figure}[H]\centering
\safeincludegraphics[width=.82\textwidth]{\figroot/deficit_absolute_by_radius.pdf}
\caption{Déficit absoluto por radio. Esta figura muestra $\Delta N=N_{\mathrm{exp}}-N_{\mathrm{obs}}$ y no debe confundirse con el déficit relativo.}
\label{fig:deficit-absolute-by-radius}
\end{figure}

\section{Tres casos finales para exposicion}
El caso regional no afirma planetas faltantes; prioriza una zona Mapper. El caso ancla no afirma un objeto ausente; prioriza una vecindad local. El caso repetido no es duplicacion erronea; puede ser una senal de transicion topologica por solapamiento de cubiertas.

Si la tabla por radio muestra que $\Delta_{\mathrm{rel}}$ es alto solo en una escala, el caso debe presentarse como exploratorio. Si el deficit se mantiene positivo en los tres radios, el caso es mas estable.

__FINAL_CASE_TABLE__

\begin{figure}[H]\centering
\safeincludegraphics[width=.95\textwidth]{\figroot/final_presentation_cases_summary.pdf}
\caption{Resumen de los tres casos finales para exposicion.}
\end{figure}

\section{Discusion}
Una region con TOI alto puede ganar por sombra fuerte, baja imputacion, continuidad fisica o soporte de red. Un ancla con ATI alto puede ganar por pertenecer a una region TOI alta, por mostrar deficit local, por baja imputacion o por ser representativa del nodo.

\section{Limitaciones}
Este modulo no introduce una funcion de completitud instrumental, no estima cantidades absolutas de objetos reales ausentes y no convierte por si solo un caso priorizado en una conclusion observacional cerrada.

\section{Conclusion}
Este modulo no calcula otro indice nuevo; abre los rankings TOI/ATI para explicar por que ciertas regiones y planetas ancla fueron priorizados. La salida principal es una anatomia de los casos top, no una afirmacion de objetos ausentes confirmados.
\end{document}
"""
    tex = tex.replace("__FINAL_CASE_TABLE__", final_case_table)
    tex = tex.replace("__TOP_ANCHOR_DEFICIT_INPUT__", top_anchor_deficit_input_path)
    forbidden = contains_forbidden_claim(tex)
    if forbidden:
        raise ValueError(f"LaTeX report contains forbidden claims: {forbidden}")
    path.write_text(tex.strip() + "\n", encoding="utf-8")


def _fmt(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _latex_radius(radius_name: object) -> str:
    mapping = {
        "r_kNN": r"$r_{\mathrm{kNN}}$",
        "r_node_median": r"$r_{\mathrm{node\_median}}$",
        "r_node_q75": r"$r_{\mathrm{node\_q75}}$",
    }
    return mapping.get(radius_name, _escape_latex(str(radius_name)))


def _stability_paragraph(label: str) -> str:
    if label == "consistent_positive_deficit":
        return "El deficit aparece en las tres escalas locales, por lo que la evidencia topologica de submuestreo es mas estable. Aun asi, no debe interpretarse como conteo absoluto de objetos reales ausentes."
    if label == "radius_sensitive_deficit":
        return "El deficit depende de la escala local elegida. Esto sugiere que el ancla es util para priorizacion, pero la evidencia es sensible al radio y debe considerarse exploratoria."
    if label == "no_consistent_deficit":
        return "No hay evidencia consistente de deficit local. El ancla puede seguir siendo relevante por vivir en una region TOI alta, pero no debe usarse como argumento fuerte de submuestreo local."
    return "No hay suficientes referencias locales para estimar deficit de forma estable."


def _latex_final_cases_table(final_cases: pd.DataFrame) -> str:
    if final_cases.empty:
        return "No hay casos finales disponibles."
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\caption{Tres casos finales para exposicion.}",
        "\\begin{tabular}{p{2.5cm}p{2.8cm}p{2.4cm}p{3.8cm}p{3.8cm}}",
        "\\toprule",
        "Caso & Objeto & Nodo & Motivo de seleccion & Cautela principal \\\\",
        "\\midrule",
    ]
    for _, row in final_cases.iterrows():
        obj = row.get("anchor_pl_name") if pd.notna(row.get("anchor_pl_name")) else row.get("selected_id")
        lines.append(
            f"{_escape_latex(str(row.get('case_type', '')))} & "
            f"{_escape_latex(str(obj or ''))} & "
            f"{_escape_latex(str(row.get('node_id', '')))} & "
            f"{_escape_latex(str(row.get('how_to_present', '')))} & "
            f"{_escape_latex(str(row.get('caution_text', '')))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = text
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped
