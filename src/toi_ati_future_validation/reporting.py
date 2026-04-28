from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

FORBIDDEN_CLAIMS = (
    "planetas faltantes confirmados",
    "descubrimos planetas",
    "faltan exactamente",
    "prediccion definitiva",
)


def contains_forbidden_claim(text: str) -> list[str]:
    lower = text.lower()
    return [phrase for phrase in FORBIDDEN_CLAIMS if phrase in lower]


def validate_prudent_text(text: str) -> None:
    forbidden = contains_forbidden_claim(text)
    if forbidden:
        raise ValueError(f"Forbidden claims found: {forbidden}")


def write_manifest(
    path: Path,
    *,
    config_path: str | None,
    config: dict,
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    warnings: Dict[str, str],
    summary_counts: Dict[str, object],
    commit_hash: str | None,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": commit_hash,
        "config_path": config_path,
        "config": config,
        "input_paths": input_paths,
        "output_paths": output_paths,
        "summary_counts": summary_counts,
        "warnings": warnings,
        "fallbacks": [warning for warning in warnings.values() if "fallback" in warning.lower()],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_summary(
    path: Path,
    *,
    regions: pd.DataFrame,
    anchors: pd.DataFrame,
    final_cases: pd.DataFrame,
    candidates: pd.DataFrame,
    technical_cases: pd.DataFrame,
) -> None:
    lines = [
        "# TOI/ATI future validation summary",
        "",
        "## Resumen ejecutivo",
        "Esta fase no crea otro indice principal. Toma TOI/ATI y los convierte en un sistema de priorizacion mas auditado al penalizar sensibilidad al radio y separar casos exploratorios de casos mas estables.",
        "",
        "## Que problema corrige esta fase",
        "- ATI original puede privilegiar un delta_rel_best alto aunque el promedio por radios sea bajo o negativo.",
        "- La nueva lectura usa estabilidad por radio, penalizacion conservadora e inspeccion de cambios de ranking.",
        "",
        "## Diferencia entre ATI original y ATI conservador",
        "- ATI original resume prioridad local con TOI, deficit best, baja imputacion y representatividad.",
        "- ATI conservador penaliza deficit sensible al radio y fragilidad en radios grandes.",
        "",
        "## Casos que bajan al penalizar sensibilidad al radio",
    ]
    if not anchors.empty:
        down = anchors.sort_values("rank_shift", ascending=False).head(5)
        lines.append(_markdown_table(down[[c for c in ["anchor_pl_name", "node_id", "ATI_original", "ATI_conservative", "rank_shift", "deficit_stability_class"] if c in down.columns]]))
    else:
        lines.append("No anchor ranking available.")
    lines.extend([
        "",
        "## Casos que suben o se mantienen por estabilidad",
    ])
    if not anchors.empty:
        up = anchors.sort_values(["ATI_conservative", "stable_deficit_score"], ascending=[False, False]).head(5)
        lines.append(_markdown_table(up[[c for c in ["anchor_pl_name", "node_id", "ATI_original", "ATI_conservative", "stable_deficit_score", "deficit_stability_class"] if c in up.columns]]))
    else:
        lines.append("No conservative ranking available.")
    lines.extend([
        "",
        "## Cinco casos finales recomendados",
    ])
    if not final_cases.empty:
        lines.append(_markdown_table(final_cases[[c for c in ["case_type", "anchor_pl_name", "node_id", "ATI_original", "ATI_conservative", "deficit_stability_class", "how_to_present"] if c in final_cases.columns]]))
    else:
        lines.append("No final future work cases available.")
    lines.extend([
        "",
        "## Tabla breve de prioridad observacional",
    ])
    if not candidates.empty:
        lines.append(_markdown_table(candidates.head(10)))
    else:
        lines.append("No observational priority candidates available.")
    lines.extend([
        "",
        "## Casos exploratorios",
        _markdown_text_from_filter(anchors, "deficit_stability_class", {"radius_sensitive_deficit", "unstable_due_to_large_radius"}),
        "",
        "## Casos robustos",
        _markdown_text_from_filter(anchors, "deficit_stability_class", {"stable_positive_deficit", "small_but_stable_deficit"}),
        "",
        "## Trabajo futuro",
        "- Incorporar completitud instrumental por metodo de descubrimiento.",
        "- Validar con catalogos futuros y con inyeccion-recuperacion sintetica.",
        "- Agregar propiedades estelares para mejorar el contexto fisico y el proxy de detectabilidad dinamica.",
        "",
        "## Advertencia de lenguaje prudente",
        "TOI/ATI no detecta planetas ausentes; prioriza regiones y anclas donde el catalogo parece observacionalmente incompleto bajo una referencia topologica local.",
    ])
    if not technical_cases.empty:
        lines.extend(["", "## Casos para auditoria tecnica", _markdown_table(technical_cases.head(10))])
    text = "\n".join(lines)
    validate_prudent_text(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_latex_report(path: Path) -> None:
    tex = r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-nodecimaldot]{babel}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,float,hyperref}
\newcommand{\figroot}{../../outputs/toi_ati_future_validation/figures_pdf}
\title{Validacion futura de TOI/ATI}
\author{Proyecto Mapper/TDA de exoplanetas}
\date{}
\begin{document}
\maketitle

\section{Introduccion}
Esta fase no crea otro indice principal. Su objetivo es validar y estabilizar TOI/ATI para convertir un ranking exploratorio en una priorizacion mas auditada.

\section{Motivacion}
ATI original puede priorizar anclas sensibles al radio porque $\Delta_{\mathrm{rel,best}}$ resume el maximo local. La fase presente separa casos estables de casos sensibles y construye una version conservadora del ranking.

\section{Datos usados}
La validacion consume tablas de TOI/ATI, deficit por radio y anatomia previa, sin recalcular Mapper ni introducir un nuevo pipeline de completitud instrumental.

\section{Estabilidad del deficit por radio}
Se comparan tres escalas locales: $r_{\mathrm{kNN}}$, $r_{\mathrm{node\_median}}$ y $r_{\mathrm{node\_q75}}$. Un caso puede ser \textit{stable\_positive\_deficit}, \textit{radius\_sensitive\_deficit}, \textit{unstable\_due\_to\_large\_radius} o \textit{no\_deficit\_or\_overpopulated}.

\section{ATI conservador}
Se construyen variantes interpretativas: ATI\_stable\_simple, ATI\_radius\_penalized y ATI\_conservative. La version conservadora penaliza sensibilidad al radio y radios grandes negativos.

\section{Resultados: casos estables vs sensibles}
\begin{figure}[H]\centering
\includegraphics[width=.82\textwidth]{\figroot/stable_vs_sensitive_deficit.pdf}
\caption{Casos estables vs sensibles al radio.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=.82\textwidth]{\figroot/deficit_profiles_selected_anchors.pdf}
\caption{Perfiles de deficit relativo por radio para anclas seleccionadas.}
\end{figure}

\section{Resultados: cambio de ranking}
\begin{figure}[H]\centering
\includegraphics[width=.82\textwidth]{\figroot/ati_vs_ati_conservative.pdf}
\caption{ATI original vs ATI conservador.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=.82\textwidth]{\figroot/rank_shift_after_stability_penalty.pdf}
\caption{Cambio de ranking tras penalizar sensibilidad al radio.}
\end{figure}

\section{Casos finales para trabajo futuro}
La tabla \texttt{final\_future\_work\_cases.csv} resume cinco casos: region top por TOI, ancla top por ATI original, ancla top por ATI conservador, ancla repetida y ancla con deficit estable.

\begin{figure}[H]\centering
\includegraphics[width=.9\textwidth]{\figroot/final_future_work_cases.pdf}
\caption{Casos finales para trabajo futuro.}
\end{figure}

\section{Prioridad observacional}
La prioridad observacional no implica confirmaciones de objetos ausentes. Es una priorizacion prudente para inspeccion futura bajo referencias topologicas locales.

\begin{figure}[H]\centering
\includegraphics[width=.82\textwidth]{\figroot/observational_priority_ranking.pdf}
\caption{Ranking de prioridad observacional futura.}
\end{figure}

\section{Discusion}
Esta fase fortalece algunos casos estables y vuelve mas fragiles otros. Si un caso como HIP 97166 c baja al penalizar sensibilidad al radio, eso indica que su lectura depende de la escala local. Si HIP 90988 b o HD 4313 b suben por estabilidad, eso sugiere una prioridad mas defendible para inspeccion futura.

\section{Limitaciones}
No hay funcion de completitud instrumental real, $R^3$ simplifica la fisica, el proxy RV no es amplitud real, Mapper solapa membresias y el deficit depende de la referencia local. Los rankings son herramientas de priorizacion, no deteccion.

\section{Trabajo futuro}
\subsection{Incorporar funciones de completitud instrumental}
Modelar $P(\mathrm{detectado}\mid M_p, P, a, estrella, metodo)$ y usar curvas de sensibilidad por metodo.

\subsection{Validar con catalogos futuros}
Correr el pipeline con versiones futuras del catalogo y observar si regiones TOI altas reciben nuevos planetas o mediciones mas finas.

\subsection{Inyeccion-recuperacion sintetica}
Insertar planetas sinteticos en $R^3$, simular seleccion por metodo y evaluar si TOI/ATI recupera regiones incompletas.

\subsection{Propiedades estelares}
Agregar masa, radio, temperatura, magnitud y distancia para mejorar el contexto fisico y la detectabilidad dinamica.

\subsection{Comparacion por metodo de descubrimiento}
Construir rankings separados para RV, transito, imaging y microlensing.

\subsection{Reporte de candidatos observacionales}
Producir fichas para top 5 candidatos y separar casos robustos de casos exploratorios.

\section{Conclusion}
La validacion futura convierte TOI/ATI de un ranking exploratorio en un sistema de priorizacion auditado: no detecta planetas ausentes, pero identifica donde el catalogo parece mas informativo para buscar incompletitud observacional.
\end{document}
"""
    validate_prudent_text(tex)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex.strip() + "\n", encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = ["" if pd.isna(value) else str(value) for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _markdown_text_from_filter(df: pd.DataFrame, column: str, labels: set[str]) -> str:
    if df.empty or column not in df.columns:
        return "No hay casos disponibles."
    subset = df[df[column].astype(str).isin(labels)]
    if subset.empty:
        return "No hay casos disponibles."
    return _markdown_table(subset[[c for c in ["anchor_pl_name", "node_id", column, "ATI_conservative", "caution_text"] if c in subset.columns]].head(10))
