from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .validation import contains_forbidden_claim


def write_manifest(path: Path, *, config_path: str | None, inputs_loaded: Dict[str, int], warnings: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "inputs_loaded_rows": inputs_loaded,
        "warnings": warnings,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_summary(
    path: Path,
    *,
    sentences: Iterable[str],
    top_regions: pd.DataFrame,
    top_anchors: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# TOI/ATI case anatomy summary", ""]
    lines.extend(f"- {s}" for s in sentences)
    lines.append("")
    lines.append("## Top regions")
    if not top_regions.empty:
        cols = [c for c in ["node_id", "TOI", "shadow_score", "I_R3", "C_phys", "S_net", "top_method"] if c in top_regions.columns]
        lines.append(top_regions[cols].head(10).to_markdown(index=False))
    else:
        lines.append("No top regions available.")
    lines.append("")
    lines.append("## Top anchors")
    if not top_anchors.empty:
        cols = [c for c in ["anchor_pl_name", "node_id", "ATI", "TOI", "delta_rel_neighbors_best", "deficit_class"] if c in top_anchors.columns]
        lines.append(top_anchors[cols].head(10).to_markdown(index=False))
    else:
        lines.append("No top anchors available.")
    lines.append("")
    lines.append("## Caution")
    lines.append("TOI y ATI priorizan regiones y anclas; no prueban planetas faltantes confirmados.")
    text = "\n".join(lines)
    forbidden = contains_forbidden_claim(text)
    if forbidden:
        raise ValueError(f"Summary contains forbidden claims: {forbidden}")
    path.write_text(text, encoding="utf-8")


def write_latex_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tex = r'''
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
\title{Anatomía de casos TOI/ATI: regiones Mapper y planetas ancla}
\author{Proyecto Mapper/TDA de exoplanetas}
\date{}
\begin{document}
\maketitle
\section{Objetivo}
Este reporte abre el ranking global TOI/ATI para explicar por qué ciertas regiones Mapper y planetas ancla fueron priorizados. El objetivo no es afirmar descubrimientos de planetas, sino auditar la anatomía de los índices.

\section{Índice regional}
\[
\mathrm{TOI}(v)=\mathrm{Shadow}(v)(1-I_{R^3}(v))C_{\mathrm{phys}}(v)S_{\mathrm{net}}(v).
\]
TOI debe leerse como una prioridad regional que combina frontera observacional, baja imputación, continuidad física con vecinos y soporte de red.

\section{Índice de ancla}
\[
\mathrm{ATI}(p^*)=\mathrm{TOI}(v)\Delta_{\mathrm{rel,best}}(p^*)(1-I_{R^3}(p^*))A(p^*).
\]
ATI debe leerse como una prioridad local de inspección. El término \(\Delta_{\mathrm{rel,best}}\) debe revisarse junto con los radios individuales.

\section{Figuras principales}
\begin{figure}[H]\centering
\safeincludegraphics[width=.9\textwidth]{\figroot/top_regions_toi_case_anatomy.pdf}
\caption{Top regiones por TOI.}
\end{figure}

\begin{figure}[H]\centering
\safeincludegraphics[width=.9\textwidth]{\figroot/top_anchors_ati_case_anatomy.pdf}
\caption{Top planetas ancla por ATI.}
\end{figure}

\begin{figure}[H]\centering
\safeincludegraphics[width=.9\textwidth]{\figroot/toi_factor_decomposition.pdf}
\caption{Descomposición visual de factores TOI.}
\end{figure}

\begin{figure}[H]\centering
\safeincludegraphics[width=.9\textwidth]{\figroot/ati_factor_decomposition.pdf}
\caption{Descomposición visual de factores ATI.}
\end{figure}

\section{Interpretación esperada}
El reporte debe explicar qué factor impulsa cada caso y qué factor limita su interpretación. Una región con TOI alto puede ganar por sombra fuerte, baja imputación, continuidad física o soporte de red. Un ancla con ATI alto puede ganar por pertenecer a una región TOI alta, por mostrar déficit local, por baja imputación o por ser representativa del nodo.

\section{Conclusión}
TOI/ATI formalizan un ranking topológico de incompletitud observacional. La lectura correcta es priorización de regiones y anclas, no conteo absoluto de objetos ausentes.
\end{document}
'''
    path.write_text(tex.strip() + "\n", encoding="utf-8")
