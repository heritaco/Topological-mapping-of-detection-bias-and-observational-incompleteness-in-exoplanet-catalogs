from __future__ import annotations

from pathlib import Path

import pandas as pd


BANNED_PHRASES = [
    "planetas faltantes confirmados",
    "descubrimos planetas",
    "faltan exactamente",
    "predicción definitiva",
]


def validate_prudent_text(text: str) -> None:
    lower = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lower:
            raise AssertionError(f"Frase prohibida detectada: {phrase}")


def build_interpretation_summary(regions: pd.DataFrame, anchors: pd.DataFrame, summary: pd.DataFrame, deficits: pd.DataFrame) -> str:
    top_regions = regions.sort_values("TOI", ascending=False).head(5) if not regions.empty else pd.DataFrame()
    top_anchors = anchors.sort_values("ATI", ascending=False).head(5) if not anchors.empty else pd.DataFrame()
    moderate_or_strong = anchors[anchors["deficit_class"].astype(str).isin(["moderate_deficit", "strong_deficit"])].copy() if not anchors.empty else pd.DataFrame()
    high_toi_methods = regions.loc[regions["region_class"].astype(str) == "high_toi_region", "top_method"].astype(str).value_counts() if not regions.empty else pd.Series(dtype=int)
    lines = [
        "# Topological Incompleteness Index",
        "",
        "## Resumen ejecutivo",
        "TOI y ATI construyen un ranking topologico de regiones Mapper y planetas ancla donde el catalogo parece observacionalmente incompleto bajo referencias locales prudentes.",
        "",
        "## Que es TOI",
        "TOI resume sombra observacional, imputacion especifica en R^3, continuidad fisica con el vecindario y soporte de red.",
        "",
        "## Que es ATI",
        "ATI combina TOI con el deficit relativo local del planeta ancla, su trazabilidad en R^3 y su representatividad dentro del nodo.",
        "",
        "## Top 5 regiones",
    ]
    if top_regions.empty:
        lines.append("No se identificaron regiones con TOI evaluable.")
    else:
        for _, row in top_regions.iterrows():
            lines.append(
                f"- {row['node_id']}: TOI={float(row['TOI']):.3f}, shadow={float(row['shadow_score']):.3f}, "
                f"clase={row['region_class']}, metodo={row.get('top_method', 'Unknown')}."
            )
    lines.extend(["", "## Top 5 planetas ancla"])
    if top_anchors.empty:
        lines.append("No se identificaron anclas con ATI evaluable.")
    else:
        for _, row in top_anchors.iterrows():
            lines.append(
                f"- {row['anchor_pl_name']} / {row['node_id']}: ATI={float(row['ATI']):.3f}, "
                f"deficit_best={float(row['delta_rel_neighbors_best']):.3f}, clase={row['deficit_class']}."
            )
    lines.extend(["", "## Regiones o anclas con deficit moderado/fuerte"])
    if moderate_or_strong.empty:
        lines.append("En esta corrida no aparecieron anclas con deficit moderado o fuerte bajo la referencia vecinal principal.")
    else:
        for _, row in moderate_or_strong.iterrows():
            lines.append(f"- {row['anchor_pl_name']} en {row['node_id']}: {row['deficit_class']} con delta_rel_best={float(row['delta_rel_neighbors_best']):.3f}.")
    lines.extend(
        [
            "",
            "## Si la mayoria son RV",
            f"Metodos dominantes mas frecuentes entre regiones high TOI: {', '.join([f'{k}={v}' for k, v in high_toi_methods.head(5).items()]) if not high_toi_methods.empty else 'sin predominio fuerte'}. "
            "Si predomina Radial Velocity, la direccion esperada de incompletitud se interpreta prudentemente hacia menor masa planetaria o menor proxy RV a escala orbital comparable.",
            "",
            "## Advertencia sobre delta_rel_neighbors_best",
            "El valor best es util para priorizacion, pero puede inflar la lectura si se interpreta aislado. Conviene revisarlo junto con el promedio, la mediana y el detalle por radio en anchor_neighbor_deficits.csv.",
            "",
            "## Advertencia general",
            "Estos resultados son rankings de submuestreo topologico y priorizacion observacional. No equivalen a una conclusion cerrada sobre objetos ausentes.",
            "",
            "Frase final para presentacion: TOI y ATI no descubren planetas faltantes; construyen un ranking topologico de regiones y planetas ancla donde el catalogo parece observacionalmente incompleto.",
        ]
    )
    text = "\n".join(lines) + "\n"
    validate_prudent_text(text)
    return text


def write_latex_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = r"""\documentclass[11pt]{article}
\usepackage[spanish]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\geometry{margin=2.5cm}

\newcommand{\maybeincludegraphics}[2][]{%
  \IfFileExists{#2}{\includegraphics[#1]{#2}}{\fbox{\parbox{0.88\linewidth}{Figura no disponible: \texttt{#2}}}}%
}

\title{Topological Incompleteness Index}
\author{Proyecto Mapper/TDA de exoplanetas}
\date{\today}

\begin{document}
\maketitle

\section{Introduccion}
Este subproyecto formaliza un indice topologico de incompletitud observacional para priorizar regiones Mapper y planetas ancla bajo lenguaje prudente.

\section{Motivacion}
El indice conecta sesgo de descubrimiento, sombra observacional, deficit local y planetas ancla como una extension metodologica del pipeline existente.

\section{Definicion de R3}
\[
x_i = (\log_{10} M_p, \log_{10} P, \log_{10} a).
\]

\section{Indice regional TOI}
\[
\mathrm{TOI}(v)=\mathrm{Shadow}(v)\,(1-I_{R^3}(v))\,C_{\mathrm{phys}}(v)\,S_{\mathrm{net}}(v).
\]

\section{Indice de ancla ATI}
\[
\mathrm{ATI}(p^*)=\mathrm{TOI}(v)\,\Delta_{\mathrm{rel,best}}(p^*)\,(1-I_{R^3}(p^*))\,A(p^*).
\]

\section{Deficit relativo de vecinos}
Se usa
\[
N_{\mathrm{obs}}(p^*,r), \qquad N_{\mathrm{exp}}(p^*,r), \qquad
\Delta_{\mathrm{rel}}(p^*,r)=\frac{N_{\mathrm{exp}}-N_{\mathrm{obs}}}{N_{\mathrm{exp}}+\epsilon}.
\]
El valor $\Delta_{\mathrm{rel,best}}$ ayuda a priorizar, pero debe interpretarse con cautela junto con el detalle por radio.

\section{Resultados}
\begin{figure}[ht]
\centering
\maybeincludegraphics[width=0.82\linewidth]{../../outputs/topological_incompleteness_index/figures_pdf/top_regions_toi_score.pdf}
\caption{Top 20 regiones Mapper por TOI.}
\end{figure}

\begin{figure}[ht]
\centering
\maybeincludegraphics[width=0.82\linewidth]{../../outputs/topological_incompleteness_index/figures_pdf/top_anchor_ati_score.pdf}
\caption{Top 20 planetas ancla por ATI.}
\end{figure}

\begin{figure}[ht]
\centering
\maybeincludegraphics[width=0.72\linewidth]{../../outputs/topological_incompleteness_index/figures_pdf/toi_vs_shadow_score.pdf}
\caption{Relacion entre shadow score y TOI.}
\end{figure}

\section{Top regiones}
Las regiones con TOI alto representan candidatas a submuestreo topologico bajo continuidad fisica, baja imputacion en $R^3$ y soporte de red suficiente.

\section{Top planetas ancla}
Los anclajes con ATI alto sirven para priorizacion observacional local, no como evidencia cerrada de objetos ausentes.

\section{Discusion}
Si se puede afirmar que TOI y ATI ordenan regiones y anclas donde el catalogo parece submuestreado bajo una referencia topologica. No se puede afirmar una cantidad absoluta de objetos reales ausentes.

\section{Limitaciones}
No hay funcion de completitud instrumental, $N_{\mathrm{exp}}$ es una referencia local, $\Delta_{\mathrm{rel,best}}$ puede inflar la lectura, $R^3$ simplifica la fisica, el proxy RV no es amplitud RV real, la imputacion puede afectar masa y los nodos pequenos pueden inflar indices.

\section{Conclusion}
TOI y ATI no detectan planetas faltantes, pero priorizan regiones y planetas ancla donde el catalogo parece topologicamente submuestreado.

\section{Trabajo futuro}
Completitud instrumental, inyeccion-recuperacion, validacion con catalogos futuros, integracion de propiedades estelares y comparacion entre metodos de descubrimiento.

\end{document}
"""
    path.write_text(content, encoding="utf-8")
