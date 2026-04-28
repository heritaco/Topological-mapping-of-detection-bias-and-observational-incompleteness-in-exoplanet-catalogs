from __future__ import annotations

from pathlib import Path

import pandas as pd


def _case_section(case_row: pd.Series) -> str:
    node_id = str(case_row["node_id"])
    anchor = str(case_row.get("anchor_pl_name", "Unknown"))
    return rf"""
\section{{Caso {node_id}}}
\subsection{{Ficha tecnica}}
Nodo objetivo: \texttt{{{node_id}}}. Metodo dominante: \texttt{{{case_row.get('top_method', 'Unknown')}}}. Planeta ancla: \texttt{{{anchor}}}.

\begin{{figure}}[ht]
\centering
\maybeincludegraphics[width=0.72\linewidth]{{../../outputs/local_shadow_case_studies/figures/case_{node_id}_ego_network.pdf}}
\caption{{Ego-network local de \texttt{{{node_id}}}.}}
\end{{figure}}

\begin{{figure}}[ht]
\centering
\maybeincludegraphics[width=0.95\linewidth]{{../../outputs/local_shadow_case_studies/figures/case_{node_id}_r3_projections.pdf}}
\caption{{Proyecciones 2D del espacio $R^3=(\log M_p,\log P,\log a)$ para \texttt{{{node_id}}}.}}
\end{{figure}}

\begin{{figure}}[ht]
\centering
\maybeincludegraphics[width=0.72\linewidth]{{../../outputs/local_shadow_case_studies/figures/case_{node_id}_method_node_vs_neighbors.pdf}}
\caption{{Composicion por metodo del nodo y su vecindario topologico.}}
\end{{figure}}

\begin{{figure}}[ht]
\centering
\maybeincludegraphics[width=0.72\linewidth]{{../../outputs/local_shadow_case_studies/figures/case_{node_id}_imputation_r3_audit.pdf}}
\caption{{Auditoria de observacion, derivacion fisica e imputacion en las variables de $R^3$.}}
\end{{figure}}

\begin{{figure}}[ht]
\centering
\maybeincludegraphics[width=0.72\linewidth]{{../../outputs/local_shadow_case_studies/figures/case_{node_id}_anchor_deficit.pdf}}
\caption{{Conteos observados y referencias locales de vecinos compatibles alrededor del planeta ancla.}}
\end{{figure}}

\begin{{figure}}[ht]
\centering
\maybeincludegraphics[width=0.72\linewidth]{{../../outputs/local_shadow_case_studies/figures/case_{node_id}_rv_proxy_distribution.pdf}}
\caption{{Proxy de detectabilidad RV para el nodo, sus vecinos y el ancla.}}
\end{{figure}}

\paragraph{{Interpretacion.}}
{case_row.get('interpretation_text', 'Interpretacion no disponible.')}
"""


def write_latex_report(latex_dir: Path, case_summary: pd.DataFrame, requested_nodes: list[str], analyzed_nodes: list[str], replacements: list[dict[str, str]]) -> None:
    latex_dir.mkdir(parents=True, exist_ok=True)
    replacement_lines = "\n".join(
        [rf"\item \texttt{{{item['requested_node_id']}}} $\rightarrow$ \texttt{{{item['replacement_node_id']}}}: {item['reason']}" for item in replacements]
    )
    if not replacement_lines:
        replacement_lines = r"\item No hubo reemplazos; los tres nodos propuestos existian y fueron analizados."
    case_sections = "\n".join(_case_section(row) for _, row in case_summary.iterrows())
    content = r"""\documentclass[11pt]{article}
\usepackage[spanish]{{babel}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{array}}
\usepackage{{hyperref}}
\geometry{{margin=2.5cm}}

\newcommand{{\maybeincludegraphics}}[2][]{%
  \IfFileExists{{#2}}{{\includegraphics[#1]{{#2}}}}{{\fbox{{\parbox{{0.88\linewidth}}{{Figura no disponible: \texttt{{#2}}}}}}}%
}

\title{{Estudio local de comunidades RV de alta sombra observacional en el Mapper orbital}}
\author{{Proyecto Mapper/TDA de exoplanetas}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Introduccion}}
El analisis global de sombra observacional detecto comunidades Mapper con contraste fuerte frente a su vecindario. Este reporte baja a tres casos concretos para construir una ficha local de incompletitud topologica sin afirmar descubrimientos definitivos.

\section{{Pregunta local}}
La pregunta central es: \emph{{Puede un exoplaneta observado funcionar como ancla de una vecindad fisico-orbital submuestreada?}}

\section{{Por que velocidad radial}}
Los top candidatos de sombra en el espacio orbital estan dominados por velocidad radial. Ese metodo favorece senales dinamicas mas fuertes, por lo que una posible incompletitud local se espera prudentemente hacia menor masa planetaria o menor proxy de detectabilidad RV.

\section{{Casos seleccionados}}
Nodos solicitados: <<REQUESTED_NODES>>. Nodos analizados: <<ANALYZED_NODES>>.
\begin{{itemize}}
<<REPLACEMENT_LINES>>
\end{{itemize}}

\section{{Espacio R3}}
Se define
\[
x_i = (\log_{10} M_p,\log_{10} P,\log_{10} a).
\]
Estas coordenadas condensan masa, periodo y escala orbital, todas relevantes para una lectura prudente del sesgo RV.

\section{{Metodologia}}
Se usan vecindarios Mapper de primer y segundo orden, metricas de red, composicion por metodo, auditoria especifica de imputacion en $R^3$, seleccion automatica de planeta ancla y un deficit local de vecinos compatibles:
\[
N_1(v)=\{u:(u,v)\in E\},
\]
\[
B(v)=\sum_m |p_v(m)-p_{N_1}(m)|,
\]
\[
I_{R^3}(v)=\frac{\# \text{ valores imputados en } M_p,P,a}{3|S_v|},
\]
\[
N_{obs}(p^*,r)=|\{q:\lVert x(q)-x(p^*)\rVert \le r\}|,
\]
\[
\Delta_N=N_{exp}-N_{obs}, \qquad
\Delta_{rel}=\frac{N_{exp}-N_{obs}}{N_{exp}+\epsilon}.
\]

<<CASE_SECTIONS>>

\section{{Comparacion entre casos}}
\begin{{figure}}[ht]
\centering
\maybeincludegraphics[width=0.68\linewidth]{{../../outputs/local_shadow_case_studies/figures/three_case_shadow_vs_physical_distance.pdf}}
\caption{{Sombra observacional frente a distancia fisica al vecindario en los tres casos.}}
\end{{figure}}

\begin{{figure}}[ht]
\centering
\maybeincludegraphics[width=0.72\linewidth]{{../../outputs/local_shadow_case_studies/figures/three_case_deficit_comparison.pdf}}
\caption{{Comparacion de deficit relativo local entre casos.}}
\end{{figure}}

\begin{{figure}}[ht]
\centering
\maybeincludegraphics[width=0.78\linewidth]{{../../outputs/local_shadow_case_studies/figures/three_case_confidence_matrix.pdf}}
\caption{{Matriz visual de sombra, imputacion, frontera metodologica, distancia fisica y deficit relativo.}}
\end{{figure}}

\section{{Discusion}}
Si se puede afirmar que algunos nodos RV concentran una vecindad fisico-orbital plausible con fuerte contraste topologico y baja imputacion en las variables clave. No se puede afirmar que descubrimos planetas ni que hay un numero absoluto de exoplanetas reales ausentes.

\section{{Limitaciones}}
No se modela una funcion de completitud instrumental. $N_{exp}$ es una referencia local, no un conteo absoluto de planetas reales. El espacio $R^3$ simplifica la fisica, el proxy RV no es amplitud fisica exacta, los nodos pequenos pueden inflar pureza y sombra, y la imputacion de masa puede debilitar conclusiones.

\section{{Conclusion}}
Un cluster RV de alta sombra puede interpretarse como una vecindad fisica plausible vista a traves de una ventana observacional incompleta. El exoplaneta ancla no prueba planetas faltantes, pero si localiza un deficit topologico de vecinos compatibles.

\section{{Trabajo futuro}}
Incorporar funciones de completitud, usar inyeccion-recuperacion, agregar propiedades estelares, validar contra datos de misiones especificas y comparar con catalogos futuros.

\end{{document}}
"""
    content = (
        content.replace("<<REQUESTED_NODES>>", ", ".join([rf"\texttt{{{value}}}" for value in requested_nodes]))
        .replace("<<ANALYZED_NODES>>", ", ".join([rf"\texttt{{{value}}}" for value in analyzed_nodes]))
        .replace("<<REPLACEMENT_LINES>>", replacement_lines)
        .replace("<<CASE_SECTIONS>>", case_sections)
        .replace("{{", "{")
        .replace("}}", "}")
    )
    (latex_dir / "main.tex").write_text(content, encoding="utf-8")
