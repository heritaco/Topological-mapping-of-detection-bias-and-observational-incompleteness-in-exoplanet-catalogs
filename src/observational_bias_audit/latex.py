from __future__ import annotations

from pathlib import Path

from .paths import PROJECT_ROOT


def _relative(target: Path, start: Path) -> str:
    return Path(__import__("os").path.relpath(target.resolve(), start.resolve())).as_posix()


def write_latex_report(
    latex_dir: Path,
    outputs_dir: Path,
    interpretation_text: str,
) -> Path:
    latex_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = outputs_dir / "figures"
    tables_dir = outputs_dir / "tables"
    main_tex = latex_dir / "main.tex"
    content = fr"""% Compile with:
% pdflatex main.tex
% pdflatex main.tex
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{amsmath}}
\\usepackage{{hyperref}}
\\usepackage{{float}}
\\title{{Auditor\'ia observacional del Mapper orbital en el cat\'alogo de exoplanetas}}
\\date{{}}
\\begin{{document}}
\\maketitle

\\section{{Introducci\'on}}
Este subproyecto audita si la topolog\'ia observada en Mapper, en particular en el espacio orbital, est\'a asociada al m\'etodo de descubrimiento de los planetas y no necesariamente a clases f\'isicas intr\'insecas.

\\section{{Hip\'otesis}}
Hip\'otesis A: la estructura topol\'ogica que aparece en Mapper no est\'a determinada principalmente por clases f\'isicas planetarias, sino por sesgos de observaci\'on y m\'etodo de descubrimiento.

\\section{{Datos y configuraci\'on analizada}}
El an\'alisis principal usa la configuraci\'on \\texttt{{orbital\\_pca2\\_cubes10\\_overlap0p35}} ya construida. Cuando hay artifacts disponibles, tambi\'en se resumen \\texttt{{phys\\_min\\_pca2\\_cubes10\\_overlap0p35}}, \\texttt{{joint\\_no\\_density\\_pca2\\_cubes10\\_overlap0p35}}, \\texttt{{joint\\_pca2\\_cubes10\\_overlap0p35}} y \\texttt{{thermal\\_pca2\\_cubes10\\_overlap0p35}}.

\\section{{Metodolog\'ia}}
Para cada nodo $v$ y m\'etodo $m$ se calculan:
\\[
p_v(m) = \\frac{{\\mathrm{{count}}_v(m)}}{{n_v}}
\\]
\\[
\\mathrm{{purity}}(v) = \\max_m p_v(m)
\\]
\\[
H(v) = -\\sum_m p_v(m) \\log p_v(m)
\\]
\\[
H_{{norm}}(v) = \\frac{{H(v)}}{{\\log(K)}}
\\]
donde $K$ es el n\'umero de m\'etodos presentes en el universo analizado.

La prueba nula mantiene fijo el grafo y la membres\'ia nodo-planeta, permuta \\texttt{{discoverymethod}} entre planetas y recalcula m\'etricas globales. El z-score reportado sigue:
\\[
z = \\frac{{\\mathrm{{observed}} - \\mathrm{{null\\_mean}}}}{{\\mathrm{{null\\_std}}}}
\\]

\\section{{M\'etricas nodo-m\'etodo}}
Se generan tablas de pureza, entrop\'ia, m\'etodo dominante, tama\~no nodal, fracci\'on de imputaci\'on media, fracci\'on de variables f\'isicamente derivadas, grado, componente e indicador de periferia.

\\section{{Prueba nula por permutaci\'on}}
Las m\'etricas globales contrastadas contra la distribuci\'on nula son pureza media ponderada, entrop\'ia media ponderada, informaci\'on mutua normalizada entre incidencias nodo-m\'etodo y assortativity del m\'etodo dominante entre nodos conectados cuando existen aristas.

\\section{{Resultados}}
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{{{_relative(figures_dir / "orbital_graph_by_dominant_discovery_method.pdf", latex_dir)}}}
\\caption{{Grafo Mapper orbital coloreado por m\'etodo de descubrimiento dominante.}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{{{_relative(figures_dir / "orbital_graph_by_method_purity.pdf", latex_dir)}}}
\\caption{{Pureza nodal por m\'etodo de descubrimiento.}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{{{_relative(figures_dir / "orbital_graph_by_method_entropy.pdf", latex_dir)}}}
\\caption{{Entrop\'ia normalizada nodo-m\'etodo en el Mapper orbital.}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{{{_relative(figures_dir / "orbital_node_method_fraction_heatmap.pdf", latex_dir)}}}
\\caption{{Heatmap nodo x m\'etodo para el espacio orbital.}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\textwidth]{{{_relative(figures_dir / "orbital_permutation_null_nmi.pdf", latex_dir)}}}
\\caption{{Distribuci\'on nula de NMI y valor observado.}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\textwidth]{{{_relative(figures_dir / "orbital_purity_vs_imputation.pdf", latex_dir)}}}
\\caption{{Pureza nodal frente a fracci\'on media de imputaci\'on.}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\textwidth]{{{_relative(figures_dir / "orbital_purity_vs_node_size.pdf", latex_dir)}}}
\\caption{{Pureza nodal frente a tama\~no de nodo.}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{{{_relative(figures_dir / "orbital_component_method_composition.pdf", latex_dir)}}}
\\caption{{Composici\'on por m\'etodo de descubrimiento a nivel de componente.}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.95\\textwidth]{{{_relative(figures_dir / "config_comparison_bias_metrics.pdf", latex_dir)}}}
\\caption{{Comparaci\'on resumida de m\'etricas de sesgo observacional entre configuraciones disponibles.}}
\\end{{figure}}

\\paragraph{{Interpretaci\'on autom\'atica.}} {interpretation_text}

\\section{{Discusi\'on}}
Los resultados deben interpretarse como una auditor\'ia de sesgo observacional sobre grafos ya construidos. No prueban por s\'i mismos la existencia de clases planetarias ni reemplazan una v\'ia contrafactual.

\\section{{Limitaciones}}
Este an\'alisis depende de la calidad de la metadata observacional, del esquema de imputaci\'on ya fijado en el pipeline original y de membres\'ias nodales con solapamiento propio de Mapper. Si nodos de alta pureza tambi\'en tienen alta imputaci\'on, la separaci\'on entre sesgo observacional y dependencia de imputaci\'on requiere an\'alisis adicional.

\\section{{Conclusi\'on}}
El subproyecto deja cuantificada la asociaci\'on entre topolog\'ia Mapper y m\'etodo de descubrimiento en t\'erminos de pureza, entrop\'ia, enriquecimiento local y contraste por permutaciones.

\\appendix
\\section{{Ap\'endice de tablas}}
\\input{{{_relative(tables_dir / "summary_global_bias_metrics.tex", latex_dir)}}}
\\input{{{_relative(tables_dir / "top_enriched_nodes.tex", latex_dir)}}}
\\input{{{_relative(tables_dir / "central_vs_peripheral_bias.tex", latex_dir)}}}

\\end{{document}}
"""
    main_tex.write_text(content, encoding="utf-8")
    readme = latex_dir / "README.md"
    readme.write_text("Compilacion esperada:\n\npdflatex main.tex\npdflatex main.tex\n", encoding="utf-8")
    return main_tex
