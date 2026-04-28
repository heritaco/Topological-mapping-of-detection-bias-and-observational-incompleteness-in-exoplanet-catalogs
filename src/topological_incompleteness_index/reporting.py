from __future__ import annotations
from pathlib import Path
import pandas as pd

def write_interpretation_summary(regions: pd.DataFrame, anchors: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append("# Topological Observational Incompleteness Index\\n\\n")
    lines.append("This output ranks Mapper regions and anchor planets as candidates for observational incompleteness.\\n\\n")
    lines.append("It does **not** prove missing exoplanets. It prioritizes regions where future observations may be informative.\\n\\n")
    if not regions.empty:
        r = regions.iloc[0]
        lines.append("## Highest TOI region\\n\\n")
        lines.append(f"- Node: `{r.get('node_id')}`\\n")
        lines.append(f"- TOI score: `{r.get('toi_score'):.4f}`\\n")
        lines.append(f"- Shadow score: `{r.get('shadow_score', float('nan'))}`\\n")
        lines.append(f"- Method: `{r.get('top_method', 'unknown')}`\\n\\n")
    if not anchors.empty:
        a = anchors.sort_values("ati_score", ascending=False).iloc[0]
        lines.append("## Highest ATI anchor\\n\\n")
        lines.append(f"- Planet: `{a.get('anchor_pl_name')}`\\n")
        lines.append(f"- Node: `{a.get('node_id')}`\\n")
        lines.append(f"- ATI score: `{a.get('ati_score'):.4f}`\\n")
        lines.append(f"- Best relative neighbor deficit: `{a.get('delta_rel_neighbors_best'):.4f}`\\n")
        lines.append("- Interpretation: local topological deficit candidate, not a confirmed missing planet.\\n\\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")

def write_latex_template(latex_dir: Path) -> None:
    latex_dir.mkdir(parents=True, exist_ok=True)
    tex = r"""
\\documentclass[11pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{amsmath}
\\usepackage{float}
\\newcommand{\\figroot}{../../outputs/topological_incompleteness_index/figures_pdf}
\\title{Topological Observational Incompleteness Index for Exoplanet Mapper Regions}
\\author{Proyecto Mapper/TDA de exoplanetas}
\\date{}
\\begin{document}
\\maketitle

\\section{Purpose}
This report introduces a regional index, TOI, and an anchor-planet index, ATI, to prioritize Mapper regions where observational bias may imply local incompleteness. The result is not a claim of confirmed missing exoplanets. It is a ranking of topological regions and anchor planets where future observations may be informative.

\\section{Definitions}
For a Mapper node $v$ and an anchor planet $p^* \\in v$, define
\\[
  x_i=(\\log_{10} M_{p,i},\\log_{10}P_i,\\log_{10}a_i)\\in\\mathbb{R}^3.
\\]
The regional index is
\\[
\\mathrm{TOI}(v)=\\mathrm{Shadow}(v)(1-I_{R^3}(v))C_{\\mathrm{phys}}(v)S_{\\mathrm{net}}(v).
\\]
The anchor index is
\\[
\\mathrm{ATI}(p^*)=\\mathrm{TOI}(v)\\Delta_{\\mathrm{rel,best}}(p^*)(1-I_{R^3}(p^*))A(p^*).
\\]

\\section{Figures}
\\begin{figure}[H]
\\centering
\\includegraphics[width=0.86\\textwidth]{\\figroot/top_regions_toi_score.pdf}
\\caption{Highest TOI Mapper regions.}
\\end{figure}

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.86\\textwidth]{\\figroot/top_anchor_ati_score.pdf}
\\caption{Highest ATI anchor planets.}
\\end{figure}

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.70\\textwidth]{\\figroot/toi_vs_anchor_deficit.pdf}
\\caption{Regional TOI score versus anchor local neighbor deficit.}
\\end{figure}

\\section{Interpretation}
A high TOI region is a Mapper node with high observational shadow, low $R^3$ imputation, continuity with its physical neighbors, and sufficient network support. A high ATI planet is a representative anchor inside such a node with positive local deficit of compatible neighbors. These metrics are exploratory and must be read as prioritization scores, not as absolute estimates of missing planets.

\\end{document}
"""
    (latex_dir / "topological_incompleteness_index.tex").write_text(tex, encoding="utf-8")
