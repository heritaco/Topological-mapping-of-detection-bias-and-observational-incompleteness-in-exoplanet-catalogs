from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_sets import SPACE_COMPARISON_ORDER, has_density_feature
from .interpretation import generate_interpretation_summary
from .io import ensure_mapper_output_tree, write_json
from visual_style import LENS_MARKERS, PROJECT_COLOR_CYCLE, SOURCE_PALETTE, apply_axis_style, configure_matplotlib, style_colorbar


def _import_matplotlib():
    import matplotlib

    configure_matplotlib(matplotlib)
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, pdf_path: Path, png_path: Path | None = None) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, format="png", dpi=180, bbox_inches="tight")


def _message_figure(pdf_path: Path, title: str, message: str, png_path: Path | None = None) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.set_title(title, loc="left")
    ax.text(0.02, 0.7, message, transform=ax.transAxes, fontsize=11, wrap=True)
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _config_labels(metrics_df: pd.DataFrame) -> list[str]:
    return [
        f"{row.space}\n{row.lens}\nc{int(row.n_cubes)} o{row.overlap:.2f}"
        for row in metrics_df.itertuples(index=False)
    ]


def build_space_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    frame = metrics_df[(metrics_df["lens"] == "pca2") & (metrics_df["n_cubes"] == 10)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["space"] = pd.Categorical(frame["space"], categories=SPACE_COMPARISON_ORDER, ordered=True)
    return frame.sort_values(["space", "overlap"]).reset_index(drop=True)


def build_lens_sensitivity(metrics_df: pd.DataFrame) -> pd.DataFrame:
    frame = metrics_df[metrics_df["lens"].isin(["pca2", "density", "domain"])].copy()
    return frame.sort_values(["space", "lens", "n_cubes", "overlap"]).reset_index(drop=True)


def build_density_feature_sensitivity(metrics_df: pd.DataFrame) -> pd.DataFrame:
    pairs = [("phys_min", "phys_density"), ("joint_no_density", "joint")]
    rows: list[dict[str, Any]] = []
    for without_density, with_density in pairs:
        left = metrics_df[(metrics_df["space"] == without_density) & (metrics_df["lens"] == "pca2")]
        right = metrics_df[(metrics_df["space"] == with_density) & (metrics_df["lens"] == "pca2")]
        if left.empty or right.empty:
            continue
        lrow = left.iloc[0]
        rrow = right.iloc[0]
        rows.append(
            {
                "comparison": f"{without_density}_vs_{with_density}",
                "without_density": without_density,
                "with_density": with_density,
                "delta_n_nodes": float(rrow["n_nodes"] - lrow["n_nodes"]),
                "delta_n_edges": float(rrow["n_edges"] - lrow["n_edges"]),
                "delta_beta_1": float(rrow["beta_1"] - lrow["beta_1"]),
                "delta_average_degree": float(rrow["average_degree"] - lrow["average_degree"]),
                "delta_mean_node_imputation_fraction": float(
                    rrow["mean_node_imputation_fraction"] - lrow["mean_node_imputation_fraction"]
                ),
                "delta_mean_node_physically_derived_fraction": float(
                    rrow["mean_node_physically_derived_fraction"] - lrow["mean_node_physically_derived_fraction"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_imputation_availability(imputation_dir: Path) -> pd.DataFrame:
    methods = ["iterative", "knn", "median", "complete_case"]
    rows = []
    for method in methods:
        rows.append(
            {
                "method": method,
                "mapper_features_exists": (imputation_dir / f"mapper_features_imputed_{method}.csv").exists()
                if method != "complete_case"
                else (imputation_dir / "mapper_features_complete_case.csv").exists(),
                "physical_csv_exists": (imputation_dir / f"PSCompPars_imputed_{method}.csv").exists()
                if method != "complete_case"
                else False,
            }
        )
    return pd.DataFrame(rows)


def write_primary_artifacts(batch_result: dict[str, Any], outputs_dir: Path) -> dict[str, Path]:
    tree = ensure_mapper_output_tree(outputs_dir)
    paths: dict[str, Path] = {}
    node_tables: list[pd.DataFrame] = []
    edge_tables: list[pd.DataFrame] = []
    for result in batch_result["results"]:
        config = result["config"]
        config_name = result["config_id"]
        graph_path = tree["graphs"] / f"graph_{config_name}.json"
        nodes_path = tree["nodes"] / f"nodes_{config_name}.csv"
        edges_path = tree["edges"] / f"edges_{config_name}.csv"
        config_path = tree["config"] / f"config_{config_name}.json"
        write_json(
            graph_path,
            {
                "config": config.__dict__,
                "mapper_metadata": result["mapper_metadata"],
                "graph_metrics": result["graph_metrics"],
                "graph": {
                    key: value
                    for key, value in result["graph"].items()
                    if key in {"nodes", "links", "simplices", "meta_data", "meta_nodes", "sample_id_lookup"}
                },
            },
        )
        result["node_table"].to_csv(nodes_path, index=False)
        result["edge_table"].to_csv(edges_path, index=False)
        write_json(config_path, {"config": config.__dict__, "config_id": config_name})
        node_tables.append(result["node_table"])
        edge_tables.append(result["edge_table"])
        paths[f"graph_{config_name}"] = graph_path

    metrics_path = tree["metrics"] / "mapper_graph_metrics.csv"
    batch_result["metrics_df"].to_csv(metrics_path, index=False)
    distance_path = tree["distances"] / "mapper_graph_distances_metric_l2.csv"
    batch_result["distances_df"].to_csv(distance_path, index=False)
    alignment_path = tree["tables"] / "mapper_input_alignment_summary.csv"
    pd.DataFrame([batch_result["alignment_summary"]]).to_csv(alignment_path, index=False)
    node_audit_path = tree["tables"] / "mapper_node_source_audit.csv"
    pd.concat(node_tables, ignore_index=True).to_csv(node_audit_path, index=False)
    edges_all_path = tree["tables"] / "mapper_edges_all.csv"
    pd.concat(edge_tables, ignore_index=True).to_csv(edges_all_path, index=False)
    paths.update(
        {
            "mapper_graph_metrics": metrics_path,
            "mapper_graph_distances_metric_l2": distance_path,
            "mapper_input_alignment_summary": alignment_path,
            "mapper_node_source_audit": node_audit_path,
            "mapper_edges_all": edges_all_path,
        }
    )
    return paths


def write_comparison_tables(batch_result: dict[str, Any], outputs_dir: Path, imputation_outputs_dir: Path) -> dict[str, Path]:
    tree = ensure_mapper_output_tree(outputs_dir)
    metrics_df = batch_result["metrics_df"]
    paths: dict[str, Path] = {}

    space_comparison = build_space_comparison(metrics_df)
    lens_sensitivity = build_lens_sensitivity(metrics_df)
    density_sensitivity = build_density_feature_sensitivity(metrics_df)
    availability = build_imputation_availability(imputation_outputs_dir)

    table_specs = {
        "mapper_space_comparison.csv": space_comparison,
        "mapper_lens_sensitivity.csv": lens_sensitivity,
        "mapper_density_feature_sensitivity.csv": density_sensitivity,
        "mapper_input_availability.csv": availability,
    }
    for filename, frame in table_specs.items():
        path = tree["tables"] / filename
        frame.to_csv(path, index=False)
        paths[path.stem] = path

    return paths


def _bar_complexity(metrics_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    if metrics_df.empty:
        _message_figure(pdf_path, "Mapper graph size and complexity", "No data available.", png_path)
        return
    plt = _import_matplotlib()
    labels = _config_labels(metrics_df)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, column, color in zip(axes, ["n_nodes", "n_edges", "beta_1"], PROJECT_COLOR_CYCLE[:3]):
        ax.bar(x, pd.to_numeric(metrics_df[column], errors="coerce").fillna(0), color=color, width=0.72)
        apply_axis_style(ax, ylabel=column)
    apply_axis_style(axes[0], title="Mapper graph size and complexity")
    axes[-1].set_xticks(x, labels, rotation=35, ha="right")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _heatmap(frame: pd.DataFrame, columns: list[str], title: str, pdf_path: Path, png_path: Path) -> None:
    if frame.empty or not any(column in frame.columns for column in columns):
        _message_figure(pdf_path, title, "No data available.", png_path)
        return
    use_columns = [column for column in columns if column in frame.columns]
    matrix = frame.loc[:, use_columns].apply(pd.to_numeric, errors="coerce")
    matrix = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, np.nan)
    matrix = matrix.fillna(0.0)
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(14, max(6, len(frame) * 0.45)))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="RdYlBu_r")
    apply_axis_style(ax, title=title)
    ax.set_xticks(range(len(use_columns)), use_columns, rotation=35, ha="right")
    ax.set_yticks(range(len(frame)), _config_labels(frame))
    cbar = fig.colorbar(image, ax=ax)
    style_colorbar(cbar, "z-score")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _distance_heatmap(distances_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    title = "Metric z-score L2 distances"
    if distances_df.empty or "metric_zscore_l2_distance" not in distances_df.columns:
        _message_figure(pdf_path, title, "No pairwise graph distances available.", png_path)
        return
    clean = distances_df.dropna(subset=["metric_zscore_l2_distance"])
    labels = sorted(set(clean["graph_a"]).union(clean["graph_b"]))
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)
    for _, row in clean.iterrows():
        value = float(row["metric_zscore_l2_distance"])
        matrix.loc[row["graph_a"], row["graph_b"]] = value
        matrix.loc[row["graph_b"], row["graph_a"]] = value
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    apply_axis_style(ax, title=title)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    cbar = fig.colorbar(image, ax=ax)
    style_colorbar(cbar, "metric_zscore_l2_distance")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _scatter_nodes_cycles(metrics_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    if metrics_df.empty:
        _message_figure(pdf_path, "Nodes vs cycles", "No data available.", png_path)
        return
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 7))
    spaces = sorted(metrics_df["space"].astype(str).unique())
    colors = {space: color for space, color in zip(spaces, PROJECT_COLOR_CYCLE)}
    for row in metrics_df.itertuples(index=False):
        ax.scatter(
            row.n_nodes,
            row.beta_1,
            color=colors.get(str(row.space), "#334155"),
            marker=LENS_MARKERS.get(str(row.lens), "o"),
            s=92,
            alpha=0.88,
            edgecolors="#ffffff",
            linewidths=0.8,
        )
    apply_axis_style(ax, title="Mapper nodes vs cycles", xlabel="n_nodes", ylabel="beta_1")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _imputation_audit(metrics_df: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    columns = ["mean_node_imputation_fraction", "max_node_imputation_fraction", "frac_nodes_high_imputation"]
    _heatmap(metrics_df, columns, "Imputation audit by configuration", pdf_path, png_path)


def _density_sensitivity_figure(table: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    if table.empty:
        _message_figure(pdf_path, "Density feature sensitivity", "No comparable density pairs available.", png_path)
        return
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 7))
    numeric = table.set_index("comparison").select_dtypes(include=[np.number])
    numeric.plot(kind="bar", ax=ax)
    apply_axis_style(ax, title="Density feature sensitivity", xlabel="", ylabel="delta")
    ax.tick_params(axis="x", rotation=25)
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _lens_sensitivity_figure(table: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    if table.empty:
        _message_figure(pdf_path, "Lens sensitivity", "No lens sensitivity table available.", png_path)
        return
    plt = _import_matplotlib()
    pivot = table.pivot_table(index="space", columns="lens", values="n_nodes", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 7))
    pivot.plot(kind="bar", ax=ax)
    apply_axis_style(ax, title="Lens sensitivity by space", xlabel="", ylabel="mean n_nodes")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _placeholder_imputation_method_figure(pdf_path: Path, png_path: Path) -> None:
    _message_figure(
        pdf_path,
        "Imputation method sensitivity",
        "This run did not compute multi-method Mapper comparisons. See mapper_input_availability.csv for available inputs.",
        png_path,
    )


def _graph_network_figure(result: dict[str, Any], color_column: str, pdf_path: Path, png_path: Path) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 8))
    nx_graph = result["nx_graph"]
    if nx_graph.number_of_nodes() == 0:
        ax.axis("off")
        ax.set_title(pdf_path.stem, loc="left")
        ax.text(0.1, 0.6, "Mapper graph vacio para esta configuracion.", transform=ax.transAxes)
        _save_figure(fig, pdf_path, png_path)
        plt.close(fig)
        return
    import networkx as nx

    layout = nx.spring_layout(nx_graph, seed=42)
    node_table = result["node_table"].set_index("node_id")
    values = pd.to_numeric(node_table.get(color_column), errors="coerce").fillna(0.0)
    nx.draw_networkx_edges(nx_graph, layout, ax=ax, edge_color="#c7d2df", width=1.0, alpha=0.8)
    nodes = nx.draw_networkx_nodes(
        nx_graph,
        layout,
        ax=ax,
        node_color=values.reindex(list(nx_graph.nodes())).fillna(0.0).to_numpy(dtype=float),
        cmap="cividis",
        linewidths=0.9,
        edgecolors="#ffffff",
        node_size=(pd.to_numeric(node_table["n_members"], errors="coerce").reindex(list(nx_graph.nodes())).fillna(1.0) * 16).to_numpy(),
    )
    cbar = fig.colorbar(nodes, ax=ax)
    style_colorbar(cbar, color_column)
    ax.set_title(pdf_path.stem, loc="left")
    ax.axis("off")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _lens_scatter_sources(result: dict[str, Any], pdf_path: Path, png_path: Path) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 7))
    frame = result["physical_df"].copy()
    lens = np.asarray(result["lens"])
    frame["lens_x"] = lens[:, 0]
    frame["lens_y"] = lens[:, 1]
    source_cols = [column for column in frame.columns if column.endswith("_was_imputed")]
    derived_cols = [column for column in frame.columns if column.endswith("_was_physically_derived")]
    imputed = frame[source_cols].apply(pd.to_numeric, errors="coerce").fillna(0).any(axis=1) if source_cols else pd.Series(False, index=frame.index)
    derived = frame[derived_cols].apply(pd.to_numeric, errors="coerce").fillna(0).any(axis=1) if derived_cols else pd.Series(False, index=frame.index)
    status = np.where(imputed, "imputed", np.where(derived, "physically_derived", "observed"))
    for label in ["observed", "physically_derived", "imputed"]:
        mask = status == label
        ax.scatter(
            frame.loc[mask, "lens_x"],
            frame.loc[mask, "lens_y"],
            s=24,
            alpha=0.74,
            label=label,
            color=SOURCE_PALETTE[label],
            edgecolors="#ffffff",
            linewidths=0.35,
        )
    apply_axis_style(ax, title=pdf_path.stem, xlabel="lens_1", ylabel="lens_2")
    ax.legend()
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _node_feature_profiles(result: dict[str, Any], pdf_path: Path, png_path: Path) -> None:
    node_table = result["node_table"].copy()
    features = [column for column in node_table.columns if column.startswith("mean_") and column.replace("mean_", "") in result["used_features"]]
    if node_table.empty or not features:
        _message_figure(pdf_path, pdf_path.stem, "No node feature profile data available.", png_path)
        return
    top = node_table.sort_values("n_members", ascending=False).head(8).set_index("node_id")[features]
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 7))
    top.T.plot(ax=ax)
    apply_axis_style(ax, title=pdf_path.stem, xlabel="feature", ylabel="mean physical value")
    _save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def write_figures(batch_result: dict[str, Any], outputs_dir: Path) -> dict[str, Path]:
    tree = ensure_mapper_output_tree(outputs_dir)
    metrics_df = batch_result["metrics_df"]
    distances_df = batch_result["distances_df"]
    density_table = build_density_feature_sensitivity(metrics_df)
    lens_table = build_lens_sensitivity(metrics_df)

    specs = [
        ("01_mapper_graph_size_complexity.pdf", _bar_complexity, [metrics_df]),
        (
            "02_mapper_metrics_zscore_heatmap.pdf",
            _heatmap,
            [metrics_df, ["n_nodes", "n_edges", "beta_0", "beta_1", "graph_density", "average_degree", "average_clustering", "mean_node_size", "mean_node_imputation_fraction", "mean_node_physically_derived_fraction"], "Mapper metrics z-score heatmap"],
        ),
        ("03_mapper_metric_l2_distances.pdf", _distance_heatmap, [distances_df]),
        ("04_mapper_nodes_vs_cycles.pdf", _scatter_nodes_cycles, [metrics_df]),
        ("05_mapper_imputation_audit_by_config.pdf", _imputation_audit, [metrics_df]),
        ("06_mapper_density_feature_sensitivity.pdf", _density_sensitivity_figure, [density_table]),
        ("07_mapper_lens_sensitivity.pdf", _lens_sensitivity_figure, [lens_table]),
        ("08_mapper_imputation_method_sensitivity.pdf", _placeholder_imputation_method_figure, []),
    ]
    paths: dict[str, Path] = {}
    for filename, func, args in specs:
        pdf_path = tree["figures_pdf"] / filename
        png_path = tree["figures_png"] / filename.replace(".pdf", ".png")
        func(*args, pdf_path, png_path)
        paths[pdf_path.stem] = pdf_path

    principal_results = [result for result in batch_result["results"] if result["config"].lens == "pca2"]
    for result in principal_results:
        config_name = result["config_id"]
        for suffix, column in [
            ("network_pl_rade", "mean_pl_rade"),
            ("network_imputation_fraction", "mean_imputation_fraction"),
            ("network_physically_derived_fraction", "physically_derived_fraction"),
        ]:
            pdf_path = tree["figures_pdf"] / f"graph_{config_name}_{suffix}.pdf"
            png_path = tree["figures_png"] / f"graph_{config_name}_{suffix}.png"
            _graph_network_figure(result, column, pdf_path, png_path)
        pdf_path = tree["figures_pdf"] / f"graph_{config_name}_lens_scatter_sources.pdf"
        png_path = tree["figures_png"] / f"graph_{config_name}_lens_scatter_sources.png"
        _lens_scatter_sources(result, pdf_path, png_path)
        pdf_path = tree["figures_pdf"] / f"graph_{config_name}_node_feature_profiles.pdf"
        png_path = tree["figures_png"] / f"graph_{config_name}_node_feature_profiles.png"
        _node_feature_profiles(result, pdf_path, png_path)

    presentation = tree["figures_pdf"] / "presentation"
    for filename, message in [
        ("slide_01_pipeline_overview.pdf", "datos crudos -> derivacion fisica -> imputacion iterative -> matriz Mapper -> grafos -> sensibilidad"),
        ("slide_02_mapper_spaces.pdf", "phys_min, phys_density, orbital, thermal, orb_thermal, joint_no_density, joint"),
        ("slide_04_density_sensitivity.pdf", "Comparacion con y sin pl_dens usando metricas y grafos principales."),
        ("slide_05_imputation_audit.pdf", "Resumen de configuraciones y nodos mas dependientes de imputacion."),
        ("slide_06_lens_sensitivity.pdf", "Comparacion entre pca2, density y domain por espacio."),
        ("slide_07_interpretation_summary.pdf", generate_interpretation_summary(metrics_df)["global_summary"]),
    ]:
        _message_figure(presentation / filename, filename.replace("_", " "), message)

    selected = [path for path in tree["figures_pdf"].glob("graph_*_network_pl_rade.pdf")][:5]
    if selected:
        shutil.copyfile(selected[0], presentation / "slide_03_main_mapper_graphs.pdf")
    else:
        _message_figure(presentation / "slide_03_main_mapper_graphs.pdf", "slide_03_main_mapper_graphs", "No principal graphs were available.")

    return paths


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _latex_inline_code(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return r"\texttt{" + _latex_escape(match.group(1)) + "}"

    return re.sub(r"`([^`]+)`", repl, text)


def _latexize_summary_text(text: str) -> str:
    text = _latex_inline_code(text)
    return text.replace("beta_1", r"\(\beta_1\)")


def _write_table_tex(frame: pd.DataFrame, path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        path.write_text("% No data available.\n", encoding="utf-8")
        return
    latex = frame.to_latex(index=False, escape=True, longtable=False, caption=caption, label=f"tab:{path.stem}", float_format="%.3f", na_rep="NA")
    latex = latex.replace(r"\begin{table}", "\\begin{table}[p]\n\\centering", 1)
    latex = latex.replace(r"\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}", 1)
    latex = latex.replace(r"\end{tabular}", "\\end{tabular}\n\\end{adjustbox}", 1)
    path.write_text(latex, encoding="utf-8")


def write_latex_report(batch_result: dict[str, Any], outputs_dir: Path, latex_dir: Path) -> dict[str, Path]:
    latex_dir.mkdir(parents=True, exist_ok=True)
    sections_dir = latex_dir / "sections"
    figures_dir = latex_dir / "figures"
    tables_dir = latex_dir / "tables"
    for path in [sections_dir, figures_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)

    metrics_df = batch_result["metrics_df"]
    density_table = build_density_feature_sensitivity(metrics_df)
    lens_table = build_lens_sensitivity(metrics_df)
    space_table = build_space_comparison(metrics_df)
    interpretation = generate_interpretation_summary(metrics_df)
    density_text = _latexize_summary_text(interpretation["density_sensitivity"])
    imputation_text = _latexize_summary_text(interpretation["imputation_audit"])
    lens_text = _latexize_summary_text(interpretation["lens_sensitivity"])
    global_summary = _latexize_summary_text(interpretation["global_summary"])

    for figure in (outputs_dir / "figures_pdf").glob("*.pdf"):
        shutil.copyfile(figure, figures_dir / figure.name)

    _write_table_tex(metrics_df.head(12), tables_dir / "mapper_graph_metrics_summary.tex", "Mapper graph metrics summary.")
    _write_table_tex(space_table, tables_dir / "mapper_space_comparison.tex", "Mapper space comparison.")
    _write_table_tex(density_table, tables_dir / "mapper_density_sensitivity.tex", "Density feature sensitivity.")
    _write_table_tex(lens_table.head(18), tables_dir / "mapper_lens_sensitivity.tex", "Lens sensitivity summary.")
    _write_table_tex(
        metrics_df[["config_id", "mean_node_imputation_fraction", "max_node_imputation_fraction", "frac_nodes_high_imputation"]].copy(),
        tables_dir / "mapper_imputation_audit_summary.tex",
        "Imputation audit summary.",
    )

    sections = {
        "00_abstract.tex": (
            "Se construyeron grafos Mapper sobre el catalogo PSCompPars completado. "
            "La imputacion principal es iterative. Los valores se trazan como observed, physically derived o imputed. "
            "El objetivo es analisis exploratorio topologico, no prueba final de clases planetarias."
        ),
        "01_introduction.tex": (
            "Este reporte resume un pipeline Mapper/TDA orientado a resultados estaticos. "
            "No interpretamos Mapper como prueba directa de la topologia real de los exoplanetas. "
            "Interpretamos los grafos como estructuras inducidas por una matriz completada con trazabilidad explicita. "
            "Las conclusiones mas confiables son las que persisten bajo cambios de espacio de variables, lens, parametros e imputacion, "
            "y que no estan dominadas por variables imputadas o derivadas."
        ),
        "02_data_and_imputation.tex": (
            "El dataset usa siete variables principales: \\texttt{pl\\_rade}, "
            "\\texttt{pl\\_bmasse}, \\texttt{pl\\_dens}, \\texttt{pl\\_orbper}, "
            "\\texttt{pl\\_orbsmax}, \\texttt{pl\\_insol} y \\texttt{pl\\_eqt}. "
            "Mapper usa una matriz transformada y escalada, mientras que la interpretacion usa el CSV fisico con unidades y banderas de procedencia. "
            "Se distinguen explicitamente \\texttt{observed}, \\texttt{physically\\_derived} e \\texttt{imputed}. "
            "Las derivaciones fisicas usadas incluyen $pl\\_dens = 5.514 \\cdot pl\\_bmasse / pl\\_rade^3$ y "
            "$pl\\_orbsmax = (st\\_mass \\cdot (pl\\_orbper / 365.25)^2)^{1/3}$. "
            "\\texttt{pl\\_dens} es mayoritariamente derivada desde \\texttt{pl\\_bmasse} y \\texttt{pl\\_rade}; por tanto, no debe tratarse como observacion independiente cuando se usa junto con masa y radio."
        ),
        "03_mapper_methodology.tex": (
            "Sea $X = \\{x_1, \\ldots, x_n\\} \\subset \\mathbb{R}^p$. Sea $f: X \\to \\mathbb{R}^d$ un lens y $\\mathcal{U}=\\{U_\\alpha\\}$ una cubierta del espacio de lentes. "
            "Para cada $U_\\alpha$ se clusteriza $f^{-1}(U_\\alpha)$ y se construye un grafo donde los nodos son clusters locales y las aristas representan interseccion no vacia de miembros. "
            "Usamos $\\beta_1 = E - V + C$ con $E$ aristas, $V$ nodos y $C=\\beta_0$ componentes conexas. "
            "La configuracion base usa n\\_cubes=10, overlap=0.35, DBSCAN, min\\_samples=4, eps\\_percentile=90 y random\\_state=42."
        ),
        "04_mapper_results.tex": (
            "Los resultados principales se resumen en las figuras estaticas globales y en "
            "los grafos \\texttt{joint\\_no\\_density} y \\texttt{joint} con lens "
            "\\texttt{pca2}. "
            f"{global_summary} "
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{figures/01_mapper_graph_size_complexity.pdf}\\caption{Tamano y complejidad de los grafos Mapper.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{figures/02_mapper_metrics_zscore_heatmap.pdf}\\caption{Heatmap de metricas estandarizadas.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.75\\linewidth]{figures/04_mapper_nodes_vs_cycles.pdf}\\caption{Relacion entre numero de nodos y ciclos.}\\end{figure}"
        ),
        "05_sensitivity_analysis.tex": (
            f"{density_text} {lens_text} {imputation_text} "
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.9\\linewidth]{figures/06_mapper_density_feature_sensitivity.pdf}\\caption{Sensibilidad al agregar \\texttt{pl\\_dens}.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.9\\linewidth]{figures/05_mapper_imputation_audit_by_config.pdf}\\caption{Auditoria de imputacion por configuracion.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.9\\linewidth]{figures/07_mapper_lens_sensitivity.pdf}\\caption{Sensibilidad al lens.}\\end{figure}"
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.9\\linewidth]{figures/08_mapper_imputation_method_sensitivity.pdf}\\caption{Sensibilidad al metodo de imputacion cuando hay insumos comparables.}\\end{figure}"
        ),
        "06_interpretation.tex": (
            "Mapper muestra estructura topologica inducida por un espacio de variables completado y auditado. "
            "Las estructuras con mayor credibilidad son las que persisten entre espacios, lenses y controles de imputacion, y que ademas tienen interpretacion fisica. "
            "Las regiones compatibles con rocosos, super-Tierras, sub-Neptunos, gigantes gaseosos o hot Jupiters solo deben enfatizarse si los nodos correspondientes muestran consistencia fisica y baja dependencia de imputacion."
        ),
        "07_limitations.tex": (
            "Las limitaciones principales incluyen sesgo observacional por "
            "\\texttt{discoverymethod}, variables derivadas no independientes, imputacion que "
            "no equivale a observacion, dependencia de parametros Mapper, y el hecho de que "
            "un \\(\\beta_1\\) alto no implica automaticamente estructura fisica. "
            "La distancia \\texttt{metric\\_zscore\\_l2} es una distancia entre vectores de "
            "metricas estandarizadas, no una distancia topologica estricta. "
            "\\texttt{observed} \\(\\neq\\) \\texttt{physically\\_derived} \\(\\neq\\) "
            "\\texttt{imputed}. % observed != physically_derived != imputed"
        ),
        "08_conclusion.tex": (
            "El pipeline Mapper quedo actualizado para usar iterative por defecto, generar outputs estaticos en outputs/mapper y producir un reporte LaTeX autocontenido. "
            "Los siguientes pasos recomendados son estabilidad por bootstrap y revision cientifica de nodos destacados."
        ),
    }
    for filename, body in sections.items():
        (sections_dir / filename).write_text(body + "\n", encoding="utf-8")

    report_body = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{float}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{longtable}
\usepackage{array}
\usepackage{placeins}
\usepackage{pdflscape}
\usepackage{adjustbox}
\title{Mapper/TDA Report for Imputed PSCompPars}
\date{}
\begin{document}
\maketitle
\input{sections/00_abstract.tex}
\section{Introduction}
\input{sections/01_introduction.tex}
\section{Data and Imputation}
\input{sections/02_data_and_imputation.tex}
\section{Mapper Methodology}
\input{sections/03_mapper_methodology.tex}
\section{Mapper Results}
\input{sections/04_mapper_results.tex}
\section{Sensitivity Analysis}
\input{sections/05_sensitivity_analysis.tex}
\section{Interpretation}
\input{sections/06_interpretation.tex}
\section{Limitations}
\input{sections/07_limitations.tex}
\section{Conclusion}
\input{sections/08_conclusion.tex}
\FloatBarrier
\clearpage
\begin{landscape}
\footnotesize
\input{tables/mapper_graph_metrics_summary.tex}
\input{tables/mapper_space_comparison.tex}
\input{tables/mapper_density_sensitivity.tex}
\input{tables/mapper_lens_sensitivity.tex}
\input{tables/mapper_imputation_audit_summary.tex}
\end{landscape}
\end{document}
"""
    report_path = latex_dir / "mapper_report.tex"
    report_path.write_text(report_body, encoding="utf-8")
    (latex_dir / "README.md").write_text(
        "Compila con `latexmk -pdf -interaction=nonstopmode -halt-on-error mapper_report.tex`. "
        "Las figuras se copian desde `outputs/mapper/figures_pdf/`.\n",
        encoding="utf-8",
    )
    (latex_dir / "Makefile").write_text(
        "all:\n\tlatexmk -pdf -interaction=nonstopmode -halt-on-error mapper_report.tex\n",
        encoding="utf-8",
    )
    return {"mapper_report_tex": report_path}
