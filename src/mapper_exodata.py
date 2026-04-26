from __future__ import annotations

import argparse

from mapper_tda.io import (
    PROJECT_ROOT,
    align_mapper_and_physical_inputs,
    load_csv,
    resolve_imputation_outputs_dir,
    resolve_mapper_features_path,
    resolve_outputs_dir,
    resolve_physical_csv_path,
)
from mapper_tda.pipeline import expand_configs_from_cli, run_mapper_batch
from mapper_tda.static_outputs import write_comparison_tables, write_figures, write_latex_report, write_primary_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye grafos Mapper/TDA estaticos sobre tablas imputadas y genera figuras PDF + LaTeX.",
    )
    parser.add_argument("--input-method", default="iterative", choices=["iterative", "knn", "median", "complete_case", "raw"])
    parser.add_argument("--mapper-features-csv", default=None, help="Ruta directa a la matriz transformada/escalada para Mapper.")
    parser.add_argument("--physical-csv", default=None, help="Ruta directa al CSV fisico con trazabilidad.")
    parser.add_argument("--outputs-dir", default="outputs/mapper", help="Carpeta principal de salida. Default: outputs/mapper.")
    parser.add_argument("--imputation-outputs-dir", default="outputs/imputation", help="Carpeta base de outputs de imputacion.")
    parser.add_argument("--space", default="all", help="Espacio Mapper a analizar: phys_min, phys_density, orbital, thermal, orb_thermal, joint_no_density, joint, all.")
    parser.add_argument("--lens", default="pca2", choices=["pca2", "density", "domain", "all"])
    parser.add_argument("--n-cubes", type=int, default=10)
    parser.add_argument("--overlap", type=float, default=0.35)
    parser.add_argument("--clusterer", default="dbscan", choices=["dbscan"])
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--eps-percentile", type=float, default=90.0)
    parser.add_argument("--k-density", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--grid", action="store_true", help="Corre sensibilidad de parametros.")
    parser.add_argument("--fast", action="store_true", help="Corre la configuracion principal base.")
    parser.add_argument("--presentation", action="store_true", help="Enfatiza figuras y tablas de presentacion.")
    parser.add_argument("--full-report", action="store_true", help="Genera todos los assets necesarios para el reporte.")
    parser.add_argument("--make-latex", action="store_true", help="Genera reporte LaTeX despues de crear figuras y tablas.")
    parser.add_argument("--no-html", action="store_true", default=True, help="No generar HTML interactivo.")
    parser.add_argument("--static-only", action="store_true", help="Alias para reforzar salida estatica.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapper_features_path = resolve_mapper_features_path(args.mapper_features_csv, input_method=args.input_method)
    physical_csv_path = resolve_physical_csv_path(args.physical_csv, input_method=args.input_method)
    outputs_dir = resolve_outputs_dir(args.outputs_dir)
    imputation_outputs_dir = resolve_imputation_outputs_dir(args.imputation_outputs_dir)

    mapper_df_raw = load_csv(mapper_features_path)
    physical_df_raw = load_csv(physical_csv_path)
    mapper_df, physical_df, alignment_summary = align_mapper_and_physical_inputs(mapper_df_raw, physical_df_raw)
    configs = expand_configs_from_cli(args)
    batch_result = run_mapper_batch(
        mapper_df=mapper_df,
        physical_df=physical_df,
        configs=configs,
        mapper_features_path=mapper_features_path,
        physical_csv_path=physical_csv_path,
        alignment_summary=alignment_summary,
    )

    write_primary_artifacts(batch_result, outputs_dir)
    write_comparison_tables(batch_result, outputs_dir, imputation_outputs_dir)
    write_figures(batch_result, outputs_dir)

    if args.make_latex or args.full_report or args.presentation:
        write_latex_report(batch_result, outputs_dir, PROJECT_ROOT / "latex" / "03_mapper")

    print("Mapper/TDA generado correctamente.")
    print(f"Mapper features: {mapper_features_path}")
    print(f"Physical CSV: {physical_csv_path}")
    print(f"Filas alineadas: {alignment_summary['n_matched_rows']}")
    print(f"Configuraciones corridas: {len(configs)}")
    print(f"Output principal: {outputs_dir}")
    print("HTML interactivo no forma parte del flujo principal.")


if __name__ == "__main__":
    main()
