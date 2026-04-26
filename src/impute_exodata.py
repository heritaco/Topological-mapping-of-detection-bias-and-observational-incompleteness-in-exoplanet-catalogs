from __future__ import annotations

import argparse

from feature_config import IMPUTATION_FEATURE_SETS
from imputation.io import PROCESSED_DIR, REPORTS_DIR, find_csv, load_pscomppars, resolve_output_dir
from imputation.pipeline import (
    ImputationConfig,
    default_log_features,
    run_imputation_pipeline,
    write_imputation_outputs,
)


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Imputa valores faltantes de PSCompPars para Mapper/TDA y ML.",
    )
    parser.add_argument("--csv", default=None, help="Ruta al CSV PSCompPars. Si se omite, se autodetecta.")
    parser.add_argument(
        "--feature-set",
        default="mapper_core",
        choices=sorted(IMPUTATION_FEATURE_SETS),
        help="Conjunto de variables a imputar.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Lista separada por comas para reemplazar --feature-set.",
    )
    parser.add_argument(
        "--methods",
        default="knn,median,iterative",
        help="Metodos separados por comas. Opciones: knn, median, iterative.",
    )
    parser.add_argument("--primary-method", default="knn", help="Metodo principal usado como referencia.")
    parser.add_argument("--n-neighbors", type=int, default=7, help="Vecinos para KNNImputer.")
    parser.add_argument("--knn-weights", default="distance", choices=["uniform", "distance"], help="Pesos de KNN.")
    parser.add_argument("--iterative-max-iter", type=int, default=20, help="Iteraciones maximas de IterativeImputer.")
    parser.add_argument(
        "--validation-mask-fraction",
        type=float,
        default=0.10,
        help="Fraccion de celdas completas ocultadas para validar imputacion.",
    )
    parser.add_argument(
        "--validation-max-complete-rows",
        type=int,
        default=2000,
        help="Maximo de casos completos usados en validacion enmascarada.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Semilla reproducible.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Carpeta para CSVs imputados. Default: data/processed/<csv>/imputation.",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Carpeta para auditorias. Default: reports/<csv>/imputation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = find_csv(args.csv)
    features = parse_csv_list(args.features) if args.features else tuple(IMPUTATION_FEATURE_SETS[args.feature_set])
    methods = parse_csv_list(args.methods)
    config = ImputationConfig(
        features=features,
        log_features=default_log_features(features),
        methods=methods,
        primary_method=args.primary_method,
        n_neighbors=args.n_neighbors,
        knn_weights=args.knn_weights,
        iterative_max_iter=args.iterative_max_iter,
        validation_mask_fraction=args.validation_mask_fraction,
        validation_max_complete_rows=args.validation_max_complete_rows,
        random_state=args.random_state,
    )

    output_dir = resolve_output_dir(args.output_dir, PROCESSED_DIR, csv_path)
    reports_dir = resolve_output_dir(args.reports_dir, REPORTS_DIR, csv_path)

    df = load_pscomppars(csv_path)
    result = run_imputation_pipeline(df, config)
    paths = write_imputation_outputs(result, config, csv_path, output_dir, reports_dir)

    print("Imputacion generada correctamente.")
    print(f"CSV: {csv_path}")
    print(f"Filas: {len(df):,}")
    print(f"Features: {', '.join(features)}")
    print(f"Metodo principal: {config.primary_method}")
    print("CSVs imputados:")
    for method in methods:
        print(f"  - {paths[f'data_{method}']}")
    print("Auditorias:")
    for key in [
        "missingness_audit",
        "complete_case_comparison",
        "validation_summary",
        "validation_by_feature",
        "config",
    ]:
        print(f"  - {paths[key]}")


if __name__ == "__main__":
    main()

