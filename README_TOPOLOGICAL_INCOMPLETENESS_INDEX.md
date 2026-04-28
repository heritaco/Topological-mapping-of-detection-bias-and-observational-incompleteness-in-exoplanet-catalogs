# Topological Incompleteness Index

TOI y ATI no descubren planetas faltantes; construyen un ranking topologico de regiones y planetas ancla donde el catalogo parece observacionalmente incompleto.

## Objetivo
Priorizar regiones Mapper y planetas ancla donde la combinacion de sombra observacional, continuidad fisica, soporte de red e imputacion en R^3 sugiere submuestreo topologico.

## Diferencia entre TOI y ATI
- TOI: indice regional para nodos Mapper.
- ATI: indice de planeta ancla que combina TOI con deficit local y representatividad.

## Como correr
python -m src.topological_incompleteness_index.run_topological_incompleteness --config configs/topological_incompleteness_index.yaml

## Inputs requeridos
- dataset imputado o base
- membership nodo-planeta
- edges Mapper
- node_observational_shadow_metrics.csv
- top_shadow_candidates.csv
- node_method_bias_metrics.csv
- node_method_fraction_matrix.csv

## Outputs generados
- CSVs en outputs/topological_incompleteness_index/tables/
- PDF en outputs/topological_incompleteness_index/figures_pdf/
- interpretation_summary.md
- metadata/run_manifest.json
- LaTeX en latex/04_topological_incompleteness/topological_incompleteness_index.tex

## Como interpretar delta_rel_neighbors_best
Es un resumen util para priorizacion, pero no debe leerse solo. Conviene revisarlo junto con los valores por radio, el promedio y la mediana.

## Por que no se afirma descubrimiento de planetas
Los indices usan referencias locales y topologicas. No modelan completitud instrumental ni equivalen a una conclusion sobre objetos ausentes.

## Como leer las figuras
- top_regions_toi_score.pdf: regiones con mayor TOI
- top_anchor_ati_score.pdf: anclas con mayor ATI
- toi_vs_shadow_score.pdf: como se apoya TOI en la sombra observacional
- deficit_distribution_by_radius.pdf: estabilidad del deficit relativo por radio

## Limitaciones
- no hay funcion de completitud instrumental
- N_exp es referencia local
- delta_rel_best puede inflar
- R^3 simplifica fisica
- proxy RV no es amplitud RV real
- la imputacion puede afectar masa

## Proximos pasos
- agregar completitud instrumental
- validar con catalogos futuros
- incorporar propiedades estelares
- comparar por metodo de descubrimiento
