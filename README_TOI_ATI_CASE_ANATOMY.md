# TOI/ATI Case Anatomy

Este modulo no calcula otro indice nuevo; abre los rankings TOI/ATI para explicar por que ciertas regiones y planetas ancla fueron priorizados. La salida principal es una anatomia de los casos top, no una afirmacion de objetos ausentes confirmados.

## Objetivo
Consumir los rankings producidos por `topological_incompleteness_index` y generar una vista explicable de los casos top:
- descomposicion de factores TOI
- descomposicion de factores ATI
- resumen del deficit por radio
- shortlist de casos para revision manual

## Como correr
```bash
python -m src.toi_ati_case_anatomy.run_case_anatomy --config configs/toi_ati_case_anatomy.yaml
```

Override opcional:
```bash
python -m src.toi_ati_case_anatomy.run_case_anatomy --config configs/toi_ati_case_anatomy.yaml --config-id orbital_pca2_cubes10_overlap0p35
```

## Inputs esperados
- `outputs/topological_incompleteness_index/tables/regional_toi_scores.csv`
- `outputs/topological_incompleteness_index/tables/anchor_ati_scores.csv`
- `outputs/topological_incompleteness_index/tables/anchor_neighbor_deficits.csv`
- `outputs/topological_incompleteness_index/tables/r3_node_geometry.csv`

Inputs opcionales:
- `outputs/observational_shadow/tables/node_observational_shadow_metrics.csv`
- `outputs/local_shadow_case_studies/tables/case_node_summary.csv`
- `outputs/local_shadow_case_studies/tables/case_anchor_planets.csv`

## Outputs
- tablas en `outputs/toi_ati_case_anatomy/tables/`
- figuras PDF en `outputs/toi_ati_case_anatomy/figures_pdf/`
- resumen en `outputs/toi_ati_case_anatomy/interpretation_summary.md`
- manifest en `outputs/toi_ati_case_anatomy/metadata/run_manifest.json`
- LaTeX en `latex/05_toi_ati_case_anatomy/toi_ati_case_anatomy.tex`

## Como interpretar
- TOI alto: region con combinacion fuerte de sombra, baja imputacion, continuidad fisica y soporte de red.
- ATI alto: ancla dentro de una region TOI alta con mayor deficit local y buena representatividad.
- `delta_rel_neighbors_best`: resumen util, pero debe leerse junto con el detalle por radio.

## Limitaciones
- no hay completitud instrumental
- el deficit local depende de la referencia elegida
- el valor best puede inflar la lectura si se usa aislado
- el modulo no valida por si solo una afirmacion observacional fuerte
