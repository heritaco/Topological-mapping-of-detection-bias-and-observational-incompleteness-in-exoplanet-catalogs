# Local Shadow Case Studies

## Pregunta
Este subproyecto pregunta si una comunidad Mapper de alta sombra observacional puede leerse como una ficha local de incompletitud topologica: un nodo, su vecindario, sus miembros en R3 y un exoplaneta ancla.

## Por que comunidades RV
Los reportes previos mostraron que varios candidatos de alta sombra en el Mapper orbital estan dominados por `Radial Velocity`. Eso hace plausible una lectura prudente de incompletitud hacia menor masa planetaria o menor proxy de detectabilidad RV.

## Que es R3
El espacio principal del caso local es:
`R3 = (log10(pl_bmasse), log10(pl_orbper), log10(pl_orbsmax))`

## Como se selecciona el planeta ancla
El ancla debe tener R3 valido, se priorizan miembros `Radial Velocity`, baja imputacion en masa/periodo/semieje, mejor trazabilidad observacional y cercania al medoid del nodo.

## Como se calcula el deficit local
Para cada ancla se compara el conteo observado de vecinos compatibles dentro de radios locales contra referencias prudentes basadas en vecinos topologicos y nodos analogos de menor sombra. Esto produce un `deficit topologico local`, no un conteo absoluto de planetas reales faltantes.

## Como correr
`python -m src.local_shadow_case_studies.run_local_shadow_cases --config configs/local_shadow_case_studies.yaml`

## Donde quedan los resultados
- Tablas CSV: `outputs/local_shadow_case_studies/tables/`
- Figuras PDF: `outputs/local_shadow_case_studies/figures/`
- Metadata y manifest: `outputs/local_shadow_case_studies/metadata/` y `outputs/local_shadow_case_studies/run_manifest.json`
- Log: `outputs/local_shadow_case_studies/logs/`
- Reporte LaTeX: `latex/local_shadow_case_studies/main.tex`

## Como interpretar
Leer cada caso como un candidato a incompletitud observacional o region fisico-orbital submuestreada. El lenguaje correcto es `deficit topologico local`, `vecinos compatibles esperados bajo referencia local` y `posible incompletitud hacia menor masa o menor senal RV`.

## Advertencias
- No afirmar `planetas faltantes confirmados`.
- No afirmar `faltan exactamente N exoplanetas reales`.
- El resultado depende del Mapper usado, de la referencia local y de la trazabilidad de imputacion.
