# Observational Shadow

## Pregunta
Este subproyecto pregunta si las comunidades Mapper pueden senalar regiones topologicas submuestreadas por la funcion de seleccion observacional del catalogo de exoplanetas.

## Inputs usados
- Grafos, nodos y aristas Mapper en `outputs/mapper/`.
- Membresias enriquecidas de la auditoria observacional cuando existen.
- Catalogos imputados/fisicos con `discoverymethod`, `disc_year`, `disc_facility` y variables fisico-orbitales.

## Como correr
`python -m src.observational_shadow.run_observational_shadow --config configs/observational_shadow.yaml`

## Outputs
- Figuras PDF: `outputs/observational_shadow/figures/`
- Tablas CSV: `outputs/observational_shadow/tables/`
- Metadata y manifest: `outputs/observational_shadow/metadata/` y `outputs/observational_shadow/run_manifest.json`
- Logs: `outputs/observational_shadow/logs/`
- Reporte LaTeX: `latex/observational_shadow/main.tex`

## Interpretacion de `shadow_score`
`shadow_score` combina pureza por metodo dominante, baja entropia, baja imputacion, contraste de composicion con vecinos y peso por tamano nodal. Un valor alto debe leerse como candidato a incompletitud observacional o posible frontera de seleccion, no como prueba de planetas faltantes.

## Limitaciones
Mapper no prueba poblaciones no observadas. Los cortes de clase son heuristicos. Nodos pequenos pueden inflar la pureza. La imputacion puede alterar algunas regiones. Sin funciones de completitud instrumental no se estiman cantidades absolutas de objetos no observados.

## Configuraciones analizadas
- `orbital_pca2_cubes10_overlap0p35`
- `phys_min_pca2_cubes10_overlap0p35`
- `joint_no_density_pca2_cubes10_overlap0p35`
- `joint_pca2_cubes10_overlap0p35`
- `thermal_pca2_cubes10_overlap0p35`
