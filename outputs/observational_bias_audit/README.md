# Observational Bias Audit

## Pregunta
Evalua si la topologia observada en los grafos Mapper, especialmente en el espacio orbital, esta asociada al metodo de descubrimiento y otros sesgos observacionales del catalogo.

## Inputs
- Artifacts de Mapper en `outputs/mapper/`
- Catalogo fisico/imputado alineado con Mapper
- Metadata observacional con `pl_name`, `discoverymethod`, `disc_year` y `disc_facility`

## Outputs
- Figuras PDF en `outputs/observational_bias_audit/figures/`
- Tablas CSV y `.tex` en `outputs/observational_bias_audit/tables/`
- Metadata, membresias reconstruidas y distribuciones nulas en `outputs/observational_bias_audit/metadata/`
- Logs en `outputs/observational_bias_audit/logs/`

## Ejecucion
`python -m src.observational_bias_audit.run_bias_audit --config configs/observational_bias_audit.yaml`

## Metricas principales
- `top_method_fraction`: pureza nodal por metodo de descubrimiento.
- `method_entropy` y `method_entropy_norm`: mezcla interna de metodos por nodo.
- `node_method_nmi`: asociacion global entre incidencias nodo-metodo.
- `enrichment_ratio`, `z_score` y `fdr_q_value`: sobrerrepresentacion local de metodos en nodos.

## Limitaciones
- No implementa la via contrafactual ni rehace Mapper.
- Los resultados dependen de la metadata observacional y de la trazabilidad de imputacion ya existente.
- Nodos pequenos o perifericos pueden mostrar pureza alta por tamano muestral.

## Configuraciones analizadas
- `orbital_pca2_cubes10_overlap0p35`
- `phys_min_pca2_cubes10_overlap0p35`
- `joint_no_density_pca2_cubes10_overlap0p35`
- `joint_pca2_cubes10_overlap0p35`
- `thermal_pca2_cubes10_overlap0p35`
