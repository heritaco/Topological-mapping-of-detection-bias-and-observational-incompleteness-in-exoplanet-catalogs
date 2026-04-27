# ExoData Exoplanet Clustering

Pipeline reproducible para imputacion y analisis Mapper/TDA sobre `PSCompPars`,
orientado ahora a salidas estaticas: PDFs, tablas auditables y reporte LaTeX.

## Principio metodologico

No interpretamos Mapper como prueba directa de la topologia real de los
exoplanetas. Interpretamos los grafos como estructuras inducidas por una matriz
completada con trazabilidad explicita. Las conclusiones mas confiables son las
que persisten bajo cambios de espacio de variables, lens, parametros e
imputacion, y que no estan dominadas por variables imputadas o derivadas.

`pl_dens` es mayoritariamente derivada desde `pl_bmasse` y `pl_rade`; por
tanto, no debe tratarse como observacion independiente cuando se usa junto con
masa y radio.

## Estructura

- `src/impute_exodata.py`: pipeline de imputacion.
- `src/mapper_exodata.py`: pipeline Mapper/TDA estatico.
- `src/mapper_tda/`: espacios, lenses, metricas, figuras y reporte LaTeX.
- `outputs/imputation/`: salidas principales de imputacion.
- `outputs/mapper/`: salidas principales de Mapper.
- `latex/`: reporte LaTeX autonomo y sus secciones.
- `reports/`: solo legado; ya no es el destino principal de Mapper.

## Outputs principales

```text
outputs/
  imputation/
    data/
    tables/
    figures_pdf/
    figures_png/
    json/
  mapper/
    data/
    graphs/
    nodes/
    edges/
    metrics/
    distances/
    tables/
    figures_pdf/
    figures_png/
    latex_assets/
    config/
  logs/

latex/
  03_mapper/
    mapper_report.tex
    sections/
    figures/
    tables/
```

## Entorno

```powershell
conda env create -f .\environment.yml
conda activate planetas
```

Si instalas manualmente:

```powershell
conda create -y -n planetas --override-channels -c conda-forge python=3.12 pandas numpy matplotlib scikit-learn scipy networkx pytest pip
conda activate planetas
python -m pip install "kmapper>=2.1"
```

## Imputacion

`iterative` es el metodo principal por defecto. KNN ya no es el default
conceptual del flujo documentado.

Comando recomendado:

```powershell
python .\src\impute_exodata.py --method compare --visualized-method iterative --outputs-dir outputs/imputation
```

Archivos clave esperados para Mapper:

- `outputs/imputation/data/mapper_features_imputed_iterative.csv` si existe.
- `outputs/imputation/data/PSCompPars_imputed_iterative.csv` si existe.
- Fallback legacy desde `reports/imputation/`.

La interpretacion distingue explicitamente:

- `observed`
- `physically_derived`
- `imputed`

En particular:

```text
pl_dens = 5.514 * pl_bmasse / pl_rade^3
pl_orbsmax = (st_mass * (pl_orbper / 365.25)^2)^(1/3)
```

## Mapper/TDA

El pipeline principal ya no genera dashboards ni HTML interactivo como salida
default. El producto final es:

1. Figuras PDF estaticas.
2. Tablas CSV/JSON.
3. Reporte LaTeX.

Espacios principales:

- `phys_min`
- `phys_density`
- `orbital`
- `thermal`
- `orb_thermal`
- `joint_no_density`
- `joint`

Aliases legacy:

- `phys -> phys_density`
- `orb -> orb_thermal`
- `all -> todos los espacios`

Lenses:

- `pca2`: principal.
- `density`: sensibilidad.
- `domain`: interpretativo auxiliar.

Default metodologico:

- `input-method=iterative`
- `lens=pca2`
- `n_cubes=10`
- `overlap=0.35`
- `clusterer=dbscan`
- `min_samples=4`
- `eps_percentile=90`
- `k_density=15`
- `random_state=42`

Comandos documentados:

```powershell
python .\src\mapper_exodata.py --space all --lens pca2 --input-method iterative --outputs-dir outputs/mapper --interpret-nodes --presentation --make-latex
python .\src\mapper_exodata.py --space all --lens all --input-method iterative --outputs-dir outputs/mapper --full-report
python .\src\mapper_exodata.py --space all --lens all --input-method iterative --outputs-dir outputs/mapper --full-validation --n-bootstrap 30 --n-null 30
```

El resolver de inputs prioriza `iterative`. Solo si no existe entra en
fallbacks legacy, incluyendo KNN.

Flags nuevos importantes:

- `--interpret-nodes`: genera etiquetas fisicas, composicion nodal, componentes y nodos destacados.
- `--bootstrap --n-bootstrap 30 --bootstrap-frac 0.8`: corre estabilidad por remuestreo.
- `--null-models --n-null 30`: corre modelos nulos `column_shuffle`.
- `--presentation`: genera figuras tipo slide.
- `--full-report`: equivale a interpretacion + presentacion + LaTeX.
- `--full-validation`: agrega bootstrap y null models al flujo anterior.

Artefactos interpretativos clave:

- `outputs/mapper/data/planet_physical_labels.csv`
- `outputs/mapper/tables/main_graph_selection.csv`
- `outputs/mapper/tables/node_physical_interpretation.csv`
- `outputs/mapper/tables/highlighted_nodes.csv`
- `outputs/mapper/tables/component_summary.csv`
- `outputs/mapper/tables/mapper_interpretive_summary.md`
- `outputs/mapper/figures_pdf/interpretation/*.pdf`
- `outputs/mapper/figures_pdf/presentation/*.pdf`

## LaTeX

El pipeline puede generar un reporte completo en `latex/03_mapper/mapper_report.tex`.

Estilo visual por defecto:

- Todas las figuras Matplotlib del proyecto deben usar el estilo compartido de `src/visual_style.py`.
- La guia para humanos e IA esta en `VISUAL_STYLE_GUIDE.md`.

Compilacion:

```powershell
cd latex/03_mapper
latexmk -pdf -interaction=nonstopmode -halt-on-error mapper_report.tex
```

## HTML interactivo

Las funciones legacy de HTML/Plotly pueden seguir existiendo para compatibilidad
interna, pero HTML interactivo no es parte del flujo principal de Mapper y no
debe considerarse output principal.

## Tests

```powershell
python -m pytest
```
