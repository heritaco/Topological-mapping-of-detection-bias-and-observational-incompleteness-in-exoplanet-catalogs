# ExoData Exoplanet Clustering

Este proyecto organiza el analisis inicial del archivo `PSCompPars` del NASA
Exoplanet Archive para el ExoData Challenge.

## Decision de variables

La guia del concurso no pide usar las 320 columnas como variables de modelado.
El foco esta en variables fisicas, orbitales, estelares y de habitabilidad:

- Planeta: `pl_rade`, `pl_bmasse`, `pl_dens`
- Orbita: `pl_orbper`, `pl_orbsmax`, `pl_orbeccen`
- Estrella/sistema: `st_teff`, `st_met`, `sy_pnum`
- Habitabilidad opcional: `pl_insol`, `pl_eqt`
- Sesgo observacional: `discoverymethod`, `disc_year`

Las 320 columnas si se perfilan para revisar nulos, rangos y metadatos, pero
para clustering conviene comenzar con subconjuntos numericos interpretables.

## Estructura

- `src/eda_exodata.py`: genera el EDA reproducible en Plotly.
- `src/feature_config.py`: define variables clave y grupos de features para clustering.
- `src/impute_exodata.py`: genera matrices imputadas y auditorias para Mapper/TDA y ML.
- `src/imputation/steps/`: pasos auditables del pipeline de imputacion.
- `notebooks/`: notebooks renderizables para revisar EDA y preparar clustering.
- `reports/`: salidas generadas: HTML interactivo, tablas de nulos, rangos y correlaciones.
- `data/`: espacio recomendado para organizar datos si despues se mueve el CSV.
- `tests/`: pruebas unitarias del pipeline de imputacion.

## Datos

Los CSV estan en `data/`.

- `PSCompPars_2026.04.25_14.43.08.csv`: tabla completa, 320 columnas.
- `PSCompPars_2026.04.25_17.36.36.csv`: tabla compacta, 84 columnas, mismas filas.

Para clustering inicial, la tabla compacta es mas manejable porque conserva las
variables centrales y elimina muchos enlaces/metadatos. La excepcion importante
es `pl_dens`: si no viene en el CSV, el script la deriva como
`5.514 * pl_bmasse / pl_rade^3`.

## Como ejecutar

Crear el entorno Conda del proyecto:

```powershell
conda env create -f .\environment.yml
conda activate planetas
```

Si tu instalacion de Conda pide aceptar terminos de los canales `defaults`,
crea el mismo entorno solo con `conda-forge`:

```powershell
conda create -y -n planetas --override-channels -c conda-forge python=3.12 pandas numpy plotly scikit-learn scipy nbformat nbconvert ipykernel jupyterlab pytest
conda activate planetas
```

```powershell
python .\src\eda_exodata.py
```

El script detecta automaticamente `PSCompPars_*.csv` en `data/` y toma el mas
reciente por nombre.
Tambien puedes pasar un archivo explicitamente:

```powershell
python .\src\eda_exodata.py --csv .\data\PSCompPars_2026.04.25_17.36.36.csv --reports-dir .\reports\PSCompPars_2026.04.25_17.36.36
```

## Imputacion para Mapper/TDA y ML

El pipeline de imputacion esta pensado para no inventar datos sin control. El
metodo principal es:

```text
derivacion fisica -> log-transform -> RobustScaler -> KNNImputer -> inversion de escala -> auditoria
```

Primero se preservan valores observados de `pl_dens` y solo se derivan faltantes
cuando hay masa y radio positivos:

```text
pl_dens = 5.514 * pl_bmasse / pl_rade**3
```

Despues se transforman en log10 las variables positivas y sesgadas, se escala
con `RobustScaler`, se imputa en el espacio escalado y se vuelve a unidades
fisicas. `KNNImputer` es el default porque Mapper depende de vecindades locales:
la imputacion se apoya en planetas cercanos en las variables observadas. `median`
queda como baseline robusto e `iterative` como sensibilidad avanzada, porque
puede imponer relaciones globales y suavizar artificialmente la topologia.

Conjuntos disponibles:

- `mapper_phys`: `pl_rade`, `pl_bmasse`, `pl_dens`
- `mapper_core`: fisicas + orbitales + estrella/sistema
- `mapper_wide`: `mapper_core` + `pl_insol`, `pl_eqt`

Ejecutar el pipeline recomendado:

```powershell
python .\src\impute_exodata.py --feature-set mapper_core
```

Tambien se pueden comparar sensibilidades en el conjunto amplio:

```powershell
python .\src\impute_exodata.py --feature-set mapper_wide --methods knn,median,iterative --n-neighbors 7
```

Salidas principales:

- `data/processed/<csv>/imputation/mapper_features_knn_imputed.csv`: matriz principal para Mapper/TDA y ML.
- `data/processed/<csv>/imputation/mapper_features_median_imputed.csv`: baseline de mediana.
- `data/processed/<csv>/imputation/mapper_features_iterative_imputed.csv`: sensibilidad avanzada.
- `reports/<csv>/imputation/imputation_missingness.csv`: nulos antes/despues, derivaciones e invalidos para log.
- `reports/<csv>/imputation/imputation_validation_summary.csv`: validacion con casos completos enmascarados.
- `reports/<csv>/imputation/imputation_complete_case_comparison.csv`: comparacion de distribuciones contra casos completos.
- `reports/<csv>/imputation/imputation_config.json`: configuracion reproducible.

## Salida principal

Abre el archivo:

```text
reports/PSCompPars_2026.04.25_17.36.36/exodata_eda_plotly.html
```

Ese reporte contiene visualizaciones interactivas en Plotly para nulos,
distribuciones, correlaciones, sesgos por metodo de descubrimiento y cobertura
de conjuntos de variables para clustering.

## Notebooks

Los notebooks estan pensados para verse en GitHub y ejecutarse localmente:

- `notebooks/01_eda_overview.ipynb`: resumen del EDA, nulos, rangos, cobertura y correlaciones.
- `notebooks/02_clustering_prep.ipynb`: seleccion de variables, transformaciones logaritmicas, escalado y PCA exploratorio.

Para ejecutarlos:

```powershell
jupyter lab
```

## Tests

```powershell
python -m pytest
```

## Repositorio sugerido

Nombre recomendado para GitHub: `exodata-exoplanet-clustering`.

Para publicarlo con GitHub CLI:

```powershell
gh auth login
gh repo create exodata-exoplanet-clustering --public --source . --remote origin --push
```

Si creas primero el repositorio desde la web de GitHub:

```powershell
git remote add origin https://github.com/<tu-usuario>/exodata-exoplanet-clustering.git
git push -u origin main
```
