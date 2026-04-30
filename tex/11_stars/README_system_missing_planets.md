# Reporte LaTeX del subproyecto `system_missing_planets`

Este reporte documenta el m\'odulo:

- `src/system_missing_planets/`

y su objetivo de priorizar intervalos orbitales dentro de sistemas planetarios conocidos donde podr\'ian existir candidatos a planetas no detectados bajo una lectura prudente.

## Archivo principal a compilar

- `tex/11_stars/system_missing_planets_report.tex`

## Compilaci\'on local

Desde la carpeta `tex/11_stars/`:

```bash
latexmk -pdf system_missing_planets_report.tex
```

Alternativa manual:

```bash
pdflatex system_missing_planets_report.tex
biber system_missing_planets_report
pdflatex system_missing_planets_report.tex
pdflatex system_missing_planets_report.tex
```

## Estructura asociada

- `tex/11_stars/sections/system_missing_planets/*.tex`
- `tex/11_stars/tables/system_missing_planets/`
- `tex/11_stars/figures/system_missing_planets/`
- `tex/11_stars/references_system_missing_planets.bib`

## Outputs esperados para que aparezcan las figuras

El reporte apunta a:

- `../../outputs/system_missing_planets/figures/top_high_priority_systems.pdf`
- `../../outputs/system_missing_planets/figures/candidate_priority_distribution.pdf`
- `../../outputs/system_missing_planets/figures/gap_ratio_vs_priority.pdf`
- `../../outputs/system_missing_planets/figures/system_architecture_<hostname_safe>.pdf`

y a tablas de referencia como:

- `../../outputs/system_missing_planets/high_priority_candidates.csv`
- `../../outputs/system_missing_planets/system_gap_summary.csv`
- `../../outputs/system_missing_planets/system_missing_planets_validation_summary.json`

## Si faltan figuras

El documento usa `\safeincludegraphics`, as\'i que la compilaci\'on no deber\'ia fallar si un PDF todav\'ia no existe. En ese caso aparecer\'a un recuadro placeholder con la ruta esperada.

## C\'omo regenerar el reporte despu\'es de correr el m\'odulo

1. Ejecutar primero el pipeline del subproyecto, por ejemplo:

```bash
conda run -n planetas python -m src.system_missing_planets.run_system_missing_planets \
  --catalog data/PSCompPars_imputed_iterative.csv \
  --output-dir outputs/system_missing_planets \
  --mode all \
  --make-figures \
  --make-latex-summary
```

2. Revisar y actualizar los placeholders `TODO` dentro de las tablas del reporte.

3. Compilar desde `tex/11_stars/`:

```bash
latexmk -pdf system_missing_planets_report.tex
```

## Nota editorial

El lenguaje del reporte est\'a dise\~nado para mantener una lectura prudente: priorizaci\'on observacional, posibles candidatos intra-sistema y se\~nal topol\'ogica exploratoria. No debe reinterpretarse como un reclamo de descubrimiento de planetas reales.
