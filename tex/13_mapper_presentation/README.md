# Presentacion Mapper/TDA de exoplanetas

Esta carpeta contiene una presentacion Beamer autocontenida para la entrega del proyecto.

Compilar desde esta carpeta:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

La narrativa sigue CRISP-DM:

1. Business Understanding: problema, pregunta y alcance.
2. Data Understanding: PSCompPars, faltantes y metadata observacional.
3. Data Preparation: imputacion, derivaciones fisicas y espacios de variables.
4. Modeling: Mapper paso a paso, DBSCAN local, lentes y 21 grafos.
5. Evaluation: comparacion de grafos, orbital principal, imputacion y sesgo.
6. Deployment/Communication: mapa final de evidencia, limitaciones y conclusion.

Mensaje central:

> Mapper/TDA funciona aqui como una herramienta para priorizar regiones de inspeccion en el catalogo de exoplanetas, no como una taxonomia final.

Las figuras fueron copiadas desde `latex/03_mapper/figures/` y `reports/imputation/outputs/figures_pdf/`.
