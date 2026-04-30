# Reporte final ejecutivo

Archivo principal:

```bash
tex/final/main.tex
```

Compilación recomendada:

```bash
cd tex/final
pdflatex main.tex
pdflatex main.tex
```

Las figuras del reporte están en:

```bash
tex/final/figures/
```

Los PDFs fuente usados para extraer figuras previas del pipeline están preservados en:

```bash
tex/final/source_reports/
```

Para regenerar los recortes de figuras desde los PDFs fuente:

```bash
python ../../scripts/final/extract_report_figures.py
```

El reporte está diseñado como entrega ejecutiva final: prioriza conclusiones, interpretación de resultados y visualizaciones generadas previamente por el pipeline.
