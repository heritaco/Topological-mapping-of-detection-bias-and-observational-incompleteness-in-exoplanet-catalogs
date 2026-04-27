# Visual Style Guide

Este archivo define el estilo visual por defecto para todas las figuras
estaticas del proyecto. Es una referencia explicita para humanos y para IA.

## Regla principal

Toda visualizacion nueva del proyecto debe usar el estilo compartido en
`src/visual_style.py`.

## Base estetica

- Base tecnica: `seaborn-v0_8-whitegrid`.
- Fondo: blanco limpio.
- Grid: sutil, gris azulado claro.
- Tipografia: sans serif limpia y neutral.
- Sensacion buscada: moderna, cientifica, clara y ligera.

## Paleta por defecto

- Azul principal: `#2563eb`
- Verde petroleo: `#0f766e`
- Ambar: `#f59e0b`
- Rojo de alerta: `#dc2626`
- Slate neutro: `#475569`
- Cian auxiliar: `#0891b2`

## Semantica recomendada

- `observed`: azul
- `physically_derived`: verde petroleo
- `imputed`: gris claro/slate

## Instrucciones para IA

- Antes de crear una figura Matplotlib, importar y aplicar `configure_matplotlib`.
- Si la figura usa ejes clasicos, usar `apply_axis_style(...)`.
- Si la figura usa colorbar, usar `style_colorbar(...)`.
- Mantener fondos blancos y evitar fondos oscuros.
- Evitar paletas estridentes salvo razon metodologica clara.
- Priorizar legibilidad en PDF.
- Mantener titulos cortos y alineados a la izquierda.
- Evitar decoracion innecesaria.

## Alcance

Este estilo es el default para:

- `src/mapper_tda/static_outputs.py`
- `src/imputation/pipeline.py`
- futuras figuras Matplotlib del proyecto
