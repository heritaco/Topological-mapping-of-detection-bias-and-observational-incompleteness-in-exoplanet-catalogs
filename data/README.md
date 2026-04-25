# Datos

El CSV actual esta en la raiz del proyecto para no mover archivos originales.
Si despues se convierte en repositorio formal, una estructura recomendable es:

```text
data/
  raw/          # archivos descargados sin modificar
  interim/      # datos filtrados o parcialmente limpios
  processed/    # matrices listas para modelado
```

Los scripts detectan el CSV en la raiz o pueden recibirlo con `--csv`.
