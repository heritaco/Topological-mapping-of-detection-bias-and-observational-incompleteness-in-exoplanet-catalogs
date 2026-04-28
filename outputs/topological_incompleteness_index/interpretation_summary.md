# Topological Incompleteness Index

## Resumen ejecutivo
TOI y ATI construyen un ranking topologico de regiones Mapper y planetas ancla donde el catalogo parece observacionalmente incompleto bajo referencias locales prudentes.

## Que es TOI
TOI resume sombra observacional, imputacion especifica en R^3, continuidad fisica con el vecindario y soporte de red.

## Que es ATI
ATI combina TOI con el deficit relativo local del planeta ancla, su trazabilidad en R^3 y su representatividad dentro del nodo.

## Top 5 regiones
- cube12_cluster0: TOI=0.077, shadow=0.085, clase=high_toi_region, metodo=Transit.
- cube17_cluster2: TOI=0.063, shadow=0.274, clase=high_toi_region, metodo=Radial Velocity.
- cube26_cluster6: TOI=0.058, shadow=0.240, clase=high_toi_region, metodo=Radial Velocity.
- cube19_cluster5: TOI=0.058, shadow=0.240, clase=high_toi_region, metodo=Radial Velocity.
- cube19_cluster4: TOI=0.058, shadow=0.243, clase=high_toi_region, metodo=Radial Velocity.

## Top 5 planetas ancla
- HIP 97166 c / cube12_cluster0: ATI=0.007, deficit_best=0.091, clase=no_deficit.
- HIP 90988 b / cube17_cluster2: ATI=0.007, deficit_best=0.111, clase=weak_deficit.
- HD 42012 b / cube26_cluster6: ATI=0.005, deficit_best=0.091, clase=no_deficit.
- HD 42012 b / cube19_cluster5: ATI=0.005, deficit_best=0.091, clase=no_deficit.
- HD 4313 b / cube17_cluster10: ATI=0.005, deficit_best=0.167, clase=weak_deficit.

## Regiones o anclas con deficit moderado/fuerte
- HD 11506 d en cube33_cluster3: moderate_deficit con delta_rel_best=0.333.
- K2-147 b en cube1_cluster4: moderate_deficit con delta_rel_best=0.333.
- GJ 3988 b en cube8_cluster2: moderate_deficit con delta_rel_best=0.333.
- K2-147 b en cube6_cluster3: moderate_deficit con delta_rel_best=0.333.
- HD 11977 b en cube17_cluster3: moderate_deficit con delta_rel_best=0.333.
- GJ 667 C b en cube13_cluster9: moderate_deficit con delta_rel_best=0.333.
- TOI-1680 b en cube8_cluster1: moderate_deficit con delta_rel_best=0.333.
- TOI-1680 b en cube7_cluster8: moderate_deficit con delta_rel_best=0.333.
- GJ 667 C b en cube7_cluster7: moderate_deficit con delta_rel_best=0.333.
- GJ 3988 b en cube7_cluster9: moderate_deficit con delta_rel_best=0.333.
- HD 111591 b en cube17_cluster8: moderate_deficit con delta_rel_best=0.333.
- HD 86950 b en cube17_cluster11: moderate_deficit con delta_rel_best=0.333.
- HD 11977 b en cube24_cluster2: moderate_deficit con delta_rel_best=0.333.
- TYC 2187-512-1 b en cube19_cluster6: moderate_deficit con delta_rel_best=0.333.
- HD 86950 b en cube18_cluster3: moderate_deficit con delta_rel_best=0.333.
- HD 11977 b en cube18_cluster1: moderate_deficit con delta_rel_best=0.333.
- HD 111591 b en cube18_cluster2: moderate_deficit con delta_rel_best=0.333.
- HD 86950 b en cube24_cluster6: moderate_deficit con delta_rel_best=0.333.
- HD 111591 b en cube24_cluster5: moderate_deficit con delta_rel_best=0.333.
- HD 86950 b en cube25_cluster8: moderate_deficit con delta_rel_best=0.333.
- HD 111591 b en cube25_cluster7: moderate_deficit con delta_rel_best=0.333.
- HD 11977 b en cube25_cluster3: moderate_deficit con delta_rel_best=0.333.
- TYC 2187-512-1 b en cube26_cluster9: moderate_deficit con delta_rel_best=0.333.

## Si la mayoria son RV
Metodos dominantes mas frecuentes entre regiones high TOI: Radial Velocity=11, Transit=2. Si predomina Radial Velocity, la direccion esperada de incompletitud se interpreta prudentemente hacia menor masa planetaria o menor proxy RV a escala orbital comparable.

## Advertencia sobre delta_rel_neighbors_best
El valor best es util para priorizacion, pero puede inflar la lectura si se interpreta aislado. Conviene revisarlo junto con el promedio, la mediana y el detalle por radio en anchor_neighbor_deficits.csv.

## Advertencia general
Estos resultados son rankings de submuestreo topologico y priorizacion observacional. No equivalen a una conclusion cerrada sobre objetos ausentes.

Frase final para presentacion: TOI y ATI no descubren planetas faltantes; construyen un ranking topologico de regiones y planetas ancla donde el catalogo parece observacionalmente incompleto.
