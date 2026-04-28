# Interpretacion automatica: sombra observacional

- Nodos con alta sombra y baja imputacion: 13
- Metodos dominantes mas frecuentes entre nodos de alta sombra:
  - Transit: 11
  - Radial Velocity: 2

## Ejemplos de comunidades candidatas

- `phys_min_pca2_cubes10_overlap0p35` / `cube51_cluster1`: Transit, shadow=0.552, n=79. menor radio y periodos mas largos.
- `phys_min_pca2_cubes10_overlap0p35` / `cube69_cluster0`: Transit, shadow=0.452, n=19. menor radio y periodos mas largos.
- `phys_min_pca2_cubes10_overlap0p35` / `cube59_cluster0`: Transit, shadow=0.359, n=4. menor radio y periodos mas largos.
- `phys_min_pca2_cubes10_overlap0p35` / `cube60_cluster2`: Transit, shadow=0.318, n=3. menor radio y periodos mas largos.
- `orbital_pca2_cubes10_overlap0p35` / `cube17_cluster2`: Radial Velocity, shadow=0.274, n=8. masas menores y senales radiales mas debiles.
- `orbital_pca2_cubes10_overlap0p35` / `cube17_cluster10`: Radial Velocity, shadow=0.249, n=5. masas menores y senales radiales mas debiles.
- `orbital_pca2_cubes10_overlap0p35` / `cube24_cluster0`: Radial Velocity, shadow=0.243, n=6. masas menores y senales radiales mas debiles.
- `orbital_pca2_cubes10_overlap0p35` / `cube17_cluster0`: Radial Velocity, shadow=0.243, n=6. masas menores y senales radiales mas debiles.

Estos resultados no prueban planetas faltantes. Senalan regiones topologicas submuestreadas o posibles fronteras de seleccion donde futuras observaciones podrian encontrar vecinos fisico-orbitales similares.

Conclusion: Mapper permite pasar de una auditoria de sesgo a una priorizacion prudente de vecindarios fisico-orbitales potencialmente incompletos.
