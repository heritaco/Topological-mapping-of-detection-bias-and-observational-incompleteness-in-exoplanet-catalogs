# Final Region Synthesis

This file classifies selected `pca2` Mapper nodes and connected components as `physical`, `observational`, `mixed`, or `weak` using explicit rule-based evidence.

Permutation `n_perm` values detected: 1000. The available permutation audit appears to be the final 1000-permutation run.

## Overall Synthesis

| config_id | region_type | final_label | n_regions |
| --- | --- | --- | --- |
| joint_no_density_pca2_cubes10_overlap0p35 | component | observational | 1 |
| joint_no_density_pca2_cubes10_overlap0p35 | component | weak | 8 |
| joint_no_density_pca2_cubes10_overlap0p35 | node | mixed | 26 |
| joint_no_density_pca2_cubes10_overlap0p35 | node | physical | 1 |
| joint_no_density_pca2_cubes10_overlap0p35 | node | weak | 35 |
| joint_pca2_cubes10_overlap0p35 | component | mixed | 3 |
| joint_pca2_cubes10_overlap0p35 | component | weak | 4 |
| joint_pca2_cubes10_overlap0p35 | node | mixed | 43 |
| joint_pca2_cubes10_overlap0p35 | node | weak | 13 |
| orbital_pca2_cubes10_overlap0p35 | component | mixed | 12 |
| orbital_pca2_cubes10_overlap0p35 | component | observational | 1 |
| orbital_pca2_cubes10_overlap0p35 | component | weak | 11 |
| orbital_pca2_cubes10_overlap0p35 | node | mixed | 80 |
| orbital_pca2_cubes10_overlap0p35 | node | weak | 44 |
| phys_density_pca2_cubes10_overlap0p35 | component | mixed | 9 |
| phys_density_pca2_cubes10_overlap0p35 | component | weak | 8 |
| phys_density_pca2_cubes10_overlap0p35 | node | mixed | 38 |
| phys_density_pca2_cubes10_overlap0p35 | node | physical | 4 |
| phys_density_pca2_cubes10_overlap0p35 | node | weak | 25 |
| phys_min_pca2_cubes10_overlap0p35 | component | mixed | 10 |
| phys_min_pca2_cubes10_overlap0p35 | component | weak | 5 |
| phys_min_pca2_cubes10_overlap0p35 | node | mixed | 44 |
| phys_min_pca2_cubes10_overlap0p35 | node | observational | 1 |
| phys_min_pca2_cubes10_overlap0p35 | node | physical | 5 |
| phys_min_pca2_cubes10_overlap0p35 | node | weak | 16 |
| thermal_pca2_cubes10_overlap0p35 | component | mixed | 2 |
| thermal_pca2_cubes10_overlap0p35 | component | observational | 2 |
| thermal_pca2_cubes10_overlap0p35 | component | weak | 34 |
| thermal_pca2_cubes10_overlap0p35 | node | mixed | 9 |
| thermal_pca2_cubes10_overlap0p35 | node | weak | 112 |

## Orbital Mapper Synthesis

| region_type | region_id | final_label | confidence | n_members | dominant_discoverymethod | dominant_discoverymethod_fraction | physical_evidence_score | observational_bias_score | imputation_risk_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| component | 12 | mixed | high | 5 | Radial Velocity | 1.000 | 1.000 | 1.000 | 0.500 |
| component | 13 | mixed | high | 5 | Radial Velocity | 0.800 | 1.000 | 1.000 | 0.500 |
| component | 16 | mixed | high | 10 | Radial Velocity | 1.000 | 1.000 | 1.000 | 0.250 |
| component | 19 | mixed | high | 22 | Radial Velocity | 0.682 | 1.000 | 1.000 | 0.667 |
| component | 21 | mixed | high | 7 | Radial Velocity | 1.000 | 1.000 | 1.000 | 0.500 |
| component | 17 | mixed | high | 33 | Radial Velocity | 0.848 | 0.970 | 1.000 | 0.667 |
| component | 20 | mixed | high | 9 | Radial Velocity | 0.667 | 0.889 | 1.000 | 0.667 |
| component | 22 | mixed | high | 6 | Radial Velocity | 0.833 | 0.833 | 1.000 | 0.667 |
| component | 2 | mixed | high | 17 | Transit | 1.000 | 0.824 | 1.000 | 0.250 |
| node | cube0_cluster0 | mixed | high | 36 | Transit | 0.972 | 1.000 | 1.000 | 0.000 |
| node | cube0_cluster1 | mixed | high | 12 | Transit | 1.000 | 1.000 | 1.000 | 0.250 |
| node | cube0_cluster2 | mixed | high | 5 | Transit | 1.000 | 1.000 | 1.000 | 0.500 |

## Most Physically Interpretable Regions

| config_id | region_type | region_id | confidence | physical_evidence_score | observational_bias_score | mean_imputation_fraction | radius_class_dominant | orbit_class_dominant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phys_density_pca2_cubes10_overlap0p35 | node | cube14_cluster0 | medium | 1.000 | 0.503 | 0.000 | jovian_size | short_period |
| phys_density_pca2_cubes10_overlap0p35 | node | cube7_cluster0 | medium | 1.000 | 0.516 | 0.000 | jovian_size | short_period |
| phys_min_pca2_cubes10_overlap0p35 | node | cube62_cluster0 | medium | 1.000 | 0.516 | 0.000 | jovian_size | short_period |
| phys_min_pca2_cubes10_overlap0p35 | node | cube63_cluster0 | medium | 1.000 | 0.574 | 0.000 | jovian_size | short_period |
| joint_no_density_pca2_cubes10_overlap0p35 | node | cube26_cluster0 | medium | 1.000 | 0.586 | 0.092 | rocky_size | intermediate_period |
| phys_min_pca2_cubes10_overlap0p35 | node | cube53_cluster0 | medium | 0.995 | 0.489 | 0.000 | jovian_size | short_period |
| phys_min_pca2_cubes10_overlap0p35 | node | cube43_cluster0 | medium | 0.975 | 0.537 | 0.000 | jovian_size | long_period |
| phys_min_pca2_cubes10_overlap0p35 | node | cube52_cluster0 | medium | 0.934 | 0.548 | 0.000 | jovian_size | short_period |

## Observationally Suspicious Regions

| config_id | region_type | region_id | confidence | dominant_discoverymethod | dominant_discoverymethod_fraction | discoverymethod_enrichment_z | discoverymethod_enrichment_p | observational_bias_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phys_min_pca2_cubes10_overlap0p35 | node | cube21_cluster0 | high | Transit | 0.922 | 26.793 | 0.001 | 1.000 |
| orbital_pca2_cubes10_overlap0p35 | component | 3 | high | Transit | 1.000 | NA | NA | 1.000 |
| thermal_pca2_cubes10_overlap0p35 | component | 25 | high | Transit | 1.000 | NA | NA | 1.000 |
| thermal_pca2_cubes10_overlap0p35 | component | 36 | high | Transit | 1.000 | NA | NA | 1.000 |
| joint_no_density_pca2_cubes10_overlap0p35 | component | 0 | medium | Transit | 0.750 | NA | NA | 0.750 |

## Mixed Regions

| config_id | region_type | region_id | confidence | physical_evidence_score | observational_bias_score | dominant_discoverymethod | radius_class_dominant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| joint_no_density_pca2_cubes10_overlap0p35 | node | cube12_cluster1 | high | 1.000 | 1.000 | Transit | rocky_size |
| joint_no_density_pca2_cubes10_overlap0p35 | node | cube16_cluster0 | high | 1.000 | 1.000 | Transit | jovian_size |
| joint_no_density_pca2_cubes10_overlap0p35 | node | cube1_cluster0 | high | 1.000 | 1.000 | Transit | rocky_size |
| joint_no_density_pca2_cubes10_overlap0p35 | node | cube2_cluster0 | high | 1.000 | 1.000 | Transit | rocky_size |
| joint_no_density_pca2_cubes10_overlap0p35 | node | cube2_cluster1 | high | 1.000 | 1.000 | Transit | rocky_size |
| joint_no_density_pca2_cubes10_overlap0p35 | node | cube2_cluster2 | high | 1.000 | 1.000 | Transit | rocky_size |
| joint_no_density_pca2_cubes10_overlap0p35 | node | cube2_cluster3 | high | 1.000 | 1.000 | Transit | rocky_size |
| joint_no_density_pca2_cubes10_overlap0p35 | node | cube35_cluster0 | medium | 1.000 | 1.000 | Microlensing | jovian_size |

## Regions Not To Overinterpret

| config_id | region_type | region_id | confidence | n_members | imputation_risk_score | mean_imputation_fraction | dominant_discoverymethod | physical_evidence_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thermal_pca2_cubes10_overlap0p35 | component | 9 | high | 3 | 1.000 | 0.833 | Radial Velocity | 1.000 |
| thermal_pca2_cubes10_overlap0p35 | node | cube12_cluster8 | high | 3 | 1.000 | 0.833 | Radial Velocity | 1.000 |
| thermal_pca2_cubes10_overlap0p35 | node | cube13_cluster4 | high | 3 | 1.000 | 0.833 | Radial Velocity | 1.000 |
| thermal_pca2_cubes10_overlap0p35 | node | cube18_cluster9 | high | 3 | 1.000 | 0.833 | Radial Velocity | 1.000 |
| thermal_pca2_cubes10_overlap0p35 | node | cube19_cluster5 | high | 3 | 1.000 | 0.833 | Radial Velocity | 1.000 |
| joint_no_density_pca2_cubes10_overlap0p35 | component | 8 | high | 4 | 1.000 | 0.500 | Imaging | 1.000 |
| joint_no_density_pca2_cubes10_overlap0p35 | node | cube23_cluster1 | high | 4 | 1.000 | 0.333 | Radial Velocity | 1.000 |
| joint_no_density_pca2_cubes10_overlap0p35 | node | cube24_cluster0 | high | 4 | 1.000 | 0.333 | Radial Velocity | 1.000 |

## Interpretation Caution

This is an evidence classification based on topology, imputation, physical coherence, and discovery-method enrichment. It is not causal proof that discovery method created a region, nor proof that a region is an astrophysical class.

## LaTeX Conclusion Paragraph

The final region synthesis classifies selected Mapper regions using four evidence streams: topology-derived regions, imputation confidence, heuristic physical coherence, and discovery-method enrichment. The strongest conclusion is not that Mapper has discovered causal astrophysical classes, but that some regions are physically interpretable, some are observationally suspicious, and several are mixed. This classification should be read as reproducible evidence triage for scientific follow-up rather than as proof of physical taxonomy.
