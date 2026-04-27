# Mapper Bias Audit Summary

Permutation settings: `n_perm=1000`, `seed=42`.

## Metadata Join Coverage

| config_id | metadata_source | join_method | join_coverage | n_unique_members | n_members_missing_metadata |
| --- | --- | --- | --- | --- | --- |
| phys_min_pca2_cubes10_overlap0p35 | outputs/mapper/data/planet_physical_labels.csv | config_id_then_sample_id_lookup_row_index | 1.000 | 5953 | 0 |
| phys_density_pca2_cubes10_overlap0p35 | outputs/mapper/data/planet_physical_labels.csv | config_id_then_sample_id_lookup_row_index | 1.000 | 5946 | 0 |
| orbital_pca2_cubes10_overlap0p35 | outputs/mapper/data/planet_physical_labels.csv | config_id_then_sample_id_lookup_row_index | 1.000 | 5799 | 0 |
| joint_no_density_pca2_cubes10_overlap0p35 | outputs/mapper/data/planet_physical_labels.csv | config_id_then_sample_id_lookup_row_index | 1.000 | 5862 | 0 |
| joint_pca2_cubes10_overlap0p35 | outputs/mapper/data/planet_physical_labels.csv | config_id_then_sample_id_lookup_row_index | 1.000 | 5843 | 0 |
| thermal_pca2_cubes10_overlap0p35 | outputs/mapper/data/planet_physical_labels.csv | config_id_then_sample_id_lookup_row_index | 1.000 | 5839 | 0 |

## Strongest Graph-Level Enrichment

The strongest graph-level discoverymethod enrichment is `joint_no_density_pca2_cubes10_overlap0p35` (purity z=37.830, NMI z=233.752, observed NMI=0.226).

## Orbital Mapper Assessment

`orbital_pca2_cubes10_overlap0p35` appears strongly dominated by discovery method labels. Observed weighted purity=0.842, observed NMI=0.254, NMI empirical p=0.001.

## Most Observationally Biased Nodes

| config_id | node_id | dominant_discoverymethod | observed_dominant_method_fraction | enrichment_z | empirical_p_value | mean_imputation_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| joint_pca2_cubes10_overlap0p35 | cube61_cluster0 | Imaging | 1.000 | 46.721 | 0.001 | 0.429 |
| joint_no_density_pca2_cubes10_overlap0p35 | cube61_cluster0 | Imaging | 1.000 | 46.556 | 0.001 | 0.500 |
| joint_pca2_cubes10_overlap0p35 | cube62_cluster0 | Imaging | 1.000 | 39.721 | 0.001 | 0.429 |
| joint_pca2_cubes10_overlap0p35 | cube47_cluster0 | Imaging | 1.000 | 39.001 | 0.001 | 0.286 |
| joint_pca2_cubes10_overlap0p35 | cube54_cluster0 | Imaging | 1.000 | 39.001 | 0.001 | 0.286 |

## Most Observationally Biased Components

| config_id | component_id | dominant_discoverymethod | dominant_discoverymethod_fraction | discoverymethod_js_divergence_vs_global | mean_imputation_fraction |
| --- | --- | --- | --- | --- | --- |
| joint_no_density_pca2_cubes10_overlap0p35 | 6 | Imaging | 1.000 | 0.943 | 0.333 |
| joint_no_density_pca2_cubes10_overlap0p35 | 7 | Imaging | 1.000 | 0.943 | 0.500 |
| joint_no_density_pca2_cubes10_overlap0p35 | 8 | Imaging | 1.000 | 0.943 | 0.500 |
| joint_pca2_cubes10_overlap0p35 | 5 | Imaging | 1.000 | 0.943 | 0.286 |
| joint_pca2_cubes10_overlap0p35 | 6 | Imaging | 1.000 | 0.943 | 0.429 |

## Bias And Imputation

104 of 291 high-bias nodes also have mean imputation fraction >= 0.30. This checks whether observational enrichment and imputation risk are co-located.

## More Physically Interpretable Regions

| config_id | node_id | radius_class_dominant | radius_class_purity | dominant_discoverymethod | discoverymethod_js_divergence_vs_global | mean_imputation_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| phys_min_pca2_cubes10_overlap0p35 | cube62_cluster0 | jovian_size | 1.000 | Transit | 0.053 | 0.000 |
| phys_min_pca2_cubes10_overlap0p35 | cube61_cluster0 | jovian_size | 1.000 | Radial Velocity | 0.105 | 0.031 |
| phys_density_pca2_cubes10_overlap0p35 | cube21_cluster0 | jovian_size | 1.000 | Transit | 0.073 | 0.039 |
| phys_min_pca2_cubes10_overlap0p35 | cube13_cluster0 | rocky_size | 1.000 | Transit | 0.060 | 0.000 |
| phys_min_pca2_cubes10_overlap0p35 | cube22_cluster0 | sub_neptune_size | 1.000 | Transit | 0.045 | 0.001 |

## Observationally Suspicious Regions

| config_id | node_id | dominant_discoverymethod | observed_dominant_method_fraction | enrichment_z | empirical_p_value | radius_class_dominant |
| --- | --- | --- | --- | --- | --- | --- |
| joint_pca2_cubes10_overlap0p35 | cube61_cluster0 | Imaging | 1.000 | 46.721 | 0.001 | jovian_size |
| joint_no_density_pca2_cubes10_overlap0p35 | cube61_cluster0 | Imaging | 1.000 | 46.556 | 0.001 | jovian_size |
| joint_pca2_cubes10_overlap0p35 | cube62_cluster0 | Imaging | 1.000 | 39.721 | 0.001 | jovian_size |
| joint_pca2_cubes10_overlap0p35 | cube47_cluster0 | Imaging | 1.000 | 39.001 | 0.001 | jovian_size |
| joint_pca2_cubes10_overlap0p35 | cube54_cluster0 | Imaging | 1.000 | 39.001 | 0.001 | jovian_size |

## Caution

This is a label-permutation bias audit, not causal proof. It keeps the Mapper topology fixed and asks whether observed discovery labels are unusually concentrated relative to random label assignments.

## Report-Ready Paragraph

A label-permutation audit was applied to the selected pca2 Mapper graphs while keeping graph topology and node memberships fixed. The audit compares observed discovery-method concentration against random relabelings of the same planets. It therefore identifies regions where Mapper structure aligns unusually strongly with observational metadata, but it does not prove that discovery method caused the topology.
