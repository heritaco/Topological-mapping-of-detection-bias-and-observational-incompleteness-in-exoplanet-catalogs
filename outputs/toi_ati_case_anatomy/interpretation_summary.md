# TOI/ATI case anatomy summary

- La region con mayor prioridad regional es cube12_cluster0, con TOI=0.07667. La lectura debe descomponerse en sombra, imputacion R^3, continuidad fisica y soporte de red antes de convertirla en conclusion astrofisica.
- El ancla con mayor ATI es HIP 97166 c / cube12_cluster0, con ATI=0.006853. Este valor prioriza inspeccion, no prueba objetos ausentes.
- El criterio central del reporte es explicar por que gana cada caso: que factor multiplica el indice, que factor lo limita y que tan robusto es el deficit por radio.

## Top regions
| node_id | TOI | shadow_score | I_R3 | C_phys | S_net | top_method |
| --- | --- | --- | --- | --- | --- | --- |
| cube12_cluster0 | 0.0766674525511436 | 0.0854837489037139 | 0.0030379216425728 | 0.9573467807645072 | 0.9396790676659268 | Transit |
| cube17_cluster2 | 0.0627572653395493 | 0.2740494357573514 | 0.0 | 0.7199315948181663 | 0.3180854949506049 | Radial Velocity |
| cube26_cluster6 | 0.0581132674842715 | 0.2398518871206464 | 0.0 | 0.7162596312006045 | 0.3382685955462278 | Radial Velocity |
| cube19_cluster5 | 0.0581132674842715 | 0.2398518871206464 | 0.0 | 0.7162596312006045 | 0.3382685955462278 | Radial Velocity |
| cube19_cluster4 | 0.0578280248129117 | 0.2427041750232117 | 0.0 | 0.7959633639616236 | 0.2993422596638316 | Radial Velocity |
| cube26_cluster4 | 0.0578280248129117 | 0.2427041750232117 | 0.0 | 0.7959633639616236 | 0.2993422596638316 | Radial Velocity |
| cube25_cluster0 | 0.0562810757788033 | 0.1564580961262041 | 0.1182795698924731 | 0.4953020073966288 | 0.8236891721311786 | Radial Velocity |
| cube24_cluster1 | 0.0523779284684389 | 0.2424045402392323 | 0.0 | 0.7218376584728403 | 0.2993422596638316 | Radial Velocity |
| cube26_cluster0 | 0.044261164864001 | 0.1975304009906056 | 0.0 | 0.9220571787615326 | 0.2430138582859247 | Radial Velocity |
| cube26_cluster10 | 0.0435710279764328 | 0.2011009153469986 | 0.0 | 0.795866096132521 | 0.2722348716114531 | Radial Velocity |

## Top anchors
| anchor_pl_name | node_id | ATI | TOI | delta_rel_neighbors_best | deficit_class |
| --- | --- | --- | --- | --- | --- |
| HIP 97166 c | cube12_cluster0 | 0.0068528480823882 | 0.0766674525511436 | 0.0909090909008264 | no_deficit |
| HIP 90988 b | cube17_cluster2 | 0.0067455387709832 | 0.0627572653395493 | 0.1111111110987654 | weak_deficit |
| HD 42012 b | cube26_cluster6 | 0.0051133340600407 | 0.0581132674842715 | 0.0909090909008264 | no_deficit |
| HD 42012 b | cube19_cluster5 | 0.0051133340600407 | 0.0581132674842715 | 0.0909090909008264 | no_deficit |
| HD 4313 b | cube17_cluster10 | 0.0050935128729779 | 0.0343040610223196 | 0.1666666666388888 | weak_deficit |
| HD 221585 b | cube25_cluster0 | 0.005070342812419 | 0.0562810757788033 | 0.0909090909008264 | no_deficit |
| 24 Sex b | cube24_cluster1 | 0.0046807935463178 | 0.0523779284684389 | 0.0909090909008264 | no_deficit |
| BD-17 63 b | cube26_cluster4 | 0.0046770858527751 | 0.0578280248129117 | 0.0909090909008264 | no_deficit |
| BD-17 63 b | cube19_cluster4 | 0.0046770858527751 | 0.0578280248129117 | 0.0909090909008264 | no_deficit |
| HD 11506 d | cube33_cluster3 | 0.0043487335823949 | 0.0137730505169408 | 0.3333333332222222 | moderate_deficit |

## Top anchors with radius-by-radius deficit
Estas tablas evitan sobreinterpretar `delta_rel_neighbors_best`: el valor maximo resume, pero no reemplaza, la lectura por los tres radios locales.
| anchor_pl_name | node_id | ATI | best_radius_type | Delta_rel_neighbors_best | mean_Delta_rel_neighbors | median_Delta_rel_neighbors | deficit_stability_label | interpretation_short |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HIP 97166 c | cube12_cluster0 | 0.0068528480823882 | r_kNN | 0.09090909090082644 | -0.21828028293803184 | -0.33228840125374487 | radius_sensitive_deficit | El deficit depende del radio y debe leerse como exploratorio. |
| HIP 90988 b | cube17_cluster2 | 0.0067455387709832 | r_node_median | 0.11111111109876542 | 0.0726310726241186 | 0.09090909090082644 | consistent_positive_deficit | El deficit aparece en las tres escalas locales y resulta mas estable. |
| HD 42012 b | cube19_cluster5 | 0.0051133340600407 | r_kNN | 0.09090909090082644 | 0.034191638084960535 | 0.008264462809849054 | consistent_positive_deficit | El deficit aparece en las tres escalas locales y resulta mas estable. |
| HD 42012 b | cube26_cluster6 | 0.0051133340600407 | r_kNN | 0.09090909090082644 | 0.034191638084960535 | 0.008264462809849054 | consistent_positive_deficit | El deficit aparece en las tres escalas locales y resulta mas estable. |
| HD 4313 b | cube17_cluster10 | 0.0050935128729779 | r_node_median | 0.16666666663888888 | 0.11363636362203472 | 0.09090909090082644 | consistent_positive_deficit | El deficit aparece en las tres escalas locales y resulta mas estable. |

## Three final presentation cases
| case_type | anchor_pl_name | node_id | TOI | ATI | Delta_rel_neighbors_best | deficit_stability_label | how_to_present | caution_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_toi_region |  | cube12_cluster0 | 0.0766674525511436 |  |  | not_anchor_case | Presentar como la region Mapper con mayor prioridad regional. Explicar que gana por la combinacion de sombra observacional, baja imputacion, continuidad fisica y soporte de red. | El caso regional no afirma objetos ausentes; prioriza una zona Mapper para inspeccion observacional. |
| top_ati_anchor | HIP 97166 c | cube12_cluster0 | 0.0766674525511436 | 0.0068528480823882 | 0.0909090909008264 | radius_sensitive_deficit | Presentar como el planeta ancla mas prioritario para inspeccion local. Explicar que combina una region TOI alta con deficit local y baja imputacion. | El caso ancla no afirma un objeto ausente; prioriza una vecindad local en R^3. |
| repeated_anchor_multi_node | HD 42012 b | cube26_cluster6 | 0.0581132674842715 | 0.0051133340600407 | 0.0909090909008264 | consistent_positive_deficit | Presentar como un caso de transicion Mapper. El planeta aparece en varios nodos por el solapamiento de la cubierta, lo que puede indicar que vive en una frontera topologica entre vecindarios. | El caso repetido no es duplicacion erronea; puede reflejar solapamiento de cubiertas y transicion topologica. |

## Recommended presentation sequence
1. Explicar primero la region top por TOI para abrir el indice regional.
2. Explicar despues el ancla top por ATI para aterrizar la priorizacion local en R^3.
3. Explicar al final el ancla repetida en varios nodos como caso de transicion Mapper por solapamiento de cubiertas.
4. Cerrar con limitaciones: no hay completitud instrumental, no hay confirmacion de objetos ausentes y el deficit puede ser sensible al radio.

## Deficit formula audit
- Raw `delta_rel` > 1 count: 0
- Recomputed `Delta_rel` > 1 count: 0
- Formula mismatches recomputed: 0
- La auditoria no encontro `Delta_rel` recomputado mayor que 1. Si una figura previa parecia exceder 1, la lectura mas probable es que estaba mostrando `Delta_N` o una escala mal rotulada.

## Figure 5 audit
- Previous y column: summary_radius_columns
- Previous y max: 0.3333333332222222
- Recomputed Delta_rel max: 0.3333333332222222
- Recomputed Delta_N max: 1.0
- Decision: plot_delta_rel
- Reason: La version corregida separa el deficit relativo recomputado desde N_obs y N_exp_neighbors del deficit absoluto, para evitar confundir escalas de conteo con fracciones normalizadas.
- Esta versión separa explícitamente déficit relativo y déficit absoluto. El primero sirve para comparar radios con escalas distintas; el segundo depende fuertemente del tamaño de la bola local.
- Ninguno debe interpretarse como cantidad confirmada de planetas ausentes.

## Caution
TOI/ATI no descubre planetas ausentes; prioriza donde buscar evidencia de incompletitud observacional.