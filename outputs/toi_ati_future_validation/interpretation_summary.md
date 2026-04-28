# TOI/ATI future validation summary

## Resumen ejecutivo
Esta fase no crea otro indice principal. Toma TOI/ATI y los convierte en un sistema de priorizacion mas auditado al penalizar sensibilidad al radio y separar casos exploratorios de casos mas estables.

## Que problema corrige esta fase
- ATI original puede privilegiar un delta_rel_best alto aunque el promedio por radios sea bajo o negativo.
- La nueva lectura usa estabilidad por radio, penalizacion conservadora e inspeccion de cambios de ranking.

## Diferencia entre ATI original y ATI conservador
- ATI original resume prioridad local con TOI, deficit best, baja imputacion y representatividad.
- ATI conservador penaliza deficit sensible al radio y fragilidad en radios grandes.

## Casos que bajan al penalizar sensibilidad al radio
| anchor_pl_name | node_id | ATI_original | ATI_conservative | rank_shift | deficit_stability_class |
| --- | --- | --- | --- | --- | --- |
| HD 221585 b | cube25_cluster0 | 0.005070342812419 | 0.0008450571354031667 | 39.0 | unstable_due_to_large_radius |
| GJ 3942 b | cube6_cluster0 | 0.0035468622316803 | 0.00059114370528005 | 39.0 | unstable_due_to_large_radius |
| HIP 97166 c | cube12_cluster0 | 0.0068528480823882 | 0.0011421413470647 | 38.0 | unstable_due_to_large_radius |
| BD-17 63 b | cube26_cluster4 | 0.0046770858527751 | 0.0013833234960151923 | 25.0 | unstable_due_to_large_radius |
| BD-17 63 b | cube19_cluster4 | 0.0046770858527751 | 0.0013833234960151923 | 25.0 | unstable_due_to_large_radius |

## Casos que suben o se mantienen por estabilidad
| anchor_pl_name | node_id | ATI_original | ATI_conservative | stable_deficit_score | deficit_stability_class |
| --- | --- | --- | --- | --- | --- |
| HIP 90988 b | cube17_cluster2 | 0.0067455387709832 | 0.006389466032847364 | 0.0726310726241186 | small_but_stable_deficit |
| HD 42012 b | cube26_cluster6 | 0.0051133340600407 | 0.0051133340600407 | 0.034191638084960535 | small_but_stable_deficit |
| HD 42012 b | cube19_cluster5 | 0.0051133340600407 | 0.0051133340600407 | 0.034191638084960535 | small_but_stable_deficit |
| HD 4313 b | cube17_cluster10 | 0.0050935128729779 | 0.004347585182760773 | 0.11363636362203472 | stable_positive_deficit |
| 24 Sex b | cube24_cluster1 | 0.0046807935463178 | 0.004153260318351299 | 0.042956929068832965 | small_but_stable_deficit |

## Cinco casos finales recomendados
| case_type | anchor_pl_name | node_id | ATI_original | ATI_conservative | deficit_stability_class | how_to_present |
| --- | --- | --- | --- | --- | --- | --- |
| top_toi_region |  | cube12_cluster0 |  |  |  | Usar para explicar el indice regional y sus factores. |
| top_ati_original_anchor | HIP 97166 c | cube12_cluster0 | 0.0068528480823882 | 0.0011421413470647 | unstable_due_to_large_radius | Usar para mostrar como el ranking original prioriza un caso local. |
| top_ati_conservative_anchor | HIP 90988 b | cube17_cluster2 | 0.0067455387709832 | 0.006389466032847364 | small_but_stable_deficit | Usar para mostrar la version mas prudente del ranking local. |
| repeated_anchor_transition_case | HD 217107 c | cube26_cluster2 | 0.0023043105024573 | 0.0023043105024573 | stable_positive_deficit | Usar para explicar por que una ancla puede vivir en la frontera entre vecindarios Mapper. |
| stable_deficit_anchor | K2-147 b | cube1_cluster4 | 0.0 | 0.0 | stable_positive_deficit | Usar para contrastar un caso estable frente a uno sensible al radio. |

## Tabla breve de prioridad observacional
| anchor_pl_name | node_id | observational_priority_score | ATI_conservative | TOI | stable_deficit_score | I_R3 | method | expected_incompleteness_direction | reason_for_priority | caution_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HIP 90988 b | cube17_cluster2 | 0.7016600163843385 | 0.006389466032847364 | 0.0627572653395493 | 0.0726310726241186 |  | Radial Velocity |  | Combina TOI alto, ATI conservador, estabilidad razonable y una direccion fisica interpretable para inspeccion futura. | Caso util para inspeccion futura, sin afirmar objetos ausentes. |
| HD 42012 b | cube26_cluster6 | 0.5915419310204659 | 0.0051133340600407 | 0.0581132674842715 | 0.034191638084960535 |  | Radial Velocity |  | Combina TOI alto, ATI conservador, estabilidad razonable y una direccion fisica interpretable para inspeccion futura. | Caso util para inspeccion futura, sin afirmar objetos ausentes. |
| HD 42012 b | cube19_cluster5 | 0.5915419310204659 | 0.0051133340600407 | 0.0581132674842715 | 0.034191638084960535 |  | Radial Velocity |  | Combina TOI alto, ATI conservador, estabilidad razonable y una direccion fisica interpretable para inspeccion futura. | Caso util para inspeccion futura, sin afirmar objetos ausentes. |
| HD 11506 d | cube33_cluster3 | 0.5296734924093389 | 0.003711876294040001 | 0.0137730505169408 | 0.277777777699074 |  | Radial Velocity |  | Combina TOI alto, ATI conservador, estabilidad razonable y una direccion fisica interpretable para inspeccion futura. | Caso util para inspeccion futura, sin afirmar objetos ausentes. |
| 24 Sex b | cube24_cluster1 | 0.5261169309169348 | 0.004153260318351299 | 0.0523779284684389 | 0.042956929068832965 |  | Radial Velocity |  | Combina TOI alto, ATI conservador, estabilidad razonable y una direccion fisica interpretable para inspeccion futura. | Caso util para inspeccion futura, sin afirmar objetos ausentes. |
| HD 4313 b | cube17_cluster10 | 0.5240063824960591 | 0.004347585182760773 | 0.0343040610223196 | 0.11363636362203472 |  | Radial Velocity |  | Combina TOI alto, ATI conservador, estabilidad razonable y una direccion fisica interpretable para inspeccion futura. | Caso util para inspeccion futura, sin afirmar objetos ausentes. |
| HD 167677 b | cube26_cluster0 | 0.48509901359263424 | 0.0039952396812571 | 0.044261164864001 | 0.03413828632129996 |  | Radial Velocity |  | Combina TOI alto, ATI conservador, estabilidad razonable y una direccion fisica interpretable para inspeccion futura. | Caso util para inspeccion futura, sin afirmar objetos ausentes. |
| HD 13931 b | cube26_cluster11 | 0.45129536445093454 | 0.0037677504861406406 | 0.0202388317156348 | 0.12222222220374486 |  | Radial Velocity |  | Combina TOI alto, ATI conservador, estabilidad razonable y una direccion fisica interpretable para inspeccion futura. | Caso util para inspeccion futura, sin afirmar objetos ausentes. |
| Kepler-48 e | cube26_cluster12 | 0.4310846115253527 | 0.0033651953995414 | 0.0384850826108858 | 0.03313711939068661 |  | Radial Velocity |  | Combina TOI alto, ATI conservador, estabilidad razonable y una direccion fisica interpretable para inspeccion futura. | Caso util para inspeccion futura, sin afirmar objetos ausentes. |
| Kepler-48 e | cube19_cluster8 | 0.4310846115253527 | 0.0033651953995414 | 0.0384850826108858 | 0.03313711939068661 |  | Radial Velocity |  | Combina TOI alto, ATI conservador, estabilidad razonable y una direccion fisica interpretable para inspeccion futura. | Caso util para inspeccion futura, sin afirmar objetos ausentes. |

## Casos exploratorios
| anchor_pl_name | node_id | deficit_stability_class | ATI_conservative | caution_text |
| --- | --- | --- | --- | --- |
| HIP 97166 c | cube12_cluster0 | unstable_due_to_large_radius | 0.0011421413470647 | Deficit sensible al radio; usar como caso exploratorio. |
| HD 221585 b | cube25_cluster0 | unstable_due_to_large_radius | 0.0008450571354031667 | Deficit sensible al radio; usar como caso exploratorio. |
| BD-17 63 b | cube26_cluster4 | unstable_due_to_large_radius | 0.0013833234960151923 | Deficit sensible al radio; usar como caso exploratorio. Nodo pequeno; el soporte local puede ser fragil. |
| BD-17 63 b | cube19_cluster4 | unstable_due_to_large_radius | 0.0013833234960151923 | Deficit sensible al radio; usar como caso exploratorio. Nodo pequeno; el soporte local puede ser fragil. |
| GJ 3942 b | cube6_cluster0 | unstable_due_to_large_radius | 0.00059114370528005 | Deficit sensible al radio; usar como caso exploratorio. |
| Gl 49 b | cube13_cluster0 | unstable_due_to_large_radius | 0.0009744350221394666 | Deficit sensible al radio; usar como caso exploratorio. |
| HIP 38594 b | cube19_cluster3 | radius_sensitive_deficit | 0.0015865168606890667 | Deficit sensible al radio; usar como caso exploratorio. |
| GJ 536 c | cube13_cluster1 | unstable_due_to_large_radius | 0.0003550826467320333 | Deficit sensible al radio; usar como caso exploratorio. |
| HD 109749 b | cube5_cluster0 | unstable_due_to_large_radius | 0.00033334311333739995 | Deficit sensible al radio; usar como caso exploratorio. |
| HD 40307 f | cube13_cluster3 | unstable_due_to_large_radius | 0.0006368765222770332 | Deficit sensible al radio; usar como caso exploratorio. |

## Casos robustos
| anchor_pl_name | node_id | deficit_stability_class | ATI_conservative | caution_text |
| --- | --- | --- | --- | --- |
| HIP 90988 b | cube17_cluster2 | small_but_stable_deficit | 0.006389466032847364 | Nodo pequeno; el soporte local puede ser fragil. |
| HD 42012 b | cube26_cluster6 | small_but_stable_deficit | 0.0051133340600407 | Caso util para priorizacion observacional prudente. |
| HD 42012 b | cube19_cluster5 | small_but_stable_deficit | 0.0051133340600407 | Caso util para priorizacion observacional prudente. |
| HD 4313 b | cube17_cluster10 | stable_positive_deficit | 0.004347585182760773 | Nodo pequeno; el soporte local puede ser fragil. |
| 24 Sex b | cube24_cluster1 | small_but_stable_deficit | 0.004153260318351299 | Nodo pequeno; el soporte local puede ser fragil. |
| HD 11506 d | cube33_cluster3 | stable_positive_deficit | 0.003711876294040001 | Nodo pequeno; el soporte local puede ser fragil. |
| HD 167677 b | cube26_cluster0 | small_but_stable_deficit | 0.0039952396812571 | Caso util para priorizacion observacional prudente. |
| HD 13931 b | cube26_cluster11 | stable_positive_deficit | 0.0037677504861406406 | Nodo pequeno; el soporte local puede ser fragil. |
| Kepler-48 e | cube19_cluster8 | small_but_stable_deficit | 0.0033651953995414 | Caso util para priorizacion observacional prudente. |
| Kepler-48 e | cube26_cluster12 | small_but_stable_deficit | 0.0033651953995414 | Caso util para priorizacion observacional prudente. |

## Trabajo futuro
- Incorporar completitud instrumental por metodo de descubrimiento.
- Validar con catalogos futuros y con inyeccion-recuperacion sintetica.
- Agregar propiedades estelares para mejorar el contexto fisico y el proxy de detectabilidad dinamica.

## Advertencia de lenguaje prudente
TOI/ATI no detecta planetas ausentes; prioriza regiones y anclas donde el catalogo parece observacionalmente incompleto bajo una referencia topologica local.

## Casos para auditoria tecnica
| issue_type | anchor_pl_name | node_id | ATI_original | ATI_conservative | delta_rel_best | delta_rel_mean | r3_imputation_score | n_members | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_ATI_but_radius_sensitive | HIP 97166 c | cube12_cluster0 | 0.0068528480823882 | 0.0011421413470647 | 0.0909090909008264 | -0.21828028293803184 | 0.0 | 3182.0 | ATI alto con deficit sensible al radio. |
| small_node_support | HIP 90988 b | cube17_cluster2 | 0.0067455387709832 | 0.006389466032847364 | 0.1111111110987654 | 0.0726310726241186 | 0.0 | 8.0 | El soporte de nodo o de red es pequeno. |
| repeated_anchor_needs_context | HD 42012 b | cube26_cluster6 | 0.0051133340600407 | 0.0051133340600407 | 0.0909090909008264 | 0.034191638084960535 | 0.0 | 11.0 | El ancla aparece en varios nodos y necesita contexto de solapamiento Mapper. |
| repeated_anchor_needs_context | HD 42012 b | cube19_cluster5 | 0.0051133340600407 | 0.0051133340600407 | 0.0909090909008264 | 0.034191638084960535 | 0.0 | 11.0 | El ancla aparece en varios nodos y necesita contexto de solapamiento Mapper. |
| small_node_support | HD 4313 b | cube17_cluster10 | 0.0050935128729779 | 0.004347585182760773 | 0.1666666666388888 | 0.11363636362203472 | 0.0 | 5.0 | El soporte de nodo o de red es pequeno. |
| high_ATI_but_radius_sensitive | HD 221585 b | cube25_cluster0 | 0.005070342812419 | 0.0008450571354031667 | 0.0909090909008264 | -0.11999633315792362 | 0.0 | 527.0 | ATI alto con deficit sensible al radio. |
| small_node_support | 24 Sex b | cube24_cluster1 | 0.0046807935463178 | 0.004153260318351299 | 0.0909090909008264 | 0.042956929068832965 | 0.0 | 6.0 | El soporte de nodo o de red es pequeno. |
| high_ATI_but_radius_sensitive | BD-17 63 b | cube26_cluster4 | 0.0046770858527751 | 0.0013833234960151923 | 0.0909090909008264 | 0.012616638628810059 | 0.0 | 6.0 | ATI alto con deficit sensible al radio. |
| small_node_support | BD-17 63 b | cube26_cluster4 | 0.0046770858527751 | 0.0013833234960151923 | 0.0909090909008264 | 0.012616638628810059 | 0.0 | 6.0 | El soporte de nodo o de red es pequeno. |
| repeated_anchor_needs_context | BD-17 63 b | cube26_cluster4 | 0.0046770858527751 | 0.0013833234960151923 | 0.0909090909008264 | 0.012616638628810059 | 0.0 | 6.0 | El ancla aparece en varios nodos y necesita contexto de solapamiento Mapper. |