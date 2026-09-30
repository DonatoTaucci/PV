# Risultati sperimentali (generati automaticamente)

## 1. Pulizia dati

| Voce | Impianto 1 | Impianto 2 |
|---|---:|---:|
| raw_rows_generation | 68778 | 67698 |
| expected_generation_rows | 71808 | 71808 |
| n_inverters | 22 | 22 |
| days | 34 | 34 |
| median_ac_dc_ratio_raw | 0.098 | 0.978 |
| dc_scale_factor_applied | 0.100 | 1.000 |
| slots_missing_generation_all_inverters | 106 | 5 |
| slots_missing_weather | 82 | 5 |
| slots_generation_without_weather | 1 | 0 |
| slots_weather_without_generation | 25 | 0 |
| weather_slots_imputed_short_gap | 5 | 5 |
| weather_slots_imputed_night | 50 | 0 |
| weather_slots_left_missing | 27 | 0 |
| generation_rows_missing | 3030 | 4110 |
| generation_rows_filled_night_zero | 2165 | 1868 |
| generation_rows_comm_loss_daylight | 865 | 2242 |
| generation_rows_comm_loss_plant_wide | 704 | 88 |
| clock_lag_best_slots | 0 | 0 |
| daylight_rows_zero_dc | 63 | 3774 |
| ac_gt_dc_rows | 18 | 17 |
| total_yield_decreases | 0 | 1161 |
| total_yield_zero_values | 0 | 563 |
| daily_yield_nonzero_at_midnight_frac | 0.215 | 0.690 |
| daily_yield_vs_ac_energy_ratio | 0.989 | 0.994 |
| k_plant_kW_per_kWm2 | 1551.000 | 1583.400 |

**Distribuzione etichette (slot diurni inverter)**

| Classe | Impianto 1 | Impianto 2 |
|---|---:|---:|
| NORMALE | 30653 | 24828 |
| SOTTOPRESTAZIONE | 9 | 370 |
| INVERTER_FERMO | 60 | 3620 |
| PERDITA_COMUNICAZIONE | 865 | 2242 |

## 2. Gemello digitale (potenza attesa normalizzata)

| Scenario | Modello | nRMSE % | MAE | R² | n |
|---|---|---:|---:|---:|---:|
| P1_temporale | lightgbm_normal_rows | 8.51 | 0.0233 | 0.9668 | 9148 |
| P1_temporale | physical_normal_rows | 8.76 | 0.0255 | 0.9649 | 9148 |
| P1_temporale | lightgbm_all_daylight_rows | 11.28 | 0.0248 | 0.9427 | 9174 |
| P2_temporale | lightgbm_normal_rows | 28.41 | 0.0495 | 0.7072 | 8090 |
| P2_temporale | physical_normal_rows | 28.12 | 0.0491 | 0.7132 | 8090 |
| P2_temporale | lightgbm_all_daylight_rows | 68.58 | 0.1032 | -0.1274 | 8998 |
| P1_to_P2_cross | lightgbm_normal_rows | 25.07 | 0.0433 | 0.7691 | 24828 |
| P1_to_P2_cross | physical_normal_rows | 25.42 | 0.0472 | 0.7627 | 24828 |
| P1_to_P2_cross | lightgbm_all_daylight_rows | 73.4 | 0.1241 | -0.1799 | 28818 |
| P2_to_P1_cross | lightgbm_normal_rows | 16.31 | 0.0454 | 0.879 | 30653 |
| P2_to_P1_cross | physical_normal_rows | 11.9 | 0.0387 | 0.9356 | 30653 |
| P2_to_P1_cross | lightgbm_all_daylight_rows | 17.51 | 0.0465 | 0.8621 | 30722 |

## 3. Previsione dello stato a +120 min (4 classi)

| Scenario | Sistema | macro-F1 | MCC | Bal. acc. | Accuracy | n test |
|---|---|---:|---:|---:|---:|---:|
| P1_temporale | hybrid_all | 0.6714 | 0.498 | 0.8271 | 0.9893 | 8998 |
| P1_temporale | persistence_all | 0.7647 | 0.6457 | 0.8289 | 0.9946 | 8998 |
| P2_temporale | hybrid_all | 0.4528 | 0.6127 | 0.6591 | 0.8865 | 8778 |
| P2_temporale | persistence_all | 0.6932 | 0.6577 | 0.6312 | 0.9103 | 8778 |
| P1_to_P2_cross | hybrid_all | 0.7887 | 0.7305 | 0.7437 | 0.8828 | 30290 |
| P1_to_P2_cross | persistence_all | 0.7886 | 0.7317 | 0.7431 | 0.8832 | 30290 |
| P2_to_P1_cross | hybrid_all | 0.5309 | 0.5151 | 0.524 | 0.9587 | 30839 |
| P2_to_P1_cross | persistence_all | 0.5996 | 0.5911 | 0.5095 | 0.9674 | 30839 |

Skill score vs persistenza (macro-F1 / MCC): P1_temporale: -0.3965 / -0.4169, P2_temporale: -0.7836 / -0.1315, P1_to_P2_cross: 0.0005 / -0.0045, P2_to_P1_cross: -0.1716 / -0.1859

## 4. Insorgenza di nuovi guasti (istanti attualmente normali)

| Scenario | Tipo | n pos. | Prevalenza | AP | Lift AP | Precision | Recall | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P1_temporale | electrical | 24 | 0.0027 | 0.1019 | 37.78 | 0.0 | 0.0 | 0.0 |
| P1_temporale | any | 24 | 0.0027 | 0.0062 | 2.31 | 0.0 | 0.0 | -0.0038 |
| P2_temporale | electrical | 648 | 0.0856 | 0.2809 | 3.28 | 0.3781 | 0.3256 | 0.295 |
| P2_temporale | any | 648 | 0.0856 | 0.2495 | 2.92 | 0.342 | 0.3256 | 0.273 |
| P1_to_P2_cross | electrical | 2500 | 0.1093 | 0.2505 | 2.29 | 0.5556 | 0.008 | 0.0568 |
| P1_to_P2_cross | any | 2834 | 0.1239 | 0.1513 | 1.22 | 0.5556 | 0.0071 | 0.052 |
| P2_to_P1_cross | electrical | 104 | 0.0035 | 0.0124 | 3.52 | 0.0323 | 0.0865 | 0.0474 |
| P2_to_P1_cross | any | 826 | 0.0278 | 0.0282 | 1.01 | 0.0358 | 0.0121 | 0.0047 |

**Recall per classe (sistema ibrido, tutti i campioni)**

| Scenario | NORMALE | SOTTOPRESTAZIONE | INVERTER_FERMO | PERDITA_COMUNICAZIONE |
|---|---:|---:|---:|---:|
| P1_temporale | 0.992 (n=8929) | 0.0 (n=0) | 0.489 (n=47) | 1.0 (n=22) |
| P2_temporale | 0.931 (n=7254) | 0.364 (n=33) | 0.683 (n=1491) | 0.0 (n=0) |
| P1_to_P2_cross | 0.982 (n=21441) | 0.588 (n=425) | 0.563 (n=5916) | 0.842 (n=2508) |
| P2_to_P1_cross | 0.985 (n=29310) | 0.2 (n=30) | 0.44 (n=134) | 0.471 (n=1365) |

**Importanza feature (gain, P1→P2)**: SLOTS_SINCE_FAULT 0.15, DELTA_T 0.13, HOUR_SIN 0.13, IRR_DIFF_1H 0.12, IRRADIATION 0.12, R_PEER_STD_1H 0.06, R_PEER_MEAN_24H 0.05, TWIN_RESID_MEAN_24H 0.05, N_COMM_24H 0.04, HOUR_COS 0.04, TWIN_RESID_MEAN_1H 0.03, R_PEER_MEAN_2H 0.02
