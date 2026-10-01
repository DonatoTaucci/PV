# Risultati del modello v1

Tabelle prodotte da `scripts/run_experiments.py`.

Split per giorni {'train': 20, 'val': 7, 'test': 7}, orizzonte 120 min, bootstrap 500 ricampionamenti per giorni.

## 1. Selezione sulla validazione

**P1**: gemello `ibrido_fisico+lightgbm`, classificatore `baseline_logistica_C0.1`, soglia 0.0128 (F1 val 0.0842); eventi di insorgenza in validazione 48, prevalenza 0.0075.

| Gemello | nRMSE % val | R² val |
|---|---:|---:|
| fisico | 10.88 | 0.9456 |
| lightgbm | 10.79 | 0.9465 |
| ibrido_fisico+lightgbm | 10.49 | 0.9494 |

| Classificatore | AP val |
|---|---:|
| baseline_logistica_C0.1 | 0.0409 |
| euristica_1-r_peer_1h | 0.0219 |
| lightgbm_l31_m100_n500 | 0.0124 |
| random_forest_leaf20 | 0.0116 |
| lightgbm_l31_m100_n200 | 0.0112 |
| baseline_logistica_C1 | 0.0109 |
| lightgbm_l15_m100_n200 | 0.0108 |
| random_forest_leaf50 | 0.0106 |
| lightgbm_l15_m20_n500 | 0.0104 |
| lightgbm_l7_m100_n200 | 0.01 |
| lightgbm_l7_m20_n200 | 0.0096 |
| lightgbm_l15_m100_n500 | 0.0096 |
| lightgbm_l7_m100_n500 | 0.0091 |
| lightgbm_l15_m20_n200 | 0.0091 |
| lightgbm_l31_m20_n500 | 0.0089 |
| lightgbm_l7_m20_n500 | 0.0088 |
| lightgbm_l31_m20_n200 | 0.008 |

**P2**: gemello `fisico`, classificatore `random_forest_leaf20`, soglia 0.2291 (F1 val 0.5295); eventi di insorgenza in validazione 661, prevalenza 0.1339.

| Gemello | nRMSE % val | R² val |
|---|---:|---:|
| fisico | 24.39 | 0.7892 |
| lightgbm | 28.78 | 0.7065 |
| ibrido_fisico+lightgbm | 27.01 | 0.7415 |

| Classificatore | AP val |
|---|---:|
| random_forest_leaf20 | 0.5307 |
| random_forest_leaf50 | 0.5293 |
| lightgbm_l15_m20_n200 | 0.4967 |
| lightgbm_l7_m20_n200 | 0.4917 |
| lightgbm_l7_m100_n200 | 0.4842 |
| lightgbm_l15_m20_n500 | 0.4738 |
| lightgbm_l15_m100_n200 | 0.472 |
| lightgbm_l7_m20_n500 | 0.4645 |
| lightgbm_l31_m20_n200 | 0.4619 |
| lightgbm_l31_m20_n500 | 0.4583 |
| lightgbm_l7_m100_n500 | 0.4578 |
| lightgbm_l31_m100_n500 | 0.4553 |
| lightgbm_l31_m100_n200 | 0.4547 |
| lightgbm_l15_m100_n500 | 0.4461 |
| baseline_logistica_C0.1 | 0.4241 |
| baseline_logistica_C1 | 0.4058 |
| euristica_1-r_peer_1h | 0.1905 |

**P1+2**: gemello `fisico`, classificatore `random_forest_leaf50`, soglia 0.2382 (F1 val 0.5067); eventi di insorgenza in validazione 709, prevalenza 0.0624.

| Gemello | nRMSE % val | R² val |
|---|---:|---:|
| fisico | 17.82 | 0.8714 |
| lightgbm | 18.86 | 0.8561 |
| ibrido_fisico+lightgbm | 18.0 | 0.8689 |

| Classificatore | AP val |
|---|---:|
| random_forest_leaf50 | 0.4637 |
| random_forest_leaf20 | 0.4619 |
| lightgbm_l15_m100_n200 | 0.434 |
| lightgbm_l31_m20_n200 | 0.4304 |
| lightgbm_l7_m100_n200 | 0.43 |
| lightgbm_l15_m100_n500 | 0.4236 |
| lightgbm_l31_m100_n200 | 0.4235 |
| lightgbm_l7_m20_n200 | 0.4231 |
| lightgbm_l31_m20_n500 | 0.4211 |
| lightgbm_l15_m20_n200 | 0.4196 |
| lightgbm_l15_m20_n500 | 0.418 |
| lightgbm_l31_m100_n500 | 0.4156 |
| lightgbm_l7_m100_n500 | 0.4137 |
| lightgbm_l7_m20_n500 | 0.4019 |
| baseline_logistica_C0.1 | 0.3479 |
| baseline_logistica_C1 | 0.3316 |
| euristica_1-r_peer_1h | 0.1077 |

## 2. Gemello digitale sul test

| Scenario | Modello | nRMSE % | R² | MAE |
|---|---|---:|---:|---:|
| P1_test | fisico | 8.45 | 0.9669 | 0.0235 |
| P1_test | lightgbm | 7.99 | 0.9704 | 0.0207 |
| P1_test | ibrido_fisico+lightgbm | 7.75 | 0.9722 | 0.0198 |
| P2_test | fisico | 31.41 | 0.6242 | 0.0471 |
| P2_test | lightgbm | 31.33 | 0.626 | 0.048 |
| P2_test | ibrido_fisico+lightgbm | 31.28 | 0.6273 | 0.0468 |
| P1+2_test | fisico | 20.49 | 0.8349 | 0.0357 |
| P1+2_test | lightgbm | 20.9 | 0.8284 | 0.0346 |
| P1+2_test | ibrido_fisico+lightgbm | 20.58 | 0.8335 | 0.0336 |
| P1_to_P2 | fisico | 25.42 | 0.7627 | 0.0472 |
| P1_to_P2 | lightgbm | 25.07 | 0.7691 | 0.0433 |
| P1_to_P2 | ibrido_fisico+lightgbm | 25.06 | 0.7693 | 0.0432 |
| P2_to_P1 | fisico | 11.9 | 0.9356 | 0.0387 |
| P2_to_P1 | lightgbm | 16.31 | 0.879 | 0.0454 |
| P2_to_P1 | ibrido_fisico+lightgbm | 14.13 | 0.9092 | 0.0415 |

## 3. Insorgenza di guasti elettrici (metrica primaria: AUC-PR)

| Scenario | Eventi (slot) | Prevalenza | AP [IC95] | Lift | AP euristica | Soglia val: P / R / F1 | Budget 2%: P / R / F1 |
|---|---:|---:|---|---:|---:|---|---|
| P1 test ultimi 7 gg | 24 | 0.0039 | 0.0824 [0.0021–0.1773] | 20.96 | 0.0632 | 0.0373 / 0.5 / 0.0694 | 0.065 / 0.3333 / 0.1088 |
| P2 test ultimi 7 gg | 311 | 0.0568 | 0.296 [0.0685–0.4373] | 5.21 | 0.1053 | 0.1855 / 0.4855 / 0.2684 | 0.4455 / 0.1576 / 0.2328 |
| P1+2 test ultimi 7 gg | 335 | 0.0289 | 0.2377 [0.0562–0.3747] | 8.22 | 0.0844 | 0.2034 / 0.3612 / 0.2602 | 0.319 / 0.2209 / 0.261 |
| P1_to_P2 cross | 2500 | 0.1093 | 0.1292 [0.0973–0.1725] | 1.18 | 0.1784 | 0.1566 / 0.1268 / 0.1401 | 0.1812 / 0.0332 / 0.0561 |
| P2_to_P1 cross | 104 | 0.0035 | 0.0221 [0.0025–0.0861] | 6.31 | 0.0135 | 0.0131 / 0.1923 / 0.0245 | 0.0168 / 0.0962 / 0.0287 |

## 4. Eventi (inizio di un guasto elettrico preceduto da stato normale)

| Scenario | Regola | Eventi | Anticipati | Recall evento | Anticipo mediano (min) | Episodi di allarme | Falsi / inverter·giorno |
|---|---|---:|---:|---:|---:|---:|---:|
| P1 test ultimi 7 gg | soglia val | 3 | 2 | 0.6667 | 105.0 | 151 | 0.955 |
| P1 test ultimi 7 gg | budget 2% | 3 | 2 | 0.6667 | 82.5 | 76 | 0.468 |
| P2 test ultimi 7 gg | soglia val | 56 | 39 | 0.6964 | 90.0 | 259 | 1.357 |
| P2 test ultimi 7 gg | budget 2% | 56 | 30 | 0.5357 | 30.0 | 59 | 0.182 |
| P1+2 test ultimi 7 gg | soglia val | 59 | 36 | 0.6102 | 60.0 | 172 | 0.425 |
| P1+2 test ultimi 7 gg | budget 2% | 59 | 33 | 0.5593 | 30.0 | 95 | 0.188 |
| P1_to_P2 cross | soglia val | 403 | 143 | 0.3548 | 30.0 | 644 | 0.648 |
| P1_to_P2 cross | budget 2% | 403 | 58 | 0.1439 | 37.5 | 225 | 0.221 |
| P2_to_P1 cross | soglia val | 15 | 11 | 0.7333 | 15.0 | 674 | 0.884 |
| P2_to_P1 cross | budget 2% | 15 | 9 | 0.6 | 15.0 | 298 | 0.386 |

## 5. Stato a +2 h (4 classi): sistema ibrido vs persistenza

| Scenario | Ibrido macro-F1 / MCC | Persistenza macro-F1 / MCC | Δ macro-F1 [IC95] |
|---|---|---|---|
| P1 test ultimi 7 gg | 0.4547 / 0.335 | 0.7643 / 0.645 | -0.3096 [-0.481–-0.0872] |
| P2 test ultimi 7 gg | 0.6269 / 0.5027 | 0.6984 / 0.668 | -0.0715 [-0.3318–-0.0187] |
| P1+2 test ultimi 7 gg | 0.6689 / 0.5713 | 0.6938 / 0.6738 | -0.0249 [-0.0843–-0.0009] |
| P1_to_P2 cross | 0.7195 / 0.6236 | 0.7886 / 0.7317 | -0.0691 [-0.0966–-0.048] |
| P2_to_P1 cross | 0.4833 / 0.3518 | 0.5996 / 0.5911 | -0.1163 [-0.2062–-0.0484] |

## 6. Sensibilità (cross-impianto, AP e lift)

| Soglia sottoprestazione | Orizzonte | P1→P2 AP (lift) | P2→P1 AP (lift) |
|---:|---:|---|---|
| 0.75 | 60 min | 0.0926 (1.67) | 0.0819 (60.71) |
| 0.75 | 120 min | 0.1329 (1.23) | 0.0304 (11.27) |
| 0.75 | 240 min | 0.2163 (1.23) | 0.019 (3.68) |
| 0.8 | 60 min | 0.0932 (1.64) | 0.0745 (42.45) |
| 0.8 | 120 min | 0.1292 (1.18) | 0.0221 (6.31) |
| 0.8 | 240 min | 0.2011 (1.14) | 0.0249 (3.67) |
| 0.85 | 60 min | 0.0975 (1.67) | 0.051 (17.96) |
| 0.85 | 120 min | 0.1301 (1.17) | 0.0231 (4.06) |
| 0.85 | 240 min | 0.1983 (1.1) | 0.0285 (2.63) |

## 7. Importanza delle feature (permutazione, calo di AP sul target)

**P1_to_P2**: R_PEER_MEAN_24H +0.027, HOUR_SIN +0.023, R_PEER_MIN_1H +0.016, TWIN_RESID +0.015, R_PEER_MEAN_2H +0.013, R_PEER +0.011, R_PEER_MEAN_1H +0.004, IRR_DIFF_1H +0.002, STATE +0.000, R_PEER_TREND_1H -0.000, N_COMM_24H -0.000, SLOTS_SINCE_FAULT -0.001
**P2_to_P1**: R_PEER +0.013, TWIN_RESID +0.012, R_PEER_TREND_1H +0.010, R_PEER_MIN_1H +0.006, IRRADIATION +0.004, R_PEER_STD_1H +0.004, TWIN_RESID_MEAN_1H +0.004, DELTA_T +0.001, HOUR_COS +0.001, PLANT_FRAC_FAULT +0.001, R_PEER_MEAN_1H +0.000, N_UNDER_24H -0.000

## 8. Modello finale

`fisico` + `random_forest_leaf50`, soglia 0.2382, addestrato su entrambi gli impianti.
