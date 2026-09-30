# PV Fault: previsione di guasti in impianti fotovoltaici (beta per tesi)

Sistema sperimentale che, a partire da dati SCADA (potenza per inverter + stazione meteo),
pulisce i dati, individua gli stati anomali degli inverter e stima il rischio che un guasto
insorga nelle prossime 2 ore, con azioni di manutenzione consigliate. Include un sito web
(Flask) per caricare nuovi dati o descrivere a parole il comportamento di un impianto.

Dati: [Kaggle, Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data),
2 impianti in India, 22 inverter ciascuno, 34 giorni a 15 min (`data/raw/`).

## Struttura

| Percorso | Contenuto |
|---|---|
| `pv_fault/data_cleaning.py` | caricamento, griglia a 15 min, gestione timestamp mancanti, allineamento meteo/generazione, controlli fisici |
| `pv_fault/labeling.py` | etichette di guasto con regole (confronto tra inverter), diagnosi d'impianto |
| `pv_fault/features.py` | feature adimensionali e target di previsione |
| `pv_fault/models.py` | gemello digitale (LightGBM e modello fisico) e sistema ibrido persistenza + LightGBM |
| `pv_fault/evaluation.py` | metriche (nRMSE, macro-F1, MCC, AUC-PR, skill score) |
| `pv_fault/recommendations.py` | base di conoscenza guasti → azioni e analisi delle descrizioni testuali |
| `scripts/run_experiments.py` | esperimenti in-sito e cross-impianto, figure, modello finale |
| `webapp/` | sito beta |
| `docs/bozza_risposte_professore.md` | bozza: pulizia dati, metrica cross-impianto, confronto con la letteratura |
| `results/` | report di pulizia, metriche, figure (generati) |

## Uso

```bash
pip install -r requirements.txt
python scripts/run_experiments.py   # rigenera results/ e models/pv_fault_models.joblib
python webapp/app.py                # http://127.0.0.1:5000
```

Formato di upload: gli stessi CSV di Kaggle (generazione + meteo), oppure un unico CSV con le colonne
`DATE_TIME, SOURCE_KEY, DC_POWER, AC_POWER, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION`
(esempio in `data/sample/`).
