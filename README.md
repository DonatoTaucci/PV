# PV Fault

Progetto di tesi sulla previsione di guasti negli impianti fotovoltaici a partire dai dati SCADA.

Per ogni inverter che funziona normalmente il modello stima la probabilità che entro due ore si fermi
o renda meno degli inverter vicini. Gli inverter già in guasto vengono riconosciuti da regole di
diagnosi. A ogni allarme sono associate le cause più probabili e i controlli da fare.

I dati sono quelli di [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
(Kaggle): due impianti in India, 22 inverter ciascuno, 34 giorni con campionamento a 15 minuti.
I CSV originali sono in `data/raw/`.

## Avvio rapido

```bash
pip install -r requirements.txt
python scripts/run_experiments.py     # addestramento, validazione e test (circa 4 minuti)
python webapp/app.py                  # dashboard su http://127.0.0.1:5000
```

Con `python scripts/run_experiments.py --fast` si salta l'analisi di sensibilità (circa 2 minuti).

Lo script rigenera:

- `results/summary.md`: tutte le tabelle dei risultati;
- `results/metrics.json`: le stesse metriche in formato leggibile da programma;
- `results/cleaning_report_plant*.json`: i numeri della pulizia dei dati;
- `results/figures/`: le figure;
- `results/predictions/`: le predizioni sul test, usate dalla dashboard;
- `models/`: il modello finale e la sua scheda (`model_card.json`).

## Struttura

```
pv_fault/
  config.py          percorsi e soglie
  data_cleaning.py   lettura dei CSV, griglia a 15 min, dati mancanti, controlli fisici
  labeling.py        etichette di guasto (confronto tra inverter) e diagnosi d'impianto
  features.py        feature e target di previsione
  models.py          gemello digitale, classificatori candidati, sistema completo
  evaluation.py      metriche, soglia, metriche per evento, bootstrap
  pipeline.py        passaggi comuni e split temporale
  plots.py           figure
  recommendations.py cause e controlli per tipo di guasto
  inference.py       applicazione del modello a nuovi dati (per il caricamento CSV, non ancora nel sito)
scripts/run_experiments.py
webapp/              dashboard Flask
docs/relazione_v1.md metodo e risultati della prima versione
```

## Metodo in breve

1. Pulizia: griglia completa timestamp × inverter, correzione della scala di DC_POWER dell'impianto 1,
   imputazione solo dove è giustificata (notte, buchi meteo fino a un'ora), verifica che meteo e
   produzione siano sincronizzati.
2. Etichette: il dataset non ne ha. Ogni inverter viene confrontato con la mediana degli altri nello
   stesso istante (`r_peer`). Le classi sono NORMALE, SOTTOPRESTAZIONE, INVERTER_FERMO e PERDITA_COMUNICAZIONE.
3. Gemello digitale: potenza attesa da irraggiamento, temperatura dei moduli e ora.
4. Classificatore: stima l'insorgenza di un guasto nelle due ore successive. Viene scelto tra
   regressione logistica, random forest e LightGBM.
5. Valutazione: 20 giorni di train, 7 di validazione e 7 di test, più i test tra impianti diversi.

La descrizione completa e i risultati sono in `docs/relazione_v1.md`.

## Deploy

`render.yaml` contiene la configurazione per Render (piano gratuito). Il servizio parte con:

```bash
gunicorn --chdir webapp app:app --bind 0.0.0.0:$PORT
```

Le versioni in `requirements.txt` sono fissate perché il modello salvato va caricato con le stesse
versioni di scikit-learn e LightGBM usate per addestrarlo.
