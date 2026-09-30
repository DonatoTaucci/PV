"""Parametri globali della pipeline (bozza v0.1).

Tutte le soglie sono raccolte qui per poterle discutere/giustificare nella tesi
e fare analisi di sensibilità senza toccare il resto del codice.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"

PLANTS = {
    1: {
        "generation": DATA_DIR / "Plant_1_Generation_Data.csv",
        "weather": DATA_DIR / "Plant_1_Weather_Sensor_Data.csv",
        # Plant 1 usa "15-05-2020 00:00" (giorno-mese-anno), Plant 2 ISO.
        "gen_datetime_format": "%d-%m-%Y %H:%M",
    },
    2: {
        "generation": DATA_DIR / "Plant_2_Generation_Data.csv",
        "weather": DATA_DIR / "Plant_2_Weather_Sensor_Data.csv",
        "gen_datetime_format": "%Y-%m-%d %H:%M:%S",
    },
}

FREQ = "15min"
SLOTS_PER_DAY = 96

# --- Pulizia ---------------------------------------------------------------
# Rapporto AC/DC atteso per un inverter di stringa moderno (~0.96-0.99).
# Se la mediana osservata è ~10 volte più bassa, DC_POWER è in scala sbagliata.
EXPECTED_EFFICIENCY_RANGE = (0.90, 1.0)
# Gap meteo interpolabili (in slot da 15 min): <= 1 ora.
MAX_WEATHER_INTERP_SLOTS = 4
# Irraggiamento (kW/m^2) sotto il quale lo slot è considerato "notte".
NIGHT_IRR = 0.005
# Irraggiamento minimo per valutare la prestazione dell'inverter
# (sotto 100 W/m^2 i rapporti diventano rumorosi).
DAYLIGHT_IRR = 0.10

# --- Etichettatura (weak supervision, basata sul confronto tra inverter) -----
# r_peer = DC_inverter / mediana DC degli inverter "vivi" dello stesso impianto
TRIP_RATIO = 0.02          # r_peer sotto questa soglia -> inverter fermo
UNDER_RATIO = 0.80         # r_peer sotto questa soglia -> sottoprestazione
UNDER_MIN_SLOTS = 3        # persistenza minima (45 min) per non confondere nuvole
TRIP_MIN_SLOTS = 2         # persistenza minima (30 min)

CLASS_NAMES = {
    0: "NORMALE",
    1: "SOTTOPRESTAZIONE",
    2: "INVERTER_FERMO",
    3: "PERDITA_COMUNICAZIONE",
}
# Ordine di gravità usato per riassumere un orizzonte futuro in una sola classe
SEVERITY = [2, 3, 1, 0]

# --- Previsione --------------------------------------------------------------
HORIZON_SLOTS = 8          # prevediamo lo stato più grave nelle prossime 2 ore
HISTORY_SLOTS_DAY = 96     # finestra storica di 24 h per le feature
TRAIN_FRACTION_TEMPORAL = 0.70
RANDOM_STATE = 42
