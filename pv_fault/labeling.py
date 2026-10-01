"""Etichette di guasto ricavate dai dati.

Il dataset non ha etichette, quindi ogni inverter viene confrontato con la
mediana degli altri inverter dello stesso impianto nello stesso istante
(r_peer). Il confronto non usa il sensore di irraggiamento e non dipende dalla
taglia dell'impianto.

Limite: un calo che colpisce tutti gli inverter insieme (es. sporcamento
uniforme) non si vede; per quello servono il gemello digitale e le diagnosi a
livello di impianto in fondo al file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def plant_scale(df: pd.DataFrame) -> float:
    """Potenza DC per unità di irraggiamento di un inverter 'tipico' (kW per kW/m^2).

    Mediana tra inverter del 95° percentile di DC/IRR con sole pieno. Serve a
    normalizzare la potenza e rendere confrontabili impianti di taglia diversa.
    """
    sunny = df[(df["IRRADIATION"] > 0.3) & (df["DC_POWER"] > 0)]
    if sunny.empty:
        return np.nan
    ratio = sunny["DC_POWER"] / sunny["IRRADIATION"]
    return float(ratio.groupby(sunny["SOURCE_KEY"]).quantile(0.95).median())


def peer_ratio(df: pd.DataFrame, k_plant: float) -> pd.Series:
    """r_peer = DC_i / mediana(DC degli inverter che stanno producendo)."""
    day = df["IRRADIATION"] >= C.DAYLIGHT_IRR
    dc = df["DC_POWER"]
    ts_max = dc.groupby(df["DATE_TIME"]).transform("max")
    alive = dc > C.TRIP_RATIO * ts_max
    ref = dc.where(alive).groupby(df["DATE_TIME"]).transform("median")
    n_alive = alive.groupby(df["DATE_TIME"]).transform("sum")
    # se quasi tutto l'impianto è fermo, il riferimento diventa fisico (k * G)
    ref = ref.where(n_alive >= 3, k_plant * df["IRRADIATION"])
    return (dc / ref).where(day)


def _drop_short_runs(mask: pd.Series, groups: pd.Series, min_len: int) -> pd.Series:
    """Tiene solo le sequenze consecutive di True lunghe almeno ``min_len``."""
    run_id = (mask != mask.groupby(groups).shift()).cumsum()
    run_len = mask.groupby(run_id).transform("size")
    return mask & (run_len >= min_len)


def label_faults(df: pd.DataFrame, k_plant: float | None = None) -> pd.DataFrame:
    """Aggiunge R_PEER e LABEL (-1 = notte/non valutabile, 0..3 vedi config)."""
    df = df.sort_values(["SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)
    k_plant = k_plant or plant_scale(df)
    df["R_PEER"] = peer_ratio(df, k_plant)
    day = df["IRRADIATION"] >= C.DAYLIGHT_IRR
    inv = df["SOURCE_KEY"]

    trip = _drop_short_runs((df["R_PEER"] < C.TRIP_RATIO).fillna(False), inv, C.TRIP_MIN_SLOTS)
    under = (df["R_PEER"] >= C.TRIP_RATIO) & (df["R_PEER"] < C.UNDER_RATIO)
    under = _drop_short_runs(under.fillna(False), inv, C.UNDER_MIN_SLOTS)

    label = np.full(len(df), -1)
    label[day.values] = 0
    label[under.values] = 1
    label[trip.values] = 2
    # comunicazione persa di giorno (IRR noto o no): prevale sulle altre
    label[df["COMM_LOSS"].values] = 3
    df["LABEL"] = label
    df.attrs["k_plant"] = k_plant
    return df


def plant_level_diagnostics(df: pd.DataFrame) -> dict:
    """Diagnosi a livello d'impianto (regole, non ML): sensore meteo e stress termico."""
    w = df.drop_duplicates("DATE_TIME").set_index("DATE_TIME")
    day = w[w["IRRADIATION"] >= C.DAYLIGHT_IRR].dropna(subset=["MODULE_TEMPERATURE"])
    out: dict = {}
    if len(day) > 20:
        # modello termico lineare tipo NOCT: Tmod - Tamb = a * G + b
        dT = day["MODULE_TEMPERATURE"] - day["AMBIENT_TEMPERATURE"]
        a, b = np.polyfit(day["IRRADIATION"], dT, 1)
        res = dT - (a * day["IRRADIATION"] + b)
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        hot = res > 3 * mad
        out["thermal_model"] = {"a_degC_per_kWm2": round(float(a), 2), "b_degC": round(float(b), 2)}
        out["thermal_anomaly_slots"] = int(hot.sum())
        out["thermal_anomaly_last"] = str(res[hot].index.max()) if hot.any() else None

    # coerenza sensore di irraggiamento: potenza d'impianto vs irraggiamento
    plant = df.groupby("DATE_TIME").agg(dc=("DC_POWER", "median"), irr=("IRRADIATION", "first"))
    k = plant_scale(df)
    prod = plant[(plant["dc"] > 0.2 * k) | (plant["irr"] > 0.2)].dropna()
    if len(prod) and np.isfinite(k):
        ratio = prod["dc"] / (k * prod["irr"].clip(lower=0.01))
        bad = (ratio > 2.0) | ((ratio < 0.3) & (prod["irr"] > 0.3))
        out["irradiance_sensor_suspect_slots"] = int(bad.sum())
    return out
