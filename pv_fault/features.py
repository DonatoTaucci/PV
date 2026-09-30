"""Feature ingegnerizzate e target di previsione.

Tutte le feature sono *adimensionali o fisiche* (rapporti, irraggiamento,
temperature) e usano solo informazioni passate: nessun valore assoluto di
potenza, che legherebbe il modello alla taglia di un impianto specifico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# Niente temperatura ambiente nel gemello: a parità di Tmod è fisicamente secondaria
# e rappresenta soprattutto il clima del sito (rischio di non trasferibilità).
TWIN_FEATURES = ["IRRADIATION", "MODULE_TEMPERATURE", "HOUR_SIN", "HOUR_COS"]

FORECAST_FEATURES = [
    # stato attuale e storia breve del singolo inverter
    "R_PEER", "R_PEER_MEAN_1H", "R_PEER_MIN_1H", "R_PEER_STD_1H", "R_PEER_MEAN_2H",
    "R_PEER_MEAN_24H", "R_PEER_TREND_1H",
    "N_UNDER_24H", "N_TRIP_24H", "N_COMM_24H", "SLOTS_SINCE_FAULT", "STATE",
    # scostamento dal gemello digitale
    "TWIN_RESID", "TWIN_RESID_MEAN_1H", "TWIN_RESID_MEAN_24H",
    # contesto meteo e temporale
    # (temperature assolute escluse: descrivono il clima del sito, non l'inverter)
    "IRRADIATION", "IRR_DIFF_1H", "DELTA_T",
    "HOUR_SIN", "HOUR_COS",
    # contesto d'impianto: quanti inverter sono già in difficoltà
    "PLANT_FRAC_FAULT",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    h = df["DATE_TIME"].dt.hour + df["DATE_TIME"].dt.minute / 60
    df["HOUR_SIN"] = np.sin(2 * np.pi * h / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * h / 24)
    return df


def add_forecast_features(df: pd.DataFrame, k_plant: float, twin=None) -> pd.DataFrame:
    """Richiede un DataFrame su griglia completa, ordinato per inverter e tempo."""
    df = add_time_features(df.sort_values(["SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True))
    g = df.groupby("SOURCE_KEY", group_keys=False)
    lab = df["LABEL"]

    df["P_NORM"] = df["DC_POWER"] / k_plant  # "irraggiamento equivalente" dell'inverter
    if twin is not None:
        ok = df[TWIN_FEATURES].notna().all(axis=1)
        pred = pd.Series(np.nan, index=df.index)
        pred[ok] = twin.predict(df.loc[ok, TWIN_FEATURES])
        df["TWIN_PRED"] = pred
        df["TWIN_RESID"] = (df["P_NORM"] - pred).where(df["IRRADIATION"] >= C.DAYLIGHT_IRR)
    else:
        df["TWIN_RESID"] = np.nan

    def roll(col, w, fn):
        return g[col].transform(lambda s: getattr(s.rolling(w, min_periods=1), fn)())

    df["R_PEER_MEAN_1H"] = roll("R_PEER", 4, "mean")
    df["R_PEER_MIN_1H"] = roll("R_PEER", 4, "min")
    df["R_PEER_STD_1H"] = roll("R_PEER", 4, "std")
    df["R_PEER_MEAN_2H"] = roll("R_PEER", 8, "mean")
    df["R_PEER_MEAN_24H"] = roll("R_PEER", C.HISTORY_SLOTS_DAY, "mean")
    df["R_PEER_TREND_1H"] = df["R_PEER"] - g["R_PEER"].shift(4)
    df["TWIN_RESID_MEAN_1H"] = roll("TWIN_RESID", 4, "mean")
    df["TWIN_RESID_MEAN_24H"] = roll("TWIN_RESID", C.HISTORY_SLOTS_DAY, "mean")

    for name, cls in [("N_UNDER_24H", 1), ("N_TRIP_24H", 2), ("N_COMM_24H", 3)]:
        df["_tmp"] = (lab == cls).astype(float)
        df[name] = roll("_tmp", C.HISTORY_SLOTS_DAY, "sum")
    df["_tmp"] = (lab > 0).astype(float)
    df["PLANT_FRAC_FAULT"] = df.groupby("DATE_TIME")["_tmp"].transform("mean")

    # slot trascorsi dall'ultimo evento anomalo (limitato a 2 giorni)
    fault = lab > 0
    last = df["DATE_TIME"].where(fault)
    last = last.groupby(df["SOURCE_KEY"]).ffill()
    since = (df["DATE_TIME"] - last).dt.total_seconds() / 900
    df["SLOTS_SINCE_FAULT"] = since.fillna(2 * C.SLOTS_PER_DAY).clip(upper=2 * C.SLOTS_PER_DAY)

    df["STATE"] = lab.where(lab >= 0, 0)
    df["IRR_DIFF_1H"] = df["IRRADIATION"] - g["IRRADIATION"].shift(4)
    df["DELTA_T"] = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]
    return df.drop(columns=["_tmp"])


def add_target(df: pd.DataFrame, horizon: int = C.HORIZON_SLOTS) -> pd.DataFrame:
    """TARGET = stato più grave nei prossimi ``horizon`` slot diurni (t+1 .. t+H)."""
    df = df.sort_values(["SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)
    g = df.groupby("SOURCE_KEY")["LABEL"]
    fut = pd.concat({h: g.shift(-h) for h in range(1, horizon + 1)}, axis=1)
    target = pd.Series(np.nan, index=df.index)
    has_day = (fut >= 0).any(axis=1)
    # scorro le classi dalla meno grave alla più grave: vince l'ultima assegnata
    for cls in reversed(C.SEVERITY):
        target[(fut == cls).any(axis=1)] = cls
    df["TARGET"] = target.where(has_day)
    # "onset": ora e nell'ultima ora tutto normale -> il guasto è davvero da prevedere
    recent_ok = df.groupby("SOURCE_KEY")["LABEL"].transform(lambda s: s.rolling(4, min_periods=1).max()) <= 0
    df["IS_ONSET_CANDIDATE"] = (df["LABEL"] == 0) & recent_ok
    return df


def forecast_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Campioni validi: istanti diurni con un target definito."""
    return df[(df["LABEL"] >= 0) & df["TARGET"].notna()].copy()
