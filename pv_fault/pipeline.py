"""Passaggi comuni: etichette, normalizzazione, feature e split temporale."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .features import add_forecast_features, add_target, add_time_features
from .labeling import label_faults, plant_scale


def label_and_normalize(clean_df: pd.DataFrame, k_plant: float | None = None) -> pd.DataFrame:
    k = k_plant or plant_scale(clean_df)
    df = label_faults(clean_df, k)
    df["P_NORM"] = df["DC_POWER"] / k
    df = add_time_features(df)
    df.attrs["k_plant"] = k
    return df


def build_features(labeled: pd.DataFrame, twin, horizon: int = C.HORIZON_SLOTS) -> pd.DataFrame:
    k = labeled.attrs["k_plant"]
    df = add_forecast_features(labeled, k, twin)
    df = add_target(df, horizon)
    df.attrs["k_plant"] = k
    return df


def day_split(df: pd.DataFrame) -> dict:
    """Assegna ogni giorno a train / val / test (in ordine temporale)."""
    days = sorted(df["DATE_TIME"].dt.normalize().unique())
    n_tr, n_va = C.SPLIT_DAYS["train"], C.SPLIT_DAYS["val"]
    return {"train": (days[0], days[n_tr]),
            "val": (days[n_tr], days[n_tr + n_va]),
            "test": (days[n_tr + n_va], days[-1] + pd.Timedelta(days=1))}


def in_period(df: pd.DataFrame, period: tuple, horizon: int = C.HORIZON_SLOTS) -> pd.Series:
    """Campioni dentro il periodo il cui target (t+1..t+H) non sconfina nel periodo dopo."""
    start, end = period
    return (df["DATE_TIME"] >= start) & (df["DATE_TIME"] < end - pd.Timedelta(minutes=15 * horizon))
