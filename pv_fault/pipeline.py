"""Funzioni che concatenano pulizia -> etichette -> feature (usate da script e sito)."""
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


def temporal_split_time(df: pd.DataFrame, frac: float = C.TRAIN_FRACTION_TEMPORAL) -> pd.Timestamp:
    """Istante di taglio: primi ``frac`` giorni per training, resto per test."""
    days = sorted(df["DATE_TIME"].dt.normalize().unique())
    return pd.Timestamp(days[int(round(frac * len(days)))])
