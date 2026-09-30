"""Analisi di nuovi dati SCADA con i modelli addestrati (usata dal sito)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data_cleaning import clean
from .labeling import plant_level_diagnostics, plant_scale
from .models import predict_hybrid
from .pipeline import build_features, label_and_normalize
from .recommendations import recommend

_MODELS = None


def load_models(path=C.MODEL_DIR / "pv_fault_models.joblib"):
    global _MODELS
    if _MODELS is None:
        _MODELS = joblib.load(path)
    return _MODELS


def analyze_scada(gen: pd.DataFrame, wea: pd.DataFrame, name: str = "upload") -> dict:
    models = load_models()
    clean_df, rep = clean(gen, wea, name=name)
    k = plant_scale(clean_df)
    k_note = None
    if not np.isfinite(k):
        k, k_note = models["k_plant_default"], "k stimato dai dati di training (poche ore di sole pieno)"
    df = build_features(label_and_normalize(clean_df, k), models["twin"])
    diag = plant_level_diagnostics(df)

    day = df[df["LABEL"] >= 0]
    if day.empty:
        raise ValueError("Nessun istante diurno nei dati caricati: servono dati con irraggiamento > 0.1 kW/m².")
    t_last = day["DATE_TIME"].max()
    now = day[day["DATE_TIME"] == t_last].copy()
    yhat, proba = predict_hybrid(models["classifier"], now)

    inverters = []
    for (_, row), yh, p in zip(now.iterrows(), yhat, proba):
        inv_hist = day[day["SOURCE_KEY"] == row["SOURCE_KEY"]]
        counts = inv_hist["LABEL"].value_counts()
        pred = C.CLASS_NAMES[int(yh)]
        inverters.append({
            "inverter": row["SOURCE_KEY"],
            "state_now": C.CLASS_NAMES[int(row["STATE"])],
            "r_peer": None if pd.isna(row["R_PEER"]) else round(float(row["R_PEER"]), 3),
            "prediction": pred,
            "risk_fault": round(float(1 - p[0]), 3),
            "proba": {C.CLASS_NAMES[i]: round(float(v), 3) for i, v in enumerate(p)},
            "history_slots": {C.CLASS_NAMES[c]: int(counts.get(c, 0)) for c in C.CLASS_NAMES},
            "recommendation": recommend(pred) if pred != "NORMALE" else None,
        })
    inverters.sort(key=lambda d: -d["risk_fault"])

    plant_alerts = []
    if diag.get("irradiance_sensor_suspect_slots", 0) > 4:
        plant_alerts.append({"category": "SENSORE_METEO", "slots": diag["irradiance_sensor_suspect_slots"],
                             **recommend("SENSORE_METEO")})
    if diag.get("thermal_anomaly_slots", 0) > 4:
        plant_alerts.append({"category": "STRESS_TERMICO", "slots": diag["thermal_anomaly_slots"],
                             **recommend("STRESS_TERMICO")})

    return {
        "reference_time": str(t_last),
        "horizon_minutes": models["horizon_slots"] * 15,
        "k_plant": round(float(k), 1),
        "k_note": k_note,
        "cleaning": {key: rep[key] for key in [
            "raw_rows_generation", "n_inverters", "days", "dc_scale_factor_applied",
            "slots_missing_weather", "weather_slots_imputed_short_gap", "weather_slots_imputed_night",
            "generation_rows_filled_night_zero", "generation_rows_comm_loss_daylight",
            "clock_lag_best_slots", "total_yield_decreases"]},
        "plant_diagnostics": diag,
        "plant_alerts": plant_alerts,
        "inverters": inverters,
        "model_version": models["version"],
    }
