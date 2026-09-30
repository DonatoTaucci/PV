"""Modelli: gemello digitale (potenza attesa) e previsore di guasti."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression

from . import config as C
from .features import FORECAST_FEATURES, TWIN_FEATURES


# ---------------------------------------------------------------------------
# Gemello digitale: P_NORM atteso dato meteo e ora
# ---------------------------------------------------------------------------
class PhysicalBaseline:
    """P = a * G * (1 + gamma * (Tmod - 25)) + b, con gamma tipico c-Si = -0.4 %/°C.

    È il modello di riferimento "da manuale": se il ML non lo batte, non serve.
    """
    gamma = -0.004

    def fit(self, X: pd.DataFrame, y):
        z = X["IRRADIATION"] * (1 + self.gamma * (X["MODULE_TEMPERATURE"] - 25))
        self.lr = LinearRegression().fit(z.to_frame(), y)
        return self

    def predict(self, X: pd.DataFrame):
        z = X["IRRADIATION"] * (1 + self.gamma * (X["MODULE_TEMPERATURE"] - 25))
        return self.lr.predict(z.to_frame())


def twin_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Solo istanti diurni etichettati NORMALE e con meteo completo."""
    ok = (df["LABEL"] == 0) & df[TWIN_FEATURES].notna().all(axis=1) & df["P_NORM"].notna()
    return df[ok]


def fit_twin(df: pd.DataFrame) -> LGBMRegressor:
    rows = twin_training_rows(df)
    model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                          min_child_samples=50, subsample=0.8, subsample_freq=1,
                          random_state=C.RANDOM_STATE, verbose=-1)
    return model.fit(rows[TWIN_FEATURES], rows["P_NORM"])


# ---------------------------------------------------------------------------
# Previsore di guasti: sistema ibrido regole + ML
# ---------------------------------------------------------------------------
# * se l'inverter è GIÀ in uno stato anomalo, la previsione a 2 h è lo stato
#   attuale (persistenza: nei dati i guasti durano in media ore);
# * se l'inverter è normale, un classificatore stima la probabilità che
#   INSORGA un guasto (e di che tipo) nelle prossime 2 ore.
# La scelta è stata fatta confrontando varianti sulla sola validazione
# temporale interna agli impianti (vedi docs/bozza_risposte_professore.md).

def make_classifier() -> LGBMClassifier:
    # Iperparametri conservativi e FISSATI a priori (niente tuning sull'impianto
    # di test). TODO tesi: tuning con TimeSeriesSplit sul solo impianto sorgente.
    return LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.05,
                          num_leaves=15, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8,
                          random_state=C.RANDOM_STATE, verbose=-1)


def fit_classifier(samples: pd.DataFrame) -> LGBMClassifier:
    """Addestra il modello di insorgenza sui soli istanti in stato NORMALE."""
    s = samples[samples["STATE"] == 0]
    clf = make_classifier()
    clf.fit(s[FORECAST_FEATURES], s["TARGET"].astype(int))
    return clf


def predict_proba_full(clf, X: pd.DataFrame) -> np.ndarray:
    """Probabilità del modello ML per tutte e 4 le classi (anche se non viste in training)."""
    p = clf.predict_proba(X[FORECAST_FEATURES])
    out = np.zeros((len(X), len(C.CLASS_NAMES)))
    out[:, clf.classes_.astype(int)] = p
    return out


def predict_hybrid(clf, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Restituisce (classe prevista, probabilità) del sistema ibrido."""
    proba = predict_proba_full(clf, X)
    state = X["STATE"].astype(int).values
    ongoing = state > 0
    proba[ongoing] = np.eye(len(C.CLASS_NAMES))[state[ongoing]]
    return proba.argmax(1), proba


def persistence_baseline(samples: pd.DataFrame) -> np.ndarray:
    """Baseline banale: 'nelle prossime 2 ore succederà ciò che succede adesso'."""
    return samples["STATE"].astype(int).values
