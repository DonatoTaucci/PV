"""Modelli: gemello digitale (potenza attesa) e previsore di insorgenza guasti.

Per entrambi i compiti si confrontano più candidati; la scelta avviene SOLO sul
set di validazione dell'impianto di addestramento (vedi scripts/run_experiments.py).
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import FORECAST_FEATURES, TWIN_FEATURES


# ---------------------------------------------------------------------------
# Gemello digitale: P_NORM atteso dato meteo e ora
# ---------------------------------------------------------------------------
class PhysicalBaseline:
    """P = a * G * (1 + gamma * (Tmod - 25)) + b, con gamma tipico c-Si = -0.4 %/°C."""
    gamma = -0.004

    def _z(self, X):
        return (X["IRRADIATION"] * (1 + self.gamma * (X["MODULE_TEMPERATURE"] - 25))).to_frame()

    def fit(self, X: pd.DataFrame, y):
        self.lr = LinearRegression().fit(self._z(X), y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.lr.predict(self._z(X))


class HybridTwin:
    """Modello fisico + LightGBM sul residuo (il ML corregge solo ciò che la fisica non spiega)."""

    def fit(self, X, y):
        self.phys = PhysicalBaseline().fit(X, y)
        self.ml = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7,
                                min_child_samples=100, random_state=C.RANDOM_STATE,
                                deterministic=True, force_row_wise=True, verbose=-1)
        self.ml.fit(X[TWIN_FEATURES], np.asarray(y) - self.phys.predict(X))
        return self

    def predict(self, X):
        return self.phys.predict(X) + self.ml.predict(X[TWIN_FEATURES])


def _lgbm_twin():
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                         subsample=0.8, subsample_freq=1, random_state=C.RANDOM_STATE,
                         deterministic=True, force_row_wise=True, verbose=-1)


TWIN_CANDIDATES = {
    "fisico": PhysicalBaseline,
    "lightgbm": _lgbm_twin,
    "ibrido_fisico+lightgbm": HybridTwin,
}


def twin_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Solo istanti diurni etichettati NORMALE e con meteo completo."""
    ok = (df["LABEL"] == 0) & df[TWIN_FEATURES].notna().all(axis=1) & df["P_NORM"].notna()
    return df[ok]


def fit_twin(df: pd.DataFrame, kind: str = "ibrido_fisico+lightgbm"):
    rows = twin_training_rows(df)
    return TWIN_CANDIDATES[kind]().fit(rows[TWIN_FEATURES], rows["P_NORM"])


# ---------------------------------------------------------------------------
# Previsore di insorgenza: candidati
# ---------------------------------------------------------------------------
def classifier_candidates() -> dict:
    """Nome -> funzione che crea un classificatore non addestrato."""
    cands = {
        "baseline_logistica_C0.1": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000)),
        "baseline_logistica_C1": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000)),
    }
    for leaf in (20, 50):
        cands[f"random_forest_leaf{leaf}"] = (lambda leaf=leaf: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf, max_features="sqrt",
                                   n_jobs=-1, random_state=C.RANDOM_STATE)))
    for leaves, mcs, n in product((7, 15, 31), (20, 100), (200, 500)):
        cands[f"lightgbm_l{leaves}_m{mcs}_n{n}"] = (lambda leaves=leaves, mcs=mcs, n=n: LGBMClassifier(
            objective="multiclass", n_estimators=n, learning_rate=0.05, num_leaves=leaves,
            min_child_samples=mcs, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=C.RANDOM_STATE, deterministic=True, force_row_wise=True, verbose=-1))
    return cands


def fit_classifier(samples: pd.DataFrame, name: str):
    """Addestra il candidato ``name`` sui soli istanti in stato NORMALE (modello di insorgenza)."""
    s = samples[samples["STATE"] == 0]
    clf = classifier_candidates()[name]()
    clf.fit(s[FORECAST_FEATURES], s["TARGET"].astype(int))
    return clf


def predict_proba_full(clf, X: pd.DataFrame) -> np.ndarray:
    """Probabilità per tutte e 4 le classi (anche se non viste in training)."""
    p = clf.predict_proba(X[FORECAST_FEATURES])
    out = np.zeros((len(X), len(C.CLASS_NAMES)))
    out[:, np.asarray(clf.classes_).astype(int)] = p
    return out


# ---------------------------------------------------------------------------
# Sistema ibrido completo
# ---------------------------------------------------------------------------
def predict_hybrid(clf, X: pd.DataFrame, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(classe prevista a +H, probabilità).

    * inverter già in stato anomalo -> persistenza;
    * inverter normale -> allarme se P(guasto elettrico) >= soglia, con la classe
      elettrica più probabile; altrimenti argmax delle probabilità.
    """
    proba = predict_proba_full(clf, X)
    yhat = proba.argmax(1)
    if threshold is not None:
        p_el = proba[:, 1] + proba[:, 2]
        el_cls = np.where(proba[:, 2] >= proba[:, 1], 2, 1)
        yhat = np.where(p_el >= threshold, el_cls, np.where(yhat == 3, 3, 0))
    state = X["STATE"].astype(int).values
    ongoing = state > 0
    yhat = np.where(ongoing, state, yhat)
    proba[ongoing] = np.eye(len(C.CLASS_NAMES))[state[ongoing]]
    return yhat, proba


def persistence_baseline(samples: pd.DataFrame) -> np.ndarray:
    """Baseline banale: 'nelle prossime 2 ore succederà ciò che succede adesso'."""
    return samples["STATE"].astype(int).values
