"""Metriche di valutazione.

- gemello digitale: nRMSE (RMSE diviso la potenza media), R², MAE;
- insorgenza dei guasti: average precision (AUC-PR) confrontata con la prevalenza,
  precision/recall alla soglia scelta, metriche per evento;
- stato a +2 h: macro-F1 e MCC, sempre insieme alla baseline di persistenza.

L'accuratezza non viene usata per scegliere nulla: con oltre il 90 % di istanti
normali, rispondere sempre "normale" darebbe già più di 0.9.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             mean_absolute_error, precision_recall_fscore_support,
                             r2_score, mean_squared_error)

from . import config as C

LABELS = sorted(C.CLASS_NAMES)


def regression_metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y), np.asarray(yhat)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    return {"nRMSE_%": round(100 * rmse / float(np.mean(y)), 2),
            "MAE": round(float(mean_absolute_error(y, yhat)), 4),
            "R2": round(float(r2_score(y, yhat)), 4),
            "n": int(len(y))}


def classification_metrics(y, yhat, proba=None) -> dict:
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    present = [c for c in LABELS if (y == c).any() or (yhat == c).any()]
    p, r, f, s = precision_recall_fscore_support(y, yhat, labels=LABELS, zero_division=0)
    out = {
        "macro_F1": round(float(f1_score(y, yhat, labels=present, average="macro", zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y, yhat)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, yhat)), 4),
        "accuracy": round(float((y == yhat).mean()), 4),
        "n": int(len(y)),
        "per_class": {C.CLASS_NAMES[c]: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                                         "F1": round(float(f[i]), 3), "support": int(s[i])}
                      for i, c in enumerate(LABELS)},
        "confusion_matrix": confusion_matrix(y, yhat, labels=LABELS).tolist(),
    }
    if proba is not None and (y > 0).any() and (y == 0).any():
        # rilevazione binaria "ci sarà un guasto?" indipendente dalla soglia
        out["AP_any_fault"] = round(float(average_precision_score(y > 0, 1 - proba[:, 0])), 4)
        out["prevalence_any_fault"] = round(float((y > 0).mean()), 4)
    return out


def skill_score(model: float, reference: float) -> float:
    """Miglioramento relativo rispetto alla baseline (1 = perfetto, 0 = come la baseline)."""
    return round((model - reference) / (1 - reference), 4) if reference < 1 else float("nan")


def onset_metrics(y, yhat, proba, classes=(1, 2, 3)) -> dict:
    """Metriche binarie 'insorgerà un guasto delle classi ``classes``?'."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    yb = np.isin(y, classes)
    pb = np.isin(yhat, classes)
    score = proba[:, list(classes)].sum(1)
    n_pos = int(yb.sum())
    out = {"n": int(len(y)), "n_positive": n_pos, "prevalence": round(float(yb.mean()), 4)}
    if 0 < n_pos < len(y):
        ap = float(average_precision_score(yb, score))
        out.update({"AP": round(ap, 4), "lift_AP_over_prevalence": round(ap / yb.mean(), 2),
                    "precision": round(float((pb & yb).sum() / max(pb.sum(), 1)), 4),
                    "recall": round(float((pb & yb).sum() / n_pos), 4),
                    "MCC": round(float(matthews_corrcoef(yb, pb)), 4)})
    return out


# ---------------------------------------------------------------------------
# Soglia di allarme
# ---------------------------------------------------------------------------
def best_f1_threshold(y_bin, score) -> tuple[float, float]:
    """Soglia che massimizza l'F1 binario (da usare SOLO sulla validazione)."""
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_bin, score)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return (float(t[i]) if len(t) else 0.5), float(f1[i])


def binary_at_threshold(y_bin, score, thr) -> dict:
    y_bin = np.asarray(y_bin).astype(bool)
    pred = np.asarray(score) >= thr
    tp = int((pred & y_bin).sum())
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / max(int(y_bin.sum()), 1)
    return {"threshold": round(float(thr), 4), "alarm_rate": round(float(pred.mean()), 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "F1": round(2 * prec * rec / max(prec + rec, 1e-12), 4),
            "MCC": round(float(matthews_corrcoef(y_bin, pred)), 4) if y_bin.any() and (~y_bin).any() else None}


# ---------------------------------------------------------------------------
# Metriche a livello di EVENTO (quello che interessa al manutentore)
# ---------------------------------------------------------------------------
def event_metrics(df, alarm_col: str = "ALARM", horizon: int = C.HORIZON_SLOTS) -> dict:
    """Un evento = inizio di una sequenza di guasto elettrico (LABEL 1/2) preceduta da NORMALE.

    * evento rilevato in anticipo se c'è almeno un allarme negli H slot precedenti;
    * anticipo = tempo tra il primo allarme nella finestra e l'inizio del guasto;
    * falso allarme = allarme non seguito da un guasto elettrico entro H slot;
      gli allarmi consecutivi sono fusi in "episodi".
    """
    d = df.sort_values(["SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)
    el = d["LABEL"].isin([1, 2]).values
    alarm = d[alarm_col].fillna(False).values.astype(bool)
    inv = d["SOURCE_KEY"].values
    prev_label = d.groupby("SOURCE_KEY")["LABEL"].shift(1).values
    starts = np.where(el & (prev_label == 0))[0]
    detected, leads = 0, []
    for i in starts:
        lo = max(i - horizon, 0)
        win = [j for j in range(lo, i) if inv[j] == inv[i] and alarm[j]]
        if win:
            detected += 1
            leads.append((d["DATE_TIME"].iloc[i] - d["DATE_TIME"].iloc[win[0]]).total_seconds() / 60)
    # episodi di allarme e falsi allarmi
    fut_el = np.zeros(len(d), dtype=bool)
    g = d.groupby("SOURCE_KEY")["LABEL"]
    for h in range(1, horizon + 1):
        fut_el |= g.shift(-h).isin([1, 2]).values
    ep_start = alarm & ~np.r_[False, alarm[:-1] & (inv[1:] == inv[:-1])]
    n_ep = int(ep_start.sum())
    ep_id = np.cumsum(ep_start)
    ep_true = pd.Series(fut_el[alarm]).groupby(ep_id[alarm]).any() if alarm.any() else pd.Series(dtype=bool)
    inv_days = d.loc[d["LABEL"] >= 0, ["SOURCE_KEY"]].assign(day=d["DATE_TIME"].dt.date).drop_duplicates().shape[0]
    return {"n_events": int(len(starts)), "events_anticipated": detected,
            "event_recall": round(detected / max(len(starts), 1), 4),
            "median_lead_min": float(np.median(leads)) if leads else None,
            "alarm_episodes": n_ep,
            "false_alarm_episodes": int((~ep_true).sum()) if n_ep else 0,
            "false_alarms_per_inverter_day": round(int((~ep_true).sum()) / max(inv_days, 1), 3) if n_ep else 0.0}


# ---------------------------------------------------------------------------
# Intervalli di confidenza: bootstrap a blocchi giornalieri
# ---------------------------------------------------------------------------
def day_block_bootstrap(days, stat_fn, n: int = C.N_BOOTSTRAP, seed: int = C.RANDOM_STATE) -> dict:
    """Ricampiona i GIORNI (non i singoli campioni, che sono autocorrelati).

    ``stat_fn(mask_indices)`` riceve gli indici di riga del campione bootstrap.
    """
    days = np.asarray(days)
    uniq = np.unique(days)
    idx_by_day = {d: np.where(days == d)[0] for d in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_day[d] for d in pick])
        v = stat_fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"low": None, "high": None}
    return {"low": round(float(np.percentile(vals, 2.5)), 4), "high": round(float(np.percentile(vals, 97.5)), 4)}
