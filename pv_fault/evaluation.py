"""Metriche di valutazione.

Scelte (motivate nella bozza, docs/bozza_risposte_professore.md):
  * gemello digitale -> nRMSE (RMSE / media della potenza misurata), MAE, R^2:
    l'nRMSE è adimensionale e quindi confrontabile tra impianti diversi;
  * previsione guasti -> macro-F1 (metrica primaria), MCC, balanced accuracy,
    F1/recall per classe, average precision "guasto sì/no", e le stesse metriche
    sul sottoinsieme "onset" (guasti non ancora in corso) e contro la baseline
    di persistenza. L'accuratezza semplice NON è usata: con >90 % di istanti
    normali un classificatore che dice sempre "normale" avrebbe accuratezza >0.9.
"""
from __future__ import annotations

import numpy as np
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
