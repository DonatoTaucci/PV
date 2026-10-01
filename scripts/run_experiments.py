"""Addestramento, validazione e test del modello.

    python scripts/run_experiments.py            # circa 4 minuti
    python scripts/run_experiments.py --fast     # senza analisi di sensibilità

Passi:
  1. pulizia ed etichette per ciascun impianto;
  2. split per giorni: 20 train, 7 validazione, 7 test;
  3. scelta di gemello digitale, classificatore e soglia sulla sola validazione;
  4. test sugli ultimi 7 giorni (P1, P2, P1+P2) e tra impianti (P1->P2, P2->P1);
  5. intervalli di confidenza, metriche per evento, sensibilità;
  6. modello finale con la configurazione scelta su P1+P2, riaddestrato su tutti i dati.

Il metodo è descritto in docs/relazione_v1.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pv_fault import config as C  # noqa: E402
from pv_fault import plots  # noqa: E402
from pv_fault.data_cleaning import load_clean_plant  # noqa: E402
from pv_fault.evaluation import (best_f1_threshold, binary_at_threshold,  # noqa: E402
                                 classification_metrics, day_block_bootstrap, event_metrics,
                                 onset_metrics, regression_metrics, skill_score)
from pv_fault.features import FORECAST_FEATURES, TWIN_FEATURES  # noqa: E402
from pv_fault.labeling import plant_level_diagnostics  # noqa: E402
from pv_fault.models import (TWIN_CANDIDATES, classifier_candidates, fit_classifier,  # noqa: E402
                             fit_twin, persistence_baseline, predict_hybrid,
                             predict_proba_full, twin_training_rows)
from pv_fault.pipeline import build_features, day_split, in_period, label_and_normalize  # noqa: E402

EL = (1, 2)  # classi "guasto elettrico"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def samples_of(feat: pd.DataFrame, mask=None) -> pd.DataFrame:
    ok = (feat["LABEL"] >= 0) & feat["TARGET"].notna()
    if mask is not None:
        ok &= mask
    return feat[ok]


def onset_view(s: pd.DataFrame, proba: np.ndarray):
    """Sottoinsieme 'insorgenza' (istanti normali da almeno 1 h) + etichetta e punteggio elettrici."""
    on = s["IS_ONSET_CANDIDATE"].values
    y = s["TARGET"].astype(int).values[on]
    return on, np.isin(y, EL), proba[on][:, list(EL)].sum(1)


def safe_ap(y_bin, score):
    return float(average_precision_score(y_bin, score)) if 0 < y_bin.sum() < len(y_bin) else np.nan


# ---------------------------------------------------------------------------
# Selezione (solo validazione dell'impianto sorgente)
# ---------------------------------------------------------------------------
def features_of(labs: list, twin) -> pd.DataFrame:
    """Feature calcolate impianto per impianto (le grandezze d'impianto non vanno mescolate)."""
    return pd.concat([build_features(lab, twin) for lab in labs], ignore_index=True)


def select_on_validation(labs: list, split: dict, log: dict) -> dict:
    """Sceglie gemello, classificatore e soglia usando solo train e validazione."""
    tr_rows = pd.concat([lab[in_period(lab, split["train"])] for lab in labs])
    va_rows = pd.concat([lab[(lab["DATE_TIME"] >= split["val"][0]) & (lab["DATE_TIME"] < split["val"][1])]
                         for lab in labs])

    # gemello digitale: vince l'nRMSE più basso in validazione
    twin_scores = {}
    va_norm = twin_training_rows(va_rows)
    for kind in TWIN_CANDIDATES:
        tw = fit_twin(tr_rows, kind)
        twin_scores[kind] = regression_metrics(va_norm["P_NORM"], tw.predict(va_norm[TWIN_FEATURES]))
    twin_kind = min(twin_scores, key=lambda k: twin_scores[k]["nRMSE_%"])
    twin = fit_twin(tr_rows, twin_kind)

    # classificatore: vince l'AP più alta sulle insorgenze in validazione
    feat = features_of(labs, twin)
    train = samples_of(feat, in_period(feat, split["train"]))
    val = samples_of(feat, in_period(feat, split["val"]))
    clf_scores = {}
    for name in classifier_candidates():
        clf = fit_classifier(train, name)
        on, yb, score = onset_view(val, predict_proba_full(clf, val))
        clf_scores[name] = {"AP_val": round(safe_ap(yb, score), 4)}
    # riferimento senza apprendimento: rischio = 1 - r_peer medio dell'ultima ora
    on, yb, _ = onset_view(val, np.zeros((len(val), 4)))
    heur = (1 - val["R_PEER_MEAN_1H"].fillna(1)).values[on]
    clf_scores["euristica_1-r_peer_1h"] = {"AP_val": round(safe_ap(yb, heur), 4)}
    prevalence = float(yb.mean())

    ranked = sorted([k for k in clf_scores if not k.startswith("euristica")],
                    key=lambda k: -np.nan_to_num(clf_scores[k]["AP_val"], nan=-1))
    best = ranked[0]
    clf = fit_classifier(train, best)
    _, yb, score = onset_view(val, predict_proba_full(clf, val))
    thr, f1v = best_f1_threshold(yb, score) if yb.any() else (0.5, 0.0)
    log.update({"twin_validation": twin_scores, "twin_selected": twin_kind,
                "classifier_validation": clf_scores, "classifier_selected": best,
                "validation_onset_prevalence": round(prevalence, 4),
                "validation_onset_events": int(yb.sum()),
                "threshold_from_validation": round(thr, 4), "validation_F1_at_threshold": round(f1v, 4)})
    return {"twin": twin_kind, "clf": best, "thr": thr}


# ---------------------------------------------------------------------------
# Valutazione completa su un insieme di test
# ---------------------------------------------------------------------------
def evaluate(clf, feat: pd.DataFrame, test_mask, thr: float, tag: str, store: dict) -> dict:
    s = samples_of(feat, test_mask)
    y = s["TARGET"].astype(int).values
    yhat, proba = predict_hybrid(clf, s, thr)
    pers = persistence_baseline(s)
    res = {"hybrid": classification_metrics(y, yhat, proba),
           "persistence": classification_metrics(y, pers)}
    res["skill_macroF1_vs_persistence"] = skill_score(res["hybrid"]["macro_F1"], res["persistence"]["macro_F1"])

    on, yb, score = onset_view(s, proba)
    yon = y[on]
    res["onset_electrical"] = onset_metrics(yon, yhat[on], proba[on], classes=EL)
    res["onset_any"] = onset_metrics(yon, yhat[on], proba[on], classes=(1, 2, 3))
    res["at_validation_threshold"] = binary_at_threshold(yb, score, thr)
    q = np.quantile(score, 1 - C.ALARM_BUDGET) if len(score) else 1.0
    res["at_alarm_budget"] = {**binary_at_threshold(yb, score, q), "budget": C.ALARM_BUDGET}
    heur = (1 - s["R_PEER_MEAN_1H"].fillna(1)).values[on]
    res["heuristic_AP"] = round(safe_ap(yb, heur), 4)

    # intervalli di confidenza (bootstrap per giorni)
    days_on = s["DATE_TIME"].dt.date.values[on]
    res["CI95_AP_electrical"] = day_block_bootstrap(days_on, lambda i: safe_ap(yb[i], score[i]))
    days = s["DATE_TIME"].dt.date.values
    res["CI95_macroF1_hybrid_minus_persistence"] = day_block_bootstrap(
        days, lambda i: f1_score(y[i], yhat[i], average="macro", labels=sorted(set(y[i]) | set(yhat[i])))
        - f1_score(y[i], pers[i], average="macro", labels=sorted(set(y[i]) | set(pers[i]))))

    # eventi: allarme = istante normale con P(guasto elettrico) >= soglia
    full = feat[test_mask].copy()
    full["ALARM"] = False
    alarm_idx = s.index[on][score >= thr]
    full.loc[alarm_idx, "ALARM"] = True
    res["events_at_validation_threshold"] = event_metrics(full)
    alarm_idx_b = s.index[on][score >= q]
    full["ALARM"] = False
    full.loc[alarm_idx_b, "ALARM"] = True
    res["events_at_alarm_budget"] = event_metrics(full)

    # materiale per i grafici
    if yb.any():
        p, r, _ = precision_recall_curve(yb, score)
        store[f"pr_{tag}"] = (r, p, float(yb.mean()))
        bins = np.unique(np.quantile(score, np.linspace(0, 1, 11)))
        idx = np.clip(np.digitize(score, bins[1:-1]), 0, len(bins) - 2)
        store[f"cal_{tag}"] = (pd.Series(score).groupby(idx).mean().values,
                               pd.Series(yb).groupby(idx).mean().values)
    store[f"cm_{tag}"] = res["hybrid"]["confusion_matrix"]
    return res


def export_predictions(clf, feat, thr, path):
    s = samples_of(feat)
    _, proba = predict_hybrid(clf, s, thr)
    out = s[["DATE_TIME", "PLANT", "SOURCE_KEY", "LABEL", "STATE", "TARGET", "R_PEER", "P_NORM",
             "TWIN_PRED", "IRRADIATION", "MODULE_TEMPERATURE", "IS_ONSET_CANDIDATE"]].copy()
    for c, n in C.CLASS_NAMES.items():
        out[f"P_{n}"] = proba[:, c].round(4)
    out["P_ELETTRICO"] = (proba[:, 1] + proba[:, 2]).round(4)
    out["ALARM"] = (out["STATE"] == 0) & (out["P_ELETTRICO"] >= thr)
    out.to_csv(path, index=False, compression="gzip")


# ---------------------------------------------------------------------------
def main(fast: bool = False):
    t0 = time.time()
    for d in (C.RESULTS_DIR, C.FIG_DIR, C.MODEL_DIR, C.RESULTS_DIR / "predictions"):
        d.mkdir(parents=True, exist_ok=True)

    # 1) pulizia ed etichette
    labeled, reports, splits = {}, {}, {}
    for p in C.PLANTS:
        clean_df, rep = load_clean_plant(p)
        lab = label_and_normalize(clean_df)
        rep["k_plant_kW_per_kWm2"] = round(lab.attrs["k_plant"], 1)
        rep["plant_diagnostics"] = plant_level_diagnostics(lab)
        day = lab[lab["LABEL"] >= 0]
        rep["label_distribution_daylight_slots"] = {C.CLASS_NAMES[c]: int((day["LABEL"] == c).sum())
                                                    for c in C.CLASS_NAMES}
        labeled[p], reports[p], splits[p] = lab, rep, day_split(lab)
        (C.RESULTS_DIR / f"cleaning_report_plant{p}.json").write_text(
            json.dumps(jsonable(rep), indent=2, ensure_ascii=False))
    print(f"[1] pulizia ed etichette ok ({time.time() - t0:.0f}s)")

    M: dict = {"protocol": {"split_days": C.SPLIT_DAYS, "horizon_min": C.HORIZON_SLOTS * 15,
                            "alarm_budget": C.ALARM_BUDGET, "n_bootstrap": C.N_BOOTSTRAP,
                            "label_thresholds": {"trip": C.TRIP_RATIO, "under": C.UNDER_RATIO},
                            "splits": {p: {k: [str(a.date()), str((b - pd.Timedelta(days=1)).date())]
                                           for k, (a, b) in splits[p].items()} for p in C.PLANTS}},
               "selection": {}, "twin_test": {}, "in_plant_test": {}, "cross_plant": {}}
    store: dict = {}

    # 2) selezione su validazione
    # i due impianti coprono gli stessi giorni, quindi lo split è identico
    groups = {1: [labeled[1]], 2: [labeled[2]], "1+2": [labeled[1], labeled[2]]}
    chosen = {}
    for g, labs in groups.items():
        log: dict = {}
        chosen[g] = select_on_validation(labs, splits[1], log)
        M["selection"][f"P{g}"] = log
        print(f"[2] P{g}: twin={chosen[g]['twin']} clf={chosen[g]['clf']} "
              f"soglia={chosen[g]['thr']:.3f} ({time.time() - t0:.0f}s)")

    # 3) test in-impianto (e combinato): train+val -> ultimi 7 giorni
    sp = splits[1]
    trval_period = (sp["train"][0], sp["val"][1])
    for g, labs in groups.items():
        ch = chosen[g]
        trval = pd.concat([lab[in_period(lab, trval_period)] for lab in labs])
        tn = twin_training_rows(pd.concat([lab[lab["DATE_TIME"] >= sp["test"][0]] for lab in labs]))
        M["twin_test"][f"P{g}_test"] = {
            k: regression_metrics(tn["P_NORM"], fit_twin(trval, k).predict(tn[TWIN_FEATURES]))
            for k in TWIN_CANDIDATES}
        twin = fit_twin(trval, ch["twin"])
        feat = features_of(labs, twin)
        clf = fit_classifier(samples_of(feat, in_period(feat, trval_period)), ch["clf"])
        M["in_plant_test"][f"P{g}"] = evaluate(clf, feat, feat["DATE_TIME"] >= sp["test"][0],
                                               ch["thr"], f"P{g}_in", store)
        if g == "1+2":
            pooled = (clf, feat[feat["DATE_TIME"] >= sp["test"][0]], ch["thr"])
            export_predictions(clf, pooled[1], ch["thr"], C.RESULTS_DIR / "predictions" / "pooled_test.csv.gz")
        o = M["in_plant_test"][f"P{g}"]
        print(f"[3] P{g} test: AP={o['onset_electrical'].get('AP')} CI={o['CI95_AP_electrical']} "
              f"prev={o['onset_electrical']['prevalence']} ({time.time() - t0:.0f}s)")

    # 4) test cross-impianto
    cross_models = {}
    for src, tgt in [(1, 2), (2, 1)]:
        ch = chosen[src]
        tn = twin_training_rows(labeled[tgt])
        M["twin_test"][f"P{src}_to_P{tgt}"] = {
            k: regression_metrics(tn["P_NORM"], fit_twin(labeled[src], k).predict(tn[TWIN_FEATURES]))
            for k in TWIN_CANDIDATES}
        twin = fit_twin(labeled[src], ch["twin"])
        f_src, f_tgt = build_features(labeled[src], twin), build_features(labeled[tgt], twin)
        clf = fit_classifier(samples_of(f_src), ch["clf"])
        key = f"P{src}_to_P{tgt}"
        M["cross_plant"][key] = evaluate(clf, f_tgt, f_tgt["DATE_TIME"].notna(), ch["thr"], key, store)
        cross_models[key] = (clf, f_tgt, ch["thr"])
        export_predictions(clf, f_tgt, ch["thr"], C.RESULTS_DIR / "predictions" / f"plant{tgt}_cross.csv.gz")
        # importanza per permutazione sul target (solo per modelli ad albero è veloce)
        s = samples_of(f_tgt)
        s = s[s["IS_ONSET_CANDIDATE"]]
        yb = s["TARGET"].isin(EL).values
        if yb.any():
            def scorer(est, X, yy):
                return safe_ap(yy, predict_proba_full(est, X.assign(STATE=0))[:, list(EL)].sum(1))
            pi = permutation_importance(clf, s[FORECAST_FEATURES], yb, scoring=scorer, n_repeats=3,
                                        random_state=C.RANDOM_STATE, n_jobs=1)
            M["cross_plant"][key]["permutation_importance_AP"] = dict(sorted(
                {f: round(float(v), 4) for f, v in zip(FORECAST_FEATURES, pi.importances_mean)}.items(),
                key=lambda kv: -kv[1])[:12])
        o = M["cross_plant"][key]
        print(f"[4] {key}: AP={o['onset_electrical'].get('AP')} CI={o['CI95_AP_electrical']} "
              f"prev={o['onset_electrical']['prevalence']} ({time.time() - t0:.0f}s)")

    # 5) analisi di sensibilità (cross-impianto, configurazione fissata)
    if not fast:
        sens = []
        base_under, base_h = C.UNDER_RATIO, C.HORIZON_SLOTS
        for under in (0.75, 0.80, 0.85):
            C.UNDER_RATIO = under
            relab = {p: label_and_normalize(load_clean_plant(p)[0]) for p in C.PLANTS}
            for h in (4, 8, 16):
                for src, tgt in [(1, 2), (2, 1)]:
                    twin = fit_twin(relab[src], chosen[src]["twin"])
                    fs = build_features(relab[src], twin, horizon=h)
                    ft = build_features(relab[tgt], twin, horizon=h)
                    clf = fit_classifier(samples_of(fs), chosen[src]["clf"])
                    st = samples_of(ft)
                    _, proba = predict_hybrid(clf, st)
                    on, yb, score = onset_view(st, proba)
                    ap = safe_ap(yb, score)
                    sens.append({"under_ratio": under, "horizon_min": h * 15, "scenario": f"P{src}_to_P{tgt}",
                                 "AP": round(ap, 4), "prevalence": round(float(yb.mean()), 4),
                                 "lift": round(ap / yb.mean(), 2) if yb.any() else None})
            print(f"[5] sensibilità under={under} ok ({time.time() - t0:.0f}s)")
        C.UNDER_RATIO, C.HORIZON_SLOTS = base_under, base_h
        M["sensitivity"] = sens

    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(jsonable(M), indent=2, ensure_ascii=False))

    # 6) modello finale: configurazione scelta sulla validazione combinata P1+P2,
    #    poi riaddestramento su tutti i 34 giorni di entrambi gli impianti
    ch = chosen["1+2"]
    both = pd.concat([labeled[1], labeled[2]], ignore_index=True)
    twin = fit_twin(both, ch["twin"])
    clf = fit_classifier(samples_of(features_of([labeled[1], labeled[2]], twin)), ch["clf"])
    card = {"version": "1.0", "trained_on": "Plant 1 + Plant 2, 15/05-17/06/2020",
            "twin": ch["twin"], "classifier": ch["clf"], "threshold": round(ch["thr"], 4),
            "horizon_min": C.HORIZON_SLOTS * 15, "features": FORECAST_FEATURES,
            "twin_features": TWIN_FEATURES, "classes": C.CLASS_NAMES,
            "selection": "AP sulla validazione combinata P1+P2 (giorni 21-27)",
            "expected_performance": "vedi in_plant_test['P1+2'] in results/metrics.json"}
    joblib.dump({"twin": twin, "classifier": clf, "threshold": ch["thr"],
                 "k_plant_default": float(np.mean([labeled[p].attrs["k_plant"] for p in C.PLANTS])),
                 "horizon_slots": C.HORIZON_SLOTS, "version": "1.0", "card": card},
                C.MODEL_DIR / "pv_fault_models.joblib")
    (C.MODEL_DIR / "model_card.json").write_text(json.dumps(jsonable(card), indent=2, ensure_ascii=False))
    print(f"[6] modello finale: {ch['twin']} + {ch['clf']} ({time.time() - t0:.0f}s)")

    # 7) figure e report
    plots.completeness_heatmap(labeled, C.FIG_DIR / "fig1_completezza_dati.png")
    plots.lag_correlation(reports, C.FIG_DIR / "fig2_sincronizzazione_meteo_generazione.png")
    plots.peer_ratio_hist(labeled, C.FIG_DIR / "fig3_distribuzione_r_peer.png")
    plots.pr_curves(store, C.FIG_DIR / "fig4_curve_precision_recall.png")
    plots.calibration(store, C.FIG_DIR / "fig5_calibrazione.png")
    plots.confusion_matrices(store, C.FIG_DIR / "fig6_matrici_confusione_cross.png")
    plots.model_selection(M["selection"], C.FIG_DIR / "fig7_selezione_modelli.png")
    if "sensitivity" in M:
        plots.sensitivity(M["sensitivity"], C.FIG_DIR / "fig8_sensibilita.png")
    clf_p, f_p, thr_p = pooled
    plots.example_timeline(f_p, clf_p, thr_p, C.FIG_DIR / "fig9_esempio_timeline.png",
                           "etichette (quadrati) e rischio previsto a 2 h, periodo di test")
    write_summary(reports, M, card)
    print(f"Fatto in {time.time() - t0:.0f}s. Vedi results/summary.md")


def write_summary(reports: dict, M: dict, card: dict):
    L = ["# Risultati del modello v1\n\nTabelle prodotte da `scripts/run_experiments.py`.\n"]
    L.append(f"Split per giorni {M['protocol']['split_days']}, orizzonte {M['protocol']['horizon_min']} min, "
             f"bootstrap {M['protocol']['n_bootstrap']} ricampionamenti per giorni.\n")
    L.append("## 1. Selezione sulla validazione\n")
    for p, s in M["selection"].items():
        L.append(f"**{p}**: gemello `{s['twin_selected']}`, classificatore `{s['classifier_selected']}`, "
                 f"soglia {s['threshold_from_validation']} (F1 val {s['validation_F1_at_threshold']}); "
                 f"eventi di insorgenza in validazione {s['validation_onset_events']}, "
                 f"prevalenza {s['validation_onset_prevalence']}.\n")
        L.append("| Gemello | nRMSE % val | R² val |\n|---|---:|---:|")
        for k, m in s["twin_validation"].items():
            L.append(f"| {k} | {m['nRMSE_%']} | {m['R2']} |")
        L.append("\n| Classificatore | AP val |\n|---|---:|")
        for k, m in sorted(s["classifier_validation"].items(), key=lambda kv: -np.nan_to_num(kv[1]["AP_val"] or 0)):
            L.append(f"| {k} | {m['AP_val']} |")
        L.append("")

    L.append("## 2. Gemello digitale sul test\n")
    L.append("| Scenario | Modello | nRMSE % | R² | MAE |\n|---|---|---:|---:|---:|")
    for sc, d in M["twin_test"].items():
        for k, m in d.items():
            L.append(f"| {sc} | {k} | {m['nRMSE_%']} | {m['R2']} | {m['MAE']} |")

    L.append("\n## 3. Insorgenza di guasti elettrici (metrica primaria: AUC-PR)\n")
    L.append("| Scenario | Eventi (slot) | Prevalenza | AP [IC95] | Lift | AP euristica | "
             "Soglia val: P / R / F1 | Budget 2%: P / R / F1 |\n|---|---:|---:|---|---:|---:|---|---|")
    rows = [(f"{k} test ultimi 7 gg", v) for k, v in M["in_plant_test"].items()] + \
           [(f"{k} cross", v) for k, v in M["cross_plant"].items()]
    for name, r in rows:
        o, ci, a, b = r["onset_electrical"], r["CI95_AP_electrical"], r["at_validation_threshold"], r["at_alarm_budget"]
        L.append(f"| {name} | {o['n_positive']} | {o['prevalence']} | {o.get('AP')} [{ci['low']}–{ci['high']}] | "
                 f"{o.get('lift_AP_over_prevalence')} | {r['heuristic_AP']} | "
                 f"{a['precision']} / {a['recall']} / {a['F1']} | {b['precision']} / {b['recall']} / {b['F1']} |")

    L.append("\n## 4. Eventi (inizio di un guasto elettrico preceduto da stato normale)\n")
    L.append("| Scenario | Regola | Eventi | Anticipati | Recall evento | Anticipo mediano (min) | "
             "Episodi di allarme | Falsi / inverter·giorno |\n|---|---|---:|---:|---:|---:|---:|---:|")
    for name, r in rows:
        for rule, k in [("soglia val", "events_at_validation_threshold"), ("budget 2%", "events_at_alarm_budget")]:
            e = r[k]
            L.append(f"| {name} | {rule} | {e['n_events']} | {e['events_anticipated']} | {e['event_recall']} | "
                     f"{e['median_lead_min']} | {e['alarm_episodes']} | {e['false_alarms_per_inverter_day']} |")

    L.append("\n## 5. Stato a +2 h (4 classi): sistema ibrido vs persistenza\n")
    L.append("| Scenario | Ibrido macro-F1 / MCC | Persistenza macro-F1 / MCC | Δ macro-F1 [IC95] |\n|---|---|---|---|")
    for name, r in rows:
        h, p_, ci = r["hybrid"], r["persistence"], r["CI95_macroF1_hybrid_minus_persistence"]
        L.append(f"| {name} | {h['macro_F1']} / {h['MCC']} | {p_['macro_F1']} / {p_['MCC']} | "
                 f"{round(h['macro_F1'] - p_['macro_F1'], 4)} [{ci['low']}–{ci['high']}] |")

    if "sensitivity" in M:
        L.append("\n## 6. Sensibilità (cross-impianto, AP e lift)\n")
        L.append("| Soglia sottoprestazione | Orizzonte | P1→P2 AP (lift) | P2→P1 AP (lift) |\n|---:|---:|---|---|")
        sd = pd.DataFrame(M["sensitivity"])
        for (u, h), g in sd.groupby(["under_ratio", "horizon_min"]):
            a = g.set_index("scenario")
            L.append(f"| {u} | {h} min | {a.loc['P1_to_P2', 'AP']} ({a.loc['P1_to_P2', 'lift']}) | "
                     f"{a.loc['P2_to_P1', 'AP']} ({a.loc['P2_to_P1', 'lift']}) |")

    L.append("\n## 7. Importanza delle feature (permutazione, calo di AP sul target)\n")
    for k, r in M["cross_plant"].items():
        if "permutation_importance_AP" in r:
            L.append(f"**{k}**: " + ", ".join(f"{f} {v:+.3f}" for f, v in r["permutation_importance_AP"].items()))
    L.append(f"\n## 8. Modello finale\n\n`{card['twin']}` + `{card['classifier']}`, soglia {card['threshold']}, "
             f"addestrato su entrambi gli impianti.")
    (C.RESULTS_DIR / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    main(**vars(ap.parse_args()))
