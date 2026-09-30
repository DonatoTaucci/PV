"""Esegue l'intera sperimentazione della versione beta e salva risultati e figure.

    python scripts/run_experiments.py

Output:
  results/cleaning_report_plant{1,2}.json   numeri della pulizia (domanda 1)
  results/metrics.json                      tutte le metriche (domanda 2)
  results/summary.md                        tabelle pronte da incollare in bozza
  results/figures/*.png                     figure per la tesi
  models/pv_fault_models.joblib             modelli finali per il sito
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pv_fault import config as C  # noqa: E402
from pv_fault.data_cleaning import load_clean_plant  # noqa: E402
from pv_fault.evaluation import (classification_metrics, onset_metrics,  # noqa: E402
                                 regression_metrics, skill_score)
from pv_fault.features import TWIN_FEATURES  # noqa: E402
from pv_fault.labeling import plant_level_diagnostics  # noqa: E402
from pv_fault.models import (PhysicalBaseline, fit_classifier, fit_twin,  # noqa: E402
                             persistence_baseline, predict_hybrid, twin_training_rows)
from pv_fault.pipeline import build_features, label_and_normalize, temporal_split_time  # noqa: E402
from pv_fault import plots  # noqa: E402


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def evaluate_forecast(clf, test: pd.DataFrame) -> dict:
    yhat, proba = predict_hybrid(clf, test)
    y = test["TARGET"].astype(int).values
    pers = persistence_baseline(test)
    onset = test["IS_ONSET_CANDIDATE"].values
    res = {"hybrid_all": classification_metrics(y, yhat, proba),
           "persistence_all": classification_metrics(y, pers)}
    res["skill_macroF1_vs_persistence"] = skill_score(res["hybrid_all"]["macro_F1"],
                                                      res["persistence_all"]["macro_F1"])
    res["skill_MCC_vs_persistence"] = skill_score(res["hybrid_all"]["MCC"], res["persistence_all"]["MCC"])
    # insorgenza: la persistenza qui predice sempre "normale" (AP = prevalenza)
    res["onset_any"] = onset_metrics(y[onset], yhat[onset], proba[onset], classes=(1, 2, 3))
    res["onset_electrical"] = onset_metrics(y[onset], yhat[onset], proba[onset], classes=(1, 2))
    return res, yhat


def main():
    C.RESULTS_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.MODEL_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ pulizia
    labeled, reports = {}, {}
    for p in C.PLANTS:
        clean_df, rep = load_clean_plant(p)
        lab = label_and_normalize(clean_df)
        rep["k_plant_kW_per_kWm2"] = round(lab.attrs["k_plant"], 1)
        rep["plant_diagnostics"] = plant_level_diagnostics(lab)
        day = lab[lab["LABEL"] >= 0]
        rep["label_distribution_daylight_slots"] = {
            C.CLASS_NAMES[c]: int((day["LABEL"] == c).sum()) for c in C.CLASS_NAMES}
        labeled[p], reports[p] = lab, rep
        (C.RESULTS_DIR / f"cleaning_report_plant{p}.json").write_text(
            json.dumps(to_jsonable(rep), indent=2, ensure_ascii=False))
        print(f"[Plant {p}] pulizia ok, k={rep['k_plant_kW_per_kWm2']}, "
              f"etichette={rep['label_distribution_daylight_slots']}")

    metrics: dict = {"twin": {}, "forecast": {}, "config": {
        "horizon_slots": C.HORIZON_SLOTS, "under_ratio": C.UNDER_RATIO,
        "trip_ratio": C.TRIP_RATIO, "train_fraction_temporal": C.TRAIN_FRACTION_TEMPORAL}}
    preds_for_plots = {}
    gap = pd.Timedelta(minutes=15 * C.HORIZON_SLOTS)

    # ------------------------------------------------------------ scenari
    scenarios = []
    for p in C.PLANTS:
        cut = temporal_split_time(labeled[p])
        scenarios.append((f"P{p}_temporale", p, p, cut))
    scenarios += [("P1_to_P2_cross", 1, 2, None), ("P2_to_P1_cross", 2, 1, None)]

    for name, src, tgt, cut in scenarios:
        src_df, tgt_df = labeled[src], labeled[tgt]
        if cut is not None:
            src_train_rows = src_df[src_df["DATE_TIME"] < cut - gap]
        else:
            src_train_rows = src_df

        # gemello digitale -------------------------------------------------
        twin = fit_twin(src_train_rows)
        rows = twin_training_rows(src_train_rows)
        phys = PhysicalBaseline().fit(rows[TWIN_FEATURES], rows["P_NORM"])
        tgt_eval = tgt_df if cut is None else tgt_df[tgt_df["DATE_TIME"] >= cut]
        normal = twin_training_rows(tgt_eval)
        allday = tgt_eval[(tgt_eval["LABEL"].isin([0, 1, 2])) & tgt_eval[TWIN_FEATURES].notna().all(axis=1)]
        metrics["twin"][name] = {
            "lightgbm_normal_rows": regression_metrics(normal["P_NORM"], twin.predict(normal[TWIN_FEATURES])),
            "physical_normal_rows": regression_metrics(normal["P_NORM"], phys.predict(normal[TWIN_FEATURES])),
            "lightgbm_all_daylight_rows": regression_metrics(allday["P_NORM"], twin.predict(allday[TWIN_FEATURES])),
        }
        if cut is None:
            preds_for_plots[f"twin_{name}"] = (normal["P_NORM"].values, twin.predict(normal[TWIN_FEATURES]))

        # previsione guasti --------------------------------------------------
        src_feat = build_features(src_df, twin)
        tgt_feat = src_feat if src == tgt else build_features(tgt_df, twin)
        src_s = src_feat[(src_feat["LABEL"] >= 0) & src_feat["TARGET"].notna()]
        tgt_s = tgt_feat[(tgt_feat["LABEL"] >= 0) & tgt_feat["TARGET"].notna()]
        if cut is not None:
            train = src_s[src_s["DATE_TIME"] < cut - gap]
            test = tgt_s[tgt_s["DATE_TIME"] >= cut]
        else:
            train, test = src_s, tgt_s
        clf = fit_classifier(train)
        res, yhat = evaluate_forecast(clf, test)
        res["n_train"], res["n_test"] = len(train), len(test)
        res["train_target_distribution"] = {C.CLASS_NAMES[int(k)]: int(v)
                                            for k, v in train["TARGET"].value_counts().sort_index().items()}
        metrics["forecast"][name] = res
        preds_for_plots[f"cm_{name}"] = res["hybrid_all"]["confusion_matrix"]
        if name == "P1_to_P2_cross":
            imp = pd.Series(clf.booster_.feature_importance("gain"), index=clf.feature_name_)
            metrics["feature_importance_gain_P1_to_P2"] = (imp / imp.sum()).round(4).sort_values(
                ascending=False).head(12).to_dict()
        m, o = res["hybrid_all"], res["onset_electrical"]
        print(f"[{name}] ibrido macro-F1={m['macro_F1']} MCC={m['MCC']} | persistenza "
              f"macro-F1={res['persistence_all']['macro_F1']} MCC={res['persistence_all']['MCC']} | "
              f"insorgenza guasti elettrici AP={o.get('AP')} (prevalenza {o['prevalence']})")

    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=2, ensure_ascii=False))

    # ------------------------------------------------------------ figure
    plots.completeness_heatmap(labeled, C.FIG_DIR / "fig1_completezza_dati.png")
    plots.lag_correlation(reports, C.FIG_DIR / "fig2_sincronizzazione_meteo_generazione.png")
    plots.peer_ratio_hist(labeled, C.FIG_DIR / "fig3_distribuzione_r_peer.png")
    plots.twin_scatter(preds_for_plots, C.FIG_DIR / "fig4_gemello_digitale_cross.png")
    plots.confusion_matrices(preds_for_plots, C.FIG_DIR / "fig5_matrici_confusione_cross.png")

    # ------------------------------------------------------------ modello finale per il sito
    both = pd.concat([labeled[1], labeled[2]], ignore_index=True)
    twin = fit_twin(both)
    feats = [build_features(labeled[p], twin) for p in C.PLANTS]
    samples = pd.concat([f[(f["LABEL"] >= 0) & f["TARGET"].notna()] for f in feats])
    clf = fit_classifier(samples)
    joblib.dump({"twin": twin, "classifier": clf,
                 "k_plant_default": float(np.mean([labeled[p].attrs["k_plant"] for p in C.PLANTS])),
                 "horizon_slots": C.HORIZON_SLOTS, "version": "0.1-beta"},
                C.MODEL_DIR / "pv_fault_models.joblib")

    write_summary(reports, metrics)
    print("Fatto. Vedi results/summary.md")


def write_summary(reports: dict, metrics: dict):
    L = ["# Risultati sperimentali (generati automaticamente)\n"]
    L.append("## 1. Pulizia dati\n")
    keys = ["raw_rows_generation", "expected_generation_rows", "n_inverters", "days",
            "median_ac_dc_ratio_raw", "dc_scale_factor_applied",
            "slots_missing_generation_all_inverters", "slots_missing_weather",
            "slots_generation_without_weather", "slots_weather_without_generation",
            "weather_slots_imputed_short_gap", "weather_slots_imputed_night", "weather_slots_left_missing",
            "generation_rows_missing", "generation_rows_filled_night_zero",
            "generation_rows_comm_loss_daylight", "generation_rows_comm_loss_plant_wide",
            "clock_lag_best_slots", "daylight_rows_zero_dc", "ac_gt_dc_rows",
            "total_yield_decreases", "total_yield_zero_values",
            "daily_yield_nonzero_at_midnight_frac", "daily_yield_vs_ac_energy_ratio", "k_plant_kW_per_kWm2"]
    L.append("| Voce | Impianto 1 | Impianto 2 |\n|---|---:|---:|")
    for k in keys:
        v = [reports[p].get(k) for p in (1, 2)]
        v = [f"{x:.3f}" if isinstance(x, float) else str(x) for x in v]
        L.append(f"| {k} | {v[0]} | {v[1]} |")
    L.append("\n**Distribuzione etichette (slot diurni inverter)**\n")
    L.append("| Classe | Impianto 1 | Impianto 2 |\n|---|---:|---:|")
    for c in C.CLASS_NAMES.values():
        a, b = (reports[p]["label_distribution_daylight_slots"][c] for p in (1, 2))
        L.append(f"| {c} | {a} | {b} |")

    L.append("\n## 2. Gemello digitale (potenza attesa normalizzata)\n")
    L.append("| Scenario | Modello | nRMSE % | MAE | R² | n |\n|---|---|---:|---:|---:|---:|")
    for s, d in metrics["twin"].items():
        for mname, m in d.items():
            L.append(f"| {s} | {mname} | {m['nRMSE_%']} | {m['MAE']} | {m['R2']} | {m['n']} |")

    L.append(f"\n## 3. Previsione dello stato a +{C.HORIZON_SLOTS * 15} min (4 classi)\n")
    L.append("| Scenario | Sistema | macro-F1 | MCC | Bal. acc. | Accuracy | n test |\n|---|---|---:|---:|---:|---:|---:|")
    for s, d in metrics["forecast"].items():
        for mname in ["hybrid_all", "persistence_all"]:
            m = d[mname]
            L.append(f"| {s} | {mname} | {m['macro_F1']} | {m['MCC']} | {m['balanced_accuracy']} | "
                     f"{m['accuracy']} | {m['n']} |")
    L.append("\nSkill score vs persistenza (macro-F1 / MCC): " + ", ".join(
        f"{s}: {d['skill_macroF1_vs_persistence']} / {d['skill_MCC_vs_persistence']}"
        for s, d in metrics["forecast"].items()))
    L.append("\n## 4. Insorgenza di nuovi guasti (istanti attualmente normali)\n")
    L.append("| Scenario | Tipo | n pos. | Prevalenza | AP | Lift AP | Precision | Recall | MCC |\n"
             "|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s, d in metrics["forecast"].items():
        for k in ["onset_electrical", "onset_any"]:
            o = d[k]
            L.append(f"| {s} | {k.replace('onset_', '')} | {o['n_positive']} | {o['prevalence']} | "
                     f"{o.get('AP', '-')} | {o.get('lift_AP_over_prevalence', '-')} | {o.get('precision', '-')} | "
                     f"{o.get('recall', '-')} | {o.get('MCC', '-')} |")
    L.append("\n**Recall per classe (sistema ibrido, tutti i campioni)**\n")
    L.append("| Scenario | " + " | ".join(C.CLASS_NAMES.values()) + " |\n|---|" + "---:|" * len(C.CLASS_NAMES))
    for s, d in metrics["forecast"].items():
        pc = d["hybrid_all"]["per_class"]
        L.append(f"| {s} | " + " | ".join(f"{pc[c]['recall']} (n={pc[c]['support']})"
                                          for c in C.CLASS_NAMES.values()) + " |")
    L.append("\n**Importanza feature (gain, P1→P2)**: " + ", ".join(
        f"{k} {v:.2f}" for k, v in metrics["feature_importance_gain_P1_to_P2"].items()))
    (C.RESULTS_DIR / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
