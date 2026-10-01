"""Dashboard del modello: risultati di addestramento/validazione/test ed esploratore delle predizioni.

Avvio locale:
    python webapp/app.py            -> http://127.0.0.1:5000

I dati mostrati vengono da results/ (prodotti da scripts/run_experiments.py).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, abort, render_template, request, send_file, send_from_directory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pv_fault import config as C  # noqa: E402
from pv_fault.recommendations import recommend  # noqa: E402

app = Flask(__name__)

METRICS = json.loads((C.RESULTS_DIR / "metrics.json").read_text())
CARD = json.loads((C.MODEL_DIR / "model_card.json").read_text())
REPORTS = {p: json.loads((C.RESULTS_DIR / f"cleaning_report_plant{p}.json").read_text()) for p in C.PLANTS}

SCENARIOS = {
    "pooled": ("Modello P1+P2, test sugli ultimi 7 giorni", "pooled_test.csv.gz"),
    "p1_to_p2": ("Addestrato su P1, applicato a P2 (34 giorni)", "plant2_cross.csv.gz"),
    "p2_to_p1": ("Addestrato su P2, applicato a P1 (34 giorni)", "plant1_cross.csv.gz"),
}
_PRED: dict = {}


@app.context_processor
def common():
    return {"card": CARD}


def predictions(key: str) -> pd.DataFrame:
    if key not in _PRED:
        df = pd.read_csv(C.RESULTS_DIR / "predictions" / SCENARIOS[key][1], parse_dates=["DATE_TIME"])
        df["DAY"] = df["DATE_TIME"].dt.date.astype(str)
        _PRED[key] = df
    return _PRED[key]


def threshold_of(key: str) -> float:
    sel = {"pooled": "P1+2", "p1_to_p2": "P1", "p2_to_p1": "P2"}[key]
    return METRICS["selection"][sel]["threshold_from_validation"]


@app.get("/")
def overview():
    final = METRICS["in_plant_test"]["P1+2"]
    return render_template("overview.html", card=CARD, m=final, M=METRICS, reports=REPORTS)


@app.get("/risultati")
def results():
    rows = [(f"P{k}, test ultimi 7 giorni", v) for k, v in METRICS["in_plant_test"].items()] + \
           [(k.replace("_to_", " → "), v) for k, v in METRICS["cross_plant"].items()]
    return render_template("results.html", M=METRICS, rows=rows, card=CARD)


@app.get("/esplora")
def explore():
    key = request.args.get("scenario", "pooled")
    if key not in SCENARIOS:
        abort(404)
    df = predictions(key)
    inverters = sorted(df["SOURCE_KEY"].unique())
    # di default l'inverter con più insorgenze nel periodo
    starts = df[df["TARGET"].isin([1, 2]) & df["IS_ONSET_CANDIDATE"]]
    default_inv = starts["SOURCE_KEY"].value_counts().index[0] if len(starts) else inverters[0]
    inv = request.args.get("inverter") or default_inv
    if inv not in inverters:
        inv = default_inv
    days = sorted(df.loc[df["SOURCE_KEY"] == inv, "DAY"].unique())
    default_day = starts.loc[starts["SOURCE_KEY"] == inv, "DAY"].min() if inv in set(starts["SOURCE_KEY"]) else days[0]
    day = request.args.get("giorno") or default_day
    if day not in days:
        day = default_day
    sub = df[(df["SOURCE_KEY"] == inv) & (df["DAY"] == day)]
    thr = threshold_of(key)
    alarms = []
    for _, r in sub[sub["ALARM"]].iterrows():
        cls = "INVERTER_FERMO" if r["P_INVERTER_FERMO"] >= r["P_SOTTOPRESTAZIONE"] else "SOTTOPRESTAZIONE"
        alarms.append({"time": r["DATE_TIME"].strftime("%H:%M"), "risk": r["P_ELETTRICO"], "cls": cls,
                       "happened": C.CLASS_NAMES.get(int(r["TARGET"]), "?")})
    rec = {c: recommend(c) for c in {a["cls"] for a in alarms}}
    return render_template("explore.html", scenarios=SCENARIOS, key=key, inverters=inverters, inv=inv,
                           days=days, day=day, alarms=alarms, rec=rec, thr=thr)


@app.get("/esplora/grafico.png")
def explore_plot():
    key = request.args.get("scenario", "pooled")
    df = predictions(key)
    sub = df[(df["SOURCE_KEY"] == request.args.get("inverter")) & (df["DAY"] == request.args.get("giorno"))]
    if sub.empty:
        abort(404)
    thr = threshold_of(key)
    t = sub["DATE_TIME"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1.3]})
    a1.plot(t, sub["R_PEER"].clip(upper=1.6), color="#2a78d6", lw=2, label="r_peer")
    colors = {1: "#eda100", 2: "#eb6834", 3: "#52514e"}
    for c, col in colors.items():
        m = sub["LABEL"] == c
        a1.scatter(t[m], np.full(m.sum(), -0.08), s=14, marker="s", color=col, label=C.CLASS_NAMES[c])
    a1.axhline(C.UNDER_RATIO, color="#52514e", lw=0.8, ls="--")
    a1.set_ylabel("r_peer")
    a1.legend(frameon=False, fontsize=7, ncol=4, loc="upper left")
    risk = sub["P_ELETTRICO"].where(sub["STATE"] == 0)
    a2.plot(t, risk, color="#eb6834", lw=2, label="P(guasto elettrico entro 2 h)")
    a2.axhline(thr, color="#52514e", lw=1, ls="--", label=f"soglia {thr:.2f}")
    a2.scatter(t[sub["ALARM"]], risk[sub["ALARM"]], color="#eb6834", s=24, zorder=3)
    a2.set_ylim(0, 1)
    a2.set_ylabel("rischio")
    a2.legend(frameon=False, fontsize=7, loc="upper left")
    a2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    for a in (a1, a2):
        a.grid(color="#e4e3df", lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.get("/figure/<path:name>")
def figure(name):
    return send_from_directory(C.FIG_DIR, name)


if __name__ == "__main__":
    app.run(debug=True)
