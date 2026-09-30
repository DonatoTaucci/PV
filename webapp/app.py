"""Sito web beta: analisi dati SCADA e descrizioni testuali.

    python webapp/app.py        # poi aprire http://127.0.0.1:5000
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pv_fault import config as C  # noqa: E402
from pv_fault.data_cleaning import parse_datetime, split_merged  # noqa: E402
from pv_fault.inference import analyze_scada  # noqa: E402
from pv_fault.recommendations import analyze_description  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
SAMPLE = C.ROOT / "data" / "sample" / "esempio_scada_plant2.csv"


def _read_csv(storage) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(storage.read()))


@app.get("/")
def index():
    return render_template("index.html", sample_available=SAMPLE.exists())


@app.post("/analizza")
def analyze():
    try:
        if request.form.get("sample") == "1":
            gen, wea = split_merged(pd.read_csv(SAMPLE))
            source = "dati di esempio (Impianto 2, ultimi giorni: già visti in addestramento, solo dimostrativo)"
        elif request.files.get("merged") and request.files["merged"].filename:
            gen, wea = split_merged(_read_csv(request.files["merged"]))
            source = request.files["merged"].filename
        else:
            gen = _read_csv(request.files["generation"])
            wea = _read_csv(request.files["weather"])
            gen["DATE_TIME"] = parse_datetime(gen["DATE_TIME"])
            wea["DATE_TIME"] = parse_datetime(wea["DATE_TIME"])
            source = f"{request.files['generation'].filename} + {request.files['weather'].filename}"
        result = analyze_scada(gen, wea)
    except (KeyError, ValueError) as exc:
        return render_template("index.html", error=f"Impossibile analizzare i file: {exc}",
                               sample_available=SAMPLE.exists()), 400
    return render_template("results.html", r=result, source=source)


@app.post("/descrivi")
def describe():
    text = request.form.get("description", "").strip()
    hypotheses = analyze_description(text) if text else []
    return render_template("describe.html", text=text, hypotheses=hypotheses)


if __name__ == "__main__":
    app.run(debug=True)
