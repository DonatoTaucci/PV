"""Figure statiche per la tesi (matplotlib, PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def completeness_heatmap(labeled: dict, path):
    """Numero di inverter che hanno inviato un record (dato grezzo), per giorno x ora."""
    fig, axes = plt.subplots(1, len(labeled), figsize=(11, 4), sharey=True)
    for ax, (p, df) in zip(np.atleast_1d(axes), labeled.items()):
        n_inv = df["SOURCE_KEY"].nunique()
        reported = ~df["RAW_MISSING"]
        m = reported.groupby([df["DATE_TIME"].dt.date, df["DATE_TIME"].dt.time]).sum().unstack()
        im = ax.imshow(m.values.T, aspect="auto", cmap="Blues", vmin=0, vmax=n_inv, origin="lower")
        ax.set_title(f"Impianto {p}: inverter con record presente")
        ax.set_xlabel("giorno (15/05 → 17/06/2020)")
        ax.set_yticks(range(0, 96, 16), [f"{h:02d}:00" for h in range(0, 24, 4)])
        ax.grid(False)
    fig.colorbar(im, ax=axes, label="n. inverter", shrink=0.8)
    fig.savefig(path)
    plt.close(fig)


def lag_correlation(reports: dict, path):
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, (p, rep) in enumerate(reports.items()):
        d = rep["clock_lag_correlations"]
        lags = [int(k) for k in d]
        ax.plot(lags, list(d.values()), marker="o", lw=2, ms=6, color=SERIES[i], label=f"Impianto {p}")
    ax.set_xlabel("sfasamento meteo rispetto a generazione (slot da 15 min)")
    ax.set_ylabel("correlazione ΣDC vs irraggiamento")
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def peer_ratio_hist(labeled: dict, path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.linspace(0, 1.6, 81)
    for i, (p, df) in enumerate(labeled.items()):
        r = df["R_PEER"].dropna().clip(upper=1.6)
        ax.hist(r, bins=bins, density=True, histtype="step", lw=2, color=SERIES[i], label=f"Impianto {p}")
    for x, t in [(C.TRIP_RATIO, "fermo"), (C.UNDER_RATIO, "sottoprestazione")]:
        ax.axvline(x, color=MUTED, lw=0.8, ls="--")
        ax.text(x + 0.01, ax.get_ylim()[1] * 0.9, t, color=MUTED, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("r_peer = DC inverter / mediana DC inverter attivi")
    ax.set_ylabel("densità (log)")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def confusion_matrices(preds: dict, path):
    keys = [k for k in preds if k.startswith("cm_") and "_to_" in k]
    names = [C.CLASS_NAMES[c][:12] for c in sorted(C.CLASS_NAMES)]
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 4.4))
    for ax, k in zip(np.atleast_1d(axes), keys):
        cm = np.array(preds[k], dtype=float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predetto")
        ax.set_ylabel("reale")
        ax.set_title(k.replace("cm_", "").replace("_to_", " → ") + " (stato a +2 h)")
        ax.grid(False)
    fig.savefig(path)
    plt.close(fig)


def _label(tag: str) -> str:
    return tag.replace("P1+2_in", "P1+P2, ultimi 7 giorni").replace("_to_", " → ")


def pr_curves(store: dict, path):
    keys = [k for k in store if k.startswith("pr_") and ("_to_" in k or "+" in k)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, k in enumerate(keys):
        r, p, prev = store[k]
        tag = k[3:]
        ax.plot(r, p, lw=2, color=SERIES[i % 4], label=f"{_label(tag)} (prevalenza {prev:.3f})")
        ax.axhline(prev, color=SERIES[i % 4], lw=0.8, ls=":")
    ax.set_xlabel("recall (insorgenze di guasto elettrico)")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Curve precision-recall sul test (linee punteggiate = classificatore casuale)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def calibration(store: dict, path):
    keys = [k for k in store if k.startswith("cal_") and ("_to_" in k or "+" in k)]
    fig, ax = plt.subplots(figsize=(5, 4))
    mx = 0.0
    for i, k in enumerate(keys):
        pm, obs = store[k]
        mx = max(mx, float(np.max(pm)), float(np.max(obs)))
        ax.plot(pm, obs, marker="o", ms=6, lw=2, color=SERIES[i % 4], label=_label(k[4:]))
    ax.plot([0, mx], [0, mx], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("probabilità prevista di guasto elettrico (media per decile)")
    ax.set_ylabel("frequenza osservata")
    ax.set_title("Calibrazione")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def model_selection(selection: dict, path):
    fig, axes = plt.subplots(1, len(selection), figsize=(6.5 * len(selection), 5), sharey=False)
    fig.subplots_adjust(wspace=0.75)
    for ax, (plant, s) in zip(np.atleast_1d(axes), selection.items()):
        items = sorted(s["classifier_validation"].items(), key=lambda kv: np.nan_to_num(kv[1]["AP_val"] or 0))
        names = [k for k, _ in items]
        vals = [np.nan_to_num(v["AP_val"] or 0) for _, v in items]
        colors = [SERIES[1] if n == s["classifier_selected"] else
                  (MUTED if n.startswith("euristica") or n.startswith("baseline") else SERIES[0]) for n in names]
        ax.barh(range(len(names)), vals, color=colors, height=0.7)
        ax.axvline(s["validation_onset_prevalence"], color=MUTED, ls=":", lw=1)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_xlabel("AP sulla validazione (linea = prevalenza)")
        ax.set_title(f"{plant}: selezione del classificatore")
        ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def sensitivity(rows: list, path):
    import pandas as pd
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (sc, g) in zip(axes, d.groupby("scenario")):
        piv = g.pivot(index="under_ratio", columns="horizon_min", values="lift")
        im = ax.imshow(piv.values, cmap="Blues", aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if piv.values[i, j] > (np.nanmin(piv.values) + np.nanmax(piv.values)) / 2 else INK)
        ax.set_xticks(range(piv.shape[1]), [f"{c} min" for c in piv.columns])
        ax.set_yticks(range(piv.shape[0]), piv.index)
        ax.set_xlabel("orizzonte di previsione")
        ax.set_ylabel("soglia sottoprestazione")
        ax.set_title(f"{sc.replace('_to_', ' → ')}: lift AP / prevalenza")
        ax.grid(False)
    fig.savefig(path)
    plt.close(fig)


def example_timeline(feat, clf, thr, path, title: str, max_days: int = 3):
    """Un inverter con molti eventi: r_peer, etichette e rischio previsto."""
    from .models import predict_proba_full
    day = feat[feat["LABEL"] >= 0]
    starts = day[(day["LABEL"].isin([1, 2])) & (day.groupby("SOURCE_KEY")["LABEL"].shift(1) == 0)]
    if starts.empty:
        return
    inv = starts["SOURCE_KEY"].value_counts().index[0]
    d0 = starts[starts["SOURCE_KEY"] == inv]["DATE_TIME"].dt.normalize().value_counts().index[0]
    sel = feat[(feat["SOURCE_KEY"] == inv) & (feat["DATE_TIME"] >= d0 - pd_days(1))
               & (feat["DATE_TIME"] < d0 + pd_days(max_days - 1))].copy()
    sel = sel[sel["IRRADIATION"].fillna(0) > 0.02]
    p = predict_proba_full(clf, sel)
    risk = np.where(sel["STATE"] == 0, p[:, 1] + p[:, 2], np.nan)
    x = np.arange(len(sel))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1.3]})
    a1.plot(x, sel["R_PEER"].clip(upper=1.6), color=SERIES[0], lw=2, label="r_peer")
    for c, col in [(1, SERIES[3]), (2, SERIES[1]), (3, MUTED)]:
        m = (sel["LABEL"] == c).values
        a1.scatter(x[m], np.full(m.sum(), -0.08), s=10, color=col, marker="s", label=C.CLASS_NAMES[c])
    a1.axhline(C.UNDER_RATIO, color=MUTED, lw=0.8, ls="--")
    a1.set_ylabel("r_peer")
    a1.legend(frameon=False, fontsize=7, ncol=4, loc="upper right")
    a1.set_title(f"Inverter {inv}: {title}")
    a2.plot(x, risk, color=SERIES[1], lw=2, label="P(guasto elettrico entro 2 h)")
    a2.axhline(thr, color=MUTED, lw=1, ls="--", label=f"soglia {thr:.2f}")
    a2.set_ylabel("rischio")
    a2.legend(frameon=False, fontsize=7)
    ticks = np.unique(sel["DATE_TIME"].dt.normalize(), return_index=True)[1]
    a2.set_xticks(ticks, [str(sel["DATE_TIME"].iloc[i].date()) for i in ticks])
    a2.set_xlabel("istanti diurni consecutivi (le notti sono omesse)")
    fig.savefig(path)
    plt.close(fig)


def pd_days(n):
    import pandas as pd
    return pd.Timedelta(days=n)
