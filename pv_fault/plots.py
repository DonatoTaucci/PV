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


def twin_scatter(preds: dict, path):
    keys = [k for k in preds if k.startswith("twin_")]
    fig, axes = plt.subplots(1, len(keys), figsize=(9, 4))
    for ax, k in zip(np.atleast_1d(axes), keys):
        y, yhat = preds[k]
        ax.hexbin(y, yhat, gridsize=50, cmap="Blues", mincnt=1, bins="log")
        lim = [0, max(y.max(), yhat.max())]
        ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
        ax.set_title(k.replace("twin_", "").replace("_", " "))
        ax.set_xlabel("P_NORM misurata")
        ax.set_ylabel("P_NORM prevista")
    fig.savefig(path)
    plt.close(fig)


def confusion_matrices(preds: dict, path):
    keys = [k for k in preds if k.startswith("cm_") and "cross" in k]
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
        ax.set_title(k.replace("cm_", "").replace("_", " "))
        ax.grid(False)
    fig.savefig(path)
    plt.close(fig)
