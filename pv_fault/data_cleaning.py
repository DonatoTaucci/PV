"""Caricamento e pulizia dei dati SCADA (generazione per inverter + stazione meteo).

Ogni passo della pulizia scrive i propri numeri in un dizionario ``report`` in
modo che la tesi possa riportare *quanto* è stato corretto, non solo *come*.

Problemi emersi dall'analisi esplorativa dei due impianti Kaggle:
  1. formati data diversi tra file (dd-mm-YYYY HH:MM vs ISO);
  2. DC_POWER dell'impianto 1 in scala x10 (AC/DC ~= 0.098, fisicamente impossibile);
  3. timestamp mancanti (per impianto, per singolo inverter, solo meteo);
  4. griglie temporali di generazione e meteo non identiche (slot presenti in un
     file e non nell'altro);
  5. contatori TOTAL_YIELD non monotoni / azzerati (impianto 2) e DAILY_YIELD che
     non si azzera a mezzanotte ma al primo campione produttivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

GEN_COLS = ["DATE_TIME", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]
WEATHER_COLS = ["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]


# ---------------------------------------------------------------------------
# Caricamento
# ---------------------------------------------------------------------------
def parse_datetime(s: pd.Series, fmt: str | None = None) -> pd.Series:
    """Parsing esplicito se il formato è noto, altrimenti inferenza con dayfirst."""
    if fmt:
        return pd.to_datetime(s, format=fmt)
    try:
        return pd.to_datetime(s, format="ISO8601")
    except (ValueError, TypeError):
        return pd.to_datetime(s, dayfirst=True, format="mixed")


def load_plant(plant_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = C.PLANTS[plant_id]
    gen = pd.read_csv(cfg["generation"])
    wea = pd.read_csv(cfg["weather"])
    gen["DATE_TIME"] = parse_datetime(gen["DATE_TIME"], cfg["gen_datetime_format"])
    wea["DATE_TIME"] = parse_datetime(wea["DATE_TIME"])
    return gen, wea


def split_merged(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per l'upload dal sito: un unico CSV con colonne di generazione e meteo."""
    df = df.copy()
    df["DATE_TIME"] = parse_datetime(df["DATE_TIME"])
    for c in ["DAILY_YIELD", "TOTAL_YIELD"]:
        if c not in df:
            df[c] = np.nan
    gen = df[GEN_COLS]
    wea = df[WEATHER_COLS].groupby("DATE_TIME", as_index=False).mean()
    return gen, wea


# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------
def _gap_runs(missing: pd.DatetimeIndex) -> pd.DataFrame:
    """Raggruppa i timestamp mancanti in 'buchi' consecutivi."""
    if len(missing) == 0:
        return pd.DataFrame(columns=["start", "end", "slots"])
    s = missing.to_series()
    run_id = (s.diff() != pd.Timedelta(C.FREQ)).cumsum()
    runs = s.groupby(run_id.values).agg(["min", "max", "count"])
    runs.columns = ["start", "end", "slots"]
    return runs.reset_index(drop=True)


def _night_slots(wea: pd.DataFrame) -> set:
    """Slot della giornata (HH:MM) in cui l'irraggiamento è sempre ~0.

    Profilo empirico sull'intero dataset; con pochi giorni (upload) si ricade
    su una regola oraria conservativa.
    """
    w = wea.dropna(subset=["IRRADIATION"])
    if w["DATE_TIME"].dt.normalize().nunique() >= 3:
        prof = w.groupby(w["DATE_TIME"].dt.time)["IRRADIATION"].max()
        return set(prof[prof < C.NIGHT_IRR].index)
    return {t for t in pd.date_range("2000-01-01", periods=C.SLOTS_PER_DAY, freq=C.FREQ).time
            if t.hour < 6 or t.hour >= 19}


def _best_lag(plant_dc: pd.Series, irr: pd.Series, max_lag: int = 4) -> tuple[int, dict]:
    """Correlazione incrociata potenza d'impianto vs irraggiamento.

    Serve a verificare che i due sistemi di acquisizione abbiano lo stesso
    orologio (nessuno sfasamento sistematico di uno o più slot).
    """
    corr = {lag: float(plant_dc.corr(irr.shift(lag))) for lag in range(-max_lag, max_lag + 1)}
    return max(corr, key=corr.get), corr


# ---------------------------------------------------------------------------
# Pulizia
# ---------------------------------------------------------------------------
def clean(gen: pd.DataFrame, wea: pd.DataFrame, name: str = "plant") -> tuple[pd.DataFrame, dict]:
    """Restituisce un DataFrame su griglia completa (timestamp x inverter) e un report."""
    rep: dict = {"name": name}
    gen = gen.copy()
    wea = wea.copy()
    rep["raw_rows_generation"] = len(gen)
    rep["raw_rows_weather"] = len(wea)
    rep["n_inverters"] = int(gen["SOURCE_KEY"].nunique())

    # 1) Valori negativi / NaN espliciti ------------------------------------
    num_g = ["DC_POWER", "AC_POWER"]
    rep["negative_power_values"] = int((gen[num_g] < 0).sum().sum())
    gen[num_g] = gen[num_g].where(gen[num_g] >= 0)
    rep["explicit_nan_generation"] = int(gen[num_g].isna().sum().sum())
    rep["explicit_nan_weather"] = int(wea[WEATHER_COLS[1:]].isna().sum().sum())

    # 2) Allineamento alla griglia di 15 minuti ------------------------------
    for df, key in [(gen, "generation"), (wea, "weather")]:
        off = df["DATE_TIME"] != df["DATE_TIME"].dt.round(C.FREQ)
        rep[f"off_grid_timestamps_{key}"] = int(off.sum())
        df["DATE_TIME"] = df["DATE_TIME"].dt.round(C.FREQ)

    # 3) Duplicati (dopo l'arrotondamento): media ------------------------------
    rep["duplicates_generation"] = int(gen.duplicated(["DATE_TIME", "SOURCE_KEY"]).sum())
    rep["duplicates_weather"] = int(wea.duplicated(["DATE_TIME"]).sum())
    gen = gen.groupby(["DATE_TIME", "SOURCE_KEY"], as_index=False).mean(numeric_only=True)
    wea = wea.groupby("DATE_TIME", as_index=False)[WEATHER_COLS[1:]].mean()

    # 4) Scala di DC_POWER -----------------------------------------------------
    prod = gen[(gen["DC_POWER"] > 0) & (gen["AC_POWER"] > 0)]
    eff = float((prod["AC_POWER"] / prod["DC_POWER"]).median()) if len(prod) else np.nan
    factor = 1.0
    if np.isfinite(eff) and not (C.EXPECTED_EFFICIENCY_RANGE[0] <= eff <= C.EXPECTED_EFFICIENCY_RANGE[1]):
        factor = float(10 ** np.round(np.log10(eff / 0.98)))  # es. 0.098 -> 0.1
        gen["DC_POWER"] = gen["DC_POWER"] * factor
    rep["median_ac_dc_ratio_raw"] = eff
    rep["dc_scale_factor_applied"] = factor

    # 5) Griglia completa --------------------------------------------------------
    t0 = min(gen["DATE_TIME"].min(), wea["DATE_TIME"].min()).normalize()
    t1 = max(gen["DATE_TIME"].max(), wea["DATE_TIME"].max()).normalize() + pd.Timedelta("23:45:00")
    grid = pd.date_range(t0, t1, freq=C.FREQ)
    rep["grid_slots"] = len(grid)
    rep["days"] = int(len(grid) / C.SLOTS_PER_DAY)

    gen_ts = pd.DatetimeIndex(gen["DATE_TIME"].unique())
    wea_ts = pd.DatetimeIndex(wea["DATE_TIME"].unique())
    rep["slots_missing_generation_all_inverters"] = int(len(grid.difference(gen_ts)))
    rep["slots_missing_weather"] = int(len(grid.difference(wea_ts)))
    rep["slots_generation_without_weather"] = int(len(gen_ts.difference(wea_ts)))
    rep["slots_weather_without_generation"] = int(len(wea_ts.difference(gen_ts)))
    rep["weather_gap_runs"] = _gap_runs(grid.difference(wea_ts)).assign(
        start=lambda d: d.start.astype(str), end=lambda d: d.end.astype(str)).to_dict("records")
    rep["generation_gap_runs"] = _gap_runs(grid.difference(gen_ts)).assign(
        start=lambda d: d.start.astype(str), end=lambda d: d.end.astype(str)).to_dict("records")

    # 6) Meteo: imputazione controllata -------------------------------------------
    night = _night_slots(wea)
    w = wea.set_index("DATE_TIME").reindex(grid)
    w.index.name = "DATE_TIME"
    was_missing = w["IRRADIATION"].isna()
    is_night = pd.Series(w.index.time, index=w.index).isin(night)
    # lunghezza del buco a cui appartiene ogni slot mancante
    run_id = (~was_missing).cumsum()
    run_len = was_missing.groupby(run_id).transform("sum")
    short_gap = was_missing & (run_len <= C.MAX_WEATHER_INTERP_SLOTS)
    run_all_night = (was_missing & is_night).groupby(run_id).transform("sum") == run_len
    long_night_gap = was_missing & ~short_gap & run_all_night

    # irraggiamento: 0 di notte (fisico), interpolazione solo per buchi brevi
    irr_interp = w["IRRADIATION"].interpolate(method="time", limit_area="inside")
    w["IRRADIATION"] = np.where(is_night & was_missing, 0.0,
                                np.where(short_gap, irr_interp, w["IRRADIATION"]))
    # temperature: buchi brevi o interamente notturni (andamento lento e monotono)
    for c in ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]:
        interp = w[c].interpolate(method="time", limit_area="inside")
        w[c] = np.where(short_gap | long_night_gap, interp, w[c])
    w["WEATHER_IMPUTED"] = was_missing & w["IRRADIATION"].notna()
    rep["weather_slots_imputed_short_gap"] = int(short_gap.sum())
    rep["weather_slots_imputed_night"] = int((was_missing & is_night & ~short_gap).sum())
    rep["weather_slots_left_missing"] = int(w["IRRADIATION"].isna().sum())
    rep["night_slots_per_day"] = len(night)

    # 7) Verifica sincronizzazione orologi generazione/meteo -----------------------
    plant_dc = gen.groupby("DATE_TIME")["DC_POWER"].sum().reindex(grid)
    lag, corr = _best_lag(plant_dc, w["IRRADIATION"])
    rep["clock_lag_best_slots"] = int(lag)
    rep["clock_lag_correlations"] = {str(k): round(v, 4) for k, v in corr.items()}
    if lag != 0:  # correggiamo solo se lo sfasamento è evidente
        w[WEATHER_COLS[1:]] = w[WEATHER_COLS[1:]].shift(lag)

    # 8) Generazione su griglia completa --------------------------------------------
    inverters = sorted(gen["SOURCE_KEY"].unique())
    full_idx = pd.MultiIndex.from_product([grid, inverters], names=["DATE_TIME", "SOURCE_KEY"])
    g = gen.set_index(["DATE_TIME", "SOURCE_KEY"]).reindex(full_idx).reset_index()
    g = g.merge(w.reset_index(), on="DATE_TIME", how="left")
    missing = g["DC_POWER"].isna()
    g_night = pd.Series(g["DATE_TIME"].dt.time).isin(night).values
    # di notte la produzione è 0 per definizione: il record mancante non è un guasto
    fill_night = missing & g_night
    g.loc[fill_night, ["DC_POWER", "AC_POWER"]] = 0.0
    n_ts_missing = g[missing & ~g_night].groupby("DATE_TIME").size()
    plant_wide = n_ts_missing[n_ts_missing == len(inverters)].index
    g["RAW_MISSING"] = missing
    g["COMM_LOSS"] = missing & ~g_night
    g["COMM_LOSS_PLANT_WIDE"] = g["COMM_LOSS"] & g["DATE_TIME"].isin(plant_wide)
    rep["expected_generation_rows"] = len(full_idx)
    rep["generation_rows_missing"] = int(missing.sum())
    rep["generation_rows_filled_night_zero"] = int(fill_night.sum())
    rep["generation_rows_comm_loss_daylight"] = int(g["COMM_LOSS"].sum())
    rep["generation_rows_comm_loss_plant_wide"] = int(g["COMM_LOSS_PLANT_WIDE"].sum())

    # 9) Controlli di plausibilità fisica ---------------------------------------------
    rep["ac_gt_dc_rows"] = int(((g["AC_POWER"] > g["DC_POWER"] * 1.001) & (g["DC_POWER"] > 1)).sum())
    rep["irradiation_gt_1_5"] = int((g["IRRADIATION"] > 1.5).sum() / max(len(inverters), 1))
    day = g["IRRADIATION"] >= C.DAYLIGHT_IRR
    rep["daylight_rows_zero_dc"] = int((day & (g["DC_POWER"] == 0)).sum())
    rep["night_rows_positive_dc"] = int(((g["IRRADIATION"] < C.NIGHT_IRR) & (g["DC_POWER"] > 1)).sum())

    # 10) Contatori energetici (solo diagnostica: NON usati come feature) ---------------
    gs = g.sort_values(["SOURCE_KEY", "DATE_TIME"])
    ty = gs.groupby("SOURCE_KEY")["TOTAL_YIELD"].apply(lambda s: s.dropna().diff())
    rep["total_yield_decreases"] = int((ty < 0).sum())
    rep["total_yield_zero_values"] = int((gs["TOTAL_YIELD"] == 0).sum())
    midnight = gs[gs["DATE_TIME"].dt.time == pd.Timestamp("00:00").time()]
    rep["daily_yield_nonzero_at_midnight_frac"] = float((midnight["DAILY_YIELD"] > 0).mean()) \
        if midnight["DAILY_YIELD"].notna().any() else None
    dy = gs.groupby(["SOURCE_KEY", gs["DATE_TIME"].dt.date])["DAILY_YIELD"].diff()
    ok = (gs["AC_POWER"] > 100) & dy.notna()
    rep["daily_yield_vs_ac_energy_ratio"] = float((dy[ok] / (gs.loc[ok, "AC_POWER"] * 0.25)).median()) \
        if ok.any() else None

    g = g.drop(columns=["PLANT_ID"], errors="ignore")
    g["PLANT"] = name
    return g.sort_values(["SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True), rep


def load_clean_plant(plant_id: int) -> tuple[pd.DataFrame, dict]:
    gen, wea = load_plant(plant_id)
    return clean(gen, wea, name=f"Plant_{plant_id}")
