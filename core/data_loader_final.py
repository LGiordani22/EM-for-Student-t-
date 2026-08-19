"""
core/data_loader_final.py

Costruttore del dataset "final" — replica NY Fed (Bok, Caratelli, Giannone,
Sbordone, Tambalotti 2018, "Macroeconomic Nowcasting with Big Data"), Tabella 3.

PERCHÉ UN LOADER NUOVO (e non un ramo di data_loader.py)
-------------------------------------------------------
`data_loader.py` è cablato su FRED-MD: scarica il pannello `current.csv`, applica
i codici di trasformazione FRED-MD `1..7`, blocchi real/financial/other. Il
dataset "final" è un'altra cosa:
  * sorgente = FRED **diretto** (33 serie) + xlsx **manuale** (4 ISM),
  * vocabolario di trasformazioni a **5 verbi** (non i codici 1..7),
  * blocchi **G/S/R/L** con overlap (nei metadati; la struttura di fattore vive
    in `config/factor_specs.json`, è scelta di MODELLO non di dato).
Riusa quindi solo lo *scheletro* (config-driven, placement a fine trimestre,
contratto di output month-end con target ultimo) — non i codici FRED-MD.

SINGOLA FONTE DI VERITÀ: `config/series_final.json` (schema "fed-v1").

CONTRATTO DI OUTPUT (identico a quello che la pipeline kalman/em già legge)
--------------------------------------------------------------------------
  * indice = date **month-end**,
  * colonne nell'ordine del config (le 3 trimestrali in coda, **GDPC1 ultima**),
  * serie trimestrali posizionate SOLO sui mesi 3/6/9/12 (NaN altrove),
  * NaN per i buchi (partenze tardive, bordo frastagliato) — gestiti dal Kalman.

COSA PRODUCE
------------
  data/processed/final/dataset_final.csv    pannello TRASFORMATO (per la stima)
  data/processed/final/raw_levels_final.csv livelli GREZZI allineati (per le inverse)
  data/processed/final/metadata_final.csv   una riga per serie (block/freq/transform/
                                            delay/ref/release/start reale/%missing)
  data/processed/final/ism_release_dates.csv date di rilascio REALI delle 4 ISM
  config/calendar_example.md                tabellina esplicativa ragged-edge

USO
---
    python core/data_loader_final.py                 # scarica e salva tutto
    python core/data_loader_final.py --no-save        # solo in memoria + report
    python core/data_loader_final.py --calendar-only  # rigenera solo la tabellina

Distinzione onesta stima vs forecast:
  * la STIMA sull'intero campione usa dataset_final.csv (allineato per periodo di
    riferimento); il calendario di pubblicazione NON gioca alcun ruolo.
  * il FORECAST real-time usa as_of(panel, metadata, D): il ragged-edge emerge
    dai delay (pseudo-real-time, publication-lag only). Per le ISM c'è la data
    di rilascio reale.
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from fredapi import Fred

load_dotenv()

# ─── Percorsi ─────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(_PROJECT_ROOT, "config")
PROC_DIR = os.path.join(_PROJECT_ROOT, "data", "processed", "final")

DEFAULT_CONFIG = "final"

VALID_TRANSFORMS = {"MoM_log", "QoQ_log_ar", "diff_ppt", "diff_level", "level"}
VALID_SOURCES = {"fred", "manual_xlsx"}
VALID_FREQS = {"M", "Q"}
VALID_BLOCKS = {"G", "S", "R", "L"}


# ─── Config (schema fed-v1) ───────────────────────────────────────────────────

def load_final_config(name: str = DEFAULT_CONFIG) -> dict:
    """Carica e valida ``config/series_<name>.json`` (schema fed-v1)."""
    path = os.path.join(CONFIG_DIR, f"series_{name}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config non trovato: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    if cfg.get("schema") != "fed-v1":
        raise ValueError(
            f"Config '{name}' ha schema {cfg.get('schema')!r}, atteso 'fed-v1'. "
            f"Questo loader NON legge lo schema FRED-MD (usa data_loader.py per quello)."
        )

    series = cfg["series"]
    ids = [s["series_id"] for s in series]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"series_id duplicati: {dupes}")

    for s in series:
        sid = s["series_id"]
        for key in ("series_id", "source", "freq", "block_loadings", "transform", "inverse_fn"):
            if key not in s:
                raise ValueError(f"Serie {sid!r}: manca la chiave '{key}'.")
        if s["source"] not in VALID_SOURCES:
            raise ValueError(f"Serie {sid!r}: source '{s['source']}' non in {VALID_SOURCES}.")
        if s["freq"] not in VALID_FREQS:
            raise ValueError(f"Serie {sid!r}: freq '{s['freq']}' non in {VALID_FREQS}.")
        if s["transform"] not in VALID_TRANSFORMS:
            raise ValueError(f"Serie {sid!r}: transform '{s['transform']}' non in {VALID_TRANSFORMS}.")
        bl = s["block_loadings"]
        if "G" not in bl or not set(bl) <= VALID_BLOCKS:
            raise ValueError(f"Serie {sid!r}: block_loadings {bl} non valido (serve G ⊆ {VALID_BLOCKS}).")
        if s["source"] == "manual_xlsx" and "xlsx_col" not in s.get("transform_params", {}):
            raise ValueError(f"Serie {sid!r} (manual_xlsx): serve transform_params.xlsx_col.")

    return cfg


# ─── Helper temporali ─────────────────────────────────────────────────────────

def _to_month_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Aggancia ogni data all'ultimo giorno del suo mese."""
    return pd.DatetimeIndex(
        [pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(1)
         for y, m in zip(idx.year, idx.month)]
    )


def _quarter_start_to_qend_monthend(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Da inizio-trimestre (convenzione FRED: 2020-01-01 = Q1) al month-end
    dell'ultimo mese del trimestre (2020-01-01 → 2020-03-31).
    """
    return _to_month_end(idx + pd.DateOffset(months=2))


# ─── Trasformazioni (5 verbi) ─────────────────────────────────────────────────

def apply_final_transform(level: pd.Series, transform: str) -> pd.Series:
    """
    Applica una delle 5 trasformazioni del vocabolario fed-v1 a una serie di
    LIVELLI. `level` deve essere alla frequenza nativa della trasformazione
    (mensile per MoM/diff/level; **trimestrale** per QoQ_log_ar).

        MoM_log     100*log(X_t/X_{t-1})
        QoQ_log_ar  400*log(X_t/X_{t-1})       (annualizzazione IN LOG)
        diff_ppt    X_t - X_{t-1}   (punti percentuali)
        diff_level  X_t - X_{t-1}   (migliaia)
        level       X_t             (identità; già stazionario)

    PERCHÉ LOG E NON VARIAZIONE PERCENTUALE
    ---------------------------------------
    Il vincolo viene da Mariano & Murasawa (2003), l'aggregatore di frequenza
    che il paper cita e che il Kalman implementa (pesi {1/3,2/3,1,2/3,1/3}):
    quei pesi derivano dall'approssimazione geometrica della media trimestrale,
    quindi l'identità

        y^Q_t = 1/3 y_t + 2/3 y_{t-1} + y_{t-2} + 2/3 y_{t-3} + 1/3 y_{t-4}

    è ESATTA solo in log-differenze. Misurato su INDPRO (l'unica serie con sia
    il livello mensile sia l'aggregato trimestrale): rmse dell'identità MM
    0.0067 in log contro 0.0211 in variazione percentuale (×3.1).

    Molto più grave del secondo ordine sui mensili è la forma COMPOSTA
    dell'annualizzazione trimestrale, `100*((X_t/X_{t-1})^4 - 1)`, usata prima
    di questa versione: è una trasformazione CONVESSA, che comprime la coda
    sinistra e allunga la destra. Sul PIL 1985-oggi ribalta il segno
    dell'asimmetria — skew +0.18 composta contro -2.22 in log (2020Q2:
    -27.98 contro -32.82; 2020Q3: +34.86 contro +29.90). Per una tesi che
    modella proprio l'asimmetria della densità del PIL, il compound cancella
    nel dato l'oggetto che il modello deve stimare.

    Il reporting in unità BEA non ci perde nulla: la conversione
    `g_BEA = 100*(exp(x/100) - 1)` è monotona, quindi i quantili commutano e
    la GaR si riporta in unità ufficiali in modo esatto (vedi `inv_QoQ_log_ar`).
    """
    if transform == "MoM_log":
        return 100.0 * np.log(level / level.shift(1))
    if transform == "QoQ_log_ar":
        return 400.0 * np.log(level / level.shift(1))
    if transform in ("diff_ppt", "diff_level"):
        return level.diff()
    if transform == "level":
        return level.copy()
    raise ValueError(f"Trasformazione ignota: {transform!r}")


# ─── Fetch FRED ───────────────────────────────────────────────────────────────

def _get_fred_api_key(provided: str | None = None) -> str:
    key = provided or os.environ.get("FRED_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY non trovata. Impostala nell'ambiente / .env, "
            "oppure passala a build_dataset_final(fred_api_key=...)."
        )
    return key


def fetch_fred_level(fred: Fred, sid: str, freq: str) -> tuple[pd.Series, pd.Series]:
    """
    Scarica una serie FRED e restituisce (raw_level_ME, transform_input_level).

    Per le MENSILI: collassa la frequenza nativa a media mensile e aggancia al
    month-end; l'input della trasformazione è la stessa serie month-end.
    Per le TRIMESTRALI: tiene la serie trimestrale (per il rapporto QoQ) come
    input della trasformazione; il raw_level month-end è a fine trimestre.
    """
    s = fred.get_series(sid)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    if freq == "M":
        s_me = s.resample("ME").mean()
        s_me.name = sid
        return s_me, s_me           # raw ME e input trasformazione coincidono
    else:  # Q
        s_q = s.copy()
        s_q.name = sid
        raw_me = s_q.copy()
        raw_me.index = _quarter_start_to_qend_monthend(s_q.index)
        raw_me.name = sid
        return raw_me, s_q          # input trasformazione = serie trimestrale


# ─── Fetch ISM (xlsx) ─────────────────────────────────────────────────────────

def fetch_ism_panel(xlsx_path: str, sheet: str) -> pd.DataFrame:
    """
    Legge il foglio `panel` dell'xlsx ISM e restituisce un DataFrame month-end
    con le 4 colonne ISM (livelli). La riga 2 (0-based) contiene le intestazioni;
    la colonna 0 è `mese_riferimento` in formato 'YYYY-MM'.
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=2)
    raw = raw.rename(columns={raw.columns[0]: "mese_riferimento"})
    raw = raw.dropna(subset=["mese_riferimento"])
    idx = pd.to_datetime(raw["mese_riferimento"].astype(str), format="%Y-%m")
    df = raw.drop(columns=["mese_riferimento"]).copy()
    df.index = _to_month_end(pd.DatetimeIndex(idx))
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def fetch_ism_release_dates(xlsx_path: str, series_sheets: dict[str, str]) -> pd.DataFrame:
    """
    Estrae le date di rilascio REALI dalle schede singole ISM (mese_riferimento,
    data_rilascio, valore). Restituisce un df lungo: series_id, ref_month (ME),
    release_date. Base per l'as_of ESATTO delle ISM.
    """
    rows = []
    for sid, sheet in series_sheets.items():
        d = pd.read_excel(xlsx_path, sheet_name=sheet, header=2)
        d = d.rename(columns={d.columns[0]: "mese_riferimento",
                              d.columns[1]: "data_rilascio",
                              d.columns[2]: "valore"})
        d = d.dropna(subset=["mese_riferimento"])
        ref = _to_month_end(pd.DatetimeIndex(
            pd.to_datetime(d["mese_riferimento"].astype(str), format="%Y-%m")))
        rel = pd.to_datetime(d["data_rilascio"], errors="coerce")
        for r, rd in zip(ref, rel):
            rows.append({"series_id": sid, "ref_month": r, "release_date": rd})
    return pd.DataFrame(rows)


# ─── Pipeline principale ──────────────────────────────────────────────────────

def build_dataset_final(
    fred_api_key: str | None = None,
    save: bool = True,
    config: str | dict = DEFAULT_CONFIG,
) -> dict:
    """
    Pipeline completa: fetch FRED + ISM → trasforma → assembla → filtra → salva.

    Ritorna un dict con: 'panel' (trasformato), 'raw_levels', 'metadata',
    'ism_release_dates', 'config'.
    """
    cfg = load_final_config(config) if isinstance(config, str) else config
    series = cfg["series"]
    ordered = [s["series_id"] for s in series]
    sample_start = pd.Timestamp(cfg.get("sample_start", "1985-01-01"))

    key = _get_fred_api_key(fred_api_key)
    fred = Fred(api_key=key)

    raw_frames: dict[str, pd.Series] = {}
    tr_frames: dict[str, pd.Series] = {}

    # ── FRED ──────────────────────────────────────────────────────────────────
    fred_series = [s for s in series if s["source"] == "fred"]
    print(f"Scarico {len(fred_series)} serie FRED ...")
    for s in fred_series:
        sid, freq, transform = s["series_id"], s["freq"], s["transform"]
        raw_me, tr_input = fetch_fred_level(fred, sid, freq)
        tr = apply_final_transform(tr_input, transform)
        if freq == "Q":
            tr.index = _quarter_start_to_qend_monthend(tr.index)
        tr.name = sid
        raw_frames[sid] = raw_me
        tr_frames[sid] = tr
        first = tr.dropna().index.min()
        print(f"  {sid:<20s} [{freq}] {transform:<11s} first valid "
              f"{first.date() if first is not None else '—'}")

    # ── ISM (manual_xlsx) ─────────────────────────────────────────────────────
    ism_series = [s for s in series if s["source"] == "manual_xlsx"]
    ism_release = pd.DataFrame(columns=["series_id", "ref_month", "release_date"])
    if ism_series:
        xlsx_path = os.path.join(_PROJECT_ROOT, cfg["ism_source_xlsx"])
        sheet = cfg.get("ism_source_sheet", "panel")
        print(f"Inietto {len(ism_series)} serie ISM da {os.path.basename(xlsx_path)} "
              f"(foglio '{sheet}') ...")
        panel = fetch_ism_panel(xlsx_path, sheet)
        # mappa nome-scheda per le date di rilascio reali
        sheet_map = {"ISM_PMI": "PMI", "ISM_PRICES": "Prices",
                     "ISM_EMP": "Employment", "ISM_NMI": "NMI"}
        for s in ism_series:
            sid = s["series_id"]
            col = s["transform_params"]["xlsx_col"]
            if col not in panel.columns:
                raise KeyError(f"{sid}: colonna '{col}' assente nel foglio panel. "
                               f"Disponibili: {list(panel.columns)}")
            lvl = panel[col].copy()
            lvl.name = sid
            raw_frames[sid] = lvl
            tr_frames[sid] = apply_final_transform(lvl, s["transform"])  # 'level' = id
            tr_frames[sid].name = sid
            print(f"  {sid:<20s} [M] level       <- '{col}'  first valid "
                  f"{lvl.dropna().index.min().date()}")
        ism_release = fetch_ism_release_dates(
            xlsx_path, {sid: sheet_map[sid] for sid in [s['series_id'] for s in ism_series]})

    # ── Assembla ──────────────────────────────────────────────────────────────
    panel = pd.concat([tr_frames[c] for c in ordered], axis=1, sort=True)
    panel.columns = ordered
    raw = pd.concat([raw_frames[c] for c in ordered], axis=1, sort=True)
    raw.columns = ordered

    # filtra al campione e ordina
    panel = panel.loc[panel.index >= sample_start, ordered].copy()
    raw = raw.reindex(panel.index)[ordered].copy()

    # trimestrali: NaN fuori dai mesi 3/6/9/12 (sicurezza)
    non_qend = ~panel.index.month.isin([3, 6, 9, 12])
    for s in series:
        if s["freq"] == "Q":
            panel.loc[non_qend, s["series_id"]] = np.nan
            raw.loc[non_qend, s["series_id"]] = np.nan

    # ── Metadata (una riga per serie) ─────────────────────────────────────────
    meta_rows = []
    for s in series:
        sid = s["series_id"]
        valid = panel[sid].dropna()
        meta_rows.append({
            "series_id": sid,
            "paper_name": s.get("paper_name", ""),
            "source": s["source"],
            "freq": s["freq"],
            "block_loadings": "".join(s["block_loadings"]),
            "transform": s["transform"],
            "inverse_fn": s["inverse_fn"],
            "publication_delay_days": s.get("publication_delay_days"),
            "reference_period": s.get("reference_period"),
            "release_name": s.get("release_name"),
            "start_actual": valid.index.min().date() if not valid.empty else None,
            "pct_missing": round(float(panel[sid].isna().mean()) * 100, 1),
            "notes": s.get("notes", ""),
        })
    metadata = pd.DataFrame(meta_rows)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"  Dataset 'final' assemblato")
    print(f"  Shape      : {panel.shape[0]} mesi × {panel.shape[1]} serie")
    print(f"  Range      : {panel.index[0].date()} → {panel.index[-1].date()}")
    print(f"  Partenze tardive (start reale oltre ~3 mesi dopo sample_start):")
    late_threshold = sample_start + pd.Timedelta(days=90)
    for _, r in metadata.iterrows():
        if r["start_actual"] and pd.Timestamp(r["start_actual"]) > late_threshold:
            print(f"    {r['series_id']:<20s} start {r['start_actual']}  "
                  f"({r['pct_missing']}% missing)  {r['notes']}")
    print("─" * 70)

    # ── Salva ─────────────────────────────────────────────────────────────────
    if save:
        os.makedirs(PROC_DIR, exist_ok=True)
        panel.to_csv(os.path.join(PROC_DIR, "dataset_final.csv"))
        raw.to_csv(os.path.join(PROC_DIR, "raw_levels_final.csv"))
        metadata.to_csv(os.path.join(PROC_DIR, "metadata_final.csv"), index=False)
        if not ism_release.empty:
            ism_release.to_csv(os.path.join(PROC_DIR, "ism_release_dates.csv"), index=False)
        print(f"  Salvato in {PROC_DIR}")

    return {"panel": panel, "raw_levels": raw, "metadata": metadata,
            "ism_release_dates": ism_release, "config": cfg}


# ─── as_of: ragged-edge pseudo-real-time ──────────────────────────────────────

def as_of(panel: pd.DataFrame, metadata: pd.DataFrame, date) -> pd.DataFrame:
    """
    Ritaglia il pannello com'era DISPONIBILE alla data `date` (pseudo-real-time,
    publication-lag only). Un valore al mese di riferimento p è disponibile sse

        end_of_month(p) + publication_delay_days  ≤  date.

    Il delay È GIÀ misurato dalla fine del periodo di riferimento (Tab. 2, nota b):
    NON va ri-sommato il reference_period, altrimenti si conterebbe due volte.
    Restituisce una copia del pannello con NaN su tutto ciò che non era ancora
    pubblicato a `date`. Le trimestrali cadono naturalmente (fine trimestre + 28).
    """
    D = pd.Timestamp(date)
    out = panel.copy()
    delay = dict(zip(metadata["series_id"], metadata["publication_delay_days"]))
    for sid in out.columns:
        avail = out.index + pd.to_timedelta(int(delay[sid]), unit="D")
        out.loc[avail > D, sid] = np.nan
    return out


# ─── Tabellina esplicativa del calendario ─────────────────────────────────────

def build_calendar_example(config: str | dict = DEFAULT_CONFIG,
                           as_of_dates=("2016-09-16", "2020-05-15")) -> str:
    """
    Genera `config/calendar_example.md`: (1) tabella delle 37 serie ordinate per
    delay, (2) uno o più esempi worked di as_of che mostrano il bordo frastagliato
    (ultimo mese di riferimento disponibile per serie). Puramente illustrativa —
    non tocca la rete: usa il config e la logica dei delay.
    """
    cfg = load_final_config(config) if isinstance(config, str) else config
    series = sorted(cfg["series"], key=lambda s: s.get("publication_delay_days", 0))

    lines = []
    lines.append("# Calendario & ragged-edge — come le serie entrano nel tempo\n")
    lines.append("> Generato da `data_loader_final.build_calendar_example`. "
                 "Illustrativo: spiega la meccanica dei delay senza scaricare nulla.\n")
    lines.append("## 1. Da dove viene il delay\n")
    lines.append(
        "Il **delay** (Tab. 2 del paper, nota b) è in **giorni di calendario "
        "dalla FINE del periodo di riferimento**, congelato sul 2017. NON misura "
        "quanto è puntuale il rilascio nel mese, ma quanto è **vecchio il dato** "
        "che contiene.\n\n"
        "Esempio chiave — stessa regola di pubblicazione, delay opposti:\n\n"
        "| Release | Publication timing | Reference | Delay |\n"
        "|---|---|---|---|\n"
        "| ISM Manufacturing | first business day of the month | 1 mese prima | **3** |\n"
        "| Construction Spending | first business day of the month | 2 mesi prima | **33** |\n\n"
        "Construction Spending esce presto nel mese, ma il report del 1° giorno "
        "lavorativo di marzo descrive **gennaio** (2 mesi prima): da fine gennaio "
        "alla pubblicazione ci sono ~33 giorni. ISM descrive il mese appena chiuso "
        "→ ~3 giorni.\n\n"
        "Regola `as_of`: periodo *p* disponibile a *D* ⇔ "
        "`fine_mese(p) + delay ≤ D`. Il delay include già lo scarto dal periodo "
        "di riferimento → il campo `reference_period` è documentazione, non si "
        "ri-somma.\n")

    lines.append("## 2. Le 37 serie ordinate per delay (dal più tempestivo)\n")
    lines.append("| Serie | Blocco | Freq | Reference | Delay (gg) | Release |")
    lines.append("|---|---|---|---|---:|---|")
    for s in series:
        lines.append(
            f"| {s['series_id']} | {''.join(s['block_loadings'])} | {s['freq']} | "
            f"{s.get('reference_period','')} | {s.get('publication_delay_days')} | "
            f"{s.get('release_name','')} |")
    lines.append("")

    lines.append("## 3. Esempi di bordo frastagliato (`as_of`)\n")
    lines.append(
        "Per ogni `as_of` D: ultimo **mese di riferimento** di cui il dato sarebbe "
        "già pubblicato a D (= ultimo p con `fine_mese(p)+delay ≤ D`). Le serie con "
        "delay grande si fermano prima → il bordo è frastagliato, non piatto.\n")
    for D in as_of_dates:
        Dts = pd.Timestamp(D)
        lines.append(f"### as_of = {D}\n")
        lines.append("| Serie | Delay | Ultimo mese disponibile |")
        lines.append("|---|---:|---|")
        for s in series:
            delay = int(s.get("publication_delay_days", 0))
            # ultimo periodo di riferimento p tale che p+delay <= D.
            # Le trimestrali esistono solo a fine trimestre (mesi 3/6/9/12).
            months = pd.date_range(end=Dts + pd.offsets.MonthEnd(0), periods=40, freq="ME")
            if s["freq"] == "Q":
                months = months[months.month.isin([3, 6, 9, 12])]
            avail = months[(months + pd.to_timedelta(delay, unit="D")) <= Dts]
            last = avail.max().strftime("%Y-%m") if len(avail) else "—"
            lines.append(f"| {s['series_id']} | {delay} | {last} |")
        lines.append("")

    text = "\n".join(lines)
    out_path = os.path.join(CONFIG_DIR, "calendar_example.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Tabellina scritta in {out_path}")
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Costruttore dataset 'final' (replica NY Fed).")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--no-save", action="store_true", help="non scrivere i CSV")
    ap.add_argument("--calendar-only", action="store_true",
                    help="rigenera solo config/calendar_example.md (nessun download)")
    args = ap.parse_args()

    if args.calendar_only:
        build_calendar_example(args.config)
        return

    build_dataset_final(fred_api_key=args.api_key, save=not args.no_save, config=args.config)
    build_calendar_example(args.config)


if __name__ == "__main__":
    main()
