"""
src/forecast/metrics_tables.py

LE TABELLE, per finestra e per famiglia.  Un lettore puro: non stima niente,
legge i CSV lunghi e li aggrega.

    python -m src.forecast.metrics_tables
    python -m src.forecast.metrics_tables --window 2007-2010

TRE POSTI, NON UNO PER METRICA
------------------------------
MDA e SignAcc sono COLONNE, non cartelle: stanno nella stessa riga di RMSE
perche' ogni domanda vera ("lo student_t e' meglio?") le vuole affiancate.
Gli assi su cui si organizza sono PERIODO e METODO.

    dfm/<spec>/rmse/metrics_<spec>.{csv,txt}   le 5 varianti di quella spec
    bvar/rmse/metrics_bvar.{csv,txt}           i 4 BVAR insieme
    comparison/                                tutti contro tutti

Un file per famiglia, non uno per finestra: il CSV e' in formato LUNGO con una
colonna `window`, quindi si pivota.  Sei finestre x tre spec x due forme
sarebbero trentasei file da aprire a mano.

LA MATRICE DEL CONFRONTO E IL SUO VINCOLO
-----------------------------------------
`comparison/` porta una matrice metodi x finestre: si legge in colonna chi
vince in quel regime, in riga chi e' stabile.

Ma un RMSE su quaranta trimestri e uno su dodici non si confrontano.  La
matrice e' quindi calcolata sull'INTERSEZIONE dei punti punteggiati da TUTTI i
metodi — stesse coppie (trimestre, settimana) per tutti — e riporta quanti
punti sono sopravvissuti.  Senza quel vincolo la tabella sembrerebbe funzionare
e direbbe una cosa falsa.

  CAVEAT DICHIARATO, non risolto: l'RMSE grezzo in punti BEA NON e'
  confrontabile fra colonne.  Nel 2020 sbagliano tutti di piu', quindi un 8 su
  2019-2021 non e' peggio di un 3 su 2014-2016 — e' un periodo piu' difficile.
  Dentro una colonna il confronto e' valido; fra colonne serve la colonna
  `RMSE_rel_ar2`, che normalizza sul benchmark dello stesso periodo.  Le due
  matrici sono affiancate apposta.

DUE COLONNE, NON UNA: `n` E `n_com`
-----------------------------------
Le tabelle di famiglia riportano OGNI metrica due volte — sul campione LIBERO
(tutto quello che quel metodo ha punteggiato) e sul campione COMUNE (i punti
punteggiati da tutti i metodi della tabella).  Non si sceglie fra le due: si
mostrano affiancate, perche' la distanza fra loro E' il dato.

Misurato sulla passata 2007-07/2010-06, tabella `fed_overlap`:

    metodo                   n     RMSE  |  n_com  RMSE_com
    fed_overlap/student_t  360    3.025  |    264     2.774
    nyfed                  309    3.079  |    264     3.305

Sul campione libero e' un pareggio; sul comune sono cinquecento millesimi di
punto BEA.  La ragione e' che la Fed non pubblica prima della settimana -4,
quindi i 96 punti di forecast profondo — i piu' difficili per tutti — sono
addebitati solo a me.  Con una colonna sola la tabella avrebbe detto "pareggio"
senza modo di accorgersene; con due, dichiara da sola su che cosa si legge.

IL CONFRONTO PER FASE, E PERCHE' IL BACKCAST HA UN PANNELLO SUO
----------------------------------------------------------------
`comparison_matrices` aggrega su TUTTE le fasi: e' un numero solo per metodo, e
il backcast — la fase in cui i modelli si distinguono di piu' e su cui poggia la
conclusione — ci sparisce dentro.  Peggio: siccome l'intersezione e' su tutti i
metodi insieme, UN modello privo di bordo (settimane > 13) cancella il backcast
a TUTTI, e la matrice continua a stampare un numero che sembra completo.

`comparison_by_phase` fa quindi l'intersezione DENTRO ciascuna fase.  Un modello
senza bordo perde il backcast per se' e non lo toglie agli altri; un modello con
copertura troppo rada in una fase viene escluso da QUELLA fase sola
(`_MIN_PHASE_COVERAGE`) invece di decimarne il campione, e il nome di chi e'
uscito finisce nella nota.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from src import output_layout as layout
from src.forecast import compute_metrics as cm

#: Le finestre in cui si riportano le metriche: le tre passate complete e i
#: tre zoom.  Le finestre "forecast" servono alle figure, non alle tabelle.
TABLE_WINDOWS: list[str] = list(layout.RMSE_PASSES) + list(layout.RMSE_ZOOM_WINDOWS)

#: Le colonne riportate, nell'ordine in cui si leggono: prima quante
#: osservazioni, poi la distanza, poi la direzione.  Le `*_com` sono le stesse
#: metriche sul campione comune e stanno APPAIATE alla loro versione libera,
#: non in fondo: si leggono a coppie.
#: `n_trimestri` sta accanto a `window` e non in fondo perche' e' parte
#: dell'intestazione, non una metrica: dice su quanti trimestri e' calcolata
#: TUTTA la riga.  E' lo stesso numero che la figura per orizzonte scrive nel
#: titolo — da quando le due leggono lo stesso campione, coincide.
_COLS = ["window", "n_trimestri", "metodo", "n", "n_com", "RMSE", "RMSE_com",
         "RMSE_rel_ar2", "MAE", "Bias", "corr", "MDA", "MDA_com", "n_dir",
         "SignAcc"]

#: Le metriche che vengono ricalcolate sul campione comune e affiancate.
_COMMON_COLS = {"n": "n_com", "RMSE": "RMSE_com", "MDA": "MDA_com"}

#: SOGLIA DI ANOMALIA, non di maggioranza.  In una fase, un metodo che copre
#: meno di questa quota dei punti (trimestre, settimana) ne viene escluso
#: invece di restringere il campione di tutti gli altri.
#:
#: DIECI PER CENTO E NON META', e la differenza e' sostanziale.  Con una soglia
#: alta si butterebbe fuori un modello che ha semplicemente girato su un
#: periodo piu' corto — il C-BVAR su dodici trimestri contro i quaranta del DFM
#: sta al 30%, ed e' un caso perfettamente legittimo in cui la risposta giusta
#: e' RESTRINGERE a dodici, non espellerlo.  Restringere e' il comportamento
#: normale; l'espulsione serve solo contro l'anomalia — un modello con una
#: manciata di punti che imporrebbe quella manciata come campione a tutti.
#: Sotto il 10% si e' in quel caso e non nell'altro.
_MIN_PHASE_COVERAGE = 0.10

#: Il nome con cui la Fed compare fra i metodi (lo stesso di `compare_nyfed`).
_NYFED = "nyfed"

#: Sotto questo numero di punti comuni la colonna della matrice non e' una
#: misura di accuratezza: e' un aneddoto su poche settimane.  Non si nasconde,
#: si etichetta — ma si etichetta in modo che non si possa leggerla per sbaglio.
_MIN_COMPARISON_POINTS = 40

_HEADER = (
    "WEEKLY NOWCAST — ACCURACY TABLES\n"
    "error in BEA percentage points (nowcast - realised)\n"
    "\n"
    "distance : RMSE, MAE, Bias.  RMSE_rel_ar2 < 1 beats the AR(2) benchmark.\n"
    "direction: MDA     = gets the direction of change vs the last published GDP\n"
    "           SignAcc = gets the sign of growth (expansion vs contraction)\n"
    "           0.50 = coin toss, 1.00 = always right; n_dir = judgeable rows\n"
    "\n"
    "TWO SAMPLES, SIDE BY SIDE — READ THEM AS A PAIR\n"
    "  n     / RMSE     / MDA       every point THAT method scored\n"
    "  n_com / RMSE_com / MDA_com   only the (quarter, week) points scored by\n"
    "                               EVERY method in this table\n"
    "Only the `_com` columns compare across rows.  The free columns can differ\n"
    "in sample size — a method that starts later, or stops at week 13 instead\n"
    "of 17, is averaged over a different set of targets — so a row with a\n"
    "smaller `n` may look better for reasons that have nothing to do with the\n"
    "model.  The gap between RMSE and RMSE_com measures exactly that.\n"
    "\n"
    "READING ACROSS WINDOWS: raw RMSE is NOT comparable between periods — a\n"
    "harder period raises it for everyone.  Compare methods WITHIN a window,\n"
    "or use RMSE_rel_ar2, which divides by the benchmark of that same period.\n"
)


# ─── Caricamento ──────────────────────────────────────────────────────────────

def load_dfm(paths: list[str] | None = None) -> pd.DataFrame:
    """I CSV settimanali del DFM, gia' punteggiati."""
    df, _, _ = cm.load_long(paths)
    return df


def load_nyfed(dfm: pd.DataFrame, paths: list[str] | None = None) -> pd.DataFrame:
    """
    Il NY Fed Staff Nowcast come UN METODO IN PIU', non come report separato.

    La sua struttura lunga (`nyfed_nowcast.load_long`) ha gia' `target_quarter`,
    `horizon_week` e `nowcast_bea` nel nostro metro; manca il realizzato, che si
    prende dal frame del DFM — e' lo stesso GDPC1, non una seconda fonte.

    Si tengono solo le righe `pre_release`: un "backcast" pubblicato DOPO
    l'uscita del PIL non e' una previsione, e includerlo regalerebbe alla Fed
    un vantaggio che non ha.
    """
    from src.forecast.nyfed_nowcast import load_long as _load_fed
    try:
        fed = _load_fed(paths)
    except Exception as exc:                       # dati Fed assenti = niente riga
        print(f"  [nyfed] non caricabile ({type(exc).__name__}: {exc}); "
              f"le tabelle useranno solo i miei metodi")
        return pd.DataFrame()

    fed = fed[fed["pre_release"]].copy()
    real = (dfm.dropna(subset=["realizzato_bea"])
            .drop_duplicates("target_quarter")
            .set_index("target_quarter")["realizzato_bea"])
    out = pd.DataFrame({
        "as_of": fed["forecast_date"],
        "target_quarter": fed["target_quarter"].astype(str),
        "horizon_week": fed["horizon_week"].astype(int),
        "nowcast_bea": fed["nowcast_bea"].astype(float),
        "metodo": _NYFED,
    })
    out["realizzato_bea"] = out["target_quarter"].map(real)
    out = out.dropna(subset=["realizzato_bea", "nowcast_bea"])
    out["errore"] = out["nowcast_bea"] - out["realizzato_bea"]
    out["fase"] = out["horizon_week"].map(cm._phase)
    return cm._add_direction(out)


def load_bvar(paths: list[str] | None = None) -> pd.DataFrame:
    """
    Gli stessi campi, dai CSV del BVAR.  Il contratto di colonne e' identico
    (`weekly_nowcast.COLUMNS`), quindi si riusa la preparazione del DFM invece
    di riscriverla — e' la ragione per cui al Gate 6 lo schema non e' cambiato.
    """
    if paths is None:
        paths = sorted(glob.glob(os.path.join(layout.bvar_csv_dir(),
                                              "bvar_realtime_*.csv")))
        if not paths:
            # Ripiego sul pilota finche' la passata vera non e' girata: e'
            # l'unico dato BVAR che esiste prima del lancio sul server.
            paths = sorted(glob.glob(os.path.join(
                os.path.dirname(layout.OUTPUT_ROOT), "_pilot_bvar", "csv",
                "bvar_realtime_*.csv")))
            if paths:
                print(f"  [bvar] passata vera assente: uso il PILOTA "
                      f"({len(paths)} file)")
    if not paths:
        return pd.DataFrame()
    return cm.load_long(list(paths))[0]


# ─── Le tabelle di famiglia ───────────────────────────────────────────────────

def _methods_of(df: pd.DataFrame, family: str) -> list[str]:
    """I metodi di una famiglia, benchmark SEMPRE inclusi (servono al relativo)."""
    keep = [m for m in df["metodo"].unique()
            if m in layout.BENCHMARKS or m == _NYFED
            or (family == "bvar" and m.split("/")[0] in layout.BVAR_MODELS)
            or (family != "bvar" and m.split("/")[0] == family)]
    return cm._ordered_methods(keep)


def _drop_fed_outside(t: pd.DataFrame) -> pd.DataFrame:
    """
    La riga NY Fed solo dove il confronto e' previsto (`NYFED_COMPARISON_PASSES`).

    Sulle finestre piu' recenti la Fed non pubblica un nowcast comparabile:
    lasciare la riga darebbe un confronto che non esiste.
    """
    if t.empty or "window" not in t.columns:
        return t
    bad = (t["metodo"] == _NYFED) & ~t["window"].isin(layout.NYFED_COMPARISON_PASSES)
    return t[~bad].reset_index(drop=True)


def _merge_common(free: pd.DataFrame, common: pd.DataFrame,
                  on: list[str]) -> pd.DataFrame:
    """
    Affianca alle colonne libere le stesse metriche sul campione comune.

    Un metodo assente dal campione comune — perche' non copre nessuno dei punti
    condivisi — resta in tabella con le sue colonne libere e `_com` a NaN: e'
    l'informazione giusta, non un buco.  Toglierlo lo nasconderebbe.
    """
    if free.empty:
        return free
    if common is None or common.empty:
        for c in _COMMON_COLS.values():
            free[c] = float("nan")
        return free
    keep = [k for k in list(on) + list(_COMMON_COLS) if k in common.columns]
    c = common[keep].rename(columns=_COMMON_COLS)
    return free.merge(c, on=on, how="left")


def common_points_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    `common_points` applicato DENTRO ciascuna fase, non sull'insieme.

    E' la differenza che fa esistere il confronto di backcast.  Sull'insieme,
    un modello privo di bordo (nessuna settimana > 13) non ha nessun punto di
    backcast da condividere, quindi l'intersezione globale scarta il backcast
    di TUTTI e la tabella continua a mostrare un RMSE che sembra completo.
    Fase per fase, quel modello e' semplicemente assente dal backcast — dove
    non ha girato non compare — e gli altri restano confrontabili fra loro.
    """
    if df.empty or "fase" not in df.columns:
        return df
    out = [common_points(g) for _, g in df.groupby("fase")]
    out = [o for o in out if not o.empty]
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


def family_tables(df: pd.DataFrame, family: str,
                  windows: list[str] = TABLE_WINDOWS
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Formato lungo: (per metodo, per metodo x fase), con la colonna `window`.

    Ogni metrica compare DUE VOLTE: sul campione libero e sul campione comune
    ai metodi della tabella (`_COMMON_COLS`).  Nella tabella per fase il comune
    e' calcolato DENTRO la fase, cosi' che il backcast di un modello senza
    bordo non cancelli quello degli altri.

    Le finestre senza dati non compaiono — non e' un errore, e' che quella
    passata non copre quel periodo, e una riga di NaN sarebbe peggio del
    silenzio.
    """
    sub = df[df["metodo"].isin(_methods_of(df, family))]
    by_m, by_mp = [], []
    for w in windows:
        # La Fed non vota sul campione: comincia a -3/-4 per scelta editoriale,
        # non per un buco, e chiederle l'asse intero lo svuoterebbe per tutti.
        # Resta pero' nella tabella, misurata sugli stessi trimestri.
        d, tenuti, _ = cm.window_sample(sub, w, skip=[_NYFED])
        if d.empty:
            continue
        c = common_points(d)
        t = _merge_common(cm.table_by_method(d),
                          cm.table_by_method(c) if not c.empty else None,
                          on=["metodo"])
        t.insert(0, "window", w)
        t.insert(1, "n_trimestri", len(tenuti))
        by_m.append(t)

        cp = common_points_by_phase(d)
        tp = _merge_common(cm.table_by_method_phase(d),
                           cm.table_by_method_phase(cp) if not cp.empty else None,
                           on=["metodo", "fase"])
        tp.insert(0, "window", w)
        by_mp.append(tp)

    m = pd.concat(by_m, ignore_index=True) if by_m else pd.DataFrame()
    mp = pd.concat(by_mp, ignore_index=True) if by_mp else pd.DataFrame()
    if not m.empty:
        m = m[[c for c in _COLS if c in m.columns]]
    if not mp.empty:
        head = ["window", "metodo", "fase"]
        mp = mp[head + [c for c in _COLS if c in mp.columns and c not in head]]
    return _drop_fed_outside(m), _drop_fed_outside(mp)


def coverage(df: pd.DataFrame, w: str) -> dict:
    """
    Quanto della finestra e' COPERTO DAI DATI, non quanto la finestra dichiara.

    Serve perche' una tabella intestata "2007-2025" calcolata su un CSV che si
    ferma al 2010 non e' vuota: e' PIENA DI NUMERI GIUSTI SOTTO UN TITOLO
    SBAGLIATO, ed e' l'errore di lettura piu' pericoloso di tutta l'uscita.
    """
    start, end = layout.window(w)
    d = layout.slice_window(df, w, column="as_of")
    if d.empty:
        return {"empty": True, "start": start, "end": end}
    a = pd.to_datetime(d["as_of"])
    days_win = (pd.Timestamp(end) - pd.Timestamp(start)).days
    days_got = (a.max() - a.min()).days
    return {"empty": False, "start": start, "end": end,
            "first": a.min().date(), "last": a.max().date(),
            "quarters": d["target_quarter"].nunique(),
            "frac": days_got / days_win if days_win else 0.0}


def _coverage_line(c: dict) -> str:
    """Una riga che dice sempre la verita' sul campione, e urla se e' parziale."""
    if c["empty"]:
        return "   NO DATA in this window"
    line = (f"   window {c['start']} .. {c['end']}\n"
            f"   data   {c['first']} .. {c['last']}   "
            f"({c['quarters']} target quarters, {100 * c['frac']:.0f}% of the window)")
    if c["frac"] < 0.95:
        line += ("\n   *** PARTIAL WINDOW — the numbers below are computed ONLY on "
                 "the dates above,\n       not on the whole window named in the "
                 "heading. Do not read them as the full period. ***")
    return line


def family_report(m: pd.DataFrame, mp: pd.DataFrame, family: str,
                  src: pd.DataFrame | None = None) -> str:
    """Il report leggibile: una sezione per finestra, piu' il taglio per fase."""
    blocks = [_HEADER, f"family: {family}"]
    if m.empty:
        return "\n".join(blocks + ["\n(no window covered by the data)"])

    for w in m["window"].unique():
        d = m[m["window"] == w].drop(columns="window")
        blocks.append(cm._section(f"{w}"))
        if src is not None:
            blocks.append(_coverage_line(coverage(src, w)))
        blocks.append(cm._fmt(d))

    if not mp.empty:
        blocks.append(cm._section(
            "BY PHASE OF THE QUARTER  (forecast / nowcast M1-M2-M3 / backcast)\n"
            "the table version of the RMSE-by-horizon figure"))
        for w in mp["window"].unique():
            blocks.append(f"\n--- {w} ---")
            blocks.append(cm._fmt(mp[mp["window"] == w].drop(columns="window")))
    return "\n".join(blocks)


# ─── La matrice del confronto ─────────────────────────────────────────────────

def common_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Le righe sulle coppie (trimestre, settimana) punteggiate da TUTTI i metodi.

    E' il vincolo che rende la matrice onesta: senza, si confronterebbe un RMSE
    su quaranta trimestri con uno su dodici.
    """
    if df.empty:
        return df
    key = ["target_quarter", "horizon_week"]
    n_methods = df["metodo"].nunique()
    cover = df.groupby(key)["metodo"].nunique()
    keep = cover[cover == n_methods].index
    return df.set_index(key).loc[keep].reset_index() if len(keep) else df.iloc[0:0]


def _both_families(dfm: pd.DataFrame, bvar: pd.DataFrame) -> pd.DataFrame:
    """
    Le due famiglie in un frame solo, con i benchmark contati una volta.

    `ar2` e `mean` sono girati da entrambe le passate: senza la deduplica la
    stessa serie entrerebbe due volte e — peggio del doppio peso — un solo
    scarto di formato su `as_of` fra le due famiglie lascerebbe passare
    entrambe le copie in silenzio.  Vedi `test_windows.check_alignment`, che
    quel formato lo sorveglia a monte.
    """
    frames = [d for d in (dfm, bvar) if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    both = pd.concat(frames, ignore_index=True)
    return both.drop_duplicates(subset=["metodo", "target_quarter",
                                        "horizon_week", "as_of"])


def _coverage(d: pd.DataFrame) -> pd.Series:
    """Quota dei punti (trimestre, settimana) della fase coperta da ogni metodo."""
    tot = d.groupby(["target_quarter", "horizon_week"]).ngroups
    if not tot:
        return pd.Series(dtype=float)
    return d.groupby("metodo").apply(
        lambda g: g.groupby(["target_quarter", "horizon_week"]).ngroups / tot,
        include_groups=False)


def _drop_thin_methods(d: pd.DataFrame, min_coverage: float
                       ) -> tuple[pd.DataFrame, list[tuple[str, float]]]:
    """
    Toglie dal confronto i metodi ANOMALI, non quelli semplicemente piu' corti.

    L'intersezione e' una congiunzione: un metodo che in backcast copre un
    trimestre su quaranta lo impone come campione a chiunque altro, e la fase
    smette di dire qualcosa.  Sotto `_MIN_PHASE_COVERAGE` il metodo esce da
    QUELLA fase — non dalla tabella, non dalle altre fasi — e il suo nome
    finisce nella nota, perche' un'esclusione taciuta e' peggio del problema
    che risolve.

    La soglia e' bassa apposta: un modello che ha girato su meno anni non e'
    un'anomalia, e per lui la risposta giusta e' restringere il campione a
    tutti, che e' precisamente cio' che il vincolo fa.  Vedi la nota su
    `_MIN_PHASE_COVERAGE`.
    """
    if d.empty:
        return d, []
    cov = _coverage(d)
    if cov.empty:
        return d, []
    thin = [(m, float(f)) for m, f in cov.items() if f < min_coverage]
    if not thin:
        return d, []
    fuori = {m for m, _ in thin}
    return d[~d["metodo"].isin(fuori)], sorted(thin, key=lambda x: x[1])


def _campione_note(d0: pd.DataFrame, tenuti: list[str], scartati: list[str],
                   non_votanti: list[str]) -> str:
    """
    La riga che DICHIARA il campione: quanti trimestri, quali, e chi lo stringe.

    DUE SPECIE DI SCARTO, che non vanno confuse.

      DI BORDO   il trimestre sta prima del primo tenuto o dopo l'ultimo.  Ha
                 l'asse tagliato perche' la finestra comincia o finisce li': lo
                 scarta il CALENDARIO, ed e' il comportamento normale.

      INTERNO    il trimestre sta in mezzo ai tenuti e viene scartato lo
                 stesso.  Allora l'asse non ce l'ha per un BUCO NEI DATI, e il
                 buco ha un nome: il metodo cui mancano le settimane.

    La distinzione conta perche' i due casi si leggono in modo opposto.  Uno e'
    la regola che funziona; l'altro e' un pezzo di passata che non ha girato, e
    finche' non gira le tabelle riportano numeri piu' bassi senza che si veda
    perche'.  Misurato il 2026-09-01: sul 2007-2025 il campione congiunto e' 63
    invece di 66 perche' `lbvar/-` non ha 2020Q3 — il rimbalzo Covid, +34.9 —
    e la sua assenza abbassa l'RMSE di tutti i metodi del ~20%.  Un numero cosi'
    non puo' comparire senza la sua ragione accanto.
    """
    if not tenuti:
        return "campione vuoto"
    testa = f"campione {len(tenuti)} trimestri ({tenuti[0]}..{tenuti[-1]})"
    if not scartati:
        return testa
    interni = [q for q in scartati if tenuti[0] < q < tenuti[-1]]
    asse = cm.standard_axis(d0)
    escluso = set(non_votanti)

    def _chi_manca(q: str) -> tuple[str, int] | None:
        g = d0[d0["target_quarter"] == q]
        mancanti = {m: len(asse - set(sub["horizon_week"]))
                    for m, sub in g.groupby("metodo")
                    if m not in escluso and not asse <= set(sub["horizon_week"])}
        return max(mancanti.items(), key=lambda kv: kv[1]) if mancanti else None

    # UNA SETTIMANA SOLA E' IL CALENDARIO, NON UN BUCO.  `horizon_week` si conta
    # dall'inizio del trimestre target, e un trimestre entra in volo all'inizio
    # di quello precedente: se quello aveva quattordici venerdi' invece di
    # tredici, il suo asse parte da -13 e non da -12.  Capita a una manciata di
    # trimestri interni e sani — 2011Q1, 2011Q2, 2017Q1, 2017Q2 sul 2007-2019 —
    # ed e' gia' documentato in `compute_metrics.standard_axis`.  Segnalarli
    # come anomalia annegherebbe quelli veri: sul 2007-2025 sono otto contro
    # tre, e i tre sono l'unica cosa da guardare.
    buchi = [(q, _chi_manca(q)) for q in interni]
    veri = [(q, c) for q, c in buchi if c and c[1] > 1]
    calendario = len(buchi) - len(veri)
    testa += (f", {len(scartati)} fuori dall'asse standard"
              f" ({len(scartati) - len(interni)} di bordo"
              + (f", {calendario} per un venerdi' di calendario" if calendario else "")
              + ")")
    if not veri:
        return testa
    dettaglio = "; ".join(
        f"{q} (a {m} mancano {n} settimane)" for q, (m, n) in veri)
    return (testa + "\n" + " " * 6
            + f"ATTENZIONE — {len(veri)} trimestri INTERNI mancano per un buco "
              f"nei dati, non per il calendario:\n"
            + " " * 8 + dettaglio)


def _non_votanti(d: pd.DataFrame) -> list[str]:
    """
    I metodi che sul campione NON votano: l'asse standard non ce l'hanno.

    E' la generalizzazione della regola gia' scritta per la NY Fed.  Il criterio
    e' "non ha l'asse MAI, su nessun trimestre della finestra": chi l'asse non
    lo produce per costruzione non l'ha perso, e chiederglielo svuoterebbe il
    campione di tutti.  Chi invece l'asse ce l'ha altrove e lo perde su un
    trimestre solo ha un BUCO — e quello deve stringere il campione, perche' e'
    esattamente l'informazione che manca a tutti.

    L'UNICO CLIENTE REALE E' LA NY FED, ed e' misurato (2026-09-01): lei copre
    28 settimane, da -5 a +22; noi 31, da -13 a +17.  Le otto settimane -13..-6
    non le ha PERSE — non le pubblica affatto, il suo forecast non parte cosi'
    presto.  (Dall'altro lato lei arriva a +22 e noi ci fermiamo a +17, ma li'
    e' `last_before_release` a scartare i suoi backcast pubblicati dopo
    l'advance del BEA: vedi l'intestazione di `compare_nyfed`.)

    GLI ALTRI DUE CASI SONO RETI, NON DATI.  Nella passata vera tutti e 21 i
    metodi coprono -13..+17, backcast compreso: nessuno e' strutturalmente
    corto, nessuno e' rado.  Le due condizioni restano perche' sono i due modi
    noti in cui questa tabella si rompe, e `test_common_sample` li costruisce
    apposta — un `qbvar` finto che si ferma alla settimana 13, un `lbvar` finto
    presente su un trimestre solo.  Sono finzioni del test, non osservazioni.

    Il caso vero e' il terzo, ed e' l'opposto: `lbvar/-` non copre 2020Q3
    perche' quel blocco non ha girato, ma copre l'asse ovunque altro.  Resta
    votante, e giustamente toglie 2020Q3 a tutti — la congiunta non puo'
    punteggiare un trimestre che uno dei metodi non ha.
    """
    fuori = {_NYFED}
    asse = cm.standard_axis(d)
    if asse:
        per = d.groupby("metodo")["horizon_week"].apply(set)
        fuori |= {m for m, weeks in per.items() if not asse <= weeks}

    # E NEANCHE I METODI TROPPO RADI, per la stessa ragione per cui esiste
    # `_drop_thin_methods`: un metodo presente su un trimestre su dodici
    # imporrebbe quel trimestre a tutti gli altri.  La differenza e' il momento:
    # `_drop_thin_methods` agisce DENTRO la fase, quando il campione dei
    # trimestri e' gia' stato deciso; qui si decide quel campione, e senza
    # questa riga un metodo rado lo deciderebbe da solo.  Stessa soglia, cosi'
    # le due guardie non possono dire cose diverse sullo stesso metodo.
    n_q = d["target_quarter"].nunique()
    if n_q:
        quota = d.groupby("metodo")["target_quarter"].nunique() / n_q
        fuori |= set(quota.index[quota < _MIN_PHASE_COVERAGE])
    return sorted(fuori)


def _binding_method(d: pd.DataFrame) -> tuple[str, int, int] | None:
    """
    Chi STRINGE il campione comune: il metodo la cui esclusione lo allargherebbe
    di piu'.

    Non e' un'accusa e non porta a nessuna esclusione automatica: e' il dato che
    manca per leggere una riga "n = 48" senza chiedersi da dove venga il 48.  Se
    il campione comune e' piccolo, il lettore ha diritto di sapere per colpa di
    chi, e decidere lui se quella colonna gli serve.

    Returns
    -------
    (metodo, punti_ora, punti_senza_di_lui) oppure None se non stringe nessuno.
    """
    metodi = list(d["metodo"].unique())
    if len(metodi) < 2:
        return None
    ora = common_points(d)
    n_ora = ora.groupby(["target_quarter", "horizon_week"]).ngroups if not ora.empty else 0
    best, n_best = None, n_ora
    for m in metodi:
        senza = d[d["metodo"] != m]
        c = common_points(senza)
        n = c.groupby(["target_quarter", "horizon_week"]).ngroups if not c.empty else 0
        if n > n_best:
            best, n_best = m, n
    return (best, n_ora, n_best) if best else None


def comparison_by_phase(dfm: pd.DataFrame, bvar: pd.DataFrame,
                        windows: list[str] = TABLE_WINDOWS,
                        min_coverage: float = _MIN_PHASE_COVERAGE
                        ) -> tuple[pd.DataFrame, str]:
    """
    Il confronto fra famiglie FASE PER FASE, ciascuna sul proprio campione
    comune.  E' l'oggetto su cui si legge il backcast.

    Perche' separato da `comparison_matrices`: quella aggrega su tutte le fasi
    e produce un numero solo per metodo.  Il backcast — trimestre chiuso, PIL
    non ancora uscito — e' la fase in cui i modelli si distinguono di piu',
    perche' e' li' che tutta l'informazione mensile e' entrata e resta solo la
    capacita' di comporla.  Dentro un aggregato pesato dalle 13 settimane di
    forecast quella differenza si annacqua.

    L'intersezione e' calcolata DENTRO la fase (vedi `common_points_by_phase`),
    e i metodi troppo radi in quella fase ne escono (`_drop_thin_methods`)
    invece di restringerla a tutti.

    Returns
    -------
    (long, note)
        `long` ha una riga per (finestra, fase, metodo) con le metriche sul
        campione comune di quella fase; `note` dice, per ogni coppia, quanti
        punti sono sopravvissuti, su quali trimestri, e chi e' stato escluso.
    """
    both = _both_families(dfm, bvar)
    if both.empty:
        return pd.DataFrame(), "  (nessun dato)"

    # UNA REGOLA SOLA PER FIGURE, TABELLE PER FAMIGLIA E TABELLE CONGIUNTE.
    #
    # Qui si tagliava su `as_of`, e la ragione scritta era che questa tabella
    # mette insieme famiglie con assi diversi PER COSTRUZIONE — la NY Fed prima
    # di tutte, che comincia a -3/-4 perche' le settimane profonde non le
    # pubblica, non le ha perse.  La ragione era vera ma provava troppo:
    # giustifica di non PRETENDERE l'asse intero da ogni metodo, non di far
    # entrare TRIMESTRI FUORI FINESTRA.  Sono due cose diverse, e il taglio su
    # `as_of` le confondeva in una sola.
    #
    # `window_sample` le tiene separate gia' di suo: `skip` esclude dal VOTO
    # sul campione i metodi che l'asse non ce l'hanno per costruzione, e li
    # lascia nelle tabelle.  Tutto il resto non cambia — `_drop_thin_methods` e
    # `common_points` continuano a governare l'asimmetria delle coperture
    # esattamente come prima.  Vedi la guardia in `test_common_sample`, che
    # costruisce apposta un q-BVAR senza backcast.
    #
    # COSA COSTAVA, misurato il 2026-09-01.  Sul 2007-2019 il taglio su `as_of`
    # faceva entrare 2020Q1 — realizzato -5.2, 195 righe di sola previsione
    # agli orizzonti -12..0 — e la finestra contava 54 trimestri invece di 46.
    # L'RMSE ne usciva gonfiato dal +2.9% (diag3/gaussian_ar1) al +5.6%
    # (bbvar), e NON in modo uniforme: il margine fra le due famiglie si
    # spostava dell'8% in una tabella che esiste apposta per confrontarle.
    #
    # VERIFICATO CHE QUI LA REGOLA NON SVUOTA NIENTE, che era il timore del
    # commento vecchio: sul frame congiunto l'asse standard esce -12..+17 per
    # ENTRAMBE le famiglie, e i trimestri tenuti sul 2007-2019 sono 46 — lo
    # stesso numero delle tabelle per famiglia e della figura per orizzonte.
    # Sul 2007-2025 sono 63 contro i 66 del solo DFM, e i tre di scarto sono il
    # blocco L-BVAR del Covid che manca: e' la regola che fa il suo mestiere
    # ("il campione e' quello del metodo peggio coperto"), non un difetto.
    # Finche' quel blocco manca, congiunte e per-famiglia riportano numeri
    # diversi sulla stessa finestra, e va detto invece che scoperto.
    rows, notes = [], []
    for w in windows:
        d0 = layout.slice_window(both, w, column="as_of")
        non_votanti = _non_votanti(d0)
        d, tenuti, scartati = cm.window_sample(both, w, skip=non_votanti)
        # La Fed esce PRIMA del calcolo, non dopo.  Toglierla a valle
        # lascerebbe gli altri metodi misurati sul campione che LEI stringeva
        # — sulle finestre di zoom la sua riga sparisce ma le settimane -12..-5
        # resterebbero tagliate lo stesso, e la tabella direbbe "forecast" su
        # cinque settimane invece di tredici senza che si veda perche'.
        if w not in layout.NYFED_COMPARISON_PASSES:
            d = d[d["metodo"] != _NYFED]
        if d.empty:
            continue
        notes.append(f"\n  --- {w} ---  "
                     + _campione_note(d0, tenuti, scartati, non_votanti))
        for ph in cm._PHASE_ORDER:
            p = d[d["fase"] == ph]
            if p.empty:
                continue
            n_prima = p["metodo"].nunique()
            p, thin = _drop_thin_methods(p, min_coverage)
            c = common_points(p) if not p.empty else p
            if c.empty:
                notes.append(f"    {ph:11s} nessun punto comune fra i "
                             f"{n_prima} metodi — saltata")
                continue

            t = cm.table_by_method(c)
            t.insert(0, "fase", ph)
            t.insert(0, "window", w)
            rows.append(t)

            n_pts = c.groupby(["target_quarter", "horizon_week"]).ngroups
            qs = sorted(c["target_quarter"].unique())
            line = (f"    {ph:11s} {n_pts:5d} punti comuni | "
                    f"{len(qs)} trimestri ({qs[0]}..{qs[-1]}) | "
                    f"settimane w{int(c['horizon_week'].min()):+d}..w"
                    f"{int(c['horizon_week'].max()):+d} | "
                    f"{c['metodo'].nunique()} metodi")
            if thin:
                line += ("\n" + " " * 16 + "esclusi da questa fase (copertura < "
                         f"{100 * min_coverage:.0f}%): "
                         + ", ".join(f"{m} {100 * f:.0f}%" for m, f in thin))
            b = _binding_method(p)
            if b:
                line += ("\n" + " " * 16 + f"campione stretto da {b[0]}: "
                         f"senza di lui sarebbero {b[2]} punti invece di {b[1]}")
            notes.append(line)

    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return long, "\n".join(notes)


def backcast_matrix(long: pd.DataFrame) -> pd.DataFrame:
    """
    Il pannello dedicato: metodi x finestre, sul solo backcast e sul campione
    comune di quella fase.  E' la tabella su cui poggia la conclusione, quindi
    esce come file a se' e non come riga dentro un'altra.
    """
    if long.empty or "fase" not in long.columns:
        return pd.DataFrame()
    bc = long[long["fase"] == "backcast"]
    if bc.empty:
        return pd.DataFrame()
    return (bc.pivot(index="metodo", columns="window", values="RMSE")
            .reindex(cm._ordered_methods(bc["metodo"].unique())))


def comparison_matrices(dfm: pd.DataFrame, bvar: pd.DataFrame,
                        windows: list[str] = TABLE_WINDOWS
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Tre matrici metodi x finestre (RMSE, RMSE_rel_ar2, MDA) piu' una nota su
    quanti punti comuni sono rimasti in ciascuna finestra.
    """
    both = _both_families(dfm, bvar)

    # Stessa regola di `comparison_by_phase`, e per la stessa ragione: il
    # campione lo decide `window_sample` sui trimestri che coprono l'asse
    # standard, con la NY Fed che resta nelle tabelle ma non vota (`skip`).
    # Il campione comune punto per punto (`common_points`) viene DOPO e resta
    # identico: e' l'altra meta' del lavoro, non un'alternativa.
    rows, notes = [], []
    for w in windows:
        d0 = layout.slice_window(both, w, column="as_of")
        non_votanti = _non_votanti(d0)
        d, tenuti, scartati = cm.window_sample(both, w, skip=non_votanti)
        if d.empty:
            continue
        c = common_points(d)
        if c.empty:
            notes.append(f"  {w}: no point is scored by every method — skipped")
            continue
        t = cm.table_by_method(c)
        t.insert(0, "window", w)
        rows.append(t)
        notes.append(
            f"  {w}: " + _campione_note(d0, tenuti, scartati, non_votanti)
            + f"; {c.groupby(['target_quarter', 'horizon_week']).ngroups} punti "
              f"comuni fra {c['metodo'].nunique()} metodi")

        # Che cosa c'e' DAVVERO dietro la colonna.  Senza questa riga la
        # matrice mostra otto punti di novembre 2008 sotto un'intestazione
        # "2007-2025" — numeri esatti, lettura completamente falsa.
        cov = coverage(c, w)
        a = pd.to_datetime(c["as_of"])
        n_pts = c.groupby(["target_quarter", "horizon_week"]).ngroups
        line = (f"  {w}: {n_pts} common (quarter, week) points | "
                f"{c['target_quarter'].nunique()} quarters "
                f"({', '.join(sorted(c['target_quarter'].unique())[:6])}"
                f"{' ...' if c['target_quarter'].nunique() > 6 else ''}) | "
                f"as_of {a.min().date()}..{a.max().date()} "
                f"({100 * cov['frac']:.0f}% of window) | "
                f"horizons w{int(c['horizon_week'].min()):+d}..w"
                f"{int(c['horizon_week'].max()):+d}")
        if cov["frac"] < 0.95 or n_pts < _MIN_COMPARISON_POINTS:
            line += "\n      *** NOT REPRESENTATIVE OF THE WINDOW — see banner ***"
        notes.append(line)

    if not rows:
        empty = pd.DataFrame()
        return empty, empty, empty, "\n".join(notes)

    long = pd.concat(rows, ignore_index=True)
    piv = lambda col: (long.pivot(index="metodo", columns="window", values=col)
                       .reindex(cm._ordered_methods(long["metodo"].unique()))
                       .reindex(columns=[w for w in windows
                                         if w in long["window"].unique()]))
    return piv("RMSE"), piv("RMSE_rel_ar2"), piv("MDA"), "\n".join(notes)


def extremes_table(dfm: pd.DataFrame, bvar: pd.DataFrame,
                   nyfed: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Nei trimestri piu' estremi, quanto del crollo ciascun metodo ha catturato.

    E' la tabella che risponde meglio di ogni RMSE alla domanda "chi vince
    nella crisi", perche' l'RMSE mescola l'errore sui trimestri normali con
    quello sugli estremi, e sono gli estremi che decidono se un modello di
    coda serve a qualcosa.  `quota_catturata = nowcast/realizzato`: 1.0 = ci
    arriva, 0.3 = ne prende un terzo, negativo = ha sbagliato segno.

    Valutata all'orizzonte PIU' PROFONDO disponibile per ogni trimestre, cioe'
    quando il modello ha visto il massimo dell'informazione: e' la sua
    prestazione migliore possibile, non una media penalizzante.

    NON e' ristretta al campione comune: qui non si confrontano medie fra
    metodi su campioni diversi, si guarda trimestre per trimestre, e ogni
    riga porta il proprio `settimana`.  Chi non ha girato su quel trimestre
    semplicemente non compare.
    """
    frames = [d for d in (dfm, bvar, nyfed) if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    t = cm.table_extremes(pd.concat(frames, ignore_index=True))
    return t if t is not None else pd.DataFrame()


def _phase_block(long: pd.DataFrame, note: str) -> list[str]:
    """La sezione per fase, col backcast staccato in fondo perche' e' il punto."""
    if long.empty:
        return [cm._section("BY PHASE — no phase has a common sample"), note]

    out = [cm._section(
        "COMPARISON BY PHASE — each phase on ITS OWN common sample\n"
        "The aggregate matrix above mixes the 13 forecast weeks with the 4\n"
        "backcast ones and hides where the models actually differ.  Here the\n"
        "intersection is taken INSIDE each phase, so a model with no border\n"
        "(no week > 13) loses its own backcast instead of deleting everyone's."),
        note]

    for w in long["window"].unique():
        d = long[long["window"] == w]
        piv = (d.pivot(index="metodo", columns="fase", values="RMSE")
               .reindex(cm._ordered_methods(d["metodo"].unique()))
               .reindex(columns=[p for p in cm._PHASE_ORDER
                                 if p in d["fase"].unique()]))
        out += [f"\n--- {w} — RMSE per fase, campione comune di ciascuna fase ---",
                piv.round(3).to_string()]

    bc = long[long["fase"] == "backcast"]
    if not bc.empty:
        out += [cm._section(
            "BACKCAST — the number the thesis rests on\n"
            "quarter closed, GDP not yet published: every monthly release is in,\n"
            "and what is left is the ability to compose it.  Same targets for\n"
            "every method in the panel, by construction."),
            cm._fmt(bc[[c for c in ("window", "metodo", "n", "RMSE",
                                    "RMSE_rel_ar2", "MAE", "Bias", "MDA")
                        if c in bc.columns]])]
    return out


def comparison_report(rmse: pd.DataFrame, rel: pd.DataFrame, mda: pd.DataFrame,
                      note: str, ext: pd.DataFrame | None = None,
                      phase_long: pd.DataFrame | None = None,
                      phase_note: str = "") -> str:
    """Le tre matrici piu' il riassunto dei vincitori: e' cio' che si legge per primo."""
    blocks = [_HEADER,
              cm._section("COMMON-SAMPLE CONSTRAINT\n"
                          "every method is scored on the SAME (quarter, week) "
                          "points; otherwise the columns would not compare"),
              note]
    if rmse.empty:
        return "\n".join(blocks + ["\n(no window has a common sample)"])

    if "NOT REPRESENTATIVE" in note:
        blocks.append(
            "\n" + "!" * 92 + "\n"
            "!!  WARNING — THE MATRIX BELOW IS NOT YET A RESULT.\n"
            "!!\n"
            "!!  One or more columns rest on a handful of (quarter, week) points, because\n"
            "!!  the common sample is the INTERSECTION of what every method has actually\n"
            "!!  been estimated on — and some models have been run on a short block only.\n"
            "!!\n"
            "!!  Those columns therefore measure a few specific weeks, NOT the window in\n"
            "!!  their heading, and if those weeks fall in a crisis the RMSE is large for\n"
            "!!  everyone. It is arithmetically correct and substantively meaningless.\n"
            "!!\n"
            "!!  Read the per-window sample above before reading any number below.\n"
            + "!" * 92)

    blocks += [cm._section("RMSE  (BEA percentage points) — compare DOWN a column"),
               rmse.round(3).to_string(),
               cm._section("RMSE RELATIVE TO AR(2) — the cross-period read; <1 beats it"),
               rel.round(3).to_string(),
               cm._section("MDA — directional accuracy; 0.50 = coin toss"),
               mda.round(3).to_string()]

    win = ["", "WINNERS BY WINDOW", "-" * 60]
    models = [m for m in rmse.index if m not in layout.BENCHMARKS]
    for w in rmse.columns:
        r = rmse.loc[models, w].dropna()
        d = mda.loc[models, w].dropna()
        if r.empty:
            continue
        # Un "vincitore" su una manciata di settimane e' la riga piu' citabile
        # e la piu' sbagliata dell'intero file: si sopprime, non si annota.
        if f"  {w}:" in note and "NOT REPRESENTATIVE" in note.split(f"  {w}:")[1][:400]:
            win.append(f"  {w:12s}  (no winner declared — sample not "
                       f"representative of the window)")
            continue
        win.append(f"  {w:12s}  best RMSE: {r.idxmin():28s} ({r.min():.3f})"
                   + (f"   best MDA: {d.idxmax():28s} ({d.max():.3f})"
                      if not d.empty else ""))

    if phase_long is not None:
        win += _phase_block(phase_long, phase_note)

    if ext is not None and not ext.empty:
        win += [cm._section(
            "EXTREME QUARTERS — how much of the swing each method captured\n"
            "at the DEEPEST horizon available (most information seen).\n"
            "quota_catturata = nowcast / realised: 1.0 = gets there, "
            "0.3 = a third, negative = wrong sign"), cm._fmt(ext)]
    return "\n".join(blocks + win)


# ─── Scrittura ────────────────────────────────────────────────────────────────

def write_all(dfm: pd.DataFrame, bvar: pd.DataFrame,
              windows: list[str] = TABLE_WINDOWS,
              nyfed: pd.DataFrame | None = None) -> list[str]:
    written: list[str] = []
    # La Fed entra come riga nelle tabelle del DFM: e' il confronto esterno del
    # lavoro a fattori, non dei BVAR.
    dfm_fed = (pd.concat([dfm, nyfed], ignore_index=True)
               if nyfed is not None and not nyfed.empty else dfm)

    for spec in layout.SPECS:
        if dfm.empty or not any(m.startswith(spec + "/") for m in dfm["metodo"]):
            continue
        m, mp = family_tables(dfm_fed, spec, windows)
        d = layout.dfm_rmse_dir(spec)
        os.makedirs(d, exist_ok=True)
        written += _write(m, mp, family_report(m, mp, spec, dfm), d,
                          f"metrics_{spec}")

    if not bvar.empty:
        m, mp = family_tables(bvar, "bvar", windows)
        d = layout.bvar_rmse_dir()
        os.makedirs(d, exist_ok=True)
        written += _write(m, mp, family_report(m, mp, "bvar", bvar), d,
                          "metrics_bvar")

    rmse, rel, mda, note = comparison_matrices(dfm, bvar, windows)
    # Il confronto per fase: la Fed entra ANCHE qui, perche' la domanda
    # "chi fa il backcast migliore" la riguarda quanto i BVAR.  Ci pensa
    # `_drop_fed_outside` a non farla comparire dove non pubblica.
    fed_ok = (nyfed if nyfed is not None and not nyfed.empty else pd.DataFrame())
    phase_long, phase_note = comparison_by_phase(
        pd.concat([d_ for d_ in (dfm, fed_ok) if not d_.empty],
                  ignore_index=True) if not dfm.empty else dfm,
        bvar, windows)
    bc = backcast_matrix(phase_long)
    ext = extremes_table(dfm, bvar, nyfed)
    d = layout.comparison_dir()
    os.makedirs(d, exist_ok=True)
    report = comparison_report(rmse, rel, mda, note, ext,
                               phase_long=phase_long, phase_note=phase_note)
    if not ext.empty:
        p = os.path.join(d, "extremes.csv")
        ext.to_csv(p, index=False)
        written.append(p)
    if not phase_long.empty:
        p = os.path.join(d, "rmse_by_phase.csv")
        phase_long.to_csv(p, index=False)
        written.append(p)
    if not bc.empty:
        p = os.path.join(d, "backcast_matrix.csv")
        bc.to_csv(p)
        written.append(p)
    for name, tab in (("rmse_matrix", rmse), ("rmse_rel_ar2_matrix", rel),
                      ("mda_matrix", mda)):
        if not tab.empty:
            p = os.path.join(d, f"{name}.csv")
            tab.to_csv(p)
            written.append(p)
    p = os.path.join(d, "summary.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    written.append(p)
    print(report)
    return written


def _write(m: pd.DataFrame, mp: pd.DataFrame, report: str,
           d: str, stem: str) -> list[str]:
    out = []
    if not m.empty:
        p = os.path.join(d, f"{stem}.csv")
        m.to_csv(p, index=False)
        out.append(p)
    if not mp.empty:
        p = os.path.join(d, f"{stem}_by_phase.csv")
        mp.to_csv(p, index=False)
        out.append(p)
    p = os.path.join(d, f"{stem}.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    out.append(p)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Tabelle di accuratezza per finestra.")
    p.add_argument("--csv", nargs="*", default=None, help="CSV DFM")
    p.add_argument("--bvar-csv", nargs="*", default=None, help="CSV BVAR")
    p.add_argument("--window", nargs="*", default=None,
                   help=f"sottoinsieme di {TABLE_WINDOWS}")
    p.add_argument("--no-nyfed", action="store_true",
                   help="non aggiungere la riga del NY Fed Staff Nowcast")
    a = p.parse_args()

    dfm = load_dfm(a.csv)
    bvar = load_bvar(a.bvar_csv)
    fed = pd.DataFrame() if a.no_nyfed else load_nyfed(dfm)
    print(f"DFM  : {len(dfm)} righe, {dfm['metodo'].nunique()} metodi")
    print(f"BVAR : {len(bvar)} righe, "
          f"{bvar['metodo'].nunique() if not bvar.empty else 0} metodi")
    print(f"NYFED: {len(fed)} righe punteggiabili"
          + ("" if fed.empty else
             f", {fed['target_quarter'].nunique()} trimestri"))

    written = write_all(dfm, bvar, a.window or TABLE_WINDOWS, nyfed=fed)
    print(f"\n{len(written)} file scritti:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
