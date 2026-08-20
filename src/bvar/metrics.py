"""
src/bvar/metrics.py

LE METRICHE DEL GATE 6 — RMSE, NY Fed, log score.  Involucro, non calcolatore.

    python -m src.bvar.metrics

Tre uscite, due delle quali sono `src/forecast/` chiamato con altri argomenti.
I percorsi li decide `src/output_layout.py`, non questo modulo:

    forecast_weekly/bvar/rmse/       RMSE dei quattro modelli, dell'AR(2),
                                     della media espandente e del NY Fed
    forecast_weekly/bvar/logscore/   il log predictive score, l'unica metrica
                                     in cui un BVAR bayesiano puo' battere un
                                     point forecast

LA FIGURA DEL LOG SCORE E' UNA SOLA, SUL PERIODO COMPLETO
---------------------------------------------------------
Deciso: un solo `logscore_by_horizon.png` sul 2007-2025, non uno per finestra
come per l'RMSE.  `--window` esiste e taglia davvero, ma non va usato nello
script: la figura confronta i QUATTRO MODELLI FRA LORO — niente AR(2), niente
media espandente, perche' un point forecast non ha densita' predittiva — e con
un solo pannello quel confronto si legge in un colpo.

Il campione non e' pero' "tutti i trimestri": `figure_logscore_by_horizon`
scarta quelli MONCHI ai bordi del blocco (il primo si vede solo in backcast,
l'ultimo solo in forecast) perche' deformerebbero le due estremita' della
curva.  Su una passata continua sono uno o due su ~76, e il titolo li dichiara.

CHE COSA E' RIUSATO, ALLA LETTERA
----------------------------------
    compute_metrics.load_long / table_by_method_phase / table_by_method /
    table_by_quarter / table_extremes / _fmt / _section
        le tabelle RMSE-MAE-Bias-MDA-SignAcc, incluse `RMSE_rel_ar2` e
        `RMSE_rel_mean`.  Girano sul CSV del BVAR senza una riga di differenza
        perche' il contratto e' lo stesso.

    compare_nyfed.build_panel / build_report / horizon_panel /
    figure_rmse_by_horizon
        il confronto col NY Fed e LA FIGURA per orizzonte con le tre fasi come
        bande di sfondo — la Figura 1 del paper.  Anche questa riusata, non
        riscritta.

Nessun file di `src/forecast/` e' modificato.  Cio' che e' nuovo qui e' solo il
log score, che nel lavoro DFM non esisteva perche' li' il nowcast era di punto.

IL CONFRONTO COL NY FED SI FA MODELLO PER MODELLO, E NON E' UN DETTAGLIO
------------------------------------------------------------------------
`compare_nyfed.aligned_sample` tiene solo i trimestri in cui TUTTI i metodi
hanno una stima finale: e' la regola che impedisce di confrontare il mio RMSE
su dieci trimestri col loro su dodici.  Ma con quattro modelli insieme diventa
una congiunzione fragile — il Q-BVAR non ha bordo (`qbvar_growth` restituisce
`None` quando il trimestre obiettivo esce dalla finestra), e un suo buco
cancellerebbe quel trimestre anche per gli altri tre.  Si gira quindi una volta
per modello, cosi' ogni modello e' allineato al NY Fed sul SUO campione
massimale, piu' una passata congiunta per la tabella d'insieme.  I campioni
possono differire, e il report lo dichiara: e' un dato, non un fastidio.

IL NY FED C'E' ANCHE SUL BLOCCO 2007-2010
------------------------------------------
La loro serie parte dal 2002Q1 — 94 trimestri — perche' 2002Q1-2015Q4 sono una
RICOSTRUZIONE RETROSPETTIVA, come la loro stessa nota dichiara ("historical
reconstructions based on real-time data").  La cartella `rmse/` si popola
quindi gia' col primo blocco.  Va letta sapendo che quei numeri non sono un
track record out-of-sample: la specificazione e' stata fissata dopo aver visto
il periodo (l'argomento completo e' nell'header di `compare_nyfed.py`).

Se il NY Fed mancasse — un blocco fuori dalla loro copertura, o il file
assente — le tabelle interne (i quattro modelli contro l'AR(2)) escono lo
stesso: il confronto con loro e' un blocco a parte che si salta con un
messaggio, non un prerequisito.

IL LOG SCORE: PERCHE' LA FIGURA E' NUOVA E LE BANDE NO
-------------------------------------------------------
`figure_rmse_by_horizon` fissa `ylim(0, max*1.18)` e l'etichetta "RMSE (punti
BEA)": il log score e' NEGATIVO, quindi riusarla darebbe un riquadro vuoto.
La figura del log score e' percio' scritta qui — ma prende da `compare_nyfed`
le bande delle fasi, i colori, il font e lo stile degli assi, cosi' le due
figure restano la stessa figura con un'altra ordinata.  E' l'adattamento
previsto: sta nell'involucro, non nell'originale.

PIU' ALTO E' MEGLIO, ed e' l'opposto dell'RMSE: chi legge le due figure in fila
deve saperlo, e infatti il titolo lo dice.
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import output_layout as layout
from src.bvar.evaluate import OUTPUT_ROOT
from src.bvar.figures import MODELS, discover_csvs
from src.forecast import compare_nyfed as cn
from src.forecast import compute_metrics as cm
from src.forecast.nyfed_nowcast import load_long as load_nyfed

#: Le fasi, con la stessa soglia di `compute_metrics._phase`.
_PHASE_ORDER = cm._PHASE_ORDER


# ─── 1. RMSE interno: i quattro modelli contro l'AR(2) ────────────────────────

def rmse_report(paths: list[str] | None, out_dir: str, tag: str = "",
                df: pd.DataFrame | None = None, n_tot: int | None = None,
                n_drop: int = 0) -> pd.DataFrame:
    """
    Le tabelle di `compute_metrics`, scritte in `output/bvar/rmse/`.

    E' la stessa sequenza del `main()` di `compute_metrics`, che non e'
    chiamabile direttamente perche' legge `sys.argv` e scrive nella cartella
    del DFM.  Le TABELLE, che sono la sostanza, sono le sue.

    `df` gia' caricato + `tag`: servono al ciclo sulle finestre, che legge i
    CSV UNA VOLTA e poi affetta.  Il `tag` entra nei nomi dei file — senza,
    sei finestre si sovrascriverebbero a vicenda e sul disco resterebbe solo
    l'ultima, con il nome di tutte.
    """
    if df is None:
        df, n_tot, n_drop = cm.load_long(paths)
    if n_tot is None:
        n_tot = len(df)
    if df.empty:
        raise SystemExit(
            "Nessuna riga punteggiabile: `realizzato_bea` e' vuoto su tutte le "
            f"{n_tot} righe.  Il CSV e' stato prodotto da una versione di "
            "`evaluate.py` che non lo riempiva — va rifatta la passata.")

    methods = cm._ordered_methods(df["metodo"].unique())
    header = (
        f"METRICHE DEL NOWCAST BVAR — GATE 6\n"
        f"errore in punti BEA (nowcast_bea - realizzato_bea)\n"
        f"metodi ({len(methods)}): {', '.join(methods)}\n"
        f"trimestri target: {df['target_quarter'].nunique()}  "
        f"({', '.join(sorted(df['target_quarter'].unique()))})\n"
        f"settimane: da {df['horizon_week'].min():+d} a "
        f"{df['horizon_week'].max():+d}\n"
        f"righe punteggiate: {len(df)} / {n_tot}  "
        f"(scartate {n_drop} senza PIL realizzato)\n"
        f"\n"
        f"RMSE_rel_ar2 < 1  =  batte l'AR(2) a parita' di orizzonte."
    )

    t_mp = cm.table_by_method_phase(df)
    t_m = cm.table_by_method(df)
    t_q = cm.table_by_quarter(df)
    t_e = cm.table_extremes(df)

    t_bal, tenuti, esclusi = cm.table_by_method_phase_balanced(df)

    blocks = [header,
              cm._section("1. METODO x FASE DEL TRIMESTRE"), cm._fmt(t_mp)]
    # 1b — LA TABELLA CHE SI LEGGE LUNGO LA RIGA.  La 1 no: agli estremi della
    # finestra i bin contengono trimestri DIVERSI (il primo obiettivo entra solo
    # da backcast, l'ultimo solo da forecast), e su 12 trimestri di crisi lo
    # sbilanciamento vale piu' del segnale.  E' il difetto trovato sul pilota
    # 2008-2010: il backcast del Q-BVAR sembrava migliorare (3.850 -> 3.731) e
    # bilanciato era piatto (3.971 -> 3.976).  Vedi `compute_metrics`.
    if not t_bal.empty and esclusi:
        blocks += [
            cm._section("1b. LO STESSO, SU PANNELLO BILANCIATO\n"
                        "    solo i trimestri presenti in TUTTE le fasi"),
            f"   tenuti  ({len(tenuti)}): {', '.join(tenuti)}\n"
            f"   esclusi ({len(esclusi)}): {', '.join(esclusi)}"
            f"   <- troncati al bordo della finestra\n",
            cm._fmt(t_bal)]
    blocks += [cm._section("2. METODO  (aggregato su fasi e date)"), cm._fmt(t_m)]
    if t_q.shape[1] > 1:
        blocks += [cm._section("3. RMSE PER METODO x TRIMESTRE"),
                   cm._fmt(t_q.reset_index())]
    if t_e is not None:
        blocks += [cm._section("4. TRIMESTRI ESTREMI, all'orizzonte piu' profondo"),
                   cm._fmt(t_e)]

    report = "\n".join(blocks)
    print(report)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metriche{tag}.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(report + "\n")
    t_mp.to_csv(os.path.join(out_dir, f"metriche_metodo_fase{tag}.csv"),
                index=False)
    t_m.to_csv(os.path.join(out_dir, f"metriche_metodo{tag}.csv"), index=False)
    print(f"\nscritto: {out_dir}")
    return df


# ─── 2. Il confronto col NY Fed, e la figura per orizzonte ────────────────────

#: Il finto nome di spec sotto cui i quattro BVAR entrano nella figura unica.
#: `horizon_panel` filtra per spec (`metodo.split("/")[0] == spec`), perche' nel
#: DFM una figura e' una spec con le sue cinque varianti.  Per il BVAR la figura
#: e' la FAMIGLIA con i suoi quattro modelli: si rietichetta `qbvar/-` in
#: `bvar/qbvar` e il filtro diventa quello giusto senza toccare `compare_nyfed`.
#: In legenda finisce `metodo.split("/", 1)[-1]`, cioe' esattamente `qbvar`.
_FAMILY = "bvar"


def _as_family(mine: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    """I quattro modelli sotto un'unica spec; benchmark e Fed intatti."""
    d = mine.copy()
    d["metodo"] = d["metodo"].map(
        lambda m: f"{_FAMILY}/{m.split('/')[0]}"
        if m.split("/")[0] in models else m)
    return d


def horizon_figures(mine: pd.DataFrame, out_dir: str,
                    models: tuple[str, ...] = MODELS, tag: str = "") -> list[str]:
    """
    L'RMSE per orizzonte: UNA FIGURA SOLA CON I QUATTRO BVAR, piu' l'AR(2), la
    media espandente e il NY Fed.  Indipendente dalla NY Fed.

    UNA FIGURA, NON QUATTRO.  La domanda e' "quale dei quattro BVAR compone
    meglio l'informazione settimana per settimana", e con un modello per
    riquadro quella domanda richiede di confrontare a memoria quattro assi con
    scale diverse.  E' anche l'analogo esatto del DFM, dove una figura porta
    una spec con le sue cinque varianti: qui porta la famiglia con i suoi
    quattro modelli.

    PERCHE' STA FUORI DA `nyfed_report`.  Ci stava dentro, ed era un difetto:
    la figura e' una metrica MIA — l'RMSE dei miei modelli settimana per
    settimana — ma se la Fed non copriva il blocco `nyfed_report` usciva col
    suo messaggio e la figura non veniva disegnata affatto.  Su tre delle sei
    finestre non ne sarebbe uscita nessuna, e non per mancanza di dati miei.

    La Fed resta nella figura quando c'e': `cn.horizon_panel` la carica da se'
    e la aggiunge come una curva in piu'.  Quando non c'e', il pannello ha solo
    i miei metodi e la figura esce lo stesso.

    Il campione lo sceglie `horizon_panel`: si passano TUTTI i trimestri e ci
    pensa lui a tenere quelli con copertura settimanale completa — la regola
    che impedisce alla curva di scendere per composizione invece che per
    apprendimento.
    """
    written: list[str] = []

    # IL CAMPIONE LO SCEGLIE `horizon_panel`, e qui si passano tutti i
    # trimestri.  Qui c'era una copia della regola — l'unione esatta delle
    # settimane — messa a guardia di un `KeyError` su frame vuoto: due copie
    # della stessa regola che potevano divergere, e infatti quella era la
    # versione che ha ridotto a due trimestri le figure di tredici anni.  Il
    # frame vuoto ora lo gestisce `horizon_panel`, che torna (vuoto, []).
    quarters = sorted(mine["target_quarter"].unique())
    if not quarters:
        print(f"  [{tag or 'completo'}] nessun trimestre nel blocco: "
              f"niente figure per orizzonte.")
        return written

    presenti = [m for m in models
                if any(c.split("/")[0] == m for c in mine["metodo"].unique())]
    if not presenti:
        print(f"  [{tag or 'completo'}] nessuno dei modelli {models} nei dati: "
              f"niente figura per orizzonte.")
        return written

    ph, sample = cn.horizon_panel(_as_family(mine, models), quarters, _FAMILY)
    if not sample or ph.empty:
        print(f"  [{tag or 'completo'}] copertura settimanale incompleta: "
              f"niente figura per orizzonte.")
        return written

    csv = os.path.join(out_dir, f"rmse_per_orizzonte_bvar{tag}.csv")
    ph.to_csv(csv, index=False)
    # IL NOME NON CONTIENE IL CAMPIONE.  Ci finiva, quando non c'e' un `tag`:
    # `_<primo>_<ultimo>` trimestre.  Cosi' pero' il file CAMBIA NOME appena
    # cambia il campione, e quello vecchio resta li' orfano: dopo la correzione
    # della regola di selezione ci si e' ritrovati
    # `..._2016Q4_2022Q4.png` (2 trimestri, vecchio) accanto a
    # `..._2007Q2_2025Q1.png` (58, nuovo), senza niente che dicesse quale dei
    # due fosse quello buono.  Un nome stabile si sovrascrive.  Il campione sta
    # nel titolo della figura e nel CSV accanto, che e' dove va cercato.
    png = cn.figure_rmse_by_horizon(
        ph, sample, "/".join(presenti),
        os.path.join(out_dir, f"rmse_per_orizzonte_bvar{tag or '_completo'}.png"))
    return [csv, png]


def nyfed_tables(mine: pd.DataFrame, models: tuple[str, ...] = MODELS,
                 window: str = "completo"
                 ) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Il confronto col NY Fed in FORMATO LUNGO: non scrive, RESTITUISCE.

    UN FILE PER TABELLA, NON UNO PER COMBINAZIONE.  `compare_nyfed` va chiamato
    una volta per modello piu' una congiunta (il perche' e' nell'intestazione:
    il campione allineato e' una congiunzione fragile, e un modello senza bordo
    la svuoterebbe per tutti).  Scrivendo a ogni giro si ottenevano sei file per
    invocazione, cioe' trenta per finestra: su sette finestre, DUECENTODIECI
    file in una cartella sola, che nessuno apre.

    Qui ogni tabella si porta due colonne in testa — `window` e `modello` — e le
    invocazioni si impilano.  Alla fine sono cinque CSV e un report, e chi vuole
    il taglio per modello o per finestra ci pivota sopra.  E' la stessa scelta
    gia' fatta in `metrics_tables` per le tabelle di famiglia, e per la stessa
    ragione.

    Se il NY Fed non copre il blocco — o il file non c'e' — si torna vuoti con
    un messaggio: le tabelle interne non dipendono da questo.

    LA COPERTURA SI CONTROLLA PRIMA DI ENTRARE, e non e' pignoleria.  Con zero
    trimestri in comune `compare_nyfed.nyfed_final` riceve un frame vuoto da
    `last_before_release`, che non ha nemmeno la colonna `as_of`, e muore con un
    `KeyError: ['as_of'] not in index` — un errore che non dice niente a chi lo
    legge e che arriverebbe DOPO la passata.  Il controllo sta qui perche'
    `compare_nyfed.py` non si tocca.
    """
    vuoto: dict[str, pd.DataFrame] = {}
    try:
        fed_quarters = set(load_nyfed()["target_quarter"].astype(str))
    except FileNotFoundError as exc:
        print(f"  [NY Fed assente] {exc}\n  confronto saltato; le metriche "
              f"interne restano valide.")
        return vuoto, []
    comuni = fed_quarters & set(mine["target_quarter"].astype(str))
    if not comuni:
        mie = sorted(set(mine["target_quarter"].astype(str)))
        print(f"  [{window}] NY Fed: nessuna sovrapposizione: i miei trimestri "
              f"({mie[0]}-{mie[-1]}) sono fuori dalla loro copertura "
              f"({min(fed_quarters)}-{max(fed_quarters)}).  NON e' un errore.")
        return vuoto, []

    out: dict[str, list[pd.DataFrame]] = {}
    testi: list[str] = []
    for spec in (None, *models):
        etichetta = "tutti" if spec is None else spec
        try:
            panel, sample, registro = cn.build_panel(
                mine, None if spec is None else [spec])
        except FileNotFoundError as exc:
            print(f"  [NY Fed assente] {exc}")
            break
        if not sample:
            print(f"  [{window}/{etichetta}] nessun trimestre allineato col "
                  f"NY Fed — salto.")
            continue

        report, tabs = cn.build_report(panel, sample, registro)
        testi.append(cm._section(f"FINESTRA {window}  —  MODELLI: {etichetta}")
                     + "\n" + report)
        for name, t in tabs.items():
            t = t.copy()
            t.insert(0, "modello", etichetta)
            t.insert(0, "window", window)
            out.setdefault(name, []).append(t)
        print(f"  [{window}/{etichetta}] {len(sample)} trimestri allineati: "
              f"{sample[0]}-{sample[-1]}")
        # La FIGURA non si disegna qui: e' una metrica mia e non deve dipendere
        # dalla copertura della Fed.  Vedi `horizon_figures`.

    return ({k: pd.concat(v, ignore_index=True) for k, v in out.items()}, testi)


def _merge_tables(dest: dict[str, list[pd.DataFrame]],
                  new: dict[str, pd.DataFrame]) -> None:
    """Impila le tabelle di un'invocazione su quelle gia' raccolte."""
    for k, v in new.items():
        dest.setdefault(k, []).append(v)


def write_nyfed(collected: dict[str, list[pd.DataFrame]], testi: list[str],
                out_dir: str) -> list[str]:
    """I cinque CSV lunghi piu' un report solo.  Scritti UNA VOLTA, in fondo."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, parts in collected.items():
        p = os.path.join(out_dir, f"nyfed_{name}.csv")
        pd.concat(parts, ignore_index=True).to_csv(p, index=False)
        written.append(p)
    if testi:
        p = os.path.join(out_dir, "confronto_nyfed.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("CONFRONTO COL NY FED STAFF NOWCAST\n"
                     "Un blocco per (finestra, modelli).  Le tabelle in CSV "
                     "sono in formato lungo:\n"
                     "si filtra su `window` e `modello` invece di aprire un "
                     "file diverso.\n" + "\n".join(testi) + "\n")
        written.append(p)
    return written


# ─── 3. Il log score ──────────────────────────────────────────────────────────

#: Le colonne che una riga di log score DEVE avere.  Servono a distinguere i
#: file GREZZI (una riga per settimana x obiettivo x modello, scritti dalla
#: passata) dai riepiloghi che questo modulo scrive nella stessa cartella: il
#: glob `logscore_*.csv` prendeva anche quelli, e il risultato era un
#: `horizon_week` pieno di NaN — un errore che si presenta come crash oscuro
#: nel casting a intero, non come "file sbagliato".  Da qui i riepiloghi si
#: chiamano `riepilogo_*`, e comunque si controlla.
_LS_COLUMNS = {"as_of", "target_quarter", "horizon_week", "spec", "variant",
               "log_score"}


def load_logscores(paths: list[str] | None = None,
                   root: str | None = None) -> pd.DataFrame:
    """I CSV GREZZI di log score prodotti dalla passata, uniti."""
    if not paths:
        # I log score GREZZI stanno accanto ai CSV che li hanno generati
        # (`csv/bvar/logscore/`), non nella cartella di consegna
        # `bvar/logscore/`, che ospita le tabelle e la figura.
        d = os.path.join(root, "logscore") if root else os.path.join(
            layout.bvar_csv_dir(), "logscore")
        paths = sorted(glob.glob(os.path.join(d, "logscore_*.csv")))
        if not paths:
            raise SystemExit(
                f"Nessun log score in {d}.\nLo scrive `evaluate.run_realtime` "
                f"durante la passata: le estrazioni non sopravvivono, quindi "
                f"non si puo' recuperare a posteriori dal CSV.")
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        missing = _LS_COLUMNS - set(d.columns)
        if missing:
            print(f"  [salto] {os.path.basename(p)}: non e' un log score "
                  f"grezzo (mancano {sorted(missing)})")
            continue
        frames.append(d)
    if not frames:
        raise SystemExit("Nessun file di log score grezzo fra quelli indicati.")
    df = pd.concat(frames, ignore_index=True)
    df["metodo"] = df["spec"] + "/" + df["variant"]
    df["fase"] = df["horizon_week"].astype(int).map(cm._phase)
    return df[df["log_score"].notna()]


def logscore_tables(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Media del log score per (metodo, fase) e per metodo.  Piu' alto e' meglio.

    La media, non la somma: i metodi possono avere un numero diverso di righe
    (il Q-BVAR ne ha meno, non avendo bordo) e una somma li penalizzerebbe per
    aver prodotto meno nowcast invece che per averli fatti peggio.

    MA LA MEDIA NON BASTA, ed e' il motivo delle colonne `_com`.  Se i quattro
    modelli hanno righe diverse, quattro medie su quattro campioni diversi non
    si confrontano nemmeno fra loro: un modello che non ha girato sui trimestri
    difficili esce davanti senza aver fatto niente di meglio.  E il modo in cui
    una riga sparisce e' silenzioso — `evaluate.run_realtime` fa `continue`
    quando le estrazioni sono `None`, quindi la riga non c'e' affatto, non c'e'
    con un NaN.  Si riportano quindi entrambe le letture affiancate: `LS_medio`
    su tutto cio' che quel modello ha prodotto, `LS_medio_com` sui soli punti
    (trimestre, settimana) prodotti da TUTTI — e nella tabella per fase il
    comune e' calcolato DENTRO la fase, cosi' che un modello senza bordo perda
    il proprio backcast invece di cancellare quello degli altri.
    """
    from src.forecast.metrics_tables import (common_points,
                                             common_points_by_phase)

    os.makedirs(out_dir, exist_ok=True)

    def _agg(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return (d.groupby(keys)["log_score"]
                .agg(["mean", "median", "std", "count"]).reset_index()
                .rename(columns={"mean": "LS_medio", "median": "LS_mediano",
                                 "std": "LS_sd", "count": "n"}))

    def _join_com(free: pd.DataFrame, com: pd.DataFrame,
                  keys: list[str]) -> pd.DataFrame:
        if com.empty:
            free["n_com"] = 0
            free["LS_medio_com"] = float("nan")
            return free
        c = (com.groupby(keys)["log_score"].agg(["mean", "count"]).reset_index()
             .rename(columns={"mean": "LS_medio_com", "count": "n_com"}))
        return free.merge(c, on=keys, how="left")

    com_ph = common_points_by_phase(df)
    com_all = common_points(df)

    by_phase = _join_com(_agg(df, ["metodo", "fase"]),
                         com_ph, ["metodo", "fase"])
    by_phase["__p"] = by_phase["fase"].map({p: i for i, p in enumerate(_PHASE_ORDER)})
    by_phase = (by_phase.sort_values(["metodo", "__p"]).drop(columns="__p")
                .reset_index(drop=True))
    by_phase = by_phase[["metodo", "fase", "n", "n_com", "LS_medio",
                         "LS_medio_com", "LS_mediano", "LS_sd"]]

    by_method = _join_com(_agg(df, ["metodo"]).drop(columns="LS_sd"),
                          com_all, ["metodo"])
    by_method = (by_method[["metodo", "n", "n_com", "LS_medio",
                            "LS_medio_com", "LS_mediano"]]
                 .sort_values("LS_medio_com", ascending=False)
                 .reset_index(drop=True))

    # `riepilogo_`, non `logscore_`: quest'ultimo e' il prefisso dei file
    # GREZZI della passata, e riusarlo li farebbe rileggere come tali.
    by_phase.to_csv(os.path.join(out_dir, "riepilogo_metodo_fase.csv"), index=False)
    by_method.to_csv(os.path.join(out_dir, "riepilogo_metodo.csv"), index=False)

    n_com_pts = (com_all.groupby(["target_quarter", "horizon_week"]).ngroups
                 if not com_all.empty else 0)
    n_pts = df.groupby(["target_quarter", "horizon_week"]).ngroups
    txt = "\n".join([
        "LOG PREDICTIVE SCORE — GATE 6",
        "LS = log p(y* | Omega_D), kernel gaussiano sulle estrazioni.",
        "PIU' ALTO E' MEGLIO (l'opposto dell'RMSE).",
        f"righe punteggiate: {len(df)}   trimestri: {df['target_quarter'].nunique()}",
        f"punti (trimestre, settimana): {n_pts} in totale, "
        f"{n_com_pts} coperti da TUTTI i modelli",
        "",
        "DUE COLONNE, SI LEGGONO IN COPPIA:",
        "  LS_medio      su tutto cio' che quel modello ha prodotto",
        "  LS_medio_com  sui soli punti prodotti da tutti i modelli",
        "Solo la seconda confronta le righe fra loro.  Se n e n_com coincidono",
        "le due letture sono la stessa e non c'e' niente da interpretare.",
        cm._section("1. METODO x FASE  (comune calcolato DENTRO la fase)"),
        by_phase.to_string(index=False),
        cm._section("2. METODO  (ordinato per LS_medio_com)"),
        by_method.to_string(index=False),
    ])
    print(txt)
    with open(os.path.join(out_dir, "logscore.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    return by_phase


def figure_logscore_by_horizon(df: pd.DataFrame, out_path: str) -> str:
    """
    Log score medio per settimana, con le tre fasi come bande di sfondo.

    Gemella di `compare_nyfed.figure_rmse_by_horizon`, da cui prende bande,
    colori, font e stile degli assi; quella non e' riusabile direttamente
    perche' fissa `ylim(0, ...)` e il log score e' negativo.

    LA REGOLA DEL CAMPIONE E' LA SUA, IN DUE PASSI, e non e' pedanteria: un
    punto calcolato su meno trimestri si muove per COMPOSIZIONE del campione e
    non per merito del modello, e su un asse che va da -12 a +17 quello e'
    l'errore di lettura piu' facile da commettere.

      1. si tengono i trimestri con copertura settimanale COMPLETA.  Ai bordi
         del blocco ce ne sono sempre di monchi — il primo si vede solo in
         backcast, l'ultimo solo in forecast — e sono quelli che
         deformerebbero le due estremita' della curva;
      2. si disegnano le settimane coperte da TUTTI i trimestri rimasti.
    """
    coverage = df.groupby("target_quarter")["horizon_week"].apply(set)
    full = set().union(*coverage) if len(coverage) else set()
    keep = [q for q in coverage.index if coverage[q] == full]
    if not keep:
        print(f"  [log score] nessun trimestre con copertura settimanale "
              f"completa su {len(coverage)}: niente figura.  "
              f"(blocco troppo corto: i trimestri ai bordi sono monchi.)")
        return ""
    df = df[df["target_quarter"].isin(keep)]

    keep_sorted = sorted(keep, key=lambda q: (int(q[:4]), int(q[-1])))
    n_target = len(keep)
    g = (df.groupby(["metodo", "horizon_week"])
         .agg(LS=("log_score", "mean"), n=("target_quarter", "nunique"))
         .reset_index())
    g = g[g["n"] == n_target]
    if g.empty:
        print("  [log score] nessuna settimana con copertura piena: niente figura.")
        return ""

    weeks = np.sort(g["horizon_week"].unique())
    lo_y, hi_y = float(g["LS"].min()), float(g["LS"].max())
    pad = 0.18 * (hi_y - lo_y or 1.0)

    with plt.rc_context(cn._serif()):
        fig, ax = plt.subplots(figsize=cn._FIG_SIZE)
        cn._apply_axes_style(ax)
        for lo, hi, name, colour in cn._PHASE_BANDS:
            ax.axvspan(lo - 0.5, hi + 0.5, color=colour, zorder=0, linewidth=0)
            ax.text((lo + hi) / 2.0, hi_y + pad * 0.92, name, ha="center",
                    va="top", fontsize=8.5, color="#4A4A4A", zorder=1)
        for _, hi, _, _ in cn._PHASE_BANDS[:-1]:
            ax.axvline(hi + 0.5, color="white", linewidth=1.2, zorder=1)

        for i, m in enumerate(sorted(g["metodo"].unique())):
            s = g[g["metodo"] == m].sort_values("horizon_week")
            ax.plot(s["horizon_week"], s["LS"],
                    color=cn._MODEL_COLORS[i % len(cn._MODEL_COLORS)],
                    linewidth=1.9, marker="o", markersize=3.2,
                    label=m.split("/")[0], zorder=3)

        ax.set_xlim(weeks.min() - 0.5, weeks.max() + 0.5)
        ax.set_ylim(lo_y - pad, hi_y + pad)
        ax.set_xlabel("Week relative to the target quarter  "
                      "(<1 forecast, 1-13 nowcast, >13 backcast)", fontsize=10)
        ax.set_ylabel("Log predictive score  (higher is better)", fontsize=10)
        # Il titolo dichiara il campione EFFETTIVO, non il periodo richiesto:
        # la regola in due passi qui sopra scarta i trimestri monchi ai bordi
        # del blocco, quindi `n_target` e' quasi sempre minore dei trimestri
        # presenti nei dati, e su una passata lunga la differenza e' di due
        # trimestri su ottanta — invisibile se non la si scrive.
        span = f"{keep_sorted[0]}–{keep_sorted[-1]}" if keep_sorted else "?"
        dropped = len(coverage) - n_target
        ax.set_title(
            f"Log score by horizon — {span}  ({n_target} quarters"
            + (f", {dropped} truncated at the block edge dropped)" if dropped
               else ")"),
            fontsize=12, pad=12)
        ax.legend(loc="lower right", fontsize=8.5, frameon=True,
                  framealpha=0.95, edgecolor="#B0B0B0", ncol=2).set_zorder(5)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, facecolor="white")
        plt.close(fig)
    print(f"  scritto: {out_path}")
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Metriche del Gate 6: RMSE, NY Fed, log score.")
    p.add_argument("--csv", nargs="*", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--salta-nyfed", action="store_true")
    p.add_argument("--window", nargs="*", default=None,
                   help="finestre da produrre (default: le tre passate RMSE "
                        "piu' i tre zoom, come il DFM). Ogni finestra scrive "
                        "file con il proprio nome nel tag.")
    p.add_argument("--solo-completo", action="store_true",
                   help="solo la passata sull'intero campione, senza le "
                        "finestre: e' il comportamento di prima")
    a = p.parse_args()

    # Le uscite vanno nell'albero di `output_layout`, non piu' accanto ai CSV
    # sorgente: i CSV grezzi restano dove la passata li scrive (`--output-root`),
    # le TABELLE e le FIGURE finiscono in bvar/rmse e bvar/logscore.
    root = a.output_root or OUTPUT_ROOT
    rmse_dir = layout.bvar_rmse_dir()
    ls_dir = layout.bvar_logscore_dir()
    # `tag` nomina il file; da solo NON basta.  Se cambiasse solo il nome,
    # `--window 2007-2010` produrrebbe un file intitolato 2007-2010 calcolato
    # su TUTTO — numeri giusti sotto un titolo sbagliato, lo stesso difetto
    # gia' chiuso nelle tabelle.  Il taglio vero e' sotto, su `ls`.
    paths = discover_csvs(a.csv)
    models = tuple(a.models.split(","))

    # I CSV si leggono UNA VOLTA e poi si affetta: sono gli stessi dati per
    # tutte le finestre, e rileggerli sei volte sarebbe solo piu' lento.
    df_all, n_tot, n_drop = cm.load_long(paths)

    print(cm._section("RMSE — CAMPIONE COMPLETO"))
    mine = rmse_report(None, rmse_dir, df=df_all, n_tot=n_tot, n_drop=n_drop)

    # Le tabelle NY Fed si ACCUMULANO e si scrivono in fondo, una volta sola.
    fed_tabs: dict[str, list[pd.DataFrame]] = {}
    fed_txt: list[str] = []
    if not a.salta_nyfed:
        print(cm._section("NY FED — l'ultima stima prima del rilascio"))
        t, txt = nyfed_tables(mine, models, "completo")
        _merge_tables(fed_tabs, t); fed_txt += txt
    horizon_figures(mine, rmse_dir, models)

    # ── Le finestre, come per il DFM ──────────────────────────────────────
    # Tre passate cumulate piu' tre zoom.  Ogni finestra e' un TAGLIO VERO dei
    # dati, non solo un nome diverso sul file: il difetto opposto — numeri su
    # tutto sotto un titolo di periodo — e' quello gia' chiuso nelle tabelle.
    finestre = ([] if a.solo_completo
                else (a.window if a.window is not None
                      else list(layout.RMSE_PASSES) + list(layout.RMSE_ZOOM_WINDOWS)))
    for w in finestre:
        d = layout.slice_window(df_all, w, column="as_of")
        if d.empty:
            print(f"\n  [{w}] nessuna riga in {layout.window(w)}: finestra "
                  f"saltata (la passata non copre questo periodo).")
            continue
        print(cm._section(f"RMSE — FINESTRA {w}  {layout.window(w)}\n"
                          f"{len(d)} righe, {d['target_quarter'].nunique()} "
                          f"trimestri"))
        tag = f"_{w}"
        # LE TABELLE PER FINESTRA NON SI SCRIVONO QUI.  `rmse_report` produce
        # tre file per finestra (metriche.txt + due CSV), cioe' ventuno su
        # sette finestre — e sarebbero DUPLICATI: `metrics_tables` scrive gia'
        # gli stessi numeri in `metrics_bvar.csv` e `metrics_bvar_by_phase.csv`,
        # in formato lungo con la colonna `window` E con le colonne n_com che
        # qui non ci sono.  Il report narrativo dettagliato resta uno, sul
        # campione completo; per finestra si legge il formato lungo.
        # Qui serve solo il frame, per la Fed e per la figura.
        mine_w = d
        if not a.salta_nyfed:
            # SULLE FINESTRE, SOLO LA PASSATA CONGIUNTA — non le cinque.
            #
            # `nyfed_report` gira una volta per modello piu' una congiunta
            # (il perche' e' nell'intestazione: il campione allineato e' una
            # congiunzione fragile e un modello senza bordo la svuoterebbe per
            # tutti), e ogni invocazione scrive sei file.  Sono trenta file per
            # finestra: per sette finestre farebbero DUECENTODIECI file in una
            # cartella sola, e a quel punto nessuno li apre piu'.
            #
            # Il dettaglio modello-per-modello serve dove si legge davvero, cioe'
            # sul campione completo, ed e' li' che resta.  Sulle finestre basta
            # la congiunta: sei file invece di trenta.
            #
            # La FIGURA per orizzonte, che e' quello che si guarda per primo,
            # continua a uscire per ogni finestra — e con la Fed dentro, perche'
            # `horizon_panel` la carica da se'.
            t, txt = nyfed_tables(mine_w, (), w)
            _merge_tables(fed_tabs, t); fed_txt += txt
        horizon_figures(mine_w, rmse_dir, models, tag=tag)

    if fed_tabs or fed_txt:
        scritti = write_nyfed(fed_tabs, fed_txt, rmse_dir)
        print(f"\n  NY Fed: {len(scritti)} file in formato lungo "
              f"(colonne `window` e `modello`), non uno per combinazione.")

    print(cm._section("LOG PREDICTIVE SCORE"))
    # `root=None` quando l'utente non l'ha imposto: cosi' `load_logscores`
    # usa la sua casa vera (`csv/bvar/logscore/`).  Passare `OUTPUT_ROOT`
    # la mandava a cercare in `forecast_weekly/logscore/`, che non esiste.
    ls = load_logscores(root=a.output_root)
    # IL LOG SCORE RESTA UNO SOLO, SUL CAMPIONE COMPLETO — scelta dichiarata
    # nell'intestazione del modulo e non cambiata qui: la figura confronta i
    # quattro modelli fra loro e con un pannello solo quel confronto si legge
    # in un colpo.  Le finestre le vuole l'RMSE, che va confrontato col
    # benchmark dello stesso periodo; il log score no.
    logscore_tables(ls, ls_dir)
    figure_logscore_by_horizon(
        ls, os.path.join(ls_dir, "logscore_by_horizon.png"))


if __name__ == "__main__":
    main()
