"""
notebooks/tabelle_risultati.py

DALLA PASSATA ALLA SEZIONE 5: inventario degli artefatti, tabelle .tex, figure.

    python notebooks/tabelle_risultati.py --check     cosa c'e' e cosa manca
    python notebooks/tabelle_risultati.py             genera tabelle e figure
    python notebooks/tabelle_risultati.py --spec diag4    cambia famiglia guida

Scrive in docs/tesi/tables/ e copia in docs/tesi/figures/, come
`tabelle_data.py` e `bai_ng_test.py`.

PERCHE' UN INVENTARIO E NON DUE `read_csv`
------------------------------------------
La passata produce un centinaio di file e i loro NOMI SONO CAMBIATI: nel
rinominare dell'agosto 2026 le tabelle italiane sono diventate inglesi
(`metriche*` -> `metrics_*`, `confronto_nyfed` -> `nyfed_comparison`,
`nyfed_campione` -> `nyfed_sample`, `riepilogo_*` -> `report_logscore_*`).
Uno script che apre due file per nome si rompe in silenzio alla prossima
passata, o peggio legge un file vecchio rimasto sul disco e produce una
tabella plausibile con i numeri sbagliati.

Qui ogni artefatto sta in `SOURCES` una volta sola, con il suo nome ATTUALE e
i nomi VECCHI come ripiego; `--check` dice, prima di generare qualunque cosa,
quali ci sono, quali mancano e quali sono stati trovati solo col nome vecchio.
Quando la passata nuova arriva, `--check` e' il primo comando da dare.

I NOMI VENGONO DAL CODICE, NON DAL DISCO
----------------------------------------
Sono letti dai moduli che li scrivono, non dalla cartella `output/` di una
passata particolare:
    src/forecast/metrics_tables.py   metrics_<spec>{,_by_phase}.csv/.txt,
                                     metrics_bvar*, e le matrici di comparison/
    src/forecast/nyfed_all.py        nyfed_<tabella>.csv, nyfed_comparison.txt,
                                     rmse_by_horizon_<spec>.csv
    src/forecast/compare_nyfed.py    le cinque tabelle: accuracy, vs_fed,
                                     fed_by_quarter, alignment, sample
    src/bvar/metrics.py              report_bvar*, report_logscore*,
                                     rmse_by_horizon_bvar_<finestra>.csv

IL LOG SCORE E' SOLO DEI BVAR, e va ricordato prima di metterlo in una
tabella. `src/forecast/` non scrive nessun log score: il DFM di questa tesi
produce una stima centrale, non una densita' predittiva. Una tabella di log
score confronta quindi i quattro BVAR fra loro, mai il DFM con loro, e
presentarla accanto alle altre inviterebbe esattamente la lettura sbagliata.

LE FIGURE SI COPIANO, NON SI RIDISEGNANO
----------------------------------------
Ridisegnarle qui vorrebbe dire tenere due volte lo stesso codice di figura e
vederle divergere alla terza modifica. La passata le disegna, questo script le
mette dove il .tex le cerca.

L'ECCEZIONE: LA NOTA A PIE' DI FIGURA, CHE IN TESI NON CI VA
-----------------------------------------------------------
Le figure della passata portano in fondo un blocco di note, e li' e' giusto:
viaggiano da sole, al relatore e nelle slide. In tesi no. La ragione e'
tipografica: la figura e' larga 11 pollici e entra a `\\textwidth`, che con
`12pt letterpaper, margin=1in` vale 6.5 — scala 0.59, e una nota a 8.5 pt
diventa 5 pt contro un corpo di 12. Illeggibile, e un blocco illeggibile e'
peggio di nessun testo; quel contenuto va nella `\\caption`, che si compone
col testo ed entra nell'elenco delle figure.

Le figure in `REDRAW` sono percio' RIGENERATE con `note=False` invece che
copiate. Questo NON viola la regola qui sopra: non si riscrive il codice della
figura, si richiama **la stessa funzione** che usa la passata
(`compare_nyfed.figure_rmse_by_horizon`), col solo interruttore girato. Se la
rigenerazione non riesce si ripiega sulla copia, dicendolo.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_TAB = ROOT / "docs" / "tesi" / "tables"
OUT_FIG = ROOT / "docs" / "tesi" / "figures"
PASS = ROOT / "output" / "forecast_weekly"

#: Le tre strutture di caricamento e i quattro BVAR, come in `output_layout`.
SPECS = ("fed_overlap", "diag4", "diag3")
BVARS = ("qbvar", "cbvar", "bbvar", "lbvar")

#: La famiglia che GUIDA le figure della sezione. Default `fed_overlap`: e'
#: quella costruita per sovrapporsi al blocco della NY Fed, quindi la sola su
#: cui il confronto esterno e' un confronto. Se una passata futura ribalta la
#: classifica delle famiglie si passa `--spec`, e si riscrive in tesi la frase
#: che dichiara la scelta: e' argomentata li', non nascosta qui.
SPEC_DEFAULT = "fed_overlap"

#: Le tre passate cumulate della tabella di famiglia, nell'ordine di lettura.
WINDOWS_FAMILY = ("2007-2019", "2007-2021", "2007-2025")

#: Le sei colonne della matrice: cumulate prima, zoom su un regime poi.
WINDOWS_ALL = ("2007-2019", "2007-2021", "2007-2025",
               "2007-2010", "2019-2021", "2024-2025")


# ─── L'inventario ─────────────────────────────────────────────────────────────

class Src:
    """
    Un artefatto della passata: dove sta ora, come si chiamava prima, a cosa
    serve. `{spec}`, `{bvar}` e `{w}` sono sostituiti dal chiamante.
    """

    def __init__(self, path: str, what: str, legacy: tuple[str, ...] = ()):
        self.path, self.what, self.legacy = path, what, legacy

    def resolve(self, **kw) -> tuple[Path | None, bool]:
        """(percorso trovato, e' un nome vecchio). (None, False) se manca."""
        p = PASS / self.path.format(**kw)
        if p.exists():
            return p, False
        for old in self.legacy:
            q = PASS / old.format(**kw)
            if q.exists():
                return q, True
        return None, False


#: OGNI artefatto che questa tesi puo' voler leggere. I nomi `legacy` sono
#: quelli precedenti al rinominare dell'agosto 2026.
SOURCES: dict[str, Src] = {
    # ── DFM, per struttura di caricamento ────────────────────────────────────
    "dfm_family": Src(
        "dfm/{spec}/rmse/metrics_{spec}.csv",
        "tabella di famiglia: 5 varianti + NY Fed + benchmark, per finestra",
        legacy=("dfm/{spec}/rmse/metriche_{spec}.csv",)),
    "dfm_family_phase": Src(
        "dfm/{spec}/rmse/metrics_{spec}_by_phase.csv",
        "la stessa, spezzata per fase (forecast / M1 M2 M3 / backcast)",
        legacy=("dfm/{spec}/rmse/metriche_{spec}_fase.csv",)),
    "dfm_by_horizon": Src(
        "dfm/{spec}/rmse/rmse_by_horizon_{spec}.csv",
        "il pannello sotto la figura: RMSE per (metodo, settimana)"),
    "dfm_nyfed_accuracy": Src(
        "dfm/{spec}/rmse/nyfed_accuracy.csv",
        "io e la Fed sull'ultima stima prima del rilascio"),
    "dfm_nyfed_vs_fed": Src(
        "dfm/{spec}/rmse/nyfed_vs_fed.csv",
        "differenze medie, correlazione, quota di trimestri in cui vinco"),
    "dfm_nyfed_by_quarter": Src(
        "dfm/{spec}/rmse/nyfed_fed_by_quarter.csv",
        "la Fed trimestre per trimestre: quanto vale il metro"),
    "dfm_nyfed_alignment": Src(
        "dfm/{spec}/rmse/nyfed_alignment.csv",
        "verifica che i due esercizi cadano sullo stesso venerdi'"),
    "dfm_nyfed_sample": Src(
        "dfm/{spec}/rmse/nyfed_sample.csv",
        "composizione del campione: cosa e' entrato e cosa no",
        legacy=("dfm/{spec}/rmse/nyfed_campione.csv",)),
    "dfm_nyfed_panel": Src(
        "dfm/{spec}/rmse/nyfed_panel_final.csv",
        "il pannello finale riga per riga"),

    # ── BVAR ─────────────────────────────────────────────────────────────────
    "bvar_family": Src(
        "bvar/rmse/metrics_bvar.csv",
        "i quattro BVAR insieme, per finestra",
        legacy=("bvar/rmse/metriche_metodo.csv", "bvar/rmse/metriche.csv")),
    "bvar_family_phase": Src(
        "bvar/rmse/metrics_bvar_by_phase.csv",
        "i quattro BVAR per fase",
        legacy=("bvar/rmse/metriche_metodo_fase.csv",)),
    "bvar_report": Src(
        "bvar/rmse/report_bvar_by_model.csv",
        "il rapporto specifico dei BVAR, per modello"),
    "bvar_report_phase": Src(
        "bvar/rmse/report_bvar_by_model_phase.csv",
        "lo stesso, per modello e fase"),
    "bvar_by_horizon": Src(
        "bvar/rmse/rmse_by_horizon_bvar_{w}.csv",
        "RMSE per orizzonte dei BVAR, un file per finestra"),

    # ── Log score: SOLO BVAR. Vedi la nota nel docstring. ────────────────────
    "logscore": Src(
        "bvar/logscore/report_logscore_by_model.csv",
        "log score dei quattro BVAR (il DFM non ne ha uno)",
        legacy=("bvar/logscore/riepilogo_metodo.csv",)),
    "logscore_phase": Src(
        "bvar/logscore/report_logscore_by_model_phase.csv",
        "log score per modello e fase",
        legacy=("bvar/logscore/riepilogo_metodo_fase.csv",)),

    # ── Confronto di tutti contro tutti ──────────────────────────────────────
    "matrix_rmse": Src(
        "comparison/rmse_matrix.csv",
        "RMSE grezza, metodi x finestre (NON confrontabile fra colonne)"),
    "matrix_rel": Src(
        "comparison/rmse_rel_ar2_matrix.csv",
        "RMSE relativa all'AR(2): l'unica confrontabile fra finestre"),
    "matrix_mda": Src(
        "comparison/mda_matrix.csv",
        "accuratezza direzionale, metodi x finestre"),
    "matrix_backcast": Src(
        "comparison/backcast_matrix.csv",
        "il solo backcast, dove il DFM legge un mese che i BVAR non hanno"),
    "phase": Src(
        "comparison/rmse_by_phase.csv",
        "tutti i metodi per fase e finestra"),
    "extremes": Src(
        "comparison/extremes.csv",
        "i trimestri estremi e la quota di movimento catturata"),
}

#: Le figure, dalla passata a docs/tesi/figures/. Aggiungerne una e' una riga.
FIGURES: dict[str, tuple[str, str]] = {
    "rmse_pre":   ("dfm/{spec}/rmse/RMSE_2007-2019.png", "fig_rmse_2007_2019.png"),
    "rmse_full":  ("dfm/{spec}/rmse/RMSE_2007-2025.png", "fig_rmse_2007_2025.png"),
    "mda":        ("dfm/{spec}/mda/MDA_2007-2025.png",   "fig_mda_2007_2025.png"),
    "traj_crisis": ("dfm/{spec}/student_t/Trajectories_2007-2010.png",
                    "fig_traj_2007_2010.png"),
    # I BVAR: la spec non entra nel percorso, la famiglia e' una sola.
    "bvar_rmse_pre":  ("bvar/rmse/RMSE_2007-2019.png",
                       "fig_bvar_rmse_2007_2019.png"),
    "bvar_rmse_full": ("bvar/rmse/RMSE_2007-2025.png",
                       "fig_bvar_rmse_2007_2025.png"),
    "bvar_traj_crisis": ("bvar/lbvar/Trajectories_2007-2010.png",
                         "fig_bvar_traj_2007_2010.png"),
    "bvar_traj_covid":  ("bvar/lbvar/Trajectories_2019-2021.png",
                         "fig_bvar_traj_2019_2021.png"),
    "traj_covid": ("dfm/{spec}/student_t/Trajectories_2019-2021.png",
                   "fig_traj_2019_2021.png"),
    "traj_recent": ("dfm/{spec}/student_t/Trajectories_2024-2025.png",
                    "fig_traj_2024_2025.png"),
}

#: Le figure che si RIGENERANO con la nota spenta invece di copiarle, e la
#: finestra con cui farlo. Vedi il docstring: in tesi la nota non ci va.
#: Chi non e' qui dentro si copia e basta, nota compresa.
REDRAW: dict[str, str] = {
    "rmse_pre":  "2007-2019",
    "rmse_full": "2007-2025",
}


#: Quali figure servono ADESSO alla sezione. Le altre restano nel registro,
#: pronte, ma non si copiano: una figura in `figures/` che nessun \includegraphics
#: cerca e' solo un file che invecchia.
FIGURES_WANTED = ("rmse_pre", "rmse_full", "traj_crisis", "traj_covid",
                  "bvar_rmse_pre", "bvar_rmse_full",
                  "bvar_traj_crisis", "bvar_traj_covid")


# ─── Etichette ────────────────────────────────────────────────────────────────

VARIANT_LABEL = {
    "gaussian":             r"Gaussian",
    "gaussian_ar1":         r"Gaussian, AR(1) idio.",
    "student_t":            r"Student-$t$",
    "student_t_ar1":        r"Student-$t$, AR(1) idio.",
    # `shared` isola i pesi per-serie dall'AR(1): stessa legge a code pesanti,
    # ma un peso solo per tutte le serie (`per_series_weights = False`).
    "student_t_ar1_shared": r"Student-$t$, AR(1) idio., shared weights",
}

BENCH_LABEL = {"nyfed": r"New York Fed", "ar2": r"\textsc{ar}(2)",
               "mean": r"Expanding mean"}

#: Le tre strutture di loading come si LEGGONO, non come si chiamano nel
#: config.  Accanto a righe scritte "Student-$t$, AR(1) idio., shared weights"
#: un identificativo Python era l'unica cosa fuori registro nella stessa
#: tabella: i due assi della griglia erano stampati in due lingue diverse.
#: Le chiavi `fed_overlap`, `diag4` e `diag3` restano stampate in un punto
#: solo, la nota della Tabella A.1, dichiarate per quello che sono, puntatori
#: a `config/factor_specs.json` per chi riproduce.
#:
#: Senza articolo davanti: il titolo di gruppo si compone "Factor model,
#: <etichetta>", e "Factor model, the overlapping structure" non si legge.
SPEC_LABEL = {
    "fed_overlap": r"overlapping structure",
    "diag4":       r"four-factor block-diagonal structure",
    "diag3":       r"three-factor block-diagonal structure",
}

BVAR_LABEL = {"qbvar/-": r"Q-\textsc{bvar}", "cbvar/authors": r"C-\textsc{bvar}",
              "bbvar/-": r"B-\textsc{bvar}", "lbvar/-": r"L-\textsc{bvar}"}


def _num(x, dec: int) -> str:
    """
    Un numero, o un trattino se manca. Mai una cella vuota.

    Il trattino e' `\\textendash{}` e non `--` scritto a mano: sono lo stesso
    segno, ma la forma esplicita dice che qui il trattino e' un SEGNO DI DATO
    ASSENTE e non un intervallo, e `chktex` smette di segnalarlo come dash di
    lunghezza sbagliata (Warning 8) in mezzo a una riga di numeri.

    IL SEGNO MENO E' `$-$` E NON IL TRATTINO DELLA TASTIERA.  In modo testo
    `-0.23` esce con un hyphen, che e' piu' corto del meno e in una colonna di
    numeri si legge male.  Le cifre restano fuori dalla matematica: cosi' la
    colonna ha un solo tipo di cifra, che `$-0.23$` non garantirebbe.
    """
    if x is None or pd.isna(x):
        return r"\textendash{}"
    s = f"{float(x):.{dec}f}"
    return f"$-${s[1:]}" if s.startswith("-") else s


def _label(metodo: str) -> str:
    if metodo in BENCH_LABEL:
        return BENCH_LABEL[metodo]
    if metodo in BVAR_LABEL:
        return BVAR_LABEL[metodo]
    spec, _, variant = metodo.partition("/")
    if spec in SPEC_LABEL:
        return f"{SPEC_LABEL[spec]}, {VARIANT_LABEL.get(variant, variant)}"
    return metodo.replace("_", r"\_")


def _win(w: str) -> str:
    """
    Il nome di una finestra come va STAMPATO: `2007--2019`, non `2007-2019`.

    Le finestre arrivano dai dati con un trattino corto, che in un intervallo
    numerico e' l'en dash: il testo della tesi scrive gia' `2007--2019`, e
    `chktex` segnala la differenza (Warning 8, "wrong length of dash") su ogni
    riga di intestazione. Si converte qui, all'ultimo momento, cosi' che le
    chiavi restino quelle dei dati e solo la stampa cambi.
    """
    return w.replace("-", "--")


def _bold(s: str, yes: bool) -> str:
    return rf"\textbf{{{s}}}" if yes else s


# ─── Tabella 1: la famiglia che guida ─────────────────────────────────────────

def tabella_famiglia(df: pd.DataFrame, spec: str) -> str:
    """
    Tre blocchi, uno per passata cumulata; dentro ogni blocco le cinque
    varianti, poi la NY Fed, poi i due benchmark. L'ordine e' FISSO e non
    dipende dai numeri: una tabella che si riordina da sola fra una passata e
    l'altra non si puo' leggere due volte.

    `spec="bvar"` la usa per la famiglia BVAR: stesse colonne e stesso ordine,
    righe i quattro modelli, poi la Fed, poi i due benchmark.  LA FED C'E' IN
    ENTRAMBE, ed e' il punto: mettendola solo nella prima, le colonne `_com`
    delle due tabelle finivano su campioni diversi e la relativa non si poteva
    confrontare fra l'una e l'altra.  Da dove viene la sua riga qui, e perche'
    non la si incolla dal CSV, sta in `_bvar_con_fed`.

    Le colonne libere e quelle sul campione comune stanno affiancate: la
    distanza fra `RMSE` e `RMSE_com` e' il dato, non un dettaglio, ed e'
    visibile soprattutto sulla riga della Fed, che non pubblica nelle
    settimane di previsione profonda.

    LA RELATIVA SI CALCOLA QUI, SUL COMUNE, E NON SI PRENDE DAL CSV
    --------------------------------------------------------------
    `RMSE_rel_ar2` arriva da `table_by_method` sul frame LIBERO: e' l'RMSE
    libero del metodo diviso l'RMSE libero dell'AR(2).  Per i miei metodi va
    bene, numeratore e denominatore hanno le stesse coppie.  PER LA FED NO: il
    numeratore ne ha 1014 e il denominatore 1382, e le 368 che mancano sono le
    previsioni profonde, cioe' le peggiori per chiunque.  Il rapporto che ne
    esce non e' una misura, e' un artefatto del campione.

    Quanto pesa, misurato: sul 2007-2019 la Fed passa da 0.864 (dentro il
    range delle cinque varianti) a 0.881 (dietro tutte e cinque); sul
    2007-2021 da 0.602, che la farebbe sembrare il metodo MIGLIORE della
    tabella, a 0.587, che e' l'ULTIMO dei sei.  La graduatoria si ribaltava.

    Percio' la colonna si ricalcola qui come `RMSE_com / RMSE_com[ar2]`: tutte
    le righe sullo stesso campione, che e' poi quello che la didascalia in tesi
    dichiara gia' ("divides by the AR(2) on the same pairs").  Dove `n` e
    `n_com` coincidono - la finestra 2007-2025, in tutte e due le tabelle - il
    numero non cambia di una cifra.

    RESTA FUORI `SignAcc`, che nel CSV c'e' solo in versione libera: darne la
    gemella comune vorrebbe dire toccare `_COMMON_COLS` in
    `src/forecast/metrics_tables.py`, e questo file non ci arriva.
    """
    modelli = (list(BVAR_LABEL) if spec == "bvar"
               else [f"{spec}/{v}" for v in VARIANT_LABEL])
    ordine = modelli + ["nyfed", "ar2", "mean"]
    righe = [
        r"\begin{tabular}{@{}l rr rr r rr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Sample} & \multicolumn{3}{c}{\textsc{rmse}} "
        r"& \multicolumn{2}{c}{Direction} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-8}",
        r"& $n$ & $n_{\mathrm{com}}$ & free & common & rel.\ \textsc{ar}(2) & "
        r"\textsc{mda} & Sign \\",
        r"\midrule",
    ]
    for k, w in enumerate(WINDOWS_FAMILY):
        sub = df[df["window"] == w].set_index("metodo")
        presenti = [m for m in ordine if m in sub.index]
        if not presenti:
            continue
        if k:
            righe.append(r"\addlinespace[0.4em]")
        n_q = int(sub.loc[presenti[0], "n_trimestri"])
        righe.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\itshape {_win(w)} "
                     rf"({n_q} target quarters)\/}} \\[0.15em]")
        vals = [sub.loc[m, "RMSE_com"] for m in presenti]
        finite = [v for v in vals if not pd.isna(v)]
        lo = min(finite) if finite else None
        # Il denominatore della relativa: l'AR(2) sulle STESSE coppie.  Se il
        # blocco non ha la riga dell'AR(2) la colonna resta vuota, che e' meglio
        # di un rapporto contro un campione diverso.
        base = (float(sub.loc["ar2", "RMSE_com"])
                if "ar2" in sub.index and not pd.isna(sub.loc["ar2", "RMSE_com"])
                else None)
        for m, v in zip(presenti, vals):
            r = sub.loc[m]
            # Il prefisso della spec e' ridondante: la tabella e' tutta quella
            # spec, e lo dice la didascalia.
            nome = (VARIANT_LABEL.get(m.partition("/")[2], _label(m))
                    if m.startswith(f"{spec}/") else _label(m))
            righe.append(
                f"{nome} & {int(r['n'])} & {int(r['n_com'])} & "
                f"{_num(r['RMSE'], 2)} & "
                f"{_bold(_num(v, 2), lo is not None and v == lo)} & "
                f"{_num(v / base if base else None, 3)} & "
                f"{_num(r['MDA_com'], 3)} & {_num(r['SignAcc'], 3)} \\\\")
    return "\n".join(righe + [r"\bottomrule", r"\end{tabular}"]) + "\n"


# ─── Tabella 2: tutti i metodi, sola RMSE relativa ────────────────────────────

def tabella_matrice(m: pd.DataFrame) -> str:
    """
    Metodi x finestre, SOLO `RMSE_rel_ar2`.

    La RMSE grezza non e' confrontabile fra finestre (nel 2020 sbagliano tutti
    di piu'), quindi una matrice di RMSE grezze inviterebbe la lettura
    sbagliata: si confronterebbe una colonna con l'altra. La relativa
    normalizza sul benchmark dello stesso periodo. Il grassetto e' per
    COLONNA, benchmark inclusi: se vince l'AR(2), lo dice la tabella.
    """
    m = m.set_index("metodo")
    gruppi = [(f"Factor model, {SPEC_LABEL[s]}",
               [f"{s}/{v}" for v in VARIANT_LABEL]) for s in SPECS]
    gruppi += [(r"Bayesian \textup{\textsc{var}}", list(BVAR_LABEL)),
               ("Benchmark", ["ar2", "mean"])]
    cols = [w for w in WINDOWS_ALL if w in m.columns]

    righe = [r"\begin{tabular}{@{}l " + "r" * len(cols) + r"@{}}", r"\toprule"]
    if len(cols) == 6:
        righe += [r"& \multicolumn{3}{c}{Cumulative} & "
                  r"\multicolumn{3}{c}{Single regime} \\",
                  r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}"]
    righe += ["& " + " & ".join(_win(c) for c in cols) + r" \\", r"\midrule"]

    tutte = [x for _, mem in gruppi for x in mem if x in m.index]
    best = {c: min((m.loc[x, c] for x in tutte if not pd.isna(m.loc[x, c])),
                   default=None) for c in cols}

    for k, (titolo, membri) in enumerate(gruppi):
        presenti = [x for x in membri if x in m.index]
        if not presenti:
            continue
        if k:
            righe.append(r"\addlinespace[0.4em]")
        righe.append(rf"\multicolumn{{{len(cols) + 1}}}{{@{{}}l}}"
                     rf"{{\itshape {titolo}\/}} \\[0.15em]")
        for x in presenti:
            celle = [_bold(_num(m.loc[x, c], 3),
                           best[c] is not None and not pd.isna(m.loc[x, c])
                           and m.loc[x, c] == best[c]) for c in cols]
            nome = (VARIANT_LABEL.get(x.partition("/")[2], _label(x))
                    if x.partition("/")[0] in SPEC_LABEL else _label(x))
            righe.append(f"{nome} & " + " & ".join(celle) + r" \\")
    return "\n".join(righe + [r"\bottomrule", r"\end{tabular}"]) + "\n"


# ─── Inventario e generazione ─────────────────────────────────────────────────

def check(spec: str) -> int:
    """
    Cosa c'e' e cosa manca, PRIMA di generare qualunque cosa. E' il comando da
    dare quando arriva una passata nuova. Torna il numero di artefatti assenti.
    """
    print(f"passata: {PASS}")
    print(f"famiglia guida: {spec}\n")
    mancanti = 0
    for key, src in SOURCES.items():
        kw = {"spec": spec, "w": WINDOWS_FAMILY[0], "bvar": BVARS[0]}
        p, vecchio = src.resolve(**kw)
        if p is None:
            stato, mancanti = "MANCA   ", mancanti + 1
        elif vecchio:
            stato = "vecchio "
        else:
            stato = "ok      "
        print(f"  {stato} {key:22s} {src.what}")
        if vecchio:
            print(f"           trovato col nome VECCHIO: "
                  f"{p.relative_to(PASS)}")
    print()
    for key in FIGURES:
        src, _ = FIGURES[key]
        p = PASS / src.format(spec=spec)
        marca = "ok      " if p.exists() else "MANCA   "
        if not p.exists():
            mancanti += 1
        usata = "  (serve alla sezione)" if key in FIGURES_WANTED else ""
        print(f"  {marca} figura {key}{usata}")
    print(f"\n{mancanti} assenti." if mancanti else "\nTutto presente.")
    return mancanti


def tabella_estremi(df: pd.DataFrame, tutte_le_spec: bool = False) -> str:
    """
    I quattro trimestri estremi, e QUANTO del movimento ciascun metodo prende.

    La colonna e' `quota_catturata`: uno vuol dire preso tutto, zero vuol dire
    fermi sulla media.  Serve alla 5.2 perche' misura in un numero solo cio'
    che le traiettorie mostrano, la COMPRESSIONE promessa da `nowcasting.tex`:
    il modello vede il verso ma non l'ampiezza.

    IL SEGNO CONTA PIU' DELLA GRANDEZZA, ed e' la ragione per cui l'AR(2) sta
    in tabella e non solo nel testo.  Una quota negativa non e' "compressione
    forte": e' il metodo che si muove NELLA DIREZIONE OPPOSTA al movimento
    vero.  Il benchmark lo fa in tre dei quattro trimestri, il che distingue
    una stima prudente da una sbagliata di verso, e toglie il sospetto che la
    compressione del fattore sia semplicemente un difetto.

    La NY Fed sta in fondo perche' comprime quanto noi: e' la prova che la
    compressione appartiene all'esercizio e non a questa implementazione.
    Sul 2021Q4 non ha una riga, per la sospensione.

    `tutte_le_spec` PER L'APPENDICE, E NON PER IL CORPO.  In tesi la 5.2 mostra
    la sola famiglia guida, come la 5.1 mostra una sola struttura: la griglia
    completa vive in appendice.  La ragione non e' solo di spazio.  Sul 2021Q4,
    che e' il meno estremo dei quattro, `diag3/student_t_ar1` prende 0.94 e
    `diag3/gaussian` 0.87: messe accanto alla frase della 5.2 ("none approaches
    the whole", vera nei CROLLI, dove il massimo delle quindici celle resta
    0.43, 0.51 e 0.68) quelle due cifre la farebbero sembrare smentita dalla
    tabella che la sostiene.  In appendice, con tutte le righe sotto gli occhi,
    il confronto si legge per quello che e'.
    """
    r = df.drop_duplicates("target").set_index("target")["realizzato"]
    cols = [t for t in ["2008Q4", "2020Q2", "2020Q3", "2021Q4"] if t in r.index]
    m = df.pivot_table(index="metodo", columns="target",
                       values="quota_catturata")
    # UN GRUPPO SOLO per i due termini di paragone: separarli dava un titolo
    # "New York Fed" sopra una riga "New York Fed", cioe' la stessa parola due
    # volte per una riga sola.
    spec_mostrate = SPECS if tutte_le_spec else (SPEC_DEFAULT,)
    gruppi = [(f"Factor model, {SPEC_LABEL[s]}",
               [f"{s}/{v}" for v in VARIANT_LABEL]) for s in spec_mostrate]
    gruppi += [(r"Bayesian \textup{\textsc{var}}", list(BVAR_LABEL)),
               ("For comparison", ["ar2", "nyfed"])]

    # DUE UNITA' IN UNA TABELLA, E VANNO DICHIARATE.  La prima riga e' in punti
    # percentuali annualizzati, tutte le altre sono quote fra zero e uno: senza
    # la banda che le separa il lettore legge la colonna per intero e trova
    # -28.0 sopra 0.51 senza sapere che non sono la stessa cosa.
    righe = [r"\begin{tabular}{@{}l " + "r" * len(cols) + r"@{}}", r"\toprule",
             "& " + " & ".join(cols) + r" \\",
             r"\cmidrule(l){2-" + str(len(cols) + 1) + "}",
             r"\itshape Realised growth & "
             + " & ".join(rf"\itshape $-${abs(r[c]):.1f}" if r[c] < 0
                          else rf"\itshape $+${r[c]:.1f}" for c in cols)
             + r" \\",
             r"\addlinespace[0.3em]",
             rf"& \multicolumn{{{len(cols)}}}{{c}}"
             r"{\itshape Share of that movement captured\/} \\",
             r"\midrule"]
    for k, (titolo, membri) in enumerate(gruppi):
        presenti = [x for x in membri if x in m.index]
        if not presenti:
            continue
        if k:
            righe.append(r"\addlinespace[0.4em]")
        righe.append(rf"\multicolumn{{{len(cols) + 1}}}{{@{{}}l}}"
                     rf"{{\itshape {titolo}\/}} \\[0.15em]")
        for x in presenti:
            celle = [_num(m.loc[x, c], 2) if c in m.columns else "" for c in cols]
            nome = (VARIANT_LABEL.get(x.partition("/")[2], _label(x))
                    if x.partition("/")[0] in SPEC_LABEL else _label(x))
            righe.append(f"{nome} & " + " & ".join(celle) + r" \\")
    return "\n".join(righe + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def _load(key: str, spec: str) -> pd.DataFrame:
    src = SOURCES[key]
    p, vecchio = src.resolve(spec=spec, w=WINDOWS_FAMILY[0], bvar=BVARS[0])
    if p is None:
        sys.exit(f"manca {PASS / src.path.format(spec=spec)}\n"
                 f"La passata non e' stata eseguita, o non e' stata copiata "
                 f"qui. Prova prima: --check")
    if vecchio:
        print(f"  ATTENZIONE: {key} letto col nome VECCHIO "
              f"({p.relative_to(PASS)}). Verifica che sia la passata giusta.")
    return pd.read_csv(p)


def _bvar_con_fed(spec: str) -> pd.DataFrame:
    """
    La tabella di famiglia dei BVAR RICALCOLATA con la Fed fra le righe.

    PERCHE' NON BASTA `metrics_bvar.csv`.  Quel CSV non ha la riga `nyfed`, e
    non per una dimenticanza: `metrics_tables.write_all` concatena la Fed al
    solo frame del DFM, perche' e' il confronto esterno del lavoro a fattori.
    La conseguenza pero' e' che le colonne `_com` delle due tabelle stanno su
    campioni DIVERSI - 1382 e 1622 coppie di qua, 1014 e 1173 di la' - e la
    colonna `rel. AR(2)`, che il lettore confronta d'istinto fra la tabella
    della famiglia e quella dei BVAR, non e' confrontabile: i punti che la Fed
    toglie sono le settimane profonde, dove i BVAR pagano di piu'.  Misurato:
    l'L-BVAR sul 2007-2021 passa da 0.712 a 0.565, il B-BVAR da 0.734 a 0.621.
    Non ribalta la conclusione della 5.3, ma la distanza che la tabella mostra
    non e' quella vera.

    INCOLLARE LA RIGA DELLA FED PRESA DAL CSV DEL DFM SAREBBE PEGGIO: sarebbe
    una riga su 1014 coppie in mezzo a righe su 1382, cioe' esattamente il
    difetto chiuso il 2026-09-06 (una riga che vive su un campione suo dentro
    una tabella che dichiara un campione comune).  Percio' si RICALCOLA, e si
    ricalcola con la stessa funzione della passata, `family_tables`, sui frame
    lunghi: nessuna metrica riscritta qui, nessun file di `output/` toccato.
    E' la stessa condotta delle figure in `REDRAW` - stessa funzione, un solo
    interruttore girato - per la stessa ragione: due implementazioni della
    stessa tabella divergono alla terza modifica.

    Il 2007-2025 NON cambia di una cifra: fuori da `NYFED_COMPARISON_PASSES` la
    Fed non entra proprio nel frame, quindi quel blocco resta a 1983 coppie con
    `n = n_com`.  Anche la colonna `free` non cambia in nessun blocco: la
    restrizione tocca solo il campione comune.

    Se i frame lunghi non ci sono (passata non copiata, o solo le tabelle) si
    ripiega sul CSV senza la Fed, dicendolo: meglio la tabella di prima che
    nessuna tabella.
    """
    try:
        from src.forecast import metrics_tables as mt
        dfm = mt.load_dfm(None)
        bvar = mt.load_bvar(None)
        fed = mt.load_nyfed(dfm)
        if bvar.empty or fed.empty:
            raise RuntimeError("frame lunghi vuoti (BVAR o Fed)")
        m, _ = mt.family_tables(pd.concat([bvar, fed], ignore_index=True),
                                "bvar", list(WINDOWS_FAMILY))
        if m.empty:
            raise RuntimeError("family_tables non ha prodotto righe")
        return m
    except Exception as exc:
        print(f"  ATTENZIONE: riga NY Fed non ricalcolabile "
              f"({type(exc).__name__}: {exc});")
        print("  la tabella BVAR uscira' SENZA la Fed, e non sara' "
              "confrontabile con quella della famiglia.")
        return _load("bvar_family", spec)


# ─── Le tabelle della 5.2.1: i test ───────────────────────────────────────────

#: Le quattro alternative al nullo gaussiano, in riga nelle tabelle dei test.
_ALTERNATIVE = ["gaussian_ar1", "student_t", "student_t_ar1",
                "student_t_ar1_shared"]

#: Etichette accorciate: in una tabella a cinque colonne di fasi non ci sta
#: "Student-$t$, AR(1) idio., shared weights" per intero.
_ALT_CORTA = {
    "gaussian":             r"Gaussian",
    "gaussian_ar1":         r"Gaussian, AR(1) idio.",
    "student_t":            r"Student-$t$",
    "student_t_ar1":        r"Student-$t$, AR(1) idio.",
    "student_t_ar1_shared": r"Student-$t$, AR(1), shared",
}

_FASI = ["forecast", "nowcast M1", "nowcast M2", "nowcast M3", "backcast"]
_FASI_CORTE = ["Forecast", "M1", "M2", "M3", "Backcast"]


def _stelle(p: float) -> str:
    """
    Tre livelli: *** all'uno per cento, ** al cinque, * al dieci.

    La terza stella NON si presenta in questa tabella (il p piu' piccolo e'
    0.033), e va bene cosi': la scala si dichiara per intero perche' il lettore
    sappia che l'assenza di *** e' un'informazione e non una convenzione
    mancante.
    """
    if p != p:
        return ""
    if p < 0.01:
        return r"\textsuperscript{***}"
    if p < 0.05:
        return r"\textsuperscript{**}"
    return r"\textsuperscript{*}" if p < 0.10 else ""


def _stat(val: float, p: float, dec: int = 2) -> str:
    """La statistica col segno meno tipografico, piu' le stelle."""
    if val != val:
        return r"\textendash{}"
    return (("$-$" if val < 0 else "") + f"{abs(val):.{dec}f}" + _stelle(p))


def tabella_test(spec: str = SPEC_DEFAULT) -> str:
    """
    I test di accuratezza predittiva: una riga per variante, una colonna per fase.

    TRE PANNELLI, E L'ORDINE E' L'ARGOMENTO.  Il primo porta il Diebold-Mariano
    modificato sul campione pre-pandemico, dove non rifiuta niente; il secondo
    il Clark-West sullo STESSO campione e sugli STESSI errori, dove rifiuta; il
    terzo il Clark-West sul campione intero, dove torna a non rifiutare.  Cosi'
    il lettore vede da se' che la differenza sta nel test e non nella finestra,
    invece di doverlo prendere per buono.

    IL QUARTO PANNELLO E' L'UNICA COPPIA NON ANNIDATA.  I primi tre confrontano
    ogni variante col GAUSSIANO, e tutte e quattro quelle coppie sono annidate:
    li' il test giusto e' il Clark-West e il Diebold-Mariano compare solo per
    mostrare quanto la correzione pesi.  Pesi per-serie contro pesi condivisi e'
    invece l'unico confronto fra due celle in cui nessuna delle due e' una
    restrizione dell'altra, quindi il Clark-West non si applica e il
    Diebold-Mariano non e' un termine di paragone ma IL test.  Senza questo
    pannello il testo prometteva un confronto che la tabella non faceva.
    Segno: negativo = i pesi condivisi hanno perso meno.

    LA RIGA DEL MCS E' UNA SOLA PERCHE' LA PROCEDURA NON ELIMINA NESSUNO.
    Quando ci si ferma al primo passo tutti i sopravvissuti ricevono lo stesso
    p-value: e' il massimo corrente del Teorema 3 di Hansen, Lunde e Nason, e
    al primo passo il massimo corrente e' uno solo.  Ripeterlo su cinque righe
    farebbe credere a cinque misure che per caso coincidono.
    """
    import numpy as np
    from src.forecast import compute_metrics as cm
    from src.forecast import predictive_accuracy as pa

    metodi = [f"{spec}/{v}" for v in VARIANT_LABEL]
    base = f"{spec}/gaussian"
    df, _, _ = cm.load_long()
    df["se"] = df["errore"] ** 2

    def blocco(win: str, quale: str):
        d, tenuti, _ = cm.window_sample(df, win)
        d = d[d["metodo"].isin(metodi)]
        val, pmcs = {}, {}
        for fase in _FASI:
            se = pa.perdite_per_trimestre(d, metodi, fase, "se")
            if se is None:
                continue
            h = pa.H_DI_FASE[fase]
            fc = pa.perdite_per_trimestre(d, metodi, fase, "nowcast_bea")
            yy = pa.perdite_per_trimestre(d, metodi, fase, "realizzato_bea")
            pmcs[fase] = float(pa.mcs(se[metodi].to_numpy(float), metodi,
                                      alpha=0.10, B=5000, ell=2,
                                      rng=np.random.default_rng(7)).iloc[0])
            for v in _ALTERNATIVE:
                m = f"{spec}/{v}"
                if quale == "dm":
                    val[(v, fase)] = pa.dm_hln(se[m].to_numpy(),
                                               se[base].to_numpy(), h)[:2]
                else:
                    val[(v, fase)] = pa.clark_west(yy[m].to_numpy(),
                                                   fc[base].to_numpy(),
                                                   fc[m].to_numpy(), h)[:2]
        return val, pmcs, len(tenuti)

    def non_annidata(win: str):
        """Pesi condivisi contro pesi per-serie: negativo = condivisi meglio."""
        d, _tenuti, _ = cm.window_sample(df, win)
        d = d[d["metodo"].isin(metodi)]
        a = f"{spec}/student_t_ar1_shared"
        b = f"{spec}/student_t_ar1"
        fuori = {}
        for fase in _FASI:
            se = pa.perdite_per_trimestre(d, metodi, fase, "se")
            if se is None:
                continue
            fuori[fase] = pa.dm_hln(se[a].to_numpy(), se[b].to_numpy(),
                                    pa.H_DI_FASE[fase])[:2]
        return fuori

    # LE FINESTRE SI NOMINANO CON GLI ANNI E COL NUMERO DI TRIMESTRI, come
    # nella tabella dell'accuratezza: "before the pandemic" e "whole sample"
    # dicono al lettore che c'e' un taglio ma non quale, e soprattutto non
    # dicono su quante osservazioni poggia la statistica che sta leggendo.
    def _titolo(win: str, n: int, test: str) -> str:
        return (r"%s, %d target quarters: %s"
                % (win.replace("-", "\\textendash{}"), n, test))

    DM = r"modified Diebold\textendash{}Mariano statistic"
    CW = r"Clark\textendash{}West statistic"

    # DUE STATISTICHE PER DUE FINESTRE, e la simmetria non e' estetica.  Il
    # Clark-West e' a UNA coda: puo' dire "la variante aiuta" e non puo' mai
    # dire "il gaussiano e' meglio".  Senza il Diebold-Mariano anche sul
    # campione intero, la tesi afferma che li' nessuno dei due e' migliore
    # mostrando l'evidenza di una direzione sola.
    b_dm = blocco("2007-2019", "dm")
    b_cw = blocco("2007-2019", "cw")
    b_dm_tot = blocco("2007-2025", "dm")
    b_tot = blocco("2007-2025", "cw")
    pannelli = [
        (_titolo("2007-2019", b_dm[2], DM), b_dm),
        (_titolo("2007-2019", b_cw[2], CW), b_cw),
        (_titolo("2007-2025", b_dm_tot[2], DM), b_dm_tot),
        (_titolo("2007-2025", b_tot[2], CW), b_tot),
    ]
    non_ann = [(r"2007\textendash{}2019", non_annidata("2007-2019")),
               (r"2007\textendash{}2025", non_annidata("2007-2025"))]
    # Il MCS NON dipende dalla statistica: e' calcolato una volta per finestra
    # sui cinque modelli insieme.  Prima stava in coda ai due pannelli del
    # Clark-West, e li' sembrava una riga DI quel test; in un pannello suo si
    # legge per quello che e', una procedura diversa che risponde a un'altra
    # domanda.  Il p del 2007-2019 e' lo stesso per i pannelli 1 e 2.
    mcs_righe = [(r"2007\textendash{}2019", b_cw[1]),
                 (r"2007\textendash{}2025", b_tot[1])]

    righe = [r"\begin{tabular}{@{}l rrrrr@{}}", r"\toprule",
             "& " + " & ".join(_FASI_CORTE) + r" \\",
             r"\cmidrule(l){2-6}"]
    for k, (titolo, dati) in enumerate(pannelli):
        val = dati[0]
        if k:
            righe.append(r"\addlinespace[0.45em]")
        righe.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\itshape {titolo}\/}} \\[0.15em]")
        for v in _ALTERNATIVE:
            celle = [_stat(*val[(v, f)]) if (v, f) in val else r"\textendash{}"
                     for f in _FASI]
            righe.append(f"{_ALT_CORTA[v]} & " + " & ".join(celle) + r" \\")

    # Titolo su UNA riga: l'inciso "the one pair that is not nested" stava
    # mandando l'intestazione a capo, e la didascalia lo dice comunque.
    righe.append(r"\addlinespace[0.45em]")
    righe.append(r"\multicolumn{6}{@{}l}{\itshape Shared against per-series "
                 r"weights: modified Diebold\textendash{}Mariano statistic\/}"
                 r" \\[0.15em]")
    for etichetta, fuori in non_ann:
        celle = [_stat(*fuori[f]) if f in fuori else r"\textendash{}"
                 for f in _FASI]
        righe.append(f"{etichetta} & " + " & ".join(celle) + r" \\")

    righe.append(r"\addlinespace[0.45em]")
    righe.append(r"\multicolumn{6}{@{}l}{\itshape Model confidence set on all "
                 r"five specifications: $p$-value\/} \\[0.15em]")
    for etichetta, pmcs in mcs_righe:
        celle = [f"{pmcs[f]:.2f}".lstrip("0") if f in pmcs
                 else r"\textendash{}" for f in _FASI]
        righe.append(f"{etichetta} & " + " & ".join(celle) + r" \\")

    return "\n".join(righe + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def tabella_lr(spec: str = SPEC_DEFAULT) -> str:
    """
    L'adattamento dentro campione, e il rapporto di verosimiglianza fra celle.

    DUE GRUPPI IN UNA TABELLA SOLA, con le colonne che cambiano significato a
    meta': sopra l'adattamento di ciascuna cella, sotto il confronto fra celle
    annidate.  E' lo stesso espediente di `tabella_estremi`, che sopra la banda
    ha punti percentuali e sotto delle quote, e per la stessa ragione: due
    tabelle da cinque righe si leggono peggio di una da dieci.

    LA COLONNA DEL VALORE CRITICO NON E' DECORAZIONE.  Con un LR nell'ordine
    delle migliaia il p-value stampato sarebbe una fila di zeri, che non dice
    niente; il critico accanto alla statistica dice quanto il rifiuto sia
    schiacciante, ed e' anche il posto dove si legge che la nulla al bordo e'
    MENO severa di un chi-quadro, cioe' che la scelta della distribuzione non
    e' stata fatta in nostro favore.
    """
    from scipy import stats as _st
    from src.em import likelihood_ratio as lrm

    adatt, conf, _n_oss = lrm.tabella(spec, radice=str(ROOT))
    a = adatt.set_index("variante")

    righe = [r"\begin{tabular}{@{}l rrr@{}}", r"\toprule",
             r"& Log-likelihood & Parameters & \textsc{bic} \\",
             r"\cmidrule(l){2-4}"]
    for v in VARIANT_LABEL:
        if v not in a.index:
            continue
        righe.append(f"{_ALT_CORTA[v]} & {_num(a.loc[v, 'ELBO'], 1)} & "
                     f"{int(a.loc[v, 'n_par'])} & {_num(a.loc[v, 'BIC'], 0)}"
                     r" \\")
    righe += [r"\addlinespace[0.45em]",
              r"\multicolumn{4}{@{}l}{\itshape Likelihood ratio against the "
              r"nested restriction\/} \\[0.15em]",
              r"& \textsc{lr} & Restrictions & Critical, 5\% \\"]
    for _, r in conf.iterrows():
        gradi = int(r["gradi"])
        crit = (lrm.chernoff_crit(gradi, 0.05)
                if str(r["nulla"]).startswith("Chernoff")
                else float(_st.chi2.ppf(0.95, gradi)))
        nome = (f"{_ALT_CORTA[r['piccolo']]} $\\rightarrow$ "
                f"{_ALT_CORTA[r['grande']]}")
        righe.append(f"{nome} & {_num(r['LR'], 1)} & {gradi} & "
                     f"{_num(crit, 1)}" r" \\")
    return "\n".join(righe + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def genera(spec: str) -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    (OUT_TAB / "tab_rmse_fed_overlap.tex").write_text(
        tabella_famiglia(_load("dfm_family", spec), spec), encoding="utf-8")
    print(f"scritto  {OUT_TAB / 'tab_rmse_fed_overlap.tex'}")

    (OUT_TAB / "tab_rmse_matrix.tex").write_text(
        tabella_matrice(_load("matrix_rel", spec)), encoding="utf-8")
    print(f"scritto  {OUT_TAB / 'tab_rmse_matrix.tex'}")

    (OUT_TAB / "tab_rmse_bvar.tex").write_text(
        tabella_famiglia(_bvar_con_fed(spec), "bvar"), encoding="utf-8")
    print(f"scritto  {OUT_TAB / 'tab_rmse_bvar.tex'}")

    (OUT_TAB / "tab_estremi.tex").write_text(
        tabella_estremi(_load("extremes", spec)), encoding="utf-8")
    print(f"scritto  {OUT_TAB / 'tab_estremi.tex'}")

    (OUT_TAB / "tab_lr.tex").write_text(tabella_lr(spec), encoding="utf-8")
    print(f"scritto  {OUT_TAB / 'tab_lr.tex'}")

    (OUT_TAB / "tab_test.tex").write_text(tabella_test(spec), encoding="utf-8")
    print(f"scritto  {OUT_TAB / 'tab_test.tex'}")

    (OUT_TAB / "tab_estremi_full.tex").write_text(
        tabella_estremi(_load("extremes", spec), tutte_le_spec=True),
        encoding="utf-8")
    print(f"scritto  {OUT_TAB / 'tab_estremi_full.tex'}")

    for key in FIGURES_WANTED:
        src, dst = FIGURES[key]
        if key in REDRAW:
            if _rigenera_senza_nota(key, spec, OUT_FIG / dst):
                continue
            print("  ripiego sulla copia: la figura avra' la nota.")
        p = PASS / src.format(spec=spec)
        if not p.exists():
            print(f"  ATTENZIONE: manca {p} — figura non copiata")
            continue
        shutil.copyfile(p, OUT_FIG / dst)
        print(f"copiato  {OUT_FIG / dst}")

    _ritaglia_per_la_tesi()


def _ritaglia_per_la_tesi() -> None:
    """
    Il passo di `figure_tesi.py`, chiamato QUI e non lasciato all'utente.

    PERCHE' E' AUTOMATICO.  Le figure appena copiate sono quelle della
    passata: portano il titolo dentro l'immagine e, alcune, la nota in fondo.
    In tesi quelle due cose stanno nella didascalia.  Finche' il ritaglio era
    un secondo comando da ricordare, bastava rigenerare UNA tabella per
    riportare in tesi otto figure col titolo, e la cosa non si vede nel log:
    lo script dice "copiato" ed e' andato tutto bene dal suo punto di vista.
    E' successo due volte l'08-09-2026, la seconda mentre si rigenerava una
    tabella che con le figure non c'entrava niente.

    Chiamarlo da qui rende l'ordine dei due passi una proprieta' del codice
    invece che una cosa da ricordare.  `figure_tesi.py` resta lanciabile da
    solo, e `--check` continua a servire per ispezionare senza scrivere.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import figure_tesi
    except Exception as exc:                       # non deve fermare le tabelle
        print(f"  ATTENZIONE: ritaglio non eseguito ({type(exc).__name__}: "
              f"{exc}); lanciare a mano notebooks/figure_tesi.py")
        return
    print()
    for nome, cfg in figure_tesi.FIGURES.items():
        figure_tesi.lavora(nome, cfg, False)


def _rigenera_senza_nota(key: str, spec: str, dst: Path) -> bool:
    """La figura RMSE con la nota spenta, per la tesi.  `True` se e' riuscito.

    Chiama la STESSA funzione della passata (`figure_rmse_by_horizon`) con
    `note=False`: nessun codice di figura duplicato qui, solo l'interruttore.
    Ritorna `False` senza sollevare se qualcosa manca, cosi' il chiamante
    ripiega sulla copia invece di lasciare la sezione senza figura.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from src import output_layout as layout
        from src.forecast import compare_nyfed as cn
    except ImportError as exc:
        print(f"  {key}: import fallito ({exc})")
        return False

    finestra = REDRAW[key]
    try:
        mine_all, _, _ = cn.load_mine(None)
        mine = layout.slice_window(mine_all, finestra, column="as_of")
        if mine.empty:
            print(f"  {key}: nessuna riga nella finestra {finestra}")
            return False
        qs = sorted(mine["target_quarter"].unique())
        ph, sample = cn.horizon_panel(mine, qs, spec)
        if not sample or ph.empty:
            print(f"  {key}: copertura incompleta su {finestra}")
            return False
        cn.figure_rmse_by_horizon(ph, sample, spec, str(dst), note=False)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"  {key}: rigenerazione fallita ({type(exc).__name__}: {exc})")
        return False
    print(f"rigenerato senza nota  {dst}  ({len(sample)} trimestri, {finestra})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--check", action="store_true",
                    help="elenca cosa c'e' e cosa manca, senza generare")
    ap.add_argument("--spec", default=SPEC_DEFAULT, choices=SPECS,
                    help=f"la famiglia che guida le figure (default: {SPEC_DEFAULT})")
    a = ap.parse_args()
    if a.check:
        sys.exit(1 if check(a.spec) else 0)
    genera(a.spec)


if __name__ == "__main__":
    main()
