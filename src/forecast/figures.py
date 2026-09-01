"""
src/forecast/figures.py

LA FIGURA DELLE TRAIETTORIE (impianto Cascaldi-Garcia 8a).

Lettore puro: legge il CSV lungo di `weekly_nowcast.py` e disegna.  Non stima
niente.

IL COLORE E' IL NUMERO DEL TRIMESTRE
------------------------------------
Q1 ha sempre lo stesso colore, in ogni figura e in ogni finestra; idem Q2, Q3,
Q4.  Quattro colori per riquadro invece di uno per trimestre disegnato, e la
legenda li dichiara.  Le due famiglie (`family='dfm'` / `'bvar'`) hanno
tavolozze diverse, cosi' i due alberi di output non si confondono a vista.
Vedi `_QUARTER_COLORS`.

COSA MOSTRA (e cosa non mostra)
-------------------------------
Un pannello per cella (spec x variante).  Sull'asse x il tempo di calendario,
cioe' le date `as_of` a risoluzione settimanale; sull'asse y il PIL in tasso
annualizzato.  Una linea continua colorata per ogni trimestre target, che segue
l'evoluzione del suo nowcast man mano che i dati arrivano; le linee di trimestri
diversi si sovrappongono nel tempo, perche' a ogni data il modello sta prevedendo
il trimestre precedente, quello corrente e il prossimo.

Le linee sono a GRADINI: piatte fra un rilascio e l'altro, saltano quando esce
un dato.  Non vanno lisciate — il gradino e' la firma della frequenza
settimanale, ed e' l'unica cosa che a frequenza mensile non si vedrebbe.

I pallini NON appartengono alle linee.  Sono il dato pubblicato: posizionati
alla data di rilascio del PIL letta dal calendario e all'altezza del valore
realizzato.  La distanza VERTICALE fra la fine della linea e il pallino e'
l'errore di nowcast finale.

LA LINEA CORRE FINO AL RILASCIO; LO STACCO RESIDUO E' UNA SOLA SETTIMANA
-----------------------------------------------------------------------
La linea di un trimestre resta viva finche' il suo PIL non esce: l'ultima
as_of disegnata e' l'ultimo venerdi' STRETTAMENTE PRECEDENTE a
`gdp_release_date(q) = fine_trimestre + 28`.  Include quindi il BACKCAST — le
settimane in cui il trimestre e' gia' chiuso ma il PIL ufficiale non e' ancora
uscito — che e' la fase PIU' ACCURATA del nowcast: arrivano occupazione, ISM,
ecc. che raffinano la stima (2008Q4 si muove da -3.01 a -3.47 fra il 2 e il 23
gennaio, avvicinandosi al realizzato).

Il pallino sta sul rilascio; il vuoto orizzontale fra la fine della linea e il
pallino e' solo il residuo fra l'ultimo venerdi' utile e la data di
pubblicazione — ~1 settimana, non quattro.  La distanza VERTICALE fra
fine-linea e pallino resta l'errore di nowcast finale.  Questa e' la stessa
regola che governa `targets_in_flight`: un trimestre e' in volo per ogni
as_of < gdp_release_date(q).  Chi vuole la linea tagliata a fine
trimestre (senza backcast) usa `--nascondi-backcast`.

E' una figura di PUNTO, non di densita': nessuna banda di quantili.

FINESTRE E CARTELLE STANNO IN `src/output_layout.py`
----------------------------------------------------
`--window 2014-2016` ritaglia il CSV sull'intervallo di date di quella
finestra e nomina la figura con quel nome; le cartelle di destinazione
(`dfm/<spec>/<variant>/`) vengono dallo stesso modulo.  Qui non c'e' nessuna
data e nessun percorso cablato.

LA SCALA E' PER FINESTRA, NON GLOBALE
-------------------------------------
Ogni figura si scala sui dati che disegna: assi (x e y) dai valori effettivi
piu' un margine, legenda posizionata verificando che non copra le curve, e
niente range fisso ereditato da un'altra finestra.  Il range fisso `(-15,+6)`
c'era, ed era tarato sulla Grande Recessione: sul 2024-2025 avrebbe schiacciato
la serie in una striscia.  Chi vuole due celle sulla stessa scala passa
`--ylim MIN MAX`.

  --style trajectories  un pannello per cella (default)
  --style compare       un trimestre, tutti i metodi sovrapposti
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src import output_layout as layout
from src.forecast.release_calendar import quarter_end

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Stile accademico ─────────────────────────────────────────────────────────
# I colori sono l'ordine MATLAB, che e' quello dell'originale.
_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE",
           "#A2142F"]

# UN COLORE PER NUMERO DI TRIMESTRE, NON PER ORDINE DI APPARIZIONE
# ----------------------------------------------------------------
# Prima i colori si assegnavano scorrendo i trimestri disegnati: sul 2014-2016
# sono quattordici, il ciclo di sei si ripeteva due volte e mezzo, e la stessa
# tinta finiva su trimestri diversi nello stesso riquadro.  Con quattordici
# linee sovrapposte quello non e' un codice colore, e' rumore.
#
# Qui il colore dipende SOLO dal numero del trimestre: Q1 e' sempre lo stesso
# colore in ogni figura, in ogni finestra, in ogni cella.  I colori per riquadro
# scendono da sei a quattro e la legenda puo' dichiararli, quindi il lettore
# legge "questa e' una linea di Q3" senza inseguire l'etichetta del pallino.
#
# Due famiglie, una per albero di output, cosi' una figura DFM e una BVAR non si
# scambiano per la stessa cosa a colpo d'occhio.  Le tinte disponibili sono
# sette e ne servono quattro per famiglia: il viola e' l'unica ripetizione, ed
# e' su Q4 in entrambe.
_QUARTER_COLORS: dict[str, dict[int, str]] = {
    "dfm":  {1: "#0072BD", 2: "#D95319", 3: "#77AC30", 4: "#7E2F8E"},
    "bvar": {1: "#4DBEEE", 2: "#EDB120", 3: "#A2142F", 4: "#7E2F8E"},
}

_FIGSIZE = (10.0, 6.0)          # ~1.67:1, il formato orizzontale dell'originale
_YLIM = None                    # scala AUTOMATICA sulla finestra: vedi _autoscale
_LINEWIDTH = 2.5
_DOT_SIZE = 55
_MIN_POINTS = 2                 # traiettorie con un solo punto non dicono niente

_BENCHMARK_SPEC = "benchmark"

# Margine attorno ai dati, in frazione dell'escursione: abbastanza da non
# tagliare un picco o un pallino, poco abbastanza da non lasciare bordi vuoti.
_PAD_Y = 0.08
_PAD_X = 0.04
# Spazio in piu' a destra per le etichette dei pallini ("08Q4"), che sono
# disegnate con un offset in punti e non rientrano nel calcolo dei limiti.
_PAD_X_LABEL = 0.05

# Le etichette dei pallini cadono spesso in mezzo alle traiettorie (nel 2008 i
# rilasci stanno dentro il grafico, non ai bordi).  Un contorno bianco sottile
# le tiene leggibili SENZA spostarle: muoverle vorrebbe dire staccarle dal
# proprio pallino, che e' l'unica cosa che dice a quale trimestre si riferiscono.
_LABEL_HALO = [pe.withStroke(linewidth=2.6, foreground="white")]


def _apply_axes_style(ax: plt.Axes) -> None:
    """Riquadro chiuso, tick interni sui quattro lati, niente griglia."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("black")
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=9, width=0.8)
    ax.grid(False)
    ax.set_facecolor("white")


def _serif() -> dict:
    return {"font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "mathtext.fontset": "dejavuserif"}


# ─── Scala e legenda: ogni figura inquadra la PROPRIA finestra ────────────────

def _autoscale(ax: plt.Axes, xs: list, ys: list) -> None:
    """
    Limiti presi dai dati EFFETTIVAMENTE disegnati in questa figura.

    Il vecchio `_YLIM=(-15,+6)` era tarato sul 2008-2010: applicato al
    2024-2025 (crescita fra +1 e +3) schiacciava la serie in una striscia
    alta un decimo del riquadro.  Qui ogni finestra si scala su se stessa.

    `xs` include le date di RILASCIO, non solo le `as_of`: il pallino del PIL
    pubblicato cade dopo l'ultimo venerdi' disegnato e deve restare dentro il
    riquadro, non a cavallo del bordo.  A destra si aggiunge un margine per le
    etichette dei pallini, che sono disegnate con un offset in punti.
    """
    if ys:
        lo, hi = float(np.nanmin(ys)), float(np.nanmax(ys))
        span = hi - lo
        pad = (span * _PAD_Y) if span > 0 else max(abs(hi) * 0.1, 0.5)
        ax.set_ylim(lo - pad, hi + pad)
    if xs:
        x0, x1 = min(xs), max(xs)
        span = (x1 - x0) or pd.Timedelta(days=7)
        ax.set_xlim(x0 - span * _PAD_X, x1 + span * (_PAD_X + _PAD_X_LABEL))


def _data_boxes(ax: plt.Axes) -> list:
    """I riquadri (in coordinate display) di tutto cio' che e' stato disegnato.

    LE SCRITTE CONTANO QUANTO LE LINEE.  Per un po' qui c'erano solo
    `ax.get_lines()`, e il risultato era una legenda che si dichiarava «libera»
    stando sopra a del testo: nella RMSE dei BVAR copriva l'etichetta della
    fase *forecast*, nelle traiettorie poteva coprire le targhette dei
    trimestri.  Il controllo diceva il vero su cio' che guardava — guardava
    troppo poco.  Le annotazioni sono `Text` come le altre e stanno in
    `ax.texts`, quindi entrano da qui.
    """
    boxes = []
    for ln in ax.get_lines():
        # `make_compare` disegna ancora una linea dello zero tratteggiata: non
        # e' un dato, e non deve spingere via la legenda.  Nelle traiettorie
        # quella linea non c'e' piu' e il filtro semplicemente non trova nulla.
        if ln.get_linestyle() == "--" and ln.get_color() == "black":
            continue
        xy = ln.get_xydata()
        if len(xy) == 0:
            continue
        pts = ax.transData.transform(xy)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts):
            boxes.append((pts[:, 0].min(), pts[:, 1].min(),
                          pts[:, 0].max(), pts[:, 1].max()))
    for txt in ax.texts:
        if not txt.get_visible() or not str(txt.get_text()).strip():
            continue
        try:                       # serve il renderer: c'e', il chiamante
            bb = txt.get_window_extent()   # ha gia' fatto `fig.canvas.draw()`
        except (RuntimeError, ValueError, AttributeError):
            continue
        if np.isfinite([bb.x0, bb.y0, bb.x1, bb.y1]).all():
            boxes.append((bb.x0, bb.y0, bb.x1, bb.y1))
    return boxes


def _legend_overlaps(fig, ax: plt.Axes, leg) -> bool:
    """La legenda tocca qualcosa di disegnato?  Verifica, non speranza."""
    fig.canvas.draw()
    lb = leg.get_window_extent()
    for x0, y0, x1, y1 in _data_boxes(ax):
        if lb.x1 > x0 and lb.x0 < x1 and lb.y1 > y0 and lb.y0 < y1:
            return True
    return False


def _place_legend(fig, ax: plt.Axes, handles: list, ncol: int = 1,
                  fallback_anchor: tuple[float, float] = (0.5, -0.09),
                  fallback_ncol: int | None = None):
    """
    Mette la legenda dove NON copre i dati.

    Si prova `best` (che minimizza la sovrapposizione con gli artisti gia'
    disegnati), poi gli angoli a mano; se nessuna posizione e' libera —
    succede quando le traiettorie riempiono il riquadro — la legenda esce
    SOTTO il riquadro, dove non puo' coprire niente per costruzione.

    Il ripiego e' parametrico perche' "sotto il riquadro" non e' lo stesso
    posto in tutte le figure: la RMSE per orizzonte ha un asse secondario a
    -0.13, e una legenda ancorata a -0.09 gli finirebbe sopra.  Chi ha roba
    sotto l'asse passa il proprio `fallback_anchor`.  `fallback_ncol` limita
    le colonne: con nove voci su una riga sola la legenda sfora a destra e
    l'ultima si taglia.

    PRIMA DI USCIRE, SI PROVA A STRINGERLA.  Una legenda a `ncol` colonne e'
    larga: nelle finestre fitte (le traiettorie 2024-2025) nessuno dei cinque
    angoli la ospita, e finiva sotto il riquadro mentre in tutte le altre
    figure sta dentro.  Ma lo spazio libero spesso c'e' — e' solo piu' stretto
    di quanto la legenda sia larga.  Quindi si riprova con meno colonne, dalla
    piu' larga alla piu' stretta: la disposizione richiesta ha la precedenza, e
    si esce sotto il riquadro solo se davvero non entra da nessuna parte.
    """
    layouts = list(dict.fromkeys([ncol, 2, 1]))
    for nc in layouts:
        for loc in ("best", "upper left", "upper right",
                    "lower left", "lower right"):
            leg = ax.legend(handles=handles, loc=loc, frameon=False,
                            fontsize=8.5, ncol=nc)
            if not _legend_overlaps(fig, ax, leg):
                return leg
            leg.remove()
    return ax.legend(handles=handles, loc="upper center", frameon=False,
                     fontsize=8.5,
                     ncol=fallback_ncol or max(ncol, len(handles)),
                     bbox_to_anchor=fallback_anchor)


# ─── Lettura ──────────────────────────────────────────────────────────────────

def discover_csvs(paths: list[str] | None = None) -> list[str]:
    """
    I CSV da disegnare: quelli dati, o TUTTI quelli presenti.

    Tutti, e non piu' il piu' recente per data di modifica.  Il piu' recente
    era corretto finche' una passata scriveva un CSV solo, con dentro tutte le
    celle (`--all-specs --all-variants` in un processo).  Da quando ogni cella
    e' un processo suo — perche' e' l'unita' che si parallelizza — in
    `dfm/csv/` ci sono QUINDICI file, uno per cella, e prenderne uno solo
    voleva dire disegnare una cella su quindici: in silenzio, senza errore,
    il file c'era e le figure uscivano — quattro invece di sessanta.

    E' la stessa regola che `compute_metrics.discover_csvs` applica gia' alle
    tabelle: le figure leggevano l'albero in un modo e le metriche in un altro.
    """
    if paths:
        return list(paths)
    d = layout.dfm_csv_dir()
    found = sorted(glob.glob(os.path.join(d, "weekly_nowcast_*.csv")))
    if not found:
        raise FileNotFoundError(
            f"Nessun CSV in {d}.\n"
            f"Generane uno con:  python scripts/run_dfm.py "
            f"--spec diag3 --variant gaussian"
        )
    return found


def load(csv_path: "str | list[str] | None" = None) -> pd.DataFrame:
    """
    Le righe di tutti i CSV richiesti, in un frame solo.

    `csv_path` accetta un percorso, una lista di percorsi, o niente (tutti).
    """
    if isinstance(csv_path, str):
        csv_path = [csv_path]
    df = pd.concat([pd.read_csv(p) for p in discover_csvs(csv_path)],
                   ignore_index=True)
    df["target_quarter"] = df["target_quarter"].astype(str)
    df["as_of_dt"] = pd.to_datetime(df["as_of"])
    df["release_dt"] = pd.to_datetime(df["gdp_release_date"])
    df["cella"] = np.where(df["spec"] == _BENCHMARK_SPEC,
                           df["variant"], df["spec"] + "/" + df["variant"])
    return df


def _quarter_key(q: str) -> tuple[int, int]:
    y, n = q.upper().split("Q")
    return int(y), int(n)


def _short(q: str) -> str:
    """'2008Q4' -> '08Q4', l'etichetta accanto al pallino."""
    y, n = q.upper().split("Q")
    return y[2:] + "Q" + n


def _quarter_number(q: str) -> int:
    """'2008Q4' -> 4.  E' questo, non l'ordine, a decidere il colore."""
    return int(q.upper().split("Q")[1])


# COME SI SCRIVE IL NOME DELLA CELLA NEL TITOLO
# ---------------------------------------------
# Nel CSV la cella e' un IDENTIFICATIVO — 'bbvar/-', 'fed_overlap/student_t_ar1'
# — fatto per essere ordinato e confrontato, non letto.  In cima a una figura ci
# va il nome, non la chiave: niente underscore, niente barra, niente trattino
# vuoto al posto della variante.  La traduzione sta qui e solo qui.
_CELL_TITLES = {
    "bbvar/-": "B-BVAR", "cbvar/authors": "C-BVAR",
    "lbvar/-": "L-BVAR", "qbvar/-": "Q-BVAR",
    "ar2": "AR(2)", "mean": "Mean",
}

#: Le celle DFM si compongono: <spec>-<variante>.  `fed_overlap` diventa
#: `FedOverlap` senza trattino interno di proposito — con `Fed-Overlap` un
#: titolo come "Fed-Overlap-Student-t AR(1)" avrebbe tre trattini e nessuno
#: capirebbe piu' quale separa che cosa.
_SPEC_TITLES = {"diag3": "Diag3", "diag4": "Diag4",
                "fed_overlap": "FedOverlap",
                # La FAMIGLIA, non un modello: `_as_family` riscrive i metodi
                # in `bvar/<modello>`, e senza questa riga un titolo costruito
                # sulla chiave di famiglia usciva `bvar` minuscolo.
                "bvar": "BVAR"}
_VARIANT_TITLES = {
    "gaussian":             "Gaussian",
    "gaussian_ar1":         "Gaussian AR(1)",
    "student_t":            "Student-t",
    "student_t_ar1":        "Student-t AR(1)",
    # `_shared` = un peso solo condiviso da tutte le serie, contro i pesi
    # per-serie di `student_t_ar1`.  Si tiene la parola dell'identificativo
    # cosi' chi legge la figura e la tabella non deve tradurre.
    "student_t_ar1_shared": "Student-t AR(1) Shared",
}


#: Gli stessi nomi, ma indicizzati sul MODELLO NUDO.  Nelle figure per
#: orizzonte il metodo arriva come `bbvar` e non come `bbvar/-`, perche' li' i
#: quattro BVAR sono aggregati per famiglia.  Si deriva dalla tabella qui sopra
#: invece di riscriverla: due elenchi degli stessi quattro nomi divergono.
_MODEL_TITLES = {k.split("/")[0]: v for k, v in _CELL_TITLES.items() if "/" in k}


def _pretty_cell(cell: str) -> str:
    """'fed_overlap/student_t_ar1' -> 'FedOverlap-Student-t AR(1)'."""
    if cell in _CELL_TITLES:
        return _CELL_TITLES[cell]
    if "/" in cell:
        spec, variant = cell.split("/", 1)
        if spec in _SPEC_TITLES and variant in _VARIANT_TITLES:
            return f"{_SPEC_TITLES[spec]}-{_VARIANT_TITLES[variant]}"
    # Cella non prevista (una spec nuova, un modello aggiunto): meglio
    # l'identificativo grezzo in figura che un KeyError a fine passata.
    return cell


def pretty_spec(spec: str) -> str:
    """Il nome della SPEC per il titolo di una figura per orizzonte.

        'diag4'                      -> 'Diag4'
        'qbvar/cbvar/bbvar/lbvar'    -> 'Q-BVAR/C-BVAR/B-BVAR/L-BVAR'

    La barra separa i modelli disegnati insieme e resta: e' come si legge in
    fretta quali quattro curve ci sono.
    """
    parts = [p for p in str(spec).split("/") if p]
    if not parts:
        return str(spec)
    if all(p in _MODEL_TITLES for p in parts):
        return "/".join(_MODEL_TITLES[p] for p in parts)
    return "/".join(_SPEC_TITLES.get(p, p) for p in parts)


def pretty_series(metodo: str) -> str:
    """Il nome di UNA curva in legenda.

        'bbvar'  o  'bbvar/-'  o  'bvar/bbvar'   -> 'B-BVAR'
        'diag4/student_t'                        -> 'Student-t'

    Nel caso DFM si tiene la sola VARIANTE, non `Diag4-Student-t`: la spec sta
    gia' nel titolo, e ripeterla su ogni voce allunga la legenda senza
    distinguere niente.

    ⚠️  `bvar/bbvar` non e' una svista: nel pannello per orizzonte i quattro
    BVAR sono aggregati per FAMIGLIA, quindi il metodo porta `bvar/` davanti
    invece della variante.  La coda va cercata in tutte e due le tabelle — con
    la sola `_VARIANT_TITLES` tornava `bbvar` minuscolo, ed e' esattamente
    l'errore che era finito in figura.
    """
    m = str(metodo)
    if m in _CELL_TITLES:
        return _CELL_TITLES[m]
    if m in _MODEL_TITLES:
        return _MODEL_TITLES[m]
    variant = m.split("/", 1)[-1]
    if variant in _MODEL_TITLES:
        return _MODEL_TITLES[variant]
    return _VARIANT_TITLES.get(variant, variant)


def _period(df: pd.DataFrame) -> str:
    m = df["as_of_dt"].dt.strftime("%Y-%m")
    return f"{m.min()}_{m.max()}"


def _line_rows(rows: pd.DataFrame, q: str, nascondi_backcast: bool) -> pd.DataFrame:
    """
    Le righe da disegnare come linea di un trimestre.

    Default: la linea corre fino al RILASCIO escluso, cioe' per ogni
    as_of < gdp_release_date(q) — backcast incluso, la fase piu' accurata del
    nowcast.  E' la stessa soglia con cui `targets_in_flight` ha generato le
    righe, quindi in pratica sono tutte le righe della cella; `release_dt`
    (colonna `gdp_release_date` del CSV) rende la regola esplicita e robusta.

    `nascondi_backcast=True` ritaglia invece a fine trimestre (con lo stacco
    di ~4 settimane fino al pallino).
    """
    if nascondi_backcast:
        return rows[rows["as_of_dt"] <= quarter_end(q)]
    release = pd.Timestamp(rows["release_dt"].iloc[0])
    return rows[rows["as_of_dt"] < release] if pd.notna(release) else rows


# ─── La figura delle traiettorie (stile Cascaldi-Garcia 8a) ───────────────────

def _draw_trajectories(ax: plt.Axes, df_cell: pd.DataFrame,
                       colors: dict[str, str],
                       ylim: tuple[float, float] | None,
                       nascondi_backcast: bool = False) -> None:
    """Un pannello: le traiettorie settimanali e i pallini dei rilasci."""
    quarters = sorted(df_cell["target_quarter"].unique(), key=_quarter_key)
    xs: list = []
    ys: list = []

    for q in quarters:
        rows = df_cell[df_cell["target_quarter"] == q].sort_values("as_of_dt")
        if len(rows) < _MIN_POINTS:
            continue
        color = colors[q]

        # Fin dove corre la linea: di default fino al RILASCIO escluso (backcast
        # incluso), col taglio a fine trimestre solo se richiesto.  E' scelta
        # GRAFICA in entrambi i casi; il pallino resta sulla data di rilascio.
        linea = _line_rows(rows, q, nascondi_backcast)

        # La traiettoria. Nessun marker: i punti sono le settimane, e sono
        # abbastanza fitti da leggersi come una linea a gradini.
        if len(linea) >= _MIN_POINTS:
            ax.plot(linea["as_of_dt"], linea["nowcast_bea"],
                    color=color, linewidth=_LINEWIDTH,
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)
            xs += [linea["as_of_dt"].min(), linea["as_of_dt"].max()]
            ys += [linea["nowcast_bea"].min(), linea["nowcast_bea"].max()]

        # Il pallino: dato pubblicato, alla data di rilascio dal calendario.
        realised = rows["realizzato_bea"].iloc[0]
        release = rows["release_dt"].iloc[0]
        if pd.notna(realised) and pd.notna(release):
            ax.plot([release], [realised], marker="o", markersize=np.sqrt(_DOT_SIZE),
                    color=color, linestyle="none", zorder=5)
            ax.annotate(_short(q), (release, realised),
                        textcoords="offset points", xytext=(6, 0),
                        ha="left", va="center", fontsize=8, color=color,
                        annotation_clip=False, zorder=6,
                        path_effects=_LABEL_HALO)
            xs.append(release)
            ys.append(realised)

    if ylim is not None:
        ax.set_ylim(*ylim)
        _autoscale(ax, xs, [])          # la x si scala comunque sulla finestra
    else:
        _autoscale(ax, xs, ys)

    # NIENTE LINEA DELLO ZERO.  C'era, tratteggiata, quando lo zero cadeva nel
    # riquadro.  Non aggiunge niente che l'asse y non dica gia' — il tick dello
    # zero c'e' comunque — e attraversa le traiettorie proprio nelle finestre
    # dove sono piu' fitte.

    _apply_axes_style(ax)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))


def _legend(fig, ax: plt.Axes, palette: dict[int, str],
            quarter_numbers: list[int]) -> None:
    """
    Senza riquadro, mai sopra ai dati, e con il codice colore dichiarato.

    Le voci di trimestre non sono decorazione: da quando il colore dipende dal
    NUMERO del trimestre e non dall'ordine, la legenda e' l'unico posto dove
    quella regola e' scritta.  Si elencano solo i numeri effettivamente
    presenti nella finestra, cosi' una finestra corta non dichiara colori che
    non ha disegnato.

    La posizione non e' piu' un default fisso.  Con `lower right` cablato, sui
    trimestri di crisi la legenda copriva proprio il pallino del PIL
    realizzato: il realizzato sta in basso, la data di rilascio e' l'ultimo
    punto a destra — cioe' l'angolo occupato dalla legenda, e nei trimestri
    che si guardano di piu'.  Ora la posizione si sceglie verificando la
    sovrapposizione; vedi `_place_legend`.
    """
    handles = [
        Line2D([0], [0], color="black", linewidth=_LINEWIDTH,
               label="Weekly forecast path"),
        Line2D([0], [0], color="black", marker="o", linestyle="none",
               markersize=np.sqrt(_DOT_SIZE), label="First-release GDP"),
        *[Line2D([0], [0], color=palette[n], linewidth=_LINEWIDTH,
                 label=f"Q{n} target")
          for n in quarter_numbers],
    ]
    _place_legend(fig, ax, handles, ncol=3)


def make_trajectories(df: pd.DataFrame, output_dir: str,
                      ylim: tuple[float, float] | None = _YLIM,
                      cells: list[str] | None = None,
                      nascondi_backcast: bool = False,
                      dir_for_cell=None,
                      window_label: str | None = None,
                      family: str = "dfm") -> list[str]:
    """
    Una figura per cella.  Restituisce i percorsi scritti.

    `dir_for_cell(cell) -> str` decide la sottocartella di ogni cella.  Il
    default replica il vecchio albero (`<output_dir>/<spec>/`); i due driver
    passano il proprio instradamento verso `output_layout`, cosi' la regola
    di posizionamento sta in un posto solo e questa funzione resta un
    disegnatore.

    `window_label` (es. "2014-2016") entra nel nome del file al posto del
    periodo dedotto dai dati: e' il nome della finestra, non quello che il
    CSV si e' trovato dentro.

    `family` sceglie la tavolozza per numero di trimestre (`_QUARTER_COLORS`):
    'dfm' o 'bvar'.  Il nome del file NON contiene la cella — ogni figura sta
    gia' nella cartella della propria cella, e ripetere il nome li' dentro non
    distingue niente.
    """
    if family not in _QUARTER_COLORS:
        raise ValueError(f"family {family!r}: attesa una di "
                         f"{tuple(_QUARTER_COLORS)}.")
    os.makedirs(output_dir, exist_ok=True)
    period = window_label or _period(df)
    palette = _QUARTER_COLORS[family]

    todo = cells if cells else sorted(df["cella"].unique())
    written: list[str] = []

    with plt.rc_context(_serif()):
        for cell in todo:
            sub = df[df["cella"] == cell]
            if sub.empty:
                print(f"  [salto] {cell}: nessuna riga")
                continue

            # I trimestri sono quelli DI QUESTA cella: una cella che si ferma
            # prima non deve dichiarare in legenda un colore che non disegna,
            # ne' annunciare nel titolo un intervallo che non copre.
            quarters = sorted(sub["target_quarter"].unique(), key=_quarter_key)
            colors = {q: palette[_quarter_number(q)] for q in quarters}
            numeri = sorted({_quarter_number(q) for q in quarters})

            fig, ax = plt.subplots(figsize=_FIGSIZE)
            fig.patch.set_facecolor("white")
            _draw_trajectories(ax, sub, colors, ylim, nascondi_backcast)
            _legend(fig, ax, palette, numeri)

            ax.set_ylabel("Annualised growth rate (%)", fontsize=10)
            ax.set_xlabel("Vintage date", fontsize=10)
            ax.set_title(f"GDP Forecast Evolution — {_pretty_cell(cell)}\n"
                         f"Target Quarters {quarters[0]}–{quarters[-1]}",
                         fontsize=11.5, pad=10)
            fig.tight_layout()

            # Dove va la figura: lo decide il chiamante via `dir_for_cell`.
            # Default = vecchio albero, una sottocartella per spec.
            if dir_for_cell is not None:
                cell_dir = dir_for_cell(cell)
            else:
                sub_dir = cell.split("/")[0] if "/" in cell else _BENCHMARK_SPEC
                cell_dir = os.path.join(output_dir, sub_dir)
            os.makedirs(cell_dir, exist_ok=True)
            fname = f"Trajectories_{period}.png"
            path = os.path.join(cell_dir, fname)
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            written.append(path)
            print(f"  scritto: {path}")

    return written


# ─── Instradamento nell'albero di output ──────────────────────────────────────

def _dfm_dir_for_cell(cell: str) -> str:
    """
    'diag3/student_t' -> output/forecast_weekly/dfm/diag3/student_t/

    I benchmark non hanno spec: `cella` e' il solo nome ('ar2', 'mean') e la
    figura non serve — sono un metro nelle tabelle, non una traiettoria.  Se
    qualcuno la chiede lo stesso, finisce accanto alle celle della spec di
    riferimento, non in giro per l'albero.
    """
    if "/" in cell:
        spec, variant = cell.split("/", 1)
        return layout.dfm_forecast_dir(spec, variant)
    return layout.dfm_benchmark_figure_dir(cell)


# ─── Confronto fra metodi su un trimestre ─────────────────────────────────────

def make_compare(df: pd.DataFrame, output_dir: str, target_quarter: str,
                 ylim: tuple[float, float] | None = _YLIM,
                 nascondi_backcast: bool = False,
                 window_label: str | None = None) -> list[str]:
    """
    Un trimestre solo, tutte le celle sovrapposte, col pallino del realizzato.
    Serve a guardare da vicino un episodio: chi ci arriva e chi resta indietro.
    """
    os.makedirs(output_dir, exist_ok=True)
    sub = df[df["target_quarter"] == target_quarter]
    if sub.empty:
        raise SystemExit(f"{target_quarter} non e' nel CSV.")

    cells = sorted(sub["cella"].unique())
    period = window_label or _period(df)

    with plt.rc_context(_serif()):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        fig.patch.set_facecolor("white")
        xs: list = []
        ys: list = []
        handles: list = []

        for i, cell in enumerate(cells):
            rows = sub[sub["cella"] == cell].sort_values("as_of_dt")
            if len(rows) < _MIN_POINTS:
                continue
            # Stessa regola grafica della 8a: fino al rilascio escluso (backcast
            # incluso) di default, a fine trimestre solo con --nascondi-backcast.
            rows = _line_rows(rows, target_quarter, nascondi_backcast)
            if len(rows) < _MIN_POINTS:
                continue
            color = _COLORS[i % len(_COLORS)]
            ax.plot(rows["as_of_dt"], rows["nowcast_bea"],
                    color=color, linewidth=_LINEWIDTH,
                    label=cell, solid_capstyle="round", zorder=3)
            handles.append(Line2D([0], [0], color=color,
                                  linewidth=_LINEWIDTH, label=cell))
            xs += [rows["as_of_dt"].min(), rows["as_of_dt"].max()]
            ys += [rows["nowcast_bea"].min(), rows["nowcast_bea"].max()]

        realised = sub["realizzato_bea"].iloc[0]
        release = sub["release_dt"].iloc[0]
        if pd.notna(realised) and pd.notna(release):
            ax.plot([release], [realised], marker="o",
                    markersize=np.sqrt(_DOT_SIZE + 25), color="black",
                    linestyle="none", zorder=5)
            ax.annotate("Published GDP", (release, realised),
                        textcoords="offset points", xytext=(7, 0),
                        ha="left", va="center", fontsize=8.5, color="black",
                        annotation_clip=False, zorder=6,
                        path_effects=_LABEL_HALO)
            xs.append(release)
            ys.append(realised)

        if ylim is not None:
            ax.set_ylim(*ylim)
            _autoscale(ax, xs, [])
        else:
            _autoscale(ax, xs, ys)
        lo, hi = ax.get_ylim()
        if lo <= 0.0 <= hi:
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", zorder=2)
        _apply_axes_style(ax)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
        ax.set_ylabel("Annualised growth rate (%)", fontsize=10)
        ax.set_xlabel("Vintage date (as_of)", fontsize=10)
        ax.set_title(f"{target_quarter} nowcast — all methods",
                     fontsize=11.5, pad=10)
        _place_legend(fig, ax, handles, ncol=2 if len(handles) > 4 else 1)
        fig.tight_layout()

        path = os.path.join(output_dir, f"compare_{_short(target_quarter)}_{period}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  scritto: {path}")

    return [path]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Figure del nowcast settimanale.")
    p.add_argument("--csv", nargs="*", default=None,
                   help="default: TUTTI i CSV di dfm/csv/ (una cella per file)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--style", choices=["trajectories", "compare"],
                   default="trajectories")
    p.add_argument("--target", default=None,
                   help="stile compare: il trimestre, es. 2008Q4")
    p.add_argument("--cell", nargs="*", default=None,
                   help="stile trajectories: solo queste celle, es. diag3/student_t")
    p.add_argument("--ylim", nargs=2, type=float, default=None,
                   metavar=("MIN", "MAX"),
                   help="scala y fissa; default: automatica sulla finestra")
    p.add_argument("--window", default=None,
                   help="nome di una finestra di output_layout, es. 2014-2016: "
                        "ritaglia il CSV e nomina le figure con quel nome")
    p.add_argument("--nascondi-backcast", action="store_true",
                   help="taglia la linea a fine trimestre (con lo stacco di "
                        "~4 settimane fino al pallino). Default: la "
                        "linea corre fino al rilascio escluso, backcast incluso, "
                        "e il vuoto fino al pallino e' ~1 settimana.")
    a = p.parse_args()

    paths = discover_csvs(a.csv)
    df = load(paths)
    ylim = tuple(a.ylim) if a.ylim else None

    if a.window:
        df = layout.slice_window(df, a.window)
        if df.empty:
            raise SystemExit(
                f"Nessuna riga dei {len(paths)} CSV cade nella finestra "
                f"{a.window} {layout.window(a.window)}.")

    # Default: il nuovo albero (dfm/<spec>/<variant>), non piu' figures/<spec>.
    out_dir = a.output_dir or layout.OUTPUT_ROOT
    dir_for_cell = None if a.output_dir else _dfm_dir_for_cell

    print(f"leggo: {len(paths)} CSV da {os.path.dirname(paths[0])}")
    print(f"  {len(df)} righe, {df['target_quarter'].nunique()} trimestri, "
          f"celle: {sorted(df['cella'].unique())}")

    if a.style == "trajectories":
        written = make_trajectories(df, out_dir, ylim=ylim, cells=a.cell,
                                    nascondi_backcast=a.nascondi_backcast,
                                    dir_for_cell=dir_for_cell,
                                    window_label=a.window, family="dfm")
    else:
        if not a.target:
            raise SystemExit("lo stile compare richiede --target YYYYQn.")
        written = make_compare(df, layout.comparison_dir(), a.target, ylim=ylim,
                               nascondi_backcast=a.nascondi_backcast,
                               window_label=a.window)

    print(f"\n{len(written)} figura/e scritte.")


if __name__ == "__main__":
    main()
