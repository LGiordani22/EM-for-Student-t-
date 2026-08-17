"""
Figura "Big data in macroeconomics" — replica della Figure 2 di
Bok, Caratelli, Giannone, Sbordone & Tambalotti (2018),
Annual Review of Economics, Vol. 10, pp. 615-643.

Adattata al pannello `final` con 37 serie NY Fed.

La figura mostra:

    1. superficie 3D delle serie temporali standardizzate;
    2. colori delle superfici per categoria economica;
    3. heatmap della stessa matrice sul piano orizzontale;
    4. NaN mantenuti come dati mancanti;
    5. NaN della heatmap visualizzati con un colore distinto.

FEDELTA' AL DATO
----------------

- nessun trimming alla prima data comune;
- nessun ffill;
- nessun riempimento con zero;
- nessuna interpolazione di default;
- NaN mantenuti nella matrice;
- standardizzazione serie per serie sulle osservazioni disponibili;
- stesso pannello dati per superficie e heatmap;
- ordine delle serie = ordine del dataset.

Riferimento:
Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M. & Tambalotti, A.
(2018), "Macroeconomic Nowcasting and Forecasting with Big Data",
Annual Review of Economics, 10, 615-643.
"""

from __future__ import annotations

from pathlib import Path

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D


# ==========================================================================
# PERCORSI
# ==========================================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "data"
    / "processed"
    / "final"
    / "dataset_final.csv"
)

METADATA = (
    ROOT
    / "data"
    / "processed"
    / "final"
    / "metadata_final.csv"
)

FIGDIR = (
    ROOT
    / "docs"
    / "tesi"
    / "figures"
)


# ==========================================================================
# CATEGORIE
# ==========================================================================

CATEGORIES: dict[str, list[str]] = {

    "Housing and construction": [
        "HOUST",
        "PERMIT",
        "HSN1F",
        "TTLCONS",
    ],

    "Retail and consumption": [
        "RSAFS",
        "PCEC96",
        "BUSINV",
        "WHLSLRIMSA",
    ],

    "International trade": [
        "BOPTEXP",
        "BOPTIMP",
        "IR",
        "IQ",
    ],

    "Manufacturing": [
        "INDPRO",
        "TCU",
        "DGORDER",
        "AMDMVS",
        "AMTMUO",
        "AMDMTI",
    ],

    "Income and output": [
        "DSPIC96",
        "A261RX1Q020SBEA",
        "GDPC1",
    ],

    "Others": [
        # Eventuali serie non assegnate a una delle categorie principali.
    ],

    "Surveys": [
        "ISM_PMI",
        "ISM_NMI",
        "ISM_EMP",
        "ISM_PRICES",
        "GACDISA066MSFRBNY",
        "GACDFSA066MSFRBPHI",
    ],

    "Labor": [
        "PAYEMS",
        "USPRIV",
        "UNRATE",
        "JTSJOL",
        "ULCNFB",
    ],

    "Prices": [
        "CPIAUCSL",
        "CPILFESL",
        "PCEPI",
        "PCEPILFE",
        "PPIFIS",
    ],
}


# ==========================================================================
# PALETTE DELLE SUPERFICI
# ==========================================================================
#
# Questa è volutamente più vicina alla Figura 2 originale:
#
#   Housing             -> rosso
#   Retail              -> verde
#   International      -> verde salvia
#   Manufacturing      -> arancio
#   Income/output       -> azzurro
#   Surveys             -> blu scuro
#   Labor               -> rosa/salmone
#   Prices              -> grigio
#
# I colori sono leggermente saturati rispetto alla figura originale,
# perché sulla superficie 3D la trasparenza e la prospettiva li rendono
# naturalmente più spenti.
#

PALETTE = {
    "Housing and construction": "#D64545",  
    "Retail and consumption": "#041610",    
    "International trade": "#760678",      
    "Manufacturing": "#E38B19",             
    "Income and output": "#716804",         
    "Others": "#AB81D0",                    
    "Surveys": "#354A8C",                   
    "Labor": "#6E102B",                     
    "Prices": "#0485A9",                    
}


# ==========================================================================
# HEATMAP
# ==========================================================================
#
# L'originale usa:
#
#   negativo -> rosso
#   vicino alla media -> arancio
#   positivo -> giallo
#
# La barra originale arriva circa da -5 a +10.
#
# SCALA DEI COLORI SEPARATA DALL'ASSE Z
# -------------------------------------
#
# Sul pannello `final` (37 serie in log-diff, dal 1985) la distribuzione
# delle z e' concentratissima:
#
#       |z| <= 2.0   ->  96.0% delle celle
#       |z| <= 2.5   ->  97.9%
#       |z| <= 3.0   ->  98.8%
#       |z| >  5.0   ->   0.27%
#
# Con la colormap tarata su +/-5 il 96% del pannello cade nel 20% centrale
# della rampa: tutto legge arancione e le recessioni non emergono.
#
# Quindi la heatmap usa una scala propria (`cclip`, default 2.0) che
# satura le code, mentre l'asse z della superficie resta su `zclip` = 5
# perche' li' servono i picchi veri (2008, 2020) senza appiattirli.
#
# La saturazione e' dichiarata sulla colorbar con <= / >=: non e' un
# dettaglio estetico, e' il 4% di celle che stanno oltre il fondoscala.
#

CMAP_FLOOR = LinearSegmentedColormap.from_list(
    "bok_floor",
    [
        (0.00, "#730000"),   # molto negativo
        (0.08, "#A30F00"),
        (0.18, "#C92F00"),
        (0.30, "#E45100"),
        (0.42, "#F47B00"),
        (0.50, "#F6A800"),   # circa zero
        (0.60, "#FFC928"),
        (0.72, "#FFD94A"),
        (0.86, "#FFE978"),
        (1.00, "#FFF2A6"),   # molto positivo
    ],
    N=512,
)


# Colore utilizzato per i dati mancanti.
#
# Non è bianco: in questo modo è immediatamente evidente dove il dataset
# non contiene osservazioni.
#
NAN_COLOR = "#09EC3E"  


# ==========================================================================
# STILE
# ==========================================================================

INK = "#0B0B0B"
INK_SOFT = "#555555"
SURFACE = "#FCFCFB"


# Spessore delle linee degli assi (x, y, z) e del bordo della heatmap.
#
# Il default di matplotlib è un filo sottile e grigio: qui gli assi e il
# perimetro del pavimento devono leggersi come una cornice marcata.
#
AXIS_LW = 1.4
BORDO_LW = 1.4


# Spessore del solo lato "Data series" all'ultima data.
#
# Quel segmento corre esattamente sopra il ragged edge finale (le celle
# verdi dell'ultimo mese): con BORDO_LW = 1.4 il tratto nero le copriva.
# Qui resta un filo sottile, che chiude il perimetro senza mangiarsi il
# dato mancante. Stessa ragione per cui l'asse y non viene ispessito.
#
BORDO_LW_FINE = 0.6


# ==========================================================================
# DATI
# ==========================================================================

def carica_pannello(
    start: str | None = None,
    end: str | None = None,
    interp_display: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Legge il dataset e standardizza ogni serie separatamente.

    La media e la deviazione standard vengono calcolate sulle sole
    osservazioni disponibili.

    I NaN rimangono NaN.

    Se interp_display=True, viene effettuata una interpolazione lineare
    soltanto all'interno dello span osservato della serie.
    """

    df = pd.read_csv(
        DATASET,
        index_col=0,
        parse_dates=True,
    )

    meta = pd.read_csv(METADATA)

    if start is not None:
        df = df.loc[
            df.index >= pd.Timestamp(start)
        ]

    if end is not None:
        df = df.loc[
            df.index <= pd.Timestamp(end)
        ]

    # Standardizzazione serie per serie.
    #
    # pandas ignora automaticamente i NaN nel calcolo di mean/std.
    #
    z = (
        df - df.mean()
    ) / df.std()

    if interp_display:
        z = z.interpolate(
            method="linear",
            limit_area="inside",
        )

    return z, meta


# ==========================================================================
# CATEGORIA
# ==========================================================================

def categoria_di(sid: str) -> str:

    for cat, ids in CATEGORIES.items():

        if sid in ids:
            return cat

    raise KeyError(
        f"{sid} non ha una categoria"
    )


# ==========================================================================
# DIAGNOSTICA
# ==========================================================================

def diagnostica(
    start: str | None = "1985-01-01",
) -> pd.DataFrame:

    z, _ = carica_pannello(start)

    righe = []

    for c in z.columns:

        s = z[c]

        fv = s.first_valid_index()
        lv = s.last_valid_index()

        if fv is not None and lv is not None:
            gap_interni = int(
                s.loc[fv:lv].isna().sum()
            )
        else:
            gap_interni = len(s)

        righe.append(
            {
                "serie": c,
                "categoria": categoria_di(c),
                "inizio": fv,
                "gap_interni": gap_interni,
                "copertura_%": round(
                    100 * s.notna().mean(),
                    1,
                ),
                "min_z": round(
                    float(s.min()),
                    1,
                ),
                "max_z": round(
                    float(s.max()),
                    1,
                ),
            }
        )

    tab = pd.DataFrame(righe)

    print(
        f"matrice: "
        f"{z.shape[0]} mesi x {z.shape[1]} serie"
    )

    print(
        f"intervallo: "
        f"{z.index.min():%Y-%m} -> "
        f"{z.index.max():%Y-%m}"
    )

    print(
        f"NaN: "
        f"{int(z.isna().sum().sum())} "
        f"({100 * z.isna().mean().mean():.1f}%)"
    )

    print(
        "serie che NON partono dall'inizio: "
        f"{int((tab['inizio'] > z.index[0]).sum())}"
    )

    print(
        "serie con gap interni: "
        f"{int((tab['gap_interni'] > 0).sum())}"
    )

    return tab


# ==========================================================================
# FIGURA
# ==========================================================================

def figura_bok2(
    start: str | None = "1985-01-01",
    end: str | None = None,

    zclip: float = 5.0,

    # Fondoscala della heatmap, indipendente da `zclip`.
    #
    # None = usa `zclip` (comportamento vecchio, pannello monocromo).
    #
    cclip: float | None = 2.0,

    interp_display: bool = False,

    elev: float = 21.0,

    # azim = -45 rende simmetriche le pendenze dei due lati del pavimento
    # (-23.2 e +23.2 gradi): con -57 il lato delle date correva quasi
    # orizzontale e quello delle serie saliva ripido, per cui i due lati
    # non si leggevano uguali nemmeno quando misuravano uguale.
    #
    azim: float = -45.0,

    figsize: tuple[float, float] = (14.0, 9.0),

    # box_aspect x = y: i due lati del pavimento hanno la stessa lunghezza
    # (misurati 619.1 px entrambi). Insieme ad azim = -45 il pavimento
    # diventa un rombo simmetrico.
    #
    box_aspect: tuple[
        float,
        float,
        float,
    ] = (2.20, 2.20, 1.00),

    zoom: float = 1.15,

    z_a_sinistra: bool = True,

    rasterize: bool = True,

    alpha: float = 0.78,

    gap_pavimento: float = 1.35,

    titolo: str | None = None,
):

    # ======================================================================
    # 1. DATI
    # ======================================================================

    z, _ = carica_pannello(
        start=start,
        end=end,
        interp_display=interp_display,
    )

    serie = list(z.columns)

    cats = [
        categoria_di(s)
        for s in serie
    ]

    n = len(serie)

    date = z.index

    x = mpl.dates.date2num(date)

    Z = z.to_numpy(
        dtype=float
    )

    # Clipping SOLO grafico.
    #
    # I NaN rimangono NaN.
    #
    Zc = np.where(
        np.isnan(Z),
        np.nan,
        np.clip(
            Z,
            -zclip,
            zclip,
        ),
    )

    # Forma:
    #
    #       serie x tempo
    #
    Zg = Zc.T

    # ======================================================================
    # 2. FIGURA
    # ======================================================================

    fig = plt.figure(
        figsize=figsize,
        facecolor=SURFACE,
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
        computed_zorder=False,
    )

    ax.set_facecolor(SURFACE)

    # ======================================================================
    # 3. BORDI TEMPORALI DELLE CELLE
    # ======================================================================
    #
    # QUESTO RISOLVE IL PROBLEMA DELLA HEATMAP TAGLIATA.
    #
    # x contiene il centro temporale di ogni osservazione.
    #
    # x_edges contiene i bordi delle celle.
    #
    # Prima:
    #
    #       ax.set_xlim(x[0], x[-1])
    #
    # tagliava metà della prima e dell'ultima cella.
    #
    # Ora l'asse copre esattamente tutta la heatmap.
    #

    if len(x) > 1:

        x_edges = np.empty(
            len(x) + 1
        )

        x_edges[1:-1] = (
            x[:-1] + x[1:]
        ) / 2.0

        x_edges[0] = (
            x[0]
            - (x[1] - x[0]) / 2.0
        )

        x_edges[-1] = (
            x[-1]
            + (x[-1] - x[-2]) / 2.0
        )

    else:

        x_edges = np.array(
            [
                x[0] - 15,
                x[0] + 15,
            ]
        )

    # ======================================================================
    # 4. LIMITI
    # ======================================================================

    base = (
        -zclip
        * float(gap_pavimento)
    )

    # IMPORTANTE:
    # uso x_edges e non x.
    #
    # Questo rende la heatmap completamente contenuta nell'area grafica.
    #

    ax.set_xlim(
        x_edges[0],
        x_edges[-1],
    )

    ax.set_ylim(
        -0.5,
        n - 0.5,
    )

    ax.set_zlim(
        base,
        zclip * 1.10,
    )

    # ======================================================================
    # 5. GRIGLIA DELLA SUPERFICIE
    # ======================================================================

    Xg, Yg = np.meshgrid(
        x,
        np.arange(n),
    )

    # ======================================================================
    # 6. HEATMAP
    # ======================================================================

    cclip_eff = (
        float(zclip)
        if cclip is None
        else float(cclip)
    )

    # La heatmap satura a +/- cclip_eff.
    #
    # Zg e' gia' clippato a +/- zclip per la superficie: Normalize
    # comprime da sola tutto cio' che eccede sui due estremi della rampa.
    #
    norm = Normalize(
        vmin=-cclip_eff,
        vmax=cclip_eff,
    )

    floor_polys = []
    floor_colors = []

    T = len(x)

    for i in range(n):

        y0 = i - 0.5
        y1 = i + 0.5

        for j in range(T):

            value = Zg[i, j]

            # --------------------------------------------------------------
            # COLORE
            # --------------------------------------------------------------

            if np.isfinite(value):

                color = CMAP_FLOOR(
                    norm(value)
                )

            else:

                # NaN = verde.
                #
                # Non viene eliminata la cella:
                # viene visualizzata come "dato mancante".
                #

                color = to_rgba(
                    NAN_COLOR,
                    alpha=1.0,
                )

            # --------------------------------------------------------------
            # CELLA
            # --------------------------------------------------------------

            x0 = x_edges[j]
            x1 = x_edges[j + 1]

            floor_polys.append(
                [
                    (x0, y0, base),
                    (x1, y0, base),
                    (x1, y1, base),
                    (x0, y1, base),
                ]
            )

            floor_colors.append(
                color
            )

    # ======================================================================
    # 7. POLIGONI HEATMAP
    # ======================================================================

    if floor_polys:

        floor = Poly3DCollection(
            floor_polys,
            facecolors=floor_colors,
            edgecolors="none",
            linewidths=0,
            antialiased=False,
            zorder=0,
        )

        ax.add_collection3d(
            floor
        )

        # ------------------------------------------------------------------
        # NIENTE CLIPPING
        # ------------------------------------------------------------------
        #
        # Con zoom > 1 il box 3D sporge fuori dal rettangolo degli assi.
        #
        # Di default matplotlib ritaglia gli artisti su quel rettangolo:
        # i due spigoli estremi del pavimento (1985 in basso a sinistra e
        # l'ultima data in alto a destra) venivano tagliati via di netto.
        #
        # Disattivando il clipping il quadrilatero viene disegnato per
        # intero e gli spigoli tornano a chiudersi a punta.
        #
        floor.set_clip_on(
            False
        )

        if rasterize:

            try:
                floor.set_rasterized(
                    True
                )
            except Exception:
                pass

    # ======================================================================
    # 7b. BORDO DELLA HEATMAP
    # ======================================================================
    #
    # Perimetro del pavimento, sugli stessi spigoli delle celle:
    # da x_edges[0] a x_edges[-1] e da -0.5 a n-0.5.
    #
    # zorder = 1: sta sopra le celle ma sotto la superficie, così i nastri
    # che scendono in basso lo coprono invece di essergli disegnati sotto.
    #
    # Il perimetro NON e' un tratto unico: il lato "Data series" all'ultima
    # data (da (x_edges[-1], -0.5) a (x_edges[-1], n-0.5)) corre sopra il
    # ragged edge finale e va disegnato sottile, gli altri tre restano
    # marcati come gli assi.
    #
    lati = [
        # (xs, ys, spessore)
        (
            [x_edges[0], x_edges[-1]],
            [-0.5, -0.5],
            BORDO_LW,
        ),
        (
            [x_edges[-1], x_edges[-1]],
            [-0.5, n - 0.5],
            BORDO_LW_FINE,
        ),
        (
            [x_edges[-1], x_edges[0]],
            [n - 0.5, n - 0.5],
            BORDO_LW,
        ),
        (
            [x_edges[0], x_edges[0]],
            [n - 0.5, -0.5],
            BORDO_LW,
        ),
    ]

    for xs, ys, lw in lati:

        ax.plot(
            xs,
            ys,
            [base] * len(xs),

            color=INK,

            linewidth=lw,

            solid_joinstyle="miter",

            clip_on=False,

            zorder=1,
        )

    # ======================================================================
    # 8. SUPERFICIE 3D
    # ======================================================================

    fc = np.zeros(
        (
            n,
            len(x),
            4,
        ),
        dtype=float,
    )

    for i, cat in enumerate(cats):

        fc[i, :, :] = to_rgba(
            PALETTE[cat],
            alpha,
        )

    surf = ax.plot_surface(
        Xg,
        Yg,
        Zg,

        facecolors=fc,

        shade=False,

        rstride=1,
        cstride=1,

        linewidth=0.08,

        edgecolor=(
            0.10,
            0.10,
            0.10,
            0.20,
        ),

        antialiased=True,

        zorder=5,
    )

    # Stessa ragione del pavimento: la superficie non va ritagliata sul
    # rettangolo degli assi.
    #
    surf.set_clip_on(
        False
    )

    if rasterize:

        try:
            surf.set_rasterized(
                True
            )
        except Exception:
            pass

    # ======================================================================
    # 9. ASSI X
    # ======================================================================

    anni = [
        a
        for a in range(
            1985,
            2031,
            5,
        )
        if (
            date[0]
            - pd.Timedelta(days=32)
            <= pd.Timestamp(
                f"{a}-01-01"
            )
            <= date[-1]
        )
    ]

    ax.set_xticks(
        [
            mpl.dates.date2num(
                pd.Timestamp(
                    f"{a}-01-01"
                )
            )
            for a in anni
        ]
    )

    ax.set_xticklabels(
        [str(a) for a in anni],
        fontsize=9,
        color=INK,
        fontweight="bold",
    )

    # ======================================================================
    # 10. ASSE Y
    # ======================================================================

    # Le 37 serie non vengono scritte individualmente.
    #

    ax.set_yticks([])

    # Scritta in orizzontale: inclinata seguiva la prospettiva dell'asse e
    # si leggeva peggio della colorbar accanto.
    #
    # Su mplot3d `rotation=0` da solo non basta: l'asse ricalcola da se'
    # l'angolo dell'etichetta a ogni disegno, e va disattivato prima.
    #
    ax.yaxis.set_rotate_label(
        False
    )

    ax.set_ylabel(
        "Data series",
        fontsize=11,
        color=INK,
        fontweight="bold",
        labelpad=14,
        rotation=0,
        ha="center",
        va="center",
    )

    # ======================================================================
    # 11. ASSE Z
    # ======================================================================

    ax.set_zticks(
        [
            -zclip,
            0,
            zclip,
        ]
    )

    ax.set_zticklabels(
        [
            f"−{zclip:g}",
            "0",
            f"{zclip:g}",
        ],
        fontsize=9,
        color=INK,
        fontweight="bold",
    )

    ax.tick_params(
        axis="z",
        pad=1,
        colors=INK,
    )

    # ======================================================================
    # 12. STILE ASSI
    # ======================================================================

    ax.grid(False)

    for pane in (
        ax.xaxis,
        ax.yaxis,
        ax.zaxis,
    ):

        pane.pane.set_visible(
            False
        )

        # L'asse y ("Data series") resta con lo stile di default: e' il lato
        # che passa sopra il ragged edge finale, ispessirlo lo copriva.
        #
        if pane is ax.yaxis:
            continue

        # Gli altri assi in nero pieno e marcati, non il filo grigio di
        # default.
        #
        pane.line.set_color(
            INK
        )

        pane.line.set_linewidth(
            AXIS_LW
        )

        pane.line.set_solid_capstyle(
            "round"
        )

    # Trattini delle tacche coerenti con lo spessore degli assi.
    #
    for asse in ("x", "y", "z"):

        ax.tick_params(
            axis=asse,
            colors=INK,
            width=1.0,
            length=4.0,
        )

    # ======================================================================
    # 13. CAMERA
    # ======================================================================

    ax.view_init(
        elev=elev,
        azim=azim,
    )

    if z_a_sinistra:

        try:

            pl = ax.zaxis._PLANES

            ax.zaxis._PLANES = (
                pl[2],
                pl[3],
                pl[0],
                pl[1],
                pl[4],
                pl[5],
            )

        except AttributeError:
            pass

    ax.set_box_aspect(
        box_aspect,
        zoom=zoom,
    )

    # ======================================================================
    # 14. DIAGNOSTICA
    # ======================================================================

    sd_per_passo = (
        box_aspect[2]
        / (
            zclip
            * (
                1.10
                + gap_pavimento
            )
        )
    ) / (
        box_aspect[1]
        / n
    )

    print(
        f"1 sd = "
        f"{sd_per_passo:.2f} "
        f"passi fra serie "
        f"("
        f"{'compenetrazione' if sd_per_passo > 1.5 else 'nastri separati'}"
        f")"
    )

    # ======================================================================
    # 15. COLORBAR
    # ======================================================================
    #
    # Più vicina al grafico rispetto alla versione precedente.
    #
    # Etichette con <= / >= quando la scala colore satura, cioe' quando
    # cclip < zclip: fuori dal fondoscala ci sono celle vere, e la barra
    # deve dirlo.
    #

    sm = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=CMAP_FLOOR,
    )

    # La barra sta SOPRA la figura, orizzontale.
    #
    # Prima era verticale a sinistra: occupava una colonna intera e
    # spingeva il disegno fuori asse. In alto sfrutta la fascia vuota
    # sopra i picchi della superficie e lascia il grafico centrato.
    #
    # x scelto in modo che barra + quadratino dei NaN, presi insieme,
    # risultino centrati sulla figura.
    #

    cax = fig.add_axes(
        (
            0.250,
            0.900,
            0.280,
            0.018,
        )
    )

    cb = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
    )

    # Tacche sotto la barra, titolo sopra.
    #
    cax.xaxis.set_ticks_position(
        "bottom"
    )

    cax.xaxis.set_label_position(
        "bottom"
    )

    cb.outline.set_edgecolor(
        INK_SOFT
    )

    cb.outline.set_linewidth(
        0.6
    )

    cb.set_ticks(
        [
            -cclip_eff,
            0,
            cclip_eff,
        ]
    )

    satura = cclip_eff < zclip

    cb.set_ticklabels(
        [
            (
                f"≤ −{cclip_eff:g}"
                if satura
                else f"−{cclip_eff:g}"
            ),
            "0",
            (
                f"≥ {cclip_eff:g}"
                if satura
                else f"{cclip_eff:g}"
            ),
        ]
    )

    cb.ax.tick_params(
        labelsize=8.5,
        colors=INK,
        length=2,
        pad=2,
    )

    for t in cb.ax.get_xticklabels():

        t.set_fontweight(
            "bold"
        )

    # Titolo sopra la barra (con la barra orizzontale, `set_label` finirebbe
    # sotto le tacche).
    #
    cb.ax.set_title(
        "Standard deviations from mean",
        fontsize=10.5,
        color=INK,
        pad=6,
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # QUADRATINO DEI DATI MANCANTI
    # ------------------------------------------------------------------
    #
    # Sta accanto alla barra, non giu' nella legenda delle categorie: il
    # verde della heatmap non e' una categoria economica, e' il ragged
    # edge del pannello e va letto insieme alla scala dei colori.
    #

    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=8,
                markerfacecolor=NAN_COLOR,
                markeredgecolor="none",
                label="Missing data (ragged edge)",
            )
        ],

        loc="center left",

        bbox_to_anchor=(
            0.565,
            0.918,
        ),

        frameon=False,

        labelcolor=INK,

        handletextpad=0.5,

        prop={
            "weight": "bold",
            "size": 10.5,
        },
    )

    # ======================================================================
    # 16. LEGENDA
    # ======================================================================
    #
    # Ordine intenzionalmente vicino alla Figura 2 originale.
    #
    # La figura originale ha 8 categorie:
    #
    # Housing and construction
    # Retail and consumption
    # International trade
    # Manufacturing
    # Income
    # Others
    # Surveys
    # Labor
    #
    # Nel nostro pannello abbiamo Prices, quindi viene mantenuta la nostra
    # categoria aggiuntiva.
    #

    legend_order = [
        "Housing and construction",
        "Retail and consumption",
        "International trade",
        "Manufacturing",
        "Income and output",
        "Others",
        "Surveys",
        "Labor",
        "Prices",
    ]

    handles = []

    for category in legend_order:

        # Se la categoria non è effettivamente presente nel dataset,
        # non la inseriamo.
        #

        if not any(
            cat == category
            for cat in cats
        ):
            continue

        handles.append(
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=9,

                markerfacecolor=(
                    PALETTE[category]
                ),

                markeredgecolor="none",

                label=category,
            )
        )

    fig.legend(
        handles=handles,

        loc="lower center",

        # Con la colorbar spostata in alto il disegno e' centrato sulla
        # figura, quindi la legenda torna al centro esatto.
        #
        # y abbassato da 0.085: il corpo e' passato da 9 a 11.5 pt e le tre
        # righe crescono verso l'alto. Ancorandola piu' in basso la legenda
        # occupa la fascia bianca sotto il pavimento invece di salirci
        # sopra, e resta comunque centrata (x = 0.50).
        #
        bbox_to_anchor=(
            0.50,
            0.022,
        ),

        ncol=4,

        frameon=False,

        labelcolor=INK,

        handletextpad=0.6,

        columnspacing=2.2,

        labelspacing=0.7,

        prop={
            "weight": "bold",
            "size": 11.5,
        },
    )

    # ======================================================================
    # 17. TITOLO
    # ======================================================================

    if titolo:

        fig.text(
            0.52,
            0.965,
            titolo,
            ha="center",
            fontsize=11,
            color=INK,
        )

    # ======================================================================
    # 18. LAYOUT
    # ======================================================================

    # Niente piu' colonna occupata a sinistra: il disegno usa tutta la
    # larghezza ed e' simmetrico rispetto al centro della figura.
    #
    # `top` > 1 NON e' un errore.
    #
    # Il riquadro degli assi 3D e' sempre molto piu' alto del cubo che ci
    # viene proiettato dentro (a elev = 21 il disegno ne occupa ~67% in
    # altezza, e per giunta sbilanciato verso il basso): il margine di
    # troppo restava come banda bianca fra la barra e i picchi della
    # superficie. Non lo si toglie ne' con `top` ne' spostando la barra,
    # perche' il ritaglio "tight" include comunque il riquadro degli assi.
    #
    # Qui il riquadro viene invece esteso oltre il bordo della figura: il
    # disegno sale e si ingrandisce, e il margine vuoto finisce fuori
    # pagina. Per questo `salva` NON usa bbox_inches="tight": e' il bordo
    # della figura a fare il taglio.
    #
    fig.subplots_adjust(
        left=0.040,
        right=0.960,
        # bottom e top scendono insieme: il disegno trasla verso la legenda
        # senza cambiare scala (l'altezza del riquadro resta 1.043).
        #
        bottom=0.085,
        top=1.128,
    )

    return fig, ax


# ==========================================================================
# SALVATAGGIO
# ==========================================================================

def salva(
    fig,
    nome: str,
) -> Path:

    FIGDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for ext in (
        "pdf",
        "png",
    ):

        # NIENTE bbox_inches="tight".
        #
        # Il riquadro degli assi 3D sporge sopra il bordo della figura
        # (vedi `subplots_adjust` in figura_bok2): il ritaglio "tight" lo
        # rimetterebbe dentro e riporterebbe la banda bianca in cima.
        # Qui il taglio lo fa il bordo della figura.
        #
        fig.savefig(
            FIGDIR / f"{nome}.{ext}",
            dpi=200,
            facecolor=SURFACE,
        )

    return (
        FIGDIR
        / f"{nome}.pdf"
    )


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":

    diagnostica()

    fig, _ = figura_bok2()

    print(
        salva(
            fig,
            "fig_panel_3d",
        )
    )