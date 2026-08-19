"""
core/bvar/qbvar.py

IL Q-BVAR — §2.1 del paper: il VAR trimestrale baseline, e lo stadio 1 del
C-BVAR.

Vedi l'header del pacchetto (`core/bvar/__init__.py`) per la mappa delle quattro
varianti e per l'asse lungo cui si dispongono.  Qui si svolge la casella Q.


================================================================================
COSA DICE IL PAPER
================================================================================
Niente di piu' dell'eq. (1): un VAR(p) trimestrale, p=5, stimato in forma
chiusa sul posterior Normal-Inverse-Wishart.

    "When all variables in the vector x_tq are available, the model can be
     readily estimated with standard Bayesian methods"                   (§2.1)

E la nota 20, per cui questo modello compare due volte nel paper:

    "The Q-BVAR corresponds to the first step needed to obtain the C-BVAR,
     see Section 2.4."                                                (nota 20)

Sull'aggregazione dei mensili, la nota 17 — che e' l'unica cosa che questo
modulo aggiunge al Gate 1:

    "As discussed in Section 2.4, for the C-BVAR monthly variables are
     transformed so as to correspond to a quarterly quantity when observed in
     the final month of each quarter BEFORE TAKING LOGS (see Giannone et al.,
     2008).  With our data, that means taking 3-months moving averages of ALL
     MONTHLY VARIABLES."                                               (nota 17)


================================================================================
COSA SIGNIFICA
================================================================================
Il Q-BVAR non RISOLVE il problema della frequenza mista: lo DISSOLVE, buttando
via la frequenza mensile prima ancora di stimare.  Tutto il suo contenuto sta a
monte, nella costruzione del pannello trimestrale — cioe' in questo modulo.

Ne seguono i due ruoli, da tenere separati in testa:

  * come MODELLO e' il TERMINE DI PARAGONE.  E' cieco dentro il trimestre per
    costruzione, e le tre varianti a frequenza mista lo battono esattamente
    dove ci si aspetta: "we find differences in performance across the three
    methods only in the first few weeks of the quarter, when no information on
    the current quarter is available.  After that, all the mixed-frequency
    models are comparable and OUTPERFORM A STANDARD QUARTERLY VAR."  Quella
    cecita' e' la definizione del vantaggio che stiamo misurando, non un
    difetto dell'implementazione.

  * come STADIO 1 e' il fornitore di (Phi, Sigma_eps) al C-BVAR.  Da cui il
    vincolo d'interfaccia in fondo a questo header.

DOVE SI APPLICA LA MEDIA MOBILE, E DOVE NO.  La nota 17 nomina il solo C-BVAR.
Vale pero' anche per il Q, PER DEDUZIONE: la nota 20 dice che il Q-BVAR *e'* lo
stadio 1 del C, quindi se stimasse su un pannello costruito diversamente, la
Phi che il Gate 3 prende a radice cubica non sarebbe la Phi del sistema che il
C-BVAR assume.  Un pannello solo, quello con le medie mobili.

Non vale invece per B ed L, che i mensili li vogliono grezzi (§2.3 e nota 11).
Mediare li' distruggerebbe il motivo per cui esistono.


================================================================================
COME SI TRADUCE IN CODICE
================================================================================
E' un WRAPPER: zero matematica nuova, zero campionatore nuovo.  La catena:

    data.build_panel(..., model_units=False)     livelli mensili mascherati
      -> three_month_average()                   media mobile a 3 mesi   <- NEW
      -> data.to_model_units()                   log / identita'
      -> campiona ai mesi 3/6/9/12
      -> data.assert_dense()                     l'invariante del Gate 0
      -> core.sample()                           il core, invariato

`to_quarterly()` fa i tre passi centrali, `build_quarterly_panel()` la catena
intera, `fit()` aggiunge la stima e impacchetta l'output per il Gate 3.


I TRE PUNTI DI ESECUZIONE DELL'AGGREGAZIONE
============================================
Sono i tre posti dove questa trasformazione, apparentemente banale, puo'
sbagliare in silenzio.  Stanno scritti qui perche' il codice che li applica e'
sotto.

--- 1. L'ORDINE DELLE OPERAZIONI: livelli -> MA 3 mesi -> log ---------------

Lo decide il paper, non noi: la nota 17 dice "BEFORE TAKING LOGS".

Il motivo e' che le due operazioni NON COMMUTANO.  log(media) e media(log) sono
quantita' diverse — la seconda e' il log della media GEOMETRICA — e per la
concavita' del logaritmo si ha sempre media(log) <= log(media).

Misurato sulle nostre 21 mensili in log del profilo q_b, 1992Q1-2025Q3:

    serie      gap medio sul livello   gap max    diff. sulla crescita, max
    HSN1F              0.11%            1.68%             166 punti base
    HOUST              0.11%            0.90%              77 pb
    PERMIT             0.05%            0.68%              62 pb
    ...
    PCEPILFE          0.0001%           0.001%            0.08 pb

Cioe': il costo si concentra dove le serie sono volatili (edilizia) ed e' nullo
sui prezzi.  Non catastrofico, ma sistematico, con segno fisso, e comunque
sbagliato di concetto — la media dei log non e' cio' che una fonte statistica
pubblica come dato trimestrale.

CONSEGUENZA SUL CODICE, ed e' il punto in cui l'errore entrerebbe non visto:
`data.build_panel` applica il log DENTRO DI SE'.  Applicare la media al suo
output darebbe esattamente media(log).  Per questo la media va infilata fra la
mascheratura e la trasformazione, ed e' la ragione per cui `build_panel` ha
ricevuto il flag `model_units=False`.  Unica modifica a codice gia' scritto che
il Gate 2 richiede, retrocompatibile.

Per le serie in LIVELLO l'ordine e' indifferente (l'identita' commuta con la
media): la regola resta comunque una sola, si media sempre sul livello grezzo.

--- 2. LE SERIE GIA' IN LIVELLO: si mediano anche loro -----------------------

La nota 17 dice "all monthly variables", senza qualificazioni, e la Tabella 1
del paper contiene mensili in Level (unemployment rate, Fed funds, credit
spread, EPU, PMI) che cadono sotto quel "all".

Nel profilo q_b sono SEI: UNRATE, TCU, Philly Fed, ISM_PMI, ISM_PRICES,
ISM_EMP.  (Empire e ISM_NMI sono in livello ma stanno fra le 7 serie escluse
per partenza tardiva: vivono solo nel profilo `l`, che la media non la usa.)
Le 3 trimestrali non ricevono nulla — sono gia' quantita' trimestrali.  La
media tocca 27 colonne su 30.

I DUE ATTRITI CHE VALE LA PENA AVER NOTATO:

  * UNRATE e TCU.  Per un tasso, "media del trimestre" e "valore a fine
    trimestre" sono davvero due convenzioni diverse.  Qui pero' non c'e'
    attrito vero: la convenzione standard per la versione trimestrale di
    entrambi e' gia' la MEDIA, non il valore terminale.  Seguendo il paper
    coincidiamo con la fonte, non divergiamo.

  * I quattro indici di diffusione (Philly + 3 ISM).  Mediarli e' ben definito
    e l'ISM stesso pubblica medie a 3 mesi.  L'attrito e' di sostanza: la media
    DILUISCE ESATTAMENTE LA TEMPESTIVITA' che rende preziose queste serie —
    l'ISM di gennaio e' il primo segnale su Q1 e dentro una MA3 pesa un terzo.
    Non e' un errore: e' il costo INTRINSECO della convenzione dati di Q e C, ed
    e' la ragione per cui esistono le varianti a frequenza mista.  Ce lo
    aspettiamo come risultato al Gate 6.

  * Il paper ammette da se' un costo affine, nota 15: definire le mensili come
    media sul trimestre "implies that we are introducing a non-invertible
    moving average in the growth rates.  Therefore modelling this monthly
    concept as autoregressive INTRODUCES SOME MIS-SPECIFICATION."  E' il costo
    dichiarato dalla fonte, non un'obiezione nostra.

--- 3. IL BORDO FRASTAGLIATO: nessuna media parziale, mai --------------------

Una MA a 3 mesi vuole 3 mesi.  All'ultimo trimestre, in nowcasting, potrebbero
essercene 1 o 2.  La regola, una sola:

    UNA CELLA MA3 ESISTE SE E SOLO SE TUTTI E TRE I MESI CI SONO.
    Altrimenti e' NaN, e chi sta a valle decide che farne.

`min_periods=3` e' scritto a mano in `three_month_average`, non lasciato al
default di pandas: e' una decisione, e va letta come tale.

L'ALTERNATIVA SCARTATA: mediare sui mesi disponibili.  Cambierebbe in silenzio
la definizione della variabile — l'osservazione non sarebbe piu' "la quantita'
trimestrale" ma un'altra variabile casuale, con media diversa e varianza molto
maggiore, data in pasto a un filtro convinto di ricevere quella vera.  E'
iniezione di errore di misura travestita da dato.

COSA SUCCEDE ALLORA AL BORDO, modello per modello:

  * IN STIMA NON SI PONE.  Il campione si chiude al 2025-09-30, che e' un
    quarter-end denso: ogni trimestre stimato ha tutti e tre i mesi.
    `assert_dense` resta il guardiano del core.

  * C-BVAR (Gate 3): lo gestisce il Kalman, NATIVAMENTE, ed e' il punto di
    §2.4.  Il modello e' MENSILE (Phi_m): i mesi gia' usciti del trimestre in
    corso entrano come osservazioni vere alle loro date mensili, e la quantita'
    di fine trimestre e' prodotta dal filtro che prevede il mese o i due mesi
    mancanti.  Nessuna media parziale viene mai formata: non esiste proprio
    come oggetto.

  * Q-BVAR: NON PUO' usare un trimestre APERTO, strutturalmente.  Non c'e'
    nessun modello mensile su cui appoggiarsi, e una media parziale non si forma
    mai: nelle settimane 1-13 il "nowcast" del trimestre corrente e' una
    previsione pura su dati fino a q-1.  E' cio' che il Q-BVAR *e'*, non un
    limite di questa implementazione — e se non fosse cieco li' dentro, al Gate
    6 non staremmo misurando niente.

    MA IL TRIMESTRE CHIUSO E' UN'ALTRA COSA, e questo punto le confondeva.
    Appena il trimestre finisce, le sue celle MA3 si completano serie per serie
    mentre il PIL non e' ancora uscito: alla settimana 14 sono gia' dentro ISM e
    finanziarie, alla 16 sono 16 serie su 30.  Quella riga trimestrale
    PARZIALMENTE osservata e' informazione vera sull'obiettivo, e il Q-BVAR ci
    condiziona attraverso il filtro — `run_Ksmoother_FM` in STEP4_QBVAR.m
    r.149.  E' il "catching up ... in week 14" di §3.2, e senza di esso il
    Q-BVAR resta piatto per un motivo implementativo, non strutturale.  Vedi la
    sezione 4 di questo modulo (`nowcast`).

CONSEGUENZA PRATICA: `assert_dense` va chiamata sul pannello di STIMA e mai su
quello di NOWCAST, dove il NaN al bordo ci deve essere.  Qui la separazione e'
nei nomi: `estimation_panel()` la chiama, `build_quarterly_panel()` no.


================================================================================
L'INTERFACCIA CHE IL GATE 3 RIUSERA'
================================================================================
Il C-BVAR parte da qui, quindi l'output e' disegnato ADESSO nella forma che il
Gate 3 consuma — cosi' non si rifattorizza.  Per ogni estrazione s, gli oggetti
dell'Appendice A:

    fit.companion(s)   (n*p, n*p)   Phi, la companion di (A_1 ... A_p)  (A.1)
    fit.Omega(s)       (n*p, n*p)   blkdiag(Sigma_eps, 0)               (A.1)
    fit.Sigma[s]       (n, n)       Sigma_eps
    fit.const[s]       (n,)         A_0
    fit.A[s]           (p, n, n)    le matrici di lag, se servono sciolte

LA REGOLA DI DISACCOPPIAMENTO: al Gate 3 la mappa cube-root sara' una FUNZIONE
PURA di (Phi, Sigma_eps) — array in, array out — non un metodo che sappia che
cos'e' un Q-BVAR.  Cosi' `cube_root.py` si testa da solo contro l'esempio AR(2)
dell'Appendice A.1 (eq. A.11-A.15), senza stimare niente, e il Gate 3 non deve
mai entrare dentro l'oggetto del Gate 2.

PUNTO APERTO REGISTRATO, da affrontare al Gate 3 e non qui: la companion (A.1)
dell'Appendice A NON HA LA COSTANTE, mentre il nostro VAR ha A_0.  Dove finisca
A_0 nella mappa verso il modello mensile il paper non lo dice.  Decisione da
§1b regola 4 (paper silente -> non si improvvisa).  Non blocca il Gate 2:
`const` viene comunque stimata, esposta e verificata dal recovery test; e' solo
il suo USO nel modello mensile a essere indeciso.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.bvar.core import BVARDraws, sample
from core.bvar.data import (
    assert_dense,
    build_panel,
    estimation_end,
    to_model_units,
)
from core.bvar.spec import BVARSpec

#: I mesi in cui una cella trimestrale esiste.  Stessa convenzione di `data.py`.
QUARTER_END_MONTHS = (3, 6, 9, 12)

#: La finestra della media mobile.  Tre, e non e' un parametro: e' il numero di
#: mesi in un trimestre (nota 17).  Nominato per leggibilita', non per essere
#: cambiato.
MA_WINDOW = 3


# ─── 1. L'aggregazione ────────────────────────────────────────────────────────

def three_month_average(levels: pd.DataFrame, spec: BVARSpec) -> pd.DataFrame:
    """
    La media mobile a 3 mesi della nota 17, applicata AI LIVELLI GREZZI.

    Tocca le sole colonne mensili: le trimestrali sono gia' quantita'
    trimestrali e portano un valore solo ai mesi 3/6/9/12, quindi una media
    mobile su di loro produrrebbe solo NaN (due dei tre mesi mancano SEMPRE, per
    costruzione del loader, non per dato mancante).

    `min_periods=MA_WINDOW` e' esplicito ed e' la regola del punto 3: una cella
    esiste se e solo se tutti e tre i mesi ci sono.  Mai una media parziale.

    Parameters
    ----------
    levels : (T_m, n)
        Pannello MENSILE in LIVELLI, gia' mascherato dal calendario.  Cioe'
        l'uscita di `data.build_panel(..., model_units=False)`.

    Returns
    -------
    pd.DataFrame
        Stesso indice e stesse colonne.  Le mensili sono mediate, le
        trimestrali sono passate attraverso intatte.
    """
    out = levels.copy()
    monthly = [c for c in out.columns if c in set(spec.monthly)]
    if monthly:
        out[monthly] = out[monthly].rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    return out


def to_monthly_units(levels: pd.DataFrame, spec: BVARSpec) -> pd.DataFrame:
    """
    I primi DUE passi di `to_quarterly`, senza il campionamento trimestrale:
    media mobile a 3 mesi sui livelli, poi log/identita'.

    E' l'input del C-BVAR (Gate 3): il suo modello e' MENSILE, quindi consuma le
    medie mobili a ogni mese invece che ai soli fine trimestre.  Vive qui e non
    in `cbvar.py` perche' la logica della media mobile — e i tre punti di
    esecuzione che la governano — stanno in questo header, e duplicarla sarebbe
    il modo piu' rapido di farle divergere.

    Le trimestrali restano NaN fuori dai mesi 3/6/9/12: e' il placement del
    loader, ed e' cio' che il filtro del C-BVAR tratta come stato latente.
    """
    return to_model_units(three_month_average(levels, spec), spec)


def build_monthly_panel(
    spec: BVARSpec,
    as_of=None,
    *,
    end=None,
    raw: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    trim: bool = True,
) -> pd.DataFrame:
    """
    La catena completa fino al pannello MENSILE in unita' del modello.

    Parameters
    ----------
    trim : bool
        True (default) taglia le prime righe con la finestra della media mobile
        incompleta — le stesse che `estimation_panel` taglia in cima.
    """
    levels = build_panel(spec, as_of, end=end, raw=raw, metadata=metadata,
                         model_units=False)
    mon = to_monthly_units(levels, spec)
    if trim:
        full = mon.notna().all(axis=1)
        if full.any():
            mon = mon.loc[full.idxmax():]
    return mon


def to_quarterly(levels: pd.DataFrame, spec: BVARSpec) -> pd.DataFrame:
    """
    Da livelli mensili mascherati al pannello TRIMESTRALE in unita' del modello.

    I tre passi, ed e' l'ordine a essere il contenuto (punto 1 dell'header):

        1. media mobile a 3 mesi        SUI LIVELLI
        2. log / identita'              DOPO la media, mai prima
        3. campionamento ai mesi 3/6/9/12

    Il passo 3 e' la meta' che il paper chiama "when observed in the final month
    of each quarter": la cella di marzo porta la media di gennaio-febbraio-marzo,
    ed *e'* il dato del primo trimestre.

    Parameters
    ----------
    levels : (T_m, n)
        Uscita di `data.build_panel(..., model_units=False)`.

    Returns
    -------
    pd.DataFrame
        Indice = i soli quarter-end, colonne = `spec.series` nell'ordine della
        config.  Puo' contenere NaN: al bordo (trimestre incompleto) e in cima
        (i primi due mesi non hanno una finestra piena).  Chi va al core deve
        passare da `estimation_panel`, che li esclude e verifica.
    """
    units = to_monthly_units(levels, spec)
    return units.loc[units.index.month.isin(QUARTER_END_MONTHS)]


def build_quarterly_panel(
    spec: BVARSpec,
    as_of=None,
    *,
    end=None,
    raw: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    La catena completa, da disco al pannello trimestrale.

    NON chiama `assert_dense`: e' la funzione che serve anche al NOWCAST, dove
    il bordo frastagliato ci deve essere.  Per il campione di stima usa
    `estimation_panel`.

    Parameters
    ----------
    as_of : date-like | None
        Data di osservazione, per il mascheramento pseudo-real-time.
    end : date-like | None
        Taglio superiore.  Per il campione di stima e' `data.estimation_end`.
    """
    levels = build_panel(spec, as_of, end=end, raw=raw, metadata=metadata,
                         model_units=False)
    return to_quarterly(levels, spec)


def estimation_panel(
    spec: BVARSpec,
    *,
    as_of=None,
    end=None,
    raw: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Il campione di STIMA: trimestrale, denso, verificato.

    Differenze da `build_quarterly_panel`, e sono entrambe deliberate:

      * chiude di default a `data.estimation_end(spec)` — per q_b il 2025-09-30,
        l'ultimo month-end prima del buco da shutdown federale di ottobre 2025;
      * taglia le righe iniziali con la finestra incompleta;
      * chiama `assert_dense`, che e' l'invariante del Gate 0 applicato.

    Se il pannello non e' denso solleva `ValueError` dicendo QUALE dei tre tipi
    di buco ha trovato.  E' voluto: meglio fallire qui che scoprire il NaN
    dentro il posterior coniugato.

    IL TAGLIO IN CODA, E PERCHE' E' ARRIVATO SOLO AL GATE 6
    -------------------------------------------------------
    Con `as_of` dato, le ultime righe sono **parzialmente** mascherate: a
    meta' trimestre alcune serie sono uscite e altre no.  Il campione di stima
    vuole un rettangolo pieno, quindi si taglia all'ultimo trimestre
    COMPLETAMENTE osservato a quella data — l'`endEstimT` degli autori
    (`bbvar.m` r.27: `find(~isnan(sum(X(3:3:end,:),2)),1,'last')`).

    Fino al Gate 5 non serviva perche' `as_of` non arrivava mai fin qui: il
    C-BVAR chiamava questa funzione SENZA `as_of` e stimava sempre sul campione
    completo.  In real time e' look-ahead, ed e' il baco che l'oracolo del Gate
    6 ha preso (`tests/test_gate6.py`).  Vedi l'header di `evaluate.py`.

    NOTA: si taglia solo la CODA, non i buchi interni.  Un NaN in mezzo deve
    ancora far fallire `assert_dense` — e' un'altra cosa, ed e' un errore.
    """
    if end is None:
        end = estimation_end(spec)
    panel = build_quarterly_panel(spec, as_of, end=end, raw=raw, metadata=metadata)
    # Le prime righe possono avere la finestra della MA incompleta: si TAGLIANO,
    # non si riempiono.  E' lo stesso principio del punto 3, applicato in cima
    # invece che in fondo.  Un buco piu' avanti NON viene tagliato: lo deve
    # vedere `assert_dense` e farlo fallire.
    full = panel.notna().all(axis=1)
    if full.any():
        panel = panel.loc[full.idxmax():full[::-1].idxmax()]
    assert_dense(panel, spec)
    return panel


# ─── 2. Il risultato, nella forma che il Gate 3 consuma ───────────────────────

@dataclass
class QBVARFit:
    """
    L'uscita del Q-BVAR: le estrazioni del core, piu' gli accessori
    dell'Appendice A che il C-BVAR consumera' al Gate 3.

    Il layout di `B` (k, n) e' quello fissato al Gate 1 — lag-major, costante in
    ULTIMA riga:

        B[s*n : (s+1)*n, :] = A_{s+1}'         s = 0 ... p-1
        B[-1, :]            = A_0

    da cui `A`, `const` e `companion` sono semplici riletture, non calcoli.
    """

    draws: BVARDraws
    panel: pd.DataFrame = field(repr=False)
    spec: BVARSpec = field(repr=False)

    # ── dimensioni ──────────────────────────────────────────────────────────
    @property
    def n(self) -> int:
        return self.spec.n

    @property
    def p(self) -> int:
        return self.spec.p

    @property
    def S(self) -> int:
        """Numero di estrazioni raccolte."""
        return int(self.draws.B.shape[0])

    @property
    def n_state(self) -> int:
        """n*p, la dimensione della companion dell'eq. (A.1)."""
        return self.n * self.p

    # ── i parametri, sciolti ────────────────────────────────────────────────
    @property
    def A(self) -> np.ndarray:
        """(S, p, n, n) — le matrici di lag.  `A[s, j]` e' A_{j+1} dell'eq. (1)."""
        n, p = self.n, self.p
        B = self.draws.B[:, : n * p, :]                 # (S, n*p, n)
        return B.reshape(self.S, p, n, n).transpose(0, 1, 3, 2)

    @property
    def const(self) -> np.ndarray:
        """(S, n) — A_0.  Vedi il punto aperto in fondo all'header."""
        return self.draws.B[:, -1, :]

    @property
    def Sigma(self) -> np.ndarray:
        """(S, n, n) — Sigma_eps."""
        return self.draws.Sigma

    # ── gli oggetti dell'Appendice A, per estrazione ────────────────────────
    def companion(self, s: int) -> np.ndarray:
        """
        Phi dell'eq. (A.1), per l'estrazione `s`.

            Phi = [[A_1 A_2 ... A_p],
                   [I    0  ...  0 ],
                   [0    I  ...  0 ],
                   [        ...    ]]

        Costruita per estrazione e non tutta insieme di proposito: a n=30, p=5,
        S=1000 la pila sarebbe 180 MB, e il Gate 3 lavora comunque draw per
        draw (la radice cubica e' una decomposizione spettrale per estrazione).
        """
        n, p, ns = self.n, self.p, self.n_state
        A = self.A[s]                                    # (p, n, n)
        Phi = np.zeros((ns, ns))
        Phi[:n] = np.hstack([A[j] for j in range(p)])
        if p > 1:
            Phi[n:, : ns - n] = np.eye(ns - n)
        return Phi

    def Omega(self, s: int) -> np.ndarray:
        """
        Omega dell'eq. (A.1): Sigma_eps nel blocco in alto a sinistra, zero
        altrove.  E' singolare per costruzione — l'innovazione entra solo nelle
        prime n componenti dello stato.
        """
        Om = np.zeros((self.n_state, self.n_state))
        Om[: self.n, : self.n] = self.draws.Sigma[s]
        return Om

    # ── diagnostica ─────────────────────────────────────────────────────────
    def spectral_radius(self) -> np.ndarray:
        """
        (S,) — il raggio spettrale di Phi, per estrazione.

        Serve al Gate 3 piu' che qui: l'Appendice A assume che Phi_m sia "real
        and stable", e la stabilita' di Phi_m e' quella di Phi (i moduli degli
        autovalori vanno alla potenza 1/3, quindi < 1 resta < 1).

        MISURATO SUL PROFILO q_b, pannello del Gate 2 (200 estrazioni), ed e' un
        risultato da portare al Gate 3:

            estrazioni con raggio spettrale < 1          0.0%
            massimo modulo, mediana                      1.0115  [1.006, 1.024]
            autovalori con |z| > 1, per estrazione       mediana 12 (su 150)
            autovalori con |z| > 1.05                    mediana 0
            dopo la radice cubica: raggio di Phi_m       mediana 1.0038, max 1.012

        COME SI LEGGE.  L'assunzione di stabilita' dell'Appendice A e'
        VIOLATA — ma di pochissimo, e nel modo in cui ci si aspetta che lo sia.
        Non e' una patologia: e' un GRAPPOLO DI RADICI QUASI UNITARIE.  Il VAR
        e' in log-LIVELLI su 30 serie non stazionarie, con il Minnesota centrato
        su un random walk e il sum-of-coefficients che spinge Pi verso zero: le
        radici unitarie sono IL PUNTO del modello, e cadono marginalmente dal
        lato esplosivo invece che esattamente su 1.  Nessun autovalore supera
        1.05, e la radice cubica avvicina ulteriormente a 1.

        LA CONSEGUENZA CONCRETA PER IL GATE 3, che e' la ragione per cui questo
        metodo esiste: il filtro del C-BVAR NON PUO' INIZIALIZZARE P_0 con la
        varianza non condizionata dello stato, perche' quella soluzione esiste
        solo se il sistema e' stabile (richiede (I - Phi (x) Phi)^-1).  Serve
        un'inizializzazione diffusa, o comunque a varianza grande.  Da decidere
        li', non qui.
        """
        return np.array([np.abs(np.linalg.eigvals(self.companion(s))).max()
                         for s in range(self.S)])

    def summary(self) -> str:
        rho = self.spectral_radius()
        return (f"{self.spec.model}-BVAR  n={self.n}  p={self.p}  "
                f"T={len(self.panel)}  S={self.S}\n"
                + self.draws.summary()
                + f"\n  raggio spettrale di Phi   mediana {np.median(rho):.4f}   "
                  f"stabili {float((rho < 1).mean()):.1%}")


# ─── 3. La stima ──────────────────────────────────────────────────────────────

def fit(
    spec: BVARSpec,
    panel: pd.DataFrame | None = None,
    *,
    as_of=None,
    n_draws: int = 1000,
    burn: int = 500,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    **kwargs,
) -> QBVARFit:
    """
    Stima il Q-BVAR.  E' `core.sample()` piu' l'impacchettamento: nessuna
    matematica che non stia gia' nel Gate 1.

    Parameters
    ----------
    panel : pd.DataFrame | None
        Il pannello TRIMESTRALE, denso, in unita' del modello.  `None` lo
        costruisce con `estimation_panel(spec, as_of=as_of)`.
    as_of : date-like | None
        Usato SOLO se `panel is None`.  In un run real-time va sempre passato:
        senza, il campione di stima arriva a `estimation_end` qualunque sia la
        data del nowcast, che e' look-ahead.  Vedi `estimation_panel`.
    **kwargs
        Passati a `core.sample` (hyperprior, c, tune, target_acceptance).
    """
    if panel is None:
        panel = estimation_panel(spec, as_of=as_of)
    if panel.shape[1] != spec.n:
        raise ValueError(
            f"il pannello ha {panel.shape[1]} colonne ma lo spec ne attende "
            f"{spec.n}"
        )
    draws = sample(spec, panel.to_numpy(dtype=float), n_draws=n_draws, burn=burn,
                   rng=rng, verbose=verbose, **kwargs)
    return QBVARFit(draws=draws, panel=panel, spec=spec)


# ─── 4. Il nowcast: il filtro sul bordo TRIMESTRALE ───────────────────────────
#
# QUESTA SEZIONE MANCAVA, ed e' il meccanismo della settimana 14 del paper.
#
# §3.2: "catching up to a certain extent only in week 14, when, at the close of
# the quarter, financial variables and the PMI and uncertainty indices for the
# full quarter become available."  Cioe': appena il trimestre CHIUDE, le sue
# celle MA3 diventano complete serie per serie — e il PIL non e' ancora uscito.
# Quella riga trimestrale, osservata in parte, e' informazione vera sul
# trimestre obiettivo, e il Q-BVAR ci puo' condizionare eccome.
#
# Il punto 3 dell'header diceva che il Q-BVAR "non puo' usare un trimestre
# incompleto, strutturalmente", e questo resta vero per la MEDIA PARZIALE (mai
# formata).  Ma confondeva due cose diverse: trimestre APERTO (settimane 1-13,
# MA incompleta, previsione pura — giusto) e trimestre CHIUSO col PIL mancante
# (settimana 14 in poi, MA completa per le serie gia' uscite — e li' si
# condiziona).  Misurato sul nostro pannello, riga 2008Q1:
#
#     as_of        serie osservate su 30      settimana
#     2008-03-28            1                    13     (trimestre aperto)
#     2008-04-04            4                    14     <- ISM x3 + Philly Fed
#     2008-04-11            7                    15     (+ occupazione)
#     2008-04-18           16                    16     (+ CPI, INDPRO, ...)
#
# Sono esattamente "financial variables and the PMI" del paper.
#
# COME LO FANNO LORO — STEP4_QBVAR.m r.144-150, ramo di riuso:
#
#     X = X(3:3:end,:);                             % pannello TRIMESTRALE
#     endEstimT = find(~isnan(sum(X,2)),1,'last');  % ultima riga PIENA
#     [X_sm,logLik] = run_Ksmoother_FM(betaHat, sigmaHat, lags,
#                                      X(endEstimT-lags:end,:));
#     X_transf = [X(lags+1:endEstimT-1,:); X_sm];
#
# ed e' la STESSA struttura del C-BVAR (finestra ancorata a endEstimT, initX
# dalle p righe precedenti, P_0 di lyapunov_symm, storia osservata anteposta),
# solo sulla companion TRIMESTRALE invece che su Phi_m.  Nessuna radice cubica:
# il sistema e' gia' quello stimato.

@dataclass
class QBVARNowcast:
    """
    L'uscita del nowcast Q-BVAR: le estrazioni del bordo trimestrale.

    Stessa convenzione dei due pezzi del C-BVAR (`cbvar.CBVARNowcast`): prima di
    `endEstimT` c'e' il dato osservato, da li' in poi quel che ha prodotto il
    simulation smoother.

    Attributes
    ----------
    draws : (S, W, n)     livelli trimestrali in unita' del modello, la finestra
    index : DatetimeIndex lungo W
    history : (H, n)      le righe osservate che precedono la finestra
    hist_index : DatetimeIndex lungo H
    """

    draws: np.ndarray
    index: pd.DatetimeIndex
    spec: BVARSpec = field(repr=False)
    history: np.ndarray | None = field(repr=False, default=None)
    hist_index: pd.DatetimeIndex | None = field(repr=False, default=None)

    @property
    def S(self) -> int:
        return int(self.draws.shape[0])

    @property
    def full_index(self) -> pd.DatetimeIndex:
        if self.hist_index is None or len(self.hist_index) == 0:
            return self.index
        return self.hist_index.append(self.index)

    def growth(self, series_id: str = "GDPC1") -> pd.DataFrame:
        """
        Crescita trimestrale annualizzata `100*((x_t/x_{t-1})^4 - 1)`.

        **t-1, non t-3**: qui ogni riga E' un trimestre.  E' la riga di
        STEP4_QBVAR.m che chiude il ciclo:

            X_transf(2:end,qSeries) = 100*((X_transf(2:end,qSeries)./
                                            X_transf(1:end-1,qSeries)).^4-1);
        """
        j = list(self.spec.series).index(series_id)
        col = self.draws[:, :, j]
        if self.history is not None and len(self.history):
            hist = np.broadcast_to(self.history[:, j], (self.S, len(self.history)))
            col = np.concatenate([hist, col], axis=1)
        lv = np.exp(col) if self.spec.transform[j] == "log" else col
        g = 100.0 * ((lv[:, 1:] / lv[:, :-1]) ** 4 - 1.0)
        return pd.DataFrame(g.T, index=self.full_index[1:])

    def summary(self, series_id: str = "GDPC1") -> str:
        q = self.growth(series_id).quantile([0.05, 0.5, 0.95], axis=1).T
        righe = "\n".join(
            f"    {d.date()}   {q.loc[d, 0.5]:7.2f}   "
            f"[{q.loc[d, 0.05]:7.2f}, {q.loc[d, 0.95]:7.2f}]" for d in q.index[-4:])
        h = 0 if self.history is None else len(self.history)
        return (f"Q-BVAR nowcast  S={self.S}  finestra {len(self.index)} trimestri "
                f"({self.index[0].date()} - {self.index[-1].date()})"
                f" + {h} di storia\n"
                f"  {series_id}, crescita annualizzata:\n{righe}")


def target_quarter_end(index: pd.DatetimeIndex, as_of) -> pd.Timestamp:
    """
    L'ultimo mese del trimestre OBIETTIVO — il `nowcastM` di STEP2/STEP4.

    E' il quarter-end del trimestre che contiene `as_of`; con `as_of=None` e'
    l'ultimo quarter-end dell'indice.  Funziona sia su un indice mensile che su
    uno trimestrale, perche' filtra sui mesi di fine trimestre in entrambi.
    """
    qe = index[index.month.isin(QUARTER_END_MONTHS)]
    if as_of is None:
        return qe[-1]
    t = pd.Timestamp(as_of)
    dopo = qe[qe >= t]
    return dopo[0] if len(dopo) else qe[-1]


def nowcast_window(
    spec: BVARSpec,
    as_of=None,
    *,
    horizon_q: int = 8,
    raw: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
):
    """
    Il pannello di NOWCAST trimestrale, la finestra di bordo e lo stato iniziale.

    E' il gemello trimestrale di `cbvar._edge`, con una differenza sola: qui i
    blocchi dello stato sono trimestri ADIACENTI, quindi `initX` e' allineato
    esatto — `[x_{t-1}, ..., x_{t-p}]` e' proprio lo stato al tempo t-1.  (Nel
    C-BVAR i blocchi distano 3 mesi e l'initX degli autori porta 2 mesi di
    gioco; qui no.)

    Returns
    -------
    (Y_win, f0, idx, history, hist_index)
    """
    p = spec.p
    panel = build_quarterly_panel(spec, as_of, raw=raw, metadata=metadata)
    # testa: come `estimation_panel`, si taglia alla prima riga piena (la
    # finestra della MA incompleta non e' un dato).
    pieno = panel.notna().all(axis=1)
    if not pieno.any():
        raise ValueError("nessuna riga completamente osservata: il Q-BVAR non "
                         "ha un punto a cui agganciare la finestra.")
    panel = panel.loc[pieno.idxmax():]
    # coda: `X = X(1:nowcastM+horizon,:)`, in trimestri invece che in mesi.
    fine = target_quarter_end(panel.index, as_of) + pd.offsets.QuarterEnd(horizon_q)
    panel = panel.loc[:fine]

    Y = panel.to_numpy(dtype=float)
    pieno = np.isfinite(Y).all(axis=1)
    i0 = int(np.flatnonzero(pieno)[-1])
    if i0 < p:
        raise ValueError(f"servono {p} trimestri pieni prima della finestra, "
                         f"ce ne sono {i0}.")
    f0 = np.concatenate([Y[i0 - 1 - j] for j in range(p)])   # piu' recente per primo
    if np.isnan(f0).any():
        raise ValueError("lo stato iniziale contiene NaN: i trimestri che "
                         "precedono la finestra non sono pienamente osservati.")
    return Y[i0:], f0, panel.index[i0:], Y[p:i0], panel.index[p:i0]


def nowcast(
    fit: QBVARFit,
    *,
    as_of=None,
    horizon_q: int = 8,
    raw: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    nugget: float = 1e-8,
    p0=None,
) -> QBVARNowcast:
    """
    Il nowcast del Q-BVAR: simulation smoother sul bordo trimestrale.

    NON e' piu' un'iterazione della companion.  Iterando, l'informazione del
    trimestre gia' chiuso — quello che alla settimana 14 ha ISM e finanziarie
    complete — non entrava mai, e il Q-BVAR restava piatto per un motivo
    implementativo invece che strutturale.  Qui le righe parzialmente osservate
    del bordo arrivano al filtro, che aggiorna sulle sole celle presenti
    (`build_selection_matrix`), esattamente come `run_Ksmoother_FM` degli autori.

    Resta vero, e va ripetuto, che il Q-BVAR NON forma mai una media parziale:
    nelle settimane 1-13 il trimestre obiettivo e' una riga tutta NaN e il
    filtro proietta.  La cecita' dentro il trimestre aperto e' intatta; quel che
    cambia e' il trimestre CHIUSO.

    Parameters
    ----------
    fit : QBVARFit         le estrazioni gia' fatte (stima as-of)
    horizon_q : int        trimestri di previsione oltre l'obiettivo.  Gli autori
                           usano `horizon = 24` MESI, cioe' 8 trimestri.
    nugget : float         varianza di osservazione relativa; vedi
                           `state_space.build_state_space`.  Loro usano R = 0.
    p0 : str | float       "lyapunov" oppure kappa per kappa*I.  None = il
                           default condiviso (`state_space.P0_DEFAULT`).  Qui
                           `initX` sono p trimestri PIENI e ADIACENTI, quindi
                           P_0 = 0 sarebbe la scelta "pulita" a priori — ma
                           misurata su tutta la griglia 2008-2010 costa 0.23 pp
                           di RMSE nel backcast.  La tabella e la spiegazione
                           stanno in `state_space.initial_covariance`.
    """
    from core.bvar.simsmoother import LinearGaussianSS, simulation_smoother
    from core.bvar.state_space import P0_DEFAULT, initial_covariance

    rng = np.random.default_rng() if rng is None else rng
    spec, n, p = fit.spec, fit.n, fit.p
    ns = fit.n_state

    Y_win, f0, idx, hist, hidx = nowcast_window(
        spec, as_of, horizon_q=horizon_q, raw=raw, metadata=metadata)

    Z = np.zeros((n, ns))
    Z[:, :n] = np.eye(n)
    draws = np.empty((fit.S, Y_win.shape[0], n))
    for s in range(fit.S):
        A = fit.companion(s)
        Q = fit.Omega(s)
        c = np.zeros(ns)
        c[:n] = fit.const[s]
        R = nugget * float(np.mean(np.abs(np.diag(fit.Sigma[s])))) * np.eye(n)
        P0, _ = initial_covariance(A, Q,
                                   kind=P0_DEFAULT if p0 is None else p0)
        lg = LinearGaussianSS(A=A, Q=Q, Z=Z, R=R, c=c, a0=f0, P0=P0)
        draws[s] = simulation_smoother(lg, Y_win, rng)[:, :n]

    return QBVARNowcast(draws=draws, index=idx, spec=spec,
                        history=hist, hist_index=hidx)


__all__ = [
    "three_month_average",
    "to_monthly_units",
    "build_monthly_panel",
    "to_quarterly",
    "build_quarterly_panel",
    "estimation_panel",
    "QBVARFit",
    "fit",
    "QBVARNowcast",
    "nowcast",
    "nowcast_window",
    "target_quarter_end",
    "QUARTER_END_MONTHS",
    "MA_WINDOW",
]
