"""
src/forecast/predictive_accuracy.py — i test di accuratezza predittiva.

PERCHE' QUESTO MODULO ESISTE
----------------------------
Sul 2007-2019 le cinque celle `fed_overlap` stanno fra 2.086 e 2.247 di RMSE;
sul 2007-2025 fra 4.98 e 5.17.  Sono differenze di centesimi, e `results.tex`
ci legge sopra un ordinamento ("the Student-$t$ leads before the pandemic").
Prima di affermare che una specificazione e' migliore — o che sono tutte
uguali, che e' un'affermazione altrettanto forte — serve un test.

Qui ce ne sono tre, e rispondono a domande DIVERSE.  Tenerle separate e' il
punto: il MCS non elimina nessuno E il Clark-West trova un vantaggio, e le due
cose non si contraddicono affatto (vedi PERCHE' TRE TEST, sotto).

L'UNITA' DI OSSERVAZIONE E' IL TRIMESTRE, NON LA SETTIMANA
-----------------------------------------------------------
Questa e' la decisione piu' importante del modulo, e non viene da nessun paper.

Il CSV di una cella ha ~1983 righe punteggiate, e sembra un campione enorme.
Non lo e': dentro un trimestre le ~30 previsioni settimanali guardano lo
STESSO PIL realizzato con informazione quasi identica.  Sono, in sostanza, lo
stesso errore ripetuto trenta volte.  Trattarle come osservazioni indipendenti
gonfia ogni statistica t di circa sqrt(30), cioe' fabbrica significativita' dal
nulla.

Percio' la perdita di un modello su un trimestre e' l'errore quadratico MEDIO
sulle settimane di quella fase, e il campione sono i 46 / 54 / 66 trimestri
della finestra.  Il campione e' quello delle tabelle: `window_sample` di
`compute_metrics`, stessa regola C, stesso asse standard.  Nessun campione
nuovo, nessuna seconda verita'.

E IL `common` DELLA TABELLA 1?  Domanda giusta, e la risposta e' misurata (il
2026-09-11, script `scratchpad/check_campione.py`).  La Tabella 1 affianca un
RMSE `free` e uno `common` e avverte che "only the common columns compare
across rows"; gli RMSE che escono di qui coincidono con la colonna `free`
(2.115 / 2.085 sul 2007-2019), e verrebbe il sospetto di star confrontando
numeri non confrontabili.

Non e' cosi', per una ragione precisa: applicare `metrics_tables.common_points`
al frame di QUESTO test non toglie NEMMENO UNA riga (1382 coppie prima e
dopo, RMSE identici alla terza cifra).  Le cinque celle e l'AR(2) coprono le
stesse coppie (trimestre, settimana), quindi fra loro `free` e `common` sono
la stessa cosa.  Il 1014 della colonna `common` nasce tutto dalla NY FED, che
non pubblica nelle settimane iniziali; e la Fed qui non e' parte in causa.

Controllato anche l'altro verso: restringendo alle sole settimane che la Fed
pubblica (le 1014 coppie, RMSE 1.993 / 1.978 = la colonna `common`), il
Clark-West resta identico alla seconda cifra in ogni cella tranne una, il
`forecast` dello `student_t`, che PASSA da 1.60 a 2.42.  Il verdetto non
dipende dalla scelta, e sul campione ristretto e' piu' forte.

PERCHE' TRE TEST E NON UNO
---------------------------
  DM-HLN      confronta DUE previsori NON annidati.  Nullo: pari accuratezza.
  Clark-West  confronta DUE modelli ANNIDATI.  Nullo: il piccolo e' il vero.
  MCS         guarda TUTTI insieme.  Restituisce l'insieme dei non-eliminabili.

Il MCS e' un test CONGIUNTO su cinque modelli con 46 osservazioni: ha pochissima
potenza, e infatti non esclude nessuno.  Il Clark-West e' un confronto A COPPIE
contro un nullo specifico: ne ha molta di piu', e il vantaggio lo trova.  Le due
letture stanno insieme, e Hansen, Lunde & Nason lo dicono esplicitamente
(p. 493): "a model is discarded only if it is found to be significantly inferior
to another model.  Models remain in the MCS until proven inferior, which has the
implication that NOT ALL MODELS IN THE MCS MAY BE JUDGED GOOD MODELS."

I PAPER, E CHE COSA PRENDIAMO DA CIASCUNO
------------------------------------------
Diebold & Mariano (1995), JBES 13(3), 253-263.
    L'impianto: si testa il differenziale di PERDITA, e il differenziale e' il
    primitivo — non il modello.  Statistica S1 = d_medio / sqrt(V(d_medio)).

Harvey, Leybourne & Newbold (1997), IJF 13(2), 281-291.
    Il DM originale RIFIUTA TROPPO in campione modesto: loro Tabella 1, con
    n=32 e h=4 un test nominale al 10% rifiuta il 21.3%.  Noi abbiamo n=46-66.
    Prendiamo la loro eq. (9) — il fattore di correzione — e i valori critici
    della t a n-1 gradi invece della normale.  E prendiamo la loro eq. (5):
    la varianza si stima col nucleo RETTANGOLARE troncato a h-1, non con
    i pesi di Bartlett.

Clark & West (2007), J. of Econometrics 138, 291-311.
    IL PAPER CHE CAMBIA LA RISPOSTA.  Il DM assume modelli non annidati; i
    nostri lo sono (`model.tex`: "three binary choices ... each of them is a
    RESTRICTION on the same system" — il gaussiano e' lo Student-t con
    1/nu = 0).  Sotto il nullo il modello grande stima parametri che in
    popolazione valgono zero: stimare rumore INIETTA rumore, quindi in campione
    finito il modello grande ha MSPE piu' alto anche quando i due sono identici
    in popolazione.  Il DM grezzo su modelli annidati e' percio' devastantemente
    sotto-dimensionato: loro riportano dimensione mediana SOTTO 0.01 per un test
    nominale 0.10, e scrivono "performed abysmally".  Prendiamo la loro
    eq. (2.1), che toglie l'handicap.

Hansen, Lunde & Nason (2011), Econometrica 79(2), 453-497.
    Il MCS: statistiche T_max e T_R (p. 465), regole di eliminazione coerenti
    (Prop. 1, p. 466), p-value come massimo corrente (Teorema 3, p. 463),
    moving-block bootstrap (p. 468).

Diebold (2015), JBES 33(1).
    Autorizza l'inquadramento: "in the DM framework the loss differential d12t
    is the primitive".  Testiamo previsioni, non strutture.

PERCHE' NON GIACOMINI-WHITE (2006), Econometrica 74(6), 1545-1578
-----------------------------------------------------------------
Sarebbe l'alternativa naturale, perche' condiziona sullo schema di stima invece
di assumerlo via.  Non si applica: GW richiede stimatori a MEMORIA LIMITATA
("our primary focus will be on rolling window methods"; valgono anche campione
fisso e pesi che scontano il passato).  Qui theta e' ri-stimato ogni mese su
finestra ESPANSIVA, quindi il loro quadro non copre questo caso.  Restiamo in
DM/HLN con i previsori come primitivi.

QUELLO CHE NON SI AFFERMA
--------------------------
  * CW e' costruito per modelli LINEARI stimati per OLS, e loro stessi avvertono
    che "certain of our asymptotic results may not generalize" al caso non
    lineare.  Il nostro e' EM su uno state-space.
  * 1/nu = 0 sta sul BORDO dello spazio parametrico, mentre in CW la restrizione
    gamma* = 0 e' all'interno.
  * Lo schema e' ricorsivo, e CW notano che il rolling e' meglio dimensionato.
Sono approssimazioni dichiarate, non ignorate.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["var_rettangolare", "dm_hln", "clark_west", "mcs",
           "perdite_per_trimestre", "ANNIDATI", "H_DI_FASE", "main"]


# ─── Chi annida chi ───────────────────────────────────────────────────────────
#
# I tre interruttori binari di `model.tex` (nu finito/infinito, rho libero/zero,
# pesi per-serie/condivisi).  Ogni cella e' una RESTRIZIONE dello stesso
# sistema, non un sistema diverso — per questo il confronto e' annidato e per
# questo serve Clark-West invece del DM.
#
# NON sono annidate fra loro le due varianti che differiscono SOLO nello schema
# dei pesi (per-serie contro condiviso): nessuna delle due si ottiene
# restringendo l'altra.  Li' vale il DM-HLN.
ANNIDATI: set[tuple[str, str]] = {
    ("gaussian", "gaussian_ar1"),
    ("gaussian", "student_t"),
    ("gaussian", "student_t_ar1"),
    ("gaussian", "student_t_ar1_shared"),
    ("gaussian_ar1", "student_t_ar1"),
    ("gaussian_ar1", "student_t_ar1_shared"),
    ("student_t", "student_t_ar1"),
}

#: L'orizzonte in TRIMESTRI di ogni fase, che e' l'`h` di HLN.
#:
#: HLN assumono che per previsioni ottimali a h passi gli errori seguano un
#: MA(h-1), quindi che le autocorrelazioni di ordine >= h del differenziale
#: siano nulle: e' `h` a fissare il troncamento della varianza e il fattore di
#: correzione.  Nelle fasi di nowcast e backcast il trimestre obiettivo e' gia'
#: cominciato e si sta prevedendo il trimestre corrente: h = 1.  In fase
#: `forecast` (settimane <= 0) il trimestre non e' ancora cominciato e l'origine
#: sta in quello precedente: h = 2.  La scelta e' conservativa — un h piu'
#: grande stringe la statistica e rende piu' difficile rifiutare.
H_DI_FASE: dict[str, int] = {
    "TUTTE": 1, "forecast": 2, "nowcast M1": 1,
    "nowcast M2": 1, "nowcast M3": 1, "backcast": 1,
}


# ─── (A) Diebold-Mariano modificato secondo Harvey-Leybourne-Newbold ─────────

def var_rettangolare(d: np.ndarray, h: int) -> float:
    """
    La varianza di `d_medio` secondo HLN (1997), eq. (1)-(2) e (5).

        V(d_medio) = n^-1 [ gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k ]

    con `gamma_k = n^-1 sum_t (d_t - d_medio)(d_{t-k} - d_medio)`.

    NUCLEO RETTANGOLARE, NON BARTLETT.  I pesi sono unitari fino a h-1 e zero
    dopo: e' il troncamento che segue dall'ipotesi MA(h-1), non uno stimatore
    HAC generico.  Con i pesi di Bartlett il termine k=1 entrerebbe dimezzato e
    la statistica in fase `forecast` (h=2) verrebbe diversa.

    Il rettangolare puo' uscire NEGATIVO quando l'autocovarianza campionaria e'
    abbastanza negativa; e' un difetto noto dello stimatore troncato, e il
    chiamante ricade sulla sola varianza contemporanea.
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    dm = d - d.mean()
    s = float(dm @ dm) / n
    for k in range(1, int(h)):
        s += 2.0 * float(dm[k:] @ dm[:-k]) / n
    return s / n


def dm_hln(perdita_a, perdita_b, h: int) -> tuple[float, float, int]:
    """
    Diebold-Mariano modificato, HLN (1997) eq. (9).  Torna `(stat, p, n)`.

        S* = sqrt( [n + 1 - 2h + h(h-1)/n] / n ) * S1,   confrontata con t_{n-1}

    NEGATIVO vuol dire che A e' migliore di B.  Il p-value e' a DUE code: il
    nullo e' pari accuratezza, e a priori non si sa chi vince.

    UN CONTROLLO CHE VALE UNA DIMOSTRAZIONE
    ---------------------------------------
    Per h = 1 questa statistica coincide ESATTAMENTE con la t ordinaria a un
    campione su `d` (con ddof=1).  Si verifica in due righe: con h=1 il fattore
    e' (n-1)/n e V(d_medio) = n^-2 * sum(d-d_medio)^2, da cui

        S* = d_medio * sqrt( n(n-1) / sum(d-d_medio)^2 ) = t_{n-1}.

    Non e' una coincidenza, ed e' il motivo per cui HLN propongono la t: a
    p. 284 scrivono che la modifica "would be precisely correct in the case of
    one-step ahead prediction if the d_t were normally distributed".  Il blocco
    `__main__` lo verifica numericamente.

    ATTENZIONE: valido per confronti NON ANNIDATI.  Su modelli annidati usare
    `clark_west`, per le ragioni nell'intestazione del modulo.
    """
    d = np.asarray(perdita_a, dtype=float) - np.asarray(perdita_b, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 8 or np.allclose(d, 0.0):
        return float("nan"), float("nan"), n
    v = var_rettangolare(d, h)
    if v <= 0.0:
        v = float(np.var(d)) / n
    s1 = d.mean() / np.sqrt(v)
    fattore = (n + 1 - 2 * h + h * (h - 1) / n) / n
    s = s1 * np.sqrt(max(fattore, 1e-12))
    return float(s), float(2.0 * stats.t.sf(abs(s), df=n - 1)), n


# ─── (B) Clark-West MSPE-adjusted ────────────────────────────────────────────

def clark_west(y, f_piccolo, f_grande, h: int) -> tuple[float, float, int]:
    """
    Clark & West (2007) eq. (2.1).  Torna `(t, p, n)`.

        f_t = (y - y1)^2 - [ (y - y2)^2 - (y1 - y2)^2 ]

    dove `y1` e' il modello PICCOLO (il nullo) e `y2` il GRANDE.  Il terzo
    termine e' l'aggiustamento: sottrae il rumore che il modello grande si
    infila da solo stimando parametri che sotto il nullo valgono zero.

    TEST A UNA CODA, con valori critici NORMALI: si rifiuta per valori positivi
    grandi (+1.282 al 10%, +1.645 al 5%).  L'alternativa e' "il modello grande
    ha MSPE minore", ed e' ordinata: e' la convenzione della letteratura sui
    modelli annidati fin da Ashley, Granger & Schmalensee (1980), e CW la
    seguono esplicitamente.

    La distribuzione asintotica NON e' normale sotto le loro condizioni
    preferite, ma CW mostrano per simulazione che i valori critici normali
    danno una dimensione "close to, but a little less than, nominal size":
    su 48 esperimenti il critico 0.10 da' dimensione effettiva fra 0.05 e 0.10,
    mediana 0.08.  Il test e' quindi LEGGERMENTE CONSERVATIVO, non liberale.

    Errore standard: CW prescrivono "for one step ahead forecast errors, the
    usual least squares standard error"; per errori autocorrelati uno stimatore
    HAC.  Qui: varianza campionaria con ddof=1 quando h=1, nucleo rettangolare
    troncato a h-1 altrimenti.

    CASO DEGENERE.  Se i due modelli producono previsioni IDENTICHE, `f_t` e'
    identicamente zero e la statistica e' 0/0: si torna NaN invece di uno zero,
    che si leggerebbe come "pari accuratezza accertata" quando invece non c'e'
    nessuna informazione.  Vale anche da sentinella: due celle che dovrebbero
    differire e danno NaN qui sono la stessa cella sotto due nomi.
    """
    y = np.asarray(y, dtype=float)
    f1 = np.asarray(f_piccolo, dtype=float)
    f2 = np.asarray(f_grande, dtype=float)
    ok = np.isfinite(y) & np.isfinite(f1) & np.isfinite(f2)
    y, f1, f2 = y[ok], f1[ok], f2[ok]
    n = y.size
    if n < 8:
        return float("nan"), float("nan"), n

    f = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)

    if h <= 1:
        v = float(np.var(f, ddof=1)) / n          # errore standard OLS
    else:
        v = var_rettangolare(f, h)
        if v <= 0.0:
            v = float(np.var(f, ddof=1)) / n
    if v <= 0.0:
        return float("nan"), float("nan"), n
    t = f.mean() / np.sqrt(v)
    return float(t), float(stats.norm.sf(t)), n


# ─── (C) Model Confidence Set ────────────────────────────────────────────────

def _indici_moving_block(n: int, B: int, ell: int, rng) -> np.ndarray:
    """
    Moving-block bootstrap di Kuensch, quello citato da HLN (2011, p. 468) via
    Goncalves & White (2005).  Blocchi contigui di lunghezza `ell` con inizio
    uniforme su {0, ..., n-ell}, SENZA avvolgimento circolare.

    Il blocco serve a conservare la dipendenza seriale del differenziale di
    perdita: con ell=1 si tornerebbe al bootstrap iid, che sotto dipendenza
    sottostima la varianza e quindi elimina troppo.
    """
    ell = max(1, min(int(ell), n))
    n_blocchi = int(np.ceil(n / ell))
    inizi = rng.integers(0, n - ell + 1, size=(B, n_blocchi))
    idx = inizi[:, :, None] + np.arange(ell)[None, None, :]
    return idx.reshape(B, -1)[:, :n]


def mcs(L: np.ndarray, nomi: list[str], alpha: float = 0.10,
        B: int = 5000, ell: int = 2, stat: str = "Tmax",
        rng=None) -> pd.Series:
    """
    Model Confidence Set, Hansen, Lunde & Nason (2011), sez. 3.

    `L` e' (n_trimestri x m) di perdite, piu' basso = meglio.  Torna il p-value
    MCS per ciascun modello: quelli con `p >= alpha` formano l'insieme.

    LE DUE STATISTICHE (p. 465), con la loro regola di eliminazione (Prop. 1):

      T_max = max_i t_i.      con  d_i. = L_medio_i - L_medio(insieme)
                              elimina argmax_i t_i.
      T_R   = max_ij |t_ij|   con  d_ij = L_medio_i - L_medio_j
                              elimina argmax_i max_j t_ij

    Si noti che `t_i.` confronta il modello con la MEDIA DELL'INSIEME corrente,
    non con un avversario: e' questo che rende la regola di eliminazione
    "coerente" col test nel senso della Def. 3 (p. 461), cioe' che impedisce di
    rifiutare l'ipotesi congiunta senza saper dire chi ne e' la causa.

    IL P-VALUE E' IL MASSIMO CORRENTE (Teorema 3, p. 462-463): il p di un
    modello eliminato al passo k e' il massimo dei p dei passi 1..k.  E' questo
    che lo rende monotono lungo l'ordine di eliminazione e interpretabile come
    un p-value classico.

    La varianza di `d_i.` viene dal bootstrap, non da una formula: e' il punto
    del disegno di HLN, perche' la distribuzione nulla di T_max dipende dalla
    matrice di correlazione fra modelli (Teorema 4) e il bootstrap la stima
    implicitamente.  Le righe si ricampionano INSIEME per tutti i modelli, cosi'
    che la dipendenza fra colonne resti intatta.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    L = np.asarray(L, dtype=float)
    n, m0 = L.shape
    idx = _indici_moving_block(n, B, ell, rng)
    Lb = L[idx].mean(axis=1)                       # (B, m0): medie bootstrap

    vivi = list(range(m0))
    p_mcs: dict[int, float] = {}
    p_corrente = 0.0

    while len(vivi) > 1:
        A, Ab = L[:, vivi], Lb[:, vivi]
        Lbar = A.mean(axis=0)
        if stat == "Tmax":
            d = Lbar - Lbar.mean()                          # d_i.
            db = Ab - Ab.mean(axis=1, keepdims=True)
            sd = np.sqrt(np.maximum(((db - d) ** 2).mean(axis=0), 1e-16))
            t = d / sd
            T, Tb = float(t.max()), (db - d).__truediv__(sd).max(axis=1)
            peggiore = int(np.argmax(t))
        elif stat == "TR":
            d = Lbar[:, None] - Lbar[None, :]                # d_ij
            db = Ab[:, :, None] - Ab[:, None, :]
            sd = np.sqrt(np.maximum(((db - d) ** 2).mean(axis=0), 1e-16))
            np.fill_diagonal(sd, np.inf)
            t = d / sd
            T = float(np.abs(t).max())
            Tb = np.abs((db - d) / sd).max(axis=(1, 2))
            peggiore = int(np.argmax(t.max(axis=1)))
        else:
            raise ValueError("stat dev'essere 'Tmax' o 'TR'")

        p = float((Tb >= T).mean())
        p_corrente = max(p_corrente, p)
        if p_corrente >= alpha:
            for j in vivi:
                p_mcs[j] = p_corrente
            break
        p_mcs[vivi[peggiore]] = p_corrente
        vivi.pop(peggiore)

    for j in vivi:                       # l'ultimo sopravvissuto, se si arriva a uno
        p_mcs.setdefault(j, 1.0)
    return pd.Series({nomi[j]: p_mcs[j] for j in range(m0)})


# ─── Costruzione delle perdite ───────────────────────────────────────────────

def perdite_per_trimestre(df: pd.DataFrame, metodi: list[str], fase: str,
                          colonna: str) -> pd.DataFrame | None:
    """
    Pivot (trimestre x metodo) della MEDIA di `colonna` dentro la fase.

    Il `dropna` finale tiene solo i trimestri che TUTTI i metodi hanno: un
    confronto vuole lo stesso campione per tutti, ed e' la stessa regola che
    `window_sample` applica a monte.  Le righe escono in ordine di tempo, che
    serve al bootstrap a blocchi (un blocco deve essere contiguo nel tempo, non
    nell'ordine alfabetico dei trimestri).
    """
    x = df[df["metodo"].isin(metodi)]
    if fase != "TUTTE":
        x = x[x["fase"] == fase]
    if x.empty:
        return None
    p = x.pivot_table(index="target_quarter", columns="metodo",
                      values=colonna, aggfunc="mean")
    p = p.reindex(columns=metodi).dropna()
    if p.empty:
        return None
    return p.loc[sorted(p.index, key=lambda q: (int(q[:4]), int(q[-1])))]


# ─── Il rapporto ─────────────────────────────────────────────────────────────

_SPEC = "fed_overlap"
_VARIANTI = ["gaussian", "gaussian_ar1", "student_t", "student_t_ar1",
             "student_t_ar1_shared"]
_BASE = "gaussian"


def main(spec: str = _SPEC, base: str = _BASE, out_dir: str | None = None) -> str:
    """Il rapporto completo sulle tre finestre cumulate.  Torna il testo."""
    from src.forecast import compute_metrics as cm
    from src import output_layout as layout

    metodi = ["%s/%s" % (spec, v) for v in _VARIANTI]
    base_m = "%s/%s" % (spec, base)
    breve = {m: m.split("/")[1] for m in metodi}
    breve["ar2"] = "ar2"

    df, _, _ = cm.load_long()
    df["se"] = df["errore"] ** 2

    R: list[str] = []
    P = R.append
    P("=" * 104)
    P("ACCURATEZZA PREDITTIVA — unita' = TRIMESTRE, campione comune (regola C)")
    P("DM-HLN (1997) eq.9  |  Clark-West (2007) eq.2.1, una coda  |  MCS (HLN 2011)")
    P("=" * 104)

    for win in layout.RMSE_PASSES:
        d, tenuti, _ = cm.window_sample(df, win)
        d = d[d["metodo"].isin(metodi + ["ar2"])]
        if d.empty:
            continue
        P("")
        P("#" * 104)
        P("# FINESTRA %s   —   %d trimestri nel campione comune" % (win, len(tenuti)))
        P("#" * 104)

        for fase in ["TUTTE"] + list(cm._PHASE_ORDER):
            se = perdite_per_trimestre(d, metodi + ["ar2"], fase, "se")
            if se is None or len(se) < 12:
                continue
            h = H_DI_FASE[fase]
            fc = perdite_per_trimestre(d, metodi, fase, "nowcast_bea")
            yy = perdite_per_trimestre(d, metodi, fase, "realizzato_bea")
            P("")
            P("--- fase: %s   n_trimestri = %d   h = %d ---" % (fase, len(se), h))
            P("%-24s%8s%9s%8s%9s%8s%8s" %
              ("metodo", "RMSE", "DM-HLN", "p_DM", "CW", "p_CW", "p_MCS"))
            pm = mcs(se[metodi].to_numpy(float), metodi, alpha=0.10,
                     B=5000, ell=2, rng=np.random.default_rng(7))
            for m in metodi + ["ar2"]:
                riga = "%-24s%8.3f" % (breve[m], float(np.sqrt(se[m].mean())))
                if m == base_m:
                    riga += "%9s%8s%9s%8s" % ("(base)", "", "", "")
                else:
                    s, p, _ = dm_hln(se[m].to_numpy(), se[base_m].to_numpy(), h)
                    riga += "%9.2f%8.3f" % (s, p)
                    if m != "ar2" and (breve[base_m], breve[m]) in ANNIDATI:
                        t, pc, _ = clark_west(yy[m].to_numpy(),
                                              fc[base_m].to_numpy(),
                                              fc[m].to_numpy(), h)
                        riga += "%9.2f%8.3f" % (t, pc)
                    else:
                        riga += "%9s%8s" % ("n.a.", "")
                riga += "%8s" % ("%.3f" % pm[m] if m in pm.index else "-")
                P(riga)

    testo = "\n".join(R)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "predictive_accuracy.txt"),
                  "w", encoding="utf-8") as fh:
            fh.write(testo)
    return testo


# ─── Self-test ───────────────────────────────────────────────────────────────
#
# Nessuna scrittura: il modulo si verifica su dati sintetici e non tocca gli
# artefatti veri (regola del progetto).

if __name__ == "__main__":
    rng = np.random.default_rng(20260911)
    ok = True

    def check(nome: str, cond: bool, extra: str = "") -> None:
        global ok
        ok &= bool(cond)
        print("  [%s] %s%s" % ("OK " if cond else "FAIL", nome,
                               ("  — " + extra) if extra else ""))

    print("1. DM-HLN a h=1 coincide con la t ordinaria (HLN p. 284)")
    for n in (16, 46, 66, 200):
        d = rng.standard_normal(n) + 0.15
        s, p, _ = dm_hln(d, np.zeros(n), h=1)
        t_rif = stats.ttest_1samp(d, 0.0)
        check("n=%3d" % n,
              abs(s - t_rif.statistic) < 1e-10 and abs(p - t_rif.pvalue) < 1e-12,
              "S*=%.6f  t=%.6f" % (s, t_rif.statistic))

    print("2. il fattore di correzione HLN eq.(9) stringe, e stringe di piu' con h")
    n = 46
    fatt = [np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) for h in (1, 2, 4, 8)]
    check("monotono decrescente in h", all(np.diff(fatt) < 0),
          "  ".join("%.4f" % f for f in fatt))
    check("< 1 sempre (il DM originale rifiuta troppo spesso)",
          all(f < 1 for f in fatt))
    check("h=1 -> sqrt((n-1)/n) esatto",
          abs(fatt[0] - np.sqrt((n - 1) / n)) < 1e-14)

    print("3. varianza rettangolare, non Bartlett")
    d = rng.standard_normal(80)
    dm_ = d - d.mean()
    g0 = float(dm_ @ dm_) / 80
    g1 = float(dm_[1:] @ dm_[:-1]) / 80
    check("h=1 -> gamma_0 / n", abs(var_rettangolare(d, 1) - g0 / 80) < 1e-14)
    check("h=2 -> (gamma_0 + 2 gamma_1) / n  [peso 1, non 0.5]",
          abs(var_rettangolare(d, 2) - (g0 + 2 * g1) / 80) < 1e-14)

    print("4. Clark-West: l'aggiustamento e' esattamente (y1 - y2)^2")
    n = 60
    y = rng.standard_normal(n)
    f1 = 0.3 * y + 0.2 * rng.standard_normal(n)
    f2 = 0.5 * y + 0.2 * rng.standard_normal(n)
    mspe1 = float(np.mean((y - f1) ** 2))
    mspe2 = float(np.mean((y - f2) ** 2))
    adj = float(np.mean((f1 - f2) ** 2))
    t_cw, _, _ = clark_west(y, f1, f2, h=1)
    f_medio = t_cw * np.sqrt(np.var((y - f1) ** 2
                                    - ((y - f2) ** 2 - (f1 - f2) ** 2),
                                    ddof=1) / n)
    check("f_medio = MSPE1 - MSPE2 + adj",
          abs(f_medio - (mspe1 - mspe2 + adj)) < 1e-9,
          "%.6f vs %.6f" % (f_medio, mspe1 - mspe2 + adj))
    check("l'aggiustamento e' sempre >= 0 (non puo' penalizzare il grande)",
          adj >= 0)

    print("5. Clark-West: previsioni identiche -> caso degenere, non uno zero finto")
    t0, p0, _ = clark_west(y, f1, f1.copy(), h=1)
    check("f_t identicamente zero -> NaN, non 0/0 mascherato",
          np.isnan(t0) and np.isnan(p0))

    print("6. Clark-West e' UNA CODA (p e' la coda destra)")
    t_hi, p_hi, _ = clark_west(y, f1, f2, h=1)
    check("p = 1 - Phi(t)", abs(p_hi - stats.norm.sf(t_hi)) < 1e-12)

    print("7. MCS: un modello palesemente peggiore viene eliminato")
    n, m = 400, 4
    base_l = rng.chisquare(3, size=(n, m)) * 0.1
    base_l[:, 3] += 3.0                                  # il quarto e' pessimo
    p_mcs = mcs(base_l, ["a", "b", "c", "PESSIMO"], alpha=0.10, B=2000, ell=2,
                rng=np.random.default_rng(3))
    check("PESSIMO fuori (p < 0.10)", p_mcs["PESSIMO"] < 0.10,
          "p=%.4f" % p_mcs["PESSIMO"])
    check("gli altri tre restano", (p_mcs[["a", "b", "c"]] >= 0.10).all())

    print("8. MCS: modelli scambiabili -> non si elimina nessuno")
    egual = rng.chisquare(3, size=(300, 4)) * 0.1
    p_eq = mcs(egual, list("abcd"), alpha=0.10, B=2000, ell=2,
               rng=np.random.default_rng(4))
    check("tutti dentro", (p_eq >= 0.10).all(), "min p = %.3f" % p_eq.min())

    print("9. MCS: il p-value e' il massimo corrente -> nessuno sotto il minimo")
    scal = rng.chisquare(3, size=(300, 5)) * 0.1 + np.arange(5) * 0.6
    p_sc = mcs(scal, list("vwxyz"), alpha=0.10, B=2000, ell=2,
               rng=np.random.default_rng(5))
    check("l'ordine dei p rispetta l'ordine delle perdite",
          p_sc["z"] <= p_sc["y"] + 1e-12 and p_sc["y"] <= p_sc["x"] + 1e-12,
          "  ".join("%s=%.3f" % (k, v) for k, v in p_sc.items()))

    print("10. blocchi del bootstrap: contigui e dentro il campione")
    idx = _indici_moving_block(50, 100, 4, rng)
    check("forma (B, n)", idx.shape == (100, 50))
    check("indici validi", idx.min() >= 0 and idx.max() <= 49)
    check("i blocchi sono contigui",
          bool((np.diff(idx[:, :4], axis=1) == 1).all()))

    print("\n%s" % ("TUTTI I CONTROLLI PASSATI" if ok
                    else "*** QUALCOSA NON TORNA ***"))
    raise SystemExit(0 if ok else 1)
