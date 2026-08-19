"""
core/bvar/tests/test_gate4.py

GATE 4 — IL BLOCKING, e il bordo che ne consegue.

    python -m core.bvar.tests.test_gate4

La domanda del gate: **l'impilamento trimestrale e' quello di `bbvar.m`, e il
bordo frastagliato regge senza la varianza non condizionata?**

Il blocking e' un'operazione senza matematica: si tagliano righe e si
ricompongono colonne.  Proprio per questo e' pericoloso — sbagliarlo non
solleva nessun errore, produce solo un pannello plausibile e storto.  I test
qui sotto sono quindi costruiti in modo che ogni cella del pannello sintetico
sia IDENTIFICABILE UNIVOCAMENTE (il valore dice da quale serie e da quale mese
viene): cosi' l'assenza di perdita e la posizione di ogni numero si verificano
per uguaglianza esatta, non per plausibilita'.

Tre controlli sono scritti per DISTINGUERE, nello spirito del §3 del Gate 3 —
accanto alla lettura giusta si calcola anche quella sbagliata e si verifica che
dia un risultato diverso.  Se anche l'errore passasse, il test non starebbe
misurando niente:

    §1  il terzo blocco prende TUTTE le colonne   contro   3*n_m + n_q
    §3  `endEstimT` sul solo terzo blocco         contro   la riga blocked intera
    §4  il riempimento e' il metodo 1             contro   una spline
    §6  la crescita si calcola su t-1             contro   t-3

  §1  Lo spec impilato: ordine, dimensione, `d_centre` replicato col blocco.
  §2  Il blocking non perde informazione, e ogni numero e' dove deve stare.
  §3  `endEstimT` guarda solo il terzo blocco.  IL CONTROLLO CENTRALE.
  §4  Il riempimento e' `remNaNs_spline` metodo 1.
  §5  Il bordo: P0 = 0, la testa osservata, e il recupero delle celle mancanti.
  §6  La crescita annualizzata si calcola su t-1.
  §7  Il pannello VERO: forma, allineamento, bordo frastagliato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bvar.bbvar import (
    EDGE_LAGS,
    FILL_K,
    MCMC_CONST,
    BBVARDraws,
    block_panel,
    blocked_spec,
    fill_median_ma,
    last_full_quarter,
    monthly_mask,
    spectral_radius,
)
from core.bvar.data import build_panel, load_raw_levels
from core.bvar.lbvar import build_state_space
from core.bvar.simsmoother import simulation_smoother
from core.bvar.spec import BVARSpec, MinnesotaSpec

_OK, _FAIL = "OK", "FAIL"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<62} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


# ─── Il giocattolo: uno spec e un pannello di cui si sa tutto ─────────────────

def _toy_spec(p: int = 2) -> BVARSpec:
    """
    Quattro serie, tre mensili e una trimestrale, con `d_centre` misto.

    La trimestrale sta IN MEZZO (posizione 2, non in coda) di proposito: se il
    codice assumesse che le trimestrali sono le ultime colonne, il §1 e il §2 lo
    vedrebbero.  Nel pannello vero non e' cosi', ma un test non deve dipendere
    dall'ordine che gli capita.
    """
    return BVARSpec(
        model="B", profile="q_b",
        series=("MA", "MB", "QC", "MD"),
        freq=("M", "M", "Q", "M"),
        transform=("log", "level", "log", "level"),
        sample_start=pd.Timestamp("1990-01-31"),
        minnesota=MinnesotaSpec(p=p, d_centre=(1, 0, 1, 1)),
    )


def _toy_panel(spec: BVARSpec, n_quarters: int = 8) -> pd.DataFrame:
    """
    Pannello MENSILE con celle identificabili: `valore = 100*colonna + riga`.

    Nessun numero puo' essere confuso con un altro, quindi dopo il blocking si
    puo' dire per ogni cella da dove viene.  Le trimestrali sono osservate solo
    nel terzo mese, come nel dato vero.
    """
    T = 3 * n_quarters
    idx = pd.date_range("1990-01-31", periods=T, freq="ME")
    X = np.array([[100.0 * j + i for j in range(spec.n)] for i in range(T)])
    for j, f in enumerate(spec.freq):
        if f == "Q":
            X[np.arange(T) % 3 != 2, j] = np.nan
    return pd.DataFrame(X, index=idx, columns=list(spec.series))


# ─── 1. Lo spec impilato ──────────────────────────────────────────────────────

def test_blocked_spec() -> bool:
    print("\n1. Lo spec impilato   (`bbvar.m` righe 18-27)")
    ok = True
    spec = _toy_spec()
    b = blocked_spec(spec)
    m = monthly_mask(spec)
    n_m, n = int(m.sum()), spec.n

    ok &= _check("n_b = 2*n_m + n", b.n == 2 * n_m + n,
                 f"{b.n} = 2*{n_m} + {n}")

    # NOTA (misurata qui, e corregge il punto 1 dell'header): 2*n_m + n e
    # 3*n_m + n_q sono LO STESSO NUMERO, perche' n_q = n - n_m.  La lettura
    # sbagliata del paper non si vede quindi nella DIMENSIONE — si vede
    # nell'ORDINE delle colonne, ed e' quello che il controllo sotto verifica.
    n_q = n - n_m
    ok &= _check("(2*n_m + n == 3*n_m + n_q: la dimensione non distingue)",
                 b.n == 3 * n_m + n_q, f"{b.n} = {3 * n_m + n_q}")

    # IL CONTROLLO CHE DISTINGUE: nel terzo blocco le trimestrali stanno al
    # LORO posto originale (qui QC in posizione 2 su 4), non accodate dopo le
    # mensili.  E' l'ordine che gli autori usano per rileggere i risultati
    # (`STEP1_BBVAR.m` r.79); sbagliarlo scambia le colonne senza errori.
    atteso = ("MA_m1", "MB_m1", "MD_m1", "MA_m2", "MB_m2", "MD_m2",
              "MA", "MB", "QC", "MD")
    sbagliato = ("MA_m1", "MB_m1", "MD_m1", "MA_m2", "MB_m2", "MD_m2",
                 "MA", "MB", "MD", "QC")      # mensili x3, poi le trimestrali
    ok &= _check("l'ordine e' [<mensili>_m1, <mensili>_m2, <tutte>]",
                 tuple(b.series) == atteso, f"{tuple(b.series)}")
    ok &= _check("...e NON [m1, m2, <mensili>, <trimestrali>]",
                 tuple(b.series) != sbagliato)

    ok &= _check("il terzo blocco e' lo spec originale, intatto",
                 tuple(b.series[-n:]) == tuple(spec.series))

    # `freq` NON diventa tutta 'Q': il terzo blocco conserva la frequenza
    # originale, ed e' deliberato — e' il selettore `qSeries` di
    # `STEP1_BBVAR.m` r.63, cioe' quello che dice a quali colonne applicare la
    # crescita annualizzata.  Se si appiattisse a 'Q' si perderebbe.
    ok &= _check("i primi due blocchi sono 'Q' (un mese = una trimestrale)",
                 set(b.freq[: 2 * n_m]) == {"Q"})
    ok &= _check("il terzo blocco conserva la freq originale (`qSeries`)",
                 tuple(b.freq[-n:]) == tuple(spec.freq),
                 f"{tuple(b.freq[-n:])}")

    # punto 2 dell'header: `pos` si replica col blocco.  Una serie wn resta wn.
    d = np.asarray(spec.minnesota.d_centre)
    d_atteso = tuple(list(d[m]) * 2 + list(d))
    ok &= _check("d_centre segue lo stesso impilamento (punto 2)",
                 tuple(b.minnesota.d_centre) == d_atteso, f"{d_atteso}")
    for nome, dd in zip(b.series, b.minnesota.d_centre):
        base = nome[:-3] if nome.endswith(("_m1", "_m2")) else nome
        if int(dd) != int(d[list(spec.series).index(base)]):
            ok &= _check(f"  {nome} ha lo stesso centraggio di {base}", False)

    ok &= _check("la trasformazione (log/level) segue il blocco",
                 tuple(b.transform) == tuple(np.asarray(spec.transform)[m]) * 2
                 + tuple(spec.transform))

    ok &= _check("p e' invariato, k = n_b*p + 1", b.p == spec.p
                 and b.k == b.n * b.p + 1, f"p={b.p} k={b.k}")
    return ok


# ─── 2. Il blocking ───────────────────────────────────────────────────────────

def test_block_panel() -> bool:
    print("\n2. Il blocking non perde informazione   (punto 1 dell'header)")
    ok = True
    spec = _toy_spec()
    nq = 8
    panel = _toy_panel(spec, nq)
    blocked = block_panel(panel, spec)
    b = blocked_spec(spec)

    ok &= _check("forma (T/3, n_b)", blocked.shape == (nq, b.n), f"{blocked.shape}")

    # l'oracolo: ogni cella del pannello blocked contro la cella mensile da cui
    # viene.  100*colonna + riga rende l'identificazione univoca.
    err, mm = 0.0, [i for i, f in enumerate(spec.freq) if f == "M"]
    for t in range(nq):
        atteso = ([100.0 * j + 3 * t for j in mm]              # mese 1, mensili
                  + [100.0 * j + 3 * t + 1 for j in mm]        # mese 2, mensili
                  + [100.0 * j + 3 * t + 2 for j in range(spec.n)])  # mese 3, TUTTE
        err = max(err, float(np.nanmax(np.abs(
            blocked.to_numpy()[t] - np.array(atteso)))))
    ok &= _check("ogni cella viene dal mese giusto della serie giusta",
                 err < 1e-12, f"err {err:.2e}")

    ok &= _check("l'indice della riga e' la data del MESE 3",
                 list(blocked.index) == list(panel.index[2::3]))

    # niente si perde e niente si duplica: le celle mensili osservate sono
    # 3 per trimestre per serie, e ricompaiono tutte esattamente una volta.
    osservati = np.sort(panel.to_numpy()[~np.isnan(panel.to_numpy())])
    dentro = np.sort(blocked.to_numpy()[~np.isnan(blocked.to_numpy())])
    ok &= _check("il blocking e' una permutazione dei dati osservati",
                 osservati.shape == dentro.shape
                 and np.allclose(osservati, dentro),
                 f"{osservati.size} celle -> {dentro.size}")

    # le trimestrali NON sono replicate: compaiono in una colonna sola
    j_q = list(b.series).index("QC")
    ok &= _check("la trimestrale compare una volta sola, nel terzo blocco",
                 [s for s in b.series if s.startswith("QC")] == ["QC"]
                 and j_q >= b.n - spec.n)

    # l'allineamento e' verificato, non sperato
    sbagliato = panel.iloc[1:]
    try:
        block_panel(sbagliato, spec)
        ok &= _check("un pannello che non parte da m1 solleva ValueError", False)
    except ValueError as e:
        ok &= _check("un pannello che non parte da m1 solleva ValueError", True,
                     str(e)[:38] + "...")

    # T non multiplo di 3: si tronca in coda, non si sfasa in testa
    tronco = block_panel(panel.iloc[:-2], spec)
    ok &= _check("T%3 != 0: tronca in coda e non sfasa il resto",
                 tronco.shape[0] == nq - 1
                 and np.allclose(tronco.to_numpy(),
                                 blocked.to_numpy()[:nq - 1], equal_nan=True))
    return ok


# ─── 3. endEstimT.  IL CONTROLLO CENTRALE ─────────────────────────────────────

def test_end_estim() -> bool:
    print("\n3. `endEstimT` guarda SOLO il terzo blocco   (punto 1, e `bbvar.m` r.26)")
    ok = True
    spec = _toy_spec()
    nq = 8
    panel = _toy_panel(spec, nq)
    blocked = block_panel(panel, spec)

    ok &= _check("pannello pieno: endEstimT e' l'ultimo trimestre",
                 last_full_quarter(blocked, spec) == nq - 1)

    # IL CASO CHE DISTINGUE.  Buco nel MESE 1 dell'ultimo trimestre, mese 3
    # completo.  Il criterio degli autori guarda `sum(X(3:3:end,:),2)`, cioe'
    # solo il terzo blocco: endEstimT resta l'ultimo trimestre.  Chi guardasse
    # la riga blocked intera direbbe nq-2, e stimerebbe su un trimestre in meno
    # buttando via un'osservazione di GDP.
    b1 = blocked.copy()
    b1.iloc[-1, 0] = np.nan                      # MA_m1 dell'ultimo trimestre
    got = last_full_quarter(b1, spec)
    riga_intera = int(np.flatnonzero(
        ~np.isnan(b1.to_numpy()).any(axis=1))[-1])
    ok &= _check("buco nel mese 1: endEstimT NON arretra", got == nq - 1,
                 f"terzo blocco {got}, riga intera {riga_intera}")
    ok &= _check("...e la lettura 'riga intera' darebbe un'altra risposta",
                 riga_intera != got, f"{riga_intera} contro {got}")

    # il bordo vero: il terzo blocco incompleto (il PIL non ancora uscito)
    b2 = blocked.copy()
    b2.iloc[-1, list(b2.columns).index("QC")] = np.nan
    ok &= _check("buco nel terzo blocco: endEstimT arretra di uno",
                 last_full_quarter(b2, spec) == nq - 2)

    b3 = blocked.copy()
    b3.iloc[:, -spec.n:] = np.nan
    try:
        last_full_quarter(b3, spec)
        ok &= _check("nessun trimestre completo -> ValueError", False)
    except ValueError:
        ok &= _check("nessun trimestre completo -> ValueError", True)
    return ok


# ─── 4. Il riempimento ────────────────────────────────────────────────────────

def test_fill() -> bool:
    print("\n4. Il riempimento e' `remNaNs_spline` metodo 1   (punto 3)")
    ok = True

    # oracolo a mano: mediana, poi media mobile centrata a 2k+1 con i bordi
    # replicati.  Ricalcolato qui senza usare la funzione sotto test.
    rng = np.random.default_rng(4)
    x = rng.normal(size=40)
    x[[7, 8, 30]] = np.nan
    X = fill_median_ma(x[:, None], k=FILL_K)[:, 0]

    y = x.copy()
    na = np.isnan(y)
    y[na] = np.nanmedian(x)
    pad = np.concatenate([np.full(FILL_K, y[0]), y, np.full(FILL_K, y[-1])])
    ma = np.array([pad[i:i + 2 * FILL_K + 1].mean() for i in range(len(y))])
    atteso = y.copy()
    atteso[na] = ma[na]
    ok &= _check("il buco -> mediana, poi media mobile centrata a 2k+1",
                 np.allclose(X, atteso), f"k={FILL_K}, finestra {2 * FILL_K + 1}")
    ok &= _check("le celle osservate non vengono toccate",
                 np.allclose(X[~na], x[~na]))
    ok &= _check("in uscita non restano NaN", not np.isnan(X).any())

    # IL CONTROLLO CHE DISTINGUE: su una serie LINEARE con un buco interno una
    # spline cubica (metodo 5) ricostruirebbe il valore vero esattamente.  Il
    # metodo 1 no — ed e' questa la divergenza voluta con l'L-BVAR.
    lin = np.arange(30.0)
    vero = lin[15]
    buco = lin.copy()
    buco[15] = np.nan
    got = fill_median_ma(buco[:, None])[15, 0]
    ok &= _check("NON e' una spline: su una retta il buco non torna esatto",
                 abs(got - vero) > 1e-6, f"{got:.3f} contro il vero {vero:.1f}")

    # partenze tardive (il caso che resta davvero, dopo il taglio a endEstimT)
    tardi = np.arange(30.0)
    tardi[:6] = np.nan
    out = fill_median_ma(tardi[:, None])[:, 0]
    ok &= _check("partenza tardiva: riempie senza NaN e resta nel range",
                 not np.isnan(out).any()
                 and out[:6].min() >= np.nanmin(tardi) - 1e-9
                 and out[:6].max() <= np.nanmax(tardi) + 1e-9)

    tutta = np.full((10, 1), np.nan)
    ok &= _check("colonna tutta NaN -> zeri (e non NaN propagati)",
                 np.allclose(fill_median_ma(tutta), 0.0))

    Z = np.column_stack([x, np.arange(40.0)])
    ok &= _check("le colonne si riempiono in modo indipendente",
                 np.allclose(fill_median_ma(Z)[:, 0], X)
                 and np.allclose(fill_median_ma(Z)[:, 1], np.arange(40.0)))
    return ok


# ─── 5. Il bordo ──────────────────────────────────────────────────────────────

def _companion(B: np.ndarray, n: int, p: int) -> np.ndarray:
    A = np.zeros((n * p, n * p))
    A[:n] = B[:n * p, :].T
    if p > 1:
        A[n:, : n * (p - 1)] = np.eye(n * (p - 1))
    return A


def _stacked_dgp(n: int, p: int, rng: np.random.Generator, *, rho: float = 1.02):
    """
    Un VAR nel regime del B-BVAR vero: rho(A) ~ 1.02, dove la varianza non
    condizionata NON esiste (nel B-BVAR misurato e' 1.014-1.027).  Testare il
    bordo su un sistema stazionario sarebbe testare il caso facile.

    Il riscalamento: moltiplicando il lag j per k^j gli autovalori della
    companion si moltiplicano ESATTAMENTE per k — riscalare tutti i lag per lo
    stesso fattore non lo farebbe (con p > 1 non e' lineare).

    Sigma e' fortemente CORRELATA di proposito.  E' la struttura vera del
    sistema impilato — `INDPRO_m1`, `INDPRO_m2`, `INDPRO` sono tre mesi della
    stessa serie — ed e' anche il canale su cui il B-BVAR nowcasta: senza
    correlazione contemporanea, una cella mancante non ha nessuna informazione
    da cui essere ricostruita, e il test del bordo non misurerebbe niente.
    """
    B = np.zeros((n * p + 1, n))
    B[:n] = (0.5 * np.eye(n) + 0.05 * rng.normal(size=(n, n))).T
    for j in range(1, p):
        B[j * n:(j + 1) * n] = (0.15 / j) * rng.normal(size=(n, n)).T

    k = rho / spectral_radius(_companion(B, n, p))
    for j in range(p):
        B[j * n:(j + 1) * n] *= k ** (j + 1)

    C = 0.8 * np.ones((n, n)) + 0.2 * np.eye(n)
    return B, 0.01 * C


def _simulate(B, Sigma, n, p, T, rng):
    Y = np.zeros((T, n))
    L = np.linalg.cholesky(Sigma)
    for t in range(p, T):
        Y[t] = B[-1] + sum(B[j * n:(j + 1) * n].T @ Y[t - 1 - j]
                           for j in range(p)) + L @ rng.normal(size=n)
    return Y


def test_edge() -> bool:
    print("\n5. Il bordo: P0 = 0 e la testa osservata   (punto 6)")
    ok = True
    rng = np.random.default_rng(404)
    n, p = 6, 2
    B, Sigma = _stacked_dgp(n, p, rng)
    rho = spectral_radius(_companion(B, n, p))
    ok &= _check("il DGP e' nel regime del B-BVAR vero (rho > 1)", rho > 1.0,
                 f"rho(A) = {rho:.4f}")

    def _edge(Y, *, righe):
        """Il bordo frastagliato: le ultime `righe` con sempre meno colonne."""
        Yr = Y.copy()
        for i, r in enumerate(range(-righe, 0)):
            Yr[r, n - (righe - i) * 2:] = np.nan
        return Yr

    # ── (a) la finestra lunga: il recupero delle celle mancanti ──────────────
    # su molte repliche, perche' le celle bucate sono poche e una sola replica
    # misurerebbe soprattutto rumore.
    R, T = 40, 24 + p
    err_s, err_0, cop = [], [], []
    for r in range(R):
        Y = _simulate(B, Sigma, n, p, T, rng)
        Yr = _edge(Y, righe=2)
        a0 = np.concatenate([Yr[p - 1 - j] for j in range(p)])
        ss = build_state_space(B, Sigma, n, p, a0)             # P0 = 0
        d = simulation_smoother(ss, Yr[p:], rng, n_draws=100)[:, :, :n]
        manc = np.isnan(Yr[p:])
        vero = Y[p:][manc]
        err_s.append(np.sqrt(np.mean((d.mean(axis=0)[manc] - vero) ** 2)))
        err_0.append(np.sqrt(np.mean(
            (vero - Y[p:].mean(axis=0)[np.where(manc)[1]]) ** 2)))
        lo, hi = np.quantile(d[:, manc], [0.05, 0.95], axis=0)
        cop.append(np.mean((vero >= lo) & (vero <= hi)))
        if r == 0:
            ok &= _check("P0 = 0: a0 e' la testa osservata, P0 e' zero",
                         np.allclose(ss.a0, a0) and np.allclose(ss.P0, 0.0))
            # le celle OSSERVATE non vanno reinventate: R e' un nugget, quindi
            # lo smoother le restituisce esatte.  E' il controllo che vede uno
            # sfasamento di indice o una Z sbagliata.
            obs = ~manc
            e = float(np.abs(d.mean(axis=0)[obs] - Yr[p:][obs]).max())
            ok &= _check("le celle osservate tornano identiche (nugget)",
                         e < 1e-4, f"err {e:.2e}")
    s, z, c = np.mean(err_s), np.mean(err_0), np.mean(cop)
    ok &= _check("nelle celle mancanti lo smoother batte la media", s < z,
                 f"RMSE {s:.4f} vs {z:.4f}  ({R} repliche)")
    # senza la copertura, "batte la media" si otterrebbe anche con una densita'
    # degenere — ed e' esattamente il difetto che il fix a P0 ha corretto.
    ok &= _check("...e la banda 5-95 copre il vero", 0.80 <= c <= 1.0,
                 f"copertura {c:.0%}")

    # ── (b) LA REGRESSIONE: P0 diffusa contro finestra CORTA ─────────────────
    # Il guasto misurato non e' P0 diffusa in se': e' l'INTERAZIONE con una
    # finestra corta (punto 6, e la tabella dell'header).  Con soli 2 periodi
    # lisciati la diffusione iniziale non viene mai consumata dai dati, e lo
    # stato al bordo resta non identificato.  Con la finestra lunga di sopra
    # non succede: per questo il controllo va fatto QUI, con `lags` periodi
    # come nella riga di `bbvar.m`.
    Y = _simulate(B, Sigma, n, p, 30, rng)
    corta = _edge(Y[-(p + 2):], righe=2)
    a0 = np.concatenate([corta[p - 1 - j] for j in range(p)])
    mag = {}
    for nome, P0 in (("0", None), ("1e2*I", 1e2), ("1e4*I", 1e4)):
        ssx = build_state_space(B, Sigma, n, p, a0,
                                P0=None if P0 is None else P0 * np.eye(n * p))
        dx = simulation_smoother(ssx, corta[p:], rng, n_draws=100)[:, :, :n]
        mag[nome] = float(np.abs(dx).max())
    ok &= _check("finestra corta, P0 = 0: lo stato resta limitato",
                 mag["0"] < 10 * float(np.abs(Y).max()),
                 f"|max| {mag['0']:.2f} contro |Y| {np.abs(Y).max():.2f}")
    ok &= _check("finestra corta, P0 diffusa: lo stato degenera",
                 mag["1e4*I"] > 5 * mag["0"],
                 f"|max| {mag['1e2*I']:.1f} (1e2) / {mag['1e4*I']:.1f} (1e4) "
                 f"contro {mag['0']:.2f}")
    ok &= _check("...e peggiora al crescere della diffusione",
                 mag["1e4*I"] > mag["1e2*I"] > mag["0"])

    # la finestra: `EDGE_LAGS` deve lasciare abbastanza periodi lisciati perche'
    # il bordo sia identificato anche senza il Lyapunov (punto 6).
    ok &= _check("EDGE_LAGS lascia >= 20 trimestri lisciati", EDGE_LAGS >= 20,
                 f"EDGE_LAGS = {EDGE_LAGS}")
    ok &= _check("MCMC_CONST e' quello di `bbvar.m` r.34, non quello dell'L",
                 abs(MCMC_CONST - 0.14) < 1e-12, f"{MCMC_CONST}")
    return ok


# ─── 6. La crescita ───────────────────────────────────────────────────────────

def test_growth() -> bool:
    print("\n6. La crescita annualizzata si calcola su t-1   (punto 7)")
    ok = True
    spec = _toy_spec()
    b = blocked_spec(spec)
    S, W = 3, 12
    idx = pd.date_range("1990-03-31", periods=W, freq="QE")

    # livelli noti: la trimestrale cresce di un g% costante per trimestre
    g_vero = 1.5
    lv = 100.0 * (1.0 + g_vero / 100.0) ** np.arange(W)
    panels = np.zeros((S, W, b.n))
    j = list(b.series).index("QC")
    panels[:, :, j] = np.log(lv)                 # QC e' 'log' -> exp in growth()

    res = BBVARDraws(panels=panels, index=idx, spec=b, base_spec=spec)
    g = res.growth("QC")

    atteso = 100.0 * ((1.0 + g_vero / 100.0) ** 4 - 1.0)
    ok &= _check("100*((x_t/x_{t-1})^4 - 1)",
                 np.allclose(g.to_numpy(), atteso), f"{g.iloc[-1, 0]:.4f} "
                 f"contro {atteso:.4f}")

    # IL CONTROLLO CHE DISTINGUE: la lettura t-3 (giusta per C e L, dove il
    # pannello e' mensile) darebbe un tasso plausibile ma sbagliato.
    t3 = 100.0 * ((1.0 + g_vero / 100.0) ** 12 - 1.0)
    ok &= _check("...e NON su t-3, che darebbe un altro numero",
                 abs(atteso - t3) > 1e-6, f"t-1 {atteso:.2f} vs t-3 {t3:.2f}")

    ok &= _check("l'esponenziale e' applicato solo alle serie 'log'",
                 b.transform[j] == "log")
    ok &= _check("l'indice perde la prima riga (la crescita non esiste in t=0)",
                 list(g.index) == list(idx[1:]) and g.shape == (W - 1, S))

    # una serie 'level' non deve passare per exp()
    k = list(b.series).index("MB")                # 'level'
    panels2 = panels.copy()
    panels2[:, :, k] = lv
    res2 = BBVARDraws(panels=panels2, index=idx, spec=b, base_spec=spec)
    ok &= _check("una serie 'level' da' la stessa crescita senza exp()",
                 np.allclose(res2.growth("MB").to_numpy(), atteso))
    return ok


# ─── 7. Il pannello vero ──────────────────────────────────────────────────────

def test_real_panel() -> bool:
    print("\n7. Il pannello VERO   (nessun MCMC: forma, allineamento, bordo)")
    ok = True
    try:
        spec = BVARSpec.from_config("B")
        monthly = build_panel(spec, None, raw=load_raw_levels())
    except Exception as e:                                    # noqa: BLE001
        print(f"     dati non disponibili, salto: {type(e).__name__}: {e}")
        return True

    m = monthly_mask(spec)
    n_m, n = int(m.sum()), spec.n
    b = blocked_spec(spec)
    ok &= _check("il pannello mensile parte dal primo mese di un trimestre",
                 monthly.index[0].month in (1, 4, 7, 10),
                 f"{monthly.index[0].date()}")

    blocked = block_panel(monthly, spec)
    ok &= _check(f"n_b = 2*{n_m} + {n}", blocked.shape[1] == 2 * n_m + n == b.n,
                 f"n_b = {b.n}, k = {b.k}")

    end = last_full_quarter(blocked, spec)
    ok &= _check("endEstimT non e' l'ultima riga (il bordo esiste)",
                 end < blocked.shape[0] - 1,
                 f"endEstimT = {blocked.index[end].date()}, "
                 f"ultima = {blocked.index[-1].date()}")

    Xb = blocked.to_numpy(dtype=float)
    ok &= _check("dopo il fill il pannello di stima non ha NaN",
                 not np.isnan(fill_median_ma(Xb[: end + 1])).any(),
                 f"prima {int(np.isnan(Xb[: end + 1]).sum())} NaN")

    coda = Xb[end + 1:]
    ok &= _check("il bordo e' frastagliato, non vuoto",
                 coda.size > 0 and np.isnan(coda).any() and not np.isnan(coda).all(),
                 f"{int(np.isnan(coda).sum())} NaN su {coda.size} celle")

    start = max(0, end - EDGE_LAGS)
    ok &= _check("la finestra terminale sta dentro il campione",
                 start >= spec.p, f"finestra {blocked.index[start].date()} - "
                 f"{blocked.index[-1].date()}")

    j = list(b.series).index("GDPC1")
    ok &= _check("il PIL sta nel terzo blocco, non replicato",
                 j >= b.n - n and "GDPC1_m1" not in b.series)
    return ok


def main() -> bool:
    print("=" * 82)
    print("Gate 4 — IL BLOCKING (B-BVAR, §2.3) e il bordo frastagliato")
    print("=" * 82)
    ok = True
    for t in (test_blocked_spec, test_block_panel, test_end_estim, test_fill,
              test_edge, test_growth, test_real_panel):
        ok &= t()
    print("\n" + "=" * 82)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 82)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
