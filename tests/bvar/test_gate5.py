"""
core/bvar/tests/test_gate5.py

GATE 5, BLOCCO 1 — il simulation smoother di Durbin-Koopman, IN ISOLAMENTO.

    python -m core.bvar.tests.test_gate5

Il documento di contesto lo impone: lo smoother va "tested in isolation first,
then inside the MCMC loop".  Questo file e' la prima meta'.

IL PRINCIPIO DEL TEST, ed e' la ragione per cui vale qualcosa.  Su un modello
lineare gaussiano PICCOLO la congiunta di (alpha_{1:T}, y_{1:T}) e' una normale
multivariata che si puo' costruire DENSAMENTE.  Quindi p(alpha | y) ha media e
covarianza in forma chiusa, calcolabili con algebra elementare e senza usare
niente del codice sotto test.  Si confrontano media e covarianza EMPIRICHE di
molte estrazioni dello smoother contro quelle ESATTE.

E' un oracolo, non un controllo di plausibilita': prende qualunque errore di
convenzione — un'intercetta contata due volte, una trasposta, un indice sfasato
di uno, il pattern dei mancanti non replicato — che un test "gira e non esplode"
lascerebbe passare.  E' lo stesso spirito dell'oracolo del Blocco 4 al Gate 1.

  §1  Lo smoother di media contro E[alpha|y] esatto.
  §2  Il simulation smoother: media E covarianza contro la congiunta esatta.
  §3  Dati mancanti — il caso che serve davvero all'L-BVAR.
  §4  Q singolare (companion di un VAR) e osservazione esatta.
  §5  La companion di un VAR con le trimestrali un mese su tre.
"""

from __future__ import annotations

import numpy as np

from core.bvar.simsmoother import (
    LinearGaussianSS,
    companion_predict,
    forward_pass,
    simulate_forward,
    simulation_smoother,
    smoothed_mean,
)
from core.kalman import kalman_predict

_OK, _FAIL = "OK", "FAIL"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<62} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


# ─── L'oracolo: la congiunta densa ────────────────────────────────────────────

def _joint(ss: LinearGaussianSS, T: int):
    """
    Media e covarianza della congiunta di alpha_{0:T-1} impilati, e la mappa
    verso y.  Costruite per ricorsione esplicita, senza usare il filtro.

    alpha_t = c + A alpha_{t-1} + eta_t,  alpha_{-1} ~ N(a0, P0)
    """
    ns = ss.ns
    m = np.empty((T, ns))
    a_prev_m = ss.a0
    for t in range(T):
        m[t] = ss.c + ss.A @ a_prev_m
        a_prev_m = m[t]

    # Cov(alpha_s, alpha_t) per ricorsione: V[t] = A V[t-1] A' + Q
    V = np.empty((T, ns, ns))
    Vp = ss.P0
    for t in range(T):
        V[t] = ss.A @ Vp @ ss.A.T + ss.Q
        Vp = V[t]

    S = np.zeros((T * ns, T * ns))
    for s in range(T):
        for t in range(s, T):
            # Cov(alpha_s, alpha_t) = V[s] (A')^(t-s)
            block = V[s] @ np.linalg.matrix_power(ss.A.T, t - s)
            S[s * ns:(s + 1) * ns, t * ns:(t + 1) * ns] = block
            S[t * ns:(t + 1) * ns, s * ns:(s + 1) * ns] = block.T
    return m.ravel(), S


def _exact_posterior(ss: LinearGaussianSS, Y: np.ndarray):
    """
    E[alpha|y] e Var[alpha|y] esatti, per condizionamento gaussiano sulla
    congiunta densa.  Le righe con NaN sono semplicemente escluse.
    """
    T, n = Y.shape
    ns = ss.ns
    mu, S = _joint(ss, T)

    # la mappa alpha -> y sulle sole osservazioni presenti
    righe, valori = [], []
    for t in range(T):
        for i in range(n):
            if not np.isnan(Y[t, i]):
                r = np.zeros(T * ns)
                r[t * ns:(t + 1) * ns] = ss.Z[i]
                righe.append(r)
                valori.append(Y[t, i])
    H = np.array(righe)                       # (m, T*ns)
    yv = np.array(valori)
    Rblk = np.zeros((len(valori), len(valori)))
    k = 0
    for t in range(T):
        for i in range(n):
            if not np.isnan(Y[t, i]):
                Rblk[k, k] = ss.R[i, i]
                k += 1

    Syy = H @ S @ H.T + Rblk
    Say = S @ H.T
    post_m = mu + Say @ np.linalg.solve(Syy, yv - H @ mu)
    post_S = S - Say @ np.linalg.solve(Syy, Say.T)
    return post_m.reshape(T, ns), post_S


def _toy(seed: int = 0, ns: int = 2, n: int = 1, *, exact_obs: bool = False,
         singular_Q: bool = False) -> LinearGaussianSS:
    rng = np.random.default_rng(seed)
    A = 0.5 * rng.normal(size=(ns, ns))
    A *= 0.85 / max(np.abs(np.linalg.eigvals(A)).max(), 1e-12)
    if singular_Q:
        Q = np.zeros((ns, ns))
        Q[0, 0] = 0.4
    else:
        L = np.tril(rng.normal(size=(ns, ns))) * 0.5
        np.fill_diagonal(L, np.abs(np.diag(L)) + 0.4)
        Q = L @ L.T
    Z = rng.normal(size=(n, ns)) if not exact_obs else np.eye(n, ns)
    R = (1e-8 if exact_obs else 0.3) * np.eye(n)
    return LinearGaussianSS(A=A, Q=Q, Z=Z, R=R,
                            c=0.2 * rng.normal(size=ns),
                            a0=rng.normal(size=ns),
                            P0=np.eye(ns) * 1.5)


# ─── 1. Lo smoother di media ──────────────────────────────────────────────────

def test_mean_smoother() -> bool:
    print("\n1. Lo smoother a disturbi contro E[alpha|y] esatto")
    ok = True
    for seed, ns, n in ((0, 2, 1), (1, 3, 2), (2, 4, 2)):
        ss = _toy(seed, ns, n)
        rng = np.random.default_rng(100 + seed)
        T = 9
        _, Y = simulate_forward(ss, T, rng)
        got = smoothed_mean(ss, forward_pass(ss, Y))
        want, _ = _exact_posterior(ss, Y)
        err = float(np.abs(got - want).max())
        ok &= _check(f"ns={ns} n={n} T={T}: media smussata esatta", err < 1e-8,
                     f"err {err:.2e}")
    return ok


# ─── 2. Il simulation smoother contro la congiunta ────────────────────────────

def test_simulation_smoother() -> bool:
    print("\n2. Il SIMULATION smoother: media e covarianza contro l'esatto")
    print("     (e' l'oracolo: media giusta ma covarianza sbagliata = imputazione")
    print("      singola travestita, ed e' l'errore che rovinerebbe il Gibbs)")
    ok = True
    ss = _toy(0, 2, 1)
    rng = np.random.default_rng(7)
    T, S = 7, 40000
    _, Y = simulate_forward(ss, T, rng)
    want_m, want_S = _exact_posterior(ss, Y)

    dr = simulation_smoother(ss, Y, rng, n_draws=S)      # (S, T, ns)
    flat = dr.reshape(S, -1)
    got_m = flat.mean(axis=0)
    got_S = np.cov(flat, rowvar=False)

    se = np.sqrt(np.diag(want_S) / S)
    z = np.abs(got_m - want_m.ravel()) / se
    ok &= _check("media empirica == media esatta (|z| < 4)", z.max() < 4.0,
                 f"|z|max {z.max():.2f}")

    rel = np.abs(got_S - want_S).max() / np.abs(want_S).max()
    ok &= _check("covarianza empirica == covarianza esatta", rel < 0.05,
                 f"scarto relativo {rel:.3f}")

    # controprova: la MEDIA smussata da sola avrebbe covarianza nulla.  Serve a
    # dimostrare che il test distingue davvero le due cose.
    var_ratio = float(np.mean(np.diag(got_S)) / np.mean(np.diag(want_S)))
    ok &= _check("...e la varianza NON e' zero (non e' imputazione singola)",
                 0.9 < var_ratio < 1.1, f"rapporto {var_ratio:.3f}")
    return ok


# ─── 3. Dati mancanti ─────────────────────────────────────────────────────────

def test_missing() -> bool:
    print("\n3. Con i dati MANCANTI — il caso che serve all'L-BVAR")
    ok = True
    ss = _toy(1, 3, 2)
    rng = np.random.default_rng(11)
    T, S = 8, 30000
    _, Y = simulate_forward(ss, T, rng)
    Y[1::3, 1] = np.nan            # una serie osservata un periodo su tre
    Y[-1, :] = np.nan              # e un bordo frastagliato in fondo

    want_m, want_S = _exact_posterior(ss, Y)
    got = smoothed_mean(ss, forward_pass(ss, Y))
    ok &= _check("la media smussata resta esatta coi buchi",
                 np.abs(got - want_m).max() < 1e-8,
                 f"err {np.abs(got - want_m).max():.2e}")

    dr = simulation_smoother(ss, Y, rng, n_draws=S).reshape(S, -1)
    se = np.sqrt(np.diag(want_S) / S)
    z = np.abs(dr.mean(axis=0) - want_m.ravel()) / se
    rel = np.abs(np.cov(dr, rowvar=False) - want_S).max() / np.abs(want_S).max()
    ok &= _check("media empirica esatta coi buchi (|z| < 4)", z.max() < 4.0,
                 f"|z|max {z.max():.2f}")
    ok &= _check("covarianza empirica esatta coi buchi", rel < 0.06,
                 f"scarto relativo {rel:.3f}")

    # il pattern dei mancanti DEVE essere replicato in y+: se non lo fosse, lo
    # smoother condizionerebbe su informazione che non abbiamo.
    _, yp = simulate_forward(ss, T, rng, pattern=Y)
    ok &= _check("simulate_forward replica il pattern dei NaN",
                 np.array_equal(np.isnan(yp), np.isnan(Y)))
    return ok


# ─── 4. Q singolare e osservazione esatta ─────────────────────────────────────

def test_singular() -> bool:
    print("\n4. Q SINGOLARE (companion) e osservazione quasi esatta")
    print("     e' la configurazione vera dell'L-BVAR, non un caso di comodo")
    ok = True
    ss = _toy(3, 4, 2, exact_obs=True, singular_Q=True)
    rng = np.random.default_rng(21)
    T, S = 8, 30000
    _, Y = simulate_forward(ss, T, rng)
    Y[1::3, 1] = np.nan

    want_m, want_S = _exact_posterior(ss, Y)
    got = smoothed_mean(ss, forward_pass(ss, Y))
    ok &= _check("media smussata esatta con Q singolare",
                 np.abs(got - want_m).max() < 1e-6,
                 f"err {np.abs(got - want_m).max():.2e}")

    dr = simulation_smoother(ss, Y, rng, n_draws=S).reshape(S, -1)
    rel = np.abs(np.cov(dr, rowvar=False) - want_S).max() / max(np.abs(want_S).max(), 1e-12)
    ok &= _check("covarianza esatta con Q singolare", rel < 0.08,
                 f"scarto relativo {rel:.3f}")
    ok &= _check("le estrazioni rispettano i dati osservati (R ~ 0)",
                 np.abs(dr.reshape(S, T, ss.ns)[:, ~np.isnan(Y[:, 0]), 0].mean(axis=0)
                        - Y[~np.isnan(Y[:, 0]), 0]).max() < 1e-3,
                 "componente 0")
    return ok


# ─── 5. La companion di un VAR con le trimestrali ─────────────────────────────

def test_var_companion() -> bool:
    print("\n5. La companion di un VAR: una serie osservata un mese su tre")
    ok = True
    rng = np.random.default_rng(31)
    n, p = 2, 3
    ns = n * p
    A1 = np.array([[0.6, 0.1], [0.05, 0.5]])
    A = np.zeros((ns, ns))
    A[:n, :n] = A1
    A[:n, n:2 * n] = 0.2 * np.eye(n)
    A[n:, :ns - n] = np.eye(ns - n)
    Q = np.zeros((ns, ns))
    Q[:n, :n] = np.array([[0.09, 0.02], [0.02, 0.04]])
    Z = np.zeros((n, ns)); Z[:, :n] = np.eye(n)
    ss = LinearGaussianSS(A=A, Q=Q, Z=Z, R=1e-10 * np.eye(n),
                          c=np.concatenate([[0.05, -0.02], np.zeros(ns - n)]),
                          a0=np.zeros(ns), P0=np.eye(ns))

    T, S = 12, 20000
    _, Y = simulate_forward(ss, T, rng)
    Y[[i for i in range(T) if i % 3 != 2], 1] = np.nan     # la "trimestrale"

    want_m, want_S = _exact_posterior(ss, Y)
    got = smoothed_mean(ss, forward_pass(ss, Y))
    ok &= _check("media smussata esatta sulla companion",
                 np.abs(got - want_m).max() < 1e-6,
                 f"err {np.abs(got - want_m).max():.2e}")

    dr = simulation_smoother(ss, Y, rng, n_draws=S)
    lat = [i for i in range(T) if i % 3 != 2]
    sd_lat = dr[:, lat, 1].std(axis=0).mean()
    sd_oss = dr[:, [i for i in range(T) if i % 3 == 2], 1].std(axis=0).mean()
    ok &= _check("i mesi LATENTI hanno incertezza, gli OSSERVATI no",
                 sd_lat > 1e-3 and sd_oss < 1e-3,
                 f"sd latenti {sd_lat:.4f}, sd osservati {sd_oss:.2e}")

    flat = dr.reshape(S, -1)
    rel = np.abs(np.cov(flat, rowvar=False) - want_S).max() / max(np.abs(want_S).max(), 1e-12)
    ok &= _check("covarianza esatta sulla companion", rel < 0.08,
                 f"scarto relativo {rel:.3f}")
    return ok


# ─── 6. La predizione companion-aware contro l'oracolo ────────────────────────

def test_companion_predict() -> bool:
    print("\n6. companion_predict contro kalman_predict (l'ORACOLO)")
    print("     una specializzazione vale solo se da' lo stesso risultato")
    ok = True
    rng = np.random.default_rng(41)
    for (n, p) in ((2, 3), (5, 4), (12, 6)):
        ns = n * p
        A = np.zeros((ns, ns))
        for j in range(p):
            A[:n, j * n:(j + 1) * n] = (0.4 / (j + 1)) * rng.normal(size=(n, n))
        A[n:, :ns - n] = np.eye(ns - n)
        L = np.tril(rng.normal(size=(n, n))) * 0.3
        np.fill_diagonal(L, np.abs(np.diag(L)) + 0.5)
        Q = np.zeros((ns, ns)); Q[:n, :n] = L @ L.T
        M = rng.normal(size=(ns, ns)); P = M @ M.T + np.eye(ns)
        f = rng.normal(size=ns)

        f1, P1 = kalman_predict(f, P, A, Q)
        f2, P2 = companion_predict(f, P, A, Q, n)
        ef = float(np.abs(f1 - f2).max())
        eP = float(np.abs(P1 - P2).max() / max(np.abs(P1).max(), 1e-300))
        ok &= _check(f"n={n} p={p} ns={ns}: media identica", ef < 1e-12,
                     f"err {ef:.1e}")
        ok &= _check(f"n={n} p={p} ns={ns}: covarianza identica", eP < 1e-12,
                     f"err rel {eP:.1e}")

    # e l'intera catena deve dare lo stesso smoothing
    ss_gen = _var_ss(4, 3, rng)
    ss_fast = LinearGaussianSS(A=ss_gen.A, Q=ss_gen.Q, Z=ss_gen.Z, R=ss_gen.R,
                               c=ss_gen.c, a0=ss_gen.a0, P0=ss_gen.P0,
                               companion_n=4)
    _, Y = simulate_forward(ss_gen, 30, np.random.default_rng(5))
    Y[1::3, -1] = np.nan
    m1 = smoothed_mean(ss_gen, forward_pass(ss_gen, Y))
    m2 = smoothed_mean(ss_fast, forward_pass(ss_fast, Y))
    err = float(np.abs(m1 - m2).max() / max(np.abs(m1).max(), 1e-300))
    ok &= _check("la catena completa da' lo stesso smoothing", err < 1e-10,
                 f"err rel {err:.1e}")
    return ok


def _var_ss(n: int, p: int, rng) -> LinearGaussianSS:
    ns = n * p
    A = np.zeros((ns, ns))
    A[:n, :n] = 0.7 * np.eye(n) + 0.05 * rng.normal(size=(n, n))
    for j in range(1, p):
        A[:n, j * n:(j + 1) * n] = (0.05 / j) * rng.normal(size=(n, n))
    A[n:, :ns - n] = np.eye(ns - n)
    L = np.tril(rng.normal(size=(n, n))) * 0.2
    np.fill_diagonal(L, np.abs(np.diag(L)) + 0.4)
    Q = np.zeros((ns, ns)); Q[:n, :n] = L @ L.T
    Z = np.zeros((n, ns)); Z[:, :n] = np.eye(n)
    return LinearGaussianSS(A=A, Q=Q, Z=Z, R=1e-8 * np.eye(n),
                            c=np.concatenate([0.02 * rng.normal(size=n),
                                              np.zeros(ns - n)]),
                            a0=np.zeros(ns), P0=np.eye(ns))


def main() -> bool:
    print("=" * 82)
    print("Gate 5, blocco 1 — IL SIMULATION SMOOTHER, IN ISOLAMENTO")
    print("=" * 82)
    ok = True
    for t in (test_mean_smoother, test_simulation_smoother, test_missing,
              test_singular, test_var_companion, test_companion_predict):
        ok &= t()
    print("\n" + "=" * 82)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 82)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
