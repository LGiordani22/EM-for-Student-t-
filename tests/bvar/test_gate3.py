"""
core/bvar/tests/test_gate3.py

GATE 3, BLOCCO 1 — la mappa cube-root.

    python -m core.bvar.tests.test_gate3

La domanda del gate: **la mappa deduce il gemello mensile giusto?**

Il controllo decisivo e' il §3, e vale la pena dire perche'.  L'Appendice A da'
due strade indipendenti per la stessa quantita' — la formula generale A.9 e la
forma chiusa A.15 dell'esempio AR(2) — e le due NON coincidono se A.9 si legge
alla lettera.  Il test verifica che la nostra implementazione riproduca A.15, e
CONTEMPORANEAMENTE che la lettura letterale non lo faccia: se anche quella
passasse, il test non starebbe distinguendo niente.

  §1  Phi_m^3 == Phi, la proprieta' che definisce la radice.
  §2  La selezione delle radici (reali negative, coppie complesse).
  §3  A.9' contro la forma chiusa A.15.  IL CONTROLLO CENTRALE.
  §4  La scorciatoia di Kronecker A.10 == la soluzione diretta.
  §5  La costante: c_m ricostruisce il passo trimestrale; c_m = c no.
  §6  La mappa su estrazioni VERE del Q-BVAR (radici quasi unitarie incluse).
  §7  Higham: e' davvero la PSD piu' vicina?
  §8  Il ciclo di Kalman sul modello mensile.
  §9  `lyapunov_symm`: P_0 sul sottospazio stabile, la riga degli autori.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bvar.cube_root import (
    coupling_matrix,
    matrix_cube_root,
    monthly_constant,
    monthly_innovation_cov,
    quarterly_to_monthly,
    select_cube_roots,
)
from core.bvar.simulate import simulate_var, stable_var_coefficients
from core.bvar.spec import BVARSpec, MinnesotaSpec
from core.bvar.state_space import (
    build_state_space,
    filter_monthly,
    lyapunov_symm,
    nearest_psd,
)

_OK, _FAIL = "OK", "FAIL"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<62} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def _companion(A: np.ndarray) -> np.ndarray:
    """(p, n, n) -> companion (n*p, n*p).  Stessa convenzione di QBVARFit."""
    p, n, _ = A.shape
    Phi = np.zeros((n * p, n * p))
    Phi[:n] = np.hstack([A[j] for j in range(p)])
    if p > 1:
        Phi[n:, : n * (p - 1)] = np.eye(n * (p - 1))
    return Phi


# ─── 1. La proprieta' che definisce la radice ─────────────────────────────────

def test_cube() -> bool:
    print("\n1. Phi_m^3 == Phi   (A.6)")
    ok = True
    rng = np.random.default_rng(11)
    for (n, p) in [(1, 2), (2, 2), (3, 5), (5, 5), (8, 3)]:
        A = stable_var_coefficients(n, p, rng)
        Phi = _companion(A)
        Phi_m, d = matrix_cube_root(Phi)
        err = float(np.abs(np.linalg.matrix_power(Phi_m, 3) - Phi).max())
        ok &= _check(f"n={n} p={p}: la radice ricostruisce Phi", err < 1e-8,
                     f"err {err:.2e}  cond(V) {d['cond_V']:.1e}  "
                     f"{d['n_real']}R/{d['n_complex']}C")
        ok &= _check(f"n={n} p={p}: Phi_m e' reale", d["max_imag"] < 1e-9,
                     f"im {d['max_imag']:.1e}")
    return ok


# ─── 2. La selezione delle radici ─────────────────────────────────────────────

def test_root_selection() -> bool:
    print("\n2. La regola di selezione dell'Appendice A")
    ok = True
    # un reale NEGATIVO: la radice reale, non il ramo principale complesso
    r = select_cube_roots(np.array([-0.283]))
    ok &= _check("un autovalore reale negativo prende la radice REALE",
                 abs(r[0].imag) < 1e-15 and r[0].real < 0,
                 f"{r[0]:.6f}  (il ramo principale darebbe "
                 f"{complex(-0.283)**(1/3):.4f})")
    ok &= _check("...e al cubo torna se stesso", abs(r[0] ** 3 + 0.283) < 1e-12)

    # una coppia complessa: argomento minimo fra i tre candidati
    z = 0.5 * np.exp(1j * 2.5)
    got = select_cube_roots(np.array([z]))[0]
    cand = [np.abs(z) ** (1 / 3) * np.exp(1j * (np.angle(z) + 2 * np.pi * k) / 3)
            for k in (0, 1, -1)]
    ok &= _check("una coppia complessa prende l'ARGOMENTO MINIMO",
                 abs(np.angle(got)) <= min(abs(np.angle(c)) for c in cand) + 1e-12,
                 f"arg {np.angle(got):+.4f} vs {[f'{np.angle(c):+.4f}' for c in cand]}")
    conj = select_cube_roots(np.array([z, np.conj(z)]))
    ok &= _check("...e i coniugati ricevono radici coniugate (Phi_m reale)",
                 abs(conj[0] - np.conj(conj[1])) < 1e-15)
    return ok


# ─── 3. A.9' contro la forma chiusa A.15 — IL CONTROLLO CENTRALE ──────────────

def test_A15() -> bool:
    print("\n3. LE TRE FORMULE contro la forma chiusa A.15 dell'esempio AR(2)")
    print("     l'analisi di riproducibilita': A.8/A.9 come STAMPATE non")
    print("     riproducono ne' (A.15) ne' il codice degli autori")
    ok = True
    casi = [(0.6, 0.25), (1.1, -0.30), (0.4, 0.50), (0.5, -0.40), (1.3, -0.45)]
    conta = {"exact_a15": 0, "authors": 0, "literal": 0}

    for (phi1, phi2) in casi:
        Phi = np.array([[phi1, phi2], [1.0, 0.0]])
        Phi_m, _ = matrix_cube_root(Phi)
        m11, m12, m21, m22 = Phi_m[0, 0], Phi_m[0, 1], Phi_m[1, 0], Phi_m[1, 1]
        A15 = m12 * m21 - m11 * m22          # (A.15): -det(Phi_m)

        vals = {v: float(coupling_matrix(Phi_m, n=1, variant=v).item())
                for v in conta}
        for v, x in vals.items():
            if abs(x - A15) < 1e-9:
                conta[v] += 1
        print(f"       phi=({phi1:>4},{phi2:>6})  A.15 {A15:+.6f}   "
              + "   ".join(f"{v} {vals[v]:+.6f}" for v in
                           ("exact_a15", "authors", "literal")))

    ok &= _check("SOLO 'exact_a15' riproduce (A.15), su tutti i casi",
                 conta["exact_a15"] == len(casi), f"{conta['exact_a15']}/{len(casi)}")
    ok &= _check("'authors' (il codice MATLAB) NON la riproduce mai",
                 conta["authors"] == 0, f"{conta['authors']}/{len(casi)}")
    ok &= _check("'literal' (A.8 come stampata) NON la riproduce mai",
                 conta["literal"] == 0, f"{conta['literal']}/{len(casi)}")

    # la conseguenza sulla varianza: (A.15) var(eps_m) = [1+A^2]^-1 var(eps_q)
    Phi = np.array([[0.6, 0.25], [1.0, 0.0]])
    Phi_m, _ = matrix_cube_root(Phi)
    A15 = Phi_m[0, 1] * Phi_m[1, 0] - Phi_m[0, 0] * Phi_m[1, 1]
    var_q = 0.37
    got = float(monthly_innovation_cov(
        coupling_matrix(Phi_m, n=1, variant="exact_a15"),
        np.array([[var_q]])).item())
    ok &= _check("...e con 'exact_a15' torna anche var(eps_m) di (A.15)",
                 abs(got - var_q / (1.0 + A15 ** 2)) < 1e-12,
                 f"{got:.8f} vs {var_q / (1.0 + A15 ** 2):.8f}")

    # a p=2 il sistema e' ESATTAMENTE determinato: e' li' che la differenza
    # fra le formule non e' una scelta fra approssimazioni ma un errore.
    C = Phi_m[1:, 0:1]
    D = (Phi_m @ Phi_m)[1:, 0:1]
    M, *_ = np.linalg.lstsq(C, D, rcond=None)
    ok &= _check("a p=2 il sistema (A.8) e' esattamente determinato",
                 float(np.abs(C @ M - D).max()) < 1e-12,
                 f"residuo {float(np.abs(C @ M - D).max()):.1e}")
    return ok


# ─── 4. La scorciatoia di Kronecker ───────────────────────────────────────────

def test_kron_shortcut() -> bool:
    print("\n4. A.10 (scorciatoia) == (I + A (x) A)^-1 diretta")
    ok = True
    rng = np.random.default_rng(5)
    for n in (1, 2, 5, 12):
        A = 0.6 * rng.normal(size=(n, n)) / np.sqrt(n)
        L = np.tril(rng.normal(size=(n, n)))
        np.fill_diagonal(L, np.abs(np.diag(L)) + 0.5)
        Sigma = (L @ L.T) / n
        s1 = monthly_innovation_cov(A, Sigma, shortcut=True)
        s0 = monthly_innovation_cov(A, Sigma, shortcut=False)
        err = float(np.abs(s1 - s0).max())
        ok &= _check(f"n={n}: le due strade coincidono", err < 1e-9, f"err {err:.2e}")
        back = s1 + A @ s1 @ A.T
        ok &= _check(f"n={n}: Sigma_m + A Sigma_m A' == Sigma_eps",
                     np.allclose(back, Sigma, atol=1e-9),
                     f"err {np.abs(back - Sigma).max():.2e}")
    return ok


# ─── 5. La costante ───────────────────────────────────────────────────────────

def test_constant() -> bool:
    print("\n5. La costante A_0 — il punto aperto ereditato dal Gate 2")
    ok = True
    rng = np.random.default_rng(3)
    n, p = 2, 2
    A = stable_var_coefficients(n, p, rng)
    Phi = _companion(A)
    Phi_m, _ = matrix_cube_root(Phi)
    a0 = np.array([0.7, -0.3])
    c = np.concatenate([a0, np.zeros(n * (p - 1))])
    c_m, cond = monthly_constant(Phi_m, a0)

    X = rng.normal(size=n * p)
    tre_mesi = c_m + Phi_m @ (c_m + Phi_m @ (c_m + Phi_m @ X))
    un_trim = c + Phi @ X
    ok &= _check("tre passi mensili con c_m == un passo trimestrale",
                 np.allclose(tre_mesi, un_trim, atol=1e-10),
                 f"err {np.abs(tre_mesi - un_trim).max():.2e}, cond {cond:.2f}")

    alt = c + Phi_m @ (c + Phi_m @ (c + Phi_m @ X))
    ok &= _check("...e con c_m = c NO — l'ipotesi alternativa e' falsificata",
                 not np.allclose(alt, un_trim, atol=1e-6),
                 f"sbaglia di {np.abs(alt - un_trim).max():.4f}")
    ok &= _check("promuovere (n,) a (n*p,) da' lo stesso risultato",
                 np.allclose(c_m, monthly_constant(Phi_m, c)[0]))
    return ok


# ─── 6. La mappa su estrazioni vere del Q-BVAR ────────────────────────────────

def test_real_draws() -> bool:
    print("\n6. La mappa nel regime del Gate 2: n=30, p=5, radici quasi unitarie")
    print("     VAR sintetico calibrato su quel regime, non estrazioni vere —")
    print("     quelle costano una stima (~2.5 min) e stanno in scripts/.")
    ok = True
    rng = np.random.default_rng(31)
    n, p = 30, 5
    A = stable_var_coefficients(n, p, rng, persistence=0.985, max_eig=0.9995)
    Phi = _companion(A)
    sim = simulate_var(n, p, 200, rng, A=A)
    Sigma = sim["Sigma"]
    a0 = 0.01 * rng.normal(size=n)

    mm = quarterly_to_monthly(Phi, Sigma, n=n, const=a0)
    chk = mm.check(Phi, Sigma)
    d = mm.diagnostics
    ok &= _check("Phi_m^3 == Phi", chk["cube_ok"], f"err {chk['cube_err']:.2e}")
    ok &= _check("Sigma_m + A Sigma_m A' == Sigma_eps", chk["cov_ok"],
                 f"err {chk['cov_err']:.2e}")
    ok &= _check("(I + Phi_m + Phi_m^2) e' ben condizionata",
                 d["cond_const"] < 1e6, f"cond {d['cond_const']:.1f}")

    # Sigma_m DEFINITA POSITIVA: NON e' un pass/fail, ed e' deliberato.
    # L'Appendice A non garantisce che (A.9) abbia una soluzione PSD: la mappa
    # Sigma_m -> Sigma_m + A Sigma_m A' e' invertibile ma la sua inversa non
    # conserva il cono delle matrici semidefinite (la serie di Neumann e' a
    # segni alterni).  Trattarlo come un FAIL vorrebbe dire far fallire il test
    # su una proprieta' che il metodo non promette.  Lo si MISURA e lo si
    # riporta: e' un punto aperto del Gate 3, non un difetto del codice.
    ev = np.linalg.eigvalsh(mm.Sigma_m)
    neg = int((ev < 0).sum())
    print(f"       [PUNTO APERTO GATE 3] Sigma_m: {neg} autovalori negativi su {n}, "
          f"|min|/max = {abs(ev.min())/ev.max():.2e}, rho(A) = "
          f"{d['A_spectral_radius']:.4f}")
    print(f"       raggio spettrale: Phi {d['spectral_radius']:.4f} -> "
          f"Phi_m {d['spectral_radius']**(1/3):.4f}   cond(V) {d['cond_V']:.1e}   "
          f"{d['n_real']}R/{d['n_complex']}C")

    # e la scorciatoia deve reggere anche qui, dove n=30 (la diretta e' 900x900)
    s0 = monthly_innovation_cov(mm.A, Sigma, shortcut=False)
    ok &= _check("A.10 == diretta anche a n=30 (900 x 900)",
                 np.allclose(mm.Sigma_m, s0, atol=1e-8),
                 f"err {np.abs(mm.Sigma_m - s0).max():.2e}")
    return ok


# ─── 7. Higham: e' davvero la PSD piu' vicina? ────────────────────────────────

def test_higham() -> bool:
    print("\n7. La proiezione di Higham (1988) sul cono PSD")
    ok = True
    rng = np.random.default_rng(9)

    # su una matrice GIA' PSD non deve toccare niente
    L = np.tril(rng.normal(size=(6, 6)))
    S_pd = L @ L.T + 0.1 * np.eye(6)
    out, rep = nearest_psd(S_pd)
    ok &= _check("su una matrice gia' PSD non fa nulla",
                 not rep.projected and np.allclose(out, S_pd))

    # costruita a mano con autovalori noti: si azzerano solo i negativi
    V, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    lam = np.array([-0.4, -0.05, 0.2, 1.0, 3.0])
    S = (V * lam) @ V.T
    out, rep = nearest_psd(S)
    ev = np.linalg.eigvalsh(out)
    ok &= _check("azzera i negativi e lascia i positivi intatti",
                 np.allclose(np.sort(ev), [0.0, 0.0, 0.2, 1.0, 3.0], atol=1e-12),
                 f"{rep.n_neg} negativi trovati")
    atteso = np.sqrt(0.4 ** 2 + 0.05 ** 2)
    ok &= _check("||S - S_+||_F == sqrt(somma dei negativi al quadrato)",
                 abs(np.linalg.norm(S - out, "fro") - atteso) < 1e-12,
                 f"{np.linalg.norm(S - out, 'fro'):.8f} vs {atteso:.8f}")
    ok &= _check("...e rel_frobenius e' quel rapporto",
                 abs(rep.rel_frobenius - atteso / np.linalg.norm(S, "fro")) < 1e-12)

    # LA PROPRIETA' CHE DEFINISCE LA PROIEZIONE: nessuna altra PSD e' piu'
    # vicina.  Si prova per campionamento — non e' una dimostrazione, ma
    # prenderebbe un errore di segno o un clipping sbagliato.
    d0 = np.linalg.norm(S - out, "fro")
    peggio = 0
    for _ in range(300):
        M = rng.normal(size=(5, 5)) * 0.3
        cand = out + M @ M.T * rng.uniform(0.01, 0.5)      # un'altra PSD
        if np.linalg.norm(S - cand, "fro") < d0 - 1e-12:
            peggio += 1
    ok &= _check("nessuna delle 300 PSD alternative e' piu' vicina", peggio == 0)
    return ok


# ─── 8. Il filtro mensile ─────────────────────────────────────────────────────

def test_filter() -> bool:
    print("\n8. Il ciclo di Kalman sul modello mensile")
    ok = True
    rng = np.random.default_rng(17)
    n, p, T = 4, 3, 240
    A = stable_var_coefficients(n, p, rng, persistence=0.95, max_eig=0.999)
    Phi = _companion(A)
    Sigma = simulate_var(n, p, 100, rng, A=A)["Sigma"]
    mm = quarterly_to_monthly(Phi, Sigma, n=n, const=0.01 * rng.normal(size=n))
    ss = build_state_space(mm)

    ok &= _check("Q = blkdiag(Sigma_m proiettata, 0), ed e' PSD",
                 np.linalg.eigvalsh(ss.Q).min() > -1e-12
                 and np.allclose(ss.Q[n:, :], 0.0))
    ok &= _check("Lambda = [I_n, 0] (i mensili SONO le prime n componenti)",
                 np.allclose(ss.Lambda[:, :n], np.eye(n))
                 and np.allclose(ss.Lambda[:, n:], 0.0))
    # R non e' zero esatto: e' il nugget di regolarizzazione.  Deve essere
    # trascurabile rispetto a Sigma_m, altrimenti smetterebbe di essere una
    # regolarizzazione e diventerebbe errore di misura vero.
    rap = np.diag(ss.R).max() / np.mean(np.abs(np.diag(ss.Q[:n, :n])))
    ok &= _check("R e' il nugget, trascurabile rispetto a Sigma_m",
                 rap < 1e-6, f"R/Sigma_m = {rap:.1e}")
    print(f"       correzione di Higham: {ss.psd}")

    # simula dal modello mensile e filtra
    ns = n * p
    X = np.zeros((T, ns))
    Lq = np.linalg.cholesky(ss.Q[:n, :n] + 1e-12 * np.eye(n))
    for t in range(1, T):
        e = np.zeros(ns)
        e[:n] = Lq @ rng.standard_normal(n)
        X[t] = ss.c_m + ss.Phi_m @ X[t - 1] + e
    Y = X[:, :n].copy()

    res = filter_monthly(ss, Y)
    ok &= _check("con dati completi il filtro riproduce lo stato osservato",
                 np.allclose(res.observables[10:], Y[10:], atol=1e-6),
                 f"err max {np.abs(res.observables[10:] - Y[10:]).max():.2e}")
    ok &= _check("la log-verosimiglianza e' finita", np.isfinite(res.loglik),
                 f"{res.loglik:.1f}")

    # ORA IL PUNTO: frequenza mista.  Le ultime due serie diventano trimestrali
    # (osservate solo nei mesi 3, 6, 9, 12) e l'ultimo mese e' bordo.
    Yq = Y.copy()
    mesi = np.arange(T) % 3
    Yq[mesi != 2, -2:] = np.nan
    Yq[-1, :2] = np.nan
    res_q = filter_monthly(ss, Yq)
    ok &= _check("con le trimestrali latenti il filtro gira e resta finito",
                 np.isfinite(res_q.loglik), f"loglik {res_q.loglik:.1f}")
    ok &= _check("...e W_t vede il numero giusto di serie",
                 res_q.n_obs[1] == n - 2 and res_q.n_obs[2] == n,
                 f"mese 2: {res_q.n_obs[1]}, mese 3: {res_q.n_obs[2]} su {n}")

    # RECOVERY: il filtro deve stimare la trimestrale latente meglio di una
    # costante.  E' la ragione per cui il C-BVAR esiste.
    lat = res_q.observables[3:-1, -2:]
    vero = Y[3:-1, -2:]
    manc = mesi[3:-1] != 2
    err_f = float(np.sqrt(np.mean((lat[manc] - vero[manc]) ** 2)))
    err_0 = float(np.sqrt(np.mean((vero[manc] - vero[manc].mean(axis=0)) ** 2)))
    ok &= _check("nei mesi latenti il filtro batte la media incondizionata",
                 err_f < err_0, f"RMSE {err_f:.4f} vs {err_0:.4f}")

    # l'inizializzazione diffusa non deve inquinare la verosimiglianza
    r1 = filter_monthly(ss, Y, kappa=1e6)
    r2 = filter_monthly(ss, Y, kappa=1e9)
    ok &= _check("la loglik non dipende da kappa (diffusi esclusi)",
                 abs(r1.loglik - r2.loglik) / abs(r1.loglik) < 1e-6,
                 f"{r1.loglik:.4f} vs {r2.loglik:.4f}")
    return ok


# ─── 9. lyapunov_symm: P_0 sul sottospazio stabile ────────────────────────────

def test_lyapunov() -> bool:
    print("\n9. lyapunov_symm — Lyapunov sul SOTTOSPAZIO STABILE (P_0 degli autori)")
    print("     e' la routine di Dynare chiamata da run_Ksmoother.m: scarta le")
    print("     direzioni con |lambda| > 1 invece di fallire su un esplosivo.")
    ok = True
    from scipy.linalg import solve_discrete_lyapunov

    # (a) sistema TUTTO stabile: deve coincidere con la Lyapunov piena
    rng = np.random.default_rng(23)
    for n in (3, 8):
        A = rng.normal(size=(n, n))
        A *= 0.7 / max(abs(np.linalg.eigvals(A)))
        L = np.tril(rng.normal(size=(n, n)))
        Q = L @ L.T + 0.1 * np.eye(n)
        P, sd = lyapunov_symm(A, Q)
        P_pieno = solve_discrete_lyapunov(A, Q)
        ok &= _check(f"n={n}: stabile -> tutte le direzioni tenute", sd == n,
                     f"sdim {sd}")
        ok &= _check(f"n={n}: coincide con solve_discrete_lyapunov",
                     np.allclose(P, P_pieno, atol=1e-9),
                     f"err {np.abs(P - P_pieno).max():.2e}")

    # (b) una direzione ESPLOSIVA: si scarta quella e solo quella.  Il caso e'
    #     costruito in modo che la risposta esatta si sappia a mano.
    lam = np.array([0.5, 0.9, 1.2])
    U, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    A = U @ np.diag(lam) @ U.T
    P, sd = lyapunov_symm(A, np.eye(3))
    atteso = U @ np.diag([1 / (1 - 0.25), 1 / (1 - 0.81), 0.0]) @ U.T
    ok &= _check("una radice > 1: il blocco stabile e' 2 su 3", sd == 2, f"sdim {sd}")
    ok &= _check("...e P e' la soluzione esatta con ZERO sull'esplosiva",
                 np.allclose(P, atteso, atol=1e-8),
                 f"err {np.abs(P - atteso).max():.2e}")
    v = U[:, 2]                                   # la direzione esplosiva
    ok &= _check("...cioe' varianza nulla lungo l'autovettore esplosivo",
                 abs(float(v @ P @ v)) < 1e-9, f"{float(v @ P @ v):.2e}")
    ok &= _check("P e' PSD", np.linalg.eigvalsh(0.5 * (P + P.T)).min() > -1e-9)

    # (c) il caso degenere: tutto esplosivo -> non c'e' niente da risolvere, e
    #     lo si dice invece di restituire numeri finti.
    P, sd = lyapunov_symm(np.diag([1.5, 2.0]), np.eye(2))
    ok &= _check("tutto esplosivo -> sdim = 0 e P = 0", sd == 0 and not P.any())
    return ok


def main() -> bool:
    print("=" * 82)
    print("Gate 3 — LA MAPPA CUBE-ROOT (blocco 1) e LO STATO-SPAZIO (blocco 2)")
    print("=" * 82)
    ok = True
    for t in (test_cube, test_root_selection, test_A15, test_kron_shortcut,
              test_constant, test_real_draws, test_higham, test_filter,
              test_lyapunov):
        ok &= t()
    print("\n" + "=" * 82)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 82)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
