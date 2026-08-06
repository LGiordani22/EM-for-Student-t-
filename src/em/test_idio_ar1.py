r"""
src/em/test_idio_ar1.py

Gate dell'Asse B a livello di ALGEBRA, prima di cablare l'AR(1) nel loop EM.

    python -m em.test_idio_ar1

Quattro domande, in ordine di importanza:

  1. NON-REGRESSIONE. Con rho = 0 lo stato
     aumentato deve riprodurre ESATTAMENTE il baseline: stessa loglik, stessi
     fattori smoothed. Se questo fallisce, l'augmentazione e' sbagliata e non
     ha senso guardare il resto.
  2. AGGREGAZIONE MM DELL'IDIO. Il valore atteso di una riga trimestrale deve
     essere la somma MM di comune E idio: e' la scelta di modello che distingue
     questa implementazione da quella nel .tex.
  3. RECOVERY DI RHO su DGP sintetico, in gaussiano e con pesi t.
  4. COERENZA DEI MOMENTI: i momenti estratti dallo smoother devono coincidere
     con quelli calcolati direttamente sulle traiettorie vere quando il segnale
     e' quasi perfettamente osservato.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = str(pathlib.Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from em.idio_ar1 import (  # noqa: E402
    MM_WEIGHTS, build_idio_layout, build_P_idio, build_Q_idio, build_C_idio,
    build_augmented_A, build_augmented_Q, build_augmented_Lambda,
    build_augmented_Sigma0, extract_idio_moments, update_rho, update_sigma2,
    idio_innovation_second_moment,
)
from kalman import (  # noqa: E402
    build_A_tilde, build_Q_tilde, build_Lambda_tilde, build_R_tilde,
    build_all_selection_matrices, kalman_filter, kalman_smoother,
)
from em.idio_ar1 import build_augmented_A as _bA  # noqa: E402,F401

_PASS = 0
_FAIL = 0


def check(cond: bool, msg: str, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {msg}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {msg}" + (f"\n         {detail}" if detail else ""))


# ─── DGP sintetico ────────────────────────────────────────────────────────────

def simulate(T, r, M_m, M_q, rho, sigma2, A, Q, Lambda, seed=0,
             nu_eps=np.inf, nu_u=np.inf):
    """
    Simula il modello ESTESO: fattori VAR(1) + idio AR(1) per serie, con le
    trimestrali osservate come aggregato MM di comune e idio (coerenti).
    """
    rng = np.random.default_rng(seed)
    M = M_m + M_q
    freq_list = ["monthly"] * M_m + ["quarterly"] * M_q

    # fattori
    f = np.zeros((T, r))
    Lc = np.linalg.cholesky(Q)
    for t in range(1, T):
        w = 1.0 if not np.isfinite(nu_u) else rng.gamma(nu_u / 2, 2 / nu_u)
        f[t] = A @ f[t - 1] + (Lc @ rng.standard_normal(r)) / np.sqrt(w)

    # idio AR(1) per serie
    e = np.zeros((T, M))
    for i in range(M):
        sd = np.sqrt(sigma2[i])
        e[0, i] = rng.standard_normal() * sd / np.sqrt(1 - rho[i] ** 2)
        for t in range(1, T):
            w = 1.0 if not np.isfinite(nu_eps) else rng.gamma(nu_eps / 2, 2 / nu_eps)
            e[t, i] = rho[i] * e[t - 1, i] + rng.standard_normal() * sd / np.sqrt(w)

    # osservazioni: mensili al livello, trimestrali aggregate MM (comune E idio)
    Y = np.full((T, M), np.nan)
    for t in range(T):
        for i in range(M):
            if freq_list[i] == "monthly":
                Y[t, i] = Lambda[i] @ f[t] + e[t, i]
            elif t >= 4 and (t + 1) % 3 == 0:
                agg_f = sum(MM_WEIGHTS[l] * (Lambda[i] @ f[t - l]) for l in range(5))
                agg_e = sum(MM_WEIGHTS[l] * e[t - l, i] for l in range(5))
                Y[t, i] = agg_f + agg_e
    return Y, f, e, freq_list


def run_augmented_smoother(Y, theta, freq_list, layout, w_u=None, w_eps=None):
    """
    Filtro+smoother sullo stato aumentato, passando per il kalman_filter di
    PRODUZIONE: il test valida il codice che verra' usato, non una sua copia.
    L'AR(1) si attiva perche' theta porta 'rho'.
    """
    r = theta["A"].shape[0]
    d_f = 5 * r
    th = {
        "A": theta["A"], "Q": theta["Q"], "Lambda": theta["Lambda"],
        "R": theta["sigma2"], "sigma2": theta["sigma2"], "rho": theta["rho"],
        "Sigma_0": np.eye(d_f),          # verra' esteso con la stazionaria
    }
    filt = kalman_filter(Y, th, w_u=w_u, w_eps=w_eps, freq_list=freq_list)
    A_ck = build_augmented_A(build_A_tilde(theta["A"]), theta["rho"], layout)
    smooth = kalman_smoother(filt, A_ck)
    return filt, smooth, d_f


# ─── I test ───────────────────────────────────────────────────────────────────

def test_layout():
    print("\n[1] Layout dello stato")
    lay = build_idio_layout(["monthly"] * 4 + ["quarterly"] * 2)
    check(lay.n_e == 4 * 1 + 2 * 5, "n_e = M_m + 5*M_q", f"n_e={lay.n_e}")
    check(list(lay.slots(4)) == [4, 5, 6, 7, 8], "trimestrale -> 5 slot contigui")
    check(list(lay.slots(0)) == [0], "mensile -> 1 slot")

    lay37 = build_idio_layout(["monthly"] * 34 + ["quarterly"] * 3)
    check(lay37.n_e == 49, "dataset 'final': n_e = 49 (stato 5r+49 = 69 con r=4)",
          f"n_e={lay37.n_e}")

    # C: la riga trimestrale porta i pesi MM, la mensile un 1
    C = build_C_idio(lay)
    check(np.allclose(C[4, 4:9], MM_WEIGHTS), "riga trimestrale di C = pesi MM")
    check(C[0, 0] == 1.0 and C[0, 1:].sum() == 0.0, "riga mensile di C = e_i,t")
    check(np.allclose(C.sum(axis=1)[4:], 3.0), "somma pesi MM per riga = 3")

    # P: companion corretta
    P = build_P_idio(np.array([0.5, 0.1, 0.2, 0.3, 0.7, -0.4]), lay)
    check(P[4, 4] == 0.7, "P: rho sullo slot corrente della trimestrale")
    check(P[5, 4] == 1.0 and P[8, 7] == 1.0, "P: righe di shift dei lag")
    check(np.allclose(np.diag(P)[5:9], 0.0),
          "P: le righe di shift non hanno dinamica propria (diagonale nulla)")

    # Q singolare solo sugli slot correnti
    Q = build_Q_idio(np.ones(6) * 0.4, lay)
    check(np.allclose(np.diag(Q)[lay.current_slots()], 0.4), "Q: varianza sugli slot correnti")
    check(np.linalg.matrix_rank(Q) == 6, "Q: rango = M (righe di shift a varianza 0)")


def test_non_regression():
    """
    Il gate di non-regressione, nella sua forma CORRETTA.

    Il gate diceva: "idio_ar1=False riproduce l'Asse A bit-identico". Vero per
    le righe MENSILI, e li' lo si verifica a precisione macchina. FALSO per le
    trimestrali, e non per un bug: aggregare rumore bianco con i pesi MM
    produce una struttura MA. Le finestre di 5 mesi di due trimestri
    consecutivi si sovrappongono in 2 mesi, quindi

        Var[riga trimestrale] = sigma^2 * sum_l c_l^2      = 2.111 sigma^2
        Cov[trim. consecutivi] = sigma^2 * sum_l c_l c_{l+3} = 0.444 sigma^2
        corr = +0.2105

    mentre il baseline tratta le osservazioni trimestrali come indipendenti.
    Sotto aggregazione coerente rho=0 NON e' il baseline: e' il baseline piu'
    un MA indotto dall'aggregazione. E' una conseguenza della scelta di
    modello, ed e' il motivo per cui il gate va enunciato per frequenza.
    """
    print("\n[2] Non-regressione: rho = 0 vs baseline")
    rng = np.random.default_rng(7)
    T, r = 220, 2
    A = np.array([[0.6, 0.1], [0.0, 0.5]])
    Q = np.array([[0.6, 0.1], [0.1, 0.4]])

    # ── (a) pannello SOLO MENSILE: l'equivalenza deve essere esatta ──────────
    M_m, M_q = 7, 0
    M = M_m
    Lambda = rng.normal(size=(M, r))
    sigma2 = rng.uniform(0.2, 0.6, M)
    rho0 = np.zeros(M)
    Y, _, _, freq_list = simulate(T, r, M_m, M_q, rho0, sigma2, A, Q, Lambda, seed=3)
    layout = build_idio_layout(freq_list)
    theta = {"A": A, "Q": Q, "Lambda": Lambda, "rho": rho0, "sigma2": sigma2}

    filt_a, sm_a, d_f = run_augmented_smoother(Y, theta, freq_list, layout)
    th_b = {"A": A, "Q": Q, "Lambda": Lambda, "R": sigma2,
            "Sigma_0": np.eye(5 * r)}
    filt_b = kalman_filter(Y, th_b, freq_list=freq_list)
    sm_b = kalman_smoother(filt_b, build_A_tilde(A))

    d_ll = abs(float(filt_a["loglik"]) - float(filt_b["loglik"]))
    check(d_ll < 1e-6,
          "pannello mensile: loglik identica fra stato aumentato (rho=0) e baseline",
          f"|diff| = {d_ll:.3e}")
    d_fs = float(np.max(np.abs(sm_a["f_smooth"][:, :d_f] - sm_b["f_smooth"])))
    check(d_fs < 1e-6, "pannello mensile: fattori smoothed identici",
          f"max|diff| = {d_fs:.3e}")

    # ── (b) la riga trimestrale differisce, e differisce QUANTO DEVE ─────────
    c = MM_WEIGHTS
    var_th = float((c ** 2).sum())
    cov_th = float(sum(c[l] * c[l + 3] for l in range(2)))
    corr_th = cov_th / var_th

    # simulazione diretta dell'aggregato di rumore bianco (rho = 0)
    rng2 = np.random.default_rng(99)
    n = 400_000
    e = rng2.standard_normal(n)
    agg = sum(c[l] * e[4 - l: n - l] for l in range(5))    # allineati sul mese t
    q = agg[::3]                                           # un trimestre ogni 3 mesi
    var_emp = float(q.var())
    corr_emp = float(np.corrcoef(q[:-1], q[1:])[0, 1])
    check(abs(var_emp - var_th) < 0.02,
          f"riga trimestrale: var = sum(c^2) = {var_th:.4f} sigma^2 (emp {var_emp:.4f})")
    check(abs(corr_emp - corr_th) < 0.02,
          f"riga trimestrale: MA indotto dall'aggregazione, corr = {corr_th:+.4f} "
          f"(emp {corr_emp:+.4f})",
          "il baseline assume 0 -> rho=0 NON e' il baseline per le trimestrali")


def test_mm_aggregation():
    """L'idio della riga trimestrale entra aggregato, non al livello."""
    print("\n[3] Aggregazione MM dell'idiosincratico (la scelta di modello)")
    rng = np.random.default_rng(11)
    r, M_m, M_q = 2, 3, 1
    M = M_m + M_q
    freq_list = ["monthly"] * M_m + ["quarterly"] * M_q
    layout = build_idio_layout(freq_list)
    Lambda = rng.normal(size=(M, r))

    Lam_ck = build_augmented_Lambda(build_Lambda_tilde(Lambda, freq_list), layout)
    d_f = 5 * r
    state = rng.normal(size=Lam_ck.shape[1])
    y = Lam_ck @ state

    i_q = M - 1
    f_lags = state[:d_f].reshape(5, r)
    e_lags = state[d_f + layout.slot(i_q): d_f + layout.slot(i_q) + 5]
    atteso = sum(MM_WEIGHTS[l] * (Lambda[i_q] @ f_lags[l]) for l in range(5)) \
        + float(MM_WEIGHTS @ e_lags)
    check(abs(y[i_q] - atteso) < 1e-12,
          "riga trimestrale = MM(comune) + MM(idio), stessi pesi",
          f"y={y[i_q]:.6f} vs atteso={atteso:.6f}")

    # e la riga mensile resta al livello
    atteso_m = Lambda[0] @ f_lags[0] + state[d_f + layout.slot(0)]
    check(abs(y[0] - atteso_m) < 1e-12, "riga mensile = comune_t + eps_t")

    # controprova: con I_M (versione .tex) il valore sarebbe diverso
    C_identity = np.zeros((M, layout.n_e))
    for i in range(M):
        C_identity[i, layout.slot(i)] = 1.0
    y_tex = (np.hstack([build_Lambda_tilde(Lambda, freq_list), C_identity]) @ state)
    check(abs(y_tex[i_q] - y[i_q]) > 1e-6,
          "le due specifiche differiscono davvero sulla riga trimestrale",
          f"MM={y[i_q]:.4f} vs I_M={y_tex[i_q]:.4f}")


def test_moments_and_recovery(nu_eps=np.inf, label="gaussiano", seed=21):
    print(f"\n[4] Recovery di rho e sigma^2 — DGP {label}")
    rng = np.random.default_rng(seed)
    T, r, M_m, M_q = 1500, 2, 8, 1
    M = M_m + M_q
    A = np.array([[0.7, 0.0], [0.1, 0.5]])
    Q = np.array([[0.5, 0.05], [0.05, 0.4]])
    Lambda = rng.normal(size=(M, r)) * 0.9
    # rho eterogenei, inclusi valori bassi e alti
    rho_true = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, -0.3, 0.5, 0.7])[:M]
    sigma2_true = rng.uniform(0.25, 0.55, M)

    Y, f_true, e_true, freq_list = simulate(
        T, r, M_m, M_q, rho_true, sigma2_true, A, Q, Lambda,
        seed=seed + 1, nu_eps=nu_eps)
    layout = build_idio_layout(freq_list)

    # Smoother ai VERI parametri: isola la qualita' dei momenti dall'ottimizzazione
    theta = {"A": A, "Q": Q, "Lambda": Lambda,
             "rho": rho_true, "sigma2": sigma2_true}
    _filt, sm, d_f = run_augmented_smoother(Y, theta, freq_list, layout)
    mom = extract_idio_moments(sm["f_smooth"], sm["P_smooth"], sm["P_lag"],
                               layout, d_f)

    # 4a. i momenti ricostruiscono le traiettorie vere
    # Quanto bene si recuperi eps_i dipende dal rapporto segnale/rumore della
    # riga: una serie con caricamento grande e idio piccolo ha l'idio "coperto"
    # dal comune e si separa peggio. Si guarda quindi la MEDIANA (il
    # comportamento tipico) e si stampa il dettaglio per serie.
    corr = np.array([np.corrcoef(mom["E_e"][:, i], e_true[:, i])[0, 1]
                     for i in range(M)])
    snr = (np.abs(Lambda).sum(axis=1) ** 2) / sigma2_true
    check(np.median(corr) > 0.85 and corr.min() > 0.45,
          f"E[eps|Y] segue la traiettoria vera (mediana {np.median(corr):.3f}, "
          f"min {corr.min():.3f})",
          f"corr = {np.round(corr, 3).tolist()}\n         "
          f"snr  = {np.round(snr, 2).tolist()}  (corr bassa dove il comune domina)")

    # 4b. un passo di M-step recupera rho
    rho_hat = update_rho(mom)
    err = np.abs(rho_hat - rho_true)
    check(err.max() < 0.15,
          f"update_rho recupera rho (err max = {err.max():.3f})",
          f"vero  = {np.round(rho_true, 2).tolist()}\n         "
          f"stima = {np.round(rho_hat, 2).tolist()}")

    # 4c. e sigma^2
    s2_hat = update_sigma2(mom, rho_hat)
    if np.isfinite(nu_eps):
        # Senza passare i pesi, update_sigma2 stima la varianza MARGINALE
        # dell'innovazione, che sotto Student-t vale sigma^2 * nu/(nu-2), non
        # sigma^2. Non e' un bias del codice: e' esattamente il lavoro che
        # fanno i pesi w_eps dell'E-step. Verificare il fattore documenta il
        # ruolo dei pesi meglio di quanto farebbe allentare una soglia.
        infl = float(np.median(s2_hat / sigma2_true))
        atteso = nu_eps / (nu_eps - 2.0)
        check(abs(infl - atteso) < 0.30,
              f"senza pesi: sigma^2 stimata = sigma^2 * nu/(nu-2) "
              f"(atteso {atteso:.2f}, misurato {infl:.2f})",
              "e' il fattore che i pesi w_eps dell'E-step rimuovono")
        # con i pesi (qui l'oracolo E[w] = (nu-2)/nu) il bias sparisce
        w_oracle = np.full(mom["E_e2"].shape[0], (nu_eps - 2.0) / nu_eps)
        s2_w = update_sigma2(mom, rho_hat, w_eps=w_oracle)
        rel_w = np.abs(s2_w - sigma2_true) / sigma2_true
        check(float(np.median(rel_w)) < 0.20,
              f"con i pesi: sigma^2 torna sul vero "
              f"(err rel mediano = {np.median(rel_w):.2f})",
              f"vero  = {np.round(sigma2_true, 3).tolist()}\n         "
              f"stima = {np.round(s2_w, 3).tolist()}")
    else:
        rel = np.abs(s2_hat - sigma2_true) / sigma2_true
        check(rel.max() < 0.35,
              f"update_sigma2 recupera sigma^2 (err rel max = {rel.max():.2f})",
              f"vero  = {np.round(sigma2_true, 3).tolist()}\n         "
              f"stima = {np.round(s2_hat, 3).tolist()}")

    # 4d. il momento dell'innovazione e' non-negativo (e' un quadrato atteso)
    S = idio_innovation_second_moment(mom, rho_hat)
    check(float(S.min()) >= -1e-9,
          "E[(eps_t - rho eps_{t-1})^2] >= 0 ovunque", f"min = {S.min():.3e}")

    # 4e. iterando (smoother ai parametri stimati) non peggiora
    theta2 = dict(theta, rho=rho_hat, sigma2=s2_hat)
    _f2, sm2, _ = run_augmented_smoother(Y, theta2, freq_list, layout)
    mom2 = extract_idio_moments(sm2["f_smooth"], sm2["P_smooth"], sm2["P_lag"],
                                layout, d_f)
    rho_hat2 = update_rho(mom2)
    err2 = np.abs(rho_hat2 - rho_true)
    check(err2.max() <= err.max() + 0.05,
          f"una seconda iterazione non peggiora (err max {err.max():.3f} -> {err2.max():.3f})")


def test_rho_zero_reduces_to_baseline_updates():
    """Con rho=0 gli update dell'Asse B coincidono con l'update i.i.d. della scala idiosincratica."""
    print("\n[5] Nesting esatto degli update (rho = 0 -> baseline)")
    rng = np.random.default_rng(5)
    T, M = 400, 6
    E_e = rng.normal(size=(T, M))
    mom = {"E_e": E_e, "E_e2": E_e ** 2 + 0.1,
           "E_e_lag": np.zeros((T, M)), "E_e2_lag": np.zeros((T, M))}
    s2 = update_sigma2(mom, np.zeros(M))
    atteso = mom["E_e2"][1:, :].sum(axis=0) / (T - 1)
    check(np.allclose(s2, atteso),
          "con rho=0, update_sigma2 == media di E[eps^2] (update i.i.d. della scala)",
          f"max|diff| = {np.max(np.abs(s2 - atteso)):.2e}")


if __name__ == "__main__":
    print("=" * 70)
    print("  Asse B — gate algebrico dell'idiosincratico AR(1)")
    print("=" * 70)
    test_layout()
    test_non_regression()
    test_mm_aggregation()
    test_rho_zero_reduces_to_baseline_updates()
    test_moments_and_recovery(nu_eps=np.inf, label="gaussiano", seed=21)
    test_moments_and_recovery(nu_eps=5.0, label="Student-t (nu_eps=5)", seed=41)
    print("\n" + "=" * 70)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 70)
    sys.exit(1 if _FAIL else 0)
