"""
GUARDIA: la radice spettrale non e' peggiore della Cholesky dove la Cholesky
funziona, e regge dove la Cholesky si arrende.

    python -m src.bvar.test_niw_robust

PERCHE' ESISTE
--------------
`niw.robust_sqrt` ha sostituito `cholesky(..., lower=True)` in `draw()`, e
`np.linalg.solve` ha sostituito `cho_factor/cho_solve` in `niw_posterior` e
`log_ml`.  Il motivo sta nella docstring di `robust_sqrt`: la Cholesky si
arrestava con "122-th leading minor ... not positive definite" e portava giu'
i due blocchi BVAR il cui campione di stima contiene il Covid.

Un cambio del genere non si accetta perche' "adesso non crasha".  Le domande
a cui questo file risponde sono tre, e vanno tenute distinte:

  1. LA RADICE E' UNA RADICE?     C @ C.T deve ricostruire S.  Se sbagliassimo
     qui, ogni estrazione di (B, Sigma) avrebbe la covarianza sbagliata e non
     se ne accorgerebbe nessuno: i numeri uscirebbero plausibili e falsi.

  2. DOVE LA CHOLESKY VA, TAGLIAMO QUALCOSA?   No, e va dimostrato: su una
     matrice ben condizionata nessun autovalore deve finire sotto la soglia.
     Altrimenti staremmo buttando incertezza vera in silenzio — l'errore che
     la soglia RELATIVA (invece dell'assoluta 1e-12 degli autori) evita.

  3. LA DISTRIBUZIONE E' LA STESSA?   La radice di una covarianza non e'
     unica: Cholesky e spettrale danno C diverse.  Cio' che deve coincidere
     non e' la matrice, e' la LEGGE delle estrazioni.  Si verifica per
     Monte Carlo che vec(B - b_bar) abbia covarianza Sigma (x) omega_bar,
     che e' la (A.9) e la promessa nella docstring di `draw`.

  4. `_spd_solve` DA' LO STESSO NUMERO DELLA LU?   E' la domanda gemella,
     arrivata dopo: `_spd_solve` ha sostituito `np.linalg.solve` in
     `niw_posterior` perche' su questa macchina il percorso LU di OpenBLAS e'
     patologico (858 ms contro 5.2 ms, 165x).  La sua docstring promette due
     cose — stesso risultato a ~1e-9 relativo, e ripiego sulla LU quando la
     Cholesky si arrende — e nessuno le verificava.  Sono la promessa piu'
     facile da rompere per sbaglio: basta invertire un `lower=` nei due
     `solve_triangular` e il numero esce plausibile e sbagliato.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky

from src.bvar.niw import NIWPosterior, _spd_solve, draw, robust_sqrt

OK, KO = "  OK  ", "  ROTTO"
_fail = 0


def check(nome: str, condizione: bool, dettaglio: str = "") -> None:
    global _fail
    print(f"{OK if condizione else KO}  {nome}" + (f"   {dettaglio}" if dettaglio else ""))
    if not condizione:
        _fail += 1


def spd(n: int, rng, cond: float = 1e3) -> np.ndarray:
    """Una SPD con numero di condizionamento controllato."""
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    d = np.logspace(0, -np.log10(cond), n)
    return Q @ np.diag(d) @ Q.T


def main() -> int:
    rng = np.random.default_rng(20260808)
    print("=" * 74)
    print("  robust_sqrt — radice spettrale al posto della Cholesky")
    print("=" * 74)

    # ── 1. E' una radice? ────────────────────────────────────────────────────
    for n, cond in ((5, 1e2), (30, 1e6), (151, 1e10)):
        S = spd(n, rng, cond)
        C = robust_sqrt(S)
        err = np.abs(C @ C.T - S).max() / np.abs(S).max()
        check(f"C @ C.T ricostruisce S   (n={n:3d}, cond={cond:.0e})",
              err < 1e-10, f"errore relativo {err:.2e}")

    # ── 2. Dove la Cholesky va, non tagliamo niente ──────────────────────────
    # Se `robust_sqrt` azzerasse autovalori su una matrice sana, C avrebbe
    # rango ridotto: si vede dal rango e dal determinante.
    S = spd(60, rng, cond=1e6)
    C = robust_sqrt(S)
    check("nessun autovalore tagliato su matrice ben condizionata",
          np.linalg.matrix_rank(C) == 60, f"rango {np.linalg.matrix_rank(C)}/60")

    L = cholesky(S, lower=True)
    check("la Cholesky su quella stessa matrice funziona (controllo)",
          np.abs(L @ L.T - S).max() / np.abs(S).max() < 1e-10)

    # ── 3. Dove la Cholesky si arrende, noi no ───────────────────────────────
    # Una matrice singolare per costruzione: rango 40 su 60.
    A = rng.standard_normal((60, 40))
    S_sing = A @ A.T
    crash = False
    try:
        cholesky(S_sing, lower=True)
    except Exception:                                        # noqa: BLE001
        crash = True
    check("la Cholesky FALLISCE sulla singolare (premessa del test)", crash)

    C = robust_sqrt(S_sing)
    err = np.abs(C @ C.T - S_sing).max() / np.abs(S_sing).max()
    check("robust_sqrt regge sulla singolare e ricostruisce S",
          err < 1e-10, f"errore relativo {err:.2e}")

    # Il caso vero: definita positiva in teoria, non in doppia precisione.
    # E' la forma di `prec = X'X + Omega^-1` coi numeri del 2020.
    n = 120
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    d = np.logspace(8, -9, n)                    # 17 ordini: sfonda il doppio
    S_bad = Q @ np.diag(d) @ Q.T
    S_bad = 0.5 * (S_bad + S_bad.T)
    crash = False
    try:
        cholesky(S_bad, lower=True)
    except Exception:                                        # noqa: BLE001
        crash = True
    C = robust_sqrt(S_bad)
    finito = np.isfinite(C).all()
    check("caso 'PD in teoria, non in doppia precisione': robust_sqrt non solleva",
          finito, f"(la Cholesky {'fallisce' if crash else 'passa'} qui)")

    # ── 4. La LEGGE delle estrazioni e' quella giusta ────────────────────────
    # Monte Carlo su `draw`: vec(B - b_bar) deve avere covarianza
    # Sigma (x) omega_bar.  Si tiene Sigma FISSA per poter confrontare, quindi
    # si controlla il pezzo condizionato — che e' quello che dipende dalla
    # radice che abbiamo cambiato.
    k, n_var = 4, 3
    omega_bar = spd(k, rng, cond=1e4)
    Sigma = spd(n_var, rng, cond=1e2)
    b_bar = rng.standard_normal((k, n_var))

    L_om, L_sig = robust_sqrt(omega_bar), robust_sqrt(Sigma)
    n_mc = 200_000
    Z = rng.standard_normal((n_mc, k, n_var))
    Bs = b_bar + L_om @ Z @ L_sig.T                 # la riga di `draw`

    # ATTENZIONE ALL'ORDINE, ed e' il punto in cui questo test si e' rotto la
    # prima volta.  La (A.9) e la docstring di `draw` dicono
    #     Cov(vec(B - b_bar)) = Sigma (x) omega_bar
    # dove `vec` e' quello DELL'ALGEBRA, che impila le COLONNE.  `reshape` in
    # numpy appiattisce per RIGHE: darebbe vec(M') e scambierebbe i due
    # fattori di Kronecker, cioe' omega_bar (x) Sigma.  Con `transpose(0,2,1)`
    # prima del reshape si impilano le colonne davvero, e si confronta con
    # l'ordine dichiarato invece che con uno comodo.
    V = (Bs - b_bar).transpose(0, 2, 1).reshape(n_mc, -1)
    emp = V.T @ V / n_mc
    teo = np.kron(Sigma, omega_bar)
    err = np.abs(emp - teo).max() / np.abs(teo).max()
    check("vec(B - b_bar) ha covarianza Sigma (x) omega_bar  (Monte Carlo)",
          err < 0.02, f"scarto relativo {err:.3f} su {n_mc:,} estrazioni")

    # ── 5. draw() end-to-end non solleva e da' Sigma simmetriche ─────────────
    post = NIWPosterior(b_bar=b_bar, omega_bar=omega_bar,
                        psi_bar=spd(n_var, rng, cond=1e3), dof_bar=40,
                        resid=np.zeros((10, n_var)))
    B, S_out = draw(post, np.random.default_rng(1), n_draws=50)
    check("draw() gira e restituisce forme giuste",
          B.shape == (50, k, n_var) and S_out.shape == (50, n_var, n_var))
    check("le Sigma estratte sono simmetriche e definite positive",
          all(np.linalg.eigvalsh(0.5 * (s + s.T)).min() > 0 for s in S_out))

    # ── 6. `_spd_solve`: stesso numero della LU, e il ripiego quando serve ───
    # La strada veloce e quella vecchia devono coincidere DOVE ENTRAMBE VANNO.
    # Non bit a bit — arrotondano in ordine diverso — ma alla tolleranza che la
    # docstring dichiara.  Si prova su una `prec` mal condizionata come quelle
    # vere, dove `Omega` spazia su dodici ordini di grandezza.
    #
    # LA TOLLERANZA SCALA COL CONDIZIONAMENTO, e non e' indulgenza.  Cholesky e
    # LU sono entrambe stabili all'indietro: su un sistema con numero di
    # condizionamento k sbagliano ENTRAMBE di ~k*eps, e quindi differiscono di
    # altrettanto.  Una soglia costante boccerebbe il codice giusto appena k
    # cresce — provato: a k=1e10 lo scarto e' 1e-7, cioe' venti volte SOTTO il
    # limite teorico 2.2e-6, e una soglia fissa a 1e-7 lo dichiarava rotto.
    # Il fattore 20 lascia margine allo stimatore di k senza rendere il
    # controllo vuoto: a k=1e3 pretende ancora 4e-12.
    for cond, nome in ((1e3, "ben condizionata"), (1e10, "mal condizionata")):
        A = spd(60, np.random.default_rng(7), cond=cond)
        Bm = np.random.default_rng(8).standard_normal((60, 37))
        tol = 20 * cond * np.finfo(float).eps
        veloce, vecchia = _spd_solve(A, Bm), np.linalg.solve(A, Bm)
        scarto = (np.abs(veloce - vecchia).max()
                  / max(np.abs(vecchia).max(), 1e-300))
        check(f"_spd_solve == np.linalg.solve  ({nome})", scarto < tol,
              f"scarto relativo {scarto:.2e}, limite {tol:.2e}")
        # E deve RISOLVERE, non solo somigliare alla LU: A @ x == B.
        res = np.abs(A @ veloce - Bm).max() / max(np.abs(Bm).max(), 1e-300)
        check(f"_spd_solve risolve davvero A x = B  ({nome})", res < tol,
              f"residuo relativo {res:.2e}, limite {tol:.2e}")

    # Il ripiego: su una matrice NON definita positiva la Cholesky si arrende e
    # deve rispondere la LU.  Se questo ramo sparisse, la marginal likelihood si
    # fermerebbe invece di degradare — e fermarla ferma il Metropolis, non solo
    # l'estrazione di (B, Sigma).
    A = spd(20, np.random.default_rng(9))
    A[3, 3] = -abs(A[3, 3])                      # rompe la definitezza positiva
    Bm = np.random.default_rng(10).standard_normal((20, 4))
    try:
        got = _spd_solve(A, Bm)
        ok = np.allclose(got, np.linalg.solve(A, Bm))
    except Exception:                                             # noqa: BLE001
        ok = False
    check("su una matrice non definita positiva ripiega sulla LU", ok)

    print("=" * 74)
    if _fail:
        print(f"  {_fail} CONTROLLI ROTTI")
        return 1
    print("  tutto verde")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
