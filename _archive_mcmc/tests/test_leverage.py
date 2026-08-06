"""
src/mcmc/tests/test_leverage.py
===============================

Il gate del **leverage** (Famiglia C, ``rho``) -- i due timing in un solo file.
Raccoglie, per la Famiglia C (``rho``):

  * i kernel e i controlli di **Branch A** (contemporaneo, Metropolis single-move);
  * i kernel e i controlli di **Branch B** (laggato, mistura Omori-10 + FFBS);
  * il gate **end-to-end condiviso** ``leverage_end_to_end`` -- una sola funzione,
    parametrizzata per ramo, cosi' una correzione (es. la semantica dell'acceptance
    di ``rho`` dopo il griddy) si fa in UN posto solo.

COPERTURA (parametri/percorsi -- notazione del README/.tex)
-----------------------------------------------------------
Livello RECOVERY del leverage (SEGNO, non magnitudine -> quella e' del validatore).
Recupera/copre:
  * rho^u   [Fam C comune]   segno del canale dominante, ordinamento dei canali
  * rho^eps [Fam C idio]     segno (mediana dei rho_eps < 0)
  * h^u_{1:T} sotto leverage (tracking del path)
  + kernel di Fam C (griddy/RW/Laplace), costanti Omori, skewness, DGP per-fattore.

Cosa verifica
-------------
  [1] costanti Omori-10 (tre check TOLLERANTI: sum q=1, sum q*m ~ -1.2704, b_j=a_j/2);
  [2] Famiglia C: ``draw_rho_scalar`` recupera un ``rho`` noto (kernel condiviso dai
      due rami -- su A regressore ``sigma*z``, su B il regressore di Omori ``sigma*g``);
  [3] nesting a ``rho=0``: Branch A path-MH senza drift; Branch B FFBS-tv == FFBS base;
  [4] griddy vs RW-Metropolis sullo stesso target (media e sd);
  [5] Branch B esce dallo start piatto ``log h=0`` senza warm-seed (Branch A non puo');
  [6] skewness del meccanismo: ``rho<0`` produce innovazioni left-skewed (il link col GaR);
  [7] il DGP per-fattore (Spec II): r volatilita' indipendenti, sd ordinate;
  [8] end-to-end per ramo: acceptance, ESS, **segno dominante di rho^u E rho^eps**,
      tracking del path h^u.

La magnitudine di ``rho`` (attenuata) NON e' un gate: la misura il validatore
(``mcmc.validate``).  Qui si asserisce il **segno** e le meccaniche.

Run
---
    python src/mcmc/tests/test_leverage.py
"""

from __future__ import annotations

import inspect
import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.constants import OMORI10, validate_mixture                     # noqa: E402
from mcmc.gibbs import load_warm_init, fit_dfm_mcmc                       # noqa: E402
from mcmc.simulate_sv import simulate_dfm_sv                             # noqa: E402
from mcmc.sample_vol import _scalar_ar1_ffbs                            # noqa: E402
from mcmc.sample_leverage_lagged import _ffbs_tv                        # noqa: E402
from mcmc.sample_leverage import (                                       # noqa: E402
    draw_rho, draw_rho_griddy, draw_rho_scalar, _lev_path_mh,
    _lev_path_laplace_mh_common, _lev_path_mh_mv_common, _logpost_A_common,
)
from mcmc.diagnostics import leverage_skewness_check                     # noqa: E402

_PASS = 0
_FAIL = 0


def _check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1; print(f"  [PASS] {name}")
    else:
        _FAIL += 1; print(f"  [FAIL] {name}   {detail}")


def _corr(x, y):
    xz, yz = x - x.mean(), y - y.mean()
    return float((xz @ yz) / (np.linalg.norm(xz) * np.linalg.norm(yz)))


def _diagonal_theta(theta):
    """Q diagonale: i whitening per-componente e full-Q coincidono, quindi Branch A e
    Branch B agganciano il leverage sullo *stesso* shock e il blocco di vol comune
    disaccoppiato e' esatto."""
    th = dict(theta)
    th["Q"] = np.diag(np.diag(np.asarray(theta["Q"], float)))
    th["Sigma_0"] = np.asarray(theta["Sigma_0"], float)
    return th


# -----------------------------------------------------------------------------
# Il gate end-to-end condiviso -- uno per ramo
# -----------------------------------------------------------------------------

#: I due rami, e cio' che li distingue *operativamente*.
BRANCH_CFG = {
    "A": {"timing": "contemporaneous", "T": 220, "n_iter": 800, "burn_in": 300,
          "path_is_metropolis": True},
    "B": {"timing": "lagged", "T": 500, "n_iter": 600, "burn_in": 250,
          "path_is_metropolis": False},
}

RHO_U_TRUE = np.array([-0.6, -0.3, -0.2])


def leverage_end_to_end(branch: str, theta, fl, bm, oc, r, check) -> dict:
    """Il gate end-to-end del leverage, **uno per ramo**.  ``check(nome, ok, dettaglio)``
    e' il reporter del chiamante, cosi' il conteggio pass/fail resta suo."""
    cfg = BRANCH_CFG[branch]
    rho_true = RHO_U_TRUE[:r]

    sim = simulate_dfm_sv(theta, T=cfg["T"], freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=9, sv_u=(0.0, 0.97, 0.22), sv_eps=(0.0, 0.95, 0.15),
                          rho_u=rho_true, rho_eps=-0.3, timing=cfg["timing"])
    res = fit_dfm_mcmc(sim["Y"], {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])},
                       fl, bm, oc, n_iter=cfg["n_iter"], burn_in=cfg["burn_in"], thin=1,
                       seed=4, sv=True, leverage=True, timing=cfg["timing"],
                       store_vol=True, verbose=False)
    acc = res["meta"]["acceptance"]

    # -- il path: Metropolis sotto A, draw diretto sotto B --------------------
    if cfg["path_is_metropolis"]:
        mh = [k for k in acc if not k.startswith("rho_")]
        check(f"[{branch}] acceptance Metropolis (path, sigma2) in (0.05, 0.95)",
              all(0.05 < acc[k] < 0.95 for k in mh),
              f"{ {k: round(acc[k], 2) for k in mh} }")
    else:
        check(f"[{branch}] il path FFBS e' un draw DIRETTO (acceptance == 1.0)",
              acc["path_u"] == 1.0 and acc["path_eps"] == 1.0, f"{acc}")
        check(f"[{branch}] sigma2 resta un Metropolis (acceptance in (0.05, 0.95))",
              0.05 < acc["sigma2"] < 0.95, f"sigma2={acc['sigma2']:.2f}")

    # -- Family C: il griddy accetta per costruzione, su ENTRAMBI i rami ------
    check(f"[{branch}] il griddy accetta per costruzione (rho acc == 1) -- NON e' una "
          f"diagnostica di mixing: si legge l'ESS",
          acc["rho_u"] == 1.0 and acc["rho_eps"] == 1.0,
          f"rho_u={acc['rho_u']}, rho_eps={acc['rho_eps']}")

    # -- nessun rho_hat senza il suo ESS --------------------------------------
    d = res["diagnostics"]["rho_u"]
    check(f"[{branch}] l'ESS di rho e' riportato (nessun rho_hat senza il suo ESS)",
          d["ess"].shape == (r,) and np.all(d["ess"] > 0),
          f"ESS={np.round(d['ess'], 1)}")

    # -- il segno dominante COMUNE.  NB: e' il SEGNO, non la MAGNITUDINE -- quest'ultima
    #    e' attenuata e la misura il validatore (mcmc.validate), non questo gate.
    rho_hat = res["theta_mean"]["rho_u"]
    j = int(np.argmax(np.abs(rho_true)))
    check(f"[{branch}] il canale comune dominante ha segno negativo",
          rho_hat[j] < 0, f"rho_hat={np.round(rho_hat, 3)}")

    # -- il segno IDIOSINCRATICO (rho_eps): il DGP mette rho_eps=-0.3 su tutte le M
    #    serie; la mediana dei rho_eps stimati deve restare negativa (segno, non valore).
    rho_eps_hat = np.asarray(res["theta_mean"]["rho_eps"])
    check(f"[{branch}] il leverage idiosincratico e' recuperato negativo (mediana < 0)",
          float(np.median(rho_eps_hat)) < 0.0,
          f"mediana rho_eps={float(np.median(rho_eps_hat)):.3f}")

    # -- i path per-fattore tracciano la volatilita' vera ---------------------
    lhu = np.log(res["draws"]["h_u"].mean(axis=0))
    truth = sim["logh_u_true"]
    if lhu.ndim == 2:
        if np.asarray(truth).ndim == 1:                 # DGP scalare-comune
            c = float(np.mean([np.corrcoef(lhu[:, k], truth)[0, 1]
                               for k in range(lhu.shape[1])]))
        else:
            c = float(np.mean([np.corrcoef(lhu[:, k], truth[:, k])[0, 1]
                               for k in range(lhu.shape[1])]))
        thr = 0.3 if branch == "A" else 0.4
        check(f"[{branch}] i path h^u per fattore tracciano il vero (corr media > {thr})",
              c > thr, f"corr={c:.3f}")
    return res


# -----------------------------------------------------------------------------
# [1] costanti Omori-10 (Branch B)
# -----------------------------------------------------------------------------

def test_constants():
    print("\n[1] Omori-10 constants present + tolerant consistency checks")
    ok_present = all(k in OMORI10 for k in ("q", "m", "v2", "a", "b")) and len(OMORI10["q"]) == 10
    _check("OMORI10 present with 10 components, keys q,m,v2,a,b", ok_present)
    try:
        out = validate_mixture(OMORI10, has_linearization=True)
        _check("sum q = 1 (tol)", out["sum_q"][1], f"{out['sum_q'][0]:.8f}")
        _check("sum q*m ~ -1.2704 (tol)", out["sum_qm"][1], f"{out['sum_qm'][0]:.5f}")
        _check("b_j = a_j/2 (tol)", out["lin"][1], f"max|b-a/2|={out['lin'][0]:.2e}")
    except AssertionError as e:
        _check("validate_mixture", False, str(e))


# -----------------------------------------------------------------------------
# [2] Famiglia C: il kernel scalare di rho (condiviso dai due rami)
# -----------------------------------------------------------------------------

def test_rho_kernel():
    print("\n[2] Family C: draw_rho_scalar recupera un rho noto (kernel dei due rami)")
    rng = np.random.default_rng(0)
    T = 4000
    rho_true, sigma2 = -0.5, 0.04
    sig = np.sqrt(sigma2)
    z = rng.standard_normal(T)
    eta = rho_true * sig * z + sig * np.sqrt(1 - rho_true ** 2) * rng.standard_normal(T)
    k = sig * z                                 # A: sigma*z; B: sigma*g (stesso kernel)
    rho = 0.0
    acc = 0.0
    draws = []
    for it in range(4000):
        rho, a = draw_rho_scalar(rho, eta, k, sigma2, 0.05, rng)
        acc += a
        if it >= 1000:
            draws.append(rho)
    mean = float(np.mean(draws))
    _check("posterior mean rho ~ true (-0.5)", abs(mean - rho_true) < 0.07,
           f"mean={mean:.3f}")
    _check("rho Metropolis acceptance in (0.1, 0.9)", 0.1 < acc / 4000 < 0.9,
           f"acc={acc/4000:.2f}")


# -----------------------------------------------------------------------------
# [3] nesting a rho=0 -- un ramo per volta
# -----------------------------------------------------------------------------

def test_nesting_A():
    print("\n[3a] Branch A: a rho=0 il path-MH non ha drift (deterministico dato rng)")
    rng = np.random.default_rng(1)
    T = 50
    logh = rng.standard_normal(T) * 0.2
    S = np.abs(rng.standard_normal(T))
    kdim = np.ones(T, int)
    has_obs = np.ones(T, bool); has_obs[0] = False
    # gcoef=0 (rho=0) -> nessun drift, var = sigma2.  Due chiamate con lo stesso rng
    # devono coincidere.
    x1, a1, _ = _lev_path_mh(logh, S, kdim, np.zeros(T), has_obs,
                             0.9, 0.04, 0.0, 0.2, np.random.default_rng(7))
    x2, a2, _ = _lev_path_mh(logh, S, kdim, np.zeros(T), has_obs,
                             0.9, 0.04, 0.0, 0.2, np.random.default_rng(7))
    _check("rho=0 path-MH is deterministic given rng (no drift terms)",
           np.allclose(x1, x2) and a1 == a2)


def test_ffbs_nesting_B():
    print("\n[3b] Branch B: a rho=0 il FFBS time-varying == FFBS base AR(1) (stesso rng)")
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    T = 60
    phi, sigma2 = 0.95, 0.05
    V_eff = np.full(T, 4.9)
    mask = np.ones(T, bool); mask[3] = False
    y_eff = np.linspace(-1, 1, T)
    x_base = _scalar_ar1_ffbs(y_eff, V_eff, mask, 0.0, phi, sigma2, rng_a)
    G = np.full(T, phi); c = np.zeros(T); W = np.full(T, sigma2)
    stat_var = sigma2 / (1 - phi * phi)
    x_tv = _ffbs_tv(y_eff, V_eff, mask, G, c, W, stat_var, rng_b)
    _check("rho=0 FFBS_tv == base FFBS (same RNG)", np.allclose(x_base, x_tv),
           f"max diff={np.max(np.abs(x_base-x_tv)):.2e}")


# -----------------------------------------------------------------------------
# [4] griddy vs RW-Metropolis sullo stesso target (Branch B, kernel di Family C)
# -----------------------------------------------------------------------------

def test_rho_griddy_same_target():
    print("\n[4] il griddy estrae dallo STESSO target della RW-Metropolis")
    rng = np.random.default_rng(0)

    # Su un (eta, k, sigma2) fisso la catena RW e i draw iid del griddy devono
    # concordare su media E sd -- non solo la media (che un kernel biased-ma-centrato
    # potrebbe pure azzeccare).
    for rho_true, n in ((-0.70, 600), (-0.15, 60), (0.45, 600), (0.0, 600)):
        s2 = 0.0625; sig = np.sqrt(s2)
        z = rng.standard_normal(n)
        eta = rho_true * sig * z + sig * np.sqrt(1 - rho_true ** 2) * rng.standard_normal(n)
        k = sig * z
        r_rw = 0.0; chain = []
        for it in range(20000):
            r_rw, _ = draw_rho_scalar(r_rw, eta, k, s2, 0.06, rng)
            if it >= 8000:
                chain.append(r_rw)
        gr = np.array([draw_rho_griddy(0.0, eta, k, s2, rng)[0] for _ in range(3000)])
        dm = abs(float(np.mean(chain)) - float(gr.mean()))
        ds = abs(float(np.std(chain)) - float(gr.std()))
        _check(f"rho_true={rho_true:+.2f}, n={n}: same posterior mean and sd",
               dm < 0.02 and ds < 0.02,
               f"|d mean|={dm:.4f}, |d sd|={ds:.4f}")

    # il griddy ignora il valore corrente (e' il punto)
    s2 = 0.05
    z = rng.standard_normal(300); eta = -0.5 * np.sqrt(s2) * z + np.sqrt(s2 * 0.75) * rng.standard_normal(300)
    k = np.sqrt(s2) * z
    a = draw_rho_griddy(-0.99, eta, k, s2, np.random.default_rng(5))[0]
    b = draw_rho_griddy(+0.99, eta, k, s2, np.random.default_rng(5))[0]
    _check("griddy draw is independent of the current value (no random walk)", a == b,
           f"from -0.99 -> {a:.4f};  from +0.99 -> {b:.4f}")
    _check("griddy always 'accepts' (flag = 1)",
           draw_rho_griddy(0.0, eta, k, s2, rng)[1] == 1)

    # il dispatcher, e il suo guard
    r1, _ = draw_rho(0.0, eta, k, s2, np.random.default_rng(9), sampler="griddy")
    r2, _ = draw_rho(0.0, eta, k, s2, np.random.default_rng(9), sampler="rw")
    _check("dispatcher: 'griddy' and 'rw' are different kernels", r1 != r2)
    try:
        draw_rho(0.0, eta, k, s2, rng, sampler="hmc"); ok = False
    except ValueError:
        ok = True
    _check("dispatcher: unknown rho_sampler raises", ok)

    # un log_prior piatto non deve muovere il draw (il hook Fisher-z e' additivo)
    c = draw_rho_griddy(0.0, eta, k, s2, np.random.default_rng(11))[0]
    d = draw_rho_griddy(0.0, eta, k, s2, np.random.default_rng(11), log_prior=lambda r: 0.0)[0]
    _check("flat log_prior leaves the draw bit-identical", c == d)


# -----------------------------------------------------------------------------
# [5] Branch B esce dallo start piatto senza warm-seed (Branch A no)
# -----------------------------------------------------------------------------

def test_branch_b_flat_start_immunity(theta, fl, bm, oc, r):
    print("\n[5] Branch B esce dallo start piatto log h=0 SENZA warm-seed (A non puo')")
    import mcmc.sample_leverage_lagged as lagged
    import mcmc.sample_leverage as contemp

    src_b = inspect.getsource(lagged)
    src_a = inspect.getsource(contemp)
    _check("Branch A porta il warm-seed di soccorso (il rescue del path-MH)",
           "sample_common_vol_mv" in src_a)
    _check("Branch B NON porta warm-seed: non gli serve (FFBS diretto)",
           "sample_common_vol_mv" not in src_b)

    # ... e non e' solo assente, e' non necessario: da log h == 0 la prima spazzata e' un
    # salto Omori/FFBS pieno (rho=0 => G=phi, c=0, W=sigma2), quindi il sampler lascia
    # subito lo stato piatto e recupera un phi persistente.  Branch A, senza seed, resta
    # nella trappola h~1 -> stati omoschedastici -> u~omosk. -> h~1.
    sim = simulate_dfm_sv(theta, T=500, freq_list=fl, block_map=bm, ordered_cols=oc, r=r,
                          seed=9, sv_u=(0.0, 0.97, 0.22), sv_eps=(0.0, 0.95, 0.15),
                          rho_u=np.array([-0.6, -0.3, -0.2])[:r], rho_eps=-0.3,
                          timing="lagged")
    res = fit_dfm_mcmc(sim["Y"], {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])},
                       fl, bm, oc, n_iter=400, burn_in=150, seed=4,
                       sv=True, leverage=True, timing="lagged",
                       store_vol=True, verbose=False)
    logh = np.log(res["draws"]["h_u"])                 # (n_keep, T, r)
    phi = res["theta_mean"]["sv_u"][:, 1]
    _check("Branch B leaves the flat path (h is not stuck at 1)",
           float(np.mean(np.std(logh, axis=1))) > 0.1,
           f"mean sd(log h) = {float(np.mean(np.std(logh, axis=1))):.3f}")
    _check("Branch B recovers a persistent phi from the flat start (all > 0.7)",
           np.all(phi > 0.7), f"phi={np.round(phi,3)}")
    _check("Branch B never degrades phi to negative (the Branch-A failure mode)",
           np.all(phi > 0.0), f"phi={np.round(phi,3)}")


# -----------------------------------------------------------------------------
# [6] skewness del meccanismo -- un ramo per volta (il link col GaR)
# -----------------------------------------------------------------------------

def test_skewness_A(theta):
    print("\n[6a] Branch A: rho<0 -> innovazioni di fattore left-skewed")
    sk = leverage_skewness_check(theta, sv_u=(0.0, 0.95, 0.15),
                                 rho_u=np.array([-0.6, -0.4, -0.3]),
                                 T=10000, seed=3)
    _check("rho<0 produces left skew (< -0.1)", sk["skew_leverage"] < -0.1,
           f"skew={sk['skew_leverage']:.3f}")
    _check("rho=0 is ~symmetric (|skew|<0.1)", abs(sk["skew_symmetric"]) < 0.1,
           f"skew={sk['skew_symmetric']:.3f}")


def test_skewness_B(theta):
    print("\n[6b] Branch B (lagged): rho<0 -> innovazione cumulata left-skewed")
    sk = leverage_skewness_check(theta, sv_u=(0.0, 0.97, 0.20),
                                 rho_u=np.array([-0.7, -0.5, -0.4]),
                                 T=12000, seed=3, timing="lagged")
    _check("rho<0 produces left skew (< -0.05)", sk["skew_leverage"] < -0.05,
           f"skew={sk['skew_leverage']:.3f} (window={sk['window']})")
    _check("rho=0 is ~symmetric (|skew|<0.07)", abs(sk["skew_symmetric"]) < 0.07,
           f"skew={sk['skew_symmetric']:.3f}")


# -----------------------------------------------------------------------------
# [7] il DGP per-fattore (Spec II): r volatilita' indipendenti (salvato da perfactor)
# -----------------------------------------------------------------------------

_SV_U_PF = np.array([[0.0, 0.97, 0.25],
                     [0.0, 0.92, 0.18],
                     [0.0, 0.95, 0.22]])
_T_PF = 600


def test_dgp_shape():
    print("\n[7] il DGP Spec II: r volatilita' per-fattore indipendenti")
    w = load_warm_init("small")
    theta = _diagonal_theta(w["theta"])
    fl, bm, oc, r = w["freq_list"], w["block_map"], w["ordered_cols"], w["r"]
    sim = simulate_dfm_sv(theta, T=_T_PF, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=21, sv_u=(0.0, 0.95, 0.0), sv_u_perfactor=_SV_U_PF[:r],
                          sv_eps=(0.0, 0.94, 0.12), rho_u=np.array([-0.70, -0.15, 0.45])[:r],
                          rho_eps=-0.3, timing="contemporaneous")
    lh = sim["logh_u_true"]
    _check("logh_u_true is (T, r)", lh.shape == (_T_PF, r), f"shape={lh.shape}")
    off = [abs(_corr(lh[:, j], lh[:, k])) for j in range(r) for k in range(j + 1, r)]
    _check("the r true paths are independent (|corr| < 0.25)", max(off) < 0.25,
           f"max|corr|={max(off):.3f}")
    sd = lh.std(axis=0)
    _check("per-factor unconditional sd ordered as sigma_k/sqrt(1-phi_k^2)",
           np.argmax(sd) == int(np.argmax(_SV_U_PF[:r, 2] / np.sqrt(1 - _SV_U_PF[:r, 1] ** 2))),
           f"sd={np.round(sd,3)}")


# -----------------------------------------------------------------------------
# [8] end-to-end per ramo (il gate condiviso)
# -----------------------------------------------------------------------------

def test_end_to_end_A(theta, fl, bm, oc, r):
    print("\n[8a] Branch A end-to-end: path Metropolis, segno dominante, tracking")
    leverage_end_to_end("A", theta, fl, bm, oc, r, _check)


def test_end_to_end_B(theta, fl, bm, oc, r):
    print("\n[8b] Branch B end-to-end: FFBS accept=1, segno dominante, tracking")
    leverage_end_to_end("B", theta, fl, bm, oc, r, _check)


# -----------------------------------------------------------------------------
# [9] Branch A, proposta a blocco (Laplace + Metropolis esatto): INVARIANZA
# -----------------------------------------------------------------------------
# La tesi non e' "mescola meglio", e' "campiona la STESSA legge": una proposta puo'
# essere rozza, se il rapporto di Metropolis e' giusto la legge invariante non si
# muove.  L'efficienza si MISURA, non si asserisce.

def _toy_target(T=60, r=2, seed=3):
    """Blocco comune di Branch A in miniatura: parametri e shock fissi, un solo target
    esatto -- cosi' i due kernel si confrontano sulla *stessa* legge."""
    rng = np.random.default_rng(seed)
    phi = np.array([0.95, 0.90])[:r]
    sigma2 = np.array([0.05, 0.08])[:r]
    rho = np.array([-0.60, 0.40])[:r]
    Q = np.array([[1.0, 0.3], [0.3, 1.2]])[:r, :r]          # Q NON diagonale: il
    qdiag = np.diag(Q)                                      # target e' quello coupled
    vals, vecs = np.linalg.eigh(Q)
    Qinv_half = vecs @ np.diag(vals ** -0.5) @ vecs.T
    has = np.zeros(T, bool); has[1:] = True
    b = np.zeros((T, r))
    b[1:] = rng.standard_normal((T - 1, r)) @ np.linalg.cholesky(Q).T
    E = b / np.sqrt(qdiag)[None, :]
    S = E ** 2
    return dict(b=b, S=S, E=E, has=has, Qinv_half=Qinv_half,
                phi=phi, sigma2=sigma2, rho=rho, T=T, r=r)


def _sweep(kernel, x, tg, rng):
    if kernel == "single":
        return _lev_path_mh_mv_common(x, tg["b"], tg["S"], tg["has"], tg["Qinv_half"],
                                      tg["phi"], tg["sigma2"], tg["rho"], 0.25, rng)
    return _lev_path_laplace_mh_common(x, tg["b"], tg["S"], tg["E"], tg["has"],
                                       tg["Qinv_half"], tg["phi"], tg["sigma2"],
                                       tg["rho"], rng)


def _run_chain(kernel, tg, n, seed, keep_from=None):
    rng = np.random.default_rng(seed)
    x = np.zeros((tg["T"], tg["r"]))
    keep_from = n // 2 if keep_from is None else keep_from
    keep = []
    for i in range(n):
        x, _, _ = _sweep(kernel, x, tg, rng)
        if i >= keep_from:
            keep.append(x.copy())
    return np.stack(keep)                                   # (n_keep, T, r)


def test_laplace_invariance(n_chains=4, n_single=3000, n_block=500):
    """I due kernel sono Metropolis sullo STESSO target esatto: la loro legge
    stazionaria coincide.  Il kernel a blocco e' nuovo; se la sua proposta o il suo
    rapporto fossero sbagliati convergerebbe a una legge *diversa* -- e si vedrebbe qui.
    La soglia non e' cablata: si stima l'errore standard dalla dispersione fra catene
    indipendenti e si confronta lo scarto con 3*SE (test a due campioni autocalibrato)."""
    print("\n[9] Branch A blocco Laplace: invarianza (stessa legge del single-move)")
    tg = _toy_target(T=40, r=2)

    def _group(kernel, n, seed0):
        means = np.zeros((n_chains, tg["r"]))
        sds = np.zeros((n_chains, tg["r"]))
        for c in range(n_chains):
            X = _run_chain(kernel, tg, n, seed=seed0 + c)
            means[c] = X.mean(axis=(0, 1))
            sds[c] = X.std(axis=(0, 1))
        return means, sds

    ms, ss = _group("single", n_single, 100)
    mb, sb = _group("laplace", n_block, 200)

    def _two_sample(As, Bs, label):
        ok = True
        detail = []
        for k in range(tg["r"]):
            a, b = As[:, k], Bs[:, k]
            se = float(np.sqrt(a.var(ddof=1) / n_chains + b.var(ddof=1) / n_chains))
            d = float(abs(a.mean() - b.mean()))
            if se > 0 and d > 3.0 * se:
                ok = False
            detail.append(f"k={k}: {a.mean():+.3f} vs {b.mean():+.3f} "
                          f"(|d|={d:.3f}, 3SE={3 * se:.3f})")
        _check(f"invarianza: {label} -- single vs blocco entro 3 SE",
               ok, " | ".join(detail))

    _two_sample(ms, mb, "media del path")
    _two_sample(ss, sb, "sd del path")

    kw = dict(b=tg["b"], S=tg["S"], has_obs=tg["has"], Qinv_half=tg["Qinv_half"],
              phi=tg["phi"], sigma2=tg["sigma2"], rho=tg["rho"])
    Xs = _run_chain("single", tg, n_single, seed=11)
    Xl = _run_chain("laplace", tg, n_block, seed=12)
    lp_s = float(np.mean([_logpost_A_common(x, **kw) for x in Xs[::20]]))
    lp_l = float(np.mean([_logpost_A_common(x, **kw) for x in Xl[::10]]))
    _check("invarianza: stesso livello medio del log-target esatto",
           abs(lp_s - lp_l) < 0.05 * abs(lp_s),
           f"single {lp_s:.1f} vs blocco {lp_l:.1f}")


def test_laplace_block_beats_single():
    """Il *punto* della riparazione, misurato: a parita' di sweep il kernel a blocco
    muove il path molto di piu' (autocorr di lag 1 del livello medio)."""
    print("\n[9b] Branch A blocco Laplace: efficienza (batte il single-move)")
    tg = _toy_target()

    def _ac1(kernel, n=600, seed=5):
        X = _run_chain(kernel, tg, n, seed, keep_from=n // 3)
        v = X[:, :, 0].mean(axis=1)
        v = v - v.mean()
        return float((v[:-1] * v[1:]).sum() / (v * v).sum())

    ac_s = _ac1("single")
    ac_b = _ac1("laplace")
    _check("il blocco batte il single-move (autocorr lag 1)",
           ac_b < ac_s, f"blocco {ac_b:.3f} vs single {ac_s:.3f}")


def test_laplace_no_warm_seed():
    """Dallo start PIATTO il kernel a blocco atterra su un path non degenere in UNO
    sweep -- senza alcun warm seed KSC."""
    print("\n[9c] Branch A blocco Laplace: nessun warm-seed necessario")
    tg = _toy_target()
    rng = np.random.default_rng(7)
    x0 = np.zeros((tg["T"], tg["r"]))
    x1, n_acc, n_prop = _sweep("laplace", x0, tg, rng)
    _check("un accept/reject a blocco per fattore", n_prop == tg["r"],
           f"n_prop={n_prop}, atteso {tg['r']}")
    _check("dal warm start piatto il path esce non degenere in 1 sweep",
           n_acc >= 1 and np.std(x1) > 0.1,
           f"n_acc={n_acc}, sd(path)={np.std(x1):.3f}")


def main():
    print("=" * 72)
    print("LEVERAGE -- Famiglia C (rho): Branch A (contemp.) + Branch B (lagged/Omori)")
    print("=" * 72)
    w = load_warm_init("small")
    theta = dict(w["theta"])
    fl, bm, oc, r = w["freq_list"], w["block_map"], w["ordered_cols"], w["r"]
    # dal piu' veloce al piu' lento
    test_constants()
    test_rho_kernel()
    test_nesting_A()
    test_ffbs_nesting_B()
    test_laplace_invariance()
    test_laplace_block_beats_single()
    test_laplace_no_warm_seed()
    test_rho_griddy_same_target()
    test_dgp_shape()
    test_skewness_A(theta)
    test_skewness_B(theta)
    test_branch_b_flat_start_immunity(theta, fl, bm, oc, r)
    test_end_to_end_A(theta, fl, bm, oc, r)
    test_end_to_end_B(theta, fl, bm, oc, r)
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
