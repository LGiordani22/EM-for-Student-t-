"""
src/mcmc/sample_asis.py
=======================

SISTEMA (equazioni dal .tex — notazione originale)
--------------------------------------------------
Cosa calcola: ri-estrae i parametri Famiglia~B (φ, σ_η²) di UN processo di log-vol
interlacciando due parametrizzazioni, per rompere la cresta path/scala e alzare l'ESS.
Wrapper sul draw Famiglia~B, μ=0.  Transizione centrata (CP):

    log h_t = φ log h_{t-1} + η_t,   η_t ~ N(0, σ_η²)             [eq:sv-logvol-u/eps]

Riparametrizzazione NON-centrata (NCP)  x̃_t = log h_t / σ_η: σ_η migra nell'equazione
di MISURA come coefficiente di regressione gaussiano —

    y*_t - m_{s_t} = σ_η · x̃_t + errore(v²_{s_t}),   x̃_t = φ x̃_{t-1} + N(0,1)   [eq:asis-ncp]
                                                     (+ drift ρ z_t sotto leverage)

Si ridisegna (σ_η signed, φ) in NCP e si riscala indietro  log h_t = σ_η x̃_t.  Le due
parametrizzazioni sono quasi-indipendenti nelle loro direzioni cattive ⇒ il kernel
interlacciato è veloce se ANCHE UNA SOLA delle due mescola bene (Yu–Meng 2011).

Ancillarity--Sufficiency Interweaving Strategy (ASIS) for the Family~B parameters
``(phi, sigma_eta^2)`` of a single log-volatility process — the mixing booster of
``docs/EM_for_student_t.tex`` §"Boosting the Mixing" (``sec:asis``).

Why
---
Step (b) draws the log-vol path ``x = log h`` given ``(phi, sigma_eta^2)``; Family~B
draws ``(phi, sigma_eta^2)`` given the path.  These sit on a narrow **path/scale
ridge** (a smoother path forces a smaller ``sigma_eta^2``, which redraws a smoother
path), so single-block Gibbs crawls along it — the common log-vol's ``phi`` shows
ESS ~ a *percent* and split-Rhat > 1.1 (``subsec:asis-diagnosis``).  Leverage
*tightens* the ridge (``sigma_eta`` now sits in the drift **and** the innovation),
so ASIS is close to necessary there (``subsec:asis-leverage``).

What
----
ASIS is a **wrapper on the Family~B draw**, per process, ``mu = 0`` (only the scale
is interwoven — no level, ``subsec:asis-cp-ncp``).  Given the CP-drawn
``sigma_eta`` and the path, it re-expresses the same posterior in the
**non-centred** coordinates ``x_tilde = x / sigma_eta`` — where ``sigma_eta``
migrates into the *measurement* equation as a Gaussian **regression coefficient**
(``eq:asis-ncp`` / ``eq:asis-ncp-lev``) — and redraws ``(sigma_eta, phi)`` there,
then rescales back.  The two draws are near-independent in their bad directions, so
the interwoven kernel is fast whenever *either* parametrisation mixes well (Yu--Meng
2011): worst-case robust.

The move (steps 2--4 of ``subsec:asis-move``; step 1, the CP draw, is done by the
caller's leverage-aware Family~B):
  (2) rescale to NCP with the just-drawn ``sigma_eta``: ``x_tilde = x / sigma_eta``;
  (3) NCP redraw — ``sigma_eta`` **signed** = coefficient of the weighted regression
      of ``(y*_t - m_{s_t})`` on ``x_tilde_t`` with known variances ``v^2_{s_t}`` and
      a **Gaussian prior** ``N(0,B)`` on the signed scale (conjugate here, and the
      same half-Normal prior Family~B uses in CP — ``subsec:asis-move``); and ``phi``
      = AR(1) coefficient of ``x_tilde`` with drift ``rho z_t`` and variance
      ``(1-rho^2)``;
  (4) rescale back with the new ``sigma_eta``: ``x = sigma_eta x_tilde``.

Leverage: the drift ``rho z_t`` carries no ``sigma_eta`` in NCP (it divides through),
so ``sigma_eta`` still migrates *entirely* into the measurement — ASIS survives the
coupling.  The residual state-dependence of ``z_t = e_t e^{-x_t/2}`` is handled the
way the block already handles it: **``z_t`` is frozen** at its current value during
the NCP redraw (``subsec:asis-leverage``), so ``sigma_eta`` is a clean measurement
regression conditional on the frozen ``z``.  ``rho`` is *not* interwoven (drawn in
Family~C) but benefits through its posterior correlation with ``sigma_eta``.

The signed ``sigma_eta``: in NCP ``(sigma_eta, x_tilde)`` and ``(-sigma_eta,
-x_tilde)`` give the same ``x`` and the same measurement, so ``sigma_eta`` is drawn
on the whole line and the sign carried into ``x = sigma_eta x_tilde`` — this *helps*,
letting the chain flip an otherwise-sticky sign (``subsec:asis-move``).
"""

from __future__ import annotations

import numpy as np

from mcmc.constants import KSC7
from mcmc.sample_leverage import _draw_phi_lev


def _draw_ksc_indicators(resid: np.ndarray, ksc: dict, rng: np.random.Generator) -> np.ndarray:
    r"""
    Draw the KSC 7-component mixture indicators ``s_t`` given the current path,
    from ``s_t \propto q_j N(resid_t; m_j, v^2_j)`` (``eq:vol-indicator-cond``),
    where ``resid_t = y*_t - x_t`` is the log-square minus the current log-vol.
    Gumbel-max sampling (one categorical per period).
    """
    m = ksc["m"]; v2 = ksc["v2"]; q = ksc["q"]
    d = resid[:, None] - m[None, :]                       # (n, 7)
    logp = np.log(q)[None, :] - 0.5 * np.log(v2)[None, :] - 0.5 * d * d / v2[None, :]
    return np.argmax(logp + rng.gumbel(size=logp.shape), axis=1)


def asis_scale_interweave(
    x: np.ndarray,
    y_star: np.ndarray,
    has_obs: np.ndarray,
    sigma2_cp: float,
    rho: float,
    z: np.ndarray,
    rng: np.random.Generator,
    *,
    has_lev: np.ndarray | None = None,
    half_normal_B: float = 1.0,
    ksc: dict = KSC7,
) -> tuple[np.ndarray, float, float]:
    r"""
    ASIS steps (2)--(4) for **one** process: given the CP-drawn ``sigma2_cp`` and
    the current path ``x`` (plus leverage ``rho`` and the **frozen** raw shock
    ``z``), redraw the signed ``sigma_eta`` and ``phi`` in the non-centred
    parametrisation and rescale the path.  ``mu = 0``.

    Parameters
    ----------
    x : (T,)              current CP log-vol path ``x_t = log h_t``.
    y_star : (T,)         KSC log-square ``y*_t = log(e_t^2 + c)`` (used where
                          ``has_obs``; the value off the observed set is ignored).
    has_obs : (T,) bool   periods carrying a KSC measurement (``t = 1..T-1`` for the
                          common factor; the observed set for a series).
    sigma2_cp : float     the CP-drawn ``sigma_eta^2`` (step 1) — the rescale anchor.
    rho : float           current leverage (0 = no leverage).
    z : (T,)              current raw-shock regressor (frozen during the redraw);
                          ``rho * z`` is the NCP transition drift.  Contemporaneous
                          (Branch A): ``z_t`` on the transition into ``t``; lagged
                          (Branch B): the Omori regressor ``g_{t-1}`` so the drift
                          ``rho g_{t-1}`` corrects the transition into ``t``.  Pass
                          zeros when ``rho = 0``.
    has_lev : (T,) bool or None  transitions that carry the leverage drift, for the
                          NCP ``phi`` draw.  ``None`` ⇒ ``has_obs`` (contemporaneous /
                          no-leverage, where measurement and leverage coincide);
                          under lagged timing pass the shifted mask (leverage into
                          ``t >= 2``), distinct from the measurement mask ``has_obs``.
    half_normal_B : float variance ``B`` of the Gaussian prior ``sigma_eta ~ N(0,B)``.

    Returns
    -------
    (x_new (T,), phi_new, sigma2_new) — the rescaled path and the NCP-drawn
    ``(phi, sigma_eta^2)``.  Degenerate ``sigma2_cp <= 0`` returns the input
    unchanged (``phi_new = None``).
    """
    x = np.asarray(x, float)
    sig_cp = float(np.sqrt(sigma2_cp)) if sigma2_cp > 0 else 0.0
    if not np.isfinite(sig_cp) or sig_cp <= 0.0:
        return x, float("nan"), float(sigma2_cp)

    # (2) rescale to NCP
    x_tilde = x / sig_cp                                   # (T,)

    idx = np.where(np.asarray(has_obs, bool))[0]
    # (3a) KSC indicators given the current path x (resid = y* - x)
    resid = np.asarray(y_star, float)[idx] - x[idx]
    s = _draw_ksc_indicators(resid, ksc, rng)
    m_s = ksc["m"][s]; v2_s = ksc["v2"][s]

    # (3b) signed sigma_eta: weighted regression of (y* - m_s) on x_tilde,
    #      variances v2_s, Gaussian prior N(0, B)  (eq:asis-ncp / -lev).
    xt = x_tilde[idx]
    resp = np.asarray(y_star, float)[idx] - m_s
    prec = 1.0 / half_normal_B + float(np.sum(xt * xt / v2_s))
    mean = float(np.sum(xt * resp / v2_s)) / prec
    sigma_new = mean + np.sqrt(1.0 / prec) * rng.standard_normal()      # SIGNED

    # (3c) phi in NCP: AR(1) of x_tilde with drift rho z (frozen), variance (1-rho^2).
    #      _draw_phi_lev with sigma2 = 1 gives drift sqrt(1)*zeta = rho z and
    #      innovation variance 1*(1-rho^2) — exactly the NCP state (eq:asis-ncp-lev).
    #      has_lev marks the leverage-bearing transitions (lagged: distinct from the
    #      measurement mask has_obs; contemporaneous/no-lev: the same mask).
    rho = float(rho)
    zeta = rho * np.asarray(z, float)
    lev_mask = np.asarray(has_obs if has_lev is None else has_lev, bool)
    phi_new = _draw_phi_lev(x_tilde, zeta, lev_mask, 1.0, rho * rho, rng)

    # (4) rescale back with the new (signed) sigma_eta
    x_new = sigma_new * x_tilde
    return x_new, float(phi_new), float(sigma_new * sigma_new)
