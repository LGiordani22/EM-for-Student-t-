"""
src/mcmc/constants.py
=====================

Tabulated mixture constants for the stochastic-volatility blocks of the Gibbs
sampler.  They are consumed by the KSC / Omori volatility blocks; with ``sv=False``
the sampler does not touch any of these tables.

Two tables live here (see ``docs/EM_for_student_t.tex`` ``tab:omori-mixture``):

* **KSC-7**  — the 7-component Gaussian mixture approximating the
  ``log chi^2_1`` density (Kim-Shephard-Chib 1998), used by the SV block
  *without* leverage (Passo 2) and by the contemporaneous-leverage branch
  (Passo 3).
* **Omori-10** — the 10-component mixture for the SV-with-leverage (lagged
  timing) branch (Omori-Chib-Shephard-Nakajima 2007, Tab. 1), used by Passo 4.
  Each component carries ``(q_j, m_j, v_j^2, a_j, b_j)``.

================================================================================
NOTA — VALIDAZIONE DELLE TABELLE (gia' implementata in ``validate_mixture`` e
coperta da ``test_passo4`` [1]).  I tre check vanno SEMPRE fatti **CON
TOLLERANZA**, MAI con uguaglianza stretta: le misture sono APPROSSIMAZIONI
numeriche della log-chi^2, i cui coefficienti NON soddisfano le identita' in
modo esatto, solo entro errore di tabulazione.  I check sono (np.isclose /
|.| < tol), MAI con ``==`` ne' confronti esatti:

    1.  Linearizzazione di Omori:   |b_j - a_j / 2| < 1e-4        (per ogni j)
        (approssimato, non esatto: b_j ~ a_j/2 solo a meno dell'errore di fit)

    2.  Media della mistura:        |sum_j q_j * m_j - (-1.2704)| < ~1e-3
        Valore di riferimento della media della log-chi^2_1 (= -1.2704...).
        NB: le due fonti tabulano costanti leggermente diverse e producono
            medie leggermente diverse, entrambe corrette entro tolleranza:
              -1.27028  (Omori et al. 2007)
              -1.27040  (Kim-Shephard-Chib 1998)
        => la tolleranza (~1e-3) deve coprire ENTRAMBE; non agganciarsi a una
           sola cifra.

    3.  Normalizzazione dei pesi:   |sum_j q_j - 1| < tolleranza numerica
        (es. 1e-8 / 1e-10): i pesi sommano a 1 solo a meno dell'arrotondamento
        con cui sono tabulati.

In sintesi: questi tre controlli vanno SEMPRE scritti con TOLLERANZA.  Un check
con uguaglianza stretta fallirebbe (o, peggio, passerebbe per caso su una sola
tabella e si romperebbe sull'altra) — non e' un bug delle costanti, e' la natura
approssimata della rappresentazione log-chi^2 come mistura di gaussiane.
================================================================================
"""

from __future__ import annotations

# Reference mean of the log chi^2_1 distribution.  Both mixtures are tuned to
# reproduce it; the two published tables land on slightly different values
# (Omori: -1.27028, KSC: -1.27040) — see the TODO above.  Kept here as a single
# named target so the (tolerant) validation does not hard-code a magic number.
LOG_CHI2_MEAN: float = -1.2704

# Tolerances for the three (tolerant!) consistency checks — see the TODO above.
# Exposed as named constants so the Passo-4 validator reads as "within TOL",
# never as an exact comparison.
TOL_OMORI_LINEARIZATION: float = 1e-4   # |b_j - a_j/2|   < this
TOL_MIXTURE_MEAN: float = 1e-3          # |sum q_j m_j - LOG_CHI2_MEAN| < this
TOL_WEIGHT_NORMALIZATION: float = 1e-8  # |sum q_j - 1|   < this

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# KSC-7 — seven-component Gaussian mixture approximating log chi^2_1 (Passo 2).
# Transcribed from docs/EM_for_student_t.tex, Table `tab:ksc-mixture` (the
# Kim-Shephard-Chib 1998 mixture as reproduced in Omori et al. 2007, Table 1,
# checked digit-by-digit there).  Order j = 1..7; signs as printed.
# ─────────────────────────────────────────────────────────────────────────────
KSC7: dict = {
    "q":  np.array([0.04395, 0.24566, 0.34001, 0.25750, 0.10556, 0.00002, 0.00730]),
    "m":  np.array([1.50746, 0.52478, -0.65098, -2.35859, -5.24321, -9.83726, -11.40039]),
    "v2": np.array([0.16735, 0.34023, 0.64009, 1.26261, 2.61369, 5.17950, 5.79596]),
}

# ─────────────────────────────────────────────────────────────────────────────
# Omori-10 — ten-component mixture for SV-with-leverage, lagged timing (Passo 4).
# Transcribed from docs/EM_for_student_t.tex, Table `tab:omori-mixture` (Omori,
# Chib, Shephard & Nakajima 2007, Table 1), where the values were entered from
# the source and checked digit-by-digit against the original PDF.  Order j=1..10;
# signs as printed.  {q,m,v2} approximate log chi^2_1 as in KSC; {a,b} are the
# per-component coefficients of the linear approximation of exp(xi/2) carrying
# the lagged leverage drift (b_j ~ a_j/2).
# ─────────────────────────────────────────────────────────────────────────────
OMORI10: dict = {
    "q":  np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
                    0.18842, 0.12047, 0.05591, 0.01575, 0.00115]),
    "m":  np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
                    -1.97278, -3.46788, -5.55246, -8.68384, -14.65000]),
    "v2": np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
                    0.98583, 1.57469, 2.54498, 4.16591, 7.33342]),
    "a":  np.array([1.01418, 1.02248, 1.03403, 1.05207, 1.08153,
                    1.13114, 1.21754, 1.37454, 1.68327, 2.50097]),
    "b":  np.array([0.50710, 0.51124, 0.51701, 0.52604, 0.54076,
                    0.56557, 0.60877, 0.68728, 0.84163, 1.25049]),
}


# ─────────────────────────────────────────────────────────────────────────────
# QML-10's single-Gaussian counterpart — the constants of the *coupled* leverage
# pass (``common_vol_coupling="qml"`` under ``leverage=True``, Branch B).
#
# The QML route (Harvey-Ruiz-Shephard 1994) replaces the mixture by ONE Gaussian
# matching the exact first two moments of xi = log z^2, z ~ N(0,1):
#
#     E[xi]   = psi(1/2) + log 2 = -gamma - log 2 = LOG_CHI2_MEAN
#     Var[xi] = pi^2 / 2                                    (LOG_CHI2_VAR)
#
# That buys a *constant* measurement covariance (pi^2/2) R_xi, which is what makes
# a FULL cross-factor R_xi tractable (the mixture cannot: a full R_xi does not
# factorise over the per-factor indicators — the 'literal' instability).  But the
# leverage drift needs exp(xi/2) = |z| to be LINEAR in xi (that is what Omori's
# per-component (a_j, b_j) deliver).  Its single-Gaussian counterpart is the best
# linear predictor of |z| given xi under the EXACT law — the same object Omori
# builds inside each component, built once, globally:
#
#     |z| ~= QML_A + QML_B * (xi - LOG_CHI2_MEAN)
#
#     QML_A = E|z|                = sqrt(2/pi)
#     QML_B = Cov(|z|, xi)/Var(xi)
#
# Both are closed-form (no table, no fitting).  With z ~ N(0,1):
#     E[|z| log z^2] = 2 E[|z| log|z|] = sqrt(2/pi) (log 2 - gamma)
#     E[|z|] E[xi]   = sqrt(2/pi) (-gamma - log 2)
#  => Cov(|z|, xi)   = sqrt(2/pi) * 2 log 2
#  => QML_B          = sqrt(2/pi) * 4 log 2 / pi^2 ~= 0.2241
# (``test_qml_leverage`` re-derives both by Monte Carlo.)
#
# Same role as (a_j, b_j), same algebra downstream — only the conditioning is
# coarser: one global line instead of ten local ones.  That is the *price* of the
# coupled pass, and the reason it is an option, not the default.
# ─────────────────────────────────────────────────────────────────────────────
LOG_CHI2_VAR: float = float(np.pi ** 2 / 2.0)          # pi^2/2 = Var(log chi^2_1)

QML_A: float = float(np.sqrt(2.0 / np.pi))             # E|z|            ~= 0.797885
QML_B: float = float(np.sqrt(2.0 / np.pi) * 4.0 * np.log(2.0) / np.pi ** 2)   # ~= 0.224149


def validate_mixture(mix: dict, *, has_linearization: bool = False) -> dict:
    """
    Run the three consistency checks of the TODO above **with tolerance**
    (never exact equality — the mixtures only approximate log chi^2_1).

    Returns a dict of (value, ok) pairs; ``has_linearization=True`` also checks
    ``|b_j - a_j/2|`` (Omori-10 only).  Raises ``AssertionError`` only on a
    *tolerant* failure, so a correctly tabulated table always passes.
    """
    q = np.asarray(mix["q"], dtype=float)
    m = np.asarray(mix["m"], dtype=float)
    sum_q = float(q.sum())
    sum_qm = float((q * m).sum())
    out = {
        "sum_q": (sum_q, abs(sum_q - 1.0) < TOL_WEIGHT_NORMALIZATION),
        "sum_qm": (sum_qm, abs(sum_qm - LOG_CHI2_MEAN) < TOL_MIXTURE_MEAN),
    }
    if has_linearization:
        a = np.asarray(mix["a"], dtype=float)
        b = np.asarray(mix["b"], dtype=float)
        out["lin"] = (float(np.max(np.abs(b - a / 2.0))),
                      bool(np.all(np.abs(b - a / 2.0) < TOL_OMORI_LINEARIZATION)))
    assert all(ok for _, ok in out.values()), f"mixture validation failed: {out}"
    return out
