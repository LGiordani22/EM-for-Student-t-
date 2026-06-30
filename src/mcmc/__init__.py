"""
src/mcmc/ — MCMC (Gibbs) estimation engine for the Student-t DFM with
stochastic volatility and leverage (master cell D1-b x D2-b).

The full sampler lives in :mod:`mcmc.gibbs` (:func:`fit_dfm_mcmc`): a complete
sweep over states (FFBS), volatility paths (KSC / Omori), Student-t weights and
parameters, with stochastic volatility and leverage (both timings) selectable by
flag.  ``shared.py`` collects the reusable mathematical kernels that, in the EM
code, live *inline* inside the E-/M-step functions; they are extracted here as
standalone, dependency-light helpers (numpy + scipy only) so the Gibbs blocks
call them without importing the EM machinery.

The EM modules (``em_e_step``, ``em_m_step``, ``kalman``, ...) are left
BYTE-FOR-BYTE UNCHANGED: this package is *additive*.  Each extracted helper is
verified against its EM counterpart in ``test_shared.py`` (same formula;
draw-vs-mean or extracted-vs-inline).
"""
