# MCMC sampler refactor — theory→code map

**Scope.** The estimation engine only (`src/mcmc/`). The second-stage density /
Growth-at-Risk module is a *downstream consumer* of the posterior draws and is
out of scope here; the only interface it needs is the draws dict produced by
`fit_dfm_mcmc` (see §7).

**How to use.** Each phase is dependency-ordered and self-contained. After every
phase the recovery tests (`test_passo1–4.py`, `test_shared.py`) must stay green —
they are the regression net, not something to rewrite first. Do **one phase per
PR/commit**, verify, then move on.

> **⚠️ WORKING RULE — always reference the `.tex`, always, always, always.**
> Before writing (or changing) any piece of code, **re-read the specific theory
> block in `docs/EM_for_student_t.tex` that governs that piece** — the exact
> equations and their surrounding derivation — and implement against it, citing
> the `eq:...`/`sec:...` labels in the code comments. Do this **piece by piece**,
> not once per phase: each function or sub-step gets its own theory re-read first.
> The `.tex` is the single source of truth; if code and text disagree, stop and
> reconcile before proceeding.
>
> **This plan is an INDEX, not a spec.** Its per-piece descriptions are
> *hypotheses* to verify against the `.tex`, never something to code from. Memory
> and this map are both lossy — the only loss-free step is opening the governing
> block at the moment of implementing. When the `.tex` disagrees with a piece's
> description here, the `.tex` wins and this file is corrected (e.g. 1d was
> re-sized from "r scalar loops" to a multivariate r-dim FFBS on 2026-07-08).
> A one-time **decomposition audit** of 15409–22070 precedes heavy coding, to
> catch mis-sized pieces before they force a reorder mid-refactor.

**Theory reference.** `docs/EM_for_student_t.tex`. Equation labels are cited as
`eq:...`. The estimation half was re-read 2026-07; the deltas below are grounded
in the actual code state (file:line) as of this map.

---

## 0. The one structural change that drives everything

The current sampler carries a **single common volatility** — the degenerate
`H^u_t = h^u_t I` case, where the Spec-I inside-sandwich `Q^{1/2} H Q^{1/2}` and
the Spec-II outside-sandwich `√H Q √H` *coincide* (both `= h^u_t Q`). (Spec I is
**not** "a scalar" — it is the per-factor sandwich with the *inside* placement;
the current code is the further `H=hI` restriction.) In code: `gibbs.py:178`
`logh_u = np.zeros(T)`, comment *"M+1 processes"*, `g_u = w_u/h_u` scalar (`:272`).

The theory now specifies **Spec II**: `r` **per-factor** common volatilities in
the sandwich

```
Var(u_t | ·) = (1/w^u_t) · √H^u_t · Q · √H^u_t ,   H^u_t = diag(h^u_1t,…,h^u_rt)
```

(`eq:sv-baseline-state` region, `eq:param-hu-matrix`). Consequences that ripple
through the whole sweep:

- `h_u`, `logh_u`: `(T,)` → `(T, r)`; `sv_u`: `(3,)` → `(r, 3)`.
- combined precision on the factor side is a **matrix**
  `G^u_t = w^u_t · H^{-1/2} Q^{-1} H^{-1/2}` (`eq:param-combined-weight-u`), no
  longer the scalar `g_u = w_u/h_u`.
- the `A`-draw stops being a closed MNIW and becomes a **vectorised Gaussian**
  (sum of Kronecker products, `eq:param-A-vec-model`, `eq:param-A-precision`);
  `Q` stays inverse-Wishart with matrix-whitened residuals (`eq:param-Q-post`).
- `rho_u` is *already* `(r,)` (`gibbs.py:185`) — the leverage side is half-migrated;
  finishing Spec II makes `h_u` and `rho_u` consistent.

**Everything below (priors, ASIS, leverage Option A, Family B M+r) layers on this.**
Do Spec II first.

---

## 1. Per-module delta table (current → target)

| Module | Current state (file:line) | Target (theory) | Change |
|---|---|---|---|
| `gibbs.py` | scalar `h_u (T,)`, `sv_u (3,)`, `g_u=w_u/h_u`; `sv_prior_b=0.05` IG; `nu_bounds=(2.001,1e3)` flat; no ASIS | Spec II per-factor; half-Normal σ_η; proper ν prior; ASIS wrapper | state shapes `(T,r)`/`(r,3)`; matrix `G^u_t`; wire ASIS; pass `log_prior` |
| `shared.py: draw_A_Q` | flat MNIW, scalar `T_eff`, `Q⊗inv(P00)` (`:267`) | matrix-whitened, vectorised `vec(A)` + IW `Q`; proper priors `Ψ0,ν0,V0` | rewrite for `G^u_t` sum-of-Kronecker; add prior args; keep flat limit as test seam |
| `shared.py: draw_lambda_r_series` | flat NIG (`:356`) | NIG with `a0,b0,m0,M0` | add proper-prior args (flat limit = current) |
| `shared.py: realized_deflated_d_u` | deflate by scalar `h_u` (`:130`) | deflate by per-factor `H^u_t` | index `h_u[t]` → `h_u[t,i]` per factor |
| `sample_params.py: draw_A_Q_block` | scalar `g_u`, `T-1` MNIW (`:60`) | per-factor whitening + vec(A) | feed `H^u`, build `G^u_t`; call new `draw_A_Q` |
| `sample_vol.py` | scalar common KSC; `IG(2,0.05)` σ²; μ via `beta_hat[0]/(1-phi)` | `r` per-factor common sub-sweeps; half-Normal σ_η; μ=0 | loop common over `r` (like idiosyncratic already); swap prior; enforce μ=0 |
| `sample_leverage.py` | `draw_rho_scalar` + `draw_rho_vec` (ρ'ρ<1, `:240`); `common_lev_scalar` | Option A: `r` scalar ρ, per-factor whitened `z^u_i` (`eq:lev-cond-common`) | **remove** `draw_rho_vec`; per-factor scalar loop; retire `common_lev_scalar` |
| `sample_leverage_lagged.py` | Branch B (Omori) under Spec I | Branch B under Spec II; Omori 10-comp (Table in .tex) | adapt to per-factor; verify Omori constants; common leverage starts one step late |
| `sample_params.py: draw_nu_griddy` | `log_prior=None` flat over bracket (`:152`) | proper prior on (2,∞) via hook | pass exponential/uniform `log_prior`; optionally tighten bracket |
| *(new)* `sample_asis.py` | — | ASIS wrapper on Family B (`sec:asis`) | CP draw → rescale NCP → redraw σ_η (meas. regression) + φ → rescale back; z_t frozen under leverage; per-process M+r |

---

## 2. Phased plan (dependency-ordered)

### Phase 1 — Spec II: per-factor common volatility  ⟵ foundational
Piece by piece (re-read the cited `.tex` block before each):

- **1a — `draw_A_Q` matrix rewrite** (`shared.py`). ✅ DONE (2026-07-08). Theory: `eq:param-hu-matrix`,
  `eq:param-combined-weight-u`, `eq:param-deflate`, `eq:param-Q-post`,
  `eq:param-A-vec-model`, `eq:param-A-precision`. New signature takes the sampled
  factor path + per-factor `H_u (T,r)` + `w_u (T,)`; builds `G^u_t` per period;
  draws `Q ~ IW(Ψ + Σ ǔǔ', ν+T_eff)` with `ǔ_t=√w_t H_t^{-1/2}(f_t-A f_{t-1})`,
  then `vec(A) ~ N(m_n,V_n)`, `V_n^{-1}=V0^{-1}+Σ (f_{t-1}f_{t-1}')⊗G^u_t`.
  **Seam:** `H≡1` collapses to the old scalar MNIW — unit-test against current `draw_A_Q`.
- **1c — `shared.realized_deflated_d_u`** ✅ DONE (2026-07-08). Deflate per factor:
  `d^u_t = (H_t^{-1/2}u_t)' Q^{-1}(H_t^{-1/2}u_t)` (eq:param-hu-matrix). Accepts
  `h_u` as `(T,)` (scalar Spec I) or `(T,r)` (per-factor Spec II); equal columns
  reduce to the scalar case. Tested in `test_shared`.
- **1d — multivariate common SV block** (`sample_vol.py`). ⚠️ **CORRECTED**: the
  common block is **NOT** `r` independent scalar sub-sweeps. Theory
  `subsec:vol-all-processes`: because `Q` is *full*, the `r` per-factor
  volatilities are a genuine **multivariate** SV problem. Per-factor log-square
  `y*_{k,t}=log(w^u_t u_{k,t}²+c)=log h^u_{k,t}+log q_{kk}+log ζ̄²_{k,t}`, with a
  **known offset** `ℓ_Q=(log q_kk)` (from current `Q`) and residuals `log ζ̄²_{k,t}`
  that are **mutually correlated** by the correlation matrix of `Q`. Solved by the
  KSC offset mixture **componentwise but retaining cross-component covariance**
  (Harvey–Ruiz–Shephard 1994) → **one r-dimensional FFBS** pass for the whole
  common block (state `log h^u_t` r-vector, diagonal AR(1) Φ/σ²). Decouples to
  scalar sub-sweeps only when `Q` is near-diagonal. *(The idiosyncratic block
  stays M scalar sub-sweeps — R diagonal.)* This is the hard core of Spec II
  sampling and needs its own focused piece (own theory re-read + r-dim FFBS +
  recovery test); do it before 1b.
- **1e — states step (a) per-factor companion covariance** (`sample_states.py`).
  ✅ DONE (2026-07-08): `forward_filter_combined` takes `Qcov (T,r,r)`, embedded as
  the top-left companion block; `ffbs_sample_states` builds `Q_t=√H Q √H/w` when
  `h_u` is `(T,r)`. Seam verified (`test_shared` [4d]: `H=hI` ≡ scalar companion).
  ⚠️ **found in audit.** Theory `eq:states-tv-cov`, `eq:states-aug-Q`: the
  companion innovation cov is `Q̃_t` with top-left block `Q_t = √H^u_t Q √H^u_t / w^u_t`.
  Current `sample_states.py:126` `build_Q_tilde(Q, g_u[t])` builds the **scalar**
  `h_u·Q/w_u` (`g_u = w_u/h_u`, `:225`). Replace with the per-factor sandwich.
  A_tilde, Lambda_tilde, MM weights are unchanged — volatility enters *only* the
  leading-block covariance. **Seam:** equal per-factor H reproduces the scalar build.
- **1b — per-factor state/shapes** (`gibbs.py`). ✅ DONE (2026-07-08): `spec2 = sv
  and not leverage` branch — `logh_u,h_u (T,r)`, `sv_u (r,3)`, storage `(n_keep,T,r)`
  /`(n_keep,r,3)`; vol via `sample_volatility_block_specII` (decoupled), states via
  1e, weights via 1c, Family~A via `draw_A_Q_perfactor` (1a). **End-to-end verified**
  (`test_passo2` [4]: per-factor `h^u_k` recover the common path, φ̂≈0.97 at T=750 —
  the √r data cost means per-factor needs ~r× more T than the old scalar-common).
  No-SV path and leverage path (scalar state) left untouched (`test_passo2` [2],
  `test_passo3` green). Leverage per-factor is Phase 4/7.

**Shared infra (audit):** 1d, 1e and Phase 4 all need matrix functions of `Q`:
`Q^{1/2}`, `Q^{-1/2}`, `diag(Q)` (the offset `ℓ_Q = log q_kk`), and `corr(Q)` (the
measurement cross-covariance in 1d). Build once (`shared.py`), reuse.
- **Test:** per-factor `h_u` recovery on a Spec-II synthetic DGP; **no-SV path
  stays bit-identical** (h≡1 ⟹ `G^u_t=w·Q^{-1}` collapses to old MNIW).
  `test_passo2`, `test_shared` green.

### Phase 2 — Family A proper priors
- **2a — `draw_A_Q_perfactor` priors** ✅ DONE (2026-07-08): args `Psi0, nu0, A0,
  V0_inv` (added in 1a, flat default). Verified against `eq:param-Q-post`
  (`Q_scale=Psi0+scatter`, `df=nu0+T_eff`) and `eq:param-AQ`
  (`A_prec=V0_inv+P00⊗Qinv`, shrinkage) — `test_shared`.
- **2b — `draw_lambda_r_series` NIG** ✅ DONE (2026-07-08): args `a0,b0,m0,M0_inv`
  (`eq:param-LR-post/hyper`); flat limit bit-preserved; proper-prior MC mean ≈
  analytic `cbar/Fbar` and IG mean — `test_shared`.
- **2c — wiring + defaults** ✅ DONE (2026-07-09): threaded the tuning-table
  hyperparameters (`ν0=r+1`, `Ψ0=(2r+2)Q̂_EM`, `A0=Â_EM`, `V0^{-1}=κI`; `a0=2`,
  `b0=(a0-1)r̂_i`, `m0=L̂_i`, `M0^{-1}=κ`) from the warm start through
  `draw_Lambda_R_block` (new `a0,b0,m0,M0_inv` args — `b0` per-series `(M,)`, `m0`
  the `(M,r)` matrix read at block col `j`) and `gibbs.py` (new
  `use_family_a_priors`, `family_a_kappa`). **Decision RESOLVED**: behind a flag,
  **flat is default** — every EM seam stays bit-identical; on is a change of args
  (thesis §From table to code). Priors frozen from the *initial* EM θ before the
  sweep mutates A/Q/Λ/R. **Scope:** A/Q priors wired on the Spec II per-factor
  path (`draw_A_Q_perfactor`); the legacy base MNIW (`draw_A_Q_block`, no-SV &
  leverage) stays flat and **warns** if priors are requested there (Spec-I remnant,
  retired in Phase 8). Λ/R priors apply on all paths. **Tests:** `test_shared` [6]
  — flat-default==explicit-flat bitwise, strong-prior per-series column routing,
  block restriction preserved; end-to-end smoke: spec2 runs priors on/off (A,Q
  finite), warning fires on legacy path. `test_passo2/3` green.
- **2d — Huang–Wand switch** ✅ DONE (2026-07-10). `shared.hw_iw_prior(a, ν*)` →
  `(Ψ0, ν0) = (2ν* diag(1/a), ν*+r-1)` and `shared.draw_hw_aux(Q, ν*, A)` →
  `a_j|Q ~ IG((ν*+r)/2, ν*(Q^{-1})_jj + 1/A_j²)` (`eq:param-Q-hw-prior/-post/-aux`).
  Conditional on `a` the prior is an ordinary IW, so it is a **drop-in swap of the
  IW hyperparameters** inside the *existing* Family A draw + `r` scalar IG draws
  appended. Wired in `gibbs.py` as `q_prior="huang_wand"` (`hw_nu_star=2.0`,
  `hw_A=1e5`), **not default** — `.tex` 20680-94: its role is a one-off robustness
  check. `draws["hw_a"]` stored. ⚠️ tuning corrected from this plan's `A≈10²` to the
  `.tex` (21520-27): `A≈10` mildly informative, `A≈10⁵` the HW near-flat choice.
- **2e — prior hook on the legacy MNIW** ✅ DONE (2026-07-10, *unplanned, needed by
  2d*): `draw_A_Q` gained `(Psi0, nu0, A0, kappa)` — the natural-conjugate MNIW
  with `A|Q ~ MN(A0, Q, (κI)^{-1})`, the Kronecker special case `V0 = Q ⊗ Ω0` the
  `.tex` names at `eq:param-AQ` (20799-802). This lets HW (and the whole tuning
  table) reach the **no-SV "current model"** cell too, and **retires the 2c
  warning**: Family A priors are now wired on *every* path. Flat default
  bit-identical (`test_shared` [8]).
- **Tests** (`test_shared` [8], 9 checks): flat-default==explicit-flat bitwise;
  `κ→∞` shrinks `A→A0`; `(Ψ0,ν0)` enter scale/df (MC mean == IW mean);
  `hw_iw_prior`/`draw_hw_aux` match the equations; and — the decisive one — the
  **construction is validated, not asserted**: iterating *only* the two HW
  conditionals with **no data** yields `sqrt(Q_jj)` with the **half-t_{ν*}(A)**
  median and marginal correlations **Uniform(-1,1)** at `ν*=2`, exactly the two
  claims of `.tex` 20655-62. Plain IW at `ν0=r+1` also checked uniform
  (`eq:param-Q-uniform-nu`).

### Phase 3 — Family B: half-Normal σ_η + μ=0 + M+r  ✅ DONE (2026-07-09)
- **Prior σ_η half-Normal** on the sd `σ_η ~ N(0,B), B≈1` (Gelman 2006 + ASIS)
  wired as a **selectable option** (`sigma_prior="inverse_gamma"|"half_normal"`,
  `half_normal_B`); **IG(2,0.05) stays the code default** — faithful to the `.tex`
  (21148–54, 21807–13): the half-Normal is the master-sampler default *when
  interweaving is on*, and the IG "is retained as the exact conjugate baseline,
  used when that strategy is switched off". ASIS is Phase 6, so IG-default here
  is correct; Phase 6 flips the default on for its NCP update.
- **CP σ_η draw = light RW-Metropolis** on `log σ²` under the half-Normal, prior
  kernel in `v=σ²` is `v^{-1/2} exp(-v/2B)` (incl. the σ_η→v Jacobian); needs the
  current `σ²` as the RW anchor. φ Gaussian, truncated `|φ|<1`. Non-leverage:
  `draw_ar1_params` (valid Gibbs order φ|σ²_cur then σ²|φ_new). Leverage:
  generalised `_draw_sigma2_lev` (same `(1-ρ²)` likelihood, prior kernel swapped)
  — used by both Branch A and Branch B.
- **μ=0 everywhere**: `draw_ar1_params` already pins μ=0 under `fix_mu0`; the
  leverage samplers **structurally** carry no intercept (μ=0 hard-wired, every
  returned `sv` row `(0,φ,σ²)`). Added `fix_mu0` to both leverage blocks
  (raises if `False` — unsupported there) and threaded `sv_fix_mu0` from
  `gibbs.py` to the leverage sampler call (was previously only on the specII call).
- `sv_u` is `(r,3)` under spec2 (Phase 1); "M+1 processes" comments → "M+r".
- **Threading**: `sv_sigma_prior`, `sv_half_normal_B` from `gibbs.py` →
  `sample_volatility_block[_specII]`, `sample_common_vol_mv`, `_sample_idio_vol`,
  and both leverage blocks. Default IG ⇒ **every seam bit-identical** (all tests
  bit-green).
- **Tests:** `test_shared` [7] (kernel: half-Normal recovery of σ²/φ on a
  synthetic AR(1), μ≡0 exactly, tight-B shrinkage below truth, guards);
  `test_passo2` [5] (e2e: half-Normal Gibbs runs, μ_u≡μ_eps≡0 across all draws).
  Smoke: leverage×{contemp,lagged} under half-Normal run, μ≡0, σ² finite. Full
  net green (shared 54, passo2 13, passo3 8, passo4 13, spec2 10).

### Phase 4 — Family C: Option A scalar ρ  (contemporaneous / Branch A)  ✅ DONE (2026-07-09, Branch A)
- **Option A implemented (Branch A / contemporaneous).** Common leverage is now
  `r` **independent scalar** `rho_i` (`draw_rho_scalar`, symmetric with the
  idiosyncratic side), each keyed on the **raw** shock
  `z^u_{i,t}=[√w Q^{-1/2}(√H^u_t)^{-1}u_t]_i` — **no vector draw, no `ρ'ρ<1`**
  (`eq:lev-cond-common`, 21295). Two whitenings kept distinct (`.tex` caution
  ~16368): measurement per-component `√(w/q_kk)u_k` (decoupled, Phase 1); raw shock
  the full symmetric `Q^{-1/2}`.
- **Coupled multivariate path draw** (`_lev_path_mh_mv_common`): the exact
  single-move Metropolis where moving `x_{k,t}` re-evaluates **all** `r`
  transition-into-`t` drifts (`z^u` mixes every factor's vol through `Q^{-1/2}`) —
  **Lorenzo's decision: exact MV, not frozen-z.** Kernel **validated bit-for-bit
  at r=1** vs the scalar `_lev_path_mh` seam.
- **gibbs.py:** `spec2` extended to `sv and (not leverage or timing=='contemporaneous')`
  — contemporaneous leverage now per-factor; `draw_A_Q_perfactor`, states FFBS,
  weights, storage all follow from `spec2`. Minimal change.
- ⚠️ **FINDING (2026-07-09) — single-move MH falls into a state–vol feedback trap.**
  From the flat warm start (`log h=0`) the per-factor χ²₁ measurement is too noisy
  for single-move MH to escape (`h~1 → homoskedastic states → u~homoskedastic →
  h~1`); `phi` **degrades** (→ negative), `σ²` stuck small. KSC-FFBS/specII escapes
  via blocked moves — why the `.tex` prefers Branch~B. The *kernel is correct*
  (proven standalone on per-factor and scalar-common DGPs, recovers `phi≈0.97`);
  the failure is **loop mixing**, not a bug. **Fix (faithful, init-only):
  warm-seed** the sweep from a blocked KSC-FFBS draw (`sample_common_vol_mv`, the
  exact `ρ=0` path) when the path enters flat — target unchanged. Recovery restored
  (`phi≈0.93`, per-factor `corr≈0.6`, comparable to specII 0.8).
- **Deferred to Phase 7/8 (Branch B migration + cleanup):** `common_lev_scalar` is
  **accepted-but-ignored** in Branch A; `draw_rho_vec`/`draw_rho_common`/
  `dominant_dir_z` are **still used by Branch B (lagged)** so their removal waits.
  Boundary first-transition offset, distinct-per-factor-ρ recovery test, A/B parity:
  TODO with Phase 7 (needs a per-factor-leverage DGP — `simulate_dfm_sv` is
  scalar-common with a vector ρ).
- **Tests:** `test_passo3` green (per-factor h^u tracks common path avg corr>0.3;
  dominant `rho_u` negative; acceptance sane; μ≡0). No-leverage (`passo2` 13,
  `shared` 54, `spec2` 10) and Branch~B (`passo4` 13) **unaffected** (lagged stays
  scalar, `spec2=False`).

### Phase 5 — Family D: proper ν prior  ✅ DONE (2026-07-09)
- Two prior constructors in `sample_params.py` (`eq:param-nu-logtarget`,
  `tab:param-prior-tuning`): `nu_log_prior_exponential(mean=20)` → `-ν/mean`
  (decreasing, log-concave — preserves the target's log-concavity) and
  `nu_log_prior_uniform(2, 50)` → flat on `(lo,hi)`, `-inf` outside (tightens the
  griddy bracket, itself already a bounded-uniform). Both return a `ν→log p(ν)`
  callable for the **existing** `draw_nu_griddy` `log_prior` hook + `nu_log_target`
  (which adds `log p(ν)` additively — already wired).
- `gibbs.py`: new `nu_log_prior=None` param (default **flat** — preserves the seam),
  threaded to both the `nu_u` and `nu_eps` griddy draws.
- **Tests** (`test_shared` [3]): prior enters additively; zero-prior == flat
  bit-for-bit; the exponential lowers the posterior mean of ν on *weak* data;
  uniform(2,50) truncates ν<50 on a wide bracket. E2e smoke: wired through gibbs,
  runs; on strong data (ν pinned near the 2.5–4 floor) the weak prior is correctly
  negligible (`.tex`: "a heavy-tailed choice here matters little, ν informed by the
  whole weight path"). `passo1` green (58 shared, 9 passo1).

### Phase 6 — ASIS wrapper on Family B  ✅ DONE (2026-07-09; Branch B deferred to Phase 7)
- New `sample_asis.py` — `asis_scale_interweave(x, y*, has_obs, sigma2_cp, rho, z, …)`:
  ASIS steps (2)–(4) for one process (`sec:asis`, `subsec:asis-move`). Given the
  CP-drawn σ_η² and path: (2) `x̃=x/σ_η`; (3) redraw **signed** σ_η as the
  weighted **measurement regression** of `(y*_t−m_{s_t})` on `x̃_t` (variances
  `v²_{s_t}`, **Gaussian prior N(0,B)** — conjugate here; the same half-Normal
  Family~B uses in CP), with KSC indicators drawn given `x`; and φ from the NCP
  state (drift `ρz`, variance `(1−ρ²)`, via `_draw_phi_lev` with σ²=1); (4)
  `x=σ_η x̃`. μ=0 ⟹ only the scale is interwoven (no level). Signed σ_η enables
  the sign-flip that unsticks the chain.
- **Under leverage (Branch A):** z is **frozen** during the NCP redraw
  (`subsec:asis-leverage`) — the block's two-pass structure (CP+ASIS with frozen
  `z_u`, then recompute `z_u` at the rescaled path, then Family C ρ); σ_η still
  migrates entirely into the measurement (the drift `ρz` carries no σ_η in NCP).
  ρ is not interwoven but benefits through its posterior correlation with σ_η.
- **Wiring:** `use_asis` in `gibbs.py` (forces `sigma_prior='half_normal'` — CP/NCP
  must share the σ_η prior) → `sample_volatility_block_specII` (common
  `sample_common_vol_mv` + idio `_sample_idio_vol`) and the Branch A leverage block
  (common two-pass + idio in-place). Fidelity note: on Branch A (exact-likelihood
  path MH) ASIS uses the **KSC** measurement for the σ_η regression — what the
  `.tex` sanctions ("ASIS sits on top of whichever branch draws the path"); the
  KSC mixture is the standard near-exact log-χ² representation.
- **Deferred:** Branch B (lagged, still scalar-common Spec I) — `use_asis`+`lagged`
  **warns** and proceeds without interweaving; ASIS folds into Branch B when it is
  migrated to per-factor (Phase 7).
- **Tests** (`test_asis.py`, 14 checks): single-process mini-Gibbs — **posterior
  invariant** (CP vs ASIS means agree) and **ESS(φ) ↑ 2.1×** on a strong-signal
  process (where CP crawls); signed-σ_η rescale identity; e2e no-leverage and e2e
  Branch-A-leverage — runs, μ≡0, posterior ~ non-ASIS within MC error. Full net
  green (shared 58, spec2 10, passo2 13, passo3 8, passo4 13, use_asis default off
  ⇒ every existing path bit-identical).

### Phase 7 — Branch B (Omori lagged) under Spec II  ✅ DONE (2026-07-09)
- **Per-factor migration (Option A).** The common block is now `r` **independent
  scalar** Omori/FFBS channels (`_branch_b_one_process` with `K=1` each, reused
  verbatim), structurally like the idiosyncratic series — replacing the old
  scalar-common **vector-ρ** block. Scalar `rho_i` via `draw_rho_scalar`.
- **Two boundary details** (`subsec:lev-branches-allproc` (i)-(ii), the delicate
  part): **(i)** the augmenting sign is `d^u_k = sign(z^u_k)` of the **full-whitening**
  raw shock `z^u = √w Q^{-1/2}(√H)^{-1}u` (computed once from the current path,
  frozen), while the **magnitude** uses the per-component `e_k=√(w/q_kk)u_k` so
  `ξ_k=y*_k−log h_k` stays linear in `log h_k` (the hybrid the `.tex` specifies;
  exact at diagonal Q, decoupled default otherwise). **(ii)** `z^u_0` undefined
  ⇒ first leverage transition into `log h_2` — honoured by `has_u[0]=False`.
- **ASIS on Branch B** (Phase 6 completion): each channel's Family~B draw wrapped
  by `asis_scale_interweave`, using the **lagged** drift `z=g_{t-1}` and the
  distinct leverage mask `has_lev=has_tr` (new `has_lev` arg — measurement mask
  `has_obs` ≠ transition mask under lagged timing); `g` recomputed at the rescaled
  path for Family~C (per-component, no two-pass needed).
- **gibbs.py:** `spec2 = bool(sv)` — **all** SV paths now per-factor (no-lev,
  Branch A, Branch B); the lagged sampler receives per-factor state and
  `draw_A_Q_perfactor`. `use_asis` wired to the lagged sampler.
- **Cleanup (Phase 8 brought forward):** removed the now-dead vector machinery
  `draw_rho_vec` / `draw_rho_common` / `dominant_dir_z` / `_rho_logpost_vec` and
  the `common_lev_scalar` param from both leverage blocks (kept as a deprecated
  ignored arg on `fit_dfm_mcmc` for API/meta stability).
- **Recovery** (Branch B mixes far better than Branch A — direct FFBS, no
  single-move MH trap, no warm-seed): at `T=900` per-factor `ρ` recovered
  `[-0.51,-0.18,-0.20]` vs true `[-0.6,-0.3,-0.2]`, `corr≈0.9`; `φ≈0.97`.
  `test_passo4` updated to `T=500` (per-factor √r cost, like `passo3`): dominant
  ρ sign, avg h-corr>0.4, FFBS accept=1, μ≡0 — 13/13. Full net green
  (`passo3` 8, `test_asis` incl. leverage, no-SV paths bit-identical, `use_asis`
  default off).
- **Still TODO (Phase 8):** a per-factor-leverage DGP (`simulate_dfm_sv` is
  scalar-common) for a distinct-per-factor-ρ recovery test + explicit A/B parity.

### Phase 8 — Variants/switch wiring + dead-code cleanup  ✅ DONE (2026-07-10)
- **Two new flags, one per restriction** (`subsec:variants-restrictions`):
  `sv_idio=False` = **D2-a** (step (b) omits the idiosyncratic half, `h^ε≡1`
  frozen, Family B drawn for the `r` common processes only, and — with leverage —
  no `ρ_ε`: with no `h^ε` there is no log-vol innovation for it to attach to);
  `tails="gaussian"` = **D1-a** (`w≡1`, step (c) *and* Family D omitted).
  Threaded to all three vol blocks (`specII`, Branch A, Branch B).
- **`VARIANT_FLAGS` + `variant_kwargs(cell, leverage=, timing=)`** in `gibbs.py`:
  the table `tab:gibbs-variants` **as code**. `timing` is passed as the *second-level*
  choice (an internal bifurcation, not a restriction — `subsec:variants-leverage-levels`).
  **D1-c raises `NotImplementedError`** (it *replaces* step (c)+Family D by outlier
  indicators; no outlier block exists in this engine — honest, not silently mapped).
  `current model + leverage` raises (nothing to correlate with).
- **Draw-schema follows the restriction:** omitted blocks contribute no keys
  (`nu_*` absent under D1-a; `sv_eps`/`h_eps` absent under D2-a; `rho_eps` too;
  `sv_*`/`rho_*` absent with `sv=False`). `meta` records the cell.
- **Cleanup:** removed `common_lev_scalar` (from `fit_dfm_mcmc` and `meta`), the
  **dead `elif sv` scalar branch** in step (d) (unreachable since `spec2=bool(sv)`),
  the scalar `logh_u/h_u/sv_u` init, the dead `sample_volatility_block` import, and
  the `spec2` variable itself (now `sv`). `M+1 processes` comments → `M+r`.
  ⚠️ **Terminology fixed while doing this (2026-07-10).** Several docstrings called
  the scalar-common block "Specification I". That is **wrong**: per
  `subsec:vol-placement` (15592-15654) *both* specifications give each factor its
  own `h^u_{i,t}` in a diagonal `H^u_t`; they differ only in **where** it enters —
  inside `Q^{1/2}H^u_t Q^{1/2}` (`eq:vol-inside`) vs outside `√H^u_t Q √H^u_t`
  (`eq:vol-outside`, adopted, so that `Var(u_{i,t})=h^u_{i,t}q_{ii}` reads factor
  `i`'s *own* volatility rather than a blend). The one-scalar block is the
  **restriction `H^u_t = h^u_t I`**, at which the two placements *coincide* and the
  fork is vacuous — it selects no specification. Corrected in `sample_vol.py`,
  `simulate_sv.py`, `sample_leverage.py`, `shared.py`, `test_perfactor_leverage.py`.
  Likewise, **`h` is never 1 in either specification**: the only frozen-at-1 cases
  are the ones the `.tex` itself freezes (`h^ε≡1` under D2-a, `h≡1` in the no-SV row
  where `σ_η²=0`). And **D1-a / D2-a are restrictions of the *master*, hence Spec II**:
  they switch blocks off, never the sandwich (asserted in `gibbs.VARIANT_FLAGS`).
  ⚠️ **Deviation from this plan, deliberate:** `sample_vol.sample_volatility_block`
  (the scalar-common `H=hI` block) is **kept, not deleted** — it is on *no*
  sampler path, but it is the **bit-exact `r=1` seam** validating
  `sample_common_vol_mv` (`test_shared` [4c]). Deleting it would remove a
  regression net for zero gain (plan §4: "do not break the seams"). Docstring now
  carries a `.. warning::` saying it is a reference implementation, not a variant.
- **`simulate_dfm_sv(sv_u_perfactor=(r,3))`** — the **Spec II + Option A DGP**
  (P0 closed): `r` independent `h^u_k` in the outside sandwich
  `u_t=√H^u_t Q^{1/2} z_t/√w`, each with its own scalar `ρ_k` coupling `η_k` to the
  raw `z_k`. The scalar-common `sv_u` path kept **bit-identical** (idio-leverage code
  factored into a helper, same RNG stream). Verified directly: the *sampler's own*
  whitening `z=√w Q^{-1/2}H^{-1/2}u` gives `corr(η_k,z_k)=[-0.694,-0.118,0.422]`
  against true `ρ=[-0.70,-0.15,0.45]`.
- **Tests.** `test_variants.py` (**37 checks**) — the grid table, D2-a at the block
  level *and* e2e, D1-a proven by **bit-identity of two runs differing only in the
  `ν` warm start** (if step (c) ran, every downstream draw would move) with the D1-b
  control showing it *does* move, the current-model row, ρ=0, Family A priors on
  every path (no warning), and Huang–Wand e2e incl. the thesis' own robustness
  check (posterior `corr(Q)` unmoved vs the plain IW). `test_perfactor_leverage.py`
  — channel separation (each `h^u_k` tracks *its own* path), distinct `ρ_k`
  ordering + signs, A/B parity at diagonal `Q` (where the P5 whitenings coincide).
- **📊 P2 quantified (2026-07-10), the headline empirical result of this phase.**
  On the new per-factor DGP (`T=600, r=3`, diagonal `Q`, `ρ=[-0.70,-0.15,+0.45]`,
  log-vol unconditional sd `[1.03, 0.46, 0.70]`):
  - **the `ρ_k` are all recovered** — ordering, dominant negative sign *and* the
    positive channel — on **both** branches, and A and B agree. Family C is robust.
  - **the `h^u_k` paths are not equally recovered:** the two strong channels give
    `corr ≈ 0.89 / 0.86` (B) and `0.87 / 0.75` (A) with `φ̂ ≈ 0.95-0.96`, while the
    **weak channel (sd 0.46) collapses**: `corr 0.26`, `φ̂ 0.62` (B); on A it is not
    even separated from the others (cross-corr `0.50` > own `0.33`).
  - So per-factor `h^u` identification needs **volatility *and* length**: `ρ` survives
    a weak channel, the *path* does not. `test_perfactor_leverage` now **asserts the
    boundary** (strong channels `corr>0.6`, weak channel `corr<0.6`) so a future
    mitigation registers as a deliberate re-read, not a silent pass. This sharpens
    P2: the short-`T` risk is concentrated in **low-volatility factors**.
- **Full regression:** shared 67, variants 37, passo1 9, passo2 13, passo3 8,
  passo4 13, asis 14, spec2 8, perfactor-leverage — all green; no-SV and Spec-II
  seams bit-identical.

---

## 3. Decisions already resolved (no re-litigation)
- **Family A priors default** = behind `use_family_a_priors` flag, **flat is default**
  (preserves the EM seam bit-for-bit; on = a change of args). (Resolved 2026-07-09.)
- **σ_η prior** = half-Normal on signed σ_η (default), IG(2,0.05) baseline. (Gelman+ASIS.)
- **Q prior** = IW with **ν_0=r+1** (uniform marginal correlations),
  Ψ_0=(2r+2)Q̂_EM; Huang–Wand as an *optional* robustness switch, not default.
- **ν prior** = proper on (2,∞); griddy bracket already acts as bounded-uniform.
- **μ=0** identification for every log-vol process.
- **Option A** for common leverage: `r` scalar channels, no vector ρ.
- **Timing** (Branch A vs B): still open, to settle empirically — both implemented.

## 4. Test / validation strategy
- Keep every `draw_*` kernel's **flat-prior limit** as an exact seam vs EM
  (`test_shared` already asserts this to machine precision) — do not break it;
  add the proper-prior behaviour behind new args whose defaults reproduce the flat case.
- Recovery on calibrated Spec-II synthetic DGPs (extend `simulate_sv.py`).
- ASIS: judged by ESS/R̂ improvement at unchanged posterior, not by a point seam.

## 5. Interface to the second stage (for later, not this refactor)
`fit_dfm_mcmc` returns `draws` (A, Q, Lambda, R, nu_*, sv_*, rho_*, optional F/h).
The density/GaR module consumes these; keep the dict schema stable so that work
is unblocked independently of the sampler internals.

---

## Appendix — 1d design: the multivariate common-volatility block

Theory: `subsec:vol-all-processes` (multivariate common block), `subsec:vol-ksc`
(scalar KSC template), `subsec:vol-placement`/`eq:vol-outside` (Spec II sandwich),
`eq:vol-logsquare`, `eq:vol-state-ar1`, `eq:ffbs-backward`. **This is a design to
validate, not yet code.**

### The model (exact)
From `u_t = √H^u_t Q^{1/2} z^u_t / √w^u_t` (Spec II), the k-th component gives
`√w^u_t u_{k,t} = √h^u_{k,t} ζ_{k,t}`, `ζ_t = Q^{1/2}z^u_t ~ N(0,Q)`. Standardise
`ζ̄_{k,t} = ζ_{k,t}/√q_kk` (unit marginal), so the **per-factor log-square** is
```
y*_{k,t} = log(w^u_t u_{k,t}² + c) = log h^u_{k,t} + log q_kk + log ζ̄²_{k,t},   k=1..r
```
- **state** `log h^u_{k,t}`, r independent AR(1)s: `Φ=diag(φ_k)`, `Var(η)=diag(σ²_k)`,
  μ=0, stationary init `log h_{k,1} ~ N(0, σ²_k/(1-φ_k²))`.
- **known offset** `ℓ_Q = (log q_kk)` from the current `Q` (diag). *Must* be
  subtracted, else the filter absorbs `log q_kk` into the state and biases the
  level against the μ=0 identification.
- **measurement noise** `ξ̄_{k,t} = log ζ̄²_{k,t}`: marginally `log χ²_1`
  (mean −1.2704, var π²/2), but the r-vector is **cross-correlated** because the
  `ζ̄_k` share the correlation matrix `corr(Q)`. `Cov(ξ̄_j,ξ̄_k)=g(corr(Q)_{jk})`,
  a fixed function `g` with `g(0)=0`, `g(±1)=π²/2` (log-squares of a bivariate
  normal; precompute on a ρ-grid or the closed series).

State transition diagonal ⇒ decouples; **the measurement couples the factors**
(full noise cov) ⇒ one genuine **r-dim FFBS**, not r scalar passes. Decouples to
r scalar sub-sweeps only when `corr(Q)≈I`.

### The measurement cross-covariance — DECIDED: follow the thesis
**Target = the thesis method** (`subsec:vol-all-processes`, literal phrasing):
KSC offset mixture **componentwise** for the per-factor mean/variance, **retaining
the cross-component covariance** HRS-1994-style on the off-diagonal. PSD is secured
by a clean **correlation-scaled** construction (no nearest-PSD repair needed):
```
Σ_ξ,t = diag(v_{s_k,t}) · R_ξ · diag(v_{s_k,t}),    R_ξ[j,k] = g(corr(Q)_{jk}) / (π²/2)
```
- diagonal = mixture variance `v²_{s_k,t}` (the KSC componentwise part);
- `R_ξ` = the **correlation matrix of the log-square vector** `ξ̄_t` (1 on the
  diagonal, `g(ρ_jk)/(π²/2)` off it). It is a genuine correlation matrix ⇒ PSD by
  construction, so `Σ_ξ,t = D^{1/2} R_ξ D^{1/2} ≻ 0` always.
- `g(ρ) = Cov(log X², log Y²)` for a standard bivariate normal of correlation ρ
  (`g(0)=0`, `g(±1)=π²/2`): precompute `R_ξ`'s entries once by numerical
  integration / a short MC on the `corr(Q)` off-diagonals each sweep (r small).

*Note on the exact-covariance variant:* gluing the exact `g(ρ_jk)` onto a
mixture diagonal (rather than correlation-scaling) is not guaranteed PSD and would
need a repair — we do **not** take that route; correlation-scaling retains the
cross-*correlation* of the log-squares while keeping the mixture's per-component
scale, and is PSD-safe.

**Implementation order (NOT a different method — same thesis target):** build in
two commits to de-risk. *Commit 1:* ✅ DONE (2026-07-08) — `sample_vol.py:sample_common_vol_mv`
with `R_ξ = I` (r independent scalar KSC sub-sweeps via `sample_log_vol_process`);
per-factor reading `e_{k,t}=√(w/q_kk)u_{k,t}`. **r=1 seam verified bit-for-bit**
against the scalar common block (`test_shared` [4c]); coupled-`R_ξ` guard raises.
*Commit 2:* ⚠️ **FINDING (2026-07-08) — coupling does not pay, and the literal
construction is unstable.** Implemented the coupled r-dim FFBS
(`sample_common_vol_mv` with `R_ξ`, `logsq_corr_matrix`, `_mv_ar1_ffbs`; g(ρ)
table verified). But on a strong-`corr(Q)=0.92` DGP:
- **Option 3 (correlation-scaled, the literal thesis phrasing)** `Σ=diag(v_s)R_ξ
  diag(v_s)` **distorts** the less-persistent factor — φ collapses 0.90→0.42,
  σ² explodes 0.12→1.27 — because it mixes the *unconditional* `R_ξ` with the
  mixture's *conditional* `v_s` (inconsistent). Verified NOT an FFBS bug: with
  `R_ξ=I` the r-dim FFBS reproduces the decoupled recovery exactly.
- **Option 2 (QML, constant exact `Σ=[[π²/2,g],[g,π²/2]]`, no mixture)** is
  *stable* (φ 0.856, σ² 0.23) but does **not** beat decoupled on point recovery.
- **Key insight:** positively-correlated measurement noise is *redundant*
  information ⇒ coupling can only *lower* point-estimate accuracy vs the
  (overconfident) decoupled sampler; its benefit is **calibration, not accuracy**.
So the premise "coupling improves recovery" was **false**. **Default = decoupled
(Commit 1)** — tested, stable, theory-sanctioned near-diagonal. The coupled path
is left `EXPERIMENTAL` behind the `R_ξ` arg. **Thesis implication:** the .tex
presents the multivariate cross-cov version as the method; consider a caveat that
(i) its benefit is calibration not point accuracy, (ii) the correlation-scaled
realisation is unstable and the decoupled/near-diagonal form is the practical
default. **Decision pending with Lorenzo.**
Near-diagonal agreement ✅ (coupled≡decoupled, validates FFBS + g(ρ) table).
Keep `R_ξ = None` as the default. **r>1 recovery** ✅ DONE (2026-07-08): `test_spec2_recovery.py` — Spec-II
DGP, diagonal Q, r=2 distinct `(φ_k,σ²_k)`; mini-Gibbs recovers per-factor paths
(corr 0.90/0.71), persistence (φ̂ 0.974/0.907, distinct) and σ² (0.049/0.097),
μ≡0. 8/8 green.

### FFBS structure (one r-dim pass; `R_ξ=I` decouples to r scalar passes)
Per Gibbs iteration, per period t:
1. **indicators** `s_{k,t}` — r independent multinomial draws (`eq:vol-indicator-cond`),
   each from the 7-comp KSC given the current `log h_{k,t}` and `y*_{k,t}−log q_kk`.
2. **path** — one r-dim FFBS: observation `ỹ_t = y*_t − ℓ_Q − m_{s_·,t}` (mean-centred),
   measurement cov `Σ_ξ,t = diag(v_{s_k}) R_ξ diag(v_{s_k})`, state transition `Φ`
   (diag), state innov `diag(σ²)`; Kalman-filter forward, backward-sample
   `log h^u_{1:T}`. With `R_ξ=I` (Commit 1) the filter separates into r scalar passes.

### Shared helpers needed (build in `shared.py`)
`Q^{1/2}` and `Q^{-1/2}` (symmetric sqrt via eigh), `diag(Q)` → `ℓ_Q`,
`corr(Q)` → off-diagonal via `g(·)`; a `g(ρ)` table/closed-form for the log-square
cross-covariance.

### Interface
`sample_common_vol_mv(u_head, Q, sv_u, logh_u_cur, rng, *, offset=1e-6, coupling=...)`
→ `{logh_u (T,r), h_u (T,r)}`, where `u_head = f_t − A f_{t-1}` (r-vector per t).
Plugs into `gibbs.py` (1b) as the common-block replacement; `sv_u` is `(r,3)`.

### Tests
- **r=1 seam (Commit 1):** with `R_ξ=I` reproduces the current scalar
  `sample_volatility_block` path bit-for-bit (same rng) — regression vs `test_passo2`.
- **Recovery:** simulate a multivariate-SV DGP (`simulate_sv.py` extended) with a
  chosen `corr(Q)`; recover per-factor `(φ_k,σ²_k)` and the `h_k` paths.
- **Coupling matters (Commit 2):** at large `|corr(Q)|`, the full `R_ξ` recovers
  `h` better than `R_ξ=I` (quantifies the bias the cross-cov removes); at
  `corr(Q)≈I` the two agree — validates the decoupling claim and the `g(ρ)` table.

---

## Appendix — Known problems & open decisions (running log)

> **⚠️ AVVISO (2026-07-10, `docs/audit_P1-P5.md`) — ogni `ρ̂` citato in questo file è
> inaffidabile.** Le stime di leverage riportate sotto (Phase 4, Phase 7, P0, P2) sono
> misurate con catene da 600–900 iterazioni. `ESS(ρ) ≈ 3–23 su 2000 draw`: l'errore
> Monte Carlo domina, e con quell'ESS non si distingue una catena non convergente da un
> bias sistematico. `ρ` è aggiornato con **una sola** mossa RW (`prop_sd=0.06`, da
> `ρ=0`) per sweep, e allargare la proposta non aiuta (accettazione crolla, ESS no).
> Fix raccomandato: **griddy-Gibbs su `ρ`** (il log-posterior è 1-D su `(−1,1)`, liscio,
> già scritto in `_rho_logpost_scalar`; il pattern esiste in `draw_nu_griddy`).
> Da fare **prima** di puntare il sampler sul pannello reale — `ρ` è il parametro che
> dà la skew alla densità del PIL. Vedi audit §P6.
>
> L'audit ha inoltre **chiuso P1, P3, P4, P5** (2026-07-10): il ramo coupled è
> irraggiungibile e assente sotto B; P3 è A-specifico e B è immune per costruzione;
> `corr(Q)` reale ≤ 0.099 rende P4 (+0.4%) e P5 (−0.1%) trascurabili. Tutti congelati
> da test (`test_variants` [8], `test_passo4` [6], `test_diagnostics`).
> **Correzione a P2**: `T` cura i parametri, **non** il tetto informativo (`corr(ĥ,h)`
> del canale debole satura a ~0.63 per qualunque `T`), e Branch A *degrada* con `T`.

A consolidated list of the substantive issues surfaced during the refactor, with
pointers to where each is detailed and its current disposition. Newest first.

**P6 — Huang–Wand: implemented, and the robustness check already passes.** *(Phase 2d,
2026-07-10 — informational, no decision pending.)* The `.tex` (20680-94) prescribes a
*use*, not just a switch: "re-estimate under `eq:param-Q-hw-prior` and confirm the
posterior correlations among the factor innovations do not move; we would adopt it as
the working prior only in the unlikely event that they do." On the synthetic panel of
`test_variants` [7] the posterior `corr(Q)` moves by `< 0.05` between the plain IW
(`ν0=r+1`) and Huang–Wand (`ν*=2, A=1e5`) — as predicted (`T` in the hundreds, `r`
small, `μ=0` keeping `diag(Q)` away from zero). **Still to do on the real panel:** run
the same comparison once on the actual FRED-MD panel before the thesis asserts the IW
critique is answered — the claim is empirical, and the check is now one flag
(`q_prior="huang_wand"`). Pairs naturally with the `corr(Q)` measurement P4 already
needs. *Not* a default; keep `inverse_wishart`.

**P3 — Branch A single-move MH: state–vol feedback trap (per-factor).** *(Phase 4,
2026-07-09.)* From the flat warm start (`log h=0`) the per-factor χ²₁ measurement
is too noisy for the single-move Metropolis to escape: `h~1 → homoskedastic states
→ u~homoskedastic → h~1`, so `φ` **degrades to negative** and `σ²` sticks small.
The KSC-FFBS (specII / no-leverage) escapes via blocked moves; the single-move MH
does not — which is exactly why the `.tex` prefers **Branch B (lagged, Omori-FFBS)**
for mixing. The **kernel is proven correct** (bit-exact r=1 seam; standalone
recovers `φ≈0.97` on per-factor and scalar-common DGPs) — the failure is *loop
mixing*, not a bug. **Fix adopted (faithful, init-only):** warm-seed the sweep's
path from a blocked KSC-FFBS draw (`sample_common_vol_mv`, the exact `ρ=0` path)
when it enters flat; target unchanged, recovery restored (`φ≈0.93`, corr `≈0.6`).
*Open watch:* the ongoing single-move mixing may still be slow on long real runs —
**check ESS / split-R̂ for the leverage `φ,σ²` once ASIS (Phase 6) is in**, and lean
on Branch B where feasible. Detail: Phase 4 section.

**P5 — Branch A vs Branch B use different leverage-magnitude whitenings.** *(Phase 7,
inherent to the `.tex` construction — benign.)* Both branches are Option A (scalar
`ρ_i`, full-whitening **sign** `d^u_k=sign(z^u_k)`), but the **drift magnitude**
differs: Branch A (single-move MH) uses the exact full-Q `|z^u_k|`; Branch B (Omori
FFBS) uses the per-component `|e_k|=exp(ξ_k/2)` with `ξ_k=y*_k−log h_k` (the
linearisation that makes the FFBS possible, `eq:lev-omori-cond`). The two coincide
when `Q` is diagonal and differ mildly for full `Q`. This is what the `.tex`
specifies (Branch B's magnitude is per-component by construction), not a bug — but
it means A/B agree on the shared parameters only up to the whitening convention;
factor in when comparing branches. Detail: Phase 7.

**P2 — √r data cost of per-factor SV — ⚠️ HIGH PRIORITY for the forecast application.**
*(Phase 1, structural; flagged important by Lorenzo 2026-07-09.)* Each per-factor
volatility is read through **one** log-square per period (vs `r` simultaneous
readings the old scalar-common enjoyed) — the √r-averaging is gone (`.tex`
`subsec:param-familyB`, delicate case (a)). Consequence: per-factor recovery needs
**~r× more T**. Empirically (scalar-common DGP, `r=3`): `T≈220–240` **collapses**
(φ→wrong, ρ sign flips), `T≈500` recovers the dominant leverage, `T≈750–900` is
comfortable. A *property*, not a bug.
- **Sharpened 2026-07-10 (Phase 8, measured on the new per-factor DGP):** the cost is
  **not uniform across factors** — it bites where the volatility is *small*. At
  `T=600, r=3`: channels with log-vol unconditional sd `1.03` / `0.70` recover
  (`corr≈0.89/0.86`, `φ̂≈0.95`); a channel with sd `0.46` **does not** (`corr 0.26`,
  `φ̂ 0.62`). Crucially, `ρ_k` *is* still recovered on the weak channel (sign and
  ordering) — it is the **path**, not the leverage, that dies first. Test:
  `test_perfactor_leverage.py`.
- **Why it matters here specifically:** the target use is **real-time nowcasting**,
  where the effective `T` is *short* — early vintages, the quarterly GDP target
  seen only at quarter-ends, and any rolling/expanding window that starts small.
  So the regime where the per-factor common SV + leverage are **weakly identified**
  is exactly the operational one. Symptoms to expect at short `T`: noisy/®sign-
  unstable `ρ_i`, over-smoothed per-factor `h^u_k`, and — because these set the
  predictive skew/scale — a **degraded tail/density forecast** (ties to
  [[project_density_nowcast_feasibility]]).
- **Mitigations to weigh (Phase 8 / forecast stage):** (a) fall back to the
  **scalar-common** SV (Spec I, the √r-averaged, better-identified block) for the
  common volatility when `T` is short, keeping per-factor only where the data
  support it; (b) tighten the `σ_η` / `ρ` priors at short `T` (Family B half-Normal
  scale, Fisher-`z` shrinkage on `ρ`); (c) **pool/partial-pool** the per-factor
  `(φ_k,σ²_k,ρ_k)` across factors; (d) accept per-factor only past a measured
  `T`-threshold. **Decision deferred**; must be settled before the per-factor
  sampler is used for the real-time forecast. Detail: Phase 1 (1b), Phase 4/7.

**P1 — Multivariate coupled common-vol (1d Commit 2): the literal thesis form is
unstable; the stable form gives no accuracy gain.** *(Phase 1, 2026-07-08 —
DECISION PENDING with Lorenzo.)* The thesis presents the **correlation-scaled**
cross-covariance `Σ_ξ,t = diag(v_{s_k}) R_ξ diag(v_{s_k})` ("mixture componentwise +
retained cross-cov") as *the* multivariate method. In practice on a strong
`corr(Q)=0.92` DGP it **distorts the less-persistent factor** (φ 0.90→0.42, σ²
0.12→1.27) because it mixes the *unconditional* `R_ξ` with the mixture's
*conditional* `v_s` (inconsistent). The consistent **QML** variant (constant exact
`Σ`, no mixture) is *stable* but **does not beat decoupled** on point recovery —
positively-correlated measurement noise is redundant info, so coupling can only
*lower* point accuracy; its benefit is **calibration, not accuracy**. **Default =
decoupled** (`R_ξ=None`); coupled left `EXPERIMENTAL`. **Two open items:**
(i) **thesis wording** — add a caveat that the coupled cross-cov's benefit is
calibration not point accuracy, the correlation-scaled realisation is unstable, and
the decoupled/near-diagonal form is the practical default; (ii) see P4. Detail:
Appendix 1d, "Commit 2 finding".

**P4 — Decoupled default vs GaR calibration.** *(Phase 1, tied to P1 — needs data.)*
The decoupled sampler is **overconfident** (compresses the posterior volatility);
for the density/Growth-at-Risk objective, *calibration is the target*, so coupling's
one benefit is aligned. Whether to switch on coupled-QML depends on **(a) the actual
`corr(Q)` magnitude on the real panel** (decoupled is exact at diagonal `Q`) and
**(b) a PIT / coverage check on the second stage** — not on point accuracy (which was
what was measured). **TODO:** measure `corr(Q)`; run the 2nd-stage calibration check;
decide coupled-QML on/off. Links: [[project_density_nowcast_feasibility]] (the QR
carries the tail, which buffers the vol-forward compression).

**P0 — Test-infra gap: no per-factor-leverage DGP. ✅ RESOLVED (Phase 8, 2026-07-10.)**
`simulate_dfm_sv` generated a **scalar-common** `h^u` with a *vector* `ρ` (the old
Spec-I DGP), so a clean **distinct-per-factor-ρ recovery** test (and Branch-A/B parity
on ρ) was not even *expressible*. Fixed: `sv_u_perfactor=(r,3)` adds the Spec II +
Option A DGP (`r` independent `h^u_k`, `r` scalar `ρ_k`); the Spec-I path is untouched
and bit-identical. `test_perfactor_leverage.py` now gates **channel separation** (each
posterior `h^u_k` tracks *its own* true path better than any other factor's — vacuous
under the old DGP, where every factor read the same truth), **distinct `ρ_k`** (ordering
+ signs, incl. a positive channel), and **A/B parity** run at **diagonal `Q`**, where the
two branches' magnitude whitenings coincide (see P5) so the comparison is clean.
