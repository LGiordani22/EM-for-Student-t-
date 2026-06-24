"""
Independent numerical verification of the math in
  - sec:gibbs-weights        (tail-weight Gibbs block, Gaussian-Gamma conjugacy)
  - subsec:lev-branch-B       (Omori et al. lagged leverage sampler)
  - subsec:lev-branch-A       (JPR contemporaneous Metropolis target)
  - subsec:lev-trunk          (common conditional law eta|z)

WHAT THIS SCRIPT CAN AND CANNOT DO
----------------------------------
CAN (first-principles / internal consistency, no external source needed):
  1. The leverage conditional law  eta|z ~ N(rho*sig*z, sig^2(1-rho^2))
     (scalar and vector/common-factor versions), via simulation + regression.
  2. The Gaussian-Gamma conjugacy of the weights block: posterior is
     Gamma((nu+M)/2, (nu+dcheck)/2), checked against a brute-force grid
     posterior; and the h=1 reduction to the EM posterior.
  3. The log-chi^2_1 moments (mean -1.2704, var pi^2/2) underlying xi_t.
  4. The sign identity z = d*exp(xi/2) and the *linear-in-log h* pivot of
     Branch B (mean affine in log h_t).
  5. The JPR/Branch-A Metropolis target: that the change-of-variables
     factorisation (level term * leverage-drifted transition * out transition)
     reproduces the true joint density of (e_t, log h_t).

CANNOT (require the Omori et al. 2007 paper; NOT attempted here):
  - The numerical values of the 7-component KSC table {q_j,m_j,v_j^2}.
  - The numerical values of the 10-component Omori table
    {q_j,m_j,v_j^2,a_j,b_j} and the literal form of their fitted
    a_j,b_j linearisation. We only check that *some* affine a+b(xi-m)
    can approximate exp(xi/2) locally, which is the structural claim.
"""
import numpy as np
from scipy import stats
from scipy.special import digamma, polygamma

rng = np.random.default_rng(20260620)
def hdr(s): print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)
def ok(b): return "PASS" if b else ">>> FAIL <<<"


# ----------------------------------------------------------------------
hdr("1. Trunk conditional  eta|z ~ N(rho*sig*z, sig^2(1-rho^2))  [scalar]")
# ----------------------------------------------------------------------
rho, sig = -0.6, 0.45
cov = np.array([[1.0, rho * sig], [rho * sig, sig**2]])
N = 4_000_000
zx = rng.multivariate_normal([0, 0], cov, size=N)
z, eta = zx[:, 0], zx[:, 1]

# regression eta = a + b z  -> b should be rho*sig ; residual var sig^2(1-rho^2)
b_hat = np.cov(eta, z, bias=True)[0, 1] / np.var(z)
a_hat = eta.mean() - b_hat * z.mean()
resid_var = np.var(eta - (a_hat + b_hat * z))
b_true, var_true = rho * sig, sig**2 * (1 - rho**2)
print(f"  slope  : emp {b_hat:+.5f}   formula rho*sig = {b_true:+.5f}   {ok(abs(b_hat-b_true)<2e-3)}")
print(f"  intcpt : emp {a_hat:+.5f}   formula 0                         {ok(abs(a_hat)<2e-3)}")
print(f"  res var: emp {resid_var:.5f}   formula sig^2(1-rho^2) = {var_true:.5f}   {ok(abs(resid_var-var_true)<2e-3)}")
# conditional check in a thin z-slice
m = (np.abs(z - 1.0) < 0.02)
print(f"  E[eta|z=1.0]: emp {eta[m].mean():+.4f}   formula {b_true*1.0:+.4f}   "
      f"{ok(abs(eta[m].mean()-b_true)<1e-2)}")
print(f"  rho=0 limit -> N(0, sig^2): residual var with rho=0 would be sig^2 = {sig**2:.5f} (mean 0)")


# ----------------------------------------------------------------------
hdr("2. Trunk conditional  eta^u|z^u  [common factor, vector rho]")
# ----------------------------------------------------------------------
r = 3
rho_v = np.array([-0.4, -0.2, 0.3])      # need rho'rho < 1
sig_u = 0.5
assert rho_v @ rho_v < 1
Sig = np.block([[np.eye(r), (sig_u * rho_v)[:, None]],
                [(sig_u * rho_v)[None, :], np.array([[sig_u**2]])]])
zx = rng.multivariate_normal(np.zeros(r + 1), Sig, size=2_000_000)
Z, ETA = zx[:, :r], zx[:, r]
# multivariate regression ETA ~ Z : coeff vector should be sig_u*rho_v
beta = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z], ETA, rcond=None)[0]
resid_var = np.var(ETA - np.c_[np.ones(len(Z)), Z] @ beta)
print(f"  slope vec emp     {np.array2string(beta[1:], precision=4, sign='+')}")
print(f"  formula sig_u*rho {np.array2string(sig_u*rho_v, precision=4, sign='+')}   "
      f"{ok(np.allclose(beta[1:], sig_u*rho_v, atol=3e-3))}")
print(f"  res var emp {resid_var:.5f}  formula sig_u^2(1-rho'rho) = "
      f"{sig_u**2*(1-rho_v@rho_v):.5f}   {ok(abs(resid_var-sig_u**2*(1-rho_v@rho_v))<3e-3)}")


# ----------------------------------------------------------------------
hdr("3. Weights block: Gaussian-Gamma conjugacy")
#    posterior  w ~ Gamma((nu+M)/2, (nu+dcheck)/2)
# ----------------------------------------------------------------------
nu_e, M = 6.0, 5
h_e = np.array([0.7, 1.3, 0.5, 2.0, 0.9])      # idiosyncratic vols at t
r_i = np.array([1.0, 0.4, 2.5, 0.8, 1.2])      # base scales
eps = np.array([0.5, -1.1, 0.3, 2.2, -0.7])    # residuals y - L f
dcheck = np.sum(eps**2 / (h_e * r_i))          # deflated Mahalanobis (eq:w-d-eps-defl)
shape = (nu_e + M) / 2
rate = (nu_e + dcheck) / 2
print(f"  dcheck (deflated) = {dcheck:.5f}")
print(f"  analytic posterior  Gamma(shape={shape:.3f}, rate={rate:.3f})")

# brute-force: log posterior(w) = log prior + log likelihood, compare to Gamma kernel
wgrid = np.linspace(0.01, 8, 4000)
# prior Gamma(nu/2, nu/2) [shape-rate], log up to const
logprior = (nu_e/2 - 1)*np.log(wgrid) - (nu_e/2)*wgrid
# likelihood N(eps; 0, diag(h_e*r_i)/w):  (w)^{M/2} exp(-w/2 * dcheck)
loglik = (M/2)*np.log(wgrid) - (wgrid/2)*dcheck
logpost = logprior + loglik
# Gamma(shape, rate) log-density up to const
loggamma = (shape - 1)*np.log(wgrid) - rate*wgrid
diff = logpost - loggamma
print(f"  (logpost - logGamma) across grid: std = {diff.std():.2e}  "
      f"(should be ~0 => identical up to constant)   {ok(diff.std() < 1e-9)}")
# also check the mode and mean match
mode_emp = wgrid[np.argmax(logpost)]
mode_th = (shape - 1)/rate
print(f"  posterior mode: grid {mode_emp:.4f}  Gamma (a-1)/b {mode_th:.4f}   "
      f"{ok(abs(mode_emp-mode_th)<5e-3)}")

# h=1 reduction to EM posterior: dcheck collapses to eps'R^-1 eps
dEM = np.sum(eps**2 / r_i)
dcheck_h1 = np.sum(eps**2 / (np.ones(M) * r_i))
print(f"  h=1 reduction: dcheck(h=1) {dcheck_h1:.5f}  ==  EM d^eps {dEM:.5f}   "
      f"{ok(abs(dcheck_h1-dEM)<1e-12)}")

# factor side: dcheck^u = d^u / h^u
Q = np.array([[2.0, 0.3], [0.3, 1.0]]); u = np.array([0.8, -0.5]); h_u = 1.7
dEM_u = u @ np.linalg.solve(Q, u)
dcheck_u = (1/h_u) * dEM_u
print(f"  factor side: dcheck^u {dcheck_u:.5f}  ==  d^u/h^u {dEM_u/h_u:.5f}   "
      f"{ok(abs(dcheck_u-dEM_u/h_u)<1e-12)}")


# ----------------------------------------------------------------------
hdr("4. log-chi^2_1 moments underlying xi_t = log z^2  (feeds KSC & Branch B)")
# ----------------------------------------------------------------------
zz = rng.standard_normal(8_000_000)
xi = np.log(zz**2)
mean_th = digamma(0.5) + np.log(2.0)     # = -gamma - log2 = -1.27036...
var_th = polygamma(1, 0.5)               # = pi^2/2 = 4.9348...
print(f"  E[xi]  emp {xi.mean():+.4f}   theory psi(1/2)+log2 = {mean_th:+.4f}   "
      f"{ok(abs(xi.mean()-mean_th)<3e-3)}")
print(f"  Var[xi] emp {xi.var():.4f}   theory pi^2/2 = {var_th:.4f}   "
      f"{ok(abs(xi.var()-var_th)<6e-3)}")
print(f"  doc claims mean -1.2704 and var pi^2/2: matches theory above.")


# ----------------------------------------------------------------------
hdr("5. Branch B: sign identity and the linear-in-log h pivot")
# ----------------------------------------------------------------------
# sign identity z = d * exp(xi/2)
zt = rng.standard_normal(100000)
xit = np.log(zt**2); dt = np.sign(zt)
zrec = dt * np.exp(xit/2)
print(f"  z == d*exp(xi/2):  max abs err {np.max(np.abs(zrec - zt)):.2e}   "
      f"{ok(np.max(np.abs(zrec-zt))<1e-9)}")

# linear-in-log h: mean(eta_{t+1}) = d*rho*sig*e^{m/2}(a + b(xi-m)), xi = ystar - logh
# => as a function of logh it must be affine: slope = -d*rho*sig*e^{m/2}*b
m_j, a_j, b_j = -1.0, 1.0, 0.5          # arbitrary stand-ins (NOT Omori values)
ystar, dts = 0.37, 1.0
f = lambda logh: dts*rho*sig*np.exp(m_j/2)*(a_j + b_j*((ystar - logh) - m_j))
xs = np.array([-2.0, 0.0, 1.5, 3.0])
ys = f(xs)
slope = np.diff(ys)/np.diff(xs)
slope_th = -dts*rho*sig*np.exp(m_j/2)*b_j
print(f"  mean(eta) vs log h: numerical slopes {np.array2string(slope, precision=5)}")
print(f"  predicted constant slope -d*rho*sig*e^(m/2)*b = {slope_th:+.5f}   "
      f"{ok(np.allclose(slope, slope_th, atol=1e-9))}  (affine => linear-Gaussian state eq)")

# structural caveat: exp(xi/2) is NOT globally affine; show local-fit error to
# motivate why Omori needs 10 components + per-component (a_j,b_j).
for half in (0.5, 1.5, 3.0):
    grid = np.linspace(m_j-half, m_j+half, 200)
    true = np.exp(grid/2)
    # best local affine fit a+b(xi-m)
    Aa = np.c_[np.ones_like(grid), grid-m_j]
    coef = np.linalg.lstsq(Aa, true, rcond=None)[0]
    relerr = np.max(np.abs(Aa@coef - true)/true)
    print(f"  affine fit of exp(xi/2) on m+/-{half}: max rel err {relerr:.3%}")


# ----------------------------------------------------------------------
hdr("6. Branch A: the JPR/Metropolis target reproduces the true joint")
# ----------------------------------------------------------------------
# True generative (contemporaneous): (z_t, eta_t) ~ N(0,[[1,rho*sig],[.,sig^2]])
# logh_t = mu + phi(logh_{t-1}-mu) + eta_t ;  e_t = sqrt(h_t) z_t.
# Doc target pi(logh_t) (eq:lev-mh-target), the part depending on logh_t:
#   level:        h^{-1/2} exp(-e^2/2h)
#   transition-in:exp( -(logh - mu - phi(loghm1-mu) - rho*sig*e/sqrt(h))^2 / (2 sig^2(1-rho^2)) )
# Claim: level * transition-in == true joint density p(e_t, logh_t | loghm1) up to const.
mu, phi = -0.1, 0.95
loghm1 = 0.2
mean_eta = mu + phi*(loghm1 - mu)     # prior mean of logh_t

def doc_logdensity(logh, e):
    h = np.exp(logh)
    level = -0.5*logh - e**2/(2*h)
    zt = e/np.sqrt(h)
    trans = -(logh - mean_eta - rho*sig*zt)**2 / (2*sig**2*(1-rho**2))
    return level + trans

def true_logjoint(logh, e):
    # p(e, logh | loghm1): change of vars from (z, eta), Jacobian |d(z,eta)/d(e,logh)| = h^{-1/2}
    eta = logh - mean_eta
    z = e/np.sqrt(logh*0 + np.exp(logh))
    rv = stats.multivariate_normal(mean=[0,0], cov=cov)
    return rv.logpdf(np.c_[z, eta]) - 0.5*logh   # + log|J|, log h^{-1/2} = -0.5 logh

# compare on a grid of (logh, e); difference should be a constant (here exactly 0 up to the
# 2*log(2pi)/normalisation already inside multivariate_normal vs our unnormalised doc form)
LH = np.linspace(-1.5, 1.5, 40)
EE = np.linspace(-3, 3, 40)
G1 = np.array([[doc_logdensity(lh, e) for e in EE] for lh in LH])
G2 = np.array([[true_logjoint(lh, e) for e in EE] for lh in LH])
d = G1 - G2
print(f"  (doc target) - (true joint) over grid: range = "
      f"[{d.min():.5f}, {d.max():.5f}],  std = {d.std():.2e}")
print(f"  constant difference (std ~ 0) => doc target == true conditional up to const   "
      f"{ok(d.std() < 1e-9)}")

hdr("SUMMARY")
print("""All items above that print PASS are verified from first principles or by
internal consistency. NOT covered (need Omori et al. 2007 at the source):
the literal {q_j,m_j,v_j^2,a_j,b_j} tables (7- and 10-component) and the exact
functional form of the Omori fitted linearisation.""")
