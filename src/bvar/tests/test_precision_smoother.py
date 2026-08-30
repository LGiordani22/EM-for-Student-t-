"""
src/bvar/tests/test_precision_smoother.py

Il campionatore a precisione da' la STESSA condizionale del simulation smoother?

    python -m src.bvar.tests.test_precision_smoother

Perche' serve un oracolo DENSO e non il confronto col DK.  Il DK e' gia'
validato contro la congiunta esatta, ma proprio dove serve il campionatore a
precisione — companion esplosiva, T lungo — il DK NON e' affidabile: e' il
motivo per cui l'altro esiste.  Confrontarli li' non proverebbe niente.  Quindi
l'oracolo si ricostruisce a mano: H, Omega e la condizionale gaussiana scritte
esplicitamente, senza importare niente da `precision_smoother`.

I quattro controlli
-------------------
  1. ASSEMBLAGGIO — i blocchi `M[e, d]` ricostruiscono Omega = H'(I(x)S^-1)H
     voce per voce, bordo di coda incluso
  2. MEDIA — con z = 0 l'estrazione e' la media condizionale esatta
  3. COVARIANZA — la covarianza empirica e' quella condizionale esatta, dentro
     l'errore Monte Carlo
  4. CODA — staccare le righe cieche finali e simularle in avanti da' la
     stessa congiunta di quando entrano in Omega (i controlli 2 e 3 girano
     anche su pannelli che finiscono con righe tutte-NaN)

piu' due controlli di STRUTTURA che tengono onesta l'implementazione:

  5. le celle OSSERVATE tornano esatte (qui si condiziona, non si filtra)
  6. il DK e il campionatore concordano dove il DK e' sano (companion stabile,
     T corto): stessa media condizionale entro il nugget
"""

from __future__ import annotations

import numpy as np

from src.bvar.lbvar import build_state_space
from src.bvar.precision_smoother import _omega_blocks, precision_draw
from src.bvar.simsmoother import forward_pass, smoothed_mean

_OK, _FAIL = "OK", "FAIL"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<62} {_OK if cond else _FAIL}"
          + (f"   {detail}" if detail else ""))
    return bool(cond)


class _ZeroRNG:
    """Un generatore che restituisce zeri: isola la MEDIA dall'estrazione."""

    def standard_normal(self, size):
        return np.zeros(size)


# ─── il caso di prova ─────────────────────────────────────────────────────────

def _case(rng, n, p, T, *, miss_rate=0.4, blind_tail=0, scale=0.3):
    """(B, Sigma, Y, head) di un VAR(p) giocattolo con buchi sparsi."""
    A = [scale * rng.standard_normal((n, n)) / np.sqrt(n) for _ in range(p)]
    B = np.vstack([np.vstack([a.T for a in A]),
                   0.1 * rng.standard_normal((1, n))])
    L = np.tril(rng.standard_normal((n, n)))
    Sigma = L @ L.T + 0.5 * np.eye(n)
    head = rng.standard_normal((p, n))
    Y = rng.standard_normal((T, n))
    Y[rng.random((T, n)) < miss_rate] = np.nan
    if blind_tail:
        Y[-blind_tail:] = np.nan
    return B, Sigma, Y, head


def _dense_pieces(B, Sigma, Y, head, n, p):
    """H, Omega, b e la condizionale x_M | x_O, costruiti a mano."""
    T = Y.shape[0]
    A = [np.asarray(B[j * n:(j + 1) * n, :]).T for j in range(p)]
    c = B[-1, :]
    H = np.zeros((T * n, T * n))
    k = np.zeros((T, n))
    for t in range(T):
        H[t * n:(t + 1) * n, t * n:(t + 1) * n] = np.eye(n)
        k[t] = c
        for j in range(1, p + 1):
            s = t - j
            if s >= 0:
                H[t * n:(t + 1) * n, s * n:(s + 1) * n] = -A[j - 1]
            else:
                k[t] = k[t] + A[j - 1] @ head[p + t - j]
    Si = np.linalg.inv(Sigma)
    W = np.kron(np.eye(T), Si)
    Om = H.T @ W @ H
    b = H.T @ W @ k.ravel()
    miss = np.isnan(Y).ravel()
    obs = ~miss
    m = np.linalg.solve(Om[np.ix_(miss, miss)],
                        b[miss] - Om[np.ix_(miss, obs)] @ Y.ravel()[obs])
    return Om, b, m, np.linalg.inv(Om[np.ix_(miss, miss)]), miss


# ─── 1. l'assemblaggio a blocchi ──────────────────────────────────────────────

def test_assemblaggio() -> bool:
    print("\n§1  i blocchi M[e, d] ricostruiscono Omega voce per voce")
    rng = np.random.default_rng(3)
    ok = True
    for (n, p, T) in [(2, 2, 15), (3, 1, 20), (2, 4, 25), (5, 6, 40), (7, 5, 60)]:
        B, Sigma, Y, head = _case(rng, n, p, T)
        Om, *_ = _dense_pieces(B, Sigma, Y, head, n, p)
        A_l = [np.asarray(B[j * n:(j + 1) * n, :]).T for j in range(p)]
        M = _omega_blocks(A_l, np.linalg.inv(Sigma), p)
        Om2 = np.zeros_like(Om)
        for a in range(T):
            e = min(p, T - 1 - a)
            for d in range(p + 1):
                bcol = a + d
                if bcol >= T:
                    break
                Om2[a * n:(a + 1) * n, bcol * n:(bcol + 1) * n] = M[e, d]
                if d:
                    Om2[bcol * n:(bcol + 1) * n, a * n:(a + 1) * n] = M[e, d].T
        err = np.max(np.abs(Om - Om2)) / np.max(np.abs(Om))
        ok &= _check(f"n={n} p={p} T={T}", err < 1e-12, f"err rel {err:.1e}")
    return ok


# ─── 2-4. media, covarianza, coda cieca ───────────────────────────────────────

def test_media() -> bool:
    print("\n§2  con z = 0 l'estrazione E' la media condizionale esatta")
    rng = np.random.default_rng(0)
    ok = True
    for (n, p, T, tail) in [(2, 2, 15, 0), (3, 1, 20, 0), (2, 4, 25, 3),
                            (4, 3, 30, 5)]:
        B, Sigma, Y, head = _case(rng, n, p, T, blind_tail=tail)
        _, _, m_ex, _, miss = _dense_pieces(B, Sigma, Y, head, n, p)
        got = precision_draw(B, Sigma, Y, head, _ZeroRNG(), n=n, p=p,
                             return_full=False)
        err = np.max(np.abs(got.ravel()[miss] - m_ex))
        ok &= _check(f"n={n} p={p} T={T} coda cieca {tail}", err < 1e-9,
                     f"scarto {err:.1e}")
    return ok


def test_covarianza() -> bool:
    print("\n§3  la covarianza empirica e' quella condizionale esatta")
    rng = np.random.default_rng(11)
    ok = True
    S = 20000
    for (n, p, T, tail) in [(2, 2, 12, 0), (3, 2, 15, 4)]:
        B, Sigma, Y, head = _case(rng, n, p, T, blind_tail=tail)
        _, _, _, C_ex, miss = _dense_pieces(B, Sigma, Y, head, n, p)
        r2 = np.random.default_rng(7)
        D = np.empty((S, int(miss.sum())))
        for s in range(S):
            D[s] = precision_draw(B, Sigma, Y, head, r2, n=n, p=p,
                                  return_full=False).ravel()[miss]
        # scarto in unita' dell'errore Monte Carlo di ciascuna voce:
        # sd(C_emp[i,j]) ~ sqrt((C_ii C_jj + C_ij^2)/S)
        d = np.diag(C_ex)
        mc = np.sqrt((np.outer(d, d) + C_ex ** 2) / S)
        z = np.max(np.abs(np.cov(D.T) - C_ex) / mc)
        ok &= _check(f"n={n} p={p} T={T} coda cieca {tail}", z < 6.0,
                     f"scarto max {z:.2f} sd MC")
    return ok


# ─── 5. le celle osservate ────────────────────────────────────────────────────

def test_osservate() -> bool:
    print("\n§5  le celle OSSERVATE tornano esatte: si condiziona, non si filtra")
    rng = np.random.default_rng(5)
    B, Sigma, Y, head = _case(rng, 4, 3, 40, blind_tail=6)
    out = precision_draw(B, Sigma, Y, head, rng, n=4, p=3, return_full=False)
    obs = ~np.isnan(Y)
    err = float(np.max(np.abs(out[obs] - Y[obs])))
    return _check("scarto sulle celle osservate", err == 0.0, f"{err:.1e}")


# ─── 6. l'accordo con il DK dove il DK e' sano ────────────────────────────────

def test_accordo_con_DK() -> bool:
    print("\n§6  dove il DK e' sano (companion stabile, T corto) i due concordano")
    rng = np.random.default_rng(21)
    n, p, T = 3, 2, 40
    B, Sigma, Y, head = _case(rng, n, p, T, miss_rate=0.35, scale=0.25)
    a0 = np.concatenate([head[p - 1 - j] for j in range(p)])
    ss = build_state_space(B, Sigma, n, p, a0)
    rho = float(np.max(np.abs(np.linalg.eigvals(ss.A))))

    # media condizionale del DK: lo smoother sul dato vero, con intercetta
    fp = forward_pass(ss, Y, with_const=True)
    mean_dk = smoothed_mean(ss, fp, with_const=True)[:, :n]
    mean_pr = precision_draw(B, Sigma, Y, head, _ZeroRNG(), n=n, p=p,
                             return_full=False)
    scale = max(1.0, float(np.max(np.abs(mean_dk))))
    err = float(np.max(np.abs(mean_dk - mean_pr))) / scale
    # il DK filtra con R = 1e-12 (sd 1e-6), il campionatore condiziona: lo
    # scarto atteso e' di quell'ordine, non zero
    return _check(f"stessa media condizionale (rho = {rho:.3f})", err < 1e-5,
                  f"scarto rel {err:.1e}")


def main() -> bool:
    print("=" * 82)
    print("Il campionatore a PRECISIONE contro un oracolo denso")
    print("=" * 82)
    ok = True
    for t in (test_assemblaggio, test_media, test_covarianza, test_osservate,
              test_accordo_con_DK):
        ok &= t()
    print("\n" + "=" * 82)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 82)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
