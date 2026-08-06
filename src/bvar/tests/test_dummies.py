"""
src/bvar/tests/test_dummies.py

Il test del Blocco 4: lo stack completo delle dummy observations, BGR eq. (5).

    python -m src.bvar.tests.test_dummies

IL TEST DELL'ORACOLO — perche' questo file esiste
--------------------------------------------------
La stessa teoria e' implementata DUE VOLTE:

  * Blocco 2, forma ANALITICA:  `minnesota_prior_mean` e `minnesota_omega_diag`
    scrivono direttamente le eq. (2)-(3) di Cimadomo;
  * Blocco 4, forma OPERATIVA:  `minnesota_dummy` costruisce le righe fittizie
    di BGR eq. (5), che sono cio' che il posterior usera' davvero.

Le due strade non si parlano: partono da fonti diverse (Cimadomo/GLP contro
BGR) e passano per algebre diverse.  Se i momenti IMPLICATI dalle dummy
coincidono con quelli analitici, allora sia la nostra LETTURA del paper sia la
nostra IMPLEMENTAZIONE sono giuste.  Un percorso solo non lo potrebbe
dimostrare: un errore di lettura si propagherebbe identico nelle due direzioni
senza mai contraddirsi.

E' il controllo della sezione 3, ed e' il motivo per cui il Blocco 2 non era
codice sprecato.
"""

from __future__ import annotations

import numpy as np

from src.bvar import data as bdata
from src.bvar.dummies import (
    DEFAULT_CONST_EPS,
    build_dummies,
    constant_dummy,
    covariance_dummy,
    default_dof,
    implied_prior_moments,
    implied_prior_scale,
    initial_observation_mean,
    minnesota_dummy,
    minnesota_omega_diag,
    minnesota_prior_mean,
)
from src.bvar.spec import BVARSpec, Hyper

_OK, _FAIL = "OK", "FAIL"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<64} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def _fixture(seed: int = 0):
    """psi ETEROGENEI: con psi tutti uguali il fattore di scala sparirebbe e
    l'oracolo non distinguerebbe psi_j da psi_i."""
    spec = BVARSpec.from_config("Q")
    rng = np.random.default_rng(seed)
    psi = np.exp(rng.normal(-4.0, 1.5, size=spec.n))
    hyp = Hyper(lam=0.6, mu=1.0, psi=psi)
    y0 = np.abs(rng.normal(10.0, 4.0, size=spec.n)) + 1.0
    return spec, hyp, y0


# ─── 1. La forma dei tre blocchi ──────────────────────────────────────────────

def test_block_shapes() -> bool:
    print("\n1. I tre blocchi di BGR eq. (5): forma e struttura")
    spec, hyp, _ = _fixture()
    n, p, k = spec.n, spec.p, spec.k
    sigma = np.sqrt(hyp.psi)
    ok = True

    Ym, Xm = minnesota_dummy(spec, hyp)
    ok &= _check("Minnesota: np righe", Ym.shape == (n * p, n) and Xm.shape == (n * p, k),
                 f"{n*p} righe")
    ok &= _check("Minnesota: Yd ha diag(delta*sigma)/lambda nelle prime n righe",
                 bool(np.allclose(Ym[:n], np.diag(spec.minnesota.d * sigma / hyp.lam))))
    ok &= _check("Minnesota: Yd e' zero sotto le prime n righe",
                 not Ym[n:].any())
    ok &= _check("Minnesota: Xd e' J_p (x) diag(sigma)/lambda, cioe' diagonale",
                 bool(np.allclose(Xm[:, :n * p],
                                  np.diag(np.concatenate(
                                      [s * sigma / hyp.lam for s in range(1, p + 1)])))))
    ok &= _check("Minnesota: colonna della costante nulla",
                 not Xm[:, -1].any())

    Yc, Xc = covariance_dummy(spec, hyp)
    ok &= _check("covarianza: n righe, Yd = diag(sigma)",
                 Yc.shape == (n, n) and np.allclose(Yc, np.diag(sigma)))
    ok &= _check("covarianza: Xd interamente NULLA (non tocca b ne' Omega)",
                 not Xc.any())

    Yk, Xk = constant_dummy(spec, DEFAULT_CONST_EPS)
    ok &= _check("costante: 1 riga, Yd = 0", Yk.shape == (1, n) and not Yk.any())
    ok &= _check("costante: Xd = [0...0, eps]",
                 Xk.shape == (1, k) and not Xk[0, :-1].any()
                 and Xk[0, -1] == DEFAULT_CONST_EPS)
    try:
        constant_dummy(spec, -1.0)
        ok &= _check("eps <= 0 rifiutato", False)
    except ValueError:
        ok &= _check("eps <= 0 rifiutato", True)
    return ok


# ─── 2. Lo stack ──────────────────────────────────────────────────────────────

def test_stack() -> bool:
    print("\n2. Lo stack assemblato")
    spec, hyp, y0 = _fixture()
    n, p, k = spec.n, spec.p, spec.k
    st = build_dummies(spec, hyp, y0)
    ok = True

    ok &= _check("Td = np + n + 1 + n (con soc)",
                 st.Td == n * p + n + 1 + n, f"Td={st.Td}")
    ok &= _check("Yd e' (Td, n), Xd e' (Td, k)",
                 st.Yd.shape == (st.Td, n) and st.Xd.shape == (st.Td, k))
    ok &= _check("quattro blocchi, nell'ordine atteso",
                 list(st.blocks) == ["minnesota", "covariance", "constant", "soc"])
    ok &= _check("i blocchi coprono tutte le righe senza buchi",
                 sum(s.stop - s.start for s in st.blocks.values()) == st.Td)

    st_no = st.without("soc")
    ok &= _check("without('soc') toglie n righe",
                 st_no.Td == st.Td - n and "soc" not in st_no.blocks)
    ok &= _check("senza y0_bar il blocco soc non c'e'",
                 "soc" not in build_dummies(spec, hyp).blocks)
    return ok


# ─── 3. L'ORACOLO: le dummy riproducono le eq. (2)-(3)? ───────────────────────

def test_oracle() -> bool:
    print("\n3. L'ORACOLO — le dummy riproducono i momenti analitici del Blocco 2?")
    spec, hyp, y0 = _fixture()
    ok = True

    # Lo stack SENZA soc deve riprodurre ESATTAMENTE le eq. (2)-(3).
    st = build_dummies(spec, hyp, y0).without("soc")
    b0, om0 = implied_prior_moments(st.Yd, st.Xd)

    b_analytic = minnesota_prior_mean(spec)
    ok &= _check("B0 implicito dalle dummy == minnesota_prior_mean  (eq. 2)",
                 bool(np.allclose(b0, b_analytic, atol=1e-10)),
                 f"max|err| = {np.abs(b0 - b_analytic).max():.2e}")

    # Omega analitica: la costante ha inf, la dummy 1/eps^2 -> confronto sui
    # regressori veri e separatamente sulla costante.
    om_analytic = minnesota_omega_diag(spec, hyp)
    got = np.diag(om0)
    ok &= _check("diag(Omega0) == minnesota_omega_diag  (eq. 3, psi_j incluso)",
                 bool(np.allclose(got[:-1], om_analytic[:-1], rtol=1e-10)),
                 f"max err rel = {np.abs(got[:-1]/om_analytic[:-1] - 1).max():.2e}")
    ok &= _check("Omega0 e' DIAGONALE senza soc (come dice l'eq. 3)",
                 bool(np.allclose(om0 - np.diag(np.diag(om0)), 0.0, atol=1e-12)))
    ok &= _check("Omega0 sulla costante = 1/eps^2 (prior piatto approssimato)",
                 bool(np.isclose(got[-1], 1.0 / DEFAULT_CONST_EPS ** 2, rtol=1e-8)),
                 f"{got[-1]:.4g}")

    # Psi implicita: solo il blocco covarianza contribuisce
    psi_hat = implied_prior_scale(st.Yd, st.Xd)
    ok &= _check("Psi implicita == diag(psi)",
                 bool(np.allclose(psi_hat, np.diag(hyp.psi), atol=1e-12)),
                 f"max|err| = {np.abs(psi_hat - np.diag(hyp.psi)).max():.2e}")

    # ---- L'ORACOLO DEVE MORDERE ----------------------------------------
    # Una nota sul perche' NON si puo' scrivere una controprova "psi_i al posto
    # di psi_j": Omega e' indicizzata dai soli REGRESSORI e non ha un indice di
    # equazione, quindi psi_i non e' nemmeno ESPRIMIBILE come sua diagonale.
    # E' precisamente l'argomento strutturale del Blocco 2 — il denominatore
    # psi_i romperebbe la fattorizzazione di Kronecker.
    #
    # La controprova sensata e' un'altra: verificare che l'oracolo si accorga
    # se i psi finissero sui regressori SBAGLIATI.  Costruisco le dummy con
    # sigma permutato e controllo che Omega implicita non torni piu'.
    n, p = spec.n, spec.p
    perm = np.roll(np.arange(n), 1)
    hyp_perm = Hyper(lam=hyp.lam, mu=hyp.mu, psi=hyp.psi[perm])
    st_perm = build_dummies(spec, hyp_perm, y0).without("soc")
    _, om_perm = implied_prior_moments(st_perm.Yd, st_perm.Xd)
    ok &= _check("controprova: con psi permutati l'oracolo FALLISCE (ha denti)",
                 not np.allclose(np.diag(om_perm)[:-1], om_analytic[:-1], rtol=1e-6),
                 f"scarto rel max = "
                 f"{np.abs(np.diag(om_perm)[:-1]/om_analytic[:-1] - 1).max():.3g}")

    # e simmetricamente: sbagliare il decadimento sul lag verrebbe visto
    lam_only = np.concatenate([
        np.tile(hyp.lam ** 2 / hyp.psi, p),        # senza il fattore 1/s^2
        [np.inf]])
    ok &= _check("controprova: senza il decadimento 1/s^2 l'oracolo FALLISCE",
                 not np.allclose(got[:-1], lam_only[:-1], rtol=1e-6))
    return ok


# ─── 4. I gradi di liberta' ───────────────────────────────────────────────────

def test_dof() -> bool:
    print("\n4. I gradi di liberta': fissati, non contati")
    spec, hyp, y0 = _fixture()
    n, k = spec.n, spec.k
    ok = True

    ok &= _check("default_dof(n) = n+2", default_dof(n) == n + 2, f"{n+2}")
    st = build_dummies(spec, hyp, y0)
    ok &= _check("lo stack porta dof = n+2 anche CON il soc",
                 st.dof == n + 2, f"dof={st.dof}")

    # il controllo incrociato di BGR vale sullo stack SENZA soc...
    st_no = st.without("soc")
    ok &= _check("cross-check BGR: Td - k + 2 = n+2 sullo stack senza soc",
                 st_no.Td - k + 2 == n + 2,
                 f"{st_no.Td} - {k} + 2 = {st_no.Td - k + 2}")
    # ...e NON vale con il soc: e' esattamente la trappola documentata
    ok &= _check("...e NON vale col soc (Td-k+2 = 2n+2): la trappola e' reale",
                 st.Td - k + 2 == 2 * n + 2,
                 f"darebbe {st.Td - k + 2} invece di {n + 2}")
    ok &= _check("dof NON dipende dal numero di righe (GLP A.8)",
                 build_dummies(spec, hyp).dof == build_dummies(spec, hyp, y0).dof)
    return ok


# ─── 5. Che cosa aggiunge il soc allo stack ───────────────────────────────────

def test_soc_effect() -> bool:
    print("\n5. Che cosa cambia quando si aggiunge il soc")
    spec, hyp, y0 = _fixture()
    n, p = spec.n, spec.p
    full = build_dummies(spec, hyp, y0)
    bare = full.without("soc")
    ok = True

    b_full, om_full = implied_prior_moments(full.Yd, full.Xd)
    b_bare, om_bare = implied_prior_moments(bare.Yd, bare.Xd)

    # (a) col soc Omega NON e' piu' diagonale: il soc vincola SOMME, e questo
    #     introduce correlazioni fra lag diversi della stessa variabile.
    #     GLP: "It also introduces correlation among the coefficients on each
    #     variable in each equation."
    off = om_full - np.diag(np.diag(om_full))
    ok &= _check("col soc Omega NON e' piu' diagonale (correlazioni fra lag)",
                 float(np.abs(off).max()) > 1e-12,
                 f"max fuori diagonale = {np.abs(off).max():.4g}")

    # (b) il soc STRINGE: aggiunge righe, quindi Xd'Xd cresce e Omega cala
    ok &= _check("il soc stringe il prior (varianze <= senza soc)",
                 bool(np.all(np.diag(om_full)[:-1] <= np.diag(om_bare)[:-1] + 1e-12)))

    # (c) la correlazione indotta e' FRA LAG DELLA STESSA VARIABILE
    j = 3
    idx = [s * n + j for s in range(p)]           # stessa variabile, tutti i lag
    sub = om_full[np.ix_(idx, idx)]
    other = om_full[idx[0], (0 * n + (j + 1) % n)]
    ok &= _check("la correlazione e' fra lag della STESSA variabile",
                 float(np.abs(sub[0, 1])) > 1e-12 and abs(other) < 1e-12,
                 f"stessa var: {sub[0,1]:.4g} | var diversa: {other:.4g}")

    # (d) LA TENSIONE wn-vs-soc, quantificata.
    #     Minnesota dice sum_s (A_s)_ii = d_i (0 per le wn); il soc dice 1 per
    #     TUTTE.  Sul centro implicito la wn viene tirata via da 0.
    lag_sum_full = np.array([b_full[[s * n + i for s in range(p)], i].sum()
                             for i in range(n)])
    lag_sum_bare = np.array([b_bare[[s * n + i for s in range(p)], i].sum()
                             for i in range(n)])
    wn = [i for i in range(n) if spec.minnesota.d[i] == 0.0]
    rw = [i for i in range(n) if spec.minnesota.d[i] == 1.0]

    ok &= _check("senza soc: somma sui propri lag = d_centre esattamente",
                 bool(np.allclose(lag_sum_bare, spec.minnesota.d, atol=1e-10)))
    # LA TENSIONE wn<->soc, RISOLTA.  Prima del fix `ydnoc(pos,pos)=0` il soc
    # tirava le serie wn da 0 verso 1 (misurato: ISM_PMI 0.996, ISM_PRICES
    # 1.000, ISM_EMP 1.000, Philly 1.000), annullando la scelta `wn` del Gate 0.
    # Azzerando le righe wn i due prior concordano: entrambi dicono 0.
    ok &= _check("col soc le wn RESTANO a 0 (i due prior concordano)",
                 bool(np.allclose(lag_sum_full[wn], 0.0, atol=1e-6)),
                 "somme wn: " + ", ".join(f"{spec.series[i]}={lag_sum_full[i]:.3f}"
                                          for i in wn))
    ok &= _check("le rw restano vicine a 1 (i due prior sono d'accordo)",
                 bool(np.all(np.abs(lag_sum_full[rw] - 1.0) < 0.05)),
                 f"scarto max rw = {np.abs(lag_sum_full[rw] - 1.0).max():.4f}")
    return ok


# ─── 6. Sui dati veri ─────────────────────────────────────────────────────────

def test_on_real_data() -> bool:
    print("\n6. Sui dati veri (q_b, campione di stima)")
    spec = BVARSpec.from_config("Q")
    est = bdata.build_panel(spec, end=bdata.estimation_end(spec))
    qe = est.index[est.index.month.isin((3, 6, 9, 12))]
    panel = est.loc[qe]
    ok = True

    y0 = initial_observation_mean(panel, spec.p)
    hyp = Hyper(lam=0.6, mu=1.0, psi=np.full(spec.n, 0.02 ** 2))
    st = build_dummies(spec, hyp, y0)

    ok &= _check("lo stack si costruisce senza NaN/inf",
                 bool(np.all(np.isfinite(st.Yd)) and np.all(np.isfinite(st.Xd))))

    # LA RAGIONE PRATICA DELLE DUMMY: X'X e' singolare, X*'X* no.
    # BGR: "Adding dummy observations works as a regularisation solution to the
    # matrix inversion problem."
    Y = panel.to_numpy()[spec.p:]
    lags = [panel.to_numpy()[spec.p - s: -s] for s in range(1, spec.p + 1)]
    X = np.column_stack(lags + [np.ones(len(Y))])
    ok &= _check("X e' piu' stretta di k: X'X SINGOLARE senza dummy",
                 X.shape[0] < spec.k,
                 f"T={X.shape[0]} osservazioni contro k={spec.k} regressori")
    rank_x = int(np.linalg.matrix_rank(X.T @ X))
    ok &= _check("...e infatti rank(X'X) < k",
                 rank_x < spec.k, f"rango {rank_x} su {spec.k}")

    Xstar = np.vstack([st.Xd, X])
    rank_s = int(np.linalg.matrix_rank(Xstar.T @ Xstar))
    ok &= _check("le dummy RIGENERANO il rango: rank(X*'X*) = k",
                 rank_s == spec.k, f"rango {rank_s} su {spec.k}")

    cond = float(np.linalg.cond(Xstar.T @ Xstar))
    ok &= _check("X*'X* e' invertibile con condizionamento gestibile",
                 np.isfinite(cond) and cond < 1e14, f"cond = {cond:.3e}")

    # e l'oracolo regge anche coi numeri veri
    bare = st.without("soc")
    b0, om0 = implied_prior_moments(bare.Yd, bare.Xd)
    ok &= _check("oracolo confermato sui dati veri (B0 e Omega)",
                 bool(np.allclose(b0, minnesota_prior_mean(spec), atol=1e-10)
                      and np.allclose(np.diag(om0)[:-1],
                                      minnesota_omega_diag(spec, hyp)[:-1], rtol=1e-10)))
    print(f"     T = {X.shape[0]} osservazioni + {st.Td} dummy = {Xstar.shape[0]} righe, k = {spec.k}")
    return ok


def main() -> bool:
    print("=" * 84)
    print("Gate 1 / Blocco 4 — lo stack delle dummy observations (BGR eq. 5)")
    print("=" * 84)
    ok = True
    for fn in (test_block_shapes, test_stack, test_oracle, test_dof,
               test_soc_effect, test_on_real_data):
        ok &= fn()
    print("\n" + "=" * 84)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 84)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
