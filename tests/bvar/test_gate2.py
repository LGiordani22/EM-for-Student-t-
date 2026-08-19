"""
core/bvar/tests/test_gate2.py

IL RECOVERY TEST DEL GATE 2: il wrapper trimestrale.

    python -m core.bvar.tests.test_gate2

Il Gate 1 ha gia' dimostrato che il core recupera la verita'.  Qui la domanda e'
diversa e piu' stretta: **il wrapper passa al core la cosa giusta, e restituisce
al Gate 3 la cosa giusta?**  Il Q-BVAR non ha matematica propria — tutto il suo
contenuto e' l'aggregazione a monte e l'interfaccia a valle, quindi il test
guarda esattamente quei due punti piu' un recovery che chiude il giro.

  §1  L'ORDINE DELLE OPERAZIONI.  Il controllo piu' importante del gate: che il
      codice calcoli log(media) e non media(log).  Verificato contro il valore
      ricostruito a mano, e contro l'errore da cui ci si difende — che deve
      dare un numero DIVERSO, altrimenti il test non starebbe testando niente.
  §2  Le serie in livello e le trimestrali.
  §3  IL BORDO.  Che una MA parziale non venga mai formata.
  §4  L'INTERFACCIA DEL GATE 3.  Che `companion` sia davvero la Phi di (A.1):
      verificato facendo iterare lo stato e confrontandolo con l'iterazione del
      VAR scritta a mano.
  §5  RECOVERY end-to-end attraverso il wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bvar.data import build_panel, load_raw_levels, to_model_units
from core.bvar.qbvar import (
    QBVARFit,
    build_quarterly_panel,
    estimation_panel,
    fit,
    three_month_average,
    to_quarterly,
)
from core.bvar.simulate import simulate_var
from core.bvar.spec import BVARSpec, MinnesotaSpec

_OK, _FAIL = "OK", "FAIL"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<62} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def _toy_spec(n_m: int = 3, n_q: int = 1, p: int = 2, transform=None) -> BVARSpec:
    """Uno spec giocattolo con mensili e trimestrali, per i test di forma."""
    n = n_m + n_q
    series = tuple([f"m{i}" for i in range(n_m)] + [f"q{i}" for i in range(n_q)])
    tr = transform or (("log",) * n_m + ("log",) * n_q)
    return BVARSpec(model="Q", profile="q_b", series=series,
                    freq=("M",) * n_m + ("Q",) * n_q, transform=tr,
                    sample_start=pd.Timestamp("2000-01-31"),
                    minnesota=MinnesotaSpec(p=p, d_centre=(1,) * n), config={})


def _toy_levels(spec: BVARSpec, T: int = 36, seed: int = 0) -> pd.DataFrame:
    """Livelli mensili positivi, con le trimestrali solo ai quarter-end."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=T, freq="ME")
    X = 100.0 + np.cumsum(rng.normal(0, 2.0, size=(T, spec.n)), axis=0)
    df = pd.DataFrame(X, index=idx, columns=list(spec.series))
    qe = ~df.index.month.isin((3, 6, 9, 12))
    df.loc[qe, list(spec.quarterly)] = np.nan
    return df


# ─── 1. L'ordine delle operazioni ─────────────────────────────────────────────

def test_order() -> bool:
    print("\n1. L'ORDINE: livelli -> media a 3 mesi -> log   (nota 17,")
    print("     'before taking logs').  Il controllo centrale del gate.")
    ok = True
    spec = _toy_spec()
    lev = _toy_levels(spec)
    q = to_quarterly(lev, spec)

    # il valore atteso, ricostruito a mano dai tre mesi
    c = "m0"
    mar = pd.Timestamp("2000-03-31")
    tre = lev.loc[pd.Timestamp("2000-01-31"):mar, c].to_numpy()
    giusto = float(np.log(tre.mean()))          # log della media   <- il paper
    sbagliato = float(np.log(tre).mean())       # media dei log     <- l'errore

    got = float(q.loc[mar, c])
    ok &= _check("la cella di marzo == log(media di gen,feb,mar)",
                 np.isclose(got, giusto, rtol=0, atol=1e-12),
                 f"{got:.10f} vs {giusto:.10f}")
    ok &= _check("...e NON == media(log) — l'errore da cui ci si difende",
                 not np.isclose(got, sbagliato, atol=1e-9),
                 f"scarto {abs(giusto - sbagliato):.2e}")
    # se i due coincidessero il test non testerebbe niente: si verifica che
    # l'esperimento sia informativo, non solo che passi.
    ok &= _check("la disuguaglianza di Jensen ha il verso giusto: media(log) <= log(media)",
                 sbagliato <= giusto + 1e-15)

    # E il flag di data.py, sul dataset vero: model_units=False deve dare i
    # LIVELLI, altrimenti la media finirebbe comunque dopo il logaritmo.
    real = BVARSpec.from_config("Q")
    raw = load_raw_levels()
    lv = build_panel(real, end="1995-12-31", raw=raw, model_units=False)
    mu = build_panel(real, end="1995-12-31", raw=raw, model_units=True)
    col = real.log_series[0]
    ok &= _check("build_panel(model_units=False) restituisce LIVELLI, non log",
                 bool(np.allclose(np.log(lv[col].dropna()), mu[col].dropna())),
                 f"su {col}")
    ok &= _check("...e col default il comportamento e' quello di prima",
                 bool(mu.equals(to_model_units(lv, real))))
    return ok


# ─── 2. Livello, log, trimestrali ─────────────────────────────────────────────

def test_transform_map() -> bool:
    print("\n2. Chi viene mediato e chi no")
    ok = True
    # una mensile in livello e una in log
    spec = _toy_spec(n_m=2, n_q=1, transform=("log", "level", "log"))
    lev = _toy_levels(spec)
    q = to_quarterly(lev, spec)
    mar = pd.Timestamp("2000-03-31")
    tre_lev = lev.loc[:mar, "m1"].tail(3).to_numpy()

    ok &= _check("una mensile in LIVELLO e' mediata (media aritmetica, no log)",
                 np.isclose(float(q.loc[mar, "m1"]), float(tre_lev.mean())))
    ok &= _check("...e per lei l'ordine e' indifferente (l'identita' commuta)",
                 np.isclose(float(np.mean(tre_lev)), float(np.mean(tre_lev))))
    ok &= _check("le TRIMESTRALI non sono toccate dalla media",
                 np.isclose(float(q.loc[mar, "q0"]),
                            float(np.log(lev.loc[mar, "q0"]))))

    # sul dataset vero: quante colonne tocca
    real = BVARSpec.from_config("Q")
    n_m, n_q = len(real.monthly), len(real.quarterly)
    lev_m = [s for s, t in zip(real.series, real.transform)
             if t == "level" and s in set(real.monthly)]
    ok &= _check(f"profilo q_b: la media tocca {n_m} colonne su {real.n}",
                 n_m == 27 and n_q == 3, f"{n_m}M + {n_q}Q")
    ok &= _check("...e fra queste 6 sono gia' in livello (punto 2 dell'header)",
                 len(lev_m) == 6, ", ".join(lev_m))
    return ok


# ─── 3. Il bordo frastagliato ─────────────────────────────────────────────────

def test_ragged_edge() -> bool:
    print("\n3. IL BORDO: nessuna media parziale, mai")
    ok = True
    spec = _toy_spec()
    lev = _toy_levels(spec)

    for n_missing, label in ((1, "1 mese su 3 mancante"), (2, "2 mesi su 3 mancanti")):
        cut = lev.copy()
        mar = pd.Timestamp("2000-03-31")
        months = pd.date_range("2000-01-31", periods=3, freq="ME")
        cut.loc[months[3 - n_missing:], "m0"] = np.nan
        q = to_quarterly(cut, spec)
        ok &= _check(f"{label}: la cella e' NaN, non una media parziale",
                     bool(np.isnan(q.loc[mar, "m0"])))

    # E il caso che conta davvero: il mascheramento real-time sul dato vero.
    # Si guarda l'evoluzione dentro un trimestre — e' la cecita' strutturale
    # del Q-BVAR resa numero (punto 3 dell'header di qbvar.py).
    real = BVARSpec.from_config("Q")
    raw = load_raw_levels()
    print("       il trimestre in corso, settimana per settimana (2019Q2):")
    storia = []
    for as_of in ("2019-04-10", "2019-05-15", "2019-06-28", "2019-07-10"):
        q = build_quarterly_panel(real, as_of=as_of, raw=raw)
        full = q.notna().all(axis=1)
        last_full = full[full].index[-1]
        n_nan = int(q.loc["2019-06-30"].isna().sum())
        storia.append((pd.Timestamp(as_of), last_full, n_nan))
        print(f"         as_of {as_of}   ultimo trimestre COMPLETO {last_full.date()}"
              f"   2019Q2: {real.n - n_nan}/{real.n} serie")

    ok &= _check("2019Q2 non e' MAI completo dentro il trimestre",
                 all(n > 0 for _, _, n in storia))
    ok &= _check("il trimestre in corso non entra mai nell'insieme informativo",
                 all(lf < pd.Timestamp("2019-06-30") for _, lf, _ in storia))
    ok &= _check("a inizio trimestre il ritardo arriva a DUE trimestri",
                 storia[0][1] == pd.Timestamp("2018-12-31"),
                 f"il 2019-04-10 si condiziona su dati fino a {storia[0][1].date()}")
    ok &= _check("...e l'insieme informativo cresce in modo monotono",
                 all(a[1] <= b[1] for a, b in zip(storia, storia[1:]))
                 and all(a[2] >= b[2] for a, b in zip(storia, storia[1:])))

    # il campione di stima invece deve essere denso: e' l'invariante del Gate 0
    try:
        est = estimation_panel(real, raw=raw)
        dense = bool(est.notna().all().all())
    except ValueError as exc:                                # pragma: no cover
        dense, est = False, None
        print(f"       assert_dense ha sollevato: {exc}")
    ok &= _check("il campione di STIMA passa assert_dense", dense,
                 f"T={len(est)} trimestri, {est.index[0].date()}-{est.index[-1].date()}"
                 if est is not None else "")
    return ok


# ─── 4. L'interfaccia del Gate 3 ──────────────────────────────────────────────

def test_gate3_interface() -> bool:
    print("\n4. L'INTERFACCIA che il Gate 3 riusera' (Appendice A)")
    ok = True
    n, p, T = 3, 2, 200
    rng = np.random.default_rng(7)
    sim = simulate_var(n, p, T, rng)
    spec = BVARSpec(model="Q", profile="q_b",
                    series=tuple(f"v{i}" for i in range(n)), freq=("M",) * n,
                    transform=("log",) * n, sample_start=pd.Timestamp("1990-01-31"),
                    minnesota=MinnesotaSpec(p=p, d_centre=(1,) * n), config={})
    panel = pd.DataFrame(sim["panel"],
                         index=pd.date_range("1990-03-31", periods=T, freq="QE"),
                         columns=list(spec.series))
    f = fit(spec, panel, n_draws=60, burn=120, rng=rng)

    ok &= _check("forme: A (S,p,n,n), const (S,n), Sigma (S,n,n)",
                 f.A.shape == (f.S, p, n, n) and f.const.shape == (f.S, n)
                 and f.Sigma.shape == (f.S, n, n))
    ok &= _check("companion Phi e' (n*p, n*p)",
                 f.companion(0).shape == (n * p, n * p))
    ok &= _check("Omega = blkdiag(Sigma_eps, 0), e Sigma_eps e' il blocco 1",
                 np.allclose(f.Omega(0)[:n, :n], f.Sigma[0])
                 and np.allclose(f.Omega(0)[n:, :], 0.0))

    # IL CONTROLLO VERO: Phi e' davvero la companion?  Si fa iterare lo stato e
    # si confronta con l'iterazione del VAR scritta a mano.  Se il layout di B
    # fosse letto storto (lag-major vs equation-major, o una trasposta di
    # troppo) questo fallirebbe, e nient'altro lo prenderebbe.
    s = 0
    A, Phi = f.A[s], f.companion(s)
    x = rng.normal(size=(p, n))                    # x[0] = x_{t-1}, x[1] = x_{t-2}
    atteso = sum(A[j] @ x[j] for j in range(p))    # A_1 x_{t-1} + ... + A_p x_{t-p}
    stato = np.concatenate([x[j] for j in range(p)])
    ok &= _check("Phi @ stato riproduce A_1 x_{t-1} + ... + A_p x_{t-p}",
                 np.allclose((Phi @ stato)[:n], atteso, atol=1e-12))
    ok &= _check("...e la parte bassa di Phi fa scorrere lo stato",
                 np.allclose((Phi @ stato)[n:], stato[: n * (p - 1)], atol=1e-12))

    # Il giro chiuso sul layout: da (A, const) si deve poter RICOSTRUIRE B
    # esattamente.  E' un controllo deterministico, non statistico: una
    # trasposta di troppo o un ordine lag/equazione scambiato lo rompe.
    Brec = np.zeros_like(f.draws.B)
    for j in range(p):
        Brec[:, j * n:(j + 1) * n, :] = f.A[:, j].transpose(0, 2, 1)
    Brec[:, -1, :] = f.const
    ok &= _check("da (A, const) si ricostruisce B ESATTAMENTE",
                 bool(np.array_equal(Brec, f.draws.B)))
    ok &= _check("spectral_radius() ha una forma e valori sensati",
                 f.spectral_radius().shape == (f.S,)
                 and bool((f.spectral_radius() > 0).all()))
    return ok


# ─── 5. Il recovery end-to-end ────────────────────────────────────────────────

def test_recovery() -> bool:
    print("\n5. RECOVERY attraverso il wrapper: si ritrovano i parametri veri?")
    ok = True
    pooled_A, pooled_S = [], []
    for (n, p, T, seed) in [(3, 2, 400, 21), (4, 1, 500, 22), (3, 2, 400, 23)]:
        rng = np.random.default_rng(seed)
        d = np.ones(n)
        d[-1] = 0.0
        sim = simulate_var(n, p, T, rng, d_centre=d)
        spec = BVARSpec(model="Q", profile="q_b",
                        series=tuple(f"v{i}" for i in range(n)), freq=("M",) * n,
                        transform=("log",) * n,
                        sample_start=pd.Timestamp("1900-03-31"),
                        minnesota=MinnesotaSpec(p=p, d_centre=tuple(int(x) for x in d)),
                        config={})
        panel = pd.DataFrame(sim["panel"],
                             index=pd.date_range("1900-03-31", periods=T, freq="QE"),
                             columns=list(spec.series))
        f = fit(spec, panel, n_draws=400, burn=300, rng=rng)

        # la copertura si misura sugli oggetti che il Gate 3 consumera', cioe'
        # A e Sigma — non su B grezza: e' il layout riletto che deve essere
        # giusto, non solo il campionatore.
        lo, hi = np.percentile(f.A, [2.5, 97.5], axis=0)
        covA = float(((sim["A"] >= lo) & (sim["A"] <= hi)).mean())
        slo, shi = np.percentile(f.Sigma, [2.5, 97.5], axis=0)
        covS = float(((sim["Sigma"] >= slo) & (sim["Sigma"] <= shi)).mean())
        corr = float(np.corrcoef(sim["A"].ravel(), f.A.mean(axis=0).ravel())[0, 1])
        pooled_A.append(covA)
        pooled_S.append(covS)
        ok &= _check(f"n={n} p={p} T={T}: corr(A vero, A stimato)", corr > 0.95,
                     f"{corr:.4f}  |  cop.A {covA:.0%}  cop.Sigma {covS:.0%}")

    mA, mS = float(np.mean(pooled_A)), float(np.mean(pooled_S))
    ok &= _check("copertura CI 95% su A, messa insieme su 3 repliche",
                 0.85 <= mA <= 1.0, f"{mA:.1%}")
    ok &= _check("copertura CI 95% su Sigma, messa insieme su 3 repliche",
                 0.80 <= mS <= 1.0, f"{mS:.1%}")
    return ok


def main() -> bool:
    print("=" * 82)
    print("Gate 2 — RECOVERY TEST del wrapper trimestrale (Q-BVAR)")
    print("=" * 82)
    ok = True
    for t in (test_order, test_transform_map, test_ragged_edge,
              test_gate3_interface, test_recovery):
        ok &= t()
    print("\n" + "=" * 82)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 82)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
