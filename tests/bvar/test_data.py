"""
core/bvar/tests/test_data.py

Il test del Gate 0: `spec.py` legge davvero la config, e `data.py` produce
davvero il pannello che il core si aspetta?

    python -m core.bvar.tests.test_data

Non e' ancora un recovery test — non c'e' niente da recuperare, il modello non
esiste.  E' il gate ingegneristico: verifica che le decisioni prese stiano
DAVVERO nei dati che usciranno, non solo nel file di configurazione.

I sette controlli
-----------------
  1. la config si carica e i conteggi sono quelli decisi (30/37, 8/29, 6/31)
  2. gli spec si costruiscono e `d_centre` viene dalla config
  3. il pannello ha le colonne giuste, nell'ordine giusto, dal 1992 per q_b
  4. le colonne `log` sono davvero logaritmate, le `level` davvero intatte
  5. NESSUNA standardizzazione: media e deviazione non sono 0 e 1
  6. il profilo q_b e' DENSO fino a `last_dense_date` (la proprieta' che il
     core esige) e le trimestrali stanno solo sui quarter-end
  7. `assert_dense` sa distinguere i tre tipi di buco, e in particolare vede il
     buco interno del 2025-10
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bvar import data as bdata
from core.bvar.spec import BVARSpec, Hyper, MinnesotaSpec, load_config

_OK, _FAIL = "OK", "FAIL"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<58} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


# ─── 1. La config ─────────────────────────────────────────────────────────────

def test_config() -> bool:
    print("\n1. La config dice quello che abbiamo deciso")
    cfg = load_config()
    s = cfg["series"]
    ok = True
    ok &= _check("37 serie in totale", len(s) == 37, f"n={len(s)}")
    ok &= _check("profilo q_b = 30 serie",
                 len(cfg["profiles"]["q_b"]["series_ids"]) == 30)
    ok &= _check("profilo l = 37 serie",
                 len(cfg["profiles"]["l"]["series_ids"]) == 37)
    n_lvl = sum(1 for e in s if e["transform"] == "level")
    n_log = sum(1 for e in s if e["transform"] == "log")
    ok &= _check("8 in livello, 29 in log", (n_lvl, n_log) == (8, 29),
                 f"{n_lvl}/{n_log}")
    n_wn = sum(1 for e in s if e["minnesota_centre"] == "wn")
    ok &= _check("6 white-noise, 31 random-walk", n_wn == 6, f"wn={n_wn}")

    # i sei wn sono ESATTAMENTE le survey: la regola del paper §3.1
    wn = {e["series_id"] for e in s if e["minnesota_centre"] == "wn"}
    expected = {"ISM_PMI", "ISM_PRICES", "ISM_EMP", "ISM_NMI",
                "GACDISA066MSFRBNY", "GACDFSA066MSFRBPHI"}
    ok &= _check("i wn sono le 6 survey/diffusione", wn == expected,
                 "" if wn == expected else str(wn ^ expected))

    # livello e white-noise sono DUE ASSI INDIPENDENTI: UNRATE e TCU sono in
    # livello ma centrate su random walk, come nella Tabella 1 del paper.
    by_id = {e["series_id"]: e for e in s}
    ok &= _check("UNRATE e TCU: livello MA random walk (assi indipendenti)",
                 all(by_id[x]["transform"] == "level"
                     and by_id[x]["minnesota_centre"] == "rw"
                     for x in ("UNRATE", "TCU")))

    # la nota che deve restare visibile finche' il relatore non decide
    ok &= _check("ISM_PRICES ha la nota 'DA VALIDARE'",
                 "DA VALIDARE" in by_id["ISM_PRICES"].get("note", ""))
    return ok


# ─── 2. Gli spec ──────────────────────────────────────────────────────────────

def test_spec() -> bool:
    print("\n2. Gli spec si costruiscono DALLA CONFIG")
    ok = True
    specs = {m: BVARSpec.from_config(m) for m in ("Q", "C", "B", "L")}
    for m, sp in specs.items():
        print(f"     {sp.summary()}")

    ok &= _check("Q, C, B condividono il profilo q_b (C = stadio 1 di Q)",
                 all(specs[m].profile == "q_b" for m in ("Q", "C", "B")))
    ok &= _check("L usa il profilo l (37 serie)",
                 specs["L"].profile == "l" and specs["L"].n == 37)
    ok &= _check("n = 30 per Q/C/B", all(specs[m].n == 30 for m in ("Q", "C", "B")))
    ok &= _check("p = 5 per Q/C/B, p = 17 per L (paper nota 10)",
                 specs["Q"].p == 5 and specs["L"].p == 17)
    ok &= _check("k = n*p+1 = 151 per Q", specs["Q"].k == 151, f"k={specs['Q'].k}")

    # d_centre viene dalla config, non da un array cablato
    q = specs["Q"]
    cfg_centre = {e["series_id"]: e["minnesota_centre"] for e in load_config()["series"]}
    from_cfg = np.array([1.0 if cfg_centre[s] == "rw" else 0.0 for s in q.series])
    ok &= _check("d_centre coincide con la config, serie per serie",
                 np.array_equal(q.minnesota.d, from_cfg))
    ok &= _check("d_centre e' 0/1 e float, pronto per np.diag(d)",
                 q.minnesota.d.dtype == float and set(np.unique(q.minnesota.d)) <= {0.0, 1.0})

    # il dummy-initial-observation resta FUORI dal default (Decisione 4)
    ok &= _check("use_dio = False di default (Cimadomo §2.1: 3 iperparametri)",
                 q.minnesota.use_dio is False)
    ok &= _check("use_soc = True di default", q.minnesota.use_soc is True)
    ok &= _check("lag_decay = 2 (paper nota 8)", q.minnesota.lag_decay == 2.0)
    ok &= _check("use_dio si accende senza cambiare firma",
                 BVARSpec.from_config("Q", use_dio=True).minnesota.use_dio is True)

    # Hyper: psi e' di sola lettura, cosi' il ciclo Metropolis non lo corrompe
    h = Hyper(lam=0.6, mu=1.0, psi=np.full(q.n, 0.02 ** 2))
    try:
        h.psi[0] = 999.0
        writable = True
    except ValueError:
        writable = False
    ok &= _check("Hyper.psi e' di sola lettura", not writable)
    ok &= _check("Hyper.delta = None di default", h.delta is None)

    # le validazioni mordono
    for label, fn in (
        ("d_centre rifiuta valori != 0/1",
         lambda: MinnesotaSpec(p=5, d_centre=(1, 0, 2))),
        ("psi rifiuta valori non positivi",
         lambda: Hyper(lam=0.6, mu=1.0, psi=np.array([1.0, -1.0]))),
        ("lam rifiuta valori non positivi",
         lambda: Hyper(lam=0.0, mu=1.0, psi=np.array([1.0]))),
    ):
        try:
            fn()
            ok &= _check(label, False)
        except ValueError:
            ok &= _check(label, True)
    return ok


# ─── 3-5. Il pannello ─────────────────────────────────────────────────────────

def test_panel() -> bool:
    print("\n3. Il pannello: colonne, ordine, campione")
    raw = bdata.load_raw_levels()
    q = BVARSpec.from_config("Q")
    lp = BVARSpec.from_config("L")
    pan_q = bdata.build_panel(q, raw=raw)
    pan_l = bdata.build_panel(lp, raw=raw)
    ok = True

    ok &= _check("q_b: 30 colonne", pan_q.shape[1] == 30, f"shape={pan_q.shape}")
    ok &= _check("l: 37 colonne", pan_l.shape[1] == 37, f"shape={pan_l.shape}")
    ok &= _check("colonne nell'ordine dello spec",
                 list(pan_q.columns) == list(q.series))
    ok &= _check("q_b parte dal 1992-01-31",
                 pan_q.index.min() == pd.Timestamp("1992-01-31"),
                 str(pan_q.index.min().date()))
    ok &= _check("l parte dal 1985-01-31",
                 pan_l.index.min() == pd.Timestamp("1985-01-31"),
                 str(pan_l.index.min().date()))
    ok &= _check("le 7 serie tardive sono in l e NON in q_b",
                 set(lp.series) - set(q.series) == {
                     "DGORDER", "TTLCONS", "ISM_NMI", "JTSJOL",
                     "GACDISA066MSFRBNY", "PCEC96", "PPIFIS"})

    print("\n4. Le trasformazioni: log dove serve, identita' dove serve")
    # una colonna log deve essere il log del grezzo; una level deve essere identica
    for col, kind in (("GDPC1", "log"), ("PAYEMS", "log"),
                      ("UNRATE", "level"), ("ISM_PMI", "level")):
        got = pan_q[col].dropna()
        ref = raw.loc[got.index, col]
        want = np.log(ref) if kind == "log" else ref
        same = bool(np.allclose(got.to_numpy(), want.to_numpy(), rtol=0, atol=1e-12))
        ok &= _check(f"{col} e' in {kind}", same)

    # e nessuna colonna 'level' e' stata logaritmata per sbaglio
    untouched = all(
        np.allclose(pan_q[c].dropna().to_numpy(),
                    raw.loc[pan_q[c].dropna().index, c].to_numpy(),
                    rtol=0, atol=1e-12)
        for c in q.level_series
    )
    ok &= _check(f"tutte e {len(q.level_series)} le 'level' sono intatte", untouched)

    # il log e' plausibile: GDPC1 in log-livello sta intorno a 9-10
    g = pan_q["GDPC1"].dropna()
    ok &= _check("GDPC1 in log-livello e' nell'ordine di grandezza giusto",
                 9.0 < g.mean() < 11.0, f"media={g.mean():.3f}")

    print("\n5. NIENTE standardizzazione (lo scaling lo fa il prior via psi)")
    mu = pan_q.mean(numeric_only=True)
    sd = pan_q.std(numeric_only=True)
    ok &= _check("le medie NON sono ~0",
                 bool((mu.abs() > 1e-6).sum() >= 25),
                 f"|media|>1e-6 su {(mu.abs() > 1e-6).sum()}/30")
    ok &= _check("le deviazioni NON sono ~1",
                 bool(((sd - 1.0).abs() > 1e-6).sum() >= 25),
                 f"|sd-1|>1e-6 su {((sd - 1.0).abs() > 1e-6).sum()}/30")
    return ok


# ─── 6-7. La densita', che e' l'invariante ────────────────────────────────────

def test_density() -> bool:
    print("\n6. Il profilo q_b e' RETTANGOLARE (l'invariante del core)")
    q = BVARSpec.from_config("Q")
    pan = bdata.build_panel(q)
    ok = True

    last = bdata.last_dense_date(pan, q)
    ok &= _check("esiste una finestra densa", last is not None, str(last.date()))

    # LA DECISIONE PRESA sullo shutdown 2025-10 dev'essere REGISTRATA, non solo
    # ottenuta per caso dal calcolo: la config fissa estimation_end e deve
    # coincidere con l'ultima data densa trovata sui dati.
    est_end = bdata.estimation_end(q)
    ok &= _check("q_b ha un estimation_end registrato in config",
                 est_end == pd.Timestamp("2025-09-30"), str(est_end))
    ok &= _check("estimation_end coincide con l'ultima data densa calcolata",
                 est_end == last)
    ok &= _check("il profilo l non fissa un estimation_end (non gli serve)",
                 bdata.estimation_end(BVARSpec.from_config("L")) is None)
    # e il campione di stima costruito con quel taglio passa l'invariante
    est = bdata.build_panel(q, end=est_end)
    try:
        bdata.assert_dense(est, q)
        ok &= _check("il campione di stima end=estimation_end e' denso", True,
                     f"{est.index.min().date()} .. {est.index.max().date()}")
    except ValueError as exc:
        ok &= _check("il campione di stima end=estimation_end e' denso", False,
                     str(exc)[:80])

    win = pan.loc[:last]
    mon = list(q.monthly)
    qrt = list(q.quarterly)
    ok &= _check(f"nessun NaN nelle {len(mon)} mensili sulla finestra densa",
                 not win[mon].isna().any().any(),
                 f"{win.index.min().date()} .. {win.index.max().date()}")

    qe = win.index[win.index.month.isin((3, 6, 9, 12))]
    ok &= _check(f"le {len(qrt)} trimestrali sono osservate a ogni quarter-end",
                 not win.loc[qe, qrt].isna().any().any(),
                 f"{len(qe)} trimestri")
    nqe = win.index[~win.index.month.isin((3, 6, 9, 12))]
    ok &= _check("e sono NaN fuori dai quarter-end (placement, non buco)",
                 int(win.loc[nqe, qrt].notna().sum().sum()) == 0)

    n_q = len(qe)
    ok &= _check("T utile >= 100 trimestri (regime del paper, T~130)",
                 n_q >= 100, f"T={n_q}, usabili con p=5: {n_q - q.p}")

    # la finestra densa passa assert_dense; il pannello intero no
    try:
        bdata.assert_dense(win, q)
        ok &= _check("assert_dense PASSA sulla finestra densa", True)
    except ValueError as exc:
        ok &= _check("assert_dense PASSA sulla finestra densa", False, str(exc)[:90])

    print("\n7. assert_dense distingue i tre tipi di buco")
    try:
        bdata.assert_dense(pan, q)
        ok &= _check("assert_dense FALLISCE sul pannello intero (bordo)", False)
    except ValueError as exc:
        ok &= _check("assert_dense FALLISCE sul pannello intero (bordo)",
                     "BORDO FRASTAGLIATO" in str(exc))

    # PARTENZE TARDIVE — attenzione a non confondere due conteggi diversi:
    #   * le 7 serie ESCLUSE da q_b sono quelle tardive rispetto al taglio
    #     1992-01, che e' il criterio con cui il profilo e' stato costruito;
    #   * nel profilo l, che parte dal 1985-01, e' tardiva ogni serie che non
    #     nasce a gennaio 1985 — quindi anche le 8 che partono nel 1992 e che
    #     in q_b sono perfettamente a posto.  Sono 17 in tutto.
    # Il punto che conta e' che le 7 escluse da q_b siano un SOTTOINSIEME delle
    # tardive di l, e che li' siano latenti anziche' fatali.
    lp = BVARSpec.from_config("L")
    pan_l = bdata.build_panel(lp)
    rep_l = bdata.gaps_report(pan_l, lp)
    late = set(rep_l[rep_l["late_start"] > 0]["series_id"])
    excluded = set(lp.series) - set(q.series)
    ok &= _check("le 7 serie escluse da q_b sono tardive anche in l",
                 excluded <= late, f"{sorted(excluded - late)}")
    ok &= _check("in l le partenze tardive sono 17 (rispetto al 1985-01)",
                 len(late) == 17, f"n={len(late)}")
    ok &= _check("PPIFIS e' la piu' tardiva: ~298 mesi latenti",
                 int(rep_l.set_index("series_id").loc["PPIFIS", "late_start"]) == 298)
    # le 3 trimestrali NON devono risultare tardive: il loro primo valore cade
    # sul primo quarter-end del campione, e i mesi non-quarter-end non contano
    ok &= _check("le trimestrali non sono contate come tardive (placement)",
                 not (set(lp.quarterly) & late))
    ok &= _check("nel profilo q_b NON ci sono partenze tardive",
                 int(bdata.gaps_report(pan, q)["late_start"].sum()) == 0)

    # IL BUCO INTERNO DEL 2025-10 (shutdown federale USA): deve essere VISTO.
    # E' il controllo che vale di piu' di tutti, perche' e' l'unico problema
    # che non si risolve cambiando profilo.
    rep = bdata.gaps_report(pan, q)
    inter = rep[rep["interior"] > 0]
    got = set(inter["series_id"])
    want = {"CPIAUCSL", "UNRATE", "CPILFESL", "IR", "IQ"}
    ok &= _check("il buco interno 2025-10 e' rilevato sulle 5 serie giuste",
                 got == want, f"{sorted(got)}")
    ok &= _check("ed e' datato 2025-10-31",
                 all("2025-10-31" in r for r in inter["interior_dates"]))

    # e assert_dense lo nomina come INTERNO, non lo confonde col bordo
    try:
        bdata.assert_dense(pan.loc[:"2025-12-31"], q)
        ok &= _check("assert_dense chiama 'BUCHI INTERNI' il caso 2025-10", False)
    except ValueError as exc:
        ok &= _check("assert_dense chiama 'BUCHI INTERNI' il caso 2025-10",
                     "BUCHI INTERNI" in str(exc))
    return ok


# ─── 8. Il calendario ─────────────────────────────────────────────────────────

def test_as_of() -> bool:
    print("\n8. Il calendario as-of si applica ai livelli (pseudo-real-time)")
    q = BVARSpec.from_config("Q")
    ok = True
    full = bdata.build_panel(q)
    masked = bdata.build_panel(q, as_of="2015-06-19")

    ok &= _check("stessa forma", full.shape == masked.shape)
    ok &= _check("mascherare AGGIUNGE NaN, non ne toglie",
                 bool((masked.isna() >= full.isna()).all().all()))
    tail = masked.loc["2015-07-31":]
    ok &= _check("dopo l'as_of non resta nulla di osservato",
                 int(tail.notna().sum().sum()) == 0)
    edge = masked.loc[:"2015-06-30"].notna()[::-1].idxmax()
    ok &= _check("il bordo a quella data e' FRASTAGLIATO, non piatto",
                 edge.nunique() > 1,
                 f"{edge.nunique()} date di fine diverse fra le 30 serie")

    # i valori osservati non vengono toccati dal mascheramento
    keep = masked.loc[:"2015-01-31"].dropna(axis=1, how="all")
    same = np.allclose(keep.to_numpy(dtype=float),
                       full.loc[keep.index, keep.columns].to_numpy(dtype=float),
                       rtol=0, atol=1e-12, equal_nan=True)
    ok &= _check("i valori pubblicati restano identici", same)
    return ok


def main() -> bool:
    print("=" * 78)
    print("Gate 0 — spec.py + data.py")
    print("=" * 78)
    ok = True
    for fn in (test_config, test_spec, test_panel, test_density, test_as_of):
        ok &= fn()
    print("\n" + "=" * 78)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 78)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
