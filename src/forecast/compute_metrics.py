"""
src/forecast/compute_metrics.py

ACCURATEZZA del nowcast settimanale.

Lettore puro: legge i CSV lunghi prodotti da `weekly_nowcast.py` e calcola le
metriche.  Non ri-stima e non ricalcola nessun nowcast — punteggia solo cio' che
e' gia' su disco contro il PIL realizzato che ogni riga si porta dietro.

L'errore e' in PUNTI BEA (`nowcast_bea - realizzato_bea`), cioe' nell'unita' in
cui il PIL si legge e si discute, non nell'unita' interna del pannello.

L'ASSE DELL'ORIZZONTE
---------------------
L'orizzonte nativo e' la settimana, e sono troppe per una tabella leggibile: da
-17 a +17 farebbero trentacinque righe per metodo.  Le tabelle aggregano quindi
per MESE del trimestre target, che e' la granularita' con cui si ragiona
(secondo mese del trimestre, terzo mese, ...); la settimana resta nel CSV e
nelle figure, dove serve davvero.

    forecast   il trimestre non e' ancora cominciato    (settimane <= 0)
    nowcast M1/M2/M3  primo/secondo/terzo mese del trimestre
    backcast   trimestre finito, PIL non ancora uscito  (settimane > 13)

IL METRO
--------
`RMSE_rel_ar2` = RMSE / RMSE(ar2) allo STESSO orizzonte: sotto 1 il metodo batte
l'AR(2).  `RMSE_rel_mean` fa lo stesso con la media espandente.  L'AR(2) e' il
metro principale perche' e' il piu' esigente dei due; la media espandente e' la
soglia sotto la quale un modello non sta aggiungendo niente.

LA DIREZIONE, NON SOLO LA DISTANZA
----------------------------------
RMSE e MAE sono simmetrici e ciechi al segno: pesano quanto il nowcast dista dal
realizzato, non se lo colloca dalla parte giusta.  Per il PIL la differenza e'
sostanziale — sbagliare di due punti restando in espansione non e' come mancare
il segno della contrazione.  Due metriche direzionali affiancano quindi le
metriche di distanza:

  `MDA`  Mean Directional Accuracy: quota di casi in cui il nowcast prende la
         DIREZIONE DEL CAMBIAMENTO rispetto all'ultimo PIL noto,
             segno(realizzato - ancora)  ==  segno(nowcast - ancora).
         L'ANCORA e' l'ultimo trimestre il cui PIL era GIA' PUBBLICATO alla
         `as_of` della riga (stessa regola dei 28 giorni che governa il
         pannello), non semplicemente il trimestre precedente: cosi' la metrica
         resta real-time come il nowcast che valuta.  Nelle settimane iniziali,
         quando nessun PIL e' ancora uscito, l'ancora non esiste e la riga non
         viene conteggiata (colonna `n_dir`).

  `SignAcc`  quota di casi in cui il nowcast azzecca il SEGNO della crescita
         (espansione vs contrazione).  E' la domanda "chiama la recessione?",
         indipendente da qualunque ancora.

Per entrambe: 1.00 = sempre giusto, 0.50 = come una monetina, sotto 0.50 =
sistematicamente controverso.  Non c'e' colonna relativa al benchmark perche'
il metro naturale e' 0.50, non l'AR(2).

Uso
---
  python -m src.forecast.compute_metrics
  python -m src.forecast.compute_metrics --csv output/forecast_weekly/csv/xxx.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: I due benchmark, per nome: sono i denominatori delle colonne relative.
_AR2 = "ar2"
_MEAN = "mean"
_BENCHMARK_SPEC = "benchmark"

#: Quanti trimestri estremi (per |realizzato|) mostrare a parte.
_N_EXTREME = 4

#: Ritardo di pubblicazione del PIL (advance BEA), in giorni dalla fine del
#: trimestre.  Serve solo a decidere quale PIL era GIA' NOTO a una `as_of`
#: quando si sceglie l'ancora della MDA — e' la stessa regola del pannello.
_GDP_DELAY_DAYS = 28

#: Quanti trimestri all'indietro cercare un'ancora prima di arrendersi.
_MAX_ANCHOR_LOOKBACK = 4


# ─── Lettura ──────────────────────────────────────────────────────────────────

def _csv_dir() -> str:
    from src import output_layout as layout
    return layout.dfm_csv_dir()


def discover_csvs(paths: list[str] | None = None) -> list[str]:
    """I CSV da leggere: quelli dati, o tutti quelli presenti."""
    if paths:
        return list(paths)
    found = sorted(glob.glob(os.path.join(_csv_dir(), "weekly_nowcast_*.csv")))
    if not found:
        raise SystemExit(
            f"Nessun CSV in {_csv_dir()}.\n"
            f"Generane uno con:  python -m src.forecast.weekly_nowcast "
            f"--start YYYY-MM-DD --end YYYY-MM-DD"
        )
    return found


def load_long(paths: list[str] | None = None) -> tuple[pd.DataFrame, int, int]:
    """
    Unisce i CSV in un unico DataFrame lungo, con la colonna `periodo`.

    Le righe il cui trimestre target non ha ancora un PIL realizzato vengono
    scartate e contate: sono i nowcast piu' recenti, non ancora punteggiabili.
    """
    frames = []
    for p in discover_csvs(paths):
        d = pd.read_csv(p)
        d["periodo"] = os.path.basename(p).replace("weekly_nowcast_", "").replace(".csv", "")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    df["target_quarter"] = df["target_quarter"].astype(str)
    df["horizon_week"] = df["horizon_week"].astype(int)
    df["metodo"] = np.where(df["spec"] == _BENCHMARK_SPEC,
                            df["variant"], df["spec"] + "/" + df["variant"])
    df["errore"] = df["nowcast_bea"] - df["realizzato_bea"]
    df["fase"] = df["horizon_week"].map(_phase)
    df = _add_direction(df)

    n_tot = len(df)
    scored = df[df["realizzato_bea"].notna() & df["nowcast_bea"].notna()].copy()
    return scored, n_tot, n_tot - len(scored)


def _quarter_end(q: str) -> pd.Timestamp:
    """'2008Q4' -> Timestamp('2008-12-31')."""
    y, n = str(q).upper().split("Q")
    return pd.Timestamp(int(y), int(n) * 3, 1) + pd.offsets.MonthEnd(0)


def _quarter_label(qe: pd.Timestamp) -> str:
    return f"{qe.year}Q{(qe.month - 1) // 3 + 1}"


def _anchor_value(q: str, as_of: pd.Timestamp, realised: dict[str, float]) -> float:
    """
    Il PIL dell'ultimo trimestre GIA' PUBBLICATO alla data `as_of`: l'ancora
    rispetto a cui si misura la direzione.

    Si va all'indietro dal trimestre target finche' non se ne trova uno il cui
    rilascio (`fine_trimestre + 28g`) e' anteriore o uguale a `as_of` e il cui
    realizzato e' noto.  Se non esiste — le prime settimane del backtest, quando
    nessun PIL e' ancora uscito — si restituisce NaN e la riga non entrera' nella
    MDA.  Usare comunque il trimestre precedente sarebbe un anacronismo: a quella
    data quel valore non era pubblicato.
    """
    qe = _quarter_end(q)
    for k in range(1, _MAX_ANCHOR_LOOKBACK + 1):
        p_end = qe - pd.offsets.QuarterEnd(k)
        if p_end + pd.Timedelta(days=_GDP_DELAY_DAYS) > as_of:
            continue                      # non ancora pubblicato a `as_of`
        val = realised.get(_quarter_label(p_end))
        if val is not None and not pd.isna(val):
            return float(val)
    return float("nan")


def _add_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge `dir_hit` (MDA) e `sign_hit` (accuratezza di segno), entrambe
    float 1.0/0.0/NaN cosi' che una media semplice dia direttamente la quota.

    L'ancora dipende solo da (trimestre target, as_of), non dal metodo: si
    calcola una volta per coppia e si riusa per tutte le celle.
    """
    realised = (df.dropna(subset=["realizzato_bea"])
                  .groupby("target_quarter")["realizzato_bea"].first().to_dict())

    as_of_dt = pd.to_datetime(df["as_of"])
    pairs = dict.fromkeys(zip(df["target_quarter"], as_of_dt))
    anchor_of = {(q, d): _anchor_value(q, d, realised) for q, d in pairs}
    anchor = np.array([anchor_of[(q, d)] for q, d in zip(df["target_quarter"], as_of_dt)])

    real = df["realizzato_bea"].to_numpy(float)
    pred = df["nowcast_bea"].to_numpy(float)

    with np.errstate(invalid="ignore"):
        s_real = np.sign(real - anchor)
        s_pred = np.sign(pred - anchor)
        hit = (s_real == s_pred).astype(float)
        # Senza ancora, o su un pareggio esatto del realizzato, non si giudica.
        hit[~np.isfinite(anchor) | ~np.isfinite(real) | ~np.isfinite(pred)] = np.nan
        hit[s_real == 0] = np.nan
        df["dir_hit"] = hit

        sg = (np.sign(real) == np.sign(pred)).astype(float)
        sg[~np.isfinite(real) | ~np.isfinite(pred)] = np.nan
        sg[np.sign(real) == 0] = np.nan
        df["sign_hit"] = sg
    return df


def _phase(week: int) -> str:
    """Settimana del trimestre -> fase leggibile."""
    if week <= 0:
        return "forecast"
    if week <= 4:
        return "nowcast M1"
    if week <= 9:
        return "nowcast M2"
    if week <= 13:
        return "nowcast M3"
    return "backcast"


_PHASE_ORDER = ["forecast", "nowcast M1", "nowcast M2", "nowcast M3", "backcast"]


# ─── Metriche ─────────────────────────────────────────────────────────────────

def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x, y = a.to_numpy(float), b.to_numpy(float)
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _metrics(g: pd.DataFrame) -> dict:
    e = g["errore"].to_numpy(float)
    n = e.size
    d = g["dir_hit"].to_numpy(float)
    n_dir = int(np.isfinite(d).sum())
    return {
        "n": n,
        "RMSE": float(np.sqrt(np.mean(e ** 2))) if n else float("nan"),
        "MAE": float(np.mean(np.abs(e))) if n else float("nan"),
        "Bias": float(np.mean(e)) if n else float("nan"),
        "corr": _safe_corr(g["nowcast_bea"], g["realizzato_bea"]),
        # Direzionali: media dei colpi, ignorando le righe non giudicabili
        # (nessuna ancora pubblicata, o pareggio esatto).  `n_dir` dice su
        # quante righe la MDA e' stata effettivamente calcolata.
        "MDA": float(np.nanmean(d)) if n_dir else float("nan"),
        "n_dir": n_dir,
        "SignAcc": (float(np.nanmean(g["sign_hit"].to_numpy(float)))
                    if np.isfinite(g["sign_hit"].to_numpy(float)).any()
                    else float("nan")),
    }


def _ordered_methods(present) -> list[str]:
    """Modelli in ordine alfabetico, i due benchmark in coda."""
    present = list(present)
    bench = [m for m in (_AR2, _MEAN) if m in present]
    models = sorted(m for m in present if m not in (_AR2, _MEAN))
    return models + bench


def _add_relative(tab: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """
    Aggiunge RMSE_rel_ar2 e RMSE_rel_mean, appaiati allo stesso `by` (fase o
    niente).  Se un benchmark manca, la sua colonna resta NaN senza rumore.
    """
    for bench, col in ((_AR2, "RMSE_rel_ar2"), (_MEAN, "RMSE_rel_mean")):
        b = tab[tab["metodo"] == bench]
        if b.empty:
            tab[col] = float("nan")
            continue
        if by is None:
            denom = float(b["RMSE"].iloc[0])
            tab[col] = tab["RMSE"] / denom if denom else float("nan")
        else:
            ref = dict(zip(b[by], b["RMSE"]))
            tab[col] = tab.apply(
                lambda r: (r["RMSE"] / ref[r[by]]
                           if ref.get(r[by]) else float("nan")), axis=1)
    return tab


def table_by_method_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Una riga per (metodo, fase del trimestre)."""
    rows = []
    for (m, ph), g in df.groupby(["metodo", "fase"]):
        rec = {"metodo": m, "fase": ph}
        rec.update(_metrics(g))
        rows.append(rec)
    tab = _add_relative(pd.DataFrame(rows), by="fase")
    order = _ordered_methods(tab["metodo"].unique())
    tab["__m"] = tab["metodo"].map({m: i for i, m in enumerate(order)})
    tab["__p"] = tab["fase"].map({p: i for i, p in enumerate(_PHASE_ORDER)})
    return (tab.sort_values(["__m", "__p"]).drop(columns=["__m", "__p"])
            .reset_index(drop=True))


def balanced_quarters(df: pd.DataFrame) -> list[str]:
    """
    I trimestri obiettivo presenti in TUTTE le fasi.

    PERCHE' SERVE.  L'RMSE per fase confronta bin che, agli estremi della
    finestra, NON contengono gli stessi trimestri: il primo obiettivo entra solo
    da backcast (quando la passata comincia, il suo trimestre e' gia' chiuso) e
    l'ultimo solo da forecast.  Con 60 trimestri l'effetto si media via; con 12,
    tutti dentro la Grande Recessione, no.

    Misurato sul pilota 2008-01/2010-06: 4 trimestri su 12 troncati, e i due
    esclusi dal nowcast erano fra i piu' facili (2007Q4 +2.54 solo backcast,
    2010Q3 +3.12 solo forecast).  Risultato: il backcast del Q-BVAR sembrava
    migliorare (3.850 -> 3.731) e bilanciato invece era PIATTO (3.971 -> 3.976).
    Un miglioramento che non c'era, prodotto dalla composizione del bin.
    """
    per_q = df.groupby("target_quarter")["fase"].agg(set)
    tutte = set(_PHASE_ORDER) & set(df["fase"].unique())
    return sorted(q for q, f in per_q.items() if tutte <= f)


def table_by_method_phase_balanced(df: pd.DataFrame
                                   ) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    La tabella 1 sul PANNELLO BILANCIATO: solo i trimestri presenti ovunque.

    Non sostituisce la tabella sbilanciata, la affianca.  Sono due domande
    diverse — "come si comporta il modello su tutto quel che ho" e "come cambia
    l'errore al crescere dell'informazione, a parita' di trimestri" — e solo la
    seconda si legge lungo la riga.

    Returns
    -------
    (tabella, trimestri tenuti, trimestri esclusi)
    """
    tenuti = balanced_quarters(df)
    esclusi = sorted(set(df["target_quarter"].unique()) - set(tenuti))
    sub = df[df["target_quarter"].isin(tenuti)]
    return (table_by_method_phase(sub) if len(sub) else pd.DataFrame(),
            tenuti, esclusi)


def table_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Una riga per metodo, aggregando fasi e date."""
    rows = []
    for m, g in df.groupby("metodo"):
        rec = {"metodo": m}
        rec.update(_metrics(g))
        rows.append(rec)
    tab = _add_relative(pd.DataFrame(rows), by=None)
    order = _ordered_methods(tab["metodo"].unique())
    tab["__m"] = tab["metodo"].map({m: i for i, m in enumerate(order)})
    return tab.sort_values("__m").drop(columns="__m").reset_index(drop=True)


def table_by_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE per metodo x trimestre target: dove ciascuno vince e dove perde."""
    rmse = (df.assign(sq=df["errore"] ** 2)
            .groupby(["metodo", "target_quarter"])["sq"].mean().pow(0.5)
            .rename("RMSE").reset_index())
    pivot = rmse.pivot(index="metodo", columns="target_quarter", values="RMSE")
    return pivot.reindex(_ordered_methods(pivot.index))


def table_extremes(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Nei trimestri piu' estremi (per |realizzato|), il nowcast di ogni metodo
    all'orizzonte piu' profondo disponibile — cioe' il piu' vicino al rilascio,
    quando il modello ha visto il massimo dell'informazione.  E' li' che si
    misura la compressione: il modello arriva al valore estremo o resta a meta'?
    """
    realised = (df.groupby("target_quarter")["realizzato_bea"].first()
                .abs().sort_values(ascending=False))
    rows = []
    for tq in realised.head(_N_EXTREME).index:
        sub = df[df["target_quarter"] == tq]
        hmax = sub["horizon_week"].max()
        deep = sub[sub["horizon_week"] == hmax]
        if deep.empty:
            continue
        real = float(deep["realizzato_bea"].iloc[0])
        for m in _ordered_methods(deep["metodo"].unique()):
            r = deep[deep["metodo"] == m]
            if r.empty:
                continue
            rows.append({
                "target": tq, "realizzato": real, "settimana": int(hmax),
                "metodo": m, "nowcast": float(r["nowcast_bea"].iloc[0]),
                "errore": float(r["errore"].iloc[0]),
                "quota_catturata": (float(r["nowcast_bea"].iloc[0]) / real
                                    if real else float("nan")),
            })
    return pd.DataFrame(rows) if rows else None


# ─── Formattazione ────────────────────────────────────────────────────────────

_FMT = {
    "RMSE": "{:.4f}", "MAE": "{:.4f}", "Bias": "{:+.4f}", "corr": "{:+.3f}",
    "MDA": "{:.3f}", "SignAcc": "{:.3f}",
    # Le gemelle sul campione comune: stesso formato, cosi' che la coppia si
    # legga a colpo d'occhio e la differenza salti.
    "RMSE_com": "{:.4f}", "MDA_com": "{:.3f}", "LS_medio": "{:+.3f}",
    "LS_medio_com": "{:+.3f}",
    "RMSE_rel_ar2": "{:.3f}", "RMSE_rel_mean": "{:.3f}",
    "realizzato": "{:+.3f}", "nowcast": "{:+.3f}", "errore": "{:+.3f}",
    "quota_catturata": "{:.2f}",
}


def _fmt(tab: pd.DataFrame) -> str:
    out = tab.copy()
    for col, f in _FMT.items():
        if col in out.columns:
            out[col] = out[col].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    return out.to_string(index=False)


def _section(t: str) -> str:
    return "\n" + "=" * 84 + "\n" + t + "\n" + "=" * 84


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Metriche del nowcast settimanale.")
    p.add_argument("--csv", nargs="*", default=None,
                   help="CSV da leggere (default: tutti quelli presenti)")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    df, n_tot, n_drop = load_long(a.csv)
    methods = _ordered_methods(df["metodo"].unique())

    header = (
        f"METRICHE DEL NOWCAST SETTIMANALE\n"
        f"errore in punti BEA (nowcast_bea - realizzato_bea)\n"
        f"metodi ({len(methods)}): {', '.join(methods)}\n"
        f"trimestri target: {df['target_quarter'].nunique()}  "
        f"({', '.join(sorted(df['target_quarter'].unique())[:6])}"
        f"{' ...' if df['target_quarter'].nunique() > 6 else ''})\n"
        f"settimane: da {df['horizon_week'].min():+d} a {df['horizon_week'].max():+d}\n"
        f"righe punteggiate: {len(df)} / {n_tot}  "
        f"(scartate {n_drop} senza PIL realizzato)\n"
        f"\n"
        f"distanza : RMSE, MAE, Bias  (punti BEA; RMSE_rel_* <1 batte il benchmark)\n"
        f"direzione: MDA     = prende la direzione del cambiamento rispetto "
        f"all'ultimo PIL pubblicato\n"
        f"           SignAcc = azzecca il segno della crescita "
        f"(espansione vs contrazione)\n"
        f"           per entrambe 0.50 = monetina, 1.00 = sempre giusto; "
        f"n_dir = righe giudicabili"
    )

    t_mp = table_by_method_phase(df)
    t_m = table_by_method(df)
    t_q = table_by_quarter(df)
    t_e = table_extremes(df)

    t_bal, tenuti, esclusi = table_by_method_phase_balanced(df)

    blocks = [
        header,
        _section("1. METODO x FASE DEL TRIMESTRE  "
                 "(RMSE_rel_* appaiato alla stessa fase; <1 batte il benchmark)"),
        _fmt(t_mp),
    ]
    if not t_bal.empty and esclusi:
        blocks += [
            _section("1b. LO STESSO, SU PANNELLO BILANCIATO\n"
                     "    solo i trimestri presenti in TUTTE le fasi: e' l'unica\n"
                     "    versione che si legge LUNGO LA RIGA, perche' la tabella 1\n"
                     "    confronta bin con dentro trimestri diversi"),
            f"   tenuti  ({len(tenuti)}): {', '.join(tenuti)}\n"
            f"   esclusi ({len(esclusi)}): {', '.join(esclusi)}"
            f"   <- troncati al bordo della finestra\n",
            _fmt(t_bal),
        ]
    elif not esclusi:
        blocks += ["\n   [pannello gia' bilanciato: ogni trimestre e' presente "
                   "in tutte le fasi]"]
    blocks += [
        _section("2. METODO  (aggregato su fasi e date)"),
        _fmt(t_m),
    ]
    if t_q.shape[1] > 1:
        blocks += [_section("3. RMSE PER METODO x TRIMESTRE"), _fmt(t_q.reset_index())]
    if t_e is not None:
        blocks += [_section(f"4. TRIMESTRI ESTREMI (i {_N_EXTREME} con |realizzato| "
                            f"maggiore), all'orizzonte piu' profondo\n"
                            f"   quota_catturata = nowcast/realizzato: 1 = ci arriva, "
                            f"0.3 = ne prende un terzo"),
                   _fmt(t_e)]

    report = "\n".join(blocks)
    print(report)

    out_dir = a.out_dir or _csv_dir()
    os.makedirs(out_dir, exist_ok=True)
    txt = os.path.join(out_dir, "metriche.txt")
    csv = os.path.join(out_dir, "metriche_metodo_fase.csv")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    t_mp.to_csv(csv, index=False)
    print(f"\nscritto: {txt}")
    print(f"scritto: {csv}")


if __name__ == "__main__":
    main()
