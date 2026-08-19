"""
core/forecast/test_common_sample.py

LA GUARDIA SUL CAMPIONE COMUNE.  Non verifica i numeri della passata: verifica
che il VINCOLO scatti, su dati sintetici costruiti perche' senza vincolo la
risposta sarebbe visibilmente sbagliata.

    python -m core.forecast.test_common_sample

PERCHE' SINTETICI E NON I CSV VERI
-----------------------------------
Sul disco le due famiglie, quando ci sono, sono quasi allineate: il vincolo non
morde e un test che girasse li' passerebbe anche se fosse rotto.  Qui i due
insiemi sono DIVERSI APPOSTA — il DFM su quaranta trimestri, il BVAR su dodici,
e un modello privo di bordo — cosi' che ogni controllo abbia un valore atteso
diverso a seconda che il vincolo funzioni o no.

    l'errore vale 5.0 sui trimestri NON condivisi, 1.0 sui condivisi
    -> con vincolo   l'RMSE del DFM e' 1.000
    -> senza vincolo e' 4.219

Costa un secondo e va lanciato PRIMA della passata, accanto a `test_windows`:
la passata dura ore e queste tabelle escono alla fine.

I QUATTRO CONTROLLI
-------------------
  1. la matrice aggregata di `comparison/` restringe all'intersezione;
  2. le tabelle di famiglia riportano `n` e `n_com` e il comune e' quello vero
     — e' la coppia di colonne su cui si legge il confronto con la NY Fed;
  3. il confronto PER FASE tiene il backcast ai modelli che ce l'hanno anche
     quando uno ne e' privo.  E' il controllo che conta di piu': sull'insieme,
     un modello senza bordo cancellava il backcast a tutti;
  4. un modello troppo rado in una fase esce da QUELLA fase invece di
     decimarne il campione, e viene dichiarato nella nota.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from core.forecast import compute_metrics as cm
from core.forecast import metrics_tables as mt

#: Le settimane di un trimestre completo, dalla convenzione di Cascaldi-Garcia.
_WEEKS = list(range(-12, 18))

#: Il bordo: un modello che non lo ha si ferma qui.
_NO_BORDER = [w for w in _WEEKS if w <= 13]


# ─── Il finto pannello ────────────────────────────────────────────────────────

def _quarters(y0: int, y1: int) -> list[str]:
    return [f"{y}Q{n}" for y in range(y0, y1 + 1) for n in range(1, 5)]


def _fridays(q: str, weeks: list[int]) -> list[str]:
    """Una `as_of` plausibile per (trimestre, settimana): sempre un venerdi'."""
    y, n = int(q[:4]), int(q[-1])
    q0 = pd.Timestamp(y, (n - 1) * 3 + 1, 1)
    q0 = q0 + pd.Timedelta(days=(4 - q0.weekday()) % 7)
    return [(q0 + pd.Timedelta(weeks=w - 1)).date().isoformat() for w in weeks]


def _build(metodo: str, quarters: list[str], weeks: list[int],
           err_fuori: float, err_dentro: float, ultimo_fuori: int = 2014,
           extra_profondo: float = 0.0) -> pd.DataFrame:
    """
    Un metodo, con errore COSTANTE dentro ciascuno dei due blocchi.

    Costante apposta: cosi' l'RMSE atteso e' esattamente `err_dentro` sul
    campione comune, e un test che fallisse non lascerebbe dubbi su quanto.

    `extra_profondo` aggiunge una penalita' sulle settimane < -4, cioe' il
    forecast profondo.  Serve a riprodurre il fenomeno vero: sono le settimane
    piu' difficili per tutti, e sono anche quelle in cui la Fed non pubblica —
    quindi vengono addebitate a me sole.  Con l'errore indipendente dalla
    settimana la restrizione toglierebbe punti senza cambiare l'RMSE, e il
    test non distinguerebbe un vincolo funzionante da uno rotto.
    """
    rows = []
    for q in quarters:
        real = 2.0 + 0.1 * int(q[-1])
        base = err_fuori if int(q[:4]) <= ultimo_fuori else err_dentro
        for w, d in zip(weeks, _fridays(q, weeks)):
            e = base + (extra_profondo if w < -4 else 0.0)
            rows.append({
                "as_of": d, "target_quarter": q, "horizon_week": w,
                "metodo": metodo, "realizzato_bea": real,
                "nowcast_bea": real + e, "gdp_release_date": "2099-01-01",
            })
    return pd.DataFrame(rows)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Le colonne derivate, con la stessa meccanica di `compute_metrics`."""
    df = df.copy()
    df["errore"] = df["nowcast_bea"] - df["realizzato_bea"]
    df["fase"] = df["horizon_week"].map(cm._phase)
    return cm._add_direction(df)


def panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Le due famiglie finte.  DFM su 40 trimestri, BVAR su 12 — un sottoinsieme —
    e `qbvar` senza bordo, che e' il caso che rompeva il backcast di tutti.
    """
    dfm_q, bvar_q = _quarters(2008, 2017), _quarters(2015, 2017)
    dfm = _finish(pd.concat([
        _build("fed_overlap/gaussian", dfm_q, _WEEKS, 5.0, 1.0),
        _build("fed_overlap/student_t", dfm_q, _WEEKS, 5.0, 1.2),
        _build("ar2", dfm_q, _WEEKS, 4.0, 2.0),
        _build("mean", dfm_q, _WEEKS, 4.5, 2.5),
    ], ignore_index=True))
    bvar = _finish(pd.concat([
        _build("cbvar/authors", bvar_q, _WEEKS, 0.0, 1.5),
        _build("qbvar/-", bvar_q, _NO_BORDER, 0.0, 0.8),      # NIENTE BACKCAST
        _build("ar2", dfm_q, _WEEKS, 4.0, 2.0),               # stessi benchmark
        _build("mean", dfm_q, _WEEKS, 4.5, 2.5),
    ], ignore_index=True))
    return dfm, bvar


# ─── I controlli ──────────────────────────────────────────────────────────────

def _esito(ok: bool, titolo: str, dettaglio: str = "") -> int:
    print(f"  {'OK   ' if ok else 'ROTTA'}  {titolo}")
    if dettaglio:
        for riga in dettaglio.splitlines():
            print(f"           {riga}")
    return 0 if ok else 1


def check_matrix(dfm: pd.DataFrame, bvar: pd.DataFrame) -> int:
    """1. La matrice aggregata restringe all'intersezione?"""
    print("\n--- 1. matrice di confronto aggregata ---")
    rmse, _, _, note = mt.comparison_matrices(dfm, bvar, windows=["2007-2019"])
    got = float(rmse.loc["fed_overlap/gaussian", "2007-2019"])
    libero = float(np.sqrt(np.mean(
        dfm[dfm["metodo"] == "fed_overlap/gaussian"]["errore"]
        .to_numpy(float) ** 2)))
    bad = _esito(abs(got - 1.0) < 1e-9,
                 "l'RMSE e' quello del campione comune",
                 f"ottenuto {got:.4f} | comune atteso 1.0000 | "
                 f"senza vincolo sarebbe {libero:.4f}")

    q = mt.common_points(mt._both_families(dfm, bvar))["target_quarter"]
    bad += _esito(q.nunique() == 12,
                  "l'intersezione e' i 12 trimestri del BVAR, non i 40 del DFM",
                  f"trimestri comuni: {q.nunique()}")
    bad += _esito("NOT REPRESENTATIVE" in note,
                  "la nota segnala che la colonna non rappresenta la finestra")
    return bad


def check_family(dfm: pd.DataFrame, bvar: pd.DataFrame) -> int:
    """
    2. Le tabelle di famiglia affiancano libero e comune?

    Il campione comune di una tabella di famiglia e' quello dei metodi CHE CI
    SONO DENTRO: `_methods_of` tiene la spec richiesta piu' i benchmark piu' la
    Fed, non i BVAR.  Il caso che conta e' quindi la RIGA NY FED, che copre
    meno settimane delle mie — sui dati veri 309 punti contro 360 — ed e' l'
    unica riga per cui `n` e `n_com` divergono davvero.  Si ricostruisce qui la
    stessa asimmetria: la Fed non pubblica prima della settimana -4.
    """
    print("\n--- 2. tabelle di famiglia: n accanto a n_com (riga NY Fed) ---")
    # Il caso vero, in piccolo: io copro le settimane profonde (dove sbaglio
    # di piu'), la Fed comincia a -4 e quelle non le vede mai.
    qs = _quarters(2008, 2017)
    mio = _finish(pd.concat([
        _build("fed_overlap/gaussian", qs, _WEEKS, 2.0, 2.0, extra_profondo=3.0),
        _build("ar2", qs, _WEEKS, 3.0, 3.0, extra_profondo=3.0),
    ], ignore_index=True))
    fed = _finish(_build("nyfed", qs, [w for w in _WEEKS if w >= -4], 2.5, 2.5))
    m, mp = mt.family_tables(pd.concat([mio, fed], ignore_index=True),
                             "fed_overlap", windows=["2007-2019"])

    bad = _esito({"n", "n_com", "RMSE", "RMSE_com"} <= set(m.columns),
                 "le quattro colonne ci sono tutte",
                 f"colonne: {list(m.columns)}")
    if bad:
        return bad

    r = m[m["metodo"] == "nyfed"]
    bad += _esito(not r.empty, "la riga NY Fed e' nella tabella")
    if r.empty:
        return bad
    r, g = r.iloc[0], m[m["metodo"] == "fed_overlap/gaussian"].iloc[0]

    bad += _esito(int(g["n"]) > int(g["n_com"]),
                  "il DFM ha n > n_com: la restrizione gli toglie punti",
                  f"n={int(g['n'])}  n_com={int(g['n_com'])}")
    bad += _esito(int(r["n"]) == int(r["n_com"]),
                  "la Fed ha n = n_com: e' lei il sottoinsieme, non perde nulla",
                  f"n={int(r['n'])}  n_com={int(r['n_com'])}")
    bad += _esito(int(g["n_com"]) == int(r["n_com"]),
                  "Fed e DFM hanno lo STESSO n_com: e' il punto di tutto",
                  f"DFM n={int(g['n'])} n_com={int(g['n_com'])} | "
                  f"Fed n={int(r['n'])} n_com={int(r['n_com'])}")
    bad += _esito(float(g["RMSE"]) > float(g["RMSE_com"]) + 1e-6,
                  "il DFM MIGLIORA sul comune: perde le sue settimane peggiori",
                  f"RMSE={float(g['RMSE']):.4f} -> "
                  f"RMSE_com={float(g['RMSE_com']):.4f}")

    # E la lettura che cambia: sul libero il confronto e' falsato, sul comune no.
    libero = "DFM" if float(g["RMSE"]) < float(r["RMSE"]) else "Fed"
    comune = "DFM" if float(g["RMSE_com"]) < float(r["RMSE_com"]) else "Fed"
    print(f"           chi vince sul campione libero: {libero}  "
          f"({float(g['RMSE']):.3f} vs {float(r['RMSE']):.3f})")
    print(f"           chi vince sul campione comune: {comune}  "
          f"({float(g['RMSE_com']):.3f} vs {float(r['RMSE_com']):.3f})")

    # Nella tabella per fase il comune e' calcolato DENTRO la fase: il
    # backcast del DFM non deve sparire solo perche' un altro non ce l'ha.
    bc = mp[(mp["fase"] == "backcast")
            & (mp["metodo"] == "fed_overlap/gaussian")]
    bad += _esito(not bc.empty and int(bc.iloc[0]["n_com"]) > 0,
                  "il backcast del DFM sopravvive nella tabella per fase",
                  f"n_com backcast = "
                  f"{int(bc.iloc[0]['n_com']) if not bc.empty else 'assente'}")
    return bad


def check_phase(dfm: pd.DataFrame, bvar: pd.DataFrame) -> int:
    """3. Il confronto per fase salva il backcast agli altri modelli?"""
    print("\n--- 3. confronto per fase: il backcast esiste ---")
    long, note = mt.comparison_by_phase(dfm, bvar, windows=["2007-2019"])

    bad = _esito(not long.empty, "il confronto per fase produce righe")
    if bad:
        return bad

    fasi = set(long["fase"].unique())
    bad += _esito("backcast" in fasi,
                  "il backcast e' fra le fasi confrontate",
                  f"fasi presenti: {sorted(fasi)}")

    bc = long[long["fase"] == "backcast"]
    metodi = set(bc["metodo"])
    bad += _esito("qbvar/-" not in metodi,
                  "il modello senza bordo non compare nel backcast (giusto)")
    bad += _esito({"cbvar/authors", "fed_overlap/gaussian"} <= metodi,
                  "i modelli CON bordo ci sono ancora",
                  f"metodi nel backcast: {sorted(metodi)}")

    # E il numero deve essere quello del campione comune di quella fase.
    r = bc[bc["metodo"] == "fed_overlap/gaussian"]
    bad += _esito(not r.empty and abs(float(r.iloc[0]["RMSE"]) - 1.0) < 1e-9,
                  "l'RMSE di backcast e' sul comune della fase",
                  f"{float(r.iloc[0]['RMSE']):.4f}" if not r.empty else "assente")

    # Tutti i metodi della fase devono avere lo STESSO n: e' la prova diretta
    # che il confronto e' sugli stessi target.
    bad += _esito(bc["n"].nunique() == 1,
                  "tutti i metodi del backcast hanno lo stesso n",
                  bc[["metodo", "n", "RMSE"]].to_string(index=False))

    # La matrice dedicata deve esistere e non essere vuota.
    bm = mt.backcast_matrix(long)
    bad += _esito(not bm.empty, "la matrice di backcast dedicata e' popolata",
                  bm.round(3).to_string() if not bm.empty else "")
    return bad


def check_thin(dfm: pd.DataFrame, bvar: pd.DataFrame) -> int:
    """4. Un metodo rado esce dalla fase invece di decimarla?"""
    print("\n--- 4. esclusione dei metodi radi ---")
    # Un quinto metodo che in backcast ha UN trimestre solo su dodici: senza
    # la regola imporrebbe quel trimestre a tutti.
    rado = _finish(_build("lbvar/-", ["2015Q1"], _WEEKS, 0.0, 3.0))
    bvar2 = pd.concat([bvar, rado], ignore_index=True)

    long, note = mt.comparison_by_phase(dfm, bvar2, windows=["2007-2019"])
    bc = long[long["fase"] == "backcast"]

    bad = _esito("lbvar/-" not in set(bc["metodo"]),
                 "il metodo rado e' escluso dal backcast")
    bad += _esito("esclusi da questa fase" in note,
                  "l'esclusione e' DICHIARATA nella nota, non taciuta")
    n_altri = int(bc[bc["metodo"] == "fed_overlap/gaussian"]["n"].iloc[0]) \
        if not bc[bc["metodo"] == "fed_overlap/gaussian"].empty else 0
    bad += _esito(n_altri == 48,
                  "gli altri conservano i loro 48 punti (12 trimestri x 4)",
                  f"n = {n_altri}")
    return bad


def check_logscore() -> int:
    """
    5. Il log score dei BVAR affianca il comune al libero?

    Il caso: un modello che ha girato solo sui trimestri facili esce davanti
    senza aver fatto niente di meglio.  Qui `qbvar` ha meta' dei punti — e i
    suoi log score sono i migliori — ma sul campione comune la classifica e'
    quella vera.
    """
    print("\n--- 5. log score sul campione comune ---")
    import tempfile
    from core.bvar import metrics as bm

    qs = _quarters(2015, 2017)
    righe = []
    for m, quarters, ls_facile, ls_difficile in (
            ("cbvar", qs, -1.0, -3.0),
            ("bbvar", qs, -1.1, -3.1),
            ("lbvar", qs, -1.2, -3.2),
            ("qbvar", qs[:6], -1.3, -3.3),      # solo la prima meta'
    ):
        for q in quarters:
            facile = int(q[-1]) in (1, 2)
            for w, d in zip(_WEEKS, _fridays(q, _WEEKS)):
                righe.append({
                    "as_of": d, "target_quarter": q, "horizon_week": w,
                    "spec": m, "variant": "authors" if m == "cbvar" else "-",
                    "log_score": ls_facile if facile else ls_difficile,
                    "realizzato_bea": 2.0,
                })
    ls = pd.DataFrame(righe)
    ls["metodo"] = ls["spec"] + "/" + ls["variant"]
    ls["fase"] = ls["horizon_week"].map(cm._phase)

    with tempfile.TemporaryDirectory() as d:
        by_phase = bm.logscore_tables(ls, d)
    # `logscore_tables` stampa; qui contano solo le colonne.
    bad = _esito({"n", "n_com", "LS_medio", "LS_medio_com"} <= set(by_phase.columns),
                 "le colonne libere e comuni ci sono tutte",
                 f"colonne: {list(by_phase.columns)}")
    if bad:
        return bad

    bc = by_phase[by_phase["fase"] == "backcast"].set_index("metodo")
    bad += _esito(int(bc.loc["cbvar/authors", "n"])
                  > int(bc.loc["cbvar/authors", "n_com"]),
                  "chi ha piu' righe vede n > n_com",
                  f"cbvar n={int(bc.loc['cbvar/authors', 'n'])} "
                  f"n_com={int(bc.loc['cbvar/authors', 'n_com'])}")
    bad += _esito(bc["n_com"].nunique() == 1,
                  "tutti i modelli hanno lo stesso n_com",
                  f"{bc['n_com'].to_dict()}")
    return bad


def main() -> int:
    print("GUARDIA SUL CAMPIONE COMUNE — dati sintetici, insiemi diversi "
          "apposta")
    dfm, bvar = panels()
    print(f"  DFM : {dfm['target_quarter'].nunique()} trimestri, "
          f"settimane {dfm['horizon_week'].min():+d}.."
          f"{dfm['horizon_week'].max():+d}")
    print(f"  BVAR: {bvar[bvar['metodo'] == 'cbvar/authors']['target_quarter'].nunique()}"
          f" trimestri; qbvar senza settimane > 13")

    bad = (check_matrix(dfm, bvar) + check_family(dfm, bvar)
           + check_phase(dfm, bvar) + check_thin(dfm, bvar)
           + check_logscore())
    print("\nCAMPIONE COMUNE OK" if not bad else f"\n{bad} CONTROLLI ROTTI")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
