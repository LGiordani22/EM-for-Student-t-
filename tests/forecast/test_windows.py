"""
core/forecast/test_windows.py

LA GUARDIA SULLE FINESTRE.  Non verifica i numeri: verifica che i SEI intervalli
di `output_layout` taglino esattamente il periodo che dichiarano.

    python -m core.forecast.test_windows            dopo la passata (severo)
    python -m core.forecast.test_windows --pre-run  prima della passata

Serve perche' la passata vera dura ore e le tabelle escono alla fine: se una
finestra fosse sbagliata di un anno, lo si scoprirebbe solo allora.  Qui il
controllo costa un secondo e gira su una griglia SINTETICA di venerdi' che
copre tutto il 2007-2025 — cioe' il caso che sul disco non esiste ancora.

`--pre-run` esiste per una ragione sola: prima della passata i CSV non ci sono
per costruzione, e li' l'assenza non e' un guasto.  DOPO la passata lo e', e il
default la tratta come tale — vedi `check_alignment`.
"""

from __future__ import annotations

import sys

import pandas as pd

from core import output_layout as layout


def _synthetic_grid() -> pd.DataFrame:
    """Tutti i venerdi' del campione pieno, con il trimestre che li contiene."""
    fri = pd.date_range(layout.FULL_SPAN[0], layout.FULL_SPAN[1], freq="W-FRI")
    return pd.DataFrame({
        "as_of": fri,
        "target_quarter": [f"{d.year}Q{d.quarter}" for d in fri],
    })


def check() -> int:
    grid = _synthetic_grid()
    print(f"griglia sintetica: {len(grid)} venerdi', "
          f"{grid['as_of'].min().date()} .. {grid['as_of'].max().date()}\n")

    failures = 0
    families = [("forecast", layout.FORECAST_WINDOWS),
                ("rmse pass", layout.RMSE_PASSES),
                ("rmse zoom", layout.RMSE_ZOOM_WINDOWS)]

    for fam, table in families:
        print(f"--- {fam} ---")
        for name, (start, end) in table.items():
            d = layout.slice_window(grid, name)
            a, b = d["as_of"].min(), d["as_of"].max()

            # 1. niente esce dai bordi dichiarati
            inside = a >= pd.Timestamp(start) and b <= pd.Timestamp(end)
            # 2. il primo/ultimo venerdi' dentro l'intervallo e' PROPRIO quello:
            #    se ne restasse fuori uno, la finestra taglierebbe dati veri
            prev = grid[grid["as_of"] < a]["as_of"]
            nxt = grid[grid["as_of"] > b]["as_of"]
            tight = ((prev.empty or prev.max() < pd.Timestamp(start))
                     and (nxt.empty or nxt.min() > pd.Timestamp(end)))
            # 3. gli anni coperti sono esattamente quelli del nome
            y0, y1 = int(start[:4]), int(end[:4])
            years = sorted({t.year for t in d["as_of"]})
            complete = years == list(range(y0, y1 + 1))

            ok = inside and tight and complete
            failures += not ok
            print(f"  {'OK ' if ok else 'ROTTA'}  {name:12s} "
                  f"{a.date()} .. {b.date()}  "
                  f"{len(d):4d} venerdi'  {len(years)} anni "
                  f"({years[0]}..{years[-1]})")
            if not ok:
                print(f"          dentro={inside} aderente={tight} "
                      f"anni_completi={complete}")
        print()

    # Ogni finestra deve stare dentro FULL_SPAN, altrimenti la passata unica
    # non la coprirebbe e la tabella uscirebbe muta senza dirlo.
    for fam, table in families:
        for name, (start, end) in table.items():
            if not (pd.Timestamp(layout.FULL_SPAN[0]) <= pd.Timestamp(start)
                    and pd.Timestamp(end) <= pd.Timestamp(layout.FULL_SPAN[1])):
                print(f"  ROTTA  {name} esce da FULL_SPAN {layout.FULL_SPAN}")
                failures += 1

    print("TUTTE LE FINESTRE OK" if not failures
          else f"{failures} FINESTRE ROTTE")
    return failures


#: I formati che le colonne CHIAVE DEL JOIN devono avere in entrambe le
#: famiglie.  Si controllano PRIMA di unire, e non e' pedanteria: un `as_of`
#: che diventa '2008-10-31 00:00:00' o un `target_quarter` scritto '2008q4'
#: non producono un confronto sbagliato — producono ZERO righe unite, cioe' un
#: controllo che si dichiara superato perche' non ha trovato niente da
#: controllare.  E' il modo di fallire del bug gia' visto su
#: `gdp_release_date`, e va reso rumoroso a monte.
_KEY_FORMATS = {
    "as_of": (r"^\d{4}-\d{2}-\d{2}$", "YYYY-MM-DD senza ora"),
    "target_quarter": (r"^\d{4}Q[1-4]$", "YYYYQn maiuscolo"),
    "gdp_release_date": (r"^(\d{4}-\d{2}-\d{2})?$", "YYYY-MM-DD o vuoto"),
}


def _check_formats(name: str, df) -> int:
    """I formati delle chiavi, famiglia per famiglia.  Ritorna quanti guasti."""
    import pandas as pd
    bad = 0
    for col, (pattern, atteso) in _KEY_FORMATS.items():
        if col not in df.columns:
            print(f"  ROTTA  [{name}] manca la colonna {col}")
            bad += 1
            continue
        s = df[col].fillna("").astype(str)
        ko = s[~s.str.match(pattern)]
        ok = ko.empty
        print(f"  {'OK ' if ok else 'ROTTA'}  [{name}] {col} in formato {atteso}"
              + ("" if ok else
                 f"  <- {len(ko)} righe fuori formato, es. {sorted(set(ko))[:3]}"))
        bad += not ok
    return bad


def check_alignment(require_both: bool = True) -> int:
    """
    DFM e BVAR sono confrontabili?  Si verifica sulle coppie (as_of, target)
    che hanno in comune, perche' e' esattamente su quelle che la matrice di
    `comparison/` li mette uno accanto all'altro.

    Tre cose devono coincidere, e se una salta il confronto e' falso senza
    sembrarlo:
      1. `horizon_week` — se le due famiglie contassero le settimane in modo
         diverso, si confronterebbero orizzonti diversi sotto lo stesso nome;
      2. `realizzato_bea` — deve essere lo stesso GDPC1, non due versioni;
      3. `gdp_release_date` — stessa stringa, non solo stesso istante.

    NIENTE PASSA IN SILENZIO.  La versione precedente ritornava 0 — cioe'
    VERDE — in tre casi che sono guasti veri e non assenze di lavoro: una
    famiglia mancante, un join vuoto, e qualunque scarto di formato sulle
    chiavi (`as_of` con l'ora, `target_quarter` minuscolo), perche' quegli
    scarti si manifestano proprio come join vuoto.  Ora sono ROSSI, e i
    formati si controllano a monte dell'unione cosi' che il messaggio dica
    quale colonna e con quale valore, non "nessuna coppia comune".

    `require_both=False` serve alla sola chiamata PRE-passata, quando i CSV
    non esistono ancora per costruzione: li' l'assenza non e' un guasto.  Dopo
    la passata si usa il default, e l'assenza e' un guasto.
    """
    import pandas as pd
    from core.forecast import metrics_tables as mt

    print("--- allineamento DFM vs BVAR ---")
    try:
        dfm = mt.load_dfm()
    except SystemExit:
        dfm = pd.DataFrame()
    bvar = mt.load_bvar()

    mancanti = [n for n, d in (("DFM", dfm), ("BVAR", bvar)) if d.empty]
    if mancanti:
        if not require_both:
            print(f"  [pre-passata] {', '.join(mancanti)} non ancora su disco: "
                  f"allineamento rimandato al dopo-passata")
            return 0
        print(f"  ROTTA  manca una delle due famiglie: {', '.join(mancanti)} "
              f"vuota.\n         Il confronto BVAR-vs-DFM non e' verificabile, "
              f"quindi NON e' verde.\n         (prima della passata usare "
              f"--pre-run, che rende questa assenza lecita)")
        return len(mancanti)

    bad = _check_formats("DFM", dfm) + _check_formats("BVAR", bvar)

    k = ["as_of", "target_quarter"]
    cols = k + ["horizon_week", "realizzato_bea", "gdp_release_date"]
    a = dfm.drop_duplicates(k)[cols]
    b = bvar.drop_duplicates(k)[cols]
    j = a.merge(b, on=k, suffixes=("_dfm", "_bvar"))
    print(f"  {len(j)} coppie (as_of, target) in comune")

    if j.empty:
        # Entrambe le famiglie hanno righe ma non si incontrano: o le due
        # passate coprono periodi disgiunti, o — molto piu' probabile — una
        # chiave non combacia.  In nessuno dei due casi il confronto e'
        # verificato, quindi in nessuno dei due si dice verde.
        sd = sorted(set(dfm["as_of"].astype(str)))
        sb = sorted(set(bvar["as_of"].astype(str)))
        sqd = sorted(set(dfm["target_quarter"].astype(str)))
        sqb = sorted(set(bvar["target_quarter"].astype(str)))
        print(f"  ROTTA  nessuna coppia comune: il join e' vuoto.\n"
              f"         as_of          DFM {sd[0]}..{sd[-1]}  "
              f"BVAR {sb[0]}..{sb[-1]}\n"
              f"         target_quarter DFM {sqd[0]}..{sqd[-1]}  "
              f"BVAR {sqb[0]}..{sqb[-1]}\n"
              f"         Se i periodi si sovrappongono, il guasto e' nel "
              f"FORMATO di una chiave, non nei dati.")
        return bad + 1

    checks = {
        "horizon_week": (j["horizon_week_dfm"] == j["horizon_week_bvar"]).all(),
        "realizzato_bea": bool((j["realizzato_bea_dfm"]
                                - j["realizzato_bea_bvar"]).abs().max() < 1e-9),
        "gdp_release_date": (j["gdp_release_date_dfm"].astype(str)
                             == j["gdp_release_date_bvar"].astype(str)).all(),
    }
    for name, ok in checks.items():
        print(f"  {'OK ' if ok else 'ROTTA'}  {name} identico")
        bad += not ok

    # La griglia deve essere di venerdi' per entrambe: se una scivolasse di un
    # giorno, l'intersezione crollerebbe a zero e la matrice uscirebbe muta.
    days = ({d.day_name() for d in pd.to_datetime(dfm["as_of"])}
            | {d.day_name() for d in pd.to_datetime(bvar["as_of"])})
    ok = days == {"Friday"}
    print(f"  {'OK ' if ok else 'ROTTA'}  griglia settimanale: {sorted(days)}")
    bad += not ok
    return bad


if __name__ == "__main__":
    pre = "--pre-run" in sys.argv
    failures = check() + check_alignment(require_both=not pre)
    print("\nTUTTO OK" if not failures else f"\n{failures} CONTROLLI ROTTI")
    sys.exit(1 if failures else 0)
