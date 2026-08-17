"""
Genera le tabelle LaTeX della Sezione 2 (Data) a partire dalle config, NON a
mano. Cosi' le tabelle della tesi non possono divergere dal dataset stimato.

    python notebooks/tabelle_data.py

Scrive in docs/tesi/tables/:
    tab_dfm_series.tex      37 serie x 3 specificazioni di fattori
    tab_bvar_series.tex     37 serie: trasformazione, centro Minnesota, profilo
    tab_calendar.tex        calendario di pubblicazione per release

ATTENZIONE: ogni file contiene il `tabular` COMPLETO (preambolo di colonne,
intestazione, corpo, \\bottomrule, chiusura), non solo il corpo. Fare
\\input delle sole righe dentro un `tabular` gia' aperto in data.tex fa
fallire \\bottomrule con "Misplaced \\noalign": la riga identica scritta
inline compila, la stessa riga letta da \\input no. Racchiudere l'ambiente
qui dentro elimina il problema alla radice; in data.tex resta soltanto
\\input dentro `table`/`threeparttable`.

Le colonne "Publication timing" e "Reference" del calendario sono trascritte
dalla Tabella 2 di Bok et al. (2018, pp. 623-624); i delay in giorni stanno in
config/series_final.json e sono gli stessi che governano la regola `as_of`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "processed" / "final" / "metadata_final.csv"
FACTOR_SPECS = ROOT / "config" / "factor_specs.json"
BVAR_CFG = ROOT / "config" / "bvar_series.json"
OUT = ROOT / "docs" / "tesi" / "tables"

# Le quattro ISM non vengono dal sito ISM ma da un foglio scaricato da
# investing.com: va detto, perche' non e' la fonte primaria.
SORGENTE = {"fred": "FRED", "manual_xlsx": "Investing.com"}

# Trascritte dalla Tabella 2 di Bok et al. (2018), colonna "Publication timing".
# UNICA eccezione: "Manufacturing and Trade Inventories", che nel paper legge
# "First full week of the month" con reference "1 month prior" — incompatibili
# coi 44 giorni di delay della stessa riga (31-gen + 44 = 16-mar). Riportiamo
# timing e reference che tornano col delay; il delay resta quello del paper.
TIMING = {
    "Construction Spending": "First business day of the month",
    "ISM Manufacturing Report": "First business day of the month",
    "ISM Non-Manufacturing Report": "Third business day of the month",
    "US International Trade in Goods & Services": "First full week of the month",
    "Manufacturers' Shipments, Inventories, Orders (M3)": "First week of the month",
    "ADP National Employment": "First Wednesday of the month",
    "Employment Situation Report": "First Friday of the month",
    "Manufacturing and Trade Inventories": "Middle of the month",
    "Job Openings and Labor Turnover (JOLTS)": "Second week of the month",
    "US Import & Export Price Indexes": "Middle of the month",
    "Retail Trade": "Ninth business day of the month",
    "Producer Price Index": "Middle of the month",
    "Wholesale Trade": "Middle of the month",
    "Empire State Mfg Survey": "15th day of the month",
    "Manufacturing Business Outlook (Philly)": "Third Thursday of the month",
    "Industrial Production & Capacity Utilization": "Middle of the month",
    "Consumer Price Index": "Middle of the month",
    "New Residential Construction": "12th business day of the month",
    "New Residential Sales": "17th business day of the month",
    "Advance Durable Goods": "Third week of the month",
    "Personal Income and Outlays": "Last week of the month",
    "Gross Domestic Product": "Last week of the month",
    "Productivity and Costs": "First week of the month",
}

FONTE = {
    "Construction Spending": "Census",
    "ISM Manufacturing Report": "ISM",
    "ISM Non-Manufacturing Report": "ISM",
    "US International Trade in Goods & Services": "BEA, Census",
    "Manufacturers' Shipments, Inventories, Orders (M3)": "Census",
    "ADP National Employment": "BLS (proxy)",
    "Employment Situation Report": "BLS",
    "Manufacturing and Trade Inventories": "Census",
    "Job Openings and Labor Turnover (JOLTS)": "BLS",
    "US Import & Export Price Indexes": "BLS",
    "Retail Trade": "Census",
    "Producer Price Index": "BLS",
    "Wholesale Trade": "Census",
    "Empire State Mfg Survey": "FRB New York",
    "Manufacturing Business Outlook (Philly)": "FRB Philadelphia",
    "Industrial Production & Capacity Utilization": "FRB Board",
    "Consumer Price Index": "BLS",
    "New Residential Construction": "Census",
    "New Residential Sales": "Census",
    "Advance Durable Goods": "Census",
    "Personal Income and Outlays": "BEA",
    "Gross Domestic Product": "BEA",
    "Productivity and Costs": "BLS",
}

REF = {"1m_prior": "1 month prior", "2m_prior": "2 months prior",
       "current_month": "Current month", "prior_quarter": "Prior quarter"}

UNITS = {
    "MoM_log": r"MoM \% change (log)",
    "QoQ_log_ar": r"QoQ \% change (log, ann.)",
    "diff_ppt": r"Ppt.\ change",
    "diff_level": "Level change (thousands)",
    "level": "Index",
}

NL = "\n"


def esc(s: str) -> str:
    return (str(s).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def _corpo(righe: list[str]) -> str:
    """Righe separate da \\\\, ultima compresa."""
    return (" \\\\" + NL).join(righe) + " \\\\"


def tabella_dfm(meta: pd.DataFrame, specs: dict) -> str:
    ordine = {"fed_overlap": ["G", "S", "R", "L"],
              "diag4": ["S", "R", "L", "N"],
              "diag3": ["A", "N", "L"]}
    macro = {"fed_overlap": r"\markA", "diag4": r"\markB", "diag3": r"\markC"}
    righe = []
    for _, r in meta.iterrows():
        sid = r["series_id"]
        celle = []
        for spec, cols in ordine.items():
            ass = specs["specs"][spec]["assignments"][sid]
            celle += [macro[spec] if c in ass else "" for c in cols]
        righe.append(
            f"{esc(r['paper_name'])} & \\texttt{{{esc(sid)}}} & "
            f"{SORGENTE[r['source']]} & {r['freq']} & "
            f"{pd.Timestamp(r['start_actual']):%Y-%m} & "
            f"{UNITS[r['transform']]} & " + " & ".join(celle)
        )
    testa = [
        r"\begin{tabular}{@{}l l l c c l cccc cccc ccc@{}}",
        r"\toprule",
        r"& & & & & & \multicolumn{4}{c}{\textcolor{specA}{\textbf{fed\_overlap}}}"
        r" & \multicolumn{4}{c}{\textcolor{specB}{\textbf{diag4}}}"
        r" & \multicolumn{3}{c}{\textcolor{specC}{\textbf{diag3}}} \\",
        r"\cmidrule(lr){7-10}\cmidrule(lr){11-14}\cmidrule(lr){15-17}",
        r"Data series & Code & Source & Freq & Start & Units"
        r" & G & S & R & L & S & R & L & N & A & N & L \\",
        r"\midrule",
    ]
    return NL.join(testa) + NL + _corpo(righe) + NL + r"\bottomrule" + NL \
        + r"\end{tabular}" + NL


def tabella_bvar(cfg: dict) -> str:
    tr = {"log": "log level", "level": "level"}
    ce = {"rw": "RW", "wn": "WN"}
    righe = []
    for s in cfg["series"]:
        prof = s.get("profiles", [])
        righe.append(
            f"{esc(s['paper_name'])} & \\texttt{{{esc(s['series_id'])}}} & "
            f"{s['freq']} & {pd.Timestamp(s['first_obs']):%Y-%m} & "
            f"{tr.get(s['transform'], s['transform'])} & "
            f"{ce.get(s['minnesota_centre'], s['minnesota_centre'])} & "
            f"{'$\\bullet$' if 'l' in prof else ''} & "
            f"{'$\\bullet$' if 'q_b' in prof else ''}"
        )
    testa = [
        r"\begin{tabular}{@{}l l c c l c cc@{}}",
        r"\toprule",
        r"& & & & & & \multicolumn{2}{c}{Profile} \\",
        r"\cmidrule(lr){7-8}",
        r"Data series & Code & Freq & Start & Transformation & Prior centre"
        r" & full & reduced \\",
        r"\midrule",
    ]
    return NL.join(testa) + NL + _corpo(righe) + NL + r"\bottomrule" + NL \
        + r"\end{tabular}" + NL


def tabella_calendario(meta: pd.DataFrame) -> str:
    g = (meta.groupby("release_name")
             .agg(delay=("publication_delay_days", "first"),
                  ref=("reference_period", "first"),
                  n=("series_id", "size"),
                  serie=("series_id", lambda s: ", ".join(sorted(s))))
             .reset_index()
             .sort_values("delay"))
    righe = []
    for _, r in g.iterrows():
        nome = r["release_name"]
        righe.append(
            f"{esc(nome)} & {esc(TIMING.get(nome, '--'))} & "
            f"{REF.get(r['ref'], r['ref'])} & {int(r['delay'])} & "
            f"{esc(FONTE.get(nome, '--'))} & {int(r['n'])} & "
            f"\\texttt{{\\scriptsize {esc(r['serie'])}}}"
        )
    testa = [
        r"\begin{tabular}{@{}l l l r l c p{4.6cm}@{}}",
        r"\toprule",
        r"Release & Publication timing & Reference & Delay & Source & $n$"
        r" & Series \\",
        r"\midrule",
    ]
    return NL.join(testa) + NL + _corpo(righe) + NL + r"\bottomrule" + NL \
        + r"\end{tabular}" + NL


def main() -> None:
    meta = pd.read_csv(META)
    specs = json.load(open(FACTOR_SPECS, encoding="utf-8"))
    bvar = json.load(open(BVAR_CFG, encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    (OUT / "tab_dfm_series.tex").write_text(
        tabella_dfm(meta, specs), encoding="utf-8")
    (OUT / "tab_bvar_series.tex").write_text(
        tabella_bvar(bvar), encoding="utf-8")
    (OUT / "tab_calendar.tex").write_text(
        tabella_calendario(meta), encoding="utf-8")

    n_l = sum(1 for s in bvar["series"] if "l" in s.get("profiles", []))
    n_qb = sum(1 for s in bvar["series"] if "q_b" in s.get("profiles", []))
    print(f"DFM       : {len(meta)} serie")
    print(f"BVAR      : profilo full = {n_l}, profilo reduced = {n_qb}")
    print(f"Calendario: {meta['release_name'].nunique()} release distinte")
    print(f"scritto in {OUT}")


if __name__ == "__main__":
    main()
