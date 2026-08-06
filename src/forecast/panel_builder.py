"""
src/forecast/panel_builder.py

Il PANNELLO REAL-TIME a una data.

Il pannello `final` e' UNO: `data/processed/final/dataset_final.csv`, 37 serie,
indice month-end.  Un "vintage" non e' un file separato — e' una VISTA su quel
pannello, generata al volo applicando il calendario di pubblicazione.  Su disco
non si scrive nessun vintage: solo i risultati.

Due cose, in ordine:

  1. **Maschera.**  `release_calendar.known_at(panel, D)` mette a NaN tutto cio'
     che a `D` non era ancora pubblicato.  Il bordo che ne esce e' frastagliato,
     non piatto: le serie con delay corto (Empire State, -14 giorni) arrivano al
     mese in corso, quelle con delay lungo (BUSINV, 44) si fermano due mesi
     prima.

  2. **Estensione.**  Il pannello si allunga con righe tutte-NaN fino al
     quarter-end del trimestre target.  E' questo che rende il nowcast un
     nowcast: su quei mesi non c'e' alcuna osservazione, quindi il filtro non
     aggiorna e propaga soltanto la dinamica dei fattori.  Il PIL del target non
     e' nel pannello per costruzione — la stessa regola di calendario decide sia
     cosa entra sia cosa e' un bersaglio, quindi barare e' strutturalmente
     impossibile.
"""

from __future__ import annotations

import pandas as pd

from src.forecast.release_calendar import known_at, load_metadata, load_panel, quarter_end


def build_panel(as_of_date, target_quarter: str | None = None,
                panel: pd.DataFrame | None = None,
                metadata: pd.DataFrame | None = None,
                exact: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Il pannello com'era noto a `as_of_date`, esteso fino al trimestre target.

    Parameters
    ----------
    as_of_date : str | pd.Timestamp
        La data dell'informazione.
    target_quarter : str, optional
        'YYYYQn'.  Se dato, il pannello viene esteso con righe tutte-NaN fino al
        quarter-end corrispondente (quando questo cade oltre la fine del
        pannello).
    panel, metadata, exact : optional
        Oggetti gia' in memoria, per non rileggere il disco a ogni settimana del
        ciclo.

    Returns
    -------
    pd.DataFrame
        (T, 37), indice month-end, bordo frastagliato, colonne nell'ordine del
        config (GDPC1 ultima).
    """
    panel = load_panel() if panel is None else panel
    metadata = load_metadata() if metadata is None else metadata

    out = known_at(panel, as_of_date, metadata=metadata, exact=exact)

    # Il pannello sorgente arriva a oggi: dopo la maschera, la coda e' fatta di
    # mesi in cui non e' pubblicato ancora nulla.  Vanno tagliati, altrimenti un
    # vintage del 2008 porterebbe con se' diciotto anni di righe vuote — il
    # Kalman e' cubico nello stato ma lineare in T, quindi e' costo puro.
    observed = out.notna().any(axis=1)
    if observed.any():
        out = out.loc[: observed[observed].index[-1]]

    if target_quarter is not None:
        target_qe = quarter_end(target_quarter)
        if target_qe < out.index[0]:
            raise ValueError(
                f"il trimestre target {target_quarter} (fine {target_qe.date()}) "
                f"precede l'inizio del pannello ({out.index[0].date()})."
            )
        if target_qe > out.index[-1]:
            out = out.reindex(pd.date_range(out.index[0], target_qe, freq="ME"))

    out.index.name = None
    return out


def ragged_edge(panel: pd.DataFrame) -> pd.Series:
    """
    L'ultimo mese osservato per ciascuna serie: il profilo del bordo.
    Diagnostica — e' la cosa da guardare quando un nowcast sembra strano.
    """
    return pd.Series(
        {c: (panel[c].dropna().index[-1] if panel[c].notna().any() else pd.NaT)
         for c in panel.columns},
        name="ultimo_mese",
    )


__all__ = ["build_panel", "ragged_edge"]


# ─── Verifica ─────────────────────────────────────────────────────────────────
# Esegui:  python -m src.forecast.panel_builder

if __name__ == "__main__":
    def _hr(t: str) -> None:
        print("\n" + "=" * 74)
        print(t)
        print("=" * 74)

    _hr("panel_builder.py — il bordo frastagliato in azione")

    full = load_panel()
    meta = load_metadata()
    print(f"pannello completo: {full.shape[0]} mesi x {full.shape[1]} serie, "
          f"{full.index[0].date()} .. {full.index[-1].date()}")

    for d in ("2008-11-14", "2020-05-15"):
        _hr(f"build_panel('{d}', target='2008Q4' / '2020Q2')")
        tq = "2008Q4" if d.startswith("2008") else "2020Q2"
        p = build_panel(d, tq, panel=full, metadata=meta)
        edge = ragged_edge(p).sort_values()
        print(f"  shape        : {p.shape[0]} mesi x {p.shape[1]} serie")
        print(f"  ultimo mese  : {p.index[-1].date()}  (esteso al quarter-end di {tq})")
        print(f"  bordo: la serie piu' avanti {edge.index[-1]} -> {edge.iloc[-1].date()}")
        print(f"         la serie piu' indietro {edge.index[0]} -> {edge.iloc[0].date()}")
        spread = (edge.max() - edge.min()).days
        print(f"  frastagliatura: {spread} giorni fra la piu' avanti e la piu' indietro "
              f"-> {'OK (non piatto)' if spread > 0 else 'FAIL (bordo piatto)'}")
        gdp = p["GDPC1"].dropna()
        hidden = quarter_end(tq) > gdp.index[-1]
        print(f"  ultimo PIL noto: {gdp.index[-1].date()}  "
              f"(target {tq} fuori dal pannello: "
              f"{'OK' if hidden else 'FAIL, il target e visibile'})")

    _hr("Il bordo avanza fra due settimane consecutive")
    a = build_panel("2008-11-07", panel=full, metadata=meta)
    b = build_panel("2008-11-14", panel=full, metadata=meta)
    obs_a, obs_b = int(a.notna().sum().sum()), int(b.notna().sum().sum())
    print(f"  osservazioni al 2008-11-07: {obs_a}")
    print(f"  osservazioni al 2008-11-14: {obs_b}")
    print(f"  -> {'OK (informazione cresciuta)' if obs_b > obs_a else 'FAIL'}"
          f"  (+{obs_b - obs_a} celle)")

    _hr("Fatto.")
