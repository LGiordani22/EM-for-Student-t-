"""
src/forecast/test_target_windows.py

La FINESTRA DI VITA di un trimestre target, e il suo riflesso nella figura 8a.

Tre cose devono valere insieme, e sono facili da rompere una alla volta:

  1. INIZIO   la linea di `q` comincia ~13 settimane prima dell'inizio di `q`,
              cioe' un trimestre di anticipo (la fase di forecast di
              Cascaldi-Garcia).  Uguale per tutti i trimestri: se uno parte piu'
              tardi degli altri o e' il bordo del backtest, o e' un bug.
  2. FINE     `q` esce dai target in volo quando il suo PIL viene pubblicato.
              L'ultima `as_of` viva e' quindi l'ultimo venerdi' PRIMA del
              rilascio.
  3. STACCO   nella FIGURA la linea si ferma alla fine del trimestre e il
              pallino sta al rilascio, 28 giorni dopo: quel vuoto e' il ritardo
              di pubblicazione, e deve esserci.  Le settimane in mezzo (il
              backcast) restano nel CSV — il taglio e' solo grafico.

I bordi del backtest troncano legittimamente le finestre: un trimestre il cui
anticipo cadrebbe prima di `--start` parte in ritardo, e non e' un difetto.  Il
test lo distingue guardando SOLO i trimestri interi.

Esegui:  python -m src.forecast.test_target_windows
Veloce: e' pura aritmetica di calendario, non stima niente.
"""

from __future__ import annotations

import pandas as pd

from src.forecast.release_calendar import (
    gdp_release_date, horizon_week, load_metadata, quarter_end, targets_in_flight,
    weekly_grid,
)

#: Finestra larga, cosi' i trimestri centrali non toccano i bordi.
_START, _END = "2007-01-01", "2010-12-31"

#: Anticipo atteso dell'inizio linea sull'inizio del trimestre, in settimane.
_LEAD_ATTESO = (12.0, 14.0)

#: Ritardo di pubblicazione del PIL, in giorni (dai metadati, non assunto).
_TARGET = "GDPC1"


def _quarter_start(q: str) -> pd.Timestamp:
    """Inizio del trimestre.  MonthBegin(-3) da una fine mese: vedi horizon_week."""
    return quarter_end(q) + pd.offsets.MonthBegin(-3)


def _finestre(meta) -> tuple[dict, list]:
    grid = weekly_grid(_START, _END)
    vivi: dict[str, list] = {}
    for d in grid:
        for q in targets_in_flight(d, n_ahead=1, target_series=_TARGET, metadata=meta):
            vivi.setdefault(q, []).append(d)
    return vivi, grid


def main() -> None:
    meta = load_metadata()
    delay = int(meta[meta["series_id"] == _TARGET]["publication_delay_days"].iloc[0])
    vivi, grid = _finestre(meta)
    ok = True

    print("\n" + "=" * 78)
    print(f"finestre dei target in volo — {grid[0].date()} .. {grid[-1].date()}, "
          f"delay PIL = {delay}g")
    print("=" * 78)
    print(f"  {'q':<8}{'inizio':>12}{'fine':>12}{'pallino':>12}"
          f"{'anticipo':>10}{'fine-Q':>9}{'stacco':>8}   esito")

    for q in sorted(vivi, key=quarter_end):
        ds = vivi[q]
        primo, ultimo = ds[0], ds[-1]
        rel = gdp_release_date(q, _TARGET, meta)
        qe = quarter_end(q)

        # Bordo del backtest: la finestra sarebbe iniziata prima di --start, o
        # sarebbe finita dopo --end.  Troncatura legittima, non si giudica.
        bordo = (primo <= grid[0]) or (ultimo >= grid[-1])

        lead = (_quarter_start(q) - primo).days / 7.0
        # Ultimo venerdi' disegnato nella figura: il taglio a fine trimestre.
        linea_fine = max([d for d in ds if d <= qe], default=None)
        stacco = (rel - linea_fine).days if linea_fine is not None else float("nan")

        if bordo:
            esito = "bordo (troncata, non giudicata)"
        else:
            c_lead = _LEAD_ATTESO[0] <= lead <= _LEAD_ATTESO[1]
            # (2) l'ultima as_of viva e' l'ultimo venerdi' prima del rilascio
            c_fine = (ultimo < rel) and (ultimo + pd.Timedelta(days=7) >= rel)
            # (3) lo stacco figura-pallino e' il ritardo di pubblicazione
            c_stacco = (delay - 7) <= stacco <= (delay + 7)
            buoni = c_lead and c_fine and c_stacco
            ok &= buoni
            esito = "OK" if buoni else "FAIL " + " ".join(
                n for n, c in (("[anticipo]", c_lead), ("[fine]", c_fine),
                               ("[stacco]", c_stacco)) if not c)

        _lf = str(linea_fine.date())[5:] if linea_fine is not None else "—"
        _st = f"{stacco:>6}g" if linea_fine is not None else f"{'—':>7}"
        print(f"  {q:<8}{str(primo.date()):>12}{str(ultimo.date()):>12}"
              f"{str(rel.date()):>12}{lead:>9.1f}s{_lf:>9}{_st}   {esito}")

    # ── horizon_week: 1 = prima settimana del trimestre ───────────────────────
    print("\n" + "=" * 78)
    print("horizon_week — 1 e' la PRIMA settimana del trimestre target")
    print("=" * 78)
    casi = [("2008-10-03", "2008Q4", 1), ("2008-12-26", "2008Q4", 13),
            ("2009-01-23", "2008Q4", 17), ("2009-01-02", "2009Q1", 1)]
    for d, q, atteso in casi:
        got = horizon_week(d, q)
        buono = abs(got - atteso) <= 1
        ok &= buono
        print(f"  as_of={d}  target={q}  h={got:+3d}  atteso~{atteso:+3d}   "
              f"{'OK' if buono else 'FAIL'}")

    print("\n" + "=" * 78)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 78)


if __name__ == "__main__":
    main()
