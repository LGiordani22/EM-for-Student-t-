"""
src/forecast/benchmarks.py

Univariate NOWCAST BENCHMARKS for the nowcasting pipeline (Fourth brick).

These are the simple, univariate yardsticks the DFM nowcast (nowcast_engine.py)
is compared against.  They see ONLY the history of quarterly GDP growth — no
monthly indicators — so they cannot react to within-quarter news.

  * arma_nowcast(as_of, target, order=(2,2))
        ARMA(p,q) on the quarterly GDP series available at as_of, forecast to
        the target quarter.  Robust fallback on non-convergence.
  * random_walk_nowcast(as_of, target)
        The trivial benchmark: nowcast = last available GDP value.

No look-ahead: both use only GDP published by as_of, obtained via
data_import.get_current_vintage / gdp_available_through (BEA advance ~1 month
after quarter end).  This module does NOT touch nowcast_engine.

Scale note
----------
For comparability with the DFM, the optional z-score uses the SAME training
scale the DFM applies to GDP: the expanding-window mean / std (ddof=1) of the
GDP quarters available at as_of.  Standardising the GDP column on exactly those
quarters is what em_initialization.standardize does inside the DFM, so the
ARMA/RW z and the DFM z are on the same ruler.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.forecast.data_import import get_current_vintage, gdp_available_through


def _quarter_end(target_quarter: str) -> pd.Timestamp:
    """Parse a 'YYYYQn' label into the month-END of that quarter's last month."""
    s = str(target_quarter).upper().replace(" ", "")
    if "Q" not in s:
        raise ValueError(f"target_quarter {target_quarter!r} is not 'YYYYQn'.")
    y_str, q_str = s.split("Q")
    y, q = int(y_str), int(q_str)
    if not 1 <= q <= 4:
        raise ValueError(f"quarter must be 1..4, got {q} in {target_quarter!r}.")
    return pd.Timestamp(y, q * 3, 1) + pd.offsets.MonthEnd(0)


def _quarters_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Number of quarter steps from quarter-end `a` to quarter-end `b` (b>=a)."""
    qa = a.year * 4 + (a.month - 1) // 3
    qb = b.year * 4 + (b.month - 1) // 3
    return qb - qa


def _available_gdp(as_of_date, config_name: str = "small") -> pd.Series:
    """
    Quarterly GDP growth (log-diff x100) available at as_of, truncated to the
    last published quarter (gdp_available_through).  No look-ahead.
    """
    return get_current_vintage("GDPC1", as_of_date, config_name=config_name).dropna()


def _fit_arma_forecast(values: np.ndarray, order: tuple[int, int], steps: int):
    """
    Fit ARMA(p,q) and forecast `steps` ahead, with a robust fallback chain on
    non-convergence / failure.

    Returns (forecast_target, order_used, converged) where forecast_target is
    the predicted value at the LAST of the `steps` horizons (the target quarter).
    """
    from statsmodels.tsa.arima.model import ARIMA  # local import (heavy dep)

    p, q = order
    # Fallback chain: requested order -> (1,1) -> (1,0) -> mean.
    candidates = [order, (1, 1), (1, 0)]
    seen = []
    for (pp, qq) in candidates:
        if (pp, qq) in seen:
            continue
        seen.append((pp, qq))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")            # silence convergence noise
                res = ARIMA(values, order=(pp, 0, qq), trend="c").fit()
            fc = np.asarray(res.forecast(steps))
            if fc.size == steps and np.all(np.isfinite(fc)):
                conv = bool(getattr(res, "mle_retvals", {}).get("converged", True))
                return float(fc[-1]), (pp, qq), conv
        except Exception:
            continue

    # Last resort: random-walk-in-mean (the unconditional mean of the series).
    return float(np.mean(values)), (0, 0), False


def arma_nowcast(as_of_date, target_quarter: str, config_name: str = "small", order: tuple[int, int] = (2, 2)) -> dict:
    """
    ARMA(p,q) nowcast of GDP growth for `target_quarter`, using only GDP
    published by `as_of_date`.

    Because it sees only the quarterly GDP history (never the monthly
    indicators), for a FIXED target quarter the ARMA nowcast changes little
    across different as_of dates within the same quarter: new monthly data
    arriving mid-quarter is invisible to it, and the GDP history it conditions
    on only changes when a new quarter is published.

    Parameters
    ----------
    as_of_date : str | datetime | (year, month) tuple
    target_quarter : str  ('YYYYQn')
    order : (p, q)   ARMA order, default (2, 2).

    Returns
    -------
    dict with keys: nowcast_livello, nowcast_z, target_quarter, as_of,
        metodo, last_gdp_used, n_obs_used, order_used, converged, fell_back.
    """
    target_qe = _quarter_end(target_quarter)
    gdp = _available_gdp(as_of_date, config_name=config_name)
    last_qe = gdp.index[-1]

    mean_gdp = float(gdp.mean())
    std_gdp = float(gdp.std(ddof=1))

    if target_qe <= last_qe:
        # Target already published in this vintage's available set: the honest
        # "nowcast" is that published value (no forecasting needed).
        level = float(gdp.loc[target_qe])
        order_used, converged, fell_back = (0, 0), True, False
    else:
        steps = _quarters_between(last_qe, target_qe)
        level, order_used, converged = _fit_arma_forecast(gdp.to_numpy(), order, steps)
        fell_back = order_used != tuple(order)

    return {
        "nowcast_livello": level,
        "nowcast_z": (level - mean_gdp) / std_gdp if std_gdp else float("nan"),
        "target_quarter": target_quarter,
        "as_of": str(as_of_date),
        "metodo": f"arma{order[0]}{order[1]}",
        "last_gdp_used": last_qe,
        "n_obs_used": int(gdp.size),
        "order_used": order_used,
        "converged": converged,
        "fell_back": fell_back,
    }


def random_walk_nowcast(as_of_date, target_quarter: str, config_name: str = "small") -> dict:
    """
    Random-walk benchmark: nowcast = last GDP value available at as_of.  The
    simplest possible yardstick; sees only the most recent published quarter.
    """
    target_qe = _quarter_end(target_quarter)
    gdp = _available_gdp(as_of_date, config_name=config_name)
    last_qe = gdp.index[-1]
    mean_gdp = float(gdp.mean())
    std_gdp = float(gdp.std(ddof=1))
    level = float(gdp.loc[last_qe])
    return {
        "nowcast_livello": level,
        "nowcast_z": (level - mean_gdp) / std_gdp if std_gdp else float("nan"),
        "target_quarter": target_quarter,
        "as_of": str(as_of_date),
        "metodo": "random_walk",
        "last_gdp_used": last_qe,
        "n_obs_used": int(gdp.size),
    }


__all__ = ["arma_nowcast", "random_walk_nowcast"]


# ─── Smoke tests ──────────────────────────────────────────────────────────────
# Run from the project root with:  python -m src.forecast.benchmarks
# Local: reads only the current processed GDP series with real-time timing.

import os  # noqa: E402  (only needed by the test block)


def _hr(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def _realized_gdp(target_qe: pd.Timestamp) -> float:
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cur = pd.read_csv(os.path.join(proj_root, "data", "processed", "dataset_small.csv"),
                      index_col=0)
    cur.index = pd.to_datetime(cur.index)
    return float(cur.loc[target_qe, "GDPC1"]) if target_qe in cur.index else float("nan")


def _run_date(as_of: str, target_quarter: str, scenario: str) -> None:
    _hr(f"{scenario}:  as_of = {as_of}   target = {target_quarter}")
    target_qe = _quarter_end(target_quarter)

    a = arma_nowcast(as_of, target_quarter, order=(2, 2))
    rw = random_walk_nowcast(as_of, target_quarter)
    realized = _realized_gdp(target_qe)

    # realised z on the same expanding-window training scale as the benchmarks
    gdp = _available_gdp(as_of)
    mean_gdp, std_gdp = float(gdp.mean()), float(gdp.std(ddof=1))
    realized_z = (realized - mean_gdp) / std_gdp if std_gdp else float("nan")

    last = a["last_gdp_used"]
    print(f"  no look-ahead: ARMA conditioned on GDP through {last.date()} "
          f"(Q{(last.month - 1)//3 + 1} {last.year}, {a['n_obs_used']} quarters); "
          f"gdp_available_through = {gdp_available_through(as_of).date()}")
    print(f"  ARMA(2,2): order_used = {a['order_used']}, converged = {a['converged']}, "
          f"fell_back = {a['fell_back']}")
    print()
    print(f"    {'method':<14}{'livello':>12}{'z':>10}{'err(liv)':>12}{'err(z)':>10}")
    print(f"    {'arma22':<14}{a['nowcast_livello']:>12.4f}{a['nowcast_z']:>10.4f}"
          f"{a['nowcast_livello'] - realized:>12.4f}{a['nowcast_z'] - realized_z:>10.4f}")
    print(f"    {'random_walk':<14}{rw['nowcast_livello']:>12.4f}{rw['nowcast_z']:>10.4f}"
          f"{rw['nowcast_livello'] - realized:>12.4f}{rw['nowcast_z'] - realized_z:>10.4f}")
    print(f"    {'realised':<14}{realized:>12.4f}{realized_z:>10.4f}"
          f"{0.0:>12.4f}{0.0:>10.4f}")

    if scenario == "CRISIS":
        print("\n  note: the ARMA is BLIND to the crisis -> it extrapolates the calm "
              "pre-crisis\n        GDP history and predicts a mild value, while the "
              "realised Q4-2008 collapses.")


if __name__ == "__main__":
    _hr("benchmarks.py smoke tests")
    _run_date("2008-11-15", "2008Q4", "CRISIS")
    _run_date("2015-05-15", "2015Q2", "CALM")
    _hr("Done.")
