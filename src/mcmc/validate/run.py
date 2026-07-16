"""
mcmc/validate/run.py
====================

**L'unico punto d'ingresso.**

    python -m mcmc.validate.run --quick
    python -m mcmc.validate.run --full
    python -m mcmc.validate.run --full --coverage 12

Che cosa sta validando — la teoria in una pagina
------------------------------------------------
Il modello è un DFM a frequenza mista con code Student-t, **volatilità stocastica
per-fattore** e **leverage**.  Due blocchi di shock, ed entrambi hanno *tutto*:

* **fattori** (``u_t``): ``A``, ``Q``, e per ogni canale ``k`` una log-vol AR(1)
  ``(phi^u_k, sigma^2_{eta,k})``, un leverage ``rho^u_k``, un path ``h^u_k``; più i pesi
  ``w^u`` e i gradi di libertà ``nu_u``;
* **osservazioni** (``eps_t``): ``Lambda``, ``R``, e — questa è la scoperta della mappa —
  **una SV per serie** ``(phi^eps_i, sigma^2_{eps,i}, h^eps_i)`` **con il suo leverage**
  ``rho^eps_i``; più ``w^eps`` e ``nu_eps``, distinti dai loro omologhi comuni.

Il blocco osservazioni è quello che **proietta il PIL**, ed era completamente privo di
recovery test.  Chiuderlo è la prima ragione per cui questo validatore esiste.

**Spec II**: la volatilità sta *fuori* dalla radice di ``Q``
(``Var(u) = sqrt(H) Q sqrt(H)/w``), quindi ``h^u_k`` è la volatilità **del fattore k** —
un oggetto che si può leggere — e non quella di una direzione ortogonalizzata che cambia
significato ogni volta che ``Q`` si muove.

**Option A**: il leverage aggancia l'innovazione di volatilità del canale ``k`` al **suo**
shock grezzo pienamente sbiancato ``z^u_k``, con ``r`` correlazioni **scalari**.

**Branch A vs B** — due *timing*, cioè due modelli, non due approssimazioni della stessa
cosa.  **A** (contemporaneo) ha un target di Metropolis **esatto**: nessuna mistura,
nessuna linearizzazione, ``Q`` piena trattata esattamente — è la **controparte esatta**,
il metro contro cui si misura B.  **B** (laggato) linearizza con la mistura di Omori-10 e
in cambio estrae il path con un **FFBS diretto** (accettazione 1).  B è il default; A è il
controllo.

Le tre regole
-------------
1. **Quattro esiti, non pass/fail** (``verdict.py``).  Rosso = *sampler rotto*, e basta.
2. **Nessuna stima senza ESS e R-hat.**
3. **La copertura si misura su N repliche**, non su un dataset.

Se un check ti sembra "quasi passare" e sei tentato di allargarne la tolleranza:
**fermati**.  O il verdetto giusto è *non identificato* / *bias noto* — e allora va detto
— oppure il sampler è rotto.  Allargare la soglia è il modo in cui un bug di
identificazione torna a nascondersi: è già successo, con una magnitudine attenuata su
``rho^u_0`` e la suite tutta verde.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.validate import dgp as D                                   # noqa: E402
from mcmc.validate import report                                     # noqa: E402
from mcmc.validate.checks import coupling, leverage, linear, variants, volatility   # noqa: E402


def main() -> int:
    # La console di Windows e' cp1252: senza questo, un carattere non-ASCII nella tabella
    # fa esplodere il validatore invece di stampare un verdetto.  Meglio riconfigurare lo
    # stream che mutilare il testo.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="CI / uso quotidiano")
    g.add_argument("--full", action="store_true",
                   help="catene lunghe — l'UNICA che ha diritto di pronunciarsi su rho")
    p.add_argument("--coverage", type=int, default=0, metavar="N",
                   help="N repliche: distingue la sfortuna dal bias")
    p.add_argument("--T", type=int, default=0, metavar="N",
                   help="override della lunghezza del pannello: un solo sample, T piu' "
                        "lungo, per vedere se AL LIMITE si recupera (l'attenuazione di rho, "
                        "se e' errors-in-variables, deve ridursi al crescere di T)")
    p.add_argument("--n-iter", type=int, default=0, metavar="N",
                   help="override del numero di iterazioni (piu' iter = piu' ESS: utile a "
                        "T grande perche' il cancello di convergenza non sospenda i verdetti)")
    p.add_argument("--burn-in", type=int, default=0, metavar="N", help="override del burn-in")
    p.add_argument("--report", type=str, default="docs/VALIDATION_REPORT.md")
    p.add_argument("--only", type=str, default="",
                   help="csv fra: linear,volatility,leverage,coupling,variants")
    a = p.parse_args()

    mode = D.FULL if a.full else D.QUICK
    if a.T or a.n_iter or a.burn_in:
        # stesso NOME (resta probante), con gli override richiesti — un solo sample.
        mode = D.Mode(mode.name, T=a.T or mode.T, n_iter=a.n_iter or mode.n_iter,
                      burn_in=a.burn_in or mode.burn_in, chains=mode.chains)
    only = set(a.only.split(",")) if a.only else None
    t0 = time.time()

    print(f"\n  validatore — modalita' {mode.name} "
          f"(T={mode.T}, n_iter={mode.n_iter}, catene={mode.chains})"
          + (f" + coverage {a.coverage}" if a.coverage else ""))
    if mode.name == "quick":
        print("  ⚠  --quick non e' probante su rho, sigma^2_eta e h: i loro OK portano il caveat.")

    out = _SRC.parent / a.report
    out.parent.mkdir(parents=True, exist_ok=True)

    verdicts = []
    modules = {"linear": linear, "volatility": volatility,
               "leverage": leverage, "coupling": coupling, "variants": variants}
    for name, mod in modules.items():
        if only and name not in only:
            continue
        print(f"\n  ── {name} " + "─" * (60 - len(name)), flush=True)
        verdicts += mod.run(mode, coverage_reps=a.coverage)
        # Salvataggio INCREMENTALE: una run in modalita' --full dura ore, e un'interruzione
        # a tre quarti non deve costare tutto.  Dopo ogni modulo il report su disco e' gia'
        # valido — parziale, ma vero.
        report.markdown(report.fill_untested(list(verdicts), mode.name), mode.name, out)
        print(f"  [salvato] {name} -> {out.name}", flush=True)

    verdicts = report.fill_untested(verdicts, mode.name)
    rc = report.console(verdicts, mode.name)
    report.markdown(verdicts, mode.name, out)
    print(f"  report: {out}    wall-clock: {time.time() - t0:.0f}s\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
