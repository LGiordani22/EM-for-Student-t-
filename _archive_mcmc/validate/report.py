"""
mcmc/validate/report.py
=======================

**La tabella unica.**  È il deliverable del validatore: una riga per parametro, per
blocco, per ramo — e la domanda a cui la tesi deve rispondere («questo parametro lo
recupero o no?») trova qui la sua risposta, senza doverla ricostruire da otto file di
test.

Due invarianti della tabella, entrambe nate da errori reali:

* **ogni riga porta la modalità in cui è stata prodotta.**  Un ``OK`` su ``rho`` ottenuto
  in ``--quick`` non vale, e deve dirlo da solo: l'ESS di ``rho`` è 40–122 su 3400 draw
  *anche* col prior half-Normal, quindi una catena corta non misura, indovina.
* **ogni stima porta ESS e R-hat.**  Un ``rho_hat`` senza ESS non è una misura.
"""

from __future__ import annotations

import pathlib

from mcmc.validate.spec import BY_KEY, PARAMS
from mcmc.validate.verdict import Outcome, Verdict


def _fmt(v, spec="{:+.3f}"):
    return "—" if v is None else spec.format(v)


def fill_untested(verdicts: list[Verdict], mode: str) -> list[Verdict]:
    """Ogni parametro dell'inventario che nessun check ha toccato entra in tabella come
    ``UNTESTED``.  **Un buco non puo' essere un'assenza silenziosa**: deve essere una
    riga vuota *dichiarata*."""
    seen = {v.param.split("_")[0] if False else v.detail.get("spec_key") for v in verdicts}
    out = list(verdicts)
    for p in PARAMS:
        if p.key in seen:
            continue
        if p.family == "-" and p.where == "NON CAMPIONATO":
            out.append(Verdict(p.tex, p.block, "-", Outcome.UNTESTED,
                               "non e' un parametro stimato: fissato per identificazione",
                               mode=mode, detail={"spec_key": p.key, "fixed": True}))
        else:
            out.append(Verdict(p.tex, p.block, "-", Outcome.UNTESTED,
                               "nessun check lo copre", mode=mode,
                               detail={"spec_key": p.key}))
    return out


def console(verdicts: list[Verdict], mode: str) -> int:
    order = {Outcome.BROKEN: 0, Outcome.RECOVERED_BIASED: 1, Outcome.NOT_IDENTIFIED: 2,
             Outcome.UNTESTED: 3, Outcome.RECOVERED: 4}
    rows = sorted(verdicts, key=lambda v: (v.block != "fattori", order[v.outcome], v.param))

    w = 132
    print("\n" + "=" * w)
    print(f"  TABELLA DEI VERDETTI — modalita': {mode}")
    print("=" * w)
    print(f"  {'parametro':<30}{'blocco':<14}{'br':<4}{'verdetto':<28}"
          f"{'vero':>8}{'stima':>9}{'ESS':>7}{'R-hat':>7}{'cov':>7}")
    print("  " + "-" * (w - 4))
    cur = None
    for v in rows:
        if v.block != cur:
            cur = v.block
            print()
        cov = "—" if v.coverage is None else f"{v.coverage:.0%}"
        name = v.param if len(v.param) <= 29 else v.param[:28] + "…"
        print(f"  {name:<30}{v.block:<14}{v.branch:<4}"
              f"{v.outcome.mark + ' ' + v.outcome.value:<28}"
              f"{_fmt(v.truth, '{:+.3f}'):>8}{_fmt(v.estimate, '{:+.3f}'):>9}"
              f"{_fmt(v.ess, '{:.0f}'):>7}{_fmt(v.r_hat, '{:.3f}'):>7}{cov:>7}")

    n_red = sum(v.outcome.is_red for v in verdicts)
    print("\n  " + "-" * (w - 4))
    for o in Outcome:
        n = sum(v.outcome is o for v in verdicts)
        if n:
            print(f"  {o.mark:<6} {o.value:<26} {n}")

    # ── Sintesi: QUALI parametri NON si recuperano pienamente (il punto della tabella) ──
    problem_order = {Outcome.BROKEN: 0, Outcome.RECOVERED_BIASED: 1,
                     Outcome.NOT_IDENTIFIED: 2, Outcome.UNTESTED: 3}
    problems = [v for v in verdicts if v.outcome in problem_order]
    if problems:
        print("\n  Parametri che NON si recuperano pienamente (da approfondire):")
        for v in sorted(problems, key=lambda x: (problem_order[x.outcome], x.block, x.param)):
            print(f"    {v.outcome.mark:<6} {v.param:<26} [{v.block[:4]}/{v.branch}]  {v.outcome.value}")
    print("=" * w)
    if n_red:
        print(f"  🔴 {n_red} ROTTI — il campionatore sbaglia.  Da riparare, non annotare.")
    else:
        print("  Nessun ROTTO: il campionatore fa quello che dice di fare.")
        print("  (I 'non identificato' NON sono fallimenti: sono il limite del dato,")
        print("   affermato invece che nascosto sotto una tolleranza larga.)")
    print("=" * w + "\n")
    return 1 if n_red else 0


def markdown(verdicts: list[Verdict], mode: str, path: pathlib.Path) -> None:
    L = [f"# Validation report — modalità `{mode}`", "",
         "Generato da `python -m mcmc.validate.run`. Una riga per parametro; il verdetto è",
         "uno dei quattro di `validate/verdict.py`. **Rosso = sampler rotto, e basta**: un",
         "parametro non identificato è un limite del dato, non un difetto del codice.", ""]

    for block in ("fattori", "osservazioni"):
        rows = [v for v in verdicts if v.block == block]
        if not rows:
            continue
        L += [f"## Blocco {block}", "",
              "| parametro | ramo | verdetto | vero | stima | ESS | R-hat | copertura | ragione |",
              "|---|---|---|---|---|---|---|---|---|"]
        order = {Outcome.BROKEN: 0, Outcome.RECOVERED_BIASED: 1, Outcome.NOT_IDENTIFIED: 2,
                 Outcome.UNTESTED: 3, Outcome.RECOVERED: 4}
        for v in sorted(rows, key=lambda x: (order[x.outcome], x.param)):
            cov = "—" if v.coverage is None else f"{v.coverage:.0%} ({v.n_reps} rep.)"
            L.append(f"| `{v.param}` | {v.branch} | **{v.outcome.value}** | "
                     f"{_fmt(v.truth)} | {_fmt(v.estimate)} | {_fmt(v.ess, '{:.0f}')} | "
                     f"{_fmt(v.r_hat, '{:.3f}')} | {cov} | {v.reason} |")
        L.append("")

    L += ["## Come leggere i verdetti", "",
          "- **recuperato** — confrontato con il vero, lo recupera.",
          "- **recuperato con bias noto** — lo recupera, ma con una distorsione *misurata*. "
          "È un risultato da scrivere in tesi, non un fallimento.",
          "- **non identificato** — i dati non lo pinzano. Il check afferma il limite e passa.",
          "- **mai testato** — riga vuota **dichiarata**.",
          "- **ROTTO** — l'unico rosso: il campionatore sbaglia.", ""]
    path.write_text("\n".join(L), encoding="utf-8")
