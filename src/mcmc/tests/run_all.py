"""
mcmc/tests/run_all.py
=====================

**Un solo comando per tutti i test:**  ``python -m mcmc.tests.run_all``

I test rispondono a una domanda **ingegneristica**: *il codice fa quello che dice la
matematica?*  Girano in **secondi/minuti**, e sono il gate che si usa dopo ogni modifica.

La domanda **scientifica** — *il modello recupera i parametri?* — è un'altra cosa, vive in
``mcmc/validate/`` e costa **ore**.  Le due non sono intercambiabili: un campionatore può
recuperare bene i parametri pur avendo un conditional sbagliato (errori che si compensano),
e il validatore non lo vedrebbe mai — ``test_shared`` sì, perché verifica che la precisione
del conditional di ``A`` sia *letteralmente* ``P00 ⊗ Q^{-1}``.

Cosa c'è, e cosa NON è coperto dal validatore
---------------------------------------------
* ``test_shared``           l'algebra dei conditional, i prior, i pesi, la deflazione per h
* ``test_passo1``           blocco lineare vs EM (senza SV)
* ``test_passo2``           mistura KSC, bit-identita' del path senza SV
* ``test_passo3``           Branch A: kernel di Family C, nesting a rho=0, end-to-end
* ``test_passo4``           Branch B: costanti di Omori, FFBS a rho=0 bit-identico, griddy
* ``test_leverage_common``  il gate end-to-end, UNA funzione parametrizzata per ramo
* ``test_asis``             invarianza dell'interweaving
* ``test_variants``         le celle D1xD2, i tripwire di P1, Huang-Wand
* ``test_diagnostics``      le funzioni diagnostiche + non consumano RNG
* ``test_spec2_recovery``   recovery Spec II senza leverage
* ``test_branchA_qml``      invarianza del kernel Laplace; costanti QML; instabilita' congelata
* ``test_sigma_eta``        la convenzione sigma / sigma^2 (la mina) + recovery di sigma^2_eta
* ``test_perfactor_leverage``  ⚠ 3 ROSSI NOTI: asseriscono proprieta' del rho del canale
                            DEBOLE, che NON e' identificato. Da ritirare: il validatore lo
                            copre meglio (magnitudine, ESS, R-hat, copertura).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

#: L'ordine e' dal piu' veloce al piu' lento: un rosso in test_shared rende inutile
#: aspettare venti minuti per gli altri.
SUITE = ["test_shared", "test_passo1", "test_passo2", "test_diagnostics", "test_variants",
         "test_asis", "test_passo3", "test_passo4", "test_spec2_recovery",
         "test_branchA_qml", "test_sigma_eta", "test_perfactor_leverage"]


def main() -> int:
    fails = []
    for name in SUITE:
        print(f"\n{'=' * 72}\n  {name}\n{'=' * 72}", flush=True)
        rc = subprocess.run([sys.executable, str(HERE / f"{name}.py")]).returncode
        if rc != 0:
            fails.append(name)
    print(f"\n{'=' * 72}")
    if fails:
        print(f"  ROSSI: {', '.join(fails)}")
        print("  (test_perfactor_leverage e test_sigma_eta hanno rossi NOTI e documentati:")
        print("   asseriscono quantita' non identificate / un bias reale del prior.)")
    else:
        print("  tutto verde")
    print("=" * 72)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
