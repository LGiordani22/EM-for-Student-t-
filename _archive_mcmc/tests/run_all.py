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

Cosa c'e', e cosa NON e' coperto dal validatore
------------------------------------------------
* ``test_shared``        l'algebra dei conditional, i prior, i pesi, la deflazione per h
* ``test_linear``        blocco lineare vs EM (senza SV): stati + (A,Q,Lambda,R,nu)
* ``test_vol_base``      SV base (KSC): filtro, FFBS, recovery per-fattore, convenzione
                         sigma/sigma^2, e la recovery del lato IDIOSINCRATICO
* ``test_leverage``      Famiglia C (rho): Branch A + Branch B, Omori, griddy, blocco
                         Laplace, skewness, DGP per-fattore, end-to-end per ramo
* ``test_variants``      le celle D1xD2 come restrizioni, i tripwire, Huang-Wand
* ``test_asis``          invarianza dell'interweaving (Famiglia B)
* ``test_diagnostics``   le funzioni diagnostiche + non consumano RNG
* ``test_coupling_qml``  il passo accoppiato del blocco comune: QML, guard, e il
                         comportamento a corr(Q)=0.8 sotto leverage (baseline dietro
                         --slow; la caratterizzazione piena e' materia del validate)

Mappa di copertura (ogni parametro/percorso -> dove, e a che livello)
--------------------------------------------------------------------
Livelli:  C = conditional (formula vs EM)    R = recovery (e2e vs verita')
          S = strutturale (c'e'/non c'e')     M = mixing (ESS/invarianza)

  percorso / parametro       test                          livello   note
  -------------------------  ----------------------------  --------  -----------------------
  f (stati)                  test_linear                   R
  A, Q        [Fam A]        test_shared, test_linear      C + R
  Lambda, R   [Fam A]        test_shared, test_linear      C + R
  w^u, w^eps  [step c]       test_shared                   C
  nu_u,nu_eps [Fam D]        test_shared, test_linear      C + R
  a_j  (HW)   [Fam A+]       test_shared, test_variants    C + S
  h^u         [percorso]     test_vol_base, test_leverage  R
  phi^u,sg2^u [Fam B com.]   test_vol_base, test_asis      R + M
  h^eps       [percorso]     test_vol_base                 R
  phi^eps     [Fam B idio]   test_vol_base                 R
  sg2^eps     [Fam B idio]   test_vol_base                 R debole   <- solo via il path
  rho^u       [Fam C com.]   test_leverage                 R (segno)  <- magnitudine: validatore
  rho^eps     [Fam C idio]   test_leverage                 R (segno)  <- magnitudine: validatore

Tutto e' coperto almeno una volta.  I punti DEBOLI, da approfondire con esperimenti:
  * sg2^eps e il canale per-fattore a bassa volatilita' -> sotto-identificati a T corto;
  * la MAGNITUDINE di rho (attenuata, non solo il segno) -> vive nel validatore, non qui.
Sigma_0 (prior su f_0) e' l'unico parametro NON estratto (tenuto fisso): niente da testare.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

#: L'ordine e' dal piu' veloce al piu' lento: un rosso in test_shared rende inutile
#: aspettare venti minuti per gli altri.
SUITE = ["test_shared", "test_linear", "test_vol_base", "test_diagnostics",
         "test_variants", "test_asis", "test_leverage", "test_coupling_qml"]


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
        print("  (la suite non porta piu' rossi noti: un rosso e' un fallimento vero.")
        print("   Le quantita' non identificate/attenuate sono materia del validatore.)")
    else:
        print("  tutto verde")
    print("=" * 72)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
