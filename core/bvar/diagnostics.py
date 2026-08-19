"""
Diagnostica di catena per il BVAR — l'Effective Sample Size.

PERCHE' STA QUI E NON IN `mcmc/`.  `ess` nasce nel lavoro MCMC del DFM
(`mcmc/diagnostics.py`), e `tests/test_mixing.py` la importava da li'.  Quando
il binario legacy `small`/`big` e' stato cancellato, pero', quel modulo ha
smesso di essere importabile — fa `from mcmc.gibbs import fit_dfm_mcmc` al
livello del modulo, cioe' trascina l'intero campionatore Gibbs dietro a una
funzione di venti righe che non ne usa niente — e con lui e' diventata
irrieseguibile la misura del mescolamento, che e' materiale di tesi.

La copia sta qui perche' il pacchetto `mcmc/` e' archiviato: `core/bvar/` non
deve avere dipendenze verso un archivio.  E' venti righe di numpy senza stato,
quindi duplicarle costa meno che tenere vivo un import fragile.

L'ESS misura *a quante estrazioni indipendenti equivale una catena*: una catena
MCMC non produce estrazioni indipendenti ma una passeggiata, e mille estrazioni
correlate possono valerne venti.  Un ESS basso non distorce — la catena resta
valida — ma rende la stima imprecisa: l'errore Monte Carlo e' ``sd/sqrt(ESS)``,
non ``sd/sqrt(S)``.  E' la quantita' su cui poggia la sezione *Il mescolamento
degli iperparametri* del README, e la legge 1/d che vi si misura.
"""

from __future__ import annotations

import numpy as np


def ess(chains: np.ndarray) -> float:
    r"""
    Effective sample size for a scalar, pooled across chains via the
    autocorrelation-based estimator (Stan/Geyer initial-positive-sequence).

    ``chains`` is ``(n_chains, n_draws)``.  Returns the total ESS.
    """
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    if n < 4:
        return float("nan")

    # Per-chain autocovariance via FFT, averaged across chains (variogram form).
    means = chains.mean(axis=1, keepdims=True)
    centered = chains - means
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    acov = np.zeros(n)
    for c in range(m):
        f = np.fft.rfft(centered[c], n=nfft)
        ac = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
        acov += ac / n
    acov /= m
    if acov[0] <= 0:
        return float("nan")
    rho = acov / acov[0]

    # Geyer initial positive sequence: sum paired autocorrelations while positive.
    tau = 1.0
    t = 1
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair
        t += 2
    return float(m * n / tau)
