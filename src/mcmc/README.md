# `src/mcmc/` — dove sta cosa

Tre mestieri, tre posti. Se cerchi qualcosa, la domanda che ti stai facendo ti dice
dove guardare.

| La tua domanda | Dove |
|---|---|
| «cosa campiona questo blocco?» | i **motori**, qui sotto — 12 file, tutti nella radice |
| «il codice fa quello che dice la matematica?» | **`tests/`** → `python -m mcmc.tests.run_all` (minuti) |
| «il modello **recupera i parametri**?» | **`validate/`** → `python -m mcmc.validate.run --full` (ore) |

Le ultime due **non sono intercambiabili**. Un campionatore può recuperare bene i
parametri pur avendo un conditional sbagliato (errori che si compensano): il validatore
non lo vedrebbe mai, `tests/test_shared` sì. E viceversa: la suite può essere tutta verde
mentre un parametro è sbagliato del 35% — **è successo**, con `rho`, perché i test
asserivano il *segno* e mai la *magnitudine*.

---

## I motori

Il modello è un DFM a frequenza mista con code Student-t, **volatilità stocastica
per-fattore** (Spec II) e **leverage** (Option A). Lo sweep di Gibbs ha quattro blocchi:

```
(a) stati        f_t              FFBS di Kalman
(b) volatilita'  h^u, h^eps       KSC (senza leverage) / Omori+FFBS (B) / Metropolis (A)
                 + phi, sigma^2_eta, rho
(c) pesi         w^u, w^eps       Gamma  (le code Student-t)
(d) parametri    A, Q             per-fattore (con SV) / MNIW (senza)
                 Lambda, R        NIG
                 nu_u, nu_eps     griddy-Gibbs
```

| File | Blocco | Cosa fa |
|---|---|---|
| **`gibbs.py`** | orchestratore | `fit_dfm_mcmc` — **l'unico punto d'ingresso**. `VARIANT_FLAGS` mappa le celle della griglia D1×D2 come *restrizioni* di questo sweep |
| `sample_states.py` | (a) | FFBS: estrae i fattori |
| `sample_vol.py` | (b) | volatilità **senza** leverage: mistura KSC-7 + FFBS. Contiene anche `logsq_corr_matrix`, `_inv_sqrt_spd`, e il passo accoppiato QML |
| `sample_leverage.py` | (b)+B+C | **Branch A** (timing contemporaneo): target di Metropolis **esatto** — nessuna mistura, nessuna linearizzazione. È la *controparte esatta*, il metro contro cui si misura B |
| `sample_leverage_lagged.py` | (b)+B+C | **Branch B** (timing laggato, il **default**): mistura di Omori-10 ⇒ il path è un **draw FFBS diretto** |
| `sample_params.py` | (d) | `Lambda`, `R` (NIG); `nu` (griddy); `A,Q` MNIW (solo senza SV) |
| `shared.py` | (c)+(d) | pesi (Gamma), `A,Q` per-fattore, prior Huang–Wand |
| `sample_asis.py` | wrapper | ASIS: riparametrizza `(phi, sigma^2_eta)` fra centrata e non-centrata. Non è un blocco: **si aggancia alla Family B** ovunque si campioni una volatilità |
| `constants.py` | — | misture KSC-7 e Omori-10, costanti QML |
| `simulate_sv.py` | — | il DGP. ⚠ **prende `sigma` in ingresso e restituisce `sigma^2`**: il sampler parla in varianza |
| `diagnostics.py` | — | ESS, split-R̂, le diagnostiche di P1/P4/P5, `recommend_coupling` |
| `bench_p6_rho.py` | — | il benchmark del mixing di `rho` (Branch B) |

**Rami morti / semi.** `sample_vol.sample_volatility_block` (restrizione scalare
`H = h·I`) **non è raggiungibile da `fit_dfm_mcmc`**: esiste solo come *seme* con cui
`tests/test_shared` verifica che il blocco per-fattore, a `r = 1`, riproduca bit-per-bit
il vecchio blocco scalare. Non è codice di produzione.

---

## Le due configurazioni che contano

**Spec II** — la volatilità sta *fuori* dalla radice di `Q`
(`Var(u) = √H · Q · √H / w`), quindi `h^u_k` è la volatilità **del fattore k**, un oggetto
che si può leggere — non quella di una direzione ortogonalizzata che cambia significato
ogni volta che `Q` si muove.

**Option A** — il leverage aggancia l'innovazione di volatilità del canale `k` al **suo**
shock grezzo pienamente sbiancato, con `r` correlazioni **scalari** `rho_k`.

**Branch B è il default; Branch A è il controllo.** Non sono due approssimazioni della
stessa cosa: sono due *timing*, cioè due modelli.

---

## Dove leggere il resto

* `docs/MCMC_ENGINES_AND_TESTS.md` — la mappa completa: motori, inventario dei parametri
  sui due blocchi, chi testa cosa, i buchi.
* `docs/VALIDATION_REPORT.md` — la tabella dei verdetti: **quale parametro è recuperato e
  quale no**.
