# `src/mcmc/` — i motori del campionatore

Tre mestieri, tre posti. La domanda che ti stai facendo ti dice dove guardare.

| La tua domanda | Dove |
|---|---|
| «cosa campiona questo blocco?» | i **motori**, questo file — 12 file nella radice di `mcmc/` |
| «il codice fa quello che dice la matematica?» | **`tests/`** → `python -m mcmc.tests.run_all` (minuti); vedi `tests/README.md` |
| «il modello **recupera i parametri**?» | **`validate/`** → `python -m mcmc.validate.run --full` (ore) |

Le ultime due **non sono intercambiabili**. Un campionatore può recuperare bene i
parametri pur avendo un conditional sbagliato (errori che si compensano): il validatore
non lo vedrebbe mai, `tests/test_shared` sì. E viceversa: la suite può essere tutta verde
mentre un parametro è sbagliato del 35% — **è successo**, con `ρ`, perché i test
asserivano il *segno* e mai la *magnitudine*.

Le equazioni citate qui (`eq:...`) stanno in `docs/EM_for_student_t.tex`.

---

## 0. Si parte da **Y**

Tutto quello che segue è condizionato a un solo dato: **`Y`**, il pannello osservato a
frequenza mista, `(T, M)` — `T` mesi, `M` serie (mensili + il PIL trimestrale posato a
fine trimestre, con i buchi ragged di fine campione). Il campionatore non vede altro.

L'unico altro ingresso è il **warm start**: la stima EM della *stessa* config, letta da
`data/processed/<config>/fit_dfm_result.npz` via `load_warm_init` (dà `theta` iniziale +
`freq_list`, `block_map`, `ordered_cols`, `r`). Serve solo a partire da un punto sensato;
non entra nel modello. `fit_dfm_mcmc` non scrive nulla su disco — i draw tornano al
chiamante, che decide se salvarli.

Da `Y` (più il warm start) il Gibbs estrae, ripetendo lo sweep, **tutto il resto**:
i percorsi latenti e i parametri elencati nella §2.

---

## 1. Il modello — cella master **D1b × D2b**, **Spec II** + **Option A**

Un DFM a frequenza mista con code Student-t, volatilità stocastica per-fattore e
leverage. Per `t = 1..T`:

```
(oss.)    y_t = Λ f_t + ε_t                                   [eq:obs, eq:idio]
(stato)   f_t = A f_{t-1} + u_t                               [eq:state]
```

**D1b — code Student-t come mistura di scala gaussiana** (un peso Gamma per periodo,
media 1; heavy tails = spike isolati, *senza* memoria):
```
ε_t | w^ε_t ~ N(0, H^ε_t R / w^ε_t),   w^ε_t ~ Gamma(ν_ε/2, ν_ε/2),   R = diag(r_1..r_M)
u_t | w^u_t ~ N(0, √H^u_t · Q · √H^u_t / w^u_t)               [eq:sv-baseline-*]
```

**D2b — volatilità stocastica su entrambi i lati** (`r` comuni + `M` idiosincratiche;
persistente, *con* memoria = volatility clustering), AR(1) su `log h`, livello `μ ≡ 0`:
```
log h_t = φ log h_{t-1} + η_t,   η_t ~ N(0, σ_η²)             [eq:sv-logvol-u/eps]
H^u_t = diag(h^u_{1,t}..h^u_{r,t}),   H^ε_t = diag(h^ε_{1,t}..h^ε_{M,t})
```

**Spec II** — `√H^u` sta **fuori** dal sandwich di `Q^{1/2}`, quindi
`Var(u_t) = √H^u_t · Q · √H^u_t / w^u_t` e `h^u_k` è la volatilità **del fattore k**
(la sua diagonale legge `h^u_k q_kk`), non quella di una direzione ortogonalizzata che
cambia significato quando `Q` si muove [eq:vol-outside vs eq:vol-inside]. Prezzo: le `r`
volatilità comuni non si separano e vanno estratte **insieme**.

**Option A** — il leverage aggancia l'innovazione di volatilità del canale `k` al **suo**
shock grezzo pienamente sbiancato `z^u_k`, con `r` correlazioni **scalari** `ρ_k`:
```
z^u_t = √w^u_t · Q^{-1/2} (√H^u_t)^{-1} u_t                   (sbiancamento)
η^u_{k,t} | z^u_{k,t} ~ N(ρ_k σ_{u,k} z^u_{k,t}, σ_{u,k}²(1-ρ_k²))   [eq:lev-cond-common]
```
Il **timing** sceglie *quale* transizione corregge: **Branch A** (contemporaneo, `z_t↔η_t`)
oppure **Branch B** (laggato, `z_t↔η_{t+1}`; è il **default**). Non sono due
approssimazioni della stessa cosa: sono due *modelli*, e Branch A fa da controllo esatto.

**Nesting.** Ogni cella della griglia D1×D2 è una *restrizione* di questo sistema:
`sv=False ⇒ h≡1`; `tails='gaussian' (D1a) ⇒ w≡1`; `leverage=False ⇒ ρ=0` (drift nullo,
varianza piena). `VARIANT_FLAGS` in `gibbs.py` mappa ogni cella ai suoi flag.

---

## 2. Cosa estrae il campionatore

Le incognite si dividono in due tipi (`subsec:gibbs-quantities`). **Nel codice** vivono
nel dizionario `res["draws"]` restituito da `fit_dfm_mcmc`.

### 2a. Percorsi latenti (crescono con `T`, ridisegnati a ogni sweep)

| Simbolo | Cosa | Forma | Step | Chi lo estrae | eq |
|:---|:---|:---|:---:|:---|:---|
| `f_{1:T}` (`F`) | percorso dei fattori (stato aumentato companion `f̃`) | `(T, r)` | **(a)** | `sample_states.ffbs_sample_states` | eq:state, eq:obs |
| `h^u_{1:T}` (`h_u`) | `r` volatilità comuni **per-fattore** (`log h^u`) | `(T, r)` | **(b)** | `sample_vol` / `sample_leverage[_lagged]` | eq:sv-logvol-u |
| `h^ε_{1:T}` (`h_eps`) | `M` volatilità idiosincratiche | `(T, M)` | **(b)** | `sample_vol._sample_idio_vol` | eq:sv-logvol-eps |
| `w^u_{1:T}`, `w^ε_{1:T}` | pesi Student-t (1 comune + 1 idio per periodo) | `(T,)`, `(T,)` | **(c)** | `shared.draw_weights` | eq:post-w-u, eq:post-w-eps |

### 2b. Parametri (numero fisso, in `θ` / `theta_mean`)

| Simbolo | Cosa | Forma | Famiglia | Step | Chi lo estrae | eq |
|:---|:---|:---|:---:|:---:|:---|:---|
| `A` | matrice VAR(1) di transizione | `(r, r)` | **A** | (d) | `shared.draw_A_Q_perfactor` / `sample_params.draw_A_Q_block` | eq:state, eq:param-AQ |
| `Q` | scala degli shock di fattore | `(r, r)` | **A** | (d) | idem | eq:state, eq:Q-update |
| `Λ` (`Lambda`) | loadings fattori→serie | `(M, r)` | **A** | (d) | `sample_params.draw_Lambda_R_block` | eq:obs, eq:Lambda-update |
| `R` | scale idiosincratiche (diag.) | `(M,)` | **A** | (d) | idem | eq:idio, eq:R-update |
| `(φ, σ_η²)^u_k` (`sv_u`) | AR(1) log-vol comuni (`μ≡0`) | `(r, 3)` | **B** | (b) | `sample_vol.draw_ar1_params` | eq:sv-logvol-u, eq:param-logvol |
| `(φ, σ_η²)^ε_i` (`sv_eps`) | AR(1) log-vol idiosincratiche (`μ≡0`) | `(M, 3)` | **B** | (b) | `sample_vol._sample_idio_vol` | eq:sv-logvol-eps |
| `ρ^u_k` (`rho_u`) | leverage comune, **scalare per-fattore** | `(r,)` | **C** | (b) | `sample_leverage.draw_rho[_scalar/_griddy]` | eq:lev-cond-common, eq:param-rho-cond |
| `ρ^ε_i` (`rho_eps`) | leverage idiosincratico | `(M,)` | **C** | (b) | idem | eq:lev-cond-scalar |
| `ν_u`, `ν_ε` (`nu_u/eps`) | gradi di libertà delle code | scalari | **D** | (d) | `sample_params.draw_nu_griddy` | eq:post-w-*, eq:g-nu-u |
| `a_j` (`hw_a`) | scale ausiliarie Huang–Wand (prior opz. su `Q`) | `(r,)` | A⁺ | (d) | `shared.draw_hw_aux` | eq:param-Q-hw-aux |

> **Nota su step e famiglia.** Lo schema del `.tex` (`subsec:gibbs-blocks`) mette *tutti* i
> parametri nel blocco **(d)**. Il **codice** invece disegna le famiglie **B** (`φ, σ_η²`) e
> **C** (`ρ`) **dentro** il blocco di volatilità **(b)**, per località: hanno bisogno del
> percorso `log h` appena campionato. In (d) restano la famiglia **A** (`A, Q, Λ, R`) e la
> **D** (`ν`). `μ` non compare perché è congelato a 0 (`eq:sv-mu-identification`): non è
> identificato separatamente dalla scala base `Q`/`R`.
>
> `Σ_0` (prior sullo stato iniziale `f_0`) è l'**unico** parametro *non* estratto: è tenuto
> fisso, è il ponte con cui il modello semina la ricorsione (`subsec:gibbs-blocks`).

---

## 3. Lo sweep: quattro blocchi, quattro famiglie

Uno sweep visita i blocchi in ordine, ognuno condizionato all'ultimo valore di tutti gli
altri (Gibbs a scansione sistematica):

```
(a) stati        f_{1:T}          FFBS: dato h, w, θ il modello è lineare-gaussiano
(b) volatilità   h^u, h^eps       il blocco NON coniugato: KSC / Omori / Metropolis
                 + φ, σ_η² (B)    +  ρ (C)     ← disegnati qui per località
(c) pesi         w^u, w^eps       draw Gamma coniugato (le code Student-t)
(d) parametri    A, Q, Λ, R (A)   +  ν (D)     ← sotto-sweep coniugato / griddy
```

Le **famiglie** raggruppano i parametri per *tipo di conditional* (`sec:gibbs-params`):

- **Famiglia A** — nucleo lineare-gaussiano **coniugato**: `A, Q` (MNIW / per-fattore),
  `Λ, R` (Normal–Inverse-Gamma). I posterior mean coincidono con i punti dell'EM.
- **Famiglia B** — AR(1) delle log-vol, **quasi-coniugata**: `(φ, σ_η²)`, uno per ciascuno
  dei `M+r` processi. Mescola male (cresta path/scala) → vedi ASIS.
- **Famiglia C** — leverage `ρ`, **Metropolis** (target non coniugato): RW su Branch A,
  griddy sul supporto `(-1,1)` su Branch B (mescola meglio della RW).
- **Famiglia D** — gradi di libertà `ν`, conditional 1-D **log-concavo** → griddy-Gibbs.

---

## 4. I file, uno per uno

### I motori (estraggono incognite)

| File | Blocco / Famiglia | Cosa fa |
|---|---|---|
| **`gibbs.py`** | orchestratore | `fit_dfm_mcmc` — **l'unico punto d'ingresso**. Monta lo sweep, applica i prior, immagazzina i draw, calcola le diagnostiche. `VARIANT_FLAGS` + `variant_kwargs` mappano le celle D1×D2 come restrizioni; `load_warm_init` legge il warm start. |
| `sample_states.py` | (a) | FFBS: `ffbs_sample_states` estrae `f_{1:T}`. `forward_filter_combined` è il filtro alimentato con la precisione combinata `g = w/h`; il backward sweep di Carter–Kohn gestisce lo stato companion singolare. |
| `sample_vol.py` | (b) + Fam. B | Volatilità **senza** leverage: mistura KSC-7 + FFBS scalare (`sample_common_vol_mv`, `_sample_idio_vol`), e i parametri AR(1) (`draw_ar1_params`). Contiene anche `logsq_corr_matrix` e il passo accoppiato QML. |
| `sample_leverage.py` | (b) + Fam. B,C | **Branch A** (timing contemporaneo): target di Metropolis **esatto** — nessuna mistura, nessuna linearizzazione. È la *controparte esatta*, il metro contro cui si misura B. `draw_rho_scalar/_griddy` per `ρ`. |
| `sample_leverage_lagged.py` | (b) + Fam. B,C | **Branch B** (timing laggato, il **default**): mistura di Omori-10 ⇒ condizionatamente a `(s, d)` il sistema è lineare-gaussiano e il percorso è un **draw FFBS diretto**. |
| `sample_params.py` | (d) + Fam. A,D | `Λ, R` (NIG, mixed-frequency); `ν` (griddy, `draw_nu_griddy` + prior opzionali); `A, Q` MNIW `draw_A_Q_block` (solo **senza** SV). |
| `shared.py` | (c)+(d) + Fam. A,D | I kernel riusabili estratti dall'EM (formula identica, «media → draw»): pesi Gamma (`draw_weights`), `A, Q` per-fattore (`draw_A_Q_perfactor`), `Λ, R` per serie, target `ν` (`nu_log_target`), prior Huang–Wand. |
| `sample_asis.py` | wrapper Fam. B | ASIS: riparametrizza `(φ, σ_η²)` fra centrata e non-centrata per rompere la cresta path/scala. Non è un blocco: **si aggancia alla Famiglia B** ovunque si campioni una volatilità. |

### Gli strumenti di supporto (non estraggono incognite)

| File | Cosa fa |
|---|---|
| `constants.py` | Le tabelle delle misture: `KSC7` (approssima `log χ²₁`, senza leverage) e `OMORI10` (leverage laggato), più `LOG_CHI2_*`, `QML_A/B`. `validate_mixture` ne controlla le identità *con tolleranza*. È la tabella che i motori SV leggono. |
| `diagnostics.py` | ESS, split-R̂ (`ess`, `split_r_hat`), le diagnostiche sul coupling della volatilità comune e sullo sbiancamento del leverage (`posterior_corr_Q`, `recommend_coupling`, `leverage_whitening_attenuation`) e la **harness di recovery a 3 livelli** (`run_recovery_mcmc[_sv/_leverage]`, `compare_branches_AB`). Misura se le catene recuperano la verità; non tocca il modello. |
| `simulate_sv.py` | Il **DGP** (il verso opposto del sampler): da `θ` noti genera un pannello `Y` sintetico con SV e leverage, per i test di recovery. ⚠ **prende `σ` in ingresso e restituisce `σ²`**: il sampler parla in varianza. |
| `__init__.py` | Docstring del package: cosa vive dove, e il contratto «additivo» rispetto all'EM. |

---

## 5. Semi (non codice di produzione)

`sample_vol.sample_volatility_block` (la restrizione scalare `H = h·I`) **non è
raggiungibile da `fit_dfm_mcmc`**: esiste solo come *seme* con cui `tests/test_shared`
verifica che il blocco per-fattore, a `r = 1`, riproduca bit-per-bit il vecchio blocco
scalare.

---

## 6. Il passo accoppiato del blocco comune (coupling / QML)

Con `Q` piena le `r` log-square comuni sono correlate: accoppiarle è teoricamente
corretto. Il **default** è `common_vol_coupling='decoupled'` (esatto a `Q` diagonale, la
via veloce vicino-diagonale altrimenti). La forma **QML** è l'opzione accoppiata *stabile*
quando `corr(Q)` è forte; `decoupled`↔`qml` si sceglie leggendo
`diagnostics.recommend_coupling`. Sotto **leverage** la QML sta dietro
`allow_experimental=True`: il suo comportamento a `corr(Q)` forte è ancora da
caratterizzare a fondo, ed è una delle cose che il **`validate`** misurerà per bene. È
un'opzione da validare, non codice morto.

---

## 7. Dove leggere il resto

* `docs/EM_for_student_t.tex` — la teoria: tutte le `eq:...` e le derivazioni dei conditional.
* `docs/MCMC_ENGINES_AND_TESTS.md` — la mappa completa: motori, inventario dei parametri
  sui due blocchi, chi testa cosa, i buchi.
* `docs/VALIDATION_REPORT.md` — la tabella dei verdetti: **quale parametro è recuperato e
  quale no**.
