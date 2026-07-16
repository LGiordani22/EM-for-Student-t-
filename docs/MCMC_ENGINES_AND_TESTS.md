# `src/mcmc/` — motori, parametri, test, buchi

**Sola lettura.** Questo documento non modifica codice. È la FASE 1: la mappa che serve
per decidere la FASE 2 (il validatore unico).

**Configurazione target:** Spec II (volatilità *outside*, `h_k` = volatilità del fattore
`k`) + Option A (`r` canali scalari `ρ_k` sugli shock grezzi) + **Branch B** (timing
laggato) come default; **Branch A** (contemporaneo) mantenuto come **controparte esatta**
(non ha linearizzazione) e usato come robustness check / gold standard.

Tutto ciò che segue è **letto dal codice**, non dalla memoria. Dove il codice contraddice
una convinzione precedente, vince il codice e lo segnalo.

---

## A. Cosa fa ogni motore

| Modulo | Blocco del Gibbs | Cosa campiona | Kernel | Chi lo chiama | Attivo sotto |
|---|---|---|---|---|---|
| **`gibbs.py`** (795) | orchestratore | — | — | l'utente (`fit_dfm_mcmc`) | sempre |
| **`sample_states.py`** (301) | **(a)** stati | `f_t` (e `f_aug`) | FFBS Kalman | `gibbs` | sempre |
| **`sample_vol.py`** (864) | **(b)** vol *senza* leverage | `h^u_k`, `h^ε_i`, `φ`, `σ²_η` | mistura KSC-7 + FFBS | `gibbs` (`sv=True, leverage=False`) | entrambi i branch (non c'è leverage) |
| **`sample_leverage.py`** (981) | **(b)** + Family B + Family C | `h`, `φ`, `σ²_η`, `ρ` | **Branch A**: Metropolis su target **esatto** | `gibbs` (`timing="contemporaneous"`) | **solo A** |
| **`sample_leverage_lagged.py`** (707) | **(b)** + Family B + Family C | idem | **Branch B**: mistura Omori-10 + FFBS | `gibbs` (`timing="lagged"`) | **solo B** |
| **`sample_params.py`** (268) | **(d)** parametri | `Λ`, `R`, `ν_u`, `ν_ε`; `A,Q` (solo no-SV) | NIG; griddy-Gibbs; MNIW | `gibbs` | sempre (MNIW solo `sv=False`) |
| **`shared.py`** (627) | **(c)** pesi + **(d)** `A,Q` | `w^u_t`, `w^ε_t`, `A`, `Q`, `hw_a` | Gamma; two-step per-fattore; Huang-Wand | `gibbs` | sempre |
| **`sample_asis.py`** (158) | wrapper su Family B | ri-estrae `σ_η` (segnato) e `φ` | interweaving CP↔NCP | i **tre** blocchi vol | entrambi i branch |
| **`simulate_sv.py`** (360) | — (DGP) | genera i dati veri | — | test / bench | — |
| **`constants.py`** (165) | — | `KSC7`, `OMORI10`, `LOG_CHI2_*`, `QML_A/B` | — | tutti | — |
| **`diagnostics.py`** (1148) | — | ESS, split-R̂, diagnostiche P1/P4/P5, harness di recovery | — | `gibbs` (diagnostica) / manuale | — |
| **`tests/bench_p6_rho.py`** (175) | — | benchmark del mixing di `ρ` (Branch B) | — | manuale | — |

### Rami morti e semi di test

| Cosa | Stato | Dettaglio |
|---|---|---|
| `sample_vol.sample_volatility_block` (scalare, `H = h·I`) | **MORTO** | Non raggiungibile da `fit_dfm_mcmc`. Unico chiamante: `test_shared.py:428`. È un **seme di test** della restrizione scalare, tenuto apposta. Va dichiarato tale, altrimenti sembra un motore vivo. |
| `sample_params.draw_A_Q_block` (MNIW) | **vivo ma solo `sv=False`** | La cella "current model". Con SV si usa `shared.draw_A_Q_perfactor`. |
| `sample_vol.sample_common_vol_mv` | **vivo, indiretto** | Chiamato da `sample_volatility_block_specII` (blocco comune senza leverage) **e** dal warm-seed di Branch A (solo con `lev_path_sampler="single"`). Mai da `gibbs` direttamente. |
| `sample_leverage._lev_path_mh` (single-move **scalare**) | **vivo** | È il path sampler del blocco **idiosincratico** sotto Branch A (`sample_leverage.py:941`). Non è morto: il nuovo kernel a blocco (Laplace) copre **solo il blocco comune**. ⚠️ Asimmetria non documentata. |
| `draw_rho_vec`, `dominant_dir_z`, `common_lev_scalar` | **rimossi** (Phase 7/8) | Nessuna traccia. |

> ⚠️ **Commento obsoleto in `gibbs.py:617`**: dice ancora «The Family C griddy is wired on
> Branch B only — Branch A keeps its RW-Metropolis untouched». **Non è più vero**: il
> griddy è ora cablato su entrambi i rami (era necessario per confrontarli senza
> confondimento). Il commento va corretto.

---

## B. Inventario dei parametri — **su DUE blocchi**

### La risposta alla domanda esplicita: **sì, il blocco idiosincratico ha tutto**

Letto dal codice (`gibbs.py` chiavi dei `draws`; `sample_leverage_lagged.py` loop
idiosincratico; `shared.draw_weights`; `sample_params.draw_nu_griddy`):

| Domanda | Risposta dal codice |
|---|---|
| Esiste una SV idiosincratica `h^ε`? | **Sì.** `draws["h_eps"]` è `(n_keep, T, M)`: **un processo per serie**, non uno comune. |
| Ha `φ^ε` e `σ²_ε` propri? | **Sì.** `draws["sv_eps"]` è `(n_keep, M, 3)` = `(μ, φ, σ²)` **per serie**. |
| Esiste un leverage idiosincratico `ρ^ε`? | **SÌ.** `draws["rho_eps"]` è `(n_keep, M)`: **un `ρ` per serie**, estratto dalla stessa Family C del blocco comune. |
| I pesi `w^ε` sono distinti da `w^u`? | **Sì.** `draw_weights` restituisce **due** vettori `(T,)`: uno per il blocco degli shock comuni, uno per quello delle osservazioni. Sono **scalari per periodo** (condivisi fra i fattori / fra le serie), non per serie. |
| Uno `ν` o due? | **Due**: `ν_u` e `ν_ε`, estratti separatamente (`draw_nu_griddy` chiamato due volte). |

**Quindi il modello è strutturalmente simmetrico fra i due blocchi.** Il refactoring 1→r
volatilità ha riguardato il blocco dei **fattori** (dove prima c'era una `h` scalare
comune); il blocco idiosincratico era **già** per-serie e non è rimasto indietro
*strutturalmente*.

**Ma è rimasto indietro sui test** — e su questo il tuo sospetto è esatto. Vedi §C.

L'unico interruttore che spegne il blocco idiosincratico è `sv_idio=False` (restrizione
**D2-a**): `h^ε ≡ 1`, niente Family B né Family C idiosincratiche. **Non è il default.**

### Blocco FATTORI / STATI (shock `u_t`)

| Simbolo (.tex) | Nome nel codice | Dove è campionato | Famiglia | Forma |
|---|---|---|---|---|
| `f_t` | `draws["F"]`, `f_aug` | `sample_states.ffbs_sample_states` | blocco (a) | `(T, r)` |
| `A` | `draws["A"]` | `shared.draw_A_Q_perfactor` (SV) / `sample_params.draw_A_Q_block` (no-SV) | **A** | `(r, r·p)` |
| `Q` | `draws["Q"]` | idem | **A** (MNIW / IW / Huang-Wand) | `(r, r)` |
| `a_j` (aux Huang-Wand) | `draws["hw_a"]` | `shared.draw_hw_aux` | **A** | `(r,)`, solo se `q_prior="huang_wand"` |
| `Σ_0` | `theta["Sigma_0"]` | **non campionato** | — | fissato |
| `μ^u_k` | `sv_u[:, 0]` | **NON CAMPIONATO — fissato a 0** | — | convenzione di identificazione |
| `φ^u_k` | `sv_u[:, 1]` | `_draw_phi_lev` (leverage) / `draw_ar1_params` | **B** | `(r,)` |
| `σ²_{η,k}` | `sv_u[:, 2]` | `_draw_sigma2_lev` / `draw_ar1_params` | **B** | `(r,)` |
| `ρ^u_k` | `draws["rho_u"]` | `draw_rho` (griddy / RW) | **C** | `(r,)` |
| `h^u_k` | `draws["h_u"]`, `logh_u` | blocco (b) | — (path latente) | `(T, r)` |
| `w^u_t` | `w_u` | `shared.draw_weights` | **D** | `(T,)` scalare per periodo |
| `ν_u` | `draws["nu_u"]` | `sample_params.draw_nu_griddy` | **D** | scalare |

### Blocco OSSERVAZIONI / IDIOSINCRATICO (shock `ε_t`)

| Simbolo (.tex) | Nome nel codice | Dove è campionato | Famiglia | Forma |
|---|---|---|---|---|
| `Λ` | `draws["Lambda"]` | `sample_params.draw_Lambda_R_block` | **A'** (NIG) | `(M, r)` |
| `R` | `draws["R"]` | idem | **A'** (NIG) | `(M,)` diagonale |
| `μ^ε_i` | `sv_eps[:, 0]` | **NON CAMPIONATO — fissato a 0** | — | idem |
| `φ^ε_i` | `sv_eps[:, 1]` | `_draw_phi_lev` / `draw_ar1_params` | **B** | `(M,)` — **una per serie** |
| `σ²_{ε,i}` | `sv_eps[:, 2]` | `_draw_sigma2_lev` / `draw_ar1_params` | **B** | `(M,)` — **una per serie** |
| `ρ^ε_i` | `draws["rho_eps"]` | `draw_rho` | **C** | `(M,)` — **una per serie** |
| `h^ε_i` | `draws["h_eps"]`, `logh_eps` | blocco (b) | — (path latente) | `(T, M)` |
| `w^ε_t` | `w_eps` | `shared.draw_weights` | **D** | `(T,)` scalare per periodo |
| `ν_ε` | `draws["nu_eps"]` | `sample_params.draw_nu_griddy` | **D** | scalare |

> **Convenzione, ora corretta ma da ricordare:** `simulate_dfm_sv` prende in ingresso
> `sv_* = (μ, φ, **σ**)` — la **deviazione standard** — mentre il sampler parla in
> **varianza** `σ²`. Le due convenzioni collidevano sullo stesso nome di campo. Il
> simulatore ora **restituisce** `sv_u`/`sv_eps` in varianza (confrontabili con i
> `draws`) ed echeggia gli input come `sv_u_sigma`/`sv_eps_sigma`.

---

## C. Chi testa cosa — e i buchi

### Cosa asserisce ogni file

| File | Cosa asserisce | Branch / config |
|---|---|---|
| `test_shared` (68) | **Algebra**, non recovery: pesi, `A,Q` conditional, prior, deflazione per `h` | agnostico |
| `test_passo1` (9) | `A`, `Q`, `Λ`, `R`, `ν_ε` ≈ **EM**; stati corr > 0.85 | **no-SV** |
| `test_passo2` (13) | mistura KSC; bit-identità no-SV; `corr(logh, vero) > 0.8`; `φ^u > 0.85` | SV, **no leverage** |
| `test_passo3` (9) | `ρ ≈ −0.5`; skew sinistro; acceptance | **A** |
| `test_passo4` (29) | costanti Omori; `ρ ≈ −0.5`; skew; griddy; immunità P3 | **B** |
| `test_asis` (14) | ASIS **non sposta** la posterior di `φ`/`σ²`; ESS(`φ`) ×1.3 | default (**A**) |
| `test_variants` (44) | celle `VARIANT_FLAGS`; tripwire P1; prior HW | tutte |
| `test_diagnostics` (39) | le funzioni diagnostiche; nessun consumo di RNG | agnostico |
| `test_spec2_recovery` (17) | recovery Spec II | SV, **no leverage** |
| `test_perfactor_leverage` (13, **3 ROSSI**) | **ordinamento e segno** di `ρ_k`; separazione dei canali `h_k`; frontiera P2; parità A/B | **A e B** |
| `test_branchA_qml` (17) | invarianza del kernel Laplace; costanti QML; instabilità QML congelata | **A e B** |
| `test_sigma_eta` (7, **1 rosso**) | convenzione σ/σ²; recovery di `σ²_η`; sensibilità al prior | **B** |

### I buchi — in ordine di gravità

1. **`ρ^ε` (leverage idiosincratico) non è MAI stato testato in recovery.** 🔴
   È impostato nel DGP di quasi ogni test (`rho_eps=-0.3`) e **nessuno lo confronta con il
   vero**. L'unico posto dove appare un confronto è `diagnostics.py` (l'harness
   `run_recovery_mcmc_leverage`, che *stampa* `rho_eps mean` e calcola `atten_eps`) — ma è
   una harness, non un test con asserzioni: nessuno la esegue in CI e nessuno fallisce se
   il numero è sbagliato.
   **È il buco più importante**, e per la ragione che dici tu: il blocco osservazioni è
   quello che proietta il PIL nel forecast.

2. **Tutto il blocco idiosincratico è privo di recovery test.** 🔴
   `φ^ε`, `σ²_ε`, `h^ε`: nessun test li confronta con la verità. La SV idiosincratica
   esiste, gira a ogni sweep, entra nella densità predittiva — e non è validata.

3. **Nessun test asserisce la MAGNITUDINE di `ρ`.** 🔴
   Solo **segno** e **ordinamento**. L'unico che guarda il valore (`test_passo3/4`) usa un
   vero di `−0.5` con tolleranza larga. **È così che un'attenuazione del 35% è rimasta
   invisibile per mesi con la suite verde.** La suite misurava la cosa sbagliata.

4. **`Λ` e `R` non sono mai testati sotto SV + leverage.** 🟠
   `test_passo1` li valida contro l'EM, ma **senza SV**. Nella configurazione target
   nessuno li guarda.

5. **`ν_u` non è mai confrontato col vero.** 🟠 Solo «resta sano (2.5–15)».

6. **ASIS non è mai testato sotto Branch B.** 🟠 `test_asis` gira col timing di default
   (contemporaneo = **A**). Sotto B l'interweave esiste (`sample_leverage_lagged.py:625`)
   e non è coperto.

7. **QML non è testata sotto entrambi i rami** — e non può esserlo: sotto A **solleva**
   per costruzione (non c'è covarianza di misura da accoppiare). Va detto, non «testato».

### Ridondanze e obsolescenze

- `test_passo3` e `test_passo4` asseriscono **le stesse tre proprietà** (`ρ ≈ −0.5`, skew
  sinistro, simmetria a `ρ=0`) sui due rami. Non è ridondanza vera (i rami sono diversi),
  ma è **struttura duplicata**: nel validatore diventa **una** funzione parametrizzata per
  branch.
- `test_passo2`, `test_spec2_recovery` e `test_perfactor_leverage` si sovrappongono sul
  recovery del path di volatilità.
- `test_shared:428` testa `sample_volatility_block` — un **ramo morto**. È un seme
  deliberato, ma va etichettato come tale.

---

## D. Stato di verità, parametro per parametro

Verdetto onesto **oggi**. `recuperato` = un test lo confronta con il **vero** e passa.

### Blocco fattori

| Parametro | Verdetto | Evidenza / storia |
|---|---|---|
| `f_t` | ✅ **recuperato** | corr > 0.85 col vero (`test_passo1`) |
| `A` | ✅ **recuperato** | ≈ EM, err. rel. < 0.15 — ma **solo no-SV** |
| `Q` | ✅ **recuperato** | ≈ EM, err. rel. < 0.20 — **solo no-SV** |
| `w^u` | ✅ **recuperato** | media MC == formula chiusa |
| `ν_u` | ⚪ **mai testato** | solo «sano» |
| `μ^u` | ⚫ **non stimato** | fissato a 0 (identificazione) |
| `φ^u_k` | 🟡 **solo canali forti** | > 0.85 sui forti; **collassa (≈0.62)** sul canale debole |
| `σ²_{η,k}` | 🔴 **bias noto** | **NUOVO**: half-Normal (B=1) sovrastima di **1.5–3.8×**; l'IG sembra migliore solo perché il suo prior è *per caso* centrato sul vero. **È debolmente identificato ⇒ è il prior a guidare la risposta.** |
| `h^u_k` | 🟡 **tetto informativo (P2)** | canali forti: corr > 0.6 e batte le cross-corr. Canale debole: **non identificato**; `corr(ĥ,h)` **satura a ~0.63 anche a T=4800 con i parametri veri**. Non è un difetto: è quanto segnale c'è **per periodo**. |
| `ρ^u_0` (dominante) | 🔴 **identificato ma ATTENUATO** | R̂ ≈ 1.0, ESS ok ⇒ converge. Ma su 12 dataset: **ρ̂ = −0.44 contro −0.70 vero**, fattore **0.63**, e il CI 90% copre il vero solo il **25%** delle volte. **Causa isolata:** con il path **vero** congelato, Family C recupera al **98%** ⇒ il conditional è **corretto** e Omori **non attenua**. L'attenuazione entra **tutta dall'incertezza sulle latenti**. |
| `ρ^u_1` (debole) | 🔴 **non identificato** | CI copre lo zero **e entrambi i segni** (`[−0.945, +0.405]`). I 3 test rossi di `test_perfactor_leverage` asseriscono proprietà di **questa quantità non identificata**: il fix li ha **smascherati**, non rotti. |
| `ρ^u_2` | 🔴 **attenuato** | fattore **0.65**, copertura 58% |

### Blocco osservazioni

| Parametro | Verdetto | Evidenza |
|---|---|---|
| `Λ` | 🟡 **recuperato solo no-SV** | ≈ EM (err. < 0.10). **Mai sotto SV+leverage.** |
| `R` | 🟡 **recuperato solo no-SV** | ≈ EM (err. < 0.15). Idem. |
| `w^ε` | ✅ **recuperato** | media MC == formula |
| `ν_ε` | ✅ **recuperato** | ≈ EM (err. < 0.30) |
| `μ^ε` | ⚫ **non stimato** | fissato a 0 |
| `φ^ε_i` | ⚪ **MAI TESTATO** | — |
| `σ²_{ε,i}` | ⚪ **MAI TESTATO** | — |
| `h^ε_i` | ⚪ **MAI TESTATO** | — |
| `ρ^ε_i` | ⚪ **MAI TESTATO** | 🔴 **il buco più grave** |

---

## E. Le due varianti algoritmiche

### ASIS (`sample_asis.py`, `use_asis`)

**Cosa riparametrizza.** La cresta **path ↔ scala**: `(φ, σ²_η)` mescolano male perché il
path `x_t = log h_t` e la sua scala `σ_η` sono fortemente correlati a posteriori.
ASIS alterna due parametrizzazioni della **stessa** coppia:
- **CP** (centered): si estrae `(φ, σ²)` dato il path `x`;
- **NCP** (non-centered): si standardizza `x̃_t = x_t / σ_η`, così `σ_η` **passa
  nell'equazione di misura** e si ri-estrae come *coefficiente di regressione con segno*;
  poi il path viene riscalato.

Il segno conta: `σ_η` è estratto **segnato**, il che permette alla catena di ribaltare un
segno che sarebbe altrimenti appiccicoso.

**Dove si aggancia.** È un **wrapper sulla Family B**, non un blocco: gira ovunque si
campioni una volatilità — `sample_vol.py` (senza leverage), `sample_leverage.py` (**A**),
`sample_leverage_lagged.py` (**B**). Quindi **entrambi i rami**, e sia sul blocco comune
sia su quello idiosincratico.

**🔴 IL CONFONDIMENTO (`gibbs.py:498`).**
```python
if use_asis and sv_sigma_prior != "half_normal":
    sv_sigma_prior = "half_normal"
```
`use_asis=True` **forza** il prior half-Normal. I due flag **non sono ortogonali**, e
qualunque esperimento che vari `use_asis` sta variando **anche il prior**.

Ma qui devo correggere la premessa della tua richiesta: **non è (solo) un bug di design —
è un vincolo matematico.** Perché l'interweaving campioni il giunto esatto, CP e NCP devono
esprimere **lo stesso prior** su `σ_η`; e il gaussiano sul `σ_η` *segnato* è ciò che rende
il draw NCP **coniugato**. Renderli ortogonali *tout court* romperebbe ASIS.

Ci sono quindi due vie, e vanno distinte:
1. **Vera ortogonalizzazione**: implementare il draw NCP **non coniugato** sotto IG (un
   passo di Metropolis in più). Fattibile, ~30 righe. Solo così `use_asis` e
   `sv_sigma_prior` diventano flag indipendenti.
2. **Disciplina sperimentale**: lasciare il vincolo (che è corretto) e **vietare** agli
   esperimenti di variare entrambi. Ogni confronto su ASIS va fatto **a prior fissato**.

⚠️ Questo confondimento ha già prodotto un errore reale: la conclusione «ASIS aiuta `ρ`»
era in realtà **il prior che aiutava `ρ`** (il GATE 4 li ha separati). E adesso sappiamo
anche che quel prior **gonfia `σ²_η` di 1.5–3.8×**. Cioè: abbiamo curato P6 con un prior
che distorce il parametro su cui agisce, e **nessun test lo controllava**.

**Chi lo testa oggi:** `test_asis` (14) — ma solo che ASIS **non sposta** la posterior e
che alza ESS(`φ`). Nessun test sotto **Branch B**. Nessun test dell'effetto su `ρ`.

### QML (`common_vol_coupling="qml"`, `recommend_coupling`)

**Cosa approssima.** Le `r` misure log-square hanno errori `ξ_k` **correlati fra loro**
quando `Q` non è diagonale (lo sono gli shock `u_k` da cui derivano). Il blocco default
(`decoupled`) **ignora** quella correlazione — è **esatto a `Q` diagonale**, approssimato
altrove. La QML (Harvey–Ruiz–Shephard 1994) sostituisce la mistura con **una sola
gaussiana** a covarianza costante esatta `Σ_ξ = (π²/2)·R_ξ`: niente indicatori, quindi una
`R_ξ` **piena** passa (la mistura non la fa passare — non fattorizza, ed è il motivo per
cui la forma `literal` è instabile).

**`recommend_coupling(Q, tol=0.05)`.** La soglia **non è su `corr(Q)`** ma sulla
**sovra-confidenza indotta** (`coupling_overconfidence`, che è del *secondo* ordine negli
elementi fuori diagonale): `≤ 5%` → `decoupled`; `> 5%` → `qml` **da considerare**, ma il
.tex la rende una condizione **congiunta** — si accende solo se anche la calibrazione della
coda risulta guasta, mai su questo numero da solo. Sul pannello reale (`corr(Q) ≈ 0.1`) dà
**0.4%** ⇒ `decoupled`.

**Dove è raggiungibile.**
- **Senza leverage** (`sample_common_vol_mv`): **stabile**. A `corr(Q)=0.92` dà `φ̂ = 0.856`
  dove la forma `literal` collassa a `0.422`.
- **Sotto leverage**: **solo Branch B**, e **gated dietro `allow_experimental=True`**.
- **Sotto Branch A**: **solleva `ValueError`** — e giustamente: A non forma **mai** una
  covarianza di misura (nessuna mistura, nessuna linearizzazione), quindi non c'è nulla da
  accoppiare. Non è una lacuna, è una proprietà.

**Stato attuale (in evoluzione).** Il primo tentativo sotto leverage **collassava** a
`corr(Q)=0.8` (φ̂ → 0.42, ρ̂ al bordo). Causa **isolata**: non il coupling della misura, ma
la **deriva troncata** (usava `z_k ≈ d_k·exp(ξ_k/2)`, cioè `M = I`, valido solo a `Q`
diagonale). La versione corretta ricostruisce `z = M·ε` esattamente
(`M = Q^{-1/2}·diag(√q_kk)`, transizione FFBS **piena**): **recupera `ρ`** (a `corr(Q)=0.8`:
`ρ̂ = −0.47` contro `−0.15` del decoupled, vero `−0.70` ⇒ **P5 eliminata**) ma **`φ` di un
fattore collassa ancora** (0.276). **Progresso, non soluzione.**

**Chi lo testa oggi:** `test_diagnostics` (le funzioni: soglia, monotonia, `R_ξ`),
`test_spec2_recovery` (stabilità senza leverage), `test_branchA_qml` (costanti QML +
instabilità **congelata**). Non esiste — e non può esistere — una copertura «sotto entrambi
i rami».

---

## E-bis. Proposta di struttura per il validatore (FASE 2)

**Non un file unico.** Un file da 2000 righe sarebbe lo stesso problema con un nome nuovo:
illeggibile, e con la tentazione di nascondere i buchi dentro una funzione lunga. Ciò che
deve essere **unico** è il **punto d'ingresso** e la **tabella dei verdetti**, non il file.

```
src/mcmc/validate/
├── __init__.py
├── spec.py        # L'INVENTARIO COME DATO: una riga per parametro
│                  #   (simbolo_tex, nome_codice, blocco, famiglia, atteso)
│                  #   È la §B di questo documento, eseguibile.
├── dgp.py         # I DGP canonici, uno per cella di configurazione.
│                  #   Un solo posto dove vive la verità.
├── checks/
│   ├── linear.py      # A, Q, Lambda, R, f, w, nu   (entrambi i blocchi)
│   ├── volatility.py  # phi, sigma2_eta, h          (comune E idiosincratico)
│   ├── leverage.py    # rho^u E rho^eps             (entrambi i branch)
│   └── variants.py    # ASIS, QML, celle D1xD2
├── report.py      # LA tabella dei verdetti + l'artefatto
└── run.py         # L'UNICO punto d'ingresso: python -m mcmc.validate.run
```

**Tre regole non negoziabili, ognuna nata da un errore che abbiamo fatto davvero:**

1. **Tre esiti, non pass/fail.** Ogni verdetto è
   `recuperato` / `recuperato con bias noto` / `non identificato` / `mai testato`.
   Un `pass/fail` binario costringe a scrivere soglie larghe per non avere rossi, ed è
   esattamente così che l'attenuazione di `ρ` è passata inosservata. Un parametro **non
   identificato** deve poter essere dichiarato tale **senza far diventare rossa la suite** —
   altrimenti si è tentati di asserire proprietà che non reggono (i 3 rossi di
   `test_perfactor_leverage`).

2. **Nessuna stima senza il suo ESS e il suo R̂.** Un `ρ̂` senza ESS non è una misura.

3. **La copertura si misura su N repliche, non su un dataset.** È l'unico modo di
   distinguere «sfortuna» da «bias»: su un dataset `ρ̂ = −0.50` sembrava rumore; su 12 la
   copertura al 25% l'ha smascherato. Il validatore deve avere una modalità `--coverage N`,
   e i verdetti «recuperato» sui parametri delicati devono poggiare su di essa.

**E una regola di scope:** ogni check gira su **entrambi i blocchi** (fattori *e*
osservazioni) e su **entrambi i rami** dove ha senso. La tabella finale deve avere le righe
`ρ^ε`, `φ^ε`, `σ²_ε`, `h^ε` — oggi vuote, ed è il punto.

---

## GATE 1 — mi fermo qui

La mappa è sopra. **Non scrivo il validatore finché non la confermi.**

Le tre cose che, se dovessi scegliere, guarderei per prime:
1. **`ρ^ε` mai testato** — il blocco che proietta il PIL.
2. **`σ²_η` gonfiato dal prior che abbiamo adottato per curare P6** — e nessuno lo controllava.
3. **Nessun test sulla magnitudine di `ρ`** — la ragione per cui la suite era verde mentre il parametro che porta lo skew era sbagliato del 35%.
