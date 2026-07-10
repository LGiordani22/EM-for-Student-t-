# `src/mcmc/` — mappa dell'engine di stima

**Cos'è.** Il motore MCMC (Gibbs) per il DFM Student-t a frequenza mista con
volatilità stocastica e leverage. È la controparte campionaria dell'EM di primo
stadio: stesso modello, stessa macchina Kalman / mixed-frequency, ma la stima è un
sampler i cui blocchi estraggono dalle **full conditional esatte**.

**Contratto strutturale (mai violato).** Il pacchetto è *additivo*: `em_e_step`,
`em_m_step`, `kalman` sono byte-for-byte intatti. Ogni kernel estratto è verificato
contro la sua controparte EM in `test_shared.py` — stessa formula, con *draw* al
posto di *mean/point*. Nel **limite a prior piatto** i draw coincidono con l'EM
bit-for-bit: è il "seam" che tiene onesto tutto il refactor.

**Teoria.** `docs/EM_for_student_t.tex` è la fonte di verità. Le etichette
`eq:...` / `sec:...` citate qui sotto sono quelle vere del `.tex`.

---

## 1. La sweep di Gibbs, in quattro passi

Una iterazione di `fit_dfm_mcmc` (Algorithm `alg:gibbs-sv`):

| passo | oggetto | metodo | modulo |
|---|---|---|---|
| **(a)** | stati `f_{0:T-1}` | FFBS (Carter–Kohn su companion singolare) | `sample_states.py` |
| **(b)** | `h^u` (r), `h^ε` (M), `sv_u`, `sv_eps` (+ `ρ_u`, `ρ_eps`) | KSC-7 / Omori-10 | `sample_vol.py`, `sample_leverage[_lagged].py` |
| **(c)** | pesi di coda `w^u_t`, `w^ε_t` | Gamma coniugata | `shared.draw_weights` |
| **(d)** | `A, Q` / `Λ, R` / `ν_u, ν_ε` | Gaussiana vettorizzata + IW / NIG / griddy | `shared.py`, `sample_params.py` |

Il **modello master** è la cella `D1-b × D2-b` con leverage. Tutto il resto è una
*restrizione*: si spengono blocchi, non si cambia il modello.

### La specificazione della volatilità comune (fondamentale, e facile da sbagliare)

`subsec:vol-placement`. **Entrambe** le specificazioni danno a ogni fattore la sua
`h^u_{i,t}`, raccolte in `H^u_t` diagonale. Differiscono solo per **dove** entra
rispetto al mixing `Q^{1/2}`:

- **Spec I (inside)**, `eq:vol-inside`: `Var(u_t) = Q^{1/2} H^u_t Q^{1/2}`. Il
  diagonale è un *blend*, `Σ_j (Q^{1/2})²_{ij} h_{j,t}`: `h^u_i` è la volatilità
  dello shock *ortogonalizzato*, per giunta non invariante alla scelta della radice.
- **Spec II (outside)**, `eq:vol-outside` — **adottata**:
  `Var(u_t) = √H^u_t · Q · √H^u_t / w^u_t`. Il diagonale è `h_{i,t} q_{ii}`: la
  volatilità del fattore `i` *stesso*.

Coincidono solo se `Q` è diagonale o `H^u_t = c·I`. ⚠️ **Il blocco a volatilità
scalare NON è "Spec I"**: è la restrizione `H = h·I`, dove il bivio è vuoto. E `h`
non è mai 1 in nessuna delle due specificazioni: gli unici casi congelati a 1 sono
quelli che il `.tex` congela (`h^ε≡1` sotto D2-a; `h≡1` nella riga no-SV, dove
`σ_η²=0` rende `φ` inoperante).

---

## 2. Moduli, uno per uno

### `constants.py` — le tabelle di mistura
`KSC7` (7 componenti, approssima `log χ²_1`) e `OMORI10` (10 componenti, con i
coefficienti `(a_j, b_j)` della linearizzazione di `exp(ξ/2)` che porta il drift di
leverage laggato). `validate_mixture()` esegue i tre check di consistenza
**sempre con tolleranza** — le misture sono approssimazioni, un `==` fallirebbe o,
peggio, passerebbe per caso su una tabella sola.

### `shared.py` — i kernel matematici riusabili
Sono gli helper che nell'EM vivono *inline* dentro le E-/M-step, estratti qui con
sole dipendenze numpy/scipy.

- `realized_deflated_d_eps/_d_u` — residui di Mahalanobis "realized" (traccia di
  covarianza posteriore assente: lo stato è *campionato*), deflazionati da `h`.
  `d_u` accetta `h_u` `(T,)` (scalare) o `(T,r)` (per-fattore, Spec II:
  `d^u_t = (H^{-1/2}u_t)' Q^{-1}(H^{-1/2}u_t)`).
- `draw_weights` — passo (c): `w^ε_t ~ Ga((ν_ε+m_t)/2, (ν_ε+ď^ε_t)/2)`, idem `w^u_t`.
  A `t=0` il peso comune è estratto dal *prior* (non c'è `u_0`).
- `nu_foc`, `nu_log_target` — la condizione del prim'ordine e il log-target 1-D
  log-concavo di `ν`.
- `draw_A_Q` — **MNIW congiunta**, valida quando `G^u_t = g^u_t Q^{-1}` si fattorizza
  (no-SV, o `H = hI`). Con i prior `(Psi0, nu0, A0, kappa)`: IW su `Q` e
  matrix-Normal coniugata `A|Q ~ MN(A0, Q, (κI)^{-1})`.
- `draw_A_Q_perfactor` — **il draw di Spec II**. Il whitening
  `C_t = √w^u_t · H_t^{-1/2}` *non commuta* con `A`, quindi i momenti pesati scalari
  non sono più sufficienti e la MNIW chiusa non sopravvive. Diventa una mossa di
  Gibbs in due tempi: `Q|A ~ IW(Ψ0 + Σ ǔ_t ǔ_t', ν0+T_eff)` (`eq:param-Q-post`) e
  poi `vec(A)|Q ~ N(m_n, V_n)` con precisione
  `V_n^{-1} = V0^{-1} + Σ_t (f_{t-1}f_{t-1}') ⊗ G^u_t` — una **somma di prodotti di
  Kronecker**, uno per periodo, che non collassa (`eq:param-A-precision`).
- `hw_iw_prior`, `draw_hw_aux` — **Huang–Wand**. Condizionatamente alle `r` scale
  ausiliarie il prior è un IW ordinario, quindi è uno swap di `(Ψ0,ν0)` dentro il
  draw esistente, più `r` scalari IG. Marginalizzando: half-t_{ν*} su ogni
  `√Q_jj` e correlazioni marginali Uniform(−1,1) a `ν*=2`.
- `composite_regressor` — il regressore composito Mariano–Murasawa per le righe
  trimestrali.
- `draw_lambda_r_series` — NIG scalare per serie (la restrizione a blocchi lascia
  ogni serie con un solo loading).

### `sample_states.py` — passo (a), FFBS
Il forward è `kalman.kalman_filter` riusato tal quale. Il backward è nuovo, e la
sottigliezza è il **companion singolare**: `f_aug[t] = (f_t,…,f_{t-4})` e il rumore
entra solo nel blocco di testa. Condizionare su `f_{t+1}` da solo **sarebbe
sbagliato** (le righe trimestrali fanno sì che `y_{t+1}` porti informazione su `f_t`
che `f_{t+1}` non riassume): si condiziona sull'intero `f_aug[t+1]`, il che pinna i
primi quattro blocchi e lascia libero solo il lag più profondo. Il path aumentato
viene poi *ricostruito* impilando le teste, così l'identità dei lag vale
esattamente e ogni lettore a valle vede un path coerente.
- `forward_filter_combined(..., Qcov (T,r,r))` — accetta la covarianza di
  innovazione **tempo-variante** `Q_t = √H^u_t Q √H^u_t / w^u_t` (`eq:states-tv-cov`).
- `ffbs_sample_states` — costruisce `Q_t` da `h_u` `(T,r)` e restituisce
  `{f_aug, F, loglik}`.

### `sample_vol.py` — passo (b), caso base (KSC, no leverage)
- `sample_log_vol_process` — il template scalare KSC: indicatori multinomiali a 7
  componenti + FFBS scalare su `log h`.
- `draw_ar1_params` — Family B `(μ, φ, σ²)` con `μ=0` imposto; prior `σ_η`
  selezionabile: `inverse_gamma` (baseline coniugato) o `half_normal` (default del
  master sampler *quando ASIS è attivo*, Gelman 2006).
- `sample_common_vol_mv` — **il blocco comune per-fattore**. Lettura per componente
  `e_{k,t} = √(w/q_kk)·u_{k,t}`, offset noto `log q_kk` (va sottratto, altrimenti il
  filtro lo assorbe nello stato e viola `μ=0`). Default **disaccoppiato**
  (`R_xi=None`): `r` sub-sweep KSC scalari.
- `sample_volatility_block_specII` — comune per-fattore + `M` idiosincratici;
  `sv_idio=False` è **D2-a**.
- `sample_volatility_block` — ⚠️ la restrizione scalare `H=h·I`. **Su nessun path
  del sampler**: è tenuto solo come *seam bit-esatto a r=1* contro cui
  `sample_common_vol_mv` è validato.

> **Il caveat del coupling multivariato (P1).** Il `.tex` presenta come *il* metodo la
> cross-covarianza `Σ_ξ,t = diag(v_{s_k}) R_ξ diag(v_{s_k})`. Implementata e testata:
> su `corr(Q)=0.92` **distorce** il fattore meno persistente (`φ` 0.90→0.42) perché
> mescola una `R_ξ` *incondizionata* con le `v_s` *condizionali*. La variante QML
> (costante, senza mistura) è stabile ma non batte il disaccoppiato sul punto.
> Motivo di fondo: rumore di misura positivamente correlato è informazione
> **ridondante** ⇒ il coupling può solo *abbassare* l'accuratezza puntuale; il suo
> beneficio è la **calibrazione**. Default = disaccoppiato; coupled = `EXPERIMENTAL`.

### `sample_leverage.py` — Branch A (timing contemporaneo, Metropolis)
Il drift di leverage rende la transizione non-lineare-gaussiana: niente KSC+FFBS,
si usa un **Metropolis single-move** sul path (`eq:lev-mh-target`).
- **Option A**: `r` correlazioni **scalari** `ρ_i`, ciascuna che accoppia `η^u_i`
  al proprio shock **grezzo** `z^u_i` — la `i`-esima componente di
  `z^u_t = √w · Q^{-1/2}(√H^u_t)^{-1} u_t` (`eq:lev-cond-common`). Niente draw
  vettoriale, niente regione `ρ'ρ<1`.
- `_lev_path_mh_mv_common` — il path draw è **accoppiato**: muovere `x_{k,t}`
  sposta il drift di *tutti* gli `r` fattori, perché `z^u` mescola ogni volatilità
  attraverso `Q^{-1/2}`. Validato bit-for-bit a `r=1` contro il kernel scalare.
- **Due whitening distinti**, e non vanno confusi: la *misura* è per componente
  `√(w/q_kk)u_k`; lo *shock grezzo* usa la `Q^{-1/2}` simmetrica piena.
- ⚠️ **Trappola nota (P3)**: dal warm start piatto (`log h = 0`) la misura χ²₁
  per-fattore è troppo rumorosa perché il single-move esca (`h~1 → stati omoschedastici
  → u omoschedastici → h~1`), e `φ` degrada. Il kernel è corretto (provato in
  standalone); il problema è il *mixing del loop*. Fix adottato: **warm-seed** del
  path da un draw KSC-FFBS a blocchi (il path esatto a `ρ=0`) quando entra piatto.
  Target invariato.

### `sample_leverage_lagged.py` — Branch B (timing laggato, Omori + FFBS)
Il timing laggato accoppia `z_t` con `η_{t+1}`. Con l'identità dei segni
`z_t = d_t e^{ξ_t/2}` e la mistura a 10 componenti di Omori, condizionatamente a
indicatore **e segno** la media condizionale è **lineare in `log h_t`**: il sistema
torna lineare-gaussiano e il path si estrae con **un FFBS diretto**. È il motivo per
cui B mescola molto meglio di A (nessun Metropolis sul path, accettazione = 1).
- `r` canali scalari indipendenti (Option A), come sul lato idiosincratico.
- Due dettagli di bordo: **(i)** il segno aumentante è `d^u_k = sign(z^u_k)` dello
  shock a whitening **pieno**, mentre la *magnitudine* usa `e_k` per componente (è
  ciò che tiene `ξ_k` lineare in `log h_k`); **(ii)** `z^u_0` non esiste, quindi la
  prima transizione con leverage è quella *dentro* `log h_2`.
- ⚠️ **P5**: A e B usano whitening di magnitudine diversi (A: `|z^u_k|` esatto con `Q`
  piena; B: `|e_k|` per componente, la linearizzazione che rende possibile l'FFBS).
  Coincidono a `Q` diagonale. È ciò che il `.tex` prescrive, non un bug — ma i due
  branch concordano sui parametri condivisi solo a meno della convenzione.

### `sample_asis.py` — il booster di mixing
`(φ, σ_η²)` e il path stanno su una **cresta path/scala** (path più liscio ⇒ `σ_η²`
più piccola ⇒ path più liscio): il Gibbs a blocchi ci striscia sopra, `ESS(φ)` è
dell'ordine del *percento*. ASIS è un **wrapper sul draw di Family B**: riscrive lo
stesso posterior in coordinate non-centrate `x̃ = x/σ_η`, dove `σ_η` migra nella
*misura* come coefficiente di regressione gaussiano, ridisegna lì `(σ_η, φ)`, e
riscala indietro. `σ_η` è estratta **con segno** (in NCP `(σ,x̃)` e `(−σ,−x̃)` danno
la stessa `x`): il segno permette il flip che sblocca la catena.
Sotto leverage `z_t` è **congelato** durante il redraw NCP. Non accende né spegne
blocchi: cavalca ovunque una volatilità venga campionata. Misurato: **ESS(φ) ×2.1**
a posterior invariato.

### `sample_params.py` — passo (d)
- `draw_A_Q_block` — la MNIW sui momenti pesati (path no-SV).
- `draw_Lambda_R_block` — NIG per serie: regressore = fattore di blocco (mensile) o
  composito MM (trimestrale); prior `(a0, b0, m0, M0_inv)`.
- `draw_nu_griddy` — griddy-Gibbs su griglia **geometrica** con pesi trapezoidali;
  `nu_log_prior_exponential(mean=20)` / `nu_log_prior_uniform(2,50)` sono i prior
  propri di Family D (entrambi preservano la log-concavità del target).

### `gibbs.py` — l'orchestratore
`fit_dfm_mcmc(...) -> {"draws", "theta_mean", "meta"}`. `load_warm_init(config)`
carica il fit EM della **stessa** config (mai una generica).

### `simulate_sv.py` — i DGP sintetici
Due DGP per la volatilità comune: **scalar-common** (`sv_u`, la restrizione `H=hI`,
con `ρ` vettoriale — usato dai gate storici, bit-identico) e **per-fattore Spec II +
Option A** (`sv_u_perfactor` `(r,3)`), l'unico in cui `(φ_k, σ_k, ρ_k)` distinti sono
identificati.

### `diagnostics.py` — convergenza e recovery
`split_r_hat`, `ess`, `diagnostics_table`; gli harness `run_recovery_mcmc[_sv,
_leverage]` e `compare_branches_AB`; `leverage_skewness_check` (verifica che `ρ<0`
produca innovazioni asimmetriche a sinistra — il meccanismo che dà skew alla
predittiva).

---

## 3. La griglia D1×D2 come restrizioni (`sec:gibbs-variants`)

Lo scheletro non cambia mai; cambia solo *quali* blocchi girano.
**Ogni cella che campiona una volatilità comune lo fa sotto Spec II.**

| cella | flag | (b) `h^u` | (b) `h^ε` | (c) code | (d) A | (d) B | (d) C `ρ` | (d) D `ν` |
|---|---|---|---|---|---|---|---|---|
| D1a×D2a | `sv=True, sv_idio=False, tails="gaussian"` | ✅ | — | — | ✅ | solo `h^u` | † | — |
| D1a×D2b | `sv=True, tails="gaussian"` | ✅ | ✅ | — | ✅ | tutti | † | — |
| D1b×D2a | `sv=True, sv_idio=False` | ✅ | — | Gamma `w` | ✅ | solo `h^u` | † | ✅ |
| **D1b×D2b** *(master)* | `sv=True` | ✅ | ✅ | Gamma `w` | ✅ | tutti | † | ✅ |
| D1c×D2* | — | | | *outlier* | | | | *outlier* |
| current model | `sv=False` | — | — | Gamma `w` | ✅ | — | — | ✅ |

† Family C e la correzione di leverage in (b) sono attive **solo** se `ρ≠0`
(`leverage=True`); allora la cella si biforca ulteriormente in **Branch A**
(`timing="contemporaneous"`) o **Branch B** (`timing="lagged"`). Questa non è una
restrizione: è una bifurcazione *interna* al caso pieno, e persiste fino al master.

`gibbs.VARIANT_FLAGS` / `variant_kwargs(cell, leverage=, timing=)` sono questa
tabella in codice.

- **D2-a** (`sv_idio=False`): `h^ε≡1` congelato, Family B solo per gli `r` comuni, e
  niente `ρ_ε` — senza `h^ε` non c'è innovazione di log-vol a cui correlarlo.
- **D1-a** (`tails="gaussian"`): `w≡1`, passo (c) e Family D omessi. *Dimostrato* dal
  fatto che due run che differiscono **solo** per `ν` nel warm start danno draw
  bit-identici (se il passo (c) girasse, tutto a valle si muoverebbe).
- **current model** (`sv=False`): tutto il passo (b) + Family B + Family C spariscono.
  Restano (a), (c), (d)-A, (d)-D: è esattamente la controparte MCMC dell'EM di primo
  stadio.
- **D1-c** (outlier espliciti): `variant_kwargs` solleva `NotImplementedError`. Il
  `.tex` la definisce come *sostituzione* del blocco pesi e di Family D con gli
  indicatori di outlier (e, nella versione stocastica, la probabilità `p`); quel
  blocco **non esiste** nel motore. **Lasciata in sospeso per scelta.**
- **ASIS** non aggiunge righe alla griglia: è un wrapper su Family B.

---

## 4. I prior (`tab:param-prior-tuning`) — tutti dietro flag, default = piatto

| famiglia | prior | flag | default |
|---|---|---|---|
| A: `Q` | `IW(Ψ0=(2r+2)Q̂_EM, ν0=r+1)` — `ν0=r+1` rende **uniformi** le correlazioni marginali | `use_family_a_priors` | off (piatto) |
| A: `Q` alt. | Huang–Wand gerarchico, margini half-t_{ν*} | `q_prior="huang_wand"`, `hw_nu_star=2`, `hw_A=1e5` | off |
| A: `A` | matrix-Normal diffusa centrata su `Â_EM`, precisione `κI` | `family_a_kappa=1e-2` | off |
| A: `Λ, R` | NIG per serie, `a0=2`, `b0=(a0−1)r̂_i`, `m0=L̂_i` | `use_family_a_priors` | off |
| B: `σ_η` | **half-Normal** `N(0,B)` sulla sd con segno (Gelman + ASIS) | `sv_sigma_prior="half_normal"` | `inverse_gamma` (baseline coniugato), forzato a half-Normal se `use_asis` |
| B: `μ` | `μ=0` **identificazione**, non un prior | `sv_fix_mu0=True` | on (i blocchi leverage non hanno intercetta *strutturalmente*) |
| C: `ρ` | Metropolis scalare per canale | — | — |
| D: `ν` | esponenziale (media 20) o uniforme (2,50) | `nu_log_prior` | flat (la griglia griddy è già un uniforme limitato proprio) |

Ogni default riproduce il seam EM **bit-for-bit**: accendere un prior è "un cambio di
argomenti", mai un cambio di kernel.

**Huang–Wand va usato una volta sola**, come prescrive il `.tex`: rifare la stima e
confermare che le correlazioni posteriori tra innovazioni dei fattori non si muovono.
Sul pannello sintetico si muovono di `< 0.05`. **Da rifare sul pannello vero.**

---

## 5. Cosa esce dal sampler (l'interfaccia per il forecast)

```python
res = fit_dfm_mcmc(Y, theta_init, freq_list, block_map, ordered_cols,
                   sv=True, leverage=True, timing="lagged",
                   store_states=True, store_vol=True, ...)
```

`res["draws"]` — **le chiavi seguono la restrizione**: un blocco omesso non produce
chiavi (non produce zeri finti).

| chiave | shape | presente quando |
|---|---|---|
| `A`, `Q` | `(n_keep, r, r)` | sempre |
| `Lambda` | `(n_keep, M, r)` | sempre |
| `R` | `(n_keep, M)` | sempre |
| `loglik` | `(n_keep,)` | sempre |
| `nu_u`, `nu_eps` | `(n_keep,)` | `tails="student_t"` |
| `F` | `(n_keep, T, r)` | `store_states=True` |
| `sv_u` | `(n_keep, r, 3)` — `(μ,φ,σ²)` per fattore | `sv=True` |
| `sv_eps` | `(n_keep, M, 3)` | `sv=True, sv_idio=True` |
| `h_u` | `(n_keep, T, r)` | `sv, store_vol` |
| `h_eps` | `(n_keep, T, M)` | `sv, sv_idio, store_vol` |
| `rho_u` | `(n_keep, r)` | `leverage=True` |
| `rho_eps` | `(n_keep, M)` | `leverage, sv_idio` |
| `hw_a` | `(n_keep, r)` | `q_prior="huang_wand"` |

`res["theta_mean"]` — medie posteriori nello stesso layout di un `theta`.
`res["meta"]` — schedule, dimensioni, seed, **la cella** (`sv, sv_idio, tails,
leverage, timing, q_prior`), wall-clock, e `acceptance` sotto leverage.

### Cosa serve al secondo stadio, e cosa manca

Per simulare la densità predittiva a `T+h` da un draw `d` servono, per ogni `d`:
`A, Q, Λ, R, ν_u, ν_ε`, lo **stato terminale** `F[d, T-1]`, la **volatilità
terminale** `h_u[d, T-1]`, `h_eps[d, T-1]`, i parametri `sv_u, sv_eps` e i `ρ`.
Tutto presente (con `store_states=True, store_vol=True`), e la ricorsione in avanti è:

```
log h_{T+1,k} = φ_k log h_{T,k} + ρ_k σ_k z_{T,k} + σ_k √(1-ρ_k²) ε      (Branch B, laggato)
w_{T+1}       ~ Gamma(ν/2, ν/2)
u_{T+1}       = √H_{T+1} · Q^{1/2} · z_{T+1} / √w_{T+1}                   (Spec II!)
f_{T+1}       = A f_T + u_{T+1}
```

⚠️ **Il sandwich Spec II va rispettato anche in simulazione**: `√H Q^{1/2} z`, non
`Q^{1/2} √H z`. `simulate_sv.simulate_dfm_sv(sv_u_perfactor=...)` è già scritto così
e serve da riferimento.

**Non esiste ancora** un modulo di forecast/GaR: la teoria del forecast non è nel
`.tex` (è tappa futura) e `src/forecast/` consuma oggi l'EM, non i draw MCMC. Lo
schema del dict sopra è **stabile per contratto** proprio per sbloccare quel lavoro.

---

## 6. Quello che il forecast deve sapere prima di partire

**P2 — il costo `√r` della volatilità per-fattore, ed è il rischio numero uno.**
Sotto Spec II ogni `h^u_k` è letta attraverso **un solo** log-quadrato per periodo
(la media su `r` letture simultanee che la restrizione scalare regalava non c'è più).
Misurato sul DGP per-fattore (`T=600, r=3`, `Q` diagonale):

- i `ρ_k` si recuperano **tutti e tre**, ordinamento e segni, su **entrambi** i branch,
  e A e B concordano;
- i **path** `h^u_k` no, e non uniformemente: canali con sd incondizionata di log-vol
  `1.03` e `0.70` → `corr` `0.89` / `0.86`, `φ̂ ≈ 0.95`; un canale con sd `0.46`
  **collassa** (`corr 0.26`, `φ̂ 0.62`; su Branch A non si separa nemmeno dagli altri).

Quindi: **il rischio a `T` corto si concentra sui fattori a bassa volatilità**, e il
leverage sopravvive più a lungo del path. Poiché sono `h` e `ρ` insieme a fissare
scala e asimmetria della predittiva, questo è esattamente ciò che degrada la coda.
Il target è il **nowcasting real-time**, dove `T` effettivo è corto: è il regime
operativo. Mitigazioni da pesare (non ancora decise): fallback alla volatilità comune
scalare a `T` corto; prior più stretti su `σ_η`/`ρ`; pooling parziale dei
`(φ_k,σ²_k,ρ_k)` tra fattori; oppure per-fattore solo oltre una soglia misurata di `T`.

**P4 — disaccoppiato vs calibrazione.** Il sampler disaccoppiato è
*sovra-sicuro* (comprime la volatilità posteriore). Per l'obiettivo densità/GaR il
target **è** la calibrazione, quindi l'unico beneficio del coupling è allineato.
Se accenderlo dipende da (a) quanto vale davvero `corr(Q)` sul pannello reale
(il disaccoppiato è *esatto* a `Q` diagonale) e (b) un check PIT/coverage sul secondo
stadio — **non** dall'accuratezza puntuale, che è ciò che finora è stato misurato.

**Timing A vs B — ancora aperto.** Entrambi implementati e derivati. B mescola molto
meglio (FFBS diretto, niente trappola single-move). Da decidere empiricamente.

**Contesto dal lavoro precedente** che pesa sul secondo stadio: la densità simulata
in avanti dal DFM propaga le code `t` (curtosi alta) ma la **scala** è
regime-invariante ⇒ comprime e manca i crash; la quantile regression si adatta, il
DFM-forward no. La volatilità è prevedibile solo quando NFCI-laggato la segnala
(cattura 2008-09, non il 2020). Da leggere insieme a P2/P4 prima di fidarsi della coda.

---

## 7. La rete di test (tutta verde)

| file | cosa garantisce | n |
|---|---|---|
| `test_shared.py` | ogni kernel vs la sua controparte EM; limite piatto **bit-for-bit**; prior Family A/D; la *costruzione* Huang–Wand (margini half-t + correlazioni uniformi, iterando le sole condizionali senza dati) | 67 |
| `test_passo1.py` | il sampler no-SV (controparte MCMC dell'EM) | 9 |
| `test_passo2.py` | SV base: filtro combinato ≡ filtro standard a `h=1`; FFBS log-vol; recovery e2e | 13 |
| `test_passo3.py` | Branch A: kernel `ρ`, meccanismo di skew, nesting a `ρ=0`, e2e | 8 |
| `test_passo4.py` | Branch B: tabelle Omori (con tolleranza), FFBS, recovery | 13 |
| `test_asis.py` | invarianza del posterior + **ESS(φ) ×2.1**; identità di riscalatura con segno | 14 |
| `test_spec2_recovery.py` | recovery per-fattore `r>1`, `(φ_k,σ²_k)` distinti | 10 |
| `test_variants.py` | la griglia come restrizioni; D1-a per bit-identità; prior su ogni path; HW e2e | 38 |
| `test_perfactor_leverage.py` | separazione dei canali, `ρ_k` distinti, parità A/B, **e la soglia P2 asserita** | 21 |

Il test di P2 asserisce *anche* che il canale debole **non** è identificato a `T=600`:
se un domani una mitigazione sposta la soglia, il test fallisce e ci obbliga a
rileggere P2, invece di lasciar passare in silenzio un miglioramento.
