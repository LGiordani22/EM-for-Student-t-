# Fix P6 — mappa (PASSO 0, sola lettura)

**Obiettivo del blocco.** `ρ` ha `ESS ≈ 3–23` su 2000 draw sotto Branch B. Causa
diagnosticata nell'audit: *blocking* (cresta a posteriori fra `ρ`, il path `log h` e
`σ²_η`), non la scala della proposta. `ρ` è il parametro che dà la **skew** alla
predittiva del PIL, quindi è l'unico problema bloccante per il forecast/GaR.

**Configurazione target, fissa:** Branch B (laggato) + Option A (`ρ_k` scalari) +
Spec II (vol outside). Nessuna modifica a Branch A.

**Contratto.** Additivo: `em_e_step`, `em_m_step`, `kalman` byte-for-byte intatti. Ogni
kernel resta verificato contro la sua controparte nel limite a prior piatto (il *seam*).

Questa mappa è **sola lettura**: nessun sorgente è stato modificato per produrla.
Tutte le righe citate sono verificate sul codice **attuale** (2026-07-10, dopo il commit
`4180044` e la chiusura P1/P3/P4/P5 non ancora committata).

---

## 1. Dove `ρ` viene campionato oggi, sotto Branch B

### Il kernel

```python
# src/mcmc/sample_leverage.py:342
def draw_rho_scalar(rho_cur, eta, k, sigma2, prop_sd, rng):
    n_lev = eta.shape[0]
    rs = rho_cur + prop_sd * rng.standard_normal()          # proposta RW
    cur = _rho_logpost_scalar(rho_cur, eta, k, n_lev, sigma2)
    new = _rho_logpost_scalar(rs,      eta, k, n_lev, sigma2)
    if np.log(rng.random()) < new - cur:
        return float(rs), 1
    return float(rho_cur), 0
```

**Firma.**

| | |
|---|---|
| `rho_cur` | `float` — valore corrente (la proposta ci si àncora: **è** la random-walk) |
| `eta` | `(n_lev,)` — innovazioni AR(1) realizzate `η_t = x_t − φ x_{t−1}`, sulle sole transizioni con leverage |
| `k` | `(n_lev,)` — regressore `k_t = σ_η · g_{t−1}` (sotto B: `g` è il regressore di Omori) |
| `sigma2` | `float` — `σ²_η` corrente |
| `prop_sd` | `float` — sd della proposta RW, **fissa** |
| `rng` | `np.random.Generator` |
| **ritorna** | `(rho_new: float, accepted: 0|1)` |

**Una sola mossa RW per sweep, `prop_sd` fisso, da `ρ=0`.** ✅ **Confermato sul codice
attuale** (l'audit non è sotto-datato):

- valore iniziale: `gibbs.py:402` `rho_u = np.zeros(r)`, `:403` `rho_eps = np.zeros(M)`;
- `prop_sd`: `gibbs.py:220` `lev_prop_rho: float = 0.06`, passato a `:467`
  `prop_rho=lev_prop_rho`;
- una sola invocazione per canale per sweep (nessun ciclo interno di raffinamento).

### I chiamanti nel loop, sotto Branch B

Entrambi in `src/mcmc/sample_leverage_lagged.py` (che importa il kernel a `:94`):

```python
# :404-407   blocco COMUNE, canale k (Option A: r canali scalari indipendenti)
eta_k = lh_k[1:] - phi_k * lh_k[:-1]
lev_k = has_tr_u[1:]
k_reg = (np.sqrt(s2_k) * g_k[:-1])[lev_k]                  # k_t = sigma_k * g_{t-1}
rho_k, a_rk = draw_rho_scalar(rho_k, eta_k[lev_k], k_reg, s2_k, prop_rho, rng)

# :455-458   blocco IDIOSINCRATICO, serie i (K = 1)
eta_i = lh_i[1:] - phi_i * lh_i[:-1]
lev_i = has_tr_i[1:]
k_i = (np.sqrt(s2_i) * g_i[:-1])[lev_i]
rho_i, a_ri = draw_rho_scalar(rho_i, eta_i[lev_i], k_i, s2_i, prop_rho, rng)
```

Entrambi girano **dopo** il draw FFBS del path (`_branch_b_one_process`) e **dopo**
Family B (`_draw_phi_lev`, `_draw_sigma2_lev`) dello stesso canale. Quindi `ρ_k` è
estratto **condizionatamente al path appena estratto**, che a sua volta è stato estratto
**condizionatamente al `ρ_k` precedente**: è esattamente la cresta che causa il blocking.

> Branch A usa lo stesso kernel a `sample_leverage.py:496, :522, :569`. **Non lo tocco.**
> Il flag del PASSO 3 (`rho_sampler`) deve quindi essere threadato in modo da *non*
> alterare il comportamento di default di Branch A.

---

## 2. Il log-posterior di `ρ` — il target che il griddy userà

```python
# src/mcmc/sample_leverage.py:334-339
def _rho_logpost_scalar(rho, eta, k, n_lev, sigma2):
    if abs(rho) >= 1.0:
        return -np.inf
    om = 1.0 - rho * rho
    res = eta - rho * k
    return -0.5 * n_lev * np.log(om) - 0.5 * np.sum(res * res) / (sigma2 * om)
```

**Forma.** Con `η_t | z ~ N(ρ k_t, σ²(1−ρ²))` sulle `n_lev` transizioni con leverage
(`eq:param-rho-cond`), il log-posterior a prior piatto è

```
log p(ρ | ·)  =  −(n_lev/2)·log(1−ρ²)  −  Σ_t (η_t − ρ k_t)² / (2 σ² (1−ρ²))   + cost.
```

**Supporto.** `(−1, 1)`, imposto *dentro* la funzione (`abs(rho) >= 1.0 → −inf`). È il
prior Uniform(−1,1) che il `.tex` dà come default (riga 21542: *"The leverage ρ defaults
to the flat Uniform(−1,1)"*), quindi **proprio** e già normalizzabile su griglia.

**Liscezza.** `C^∞` su `(−1,1)`: composizione di `log(1−ρ²)` e di un polinomio di
secondo grado in `ρ` diviso `(1−ρ²)`. Diverge a `−∞` ai bordi (`log(1−ρ²) → −∞` con
coefficiente `−n_lev/2 < 0`), quindi la massa non tocca mai `±1`.

**Log-concavità.** ⚠️ **NO — e questo è un punto che l'audit dava per scontato e che va
corretto.** Il primo termine è **convesso**:

```
d²/dρ² [ −(n/2)·log(1−ρ²) ]  =  n(1+ρ²)/(1−ρ²)²  >  0        (verificato numericamente)
```

quindi *a priori* nulla garantisce la concavità del log-posterior. Verificato su 400
configurazioni casuali di `(n_lev, σ², scala di η e k)`: **3 sono non concave** in
almeno un punto (0.8%) e **1 è bimodale**. Nel regime tipico (`n_lev ≈ T` grande, dati
informativi) il target è concavo e unimodale, ma **non lo è per costruzione**.

Il che è **irrilevante per il griddy**, e va detto esplicitamente perché la
giustificazione è *diversa* da quella di `ν`:

- per `ν`, la log-concavità (risultato della tesi) giustifica che *una griglia
  grossolana basta* su un supporto **illimitato** `(2, ∞)`;
- per `ρ`, la giustificazione è più semplice e più forte: **il supporto è compatto**,
  `(−1,1)`. Una griglia uniforme fine (400–800 punti) risolve qualunque forma —
  unimodale, bimodale o asimmetrica — perché non c'è coda da inseguire.

⚠️ **Da scrivere così nel `.tex`, non copiando l'argomento di `ν`.** Sarebbe un errore
sostanziale affermare che il target di `ρ` è log-concavo.

**Riusabile as-is?** ✅ **Sì, senza modifiche.** È già vettorizzabile su `ρ` con un
`np.vectorize`/list-comp (come fa `draw_nu_griddy` con `nu_log_target`), e il costo per
punto è `O(n_lev)`. Con `grid_size=400` e `n_lev ≈ T`, il costo per canale per sweep è
`400·T` flop — dello stesso ordine dell'FFBS che lo precede. Accettabile; se servisse,
si vettorizza in una sola operazione matriciale `(grid, n_lev)`.

---

## 3. Il pattern griddy esistente — `draw_nu_griddy`

```python
# src/mcmc/sample_params.py:224-252
w = np.asarray(weights, dtype=float)
sum_w = float(np.sum(w)); sum_log_w = float(np.sum(np.log(w)))
lo, hi = nu_bounds
grid = np.geomspace(lo, hi, grid_size)                       # (a) griglia GEOMETRICA
logp = np.array([nu_log_target(nu, sum_log_w, sum_w, n, log_prior) for nu in grid])
logp -= logp.max()                                           # (b) stabilizzazione
dens = np.exp(logp)
cell = np.gradient(grid)                                     # (c) peso = ampiezza cella
probs = dens * cell
total = probs.sum()
if not np.isfinite(total) or total <= 0:
    return float(grid[grid_size // 2])                       # (d) guardia degenere
probs /= total
return float(rng.choice(grid, p=probs))                      # (e) draw discreto
```

### Cosa è riusabile e cosa va adattato

| elemento | riusabile? | adattamento per `ρ` |
|---|---|---|
| **(b)** stabilizzazione `logp −= logp.max()` | ✅ tal quale | — |
| **(c)** peso trapezoidale `np.gradient(grid)` | ✅ tal quale | su griglia **uniforme** `np.gradient` è costante ⇒ i pesi si semplificano, ma tenerlo mantiene il pattern identico e la formula corretta se un domani la griglia diventasse non uniforme |
| **(d)** guardia su `total` non finito / ≤ 0 | ✅ tal quale | fallback: `ρ = 0` (il punto centrale della griglia), che è anche il valore "nessun leverage" — semanticamente giusto |
| **(e)** `rng.choice(grid, p=probs)` | ✅ tal quale | **consuma la stessa quantità di RNG** di prima? No: cambia lo stream. Vedi §6. |
| **(a)** griglia `np.geomspace(lo, hi)` | ❌ **da sostituire** | `ν` vive su `(2, 1000)` con code lunghe ⇒ geometrica. `ρ` vive su `(−1, 1)`, **compatto e simmetrico attorno a 0** ⇒ **`np.linspace(-1+ε, 1-ε, grid_size)`**, uniforme |
| bordi | — | `_rho_logpost_scalar` dà `−inf` a `|ρ| ≥ 1`; la griglia deve **escludere** gli estremi (`ε ≈ 1e-6`), altrimenti `exp(−inf)=0` è innocuo ma inutile |
| `log_prior` hook | ✅ da replicare | permette lo shrinkage Fisher-`z` del `.tex` (riga 21544) **senza toccare il kernel**, esattamente come `nu_log_prior_*` |

**Nota importante sulla firma.** `draw_rho_scalar` ritorna `(rho, accepted)`. Un griddy
**non ha accettazione**: ogni draw è accettato per costruzione. I chiamanti sommano
`a_rk` in `acc["rho_u"]`, e `test_passo3`/`test_passo4` asseriscono
`0.05 < acc["rho_u"] < 0.95`. ⚠️ **Se il griddy ritornasse `1`, quei test fallirebbero**
(`acc = 1.0 > 0.95`). Il draw FFBS del path ha lo stesso problema e lo risolve
dichiarando `acc["path_u"] = 1.0` e testando `== 1.0`. Serve la stessa scelta esplicita
per `ρ`, e i test vanno adattati di conseguenza — **è un punto di rottura da gestire nel
PASSO 3, non una sorpresa da scoprire a valle.**

---

## 4. Agganci di diagnostica (da implementare al PASSO 1, non ora)

### Dove vivono

| funzione | file:riga | firma | ritorna |
|---|---|---|---|
| `split_r_hat` | `diagnostics.py:31` | `(chains: (n_chains, n_draws))` | `float` — split-R̂ di Gelman-Rubin; **una singola catena lunga basta** (la divide in due) |
| `ess` | `diagnostics.py:56` | `(chains: (n_chains, n_draws))` | `float` — ESS totale, autocorrelazione via FFT + sequenza positiva iniziale di Geyer |
| `diagnostics_table` | `diagnostics.py:96` | `(per_chain: dict[str, (n_chains, n_draws)])` | `dict[str, {"r_hat", "ess"}]` |

Entrambe accettano `(n_chains, n_draws)`: da una singola run di `fit_dfm_mcmc` si passa
`draws[k][None, :]` (una catena).

### Dove agganciarle in `fit_dfm_mcmc`

Il punto è **uno solo**, e sta dopo la chiusura del loop e dopo il trim delle storage,
prima della costruzione di `meta`:

```
gibbs.py:576   for k in list(draws):            # trim di n_keep
gibbs.py:579   theta_mean = { ... }
      ^^^^     <-- QUI: calcolare le diagnostiche dai draw già trimmati
gibbs.py:598   meta = { ... }
gibbs.py:610   return {"draws": draws, "theta_mean": theta_mean, "meta": meta}
```

**Chiavi da diagnosticare** (solo quelle presenti — lo schema dei draw segue la
restrizione, vedi `MCMC_MAP.md` §3):

| quantità | sorgente | scalari da estrarre |
|---|---|---|
| `ρ_u` | `draws["rho_u"]` `(n_keep, r)` — solo se `leverage` | `r` canali |
| `ρ_ε` | `draws["rho_eps"]` `(n_keep, M)` — solo se `leverage and sv_idio` | `M` serie (o solo un riassunto: min/mediana ESS) |
| `φ_u`, `σ²_u` | `draws["sv_u"]` `(n_keep, r, 3)`, colonne `1` e `2` — solo se `sv` | `2r` |
| `φ_ε`, `σ²_ε` | `draws["sv_eps"]` `(n_keep, M, 3)` — solo se `sv and sv_idio` | `2M` (riassunto) |

**Dove metterlo nel risultato.** `res["diagnostics"]` (chiave nuova, non `meta`): `meta`
è metadati della run, le diagnostiche sono *risultati*. Struttura proposta:

```python
res["diagnostics"] = {
    "rho_u":   {"ess": (r,), "r_hat": (r,)},
    "sv_u_phi":  {...}, "sv_u_sigma2": {...},
    "rho_eps": {"ess_min": float, "ess_median": float, "r_hat_max": float},   # riassunto
    ...
}
```

⚠️ **Vincolo del GATE 1**: la diagnostica **legge** `draws`, non li altera, e **non
consuma RNG**. Quindi a parità di seed i draw devono essere **byte-for-byte identici**
prima e dopo. Va reso un test, non un'asserzione.

⚠️ **Costo**: `ess` fa una FFT per catena per scalare. Con `M ≈ 20` serie e `n_keep`
grande, `2M + 2r + r + M` scalari sono ~80 FFT: trascurabile rispetto a una run. Ma per
`ρ_ε`/`sv_eps` conviene riportare un **riassunto** (min/mediana) invece di `M` vettori,
per non gonfiare il dict.

---

## 5. I valori di `ρ̂` sotto-convergiuti — la lista (da correggere al PASSO 4)

Regola nuova da introdurre: **nessun `ρ̂` senza il suo ESS.**

Tutte le run che li hanno prodotti usano `n_iter ≤ 900`. Per riferimento, l'audit misura
`ESS(ρ) = [21, 13, 20]` su **1500** draw tenuti (`n_iter=2500, burn=1000`), quindi a
`n_iter ≤ 900` l'ESS è nell'ordine di **pochi draw effettivi**.

| # | file:riga | valore citato | `n_iter` della run | stato |
|---|---|---|---|---|
| 1 | `REFACTOR_PLAN.md:333-334` | *"at `T=900` per-factor `ρ` recovered `[-0.51,-0.18,-0.20]` vs true `[-0.6,-0.3,-0.2]`"* | ignoto, ma la run di Phase 7 usava `n_iter ≤ 900` | ⚠️ **da rigenerare** |
| 2 | `REFACTOR_PLAN.md:398-401` (blocco P2 quantificato) | *"i `ρ_k` sono tutti recuperati — ordinamento, segno dominante e canale positivo — su entrambi i branch"* | `n_iter=700` (`test_perfactor_leverage`) | ⚠️ **affermazione qualitativa**: l'ordinamento regge, ma va ri-verificato con ESS |
| 3 | `REFACTOR_PLAN.md:629` | *"`ρ_k` *is* still recovered on the weak channel (sign and ordering)"* | `n_iter=600` | ⚠️ **da riverificare** (nell'audit il canale debole cambia segno fra finestre) |
| 4 | `MCMC_MAP.md:385` | Branch A a `T=1200`: *"`ρ̂` collassa a `[-0.18, +0.06, +0.09]`"* | `n_iter=600` | ⚠️ **Branch A**: non lo tocchiamo, ma il numero va marcato come "a ESS ignoto" |
| 5 | `audit_P1-P5.md:593-596` (tabella `lev_prop_rho`) | `[−0.49,−0.30,+0.39]`, `[−0.49,−0.10,+0.41]`, `[−0.44,−0.18,+0.43]` | `n_iter=2500, burn=1000` | ✅ **già accompagnati dal loro ESS** — questi restano, sono il baseline |
| 6 | `audit_P1-P5.md` §P2 (tabella `T=600/1200/2400`, colonna `ρ̂`) | `[-0.44,-0.34,0.41]` / `[-0.50,-0.17,0.37]` / `[-0.49,-0.03,0.34]` | `n_iter=600` | ⚠️ **da rigenerare** (le colonne `corr(h)` e `φ̂` restano valide: non dipendono da `ρ`) |
| 7 | `test_perfactor_leverage.py:77` `RHO_TRUE` | `[-0.70,-0.15,0.45]` | — | ✅ è il **vero** del DGP, non una stima |
| 8 | `test_passo3.py:116`, `test_passo4.py:141` `rho_u_true` | `[-0.6,-0.3,-0.2]` | — | ✅ vero del DGP |

**Nessun valore numerico di `ρ̂` è hard-coded come soglia nei test**: i test asseriscono
*segno*, *ordinamento* e *acceptance*, mai un valore. ✅ Buona notizia — le correzioni
del PASSO 4 sono confinate ai documenti, non alla suite.

⚠️ **Eccezione da tenere d'occhio**: `test_passo3.py:77` asserisce
`abs(mean - rho_true) < 0.07` sul **kernel isolato** `draw_rho_scalar` (4000 iterazioni
su regressore esatto). Quel test misura il *kernel*, non il sampler, ed è passato: il
kernel non è distorto. Se il griddy sostituisce il kernel, quel test è il primo seam di
correttezza da far restare verde.

---

## 6. Rischi noti prima di iniziare

1. **Lo stream RNG cambia.** Sostituire una `standard_normal()` + `random()` con una
   `rng.choice(...)` altera la sequenza di numeri casuali consumati. Quindi **le run con
   leverage non saranno bit-identiche** al pre-fix, e non devono esserlo — cambia il
   kernel. Ma le run **senza** leverage (`test_passo1`, `test_passo2`, `test_variants`
   no-SV/spec2, `test_shared`) devono restare **bit-for-bit identiche**: il seam EM non
   si tocca. Da verificare esplicitamente.

2. **`acc["rho_u"]` perde significato.** Vedi §3. Va deciso, non subìto.

3. **La harness di recovery non supporta il DGP per-fattore.**
   `diagnostics.run_recovery_mcmc_leverage` (`:734`) accetta `sv_u` come **tupla**
   `(μ,φ,σ)` e la passa a `simulate_dfm_sv(sv_u=sv_u, ...)` (`:786`): genera quindi il
   DGP **scalar-common** `H = h·I` con `ρ` vettoriale, non il per-fattore Spec II.
   Per il PASSO 2 (baseline sul *DGP per-fattore vero*) servirà o (a) un passthrough
   `sv_u_perfactor` nella harness — additivo, ~4 righe — o (b) uno script di misura
   dedicato. **Preferisco (a)**: la harness è il posto giusto e serve anche al GATE 3.

4. **Il costo per sweep sale.** Il griddy valuta il target su `grid_size` punti contro i
   2 dell'RW. Con `grid_size=400`, `r=3` canali + `M≈20` serie, sono `~9200` valutazioni
   `O(n_lev)` per sweep. Va misurato: se il wall-clock raddoppia, si vettorizza il target
   su griglia (una `(grid, n_lev)` matrice, un `einsum`), che riporta il costo a ~1×.

---

## 7. Baseline

*(Da compilare al PASSO 2, dopo il GATE 1. Sezione riservata.)*

| | `ESS(ρ_k)` | `split-R̂(ρ_k)` | `ρ̂_k` [CI 90%] |
|---|---|---|---|
| RW (baseline) | — | — | — |
| griddy | — | — | — |

DGP: per-fattore Spec II, `r=3`, `Q` diagonale, `ρ_vero = [−0.70, −0.15, +0.45]`,
`sv_u = [(0,.97,.25), (0,.92,.18), (0,.95,.22)]`, Branch B (laggato).

---

## GATE 0 — stato

✅ Mappa completa. **Nessun sorgente modificato.**

Quattro cose che la mappa ha trovato e che l'audit non diceva, e che cambiano il PASSO 3:

1. **Il target di `ρ` NON è log-concavo** — il termine `−(n/2)log(1−ρ²)` è convesso, e
   su 400 configurazioni casuali lo 0.8% risulta non concavo, una bimodale. L'audit
   scriveva "1-D su `(−1,1)`, liscio" lasciando intendere la log-concavità di `ν`. Non
   serve che lo sia (supporto **compatto**), ma la giustificazione va scritta *diversa*
   nel `.tex`, non copiata da quella di `ν`.
2. **`acc["rho_u"]` diventa privo di significato** con un griddy (ogni draw è accettato),
   e `test_passo3`/`test_passo4` lo asseriscono in `(0.05, 0.95)`. Punto di rottura noto:
   va deciso come il path FFBS di Branch B, che dichiara `acc = 1.0` e testa `== 1.0`.
3. **La harness di recovery non genera il DGP per-fattore** (`diagnostics.py:786` passa
   `sv_u` scalare a `simulate_dfm_sv`): il PASSO 2 richiede un passthrough
   `sv_u_perfactor`, ~4 righe additive.
4. **Lo stream RNG delle run con leverage cambierà** (`rng.choice` al posto di
   `standard_normal`+`random`). È inevitabile e corretto — ma le run **senza** leverage
   devono restare bit-for-bit identiche, e va verificato, non assunto.

**In attesa di conferma per procedere al PASSO 1.**
