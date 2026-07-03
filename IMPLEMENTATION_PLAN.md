# Mappa di implementazione — Sampler MCMC (Gibbs) per il DFM Student-t con stochastic volatility e leverage

**Cella master: D1-b × D2-b** (Student-t su fattore + idiosincratici, SV su tutti gli M+1 processi, leverage), con **entrambi i timing del leverage** (contemporaneo e lagged) selezionabili da un flag.

> Questo è un **piano**, non codice. Definisce moduli, ordine, riuso, e test.
> Principio guida: **conservare modello + Kalman, riscrivere solo il motore di stima.**
> Riferimento teorico: `docs/EM_for_student_t.tex`, sezioni Gibbs step (a)–(d), "Two Routes for the Leverage", e la mixed-frequency / regressore composito $\phi_t$.

---

## 0. Sintesi della ricognizione del repository

Il repo è già strutturato in modo ideale per questa estensione: **il modello e tutto lo state-space (Kalman/smoother, mixed-frequency, selection matrix, simulatore, harness di recovery) sono separati dal motore EM**. Il sampler MCMC riusa il primo gruppo e sostituisce solo il secondo.

### 1.A — Riusabili COSÌ COME SONO (chiamabili da un nuovo orchestratore, zero modifiche)

| Funzione / modulo | File | Ruolo nel sampler |
|---|---|---|
| `build_A_tilde`, `build_Lambda_tilde` | `src/kalman.py` | Companion $\tilde{\mathbf A}$ e loading $\tilde{\mathbf L}$ con pesi MM — invarianti, identici per EM e MCMC |
| `build_Q_tilde(Q, w_u_t)`, `build_R_tilde(R, w_eps_t)` | `src/kalman.py` | **Riuso chiave per la SV**: accettano già un peso scalare. Passando il **peso combinato** `w/h` (precisione $g_t = w_t/h_t$) producono la covarianza $\tilde{\mathbf Q}_t = (h^u_t/w^u_t)\mathbf Q$ senza modifiche |
| `build_selection_matrix`, `build_all_selection_matrices` | `src/kalman.py` | Maschera $\mathbf W_t$ (ragged + mask trimestrale) — invariata |
| `kalman_predict`, `kalman_update`, `kalman_filter` | `src/kalman.py` | **Forward pass del simulation smoother**: si chiama identico, alimentato con `Q_tilde_t`/`R_tilde_t` tempo-varianti |
| `kalman_smoother` | `src/kalman.py` | Serve per il sampler Durbin–Koopman (alternativa al FFBS) e per i valori iniziali |
| `compute_weighted_moments` | `src/em_m_step.py` | **Momenti sufficienti** $\mathcal P_{11},\mathcal P_{10},\mathcal P_{00}$ e momenti del loading — base sia per il punto M-step sia per il **draw** coniugato |
| `simulate_factors`, `simulate_observations`, `apply_missing_pattern`, `simulate_dfm` | `src/simulate_dfm.py` | Generano il panel Student-t + mixed-frequency. Riusabili per il livello base; **da estendere** per SV+leverage (vedi §6) |
| `fit_dfm`, `load_dfm_fit` | `src/em_main.py` | Producono $\theta^{(0)}$ + percorsi smoothed per **inizializzare** il sampler (caldo, non da PCA) |
| `init_theta_from_synthetic`, pattern di `run_recovery` | `src/monte_carlo_recovery.py` | Template dell'**harness di recovery** (simula → init → re-fit → score) da clonare per l'MCMC |
| `data_loader`, `config_utils` | `src/` | Caricamento panel/config (small/big) — invariati |

### 1.B — "Incastrate" nell'EM, da ESTRARRE in helper condivisi (piccolo refactoring)

| Logica esistente | Dove vive ora | Perché va estratta | Cosa estrarre |
|---|---|---|---|
| **Residuo di Mahalanobis** $d^\varepsilon_t, d^u_t$ | `compute_d_eps`, `compute_d_u` (`em_e_step.py`) | Calcolano il $d$ **atteso** (col termine di traccia su `P_smooth`) e dividono per `R`, non per `h`. Il sampler vuole il $d$ **realizzato** dallo stato campionato e **deflazionato da $h_t$** | Helper `realized_deflated_d(states_sampled, Lambda_tilde, R, h_path, W_list)` (il trucco `P=0` annulla la traccia; aggiungere la deflazione per-serie da $h$) |
| **Draw dei pesi** $w$ | `compute_weights` (`em_e_step.py`) ritorna la **media** Gamma | Il sampler vuole un **campione** da `Gamma((nu+m)/2,(nu+d)/2)` | Helper `draw_weights(d_eps,d_u,m_obs,nu_eps,nu_u,r)` che riusa la stessa parametrizzazione ma estrae invece di mediare |
| **FOC / log-target di $\nu$** | dentro `update_nu` (`em_m_step.py`), funzione interna `g(nu)` | Serve per il **draw** di $\nu$ (griddy-Gibbs o Metropolis), con statistiche sufficienti dai pesi **campionati** | Estrarre `nu_log_target(nu, sum_log_w, sum_w, T, prior)` e/o `nu_foc(nu, ...)` come funzioni pubbliche riusabili |
| **Completamento del quadrato / draw coniugati** | dentro `update_A_Q`, `update_Lambda`, `update_R` (ritornano il **punto**) | Il sampler vuole il **draw** dalle stesse posteriori (MNIW per $\mathbf A,\mathbf Q$; NIG per $\mathbf L,r$). I momenti sono già in `compute_weighted_moments` | Aggiungere `draw_A_Q(moments, prior)`, `draw_Lambda_R(moments, prior)` che riusano i momenti e campionano invece di risolvere |
| **Regressore composito $\phi_t$ (trimestrale)** | dentro la M-step mixed-frequency di `update_Lambda` | Il draw del loading trimestrale usa $\phi_t$ (media MM dei 5 blocchi) — stessa algebra, draw invece di punto | Estrarre `composite_regressor(states_aug, mm_weights)` → $\phi_{1:T}$ |

**Conclusione del refactoring**: nessuna riscrittura, solo **estrazione di ~5 helper** da codice già scritto e testato. Il rischio è basso perché ogni helper è verificabile contro la sua controparte EM (stessa formula, draw vs media/punto).

---

## 1. Architettura — CONFERMATA con precisazioni

**La scelta proposta è corretta.** Orchestratore MCMC in **moduli NUOVI e separati dall'EM**, che **importano e chiamano** le funzioni esistenti (Kalman, smoother, mixed-frequency, build matrici, momenti, pesi) — **non le riscrivono**.

Precisazioni:

1. **Nessuna modifica ai file EM esistenti.** Gli helper di §1.B si aggiungono come funzioni nuove (in `kalman.py` / `em_m_step.py` o in un nuovo `src/mcmc/shared.py`), lasciando intatte le firme attuali. L'EM continua a funzionare identico (vincolo: i test EM/recovery esistenti devono restare verdi dopo l'estrazione — è il primo gate).
2. **Il riuso della SV passa per il peso combinato.** Non serve un nuovo `build_Q_tilde`: il sampler passa $g_t = w_t/h_t$ dove l'EM passava $w_t$. Questo è il singolo aggancio che rende lo state-space SV "gratis".
3. **Confine netto stato-vs-stima.** Tutto ciò che è "modello + filtro" resta in `kalman.py`/`simulate_dfm.py`; tutto ciò che è "draw da posteriori" vive nel nuovo package `src/mcmc/`.

```
src/mcmc/
  gibbs.py            # orchestratore (loop, burn-in, thinning, storage)
  sample_states.py    # step (a): FFBS (riusa kalman_filter) o Durbin-Koopman
  sample_vol.py       # step (b): KSC mixture — NUOVO, biforcato per timing
  sample_weights.py   # step (c): draw Gamma (riusa logica Student-t)
  sample_params.py    # step (d): A,Q / Lambda,R / log-vol AR(1) / leverage / nu
  shared.py           # helper estratti da §1.B + costanti KSC-7, Omori-10
  diagnostics.py      # R-hat, ESS, trace, recovery scoring
  constants.py        # tabelle KSC-7 e Omori-10 (inserite dall'utente)
```

---

## 2. I moduli (nuovi vs riusati) e il loro ruolo

### Orchestratore Gibbs — `src/mcmc/gibbs.py` (NUOVO)
- `fit_dfm_mcmc(Y, theta_init, freq_list, *, timing, n_iter, burn_in, thin, sv=True, leverage=True, seed, priors)`.
- **Init caldo da EM**: `theta_init` da `fit_dfm`/`load_dfm_fit`; percorsi $h$ inizializzati a 1 (log $h=0$), pesi $w=1$, leverage $\rho=0$.
- **Ciclo sweep**: (a) stati → (b) vol → (c) pesi → (d) parametri, in quest'ordine; gestione burn-in, thinning, storage dei draw (memmap o `.npz` a blocchi per non saturare la RAM su $5r\cdot T$).
- **Flag**: `timing ∈ {"contemporaneous","lagged"}`, `sv`, `leverage` (per abilitare le versioni ridotte usate nei test incrementali §4).

### (a) Sampler stati $f_{1:T}$ — `src/mcmc/sample_states.py` (NUOVO sottile)
- **Riusa** `kalman_filter` per il forward pass (alimentato da `Q_tilde_t = build_Q_tilde(Q, w/h)`).
- **Aggiunge** il **backward sampling** FFBS (eq. `ffbs-backward`): gain $J_t$ già identico allo smoother gain.
- **Caveat companion singolare** (già documentato in `kalman_smoother`): si campiona solo il blocco $r$-dimensionale di testa; i 4 blocchi lag si leggono da $\tilde f_{t+1}$.
- **Alternativa Durbin–Koopman**: riusa `kalman_smoother` due volte (dati reali + pseudo-dati) — preferibile a $5r$ grande. Entrambe selezionabili.

### (b) Sampler volatilità $h_{1:T}$ — `src/mcmc/sample_vol.py` (NUOVO — il pezzo principale)
- KSC (Kim–Shephard–Chib): trasformazione log-quadratica $y^*_t=\log e_t^2$, mistura a 7 componenti, FFBS sul sub-state-space lineare-gaussiano della log-vol.
- **Biforcazione leverage** (vedi §3): drift sulla transizione AR(1) della log-vol.
- Eseguito $M+1$ volte (fattore comune + $M$ idiosincratici), ciascuno scalare.

### (c) Sampler pesi $w$ — `src/mcmc/sample_weights.py` (NUOVO sottile)
- **Riusa** l'helper `realized_deflated_d` (§1.B) per $\check d^\varepsilon_t,\check d^u_t$ (residuo deflazionato da $h$).
- **Draw** `Gamma((nu+m)/2, (nu+\check d)/2)` al posto della media — un'unica sostituzione rispetto alla logica E-step esistente.
- Default **separabile**: il leverage non tocca la posteriori del peso (resta coniugata). Opzione "accoppiamento esatto" (Metropolis sul kernel $\propto\exp(A\sqrt w)$) prevista ma non default — vedi tesi, sezione weight-axis.

### (d) Sampler parametri — `src/mcmc/sample_params.py` (NUOVO, misto riuso)
- **Family A** $\mathbf A,\mathbf Q$: **riusa** `compute_weighted_moments` (con $g_t=w_t/h_t$) → **draw MNIW** (nuovo, ~10 righe).
- **Family A** $\mathbf L,r$: riusa i momenti pesati → **draw NIG**; per le serie **trimestrali** usa il **regressore composito $\phi_t$** (helper `composite_regressor`) — esattamente come la M-step mixed-frequency, ma draw.
- **Family B** log-vol AR(1) $(\mu,\phi,\sigma_\eta^2)$: regressione NIG sulla path di log-vol campionata (nuovo, ma è la stessa macchina NIG di Family A); vincolo $|\phi|<1$.
- **Family C** leverage $\rho,\rho_{\varepsilon,i}$: **Metropolis** sul full-conditional non standard (prefattore $(1-\rho^2)^{-(T-1)/2}$) — **biforca per timing** (vedi §3); vincolo $|\rho|<1$, $\boldsymbol\rho'\boldsymbol\rho<1$.
- **Family D** $\nu_u,\nu_\varepsilon$: **riusa** `nu_log_target`/`nu_foc` estratti → **griddy-Gibbs** (sfrutta la log-concavità dimostrata in tesi) o Metropolis, con statistiche dai pesi campionati.

---

## 3. I DUE TIMING — architettura

I due timing **differiscono SOLO nel blocco volatilità (b) e nel draw del leverage (Family C)**. Tutto il resto (stati, pesi, $\mathbf A,\mathbf Q,\mathbf L,r$, $\nu$) è **identico**.

**Pattern: un'unica interfaccia, due implementazioni selezionate da `timing`.**

```
sample_vol.py
  def sample_volatility_block(..., timing, leverage):
      if not leverage:                 -> _ksc_base(...)              # mistura 7, nessun drift
      elif timing == "contemporaneous":-> _ksc_leverage_metropolis(...)  # Ramo A
      elif timing == "lagged":         -> _ksc_leverage_omori(...)        # Ramo B
```

- **Ramo A (contemporaneo, Metropolis)** — *implementabile da principi primi*, **nessuna costante Omori**. La drift $\rho\sigma_\eta z_t$ dipende dallo stato $h_t$ campionato → target valutato per densità, mossa single-move Metropolis sulla log-vol. Family C: stessa macchina Metropolis.
- **Ramo B (lagged, Omori)** — richiede le **costanti tabulate**: mistura **a 10 componenti** (Omori–Chib–Shephard–Nakajima 2007) + indicatore di **segno** $d_t$. La drift cade sulla transizione $t\to t+1$, lasciando il sub-state-space lineare-gaussiano → FFBS sopravvive. Family C: il condizionamento su segni e indicatori rende il mean lineare in $\rho$ → mossa Metropolis quasi-gaussiana ad alta accettazione.

La tabella delle costanti (KSC-7 e Omori-10) vive in `constants.py`, **inserita dall'utente dalla fonte**; il Ramo B è gated su di esse (vedi §7).

---

## 4. Ordine di implementazione (dal più sicuro al più rischioso) + test a ogni passo

> Ogni passo ha un **recovery test** dedicato (§5). I bug MCMC sono silenziosi: si avanza solo a gate verde.

**Passo 0 — Refactoring estrattivo (gate di non-regressione).**
Estrarre i ~5 helper di §1.B. *Test*: la suite EM/recovery esistente (`run_recovery`, `test_monte_carlo_engine`) resta **bit-identica** → l'estrazione non ha cambiato nulla.

**Passo 1 — Orchestratore + blocchi "amici" SENZA SV (= versione MCMC dell'EM).**
Stati (FFBS) + pesi (Gamma draw) + parametri (A,Q,L,r,ν), $h\equiv1$, $\rho\equiv0$.
*Test (recovery)*: simulare da $\theta$ Student-t noto (simulatore attuale) → il sampler **recupera $\theta$**? Le medie posteriori di $\mathbf A,\mathbf Q,\mathbf L,r,\nu$ coprono il vero entro gli intervalli credibili; le path dei fattori coprono $F_{true}$. **Confronto incrociato**: media posteriori MCMC ≈ stima EM sullo stesso panel.

**Passo 2 — Blocco vol base (KSC senza leverage).**
Aggiungere `_ksc_base`; abilitare `sv=True`, `leverage=False`.
*Test*: estendere il simulatore con **path di vol dinamica** (§6, log-AR(1), $\rho=0$) → il sampler **recupera i percorsi $h_{1:T}$** (correlazione path stimata vs vera; recupero di $\mu,\phi,\sigma_\eta$ del processo log-vol) e ancora $\theta$.

**Passo 3 — Leverage Ramo A (contemporaneo + Metropolis).**
*Implementabile per primo perché non richiede costanti Omori.* Aggiungere `_ksc_leverage_metropolis` + Family C Metropolis.
*Test*: simulatore con **leverage** (§6) e $\rho<0$ → recupero di $\rho$ (comune e idiosincratico) con segno e magnitudine; le path di vol mostrano l'asimmetria; accettazione Metropolis in range ragionevole (tuning della proposta).

**Passo 4 — Leverage Ramo B (lagged + Omori).**
Richiede le costanti **KSC-7 + Omori-10** inserite. Aggiungere `_ksc_leverage_omori` + segno + Family C versione lagged.
*Test*: simulare con **timing lagged** → recupero di $\rho$ e $h$; **coerenza incrociata** (su dati lagged, Ramo B ≈ oracle; mixing migliore/senza tuning rispetto al Ramo A); su dati contemporanei i due rami divergono come atteso.

---

## 5. Infrastruttura di test — il recovery come strumento centrale

**Principio**: i bug MCMC non crashano, producono posteriori sottilmente sbagliate. L'unico controllo affidabile è il **self-recovery**: genero da parametri noti, il sampler deve recuperarli.

**Struttura (clonando il pattern `run_recovery`):**
- `src/mcmc/diagnostics.py :: run_recovery_mcmc(theta_true, *, T, seed, timing, sv, leverage, n_iter, burn_in)`:
  1. **Simula** panel + ground truth ($F$, $w$, **$h$**, **$\rho$** quando attivi) dal simulatore esteso.
  2. **Init** da EM (caldo) o da PCA (onesto) sul panel sintetico.
  3. **Re-fit** col sampler (`save_path` separato, mai sovrascrivere la cache reale).
  4. **Score** a 3 livelli:
     - *Livello 1 — parametri*: copertura del vero negli intervalli credibili (al netto dell'indeterminatezza rotazionale: riusare `procrustes_*`/`align_sign_per_factor` da `monte_carlo_recovery.py`).
     - *Livello 2 — stati latenti*: correlazione path $\hat F$ vs $F_{true}$, $\hat h$ vs $h_{true}$, $\hat w$ vs $w_{true}$.
     - *Livello 3 — diagnostica MCMC*: R-hat (multi-catena), ESS, trace plot — vedi §7.
- **Toggle progressivi**: lo stesso harness con `sv`/`leverage`/`timing` copre i Passi 1–4. Recovery con tutto spento = test del Passo 1; si accende un pezzo per volta.

**Estensione del simulatore** (`src/simulate_dfm.py`, additiva e backward-compatible — `sv=False,leverage=False` deve restare bit-identico all'attuale):
- generare le **path di log-vol** $\log h_t$ come AR(1) ($\mu,\phi,\sigma_\eta$) per il fattore e per gli $M$ idiosincratici;
- accoppiare shock di livello e innovazione log-vol con correlazione $\rho$ (leverage), **per entrambi i timing** (contemporaneo: $z_t\!\leftrightarrow\!\eta_t$; lagged: $z_t\!\leftrightarrow\!\eta_{t+1}$);
- restituire $h_{true},\rho_{true}$ nel ground truth.

---

## 6. Dipendenze, rischi, diagnostica

**Dipendenze esterne / dati:**
- **Costanti tabulate** (le inserisce l'utente dalla fonte): **KSC-7** (mistura a 7 componenti, log-$\chi^2_1$) e **Omori-10** (mistura a 10 componenti per la SV con leverage lagged, Tab. 1 di Omori et al. 2007). Già riportate e verificate cifra-per-cifra in `docs/EM_for_student_t.tex` (`tab:omori-mixture`) — da trascrivere in `src/mcmc/constants.py` con i medesimi controlli di coerenza ($\sum q_j=1$, $\sum q_j m_j\approx-1.2704$, $b_j=a_j/2$).
- Il **Ramo B è gated** su queste tabelle; il **Ramo A no** (principi primi) → si implementa e testa per primo.

**Punti più delicati (massima attenzione + test mirati):**
1. **Sampler della volatilità (b)** — il blocco più tecnico; bug nella mistura/segno producono path plausibili ma sbagliate → recovery su $h$ obbligatorio.
2. **Leverage (Family C + drift)** — full-conditional non standard, vincoli $|\rho|<1$; segno scartato dalla mappa log-quadratica (Ramo B) → test su recupero di $\rho$ con segno.
3. **Companion singolare nel FFBS** — campionare solo il blocco di testa (già gestito nello smoother).
4. **Indeterminatezza rotazionale** nello scoring — riusare le routine Procrustes esistenti.
5. **Deflazione per $h$** nei pesi e nei momenti dei parametri — coerenza $g_t=w_t/h_t$ ovunque (un solo punto di verità).

**Diagnostica di convergenza (da prevedere in `diagnostics.py`):**
- **R-hat** (Gelman–Rubin) su più catene con seed diversi, per i parametri scalari chiave ($\mathbf A,\mathbf Q,\nu,\mu,\phi,\sigma_\eta,\rho$);
- **ESS** (effective sample size) per quantificare l'autocorrelazione dei draw (il blocco vol e il leverage mixano lentamente);
- **trace plot** + running-mean per ispezione visiva del burn-in;
- regola pratica: scartare burn-in finché R-hat $<1.1$ e ESS sufficiente prima di leggere le posteriori.

---

## 7. Cosa si riusa vs cosa è nuovo — quadro finale

| Blocco | Riuso | Nuovo |
|---|---|---|
| State-space (companion, $\tilde{\mathbf L}$, $\mathbf W_t$, SV via peso combinato) | **tutto** `kalman.py` | — |
| Stati (a) | forward `kalman_filter` / `kalman_smoother` | backward sampling (FFBS) o Durbin–Koopman |
| Volatilità (b) | — | **KSC mixture**, biforcata per timing (il pezzo principale) |
| Pesi (c) | logica residuo/Gamma E-step | draw al posto della media; deflazione per $h$ |
| Parametri (d) — A,Q,L,r | `compute_weighted_moments`, regressore composito $\phi_t$ | draw MNIW / NIG |
| Parametri (d) — log-vol AR(1) | macchina NIG di Family A | regressione sulla path log-vol |
| Parametri (d) — leverage | — | Metropolis, biforcato per timing |
| Parametri (d) — $\nu$ | FOC/log-target di `update_nu` | griddy-Gibbs / Metropolis (draw) |
| Test | pattern `run_recovery`, Procrustes, simulatore base | recovery MCMC 3-livelli; estensione simulatore (SV+leverage); R-hat/ESS/trace |

**In una frase**: si scrive un nuovo package `src/mcmc/` di orchestrazione e draw; tutto il modello, il filtro, la mixed-frequency, i momenti e l'harness di recovery sono importati dall'esistente. Il lavoro genuinamente nuovo è concentrato nel **sampler della volatilità** e nel **leverage**, ed è esattamente lì che va puntata la batteria di recovery test, un timing per volta (prima il Ramo A, poi il Ramo B).
