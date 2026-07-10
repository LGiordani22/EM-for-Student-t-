# Audit P1–P5 sotto la configurazione scelta

**Configurazione fissata per tutta l'analisi** (decisione di modellazione chiusa, non
si rimette in discussione qui):

- **Branch B** — timing laggato, mistura sign-augmented di Omori + FFBS diretto.
- **Option A** — `r` canali scalari `ρ_k`, ciascuno sul proprio errore grezzo `z^u_k`.
- **Spec II** — volatilità *outside*, `Var(u_t|·) = √H^u_t · Q · √H^u_t / w^u_t`, cioè
  `h_k` è la volatilità del **fattore `k`**, non di una direzione ortogonalizzata.

Branch A è valutato solo dove serve al confronto (e perché è ancora nel codice), ma
non è la configurazione target.

**Metodo.** Ogni problema è stato riletto **dal codice**, non dalla mappa, e dove
possibile **misurato**. Gli esperimenti girano sul DGP per-fattore vero
(`simulate_dfm_sv(sv_u_perfactor=...)`), `r=3`, `Q` diagonale salvo dove indicato,
`ρ_vero = [-0.70, -0.15, +0.45]`, `sv_u = [(0,.97,.25), (0,.92,.18), (0,.95,.22)]`
(sd incondizionate della log-vol: `1.03`, `0.46`, `0.70`).

> **Risultato più importante dell'audit, e non è nessuno dei cinque.** Vedi **P6** in
> fondo: `ESS(ρ) ≈ 3–23 su 2000 draw` sotto Branch B. `ρ` è il parametro che dà la
> **skew** alla predittiva del PIL, ed è oggi il meno affidabilmente stimato del
> modello. Tre dei cinque problemi catalogati sono, sotto B, non-problemi; questo è
> reale e non era catalogato.

---

## Sommario dei verdetti

| | si pone sotto B? | bloccante per GaR? | azione ora | chi decide |
|---|---|---|---|---|
| **P1** coupling `R_ξ` | **No** — strutturalmente assente | No | ✅ **chiuso** (guard + tripwire) | tecnico (Lorenzo) |
| **P2** costo `√r` | **Sì**, ma va scomposto | **Parzialmente** | misurare `σ̂_η,k` sul pannello reale | modellazione (Ciganovic) |
| **P3** trappola single-move | **No** — A-specifico | No | ✅ **chiuso** (immunità di B asserita) | tecnico |
| **P4** disaccoppiato vs calibrazione | **Attenuato** a ~0.4% | No | ✅ **chiuso** (diagnostica); agganci nel forecast | modellazione, ma bassa posta |
| **P5** whitening A-vs-B | **No** come A-vs-B; ~0 come bias interno | No | ✅ **chiuso** (forma chiusa + diagnostica) | tecnico |
| **P6** *(nuovo)* mixing di `ρ` | **Sì**, grave | **Sì** | risolvere **prima** del pannello reale | tecnico, con esito da mostrare |

### Stato di chiusura (2026-07-10, dopo l'audit)

P1, P3, P4, P5 sono stati **chiusi nel codice**, senza alcun cambio di modello:

- **P1** — `sample_common_vol_mv(R_xi=...)` ora **solleva** senza
  `allow_experimental=True`. `test_variants` [8] congela l'irraggiungibilità:
  `fit_dfm_mcmc` non espone `R_xi`, `gibbs.py` non lo nomina, `sample_leverage_lagged`
  non referenzia il blocco multivariato, e un **tripwire** verifica che una run di
  Branch B non vi entri mai.
- **P3** — `test_passo4` [6] asserisce che il warm-seed esiste *solo* in Branch A e che
  B, senza alcun rimedio, esce dal warm start piatto e recupera `φ̂ > 0.7` su tutti i
  canali (mai negativo, il modo di fallimento di A).
- **P4/P5** — tre funzioni pure in `diagnostics.py` rendono i numeri ispezionabili a
  ogni run: `posterior_corr_Q(draws)`, `coupling_overconfidence(Q)`,
  `leverage_whitening_attenuation(Q)`. `test_diagnostics.py` (21 check) verifica la
  forma chiusa di `λ` **contro Monte Carlo sulla definizione**, non contro sé stessa,
  e sull'EM reale riporta `+0.04%` di sovra-confidenza e `−0.13%` di bias su `ρ`.

Una precisazione emersa scrivendo i test, e vale la pena registrarla: **a `Q` diagonale
`R_ξ` non è esattamente `I`**. La teoria dà `g(0)=0`, ma `g(ρ)` è tabulata per Monte
Carlo con `n_mc=3·10⁶`, il cui errore standard su una correlazione è `≈5.8·10⁻⁴`; la
tabella restituisce `−4.2·10⁻⁴`, cioè una deviazione standard da zero. Non è un bug —
è il rumore della tabulazione — ma significa che "disaccoppiato ≡ coupled a `Q`
diagonale" vale a meno di `~4·10⁻⁴`, non esattamente. Il test lo asserisce con quella
soglia, nominandola.

Resta aperto solo **P2** (che richiede una misura sul pannello reale) e **P6** (che
richiede il griddy-Gibbs su `ρ`).

---

## P1 — Coupling multivariato `R_ξ`

### Cos'è davvero

La mappa e il `REFACTOR_PLAN` lo descrivono come "coupled lasciato EXPERIMENTAL
dietro l'argomento `R_xi`, default disaccoppiato". **Leggendo il codice la situazione
è più netta di così, e la mappa va corretta.**

`R_xi` esiste in **una sola funzione**, `sample_vol.sample_common_vol_mv`
(`sample_vol.py:544`, argomento `R_xi: np.ndarray | None = None`, ramo coupled a
`sample_vol.py:665-690`). I suoi chiamanti sono esattamente tre:

1. `sample_vol.sample_volatility_block_specII` (`sample_vol.py:778`) — il blocco
   **senza leverage**, che gli passa il proprio `R_xi` (default `None`);
2. `sample_leverage.py:464-466` — il **warm-seed di Branch A**, che non passa `R_xi`;
3. i test (`test_shared.py`, `test_spec2_recovery.py`).

E il punto decisivo: **`gibbs.py` non contiene la stringa `R_xi`** (zero occorrenze).
Quindi `fit_dfm_mcmc` non espone il parametro su nessun path: il ramo coupled **non è
raggiungibile dall'orchestratore**, nemmeno per errore, nemmeno volendo.

Sotto **Branch B** è ancora più forte: `sample_leverage_lagged.py` **non chiama mai**
`sample_common_vol_mv`. Il blocco comune sono `r` canali Omori/FFBS *scalari e
indipendenti* (`_branch_b_one_process` con `K=1`, `sample_leverage_lagged.py:364-366`),
ognuno con la propria misura per componente `e_k = √(w/q_kk)·u_k`. Non esiste, in
Branch B, un oggetto in cui inserire una covarianza di misura fra canali.

### Sotto B, si pone?

**Non si pone.** Non per decisione, ma per costruzione: sotto B il codice non ha un
FFBS `r`-dimensionale in cui `R_ξ` possa entrare. Attivare il coupling sotto B non
sarebbe cambiare un flag, sarebbe **scrivere un blocco nuovo**.

### È bloccante per il forecast/GaR?

**No.** E c'è un motivo empirico che va oltre l'argomento strutturale: il coupling
serve solo se `corr(Q)` è materiale, e **sul pannello reale non lo è**. Letto dal fit
EM già su disco (`fit_dfm_result.npz`, entrambe le config):

| config | off-diag di `corr(Q)` | max &#124;·&#124; | `R_ξ` off-diag implicata |
|---|---|---|---|
| `small` | `[0.053, −0.023, 0.039]` | 0.053 | max `0.0009` |
| `big` | `[−0.006, −0.046, 0.099]` | 0.099 | max `0.0036` |

(`R_ξ` calcolata con la funzione vera del codice, `sample_vol.logsq_corr_matrix`.)
La correlazione dei **log-quadrati** è di secondo ordine in `corr(Q)`: a `corr(Q)=0.1`
vale `0.0037`, a `0.3` vale `0.037`, e serve `corr(Q) ≈ 0.9` per arrivare a `0.51`.
Il disaccoppiato è **esatto a `Q` diagonale** e qui `Q` è quasi diagonale.

### Come si risolverebbe — ✅ FATTO

Non c'era da correggere il modello, solo da rendere il fatto **verificato invece che
constatato** (oggi lo sappiamo perché *ho letto* `gibbs.py`; domani qualcuno aggiunge
un parametro e non ce ne accorgiamo):

1. `sample_common_vol_mv(R_xi=...)` ora **solleva** senza `allow_experimental=True`,
   con un messaggio che riporta l'instabilità (`φ 0.90→0.42`) e il `+0.4%`.
2. `test_variants` [8] congela: `fit_dfm_mcmc` non ha `R_xi` nella firma, `gibbs.py`
   non lo nomina, `sample_leverage_lagged` non referenzia il blocco multivariato, e un
   **tripwire** (monkeypatch) verifica che una run di Branch B non vi entri mai.
3. `MCMC_MAP.md` corretto: diceva "dietro il flag `R_xi`", suggerendo una disponibilità
   che non c'è.

Il ramo coupled **resta** come implementazione di riferimento del metodo del `.tex` —
serve a poter *mostrare* che la forma literal è instabile, invece di asserirlo.

⚠️ **Caveat onesto**: `corr(Q)` qui viene dal fit EM del modello **senza SV**. Sotto SV
la `Q` posteriore potrebbe cambiare. Il numero va rimisurato sui draw MCMC del
pannello reale (è la stessa misura che serve a P4 — una riga di codice sul dict dei
draw, nessun modulo nuovo).

### Chi decide

**Tecnico (Lorenzo).** Non c'è nulla da validare col professore: è una constatazione
sul codice più una misura.

**P1 ≡ P4?** Sono lo stesso oggetto visto da due lati — il coupling è la *causa*, la
sovra-confidenza la *conseguenza*. Li tratto separatamente perché la domanda di P1 è
"il default è giusto?" (sì, e non c'è alternativa sotto B) e quella di P4 è "quanto
costa il default?" (misurabile, vedi sotto).

---

## P2 — Il costo `√r` e il canale a bassa volatilità

Questo è il problema segnalato come rischio numero uno, e la richiesta era di
distinguere **informazione** da **mixing**. La risposta è: *entrambi, ma in parti
diverse dello stesso fenomeno, e la scomposizione cambia le mitigazioni*.

### Cos'è davvero (dal codice)

Sotto Spec II ogni `h^u_k` ha il **proprio stato**, letto attraverso **una sola**
misura per periodo. In Branch B, `_branch_b_one_process` è invocato con `K=1` per
fattore (`sample_leverage_lagged.py:364`), e dentro combina le `K` misure in una sola
osservazione efficace (`sample_leverage_lagged.py:249-255`): con `K=1` la precisione
per periodo è `1/v²_{s_t}`, cioè **una** log-χ²₁.

Nella restrizione scalare `H = h·I` che il refactor ha sostituito
(`sample_vol.sample_volatility_block:422-426`), lo **stesso** stato riceveva `r`
misure per periodo (`ys_u = log(tilde_u.reshape(-1)²)`, `r(T-1)` osservazioni). Da qui
il `√r`: la varianza del rumore di misura per stato passa da `(π²/2)/r` a `π²/2`.

La descrizione nella mappa è quindi **corretta come meccanismo**, ma la conseguenza
che ne trae ("serve `~r×` più `T`") è **sbagliata**, e questo audit la corregge.

### Sotto B, si pone? — sì, e va scomposto in due cose diverse

Misurato sul sampler vero, Branch B, stesso DGP, `n_iter=600`:

| | `corr(ĥ_k, h_k)` | `φ̂_k` (vero `.97/.92/.95`) |
|---|---|---|
| T=600 | `[0.89, 0.22, 0.86]` | `[0.96, 0.45, 0.95]` |
| T=1200 | `[0.92, 0.59, 0.81]` | `[0.97, 0.88, 0.96]` |
| T=2400 | `[0.91, 0.63, 0.82]` | `[0.98, 0.90, 0.95]` |

E il **tetto informativo**, calcolato indipendentemente dal sampler con uno smoother
di Kalman a **parametri veri** su una singola log-χ²₁ per periodo (quindi: nessun
errore di mixing, nessuna incertezza sui parametri — è quanto *chiunque* può estrarre):

| canale | sd inc. | tetto, 1 misura/periodo | tetto, `r=3` misure (scalar-common) |
|---|---|---|---|
| forte `k=0` | 1.03 | 0.831 | 0.901 |
| **debole `k=1`** | **0.46** | **0.524** | 0.688 |
| forte `k=2` | 0.70 | 0.711 | 0.827 |

e questo tetto **non si muove con `T`**: `0.535` (T=600) → `0.549` (T=1200) → `0.551`
(T=4800). Ovvio col senno di poi: la correlazione fra path stimato e path vero è un
problema di **estrazione di segnale per periodo**; più periodi aggiungono periodi, non
informazione *per* periodo.

**La scomposizione, quindi:**

1. **La parte "mixing/stima"**: a T=600 il canale debole ha `φ̂ = 0.45` contro `0.92`
   vero, e `corr = 0.22` contro un tetto di `0.52`. A T=1200 `φ̂ = 0.88` e `corr = 0.59`.
   Cioè: **i parametri `(φ_k, σ²_k)` si recuperano con `T`**, e con essi buona parte
   della correlazione persa. Questo `T` *lo cura*.
2. **La parte "informazione"**: oltre T≈1200 la `corr` satura a `~0.63` contro il tetto
   `~0.55` (il sampler lo supera leggermente perché la mistura KSC/Omori sfrutta la
   non-gaussianità della log-χ², cosa che lo smoother QML non fa). Questo `T` **non lo
   cura**, e nessun miglioramento di mixing lo curerà.

**È mixing?** No — e c'è la prova diretta. ASIS esiste apposta per rompere la cresta
path/scala. Attivato sotto B, stesso DGP, T=600:

```
use_asis=False   corr=[0.890, 0.215, 0.861]   phi=[0.963,  0.454, 0.952]
use_asis=True    corr=[0.895, 0.146, 0.800]   phi=[0.944, -0.114, 0.873]
```

ASIS **peggiora** il canale debole (`φ̂` va a `−0.11`). Se il collasso fosse la cresta
path/scala, ASIS lo curerebbe. Non lo cura: non c'è segnale da mescolare meglio.

**Il collasso è indipendente dal branch?** Sì per il canale debole, e Branch A è
peggio ovunque. Branch A, stesso DGP e `n_iter`:

```
T=600    corr=[0.86, 0.33, 0.76]   phi=[0.962, 0.947, 0.938]   rho=[-0.33, -0.06, +0.31]
T=1200   corr=[0.80, 0.24, 0.60]   phi=[0.944, 0.724, 0.881]   rho=[-0.18, +0.06, +0.09]
```

⚠️ **Branch A *degrada* al crescere di `T`.** A T=1200 il leverage è di fatto perso e il
segno del canale debole si inverte. Il motivo è nel codice: `_lev_path_mh` fa `for t in
range(T)` (`sample_leverage.py:128`), **una mossa single-move per coordinata per
sweep**, con `prop_sd=0.25` fisso. Raddoppiare `T` raddoppia i gradi di libertà del
path senza dare una sweep in più: il livello/scala del path diffonde più lentamente.
Branch B, con FFBS diretto (`acc["path_u"]=1.0`, `sample_leverage_lagged.py:331`), non
ha questo problema, e infatti **migliora** con `T`. Questo è un argomento forte, e
misurato, a favore della scelta di B.

### È bloccante per il forecast/GaR?

**Parzialmente, e in un modo preciso.** Cosa degrada se lo lascio così:

- **La scala della predittiva** nei periodi in cui il fattore debole guida la varianza.
  `h^u_k` entra nel sandwich `√H Q √H`: se `ĥ_k` è sovra-lisciata, la varianza
  condizionale del fattore `k` è compressa nei picchi e gonfiata nelle calme.
- **La coda sinistra**, indirettamente e non tramite `ρ`: la skew nasce da `ρ_k`, ma
  l'*ampiezza* della coda nasce da `h`. Un `h` compresso dà code strette proprio dove
  servirebbero larghe — è la stessa **compressione di scala** già documentata in
  `project_density_nowcast_feasibility` (la predittiva DFM-forward propaga le code `t`
  ma non adatta la scala, e manca i crash).
- **La calibrazione PIT** ne risente in modo asimmetrico: PIT schiacciati verso gli
  estremi nei periodi di alta volatilità del fattore debole.

**Ma non è bloccante in senso stretto**, perché dipende da un fatto che **non
conosciamo ancora**: quanto vale `σ_η,k` sui fattori *veri*. Il "canale debole" del
mio DGP è una scelta mia (`σ=0.18, φ=0.92` ⇒ sd `0.46`). Se sul pannello reale tutti e
tre i fattori hanno volatilità che si muove come i miei canali forti (sd ≳ 0.7), P2
semplicemente non morde. **Non è misurabile in astratto.**

### Come si risolverebbe — dalla più leggera alla più invasiva

0. **(Prima di tutto) Misurare.** Un run MCMC sul pannello reale con `sv=True,
   leverage=True, timing="lagged"` e leggere `σ̂_η,k / √(1−φ̂_k²)` per `k=1..r`.
   Zero righe di codice: sono già nei draw (`draws["sv_u"]`). **Se tutte le sd sono
   ≳ 0.7, P2 si chiude qui.**
1. **Prior più stretto su `σ_η`** (half-Normal con `B` piccolo). Costo: **zero righe** —
   `sv_sigma_prior="half_normal"`, `sv_half_normal_B=...` sono già flag di
   `fit_dfm_mcmc`. Effetto: regolarizza `σ̂²_k` dove i dati non parlano. **Non** alza il
   tetto informativo, ma evita il collasso di `φ̂` a `T` corto (che è la parte curabile).
2. **Fisher-`z` shrinkage su `ρ`** — il `.tex` la nomina esplicitamente
   (riga 21544: *"a mild shrinkage toward zero, if desired, is cleanest as a zero-mean
   normal on the Fisher-z transform atanh ρ"*). Costo: ~10 righe in
   `_rho_logpost_scalar` + un flag. Serve un recovery test nuovo.
3. **Pooling parziale dei `(φ_k, σ²_k, ρ_k)` fra fattori** (gerarchico). Costo: medio —
   un iperprior sui tre canali dentro Family B/C, più il draw degli iperparametri.
   Tocca `sample_vol.draw_ar1_params`, `_draw_sigma2_lev`, `draw_rho_scalar` e
   `gibbs.py`. Recovery test da riscrivere. **Attenzione**: prende in prestito
   informazione fra fattori, il che è esattamente ciò che Spec II voleva evitare
   (l'interpretazione "h_k è la vol del fattore k"). È una scelta di modellazione.
4. **Fallback alla volatilità comune scalare `H = h·I` sotto una soglia di `T`.**
   Costo: **alto**. Il blocco scalare esiste (`sample_volatility_block`) ma è
   dichiarato *seam di test*, non sta su nessun path, non ha leverage per-fattore, e
   `draw_A_Q_perfactor`/`ffbs_sample_states` sono cablati su `h_u` `(T,r)`. Riportarlo
   in produzione significa reintrodurre un ramo che il refactor ha appena eliminato, e
   **contraddice la scelta Spec II** (con `H=hI` il bivio inside/outside è vuoto e i
   tre `h_k` tornano uno). Sconsigliato: se serve, tanto vale porre `r=1` per la vol.

**Raccomandazione: (0), poi (1) se serve. Non fare (3)-(4) adesso.**

### Chi decide

- **Tecnico (Lorenzo)**: misurare `σ̂_η,k`; attivare il prior half-Normal; alzare
  `n_iter`/`T` nei recovery test.
- **Modellazione (Ciganovic)**: il **pooling** (3) e il **fallback scalare** (4) sono
  cambiamenti di modello, non di sampler. In particolare (3) indebolisce la lettura
  "per-fattore" che è la ragione dichiarata per adottare Spec II. Va portato al
  professore **solo se** la misura (0) mostra che almeno un fattore reale ha
  volatilità debole.

---

## P3 — Trappola del single-move dal warm start piatto

### Cos'è davvero

Dal warm start `log h = 0`, la misura χ²₁ per-fattore è troppo rumorosa perché il
Metropolis single-move esca: `h~1 → stati omoschedastici → u omoschedastici → h~1`.
Il rimedio adottato è un **warm-seed**: se il path entra piatto, lo si inizializza con
un draw KSC-FFBS a blocchi.

Nel codice il rimedio sta in **un solo posto**:

```python
# sample_leverage.py:463   (Branch A)
if not np.any(np.abs(logh_u) > 1e-9):
    from mcmc.sample_vol import sample_common_vol_mv
    logh_u = sample_common_vol_mv(u, Q, w_u, logh_u, seed_sv, rng, offset=1e-6)["logh_u"]
```

`sample_leverage_lagged.py` **non contiene nulla di analogo** (verificato: la stringa
non compare).

### Sotto B, si pone?

**Non si pone. B è immune per costruzione.** Il path non viene raggiunto con mosse
locali: è **estratto dalla sua full conditional** con un FFBS diretto
(`_ffbs_tv`, `sample_leverage_lagged.py:85`), il che è esattamente ciò che rende
`acc["path_u"] = 1.0` (`:331`) non un'approssimazione ma un fatto. Da `log h = 0` con
`ρ = 0` iniziale la transizione è `G_t = φ`, `c_t = 0`, `W_t = σ²`
(`sample_leverage_lagged.py:241-246`): il primo draw è già un FFBS KSC/Omori pieno, che
"salta" direttamente in una regione di path plausibile. Non c'è nessuna trappola da cui
uscire. E infatti B recupera `φ̂ ≈ 0.96` da init piatto **senza alcun warm-seed**.

Non è un caso: è la ragione teorica per cui il `.tex` preferisce B. La mappa dice che
B "mescola meglio"; l'audit precisa **perché**, ed è più forte di "meglio": *A ha un
modo di fallimento che B non ha*.

### È bloccante per il forecast/GaR?

**No.** Sotto B non esiste.

**Ma va documentato come argomento pro-B**, perché è un fatto misurato e non
un'opinione, e si somma alla degradazione di A con `T` documentata in P2. Se un giorno
qualcuno riproponesse A, questi sono i due numeri da mostrare.

### Come si risolverebbe — ✅ FATTO

Nulla da correggere sotto B; ma l'immunità non era asserita da nessuna parte, e un
domani qualcuno potrebbe toccare `_ffbs_tv` senza accorgersene. `test_passo4` [6] ora
asserisce, in un colpo, il fatto strutturale e quello empirico:

- Branch A **contiene** il rimedio (`sample_common_vol_mv` nel sorgente), Branch B
  **non lo contiene**;
- e non gli serve: da `log h = 0` B abbandona il path piatto (`sd(log h) > 0.1`),
  recupera `φ̂ > 0.7` su tutti i canali, e **non degrada mai `φ` a negativo** — che è
  esattamente il modo di fallimento di A.

Se si volesse tenere A vivo come robustness check, l'unica cosa onesta sarebbe rendere
il warm-seed incondizionato (oggi si attiva solo se il path è *esattamente* piatto
entro `1e-9`, quindi solo alla prima sweep) e adattare `prop_path` a `T` — che è anche
la causa della degradazione con `T` documentata in P2. **Non urgente, non serve al
target.**

### Chi decide

**Tecnico.** Nessuna implicazione di modellazione: A e B campionano dallo stesso
target sotto lo stesso *timing*; qui si parla di come lo raggiungono.

*(Nota: la scelta A-vs-B **in sé** era una decisione di modellazione — il timing è
un'ipotesi sostantiva — ma è già stata presa, e questo audit la conferma sul piano
computazionale.)*

---

## P4 — Il disaccoppiato è sovra-sicuro; il target è la calibrazione

### Cos'è davvero

Il sampler disaccoppiato tratta gli `r` log-quadrati come indipendenti. Se in verità
sono correlati (attraverso `corr(Q)`), sta usando **più informazione di quanta ce ne
sia**, e la sua posterior sulla volatilità è troppo stretta. Per un obiettivo di
**densità/GaR**, dove il target *è* la calibrazione, questo è il tipo di errore che
conta — a differenza dell'accuratezza puntuale, che è ciò che era stato misurato
quando si è deciso il default (il "coupling non paga" del `REFACTOR_PLAN`).

Quel ragionamento resta valido, **ma il suo peso dipende da un numero**, e il numero
adesso ce l'abbiamo. La sovra-confidenza sulla precisione congiunta dei tre canali
scala come `√(1 + (r−1)·R_ξ)`:

| `corr(Q)` | `R_ξ` off-diag | sovra-confidenza |
|---|---|---|
| **0.099** *(pannello `big`)* | **0.0036** | **+0.4 %** |
| 0.30 | 0.037 | +3.6 % |
| 0.50 | 0.111 | +10.5 % |
| 0.90 | 0.508 | +42 % |

Sotto Spec II, per giunta, ogni canale ha il **proprio** stato: il coupling agisce sulla
covarianza del rumore *fra* canali, non su un unico stato condiviso, quindi l'effetto
sulla singola `ĥ_k` è ancora più debole di così.

### Sotto B, si pone?

**In forma molto attenuata, e senza rimedio *sotto leverage*.** Attenuata perché
`corr(Q)` reale è ≤ 0.099 ⇒ +0.4%. Sotto **Branch B** non c'è un FFBS `r`-dimensionale
in cui inserire il coupling: il blocco comune sono `r` canali Omori scalari, e
accenderlo lì richiederebbe una mistura di Omori congiunta che non deriviamo
(`.tex` `subsec:lev-branches-allproc` (iii)).

Sul **path senza leverage** (Spec II, `sv=True, leverage=False`), invece, il coupling
ora **c'è**: `common_vol_coupling="qml"` (2026-07-10). È la forma **QML** — covarianza
di misura costante `Σ_ξ = (π²/2)R_ξ`, senza mistura — non la literal instabile.
Verificata stabile a `corr(Q)=0.92` (`φ̂ = 0.856` sul fattore meno persistente, contro
`0.422` della literal; `test_spec2_recovery` [3]). Resta **non-default**: con
`corr(Q) ≈ 0.1` non paga il doppio costo e la perdita del raffinamento della mistura.

### È bloccante per il forecast/GaR?

**No.** +0.4% di sovra-confidenza sulla precisione della log-vol è invisibile rispetto
agli altri effetti in gioco (P2 satura la `corr` di `h` a `0.6` sui canali deboli; P6
lascia `ρ` con `ESS ≈ 15`). Investire qui prima di aver risolto P6 sarebbe ottimizzare
il terzo decimale mentre il primo è sbagliato.

### Come si risolverebbe, e la mossa giusta — ✅ diagnostica FATTA

Il coupling non va acceso. Ma la misura che lo deciderebbe non esisteva come funzione.
Ora sì, in `diagnostics.py` (pure, nessun effetto sul sampler):

- `posterior_corr_Q(draws)` — correlazioni posteriori fra innovazioni dei fattori, con
  banda credibile. **Chiude P1, P4 e P5 insieme**, perché tutti e tre dipendono da quel
  numero. È anche il robustness check di Huang–Wand che il `.tex` prescrive.
- `coupling_overconfidence(Q)` — `√(1+(r−1)R̄_ξ) − 1`.
- `leverage_whitening_attenuation(Q)` — vedi P5.

Sull'EM reale: `+0.04%` di sovra-confidenza. Gate: `test_diagnostics.py`.

E ora c'è anche il **rimedio**, non solo la diagnostica (2026-07-10): la **QML** è
implementata come opzione esplicita, `common_vol_coupling="qml"` (path senza leverage).
`diagnostics.recommend_coupling(Q)` dà la raccomandazione: `"decoupled"` sotto una
soglia di sovra-confidenza (5% ⇒ `corr(Q)≈0.35`), `"qml"` sopra — ma solo *"IFF a
PIT/coverage check also shows tail mis-calibration"*, mai sul numero da solo, come il
`.tex` prescrive.

**La mossa giusta resta: default disaccoppiato ora, misura sul pannello reale, decidi
dopo — e con ogni probabilità la risposta resterà "no".** Confermo la lettura della
mappa, con la correzione che `corr(Q)` non è più ignoto: sappiamo già che è ~0.1
sull'EM. Va riconfermato sui draw MCMC (sotto SV la `Q` può cambiare), ma la prior è
forte.

**Non c'è motivo di accendere il coupling prima.** A `Q` diagonale il disaccoppiato è
**esatto**, e la QML per giunta *perde* il raffinamento della mistura (a `Q` diagonale è
una singola gaussiana `π²/2`, non la mistura KSC), quindi non conviene finché `corr(Q)`
resta piccolo. La literal (`Σ = diag(v_s) R_ξ diag(v_s)`) resta **instabile** (`φ
0.90→0.42`) e dietro `allow_experimental`, tenuta solo per esibire quel finding.

**Agganci che il modulo di forecast deve avere perché accendere il coupling *dopo* non
costringa a riscriverlo.** Il forecast consuma i draw, non il sampler, quindi la regola
è semplice: **non assumere mai indipendenza fra i canali di volatilità a valle.**

1. **Simulare `log h_{T+h}` come vettore, non come `r` scalari.** Anche se oggi
   l'innovazione è `diag(σ²_k)`, il codice deve propagare una **matrice** di covarianza
   `Σ_η` (oggi diagonale). Se un domani il coupling introducesse correlazione, cambia
   una matrice, non un ciclo.
2. **Leggere `Σ_η` (e `φ`) dai draw, non da costanti**: `draws["sv_u"][d]` è `(r,3)` e
   contiene già tutto per draw.
3. **Non "mediare" `h_u` sui draw prima di simulare.** La compressione di scala nasce
   proprio da lì. Ogni traiettoria predittiva parte dal **proprio** `h_u[d, T-1]`, e
   l'incertezza posteriore su `h` deve entrare nella densità. Se il sampler è
   sovra-sicuro, si vedrà nel PIT; se il forecast media prima, non si vedrà più nulla.
4. **Esporre `corr(Q)` e il PIT per periodo come output diagnostici**, non solo il
   quantile finale: sono le due misure che chiudono P4.

Con questi quattro punti, accendere il coupling in futuro è un cambio nel sampler e
nulla nel forecast.

### Chi decide

- **Tecnico**: misurare `corr(Q)` posteriore; scrivere il forecast con gli agganci sopra.
- **Modellazione (Ciganovic)**: solo se `corr(Q)` posteriore risultasse **grande** (≳0.4,
  cosa che l'EM non suggerisce affatto) andrebbe portata la domanda "vale la pena un
  blocco Omori multivariato per calibrare la coda?". Con `0.1`, la posta è troppo bassa
  per disturbarlo.

---

## P5 — Whitening di magnitudine diverso fra A e B

### Cos'è davvero

Entrambi i branch sono Option A e usano, come **segno** del leverage, `d^u_k =
sign(z^u_k)` dello shock a **whitening pieno** `z^u = √w · Q^{-1/2}(√H)^{-1} u`
(`sample_leverage_lagged.py:349-351`). Differiscono nella **magnitudine**:

- Branch A usa l'esatto `|z^u_k|`;
- Branch B usa la **componente** `|e_k| = √(w/q_kk)|u_k|`, perché è ciò che rende
  `ξ_k = y*_k − log h_k` **lineare** in `log h_k` e quindi l'FFBS possibile
  (`sample_leverage_lagged.py:346-347`, la ragione strutturale della mistura di Omori).

Le due coincidono a `Q` diagonale.

### Sotto B, si pone?

**Come "discrepanza A-vs-B": no, è irrilevante.** Avendo scelto B, non c'è nessun
confronto da fare: la magnitudine per componente *è* la specificazione del sampler, e
il `.tex` la prescrive.

**Come bias interno a B: sì in linea di principio, e l'ho quantificato in forma
chiusa.** Il regressore di Family C sotto B è `g_k ≈ sign(z_k)·|ζ̄_k|`, dove
`ζ̄_k = ζ_k/√q_kk` e `ζ = Q^{1/2} z`. Il vero drift è `ρ z_k`. Poiché `Var(ζ̄_k)=1`, la
stima di `ρ` è attenuata del fattore

```
ρ̂/ρ  =  E[z_k · g_k]  =  E[|z_k|·|ζ̄_k|]  =  λ(c_k) = (2/π)·( c·arcsin c + √(1−c²) )
```

con `c_k = Corr(z_k, ζ̄_k) = (Q^{1/2})_{kk} / √q_kk`. Verificata per Monte Carlo
(errore < 0.3%). Proprietà: `λ(1)=1` (nessuna attenuazione a `Q` diagonale),
`λ(0)=0.637`, ed è **sempre un'attenuazione, mai un'inversione di segno** — il segno è
esatto perché usa il whitening pieno.

| `corr(Q)` (`r=3` equicorrelata) | `c_k` | `λ` | bias su `ρ` | `ρ=−0.6` diventa |
|---|---|---|---|---|
| **≈0.05–0.10** *(reale)* | ~0.999 | ~0.999 | **−0.1 %** | −0.599 |
| 0.30 | 0.979 | 0.981 | −1.9 % | −0.589 |
| 0.50 | 0.943 | 0.951 | −4.9 % | −0.571 |
| 0.70 | 0.882 | 0.906 | −9.4 % | −0.544 |
| 0.80 | 0.836 | 0.876 | −12.4 % | −0.526 |

**A quale `corr(Q)` diventa preoccupante per la coda?** Il bias è un'attenuazione di
`|ρ|`, e la skew della predittiva è monotona in `|ρ|`: quindi il sampler
**sottostima la skew sinistra**, cioè *sottostima il rischio di coda* — la direzione
sbagliata per il GaR. Ma la soglia pratica è alta: serve `corr(Q) ≳ 0.5` per superare
il 5% di attenuazione, e `≳ 0.8` per superare il 10%.

Sul pannello reale, `c_k` calcolato dal fit EM: `[0.9998, 0.9987, 0.9999]` (`small`) e
`[0.9996, 0.9988, 0.9986]` (`big`) ⇒ **attenuazione −0.1%**. Irrilevante.

Va inoltre notato che la linearizzazione di Omori in sé (`a_j + b_j(ξ−m_j)` al posto di
`exp(ξ/2)`) **non attenua**: misurata con la tabella vera del codice,
`E[zg]/E[g²] = 1.0000` e `corr(g, z) = 0.997`. Il regressore di Branch B è, di suo,
eccellente.

### È bloccante per il forecast/GaR?

**No.** −0.1% su `ρ`, contro un `ρ` che oggi ha `ESS ≈ 15` (P6). L'errore di P5 è
quattro ordini di grandezza sotto il rumore Monte Carlo con cui misuriamo `ρ`.

### Come si risolverebbe — ✅ precisato, non corretto

**Il bias non va corretto** (−0.1%): dividere il regressore per `λ(c_k)` sarebbe una
deviazione dal `.tex` per guadagnare un millesimo. Ciò che serviva era la **precisione**,
e ora c'è:

1. `diagnostics.leverage_whitening_attenuation(Q)` restituisce `c_k`, `λ_k` e
   `bias_pct` per fattore, a ogni `Q`.
2. `test_diagnostics.py` [3] verifica la forma chiusa **contro Monte Carlo sulla
   definizione** `E[|z_k||z̄_k|]` — non contro sé stessa — a `corr(Q) ∈ {0.3, 0.6, 0.9}`,
   più le proprietà su cui l'audit poggia: `λ ≤ 1` sempre (mai amplificazione, mai
   inversione di segno) e `λ(1)=1` a `Q` diagonale.
3. Il docstring di `sample_leverage_lagged.py` diceva che le due magnitudini
   "differiscono *mildly* for full Q" — vago. Ora riporta la forma chiusa, i valori
   (`0.98` a `corr(Q)=0.3`, `0.88` a `0.8`), la direzione dell'errore (**sottostima la
   skew sinistra**, cioè erra *contro* l'obiettivo GaR) e il fatto che la
   linearizzazione di Omori di suo non attenua.

Se un giorno `corr(Q)` posteriore risultasse ≳ 0.5 (che l'EM non suggerisce), la
correzione `1/λ(c_k)` è una riga, calcolabile da `Q` a ogni sweep — ma andrebbe
discussa, perché è una deviazione dal `.tex`. **Non ora.**

### Chi decide

**Tecnico** — è una constatazione quantitativa. Diventa **modellazione** solo nello
scenario `corr(Q) ≳ 0.5`, che va prima misurato.

---

## P6 *(nuovo, non catalogato)* — `ρ` non è affidabilmente stimato: `ESS ≈ 3–23`

Questo problema non è in `MCMC_MAP.md`. È emerso inseguendo un'anomalia: `ρ̂_0`
usciva `≈ −0.49` contro `−0.70` vero, **con `Q` diagonale** (quindi `λ=1`, niente P5),
**a `T=2400`** e persino con `σ_η = 0.6` e `corr(ĥ) = 0.95` (quindi niente rumore sul
path). Ho escluso le tre cause ovvie, una per una:

1. **Il kernel di Family C è non distorto.** `draw_rho_scalar` con regressore esatto e
   parametri veri recupera `[−0.710, −0.147, +0.442]` a T=600 e `[−0.706, −0.167,
   +0.467]` a T=2400.
2. **La linearizzazione di Omori non attenua**: `E[zg]/E[g²] = 1.0000`.
3. **P5 non c'entra**: `Q` è diagonale in tutti questi run.

La causa vera, misurata su una catena lunga (Branch B, T=600, 4000 iterazioni):

```
ESS(rho_0) =  13     ESS(rho_1) =  3     ESS(rho_2) =  23      su 2000 draw
medie per finestre di 500 it, rho_1:  -0.347, -0.038, -0.298, -0.309,
                                      -0.301, -0.170, +0.222, +0.520
```

`ρ_1` **cambia segno** fra finestre della stessa catena. `ESS(ρ)/N ≈ 0.6–1.2 %`. Con
un ESS di 15, la media posteriore di `ρ` è un numero con un errore Monte Carlo che
domina qualunque effetto discusso in P1–P5.

**E non è la scala della proposta.** `ρ` è aggiornato con **una sola** mossa RW per
sweep, `prop_sd = 0.06` fisso, partendo da `ρ=0` (`gibbs.py:402`, `:220`;
`draw_rho_scalar`, `sample_leverage.py:342`). Ho provato ad allargarla:

| `lev_prop_rho` | accettazione | `ρ̂` | `ESS` |
|---|---|---|---|
| 0.06 | 0.50 | `[−0.49, −0.30, +0.39]` | `[21, 13, 20]` |
| 0.15 | 0.26 | `[−0.49, −0.10, +0.41]` | `[20, 9, 20]` |
| 0.30 | 0.13 | `[−0.44, −0.18, +0.43]` | `[11, 10, 13]` |

Allargare non aiuta: l'accettazione crolla e l'ESS non sale. Il collo di bottiglia **non
è la proposta, è il blocking**: `ρ` è fortemente correlato a posteriori con il path `h`
e con `σ²_η`, e viene aggiornato **condizionatamente al path**, che a sua volta è stato
estratto **condizionatamente a `ρ`**. È la stessa cresta che ASIS risolve per `(φ, σ²)`
— e infatti il `.tex` (`subsec:asis-leverage`) nota che `ρ` "beneficia attraverso la
sua correlazione posteriore con `σ_η`" ma **non è interwoven**.

### È bloccante per il forecast/GaR?

**Sì, ed è l'unico dei sei che lo è senza riserve.** `ρ` è il parametro che genera la
**skew** della predittiva: è *la* ragione per cui abbiamo messo il leverage nel
modello, ed è ciò che distingue una densità del PIL con coda sinistra pesante da una
simmetrica. Se `ρ` ha `ESS ≈ 15`:

- la **probabilità di recessione** e i quantili bassi (GaR al 5%) ereditano un errore MC
  che non si riduce con `n_keep` (l'ESS non cresce);
- non possiamo distinguere `ρ = −0.7` da `ρ = −0.45` sui dati veri, cioè non possiamo
  dire **quanto** è asimmetrica la densità;
- e non possiamo nemmeno escludere che i `ρ̂ ≈ −0.49` sistematici siano un **bias reale**
  (un errore di timing o di scala nel regressore) invece che una catena non convergente.
  **Con `ESS = 15` i due casi non sono distinguibili.**

### Come si risolverebbe — dalla più leggera alla più invasiva

1. **Catene molto più lunghe + diagnostica obbligatoria.** Costo: zero righe.
   `split_r_hat` e `ess` esistono già in `diagnostics.py`. Va reso **impossibile**
   riportare un `ρ̂` senza il suo ESS. Questo è il minimo sindacale e va fatto subito:
   **tutti i valori di `ρ` citati nel `REFACTOR_PLAN` e in `MCMC_MAP` sono stati
   misurati a 600–900 iterazioni e sono quindi inaffidabili.**
2. **Adattare `prop_rho` durante il burn-in** (target di accettazione ~0.3). Costo:
   ~10 righe in `gibbs.py`. Dai numeri sopra, **da solo non basta**.
3. **Griddy-Gibbs su `ρ`.** Il log-posterior `_rho_logpost_scalar` è 1-D su `(−1,1)`,
   liscio e già scritto: un griddy come quello di `ν` (`draw_nu_griddy`) darebbe draw
   **indipendenti dalla posizione corrente**, eliminando la random-walk. Costo: ~20
   righe, riusa il pattern esistente. **Questa è, secondo me, la mossa giusta**: è la
   più economica che attacca la causa (RW lenta), non il sintomo.
4. **Interweaving anche su `ρ`** (estendere ASIS oltre `(φ,σ²)`), o un aggiornamento
   congiunto `(σ²_η, ρ)`. Costo: alto, teoria nuova, va derivato nel `.tex`.
5. **Prior Fisher-`z` su `ρ`** (già previsto dal `.tex`, riga 21544). Non risolve il
   mixing ma **stabilizza** la posterior a `T` corto, e riduce l'errore MC accorciando
   la coda del target.

**Raccomandazione: (1) subito, (3) prima del pannello reale, (5) come complemento.
Non (4).**

### Chi decide

- **Tecnico (Lorenzo)**: (1), (2), (3) sono pura efficienza computazionale — stesso
  target, stessa posterior. Nessuna validazione richiesta.
- **Modellazione (Ciganovic)**: (5), il prior su `ρ`, cambia la posterior. E soprattutto
  va portata al professore **la domanda a monte**: se dopo (3) `ρ` risultasse
  debolmente identificato anche sul pannello reale, *il leverage è sostenuto dai dati?*
  Quella è una domanda di modellazione, non di sampler.

---

## Ordine di lavoro raccomandato verso il forecast

### Prima di puntare il sampler sul pannello reale

1. **Risolvere P6, punto (3)**: griddy-Gibbs su `ρ`. È l'unico intervento sul sampler
   che raccomando prima dei dati veri. Senza, ogni numero sul leverage — e quindi sulla
   coda — è rumore. ⏳ **DA FARE.**
2. **Rendere obbligatoria la diagnostica** `ESS`/`split-R̂` su `ρ`, `φ`, `σ²` in ogni run
   (P6 punto 1). Zero righe: le funzioni ci sono. ⏳ **DA FARE.**
3. **Rifare i recovery test del leverage con catene adeguate**, e correggere i valori di
   `ρ̂` citati in `REFACTOR_PLAN.md` e `MCMC_MAP.md`, che sono sotto-convergiuti.
   ⏳ **DA FARE** (nel frattempo entrambi i documenti portano l'avviso).
4. ✅ **FATTO** — `MCMC_MAP.md` corretto su P1 (irraggiungibile, assente sotto B), su P2
   (`T` cura i parametri, non il tetto informativo; e Branch A *degrada* con `T`), su P3
   (B immune) e su P5 (forma chiusa). P1/P3/P4/P5 chiusi anche nel codice: guard +
   tripwire, test di immunità, tre diagnostiche. Vedi *Stato di chiusura* in cima.

### Da misurare sul pannello reale, al primo run

5. **`σ̂_η,k / √(1−φ̂_k²)` per ogni fattore** ⇒ chiude P2. È già nei draw.
6. **`corr(Q)` posteriore** ⇒ chiude P1, P4 e P5 in un colpo (tutti e tre dipendono da
   quello stesso numero). È una riga sul dict dei draw.
7. **`ESS(ρ_k)`** ⇒ verifica che (1) abbia funzionato.

### Da decidere solo dopo aver visto PIT/coverage

8. Se accendere una qualunque forma di coupling (P4) — e ricordando che sotto B **non
   esiste** e andrebbe scritta.
9. Se serve pooling/prior più stretti (P2 opzioni 2-3).
10. Se il leverage è sostenuto dai dati (la domanda a monte di P6).

### Cosa NON fare adesso

- ❌ **Non** scrivere il blocco Omori multivariato per `R_ξ`. Guadagno stimato: +0.4% di
  calibrazione. Costo: un blocco FFBS nuovo. È il caso da manuale di scope creep.
- ❌ **Non** reintrodurre il fallback alla volatilità scalare `H = h·I`. Contraddice
  Spec II, e il refactor ha appena tolto quel ramo.
- ❌ **Non** implementare D1-c (outlier). È in sospeso per decisione.
- ❌ **Non** correggere P5 con il fattore `1/λ(c_k)`: il bias è −0.1% sul pannello reale.
- ❌ **Non** toccare Branch A per riparare P3. Non è la configurazione target.
- ❌ **Non** ottimizzare il forecast per il coupling futuro oltre i quattro agganci di
  P4 (vettore, non `r` scalari; niente medie di `h` prima di simulare).

---

## Domande residue per Ciganovic

Solo modellazione sostanziale. Tutto il resto lo decido io.

1. **Il leverage è sostenuto dai dati?** Se, risolto il mixing (P6), `ρ_k` risultasse
   debolmente identificato sul pannello reale — intervalli posteriori che contengono
   zero — la domanda non è più computazionale: è *se tenere il leverage nel modello*.
   È la stessa questione già emersa nell'audit di Passo 4 (il controllo `ρ=0` con falso
   positivo reale).

2. **Prior informativo su `ρ`?** Il `.tex` prevede la Uniform(−1,1) come default e
   nomina lo shrinkage Fisher-`z` come opzione. A `T` corto (il regime del nowcast
   real-time) la scelta **cambia la coda della predittiva**, non solo la sua varianza
   MC. Serve la sua validazione: quanta credenza a priori sull'asimmetria è lecito
   mettere in un esercizio di Growth-at-Risk?

3. **Pooling dei parametri di volatilità fra fattori?** Se un fattore reale ha
   volatilità debole, il pooling parziale dei `(φ_k, σ²_k, ρ_k)` lo salverebbe
   prendendo in prestito informazione dagli altri. Ma prendere a prestito fra fattori
   **erode la lettura "`h_k` è la volatilità del fattore `k`"**, che è la ragione
   dichiarata (`subsec:vol-placement`) per cui abbiamo adottato Spec II invece di
   Spec I. È un trade-off fra identificazione e interpretabilità, e va deciso da chi
   possiede l'interpretazione.

4. **Il tetto informativo è accettabile?** Sotto Spec II la correlazione fra `ĥ_k` e la
   vera `h_k` di un fattore debole **non supera ~0.6, per nessun `T`**. Sotto la
   restrizione scalare sarebbe ~0.69, ma perderemmo la lettura per-fattore. La domanda
   è se una volatilità per-fattore stimata al 60% di correlazione serva alla tesi
   meglio di una volatilità comune stimata al 90%. **Questa è la vera domanda di
   modellazione aperta**, e P2 la pone senza risolverla.

---

## Appendice — provenienza delle misure

Tutti i numeri di questo documento sono riproducibili. Esperimenti eseguiti il
2026-07-10 in scratchpad, **senza modificare i sorgenti**:

| misura | come |
|---|---|
| `corr(Q)`, `c_k`, `R_ξ` reali | `load_warm_init('small'/'big')` + `sample_vol.logsq_corr_matrix` |
| tetto informativo `h_k` | smoother di Kalman a parametri veri, rumore log-χ²₁ (60 repliche) |
| P2 Branch B, `T ∈ {600,1200,2400}` | `fit_dfm_mcmc(sv=True, leverage=True, timing='lagged')`, `n_iter=600` |
| P2 Branch A, `T ∈ {600,1200}` | idem, `timing='contemporaneous'` |
| ASIS on/off | idem, `use_asis=True/False` |
| `λ(c)` | forma chiusa `(2/π)(c·arcsin c + √(1−c²))`, verificata MC (`N=4·10⁵`) |
| attenuazione di Omori | `E[zg]/E[g²]` con `OMORI10` vera, `N=2·10⁶` |
| kernel Family C non distorto | `draw_rho_scalar` su regressore esatto, `T ∈ {300..2400}` |
| `ESS(ρ)`, traccia | `fit_dfm_mcmc(n_iter=4000, burn_in=0)` + `diagnostics.ess` |
| `prop_rho ∈ {0.06,0.15,0.30}` | `fit_dfm_mcmc(n_iter=2500, burn_in=1000, lev_prop_rho=·)` |

Un esperimento è stato **scartato** perché mal costruito: un proxy "errors-in-variables"
che contaminava `z` con rumore indipendente, distruggendo per costruzione
`corr(η, z)` e producendo un collasso di `ρ̂` a zero che è un artefatto del proxy, non
del sampler (nel Gibbs `η` e `z` sono letti dallo **stesso** draw del path e conservano
l'accoppiamento). È segnalato qui perché un lettore potrebbe rifare lo stesso errore.
