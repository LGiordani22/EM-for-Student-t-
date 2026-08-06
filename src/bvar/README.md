# `src/bvar/` — BVAR per il nowcasting del PIL USA

Replica di **Cimadomo, Giannone, Lenza, Monti & Sokol (2022)**, *"Nowcasting with
large Bayesian vector autoregressions"*, Journal of Econometrics 231, 500–519.

Componente **indipendente** dal lavoro DFM / Growth-at-Risk del resto del repo.
Con il resto condivide solo l'infrastruttura dati stabile, **per import e mai per
modifica**: `src/forecast/release_calendar.py` (calendario dei rilasci) e
`data/processed/final/raw_levels_final.csv` (livelli grezzi).

Fonte di verità: il paper di Cimadomo. Dove il paper rimanda a un altro lavoro
per una formula, l'autorità su quel pezzo è il lavoro citato — GLP (2015) per il
campionatore degli iperparametri, BGR (2010) per le dummy observations, BGL
(2015) per il filtering condizionale, GRS (2008) per la trasformazione
mensile→trimestrale, GMR (2016) per la radice cubica. Vedi
`docs/BVARs/BVAR_CONTEXT.md`.

---

## I quattro modelli, un solo core

| Modello | Paper § | Che cos'è | Profilo dati | `p` |
|---|---|---|---|---|
| **Q-BVAR** | 2.1 | VAR trimestrale baseline. È anche lo **stadio 1 del C-BVAR** (nota 20) | `q_b` (30 serie) | 5 |
| **C-BVAR** | 2.4 + App. A | Radice cubica: da Φ, Ω del Q-BVAR a Φ_m = Φ^(1/3), poi Σ_εm. Filtro sulla **finestra terminale** — vedi sotto | `q_b` (30 serie) | 5 |
| **B-BVAR** | 2.3 | Blocking: ogni mensile → 3 serie trimestrali | `q_b` (30 serie) | 5 |
| **L-BVAR** | 2.2 | VAR mensile con le trimestrali latenti | `l` (37 serie) | 17 |

Il **core** è uno solo: NIW coniugato + Minnesota (eq. 2–3) + sum-of-coefficients,
imposti via dummy observations, più il campionatore GLP(2015) per λ, ψ, μ.
L'architettura **non si ramifica**: i quattro modelli differiscono in *cosa
passano al core*, non in *quale* core chiamano.

### L'invariante che tiene insieme tutto

> **Il core sampler non vede mai un NaN.**

Il dato mancante è gestito **fuori** dal core, sempre: dal Kalman (B, C) o dal
simulation smoother (L). Il core riceve una matrice `(T, n)` densa e restituisce
estrazioni di `(B, Σ)`.

---

## Perché due profili di variabili — 30 per Q/C/B, 37 per L

> **DECISO E CHIUSO** (approvato dal relatore). Per Q/C/B: **30 serie dal
> 1992-01**, con le 7 a partenza tardiva escluse. Non si riapre. Il seguito di
> questa sezione è il *perché*, che serve alla tesi, non una discussione aperta.

È la scelta teorica più importante del Gate 0 e vale la pena leggerla per intero:
sta in `config/bvar_series.json` → `notes.perche_due_profili`, e nel docstring di
`data.py`. In breve:

**La risposta non è "questi modelli non usano il Kalman".** Il Kalman c'è in
tutti e quattro. La differenza è **dove agisce**:

- **Q / C / B — Kalman a valle.** La stima è in forma chiusa sul posterior
  Normal-Inverse-Wishart e vuole un pannello pieno: §2.1, *"When all variables in
  the vector x_tq are available, the model can be readily estimated with standard
  Bayesian methods"*. Il Kalman entra **dopo**, solo per i nowcast come previsioni
  condizionate (§2.3).
- **L — Kalman a monte.** Il simulation smoother sta **dentro** il ciclo MCMC:
  §2.2, si estrae il dataset mensile completo *"including draws of the latent
  missing values"*.

Da cui la distinzione fra i due tipi di dato mancante:

| tipo | dov'è | chi lo risolve |
|---|---|---|
| **bordo frastagliato** | in fondo (non ancora pubblicato) | il Kalman, in **tutti** i modelli |
| **partenza tardiva** | in cima (la serie non esisteva) | **solo** l'L-BVAR, con lo smoother |

Q e B scartano le 7 serie a storia corta non perché non sappiano usare il Kalman,
ma perché il loro Kalman interviene **troppo tardi** per salvare i buchi in cima.

Le 7 escluse, con la prima osservazione: DGORDER (1992-02), TTLCONS (1993-01),
ISM_NMI (1997-07), JTSJOL (2000-12), Empire State (2001-07), PCEC96 (2007-01),
PPIFIS (2009-11). Tenerle tutte significherebbe partire dal **2009-11**, cioè
T≈66 con n=37: un regime large-n-small-T molto più estremo di quello del paper
(n=18, T≈130), dove non si distingue un bug da un artefatto del regime. Con
1992/n=30 abbiamo T=135, e la Tabella B.1 resta un sanity check valido.

## Il centraggio delle survey — `wn` per tutte e sei

> **DECISO E CHIUSO** (approvato dal relatore). Tutte e 6 le survey prendono
> `minnesota_centre = "wn"`, **`ISM_PRICES` compreso**. Non si riapre.

Le sei: le 4 ISM (PMI, PRICES, EMP, NMI), Empire State, Philly Fed. Nel profilo
`q_b` ne entrano 4 (NMI e Empire partono troppo tardi).

Il criterio è che **la forma del dato prevale sulla materia**: un indice di
diffusione è ancorato a una soglia neutra, limitato, mean-reverting per
costruzione, e centrarlo su `rw` metterebbe massa a priori su un processo che il
dato non può generare. `ISM_PRICES` era l'unico caso in tensione — misura
*prezzi*, e tutti i prezzi del pannello sono `rw` — ma è la **derivata
qualitativa** dei prezzi, non un livello di prezzi: la sua controparte in
log-livello è il PPI, non lei.

**Il codice degli autori conferma la regola**, ed è una conferma forte perché
non viene dal paper. Il vettore `stationary` di `Data/datasetCGLMS.mat` — quello
che diventa `pos`, e quindi sia `diagb(pos) = 0` sia `ydnoc(pos,pos) = 0` — ha
**3 elementi accesi su 28**:

| serie | `stationary` | natura |
|---|---|---|
| `USEPUINDXM` (EPU) | 1 | indice di incertezza |
| `UMCSENT` (Michigan) | 1 | **survey di sentiment** |
| `NAPM` (= ISM PMI) | 1 | **survey di diffusione** |
| le altre 25 | 0 | serie "dure", **prezzi compresi** |

Cioè: il **100%** dei soft indicator del loro pannello è `wn` e il **100%** delle
loro serie di prezzo è `rw`, senza eccezioni da nessuna delle due parti. Il
paper nomina solo PMI ed EPU (§3.1); UMCSENT mostra che la regola vera è "le
survey", non "quelle due serie". Il loro pannello **non contiene survey di
prezzi**, quindi su `ISM_PRICES` non poteva arbitrare: la nostra è
un'estensione della *loro* regola al caso che loro non hanno.

Conseguenza operativa, da non perdere di vista: `wn` non tocca solo la media a
priori del primo lag proprio, **spegne anche il sum-of-coefficients** su quella
serie (`ydnoc(pos,pos) = 0`). È il canale per cui il fix del Gate 5 ha portato
μ da 1.20 a 2.25.

---

## Mappa della teoria — dove sta scritta la §2.1

La teoria non sta in un documento separato: vive negli header dei moduli, accanto
al codice che implementa. Questo è l'indice.

| argomento | dove | sezione |
|---|---|---|
| **Il VAR e le due incognite** — `Y = XB + U`, B e Σ | `niw.py` | *IL VAR E LE DUE INCOGNITE* |
| Il prior NIW, coniugatezza, perché `dof = n+2` | `niw.py` | header + `default_dof` |
| Marginal likelihood: A.13, A.14 stabile, dummy (A.3) | `niw.py` | *NUMERICA* + `log_ml` |
| Paradosso di Bartlett (la ML dipende da `const_var`) | `niw.py` | `build_prior` |
| **Minnesota letta per righe** + filosofia BVAR | `dummies.py` | *LA MINNESOTA LETTA PER RIGHE* |
| **Media vs covarianza: i due assi** | `dummies.py` | *LA MEDIA E LA COVARIANZA...* |
| **λ e i due casi limite** (prior puro ↔ OLS) | `dummies.py` | *LAMBDA E I DUE CASI LIMITE* |
| **Eq. (3) fattore per fattore** (λ², 1/s², scale, diagonalità) | `dummies.py` | *L'EQ. (3) FATTORE PER FATTORE* |
| «Centrato su zero» ≠ «vincolato a zero» | `dummies.py` | omonima |
| Il refuso dell'eq. (3): perché `ψ_j` e non `ψ_i` | `dummies.py` | *IL REFUSO DELL'EQ. (3)* |
| Minnesota vs sum-of-coefficients: distinti e additivi | `dummies.py` | omonima |
| Sum-of-coefficients, Π = 0, *inexact differencing* | `dummies.py` | *IL SUM-OF-COEFFICIENTS* |
| Perché limita la componente deterministica | `dummies.py` | omonima |
| μ, casi limite, **tensione wn↔soc misurata** | `dummies.py` | *MU* + *LA TENSIONE wn↔soc* |
| Il meccanismo delle dummy observations | `dummies.py` | omonima |
| Contabilità dei gradi di libertà (la trappola) | `dummies.py` | `DummyStack` |
| **La mappa delle quattro varianti** — l'asse Q/C/B/L, cosa condividono | `__init__.py` | *LA MAPPA DELLE QUATTRO VARIANTI* |
| **La discrepanza dentro l'Appendice A** (A.9 letterale ≠ A.15) | `cube_root.py` | *UNA DISCREPANZA...* |
| A₀ nella mappa: c_m = (I+Φ_m+Φ_m²)⁻¹c, e l'ipotesi falsificata | `cube_root.py` | *LA COSTANTE A_0* |
| **Il C-BVAR fuori dalle ipotesi di GMR** — cosa dice GMR, le alternative, il fenomeno come risultato | `cube_root.py` | *CHE COSA DICE GMR* + *IL FENOMENO COME RISULTATO* |
| **P₀: `lyapunov_symm` sul sottospazio stabile**, e perché su finestra corta conta | `state_space.py` | punto 2 + `lyapunov_symm` |
| **Higham: perché, e quanto sposta** | `state_space.py` | punto 2 + `nearest_psd` |
| Chi usa la media mobile e chi no (Q e C sì, B e L no) | `__init__.py` | *L'AGGREGAZIONE MENSILE → TRIMESTRALE* |
| **L'ordine delle operazioni**: livelli → MA 3 mesi → log (nota 17) | `qbvar.py` | punto 1 |
| Le serie già in livello, e i due attriti (tassi, diffusione) | `qbvar.py` | punto 2 |
| **Il bordo × la MA**: mai una media parziale; la cecità del Q-BVAR | `qbvar.py` | punto 3 |
| L'interfaccia (Φ, Σ_ε, A₀) che il C-BVAR riuserà | `qbvar.py` | *L'INTERFACCIA CHE IL GATE 3...* |
| **I tre iperparametri: che cosa sono davvero** | `hyper.py` | *CHE COSA SONO DAVVERO...* |
| **Il salto gerarchico**, GLP vs Litterman | `hyper.py` | idem + header |
| Iperprior di GLP §III, nota sul ri-controllo dei ψ | `hyper.py` | *GLI IPERPRIOR* |
| Log-parametrizzazione della proposta (interpretazione) | `hyper.py` | *LA PARAMETRIZZAZIONE...* |
| `sample()` vs `step()`, requisito dell'L-BVAR | `core.py` | header |

**Notazione** (fissata ai Blocchi 1–2, da usare sempre — mai quella grezza dei
paper): `Y` (T×n), `X` (T×k), `B` (k×n), `Sigma` (n×n), `Psi = diag(psi)`,
`Omega` (k×k, diagonale), `dof = n+2`, `d_centre` (1=rw, 0=wn). Le due collisioni
di notazione dei paper (`d` usato da Cimadomo per due oggetti; `Ψ` che in BGR è la
covarianza e non la scala) sono documentate in `dummies.py`.

**I nostri numeri**, profilo `q_b` (n=30, p=5, T=135 trimestri, 1992Q1–2025Q3),
sul pannello del Gate 2 — medie mobili a 3 mesi, come da nota 17:

| | fine trimestre | medie mobili (Gate 2) | **+ fix soc (Gate 5)** | banda Tab. B.1 |
|---|---|---|---|---|
| λ | 0.5545 [0.506, 0.608] | 0.4875 [0.443, 0.536] | **0.5182** [0.490, 0.559] | 0.59–0.75 |
| μ | 1.0068 [0.819, 1.279] | 1.1983 [1.029, 1.414] | **2.2508** [1.848, 2.854] | 0.97–1.72 |

Stessa finestra, stesso seme. Fra la prima e la seconda colonna cambia solo
l'aggregazione (medie mobili). Fra la seconda e la terza cambia solo il fix al
sum-of-coefficients — l'azzeramento delle righe `wn`, `ydnoc(pos,pos)=0`.

**μ quasi raddoppia col fix**, e il verso conta: μ sta al *denominatore* della
dummy, quindi μ più grande = soc più **debole**. Tolto il conflitto col
Minnesota sulle quattro survey, i dati chiedono un soc molto meno intenso —
segno che prima parte della sua forza serviva a vincere una battaglia che non
avrebbe dovuto combattere.

**Ora λ è sotto la banda e μ sopra** (μ=2.25 contro 0.97–1.72), e non è un'anomalia: la Tabella B.1 riporta
**B, C e L — non il Q-BVAR**, e il λ ottimale cala al crescere di *n* (BGR 2010).
Il nostro sistema ha n=30 contro i 18 del paper con la stessa T, quindi la
marginal likelihood chiede più shrinkage. Il confronto resta un sanity check di
plausibilità, non un bersaglio. E i limiti ammissibili nel codice degli autori
sono `miu_max = 5`, quindi 2.25 è comodamente dentro il dominio che considerano.

## Gate 3 — il C-BVAR funziona, e la strada per arrivarci è il contributo

**Il C-BVAR è un modello vivo**, a due condizioni che il paper non dichiara e
che abbiamo dovuto ricostruire dal codice di replica degli autori.

### Le due condizioni

1. **La formula di accoppiamento** (`cube_root.coupling_matrix`). L'Appendice A
   ha **tre letture inequivalenti** e la scelta decide se Σ_εm è una matrice di
   covarianza:

   | formula | riproduce (A.15)? | spostamento Higham (mediana / p90) | autoval. neg. | ρ(A) |
   |---|---|---|---|---|
   | `literal` — (A.8) come stampata | no | — | — | — |
   | `authors` — codice MATLAB, **default** | no | **0.00% / 1.84%** | 2 su 30 | 1.36 |
   | `exact_a15` — la derivazione da (A.7) | **sì** | 75.77% / 99.16% | 9 su 30 | 2.28 |

   Default `authors`: una replica deve replicare i risultati pubblicati, e a
   p>2 il sistema (A.8) non ha soluzione esatta — il paper stesso dice «can be
   *approximately* solved».

   **`p=2` è il perno, e va raccontato bene** — trattazione completa
   nell'header di `cube_root.py`, sezione *«p = 2 È IL PERNO»*. In sintesi, è
   un conteggio: `C M = D` ha `(p−1)n²` equazioni e `n²` incognite, quindi è
   **esattamente determinato a p=2** e **sovradeterminato a p>2**. Da cui due
   regimi da non confondere:

   | | p = 2 | p = 5 (quello che il paper stima) |
   |---|---|---|
   | il sistema | quadrato, **soluzione unica** | sovradeterminato, **nessuna soluzione** |
   | residuo minimi quadrati sul **proprio** sistema | 0.0000 per **tutte e tre** | `exact_a15` **0.40%**, `authors`/`literal` **93%** |
   | che cosa distingue le tre | risolvono esattamente **sistemi diversi**: due sono **errori**, non approssimazioni | sono tutte approssimazioni, ma di sistemi mal posti in modo molto diverso |
   | conseguenza | l'incoerenza si **dimostra**, senza margini numerici | la scelta — mai dichiarata — decide la **stimabilità** |

   > **Attenzione all'errore facile:** *non* è vero che le tre letture
   > coincidano a p=2. Misurato su radici cubiche vere, a p=2 differiscono già
   > del 40–95% in norma. (Coincidono solo se si passa una companion invece di
   > una radice cubica — lì `Φ_m.1 = I` per costruzione e il fenomeno sparisce.
   > È una trappola in cui si cade facilmente scrivendo il test.)

   **E il paradosso da scrivere in tesi:** la lettura matematicamente corretta
   è quella numericamente inutilizzabile. `exact_a15` dà ρ(A)=2.28 e Σ_m
   indefinita in modo massiccio; `authors`, che è mal posta, dà ρ(A)=1.36 e una
   violazione da arrotondamento. Il meccanismo è identificato: (A.9) inverte
   `X ↦ X + A X A'`, che conserva il cono PSD solo per ρ(A)<1.

2. **La finestra del filtro** (`state_space.edge_window`). Si filtra solo il
   bordo e si prende la storia dai dati osservati, come in `crbvar.m`. Non è
   un espediente: sul nostro sistema è obbligatorio, e il degrado è **monotono**
   nella lunghezza della ricorsione —

   | finestra | 16 mesi | 24 | 48 | 96 | 200 | 403 |
   |---|---|---|---|---|---|---|
   | errore sui mesi osservati | **2.7e-04** | 1.1e-03 | 2.1e-03 | 4.6e-03 | 1.3e-02 | 1.6e-02 |

   La causa è che Φ^(1/3) di una companion quasi-difettiva è fortemente **non
   normale**: raggio spettrale 1.005 ma entrate fino a 1.8e4 e cond(V) ≈ 1e6.
   Su 16 passi non si vede, su 400 domina. **Il paper non spiega mai perché il
   filtro giri su una finestra corta: questa tabella è la risposta.**

   **Dove sta l'ancora**, ed è una correzione del 2026-08-02. La finestra parte
   da `endEstimT` — l'ultimo quarter-end *pienamente osservato*
   (`STEP2_CRBVAR.m` r.144) — e arriva in fondo, **righe di previsione NaN
   comprese**: sono quelle che il filtro riempie proiettando, ed è così che si
   ottiene la previsione a `horizon = 24`. Qui l'ancora stava invece in fondo
   all'**indice** (`i0 = T − (3p+1)`), che coincide con la loro convenzione solo
   su un pannello senza righe appese: con 24 mesi di NaN in coda finiva nel
   vuoto. Era il bloccante del C-BVAR nel Gate 6.

### Cosa è stato escluso, misurando

- **la dimensione *n*** — nessuna soglia: da n=6 a n=30 il comportamento non
  cambia qualitativamente.

### L'inizializzazione: era in questa lista, e non ci sta più

Ci stava scritto che «P₀ su otto ordini di grandezza (1e−2 → 1e6) lascia
l'errore a 1.59e-02 identico a tre cifre» e che «la varianza non condizionata
non esiste, quindi il `lyapunov_symm` degli autori fallirebbe anche da loro».
**Sono cadute tutte e due**, e per motivi diversi.

`lyapunov_symm.m` **non è** `solve_discrete_lyapunov`: è la routine di *Dynare*,
e la sua intestazione lo dice — *«If a has some unit roots, the function computes
only the solution of the stable subsystem»*. Schur ordinato, via le direzioni con
|λ|>1, Lyapunov sul resto, riproiezione: su un sistema esplosivo è **il caso per
cui è scritta**, non un caso che la rompe. Sul nostro Φ_m il blocco stabile è
135–142 direzioni su 150.

E la misura «l'init non conta» era fatta **sul campione pieno**, dove sono le
centinaia di osservazioni a fissare lo stato. Sulla finestra ancorata a
`endEstimT` le righe informative sono poche, e P₀ torna a essere un ingrediente
del nowcast — banda 90% su 2019Q2 (BEA 3.38), stesse estrazioni del Q-BVAR:

| P₀ | mediana | banda 90% | mediana e banda a `horizon = 24` |
|---|---|---|---|
| `1e0·I` (il vecchio ripiego) | 7.68 | **[−73.68, +314.44]** | **34.87** [−75.87, +207.21] |
| `1e-2·I` | 2.50 | [−15.93, +17.32] | 4.36 [−16.30, +15.50] |
| `1e-4·I` | 1.16 | [−3.05, +4.71] | 0.77 [−3.74, +5.58] |
| **`lyapunov_symm`** (gli autori) | **1.58** | **[−0.81, +5.38]** | 1.82 [−1.23, +4.50] |

Una sola stima (S=40, seme 1) e un solo seme per lo smoother: fra le righe
cambia **solo P₀**. A `horizon = 24` il ripiego isotropo non allarga soltanto la
banda, sposta la **mediana** a 34.87 — cioè a quell'orizzonte il nowcast non era
impreciso, era finto.

Il ripiego isotropo non era sbagliato per caso: era **invisibile** finché la
finestra sbagliata era lunga. È il secondo caso, in questo gate, in cui una
scelta implementativa non dichiarata decide se il modello è usabile.

### Il contributo per la tesi

Un'**analisi di riproducibilità**, non un risultato negativo. Cinque punti, e
il secondo è il perno:

1. **l'Appendice A è internamente incoerente** — (A.8)/(A.9) come stampate non
   riproducono (A.15), la forma chiusa che il paper stesso deriva due pagine
   dopo, e il codice degli autori usa una terza formula ancora;
2. **a p=2 questo si dimostra senza margini.** Il sistema è quadrato, la
   risposta giusta esiste ed è unica, e due letture su tre non la danno: non
   sono approssimazioni concorrenti, sono errori. È il caso che il paper
   *illustra*;
3. **a p=5 — il p che il paper stima** — la rappresentazione mensile esiste solo
   in modo approssimato, e la scelta fra le tre letture, mai dichiarata, decide
   se il modello è **stimabile** (0% contro 76% di sostituzione della matrice di
   covarianza; ρ(A) 1.36 contro 2.28). Col paradosso che la lettura corretta è
   quella inutilizzabile;
4. **gli autori incontrano la non-PSD e la gestiscono in silenzio** — `try/catch`
   con `D(D<0) = 1e-10` e un `qqFlag` mai pubblicato, più un riferimento a
   Higham *commentato*. Né Cimadomo né GMR (2016) discutono mai la semidefinita
   positività, e le ipotesi di GMR (sistema stabile, dati stazionari) non
   coprono il nostro caso;
5. **il filtro richiede una finestra corta e il paper non dice perché.** La
   tabella del degrado monotono è la risposta.

*Correzioni a versioni precedenti di questa sezione*, tenute perché il percorso
è istruttivo: si era scritto che il problema **non** dipendeva dalla lettura di
A.9 (dipende quasi solo da quella), che c'era una **soglia in n** (non c'è, era
misurata su VAR sintetici troppo benigni), che il C-BVAR **non era
recuperabile** (lo è), che a p=2 le tre letture **coincidessero** (non
coincidono: differiscono del 40–95%; coincidono solo se si passa per errore una
companion invece di una radice cubica) e — la più recente, 2026-08-02 — che
**l'inizializzazione non contasse** e che il `lyapunov_symm` degli autori
«fallirebbe anche da loro» (non fallisce: risolve il sottospazio stabile, ed è
la differenza fra una banda `[−2.4, +6.7]` e una `[−74, +314]`). Cinque
conclusioni ribaltate da misure successive: un buon esempio di quanto sia facile
attribuire a un metodo ciò che dipende da una scelta implementativa non
dichiarata.

Trafila completa negli header di `cube_root.py` e `state_space.py`.

---

## Struttura del pacchetto

Un file nasce al gate che lo richiede, non prima.

```
spec.py          ✅ Gate 0   MinnesotaSpec, Hyper, BVARSpec + lettura config
data.py          ✅ Gate 0   wrapper dati: raw levels → known_at → log/livello
tests/           ✅ Gate 0   un test per gate

dummies.py       ✅ Gate 1   Yd, Xd: Minnesota + sum-of-coeff (BGR 2010 eq. 5)
niw.py           ✅ Gate 1   posterior NIW, draw di (B,Σ), marginal likelihood
hyper.py         ✅ Gate 1   iperprior + Metropolis su γ (GLP 2015 §III, App. B)
core.py          ✅ Gate 1   IL CORE SAMPLER — sample() e step()
simulate.py      ✅ Gate 1   DGP per i recovery test

qbvar.py         ✅ Gate 2   wrapper trimestrale: MA 3 mesi + core + output App. A
cube_root.py     ✅ Gate 3   Φ^(1/3), selezione radici, Σ_εm (A.9' + A.10), c_m
state_space.py   ✅ Gate 3   Higham + finestra/P₀ + ciclo filtro su kalman.py
                            — è QUI che sta il filtro condizionale (BGL 2015)
cbvar.py         ✅ Gate 3   i tre stadi + nowcast (crescita annualizzata)
bbvar.py         ✅ Gate 4   impilamento + filtro  (dopo l'L-BVAR)
simsmoother.py   ✅ Gate 5   Durbin-Koopman, codice NUOVO
lbvar.py         ✅ Gate 5   il ciclo MCMC a tre passi (§2.2)
evaluate.py      ✅ Gate 6   il ciclo settimanale: calendario, cache, CSV
                            + checkpoint: la passata si interrompe e riprende
                            + parallel_blocks: i confini dei blocchi paralleli
figures.py       ✅ Gate 6   involucro su src/forecast/figures.py
metrics.py       ✅ Gate 6   involucro su compute_metrics + compare_nyfed
```

L'orchestratore delle passate non sta qui ma in **`scripts/run_all.sh`**: DFM e
BVAR sullo stesso calendario, poi figure, NY Fed e tabelle di confronto.

### Gate 6 — l'uscita, e perché è tutta riusata

Non si lanciano più a mano: l'orchestratore è **`scripts/run_all.sh`**, che
gira DFM e BVAR sullo stesso calendario e poi figure, NY Fed e tabelle. I tre
comandi restano per girare un pezzo solo:

```bash
OMP_NUM_THREADS=1 python -m src.bvar.evaluate --start 2007-05-04 --end 2007-07-27
python -m src.bvar.figures
python -m src.bvar.metrics
```

I percorsi **non li decide questo pacchetto**: li decide `src/output_layout.py`,
e tutto sta sotto `forecast_weekly/` insieme al DFM. Non è cosmetico — i CSV del
BVAR sono l'ingresso della stessa catena di tabelle del DFM, e un albero
parallelo `output/bvar/` (come diceva questo README fino al 2026-08-06) era il
motivo per cui le uscite finivano sparse.

```
output/forecast_weekly/
  csv/bvar/               il CSV lungo + i quantili .npz   (l'ingresso di tutto)
  csv/bvar/logscore/      i log score grezzi, uno per blocco
  bvar/qbvar|cbvar|bbvar|lbvar/   una figura cg8a per modello
  bvar/rmse/              RMSE vs AR(2), media espandente e NY Fed
                          + la figura RMSE per orizzonte con le tre fasi
  bvar/logscore/          log predictive score: riepiloghi + figura
  comparison/             BVAR-vs-DFM, per fase, backcast compreso
```

`figures.py` e `metrics.py` **non disegnano e non calcolano**: chiamano
`src/forecast/{figures,compute_metrics,compare_nyfed}.py` con altri argomenti.
È la ragione per cui il contratto CSV è `weekly_nowcast.COLUMNS` alla lettera —
e il contratto vale solo se le colonne sono anche RIEMPITE: `realizzato_bea`
vuoto non dà errore, dà figure senza pallini e RMSE su zero righe.

### L'unità di parallelismo, e dove si taglia

`evaluate.py` è **seriale di proposito**: nessun thread, nessun pool. Il
parallelismo sta fuori, un processo per blocco, ed è `scripts/run_all.sh` a
lanciarli.

L'unità è il **blocco-trimestre** — una stima completa più le sue settimane di
riuso — indipendente dalle altre per costruzione, perché ogni stima riparte dal
pannello a `as_of` e non dallo stato della precedente.

**I confini li dà `parallel_blocks`** (flag `--print-blocks`), e cadono sulle
settimane di **stima piena**, cioè dove le release BEA le mettono. Non è un
dettaglio di comodità: la prima settimana di ogni blocco viene comunque forzata a
piena da `estimation_weeks`, quindi tagliare *lì* è l'unico confine che non costa
una stima in più e non sposta una riga. Misurato sul 2007-2025:

| | blocchi | stime piene | settimane che cambiano valore |
|---|---|---|---|
| passata continua (il riferimento) | 1 | 77 | — |
| **taglio sulle stime piene** | **77** | **77** | **0** |
| taglio annuale (com'era prima) | 19 | 95 | 18 |

Il taglio annuale cadeva a Capodanno, in mezzo a un trimestre: promuoveva 18
settimane da riuso a stima fresca, e con loro il nowcast di quelle righe
dipendeva da come era stato affettato il lavoro per il cluster, non dal modello.

I 77 blocchi hanno mediana 13 settimane (min 4, max 14) e stanno in una sola
ondata su una macchina da qualche decina di core in su. Il vincolo pratico non
sono i core ma la **RAM**: ogni processo carica il pannello e tiene la propria
cache (tetto `--max-cache-mb`, default 1500).

### Cosa si importa da `kalman.py` (e cosa no)

`src/kalman.py` **non si tocca**: è condiviso da `em/`, `mcmc/`, `forecast/`.

Riusabili così come sono, perché sono pura algebra matriciale:

- `kalman_predict(f_prev, P_prev, A_tilde, Q_tilde)`
- `kalman_update(f_pred, P_pred, y_t, W_t, Lambda_tilde, R_tilde)`
- `build_selection_matrix(y_t)` / `build_all_selection_matrices(Y)`

**Non** riusabili, perché cablano la struttura DFM: `build_A_tilde`,
`build_Q_tilde`, `build_Lambda_tilde`, `build_R_tilde` (companion a 5 blocchi e
pesi Mariano-Murasawa), `kalman_filter`, `run_kalman` (firmano su `theta` del
DFM). Il BVAR ha una companion form diversa e non usa MM: nel C-BVAR
l'aggregazione è la media mobile a 3 mesi **sui dati** (nota 17), non un peso
nell'equazione di osservazione.

Quindi `bvar/state_space.py` scriverà il proprio ciclo `predict`/`update` sulle
primitive importate. **Riuso per import, non riscrittura del filtro.**

---

## Convenzioni

- **Import**: sempre `from src.…`.
- **Configurazione**: nessuna scelta di modellazione è cablata nel codice.
  Serie, profili, mappa log/livello e centraggio rw/wn stanno in
  `config/bvar_series.json`, con la motivazione accanto. Il vettore `d_centre`
  dell'eq. (2) è **costruito dalla config**, mai scritto a mano.
- **Dati**: pseudo-real-time. Si maschera un unico file corrente col calendario;
  si riproduce la tempistica dei rilasci, non le revisioni. Il paper invece
  ricostruisce vintage veri (§3.1). Semplificazione **dichiarata**.
- **Recovery test a ogni gate** prima di passare al successivo.

## Test

Uno per gate, più i test di blocco del Gate 1. Ognuno si esegue da solo:

```
python -m src.bvar.tests.test_data        Gate 0   dati, profili, buchi
python -m src.bvar.tests.test_minnesota   Gate 1   eq. (2)-(3) analitica
python -m src.bvar.tests.test_dummies     Gate 1   Minnesota via dummy (l'oracolo)
python -m src.bvar.tests.test_soc         Gate 1   sum-of-coefficients
python -m src.bvar.tests.test_gate1       Gate 1   RECOVERY del core sampler
python -m src.bvar.tests.test_gate2       Gate 2   RECOVERY del wrapper trimestrale
python -m src.bvar.tests.test_gate3      Gate 3   mappa cube-root + stato-spazio
python -m src.bvar.tests.test_gate4      Gate 4   blocking + bordo frastagliato
python -m src.bvar.tests.test_gate5      Gate 5   simulation smoother (oracolo esatto)
python -m src.bvar.tests.test_gate6      Gate 6   calendario: look-ahead, dato BEA, le tre fasi
python -m src.bvar.tests.test_resume     Gate 6   ripresa: interrotta == non interrotta (~3 min)
python -m src.bvar.tests.test_p0         —        P₀ centralizzata sui 4 modelli
python -m src.bvar.tests.test_mixing     —        ESPERIMENTO sul mescolamento (~6 min)
python -m src.bvar.cbvar                 Gate 3   CHECK END-TO-END (~3 min)
```

`test_mixing` non è un test di gate: non c'è niente da «passare», produce
**numeri**. Sta in `tests/` perché vada rieseguito quando serve — è la misura
che chiude il punto 5 del registro, e i suoi output sono materiale per la tesi.

L'ultimo non sta in `tests/` perché richiede una stima vera. È il controllo che
chiude il Gate 3: sui trimestri **già osservati** il C-BVAR deve *riprodurre* il
dato BEA, non approssimarlo — lì lo smoother non ha nulla da stimare, quindi uno
scarto rivelerebbe un errore in un punto qualsiasi della catena. Misurato: entro
**0.03 punti percentuali** su 2018Q3, 2018Q4 e 2019Q1. Sul trimestre obiettivo
si controlla invece la *forma*: banda larga e contenente il realizzato.

Il Gate 2 controlla le tre cose che il wrapper può sbagliare in silenzio:
l'ordine `log(media)` vs `media(log)`, che una media parziale non venga **mai**
formata al bordo, e che `companion()` sia davvero la Φ dell'eq. (A.1) — questo
verificato facendo iterare lo stato, non per ispezione.

---

## Stato dei quattro modelli — dove siamo al 2026-08-06

**Gate 0-1-2-3-4-5 CHIUSI DEFINITIVAMENTE.** Tutti e quattro i modelli girano
end-to-end e producono un nowcast sano; le scelte di modellazione sono state
approvate dal relatore e non si riaprono. **Resta solo il Gate 6**
(valutazione real-time + confronto NY Fed): il codice c'è ed è stato provato
end-to-end da uno SMOKE, la **passata vera non è ancora stata lanciata**.

| modello | gate | nowcast 2026Q2 | stato | note |
|---|---|---|---|---|
| **Q-BVAR** | 2 | — (baseline) | **CHIUSO** | nessun punto aperto. ρ(A) ≈ 1.012 è documentato, non è una questione |
| **C-BVAR** | 3 | — | **CHIUSO come modello** | il residuo di (A.8) a p>2 non è più un punto aperto ma **il contributo di riproducibilità**: vedi la sezione Gate 3 e l'header di `cube_root.py`. Due divergenze paper↔codice da dichiarare: la formula di accoppiamento e la non-PSD (Higham contro il loro `D(D<0)=1e-10`) |
| **B-BVAR** | 4 | **1.46%** [−0.98, 3.95] | **CHIUSO** | ESS/iter di λ = 0.015 con accettazione al 20.8%: **misurato e spiegato** (vedi *Il mescolamento…*), è la legge 1/d, non un bug. Nowcast e test sani |
| **L-BVAR** | 5 | 2.21% [−0.20, 4.53] | **CHIUSO** | `sample_start` 1985-01 confermato. Due divergenze da dichiarare: il box sui `psi` (non è nel paper) e `P0 = 0` |

---

## Il registro dei punti aperti

**Tutti i punti di modellazione sono chiusi.** Il registro resta come traccia di
*come* si è deciso — serve alla tesi, non al lavoro.

| # | punto | esito | dove sta scritto |
|---|---|---|---|
| 1 | Il residuo di (A.8) a p>2 | **CHIUSO — diventa il contributo.** Non è una scelta da fare ma un risultato da scrivere: p=2 esatto / p=5 sovradeterminato, residuo 0.40% contro 93% | `cube_root.py`, sezione *«p = 2 È IL PERNO»* |
| 2 | `ISM_PRICES`, rw o wn? | **CHIUSO: `wn`**, con tutte le survey. Confermato dal vettore `stationary` degli autori | sezione *Il centraggio delle survey*, `config/bvar_series.json` |
| 3 | `sample_start` del profilo `l` | **CHIUSO: 1985-01.** Lo smoother ha girato davvero al Gate 5 e le 298 righe latenti di PPIFIS non hanno prodotto patologie | `config/bvar_series.json` |
| 4 | Mancano le righe di previsione | **CHIUSO** — `data.append_forecast_rows`, e `horizon` è ora parametro di tutti e quattro i `fit()`/`fit_reuse()` | `data.py`, `evaluate.py` |
| 5 | Il mixing degli iperparametri | **CHIUSO — misurato.** È la legge 1/d di un random-walk Metropolis, una proprietà del metodo: si descrive, non si risolve | sezione *Il mescolamento…*, `tests/test_mixing.py` |
| 6 | Il profilo `l` non ha campione prima del 2010 | **CHIUSO: strada (a)**, `data.drop_empty_series`, approvata dal relatore. `n` varia nel tempo (35→36→37 fra 2007 e 2010) e va dichiarato in tesi | `config/bvar_series.json`, `lbvar.last_full_row`, sezione sotto |

### 6 — Il profilo `l` non era utilizzabile prima del 2010

Trovato dall'oracolo del Gate 6 (`tests/test_gate6.py`), a `as_of = 2008-06-20`.

`lbvar` cerca `last_full_row`, l'ultima riga con **tutte e 37** le serie
osservate — è il `lastFull` degli autori (`lbvar.m` r.18). Ma il profilo `l`
contiene PPIFIS (parte **2009-11**) e PCEC96 (**2007-01**): prima di quelle date
nessuna riga è piena e l'L-BVAR non ha campione di stima. Solleva `ValueError`,
quindi non è un rischio silenzioso — ma **blocca il blocco 2007-2010**.

Non è un baco del calendario: l'oracolo conferma che i pannelli costruiti sono
puliti. È una **decisione di modello** — quali serie entrano a quale vintage — e
va presa dal relatore. Le tre strade:

| | costo | effetto |
|---|---|---|
| **a)** profilo `l` dipendente da `as_of` — **SCELTA, approvata dal relatore** | nullo | `n` varia nel tempo. È la ricostruzione onesta del real-time, non un compromesso |
| **b)** profilo `l` ridotto e **fisso** per il blocco 2007-2010 | una config in più | due modelli L diversi nei due blocchi, da dichiarare |
| **c)** L-BVAR solo sul blocco 2016-2019 | nullo | si perde il confronto L-BVAR ↔ DFM sulla crisi |

### La (a), implementata — `data.drop_empty_series`

**L'argomento non è «difendibile», è che (a) è la scelta più fedele al
real-time.** Nel 2008 PPIFIS non esisteva: un previsore reale non poteva usarla
in nessun modo, nemmeno come latente. Una colonna interamente NaN **non è uno
stato latente, è puro prior** — nessuna osservazione propria, nessuna
correlazione da cui informarsi — e per giunta aggiunge un ψ alla proposta del
Metropolis, peggiorando il mixing per la legge 1/d che abbiamo appena misurato.
È anche coerente con quel che già facciamo: le 7 serie tardive sono escluse da
Q/C/B perché non esistono ancora; qui l'L le include quando esistono e le scarta
quando non esistono — **la stessa regola, applicata alla data invece che al
profilo**.

Le serie **parzialmente** osservate restano, e il loro passato lo riempie lo
smoother: PCEC96 a `as_of = 2008-06` ha 17 mesi di dati e ~260 latenti, ed è
esattamente il caso che il profilo `l` esiste per trattare.

La regola ha **due fasi**, e la seconda non è un'eccezione ma la stessa regola
portata alla sua conseguenza — trovata misurando, a `as_of = 2010-01-01`:

1. esce chi ha **zero** osservazioni;
2. finché non esiste una riga pienamente osservata **sull'insieme trattenuto**,
   esce la serie che comincia più tardi. (A 2010-01 PPIFIS ha 2 mesi di storia,
   quindi la fase 1 la terrebbe — ma nessun quarter-end è ancora pieno: il
   2009-09 non ha PPIFIS, il 2009-12 non ha ancora il PIL.)

Così **`last_full_row` è definito sull'insieme delle serie effettivamente
trattenute, per costruzione e non come caso particolare**. Misurato:

| `as_of` | n | `lastFull` | non ancora disponibili |
|---|---|---|---|
| 2007-01 | 35 | 257 | PCEC96, PPIFIS |
| 2007-06 | 36 | 263 | PPIFIS |
| 2008-06 | 36 | 275 | PPIFIS |
| 2010-01 | 36 | 293 | PPIFIS |
| 2010-04 | **37** | 296 | — |
| 2016-06 | 37 | 371 | — |

Monotono e senza salti. **Da dichiarare in tesi: `n` varia nel tempo** (35 → 36
→ 37 fra il 2007 e il 2010, poi stabile).

> **Il punto in cui è facile confondersi, e va detto:** *non è lo smoother che si
> rompe.* L'L-BVAR ha il simulation smoother dentro la stima ed è proprio per
> questo che il profilo `l` ha 37 serie invece di 30 — lo smoother sa fare il suo
> lavoro. Quel che manca è **l'ancoraggio**: `last_full_row` è una convenzione
> che fissa la fine del campione di *stima* pretendendo una riga con tutte le
> serie osservate. È il campione che resta senza appiglio, non lo smoother che
> fallisce.

Il blocco **2016-2019 non è toccato**: lì tutte e 37 le serie esistono.

Le due decisioni di perimetro (campione 1992/n=30; `wn` per tutte le survey)
sono state approvate dal relatore e hanno la loro sezione dedicata sopra.

## Il mescolamento degli iperparametri — misurato, spiegato, chiuso

> **CHIUSO.** È la legge 1/d di un random-walk Metropolis: una proprietà nota
> del metodo che il paper prescrive, non un difetto dei nostri modelli. Si
> descrive in tesi, non si "risolve". Trattazione completa nell'header di
> `hyper.py`, sezione *IL MESCOLAMENTO*; la misura è
> `python -m src.bvar.tests.test_mixing`.

### Cos'è l'ESS, e perché conta

Una catena MCMC non produce estrazioni indipendenti: produce una passeggiata.
«Mescolare male» vuol dire **strisciare** — muoversi per passi così piccoli, o
così correlati, che mille estrazioni raccontano quello che ne racconterebbero
venti indipendenti. La misura è l'**Effective Sample Size**:

$$\text{ESS} = \frac{S}{\tau}, \qquad \tau = 1 + 2\sum_{t\ge1}\rho_t$$

cioè *a quante estrazioni indipendenti equivale la catena*. `ESS/iterazione =
0.015` significa che ne servono ~67 per una indipendente: 1000 estrazioni ne
valgono 15. Un ESS basso **non distorce** — la catena resta valida e converge
alla posteriore giusta — ma rende la stima **imprecisa**: l'errore Monte Carlo
è `sd/√ESS`, non `sd/√S`. L'autocorrelazione a lag 1 è la stessa cosa vista da
vicino: `ρ₁ = 0.986` vuol dire che λ è quasi identico a quello di prima.

**ESS e accettazione non sono la stessa cosa**, e il B-BVAR lo dimostra:
accettazione **20.8%** (il bersaglio dell'Appendice B) e ESS/iterazione
**0.015**. Cioè accettare quasi sempre e muoversi pochissimo.

### Perché qui si striscia — e perché *non* è la ragione che sembra

La spiegazione naturale è la **cresta**: gli iperparametri fanno tutti la stessa
cosa da direzioni diverse (λ e i ψ entrano *moltiplicati* nell'eq. 3; μ regola
la persistenza che la Minnesota già governa), quindi la posteriore sarebbe un
insieme di combinazioni quasi equivalenti, e un random-walk lungo una cresta
avanza per costruzione lentamente.

**Misurato, non regge come causa principale.** Sulla posteriore vera (Q-BVAR,
n=30, correlazioni implicate da W al modo):

| | |
|---|---|
| corr(λ, μ) | **−0.410** |
| \|corr\| massima fuori diagonale | −0.496 |
| \|corr\| **mediana** fuori diagonale | **0.028** |
| cond(W) | 58.8 |

Le correlazioni ci sono, e sono quelle attese — λ contro μ a −0.41 è
esattamente la tensione Minnesota↔sum-of-coefficients. Ma la posteriore **non è
una cresta patologica**: la correlazione tipica fra due coordinate è 0.03.

E c'è una ragione strutturale per cui non poteva esserlo: **la proposta
dell'Appendice B usa già W**, l'Hessiana inversa al modo, che *è* la forma
locale della posteriore, correlazioni comprese. Proporre con covarianza `c·W`
significa proporre già allineati alla cresta. Ciò che W non può correggere è la
**dimensione**.

### La causa vera: la legge 1/d

Per un random-walk Metropolis in dimensione *d*, anche con la scala ottimale
l'efficienza per iterazione decade come **1/d** (Roberts–Gelman–Gilks 1997): la
proposta muove tutte e *d* le coordinate insieme, e la probabilità che una
mossa congiunta sia buona in tutte le direzioni cala con *d*. Qui `d = 2 + n`.

L'esperimento isola *d* a modello fermo — Q-BVAR (forma chiusa, nessuno
smoother), stessa T, stesso seme, stesso Metropolis, sottoinsiemi annidati:

| n | d | acc | c | ESS/it λ | ρ₁ λ | **ESS/it × d** |
|---|---|---|---|---|---|---|
| 6 | 8 | 17.3% | 1.091 | 0.0355 | 0.938 | **0.28** |
| 12 | 14 | 18.2% | 0.514 | 0.0269 | 0.955 | **0.38** |
| 18 | 20 | 20.5% | 0.307 | 0.0117 | 0.973 | **0.23** |
| 24 | 26 | 19.5% | 0.285 | 0.0104 | 0.976 | **0.27** |
| 30 | 32 | 19.9% | 0.220 | 0.0087 | 0.977 | **0.28** |
| *ψ congelati* | 2 | 19.9% | 9.403 | 0.1099 | 0.781 | **0.22** |

**pendenza di log(ESS/it) su log(d) = −1.10**, contro la previsione teorica
−1.00. Tre letture:

1. **il prodotto ESS/it × d è piatto** su un fattore 4 di dimensione (spread
   1.6×). Il mescolamento è governato da *d*, punto;
2. **l'accettazione è a bersaglio in ogni cella** (17–21%), e `c` scende da 1.09
   a 0.22 al crescere di *d* — la taratura 1/√d che la procedura trova da sola.
   Non è un passo mal tarato;
3. **congelare i ψ non è speciale.** L'ESS salta 13× (previsti 32/2 = 16×) ma il
   prodotto resta 0.22, in linea con tutti gli altri. I ψ sono 30 coordinate
   come le altre, non un blocco patologico da isolare.

I due modelli veri stanno sulla stessa legge: L-BVAR d=39 → 0.053 (prodotto
2.07), B-BVAR d=86 → 0.015 (prodotto 1.29). La *costante* è diversa da quella
del Q-BVAR (0.28) — dipende da modello e dati — ma i due modelli veri stanno
entro 1.6× l'uno dall'altro e l'ordinamento è quello previsto.

> Da cui la lettura del B-BVAR, che ribalta come lo si raccontava:
> **non mescola peggio perché sia un modello peggiore, ma perché il blocking gli
> raddoppia la dimensione degli iperparametri** (84 ψ invece di 37).

**Traccia nel codice degli autori**, che non lo dicono mai: `MCMCconst` è
cablato a mano, un valore per modello — `qbvar.m` **1**, `crbvar.m` **0.5**,
`bbvar.m` **0.14**, `lbvar.m` **1.6**. Hanno dovuto stringere la proposta di un
ordine di grandezza proprio dove *d* è più grande. È la taratura 1/√d fatta a
occhio.

### Perché il nowcast non eredita il problema

Il ciclo ha **due popolazioni**, e vanno tenute distinte:

| | | ESS/iterazione |
|---|---|---|
| λ, μ, ψ | Metropolis | **0.01 – 0.05** |
| B, Σ, pannello latente | coniugata / smoother | **0.95 – 1.00** |

La seconda riga non è fortuna, è come è costruito il ciclo: dato γ, l'estrazione
di **(B, Σ) è esatta dalla Normal-Inverse-Wishart** — forma chiusa, nessun
rifiuto, nessuna memoria dell'iterazione precedente — e il simulation smoother
gira con **randomness fresca**. Il nowcast è funzione di (B, Σ, stato), non di γ;
γ entra solo come regolatore dello shrinkage, e la sua posteriore è concentrata
(sd(λ) ≈ 0.016). Quindi la lentezza di γ modula un ingrediente che quasi non
varia, e non si propaga.

Misurato su **due** modelli:

| | L-BVAR | B-BVAR |
|---|---|---|
| ESS del nowcast | ~1000/1100 | **1000/1000** |
| corr(nowcast, λ) | — | −0.047 |
| corr(nowcast, μ) | −0.045 | −0.010 |
| mediana del nowcast, estrazioni con μ alto vs basso | −0.14 pp | dentro l'errore MC |

**La densità del nowcast è di fatto iid** anche mentre gli iperparametri
strisciano.

**Il limite onesto, da scrivere così e non più forte di così:** le bande
esplorano bene l'incertezza su **parametri e stato**, meno bene quella sugli
**iperparametri**. È un'attenuante *misurata* — sd(λ) piccola, correlazione ≈ 0
— non un'assoluzione teorica. Se il deliverable fosse una banda *su* λ, il
problema sarebbe reale; il deliverable è il nowcast, e lì l'impatto è
trascurabile.

### Perché non si "risolve", e cosa fanno gli autori

Le vie d'uscita esistono — bloccare i ψ, fissarli al modo (`MNpsi = 0` esiste
anche da loro), passare a un campionatore con gradiente (MALA/HMC, che scalano
come d^(1/3) o d^(1/4)) — ma sono tutte **deviazioni dall'Appendice B**, e il
beneficio ricadrebbe su una banda che non è il deliverable. In più il blocking
**non è un rimedio ma un cambio di modello**: congelare i ψ toglie la loro
incertezza dalla posteriore invece di campionarla male, e guadagna esattamente
il fattore che la legge 1/d prevede, né più né meno.

Tentativi fatti e falliti, tenuti perché il percorso è istruttivo: taratura di
`c` + spazzate multiple sull'L-BVAR (ESS 0.053 → 0.030, **peggiorato** — lì
agisce anche il target mobile); alzare l'accettazione al 20% sul B-BVAR (fatto,
ESS invariato).

**E il confronto che vale in tesi:** nel codice di replica degli autori **non
c'è nessun ESS, nessun R̂, nessuna autocorrelazione, nessun thinning**. L'unica
diagnostica è `r.mcmc.ACCrate`, calcolata a posteriori come frazione di
iterazioni in cui λ è cambiato, e mai usata per agire; l'unico accenno alla
convergenza in tutto il pacchetto è il commento su `Ndrawsdiscard`
(`setpriors.m` r.54). **Con la sola accettazione a bersaglio il loro
campionatore sembra sano** — che è esattamente la situazione del nostro B-BVAR.
Che il mescolamento degli iperparametri sia povero è un fatto che il loro
strumentario diagnostico non può vedere.

---

### 4 — Il pannello si ferma ai dati: mancano le righe di previsione

Trovato al **Gate 4**, verificando altro, e vale per **tutti e quattro** i
modelli. Gli autori troncano il pannello a `nowcastM + horizon` con
`horizon = 24` mesi (`STEP1_BBVAR.m` r.98): le righe oltre l'ultimo dato sono
**tutte-NaN**, e lo smoother, non avendo nulla da cui aggiornare, le riempie
iterando la transizione — cioè produce **il sentiero di previsione a 8
trimestri**, con bande vere, nello stesso passaggio del nowcast.

Il nostro `fit()` si ferma invece all'ultima riga di dati: dà il nowcast e
basta. Non è un difetto del Gate 4 — il gate è il nowcast — ma il Gate 6 si
chiama *forecast/evaluate* e lì gli orizzonti servono. La modifica è piccola e
localizzata (appendere `horizon/3` righe di NaN al pannello blocked prima dello
smoother), ma **va fatta prima** di lanciare la passata real-time: rifarla dopo
significherebbe ripagare i 107 vintage.

Corollario di calendario: nel loro schema `X` è già tagliato a
`nowcastM + horizon` **prima** del blocking, quindi le righe future entrano
nell'impilamento come trimestri interi. Replicandolo va tenuto l'allineamento
del blocking (primo mese del trimestre), che `block_panel` verifica.

### 5 — Il mixing degli iperparametri: come ci si è arrivati

> **Questa sottosezione è il PERCORSO, non la conclusione.** La conclusione —
> misurata e definitiva — sta sopra, in *«Il mescolamento degli iperparametri»*.
> Si tiene per intero perché tre diagnosi successive sono state falsificate
> l'una dall'altra, ed è materiale onesto per la tesi.
>
> La traiettoria: *«è il target mobile»* → falsificata dal B-BVAR → *«è la
> dimensione della proposta»* (sospetto, due punti) → **confermata dalla misura**
> (`test_mixing`, pendenza −1.10 contro −1.00 previsto). E lungo la strada è
> caduta anche la spiegazione «è una cresta»: la cresta è mite (|corr| mediana
> 0.028) e la proposta `c·W` è già allineata ad essa.

**Deciso con Lorenzo: è un risultato, non un bug.** La taratura di `c`
e le spazzate multiple di Metropolis furono provate e **peggiorarono** l'ESS
(0.053 → 0.030), perché il target si muove sotto la catena — il pannello latente
è diverso a ogni iterazione. Non ri-proporle.

**IL PRIMO ESPERIMENTO DI CONTROLLO HA FALSIFICATO LA PRIMA DIAGNOSI.**
Il B-BVAR è il termine di paragone giusto: stesso core, stesso Metropolis, ma
**Kalman a valle** invece che dentro il ciclo — il pannello di stima è calcolato
una volta sola prima di `sample()`, quindi il target degli iperparametri **non
si muove**. L'aspettativa era: se lì il mixing è sano, «è il pannello latente»
è confermata per differenza. Misurato sulla catena definitiva (1000/500):

| | L-BVAR (target mobile) | B-BVAR (target **fisso**) |
|---|---|---|
| ESS/iter di λ | 0.053 | **0.015** |
| autocorr lag 1 di λ | 0.939 | **0.986** |
| accettazione | 5.8% | **20.8%** (a bersaglio) |
| dim. della proposta | 39 | 86 |

**Il B-BVAR mescola peggio, col target fisso e l'accettazione centrata sul 20%
dell'App. B.** Quindi il target mobile *non* può essere la causa dell'ESS basso:
il fenomeno si osserva, più marcato, dove per costruzione il target non si muove.

Cosa sopravvive e cosa no:

- **resta valido** che nell'L-BVAR le spazzate multiple *peggiorarono* l'ESS —
  quello sì è spiegato dal target mobile (si equilibra sul pannello corrente,
  poi il pannello cambia). Il tentativo fallito resta ben diagnosticato;
- **non regge** che il target mobile spieghi l'ESS basso in sé;
- **nuovo sospetto: la dimensione della proposta** (39 → 86). Era un'ipotesi
  **con due punti, non una misura**.

**IL SECONDO ESPERIMENTO HA CONFERMATO IL SOSPETTO** (`tests/test_mixing`,
2026-08-01): isolando *d* a modello fermo, ESS/it × d è piatto e la pendenza è
−1.10 contro il −1.00 previsto dalla teoria del random-walk Metropolis. È la
legge 1/d. **Ed è anche caduta la lettura «è una cresta»**: la posteriore è mite
(|corr| mediana 0.028, cond(W) 58.8) e la proposta `c·W` è già allineata alle
correlazioni per costruzione. Vedi la sezione dedicata sopra.

Quel che resta valido dell'intuizione di allora: la firma non è quella di un
passo mal tarato. Un Metropolis *proposal-limited* si sblocca alzando
l'accettazione, e qui l'abbiamo alzata al 20.8% senza che l'ESS si muovesse.

**Perché non contamina il deliverable, misurato anche sul B-BVAR:** ESS del
nowcast **1000/1000**, corr(nowcast, λ) = −0.047, corr(nowcast, μ) = −0.010 —
gli stessi ordini di grandezza dell'L-BVAR (−0.045 su μ), ora su due modelli.
Ogni iterazione estrae (B, Σ) esatte dalla NIW condizionatamente a γ e lo
smoother gira con randomness fresca, quindi la densità è di fatto iid. Il limite
da dichiarare in tesi: la banda esplora bene l'incertezza su parametri e stato,
**male quella sugli iperparametri** — attenuante misurata (sd(λ) = 0.016 e
correlazione ≈ 0), non assoluzione teorica.

*Chiuso al Gate 5 (leggendo il codice degli autori):* **il modo a posteriori
costava 4.8 ore, e la colpa era di tre scelte nostre, non del problema.**
Misurato sul profilo `l` vero (n=37, p=17, k=630): una valutazione della
log-posterior costa **220 ms**, e Nelder-Mead al budget di allora
(`maxfev = 2000·39`) ne bruciava 78.000 — **4.78 h senza convergere**, contro
0.44 s per iterazione del ciclo MCMC.

*Correzione misurata al pilota:* quel «0.44 s» contava **solo** le due valutazioni
di marginal likelihood del Metropolis, ed era sbagliato di 70×. Il ciclo vero
costa **~31 s per iterazione**, perché il costo dominante è il **simulation
smoother** — stato companion di dimensione `n·p = 629` su 479 mesi — non la ML.
Resta comunque vero che il modo andava sistemato: su un pilota da 150 pesa 31
min su 108.

`bvarGLPmf.m` risolve tutte e tre le cose, e nessuna era `maxiter`:

| | prima (nostro) | dopo (autori) | dove |
|---|---|---|---|
| ottimizzatore | Nelder-Mead, 39 dim | quasi-Newton (`csminwel`) | riga 248 |
| trasformazione | log, non vincolata | **logistica su un box** | righe 216–240 |
| partenza dei `psi` | `var(Y)` | **`SS`** = residuo AR(1) | righe 205–213 |
| Jacobiano nel modo | sì | **no** (log-post naturale) | `formin` riga 161 |

Il box rende irraggiungibili i limiti *per costruzione*, quindi la regione dove
la log-posterior vale `-inf` non esiste: è il motivo per cui gli autori non
hanno mai avuto il problema che ci aveva spinti verso un metodo senza gradienti.
E il punto di partenza contava più di tutto — `var(Y)/SS` ha **mediana 225×**
(max 41.395×, CPILFESL), 22 serie su 37 cadevano fuori dal box, e partivamo
**11.037 nat** sotto il punto degli autori.

Servivano però altre due correzioni, trovate solo perché il criterio di verifica
non era «è veloce» ma **«è un massimo»** — sondando ±h su tutte e 39 le
coordinate:

- **l'Hessiana non era mai definita positiva**, e il ripiego isotropo `W = 0.01·I`
  scattava *sempre*. La proposta perdeva così ogni informazione sulla forma del
  posterior — proprio ciò che l'Appendice B chiede di usare — e con `psi` che
  coprono nove ordini di grandezza è senza speranza. Rimedio, di nuovo dal loro
  codice (righe 359–362): `[V,E]=eig(HH); HH=V*abs(E)*V'`, cioè si prende il
  **valore assoluto** degli autovalori invece di rinunciare. Le direzioni quasi
  piatte sono tali, non un errore;
- **il passo delle differenze finite dev'essere ASSOLUTO**, non relativo. Il
  passo relativo scala con `|x|`, ma le coordinate del box sono **logit** e un
  logit passa per lo zero: a μ=2.53 corrisponde x=0.024, e un passo relativo di
  1e-5 diventa 2.4e-7 assoluto — sotto il rumore. Misurato: 32 direzioni
  miglioravano ancora. È esattamente perché le loro `x` sono O(1) che
  `numgrad.m` usa un passo assoluto.

Sul secondo punto ci si discosta dal loro **numero** con una ragione misurata:
loro usano `delta=1e-6` in avanti, noi differenze **centrate**, e a passo piccolo
le centrate amplificano il rumore. Provati sul profilo `q_b`: 1e-6 → 55 direzioni
residue; 1e-5 → 23; 1e-4 → 1; **1e-3 → 0, ed è anche 8× più veloce**.

**Risultato sul profilo `l`: 31.3 min contro 4.78 h (9×), e stavolta convergente
davvero** — 0 direzioni che migliorano, W definita positiva vera (cond 2e+02).
Il modo è **λ = 0.5128, μ = 3.5669**, entrambi interni, nessun `psi` al bordo del
box. La log-posterior al modo è 2.537 nat sopra il punto di partenza degli
autori.

*Controprova indipendente:* sul profilo `q_b` il modo cade a **λ = 0.5355,
μ = 2.2697**, contro le mediane a posteriori di 0.5182 e 2.2508 riportate nella
tabella qui sopra — ottenute per una strada completamente diversa (catena MCMC,
non ottimizzazione). Due metodi indipendenti sullo stesso punto.

*Su λ e la dimensione n:* λ passa da 0.5355 (n=30) a 0.5128 (n=37), quindi cala
nella direzione che BGR (2010) prevede, ma **di poco (−4%)** — molto meno di
quanto suggerisse una versione precedente di questa nota, che riportava 0.26 a
n=37 e parlava di un pattern monotono netto. Quel numero veniva da un modo **non
convergente**, fermo 1.400 nat sotto il vero: un buon promemoria che un
ottimizzatore che si ferma da solo non ha per questo trovato un massimo. Il
confronto con la banda 0.59–0.75 della Tab. B.1 resta comunque indicativo e non
un bersaglio: quella riporta mediane a posteriori di B, C e L, non modi.

*Corollario misurato, contro un sospetto:* λ **non** vuole il tetto sul profilo
vero. Profilato direttamente, il massimo è interno a λ≈0.2 e il tetto è ~9.700
nat peggiore, con *entrambi* i punti di partenza. E il termine di Jacobiano, pur
essendo una divergenza reale dal loro codice, è innocuo: vale 4,6 nat su tutto
il range di λ contro ~9.700 nat di curvatura della ML.

*Chiuso al Gate 5 (leggendo il codice degli autori):* **c'è un box sui `psi` che
nel paper non esiste.** `bvarGLPmf.m` righe 211–212 impongono
`psi ∈ [SS/100, SS·100]`, e `lbvar.m` righe 35–36 e 100 rifiutano seccamente le
proposte MCMC che ne escono. Né Cimadomo §2.1 né GLP §III lo menzionano: GLP
dichiara un iperprior inverse-Gamma «proper but quite disperse», e un box è un
prior in tutto e per tutto — tronca il supporto, e con esso il posterior.
**Adottato** per la regola di sempre (dove il paper tace, decide il loro
codice), ma **da dichiarare in tesi** accanto alle altre divergenze
paper↔codice: `P0 = 0` (`lbvar.py` punto 3) e il soc sulle `wn` (qui sotto).
*Attenuante misurata:* al modo del profilo `l` nessun `psi` tocca il bordo — il
box vincola il **percorso** dell'ottimizzatore, non il **risultato**.

*Chiuso al Gate 5 (leggendo il codice degli autori):* **il sum-of-coefficients
va azzerato sulle serie `wn`.** `logMLVAR_formin.m` fa `ydnoc(pos,pos) = 0`: per
le serie stazionarie il soc tira la somma dei propri lag verso **0**, non verso 1,
allineandosi al centraggio Minnesota. Prima del fix le quattro survey venivano
tirate da 0 a ~1.00, annullando la scelta `wn` del Gate 0. Il punto era riservato
al relatore perché *sembrava* una deviazione dal paper: non lo è — il paper tace,
il codice no. Impatto: μ da 1.20 a 2.25, λ da 0.49 a 0.52.

*Chiuso al Gate 3:* **Σ_εm non semidefinita positiva.** Su 100 estrazioni vere (n=30) è PSD nello
**0%** dei casi, mediana **9 autovalori negativi su 30**, `|λmin|/λmax` mediana 0.74. Non è un bug:
l'inversa di `X ↦ X + AXA'` non conserva il cono PSD quando ρ(A)>1 (qui ρ(A)≈2.25), e **né Cimadomo
né GMR (2016) coprono il caso** — GMR Prop. 1 assume «T_m is real and stable» su dati stazionari,
noi siamo in log-livelli con ρ(Φ)≈1.012. **Deciso: proiezione di Higham (1988) sulla PSD più vicina,
dichiarata apertamente come approssimazione e misurata a ogni estrazione** (`state_space.nearest_psd`,
`psd_summary`). Alternative scartate su evidenza: la (7) di GMR peggiora di ~35×; ridurre p non aiuta
(a p=2 il sistema è esatto e Σ_m è indefinita lo stesso); rendere stazionario il sistema non
risolverebbe (la PSD si perde anche con ρ(Φ)<1) e contraddirebbe §2 del paper. **Ridurre il profilo non è un rimedio**: rifatta la prova su profili annidati con Q-BVAR ristimato, Σ_m non è PSD e il filtro diverge a ogni n da 6 a 30, **n=18 (la dimensione del paper) compreso**. Trafila
completa e materiale per la tesi nell'header di `cube_root.py`.

*Chiuso al Gate 2:* il buco interno a **2025-10** (shutdown federale USA —
CPIAUCSL, UNRATE, CPILFESL, IR, IQ senza l'osservazione di ottobre). Deciso:
il campione di stima si chiude al **2025-09-30**, senza interpolare. Registrato
in `profiles.q_b.estimation_end`, letto da `data.estimation_end()`.
