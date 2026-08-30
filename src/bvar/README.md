# `src/bvar/` — BVAR per il nowcasting del PIL USA

Replica di **Cimadomo, Giannone, Lenza, Monti & Sokol (2022)**, *"Nowcasting with
large Bayesian vector autoregressions"*, Journal of Econometrics 231, 500–519.

Componente **indipendente** dal lavoro DFM del resto del repo. Con il resto
condivide solo l'infrastruttura dati stabile, **per import e mai per modifica**:
`src/forecast/release_calendar.py` e `data/processed/final/raw_levels_final.csv`.

Fonte di verità: il paper. Dove il paper rimanda a un altro lavoro per una
formula, l'autorità su quel pezzo è il lavoro citato — GLP (2015) per gli
iperparametri, BGR (2010) per le dummy, BGL (2015) per il filtering
condizionale, GRS (2008) per l'aggregazione, GMR (2016) per la radice cubica.

⚠️ **Regola permanente:** prima di implementare qualsiasi cosa qui dentro,
leggere il codice degli autori in `docs/BVARs/code/`. Ha già ribaltato tre
conclusioni. (La cartella è gitignorata: materiale altrui, va riscaricato dal
pacchetto di replica del *Journal of Econometrics*.)

---

## I quattro modelli, un solo core

| Modello | Paper § | Che cos'è | Profilo | `p` |
|---|---|---|---|---|
| **Q-BVAR** | 2.1 | VAR trimestrale baseline. È anche lo **stadio 1 del C-BVAR** (nota 20) | `q_b` (30 serie) | 5 |
| **C-BVAR** | 2.4 + App. A | Radice cubica: da Φ, Ω del Q a Φ_m = Φ^(1/3), poi Σ_εm. Filtro sulla finestra terminale | `q_b` (30) | 5 |
| **B-BVAR** | 2.3 | Blocking: ogni mensile → 3 serie trimestrali | `q_b` (30) | 5 |
| **L-BVAR** | 2.2 | VAR mensile con le trimestrali latenti | `l` (37 serie) | 17 |

Il **core** è uno solo: NIW coniugato + Minnesota (eq. 2–3) + sum-of-coefficients
via dummy observations, più il campionatore GLP (2015) per λ, ψ, μ.
L'architettura **non si ramifica**: i quattro modelli differiscono in *cosa*
passano al core, non in *quale* core chiamano.

> **L'invariante che tiene insieme tutto: il core sampler non vede mai un NaN.**

Il dato mancante è gestito **fuori** dal core, sempre: dal Kalman (B, C) o dal
simulation smoother (L). Il core riceve una `(T, n)` densa e restituisce
estrazioni di `(B, Σ)`.

---

## Le due decisioni di perimetro

Entrambe **approvate dal relatore, chiuse, non si riaprono**. Qui c'è l'esito;
il perché per esteso sta in `config/bvar_series.json` e in `data.py`.

**1. Due profili: 30 serie dal 1992 per Q/C/B, 37 per L.** Non è che «questi
modelli non usano il Kalman» — c'è in tutti e quattro. Cambia **dove agisce**:
in Q/C/B a valle (la stima è in forma chiusa e vuole un pannello pieno), nell'L
**dentro** il ciclo MCMC. Da cui i due tipi di dato mancante:

| tipo | dov'è | chi lo risolve |
|---|---|---|
| bordo frastagliato | in fondo | il Kalman, in **tutti** |
| partenza tardiva | in cima | **solo** l'L, con lo smoother |

Tenere tutte e 37 in Q/C/B significherebbe partire dal 2009-11: T≈66 con n=37,
un regime large-n-small-T più estremo di quello del paper (n=18, T≈130), dove
non si distingue un bug da un artefatto. Con 1992/n=30 si ha T=135.

**2. Tutte e sei le survey prendono `wn`, `ISM_PRICES` compreso.** La forma del
dato prevale sulla materia: un indice di diffusione è ancorato a una soglia
neutra e mean-reverting per costruzione. `ISM_PRICES` è la *derivata
qualitativa* dei prezzi, non un livello.

Il codice degli autori conferma la regola, ed è una conferma forte perché non
viene dal paper: il vettore `stationary` ha 3 elementi accesi su 28 — EPU,
Michigan, ISM PMI. Il **100%** dei loro soft indicator è `wn` e il 100% dei
prezzi è `rw`. Il loro pannello non contiene survey di prezzi, quindi su
`ISM_PRICES` non poteva arbitrare: la nostra è un'estensione della *loro* regola.

⚠️ `wn` non tocca solo la media a priori del primo lag: **spegne anche il
sum-of-coefficients** su quella serie (`ydnoc(pos,pos) = 0`).

---

## Dove sta scritta la teoria

Non in un documento separato: negli header dei moduli, accanto al codice. Questo
è l'indice.

| argomento | dove |
|---|---|
| Il VAR e le due incognite; prior NIW, coniugatezza, `dof = n+2` | `niw.py` |
| Marginal likelihood A.13 / A.14 stabile; paradosso di Bartlett | `niw.py` |
| Minnesota letta per righe; media vs covarianza; λ e i casi limite | `dummies.py` |
| Eq. (3) fattore per fattore; il refuso `ψ_j` vs `ψ_i` | `dummies.py` |
| Sum-of-coefficients, Π = 0, *inexact differencing*; μ; tensione wn↔soc | `dummies.py` |
| Contabilità dei gradi di libertà (la trappola) | `dummies.py` |
| La mappa delle quattro varianti | `__init__.py` |
| La discrepanza dentro l'Appendice A; A₀; il C fuori dalle ipotesi di GMR | `cube_root.py` |
| P₀ (`lyapunov_symm` sul sottospazio stabile); Higham | `state_space.py` |
| L'ordine livelli → MA 3 mesi → log (nota 17); il bordo × la MA | `qbvar.py` |
| I tre iperparametri; il salto gerarchico; iperprior GLP §III | `hyper.py` |
| Il mescolamento: ESS, la legge 1/d, perché non si «risolve» | `hyper.py` |
| Durbin-Koopman; la companion mensile | `simsmoother.py` |
| Il campionatore a precisione bandata | `precision_smoother.py` |
| `sample()` vs `step()` | `core.py` |

**Notazione** (da usare sempre, mai quella grezza dei paper): `Y` (T×n), `X`
(T×k), `B` (k×n), `Sigma` (n×n), `Psi = diag(psi)`, `Omega` (k×k diagonale),
`dof = n+2`, `d_centre` (1=rw, 0=wn).

**I nostri iperparametri**, profilo `q_b` (n=30, p=5, T=135, 1992Q1–2025Q3):

| | fine trimestre | medie mobili | **+ fix soc** | banda Tab. B.1 |
|---|---|---|---|---|
| λ | 0.5545 | 0.4875 | **0.5182** [0.490, 0.559] | 0.59–0.75 |
| μ | 1.0068 | 1.1983 | **2.2508** [1.848, 2.854] | 0.97–1.72 |

μ quasi raddoppia col fix, e il verso conta: μ sta al *denominatore*, quindi μ
grande = soc più **debole**. Tolto il conflitto col Minnesota sulle survey, i
dati chiedono un soc meno intenso. Che λ sia sotto la banda e μ sopra non è
un'anomalia: la Tabella B.1 riporta **B, C e L, non il Q**, e λ cala al crescere
di *n* (BGR 2010) — n=30 contro i loro 18.

---

## Il C-BVAR: un'analisi di riproducibilità

Il C-BVAR **funziona**, a due condizioni che il paper non dichiara e che sono
state ricostruite dal codice di replica. Trattazione completa in `cube_root.py`
e `state_space.py`; qui l'esito.

**1. La formula di accoppiamento.** L'Appendice A ha **tre letture
inequivalenti**, e la scelta decide se Σ_εm è una matrice di covarianza:

| formula | riproduce (A.15)? | Higham (mediana / p90) | autoval. neg. | ρ(A) |
|---|---|---|---|---|
| `literal` — (A.8) come stampata | no | — | — | — |
| `authors` — codice MATLAB, **default** | no | **0.00% / 1.84%** | 2 su 30 | 1.36 |
| `exact_a15` — la derivazione da (A.7) | **sì** | 75.77% / 99.16% | 9 su 30 | 2.28 |

**`p = 2` è il perno**, ed è un conteggio: `C M = D` ha `(p−1)n²` equazioni e
`n²` incognite, quindi è esattamente determinato a p=2 e sovradeterminato a p>2.
A p=2 il residuo è 0 per tutte e tre — risolvono esattamente **sistemi diversi**,
cioè due sono *errori*, non approssimazioni. A p=5 (quello che il paper stima) i
residui sono 0.40% per `exact_a15` contro 93% per le altre due.

> ⚠️ **Errore facile:** *non* è vero che le tre letture coincidano a p=2. Su
> radici cubiche vere differiscono già del 40–95%. Coincidono solo se si passa
> una companion invece di una radice cubica — trappola in cui si cade
> scrivendo il test.

**Il paradosso da scrivere in tesi:** la lettura matematicamente corretta è
quella numericamente inutilizzabile. (A.9) inverte `X ↦ X + A X A'`, che
conserva il cono PSD solo per ρ(A) < 1.

**2. La finestra del filtro.** Si filtra solo il bordo, come in `crbvar.m`. Non
è un espediente: il degrado è **monotono** nella lunghezza della ricorsione.

| finestra | 16 mesi | 24 | 48 | 96 | 200 | 403 |
|---|---|---|---|---|---|---|
| errore sui mesi osservati | **2.7e-04** | 1.1e-03 | 2.1e-03 | 4.6e-03 | 1.3e-02 | 1.6e-02 |

La causa: Φ^(1/3) di una companion quasi-difettiva è fortemente **non normale** —
raggio 1.005 ma entrate fino a 1.8e4 e cond(V) ≈ 1e6. **Il paper non spiega mai
perché il filtro giri su una finestra corta: questa tabella è la risposta.**

**Il contributo, in cinque punti.** (1) L'Appendice A è internamente incoerente;
(2) a p=2 lo si dimostra senza margini numerici; (3) a p=5 la scelta fra le tre
letture, mai dichiarata, decide se il modello è **stimabile**; (4) gli autori
incontrano la non-PSD e la gestiscono in silenzio (`D(D<0) = 1e-10`, un `qqFlag`
mai pubblicato, un riferimento a Higham *commentato*); (5) il filtro richiede una
finestra corta e il paper non dice perché.

> **Cinque conclusioni nostre ribaltate da misure successive** sono elencate in
> fondo all'header di `cube_root.py`. Vale la pena leggerle: sono un buon
> esempio di quanto sia facile attribuire a un metodo ciò che dipende da una
> scelta implementativa non dichiarata.

---

## Il blocco Covid dell'L-BVAR

Il blocco `2020-07-31 … 2020-10-23` è stato l'unico a fallire, in **tutte** le
passate, e **solo per l'L-BVAR**. Contiene 2020Q3, il rimbalzo del +34,9%.

Ne esce col **ripiego di settimana**. Le due correzioni numeriche che lo
precedono sono giuste per conto loro e allungano la catena di quattro volte, ma
**non chiudono il blocco** — e il perché è un risultato da scrivere in tesi.

### Durbin–Koopman cancella

DK ottiene l'estrazione come `α̃ = α⁺ + E[α | y − y⁺]`, con `α⁺` simulato **dalla
prior**. L'identità è esatta in aritmetica reale; in doppia precisione l'errore è
proporzionale alla scala dei **due addendi**, non della loro somma. Qui le scale
divergono: `lbvar.m` non ha alcun controllo di stabilità (verificato), quindi
`α⁺` cresce come ρ^T su T ≈ 500 mesi mentre il risultato resta alla scala del
dato, perché `R = 1e-12` inchioda le celle osservate.

Misurato (`tests/law_dk_error.py`, T = 500, residuo sulle celle osservate — che
devono tornare entro ~1e−6):

| ρ^T | residuo DK | residuo precisione |
|---|---|---|
| 3.7e+05 | 8.6e−07 | 0 |
| 4.1e+11 | 2.3e−04 | 0 |
| 3.7e+15 | **2.7e+00** | 0 |

All'ultima riga il pannello estratto non riproduce più il dato.

**Il rimedio è `precision_smoother.py`**: la stessa condizionale scritta sulla
precisione bandata `Ω = H'(I⊗Σ⁻¹)H`, dove condizionare sulle celle osservate è
una **partizione** e non un filtro. Le voci di Ω sono O(‖A‖²) — nessun ρ^T —
quindi non c'è niente da cancellare. **0,26 s contro ~32 s.** Il DK resta il ramo
normale: `_finite_smoother` prova quello per primo, con le stesse chiamate
all'RNG, e scende qui solo quando la guardia di cancellazione scatta.

### Un cammino in fuga avvelenava la catena

Il ripiego, nella prima versione, controllava solo `isfinite`. Un cammino a
**1e26** — finito — veniva accettato, entrava nella stima, e da lì ogni
estrazione di (B, Σ) era degenere: `max diag(Σ) = 1,5e37`, ρ = 70, precisione non
più definita positiva. La catena non si fermava dove si rompeva.

`_path_within_support` applica ora la stessa guardia **alle due strade**.

> ⚠️ **Da dichiarare in tesi:** rifiutare per magnitudine **non è un rifiuto di
> Metropolis**, è una troncatura del target. Si campiona dalla posterior
> ristretta alla regione rappresentabile.

### Perché non bastano, ed è il risultato

| | iterazioni raggiunte |
|---|---|
| solo DK | 11 |
| + campionatore a precisione | 42 |
| + guardia di magnitudine | **196** |

Quattro volte più lunga, e finisce lo stesso; gli iperparametri erano già
congelati alla 125. La causa è **specifica del profilo `l`**: PPIFIS ha 278 mesi
di pura latenza e PCEC96 244. Appena il VAR esce dal cerchio unitario la loro
storia ricostruita è enormemente diffusa — a **ρ = 1,136**, che è mite, il
cammino arriva a 1,9e9 contro un dato che sta a 91,5. Le estrazioni che la
posterior vuole sono esattamente quelle che la stima non digerisce: **a quel
vintage il Gibbs non ha un punto fisso praticabile.**

### Il ripiego di settimana, per tutti e quattro i modelli

`run_model`: stima piena → se fallisce e c'è cache, **riuso** → se fallisce e la
cache è vuota, `EstimationFailed`. Il terzo caso è quello del Covid, perché
`parallel_blocks` taglia sulle settimane piene e la prima settimana di ogni
blocco ha la cache vuota per costruzione. Lì `run_realtime` fa la **stima di
riscaldamento** all'ultima settimana piena precedente (`previous_full_week`) e
riprova in riuso.

**Il parallelismo resta intatto**: il blocco si ricalcola la stima da solo, non
aspetta nessun processo. Costa una stima in più *a quel blocco*.

Non è «saltare la settimana»: `fit_reuse` ricostruisce il pannello a `as_of` e
rifà il bordo, quindi il flusso dati nuovo entra tutto nel nowcast — si ereditano
solo i parametri. È quello che fanno gli autori nell'esercizio Covid, dove
`STEP3a_LBVAR_covid.m` r. 88 **commenta** il trigger di ri-stima e manda tutte le
settimane dell'anno a `lbvar_NE`. E il loro driver ha
`X_draws(X_draws==inf) = 1e16;`: l'infinito lo conoscevano.

Sul CSV nessun cambio di schema: `reestimated` esisteva già, e ora porta il
**fatto** (`estimated`) invece dell'intenzione (`full`).

**Provato sul blocco vero** (solo L, `n_draws=40`, fallimento al 2020-07-31
iniettato perché già misurato tre volte): riscaldamento al 2020-05-01 riuscito in
31,5 min, **13 settimane su 13**, 37,4 min totali.

**Il corollario che spiega il resto:** il ramo di riuso non falliva perché la sua
finestra è di ~45 mesi e non ~500. Stesso DK, altro esponente — ed è anche il
motivo per cui il ripiego funziona.

---

## Struttura del pacchetto

```
spec.py            MinnesotaSpec, Hyper, BVARSpec + lettura config
data.py            raw levels → known_at → log/livello; taglio al vintage
dummies.py         Yd, Xd: Minnesota + sum-of-coeff (BGR 2010 eq. 5)
niw.py             posterior NIW, draw di (B,Σ), marginal likelihood
hyper.py           iperprior + Metropolis su γ (GLP 2015 §III, App. B)
core.py            IL CORE SAMPLER — sample() e step()
simulate.py        DGP per i recovery test

qbvar.py           wrapper trimestrale: MA 3 mesi + core + output App. A
cube_root.py       Φ^(1/3), selezione radici, Σ_εm (A.9' + A.10), c_m
state_space.py     Higham + finestra/P₀ + ciclo filtro su kalman.py
cbvar.py           i tre stadi + nowcast
bbvar.py           impilamento + filtro
simsmoother.py     Durbin-Koopman
precision_smoother.py  la condizionale sulla precisione bandata (il ripiego)
lbvar.py           il ciclo MCMC a tre passi (§2.2)

evaluate.py        il ciclo settimanale: calendario, cache, CSV, checkpoint,
                   parallel_blocks, ripiego su stima fallita
figures.py         involucro su src/forecast/figures.py
metrics.py         involucro su compute_metrics + compare_nyfed
```

L'orchestratore è **`run_all.py`** nella radice del repo (`python run_all.py`,
nessuna opzione). Lancia in parallelo le 15 celle DFM più i benchmark e ogni
coppia modello × blocco del BVAR, poi i merge per blocco, poi le uscite. I
singoli pezzi restano lanciabili a mano da `scripts/`: `run_bvar.py`,
`run_dfm.py`, `run_outputs.py`.

Quattro cose di `run_all.py` che vale la pena sapere prima di leggere una
passata:

- **ogni job ha il suo log**, `output/_logs/run_all/<job>.log`, con
  `stderr` dentro lo stesso file: se un blocco cade, il traceback è lì.
  La cartella è gitignorata;
- **un job che fallisce non ferma gli altri** (`check=False`), e i falliti
  sono elencati in fondo con il percorso del loro log. Il merge gira con
  `--allow-missing` per la stessa ragione: un modello morto non deve portarsi
  via i tre che hanno finito;
- **un thread numerico per processo** (`OMP_NUM_THREADS=1` e compagnia in
  `CHILD_ENV`): è politica misurata, più thread BLAS rendono più lenti i job
  dominati dall'ottimizzazione;
- **i blocchi BVAR scrivono in shard separate** (`output/_bvar_shards/<blocco>/<modello>`)
  e vengono ricomposti dai merge. È il motivo per cui `merge_bvar_models.py`
  esiste e per cui `test_parallel_models` verifica che shard + merge diano
  byte per byte gli stessi artefatti della passata seriale.

I percorsi **non li decide questo pacchetto**: li decide
`src/output_layout.py`, e tutto sta sotto `forecast_weekly/` insieme al DFM.

```
output/forecast_weekly/
  csv/bvar/               il CSV lungo + i quantili .npz   (l'ingresso di tutto)
  csv/bvar/logscore/      i log score grezzi, uno per blocco
  bvar/<modello>/         Trajectories_<finestra>.png
  bvar/rmse/              RMSE_<finestra>.png + tabelle vs AR(2), media, NY Fed
  bvar/logscore/          LOGSCORE_<finestra>.png + riepiloghi
  comparison/             BVAR-vs-DFM, per fase, backcast compreso
```

`figures.py` e `metrics.py` **non disegnano e non calcolano**: chiamano
`src/forecast/{figures,compute_metrics,compare_nyfed}.py` con altri argomenti.
Il contratto CSV è `weekly_nowcast.COLUMNS` alla lettera — e vale solo se le
colonne sono anche **riempite**: `realizzato_bea` vuoto non dà errore, dà figure
senza pallini e RMSE su zero righe.

### L'unità di parallelismo

`evaluate.py` è **seriale di proposito**: nessun thread, nessun pool. Il
parallelismo sta fuori, un processo per blocco.

L'unità è il **blocco-trimestre** — una stima completa più le sue settimane di
riuso — indipendente dalle altre perché ogni stima riparte dal pannello a
`as_of`. I confini li dà `parallel_blocks` (flag `--print-blocks`) e cadono sulle
settimane di **stima piena**: la prima settimana di ogni blocco è comunque
forzata a piena, quindi tagliare lì è l'unico confine che non costa una stima in
più e non sposta una riga. Misurato sul 2007-2025:

| | blocchi | stime piene | settimane che cambiano valore |
|---|---|---|---|
| passata continua (riferimento) | 1 | 77 | — |
| **taglio sulle stime piene** | **77** | **77** | **0** |
| taglio annuale (com'era) | 19 | 95 | 18 |

I 77 blocchi hanno mediana 13 settimane (min 4, max 14). Il vincolo pratico non
sono i core ma la **RAM**: ogni processo carica il pannello e tiene la propria
cache (`--max-cache-mb`, default 1500).

### Cosa si importa da `kalman.py`

`src/kalman.py` **non si tocca**: è condiviso da `em/`, `mcmc/`, `forecast/`.
Riusabili perché pura algebra: `kalman_predict`, `kalman_update`,
`build_selection_matrix`. **Non** riusabili perché cablano la struttura DFM:
`build_A_tilde/Q_tilde/Lambda_tilde/R_tilde`, `kalman_filter`, `run_kalman`. Il
BVAR ha un'altra companion e non usa Mariano-Murasawa. **Riuso per import, non
riscrittura del filtro.**

---

## Convenzioni

- **Import**: sempre `from src.…`.
- **Configurazione**: nessuna scelta di modellazione cablata nel codice. Serie,
  profili, mappa log/livello e centraggio rw/wn stanno in
  `config/bvar_series.json`, con la motivazione accanto. `d_centre` è
  **costruito dalla config**, mai scritto a mano.
- **Dati**: pseudo-real-time. Si maschera un unico file corrente col calendario:
  si riproduce la tempistica dei rilasci, non le revisioni. Il paper ricostruisce
  vintage veri (§3.1). Semplificazione **dichiarata**.
- **I self-test non scrivono fra gli artefatti veri.**

## Test

```
python -m src.bvar.tests.test_data                 dati, profili, buchi
python -m src.bvar.tests.test_minnesota            eq. (2)-(3) analitica
python -m src.bvar.tests.test_dummies              Minnesota via dummy (l'oracolo)
python -m src.bvar.tests.test_soc                  sum-of-coefficients
python -m src.bvar.tests.test_niw_robust           NIW su matrici al limite
python -m src.bvar.tests.test_gate1                RECOVERY del core sampler
python -m src.bvar.tests.test_gate2                RECOVERY del wrapper trimestrale
python -m src.bvar.tests.test_gate3                mappa cube-root + stato-spazio
python -m src.bvar.tests.test_gate4                blocking + bordo frastagliato
python -m src.bvar.tests.test_gate5                simulation smoother (oracolo esatto)
python -m src.bvar.tests.test_gate6                calendario: look-ahead, dato BEA, le tre fasi
python -m src.bvar.tests.test_resume               ripresa + ripiego su stima fallita (~3 min)
python -m src.bvar.tests.test_parallel_models      sharding, ripiego, guardie numeriche
python -m src.bvar.tests.test_precision_smoother   il campionatore contro un oracolo denso
python -m src.bvar.tests.test_p0                   P₀ centralizzata sui 4 modelli
python -m src.bvar.tests.test_mixing               ESPERIMENTO: il mescolamento (~6 min)
python -m src.bvar.tests.law_dk_error              ESPERIMENTO: l'errore del DK contro ρ e T
python -m src.bvar.cbvar                           CHECK END-TO-END (~3 min)
```

Gli ultimi tre non sono test di gate: non c'è niente da «passare», producono
**numeri**, e i loro output sono materiale per la tesi.

`python -m src.bvar.cbvar` non sta in `tests/` perché richiede una stima vera:
sui trimestri **già osservati** il C-BVAR deve *riprodurre* il dato BEA, non
approssimarlo — lì lo smoother non ha nulla da stimare, quindi uno scarto
rivelerebbe un errore in un punto qualsiasi della catena. Misurato: entro
**0.03 punti percentuali**.

---

## Stato

Tutte le scelte di modellazione sono **chiuse e approvate dal relatore**. I
quattro modelli girano end-to-end.

| modello | stato | da dichiarare in tesi |
|---|---|---|
| **Q-BVAR** | chiuso | ρ(A) ≈ 1.012 è documentato, non è una questione |
| **C-BVAR** | chiuso come modello | due divergenze paper↔codice: la formula di accoppiamento e la non-PSD (Higham contro il loro `D(D<0)=1e-10`). Il residuo di (A.8) a p>2 **è il contributo**, non un punto aperto |
| **B-BVAR** | chiuso | ESS/iter di λ = 0.015 con accettazione al 20.8%: è la legge 1/d, non un bug |
| **L-BVAR** | chiuso | il box sui `psi` (non è nel paper), `P0 = 0`, e il blocco Covid: a quel vintage la stima piena non è possibile e si eredita |

**Aperto, e va verificato sulla passata del server:** che i 77 blocchi chiudano
77/77 per tutti e quattro i modelli. Il taglio del pannello al vintage
(`data.truncate_at_vintage`, applicato a L e B) **cambia il flusso casuale**:
i loro numeri si spostano, senza direzione sistematica.

### Tre cose misurate che vale la pena non riscoprire

- **Il mescolamento degli iperparametri è la legge 1/d** di un random-walk
  Metropolis — una proprietà nota del metodo che il paper prescrive, non un
  difetto nostro. `tests/test_mixing.py` la isola a modello fermo: la pendenza
  di log(ESS/it) su log(d) è **−1.10** contro il −1.00 teorico. Si descrive, non
  si «risolve». Trattazione in `hyper.py`, sezione *IL MESCOLAMENTO*.
- **Il profilo `l` non aveva campione prima del 2010.** `last_full_row` pretende
  una riga con *tutte* le serie osservate; con PPIFIS dal 2009-11 nessuna riga è
  piena prima. Rimedio: `data.drop_empty_series` scarta le serie con zero
  osservazioni a quella data. **`n` varia nel tempo** (35 → 36 → 37 fra 2007 e
  2010) e va dichiarato. Non è lo smoother che si rompe: è il *campione* che
  resta senza appiglio.
- **`np.linalg.solve` è 165× più lento di `cho_solve`** su questa build di
  OpenBLAS (percorso LU patologico; il gemm 630³ gira in 13 ms). Era il collo di
  bottiglia della ricerca del modo. Vedi `niw._spd_solve`.
