# DATASET_MAP — il pannello `final`, come è costruito

> **Cos'è questo documento.** La mappa del dataset **già costruito** che sta in
> `data/processed/final/`: quali serie ci sono, da dove vengono, come sono
> trasformate, quando diventano note. Descrive lo stato dei fatti, non un piano.
>
> **Fonte.** Bok, Caratelli, Giannone, Sbordone, Tambalotti (2018), *Macroeconomic
> Nowcasting with Big Data*, in
> `docs/Paper tesi/Bok, Caratelli, Giannone, Sbordone, Tambalotti.pdf`
> (cartella locale, non versionata) —
> **Tabella 3** (serie, loadings, Units) pp. 631-632; **Tabella 2** (release e
> ritardi di pubblicazione) pp. 623-624.
>
> **Single source of truth:** `config/series_final.json` (schema `fed-v1`).
> Il costruttore è `src/data_loader_final.py`; il documento riflette il config,
> non lo sostituisce. Se i due divergono, ha ragione il config.
>
> Ultima revisione: 2026-08-04 · branch `refactor/mcmc-spec2`

---

## 0. In due righe

**37 serie** (34 mensili + 3 trimestrali), **499 mesi** dal 1985-01 al 2026-07,
**16.9% di celle mancanti** — quasi tutte per partenza tardiva o per il layout
trimestrale, non per buchi interni. Fonte: **33 da FRED** + **4 ISM** da
`data/raw/ISM_series_pulite.xlsx`. Il target `GDPC1` è l'**ultima colonna**.

| Artefatto | Cosa contiene |
|---|---|
| `data/processed/final/dataset_final.csv` | il pannello trasformato, indice month-end, 499×37 |
| `data/processed/final/raw_levels_final.csv` | gli stessi dati **in livello**, per invertire la catena di scala |
| `data/processed/final/metadata_final.csv` | una riga per serie: blocco, trasformazione, delay, inizio effettivo, % missing |
| `data/processed/final/ism_release_dates.csv` | date di rilascio **reali** delle 4 ISM (mensili, dal 1979) |
| `config/calendar_example.md` | tabella generata: le 37 serie per delay + esempi di bordo frastagliato |

Ricostruzione: `python src/data_loader_final.py` (serve `FRED_API_KEY` in `.env`).

---

## 1. Le 37 serie

**Blocchi.** G = global (carica su tutte), S = soft/surveys, R = real, L = labor.
Verificati uno-a-uno dalle "X" di Tabella 3 (p. 632): **nessuna serie carica su
due blocchi locali** — sempre G più al massimo uno fra S/R/L. Distribuzione
effettiva: 19 `GR`, 7 `G` (solo globale), 6 `GS`, 5 `GL`.

**Trasformazioni** (vocabolario in §2): `MoM_log` 22 · `level` 6 · `diff_level` 4
· `QoQ_log_ar` 3 · `diff_ppt` 2.

**Inizio effettivo** è la prima osservazione non-NaN *dopo* la trasformazione
(quindi già al netto della differenza persa). 17 serie mensili coprono l'intero
campione dal 1985-01, le 3 trimestrali dal 1985-03.

> ⚠ La numerazione `#` di questa tabella è l'**ordine delle colonne del pannello**,
> non quello di Tabella 3 del paper. Per la corrispondenza col paper vale la
> colonna `paper_name` di `metadata_final.csv`.

| # | Serie (paper) | Bloc. | Freq | ID | Trasf. | Delay | Rif. | Inizio | % NaN |
|---|---|---|---|---|---|--:|---|---|--:|
| 1 | All employees: total nonfarm | GL | M | `PAYEMS` | diff_level | 7 | 1m | 1985-01 | 0.2 |
| 2 | CPI-U: all items | G | M | `CPIAUCSL` | MoM_log | 18 | 1m | 1985-01 | 0.6 |
| 3 | Mfrs new orders: durable goods | GR | M | `DGORDER` | MoM_log | 26 | 1m | 1992-03 | 17.6 |
| 4 | Retail sales and food services | GR | M | `RSAFS` | MoM_log | 14 | 1m | 1992-02 | 17.2 |
| 5 | New single family houses sold | GR | M | `HSN1F` | MoM_log | 26 | 1m | 1985-01 | 0.4 |
| 6 | Housing starts | GR | M | `HOUST` | MoM_log | 16 | 1m | 1985-01 | 0.2 |
| 7 | Civilian unemployment rate | GL | M | `UNRATE` | diff_ppt | 7 | 1m | 1985-01 | 0.6 |
| 8 | Industrial production index | GR | M | `INDPRO` | MoM_log | 17 | 1m | 1985-01 | 0.2 |
| 9 | PPI: final demand | G | M | `PPIFIS` | MoM_log | 14 | 1m | **2009-12** | 60.1 |
| 10 | ADP nonfarm private payroll *(proxy)* | GL | M | `USPRIV` | diff_level | 5 | 1m | 1985-01 | 0.2 |
| 11 | Empire State Mfg: gen. bus. cond. | GS | M | `GACDISA066MSFRBNY` | level | −14 | corr. | 2001-07 | 39.7 |
| 12 | Merchant wholesalers: inventories | GR | M | `WHLSLRIMSA` | MoM_log | 37 | 2m | 1992-02 | 17.4 |
| 13 | Value of construction put in place | GR | M | `TTLCONS` | MoM_log | 33 | 2m | 1993-02 | 19.8 |
| 14 | Philly Fed Mfg: current activity | GS | M | `GACDFSA066MSFRBPHI` | level | −11 | corr. | 1985-01 | 0.0 |
| 15 | Import price index | G | M | `IR` | MoM_log | 13 | 1m | 1989-01 | 10.2 |
| 16 | Building permits | GR | M | `PERMIT` | diff_level | 16 | 1m | 1985-01 | 0.2 |
| 17 | Capacity utilization | GR | M | `TCU` | diff_ppt | 17 | 1m | 1985-01 | 0.2 |
| 18 | Core PCE: chain price index | G | M | `PCEPILFE` | MoM_log | 30 | 1m | 1985-01 | 0.4 |
| 19 | CPI-U less food & energy | G | M | `CPILFESL` | MoM_log | 18 | 1m | 1985-01 | 0.6 |
| 20 | Inventories: total business | GR | M | `BUSINV` | MoM_log | 44 | 2m | 1992-02 | 17.4 |
| 21 | JOLTS: job openings: total | GL | M | `JTSJOL` | diff_level | 42 | 2m | 2001-01 | 38.9 |
| 22 | Real personal consumption expend. | GR | M | `PCEC96` | MoM_log | 30 | 1m | **2007-02** | 53.5 |
| 23 | PCE: chain price index | G | M | `PCEPI` | MoM_log | 30 | 1m | 1985-01 | 0.4 |
| 24 | Export price index | G | M | `IQ` | MoM_log | 13 | 1m | 1989-01 | 10.2 |
| 25 | Mfrs shipments: durable goods | GR | M | `AMDMVS` | MoM_log | 26 | 1m | 1992-02 | 17.4 |
| 26 | Mfrs unfilled orders: all mfg | GR | M | `AMTMUO` | MoM_log | 35 | 2m | 1992-02 | 17.4 |
| 27 | Mfrs inventories: durable goods | GR | M | `AMDMTI` | MoM_log | 26 | 1m | 1992-02 | 17.4 |
| 28 | Real disposable personal income | GR | M | `DSPIC96` | MoM_log | 30 | 1m | 1985-01 | 0.4 |
| 29 | Exports: goods and services | GR | M | `BOPTEXP` | MoM_log | 35 | 2m | 1992-02 | 17.4 |
| 30 | Imports: goods and services | GR | M | `BOPTIMP` | MoM_log | 35 | 2m | 1992-02 | 17.4 |
| 31 | ISM mfg.: PMI composite | GS | M | `ISM_PMI` 🔒 | level | 3 | 1m | 1985-01 | 0.2 |
| 32 | ISM mfg.: prices index | GS | M | `ISM_PRICES` 🔒 | level | 3 | 1m | 1985-01 | 0.2 |
| 33 | ISM mfg.: employment index | GS | M | `ISM_EMP` 🔒 | level | 3 | 1m | 1985-01 | 0.2 |
| 34 | ISM non-mfg.: NMI composite | GS | M | `ISM_NMI` 🔒 | level | 5 | 1m | 1997-07 | 30.3 |
| 35 | Nonfarm business: unit labor cost | GL | **Q** | `ULCNFB` | QoQ_log_ar | 34 | prec. | 1985-03 | 66.9 |
| 36 | Real gross domestic income | GR | **Q** | `A261RX1Q020SBEA` | QoQ_log_ar | 28 | prec. | 1985-03 | 66.9 |
| 37 | **Real GDP** *(target, ultima colonna)* | GR | **Q** | `GDPC1` | QoQ_log_ar | 28 | prec. | 1985-03 | 66.9 |

🔒 = non su FRED, letta da `data/raw/ISM_series_pulite.xlsx` (foglio `panel`).
Rif.: `1m`/`2m` = il dato descrive il mese precedente / di due mesi prima;
`corr.` = mese corrente; `prec.` = trimestre precedente.
Il **66.9% di NaN delle trimestrali è il layout**, non un buco: stanno solo sui
mesi 3/6/9/12 (2 mesi su 3 sono NaN per costruzione).

### 1.1 Le cinque sostituzioni, e perché

| Serie del paper | Cosa c'è nel dataset | Motivo |
|---|---|---|
| ISM PMI, prices, employment, NMI (4) | `ISM_PMI`, `ISM_PRICES`, `ISM_EMP`, `ISM_NMI` da xlsx locale | l'ISM ritirò le proprie serie da FRED nel 2016; le vecchie `NAPM*` si fermano lì e non esistono in real-time forward. Sono il grosso del blocco soft: prenderle da file evita di svuotare S |
| ADP nonfarm private payroll | `USPRIV` (BLS, All Employees: Total Private) | il dato ADP è proprietario. `USPRIV` è l'analogo BLS con la stessa unità (`diff_level`) e storia lunga — **proxy, non identico**, dichiarato come tale |

### 1.2 Cosa NON entra, di proposito

- **Variabili finanziarie.** Il paper le esclude esplicitamente (p. 631:
  *"financial variables are not included in the model"*). Nessun S&P 500, spread
  o NFCI in questo pannello.
- **Disaggregati.** Solo gli *headline* di ogni release (p. 630): niente
  scomposizioni settoriali o per classe di età.
- **Rebasing degli indici.** Non serve: le trasformazioni (rapporti,
  log-differenze, differenze prime) sono scale-invarianti e la standardizzazione
  toglie la scala residua. Le uniche serie in livello sono gli indici di
  diffusione ISM/Empire/Philly, già su scala 0-100 comune.

---

## 2. Trasformazioni e inverse

Cinque trasformazioni, tutte prese dalla colonna *Units* di Tabella 3, tutte
invertibili. Implementate in `data_loader_final.apply_final_transform`
(`src/data_loader_final.py:140`).

| `transform` | Units (paper) | Formula | `inverse_fn` | Ricostruzione |
|---|---|---|---|---|
| `MoM_log` | MoM % change | `100·log(X_t/X_{t−1})` | `inv_MoM_log` | `X_t = X_{t−1}·exp(g/100)` |
| `QoQ_log_ar` | QoQ % change, annual rate | `400·log(X_t/X_{t−1})` | `inv_QoQ_log_ar` | `X_t = X_{t−1}·exp(g/400)` |
| `diff_ppt` | Ppt. change | `X_t − X_{t−1}` (punti %) | `inv_diff` | `X_t = X_{t−1} + d` |
| `diff_level` | Level change (thousands) | `X_t − X_{t−1}` (migliaia) | `inv_diff` | `X_t = X_{t−1} + d` |
| `level` | Index | `X_t` (identità) | `identity` | — |

L'output del modello è **standardizzato**: la catena inversa completa è
`z → ·std + mean → inverti la trasformazione`, con `mean`/`std` calcolati **sul
vintage** (`src/forecast/scale.py`), mai sul campione pieno — usare la std piena
sarebbe look-ahead.

### 2.1 Log e non variazione percentuale: la scelta che conta

La versione iniziale del dataset usava l'annualizzazione **composta** della BEA,
`100·((X_t/X_{t−1})⁴ − 1)`, classificando lo scarto dal log come «di secondo
ordine». **Misurato sul campione 1985-oggi, è falso**: `(1+g)⁴` è **convessa**,
comprime la coda sinistra e allunga la destra.

| | composta | log (`400·Δlog`) |
|---|---:|---:|
| skewness `GDPC1` | **+0.173** | **−2.232** |
| 2020Q2 | −27.98 | −32.82 |
| 2020Q3 | +34.86 | +29.90 |
| sd | 4.175 | 4.177 |

La composta **ribalta il segno dell'asimmetria** del PIL. Per una tesi che
modella l'asimmetria della densità del PIL, cancella *nel dato* l'oggetto che il
modello deve stimare. Stesso effetto sulle altre due trimestrali: GDI da −1.44 a
−3.34, ULC da +0.02 a −0.27.

Il secondo motivo è l'aggregatore. I pesi di Mariano-Murasawa {1/3, 2/3, 1, 2/3,
1/3} derivano dall'approssimazione geometrica della media trimestrale: l'identità
è **esatta solo in log-differenze**. Errore misurato su INDPRO (unica serie con
sia il livello mensile sia l'aggregato trimestrale): **rmse 0.0067 in log contro
0.0211 in variazione percentuale**, ×3.1.

**Il reporting non ci perde nulla:** `g_BEA = 100·(exp(x/100) − 1)` è
**monotona**, quindi i quantili commutano e la GaR si riporta in unità BEA
ufficiali in modo *esatto*, non approssimato.

> ⚠ **Nota di manutenzione.** Il campo `notes.transform_vocab` di
> `config/series_final.json` descrive ancora le formule **percentuali**
> (`MoM_pct`, `QoQ_pct_ar` composta): è testo residuo della versione precedente.
> Il codice e i dati usano le log-differenze qui sopra. Vale il codice.

### 2.2 L'annualizzazione non tocca Kalman né mixed-frequency

Tre ragioni, tutte verificate:

1. **La standardizzazione cancella la scala.** `standardize(4g) = (4g − 4·mean)/(4·std)
   = standardize(g)`: l'input standardizzato è **bit-identico** a quello non
   annualizzato. Il modello vede la stessa serie, i pesi MM non cambiano.
2. **Il loading assorbe la costante.** Anche senza standardizzare, l'MM è lineare
   (`Y^Q = Λ·Σ w_l f_{t−l}`): moltiplicare `Y^Q` per 4 fa stimare all'M-step
   `Λ_new = 4·Λ`. I pesi restano identici.
3. **La de-standardizzazione restituisce la scala d'ingresso**, quindi
   `inv_QoQ_log_ar` legge direttamente la crescita annualizzata.

---

## 3. Il layout che il Kalman si aspetta

Il costruttore del dataset produce questo formato; il modello non è stato
adattato al dataset.

1. **Il PIL sta al mese di fine trimestre.** Le tre trimestrali stanno **solo**
   sui mesi 3/6/9/12, all'indice month-end: il PIL di gennaio-marzo sta su
   **marzo**. `data_loader_final.py:130` (`_quarter_start_to_qend_monthend`)
   sposta l'indice da inizio a fine trimestre; `data_loader_final.py:345` forza a
   `NaN` le trimestrali su ogni mese non ∈ {3,6,9,12}.
2. **Mesi mancanti = `NaN`**, non zero né placeholder. Il Kalman li salta via
   matrice di selezione `W_t` (`src/kalman.py:399`,
   `build_selection_matrix`): le righe `NaN` non entrano nell'update. Nessun
   trattamento speciale a monte.
3. **`GDPC1` è l'ultima colonna** della matrice.
4. **L'aggregazione MM è già nel modello.** `kalman.build_Lambda_tilde`
   (`src/kalman.py:222`) legge `freq_list`: per ogni riga `"quarterly"` spalma il
   loading sui 5 lag con i pesi {1/3, 2/3, 1, 2/3, 1/3}; le `"monthly"` caricano
   solo sul blocco contemporaneo. Il ciclo è su `freq_list`, **non c'è cablatura a
   un unico PIL**: le tre trimestrali (GDP, GDI, ULC) sono aggregate tutte senza
   modifiche al codice.

---

## 4. Ragged edge: `as_of`

La regola di disponibilità è **una sola** e vive in `data_loader_final.as_of`
(`src/data_loader_final.py:402`):

> il periodo *p* della serie *i* è noto alla data *D* ⇔ `fine_periodo(p) + delay_i ≤ D`

- Il `publication_delay_days` di Tabella 2 è **già misurato dalla fine del periodo
  di riferimento** (nota *b*): **non va ri-sommato** il `reference_period`,
  altrimenti si conta due volte. Il campo `reference_period` è documentazione.
- I **delay negativi** (Empire −14, Philly −11: survey pubblicate prima della fine
  del mese) funzionano con la stessa formula, senza casi speciali.
- Le trimestrali cadono naturalmente: fine trimestre + 28. Il PIL del trimestre in
  corso **non è mai nel pannello** — è esattamente il caso "nowcast".
- Il delay è **per-release**, non per-serie: serie della stessa release
  condividono il ritardo (CPI all + core CPI → 18; payroll + unemployment → 7).

**Sopra questa regola, il forecast aggiunge un livello.**
`src/forecast/release_calendar.py:100` (`known_at`) importa `as_of` e per le
**4 ISM** sostituisce il delay con la **data di rilascio reale**
(`ism_release_dates.csv`, copertura mensile dal 1979 — dal 1997-07 per `ISM_NMI`);
dove il file non copre, si ricade sul delay. Questa è la funzione che la pipeline
di nowcasting usa davvero: vedi `docs/FORECAST_CALENDAR_CHECK.md`.

### 4.1 I tre caveat da dichiarare in tesi

1. **Pseudo-real-time, publication-lag only.** Si riproduce la *tempistica* di
   pubblicazione, non le **revisioni**: un nowcast del 2008 vede i dati del 2008
   nella loro versione di oggi. Il vero real-time del paper (Fig. 4) ricostruisce
   anche i vintage. Semplificazione dichiarata, non difetto nascosto.
2. **Delay congelati sul calendario 2017.** La colonna *Publication timing* del
   paper (es. "first business day") è la regola; il delay è quella regola
   convertita in giorni fissi usando il 2017. Approssima la regola negli altri
   anni — tranne per le 4 ISM, che hanno la data vera.
3. **Un id ancora da riverificare:** `IQ` (export price index, all commodities,
   2000=100). `IR` è stato verificato ed è corretto — con l'avvertenza che è
   **trimestrale prima del 1989** e contribuisce quindi solo dal 1989.

---

## 5. Copertura temporale: il quadro onesto

Il pannello comincia nel **1985-01** e il Kalman gestisce nativamente lo
sbilanciamento (`W_t` salta i `NaN`), ma le serie non partono tutte insieme.

| Fascia | Serie | Nota |
|---|---|---|
| **1985 (17 M + 3 Q)** | PAYEMS, CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, DSPIC96, INDPRO, TCU, UNRATE, HOUST, HSN1F, PERMIT, USPRIV, Philly, ISM_PMI/PRICES/EMP; le 3 trimestrali da 1985-03 | copertura piena |
| **1989** | `IR`, `IQ` | mensili solo dal 1989 (prima trimestrali) |
| **1992-93 (9)** | DGORDER, RSAFS, AMDMVS, AMTMUO, AMDMTI, BUSINV, WHLSLRIMSA, BOPTEXP, BOPTIMP (1992), TTLCONS (1993) | base NAICS corrente del Census: niente pre-1992 su FRED |
| **1997-2001** | ISM_NMI (1997-07), JTSJOL (2001-01), Empire (2001-07) | accettate |
| **2007** | `PCEC96` | la serie **mensile** FRED parte 2007-02 → 53.5% NaN |
| **2009** | `PPIFIS` | concetto BLS nato nel 2009 → 60.1% NaN |

**Le due decisioni sui buchi lunghi.** `PPIFIS` e `PCEC96` restano `NaN` prima
della loro partenza: **nessun raccordo** con `PPIFGS`/`PPIACO`, **nessun
back-splicing**. Incollare serie con definizione diversa introdurrebbe una rottura
di livello proprio nella trasformazione in differenze; il `NaN` è onesto e il
filtro lo assorbe.

Per riferimento: il paper stima su **15 anni mobili** (p. 631) e fa backtesting
**dal 2000** — non ha mai avuto bisogno del 1985. Il tratto 1985-1992 del nostro
pannello poggia sulle 17 serie lunghe.

---

## 6. Le anomalie riprodotte fedelmente (non sono bug)

| Voce | Cosa | Perché è così |
|---|---|---|
| `BUSINV` | delay **44** del paper, reference **corretto** da `1m_prior` a `2m_prior` il **2026-08-11** | la riga di Tabella 2 ("Manufacturing and Trade Inventories") non regge internamente: prima settimana piena del mese su dati del mese prima farebbe ~7 giorni, non 44. La realtà Census è metà mese su dati di **due** mesi prima (31-gen + 44 = 16-mar). Teniamo il delay del paper (che è il campo giusto) e allineiamo il reference. **Non cambia una cella**: `as_of`/`release_date` usano solo il delay |
| `BOPTEXP` / `BOPTIMP` | mappate a *US International Trade in Goods & Services* (35 gg, 2m_prior) | sono BEA goods+services, BOP basis. Il rilascio più tempestivo (*Advance Economic Indicators*, 28 gg, 1m) è Census **goods-only**: serie diversa, non usata |
| Ordini/inventari manifatturieri | #3/#25/#27 → *Advance Durable Goods* (26 gg); #26 → *M3 full* (35 gg); #20 → *Manufacturing and Trade Inventories* (44 gg); #12 → *Wholesale Trade* (37 gg) | assegnazione advance-vs-full scelta a favore del report più tempestivo dove il paper lascia ambiguità |

Delay e `reference_period` sono stati verificati riga per riga contro Tabella 2
(pp. 623-624) e le trasformazioni contro Tabella 3 (p. 632): **zero discrepanze**
(campo `notes.delays_verified` del config, 2026-07-17).

---

## 7. Blocchi del dataset ≠ numero di fattori

Il dataset trasporta i `block_loadings` G/S/R/L di Tabella 3 e **basta**. Quanti
fattori stimare (r = 4 con G+S+R+L, oppure r = 3) e con quale maschera di
caricamento è una scelta di **modello**, non di dato: vive in
`config/factor_specs.json` e nelle tre spec `fed_overlap` / `diag4` / `diag3`
(vedi `src/forecast/nowcast_engine.py`). Cambiare spec non richiede ricostruire
il dataset.
