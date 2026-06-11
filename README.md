# Two-Stage Quantile Nowcasting — First Stage

**Tesi di laurea magistrale, Sapienza Università di Roma**
Relatori: Massimiliano Tancioni, Milos Ciganovic — Candidato: Lorenzo Giordani

Questa repository contiene il **First Stage** di una metodologia di nowcasting in
due stadi per il PIL USA: un **Dynamic Factor Model mixed-frequency con errori
Student-_t_**, stimato via **algoritmo EM** (Kalman filter/smoother + scale-mixture
representation), validato con un esperimento **Monte Carlo** calibrato sui dati reali,
e applicato al **nowcasting real-time** del PIL statunitense su più regimi macroeconomici
(crisi 2008, COVID-19, inflazione 2022, ecc.). Il modello esiste in due configurazioni
intercambiabili: **`small`** (20 serie) e **`big`** (50 serie). Il Second Stage
(regressione quantilica stile Growth-at-Risk) è _future work_ e non è incluso qui
(vedi §6).

Tutta la derivazione teorica — Kalman filter/smoother, aggiornamenti EM, augmentazione
Mariano–Murasawa, pesi della scale-mixture, identificazione rotazionale — è in
**`EM_for_student_t.tex`** (sorgente della tesi), riferimento autoritativo per ogni
scelta implementativa.

---

## 1. Mappa delle cartelle

| Cartella | Contenuto |
|---|---|
| `src/` | Libreria core del DFM Student-_t_: simulazione, inizializzazione, Kalman, E-step / M-step, orchestratore EM, e i tre esperimenti Monte Carlo (`run_experiment_a/b/c.py`). |
| `src/forecast/` | Pipeline di **nowcasting real-time**: import dei vintage, costruzione del panel, motore DFM, benchmark univariati, orchestratore rolling, metriche e figure. |
| `scripts/` | Due orchestratori PowerShell: `run_experiments.ps1` (Monte Carlo A/B/C) e `run_forecasts.ps1` (rolling nowcast su tutti i periodi), con parallelismo controllato e _resume_ automatico. |
| `notebooks/` | Tre notebook di analisi/validazione: `01` esplorazione dati, `02` controllo inizializzazione θ⁽⁰⁾, `03` validazione della struttura a 3 blocchi. |
| `config/` | I due _single source of truth_ delle configurazioni: `series_small.json` (20 serie) e `series_big.json` (50 serie). Definiscono serie, blocco (real/financial/other), frequenza, codice di trasformazione FRED-MD, ordine delle colonne e dimensione dei fattori. |
| `data/raw/` | I **323 vintage real-time di FRED-MD** (1999-08 → 2026-03), in tre cartelle per intervallo, più `fredmd_current.csv`. Vedi §5. |
| `data/processed/` | Dataset costruiti (`dataset_small.csv`, `dataset_big.csv` + metadata) e, nelle sottocartelle `small/` e `big/`, gli artefatti di stima: `theta_initial.npz` (θ⁽⁰⁾) e `fit_dfm_result.npz` (θ\* = stima EM in-sample). |
| `output/` | Tutti i risultati prodotti: diagnostiche in-sample (`small/figures/`, `big/figures/`), summary Monte Carlo (`monte_carlo/`), test di recovery (`recovery/`), nowcast real-time (`forecast_realtime/`). Vedi §4. |
| `_archive_second_stage/` | Prototipo **archiviato** del Second Stage (regressione quantilica del PIL sui fattori). Materiale di sviluppo futuro, **non parte** del First Stage. Vedi §6. |
| _root_ | `EM_for_student_t.tex` (derivazioni teoriche), `requirements.txt`, `.env` (chiave API FRED), `README.md`. |

---

## 2. Ordine di esecuzione (il cuore)

Il flusso è una catena: ogni passo consuma l'output del precedente. Il flag
**`--small` / `--big`** (o `--config <nome>`) seleziona la configurazione e instrada
automaticamente tutti i path verso la cartella giusta — `small` riproduce il dataset
originale a 20 serie bit-per-bit, `big` lo estende a 50. Tutti i comandi vanno lanciati
dalla **root del progetto**, con il `.venv` attivo.

### Passo 1 — Costruzione del dataset
```bash
python src/data_loader.py --config small      # oppure --config big
```
Scarica FRED-MD + GDPC1 + NFCI (serve `FRED_API_KEY` in `.env`), applica i codici di
trasformazione FRED-MD per la stazionarietà, posiziona il PIL solo a fine trimestre, e
scrive `data/processed/dataset_<cfg>.csv`. La lista delle serie è letta interamente da
`config/series_<cfg>.json`.

### Passo 2 — Inizializzazione EM (θ⁽⁰⁾)
```bash
python src/em_initialization.py --small
```
Standardizza il panel, riempie i buchi mixed-frequency (Mariano–Murasawa) e ragged-edge,
esegue la PCA per blocco e calcola il vettore di parametri iniziale.
**Produce:** `data/processed/<cfg>/theta_initial.npz` (+ metadata) e le figure diagnostiche
`pca_factors.png`, `sim_factors.png`, `mm_fill_verification.png` in `output/<cfg>/figures/`.

### Passo 3 — Stima EM in-sample (θ\*)
```bash
python src/em_main.py --small --max-iter 250
```
Esegue il loop EM esterno (E-step Student-_t_ + M-step + Kalman), normalizza segni e scala
(sign convention + Convention 1 a varianza totale unitaria) per rendere i fattori
economicamente interpretabili.
**Produce:** `data/processed/<cfg>/fit_dfm_result.npz` — **θ\***, la stima usata come DGP del
Monte Carlo — e le figure `em_loglik_convergence.png`, `kalman_filtered_factors.png`,
`kalman_filtered_vs_smoothed_real.png`.

### Passo 4 — Recovery / identificabilità
```bash
python src/monte_carlo_recovery.py --small
```
Test di auto-recovery: simula un panel sintetico da θ\*, ri-stima l'EM **da zero** (PCA
indipendente, senza sbirciare θ\*) e confronta — su scalari invarianti (ν, autovalori di
_A_), loadings (con allineamento di Procuste) e _path latente_ — quanto bene l'EM ricostruisce
la verità. Due repliche, `T=497` e `T=2000`.
**Produce:** `output/recovery/<cfg>/` (`recovery_T497_<cfg>.txt`, `recovery_T2000_<cfg>.txt`,
`recovery_summary_<cfg>.txt`, e gli `.npz`).

### Passo 5 — Esperimenti Monte Carlo (A / B / C)
```powershell
.\scripts\run_experiments.ps1               # small, S=20 (test rapido, A→B→C sequenziale)
.\scripts\run_experiments.ps1 -big          # big, S=20
.\scripts\run_experiments.ps1 -both -full   # small+big, S=1000 (run da tesi; A∥C in parallelo)
.\scripts\run_experiments.ps1 -Force        # ignora la cache (fingerprint resume) e ricalcola
```
Lo script orchestra i tre esperimenti (lanciabili anche singolarmente con
`python src/run_experiment_a.py --small [--full]`), rispettando la dipendenza B→A e con
_resume_ basato sul fingerprint SHA-1 di θ\*:

- **Exp A** — DGP Student-_t_ (coda pesante, ν≈4): lo stimatore Student-_t_ deve **battere** il
  Gaussiano (che, privo dell'attenuazione dei pesi, gonfia _Q_ ed _R_).
- **Exp B** — DGP Gaussiano (ν→∞): _nesting check_, i due stimatori devono **coincidere** (gap≈0).
- **Exp C** — contaminazione da outlier idiosincratici (π>0): robustezza dello Student-_t_.

**Produce:** `output/monte_carlo/<cfg>/exp{A,B,C}/*.json` e i summary `SUMMARY_<cfg>.txt`.

### Passo 6 — Nowcast real-time rolling
```powershell
.\scripts\run_forecasts.ps1                              # small, tutti i periodi, MaxParallel 3
.\scripts\run_forecasts.ps1 -Config big -MaxParallel 2   # big (più pesante)
.\scripts\run_forecasts.ps1 -Config both -Figures        # entrambi, con figure a fine run
.\scripts\run_forecasts.ps1 -Force                       # ricalcolo da zero (rimuove i CSV)
```
Orchestra `python -m src.forecast.rolling_nowcast --start YYYY-MM --end YYYY-MM --small` su
**7 periodi** (dotcom 2001, crisi 2008–09, debito EU 2011–12, calma 2015, dazi 2018–19, COVID
2020, inflazione 2022–23). Per ogni mese (`as_of` = giorno 15) il motore identifica
automaticamente i trimestri "in volo", ri-stima il DFM **sul vintage real-time corrispondente**
(nessun look-ahead: il PIL del trimestre target non è mai nel panel) e produce il nowcast con
DFM Student-_t_, DFM Gaussiano e i benchmark univariati (ARMA, random walk). Parallelismo
controllato e _resume_ automatico su tripla `as_of × target × method`.
**Produce:** `output/forecast_realtime/csv/<cfg>/rolling_nowcast_<start>_<end>.csv` (formato
"long": una riga per `as_of × target × metodo`).

### Passo 7 — Metriche di accuratezza
```bash
python -m src.forecast.compute_metrics --small      # oppure --big
```
Legge i CSV del Passo 6 (non ri-stima nulla) e calcola RMSE, MAE, bias, correlazione, RMSE
relativo ai benchmark e la metrica z (Volatility Paradox), aggregati per `metodo × horizon`,
per metodo e per periodo.
**Produce:** `output/forecast_realtime/csv/<cfg>/metrics_summary.txt` (leggibile) e
`metrics_by_method_horizon.csv` (machine-readable).

### Passo 8 — Figure dei nowcast
```bash
python -m src.forecast.figures --small                          # griglia 4 pannelli, CSV più recente
python -m src.forecast.figures --small --style compare --target 2008Q4
```
**Produce:** le figure delle traiettorie in `output/forecast_realtime/figures/<cfg>/`.

> **Helper diagnostico (opzionale, fuori pipeline):** `python -m src.forecast.extract_weps
> --as-of 2009-01-15 --target 2008Q4 --window 2008-01:2009-12` estrae i pesi mensili `w_eps_t`
> della scale-mixture per documentare il down-weighting dei mesi di crisi.

---

## 3. I notebook

| Notebook | Cosa analizza |
|---|---|
| `01_data_exploration.ipynb` | Carica `dataset_<cfg>.csv`, visualizza le serie per blocco, riporta il pattern di dati mancanti e le statistiche descrittive. Config-aware (`CONFIG = "small"`/`"big"`). |
| `02_initialization_check.ipynb` | Visualizza il vettore iniziale θ⁽⁰⁾ (`theta_initial.npz`) come sanity check prima dell'EM (Sezione 4 della tesi). |
| `03_model_specification.ipynb` | Valida la struttura a **3 blocchi** (1 fattore per blocco: real/financial/nominal): verifica che i blocchi imposti per design economico siano statisticamente distinti e ben identificati. |

---

## 4. Dove guardare i risultati (senza eseguire nulla)

Un lettore che voglia solo vedere gli output già prodotti:

- **Diagnostiche in-sample del DFM** — `output/small/figures/` e `output/big/figures/`:
  convergenza della log-likelihood EM, fattori filtrati di Kalman, confronto filtrato vs smoothed.
- **Validazione Monte Carlo** — `output/monte_carlo/small/SUMMARY_small.txt` e
  `output/monte_carlo/big/SUMMARY_big.txt` (esiti aggregati degli esperimenti A/B/C).
- **Test di recovery / identificabilità** — `output/recovery/<cfg>/recovery_summary_<cfg>.txt`.
- **Tabelle di accuratezza dei nowcast** — `output/forecast_realtime/csv/<cfg>/metrics_summary.txt`
  e `metrics_by_method_horizon.csv`.
- **Nowcast grezzi** — i CSV `rolling_nowcast_<start>_<end>.csv` nella stessa cartella.
- **Figure dei nowcast** — `output/forecast_realtime/figures/<cfg>/`.

---

## 5. Dati e riproducibilità

La repository è **autocontenuta** sul lato dati storici:

- `data/raw/` contiene i **vintage real-time di FRED-MD** (McCracken & Ng, Federal Reserve Bank
  of St. Louis) — un file per mese di pubblicazione, da `1999-08` a `2026-03` — organizzati in
  tre cartelle (`1999-08…2014-12`, `2015-01…2024-12`, `2025-01…2026-03`). Ogni file contiene
  l'intera storia nota _a quella data di rilascio_, con il ragged-edge reale del momento.
- `data/processed/` contiene i dataset stazionarizzati costruiti dal Passo 1.

**Caveat ibrido (da dichiarare in tesi):** le serie FRED-MD usano i veri vintage real-time
(valori e disponibilità corretti a ciascuna data); GDPC1 e NFCI usano invece i valori
_correnti_ (revisionati) ma con la disponibilità temporale ricostruita per rispettare il
calendario di pubblicazione reale (ALFRED espone i vintage NFCI solo dal 2011). C'è quindi
look-ahead sui _valori_ di GDPC1/NFCI ma non sulla loro _disponibilità_. Migrare GDPC1 ai veri
vintage ALFRED è un upgrade futuro.

Fonte: **FRED-MD**, McCracken & Ng (2016), *"FRED-MD: A Monthly Database for Macroeconomic
Research"*, Journal of Business & Economic Statistics.

---

## 6. Cosa NON è incluso (future work)

Questo First Stage è una componente di un disegno più ampio. Sono **archiviati fuori dalla
pipeline** e non necessari per riprodurre i risultati qui descritti:

- **Second Stage** — l'esplorazione preliminare della **regressione quantilica del PIL sui
  fattori del DFM** (stile Growth-at-Risk, Adrian–Boyarchenko–Giannone 2019: predizione dei
  quantili condizionali e fit di una skew-_t_). Il prototipo end-to-end è in
  `_archive_second_stage/` ma è **sviluppo futuro**, non parte di questa consegna.
- Eventuali file pre-refactor della configurazione (prima del passaggio config-driven JSON)
  sono anch'essi archiviati e non più usati.

---

## 7. Ambiente

Python (≥ 3.12; sviluppato su 3.14) con virtual environment dedicato:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dipendenze principali: **numpy**, **scipy**, **pandas**, **matplotlib**, **statsmodels**,
**fredapi**, **joblib**, **seaborn**, **tqdm**, **python-dotenv**, **requests**, **jupyter**.
La costruzione del dataset (Passo 1) richiede una chiave API FRED gratuita
(<https://fred.stlouisfed.org/docs/api/api_key.html>) salvata come `FRED_API_KEY` nel file
`.env`; tutti gli altri passi girano sui dati già presenti nella repository.
