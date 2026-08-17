# CLAUDE.md — orientamento per l'assistente

Tesi di laurea magistrale: **DFM mixed-frequency con errori Student-t** per il
nowcasting del PIL USA. Repo in italiano; codice e docstring in italiano/inglese
misto, commenti in italiano.

## La teoria: dove sta

**`docs/tesi/EM_for_student_t.tex`** è il **riferimento autoritativo** per ogni
scelta implementativa: Kalman filter/smoother, aggiornamenti EM, augmentazione
Mariano-Murasawa, scale-mixture, identificazione rotazionale, e tutti i
conditional del Gibbs sampler.

⚠️ **`docs/tesi/` non è più nel repository** (gitignorato dal 2026-08-17,
insieme al resto di `docs/`): il `.tex`, il PDF e la tesi in scrittura vivono
**solo in locale**. La regola qui sotto vale comunque, ma su un clone fresco
quei file **non ci sono**.

- Il PDF compilato e i file ausiliari di build LaTeX stanno **nella stessa
  cartella** `docs/tesi/`.
- Le docstring del codice citano il file per **nome soltanto**
  (`EM_for_student_t.tex`, `eq:mm-aggregation`, `subsec:asis`, …): il path è
  sempre `docs/tesi/`. Le etichette `eq:*` / `sec:*` / `subsec:*` si cercano
  direttamente nel `.tex`.
- Il `.tex` è grande (~1.3 MB): cercarci dentro con Grep sull'etichetta, non
  leggerlo per intero.

## Dati e config

- **Single source of truth del dataset:** `config/series_final.json` (schema
  `fed-v1`, 37 serie). Costruttore: `src/data_loader_final.py`.
  Artefatti in `data/processed/final/`. Mappa leggibile: `docs/DATASET_MAP.md`.
- **Struttura dei fattori** (spec `fed_overlap` / `diag4` / `diag3`):
  `config/factor_specs.json`. È una scelta di *modello*, separata dal dato.
- Il binario legacy `small`/`big` (`series_small.json`, `series_big.json`,
  `src/data_loader.py`) è stato **cancellato il 2026-08-03**.
- Il pacchetto MCMC (Gibbs, SV, leverage, ASIS, `validate/`) è stato
  **archiviato in `_archive_mcmc/` il 2026-08-06**: non era più importabile
  dopo la cancellazione del binario legacy e non fa parte della consegna.
  **Non ha dipendenze entranti**: l'unica che c'era — `ess` usata da
  `src/bvar/tests/test_mixing.py` — è stata copiata in
  `src/bvar/diagnostics.py`.

## Gli archivi: in locale, fuori dal repo

Tre cartelle `_archive_*` raccolgono materiale che non fa parte della consegna.
Sono tutte **gitignorate dal 2026-08-17**: restano sul disco, non nel repo.

| Cartella | Cosa contiene |
|---|---|
| `_archive_mcmc/` | il pacchetto MCMC (vedi sopra); la teoria resta nel `.tex` |
| `_archive_mc_diagnostics/` | le diagnosi Monte Carlo: `diagnose_mc_recovery*.py`, `_diagnostic_em/`, gli output `diagnostic_mc_v2/` e `recovery_diagnostics/`, e i due `MONTECARLO_*.md` |
| `_archive_notebooks/` | i tre notebook `0X_*.ipynb` che parlano ancora del binario `small`/`big`: **non possono più girare**, il loader è stato cancellato |

`notebooks/` resta al suo posto con quello che è ancora vivo
(`tabelle_data.py` usa `series_final`, più `tesi.ipynb`, `bok_fig2.py`,
`bai_ng_test.py`).

## Documenti in `docs/`

**Dal 2026-08-17 `docs/` è quasi tutta fuori dal repo**: la regola è `docs/*`
ignorata, con tre eccezioni. Solo le prime tre righe della tabella sono
versionate; tutto il resto vive in locale.

| File | Versionato | Cosa contiene |
|---|---|---|
| `DATASET_MAP.md` | sì | il pannello `final`: serie, trasformazioni, delay, copertura |
| `FORECAST_CALENDAR_CHECK.md` | sì | verifiche sul calendario di pubblicazione (no look-ahead) |
| `Variabili NYFED.docx` | sì | le variabili della replica NY Fed |
| `tesi/EM_for_student_t.tex` | **no** | la teoria (autoritativo) |
| `tesi/` | **no** | la tesi in scrittura: `Tesi.tex`, `sections/`, `tables/`, `figures/`, PDF |
| `Paper tesi/` | **no** | i paper di riferimento (96 MB di PDF altrui) |
| `BVARs/` | **no** | codice degli autori dei BVAR + materiale di riferimento |

I `MONTECARLO_*.md` sono passati in `_archive_mc_diagnostics/docs/`.

**Regola permanente sui BVAR:** prima di implementare qualsiasi cosa in
`src/bvar/`, leggere nel dettaglio il codice degli autori in `docs/BVARs/code/`.

⚠️ **`docs/BVARs/` non è nel repository** (gitignorato dal 2026-08-06): è
materiale altrui e la repo è pubblica. La regola qui sopra vale comunque, ma su
un clone fresco quei file **non ci sono**: vanno riscaricati dal *Journal of
Econometrics* (pacchetto di replica di Cimadomo, Giannone, Lenza, Monti &
Sokol 2022) e rimessi in `docs/BVARs/`.

Dal 2026-08-17 le due cartelle hanno confini netti, senza più duplicati:

- **`docs/BVARs/`** = solo il **codice** (52 MB): `code/` — il bersaglio della
  regola qui sopra — più `CGLMSreplicationWeb.zip`, il pacchetto originale
  della rivista (101 file, contiene anche i dati che `code/` non ha).
- **`docs/Paper tesi/`** = tutti i **paper**, BVAR inclusi in `Paper tesi/BVARs/`.

## Note operative

- Windows + PowerShell; virtualenv in `.venv`.
- I self-test non devono scrivere fra gli artefatti veri.
- `README.md` descrive ancora in parte la pipeline pre-refactor (`small`/`big`):
  in caso di conflitto valgono il codice e i documenti di `docs/`.
