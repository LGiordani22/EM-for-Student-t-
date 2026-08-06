# CLAUDE.md — orientamento per l'assistente

Tesi di laurea magistrale: **DFM mixed-frequency con errori Student-t** per il
nowcasting del PIL USA. Repo in italiano; codice e docstring in italiano/inglese
misto, commenti in italiano.

## La teoria: dove sta

**`docs/tesi/EM_for_student_t.tex`** è il **riferimento autoritativo** per ogni
scelta implementativa: Kalman filter/smoother, aggiornamenti EM, augmentazione
Mariano-Murasawa, scale-mixture, identificazione rotazionale, e tutti i
conditional del Gibbs sampler.

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
  dopo la cancellazione del binario legacy e non fa parte della consegna. Resta
  versionato perché la teoria corrispondente è nel `.tex`. **Non ha dipendenze
  entranti**: l'unica che c'era — `ess` usata da `src/bvar/tests/test_mixing.py`
  — è stata copiata in `src/bvar/diagnostics.py`.

## Documenti in `docs/`

| File | Cosa contiene |
|---|---|
| `tesi/EM_for_student_t.tex` | la teoria (autoritativo) |
| `DATASET_MAP.md` | il pannello `final`: serie, trasformazioni, delay, copertura |
| `FORECAST_CALENDAR_CHECK.md` | verifiche sul calendario di pubblicazione (no look-ahead) |
| `MONTECARLO_PROBLEMI.md` | problemi aperti degli esperimenti Monte Carlo |
| `BVARs/` | codice degli autori dei BVAR + materiale di riferimento — **solo in locale, non versionato** |

**Regola permanente sui BVAR:** prima di implementare qualsiasi cosa in
`src/bvar/`, leggere nel dettaglio il codice degli autori in `docs/BVARs/code/`.

⚠️ **`docs/BVARs/` non è nel repository** (gitignorato dal 2026-08-06): sono
59 MB di materiale altrui e la repo è pubblica. La regola qui sopra vale
comunque, ma su un clone fresco quei file **non ci sono**: vanno riscaricati
dal *Journal of Econometrics* (pacchetto di replica di Cimadomo, Giannone,
Lenza, Monti & Sokol 2022) e rimessi in `docs/BVARs/`.

## Note operative

- Windows + PowerShell; virtualenv in `.venv`.
- I self-test non devono scrivere fra gli artefatti veri.
- `README.md` descrive ancora in parte la pipeline pre-refactor (`small`/`big`):
  in caso di conflitto valgono il codice e i documenti di `docs/`.
