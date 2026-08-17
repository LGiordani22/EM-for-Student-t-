# Check diagnostico del calendario di forecast — sola lettura

> **Cosa contiene.** Due verifiche sul calendario di pubblicazione della pipeline
> di nowcasting: (1) il PIL realizzato rientra nel pannello e muove il Kalman,
> (2) il calendario è corretto e senza look-ahead per tutte le 37 serie.
> Diagnostica pura: nessuna modifica a codice o dati.
>
> **Prima esecuzione:** 2026-07-21 (dopo il fix `c759cd7` del mix raw/canonico
> nell'estrazione del nowcast — i valori qui sotto sono quindi post-fix).
> **Ri-verificata integralmente il 2026-08-04**: tutti i numeri del documento
> riproducono alla quinta cifra. Ricetta in §Riproduzione.
>
> Il dataset e i suoi delay sono descritti in `docs/DATASET_MAP.md`; qui si
> verifica solo la **meccanica del calendario**, non la scelta dei delay.

Meccanica di riferimento (`data_loader_final.as_of` → `forecast/release_calendar`):

> periodo *p* della serie *i* è noto a *D* ⇔ `fine_mese(p) + delay_i ≤ D`
> (per le 4 ISM: `data_rilascio_reale ≤ D`, con il delay come solo fallback).

---

## Check 1 — Il PIL realizzato rientra nel pannello e muove il Kalman?

**Sì, entrambe le cose, e con il delay corretto.** Non è un bug. Test attorno al
rilascio del PIL di **2008Q3** (fine trimestre 2008‑09‑30, delay 28g →
rilascio **2008‑10‑28**), target in volo **2008Q4**, cella `diag3 / gaussian`.

### (a) Il valore entra come osservazione nota per i vintage successivi

`known_at(panel, D)` — ragged edge di GDPC1 ai due venerdì a cavallo del rilascio:

| as_of | Ultimo PIL noto nel pannello | Target 2008Q4 nascosto? |
|---|---|---|
| 2008‑10‑24 (pre) | **2008Q2** (2008‑06‑30, z=+2.3747) | sì |
| 2008‑10‑31 (post) | **2008Q3** (2008‑09‑30, z=−2.1066) | sì |

Passato il rilascio (2008‑10‑28), la cella 2008Q3 — valore realizzato z=−2.1066 —
compare come osservazione **nota** nel vintage; il target 2008Q4 resta fuori dal
pannello per costruzione. La regola `as_of` è la stessa che decide sia cosa entra
sia cosa è bersaglio (`release_calendar.targets_in_flight`). ✔

### (b) Influenza lo stato del Kalman e il nowcast del trimestre in volo

Due prove, `theta` congelato (stimato una volta al venerdì pre‑rilascio, come la
pipeline con EM mensile).

**B1 — ablazione pura** (stesso `as_of` = 2008‑10‑31, stesso `theta`, pannello
identico *tranne* la singola cella PIL 2008Q3): isola l'effetto del solo PIL.

| Pannello | nowcast 2008Q4 (z) | livello (log‑ann.) | BEA % |
|---|---|---|---|
| **con** PIL 2008Q3 | −0.91565 | +1.0750 | **+1.0808** |
| **senza** PIL 2008Q3 | −0.91319 | +1.1865 | **+1.1935** |
| **Δ (con − senza)** | **−0.00246** | −0.111 | **−0.1127 pt** |

Rimuovere la sola cella del PIL realizzato sposta il nowcast di 2008Q4 di
**−0.11 punti BEA** → **il PIL entra nel filtro**. Segno coerente: un 2008Q3 molto
negativo (z=−2.1) abbassa il fattore e trascina giù 2008Q4.

**B2 — due venerdì consecutivi** (`theta` congelato, replica della pipeline):

| as_of | nowcast 2008Q4 (z) | BEA % | celle note |
|---|---|---|---|
| 2008‑10‑24 (pre) | −0.89960 | +1.2213 | 7910 |
| 2008‑10‑31 (post) | −0.91565 | +1.0808 | 7920 |
| **Δ (post − pre)** | **−0.01605** | **−0.1404 pt** | +10 |

Nella settimana escono **10 serie** (non solo il PIL — anche GDI, durable goods,
PCE…); B1 isola che il PIL da solo pesa −0.11 dei −0.14 punti complessivi.

> **Nota onesta sulla magnitudine.** L'effetto è reale ma modesto (~0.1 pt) perché
> il PIL realizzato di un trimestre *passato* (Q3) agisce sul nowcast di un
> trimestre *futuro* (Q4) solo **indirettamente**, ancorando il livello del
> fattore in Q3 che poi si propaga a Q4 via la dinamica VAR; e il loading del PIL
> sul fattore è piccolo (in `diag3`, R²≈0.19). L'entità cambia con spec/variante,
> ma il **meccanismo è confermato**: l'informazione non viene buttata.

### (c) Delay del PIL corretto

`publication_delay_days(GDPC1)` letto dai metadati = **28** (advance BEA);
`gdp_release_date("2008Q3")` = **2008‑10‑28** = 2008‑09‑30 + 28. ✔
(Identico per la GDI `A261RX1Q020SBEA`, 28g.)

**Verdetto Check 1: OK.** Il PIL realizzato (a) rientra nel pannello dopo il suo
rilascio, (b) modifica dimostrabilmente lo stato del Kalman e il nowcast dei
trimestri in volo, (c) col delay corretto di 28 giorni. Nessun bug.

---

## Check 2 — Il calendario è corretto per tutte le 37 serie?

**Fonte di TUTTI i delay:** `publication_delay_days` in
`config/series_final.json`, verificato vs **Tabella 2 del paper** (campo
`delays_verified`, 2026‑07‑17). Le **4 ISM** usano in più la **data di rilascio
reale** (`data/processed/final/ism_release_dates.csv`); il delay è solo fallback.

### Tabella delle 37 serie (ordinate per delay)

| Serie | Freq | Delay (gg) | Reference | Fonte delay | Plausibilità |
|---|:--:|--:|---|---|---|
| GACDISA066MSFRBNY (Empire) | M | −14 | current_month | config/Tab.2 | survey pre‑fine‑mese ✔ |
| GACDFSA066MSFRBPHI (Philly) | M | −11 | current_month | config/Tab.2 | survey pre‑fine‑mese ✔ |
| ISM_PMI | M | 3 | 1m_prior | **date reali** (+fallback) | ISM ~1‑3g ✔ |
| ISM_PRICES | M | 3 | 1m_prior | **date reali** (+fallback) | ISM ~1‑3g ✔ |
| ISM_EMP | M | 3 | 1m_prior | **date reali** (+fallback) | ISM ~1‑3g ✔ |
| ISM_NMI | M | 5 | 1m_prior | **date reali** (+fallback) | ISM non‑mfg ~3‑5g ✔ |
| USPRIV (ADP proxy) | M | 5 | 1m_prior | config/Tab.2 | ADP ~2‑5g ✔ ⚑proxy |
| PAYEMS | M | 7 | 1m_prior | config/Tab.2 | BLS occ. ~7g ✔ |
| UNRATE | M | 7 | 1m_prior | config/Tab.2 | BLS occ. ~7g ✔ |
| IR (import prices) | M | 13 | 1m_prior | config/Tab.2 | ~2 settimane ✔ ⚑1989+ |
| IQ (export prices) | M | 13 | 1m_prior | config/Tab.2 | ~2 settimane ✔ ⚑id ⚑1989+ |
| RSAFS (retail) | M | 14 | 1m_prior | config/Tab.2 | Census retail ~14g ✔ |
| PPIFIS | M | 14 | 1m_prior | config/Tab.2 | BLS PPI ~14g ✔ ⚑2009+ |
| HOUST | M | 16 | 1m_prior | config/Tab.2 | Census ~17g ✔ |
| PERMIT | M | 16 | 1m_prior | config/Tab.2 | Census ~17g ✔ |
| INDPRO | M | 17 | 1m_prior | config/Tab.2 | Fed IP ~17g ✔ |
| TCU | M | 17 | 1m_prior | config/Tab.2 | Fed IP ~17g ✔ |
| CPIAUCSL | M | 18 | 1m_prior | config/Tab.2 | BLS CPI ~18g ✔ |
| CPILFESL | M | 18 | 1m_prior | config/Tab.2 | BLS CPI ~18g ✔ |
| DGORDER | M | 26 | 1m_prior | config/Tab.2 | Advance Durable ~26g ✔ |
| HSN1F | M | 26 | 1m_prior | config/Tab.2 | New Res. Sales ~26g ✔ |
| AMDMVS | M | 26 | 1m_prior | config/Tab.2 | Advance Durable ~26g ✔ |
| AMDMTI | M | 26 | 1m_prior | config/Tab.2 | Advance Durable ~26g ✔ |
| A261RX1Q020SBEA (GDI) | Q | 28 | prior_quarter | config/Tab.2 | advance BEA ~28g ✔ |
| **GDPC1 (target)** | Q | 28 | prior_quarter | config/Tab.2 | advance BEA ~28g ✔ |
| PCEPILFE | M | 30 | 1m_prior | config/Tab.2 | Pers. Income ~30g ✔ |
| PCEC96 | M | 30 | 1m_prior | config/Tab.2 | Pers. Income ~30g ✔ ⚑2007+ |
| PCEPI | M | 30 | 1m_prior | config/Tab.2 | Pers. Income ~30g ✔ |
| DSPIC96 | M | 30 | 1m_prior | config/Tab.2 | Pers. Income ~30g ✔ |
| TTLCONS | M | 33 | 2m_prior | config/Tab.2 | Construction ~33g ✔ |
| ULCNFB | Q | 34 | prior_quarter | config/Tab.2 | Prod.&Costs ~34g ✔ |
| AMTMUO | M | 35 | 2m_prior | config/Tab.2 | M3 ~35g ✔ |
| BOPTEXP | M | 35 | 2m_prior | config/Tab.2 | BEA trade ~35g ✔ ⚑mapping |
| BOPTIMP | M | 35 | 2m_prior | config/Tab.2 | BEA trade ~35g ✔ ⚑mapping |
| WHLSLRIMSA | M | 37 | 2m_prior | config/Tab.2 | Wholesale ~37g ✔ |
| JTSJOL | M | 42 | 2m_prior | config/Tab.2 | JOLTS ~42g ✔ |
| BUSINV | M | 44 | 2m_prior | config/Tab.2 (reference corretto) | ⚑44g — vedi sotto |

Legenda ⚑ (documentati in `config/series_final.json` e in `DATASET_MAP.md` §6,
**non** bug): `proxy` USPRIV proxy BLS di ADP; `id` IQ id da riverificare (all
commodities — l'id di IR è invece stato verificato); `1989+` IR/IQ sono
trimestrali prima del 1989, contribuiscono dal 1989; `2009+`/`2007+` partenza
tardiva (buco NaN gestito dal Kalman); `mapping` BOPTEXP/IMP mappate a BEA
goods+services (non Census advance 28/1m); `BUSINV` 44g: il reference
`1m_prior` di Tab. 2 non regge coi 44 giorni ed è
stato **corretto in `2m_prior` il 2026-08-11** (il delay, unico campo usato nei
calcoli, resta quello del paper: nessun numero di questo documento cambia).

### (a) Delay implausibili

L'euristica (soglie: `<−20` troppo anticipato, `>60` troppo lungo, coerenza
delay↔reference) **non segnala nessuna serie**. Confronto coi valori tipici
attesi (ISM ~1‑3g, occupazione BLS ~7g, PIL ~28g, Census ~30‑45g): tutti
coerenti. Nessun delay troppo corto (rischio look‑ahead) né sprecato.
L'unico borderline era **BUSINV (44g / 1m_prior)** — incoerenza della Tab. 2,
non del codice: reference corretto in `2m_prior` il 2026-08-11, delay invariato.

### (b) Le 4 ISM usano le date reali (non il delay generico) ✔

Test discriminante: mesi in cui la data **reale** è *più tarda* di `ref+delay`
(direzione critica per il look‑ahead). Con `as_of` fra `ref+delay` e la data
reale, il valore deve restare **nascosto**:

| Serie | Ref | ref+delay | Data REALE | as_of testata | Visibile? |
|---|---|---|---|---|---|
| ISM_PRICES | 2008‑08 | 2008‑09‑03 | **2008‑10‑01** (+31g) | 2008‑09‑17 | **No** ✔ |
| ISM_PMI | 2025‑12 | 2026‑01‑03 | 2026‑01‑05 (+5g) | 2026‑01‑04 | No ✔ |
| ISM_EMP | 2025‑12 | 2026‑01‑03 | 2026‑01‑05 (+5g) | 2026‑01‑04 | No ✔ |
| ISM_NMI | 2025‑12 | 2026‑01‑05 | 2026‑01‑07 (+7g) | 2026‑01‑06 | No ✔ |

`ISM_PRICES 2008‑08` è il caso forte: con il delay generico (3g) sarebbe stata
visibile dal 3 settembre; la data reale la tiene nascosta fino al 1° ottobre. Il
codice rispetta la data reale → **evita un look‑ahead di ~28 giorni**. Delay
implicito reale ISM: mediana 1‑3g, config 3‑5g (fallback conservativo).

### (c) Test di look‑ahead su tutte le 37 serie ✔

Oracolo indipendente della data di rilascio effettiva (ISM = data reale;
altrimenti ref+delay). Per ogni cella **non‑NaN** di `known_at(panel, D)` si
verifica `rilascio ≤ D`:

| as_of | Celle note | Look‑ahead |
|---|--:|---|
| 2008‑10‑31 | 7 920 | **0 violazioni** |
| 2016‑09‑16 | 11 217 | **0 violazioni** |
| 2020‑05‑15 | 12 755 | **0 violazioni** |
| 2012‑03‑30 | 9 340 | **0 violazioni** |
| 2023‑07‑14 | 14 083 | **0 violazioni** |

**0 violazioni su 5 date** (55 315 celle). Nessuna serie visibile prima del suo
rilascio.

### (d) Serie mancanti / delay mancanti ✔

37 colonne del pannello ↔ 37 righe di metadati, corrispondenza esatta: nessuna
colonna senza metadati, **nessun delay mancante o di default non verificato**,
nessuna serie orfana.

**Verdetto Check 2: OK.** Calendario completo e coerente per tutte le 37 serie;
ISM sulle date reali; zero look‑ahead. Nessun bug. Voci da tenere d'occhio (già
documentate, non azioni): **BUSINV 44g** (reference corretto in 2m_prior il 2026-08-11), **IQ**
(id da riverificare), **BOPTEXP/IMP** (scelta di mapping BEA vs Census advance).

---

## Su quali giorni gira il forecast?

**Ogni venerdì.** `release_calendar.weekly_grid` usa `_WEEK_ANCHOR = "W-FRI"`: le
date `as_of` sono tutti i venerdì in `[start, end]` (il venerdì chiude la
settimana lavorativa → tutto ciò che è uscito nella settimana è dentro).

- **Non** è un giorno fisso del mese: nei venerdì 2008‑2009 il giorno‑del‑mese
  assume tutti i valori 1..31; ogni mese ha **4 o 5 venerdì** su date diverse.
- Verificato ott‑nov 2008: 3, 10, 17, 24, 31 ott · 7, 14, 21, 28 nov — tutti
  venerdì.
- La **ri‑stima EM** (non il nowcast) ha cadenza separata: default `monthly` =
  EM sul **primo venerdì del mese** (`weekly_nowcast.py`, confronto
  `month != last_em_month`), poi solo filtro a `theta` congelato nei venerdì
  successivi; `--em-frequency weekly` ri‑stima ogni venerdì.

---

## Verdetti in sintesi

| | Esito | Dettaglio |
|---|---|---|
| **Check 1** (PIL rientra + muove il Kalman) | **OK** | (a) rientra dopo il rilascio; (b) ablazione −0.11 pt BEA su 2008Q4 → entra nel filtro; (c) delay 28g corretto. Magnitudine modesta ma meccanismo confermato. |
| **Check 2** (calendario 37 serie) | **OK** | 37/37 coperte, ISM su date reali, 0 look‑ahead su 55 315 celle, nessun delay mancante. Da verificare a mano (documentati, non bug): BUSINV 44g (reference corretto in 2m il 2026-08-11), IQ id, BOPTEXP/IMP mapping. |
| **Giorni del forecast** | **Venerdì** | griglia W‑FRI, variabile nel mese (4‑5/mese); EM ri‑stimato il 1° venerdì del mese (default). |

---

## Cosa questo documento NON verifica

- **Le revisioni dei dati.** Il calendario riproduce la *tempistica*, non i
  vintage: pseudo‑real‑time publication‑lag only (`DATASET_MAP.md` §4.1).
- **La scelta dei delay.** Che i valori di Tabella 2 siano quelli giusti è
  verificato altrove (campo `delays_verified` del config); qui si verifica che
  siano applicati correttamente.
- **L'inizio delle linee della figura 8a** (~1 trimestre di anticipo, stile
  Cascaldi‑Garcia): è `targets_in_flight` + il taglio grafico di `figures.py`,
  fuori dallo scopo di questi due check.

---

## Riproduzione

I check di puro calendario (Check 2 a/b/c/d + la griglia settimanale) non
richiedono stima: girano in secondi su `release_calendar`.

```python
from src.forecast import release_calendar as rc
panel, meta, exact = rc.load_panel(), rc.load_metadata(), rc.load_exact_releases()

# (c) oracolo look-ahead: per ogni cella nota, rilascio <= D
kn = rc.known_at(panel, "2008-10-31", meta, exact)     # 7 920 celle note

# (a) ragged edge del PIL a cavallo del rilascio
rc.gdp_release_date("2008Q3", metadata=meta)            # 2008-10-28
```

Il Check 1(b) richiede una stima EM (`diag3/gaussian`, ~4 s sul vintage
2008‑10‑24) e poi due chiamate a `theta` congelato:

```python
from src.forecast.nowcast_engine import nowcast
pre  = nowcast("2008-10-24", "2008Q4", "diag3", "gaussian")          # stima
post = nowcast("2008-10-31", "2008Q4", "diag3", "gaussian", theta=pre["theta"])
abl  = panel.copy(); abl.loc["2008-09-30", "GDPC1"] = float("nan")
post_abl = nowcast("2008-10-31", "2008Q4", "diag3", "gaussian",
                   theta=pre["theta"], panel=abl)
```
