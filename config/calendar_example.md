# Calendario & ragged-edge — come le serie entrano nel tempo

> Generato da `data_loader_final.build_calendar_example`. Illustrativo: spiega la meccanica dei delay senza scaricare nulla.

## 1. Da dove viene il delay

Il **delay** (Tab. 2 del paper, nota b) è in **giorni di calendario dalla FINE del periodo di riferimento**, congelato sul 2017. NON misura quanto è puntuale il rilascio nel mese, ma quanto è **vecchio il dato** che contiene.

Esempio chiave — stessa regola di pubblicazione, delay opposti:

| Release | Publication timing | Reference | Delay |
|---|---|---|---|
| ISM Manufacturing | first business day of the month | 1 mese prima | **3** |
| Construction Spending | first business day of the month | 2 mesi prima | **33** |

Construction Spending esce presto nel mese, ma il report del 1° giorno lavorativo di marzo descrive **gennaio** (2 mesi prima): da fine gennaio alla pubblicazione ci sono ~33 giorni. ISM descrive il mese appena chiuso → ~3 giorni.

Regola `as_of`: periodo *p* disponibile a *D* ⇔ `fine_mese(p) + delay ≤ D`. Il delay include già lo scarto dal periodo di riferimento → il campo `reference_period` è documentazione, non si ri-somma.

## 2. Le 37 serie ordinate per delay (dal più tempestivo)

| Serie | Blocco | Freq | Reference | Delay (gg) | Release |
|---|---|---|---|---:|---|
| GACDISA066MSFRBNY | GS | M | current_month | -14 | Empire State Mfg Survey |
| GACDFSA066MSFRBPHI | GS | M | current_month | -11 | Manufacturing Business Outlook (Philly) |
| ISM_PMI | GS | M | 1m_prior | 3 | ISM Manufacturing Report |
| ISM_PRICES | GS | M | 1m_prior | 3 | ISM Manufacturing Report |
| ISM_EMP | GS | M | 1m_prior | 3 | ISM Manufacturing Report |
| USPRIV | GL | M | 1m_prior | 5 | ADP National Employment |
| ISM_NMI | GS | M | 1m_prior | 5 | ISM Non-Manufacturing Report |
| PAYEMS | GL | M | 1m_prior | 7 | Employment Situation Report |
| UNRATE | GL | M | 1m_prior | 7 | Employment Situation Report |
| IR | G | M | 1m_prior | 13 | US Import & Export Price Indexes |
| IQ | G | M | 1m_prior | 13 | US Import & Export Price Indexes |
| RSAFS | GR | M | 1m_prior | 14 | Retail Trade |
| PPIFIS | G | M | 1m_prior | 14 | Producer Price Index |
| HOUST | GR | M | 1m_prior | 16 | New Residential Construction |
| PERMIT | GR | M | 1m_prior | 16 | New Residential Construction |
| INDPRO | GR | M | 1m_prior | 17 | Industrial Production & Capacity Utilization |
| TCU | GR | M | 1m_prior | 17 | Industrial Production & Capacity Utilization |
| CPIAUCSL | G | M | 1m_prior | 18 | Consumer Price Index |
| CPILFESL | G | M | 1m_prior | 18 | Consumer Price Index |
| DGORDER | GR | M | 1m_prior | 26 | Advance Durable Goods |
| HSN1F | GR | M | 1m_prior | 26 | New Residential Sales |
| AMDMVS | GR | M | 1m_prior | 26 | Advance Durable Goods |
| AMDMTI | GR | M | 1m_prior | 26 | Advance Durable Goods |
| A261RX1Q020SBEA | GR | Q | prior_quarter | 28 | Gross Domestic Product |
| GDPC1 | GR | Q | prior_quarter | 28 | Gross Domestic Product |
| PCEPILFE | G | M | 1m_prior | 30 | Personal Income and Outlays |
| PCEC96 | GR | M | 1m_prior | 30 | Personal Income and Outlays |
| PCEPI | G | M | 1m_prior | 30 | Personal Income and Outlays |
| DSPIC96 | GR | M | 1m_prior | 30 | Personal Income and Outlays |
| TTLCONS | GR | M | 2m_prior | 33 | Construction Spending |
| ULCNFB | GL | Q | prior_quarter | 34 | Productivity and Costs |
| AMTMUO | GR | M | 2m_prior | 35 | Manufacturers' Shipments, Inventories, Orders (M3) |
| BOPTEXP | GR | M | 2m_prior | 35 | US International Trade in Goods & Services |
| BOPTIMP | GR | M | 2m_prior | 35 | US International Trade in Goods & Services |
| WHLSLRIMSA | GR | M | 2m_prior | 37 | Wholesale Trade |
| JTSJOL | GL | M | 2m_prior | 42 | Job Openings and Labor Turnover (JOLTS) |
| BUSINV | GR | M | 1m_prior | 44 | Manufacturing and Trade Inventories |

## 3. Esempi di bordo frastagliato (`as_of`)

Per ogni `as_of` D: ultimo **mese di riferimento** di cui il dato sarebbe già pubblicato a D (= ultimo p con `fine_mese(p)+delay ≤ D`). Le serie con delay grande si fermano prima → il bordo è frastagliato, non piatto.

### as_of = 2016-09-16

| Serie | Delay | Ultimo mese disponibile |
|---|---:|---|
| GACDISA066MSFRBNY | -14 | 2016-09 |
| GACDFSA066MSFRBPHI | -11 | 2016-08 |
| ISM_PMI | 3 | 2016-08 |
| ISM_PRICES | 3 | 2016-08 |
| ISM_EMP | 3 | 2016-08 |
| USPRIV | 5 | 2016-08 |
| ISM_NMI | 5 | 2016-08 |
| PAYEMS | 7 | 2016-08 |
| UNRATE | 7 | 2016-08 |
| IR | 13 | 2016-08 |
| IQ | 13 | 2016-08 |
| RSAFS | 14 | 2016-08 |
| PPIFIS | 14 | 2016-08 |
| HOUST | 16 | 2016-08 |
| PERMIT | 16 | 2016-08 |
| INDPRO | 17 | 2016-07 |
| TCU | 17 | 2016-07 |
| CPIAUCSL | 18 | 2016-07 |
| CPILFESL | 18 | 2016-07 |
| DGORDER | 26 | 2016-07 |
| HSN1F | 26 | 2016-07 |
| AMDMVS | 26 | 2016-07 |
| AMDMTI | 26 | 2016-07 |
| A261RX1Q020SBEA | 28 | 2016-06 |
| GDPC1 | 28 | 2016-06 |
| PCEPILFE | 30 | 2016-07 |
| PCEC96 | 30 | 2016-07 |
| PCEPI | 30 | 2016-07 |
| DSPIC96 | 30 | 2016-07 |
| TTLCONS | 33 | 2016-07 |
| ULCNFB | 34 | 2016-06 |
| AMTMUO | 35 | 2016-07 |
| BOPTEXP | 35 | 2016-07 |
| BOPTIMP | 35 | 2016-07 |
| WHLSLRIMSA | 37 | 2016-07 |
| JTSJOL | 42 | 2016-07 |
| BUSINV | 44 | 2016-07 |

### as_of = 2020-05-15

| Serie | Delay | Ultimo mese disponibile |
|---|---:|---|
| GACDISA066MSFRBNY | -14 | 2020-04 |
| GACDFSA066MSFRBPHI | -11 | 2020-04 |
| ISM_PMI | 3 | 2020-04 |
| ISM_PRICES | 3 | 2020-04 |
| ISM_EMP | 3 | 2020-04 |
| USPRIV | 5 | 2020-04 |
| ISM_NMI | 5 | 2020-04 |
| PAYEMS | 7 | 2020-04 |
| UNRATE | 7 | 2020-04 |
| IR | 13 | 2020-04 |
| IQ | 13 | 2020-04 |
| RSAFS | 14 | 2020-04 |
| PPIFIS | 14 | 2020-04 |
| HOUST | 16 | 2020-03 |
| PERMIT | 16 | 2020-03 |
| INDPRO | 17 | 2020-03 |
| TCU | 17 | 2020-03 |
| CPIAUCSL | 18 | 2020-03 |
| CPILFESL | 18 | 2020-03 |
| DGORDER | 26 | 2020-03 |
| HSN1F | 26 | 2020-03 |
| AMDMVS | 26 | 2020-03 |
| AMDMTI | 26 | 2020-03 |
| A261RX1Q020SBEA | 28 | 2020-03 |
| GDPC1 | 28 | 2020-03 |
| PCEPILFE | 30 | 2020-03 |
| PCEC96 | 30 | 2020-03 |
| PCEPI | 30 | 2020-03 |
| DSPIC96 | 30 | 2020-03 |
| TTLCONS | 33 | 2020-03 |
| ULCNFB | 34 | 2020-03 |
| AMTMUO | 35 | 2020-03 |
| BOPTEXP | 35 | 2020-03 |
| BOPTIMP | 35 | 2020-03 |
| WHLSLRIMSA | 37 | 2020-03 |
| JTSJOL | 42 | 2020-03 |
| BUSINV | 44 | 2020-03 |
