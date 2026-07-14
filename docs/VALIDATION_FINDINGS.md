# Cosa dice il validatore — i risultati, e cosa significano

Compagno di `VALIDATION_REPORT.md` (la tabella, **auto-generata**: non scrivere lì, una
nuova run la sovrascrive). Qui sta l'interpretazione, che è scritta a mano e resta.

Run: `python -m mcmc.validate.run --full --coverage 8` — T=600, 1200 iter × 2 catene,
entrambi i rami, 8 dataset per la copertura. Wall-clock 2h 13m.

```
recuperato                 22
recuperato con bias noto   10
non identificato           31
mai testato                 6
ROTTO                       0     <- il campionatore fa quello che dice di fare
```

---

## 1. Nessun ROTTO — ma non è la notizia

Zero rossi significa **una cosa sola**: il campionatore non fa nulla di *sbagliato*. Non
significa che il modello funzioni. Le 31 righe «non identificato» sono la notizia vera, e
si dividono in **due categorie che è fondamentale non confondere**, perché hanno rimedi
opposti:

* **«il DATO non lo pinza»** — limite informativo. Nessun rimedio nel campionatore: o più
  dati, o una specificazione diversa, o lo si dichiara.
* **«la CATENA non converge»** — limite del *campionatore*. Un rimedio esiste.

Il validatore le distingue e lo scrive in ogni riga.

---

## 2. Il risultato che pesa sulla tesi: `rho` è attenuato su **entrambi** i blocchi

`rho` è ciò che dà lo **skew** alla densità predittiva del PIL. E poiché **non c'è nessun
secondo stadio quantilico**, quella densità *è* il deliverable: non esiste un passo a valle
che assorba l'errore.

| | vero | stimato | rapporto | **copertura CI 90%** |
|---|---|---|---|---|
| `rho^u_0` (dominante) | −0.70 | −0.49 | **0.71** | **25 %** |
| `rho^u_2` | +0.45 | +0.34 | **0.76** | **50 %** |
| **`rho^eps`** (idiosincratico) | −0.30 | −0.16 | **0.54** | **12 %** |
| `rho^u_1` (debole) | −0.15 | −0.19 | — | 75 % ⟵ *copre perché il CI è larghissimo* |

Un CI al 90% deve coprire il vero **circa il 90% delle volte**. Ne copre il **12–50%** sui
canali identificati, su **8 dataset indipendenti**. Non è sfortuna: è **bias sistematico**.

E la direzione è la peggiore possibile: un `|rho|` compresso verso zero **sottostima la
coda sinistra**, cioè sottostima il rischio — l'errore che un esercizio di Growth-at-Risk
non può permettersi.

> `rho^eps` — il leverage **idiosincratico** — era nel DGP di quasi ogni test e **non era
> mai stato confrontato col vero**. È il blocco che proietta il PIL, ed è il *più*
> attenuato dei tre.

---

## 3. La causa, isolata: **errors-in-variables**, non un bug, non Omori

Il check `check_oracle` congela il path, la volatilità e i pesi **veri**, e fa girare
**solo** il draw di Family C:

```
scarto assoluto dal vero:   z esatto  [0.004, 0.048, 0.032]
                            Omori     [0.007, 0.052, 0.032]
```

Due conclusioni, entrambe forti:

1. **Il conditional di Family C è corretto.** Con le latenti note, `rho` si recupera
   praticamente esatto su tutti e tre i canali. Non c'è nessun bug nell'algebra.
2. **La linearizzazione di Omori NON attenua.** Il regressore di Omori e quello esatto
   danno lo *stesso* numero (scarti che differiscono di 0.003–0.004).

⇒ L'attenuazione entra **tutta dall'incertezza sulle latenti**: il path di volatilità è
*stimato*, quindi il regressore `z` è rumoroso, e una regressione con regressore rumoroso
attenua verso zero. È **una proprietà del problema**, non un difetto riparabile nel codice.

Confermato per esclusione: l'attenuazione **sopravvive** al cambio di prior (×0.74 anche
con inverse-Gamma), al cambio di kernel (griddy o RW), e al cambio di ramo (Branch A, che
è **esatto**, attenua *di più*).

**Questa è la domanda da portare a Ciganovic**, e ora è posta con precisione: non «il mio
sampler ha un bug», ma «il parametro che porta la coda è attenuato per costruzione — quanto
ci si può fidare della coda?».

---

## 4. Un'ipotesi elegante, e falsa: il prior **non** è la causa

Avevo costruito questa catena: *il half-Normal ha B=1 quando i `sigma` veri sono ~0.2 ⇒ è
largo dieci volte troppo ⇒ gonfia `sigma^2_eta` ⇒ il path esce troppo ondulato ⇒ un AR(1)
adattato a un path ondulato dà `phi` basso ⇒ e un path rumoroso attenua `rho`.*

Spiegava tutto. **I dati la rifiutano su ogni anello** (esperimento `tune_B`, 4 bracci):

| B | σ²_η / vero | φ^ε | ρ/vero (k=0) | ESS(ρ) |
|---|---|---|---|---|
| inverse-Gamma | [1.20, **0.46**, 0.68] | **0.32** | **0.74** | [52, 18, 75] |
| **1.0** (attuale) | [1.77, 1.25, 5.82] | 0.35 | 0.71 | **[63, 18, 88]** |
| 0.5 | [1.51, **2.09**, 2.48] | 0.35 | 0.68 | [62, 29, 60] |
| 0.25 | [1.49, **3.60**, 2.74] | 0.36 | 0.75 | [35, 25, 62] |

* `phi^eps` collassa **anche con l'inverse-Gamma** (0.32) ⇒ non è il prior;
* `rho` è attenuato **anche con l'inverse-Gamma** (×0.74) ⇒ non è il prior;
* abbassare `B` **peggiora** `sigma^2_eta` sul canale debole (×1.25 → ×2.09 → ×3.60) e
  **abbassa** l'ESS di `rho`.

**Decisione: `half_normal_B = 1.0` resta.** È il migliore dei quattro valori testati.

---

## 5. `phi^eps = 0.33` è **P2**, non un bug

Il campionatore legge la volatilità idiosincratica (vera persistenza **0.94**) come
**rumore bianco** (0.07–0.35). Il conto che spiega perché:

```
sd incondizionata di log h^eps  =  0.12 / sqrt(1 - 0.94^2)  =  0.35
```

Il canale comune **debole** — che sappiamo **non essere identificato** a T=600 — ha sd
incondizionata **0.46**. Quello idiosincratico ne ha **0.35**: è *sotto* la soglia già
misurata. La SV idiosincratica, con quei parametri e quel `T`, **non porta abbastanza
segnale per essere identificata**.

Verdetto: *non identificato*. Ed è un risultato **di sostanza**: mette in discussione se la
SV idiosincratica valga la complessità che costa.

---

## 6. Scoperta nuova: `Q` e `Lambda` **non convergono** sotto SV

| | ESS | R-hat |
|---|---|---|
| `A` (VAR) | 510 / 232 | 1.00 / 1.07 ✓ |
| **`Q`** | **15 / 10** | **1.23 / 1.91** |
| **`Lambda`** | **6 / 7** | **2.00 / 2.32** |
| `R` | 109 / 52 | 1.03 / 1.16 ✓ |

`A` e `R` mescolano benissimo; `Q` e `Lambda` no, **su entrambi i rami**. E non è casuale
*quali* due: sono esattamente i parametri con una **relazione di scala** — `Lambda` con i
fattori (`f → cf`, `Lambda → Lambda/c`), `Q` con il livello di `log h`
(`Var(u) = √H·Q·√H`). Il sospetto è una cresta di scala percorsa lentamente.

**Nessun test guardava `Q` con la volatilità accesa** (`test_passo1` la valida contro l'EM,
ma *senza* SV). Era invisibile. **Da indagare.**

---

## 7. ASIS: corretto, ma la leva su `rho` è il **prior**

* **Correttezza** ✓ — a prior fissato, ASIS on/off campionano lo *stesso* target
  (|Δφ| = 0.009 su B, 0.001 su A). Se divergessero, ASIS sarebbe **rotto**.
* **Efficienza** — il guadagno di ESS su `phi` è modesto (×0.31 in ESS/draw: ASIS *non*
  paga, in questa configurazione).
* **La leva su `rho` è il PRIOR, non l'interweaving**: a ASIS fissato, passare da IG a
  half-Normal moltiplica l'ESS/draw di `rho` per **×3.8** (Branch B).

⚠ **Il confondimento è strutturale**: `gibbs.py` **forza** `half_normal` quando
`use_asis=True`, e **non è un bug** — CP e NCP devono esprimere lo *stesso* prior perché
l'interweaving campioni il giunto esatto. Il validatore **fa fallire** ogni confronto su
ASIS che vari anche il prior (`_guard_asis_comparison`). La via alternativa (draw NCP non
coniugato sotto IG, ~30 righe) è documentata e **non implementata**.

⇒ La giustificazione di ASIS su `rho` nel `.tex` («meglio sigma ⇒ meglio rho») **è
falsificata**, ed è già stata riscritta.

---

## 8. QML sotto leverage: progresso, non soluzione

La deriva corretta (`z = M·eps` esatto, transizione FFBS **piena**) **elimina P5** — a
`corr(Q)=0.8` recupera `rho` dove il decoupled lo distrugge — ma **`phi` di un fattore
collassa ancora**. Resta dietro `allow_experimental=True`, con il fallimento **congelato**
in un check invece che nascosto.

Sotto **Branch A** solleva `ValueError` **per costruzione**: A non forma *mai* una
covarianza di misura (nessuna mistura, nessuna linearizzazione). **Non è una lacuna, è una
proprietà** — ed è la ragione per cui A è la controparte esatta.

---

## 9. Tre difetti che il validatore ha trovato **in sé stesso**

Vale la pena registrarli: sono la stessa patologia in forme diverse, e sono la ragione per
cui le tre regole non sono negoziabili.

1. Dichiarava `Q` **«recuperato»** leggendo un errore relativo da una catena con **ESS 6 e
   R-hat 2.3**. Un numero così non è una stima sbagliata: **non è una stima**.
   ⇒ **cancello di convergenza**: nessun verdetto positivo da una catena che non ha esplorato.
2. Dava **ROTTO** all'oracolo perché asseriva un **rapporto** sul canale **debole**, dove
   `rho` vero è −0.15 e il rapporto è rumore puro. Asseriva una proprietà di una quantità
   **non identificata** — il peccato esatto che il validatore esiste per impedire.
   ⇒ soglia sull'**errore assoluto**.
3. Misurava il guadagno del prior confrontando **ESS grezzi fra fit con numero di catene
   diverso**, gonfiando ×2 un risultato che era *già vero*: il modo più insidioso di
   sbagliare. ⇒ **ESS per draw**.

---

## Cosa resta aperto

1. **L'attenuazione di `rho`** — è il bloccante per il GaR, ed è una domanda di
   **identificazione**, non di codice. Rimedi possibili: prior informativo su `rho`
   (economicamente difendibile: il leverage è negativo, lo sappiamo da trent'anni), oppure
   una **correzione del bias calibrata per simulazione** (il fattore ora lo conosciamo).
2. **`Q` e `Lambda` che non convergono** sotto SV — nuovo, mai guardato.
3. **La SV idiosincratica non è identificata** — vale la complessità che costa?
4. **`w^u`, `w^eps`, `mu`** restano «mai testati» (i pesi non sono immagazzinati nei draws;
   `mu` è fissato a 0 per identificazione — la sua *assenza* è una scelta, non un buco).
