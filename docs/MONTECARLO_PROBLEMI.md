# Problemi emersi dalle recovery Monte Carlo

> **Stato: 2026-07-25.** Aggiorna la versione del 2026-07-20 (che a sua volta
> riscriveva quella del 2026-07-19). La sezione finale "Ipotesi falsificate"
> elenca esplicitamente cosa è stato scartato e perché: serve a non riprendere
> per buone conclusioni che sono state smentite.
>
> **Cosa cambia rispetto al 2026-07-20.** Le recovery sono state rigirate da
> zero su tutte e 15 le celle, seed invariato. **I risultati sono identici cifra
> per cifra** (§0-bis): nessun numero della versione precedente è stato
> corretto. Sono invece state aggiunte tre metriche che prima non venivano
> calcolate — `Q` fuori diagonale (PROBLEMA G), `Λ` per colonna (§4-bis), `φ`
> idiosincratico (§4-ter) — più la scomposizione di `A` fra raggio spettrale e
> matrice intera.
>
> **Come leggere.** Le recovery girano su tutte e 15 le celle (3 spec × 5
> varianti), a due lunghezze campionarie (T = 500 e T = 2000), seed 42. Per ogni
> cella si simula dal `theta` stimato su quella stessa cella e si ri-stima da
> zero con una PCA di partenza indipendente. Artefatti in
> `output/recovery/final/<spec>/<variante>/`.
>
> Comando: `python src/monte_carlo_recovery.py --spec X --variant Y`
> (`--all` per tutte, `--force` per ignorare la cache; una run completa va
> spezzata per cella — ~2,5 h in totale, dominata dalle tre celle
> `student_t_ar1`).
>
> Il criterio di lettura è: **un errore deve calare passando da T = 500 a
> T = 2000.** Se cala è varianza campionaria; se resta fermo o cresce è
> distorsione o mancata identificazione. **Attenzione**: questo criterio
> presuppone che le due run siano ugualmente convergiute — vedi la nota sulla
> tolleranza nelle ipotesi falsificate.

---

## 0. La premessa: le celle gaussiane validano la macchina

**Tutte e sei le celle gaussiane recuperano bene e migliorano con T.**

| | Λ (Procruste-block) T=500 → T=2000 | corr. min fra fattori |
|---|---|---|
| `fed_overlap/gaussian` | 0.104 → 0.060 | 0.979 → 0.982 |
| `fed_overlap/gaussian_ar1` | 0.106 → 0.055 | 0.959 → 0.969 |
| `diag4/gaussian` | 0.049 → 0.028 | 0.985 → 0.988 |
| `diag4/gaussian_ar1` | 0.072 → 0.033 | 0.985 → 0.988 |
| `diag3/gaussian` | 0.055 → 0.033 | 0.977 → 0.982 |
| `diag3/gaussian_ar1` | 0.068 → 0.029 | 0.977 → 0.982 |

Segni dei fattori corretti ovunque (0 flip). Autovalori di `A` con errore ≤ 4.5%
in tutte le run, incluse le celle problematiche: **la dinamica dei fattori non è
fra i problemi**. L'AR(1) idiosincratico **di per sé non rompe nulla**: le celle
`gaussian_ar1` sono pulite quanto le `gaussian`.

**Questo è il discriminante che regge tutte le diagnosi che seguono.** Le celle
gaussiane girano sullo stesso identico codice — stesso smoother, stesso M-step,
stessa inizializzazione PCA — e recuperano tutto: `Q` diagonale 0.87–1.25, `Q`
fuori diagonale entro 0.02, `Λ` entro 0.15, fattori ≥ 0.959. Un difetto di
implementazione non saprebbe risparmiare sei celle su quindici. Ogni volta che
sotto si scrive «limite di identificazione, non bug», l'argomento è questo.

**Caveat sulla mediana.** La tabella §0 storica riportava `diag(Q)` come mediana
sugli elementi, e questo nascondeva un caso: `diag3/gaussian` ha `Q` sul fattore
A che *peggiora* con T (rapporto 1.16 → 1.25) mentre gli altri due elementi
convergono a 1.00. La mediana (3.0% → 2.5%) non lo mostra. È un caso isolato e
di entità modesta, ma la frase «`Q` è recuperato esattamente nelle gaussiane»
va riferita agli elementi, non alla mediana.

---

## 0-bis. La tornata del 2026-07-25: riproducibilità confermata

Le 15 celle sono state ricalcolate da zero con `--force`. Verifica a livello di
contenuto: **30 archivi, 1542 array confrontati elemento per elemento, zero
differenze** (confronto con `equal_nan`, quindi coincidono anche i NaN delle
celle gaussiane).

La cache è stata esclusa: zero `[riuso]` nei log, EM rigirato in tutte le celle
(`n_iter` 12–213 con le traiettorie di verosimiglianza complete), 41 minuti di
calcolo effettivo per `diag3/student_t_ar1`.

**Perché conta.** Gli artefatti precedenti sono del 19-07; i sorgenti che
producono la stima (`kalman.py`, `em_main.py`, `simulate_dfm.py`,
`em_initialization.py`) sono stati modificati il 23–24 luglio. I numeri non si
sono mossi di un bit: **quel refactor è behavior-preserving**. È un test di
regressione superato a seed fisso, non banale viste le ~275 righe di codice
toccate nel solo `kalman.py`.

**Corollario da tenere presente.** Gli interventi su `monte_carlo_recovery.py` e
su `aggregate_replications` (righe generate su `r`, appaiamento dinamico
`diagQ_hat_<f>` → `diagQ_star_<f>`) sono **codice di reporting**: cambiano quale
numero viene stampato e con che etichetta, non l'M-step. Non possono migliorare
`diag(Q)`, e infatti non l'hanno migliorata. Vedi "Difetti di reporting".

**Le due run non convergiute si riproducono identiche**: `diag4/student_t_ar1` a
T=500 (24 iterazioni, 9 violazioni di monotonicità) e `fed_overlap/student_t_ar1`
a T=500 (25 iterazioni, 10 violazioni), contro le 130–213 iterazioni e 0
violazioni delle celle sorelle. Vedi PROBLEMA H.

---

## PROBLEMA A — `nu_eps`: **RISOLTO, causa isolata**

Causa: **i pesi idiosincratici per serie**, non l'AR(1).

La quinta variante `student_t_ar1_shared` (idio AR(1) + peso condiviso) è stata
aggiunta il 2026-07-20 proprio per separare le due caratteristiche, che nelle
quattro varianti storiche erano confuse. Varia **una cosa sola** rispetto a
`student_t_ar1`.

| schema pesi | `nu_eps` rel.err T=2000 | `corr(w_eps stimati, veri)` |
|---|---|---|
| condiviso, no AR(1) (`student_t`) | 1.5% / 1.6% / 1.5% | 0.899 / 0.891 / 0.887 |
| **per serie** (`student_t_ar1`) | **48.4% / 72.6% / 84.6%** | **0.314 / 0.289 / 0.290** |
| **condiviso + AR(1)** (`..._shared`) | **0.96% / 0.14% / 0.11%** | **0.887 / 0.885 / 0.885** |

Cambiando **solo** lo schema dei pesi, con l'AR(1) invariato, l'errore passa da
48–85% a **sotto l'1%**. È un esperimento controllato: la conclusione è causale,
non congetturale.

**Meccanismo.** Col peso condiviso ogni peso è informato da `M` osservazioni
(posteriore stretto); coi pesi per serie da **una sola** (`m_obs = 1`, posteriore
largo). `nu_eps` andrebbe stimato dalla dispersione di quantità che sono esse
stesse quasi tutte rumore. Conferma indipendente: `corr(w_eps)` è **identica** a
T=500 e T=2000 (0.3157 → 0.3138), incompatibile con un problema di convergenza o
di quantità di dati.

**Sistematico o casuale?** Nelle 6 celle a peso condiviso l'errore è **casuale e
si cura**: segno negativo (−0.47 a −0.01 in livelli), e a T=2000 scende sotto
l'1% in tre celle su sei. Nelle 3 celle a pesi per serie è **sistematico**:
segno positivo in tutte e tre le spec, +2.6 / +4.4 / +5.2 in livelli, e non cala
con T.

**Direzione.** `nu_eps` sovrastimato (vero ≈ 6.1, stimato ≈ 10.5–11.5) significa
code più leggere del vero, quindi meno abbattimento degli outlier: erode la
robustezza che è la ragione d'essere della specificazione Student-t.

---

## PROBLEMA C — `R`: **RISOLTO, stessa causa di A**

| variante | `R` mediana rel.err T=2000 |
|---|---|
| gaussiane e `student_t` | 2.0–3.1% |
| `student_t_ar1` (per serie) | **14.0–16.5%** |
| `student_t_ar1_shared` | **2.14% / 2.37% / 2.35%** |

Il peso condiviso riporta `R` al livello delle celle gaussiane. Conferma
l'ipotesi che C fosse a valle di A: sotto AR(1) `R` è la varianza
dell'innovazione fresca e il suo aggiornamento è una **varianza pesata** da
`w_eps`; pesi sbagliati, `R` sbagliata. **A e C sono un problema solo.**

---

## PROBLEMA D — `diag(Q)` sul fattore lavoro: **APERTO, ma riqualificato**

Non è un difetto dello stimatore. È il **fit sui dati veri** che spegne il
fattore lavoro sotto code pesanti.

| | `Q[L]` vero | stimato | rapporto |
|---|---|---|---|
| 6 celle **gaussiane** | 0.86 – 0.95 | 0.86 – 0.93 | **0.98 – 1.00** |
| 9 celle **Student-t** | 0.010 – 0.033 | 0.16 – 0.33 | 5 – 31 |

**Sotto gaussiana `Q[L]` è recuperato perfettamente.** Il rapporto di 20–30 nelle
celle Student-t è quasi tutto denominatore: è il *vero* a crollare di novanta
volte, non la stima a impazzire. E la stima è stabile (`diag4` dà 0.2210, 0.2226,
0.2223 su tre varianti diverse) perché converge all'**attrattore** che rende
`sd(f_L) ≈ 1`, cioè al fattore coerente con la Convenzione 1.

**Rapporto stimato/vero per fattore, per spec** (le 9 celle Student-t):

| spec | fattore L | fattore R | altri fattori |
|---|---|---|---|
| `diag3` | **16.3 – 30.7×** | — | A: 0.75–1.09, **ma `_shared` 3.40–3.99×** |
| `diag4` | **14.4 – 20.3×** | 1.67 – 2.31× | S: 0.50–1.05, N: 0.59–0.94 |
| `fed_overlap` | **5.4 – 19.3×** | 1.67 – 2.41× | G: 0.79–1.14, S: 0.52–0.97 |

Peggiore cella in assoluto: `diag3/student_t_ar1_shared`, `Q[L]` a 30.1 → 30.7×.

**Quote della varianza comune del fattore L:**

| spec | gaussiana | Student-t (tutte e 3 le varianti) |
|---|---|---|
| `diag3` | 22.3% | 1.6% / 1.7% / 2.0% |
| `diag4` | 19.4% | 1.5% / 1.3% / 1.5% |
| `fed_overlap` | 17.4% | 2.4% / 2.3% / 1.7% |

**Passando allo Student-t il blocco occupazione sparisce dal modello.** Non
dipende dai pesi (il condiviso non lo corregge) né dall'AR(1).

**Indiziato**: la sottigliezza del blocco. L = `PAYEMS, UNRATE, USPRIV, JTSJOL,
ULCNFB` — cinque serie in tutte e tre le spec, di cui PAYEMS e USPRIV correlate
0.998, quindi ~4 indipendenti. Le sue innovazioni genuine diventano
indistinguibili da osservazioni anomale, e il modello le abbatte come outlier
invece di trattarle come innovazioni.

**NUOVO — un caso che tocca il deliverable.** In `diag3/student_t_ar1_shared` la
stessa compressione colpisce il fattore **A**, cioè i 25 serie usati nel nowcast
del PIL: `Q[A]` a 3.40–3.99× e (§4-bis) `Λ[A]` a 0.57–0.63 della norma vera. È
l'unica cella in cui il fenomeno esce dal blocco lavoro e raggiunge il fattore
del forecast. Va tenuto presente se `_shared` viene usata come controllo su
`diag3`.

**Impatto sul deliverable altrove: nullo.** Il forecast del PIL gira sul fattore
reale/globale, mai su L.

---

## PROBLEMA B — fattori sotto sovrapposizione: **è il fattore L**

| cella | corr. minima T=500 → T=2000 |
|---|---|
| `fed_overlap/student_t` | 0.700 → 0.744 |
| `fed_overlap/student_t_ar1` | 0.657 → 0.724 |
| `fed_overlap/student_t_ar1_shared` | 0.759 → 0.785 |
| tutte le altre | ≥ 0.95 |

Il fattore mal correlato è **L**, lo stesso di D. In `diag3`/`diag4` L ha
correlazione 0.99 e solo `Q` è rotto; in `fed_overlap` L compete col globale G
(37 serie) che carica sulle stesse serie, e la sovrapposizione aggrava un
fattore già fragile. **B e D sono lo stesso fenomeno**, che si manifesta anche
sulla correlazione solo dove c'è sovrapposizione.

Migliora con T in tutte e tre le varianti (0.66 → 0.72, 0.70 → 0.74,
0.76 → 0.79), quindi **non è pura non-identificazione**: c'è informazione che si
accumula, semplicemente troppo poca.

Degrada anche il fattore **R** sotto Student-t nella stessa spec: 0.948–0.978
contro 0.993–0.995 delle celle gaussiane.

**Caveat che resta valido**: sotto sovrapposizione i singoli fattori non sono
separatamente identificati per costruzione, quindi una correlazione bassa sul
fattore individuale è in parte attesa. La metrica invariante a rotazioni
(componente comune) non è ancora fra quelle calcolate.

---

## PROBLEMA E — `nu_u` distorto verso l'alto in **tutte** le celle Student-t

`nu_u` è l'indice di coda delle innovazioni **dei fattori**, quindi tocca anche
il fattore usato nel forecast.

| cella | vero | T=500 | T=2000 |
|---|---|---|---|
| `diag3/student_t` | 3.895 | 12.4% | **41.9%** |
| `diag3/student_t_ar1` | 3.578 | 17.0% | 32.0% |
| `diag3/..._shared` | 4.574 | 20.3% | **65.7%** |
| `diag4/student_t` | 3.868 | 0.2% | 13.0% |
| `diag4/student_t_ar1` | 3.755 | 2.5% | 8.5% |
| `diag4/..._shared` | 3.871 | 2.4% | 13.2% |
| `fed_overlap/student_t` | 3.856 | 1.3% | 9.0% |
| `fed_overlap/student_t_ar1` | 3.952 | 8.5% | 16.0% |
| `fed_overlap/..._shared` | 4.191 | 22.8% | 32.1% |

**Nove celle su nove peggiorano con T**, e in tutte la stima si sposta verso
l'alto. Non dipende dall'AR(1) né dallo schema dei pesi (peggiora anche col
condiviso). `nu_u` sovrastimato = code dei fattori più leggere del vero.

**Sistematico, senza ambiguità.** In segno con cui l'errore si presenta:

| | celle con errore positivo | errore medio in livelli |
|---|---|---|
| T=500 | 5/9 | +0.268 |
| T=2000 | **9/9** | **+1.048** |

Il segno diventa unanime e l'errore **quadruplica** raddoppiando due volte la
dimensione campionaria. Questo è esattamente il criterio di lettura del
preambolo, applicato al contrario: non solo non cala, cresce.

**Ipotesi in piedi**: `nu_u` eredita la distorsione di `Q` attraverso i pesi
`w_u`, che si costruiscono da residui di stato normalizzati per `Q`. Due
supporti concordanti:

1. `corr(w_u stimati, veri)` sta a **0.476 – 0.615 in tutte e nove le celle e
   non migliora con T** (es. `diag3/student_t`: 0.500 → 0.511). I pesi sono metà
   rumore a qualunque dimensione campionaria, e `nu_u` viene stimato dalla loro
   dispersione.
2. L'ordinamento fra spec torna: `diag3` ha il `Q` peggiore *e* il `nu_u`
   peggiore, `diag4` il più contenuto in entrambi.

**Non è però provata.** Un limite di identificazione puro di norma si stabilizza;
un errore che *cresce* con T è insolito e lascia aperta la possibilità che il
problema stia nell'aggiornamento di `nu_u` stesso. Test che deciderebbe: fissare
`Q` al vero e ri-stimare solo `nu_u` — se il bias sparisce è ereditato, se resta
è nell'M-step di `nu_u`.

---

## PROBLEMA F — ottimi locali con i pesi per serie

Test: `python src/compare_theta_star_hat.py --all`, output in
`output/recovery/theta_star_vs_hat.txt`.

Si valutano `theta_star` (fit reale, che genera i dati) e `theta_hat` (ri-stima
sul simulato) **sul pannello simulato**, dove `theta_star` è il vero per
costruzione. Se `theta_hat` ha ELBO più basso, l'EM non ha raggiunto il punto
che ha generato i dati.

| celle | ELBO(`theta_hat`) − ELBO(`theta_star`) sul simulato |
|---|---|
| 6 gaussiane | da +41 a +70 (entro rumore) |
| 3 `student_t` | da −23 a +49 (entro rumore) |
| 3 `..._ar1_shared` | da −24 a +21 (entro rumore) |
| **3 `student_t_ar1`** (per serie) | **−1663, −1944, −2358** |

**Dodici celle su quindici non hanno ottimi locali. Le tre che ce l'hanno sono
esattamente quelle a pesi per serie**, con un divario di due ordini di grandezza.

Il confronto è controllato: `student_t_ar1` e `student_t_ar1_shared` condividono
tutto e differiscono solo per lo schema dei pesi. Passare da `T` a `T×M`
variabili latenti porta il divario da ~20 a ~2000 punti.

**È lo stesso strumento con cui `theta_star` compra il suo vantaggio.** Sui dati
veri `theta_star` vince di +582/+622/+902 col peso per serie contro
+69/+51/+63 col condiviso. La granularità che permette di abbattere singole
serie alza la verosimiglianza *e* scolpisce i massimi locali: guadagno in
adattamento e perdita in recuperabilità hanno la stessa origine.

**Conseguenza**: sulle tre celle a pesi per serie, parte dell'errore su `Q` è
artefatto di ottimizzazione e le metriche vanno lette come limite inferiore
della qualità raggiungibile.

---

## PROBLEMA G (NUOVO) — `Q` fuori diagonale attenuata verso zero

Metrica introdotta il 2026-07-25: non era mai stata calcolata. Si confronta la
**correlazione implicata** da `Q` (l'oggetto identificato: la scala per fattore
è assorbita da `Λ`), vero → stimato.

| cella | coppia | vero → stimato |
|---|---|---|
| `fed_overlap/student_t` T=2000 | G–L | **−0.79 → +0.01** |
| `fed_overlap/student_t` T=500 | G–L | −0.79 → −0.04 |
| `fed_overlap/student_t_ar1` T=2000 | G–L | −0.79 → −0.11 |
| `fed_overlap/student_t_ar1_shared` T=2000 | G–L | −0.56 → −0.29 |
| `fed_overlap/gaussian` T=2000 | R–L | 0.75 → 0.73 ✓ |
| `diag4/gaussian` T=2000 | R–L | 0.75 → 0.76 ✓ |
| `diag4/gaussian_ar1` T=2000 | S–R | 0.44 → 0.45 ✓ |

Il pattern è netto e vale su tutte e tre le spec: **le celle gaussiane
recuperano le correlazioni fuori diagonale entro 0.02, le Student-t le attenuano
verso zero**, e tanto più quanto più la correlazione vera è forte. Su
`fed_overlap` la correlazione più marcata del modello (G–L, −0.79) viene
azzerata. `diag3` e `diag4` mostrano la stessa attenuazione ma partono da
correlazioni vere piccole (0.04–0.36), quindi il fenomeno si nota meno.

`student_t_ar1_shared` è la meno peggio fra le Student-t (G–L −0.56 → −0.29),
ma resta attenuata di un fattore 2.

**Verdetto: limite di identificazione, stessa radice di E.** I pesi `w_u` sono
recuperati a `corr ≈ 0.5`: metà del segnale di scala è rumore, e quel rumore è
condiviso fra i fattori, quindi attenua la correlazione stimata. È la stessa
struttura errors-in-variables già documentata su `ρ` nel Gibbs
(`project_rho_attenuation_bias`).

**Da dichiarare in tesi**: `Q` non governa solo la larghezza della densità
predittiva, ma la sua **forma congiunta**. Una `Q` con le correlazioni schiacciate
sottostima la co-movimentazione degli shock ai fattori.

---

## PROBLEMA H (NUOVO) — due fit non convergono, riproducibili

| cella | T | iterazioni | violazioni di monotonicità |
|---|---|---|---|
| `diag4/student_t_ar1` | 500 | 24 | **9** |
| `fed_overlap/student_t_ar1` | 500 | 25 | **10** |
| celle sorelle | 500 / 2000 | 130 – 213 | 0 |

Escono per stallo con ΔL negativo. Erano già stati osservati il 2026-07-19 e si
**riproducono identici** il 2026-07-25 (stesso numero di iterazioni, stesse
violazioni), quindi non sono un incidente numerico occasionale.

**È l'unico candidato a un difetto di implementazione in tutto il documento.**
La monotonicità dell'ELBO è una garanzia teorica dell'EM: violarla nove o dieci
volte non è il "transito vicino a ν piccolo" che spiega le violazioni transitorie
isolate. Entrambi i casi stanno nella variante a pesi per serie a T=500 — la
stessa in cui il criterio del ciclo interno è `rms` invece di `max`.

**Conseguenza operativa**: i numeri di quelle due run vanno considerati non
validi. In particolare `diag4/student_t_ar1` T=500 dà `Q[S]` a 0.500 e
`Λ[S]` rel.err 0.402, entrambi fuori linea rispetto alla stessa cella a T=2000
(0.703 e 0.223) e alle celle sorelle — sono artefatti di non convergenza, non
misure.

---

## §4-bis (NUOVO) — `Λ` per colonna: la controparte esatta di D

Metrica introdotta il 2026-07-25. Errore relativo per fattore calcolato **sui
soli membri della mask**, affiancato dal rapporto di norma ‖Λ̂‖/‖Λ*‖ che separa
"colonna riscalata" da "colonna sbagliata".

| | gaussiane (6 celle) | Student-t (9 celle) |
|---|---|---|
| tutti i fattori | rel.err 0.006 – 0.146, norma 0.87 – 1.09 | — |
| fattore **L** | — | rel.err **0.73 – 0.81**, norma **0.19 – 0.27** |
| fattore **R** (`diag4`, `fed_overlap`) | — | rel.err 0.25 – 0.35, norma 0.66 – 0.76 |
| fattore **A** (solo `diag3/_shared`) | — | rel.err 0.37 – 0.44, norma 0.57 – 0.63 |

**Questo chiude il cerchio con D**: `Λ[L]` a 0.24 della norma vera e `Q[L]` a
20× sono lo stesso errore visto da due lati. È una scala **per fattore** che si
compensa — il prodotto `Λ f` resta corretto, i due pezzi separatamente no.

È anche la misura diretta che mancava per l'ipotesi falsificata n. 2: prima si
poteva solo dedurre che non fosse una scala comune (riscalando con lo scalare
ottimo l'errore passava da 0.514 a 0.465); ora la si legge colonna per colonna.

---

## §4-ter (NUOVO) — `φ` (ρ idiosincratico): **il parametro meglio recuperato**

Metrica introdotta il 2026-07-25. 18 run (3 varianti AR(1) × 3 spec × 2 T).

| | corr(φ̂, φ*) | pendenza di φ̂ su φ* | bias medio |
|---|---|---|---|
| T=500 | 0.931 – 0.988 | 0.78 – 0.94 | +0.010 – +0.047 |
| T=2000 | 0.950 – **0.997** | 0.82 – **0.978** | +0.006 – +0.043 |

Due fatti, in tensione fra loro:

1. **Attenuazione sistematica**: pendenza < 1 in tutte e 18 le run e bias
   positivo in tutte e 18. È la firma errors-in-variables, la stessa di G ed E.
2. **Ma qui si cura con T**, al contrario di quanto trovato nel Gibbs.
   `diag4/student_t_ar1` passa da pendenza 0.782 / bias 0.036 a pendenza
   **0.978** / bias 0.010. È distorsione di campione finito, **non** il pavimento
   strutturale documentato in `project_rho_attenuation_bias`.

**Unica eccezione**: `fed_overlap/student_t_ar1`, dove il bias *cresce*
(0.031 → 0.043) e la pendenza resta 0.79 → 0.82. È anche la cella con il fit
T=500 non convergiuto (PROBLEMA H), quindi il dato va riletto dopo aver risolto
quello.

**Conclusione**: `φ` non è fra i problemi. Va segnalato come risultato positivo —
l'Asse B recupera il suo parametro caratteristico.

---

## §4-quater (NUOVO) — `A`: raggio spettrale sano, matrice intera meno

Il raggio spettrale è recuperato con errore ≤ 4.4% in tutte e 30 le run,
tipicamente < 2% (§0). Ma la matrice `A` **intera** racconta un'altra storia:

| variante | ‖Â − A*‖ / ‖A*‖ (range sulle 3 spec) |
|---|---|
| `gaussian`, `gaussian_ar1` | 0.054 – 0.168 |
| `student_t`, `student_t_ar1` | 0.145 – 0.321 |
| `student_t_ar1_shared` | 0.240 – **0.531** |

Peggiore: `diag3/student_t_ar1_shared` (0.42 → 0.53, e peggiora con T). Sono le
interazioni **fuori diagonale** fra fattori a non essere recuperate, non la
persistenza. Coerente con l'essere a valle della distorsione di scala di D/§4-bis
piuttosto che un problema autonomo: la stessa cella ha `Q[A]` a 3.4–4.0× e
`Λ[A]` a 0.6.

La frase «la dinamica dei fattori non è fra i problemi» resta vera **se riferita
al raggio spettrale**, che è ciò che governa la persistenza delle previsioni.

---

## Difetti di reporting corretti

* **Il quarto fattore era omesso** (corretto il 2026-07-20). Il summary stampava
  3 righe di `diag(Q)` e 3 correlazioni con etichette fisse (`f_R`/`f_F`/`f_X`,
  `[0=real]`/`[1=fin.]`/`[2=other]`), ma `diag4` e `fed_overlap` hanno r=4. Su
  `fed_overlap` il quarto fattore è **il peggiore della cella** (corr 0.744, `Q`
  a 5.4×): leggendo il summary quella cella sembrava la più sana delle tre.
  Corretto in `monte_carlo_recovery.py`: righe generate su `r` con i nomi veri.
  **Precisazione (2026-07-25)**: il quarto fattore non è mai stato *droppato
  dalla stima* — era omesso dalla *tabella*. Prova nei timestamp: il `.npz` è del
  19-07 16:38–17:06, ma il `recovery_summary_*.txt` è stato riscritto il 20-07
  09:51, cioè il fix ha ri-stampato le **stesse** stime aggiungendo la riga.
* **L'appaiamento dinamico è anch'esso reporting.** L'accoppiamento
  `diagQ_hat_<f>` → `diagQ_star_<f>` in `aggregate_replications` serve a produrre
  bias/RMSE per fattore su `r` generico. Non tocca l'M-step: non può migliorare
  `diag(Q)` e non l'ha migliorata (§0-bis).
* **Due run non convergiute** nella tornata del 2026-07-19: si riproducono
  identiche e sono state promosse a PROBLEMA H invece di restare una nota.
* Conteggi celle, `T=497` residuo del vecchio `dataset_small` (le config final
  sono T=499), larghezze di colonna.

---

## Ipotesi falsificate — non riprenderle

1. **«Il peggioramento con T è sotto-convergenza.»** `tol_outer` è *relativa*
   (`|ΔL|/|L| < 1e-5`) e `|L|` cresce con T, quindi a T=2000 la soglia assoluta
   è ~4× più lasca (l'EM si ferma guadagnando ancora +0.55 contro +0.14 a
   T=500). L'osservazione è vera, la conclusione no: rigirando `diag3/student_t`
   a T=2000 con `tol_outer=2.5e-6` (59 iterazioni invece di 38) `nu_u` passa da
   **5.527 a 5.468** e `Q[min]` da **21.15 a 21.08**. Nulla. La distorsione è
   reale. `tol_outer` resta a `1e-5`: stringerlo non compra niente e romperebbe
   la confrontabilità fra celle.

2. **«L'errore su ΛΛ' è un artefatto della Convenzione 1 (scala comune).»**
   ΛΛ' è invariante a rotazioni ed è 0.46–0.59 in tutte le celle Student-t
   contro 0.07–0.20 nelle gaussiane. Ma riscalando con lo scalare comune ottimo
   l'errore passa solo da 0.514 a 0.465: **non è una scala comune**. È una scala
   **per fattore**, con Λ e Q che si compensano — su `diag3/student_t` la norma
   della colonna L di Λ sta a 0.233 del vero mentre `Q[L]` sta a 21×.
   *Ora misurato direttamente in §4-bis, colonna per colonna.*

3. **«Lo stimatore restituisce un modello sano, il DGP è patologico.»** Metà
   vera, ma la conclusione era sbagliata. È vero che `theta_hat` è internamente
   coerente (`sd(f)` 0.88–1.08 su tutti i fattori) e `theta_star` no
   (`sd(f_L)` 0.19–0.34). Ma `theta_star` ha ELBO **più alto su entrambi i
   pannelli**: non è lo stimatore che si rifiuta di riprodurre un punto cattivo,
   è l'EM che non raggiunge un punto migliore (PROBLEMA F).

4. **«La quinta configurazione non vale il costo.»** Valeva: ha chiuso A e C con
   attribuzione causale, ed è servita a isolare F. Tre celle e sei recovery.

5. **«I fix di reporting (quarto fattore, appaiamento dinamico) potevano
   sistemare `diag(Q)`.»** Falsificata dalla ri-stima del 2026-07-25: 30 run su
   30 identiche cifra per cifra. Sono codice di stampa; la stima non li attraversa.

---

## Cosa resta aperto

1. **Perché lo Student-t spegne il fattore lavoro** (quota 20% → 2%). È una
   questione di *specificazione*, non di stima: l'indiziato è il blocco da 5
   serie con due correlate 0.998. Test proposto: rifittare togliendo USPRIV da
   L e vedere se `sd(f_L)` implicata risale verso 1.
2. **`nu_u`** (PROBLEMA E) — probabilmente a valle di D. Test che decide:
   fissare `Q` al vero e ri-stimare solo `nu_u`.
3. **PROBLEMA H** — le due non convergenze riproducibili sono l'unico sospetto
   di bug rimasto. Da guardare nel ciclo ECM interno sotto pesi per serie.
4. **Raggiungibilità del punto migliore** nelle celle a pesi per serie: un warm
   start da `theta_star` direbbe se il massimo è raggiungibile. Ma una recovery
   inizializzata al vero **non è più una recovery**: diagnostica la superficie,
   non valida lo stimatore.
5. **La metrica sulla componente comune** (invariante a rotazioni) per chiudere
   il caveat di B.
6. **Replicazione su più semi.** Tutto è su seed 42. Le separazioni osservate
   sono di due ordini di grandezza, quindi non plausibilmente casuali, ma una
   conferma rafforzerebbe le tabelle. Nota: la ri-stima del 2026-07-25 conferma
   la **riproducibilità** a seed fisso, che è cosa diversa dalla robustezza al
   seme.

---

## Implicazioni per il forecast

Il fattore usato nel nowcast del PIL è recuperato bene, con una sola eccezione:

| spec | fattore | `Q` rapporto T=2000 | giudizio |
|---|---|---|---|
| `diag3` | A (25 serie) | 0.85 – 1.09 | pulito — **tranne `_shared`: 3.40** |
| `fed_overlap` | G (37 serie) | 0.83 – 0.98 | pulito |
| `fed_overlap` / `diag4` | R (19 serie) | 1.90 – 2.41 | gonfiato ~2×, in peggioramento con T |

I fattori aggregati (A, G) sono puliti. Il fattore R, dove l'attività reale è
spezzata in sotto-blocchi, ha `Q` gonfiata ~2× e in peggioramento con T — da
dichiarare, perché `Q` governa la larghezza della densità predittiva, ma non
bloccante.

**Raccomandazione**: `student_t_ar1` come specificazione principale (vince la
verosimiglianza sui dati veri di 582–902 punti e non degrada il fattore reale),
`student_t_ar1_shared` come controllo — **con l'avvertenza che su `diag3` la
variante `_shared` distorce il fattore A** (PROBLEMA D), quindi lì il controllo
va letto con cautela. Il limite su `nu_eps` nella per-serie va dichiarato:
riguarda le code idiosincratiche, che nel nowcast del PIL pesano poco. Da
dichiarare anche l'attenuazione di `Q` fuori diagonale (PROBLEMA G), che
riguarda la forma congiunta della densità predittiva.
