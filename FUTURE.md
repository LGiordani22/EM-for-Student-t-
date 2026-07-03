# Sviluppi futuri — estensioni valutate e rinviate

Questo file raccoglie le estensioni del modello che sono state **valutate ma
rinviate**, ciascuna con il relativo **nodo/difficoltà principale**, così da non
perdere il ragionamento già fatto e da avere materiale per la sezione "sviluppi
futuri" della tesi.

Lo **stato attuale della tesi** — First Stage DFM Student-t mixed-frequency,
completo e validato (EM con down-weighting Student-t su fattori e idiosincratici,
identificazione block-diagonale, Monte Carlo di recovery, pipeline di nowcasting
real-time) — è il **contributo principale**. Le voci qui sotto sono direzioni
future, non lacune del lavoro presente.

Il valore di questo documento sta nei **nodi**: per ogni estensione registro non
solo *cosa* fare, ma soprattutto *perché è difficile* e *dove sta il rischio*.

---

## 1. Errori idiosincratici AR(1) con code grasse (Student-t)

**Cosa.** Rendere gli idiosincratici **persistenti** invece che white noise:

```
eps_{i,t} = rho_i * eps_{i,t-1} + u_{i,t}
```

con persistenza sul *livello* (`rho_i`) e code grasse sull'*innovazione* `u_{i,t}`
(**Via A**), coerente con la rappresentazione scale-mixture Student-t già usata
sui fattori e sugli idiosincratici contemporanei. Il modello attuale è annidato
come caso particolare `rho_i = 0` per ogni `i`.

**Motivazione.** Cattura la **persistenza** degli shock idiosincratici — anomalie
series-specific che *durano* nel tempo — distinta dagli **outlier transitori**,
già gestiti dallo Student-t. Utile su serie in cui la dinamica specifica (non
spiegata dai fattori comuni) ha memoria propria.

**Nodo / costo.**
- Espande lo state-space da `5r` a `5r + M`: gli `M` idiosincratici persistenti
  entrano nello stato latente (companion form allargata).
- Riscrittura di **E-step e M-step** per il blocco `eps`: nuove sufficient
  statistics, nuovo update di `rho_i` (oltre a `R`), propagazione dei pesi
  Student-t sul blocco idiosincratico autoregressivo.
- **Monte Carlo di recovery da rifare** da capo (nuovo θ\*, nuovo simulatore per
  il blocco persistente).
- **Riapre l'identificazione**: persistenza *comune* (fattori) vs persistenza
  *series-specific* (idiosincratici). Con idiosincratici autoregressivi, parte
  della dinamica può essere attribuita all'uno o all'altro canale: va vincolato.
- Ordine di grandezza: **settimane**.

**Riferimenti.** Bańbura–Modugno (2014); Cascaldi-Garcia. Esiste già un paragrafo
"Serial uncorrelation" nel `.tex` (assunzione che questa estensione rilasserebbe),
più l'eventuale sezione metodologica estesa in scrittura.

---

## 2. Struttura 3+1 fattori (gerarchica, alla NY Fed)

**Cosa.** Un fattore **GLOBALE** su cui caricano **tutte** le serie, più **3
fattori di categoria** (reale / finanziario / nominale) su cui carica solo il
rispettivo blocco. Ogni serie carica quindi su **due** fattori: globale + sua
categoria. **Non è block-diagonal**: la `Lambda` ha una colonna piena (globale) +
i blocchi di categoria.

**Motivazione.** Avvicinarsi alla struttura tipo NY Fed / Giannone; far dipendere
il GDP da un **ciclo comune** oltre che dal solo fattore reale.

**Nodo PRINCIPALE — IDENTIFICAZIONE.** È qui che sta tutto il problema, ed è di
natura **teorica**, non di codice.
- Rompe l'identificazione block-diagonale ("What Block Restrictions Identify").
- Le colonne di **categoria** restano vincolate (possono solo riscalarsi /
  cambiare segno: per restare zero sulle righe degli altri blocchi i coefficienti
  di mescolamento devono annullarsi). Ma la **colonna globale non ha vincoli di
  zero** (carica su tutto): qualsiasi `g_new = g + β_R c_R + β_F c_F + β_X c_X`
  preserva il pattern di sparsità.
- Conseguenza: lo split **globale-vs-categoria NON è identificato**. Non più
  un'indeterminazione *finita* di segno/scala (gruppo `{±1}^r × R^r_+`, gestita da
  `normalize_signs` + `apply_convention_1`), ma un'indeterminazione **continua a
  3 parametri** (i `β`). E poiché `A` e `Q` sono *piene/non vincolate*, il VAR non
  impone alcun vincolo identificante: la rotazione `(Λ B, B⁻¹ f)` resta ammissibile.
- Servono **restrizioni aggiuntive** (es. ortogonalità dei loadings globali
  rispetto ai loadings di categoria entro blocco, oppure ancoraggi) — letteratura
  DFM gerarchica: **Kose–Otrok–Whiteman, Moench–Ng–Potter**.
- Implica una **M-step vincolata** (la separazione globale/categoria richiede una
  normalizzazione *tra righe*, non più riga-per-riga indipendente), la riscrittura
  di `normalize_signs` / `apply_convention_1`, e soprattutto la **RI-DERIVAZIONE
  della teoria di identificazione nella tesi**.

**Costo.**
- **Plumbing contenuto**: ogni riga "caricata" diventa una OLS pesata 2×2 (i
  momenti incrociati `E[f^g_t f^k_t | Y]` sono già disponibili negli off-diagonal
  di `P_smooth`); lo state-space è quasi gratis perché il core è **già generico in
  r** (`5r`: 15 → 20, costo Kalman ~2.4× per step, trascurabile). ~1 giorno.
- **Identificazione**: ~**2–3 settimane** di lavoro **teorico** + recovery Monte
  Carlo (che è bloccata dall'identificazione — vedi sotto).
- **Il rischio e il lavoro vero sono nell'identificazione, non nel codice.**

**Caveat empirico.** Nel prototipo del Second Stage **dominava il fattore reale**:
il guadagno di uno split globale/categoria sulla **coda del GDP** è tutto da
dimostrare. Da valutare se valga rispetto al restare block-diagonal o al "1
fattore globale" (estensione economica, ~1 giorno, block-diagonale a 1 blocco,
identificata di segno/scala).

**Nota recovery MC.** Senza la restrizione identificante, confrontare `Λ_stimata`
vs `Λ_vera` è **mal posto**: lo stimatore recupera una qualsiasi rotazione dello
split globale/categoria, quindi gli RMSE-vs-verità sono privi di senso finché non
si applica la *stessa* normalizzazione a verità e stima. La recovery MC è quindi
**necessaria e subordinata** alla soluzione del nodo identificazione.

---

## 3. Config-aware sulla struttura dei fattori (loading mask)

**Cosa.** Guidare il codice da una **"loading mask"** — matrice serie × fattori di
0/1 — specificata da JSON, così che struttura dei fattori (block-diagonal,
1-fattore, 3+1, ecc.), numero di serie, quali serie monthly/quarterly e numero di
fattori siano **tutti configurabili senza cablare nulla nel codice**. Obiettivo:
un framework **riutilizzabile** su altri dataset / paesi scrivendo solo un JSON.

**Stato attuale.**
- Il **core** (Kalman / E-step / M-step: `build_*_tilde`, `compute_weighted_moments`,
  `update_A_Q`, `update_R`, `update_nu`) è **già generico in r** (numero di fattori,
  letto dalle shape).
- La **struttura** (block-diagonal, 3 nomi `real/financial/other`) è cablata in
  **~3 punti**: `update_Lambda` (ogni riga → 1 colonna), `pca_initialization` /
  `compute_theta_initial` (PCA blocco-per-blocco, Lambda block-diagonale), 
  `normalize_signs` (un fattore per blocco) — più la costante `_BLOCK_ORDER` /
  `_BLOCK_TO_COL` **triplicata** in `em_m_step.py`, `em_main.py`,
  `em_initialization.py`.

**Costo plumbing.** ~**3–5 giorni**, **rischio BASSO**. `update_Lambda` diventa,
per riga, una OLS pesata multivariata sul sottoinsieme di colonne dove
`mask[i,:]==1` (i cross-moment servono già esistono in `P_smooth`); l'init
inizializza ogni fattore da PCA sull'unione delle serie che lo caricano. Block-
diagonal, 1-fattore e 3+1 diventano tutti casi particolari della maschera.
**Test di regressione fortissimo**: la config "small" deve riprodurre i risultati
attuali **BIT-FOR-BIT** → blinda il refactoring (stesso criterio già usato per il
dataset).

**DISTINZIONE CRUCIALE (da tenere ben presente).** Il config-aware vale per il
**PLUMBING** (struttura dei loadings, serie, frequenze, numero di fattori) **ma
NON per l'IDENTIFICAZIONE in generale.**
- Una mask **block-diagonale** è identificata (segno/scala).
- Una mask con **sovrapposizioni** (es. 3+1) in generale **NO** → serve teoria
  specifica caso per caso.
- Quindi: si può avere un sistema config-aware che fa **GIRARE** qualsiasi
  struttura, ma l'**identificazione** resta **specifica della struttura** e **non
  è generalizzabile via config**.
- In una frase: **il plumbing è ingegneria; l'identificazione è ricerca.**

**Relazione con la voce 2.** La loading mask rende **banale la plumbing** di 3+1
(la maschera `[globale tutto-1 | blocchi di categoria]`), lasciando **intatto** il
nodo identificazione. Quindi: `3+1 = maschera (gratis con la generalizzazione) +
identificazione (~2–3 settimane, il vero costo)`. Conviene fare **prima** la mask
(utile di per sé: rimuove l'hardcoding, basso rischio, test bit-exact), ma senza
illudersi che renda 3+1 "trivial".

---

## 4. Densità predittiva e Growth-at-Risk: oltre la QR "appiccicata"

Questa voce raccoglie le strade per la **coda del GDP** (Growth-at-Risk), cioè per
una densità predittiva che si **allarghi nei crolli**. È il filo conduttore di una
serie di test diagnostici già eseguiti (script in `diagnostics/`): il prototipo di
Second Stage, la diagnosi del perché domina il fattore reale, la feasibility della
density nowcasting dal DFM, il test oracolo del peso `w` e il test del `w`
forecastato. Da qui partono tre direzioni future, in ordine crescente di
ambizione e di rottura col framework attuale.

### 4.0 Baseline attuale — Quantile Regression sul fattore reale (lo stato dell'arte del progetto)

**Cosa.** Secondo stadio à la Adrian–Boyarchenko–Giannone: regressione quantile del
GDP sui fattori smoothed del DFM (in pratica **domina il solo fattore reale**),
`Q_τ(GDP_t) = β_0(τ) + β_1(τ) f^R_t`. Prototipo già funzionante end-to-end.

**Cosa regge e cosa no.**
- **Regge in full-sample**: la QR mappa il fattore reale estremo attraverso una
  pendenza di coda ripida e **traccia i crolli** — anche il 2020Q2 (q05 ≈ −8.2 =
  realizzato). È il meccanismo che oggi cattura la coda, là dove la densità
  generata *forward* dal DFM no (scale regime-invariante, std ≈ 0.63 ovunque).
- **MA**: (i) è un modello **appiccicato dopo il DFM** — una regressione separata
  sopra i fattori, non una densità coerente generata dal modello; (ii) la cattura
  del 2020 è in buona parte **in-sample / look-ahead** (usa il valore realizzato e
  smoothed del fattore); in real-time onesto la QR sbatte contro **lo stesso muro
  del 2020** (il fattore stesso è compresso/robustizzato dallo Student-t); (iii)
  metodologicamente è **poco originale e poco elegante** — è la ricetta GaR
  standard riapplicata.

**Verdetto.** Buona come *benchmark* e come pragmatico "first cut" del tail risk,
ma non è il punto di arrivo ambizioso della tesi. Le tre voci sotto sono i modi di
sostituirla con qualcosa di **generativo e coerente**.

### 4.1 DFM con Stochastic Volatility (DFM-SV)

**Cosa (nucleo comune a tutte le varianti).** Rendere lo **scale** degli shock
**persistente** invece che iid. Oggi il modello Student-t è uno scale-mixture con
peso `w_t ~ Gamma(ν/2, 2/ν)` **iid** (la vol di domani è scorrelata da quella di
oggi → il modello non sa *prevedere* un regime turbolento, lo vede solo a
posteriori). La SV dà **memoria** alla vol tramite una log-vol latente AR(1):

```
log h_t = μ + φ (log h_{t-1} − μ) + η_t          (lo shock usa varianza h_t)
```

Il modello attuale è il caso `φ = 0`. Il `w_t` latente che già stimi **è** una vol:
la SV è la sua versione con dinamica (`h_t ≈ 1/w_t` reso persistente).

**Cosa comprano i test (vale per tutte le varianti).**
- **Test oracolo** (`diagnostics/density_nowcast_feasibility.py`, sez. "TEST
  ORACOLO"): forzando lo scale al `w` *smoothed*, la densità si allarga e cattura i
  crolli (2020Q2 PIT 3e-5 → 0.007). Lo scale regime-dipendente **ha il range** (ma
  è informato dal realizzato → upper bound).
- **Test del `w` forecastato** (`diagnostics/vol_forecastability.py`): la vol è
  forecastabile **solo dove c'è il driver finanziario**. Cattura il **2008-09**
  (2009Q1 `w` onesto ≈ oracolo), **non** il **2020** reale (resta fuori sotto ogni
  forecast onesto). → la SV **non risolve il 2020 ex-ante** (non-forecastabile), ma
  allarga *prima* le crisi finanziarie e dà una densità **a una fase**, generativa,
  che **sostituisce la QR appiccicata** della 4.0.

Su questo nucleo ci sono **due scelte di design ortogonali**, che enumero per
intero perché vanno valutate separatamente: **(D1)** che fine fa lo Student-t; e
**(D2)** dove vive la vol tempo-variante.

---

#### D1 — Meccanismo delle code grasse: la SV sostituisce il t, o convivono?

Code grasse = eccesso di curtosi. La SV e lo Student-t lo producono in **due modi
diversi**: la SV con **clustering** (vol che dura), il t con **outlier iid** (salti
isolati). La scelta è *quale* meccanismo tieni.

- **Opzione D1-A — SV pura (la vol persistente SOSTITUISCE il t).**
  Lo shock è gaussiano-condizionato-alla-vol; tutta la curtosi nasce dal clustering.
  - *Pro*: massimamente **pulito e parsimonioso** — un solo meccanismo di
    volatilità, nidifica il modello attuale a `φ=0` *e* `ν→∞`. Niente parametro `ν`.
  - *Contro*: assume che **ogni** picco sia vol persistente. Un crollo-rimbalzo di
    **un solo trimestre** (2020 Q2↓/Q3↑) è un **salto**, non un regime: una log-vol
    liscia AR(1) lo insegue male (alza la vol e la tiene su per persistenza →
    *smearing* sui trimestri vicini, e fatica a raggiungere il `−8.20`).
  - *Identificazione*: regge (scale identificato di segno/scala); rischio solo di
    stima.
  - *Verdetto dai test*: è la variante che **storicamente rompe sul Covid** (è il
    motivo per cui la letteratura post-2020 ha rimesso le code grasse, vedi sotto).

- **Opzione D1-B — SV + Student-t (vol persistente + code iid).**
  Tieni `h_t` persistente **e** un residuo Student-t (`ν` da stimare, tipicamente
  alto) sopra la log-vol.
  - *Pro*: la SV cattura il *clima*, il t assorbe gli **outlier one-off** che la SV
    liscia non prende (il 2020). È la combinazione **"vol + jump"** classica della
    finanza econometrica — interpretabile, non un'accozzaglia.
  - *Contro*: **due meccanismi** → meno pulito; un parametro in più.
  - *Identificazione (il vero nodo di D1-B)*: separare "vol persistente" (`φ, σ_η`)
    da "outlier iid" (`ν`). Possono scambiarsi lavoro. Si vincola con **prior
    informativi sulla persistenza** e `ν` non troppo basso.
  - *Verdetto dai test*: copre **sia** i build-up (2008, via SV) **sia** il salto
    idiosincratico del 2020 (via t).

- **Opzione D1-C — SV + componente outlier ESPLICITA (outlier-adjusted SV).**
  Variante "più chirurgica" di D1-B: invece del t-continuo, un moltiplicatore di
  scala `o_t` che vale 1 quasi sempre e occasionalmente esplode (es. `o_t` con prior
  a coda grassa, o un jump bernoulliano con probabilità `p` e size propria).
  - *Pro*: **separazione più netta** vol-vs-outlier rispetto al t (l'outlier è un
    evento *raro e datato*, con probabilità di occorrenza propria → identificazione
    più pulita di D1-B). È **esattamente** lo strumento con cui Carriero–Clark–
    Marcellino–Mertens (2022) gestiscono il Covid senza far esplodere la SV.
  - *Contro*: un blocco in più (indicatori di outlier come variabili latenti
    aggiuntive nel sampler); leggermente più codice di D1-B.
  - *Identificazione*: la **migliore** delle tre per il caso-Covid (l'outlier è
    timato, non confuso con la persistenza).
  - *Verdetto dai test*: è la risposta "stato dell'arte" al fatto che il 2020 nei
    nostri test è un **singolo trimestre** idiosincratico ed estremo.

---

#### D2 — Dove vive la SV: solo sul fattore comune, o anche sull'idiosincratico?

- **Opzione D2-A — solo SV comune sul fattore (1 `h_t` scalare).**
  Un unico processo di vol che scala le innovazioni del fattore: il "clima di
  volatilità" macro (common-SV à la CCM; coincide con una misura di incertezza
  macro à la JLN).
  - *Pro*: **massima parsimonia ed eleganza** — un solo AR(1) latente,
    interpretabile, un solo parametro di persistenza.
  - *Contro DECISIVO (dai test)*: nel **2020 il crollo del GDP passava per
    l'IDIOSINCRATICO** (`w_eps ≈ 0` sul GDP Apr–Giu), **non** per il fattore comune
    (che lo Student-t comprimeva). Anche conoscendo i fattori smoothed (V1, full-
    info) il 2020 falliva. → una SV **solo comune** *struttralmente non cattura il
    tuo episodio-vetrina*. Va bene per il 2008 (comune), non per il 2020.
  - *Costo*: il più basso (1 processo latente).

- **Opzione D2-B — comune + idiosincratica su TUTTE le serie.**
  Un `h_t` comune **più** una log-vol idiosincratica per ciascuna delle M serie.
  - *Pro*: il più **completo** — cattura vol series-specific ovunque; è lo standard
    nelle BVAR large-scale (CCM).
  - *Contro*: **M+1 processi latenti** → molti parametri, meno parsimonioso, costo
    di stima alto; meno interpretabile per il *solo* tail del GDP.
  - *Identificazione*: separare vol **comune** vs **idiosincratica** (analoga al
    problema di scala dei loadings; gestibile ma da vincolare).

- **Opzione D2-C — comune + idiosincratica SOLO sul GDP (o sul blocco reale).**
  Via di mezzo mirata: `h_t` comune + una sola log-vol idiosincratica sull'equazione
  del GDP (eventualmente estesa al blocco reale).
  - *Pro*: **parsimonioso** ma centra il canale che conta — il 2020 del GDP era
    idiosincratico, quindi mette la vol tempo-variante *esattamente dove il tuo
    caso-limite la richiede*. GDP è la variabile-target → giustificato.
  - *Contro*: leggermente **ad hoc** ("perché solo il GDP?"); difendibile proprio
    perché GDP è l'oggetto della tesi (Growth-at-Risk).
  - *Costo*: medio-basso (2 processi latenti).

---

#### Matrice di design (D1 × D2) e raccomandazione

Le due scelte sono **indipendenti**: 3 × 3 = 9 combinazioni. Le celle agli estremi:

| | **D2-A** comune | **D2-B** comune+tutti idio | **D2-C** comune+idio GDP |
|---|---|---|---|
| **D1-A** SV pura | massima eleganza, **manca il 2020** | completo ma pesante, **smearing 2020** | pulito, **smearing 2020** |
| **D1-B** SV+t | manca canale idio 2020 | molto robusto, **pesante** | **buon compromesso** |
| **D1-C** SV+outlier | manca canale idio 2020 | stato dell'arte, **pesante** | ⭐ **consigliata** |

**Raccomandazione (da rivedere insieme):** **D1-C × D2-C** — SV comune sul fattore
(il clima macro / parte forecastabile finanziaria) **+** una log-vol idiosincratica
sul GDP con **componente outlier esplicita** per il salto 2020. Mette ogni
meccanismo dove è economicamente giusto: **SV dove la vol dura (comune), outlier
datato dove lo shock è un one-off (idiosincratico GDP)**. Copre 2008 *e* 2020,
resta parsimonioso, racconta una storia interpretabile. **D1-B × D2-C** è la
variante "più semplice da scrivere" (t continuo invece dell'outlier esplicito), a
costo di un'identificazione vol-vs-jump un po' meno netta. **D1-A × D2-A** è la più
elegante sulla carta ma è quella che i nostri test bocciano sul 2020 (manca sia il
canale idiosincratico sia il salto one-off).

---

#### Sistemi di equazioni (cosa cambia, esplicito)

**Modello ATTUALE (baseline).** `f~_t` = stato companion MM-aggregato, `λ_i` i
loadings, `z` rumori `N(0,1)`:

```
f_t      = A f_{t-1} + u_t,        u_t     = (1/√w^u_t)  · Q^{1/2} z^u_t
y_{i,t}  = λ_i' f~_t + e_{i,t},     e_{i,t} = √(R_i / w^e_{i,t}) · z^e_{i,t}
w^u_t    ~ Gamma(ν_u/2, 2/ν_u)  iid      ⇒  u_t     ~ t_{ν_u}(0, Q)
w^e_{i,t}~ Gamma(ν_e/2, 2/ν_e)  iid      ⇒  e_{i,t} ~ t_{ν_e}(0, R_i)
```

I pesi `w` sono **iid** (nessuna memoria): è la radice della compressione.

**Forma-madre SV (con "interruttori", copre tutte le 9 celle).** La varianza di
ogni shock = `h` (SV persistente) × `s` (coda, scelta D1). `h ≡ 1` se la SV è
spenta su quel blocco (scelta D2); `s ≡ 1` se la coda è spenta:

```
u_t      = √(h^f_t)        · s^u_t      · Q^{1/2} z^u_t
e_{i,t}  = √(h^i_t · R_i)   · s^e_{i,t}  · z^e_{i,t}
log h^•_t = μ_• + φ_•(log h^•_{t-1} − μ_•) + η^•_t ,   η^•_t ~ N(0, σ²_•)
```

- **D2 sceglie quali `h` sono attivi:**
  - D2-A: `h^f` attivo; `h^i ≡ 1` ∀i.
  - D2-B: `h^f` attivo; `h^i` attivo ∀i.
  - D2-C: `h^f` attivo; `h^i` attivo **solo** per i = GDP, `≡1` altrove.
- **D1 sceglie la coda `s`:**
  - D1-A: `s ≡ 1` (gaussiano condizionato alla vol).
  - D1-B: `s^•_t = 1/√w^•_t`, `w^•_t ~ Gamma(ν/2, 2/ν)` iid (Student-t residuo).
  - D1-C: `s^•_t = 1` con prob `1−p`; `= √o^•_t` con prob `p`, `o` grande
    (outlier *datato*, salto raro).

Il modello attuale è la forma-madre con `φ_•=0, σ_•=0` (h costante) e `s=1/√w`,
`w` iid: **tutto nidificato**.

**Cella consigliata D1-C × D2-C, scritta per esteso:**

```
# Fattore comune: SV pura
f_t       = A f_{t-1} + u_t ,     u_t = √(h_t) · Q^{1/2} z^u_t ,   z^u_t ~ N(0, I_r)
log h_t   = μ + φ(log h_{t-1} − μ) + η_t ,        η_t ~ N(0, σ²_η)

# Idiosincratici i ≠ GDP: gaussiani omoschedastici (come ora, senza t)
e_{i,t}   ~ N(0, R_i)

# Idiosincratico GDP: SV (vol lenta) + outlier datato (il salto 2020)
e_{g,t}   = √(h^g_t · R_g) · o_{g,t} · z^e_{g,t} ,   z^e_{g,t} ~ N(0, 1)
log h^g_t = μ_g + φ_g(log h^g_{t-1} − μ_g) + η^g_t
o_{g,t}   = 1                con prob 1 − p          # quasi sempre
o_{g,t}   ~ coda-grassa (≥1) con prob p              # raro, es. 2020

y_{i,t}   = λ_i' f~_t + e_{i,t}
```

**Cella più semplice D1-B × D2-C** (t continuo invece dell'outlier datato): identica,
ma sul GDP

```
e_{g,t}   = √(h^g_t · R_g / w_{g,t}) · z^e_{g,t} ,   w_{g,t} ~ Gamma(ν_g/2, 2/ν_g) iid
```

(il vecchio `update_nu` sopravvive qui; in D1-C è sostituito dal passo sull'outlier).

#### Devo buttare via l'EM? (cosa sopravvive, cosa cambia)

**No, non tutto — e soprattutto NON il pezzo difficile.** Butti via l'EM come
*algoritmo esterno* e gli update in forma chiusa della vol; **tieni** tutta la
rappresentazione state-space e il Kalman.

Il punto che rende tutto modulare: **condizionatamente al percorso di volatilità
`h_{1:T}` (e agli `s_t`), il modello torna lineare-gaussiano**, solo con varianze
**tempo-varianti** `Q_t = h^f_t · Q` e `R_{i,t} = h^i_t s²_{i,t} · R_i`. Il
Kalman/smoother gira identico, con matrici che cambiano nel tempo.

**Sopravvive (riuso quasi diretto):**
- `build_A_tilde / build_Q_tilde / build_Lambda_tilde / build_R_tilde` + companion
  MM → **invariati** (basta passare `Q_t, R_t` tempo-varianti).
- Kalman filter + smoother (`kalman.py`) → **riuso**; aggiungi il **simulation
  smoother** (Durbin–Koopman) per *campionare* gli stati, non solo lisciarli.
- gli update weighted-OLS di Λ, A, Q, R (`em_m_step.py`) → **sopravvivono
  reinterpretati** come draw condizionali del Gibbs (posterior coniugata: Normale
  per Λ/A, Inverse-Gamma/Wishart per le scale `R, Q`). Stessa algebra dei momenti
  pesati, con `h_t` al posto dei pesi `w_t`.

**Cambia / si aggiunge:**
- l'**outer loop EM → Gibbs/MCMC** (`em_main.py`): non più E/M a convergenza, ma un
  sampler che cicla i blocchi.
- nuovi blocchi sampler: **Kim–Shephard–Chib** (mixture-of-normals) per `h | stati`;
  passi per `(μ, φ, σ_η)`; in D1-B il passo `w | ·` (resta Gamma → `update_nu`
  sopravvive); in D1-C i passi su `indicatori/size di outlier | ·`.
- il **Monte Carlo di recovery** va rifatto.

In una riga: **conservi "modello + Kalman" (la metà difficile, già costruita) e
riscrivi "il motore di stima" (EM → MCMC).** Il lavoro nuovo è il sampler della vol,
non lo state-space.

*Alternativa per restare in EM:* **MCEM / particle-EM** (tieni l'EM ma l'E-step su
`h` diventa Monte Carlo). Più vicino al codice attuale, ma più fragile e meno
standard: per la SV la via bayesiana è lo standard ed è più pulita.

---

**Nodo / costo (comune a tutte le varianti).**
- **Motore di stima**: EM → MCMC (Gibbs), vedi sotto "Devo buttare via l'EM?". Lo
  state-space e il Kalman si riusano; il lavoro nuovo è il sampler della vol.
- **Monte Carlo di recovery da rifare** (nuovo θ\*, nuovo simulatore con la dinamica
  della vol — e, per D1-B/C, verificare di recuperare *separatamente* persistenza e
  outlier; per D2-B/C, comune vs idiosincratica).
- **Identificazione**: lo scale è identificato; i nodi veri sono **interni a D1**
  (vol-vs-jump, peggiore in D1-B, migliore in D1-C) e **interni a D2** (comune-vs-
  idiosincratica in D2-B).
- Ordine di grandezza: **settimane** (D1-C × D2-C: alto ma gestibile; D2-B alza il
  costo per il numero di processi latenti).

**Driver osservabile (ortogonale, opzionale) — SV-X.** Sia D1 sia D2 ammettono un
**add-on**: far dipendere la log-vol *comune* da NFCI laggato,
`log h_t = μ + φ(log h_{t-1}−μ) + γ·NFCI_{t-1} + η_t`. Sfrutta il segnale finanziario
anticipatore che il test del `w` forecastato ha documentato (migliora l'anticipo
sulle crisi finanziarie). **Non serve per l'eleganza** — è una sezione robustezza,
e non aiuta comunque il 2020 (NFCI non esplose). Senza il termine `γ`, la SV è
**puramente latente** (nessuna variabile esterna): prevede la vol solo per
persistenza.

**Riferimenti.** Carriero–Clark–Marcellino ("common stochastic volatility");
Carriero–Clark–Marcellino–Mertens (2022, *outlier-adjusted SV* per il Covid);
Kim–Shephard–Chib (mixture sampler per SV); Durbin–Koopman (simulation smoother);
Mumtaz–Surico; Jurado–Ludvigson–Ng (macro uncertainty); Adrian–Boyarchenko–Giannone
(GaR).

### 4.2 DFM con Markov-Switching (regime detection)

**Cosa.** La versione **discreta** della 4.1: lo scale (e/o la dinamica `A`, `Q`)
governato da una **catena di Markov a 2-3 stati** — *calmo* / *crisi* — con il
filtro di **Hamilton** sopra il Kalman che hai già. Il modello attuale è il caso "1
solo stato".

**Motivazione.** È forse **più naturale** della SV continua per la **dicotomia
finanziaria/reale** che i test hanno documentato: gli stati "calmo/crisi" sono
interpretabili, e la probabilità di stato-crisi può essere **guidata da NFCI**
(transizione tempo-variante) — esattamente il canale che il test del `w`
forecastato dice essere forecastabile. La *regime detection* (probabilità filtrata
di crisi) è un output di per sé interessante e diagnostico: scatterebbe *dopo*
l'onset del 2020 (coerente con la non-forecastabilità), *prima* nel 2008.

**Nodo / costo.**
- Più **gestibile** della SV: il filtro di Hamilton è uno strato sopra il Kalman,
  niente MCMC obbligatorio; l'EM diventa un EM con probabilità di regime come
  ulteriori variabili latenti (Kim–Nelson).
- **Identificato** (gli stati sono ordinabili per varianza, label-switching
  risolvibile con un vincolo `σ_crisi > σ_calmo`).
- **Monte Carlo di recovery da rifare**.
- Ordine di grandezza: **settimane**, **rischio medio** (più basso della SV).

**Riferimenti.** Hamilton (1989); Kim–Nelson (state-space with regime switching);
Chauvet (DFM con Markov-switching per il business cycle).

**Raccomandazione tra 4.1 e 4.2.** Per *questa* tesi, il **Markov-switching
guidato da NFCI** è il candidato migliore: più economico da stimare, identificato,
e calzante sulla distinzione finanziaria/reale già documentata. La SV continua è
più liscia ed elegante ma con costo di stima superiore.

### 4.3 SOLUZIONE FUTURA — Quantile Dynamic Factor Model (QDFM)

**Idea.** I **loadings dipendono dal quantile**: la struttura del ciclo cambia
lungo la distribuzione (effetti diversi in crisi vs boom).

```
Q_τ(y_it | f_t) = λ_i(τ)' f_t
```

**Significato.** Non più "un fattore + una coda mappata sopra", ma un modello in cui
la **struttura fattoriale stessa è quantile-specifica**: spillover asimmetrici,
rischio strutturale **endogeno**, fattori diversi per ogni quantile.

**Nodo PRINCIPALE — rompe l'intero framework attuale.** Qui non si estende il
modello: lo si **cambia**.
- ❌ niente Kalman filter
- ❌ niente EM standard
- ❌ niente likelihood gaussiana

**Tecniche necessarie.**
- **Asymmetric Laplace (ALD)** come pseudo-likelihood per i quantili (con la sua
  rappresentazione come location-scale mixture, che riapre una struttura
  trattabile).
- **Inferenza bayesiana**: Gibbs sampling / MCMC / metodi particellari sui fattori
  latenti quantile-specifici.
- **Variational inference** (più scalabile): ELBO sull'ALD, approssimazione della
  posterior.

**Problema chiave — esplosione di parametri.** `λ_i(τ)` per ogni serie e ogni
quantile → servono vincoli forti: **shrinkage**, **spline su τ** (loadings lisci nel
quantile), **struttura low-rank**. Senza questi è non identificato / sovra-
parametrizzato.

**Output.** Fattori diversi per ogni quantile; spillover asimmetrici; rischio
strutturale endogeno (la coda non è mappata *sopra* il modello, **è** il modello).

**In sintesi.**
- ✔ molto **innovativo** — materiale da **paper nuovo**, non da semplice estensione
- ✔ risponde alla critica "QR appiccicata" in modo radicale
- ❌ **complesso**, **richiede una stima completamente nuova** (addio Kalman/EM
  gaussiano)
- ❌ ordine di grandezza: **mesi**, **rischio alto** (sia teorico sia di stima)

**Posizionamento.** È la direzione di **ricerca pura**: oltre la tesi corrente, come
progetto di dottorato / paper a sé. Le voci 4.1–4.2 sono il percorso *incrementale*
(generalizzano ciò che hai); la 4.3 è il *salto*.
