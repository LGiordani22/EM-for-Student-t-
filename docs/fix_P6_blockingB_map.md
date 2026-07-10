# Fix P6 — piano B: la cresta `ρ ↔ path ↔ σ_η` (mappa, sola lettura)

**Perché questo documento esiste.** Il GATE 3 ha dato il suo verdetto: il griddy-Gibbs
su `ρ` **elimina la random-walk e non muove l'ESS**. Il collo di bottiglia non era la
mossa di `ρ`; è il **blocking** del Gibbs. Il criterio del PASSO 3 prescrive, in questo
ramo, di **fermarsi** e mappare il piano B senza implementarlo. È ciò che segue.

**Nessun sorgente è stato modificato per produrre questa mappa.**

---

## 1. Il fatto, e cosa dimostra

| | `ESS/draw` `k=0` | `k=1` | `k=2` | `R̂(ρ_1)` | wall-clock | **ESS/secondo** |
|---|---|---|---|---|---|---|
| RW-Metropolis | 1.25% | 0.35% | 1.17% | 1.156 | 1632 s | **0.038** |
| griddy | 1.17% | 0.32% | 1.39% | 1.135 | 3170 s | 0.018 |

Il griddy fa esattamente ciò per cui è stato scritto — verificato in `test_passo4` [7]:

- campiona dallo **stesso target** dell'RW (media e sd del posterior coincidono a tre
  decimali su `ρ_vero ∈ {−0.70, −0.15, +0.45, 0}` e `n_lev ∈ {60, 600}`);
- è **indipendente dal valore corrente**: partendo da `ρ = −0.99` o da `ρ = +0.99`, con
  lo stesso seed, restituisce lo **stesso identico draw**.

Cioè: **lungo `ρ`, condizionatamente al resto, l'autocorrelazione è ora zero.** E l'ESS
non si muove.

**Conclusione per esclusione.** L'autocorrelazione di `ρ` non entra dalla sua mossa.
Entra dalla catena `ρ → path → σ²_η → ρ`. Ogni sweep `ρ` è estratto *dato* un path che è
stato estratto *dato* il `ρ` precedente. Perfezionare il draw di `ρ` dato il path è
inutile se è il path a muoversi lentamente nella direzione che conta. **La cresta è
reale**, e questo è il primo esperimento che lo dimostra invece di ipotizzarlo.

Costo aggiuntivo: il griddy costa `2×` in wall-clock, quindi in **ESS al secondo è due
volte peggiore dell'RW**. Rilevante per la scelta del default (§6).

---

## 2. Cosa dice il `.tex`, e cosa non abbiamo mai testato

`subsec:asis-leverage` (21892–21979) è già lucido sulla diagnosi. Osserva che il
leverage **stringe** la cresta che ASIS esiste per allentare, perché `σ_η` è legato al
path in **due** punti anziché uno: attraverso gli incrementi AR(1) *e* attraverso il
drift `ρ σ_η z_t` (`eq:asis-cp-lev`). *"The block that mixed worst in the base case
mixes worse still under leverage."*

E chiude con una previsione precisa (21971–79):

> *"Both updates condition on the current leverage `ρ`, which is drawn in its own
> Family~C Metropolis block and is **not** itself interwoven. But `ρ` benefits
> nonetheless, and substantially: it is strongly posterior-correlated with `σ_η` ...
> so a better-mixing `σ_η` unlocks a better-mixing `ρ`."*

⚠️ **Quella previsione non è mai stata verificata.** Tutti i run di P6 — baseline, GATE 3,
gli esperimenti dell'audit — girano con `use_asis=False`. Se il `.tex` ha ragione,
`ESS(ρ)` dovrebbe salire **senza scrivere una riga**, semplicemente accendendo ASIS.
È l'esperimento zero, e va fatto prima di qualunque derivazione nuova.

*(L'unico dato che abbiamo è indiretto e contrario: nell'audit, ASIS sotto B a `T=600`
**peggiorava** il canale debole, `φ̂ → −0.11`. Ma quella misura guardava `corr(h)` e `φ̂`,
non `ESS(ρ)`, e con l'RW su `ρ`. Non decide nulla.)*

---

## 3. Le opzioni, dalla più leggera alla più invasiva

### Opzione 0 — accendere ASIS e misurare `ESS(ρ)` — **zero righe**

`fit_dfm_mcmc(..., use_asis=True, rho_sampler=...)` è già cablato su Branch B
(Phase 6/7). Il bench `bench_p6_rho.py` accetta `**fit_kwargs`, quindi il confronto è
un flag.

- **Costo:** due run del bench (~1 h).
- **Cosa decide:** se la previsione del `.tex` regge. Se `ESS(ρ)` sale a `>3%`, P6 è
  chiuso e non serve altro.
- **Rischio:** ASIS forza `sigma_prior="half_normal"`, quindi cambia il prior su `σ_η`
  rispetto al baseline. Il confronto va fatto **a prior uguale** (half-Normal in
  entrambi i bracci), altrimenti si misurano due cose insieme.

**Questa va fatta per prima, sempre.** Se funziona, le opzioni sotto non servono.

---

### Opzione 1 — riparametrizzare `(σ_η, ρ) → (a, b)`, il draw congiunto coniugato ⭐

**È la mia raccomandazione**, se l'Opzione 0 fallisce. Non è "interweaving su `ρ`": è più
semplice, ed è *esatta*.

Scrivi la transizione CP con leverage nella sua forma decomposta — che il `.tex` **già
usa** (`eq:asis-cp-lev`):

```
x_t = φ x_{t-1} + ρ σ_η z_t + σ_η √(1−ρ²) ε_t ,      ε_t ~ N(0,1)
```

e cambia coordinate:

```
a ≡ ρ σ_η        (coefficiente di drift)
b ≡ σ_η √(1−ρ²)  (scala dell'innovazione indipendente)
```

Allora, **dato il path e dato `z` congelato** (la stessa presa di posizione separabile che
il `.tex` adotta già per `σ_η`, 21950–56), gli incrementi `η_t = x_t − φ x_{t-1}`
obbediscono a

```
η_t | a, b, z  ~  N(a z_t , b²)
```

che è una **regressione lineare gaussiana** con coefficiente `a` e varianza `b²`. Il suo
posterior coniugato è una **Normal-Inverse-Gamma**: `(a, b²)` si estrae **in forma
chiusa, congiuntamente, in un colpo solo**. Poi si torna indietro:

```
σ_η = √(a² + b²)          ρ = a / √(a² + b²)
```

Tre proprietà che rendono questa strada attraente:

1. **La cresta `ρ ↔ σ_η` scompare per costruzione.** `ρ` e `σ_η` sono *entrambi* funzioni
   di `(a,b)`, e `(a,b)` sono le coordinate naturali della regressione: il loro posterior
   congiunto è quasi ortogonale dove `(ρ, σ_η)` è una banana. Non si mescola meglio la
   cresta: **la si elimina cambiando carta**.
2. **Il vincolo `|ρ| < 1` è automatico.** `ρ = a/√(a²+b²) ∈ (−1,1)` sempre, e
   `b² = σ²(1−ρ²) ≥ 0` sempre. Spariscono il rigetto ai bordi e la regione ammissibile.
3. **Sostituisce due mosse Metropolis con un draw coniugato.** Oggi `σ²_η` è un
   RW-Metropolis (`_draw_sigma2_lev`, non coniugato per via del `(1−ρ²)` e del drift) e
   `ρ` è un altro Metropolis (o il griddy). Con `(a,b)` **entrambe** diventano un'unica
   NIG esatta.

**Cosa va derivato nel `.tex`** (è teoria nuova, non solo codice):

- il **Jacobiano** della mappa `(a,b) → (ρ, σ_η)`, per esprimere quale prior su `(a,b)`
  induce il prior che il `.tex` adotta su `(ρ, σ_η)` — Uniform(−1,1) su `ρ`
  (riga 21542) e half-Normal su `σ_η` (21503). La NIG coniugata su `(a,b²)` **non**
  induce esattamente quella coppia: va scelto se (i) adottare la NIG e documentare il
  prior implicito su `(ρ,σ_η)`, oppure (ii) tenere il prior voluto e correggere con un
  passo Metropolis-Hastings leggero sul rapporto dei Jacobiani. **(i) è più pulito.**
- che il draw resta **condizionale a `z` congelato**, esattamente come già fa `σ_η`
  in ASIS-leverage. La stessa "conditionally exact" caveat, nessuna in più.
- il caso `lagged` (Branch B, la nostra config): `z_t → g_{t−1}`, il regressore di Omori.
  La struttura è identica.

**Punti d'innesto nel codice** (per il preventivo, non per implementare ora):

| dove | cosa |
|---|---|
| `sample_leverage.py:_draw_sigma2_lev` (`:281`) | rimpiazzato dal draw NIG di `(a,b²)` |
| `sample_leverage.py:draw_rho_scalar` / `draw_rho_griddy` (`:342`, `:365`) | non più chiamati sul path B (restano per A e per il confronto) |
| `sample_leverage_lagged.py:404-407` e `:455-458` | un'unica chiamata `draw_ab_nig(...)` al posto di due |
| `sample_leverage_lagged.py:370-373` | `φ` resta il suo draw (invariato) |
| `sample_asis.py` | ASIS interviene su `σ_η` **dopo** la ri-mappatura, oppure diventa superfluo su quell'asse |
| `gibbs.py` | un flag `familyBC_sampler="joint_ab" \| "separate"` |

**Costo:** ~60 righe di codice, ~1 pagina di `.tex`, un recovery test nuovo, il bench
per il confronto. **Rischio principale:** l'interazione con ASIS (che riparametrizza
`σ_η`) va pensata, non sovrapposta meccanicamente.

**Cosa NON risolve:** la cresta residua `(a,b) ↔ path`. Se dopo questo `ESS(ρ)` resta
basso, il colpevole è il path, e l'unica risposta è l'Opzione 3.

---

### Opzione 2 — interweaving esplicito su `ρ` (ASIS esteso)

Il `.tex` dice che `ρ` *"is not itself interwoven"*. Estenderlo significa trovare, per
`ρ`, una coppia di parametrizzazioni sufficiente/ancillare come `(CP, NCP)` lo è per
`σ_η`. In NCP (`eq:asis-ncp-lev`) `ρ` **resta nello stato**: `x̃_t = φ x̃_{t-1} + ρ z_t +
√(1−ρ²) ε_t`. Non migra nella misura, quindi non c'è la regressione pulita che rende
ASIS elegante su `σ_η`.

**Valutazione:** poco promettente e caro. L'Opzione 1 ottiene lo stesso effetto (rompere
la correlazione `ρ`–`σ_η`) con un cambio di coordinate invece che con un interweaving,
e in forma **chiusa**. Non la raccomando.

---

### Opzione 3 — attaccare la cresta `path ↔ (ρ, σ_η)`

Se anche dopo l'Opzione 1 l'ESS resta basso, il blocking residuo è fra il **path** e i
parametri di volatilità. Le strade note:

- **ASIS sul path** (già presente: `use_asis`, riscalatura `x̃ = x/σ_η`). È esattamente
  ciò che l'Opzione 0 testa.
- **Draw congiunto `(path, σ_η)`** in coordinate non-centrate — cioè estrarre `x̃` e poi
  `σ_η`, mai `x` e `σ_η` separatamente. È la forma forte di ASIS, e sotto Branch B
  l'FFBS può girare direttamente su `x̃` (la transizione NCP è ancora lineare-gaussiana).
- **Marginalizzare `ρ` fuori dal path draw**: non fattibile in forma chiusa.

**Costo:** alto, teoria nuova. **Da valutare solo con i numeri dell'Opzione 1 in mano.**

---

## 4. Il secondo risultato del GATE 3: `ρ_1` non è identificato

Il CI al 90% del canale debole **si allarga** col griddy: `[−0.945, +0.405]` contro
`[−0.727, +0.304]` dell'RW. Copre lo zero **ed entrambi i segni**. La sua media a catena
finita è un numero arbitrario dentro quell'intervallo: `−0.264` (RW, 5000 draw),
`−0.361` (griddy, 5000 draw), `−0.607` (griddy, 700 draw).

**Conseguenza sui test.** Tre check di `test_perfactor_leverage` falliscono ora:

```
[FAIL] Branch B: rho_k ordering recovered      rho=[-0.532 -0.607  0.501]  true=[-0.70 -0.15 0.45]
[FAIL] A/B agree on the rho ordering           A=[-0.304 -0.058  0.295]
[FAIL] A/B agree on rho within MC error        max|d| = 0.549
```

Tutti e tre dipendono dal **canale debole**. Il canale dominante (`−0.532`) e quello
positivo (`+0.501`) — che *sono* identificati (`R̂ ≈ 1.03–1.04`) — passano con entrambi i
kernel, su entrambi i branch.

⚠️ **Il fix non ha rotto quei test: li ha smascherati.** Asserivano l'ordinamento e la
parità A/B di una quantità **non identificata**; erano verdi perché l'RW, non
convergendo, restava dove il path lo portava. È lo stesso meccanismo per cui
`acceptance(ρ) = 0.50` sembrava "sana" mentre `ESS/draw` era `0.35%`.

**Correzione proposta** (non applicata — è materia del PASSO 4, che il GATE 3 ha
sospeso): asserire su `ρ` **solo sui canali identificati**, e trasformare il canale
debole in un check esplicito di **non**-identificazione (CI che copre lo zero) — la
stessa forma che il test già usa per la soglia P2 su `corr(h)`. La parità A/B va
ristretta ai canali identificati, e comunque va ripensata: ora che B mescola e A no,
confronta un point estimate convergiuto con uno che non lo è.

---

## 5. Il fatto sgradevole che nessuna opzione risolve

`ρ̂_0 = −0.513`, CI `[−0.705, −0.320]`, `R̂ = 1.043`, `ESS = 58` — e il vero è `−0.70`.
Il canale **dominante**, quello identificato e convergiuto, ha un posterior che sfiora
appena il valore vero.

Questo **non è mixing**, e nessun sampler lo cura: il griddy campiona *meglio lo stesso
posterior*, e il posterior sta lì. Le due letture restano quelle dell'audit:

1. a `T=600` il posterior di `ρ` è genuinamente concentrato lontano dal vero, perché `ρ`
   correla `η_t` e `z_t` che sono **entrambi latenti**, letti attraverso un `ĥ` rumoroso
   (un errors-in-variables che `T` non cura, come misurato: `−0.49` anche a `T=2400`);
2. oppure c'è un bias residuo nel regressore di Family C che non abbiamo trovato — ma il
   kernel è non distorto su regressore esatto, la linearizzazione di Omori ha
   `E[zg]/E[g²] = 1.0000`, e `Q` è diagonale in tutti questi run.

**È la vera domanda aperta per il forecast**, più di P6 stesso: se la skew della
predittiva è governata da un `ρ` sistematicamente attenuato del 25–30%, il GaR
sottostima il rischio di coda. Va deciso con Ciganovic, perché è una questione di
**identificazione del modello**, non di sampler.

---

## 6. Decisioni immediate che il GATE 3 impone

1. **Il default di `rho_sampler`.** Il griddy non compra ESS e costa `2×`. Ma è
   *value-independent*, non ha `prop_sd` da tarare, e prova di campionare dalla
   conditional esatta. Tre strade: (a) tenerlo default per correttezza e accettare il
   costo; (b) tornare a `"rw"` finché la cresta non è risolta; (c) tenerlo e **ridurre la
   griglia** da 401 a ~101 punti (il target è liscio su un supporto compatto: il costo
   scala linearmente e l'errore di griglia è trascurabile). **Raccomando (c)**, che
   recupera gran parte del `2×`.

2. **I tre test rossi.** Vanno corretti come in §4. Non è "aggiustare il test per farlo
   passare": è smettere di asserire una proprietà che il posterior non ha.

3. **L'Opzione 0 va eseguita prima di ogni altra cosa.** Costa due run e verifica una
   previsione esplicita del `.tex`.

---

## 7. Cosa NON fare

- ❌ **Non** implementare l'Opzione 1 prima di aver eseguito l'Opzione 0. Se ASIS basta,
  60 righe e una pagina di tesi sono sprecate.
- ❌ **Non** allargare la griglia del griddy né tarare `prop_sd`: il GATE 3 dimostra che
  la mossa di `ρ` non è il collo di bottiglia.
- ❌ **Non** interpretare `ρ̂_0 = −0.513` come un difetto del sampler. È il posterior.
- ❌ **Non** toccare Branch A.
