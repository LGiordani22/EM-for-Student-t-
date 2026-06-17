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
