# `scripts/` — i tre entry point, e come si parallelizzano

La passata completa, senza opzioni da configurare, parte dalla radice:

```
python run_all.py
```

Durante la passata, un secondo terminale puo' mostrare il dashboard live
(sola lettura, refresh automatico, ETA appresa dai checkpoint):

```
python check_progress.py
```

`run_all.py` scopre le celle e i blocchi dagli entry point sotto, separa anche
i quattro modelli di ogni blocco BVAR e ne tiene fino a **224 in volo insieme**,
con un thread BLAS ciascuno — 224 *lavori*, che non sono 224 processi: il conto
vero è più sotto. Poi riunisce i quattro shard di ogni blocco e
lancia tutte le figure e tabelle. Un lavoro fallito non interrompe gli altri:
viene elencato nel riepilogo finale e il suo dettaglio resta in
`output/_logs/run_all/`.

Tre script, uno per impostazione, ciascuno lanciabile da solo.

```
python scripts/run_dfm.py --spec diag3 --variant student_t      una cella DFM
python scripts/run_bvar.py --start 2007-01-05 --end 2007-04-27   un blocco BVAR
python scripts/run_outputs.py                                    figure, metriche, tabelle
```

Ogni script ha il suo docstring in testa, con i dettagli. Qui c'è solo quello
che serve per decidere **come lanciarli in parallelo**, e in particolare i
numeri che sono stati *misurati* su questo progetto invece che stimati.

## L'unità atomica, e perché è quella

| | unità | quante | indipendenti? |
|---|---|---|---|
| DFM | la **cella** (spec × variante), più il lavoro dei **benchmark** | 15 + 1 | sì, fra loro |
| BVAR | **modello × blocco** (1 stima piena + riusi) | 4 × 77 sul 2007-2025 | sì, per costruzione |

```
python scripts/run_dfm.py --list          le 16 unità, una per riga
python scripts/run_dfm.py --benchmark     ar2 e media espandente, da soli
python scripts/run_bvar.py --list-blocks  i 77 blocchi, 'inizio fine' per riga
```

**Il BVAR è indipendente per costruzione**: ogni stima parte dal pannello a
`as_of`, non dallo stato della precedente. I confini li dà `--list-blocks` e
non vanno riscritti a mano: cadono sulle settimane di **stima piena**, che
seguono le release BEA, ed è l'unico taglio che non costa una stima in più e
non cambia una riga rispetto a una passata continua. Un taglio annuale a
Capodanno, per dire, cadrebbe in mezzo a un trimestre e promuoverebbe
diciannove settimane da riuso a stima fresca: numeri che dipendono da come si è
affettato il lavoro, non dal modello.

Anche i quattro modelli di uno stesso blocco sono indipendenti: ciascuno parte
dallo stesso stream deterministico derivato dalla data e non condivide stato
mutabile con gli altri. `run_all.py` li scrive sotto radici temporanee diverse,
così checkpoint e file non collidono, e pubblica il blocco normale solo quando
tutti e quattro sono presenti. I benchmark vengono calcolati dal solo Q-BVAR.

**Il DFM è parallelo due volte: fra le celle, e dentro ognuna.** Delle 991
settimane di una cella solo quelle di **ri-stima** sono in catena — ognuna parte
dal θ del vintage precedente. Le altre, il 58-76% del lavoro, sono funzioni pure
di (θ del mese, vintage, trimestre) e girano in parallelo: `--workers` le manda
in un pool, a numeri **identici bit per bit** (verificato su
`diag3/student_t_ar1`, 991 venerdì, contro la passata sequenziale).

**Quattro è il tetto utile, non un limite arbitrario.** Dopo una ri-stima una
cella ha in volo le ~3 settimane congelate del mese più la ri-stima del mese
dopo: quattro lavori, e il quinto dipende da un θ che non esiste ancora.
Misurato su `diag3/student_t_ar1`: **155,4 → 62,1 minuti**. Alzarlo è lecito e
non rompe niente — i processi in più restano fermi — e servirebbe solo se la
cadenza di ri-stima diventasse più rada di quella mensile.

Il tempo di parete di una passata resta quello della **cella più lenta**, non
la media: le varianti `_ar1` stanno sopra le altre.

**I benchmark sono la sedicesima unità, non un passeggero.** AR(2) e media
espandente non dipendono da spec né da variante: si calcolano una volta sola.
Prima quel «una volta» era realizzato attaccandoli alla prima cella dell'ordine
canonico, e le loro righe finivano nel CSV di `diag3/gaussian` — tre serie in un
file che porta il nome di una, 6831 righe dove ne dichiara 2277. Ora hanno la
loro cartella (`dfm/_cells/benchmark/` → `dfm/csv/benchmark/`), il loro stato di
ripresa, e girano in parallelo alle celle invece che dentro una.

**L'asse è la cella e non il periodo, ed è voluto.** Percorrendo il 2007-2025
in un processo solo, la catena di θ resta continua per tutti e diciannove gli
anni. Spezzando per periodo si interrompe a ogni confine di shard, e il costo è
misurato: la ripresa non eredita un θ che non ha prodotto, quindi ogni confine
vale un gradino di **~2e-03 punti BEA** (`src/forecast/test_resume.py`). È
tre ordini di grandezza sotto le differenze di RMSE fra metodi (0.1–1.0), ma
va saputo prima di decidere, non dopo.

## Thread per processo: **uno**

Il parallelismo di questo progetto è **fra lavori indipendenti**, non dentro
l'algebra: 4 × 77 shard BVAR + 15 celle + 1 benchmark = 324 unità. Con molti
core la quantità da
minimizzare sono i **core-secondi per blocco**, cioè il rendimento per core,
non il tempo del singolo blocco. Quindi:

```
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

**Non è una scelta ovvia, ed è per questo che va scritta.** Dentro lo stesso
codice convivono due regimi opposti: la ricerca della moda va *meglio* con 1
thread (il multi-thread le fa perdere tempo in sincronizzazione), mentre lo
smoother dell'L-BVAR ne vorrebbe molti, perché è algebra densa. Un thread per
processo sacrifica il secondo a favore del primo e del parallelismo esterno.
Su questo progetto le stime a priori su "quanti thread" hanno sbagliato
ripetutamente: se si vuole cambiare, si misuri.

## La memoria: nessuno la guarda, la valvola è il numero di processi

**Non c'è un solo punto in tutto il codice che legga la RAM disponibile**, e
non è una svista: non esiste un numero giusto da cablare. Dipende dal modello
(l'L-BVAR tiene uno stato companion da `n·p` dimensioni nel simulation
smoother), dal numero di estrazioni, e dal tetto della cache su disco
(`--max-cache-mb`, default 1500 MB per blocco).

La valvola è **quanti processi si lanciano insieme**, e abbassarla non costa
niente, perché la ripresa è garantita: se la passata muore per memoria esaurita
al blocco 50, si rilancia con meno processi e i 50 finiti restano dove sono. I
blocchi restano 77 in ogni caso — cambia solo in quante ondate.

## Il conto dei processi, per intero

Due numeri diversi che è facile confondere: i **lavori** (le unità che la coda
distribuisce) e i **processi** (quello che il sistema operativo vede davvero).
Non coincidono, perché una cella DFM è un lavoro solo che ne apre quattro.

| | lavori | processi per lavoro | processi |
|---|---:|---:|---:|
| shard BVAR — 4 modelli × 77 blocchi | **308** | 1 | **308** |
| celle DFM — 15 celle | **15** | **4** (`DFM_WORKERS`) | **60** |
| benchmark DFM | **1** | 1 | **1** |
| **totale** | **324** | | **369** |

Il benchmark è a un processo per costruzione, non per scelta di configurazione:
`run_dfm.py` **rifiuta** `--workers` insieme a `--benchmark`, perché l'AR(2) e la
media espandente non hanno né catena di θ né settimane da riusare.

**Ma 369 non girano mai insieme.** `run_all.py` mette i 324 lavori in un pool da
`min(MAX_PROCESSES, core disponibili)`, con `MAX_PROCESSES = 224`, e li sottomette
in quest'ordine: prima il benchmark, poi le 15 celle, poi i 308 shard. I primi
224 lavori in volo sono quindi 1 benchmark + 15 celle + 208 shard, cioè

```
1 + 15×4 + 208 = 269 processi su 224 core = 1,20×
```

I 100 shard restanti aspettano un posto. **La sovrascrizione è quei 45 processi
in più** (15 celle × 3 worker oltre il loro slot), e dura solo finché le celle
DFM sono vive — al massimo le prime ~4 ore, dopo di che restano solo shard BVAR
uno per slot. È misurata benigna: a doppia sovrascrizione dei core il costo per
compito raddoppia ma la produttività continua a salire.

**Su una macchina più piccola si abbassa `MAX_PROCESSES`, non `DFM_WORKERS`**:
il primo è la valvola vera, il secondo toglierebbe l'unico parallelismo che il
cammino critico del DFM abbia.

## Ripresa

Rilanciare lo stesso comando. Niente viene cancellato all'avvio.

Un solo spigolo, e sta in `run_dfm.py`: la ripresa salta ogni riga già presente
nel CSV, **comprese quelle in errore** — sono righe scritte, quindi "fatte".
Rilanciare una cella guasta non la ripara e non lo dice. Per ripararla serve
`--fresh`, che cancella il CSV della cella e la ristima da capo. Lo script
stampa all'avvio quante righe trova e quante sono in errore.

## Ordine

Le stime (DFM e BVAR) sono indipendenti fra loro e si possono lanciare tutte
insieme. `run_outputs.py` va **dopo**: legge solo i CSV già su disco, non
stima niente, e si ferma se il DFM non ha prodotto nowcast invece di disegnare
figure sul vuoto. Dura minuti, non ore, e ha `--from` per riprendere da un
passo.
