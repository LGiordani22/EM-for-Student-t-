"""
src/bvar/ — BVAR per il nowcasting del PIL USA.

Replica di Cimadomo, Giannone, Lenza, Monti & Sokol (2022), "Nowcasting with
large Bayesian vector autoregressions", J. Econometrics 231, 500-519.

Componente INDIPENDENTE dal lavoro DFM / Growth-at-Risk del resto del repo:
modello diverso, codice diverso, cartella diversa.  Con il resto condivide solo
l'infrastruttura dati stabile (calendario dei rilasci, livelli grezzi), per
IMPORT e mai per modifica.

Vedi `src/bvar/README.md` per la struttura dei file e il piano a gate.


################################################################################
#                                                                              #
#   LA MAPPA DELLE QUATTRO VARIANTI  —  §2.1, §2.2, §2.3, §2.4                 #
#                                                                              #
#   Questa e' la sezione introduttiva comune ai quattro modelli.  I moduli      #
#   `qbvar.py`, `cbvar.py`, `bbvar.py`, `lbvar.py` nascono ai rispettivi gate   #
#   e si aprono rimandando QUI: la teoria che li distingue sta scritta una      #
#   volta sola, in un posto dove si legge tutta insieme.                        #
#                                                                              #
################################################################################


L'ASSE LUNGO CUI LE QUATTRO VARIANTI SI DISPONGONO
===================================================
Il paper presenta quattro modelli, ma non sono quattro idee diverse: sono
quattro RISPOSTE ALLA STESSA DOMANDA, e la domanda e' una sola.

    Il PIL esce ogni tre mesi.  L'occupazione, l'ISM, la produzione industriale
    escono ogni mese.  Un VAR ha UNA frequenza.  Dove metti la giuntura?

Le quattro risposte sono quattro POSTI diversi in cui mettere quella giuntura, e
l'ordine dei gate segue esattamente questo asse — da "il piu' lontano possibile
dal core" a "dentro il core":

    variante   dove avviene la riconciliazione        cosa tocca del core
    --------   -----------------------------------    -------------------------
    Q  §2.1    NEI DATI, a monte.  Si aggrega e si    niente: e' il core, punto
               butta via la frequenza mensile
    C  §2.4    NEI PARAMETRI, a valle.  Si stima      niente: una mappa
               trimestrale, poi Phi_m = Phi^(1/3)     deterministica DOPO
    B  §2.3    NEI DATI, a monte, senza buttare       niente: lo stesso core su
               via.  Ogni mensile -> 3 colonne        un pannello piu' largo
    L  §2.2    NEGLI STATI, DENTRO la stima.  Le      l'UNICO: si intreccia col
               trimestrali sono mensili latenti       core via `step()`

Da leggere cosi': **tre varianti su quattro non toccano il campionatore.**  Q, C
e B chiamano `core.sample()` una volta sola, su una matrice densa, e tutto il
loro contenuto sta in COME quella matrice e' stata costruita (Q, B) o in COSA si
fa dei parametri che ne escono (C).  Solo l'L-BVAR ha bisogno di `step()`, e
`step()` esiste dal Gate 1 proprio per questo.

E' l'architettura che il documento di contesto impone ("l'architettura non si
ramifica") ed e' anche il motivo per cui l'ordine dei gate e' Q -> C -> B -> L:
ogni gate aggiunge un pezzo e nessuno rimette in discussione il precedente.


CHE COSA CONDIVIDONO — il core, ed e' quasi tutto
==================================================
Il paper e' esplicito su ciascuno dei tre punti, e le citazioni contano perche'
e' cio' che ci autorizza a NON ramificare:

  1. LO STESSO PRIOR.  Normal-Inverse-Wishart, Psi = diag(psi), dof = n+2,
     Minnesota (eq. 2-3) + sum-of-coefficients via dummy observations.
       §2.2 (L-BVAR): "We adopt a Normal-Inverse Wishart prior with THE SAME
       PARAMETRIZATION AS THE BASELINE CASE, which combines the Minnesota prior
       with the sum-of-coefficients prior."
       §2.3 (B-BVAR): "We adopt THE SAME PRIOR that we use for the quarterly
       model."
     Il C-BVAR non ha un prior suo: eredita quello del Q-BVAR, perche' il
     Q-BVAR *e'* il suo stadio 1 (nota 20).

  2. LO STESSO TRATTAMENTO DEGLI IPERPARAMETRI.  Gerarchico alla Giannone,
     Lenza & Primiceri (2015): lambda, mu, psi sono variabili casuali estratte
     dal loro posterior, non costanti fissate a occhio.  §2.3: "As for the
     L-BVAR, we conduct posterior inference on the hyperparameters describing
     the informativeness of the priors, following Giannone et al. (2015)."

  3. LO STESSO p, nel senso giusto — cioe' lo stesso INSIEME INFORMATIVO.
     5 lag trimestrali per Q/C/B, 17 mensili per L.  Non e' una scelta libera:
     la nota 10 la deriva.  "17 monthly lags ensure consistency with the
     information sets of the B-BVAR and C-BVAR models, which are estimated with
     5 quarterly lags.  For example, with data available until the end of
     March, ... the B-BVAR and C-BVAR include lagged monthly information up
     until October of the year before the last (the former BECAUSE OF ITS BLOCK
     STRUCTURE, the latter BECAUSE MONTHLY VARIABLES ENTER AS THREE-MONTH
     MOVING AVERAGES)."

     Quella parentesi finale e' importante e va letta due volte: B e C arrivano
     alla stessa profondita' PER DUE MECCANISMI DIVERSI.  E' la prima traccia
     della differenza che il resto di questa mappa svolge.

  4. L'INVARIANTE DEL GATE 0: il core non vede mai un NaN.  Il dato mancante e'
     gestito fuori — dal Kalman a valle (Q, C, B) o dal simulation smoother a
     monte (L).  Vedi l'header di `data.py` per i due tipi di buco (bordo
     frastagliato vs partenza tardiva) e perche' solo l'L-BVAR risolve il
     secondo.


================================================================================
Q-BVAR  §2.1  —  il baseline, e lo stadio 1 del C-BVAR
================================================================================

COSA DICE IL PAPER
------------------
E' l'eq. (1) e basta: un VAR(p) trimestrale, p=5, stimato in forma chiusa.

    "When all variables in the vector x_tq are available, the model can be
     readily estimated with standard Bayesian methods"                   (§2.1)

E la nota 20, che e' la ragione per cui questo modello compare due volte nel
paper:

    "The Q-BVAR corresponds to the first step needed to obtain the C-BVAR,
     see Section 2.4."                                                (nota 20)

COSA SIGNIFICA
--------------
Il Q-BVAR non RISOLVE il problema della frequenza mista: lo DISSOLVE, buttando
via la frequenza mensile prima ancora di stimare.  Tutto sta a monte, nella
costruzione del pannello trimestrale (vedi la sezione sull'aggregazione, sotto).

Il che gli da' due ruoli distinti, che conviene tenere separati in testa:

  * come MODELLO e' il termine di paragone — quello che le tre varianti a
    frequenza mista devono battere, e lo battono solo dove ci si aspetta:
    "we find differences in performance across the three methods only in the
    first few weeks of the quarter, when no information on the current quarter
    is available.  After that, all the mixed-frequency models are comparable
    and OUTPERFORM A STANDARD QUARTERLY VAR."  Il Q-BVAR e' cieco dentro il
    trimestre per costruzione: e' la definizione del vantaggio che stiamo
    misurando, non un difetto dell'implementazione.

  * come STADIO 1 e' il fornitore di (Phi, Sigma_eps) per il C-BVAR.  Da cui il
    vincolo di interfaccia al Gate 2: l'output del Q-BVAR dev'essere gia' nella
    forma che il Gate 3 consuma, altrimenti si rifattorizza.

COME SI TRADUCE IN CODICE
-------------------------
`qbvar.py`, Gate 2.  E' un WRAPPER: zero matematica nuova, zero campionatore
nuovo.

    pannello mensile in livelli grezzi (data.build_panel)
      -> media mobile a 3 mesi          <- l'unico passo nuovo
      -> log / identita' (to_model_units)
      -> campiona ai mesi 3/6/9/12
      -> assert_dense
      -> core.sample()

E in uscita, per ogni estrazione s, gli oggetti dell'Appendice A:

    Phi[s]     (np, np)   companion di (A_1 ... A_p)             — eq. (A.1)
    Sigma[s]   (n, n)     Sigma_eps, da cui Omega = blkdiag(Sigma_eps, 0)
    const[s]   (n,)       A_0

La regola che tiene il Gate 3 disaccoppiato: la mappa cube-root sara' una
FUNZIONE PURA di (Phi, Sigma_eps) — array in, array out — non un metodo che
sappia che cos'e' un Q-BVAR.  Cosi' `cube_root.py` si testa da solo contro
l'esempio AR(2) dell'Appendice A.1 (eq. A.11-A.15), senza stimare niente.


================================================================================
C-BVAR  §2.4 + Appendice A  —  la radice cubica
================================================================================

COSA DICE IL PAPER
------------------
Si assume che il VAR trimestrale sia l'ITERATA A TRE PASSI di un VAR mensile
non osservato.  In companion form:

    X_tq = Phi X_tq-1 + nu_tq                                          (A.1)
    X_tm = Phi_m X_tm-1 + nu_m,tm                                      (A.3)

e iterando tre volte la mensile e uguagliandola alla trimestrale:

    Phi_m = Phi^(1/3)                                                  (A.6)
    nu_tm = nu_m,tm + Phi_m nu_m,tm-1 + Phi_m^2 nu_m,tm-2              (A.7)

Da (A.7), risolvendo il sistema sovradeterminato delle ultime n(p-1) righe per
eps_m,tm-1 (A.8), si recupera la covarianza mensile:

    vec(Sigma_eps_m) = (I + A (x) A)^-1 vec(Sigma_eps)                 (A.9)
    A = Phi^2_m11 - Phi_m11 (J'J)^-1 J' Phi_m.1,   J = [I_n ... I_n]'

e (A.10) e' la scorciatoia di Kronecker che evita di invertire una n^2 x n^2.

Selezione della radice, quando ce n'e' piu' d'una: autovalori reali -> la loro
radice cubica reale; coppie complesse coniugate -> "the cube root which is
characterized by the LEAST OSCILLATORY BEHAVIOUR, i.e., the cube root with the
smallest argument", come in Giannone, Monti & Reichlin (2016).

COSA SIGNIFICA
--------------
E' l'unica delle quattro varianti in cui la frequenza mensile e' RICOSTRUITA
invece che osservata o aggregata.  Non si stima niente di mensile: si stima
trimestrale, e poi si DEDUCE la legge mensile che, campionata a fine trimestre,
riprodurrebbe quella trimestrale.

Il prezzo e' una restrizione precisa, ed e' scritta in (A.4): il valore mensile
corrente dipende da UN SOLO MESE dentro ogni trimestre passato.  Non e' un VAR
mensile qualsiasi — e' il sottoinsieme di VAR mensili compatibili con quello
trimestrale.

E il prezzo dell'aggregazione: la nota 15 lo ammette apertamente.  "If instead
the variables considered are flows, then our definition of the monthly
variables as an average over the quarter implies that we are introducing a
non-invertible moving average in the growth rates.  Therefore modelling this
monthly concept as autoregressive INTRODUCES SOME MIS-SPECIFICATION."  Va
citato: e' il costo dichiarato dalla fonte, non un'obiezione nostra.

COME SI TRADUCE IN CODICE
-------------------------
Gate 3, tre file, e nessuno di loro e' un campionatore:

    cube_root.py    (Phi, Sigma_eps) -> (Phi_m, Sigma_eps_m).  Funzione PURA,
                    deterministica, applicata draw per draw.  Autovalori,
                    selezione della radice, A.9 con la scorciatoia A.10.
    state_space.py  companion mensile + ciclo predict/update scritto sulle
                    primitive importate da `src/kalman.py` (mai riscritte, mai
                    modificate — vincolo del documento di contesto).
    cbvar.py        la colla: stadio 1 = qbvar, stadio 2 = mappa, stadio 3 =
                    filtro sul flusso dati reale.

PUNTO APERTO da portare al Gate 3, non da decidere qui: la companion (A.1)
dell'Appendice A NON HA LA COSTANTE.  Il nostro VAR ha A_0.  Dove finisce A_0
nel modello mensile il paper non lo dice.  E' una decisione da §1b regola 4.


================================================================================
B-BVAR  §2.3  —  blocking / impilamento
================================================================================

COSA DICE IL PAPER
------------------
Si allinea tutto alla frequenza PIU' BASSA, trattando ogni mensile come tre
serie trimestrali distinte — una per mese del trimestre:

    x^q_tq = [ x'_tm-2  x'_tm-1  x'_tm ]'                               (§2.3)

impilate con le trimestrali vere y_tq.  "x_tq is a vector of length n = q + 3m,
where q is the number of quarterly variables and m is the number of monthly
variables in our system."  Poi: VAR(5) trimestrale, stesso prior, stimato in
forma chiusa.

E la nota 13, che e' la parte onesta:

    "These priors do not explicitly take into account that, in the B-BVAR, some
     variables reflect the observations for three consecutive months of the
     same monthly time series.  WE MAINTAIN THE SAME PRIORS used in other
     models TO PRESERVE COMPARABILITY."

COSA SIGNIFICA
--------------
E' la variante senza trucchi: nessuno stato latente, nessuna radice, nessuna
media.  Solo un cambio di etichette.  Niente viene buttato via e niente viene
approssimato — il mese di gennaio resta il mese di gennaio, e' solo diventato
una colonna a se'.

Il costo e' tutto dimensionale, e per noi e' grosso.  Nel paper: 14 mensili + 4
trimestrali -> n = 4 + 42 = 46.  Da noi, sul profilo `q_b`:

    27 mensili x 3 + 3 trimestrali  =  n = 84
    k = 84*5 + 1 = 421 coefficienti per equazione
    B ha 84 * 421 = 35 364 elementi, su T ~ 130 osservazioni trimestrali

E' il regime "large n, small T" che il metodo esiste per affrontare, ma e' anche
il sistema piu' largo dei tre in forma chiusa: e' li' che il dimensionamento dei
test va rifatto (vedi `notes.dimensionamento` nella config).

DIFFERENZA DA NON PERDERE, ed e' quella che la nota 10 anticipava: nel B-BVAR i
mensili entrano GREZZI, mese per mese.  NON mediati.  La media mobile a 3 mesi
e' della famiglia Q/C — qui sarebbe esattamente l'informazione che il blocking
esiste per non perdere.

COME SI TRADUCE IN CODICE
-------------------------
Gate 4.  `bbvar.py` fa l'impilamento (contabilita', non matematica) e riusa
`state_space.py` del Gate 3 per il bordo frastagliato.  Il core e' chiamato una
volta sola, come per Q.


================================================================================
L-BVAR  §2.2  —  le trimestrali come processi latenti
================================================================================

COSA DICE IL PAPER
------------------
Il VAR e' MENSILE, p=17, e le trimestrali sono mensili con due osservazioni
mancanti su tre.  La stima e' un ciclo MCMC in tre passi:

    "Using the simulation smoother of Durbin and Koopman (2001), we draw the
     complete monthly dataset (i.e., including draws of the latent missing
     values) conditional on the model parameters A_m's and Sigma_m; then, using
     the posterior sampler of Giannone et al. (2015), we draw the
     hyperparameters lambda, mu and psi conditional on the complete monthly
     dataset, and finally, we draw the model parameters conditional on the
     hyperparameters and the complete monthly dataset."                 (§2.2)

Inizializzazione, che il paper specifica per intero: si interpolano le
trimestrali con delle spline per avere un pannello mensile completo
preliminare; le condizioni iniziali sono Normali con media pari ai primi p mesi
di quel pannello e varianza zero o Psi_ii a seconda che il dato sia osservato o
stimato; i parametri partono dalla loro media a priori.

E la nota 11, sulla natura della variabile latente: "We treat quarterly data as
monthly data available only in the last month of the quarter.  Hence, the latent
variable we estimate INHERITS THE FEATURES OF THE QUARTERLY VARIABLE."  Hanno
provato a imporre esplicitamente la restrizione di aggregazione nello state
space e "we do not find improvements given the very general lag structure of the
model".

COSA SIGNIFICA
--------------
E' l'unica variante in cui il dato mancante e' DENTRO la stima invece che prima
o dopo.  Da cui la conseguenza che decide i profili di variabili di tutto il
pacchetto: solo l'L-BVAR puo' usare tutte e 37 le serie, perche' solo lui puo'
trattare una PARTENZA TARDIVA (PPIFIS che non esiste prima del 2009) come stato
latente da riempire.  Q, C e B ne usano 30.  Vedi `data.py` per intero.

Ed e' l'unica in cui la giuntura fra le frequenze non e' una trasformazione dei
dati ne' una mappa sui parametri: e' una VARIABILE CASUALE che si estrae a ogni
iterazione.  Niente medie, niente radici, niente colonne triplicate — il
pannello mensile completo e' un oggetto stocastico.

COME SI TRADUCE IN CODICE
-------------------------
Gate 5, ed e' l'unico gate che tocca il core.

    simsmoother.py  Durbin-Koopman, codice NUOVO scritto in bvar/, sopra le
                    primitive importate da `src/kalman.py`.  Testato in
                    isolamento PRIMA di entrare nel ciclo.
    lbvar.py        il ciclo dei tre passi.

L'aggancio esiste gia' dal Gate 1 e non va inventato: `core.step(state, rng,
panel=...)` ricostruisce il target sul nuovo pannello, rivaluta la log-posterior
corrente e fa UNA spazzata, mantenendo la catena degli iperparametri da dove
stava.  E' esattamente il terzo e secondo passo del ciclo di §2.2; il primo
passo — l'estrazione del pannello — e' cio' che il Gate 5 aggiunge.

E' anche il modello piu' pesante: 37 variabili x 17 lag mensili e' lo state
space piu' grande dei quattro.  Ultimo, per questo.


================================================================================
L'AGGREGAZIONE MENSILE -> TRIMESTRALE  —  chi la usa e chi no
================================================================================
Riguarda Q e C.  Non riguarda B ne' L.  Vale la pena fissarlo qui perche' e' la
prima cosa che si sbaglia leggendo la nota 17 di corsa.

    nota 17: "As discussed in Section 2.4, FOR THE C-BVAR monthly variables are
    transformed so as to correspond to a quarterly quantity when observed in the
    final month of each quarter BEFORE TAKING LOGS (see Giannone et al., 2008).
    With our data, that means taking 3-months moving averages of ALL MONTHLY
    VARIABLES."

  * C-BVAR: si', ed e' obbligatorio.  La radice cubica ha senso solo se la
    variabile mensile e' lo STESSO CONCETTO della trimestrale, cosi' che
    campionarla a fine trimestre restituisca la trimestrale.  E' la definizione
    stessa di "quarterly quantity when observed at end of the quarter".

  * Q-BVAR: si', ma per deduzione, non perche' la nota 17 lo dica.  La nota 17
    nomina il solo C-BVAR; la nota 20 dice pero' che il Q-BVAR *e'* lo stadio 1
    del C-BVAR.  Se il Q-BVAR stimasse su un pannello costruito diversamente, la
    Phi che il Gate 3 prende a radice cubica non sarebbe la Phi del sistema che
    il C-BVAR assume.  L'unica lettura internamente coerente e' un solo pannello
    trimestrale, quello con le medie mobili.

  * B-BVAR: no.  I mensili entrano grezzi, mese per mese (§2.3).  Mediare qui
    distruggerebbe il motivo per cui il blocking esiste.

  * L-BVAR: no.  Lavora sui mensili cosi' come sono; le trimestrali sono mensili
    osservati solo nell'ultimo mese del trimestre (nota 11).

I tre punti di esecuzione — ordine delle operazioni, serie gia' in livello,
bordo frastagliato — sono discussi nell'header di `qbvar.py`, dove il codice che
li applica sta accanto.
"""
