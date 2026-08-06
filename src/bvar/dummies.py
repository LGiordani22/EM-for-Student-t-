"""
src/bvar/dummies.py

IL PRIOR SUI COEFFICIENTI: prima in forma analitica, poi (Blocco 4) in forma
operativa come dummy observations.

Questo modulo ospita le due facce della stessa cosa:

  * BLOCCO 2 (questo file, ora) — i MOMENTI ANALITICI del prior di Minnesota,
    cioe' le eq. (2)-(3) di Cimadomo scritte letteralmente: la media `b` e la
    diagonale di `Omega`.
  * BLOCCO 4 (in arrivo) — le DUMMY OBSERVATIONS alla Banbura, Giannone &
    Reichlin (2010) eq. (5), cioe' le righe fittizie (Yd, Xd) da impilare sopra
    i dati veri perche' il posterior coniugato riproduca quel prior.

PERCHE' IMPLEMENTARE LA STESSA COSA DUE VOLTE
---------------------------------------------
Non e' ridondanza, e' l'architettura del test.  Il percorso di calcolo vero
passa dalle dummy (sono l'unico modo comodo di imporre anche il
sum-of-coefficients, e regolarizzano l'inversione: BGR, "Adding dummy
observations works as a regularisation solution to the matrix inversion
problem").  Ma dalle dummy si risale ai momenti impliciti con le identita' di
BGR:

    B0 = (Xd' Xd)^-1 Xd' Yd        Omega0 = (Xd' Xd)^-1

Se quei momenti impliciti coincidono con le eq. (2)-(3) calcolate qui in modo
diretto, allora sia la nostra LETTURA del paper sia la nostra IMPLEMENTAZIONE
sono giuste.  Un percorso solo non lo potrebbe dimostrare: un errore di lettura
si propagherebbe identico nelle due direzioni senza mai contraddirsi.


LE EQUAZIONI (Cimadomo §2.1)
============================
Medie, eq. (2):

    E(A_1) = diag(d),    E(A_2) = ... = E(A_p) = 0_n

dove `d` e' il vettore n x 1 di 1 (random walk) e 0 (white noise) — nel nostro
codice `MinnesotaSpec.d_centre`, costruito dalla config al Gate 0.  Solo la
DIAGONALE del primo lag e' centrata su d: gli elementi fuori diagonale di A_1
sono centrati su zero, cioe' a priori la variabile j non aiuta a prevedere la
variabile i.

Varianze, eq. (3):

    Cov[(A_s)_ij, (A_r)_hm | Sigma] = lambda^2 * Sigma_ih / (s^2 * psi_j)
                                      se m = j e r = s,  zero altrimenti

che nella forma coniugata Cov(vec(B)|Sigma) = Sigma (x) Omega significa

    Omega[(s,j), (s,j)] = lambda^2 / (s^lag_decay * psi_j)

e Omega e' DIAGONALE (la covarianza e' non nulla solo per (r,m) = (s,j)).


IL REFUSO DELL'EQ. (3): perche' il codice usa psi_j e non psi_i
================================================================
L'eq. (3) COME STAMPATA ha `Psi_ii` al denominatore.  Non e' quello che va
usato, e non e' una nostra deviazione dal paper: e' un refuso che il paper
stesso smentisce.  I fatti, separando i due indici:

    fonte                          numeratore      denominatore
    ---------------------------------------------------------------
    Cimadomo eq. (3), stampata     Sigma_ih  OK    Psi_ii    SBAGLIATO
    Cimadomo, prosa 2 righe dopo   Sigma_ii  (*)   Psi_jj    OK
    GLP (2015) §III                Sigma_ih  OK    psi_j     OK

    (*) la prosa scrive `ii` perche' sta descrivendo a parole il caso i = h,
        non perche' l'indice sia quello.

Ma l'argomento decisivo non e' il conteggio delle fonti, e' strutturale.  Nel
prior coniugato Cov(vec(B)|Sigma) = Sigma (x) Omega, dove Sigma e' indicizzata
dalle EQUAZIONI e Omega dai REGRESSORI:

    Cov[(B_s)_ij, (B_r)_hm] = Sigma_ih * Omega[(s,j),(r,m)]

Omega NON PUO' dipendere dall'indice di equazione i: se dipendesse, la
fattorizzazione di Kronecker cadrebbe e il prior non sarebbe piu' coniugato —
cioe' verrebbe meno tutto il Blocco 1.  Un denominatore psi_i renderebbe Omega
funzione dell'equazione: e' strutturalmente impossibile.  `psi_j` e' l'unico
indice compatibile con l'eq. (4) di GLP.

Infine: siccome d = n+2 implica d-n-1 = 1, si ha psi_j/(d-n-1) = psi_j, quindi
le scritture di Cimadomo e GLP coincidono NUMERICAMENTE.  Il disaccordo e'
soltanto di indice.

(Decisione 5 del Gate 0, presa con il relatore.)


DUE COLLISIONI DI NOTAZIONE DA NON DIMENTICARE
==============================================
1. Cimadomo usa la lettera `d` per DUE oggetti diversi nella stessa
   sottosezione: `d = n+2` (gradi di liberta', scalare) e `E(A_1) = diag(d)`
   (vettore rw/wn).  Nel codice: `dof` per il primo, `d_centre` per il secondo.
   Mai `d` da solo.
2. BGR (2010) usa `Psi` per la COVARIANZA dei residui e `S0` per la scala
   dell'IW — l'opposto di Cimadomo e GLP.  Da tenere a mente al Blocco 4.


================================================================================
LA MINNESOTA IN CONCRETO: un esempio a 3 variabili, con numeri veri
================================================================================
Tre serie del profilo q_b, in unita' del modello, campione di stima
1992Q1-2025Q3:

                   GDPC1     ISM_PMI   UNRATE
    1992-03-31    9.2337      54.6      7.4        <- log-livello / livello / livello
    2025-09-30   10.0869      49.1      4.4
    d_centre         1          0         1        <- rw, wn, rw

IL CENTRO A PRIORI, EQ. (2), scritto come matrice:

                    GDPC1  ISM  UNRATE
                   +---------------------
    E(A_1) = GDPC1 |   1     0     0
             ISM   |   0     0     0          E(A_2) = ... = E(A_5) = 0_{3x3}
             UNRATE|   0     0     1

Sulla diagonale c'e' `d_centre`: 1 per GDPC1 e UNRATE (persistenti), 0 per
l'ISM (survey mean-reverting).  Fuori diagonale SEMPRE zero, in tutti i lag: a
priori nessuna variabile aiuta a prevederne un'altra.  Se il prior vincesse del
tutto il sistema sarebbe

    GDP_t = GDP_{t-1} + eps      ISM_t = c + eps      U_t = U_{t-1} + eps

LE SD A PRIORI, EQ. (3), con lambda = 0.6 e psi dai residui AR(5) univariati
(psi_GDP = 1.3e-4, psi_ISM = 8.58, psi_UNRATE = 0.517).

NOTA SU QUESTI psi: sono presi dai residui di un AR(5) univariato SOLO per dare
numeri concreti all'esempio — e' il modo in cui li FISSA BGR (2010).  Nel modello
vero psi NON e' fissato: e' un iperparametro CAMPIONATO dal suo posterior
insieme a lambda e mu.  GLP (2015) §III: "We treat psi as a hyperparameter,
which differs from the existing literature that has been fixing this parameter
using sample information."  Vedi il Blocco 5.

    lag 1:      GDPC1   ISM_PMI  UNRATE      lag 5:    GDPC1  ISM_PMI  UNRATE
    GDPC1      0.6000   0.0023   0.0095      GDPC1    0.1200  0.0005   0.0019
    ISM_PMI  153.6993   0.6000   2.4442      ISM_PMI 30.7399  0.1200   0.4888
    UNRATE    37.7300   0.1473   0.6000      UNRATE   7.5460  0.0295   0.1200

Tre cose da leggere qui dentro:
  * la DIAGONALE e' esattamente lambda = 0.6, in ogni equazione: lambda E' la sd
    a priori del coefficiente di ogni variabile su se stessa;
  * 0.0023 e 153.70 non sono un bug, sono le UNITA'.  Un punto di ISM non puo'
    muovere il log-PIL di piu' di qualche millesimo; viceversa un'unita' di
    log-PIL (il PIL moltiplicato per e!) e' un movimento enorme, che in punti
    ISM vale moltissimo.  E' il fattore sqrt(psi_i/psi_j) che converte le unita'
    — ed e' cio' che permette al prior di lavorare su dati NON standardizzati;
  * il lag 5 ha esattamente UN QUINTO delle sd del lag 1 (0.12 = 0.6/5).


LA MINNESOTA LETTA PER RIGHE: che cosa dice, equazione per equazione
=====================================================================
La matrice E(A_1) qui sopra si legge meglio UNA RIGA ALLA VOLTA, perche' ogni
riga E' un'equazione del VAR.

Prendi la prima riga, l'equazione del PIL:

    GDPC1_t = a_11 GDPC1_{t-1} + a_12 ISM_{t-1} + a_13 UNRATE_{t-1} + ...

La Minnesota dice:

    E(a_11) = d_centre[GDPC1] = 1     <- il proprio primo lag
    E(a_12) = 0                       <- il lag dell'ISM
    E(a_13) = 0                       <- il lag di UNRATE

cioe': "a priori penso che il PIL sia spiegato soprattutto dal proprio passato,
e non dal passato delle altre variabili".  Lo stesso vale per ogni altra riga.
Quindi la Minnesota non parla solo dei LAG: incorpora anche l'idea che gli
EFFETTI INCROCIATI siano inizialmente poco importanti.

PERCHE' HA SENSO ECONOMICAMENTE.  Nel nostro Q-BVAR ci sono 30 variabili e 4530
coefficienti.  E' implausibile che ogni variabile dipenda fortemente da tutte le
altre.  La Minnesota dice: "parto pensando che ogni variabile sia
principalmente autoregressiva; se i dati mostrano che l'ISM anticipa il PIL o
che i prezzi all'import muovono il CPI, lo accetto — ma deve essere l'evidenza
a convincermi".

E ALLORA IL VAR NON PERDE IL SUO SENSO?  No, ed e' il punto elegante.  IL PRIOR
E' SOLO IL PUNTO DI PARTENZA.  Se nei dati c'e' che l'ISM anticipa il PIL, il
POSTERIOR assegnera' a quel coefficiente fuori diagonale un valore diverso da
zero.  Il VAR resta un vero VAR, ricco di interazioni: il prior impedisce
soltanto di attribuire automaticamente un ruolo importante a TUTTE le relazioni
possibili quando non c'e' evidenza sufficiente.

E' la filosofia dei BVAR: partire da un modello semplice — quasi un insieme di
AR univariati — e lasciare che siano i dati ad aggiungere le interazioni
realmente necessarie.  E' uno dei motivi per cui i BVAR prevedono meglio dei VAR
senza prior quando le variabili sono tante.


LA MEDIA E LA COVARIANZA SONO DUE ASSI DISTINTI  <<< IL CUORE
==============================================================
Questa distinzione e' la chiave di tutto il prior, e va tenuta separata:

    LA MEDIA (eq. 2)       dice DOVE sono centrati i coefficienti.
    LA COVARIANZA (eq. 3)  dice QUANTO sei disposto a discostarti dal centro.

La media e' la parte "economica": ogni variabile dipende soprattutto dal proprio
passato, gli effetti incrociati partono da zero.

La covarianza e' la parte che decide QUANTO PESO DARE AL PRIOR RISPETTO AI DATI
(cioe' alla verosimiglianza).  E' letteralmente il meccanismo dello shrinkage:

    covarianza PICCOLA  ->  "sono molto sicuro della mia convinzione iniziale"
                            -> forte shrinkage verso il prior
                            -> la verosimiglianza conta poco
    covarianza GRANDE   ->  "non sono sicuro, lascio parlare i dati"
                            -> shrinkage debole
                            -> la verosimiglianza pesa di piu'

Un esempio concreto sul coefficiente dell'ISM nell'equazione del PIL, a_12.  Il
prior dice E(a_12) = 0 in ogni caso.  Ma:
    varianza piccola -> "non credo molto che l'ISM muova il PIL";
    varianza grande  -> "potrebbe contare, vediamo cosa dicono i dati".
Stesso centro, conclusioni diverse.

In una frase: LA MINNESOTA ASSUME CHE OGNI VARIABILE SIA SPIEGATA
PRINCIPALMENTE DAI PROPRI LAG, MENTRE GLI EFFETTI DEGLI ALTRI REGRESSORI SIANO
INIZIALMENTE PICCOLI; LA MATRICE DI COVARIANZA DETERMINA QUANTO FORTE E' QUESTA
CONVINZIONE.


LAMBDA E I DUE CASI LIMITE
===========================
lambda governa la covarianza COMPLESSIVA (moltiplica ogni elemento dell'eq. 3).
Il punto da non perdere: LAMBDA NON SPOSTA IL CENTRO.  Sposta solo quanto i dati
possono allontanartene.

    lambda -> 0     la varianza del prior collassa.  I coefficienti SONO la
                    media a priori: IL POSTERIOR COINCIDE COL PRIOR, i dati non
                    contano nulla, e il VAR si riduce a un insieme di random
                    walk (o white noise) INDIPENDENTI.

    lambda -> inf   la varianza esplode, il prior svanisce: SI RICADE NEL VAR
                    STIMATO CON OLS CLASSICO, SENZA PRIOR.  Che per noi vuol
                    dire 151 regressori su 130 osservazioni, X'X singolare,
                    stima indefinita.

Cimadomo §2.1 lo dice cosi': "For lambda = 0 the posterior equals the prior and
the data do not influence the estimates.  If lambda -> infinity, posterior
expectations coincide with the Ordinary Least Squares (OLS) estimates."

Ne segue che la scelta di lambda e' cruciale, e che il modo giusto di leggerla
non e' "quanto e' informativo il prior" in astratto, ma QUANTO PERMETTI AI DATI
DI SPOSTARTI DAL PUNTO DI PARTENZA.

IL NOSTRO NUMERO, misurato sul profilo q_b col pannello del Gate 2 (medie mobili
a 3 mesi, nota 17; T=135 trimestri):

    lambda ~ 0.52   [90%: 0.490, 0.559]      banda del paper (Tab. B.1): 0.59-0.75

e siccome lambda E' la deviazione standard a priori del coefficiente del proprio
primo lag (vedi `prior_coefficient_sd`), si legge cosi': A PRIORI QUEL
COEFFICIENTE E' 1 +/- 0.52.  Abbastanza stretto da regolarizzare 4530
coefficienti, abbastanza largo da lasciare che i dati lo spostino sul serio.

Sta SOTTO la banda del paper, e ci si aspetta che ci stia: la Tabella B.1
riporta B, C ed L — non il Q-BVAR — e il lambda ottimale cala al crescere di n
(BGR 2010).  Il nostro n=30 contro il loro n=18, a T quasi uguale, chiede piu'
shrinkage.  Sul pannello a fine trimestre (prima del Gate 2) veniva 0.55; prima del fix
al sum-of-coefficients (righe wn azzerate) veniva 0.49.  Vedi README, "I nostri numeri".


L'EQ. (3) FATTORE PER FATTORE
==============================
Nella nostra notazione, la varianza a priori del coefficiente della variabile j
al lag s nell'equazione i e'

    Var[(A_s)_ij | Sigma] = lambda^2 * (1/s^2) * Sigma_ii / psi_j

e ogni fattore e' una delle intuizioni di sopra, resa formula:

 1. lambda^2      LA TIGHTNESS COMPLESSIVA.  Il peso prior-contro-dati, uguale
                  per tutti i coefficienti.  E' il fattore descritto sopra.

 2. 1/s^2         IL DECADIMENTO COI LAG.  `s` e' l'INDICE DEL LAG (1, 2, ..., p),
                  non un parametro: quindi 1/s^2 fa si' che i lag piu' lontani
                  siano stretti PIU' FORTE verso zero.  In numeri:

                      lag 1 -> 1/1 = 1.00
                      lag 2 -> 1/4 = 0.25
                      lag 3 -> 1/9 = 0.11
                      lag 5 -> 1/25 = 0.04

                  E' questo che FORMALIZZA E(A_2) = ... = E(A_p) = 0: non basta
                  centrarli su zero, bisogna anche dire quanto ci si crede — e
                  ci si crede tanto piu' quanto il lag e' lontano.
                  (L'esponente 2 e' fissato dal paper, nota 8.)

 3. Sigma_ii/psi_j  IL RAPPORTO DI SCALE.  Serve a tenere conto che variabili di
                  taglia diversa avrebbero automaticamente coefficienti di
                  taglia diversa.  Usando E(Sigma) = Psi (Blocco 1) il rapporto
                  e' psi_i/psi_j, cioe' la scala dell'EQUAZIONE diviso quella
                  del REGRESSORE: esattamente il fattore di conversione fra le
                  unita' delle due variabili.  Senza, il prior sarebbe
                  inesistente per le serie a scala grande e soffocante per
                  quelle a scala piccola — e non si potrebbe lavorare su dati
                  non standardizzati.
                  ATTENZIONE all'indice: psi_J, la variabile RITARDATA.  Vedi
                  la sezione "IL REFUSO DELL'EQ. (3)".

 4. (struttura)   LA DIAGONALITA'.  La covarianza e' non nulla solo quando
                  (r,m) = (s,j), cioe' A PRIORI I COEFFICIENTI SONO TRATTATI
                  COME INDIPENDENTI.  E' il motivo per cui Omega si tiene come
                  vettore (k,) e non come matrice (k,k).
                  (L'unica eccezione e' la correlazione fra equazioni via
                  Sigma_ih, che e' un sottoprodotto della struttura di
                  Kronecker — vedi sotto.)

Tutto insieme, la formula dice: "credo che ogni variabile sia principalmente
autoregressiva, credo poco negli effetti incrociati, credo ancora meno nei lag
lontani, e tengo conto delle scale — ma lascio che i dati cambino queste
convinzioni, tanto piu' quanto lambda e' grande".


"CENTRATO SU ZERO" NON VUOL DIRE "VINCOLATO A ZERO"
===================================================
E(A_2) = ... = E(A_p) = 0 NON significa che i lag oltre il primo siano esclusi.
Ci sono tutti, con tutti i loro coefficienti, e sono LIBERI di essere diversi da
zero.  Il prior dice: "in assenza di evidenza la mia scommessa e' zero, ma sono
disposto a cambiare idea — e questa e' quanta evidenza mi serve".

Il "quanta evidenza mi serve" e' la VARIANZA, cioe' l'eq. (3):

    centrato su 0, varianza 0      -> dogmatico: il coefficiente E' zero, il lag
                                      e' escluso davvero.  (E' il caso lambda=0.)
    centrato su 0, varianza (3)    -> informativo: parte da zero e i dati lo
                                      spostano in proporzione a quanto lo chiedono.
    nessun prior, varianza infinita-> piatto: il coefficiente e' l'OLS, rumore
                                      compreso.

Il posterior e' in sostanza una MEDIA PESATA fra il centro a priori (zero) e cio'
che dicono i dati, con pesi pari alle rispettive precisioni.  Segnale forte sul
lag 3 -> il coefficiente si sposta; segnale debole -> resta vicino a zero e non
aggiunge rumore alla previsione.

Il decadimento 1/s^2 dice quindi: PIU' IL LAG E' LONTANO, PIU' FORTE DEV'ESSERE
L'EVIDENZA PER SPOSTARLO.  Non "il lag 5 non esiste", ma "il lag 5 deve
dimostrare di servire cinque volte piu' convincentemente del lag 1".  E' per
questo che Cimadomo puo' scrivere che "inference tends to be robust to the
specific value of p, provided that it is large enough": i lag inutili arrivano
gia' quasi spenti e non fanno danni.


MINNESOTA E SUM-OF-COEFFICIENTS: DUE PRIOR DISTINTI E ADDITIVI
===============================================================
Tre affermazioni nette, perche' e' qui che nasce la confusione.

(i) LA MINNESOTA NON "E'" RW O WN.  La Minnesota e' la STRUTTURA del prior: la
    forma delle medie (eq. 2) e delle varianze (eq. 3), col decadimento sul lag
    e il fattore di scala.  `rw` vs `wn` e' la scelta del CENTRO dentro quella
    struttura, fatta serie per serie — il nostro `d_centre`.  Un dataset di sole
    survey userebbe la stessa identica Minnesota con tutti d=0.  Struttura e
    centro sono due livelli diversi.

(ii) AGISCONO SU OGGETTI DIVERSI.

     Minnesota            -> ogni SINGOLO coefficiente (A_s)_ij preso da solo:
                             dove sta e quanto puo' muoversi.   ~k*n affermazioni
     sum-of-coefficients  -> la SOMMA sui lag, sum_s (A_s)_ij: una sola
                             affermazione per coppia (i,j).      ~n*n affermazioni

     La Minnesota non dice NULLA sulle somme; il soc non dice NULLA sui singoli
     coefficienti.  Parlano di livelli di aggregazione diversi degli stessi
     parametri.

(iii) SONO ADDITIVI, NON ALTERNATIVI.  Convivono nello stesso modello e il
     posterior li rispetta entrambi.  Cimadomo: "we COMBINE the Minnesota prior
     ... WITH the sum-of-coefficients prior".  Operativamente si vede benissimo:
     sono due BLOCCHI DI RIGHE DIVERSE della stessa matrice di dummy, impilati
     uno sopra l'altro.

L'immagine da tenere:

    LA MINNESOTA DICE DOVE STA OGNI SINGOLO MATTONE.
    LA SUM-OF-COEFFICIENTS DICE COME SI COMPORTA IL MURO NEL LUNGO PERIODO.

E non sono ridondanti, perche' LA SOMMA PUO' ESSERE GIUSTA CON I SINGOLI
SBAGLIATI E VICEVERSA:
    (A_1)_ii = 1.4, (A_2)_ii = -0.4  -> somma 1: il soc e' contento, ma i singoli
                                        sono lontani dai centri (1 e 0) e la
                                        Minnesota li penalizza;
    (A_1)_ii = 0.9, resto a zero     -> piace alla Minnesota, ma la somma e' 0.9
                                        e il soc protesta.
I due prior vincolano direzioni diverse dello spazio dei parametri.


IL SUM-OF-COEFFICIENTS: CHE COSA POSTULA
=========================================
Per ogni variabile i:

    sum_{s=1..p} (A_s)_ii = 1        (somma sui PROPRI lag)
    sum_{s=1..p} (A_s)_hi = 0        (somma sui lag ALTRUI, per ogni h != i)

Messe insieme su tutte le i, dicono che la colonna i di sum_s A_s e' il versore
e_i, cioe' in forma matriciale

    sum_{s=1..p} A_s = I_n     <=>     Pi = I_n - A_1 - ... - A_p = 0

PI E' LA MATRICE DI IMPATTO DI LUNGO PERIODO: governa cosa succede al sistema
quando lo si lascia andare.  Se Pi e' invertibile il sistema e' stazionario e
torna alla media Pi^-1 A_0; se Pi = 0 non torna da nessuna parte — ha n radici
unitarie e ogni shock e' permanente.

Un VAR in livelli con Pi = 0 E' un VAR in differenze prime.  Da qui la frase di
BGR (2010): "A VAR in first differences implies the restriction
(I_n - A_1 - ... - A_p) = 0.  We follow Doan, Litterman and Sims (1984) and set a
prior that shrinks Pi to zero.  This can be understood as INEXACT DIFFERENCING."
Non differenziamo i dati: mettiamo un prior che tira il modello verso quella
restrizione, lasciando ai dati la possibilita' di allontanarsene.

Quanto e' lontana la restrizione dai nostri dati?  Misurato: per il log-PIL la
somma dei coefficienti di un AR(5) stimato liberamente e' 0.9930, cioe'
Pi = 0.0070.  Il prior spinge verso 1 una somma che i dati mettono gia' a 0.993.


PERCHE' LIMITA LA COMPONENTE DETERMINISTICA (e perche' serve solo se NON differenzi)
====================================================================================
Un VAR stimato condizionatamente alle prime p osservazioni si scompone in una
COMPONENTE DETERMINISTICA  tau_t = E(y_t | y_1..y_p, beta_hat)  — dove il modello
dice che andrebbero le variabili se non arrivasse piu' nessuno shock — piu' la
parte guidata dagli shock.

Stimando liberamente un VAR in LOG-LIVELLI sul nostro campione (il log-PIL va da
9.23 a 10.09: trend netto), il VAR ha due modi di riprodurre quel trend:

  1. ONESTAMENTE, con Pi ~ 0: il PIL e' un random walk con drift e il trend e'
     l'accumularsi di shock permanenti;
  2. BARANDO, con Pi != 0: il sistema e' stazionario attorno a una media
     lontanissima, e il trend osservato e' il TRANSITORIO DETERMINISTICO che
     dalle condizioni iniziali del 1992 si avvicina lentamente a quella media.

Con 135 trimestri le due spiegazioni sono quasi indistinguibili IN CAMPIONE, e
la seconda spesso fitta MEGLIO perche' ha piu' liberta'.  E' la diagnosi di Sims
(1992a) che GLP cita: tau_t finisce per "explain an implausibly high share of the
variation", con "temporal heterogeneity" — comportamento molto diverso a inizio
e a fine campione.

Il guaio non si vede in-sample: si vede IN PREVISIONE.  Un modello che ha
spiegato il trend come transitorio deterministico, appena lo proietti in avanti,
fa tornare tutto verso una media inventata.

Il sum-of-coefficients chiude questa strada: spingendo Pi verso 0 toglie al
modello la possibilita' di attribuire il trend a un transitorio deterministico.

E QUI SI CHIUDE IL CERCHIO COL GATE 0.  Se avessimo differenziato, il problema
non esisterebbe — ma avremmo buttato via l'informazione sui livelli e ogni
possibilita' di cointegrazione.  La scelta di Cimadomo e': TIENI I LIVELLI E
GOVERNA LA NON-STAZIONARIETA' CON I PRIOR.  La Minnesota lo fa centrando su un
random walk; il soc lo fa spingendo Pi verso zero.  Sono le due gambe della
stessa scelta, e LA SECONDA ESISTE SOLO PERCHE' NON ABBIAMO DIFFERENZIATO: su
dati differenziati il soc assumerebbe una radice unitaria nelle differenze,
cioe' I(2).


MU: I CASI LIMITE, E IL VERSO
==============================
In GLP mu compare al DENOMINATORE: y+ = diag(y0_bar / mu).  Coi nostri numeri
(y0_bar = media delle prime p=5 osservazioni):

                  GDPC1   ISM_PMI  UNRATE
    y0_bar        9.253    53.12    7.44
    mu = 0.2  ->  46.26   265.6    37.2     dummy grande  -> prior FORTE
    mu = 1.0  ->   9.253   53.12    7.44    dummy ~ dati  -> prior moderato
    mu = 5.0  ->   1.851   10.62    1.488   dummy piccola -> prior DEBOLE

Il meccanismo: una dummy con valori grandi ha molta LEVA nella regressione,
quindi vincola forte; una dummy piccola conta poco.

    mu -> 0     dummy esplode  -> prior DOGMATICO: Pi = 0 esatto, radice unitaria
                                  in ogni equazione, "rules out cointegration"
    mu -> inf   dummy svanisce -> prior INESISTENTE: resta la sola Minnesota

IL VERSO E' LO STESSO DI LAMBDA: iperparametro piccolo = prior stretto.  Comodo
da ricordare, perche' mu al denominatore suggerirebbe il contrario.

I valori del paper (Tab. B.1: mu ~ 0.97-1.72) cadono dove la dummy ha grandezza
confrontabile con i dati, cioe' "vale piu' o meno un'osservazione in piu' per
variabile": un prior presente ma non prepotente.

IL NOSTRO NUMERO, misurato sul profilo q_b (Gate 1):

    mu ~ 1.10   [90%: 0.975, 1.272]      banda del paper (Tab. B.1): 0.97-1.72

cioe' in pieno centro della banda: il soc e' presente e non prepotente.


LA TENSIONE wn <-> soc  —  RISOLTA, con la riga degli autori
=============================================================
Impilando Minnesota e soc c'era un disaccordo GENUINO sulle serie centrate su
white noise, e vale la pena raccontarlo per intero perche' e' costato una
decisione.

    la Minnesota dice   sum_s (A_s)_ii = d_centre_i   ->  0 per le survey
    il soc, COM'ERA     sum_s (A_s)_ii = 1            ->  per TUTTE le variabili

MISURATO PRIMA DEL FIX (test_dummies §5, mu = 1.0): sul centro implicito dello
stack le somme dei propri lag delle serie wn venivano tirate da 0 a

    ISM_PMI 0.996   ISM_PRICES 1.000   ISM_EMP 1.000   Philly 1.000

cioe' SULLE SERIE WHITE-NOISE IL PRIOR NETTO NON ERA WHITE NOISE, ERA
PRATICAMENTE RANDOM WALK: il soc vinceva e la scelta `wn` del Gate 0 veniva
annullata.

LA SOLUZIONE, che sta nel codice di replica degli autori e non nel paper:

    ydnoc = (1/miu)*diag(y0);   ydnoc(pos,pos) = 0;     % logMLVAR_formin.m

Sulle serie stazionarie il lato sinistro della dummy e' azzerato, quindi il soc
tira la somma dei propri lag verso 0 invece che verso 1.  I due prior parlano
entrambi di persistenza e ora dicono la stessa cosa.  Dopo il fix le somme wn
misurate sono 0.000 su tutte e quattro, e le rw restano a 1 (scarto 0.0000).

PERCHE' NON E' UNA DEVIAZIONE.  Il punto era stato riservato al relatore proprio
perche' SEMBRAVA una deviazione dal paper da dichiarare.  Non lo e': il paper
tace — ne' §2.1 ne' la nota 13 lo menzionano — ma il codice degli autori e'
inequivocabile.  Dove testo e codice divergono, per la replica vince il codice.

L'IMPATTO NON E' COSMETICO.  Ristimando il Q-BVAR sul profilo q_b:

                prima del fix              dopo
    lambda   0.4875 [0.443, 0.536]   0.5182 [0.490, 0.559]
    mu       1.1983 [1.029, 1.414]   2.2508 [1.848, 2.854]

mu QUASI RADDOPPIA.  Il verso conta: mu sta al DENOMINATORE della dummy, quindi
mu piu' grande = soc piu' DEBOLE.  Tolto il conflitto, i dati chiedono un soc
molto meno intenso — segno che prima parte della sua forza serviva a vincere una
battaglia che non avrebbe dovuto combattere.

CONSEGUENZA DA RICORDARE: `d_centre` governa ORA DUE assi, non uno — il centro
del primo lag (Minnesota) e il bersaglio della somma dei lag (soc).  Cambiare
una serie da rw a wn nella config li muove entrambi.


CAVEAT SULLA COINTEGRAZIONE, da tenere e da guardare al Blocco 5.  Pi = 0
significa rank(Pi) = 0, mentre la cointegrazione richiede 0 < rank(Pi) < n.  Il
soc spinto al limite la esclude — ed e' esattamente il motivo per cui Sims (1993)
aggiunse il dummy-initial-observation, che invece con la cointegrazione e'
compatibile.  E' il delta che abbiamo deciso di lasciare FUORI (Decisione 4, per
fedelta' a Cimadomo §2.1).  Con mu ~ 1 il prior e' abbastanza lasco da non
imporlo davvero, ma se il mu stimato uscisse molto basso il caveat diventerebbe
sostanziale.


IL MECCANISMO DELLE DUMMY OBSERVATIONS
=======================================
L'IDEA.  Un prior gaussiano su coefficienti di regressione e' OSSERVAZIONALMENTE
EQUIVALENTE ad avere osservato dei dati in piu'.  Invece di scrivere il prior
come distribuzione, si scrivono righe fittizie (Yd, Xd) e le si impila sopra i
dati veri,

    Y* = [Yd ; Y]        X* = [Xd ; X]

e si stima COME SE fossero tutti dati veri.  E' la "Theil mixed estimation" che
nomina GLP.

PERCHE' FUNZIONA.  Una riga fittizia yd = xd B + u e' un'affermazione: "esiste
un'osservazione in cui questa combinazione di coefficienti da' questo valore".
La stima cerca di soddisfarla come qualunque altra osservazione, e quanto ci
tiene dipende dalla GRANDEZZA della riga: valori grandi = molta leva = vincolo
forte.  LA MAGNITUDINE DELLA DUMMY E' L'INTENSITA' DEL PRIOR.  E' per questo che
mu e lambda compaiono come DIVISORI nelle formule, e non come pesi separati.

PERCHE' USARLO INVECE DI SCRIVERE IL PRIOR DIRETTAMENTE.  Tre ragioni pratiche:
  1. IL SOC NON HA UNA FORMA COMODA COME OMEGA.  La Minnesota si' (Omega
     diagonale, Blocco 2).  Il soc vincola SOMME, e questo introduce
     CORRELAZIONI fra i coefficienti di lag diversi della stessa variabile — GLP:
     "It also introduces correlation among the coefficients on each variable in
     each equation."  Come matrice Omega piena k x k sarebbe scomodo e costoso;
     come n righe di dummy e' banale.
  2. LA CONIUGATEZZA SI CONSERVA GRATIS.  Impilare righe e rifare il posterior
     NIW da' automaticamente un posterior NIW: nessuna matematica nuova rispetto
     al Blocco 1.
  3. REGOLARIZZA NUMERICAMENTE.  BGR: "Adding dummy observations works as a
     regularisation solution to the matrix inversion problem."  Con k=151 e
     T=130, X'X e' singolare; X*'X* no.

COME SI TORNA INDIETRO.  Da un blocco di dummy si recuperano i momenti del prior
che esso implica, con le identita' di BGR:

    B0 = (Xd' Xd)^-1 Xd' Yd        Omega0 = (Xd' Xd)^-1

E' lo strumento che al Blocco 4 usiamo come ORACOLO (vedi la sezione "PERCHE'
IMPLEMENTARE LA STESSA COSA DUE VOLTE" in cima).

AVVERTENZA SULL'ORDINE.  GLP mette la costante PRIMA (x+ = [0, y+, ..., y+]);
BGR e noi la mettiamo IN FONDO.  Il nostro blocco soc e' quindi
[y+, ..., y+, 0].  Lo zero nella colonna della costante dice che IL SOC NON
AFFERMA NULLA SULLA COSTANTE — coerente col suo prior piatto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bvar.spec import BVARSpec, Hyper

#: L'`eps` del blocco della costante di BGR eq. (5).  NON E' FISSATO DA NESSUNO
#: DEI SEI PAPER — BGR dice solo "eps is a very small number".  Scelta nostra:
#: da' varianza a priori 1/eps^2 = 1e8 sulla costante, piattissima rispetto a
#: dati in log-livelli dell'ordine di 10, senza rovinare il condizionamento di
#: X*'X*.  Vedi `constant_dummy`.
DEFAULT_CONST_EPS: float = 1e-4

#: Il prior sulla costante e' PIATTO (Cimadomo §2.1: "For the constant A0 term,
#: we use a flat prior"), cioe' varianza infinita.  Nella forma analitica la
#: rappresentiamo come `np.inf`, che e' la verita' matematica.  Le dummy del
#: Blocco 4 la approssimeranno con `1/eps**2` per un eps piccolo, che e' il
#: trucco di BGR eq. (5) — l'ultima riga di Xd, `[0_{1 x np}, eps]`.
CONST_PRIOR_VAR = np.inf


# ─── 1. La mappa dei regressori ───────────────────────────────────────────────

def regressor_layout(spec: BVARSpec) -> list[tuple[int, str]]:
    """
    L'ordine delle colonne di X, che e' anche l'ordine delle righe di B.

    Convenzione LAG-MAJOR con la COSTANTE IN FONDO, presa da BGR (2010):

        x_t = (x'_{t-1}, ..., x'_{t-p}, 1)'

    quindi il regressore (lag s, variabile j) sta all'indice (s-1)*n + j, e la
    costante all'indice n*p.  Non e' una convenzione arbitraria: deve combaciare
    con le dummy di BGR eq. (5), dove il blocco `J_p (x) diag(sigma)` con
    `J_p = diag(1,...,p)` e' organizzato esattamente per lag.

    Returns
    -------
    list[tuple[int, str]]
        Lunga k = n*p+1.  Ogni voce e' `(lag, series_id)`; la costante e'
        `(0, '<const>')`.
    """
    out: list[tuple[int, str]] = []
    for s in range(1, spec.p + 1):
        out.extend((s, sid) for sid in spec.series)
    out.append((0, "<const>"))
    return out


# ─── 2. Le medie a priori, eq. (2) ────────────────────────────────────────────

def minnesota_prior_mean(spec: BVARSpec) -> np.ndarray:
    """
    La matrice `b` (k x n) delle medie a priori dei coefficienti — eq. (2).

    Struttura: tutto zero, TRANNE la diagonale del blocco del primo lag, che
    porta `d_centre`.

        b[(1-1)*n + i, i] = d_centre[i]        i = 0, ..., n-1

    Cioe': se il prior vincesse del tutto, l'equazione della variabile i
    sarebbe  x_{i,t} = d_i * x_{i,t-1} + eps_{i,t}  — un random walk se
    d_i = 1, un white noise se d_i = 0.  E' l'ipotesi nulla di ciascuna serie.

    La riga della costante resta zero: non perche' crediamo che sia zero, ma
    perche' il suo prior e' piatto e la media di un prior piatto non entra
    nella stima (vedi CONST_PRIOR_VAR).

    Notare che `d_centre` NON e' cablato qui: viene da
    config/bvar_series.json tramite `BVARSpec.from_config`.
    """
    b = np.zeros((spec.k, spec.n), dtype=float)
    b[: spec.n, :][np.diag_indices(spec.n)] = spec.minnesota.d
    return b


# ─── 3. Le varianze a priori, eq. (3) ─────────────────────────────────────────

def minnesota_omega_diag(
    spec: BVARSpec,
    hyper: Hyper,
    *,
    const_var: float = CONST_PRIOR_VAR,
) -> np.ndarray:
    """
    La diagonale di `Omega` (k,) — eq. (3).

        Omega[(s,j)] = lambda^2 / (s^lag_decay * psi_j)
        Omega[const] = const_var  (infinito: prior piatto)

    Omega e' DIAGONALE perche' l'eq. (3) e' non nulla solo quando (r,m)=(s,j).
    La teniamo quindi come vettore di lunghezza k e non come matrice k x k: a
    k=151 (Q-BVAR) e k=421 (B-BVAR) non e' un dettaglio di stile.

    ATTENZIONE ALL'INDICE: al denominatore c'e' `psi_j`, la scala della
    VARIABILE RITARDATA, non `psi_i` dell'equazione.  Vedi la sezione
    "IL REFUSO DELL'EQ. (3)" nell'header del modulo.

    Parameters
    ----------
    const_var : float
        Varianza a priori della costante.  Default `np.inf` (prior piatto,
        la verita' matematica).  Passare un valore finito serve solo ai test
        che devono confrontare Omega con l'inversa di una matrice.
    """
    if hyper.n != spec.n:
        raise ValueError(
            f"hyper.psi ha lunghezza {hyper.n} ma lo spec ha n={spec.n}"
        )
    lam2 = hyper.lam ** 2
    psi = hyper.psi                       # (n,)  scala della variabile RITARDATA
    lags = np.arange(1, spec.p + 1, dtype=float)          # s = 1..p
    decay = lags ** spec.minnesota.lag_decay              # s^2  (nota 8)

    # blocco (p, n): riga s, colonna j  ->  lambda^2 / (s^decay * psi_j)
    block = lam2 / (decay[:, None] * psi[None, :])
    return np.concatenate([block.ravel(), [const_var]])


# ─── 4. Diagnostica: le deviazioni standard, in chiaro ────────────────────────

def prior_coefficient_sd(spec: BVARSpec, hyper: Hyper) -> np.ndarray:
    """
    Deviazioni standard a priori dei coefficienti (k x n), in unita' dei dati.

    E' una funzione DIAGNOSTICA, non un pezzo del percorso di stima: serve a
    guardare in faccia i numeri che il prior sta imponendo.

        sd[(s,j), i] = (lambda / s^(decay/2)) * sqrt(psi_i / psi_j)

    ottenuta da  Var = Sigma_ii * Omega[(s,j)]  usando il risultato del
    Blocco 1 che con dof = n+2 la media a priori di Sigma e' esattamente
    Psi = diag(psi), quindi Sigma_ii ~ psi_i.

    DUE LETTURE CHE VALE LA PENA AVERE SOTTO GLI OCCHI
    --------------------------------------------------
    1. `sd[(1,i), i] = lambda` esattamente.  Cioe' LAMBDA E' LA DEVIAZIONE
       STANDARD A PRIORI DEL COEFFICIENTE DI OGNI VARIABILE SU SE STESSA AL
       PRIMO LAG.  E' il modo piu' concreto di leggere la banda della Tabella
       B.1: lambda ~ 0.6 vuol dire "a priori quel coefficiente e' 1 +/- 0.6".
    2. Il fattore sqrt(psi_i/psi_j) e' un RAPPORTO DI SCALE, e ha
       un'interpretazione dimensionale esatta: quel coefficiente converte
       unita' di j in unita' di i, quindi la sua grandezza naturale e' il
       rapporto fra le due scale.  E' cio' che permette al prior di funzionare
       su dati NON standardizzati — e il motivo per cui standardizzare a monte
       avrebbe distrutto l'informazione sulle scale relative.

    La riga della costante e' `inf` (prior piatto).
    """
    omega = minnesota_omega_diag(spec, hyper)             # (k,)
    sigma_ii = hyper.psi                                  # (n,)  E[Sigma] = Psi
    with np.errstate(invalid="ignore"):
        return np.sqrt(omega[:, None] * sigma_ii[None, :])


# ─── 5. La scala del sum-of-coefficients: y0_bar ──────────────────────────────

def initial_observation_mean(panel, p: int) -> np.ndarray:
    """
    `y0_bar`: la media delle PRIME p osservazioni, variabile per variabile.

    E' la scala su cui il sum-of-coefficients costruisce le proprie dummy.
    GLP §III: "where y0_bar is an n x 1 vector containing the average of the
    first p observations for each variable".

    PERCHE' LE PRIME p E NON LA MEDIA DI TUTTO IL CAMPIONE
    ------------------------------------------------------
    Perche' il prior e' un'affermazione sulle CONDIZIONI INIZIALI: GLP lo
    descrive come "stating that a no-change forecast is a good forecast AT THE
    BEGINNING OF THE SAMPLE".  Il soc serve proprio a limitare il transitorio
    deterministico che parte dalle condizioni iniziali (vedi la sezione
    "PERCHE' LIMITA LA COMPONENTE DETERMINISTICA" nell'header), quindi la scala
    naturale e' quella del punto di partenza, non del campione intero.

    E c'e' una ragione statistica in piu': usare la media dell'intero campione
    farebbe dipendere il PRIOR dai dati che il prior deve poi giudicare — le
    prime p osservazioni sono invece esattamente quelle su cui il VAR condiziona
    e che non entrano nella verosimiglianza (l'inferenza usa y_{p+1}, ..., y_T).

    Parameters
    ----------
    panel : (T, n) array o DataFrame
        Il pannello in unita' del modello, gia' denso.
    p : int
        Numero di lag del modello.

    Returns
    -------
    np.ndarray, shape (n,)
    """
    arr = np.asarray(getattr(panel, "to_numpy", lambda: panel)(), dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"panel deve essere 2-D (T, n), ricevuto {arr.shape}")
    if arr.shape[0] < p:
        raise ValueError(
            f"servono almeno p={p} osservazioni per y0_bar, ce ne sono {arr.shape[0]}"
        )
    head = arr[:p]
    if np.isnan(head).any():
        bad = int(np.isnan(head).sum())
        raise ValueError(
            f"le prime {p} righe del pannello contengono {bad} NaN: y0_bar non e' "
            f"definita.  Il core non deve mai vedere un NaN — vedi "
            f"src/bvar/data.py::assert_dense."
        )
    return head.mean(axis=0)


# ─── 6. Le dummy del sum-of-coefficients ──────────────────────────────────────

def sum_of_coefficients_dummy(
    spec: BVARSpec,
    hyper: Hyper,
    y0_bar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Le n righe fittizie del sum-of-coefficients — GLP §III.

        y+ (n x n)      = diag(y0_bar / mu)
        x+ (n x k)      = [y+, y+, ..., y+, 0_{n x 1}]      (p copie di y+)

    ORDINE DELLE COLONNE.  GLP scrive x+ = [0_{n x 1}, y+, ..., y+], con la
    COSTANTE PRIMA.  Noi la teniamo IN FONDO, coerentemente con BGR e con
    `regressor_layout`.  Lo zero nella colonna della costante dice che il soc
    NON AFFERMA NULLA sulla costante — coerente col suo prior piatto.

    CHE COSA AFFERMA, RIGA PER RIGA
    -------------------------------
    La riga i ha `y0_bar_i/mu` in posizione i (e zero altrove) sia in y+ sia in
    ciascuna delle p copie dentro x+.  Letta come osservazione `y = x B`, per
    l'equazione h dice:

        h = i :   y0_i/mu = (y0_i/mu) * sum_s (A_s)_ii   =>  sum_s (A_s)_ii = d_i
        h != i:        0  = (y0_i/mu) * sum_s (A_s)_hi   =>  sum_s (A_s)_hi = 0

    cioe' il postulato di Cimadomo §2.1 — "the sum of the coefficients
    associated with the own lags of each variable equals one, while the sum of
    the coefficients associated with the lags of the other variables equals
    zero" — MA CON d_i AL POSTO DI 1 SULLE SERIE WHITE-NOISE.  Vedi sotto.


    LE RIGHE white-noise SONO AZZERATE  (`ydnoc(pos,pos) = 0`)
    ==========================================================
    E' la sottigliezza piu' importante di questa funzione, e nel paper NON C'E'.
    Sta nel codice di replica degli autori, `fromGLP/logMLVAR_formin.m`:

        if noc==1;
            ydnoc = (1/miu)*diag(y0);   ydnoc(pos,pos) = 0;
            xdnoc = [zeros(n,1)  (1/miu)*repmat(diag(y0),1,lags)];
        end

    dove `pos` sono le serie STAZIONARIE — le stesse che nel Minnesota ricevono
    il centraggio white-noise.  Per quelle il lato sinistro della dummy e'
    azzerato, quindi la dummy impone `sum_s (A_s)_ii = 0` invece di `= 1`.
    `xdnoc` NON viene toccata: si azzera solo `y+`.

    PERCHE' E' LA COSA GIUSTA.  Minnesota e sum-of-coefficients parlano
    ENTRAMBI di persistenza, su oggetti diversi:

        Minnesota                sul PRIMO lag        (A_1)_ii  -> d_i
        sum-of-coefficients      sulla SOMMA dei lag  sum_s (A_s)_ii -> ?

    Se il soc dicesse 1 anche dove il Minnesota dice 0, i due prior si
    contraddirebbero proprio sulle serie in cui la scelta conta.  Azzerando le
    righe wn i due assi si muovono insieme: per una survey mean-reverting
    entrambi tirano verso zero.

    NOTA STORICA, tenuta perche' spiega un passaggio del progetto.  Una versione
    precedente di questa funzione applicava il soc UNIFORMEMENTE, e l'header
    documentava la contraddizione come "tensione wn<->soc, PUNTO APERTO":
    misurato, sul centro implicito le somme dei propri lag delle serie wn
    venivano tirate da 0 a ~1.00 (ISM_PMI 0.996, ISM_PRICES 1.000, ISM_EMP
    1.000, Philly 1.000), annullando in pratica la scelta `wn` del Gate 0.  Il
    punto era riservato al relatore perche' cambiarlo SEMBRAVA una deviazione
    dal paper.  Non lo e': e' cio' che fanno gli autori.  Il paper tace, il
    codice no.

    ATTENZIONE: `d_centre` governa quindi DUE cose, non una.  Cambiare una serie
    da rw a wn nella config sposta sia il centro del primo lag sia la somma dei
    lag.  E' voluto.

    NOTA: la dummy e' soddisfatta ESATTAMENTE (residuo nullo) da qualunque B con
    sum_s A_s = I, INDIPENDENTEMENTE dal valore di mu e di y0_bar.  mu e y0_bar
    non spostano il PUNTO verso cui si tira, solo la FORZA con cui si tira — e'
    il meccanismo "la magnitudine della dummy e' l'intensita' del prior"
    descritto nell'header.  Il test lo verifica.

    Parameters
    ----------
    y0_bar : np.ndarray, shape (n,)
        Da `initial_observation_mean`.

    Returns
    -------
    (Yd, Xd) : (n, n) e (n, k)
    """
    y0 = np.asarray(y0_bar, dtype=float)
    if y0.shape != (spec.n,):
        raise ValueError(
            f"y0_bar deve avere shape ({spec.n},), ricevuto {y0.shape}"
        )
    if hyper.mu <= 0:
        raise ValueError(f"mu deve essere > 0, ricevuto {hyper.mu}")

    y_plus = np.diag(y0 / hyper.mu)                      # (n, n)

    Xd = np.zeros((spec.n, spec.k), dtype=float)
    for s in range(spec.p):                              # p copie, lag-major
        Xd[:, s * spec.n: (s + 1) * spec.n] = y_plus
    # l'ultima colonna (la costante) resta zero: il soc tace sulla costante

    # `ydnoc(pos,pos) = 0`: sulle serie white-noise il soc tira la somma dei
    # propri lag verso 0, non verso 1.  Solo Yd; Xd resta intatta.
    Yd = y_plus.copy()
    wn = spec.minnesota.d == 0.0
    Yd[wn, wn] = 0.0
    return Yd, Xd


# ─── 7. Il meccanismo, al contrario: da dummy a momenti ───────────────────────

def implied_prior_moments(
    Yd: np.ndarray,
    Xd: np.ndarray,
    *,
    rcond: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    I momenti del prior IMPLICATI da un insieme di dummy observations — BGR (2010).

        B0     = (Xd' Xd)^-1 Xd' Yd
        Omega0 = (Xd' Xd)^-1

    BGR: "It can be shown that adding Td dummy observations Yd and Xd to the
    system is equivalent to imposing the Normal inverted Wishart prior with
    B0 = (Xd'Xd)^-1 Xd'Yd, Omega0 = (Xd'Xd)^-1, ..."

    E' il MECCANISMO DELLE DUMMY letto al contrario, ed e' lo strumento con cui
    al Blocco 4 verifichiamo che le dummy della Minnesota riproducano davvero i
    momenti analitici delle eq. (2)-(3) calcolati al Blocco 2.

    PERCHE' NON FUNZIONA SU UN BLOCCO ISOLATO
    -----------------------------------------
    Un singolo blocco di dummy non identifica tutti i k coefficienti: il blocco
    soc, per esempio, ha n righe per k = n*p+1 colonne, quindi Xd'Xd e'
    singolare (rango <= n << k).  Non e' un difetto: I BLOCCHI DI DUMMY
    IDENTIFICANO IL PRIOR SOLO INSIEME — il blocco Minnesota vincola i singoli
    coefficienti, il soc le somme, quello della costante l'intercetta.

    Questa funzione quindi SOLLEVA UN ERRORE se Xd'Xd e' singolare, invece di
    restituire silenziosamente una pseudo-inversa: una pseudo-inversa darebbe
    numeri plausibili e sbagliati, che e' il modo peggiore di fallire.

    Raises
    ------
    np.linalg.LinAlgError
        Se Xd'Xd non e' invertibile, con un messaggio che spiega perche'.
    """
    Yd = np.asarray(Yd, dtype=float)
    Xd = np.asarray(Xd, dtype=float)
    if Yd.shape[0] != Xd.shape[0]:
        raise ValueError(
            f"Yd e Xd devono avere lo stesso numero di righe, "
            f"ricevuto {Yd.shape[0]} e {Xd.shape[0]}"
        )

    xtx = Xd.T @ Xd
    k = xtx.shape[0]
    rank = int(np.linalg.matrix_rank(xtx, tol=rcond * max(1.0, float(np.abs(xtx).max()))))
    if rank < k:
        raise np.linalg.LinAlgError(
            f"Xd'Xd e' singolare (rango {rank} su {k}): queste dummy da sole non "
            f"identificano il prior.  I blocchi di dummy identificano il prior solo "
            f"INSIEME — un blocco isolato (es. il solo sum-of-coefficients, che ha "
            f"{Xd.shape[0]} righe per {k} colonne) non basta.  Impila i blocchi e "
            f"richiama su tutto lo stack."
        )

    omega0 = np.linalg.inv(xtx)
    b0 = omega0 @ (Xd.T @ Yd)
    return b0, omega0


# ─── 8. I tre blocchi di BGR eq. (5) ──────────────────────────────────────────

def minnesota_dummy(spec: BVARSpec, hyper: Hyper) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Il PRIMO blocco di BGR eq. (5): il prior sui coefficienti autoregressivi.

        Yd = [ diag(delta_1 sigma_1, ..., delta_n sigma_n) / lambda ]   (n righe)
             [ 0_{n(p-1) x n}                                      ]
        Xd = [ J_p (x) diag(sigma_1, ..., sigma_n) / lambda , 0_{np x 1} ]

    con J_p = diag(1, 2, ..., p), delta = d_centre, sigma_i = sqrt(psi_i).
    Sono np righe.

    PERCHE' RIPRODUCE ESATTAMENTE LE EQ. (2)-(3)
    --------------------------------------------
    Xd e' DIAGONALE, con elemento (s,j) pari a s*sigma_j/lambda.  Quindi

        (Xd' Xd)_{(s,j)} = s^2 psi_j / lambda^2
        Omega_{(s,j)}    = lambda^2 / (s^2 psi_j)      <- eq. (3), psi_j incluso

    e sulla riga (1,i)

        B0_{(1,i),i} = Omega_{(1,i)} * (sigma_i/lambda) * (delta_i sigma_i/lambda)
                     = (lambda^2/psi_i) * (delta_i psi_i / lambda^2) = delta_i

    cioe' l'eq. (2).  Si vede anche PERCHE' la struttura e' quella: il fattore s
    di J_p produce il decadimento 1/s^2 nella varianza (Omega e' l'inversa di un
    quadrato), e il fattore sigma_j produce lo scaling 1/psi_j.  Il prior e'
    codificato nella GEOMETRIA delle righe fittizie.

    SIGMA vs PSI.  BGR scrive sigma_i, Cimadomo e GLP scrivono psi_i: e' la
    stessa cosa a meno della radice, sigma_i = sqrt(psi_i), perche' psi_i e' la
    varianza attesa a priori del residuo (Blocco 1, con dof = n+2).  La
    differenza SOSTANZIALE fra i due paper e' che BGR FISSA sigma (residui di un
    AR(p) univariato) mentre GLP lo CAMPIONA.  Noi seguiamo GLP: psi arriva da
    `hyper`, che al Blocco 5 sara' un'estrazione del Metropolis.
    """
    n, p, k = spec.n, spec.p, spec.k
    sigma = np.sqrt(hyper.psi)                      # (n,)
    lam = hyper.lam

    Yd = np.zeros((n * p, n), dtype=float)
    Yd[:n, :] = np.diag(spec.minnesota.d * sigma / lam)

    Xd = np.zeros((n * p, k), dtype=float)
    for s in range(1, p + 1):                       # J_p = diag(1, ..., p)
        lo, hi = (s - 1) * n, s * n
        Xd[lo:hi, lo:hi] = np.diag(s * sigma / lam)
    # l'ultima colonna (costante) resta zero
    return Yd, Xd


def covariance_dummy(spec: BVARSpec, hyper: Hyper) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Il SECONDO blocco di BGR eq. (5): il prior sulla matrice di covarianza.

        Yd = diag(sigma_1, ..., sigma_n)        (n righe)
        Xd = 0_{n x k}

    Righe con regressori TUTTI NULLI: sembra strano, ed e' esattamente il punto.
    Poiche' Xd = 0, questo blocco NON contribuisce ne' a Xd'Xd ne' a Xd'Yd,
    quindi NON tocca b ne' Omega.  Contribuisce solo al residuo:

        S0 <- (Yd - Xd B0)'(Yd - Xd B0) = diag(sigma)' diag(sigma)
                                        = diag(psi) = Psi

    Cioe': queste righe dicono "ho osservato residui di questa grandezza" e
    nient'altro.  Sono il modo di iniettare Psi nello stack senza disturbare il
    prior sui coefficienti.
    """
    sigma = np.sqrt(hyper.psi)
    return np.diag(sigma), np.zeros((spec.n, spec.k), dtype=float)


def constant_dummy(spec: BVARSpec, eps: float) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Il TERZO blocco di BGR eq. (5): il prior (piatto) sull'intercetta.

        Yd = 0_{1 x n}
        Xd = [0_{1 x np}, eps]                  (1 riga)

    Da' (Xd'Xd)_const = eps^2, quindi Omega_const = 1/eps^2: con eps piccolo la
    varianza a priori sulla costante e' enorme, cioe' e' L'APPROSSIMAZIONE
    NUMERICA DEL PRIOR PIATTO di Cimadomo §2.1 ("For the constant A0 term, we use
    a flat prior").  Non si puo' scrivere "varianza infinita" dentro una matrice;
    si scrive una varianza enorme.

    IL VALORE DI eps NON E' FISSATO DA NESSUNO DEI SEI PAPER.  BGR dice soltanto
    "eps is a very small number".  E' quindi una scelta NOSTRA, e va dichiarata
    come tale.  Il trade-off: eps piccolo = prior piu' piatto (voluto) ma
    X*'X* peggio condizionata (non voluto).  Coi nostri dati in log-livelli
    dell'ordine di 10, una varianza a priori di 1/eps^2 = 1e8 sulla costante e'
    gia' piattissima in rapporto alla scala.  Default DEFAULT_CONST_EPS = 1e-4.
    """
    if eps <= 0:
        raise ValueError(f"eps deve essere > 0, ricevuto {eps}")
    Xd = np.zeros((1, spec.k), dtype=float)
    Xd[0, -1] = eps
    return np.zeros((1, spec.n), dtype=float), Xd


# ─── 9. Lo stack completo ─────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class DummyStack:
    """
    Lo stack completo delle dummy observations, pronto per essere impilato sopra
    i dati veri.

    Attributes
    ----------
    Yd : (Td, n)
    Xd : (Td, k)
    blocks : dict[str, slice]
        Dove comincia e finisce ogni blocco, per diagnostica e test.
    dof : int
        I gradi di liberta' del prior su Sigma.  **Vale n+2 ed e' FISSATO, non
        dedotto dal numero di righe** — vedi la nota qui sotto.

    LA CONTABILITA' DEI GRADI DI LIBERTA': UNA TRAPPOLA
    ----------------------------------------------------
    BGR scrive `alpha0 = Td - k`.  Sul LORO stack (np + n + 1 righe, k = np+1)
    fa `n`; poi aggiungono il prior improprio |Psi|^-(n+3)/2, che porta il
    posterior a `Td + 2 + T - k` = `n + 2 + T`.  Torna con d = n+2.

    MA se si aggiunge il blocco sum-of-coefficients (altre n righe), quella
    formula darebbe 2n+2 — SBAGLIATO.

    La risoluzione sta in GLP Appendix A, eq. (A.8): il posterior e'

        Sigma | Y ~ IW( Psi + eps'eps + (B-b)' Omega^-1 (B-b),  T - p + d )

    dove `d` e' un PARAMETRO DEL PRIOR che si fissa, non si conta.  Le dummy
    servono a costruire b, Omega e Psi; `d = n+2` viene dalla scelta di
    Kadiyala-Karlsson discussa al Blocco 1 (il minimo che da' media a priori
    finita).  Il numero di righe fittizie e' IRRILEVANTE per d.

    Quindi qui `dof` e' impostato direttamente a n+2, e la coincidenza
    `Td - k + 2 = n + 2` vale come CONTROLLO INCROCIATO sullo stack senza soc,
    non come definizione.
    """

    Yd: np.ndarray
    Xd: np.ndarray
    blocks: dict
    dof: int

    @property
    def Td(self) -> int:
        return int(self.Yd.shape[0])

    def block(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Le righe di un singolo blocco, per ispezione."""
        sl = self.blocks[name]
        return self.Yd[sl], self.Xd[sl]

    def without(self, *names: str) -> "DummyStack":
        """
        Lo stesso stack senza i blocchi indicati.

        Serve al test dell'oracolo: lo stack SENZA il sum-of-coefficients deve
        riprodurre esattamente i momenti analitici delle eq. (2)-(3), mentre lo
        stack CON il soc no — e non deve, perche' il soc e' un prior in piu'.
        """
        keep = [nm for nm in self.blocks if nm not in names]
        Yd = np.vstack([self.Yd[self.blocks[nm]] for nm in keep])
        Xd = np.vstack([self.Xd[self.blocks[nm]] for nm in keep])
        out, pos = {}, 0
        for nm in keep:
            h = self.blocks[nm].stop - self.blocks[nm].start
            out[nm] = slice(pos, pos + h)
            pos += h
        return DummyStack(Yd=Yd, Xd=Xd, blocks=out, dof=self.dof)


def build_dummies(
    spec: BVARSpec,
    hyper: Hyper,
    y0_bar: np.ndarray | None = None,
    *,
    eps: float = DEFAULT_CONST_EPS,
) -> DummyStack:
    """
    Assembla lo stack completo: BGR eq. (5) piu' il sum-of-coefficients.

    Ordine dei blocchi (l'ordine delle righe e' irrilevante per il posterior —
    conta solo l'insieme — ma tenerlo fisso rende i test leggibili):

        'minnesota'   np righe   prior sui coefficienti autoregressivi
        'covariance'   n righe   prior su Sigma  (inietta Psi)
        'constant'     1 riga    prior piatto sull'intercetta
        'soc'          n righe   sum-of-coefficients   (se y0_bar e' dato)

    Parameters
    ----------
    y0_bar : (n,) o None
        Media delle prime p osservazioni, da `initial_observation_mean`.
        Se None il blocco soc viene OMESSO — utile per i test dell'oracolo e
        per isolare l'effetto del soc, NON per la stima vera (Cimadomo §2.1 usa
        sempre Minnesota + sum-of-coefficients).
    eps : float
        Il numero piccolo del blocco della costante.  Scelta nostra, non del
        paper: vedi `constant_dummy`.

    Returns
    -------
    DummyStack
    """
    parts: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = [
        ("minnesota", minnesota_dummy(spec, hyper)),
        ("covariance", covariance_dummy(spec, hyper)),
        ("constant", constant_dummy(spec, eps)),
    ]
    if y0_bar is not None:
        parts.append(("soc", sum_of_coefficients_dummy(spec, hyper, y0_bar)))

    blocks, pos = {}, 0
    for name, (Yb, _) in parts:
        h = Yb.shape[0]
        blocks[name] = slice(pos, pos + h)
        pos += h

    Yd = np.vstack([Yb for _, (Yb, _) in parts])
    Xd = np.vstack([Xb for _, (_, Xb) in parts])
    return DummyStack(Yd=Yd, Xd=Xd, blocks=blocks, dof=default_dof(spec.n))


def default_dof(n: int) -> int:
    """
    d = n + 2: i gradi di liberta' del prior Inverse-Wishart su Sigma.

    Cimadomo §2.1: "d = n + 2 degrees of freedom, which is the minimum number of
    degrees of freedom that guarantees the existence of the prior mean of Sigma
    (equal to Psi/(d-n-1) = Psi)".  Kadiyala & Karlsson (1997) via GLP §III.

    E' il punto in cui la media a priori esiste ma la VARIANZA no (che
    richiederebbe d > n+3): il prior piu' diffuso possibile che sia ancora
    ancorato.  Vedi Blocco 1.

    NON si deduce dal numero di righe di dummy: vedi la nota in DummyStack.
    """
    return n + 2


# ─── 10. La verifica: da stack a momenti impliciti ────────────────────────────

def implied_prior_scale(Yd: np.ndarray, Xd: np.ndarray) -> np.ndarray:
    """
    La matrice di scala Psi implicata dalle dummy — BGR:

        S0 = (Yd - Xd B0)' (Yd - Xd B0)     con  B0 = (Xd'Xd)^-1 Xd'Yd

    Sullo stack completo (senza soc) deve venire esattamente `diag(psi)`,
    perche' il blocco Minnesota e quello della costante hanno residuo nullo per
    costruzione e l'unico che contribuisce e' il blocco della covarianza.
    """
    b0, _ = implied_prior_moments(Yd, Xd)
    resid = np.asarray(Yd, dtype=float) - np.asarray(Xd, dtype=float) @ b0
    return resid.T @ resid


__all__ = [
    "regressor_layout",
    "minnesota_prior_mean",
    "minnesota_omega_diag",
    "prior_coefficient_sd",
    "initial_observation_mean",
    "sum_of_coefficients_dummy",
    "implied_prior_moments",
    "minnesota_dummy",
    "covariance_dummy",
    "constant_dummy",
    "build_dummies",
    "default_dof",
    "implied_prior_scale",
    "DummyStack",
    "CONST_PRIOR_VAR",
    "DEFAULT_CONST_EPS",
]
