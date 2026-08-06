"""
src/bvar/cube_root.py

GATE 3, BLOCCO 1 — LA MAPPA CUBE-ROOT: dal VAR trimestrale al suo gemello
mensile.  §2.4 del paper, derivata per esteso nell'Appendice A.

Vedi `src/bvar/__init__.py` per la mappa delle quattro varianti e la casella C.

Questo modulo e' una FUNZIONE PURA: array in, array out.  Non sa che cosa sia un
Q-BVAR, non legge dati, non estrae niente.  E' la regola di disaccoppiamento
fissata al Gate 2, e serve a poterlo testare contro l'esempio AR(2)
dell'Appendice A.1 senza stimare nulla.


================================================================================
COSA DICE IL PAPER
================================================================================
L'ipotesi di partenza (§2.4): il VAR trimestrale e' l'ITERATA A TRE PASSI di un
VAR mensile non osservato.  In companion form:

    X_tq = Phi   X_tq-1  + nu_tq                                       (A.1)
    X_tm = Phi_m X_tm-1  + nu_m,tm                                     (A.3)

Iterando tre volte la mensile e uguagliandola alla trimestrale:

    Phi_m = Phi^(1/3)                                                  (A.6)
    nu_tm = nu_m,tm + Phi_m nu_m,tm-1 + Phi_m^2 nu_m,tm-2              (A.7)

Le ultime n(p-1) righe di (A.7) sono un sistema sovradeterminato che si risolve
per eps_m,tm-1 (A.8); sostituendo nelle prime n righe si ricava

    vec(Sigma_eps_m) = (I + A (x) A)^-1 vec(Sigma_eps)                 (A.9)

e (A.10) e' la scorciatoia di Kronecker che evita di invertire una n^2 x n^2.

Selezione della radice: autovalori reali -> la loro radice cubica reale; coppie
complesse coniugate -> "the cube root which is characterized by the LEAST
OSCILLATORY BEHAVIOUR, i.e., the cube root with the smallest argument", come in
Giannone, Monti & Reichlin (2016).


================================================================================
COSA SIGNIFICA
================================================================================
Non si stima niente di mensile.  Si stima trimestrale e si DEDUCE la legge
mensile che, campionata a fine trimestre, riprodurrebbe quella trimestrale.  Il
prezzo e' la restrizione (A.4): il valore mensile corrente dipende da UN SOLO
MESE dentro ogni trimestre passato.  Non e' un VAR mensile qualsiasi, e' il
sottoinsieme compatibile con quello trimestrale.

TRE COSE CHE VALE LA PENA AVER CAPITO PRIMA DI LEGGERE IL CODICE.

1. PERCHE' LE RADICI SONO PIU' D'UNA, E PERCHE' LA SCELTA E' SENSATA.  Un numero
   reale ha una sola radice cubica reale; un numero complesso ne ha tre, sfasate
   di 2pi/3.  Se Phi ha k coppie coniugate ci sono 3^k radici cubiche REALI di
   Phi.  Sceglierne una e' scegliere QUANTO OSCILLA il processo mensile fra un
   trimestre e l'altro: i tre candidati riproducono tutti Phi dopo tre mesi, ma
   percorrendo strade diverse.  L'argomento piu' piccolo e' la strada piu'
   diretta — quella che non aggiunge cicli infra-trimestrali che i dati
   trimestrali non potrebbero mai vedere.  E' la scelta di regolarita', non
   un'approssimazione.

2. PERCHE' Sigma_eps_m NON E' Sigma_eps / 3.  Se i tre shock mensili fossero
   indipendenti e si sommassero, la varianza trimestrale sarebbe tre volte
   quella mensile.  Ma (A.7) dice che NON si sommano liberamente: le righe
   inferiori impongono che gli shock dei mesi 1 e 2 siano LEGATI fra loro
   (A.8), perche' gli stati ritardati devono restare coerenti a fine trimestre.
   Sopravvivono due gradi di liberta', non tre, e il legame e' la matrice A.  Da
   cui la forma Sigma_eps = Sigma_eps_m + A Sigma_eps_m A', cioe' (A.9).

3. LA COSTANTE.  Vedi la sezione dedicata piu' sotto: era un punto aperto
   registrato al Gate 2, ed e' risolto qui.


================================================================================
L'ANALISI DI RIPRODUCIBILITA' — il contributo principale di questa sezione
================================================================================
>>> Da leggere prima di usare l'uscita.  L'Appendice A ha TRE letture
>>> inequivalenti, e la scelta decide se Sigma_eps_m e' una matrice di
>>> covarianza.  Dettagli operativi in `coupling_matrix`.

I TRE FATTI, tutti verificabili eseguendo `test_gate3`:

  1. (A.8)/(A.9) COME STAMPATE non riproducono (A.15), la forma chiusa che il
     paper stesso deriva nell'esempio AR(2) dell'Appendice A.1.  Su 5 casi, mai.

  2. IL CODICE DI REPLICA DEGLI AUTORI usa una TERZA formula, diversa da
     entrambe (`build_monthly_ss.m`: J come matrice dei coefficienti,
     [Phi_m^2].1 come termine noto).  Neanche questa riproduce (A.15).

  3. SOLO la lettura che discende correttamente da (A.7) — dove il coefficiente
     su eps_{t-1} nel blocco i e' Phi_mi1 e non l'identita' — riproduce (A.15),
     a 8 decimali su tutti i casi.  E' la variante "exact_a15".

E LA SCELTA NON E' INNOCUA.  Misurato su 100 estrazioni vere, n=30, p=5:

    formula      Frobenius mediana   p90      autoval.neg   lambda_min/lambda_max
    exact_a15          75.77%      99.16%       9 su 30           -1.22
    authors             0.00%       1.84%       2 su 30           -5.8e-06

dove "Frobenius" e' di quanto la proiezione di Higham deve spostare Sigma_m per
renderla PSD.  Con "exact_a15" ne sostituisce tre quarti; con "authors" e' una
correzione di arrotondamento.  E rho(A) passa da 2.28 a 1.36.

PERCHE' IL DEFAULT E' "authors" pur non riproducendo (A.15).  Perche' a p > 2 il
sistema (A.8) NON HA SOLUZIONE ESATTA — l'Appendice A dice essa stessa "can be
APPROXIMATELY solved" — quindi il testo licenzia una qualunque approssimazione,
e fra approssimazioni si sceglie con criteri numerici: una replica deve
replicare i risultati pubblicati.  A p = 2 la faccenda cambia natura, ed e' il
perno dell'argomento: vedi la sezione qui sotto.


================================================================================
p = 2 E' IL PERNO — dove l'incoerenza si vede, e dove diventa consequenziale
================================================================================
Questa e' la sezione da leggere per capire il contributo di riproducibilita'.
Il punto e' un CONTEGGIO, e da li' discende tutto il resto.

--- IL CONTEGGIO: quante equazioni, quante incognite -------------------------

Le righe di vincolo di (A.7) sono, per ogni blocco i = 2 ... p,

    0 = Phi_m,i1 eps_{t-1} + [Phi_m^2]_i1 eps_{t-2}

e l'incognita e' la matrice M (n x n) che lega i due shock, eps_{t-1} = M
eps_{t-2}, cioe' proprio la (A.8).  In forma matriciale: C M = D, con C e D di
forma ((p-1)n, n).  Quindi

    equazioni scalari   (p-1) * n^2
    incognite scalari         n^2

    p = 2   ->  (p-1) = 1   ESATTAMENTE DETERMINATO   soluzione unica
    p > 2   ->  (p-1) > 1   SOVRADETERMINATO          nessuna soluzione esatta

**Il paper usa p = 5.**  Cioe' la rappresentazione mensile su cui poggia tutto
il C-BVAR esiste, alla lettera, SOLO nel caso a due ritardi — l'unico che il
paper illustra (l'esempio AR(2) dell'Appendice A.1, da cui viene la forma
chiusa A.15) — e NON esiste nel caso che il paper effettivamente stima.  Il
passaggio da "esatto" a "approssimato" avviene fra l'esempio didattico e
l'applicazione, e il paper lo segnala con tre parole in mezzo a una frase ("can
be approximately solved") senza mai dire che cosa si stia approssimando ne'
quanto bene.

--- A p = 2: LE TRE LETTURE NON COINCIDONO, E DUE SONO SEMPLICEMENTE SBAGLIATE

ATTENZIONE, e' l'errore facile da fare: NON e' vero che le tre letture
coincidano a p = 2 e divergano dopo.  Misurato su radici cubiche vere (non su
companion, che invece hanno Phi_m.1 = I per costruzione e mascherano il
fenomeno), a p = 2 le tre A differiscono gia' fra loro del 40-95% in norma.

Quel che e' vero — e che e' MOLTO piu' forte — e' questo:

    a p = 2 tutte e tre risolvono ESATTAMENTE il proprio sistema
    (residuo 0.0000 su tutte le estrazioni), ma sono TRE SISTEMI DIVERSI.

Cioe' a p = 2 non c'e' nessuna approssimazione da nessuna parte, e quindi
nessuna scusa: c'e' una risposta giusta, e' unica, ed e' quella di "exact_a15"
— l'unica che riproduce (A.15), la forma chiusa che il paper stesso deriva
(verificato a 8 decimali in `test_gate3`, §3).  Le altre due sostituiscono a C
la pila di identita' J = [I ... I]', che e' un'ALTRA equazione, e ne danno la
soluzione esatta.  **A p = 2 non sono approssimazioni: sono errori**, e si vede
senza ambiguita' numerica.

--- A p = 5: SI RIBALTA TUTTO -----------------------------------------------

Sopra p = 2 non esiste piu' una risposta esatta e ogni lettura diventa un
minimo-quadrati.  Ma i tre minimi quadrati non sono equivalenti.  Misurato sul
pannello vero, n = 30, 40 estrazioni del Q-BVAR (seme 7), residuo relativo dei
minimi quadrati sul PROPRIO sistema, mediana:

    lettura        p = 2        p = 5
    exact_a15      0.0000       0.0040      quasi consistente
    authors        0.0000       0.9303      NON consistente
    literal        0.0000       0.9278      NON consistente

e sul sistema VERO, quello implicato da (A.7) (mediana, p = 5):

    exact_a15      0.0040   |   authors  1.8192   |   literal  2.0492

Da leggere cosi': il sistema che (A.7) implica davvero **e'** approssimabile —
un residuo dello 0.4% e' esattamente cio' che "can be approximately solved"
descrive.  Il sistema stampato in (A.8), e quello del codice degli autori, NON
lo sono: chiedono a p-1 = 4 blocchi diversi di accordarsi sulla stessa M, e il
residuo e' del 93%.  La frase di licenza del paper copre una cosa che il paper
stesso non fa.

--- E QUI STA IL PARADOSSO CHE VA SCRITTO IN TESI ---------------------------

La lettura matematicamente corretta e' quella NUMERICAMENTE INUTILIZZABILE.
Sulle stesse 100 estrazioni (n = 30, p = 5):

    lettura        Frobenius Higham (mediana / p90)   autoval. neg.   rho(A)
    exact_a15            75.77% / 99.16%                 9 su 30       2.28
    authors               0.00% /  1.84%                 2 su 30       1.36

Il meccanismo e' identificato: e' rho(A) a decidere.  La (A.9) inverte la mappa
X -> X + A X A', che conserva il cono PSD solo quando la serie di Neumann
converge, cioe' per rho(A) < 1.  Con "exact_a15" rho(A) = 2.28 e Sigma_m esce
indefinita in modo massiccio; con "authors" rho(A) = 1.36 e la violazione e'
dell'ordine dell'arrotondamento.  **La formula sbagliata sta piu' vicina al cono
PSD di quella giusta**, e questo non e' un caso fortunato da nascondere: e' il
sintomo che a p = 5 l'oggetto "VAR mensile compatibile" non esiste, e ogni
lettura sta scegliendo quale pezzo del vincolo sacrificare.

--- CHE GLI AUTORI LO SAPPIANO, SI VEDE -------------------------------------

Non e' un problema che sfugge loro: e' un problema che GESTISCONO IN SILENZIO.
In `build_monthly_ss.m` la decomposizione di Sigma_m sta dentro un try/catch,
gli autovalori negativi vengono schiacciati con `D(D<0) = 1e-10`, un `qqFlag`
registra quando succede, e c'e' un riferimento a Higham COMMENTATO — cioe'
qualcuno ha guardato la letteratura sulla PSD piu' vicina e poi ha lasciato
perdere.  Ma:

  * nel PAPER non se ne parla mai.  Ne' §2.4 ne' l'Appendice A menzionano la
    semidefinita positivita';
  * in GMR (2016), la fonte citata per la radice cubica, nemmeno: la Prop. 1
    assume "T_m is real and stable" su dati stazionari, e noi siamo in
    log-livelli con rho(Phi) ~ 1.012, cioe' FUORI dalle loro ipotesi;
  * il `qqFlag` non compare in nessun output pubblicato.

Noi divergiamo qui, e apertamente: proiezione di Higham (1988) sulla PSD piu'
vicina invece dello schiacciamento degli autovalori, con lo spostamento
MISURATO a ogni estrazione (`state_space.nearest_psd`, `psd_summary`).  Non
perche' sia piu' giusta — e' un'approssimazione anche lei — ma perche' e'
quantificata: la tabella qui sopra esiste solo perche' lo spostamento si misura.

--- IL PREZZO SI PAGA ANCHE DOVE NON C'E' NULLA DA STIMARE -------------------

Aggiunto al Gate 6, ed e' un'altra faccia dello stesso risultato.

Sui trimestri GIA' OSSERVATI non c'e' niente da stimare: il PIL e' un dato, e il
modello deve restituirlo tale e quale.  E' il controllo che chiude i gate.
Misurato a `as_of = 2018-11-16` (`tests/test_gate6.py`):

    B-BVAR    0.0008 pp   con 3 estrazioni    la storia e' il dato REPLICATO
    L-BVAR    0.0000 pp   con 3 estrazioni    le celle osservate, R = 0
    C-BVAR    0.010  pp   con 150 estrazioni  <-- e con 8 ne serviva 0.066

Nel B e nell'L la riproduzione e' esatta **estrazione per estrazione**, e tre
bastano.  Nel C-BVAR **no**: il PIL trimestrale osservato non entra come
vincolo diretto, viene RICOSTRUITO attraverso il processo mensile latente, il
cui Sigma_m e' passato per la proiezione di Higham.  Ogni estrazione lo
riproduce a meno di una dispersione residua di **~0.10 pp**, e la riproduzione
e' esatta solo NELLA MEDIANA:

    S =   8    errore della mediana  0.066 pp
    S =  40                          0.020 pp
    S = 150                          0.010 pp        (scende come 1/sqrt(S))

Cioe': **il C-BVAR paga la mappa cube-root anche dove non c'e' nessuna
incertezza da rappresentare.**  Non e' un difetto dell'implementazione — e' il
costo di far passare un dato osservato attraverso una rappresentazione mensile
che a p = 5 esiste solo in modo approssimato.  Da riportare accanto al resto: e'
la stessa storia dell'Appendice A vista dal lato operativo.

Conseguenza pratica per il Gate 6: il C-BVAR ha bisogno di piu' estrazioni degli
altri per una mediana stabile, e questo entra nel preventivo.

--- IL CONTRIBUTO, IN QUATTRO RIGHE -----------------------------------------

  1. l'Appendice A e' internamente incoerente: (A.8) come stampata non
     riproduce (A.15), che il paper deriva due pagine dopo, e il codice di
     replica usa una TERZA formula ancora;
  2. a p = 2 questo si dimostra senza margini — il sistema e' quadrato, la
     risposta giusta e' unica, e due letture su tre non la danno;
  3. a p = 5, il p che il paper stima, la rappresentazione mensile esiste solo
     in modo approssimato, e la scelta fra le tre letture — mai dichiarata —
     decide se il modello e' STIMABILE (0% contro 76% di sostituzione della
     matrice di covarianza; rho(A) 1.36 contro 2.28);
  4. la non-PSD che ne segue e' nota agli autori e gestita in silenzio, non
     discussa dal paper, e fuori dalle ipotesi della fonte che il paper cita.

Riprodurre i numeri: `test_gate3` §3 (la corrispondenza con A.15) e la tabella
dei residui con `coupling_matrix(..., variant=...)` su estrazioni di `qbvar.fit`.


IL C-BVAR FUNZIONA — ma solo sulla finestra terminale
------------------------------------------------------
Con la formula "authors" Sigma_m e' sostanzialmente PSD, ma NON basta: il filtro
diverge lo stesso se lo si fa girare su tutto il campione.  La causa non e' la
covarianza, e' Phi_m:

    raggio spettrale di Phi_m      1.005      (mite)
    |Phi_m| max                    1.8e+04    (enorme)
    cond(V) della radice cubica    1.05e+06

cioe' una matrice fortemente NON NORMALE: autovalori miti, amplificazione per
passo grande.  Su 16 passi non si vede, su 400 domina, e il degrado e' monotono
nella lunghezza della finestra.  Il rimedio e' quello degli autori — filtrare
solo il bordo, prendere la storia dai dati — ed e' documentato con la tabella in
`state_space.edge_window`.

Il paper non spiega mai perche' il loro filtro giri su una finestra corta.
Quella tabella e' la risposta, ed e' un risultato nostro.

ESCLUSE, misurate: l'inizializzazione (P_0 su otto ordini di grandezza non
cambia nulla; la varianza non condizionata non esiste perche' il sistema e'
esplosivo) e la dimensione n (nessuna soglia: il comportamento non cambia
qualitativamente fra n=6 e n=30).

CORREZIONI A VERSIONI PRECEDENTI DI QUESTA NOTA, tenute perche' il percorso e'
istruttivo: si era scritto (a) che il problema non dipendeva dalla lettura di
A.9 — falso, ne dipende quasi interamente; (b) che c'era una soglia in n — era
misurato su VAR sintetici troppo benigni; (c) che il C-BVAR non era recuperabile
— lo e', con la finestra corta.


CHE COSA DICE GMR (2016) SU QUESTO — la fonte a cui Cimadomo rimanda
--------------------------------------------------------------------
Cimadomo §2.4 dice che la sezione "reflects and EXPANDS the results previously
derived for DSGE models by Giannone et al. (2016)", e la nota 28 rimanda a GMR
per il caso non diagonalizzabile.  Letto GMR per intero (15 pagine):

  * SULLA PSD, GMR NON DICE NULLA.  Le parole "positive", "definite",
    "semidefinite" hanno ZERO occorrenze nel paper.  Nessun rimedio prescritto,
    perche' il problema non viene mai incontrato.

  * IL CASO NON DIAGONALIZZABILE DELLA NOTA 28 E' UN ALTRO PROBLEMA.  GMR p.203
    cita Higham (2008): "there exists no p-th (so also no cube) root of a matrix
    that has zero-valued eigenvalues that are DEFECTIVE".  Nasce dagli stati
    ridondanti dei DSGE e il rimedio e' "work on the model to try to reduce it
    to a minimal state space".  Non c'entra con la PSD.

  * GMR ASSUME LA STABILITA', ESPLICITAMENTE E DUE VOLTE.  Prima dell'eq. (4):
    "Assume that the monthly states can be written as ... and that T_m IS REAL
    AND STABLE".  E nella Proposizione 1: "where T_m is real and stable and
    eps_m,tm are orthonormal shocks".  In piu' l'eq. (1) di GMR parte da
    osservabili "which are transformed to be STATIONARY".  Il loro T_theta e'
    la soluzione log-linearizzata di un DSGE, quindi stabile per costruzione:
    nel loro setup il problema non puo' presentarsi.

  * L'EQUAZIONE DI GMR NON E' QUELLA DI CIMADOMO.  GMR (7) e'

        vec(B_m B_m') = (I + T_m (x) T_m + T_m^2 (x) T_m^2)^-1 vec(B_th B_th')

    cioe' i TRE shock mensili sono indipendenti e le varianze si sommano in tre
    termini.  Cimadomo (A.9) ha invece (I + A (x) A)^-1 con un solo A, perche'
    aggiunge la restrizione (A.8) che lega eps_m,tm-1 a eps_m,tm-2 — restrizione
    che nasce dalla struttura COMPANION del VAR e che lo stato DSGE di GMR non
    ha.  La parte sulla covarianza dell'Appendice A e' dunque un'ESTENSIONE di
    Cimadomo, e su quella GMR non e' la fonte vincolante.  Lo e' invece, alla
    lettera, sulla radice cubica e sulla selezione delle radici: li' il testo di
    Cimadomo ricalca GMR quasi parola per parola.

  * PROVATA ANCHE LA (7) DI GMR SUI NOSTRI DRAW: VA PEGGIO.

        Cimadomo A.9   lambda_min/lambda_max  mediana  -1.5     PSD in 0%
        GMR (7)        lambda_min/lambda_max  mediana -55.5     PSD in 0%
        GMR (7) identita' in avanti, errore relativo mediano 2.6e-13

    L'identita' in avanti torna a 1e-13, quindi la (7) e' implementata bene: e'
    proprio la sua soluzione a non essere una covarianza.  NON E' L'EQUAZIONE,
    E' IL SISTEMA.  Cambiare formula non e' un rimedio.

  * E LA STABILITA' DA SOLA NON BASTEREBBE.  Sui VAR sintetici la PSD si perde
    da n >= 10 anche con rho(Phi) < 1 imposto (max_eig = 0.9995); sui dati veri
    si perde a ogni n provato, da n = 6 in su.  Rendere stazionario il sistema
    non e' garanzia di rimedio.

CONCLUSIONE: siamo su un punto che ne' Cimadomo ne' GMR coprono.


COSA PORTARE IN TESI
--------------------------------------------------------------------------
Non un risultato negativo — quello era la lettura di una versione precedente,
basata su una formula di accoppiamento che si e' poi rivelata solo una delle
tre possibili.  Il contributo e' UN'ANALISI DI RIPRODUCIBILITA', ed e' piu'
forte:

  1. L'Appendice A del paper e' INTERNAMENTE INCOERENTE.  (A.8)/(A.9) come
     stampate non riproducono (A.15), la forma chiusa che il paper stesso
     deriva; e il codice di replica degli autori usa una terza formula ancora.
     Tre letture, nessuna coincidenza fra le prime due e la terza.

  2. LA SCELTA FRA LORO DECIDE SE IL MODELLO E' STIMABILE.  Non e' una
     sottigliezza notazionale: lo spostamento che la proiezione PSD deve
     applicare passa dallo 0.0% al 75.8% mediano a seconda della lettura.

  3. GLI AUTORI INCONTRANO LA NON-PSD E LA GESTISCONO IN SILENZIO.  In
     `build_monthly_ss.m` c'e' un `try/catch` che azzera gli autovalori
     negativi (`D(D<0) = 1e-10`) con un `qqFlag` che lo registra, e un
     riferimento commentato alla "Cheng and N. J. Higham approximation".  Il
     paper non ne parla.  Ne' Cimadomo ne' GMR (2016) discutono mai la
     semidefinita positivita' — vedi la sezione su GMR.

  4. IL FILTRO VA FATTO GIRARE SU UNA FINESTRA CORTA, E IL PAPER NON DICE
     PERCHE'.  Gli autori filtrano gli ultimi 3*lags+1 mesi e incollano la
     storia dai dati (con la versione a campione pieno scritta e commentata).
     Noi abbiamo misurato la ragione: Phi^(1/3) di una companion quasi-difettiva
     e' fortemente non normale (entrate 1.8e4, cond(V) 1e6), e l'errore cresce
     in modo monotono con la lunghezza della ricorsione.  Tabella in
     `state_space.edge_window`.

  5. IL REGIME LARGE-n NON STAZIONARIO E' PIU' OSTILE DI QUELLO DEL PAPER, e
     GMR Proposizione 1 — che assume "T_m real and stable" su osservabili
     "transformed to be stationary" — non ci copre.  Ma non e' questo a
     impedire il modello: e' una questione di condizionamento, non di ipotesi
     violate.

Il percorso che ha portato qui e' esso stesso materiale: tre conclusioni
successive ribaltate da misure successive (non dipende dalla lettura -> dipende
quasi solo da quella; c'e' una soglia in n -> non c'e'; non e' recuperabile ->
lo e').  Vale la pena raccontarlo come esempio di quanto sia facile attribuire
a un metodo cio' che dipende da una scelta implementativa non dichiarata.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: La variante di default della matrice di accoppiamento.  "authors" replica il
#: codice MATLAB degli autori — vedi `coupling_matrix` e la sezione
#: "TRE FORMULE, UNA SOLA REPLICA" nell'header.
DEFAULT_COUPLING = "authors"


#: Sotto questa soglia sulla parte immaginaria un autovalore e' trattato come
#: REALE, e prende la sua radice cubica reale (Appendice A).  Non e' una
#: tolleranza qualsiasi: distingue i due rami della regola di selezione.
REAL_TOL = 1e-9

#: Soglia sulla parte immaginaria residua di una matrice che DEVE essere reale.
IMAG_TOL = 1e-7


@dataclass
class MonthlyMap:
    """
    Il gemello mensile di un VAR trimestrale: l'uscita della mappa di §2.4.

    Attributes
    ----------
    Phi_m : (n*p, n*p)      Phi^(1/3), eq. (A.6)
    Sigma_m : (n, n)        Sigma_eps_m, eq. (A.9')
    const_m : (n*p,) | None  c_m, la costante mensile.  None se non e' stata data
                             una costante trimestrale.
    A : (n, n)              la matrice di accoppiamento di (A.9')
    n, p : int
    diagnostics : dict      condizionamento e residui, per il Gate 3
    """

    Phi_m: np.ndarray
    Sigma_m: np.ndarray
    const_m: np.ndarray | None
    A: np.ndarray
    n: int
    p: int
    diagnostics: dict
    variant: str = DEFAULT_COUPLING

    def check(self, Phi: np.ndarray, Sigma: np.ndarray, tol: float = 1e-7) -> dict:
        """
        Le due identita' che definiscono la mappa, verificate all'indietro.
        E' il controllo che il Gate 3 puo' chiamare su ogni estrazione.

            Phi_m^3 == Phi                          (A.6)
            Sigma_m + A Sigma_m A' == Sigma_eps     (A.9')
        """
        cube = np.linalg.matrix_power(self.Phi_m, 3)
        back = self.Sigma_m + self.A @ self.Sigma_m @ self.A.T
        return {
            "cube_err": float(np.abs(cube - Phi).max()),
            "cov_err": float(np.abs(back - Sigma).max()),
            "cube_ok": bool(np.allclose(cube, Phi, atol=tol)),
            "cov_ok": bool(np.allclose(back, Sigma, atol=tol)),
        }


# ─── 1. La radice cubica della companion — A.6 ────────────────────────────────

def select_cube_roots(ev: np.ndarray, *, real_tol: float = REAL_TOL) -> np.ndarray:
    """
    La regola di selezione dell'Appendice A, applicata agli autovalori.

      * autovalore REALE  -> la sua radice cubica reale, `sign(l) |l|^(1/3)`.
        Serve scriverla a mano: per un reale NEGATIVO il ramo principale di
        `l**(1/3)` restituisce un complesso (modulo giusto, argomento pi/3), non
        la radice reale.  E' un caso che capita subito — il primo esempio AR(2)
        del paper, phi=(0.6, 0.25), ha un autovalore a -0.283.

      * autovalore COMPLESSO -> "the cube root with the smallest argument".  Per
        l = r e^(i theta) con theta in (-pi, pi], i tre candidati hanno argomento
        theta/3, (theta+2pi)/3, (theta-2pi)/3; il primo ha modulo |theta|/3 <=
        pi/3, gli altri almeno pi/3.  Il piu' piccolo e' dunque SEMPRE il ramo
        principale, che e' esattamente cio' che `l**(1/3)` calcola.

    Le coppie coniugate ricevono radici coniugate (il ramo principale commuta col
    coniugio), che e' cio' che rende reale la matrice ricostruita.
    """
    ev = np.asarray(ev, dtype=complex)
    out = np.empty_like(ev)
    is_real = np.abs(ev.imag) < real_tol
    r = ev[is_real].real
    out[is_real] = np.sign(r) * np.abs(r) ** (1.0 / 3.0)
    out[~is_real] = ev[~is_real] ** (1.0 / 3.0)
    return out


def matrix_cube_root(Phi: np.ndarray, *, real_tol: float = REAL_TOL,
                     imag_tol: float = IMAG_TOL) -> tuple[np.ndarray, dict]:
    """
    Phi_m = Phi^(1/3), eq. (A.6), via decomposizione spettrale.

    Se Phi = V D V^-1 allora Phi^(1/3) = V D^(1/3) V^-1, con D^(1/3) le radici
    scelte da `select_cube_roots`.

    IL CASO NON DIAGONALIZZABILE.  La nota 28 del paper rimanda a Giannone,
    Monti & Reichlin (2016) e noi non l'abbiamo implementato: una companion e'
    non-derogatoria, quindi e' diagonalizzabile se e solo se i suoi autovalori
    sono distinti, il che per una matrice estratta da un posterior continuo
    accade con probabilita' 1.  Il rischio VERO non e' l'esatta degenerazione ma
    la QUASI degenerazione, che rende V mal condizionata senza che nulla
    fallisca.  Per questo il condizionamento di V e' RESTITUITO e non ignorato:
    e' la spia da guardare, ed e' compito del chiamante decidere una soglia.

    Returns
    -------
    (Phi_m, diagnostics)
        `diagnostics` ha `cond_V`, `max_imag`, `n_real`, `n_complex`,
        `spectral_radius`.
    """
    Phi = np.asarray(Phi, dtype=float)
    if Phi.ndim != 2 or Phi.shape[0] != Phi.shape[1]:
        raise ValueError(f"Phi deve essere quadrata, ricevuta {Phi.shape}")

    ev, V = np.linalg.eig(Phi.astype(complex))
    roots = select_cube_roots(ev, real_tol=real_tol)
    Phi_m_c = V @ np.diag(roots) @ np.linalg.inv(V)

    # LA TOLLERANZA E' RELATIVA, e non e' un dettaglio.  Una soglia ASSOLUTA su
    # `max_imag` e' cieca alla scala e al condizionamento: su una companion
    # 150 x 150 con cond(V) ~ 1e3 il residuo immaginario di puro arrotondamento
    # vale ~1e-7 in valore assoluto, e una soglia assoluta a 1e-7 la scambia per
    # un errore di conjugacy.  E' successo alla prima estrazione vera del
    # Q-BVAR (1.465e-07).
    #
    # Le due cause vanno distinte, perche' i rimedi sono opposti:
    #   * RESIDUO DI ARROTONDAMENTO — piccolo RISPETTO A |Phi_m|, cresce con
    #     cond(V).  Benigno: si prende la parte reale.
    #   * CONJUGACY ROTTA — una coppia coniugata ha ricevuto radici non
    #     coniugate perche' `real_tol` l'ha classificata male.  Allora la parte
    #     immaginaria e' dello stesso ordine di |Phi_m|, non un epsilon.
    scale = max(1.0, float(np.abs(Phi_m_c.real).max()))
    max_imag = float(np.abs(Phi_m_c.imag).max())
    rel_imag = max_imag / scale
    if rel_imag > imag_tol:
        raise ValueError(
            f"la radice cubica non e' reale: parte immaginaria massima "
            f"{max_imag:.3e}, relativa {rel_imag:.3e} > {imag_tol:.1e}.\n"
            f"  cond(V) = {np.linalg.cond(V):.2e}, {int((np.abs(ev.imag) < real_tol).sum())} "
            f"autovalori reali su {ev.size}.\n"
            f"  Se il residuo relativo e' dell'ordine di 1 la conjugacy e' rotta "
            f"(alza `real_tol`, ora {real_tol:.1e}); se e' piccolo ma sopra "
            f"soglia e' arrotondamento su una matrice mal condizionata (alza "
            f"`imag_tol`)."
        )
    diag = {
        "cond_V": float(np.linalg.cond(V)),
        "max_imag": max_imag,
        "rel_imag": rel_imag,
        "n_real": int((np.abs(ev.imag) < real_tol).sum()),
        "n_complex": int((np.abs(ev.imag) >= real_tol).sum()),
        "spectral_radius": float(np.abs(ev).max()),
    }
    return Phi_m_c.real, diag


# ─── 2. La matrice di accoppiamento — A.9' ────────────────────────────────────

def coupling_matrix(Phi_m: np.ndarray, n: int, *,
                    variant: str = DEFAULT_COUPLING) -> np.ndarray:
    r"""
    La matrice A di (A.9).  ESISTONO TRE FORMULE E NON COINCIDONO.

    Tutte hanno la forma  A = [Phi_m^2]_11 - Phi_m11 M,  e differiscono solo in
    come si ricava M dal sistema sovradeterminato delle righe inferiori di (A.7).

    variant = "authors"   (DEFAULT)  M risolve ai minimi quadrati  J M = [Phi_m^2].1
    variant = "exact_a15"            M risolve ai minimi quadrati  Phi_m.1 M = [Phi_m^2].1
    variant = "literal"              M risolve ai minimi quadrati  J M = Phi_m.1

    con J = [I_n ... I_n]' di forma ((p-1)n, n), Phi_m.1 = Phi_m[n:, :n] e
    [Phi_m^2].1 = (Phi_m @ Phi_m)[n:, :n].

    QUALE E' "GIUSTA" — la risposta onesta e' che la domanda e' mal posta.

      * "exact_a15" e' quella che discende correttamente da (A.7): siccome
        nu_m,t = (eps_m,t', 0)', il blocco i-esimo da'
            0 = Phi_mi1 eps_{t-1} + [Phi_m^2]_i1 eps_{t-2}
        e il coefficiente su eps_{t-1} e' Phi_mi1, NON l'identita'.  A p = 2 il
        sistema e' ESATTAMENTE determinato e questa formula lo risolve esatto,
        riproducendo (A.15) a 8 decimali (verificato su 5 AR(2)).  Le altre due
        no.

      * "authors" e' quella del codice di replica degli autori
        (`docs/BVARs/code/functionsGMS/build_monthly_ss.m`):
            kronMat = aa2(1:n,1:n) - aa(1:n,1:n)*(repmat(eye(n),lags-1,1) \ aa2(n+1:end,1:n))
        Usa J come matrice dei coefficienti e [Phi_m^2].1 come termine noto.
        NON riproduce (A.15) — ma e' quella che ha prodotto i risultati
        PUBBLICATI.

      * "literal" e' l'Appendice A stampata, (A.8) alla lettera: J come
        coefficienti e Phi_m.1 come termine noto.  Non riproduce (A.15) e non
        corrisponde al codice.  Tenuta solo per documentare la discrepanza.

    IL CONTEGGIO CHE SPIEGA TUTTO.  C M = D ha (p-1)n^2 equazioni scalari e n^2
    incognite: ESATTAMENTE DETERMINATO a p = 2, SOVRADETERMINATO a p > 2.  Da
    cui i due regimi, che vanno tenuti distinti (trattazione completa nella
    sezione "p = 2 E' IL PERNO" dell'header):

      * a p = 2 tutte e tre risolvono esattamente il PROPRIO sistema (residuo
        0.0000), ma i sistemi sono tre e uno solo e' quello di (A.7).  Non sono
        approssimazioni concorrenti: due sono errori, e si vede;
      * a p = 5 nessuna e' esatta, ma non sono equivalenti: il residuo dei
        minimi quadrati sul proprio sistema e' 0.40% per "exact_a15" e 93% per
        "authors"/"literal".  La licenza del paper ("can be APPROXIMATELY
        solved") descrive bene il sistema di (A.7), male quello stampato.

    PERCHE' IL DEFAULT E' "authors", pur non riproducendo (A.15) ed essendo la
    lettura peggio posta: perche' a p > 2 nessuna e' esatta, il testo licenzia
    l'approssimazione, e una REPLICA deve replicare i risultati pubblicati.  Fra
    approssimazioni si sceglie con criteri numerici, e la differenza a p = 5 e'
    drammatica — nel verso paradossale.  Misurato su 100 estrazioni vere,
    n = 30:

        formula      Frob mediana   Frob p90   autoval.neg   lambda_min/lambda_max
        exact_a15         75.77%      99.16%       9 su 30         -1.22
        authors            0.00%       1.84%       2 su 30         -5.8e-06

    dove "Frob" e' di quanto la proiezione di Higham deve spostare Sigma_m.  Con
    "authors" la proiezione e' una VERA correzione di arrotondamento (gli
    autovalori negativi stanno sei ordini di grandezza sotto il maggiore); con
    "exact_a15" sostituisce tre quarti della matrice.  E rho(A) passa da 2.28 a
    1.36, cioe' dal regime in cui la serie di Neumann diverge a quello in cui
    quasi converge.

    Da cui la regola operativa: una REPLICA deve replicare i risultati
    pubblicati, quindi default "authors"; "exact_a15" resta accessibile perche'
    e' il perno dell'analisi di riproducibilita' (vedi header).

    Raises
    ------
    NotImplementedError
        Se p = 1 (nota 31 del paper: rimanda a GMR 2016).
    ValueError
        Se `variant` non e' uno dei tre.
    """
    Phi_m = np.asarray(Phi_m, dtype=float)
    np_ = Phi_m.shape[0]
    p = np_ // n
    if n * p != np_:
        raise ValueError(f"Phi_m e' {Phi_m.shape} ma n={n} non la divide")
    if p < 2:
        raise NotImplementedError(
            "p = 1: (A.7) non produce righe di vincolo e la derivazione "
            "dell'Appendice A non si applica.  La nota 31 del paper rimanda a "
            "Giannone, Monti & Reichlin (2016), che copre il caso a un solo "
            "ritardo trimestrale.  Il nostro p e' 5."
        )

    Phi_m2 = Phi_m @ Phi_m
    J = np.tile(np.eye(n), (p - 1, 1))            # ((p-1)n, n)
    Phi_m_1 = Phi_m[n:, :n]                       # Phi_m.1
    Phi_m2_1 = Phi_m2[n:, :n]                     # [Phi_m^2].1

    if variant == "authors":
        C, D = J, Phi_m2_1
    elif variant == "exact_a15":
        C, D = Phi_m_1, Phi_m2_1
    elif variant == "literal":
        C, D = J, Phi_m_1
    else:
        raise ValueError(
            f"variant ignota {variant!r}; attese 'authors', 'exact_a15', 'literal'"
        )

    M, *_ = np.linalg.lstsq(C, D, rcond=None)
    return Phi_m2[:n, :n] - Phi_m[:n, :n] @ M


# ─── 3. La covarianza mensile — A.9 con la scorciatoia A.10 ───────────────────

def monthly_innovation_cov(A: np.ndarray, Sigma: np.ndarray, *,
                           shortcut: bool = True,
                           imag_tol: float = IMAG_TOL) -> np.ndarray:
    r"""
    Sigma_eps_m da (A.9):   vec(Sigma_eps_m) = (I + A (x) A)^-1 vec(Sigma_eps)

    Parameters
    ----------
    shortcut : bool
        True (default) usa (A.10).  False forma esplicitamente la n^2 x n^2 e la
        inverte: serve al test di equivalenza, ed e' impraticabile al crescere
        di n — che e' il motivo per cui il paper scrive A.10.

    LA SCORCIATOIA (A.10), svolta.  Se A = P L P^-1 con L diagonale, allora

        (I + A (x) A)^-1 = (P (x) P) (I + L (x) L)^-1 (P^-1 (x) P^-1)

    e usando vec(X Y Z) = (Z' (x) X) vec(Y) non serve MAI costruire un prodotto
    di Kronecker:

        1.  S~ = P^-1 Sigma_eps P^-T           <- (P^-1 (x) P^-1) vec(Sigma)
        2.  M_ij = S~_ij / (1 + l_i l_j)       <- (I + L (x) L)^-1, e' diagonale
        3.  Sigma_eps_m = P M P'               <- (P (x) P)

    Costo: due decomposizioni n x n e tre prodotti n x n, invece di invertire una
    n^2 x n^2.  A n=30 sono 900 x 900 contro 30 x 30.

    Nota sulle trasposte: e' P^-T, la trasposta SEMPLICE, non la coniugata —
    `vec(A X B) = (B' (x) A) vec(X)` non coniuga.  Con autovalori complessi i
    passi intermedi sono complessi e il risultato torna reale; la parte
    immaginaria residua e' controllata.
    """
    A = np.asarray(A, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = Sigma.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A e' {A.shape} ma Sigma e' {Sigma.shape}")

    if not shortcut:
        K = np.eye(n * n) + np.kron(A, A)
        out = np.linalg.solve(K, Sigma.reshape(-1, order="F")).reshape(n, n, order="F")
        return 0.5 * (out + out.T)

    lam, P = np.linalg.eig(A.astype(complex))
    P_inv = np.linalg.inv(P)
    S_tilde = P_inv @ Sigma.astype(complex) @ P_inv.T
    denom = 1.0 + np.outer(lam, lam)
    if np.abs(denom).min() < 1e-12:
        raise ValueError(
            "(I + A (x) A) e' singolare: esiste una coppia di autovalori di A "
            "con l_i l_j = -1, quindi (A.9) non ha soluzione unica."
        )
    out_c = P @ (S_tilde / denom) @ P.T
    max_imag = float(np.abs(out_c.imag).max())
    if max_imag > imag_tol * max(1.0, float(np.abs(Sigma).max())):
        raise ValueError(
            f"Sigma_eps_m non e' reale: parte immaginaria massima {max_imag:.3e}"
        )
    out = out_c.real
    return 0.5 * (out + out.T)


# ─── 4. La costante — il punto aperto del Gate 2 ──────────────────────────────

def monthly_constant(Phi_m: np.ndarray, const: np.ndarray) -> tuple[np.ndarray, float]:
    """
    c_m = (I + Phi_m + Phi_m^2)^-1 c.

    Derivazione e falsificazione dell'ipotesi alternativa (c_m = c) nella
    sezione "LA COSTANTE A_0" in cima al modulo.

    Parameters
    ----------
    const : (n*p,) oppure (n,)
        Se e' lunga n viene promossa a (n*p,) mettendo zeri sotto — che e' la
        forma c = (A_0', 0 ... 0)' della companion.

    Returns
    -------
    (c_m, cond)  con `cond` il numero di condizionamento di (I + Phi_m + Phi_m^2).
    """
    Phi_m = np.asarray(Phi_m, dtype=float)
    np_ = Phi_m.shape[0]
    c = np.asarray(const, dtype=float).ravel()
    if c.size != np_:
        padded = np.zeros(np_)
        padded[: c.size] = c
        c = padded

    S = np.eye(np_) + Phi_m + Phi_m @ Phi_m
    cond = float(np.linalg.cond(S))
    return np.linalg.solve(S, c), cond


# ─── 5. La mappa completa ─────────────────────────────────────────────────────

def quarterly_to_monthly(Phi: np.ndarray, Sigma: np.ndarray, n: int, *,
                         const: np.ndarray | None = None,
                         shortcut: bool = True,
                         variant: str = DEFAULT_COUPLING) -> MonthlyMap:
    """
    I quattro passi di §2.4 insieme, per UNA estrazione.

    Parameters
    ----------
    Phi : (n*p, n*p)    la companion trimestrale — `QBVARFit.companion(s)`
    Sigma : (n, n)      Sigma_eps                — `QBVARFit.Sigma[s]`
    n : int             numero di variabili (non n*p)
    const : (n,) | None A_0                      — `QBVARFit.const[s]`
    variant : str       quale delle tre formule di accoppiamento — vedi
                        `coupling_matrix`.  Default: quella del codice degli
                        autori, che e' cio' che una replica deve replicare.

    Returns
    -------
    MonthlyMap
    """
    Phi = np.asarray(Phi, dtype=float)
    p = Phi.shape[0] // n
    Phi_m, diag = matrix_cube_root(Phi)
    A = coupling_matrix(Phi_m, n, variant=variant)
    Sigma_m = monthly_innovation_cov(A, Sigma, shortcut=shortcut)

    const_m = None
    if const is not None:
        const_m, cond_c = monthly_constant(Phi_m, const)
        diag["cond_const"] = cond_c

    diag["A_spectral_radius"] = float(np.abs(np.linalg.eigvals(A)).max())
    diag["Sigma_m_min_eig"] = float(np.linalg.eigvalsh(Sigma_m).min())
    return MonthlyMap(Phi_m=Phi_m, Sigma_m=Sigma_m, const_m=const_m, A=A,
                      n=n, p=p, diagnostics=diag, variant=variant)


__all__ = [
    "MonthlyMap",
    "select_cube_roots",
    "matrix_cube_root",
    "coupling_matrix",
    "DEFAULT_COUPLING",
    "monthly_innovation_cov",
    "monthly_constant",
    "quarterly_to_monthly",
    "REAL_TOL",
    "IMAG_TOL",
]
