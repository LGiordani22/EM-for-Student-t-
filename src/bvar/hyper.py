"""
src/bvar/hyper.py

IL SALTO GERARCHICO: lambda, psi, mu diventano variabili casuali.

Fin qui (`niw.py`) i parametri del VAR erano in forma chiusa CONDIZIONATAMENTE
agli iperparametri.  Qui gli iperparametri smettono di essere costanti da
fissare a occhio e diventano essi stessi oggetto di inferenza, con un loro
iperprior e un loro posterior.  E' cio' che rende il modello GERARCHICO, ed e'
la ragione per cui il paper cita Giannone, Lenza & Primiceri (2015) e non solo
Litterman.

Cimadomo §2.1: "As in Giannone et al. (2015), we treat these hyperparameters as
random variables, and we draw them from their posterior distributions.  For the
hyperparameters, we choose the same rather diffuse priors described in Giannone
et al. (2015)."

Fonte vincolante: GLP (2015) §III (gli iperprior) e Appendice B (il Metropolis).


L'ARCHITETTURA A DUE STRATI
============================
    strato 2   gamma = (lambda, mu, psi)   NON coniugato  ->  Metropolis
                     |
                     v
    strato 1   (B, Sigma) | gamma          coniugato      ->  forma chiusa

Ogni proposta di gamma richiede di ricostruire il prior, ricalcolare il
posterior e valutare la marginal likelihood.  E' costoso ma esatto: non ci sono
approssimazioni fra i due strati.


GLI IPERPRIOR (GLP §III)
=========================
"As hyperpriors for lambda, mu, and delta, we choose GAMMA densities with mode
equal to 0.2, 1, and 1, the values recommended by Sims and Zha (1998), and
standard deviations equal to 0.4, 1, and 1, respectively."

(delta e' il dummy-initial-observation, che noi teniamo FUORI dal default —
Decisione 4, per fedelta' a Cimadomo §2.1 che elenca tre iperparametri.)

"Finally, the choice of the hyperprior for each element of the vector
psi/(d-n-1), that is, the prior mean of the main diagonal of Sigma, should be
loosely related to the scale of the variables in the model.  We pick an
INVERSE-GAMMA with scale and shape equal to (0.02)^2 because it seems
appropriate for our data expressed in annualized log-terms.  This hyperprior
peaks at approximately (0.02)^2, and it is proper but quite disperse since it
does not have either a variance or a mean."

Nota che psi/(d-n-1) = psi, perche' d = n+2 (Blocco 1).

UN PUNTO DA RI-CONTROLLARE SUI psi STIMATI  <<< IMPORTANTE
-----------------------------------------------------------
GLP calibra quel (0.02)^2 su dati "expressed in annualized log-terms".  Il
NOSTRO pannello e' misto: 24 serie in log-livello (residui dell'ordine di
0.005-0.01, per cui 0.02 e' la scala giusta) ma 6 serie in LIVELLO — UNRATE,
TCU, ISM_PMI, ISM_PRICES, ISM_EMP, Philly — i cui residui sono dell'ordine di
0.2 - 3, cioe' 10-150 volte piu' grandi.  Un unico iperprior centrato su 0.02 e'
quindi mal scalato per quelle sei.

DECISIONE PRESA: si implementa GLP ESATTAMENTE com'e' scritta e si MISURA.
Le ragioni: (a) quell'inverse-Gamma e' estremamente diffusa — shape = 4e-4, e
GLP stessa dice che "does not have either a variance or a mean" — quindi i dati
dovrebbero dominare senza fatica; (b) il paper ha lo STESSO problema con le sue
variabili in livello (UNRATE, FEDFUNDS, spread, PMI, EPU) e ci convive; (c) non
si devia dalla fonte-verita' su un sospetto.

DA FARE AL RECOVERY TEST E SUI DATI VERI: guardare i psi stimati SERIE PER
SERIE.  Se quelli delle sei serie in livello risultano schiacciati verso 0.02
invece di andare dove li porterebbero i dati, si torna qui con il numero in mano
e si valuta un iperprior per-serie.  Finche' non c'e' quel numero, non si tocca.


LA PARAMETRIZZAZIONE DELLA PROPOSTA: UNA SCELTA ALGORITMICA, DICHIARATA
=======================================================================
L'Appendice B dice: "Draw a candidate value of the hyperparameters gamma* from a
GAUSSIAN PROPOSAL DISTRIBUTION, with mean equal to gamma^(j-1) and variance
equal to c*W, where W is the inverse Hessian of the negative of the log-posterior
of the hyperparameters at the peak, and c is a scaling constant chosen to obtain
an acceptance rate of approximately 20 percent."

Ma NON dice su quale scala.  lambda, mu e psi sono tutti POSITIVI: una proposta
gaussiana su gamma direttamente puo' proporre valori negativi, che hanno prior
nullo e vengono rifiutati — corretto ma inefficiente, e vicino allo zero
l'accettazione crolla e la catena si impasta contro il bordo.

SCELTA: si propone in LOG, con lo Jacobiano esplicito nel target.  Se
theta = log(gamma), allora

    p(theta) = p(gamma) * |d gamma / d theta| = p(gamma) * gamma
    log p(theta) = log p(gamma) + sum(log gamma)

E' una scelta di ALGORITMO, non di modello: il posterior campionato e'
identico, cambia solo l'efficienza di mescolamento.  Va dichiarata in tesi come
INTERPRETAZIONE dell'Appendice B, non come lettura letterale — e la fedelta' e'
DIMOSTRATA, non asserita: `tests/test_gate1.py` verifica che le due
parametrizzazioni diano lo stesso posterior.

L'ottimizzatore per il modo (passo 1 dell'Appendice B, che dice solo "requires a
numerical maximization" senza specificare come) e' `scipy.optimize.minimize`.
Libertà implementativa, nessun vincolo dal paper.


================================================================================
IL MESCOLAMENTO — perche' questa catena STRISCIA, e perche' va bene lo stesso
================================================================================
Questa sezione e' scritta per chi legge la tesi senza aver mai tarato un
Metropolis.  E' il risultato metodologico del pacchetto, e i numeri che la
sostengono si rifanno con `python -m src.bvar.tests.test_mixing`.


--- 1. CHE COSA VUOL DIRE "MESCOLARE" ---------------------------------------

Una catena MCMC non produce estrazioni indipendenti: produce una PASSEGGIATA
che, se la si lascia camminare abbastanza, visita ogni regione della posteriore
con la frequenza giusta.  "Mescolare bene" vuol dire attraversare in fretta
tutta la distribuzione; "mescolare male" vuol dire STRISCIARE — muoversi per
passi cosi' piccoli, o cosi' correlati fra loro, che mille iterazioni
raccontano quello che ne racconterebbero venti indipendenti.

La misura si chiama EFFECTIVE SAMPLE SIZE (ESS):

    ESS = S / tau        con   tau = 1 + 2 * somma_{t>=1} rho_t

dove S sono le estrazioni tenute e rho_t l'autocorrelazione a distanza t.  Si
legge letteralmente: **a quante estrazioni INDIPENDENTI equivale la catena**.
tau e' il "tempo di autocorrelazione integrato", cioe' ogni quante iterazioni la
catena si dimentica di dov'era.

    ESS/iterazione = 1.00    ogni estrazione e' nuova informazione (iid)
    ESS/iterazione = 0.05    ne servono ~20 per una indipendente
    ESS/iterazione = 0.015   ne servono ~67; 1000 estrazioni valgono 15

Il legame con quel che interessa e' diretto: l'errore Monte Carlo su una media a
posteriori e' sd/sqrt(ESS), NON sd/sqrt(S).  Un ESS basso non introduce
distorsione — la catena resta valida e converge alla posteriore giusta — ma
rende la stima IMPRECISA, come se si fossero raccolte molte meno estrazioni.

L'autocorrelazione a lag 1 e' la stessa cosa vista da vicino: rho_1 = 0.986
significa che il valore di λ a questa iterazione e' quasi identico a quello di
prima.  Da li' tau esplode, e l'ESS con lui.

**ESS e ACCETTAZIONE non sono la stessa cosa, e il B-BVAR lo dimostra.**
L'accettazione dice solo in quale quadrante sei: proporre passi enormi fa
rifiutare tutto (catena ferma), proporre passi minuscoli fa accettare tutto
(catena ferma lo stesso).  L'ottimo teorico per un random-walk in dimensione
alta e' 0.234, e l'Appendice B chiede "circa 20%".  Ma si puo' avere
l'accettazione perfetta e l'ESS pessimo: e' esattamente il B-BVAR, 20.8% di
accettazione e ESS/iterazione 0.015.  Vuol dire **accettare quasi sempre, e
muoversi pochissimo**.


--- 2. LA CRESTA: c'e', ma NON e' la causa principale (misurato) -------------

La spiegazione naturale — e quella che avevamo scritto per prima — e' la
CRESTA.  Gli iperparametri del BVAR non sono indipendenti nella posteriore, e
non possono esserlo: **fanno tutti la stessa cosa da direzioni diverse**, cioe'
regolano quanto il prior stringe rispetto ai dati.

  * lambda e' la tightness complessiva della Minnesota.  La deviazione standard
    a priori del coefficiente della variabile j al lag s e' proporzionale a
    lambda * sqrt(psi_j) / (s * sqrt(psi_i)) — vedi `dummies.py`, "L'EQ. (3)
    FATTORE PER FATTORE".  Quindi **lambda e i psi entrano moltiplicati**: se
    lambda raddoppia e i psi si dividono per quattro, il prior sui coefficienti
    e' quasi lo stesso.  Non e' una correlazione accidentale, e' una quasi
    ridondanza della parametrizzazione;
  * mu governa il sum-of-coefficients, che tira la somma dei lag propri verso 1
    (o verso 0 sulle `wn`).  E' un altro modo di irrigidire lo stesso oggetto —
    la persistenza — che la Minnesota gia' governa.  Da cui la tensione
    documentata in `dummies.py`, "LA TENSIONE wn <-> soc": mu e lambda si
    compensano;
  * i psi sono le scale delle n equazioni, e nel BVAR la scala e' l'unita' di
    misura dello shrinkage: cambiarle tutte insieme di un fattore e' quasi
    equivalente a cambiare lambda.

Su una cresta, un random-walk Metropolis striscia per costruzione: la proposta
e' gaussiana con covarianza c*W, la STESSA a ogni iterazione.  Se il passo e'
grande abbastanza da percorrere la cresta in fretta esce lateralmente, dove la
densita' crolla, e viene rifiutato; se e' piccolo abbastanza da restarci dentro,
lungo la cresta avanza di pochissimo.  Il passo lo governa la direzione piu'
STRETTA, il progresso quella piu' LUNGA.

MA LA MISURA DICE CHE QUI NON E' QUESTO IL PROBLEMA, e va detto — perche' e'
una spiegazione che suona bene e sarebbe sbagliata.  `test_mixing` §1 stampa la
matrice di correlazione implicata da W al modo, sul pannello vero (Q-BVAR,
n=30):

    corr(lambda, mu)                     -0.410
    |corr| massima fuori diagonale       -0.496   (psi[HSN1F] vs psi[UNRATE])
    |corr| MEDIANA fuori diagonale        0.028
    anisotropia (autoval. max/min di R)      9x
    cond(W)                                58.8

Cioe': le correlazioni ci sono e sono quelle attese — lambda contro mu a -0.41,
esattamente la tensione Minnesota <-> sum-of-coefficients — ma **la posteriore
non e' una cresta patologica**.  La correlazione tipica fra due coordinate e'
0.03, e cond(W) = 59 e' un condizionamento mite.

E c'e' una ragione strutturale per cui non poteva essere quello: **la proposta
dell'Appendice B usa gia' W, l'Hessiana inversa al modo.**  W E' la forma locale
della posteriore, correlazioni comprese: proporre con covarianza c*W significa
proporre GIA' allineati alla cresta.  Cio' che W non puo' correggere e' che la
cresta sia curva (W e' fissa, valutata in un punto solo) — ma con
un'anisotropia di 9x quel residuo e' piccolo.

E' anche il motivo per cui alzare l'accettazione non serve — provato, sul
B-BVAR l'abbiamo portata al 20.8% e l'ESS non si e' mosso.  Se il problema
fosse un passo mal tarato, si sbloccherebbe.

Resta un fatto da spiegare, ed e' grosso: l'ESS/iterazione e' 0.0087 a d=32.
Se non e' la correlazione, che cos'e'?


--- 3. LA LEGGE 1/d: IL COSTO E' LA DIMENSIONE, E BASTA ---------------------

E' la DIMENSIONE, e la misura lo dice in modo netto.  Per un
random-walk Metropolis in dimensione d, la teoria del regime asintotico
(Roberts, Gelman & Gilks 1997) dice che anche con la scala OTTIMALE l'efficienza
per iterazione decade come 1/d:

    ESS/iterazione  x  d  ~=  costante

La ragione intuitiva: la proposta muove tutte e d le coordinate insieme, e la
probabilita' che una mossa congiunta sia buona in TUTTE le direzioni cala con d.
Per restare accettabili i passi vanno accorciati come 1/sqrt(d), e il percorso
compiuto per iterazione si riduce di conseguenza.

Qui d = 2 + n: lambda, mu e un psi per equazione.  Quindi

    Q-BVAR (n=30)    d = 32
    L-BVAR (n=37)    d = 39
    B-BVAR (n=84)    d = 86      <- il blocking TRIPLICA i mensili, e con loro d

**Il B-BVAR mescola peggio non perche' sia un modello peggiore, ma perche' il
blocking gli raddoppia la dimensione degli iperparametri.**  L'esperimento che
lo isola e' `tests/test_mixing`: stesso modello (Q-BVAR), stessa T, stesso seme,
stesso Metropolis, e si muove SOLO d togliendo colonne al pannello.  Misurato,
5000 estrazioni per cella, 2 semi, c tarato al 20% in ogni cella:

    n     d    acc      c      ESS/it lam   rho1 lam   ESS/it x d
    6     8   17.3%   1.091      0.0355      0.938        0.28
    12   14   18.2%   0.514      0.0269      0.955        0.38
    18   20   20.5%   0.307      0.0117      0.973        0.23
    24   26   19.5%   0.285      0.0104      0.976        0.27
    30   32   19.9%   0.220      0.0087      0.977        0.28

    psi congelati, si campiona (lambda, mu):
          2   19.9%   9.403      0.1099      0.781        0.22

    pendenza di log(ESS/it) su log(d):  -1.10   (la previsione 1/d e' -1.00)
    prodotto ESS/it x d:  fra 0.22 e 0.38, spread 1.6x

Tre cose da leggere in questa tabella:

  1. **il prodotto ESS/it x d e' piatto** su un fattore 4 di dimensione (d da 8
     a 32) e la pendenza e' -1.10 contro la previsione teorica -1.00.  Il
     mescolamento e' governato dalla dimensione, punto;
  2. **l'accettazione e' a bersaglio in OGNI cella** (17-21%): non e' taratura
     sbagliata.  Si vede anche in c, che scende da 1.09 a 0.22 al crescere di d
     — la taratura 1/sqrt(d), trovata dalla procedura senza che gliela si
     imponga;
  3. **congelare i psi (d=2) non e' speciale.**  L'ESS salta 13x, ma il prodotto
     resta 0.22, in linea con tutti gli altri.  Cioe' i psi non sono un blocco
     patologico da isolare: sono 30 coordinate come le altre, e il guadagno del
     blocking e' *esattamente* quello che la legge 1/d prevede (32/2 = 16x
     previsto, 13x misurato).  **Non c'e' niente da riparare.**

Il confronto con i due modelli veri chiude il cerchio (d, ESS/it, prodotto):
L-BVAR 39, 0.053, 2.07;  B-BVAR 86, 0.015, 1.29.  La COSTANTE e' diversa da
quella del Q-BVAR (0.28) — dipende da modello e dati, e non c'e' ragione perche'
sia la stessa — ma i due modelli veri stanno entro 1.6x l'uno dall'altro, e
l'ordinamento e' quello che la legge prevede.

Che sia la dimensione lo si vede anche nel codice degli autori, senza che loro
lo dicano: `MCMCconst` e' cablato a mano, un valore per modello, e vale 1 nel
Q-BVAR, 0.5 nel C-BVAR, **0.14 nel B-BVAR** e 1.6 nell'L-BVAR.  Cioe' hanno
dovuto stringere la proposta di un ordine di grandezza proprio dove d e' piu'
grande.  E' la taratura 1/sqrt(d) fatta a occhio, ed e' la traccia del fenomeno
nel loro codice.


--- 4. E ALLORA PERCHE' IL NOWCAST E' SANO ----------------------------------

Perche' **il nowcast non e' un funzionale della catena lenta.**  Il ciclo ha due
popolazioni, e vanno tenute distinte:

    lambda, mu, psi        Metropolis           ESS/iterazione ~ 0.01-0.05
    B, Sigma, pannello     coniugata / smoother ESS/iterazione ~ 0.95-1.00

La seconda riga non e' un colpo di fortuna, e' come e' costruito il ciclo:

  * dato gamma, l'estrazione di (B, Sigma) e' ESATTA dalla Normal-Inverse-Wishart
    — forma chiusa, nessun rifiuto, nessuna memoria dell'iterazione precedente;
  * il simulation smoother gira con randomness FRESCA a ogni iterazione;
  * il nowcast e' una funzione di (B, Sigma, stato), non di gamma.

Gamma entra solo indirettamente, come regolatore dello shrinkage — e la
posteriore di gamma e' concentrata: sd(lambda) ~ 0.016.  Quindi la lentezza di
gamma modula un ingrediente che quasi non varia, e non si propaga.

MISURATO su DUE modelli (Gate 4 e 5):

    ESS del nowcast              1000 / 1000     (L-BVAR ~1000/1100)
    corr(nowcast, lambda)        -0.047
    corr(nowcast, mu)            -0.010          (L-BVAR: -0.045)
    mediana del nowcast, estrazioni con mu alto vs basso:  -0.14 pp,
                                                  dentro l'errore Monte Carlo

Cioe' **la densita' del nowcast e' di fatto iid** anche mentre gli
iperparametri strisciano.

IL LIMITE ONESTO, da scrivere cosi' e non piu' forte di cosi': le bande
esplorano bene l'incertezza sui PARAMETRI e sullo STATO, meno bene quella sugli
IPERPARAMETRI.  E' un'attenuante MISURATA — sd(lambda) piccola, correlazione ~0
— non un'assoluzione teorica.  Se il deliverable fosse una banda SU lambda, il
problema sarebbe reale; il deliverable e' il nowcast, e li' l'impatto e'
trascurabile.


--- 5. PERCHE' NON SI "RISOLVE" ---------------------------------------------

Perche' e' una proprieta' dell'algoritmo che il paper prescrive, non un difetto
dell'implementazione.  E perche' la misura toglie di mezzo la via d'uscita che
sembrava piu' promettente.

**Il blocking dei psi non e' un rimedio, e' un CAMBIO DI MODELLO.**  L'esito del
§3 e' chiaro: bloccare rende ESS/iterazione ~ 1/d_blocco, cioe' guadagna
esattamente quel che si toglie in dimensione, ne' piu' ne' meno.  Congelare i
psi al modo fa 13x sull'ESS, ma i psi smettono di essere iperparametri: la loro
incertezza sparisce dalla posteriore invece di essere campionata male.  Si
comprerebbe una diagnostica migliore rinunciando a un pezzo di inferenza, e per
di piu' su un oggetto — la banda degli iperparametri — che nel §4 abbiamo
misurato non contare per il nowcast.  (L'interruttore esiste anche da loro:
`bvarGLP_fixedhyp` con `'MNpsi', 0`, riga commentata di `bbvar.m` r.35, dove i
psi restano fissi a `SS`.)

Le altre vie — blocchi piu' piccoli aggiornati a rotazione, o un campionatore
con gradiente (MALA, HMC), che scala come d^(1/3) o d^(1/4) invece che 1/d —
sono **deviazioni dall'Appendice B**, e il beneficio ricadrebbe su una banda che
non e' il deliverable.  Costo alto, guadagno su cio' che non serve: si descrive,
non si risolve.

TENTATIVI FATTI E FALLITI, tenuti perche' il percorso e' istruttivo:

  * taratura di c e spazzate multiple sull'L-BVAR: ESS/iterazione 0.053 -> 0.030,
    PEGGIORATO.  Li' agisce anche il target mobile (il pannello latente cambia a
    ogni iterazione, quindi le spazzate multiple equilibrano gli iperparametri
    su un pannello che poi cambia).  Vedi l'header di `lbvar.py`;
  * alzare l'accettazione al 20% sul B-BVAR: fatto, ESS invariato.  Vedi §2.

QUEL CHE FANNO GLI AUTORI, per il confronto in tesi.  Nel loro codice di replica
**non c'e' nessun ESS, nessun R-hat, nessuna autocorrelazione, nessun thinning**:
l'unica diagnostica e' `r.mcmc.ACCrate`, calcolata a posteriori come frazione di
iterazioni in cui lambda e' cambiato, e mai usata per agire.  L'unico accenno
alla convergenza in tutto il pacchetto e' il commento su `Ndrawsdiscard`
("number of draws initially discarded to allow convergence", `setpriors.m`
r.54).  **Con la sola accettazione a bersaglio il loro campionatore sembra
sano** — ed e' precisamente la situazione del nostro B-BVAR.  Che il
mescolamento degli iperparametri sia povero e' un fatto che il loro strumentario
diagnostico non puo' vedere: misurarlo, quantificarlo e mostrare che non
contamina il deliverable e' un contributo di riproducibilita' a tutti gli
effetti.


================================================================================
CHE COSA SONO DAVVERO I TRE IPERPARAMETRI  —  la sintesi della §2.1
================================================================================
Torniamo all'equazione del VAR (la forma completa e' in `niw.py`):

    Y = X B + U        eps_t ~ N(0, Sigma)

Le incognite del MODELLO sono due: B (i coefficienti) e Sigma (la matrice di
varianza-covarianza degli errori veri del VAR).  Gli IPERPARAMETRI non sono
altre incognite dello stesso tipo: sono i parametri del PRIOR su quelle due.


psi  —  UN PRIOR SULLA COVARIANZA DEGLI ERRORI DEL VAR
------------------------------------------------------
psi e' la diagonale della matrice di scala Psi dell'Inverse-Wishart su Sigma:

    Sigma ~ IW(Psi, dof),     Psi = diag(psi),     dof = n + 2

e siccome dof = n+2 implica dof - n - 1 = 1, la media a priori e'

    E(Sigma) = Psi / (dof - n - 1) = Psi = diag(psi)

Quindi, senza giri di parole: PSI_I E' LA VARIANZA ATTESA A PRIORI DELL'ERRORE
DELL'EQUAZIONE i DEL VAR.  Si', e' letteralmente un prior sulla matrice di
varianza-covarianza degli errori veri del VAR — sulla sua DIAGONALE, perche'
Psi e' diagonale: a priori non diciamo nulla sulle correlazioni fra equazioni,
quelle le impara il posterior (ed e' `psi_bar`, che infatti NON e' diagonale).

E' anche il posto da cui la scala delle variabili entra nel modello: e' per
questo che i dati NON vanno standardizzati a monte (lo scaling lo fa il prior
via psi, eq. 3 fattore 3), e per questo che psi compare al denominatore
dell'eq. (3).

I nostri numeri: sqrt(psi) va da 0.0084 (PCEPILFE, log-livello) a 61.6 (Philly,
livello) — sei ordini di grandezza, esattamente le scale delle serie.


lambda  —  LA TIGHTNESS COMPLESSIVA DELLA MINNESOTA
---------------------------------------------------
Moltiplica tutte le varianze a priori dei coefficienti (eq. 3).  NON sposta il
centro: sposta quanto i dati possono allontanartene.  lambda -> 0 il posterior
coincide col prior; lambda -> inf si ricade nell'OLS senza prior.  Ed e' anche,
letteralmente, la sd a priori del coefficiente del proprio primo lag.
Trattazione completa in `dummies.py`, sezione "LAMBDA E I DUE CASI LIMITE".

I nostri numeri (profilo q_b, pannello del Gate 2): lambda ~ 0.52, cioe' "quel
coefficiente e' 1 +/- 0.52".


mu  —  L'INTENSITA' DEL SUM-OF-COEFFICIENTS
-------------------------------------------
Governa quanto forte si spinge Pi = I - A_1 - ... - A_p verso zero, cioe' quanto
si limita la componente deterministica di un VAR in livelli.  Compare al
DENOMINATORE della dummy (y+ = diag(y0_bar/mu)), quindi mu piccolo = dummy
grande = prior forte — stesso verso di lambda.  Trattazione completa in
`dummies.py`.

I nostri numeri (profilo q_b, pannello del Gate 2): mu ~ 2.25.  Attenzione al verso: mu sta al
DENOMINATORE della dummy, quindi mu grande = soc DEBOLE.


IL SALTO GERARCHICO: perche' GLP e non solo Litterman
------------------------------------------------------
Questi tre NON sono fissati a mano.  Litterman li sceglieva a occhio o
ottimizzando la previsione su un presample; BGR (2010) fissa psi con la varianza
dei residui di un AR(p) univariato.  GLP li tratta come VARIABILI CASUALI con un
iperprior, e li CAMPIONA dal loro posterior

    p(gamma | Y)  proporzionale a  ML(Y | gamma) * p(gamma)

dove la marginal likelihood ML(Y|gamma) — la densita' dei dati dopo aver
integrato via B e Sigma — misura quanto bene QUEL grado di shrinkage prevede i
dati un passo avanti, con la penalita' per complessita' incorporata
automaticamente (il rasoio di Occam bayesiano: A.4).  E' cio' che significa
"lasciare che siano i dati a scegliere il grado di shrinkage": prior piu'
stretti quando i parametri sono tanti rispetto ai dati, piu' larghi nel caso
opposto, senza che nessuno lo imponga dall'esterno.

PERCHE' CI IMPORTA PER LA TESI: con gli iperparametri FISSATI, le bande di
previsione condizionano su valori creduti noti, e sono percio' TROPPO STRETTE.
Campionandoli, ogni draw del forecast usa un (lambda, mu, psi) diverso e la
densita' predittiva finale incorpora anche l'incertezza sul GRADO DI SHRINKAGE.
Per un lavoro il cui deliverable e' una densita' e non un punto, e' la
differenza fra bande oneste e bande ottimistiche.  Cimadomo §2 lo rivendica
esplicitamente come vantaggio sui DFM: "we produce probabilistic forecasts that
reflect ALL sources of uncertainty, including that coming from the setting of
hyperparameters underlying the prior distributions".
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from src.bvar.dummies import initial_observation_mean, sum_of_coefficients_dummy
from src.bvar.niw import build_prior, log_ml_with_dummies
from src.bvar.spec import BVARSpec, Hyper


# ─── 1. Gli iperprior (GLP §III) ──────────────────────────────────────────────

def gamma_from_mode_sd(mode: float, sd: float) -> tuple[float, float]:
    """
    Converte (moda, deviazione standard) nei parametri (shape, scale) di una
    Gamma.  GLP specifica gli iperprior in termini di moda e sd, non di shape e
    scale, quindi la conversione serve.

    Per Gamma(a, theta):  moda = (a-1) theta,  sd = sqrt(a) theta.
    Ponendo u = sqrt(a):  sd*(u - 1/u) = moda  ->  sd*u^2 - moda*u - sd = 0
                          u = (moda + sqrt(moda^2 + 4 sd^2)) / (2 sd)
    """
    u = (mode + np.sqrt(mode ** 2 + 4.0 * sd ** 2)) / (2.0 * sd)
    return float(u ** 2), float(sd / u)


@dataclass(frozen=True, eq=False)
class HyperPrior:
    """
    Gli iperprior di GLP §III.  I default SONO i valori del paper.

    lam_mode/lam_sd : 0.2 / 0.4   (Sims & Zha 1998, via GLP)
    mu_mode/mu_sd   : 1.0 / 1.0
    psi_scale/shape : (0.02)^2 entrambi — vedi la nota nell'header sul
                      ri-controllo dei psi stimati.
    """

    lam_mode: float = 0.2
    lam_sd: float = 0.4
    mu_mode: float = 1.0
    mu_sd: float = 1.0
    psi_shape: float = 0.02 ** 2
    psi_scale: float = 0.02 ** 2
    lam_max: float | None = None
    mu_max: float | None = None

    def log_pdf(self, lam: float, mu: float, psi: np.ndarray) -> float:
        """
        Log densita' congiunta dell'iperprior (i tre blocchi indipendenti).

        I LIMITI SUPERIORI `lam_max` / `mu_max` sono `None` per default — GLP
        §III non li ha.  Il codice di replica degli autori invece li impone come
        rifiuto secco (`lbvar.m` righe 33-34 e 100):

            lambda_max = 5;  miu_max = 5;
            if lambda_new<lambda_min || lambda_new>lambda_max || ... p_acc=0;

        Metterli QUI, come -inf nell'iperprior, e' ESATTAMENTE equivalente a
        quel rifiuto — e non a valle contando le violazioni, che le lascerebbe
        entrare nella catena.  I limiti INFERIORI non servono: proponiamo in
        scala log, quindi la positivita' e' gratis.
        """
        if lam <= 0 or mu <= 0 or np.any(psi <= 0):
            return -np.inf
        if (self.lam_max is not None and lam > self.lam_max) or            (self.mu_max is not None and mu > self.mu_max):
            return -np.inf
        out = 0.0
        for val, (mode, sd) in ((lam, (self.lam_mode, self.lam_sd)),
                                (mu, (self.mu_mode, self.mu_sd))):
            a, th = gamma_from_mode_sd(mode, sd)
            out += (a - 1.0) * np.log(val) - val / th - gammaln(a) - a * np.log(th)
        # inverse-Gamma(shape, scale) su ogni psi_i
        a, b = self.psi_shape, self.psi_scale
        out += float(np.sum(a * np.log(b) - gammaln(a)
                            - (a + 1.0) * np.log(psi) - b / psi))
        return float(out)


# ─── 2. Il target: log-posterior degli iperparametri ──────────────────────────

@dataclass(frozen=True, eq=False)
class HyperTarget:
    """
    La funzione che il Metropolis campiona:

        log p(gamma | Y)  =  log ML(Y | gamma)  +  log p(gamma)   + cost.

    Il primo termine e' GLP (A.14) col trucco delle dummy (A.3), il secondo e'
    l'iperprior.  Tenuti insieme qui perche' il Metropolis li valuta sempre
    in coppia.
    """

    spec: BVARSpec
    Y: np.ndarray               # (T, n)  dati veri (gia' senza le prime p righe)
    X: np.ndarray               # (T, k)
    y0_bar: np.ndarray          # (n,)    per il blocco soc
    hyperprior: HyperPrior
    # I due momenti del blocco dati che NON dipendono dagli iperparametri.
    # Calcolati una volta in `build_target` e riusati a ogni valutazione: e' la
    # cache descritta in `niw.log_ml_with_dummies`.  Al modo del profilo `l`
    # sono migliaia di valutazioni sullo STESSO pannello, e X'X da solo pesa il
    # 26% di una valutazione.
    XtX: np.ndarray | None = None       # (k, k)
    XtY: np.ndarray | None = None       # (k, n)

    @property
    def moments(self) -> tuple[np.ndarray, np.ndarray] | None:
        return None if self.XtX is None else (self.XtX, self.XtY)

    def unpack(self, gamma: np.ndarray) -> Hyper:
        """gamma = [lam, mu, psi_1..psi_n]  ->  Hyper."""
        return Hyper(lam=float(gamma[0]), mu=float(gamma[1]),
                     psi=np.asarray(gamma[2:], dtype=float))

    def log_posterior(self, gamma: np.ndarray) -> float:
        """Su scala NATURALE (gamma)."""
        g = np.asarray(gamma, dtype=float)
        if np.any(g <= 0) or not np.all(np.isfinite(g)):
            return -np.inf
        try:
            hyp = self.unpack(g)
        except ValueError:
            return -np.inf
        lp = self.hyperprior.log_pdf(hyp.lam, hyp.mu, hyp.psi)
        if not np.isfinite(lp):
            return -np.inf
        prior = build_prior(self.spec, hyp)
        Yd, Xd = sum_of_coefficients_dummy(self.spec, hyp, self.y0_bar)
        try:
            ml = log_ml_with_dummies(self.Y, self.X, Yd, Xd, prior,
                                     moments=self.moments)
        except np.linalg.LinAlgError:
            return -np.inf
        return -np.inf if not np.isfinite(ml) else float(ml + lp)

    def log_posterior_log_scale(self, theta: np.ndarray) -> float:
        """
        Su scala LOG, con lo Jacobiano — e' il target del Metropolis.

            log p(theta|Y) = log p(gamma|Y) + sum(theta)

        perche' |d gamma / d theta| = prod(gamma) = exp(sum(theta)).
        """
        th = np.asarray(theta, dtype=float)
        if not np.all(np.isfinite(th)):
            return -np.inf
        val = self.log_posterior(np.exp(th))
        return -np.inf if not np.isfinite(val) else float(val + np.sum(th))


def build_target(spec: BVARSpec, panel: np.ndarray,
                 hyperprior: HyperPrior | None = None) -> HyperTarget:
    """
    Costruisce il target a partire dal pannello (T, n) DENSO in unita' del
    modello.  Fa il lavoro di impilamento dei ritardi una volta sola: dentro il
    Metropolis si valuta solo la ML.
    """
    panel = np.asarray(panel, dtype=float)
    if np.isnan(panel).any():
        raise ValueError(
            "il pannello contiene NaN: il core non deve mai vederne "
            "(vedi src/bvar/data.py::assert_dense)."
        )
    p, n = spec.p, spec.n
    T = panel.shape[0]
    Y = panel[p:]
    lags = [panel[p - s: T - s] for s in range(1, p + 1)]
    X = np.column_stack(lags + [np.ones(T - p)])
    return HyperTarget(
        spec=spec, Y=Y, X=X,
        y0_bar=initial_observation_mean(panel, p),
        hyperprior=hyperprior or HyperPrior(),
        XtX=X.T @ X, XtY=X.T @ Y,
    )


# ─── 3. Il modo a posteriori (Appendice B, passo 1) ───────────────────────────
#
# QUESTA SEZIONE SEGUE `bvarGLPmf.m` RIGA PER RIGA, E LA STORIA E' ISTRUTTIVA
# ============================================================================
# Una prima versione massimizzava con Nelder-Mead in scala log, partendo da
# psi = var(Y).  Sul profilo `l` vero (n=37, p=17, k=630) e' stata MISURATA:
#
#     una valutazione della log-posterior      220 ms
#     Nelder-Mead a maxfev = 2000*(2+n)        4.78 h   <- e senza convergere
#     Hessiana numerica                       11.5 min
#
# 4.8 ore che NON sono convergenza: Nelder-Mead in 39 dimensioni non raggiunge
# `xatol=1e-4`, esaurisce il budget e si ferma.
#
# (Una stima iniziale dava il ciclo MCMC a 0.44 s/iterazione, e quindi il modo a
# 260 volte l'intero pilota.  Sbagliata di 70x: contava solo le due valutazioni
# di ML del Metropolis, mentre il costo vero e' il SIMULATION SMOOTHER — stato
# companion di dimensione n*p = 629 su 479 mesi — a ~31 s/iterazione.  Misurato
# al pilota: 31 min di modo su 108 di run.  Il modo andava sistemato lo stesso.)
#
# Il codice degli autori risolve tutte e tre le cose, e nessuna era `maxiter`:
#
#   1. OTTIMIZZATORE.  `csminwel` (quasi-Newton di Sims, gradiente numerico),
#      non un metodo a simplesso.  bvarGLPmf.m riga 248.
#   2. TRASFORMAZIONE LOGISTICA SU UN BOX, non log.  Righe 216-240:
#          x = -log((MAX - g)/(g - MIN))
#      I limiti sono irraggiungibili per costruzione, quindi la regione dove la
#      log-posterior vale -inf NON ESISTE.  E' il motivo per cui loro non hanno
#      mai avuto il problema che ci aveva spinti verso Nelder-Mead.
#   3. PUNTO DI PARTENZA DEI psi.  Righe 205-213: `SS`, la varianza dei residui
#      di un AR(1) per serie — non var(Y).  Misurato sul pannello vero, il
#      rapporto var(Y)/SS ha mediana 225x e massimo 41395x (CPILFESL), e 22
#      serie su 37 cadono FUORI dal box degli autori.  La log-posterior al
#      nostro vecchio punto di partenza era 11037 nat peggiore del loro.
#
# Ma l'allineamento da solo non bastava, e si e' visto solo perche' il criterio
# di verifica non era "e' veloce" ma "E' UN MASSIMO", sondando +/- h su tutte e
# 39 le coordinate.  Servivano anche la regolarizzazione dell'Hessiana e il
# passo assoluto: vedi i due commenti piu' sotto, ciascuno accanto al suo pezzo.
#
# Misurato a valle di TUTTE e quattro le correzioni, profilo `l`:
#
#     31.3 min contro 4.78 h (9x), e stavolta convergente davvero
#     0 direzioni che migliorano, W definita positiva vera (cond 2e+02)
#     lambda = 0.5128, mu = 3.5669, entrambi interni, nessun psi al bordo
#     +2537 nat sopra il punto di partenza degli autori
#
# ATTENZIONE, errore da non ripetere: una versione precedente di questa nota
# riportava 6.4 min e lambda = 0.262, e ci costruiva sopra un "pattern monotono
# in n".  Quei numeri venivano da un modo NON CONVERGENTE, fermo 1400 nat sotto
# il vero.  Un ottimizzatore che si ferma da solo non ha per questo trovato un
# massimo — il sondaggio direzionale e' l'unico controllo che lo dimostra, ed e'
# per questo che `verify_mode` lo fa.  Il vero lambda a n=37 e' 0.5128, contro
# 0.5355 a n=30: cala nella direzione che BGR (2010) prevede, ma solo del 4%.
#
#
# IL BOX SUI psi — UNA SCOPERTA CODICE-vs-PAPER, NON UNA SCELTA NOSTRA
# --------------------------------------------------------------------
# `bvarGLPmf.m` righe 211-212 vincolano
#
#     MIN.psi = SS/100        MAX.psi = SS*100
#
# e `lbvar.m` righe 35-36 e 100 rifiutano seccamente le proposte MCMC fuori da
# quel box.  QUESTO NEL PAPER NON C'E'.  Ne' Cimadomo §2.1 ne' GLP §III
# menzionano limiti sui psi: GLP dichiara un iperprior inverse-Gamma "proper but
# quite disperse", e un box e' un prior in tutto e per tutto — tronca il
# supporto, e con esso il posterior.
#
# ADOTTATO, per la regola di sempre: dove il paper tace, decide il loro codice.
# Ma va DICHIARATO in tesi come divergenza paper<->codice, accanto alle altre
# gia' registrate (P0 = 0 in `lbvar.py` punto 3, il soc sulle wn in
# `dummies.py`).  Nota attenuante MISURATA: al modo del profilo `l` nessun psi
# tocca il bordo del box (0 in basso, 0 in alto), quindi qui il box vincola il
# PERCORSO dell'ottimizzatore ma non il RISULTATO.
#
#
# NIENTE JACOBIANO NEL MODO
# --------------------------
# `logMLVAR_formin.m` riga 161 nega una log-posterior NATURALE: la logistica e'
# puro artificio di vincolo, non cambio di misura.  Quindi qui si massimizza
# `log_posterior`, non `log_posterior_log_scale`.
#
# La W restituita resta invece in scala LOG, perche' la nostra proposta
# Metropolis e' in scala log.  Non c'e' contraddizione, ed e' un'identita' esatta
# e non un'approssimazione: se h(theta) = g(theta) + sum(theta) e' il target in
# scala log, il termine del Jacobiano e' LINEARE in theta, quindi
#
#     Hess h  ==  Hess g        (il Jacobiano sposta il gradiente, non la curvatura)
#
# Il modo naturale theta* = log(gamma*) e' un punto stazionario di g ma non di h;
# usarlo per valutare la curvatura e' legittimo e standard — serve la FORMA della
# proposta, non il suo centro esatto.

#: I limiti di `bvarGLPmf.m` righe 155-162 (e `lbvar.m` righe 31-36).
LAM_MIN, LAM_MAX = 1e-4, 5.0
MU_MIN, MU_MAX = 1e-4, 5.0
PSI_BOX_FACTOR = 100.0          #: MIN.psi = SS/100, MAX.psi = SS*100

#: Valori di partenza di `bvarGLPmf.m` righe 200-213.
LAM_START, MU_START = 0.2, 1.0

_XCLIP = 700.0                  #: evita overflow in exp(-x) ai bordi del box
_MODE_TOL = 1e-3                #: guadagno sotto il quale le ripartenze si fermano (nat)

#: Passo ASSOLUTO delle differenze centrate per il gradiente, in coordinate del
#: box logit.  E' l'analogo del `delta` di `numgrad.m` (loro 1e-6, in avanti);
#: 1e-3 e' misurato, vedi la tabella nel corpo di `find_mode`.
FD_STEP = 1e-3


def ar1_residual_var(Y: np.ndarray) -> np.ndarray:
    """
    `SS` di `bvarGLPmf.m` righe 205-213: la varianza dei residui di un AR(1) con
    intercetta, stimato per OLS separatamente su ogni serie.

    Il denominatore e' `nobs - nvar` (cioe' T-1-2), non `nobs`: e' quello di
    `ols1.m`, `sig2hatols = (res'res)/(nobs-nvar)`.

    E' il punto di partenza dei psi E il centro del loro box.  Ha senso: psi_i e'
    la varianza attesa a priori dell'errore dell'equazione i del VAR, e il
    residuo di un AR(1) univariato ne e' la stima piu' grezza e piu' robusta —
    esattamente la calibrazione di BGR (2010).  `var(Y)`, che usavamo prima, e'
    invece la varianza NON CONDIZIONATA: su serie in log-livello molto persistenti
    e' di ordini di grandezza piu' grande, e non e' affatto la stessa cosa.
    """
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    out = np.empty(n)
    Z = np.empty((T - 1, 2))
    Z[:, 0] = 1.0
    for i in range(n):
        Z[:, 1] = Y[:-1, i]
        beta, *_ = np.linalg.lstsq(Z, Y[1:, i], rcond=None)
        res = Y[1:, i] - Z @ beta
        out[i] = float(res @ res) / (Z.shape[0] - 2)        # ols1.m
    return out


def hyper_box(target: HyperTarget) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Il box `[MIN, MAX]` e il punto di partenza `gamma0` degli autori.

    I limiti superiori su lambda e mu vengono dall'IPERPRIOR quando li fissa
    (`HyperPrior.lam_max` / `mu_max`), altrimenti dai valori degli autori: cosi'
    il box del modo e il rifiuto secco del Metropolis restano lo stesso numero,
    che e' la situazione di `lbvar.m`.
    """
    hp = target.hyperprior
    SS = ar1_residual_var(target.Y)
    lam_hi = LAM_MAX if hp.lam_max is None else float(hp.lam_max)
    mu_hi = MU_MAX if hp.mu_max is None else float(hp.mu_max)
    lo = np.concatenate([[LAM_MIN, MU_MIN], SS / PSI_BOX_FACTOR])
    hi = np.concatenate([[lam_hi, mu_hi], SS * PSI_BOX_FACTOR])
    g0 = np.clip(np.concatenate([[LAM_START, MU_START], SS]), lo, hi)
    return lo, hi, g0


def _to_x(g: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """`x = -log((MAX-g)/(g-MIN))` — bvarGLPmf.m riga 216."""
    return -np.log((hi - g) / (g - lo))


def _to_g(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """`g = MIN + (MAX-MIN)/(1+exp(-x))` — logMLVAR_formin.m riga 13."""
    return lo + (hi - lo) / (1.0 + np.exp(-np.clip(x, -_XCLIP, _XCLIP)))


def find_mode(target: HyperTarget, gamma0: np.ndarray | None = None,
              *, maxiter: int = 500, n_restarts: int = 8
              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Massimizza numericamente il log-posterior degli iperparametri.

    Appendice B, passo 1: "Initialize the hyperparameters gamma at their
    posterior mode, which requires a numerical maximization."  Il paper non dice
    come; il codice degli autori si': box logistico + quasi-Newton, partendo da
    `SS`.  Vedi la trafila completa nel commento in cima a questa sezione.

    Parameters
    ----------
    gamma0 : (2+n,) or None
        None usa il punto degli autori (lambda=0.2, mu=1, psi=SS).
    n_restarts : int
        Quante volte rilanciare l'ottimizzatore dal punto raggiunto.  Si ferma
        appena il guadagno scende sotto `_MODE_TOL`.  Vedi il commento accanto
        al ciclo: con una chiamata sola il modo resta incompleto.

    Returns
    -------
    (gamma_mode, W) : (2+n,) e (2+n, 2+n)
        `W` e' l'Hessiana inversa nel picco, che l'Appendice B usa come
        covarianza della proposta.  E' in scala LOG, coerente con la proposta.
    """
    lo, hi, g0 = hyper_box(target)
    if gamma0 is not None:
        g0 = np.clip(np.asarray(gamma0, dtype=float), lo, hi)
    x0 = _to_x(g0, lo, hi)

    # Il box rende il dominio interamente ammissibile, quindi la penalita' non
    # dovrebbe mai scattare; resta come rete perche' la ML puo' comunque fallire
    # per ragioni di algebra lineare.
    _PEN = 1e12

    def neg_nat(x):
        v = target.log_posterior(_to_g(np.asarray(x, dtype=float), lo, hi))
        return _PEN if not np.isfinite(v) else -v

    # IL PASSO DEL GRADIENTE: ASSOLUTO, COME `numgrad.m` — E PERCHE'.
    #
    # Il gradiente e' scritto a mano invece di usare `jac="3-point"` di scipy per
    # una ragione precisa: scipy offre `finite_diff_rel_step`, che scala il passo
    # con |x|, e QUI E' SBAGLIATO.  Le coordinate del box sono LOGIT, e un logit
    # passa per lo zero: al modo del profilo q_b, mu = 2.53 sta a x = 0.024, e un
    # passo relativo di 1e-5 diventa 2.4e-7 assoluto — di nuovo sotto il rumore
    # della log-posterior.  Misurato: l'ottimizzatore si fermava lasciando 32
    # direzioni che miglioravano ancora.  (Il passo relativo sarebbe corretto in
    # scala LOG, dove le coordinate non passano per lo zero; nel box logit no.)
    #
    # Gli autori usano un passo ASSOLUTO, `delta = 1e-6` in `numgrad.m`, proprio
    # perche' le loro x trasformate sono O(1).  Adottiamo l'idea, non il numero:
    # loro fanno differenze IN AVANTI, noi CENTRATE (niente bias del prim'ordine,
    # al costo di 2m valutazioni per gradiente), e a passo troppo piccolo le
    # differenze centrate amplificano il rumore.  Misurato sul profilo q_b, con
    # ripartenze e sondaggio a +/-0.01 e +/-0.05 in scala log:
    #
    #     passo    logpost    direzioni che migliorano ancora    secondi
    #     1e-6    8466.544              55  (max +0.553 nat)        225
    #     1e-5    8473.779              23  (max +0.047 nat)        258
    #     1e-4    8473.890               1  (max +0.000 nat)         83
    #     1e-3    8473.892               0                           28   <--
    #
    # 1e-3 e' insieme il piu' preciso e il piu' veloce: sotto quel passo il
    # gradiente e' rumore e l'ottimizzatore consuma valutazioni per non muoversi.
    # Controprova di merito: il modo cosi' trovato (lambda 0.5355, mu 2.2697)
    # cade praticamente sulle mediane a posteriori gia' documentate nel README
    # per il Q-BVAR (0.5182, 2.2508), che sono state ottenute per una strada
    # completamente diversa.
    # PERCHE' UN CICLO DI RIPARTENZE E NON UNA CHIAMATA SOLA.  Con differenze
    # finite rumorose L-BFGS-B soddisfa i propri criteri (`ftol`/`gtol`) in un
    # punto dove il gradiente vero non e' ancora nullo, e si ferma.  Misurato
    # sul profilo `l`: una chiamata sola lasciava 50 direzioni su 39 coordinate
    # che MIGLIORAVANO ancora la log-posterior.  Ripartire dal punto raggiunto
    # ricostruisce l'approssimazione dell'Hessiana da zero e sblocca la discesa
    # — e' lo stesso rimedio che `csminwel` applica internamente quando
    # incontra un "cliff" (righe 115-160: riparte con Hcliff, poi con l'identita').
    def grad_nat(x):
        """Differenze centrate a passo ASSOLUTO — `numgrad.m`, ma centrato."""
        x = np.asarray(x, dtype=float)
        out = np.empty(x.size)
        e = np.zeros(x.size)
        for i in range(x.size):
            e[i] = FD_STEP
            out[i] = (neg_nat(x + e) - neg_nat(x - e)) / (2.0 * FD_STEP)
            e[i] = 0.0
        return out

    best_x = x0
    best_f = neg_nat(x0)
    for _ in range(max(1, n_restarts)):
        res = minimize(neg_nat, best_x, method="L-BFGS-B", jac=grad_nat,
                       options={"maxiter": maxiter,
                                "maxfun": 500 * (2 + target.spec.n),
                                "ftol": 1e-14, "gtol": 1e-10})
        if not np.isfinite(res.fun) or res.fun >= best_f - _MODE_TOL:
            if np.isfinite(res.fun) and res.fun < best_f:
                best_x, best_f = res.x, float(res.fun)
            break
        best_x, best_f = res.x, float(res.fun)

    g_hat = _to_g(best_x, lo, hi)
    # il modo DEVE essersi mosso e DEVE essere migliorato: e' il controllo che
    # smaschera l'early-stop silenzioso (il "modo" uguale a gamma0).
    if (not np.isfinite(target.log_posterior(g_hat))
            or target.log_posterior(g_hat) < target.log_posterior(g0)):
        g_hat = g0
    th_hat = np.log(g_hat)

    # ── Hessiana in scala LOG (vedi la nota "NIENTE JACOBIANO NEL MODO").
    # Differenze finite centrate sul log-posterior NEGATIVO, in theta = log gamma.
    def neg_log(th):
        v = target.log_posterior(np.exp(np.asarray(th, dtype=float)))
        return _PEN if not np.isfinite(v) else -v

    m = th_hat.size
    h = 1e-3 * np.maximum(1.0, np.abs(th_hat))
    f0 = neg_log(th_hat)
    H = np.zeros((m, m))
    for i in range(m):
        ei = np.zeros(m); ei[i] = h[i]
        # diagonale: lo schema a 4 punti degenera in quello a 3, con f0 al centro
        H[i, i] = (neg_log(th_hat + 2 * ei) - 2 * f0
                   + neg_log(th_hat - 2 * ei)) / (4.0 * h[i] * h[i])
        for j in range(i + 1, m):
            ej = np.zeros(m); ej[j] = h[j]
            fpp = neg_log(th_hat + ei + ej)
            fpm = neg_log(th_hat + ei - ej)
            fmp = neg_log(th_hat - ei + ej)
            fmm = neg_log(th_hat - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
    H = 0.5 * (H + H.T)

    # REGOLARIZZAZIONE DELL'HESSIANA — la ricetta e' degli autori.
    # `bvarGLPmf.m` righe 359-362 (blocco commentato, ma esplicito):
    #
    #     % regularizing the Hessian (making sure it is positive definite)
    #     [V,E]=eig(HH);  HH=V*abs(E)*V';
    #
    # cioe' si prende il VALORE ASSOLUTO degli autovalori invece di rinunciare.
    # Serve perche' un'Hessiana per differenze finite, valutata in un punto solo
    # QUASI stazionario, ha quasi sempre qualche autovalore leggermente negativo
    # o quasi nullo: sono le direzioni piatte, non un errore.
    #
    # PERCHE' NON BASTAVA IL RIPIEGO ISOTROPO.  Prima qui c'era `W = 0.01*I` in
    # caso di Hessiana non definita positiva, e sul profilo `l` scattava SEMPRE:
    # la proposta perdeva ogni informazione sulla forma del posterior, che e'
    # proprio cio' che l'Appendice B chiede di usare (`c*W`, con W l'Hessiana
    # inversa nel picco).  I psi coprono nove ordini di grandezza, quindi una
    # proposta isotropa e' senza speranza — ed e' il meccanismo che in passato
    # aveva gia' prodotto un'accettazione del 3.8%.
    ev, V = np.linalg.eigh(H)
    floor = 1e-10 * max(1.0, float(np.abs(ev).max()))
    ev_reg = np.maximum(np.abs(ev), floor)
    W = (V / ev_reg) @ V.T                        # = V diag(1/|e|) V'
    W = 0.5 * (W + W.T)
    if not np.all(np.isfinite(W)) or np.any(np.linalg.eigvalsh(W) <= 0):
        W = np.eye(m) * 0.01                      # rete, ora davvero eccezionale
    return np.exp(th_hat), W


# ─── 4. Il Metropolis (Appendice B) ───────────────────────────────────────────

@dataclass
class MetropolisState:
    """Lo stato della catena: quanto basta per riprendere da dove si era."""
    theta: np.ndarray           # iperparametri in scala LOG
    logpost: float
    W: np.ndarray               # covarianza di base della proposta (scala LOG)
    c: float                    # fattore di scala (tarato per accettazione ~20%)
    n_accept: int = 0
    n_prop: int = 0
    W_nat: np.ndarray | None = None   # la stessa, riportata in scala NATURALE

    def __post_init__(self) -> None:
        if self.W_nat is None:
            # Delta method: se theta = log(gamma), allora una covarianza W in
            # scala log corrisponde a  diag(gamma) W diag(gamma)  in scala
            # naturale.  Serve SOLO al test di equivalenza (log_scale=False):
            # senza questa conversione la proposta naturale userebbe passi
            # tarati per il log — enormi rispetto a psi ~ 1e-3 — e verrebbe
            # rifiutata sempre, rendendo il confronto privo di significato.
            # E' fissata al modo (non allo stato corrente) per non rompere la
            # simmetria della proposta, che renderebbe necessaria una
            # correzione di Hastings.
            g = np.exp(self.theta)
            self.W_nat = (g[:, None] * self.W) * g[None, :]

    @property
    def gamma(self) -> np.ndarray:
        return np.exp(self.theta)

    @property
    def acceptance(self) -> float:
        return self.n_accept / self.n_prop if self.n_prop else 0.0


def metropolis_step(target: HyperTarget, state: MetropolisState,
                    rng: np.random.Generator, *, log_scale: bool = True) -> MetropolisState:
    """
    Un passo di Metropolis — Appendice B, passi 2 e 3.

        gamma* ~ N(gamma^(j-1), c*W)
        alpha  = min(1, p(gamma*|y) / p(gamma^(j-1)|y))

    `log_scale=False` esegue la proposta sulla scala NATURALE, cioe' la lettura
    letterale dell'Appendice B.  Serve al test di equivalenza: le due strade
    devono dare lo stesso posterior (vedi la nota nell'header).
    """
    base = state.W if log_scale else state.W_nat
    L = np.linalg.cholesky(state.c * base)
    step = L @ rng.standard_normal(state.theta.size)

    if log_scale:
        cand = state.theta + step
        lp_cand = target.log_posterior_log_scale(cand)
    else:
        cand_g = np.exp(state.theta) + step
        if np.any(cand_g <= 0):
            state.n_prop += 1
            return state
        cand = np.log(cand_g)
        lp_cand = target.log_posterior(cand_g)

    lp_curr = (state.logpost if log_scale
               else state.logpost - float(np.sum(state.theta)))

    state.n_prop += 1
    if np.log(rng.random()) < lp_cand - lp_curr:
        state.theta = cand
        state.logpost = (lp_cand if log_scale
                         else lp_cand + float(np.sum(cand)))
        state.n_accept += 1
    return state


def init_metropolis(target: HyperTarget, rng: np.random.Generator,
                    *, c: float = 0.5, gamma0=None) -> MetropolisState:
    """Passo 1 dell'Appendice B: modo + Hessiana inversa."""
    gamma_mode, W = find_mode(target, gamma0)
    theta = np.log(gamma_mode)
    return MetropolisState(theta=theta,
                           logpost=target.log_posterior_log_scale(theta),
                           W=W, c=c)


__all__ = [
    "HyperPrior",
    "HyperTarget",
    "MetropolisState",
    "gamma_from_mode_sd",
    "build_target",
    "ar1_residual_var",
    "hyper_box",
    "find_mode",
    "metropolis_step",
    "init_metropolis",
    "LAM_MIN", "LAM_MAX", "MU_MIN", "MU_MAX",
    "PSI_BOX_FACTOR", "LAM_START", "MU_START", "FD_STEP",
]
