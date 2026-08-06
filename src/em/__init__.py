"""
src/em/ — Motore di stima EM per il Dynamic Factor Model Student-t a
frequenza mista.

IL MODELLO
----------
Un DFM a frequenza mista con innovazioni Student-t, stimato via EM sul dataset
'final' (M = 37 serie: 34 mensili + 3 trimestrali). La struttura di caricamento
e' scelta fra tre spec di config/factor_specs.json:

    diag3        r = 3 fattori, Lambda diagonale a blocchi (1 fattore per serie)
    diag4        r = 4 fattori, Lambda diagonale a blocchi
    fed_overlap  r = 4 fattori (G, S, R, L), NON diagonale: i prezzi caricano
                 su 1 fattore, le altre serie su 2 (globale + locale)

Le serie trimestrali sono osservate come aggregato many-to-many (MM) della loro
controparte mensile latente, con pesi {1/3, 2/3, 1, 2/3, 1/3}: lo stato dei
fattori impila 5 lag (dimensione 5r).

Cinque varianti, combinazione di tre flag ortogonali — code pesanti sui pesi
(gaussian), persistenza dell'idiosincratico (idio_ar1), e pesi idiosincratici
per-serie w^eps_{i,t} contro condivisi w^eps_t (per_series_weights):

    gaussian               nu = inf (pesi === 1)  | idio i.i.d.
    gaussian_ar1           nu = inf               | idio AR(1)
    student_t              code pesanti           | idio i.i.d.  | pesi condivisi
    student_t_ar1          code pesanti           | idio AR(1)   | pesi per-serie
    student_t_ar1_shared   code pesanti           | idio AR(1)   | pesi condivisi

(``student_t_ar1_shared`` e' il controllo che isola i pesi per-serie dall'AR(1):
identico a ``student_t_ar1`` tranne lo schema dei pesi. Delle 8 combinazioni dei
tre flag, queste 5 sono quelle istanziate in run_final_artifacts.VARIANTS.)

Parametri stimati:  A (VAR r x r), Q (innovazione r x r), Lambda (M x r), la
scala idiosincratica (R, oppure la coppia sigma^2 e rho sotto l'Asse B), e i
gradi di liberta' nu_u (fattori) e nu_eps (idiosincratico).

L'ALGORITMO E IL GRAFO DEI MODULI
---------------------------------
    em_initialization   theta^(0): standardizzazione, MM-fill delle trimestrali
                        + gaussian-fill del bordo frastagliato, PCA mask-driven
                        e primo M-step.

              (src/kalman.py, fuori dal pacchetto: filtro + smoother a
               frequenza mista che forniscono i momenti smoothed dei fattori)

    em_e_step  (E-step) residui di Mahalanobis d^u (fattori) e d^eps (idio) ->
                        pesi Student-t w^u, w^eps dal posteriore Gamma
                        coniugato; ciclo interno ECM che scioglie
                        l'accoppiamento pesi <-> momenti smoothed.

    em_m_step  (M-step) statistiche sufficienti pesate -> update in forma
                        chiusa di (Lambda, R) e (A, Q) in ordine sequenziale
                        ECM, e nu_u/nu_eps via root-finding di Brent.

    em_main   (loop)    itera E/M con arresto sull'ELBO, poi fissa un
                        rappresentante canonico della classe di equivalenza dei
                        fattori (normalize_signs, apply_convention_1). fit_dfm
                        e' l'entry point completo del First Stage.

Due assi trasversali, richiamati da tutti i moduli:

    factor_structure    la LOADING MASK M x r di 0/1 (build_loading_mask):
                        l'unico punto del codice che conosce i nomi dei fattori.
    idio_ar1            Asse B: idiosincratico AR(1) portato DENTRO lo stato
                        (stato aumentato 5r + n_e, con n_e = M_m + 5*M_q = 49).

selftest_fixture fornisce la fixture unica ai self-test __main__ dei moduli;
test_idio_ar1 e' il gate algebrico dell'Asse B.
"""
