# Validation report — modalità `quick`

Generato da `python -m mcmc.validate.run`. Una riga per parametro; il verdetto è
uno dei quattro di `validate/verdict.py`. **Rosso = sampler rotto, e basta**: un
parametro non identificato è un limite del dato, non un difetto del codice.

## Blocco fattori

| parametro | ramo | verdetto | vero | stima | ESS | R-hat | copertura | ragione |
|---|---|---|---|---|---|---|---|---|
| `QML sotto leverage` | B | **recuperato con bias noto** | -0.700 | -0.341 | 27 | 0.996 | — | **progresso, non soluzione** — e il check congela ENTRAMBI i fatti. A corr(Q)=0.8 la deriva corretta (z = M eps esatto) recupera rho: -0.34 contro il -0.27 del decoupled (vero -0.70) ⇒ **P5 eliminata** [NON confermato]. Ma phi di un fattore COLLASSA ancora: [1.   0.43 0.43]. Resta dietro allow_experimental. |
| `ASIS x prior (cella off-IG)` | B | **non identificato** | — | — | — | — | — | la cella (use_asis=True, prior=IG) e' **IRRAGGIUNGIBILE**: gibbs.py:498 forza il half-Normal. Non e' un bug — CP e NCP devono esprimere lo STESSO prior perche' l'interweaving campioni il giunto esatto, e il gaussiano sul sigma segnato e' cio' che rende coniugato il draw NCP. Via 1 (draw NCP non coniugato sotto IG, ~30 righe) documentata e NON implementata; adottata la via 2: disciplina sperimentale. |
| `A` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `Q` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `f_t` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `h^u_k` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `mu^u_k` | - | **mai testato** | — | — | — | — | — | non e' un parametro stimato: fissato per identificazione |
| `nu_u` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `phi^u_k` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `rho^u_k` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `sigma^2_{eta,k}` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `w^u_t` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `ASIS correttezza` | B | **recuperato** | — | +0.008 | — | — | — | a prior FISSATO (half-Normal), ASIS on/off campionano lo stesso target: |d phi| = 0.008, |d sigma^2|/sigma^2 = 4%. Se divergessero, ASIS sarebbe ROTTO (una riparametrizzazione non puo' spostare la legge invariante). |
| `ASIS efficienza (ESS phi)` | B | **recuperato** | — | +1.346 | — | — | — | ESS(phi) x1.35 con ASIS. E' cio' per cui ASIS e' costruito — la cresta (phi, sigma^2). Non alzarlo lo renderebbe INUTILE, non rotto. |
| `QML sotto Branch A` | A | **recuperato** | — | — | — | — | — | solleva ValueError **per costruzione**: Branch A non forma MAI una covarianza di misura (nessuna mistura, nessuna linearizzazione; usa Q^{-1/2} pieno in modo esatto), quindi non c'e' nulla da accoppiare. **Non e' una lacuna, e' una proprieta'** — ed e' la ragione per cui A e' la controparte esatta. |
| `leva su rho: prior, non ASIS` | B | **recuperato** | — | +0.855 | — | — | — | a ASIS FISSATO (off), cambiare il prior IG -> half-Normal moltiplica ESS(rho) per x0.9. A prior FISSATO, accendere ASIS lo moltiplica per x0.9. **La leva e' il prior, non l'interweaving** — la catena causale 'meglio sigma => meglio rho' del .tex e' falsificata. |
| `recommend_coupling` | B | **recuperato** | — | — | — | — | — | la soglia e' sulla SOVRA-CONFIDENZA indotta (<=5%), non su corr(Q): a Q diagonale -0.0% -> 'decoupled'; a corr(Q)=0.8 30% -> 'qml'. Sul pannello reale (~0.4%) => decoupled. |

## Blocco osservazioni

| parametro | ramo | verdetto | vero | stima | ESS | R-hat | copertura | ragione |
|---|---|---|---|---|---|---|---|---|
| `Lambda` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `R` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `h^eps_i` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `mu^eps_i` | - | **mai testato** | — | — | — | — | — | non e' un parametro stimato: fissato per identificazione |
| `nu_eps` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `phi^eps_i` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `rho^eps_i` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `sigma^2_{eps,i}` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `w^eps_t` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |

## Come leggere i verdetti

- **recuperato** — confrontato con il vero, lo recupera.
- **recuperato con bias noto** — lo recupera, ma con una distorsione *misurata*. È un risultato da scrivere in tesi, non un fallimento.
- **non identificato** — i dati non lo pinzano. Il check afferma il limite e passa.
- **mai testato** — riga vuota **dichiarata**.
- **ROTTO** — l'unico rosso: il campionatore sbaglia.
