# Validation report — modalità `full`

Generato da `python -m mcmc.validate.run`. Una riga per parametro; il verdetto è
uno dei quattro di `validate/verdict.py`. **Rosso = sampler rotto, e basta**: un
parametro non identificato è un limite del dato, non un difetto del codice.

## Blocco fattori

| parametro | ramo | verdetto | vero | stima | ESS | R-hat | copertura | ragione |
|---|---|---|---|---|---|---|---|---|
| `Q` | B | **recuperato con bias noto** | +0.248 | +0.112 | 15 | 1.230 | — | err. rel. 49% contro il vero, **sotto SV + leverage** (prima era validato solo contro l'EM e senza SV) |
| `Q` | A | **recuperato con bias noto** | +0.248 | +0.055 | 10 | 1.912 | — | err. rel. 38% contro il vero, **sotto SV + leverage** (prima era validato solo contro l'EM e senza SV) |
| `h^u_k` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `mu^u_k` | - | **mai testato** | — | — | — | — | — | non e' un parametro stimato: fissato per identificazione |
| `phi^u_k` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `rho^u_k` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `sigma^2_{eta,k}` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `w^u_t` | B | **mai testato** | — | — | — | — | — | i pesi non sono immagazzinati nei draws (sono interni allo sweep): l'algebra del loro conditional e' verificata in test_shared, ma il path non e' confrontabile col vero senza esporli |
| `w^u_t` | A | **mai testato** | — | — | — | — | — | i pesi non sono immagazzinati nei draws (sono interni allo sweep): l'algebra del loro conditional e' verificata in test_shared, ma il path non e' confrontabile col vero senza esporli |
| `A` | B | **recuperato** | +0.108 | +0.140 | 510 | 1.001 | — | err. rel. 17% contro il vero, **sotto SV + leverage** (prima era validato solo contro l'EM e senza SV) |
| `A` | A | **recuperato** | +0.108 | +0.139 | 232 | 1.067 | — | err. rel. 23% contro il vero, **sotto SV + leverage** (prima era validato solo contro l'EM e senza SV) |
| `f_t` | B | **recuperato** | — | +0.991 | — | — | — | corr. media col vero 0.99 (un path si giudica sulla traccia, non sul valore) |
| `f_t` | A | **recuperato** | — | +0.992 | — | — | — | corr. media col vero 0.99 (un path si giudica sulla traccia, non sul valore) |
| `nu_u` | B | **recuperato** | +4.080 | +4.386 | 138 | 1.011 | — | CI copre il vero, rapporto 1.07 |
| `nu_u` | A | **recuperato** | +4.080 | +4.438 | 134 | 1.020 | — | CI copre il vero, rapporto 1.09 |

## Blocco osservazioni

| parametro | ramo | verdetto | vero | stima | ESS | R-hat | copertura | ragione |
|---|---|---|---|---|---|---|---|---|
| `Lambda` | A | **recuperato con bias noto** | +0.938 | +1.920 | 7 | 2.320 | — | err. rel. 67% contro il vero, **sotto SV + leverage** (buco chiuso) |
| `h^eps_i` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `mu^eps_i` | - | **mai testato** | — | — | — | — | — | non e' un parametro stimato: fissato per identificazione |
| `phi^eps_i` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `rho^eps_i` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `sigma^2_{eps,i}` | - | **mai testato** | — | — | — | — | — | nessun check lo copre |
| `w^eps_t` | B | **mai testato** | — | — | — | — | — | i pesi non sono immagazzinati nei draws (sono interni allo sweep): l'algebra del loro conditional e' verificata in test_shared, ma il path non e' confrontabile col vero senza esporli |
| `w^eps_t` | A | **mai testato** | — | — | — | — | — | i pesi non sono immagazzinati nei draws (sono interni allo sweep): l'algebra del loro conditional e' verificata in test_shared, ma il path non e' confrontabile col vero senza esporli |
| `Lambda` | B | **recuperato** | +0.938 | +1.374 | 6 | 2.004 | — | err. rel. 30% contro il vero, **sotto SV + leverage** (buco chiuso) |
| `R` | B | **recuperato** | +0.015 | +0.020 | 109 | 1.025 | — | err. rel. 14% contro il vero, **sotto SV + leverage** (buco chiuso) |
| `R` | A | **recuperato** | +0.015 | +0.020 | 52 | 1.155 | — | err. rel. 11% contro il vero, **sotto SV + leverage** (buco chiuso) |
| `nu_eps` | B | **recuperato** | +4.400 | +4.822 | 454 | 1.009 | — | CI copre il vero, rapporto 1.10 |
| `nu_eps` | A | **recuperato** | +4.400 | +4.803 | 445 | 1.000 | — | CI copre il vero, rapporto 1.09 |

## Come leggere i verdetti

- **recuperato** — confrontato con il vero, lo recupera.
- **recuperato con bias noto** — lo recupera, ma con una distorsione *misurata*. È un risultato da scrivere in tesi, non un fallimento.
- **non identificato** — i dati non lo pinzano. Il check afferma il limite e passa.
- **mai testato** — riga vuota **dichiarata**.
- **ROTTO** — l'unico rosso: il campionatore sbaglia.
