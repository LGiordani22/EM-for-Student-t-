# `src/mcmc/tests/` — i test dei motori

Un solo comando:

```bash
python -m mcmc.tests.run_all
```

I test rispondono a una domanda ingegneristica: **il codice fa quello che dice la
matematica?** Girano in minuti. La domanda scientifica — *il modello recupera i veri
parametri?* — vive in `mcmc/validate/` (ore).

## Due livelli, due posti

I test qui distinguono per **livello di ciò che accertano**:

- **conditional** — la formula del full conditional è quella giusta (confronto con l'EM);
- **recovery** — a partire da un pannello simulato da `θ` noti, si ritrova la verità;
- **strutturale** — quali parametri vengono estratti in ciascuna cella del modello;
- **mixing** — l'efficienza del campionatore (ESS, invarianza).

Per il **segno** e la **correttezza** di `rho` (Famiglia C) il posto è qui; per la sua
**magnitudine** (rapporto `rho_hat / rho_vero`, copertura del CI) il posto è
`mcmc/validate/checks/leverage.py`.

## Chi fa cosa, e quali parametri testa

| File | Cosa fa | Parametri / percorsi testati | Livello |
|---|---|---|---|
| `test_shared` | i kernel dei conditional vs EM | `w^u,w^ε`; `A,Q`; `Λ,R`; `ν_u,ν_ε`; `(φ,σ²)`; `a_j` (Huang-Wand) | conditional |
| `test_linear` | il campionatore **senza SV** = MCMC dell'EM | `f`; `A,Q,Λ,R`; `ν_ε` | recovery |
| `test_vol_base` | volatilità stocastica base (KSC), correttezza + recupero | `h^u,(φ,σ²)^u` [comune]; `h^ε,φ^ε` [idio] | recovery |
| `test_leverage` | Famiglia C (`rho`): Branch A + Branch B, Omori, kernel griddy/RW/Laplace, skewness, DGP per-fattore, end-to-end | `ρ^u`, `ρ^ε` (segno); `h^u` sotto leverage | recovery (segno) |
| `test_variants` | le celle D1×D2 come restrizioni del sampler | presenza/assenza di `w`, `h^u`, `h^ε`, `ρ^u`, `ρ^ε`, `ν`, `a_j` | strutturale |
| `test_asis` | l'interweaving ASIS | `(φ,σ²)` [Famiglia B] | mixing |
| `test_diagnostics` | le funzioni diagnostiche | nessun parametro del modello: `corr(Q)`, ESS, R-hat, no-RNG | — |
| `test_coupling_qml` | il passo accoppiato del blocco comune (QML) | `h^u,(φ,σ²)^u` sotto coupling | recovery |

## Copertura dei parametri del modello

Il modello ha due blocchi di shock, entrambi con l'insieme completo:

- **fattori** (`u_t`): `A`, `Q`, e per canale `k` una log-vol AR(1) `(φ^u_k, σ²_k)`, un
  leverage `ρ^u_k`, un path `h^u_k`; più i pesi `w^u` e i gradi di libertà `ν_u`;
- **osservazioni** (`ε_t`): `Λ`, `R`, e per serie `i` una SV `(φ^ε_i, σ²_{ε,i}, h^ε_i)`
  col suo leverage `ρ^ε_i`; più `w^ε` e `ν_ε`.

Ogni parametro/percorso di entrambi i lati è testato almeno una volta (tabella sopra).
`μ` è fissato a 0 per identificazione (verificato `μ==0` in ogni draw dove la SV è attiva);
`Σ_0` (prior su `f_0`) è tenuto fisso e non viene estratto.

## Note

- `test_coupling_qml` ha un gate lento dietro `--slow`.
- Il recupero del **valore** dei parametri (magnitudine, CI, ESS/R-hat su repliche) è in
  `mcmc/validate/` — vedi `mcmc/validate/run.py`.
