# Piano di pulizia / consolidamento `src/mcmc/`

> **Quando.** La pulizia va fatta **DOPO la validazione del modello completo**,
> non prima. Questo file è separato da `IMPLEMENTATION_PLAN.md` apposta:
> *implementazione* e *pulizia* sono cose diverse.

---

## Principio chiave (leggere prima di tutto)

C'è **una sola** attività in questo piano che NON è pulizia:

- **La Parte A — il DRIVER DI RECOVERY UNICO — è parte della VALIDAZIONE del
  modello completo, non cosmesi.** È lo strumento con cui si valida il
  **master cell SV + leverage accesi insieme (D1-b × D2-b)**, la cella che
  **oggi nessun recovery testa** (ogni `run_recovery_*` accende un solo asse).
  Va fatta **ORA, insieme alla validazione**, perché *è* la validazione. Il
  driver non ha senso "a freddo": si scrive contro il modello che si sta
  validando. Farlo dopo significherebbe scriverlo due volte.

- **La Parte B — `_testutil`, archiviare i gate `test_passoN`, allineare i
  docstring "Passo N" — è COSMETICA.** Zero impatto sul comportamento. Va fatta
  **DOPO**, in un **commit di pulizia isolato** e facilmente revertibile, quando
  il package si è stabilizzato. Non blocca la scienza.

Non mescolare i due blocchi nello stesso commit: il primo cambia *cosa
testiamo*, il secondo solo *come è organizzato il codice*.

---

## PARTE A — Consolidamento utile SUBITO (= validazione del modello completo)

> Questo NON è pulizia. È la prossima tappa scientifica. Farlo ORA con la
> validazione, non al commit cosmetico.

### A1. Driver di recovery parametrico unico

**Obiettivo.** Sostituire le quattro funzioni "una per asse" di
`diagnostics.py` (`run_recovery_mcmc`, `run_recovery_mcmc_sv`,
`run_recovery_mcmc_leverage`, `compare_branches_AB`) con **un solo driver** che
copre l'intera griglia di celle, incluso il caso oggi mancante:
**SV + leverage insieme** (il master cell).

**Firma proposta** (in `diagnostics.py`, nessun file nuovo):

```
run_recovery(*, sv: bool, leverage: bool, timing: str = "contemporaneous",
             T=400, n_chains=4, n_iter, burn_in, seed=0,
             theta_true=None, config_name="small") -> dict
```

- `theta_true=None` → modalità **dati veri** (vedi A3): niente coverage, ma
  R-hat/ESS, cross-check vs EM, stabilità.
- `theta_true` fornito → modalità **recovery sintetica**: coverage degli
  intervalli credibili + correlazione dei path latenti + R-hat/ESS.
- DGP: `simulate_sv.simulate_dfm_sv` genera già SV **e** leverage con la
  struttura mixed-freq / ragged corretta → il master cell combinato è
  generabile **senza codice nuovo di simulazione**.

**Cosa assorbe / manda in pensione:**

| Vecchio | Destino |
|---|---|
| `run_recovery_mcmc` (no-SV + cross-check EM) | caso `sv=False, leverage=False` |
| `run_recovery_mcmc_sv` | caso `sv=True, leverage=False` |
| `run_recovery_mcmc_leverage` | casi `leverage=True`, `timing ∈ {contemporaneous, lagged}` |
| `compare_branches_AB` | helper sottile sopra due chiamate `timing=...` |
| **(nuovo, mancante oggi)** | **`sv=True, leverage=True` — il master cell** |

**Suite sopra il driver** (la vera tappa di validazione):

```
run_recovery_suite(cells=[...], ...) -> tabella unica
```

gira la griglia di celle e produce **un'unica tabella** R-hat / ESS / coverage /
relerr-vs-EM (riusando `diagnostics_table` e una versione unificata dei
`_print_recovery*`). È questo l'output che certifica il modello completo.

**File toccati:** solo `diagnostics.py`.
**Rischi:** medi — è logica di test, ma le quattro funzioni hanno
asserzioni/soglie leggermente diverse; vanno riconciliate con cura (soglie
coverage, allineamento di segno per fattore via `monte_carlo_recovery`,
gestione `nu`). Da fare con il modello sotto mano, non a freddo.
**Accettazione:** la suite passa su tutte le celle a T pieno multi-chain, **e in
particolare il master cell combinato recupera** parametri + path con R-hat sani;
le celle che riproducono i vecchi `run_recovery_*` danno gli stessi verdetti di
prima (non-regressione).

### A2. Caso "master cell" come gate end-to-end veloce

Aggiungere **un** caso veloce (T piccolo, 1 chain, ~1-2 min)
`sv=True, leverage=True` come smoke test — colma il buco lasciato dai
`test_passoN` (ognuno accende un solo asse). Vive già dentro il driver A1 (è solo
una configurazione), quindi **non** è codice separato.

### A3. Entry per dati veri (tappa successiva, ma stesso driver)

La modalità `theta_true=None` di A1 è il punto d'aggancio per la validazione sui
dati veri (no coverage; R-hat/ESS, cross-check EM su `fit_dfm_result.npz`,
stabilità tra vintage). **Non** aggiungere ora un ramo separato: progettare A1
già con questo branch dall'inizio, così non si riscrive.

**Perché A1-A3 ora:** sono *la stessa attività* della validazione del modello
completo. Il driver va scritto contro il modello che si sta validando.

---

## PARTE B — Cosmetico / igiene → DOPO (commit di pulizia dedicato)

> Zero impatto sul comportamento. Da fare quando il package è stabile, in un
> commit isolato e facilmente revertibile. Nessuno di questi blocca la scienza.

### B1. `mcmc/_testutil.py` — deduplica gli helper di test

Estrarre i helper ripetuti in un solo modulo:

- `_check(name, ok, detail)` — oggi **5 copie** (`test_shared` + 4 `test_passoN`).
- correlazioni: `_abscorr` (test_passo1, diagnostics), `_corr` (test_passo2),
  `_signedcorr` (diagnostics), `_relerr`.
- unificare `_print_recovery` / `_print_recovery_sv` / `_print_recovery_lev` in
  **una** funzione parametrica.

**File toccati:** nuovo `_testutil.py`; import aggiornati in `diagnostics.py` e
nei test sopravvissuti.
**Rischio:** basso. **Accettazione:** test verdi identici a prima.
**Sequenza:** farlo **dopo A1**, perché A1 riscrive proprio i `_print_recovery*`
— altrimenti li si tocca due volte.

### B2. Archiviare i gate `test_passoN.py`

Una volta che A1/A2 coprono gli stessi casi (e meglio), i quattro
`test_passo1..4.py` sono scaffolding storico superato.

- **Opzione consigliata:** spostarli in `src/mcmc/_archive_gates/` (traccia dei
  singoli passi senza che girino nella suite).
- Alternativa: cancellarli, dato che i loro casi end-to-end vivono nel driver.
- **`test_shared.py` NON si archivia:** è una *regression guard* permanente
  (invariante "shared == EM"), categoria validazione-utile, resta dov'è.

**Rischio:** basso. **Ordine:** A1 → B2 (archiviare prima che A1 copra quei casi
sarebbe prematuro).

### B3. Allineare i docstring "Passo N" allo stato reale

Diversi banner dicono ancora "**Passo 1: no SV, no leverage**" mentre il codice è
già al master cell completo (SV + Branch A + Branch B cablati in `gibbs.py`). Da
correggere header/docstring in: `gibbs.py`, `sample_params.py`,
`sample_states.py`, `shared.py`, `__init__.py` (e i riferimenti "Passo 2-4" in
`constants.py` / `sample_vol.py` dove ormai sono implementati).
**Rischio:** nullo (solo commenti). **Accettazione:** lettura coerente; nessun
banner promette uno stato più arretrato del codice.

### B4. (minori, opzionali) micro-deduplica

- `_inv_sqrt_spd` (sample_vol) e `_sqrt_spd` (simulate_sv): coppia diretta /
  inversa SPD → eventualmente in `shared`. Marginale.
- Nessun'altra duplicazione pesante: `sample_leverage_lagged` già riusa
  `draw_rho_*` e `_inv_sqrt_spd`; `sample_params` riusa l'EM. L'architettura è
  pulita — non inventare refactor.

---

## Ordine di esecuzione consigliato

1. **ORA, con la validazione:** A1 → A2 → (predisporre A3).
2. **DOPO, commit di pulizia isolato:** B1 → B2 → B3 (→ B4 se ne hai voglia).
3. Il commit iniziale del package (`src/mcmc/` è interamente untracked, +
   `IMPLEMENTATION_PLAN.md`) va fatto **prima** di tutto questo, così A e B sono
   diff leggibili sopra una baseline committata — altrimenti il primo commit
   ingloberebbe sia il codice sia il refactor e diventa illeggibile.

## Cosa NON fare

- Non archiviare i `test_passoN` prima che A1 ne copra i casi.
- Non toccare `shared.py` / `sample_*` "per pulizia": sono foglia e validati; il
  rischio non vale il guadagno cosmetico.
- Non fondere A (validazione) e B (cosmesi) nello stesso commit.
