"""
core/config_utils.py

Parsing degli argomenti e costruzione dei percorsi, in un posto solo.

UN MONDO SOLO (`final`)
-----------------------
Il pannello e' unico (37 serie NY Fed,
`data/processed/final/dataset_final.csv`) e le celle sono indicizzate da due
assi ortogonali:

    spec     struttura di caricamento : fed_overlap | diag4 | diag3
    variante flag (gaussian, idio_ar1, per_series_weights) :
                 gaussian | gaussian_ar1 | student_t | student_t_ar1
                 | student_t_ar1_shared

    artefatti -> data/processed/final/<spec>/<variante>/
                 output/final/<spec>/<variante>/

API: `parse_spec_variant_args()`, `resolve_final_path()`, `SPECS`, `VARIANTS`.

**Il binario legacy non c'e' piu'.**  Il mondo vecchio era `--config <nome>`:
un nome sceglieva un pannello intero (`config/series_<nome>.json` ->
`dataset_<nome>.csv`), letto da `src/data_loader.py`.  Cancellati entrambi:
`config/series_{small,big}.json`, `data_loader.py`, e da qui
`resolve_output_path()` (percorsi `output/<config>/` e
`data/processed/<config>/`) assieme alle scorciatoie `--small`/`--big` e a
`parse_config_args`/`get_project_root`.

L'UNICO chiamante rimasto era `mcmc/gibbs.py:load_warm_init`, che ora e' rotto
di proposito: `src/mcmc/` va migrato al mondo `final` (`series_final.json` +
`resolve_final_path`) sul pattern di `em/selftest_fixture.py`.  Nient'altro nel
progetto dipendeva dal binario legacy: `core/forecast/`, `core/dfm/`, i motori
Monte Carlo e i self-test di `kalman`/`simulate_dfm` sono gia' tutti su
`parse_spec_variant_args`/`resolve_final_path`.
"""

from __future__ import annotations

import argparse
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# ─── Mondo nuovo: gli assi delle celle ────────────────────────────────────────
# Unica definizione: `run_final_artifacts.VARIANTS` porta i flag, qui stanno i
# soli nomi, per non duplicare la semantica in due posti.
SPECS: tuple[str, ...] = ("fed_overlap", "diag4", "diag3")
VARIANTS: tuple[str, ...] = ("gaussian", "gaussian_ar1", "student_t",
                             "student_t_ar1", "student_t_ar1_shared")
DEFAULT_SPEC = "fed_overlap"
DEFAULT_VARIANT = "gaussian"


# ═══ Mondo nuovo ══════════════════════════════════════════════════════════════

def parse_spec_variant_args(
    description: str = "",
    extra=None,
    *,
    with_variant: bool = True,
) -> argparse.Namespace:
    """
    Parser per il dataset `final`: ``--spec`` e (opzionale) ``--variant``.

    Parameters
    ----------
    description : str
        Descrizione passata ad ArgumentParser.
    extra : callable or None
        Callable ``extra(parser)`` per aggiungere flag specifici del modulo,
        invocato prima del parsing.
    with_variant : bool
        Se False espone il solo ``--spec`` (per i moduli che non dipendono
        dalla variante, es. l'inizializzazione).

    Returns
    -------
    argparse.Namespace
        Con ``spec`` e, se richiesto, ``variant``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--spec", choices=SPECS, default=DEFAULT_SPEC,
        help=f"struttura di caricamento da config/factor_specs.json "
             f"(default: {DEFAULT_SPEC})",
    )
    if with_variant:
        parser.add_argument(
            "--variant", choices=VARIANTS, default=DEFAULT_VARIANT,
            help=f"combinazione dei flag (gaussian, idio_ar1, "
                 f"per_series_weights) (default: {DEFAULT_VARIANT})",
        )
    if extra is not None:
        extra(parser)
    return parser.parse_args()


def resolve_final_path(
    kind: str,
    filename: str,
    spec: str,
    variant: str | None = None,
    *,
    mkdir: bool = True,
) -> pathlib.Path:
    """
    Percorso di una cella del dataset `final`, con la directory gia' creata.

    Parameters
    ----------
    kind : {"figures", "processed", "dataset"}
        "figures"   -> output/final/<spec>/<variant>/<filename>
        "processed" -> data/processed/final/<spec>/<variant>/<filename>
        "dataset"   -> data/processed/final/dataset_final.csv (filename e
                       variant ignorati: il pannello e' UNO, comune a tutte
                       le celle — e' la stima a variare, non i dati)
    filename : str
        Nome del file con estensione.  Ignorato per kind=="dataset".
    spec : str
        Una di `SPECS`.
    variant : str or None
        Una di `VARIANTS`.  Obbligatoria per "figures" e "processed": senza,
        due celle diverse scriverebbero sullo stesso file.
    mkdir : bool
        Se True crea la directory genitore.  Metti False per costruire un
        percorso da *leggere* senza creare cartelle vuote come effetto
        collaterale.

    Raises
    ------
    ValueError
        Se `kind` e' ignoto, se `spec`/`variant` non sono nelle liste, o se
        manca `variant` dove serve.
    """
    if spec not in SPECS:
        raise ValueError(f"spec {spec!r} ignota. Attesa una di {SPECS}.")

    if kind == "dataset":
        p = _PROJECT_ROOT / "data" / "processed" / "final" / "dataset_final.csv"
        if mkdir:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    if kind not in ("figures", "processed"):
        raise ValueError(
            f"kind {kind!r} ignoto. Atteso uno di "
            f"'figures', 'processed', 'dataset'."
        )
    if variant is None:
        raise ValueError(
            f"kind={kind!r} richiede `variant`: senza, celle diverse "
            f"scriverebbero sullo stesso percorso."
        )
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} ignota. Attesa una di {VARIANTS}.")

    if kind == "figures":
        p = _PROJECT_ROOT / "output" / "final" / spec / variant / filename
    else:
        p = _PROJECT_ROOT / "data" / "processed" / "final" / spec / variant / filename
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p
