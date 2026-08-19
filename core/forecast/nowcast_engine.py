"""
core/forecast/nowcast_engine.py

Il MOTORE: una cella (spec x variante), una data, un trimestre target -> un
nowcast del PIL.

Nessuna orchestrazione qui dentro: il ciclo settimanale sta in
`weekly_nowcast.py`.  Questo modulo sa fare una cosa sola, e la fa bene.

LE TRE SPEC E LE CINQUE VARIANTI
--------------------------------
La spec (`fed_overlap` | `diag4` | `diag3`) e' la struttura di caricamento, cioe'
quali fattori ciascuna serie vede; viene da `config/factor_specs.json` tramite
`em.factor_structure.build_loading_mask`.  La variante combina tre flag
ortogonali (`gaussian`, `idio_ar1`, `per_series_weights`) nei cinque modelli
stimati: `gaussian`, `gaussian_ar1`, `student_t`, `student_t_ar1`,
`student_t_ar1_shared`.

DUE MODI DI PRODURRE UN NOWCAST
-------------------------------
`estimate(...)`     stima i parametri sul vintage (EM completo).  Costoso.
`filter_only(...)`  tiene i parametri fissi e ricalcola solo lo stato.  Molto
                    piu' rapido, ed e' cio' che serve nelle settimane in cui non
                    si ri-stima: i parametri si muovono lentamente, lo stato no.
                    Non e' un filtro gaussiano travestito — sotto Student-t
                    ripassa dal ciclo ECM, quindi i pesi w_t sono ricalcolati
                    sull'informazione nuova, con theta congelato.

L'ESTRAZIONE DEL PIL: LA RIGA DI LAMBDA, NON UN INDICE
------------------------------------------------------
Il valore modellato della serie target al mese `t` e'

    yhat_t = check_Lambda[target, :] @ stato_t

dove `check_Lambda` e' *la stessa* matrice di osservazione che il filtro di
Kalman ha usato — restituita da `run_e_step` come `Lambda_tilde`.  Non si
ricostruisce niente a mano, e questo risolve da solo due cose che una
ricostruzione manuale sbaglia:

  * **Fattori multipli.**  Sotto `fed_overlap` il PIL carica su DUE fattori
    (G e R).  La riga di Lambda li contiene entrambi, con i loro pesi; prendere
    "il fattore del blocco del PIL" ne butterebbe uno, senza errore visibile.
  * **Idiosincratico con memoria.**  Sotto le varianti `*_ar1` l'idiosincratico
    e' uno stato, la sua media condizionata non e' zero, e per una trimestrale
    e' aggregato con gli stessi pesi MM della parte comune.  `check_Lambda`
    include quel blocco (`[Lambda_tilde | C_idio]`), quindi la componente entra
    da sola.  Ometterla distorcerebbe tre varianti su cinque.

L'aggregazione di Mariano-Murasawa {1/3, 2/3, 1, 2/3, 1/3} e' anch'essa gia'
dentro `check_Lambda` per le righe trimestrali: il nowcast del PIL e' il valore
TRIMESTRALE, letto al mese di fine trimestre.

  * **Una parametrizzazione sola.**  `check_Lambda` e lo stato che moltiplica
    devono venire dallo STESSO dizionario, perche' `fit_dfm` ne restituisce due
    versioni osservazionalmente equivalenti ma non mescolabili (canonica in
    `fit["f_smooth"]`, raw in `fit["e_step_output"]`).  Si usa la raw, che e'
    l'unica completa sotto `*_ar1`.  Il dettaglio sta in `extract_target`.

LA SCALA
--------
Standardizzazione e de-standardizzazione usano media e deviazione dello STESSO
vintage (`scale.standardize_asof` sul pannello gia' mascherato), mai del
campione pieno.  La catena intera e le sue inverse stanno in `scale.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.forecast import scale
from core.forecast.panel_builder import build_panel
from core.forecast.release_calendar import quarter_end

from dfm.em_initialization import (
    mm_fill_quarterly,
    gaussian_fill_ragged,
    pca_initialization,
    compute_theta_initial,
)
from dfm.em_e_step import run_e_step
from dfm.em_main import fit_dfm
from dfm.factor_structure import build_loading_mask

# ─── Le cinque varianti: combinazioni dei flag ortogonali ─────────────────────
# Stessa semantica di `run_final_artifacts.VARIANTS`, ridotta a cio' che il
# nowcast deve sapere.  `inner_criterion` compare solo dove serve: con i pesi
# per-serie il criterio `max` e' preso su T*M variazioni invece che su T, quindi
# la stessa tolleranza diventa una richiesta piu' severa per pura dimensione.
VARIANTS: dict[str, dict] = {
    "gaussian":             {"gaussian": True,  "idio_ar1": False, "per_series_weights": False},
    "gaussian_ar1":         {"gaussian": True,  "idio_ar1": True,  "per_series_weights": False},
    "student_t":            {"gaussian": False, "idio_ar1": False, "per_series_weights": False},
    "student_t_ar1":        {"gaussian": False, "idio_ar1": True,  "per_series_weights": True,
                             "inner_criterion": "rms"},
    "student_t_ar1_shared": {"gaussian": False, "idio_ar1": True,  "per_series_weights": False},
}

SPECS: tuple[str, ...] = ("fed_overlap", "diag4", "diag3")


# ─── Il vintage: pannello, scala, struttura, punto d'innesco ──────────────────

def prepare_vintage(as_of_date, target_quarter: str, spec: str, variant: str,
                    theta_warm: dict | None = None, seed: int = 42,
                    panel: pd.DataFrame | None = None,
                    metadata: pd.DataFrame | None = None,
                    exact: pd.DataFrame | None = None) -> dict:
    """
    Tutto cio' che serve per stimare o filtrare a una data, e nient'altro.

    `theta_warm`, se dato, sostituisce l'inizializzazione PCA come punto di
    partenza dell'EM.  Il warm start dal vintage PRECEDENTE non e' look-ahead:
    e' solo un innesco migliore per lo stesso ottimo, e non contiene alcuna
    informazione futura.  (Cosa diversa sarebbe partire dal theta di campione
    pieno, che quell'informazione la contiene: non si fa.)
    """
    if spec not in SPECS:
        raise ValueError(f"spec {spec!r} ignota; attesa una di {SPECS}.")
    if variant not in VARIANTS:
        raise ValueError(f"variante {variant!r} ignota; attesa una di {tuple(VARIANTS)}.")
    cfg = VARIANTS[variant]

    panel_asof = build_panel(as_of_date, target_quarter, panel=panel,
                             metadata=metadata, exact=exact)

    # Serie con UNA SOLA osservazione nel vintage: il fattore la interpola
    # esattamente, il residuo idiosincratico va a zero e `update_R` sbatte sulla
    # guardia di degenerazione (r_i = 0) facendo abortire l'EM.  Una sola
    # osservazione non porta informazione stimabile: la si AZZERA (NaN),
    # riportando la colonna al caso "zero osservazioni" che il Kalman gia'
    # gestisce via W_t.  Non si RIMUOVE la colonna — `build_loading_mask`
    # pretende la bijezione serie<->assegnazioni, e togliere la colonna la
    # romperebbe; azzerarla lascia intatti cols, mask e freq_list.  Le colonne
    # gia' a zero oss restano tali (comportamento invariato).  Il TARGET
    # (ultima colonna) non si tocca mai.  Esempio: PPIFIS a inizio 2010.
    target_col = panel_asof.columns[-1]
    n_obs = panel_asof.notna().sum()
    too_thin = [c for c in panel_asof.columns
                if c != target_col and int(n_obs[c]) == 1]
    if too_thin:
        panel_asof = panel_asof.copy()
        panel_asof[too_thin] = np.nan

    cols = list(panel_asof.columns)

    meta = metadata if metadata is not None else None
    freq_list = _freq_list_for(cols, meta)
    structure = build_loading_mask(spec, cols)

    # Scala del vintage: il pannello e' gia' mascherato, quindi media e
    # deviazione sono espandenti per costruzione.
    Y_std_df, mean, std = scale.standardize_asof(panel_asof)

    if theta_warm is not None:
        theta0 = dict(theta_warm)
    else:
        Y_mm = Y_std_df.copy()
        for c, fr in zip(cols, freq_list):
            if fr == "quarterly":
                Y_mm[c] = mm_fill_quarterly(Y_std_df[c])
        Y_filled = gaussian_fill_ragged(Y_mm, random_state=seed)
        F0, _info = pca_initialization(Y_filled, structure)
        theta0 = compute_theta_initial(Y_filled, F0, structure,
                                       idio_ar1=cfg["idio_ar1"],
                                       freq_list=freq_list)

    if cfg.get("per_series_weights"):
        theta0["per_series_weights"] = True
    if cfg.get("inner_criterion"):
        theta0["inner_criterion"] = cfg["inner_criterion"]

    return {
        "as_of": pd.Timestamp(as_of_date),
        "target_quarter": target_quarter,
        "spec": spec,
        "variant": variant,
        "panel": panel_asof,
        "Y_std_df": Y_std_df,
        "mean": mean,
        "std": std,
        "cols": cols,
        "freq_list": freq_list,
        "structure": structure,
        "theta0": theta0,
        "gaussian": cfg["gaussian"],
    }


def _freq_list_for(cols: list[str], metadata: pd.DataFrame | None) -> list[str]:
    """'monthly'/'quarterly' per colonna, dai metadati del pannello."""
    from core.forecast.release_calendar import load_metadata
    meta = load_metadata() if metadata is None else metadata
    freq_of = dict(zip(meta["series_id"], meta["freq"]))
    missing = [c for c in cols if c not in freq_of]
    if missing:
        raise KeyError(f"serie senza frequenza nei metadati: {missing}")
    return ["monthly" if freq_of[c] == "M" else "quarterly" for c in cols]


# ─── I due modi di produrre lo stato ──────────────────────────────────────────

def estimate(vintage: dict, max_iter: int = 250, verbose: bool = False) -> dict:
    """EM completo sul vintage.  Restituisce l'output di `fit_dfm`."""
    return fit_dfm(
        Y=vintage["Y_std_df"].to_numpy(),
        theta_init=vintage["theta0"],
        freq_list=vintage["freq_list"],
        block_map=vintage["structure"],
        ordered_cols=vintage["cols"],
        gaussian=vintage["gaussian"],
        use_full_elbo=True,
        max_iter=max_iter,
        verbose=verbose,
        save_path=None,
    )


def filter_only(vintage: dict, theta: dict, verbose: bool = False) -> dict:
    """
    Solo lo stato, parametri congelati: E-step su `theta` senza M-step.

    Sotto Student-t il ciclo ECM interno gira comunque, quindi i pesi w_t
    reagiscono all'informazione nuova della settimana; e' solo theta a non
    muoversi.
    """
    return run_e_step(
        Y=vintage["Y_std_df"].to_numpy(),
        theta=theta,
        freq_list=vintage["freq_list"],
        gaussian=vintage["gaussian"],
        verbose=verbose,
    )


# ─── L'estrazione del target ──────────────────────────────────────────────────

def extract_target(vintage: dict, state: np.ndarray, Lambda_full: np.ndarray,
                   P_smooth: np.ndarray | None = None,
                   theta: dict | None = None,
                   target_series: str | None = None) -> dict:
    """
    Il valore modellato della serie target al mese di fine trimestre.

    `Lambda_full` e' la matrice di osservazione usata dal filtro
    (`run_e_step(...)["Lambda_tilde"]` o `fit_dfm(...)["e_step_output"]`
    equivalente): include gia' i pesi MM sulle righe trimestrali e, sotto
    `idio_ar1`, il blocco idiosincratico.  La riga del target moltiplica lo stato
    intero: nessun indice di fattore da indovinare.

    `state` E `Lambda_full` DEVONO VENIRE DALLA STESSA PARAMETRIZZAZIONE
    ------------------------------------------------------------------
    `fit_dfm` restituisce DUE mondi, entrambi legittimi ma non mescolabili:

      * CANONICO — `fit["theta"]`, `fit["f_smooth"]`, `fit["P_smooth"]`: dopo
        ortogonalizzazione, normalizzazione dei segni e Convenzione 1
        (`em_main.py`, sez. 2c/3/4);
      * RAW — tutto cio' che sta in `fit["e_step_output"]`, che la docstring di
        `fit_dfm` dichiara esplicitamente *pre-post-processing*.

    I due sono osservazionalmente equivalenti: la Convenzione 1 riscala
    `f_can = f_raw / c` e `Lambda_can = Lambda_raw * c`, quindi il prodotto
    `Lambda @ f` e' lo stesso — ma SOLO se i due fattori del prodotto vengono
    dallo stesso mondo.  Mescolarli restituisce `z_vero / c`, cioe' un nowcast
    silenziosamente compresso (fino a ~2.6x su `diag3`), e per giunta solo nelle
    settimane in cui si ri-stima: le settimane a theta congelato passano da
    `filter_only`, che e' coerente per costruzione.  Il risultato e' una
    seghettatura mensile che sembra segnale e non lo e'.

    Qui si usa la coppia RAW, perche' e' l'unica disponibile per intero: sotto
    le varianti `*_ar1` la riga di `Lambda_tilde` ha in coda il blocco
    idiosincratico, che `kalman.build_Lambda_tilde` non ricostruisce (rende
    (M, 5r), senza `C_idio`).  `P_smooth` viene dallo stesso dizionario, quindi
    `z` e `sd_z` sono sulla stessa scala.
    """
    cols = vintage["cols"]
    target = target_series or cols[-1]
    if target not in cols:
        raise KeyError(f"serie target {target!r} assente dal pannello.")
    i = cols.index(target)

    target_qe = quarter_end(vintage["target_quarter"])
    idx = vintage["Y_std_df"].index
    if target_qe not in idx:
        raise ValueError(
            f"il mese di fine trimestre {target_qe.date()} non e' nel pannello "
            f"(ultimo mese: {idx[-1].date()}).  Il pannello va esteso fino al "
            f"target: passa `target_quarter` a build_panel."
        )
    t = idx.get_loc(target_qe)

    row = np.asarray(Lambda_full, dtype=float)[i, :]
    z = float(row @ np.asarray(state, dtype=float)[t, :])

    # Incertezza dello stato proiettata sul target.  Sotto le varianti senza
    # idio_ar1 l'idiosincratico NON e' nello stato, quindi la sua varianza va
    # aggiunta a mano; sotto *_ar1 e' gia' dentro P_smooth.
    sd = float("nan")
    if P_smooth is not None:
        var = float(row @ np.asarray(P_smooth, dtype=float)[t] @ row)
        if theta is not None and not _has_idio_state(theta):
            R = np.asarray(theta["R"], dtype=float).ravel()
            var += float(R[i])
        sd = float(np.sqrt(max(var, 0.0)))

    out = scale.unscale(z, float(vintage["mean"][target]), float(vintage["std"][target]))
    out["sd_z"] = sd
    out["target_series"] = target
    return out


def _has_idio_state(theta) -> bool:
    """True se l'idiosincratico e' uno stato AR(1) invece che rumore bianco."""
    try:
        return ("rho" in theta) and (theta["rho"] is not None)
    except TypeError:
        return False


# ─── L'entrata unica ──────────────────────────────────────────────────────────

def nowcast(as_of_date, target_quarter: str, spec: str, variant: str,
            theta: dict | None = None, theta_warm: dict | None = None,
            max_iter: int = 250, verbose: bool = False,
            target_series: str | None = None,
            panel: pd.DataFrame | None = None,
            metadata: pd.DataFrame | None = None,
            exact: pd.DataFrame | None = None) -> dict:
    """
    Nowcast del PIL per `target_quarter` con l'informazione nota a `as_of_date`.

    Parameters
    ----------
    theta : dict, optional
        Se dato, i parametri sono CONGELATI a questo valore e si ricalcola solo
        lo stato (settimana senza ri-stima).  Se None, si stima l'EM.
    theta_warm : dict, optional
        Punto di partenza dell'EM quando si stima (il theta del vintage
        precedente).  Ignorato se `theta` e' dato.

    Returns
    -------
    dict
        `nowcast_bea`     tasso di crescita BEA, %  (l'unita' di lettura)
        `nowcast_livello` log-differenza annualizzata (l'unita' del pannello)
        `nowcast_z`       standardizzata (l'unita' del modello)
        `sd_z`            deviazione dello stato proiettata sul target, in z
        `theta`           i parametri usati (da riciclare la settimana dopo)
        + metadati della cella e della stima
    """
    v = prepare_vintage(as_of_date, target_quarter, spec, variant,
                        theta_warm=theta_warm, panel=panel,
                        metadata=metadata, exact=exact)

    if theta is None:
        fit = estimate(v, max_iter=max_iter, verbose=verbose)
        theta_used = fit["theta"]
        eso = fit["e_step_output"]
        n_iter, converged, reestimated = int(fit["n_iter"]), bool(fit["converged"]), True
    else:
        theta_used = theta
        eso = filter_only(v, theta_used, verbose=verbose)
        n_iter, converged, reestimated = 0, True, False

    # STATO, LAMBDA E P DALLO STESSO DIZIONARIO, SEMPRE.  E' l'unico invariante
    # che tiene in piedi la scala del nowcast: `fit["f_smooth"]` e' CANONICO
    # mentre `fit["e_step_output"]` e' RAW, e mescolare i due mondi comprime il
    # nowcast del fattore della Convenzione 1 — silenziosamente, e solo nelle
    # settimane con ri-stima.  `filter_only` restituisce lo stesso dizionario
    # nella stessa parametrizzazione, quindi le due branch sono omogenee e una
    # settimana EM e una congelata sono confrontabili.  Vedi `extract_target`.
    res = extract_target(v, np.asarray(eso["f_smooth"], dtype=float),
                         eso["Lambda_tilde"],
                         P_smooth=eso.get("P_smooth"), theta=theta_used,
                         target_series=target_series)

    nu_eps = _maybe_float(theta_used, "nu_eps")
    return {
        "as_of": str(pd.Timestamp(as_of_date).date()),
        "target_quarter": target_quarter,
        "spec": spec,
        "variant": variant,
        "target_series": res["target_series"],
        "nowcast_bea": res["bea"],
        "nowcast_livello": res["level"],
        "nowcast_z": res["z"],
        "sd_z": res["sd_z"],
        "n_iter": n_iter,
        "converged": converged,
        "reestimated": reestimated,
        "nu_eps_hat": (None if v["gaussian"] else nu_eps),
        "mean_train": float(v["mean"][res["target_series"]]),
        "std_train": float(v["std"][res["target_series"]]),
        "n_obs_panel": int(v["panel"].notna().sum().sum()),
        "theta": theta_used,
    }


def _maybe_float(theta, key: str):
    try:
        val = theta[key]
    except (KeyError, IndexError, TypeError):
        return None
    return None if val is None else float(np.asarray(val).ravel()[0])


__all__ = ["VARIANTS", "SPECS", "prepare_vintage", "estimate", "filter_only",
           "extract_target", "nowcast"]


# ─── Verifica ─────────────────────────────────────────────────────────────────
# Esegui:  python -m core.forecast.nowcast_engine
# Una stima vera per cella: lento (minuti).  Gira due celle, non quindici.

if __name__ == "__main__":
    def _hr(t: str) -> None:
        print("\n" + "=" * 78)
        print(t)
        print("=" * 78)

    from core.forecast.release_calendar import load_metadata, load_panel

    _hr("nowcast_engine.py — una cella, una data")

    full, meta = load_panel(), load_metadata()
    AS_OF, TARGET = "2008-11-14", "2008Q4"

    realised_level = float(full.loc[quarter_end(TARGET), "GDPC1"])
    realised_bea = float(scale.to_bea(realised_level))

    for spec, variant in (("diag3", "gaussian"), ("fed_overlap", "gaussian")):
        _hr(f"{spec} / {variant}   as_of={AS_OF}  target={TARGET}")
        r = nowcast(AS_OF, TARGET, spec, variant, panel=full, metadata=meta,
                    max_iter=60, verbose=False)
        print(f"  convergenza    : {r['converged']} (n_iter={r['n_iter']})")
        print(f"  scala vintage  : mean={r['mean_train']:+.4f}  std={r['std_train']:.4f}")
        print(f"  nowcast        : z={r['nowcast_z']:+.4f}  "
              f"livello={r['nowcast_livello']:+.4f} (log ann.)  "
              f"BEA={r['nowcast_bea']:+.4f}%")
        print(f"  realizzato     : livello={realised_level:+.4f}  BEA={realised_bea:+.4f}%")
        print(f"  errore (BEA)   : {r['nowcast_bea'] - realised_bea:+.4f} punti")

        # Il riciclo dei parametri: stessa cella, settimana dopo, theta congelato.
        r2 = nowcast("2008-11-21", TARGET, spec, variant, theta=r["theta"],
                     panel=full, metadata=meta)
        print(f"  settimana dopo (theta congelato): BEA={r2['nowcast_bea']:+.4f}%  "
              f"(mosso di {r2['nowcast_bea'] - r['nowcast_bea']:+.4f} coi dati nuovi)")

    _hr("Fatto.")
