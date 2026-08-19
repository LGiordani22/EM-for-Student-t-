"""
core/run_final_artifacts.py

Esegue l'EM sul dataset FINAL per una (o tutte) le tre strutture di fattore di
`config/factor_specs.json` e produce l'INSIEME COMPLETO di artefatti per la
spec, in un solo posto e parametrici sulla spec.

PERCHE' UN RUNNER DEDICATO
--------------------------
Senza di esso le figure e i JSON nascerebbero sparsi dentro i `__main__` dei
singoli moduli (`em_initialization` -> pca_factors.png, `kalman` -> le sue,
`em_main` -> la convergenza), ognuno con i suoi percorsi: per il quadro completo
di una spec servirebbe lanciare piu' moduli in un ordine non scritto da nessuna
parte, e `fit_dfm_result.json` non lo scriverebbe nessuno. Qui l'insieme e' UNO,
esplicito e riproducibile.

I `__main__` dei moduli EM restano quello che devono essere: self-test con
assert (`python -m em.em_m_step --spec ...`), non generatori di output.

LE CELLE: 3 SPEC x 5 VARIANTI
-----------------------------
I flag `gaussian`, `idio_ar1` e `per_series_weights` danno cinque varianti per
ogni struttura di fattore, e ognuna ha la sua cartella:

    gaussian          shock normali,      idiosincratico i.i.d.
    gaussian_ar1      shock normali,      idiosincratico AR(1)
    student_t         shock a code pesanti, idiosincratico i.i.d.
    student_t_ar1     shock a code pesanti, idiosincratico AR(1), pesi PER SERIE
    student_t_ar1_shared  come sopra ma con UN peso condiviso su tutte le serie
                          (isola l'effetto dei pesi per-serie da quello dell'AR(1))

COSA PRODUCE
------------
  data/processed/final/<spec>/<variante>/
      theta_initial.npz              theta^(0) (PCA mask-driven, + seed rho)
      theta_initial_metadata.json    T, M, r, serie per fattore, autovalori
      fit_dfm_result.npz             theta stimato + fattori smoothed + pesi
      fit_dfm_result.json            versione leggibile: Lambda per serie, R,
                                     A, Q, nu, rho, sign_flips, scale, ELBO
  output/final/<spec>/<variante>/
      em_loglik_convergence.png      ELBO per iterazione + Delta (monotonia)
      pca_factors.png                fattori iniziali F^(0), bande NBER
      factors_smoothed.png           fattori finali smoothed, bande NBER
      mm_fill_verification.png       MM fill vs osservazioni trimestrali
      lambda_heatmap.png             Lambda: mappa serie x fattore (mask visibile)
      weights_student_t.png          w_u / w_eps nel tempo (varianti student_t)
      idio_ar1_rho.png               rho stimati per serie (varianti *_ar1)

USO
---
    python core/run_final_artifacts.py --spec fed_overlap --variant student_t_ar1
    python core/run_final_artifacts.py --all --all-variants   # tutte le 15 celle
    python core/run_final_artifacts.py --all                  # solo gaussian
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "core")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dfm.em_initialization import (  # noqa: E402
    standardize, mm_fill_quarterly, gaussian_fill_ragged,
    pca_initialization, compute_theta_initial,
)
from dfm.factor_structure import build_loading_mask  # noqa: E402
from dfm.em_main import fit_dfm  # noqa: E402

SPECS = ("fed_overlap", "diag4", "diag3")

# Le 5 varianti come combinazione dei flag ortogonali.
VARIANTS: dict[str, dict] = {
    "gaussian":       {"gaussian": True,  "idio_ar1": False, "per_series_weights": False},
    "gaussian_ar1":   {"gaussian": True,  "idio_ar1": True,  "per_series_weights": False},
    "student_t":      {"gaussian": False, "idio_ar1": False, "per_series_weights": False},
    # Pesi idiosincratici PER SERIE (w^eps_{i,t}), come nel .tex: con la
    # persistenza modellata serie per serie, un abbattimento delle code globale
    # sarebbe internamente incoerente.  Misurato sul pannello: il peso condiviso
    # declassa l'INTERO cross-section per anomalie di una sola serie (2021-03,
    # w=0.17, ma mediana per-serie 1.03 — era lo shock agli assegni di stimolo
    # su DSPIC96), mentre uno shock davvero comune (2020-04) resta declassato da
    # entrambi.  Costa ~16% di iterazioni interne in piu'.
    # Nota: il flag ha effetto SOLO qui.  Sotto `gaussian` i pesi sono ≡1 (quindi
    # per-serie e condiviso coincidono) e senza `idio_ar1` non e' applicabile.
    # `inner_criterion="rms"` SOLO qui, e solo qui deve restare.
    #
    # LA RAGIONE E' `per_series_weights`, NON LA VELOCITA'. Con i pesi per serie
    # il criterio `max` e' preso su T*M variazioni invece che su T: il massimo di
    # un insieme piu' grande e' sistematicamente piu' grande, quindi la STESSA
    # tolleranza diventa una richiesta piu' severa per pura dimensione. `max` qui
    # non e' confrontabile con `max` sulle celle a pesi condivisi; `rms` de-
    # distorce. Il guadagno di tempo (581 s -> 157 s a tolleranza esterna
    # invariata, loglik NON peggiore) e' una conseguenza, non la motivazione.
    #
    # Perche' le altre tre restano su `max`: le due gaussiane saltano del tutto
    # il ciclo interno (nu->inf => pesi ≡ 1, criterio irrilevante); `student_t`
    # ha pesi CONDIVISI, quindi il max e' su T e la distorsione dimensionale non
    # si presenta — e col solo warm start sta gia' a ~10 iterazioni interne
    # (69 s -> 35 s, theta invariato a 1.1e-06). Restano percio' confrontabili
    # con tutti i risultati precedenti.
    #
    # Caveat da dichiarare se il criterio viene discusso: poiche' il rapporto
    # max/RMS si stabilizza (~36.7 misurato), il cambio equivale a RISCALARE la
    # tolleranza interna per la dimensione — le quattro celle non condividono
    # quindi la stessa tolleranza interna effettiva. E' l'intento, non un
    # effetto collaterale. L'alternativa omogenea sarebbe `rms` ovunque con
    # `tol_inner` riscalato, ma andrebbe riverificato che le altre tre celle non
    # si muovano.
    "student_t_ar1":  {"gaussian": False, "idio_ar1": True,  "per_series_weights": True,
                       "inner_criterion": "rms"},
    # ─── Il controllo che isola i pesi per-serie ──────────────────────────────
    # Identica a `student_t_ar1` TRANNE lo schema dei pesi: un solo w^eps_t
    # condiviso da tutte le M serie, invece di w^eps_{i,t} per serie.
    #
    # Perche' esiste. Fra le quattro varianti iniziali, `per_series_weights` e
    # `idio_ar1` erano CONFUSI: l'unica cella con pesi per-serie era anche
    # l'unica con idiosincratico AR(1). Le recovery mostrano che `nu_eps` non e'
    # recuperato proprio li' (errore 48-85% contro l'1.5% delle celle a peso
    # condiviso, e in calo lentissimo con T), ma col disegno a quattro celle non
    # si poteva dire QUALE delle due caratteristiche lo causi. Questa cella
    # varia una cosa sola e chiude la domanda.
    #
    # Non e' solo diagnostica: e' un modello candidato. I due schemi hanno
    # difetti OPPOSTI e misurati. Il peso condiviso declassa l'intero
    # cross-section per l'anomalia di una serie sola (2021-03, vedi sopra); il
    # peso per-serie costa l'identificazione dell'indice di coda, perche' ogni
    # peso e' informato da UNA osservazione (corr(w_eps stimati, veri) = 0.29
    # contro 0.89, e non migliora con T). Tenendole entrambe il trade-off si
    # mostra sui dati invece di deciderlo a priori — e non e' detto che il
    # modello internamente piu' coerente sia quello che nowcasta meglio.
    #
    # `inner_criterion` resta il DEFAULT `max`, deliberatamente: la distorsione
    # dimensionale che giustifica `rms` su `student_t_ar1` e' causata dai pesi
    # per-serie (massimo su T*M invece che su T) e qui non si presenta. Tenere
    # `max` la rende confrontabile con `student_t`, che e' il confronto che
    # conta per isolare l'effetto AR(1).
    "student_t_ar1_shared": {"gaussian": False, "idio_ar1": True,
                             "per_series_weights": False},
}

# Recessioni NBER usate come bande grigie nei grafici dei fattori.
NBER = [("1990-07", "1991-03"), ("2001-03", "2001-11"),
        ("2007-12", "2009-06"), ("2020-02", "2020-04")]


# ─── Percorsi ─────────────────────────────────────────────────────────────────

def _proc_dir(spec: str, variant: str) -> str:
    p = os.path.join(_ROOT, "data", "processed", "final", spec, variant)
    os.makedirs(p, exist_ok=True)
    return p


def _fig_dir(spec: str, variant: str) -> str:
    p = os.path.join(_ROOT, "output", "final", spec, variant)
    os.makedirs(p, exist_ok=True)
    return p


def _jsonable(x):
    """
    Rende un valore serializzabile in JSON STANDARD.

    `json.dump` scrive `Infinity` per float('inf'): sintassi accettata da
    Python ma NON dal JSON standard, quindi illeggibile da qualunque altro
    parser. Sotto l'estimatore gaussiano nu_u e nu_eps sono esattamente inf
    (e' la definizione del limite gaussiano), quindi il caso capita sempre.
    Si scrive `null`, che significa la stessa cosa — "non e' un parametro
    stimato qui" — ed e' JSON valido.
    """
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


# ─── Preparazione ─────────────────────────────────────────────────────────────

def prepare(spec: str, seed: int = 42, idio_ar1: bool = False):
    """Pannello 'final' + struttura `spec` + theta^(0). Deterministico dato seed."""
    df = pd.read_csv(os.path.join(_ROOT, "data", "processed", "final",
                                  "dataset_final.csv"),
                     index_col=0, parse_dates=True)
    with open(os.path.join(_ROOT, "config", "series_final.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    freq_of = {s["series_id"]: ("monthly" if s["freq"] == "M" else "quarterly")
               for s in cfg["series"]}
    cols = list(df.columns)
    freq_list = [freq_of[c] for c in cols]
    fs = build_loading_mask(spec, cols)

    Y_std_df, mean, std = standardize(df)
    Y_mm = Y_std_df.copy()
    for c, fr in zip(cols, freq_list):
        if fr == "quarterly":
            Y_mm[c] = mm_fill_quarterly(Y_std_df[c])
    Y_filled = gaussian_fill_ragged(Y_mm, random_state=seed)

    F0, _info = pca_initialization(Y_filled, fs)
    theta0 = compute_theta_initial(Y_filled, F0, fs, idio_ar1=idio_ar1,
                                   freq_list=freq_list)
    return dict(df=df, Y_std_df=Y_std_df, Y_filled=Y_filled, mean=mean, std=std,
                cols=cols, freq_list=freq_list, fs=fs, F0=np.asarray(F0),
                theta0=theta0)


# ─── Salvataggi ───────────────────────────────────────────────────────────────

def save_theta_initial(prep: dict, spec: str, variant: str) -> None:
    d = _proc_dir(spec, variant)
    th, fs = prep["theta0"], prep["fs"]
    np.savez_compressed(
        os.path.join(d, "theta_initial.npz"),
        Lambda=th["Lambda"], A=th["A"], Q=th["Q"], R=th["R"],
        w_u=th["w_u"], w_eps=th["w_eps"], Sigma_0=th["Sigma_0"],
        F=prep["F0"], nu_u=np.array(th["nu_u"]), nu_eps=np.array(th["nu_eps"]),
    )
    A = np.asarray(th["A"])
    meta = {
        "spec": spec,
        "variant": variant,
        "factor_names": fs.factor_names,
        "diagonal": bool(fs.diagonal),
        "T": int(prep["Y_std_df"].shape[0]),
        "M": int(prep["Y_std_df"].shape[1]),
        "r": int(fs.r),
        "sample_start": str(prep["df"].index[0].date()),
        "sample_end": str(prep["df"].index[-1].date()),
        "series_per_factor": {fs.factor_names[j]: int(fs.mask[:, j].sum())
                              for j in range(fs.r)},
        "series_mean": {k: float(v) for k, v in prep["mean"].items()},
        "series_std": {k: float(v) for k, v in prep["std"].items()},
        "Lambda_sv": [float(v) for v in
                      np.linalg.svd(np.asarray(th["Lambda"]), compute_uv=False)],
        "A_eigenvalue_moduli": sorted(
            (float(abs(v)) for v in np.linalg.eigvals(A)), reverse=True),
        "Q_eigenvalues": [float(v) for v in np.linalg.eigvalsh(np.asarray(th["Q"]))],
    }
    with open(os.path.join(d, "theta_initial_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def load_final_fit(spec: str, variant: str) -> dict:
    """
    Rilegge `fit_dfm_result.{npz,json}` di una cella. Restituisce
    ``{"theta": dict, "meta": dict}``.

    **Non usare `em_main.load_dfm_fit` su questi file.** Sono due formati
    diversi sotto lo stesso nome: `load_dfm_fit` legge l'archivio che scrive
    `fit_dfm(save_path=...)`, con le chiavi prefissate `theta__`/`estep__` e
    tutte le diagnostiche; `save_fit` qui sopra scrive un archivio **piatto e
    ridotto** (gli array servono alle figure, non a riprendere l'EM).
    Passandogli un file di questa forma, `load_dfm_fit` non trova nessuna
    chiave `theta__` e restituirebbe **theta = {} in silenzio** — oggi si
    salva solo perche' inciampa prima su una chiave mancante. Questa funzione
    esiste per non dipendere da quell'inciampo.

    Nota su `sigma2`: l'archivio non lo contiene perche' sotto AR(1) il M-step
    pone `R = sigma2` (em_m_step riga ~1879), quindi `R` **e'** la varianza
    dell'innovazione fresca. Chi simula legge `sigma2` se c'e', `R` altrimenti:
    in entrambi i casi ottiene la stessa quantita'.
    """
    d = _proc_dir(spec, variant)
    npz_path = os.path.join(d, "fit_dfm_result.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Cella {spec}/{variant} non stimata: {npz_path} non esiste.\n"
            f"Calcolala con:  python core/run_final_artifacts.py "
            f"--spec {spec} --variant {variant}"
        )
    a = np.load(npz_path)
    theta = {k: a[k] for k in ("Lambda", "A", "Q", "R", "Sigma_0")}
    theta["nu_u"] = float(a["nu_u"])
    theta["nu_eps"] = float(a["nu_eps"])
    if "rho" in a.files:
        theta["rho"] = np.asarray(a["rho"], float)
        theta["sigma2"] = np.asarray(a["R"], float)   # R == sigma2, vedi sopra

    json_path = os.path.join(d, "fit_dfm_result.json")
    meta: dict = {}
    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    if meta.get("per_series_weights"):
        theta["per_series_weights"] = True
    return {"theta": theta, "meta": meta}


def save_fit(out: dict, prep: dict, spec: str, variant: str) -> None:
    """fit_dfm_result.{npz,json} — gli array e la versione leggibile."""
    d = _proc_dir(spec, variant)
    suffix = ""
    th, fs, cols = out["theta"], prep["fs"], prep["cols"]
    est = out["e_step_output"]

    np.savez_compressed(
        os.path.join(d, f"fit_dfm_result{suffix}.npz"),
        Lambda=th["Lambda"], A=th["A"], Q=th["Q"], R=th["R"],
        Sigma_0=th["Sigma_0"], nu_u=np.array(th["nu_u"]),
        nu_eps=np.array(th["nu_eps"]),
        f_smooth=out["f_smooth"], P_smooth=out["P_smooth"],
        w_u=est["w_u"], w_eps=est["w_eps"],
        loglik_history=np.asarray(out["loglik_history"], dtype=float),
        **({"rho": np.asarray(th["rho"], float)} if "rho" in th else {}),
    )

    Lam = np.asarray(th["Lambda"])
    R = np.asarray(th["R"])
    ll = np.asarray(out["loglik_history"], dtype=float)
    dll = np.diff(ll)
    f = np.asarray(out["f_smooth"])[:, :fs.r]

    payload = {
        "spec": spec,
        "variant": variant,
        "gaussian": bool(VARIANTS[variant]["gaussian"]),
        "idio_ar1": bool(VARIANTS[variant]["idio_ar1"]),
        "per_series_weights": bool(VARIANTS[variant].get("per_series_weights", False)),
        "factor_names": fs.factor_names,
        "diagonal": bool(fs.diagonal),
        "T": int(out["T"]), "M": int(out["M"]), "r": int(out["r"]),
        "n_iter": int(out["n_iter"]),
        "converged": bool(out["converged"]),
        "loglik_first": float(ll[0]), "loglik_last": float(ll[-1]),
        "elbo_monotone": bool((dll >= -1e-6).all()),
        "elbo_min_delta": _jsonable(float(dll.min())) if dll.size else None,
        "monotonicity_violations": list(out["monotonicity_violations"]),
        # null (non Infinity) sotto l'estimatore gaussiano: vedi _jsonable
        "nu_u": _jsonable(float(th["nu_u"])),
        "nu_eps": _jsonable(float(th["nu_eps"])),
        "rho_A": float(np.max(np.abs(np.linalg.eigvals(np.asarray(th["A"]))))),
        "var_f_smooth": [float(v) for v in f.var(axis=0)],
        "sign_flips": {k: int(v) for k, v in out["sign_flips"].items()},
        "scale_factors": {k: float(v) for k, v in out["scale_factors"].items()}
        if isinstance(out["scale_factors"], dict)
        else [float(v) for v in np.asarray(out["scale_factors"]).ravel()],
        "leak_coef": {k: float(v) for k, v in (out.get("leak_coef") or {}).items()},
        "A": np.asarray(th["A"]).tolist(),
        "Q": np.asarray(th["Q"]).tolist(),
        # Lambda per serie, con i nomi dei fattori: la mask e' leggibile a occhio
        # (gli zeri esatti sono le entrate fuori-mask).
        "Lambda": {c: {fs.factor_names[j]: float(Lam[i, j]) for j in range(fs.r)}
                   for i, c in enumerate(cols)},
        "R": {c: float(R[i]) for i, c in enumerate(cols)},
    }
    if "rho" in th:
        _rho = np.asarray(th["rho"], float).ravel()
        payload["rho"] = {c: float(_rho[i]) for i, c in enumerate(cols)}
        payload["rho_summary"] = {
            "mean": float(_rho.mean()), "median": float(np.median(_rho)),
            "min": float(_rho.min()), "max": float(_rho.max()),
            "n_above_0.5": int((_rho > 0.5).sum()),
        }
    with open(os.path.join(d, f"fit_dfm_result{suffix}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ─── Figure ───────────────────────────────────────────────────────────────────

def _shade(ax, index):
    for a, b in NBER:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b >= index[0] and a <= index[-1]:
            ax.axvspan(a, b, alpha=0.15, color="grey", zorder=0)


def make_figures(out: dict, prep: dict, spec: str, variant: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib non disponibile — nessuna figura.")
        return []

    fs, cols = prep["fs"], prep["cols"]
    idx = prep["Y_std_df"].index
    fdir = _fig_dir(spec, variant)
    written: list[str] = []

    def _save(fig, name):
        p = os.path.join(fdir, name)
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    # 1. Convergenza ELBO + Delta (la "monotonia")
    ll = np.asarray(out["loglik_history"], dtype=float)
    dll = np.diff(ll)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=False,
                                 gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(np.arange(1, ll.size + 1), ll, lw=1.3, color="steelblue", marker="o", ms=2.5)
    a1.set_ylabel("log-verosimiglianza (ELBO)")
    a1.set_title(f"Convergenza EM — spec '{spec}', variante '{variant}'  "
                 f"(n_iter={out['n_iter']}, converged={out['converged']})")
    a1.grid(alpha=0.3)
    colors = ["seagreen" if d >= 0 else "crimson" for d in dll]
    a2.bar(np.arange(2, ll.size + 1), dll, color=colors, width=0.8)
    a2.axhline(0, lw=0.8, color="black")
    a2.set_yscale("symlog", linthresh=1e-6)
    a2.set_xlabel("iterazione EM")
    a2.set_ylabel(r"$\Delta$ ELBO (symlog)")
    n_neg = int((dll < -1e-6).sum())
    a2.set_title(f"Incrementi: {'tutti >= 0 (monotona)' if n_neg == 0 else f'{n_neg} VIOLAZIONI'}"
                 f"   min Delta = {dll.min():.2e}", fontsize=10)
    a2.grid(alpha=0.3, axis="y")
    _save(fig, "em_loglik_convergence.png")

    # 2. Fattori iniziali (PCA) e 3. fattori finali (smoothed)
    for arr, name, title in (
        (prep["F0"], "pca_factors.png",
         f"Fattori iniziali $F^{{(0)}}$ (PCA mask-driven) — spec '{spec}'"),
        (np.asarray(out["f_smooth"])[:, :fs.r], "factors_smoothed.png",
         f"Fattori smoothed finali — spec '{spec}', variante '{variant}'"),
    ):
        fig, axes = plt.subplots(fs.r, 1, figsize=(12, 2.2 * fs.r), sharex=True)
        axes = np.atleast_1d(axes)
        for j, ax in enumerate(axes):
            ax.plot(idx, arr[:, j], lw=1.0, color=f"C{j}")
            _shade(ax, idx)
            ax.axhline(0, lw=0.5, color="black", ls="--")
            ax.set_ylabel(f"{fs.factor_names[j]}\n({int(fs.mask[:, j].sum())} serie)",
                          fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].set_title(title + "   (bande grigie = recessioni NBER)", fontsize=11)
        _save(fig, name)

    # 4. Verifica del MM fill su ogni serie trimestrale
    qcols = [c for c, f_ in zip(cols, prep["freq_list"]) if f_ == "quarterly"]
    if qcols:
        fig, axes = plt.subplots(len(qcols), 1, figsize=(12, 2.6 * len(qcols)),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, qcols):
            raw = prep["Y_std_df"][c]
            filled = mm_fill_quarterly(raw)
            ax.plot(idx, filled, lw=1.1, color="steelblue", label=r"MM-filled $\xi_m$")
            obs = raw.dropna()
            ax.scatter(obs.index, obs.values, s=14, color="crimson", zorder=5,
                       label="osservazione trimestrale")
            _shade(ax, idx)
            ax.set_ylabel(c, fontsize=9)
            ax.legend(fontsize=7, loc="lower left")
            ax.grid(alpha=0.3)
        axes[0].set_title("MM fill: mensile latente vs osservazioni trimestrali "
                          r"($x^Q_m = 2\xi_m + \xi_{m-1}$)", fontsize=11)
        _save(fig, "mm_fill_verification.png")

    # 5. Heatmap di Lambda — la mask e' visibile a occhio (bianco = fuori-mask)
    Lam = np.asarray(out["theta"]["Lambda"])
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * fs.r, 0.26 * len(cols) + 1.6))
    vmax = float(np.abs(Lam).max()) or 1.0
    masked = np.ma.masked_where(fs.mask == 0, Lam)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(fs.r), fs.factor_names)
    ax.set_yticks(range(len(cols)), cols, fontsize=6)
    for i in range(len(cols)):
        for j in range(fs.r):
            if fs.mask[i, j]:
                ax.text(j, i, f"{Lam[i, j]:+.2f}", ha="center", va="center", fontsize=5)
    ax.set_title(f"$\\Lambda$ — spec '{spec}', variante '{variant}'\n"
                 f"bianco = fuori mask (zero esatto)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6)
    _save(fig, "lambda_heatmap.png")

    # 6. Pesi Student-t nel tempo (solo se stimati)
    if not VARIANTS[variant]["gaussian"]:
        est = out["e_step_output"]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        a1.plot(idx, est["w_u"], lw=0.9, color="darkorange")
        a1.set_ylabel(r"$w_u$ (fattori)")
        a2.plot(idx, est["w_eps"], lw=0.9, color="seagreen")
        a2.set_ylabel(r"$w_\epsilon$ (idiosincratici)")
        for ax in (a1, a2):
            _shade(ax, idx)
            ax.axhline(1.0, lw=0.6, color="black", ls="--")
            ax.grid(alpha=0.3)
        a1.set_title(f"Pesi Student-t — spec '{spec}', variante '{variant}'  "
                     f"($\\nu_u$={float(out['theta']['nu_u']):.1f}, "
                     f"$\\nu_\\epsilon$={float(out['theta']['nu_eps']):.1f})   "
                     f"basso = osservazione declassata", fontsize=11)
        _save(fig, "weights_student_t.png")

    # 7. rho stimati per serie (solo varianti con idio AR(1))
    if VARIANTS[variant]["idio_ar1"] and "rho" in out["theta"]:
        rho = np.asarray(out["theta"]["rho"], float).ravel()
        order = np.argsort(rho)
        fig, ax = plt.subplots(figsize=(9, 0.24 * len(cols) + 1.8))
        colors = ["crimson" if prep["freq_list"][i] == "quarterly" else "steelblue"
                  for i in order]
        ax.barh(range(len(cols)), rho[order], color=colors)
        ax.set_yticks(range(len(cols)), [cols[i] for i in order], fontsize=6)
        ax.axvline(0, lw=0.8, color="black")
        ax.set_xlabel("rho_i stimato")
        ax.set_title(
            "Persistenza idiosincratica - spec '" + spec + "', variante '"
            + variant + "'" + chr(10) + "(rosso = trimestrale)   "
            + f"mediana {np.median(rho):+.2f}, "
            + f"{int((rho > 0.5).sum())}/{len(rho)} sopra 0.5",
            fontsize=10)
        ax.grid(alpha=0.3, axis="x")
        _save(fig, "idio_ar1_rho.png")

    return written


# ─── Un giro completo ─────────────────────────────────────────────────────────

def run_one(spec: str, variant: str, max_iter: int, verbose: bool) -> dict:
    cfg = VARIANTS[variant]
    print("\n" + "=" * 78)
    print(f"  SPEC '{spec}'  |  variante '{variant}'  "
          f"(gaussian={cfg['gaussian']}, idio_ar1={cfg['idio_ar1']})")
    print("=" * 78)

    prep = prepare(spec, idio_ar1=cfg["idio_ar1"])
    fs = prep["fs"]
    if cfg.get("per_series_weights"):
        prep["theta0"]["per_series_weights"] = True
    if cfg.get("inner_criterion"):
        prep["theta0"]["inner_criterion"] = cfg["inner_criterion"]
    save_theta_initial(prep, spec, variant)
    print(f"  struttura : r={fs.r}, fattori={fs.factor_names}, "
          f"{'diagonale' if fs.diagonal else 'NON diagonale'}  "
          f"(serie/fattore: {[int(fs.mask[:, j].sum()) for j in range(fs.r)]})")

    out = fit_dfm(
        prep["Y_std_df"].to_numpy(), prep["theta0"],
        freq_list=prep["freq_list"], block_map=fs, ordered_cols=prep["cols"],
        gaussian=cfg["gaussian"], max_iter=max_iter, verbose=verbose,
    )

    save_fit(out, prep, spec, variant)
    figs = make_figures(out, prep, spec, variant)

    ll = np.asarray(out["loglik_history"], dtype=float)
    dll = np.diff(ll)
    f = np.asarray(out["f_smooth"])[:, :fs.r]
    print(f"\n  converged        : {out['converged']} (n_iter={out['n_iter']})")
    print(f"  loglik           : {ll[0]:.2f} -> {ll[-1]:.2f}")
    print(f"  ELBO monotono    : {(dll >= -1e-6).all()}  "
          f"(min Delta = {dll.min():.2e}, violazioni={out['monotonicity_violations']})")
    print(f"  var(f) [~1]      : {np.array2string(f.var(axis=0), precision=3)}")
    print(f"  rho(A)           : {np.max(np.abs(np.linalg.eigvals(np.asarray(out['theta']['A'])))):.4f}")
    if out.get("leak_coef"):
        print(f"  leak G-locali    : "
              f"{{{', '.join(f'{k}:{v:+.3f}' for k, v in out['leak_coef'].items())}}}")
    if not cfg["gaussian"]:
        print(f"  nu_u, nu_eps     : {float(out['theta']['nu_u']):.2f}, "
              f"{float(out['theta']['nu_eps']):.2f}")
    if cfg["idio_ar1"] and "rho" in out["theta"]:
        _r = np.asarray(out["theta"]["rho"], float).ravel()
        print(f"  rho idio         : mediana {np.median(_r):+.3f}, "
              f"range [{_r.min():+.3f}, {_r.max():+.3f}], "
              f"{int((_r > 0.5).sum())}/{len(_r)} sopra 0.5")
    print(f"  artefatti        : {_proc_dir(spec, variant)}")
    for p in figs:
        print(f"                     {os.path.relpath(p, _ROOT)}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="EM sul dataset 'final' + tutti gli artefatti, per spec.")
    ap.add_argument("--spec", choices=SPECS, default=None)
    ap.add_argument("--all", action="store_true", help="tutte e tre le spec")
    ap.add_argument("--variant", choices=list(VARIANTS), default="gaussian")
    ap.add_argument("--all-variants", action="store_true",
                    help=f"tutte e {len(VARIANTS)} le varianti per ogni spec "
                         f"({len(SPECS) * len(VARIANTS)} celle con --all)")
    ap.add_argument("--max-iter", type=int, default=500,
                    help="tetto di sicurezza, non un parametro di taratura: "
                         "raggiungerlo viene riportato come NON convergenza")
    ap.add_argument("--skip-existing", action="store_true",
                    help="salta le celle che hanno gia' un fit_dfm_result.json "
                         "(per riprendere una run interrotta senza rifare tutto)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if not a.spec and not a.all:
        ap.error("passa --spec <nome> oppure --all")
    specs = list(SPECS) if a.all else [a.spec]
    variants = list(VARIANTS) if a.all_variants else [a.variant]

    summary = []
    for spec in specs:
        for var in variants:
            done = os.path.join(_ROOT, "data", "processed", "final", spec, var,
                                "fit_dfm_result.json")
            if a.skip_existing and os.path.isfile(done):
                with open(done, encoding="utf-8") as fh:
                    d = json.load(fh)
                print(f"\n  [skip] {spec}/{var} gia' presente "
                      f"(n_iter={d['n_iter']}, loglik={d['loglik_last']:.2f})")
                summary.append((spec, var, d["converged"], d["n_iter"],
                                d["loglik_last"], d["elbo_monotone"]))
                continue
            out = run_one(spec, var, a.max_iter, verbose=not a.quiet)
            ll = np.asarray(out["loglik_history"], dtype=float)
            summary.append((spec, var, out["converged"], out["n_iter"], ll[-1],
                            bool((np.diff(ll) >= -1e-6).all())))

    print("\n" + "=" * 78)
    print("  RIEPILOGO")
    print("=" * 78)
    # La colonna `variante` si dimensiona sul nome piu' lungo EFFETTIVAMENTE
    # stampato (minimo 16, il default): con `student_t_ar1_shared` la
    # larghezza fissa veniva sfondata e il nome si incollava alla colonna
    # `conv`, producendo righe come "student_t_ar1_sharedTrue".
    _wv = max(16, max((len(e) for _, e, *_ in summary), default=16) + 1)
    print(f"  {'spec':<14s}{'variante':<{_wv}s}{'conv':<7s}{'n_iter':>7s}"
          f"{'loglik':>13s}   ELBO monotono")
    for s, e, c, n, l, m in summary:
        print(f"  {s:<14s}{e:<{_wv}s}{str(c):<7s}{n:>7d}{l:>13.2f}   {m}")


if __name__ == "__main__":
    main()
