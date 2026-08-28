"""
QUANTO PESA DAVVERO Sigma_0 = I: la misura dietro il Remark 29.

    cd <radice del repo>
    python scripts/diag_sigma0_equivariance.py --estimator gaussian

PERCHE' ESISTE
--------------
Il Remark 29 dell'appendice sull'inizializzazione (`rem:sigma0-equivariance`)
dichiara due numeri: ruotando theta^(0) con un elemento NON ortogonale di G,
la log-verosimiglianza convergente si sposta "di circa 10^-5 del suo valore"
e i loadings canonici "di circa 10^-3 dei loro".  Servono a chiudere il
caveat: la Proposizione di equivarianza non vale alla lettera perche'
Sigma_0 = I non e' equivariante (lo e' la stazionaria di Lyapunov), ma
l'effetto residuo e' trascurabile.

Il problema e' che 10^-5 E' LA TOLLERANZA DI ARRESTO.  Il criterio di
`run_em` e' relativo — |L^(j) - L^(j-1)| / |L^(j-1)| < tol_outer — e il
default, quello che usano le passate vere (`run_final_model.py` non lo
sovrascrive), e' tol_outer = 1e-5 (`em_main.py`, riga ~1696).  In unita'
assolute, su |L| ~ 9700, la banda vale ~0.097 di log-verosimiglianza: e' il
conto che sta gia' in `src/check_final_artifacts.py` riga ~147.

Due run che partono da rappresentanti diversi dell'orbita si fermano a
iterazioni diverse e possono differire di quell'ordine PER QUALUNQUE
RAGIONE, senza bisogno di alcuna non-equivarianza.  Il numero pubblicato,
cosi' com'e', non misura l'effetto che il remark gli attribuisce: e' un
maggiorante che coincide col pavimento numerico dell'algoritmo.  Stesso
sospetto, indiretto, sul 10^-3 dei loadings: vicino a un punto stazionario
la verosimiglianza e' piatta, quindi uno scarto ENTRO tolleranza sulla L e'
compatibile con uno scarto molto piu' grande sui parametri.

Questo script separa le due cose facendo variare la tolleranza.  Se lo
scarto scende con `tol_outer`, era il criterio di arresto e la formulazione
onesta e' "al di sotto della tolleranza di convergenza" — che e' un'
affermazione PIU' FORTE di quella attuale.  Se resta inchiodato a ~10^-5
mentre la tolleranza scende di due ordini, e' un effetto vero di Sigma_0 e
il numero va tenuto, con il setup dichiarato.

I QUATTRO BRACCI
----------------
Tutti partono dallo STESSO theta^(0) (stesso seme del fill gaussiano: due
estrazioni diverse possono convergere altrove, ed e' il Remark 28, non
questo).  Cambia solo il rappresentante dell'orbita da cui si parte:

    base   theta^(0) cosi' com'e'                        Sigma_0 = I
    rotI   T_G(theta^(0)), G NON ortogonale              Sigma_0 = I   <-- il codice vero
    rotS   T_G(theta^(0)), G NON ortogonale              Sigma_0 ruotato (equivariante)
    sign   T_D(theta^(0)), D = diag(+-1) ORTOGONALE      Sigma_0 = I   <-- controllo

`rotI` e' quello che l'implementazione fa davvero: `compute_theta_initial`
scrive I_{5r} qualunque sia il rappresentante, e da li' in poi Sigma_0 e'
congelato (`em_m_step.py` riga ~1955, con assert a riga ~2878).  E' il
braccio che ha prodotto i numeri pubblicati.

`rotS` e' il caso "Sigma_0 trasformato" che il remark dice coincidere a
precisione macchina; `sign` e' il caso ortogonale, che il remark dice
coprire alla lettera.  Se questi due NON tornano a precisione macchina, il
difetto non e' Sigma_0 ma il banco di prova, e i numeri di `rotI` non
significano nulla: sono i controlli dell'esperimento, non un contorno.

LA G
----
`data/processed/_rot_G.npz` — l'unica traccia rimasta dell'esperimento del
18 agosto.  E' un elemento di G per `fed_overlap`: diag(1.7, 0.6, 1.4, 0.8)
piu' la colonna di leak (., 0.4, -0.3, 0.25) sul fattore globale, cioe' i
2r-1 = 7 parametri liberi.  La convenzione e' quella di `eq:app-leak`:
Lambda -> Lambda G, f -> G^{-1} f, A -> G^{-1} A G, Q -> G^{-1} Q G^{-T}.
Non e' ortogonale (G G' != I, det = 1.142), che e' esattamente il caso in
cui il remark ammette il difetto.

NON SCRIVE FRA GLI ARTEFATTI VERI: `fit_dfm` e' chiamato con
`save_path=None` (nessuna persistenza, nessuna cache toccata) e il JSON di
uscita finisce dove dice `--out`, che di default sta fuori dal repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from em.em_initialization import (  # noqa: E402
    standardize, mm_fill_quarterly, gaussian_fill_ragged,
    pca_initialization, compute_theta_initial,
)
from em.em_main import fit_dfm  # noqa: E402
from run_final_model import _prep_final  # noqa: E402  (stessa preparazione del runner vero)

SEP = "=" * 100
ARMS = ("base", "rotI", "rotS", "sign")


# ─────────────────────────────────────────────────────────────────────────────
#  Il punto di partenza, e i suoi rappresentanti
# ─────────────────────────────────────────────────────────────────────────────

def build_theta0(spec: str):
    """
    theta^(0) IDENTICO a quello di `run_final_model.run` (stesse chiamate,
    stesso seme 42 nel fill gaussiano).  Il pannello standardizzato Y e il
    fill si costruiscono UNA volta sola e si riusano per tutti i bracci:
    l'esperimento deve far variare il rappresentante dell'orbita e nient'altro.
    """
    df, cols, freq_list, structure = _prep_final(spec)

    Y_std_df, _mean, _std = standardize(df)
    Y_mm = Y_std_df.copy()
    for c, fr in zip(cols, freq_list):
        if fr == "quarterly":
            Y_mm[c] = mm_fill_quarterly(Y_std_df[c])
    Y_filled = gaussian_fill_ragged(Y_mm, random_state=42)
    F0, _info = pca_initialization(Y_filled, structure)
    theta0 = compute_theta_initial(Y_filled, F0, structure)

    return Y_std_df.to_numpy(), cols, freq_list, structure, theta0


def rotate_theta0(theta0: dict, G: np.ndarray, rotate_sigma0: bool) -> dict:
    r"""
    T_G(theta^(0)) nella convenzione di `eq:app-leak`:

        Lambda -> Lambda G,   A -> G^{-1} A G,   Q -> G^{-1} Q G^{-T},

    e, SE `rotate_sigma0`, anche

        Sigma_0 -> G_aug^{-1} Sigma_0 G_aug^{-T},   G_aug = blkdiag(G, ..., G).

    E' la stessa algebra di `monte_carlo_recovery.apply_factor_rotation`
    (che pero' non tocca Sigma_0, ed e' proprio il perno di questa misura).
    theta^(0) non porta momenti smoothed: non c'e' altro da ruotare.
    """
    G = np.asarray(G, float)
    r = np.asarray(theta0["A"]).shape[0]
    if G.shape != (r, r):
        raise ValueError(f"G ha forma {G.shape}, attesa ({r}, {r}): la spec ha r={r}.")
    G_inv = np.linalg.inv(G)

    out = {k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else v)
           for k, v in theta0.items()}
    out["Lambda"] = np.asarray(theta0["Lambda"]) @ G
    out["A"] = G_inv @ np.asarray(theta0["A"]) @ G
    out["Q"] = G_inv @ np.asarray(theta0["Q"]) @ G_inv.T

    Sig0 = np.asarray(theta0["Sigma_0"])
    if Sig0.shape != (5 * r, 5 * r):
        raise ValueError(
            f"Sigma_0 ha forma {Sig0.shape}, attesa ({5*r}, {5*r}). Lo script "
            "misura il blocco dei fattori: con l'idio nello stato (Asse B) il "
            "blocco idiosincratico e' la stazionaria dell'AR(1) e va trattato a "
            "parte (Remark 26)."
        )
    if rotate_sigma0:
        G_aug_inv = np.kron(np.eye(5), G_inv)
        out["Sigma_0"] = G_aug_inv @ Sig0 @ G_aug_inv.T

    return out


def check_mask_preserved(Lam0: np.ndarray, Lam_rot: np.ndarray, tag: str) -> None:
    """
    G preserva il pattern di zeri (e' il senso del gruppo): la colonna del
    globale non ha zeri, le locali sono riscalate.  Se questo salta, il punto
    di partenza ruotato NON e' nello spazio dei parametri e il confronto e'
    senza senso — meglio fermarsi che pubblicare il numero.
    """
    zeros = np.asarray(Lam0) == 0.0
    leak = float(np.abs(np.asarray(Lam_rot)[zeros]).max()) if zeros.any() else 0.0
    print(f"  [{tag}] maschera preservata: max |Lambda_rot| sugli zeri di "
          f"Lambda^(0) = {leak:.3e}")
    if leak > 1e-12:
        raise RuntimeError(
            f"[{tag}] G non preserva la maschera (leak {leak:.3e}): non e' un "
            "elemento del gruppo per questa spec."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Un braccio = una stima
# ─────────────────────────────────────────────────────────────────────────────

def run_arm(arm, theta_init, Y, cols, freq_list, structure,
            estimator, tol_outer, max_iter):
    t0 = time.time()
    out = fit_dfm(
        Y, theta_init,
        freq_list=freq_list,
        block_map=structure,
        ordered_cols=cols,
        gaussian=(estimator == "gaussian"),
        tol_outer=tol_outer,
        max_iter=max_iter,
        verbose=False,
        save_path=None,          # nessuna persistenza: non tocca gli artefatti veri
    )
    ll = np.asarray(out["loglik_history"], dtype=float)
    return {
        "arm": arm,
        "converged": bool(out["converged"]),
        "n_iter": int(out["n_iter"]),
        "loglik": float(ll[-1]),
        "Lambda": np.asarray(out["theta"]["Lambda"], dtype=float),
        "leak_coef": {k: float(v) for k, v in (out.get("leak_coef") or {}).items()},
        "seconds": time.time() - t0,
    }


def compare(res, base):
    """Scarti del braccio `res` rispetto al `base` DELLA STESSA tolleranza."""
    L_b = base["loglik"]
    dL = abs(res["loglik"] - L_b)
    Lam_b = base["Lambda"]
    dLam = np.linalg.norm(res["Lambda"] - Lam_b) / np.linalg.norm(Lam_b)
    return {
        "dL_abs": dL,
        "dL_rel": dL / abs(L_b),
        "dLambda_rel": float(dLam),
        "dLambda_max": float(np.abs(res["Lambda"] - Lam_b).max()),
    }


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Misura l'effetto di Sigma_0 = I sulla non-equivarianza, "
                    "separandolo dalla tolleranza di arresto (Remark 29).")
    ap.add_argument("--spec", choices=["fed_overlap", "diag4", "diag3"],
                    default="fed_overlap",
                    help="la G salvata e' 4x4 con leak sul globale: fed_overlap.")
    ap.add_argument("--estimator", choices=["gaussian", "student_t"], default="gaussian")
    ap.add_argument("--tol", default="1e-5,1e-6,1e-7",
                    help="tolleranze da confrontare (la prima e' quella pubblicata).")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help=f"bracci da girare, fra {ARMS} ('base' e' obbligatorio).")
    ap.add_argument("--max-iter", type=int, default=500,
                    help="alzato rispetto ai 200 del runner: a tolleranza stretta "
                         "il tetto di iterazioni diventa lui il vincolo.")
    ap.add_argument("--G", default=os.path.join(_ROOT, "data", "processed", "_rot_G.npz"))
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("TEMP", "/tmp"), "sigma0_equivariance.json"))
    a = ap.parse_args()

    tols = [float(x) for x in a.tol.split(",") if x.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if "base" not in arms:
        arms.insert(0, "base")
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"bracci sconosciuti: {sorted(unknown)}; ammessi {ARMS}")

    print(SEP)
    print("  Sigma_0 = I: effetto vero o pavimento numerico?   (Remark 29)")
    print(SEP)

    G = np.load(a.G)["G"]
    Y, cols, freq_list, structure, theta0 = build_theta0(a.spec)
    r = np.asarray(theta0["A"]).shape[0]

    np.set_printoptions(precision=4, suppress=True)
    print(f"  spec={a.spec}  estimator={a.estimator}  r={r}  "
          f"T x M = {Y.shape[0]} x {Y.shape[1]}  max_iter={a.max_iter}")
    print(f"  G da {os.path.relpath(a.G, _ROOT)}:  det={np.linalg.det(G):+.4f}   "
          f"max|G G' - I| = {np.abs(G @ G.T - np.eye(r)).max():.3f}  "
          f"(ortogonale: {'si' if np.allclose(G @ G.T, np.eye(r)) else 'NO'})")

    # I rappresentanti.  D = diag(+-1) alternati: e' ortogonale, sta in G sotto
    # ogni pattern della tabella, e normalize_signs lo deve annullare.
    D = np.diag([(-1.0) ** k for k in range(r)])
    starts = {
        "base": theta0,
        "rotI": rotate_theta0(theta0, G, rotate_sigma0=False),
        "rotS": rotate_theta0(theta0, G, rotate_sigma0=True),
        "sign": rotate_theta0(theta0, D, rotate_sigma0=False),
    }
    Lam0 = np.asarray(theta0["Lambda"])
    for tag in ("rotI", "sign"):
        if tag in arms:
            check_mask_preserved(Lam0, starts[tag]["Lambda"], tag)

    # Il metro "di ordine uno" con cui il remark confronta lo scarto finale:
    # quanto la rotazione sposta i loadings AL PUNTO DI PARTENZA.
    shift0 = float(np.linalg.norm(Lam0 @ G - Lam0) / np.linalg.norm(Lam0))
    print(f"  rotazione iniziale sui loadings: ||Lambda^(0) G - Lambda^(0)||_F / "
          f"||Lambda^(0)||_F = {shift0:.4f}   <- il termine di paragone")

    rows = []
    for tol in tols:
        print(f"\n{'-'*100}\n  tol_outer = {tol:.0e}\n{'-'*100}")
        res = {}
        for arm in arms:
            res[arm] = run_arm(arm, starts[arm], Y, cols, freq_list, structure,
                               a.estimator, tol, a.max_iter)
            r_ = res[arm]
            print(f"    {arm:5s}  conv={str(r_['converged']):5s}  "
                  f"n_iter={r_['n_iter']:4d}  L={r_['loglik']:.6f}  "
                  f"({r_['seconds']:.1f}s)")
            if not r_["converged"]:
                print(f"    !! {arm} NON converge entro max_iter={a.max_iter}: "
                      "lo scarto misura il tetto di iterazioni, non Sigma_0.")

        base = res["base"]
        band = tol * abs(base["loglik"])          # il passo massimo che l'arresto ammette
        print(f"\n    banda dell'arresto = tol*|L| = {band:.6f} unita' di loglik")
        print(f"    {'arm':5s} {'|dL|':>12s} {'|dL|/|L|':>11s} {'|dL|/banda':>11s} "
              f"{'dLambda_rel':>12s} {'dLambda_max':>12s}")
        for arm in arms:
            if arm == "base":
                continue
            c = compare(res[arm], base)
            print(f"    {arm:5s} {c['dL_abs']:12.3e} {c['dL_rel']:11.3e} "
                  f"{c['dL_abs']/band:11.3f} {c['dLambda_rel']:12.3e} "
                  f"{c['dLambda_max']:12.3e}")
            rows.append({"tol": tol, **{k: v for k, v in res[arm].items()
                                        if k != "Lambda"}, **c,
                         "base_loglik": base["loglik"],
                         "base_n_iter": base["n_iter"],
                         "tol_band": band})

    # ── Lettura ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}\n  COME SI LEGGE\n{SEP}")
    print("  rotI  scende con tol   -> i 10^-5 / 10^-3 pubblicati erano il criterio di")
    print("        arresto, non Sigma_0.  In tesi: 'al di sotto della tolleranza di")
    print("        convergenza', piu' forte di quanto scritto ora.")
    print("  rotI  resta piatto     -> effetto vero di Sigma_0: il numero si tiene, ma")
    print("        va dichiarato il setup (spec, stimatore, G, tolleranza, norma).")
    print("  rotS  e  sign  devono stare a precisione macchina a OGNI tolleranza:")
    print("        se non lo fanno, il banco di prova e' rotto e rotI non dice nulla.")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({
            "spec": a.spec, "estimator": a.estimator, "r": r,
            "T": int(Y.shape[0]), "M": int(Y.shape[1]),
            "max_iter": a.max_iter, "tolerances": tols, "arms": arms,
            "G": np.asarray(G).tolist(),
            "G_orthogonal": bool(np.allclose(G @ G.T, np.eye(r))),
            "lambda_shift_at_start": shift0,
            "numpy": np.__version__,
            "rows": rows,
        }, fh, indent=2)
    print(f"\n  JSON -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
