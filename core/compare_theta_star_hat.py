"""
`theta_star` non e' recuperabile, oppure l'EM non lo trova?

IL FATTO DA SPIEGARE
--------------------
Le recovery Monte Carlo mostrano che su alcune celle Student-t la ri-stima non
riproduce `theta_star`: in particolare `diag(Q)` sul fattore L esce 15-30 volte
il valore vero.  Ma i due `theta` differiscono in modo SISTEMATICO, non casuale:

  theta_star (fit sui dati VERI)      sd(f_L) implicata da (A,Q) ~ 0.20
  theta_hat  (ri-stima sul SIMULATO)  sd(f_L) implicata da (A,Q) ~ 0.97

cioe' `theta_hat` e' internamente COERENTE con la Convenzione 1 (varianza totale
unitaria per fattore) e `theta_star` no.  La ri-stima non produce rumore attorno
al vero: produce un modello diverso e piu' coerente.

LE DUE SPIEGAZIONI, E PERCHE' SERVONO DUE PANNELLI
--------------------------------------------------
(a) IDENTIFICAZIONE.  `theta_star` non e' recuperabile dai dati che genera esso
    stesso: la verosimiglianza e' piatta o multimodale in quella direzione, e
    l'EM approda correttamente a un punto equivalente o migliore.
(b) OTTIMO LOCALE.  L'EM sul pannello simulato si ferma in un punto peggiore di
    quello che ha generato i dati.

Le due si distinguono SOLO valutando i due `theta` sul pannello SIMULATO:

    ELBO(theta_hat) > ELBO(theta_star)  sul simulato  ->  caso (a)
    ELBO(theta_star) > ELBO(theta_hat)  sul simulato  ->  caso (b)

Valutarli sul pannello REALE risponde a una domanda DIVERSA — quale dei due
descriva meglio i dati veri — e non e' informativa sul meccanismo: `theta_star`
e' stato stimato proprio massimizzando quella quantita', quindi vince per
costruzione (misurato: da +51 a +902 punti su tutte e sei le celle Student-t
AR(1)).  Riportiamo entrambi i pannelli perche' insieme raccontano la storia
completa, ma il verdetto sul meccanismo lo da' il simulato.

COSTO
-----
Nessun EM.  Il pannello simulato e' deterministico dato (`theta_star`, seed), e
i due `theta` sono gia' su disco: servono due E-step per pannello e per cella.

USO
---
    python core/compare_theta_star_hat.py --spec diag3 --variant student_t_ar1
    python core/compare_theta_star_hat.py --all
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent

from config_utils import SPECS, VARIANTS, parse_spec_variant_args  # noqa: E402
from dfm.em_e_step import run_e_step                                # noqa: E402
from dfm.em_main import compute_elbo_correction                     # noqa: E402
from dfm.selftest_fixture import load_fixture                       # noqa: E402
from run_final_artifacts import VARIANTS as _VFLAGS, prepare       # noqa: E402
from simulate_dfm import simulate_dfm                              # noqa: E402

# Devono coincidere con `monte_carlo_recovery.__main__`, altrimenti il pannello
# rigenerato qui non e' quello su cui `theta_hat` e' stato stimato e il
# confronto non ha senso.
SEED = 42
T_SIM = 2000

# Le chiavi che definiscono il modello.  `rho` c'e' solo sotto idio-AR(1) e
# viene semplicemente omessa dove manca.
_KEYS = ("Lambda", "A", "Q", "R", "Sigma_0", "nu_u", "nu_eps", "rho")


def _theta_from(npz, prefix: str = "") -> dict:
    """Ricostruisce il dict `theta` dalle chiavi appiattite dentro l'`.npz`."""
    th: dict = {}
    for k in _KEYS:
        kk = f"{prefix}{k}"
        if kk in npz.files:
            v = npz[kk]
            th[k] = float(v) if v.ndim == 0 else np.asarray(v)
    return th


def _elbo(Y, theta, freq_list, r, vcfg) -> float:
    """
    ELBO a `theta` fissato: loglik del Kalman + correzione, come nel ciclo EM
    (`em_main.run_em`, punto 1).

    I flag di variante vanno applicati a ENTRAMBI i `theta`: l'E-step sotto pesi
    per-serie calcola una quantita' diversa da quello sotto peso condiviso, e
    confrontarne i valori sarebbe un errore di categoria.
    """
    th = dict(theta)
    if vcfg.get("per_series_weights"):
        th["per_series_weights"] = True
    if vcfg.get("inner_criterion"):
        th["inner_criterion"] = vcfg["inner_criterion"]
    e_out = run_e_step(Y, th, freq_list=freq_list, verbose=False,
                       gaussian=vcfg["gaussian"])
    return float(e_out["loglik"]) + compute_elbo_correction(e_out, th, r, Y)


def _sd_implied(theta: dict, r: int) -> np.ndarray:
    """
    Deviazione standard stazionaria per fattore implicata da (A, Q).

    Sotto la Convenzione 1 deve valere ~1.  `Q` e' la matrice di SCALA, non la
    covarianza: sotto la mistura di scala Student-t la covarianza marginale e'
    `Q * nu/(nu-2)`, e il fattore va incluso o i due rami (gaussiano e
    Student-t) non sono confrontabili.
    """
    A = np.asarray(theta["A"])[:r, :r]
    Q = np.asarray(theta["Q"])[:r, :r]
    S = np.linalg.solve(np.eye(r * r) - np.kron(A, A), Q.ravel()).reshape(r, r)
    sd = np.sqrt(np.diag(S))
    nu = float(theta["nu_u"])
    return sd * np.sqrt(nu / (nu - 2)) if np.isfinite(nu) and nu > 2 else sd


def compare(spec: str, variant: str) -> dict | None:
    vcfg = _VFLAGS[variant]
    fit_p = _ROOT / "data" / "processed" / "final" / spec / variant / "fit_dfm_result.npz"
    rec_p = (_ROOT / "output" / "recovery" / "final" / spec / variant
             / f"mc_recovery_T{T_SIM}.npz")
    for p in (fit_p, rec_p):
        if not p.exists():
            print(f"  [salto] {spec}/{variant}: manca {p.name}")
            return None

    th_star = _theta_from(np.load(fit_p, allow_pickle=True))
    th_hat = _theta_from(np.load(rec_p, allow_pickle=True), prefix="theta_hat__")

    fx = load_fixture(spec)
    r = fx.structure.r
    names = list(fx.structure.factor_names)

    # ── Pannello SIMULATO: rigenerato con gli stessi (theta_star, seed, T) di
    #    `monte_carlo_recovery`, quindi identico a quello su cui `theta_hat` e'
    #    stato stimato. Nessun EM: solo la simulazione.
    freq_map = dict(zip(fx.ordered_cols, fx.freq_list))
    sim = simulate_dfm(
        theta=th_star, T=T_SIM,
        freq_list=[freq_map[c] for c in fx.ordered_cols],
        block_map=fx.structure.display_block_map(),
        ordered_cols=fx.ordered_cols,
        r=r, seed=SEED,
        per_series_weights=vcfg.get("per_series_weights", False),
    )
    Y_sim = sim["Y"]
    fl_sim = [freq_map[c] for c in fx.ordered_cols]

    # ── Pannello REALE
    prep = prepare(spec, idio_ar1=vcfg["idio_ar1"])
    Y_real = prep["Y_std_df"].to_numpy()

    out = {
        "sim_star": _elbo(Y_sim, th_star, fl_sim, r, vcfg),
        "sim_hat": _elbo(Y_sim, th_hat, fl_sim, r, vcfg),
        "real_star": _elbo(Y_real, th_star, prep["freq_list"], r, vcfg),
        "real_hat": _elbo(Y_real, th_hat, prep["freq_list"], r, vcfg),
        "sd_star": _sd_implied(th_star, r),
        "sd_hat": _sd_implied(th_hat, r),
        "names": names,
    }

    d_sim = out["sim_hat"] - out["sim_star"]

    # Soglia di RUMORE sul verdetto. Il test sul solo segno di `d_sim` e'
    # fuorviante: `theta_hat` e' stimato su un pannello finito, quindi anche
    # senza alcun ottimo locale i due ELBO differiscono di qualche decina di
    # punti in un verso o nell'altro. Misurato sulle 15 celle: dodici stanno
    # entro |65| e il segno vi si distribuisce in modo casuale, tre stanno fra
    # 1663 e 2358 — due ordini di grandezza piu' in la'. La soglia separa i due
    # regimi senza tarare nulla di fine: e' in scala RELATIVA a |ELBO| perche'
    # il livello dipende da T e dalla cella.
    _noise = 5e-3 * abs(out["sim_star"])   # ~250 punti su ~50.000
    print(f"\n{spec}/{variant}")
    print(f"   sd(f) implicata (atteso ~1 sotto Convenzione 1)")
    print(f"     theta_star  " + "  ".join(
        f"{names[j]}={out['sd_star'][j]:.2f}" for j in range(r)))
    print(f"     theta_hat   " + "  ".join(
        f"{names[j]}={out['sd_hat'][j]:.2f}" for j in range(r)))
    print(f"   ELBO sul pannello SIMULATO  (il verdetto sul meccanismo)")
    print(f"     theta_star  {out['sim_star']:12.2f}")
    print(f"     theta_hat   {out['sim_hat']:12.2f}")
    if abs(d_sim) < _noise:
        print(f"     -> differenza {d_sim:+.2f}, entro il rumore "
              f"(soglia {_noise:.0f}): NESSUN ottimo locale rilevabile. "
              f"L'EM approda a un punto equivalente a theta_star.")
    elif d_sim > 0:
        print(f"     -> theta_hat vince di {d_sim:+.2f}: (a) IDENTIFICAZIONE. "
              f"L'EM trova un punto MIGLIORE di quello che ha generato i dati, "
              f"quindi theta_star non e' recuperabile — non e' un difetto "
              f"dell'ottimizzatore.")
    else:
        print(f"     -> theta_star vince di {-d_sim:.2f}: (b) OTTIMO LOCALE. "
              f"L'EM non ha trovato il punto che ha generato i dati pur "
              f"essendo migliore: qui l'ottimizzazione atterra altrove.")
    print(f"   ELBO sul pannello REALE  (domanda diversa: chi descrive i dati)")
    print(f"     theta_star  {out['real_star']:12.2f}")
    print(f"     theta_hat   {out['real_hat']:12.2f}")
    print(f"     -> differenza {out['real_star'] - out['real_hat']:+.2f} "
          f"(theta_star e' stimato massimizzando questa quantita': "
          f"vince per costruzione, non e' una prova di nulla)")
    return out


# ─── Scrittura INCREMENTALE del riassunto ─────────────────────────────────────
# Il file si apre con l'intestazione PRIMA di calcolare qualunque cella, e ogni
# cella appende la sua riga appena e' pronta.
#
# La versione precedente componeva tutto in memoria e scriveva alla fine: una
# run interrotta a meta' — o un'eccezione su una singola cella — lasciava zero
# byte su disco e buttava via tutte le celle gia' calcolate. Non e' teorico, e'
# successo (un errore su una chiave nell'ultima riga di stampa ha perso i
# risultati di una cella da 183 iterazioni). Con `--all` la run dura 10-15
# minuti: e' esattamente il caso in cui perdere tutto costa.
#
# Per lo stesso motivo `sd(f)` NON e' piu' una sezione in coda ma sta accanto
# alla riga della sua cella: una coda scritta alla fine avrebbe reintrodotto il
# problema che questa modifica elimina.

_HDR_BAR = "=" * 78


def open_txt(out_path: pathlib.Path, n_celle: int) -> None:
    """Crea il file e scrive l'intestazione. Da chiamare PRIMA del ciclo."""
    import datetime

    L = [_HDR_BAR,
         "  THETA_STAR vs THETA_HAT  —  identificazione o ottimo locale?",
         f"  Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
         f"  T_sim = {T_SIM}, seed = {SEED}, celle attese = {n_celle}",
         "",
         "  theta_star = fit sul pannello REALE",
         "  theta_hat  = ri-stima sul pannello SIMULATO da theta_star (init PCA)",
         "",
         "  Il verdetto lo da' la colonna SIMULATO, dove theta_star e' il vero:",
         "    positivo -> l'EM trova un punto migliore del vero (nessun ottimo locale)",
         "    negativo oltre soglia -> OTTIMO LOCALE: l'EM non raggiunge il vero",
         "  La colonna REALE risponde a un'altra domanda (chi descrive i dati veri)",
         "  e non e' informativa sul meccanismo: theta_star vince per costruzione.",
         "",
         "  sd(f) = deviazione standard implicata da (A,Q); sotto Convenzione 1 ~1.",
         _HDR_BAR, "",
         f"  {'spec':<13s}{'variante':<23s}"
         f"{'SIM hat-star':>14s}{'REAL star-hat':>15s}   esito",
         "  " + "-" * 74]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def append_txt(out_path: pathlib.Path, spec: str, variant: str, m: dict) -> None:
    """Appende la riga di UNA cella. Da chiamare subito dopo averla calcolata."""
    d_sim = m["sim_hat"] - m["sim_star"]
    d_real = m["real_star"] - m["real_hat"]
    noise = 5e-3 * abs(m["sim_star"])
    if abs(d_sim) < noise:
        esito = "entro rumore"
    elif d_sim > 0:
        esito = "nessun ottimo locale"
    else:
        esito = "*** OTTIMO LOCALE ***"

    nm = m["names"]
    r = len(nm)
    riga = (f"  {spec:<13s}{variant:<23s}"
            f"{d_sim:>14.2f}{d_real:>15.2f}   {esito}")
    sd = ("  " + " " * 36 + "sd(f) star "
          + " ".join(f"{nm[j]}={m['sd_star'][j]:.2f}" for j in range(r))
          + "  |  hat "
          + " ".join(f"{nm[j]}={m['sd_hat'][j]:.2f}" for j in range(r)))
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(riga + "\n" + sd + "\n")


def close_txt(out_path: pathlib.Path, n_fatte: int, n_attese: int) -> None:
    """Chiude il file. Segnala se mancano celle, cosi' un file troncato si
    riconosce a colpo d'occhio invece di sembrare completo."""
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write("\n")
        if n_fatte < n_attese:
            fh.write(f"  *** INCOMPLETO: {n_fatte}/{n_attese} celle. "
                     f"Run interrotta o celle saltate. ***\n")
        else:
            fh.write(f"  Completo: {n_fatte}/{n_attese} celle.\n")
        fh.write(_HDR_BAR + "\n")


def main() -> None:
    def _extra(parser):
        parser.add_argument(
            "--all", action="store_true",
            help=f"tutte le {len(SPECS) * len(VARIANTS)} celle "
                 f"(ignora --spec/--variant)")

    args = parse_spec_variant_args(
        "Confronto theta_star (fit reale) vs theta_hat (recovery): "
        "identificazione o ottimo locale?",
        extra=_extra,
    )
    celle = (list(itertools.product(SPECS, VARIANTS)) if args.all
             else [(args.spec, args.variant)])

    print("=" * 78)
    print(f"  theta_star vs theta_hat  |  {len(celle)} cella/e  |  "
          f"T_sim = {T_SIM}, seed = {SEED}")
    print("=" * 78)
    out_path = _ROOT / "output" / "recovery" / "theta_star_vs_hat.txt"
    open_txt(out_path, len(celle))

    n_fatte = 0
    for spec, variant in celle:
        m = compare(spec, variant)
        if m is not None:
            append_txt(out_path, spec, variant, m)
            n_fatte += 1
    close_txt(out_path, n_fatte, len(celle))

    print("\n" + "=" * 78)
    print(f"  Scritto: {out_path}  ({n_fatte}/{len(celle)} celle)")


if __name__ == "__main__":
    main()
