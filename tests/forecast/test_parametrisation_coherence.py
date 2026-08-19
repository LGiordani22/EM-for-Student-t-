"""
core/forecast/test_parametrisation_coherence.py

Il GUARDIANO della scala del nowcast.

`fit_dfm` restituisce due parametrizzazioni dello stesso modello, equivalenti in
distribuzione ma non mescolabili:

    CANONICA  fit["theta"], fit["f_smooth"], fit["P_smooth"]
              (ortogonalizzazione + segno + Convenzione 1)
    RAW       tutto cio' che sta in fit["e_step_output"]

La Convenzione 1 riscala `f_can = f_raw / c` e `Lambda_can = Lambda_raw * c`.
Il prodotto `Lambda @ f` e' quindi invariante — ma solo se i due fattori vengono
dallo stesso mondo.  Prendere `Lambda_tilde` dal mondo raw e lo stato da quello
canonico restituisce `z_vero / c`: un nowcast compresso, senza alcun errore
visibile.  Su `diag3` valeva 2.58x.

La compressione ha una firma temporale, ed e' la ragione per cui questo test
esiste: si manifesta SOLO nelle settimane in cui si ri-stima, perche' le
settimane a theta congelato passano da `filter_only`, che e' coerente per
costruzione.  Sotto `--em-frequency monthly` il primo venerdi' del mese
uscirebbe compresso e gli altri no — una seghettatura mensile che sembra
segnale e non lo e'.

I due test, quindi:

  1. COERENZA  A parita' di vintage, una settimana con ri-stima e una congelata
               devono dare lo stesso z: raw e canonico non vanno mescolati.
  2. INVARIANZA `Lambda_can @ f_can` deve valere quanto `Lambda_raw @ f_raw`.
               Verifica che i due mondi siano davvero equivalenti e che
               `nowcast` ne stia usando uno intero, non mezzo per uno.
               Girato dove `build_Lambda_tilde` basta a ricostruire la Lambda
               canonica, cioe' sulle varianti senza idio nello stato.

Esegui:  python -m core.forecast.test_parametrisation_coherence
Lento: stima l'EM per davvero (qualche minuto).
"""

from __future__ import annotations

import numpy as np

from core.forecast import scale
from core.forecast.nowcast_engine import (
    SPECS, prepare_vintage, estimate, filter_only, extract_target, nowcast,
)
from core.forecast.release_calendar import load_metadata, load_panel, quarter_end

from kalman import build_Lambda_tilde

#: Il vintage di riferimento del test.
AS_OF, TARGET = "2008-12-19", "2008Q4"

#: Valore atteso di z per diag3/gaussian a quel vintage (la parametrizzazione
#: mista darebbe -0.4950).
_Z_ATTESO_DIAG3 = -1.2765

#: Tolleranza relativa fra settimana EM e settimana congelata.  Non e' zero: la
#: settimana congelata ri-gira l'E-step da capo (e sotto Student-t anche il
#: ciclo ECM interno), quindi lo stato puo' muoversi all'ultima cifra.  Una
#: differenza di parametrizzazione, invece, vale decine di punti percentuali.
_TOL_REL = 5e-3


def _hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def test_coerenza_em_vs_congelato(spec: str, variant: str, panel, meta) -> bool:
    """
    Stesso vintage, due strade: EM e theta congelato.  Lo z deve coincidere.

    Verifica l'invariante: la strada EM e quella congelata devono dare lo stesso
    z.  Una parametrizzazione mista restituirebbe `z_vero / c` (con `c` fino a 2.58).
    """
    v = prepare_vintage(AS_OF, TARGET, spec, variant, panel=panel, metadata=meta)

    fit = estimate(v, max_iter=250)
    eso_em = fit["e_step_output"]
    z_em = extract_target(v, np.asarray(eso_em["f_smooth"], float),
                          eso_em["Lambda_tilde"],
                          P_smooth=eso_em.get("P_smooth"),
                          theta=fit["theta"])

    # Stesso vintage, parametri congelati al theta appena stimato.
    eso_fr = filter_only(v, fit["theta"])
    z_fr = extract_target(v, np.asarray(eso_fr["f_smooth"], float),
                          eso_fr["Lambda_tilde"],
                          P_smooth=eso_fr.get("P_smooth"),
                          theta=fit["theta"])

    den = max(abs(z_fr["z"]), 1e-8)
    err_z = abs(z_em["z"] - z_fr["z"]) / den
    err_sd = (abs(z_em["sd_z"] - z_fr["sd_z"]) / max(abs(z_fr["sd_z"]), 1e-8)
              if np.isfinite(z_fr["sd_z"]) else 0.0)
    ok = bool(err_z < _TOL_REL and err_sd < _TOL_REL)

    print(f"  {spec:12s} {variant:22s} "
          f"z(EM)={z_em['z']:+8.4f}  z(congelato)={z_fr['z']:+8.4f}  "
          f"scarto={err_z:.2e}   "
          f"sd={z_em['sd_z']:.4f}/{z_fr['sd_z']:.4f}   "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        rap = z_em["z"] / z_fr["z"] if z_fr["z"] else float("nan")
        print(f"      -> rapporto z(EM)/z(congelato) = {rap:.4f}.  Se somiglia a un "
              f"fattore di Convenzione 1, e' tornato il mix raw/canonico.")
    return ok


def test_invarianza_raw_canonico(spec: str, variant: str, panel, meta) -> bool:
    """
    `Lambda_raw @ f_raw` == `Lambda_can @ f_can`: i due mondi sono equivalenti.

    Ricostruisce la Lambda canonica con `build_Lambda_tilde(theta["Lambda"])`,
    che rende (M, 5r): vale quindi solo dove l'idio NON e' nello stato.  Dove lo
    e', la coda `C_idio` non e' ricostruibile e il confronto si salta — ed e'
    esattamente la ragione per cui `nowcast` usa il mondo raw, che e' completo.
    """
    v = prepare_vintage(AS_OF, TARGET, spec, variant, panel=panel, metadata=meta)
    fit = estimate(v, max_iter=250)
    eso = fit["e_step_output"]

    f_raw = np.asarray(eso["f_smooth"], float)
    f_can = np.asarray(fit["f_smooth"], float)
    Lt_raw = np.asarray(eso["Lambda_tilde"], float)
    r = v["structure"].r

    if Lt_raw.shape[1] != 5 * r:
        print(f"  {spec:12s} {variant:22s} [saltato] idio nello stato "
              f"({Lt_raw.shape[1]} colonne > 5r={5*r}): Lambda canonica non "
              f"ricostruibile, ed e' il motivo per cui si usa la raw.")
        return True

    Lt_can = build_Lambda_tilde(np.asarray(fit["theta"]["Lambda"], float),
                                v["freq_list"])
    i = v["cols"].index("GDPC1")
    t = v["Y_std_df"].index.get_loc(quarter_end(TARGET))

    z_raw = float(Lt_raw[i] @ f_raw[t])
    z_can = float(Lt_can[i] @ f_can[t])
    z_misto = float(Lt_raw[i] @ f_can[t])          # parametrizzazione mista

    err = abs(z_raw - z_can) / max(abs(z_can), 1e-8)
    ok = bool(err < _TOL_REL)
    print(f"  {spec:12s} {variant:22s} "
          f"z(raw)={z_raw:+8.4f}  z(can)={z_can:+8.4f}  scarto={err:.2e}   "
          f"{'OK' if ok else 'FAIL'}")
    print(f"      (il prodotto misto darebbe {z_misto:+8.4f}: "
          f"compressione {abs(z_can / z_misto) if z_misto else float('nan'):.3f}x)")
    return ok


def test_valore_atteso_diag3(panel, meta) -> bool:
    """Caso di regressione."""
    r = nowcast(AS_OF, TARGET, "diag3", "gaussian", panel=panel, metadata=meta,
                max_iter=250)
    err = abs(r["nowcast_z"] - _Z_ATTESO_DIAG3)
    ok = bool(err < 0.02)
    print(f"  diag3/gaussian  z={r['nowcast_z']:+.4f}  atteso={_Z_ATTESO_DIAG3:+.4f}  "
          f"(la parametrizzazione mista darebbe -0.4950)   {'OK' if ok else 'FAIL'}")
    print(f"      BEA={r['nowcast_bea']:+.4f}%  "
          f"mean={r['mean_train']:+.4f}  std={r['std_train']:.4f}")
    return ok


def main() -> None:
    panel, meta = load_panel(), load_metadata()
    ok = True

    _hr("1. COERENZA — stesso vintage, settimana EM vs settimana congelata")
    print("   (lo z deve coincidere: raw e canonico non vanno mescolati)\n")
    for spec in SPECS:
        for variant in ("gaussian", "student_t"):
            ok &= test_coerenza_em_vs_congelato(spec, variant, panel, meta)

    _hr("2. COERENZA sotto *_ar1 — la coda idiosincratica nello stato")
    print("   (la riga di Lambda_tilde ha in coda C_idio: la parametrizzazione\n"
          "    dev'essere coerente anche li', non solo sui primi 5r)\n")
    for variant in ("gaussian_ar1", "student_t_ar1_shared"):
        ok &= test_coerenza_em_vs_congelato("diag3", variant, panel, meta)

    _hr("3. INVARIANZA — i due mondi valgono lo stesso")
    print()
    for spec in SPECS:
        ok &= test_invarianza_raw_canonico(spec, "gaussian", panel, meta)

    _hr("4. REGRESSIONE — il caso di riferimento")
    print()
    ok &= test_valore_atteso_diag3(panel, meta)

    _hr("TUTTO OK" if ok else "QUALCOSA NON TORNA")


if __name__ == "__main__":
    main()
