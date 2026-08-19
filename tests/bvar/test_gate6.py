"""
core/bvar/tests/test_gate6.py

GATE 6 — I DUE TEST DI CALENDARIO, da passare PRIMA della passata real-time.

    python -m core.bvar.tests.test_gate6

Il Gate 6 non introduce stima nuova: lo stimatore ha gia' i recovery ai Gate 1
(core sampler) e 2 (wrapper trimestrale), e i quattro modelli hanno gia' i loro
end-to-end ai Gate 3-5.  Quel che e' NUOVO qui e' il calendario, e i due test
sono esattamente i due modi in cui il calendario puo' rompersi.

  §1  L'ORACOLO ANTI-LOOK-AHEAD.  Il pannello a `as_of` contiene solo celle
      gia' pubblicate a quella data?  E' il rischio principale del gate.
  §2  LA RIPRODUZIONE DEL DATO BEA.  Sui trimestri gia' osservati il modello
      deve RIPRODURRE il dato, non approssimarlo.


================================================================================
PERCHE' L'ORACOLO E' IL TEST CHE CONTA
================================================================================
Il look-ahead non da' crash e non da' numeri assurdi: da' **performance troppo
buona**.  Un RMSE che batte il NY Fed di tre volte non insospettisce nessuno
finche' non lo si va a cercare, ed e' il modo peggiore in cui una tesi puo'
sbagliare.  Nella pipeline DFM e' gia' successo.

E' successo anche qui, e questo test l'ha preso: `cbvar.fit` chiamava
`estimation_panel(spec, raw=raw)` **senza `as_of`**, quindi il campione di stima
arrivava a `estimation_end` (2025-09-30) qualunque fosse la data del nowcast.
Un nowcast del 2008 avrebbe stimato su dati fino al 2025.

DISEGNO — un oracolo ESATTO, non di plausibilita':

  a) si INTERCETTA ogni pannello che il codice costruisce, con la `as_of` con
     cui l'ha costruito.  I due punti di passaggio obbligati sono
     `data.build_panel` (mensile) e `qbvar.build_quarterly_panel` (trimestrale,
     medie mobili): tutti e quattro i modelli passano di li';
  b) si ABORTISCE appena il pannello arriva allo stimatore.  `hyper.build_target`
     e' la porta unica del core — l'invariante «il core non vede mai un NaN» —
     quindi basta alzare li' una sentinella per esercitare tutta la catena del
     calendario senza pagare `find_mode` (43 min sul B-BVAR);
  c) su ogni pannello registrato si verifica, cella per cella, che il dato fosse
     gia' pubblicato.  Due controlli indipendenti:
        - VETTORIALE: la maschera che `known_at` produrrebbe a quella `as_of`
          non deve avere NaN dove il pannello ha un numero;
        - PUNTUALE su un campione di celle: `release_date(serie, mese) <= as_of`,
          che NON passa da `known_at` e quindi non e' circolare.
  d) CONTROLLO NEGATIVO.  Si ri-costruisce di proposito il pannello col baco
     (`as_of=None`) e si pretende che l'oracolo lo BOCCI.  Senza questo, un test
     verde non dimostrerebbe niente — potrebbe essere verde perche' non guarda.


================================================================================
PERCHE' LA RIPRODUZIONE BEA SI FA SUL RAMO DI RIUSO
================================================================================
Sui trimestri gia' osservati non c'e' niente da stimare: il PIL e' un dato, e lo
smoother deve restituirlo tale e quale (le celle osservate entrano con R = 0, o
col nostro nugget 1e-12).  Uno scarto rivelerebbe un errore in un punto
qualsiasi della catena — blocking, indici, finestra, de-trasformazione.

E' il controllo che ha chiuso il Gate 3 (entro 0.03 pp).  Qui si applica al
**ramo di riuso**, ed e' la scelta giusta per due ragioni:

  1. il ramo di riuso e' cio' che il Gate 6 aggiunge di nuovo — il ramo pieno e'
     gia' stato verificato end-to-end ai Gate 3, 4 e 5;
  2. il ramo di riuso e' quello che gira in **48 settimane su 52**.

E siccome la riproduzione di una cella OSSERVATA e' una proprieta' del
cablaggio stato-spazio e non dei parametri, il test puo' usare (B, Sigma)
sintetici — un random walk — invece di pagare `find_mode`.  Se il cablaggio e'
giusto la cella torna identica qualunque sia la legge di moto; se e' storto, non
torna con nessuna.  E' un test PIU' severo, non meno: toglie di mezzo la
possibilita' che un parametro ben stimato copra un indice sbagliato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.bvar import bbvar, cbvar, data as bdata, evaluate, lbvar, qbvar
from core.bvar.evaluate import TARGET
from core.bvar.spec import BVARSpec
from core.forecast.release_calendar import (
    horizon_week,
    known_at,
    load_exact_releases,
    load_metadata,
    release_date,
    weekly_grid,
)

_OK, _FAIL = "OK", "FAIL"
_QM = (3, 6, 9, 12)


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<62} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


# ─── La strumentazione: spie e sentinella ─────────────────────────────────────

class _Enough(Exception):
    """Sentinella: il calendario ha finito il suo lavoro, la stima non serve."""


class Spy:
    """
    Registra ogni pannello costruito, con la `as_of` usata per costruirlo.

    Si sostituisce alle funzioni vere per la durata del blocco `with`.  Non
    riscrive niente: chiama l'originale e osserva.
    """

    def __init__(self, *, stop_at_target: bool = True):
        self.panels: list[tuple] = []       # (kind, as_of, DataFrame)
        self.stop = stop_at_target
        self._saved: list[tuple] = []

    def __enter__(self):
        import core.bvar.data as m_data
        import core.bvar.hyper as m_hyper
        import core.bvar.qbvar as m_qbvar

        def wrap(mod, name, kind):
            orig = getattr(mod, name)
            self._saved.append((mod, name, orig))

            def f(spec, as_of=None, *a, **k):
                out = orig(spec, as_of, *a, **k)
                self.panels.append((kind, as_of, out))
                return out
            setattr(mod, name, f)

        wrap(m_data, "build_panel", "monthly")
        wrap(m_qbvar, "build_quarterly_panel", "quarterly")
        # i moduli importano i simboli per nome: vanno ripuntati anche li'
        for mod in (m_qbvar, bbvar, lbvar):
            if hasattr(mod, "build_panel"):
                self._saved.append((mod, "build_panel", mod.build_panel))
                mod.build_panel = m_data.build_panel
        for mod in (cbvar, qbvar):
            if hasattr(mod, "build_quarterly_panel"):
                self._saved.append((mod, "build_quarterly_panel",
                                    mod.build_quarterly_panel))
                mod.build_quarterly_panel = m_qbvar.build_quarterly_panel

        if self.stop:
            def bt(*a, **k):
                raise _Enough

            # `build_target` va sostituito in OGNI modulo che l'ha importato
            # per nome, non solo in `hyper`.  Un `from ... import build_target`
            # crea un binding NUOVO nel modulo importatore, e ripuntare
            # l'originale non lo tocca.  Errore fatto: `lbvar` non abortiva e il
            # test girava `find_mode` sul profilo `l` per intero, 30 minuti.
            import core.bvar.core as m_core
            for mod in (m_hyper, m_core, qbvar, cbvar, bbvar, lbvar):
                if getattr(mod, "build_target", None) is not None:
                    self._saved.append((mod, "build_target", mod.build_target))
                    mod.build_target = bt
        return self

    def __exit__(self, *exc):
        for mod, name, orig in reversed(self._saved):
            setattr(mod, name, orig)
        return False


# ─── §1  L'oracolo ────────────────────────────────────────────────────────────

def audit_panel(kind: str, as_of, panel: pd.DataFrame, meta: pd.DataFrame,
                raw: pd.DataFrame, rng, exact: pd.DataFrame | None = None,
                n_sample: int = 200) -> list[str]:
    """
    Le violazioni di un singolo pannello.  Lista vuota = pulito.

    `as_of = None` e' gia' di per se' una violazione in regime real-time: vuol
    dire che quel pannello e' stato costruito senza guardare il calendario.

    `meta` ed `exact` vanno passati SEMPRE.  Senza, `known_at` e `release_date`
    rileggono il disco a ogni chiamata: misurato, il test passava da secondi a
    oltre 45 minuti.  E' lo stesso motivo per cui `build_panel` li accetta.
    """
    if as_of is None:
        return ["pannello costruito con as_of=None (nessun mascheramento)"]

    D = pd.Timestamp(as_of)
    bad: list[str] = []
    cols = [c for c in panel.columns if c in raw.columns]

    # (a) controllo VETTORIALE contro la maschera che known_at produrrebbe
    if kind == "monthly":
        exp = known_at(raw.loc[:, cols], D, metadata=meta, exact=exact)
        common = panel.index.intersection(exp.index)
        got = panel.loc[common, cols].to_numpy(dtype=float)
        want = exp.loc[common, cols].to_numpy(dtype=float)
        viol = np.isfinite(got) & ~np.isfinite(want)
        if viol.any():
            i, j = np.argwhere(viol)[0]
            bad.append(f"{int(viol.sum())} celle osservate ma non ancora "
                       f"pubblicate (es. {cols[j]} @ {common[i].date()})")

    # (b) controllo PUNTUALE, indipendente da known_at
    finite = np.argwhere(np.isfinite(panel.loc[:, cols].to_numpy(dtype=float)))
    if len(finite):
        pick = finite[rng.choice(len(finite), size=min(n_sample, len(finite)),
                                 replace=False)]
        for i, j in pick:
            t, sid = panel.index[i], cols[j]
            # una cella trimestrale e' la media di 3 mesi: vale il piu' tardivo
            months = ([t] if kind == "monthly"
                      else list(pd.date_range(t - pd.offsets.MonthEnd(2),
                                              t, freq="ME")))
            for m in months:
                try:
                    rd = release_date(sid, m, metadata=meta, exact=exact)
                except Exception:
                    continue
                if rd is not None and pd.Timestamp(rd) > D:
                    bad.append(f"{sid} @ {m.date()} rilasciata il "
                               f"{pd.Timestamp(rd).date()} > as_of {D.date()}")
                    break
            if bad:
                break
    return bad


#: Il messaggio con cui un modello dichiara che a quella `as_of` non ha un
#: campione utilizzabile.  NON e' un look-ahead ed e' un'altra cosa: vedi
#: `test_oracle`, sezione "LIMITAZIONE DI COPERTURA".
_NO_SAMPLE = "nessuna riga completamente osservata"


def _exercise(model: str, as_of, spec, raw) -> tuple[list[tuple], str | None]:
    """
    Fa costruire al modello i suoi pannelli, poi abortisce.

    Returns
    -------
    (pannelli registrati, messaggio d'errore non-sentinella o None)
    """
    err = None
    with Spy() as spy:
        try:
            if model == "qbvar":
                # DUE pannelli, non uno: la stima e il BORDO.  Il secondo e'
                # arrivato con `qbvar.nowcast` (il meccanismo della settimana
                # 14) e va sorvegliato quanto il primo — e' quello che tocca le
                # righe piu' recenti, quindi quello dove un look-ahead farebbe
                # piu' danno.
                qbvar.nowcast_window(spec, as_of, raw=raw)
                qbvar.fit(spec, as_of=as_of, n_draws=2, burn=2)
            elif model == "cbvar":
                cbvar.fit(spec, as_of=as_of, n_draws=2, burn=2, raw=raw)
            elif model == "bbvar":
                bbvar.fit(spec, as_of=as_of, n_draws=2, burn=2, raw=raw,
                          verbose=False)
            elif model == "lbvar":
                lbvar.fit(spec, as_of=as_of, n_draws=2, burn=2, raw=raw,
                          verbose=False)
        except _Enough:
            pass
        except Exception as e:                       # noqa: BLE001
            err = str(e)
    return spy.panels, err


def test_oracle(dates=("2018-11-16", "2008-06-20")) -> bool:
    """
    L'oracolo, su piu' di una data.

    Due date e non una: il 2018 e' il regime in cui tutti e quattro i modelli
    hanno un campione pieno, il 2008 e' il regime in cui le 7 serie a partenza
    tardiva del profilo `l` non esistono ancora.  Un look-ahead puo' nascondersi
    in uno dei due e non nell'altro — per esempio in un ripiego che scatta solo
    quando il campione e' corto.
    """
    ok = True
    meta, raw = load_metadata(), bdata.load_raw_levels()
    exact = load_exact_releases()
    rng = np.random.default_rng(0)
    limiti: list[str] = []

    for as_of in dates:
        print(f"\n[1] ORACOLO ANTI-LOOK-AHEAD   as_of = {as_of}")
        for model in ("qbvar", "cbvar", "bbvar", "lbvar"):
            spec = BVARSpec.from_config(model[0].upper())
            panels, err = _exercise(model, as_of, spec, raw)

            # I pannelli COSTRUITI PRIMA dell'errore si controllano lo stesso:
            # un modello che esplode dopo aver guardato nel futuro ha comunque
            # guardato nel futuro.
            viol: list[str] = []
            for kind, ao, p in panels:
                viol += [f"[{kind}] {v}"
                         for v in audit_panel(kind, ao, p, meta, raw, rng, exact)]

            if err and _NO_SAMPLE in err:
                limiti.append(f"{model} @ {as_of}")
                ok &= _check(f"{model}: {len(panels)} pannelli, nessun look-ahead",
                             not viol, (viol[0] if viol else
                                        "-> ma NIENTE CAMPIONE, vedi sotto"))
                continue
            if err:
                ok &= _check(f"{model}: gira senza errori", False, err[:70])
                continue
            ok &= _check(f"{model}: {len(panels)} pannelli, nessun look-ahead",
                         bool(panels) and not viol,
                         viol[0] if viol else ("nessun pannello!" if not panels else ""))

    if limiti:
        print("\n    LIMITAZIONE DI COPERTURA (non e' un look-ahead, e' un'altra cosa):")
        for x in limiti:
            print(f"      {x}: nessuna riga completamente osservata")
        print("      Il profilo `l` ha 37 serie, di cui PPIFIS parte 2009-11 e")
        print("      PCEC96 2007-01.  Prima di quelle date `last_full_row` non")
        print("      trova nessuna riga piena e l'L-BVAR non ha campione di")
        print("      stima.  E' una DECISIONE DI MODELLO (quali serie entrano a")
        print("      quale vintage), non un baco del calendario: va portata al")
        print("      relatore prima del blocco 2007-2010.  Il blocco 2016-2019")
        print("      non e' toccato.")

    # ── il controllo negativo: l'oracolo DEVE bocciare il baco
    print("\n    controllo negativo — si ri-introduce il baco di proposito:")
    spec = BVARSpec.from_config("C")
    bugged = qbvar.estimation_panel(spec, raw=raw)        # <- senza as_of
    viol = audit_panel("quarterly", None, bugged, meta, raw, rng, exact)
    ok &= _check("as_of=None viene BOCCIATO", bool(viol),
                 viol[0] if viol else "NON bocciato -> l'oracolo non guarda!")

    # e la stessa cosa con una as_of finta ma sbagliata (dati dal futuro)
    late = qbvar.build_quarterly_panel(spec, "2025-09-30", raw=raw)
    viol = audit_panel("quarterly", pd.Timestamp(dates[-1]), late, meta, raw, rng, exact)
    ok &= _check("un pannello del 2025 spacciato per 2008 viene BOCCIATO",
                 bool(viol), viol[0] if viol else "NON bocciato!")
    return ok


# ─── §2  La riproduzione del dato BEA ─────────────────────────────────────────

def _random_walk_draws(n: int, p: int, k: int, S: int, scale: float = 1e-4):
    """
    (B, Sigma) sintetici: ogni serie e' un random walk indipendente.

    Non e' una stima e non vuole esserlo.  La riproduzione di una cella
    OSSERVATA e' una proprieta' del cablaggio stato-spazio (R = 0 la inchioda al
    suo valore), non dei parametri: se il cablaggio e' giusto la cella torna
    con qualunque legge di moto.  Usare parametri neutri rende il test PIU'
    severo, perche' toglie la possibilita' che una buona stima mascheri un
    indice sbagliato.
    """
    B = np.zeros((S, k, n))
    for j in range(n):
        B[:, 1 + j, j] = 1.0
    Sig = np.repeat((np.eye(n) * scale)[None], S, axis=0)
    return B, Sig


def _bea_truth(spec: BVARSpec, raw: pd.DataFrame) -> pd.Series:
    """La crescita annualizzata del PIL dai livelli grezzi, ai quarter-end."""
    lv = raw[TARGET].dropna()
    lv = lv[lv.index.month.isin(_QM)]
    return 100.0 * ((lv / lv.shift(1)) ** 4 - 1.0)


def _compare(name: str, g: pd.DataFrame, truth: pd.Series, as_of,
             tol: float = 0.03, n_check: int = 3) -> tuple[bool, str]:
    """Confronta la MEDIANA delle estrazioni col dato BEA sui trimestri osservati."""
    D = pd.Timestamp(as_of)
    med = g.median(axis=1)
    idx = pd.DatetimeIndex(med.index).normalize()
    med.index = idx
    # Solo i trimestri il cui PIL era GIA' PUBBLICATO a `as_of`.  Il taglio a
    # due trimestri indietro esclude sia il trimestre obiettivo sia quello
    # precedente, il cui dato potrebbe non essere ancora uscito: su quelli il
    # modello sta facendo il suo mestiere (nowcast), e confrontarli col
    # realizzato misurerebbe l'accuratezza, non la riproduzione.
    cand = [t for t in idx if t in truth.index and t <= D - pd.offsets.QuarterEnd(2)]
    if not cand:
        return False, "nessun trimestre osservato nella finestra"
    cand = cand[-n_check:]
    err = {t: abs(float(med.loc[t]) - float(truth.loc[t])) for t in cand}
    worst = max(err, key=err.get)
    return (err[worst] <= tol,
            f"peggiore {worst.date()}: {err[worst]:.4f} pp "
            f"(modello {float(med.loc[worst]):.3f} vs BEA {float(truth.loc[worst]):.3f})")


def _compare_finestra(res, truth: pd.Series, tol: float = 0.03
                      ) -> tuple[bool, str]:
    """
    Il PRIMO quarter-end della finestra del C-BVAR, cioe' `endEstimT`.

    E' il solo trimestre gia' pubblicato che il modello RICOSTRUISCE invece di
    copiarlo: il livello di `endEstimT` esce dallo smoother (riga 0 della
    finestra), quello di `endEstimT-3` dalla storia osservata.  Senza questo
    controllo il confronto col BEA, dopo il cambio di ancoraggio, non
    toccherebbe piu' lo stato-spazio.
    """
    d = pd.Timestamp(res.index[0]).normalize()
    g = res.growth(TARGET)
    g.index = pd.DatetimeIndex(g.index).normalize()
    if d not in g.index:
        return False, f"{d.date()} non ha una crescita (finestra troppo corta?)"
    if d not in truth.index:
        return False, f"{d.date()} non e' un trimestre BEA"
    med, atteso = float(g.loc[d].median()), float(truth.loc[d])
    return abs(med - atteso) <= tol, (f"{d.date()}: {abs(med - atteso):.4f} pp "
                                      f"(modello {med:.3f} vs BEA {atteso:.3f})")


def test_bea(as_of="2018-11-16") -> bool:
    print(f"\n[2] RIPRODUZIONE DEL DATO BEA   as_of = {as_of}   (ramo di RIUSO)")
    ok = True
    raw = bdata.load_raw_levels()
    rng = np.random.default_rng(7)

    # --- B-BVAR: riuso sul sistema bloccato
    specB = BVARSpec.from_config("B")
    bspec = bbvar.blocked_spec(specB)
    B, Sig = _random_walk_draws(bspec.n, bspec.p, bspec.k, S=3)
    resB = bbvar.fit_reuse(B, Sig, specB, as_of=as_of, raw=raw, rng=rng)
    good, det = _compare("bbvar", resB.growth(TARGET), _bea_truth(specB, raw), as_of)
    ok &= _check("B-BVAR riproduce il PIL osservato entro 0.03 pp", good, det)

    # --- L-BVAR: riuso mensile
    specL = BVARSpec.from_config("L")
    B, Sig = _random_walk_draws(specL.n, specL.p, specL.k, S=3)
    resL = lbvar.fit_reuse(B, Sig, specL, as_of=as_of, raw=raw, rng=rng,
                           verbose=False)
    good, det = _compare("lbvar", resL.growth(TARGET), _bea_truth(specL, raw), as_of)
    ok &= _check("L-BVAR riproduce il PIL osservato entro 0.03 pp", good, det)

    # --- C-BVAR: qui il ramo pieno e' economico (find_mode ~38 s), e serve
    #     perche' la cache del riuso si costruisce solo passando di la'.
    #
    #     DOVE STA LA RIPRODUZIONE, DOPO IL CAMBIO DI ANCORAGGIO.  Da quando la
    #     finestra parte da `endEstimT` (l'ultimo quarter-end osservato, la
    #     convenzione di STEP2_CRBVAR.m r.144-150) i trimestri che `_compare`
    #     guarda — quelli chiusi da almeno due trimestri — stanno nella STORIA
    #     OSSERVATA che `cbvar` antepone alla finestra, esattamente come fanno
    #     gli autori.  Li' il confronto col BEA verifica la catena delle
    #     trasformazioni (media mobile, log, esponenziale, crescita
    #     annualizzata) e torna ESATTO, non piu' entro 0.03 pp.
    #
    #     La riproduzione che passa DAVVERO per lo smoother e' quella del primo
    #     quarter-end della finestra, ed e' un controllo a parte
    #     (`_compare_finestra`): il PIL di `endEstimT` e' pubblicato per
    #     costruzione, ma il modello lo ricostruisce attraverso il processo
    #     mensile latente il cui Sigma_m e' passato dalla proiezione di Higham.
    #
    #     PERCHE' LI' SERVONO 150 ESTRAZIONI E A B/L NE BASTAVANO 3.  Non e' una
    #     tolleranza allentata: e' una proprieta' del modello, misurata.  Nel
    #     B-BVAR la storia prima della finestra e' il dato osservato REPLICATO
    #     fra le estrazioni (`bbvar.m` r.47-51), e nell'L-BVAR le celle osservate
    #     sono inchiodate dal loro valore (R = 0 / nugget): in entrambi la
    #     riproduzione e' esatta ESTRAZIONE PER ESTRAZIONE.  Nel C-BVAR ogni
    #     estrazione sbaglia di ~0.10 pp e solo la MEDIANA si assesta:
    #
    #         S =   8   0.066 / 0.010 pp
    #         S =  40   0.020 / 0.021 pp
    #         S = 150   0.010 / 0.011 pp        <- dentro 0.03, e scende come 1/sqrt(S)
    #
    #     E' un risultato da riportare in tesi, non un dettaglio di test: dice
    #     che il C-BVAR paga la mappa cube-root anche dove non c'e' nulla da
    #     stimare.
    specC = BVARSpec.from_config("C")
    truthC = _bea_truth(specC, raw)
    resC = cbvar.fit(specC, as_of=as_of, n_draws=150, burn=75, raw=raw, rng=rng)
    good, det = _compare("cbvar", resC.growth(TARGET), truthC, as_of)
    ok &= _check("C-BVAR (ramo pieno) riproduce il PIL entro 0.03 pp", good, det)
    good, det = _compare_finestra(resC, truthC)
    ok &= _check("C-BVAR: il primo trimestre della FINESTRA entro 0.03 pp", good, det)

    resC2 = cbvar.fit_reuse(resC.systems, specC, as_of=as_of, raw=raw,
                            rng=np.random.default_rng(7))
    good, det = _compare("cbvar-riuso", resC2.growth(TARGET), truthC, as_of)
    ok &= _check("C-BVAR (ramo di riuso) riproduce il PIL entro 0.03 pp", good, det)
    good, det = _compare_finestra(resC2, truthC)
    ok &= _check("C-BVAR riuso: il primo trimestre della FINESTRA entro 0.03 pp",
                 good, det)
    return ok


# ─── §3  Le tre fasi esistono davvero ─────────────────────────────────────────

def test_fasi(start: str = "2008-01-01", end: str = "2009-12-31") -> bool:
    """
    Il calendario produce TUTTE E TRE le fasi, non solo il nowcast?

    E' un test di CALENDARIO come gli altri due, e come loro non stima niente:
    percorre la griglia e guarda che `horizon_week` esca dall'intervallo 1..13.
    Nasce da un difetto vero, trovato sul pilota: il ciclo chiamava
    `target_quarter`, che restituisce UN obiettivo solo — quello corrente o il
    prossimo — e cosi' `horizon_week` non usciva mai da 1..13.

    Le conseguenze erano due, e nessuna delle due dava errore:

      1. le bande 'forecast' e 'backcast' delle figure restavano VUOTE, cioe'
         mancava proprio il confronto che la §3 esiste per fare;
      2. `compare_nyfed` prende di ogni metodo la riga a `horizon_week` massima:
         13 (meta' trimestre) invece di 17 (l'ultimo venerdi' prima del
         rilascio), quindi la mia stima di meta' trimestre sarebbe finita sulla
         stessa riga del loro backcast maturo.  Quattro settimane di dati in
         piu' da una parte sola, e il numero che ne esce sembra un risultato di
         modello mentre e' un artefatto di calendario.

    Si controlla anche il CONTROLLO NEGATIVO — che con `target_quarter` il test
    fallirebbe davvero — perche' un test verde che non sa fallire non dimostra
    niente.  E' la stessa disciplina dell'oracolo del §1.
    """
    print("\n" + "=" * 82)
    print("§3  LE TRE FASI — forecast (h<1), nowcast (1..13), backcast (h>13)")
    print("=" * 82)

    meta = load_metadata()
    grid = weekly_grid(start, end)
    hs = [horizon_week(D, q)
          for D in grid for q in evaluate.targets(D, metadata=meta)]
    fasi = {"forecast": [h for h in hs if h < 1],
            "nowcast": [h for h in hs if 1 <= h <= 13],
            "backcast": [h for h in hs if h > 13]}

    ok = True
    for nome, v in fasi.items():
        ok &= _check(f"la fase '{nome}' e' popolata", bool(v),
                     f"{len(v)} righe" if v else "VUOTA")
    ok &= _check("l'orizzonte massimo e' 17 (ultimo venerdi' pre-rilascio)",
                 max(hs) == 17, f"max = {max(hs):+d}")
    ok &= _check("l'orizzonte minimo e' negativo (esiste il forecast)",
                 min(hs) < 0, f"min = {min(hs):+d}")

    # Controllo negativo: col vecchio `target_quarter` le fasi si riducono a una.
    hs_old = [horizon_week(D, evaluate.target_quarter(D)) for D in grid]
    degenere = min(hs_old) >= 1 and max(hs_old) <= 13
    ok &= _check("controllo negativo: un obiettivo solo -> una fase sola",
                 degenere, f"con target_quarter: h in [{min(hs_old):+d}, "
                           f"{max(hs_old):+d}]")
    return ok


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> bool:
    print("=" * 82)
    print("GATE 6 — I TEST DI CALENDARIO")
    print("=" * 82)
    ok = True
    ok &= test_oracle()
    ok &= test_bea()
    ok &= test_fasi()
    print("\n" + "=" * 82)
    print("TUTTO OK — si puo' lanciare la passata" if ok
          else "QUALCOSA NON TORNA — NON lanciare la passata")
    print("=" * 82)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
