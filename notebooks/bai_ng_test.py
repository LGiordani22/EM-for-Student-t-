"""
Test di Bai e Ng (2005) su asimmetria, curtosi e normalita' per le 37 serie
del pannello `final`, piu' la tabella LaTeX dell'appendice.

    python notebooks/bai_ng_test.py

Scrive in docs/tesi/tables/:
    tab_fat_tails.tex

RIFERIMENTO
-----------
Bai, J. & Ng, S. (2005), "Tests for Skewness, Kurtosis, and Normality for
Time Series Data", Journal of Business & Economic Statistics 23(1), 49-60.

Le formule sono i Teoremi 2 e 3 e la Sezione 2.4 del paper. La logica: la
curtosi campionaria da sola non e' un test, perche' sotto dipendenza seriale
la sua varianza asintotica NON e' 24/T; serve la varianza di lungo periodo
dei momenti. Bai e Ng la costruiscono col metodo delta, tenendo conto che
media e varianza sono stimate.

    ASIMMETRIA (Teorema 2), sotto H0: mu_3 = 0

        pi_3 = sqrt(T) * mu_3^ / sqrt(alpha' Gamma alpha)
        alpha = [1, -3 sigma^2]
        Z_t   = [ (x_t - xbar)^3 , (x_t - xbar) ]

    CURTOSI (Teorema 3), valutata in kappa = 3

        pi_4 = sqrt(T) * (mu_4^ - 3 sigma^4) / sqrt(beta' Omega beta)
        beta  = [1, -4 mu_3, -2 sigma^2 kappa]  -> [1, 0, -6 sigma^2]
        W_t   = [ (x-xbar)^4 - 3 sigma^4 , (x-xbar) , (x-xbar)^2 - sigma^2 ]

    NORMALITA' (Sezione 2.4)

        pi_34 = pi_3^2 + pi_4^2  ~  chi^2(2)

    perche' pi_3 e pi_4 sono asintoticamente indipendenti sotto normalita'
    anche con dati dipendenti.

Il null e' imposto ovunque (mu_3 = 0, mu_4 = 3 sigma^4, kappa = 3), come nella
Sezione 2.4: e' li' che gli autori scrivono esplicitamente il vettore da usare
per la covarianza di lungo periodo.

Gamma e Omega sono stimate col kernel di Newey-West senza prewhitening, che e'
quello usato nelle simulazioni del paper (Tabelle 1, 2 e 4). L'ampiezza di
banda segue la regola pratica m = floor(4 (T/100)^(2/9)).

AVVERTENZA DEGLI AUTORI (p. 49): le distorsioni di size rendono il test sulla
sola curtosi poco affidabile, salvo contro alternative a code sottili, perche'
(x - mu)^4 e' fortemente asimmetrico e converge lentamente. pi_4 sta in tabella
per completezza; l'evidenza si legge su pi_34, che nelle loro simulazioni ha
size e potenza buone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "final" / "dataset_final.csv"
METADATA = ROOT / "data" / "processed" / "final" / "metadata_final.csv"
OUT = ROOT / "docs" / "tesi" / "tables"

NL = "\n"


# ==========================================================================
# VARIANZA DI LUNGO PERIODO
# ==========================================================================

def banda_nw(n_oss: int) -> int:
    """Regola pratica di Newey-West: m = floor(4 (T/100)^(2/9))."""

    return int(
        np.floor(
            4.0 * (n_oss / 100.0) ** (2.0 / 9.0)
        )
    )


def lrv(z: np.ndarray, m: int) -> np.ndarray:
    """
    Covarianza di lungo periodo di `z` (T x k) col kernel di Bartlett.

        Gamma = C_0 + sum_{j=1}^{m} w_j (C_j + C_j')
        w_j   = 1 - j / (m + 1)

    NON si demeana. Le componenti di Z_t sono gia' centrate sui valori che
    la nulla impone (Sezione 2.4 del paper): (x-xbar) e (x-xbar)^2-sigma^2
    hanno media campionaria esattamente zero, mentre (x-xbar)^3 e
    (x-xbar)^4-3sigma^4 hanno per media proprio le quantita' sotto test.
    Sottrarre quella media toglierebbe dalla varianza il segnale che si sta
    testando, e renderebbe il test conservativo.
    """

    z = np.atleast_2d(z)

    if z.ndim == 1:
        z = z[:, None]

    n_oss = z.shape[0]
    zc = z

    gamma = (zc.T @ zc) / n_oss

    for j in range(1, m + 1):

        c_j = (zc[j:].T @ zc[:-j]) / n_oss
        w_j = 1.0 - j / (m + 1.0)

        gamma = gamma + w_j * (c_j + c_j.T)

    return gamma


# ==========================================================================
# I TRE TEST
# ==========================================================================

def bai_ng(x: np.ndarray) -> dict:
    """
    Restituisce momenti campionari e statistiche pi_3, pi_4, pi_34.

    `x` deve essere gia' privo di NaN e ordinato nel tempo.
    """

    x = np.asarray(x, dtype=float)
    n_oss = x.size

    e = x - x.mean()

    sigma2 = float((e ** 2).mean())
    sigma = float(np.sqrt(sigma2))

    mu3 = float((e ** 3).mean())
    mu4 = float((e ** 4).mean())

    skew = mu3 / sigma ** 3
    kurt = mu4 / sigma ** 4

    m = banda_nw(n_oss)

    # --- la matrice 4x4 della Sezione 2.4 ---------------------------------
    #
    #     Z_t = [ (x-xbar), (x-xbar)^2 - sigma^2,
    #             (x-xbar)^3, (x-xbar)^4 - 3 sigma^4 ]
    #
    # e' il vettore che gli autori indicano esplicitamente per il test di
    # normalita'. Le due condizioni di momento si ottengono da Z_t col
    # metodo delta, tenendo conto che media e varianza sono stimate:
    #
    #     sqrt(T) mu_3^          = A[0] . sum Z_t / sqrt(T) + o_p(1)
    #     sqrt(T)(mu_4^ - 3 s^4) = A[1] . sum Z_t / sqrt(T) + o_p(1)
    #
    # Le righe di A sono i gradienti del Teorema 2 e del Teorema 3 col null
    # imposto (mu_3 = 0, kappa = 3), riscritti sulle quattro componenti.
    z = np.column_stack([
        e,
        e ** 2 - sigma2,
        e ** 3,
        e ** 4 - 3.0 * sigma2 ** 2,
    ])

    A = np.array([
        [-3.0 * sigma2, 0.0, 1.0, 0.0],
        [0.0, -6.0 * sigma2, 0.0, 1.0],
    ])

    V = A @ lrv(z, m) @ A.T

    # Y = sqrt(T) * (mu_3^, mu_4^ - 3 sigma^4)
    y = np.sqrt(n_oss) * np.array([mu3, mu4 - 3.0 * sigma2 ** 2])

    # --- asimmetria (Teorema 2) e curtosi (Teorema 3): le diagonali -------
    pi3 = y[0] / np.sqrt(V[0, 0])
    pi4 = y[1] / np.sqrt(V[1, 1])

    # --- normalita' (Sezione 2.4): la FORMA QUADRATICA, non pi3^2 + pi4^2 -
    #
    # Il paper presenta anche pi_3^2 + pi_4^2 come generalizzazione diretta
    # del Jarque-Bera, ed e' asintoticamente equivalente perche' sotto
    # normalita' le due statistiche sono indipendenti. Ma la statistica che
    # gli autori riportano nella loro Tabella 5 e' Y'(A Phi A')^{-1} Y, che
    # in campioni finiti differisce (li' pi_34 non e' mai la somma dei
    # quadrati: per il tasso di disoccupazione danno .913, 1.323 e 4.853).
    # E' anche l'unica che giustifica la frase dell'abstract secondo cui il
    # test congiunto richiede una covarianza di lungo periodo a 4 dimensioni.
    pi34 = float(y @ np.linalg.solve(V, y))

    # --- Jarque-Bera, per confronto ---------------------------------------
    #     E' il caso iid: usa la varianza TEORICA gaussiana (6/T e 24/T)
    #     invece di stimarla. Sotto dipendenza seriale sovra-rifiuta - e'
    #     esattamente il motivo per cui Bai e Ng hanno scritto il paper -
    #     ma dove l'autocorrelazione e' trascurabile e' un test valido.
    jb = n_oss * (skew ** 2 / 6.0 + (kurt - 3.0) ** 2 / 24.0)

    # Autocorrelazione di primo ordine: serve a dire, serie per serie,
    # quanto l'ipotesi iid di JB sia plausibile.
    ac1 = float(np.corrcoef(e[1:], e[:-1])[0, 1])

    return {
        "T": n_oss,
        "banda": m,
        "skew": skew,
        "kurt": kurt,
        "pi3": pi3,
        "pi4": pi4,
        "pi34": pi34,
        "jb": jb,
        "ac1": ac1,
        "p3": 2.0 * (1.0 - stats.norm.cdf(abs(pi3))),
        "p4": 2.0 * (1.0 - stats.norm.cdf(abs(pi4))),
        "p34": 1.0 - stats.chi2.cdf(pi34, df=2),
        "pjb": 1.0 - stats.chi2.cdf(jb, df=2),
    }


# ==========================================================================
# GAUSSIANA CONTRO t: STIMA ML E TEST LR
# ==========================================================================
#
# I test di Bai e Ng dicono se i momenti sono compatibili con la normalita'.
# Non dicono quale distribuzione descriva meglio i dati. Qui si stimano per
# massima verosimiglianza due modelli annidati sulle STESSE osservazioni,
#
#     y_t = c + sum_i phi_i y_{t-i} + sigma e_t
#
# con e_t gaussiano oppure e_t ~ t_nu, e si confrontano con un LR.
#
# PERCHE' UN AR E NON I LIVELLI. La verosimiglianza t su y_t direttamente
# attribuirebbe alla coda tutta la dipendenza seriale: una serie gaussiana
# ma persistente ha una distribuzione marginale con code apparentemente
# grasse. Filtrando con un AR, nu misura la coda delle INNOVAZIONI, che e'
# l'oggetto di cui parla la tesi. L'ordine p e' scelto col BIC sul modello
# gaussiano, fra 0 e 4, e poi tenuto fisso per entrambi: il LR confronta due
# modelli che differiscono SOLO per la distribuzione.
#
# IL BORDO. Sotto H0 la gaussiana e' il limite nu -> infinito, cioe'
# eta = 1/nu = 0, che sta sul bordo dello spazio parametrico. Il LR non e'
# chi^2(1): per Chernoff (1954) e' la mistura 0.5 chi^2(0) + 0.5 chi^2(1),
# quindi il p-value e' meta' di quello chi^2(1). Usare chi^2(1) e' il modo
# standard di sovrastimare la significativita' di questo test.

def _ar_matrici(x: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Regressori dell'AR(p) con intercetta, sulle sole righe complete."""

    n_oss = x.size
    y = x[p:]

    reg = [np.ones(n_oss - p)]

    for i in range(1, p + 1):
        reg.append(x[p - i:n_oss - i])

    return np.column_stack(reg), y


def _ll_gauss(x: np.ndarray, p: int) -> tuple[float, int]:
    """Log-verosimiglianza dell'AR(p) gaussiano stimato per OLS/ML."""

    X, y = _ar_matrici(x, p)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta

    n_oss = res.size
    s2 = float((res ** 2).mean())

    ll = -0.5 * n_oss * (np.log(2.0 * np.pi * s2) + 1.0)

    return ll, n_oss


def _ll_t(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Meno la log-verosimiglianza dell'AR(p) con innovazioni t_nu."""

    k = X.shape[1]

    beta = theta[:k]
    sigma = np.exp(theta[k])
    nu = 2.0 + np.exp(theta[k + 1])

    res = (y - X @ beta) / sigma

    ll = (
        stats.t.logpdf(res, df=nu).sum()
        - res.size * np.log(sigma)
    )

    return -ll


def gauss_vs_t(x: np.ndarray, p_max: int = 4) -> dict:
    """
    Stima AR(p)-gaussiano e AR(p)-t sulle stesse osservazioni e le confronta.

    Restituisce l'ordine scelto, nu stimato, la statistica LR e il p-value
    con la correzione di bordo.
    """

    # --- ordine col BIC sul modello gaussiano ------------------------------
    bic = {}

    for p in range(0, p_max + 1):

        ll, n_oss = _ll_gauss(x, p)
        n_par = p + 2                      # intercetta, p coefficienti, sigma

        bic[p] = -2.0 * ll + n_par * np.log(n_oss)

    p = min(bic, key=bic.get)

    # --- i due modelli, sullo stesso campione effettivo --------------------
    ll_g, n_oss = _ll_gauss(x, p)

    X, y = _ar_matrici(x, p)

    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    s0 = float(np.sqrt(((y - X @ beta0) ** 2).mean()))

    theta0 = np.concatenate([beta0, [np.log(s0)], [np.log(6.0)]])

    opt = minimize(
        _ll_t,
        theta0,
        args=(X, y),
        method="Nelder-Mead",
        options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-8,
                 "fatol": 1e-8},
    )

    ll_t = -float(opt.fun)
    nu = 2.0 + float(np.exp(opt.x[-1]))

    # Il LR non puo' essere negativo: la gaussiana e' un caso limite della t.
    # Se l'ottimizzatore non ha raggiunto almeno la verosimiglianza gaussiana
    # il confronto e' inaffidabile e va segnalato, non arrotondato a zero.
    lr = 2.0 * (ll_t - ll_g)
    ok = lr > -1e-6

    return {
        "p_ar": p,
        "nu": nu,
        "lr": max(lr, 0.0),
        "p_lr": 0.5 * (1.0 - stats.chi2.cdf(max(lr, 0.0), df=1)),
        "lr_ok": ok,
        "T_ar": n_oss,
    }


# ==========================================================================
# TABELLA
# ==========================================================================

def stelle(p: float) -> str:
    """Convenzione della tesi: *** 1%, ** 5%, * 10%."""

    if p < 0.01:
        return r"$^{***}$"

    if p < 0.05:
        return r"$^{**}$"

    if p < 0.10:
        return r"$^{*}$"

    return ""


# Nomi accorciati SOLO per questa tabella. Il float e' alto 470pt e ogni
# nome che va a capo ne costa ~10: con i nomi per esteso le note non ci
# stanno e LaTeX le butta via ("Float too large for page"). Il nome completo
# resta nella Tabella A.1, a cui questa e' agganciata dal codice della serie.
NOMI_CORTI = {
    "ADP nonfarm private payroll (proxy)": "ADP nonfarm private payroll",
    "Core CPI: all items less food & energy": "Core CPI: less food \\& energy",
    "Empire State Mfg: gen. bus. cond.": "Empire State Mfg survey",
    "Merchant wholesalers: inventories: total": "Wholesalers: inventories",
    "Philly Fed Mfg: current activity": "Philly Fed Mfg survey",
    "Real personal consumption expenditures": "Real consumption",
    "ISM nonmanufacturing: NMI composite": "ISM nonmfg: NMI composite",
    "Value of construction put in place": "Construction put in place",
    "Nonfarm business: unit labor cost": "Nonfarm business: unit labor",
    "Real disposable personal income": "Real disposable income",
    "Mfrs unfilled orders: all mfg": "Mfrs unfilled orders",
    "Mfrs inventories: durable goods": "Mfrs inventories: durables",
    "Mfrs shipments: durable goods": "Mfrs shipments: durables",
    "Mfrs new orders: durable goods": "Mfrs new orders: durables",
    "New single family houses sold": "New single family homes sold",
    "Civilian unemployment rate": "Unemployment rate",
    "All employees: total nonfarm": "Total nonfarm employment",
    "Industrial production index": "Industrial production",
}


def esc(s: str) -> str:

    return (
        str(s)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def tabella(ris: pd.DataFrame) -> str:

    testa = [
        # Prima colonna a larghezza fissa: con `l` la tabella misura 521pt
        # contro i 469.75pt del blocco di testo (lscape ruota la pagina ma
        # NON allarga \textwidth), e finisce tagliata.
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.6cm}"
        r" l r r r r r r r r r@{}}",
        r"\toprule",
        r"& & & & \multicolumn{2}{c}{$\hat\kappa$} & &"
        r" \multicolumn{2}{c}{Bai--Ng} &"
        r" \multicolumn{2}{c}{Gaussian vs.\ $t$} \\",
        r"\cmidrule(lr){5-6} \cmidrule(lr){8-9} \cmidrule(lr){10-11}",
        r"Data series & Code & $T$ & $\hat\tau$ & all & excl.\ 2020 &"
        r" JB & $\pi_3$ & $\pi_{34}$ & $\hat\nu$ & LR \\",
        r"\midrule",
    ]

    righe = []

    for _, r in ris.iterrows():

        jb = (
            f"{r['jb']:,.0f}".replace(",", r"\,")
            if r["jb"] < 1e6
            else f"{r['jb'] / 1e6:.1f}M"
        )

        # Due casi vanno dichiarati invece che stampati come stime precise:
        # nu grandissimo (coda indistinguibile dalla gaussiana) e nu sul
        # bordo inferiore, dove la varianza della t stimata non e' finita.
        if r["nu"] > 100:
            nu = r"$>$100"
        elif r["nu"] <= 2.05:
            nu = r"2.0$^{\dagger}$"
        else:
            nu = f"{r['nu']:.1f}"

        nome = NOMI_CORTI.get(r["name"], r["name"])

        # I nomi accorciati contengono gia' l'escape di '&': non ri-escapare.
        nome = nome if nome != r["name"] else esc(nome)

        righe.append(
            f"{nome} & \\texttt{{\\scriptsize {esc(r['code'])}}} & "
            f"{int(r['T'])} & "
            f"{r['skew']:.2f} & {r['kurt']:.1f} & {r['kurt_ex']:.1f} & "
            f"{jb}{stelle(r['pjb'])} & "
            f"{r['pi3']:.2f}{stelle(r['p3'])} & "
            f"{r['pi34']:.1f}{stelle(r['p34'])} & "
            f"{nu} & "
            f"{r['lr']:.1f}{stelle(r['p_lr'])} \\\\"
        )

    return (
        NL.join(testa) + NL
        + NL.join(righe) + NL
        + r"\bottomrule" + NL
        + r"\end{tabular}" + NL
    )


# ==========================================================================
# MAIN
# ==========================================================================

def main() -> None:

    df = pd.read_csv(DATASET, index_col=0, parse_dates=True)
    meta = pd.read_csv(METADATA)

    nome = dict(zip(meta["series_id"], meta["paper_name"]))
    freq = dict(zip(meta["series_id"], meta["freq"]))

    out = []

    for code in df.columns:

        s = df[code].dropna()

        # Buchi interni: le trimestrali sono osservate 1 mese su 3 (regolare,
        # e' la loro frequenza), ma un buco IRREGOLARE romperebbe la struttura
        # di dipendenza su cui poggia la varianza di lungo periodo.
        passo = s.index.to_series().diff().dt.days.dropna()
        atteso = passo.median()
        irregolare = int((passo > atteso * 1.6).sum())

        r = bai_ng(s.to_numpy())

        # Stesso calcolo senza il 2020: serve a dire se le code grasse sono
        # solo la pandemia. (Non e' un secondo test, e' un controllo: la
        # colonna riportata e' la sola curtosi.)
        s_ex = s[(s.index < "2020-01-01") | (s.index > "2020-12-31")]
        r_ex = bai_ng(s_ex.to_numpy())

        r["kurt_ex"] = r_ex["kurt"]
        r["p34_ex"] = r_ex["p34"]

        r.update(gauss_vs_t(s.to_numpy()))

        r["code"] = code
        r["name"] = nome.get(code, code)
        r["freq"] = freq.get(code, "?")
        r["start"] = s.index.min().date()
        r["gap"] = irregolare

        out.append(r)

    ris = pd.DataFrame(out)

    # Ordine: come nel dataset (= come le altre tabelle della tesi).
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_fat_tails.tex").write_text(tabella(ris), encoding="utf-8")

    # ---------------------------------------------------------------- log
    n = len(ris)

    print(f"serie                       : {n}")
    print(f"T mediano                   : {int(ris['T'].median())}")
    print(f"banda NW (min-max)          : "
          f"{int(ris['banda'].min())}-{int(ris['banda'].max())}")
    print(f"serie con buchi irregolari  : {int((ris['gap'] > 0).sum())}")
    print()
    print(f"curtosi mediana             : {ris['kurt'].median():.2f}"
          f"   (senza 2020: {ris['kurt_ex'].median():.2f})")
    print(f"curtosi > 3                 : {int((ris['kurt'] > 3).sum())}/{n}"
          f"   (senza 2020: {int((ris['kurt_ex'] > 3).sum())}/{n})")
    print(f"curtosi > 5                 : {int((ris['kurt'] > 5).sum())}/{n}"
          f"   (senza 2020: {int((ris['kurt_ex'] > 5).sum())}/{n})")
    print(f"curtosi min / max           : {ris['kurt'].min():.1f}"
          f" ({ris.loc[ris['kurt'].idxmin(), 'code']})"
          f"  /  {ris['kurt'].max():.1f}"
          f" ({ris.loc[ris['kurt'].idxmax(), 'code']})")
    print()
    print(f"JB rifiuta al 5%            : "
          f"{int((ris['pjb'] < 0.05).sum())}/{n}"
          f"   (al 1%: {int((ris['pjb'] < 0.01).sum())}/{n})")
    print(f"pi_34 rifiuta al 5%         : "
          f"{int((ris['p34'] < 0.05).sum())}/{n}"
          f"   (al 1%: {int((ris['p34'] < 0.01).sum())}/{n})")
    print(f"pi_3 (simmetria) al 5%      : "
          f"{int((ris['p3'] < 0.05).sum())}/{n}")
    print(f"pi_4 (curtosi) al 5%        : "
          f"{int((ris['p4'] < 0.05).sum())}/{n}")
    print()
    print(f"|ac1| > 0.2                 : "
          f"{int((ris['ac1'].abs() > 0.2).sum())}/{n}"
          f"   (mediana {ris['ac1'].median():.2f})")
    print()
    print("serie che rifiutano la normalita' con pi_34 al 5%:")

    for _, r in ris[ris["p34"] < 0.05].sort_values("p34").iterrows():
        print(f"    {r['code']:<18} kurt={r['kurt']:6.2f}  "
              f"pi34={r['pi34']:6.2f}  p={r['p34']:.3f}")

    print()
    print("serie che rifiutano la simmetria al 5%:")

    for _, r in ris[ris["p3"] < 0.05].iterrows():
        print(f"    {r['code']:<18} skew={r['skew']:6.2f}  "
              f"pi3={r['pi3']:6.2f}  p={r['p3']:.3f}")

    print()
    print("--- gaussiana contro t, su innovazioni AR(p) ---")
    print(f"LR rifiuta la gaussiana al 5% : "
          f"{int((ris['p_lr'] < 0.05).sum())}/{n}"
          f"   (al 1%: {int((ris['p_lr'] < 0.01).sum())}/{n})")
    print(f"nu mediano                    : {ris['nu'].median():.1f}")
    print(f"nu < 10                       : {int((ris['nu'] < 10).sum())}/{n}")
    print(f"nu > 100 (indistinguibile)    : "
          f"{int((ris['nu'] > 100).sum())}/{n}")
    print(f"ordine AR scelto (BIC)        : "
          f"{dict(ris['p_ar'].value_counts().sort_index())}")
    print(f"ottimizzazioni sospette       : "
          f"{int((~ris['lr_ok']).sum())}")

    print()
    print(f"scritto in {OUT / 'tab_fat_tails.tex'}")


if __name__ == "__main__":
    main()
