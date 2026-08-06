"""
src/em/selftest_fixture.py

Fixture UNICA per i self-test dei moduli EM (i blocchi ``__main__`` di
em_initialization, em_e_step, em_m_step, em_main).

PERCHE' ESISTE
--------------
Un solo ingresso parametrico sulle tre spec di `config/factor_specs.json`
(`fed_overlap`, `diag4`, `diag3`) applicate al dataset `final` (37 serie).
Nessun nome di serie, nome di fattore o valore di `r` e' cablato qui: tutto
viene dai config. E' l'unico punto del percorso di test che conosce i percorsi
su disco.

USO (dentro un blocco ``__main__``)
-----------------------------------
    from em.selftest_fixture import parse_spec_args, load_fixture

    args = parse_spec_args("em_m_step self-test")
    fx   = load_fixture(args.spec)

    fx.Y            # pannello standardizzato, np.ndarray (T, M), NaN preservati
    fx.Y_filled     # pannello riempito (MM + gaussian fill), per PCA/init
    fx.structure    # FactorStructure (mask M x r)
    fx.freq_list    # ['monthly'|'quarterly'] x M
    fx.ordered_cols # nomi colonna, nell'ordine
    fx.theta0       # theta^(0) da PCA mask-driven
    fx.r, fx.T, fx.M
    fx.target       # nome della serie target (l'ultima colonna del config)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASET = os.path.join(_PROJECT_ROOT, "data", "processed", "final", "dataset_final.csv")
_SERIES_CONFIG = os.path.join(_PROJECT_ROOT, "config", "series_final.json")

SPEC_CHOICES = ("fed_overlap", "diag4", "diag3")
DEFAULT_SPEC = "fed_overlap"


@dataclass
class Fixture:
    """Tutto cio' che serve a un self-test EM, coerente per costruzione."""
    spec_name: str
    df: pd.DataFrame                 # pannello trasformato, non standardizzato
    Y_std_df: pd.DataFrame           # standardizzato (NaN preservati)
    Y_filled: pd.DataFrame           # standardizzato + MM fill + gaussian fill
    mean: pd.Series
    std: pd.Series
    structure: object                # FactorStructure
    freq_list: list[str]
    ordered_cols: list[str]
    theta0: dict
    F0: np.ndarray
    target: str
    _pca_info: dict = field(default_factory=dict)

    @property
    def Y(self) -> np.ndarray:
        return self.Y_std_df.to_numpy()

    @property
    def T(self) -> int:
        return self.Y_std_df.shape[0]

    @property
    def M(self) -> int:
        return self.Y_std_df.shape[1]

    @property
    def r(self) -> int:
        return int(np.asarray(self.theta0["A"]).shape[0])

    @property
    def quarterly_cols(self) -> list[str]:
        return [c for c, f in zip(self.ordered_cols, self.freq_list)
                if f == "quarterly"]

    def col(self, name: str) -> int:
        """Indice di colonna di una serie."""
        return self.ordered_cols.index(name)

    def factors_of(self, name: str) -> np.ndarray:
        """Colonne-fattore su cui carica la serie `name` (dalla mask)."""
        return self.structure.factors_of_series(self.col(name))

    def describe(self) -> str:
        fs = self.structure
        return (f"fixture: dataset 'final' | spec '{self.spec_name}' "
                f"(r={fs.r}, fattori={fs.factor_names}, "
                f"{'diagonale' if fs.diagonal else 'non-diagonale'}) | "
                f"T x M = {self.T} x {self.M} | target={self.target}")


def selftest_scratch(tag: str = "em") -> str:
    """
    Cartella TEMPORANEA per gli output dei self-test, cancellata all'uscita.

    I self-test scrivono file (cache di fit_dfm, theta^(0), round-trip su
    disco): se li scrivessero in `data/processed/final/<spec>/` o in
    `output/`, si mescolerebbero agli artefatti VERI prodotti da
    `run_final_artifacts.py`. E' gia' successo: una run di self-test con
    --max-iter 3 ha sovrascritto il fit a 83 iterazioni di fed_overlap, e da
    fuori i due file sono indistinguibili.

    Regola: i self-test verificano, non producono. Tutto cio' che scrivono
    vive qui e sparisce quando il processo termina.
    """
    import atexit  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    d = tempfile.mkdtemp(prefix=f"selftest_{tag}_")
    atexit.register(lambda: shutil.rmtree(d, ignore_errors=True))
    return d


def parse_spec_args(description: str) -> argparse.Namespace:
    """Parser comune ai self-test: `--spec {fed_overlap,diag4,diag3}`."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--spec", choices=SPEC_CHOICES, default=DEFAULT_SPEC,
                    help=f"struttura di fattore da config/factor_specs.json "
                         f"(default: {DEFAULT_SPEC})")
    return ap.parse_args()


def load_fixture(spec_name: str = DEFAULT_SPEC, *, seed: int = 42) -> Fixture:
    """
    Costruisce la fixture: pannello `final` + struttura `spec_name` + theta^(0).

    Deterministica dato `seed` (il gaussian fill del bordo frastagliato e'
    l'unica sorgente di casualita').
    """
    # import locali: evitano cicli quando la fixture e' importata da un modulo EM
    from em.em_initialization import (  # noqa: PLC0415
        standardize, mm_fill_quarterly, gaussian_fill_ragged,
        pca_initialization, compute_theta_initial,
    )
    from em.factor_structure import build_loading_mask  # noqa: PLC0415

    if not os.path.isfile(_DATASET):
        raise FileNotFoundError(
            f"Dataset 'final' non trovato: {_DATASET}\n"
            f"Costruiscilo con:  python src/data_loader_final.py"
        )

    df = pd.read_csv(_DATASET, index_col=0, parse_dates=True)
    with open(_SERIES_CONFIG, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    freq_of = {s["series_id"]: ("monthly" if s["freq"] == "M" else "quarterly")
               for s in cfg["series"]}
    ordered_cols = list(df.columns)
    freq_list = [freq_of[c] for c in ordered_cols]
    structure = build_loading_mask(spec_name, ordered_cols)

    Y_std_df, mean, std = standardize(df)
    Y_mm = Y_std_df.copy()
    for c, fr in zip(ordered_cols, freq_list):
        if fr == "quarterly":
            Y_mm[c] = mm_fill_quarterly(Y_std_df[c])
    Y_filled = gaussian_fill_ragged(Y_mm, random_state=seed)

    F0, pca_info = pca_initialization(Y_filled, structure)
    theta0 = compute_theta_initial(Y_filled, F0, structure)

    return Fixture(
        spec_name=spec_name, df=df, Y_std_df=Y_std_df, Y_filled=Y_filled,
        mean=mean, std=std, structure=structure, freq_list=freq_list,
        ordered_cols=ordered_cols, theta0=theta0, F0=np.asarray(F0),
        target=ordered_cols[-1],           # per contratto il target e' l'ultima
        _pca_info=pca_info if isinstance(pca_info, dict) else {},
    )


if __name__ == "__main__":
    _a = parse_spec_args("self-test della fixture stessa")
    _fx = load_fixture(_a.spec)
    print(_fx.describe())
    assert _fx.Y.shape == (_fx.T, _fx.M)
    assert len(_fx.freq_list) == _fx.M == _fx.structure.M
    assert _fx.structure.mask.sum(axis=1).min() >= 1, "riga di mask vuota"
    assert np.asarray(_fx.theta0["Lambda"]).shape == (_fx.M, _fx.r)
    assert not np.isnan(_fx.Y_filled.to_numpy()).any(), "Y_filled contiene NaN"
    assert _fx.quarterly_cols, "nessuna serie trimestrale nel pannello"
    print(f"  trimestrali : {_fx.quarterly_cols}")
    print(f"  target '{_fx.target}' carica sui fattori "
          f"{[_fx.structure.factor_names[j] for j in _fx.factors_of(_fx.target)]}")
    print("fixture OK.")
