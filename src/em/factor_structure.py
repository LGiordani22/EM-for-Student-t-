"""
src/em/factor_structure.py

La LOADING MASK.

Rappresenta la struttura di caricamento come una matrice mask M x r di 0/1
derivata da `config/factor_specs.json`. Le tre strutture di fattore
(`fed_overlap`, `diag4`, `diag3`) sono TRE MASK diverse dello stesso oggetto:

  * `mask[i, j] = 1`  <=>  la serie i puo' caricare sul fattore j.

"Diagonale vs non-diagonale" e' solo QUANTE entrate ha una riga:
  * diag3 / diag4  -> esattamente 1 entrata per riga (Lambda diagonale a blocchi),
  * fed_overlap    -> 1 o 2 entrate per riga (globale + locale; 1 per i prezzi).

Questo modulo NON conosce la matematica dell'EM: fornisce solo la mask, i nomi dei
fattori, il flag `diagonal`, e (per l'init A2) i membri di ciascun fattore e
l'ordine di estrazione (dal fattore piu' largo al piu' stretto). L'unico punto del
codice che conosce i nomi dei fattori.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FACTOR_SPECS_PATH = os.path.join(_PROJECT_ROOT, "config", "factor_specs.json")


@dataclass
class FactorStructure:
    """
    Struttura di fattore risolta per un dato ordine di colonne (serie).

    Attributes
    ----------
    spec_name : str
        Nome della spec (es. "fed_overlap", "diag4", "diag3").
    factor_names : list[str]
        Nomi dei fattori nell'ordine delle colonne di `mask` (es. ["G","S","R","L"]).
    mask : np.ndarray, shape (M, r), dtype int
        mask[i, j] = 1 se la serie i carica sul fattore j.
    diagonal : bool
        True se ogni riga ha esattamente una entrata (Lambda diagonale a blocchi).
    ordered_cols : list[str]
        L'ordine delle serie (righe di mask), = colonne del dataset.
    """
    spec_name: str
    factor_names: list[str]
    mask: np.ndarray
    diagonal: bool
    ordered_cols: list[str]

    @property
    def M(self) -> int:
        return self.mask.shape[0]

    @property
    def r(self) -> int:
        return self.mask.shape[1]

    def members(self, j: int) -> np.ndarray:
        """Indici (di riga) delle serie che caricano sul fattore j."""
        return np.where(self.mask[:, j] == 1)[0]

    def factors_of_series(self, i: int) -> np.ndarray:
        """Colonne (fattori) su cui carica la serie i."""
        return np.where(self.mask[i, :] == 1)[0]

    def factor_col(self, name: str) -> int:
        """Colonna del fattore dal nome."""
        return self.factor_names.index(name)

    def display_block_map(self) -> dict[str, str]:
        """
        Mappa serie -> UN nome di blocco, per raggruppare le righe nei report.

        E' l'inverso approssimato di `from_block_map`, e serve ai moduli di
        Monte Carlo che stampano tabelle raggruppate per blocco. Sotto una spec
        NON diagonale una serie carica su piu' fattori, ma una tabella ha
        bisogno di una riga per serie: si sceglie il fattore **piu' specifico**,
        cioe' quello con MENO membri, a parita' il primo per indice di colonna.
        E' una scelta di presentazione e basta — nessun calcolo deve passare di
        qui: chi fa algebra usa `mask`, che non perde informazione.

        Il criterio "meno membri" non e' arbitrario. Prendere il primo fattore
        in ordine di colonna sarebbe degenere proprio dove serve: sotto una
        spec con un fattore GLOBALE su cui caricano tutte le serie, il globale
        e' il primo per tutte, e la mappa collasserebbe l'intero pannello in un
        unico gruppo — una tabella con un solo blocco, che non raggruppa nulla.
        Il fattore piu' stretto e' invece quello che distingue le serie fra
        loro, ed e' l'informazione che una tabella raggruppata deve mostrare.

        Le serie senza alcun caricamento (che `build_loading_mask` gia' vieta)
        finirebbero in "?" invece di sparire silenziosamente.
        """
        sizes = self.mask.sum(axis=0)          # membri per fattore
        out: dict[str, str] = {}
        for i, c in enumerate(self.ordered_cols):
            js = self.factors_of_series(i)
            if not js.size:
                out[c] = "?"
                continue
            j_best = min((int(j) for j in js), key=lambda j: (int(sizes[j]), j))
            out[c] = self.factor_names[j_best]
        return out

    def extraction_order(self) -> list[int]:
        """
        Ordine di estrazione dei fattori per la PCA di init (A2): dal fattore con
        PIU' membri (il globale, se presente) al piu' stretto. Per le mask diagonali
        (insiemi disgiunti) l'ordine e' irrilevante e questo restituisce comunque un
        ordine valido. Il globale va estratto per primo cosi' i locali si estraggono
        sui RESIDUI (co-movimento specifico del blocco, ortogonale al globale).
        """
        sizes = self.mask.sum(axis=0)                 # membri per fattore
        # ordine per numerosita' decrescente, stabile sull'indice colonna
        return sorted(range(self.r), key=lambda j: (-int(sizes[j]), j))


    @classmethod
    def from_block_map(cls, block_map: dict[str, str],
                       ordered_cols: list[str]) -> "FactorStructure":
        """
        Adattatore da una mappa serie->blocco (una serie in UN blocco) a una
        FactorStructure DIAGONALE. I nomi dei fattori sono i blocchi nell'ordine
        di PRIMA APPARIZIONE in `ordered_cols` — quindi NON viene cablato alcun
        nome: l'ordine emerge dai dati passati dal chiamante. Per il dataset
        `final` si usa invece build_loading_mask con le spec nominate.
        """
        factor_names: list[str] = []
        for c in ordered_cols:
            b = block_map[c]
            if b not in factor_names:
                factor_names.append(b)
        col_of = {b: j for j, b in enumerate(factor_names)}
        M, r = len(ordered_cols), len(factor_names)
        mask = np.zeros((M, r), dtype=int)
        for i, c in enumerate(ordered_cols):
            mask[i, col_of[block_map[c]]] = 1
        return cls(spec_name="legacy_block_map", factor_names=factor_names,
                   mask=mask, diagonal=True, ordered_cols=list(ordered_cols))


def as_structure(structure, ordered_cols) -> "FactorStructure":
    """
    Normalizza l'argomento a FactorStructure. Accetta:
      * una FactorStructure (mask-driven),
      * un dict serie->blocco -> struttura diagonale generica.
    """
    if isinstance(structure, FactorStructure):
        return structure
    if isinstance(structure, dict):
        return FactorStructure.from_block_map(structure, list(ordered_cols))
    raise TypeError(f"structure deve essere FactorStructure o dict, non {type(structure)}")


def load_factor_specs(path: str | None = None) -> dict:
    """Carica `config/factor_specs.json`."""
    p = path or _FACTOR_SPECS_PATH
    if not os.path.isfile(p):
        raise FileNotFoundError(f"factor_specs non trovato: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_loading_mask(
    spec_name: str,
    ordered_cols: list[str],
    specs: dict | None = None,
) -> FactorStructure:
    """
    Costruisci la loading mask M x r per una spec e un ordine di serie.

    Parameters
    ----------
    spec_name : str
        "fed_overlap" | "diag4" | "diag3" (chiavi di factor_specs.json["specs"]).
    ordered_cols : list[str]
        L'ordine delle serie = colonne del dataset (righe della mask).
    specs : dict, optional
        Contenuto gia' caricato di factor_specs.json; altrimenti viene letto.

    Returns
    -------
    FactorStructure
    """
    data = specs or load_factor_specs()
    if spec_name not in data["specs"]:
        raise KeyError(f"spec '{spec_name}' non in {list(data['specs'])}.")
    spec = data["specs"][spec_name]
    factor_names = list(data["factor_column_order"][spec_name])
    col_of = {name: j for j, name in enumerate(factor_names)}
    assignments = spec["assignments"]

    # validazione: ogni serie del dataset deve avere un'assegnazione, e viceversa.
    missing = [c for c in ordered_cols if c not in assignments]
    if missing:
        raise KeyError(f"spec '{spec_name}': serie senza assegnazione: {missing}")
    extra = [c for c in assignments if c not in ordered_cols]
    if extra:
        raise KeyError(f"spec '{spec_name}': assegnazioni per serie assenti dal "
                       f"dataset: {extra}")

    M, r = len(ordered_cols), len(factor_names)
    mask = np.zeros((M, r), dtype=int)
    for i, col in enumerate(ordered_cols):
        for fname in assignments[col]:
            if fname not in col_of:
                raise KeyError(f"spec '{spec_name}': fattore '{fname}' della serie "
                               f"{col} non in factor_column_order {factor_names}.")
            mask[i, col_of[fname]] = 1

    # sanity: nessuna riga vuota (ogni serie carica su almeno un fattore),
    # nessuna colonna vuota (ogni fattore ha almeno una serie).
    empty_rows = [ordered_cols[i] for i in range(M) if mask[i].sum() == 0]
    if empty_rows:
        raise ValueError(f"spec '{spec_name}': serie senza alcun fattore: {empty_rows}")
    empty_cols = [factor_names[j] for j in range(r) if mask[:, j].sum() == 0]
    if empty_cols:
        raise ValueError(f"spec '{spec_name}': fattori senza serie: {empty_cols}")

    diagonal = bool((mask.sum(axis=1) == 1).all())
    declared = bool(spec.get("diagonal", diagonal))
    if diagonal != declared:
        raise ValueError(
            f"spec '{spec_name}': diagonal derivato={diagonal} != dichiarato={declared} "
            f"(righe con >1 entrata: {int((mask.sum(axis=1) > 1).sum())}).")

    return FactorStructure(
        spec_name=spec_name,
        factor_names=factor_names,
        mask=mask,
        diagonal=diagonal,
        ordered_cols=list(ordered_cols),
    )
