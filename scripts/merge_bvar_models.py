#!/usr/bin/env python3
"""Merge the model shards of one BVAR block into canonical outputs.

Di default vuole tutti e quattro i modelli: se ne manca uno, non scrive niente.
`--allow-missing` pubblica quelli che ci sono e dichiara chi manca — serve
perche' un modello morto non deve portarsi via il lavoro degli altri tre, gia'
calcolato e gia' sul disco.

Rifondere e' idempotente. Per recuperare il modello che mancava non si rilancia
il blocco intero, si rilancia LUI e poi si rifonde:

    python scripts/run_bvar.py --start 2020-07-31 --end 2020-10-23 \\
        --models lbvar --output-root output/_bvar_shards/2020-07-31_2020-10-23/lbvar
    python scripts/merge_bvar_models.py --start 2020-07-31 --end 2020-10-23
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import output_layout as layout
from core.bvar.evaluate import merge_model_runs, shard_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--allow-missing", action="store_true",
                        help="pubblica i modelli presenti invece di non "
                             "scrivere niente quando ne manca uno")
    args = parser.parse_args()

    base = ROOT / "output" / "_bvar_shards" / f"{args.start}_{args.end}"
    roots = {model: str(base / model) for model in layout.BVAR_MODELS}

    # Si dice PRIMA che cosa si sta per pubblicare, invece di scoprirlo da
    # un'eccezione o, peggio, di non accorgersene affatto.
    stato = shard_status(args.start, args.end, roots)
    assenti = [m for m, p in stato.items() if p != "pronto"]
    for model in layout.BVAR_MODELS:
        print(f"  {model:7s} {stato[model]}")
    if assenti and args.allow_missing:
        print(f"  !! blocco PARZIALE: {', '.join(assenti)} non entra")
        if "qbvar" in assenti:
            print("  !! il q-BVAR porta anche i benchmark del blocco: "
                  "escono con lui")

    merged = merge_model_runs(args.start, args.end, roots,
                              require_all=not args.allow_missing)
    modelli = sorted(set(merged["spec"].astype(str)))
    print(f"  scritte {len(merged)} righe; nel blocco: {', '.join(modelli)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
