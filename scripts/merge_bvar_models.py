#!/usr/bin/env python3
"""Merge the four model shards of one BVAR block into canonical outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import output_layout as layout
from core.bvar.evaluate import merge_model_runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    base = ROOT / "output" / "_bvar_shards" / f"{args.start}_{args.end}"
    roots = {model: str(base / model) for model in layout.BVAR_MODELS}
    merged = merge_model_runs(args.start, args.end, roots)
    print(f"  merged {len(merged)} rows from {len(roots)} model shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
