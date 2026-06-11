"""
src/config_utils.py

Shared utility for config-aware CLI entry points.

Every module that accepts --small / --big / --config <name> imports from
here.  This keeps the parsing logic and path-building logic in one place.

Public API
----------
parse_config_args(description, extra=None) -> argparse.Namespace
    Parse --small, --big, --config <name>.  Default: "small".
    Pass a callable ``extra(parser)`` to add module-specific flags.

resolve_output_path(kind, filename, config_name) -> pathlib.Path
    Build a config-specific output path and create the parent dir:
      "figures"   -> output/<config>/figures/<filename>
      "processed" -> data/processed/<config>/<filename>
      "dataset"   -> data/processed/dataset_<config>.csv  (filename ignored)

get_project_root() -> pathlib.Path
    Return the project root (parent of src/).
"""

import argparse
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def get_project_root() -> pathlib.Path:
    return _PROJECT_ROOT


def parse_config_args(
    description: str = "",
    extra=None,
) -> argparse.Namespace:
    """
    Parse --small, --big, --config <name>.

    --small and --big are mutually exclusive shortcuts for --config small/big.
    If none is specified, the default is "small".

    Parameters
    ----------
    description : str
        Description passed to ArgumentParser.
    extra : callable or None
        Optional callable ``extra(parser)`` that can add module-specific
        arguments to the parser before parsing.  The function receives
        the ArgumentParser instance and should call parser.add_argument().

    Returns
    -------
    argparse.Namespace
        Namespace with attribute ``config`` set to the resolved config name.
    """
    parser = argparse.ArgumentParser(description=description)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--small",
        dest="config",
        action="store_const",
        const="small",
        help="Use the 'small' config (20 series). Equivalent to --config small.",
    )
    grp.add_argument(
        "--big",
        dest="config",
        action="store_const",
        const="big",
        help="Use the 'big' config (50 series). Equivalent to --config big.",
    )
    grp.add_argument(
        "--config",
        dest="config",
        metavar="NAME",
        help=(
            "Config name — loads config/series_<NAME>.json. "
            "Default: small."
        ),
    )
    if extra is not None:
        extra(parser)
    args = parser.parse_args()
    if args.config is None:
        args.config = "small"
    return args


def resolve_output_path(
    kind: str,
    filename: str,
    config_name: str,
) -> pathlib.Path:
    """
    Build a config-specific output path and ensure the parent directory exists.

    Parameters
    ----------
    kind : {"figures", "processed", "dataset"}
        "figures"   -> output/<config>/figures/<filename>
        "processed" -> data/processed/<config>/<filename>
        "dataset"   -> data/processed/dataset_<config>.csv  (filename ignored)
    filename : str
        File name (including extension).  Ignored when kind=="dataset".
    config_name : str
        The active config name (e.g. "small" or "big").

    Returns
    -------
    pathlib.Path
        Absolute path with parent directory already created.
    """
    root = _PROJECT_ROOT
    if kind == "figures":
        p = root / "output" / config_name / "figures" / filename
    elif kind == "processed":
        p = root / "data" / "processed" / config_name / filename
    elif kind == "dataset":
        p = root / "data" / "processed" / f"dataset_{config_name}.csv"
    else:
        raise ValueError(
            f"Unknown kind: {kind!r}.  "
            f"Expected one of 'figures', 'processed', 'dataset'."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
