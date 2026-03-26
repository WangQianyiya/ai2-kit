#!/usr/bin/env python3
"""
Run LAMMPS using the in-process shared library (Python lammps API).

PyPI package ``lammps`` ships a small ``lmp`` executable that invokes this
library in a subprocess; on some clusters that stub **segfaults**, while
``from lammps import lammps`` + ``CDLL(liblammps.so)`` works.

ai2-kit invokes: ``<lammps_cmd> -i lammps.input [-v restart 0|1]``

This driver parses the same argv and runs the input via ``lammps.file()``,
which is reliable for DeepMD pair styles in the affected wheels.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="lmp_via_liblammps", add_help=True)
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        default=None,
        help="LAMMPS input script (same as lmp -i)",
    )
    parser.add_argument(
        "-v",
        dest="v_pair",
        nargs=2,
        metavar=("NAME", "VALUE"),
        action="append",
        default=[],
        help='Index variable, e.g. -v restart 0 (repeatable)',
    )
    parser.add_argument(
        "-echo",
        choices=["none", "screen", "log", "both"],
        default="screen",
    )
    parser.add_argument(
        "-nocite",
        action="store_true",
        help="Ignored for compatibility with lmp CLI; LAMMPS may still print cite info.",
    )
    # Keep unknown args for future compatibility (e.g. -log)
    args, unknown = parser.parse_known_args()
    if unknown:
        # Do not fail on flags we do not implement (e.g. -log)
        pass

    if not args.input_file:
        print("lmp_via_liblammps: missing -i INPUT", file=sys.stderr)
        return 2

    # Import after argparse so ``-h`` works without loading TF/DeepMD.
    from lammps import lammps  # noqa: WPS433

    cmdargs = ["-echo", args.echo]
    if args.nocite:
        cmdargs.append("-nocite")

    L = lammps(cmdargs=cmdargs)
    try:
        for name, value in args.v_pair:
            L.command(f"variable {name} index {value}")
        L.file(args.input_file)
    finally:
        try:
            L.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
