"""
Compute per-frame LLPR scores after a LAMMPS MD simulation.

Runs on the same compute node as the LAMMPS job (appended as a BashStep).
Reads traj.lammpstrj for structures, model_devi.out for step numbers,
llpr_cov.npy for the pre-computed covariance, and writes llpr.out.

Usage:
    python -m ai2_kit.tool.llpr_score_lammps_traj \
        --task_dir /path/to/lammps/task \
        --model_path /path/to/frozen_model.pth \
        --cov_path /path/to/llpr_cov.npy \
        --sigma 0.01 \
        --type_map O,H
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _parse_lammpstrj_steps(path: str) -> list:
    """Extract timestep numbers from LAMMPS dump file headers."""
    steps = []
    with open(path) as f:
        expect_step = False
        for line in f:
            if 'ITEM: TIMESTEP' in line:
                expect_step = True
            elif expect_step:
                try:
                    steps.append(int(line.strip()))
                except ValueError:
                    pass
                expect_step = False
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cov_path", required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--type_map", required=True,
                        help="Comma-separated type map, e.g. O,H")
    args = parser.parse_args()

    type_map = args.type_map.split(",")
    task_dir = args.task_dir

    # Read pre-computed covariance
    if not os.path.isfile(args.cov_path):
        print(f"[llpr_score] WARNING: {args.cov_path} not found, skipping", file=sys.stderr)
        return
    cov = np.load(args.cov_path)
    D = cov.shape[0]
    inv_M = np.linalg.inv(cov + (args.sigma ** 2) * np.eye(D, dtype=cov.dtype))

    # Read calibration C from llpr_meta.json (if available)
    C_val = None
    meta_path = os.path.join(os.path.dirname(args.cov_path), "llpr_meta.json")
    if os.path.isfile(meta_path):
        import json
        with open(meta_path) as mf:
            meta = json.load(mf)
        C_val = meta.get("C")
        if C_val is not None:
            print(f"[llpr_score] loaded C = {C_val:.6e} from {meta_path}")

    # Read trajectory
    traj_path = os.path.join(task_dir, "traj.lammpstrj")
    if not os.path.isfile(traj_path):
        print(f"[llpr_score] WARNING: {traj_path} not found, skipping", file=sys.stderr)
        return

    # Parse step numbers directly from LAMMPS dump header (ITEM: TIMESTEP)
    steps = _parse_lammpstrj_steps(traj_path)

    import ase.io
    try:
        atoms_list = ase.io.read(traj_path, ":", format="lammps-dump-text",
                                 specorder=type_map)
    except Exception:
        atoms_list = ase.io.read(traj_path, ":", format="extxyz")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    if not steps:
        print("[llpr_score] WARNING: no timesteps in dump, using frame indices",
              file=sys.stderr)
        steps = list(range(len(atoms_list)))

    n_frames = min(len(atoms_list), len(steps))

    # Load model
    from deepmd.infer import DeepPot
    dp = DeepPot(args.model_path)
    try:
        model_type_map = list(dp.get_type_map())
    except Exception:
        model_type_map = type_map
    n_types = len(model_type_map)

    from ai2_kit.domain.llpr import get_h_mol_per_type, u_raw_score

    # Score each frame
    print(f"[llpr_score] scoring {n_frames} frames ...")
    results = []
    for i in range(n_frames):
        at = atoms_list[i]
        coords = at.get_positions().astype(np.float64)
        symbols = at.get_chemical_symbols()
        atype = np.array([model_type_map.index(s) for s in symbols], dtype=np.int32)
        cell = at.get_cell()
        if cell.rank == 3 and np.any(at.get_pbc()):
            box = cell[:].flatten()
        else:
            box = np.eye(3, dtype=np.float64).flatten() * 30.0

        f = get_h_mol_per_type(dp, coords, box, atype, n_types)
        u = u_raw_score(f, inv_M)
        results.append((steps[i], u))

    # Write llpr.out
    out_path = os.path.join(task_dir, "llpr.out")
    with open(out_path, "w") as fp:
        if C_val is not None:
            fp.write("#         step       llpr_u_raw  sigma_e_per_atom\n")
            for step, u in results:
                sigma_e = np.sqrt(max(C_val * u, 0.0))
                fp.write(f"{step:14d} {u:18.8e} {sigma_e:18.8e}\n")
        else:
            fp.write("#         step       llpr_u_raw\n")
            for step, u in results:
                fp.write(f"{step:14d} {u:18.8e}\n")

    print(f"[llpr_score] wrote {len(results)} scores -> {out_path}")


if __name__ == "__main__":
    main()
