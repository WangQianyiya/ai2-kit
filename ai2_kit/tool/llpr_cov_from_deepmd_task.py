"""
Compute LLPR covariance matrix and calibration constant C from a DeepMD
training task directory.

Runs on the GPU training node after model freezing.
Reads training data from input.json's training_data.systems (DeepMD NPY format),
loads the frozen model, extracts last-layer features, saves the covariance
matrix as llpr_cov.npy, and calibrates C = mean((err/N)^2 / u_raw) using the
same training data.  Results are written to llpr_meta.json.

Usage:
    python -m ai2_kit.tool.llpr_cov_from_deepmd_task \
        --dp_task_dir /path/to/000 --sigma 0.01
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def _iter_deepmd_npy_frames(system_dir: str):
    """
    Yield (coords, atype, box, energy_or_None) per frame in a DeepMD NPY
    system directory.  Pure numpy reader -- no dpdata dependency.
    """
    type_raw_path = os.path.join(system_dir, "type.raw")
    atype = np.loadtxt(type_raw_path, dtype=np.int32).flatten()

    set_dirs = sorted(glob.glob(os.path.join(system_dir, "set.*")))
    if not set_dirs:
        return

    for sd in set_dirs:
        coord_path = os.path.join(sd, "coord.npy")
        box_path = os.path.join(sd, "box.npy")
        energy_path = os.path.join(sd, "energy.npy")
        if not os.path.isfile(coord_path):
            continue
        coords_all = np.load(coord_path)
        n_frames = coords_all.shape[0]
        n_atoms = len(atype)
        coords_all = coords_all.reshape(n_frames, n_atoms, 3)

        if os.path.isfile(box_path):
            boxes_all = np.load(box_path).reshape(n_frames, 9)
        else:
            boxes_all = np.tile(np.eye(3).flatten() * 100.0, (n_frames, 1))

        if os.path.isfile(energy_path):
            energies = np.load(energy_path).flatten()
        else:
            energies = None

        for i in range(n_frames):
            e = float(energies[i]) if energies is not None else None
            yield coords_all[i], atype, boxes_all[i], e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_task_dir", required=True,
                        help="DeepMD task directory (e.g. .../tasks/000)")
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="Regularization sigma for inv_M (default: 0.01)")
    args = parser.parse_args()

    dp_task_dir = args.dp_task_dir
    sigma = args.sigma

    model_path = None
    for name in ("frozen_model.pth", "frozen_model.pb"):
        p = os.path.join(dp_task_dir, name)
        if os.path.isfile(p):
            model_path = p
            break
    if model_path is None:
        print("ERROR: no frozen model found in", dp_task_dir, file=sys.stderr)
        sys.exit(1)

    input_json_path = os.path.join(dp_task_dir, "input.json")
    with open(input_json_path) as f:
        dp_input = json.load(f)
    systems = dp_input["training"]["training_data"]["systems"]

    print(f"[llpr_cov] model: {model_path}")
    print(f"[llpr_cov] training systems: {len(systems)}")
    print(f"[llpr_cov] sigma: {sigma}")

    from deepmd.infer import DeepPot
    dp = DeepPot(model_path)

    try:
        model_type_map = list(dp.get_type_map())
    except Exception:
        type_map_path = os.path.join(dp_task_dir, "type_map.raw")
        if os.path.isfile(type_map_path):
            with open(type_map_path) as f:
                model_type_map = f.read().strip().split()
        else:
            print("ERROR: cannot determine type_map", file=sys.stderr)
            sys.exit(1)

    n_types = len(model_type_map)

    from ai2_kit.domain.llpr import get_h_mol_per_type, u_raw_score

    # Collect features, coords/box/atype (for E_pred later), and E_ref
    f_list = []
    coords_list = []
    atype_list = []
    box_list = []
    e_ref_list = []
    has_energy = True
    n_frames_total = 0

    for sys_dir in systems:
        for coords, atype, box, e_ref in _iter_deepmd_npy_frames(sys_dir):
            f = get_h_mol_per_type(dp, coords, box, atype, n_types)
            f_list.append(f)
            coords_list.append(coords)
            atype_list.append(atype)
            box_list.append(box)
            if e_ref is not None:
                e_ref_list.append(e_ref)
            else:
                has_energy = False
            n_frames_total += 1
            if n_frames_total % 100 == 0:
                print(f"[llpr_cov] processed {n_frames_total} frames ...")

    if not f_list:
        print("ERROR: no training frames loaded", file=sys.stderr)
        sys.exit(1)

    F = np.stack(f_list, axis=0)
    n_train, D = F.shape
    cov = (F.T @ F) / n_train

    out_path = os.path.join(dp_task_dir, "llpr_cov.npy")
    np.save(out_path, cov)
    print(f"[llpr_cov] saved covariance ({D}x{D}) from {n_train} frames -> {out_path}")

    # Calibrate C = mean((err/N_atoms)^2 / u_raw) on training set
    C_val = None
    if has_energy and len(e_ref_list) == n_train:
        inv_M = np.linalg.inv(cov + (sigma ** 2) * np.eye(D, dtype=cov.dtype))
        ratios = []
        for i in range(n_train):
            u = u_raw_score(f_list[i], inv_M)
            if u < 1e-12:
                continue
            n_atoms = len(atype_list[i])
            coords_i = np.asarray(coords_list[i], dtype=np.float64).reshape(1, n_atoms, 3)
            box_i = np.asarray(box_list[i], dtype=np.float64).reshape(1, 9)
            atype_i = np.asarray(atype_list[i], dtype=np.int32)
            out = dp.eval(coords_i, box_i, atype_i, atomic=False)
            if isinstance(out, (list, tuple)):
                e_pred = float(np.asarray(out[0]).flatten()[0])
            elif isinstance(out, dict):
                val = out.get("energy", out.get("o_energy", 0))
                e_pred = float(np.asarray(val).flatten()[0])
            else:
                e_pred = float(np.asarray(out).flatten()[0])
            err_per_atom = (e_pred - e_ref_list[i]) / n_atoms
            ratios.append(err_per_atom ** 2 / u)

        if ratios:
            C_val = float(np.mean(ratios))
            print(f"[llpr_cov] calibrated C = {C_val:.6e}  "
                  f"(from {len(ratios)}/{n_train} frames, per-atom)")
        else:
            print("[llpr_cov] WARNING: all u_raw < 1e-12, cannot calibrate C",
                  file=sys.stderr)
    else:
        print("[llpr_cov] WARNING: energy.npy missing in some systems, "
              "skipping C calibration", file=sys.stderr)

    meta = {
        "n_train": n_train,
        "D": D,
        "n_types": n_types,
        "type_map": model_type_map,
        "model": os.path.basename(model_path),
        "sigma": sigma,
        "C": C_val,
        "C_unit": "eV/atom (sqrt(C*u_raw) gives per-atom energy error estimate)",
    }
    meta_path = os.path.join(dp_task_dir, "llpr_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[llpr_cov] wrote {meta_path}")


if __name__ == "__main__":
    main()
