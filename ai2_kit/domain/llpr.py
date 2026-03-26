"""
LLPR (Last Layer Probabilistic Regression) core functions for DeepMD backend.

Computes per-structure uncertainty via the last-layer feature covariance:
    Sigma = F^T F / N
    inv_M = (Sigma + sigma^2 * I)^{-1}
    u_raw = f^T inv_M f
    C = mean((err/N_atoms)^2 / u_raw)   (calibrated on training set)
    sigma_e_per_atom = sqrt(C * u_raw)   (eV/atom)

Requires deepmd-kit with eval_fitting_last_layer API support.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def get_h_mol_per_type(dp, coords: np.ndarray, box: np.ndarray,
                       atype: np.ndarray, n_types: int) -> np.ndarray:
    """
    Extract per-type aggregated last-layer features from a DeepPot model.

    For each atom type t, sum the last-layer features of all atoms of that type
    to get a per-type feature block. Concatenate all blocks into a single vector f.

    Returns f of shape (n_types * d,) where d is the feature dimension per atom.
    """
    natoms = coords.shape[0]
    coords_batch = np.asarray(coords, dtype=np.float64).reshape(1, natoms, 3)
    cells_batch = np.asarray(box, dtype=np.float64).reshape(1, 9)
    atype_flat = np.asarray(atype, dtype=np.int32)

    h = dp.eval_fitting_last_layer(coords_batch, cells_batch, atype_flat)
    h = np.asarray(h)
    if h.ndim == 3:
        h = h[0]

    d = h.shape[1]
    f = np.zeros(n_types * d, dtype=h.dtype)
    for t in range(n_types):
        mask = (atype_flat == t)
        if mask.any():
            f[t * d: (t + 1) * d] = np.sum(h[mask], axis=0)
    return f


def get_f_and_energy(dp, coords: np.ndarray, box: np.ndarray,
                     atype: np.ndarray, n_types: int) -> Tuple[np.ndarray, float]:
    """
    Get both the per-type feature vector f and the predicted energy E from a DeepPot model.
    """
    f = get_h_mol_per_type(dp, coords, box, atype, n_types)
    natoms = coords.shape[0]
    coords_batch = np.asarray(coords, dtype=np.float64).reshape(1, natoms, 3)
    cells_batch = np.asarray(box, dtype=np.float64).reshape(1, 9)
    atype_flat = np.asarray(atype, dtype=np.int32)

    out = dp.eval(coords_batch, cells_batch, atype_flat, atomic=False)
    if isinstance(out, (list, tuple)):
        E_pred = float(np.asarray(out[0]).flatten()[0])
    elif isinstance(out, dict):
        val = out.get("energy", out.get("o_energy", 0))
        E_pred = float(np.asarray(val).flatten()[0])
    else:
        E_pred = float(np.asarray(out).flatten()[0])
    return f, E_pred


def build_inv_cov(dp, coords_list: List[np.ndarray], box_list: List[np.ndarray],
                  atype_list: List[np.ndarray], n_types: int,
                  sigma: float) -> Tuple[np.ndarray, int, int]:
    """
    Build the regularized inverse covariance matrix from training data.

    Returns (inv_M, n_train, D) where D = n_types * feature_dim.
    """
    f_list = []
    for coords, box, atype in zip(coords_list, box_list, atype_list):
        f = get_h_mol_per_type(dp, coords, box, atype, n_types)
        f_list.append(f)
    F = np.stack(f_list, axis=0)
    n_train, D = F.shape
    cov = (F.T @ F) / n_train
    M = cov + (sigma ** 2) * np.eye(D, dtype=cov.dtype)
    inv_M = np.linalg.inv(M)
    return inv_M, n_train, D


def u_raw_score(f: np.ndarray, inv_M: np.ndarray) -> float:
    """Compute raw LLPR uncertainty: u_raw = f^T inv_M f."""
    return float(np.dot(f, inv_M @ f))
