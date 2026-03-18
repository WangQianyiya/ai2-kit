from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.log import get_logger
from ai2_kit.core.util import dump_json, dump_text, flush_stdio, limit
from ai2_kit.core.pydantic import BaseModel

from typing import List, Optional, Tuple, Dict
from io import StringIO
from dataclasses import dataclass
import pandas as pd
from tabulate import tabulate
from itertools import groupby
from functools import lru_cache
import traceback

import ase.io
import numpy as np
import os

from .data import get_data_format, DataFormat, artifacts_to_ase_atoms
from .iface import ICllSelectorOutput, BaseCllContext
from .constant import DEFAULT_ASAP_SOAP_DESC, DEFAULT_ASAP_PCA_REDUCER


logger = get_logger(__name__)


class CllModelDeviSelectorInputConfig(BaseModel):

    class AsapOptions(BaseModel):
        disable: bool = False
        limit_per_cluster: int = 1
        """
        limit the number of structures to be selected from the same cluster
        """
        sort_by_ssw_energy: bool = False
        """
        sorted the structures by ssw_energy in each cluster
        """
        descriptor: dict = {'soap': { **DEFAULT_ASAP_SOAP_DESC, 'preset': 'minimal'}}
        dim_reducer: dict = {'pca': DEFAULT_ASAP_PCA_REDUCER}
        cluster: dict = {'dbscan': {}}

    f_trust_lo: float = 0.
    """
    the lower bound of model_devi score to select the structure for labeling
    """
    f_trust_hi: float = 65535.
    """
    the upper bound of model_devi score to select the structure for labeling
    """
    new_explore_system_q: float = 0.25
    """
    the quantile of model_devi score to select the structure for next round of exploration
    """
    asap_options: Optional[AsapOptions] = None
    """
    options for ASAP to further select candidates
    """
    screening_fn: Optional[str] = None
    """
    the function to screen the candidates, e.g
    "lambda x: x['ssw_energy'] < -1000"
    """
    max_decent_per_traj: int = -1
    """
    limit the max number of decent structures per trajectory, -1 means unlimited
    """
    workers: int = 4
    """
    number of workers to run the analysis
    """


@dataclass
class CllModelDevSelectorContext(BaseCllContext):
    ...


@dataclass
class CllModelDeviSelectorOutput(ICllSelectorOutput):
    candidates: List[Artifact]
    passing_rate: float
    new_explore_systems: List[Artifact]

    def get_model_devi_dataset(self):
        return self.candidates

    def get_passing_rate(self) -> float:
        return self.passing_rate

    def get_new_explore_systems(self) -> List[Artifact]:
        return self.new_explore_systems


@dataclass
class CllModelDeviSelectorInput:
    config: CllModelDeviSelectorInputConfig
    model_devi_data: List[Artifact]
    model_devi_file: str
    type_map: List[str]

    def set_model_devi_dataset(self, data: List[Artifact]):
        self.model_devi_data = data


async def cll_model_devi_selector(input: CllModelDeviSelectorInput, ctx: CllModelDevSelectorContext):
    executor = ctx.resource_manager.default_executor
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    executor.mkdir(work_dir)

    f_trust_lo = input.config.f_trust_lo
    f_trust_hi = input.config.f_trust_hi

    results = executor.run_python_fn(bulk_select_structures_by_model_devi)(
        model_devi_outputs=[a.to_dict() for a in input.model_devi_data],
        model_devi_file=input.model_devi_file,
        f_trust_lo=f_trust_lo, f_trust_hi=f_trust_hi,
        type_map=input.type_map, work_dir=work_dir,
        new_explore_system_q=input.config.new_explore_system_q,
        max_decent_per_traj=input.config.max_decent_per_traj,
        screening_fn=input.config.screening_fn,
        workers=input.config.workers,
    )

    candidates = [ result['decent'] for result, _ in results if 'decent' in result ]
    new_systems = [ result['next'] for result, _ in results if 'next' in result ]
    stats = [ stats for _, stats in results ]

    # group next_structures by `attrs.source` and keep the first one
    # so that the total number of explore structures will be the same as the original one
    get_source = lambda s: s['attrs']['source']
    # filter out the structures whose ancestor ends with '-fin'
    # FIXME: use a dedicated field to indicate the final structure
    new_systems = [s for s in new_systems if not s['attrs']['ancestor'].endswith('-fin')]
    new_systems = sorted(new_systems, key=get_source)
    new_systems = [next(group) for _source, group in groupby(
        new_systems, key=get_source)]

    # write model_deviation stats report
    # TODO: refactor into a function
    def _get_row(stat: dict):
        url = stat['src']
        url = f'...{url[-30:]}' if len(url) > 30 else url  # wrap url if it is too long
        total, good, decent, poor = stat['total'], stat['good'], stat['decent'], stat['poor']
        return [
            url, total, good, decent, poor,
            f'{good / total * 100:.2f}%', f'{decent / total * 100:.2f}%', f'{poor / total * 100:.2f}%',
        ]
    headers = ['file', 'total', 'good', 'decent', 'poor', 'good%', 'decent%', 'poor%']
    table = [ _get_row(stat) for stat in stats]
    total = sum(row[1] for row in table)
    total_good = sum(row[2] for row in table)

    stats_report = tabulate(table, headers=headers, tablefmt='tsv')
    logger.info('stats report: \n%s\n', stats_report)
    executor.dump_text(stats_report, os.path.join(work_dir, 'stats.tsv'))

    # further select candidates by ASAP
    if input.config.asap_options and not input.config.asap_options.disable:
        asap_options = input.config.asap_options
        candidates = executor.run_python_fn(bulk_select_distinct_structures)(
            candidates=candidates,
            descriptor_opt=asap_options.descriptor,
            dim_reducer_opt=asap_options.dim_reducer,
            cluster_opt=asap_options.cluster,
            type_map=input.type_map,
            work_dir=work_dir,
            limit_per_cluster=asap_options.limit_per_cluster,
            sort_by_energy=asap_options.sort_by_ssw_energy,
            workers=input.config.workers,
        )

    return CllModelDeviSelectorOutput(
        candidates=[Artifact.of(**a) for a in candidates],
        new_explore_systems=[Artifact.of(**a) for a in new_systems],
        passing_rate=total_good / total,
    )


class CllLlprSelectorInputConfig(BaseModel):
    """
    Config for LLPR-based selector (LLPR-DP-MACE).
    Selects the top-n structures by LLPR uncertainty for labeling.
    Optional SOAP/ASAP clustering can be applied on top of LLPR top-n.
    """

    n_candidates: int = 100
    """Number of structures to select (uncertainty ranked, highest first)."""
    sigma: float = 0.01
    """LLPR covariance regularization."""
    train_xyz: str = ""
    """
    Path to training XYZ for building LLPR covariance.
    If empty (default), automatically uses the training dataset from the Train stage
    (DeepMD NPY format, read via dpdata).
    """
    val_xyz: Optional[str] = None
    """Path to validation XYZ for C calibration. Only used when train_xyz is manually set."""
    energy_key: str = "U0"
    """Energy key in XYZ info dict. Only used when train_xyz is manually set."""

    asap_options: Optional[CllModelDeviSelectorInputConfig.AsapOptions] = None
    """
    Optional: after LLPR top-n selection, run SOAP/ASAP clustering to pick diverse structures
    (one per cluster by default). If representatives are fewer than n_candidates, the rest are
    filled from the LLPR-ordered pool by uncertainty only (no SOAP for fill-up).
    Same interface as model_devi selector (descriptor=SOAP, dim_reducer=PCA, cluster=DBSCAN).
    Set disable: true or omit to skip clustering.
    """
    workers: int = 4
    """Number of workers for ASAP clustering when asap_options is used."""


@dataclass
class CllLlprSelectorInput:
    config: CllLlprSelectorInputConfig
    model_devi_data: List[Artifact]
    model_devi_file: str
    type_map: List[str]
    models: List[Artifact]
    """Trained ML models (e.g. from train_output.get_mlp_models()); first is used for LLPR."""
    training_dataset: List[Artifact]
    """Training dataset from Train stage; used to auto-build LLPR covariance when train_xyz is empty."""


@dataclass
class CllLlprSelectorContext(BaseCllContext):
    ...


def _atoms_to_coords_atype_box(atoms, type_map: List[str]):
    """Convert ASE atoms to coords, atype (0..n_types-1), box for DeepMD."""
    coords = atoms.get_positions().astype(np.float64)
    symbols = atoms.get_chemical_symbols()
    atype = np.array([type_map.index(s) for s in symbols], dtype=np.int32)
    cell = atoms.get_cell()
    if cell.rank == 3 and np.any(atoms.get_pbc()):
        box = cell[:].flatten()
    else:
        box = np.eye(3, dtype=np.float64).flatten() * 30.0
    return coords, atype, box


def _load_training_data_from_deepmd_npy(
    training_datasets: List[ArtifactDict],
    type_map: List[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Read DeepMD NPY directories (via dpdata) and return
    (coords_list, atype_list, box_list, energies_list) for LLPR.
    """
    import dpdata
    all_coords: List[np.ndarray] = []
    all_atype: List[np.ndarray] = []
    all_box: List[np.ndarray] = []
    all_energies: List[float] = []
    for ds_dict in training_datasets:
        url = ds_dict["url"]
        fmt = ds_dict.get("format", "")
        if fmt == DataFormat.DEEPMD_NPY:
            ds = dpdata.LabeledSystem(url, fmt='deepmd/npy')
        elif fmt == DataFormat.EXTXYZ:
            ds = dpdata.LabeledSystem(url, fmt='extxyz')
        else:
            logger.warning("Skip unsupported training data format %s for LLPR auto-load", fmt)
            continue
        if len(ds) == 0:
            continue
        atom_names = list(ds['atom_names'])
        atom_types_raw = ds['atom_types']
        symbols = [atom_names[t] for t in atom_types_raw]
        atype = np.array([type_map.index(s) for s in symbols], dtype=np.int32)
        for i in range(len(ds)):
            all_coords.append(ds['coords'][i].astype(np.float64))
            all_atype.append(atype.copy())
            cell = ds['cells'][i]
            if np.abs(np.linalg.det(cell)) > 1e-6:
                all_box.append(cell.flatten().astype(np.float64))
            else:
                all_box.append(np.eye(3, dtype=np.float64).flatten() * 30.0)
            all_energies.append(float(ds['energies'][i]))
    return all_coords, all_atype, all_box, all_energies


def _run_llpr_select(
    model_devi_outputs: List[ArtifactDict],
    model_devi_file: str,
    type_map: List[str],
    work_dir: str,
    model_path: str,
    sigma: float,
    n_candidates: int,
    training_datasets: Optional[List[ArtifactDict]] = None,
    train_xyz: str = "",
    val_xyz: Optional[str] = None,
    energy_key: str = "U0",
) -> Tuple[List[ArtifactDict], List[ArtifactDict], float]:
    """
    Run LLPR on explore structures and return top-n by uncertainty.
    Designed to be run via executor.run_python_fn().

    If training_datasets is provided (and train_xyz is empty), reads training data
    directly from DeepMD NPY dirs via dpdata. Otherwise falls back to reading
    train_xyz as an XYZ file via DeepMDDataLoader.
    """
    os.makedirs(work_dir, exist_ok=True)

    # Collect (url, attrs, idx, atoms) for every frame from explore outputs
    flat_list: List[Tuple[str, dict, int, "ase.Atoms"]] = []
    for out in model_devi_outputs:
        data_format = get_data_format(out)
        url = out["url"]
        attrs = dict(out.get("attrs", {}))
        attrs.pop("model_devi_file", None)
        if data_format in (DataFormat.LAMMPS_OUTPUT_DIR, DataFormat.LASP_LAMMPS_OUT_DIR):
            if data_format == DataFormat.LASP_LAMMPS_OUT_DIR:
                traj_path = os.path.join(url, "structures.xyz")
            else:
                traj_path = os.path.join(url, attrs.get("structures", "traj.lammpstrj"))
            if not os.path.isfile(traj_path):
                traj_path = os.path.join(url, "traj.lammpstrj")
            try:
                atoms_list = ase.io.read(traj_path, ":", format="lammps-dump-text", specorder=type_map)
            except Exception:
                atoms_list = ase.io.read(traj_path, ":", format="extxyz")
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            for idx, at in enumerate(atoms_list):
                flat_list.append((url, dict(attrs), idx, at))
        elif data_format == DataFormat.ANYWARE_OUTPUT_DIR:
            structures_file = os.path.join(url, "structures.xyz")
            atoms_list = ase.io.read(structures_file, ":", format="extxyz")
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            for idx, at in enumerate(atoms_list):
                flat_list.append((url, dict(attrs), idx, at))
        else:
            logger.warning("Skip unsupported format %s for LLPR", data_format)

    if not flat_list:
        return [], [], 0.0

    from deepmd.infer import DeepPot
    from ai2_kit.domain.llpr import build_inv_cov, u_raw_score, get_f_and_energy, get_h_mol_per_type

    dp = DeepPot(model_path)
    try:
        model_type_map = list(dp.get_type_map())
    except Exception:
        model_type_map = type_map
    n_types = len(model_type_map)

    use_auto_dataset = bool(training_datasets) and not train_xyz
    if use_auto_dataset:
        coords_tr, atype_tr, box_tr, energies_tr = _load_training_data_from_deepmd_npy(
            training_datasets, type_map)
        if not coords_tr:
            raise ValueError("No frames loaded from training_datasets for LLPR covariance.")
        inv_M, _, _ = build_inv_cov(dp, coords_tr, box_tr, atype_tr, n_types, sigma)
        u_raw_list, sq_err_list = [], []
        for coords, atype, box, e_true in zip(coords_tr, atype_tr, box_tr, energies_tr):
            f, E_pred = get_f_and_energy(dp, coords, box, atype, n_types)
            u_raw_list.append(u_raw_score(f, inv_M))
            sq_err_list.append((E_pred - e_true) ** 2)
    else:
        if not train_xyz or not os.path.isfile(train_xyz):
            raise FileNotFoundError(f"LLPR train_xyz must exist: {train_xyz}")
        val_xyz = val_xyz or train_xyz

        def _load_xyz(path, ek):
            atoms_list = ase.io.read(path, ":", format="extxyz")
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            cl, al, bl, el = [], [], [], []
            for at in atoms_list:
                c, a, b = _atoms_to_coords_atype_box(at, type_map)
                cl.append(c); al.append(a); bl.append(b)
                el.append(float(at.info.get(ek, at.info.get("energy", 0.0))))
            return cl, al, bl, el

        coords_tr, atype_tr, box_tr, _ = _load_xyz(train_xyz, energy_key)
        inv_M, _, _ = build_inv_cov(dp, coords_tr, box_tr, atype_tr, n_types, sigma)
        coords_v, atype_v, box_v, energies_v = _load_xyz(val_xyz, energy_key)
        u_raw_list, sq_err_list = [], []
        for coords, atype, box, e_true in zip(coords_v, atype_v, box_v, energies_v):
            f, E_pred = get_f_and_energy(dp, coords, box, atype, n_types)
            u_raw_list.append(u_raw_score(f, inv_M))
            sq_err_list.append((E_pred - e_true) ** 2)

    u_raw_val = np.array(u_raw_list)
    sq_err_val = np.array(sq_err_list)
    u_raw_safe = np.maximum(u_raw_val, 1e-12)
    C = float(np.mean(sq_err_val / u_raw_safe))

    # Score explore structures
    use_type_map = model_type_map if model_type_map else type_map
    coords_list = []
    atype_list = []
    box_list = []
    for _, _, _, at in flat_list:
        c, a, b = _atoms_to_coords_atype_box(at, use_type_map)
        coords_list.append(c)
        atype_list.append(a)
        box_list.append(b)
    u_raw_test = []
    for coords, atype, box in zip(coords_list, atype_list, box_list):
        f = get_h_mol_per_type(dp, coords, box, atype, n_types)
        u_raw_test.append(u_raw_score(f, inv_M))
    u = C * np.array(u_raw_test)

    # Top-n by uncertainty (descending)
    order = np.argsort(-u)
    n_sel = min(n_candidates, len(order))
    top_indices = order[:n_sel]

    # Build candidates: one file with top-n frames
    candidates_xyz = os.path.join(work_dir, "llpr_candidates.xyz")
    selected_atoms = [flat_list[i][3] for i in top_indices]
    ase.io.write(candidates_xyz, selected_atoms, format="extxyz")
    first_attrs = flat_list[top_indices[0]][1] if n_sel else {}
    candidates = [
        {"url": candidates_xyz, "format": DataFormat.EXTXYZ, "attrs": {**first_attrs, "ancestor": first_attrs.get("ancestor", "llpr")}}
    ]

    # New explore systems: one per task (url), pick frame with max variance in that task
    url_to_indices: Dict[str, List[int]] = {}
    for i, (url, _, _, _) in enumerate(flat_list):
        url_to_indices.setdefault(url, []).append(i)
    new_systems = []
    next_dir = os.path.join(work_dir, "next")
    os.makedirs(next_dir, exist_ok=True)
    for url, indices in url_to_indices.items():
        best_i = indices[np.argmax(u[indices])]
        _, attrs, _, at = flat_list[best_i]
        next_xyz = os.path.join(next_dir, f"next_{len(new_systems):06d}.xyz")
        ase.io.write(next_xyz, [at], format="extxyz")
        new_systems.append({"url": next_xyz, "format": DataFormat.EXTXYZ, "attrs": dict(attrs)})

    passing_rate = n_sel / len(flat_list) if flat_list else 0.0
    return candidates, new_systems, passing_rate


def _fill_up_llpr_candidates(
    work_dir: str,
    llpr_candidates_path: str,
    asap_candidates: List[ArtifactDict],
    n_candidates: int,
    type_map: List[str],
) -> List[ArtifactDict]:
    """
    When ASAP is used, SOAP representatives may be fewer than n_candidates.
    Fill up to n_candidates from the LLPR-ordered pool in file order (no SOAP for fill-up).
    Structures already in asap_candidates are excluded; the rest are taken in llpr_candidates
    file order (= LLPR uncertainty descending).
    """
    if not asap_candidates:
        return asap_candidates
    llpr_atoms = ase.io.read(llpr_candidates_path, ":", format="extxyz")
    if not isinstance(llpr_atoms, list):
        llpr_atoms = [llpr_atoms]
    distinct_atoms = []
    for a in asap_candidates:
        url = a.get("url")
        fmt = a.get("format") or DataFormat.EXTXYZ
        if fmt == DataFormat.EXTXYZ:
            at_list = ase.io.read(url, ":", format="extxyz")
        else:
            at_list = ase.io.read(url, ":", format=fmt)
        if not isinstance(at_list, list):
            at_list = [at_list]
        distinct_atoms.extend(at_list)
    if len(distinct_atoms) >= n_candidates:
        return asap_candidates
    need = n_candidates - len(distinct_atoms)
    # Which indices in llpr_atoms are already in distinct (match by positions)
    def same_struct(a, b, atol=1e-5):
        return a.get_chemical_formula() == b.get_chemical_formula() and np.allclose(
            a.get_positions(), b.get_positions(), atol=atol
        )
    selected_in_llpr = set()
    for da in distinct_atoms:
        for i, la in enumerate(llpr_atoms):
            if i in selected_in_llpr:
                continue
            if same_struct(la, da):
                selected_in_llpr.add(i)
                break
    remaining_indices = [i for i in range(len(llpr_atoms)) if i not in selected_in_llpr]
    fill_indices = remaining_indices[:need]
    fill_atoms = [llpr_atoms[i] for i in fill_indices]
    final_atoms = distinct_atoms + fill_atoms
    out_path = os.path.join(work_dir, "llpr_candidates_filled.xyz")
    ase.io.write(out_path, final_atoms, format="extxyz")
    attrs = dict(asap_candidates[0].get("attrs", {}))
    return [
        {"url": out_path, "format": DataFormat.EXTXYZ, "attrs": {**attrs, "ancestor": attrs.get("ancestor", "llpr")}}
    ]


async def cll_llpr_selector(input: CllLlprSelectorInput, ctx: CllLlprSelectorContext) -> ICllSelectorOutput:
    """
    LLPR selector: compute uncertainty via LLPR-DP-MACE and select top-n structures.
    """
    executor = ctx.resource_manager.default_executor
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    executor.mkdir(work_dir)

    if not input.models:
        raise ValueError("LLPR selector requires at least one model (train_output.get_mlp_models()).")
    model_path = input.models[0].url
    cfg = input.config
    if not cfg.train_xyz and not input.training_dataset:
        raise ValueError(
            "LLPR selector requires either train_xyz in config or "
            "training_dataset from the Train stage."
        )

    candidates, new_systems, passing_rate = executor.run_python_fn(_run_llpr_select)(
        model_devi_outputs=[a.to_dict() for a in input.model_devi_data],
        model_devi_file=input.model_devi_file,
        type_map=input.type_map,
        work_dir=work_dir,
        model_path=model_path,
        sigma=cfg.sigma,
        n_candidates=cfg.n_candidates,
        training_datasets=[a.to_dict() for a in input.training_dataset] if input.training_dataset else None,
        train_xyz=cfg.train_xyz,
        val_xyz=cfg.val_xyz,
        energy_key=cfg.energy_key,
    )

    # Optional: SOAP/ASAP clustering on LLPR top-n (same interface as model_devi)
    if cfg.asap_options and not cfg.asap_options.disable:
        asap = cfg.asap_options
        candidates = executor.run_python_fn(bulk_select_distinct_structures)(
            candidates=candidates,
            descriptor_opt=asap.descriptor,
            dim_reducer_opt=asap.dim_reducer,
            cluster_opt=asap.cluster,
            type_map=input.type_map,
            work_dir=work_dir,
            limit_per_cluster=asap.limit_per_cluster,
            sort_by_energy=asap.sort_by_ssw_energy,
            workers=input.config.workers,
        )
        # Fill up to n_candidates from LLPR-ordered pool by score only (no SOAP for fill-up)
        candidates = executor.run_python_fn(_fill_up_llpr_candidates)(
            work_dir=work_dir,
            llpr_candidates_path=os.path.join(work_dir, "llpr_candidates.xyz"),
            asap_candidates=candidates,
            n_candidates=cfg.n_candidates,
            type_map=input.type_map,
        )

    return CllLlprSelectorOutput(
        candidates=[Artifact.of(**a) for a in candidates],
        new_explore_systems=[Artifact.of(**a) for a in new_systems],
        passing_rate=passing_rate,
    )


@dataclass
class CllLlprSelectorOutput(ICllSelectorOutput):
    candidates: List[Artifact]
    new_explore_systems: List[Artifact]
    passing_rate: float

    def get_model_devi_dataset(self) -> List[Artifact]:
        return self.candidates

    def get_passing_rate(self) -> float:
        return self.passing_rate

    def get_new_explore_systems(self) -> List[Artifact]:
        return self.new_explore_systems


def bulk_select_structures_by_model_devi(model_devi_outputs: List[ArtifactDict],
                                            model_devi_file: str,
                                            f_trust_lo: float,
                                            f_trust_hi: float,
                                            new_explore_system_q: float,
                                            type_map: List[str],
                                            work_dir: str,
                                            max_decent_per_traj: int,
                                            screening_fn: Optional[str],
                                            workers: int = 4,
                                            ) -> List[Tuple[Dict[str, ArtifactDict], dict]]:
    import joblib
    return joblib.Parallel(n_jobs=workers)(
        joblib.delayed(select_structures_by_model_devi)(
            model_devi_output=output,
            model_devi_file=model_devi_file,
            f_trust_lo=f_trust_lo, f_trust_hi=f_trust_hi,
            type_map=type_map,
            work_dir=os.path.join(work_dir, 'model_devi', f'{i:06}'),
            max_decent_per_traj=max_decent_per_traj,
            new_explore_system_q=new_explore_system_q,
            screening_fn=screening_fn,
        )
        for i, output in enumerate(model_devi_outputs)
    )  # type: ignore


def select_structures_by_model_devi(model_devi_output: ArtifactDict,
                                    model_devi_file: str,
                                    f_trust_lo: float,
                                    f_trust_hi: float,
                                    type_map: List[str],
                                    work_dir: str,
                                    new_explore_system_q: float,
                                    max_decent_per_traj: int,
                                    screening_fn: Optional[str],
                                    ) -> Tuple[Dict[str, ArtifactDict], dict]:
    """
    analysis the model_devi output of explore stage and select candidates

    :param next_explore_system_q: the quantile of model_devi score to select the structure for next round of exploration
    """
    os.makedirs(work_dir, exist_ok=True)
    dump_json(model_devi_output, os.path.join(work_dir, 'model_devi_output.debug.json'))

    model_devi_dir = model_devi_output['url']
    model_devi_file = model_devi_output['attrs'].pop('model_devi_file', model_devi_file)

    force_col = 'max_devi_f'
    print(f'criteria: {f_trust_lo} <= {force_col} < {f_trust_hi}')

    # get path of model_devi file
    data_format = get_data_format(model_devi_output)  # type: ignore
    if data_format in (DataFormat.LAMMPS_OUTPUT_DIR, DataFormat.LASP_LAMMPS_OUT_DIR):
        model_devi_file = os.path.join(model_devi_dir, model_devi_file)
    elif data_format == DataFormat.ANYWARE_OUTPUT_DIR:
        model_devi_file = os.path.join(model_devi_dir, 'model_devi.out')
    else:
        raise ValueError('unknown model_devi_data types')
    logger.info('start to analysis file: %s', model_devi_file)

    # load model_devi data
    with open(model_devi_file, 'r') as f:
        text = f.read()
    df = pd.read_csv(StringIO(text.lstrip('#')), delim_whitespace=True)
    # layout:
    #        step  max_devi_v  min_devi_v  avg_devi_v  max_devi_f  min_devi_f  avg_devi_f
    # 0        0    0.006793    0.000672    0.003490    0.143317    0.005612    0.026106
    # 1      100    0.006987    0.000550    0.003952    0.128178    0.006042    0.022608

    # load structures
    atoms_list = []
    if data_format == DataFormat.LAMMPS_OUTPUT_DIR:
        lammpstrj_file = model_devi_output['attrs'].pop('structures', 'traj.lammpstrj')
        atoms_list += ase.io.read(os.path.join(model_devi_dir, lammpstrj_file), ':', format='lammps-dump-text', specorder=type_map)
    elif data_format in (DataFormat.LASP_LAMMPS_OUT_DIR, DataFormat.ANYWARE_OUTPUT_DIR):
        structures_file = os.path.join(model_devi_dir, 'structures.xyz')
        atoms_list += ase.io.read(structures_file, ':', format='extxyz')
    else:
        raise ValueError('unknown model_devi_data types')

    # screening structure before model_devi analysis
    # FIXME: this should be moved to after the model_devi analysis
    if screening_fn is not None:
        if 'ssw_energy' in atoms_list[0].info:
            s_ssw_energy = pd.Series(map(lambda atoms: atoms.info['ssw_energy'], atoms_list))  # type: ignore

            # the following ssw_* methods are for the screening_fn
            # so don't need to worry about the unused warning
            ssw_energy_max = s_ssw_energy.max()
            ssw_energy_min = s_ssw_energy.min()
            def ssw_energy_quantile(q):
                # the quantile will be evaluated every time, which is not efficient
                # so here we cache the result, carefully
                return lru_cache()(s_ssw_energy.quantile)(q)
            ssw_enenrgy_quantile = ssw_energy_quantile

        # return the df row whose atoms pass the screening_fn
        _screening_fn = eval(screening_fn, locals())  # str to function
        df = df[[ _screening_fn(atoms) for atoms in atoms_list ]]


    # evaluate new found structures by their model_devi score in 3 levels: good, decent, poor
    good_df   = df[df[force_col] < f_trust_lo]
    decent_df = df[(df[force_col] >= f_trust_lo) & (df[force_col] < f_trust_hi)]
    poor_df   = df[df[force_col] >= f_trust_hi]

    # select the last frame from df whose model_devi score is less than the quantile
    # as the initial structure for next round of exploration to replace the original one
    # the next frame should be selected from good or decent frame
    # if there is no good or decent frame, use the first frame as the initial structure
    # NOTE: equal is essential to ensure the existence of next structure
    # NOTE: select the last frame can increase the diversity of structures
    _ndf = df[df[force_col] < f_trust_hi]
    if len(_ndf) == 0:
        next_df = df.head(1)  # the first frame is the initial structure
    else:
        next_df = _ndf[_ndf[force_col] <= _ndf[force_col].quantile(new_explore_system_q)].tail(1)

    stats = {
        'src': model_devi_file,
        'total': len(df),
        'good': len(good_df),
        'decent': len(decent_df),
        'poor': len(poor_df),
    }

    result: Dict[str, ArtifactDict] = {}
    # TODO: refactor the following repeating code
    # TODO: dump good may lead to storage issue, so disable it for now
    if len(good_df) > 0:
        good_file = os.path.join(work_dir, 'good.xyz')
        # ase.io.write(good_file, [atoms_list[_i] for _i in good_df.index], format='extxyz')
        result['good'] = {'url': good_file, 'format': DataFormat.EXTXYZ,  # type: ignore
                            'attrs': {**model_devi_output['attrs']}}

    if len(poor_df) > 0:
        poor_file = os.path.join(work_dir, 'poor.xyz')
        # ase.io.write(poor_file, [atoms_list[_i] for _i in poor_df.index], format='extxyz')
        result['poor'] = {'url': poor_file, 'format': DataFormat.EXTXYZ,  # type: ignore
                            'attrs': {**model_devi_output['attrs']}}
    if len(decent_df) > 0:
        decent_file = os.path.join(work_dir, 'decent.xyz')
        ase.io.write(decent_file,
                        list(limit((atoms_list[_i] for _i in decent_df.index), max_decent_per_traj)),
                        format='extxyz')
        result['decent'] = {'url': decent_file, 'format': DataFormat.EXTXYZ,  # type: ignore
                            'attrs': {**model_devi_output['attrs']}}
    if len(next_df) > 0:
        next_file = os.path.join(work_dir, 'next.xyz')
        ase.io.write(next_file, [atoms_list[_i] for _i in next_df.index], format='extxyz')
        result['next'] = {'url': next_file, 'format': DataFormat.EXTXYZ,  # type: ignore
                            'attrs': {**model_devi_output['attrs']}}
    dump_json([result, stats, list(decent_df.index), list(next_df.index)], os.path.join(work_dir, 'result.debug.json'))
    return result, stats


def bulk_select_distinct_structures(candidates: List[ArtifactDict],
                                    descriptor_opt: dict,
                                    dim_reducer_opt: dict,
                                    cluster_opt: dict,
                                    type_map: List[str],
                                    work_dir: str,
                                    limit_per_cluster: int = -1,
                                    sort_by_energy: bool = False,
                                    workers: int = 4,
                                    ) -> List[ArtifactDict]:
    try:
        dump_json(candidates, os.path.join(work_dir, 'candidates.debug.json'))
    except Exception as e:
        pass
    get_ancestor = lambda c: c['attrs']['ancestor']
    candidates = sorted(candidates, key=get_ancestor)
    inputs = []
    for i, (ancestor_key, candidate_group) in enumerate(groupby(candidates, key=get_ancestor)):
        candidate_group = list(candidate_group)
        inputs.append((candidate_group, candidate_group[0]['attrs']))

    import joblib
    return joblib.Parallel(n_jobs=workers)(
        joblib.delayed(select_distinct_structures)(
            candidates=group,
            attrs=attrs,
            descriptor_opt=descriptor_opt,
            dim_reducer_opt=dim_reducer_opt,
            cluster_opt=cluster_opt,
            type_map=type_map,
            work_dir=os.path.join(work_dir, 'asap', f'{i:06}'),
            limit_per_cluster=limit_per_cluster,
            sort_by_energy=sort_by_energy,
        ) for i, (group, attrs) in enumerate(inputs)
    )  # type: ignore


def select_distinct_structures(candidates: List[ArtifactDict],
                                attrs: dict,
                                descriptor_opt: dict,
                                dim_reducer_opt: dict,
                                cluster_opt: dict,
                                type_map: List[str],
                                work_dir: str,
                                limit_per_cluster: int = -1,
                                sort_by_energy: bool = False,
                                ):

    from .asap import get_descriptor, reduce_dimension, get_trainer, get_cluster
    from asaplib.data.xyz import ASAPXYZ

    os.makedirs(work_dir, exist_ok=True)

    dump_json(attrs, os.path.join(work_dir, 'attrs.debug.json'))
    # load structures and save it to a tmp file for ASAP to load
    atoms_list = [atoms for _, atoms in artifacts_to_ase_atoms(candidates, type_map=type_map)]


    if len(atoms_list) < 20:
        # FIXME: there are a lot of potential issue when the number of atoms is small
        # the root cause is in asaplib, which I guess has not been tested with small dataset
        selected_atoms_list = atoms_list
    elif limit_per_cluster <= 0:
        selected_atoms_list = atoms_list
    else:
        if sort_by_energy and 'ssw_energy' in atoms_list[0].info:
            atoms_list = sorted(atoms_list, key=lambda atoms: atoms.info['ssw_energy'])

        # use asaplib to group structures
        # load structures to ASAP
        tmp_structures_file = os.path.join(work_dir, '.tmp-structures.xyz')
        ase.io.write(tmp_structures_file, atoms_list, format='extxyz')
        asapxyz = ASAPXYZ(tmp_structures_file)

        # group structures
        try:
            asap_path_prefix = os.path.join(work_dir, 'asap')
            descriptors, _ = get_descriptor(asapxyz, descriptor_opt, path_prefix=asap_path_prefix)
            reduced_descriptors = reduce_dimension(descriptors, dim_reducer_opt)
            trainer = get_trainer(reduced_descriptors, cluster_opt)
            cluster_labels = get_cluster(asapxyz, reduced_descriptors, trainer, path_prefix=asap_path_prefix)

            # dump_json(cluster_labels, os.path.join(work_dir, 'cluster.debug.json'))
            selected_frames = []
            for frames in cluster_labels.values():
                if len(frames) < limit_per_cluster:
                    selected_frames += list(frames)
                else:
                    selected_frames += list(frames[:limit_per_cluster])
            selected_atoms_list = [atoms_list[i] for i in selected_frames]
        except Exception as e:
            print('asaplib failed: %s', e)
            # dump exception to file
            dump_text(traceback.format_exc(), os.path.join(work_dir, 'asaplib-exception.txt'))
            selected_atoms_list = atoms_list

    # write selected structures to file
    distinct_structures_file = os.path.join(work_dir,  'distinct_structures.xyz')
    ase.io.write(distinct_structures_file, selected_atoms_list, format='extxyz')

    output = {
        'url': distinct_structures_file,
        'format': DataFormat.EXTXYZ,
        'attrs': attrs,
    }
    dump_json(output, os.path.join(work_dir, 'output.debug.json'))
    flush_stdio()  # flush joblib stdio buffer
    return output
