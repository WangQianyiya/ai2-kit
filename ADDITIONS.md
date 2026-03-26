# 相对原 ai2kit 项目的新增内容总结

本文档总结本 fork（ai2-kit-FT）相对于上游 ai2kit 原项目所**新增**的功能与改动。

---

## 1. 概述

在 CLL（Closed-Loop Learning）工作流的 **Select（筛选）** 阶段，除原有的 **model_devi** 选择器外，新增了基于 **LLPR（Last Layer Probabilistic Regression）** 的筛选方式，并支持与 **SOAP/ASAP** 联用，实现「高不确定性 + 结构多样性」的筛选。LLPR 核心逻辑已内置于本仓库，无需外部 LLPR-DP-MACE 目录；训练数据可自动从 Train 阶段获取，无需手动配置 `train_xyz`。

---

## 2. 新增文件

| 文件 | 说明 |
|------|------|
| `ai2_kit/domain/llpr.py` | LLPR 核心算法：`get_h_mol_per_type`、`get_f_and_energy`、`build_inv_cov`、`u_raw_score`。依赖 deepmd-kit 的 `DeepPot` 与 `eval_fitting_last_layer` API。 |
| `ai2_kit/tool/llpr_cov_from_deepmd_task.py` | 训练作业内：计算协方差 `llpr_cov.npy` 并标定 `C`（写入 `llpr_meta.json`）；接受 `--sigma` 参数。 |
| `ai2_kit/tool/llpr_score_lammps_traj.py` | MD 作业内：读轨迹与 `llpr_cov.npy`，读 C 后写 `llpr.out`（含 `sigma_e_per_atom` 列）。 |

---

## 3. 修改文件与改动要点

| 文件 | 改动要点 |
|------|----------|
| `ai2_kit/domain/selector.py` | 新增 `CllLlprSelectorInputConfig`、`CllLlprSelectorInput`、`CllLlprSelectorContext`、`_load_training_data_from_deepmd_npy`、`_run_llpr_select`、`_fill_up_llpr_candidates`、`cll_llpr_selector`、`CllLlprSelectorOutput`；LLPR 可选用 SOAP/ASAP 聚类，不足 `n_candidates` 时按 LLPR 顺序补足（后补不做 SOAP）。 |
| `ai2_kit/workflow/cll_mlp.py` | `WorkflowConfig.Select` 增加 `llpr: Optional[CllLlprSelectorInputConfig]`；在 select 分支中增加 `elif workflow_config.select.llpr`，构造 `CllLlprSelectorInput`（含 `models`、`training_dataset`）并调用 `cll_llpr_selector`。 |
| `ai2_kit/domain/deepmd.py` | Split-LLPR：对 model0 在 freeze 后追加 `llpr_cov_from_deepmd_task --sigma`；`CllDeepmdInput` 新增 `llpr_sigma` 字段。 |
| `ai2_kit/domain/lammps.py` | 配置 `llpr_sigma` 时在 MD 后追加 `llpr_score_lammps_traj`；同样用 `dp` 同目录 Python。 |
| `ai2_kit/core/queue_system.py` | `QueueJobFuture.__repr__` 避免对 `JobState(bytes, Enum)` 做 `repr(dict)` 导致二次异常。 |

---

## 4. 功能说明

### 4.1 LLPR 选择器

- **作用**：用 LLPR 对探索阶段产生的结构计算能量不确定性（`sigma_e_per_atom`，eV/atom），通过上下限阈值将结构分为三类，筛选候选结构送标注。
- **三分类**（模仿 model_devi 的 `f_trust_lo` / `f_trust_hi`）：
  - **good**：`sigma_e_per_atom < e_trust_lo` — 模型预测良好，无需重新标注
  - **decent**：`e_trust_lo <= sigma_e_per_atom < e_trust_hi` — 不确定性适中，选为候选送 DFT 标注
  - **poor**：`sigma_e_per_atom >= e_trust_hi` — 偏差过大，结构可能不合理，丢弃
- **候选选取**：从 decent 中按不确定性降序取前 `n_candidates` 个；可通过 `max_decent_per_traj` 限制每条轨迹的最大 decent 数量。
- **next explore system**：从 good+decent（score < hi）中按 `new_explore_system_q` 分位数选取下轮探索初始结构（每条轨迹一个），逻辑与 model_devi 一致。
- **passing_rate**：`good / total`，用于 update walkthrough 阶段判断是否推进配置表。
- **统计报告**：每条轨迹输出 good/decent/poor 数量与比例（`llpr_stats.tsv`），格式与 model_devi 的 `stats.tsv` 一致。
- **向后兼容**：`e_trust_lo` 默认为 0，`e_trust_hi` 默认为 65535，此时所有结构均为 decent，行为退化为原来的 top-n 模式。
- **输入**：探索数据（`model_devi_data`）、训练好的模型（`models[0]`）、训练数据集（用于建协方差与校准，见下）、`type_map` 等。
- **输出**：与原有 selector 一致，实现 `ICllSelectorOutput`（`get_model_devi_dataset`、`get_passing_rate`、`get_new_explore_systems`）。

### 4.2 自动使用 Train 阶段训练数据

- **默认**：不配置 `train_xyz` 时，LLPR 直接使用 `train_output.get_training_dataset()`（DeepMD NPY 格式），通过 dpdata 读取，用于构建 LLPR 协方差矩阵和校准常数 C。
- **可选**：仍可手动设置 `train_xyz`（及 `val_xyz`、`energy_key`）使用 XYZ 文件，兼容旧用法。

### 4.3 SOAP/ASAP 联用

- **配置**：在 `select.llpr` 下设置 `asap_options`（与 model_devi 的 ASAP 同套接口：`descriptor`、`dim_reducer`、`cluster`、`limit_per_cluster` 等）。
- **流程**：先按 LLPR 取 top-`n_candidates` 作为候补池 → 对该池做 SOAP 聚类 → 每簇取代表（默认每簇 1 个）→ 若代表数不足 `n_candidates`，从候补池中按 LLPR 顺序补足；**补足阶段不再做 SOAP**。
- **候补池大小**：等于 `n_candidates`（若探索总帧数不足则为总帧数）。

### 4.4 内联 LLPR 核心，无外部目录依赖

- **原状**：曾依赖外部 `LLPR-DP-MACE` 目录（`llpr_dp_mace_dir`），需在配置中指定路径。
- **现状**：LLPR 核心已迁入 `ai2_kit/domain/llpr.py`，仅依赖 **deepmd-kit**（`deepmd.infer.DeepPot` 及 `eval_fitting_last_layer`），不再需要 `llpr_dp_mace_dir`。

### 4.5 Split LLPR（训练协方差 + C 标定 + MD 内评分）

- **Train**：在 DeepMD 训练/冻结后，于同一 GPU 作业内对 **model0** 运行 `python -m ai2_kit.tool.llpr_cov_from_deepmd_task --sigma <sigma>`，在任务目录写出 `llpr_cov.npy` 及 `llpr_meta.json`（含标定常数 `C`）。
- **C 标定**：使用与训练相同的数据集，按 `C = mean((err/N)^2 / u_raw)` 计算（`err = E_pred - E_DFT`，`N` = 原子数），使得 `sqrt(C * u_raw)` 的量纲为 **eV/atom**，可直接设阈值。
- **Explore (lammps)**：LAMMPS 结束后在同一作业内运行 `llpr_score_lammps_traj`，写出 `llpr.out`。若 `llpr_meta.json` 含 `C`，则输出三列：`step`、`u_raw`、`sigma_e_per_atom`（= `sqrt(C * u_raw)`，eV/atom）。
- **Selector 快路径**：解析 `llpr.out` 时优先使用第三列（`sigma_e_per_atom`）；若只有两列则回退到 `u_raw`。读取后按 `e_trust_lo` / `e_trust_hi` 将结构分为 good/decent/poor 三类，只取 decent 候选。
- **Python 解释器**：上述两步在生成的 sbatch 中使用 `"$(dirname "$(command -v dp)")/python"`，与当前环境中 **`dp`（deepmd-kit）同目录的 Python** 一致；避免非交互 `source activate` 后裸写 `python` 仍指向 Anaconda base、导致 `ModuleNotFoundError: deepmd`。
- **前提**：作业 `setup` 后 `dp` 须在 `PATH` 中（与运行 `dp train` / `dp freeze` 相同）。

---

## 5. 配置示例

**LLPR 带阈值筛选（推荐）：**

```yaml
select:
  llpr:
    e_trust_lo: 0.005          # eV/atom，低于此为 good
    e_trust_hi: 0.05           # eV/atom，高于此为 poor
    n_candidates: 50           # decent 最大选取数
    new_explore_system_q: 0.25 # 下轮探索初始结构分位数
    max_decent_per_traj: -1    # 每条轨迹 decent 上限，-1 不限
    sigma: 0.01
```

**LLPR 不设阈值（top-n 模式，向后兼容）：**

```yaml
select:
  llpr:
    n_candidates: 50
    sigma: 0.01
```

**LLPR + SOAP 联用：**

```yaml
select:
  llpr:
    e_trust_lo: 0.005
    e_trust_hi: 0.05
    n_candidates: 50
    sigma: 0.01
    asap_options:
      disable: false
      limit_per_cluster: 1
      descriptor:
        soap:
          preset: minimal
      dim_reducer:
        pca: {}
      cluster:
        dbscan: {}
    workers: 4
```

无需再配置 `train_xyz` 或 `llpr_dp_mace_dir`。

---

## 6. 依赖

- **deepmd-kit**：需支持 `DeepPot.eval_fitting_last_layer`。
- **dpdata**：用于 selector 在登录节点从 DeepMD NPY 加载训练数据（可选）；训练作业内 LLPR 协方差工具不依赖 dpdata。
- **ase**：用于结构读写与坐标/类型/盒子转换。
- 使用 SOAP 时与原有 model_devi+ASAP 相同（asaplib 等）。

---

## 7. 相关提交（本 fork）

- `1ebd429` Add LLPR selector entry and workflow hook  
- `69f8310` Add LLPR fill-up by uncertainty only after ASAP (no SOAP for fill-up)  
- `b8ac66f` Auto-load training dataset for LLPR covariance from Train stage  
- `1aefdba` Inline LLPR core into ai2_kit/domain/llpr.py, remove external dependency  
