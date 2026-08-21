#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLGAFormer vs SOTA Models Comparison Study (Experiment 2)
- Compares PLGAFormer against state-of-the-art time series forecasting models
- Uses unified configuration system for consistent parameter management
- Goal: Demonstrate PLGAFormer's superiority over existing SOTA methods
"""

import warnings
warnings.filterwarnings('ignore', message='.*flash attention.*', category=UserWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
from collections import OrderedDict
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import json
import hashlib
import copy
import csv
import shutil
 
# Set matplotlib to non-interactive mode
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 0

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Import unified configuration and models
import sys
from pathlib import Path

# 项目根目录：所有数据/结果路径基于此，与当前工作目录无关
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import (
    ECEFTrajectoryLoss, HGVPhysicsLoss, HGVConfig,
    create_registered_model, get_supported_model_types
)
from utils.repro import set_global_seed, seed_worker, build_torch_generator
from utils.train_protocol import (
    build_adamw_optimizer,
    build_warmup_cosine_scheduler,
    should_update_best,
    build_optimizer_by_profile,
    build_scheduler_by_profile,
    clip_gradients_with_finite_check,
)
from utils.experiment_io import get_experiment_dirs, save_run_metadata
from utils.seq2seq_protocol import (
    align_source_position_scale,
    build_supervision_windows as build_seq2seq_supervision_windows,
    training_forward as seq2seq_training_forward,
)
from utils.inference_protocol import predict_by_eval_protocol as unified_predict_by_eval_protocol
from utils.physics_prior_cache import (
    build_cached_physics_prior_loader,
    unpack_batch_with_optional_prior,
)
from utils.model_provenance import (
    get_model_provenance,
    build_comparison_model_provenance,
    validate_comparison_model_types,
)
from utils.trajectory_protocol import (
    dataset_protocol_name,
    get_split_maneuver_labels,
    validate_npz_trajectory_splits,
)
from utils.trajectory_metrics import (
    aggregate_window_metrics_by_trajectory,
    holm_adjust,
    paired_permutation_test,
)
from data_generation.data_paths import (
    get_dataset_npz_path,
    get_input_scaler_path,
    get_output_scaler_path,
    get_processed_data_dir,
)
from data_provider.hgv_data import load_hgv_dataset

# ======================================================================================
# 统一配置系统 (Unified Configuration System)
# ======================================================================================
# 随机种子：每轮运行仅在一处设置——主循环内「每 run」开始时调用 set_random_seed(seed)，
# 数据加载时传入同一 seed 给 DataLoader generator，不再在其它位置写死 manual_seed。

# 获取统一配置（唯一真相源：全部从 HGVConfig 直接读取，不做 experiment_type 覆盖）
_base_train_config = HGVConfig.get_train_config()
MODEL_CONFIG = HGVConfig.get_model_config('plgaformer')
PHYSICS_CONFIG = HGVConfig.get_physics_config()

# ======================================================================================
# SOTA实验专用配置 - 符合顶刊标准
# ======================================================================================
# 🎯 SOTA实验 vs 消融实验的区别：
#   - SOTA需要完全收敛（更多epochs）
#   - SOTA需要完整评估（全量数据 + 多预测长度）
#   - SOTA需要更高统计可信度（5次运行）
#
# 顶刊参考：
#   - PatchTST/iTransformer: 100 epochs, patience=10-15
#   - 本实验: epochs/patience/warmup 从 get_train_config() 读取（与 __init__.py 一致）

# ======================================================================================
# 多次运行配置 - 符合顶刊标准
# ======================================================================================
# 🎯 设计理念：通过多次独立运行消除随机性影响，提供统计显著性
# 
# 顶刊标准：
# - NeurIPS/ICML/ICLR: 3-5次运行，报告 mean ± std
# - TPAMI/TIP: 5次以上，需统计检验 (p < 0.05)

NUM_RUNS = _base_train_config['num_runs']
RANDOM_SEEDS = _base_train_config['random_seeds']

# Report configuration
REPORT_ON_PHYSICAL_SCALE = True   # 若 True：FDE/ADE 在物理空间（米）；主表可注明 "FDE (m)", "ADE (m)"
REPORT_MSE_ON_SCALED = True      # 若 True：MSE/MAE 在标准化空间（无量纲）；若 False：物理空间（MSE 单位 m² 等）
NUMERICAL_GUARDS = True
FORCE_ONESHOT_PREDICTION = True
# 顶会协议对齐：评估时自回归模型默认不使用未来真值作seed，避免信息泄漏。
# 可选: "zero"（默认）, "gt_first"（兼容旧结果）
EVAL_AR_SEED_MODE = _base_train_config.get('eval_ar_seed_mode', 'zero')
# 评估协议:
# - strict_autoregressive: 自回归模型按 seed rollout，SOTA模型一次性预测
# - official_like_decoder: 对支持 decoder_context 的模型使用 label_len+zero 语义
EVAL_PROTOCOL = _base_train_config.get('eval_protocol', 'strict_autoregressive')
# 训练监督协议:
# - shifted_next_step: tgt[:-1] -> tgt[1:]（当前项目兼容模式）
# - pred_window: 统一 pred_len 监督窗口（顶会语义）
TRAIN_SUPERVISION_PROTOCOL = _base_train_config.get('train_supervision_protocol', 'shifted_next_step')
# 优化器协议:
# - enhanced: AdamW + warmup cosine
# - official_like: Adam + type1-lrdecay
OPTIMIZER_PROFILE = _base_train_config.get('optimizer_profile', 'enhanced')
# 严格复现模式：True 时启用严格 deterministic（可能影响部分模型可训练性）
STRICT_REPRO_MODE = _base_train_config.get('strict_repro_mode', False)
# 指标与单位说明（正文主表一致）：MSE/MAE 见 REPORT_MSE_ON_SCALED；FDE/ADE 恒为笛卡尔位移误差，单位米 (m)

VERBOSE = _base_train_config.get('verbose', False)
def info(msg):
    if VERBOSE:
        print(msg)


def _is_cuda_illegal_access_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        'illegal memory access' in msg
        or 'cuda error: an illegal memory access was encountered' in msg
    )
info(f"\n{'='*60}")
info(f"SOTA COMPARISON EXPERIMENT CONFIGURATION (顶刊标准)")
info(f"{'='*60}")
info(f"训练轮数: {_base_train_config['epochs']} epochs")
info(f"早停Patience: {_base_train_config.get('early_stopping_patience', 10)}")
info(f"Warmup: {_base_train_config.get('warmup_epochs', 0)} epochs")
info(f"独立运行次数: {NUM_RUNS}")
info(f"随机种子: {RANDOM_SEEDS}")
info(f"Batch Size: {_base_train_config['batch_size']}")
info(f"Learning Rate: {_base_train_config['learning_rate']}")
info(f"Model Dimension: {MODEL_CONFIG['d_model']}")
info(f"Attention Heads: {MODEL_CONFIG['nhead']}")
info(f"Physics Loss Weight: {PHYSICS_CONFIG['alpha']}")
info(f"{'='*60}\n")

# Paper-facing comparison matrix.
COMPARISON_MODELS = OrderedDict([
    ("Transformer (baseline)", {
        'model_type': 'transformer',
        'description': 'Standard Transformer baseline (data-driven only, no physics loss)',
        'physics_loss_weight': 0.0,
        'innovations': []
    }),
    ("PLGAFormer (proposed)", {
        'model_type': 'plgaformer',
        'description': 'Validation-selected PLGAFormer with confidence-gated rotating-Earth prior fusion',
        'physics_loss_weight': 0.0,
        'innovations': [
            'C: identified rotating-Earth 3-DOF prior with state-adaptive bounded fusion',
        ]
    }),
    ("Spherical kinematics", {
        'model_type': 'kinematic',
        'description': 'Parameter-free frozen-state analytical spherical kinematic baseline',
        'physics_loss_weight': 0.0,
        'innovations': []
    }),
    ("Rotating-Earth 3-DOF", {
        'model_type': 'rotating_3dof',
        'description': 'Parameter-free identified rotating-Earth 3-DOF RK4 propagation',
        'physics_loss_weight': 0.0,
        'innovations': [],
    }),
    ("DLinear", {
        'model_type': 'dlinear',
        'description': 'Decomposition-linear forecasting baseline (AAAI 2023)',
        'physics_loss_weight': 0.0,
        'innovations': []
    }),
    ("PatchTST", {
        'model_type': 'patchtst',
        'description': 'PatchTST (ICLR 2023)',
        'physics_loss_weight': 0.0,
        'innovations': []
    }),
    ("iTransformer", {
        'model_type': 'itransformer',
        'description': 'iTransformer inverted (ICLR 2024)',
        'physics_loss_weight': 0.0,
        'innovations': []
    }),
])

# 预测长度 - 从统一配置 get_train_config() 直接读取
def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _model_aliases(model_name: str, model_config: dict) -> set[str]:
    compact_name = (
        model_name.lower()
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
        .replace("_", "")
    )
    model_type = str(model_config.get("model_type", "")).lower()
    compact_type = model_type.replace("-", "").replace("_", "").replace(" ", "")
    aliases = {compact_name, model_type, compact_type}
    return aliases


def _selected_comparison_models() -> OrderedDict:
    raw = os.getenv("HGV_SOTA_MODELS")
    if raw is None or raw.strip() == "":
        return OrderedDict(COMPARISON_MODELS)

    requested = {
        item.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        for item in raw.split(",")
        if item.strip()
    }
    selected = OrderedDict()
    matched = set()
    for model_name, model_config in COMPARISON_MODELS.items():
        aliases = _model_aliases(model_name, model_config)
        matches = requested & aliases
        if matches:
            selected[model_name] = model_config
            matched.update(matches)

    unmatched = requested - matched
    if unmatched:
        raise ValueError(
            f"HGV_SOTA_MODELS did not match any comparison model: {raw}. "
            f"Unmatched selections: {sorted(unmatched)}. "
            f"Available model_types: {[cfg['model_type'] for cfg in COMPARISON_MODELS.values()]}"
        )
    return selected


PREDICTION_HORIZONS = _base_train_config.get('prediction_horizons', [32, 64, 128, 256])
EVAL_METRICS = ['mse', 'mae', 'rmse', 'fde', 'ade', 'rmse_cart_m', 'mse_cart_m2', 'mae_cart_m']
_RUN_SIGNATURE = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_run_signature() -> str:
    """Hash data, implementation, and training protocol for cache isolation."""
    global _RUN_SIGNATURE
    if _RUN_SIGNATURE is not None:
        return _RUN_SIGNATURE
    tracked_files = {
        "dataset": get_dataset_npz_path(PROJECT_ROOT),
        "input_scaler": get_input_scaler_path(PROJECT_ROOT),
        "output_scaler": get_output_scaler_path(PROJECT_ROOT),
        "plgaformer": PROJECT_ROOT / "models" / "plgaformer.py",
        "model_factory": PROJECT_ROOT / "models" / "model_factory.py",
        "baseline_models": PROJECT_ROOT / "models" / "baseline_models.py",
        "generator": PROJECT_ROOT / "data_generation" / "data_generator.py",
        "experiment": Path(__file__).resolve(),
    }
    selected_model_types = {
        str(config.get("model_type", "")).lower()
        for config in _selected_comparison_models().values()
    }
    tslib_types = selected_model_types & {"transformer", "dlinear", "patchtst", "itransformer"}
    tslib_root = os.getenv("HGV_TSLIB_ROOT", "").strip()
    TSLIB_OFFICIAL_COMMIT = None
    if tslib_types:
        from models.public_baselines import TSLIB_OFFICIAL_COMMIT, tslib_signature_files

        if tslib_root:
            for relative, path in tslib_signature_files(tslib_types, root=tslib_root).items():
                tracked_files[f"tslib:{relative}"] = path
    payload = {
        "files": {name: _sha256_file(path) for name, path in tracked_files.items()},
        "train_config": _base_train_config,
        "model_config": MODEL_CONFIG,
        "physics_config": PHYSICS_CONFIG,
        "evaluation_metrics": EVAL_METRICS,
        "selected_model_types": sorted(selected_model_types),
        "tslib_commit": TSLIB_OFFICIAL_COMMIT if tslib_types else None,
    }
    serialized = json.dumps(convert_to_serializable(payload), sort_keys=True, separators=(",", ":"))
    _RUN_SIGNATURE = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return _RUN_SIGNATURE


def _partial_key(seed: int, model_name: str, model_config: dict) -> str:
    horizons = ",".join(str(h) for h in PREDICTION_HORIZONS)
    return (
        f"seed={int(seed)}|model={model_config['model_type']}|name={model_name}|"
        f"pred_len={_base_train_config.get('pred_len')}|horizons={horizons}|"
        f"signature={_current_run_signature()}"
    )


def _load_partial_runs(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 2, "runs": OrderedDict()}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {"schema_version": 2, "runs": OrderedDict()}
    runs = payload.get("runs", {})
    if isinstance(runs, list):
        runs = {
            item.get("key", f"legacy_{idx}"): item
            for idx, item in enumerate(runs)
            if isinstance(item, dict)
        }
    payload["runs"] = OrderedDict(runs)
    payload.setdefault("schema_version", 2)
    return payload


def _save_partial_runs(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(payload), f, indent=2, ensure_ascii=False)


def _upsert_partial_run(
    payload: dict,
    key: str,
    seed: int,
    model_name: str,
    model_config: dict,
    results: dict,
    duration_sec: float,
    checkpoint_path: str | None = None,
    training_history: dict | None = None,
    model_audit: dict | None = None,
) -> None:
    payload.setdefault("runs", OrderedDict())
    payload["runs"][key] = {
        "key": key,
        "seed": int(seed),
        "model_name": model_name,
        "model_type": model_config["model_type"],
        "pred_len": int(_base_train_config.get("pred_len", 0)),
        "prediction_horizons": [int(h) for h in PREDICTION_HORIZONS],
        "run_signature": _current_run_signature(),
        "results": convert_to_serializable(results),
        "duration_sec": float(duration_sec),
        "checkpoint_path": checkpoint_path,
        "training_history": convert_to_serializable(training_history or {}),
        "model_audit": convert_to_serializable(model_audit or {}),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _multi_run_from_partial(payload: dict, selected_models: OrderedDict) -> dict:
    multi = {model_name: [] for model_name in selected_models.keys()}
    runs = payload.get("runs", {})
    for seed in RANDOM_SEEDS:
        for model_name, model_config in selected_models.items():
            item = runs.get(_partial_key(seed, model_name, model_config))
            if item and item.get("results") is not None:
                normalized_results = {}
                for horizon, metrics in item["results"].items():
                    try:
                        normalized_horizon = int(horizon)
                    except (TypeError, ValueError):
                        normalized_horizon = horizon
                    normalized_results[normalized_horizon] = metrics
                multi[model_name].append(normalized_results)
    return multi

# 数据子集比例（与消融 30% 类似）：None = 100%，0.5 = 50%；多轮运行使用同一子集（固定种子 42）
SOTA_DATA_SUBSET_RATIO = 1.0

# ======================================================================================
# 统计分析函数 - 多次运行结果的统计处理
# ======================================================================================

from scipy import stats

def set_random_seed(seed):
    """每轮运行开始时唯一调用的种子设置入口，确保可复现性；DataLoader 使用同一 seed 的 generator。
    注意：使用 warn_only=True 允许某些操作（如 upsample_linear1d）的非确定性实现。
    """
    # 使用统一入口，避免各实验脚本出现细微差异。
    set_global_seed(seed, deterministic=True, strict_deterministic=STRICT_REPRO_MODE)

def compute_statistics(multi_run_results):
    """
    计算多次运行的统计量
    
    Args:
        multi_run_results: dict, {model_name: [{run1_metrics}, {run2_metrics}, ...]}
    
    Returns:
        stats_results: dict, {model_name: {horizon: {metric: {'mean': x, 'std': y, 'values': [...]}}}}
    """
    stats_results = {}
    
    for model_name, run_results in multi_run_results.items():
        stats_results[model_name] = {}
        
        # 提取所有预测长度的指标
        for horizon in PREDICTION_HORIZONS:
            stats_results[model_name][horizon] = {}
            
            for metric in EVAL_METRICS:
                values = []
                for run_result in run_results:
                    if horizon in run_result and metric in run_result[horizon]:
                        values.append(run_result[horizon][metric])
                
                if len(values) > 0:
                    stats_results[model_name][horizon][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values
                    }
    
    return stats_results

def perform_statistical_tests(stats_results, baseline_name="Transformer (baseline)"):
    """
    执行统计显著性检验（配对t-test）
    
    Args:
        stats_results: 从compute_statistics返回的统计结果
        baseline_name: 基线模型名称
    
    Returns:
        test_results: dict, {model_name: {horizon: {metric: {'t_stat': t, 'p_value': p, 'significant': bool}}}}
    """
    test_results = {}
    
    if baseline_name not in stats_results:
        print(f"⚠️ 基线模型 '{baseline_name}' 不在结果中，跳过统计检验")
        return test_results
    
    baseline_stats = stats_results[baseline_name]
    
    for model_name, model_stats in stats_results.items():
        if model_name == baseline_name:
            continue
        
        test_results[model_name] = {}
        
        for horizon in PREDICTION_HORIZONS:
            test_results[model_name][horizon] = {}
            
            for metric in EVAL_METRICS:
                if (horizon in baseline_stats and metric in baseline_stats[horizon] and
                    horizon in model_stats and metric in model_stats[horizon]):
                    
                    baseline_values = baseline_stats[horizon][metric]['values']
                    model_values = model_stats[horizon][metric]['values']
                    
                    # 确保样本数相同
                    min_len = min(len(baseline_values), len(model_values))
                    if min_len >= 2:  # 至少需要2个样本进行t检验
                        try:
                            t_stat, p_value = stats.ttest_rel(
                                baseline_values[:min_len], 
                                model_values[:min_len]
                            )
                            test_results[model_name][horizon][metric] = {
                                't_stat': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'highly_significant': p_value < 0.01
                            }
                        except Exception as e:
                            print(f"⚠️ t检验失败 ({model_name}, {horizon}, {metric}): {e}")
    
    return test_results

def perform_trajectory_level_tests(multi_run_results, baseline_name="Transformer (baseline)"):
    """Perform paired inference with held-out trajectories as independent units.

    Seed repetitions estimate optimization variability; they are averaged within
    each trajectory before the paired sign-flip tests. Holm correction controls
    family-wise error across all model, horizon, and displacement-metric tests.
    """
    if baseline_name not in multi_run_results:
        print(f"WARNING: baseline '{baseline_name}' is absent; skipping inferential tests.")
        return {}

    def average_by_trajectory(run_results, horizon, metric):
        values_by_id = {}
        for run_result in run_results:
            horizon_result = run_result.get(horizon, {})
            for trajectory_id, item in horizon_result.get("trajectory_metrics", {}).items():
                values_by_id.setdefault(int(trajectory_id), []).append(float(item[metric]))
        return {key: float(np.mean(values)) for key, values in values_by_id.items()}

    test_results = {}
    pending_adjustment = []
    baseline_runs = multi_run_results[baseline_name]
    for model_name, model_runs in multi_run_results.items():
        if model_name == baseline_name:
            continue
        test_results[model_name] = {}
        for horizon in PREDICTION_HORIZONS:
            test_results[model_name][horizon] = {}
            for metric in ("ade", "fde"):
                baseline_values = average_by_trajectory(baseline_runs, horizon, metric)
                model_values = average_by_trajectory(model_runs, horizon, metric)
                common_ids = sorted(set(baseline_values) & set(model_values))
                if len(common_ids) < 2:
                    continue
                comparison = paired_permutation_test(
                    np.asarray([model_values[key] for key in common_ids]),
                    np.asarray([baseline_values[key] for key in common_ids]),
                    seed=2026 + int(horizon),
                )
                comparison["independent_unit"] = "held-out source trajectory"
                comparison["seed_aggregation"] = "mean within trajectory before inference"
                comparison["trajectory_ids"] = common_ids
                test_results[model_name][horizon][metric] = comparison
                pending_adjustment.append((comparison, float(comparison["p_value"])))

    if pending_adjustment:
        adjusted = holm_adjust([item[1] for item in pending_adjustment])
        for (comparison, _), p_adjusted in zip(pending_adjustment, adjusted):
            comparison["p_value_holm"] = float(p_adjusted)
            comparison["significant_fwer_0_05"] = bool(p_adjusted < 0.05)
    return test_results


def print_statistical_summary(stats_results, test_results, baseline_name="Transformer (baseline)"):
    """
    打印统计分析摘要
    """
    print(f"\n{'='*100}")
    print(f"📊 统计分析摘要 ({NUM_RUNS}次独立运行)")
    print(f"{'='*100}")
    
    for horizon in PREDICTION_HORIZONS:
        print(f"\n📏 预测长度: {horizon} 步")
        print(f"{'-'*100}")
        # 表头与论文主表一致：MSE/MAE 标注是否 scaled，FDE/ADE 标注单位 (m)
        mse_label = "MSE (scaled)" if REPORT_MSE_ON_SCALED else "MSE"
        mae_label = "MAE (scaled)" if REPORT_MSE_ON_SCALED else "MAE"
        print(f"{'模型':<25} {mse_label:<18} {mae_label:<18} {'FDE (m)':<18} {'FDE p-Holm':<12}")
        print(f"{'-'*100}")
        
        for model_name, model_stats in stats_results.items():
            if horizon not in model_stats:
                continue
            
            mse_stats = model_stats[horizon].get('mse', {})
            mae_stats = model_stats[horizon].get('mae', {})
            fde_stats = model_stats[horizon].get('fde', {})
            
            mse_str = f"{mse_stats.get('mean', 0):.4f}±{mse_stats.get('std', 0):.4f}"
            mae_str = f"{mae_stats.get('mean', 0):.4f}±{mae_stats.get('std', 0):.4f}"
            fde_str = f"{fde_stats.get('mean', 0):.1f}±{fde_stats.get('std', 0):.1f}"
            
            # Trajectory-level FDE comparison against the Transformer baseline.
            p_value_str = "-"
            if model_name != baseline_name and model_name in test_results:
                if horizon in test_results[model_name] and 'fde' in test_results[model_name][horizon]:
                    comparison = test_results[model_name][horizon]['fde']
                    p_val = comparison.get('p_value_holm', comparison['p_value'])
                    if p_val < 0.01:
                        p_value_str = f"{p_val:.4f}**"
                    elif p_val < 0.05:
                        p_value_str = f"{p_val:.4f}*"
                    else:
                        p_value_str = f"{p_val:.4f}"
            
            print(f"{model_name:<25} {mse_str:<18} {mae_str:<18} {fde_str:<18} {p_value_str:<12}")
        
        print(f"{'-'*100}")
    
    print(
        "注: mean ± std 跨随机种子计算；p 值来自留出源轨迹上的配对符号置换检验，"
        "并对全部模型、预测长度和 ADE/FDE 检验作 Holm 校正。"
    )

def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型以便JSON序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj

# ======================================================================================
# 模型创建工厂函数
# ======================================================================================

def create_model(model_type, input_dim=6, device=torch.device('cpu'), plgaformer_kwargs=None):
    """
    创建不同类型的模型 - 唯一入口，与 COMPARISON_MODELS 一致。
    统一走 models.model_factory.create_registered_model。
    """
    model_type = str(model_type).lower().strip()
    supported = set(get_supported_model_types())
    if model_type not in supported:
        raise ValueError(
            f"Unsupported model_type for SOTA comparison: {model_type}. "
            f"Supported: {sorted(supported)}"
        )
    if model_type in {'transformer', 'dlinear', 'patchtst', 'itransformer'}:
        tslib_root = os.getenv("HGV_TSLIB_ROOT", "").strip()
        if not tslib_root:
            raise RuntimeError(
                "Paper-facing public baselines require the pinned TSLib checkout. "
                "Pass --tslib-root or set HGV_TSLIB_ROOT before creating the model."
            )
        public_kwargs = dict(plgaformer_kwargs or {})
        public_kwargs.update({"source": "tslib", "tslib_root": tslib_root})
        plgaformer_kwargs = public_kwargs
    return create_registered_model(
        model_type=model_type,
        input_dim=input_dim,
        device=device,
        plgaformer_kwargs=plgaformer_kwargs,
    )


def _plgaformer_physical_kwargs(input_scaler, output_scaler) -> dict:
    """Build the exact physical-scaler contract used for every model reconstruction."""
    source_mean = np.concatenate([output_scaler.mean_, input_scaler.mean_[3:]])
    source_scale = np.concatenate([output_scaler.scale_, input_scaler.scale_[3:]])
    return {
        'input_scaler_mean': source_mean,
        'input_scaler_scale': source_scale,
        'output_scaler_mean': output_scaler.mean_,
        'output_scaler_scale': output_scaler.scale_,
        'sampling_interval_s': float(_base_train_config.get('sampling_interval_s', 1.0)),
        'require_physical_scaler': True,
    }


def _model_reconstruction_kwargs(model_type, input_scaler, output_scaler):
    kwargs = {}
    if model_type in {'plgaformer', 'kinematic', 'rotating_3dof'}:
        kwargs.update(_plgaformer_physical_kwargs(input_scaler, output_scaler))
    if model_type in {'transformer', 'dlinear', 'patchtst', 'itransformer'}:
        tslib_root = os.getenv("HGV_TSLIB_ROOT", "").strip()
        if tslib_root:
            kwargs.update({"source": "tslib", "tslib_root": tslib_root})
    return kwargs or None

def get_model_save_path(model_type, models_save_dir):
    """
    获取模型保存路径 - 统一路径命名逻辑
    
    Args:
        model_type: 模型类型 (如 'plgaformer', 'transformer', 'patchtst' 等)
        models_save_dir: 模型保存目录
    
    Returns:
        模型保存路径，如果模型不应该保存则返回 None
    """
    path_mapping = {
        'plgaformer': 'best_plgaformer_sota.pth',
        'transformer': 'best_transformer_sota.pth',
        'patchtst': 'best_patchtst_sota.pth',
        'itransformer': 'best_itransformer_sota.pth',
        'dlinear': 'best_dlinear_sota.pth',
        'kinematic': None,
        'rotating_3dof': None,
    }
    
    filename = path_mapping.get(model_type)
    if filename is None:
        if model_type in {'kinematic', 'rotating_3dof'}:
            return None
        return os.path.join(models_save_dir, f"best_{model_type}_sota.pth")
    
    return os.path.join(models_save_dir, filename)

# ======================================================================================
# 数据加载
# ======================================================================================

def _stable_indices(total: int, ratio: float | None, rng_seed: int = 42) -> np.ndarray:
    if ratio is None or ratio >= 1.0:
        return np.arange(total)
    n = max(1, min(int(total * ratio), total))
    rng = np.random.default_rng(rng_seed)
    return np.sort(rng.choice(total, size=n, replace=False))


def _build_scaler_signature(
    data: dict,
    subset_ratio: float | None,
    train_idx: np.ndarray,
) -> dict:
    idx_bytes = train_idx.astype(np.int64).tobytes()
    return {
        "subset_ratio": None if subset_ratio is None else float(subset_ratio),
        "dataset_protocol": dataset_protocol_name(data),
        "train_shape": list(data["X_train"].shape),
        "target_shape": list(data["y_train"].shape),
        "train_idx_hash": hashlib.sha256(idx_bytes).hexdigest(),
    }


def _scaler_signature_matches(meta_path: Path, expected: dict) -> bool:
    if not meta_path.exists():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            current = json.load(f)
    except Exception:
        return False
    return current == expected


def load_and_prepare_data(batch_size, seed=None, subset_ratio=None):
    """加载基于物理仿真的数据集；subset_ratio 为 None 时使用全量，否则使用该比例子集（固定种子，多轮一致）。"""
    info("="*50)
    info("Loading physics-based dataset for SOTA comparison...")
    
    try:
        # 加载数据集（路径基于 PROJECT_ROOT）
        data_path = str(get_dataset_npz_path(PROJECT_ROOT))
        info(f"📂 Loading dataset: {data_path}")
        data_bundle = load_hgv_dataset(data_path, require_trajectory_level=True)
        data = data_bundle.raw
        _base_train_config["sampling_interval_s"] = float(
            data_bundle.time_metadata["sampling_interval_s"]
        )
        
        info(f"数据集规模: 训练集{data['X_train'].shape}, 验证集{data['X_val'].shape}, 测试集{data['X_test'].shape}")
        protocol = data_bundle.protocol
        split_report = data_bundle.split_report
        if split_report is not None:
            print(f"📌 数据协议: {protocol} | 轨迹级划分重叠检查: {split_report}")
        else:
            print(f"📌 数据协议: {protocol}")
        
        # 验证数据集规模
        total_samples = len(data['X_train']) + len(data['X_val']) + len(data['X_test'])
        info(f"  总样本数: {total_samples:,}")
        info(f"  输入序列长度: {data['X_train'].shape[1]}")
        info(f"  预测序列长度: {data['y_train'].shape[1]}")
        info(f"  输入特征维度: {data['X_train'].shape[2]}")
        info(f"  输出特征维度: {data['y_train'].shape[2]}")
        
        # 先确定子集索引（若启用），确保 scaler 与训练子集一致，避免“全量拟合+子集训练”偏差。
        train_subset_idx = _stable_indices(len(data['X_train']), subset_ratio, rng_seed=42)
        val_subset_idx = _stable_indices(len(data['X_val']), subset_ratio, rng_seed=42)
        test_subset_idx = _stable_indices(len(data['X_test']), subset_ratio, rng_seed=42)

        # 加载或创建标准化器（带签名校验）
        _data_dir = get_processed_data_dir(PROJECT_ROOT)
        scaler_path = str(get_input_scaler_path(PROJECT_ROOT))
        output_scaler_path = str(get_output_scaler_path(PROJECT_ROOT))
        scaler_meta_path = Path(
            os.getenv(
                "HGV_SCALER_SIGNATURE_PATH",
                str(Path(_data_dir) / "scaler_signature_exp1_sota.json"),
            )
        ).expanduser().resolve()
        scaler_meta_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_signature = _build_scaler_signature(
            data=data,
            subset_ratio=subset_ratio,
            train_idx=train_subset_idx,
        )

        scaler = None
        output_scaler = None
        out_dim = data['y_train'].shape[-1]
        signature_ok = _scaler_signature_matches(scaler_meta_path, scaler_signature)
        if signature_ok:
            try:
                scaler = joblib.load(scaler_path)
                output_scaler = joblib.load(output_scaler_path)
                if output_scaler.mean_.shape[0] != out_dim:
                    raise ValueError("output_scaler dim mismatch")
                info(f"✅ 已加载签名匹配的标准化器: {scaler_path}, {output_scaler_path}")
            except Exception:
                scaler = None
                output_scaler = None

        if scaler is None or output_scaler is None:
            info("🔧 创建新的输入/输出标准化器（按当前训练子集拟合）...")
            scaler = StandardScaler()
            train_input_data = data['X_train'][train_subset_idx].reshape(-1, data['X_train'].shape[-1])
            scaler.fit(train_input_data)

            output_scaler = StandardScaler()
            train_output_data = data['y_train'][train_subset_idx].reshape(-1, out_dim)
            output_scaler.fit(train_output_data)

            _data_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            joblib.dump(output_scaler, output_scaler_path)
            with scaler_meta_path.open("w", encoding="utf-8") as f:
                json.dump(scaler_signature, f, indent=2, ensure_ascii=False)
            info(f"✅ 新标准化器与签名已保存: {scaler_meta_path}")
        
        info("🔄 开始批量标准化处理...")
        
        # 高效批量标准化输入数据
        info("  - 标准化输入数据...")
        X_train_reshaped = data['X_train'].reshape(-1, data['X_train'].shape[-1])
        X_val_reshaped = data['X_val'].reshape(-1, data['X_val'].shape[-1])
        X_test_reshaped = data['X_test'].reshape(-1, data['X_test'].shape[-1])
        
        X_train_scaled = scaler.transform(X_train_reshaped).reshape(data['X_train'].shape)
        X_val_scaled = scaler.transform(X_val_reshaped).reshape(data['X_val'].shape)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(data['X_test'].shape)
        
        # 高效批量标准化输出数据
        info("  - 标准化输出数据...")
        y_train_reshaped = data['y_train'].reshape(-1, data['y_train'].shape[-1])
        y_val_reshaped = data['y_val'].reshape(-1, data['y_val'].shape[-1])
        y_test_reshaped = data['y_test'].reshape(-1, data['y_test'].shape[-1])
        
        y_train_scaled = output_scaler.transform(y_train_reshaped).reshape(data['y_train'].shape)
        y_val_scaled = output_scaler.transform(y_val_reshaped).reshape(data['y_val'].shape)
        y_test_scaled = output_scaler.transform(y_test_reshaped).reshape(data['y_test'].shape)

        X_train_scaled = align_source_position_scale(
            X_train_scaled,
            input_mean=scaler.mean_,
            input_scale=scaler.scale_,
            output_mean=output_scaler.mean_,
            output_scale=output_scaler.scale_,
            output_dim=data['y_train'].shape[-1],
        ).astype(np.float32, copy=False)
        X_val_scaled = align_source_position_scale(
            X_val_scaled,
            input_mean=scaler.mean_,
            input_scale=scaler.scale_,
            output_mean=output_scaler.mean_,
            output_scale=output_scaler.scale_,
            output_dim=data['y_val'].shape[-1],
        ).astype(np.float32, copy=False)
        X_test_scaled = align_source_position_scale(
            X_test_scaled,
            input_mean=scaler.mean_,
            input_scale=scaler.scale_,
            output_mean=output_scaler.mean_,
            output_scale=output_scaler.scale_,
            output_dim=data['y_test'].shape[-1],
        ).astype(np.float32, copy=False)
        
        info("🔄 转换为PyTorch张量...")
        # 转换为tensor
        X_train = torch.from_numpy(X_train_scaled.astype(np.float32))
        y_train = torch.from_numpy(y_train_scaled.astype(np.float32))
        X_val = torch.from_numpy(X_val_scaled.astype(np.float32))
        y_val = torch.from_numpy(y_val_scaled.astype(np.float32))
        X_test = torch.from_numpy(X_test_scaled.astype(np.float32))
        y_test = torch.from_numpy(y_test_scaled.astype(np.float32))
        
        # 数据子集（与 scaler 拟合索引保持一致）
        if subset_ratio is not None and subset_ratio < 1.0:
            X_train = X_train[train_subset_idx]
            y_train = y_train[train_subset_idx]
            X_val = X_val[val_subset_idx]
            y_val = y_val[val_subset_idx]
            X_test = X_test[test_subset_idx]
            y_test = y_test[test_subset_idx]
            print(f"🔧 SOTA 数据子集：使用 {subset_ratio*100:.0f}% 数据（固定种子 42，多轮一致）")
            print(f"    训练: {len(X_train):,} | 验证: {len(X_val):,} | 测试: {len(X_test):,}")
        
        info("🔄 创建数据加载器...")
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        generator = build_torch_generator(seed) if seed is not None else None
        num_workers = _base_train_config.get('dataloader_workers', 0)
        pin_memory = _base_train_config.get('pin_memory', False)
        persistent_workers = _base_train_config.get('persistent_workers', False)
        prefetch_factor = _base_train_config.get('prefetch_factor', 2)
        loader_kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers if num_workers > 0 else False,
        }
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = prefetch_factor
            loader_kwargs['worker_init_fn'] = seed_worker
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, **loader_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
        
        info(f"数据加载完成. 训练样本: {len(X_train)}, 验证样本: {len(X_val)}, 测试样本: {len(X_test)}")
        return train_loader, val_loader, test_loader, scaler, output_scaler
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("请先运行 python data_generator.py 生成数据集")
        return None, None, None, None, None

# ======================================================================================
# 训练与评估
# ======================================================================================

def generate_causal_mask(size, device):
    """生成因果掩码"""
    mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# 定义使用一次性预测的SOTA模型类名列表
ONESHOT_MODELS = [
    'SphericalKinematicBaseline', 'RotatingEarth3DOFBaseline',
    'TSLibForecastAdapter',
]

def is_oneshot_model(model):
    """判断模型是否使用一次性预测（而非自回归）"""
    return bool(getattr(model, "is_oneshot", False)) or (
        hasattr(model, '__class__') and model.__class__.__name__ in ONESHOT_MODELS
    )


def build_eval_seed(y_true_scaled: torch.Tensor, model, device) -> torch.Tensor | None:
    """按统一评估协议构造自回归 seed。"""
    if is_oneshot_model(model):
        return None

    seed_mode = str(EVAL_AR_SEED_MODE).lower().strip()
    if seed_mode == 'gt_first':
        return y_true_scaled[:, :1, :].clone()

    # strict 模式：零起步，不向模型注入未来真值。
    batch_size = y_true_scaled.size(0)
    output_dim = y_true_scaled.size(-1)
    return torch.zeros(batch_size, 1, output_dim, device=device, dtype=y_true_scaled.dtype)


def build_decoder_context(
    y_ref: torch.Tensor,
    target_length: int,
    protocol: str = "strict_autoregressive",
) -> torch.Tensor | None:
    """
    Build decoder context following official-like label_len + zero semantics.
    Returns None when protocol does not require decoder context.
    """
    p = str(protocol).lower().strip()
    if p != "official_like_decoder":
        return None
    label_len = int(_base_train_config.get("label_len", max(1, min(48, y_ref.size(1)))))
    label_len = max(1, min(label_len, y_ref.size(1)))
    context = y_ref[:, :label_len, :]
    zero_block = torch.zeros(
        y_ref.size(0),
        max(0, target_length),
        y_ref.size(-1),
        device=y_ref.device,
        dtype=y_ref.dtype,
    )
    return torch.cat([context, zero_block], dim=1)

def unified_predict(
    model,
    x,
    target_length,
    y_true_scaled=None,
    y_seed=None,
    device=None,
    eval_protocol: str = "strict_autoregressive",
    decoder_context: torch.Tensor | None = None,
):
    """
    统一的模型预测接口
    
    Args:
        model: 模型实例
        x: 输入序列 [batch_size, seq_len, input_dim]
        target_length: 目标预测长度
        y_seed: 自回归预测的种子序列 (可选，用于自回归模型)
        device: 计算设备
    
    Returns:
        预测序列 [batch_size, target_length, output_dim]
    """
    if device is None:
        device = x.device
    
    protocol = str(eval_protocol).lower().strip()
    if y_true_scaled is not None:
        return unified_predict_by_eval_protocol(
            model=model,
            x=x,
            y_true_scaled=y_true_scaled,
            pred_length=target_length,
            device=device,
            eval_protocol=protocol,
            eval_ar_seed_mode=EVAL_AR_SEED_MODE,
            label_len=int(_base_train_config.get("label_len", 48)),
            oneshot_models=ONESHOT_MODELS,
        )

    if FORCE_ONESHOT_PREDICTION and model.__class__.__name__ == 'StandardTransformer':
        return model(x, target_length=target_length)
    if is_oneshot_model(model):
        # SOTA模型：一次性预测全部序列
        if protocol == "official_like_decoder" and model.__class__.__name__ == "TSLibForecastAdapter":
            return model(x, target_length=target_length, decoder_context=decoder_context)
        return model(x, target_length=target_length)
    else:
        # PLGAFormer/Transformer：自回归预测
        if y_seed is None:
            # 创建初始种子 (全零或使用输入的最后一帧)
            batch_size = x.size(0)
            output_dim = 3
            y_input = torch.zeros(batch_size, 1, output_dim, device=device)
        else:
            y_input = y_seed.clone()
        
        for _ in range(target_length):
            tgt_mask = generate_causal_mask(y_input.size(1), device)
            pred = model(x, y_input, tgt_mask=tgt_mask)
            y_input = torch.cat([y_input, pred[:, -1:, :]], dim=1)
        
        # 返回预测部分（去掉初始种子）
        return y_input[:, 1:, :]

def unscale_data(scaled_data, mean, scale):
    """反标准化数据"""
    return scaled_data * scale + mean

# 从统一配置获取地球半径（确保全项目一致）
_physical_constraints = HGVConfig.get_physical_constraints()
R_EARTH = _physical_constraints['earth_radius']

def spherical_to_cartesian(spherical_data):
    """
    将球坐标转换为笛卡尔坐标，用于正确计算位移误差
    
    Args:
        spherical_data: 球坐标数据 [..., 3] 包含 [r, λ, φ]
                        r: 地心距离 (m)
                        λ: 经度 (rad)
                        φ: 纬度 (rad)
    
    Returns:
        cartesian_data: 笛卡尔坐标 [..., 3] 包含 [x, y, z]
    """
    r = spherical_data[..., 0]      # 地心距离
    lon = spherical_data[..., 1]    # 经度 λ
    lat = spherical_data[..., 2]    # 纬度 φ
    
    # 球坐标到笛卡尔坐标转换
    # x = r * cos(lat) * cos(lon)
    # y = r * cos(lat) * sin(lon)
    # z = r * sin(lat)
    cos_lat = torch.cos(lat)
    x = r * cos_lat * torch.cos(lon)
    y = r * cos_lat * torch.sin(lon)
    z = r * torch.sin(lat)
    
    return torch.stack([x, y, z], dim=-1)

def compute_displacement_error(pred_pos, true_pos):
    """
    计算笛卡尔空间中的位移误差
    
    Args:
        pred_pos: 预测的球坐标位置 [..., 3] 包含 [r, λ, φ]
        true_pos: 真实的球坐标位置 [..., 3] 包含 [r, λ, φ]
    
    Returns:
        displacement_error: 欧氏距离误差 (米)
    """
    pred_cart = spherical_to_cartesian(pred_pos)
    true_cart = spherical_to_cartesian(true_pos)
    
    # 计算欧氏距离
    displacement = torch.norm(pred_cart - true_cart, dim=-1)
    return displacement

def compute_cartesian_component_errors(pred_pos, true_pos):
    """Return component-wise Cartesian RMSE (m), MSE (m^2), and MAE (m)."""
    pred_cart = spherical_to_cartesian(pred_pos)
    true_cart = spherical_to_cartesian(true_pos)
    diff = pred_cart - true_cart
    mse = (diff ** 2).mean()
    return torch.sqrt(mse), mse, torch.abs(diff).mean()

def build_supervision_windows(
    tgt: torch.Tensor,
    protocol: str,
    src: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return build_seq2seq_supervision_windows(
        tgt=tgt,
        protocol=protocol,
        pred_len=int(_base_train_config.get("pred_len", tgt.size(1))),
        label_len=int(_base_train_config.get("label_len", max(1, min(48, tgt.size(1))))),
        src=src,
    )


def training_forward(
    model,
    src,
    decoder_input,
    tgt_mask,
    device,
    supervision_protocol: str = "shifted_next_step",
    physics_prior_override: torch.Tensor | None = None,
):
    model_forward_kwargs = (
        {"physics_prior_override": physics_prior_override}
        if physics_prior_override is not None
        else None
    )
    return seq2seq_training_forward(
        model=model,
        src=src,
        decoder_input=decoder_input,
        tgt_mask=tgt_mask,
        supervision_protocol=supervision_protocol,
        pred_len=int(_base_train_config.get("pred_len", max(1, decoder_input.size(1) // 2))),
        oneshot_models=ONESHOT_MODELS,
        model_forward_kwargs=model_forward_kwargs,
    )

def run_evaluation(model, loader, scaler_mean, scaler_scale, device, horizons, trajectory_ids=None):
    """评估模型性能（包含FDE/ADE指标）"""
    model.eval()
    results = {h: {metric: [] for metric in EVAL_METRICS if metric != 'rmse'} for h in horizons}
    result_weights = {
        h: {metric: [] for metric in EVAL_METRICS if metric != 'rmse'}
        for h in horizons
    }
    trajectory_cache = {
        h: {"pred": [], "true": [], "ids": []}
        for h in horizons
    } if trajectory_ids is not None else None
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64) if trajectory_ids is not None else None
    sample_offset = 0
    max_horizon = max(horizons)
    
    with torch.no_grad():
        for x, y_true_scaled in loader:
            batch_size = x.size(0)
            batch_ids = None
            if trajectory_ids is not None:
                batch_ids = trajectory_ids[sample_offset:sample_offset + batch_size]
            sample_offset += batch_size

            x, y_true_scaled = x.to(device), y_true_scaled.to(device)
            
            if y_true_scaled.size(1) > max_horizon:
                y_true_scaled = y_true_scaled[:, :max_horizon, :]
            
            # 使用统一预测接口
            pred_length = min(y_true_scaled.size(1), max_horizon)
            y_seed = build_eval_seed(y_true_scaled, model, device)
            decoder_context = build_decoder_context(y_true_scaled, pred_length, protocol=EVAL_PROTOCOL)
            y_pred_scaled = unified_predict(
                model,
                x,
                pred_length,
                y_true_scaled=y_true_scaled,
                y_seed=y_seed,
                device=device,
                eval_protocol=EVAL_PROTOCOL,
                decoder_context=decoder_context,
            )
            if y_pred_scaled.ndim != 3 or y_pred_scaled.size(0) != batch_size:
                raise RuntimeError(
                    f"Invalid evaluation output shape {tuple(y_pred_scaled.shape)} for batch {batch_size}."
                )
            if y_pred_scaled.size(1) < pred_length or y_pred_scaled.size(2) < 3:
                raise RuntimeError(
                    f"Incomplete evaluation output {tuple(y_pred_scaled.shape)} for horizon {pred_length}."
                )
            if not torch.isfinite(y_pred_scaled).all() or not torch.isfinite(y_true_scaled).all():
                raise FloatingPointError("Non-finite prediction or target in formal evaluation.")
            
            pred_scaled = y_pred_scaled
            true_scaled = y_true_scaled
            if REPORT_ON_PHYSICAL_SCALE:
                pred_phys = unscale_data(y_pred_scaled, scaler_mean, scaler_scale)
                true_phys = unscale_data(y_true_scaled, scaler_mean, scaler_scale)
            else:
                pred_phys = pred_scaled
                true_phys = true_scaled
            
            for h in horizons:
                actual_h = min(h, pred_scaled.size(1), true_scaled.size(1))
                if actual_h <= 0:
                    continue
                
                pred_scaled_h = pred_scaled[:, :actual_h, :]
                true_scaled_h = true_scaled[:, :actual_h, :]
                pred_phys_h = pred_phys[:, :actual_h, :]
                true_phys_h = true_phys[:, :actual_h, :]
                
                if REPORT_MSE_ON_SCALED:
                    mse = ((pred_scaled_h - true_scaled_h) ** 2).mean().item()
                    mae = torch.abs(pred_scaled_h - true_scaled_h).mean().item()
                else:
                    mse = ((pred_phys_h - true_phys_h) ** 2).mean().item()
                    mae = torch.abs(pred_phys_h - true_phys_h).mean().item()
                
                # FDE (Final Displacement Error) - 最后一个时间步的位置误差
                # 使用笛卡尔坐标计算真实的空间位移误差（单位：米）
                # 位置是前3个维度 [r, λ, φ]
                final_pred_pos = pred_phys_h[:, -1, :3]
                final_true_pos = true_phys_h[:, -1, :3]
                # 使用球坐标→笛卡尔转换计算真实位移
                fde = compute_displacement_error(final_pred_pos, final_true_pos).mean().item()
                
                # ADE (Average Displacement Error) - 所有时间步的平均位置误差
                all_pred_pos = pred_phys_h[:, :, :3]
                all_true_pos = true_phys_h[:, :, :3]
                # 逐时间步计算位移误差后取平均
                ade = compute_displacement_error(all_pred_pos, all_true_pos).mean().item()
                rmse_cart_m, mse_cart_m2, mae_cart_m = compute_cartesian_component_errors(all_pred_pos, all_true_pos)
                
                results[h]['mse'].append(mse)
                results[h]['mae'].append(mae)
                results[h]['fde'].append(fde)
                results[h]['ade'].append(ade)
                results[h]['rmse_cart_m'].append(rmse_cart_m.item())
                results[h]['mse_cart_m2'].append(mse_cart_m2.item())
                results[h]['mae_cart_m'].append(mae_cart_m.item())
                result_weights[h]['mse'].append(pred_scaled_h.numel())
                result_weights[h]['mae'].append(pred_scaled_h.numel())
                result_weights[h]['fde'].append(final_pred_pos.size(0))
                result_weights[h]['ade'].append(all_pred_pos.size(0) * all_pred_pos.size(1))
                cartesian_component_count = all_pred_pos.size(0) * all_pred_pos.size(1) * 3
                result_weights[h]['rmse_cart_m'].append(cartesian_component_count)
                result_weights[h]['mse_cart_m2'].append(cartesian_component_count)
                result_weights[h]['mae_cart_m'].append(cartesian_component_count)

                if trajectory_cache is not None and batch_ids is not None:
                    pred_cart_h = spherical_to_cartesian(pred_phys_h[:, :, :3]).detach().cpu().numpy()
                    true_cart_h = spherical_to_cartesian(true_phys_h[:, :, :3]).detach().cpu().numpy()
                    trajectory_cache[h]["pred"].append(pred_cart_h)
                    trajectory_cache[h]["true"].append(true_cart_h)
                    trajectory_cache[h]["ids"].append(batch_ids[: pred_cart_h.shape[0]])
    
    if trajectory_ids is not None and sample_offset != len(trajectory_ids):
        raise RuntimeError(
            f"Evaluation sample/trajectory mismatch: {sample_offset} samples vs "
            f"{len(trajectory_ids)} trajectory IDs."
        )

    for h in horizons:
        if results[h]['mse']:
            for metric in ('mse', 'mae', 'fde', 'ade', 'mse_cart_m2', 'mae_cart_m'):
                results[h][metric] = float(np.average(results[h][metric], weights=result_weights[h][metric]))
            results[h]['rmse_cart_m'] = float(np.sqrt(results[h]['mse_cart_m2']))
        else:
            raise RuntimeError(f"No evaluation samples were accumulated for horizon {h}.")
        metric_values = [
            results[h][metric]
            for metric in ('mse', 'mae', 'fde', 'ade', 'rmse_cart_m', 'mse_cart_m2', 'mae_cart_m')
        ]
        if not np.all(np.isfinite(np.asarray(metric_values, dtype=np.float64))):
            raise FloatingPointError(f"Non-finite aggregate evaluation metrics at horizon {h}.")

        if trajectory_cache is not None and trajectory_cache[h]["pred"]:
            trajectory_report = aggregate_window_metrics_by_trajectory(
                np.concatenate(trajectory_cache[h]["pred"], axis=0),
                np.concatenate(trajectory_cache[h]["true"], axis=0),
                np.concatenate(trajectory_cache[h]["ids"], axis=0),
            )
            results[h]["trajectory_window_ade"] = trajectory_report["mean_ade"]
            results[h]["trajectory_window_fde"] = trajectory_report["mean_fde"]
            results[h]["trajectory_count"] = trajectory_report["trajectory_count"]
            results[h]["trajectory_ade_ci"] = trajectory_report["ade_ci"]
            results[h]["trajectory_fde_ci"] = trajectory_report["fde_ci"]
            results[h]["trajectory_metrics"] = {
                str(key): value for key, value in trajectory_report["trajectories"].items()
            }
    
    return results

def save_predictions_for_visualization(models_dict, test_loader, scaler_mean, scaler_scale, device, save_dir, num_samples=100):
    """
    保存模型预测结果用于后续快速可视化
    
    Args:
        models_dict: {model_name: model} 字典
        test_loader: 测试数据加载器
        scaler_mean, scaler_scale: 标准化参数
        device: 计算设备
        save_dir: 保存目录
        num_samples: 保存的样本数量
    """
    print(f"\n{'='*80}")
    print(f"保存预测结果用于快速可视化...")
    print(f"{'='*80}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集测试样本
    X_samples = []
    y_samples = []
    
    for x, y in test_loader:
        X_samples.append(x)
        y_samples.append(y)
        if len(X_samples) * x.size(0) >= num_samples:
            break
    
    X_test = torch.cat(X_samples, dim=0)[:num_samples].to(device)
    y_test = torch.cat(y_samples, dim=0)[:num_samples].to(device)
    pred_len = min(
        int(_base_train_config.get("pred_len", y_test.size(1))),
        int(max(PREDICTION_HORIZONS)) if PREDICTION_HORIZONS else int(y_test.size(1)),
        int(y_test.size(1)),
    )
    if y_test.size(1) > pred_len:
        y_test = y_test[:, :pred_len, :]
    
    print(f"  收集了 {len(X_test)} 个测试样本 (预测长度 {pred_len} 步)")
    
    # 保存真实值（与预测长度对齐，避免后续可视化时形状不匹配）
    predictions_cache = {
        'X_test': X_test.cpu().numpy(),
        'y_test': y_test.cpu().numpy(),
        'scaler_mean': scaler_mean.cpu().numpy() if isinstance(scaler_mean, torch.Tensor) else scaler_mean,
        'scaler_scale': scaler_scale.cpu().numpy() if isinstance(scaler_scale, torch.Tensor) else scaler_scale,
        'prediction_length': np.asarray(pred_len, dtype=np.int64),
        'prediction_horizons': np.asarray(PREDICTION_HORIZONS, dtype=np.int64),
        'seq_len': np.asarray(_base_train_config.get("seq_len", X_test.size(1)), dtype=np.int64),
        'predictions': {}
    }
    
    # 对每个模型生成预测
    for model_name, model in models_dict.items():
        print(f"  生成 {model_name} 的预测...")
        model.eval()
        
        all_predictions = []
        
        with torch.no_grad():
            # 分批处理避免内存溢出
            batch_size = 32
            for i in range(0, len(X_test), batch_size):
                x_batch = X_test[i:i+batch_size]
                y_batch = y_test[i:i+batch_size]
                
                y_seed = build_eval_seed(y_batch, model, device)
                decoder_context = build_decoder_context(y_batch, pred_len, protocol=EVAL_PROTOCOL)
                y_pred = unified_predict(
                    model,
                    x_batch,
                    pred_len,
                    y_seed=y_seed,
                    device=device,
                    eval_protocol=EVAL_PROTOCOL,
                    decoder_context=decoder_context,
                )
                
                all_predictions.append(y_pred.cpu().numpy())
        
        predictions_cache['predictions'][model_name] = np.concatenate(all_predictions, axis=0)
        print(f"    ✓ 预测形状: {predictions_cache['predictions'][model_name].shape}")
    
    # 保存到文件
    save_path = os.path.join(save_dir, "predictions_cache.npz")
    np.savez_compressed(save_path, **predictions_cache)
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n✅ 预测结果已保存: {save_path}")
    print(f"   文件大小: {file_size_mb:.2f} MB")
    print(f"   包含模型: {list(predictions_cache['predictions'].keys())}")
    print(f"   样本数量: {num_samples}")
    print(f"{'='*80}\n")


def evaluate_per_maneuver(model, test_data, maneuver_labels, scaler_mean, scaler_scale, device, horizon=64):
    """
    按机动类型分别评估模型性能
    
    Args:
        model: 训练好的模型
        test_data: 测试数据 (X_test, y_test)
        maneuver_labels: 机动类型标签数组
        scaler_mean: 输出标准化器的均值
        scaler_scale: 输出标准化器的缩放
        device: 计算设备
        horizon: 预测时间步长
        
    Returns:
        dict: 每种机动类型的MSE, MAE, FDE, ADE
    """
    model.eval()
    X_test, y_test = test_data
    
    # 确保数据是tensor
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.from_numpy(X_test.astype(np.float32))
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.from_numpy(y_test.astype(np.float32))
    
    # 获取唯一的机动类型
    unique_maneuvers = np.unique(maneuver_labels)
    results = {}
    
    eval_batch_size = int(_base_train_config.get('batch_size', 64))
    with torch.no_grad():
        for maneuver_type in unique_maneuvers:
            # 获取该机动类型的索引
            indices = np.where(maneuver_labels == maneuver_type)[0]
            
            if len(indices) == 0:
                continue
            
            totals = {'squared': 0.0, 'absolute': 0.0, 'fde': 0.0, 'ade': 0.0}
            counts = {'components': 0, 'fde': 0, 'ade': 0}
            for start in range(0, len(indices), eval_batch_size):
                batch_indices = indices[start:start + eval_batch_size]
                X_batch = X_test[batch_indices].to(device)
                y_batch = y_test[batch_indices].to(device)
                if y_batch.size(1) > horizon:
                    y_batch = y_batch[:, :horizon, :]
                pred_length = min(y_batch.size(1), horizon)
                y_pred = unified_predict(
                    model,
                    X_batch,
                    pred_length,
                    y_true_scaled=y_batch,
                    y_seed=build_eval_seed(y_batch, model, device),
                    device=device,
                    eval_protocol=EVAL_PROTOCOL,
                    decoder_context=build_decoder_context(y_batch, pred_length, protocol=EVAL_PROTOCOL),
                )

                mean_tensor = scaler_mean.to(device) if isinstance(scaler_mean, torch.Tensor) else torch.as_tensor(scaler_mean, device=device)
                scale_tensor = scaler_scale.to(device) if isinstance(scaler_scale, torch.Tensor) else torch.as_tensor(scaler_scale, device=device)
                mean_tensor = mean_tensor[:y_pred.size(-1)]
                scale_tensor = scale_tensor[:y_pred.size(-1)]
                y_pred_phys = unscale_data(y_pred, mean_tensor, scale_tensor) if REPORT_ON_PHYSICAL_SCALE else y_pred
                y_true_phys = unscale_data(y_batch, mean_tensor, scale_tensor) if REPORT_ON_PHYSICAL_SCALE else y_batch
                metric_pred = y_pred if REPORT_MSE_ON_SCALED else y_pred_phys
                metric_true = y_batch if REPORT_MSE_ON_SCALED else y_true_phys
                error = metric_pred - metric_true
                totals['squared'] += float(error.square().sum())
                totals['absolute'] += float(error.abs().sum())
                counts['components'] += error.numel()

                fde_values = compute_displacement_error(y_pred_phys[:, -1, :3], y_true_phys[:, -1, :3])
                ade_values = compute_displacement_error(y_pred_phys[:, :, :3], y_true_phys[:, :, :3])
                totals['fde'] += float(fde_values.sum())
                totals['ade'] += float(ade_values.sum())
                counts['fde'] += fde_values.numel()
                counts['ade'] += ade_values.numel()
                del X_batch, y_batch, y_pred, y_pred_phys, y_true_phys

            mse = totals['squared'] / counts['components']
            mae = totals['absolute'] / counts['components']
            fde = totals['fde'] / counts['fde']
            ade = totals['ade'] / counts['ade']
            
            metric_values = np.asarray([mse, mae, fde, ade], dtype=np.float64)
            if not np.all(np.isfinite(metric_values)):
                raise FloatingPointError(
                    f"Non-finite per-maneuver metrics for {maneuver_type}: "
                    f"MSE={mse}, MAE={mae}, FDE={fde}, ADE={ade}."
                )
            
            results[maneuver_type] = {
                'mse': mse,
                'mae': mae,
                'fde': fde,
                'ade': ade,
                'num_samples': len(indices)
            }
    
    return results

def benchmark_model_efficiency(model, test_input, device, num_runs=100):
    """
    测量模型的计算效率
    
    Args:
        model: 训练好的模型
        test_input: 测试输入数据 (batch_size, seq_len, input_dim)
        device: 计算设备
        num_runs: 运行次数用于平均
        
    Returns:
        dict: 包含latency, throughput, memory的字典
    """
    model.eval()
    device_obj = torch.device(device) if isinstance(device, str) else device
    
    # 确保输入是tensor
    if not isinstance(test_input, torch.Tensor):
        test_input = torch.from_numpy(test_input.astype(np.float32))
    test_input = test_input[:1].to(device_obj)
    target_length = int(max(PREDICTION_HORIZONS))
    dummy_target = torch.zeros(test_input.size(0), target_length, 3, device=device_obj)

    def predict_once():
        return unified_predict(
            model,
            test_input,
            target_length,
            y_true_scaled=dummy_target,
            device=device_obj,
            eval_protocol=EVAL_PROTOCOL,
        )
    
    # 预热 - 使用统一预测接口
    with torch.no_grad():
        for _ in range(10):
            _ = predict_once()
    
    # 测量推理时间
    if device_obj.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = predict_once()
    
    if device_obj.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_latency = (end_time - start_time) / num_runs * 1000  # ms
    throughput = num_runs * test_input.size(0) / (end_time - start_time)  # samples/sec
    
    # 测量内存占用
    if device_obj.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device_obj)
        with torch.no_grad():
            _ = predict_once()
        memory_mb = torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)
    else:
        memory_mb = 0.0
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'latency_ms': avg_latency,
        'throughput_samples_per_sec': throughput,
        'memory_mb': memory_mb,
        'total_params': total_params,
        'trainable_params': trainable_params
    }

def train_model(model_name, model_config, train_loader, val_loader, scaler_mean, scaler_scale, model_override=None):
    """训练模型 - 优化版本，参考消融实验的成功经验"""
    info("-" * 50)
    info(f"训练模型: {model_name}")
    
    device = torch.device(_base_train_config['device'])
    
    # 验证输入参数
    if model_override is None:
        raise ValueError("model_override is required for training")
    
    if not hasattr(model_override, 'parameters'):
        raise ValueError("model_override must be a valid PyTorch model")
    
    model = model_override.to(device)
    
    # Analytical fixed-parameter models skip optimization.
    model_type = model_config.get('model_type')
    if model_type in {'kinematic', 'rotating_3dof'}:
        print(f"ℹ️  {model_type} 为固定参数模型，无需训练，直接使用")
        model.eval()
        # 快速验证：运行一次前向传播确保模型可用。
        with torch.no_grad():
            test_batch = next(iter(val_loader))
            test_input = test_batch[0][:2].to(device)
            test_output = unified_predict(model, test_input, 64, device=device)
            if not torch.isfinite(test_output).all():
                raise FloatingPointError(f"Non-finite {model_type} validation output.")
            print(f"  ✓ {model_type} 模型验证通过")
        return model, {
            "epochs_completed": 0,
            "best_epoch": None,
            "best_val_loss": None,
            "epoch_records": [],
            "fixed_parameter_model": True,
        }
    
    model.train()

    prior_cache_stats = {
        "train": {"enabled": False, "reason": "disabled_by_config"},
        "validation": {"enabled": False, "reason": "disabled_by_config"},
    }
    if bool(_base_train_config.get("cache_physics_prior", True)):
        cache_batch_size = int(
            _base_train_config.get("physics_prior_cache_batch_size", train_loader.batch_size or 128)
        )
        train_loader, train_prior_cache = build_cached_physics_prior_loader(
            train_loader,
            model,
            device=device,
            target_length=int(_base_train_config.get("pred_len", 256)),
            cache_batch_size=cache_batch_size,
        )
        val_loader, val_prior_cache = build_cached_physics_prior_loader(
            val_loader,
            model,
            device=device,
            target_length=int(_base_train_config.get("pred_len", 256)),
            cache_batch_size=cache_batch_size,
        )
        prior_cache_stats = {
            "train": train_prior_cache.as_dict(),
            "validation": val_prior_cache.as_dict(),
        }
        if train_prior_cache.enabled:
            print(
                "物理先验缓存: "
                f"train={train_prior_cache.samples:,}, val={val_prior_cache.samples:,}, "
                f"build={train_prior_cache.elapsed_sec + val_prior_cache.elapsed_sec:.2f}s"
            )
    
    # 使用统一的物理损失函数
    physics_loss_weight = model_config.get('physics_loss_weight', 0.0)
    use_physics_loss = physics_loss_weight > 0.0
    
    model_type = model_config.get('model_type')
    primary_criterion = ECEFTrajectoryLoss(
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        distance_scale_m=float(_base_train_config.get('ecef_distance_scale_m', 100_000.0)),
        scaled_mse_weight=float(_base_train_config.get('scaled_mse_weight', 0.1)),
    ).to(device)
    physics_criterion = None
    if use_physics_loss:
        # 传入 scaler 使物理约束在反标准化后的物理空间计算（与顶刊/消融一致）
        scaler_mean_np = scaler_mean.cpu().numpy() if hasattr(scaler_mean, 'cpu') else scaler_mean
        scaler_std_np = scaler_scale.cpu().numpy() if hasattr(scaler_scale, 'cpu') else scaler_scale
        physics_criterion = HGVPhysicsLoss(
            alpha=physics_loss_weight,
            scaler_mean=scaler_mean_np,
            scaler_std=scaler_std_np,
        )
        # 确保 criterion 在正确设备上（register_buffer 会自动跟随）
        physics_criterion = physics_criterion.to(device)
    
    # 统一学习率和正则化策略（与消融一致：AdamW + Cosine + Warmup，顶刊常见）
    base_lr = _base_train_config.get('learning_rate', 1e-3)
    weight_decay = _base_train_config.get('weight_decay', 5e-5)
    total_epochs = _base_train_config['epochs']
    warmup_epochs = _base_train_config.get('warmup_epochs', 5)

    optimizer = build_optimizer_by_profile(
        model=model,
        learning_rate=base_lr,
        weight_decay=weight_decay,
        profile=OPTIMIZER_PROFILE,
    )
    scheduler = build_scheduler_by_profile(
        optimizer=optimizer,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        profile=OPTIMIZER_PROFILE,
        min_lr_ratio=0.01,
    )
    
    # 梯度缩放器（混合精度训练）
    device_obj = torch.device(device) if isinstance(device, str) else device
    use_mixed_precision = _base_train_config.get('use_mixed_precision', True)
    mixed_precision_dtype_name = str(
        _base_train_config.get('mixed_precision_dtype', 'bfloat16')
    ).lower()
    mixed_precision_dtype = (
        torch.bfloat16 if mixed_precision_dtype_name == 'bfloat16' else torch.float16
    )
    amp_init_scale = float(_base_train_config.get('amp_init_scale', 128.0))
    scaler = (
        torch.amp.GradScaler(
            'cuda',
            init_scale=amp_init_scale,
            enabled=mixed_precision_dtype == torch.float16,
        )
        if (device_obj.type == 'cuda' and use_mixed_precision)
        else None
    )
    
    best_val_loss = float('inf')
    best_state_dict = None
    best_epoch = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    epoch_records = []
    physics_gate_history = []
    expected_train_batches = len(train_loader)
    expected_val_batches = len(val_loader)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"物理约束权重: {physics_loss_weight}, 混合精度: {use_mixed_precision}, "
        f"dtype={mixed_precision_dtype_name if use_mixed_precision else 'float32'}"
    )
    
    for epoch in range(_base_train_config['epochs']):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        train_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            src, tgt, physics_prior_override = unpack_batch_with_optional_prior(batch)
            src, tgt = src.to(device), tgt.to(device)
            if physics_prior_override is not None:
                physics_prior_override = physics_prior_override.to(device, non_blocking=True)
            if not torch.isfinite(src).all() or not torch.isfinite(tgt).all():
                raise FloatingPointError(f"Non-finite training input at batch {batch_idx}.")
            if src.size(0) == 0 or tgt.size(0) == 0:
                raise RuntimeError(f"Empty training batch at index {batch_idx}.")
            
            # 准备输入数据
            decoder_input, tgt_output = build_supervision_windows(tgt, TRAIN_SUPERVISION_PROTOCOL, src=src)
            
            # 生成掩码
            tgt_mask = generate_causal_mask(decoder_input.size(1), device)
            
            optimizer.zero_grad()
            
            # 混合精度训练（添加数值稳定性检查）
            clip_norm = _base_train_config.get('gradient_clip_norm', None)
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=mixed_precision_dtype):
                    # 使用统一的训练前向传播接口
                    outputs = training_forward(
                        model,
                        src,
                        decoder_input,
                        tgt_mask,
                        device,
                        supervision_protocol=TRAIN_SUPERVISION_PROTOCOL,
                        physics_prior_override=physics_prior_override,
                    )
                    
                    if not torch.isfinite(outputs).all():
                        raise FloatingPointError(
                            f"Non-finite AMP model output: model={model_name}, batch={batch_idx}."
                        )
                    
                    if outputs.size(1) != tgt_output.size(1):
                        outputs = outputs[:, -tgt_output.size(1):, :]
                    loss = primary_criterion(outputs, tgt_output)
                    if physics_criterion is not None:
                        loss = loss + physics_criterion(
                            outputs.float(),
                            tgt_output.float(),
                            dt=float(_base_train_config.get('sampling_interval_s', 1.0)),
                            include_mse=False,
                        )
                
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite AMP loss: model={model_name}, batch={batch_idx}."
                    )
                
                scaler.scale(loss).backward()
                
                if NUMERICAL_GUARDS:
                    try:
                        scaler.unscale_(optimizer)
                        
                        clip_gradients_with_finite_check(
                            model,
                            clip_norm,
                            context=f"model={model_name}, batch={batch_idx}, amp=True",
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        
                    except RuntimeError as e:
                        if _is_cuda_illegal_access_error(e):
                            raise RuntimeError(
                                f"CUDA illegal memory access during AMP unscale (model={model_name}, batch={batch_idx}). "
                                "建议降低 batch size 或关闭该模型 AMP。"
                            ) from e
                        optimizer.zero_grad()
                        raise RuntimeError(
                            f"AMP unscale/step failed: model={model_name}, batch={batch_idx}: {e}"
                        ) from e
                else:
                    if clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # 使用统一的训练前向传播接口
                outputs = training_forward(
                    model,
                    src,
                    decoder_input,
                    tgt_mask,
                    device,
                    supervision_protocol=TRAIN_SUPERVISION_PROTOCOL,
                    physics_prior_override=physics_prior_override,
                )
                
                if not torch.isfinite(outputs).all():
                    raise FloatingPointError(
                        f"Non-finite model output: model={model_name}, batch={batch_idx}."
                    )
                
                if outputs.size(1) != tgt_output.size(1):
                    outputs = outputs[:, -tgt_output.size(1):, :]
                loss = primary_criterion(outputs, tgt_output)
                if physics_criterion is not None:
                    loss = loss + physics_criterion(
                        outputs.float(),
                        tgt_output.float(),
                        dt=float(_base_train_config.get('sampling_interval_s', 1.0)),
                        include_mse=False,
                    )
                
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss: model={model_name}, batch={batch_idx}."
                    )
            
                loss.backward()
                
                if NUMERICAL_GUARDS:
                    clip_gradients_with_finite_check(
                        model,
                        clip_norm,
                        context=f"model={model_name}, batch={batch_idx}, amp=False",
                    )
                    
                else:
                    if clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                
                optimizer.step()
            
            batch_size = int(src.size(0))
            epoch_train_loss += loss.item() * batch_size
            train_samples += batch_size
            num_batches += 1
            
            # 智能GPU内存管理优化
            if device.type == 'cuda':
                # 减少频繁 empty_cache/sync 对吞吐和稳定性的干扰，仅在高占用时触发。
                if batch_idx % 100 == 0:
                    total_mem = torch.cuda.get_device_properties(device).total_memory
                    allocated = torch.cuda.memory_allocated(device)
                    if allocated > 0.9 * total_mem:
                        torch.cuda.empty_cache()
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src, tgt, physics_prior_override = unpack_batch_with_optional_prior(batch)
                src, tgt = src.to(device), tgt.to(device)
                if physics_prior_override is not None:
                    physics_prior_override = physics_prior_override.to(device, non_blocking=True)
                
                decoder_input, tgt_output = build_supervision_windows(tgt, TRAIN_SUPERVISION_PROTOCOL, src=src)
                tgt_mask = generate_causal_mask(decoder_input.size(1), device)
                
                if scaler is not None:
                    with torch.amp.autocast(device_type='cuda', dtype=mixed_precision_dtype):
                        outputs = training_forward(
                            model,
                            src,
                            decoder_input,
                            tgt_mask,
                            device,
                            supervision_protocol=TRAIN_SUPERVISION_PROTOCOL,
                            physics_prior_override=physics_prior_override,
                        )
                else:
                    outputs = training_forward(
                        model,
                        src,
                        decoder_input,
                        tgt_mask,
                        device,
                        supervision_protocol=TRAIN_SUPERVISION_PROTOCOL,
                        physics_prior_override=physics_prior_override,
                    )
                
                if not torch.isfinite(outputs).all():
                    raise FloatingPointError(
                        f"Non-finite validation output: model={model_name}, epoch={epoch}."
                    )
                # 验证目标与训练目标保持一致，确保 early-stopping / best checkpoint 选择协议一致。
                if outputs.size(1) != tgt_output.size(1):
                    outputs = outputs[:, -tgt_output.size(1):, :]
                val_loss = primary_criterion(outputs, tgt_output)
                if physics_criterion is not None:
                    val_loss = val_loss + physics_criterion(
                        outputs.float(),
                        tgt_output.float(),
                        dt=float(_base_train_config.get('sampling_interval_s', 1.0)),
                        include_mse=False,
                    )
                if not torch.isfinite(val_loss):
                    raise FloatingPointError(
                        f"Non-finite validation loss: model={model_name}, epoch={epoch}."
                    )
                batch_size = int(src.size(0))
                epoch_val_loss += val_loss.item() * batch_size
                val_samples += batch_size
                val_batches += 1
        
        if num_batches != expected_train_batches or train_samples <= 0:
            raise RuntimeError(
                f"Incomplete training epoch {epoch}: {num_batches}/{expected_train_batches} batches."
            )
        if val_batches != expected_val_batches or val_samples <= 0:
            raise RuntimeError(
                f"Incomplete validation epoch {epoch}: {val_batches}/{expected_val_batches} batches."
            )
        avg_train_loss = epoch_train_loss / train_samples
        avg_val_loss = epoch_val_loss / val_samples
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epoch_lr = float(optimizer.param_groups[0]['lr'])
        epoch_time = time.time() - epoch_start_time
        gate_snapshot = None
        if hasattr(model, 'physics_gate_snapshot'):
            gate_snapshot = model.physics_gate_snapshot()
            gate_snapshot['epoch'] = int(epoch + 1)
            physics_gate_history.append(gate_snapshot)
        
        min_delta = _base_train_config.get('min_delta', 1e-6)  # 改为小正值，避免微小波动重置patience
        early_stopping_patience = _base_train_config.get('early_stopping_patience', 15)
        
        if should_update_best(avg_val_loss, best_val_loss, min_delta=min_delta):
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = int(epoch + 1)
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_records.append({
            'epoch': int(epoch + 1),
            'train_loss': float(avg_train_loss),
            'val_loss': float(avg_val_loss),
            'learning_rate': epoch_lr,
            'epoch_time_sec': float(epoch_time),
            'train_batches': int(num_batches),
            'val_batches': int(val_batches),
            'train_samples': int(train_samples),
            'val_samples': int(val_samples),
            'amp_scale': float(scaler.get_scale()) if scaler is not None else None,
            'is_best': bool(best_epoch == epoch + 1),
            'physics_gate_snapshot': gate_snapshot,
        })

        # 学习率调整
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == _base_train_config['epochs'] - 1:
            print(f'Epoch {epoch}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, LR={epoch_lr:.6f}, Time={epoch_time:.2f}s')
        
        # Early Stopping 检查：如果验证损失连续 patience 个 epoch 未改善，提前停止训练
        if patience_counter >= early_stopping_patience:
            print(f'⏹️  Early Stopping: 验证损失连续 {early_stopping_patience} 个 epoch 未改善（最佳: {best_val_loss:.6f} @ Epoch {best_epoch}），提前停止训练')
            break
    
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    info(f'模型训练完成: {model_name}')
    info(f'最佳验证损失: {best_val_loss:.6f}')
    training_history = {
        'epochs_requested': int(_base_train_config['epochs']),
        'epochs_completed': int(len(epoch_records)),
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epoch_records': epoch_records,
        'physics_gate_history': physics_gate_history,
        'best_model_gate_snapshot': model.physics_gate_snapshot()
        if hasattr(model, 'physics_gate_snapshot') else None,
        'physics_prior_cache': prior_cache_stats,
        'fixed_parameter_model': False,
        'optimizer_profile': OPTIMIZER_PROFILE,
        'mixed_precision': bool(scaler is not None),
        'mixed_precision_dtype': mixed_precision_dtype_name
        if scaler is not None else 'float32',
        'supervision_protocol': TRAIN_SUPERVISION_PROTOCOL,
    }
    return model, training_history

# ======================================================================================
# 主执行流程
# ======================================================================================

def main():
    """主函数：运行SOTA模型对比实验 - 符合顶刊标准（多次运行 + 统计分析）"""
    
    info(f"\n{'#'*80}")
    info(f"# SOTA对比实验 - 顶刊标准")
    info(f"# 训练轮数: {_base_train_config['epochs']} epochs")
    info(f"# 独立运行: {NUM_RUNS}次")
    info(f"# 预测长度: {PREDICTION_HORIZONS}")
    info(f"{'#'*80}\n")
    
    device = _base_train_config['device']
    
    # 加载maneuver labels用于per-maneuver评估
    try:
        data_path = str(get_dataset_npz_path(PROJECT_ROOT))
        data = load_hgv_dataset(data_path, require_trajectory_level=True).raw
        maneuver_labels_test = get_split_maneuver_labels(data, "test")
        trajectory_ids_test = np.asarray(data["trajectory_ids_test"], dtype=np.int64)
        if maneuver_labels_test is None:
            # 兼容旧数据：若无显式索引，回退到旧约定（连续切分）。
            all_maneuver_labels = data['maneuver_labels']
            total_samples = len(all_maneuver_labels)
            test_start_idx = int(total_samples * 0.9)
            maneuver_labels_test = all_maneuver_labels[test_start_idx:]
        print(f"✅ 加载maneuver labels成功，测试集样本数: {len(maneuver_labels_test)}")
    except Exception as e:
        print(f"⚠️  无法加载maneuver labels: {e}")
        maneuver_labels_test = None
        trajectory_ids_test = None

    trajectory_ids_for_eval = trajectory_ids_test
    if trajectory_ids_for_eval is not None and SOTA_DATA_SUBSET_RATIO is not None and SOTA_DATA_SUBSET_RATIO < 1.0:
        _test_idx_for_eval = _stable_indices(len(trajectory_ids_for_eval), SOTA_DATA_SUBSET_RATIO, rng_seed=42)
        trajectory_ids_for_eval = trajectory_ids_for_eval[_test_idx_for_eval]

    selected_comparison_models = _selected_comparison_models()
    unknown_model_types = validate_comparison_model_types(selected_comparison_models)
    if unknown_model_types:
        raise ValueError(f"Unknown comparison model_type(s): {unknown_model_types}")
    
    # ======================================================================================
    # 多次运行实验循环 - 符合顶刊标准
    # ======================================================================================
    
    total_models = len(selected_comparison_models)
    
    # 存储多次运行的结果 {model_name: [run1_results, run2_results, ...]}
    multi_run_results = {model_name: [] for model_name in selected_comparison_models.keys()}
    
    # 创建统一实验目录（基于 PROJECT_ROOT）
    results_dir_path, models_dir_path = get_experiment_dirs(PROJECT_ROOT, "exp1_sota")
    models_save_dir = str(models_dir_path)
    partial_runs_path = results_dir_path / "partial_runs.json"
    partial_payload = _load_partial_runs(partial_runs_path)
    partial_payload.setdefault("experiment", "exp1_sota")
    partial_payload.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    partial_payload["active_config"] = {
        "pred_len": int(_base_train_config.get("pred_len", 0)),
        "prediction_horizons": [int(h) for h in PREDICTION_HORIZONS],
        "random_seeds": [int(seed) for seed in RANDOM_SEEDS],
        "selected_models": list(selected_comparison_models.keys()),
        "train_mode": os.getenv("HGV_TRAIN_MODE", HGVConfig.TRAIN_MODE),
        "run_signature": _current_run_signature(),
    }
    skip_existing = _env_flag("HGV_SOTA_SKIP_EXISTING", True)
    run_checkpoint_paths = {}
    if partial_payload.get("runs"):
        print(f"Loaded partial SOTA runs: {len(partial_payload['runs'])} entries from {partial_runs_path}")
    
    # 多次运行外层循环
    for run_idx, seed in enumerate(RANDOM_SEEDS):
        print(f"\n{'#'*80}")
        print(f"# 第 {run_idx + 1}/{NUM_RUNS} 次运行 (随机种子: {seed})")
        print(f"{'#'*80}")
        
        # 单入口：本 run 仅在此处设置随机种子，随后数据加载使用同一 seed
        set_random_seed(seed)
        train_loader, val_loader, test_loader, scaler, output_scaler = load_and_prepare_data(
            _base_train_config['batch_size'], seed=seed, subset_ratio=SOTA_DATA_SUBSET_RATIO
        )
        if train_loader is None:
            return
        
        scaler_mean = torch.from_numpy(scaler.mean_.astype(np.float32)).to(device)
        scaler_scale = torch.from_numpy(scaler.scale_.astype(np.float32)).to(device)
        output_scaler_mean = torch.from_numpy(output_scaler.mean_.astype(np.float32)).to(device)
        output_scaler_scale = torch.from_numpy(output_scaler.scale_.astype(np.float32)).to(device)
        
        # 本次运行的结果
        run_results = OrderedDict()
        
        # 训练所有模型
        for model_idx, (model_name, model_config) in enumerate(selected_comparison_models.items(), 1):
            start_time_model = time.time()
            partial_key = _partial_key(seed, model_name, model_config)
            if skip_existing and partial_key in partial_payload.get("runs", {}):
                cached_checkpoint = partial_payload["runs"][partial_key].get("checkpoint_path")
                if cached_checkpoint:
                    run_checkpoint_paths[model_name] = cached_checkpoint
                print(f"Skipping completed run from partial cache: seed={seed}, model={model_name}")
                continue
            print(f"\n{'='*80}")
            print(f"[Run {run_idx+1}/{NUM_RUNS}] [{model_idx}/{total_models}] 模型: {model_name}")
            print(f"{'='*80}")
            print(f"模型类型: {model_config['model_type']}")
            print(f"物理损失权重: {model_config.get('physics_loss_weight', 0.0)}")
            provenance = get_model_provenance(model_config['model_type'])
            print(
                "实现来源: "
                f"{provenance['source_type']} | {provenance['implementation_level']} | {provenance['reference_family']}"
            )

            # Every method in a paired seed run receives the same initialization
            # seed and minibatch order. Otherwise earlier methods consume RNG state
            # and silently change the optimization protocol for later methods.
            set_random_seed(seed)
            loader_generator = getattr(train_loader, "generator", None)
            if loader_generator is not None:
                loader_generator.manual_seed(int(seed))
            
            try:
                plgaformer_kwargs = _model_reconstruction_kwargs(
                    model_config['model_type'], scaler, output_scaler
                )
                model = create_model(
                    model_config['model_type'],
                    input_dim=6,
                    device=device,
                    plgaformer_kwargs=plgaformer_kwargs,
                )
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"总参数量: {total_params:,}")
                print(f"可训练参数量: {trainable_params:,}")
                model_audit = {
                    "total_parameters": int(total_params),
                    "trainable_parameters": int(trainable_params),
                    "provenance": provenance,
                    "model_type": model_config['model_type'],
                    "description": model_config.get('description'),
                    "innovations": model_config.get('innovations', []),
                }
                if hasattr(model, "source_audit"):
                    model_audit["external_source"] = model.source_audit
                
                trained_model, training_history = train_model(
                    model_name,
                    model_config,
                    train_loader,
                    val_loader,
                    output_scaler_mean,
                    output_scaler_scale,
                    model_override=model,
                )
                eval_results = run_evaluation(
                    trained_model,
                    test_loader,
                    output_scaler_mean,
                    output_scaler_scale,
                    device,
                    PREDICTION_HORIZONS,
                    trajectory_ids=trajectory_ids_for_eval,
                )
                
                # Save every seed under a configuration-specific path. This is
                # required for exact reruns and avoids collisions with open files.
                checkpoint_path = None
                model_type = model_config['model_type']
                if get_model_save_path(model_type, models_save_dir) is not None:
                    signature_short = _current_run_signature()[:12]
                    checkpoint_dir = Path(models_save_dir) / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    save_path = checkpoint_dir / (
                        f"best_{model_type}_{signature_short}_seed{seed}_pred"
                        f"{_base_train_config.get('pred_len')}.pth"
                    )
                    temp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                    try:
                        torch.save(trained_model.state_dict(), temp_path)
                        os.replace(temp_path, save_path)
                        checkpoint_path = str(save_path)
                        print(f"  ✓ 模型已保存: {save_path}")
                    except Exception as e:
                        if temp_path.exists():
                            temp_path.unlink(missing_ok=True)
                        raise RuntimeError(f"Checkpoint save failed for {model_name}: {e}") from e
                if checkpoint_path:
                    run_checkpoint_paths[model_name] = checkpoint_path
                
                end_time_model = time.time()
                run_results[model_name] = eval_results
                _upsert_partial_run(
                    partial_payload,
                    partial_key,
                    seed,
                    model_name,
                    model_config,
                    eval_results,
                    end_time_model - start_time_model,
                    checkpoint_path=checkpoint_path,
                    training_history=training_history,
                    model_audit=model_audit,
                )
                _save_partial_runs(partial_runs_path, partial_payload)
                
                print(f"✅ 模型 {model_name} 训练完成，用时: {end_time_model - start_time_model:.2f}秒")
            
            except Exception as e:
                print(f"❌ 模型 {model_name} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                if isinstance(e, FloatingPointError):
                    raise
                if _is_cuda_illegal_access_error(e):
                    raise RuntimeError(
                        f"检测到 CUDA illegal memory access（模型: {model_name}）。"
                        "为避免 CUDA 上下文污染导致后续模型连锁失败，当前运行已中止。"
                    ) from e
                raise RuntimeError(f"Formal SOTA run failed for model {model_name}.") from e
            finally:
                if 'trained_model' in locals():
                    del trained_model
                if 'model' in locals():
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 将本次运行结果添加到多次运行结果中
        for model_name, results in run_results.items():
            multi_run_results[model_name].append(results)
    
        # 清理GPU缓存
        torch.cuda.empty_cache()
    
    # ======================================================================================
    # 统计分析 - 多次运行结果汇总
    # ======================================================================================
    
    print(f"\n{'='*100}")
    print(f"📊 统计分析 ({NUM_RUNS}次独立运行)")
    print(f"{'='*100}")
    
    # Rebuild from the durable partial cache so skipped and freshly completed runs use one source.
    multi_run_results = _multi_run_from_partial(partial_payload, selected_comparison_models)

    # 计算统计量
    stats_results = compute_statistics(multi_run_results)
    
    # 执行统计检验
    test_results = perform_trajectory_level_tests(
        multi_run_results,
        baseline_name="Transformer (baseline)",
    )
    
    # 打印统计摘要
    print_statistical_summary(
        stats_results, test_results, baseline_name="Transformer (baseline)"
    )
    
    # 为了兼容后续代码，创建final_results（使用均值）
    final_results = OrderedDict()
    for model_name, model_stats in stats_results.items():
        final_results[model_name] = {}
        for horizon in PREDICTION_HORIZONS:
            if horizon in model_stats:
                final_results[model_name][horizon] = {
                    metric: data['mean'] 
                    for metric, data in model_stats[horizon].items()
                }
    
    # 打印结果（使用mean±std格式）
    print("\n" + "="*120)
    print(f"SOTA COMPARISON RESULTS - Multi-Horizon Prediction ({NUM_RUNS}次独立运行, mean±std)")
    print("="*120)
    
    # 打印详细的多步长结果（表头与论文主表一致）
    mse_header = "MSE (scaled)" if REPORT_MSE_ON_SCALED else "MSE"
    header = f"{'Model':<28} | " + " | ".join([f"{mse_header}-{h:<12}" for h in PREDICTION_HORIZONS]) + " | Improvement"
    print(header)
    print("-" * 120)
    
    # 获取baseline MSE用于计算改善幅度
    baseline_mse_128 = stats_results.get("Transformer (baseline)", {}).get(128, {}).get('mse', {}).get('mean', 1.0)
    
    for model_name in final_results.keys():
        if model_name in stats_results:
            mse_strs = []
            for h in PREDICTION_HORIZONS:
                if h in stats_results[model_name] and 'mse' in stats_results[model_name][h]:
                    mse_mean = stats_results[model_name][h]['mse']['mean']
                    mse_std = stats_results[model_name][h]['mse']['std']
                    mse_strs.append(f"{mse_mean:.4f}±{mse_std:.4f}")
                else:
                    mse_strs.append("N/A")
            
            # 计算128步改善幅度
            if model_name == "Transformer (baseline)":
                improvement = "Baseline"
            else:
                mse_128_mean = stats_results[model_name].get(128, {}).get('mse', {}).get('mean', None)
                if mse_128_mean is None or baseline_mse_128 <= 0:
                    improvement = "N/A"
                else:
                    improvement_pct = (baseline_mse_128 - mse_128_mean) / baseline_mse_128 * 100
                    improvement = f"{improvement_pct:+.1f}%"
            
            mse_str = " | ".join([f"{s:<12}" for s in mse_strs])
            row = f"{model_name:<28} | {mse_str} | {improvement}"
            print(row)
    
    print("="*120)
    print(f"注: 结果为{NUM_RUNS}次独立运行的 mean±std")
    
    # 结果落盘（使用统一实验目录）
    results_dir = str(results_dir_path)

    # 保存统计结果（包含mean, std, p-value）
    stats_json_path = os.path.join(results_dir, "statistical_results.json")
    try:
        with open(stats_json_path, "w", encoding="utf-8") as f:
            json.dump({
                'num_runs': NUM_RUNS,
                'random_seeds': RANDOM_SEEDS,
                'statistics': convert_to_serializable(stats_results),
                'significance_tests': convert_to_serializable(test_results)
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ 统计结果已保存: {stats_json_path}")
    except Exception as e:
        print(f"⚠️ 统计结果保存失败: {e}")

    # 保存运行元信息（用于复现与审计）
    trajectory_level_results = OrderedDict()
    for model_name, run_results in multi_run_results.items():
        trajectory_level_results[model_name] = OrderedDict()
        for horizon in PREDICTION_HORIZONS:
            ade_values = [
                run_result[horizon]["trajectory_window_ade"]
                for run_result in run_results
                if horizon in run_result and "trajectory_window_ade" in run_result[horizon]
            ]
            fde_values = [
                run_result[horizon]["trajectory_window_fde"]
                for run_result in run_results
                if horizon in run_result and "trajectory_window_fde" in run_result[horizon]
            ]
            trajectory_counts = [
                run_result[horizon]["trajectory_count"]
                for run_result in run_results
                if horizon in run_result and "trajectory_count" in run_result[horizon]
            ]
            if ade_values and fde_values:
                trajectory_level_results[model_name][horizon] = {
                    "trajectory_window_ade_mean": float(np.mean(ade_values)),
                    "trajectory_window_ade_std": float(np.std(ade_values)),
                    "trajectory_window_fde_mean": float(np.mean(fde_values)),
                    "trajectory_window_fde_std": float(np.std(fde_values)),
                    "trajectory_count": int(max(trajectory_counts)) if trajectory_counts else 0,
                }
    trajectory_json_path = os.path.join(results_dir, "trajectory_level_results.json")
    try:
        with open(trajectory_json_path, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(trajectory_level_results), f, indent=2, ensure_ascii=False)
    except PermissionError as exc:
        fallback_name = f"trajectory_level_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        trajectory_json_path = os.path.join(results_dir, fallback_name)
        with open(trajectory_json_path, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(trajectory_level_results), f, indent=2, ensure_ascii=False)
        print(f"WARNING: trajectory_level_results.json is not writable ({exc}); saved fallback: {trajectory_json_path}")
    print(f"鉁?杞ㄨ抗绾ц仛鍚堟寚鏍囧凡淇濆瓨: {trajectory_json_path}")

    metadata_paths = save_run_metadata(
        results_dir=results_dir_path,
        metadata={
            "experiment": "exp1_sota",
            "num_runs": NUM_RUNS,
            "random_seeds": RANDOM_SEEDS,
            "prediction_horizons": PREDICTION_HORIZONS,
            "data_subset_ratio": SOTA_DATA_SUBSET_RATIO,
            "report_on_physical_scale": REPORT_ON_PHYSICAL_SCALE,
            "report_mse_on_scaled": REPORT_MSE_ON_SCALED,
            "eval_ar_seed_mode": EVAL_AR_SEED_MODE,
            "eval_protocol": EVAL_PROTOCOL,
            "train_supervision_protocol": TRAIN_SUPERVISION_PROTOCOL,
            "optimizer_profile": OPTIMIZER_PROFILE,
            "strict_repro_mode": STRICT_REPRO_MODE,
            "comparison_models": list(selected_comparison_models.keys()),
            "comparison_model_provenance": build_comparison_model_provenance(selected_comparison_models),
            "train_config": _base_train_config,
            "model_config": MODEL_CONFIG,
            "physics_config": PHYSICS_CONFIG,
        },
    )
    print(f"✅ 运行元信息已保存: {metadata_paths['latest']}")

    # 保存均值结果（兼容旧格式）
    json_path = os.path.join(results_dir, "baseline_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(final_results), f, indent=2)

    csv_path = os.path.join(results_dir, "baseline_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Model"]
        for h in [32, 64, 128, 256]:
            header.extend([
                f"mse_{h}", f"mse_{h}_std", f"mae_{h}", f"mae_{h}_std",
                f"rmse_cart_m_{h}", f"rmse_cart_m_{h}_std",
                f"mse_cart_m2_{h}", f"mse_cart_m2_{h}_std",
                f"mae_cart_m_{h}", f"mae_cart_m_{h}_std",
                f"fde_{h}", f"fde_{h}_std", f"ade_{h}", f"ade_{h}_std",
            ])
        writer.writerow(header)
        for model_name in final_results.keys():
            if model_name in stats_results:
                row = [model_name]
                for h in [32, 64, 128, 256]:
                    horizon_stats = stats_results[model_name].get(h, {})
                    for metric_name in ["mse", "mae", "rmse_cart_m", "mse_cart_m2", "mae_cart_m", "fde", "ade"]:
                        metric_stats = horizon_stats.get(metric_name, {})
                        row.extend([metric_stats.get('mean', 0), metric_stats.get('std', 0)])
                writer.writerow(row)

    baseline_long_csv_path = os.path.join(results_dir, "baseline_results_long.csv")
    with open(baseline_long_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "phase", "seed", "scenario_type", "scenario_value",
            "model", "horizon", "metric", "value"
        ])
        for model_name in final_results.keys():
            if model_name not in stats_results:
                continue
            for h in PREDICTION_HORIZONS:
                horizon_stats = stats_results[model_name].get(h, {})
                for metric_name in EVAL_METRICS:
                    stats_item = horizon_stats.get(metric_name, {})
                    if "mean" in stats_item:
                        writer.writerow(["exp1_sota", "", "", "", "", model_name, h, f"{metric_name}_mean", stats_item["mean"]])
                    if "std" in stats_item:
                        writer.writerow(["exp1_sota", "", "", "", "", model_name, h, f"{metric_name}_std", stats_item["std"]])
    print(f"✅ 结果已保存: {json_path}, {csv_path}")
    print(f"✅ 统一长表已保存: {baseline_long_csv_path}")
    
    # Per-Maneuver评估
    if maneuver_labels_test is not None:
        print("\n" + "="*80)
        print("PER-MANEUVER TYPE PERFORMANCE ANALYSIS")
        print("="*80)
        
        # 加载测试数据并标准化（与主评估一致）
        data_path = str(get_dataset_npz_path(PROJECT_ROOT))
        data = load_hgv_dataset(data_path, require_trajectory_level=True).raw
        X_test_raw = data['X_test']
        y_test_raw = data['y_test']
        
        # 加载标准化器（与主流程一致）
        _data_dir = get_processed_data_dir(PROJECT_ROOT)
        scaler_path = str(get_input_scaler_path(PROJECT_ROOT))
        output_scaler_path = str(get_output_scaler_path(PROJECT_ROOT))
        scaler = joblib.load(scaler_path)
        output_scaler = joblib.load(output_scaler_path)
        
        # 标准化测试数据
        X_test_reshaped = X_test_raw.reshape(-1, X_test_raw.shape[-1])
        y_test_reshaped = y_test_raw.reshape(-1, y_test_raw.shape[-1])
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test_raw.shape)
        y_test_scaled = output_scaler.transform(y_test_reshaped).reshape(y_test_raw.shape)
        X_test_scaled = align_source_position_scale(
            X_test_scaled,
            input_mean=scaler.mean_,
            input_scale=scaler.scale_,
            output_mean=output_scaler.mean_,
            output_scale=output_scaler.scale_,
            output_dim=y_test_raw.shape[-1],
        )
        X_test = torch.from_numpy(X_test_scaled.astype(np.float32))
        y_test = torch.from_numpy(y_test_scaled.astype(np.float32))

        # 与主评估保持一致：若启用子集评估，per-maneuver 使用同一固定子集策略。
        if SOTA_DATA_SUBSET_RATIO is not None and SOTA_DATA_SUBSET_RATIO < 1.0:
            rng = np.random.default_rng(42)
            n_test = min(int(len(X_test) * SOTA_DATA_SUBSET_RATIO), len(X_test))
            test_idx = rng.choice(len(X_test), size=n_test, replace=False)
            X_test = X_test[test_idx]
            y_test = y_test[test_idx]
            maneuver_labels_test = maneuver_labels_test[test_idx]
        
        per_maneuver_results = {}
        
        # 对每个模型进行per-maneuver评估
        for model_name in final_results.keys():
            model_config = COMPARISON_MODELS[model_name]
            model_path = run_checkpoint_paths.get(
                model_name,
                get_model_save_path(model_config['model_type'], models_save_dir),
            )
            
            is_fixed_model = model_config['model_type'] in {'kinematic', 'rotating_3dof'}
            if is_fixed_model or (model_path and os.path.exists(model_path)):
                print(f"\n评估 {model_name} 的per-maneuver性能...")
                
                # 重新创建模型
                plgaformer_kwargs = _model_reconstruction_kwargs(
                    model_config['model_type'], scaler, output_scaler
                )
                model = create_model(
                    model_config['model_type'], input_dim=6, device=device,
                    plgaformer_kwargs=plgaformer_kwargs,
                )
                if not is_fixed_model:
                    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                model.eval()
                
                # 评估（传入标准化后的数据）
                maneuver_perf = evaluate_per_maneuver(
                    model, 
                    (X_test, y_test), 
                    maneuver_labels_test,
                    output_scaler_mean,
                    output_scaler_scale,
                    device,
                    horizon=max(PREDICTION_HORIZONS)
                )
                
                per_maneuver_results[model_name] = maneuver_perf
                
                # 打印结果
                for maneuver_type, metrics in maneuver_perf.items():
                    print(f"  {maneuver_type}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
                          f"FDE={metrics['fde']:.4f}, ADE={metrics['ade']:.4f}, N={metrics['num_samples']}")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 保存per-maneuver结果
        per_maneuver_json = os.path.join(results_dir, "per_maneuver_results.json")
        with open(per_maneuver_json, "w", encoding="utf-8") as f:
            json.dump(per_maneuver_results, f, indent=2)
        print(f"\n✅ Per-maneuver结果已保存: {per_maneuver_json}")
    
    # 计算效率测试
    print("\n" + "="*80)
    print("COMPUTATIONAL EFFICIENCY BENCHMARK")
    print("="*80)
    
    # 准备测试输入
    data_path = str(get_dataset_npz_path(PROJECT_ROOT))
    data = load_hgv_dataset(data_path, require_trajectory_level=True).raw
    X_test = data['X_test']
    test_batch = X_test[:32]  # 使用32个样本进行测试
    
    efficiency_results = {}
    
    for model_name in final_results.keys():
        model_config = COMPARISON_MODELS[model_name]
        model_path = run_checkpoint_paths.get(
            model_name,
            get_model_save_path(model_config['model_type'], models_save_dir),
        )
        
        is_fixed_model = model_config['model_type'] in {'kinematic', 'rotating_3dof'}
        if is_fixed_model or (model_path and os.path.exists(model_path)):
            print(f"\n测试 {model_name} 的计算效率...")
            
            # 重新创建模型
            plgaformer_kwargs = _model_reconstruction_kwargs(
                model_config['model_type'], scaler, output_scaler
            )
            model = create_model(
                model_config['model_type'], input_dim=6, device=device,
                plgaformer_kwargs=plgaformer_kwargs,
            )
            if not is_fixed_model:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            
            # 效率测试
            efficiency = benchmark_model_efficiency(model, test_batch, device, num_runs=100)
            efficiency_results[model_name] = efficiency
            
            print(f"  Latency: {efficiency['latency_ms']:.2f} ms")
            print(f"  Throughput: {efficiency['throughput_samples_per_sec']:.2f} samples/sec")
            print(f"  Memory: {efficiency['memory_mb']:.2f} MB")
            print(f"  Parameters: {efficiency['total_params']:,}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 保存效率结果
    efficiency_json = os.path.join(results_dir, "efficiency_results.json")
    with open(efficiency_json, "w", encoding="utf-8") as f:
        json.dump(efficiency_results, f, indent=2)
    print(f"\n✅ 效率测试结果已保存: {efficiency_json}")
    
    # 生成效率对比表格
    print("\n" + "="*80)
    print("Efficiency Comparison Summary")
    print("="*80)
    header = f"{'Model':<30} | {'Latency (ms)':<15} | {'Memory (MB)':<15} | {'Params (K)':<15}"
    print(header)
    print("-" * len(header))
    for model_name, eff in efficiency_results.items():
        row = f"{model_name:<30} | {eff['latency_ms']:<15.2f} | {eff['memory_mb']:<15.2f} | {eff['total_params']/1000:<15.1f}"
        print(row)
    print("="*80)

if __name__ == '__main__':
    print("Starting SOTA comparison experiment...")
    try:
        main()
        print("SOTA comparison experiment completed!")
    except Exception as e:
        print(f"SOTA comparison experiment error: {e}")
        import traceback
        traceback.print_exc()
