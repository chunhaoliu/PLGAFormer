#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLGAFormer Ablation Study

结构消融比较标准 Transformer、匹配主干、A、入选 C 结构以及
经验证淘汰的 A+C 与 A+B+C 候选。训练目标消融在固定 C 结构上比较 MSE-only
与无量纲物理目标。正式模式使用完整数据、预注册时域和轨迹级物理指标。
"""

# Version compatibility resolved: PyTorch 2.2.2 + NumPy 1.26.4 + Python 3.10
import warnings
# Filter out Flash Attention related warnings
warnings.filterwarnings('ignore', message='.*flash attention.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash attention.*', category=UserWarning)

# 修复多进程兼容性问题
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 将异常检测留到特定模型训练时启用
# torch.autograd.set_detect_anomaly(True)
import copy
import numpy as np
import time
import os
from collections import OrderedDict
import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import stats

# Set matplotlib to non-interactive mode to ensure charts don't pop up
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set matplotlib fonts for English display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False  # Display minus sign normally
plt.rcParams['figure.max_open_warning'] = 0  # Disable figure count warning

# Disable font-related warnings
# warnings已在文件开头导入，这里只设置过滤规则
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*font.*')

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Temporarily comment out complex models, use simplified baseline for debugging
# from main import PLGAFormer, PLGA_WINDOW_SIZE, PLGA_ADAPTIVE_WINDOW
# Import the physics loss function we need
# from models.physics_loss import PhysicsInformedLoss  # Commented out as we use local implementation
import sys
from pathlib import Path

# 项目根目录：所有数据/结果路径基于此
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import ECEFTrajectoryLoss, HGVConfig, create_registered_model, HGVPhysicsLoss
from utils.repro import set_global_seed, seed_worker, build_torch_generator, stable_seed_from_name
from utils.train_protocol import (
    should_update_best,
    build_optimizer_by_profile,
    build_scheduler_by_profile,
    clip_gradients_with_finite_check,
)
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
from utils.model_provenance import get_model_provenance, validate_comparison_model_types
from utils.trajectory_metrics import (
    aggregate_window_metrics_by_trajectory,
    holm_adjust,
    paired_permutation_test,
)
from utils.experiment_io import get_experiment_dirs, save_run_metadata
from utils.trajectory_protocol import dataset_protocol_name, validate_npz_trajectory_splits
from data_generation.data_paths import (
    get_dataset_npz_path,
    get_input_scaler_path,
    get_output_scaler_path,
    get_processed_data_dir,
)
from data_provider.hgv_data import load_hgv_dataset

# ======================================================================================
# 模型创建函数
# ======================================================================================

def _safe_filename_token(name: str) -> str:
    """Create a stable filesystem-safe token for experiment artifacts."""
    token = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(name))
    token = '_'.join(part for part in token.split('_') if part)
    return token or 'model'


def ablation_run_seed(seed: int, phase_name: str) -> int:
    """Use one deterministic seed per run/phase so variants are compared fairly."""
    return int(seed)

def create_model(
    model_type,
    input_dim=6,
    device=torch.device('cpu'),
    innovations=None,
    model_config_override=None,
    input_scaler_mean=None,
    input_scaler_scale=None,
    output_scaler_mean=None,
    output_scaler_scale=None,
):
    """
    创建不同类型的模型 - 公平消融实验设计

    - 基线（baseline）：使用标准 Transformer（StandardTransformer），
      唯一真相源 HGVConfig.get_model_config('transformer')。
    - PLGAFormer 及其消融变体：唯一真相源 HGVConfig.get_model_config('plgaformer')，
      通过 innovations 字典控制 A/B/C 开关。

    Args:
        model_type: 'baseline' 或 'plgaformer'
        input_dim: 输入维度（默认 6）
        device: 计算设备
        innovations: dict, 创新点开关，例如
            {'use_sparse_attention': True, 'use_physics_corrector': True, 'use_multi_head_output': True}
        model_config_override: dict, 可选，覆盖 dropout 等参数（消融时增强 dropout 减轻过拟合）
    """
    if model_type == 'baseline':
        return create_registered_model(
            model_type='baseline',
            input_dim=input_dim,
            device=device,
        )

    model_config = HGVConfig.get_model_config('plgaformer')
    overrides = model_config_override or {}
    plgaformer_kwargs = {
        'd_model': model_config.get('d_model', 256),
        'nhead': model_config.get('nhead', 8),
        'num_encoder_layers': model_config.get('num_encoder_layers', 3),
        'num_decoder_layers': model_config.get('num_decoder_layers', 2),
        'dim_feedforward': model_config.get('dim_feedforward', 1024),
        'dropout': overrides.get('dropout', model_config.get('dropout', 0.1)),
        'output_dim': model_config.get('output_dim', 3),
        # 默认全关，由 innovations 覆盖，符合消融实验需求
        'use_sparse_attention': False,
        'use_physics_corrector': False,
        'use_multi_head_output': False,
        'use_adaptive_fusion': False,
        'use_prior_fusion': True,
        'use_channel_residual': True,
    }
    if innovations:
        plgaformer_kwargs.update(innovations)
    plgaformer_kwargs.update({
        'input_scaler_mean': input_scaler_mean,
        'input_scaler_scale': input_scaler_scale,
        'output_scaler_mean': output_scaler_mean,
        'output_scaler_scale': output_scaler_scale,
        'sampling_interval_s': float(train_config.get('sampling_interval_s', 1.0)),
        'require_physical_scaler': bool(
            plgaformer_kwargs.get('use_sparse_attention', False)
            or plgaformer_kwargs.get('use_multi_head_output', False)
        ),
    })

    return create_registered_model(
        model_type='plgaformer',
        input_dim=input_dim,
        device=device,
        plgaformer_kwargs=plgaformer_kwargs,
    )

# ======================================================================================
# 实验配置 (Centralized Configuration)
# ======================================================================================

# 唯一真相源：SOTA 与消融均从 HGVConfig 读取，避免两套逻辑
train_config = HGVConfig.get_train_config()
_ablation_config = HGVConfig.get_experiment_config('ablation')

# Report configuration（与 SOTA 一致）
REPORT_ON_PHYSICAL_SCALE = True   # FDE/ADE 恒为物理空间（米）
REPORT_MSE_ON_SCALED = True      # MSE/MAE 在标准化空间（无量纲）；与 SOTA 主表一致
NUMERICAL_GUARDS = True          # 验证阶段数值保护（clamp、NaN/Inf 检查）
# 评估协议与 SOTA 保持一致（可切换）
EVAL_AR_SEED_MODE = train_config.get('eval_ar_seed_mode', 'zero')  # zero / gt_first
EVAL_PROTOCOL = train_config.get('eval_protocol', 'strict_autoregressive')
TRAIN_SUPERVISION_PROTOCOL = train_config.get('train_supervision_protocol', 'shifted_next_step')
OPTIMIZER_PROFILE = train_config.get('optimizer_profile', 'enhanced')
STRICT_REPRO_MODE = train_config.get('strict_repro_mode', False)

# 随机种子：仅在每轮运行开始时通过 set_random_seed(seed) 设置（见 run_ablation_phase），不在此处写死。

# 训练配置 - 从统一配置获取
# ======================================================================================
# 消融实验专用配置
# ======================================================================================
# 正式模式与主实验共享完整256步目标和32/64/128/256步评估时域；
# speed模式仅用于开发诊断，不作为论文证据。

# 消融实验专用参数（均从 HGVConfig 读取）
_subset_override = os.getenv('HGV_ABLATION_SUBSET_RATIO')
ABLATION_SUBSET_RATIO = (
    float(_subset_override)
    if _subset_override not in (None, '')
    else _ablation_config.get('data_subset_ratio')
)
ABLATION_PRED_LEN = 256
ABLATION_EPOCHS = int(os.getenv('HGV_ABLATION_EPOCHS', os.getenv('HGV_EPOCHS', train_config.get('epochs', 50))))
PHYSICS_DT = float(train_config.get('sampling_interval_s', 1.0))
_ABLATION_WARMUP_EPOCHS = int(train_config.get('warmup_epochs', 5))

# 多次运行配置：从 HGVConfig 统一读取（消融优先用 get_experiment_config('ablation')）
BASE_RANDOM_SEEDS = (
    train_config.get('random_seeds', [42, 123, 456])
    if os.getenv('HGV_RANDOM_SEEDS')
    else _ablation_config.get('random_seeds', train_config.get('random_seeds', [42, 123, 456]))
)
NUM_RUNS = int(os.getenv(
    'HGV_NUM_RUNS',
    _ablation_config.get('num_runs', train_config.get('num_runs', 3)),
))
RANDOM_SEEDS = list(BASE_RANDOM_SEEDS)[:NUM_RUNS]

# ======================================================================================
# 训练配置 - 全部从 HGVConfig 读取，与 SOTA 一致
# ======================================================================================

TRAIN_CONFIG = {
    'batch_size': int(train_config.get('batch_size', 64)),
    'epochs': ABLATION_EPOCHS,
    'lr': train_config['lr'],
    'lr_decay_factor': train_config.get('lr_decay_factor', 0.8),
    'weight_decay': train_config.get('weight_decay', 5e-5),
    'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
    'early_stopping_patience': train_config.get('early_stopping_patience', 15),
    'min_delta': train_config.get('min_delta', 1e-6),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_mixed_precision': bool(train_config.get('use_mixed_precision', True)),
    'gradient_clip_norm': train_config.get('gradient_clip_norm', 1.0),
    'dataloader_workers': train_config.get('dataloader_workers', 0),
    'pin_memory': train_config.get('pin_memory', False),
    'persistent_workers': train_config.get('persistent_workers', False),
    'prefetch_factor': train_config.get('prefetch_factor', 2),
}

def get_runtime_dataloader_config():
    """根据当前设备自动选择 DataLoader 配置，减少 CPU 侧供数瓶颈。"""
    use_cuda = (TRAIN_CONFIG.get('device') == 'cuda') and torch.cuda.is_available()
    if not use_cuda:
        return {
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': None,
        }

    workers = max(0, int(TRAIN_CONFIG.get('dataloader_workers', 0)))
    return {
        'num_workers': workers,
        'pin_memory': bool(TRAIN_CONFIG.get('pin_memory', True)),
        'persistent_workers': bool(TRAIN_CONFIG.get('persistent_workers', True)) and workers > 0,
        'prefetch_factor': int(TRAIN_CONFIG.get('prefetch_factor', 2)) if workers > 0 else None,
    }

# ======================================================================================
# 消融实验配置 - 常规消融（Standard Ablation）
# ======================================================================================
# 设计理念：从完整模型中逐一移除组件，观察性能下降 → 证明该组件有贡献
#
# Selected contribution: independently ablate the analytical prior and residual heads.
#
# 物理损失和 dropout 均来自统一主实验配置，避免消融混杂超参影响。

_ABLATION_PHYSICS_ALPHA = float(HGVConfig.get_physics_config().get('alpha', 1e-4))
_ABLATION_DROPOUT = float(HGVConfig.get_model_config('plgaformer').get('dropout', 0.1))

# 结构消融：匹配主干、独立组件、入选 C 与淘汰 A+C/A+B+C 候选
ABLATION_STRUCTURE_MODELS = OrderedDict([
    ("Transformer (baseline)", {
        'model_type': 'baseline',
        'description': 'Matched Transformer with the shared ECEF composite objective',
        'physics_loss_weight': 0.0,
        'innovations': None,
    }),
    ("PLGAFormer backbone (no A/B/C)", {
        'model_type': 'plgaformer',
        'description': 'Matched PLGAFormer backbone with all proposed components disabled',
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': False,
            'use_physics_corrector': False,
            'use_multi_head_output': False,
        },
    }),
    ("PLGAFormer prior fusion only", {
        'model_type': 'plgaformer',
        'description': 'Rotating-Earth prior fusion without the channel residual',
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': False,
            'use_physics_corrector': False,
            'use_multi_head_output': True,
            'use_prior_fusion': True,
            'use_channel_residual': False,
        },
    }),
    ("PLGAFormer channel residual only", {
        'model_type': 'plgaformer',
        'description': 'Bounded channel residual without the rotating-Earth prior fusion',
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': False,
            'use_physics_corrector': False,
            'use_multi_head_output': True,
            'use_prior_fusion': False,
            'use_channel_residual': True,
        },
    }),
    ("PLGAFormer (proposed)", {
        'model_type': 'plgaformer',
        'description': 'Rotating-Earth prior fusion with bounded channel residual',
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': False,
            'use_physics_corrector': False,
            'use_multi_head_output': True,
            'use_prior_fusion': True,
            'use_channel_residual': True,
        },
    }),
])

def build_ablation_models_with_physics_weight(base_models, physics_weight):
    """基于统一结构配置构建对照组，便于拆分结构贡献与物理损失贡献。"""
    models = OrderedDict()
    for model_name, cfg in base_models.items():
        new_cfg = copy.deepcopy(cfg)
        new_cfg['physics_loss_weight'] = float(physics_weight)
        if physics_weight <= 0:
            new_cfg['description'] = f"{cfg['description']} | shared ECEF composite objective"
        else:
            new_cfg['description'] = (
                f"{cfg['description']} | +dimensionless dynamics residual "
                f"(alpha={physics_weight:.1e})"
            )
        models[model_name] = new_cfg
    return models

# Structural isolation under the same ECEF composite objective.
ABLATION_STRUCTURE_MODELS_MSE_ONLY = build_ablation_models_with_physics_weight(
    ABLATION_STRUCTURE_MODELS, physics_weight=0.0
)
# 套B：结构 + 同权重物理约束
_PROPOSED_C_CONFIG = copy.deepcopy(ABLATION_STRUCTURE_MODELS["PLGAFormer (proposed)"])
ABLATION_OBJECTIVE_MODELS = OrderedDict([
    ("PLGAFormer (ECEF composite)", {
        **copy.deepcopy(_PROPOSED_C_CONFIG),
        'description': 'Selected architecture with the shared ECEF composite objective',
        'physics_loss_weight': 0.0,
    }),
    ("PLGAFormer (+ dynamics residual)", {
        **copy.deepcopy(_PROPOSED_C_CONFIG),
        'description': 'Selected architecture with an additional dimensionless dynamics residual',
        'physics_loss_weight': _ABLATION_PHYSICS_ALPHA,
    }),
])

ABLATION_PRIOR_MODELS = OrderedDict([
    ("Transformer (baseline)", {
        'model_type': 'baseline',
        'description': 'Matched Transformer control for attention-prior isolation',
        'physics_loss_weight': 0.0,
        'innovations': None,
    }),
    ("PLGAFormer A only", {
        'model_type': 'plgaformer',
        'description': 'Physics-aware attention only, with all three explicit priors',
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': True,
            'use_physics_corrector': False,
            'use_multi_head_output': False,
            'attention_prior_mask': (True, True, True),
        },
    }),
    ("A: temporal only", {
        'model_type': 'plgaformer',
        'description': 'Attention prior isolation: temporal continuity only',
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': True,
            'use_physics_corrector': False,
            'use_multi_head_output': False,
            'attention_prior_mask': (True, False, False),
        },
    }),
    ("A: phase only", {
        'model_type': 'plgaformer',
        'description': 'Attention prior isolation: phase coherence only',
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': True,
            'use_physics_corrector': False,
            'use_multi_head_output': False,
            'attention_prior_mask': (False, True, False),
        },
    }),
    ("A: geometry only", {
        'model_type': 'plgaformer',
        'description': 'Attention prior isolation: ECEF geometry only',
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': True,
            'use_physics_corrector': False,
            'use_multi_head_output': False,
            'attention_prior_mask': (False, False, True),
        },
    }),
    ("A w/o temporal", {
        'model_type': 'plgaformer',
        'description': 'All explicit attention priors except temporal continuity',
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': True,
            'use_physics_corrector': False,
            'use_multi_head_output': False,
            'attention_prior_mask': (False, True, True),
        },
    }),
    ("A w/o phase", {
        'model_type': 'plgaformer',
        'description': 'All explicit attention priors except phase coherence',
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': True,
            'use_physics_corrector': False,
            'use_multi_head_output': False,
            'attention_prior_mask': (True, False, True),
        },
    }),
    ("A w/o geometry", {
        'model_type': 'plgaformer',
        'description': 'All explicit attention priors except ECEF geometry',
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': True,
            'use_physics_corrector': False,
            'use_multi_head_output': False,
            'attention_prior_mask': (True, True, False),
        },
    }),
])

ABLATION_FINAL_MECHANISM_CONTROLS = OrderedDict([
    ("PLGAFormer spherical-prior fusion", {
        'model_type': 'plgaformer',
        'description': (
            'Replace the identified rotating-Earth 3-DOF proposal with '
            'spherical constant-velocity kinematics while retaining adaptive fusion'
        ),
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': False,
            'use_physics_corrector': False,
            'use_multi_head_output': True,
            'use_prior_fusion': True,
            'use_channel_residual': False,
            'prior_type': 'spherical_kinematic',
            'prior_blend_mode': 'adaptive',
        },
    }),
    ("PLGAFormer rotating-prior schedule only", {
        'model_type': 'plgaformer',
        'description': (
            'Retain the identified rotating-Earth 3-DOF proposal but remove '
            'learned state/disagreement gating from the fusion weight'
        ),
        'physics_loss_weight': 0.0,
        'warmup_epochs': _ABLATION_WARMUP_EPOCHS,
        'dropout': _ABLATION_DROPOUT,
        'innovations': {
            'use_sparse_attention': False,
            'use_physics_corrector': False,
            'use_multi_head_output': True,
            'use_prior_fusion': True,
            'use_channel_residual': False,
            'prior_type': 'rotating_3dof',
            'prior_blend_mode': 'schedule_only',
        },
    }),
])

ABLATION_PHASES = OrderedDict([
    ("phase1_structural", {
        'num_runs': NUM_RUNS,
        'random_seeds': RANDOM_SEEDS,
        'models': ABLATION_STRUCTURE_MODELS_MSE_ONLY,
    }),
    ("phase2_objective_isolation", {
        'num_runs': NUM_RUNS,
        'random_seeds': RANDOM_SEEDS,
        'models': ABLATION_OBJECTIVE_MODELS,
    }),
    ("phase3_attention_prior_isolation", {
        'num_runs': NUM_RUNS,
        'random_seeds': RANDOM_SEEDS,
        'models': ABLATION_PRIOR_MODELS,
    }),
    ("phase4_final_mechanism_controls", {
        'num_runs': NUM_RUNS,
        'random_seeds': RANDOM_SEEDS,
        'models': ABLATION_FINAL_MECHANISM_CONTROLS,
    }),
])


def _normalized_selection_token(value: str) -> str:
    return ''.join(character for character in str(value).lower() if character.isalnum())


def _ablation_model_aliases(model_name: str) -> set[str]:
    normalized = _normalized_selection_token(model_name)
    aliases = {normalized}
    lowered = model_name.lower()
    if 'baseline' in lowered:
        aliases.add('baseline')
    if 'backbone' in lowered:
        aliases.update({'backbone', 'noabc'})
    if '(c, proposed' in lowered or '(proposed)' in lowered:
        aliases.update({'full', 'c', 'conly', 'proposed'})
    if 'prior fusion only' in lowered:
        aliases.update({'prioronly', 'fusiononly'})
    if 'spherical-prior' in lowered:
        aliases.update({'sphericalprior'})
    if 'schedule only' in lowered:
        aliases.update({'scheduleonly'})
    if 'channel residual only' in lowered:
        aliases.update({'residualonly', 'channelonly'})
    if '(a+c' in lowered:
        aliases.update({'excluded', 'ac'})
    if '(a+b+c' in lowered:
        aliases.update({'excluded', 'abc'})
    for component in ('a', 'b', 'c'):
        if f'w/o {component}' in lowered:
            aliases.add(f'wo{component}')
    if 'a only' in lowered:
        aliases.add('aonly')
    if 'c only' in lowered:
        aliases.add('conly')
    return aliases


def _select_ablation_phases(phase_definitions: OrderedDict) -> OrderedDict:
    phase_request = os.getenv('HGV_ABLATION_PHASES', '').strip()
    model_request = os.getenv('HGV_ABLATION_MODELS', '').strip()
    requested_phases = {
        _normalized_selection_token(value)
        for value in phase_request.split(',') if value.strip()
    }
    requested_models = {
        _normalized_selection_token(value)
        for value in model_request.split(',') if value.strip()
    }
    selected = OrderedDict()
    for phase_name, phase_config in phase_definitions.items():
        phase_token = _normalized_selection_token(phase_name)
        if requested_phases and not any(
            token == phase_token or token in phase_token for token in requested_phases
        ):
            continue
        config_copy = copy.deepcopy(phase_config)
        if requested_models:
            config_copy['models'] = OrderedDict(
                (model_name, model_config)
                for model_name, model_config in config_copy['models'].items()
                if requested_models & _ablation_model_aliases(model_name)
            )
        if config_copy['models']:
            selected[phase_name] = config_copy
    if not selected:
        raise ValueError(
            'Ablation phase/model selection matched no configurations: '
            f'phases={phase_request!r}, models={model_request!r}.'
        )
    return selected


ABLATION_PHASES = _select_ablation_phases(ABLATION_PHASES)


def _build_phase_model_provenance(phase_defs: OrderedDict) -> dict:
    out = {}
    for phase_name, phase_cfg in phase_defs.items():
        phase_models = phase_cfg.get("models", {})
        out[phase_name] = {
            model_name: get_model_provenance(model_cfg.get("model_type", ""))
            for model_name, model_cfg in phase_models.items()
        }
    return out

def get_data_subset_for_experiment():
    """获取实验数据子集"""
    print("📊 Loading data for experiment...")
    return load_and_prepare_data(TRAIN_CONFIG['batch_size'])

# Match the primary experiment so component claims cover the full horizon.
PREDICTION_HORIZONS = [32, 64, 128, 256]

from itertools import product

# ======================================================================================
# 统计分析函数 - 多次运行结果的统计处理
# ======================================================================================

def compute_statistics(multi_run_results):
    """
    计算多次运行的统计量
    
    Args:
        multi_run_results: dict, {model_name: [{run1_metrics}, {run2_metrics}, ...]}
    
    Returns:
        stats_results: dict, {model_name: {metric: {'mean': x, 'std': y, 'values': [...]}}}
    """
    stats_results = {}
    
    for model_name, run_results in multi_run_results.items():
        stats_results[model_name] = {}
        
        # 提取所有预测长度的指标（仅 MSE/MAE/RMSE）
        for horizon in PREDICTION_HORIZONS:
            stats_results[model_name][horizon] = {}
            
            for metric in [
                'mse', 'mae', 'rmse', 'fde', 'ade',
                'trajectory_window_fde', 'trajectory_window_ade',
            ]:
                values = []
                for run_result in run_results:
                    if horizon in run_result and metric in run_result[horizon]:
                        values.append(run_result[horizon][metric])
                
                if len(values) > 0:
                    values_np = np.asarray(values, dtype=np.float64)
                    n = len(values_np)
                    mean = float(np.mean(values_np))
                    std = float(np.std(values_np, ddof=0))
                    median = float(np.median(values_np))
                    q1 = float(np.percentile(values_np, 25))
                    q3 = float(np.percentile(values_np, 75))
                    if n >= 2:
                        sem = float(std / np.sqrt(n))
                        t_crit = float(stats.t.ppf(0.975, n - 1))
                        ci95_low = mean - t_crit * sem
                        ci95_high = mean + t_crit * sem
                    else:
                        ci95_low = mean
                        ci95_high = mean
                    stats_results[model_name][horizon][metric] = {
                        'mean': mean,
                        'std': std,
                        'median': median,
                        'q1': q1,
                        'q3': q3,
                        'ci95_low': float(ci95_low),
                        'ci95_high': float(ci95_high),
                        'n': n,
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
        test_results: dict, {model_name: {metric: {'t_stat': t, 'p_value': p, 'significant': bool}}}
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
            
            for metric in ['mse', 'mae', 'rmse']:
                if (horizon in baseline_stats and metric in baseline_stats[horizon] and
                    horizon in model_stats and metric in model_stats[horizon]):
                    
                    baseline_values = baseline_stats[horizon][metric]['values']
                    model_values = model_stats[horizon][metric]['values']
                    
                    # 确保样本数相同
                    min_len = min(len(baseline_values), len(model_values))
                    if min_len >= 3:  # 至少3个样本再执行t检验，降低小样本误导
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
                            print(f"⚠️ t检验失败 ({model_name}, {metric}): {e}")
    
    return test_results

def perform_trajectory_level_tests(multi_run_results, baseline_name="Transformer (baseline)"):
    """Use held-out source trajectories, rather than seeds, as independent units."""
    if baseline_name not in multi_run_results:
        return {}

    def average_by_trajectory(run_results, horizon, metric):
        values_by_id = {}
        for run_result in run_results:
            for trajectory_id, item in run_result.get(horizon, {}).get('trajectory_metrics', {}).items():
                values_by_id.setdefault(int(trajectory_id), []).append(float(item[metric]))
        return {key: float(np.mean(values)) for key, values in values_by_id.items()}

    output = {}
    pending = []
    baseline_runs = multi_run_results[baseline_name]
    for model_name, model_runs in multi_run_results.items():
        if model_name == baseline_name:
            continue
        output[model_name] = {}
        for horizon in PREDICTION_HORIZONS:
            output[model_name][horizon] = {}
            for metric in ('ade', 'fde'):
                baseline_values = average_by_trajectory(baseline_runs, horizon, metric)
                model_values = average_by_trajectory(model_runs, horizon, metric)
                common_ids = sorted(set(baseline_values) & set(model_values))
                if len(common_ids) < 2:
                    continue
                comparison = paired_permutation_test(
                    np.asarray([model_values[key] for key in common_ids]),
                    np.asarray([baseline_values[key] for key in common_ids]),
                    seed=3100 + int(horizon),
                )
                comparison['independent_unit'] = 'held-out source trajectory'
                comparison['seed_aggregation'] = 'mean within trajectory before inference'
                output[model_name][horizon][metric] = comparison
                pending.append((comparison, float(comparison['p_value'])))
    if pending:
        adjusted = holm_adjust([item[1] for item in pending])
        for (comparison, _), adjusted_p in zip(pending, adjusted):
            comparison['p_value_holm'] = float(adjusted_p)
            comparison['significant_fwer_0_05'] = bool(adjusted_p < 0.05)
    return output


def save_phase_outputs_to_files(phase_outputs, output_dir):
    import csv
    import json

    os.makedirs(output_dir, exist_ok=True)

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    phases_payload = {}
    for phase_name, out in phase_outputs.items():
        phases_payload[phase_name] = {
            'num_runs': out.get('num_runs'),
            'random_seeds': out.get('random_seeds'),
            'models_to_run': convert_to_serializable(out.get('models_to_run', {})),
            'final_results': convert_to_serializable(out.get('final_results', {})),
            'statistics': convert_to_serializable(out.get('statistics', {})),
            'significance_tests': convert_to_serializable(out.get('significance_tests', {})),
            'multi_run_results': convert_to_serializable(out.get('multi_run_results', {})),
            'training_histories': convert_to_serializable(out.get('training_histories', {})),
        }

    latest_json_path = os.path.join(output_dir, "latest_results.json")
    with open(latest_json_path, 'w', encoding='utf-8') as f:
        json.dump({'base_random_seeds': BASE_RANDOM_SEEDS, 'phases': phases_payload}, f, indent=2, ensure_ascii=False)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamped_json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(timestamped_json_path, 'w', encoding='utf-8') as f:
        json.dump({'base_random_seeds': BASE_RANDOM_SEEDS, 'phases': phases_payload}, f, indent=2, ensure_ascii=False)

    latest_runs_csv_path = os.path.join(output_dir, "latest_runs.csv")
    with open(latest_runs_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "seed", "model", "horizon", "metric", "value"])
        for phase_name, out in phase_outputs.items():
            seeds = out.get('random_seeds') or []
            multi_run_results = out.get('multi_run_results') or {}
            for model_name, runs in multi_run_results.items():
                for run_idx, run_result in enumerate(runs):
                    seed = seeds[run_idx] if run_idx < len(seeds) else ""
                    for horizon, metrics in (run_result or {}).items():
                        for metric_name, value in (metrics or {}).items():
                            writer.writerow([phase_name, seed, model_name, horizon, metric_name, value])

    timestamped_runs_csv_path = os.path.join(output_dir, f"runs_{timestamp}.csv")
    try:
        import shutil
        shutil.copyfile(latest_runs_csv_path, timestamped_runs_csv_path)
    except Exception:
        pass

    latest_stats_csv_path = os.path.join(output_dir, "latest_stats.csv")
    with open(latest_stats_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "model", "horizon", "metric",
            "n", "mean", "std", "median", "q1", "q3", "ci95_low", "ci95_high", "values"
        ])
        for phase_name, out in phase_outputs.items():
            stats_results = out.get('statistics') or {}
            for model_name, model_stats in stats_results.items():
                for horizon, horizon_stats in model_stats.items():
                    for metric_name, stats in horizon_stats.items():
                        n = stats.get('n', "")
                        mean = stats.get('mean', "")
                        std = stats.get('std', "")
                        median = stats.get('median', "")
                        q1 = stats.get('q1', "")
                        q3 = stats.get('q3', "")
                        ci95_low = stats.get('ci95_low', "")
                        ci95_high = stats.get('ci95_high', "")
                        values = stats.get('values', [])
                        values_str = ",".join(str(v) for v in values)
                        writer.writerow([
                            phase_name, model_name, horizon, metric_name,
                            n, mean, std, median, q1, q3, ci95_low, ci95_high, values_str
                        ])

    timestamped_stats_csv_path = os.path.join(output_dir, f"stats_{timestamp}.csv")
    try:
        import shutil
        shutil.copyfile(latest_stats_csv_path, timestamped_stats_csv_path)
    except Exception:
        pass

    return {
        'latest_json': latest_json_path,
        'timestamped_json': timestamped_json_path,
        'latest_runs_csv': latest_runs_csv_path,
        'timestamped_runs_csv': timestamped_runs_csv_path,
        'latest_stats_csv': latest_stats_csv_path,
        'timestamped_stats_csv': timestamped_stats_csv_path,
    }

def print_statistical_summary(stats_results, test_results, num_runs, baseline_name="Transformer (baseline)"):
    """
    打印统计分析摘要
    """
    print(f"\n{'='*80}")
    print(f"📊 统计分析摘要 ({num_runs}次独立运行)")
    print(f"{'='*80}")
    
    for horizon in PREDICTION_HORIZONS:
        print(f"\n📏 预测长度: {horizon} 步")
        print(f"{'-'*128}")
        print(f"{'模型':<30} {'MSE(mean±std)':<20} {'MSE median [95%CI]':<30} {'MAE(mean±std)':<20} {'MAE median [95%CI]':<30} {'FDE p-Holm':<12}")
        print(f"{'-'*128}")
        
        for model_name, model_stats in stats_results.items():
            if horizon not in model_stats:
                continue
            
            mse_stats = model_stats[horizon].get('mse', {})
            mae_stats = model_stats[horizon].get('mae', {})
            
            mse_str = f"{mse_stats.get('mean', 0):.4f}±{mse_stats.get('std', 0):.4f}"
            mae_str = f"{mae_stats.get('mean', 0):.4f}±{mae_stats.get('std', 0):.4f}"
            mse_robust = (
                f"{mse_stats.get('median', 0):.4f} "
                f"[{mse_stats.get('ci95_low', 0):.4f},{mse_stats.get('ci95_high', 0):.4f}]"
            )
            mae_robust = (
                f"{mae_stats.get('median', 0):.4f} "
                f"[{mae_stats.get('ci95_low', 0):.4f},{mae_stats.get('ci95_high', 0):.4f}]"
            )
            
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
            
            print(f"{model_name:<30} {mse_str:<20} {mse_robust:<30} {mae_str:<20} {mae_robust:<30} {p_value_str:<12}")
        
        print(f"{'-'*128}")
        print(
            "注: MSE/MAE 为标准化空间；seed 仅描述优化波动；p 值来自留出源轨迹上的"
            "配对符号置换检验，并对全部模型、时域和 ADE/FDE 检验作 Holm 校正。"
        )

def set_random_seed(seed):
    """每轮运行开始时唯一调用的种子设置入口，确保可复现性；DataLoader 使用同一 seed 的 generator。
    使用 warn_only=True 与 SOTA 实验保持一致，避免某些操作缺少确定性实现时直接报错。
    """
    # 使用统一入口，避免各实验脚本出现细微差异。
    set_global_seed(seed, deterministic=True, strict_deterministic=STRICT_REPRO_MODE)

def validate_configuration():
    """验证配置参数的有效性"""
    errors = []
    warnings = []
    
    # 验证TRAIN_CONFIG - 只验证训练相关参数
    required_train_keys = ['batch_size', 'epochs', 'lr', 'device']
    for key in required_train_keys:
        if key not in TRAIN_CONFIG:
            errors.append(f"TRAIN_CONFIG缺少必需参数: {key}")
    
    # 验证数值范围
    if TRAIN_CONFIG.get('batch_size', 0) <= 0:
        errors.append("batch_size必须大于0")
    if TRAIN_CONFIG.get('epochs', 0) <= 0:
        errors.append("epochs必须大于0")
    if TRAIN_CONFIG.get('lr', 0) <= 0:
        errors.append("学习率lr必须大于0")
    if ABLATION_SUBSET_RATIO is not None and not (0.0 < ABLATION_SUBSET_RATIO <= 1.0):
        errors.append("ABLATION_SUBSET_RATIO必须在(0, 1]内或为None")
    
    # 验证设备配置
    device = TRAIN_CONFIG.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        warnings.append("配置使用CUDA但CUDA不可用，将自动切换到CPU")
        TRAIN_CONFIG['device'] = 'cpu'
    
    if not ABLATION_PHASES:
        errors.append("ABLATION_PHASES不能为空")
    else:
        for phase_name, phase_cfg in ABLATION_PHASES.items():
            if not isinstance(phase_cfg, dict):
                errors.append(f"Phase {phase_name} 配置必须是dict")
                continue

            models = phase_cfg.get('models')
            num_runs = phase_cfg.get('num_runs')
            seeds = phase_cfg.get('random_seeds')

            if not models:
                errors.append(f"Phase {phase_name} 的模型列表为空")
                continue
            if not isinstance(num_runs, int) or num_runs <= 0:
                errors.append(f"Phase {phase_name} 的num_runs必须是正整数")
            if not isinstance(seeds, list) or len(seeds) == 0:
                errors.append(f"Phase {phase_name} 的random_seeds不能为空")
            elif isinstance(num_runs, int) and num_runs > len(seeds):
                errors.append(f"Phase {phase_name} 的num_runs超过random_seeds长度")

            for model_name, config in models.items():
                if 'model_type' not in config:
                    errors.append(f"Phase {phase_name} 模型 {model_name} 缺少model_type配置")
                if 'physics_loss_weight' not in config:
                    warnings.append(f"Phase {phase_name} 模型 {model_name} 缺少physics_loss_weight配置，将使用默认值0.0")
                    config['physics_loss_weight'] = 0.0

                weight = config.get('physics_loss_weight', 0.0)
                if weight < 0 or weight > 1.0:
                    warnings.append(f"Phase {phase_name} 模型 {model_name} 的physics_loss_weight ({weight}) 超出推荐范围[0, 1]")
    
    # 验证PREDICTION_HORIZONS
    if not PREDICTION_HORIZONS:
        errors.append("PREDICTION_HORIZONS不能为空")
    for horizon in PREDICTION_HORIZONS:
        if not isinstance(horizon, int) or horizon <= 0:
            errors.append(f"预测步长 {horizon} 必须是正整数")
    
    # 输出验证结果
    if errors:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    if warnings:
        print("⚠️  配置警告:")
        for warning in warnings:
            print(f"   - {warning}")
    
    print("✅ 配置验证通过")
    return True

# ======================================================================================
# 数据加载 (Data Loading)
# ======================================================================================

def load_and_prepare_data(batch_size, subset_ratio=None, initial_load_seed=None):
    """加载基于物理仿真的数据集并返回scaler - 支持数据子集以加速消融实验。
    初始加载阶段使用的种子由 initial_load_seed 指定（仅用于子集采样与初始 DataLoader）；
    每轮训练种子在 run_ablation_phase 内通过 set_random_seed(seed) 单入口设置。
    
    Args:
        batch_size: 批次大小
        subset_ratio: 数据子集比例（0-1之间），None表示使用全部数据
        initial_load_seed: 初始加载阶段种子（子集采样与初始 DataLoader）；None 则用 BASE_RANDOM_SEEDS[0]
    """
    if initial_load_seed is None:
        initial_load_seed = BASE_RANDOM_SEEDS[0]
    # 使用消融实验专用的子集比例
    if subset_ratio is None:
        subset_ratio = ABLATION_SUBSET_RATIO
    
    # 检查数据文件是否存在（路径基于 PROJECT_ROOT）
    _data_dir = get_processed_data_dir(PROJECT_ROOT)
    data_path = str(get_dataset_npz_path(PROJECT_ROOT))

    if not os.path.exists(data_path):
        print("🔄 数据文件不存在，自动调用数据生成器...")
        import subprocess
        try:
            gen_script = str(PROJECT_ROOT / 'data_generation' / 'data_generator.py')
            result = subprocess.run([sys.executable, gen_script], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ 数据生成完成")
            else:
                print(f"⚠️  数据生成警告: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⚠️  数据生成超时，但可能已部分完成")
        except Exception as e:
            print(f"⚠️  数据生成过程中出现问题: {e}")
    else:
        print("✅ 发现现有数据文件，直接加载...")
    
    scaler = None
    try:
        # 加载数据集
        print(f"📂 加载数据集: {data_path}")
        data_bundle = load_hgv_dataset(data_path, require_trajectory_level=True)
        data = data_bundle.raw
        trajectory_ids_test = np.asarray(data['trajectory_ids_test'], dtype=np.int64)
        
        print(f"数据集规模信息:")
        print(f"  训练集: {data['X_train'].shape}")
        print(f"  验证集: {data['X_val'].shape}")  
        print(f"  测试集: {data['X_test'].shape}")
        protocol = data_bundle.protocol
        split_report = data_bundle.split_report
        if split_report is not None:
            print(f"  数据协议: {protocol} | 轨迹级划分重叠检查: {split_report}")
        else:
            print(f"  数据协议: {protocol}")
        
        # 验证数据集规模
        total_samples = len(data['X_train']) + len(data['X_val']) + len(data['X_test'])
        print(f"  总样本数: {total_samples:,}")
        print(f"  输入序列长度: {data['X_train'].shape[1]}")
        print(f"  预测序列长度: {data['y_train'].shape[1]}")
        print(f"  输入特征维度: {data['X_train'].shape[2]}")
        print(f"  输出特征维度: {data['y_train'].shape[2]}")
        
        # 检查序列长度是否与配置一致
        from models import HGVConfig
        train_cfg = HGVConfig.get_train_config()
        expected_input_len = train_cfg.get('seq_len', 64)
        expected_output_len = train_cfg.get('pred_len', 256)
        if data['X_train'].shape[1] != expected_input_len:
            print(f"⚠️  输入序列长度不匹配: 期望 {expected_input_len}, 实际 {data['X_train'].shape[1]}")
        # 注：输出长度可能小于pred_len（评估时按需截取）
        
        # 创建标准化器
        scaler_path = str(get_input_scaler_path(PROJECT_ROOT))
        
        try:
            scaler = joblib.load(scaler_path)
            print(f"✅ 已加载现有标准化器: {scaler_path}")
        except FileNotFoundError:
            print("🔧 创建新的标准化器...")
            scaler = StandardScaler()
            
            # 仅使用训练 split 拟合，避免验证/测试轨迹信息进入标准化器。
            train_input_data = data['X_train'].reshape(-1, data['X_train'].shape[-1])
            scaler.fit(train_input_data)
            
            # 保存标准化器
            _data_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            print(f"✅ 新标准化器已保存到: {scaler_path}")
        
    except FileNotFoundError as e:
        print(f"❌ 数据文件未找到: {e}")
        print("请检查数据生成器是否正常运行")
        return None, None, None, None, None
    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        return None, None, None, None, None

    # 应用标准化处理到数据
    print("🔧 应用数据标准化处理...")
    
    # 标准化输入数据
    n_train, in_len, in_dim = data['X_train'].shape
    n_val = data['X_val'].shape[0]
    n_test = data['X_test'].shape[0]
    X_train_2d = data['X_train'].reshape(-1, in_dim)
    X_val_2d   = data['X_val'].reshape(-1, in_dim)
    X_test_2d  = data['X_test'].reshape(-1, in_dim)
    X_train_scaled_2d = scaler.transform(X_train_2d)
    X_val_scaled_2d   = scaler.transform(X_val_2d)
    X_test_scaled_2d  = scaler.transform(X_test_2d)
    X_train_scaled = X_train_scaled_2d.reshape(n_train, in_len, in_dim).astype(np.float32, copy=False)
    X_val_scaled   = X_val_scaled_2d.reshape(n_val, in_len, in_dim).astype(np.float32, copy=False)
    X_test_scaled  = X_test_scaled_2d.reshape(n_test, in_len, in_dim).astype(np.float32, copy=False)
    
    # 创建输出专用的标准化器
    output_scaler_path = str(get_output_scaler_path(PROJECT_ROOT))
    
    out_dim = data['y_train'].shape[-1]
    try:
        output_scaler = joblib.load(output_scaler_path)
        print(f"✅ 已加载现有输出标准化器: {output_scaler_path}")
        if output_scaler.mean_.shape[0] != out_dim:
            raise ValueError("output_scaler dim mismatch")
    except (FileNotFoundError, ValueError):
        print("🔧 为输出数据创建专用标准化器...")
        output_scaler = StandardScaler()
        train_y_data = data['y_train'].reshape(-1, out_dim)
        output_scaler.fit(train_y_data)
        
        _data_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(output_scaler, output_scaler_path)
        print(f"✅ 输出标准化器已保存到: {output_scaler_path}")

    # 标准化输出数据
    n_train_y, out_len, out_dim = data['y_train'].shape
    n_val_y = data['y_val'].shape[0]
    n_test_y = data['y_test'].shape[0]
    y_train_2d = data['y_train'].reshape(-1, out_dim)
    y_val_2d   = data['y_val'].reshape(-1, out_dim)
    y_test_2d  = data['y_test'].reshape(-1, out_dim)
    y_train_scaled_2d = output_scaler.transform(y_train_2d)
    y_val_scaled_2d   = output_scaler.transform(y_val_2d)
    y_test_scaled_2d  = output_scaler.transform(y_test_2d)
    y_train_scaled = y_train_scaled_2d.reshape(n_train_y, out_len, out_dim).astype(np.float32, copy=False)
    y_val_scaled   = y_val_scaled_2d.reshape(n_val_y, out_len, out_dim).astype(np.float32, copy=False)
    y_test_scaled  = y_test_scaled_2d.reshape(n_test_y, out_len, out_dim).astype(np.float32, copy=False)

    # Keep decoder label/context semantics aligned with Autoformer/FEDformer:
    # source position channels are provided to the decoder in the same normalized
    # space as future targets, while non-position source features keep input scaling.
    X_train_scaled = align_source_position_scale(
        X_train_scaled,
        input_mean=scaler.mean_,
        input_scale=scaler.scale_,
        output_mean=output_scaler.mean_,
        output_scale=output_scaler.scale_,
        output_dim=out_dim,
    ).astype(np.float32, copy=False)
    X_val_scaled = align_source_position_scale(
        X_val_scaled,
        input_mean=scaler.mean_,
        input_scale=scaler.scale_,
        output_mean=output_scaler.mean_,
        output_scale=output_scaler.scale_,
        output_dim=out_dim,
    ).astype(np.float32, copy=False)
    X_test_scaled = align_source_position_scale(
        X_test_scaled,
        input_mean=scaler.mean_,
        input_scale=scaler.scale_,
        output_mean=output_scaler.mean_,
        output_scale=output_scaler.scale_,
        output_dim=out_dim,
    ).astype(np.float32, copy=False)

    maneuver_labels_train = None
    maneuver_labels_val = None
    maneuver_labels_test = None
    if 'maneuver_labels_train' in data and 'maneuver_labels_val' in data and 'maneuver_labels_test' in data:
        if (len(data['maneuver_labels_train']) == n_train and
            len(data['maneuver_labels_val']) == n_val and
            len(data['maneuver_labels_test']) == n_test):
            maneuver_labels_train = data['maneuver_labels_train']
            maneuver_labels_val = data['maneuver_labels_val']
            maneuver_labels_test = data['maneuver_labels_test']
    elif 'maneuver_labels' in data:
        all_labels = data['maneuver_labels']
        if len(all_labels) == n_train:
            maneuver_labels_train = all_labels
        elif len(all_labels) == (n_train + n_val + n_test):
            maneuver_labels_train = all_labels[:n_train]
            maneuver_labels_val = all_labels[n_train:n_train + n_val]
            maneuver_labels_test = all_labels[n_train + n_val:n_train + n_val + n_test]
        else:
            print("⚠️  maneuver_labels长度与数据集不匹配，分层采样将回退为随机采样")
    
    # 转换为torch tensor 
    X_train = torch.from_numpy(X_train_scaled.astype(np.float32))
    y_train = torch.from_numpy(y_train_scaled.astype(np.float32))
    X_val = torch.from_numpy(X_val_scaled.astype(np.float32))
    y_val = torch.from_numpy(y_val_scaled.astype(np.float32))
    X_test = torch.from_numpy(X_test_scaled.astype(np.float32))
    y_test = torch.from_numpy(y_test_scaled.astype(np.float32))
    
    # 🚀 消融实验优化：使用数据子集加速训练
    if subset_ratio is not None and subset_ratio < 1.0:
        print(f"🔧 消融实验优化：使用 {subset_ratio*100:.0f}% 数据子集")

        def stratified_indices(labels, subset_size, rng):
            if labels is None:
                return None
            labels = np.asarray(labels)
            if subset_size >= len(labels):
                return np.arange(len(labels))
            unique, counts = np.unique(labels, return_counts=True)
            per_class = np.floor(counts * (subset_size / len(labels))).astype(int)
            per_class = np.maximum(per_class, 1)
            per_class = np.minimum(per_class, counts)
            current = int(per_class.sum())
            if current > subset_size:
                excess = current - subset_size
                reducible = per_class > 1
                reducible_indices = np.where(reducible)[0]
                if len(reducible_indices) > 0:
                    reduce_order = rng.permutation(reducible_indices)
                    for idx in reduce_order:
                        if excess <= 0:
                            break
                        can_reduce = per_class[idx] - 1
                        delta = min(can_reduce, excess)
                        per_class[idx] -= delta
                        excess -= delta
            elif current < subset_size:
                deficit = subset_size - current
                available = counts - per_class
                while deficit > 0 and np.any(available > 0):
                    candidates = np.where(available > 0)[0]
                    pick = rng.choice(candidates)
                    per_class[pick] += 1
                    available[pick] -= 1
                    deficit -= 1

            chosen = []
            for u, k in zip(unique, per_class):
                class_idx = np.where(labels == u)[0]
                if k >= len(class_idx):
                    chosen.append(class_idx)
                else:
                    chosen.append(rng.choice(class_idx, size=int(k), replace=False))
            return np.concatenate(chosen, axis=0)

        rng = np.random.default_rng(initial_load_seed)
        n_train_subset = int(len(X_train) * subset_ratio)
        n_val_subset = int(len(X_val) * subset_ratio)
        n_test_subset = int(len(X_test) * subset_ratio)

        train_indices = stratified_indices(maneuver_labels_train, n_train_subset, rng)
        val_indices = stratified_indices(maneuver_labels_val, n_val_subset, rng)
        test_indices = stratified_indices(maneuver_labels_test, n_test_subset, rng)

        if train_indices is None:
            train_indices = rng.choice(len(X_train), n_train_subset, replace=False)
        if val_indices is None:
            val_indices = rng.choice(len(X_val), n_val_subset, replace=False)
        if test_indices is None:
            test_indices = rng.choice(len(X_test), n_test_subset, replace=False)

        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_val = X_val[val_indices]
        y_val = y_val[val_indices]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]
        trajectory_ids_test = trajectory_ids_test[test_indices]
        
        print(f"    子集训练集: {len(X_train):,} 样本")
        print(f"    子集验证集: {len(X_val):,} 样本")
        print(f"    子集测试集: {len(X_test):,} 样本")
    
    # 🚀 消融实验优化：截取预测长度到 ABLATION_PRED_LEN
    if y_train.shape[1] > ABLATION_PRED_LEN:
        print(f"🔧 消融实验优化：预测长度从 {y_train.shape[1]} 截取到 {ABLATION_PRED_LEN}")
        y_train = y_train[:, :ABLATION_PRED_LEN, :]
        y_val = y_val[:, :ABLATION_PRED_LEN, :]
        y_test = y_test[:, :ABLATION_PRED_LEN, :]
    
    print(f"✅ 数据标准化完成")
    print(f"    标准化后数值范围检查:")
    print(f"    X_train: min={X_train.min().item():.3f}, max={X_train.max().item():.3f}")
    print(f"    y_train: min={y_train.min().item():.3f}, max={y_train.max().item():.3f}")
    
    # 创建数据集和DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(
        X_test,
        y_test,
        torch.from_numpy(np.asarray(trajectory_ids_test, dtype=np.int64)),
    )

    dataloader_cfg = get_runtime_dataloader_config()
    print(
        f"🔧 DataLoader配置: workers={dataloader_cfg['num_workers']}, "
        f"pin_memory={dataloader_cfg['pin_memory']}, "
        f"persistent_workers={dataloader_cfg['persistent_workers']}"
    )

    generator = build_torch_generator(initial_load_seed)
    train_loader_kwargs = {
        'num_workers': dataloader_cfg['num_workers'],
        'pin_memory': dataloader_cfg['pin_memory'],
        'generator': generator,
    }
    eval_loader_kwargs = {
        'num_workers': dataloader_cfg['num_workers'],
        'pin_memory': dataloader_cfg['pin_memory'],
    }
    if dataloader_cfg['num_workers'] > 0:
        train_loader_kwargs['persistent_workers'] = dataloader_cfg['persistent_workers']
        train_loader_kwargs['prefetch_factor'] = dataloader_cfg['prefetch_factor']
        train_loader_kwargs['worker_init_fn'] = seed_worker
        eval_loader_kwargs['persistent_workers'] = dataloader_cfg['persistent_workers']
        eval_loader_kwargs['prefetch_factor'] = dataloader_cfg['prefetch_factor']
        eval_loader_kwargs['worker_init_fn'] = seed_worker

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **train_loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **eval_loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **eval_loader_kwargs
    )
    
    print(f"✅ 数据集加载成功")
    print(f"    训练集: {len(X_train):,} 样本")
    print(f"    验证集: {len(X_val):,} 样本") 
    print(f"    测试集: {len(X_test):,} 样本")
    print(f"    输入序列长度: {X_train.shape[1]}")
    print(f"    预测序列长度: {y_train.shape[1]}")
    
    return train_loader, val_loader, test_loader, scaler, output_scaler

# ======================================================================================
# 训练与评估 (重构以支持反标准化)
# ======================================================================================

def unscale_data(scaled_data, mean, scale):
    """[FIX] 辅助函数：使用torch操作将tensor反标准化，保留计算图"""
    # Broadcasting should handle the dimensions correctly
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

def generate_causal_mask(size, device):
    """生成用于解码器的因果掩码"""
    mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def build_supervision_windows(
    tgt: torch.Tensor,
    protocol: str,
    src: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return build_seq2seq_supervision_windows(
        tgt=tgt,
        protocol=protocol,
        pred_len=int(ABLATION_PRED_LEN),
        label_len=int(train_config.get('label_len', max(1, min(48, tgt.size(1))))),
        src=src,
    )

def manage_gpu_memory(device, batch_idx, force_cleanup_threshold=6.0, monitor_threshold=4.0):
    """管理GPU内存使用（已移除废弃的 memory_cached API）"""
    if device.type == 'cuda':
        # 低频清理：避免频繁同步导致吞吐下降
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
        
        if batch_idx % 50 == 0:
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
            
            if memory_allocated > force_cleanup_threshold:
                torch.cuda.empty_cache()
                print(f"⚠️  内存清理: 分配 {memory_allocated:.2f}GB, 保留 {memory_reserved:.2f}GB")
            elif memory_allocated > monitor_threshold and batch_idx % 200 == 0:
                print(f"📊 内存状态: 分配 {memory_allocated:.2f}GB, 保留 {memory_reserved:.2f}GB")

def check_tensor_validity(tensor, name="tensor"):
    """检查张量的数值稳定性"""
    if torch.isnan(tensor).any():
        print(f"⚠️  检测到NaN值在 {name}")
        return False
    if torch.isinf(tensor).any():
        print(f"⚠️  检测到Inf值在 {name}")
        return False
    return True

def run_evaluation(model, loader, scaler_mean, scaler_scale, device, horizons):
    """Evaluate windows and retain trajectory-level displacement statistics."""
    model.eval()
    sums = {
        h: {
            'sq_sum': 0.0, 'abs_sum': 0.0, 'n_elem': 0,
            'cart_sq_sum': 0.0, 'cart_abs_sum': 0.0, 'cart_n_elem': 0,
            'fde_sum': 0.0, 'fde_n': 0, 'ade_sum': 0.0, 'ade_n': 0,
        }
        for h in horizons
    }
    trajectory_cache = {h: {'pred': [], 'true': [], 'ids': []} for h in horizons}
    device_obj = torch.device(device) if isinstance(device, str) else device
    max_horizon = max(horizons)

    with torch.no_grad():
        for batch in loader:
            x, y_true_scaled = batch[:2]
            trajectory_ids = batch[2] if len(batch) > 2 else None
            to_cuda = device_obj.type == 'cuda'
            x = x.to(device, non_blocking=to_cuda)
            y_true_scaled = y_true_scaled.to(device, non_blocking=to_cuda)
            if y_true_scaled.size(1) > max_horizon:
                y_true_scaled = y_true_scaled[:, :max_horizon, :]
            pred_length = min(y_true_scaled.size(1), max_horizon)
            y_pred_scaled = unified_predict_by_eval_protocol(
                model=model,
                x=x,
                y_true_scaled=y_true_scaled,
                pred_length=pred_length,
                device=device,
                eval_protocol=EVAL_PROTOCOL,
                eval_ar_seed_mode=EVAL_AR_SEED_MODE,
                label_len=int(train_config.get('label_len', max(1, min(48, y_true_scaled.size(1))))),
                causal_mask_builder=generate_causal_mask,
                oneshot_models=[],
            )
            for h in horizons:
                actual_h = min(h, y_pred_scaled.size(1), y_true_scaled.size(1))
                if actual_h <= 0:
                    continue
                pred_h_scaled = y_pred_scaled[:, :actual_h, :]
                true_h_scaled = y_true_scaled[:, :actual_h, :]
                pred_h_phys = unscale_data(pred_h_scaled, scaler_mean, scaler_scale)
                true_h_phys = unscale_data(true_h_scaled, scaler_mean, scaler_scale)
                diff_mse = (
                    pred_h_scaled - true_h_scaled
                    if REPORT_MSE_ON_SCALED
                    else pred_h_phys - true_h_phys
                )
                sums[h]['sq_sum'] += float(diff_mse.square().sum())
                sums[h]['abs_sum'] += float(diff_mse.abs().sum())
                sums[h]['n_elem'] += diff_mse.numel()

                pred_cart = spherical_to_cartesian(pred_h_phys[..., :3])
                true_cart = spherical_to_cartesian(true_h_phys[..., :3])
                cart_diff = pred_cart - true_cart
                displacement = torch.linalg.vector_norm(cart_diff, dim=-1)
                sums[h]['cart_sq_sum'] += float(cart_diff.square().sum())
                sums[h]['cart_abs_sum'] += float(cart_diff.abs().sum())
                sums[h]['cart_n_elem'] += cart_diff.numel()
                sums[h]['fde_sum'] += float(displacement[:, -1].sum())
                sums[h]['fde_n'] += displacement.size(0)
                sums[h]['ade_sum'] += float(displacement.sum())
                sums[h]['ade_n'] += displacement.numel()
                if trajectory_ids is not None:
                    trajectory_cache[h]['pred'].append(pred_cart.detach().cpu().numpy())
                    trajectory_cache[h]['true'].append(true_cart.detach().cpu().numpy())
                    trajectory_cache[h]['ids'].append(trajectory_ids.detach().cpu().numpy())

    results = {}
    for h in horizons:
        n_elem = sums[h]['n_elem']
        if n_elem <= 0:
            results[h] = {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}
            continue
        mse = sums[h]['sq_sum'] / n_elem
        mse_cart_m2 = sums[h]['cart_sq_sum'] / sums[h]['cart_n_elem']
        results[h] = {
            'mse': mse,
            'mae': sums[h]['abs_sum'] / n_elem,
            'rmse': float(np.sqrt(mse)),
            'mse_cart_m2': mse_cart_m2,
            'mae_cart_m': sums[h]['cart_abs_sum'] / sums[h]['cart_n_elem'],
            'rmse_cart_m': float(np.sqrt(mse_cart_m2)),
            'fde': sums[h]['fde_sum'] / sums[h]['fde_n'],
            'ade': sums[h]['ade_sum'] / sums[h]['ade_n'],
        }
        if trajectory_cache[h]['ids']:
            report = aggregate_window_metrics_by_trajectory(
                np.concatenate(trajectory_cache[h]['pred'], axis=0),
                np.concatenate(trajectory_cache[h]['true'], axis=0),
                np.concatenate(trajectory_cache[h]['ids'], axis=0),
            )
            results[h]['trajectory_window_ade'] = report['mean_ade']
            results[h]['trajectory_window_fde'] = report['mean_fde']
            results[h]['trajectory_count'] = report['trajectory_count']
            results[h]['trajectory_ade_ci'] = report['ade_ci']
            results[h]['trajectory_fde_ci'] = report['fde_ci']
            results[h]['trajectory_metrics'] = {
                str(key): value for key, value in report['trajectories'].items()
            }
    return results

def train_model(model_name, model_config, train_loader, val_loader, scaler_mean, scaler_scale, model_override=None):
    """完整的模型训练流程 - 优化版"""
    print("-" * 50)
    print(f"训练模型: {model_name}")
    
    device = torch.device(TRAIN_CONFIG['device'])
    
    # 验证输入参数
    if model_override is None:
        raise ValueError("model_override is required for training")
    
    if not hasattr(model_override, 'parameters'):
        raise ValueError("model_override must be a valid PyTorch model")
    
    model = model_override.to(device)
    
    # 验证数据加载器
    if train_loader is None or val_loader is None:
        raise ValueError("train_loader and val_loader cannot be None")
    
    # 验证scaler参数
    if scaler_mean is None or scaler_scale is None:
        raise ValueError("scaler_mean and scaler_scale are required for evaluation")

    prior_cache_stats = {
        "train": {"enabled": False, "reason": "disabled_by_config"},
        "validation": {"enabled": False, "reason": "disabled_by_config"},
    }
    if bool(TRAIN_CONFIG.get("cache_physics_prior", True)):
        cache_batch_size = int(
            TRAIN_CONFIG.get("physics_prior_cache_batch_size", train_loader.batch_size or 128)
        )
        train_loader, train_prior_cache = build_cached_physics_prior_loader(
            train_loader,
            model,
            device=device,
            target_length=int(ABLATION_PRED_LEN),
            cache_batch_size=cache_batch_size,
        )
        val_loader, val_prior_cache = build_cached_physics_prior_loader(
            val_loader,
            model,
            device=device,
            target_length=int(ABLATION_PRED_LEN),
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
    
    # 损失函数选择
    physics_loss_weight = model_config.get('physics_loss_weight', 0.0)
    physics_alpha = model_config.get('physics_alpha', physics_loss_weight)
    use_physics_loss = physics_loss_weight > 0.0
    
    primary_criterion = ECEFTrajectoryLoss(
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        distance_scale_m=float(TRAIN_CONFIG.get('ecef_distance_scale_m', 100_000.0)),
        scaled_mse_weight=float(TRAIN_CONFIG.get('scaled_mse_weight', 0.1)),
    ).to(device)
    physics_criterion = None
    if use_physics_loss:
        physics_criterion = HGVPhysicsLoss(alpha=physics_alpha, scaler_mean=scaler_mean, scaler_std=scaler_scale)
    
    # 统一学习率和正则化策略 - 所有模型使用相同的参数
    base_lr = TRAIN_CONFIG['lr']  # 所有模型使用相同的学习率
    weight_decay = TRAIN_CONFIG['weight_decay']
    warmup_epochs = model_config.get('warmup_epochs', TRAIN_CONFIG.get('warmup_epochs', 5))
    total_epochs = TRAIN_CONFIG['epochs']
    
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
    
    # 梯度缩放器（基于配置的混合精度训练）
    device_obj = torch.device(device) if isinstance(device, str) else device
    use_mixed_precision = TRAIN_CONFIG.get('use_mixed_precision', True)
    mixed_precision_dtype_name = str(
        TRAIN_CONFIG.get('mixed_precision_dtype', 'bfloat16')
    ).lower()
    mixed_precision_dtype = (
        torch.bfloat16 if mixed_precision_dtype_name == 'bfloat16' else torch.float16
    )
    scaler = (
        torch.amp.GradScaler(
            'cuda',
            init_scale=128.0,
            enabled=mixed_precision_dtype == torch.float16,
        )
        if (device_obj.type == 'cuda' and use_mixed_precision)
        else None
    )
    
    best_val_loss = float('inf')
    best_state_dict = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    physics_gate_history = []
    epoch_records = []
    best_epoch = None
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"物理约束权重: {physics_alpha}")
    
    total_batches_per_epoch = len(train_loader)
    NAN_WARN_LIMIT = 5  # 同类 warning 仅打印前 5 次

    for epoch in range(TRAIN_CONFIG['epochs']):
        epoch_start_time = time.time()
        epoch_train_samples = 0
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            src, tgt, physics_prior_override = unpack_batch_with_optional_prior(batch)
            to_cuda = (device.type == 'cuda')
            src = src.to(device, non_blocking=to_cuda)
            tgt = tgt.to(device, non_blocking=to_cuda)
            if physics_prior_override is not None:
                physics_prior_override = physics_prior_override.to(device, non_blocking=to_cuda)
            if not torch.isfinite(src).all() or not torch.isfinite(tgt).all():
                raise FloatingPointError(f"Non-finite training input at batch {batch_idx}.")
            if src.size(0) == 0 or tgt.size(0) == 0:
                raise RuntimeError(f"Empty training batch at index {batch_idx}.")
            
            # 准备输入数据
            tgt_input, tgt_output = build_supervision_windows(tgt, TRAIN_SUPERVISION_PROTOCOL, src=src)
            
            # 生成掩码
            tgt_mask = generate_causal_mask(tgt_input.size(1), device)
            
            optimizer.zero_grad()
            
            # 混合精度训练（添加数值稳定性检查）
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=mixed_precision_dtype):
                    outputs = seq2seq_training_forward(
                        model=model,
                        src=src,
                        decoder_input=tgt_input,
                        tgt_mask=tgt_mask,
                        supervision_protocol=TRAIN_SUPERVISION_PROTOCOL,
                        pred_len=int(ABLATION_PRED_LEN),
                        oneshot_models=[],
                        model_forward_kwargs={"physics_prior_override": physics_prior_override}
                        if physics_prior_override is not None else None,
                    )
                    
                    # 模型输出数值稳定性检查
                    if not torch.isfinite(outputs).all():
                        raise FloatingPointError(f"Non-finite model output at batch {batch_idx}.")
                    
                    if outputs.size(1) != tgt_output.size(1):
                        outputs = outputs[:, -tgt_output.size(1):, :]
                    loss_mse = primary_criterion(outputs, tgt_output)
                
                # 损失数值稳定性检查
                    loss = loss_mse
                if physics_criterion is not None:
                    # 使用内置反归一化的物理损失计算
                    loss_phys = physics_criterion(outputs.float(), tgt_output.float(), dt=PHYSICS_DT, include_mse=False)
                    loss = loss + loss_phys
                
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at batch {batch_idx}.")
                
                scaler.scale(loss).backward()
                
                # 先尝试unscale以检查梯度
                try:
                    scaler.unscale_(optimizer)
                    
                    clip_gradients_with_finite_check(
                        model,
                        TRAIN_CONFIG['gradient_clip_norm'],
                        context=f"ablation model={model_name}, batch={batch_idx}, amp=True",
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    
                except RuntimeError as e:
                    optimizer.zero_grad()
                    raise RuntimeError(f"AMP unscale failed at batch {batch_idx}: {e}") from e
            else:
                outputs = seq2seq_training_forward(
                    model=model,
                    src=src,
                    decoder_input=tgt_input,
                    tgt_mask=tgt_mask,
                    supervision_protocol=TRAIN_SUPERVISION_PROTOCOL,
                    pred_len=int(ABLATION_PRED_LEN),
                    oneshot_models=[],
                    model_forward_kwargs={"physics_prior_override": physics_prior_override}
                    if physics_prior_override is not None else None,
                )
                
                # 模型输出数值稳定性检查
                if not torch.isfinite(outputs).all():
                    raise FloatingPointError(f"Non-finite model output at batch {batch_idx}.")
                
                if outputs.size(1) != tgt_output.size(1):
                    outputs = outputs[:, -tgt_output.size(1):, :]
                loss_mse = primary_criterion(outputs, tgt_output)
                loss = loss_mse
                if physics_criterion is not None:
                    # 使用内置反归一化的物理损失计算
                    loss_phys = physics_criterion(outputs.float(), tgt_output.float(), dt=PHYSICS_DT, include_mse=False)
                    loss = loss + loss_phys
                
                # 损失数值稳定性检查
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at batch {batch_idx}.")
            
                loss.backward()
                
                clip_gradients_with_finite_check(
                    model,
                    TRAIN_CONFIG['gradient_clip_norm'],
                    context=f"ablation model={model_name}, batch={batch_idx}, amp=False",
                )
                
                optimizer.step()
            
            batch_size = src.size(0)
            epoch_train_loss += loss.item() * batch_size
            epoch_train_samples += batch_size
            num_batches += 1
            
            # 智能GPU内存管理优化
            if device.type == 'cuda':
                # 避免高频 empty_cache() 触发 CUDA 同步，仅在高占用时清理
                if batch_idx % 50 == 0:
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
                    
                    # 内存使用率超过阈值时进行强制清理
                    if memory_allocated > 20.0:
                        torch.cuda.empty_cache()
            
            # 移除batch级别打印
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src, tgt, physics_prior_override = unpack_batch_with_optional_prior(batch)
                to_cuda = (device.type == 'cuda')
                src = src.to(device, non_blocking=to_cuda)
                tgt = tgt.to(device, non_blocking=to_cuda)
                if physics_prior_override is not None:
                    physics_prior_override = physics_prior_override.to(device, non_blocking=to_cuda)
                
                tgt_input, tgt_output = build_supervision_windows(tgt, TRAIN_SUPERVISION_PROTOCOL, src=src)
                tgt_mask = generate_causal_mask(tgt_input.size(1), device)
                
                if scaler is not None:
                    with torch.amp.autocast(device_type='cuda', dtype=mixed_precision_dtype):
                        outputs = seq2seq_training_forward(
                            model=model,
                            src=src,
                            decoder_input=tgt_input,
                            tgt_mask=tgt_mask,
                            supervision_protocol=TRAIN_SUPERVISION_PROTOCOL,
                            pred_len=int(ABLATION_PRED_LEN),
                            oneshot_models=[],
                            model_forward_kwargs={"physics_prior_override": physics_prior_override}
                            if physics_prior_override is not None else None,
                        )
                else:
                    outputs = seq2seq_training_forward(
                        model=model,
                        src=src,
                        decoder_input=tgt_input,
                        tgt_mask=tgt_mask,
                        supervision_protocol=TRAIN_SUPERVISION_PROTOCOL,
                        pred_len=int(ABLATION_PRED_LEN),
                        oneshot_models=[],
                        model_forward_kwargs={"physics_prior_override": physics_prior_override}
                        if physics_prior_override is not None else None,
                    )
                
                if not torch.isfinite(outputs).all():
                    raise FloatingPointError(
                        f"Non-finite validation output: model={model_name}, epoch={epoch}."
                    )
                
                if outputs.size(1) != tgt_output.size(1):
                    outputs = outputs[:, -tgt_output.size(1):, :]
                loss_mse = primary_criterion(outputs, tgt_output)
                loss = loss_mse
                if physics_criterion is not None:
                    loss_phys = physics_criterion(outputs.float(), tgt_output.float(), dt=PHYSICS_DT, include_mse=False)
                    loss = loss + loss_phys
                
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite validation loss: model={model_name}, epoch={epoch}."
                    )
                
                batch_size = src.size(0)
                epoch_val_loss += loss.item() * batch_size
                val_samples += batch_size
                val_batches += 1
        
        if num_batches != total_batches_per_epoch or epoch_train_samples <= 0:
            raise RuntimeError(
                f"Incomplete training epoch {epoch}: {num_batches}/{total_batches_per_epoch} batches."
            )
        if val_batches != len(val_loader) or val_samples <= 0:
            raise RuntimeError(
                f"Incomplete validation epoch {epoch}: {val_batches}/{len(val_loader)} batches."
            )
        avg_train_loss = epoch_train_loss / epoch_train_samples
        avg_val_loss = epoch_val_loss / val_samples
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        if hasattr(model, 'physics_gate_snapshot'):
            gate_snapshot = model.physics_gate_snapshot()
            gate_snapshot['epoch'] = int(epoch + 1)
            physics_gate_history.append(gate_snapshot)
        
        current_lr = float(optimizer.param_groups[0]['lr'])

        # 早停检查
        min_delta = TRAIN_CONFIG.get('min_delta', 1e-6)
        if should_update_best(avg_val_loss, best_val_loss, min_delta=min_delta):
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = int(epoch + 1)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 计算epoch运行时间
        epoch_time = time.time() - epoch_start_time
        
        # 打印进度（每个epoch都打印）
        print(f'Epoch {epoch}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, LR={current_lr:.6f}, Time={epoch_time:.2f}s')
        epoch_records.append({
            'epoch': int(epoch + 1),
            'train_loss': float(avg_train_loss),
            'val_loss': float(avg_val_loss),
            'learning_rate': float(current_lr),
            'epoch_time_sec': float(epoch_time),
            'train_batches': int(num_batches),
            'val_batches': int(val_batches),
            'train_samples': int(epoch_train_samples),
            'val_samples': int(val_samples),
            'amp_scale': float(scaler.get_scale()) if scaler is not None else None,
            'is_best': bool(best_epoch == epoch + 1),
            'physics_gate_snapshot': gate_snapshot if hasattr(model, 'physics_gate_snapshot') else None,
        })

        scheduler.step()
        
        # 早停
        if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
            print(f'早停于 epoch {epoch}')
            break

    # 恢复最佳模型（与 SOTA 一致）
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
            
    print(f'✅ {model_name} 训练完成 (最佳验证损失: {best_val_loss:.6f})')
    
    # 返回模型和训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'epochs_completed': len(epoch_records),
        'epoch_records': epoch_records,
        'physics_gate_history': physics_gate_history,
        'best_model_gate_snapshot': model.physics_gate_snapshot()
        if hasattr(model, 'physics_gate_snapshot') else None,
        'physics_prior_cache': prior_cache_stats,
        'mixed_precision': bool(scaler is not None),
        'mixed_precision_dtype': mixed_precision_dtype_name
        if scaler is not None else 'float32',
    }
    
    return model, training_history

# ======================================================================================
# 主执行流程 (Main Execution Flow)
# ======================================================================================

def run_ablation_phase(
    phase_name,
    models_to_run,
    train_loader,
    val_loader,
    test_loader,
    scaler_mean,
    scaler_scale,
    input_scaler_mean,
    input_scaler_scale,
    device,
    num_runs,
    random_seeds,
    models_save_dir,
):
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}")
    print(f"{'='*60}")

    total_models = len(models_to_run)
    multi_run_results = {model_name: [] for model_name in models_to_run.keys()}
    all_training_histories = {model_name: [] for model_name in models_to_run.keys()}

    seeds_to_use = list(random_seeds)[:num_runs]
    for run_idx, seed in enumerate(seeds_to_use):
        print(f"\n{'#'*80}")
        print(f"# 第 {run_idx + 1}/{num_runs} 次运行 (随机种子: {seed})")
        print(f"{'#'*80}")

        # 单入口：本 run 仅在此处设置随机种子；DataLoader 使用同一 seed 的 generator
        set_random_seed(seed)

        run_results = OrderedDict()
        run_histories = {}
        completed_models = 0

        for model_idx, (model_name, model_config) in enumerate(models_to_run.items(), 1):
            print(f"\n{'='*50}")
            print(f"🚀 [Phase {phase_name}] [Run {run_idx+1}/{num_runs}] [{model_idx}/{total_models}] 训练模型: {model_name}")
            print(f"配置: {model_config['description']}")
            print(f"{'='*50}")

            try:
                model_seed = ablation_run_seed(seed, phase_name)
                set_random_seed(model_seed)
                generator = build_torch_generator(model_seed)
                loader_kwargs = {
                    'num_workers': train_loader.num_workers,
                    'pin_memory': train_loader.pin_memory,
                    'generator': generator
                }
                if train_loader.num_workers > 0:
                    loader_kwargs['persistent_workers'] = train_loader.persistent_workers
                    loader_kwargs['prefetch_factor'] = train_loader.prefetch_factor
                    loader_kwargs['worker_init_fn'] = seed_worker
                train_loader_model = DataLoader(
                    train_loader.dataset,
                    batch_size=train_loader.batch_size,
                    shuffle=True,
                    **loader_kwargs
                )

                model_type = model_config['model_type']
                start_time = time.time()

                model = create_model(
                    model_type=model_type,
                    input_dim=6,
                    device=device,
                    innovations=model_config.get('innovations'),
                    model_config_override=model_config,
                    input_scaler_mean=input_scaler_mean,
                    input_scaler_scale=input_scaler_scale,
                    output_scaler_mean=scaler_mean,
                    output_scaler_scale=scaler_scale,
                )

                if model is None:
                    raise RuntimeError(f"模型创建失败: {model_name}")

                trained_model, training_history = train_model(
                    model_name,
                    model_config,
                    train_loader_model,
                    val_loader,
                    scaler_mean,
                    scaler_scale,
                    model_override=model
                )

                if trained_model is None:
                    raise RuntimeError(f"模型训练失败: {model_name}")

                run_histories[model_name] = training_history

                print(f"📊 开始评估模型: {model_name}")
                eval_results = run_evaluation(
                    trained_model,
                    test_loader,
                    scaler_mean,
                    scaler_scale,
                    device,
                    PREDICTION_HORIZONS
                )

                if not eval_results:
                    raise RuntimeError(f"模型评估失败: {model_name}")

                run_results[model_name] = eval_results

                if run_idx == num_runs - 1:
                    os.makedirs(models_save_dir, exist_ok=True)
                    phase_token = _safe_filename_token(phase_name)
                    model_token = _safe_filename_token(model_name)
                    save_path = os.path.join(
                        models_save_dir,
                        f"best_{model_type}_{phase_token}_{model_token}_ablation.pth",
                    )
                    try:
                        torch.save(trained_model.state_dict(), save_path)
                        print(f"  ✓ 模型已保存: {save_path}")
                    except Exception as e:
                        raise RuntimeError(f"模型保存失败: {model_name}: {e}") from e

                training_time = time.time() - start_time
                horizon_key = max(eval_results.keys())
                metrics = eval_results.get(horizon_key, {})
                mse = metrics.get('mse', float('nan'))
                mae = metrics.get('mae', float('nan'))

                completed_models += 1

                print(f"✅ 模型 {model_name} 训练完成")
                print(f"⏱️  训练时间: {training_time:.2f}秒")
                print(f"📈 Pred-{horizon_key} MSE: {mse:.4f}, MAE: {mae:.4f}")
                print(f"📊 进度: {completed_models}/{total_models} 模型完成")

            except Exception as model_error:
                print(f"❌ 模型 {model_name} 处理过程中发生错误: {model_error}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"Ablation phase {phase_name} failed for model {model_name}."
                ) from model_error

        for model_name, results in run_results.items():
            multi_run_results[model_name].append(results)
        for model_name, history in run_histories.items():
            all_training_histories[model_name].append(history)

        torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"📊 统计分析 (Phase {phase_name}, {num_runs}次独立运行)")
    print(f"{'='*80}")

    stats_results = compute_statistics(multi_run_results)
    test_results = perform_trajectory_level_tests(
        multi_run_results,
        baseline_name="Transformer (baseline)",
    )
    print_statistical_summary(stats_results, test_results, num_runs=num_runs, baseline_name="Transformer (baseline)")

    final_results = OrderedDict()
    for model_name, model_stats in stats_results.items():
        final_results[model_name] = {}
        for horizon in PREDICTION_HORIZONS:
            if horizon in model_stats:
                final_results[model_name][horizon] = {
                    metric: data['mean']
                    for metric, data in model_stats[horizon].items()
                }

    return {
        'phase_name': phase_name,
        'num_runs': num_runs,
        'random_seeds': seeds_to_use,
        'models_to_run': models_to_run,
        'multi_run_results': multi_run_results,
        'training_histories': all_training_histories,
        'statistics': stats_results,
        'significance_tests': test_results,
        'final_results': final_results,
        'total_models': total_models
    }

def main(external_config=None):
    """Main function: Run systematic ablation study to validate three innovations
    
    Args:
        external_config: 外部传入的配置字典，用于AutoML超参数优化
    """
    
    # 设置多进程保护
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过，忽略错误
    
    # 若有外部配置（如 AutoML），仅更新 TRAIN_CONFIG；模型配置仍以 HGVConfig.get_model_config 为唯一真相源
    global TRAIN_CONFIG
    if external_config is not None:
        print("🤖 使用AutoML提供的配置参数（仅影响训练超参，模型配置仍从 HGVConfig 读取）")
        for key, value in external_config.items():
            if key in TRAIN_CONFIG:
                TRAIN_CONFIG[key] = value
    
    train_mode_name = os.getenv(
        'HGV_TRAIN_MODE',
        HGVConfig.TRAIN_MODE if hasattr(HGVConfig, 'TRAIN_MODE') else 'custom',
    )
    print(f"\n{'='*60}")
    print(f"PLGAFormer 消融实验 ({train_mode_name} 模式)")
    print(f"{'='*60}")
    print(f"🚀 优化配置:")
    subset_percent = 100.0 if ABLATION_SUBSET_RATIO is None else 100.0 * ABLATION_SUBSET_RATIO
    print(f"   - 数据子集: {subset_percent:.0f}%")
    print(f"   - 预测长度: {ABLATION_PRED_LEN} 步（与主实验一致）")
    print(f"   - 训练轮数: {ABLATION_EPOCHS} epochs")
    print(f"   - Batch Size: {TRAIN_CONFIG['batch_size']}")
    print(f"   - 评估步长: {PREDICTION_HORIZONS}")
    if external_config is not None:
        print(f"AutoML配置: batch_size={TRAIN_CONFIG['batch_size']}, lr={TRAIN_CONFIG['lr']:.2e}")
    else:
        model_cfg = HGVConfig.get_model_config('plgaformer')
        print(
            f"模型配置: d_model={model_cfg['d_model']}, nhead={model_cfg['nhead']}, "
            f"{model_cfg['num_encoder_layers']}+{model_cfg['num_decoder_layers']}层"
        )
    for phase_name, phase_cfg in ABLATION_PHASES.items():
        num_runs = phase_cfg.get('num_runs', NUM_RUNS)
        seeds = phase_cfg.get('random_seeds', RANDOM_SEEDS)
        unknown_model_types = validate_comparison_model_types(phase_cfg.get('models', {}))
        if unknown_model_types:
            raise ValueError(f"Unknown model_type(s) in {phase_name}: {unknown_model_types}")
        print(f"   - {phase_name}: {num_runs} runs (seeds: {list(seeds)[:num_runs]})")
    print(f"{'='*60}")
    
    # 验证配置
    if not validate_configuration():
        return {'error': 'Configuration validation failed'}

    # 初始加载阶段：仅在此处设置一次种子，使数据子集与初始 DataLoader 可复现；每轮训练种子在 run_ablation_phase 内设置
    set_random_seed(BASE_RANDOM_SEEDS[0])
    # 加载数据 - 简化输出
    print("📦 加载数据集...")
    train_loader, val_loader, test_loader, x_scaler, y_scaler = load_and_prepare_data(
        TRAIN_CONFIG['batch_size'], initial_load_seed=BASE_RANDOM_SEEDS[0]
    )
    if train_loader is None:
        return {'error': 'Data loading failed'}

    # 数据加载比例可视化（与 SOTA 一致，便于复现）
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    ratio_pct = (ABLATION_SUBSET_RATIO * 100) if (ABLATION_SUBSET_RATIO is not None and ABLATION_SUBSET_RATIO < 1.0) else 100
    print(f"📊 消融实验数据: 使用 {ratio_pct:.0f}% 子集 | 训练 {n_train:,} | 验证 {n_val:,} | 测试 {n_test:,}")

    # [FIX] 将scaler的参数转换为torch tensor以用于GPU计算 (版本兼容性已解决)
    device = TRAIN_CONFIG['device']
    scaler_mean = torch.from_numpy(y_scaler.mean_.astype(np.float32)).to(device)
    scaler_scale = torch.from_numpy(y_scaler.scale_.astype(np.float32)).to(device)
    source_scaler_mean = np.concatenate([y_scaler.mean_, x_scaler.mean_[3:]]).astype(np.float32)
    source_scaler_scale = np.concatenate([y_scaler.scale_, x_scaler.scale_[3:]]).astype(np.float32)

    phase_outputs = OrderedDict()
    results_dir_path, models_dir_path = get_experiment_dirs(PROJECT_ROOT, "exp2_ablation")
    results_dir = str(results_dir_path)
    models_save_dir = str(models_dir_path)
    for phase_name, phase_cfg in ABLATION_PHASES.items():
        phase_outputs[phase_name] = run_ablation_phase(
            phase_name=phase_name,
            models_to_run=phase_cfg['models'],
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            input_scaler_mean=source_scaler_mean,
            input_scaler_scale=source_scaler_scale,
            device=device,
            num_runs=phase_cfg['num_runs'],
            random_seeds=phase_cfg['random_seeds'],
            models_save_dir=models_save_dir,
        )
    try:
        saved_paths = save_phase_outputs_to_files(phase_outputs, results_dir)
        print(f"✓ 结果已保存: {saved_paths['latest_json']}")
        print(f"✓ 结果已保存: {saved_paths['latest_runs_csv']}")
        print(f"✓ 结果已保存: {saved_paths['latest_stats_csv']}")
    except Exception as e:
        print(f"⚠️ 结果保存失败: {e}")

    figures_dir = results_dir
    os.makedirs(figures_dir, exist_ok=True)
    for phase_name, phase_out in phase_outputs.items():
        if not phase_out:
            continue
        final_results = phase_out.get('final_results', {})
        stats_results = phase_out.get('statistics', {})
        all_training_histories = phase_out.get('training_histories', {})
        phase_runs = phase_out.get('num_runs', NUM_RUNS)
        phase_seeds = phase_out.get('random_seeds', [])

        print(f"\n{'='*70}")
        print(f"PLGAFormer 结构消融实验结果 ({phase_name})")
        print(f"{'='*70}")

        if final_results:
            try:
                print(f"\n📊 生成可视化结果（含误差棒，{phase_runs}次运行）...")
                generate_ablation_study_visualization(
                    final_results, figures_dir, stats_results=stats_results, num_runs=phase_runs,
                    filename_prefix=f"{phase_name}_ablation_study_comprehensive"
                )

                # 1) 每个模型仅展示最后一次运行曲线
                last_run_histories = {}
                for model_name, histories in all_training_histories.items():
                    if histories:
                        last_run_histories[model_name] = histories[-1]
                if last_run_histories:
                    print(f"\n📈 生成训练曲线（最后一次运行）...")
                    plot_training_curves(
                        last_run_histories, figures_dir,
                        filename_prefix=f"{phase_name}_training_curves_last_run",
                        title_suffix=phase_name
                    )

                # 2) run-level 曲线：每个模型展示所有 seed 的验证曲线
                print(f"\n📈 生成 run-level 训练曲线（按 seed）...")
                plot_run_level_training_curves(
                    all_training_histories, phase_seeds, figures_dir,
                    filename_prefix=f"{phase_name}_training_curves_runs",
                    title_suffix=phase_name
                )

                print(f"✅ 结构消融可视化已保存到: {figures_dir}")
            except Exception as viz_error:
                print(f"❌ 可视化生成失败: {viz_error}")

        baseline_name = "Transformer (baseline)"
        if baseline_name in stats_results:
            summary_horizon = max(PREDICTION_HORIZONS)
            baseline_mse = stats_results[baseline_name][summary_horizon].get('mse', {}).get('mean', 0)
            if baseline_mse > 0:
                print(f"\n📈 {phase_name} vs Baseline at horizon {summary_horizon} (MSE 改善%):")
                vs_models = [
                    "PLGAFormer backbone (no A/B/C)",
                    "PLGAFormer A only",
                    "PLGAFormer (C, proposed)",
                    "PLGAFormer (A+C, excluded)",
                    "PLGAFormer (A+B+C, excluded)",
                ]
                for model_name in vs_models:
                    if model_name in stats_results:
                        mse = stats_results[model_name][summary_horizon].get('mse', {}).get('mean', 0)
                        improvement = (baseline_mse - mse) / baseline_mse * 100
                        print(f"   {model_name}: {improvement:+.1f}%")

    metadata_paths = save_run_metadata(
        results_dir=results_dir_path,
        metadata={
            "experiment": "exp2_ablation",
            "base_num_runs": NUM_RUNS,
            "base_random_seeds": BASE_RANDOM_SEEDS,
            "ablation_subset_ratio": ABLATION_SUBSET_RATIO,
            "prediction_horizons": PREDICTION_HORIZONS,
            "eval_protocol": EVAL_PROTOCOL,
            "eval_ar_seed_mode": EVAL_AR_SEED_MODE,
            "train_supervision_protocol": TRAIN_SUPERVISION_PROTOCOL,
            "optimizer_profile": OPTIMIZER_PROFILE,
            "strict_repro_mode": STRICT_REPRO_MODE,
            "phase_definitions": ABLATION_PHASES,
            "phase_model_provenance": _build_phase_model_provenance(ABLATION_PHASES),
            "train_config": TRAIN_CONFIG,
            "model_config": HGVConfig.get_model_config('plgaformer'),
            "physics_config": HGVConfig.get_physics_config(),
        },
    )
    print(f"✅ 运行元信息已保存: {metadata_paths['latest']}")

    if external_config is not None:
        best_mse = float('inf')
        target_phase = phase_outputs.get('phase2_objective_isolation') or next(iter(phase_outputs.values()), None)
        if target_phase and target_phase.get('final_results'):
            for model_name, results in target_phase['final_results'].items():
                target_horizon = max(PREDICTION_HORIZONS)
                if target_horizon in results:
                    mse = results[target_horizon].get('mse', float('inf'))
                    if mse < best_mse:
                        best_mse = mse
            return {
                'val_loss': best_mse,
                'completed_models': target_phase.get('total_models', 0),
                'total_models': target_phase.get('total_models', 0),
                'final_results': target_phase['final_results']
            }
        return {'val_loss': float('inf'), 'error': 'No successful results'}

    return None


def generate_ablation_study_visualization(final_results, save_dir, stats_results=None, num_runs=None, filename_prefix="ablation_study_comprehensive"):
    """
    生成消融实验可视化 - 仅 MSE/MAE（标准化空间）
    
    Args:
        final_results: 均值结果字典
        save_dir: 保存目录
        stats_results: 统计结果字典（包含mean和std），用于绘制误差棒
    """
    os.makedirs(save_dir, exist_ok=True)
    
    main_horizon = max(PREDICTION_HORIZONS)
    models_data = []
    
    # Structural main line for the validation-selected C model.
    model_order = [
        ("Transformer (baseline)", "Baseline"),
        ("PLGAFormer backbone (no A/B/C)", "Matched backbone"),
        ("PLGAFormer A only", "A only"),
        ("PLGAFormer (C, proposed)", "C proposed"),
        ("PLGAFormer (A+C, excluded)", "A+C excluded"),
        ("PLGAFormer (A+B+C, excluded)", "A+B+C excluded"),
    ]

    for model_key, display_name in model_order:
        if model_key in final_results and main_horizon in final_results[model_key]:
            mae_val = final_results[model_key][main_horizon].get('mae', float('nan'))
            mse_val = final_results[model_key][main_horizon].get('mse', float('nan'))

            if stats_results and model_key in stats_results and main_horizon in stats_results[model_key]:
                mae_std = stats_results[model_key][main_horizon].get('mae', {}).get('std', 0)
                mse_std = stats_results[model_key][main_horizon].get('mse', {}).get('std', 0)
            else:
                mae_std = mse_std = 0

            models_data.append((display_name, mae_val, mse_val, mae_std, mse_std))
    
    if len(models_data) < 2:
        print("⚠️ 数据不足，无法生成可视化")
        return
    
    baseline_mae = models_data[0][1]
    baseline_mse = models_data[0][2]

    model_names = [item[0] for item in models_data]
    mae_values = [item[1] for item in models_data]
    rmse_values = [np.sqrt(item[2]) for item in models_data]
    mae_stds = [item[3] for item in models_data]
    rmse_stds = [np.sqrt(item[4]) / 2 if item[4] > 0 else 0 for item in models_data]
    
    # 配色：Baseline 蓝、Full 红、w/o 绿（顶刊标准）
    colors = []
    for name in model_names:
        if name == 'PLGAFormer (Full)':
            colors.append('#d62728')  # 红色（完整模型）
        elif name == 'Baseline':
            colors.append('#1f77b4')  # 蓝色
        elif 'w/o' in name:
            colors.append('#2ca02c')  # 绿色（消融变体）
        else:
            colors.append('#7f7f7f')  # 灰色（默认）
    
    # 设置matplotlib为顶刊标准
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 1.0
    })
    
    # 生成综合图 - 1x2 (MAE, RMSE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 辅助函数：绘制带误差棒的条形图并标注改进百分比
    def plot_metric_bars_with_error(ax, values, stds, baseline_val, xlabel, unit=''):
        y_pos = range(len(model_names))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.85, xerr=stds, 
                       error_kw={'elinewidth': 1.5, 'capsize': 3, 'capthick': 1.5, 'ecolor': 'black'})
        for i, (val, std) in enumerate(zip(values, stds)):
            if i > 0 and baseline_val > 0:  # 跳过基线
                improvement = (baseline_val - val) / baseline_val * 100
                label = f'{improvement:+.1f}%'
                text_x = val + std + max(values)*0.02 if std > 0 else val + max(values)*0.02
                ax.text(text_x, i, label, 
                    ha='left', va='center', fontsize=10,
                    color='#2ca02c' if improvement > 0 else '#666666', fontweight='bold' if improvement > 20 else 'normal')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        max_val = max(v + s for v, s in zip(values, stds))
        ax.set_xlim(0, max_val * 1.25 if max_val > 0 else 1.0)
    
    ax1.set_title('(a) MAE', fontsize=13, fontweight='bold', loc='left')
    plot_metric_bars_with_error(ax1, mae_values, mae_stds, baseline_mae, 'MAE (scaled)')
    ax2.set_title('(b) RMSE', fontsize=13, fontweight='bold', loc='left')
    baseline_rmse = rmse_values[0]
    plot_metric_bars_with_error(ax2, rmse_values, rmse_stds, baseline_rmse, 'RMSE (scaled)')
    
    plt.tight_layout()
    
    # 添加运行次数标注
    if stats_results:
        runs = num_runs if num_runs is not None else NUM_RUNS
        fig.text(0.99, 0.01, f'Results averaged over {runs} runs', 
                 ha='right', va='bottom', fontsize=9, style='italic')
    
    # 保存综合图
    save_path = os.path.join(save_dir, f"{filename_prefix}.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path_png = os.path.join(save_dir, f"{filename_prefix}.png")
    plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 恢复默认设置
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"✅ 消融实验可视化已保存: {save_dir}")
    print(f"  - PDF: {filename_prefix}.pdf (含误差棒)")
    print(f"  - PNG: {filename_prefix}.png (含误差棒)")
    print(f"  (也可单独运行 visualize_results.py 重新生成图表，无需重跑训练)")

def plot_training_curves(training_histories, save_dir, filename_prefix="training_curves", title_suffix=""):
    """
    绘制训练曲线（结构消融主线）
    
    Args:
        training_histories: 字典，包含所有模型的训练历史
        save_dir: 保存目录
    """
    if not training_histories:
        print("⚠️  没有训练历史数据，跳过训练曲线绘制")
        return
    
    plot_models = [
        "Transformer (baseline)",
        "PLGAFormer backbone (no A/B/C)",
        "PLGAFormer A only",
        "PLGAFormer (C, proposed)",
        "PLGAFormer (A+C, excluded)",
        "PLGAFormer (A+B+C, excluded)",
    ]
    filtered = [(k, training_histories[k]) for k in plot_models if k in training_histories]
    if not filtered:
        filtered = list(training_histories.items())
    
    color_map = {
        "Transformer (baseline)": "#1f77b4",
        "PLGAFormer backbone (no A/B/C)": "#7f7f7f",
        "PLGAFormer A only": "#2ca02c",
        "PLGAFormer (C, proposed)": "#d62728",
        "PLGAFormer (A+C, excluded)": "#9467bd",
        "PLGAFormer (A+B+C, excluded)": "#ff7f0e",
    }
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('default')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.linewidth': 1.5})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (model_name, history) in enumerate(filtered):
        color = color_map.get(model_name, plt.cm.tab10(idx))
        epochs = range(1, len(history['train_losses']) + 1)
        ax1.plot(epochs, history['train_losses'], label=model_name, color=color, linewidth=2, alpha=0.85)
        ax2.plot(epochs, history['val_losses'], label=model_name, color=color, linewidth=2, alpha=0.85)
    
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Training Loss (MSE)', fontsize=13)
    title_1 = '(a) Training Loss' if not title_suffix else f"(a) Training Loss - {title_suffix}"
    title_2 = '(b) Validation Loss' if not title_suffix else f"(b) Validation Loss - {title_suffix}"
    ax1.set_title(title_1, fontsize=13, fontweight='bold', loc='left')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数尺度以便更好地观察
    
    ax2.set_xlabel('Epoch', fontsize=13)
    ax2.set_ylabel('Validation Loss (MSE)', fontsize=13)
    ax2.set_title(title_2, fontsize=13, fontweight='bold', loc='left')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图片
    save_path_pdf = os.path.join(save_dir, f"{filename_prefix}.pdf")
    save_path_png = os.path.join(save_dir, f"{filename_prefix}.png")
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✅ 训练曲线已保存:")
    print(f"  - PDF: {save_path_pdf}")
    print(f"  - PNG: {save_path_png}")

def plot_run_level_training_curves(training_histories, random_seeds, save_dir, filename_prefix="training_curves_runs", title_suffix=""):
    """
    绘制 run-level 验证曲线：同一模型下不同 seed 的曲线并排对比。

    Args:
        training_histories: dict[str, list[history]]，每个模型包含多次运行历史
        random_seeds: list[int]，运行使用的随机种子
        save_dir: 保存目录
    """
    if not training_histories:
        return

    ordered_models = [
        "Transformer (baseline)",
        "PLGAFormer backbone (no A/B/C)",
        "PLGAFormer A only",
        "PLGAFormer (C, proposed)",
        "PLGAFormer (A+C, excluded)",
        "PLGAFormer (A+B+C, excluded)",
    ]
    model_names = [m for m in ordered_models if m in training_histories and training_histories[m]]
    if not model_names:
        model_names = [m for m, runs in training_histories.items() if runs]
    if not model_names:
        return

    cols = 2
    rows = int(np.ceil(len(model_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.8 * rows), dpi=300)
    axes = np.atleast_1d(axes).flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(random_seeds))))

    for ax_idx, model_name in enumerate(model_names):
        ax = axes[ax_idx]
        runs = training_histories.get(model_name, [])
        for run_idx, history in enumerate(runs):
            val_losses = history.get('val_losses', [])
            if not val_losses:
                continue
            epochs = range(1, len(val_losses) + 1)
            seed = random_seeds[run_idx] if run_idx < len(random_seeds) else f"run{run_idx+1}"
            ax.plot(epochs, val_losses, linewidth=1.8, alpha=0.9, color=colors[run_idx % len(colors)], label=f"seed={seed}")

        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Loss')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.legend(fontsize=8, loc='upper right')

    for extra_idx in range(len(model_names), len(axes)):
        fig.delaxes(axes[extra_idx])

    suptitle = "Run-level Validation Curves" if not title_suffix else f"Run-level Validation Curves - {title_suffix}"
    fig.suptitle(suptitle, fontsize=13, y=1.0)
    plt.tight_layout()

    save_path_pdf = os.path.join(save_dir, f"{filename_prefix}.pdf")
    save_path_png = os.path.join(save_dir, f"{filename_prefix}.png")
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ run-level 训练曲线已保存:")
    print(f"  - PDF: {save_path_pdf}")
    print(f"  - PNG: {save_path_png}")

# 已移除无意义的可视化函数：generate_component_contribution_analysis 和 generate_metrics_radar_chart

def enable_full_plgaformer():
    """
    启用完整PLGAFormer模型的函数
    
    当单创新点测试成功后，调用此函数重新启用PLGAFormer (proposed)模型
    只需要调整ABLATION_PHASES中的结构消融模型配置即可
    """
    print("💡 启用完整PLGAFormer模型测试：")
    print("1. 找到ABLATION_PHASES中的结构消融配置")
    print("2. 调整要对比的结构组合（B/C开关）")
    print("3. 重新运行实验")
    print("4. 预期结果：B+C >= 单组件 >= Baseline")


if __name__ == '__main__':
    # 设置多进程保护，避免sympy兼容性问题
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过，忽略错误
    
    train_mode_name = os.getenv(
        'HGV_TRAIN_MODE',
        HGVConfig.TRAIN_MODE if hasattr(HGVConfig, 'TRAIN_MODE') else 'custom',
    )
    print(f"🚀 开始 PLGAFormer 消融实验 ({train_mode_name} 模式)...")
    print("📋 结构消融对照: 套A(MSE-only) + 套B(Physics-on)")
    print("=" * 80)
    try:
        main()
        print("✅ 消融实验完成!")
        print("📊 请检查结果：每个创新点是否都优于基线模型")
        print("📋 下一步：如果结果良好，可启用PLGAFormer (proposed)进行完整测试")
    except Exception as e:
        print(f"❌ 实验错误: {e}")
        import traceback
        traceback.print_exc()
