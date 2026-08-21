#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HGV Trajectory Prediction Models Package

This package provides:
- PLGAFormerTransformer: Main proposed model with bounded rotating-Earth prior fusion
- HGVPhysicsLoss: Physics-informed loss function
- create_registered_model: Factory for the active paper-mainline models
- create_baseline_model: Active baseline/analytical implementation factory
- create_sota_model/create_pit_model: Legacy import shims gated by the active registry
- HGVConfig: Unified configuration center for all models and experiments
"""

import os
from pathlib import Path

import torch
from .plgaformer import PLGAFormerTransformer, HGVPhysicsLoss, ECEFTrajectoryLoss
from .baseline_models import create_baseline_model
from .model_factory import create_registered_model, get_supported_model_types


_PHYSICAL_SCALER_KWARGS = {
    'input_scaler_mean',
    'input_scaler_scale',
    'output_scaler_mean',
    'output_scaler_scale',
    'sampling_interval_s',
    'require_physical_scaler',
}
_PUBLIC_SOURCE_KWARGS = {'source', 'source_root', 'tslib_root'}


def _unsupported_active_model(model_type):
    normalized = str(model_type).lower().strip()
    supported = get_supported_model_types()
    if normalized not in supported:
        raise ValueError(
            f"Unsupported active model type: {normalized}; supported={supported}"
        )
    return normalized


def _normalize_public_source_kwargs(kwargs):
    unknown = set(kwargs) - _PUBLIC_SOURCE_KWARGS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"Unsupported create_sota_model keyword(s): {names}")

    normalized = dict(kwargs)
    source_root = normalized.pop('source_root', None)
    tslib_root = normalized.get('tslib_root')
    if source_root is not None and tslib_root is not None:
        source_path = Path(source_root).expanduser().resolve()
        tslib_path = Path(tslib_root).expanduser().resolve()
        if source_path != tslib_path:
            raise ValueError("source_root conflicts with tslib_root")
    if source_root is not None:
        normalized['tslib_root'] = source_root

    if normalized.get('source') is not None:
        source = str(normalized['source']).lower().strip()
        if source != 'tslib':
            raise ValueError("create_sota_model source must be 'tslib'")
        normalized['source'] = source
    elif normalized.get('tslib_root') is not None:
        normalized['source'] = 'tslib'
    return normalized or None


def create_sota_model(
    model_type,
    input_dim=6,
    device=torch.device('cpu'),
    **kwargs,
):
    """Legacy import shim gated by the exact active registry and constructors."""
    normalized = _unsupported_active_model(model_type)
    if normalized in {'transformer', 'dlinear', 'patchtst', 'itransformer'}:
        active_kwargs = _normalize_public_source_kwargs(kwargs)
    elif normalized in {'plgaformer', 'kinematic', 'rotating_3dof'}:
        unknown = set(kwargs) - _PHYSICAL_SCALER_KWARGS
        if unknown:
            names = ", ".join(sorted(unknown))
            if normalized == 'plgaformer':
                raise TypeError(
                    "PLGAFormer compatibility shim does not accept architecture "
                    f"or active-flag overrides: {names}"
                )
            raise TypeError(f"Unsupported create_sota_model keyword(s): {names}")
        active_kwargs = dict(kwargs) or None
    else:  # pragma: no cover - guarded by the exact active registry above
        active_kwargs = None
    return create_registered_model(
        model_type=normalized,
        input_dim=input_dim,
        device=device,
        plgaformer_kwargs=active_kwargs,
    )


def create_pit_model(
    input_dim=6,
    device=torch.device('cpu'),
    **kwargs,
):
    """Legacy import shim; PIT is not an active constructible model."""
    return create_registered_model(
        model_type='pit',
        input_dim=input_dim,
        device=device,
    )


class HGVConfig:
    """HGV项目统一配置中心
    
    统一管理所有模型架构参数、物理损失参数和训练参数，
    解决项目中参数分散和不一致的问题。
    """
    
    # ===== 核心模型架构参数 =====
    # 注：只保留主要参数名，避免冗余
    MODEL_PARAMS = {
        'input_dim': 6,              # 输入维度 [r, λ, φ, V, γ, ψ] 球坐标系
        'd_model': 256,              # 模型维度（统一轻量级配置）
        'nhead': 8,                   # 注意力头数（与 ATTENTION_PARAMS 一致，顶刊常用 8）
        'output_dim': 3,             # 标准输出维度 [r, λ, φ]，与数据/评估一致
        'num_encoder_layers': 3,     # 编码器层数
        'num_decoder_layers': 2,     # 解码器层数  
        'dim_feedforward': 1024,     # 前馈网络维度（4*d_model）
        'dropout': 0.1,              # Dropout率
        'max_seq_length': 512,       # 最大序列长度
    }
    
    # ===== 注意力头配置 =====
    # 顶刊常见为 8 头；若为效率采用 4 头，可改为 4 并在实验说明中注明“为效率采用 4 头”（配置出处：本文件 ATTENTION_PARAMS）
    ATTENTION_PARAMS = {
        'plgaformer_nhead': 8,   # PLGAFormer 注意力头数（与顶刊一致）
        'baseline_nhead': 8,     # 基线 Transformer 注意力头数
    }
    
    # ===== 物理损失参数 =====
    PHYSICS_PARAMS = {
        'alpha': 0.0001,              # 统一无量纲物理残差的总权重
        'acceleration_weight': 0.1,   # 物理残差内部的加速度相对权重
    }
    
    # ===== HGV物理约束参数 =====
    # 这些参数应与数据生成器中的初始条件保持一致
    PHYSICAL_CONSTRAINTS = {
        # 高度约束 (m)
        'min_height': 0.0,           # 最小高度（地表）
        'max_height': 120000.0,      # 最大高度（120km）
        
        # 速度约束 (m/s)
        'min_velocity': 100.0,       # 最小速度
        'max_velocity': 15000.0,     # 最大速度
        
        # 角度约束 (rad)
        'max_gamma': 1.5708,         # 最大航迹角 (π/2)
        'max_latitude': 1.5708,      # 最大纬度 (π/2)
        'max_azimuth_rate': 3.1416,  # 最大方位角变化率 (π)
        
        # 物理常数
        'g': 9.81,                   # 重力加速度 (m/s²)
        'earth_radius': 6378000.0,   # 地球半径 (m) - 与数据生成器一致
    }
    
    # ===== 训练参数 =====
    # 注：learning_rate 是主要参数名，需要 lr 别名时通过方法动态获取
    #
    # ---------- 主实验超参表（正文/附录可引用，审稿复现用）----------
    # 所有对比方法（含基线）使用相同或对应设置，未做大规模超参搜索。
    # 优化器: AdamW, lr=0.001, weight_decay=5e-5, betas=(0.9,0.999)
    # 学习率调度: Linear Warmup (warmup_epochs) + Cosine Annealing
    # 训练: epochs=50, batch_size=64(或 speed 预设 128), gradient_clip_norm=1.0
    # 早停: early_stopping_patience=15, min_delta=1e-6
    # 模型: d_model=256, nhead=8, encoder_layers=3, decoder_layers=2, dim_feedforward=1024, dropout=0.1
    # Primary loss: ECEF composite objective; optional dynamics residual uses PHYSICS_PARAMS.
    # 多轮运行: num_runs/random_seeds 由 TRAIN_MODE(speed/repro) 决定
    # ---------------------------------------------------------------
    TRAIN_MODE = 'repro'

    TRAIN_MODE_PRESETS = {
        'speed': {
            'batch_size': 128,
            'deterministic': False,
            'cudnn_benchmark': True,
            'allow_tf32': True,
            'matmul_precision': 'high',
            'dataloader_workers': 12,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
            'num_runs': 2,
            'random_seeds': [42, 123],
        },
        'max_perf': {
            'batch_size': 256,
            'deterministic': False,
            'cudnn_benchmark': True,
            'allow_tf32': True,
            'matmul_precision': 'high',
            'dataloader_workers': 24,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
            'num_runs': 2,
            'random_seeds': [42, 123],
        },
        'repro': {
            'batch_size': 64,
            'deterministic': True,
            'cudnn_benchmark': False,
            'allow_tf32': False,
            'matmul_precision': 'high',
            'dataloader_workers': 0,
            'persistent_workers': False,
            'num_runs': 5,
            'random_seeds': [42, 123, 456, 789, 1024],
        },
    }

    TRAIN_PARAMS = {
        'batch_size': 64,              # 批次大小（统一配置）
        'learning_rate': 0.001,        # 学习率（顶刊标准：0.001）
        'epochs': 50,                  # 训练轮数（统一配置）
        'seq_len': 256,
        'pred_len': 256,
        'label_len': 128,
        'train_supervision_protocol': 'source_context_pred_window',
        'eval_protocol': 'source_context_decoder',
        'eval_ar_seed_mode': 'zero',
        'trajectory_loss': 'ecef_composite',
        'ecef_distance_scale_m': 100000.0,
        'scaled_mse_weight': 0.1,
        'prediction_horizons': [32, 64, 128, 256],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataloader_workers': 0,       # RAM TensorDataset; avoids Windows spawn overhead
        'pin_memory': False,           # workers=0 measured fastest for RAM tensors
        'persistent_workers': False,
        'prefetch_factor': 2,
        'deterministic': False,
        'cudnn_benchmark': True,
        'allow_tf32': True,
        'matmul_precision': 'high',
        'num_runs': 3,
        'random_seeds': [42, 123, 456],
        'data_subset_ratio': None,      # None=全量数据；消融时由 get_experiment_config('ablation') 覆盖
        'verbose': False,
        'lr_decay_factor': 0.8,        # 学习率衰减因子
        'weight_decay': 5e-5,         # 权重衰减（与 SOTA/消融一致，顶刊常用量级）
        'gradient_clip_norm': 1.0,     # 梯度裁剪范数
        'use_mixed_precision': True,   # 启用混合精度训练
        'mixed_precision_dtype': 'bfloat16',
        'cache_physics_prior': True,
        'physics_prior_cache_batch_size': 512,
        'early_stopping_patience': 15,  # 早停耐心参数
        'patience': 10,
        'warmup_epochs': 5,   # 顶刊常用 5–10；0 表示无 warmup
        'min_delta': 1e-6,    # 早停最小改善阈值（避免微小波动重置patience）
    }
    
    # ===== 特定模型参数 =====
    SPECIFIC_PARAMS = {
        'patchtst': {'patch_len': 16, 'stride': 8},
    }
    
    @classmethod
    def get_config(cls):
        """获取统一配置 - 主要接口方法
        
        Returns:
            dict: 包含所有配置的字典，结构化为不同类别
        """
        return {
            'model': {
                'input_dim': cls.MODEL_PARAMS['input_dim'],
                'd_model': cls.MODEL_PARAMS['d_model'],
                'nhead': cls.ATTENTION_PARAMS['plgaformer_nhead'],  # 默认使用PLGAFormer配置
                'num_encoder_layers': cls.MODEL_PARAMS['num_encoder_layers'],
                'num_decoder_layers': cls.MODEL_PARAMS['num_decoder_layers'],
                'dim_feedforward': cls.MODEL_PARAMS['dim_feedforward'],
                'dropout': cls.MODEL_PARAMS['dropout'],
            },
            'training': {
                'batch_size': cls.TRAIN_PARAMS['batch_size'],
                'learning_rate': cls.TRAIN_PARAMS['learning_rate'],
                'epochs': cls.TRAIN_PARAMS['epochs'],
                'seq_len': cls.TRAIN_PARAMS['seq_len'],
                'pred_len': cls.TRAIN_PARAMS['pred_len'],
            },
            'physics': cls.PHYSICS_PARAMS.copy()
        }
    
    @classmethod
    def get_model_config(cls, model_type='plgaformer'):
        """获取指定模型的完整配置
        
        Args:
            model_type: 模型类型 ('plgaformer', 'transformer', 'patchtst', 等)
            
        Returns:
            dict: 包含所有必要参数的配置字典
        """
        config = cls.MODEL_PARAMS.copy()
        
        # 根据模型类型设置注意力头数
        if model_type == 'plgaformer':
            config['nhead'] = cls.ATTENTION_PARAMS['plgaformer_nhead']
        else:
            config['nhead'] = cls.ATTENTION_PARAMS['baseline_nhead']
            
        # 添加特定模型参数
        if model_type in cls.SPECIFIC_PARAMS:
            config.update(cls.SPECIFIC_PARAMS[model_type])
            
        return config
    
    @classmethod  
    def get_physics_config(cls):
        """获取物理损失配置"""
        return cls.PHYSICS_PARAMS.copy()
    
    @classmethod
    def get_physical_constraints(cls):
        """获取HGV物理约束参数"""
        return cls.PHYSICAL_CONSTRAINTS.copy()
        
    @classmethod
    def get_train_config(cls, mode=None):
        """获取训练配置 - 包含兼容性别名"""
        config = cls.TRAIN_PARAMS.copy()
        selected_mode = os.getenv("HGV_TRAIN_MODE", mode or cls.TRAIN_MODE)
        preset = cls.TRAIN_MODE_PRESETS.get(selected_mode, {})
        config.update(preset)
        # 环境变量覆盖：便于在不改代码的情况下压榨本地硬件
        env_int_overrides = {
            "HGV_BATCH_SIZE": "batch_size",
            "HGV_DATALOADER_WORKERS": "dataloader_workers",
            "HGV_PREFETCH_FACTOR": "prefetch_factor",
            "HGV_NUM_RUNS": "num_runs",
            "HGV_SEQ_LEN": "seq_len",
            "HGV_PRED_LEN": "pred_len",
            "HGV_LABEL_LEN": "label_len",
            "HGV_EPOCHS": "epochs",
            "HGV_EARLY_STOPPING_PATIENCE": "early_stopping_patience",
            "HGV_PATIENCE": "patience",
            "HGV_WARMUP_EPOCHS": "warmup_epochs",
        }
        for env_key, conf_key in env_int_overrides.items():
            val = os.getenv(env_key)
            if val is None or val == "":
                continue
            try:
                config[conf_key] = int(val)
            except ValueError:
                pass

        env_float_overrides = {
            "HGV_LEARNING_RATE": "learning_rate",
            "HGV_WEIGHT_DECAY": "weight_decay",
            "HGV_MIN_DELTA": "min_delta",
        }
        for env_key, conf_key in env_float_overrides.items():
            val = os.getenv(env_key)
            if val is None or val == "":
                continue
            try:
                config[conf_key] = float(val)
            except ValueError:
                pass

        horizons = os.getenv("HGV_PREDICTION_HORIZONS")
        if horizons:
            try:
                config["prediction_horizons"] = [
                    int(item.strip()) for item in horizons.split(",") if item.strip()
                ]
            except ValueError:
                pass

        seeds = os.getenv("HGV_RANDOM_SEEDS")
        if seeds:
            try:
                config["random_seeds"] = [
                    int(item.strip()) for item in seeds.split(",") if item.strip()
                ]
                config["num_runs"] = len(config["random_seeds"])
            except ValueError:
                pass

        env_bool_overrides = {
            "HGV_PIN_MEMORY": "pin_memory",
            "HGV_PERSISTENT_WORKERS": "persistent_workers",
            "HGV_ALLOW_TF32": "allow_tf32",
            "HGV_CUDNN_BENCHMARK": "cudnn_benchmark",
            "HGV_DETERMINISTIC": "deterministic",
        }
        for env_key, conf_key in env_bool_overrides.items():
            val = os.getenv(env_key)
            if val is None or val == "":
                continue
            config[conf_key] = str(val).strip().lower() in {"1", "true", "yes", "on"}

        if config.get("dataloader_workers", 0) > 0:
            max_workers = max(1, (os.cpu_count() or 8) - 2)
            config["dataloader_workers"] = int(min(config["dataloader_workers"], max_workers))
        # 添加 lr 别名以兼容旧代码
        config['lr'] = config['learning_rate']
        return config
        
    @classmethod
    def get_experiment_config(cls, experiment_type='default'):
        """获取实验专用配置
        
        Args:
            experiment_type: 实验类型 ('ablation', 'baseline', 'longterm', 'efficiency')
            
        Returns:
            dict: 合并的配置字典
        """
        config = {}
        config.update(cls.MODEL_PARAMS)
        config.update(cls.PHYSICS_PARAMS)
        config.update(cls.TRAIN_PARAMS)
        
        # 实验特定调整
        if experiment_type == 'efficiency':
            config['dropout'] = 0.03  # 效率实验使用更低dropout
        elif experiment_type == 'longterm':
            config['alpha'] = 0.001   # 长期预测保持标准物理损失
        elif experiment_type == 'ablation':
            selected_mode = os.getenv("HGV_TRAIN_MODE", cls.TRAIN_MODE)
            if selected_mode == 'speed':
                config['data_subset_ratio'] = 0.5
                config['num_runs'] = 2
                config['random_seeds'] = [42, 123]
            else:
                config['data_subset_ratio'] = None
                config['num_runs'] = 3
                config['random_seeds'] = [42, 123, 456]

        return config
    
    @classmethod
    def from_trial(cls, trial):
        """从Optuna trial创建动态配置 - AutoML支持
        
        Args:
            trial: Optuna trial对象
            
        Returns:
            dict: 包含所有配置的字典，参数由AutoML动态选择
        """
        # 创建基础配置副本
        config = {}
        config.update(cls.MODEL_PARAMS)
        config.update(cls.PHYSICS_PARAMS)
        config.update(cls.TRAIN_PARAMS)
        
        # AutoML搜索的关键参数
        config['d_model'] = trial.suggest_categorical('d_model', [64, 128, 256])
        config['nhead'] = trial.suggest_categorical('nhead', [4, 8, 16])
        config['num_encoder_layers'] = trial.suggest_int('num_encoder_layers', 2, 6)
        config['num_decoder_layers'] = trial.suggest_int('num_decoder_layers', 1, 4)
        config['dropout'] = trial.suggest_float('dropout', 0.05, 0.3)
        
        # 训练参数优化 - 以顶刊标准0.001为中心进行搜索
        config['learning_rate'] = trial.suggest_float('learning_rate', 5e-4, 2e-3, log=True)
        config['lr'] = config['learning_rate']  # 兼容性别名
        config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
        config['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # 物理损失权重优化
        config['alpha'] = trial.suggest_float('alpha', 1e-5, 1e-3, log=True)
        config['acceleration_weight'] = trial.suggest_float(
            'acceleration_weight', 0.05, 0.5, log=True
        )
        
        # 确保dim_feedforward与d_model成比例
        config['dim_feedforward'] = config['d_model'] * 4
        
        return config


__all__ = [
    'PLGAFormerTransformer',
    'HGVPhysicsLoss',
    'ECEFTrajectoryLoss',
    'create_baseline_model',
    'create_sota_model',
    'create_pit_model',
    'create_registered_model',
    'get_supported_model_types',
    'HGVConfig',
]
