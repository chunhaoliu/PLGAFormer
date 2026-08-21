#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLGAFormer Physics Consistency Analysis (Experiment 4)
- Validates PLGAFormer's ability to maintain physical consistency in predictions
- Evaluates constraint violation rates and trajectory smoothness
- Goal: Demonstrate PLGAFormer's superior physics-informed prediction capabilities
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

# 项目根目录：与 exp1/exp2/exp3 保持一致，所有路径基于此
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import (
    PLGAFormerTransformer,
    HGVPhysicsLoss,
    create_baseline_model,
    create_sota_model,
    create_pit_model,
    create_registered_model,
    HGVConfig,
)
from utils.repro import set_global_seed
from utils.experiment_io import (
    get_experiment_dirs,
    resolve_exp1_checkpoint,
    save_experiment_results,
    save_run_metadata,
)
from utils.inference_protocol import predict_by_eval_protocol as unified_predict_by_eval_protocol
from utils.seq2seq_protocol import align_source_position_scale
from utils.model_provenance import (
    build_comparison_model_provenance,
    validate_comparison_model_types,
)
from utils.trajectory_protocol import dataset_protocol_name, validate_npz_trajectory_splits
from data_generation.data_paths import get_processed_data_dir
from data_provider.hgv_data import load_hgv_dataset

REPORT_ON_PHYSICAL_SCALE = True

# 获取统一配置
TRAIN_CONFIG = HGVConfig.get_train_config()
MODEL_CONFIG = HGVConfig.get_model_config('plgaformer')
PHYSICS_CONFIG = HGVConfig.get_physics_config()
EVAL_PROTOCOL = TRAIN_CONFIG.get('eval_protocol', 'strict_autoregressive')
EVAL_AR_SEED_MODE = TRAIN_CONFIG.get('eval_ar_seed_mode', 'zero')
TRAIN_SUPERVISION_PROTOCOL = TRAIN_CONFIG.get('train_supervision_protocol', 'shifted_next_step')
OPTIMIZER_PROFILE = TRAIN_CONFIG.get('optimizer_profile', 'enhanced')
STRICT_REPRO_MODE = TRAIN_CONFIG.get('strict_repro_mode', False)
# 以统一配置覆盖初始化默认seed策略。
set_global_seed(42, deterministic=True, strict_deterministic=STRICT_REPRO_MODE)

print(f"\n{'='*60}")
print(f"PHYSICS CONSISTENCY ANALYSIS CONFIGURATION")
print(f"Batch Size: {TRAIN_CONFIG['batch_size']}")
print(f"Learning Rate: {TRAIN_CONFIG['learning_rate']}")
print(f"Model Dimension: {MODEL_CONFIG['d_model']}")
print(f"{'='*60}\n")

# 物理一致性评估配置 - 从统一配置读取
_physical_constraints = HGVConfig.get_physical_constraints()
PREDICTION_LENGTH = TRAIN_CONFIG.get('pred_len', 256)  # 从配置读取预测长度
EARTH_RADIUS = _physical_constraints['earth_radius']   # 从配置读取地球半径
MIN_HEIGHT = _physical_constraints['min_height']       # 最小高度约束 (m)
MAX_HEIGHT = _physical_constraints['max_height']       # 最大高度约束 (m)
VELOCITY_RELATIVE_TOLERANCE = 0.50
ACCELERATION_RELATIVE_TOLERANCE = 0.25
SAMPLING_INTERVAL_S = 1.0

# 统一的数据 / 模型 / 结果目录（与其它实验保持一致）
DATA_DIR = get_processed_data_dir(PROJECT_ROOT)
RESULTS_DIR, _ = get_experiment_dirs(PROJECT_ROOT, 'exp4_physics_consistency')
_, MODELS_DIR = get_experiment_dirs(PROJECT_ROOT, 'exp1_sota')
COPY_FIGURES_TO_LATEX = False

# 对比模型配置：基线 Transformer + 提出方法 PLGAFormer
COMPARISON_MODELS = OrderedDict([
    ("Transformer (baseline)", {
        'model_type': 'transformer',
        'description': 'Standard Transformer baseline (data-driven only)',
    }),
    ("PLGAFormer (proposed)", {
        'model_type': 'plgaformer',
        'description': 'Validation-selected PLGAFormer with bounded rotating-Earth prior fusion (C)',
    }),
])

# ======================================================================================
# 数据加载和准备
# ======================================================================================

def load_and_prepare_data(batch_size):
    """加载数据集"""
    data_path = DATA_DIR / 'hgv_trajectory_dataset.npz'
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件未找到: {data_path}")
    
    print(f"📂 加载数据集: {data_path}")
    data_bundle = load_hgv_dataset(data_path, require_trajectory_level=True)
    data = data_bundle.raw
    global SAMPLING_INTERVAL_S
    SAMPLING_INTERVAL_S = float(np.asarray(data.get('sampling_interval_s', 1.0)).reshape(-1)[0])
    
    print(f"数据集规模: 训练集{data['X_train'].shape}, 验证集{data['X_val'].shape}, 测试集{data['X_test'].shape}")
    protocol = data_bundle.protocol
    split_report = data_bundle.split_report
    if split_report is not None:
        print(f"数据协议: {protocol} | 轨迹级划分重叠检查: {split_report}")
    else:
        print(f"数据协议: {protocol}")
    
    # Evaluation is read-only: never refit or overwrite training scalers here.
    scaler_path = DATA_DIR / 'scaler_hgv_trajectory.joblib'
    output_scaler_path = DATA_DIR / 'output_scaler_hgv_trajectory.joblib'
    
    scaler = joblib.load(str(scaler_path))
    output_scaler = joblib.load(str(output_scaler_path))
    
    # 应用标准化
    n_test = data['X_test'].shape[0]
    in_dim = data['X_test'].shape[-1]
    
    X_test_scaled = scaler.transform(data['X_test'].reshape(-1, in_dim)).reshape(data['X_test'].shape)
    
    n_test_y = data['y_test'].shape[0]
    out_dim = data['y_test'].shape[-1]
    
    if output_scaler.mean_.shape[0] != out_dim:
        raise ValueError(
            "Stored output scaler is incompatible with the test targets: "
            f"scaler_dim={output_scaler.mean_.shape[0]}, target_dim={out_dim}."
        )
    
    y_test_scaled = output_scaler.transform(data['y_test'].reshape(-1, out_dim)).reshape(data['y_test'].shape)
    X_test_scaled = align_source_position_scale(
        X_test_scaled,
        input_mean=scaler.mean_,
        input_scale=scaler.scale_,
        output_mean=output_scaler.mean_,
        output_scale=output_scaler.scale_,
        output_dim=out_dim,
    ).astype(np.float32, copy=False)
    
    # 转换为torch tensor
    X_test = torch.from_numpy(X_test_scaled.astype(np.float32))
    y_test = torch.from_numpy(y_test_scaled.astype(np.float32))
    
    # 创建DataLoader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    return test_loader, scaler, output_scaler

# ======================================================================================
# 物理一致性评估函数
# ======================================================================================

def generate_causal_mask(size, device):
    """生成因果掩码"""
    mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def unscale_data(scaled_data, mean, scale):
    return scaled_data * scale + mean


def spherical_to_cartesian_torch(position):
    """Convert [r, lon, lat] trajectories to Cartesian meters."""
    radius = position[..., 0]
    lon = position[..., 1]
    lat = position[..., 2]
    cos_lat = torch.cos(lat)
    return torch.stack(
        [
            radius * cos_lat * torch.cos(lon),
            radius * cos_lat * torch.sin(lon),
            radius * torch.sin(lat),
        ],
        dim=-1,
    )


def compute_constraint_violations(pred_traj, true_traj, sampling_interval_s=None):
    """
    计算物理约束违反率。

    输入在物理尺度下为球坐标 [r, lon, lat]。高度约束直接在半径上检查；
    速度/加速度约束先转到 Cartesian 米制空间，再用相对 RMSE 阈值统计轨迹级违反率。
    """
    violations = {}
    dt = float(SAMPLING_INTERVAL_S if sampling_interval_s is None else sampling_interval_s)
    if dt <= 0.0:
        raise ValueError("sampling_interval_s must be positive.")
    
    # 1. 高度约束违反（使用物理约束区间，而不是统计阈值）
    # 假设第 0 维是径向距离 r（物理空间）或与 r 单调相关的高度量
    if pred_traj.shape[-1] >= 1:
        radius_or_height = pred_traj[:, :, 0]
        
        if REPORT_ON_PHYSICAL_SCALE:
            # 在物理空间下，pred_traj[..., 0] 为地心距离 r
            height = radius_or_height - EARTH_RADIUS
        else:
            # 在标准化空间下无法直接换算高度，这里保留原值，仅做相对约束检查
            height = radius_or_height
        
        # 违反率：任一时间步高度超出 [MIN_HEIGHT, MAX_HEIGHT] 的轨迹比例
        height_violation_mask = (height < (MIN_HEIGHT - 1e-3)) | (height > (MAX_HEIGHT + 1e-3))
        height_violation_rate = height_violation_mask.any(dim=1).float().mean().item()
        violations['height_violation_rate'] = height_violation_rate
    
    pred_xyz = spherical_to_cartesian_torch(pred_traj)
    true_xyz = spherical_to_cartesian_torch(true_traj)

    if pred_xyz.size(1) >= 2:
        pred_velocity = torch.diff(pred_xyz, dim=1) / dt
        true_velocity = torch.diff(true_xyz, dim=1) / dt
        velocity_error = torch.linalg.norm(pred_velocity - true_velocity, dim=-1)
        velocity_scale = torch.linalg.norm(true_velocity, dim=-1)
        velocity_relative_rmse = torch.sqrt(torch.mean(velocity_error ** 2, dim=1)) / (
            torch.sqrt(torch.mean(velocity_scale ** 2, dim=1)) + 1e-8
        )
        violations['velocity_relative_rmse'] = velocity_relative_rmse.mean().item()
        violations['velocity_violation_rate'] = (
            velocity_relative_rmse > VELOCITY_RELATIVE_TOLERANCE
        ).float().mean().item()
    else:
        violations['velocity_relative_rmse'] = 0.0
        violations['velocity_violation_rate'] = 0.0

    if pred_xyz.size(1) >= 3:
        pred_acceleration = torch.diff(pred_xyz, n=2, dim=1) / (dt ** 2)
        true_acceleration = torch.diff(true_xyz, n=2, dim=1) / (dt ** 2)
        accel_error = torch.linalg.norm(pred_acceleration - true_acceleration, dim=-1)
        accel_scale = torch.linalg.norm(true_acceleration, dim=-1)
        accel_relative_rmse = torch.sqrt(torch.mean(accel_error ** 2, dim=1)) / (
            torch.sqrt(torch.mean(accel_scale ** 2, dim=1)) + 1e-8
        )
        violations['acceleration_relative_rmse'] = accel_relative_rmse.mean().item()
        violations['acceleration_violation_rate'] = (
            accel_relative_rmse > ACCELERATION_RELATIVE_TOLERANCE
        ).float().mean().item()
    else:
        violations['acceleration_relative_rmse'] = 0.0
        violations['acceleration_violation_rate'] = 0.0
    
    return violations

def compute_trajectory_smoothness(pred_traj, true_traj, sampling_interval_s=None):
    """
    计算 Cartesian 物理空间下的轨迹平滑度。
    """
    smoothness = {}
    dt = float(SAMPLING_INTERVAL_S if sampling_interval_s is None else sampling_interval_s)
    if dt <= 0.0:
        raise ValueError("sampling_interval_s must be positive.")

    pred_xyz = spherical_to_cartesian_torch(pred_traj)
    true_xyz = spherical_to_cartesian_torch(true_traj)

    if pred_xyz.size(1) >= 3:
        pred_second_diff = torch.diff(pred_xyz, n=2, dim=1) / (dt ** 2)
        true_second_diff = torch.diff(true_xyz, n=2, dim=1) / (dt ** 2)
        pred_smoothness = torch.linalg.norm(pred_second_diff, dim=-1).mean().item()
        true_smoothness = torch.linalg.norm(true_second_diff, dim=-1).mean().item()
    else:
        pred_smoothness = 0.0
        true_smoothness = 0.0
    
    smoothness['position_smoothness_pred'] = pred_smoothness
    smoothness['position_smoothness_true'] = true_smoothness
    smoothness['position_smoothness_ratio'] = pred_smoothness / (true_smoothness + 1e-8)
    
    if pred_xyz.size(1) >= 3:
        pred_speed = torch.linalg.norm(torch.diff(pred_xyz, dim=1) / dt, dim=-1)
        true_speed = torch.linalg.norm(torch.diff(true_xyz, dim=1) / dt, dim=-1)
        pred_velocity_smoothness = (torch.abs(torch.diff(pred_speed, dim=1)) / dt).mean().item()
        true_velocity_smoothness = (torch.abs(torch.diff(true_speed, dim=1)) / dt).mean().item()
    else:
        pred_velocity_smoothness = 0.0
        true_velocity_smoothness = 0.0
    
    smoothness['velocity_smoothness_pred'] = pred_velocity_smoothness
    smoothness['velocity_smoothness_true'] = true_velocity_smoothness
    smoothness['velocity_smoothness_ratio'] = pred_velocity_smoothness / (true_velocity_smoothness + 1e-8)
    
    return smoothness


def format_physics_metric(metric_name, value):
    """Format rates as percentages and continuous metrics as scalars."""
    if str(metric_name).endswith("_rate"):
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def evaluate_physics_consistency(model, test_loader, device, scaler_mean, scaler_scale):
    """评估模型的物理一致性
    
    - 约束/平滑度指标：在物理空间上计算（反标准化后）
    - Accuracy metrics: scaled MSE and ECEF errors in meters
    """
    model.eval()
    
    all_violations = []
    all_smoothness = []
    batch_weights = []
    total_mse_scaled = 0.0
    total_mse_physical = 0.0
    total_mae_cart_m = 0.0
    total_ade_m = 0.0
    total_fde_m = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x, y_true in test_loader:
            x, y_true = x.to(device), y_true.to(device)
            batch_size = x.size(0)
            
            # 限制预测长度
            if y_true.size(1) > PREDICTION_LENGTH:
                y_true = y_true[:, :PREDICTION_LENGTH, :]
            
            pred_length = y_true.size(1)
            y_pred = unified_predict_by_eval_protocol(
                model=model,
                x=x,
                y_true_scaled=y_true,
                pred_length=pred_length,
                device=device,
                eval_protocol=EVAL_PROTOCOL,
                eval_ar_seed_mode=EVAL_AR_SEED_MODE,
                label_len=int(TRAIN_CONFIG.get('label_len', max(1, min(48, y_true.size(1))))),
                causal_mask_builder=generate_causal_mask,
            )
            if not torch.isfinite(y_pred).all():
                raise FloatingPointError("Non-finite prediction in physics-consistency evaluation.")
            
            # 1）标准化空间 MSE（与主实验保持一致）
            y_pred_scaled = y_pred
            y_true_scaled = y_true
            mse_scaled = ((y_pred_scaled - y_true_scaled) ** 2).mean().item()
            total_mse_scaled += mse_scaled * batch_size
            
            # 2）物理空间：用于物理约束和可选的物理 MSE
            if REPORT_ON_PHYSICAL_SCALE:
                y_pred_eval = unscale_data(y_pred, scaler_mean, scaler_scale)
                y_true_eval = unscale_data(y_true, scaler_mean, scaler_scale)
            else:
                y_pred_eval = y_pred
                y_true_eval = y_true
            
            # 计算约束违反率（物理空间）
            violations = compute_constraint_violations(y_pred_eval, y_true_eval)
            all_violations.append(violations)
            
            # 计算轨迹平滑度（物理空间）
            smoothness = compute_trajectory_smoothness(y_pred_eval, y_true_eval)
            all_smoothness.append(smoothness)
            batch_weights.append(batch_size)
            
            # ECEF component errors preserve a single physical unit (meters).
            pred_xyz = spherical_to_cartesian_torch(y_pred_eval)
            true_xyz = spherical_to_cartesian_torch(y_true_eval)
            cart_error = pred_xyz - true_xyz
            displacement = torch.linalg.norm(cart_error, dim=-1)
            mse_physical = torch.mean(cart_error ** 2).item()
            mae_cart_m = torch.mean(torch.abs(cart_error)).item()
            ade_m = torch.mean(displacement).item()
            fde_m = torch.mean(displacement[:, -1]).item()
            total_mse_physical += mse_physical * batch_size
            total_mae_cart_m += mae_cart_m * batch_size
            total_ade_m += ade_m * batch_size
            total_fde_m += fde_m * batch_size
            
            total_samples += batch_size
    
    # 聚合结果
    avg_violations = {}
    for key in all_violations[0].keys():
        avg_violations[key] = float(np.average([v[key] for v in all_violations], weights=batch_weights))
    
    avg_smoothness = {}
    for key in all_smoothness[0].keys():
        avg_smoothness[key] = float(np.average([s[key] for s in all_smoothness], weights=batch_weights))
    
    avg_mse_scaled = total_mse_scaled / total_samples if total_samples > 0 else float('inf')
    avg_mse_physical = total_mse_physical / total_samples if total_samples > 0 else float('inf')
    avg_mae_cart_m = total_mae_cart_m / total_samples if total_samples > 0 else float('inf')
    avg_ade_m = total_ade_m / total_samples if total_samples > 0 else float('inf')
    avg_fde_m = total_fde_m / total_samples if total_samples > 0 else float('inf')
    
    return {
        'violations': avg_violations,
        'smoothness': avg_smoothness,
        'mse_scaled': avg_mse_scaled,
        'mse_physical': avg_mse_physical,
        'mse_cart_m2': avg_mse_physical,
        'rmse_cart_m': float(np.sqrt(avg_mse_physical)),
        'mae_cart_m': avg_mae_cart_m,
        'ade_m': avg_ade_m,
        'fde_m': avg_fde_m,
    }

# ======================================================================================
# 可视化函数
# ======================================================================================

def generate_physics_consistency_visualization(results, save_dir):
    """生成物理一致性可视化 - 顶刊标准：每个指标独立成图（不再绘制雷达图）"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = list(results.keys())
    
    # 定义模型颜色和标记（与exp2保持一致）
    model_colors = {
        'Transformer (baseline)': '#1f77b4',
        'Informer': '#ff7f0e',
        'PatchTST': '#2ca02c',
        'FEDformer': '#d62728',
        'TimesNet': '#9467bd',
        'iTransformer': '#17becf',
        'PLGAFormer (proposed)': '#8c564b'
    }
    
    model_markers = {
        'Transformer (baseline)': 'o',
        'Informer': 's',
        'PatchTST': '^',
        'FEDformer': 'D',
        'TimesNet': 'v',
        'iTransformer': '<',
        'PLGAFormer (proposed)': 'p'
    }
    
    # 设置matplotlib为顶刊标准
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
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
        'legend.fontsize': 11,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 1.0
    })
    
    # 1. 生成约束违反率图（折线图，三种约束类型）
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    
    violation_types = ['height_violation_rate', 'velocity_violation_rate', 'acceleration_violation_rate']
    violation_labels = ['Height Constraint', 'Velocity Constraint', 'Acceleration Constraint']
    x_pos = np.arange(len(violation_types))
    
    for model_name in model_names:
        if 'violations' in results[model_name]:
            violation_values = [results[model_name]['violations'][vt] * 100 for vt in violation_types]
            ax.plot(x_pos, violation_values,
                   marker=model_markers.get(model_name, 'o'),
                   color=model_colors.get(model_name, '#7f7f7f'),
                   label=model_name.replace('(baseline)', '').replace('(proposed)', '').strip(),
                   linewidth=2.5, markersize=8)
    
    ax.set_xlabel('Constraint Type', fontsize=13)
    ax.set_ylabel('Violation Rate (%)', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(violation_labels, fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.2, len(violation_types)-0.8)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "constraint_violation_rates.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path_png = os.path.join(save_dir, "constraint_violation_rates.png")
    plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. 生成轨迹平滑度误差图（柱状图，更专业的分组形式）
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    
    smoothness_types = ['position_smoothness_ratio', 'velocity_smoothness_ratio']
    smoothness_labels = ['Position Smoothness', 'Velocity Smoothness']
    x_pos = np.arange(len(smoothness_types))
    width = 0.15  # 更窄的柱子
    
    for i, model_name in enumerate(model_names):
        if 'smoothness' in results[model_name]:
            smoothness_values = [results[model_name]['smoothness'][st] for st in smoothness_types]
            offset = (i - len(model_names)/2) * width
            bars = ax.bar(x_pos + offset, smoothness_values, width,
                   label=model_name.replace('(baseline)', '').replace('(proposed)', '').strip(),
                   color=model_colors.get(model_name, '#7f7f7f'),
                   alpha=0.85, edgecolor='black', linewidth=1.0)
            
            # 添加数值标签
            for bar, val in zip(bars, smoothness_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Smoothness Metric', fontsize=13)
    ax.set_ylabel('Error Ratio (Pred/True)', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(smoothness_labels, fontsize=11)
    ax.legend(loc='best', fontsize=10, ncol=1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect (Ratio=1.0)')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "trajectory_smoothness_error.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path_png = os.path.join(save_dir, "trajectory_smoothness_error.png")
    plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 恢复默认设置
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"✅ 物理一致性分析可视化已保存: {save_dir}")
    print(f"  - 约束违反率图: constraint_violation_rates.pdf/png")
    print(f"  - 轨迹平滑度图: trajectory_smoothness_error.pdf/png")

# ======================================================================================
# 主实验函数
# ======================================================================================

def create_model_for_physics(
    model_type,
    input_dim=6,
    device=torch.device('cpu'),
    input_scaler_mean=None,
    input_scaler_scale=None,
    output_scaler_mean=None,
    output_scaler_scale=None,
):
    """创建物理一致性测试模型（支持所有SOTA模型）"""
    # PLGAFormer：使用统一配置中的模型参数，保持与训练时一致
    if model_type == 'plgaformer':
        model_config = HGVConfig.get_model_config('plgaformer')
        model = PLGAFormerTransformer(
            input_dim=input_dim,
            d_model=model_config.get('d_model', 256),
            nhead=model_config.get('nhead', 8),
            num_encoder_layers=model_config.get('num_encoder_layers', 3),
            num_decoder_layers=model_config.get('num_decoder_layers', 2),
            dim_feedforward=model_config.get('dim_feedforward', 1024),
            dropout=model_config.get('dropout', 0.1),
            output_dim=model_config.get('output_dim', 3),
            use_sparse_attention=False,
            use_physics_corrector=False,
            use_multi_head_output=True,
            use_adaptive_fusion=True,
            input_scaler_mean=input_scaler_mean,
            input_scaler_scale=input_scaler_scale,
            output_scaler_mean=output_scaler_mean,
            output_scaler_scale=output_scaler_scale,
            sampling_interval_s=SAMPLING_INTERVAL_S,
            require_physical_scaler=True,
        )
        return model.to(device)

    # PIT：使用专门的工厂函数
    if model_type == 'pit':
        return create_pit_model(input_dim=input_dim, device=device)

    # 传统 Transformer / Kalman 等基线模型
    if model_type == 'transformer':
        tslib_root = os.getenv("HGV_TSLIB_ROOT", "").strip()
        if not tslib_root:
            raise RuntimeError(
                "Physics-consistency Transformer comparison requires the pinned TSLib checkout. "
                "Set HGV_TSLIB_ROOT before running this diagnostic."
            )
        return create_registered_model(
            'transformer',
            input_dim=input_dim,
            device=device,
            plgaformer_kwargs={"source": "tslib", "tslib_root": tslib_root},
        )
    if model_type == 'kalman':
        return create_baseline_model(model_type, input_dim=input_dim, device=device)

    # 其它 SOTA 时序模型（Informer, PatchTST, FEDformer, TimesNet, iTransformer 等）
    return create_sota_model(model_type, input_dim=input_dim, device=device)

def get_hgv_config():
    """延迟导入HGVConfig"""
    try:
        from models import HGVConfig
        return HGVConfig
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from models import HGVConfig
        return HGVConfig

def main():
    print("="*80)
    print("实验4：PLGAFormer物理一致性分析")
    print("="*80)
    
    # 加载数据
    test_loader, scaler, output_scaler = load_and_prepare_data(TRAIN_CONFIG['batch_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_scaler_mean = torch.from_numpy(output_scaler.mean_.astype(np.float32)).to(device)
    output_scaler_scale = torch.from_numpy(output_scaler.scale_.astype(np.float32)).to(device)
    source_scaler_mean = np.concatenate([output_scaler.mean_, scaler.mean_[3:]]).astype(np.float32)
    source_scaler_scale = np.concatenate([output_scaler.scale_, scaler.scale_[3:]]).astype(np.float32)
    
    print(f"\n使用设备: {device}")
    print(f"预测长度: {PREDICTION_LENGTH}步\n")
    
    # 加载已训练模型（从统一的 MODELS_DIR 目录，与 exp1/exp2 保持一致）
    trained_models = {}
    checkpoint_audits = {}
    models_dir = MODELS_DIR
    
    print("加载已训练模型...")
    unknown_model_types = validate_comparison_model_types(COMPARISON_MODELS)
    if unknown_model_types:
        raise ValueError(f"Unknown comparison model_type(s): {unknown_model_types}")
    for model_name, model_config in COMPARISON_MODELS.items():
        model_type = model_config['model_type']
        try:
            model_path, checkpoint_audit = resolve_exp1_checkpoint(
                PROJECT_ROOT, model_type
            )
            try:
                print(f"  Loading model: {model_name} from {model_path}")
                model = create_model_for_physics(
                    model_type,
                    input_dim=6,
                    device=device,
                    input_scaler_mean=source_scaler_mean,
                    input_scaler_scale=source_scaler_scale,
                    output_scaler_mean=output_scaler.mean_,
                    output_scaler_scale=output_scaler.scale_,
                )
                model.load_state_dict(
                    torch.load(model_path, map_location=device, weights_only=True)
                )
                model.eval()
                trained_models[model_name] = model
                checkpoint_audits[model_name] = checkpoint_audit
            except Exception as e:
                raise RuntimeError(f"Failed to reconstruct {model_name}: {e}") from e
        except (FileNotFoundError, RuntimeError) as e:
            raise RuntimeError(f"Cannot resolve formal checkpoint for {model_name}: {e}") from e
    
    if not trained_models:
        print("\n❌ 没有找到任何已训练模型！")
        print("请先运行 exp1 或 exp2 训练模型，或手动将模型文件放入 trained_models/ 目录")
        return
    
    print(f"\n成功加载 {len(trained_models)} 个模型\n")
    
    # 运行物理一致性评估
    results = {}
    for model_name, model in trained_models.items():
        print(f"\n{'='*80}")
        print(f"评估模型: {model_name}")
        print(f"{'='*80}")
        
        results[model_name] = evaluate_physics_consistency(model, test_loader, device, output_scaler_mean, output_scaler_scale)
        
        # Report dimensionless optimization error and meter-based ECEF errors separately.
        print(f"\n  MSE (scaled):   {results[model_name]['mse_scaled']:.4f}")
        print(f"  ECEF RMSE (m):  {results[model_name]['rmse_cart_m']:.2f}")
        print(f"  ADE / FDE (m):  {results[model_name]['ade_m']:.2f} / {results[model_name]['fde_m']:.2f}")
        print(f"\n  约束违反率:")
        for k, v in results[model_name]['violations'].items():
            print(f"    {k}: {format_physics_metric(k, v)}")
        print(f"\n  轨迹平滑度:")
        for k, v in results[model_name]['smoothness'].items():
            print(f"    {k}: {v:.4f}")
    
    # 生成可视化
    print(f"\n{'='*80}")
    print("生成可视化结果...")
    save_dir = str(RESULTS_DIR)
    generate_physics_consistency_visualization(results, save_dir)

    # 保存结果（JSON/CSV）并复制图到LaTeX图目录
    try:
        os.makedirs(save_dir, exist_ok=True)
        result_paths = save_experiment_results(
            results_dir=RESULTS_DIR,
            filename="physics_consistency_results.json",
            experiment="exp4_physics_consistency",
            config={
                "prediction_length": PREDICTION_LENGTH,
                "report_on_physical_scale": REPORT_ON_PHYSICAL_SCALE,
                "eval_protocol": EVAL_PROTOCOL,
                "eval_ar_seed_mode": EVAL_AR_SEED_MODE,
                "train_supervision_protocol": TRAIN_SUPERVISION_PROTOCOL,
                "optimizer_profile": OPTIMIZER_PROFILE,
                "strict_repro_mode": STRICT_REPRO_MODE,
            },
            results=results,
            extra={
                "metrics": [
                    "mse_scaled",
                    "mse_cart_m2",
                    "rmse_cart_m",
                    "mae_cart_m",
                    "ade_m",
                    "fde_m",
                    "violations.height_violation_rate",
                    "violations.velocity_violation_rate",
                    "violations.acceleration_violation_rate",
                    "violations.velocity_relative_rmse",
                    "violations.acceleration_relative_rmse",
                    "smoothness.position_smoothness_ratio",
                    "smoothness.velocity_smoothness_ratio",
                ],
            },
        )
        json_path = result_paths["latest"]

        csv_path = Path(save_dir) / "physics_consistency_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model","MSE_scaled","MSE_cart_m2","RMSE_cart_m","MAE_cart_m","ADE_m","FDE_m",
                "height_violation","velocity_violation","accel_violation",
                "velocity_relative_rmse","accel_relative_rmse",
                "pos_smooth_ratio","vel_smooth_ratio"
            ])
            for m, r in results.items():
                writer.writerow([
                    m,
                    r.get("mse_scaled", float('nan')),
                    r.get("mse_cart_m2", float('nan')),
                    r.get("rmse_cart_m", float('nan')),
                    r.get("mae_cart_m", float('nan')),
                    r.get("ade_m", float('nan')),
                    r.get("fde_m", float('nan')),
                    r.get("violations", {}).get("height_violation_rate", float('nan')),
                    r.get("violations", {}).get("velocity_violation_rate", float('nan')),
                    r.get("violations", {}).get("acceleration_violation_rate", float('nan')),
                    r.get("violations", {}).get("velocity_relative_rmse", float('nan')),
                    r.get("violations", {}).get("acceleration_relative_rmse", float('nan')),
                    r.get("smoothness", {}).get("position_smoothness_ratio", float('nan')),
                    r.get("smoothness", {}).get("velocity_smoothness_ratio", float('nan')),
                ])

        long_csv_path = Path(save_dir) / "physics_consistency_results_long.csv"
        with open(long_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "experiment", "phase", "seed", "scenario_type", "scenario_value",
                "model", "horizon", "metric", "value"
            ])
            for m, r in results.items():
                writer.writerow(["exp4_physics_consistency", "", "", "physics", "overall", m, PREDICTION_LENGTH, "mse_scaled", r.get("mse_scaled", float("nan"))])
                for metric in ["mse_cart_m2", "rmse_cart_m", "mae_cart_m", "ade_m", "fde_m"]:
                    writer.writerow(["exp4_physics_consistency", "", "", "physics", "overall", m, PREDICTION_LENGTH, metric, r.get(metric, float("nan"))])
                writer.writerow(["exp4_physics_consistency", "", "", "physics", "overall", m, PREDICTION_LENGTH, "height_violation_rate", r.get("violations", {}).get("height_violation_rate", float("nan"))])
                writer.writerow(["exp4_physics_consistency", "", "", "physics", "overall", m, PREDICTION_LENGTH, "velocity_violation_rate", r.get("violations", {}).get("velocity_violation_rate", float("nan"))])
                writer.writerow(["exp4_physics_consistency", "", "", "physics", "overall", m, PREDICTION_LENGTH, "acceleration_violation_rate", r.get("violations", {}).get("acceleration_violation_rate", float("nan"))])
                writer.writerow(["exp4_physics_consistency", "", "", "physics", "overall", m, PREDICTION_LENGTH, "velocity_relative_rmse", r.get("violations", {}).get("velocity_relative_rmse", float("nan"))])
                writer.writerow(["exp4_physics_consistency", "", "", "physics", "overall", m, PREDICTION_LENGTH, "acceleration_relative_rmse", r.get("violations", {}).get("acceleration_relative_rmse", float("nan"))])
                writer.writerow(["exp4_physics_consistency", "", "", "physics", "overall", m, PREDICTION_LENGTH, "position_smoothness_ratio", r.get("smoothness", {}).get("position_smoothness_ratio", float("nan"))])
                writer.writerow(["exp4_physics_consistency", "", "", "physics", "overall", m, PREDICTION_LENGTH, "velocity_smoothness_ratio", r.get("smoothness", {}).get("velocity_smoothness_ratio", float("nan"))])
        latex_fig_dir = os.path.join(os.path.dirname(__file__), "..", "..", "els-cas-AST-main-Trae1030", "figures")
        copied_to_latex = False
        if COPY_FIGURES_TO_LATEX:
            os.makedirs(latex_fig_dir, exist_ok=True)
            for fname in [
                "constraint_violation_rates.pdf","constraint_violation_rates.png",
                "trajectory_smoothness_error.pdf","trajectory_smoothness_error.png",
            ]:
                src = os.path.join(save_dir, fname)
                dst = os.path.join(latex_fig_dir, fname)
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
                    copied_to_latex = True
        metadata_paths = save_run_metadata(
            results_dir=RESULTS_DIR,
            metadata={
                "experiment": "exp4_physics_consistency",
                "seed": 42,
                "prediction_length": PREDICTION_LENGTH,
                "comparison_models": list(COMPARISON_MODELS.keys()),
                "comparison_model_provenance": build_comparison_model_provenance(COMPARISON_MODELS),
                "checkpoint_audits": checkpoint_audits,
                "eval_protocol": EVAL_PROTOCOL,
                "eval_ar_seed_mode": EVAL_AR_SEED_MODE,
                "train_supervision_protocol": TRAIN_SUPERVISION_PROTOCOL,
                "optimizer_profile": OPTIMIZER_PROFILE,
                "strict_repro_mode": STRICT_REPRO_MODE,
                "train_config": TRAIN_CONFIG,
                "model_config": MODEL_CONFIG,
                "physics_config": PHYSICS_CONFIG,
                "model_source_dir": str(MODELS_DIR),
            },
        )
        print(f"✅ 物理一致性结果已保存: {json_path}, {csv_path}")
        print(f"✅ 统一长表已保存: {long_csv_path}")
        print(f"✅ 运行元信息已保存: {metadata_paths['latest']}")
        if copied_to_latex:
            print(f"✅ 图已复制到LaTeX目录: {latex_fig_dir}")
        else:
            print("INFO: LaTeX figure copy skipped for this run")
    except Exception as e:
        print(f"⚠️ 结果保存或复制到LaTeX目录失败: {e}")
    
    print(f"\n{'='*80}")
    print("✅ 物理一致性分析实验完成！")
    print(f"{'='*80}")

if __name__ == '__main__':
    try:
        main()
        print("物理一致性分析实验框架创建完成!")
    except Exception as e:
        print(f"实验错误: {e}")
        import traceback
        traceback.print_exc()
