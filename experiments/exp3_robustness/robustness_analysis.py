#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 3: robustness under noisy, missing, and shortened histories.

Formal outputs use the same de-normalization, ECEF conversion, and
trajectory-level aggregation as Experiment 1. Scaled-space errors are retained
only as optimization diagnostics.
"""

import warnings
warnings.filterwarnings('ignore', message='.*flash attention.*', category=UserWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import csv
from pathlib import Path
from collections import OrderedDict
import joblib
from sklearn.preprocessing import StandardScaler

# Import unified configuration and models（路径与 exp1/exp2 一致）
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import (
    PLGAFormerTransformer, HGVPhysicsLoss, create_baseline_model, create_sota_model,
    create_pit_model, HGVConfig
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
from utils.trajectory_metrics import aggregate_window_metrics_by_trajectory
from utils.final_plgaformer import (
    final_evidence_signature,
    final_plgaformer_kwargs,
    resolve_final_plgaformer,
)
from utils.formal_evidence import load_formal_config
from data_generation.data_paths import get_processed_data_dir
from data_provider.hgv_data import load_hgv_dataset

# 获取统一配置（唯一真相源：从 HGVConfig 读取）
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

# 鲁棒性测试配置 - 从统一配置读取关键参数
_physical_constraints = HGVConfig.get_physical_constraints()
EARTH_RADIUS = _physical_constraints['earth_radius']
_seq_len = TRAIN_CONFIG.get('seq_len', 64)
# 固定预测长度 64 步用于鲁棒性评估（与消融 horizon 一致）
PREDICTION_LENGTH = int(TRAIN_CONFIG.get('pred_len', 256))

MISSING_RATES = [0.0, 0.05, 0.10, 0.15, 0.20]  # 随机缺失率
INPUT_LENGTHS = [64, 128, 192, 256]
PERTURBATION_SEEDS = [1001, 1002, 1003]
MODEL_SEEDS = [42, 123, 456]
_FORMAL_CONFIG = load_formal_config(PROJECT_ROOT / "configs" / "formal_v3.json")
NOISE_LEVELS = [
    float(value)
    for value in _FORMAL_CONFIG["studies"]["robustness"]["sensor_noise"]["scale_multipliers"]
]
SENSOR_NOISE_STD_PHYSICAL = np.asarray(
    [
        100.0,
        2.0e-5,
        2.0e-5,
        10.0,
        np.deg2rad(0.1),
        np.deg2rad(0.1),
    ],
    dtype=np.float32,
)

# 鲁棒性实验仅评估本文方法：PLGAFormer vs 真实数据（顶刊常见设置）
COMPARISON_MODELS = OrderedDict([
    ("Transformer (baseline)", {
        'model_type': 'transformer',
        'description': 'Matched data-driven Transformer baseline',
    }),
    ("PLGAFormer (proposed)", {
        'model_type': 'plgaformer',
        'description': 'Validation-selected PLGAFormer with bounded rotating-Earth prior fusion (C)',
    }),
])

# 结果与模型路径（统一实验目录协议）
RESULTS_DIR, _ = get_experiment_dirs(PROJECT_ROOT, 'exp3_robustness')
_, MODELS_DIR = get_experiment_dirs(PROJECT_ROOT, 'exp1_sota')
DATA_DIR = get_processed_data_dir(PROJECT_ROOT)

print(f"\n{'='*60}")
print(f"ROBUSTNESS ANALYSIS EXPERIMENT CONFIGURATION")
print(f"Batch Size: {TRAIN_CONFIG['batch_size']}")
print(f"Model Dimension: {MODEL_CONFIG['d_model']}")
print(f"Prediction Length: {PREDICTION_LENGTH} | Seq Len: {_seq_len}")
print(f"Results Dir: {RESULTS_DIR}")
print(f"Models Dir: {MODELS_DIR}")
print(f"{'='*60}\n")

# ======================================================================================
# 数据加载和准备
# ======================================================================================

def load_and_prepare_data(batch_size):
    """加载数据集（路径基于 PROJECT_ROOT，与 exp1/exp2 一致）"""
    data_path = DATA_DIR / 'hgv_trajectory_dataset.npz'
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件未找到: {data_path}")
    print(f"📂 加载数据集: {data_path}")
    data_bundle = load_hgv_dataset(data_path, require_trajectory_level=True)
    data = data_bundle.raw
    print(f"数据集规模: 训练集{data['X_train'].shape}, 验证集{data['X_val'].shape}, 测试集{data['X_test'].shape}")
    protocol = data_bundle.protocol
    split_report = data_bundle.split_report
    if split_report is not None:
        print(f"数据协议: {protocol} | 轨迹级划分重叠检查: {split_report}")
    else:
        print(f"数据协议: {protocol}")

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
        output_scaler = StandardScaler()
        train_y_data = data['y_train'].reshape(-1, out_dim)
        output_scaler.fit(train_y_data)
        joblib.dump(output_scaler, str(output_scaler_path))
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
    trajectory_ids = np.asarray(data_bundle.splits["test"].trajectory_ids, dtype=np.int64)
    central_indices = np.asarray(
        [
            indices[len(indices) // 2]
            for trajectory_id in sorted(np.unique(trajectory_ids))
            for indices in [np.flatnonzero(trajectory_ids == trajectory_id)]
        ],
        dtype=np.int64,
    )
    X_test_scaled = X_test_scaled[central_indices]
    y_test_scaled = y_test_scaled[central_indices]
    trajectory_ids = trajectory_ids[central_indices]

    X_test = torch.from_numpy(X_test_scaled.astype(np.float32))
    y_test = torch.from_numpy(y_test_scaled.astype(np.float32))
    
    # 创建DataLoader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    return test_loader, scaler, output_scaler, trajectory_ids

# ======================================================================================
# 鲁棒性测试函数
# ======================================================================================

def add_noise_to_data(x, noise_level, generator=None, noise_std_scaled=None):
    """添加高斯噪声到输入数据"""
    if noise_level == 0.0:
        return x
    noise = torch.randn(
        x.shape, device=x.device, dtype=x.dtype, generator=generator
    )
    if noise_std_scaled is None:
        noise = noise * noise_level
    else:
        noise_scale = torch.as_tensor(
            noise_std_scaled, device=x.device, dtype=x.dtype
        ).view(1, 1, -1)
        noise = noise * noise_scale * noise_level
    return x + noise

def apply_random_missing(x, missing_rate, generator=None):
    """随机缺失数据（将缺失位置置零）"""
    if missing_rate == 0.0:
        return x
    observed = torch.rand(
        (x.size(0), x.size(1), 1),
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    ) > missing_rate
    observed[:, 0, :] = True
    filled = x.clone()
    for index in range(1, x.size(1)):
        filled[:, index, :] = torch.where(
            observed[:, index, :],
            x[:, index, :],
            filled[:, index - 1, :],
        )
    return filled

def adjust_input_length(x, y, target_length):
    """调整输入长度"""
    current_length = x.size(1)
    if target_length == current_length:
        return x, y
    elif target_length < current_length:
        # Keep the forecast origin fixed by retaining the most recent context.
        return x[:, -target_length:, :], y
    else:
        # 填充（使用最后一个时间步的值进行填充）
        last_step = x[:, -1:, :].repeat(1, target_length - current_length, 1)
        return torch.cat([x, last_step], dim=1), y

def generate_causal_mask(size, device):
    """生成因果掩码"""
    mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def unscale_data(scaled_data, mean, scale):
    return scaled_data * scale + mean


def spherical_to_cartesian(spherical_data):
    """Convert [radius, longitude, latitude] to ECEF coordinates in meters."""
    radius = spherical_data[..., 0]
    longitude = spherical_data[..., 1]
    latitude = spherical_data[..., 2]
    cos_latitude = torch.cos(latitude)
    return torch.stack(
        [
            radius * cos_latitude * torch.cos(longitude),
            radius * cos_latitude * torch.sin(longitude),
            radius * torch.sin(latitude),
        ],
        dim=-1,
    )

def evaluate_model_robustness(
    model,
    test_loader,
    device,
    scaler_mean,
    scaler_scale,
    test_type='noise',
    test_param=0.0,
    perturbation_seeds=None,
    trajectory_ids=None,
    noise_std_scaled=None,
):
    """Evaluate one perturbation with physical and trajectory-level metrics."""
    model.eval()
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
    if len(trajectory_ids) != len(test_loader.dataset):
        raise ValueError("trajectory_ids must align one-to-one with test windows")
    stochastic = test_type in {'noise', 'missing'} and float(test_param) > 0.0
    seeds = list(perturbation_seeds or PERTURBATION_SEEDS) if stochastic else [None]
    replicate_metrics = []

    for replicate_seed in seeds:
        scaled_squared_error = 0.0
        scaled_element_count = 0
        cartesian_squared_error = 0.0
        cartesian_element_count = 0
        pred_cartesian = []
        true_cartesian = []
        replicate_ids = []
        sample_offset = 0
        generator = None
        if replicate_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(replicate_seed))

        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(device), y_true.to(device)
                batch_size = x.size(0)
            
            # 应用不同类型的扰动
                if test_type == 'noise':
                    x = add_noise_to_data(
                        x,
                        test_param,
                        generator=generator,
                        noise_std_scaled=noise_std_scaled,
                    )
                elif test_type == 'missing':
                    x = apply_random_missing(x, test_param, generator=generator)
                elif test_type == 'input_length':
                    x, y_true = adjust_input_length(x, y_true, int(test_param))
            
                if y_true.size(1) > PREDICTION_LENGTH:
                    y_true = y_true[:, :PREDICTION_LENGTH, :]
            
                pred_length = min(PREDICTION_LENGTH, y_true.size(1))
                y_pred = unified_predict_by_eval_protocol(
                    model=model,
                    x=x,
                    y_true_scaled=y_true,
                    pred_length=pred_length,
                    device=device,
                    eval_protocol=EVAL_PROTOCOL,
                    eval_ar_seed_mode=EVAL_AR_SEED_MODE,
                    label_len=int(
                        TRAIN_CONFIG.get('label_len', max(1, min(48, y_true.size(1))))
                    ),
                    causal_mask_builder=generate_causal_mask,
                )

                if not torch.isfinite(y_pred).all():
                    raise FloatingPointError("Non-finite prediction in robustness evaluation")
                pred_physical = unscale_data(y_pred, scaler_mean, scaler_scale)
                true_physical = unscale_data(y_true, scaler_mean, scaler_scale)
                pred_cart = spherical_to_cartesian(pred_physical[..., :3])
                true_cart = spherical_to_cartesian(true_physical[..., :3])
                cart_diff = pred_cart - true_cart

                scaled_squared_error += float(((y_pred - y_true) ** 2).sum().item())
                scaled_element_count += int(y_pred.numel())
                cartesian_squared_error += float((cart_diff ** 2).sum().item())
                cartesian_element_count += int(cart_diff.numel())
                pred_cartesian.append(pred_cart.detach().cpu().numpy())
                true_cartesian.append(true_cart.detach().cpu().numpy())
                replicate_ids.append(trajectory_ids[sample_offset:sample_offset + batch_size])
                sample_offset += batch_size

        if sample_offset != len(trajectory_ids):
            raise RuntimeError("Robustness evaluation did not consume all test windows")
        trajectory_report = aggregate_window_metrics_by_trajectory(
            np.concatenate(pred_cartesian, axis=0),
            np.concatenate(true_cartesian, axis=0),
            np.concatenate(replicate_ids, axis=0),
        )
        replicate_metrics.append(
            {
                'seed': replicate_seed,
                'scaled_mse': scaled_squared_error / scaled_element_count,
                'rmse_cart_m': float(np.sqrt(cartesian_squared_error / cartesian_element_count)),
                'ade_m': trajectory_report['mean_ade'],
                'fde_m': trajectory_report['mean_fde'],
                'ade_ci_m': trajectory_report['ade_ci'],
                'fde_ci_m': trajectory_report['fde_ci'],
                'trajectory_count': trajectory_report['trajectory_count'],
            }
        )

    result = {'replicates': replicate_metrics}
    for metric in ('scaled_mse', 'rmse_cart_m', 'ade_m', 'fde_m'):
        values = np.asarray([item[metric] for item in replicate_metrics], dtype=float)
        result[metric] = float(values.mean())
        result[f'{metric}_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    result['trajectory_count'] = int(replicate_metrics[0]['trajectory_count'])
    return result


def aggregate_model_seed_results(seed_results):
    """Pool model-initialization and perturbation replicates."""
    pooled_replicates = []
    for model_seed, result in seed_results:
        for replicate in result["replicates"]:
            pooled_replicates.append({**replicate, "model_seed": int(model_seed)})
    if not pooled_replicates:
        raise RuntimeError("No robustness replicates were produced.")
    output = {"replicates": pooled_replicates}
    for metric in ("scaled_mse", "rmse_cart_m", "ade_m", "fde_m"):
        values = np.asarray([item[metric] for item in pooled_replicates], dtype=float)
        output[metric] = float(values.mean())
        output[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    output["trajectory_count"] = int(pooled_replicates[0]["trajectory_count"])
    output["model_seeds"] = sorted({int(item["model_seed"]) for item in pooled_replicates})
    output["replicate_count"] = len(pooled_replicates)
    return output

# ======================================================================================
# 结果保存（与 exp1/exp2 一致：JSON + CSV）
# ======================================================================================

def _to_serializable(obj):
    """将 numpy 标量转为 Python 原生类型，便于 JSON 保存"""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj

def save_robustness_results(results, results_dir, checkpoint_audits=None):
    """保存鲁棒性结果（JSON + 兼容CSV + 统一长表CSV）"""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    transformer_audits = (checkpoint_audits or {}).get("Transformer (baseline)", {})
    first_transformer_audit = transformer_audits.get(42, transformer_audits.get("42", {}))
    base_signature = str(first_transformer_audit.get("run_signature", ""))
    final_signature, final_signature_audit = final_evidence_signature(base_signature)
    saved = save_experiment_results(
        results_dir=results_dir,
        filename='robustness_results.json',
        experiment='exp3_robustness',
        config={
            'noise_levels': NOISE_LEVELS,
            'run_signature': final_signature,
            'base_exp1_signature': base_signature,
            'final_signature_audit': final_signature_audit,
            'noise_level_semantics': 'multiplier of per-channel physical sensor-noise standard deviations',
            'sensor_noise_std_physical': SENSOR_NOISE_STD_PHYSICAL.tolist(),
            'sensor_noise_units': ['m', 'rad', 'rad', 'm/s', 'rad', 'rad'],
            'missing_rates': MISSING_RATES,
            'input_lengths': INPUT_LENGTHS,
            'model_seeds': MODEL_SEEDS,
            'prediction_length': PREDICTION_LENGTH,
            'seq_len': _seq_len,
            'window_selection': 'central window per held-out source trajectory',
            'eval_protocol': EVAL_PROTOCOL,
            'eval_ar_seed_mode': EVAL_AR_SEED_MODE,
            'train_supervision_protocol': TRAIN_SUPERVISION_PROTOCOL,
            'optimizer_profile': OPTIMIZER_PROFILE,
            'strict_repro_mode': STRICT_REPRO_MODE,
        },
        results=_to_serializable(results),
        extra={
            'metrics': ['scaled_mse', 'rmse_cart_m', 'ade_m', 'fde_m'],
            'primary_metric_protocol': 'ECEF meters, trajectory-level ADE/FDE',
            'checkpoint_audits': checkpoint_audits or {},
        },
    )
    json_path = saved['latest']
    csv_path = results_dir / 'robustness_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['test_type', 'test_param', 'model', 'scaled_mse', 'rmse_cart_m', 'ade_m', 'fde_m'])
        for model_name, data in results.items():
            for level in NOISE_LEVELS:
                if 'noise' in data and level in data['noise']:
                    r = data['noise'][level]
                    writer.writerow(['noise', level, model_name, r['scaled_mse'], r['rmse_cart_m'], r['ade_m'], r['fde_m']])
            for rate in MISSING_RATES:
                if 'missing' in data and rate in data['missing']:
                    r = data['missing'][rate]
                    writer.writerow(['missing', rate, model_name, r['scaled_mse'], r['rmse_cart_m'], r['ade_m'], r['fde_m']])
            for length in INPUT_LENGTHS:
                if 'input_length' in data and length in data['input_length']:
                    r = data['input_length'][length]
                    writer.writerow(['input_length', length, model_name, r['scaled_mse'], r['rmse_cart_m'], r['ade_m'], r['fde_m']])

    long_csv_path = results_dir / 'robustness_results_long.csv'
    with open(long_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'experiment', 'phase', 'seed', 'scenario_type', 'scenario_value',
            'model', 'horizon', 'metric', 'value'
        ])
        for model_name, data in results.items():
            for level in NOISE_LEVELS:
                if 'noise' in data and level in data['noise']:
                    r = data['noise'][level]
                    for metric in ('scaled_mse', 'rmse_cart_m', 'ade_m', 'fde_m'):
                        writer.writerow(['exp3_robustness', '', '', 'noise', level, model_name, PREDICTION_LENGTH, metric, r[metric]])
            for rate in MISSING_RATES:
                if 'missing' in data and rate in data['missing']:
                    r = data['missing'][rate]
                    for metric in ('scaled_mse', 'rmse_cart_m', 'ade_m', 'fde_m'):
                        writer.writerow(['exp3_robustness', '', '', 'missing', rate, model_name, PREDICTION_LENGTH, metric, r[metric]])
            for length in INPUT_LENGTHS:
                if 'input_length' in data and length in data['input_length']:
                    r = data['input_length'][length]
                    for metric in ('scaled_mse', 'rmse_cart_m', 'ade_m', 'fde_m'):
                        writer.writerow(['exp3_robustness', '', '', 'input_length', length, model_name, PREDICTION_LENGTH, metric, r[metric]])
    return str(json_path), str(csv_path), str(long_csv_path)

# ======================================================================================
# 模型创建与加载（与 exp1 一致）
# ======================================================================================

def create_model_for_robustness(
    model_type,
    input_dim=6,
    device=torch.device('cpu'),
    input_scaler_mean=None,
    input_scaler_scale=None,
    output_scaler_mean=None,
    output_scaler_scale=None,
):
    """创建鲁棒性测试用模型（与 exp1 create_model 对齐）"""
    if model_type == 'transformer':
        return create_baseline_model('transformer', input_dim=input_dim, device=device)
    if model_type == 'pit':
        return create_pit_model(input_dim=input_dim, device=device)
    if model_type == 'plgaformer':
        cfg = HGVConfig.get_model_config('plgaformer')
        return PLGAFormerTransformer(
            input_dim=input_dim,
            d_model=cfg.get('d_model', 256),
            nhead=cfg.get('nhead', 8),
            num_encoder_layers=cfg.get('num_encoder_layers', 3),
            num_decoder_layers=cfg.get('num_decoder_layers', 2),
            dim_feedforward=cfg.get('dim_feedforward', 1024),
            dropout=cfg.get('dropout', 0.1),
            output_dim=cfg.get('output_dim', 3),
            use_adaptive_fusion=True,
            **final_plgaformer_kwargs(),
            input_scaler_mean=input_scaler_mean,
            input_scaler_scale=input_scaler_scale,
            output_scaler_mean=output_scaler_mean,
            output_scaler_scale=output_scaler_scale,
            sampling_interval_s=float(TRAIN_CONFIG.get('sampling_interval_s', 1.0)),
            require_physical_scaler=True,
        ).to(device)
    try:
        return create_sota_model(model_type, input_dim=input_dim, device=device)
    except ValueError:
        raise ValueError(f"Unsupported model_type: {model_type}")

def main():
    print("="*80)
    print("实验3：PLGAFormer鲁棒性分析")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载数据
    test_loader, scaler, output_scaler, trajectory_ids = load_and_prepare_data(TRAIN_CONFIG['batch_size'])
    output_scaler_mean = torch.from_numpy(output_scaler.mean_.astype(np.float32)).to(device)
    output_scaler_scale = torch.from_numpy(output_scaler.scale_.astype(np.float32)).to(device)
    source_scaler_mean = np.concatenate([output_scaler.mean_, scaler.mean_[3:]]).astype(np.float32)
    source_scaler_scale = np.concatenate([output_scaler.scale_, scaler.scale_[3:]]).astype(np.float32)
    sensor_noise_std_scaled = SENSOR_NOISE_STD_PHYSICAL / source_scaler_scale

    print("测试配置:")
    print(f"  - 噪声水平: {NOISE_LEVELS}")
    print(f"  - 缺失率: {MISSING_RATES}")
    print(f"  - 输入长度: {INPUT_LENGTHS}")
    print(f"  - 预测长度: {PREDICTION_LENGTH}\n")

    # 加载已训练模型（从 exp1_sota/trained_models）
    trained_models = {}
    checkpoint_audits = {}
    print("加载已训练模型...")
    unknown_model_types = validate_comparison_model_types(COMPARISON_MODELS)
    if unknown_model_types:
        raise ValueError(f"Unknown comparison model_type(s): {unknown_model_types}")
    for model_name, model_config in COMPARISON_MODELS.items():
        model_type = model_config['model_type']
        trained_models[model_name] = {}
        checkpoint_audits[model_name] = {}
        for model_seed in MODEL_SEEDS:
            try:
                if model_type == "plgaformer":
                    model_path, _, checkpoint_audit = resolve_final_plgaformer(model_seed)
                else:
                    model_path, checkpoint_audit = resolve_exp1_checkpoint(
                        PROJECT_ROOT, model_type, seed=model_seed
                    )
                try:
                    print(f"  Loading model: {model_name}, seed={model_seed} from {model_path}")
                    model = create_model_for_robustness(
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
                    trained_models[model_name][model_seed] = model
                    checkpoint_audits[model_name][model_seed] = checkpoint_audit
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to reconstruct {model_name}, seed={model_seed}: {e}"
                    ) from e
            except (FileNotFoundError, RuntimeError) as e:
                raise RuntimeError(
                    f"Cannot resolve formal checkpoint for {model_name}, seed={model_seed}: {e}"
                ) from e

    if not trained_models:
        print("\n❌ 没有找到任何已训练模型！")
        print("请先运行 experiments/exp1_sota/SOTA_comparison.py 生成 trained_models 后再运行本实验。")
        return
    print(f"\n成功加载 {len(trained_models)} 个模型\n")
    
    # 运行鲁棒性测试
    results = {}
    total_tests = len(trained_models) * (len(NOISE_LEVELS) + len(MISSING_RATES) + len(INPUT_LENGTHS))
    current_test = 0
    
    for model_name, seeded_models in trained_models.items():
        print(f"\n{'='*80}")
        print(f"测试模型: {model_name}")
        print(f"{'='*80}")
        
        results[model_name] = {
            'noise': {},
            'missing': {},
            'input_length': {}
        }
        
        # 1. 噪声鲁棒性测试
        print("\n[1/3] 噪声鲁棒性测试...")
        for noise_level in NOISE_LEVELS:
            current_test += 1
            print(f"  进度: [{current_test}/{total_tests}] 噪声水平 {noise_level*100:.0f}%", end=" ")
            result = aggregate_model_seed_results([
                (
                    model_seed,
                    evaluate_model_robustness(
                        model, test_loader, device, output_scaler_mean, output_scaler_scale,
                        'noise', noise_level, trajectory_ids=trajectory_ids,
                        noise_std_scaled=sensor_noise_std_scaled,
                    ),
                )
                for model_seed, model in seeded_models.items()
            ])
            results[model_name]['noise'][noise_level] = result
            print(f"✓ ADE={result['ade_m']:.1f} m, FDE={result['fde_m']:.1f} m")
        
        # 2. 数据缺失鲁棒性测试
        print("\n[2/3] 数据缺失鲁棒性测试...")
        for missing_rate in MISSING_RATES:
            current_test += 1
            print(f"  进度: [{current_test}/{total_tests}] 缺失率 {missing_rate*100:.0f}%", end=" ")
            result = aggregate_model_seed_results([
                (
                    model_seed,
                    evaluate_model_robustness(
                        model, test_loader, device, output_scaler_mean, output_scaler_scale,
                        'missing', missing_rate, trajectory_ids=trajectory_ids,
                    ),
                )
                for model_seed, model in seeded_models.items()
            ])
            results[model_name]['missing'][missing_rate] = result
            print(f"✓ ADE={result['ade_m']:.1f} m, FDE={result['fde_m']:.1f} m")
        
        # 3. 输入长度变化测试
        print("\n[3/3] 输入长度变化鲁棒性测试...")
        for input_length in INPUT_LENGTHS:
            current_test += 1
            print(f"  进度: [{current_test}/{total_tests}] 输入长度 {input_length}步", end=" ")
            result = aggregate_model_seed_results([
                (
                    model_seed,
                    evaluate_model_robustness(
                        model, test_loader, device, output_scaler_mean, output_scaler_scale,
                        'input_length', input_length, trajectory_ids=trajectory_ids,
                    ),
                )
                for model_seed, model in seeded_models.items()
            ])
            results[model_name]['input_length'][input_length] = result
            print(f"✓ ADE={result['ade_m']:.1f} m, FDE={result['fde_m']:.1f} m")
    
    # 保存结果（JSON + CSV），与 exp1/exp2 一致
    print(f"\n{'='*80}")
    print("保存实验结果...")
    json_path, csv_path, long_csv_path = save_robustness_results(
        results, RESULTS_DIR, checkpoint_audits=checkpoint_audits
    )
    print(f"✅ 已保存: {json_path}")
    print(f"✅ 已保存: {csv_path}")
    print(f"✅ 已保存: {long_csv_path}")
    metadata_paths = save_run_metadata(
        results_dir=RESULTS_DIR,
        metadata={
            "experiment": "exp3_robustness",
            "seed": 42,
            "model_seeds": MODEL_SEEDS,
            "prediction_length": PREDICTION_LENGTH,
            "seq_len": _seq_len,
            "noise_levels": NOISE_LEVELS,
            "noise_level_semantics": "multiplier of per-channel physical sensor-noise standard deviations",
            "sensor_noise_std_physical": SENSOR_NOISE_STD_PHYSICAL.tolist(),
            "sensor_noise_units": ["m", "rad", "rad", "m/s", "rad", "rad"],
            "missing_rates": MISSING_RATES,
            "input_lengths": INPUT_LENGTHS,
            "perturbation_seeds": PERTURBATION_SEEDS,
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
    print(f"✅ 运行元信息已保存: {metadata_paths['latest']}")
    print(f"\n📊 生成图表请运行: python experiments/exp3_robustness/visualize_results.py")
    print(f"{'='*80}")
    print("✅ 鲁棒性分析实验完成！")
    print(f"{'='*80}")

if __name__ == '__main__':
    try:
        main()
        print("鲁棒性分析实验框架创建完成!")
    except Exception as e:
        print(f"实验错误: {e}")
        import traceback
        traceback.print_exc()
        raise

