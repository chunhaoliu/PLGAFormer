"""
独立可视化脚本 - 基于已保存的预测结果快速生成顶刊级别图表
无需重新训练或评估，运行时间 < 1分钟

使用方法:
    python experiments/exp1_sota/visualize_results.py

功能:
    1. 从predictions_cache.npz加载预测结果
    2. 从baseline_results.json加载性能指标
    3. 生成多种顶刊级别可视化图表

作者: PLGAFormer Team
日期: 2024
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import shutil

# 设置matplotlib后端
matplotlib.use('Agg')

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models import HGVConfig

# 从统一配置获取地球半径（确保全项目一致）
_physical_constraints = HGVConfig.get_physical_constraints()
R_EARTH = _physical_constraints['earth_radius']

# ======================================================================================
# 辅助函数
# ======================================================================================

def load_cached_predictions(cache_path):
    """加载缓存的预测结果"""
    print(f"📂 加载预测结果: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    
    scaler_mean = data['scaler_mean']
    scaler_scale = data['scaler_scale']
    y_test = data['y_test']
    predictions = data['predictions'].item()
    
    if scaler_mean.shape[0] != y_test.shape[-1]:
        dim = min(scaler_mean.shape[0], y_test.shape[-1])
        scaler_mean = scaler_mean[:dim]
        scaler_scale = scaler_scale[:dim]
        y_test = y_test[..., :dim]
        for key, value in predictions.items():
            predictions[key] = value[..., :dim]
    
    predictions_cache = {
        'X_test': data['X_test'],
        'y_test': y_test,
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
        'predictions': predictions
    }
    
    print(f"   ✓ 样本数量: {len(predictions_cache['y_test'])}")
    print(f"   ✓ 包含模型: {list(predictions_cache['predictions'].keys())}")
    
    return predictions_cache


def load_performance_results(json_path):
    """加载性能指标结果"""
    print(f"📂 加载性能指标: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"   ✓ 包含模型: {list(results.keys())}")
    return results


def unscale_data(scaled_data, mean, scale):
    """反标准化数据"""
    return scaled_data * scale + mean


def spherical_to_geo(data):
    """将球坐标(r, λ, φ)转换为地理坐标(lon, lat, h)"""
    r = data[:, 0] if data.ndim == 2 else data[:, :, 0]
    lon = np.rad2deg(data[:, 1] if data.ndim == 2 else data[:, :, 1])
    lat = np.rad2deg(data[:, 2] if data.ndim == 2 else data[:, :, 2])
    h = (r - R_EARTH) / 1000.0  # 高度（km）
    return lon, lat, h


def set_publication_style():
    """顶刊级别绘图样式（TPAMI/TIP/NeurIPS 风格）"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#333333',
        'legend.fancybox': False,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'axes.spines.top': True,
        'axes.spines.right': True,
    })


# ======================================================================================
# SOTA 模型配色（根据实际数据动态选择）
# ======================================================================================
SOTA_MODEL_COLORS = {
    "PIT": '#1f77b4',
    "Transformer (baseline)": '#1f77b4',
    "PLGAFormer (proposed)": '#d62728',
    "Kalman": '#7f7f7f',
    "FEDformer": '#2ca02c',
    "TimesNet": '#d62728',
    "iTransformer": '#9467bd',
    "PLGAFormer": '#ff7f0e',
    "Informer": '#17becf',
}
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']


def _get_model_color(model_name, idx=0):
    """根据模型名或索引获取颜色"""
    return SOTA_MODEL_COLORS.get(model_name, COLOR_PALETTE[idx % len(COLOR_PALETTE)])


# ======================================================================================
# 可视化函数
# ======================================================================================

def plot_error_accumulation(predictions_cache, save_dir):
    """图1: 误差累积曲线 + 置信区间（单图，顶刊标准）"""
    print("\n📊 生成图: 误差累积曲线...")
    preds = predictions_cache.get('predictions', {})
    if not preds:
        print("   ⚠️ 无预测数据，跳过")
        return
    set_publication_style()
    y_test = predictions_cache['y_test']
    scaler_mean = predictions_cache['scaler_mean']
    scaler_scale = predictions_cache['scaler_scale']
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, (model_name, y_pred) in enumerate(preds.items()):
        color = _get_model_color(model_name, idx)
        y_pred_phys = unscale_data(y_pred, scaler_mean, scaler_scale)
        y_true_phys = unscale_data(y_test, scaler_mean, scaler_scale)
        seq_len = min(y_pred_phys.shape[1], y_true_phys.shape[1])
        mse_per_step = np.array([
            ((y_pred_phys[:, t, :] - y_true_phys[:, t, :]) ** 2).mean(axis=1).mean()
            for t in range(seq_len)
        ])
        std_per_step = np.array([
            ((y_pred_phys[:, t, :] - y_true_phys[:, t, :]) ** 2).mean(axis=1).std()
            for t in range(seq_len)
        ])
        t_steps = np.arange(seq_len)
        ax.plot(t_steps, mse_per_step, color=color, linewidth=2.5,
                label=model_name.replace('(baseline)', '').replace('(proposed)', '').strip())
        ax.fill_between(t_steps, mse_per_step - std_per_step, mse_per_step + std_per_step,
                        color=color, alpha=0.2)
    ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.15, linestyle=':')
    ax.set_yscale('log')
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        p = os.path.join(save_dir, f"error_accumulation.{fmt}")
        plt.savefig(p, format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()
    print(f"   ✓ error_accumulation.pdf/png")


def _horizon_key(h):
    """JSON 中 horizon 键可能是 str 或 int"""
    return str(h) if isinstance(h, int) else h

def plot_multi_horizon_mse(performance_results, save_dir):
    """图2: 多步长 MSE 柱状图（单图）"""
    print("\n📊 生成图: 多步长 MSE...")
    if not performance_results:
        return
    set_publication_style()
    _train_cfg = HGVConfig.get_train_config()
    horizons = _train_cfg.get('prediction_horizons', [32, 64, 96, 128])
    first_model = next(iter(performance_results.values()))
    available_horizons = [h for h in horizons if _horizon_key(h) in first_model]
    if not available_horizons:
        return
    x_pos = np.arange(len(available_horizons))
    width = 0.8 / max(len(performance_results), 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, model_name in enumerate(performance_results.keys()):
        if model_name not in performance_results:
            continue
        mse_values = [performance_results[model_name][_horizon_key(h)]['mse'] for h in available_horizons]
        offset = (i - len(performance_results) / 2 + 0.5) * width
        ax.bar(x_pos + offset, mse_values, width,
               label=model_name.replace('(baseline)', '').replace('(proposed)', '').strip(),
               color=_get_model_color(model_name, i), alpha=0.85, edgecolor='#333', linewidth=0.8)
    ax.set_xlabel('Prediction Horizon (steps)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MSE (scaled)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(h) for h in available_horizons])
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.minorticks_on()
    ax.grid(True, which='minor', axis='y', alpha=0.15, linestyle=':')
    ax.set_yscale('log')
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        p = os.path.join(save_dir, f"multi_horizon_mse.{fmt}")
        plt.savefig(p, format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()
    print(f"   ✓ multi_horizon_mse.pdf/png")


def plot_performance_matrices(performance_results, save_dir):
    """
    性能矩阵热图（原先在 SOTA_comparison.py 中）
    使用 baseline_results.json 中的均值结果生成 MSE/MAE 矩阵。
    """
    print("\n📊 生成图: 性能矩阵热图...")
    if not performance_results:
        return

    set_publication_style()

    model_names = list(performance_results.keys())
    train_cfg = HGVConfig.get_train_config()
    horizons = train_cfg.get('prediction_horizons', [32, 64, 128, 256])

    def _hk(h):
        return str(h)

    # 1. MSE 矩阵
    mse_matrix = np.zeros((len(model_names), len(horizons)))
    for i, m in enumerate(model_names):
        for j, h in enumerate(horizons):
            mse_matrix[i, j] = performance_results[m].get(_hk(h), {}).get('mse', np.nan)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    im = ax.imshow(mse_matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(horizons)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([str(h) for h in horizons], fontsize=10)
    ax.set_yticklabels(
        [name.replace('(baseline)', '').replace('(proposed)', '').strip() for name in model_names],
        fontsize=10,
    )

    ax.set_xlabel('Prediction Horizon (steps)', fontsize=13)
    ax.set_ylabel('Model', fontsize=13)

    for i in range(len(model_names)):
        for j in range(len(horizons)):
            val = mse_matrix[i, j]
            txt = 'N/A' if np.isnan(val) else f'{val:.3f}'
            ax.text(j, i, txt, ha='center', va='center', color='black', fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MSE (scaled)', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        p = os.path.join(save_dir, f"performance_matrix_mse.{fmt}")
        plt.savefig(p, format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()

    # 2. MAE 矩阵
    mae_matrix = np.zeros((len(model_names), len(horizons)))
    for i, m in enumerate(model_names):
        for j, h in enumerate(horizons):
            mae_matrix[i, j] = performance_results[m].get(_hk(h), {}).get('mae', np.nan)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    im = ax.imshow(mae_matrix, cmap='YlGnBu', aspect='auto')

    ax.set_xticks(np.arange(len(horizons)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([str(h) for h in horizons], fontsize=10)
    ax.set_yticklabels(
        [name.replace('(baseline)', '').replace('(proposed)', '').strip() for name in model_names],
        fontsize=10,
    )

    ax.set_xlabel('Prediction Horizon (steps)', fontsize=13)
    ax.set_ylabel('Model', fontsize=13)

    for i in range(len(model_names)):
        for j in range(len(horizons)):
            val = mae_matrix[i, j]
            txt = 'N/A' if np.isnan(val) else f'{val:.3f}'
            ax.text(j, i, txt, ha='center', va='center', color='black', fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MAE (scaled)', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        p = os.path.join(save_dir, f"performance_matrix_mae.{fmt}")
        plt.savefig(p, format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()

    # 复制到论文图目录（与原逻辑保持一致）
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    latex_fig_dir = os.path.join(project_root, "els-cas-AST-main-Trae1030", "figures")
    try:
        os.makedirs(latex_fig_dir, exist_ok=True)
        for fname in [
            "performance_matrix_mse.pdf",
            "performance_matrix_mse.png",
            "performance_matrix_mae.pdf",
            "performance_matrix_mae.png",
        ]:
            src = os.path.join(save_dir, fname)
            dst = os.path.join(latex_fig_dir, fname)
            if os.path.exists(src):
                shutil.copyfile(src, dst)
        print(f"   ✓ 性能矩阵图已复制到 LaTeX 目录: {latex_fig_dir}")
    except Exception as e:
        print(f"   ⚠️ 性能矩阵图复制到 LaTeX 目录失败: {e}")


def plot_3d_trajectory(predictions_cache, save_dir, sample_idx=0):
    """图: 3D 轨迹对比（单图）"""
    print("\n📊 生成图: 3D 轨迹...")
    preds = predictions_cache.get('predictions', {})
    if not preds:
        return
    set_publication_style()
    y_test = predictions_cache['y_test']
    scaler_mean = predictions_cache['scaler_mean']
    scaler_scale = predictions_cache['scaler_scale']
    y_true_phys = unscale_data(y_test[sample_idx], scaler_mean, scaler_scale)
    pred_0 = next(iter(preds.values()))[sample_idx]
    min_len = min(y_true_phys.shape[0], pred_0.shape[0])
    y_true_phys = y_true_phys[:min_len]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    lon_true, lat_true, h_true = spherical_to_geo(y_true_phys)
    ax.plot(lon_true, lat_true, h_true, color='#000000', linestyle='-', linewidth=3.0, label='Ground Truth', zorder=10)
    for idx, (model_name, y_pred) in enumerate(preds.items()):
        y_pred_phys = unscale_data(y_pred[sample_idx][:min_len], scaler_mean, scaler_scale)
        lon_pred, lat_pred, h_pred = spherical_to_geo(y_pred_phys)
        ax.plot(lon_pred, lat_pred, h_pred, color=_get_model_color(model_name, idx), linestyle='--', linewidth=2.2,
                label=model_name.replace('(baseline)', '').replace('(proposed)', '').strip(), alpha=0.85)
    ax.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_zlabel('Altitude (km)', fontsize=12, fontweight='bold', labelpad=8)
    ax.legend(loc='upper right', fontsize=9)
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(os.path.join(save_dir, f"trajectory_3d.{fmt}"), format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()
    print(f"   ✓ trajectory_3d.pdf/png")


def _select_representative_sample(predictions_cache, reference_model=None, horizon_steps=None):
    """选择一个“代表性样本”（误差中位数附近），用于轨迹可视化。"""
    preds = predictions_cache.get('predictions', {})
    if not preds:
        return 0

    y_test = predictions_cache['y_test']
    scaler_mean = predictions_cache['scaler_mean']
    scaler_scale = predictions_cache['scaler_scale']

    # 选择参考模型：优先 PLGAFormer，其次 Transformer，再其次第一个模型
    model_names = list(preds.keys())
    ref_name = reference_model
    if ref_name is None:
        for cand in ["PLGAFormer (proposed)", "PLGAFormer", "Transformer (baseline)"]:
            if cand in preds:
                ref_name = cand
                break
        if ref_name is None:
            ref_name = model_names[0]

    y_pred = preds[ref_name]
    y_true_phys = unscale_data(y_test, scaler_mean, scaler_scale)
    y_pred_phys = unscale_data(y_pred, scaler_mean, scaler_scale)

    seq_len = min(y_true_phys.shape[1], y_pred_phys.shape[1])
    if horizon_steps is not None:
        seq_len = min(seq_len, horizon_steps)

    diff = y_pred_phys[:, :seq_len, :] - y_true_phys[:, :seq_len, :]
    mse_per_sample = (diff ** 2).mean(axis=(1, 2))
    median_idx = int(np.argsort(mse_per_sample)[len(mse_per_sample) // 2])
    return median_idx


def plot_pit_style_trajectory(predictions_cache, save_dir, sample_idx=None):
    """PIT 风格: 左 3D 轨迹 + 右侧高度-时间曲线及局部放大。"""
    print("\n📊 生成图: PIT 风格 3D+高度 曲线...")
    preds = predictions_cache.get('predictions', {})
    if not preds:
        print("   ⚠️ 无预测数据，跳过")
        return

    set_publication_style()
    y_test = predictions_cache['y_test']
    scaler_mean = predictions_cache['scaler_mean']
    scaler_scale = predictions_cache['scaler_scale']

    if sample_idx is None:
        sample_idx = _select_representative_sample(predictions_cache, horizon_steps=None)
        print(f"   ✓ 自动选取代表样本: index={sample_idx}")

    y_true_phys = unscale_data(y_test[sample_idx], scaler_mean, scaler_scale)
    first_pred = next(iter(preds.values()))[sample_idx]
    min_len = min(y_true_phys.shape[0], first_pred.shape[0])
    y_true_phys = y_true_phys[:min_len]

    # 时间轴（秒），数据生成器使用 2s 间隔
    dt = 2.0
    t = np.arange(min_len) * dt

    # 预先计算所有模型的经纬高序列
    model_series = {}
    for idx, (model_name, y_pred_scaled) in enumerate(preds.items()):
        y_pred_phys = unscale_data(y_pred_scaled[sample_idx], scaler_mean, scaler_scale)[:min_len]
        lon_pred, lat_pred, h_pred = spherical_to_geo(y_pred_phys)
        model_series[model_name] = {
            'lon': lon_pred,
            'lat': lat_pred,
            'h': h_pred,
            'color': _get_model_color(model_name, idx),
        }

    lon_true, lat_true, h_true = spherical_to_geo(y_true_phys)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, width_ratios=[1.4, 1.6], figure=fig)

    # 左侧 3D 轨迹
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.plot(lon_true, lat_true, h_true, color='#000000', linestyle='-', linewidth=3.0,
              label='Real trajectory', zorder=10)
    for name, series in model_series.items():
        ax3d.plot(series['lon'], series['lat'], series['h'],
                  color=series['color'], linewidth=2.0,
                  label=name.replace('(baseline)', '').replace('(proposed)', '').strip())
    ax3d.set_xlabel('Longitude (deg)', fontsize=11)
    ax3d.set_ylabel('Latitude (deg)', fontsize=11)
    ax3d.set_zlabel('Altitude (km)', fontsize=11)
    ax3d.view_init(elev=20, azim=-60)
    ax3d.grid(True, alpha=0.3)
    ax3d.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # 右侧高度-时间主图
    ax_main = fig.add_subplot(gs[0, 1])
    ax_main.plot(t, h_true, color='#000000', linewidth=3.0, label='Real trajectory', zorder=10)
    for name, series in model_series.items():
        ax_main.plot(t, series['h'],
                     color=series['color'],
                     linewidth=2.0,
                     label=name.replace('(baseline)', '').replace('(proposed)', '').strip())
    ax_main.set_xlabel('Time (s)', fontsize=12)
    ax_main.set_ylabel('Altitude (km)', fontsize=12)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.legend(loc='upper right', fontsize=9, framealpha=0.95)

    # 两个局部放大区域（时间段按比例自动选取）
    if min_len > 40:
        start1, end1 = int(0.1 * min_len), int(0.25 * min_len)
        start2, end2 = int(0.55 * min_len), int(0.75 * min_len)

        def add_inset(ax_parent, t_arr, h_true_arr, model_series_dict,
                      t_start, t_end, width_inch, height_inch, anchor_xy, loc):
            # 使用绝对尺寸(英寸)，以便 bbox_to_anchor 可用 (x,y) 二元组
            ax_ins = inset_axes(
                ax_parent,
                width=width_inch,
                height=height_inch,
                loc=loc,
                bbox_to_anchor=anchor_xy,
                bbox_transform=ax_parent.transAxes,
                borderpad=0.5,
            )
            mask = (t_arr >= t_start) & (t_arr <= t_end)
            ax_ins.plot(t_arr[mask], h_true_arr[mask], color='#000000', linewidth=2.0)
            for name, series in model_series_dict.items():
                ax_ins.plot(t_arr[mask], series['h'][mask],
                            color=series['color'], linewidth=1.5)
            ax_ins.set_xlim(t_start, t_end)
            ax_ins.grid(True, alpha=0.3, linestyle='--')
            for label in ax_ins.get_xticklabels():
                label.set_fontsize(8)
            for label in ax_ins.get_yticklabels():
                label.set_fontsize(8)
            ax_parent.axvspan(t_start, t_end, color='orange', alpha=0.08)

        add_inset(
            ax_main, t, h_true, model_series,
            t[start1], t[end1],
            width_inch=2.8, height_inch=1.6,
            anchor_xy=(0.02, 0.98),
            loc='upper left',
        )
        add_inset(
            ax_main, t, h_true, model_series,
            t[start2], t[end2],
            width_inch=2.8, height_inch=1.6,
            anchor_xy=(0.02, 0.02),
            loc='lower left',
        )

    # 含 3D 与 inset 时不用 tight_layout，改用 subplots_adjust 避免警告
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.12, top=0.95, wspace=0.25)
    os.makedirs(save_dir, exist_ok=True)
    for fmt in ['pdf', 'png']:
        plt.savefig(os.path.join(save_dir, f"pit_style_trajectory_sample{sample_idx}.{fmt}"),
                    format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()
    print(f"   ✓ pit_style_trajectory_sample{sample_idx}.pdf/png")


def plot_altitude_evolution(predictions_cache, save_dir, sample_idx=0):
    """图: 高度演化对比（单图）"""
    print("\n📊 生成图: 高度演化...")
    preds = predictions_cache.get('predictions', {})
    if not preds:
        return
    set_publication_style()
    y_test = predictions_cache['y_test']
    scaler_mean = predictions_cache['scaler_mean']
    scaler_scale = predictions_cache['scaler_scale']
    y_true_phys = unscale_data(y_test[sample_idx], scaler_mean, scaler_scale)
    pred_0 = next(iter(preds.values()))[sample_idx]
    min_len = min(y_true_phys.shape[0], pred_0.shape[0])
    y_true_phys = y_true_phys[:min_len]
    _, _, h_true = spherical_to_geo(y_true_phys)
    time_steps = np.arange(min_len)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_steps, h_true, color='#000000', linewidth=2.5, label='Ground Truth')
    for idx, model_name in enumerate(preds.keys()):
        y_pred_phys = unscale_data(preds[model_name][sample_idx][:min_len], scaler_mean, scaler_scale)
        _, _, h_pred = spherical_to_geo(y_pred_phys)
        ax.plot(time_steps, h_pred, color=_get_model_color(model_name, idx), linestyle='--', linewidth=2.0,
                label=model_name.replace('(baseline)', '').replace('(proposed)', '').strip(), alpha=0.85)
    ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Altitude (km)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig(os.path.join(save_dir, f"altitude_evolution.{fmt}"), format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()
    print(f"   ✓ altitude_evolution.pdf/png")


def plot_error_heatmap(predictions_cache, save_dir, sample_idx=0):
    """图: 误差热图（单图） - 已弃用"""
    print("⚠️ 已跳过误差热图绘制（函数保留占位但不再生成图）。")


def plot_efficiency_performance_tradeoff(performance_results, efficiency_results, save_dir):
    """效率-性能权衡散点图 - 已弃用"""
    print("⚠️ 已跳过效率-性能权衡图绘制（函数保留占位但不再生成图）。")


def plot_error_distribution_boxplot(predictions_cache, save_dir):
    """误差分布箱线图 - 已弃用"""
    print("⚠️ 已跳过误差分布箱线图绘制（函数保留占位但不再生成图）。")


# ======================================================================================
# 主函数
# ======================================================================================

def main():
    """主函数：生成所有可视化图表"""
    print("\n" + "="*80)
    print("独立可视化脚本 - 快速生成顶刊级别图表")
    print("="*80)
    
    # 设置路径
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    figures_dir = os.path.join(results_dir, "figures_advanced")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 加载数据
    cache_path = os.path.join(results_dir, "predictions_cache.npz")
    json_path = os.path.join(results_dir, "baseline_results.json")
    
    if not os.path.exists(cache_path):
        print(f"\n❌ 错误: 找不到预测缓存文件: {cache_path}")
        print("   请先运行 SOTA_comparison.py 生成预测结果")
        return
    
    if not os.path.exists(json_path):
        print(f"\n❌ 错误: 找不到性能结果文件: {json_path}")
        return
    
    predictions_cache = load_cached_predictions(cache_path)
    performance_results = load_performance_results(json_path)
    
    # 生成图表
    print("\n" + "="*80)
    print("开始生成顶刊级别可视化图表...")
    print("="*80)
    
    # 图1-2: 误差累积曲线、多步长 MSE（各自单图）
    plot_error_accumulation(predictions_cache, figures_dir)
    plot_multi_horizon_mse(performance_results, figures_dir)
    # 性能矩阵热图（原本在训练脚本中）
    plot_performance_matrices(performance_results, figures_dir)
    
    # 图3-4: 3D 轨迹、PIT 风格 3D+高度、高度演化（各自单图）
    plot_3d_trajectory(predictions_cache, figures_dir, sample_idx=0)
    plot_pit_style_trajectory(predictions_cache, figures_dir, sample_idx=None)
    plot_altitude_evolution(predictions_cache, figures_dir, sample_idx=0)
    
    print("\n" + "="*80)
    print("✅ 所有图表生成完成！")
    print(f"   保存位置: {figures_dir}")
    print("="*80)
    
    # 恢复默认设置
    plt.rcParams.update(plt.rcParamsDefault)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

