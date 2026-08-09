"""
独立可视化脚本 - 基于已保存的消融实验结果快速生成顶刊级别图表
无需重新训练，运行时间 < 1 分钟

使用方法:
    python experiments/exp2_ablation/visualize_results.py

功能:
    1. 从 results/latest_results.json 加载消融统计结果
    2. 生成结构消融柱状图（仅 Baseline、Full、w/o A/B/C，5 个模型）

作者: PLGAFormer Team
日期: 2024
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# 仅展示的 5 个模型；顺序为图上自下而上（barh 中 index 0 在最下）
# 目标自上而下：Transformer(baseline), w/o A, w/o B, w/o C, PLGAFormer(proposed)
# 故数据顺序自下而上：PLGAFormer, w/o C, w/o B, w/o A, Transformer
ABLATION_MODEL_ORDER = [
    ("PLGAFormer (A+B+C)", "PLGAFormer (proposed)"),
    ("PLGAFormer w/o C", "w/o C"),
    ("PLGAFormer w/o B", "w/o B"),
    ("PLGAFormer w/o A", "w/o A"),
    ("Transformer (baseline)", "Transformer (baseline)"),
]

HORIZON = 64
RESULTS_JSON = "latest_results.json"
PHASE_KEY = "phase1_structural_ablation"


def set_publication_style():
    """顶刊级别绘图样式（与 Informer/PatchTST 等一致）"""
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
        'grid.linewidth': 1.0,
        'figure.dpi': 100,
        'savefig.dpi': 300,
    })


def load_ablation_results(json_path):
    """从 latest_results.json 加载消融统计"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    phases = data.get('phases', {})
    phase = phases.get(PHASE_KEY, {})
    stats = phase.get('statistics', {})
    num_runs = phase.get('num_runs', 2)
    return stats, num_runs


def plot_ablation_bars(stats, num_runs, save_dir):
    """
    生成结构消融柱状图（顶刊标准）
    仅包含 Baseline、Full、w/o A/B/C
    """
    main_horizon = HORIZON
    horizon_key = str(main_horizon)  # JSON 中 horizon 为字符串键
    models_data = []

    for model_key, display_name in ABLATION_MODEL_ORDER:
        if model_key not in stats or horizon_key not in stats[model_key]:
            continue
        h = stats[model_key][horizon_key]
        mae_mean = h.get('mae', {}).get('mean', float('nan'))
        mse_mean = h.get('mse', {}).get('mean', float('nan'))
        mae_std = h.get('mae', {}).get('std', 0)
        mse_std = h.get('mse', {}).get('std', 0)
        models_data.append((display_name, mae_mean, mse_mean, mae_std, mse_std))

    if len(models_data) < 2:
        print("⚠️ 数据不足，至少需要 Baseline 与一个变体")
        return

    model_names = [item[0] for item in models_data]
    mae_values = [item[1] for item in models_data]
    rmse_values = [np.sqrt(item[2]) for item in models_data]
    mae_stds = [item[3] for item in models_data]
    # RMSE 误差棒：std(RMSE) ≈ std(MSE) / (2*sqrt(MSE))
    rmse_stds = [0.5 * item[4] / np.sqrt(item[2]) if item[2] > 1e-12 else 0 for item in models_data]

    # 配色：Transformer 蓝、PLGAFormer 红、w/o 绿
    colors = []
    for name in model_names:
        if name == 'PLGAFormer (proposed)':
            colors.append('#d62728')
        elif name == 'Transformer (baseline)':
            colors.append('#1f77b4')
        else:
            colors.append('#2ca02c')

    baseline_idx = model_names.index('Transformer (baseline)')  # 用于计算改善%
    set_publication_style()

    def plot_single_bars(ax, values, stds, xlabel):
        y_pos = np.arange(len(model_names))
        ax.barh(y_pos, values, color=colors, alpha=0.9, xerr=stds,
                error_kw={'elinewidth': 1.5, 'capsize': 4, 'capthick': 1.5, 'ecolor': '#333333'})
        baseline_val = values[baseline_idx]
        for i, (val, std) in enumerate(zip(values, stds)):
            if i != baseline_idx and baseline_val > 0:
                improvement = (baseline_val - val) / baseline_val * 100
                lbl = f'{improvement:+.1f}%'
                tx = val + std + max(values) * 0.03
                ax.text(tx, i, lbl, ha='left', va='center', fontsize=10,
                        color='#2ca02c' if improvement > 0 else '#666666', fontweight='bold' if improvement > 20 else 'normal')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        x_max = max(v + (s or 0) for v, s in zip(values, stds)) * 1.2
        ax.set_xlim(0, max(x_max, 0.01))

    os.makedirs(save_dir, exist_ok=True)

    # 图1: MAE 单独
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    plot_single_bars(ax1, mae_values, mae_stds, 'MAE (scaled)')
    fig1.text(0.99, 0.01, f'Averaged over {num_runs} runs', ha='right', va='bottom', fontsize=9, style='italic')
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        path = os.path.join(save_dir, f"ablation_mae.{fmt}")
        plt.savefig(path, format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()
    print(f"   ✓ ablation_mae.pdf / ablation_mae.png")

    # 图2: RMSE 单独
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_single_bars(ax2, rmse_values, rmse_stds, 'RMSE (scaled)')
    fig2.text(0.99, 0.01, f'Averaged over {num_runs} runs', ha='right', va='bottom', fontsize=9, style='italic')
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        path = os.path.join(save_dir, f"ablation_rmse.{fmt}")
        plt.savefig(path, format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()
    print(f"   ✓ ablation_rmse.pdf / ablation_rmse.png")


def main():
    print("\n" + "=" * 70)
    print("PLGAFormer 消融实验 - 独立可视化")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    json_path = os.path.join(results_dir, RESULTS_JSON)

    if not os.path.exists(json_path):
        print(f"\n❌ 未找到结果文件: {json_path}")
        print("   请先运行: python experiments/exp2_ablation/ablation_study.py")
        return

    stats, num_runs = load_ablation_results(json_path)
    print(f"\n📂 加载: {RESULTS_JSON} ({num_runs} 次运行)")
    print(f"   模型: {[d[1] for d in ABLATION_MODEL_ORDER]}")

    print("\n📊 生成消融柱状图...")
    plot_ablation_bars(stats, num_runs, results_dir)

    print("\n" + "=" * 70)
    print("✅ 可视化完成")
    print(f"   保存目录: {results_dir}")
    print("=" * 70)

    plt.rcParams.update(plt.rcParamsDefault)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
