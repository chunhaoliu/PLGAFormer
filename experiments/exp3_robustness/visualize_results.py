"""
独立可视化脚本 - 基于已保存的鲁棒性分析结果快速生成顶刊级别图表
仅展示 PLGAFormer (proposed) 在噪声/缺失/输入长度下的鲁棒性，指标为标准化空间 MSE/MAE

使用方法:
    python experiments/exp3_robustness/visualize_results.py

功能:
    1. 从 results/robustness_results.json 加载结果
    2. 生成综合 2×3 子图（MSE 行 + MAE 行）及三张分图（noise / missing / input_length）
    3. 纵轴标注 MSE (scaled) / MAE (scaled)，与主表一致

作者: PLGAFormer Team
日期: 2024
"""

import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 项目根目录（与 exp1/exp2 一致）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from utils.console import ensure_utf8_console

RESULTS_JSON = "robustness_results.json"

# 仅 PLGAFormer 时使用的配色（顶刊风格）
PLGAFORMER_COLOR = '#c44e52'
PLGAFORMER_MARKER = 'o'


def configure_console_for_unicode():
    """Allow Unicode status output on Windows GBK consoles."""
    ensure_utf8_console()


def set_publication_style():
    """顶刊级别绘图样式（与 exp1/exp2 一致）"""
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
        'lines.markersize': 9,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'grid.linewidth': 1.0,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'axes.spines.top': True,
        'axes.spines.right': True,
    })


def load_robustness_results(json_path):
    """从 robustness_results.json 加载"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    config = data.get('config', {})
    results = data.get('results', {})
    return config, results


def _get_series(data, key_sub, levels, key_name='mse'):
    """从 data[key_sub] 按 levels 顺序取出 key_name 序列，键可能是 str 或 number"""
    out = []
    for v in levels:
        key = str(v) if str(v) in data.get(key_sub, {}) else v
        if key in data.get(key_sub, {}):
            out.append(data[key_sub][key].get(key_name, float('nan')))
        else:
            out.append(float('nan'))
    return out


def plot_robustness_overview(results, config, save_dir):
    """一张图 2×3：上行 MSE (scaled)，下行 MAE (scaled)；三列分别为噪声、缺失率、输入长度"""
    noise_levels = config.get('noise_levels', [0.0, 0.05, 0.10, 0.15, 0.20])
    missing_rates = config.get('missing_rates', [0.0, 0.05, 0.10, 0.15, 0.20])
    input_lengths = config.get('input_lengths', [16, 32, 48, 64])
    set_publication_style()

    # 取第一个模型（实验仅 PLGAFormer）
    if not results:
        return
    model_name = next(iter(results.keys()))
    data = results[model_name]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.14, wspace=0.28, hspace=0.32)

    labels = [
        '(a) Noise robustness',
        '(b) Missing data robustness',
        '(c) Input length robustness',
        '(d) Noise robustness',
        '(e) Missing data robustness',
        '(f) Input length robustness',
    ]
    # Row 0: MSE
    if 'noise' in data:
        mse_noise = _get_series(data, 'noise', noise_levels, 'mse')
        x = [n * 100 for n in noise_levels]
        axes[0, 0].plot(x, mse_noise, color=PLGAFORMER_COLOR, marker=PLGAFORMER_MARKER,
                        linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1.2)
        axes[0, 0].set_xlabel('Noise level (%)')
        axes[0, 0].set_ylabel('MSE (scaled)')
        axes[0, 0].set_xlim(-2, 22)
        axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].set_title(labels[0], fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.35, linestyle='--')

    if 'missing' in data:
        mse_miss = _get_series(data, 'missing', missing_rates, 'mse')
        x = [r * 100 for r in missing_rates]
        axes[0, 1].plot(x, mse_miss, color=PLGAFORMER_COLOR, marker=PLGAFORMER_MARKER,
                        linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1.2)
        axes[0, 1].set_xlabel('Missing rate (%)')
        axes[0, 1].set_ylabel('MSE (scaled)')
        axes[0, 1].set_xlim(-2, 22)
        axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].set_title(labels[1], fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.35, linestyle='--')

    if 'input_length' in data:
        mse_len = _get_series(data, 'input_length', input_lengths, 'mse')
        axes[0, 2].plot(input_lengths, mse_len, color=PLGAFORMER_COLOR, marker=PLGAFORMER_MARKER,
                        linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1.2)
        axes[0, 2].set_xlabel('Input length (steps)')
        axes[0, 2].set_ylabel('MSE (scaled)')
        axes[0, 2].set_xticks(input_lengths)
        axes[0, 2].set_ylim(bottom=0)
    axes[0, 2].set_title(labels[2], fontsize=13, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.35, linestyle='--')

    # Row 1: MAE
    if 'noise' in data:
        mae_noise = _get_series(data, 'noise', noise_levels, 'mae')
        x = [n * 100 for n in noise_levels]
        axes[1, 0].plot(x, mae_noise, color=PLGAFORMER_COLOR, marker=PLGAFORMER_MARKER,
                        linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1.2)
        axes[1, 0].set_xlabel('Noise level (%)')
        axes[1, 0].set_ylabel('MAE (scaled)')
        axes[1, 0].set_xlim(-2, 22)
        axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].set_title(labels[3], fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.35, linestyle='--')

    if 'missing' in data:
        mae_miss = _get_series(data, 'missing', missing_rates, 'mae')
        x = [r * 100 for r in missing_rates]
        axes[1, 1].plot(x, mae_miss, color=PLGAFORMER_COLOR, marker=PLGAFORMER_MARKER,
                        linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1.2)
        axes[1, 1].set_xlabel('Missing rate (%)')
        axes[1, 1].set_ylabel('MAE (scaled)')
        axes[1, 1].set_xlim(-2, 22)
        axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].set_title(labels[4], fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.35, linestyle='--')

    if 'input_length' in data:
        mae_len = _get_series(data, 'input_length', input_lengths, 'mae')
        axes[1, 2].plot(input_lengths, mae_len, color=PLGAFORMER_COLOR, marker=PLGAFORMER_MARKER,
                        linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1.2)
        axes[1, 2].set_xlabel('Input length (steps)')
        axes[1, 2].set_ylabel('MAE (scaled)')
        axes[1, 2].set_xticks(input_lengths)
        axes[1, 2].set_ylim(bottom=0)
    axes[1, 2].set_title(labels[5], fontsize=13, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.35, linestyle='--')

    fig.text(0.5, 0.02,
             'Metrics in scaled space, consistent with main results (Table 1).',
             ha='center', fontsize=11, style='italic')
    for fmt in ['pdf', 'png']:
        path = os.path.join(save_dir, f"robustness_overview.{fmt}")
        plt.savefig(path, format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()
    print("   ✓ robustness_overview.pdf/png")


def plot_noise_robustness(results, config, save_dir):
    """噪声鲁棒性：MSE (scaled) 单图"""
    noise_levels = config.get('noise_levels', [0.0, 0.05, 0.10, 0.15, 0.20])
    set_publication_style()
    if not results:
        return
    data = next(iter(results.values()))
    if 'noise' not in data:
        return
    mse_list = _get_series(data, 'noise', noise_levels, 'mse')
    x = [n * 100 for n in noise_levels]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x, mse_list, color=PLGAFORMER_COLOR, marker=PLGAFORMER_MARKER,
            linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1.2,
            label='PLGAFormer (proposed)')
    ax.set_xlabel('Noise level (%)', fontsize=13)
    ax.set_ylabel('MSE (scaled, consistent with Table 1)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.set_xlim(-2, 22)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        path = os.path.join(save_dir, f"noise_robustness.{fmt}")
        plt.savefig(path, format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()
    print("   ✓ noise_robustness.pdf/png")


def plot_missing_robustness(results, config, save_dir):
    """缺失率鲁棒性：MSE (scaled) 单图"""
    missing_rates = config.get('missing_rates', [0.0, 0.05, 0.10, 0.15, 0.20])
    set_publication_style()
    if not results:
        return
    data = next(iter(results.values()))
    if 'missing' not in data:
        return
    mse_list = _get_series(data, 'missing', missing_rates, 'mse')
    x = [r * 100 for r in missing_rates]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x, mse_list, color=PLGAFORMER_COLOR, marker=PLGAFORMER_MARKER,
            linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1.2,
            label='PLGAFormer (proposed)')
    ax.set_xlabel('Missing data rate (%)', fontsize=13)
    ax.set_ylabel('MSE (scaled, consistent with Table 1)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.set_xlim(-2, 22)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        path = os.path.join(save_dir, f"missing_data_robustness.{fmt}")
        plt.savefig(path, format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()
    print("   ✓ missing_data_robustness.pdf/png")


def plot_input_length_robustness(results, config, save_dir):
    """输入长度鲁棒性：MSE (scaled) 单图"""
    input_lengths = config.get('input_lengths', [16, 32, 48, 64])
    set_publication_style()
    if not results:
        return
    data = next(iter(results.values()))
    if 'input_length' not in data:
        return
    mse_list = _get_series(data, 'input_length', input_lengths, 'mse')
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(input_lengths, mse_list, color=PLGAFORMER_COLOR, marker=PLGAFORMER_MARKER,
            linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1.2,
            label='PLGAFormer (proposed)')
    ax.set_xlabel('Input sequence length (steps)', fontsize=13)
    ax.set_ylabel('MSE (scaled, consistent with Table 1)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.set_xticks(input_lengths)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        path = os.path.join(save_dir, f"input_length_robustness.{fmt}")
        plt.savefig(path, format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()
    print("   ✓ input_length_robustness.pdf/png")


def main():
    configure_console_for_unicode()
    print("\n" + "=" * 70)
    print("实验3 鲁棒性分析 - 独立可视化 (PLGAFormer vs 真实数据)")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    json_path = os.path.join(results_dir, RESULTS_JSON)

    if not os.path.exists(json_path):
        print(f"\n❌ 未找到结果文件: {json_path}")
        print("   请先运行: python experiments/exp3_robustness/robustness_analysis.py")
        return

    config, results = load_robustness_results(json_path)
    print(f"\n📂 加载: {RESULTS_JSON}")
    print(f"   模型: {list(results.keys())}")

    print("\n📊 生成鲁棒性图表 (MSE/MAE scaled)...")
    plot_robustness_overview(results, config, results_dir)
    plot_noise_robustness(results, config, results_dir)
    plot_missing_robustness(results, config, results_dir)
    plot_input_length_robustness(results, config, results_dir)

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
