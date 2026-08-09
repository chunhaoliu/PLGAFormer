"""
独立可视化脚本 - PLGAFormer 物理一致性分析（实验 4）

功能（参考 exp3 的可视化风格，面向顶刊水平）:
    1. 从 results/physics_consistency_results.json 加载物理一致性评估结果
    2. 生成对比图 (Transformer vs PLGAFormer)：
        - 约束违反率折线图 (height / velocity / acceleration)
        - 轨迹平滑度误差柱状图 (position / velocity smoothness ratio)
        - 物理一致性综合雷达图
    3. 与实验 1/2/3 使用统一的出版级绘图配置

使用方法:
    python experiments/exp4_physics_consistency/visualize_physics_consistency.py
"""

import os
import sys
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 项目根目录（与其它实验保持一致）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_JSON = "physics_consistency_results.json"


def set_publication_style():
    """顶刊级别绘图样式（与 exp1/exp2/exp3 一致）"""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "black",
            "legend.fancybox": False,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "grid.linewidth": 1.0,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def load_physics_results(json_path):
    """从 physics_consistency_results.json 加载结果"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容两种格式：
    # 1) 旧格式：顶层直接是 results 字典
    # 2) 新格式：{"experiment": "...", "config": {...}, "results": {...}}
    if isinstance(data, dict) and "results" in data and isinstance(data.get("results"), dict):
        return data["results"]
    return data


def _get_metric_for_models(results, key_path):
    """
    按模型顺序提取指定 metric 序列。

    key_path 形如 ('violations', 'height_violation_rate') 或 ('smoothness', 'position_smoothness_ratio')
    返回:
        model_names: [str]
        values: [float]
    """
    models = sorted(results.keys())
    vals = []
    for m in models:
        cur = results[m]
        v = cur
        for k in key_path:
            v = v.get(k, np.nan)
            if v is None:
                v = np.nan
                break
        vals.append(float(v))
    return models, vals


def plot_constraint_violations(results, save_dir):
    """约束违反率折线 / 点图：height / velocity / acceleration，横轴为约束类型，纵轴为违反率 (%)"""
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    model_names = sorted(results.keys())
    if not model_names:
        return

    # 三个约束类型
    violation_keys = [
        ("violations", "height_violation_rate"),
        ("violations", "velocity_violation_rate"),
        ("violations", "acceleration_violation_rate"),
    ]
    violation_labels = ["Height constraint", "Velocity constraint", "Acceleration constraint"]
    x = np.arange(len(violation_keys))

    # 颜色/标记映射（与其它实验保持一致）
    model_colors = {
        "Transformer (baseline)": "#1f77b4",
        "PLGAFormer (proposed)": "#c44e52",
    }
    model_markers = {
        "Transformer (baseline)": "o",
        "PLGAFormer (proposed)": "s",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for m in model_names:
        vals = []
        for kp in violation_keys:
            v = results[m].get(kp[0], {}).get(kp[1], np.nan)
            vals.append(float(v) * 100.0 if np.isfinite(v) else np.nan)
        ax.plot(
            x,
            vals,
            marker=model_markers.get(m, "o"),
            color=model_colors.get(m, "#7f7f7f"),
            linewidth=2.5,
            markersize=8.0,
            markeredgecolor="white",
            markeredgewidth=1.1,
            label=m.replace("(baseline)", "").replace("(proposed)", "").strip(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(violation_labels, fontsize=11)
    ax.set_ylabel("Violation rate (%)", fontsize=13)
    ax.set_xlabel("Constraint type", fontsize=13)
    ax.set_title("(a) Constraint violation rates", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(-0.2, len(x) - 0.8)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        path = os.path.join(save_dir, f"physics_constraint_violations.{fmt}")
        plt.savefig(path, format=fmt, bbox_inches="tight", dpi=300 if fmt == "png" else None)
    plt.close()
    print("   ✓ physics_constraint_violations.pdf/png")


def plot_smoothness_error(results, save_dir):
    """轨迹平滑度误差柱状图：position / velocity smoothness ratio (Pred / True)"""
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    model_names = sorted(results.keys())
    if not model_names:
        return

    smooth_keys = [
        ("smoothness", "position_smoothness_ratio"),
        ("smoothness", "velocity_smoothness_ratio"),
    ]
    smooth_labels = ["Position smoothness", "Velocity smoothness"]
    x = np.arange(len(smooth_keys))
    width = 0.22

    model_colors = {
        "Transformer (baseline)": "#1f77b4",
        "PLGAFormer (proposed)": "#c44e52",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for i, m in enumerate(model_names):
        vals = []
        for kp in smooth_keys:
            v = results[m].get(kp[0], {}).get(kp[1], np.nan)
            vals.append(float(v) if np.isfinite(v) else np.nan)
        offset = (i - (len(model_names) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=m.replace("(baseline)", "").replace("(proposed)", "").strip(),
            color=model_colors.get(m, "#7f7f7f"),
            edgecolor="black",
            linewidth=1.0,
            alpha=0.9,
        )
        # 数值标签
        for bar, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(smooth_labels, fontsize=11)
    ax.set_ylabel("Smoothness ratio (Pred / True)", fontsize=13)
    ax.set_xlabel("Smoothness metric", fontsize=13)
    ax.set_title("(b) Trajectory smoothness error", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        path = os.path.join(save_dir, f"physics_smoothness_error.{fmt}")
        plt.savefig(path, format=fmt, bbox_inches="tight", dpi=300 if fmt == "png" else None)
    plt.close()
    print("   ✓ physics_smoothness_error.pdf/png")


def main():
    print("=" * 80)
    print("可视化：实验 4 - 物理一致性分析")
    print("=" * 80)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    json_path = os.path.join(results_dir, RESULTS_JSON)

    if not os.path.exists(json_path):
        print(f"❌ 未找到结果文件: {json_path}")
        print("请先运行 physics_consistency.py 生成 physics_consistency_results.json")
        return

    print(f"📂 加载结果: {json_path}")
    results = load_physics_results(json_path)

    print("生成物理一致性图像（Transformer vs PLGAFormer）...")
    plot_constraint_violations(results, results_dir)
    plot_smoothness_error(results, results_dir)

    print("\n✅ 物理一致性可视化完成！")


if __name__ == "__main__":
    main()

