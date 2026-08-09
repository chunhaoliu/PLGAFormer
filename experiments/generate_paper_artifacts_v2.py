#!/usr/bin/env python3
"""
Legacy / diagnostic Paper Artifact Generator (not formal-v3 paper-facing)
==========================================================================
Generates ALL publication-quality figures and tables:
  - Fig: Single maneuver examples with 3D trajectory and control history
  - Fig: Grouped bar chart (SOTA comparison, 3 horizons)
  - Fig: Ablation radar chart
  - Fig: Missing data robustness curves
  - Fig: Efficiency Pareto frontier
  - Fig: Layer ablation + physics sensitivity
  - Fig: Prediction trajectory visualization (3 maneuver types)
  - Fig: Model architecture overview
  - Table: SOTA comparison (best bold, second underline)
  - Table: Missing data MAPE matrix
  - Table: Ablation study with Δ%
  - Table: Efficiency comparison
  - Table: Physics consistency metrics
"""
import warnings; warnings.filterwarnings('ignore')
import csv
import numpy as np, json, os, sys
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                     'axes.titlesize': 11, 'axes.labelsize': 10,
                     'legend.fontsize': 7, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight'})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.data_paths import get_dataset_npz_path
from utils.trajectory_protocol import (
    DEFAULT_POINTS_PER_TRAJECTORY,
    DEFAULT_SAMPLING_INTERVAL_S,
    dataset_time_metadata,
    horizon_label,
)

ARTIFACTS_DIR = PROJECT_ROOT / 'experiments' / 'paper_artifacts_v2'
ARTIFACTS_DIR.mkdir(exist_ok=True)


def _load_active_dataset_time_metadata():
    dataset_path = get_dataset_npz_path(PROJECT_ROOT)
    if not dataset_path.exists():
        return {"sampling_interval_s": DEFAULT_SAMPLING_INTERVAL_S, "has_sampling_interval_s": False}
    loaded = np.load(dataset_path, allow_pickle=True)
    try:
        data = {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()
    metadata = dataset_time_metadata(data)
    metadata["has_sampling_interval_s"] = "sampling_interval_s" in data
    return metadata


def _export_figure(fig, stem, *, png_dpi=600, svg=False):
    fig.savefig(ARTIFACTS_DIR / f'{stem}.png', dpi=png_dpi)
    fig.savefig(ARTIFACTS_DIR / f'{stem}.pdf')
    if svg:
        fig.savefig(ARTIFACTS_DIR / f'{stem}.svg')

# ==================== Color Scheme ====================
COLORS = {
    'PLGAFormer': '#E31A1C', 'Transformer': '#1F78B4', 'iTransformer': '#FF7F00',
    'Informer': '#33A02C', 'Autoformer': '#6A3D9A', 'FEDformer': '#B15928',
    'PatchTST': '#FB9A99', 'PIT': '#CAB2D6', 'Kalman': '#7F7F7F',
}
MODEL_ORDER = ['PLGAFormer', 'iTransformer', 'PIT', 'FEDformer', 'PatchTST',
               'Autoformer', 'Informer', 'Transformer', 'Kalman']
MODEL_LABELS = {
    'PLGAFormer': 'PLGAFormer\n(Ours)', 'Transformer': 'Transformer\n(Baseline)',
    'iTransformer': 'iTransformer', 'Informer': 'Informer',
    'Autoformer': 'Autoformer', 'FEDformer': 'FEDformer',
    'PatchTST': 'PatchTST', 'PIT': 'PIT', 'Kalman': 'Kalman',
}
REQUESTED_HORIZONS = [32, 64, 128, 256]
HORIZONS = [32, 64, 128]
DATASET_TIME_METADATA = _load_active_dataset_time_metadata()
SAMPLING_INTERVAL_S = float(DATASET_TIME_METADATA.get("sampling_interval_s", DEFAULT_SAMPLING_INTERVAL_S))
HORIZON_LABELS = {
    h: horizon_label(h, SAMPLING_INTERVAL_S, compact=True)
    if DATASET_TIME_METADATA.get("has_sampling_interval_s", False)
    else f"{h} steps"
    for h in REQUESTED_HORIZONS
}
SOTA_PHYSICAL = {}
SOTA_SOURCE = 'seeded_defaults'


def _dataset_timing_summary():
    if DATASET_TIME_METADATA.get("has_sampling_interval_s", False):
        return (
            f'Dataset timing: sampling_interval_s={SAMPLING_INTERVAL_S:g}, '
            f'input={DATASET_TIME_METADATA.get("seq_len", "--")} steps '
            f'({DATASET_TIME_METADATA.get("input_duration_s", "--"):g} s), '
            f'prediction={DATASET_TIME_METADATA.get("pred_len", "--")} steps '
            f'({DATASET_TIME_METADATA.get("prediction_duration_s", "--"):g} s)'
        )
    return (
        f'Dataset timing: sampling metadata missing; '
        f'input={DATASET_TIME_METADATA.get("seq_len", "--")} steps, '
        f'prediction={DATASET_TIME_METADATA.get("pred_len", "--")} steps. '
        f'Regenerate data with current code to populate physical seconds.'
    )

# ==================== Paper Data (from Exp1 artifacts) ====================
SOTA_MSE = {
    'PLGAFormer':  {32: (0.000114, 0.000008), 64: (0.000136, 0.000005), 128: (0.000172, 0.000008)},
    'Transformer': {32: (0.000503, 0.000051), 64: (0.000555, 0.000058), 128: (0.000628, 0.000041)},
    'iTransformer':{32: (0.000228, 0.000028), 64: (0.000243, 0.000024), 128: (0.000286, 0.000028)},
    'Informer':    {32: (0.129599, 0.009270), 64: (0.136265, 0.011206), 128: (0.126949, 0.003643)},
    'Autoformer':  {32: (0.016851, 0.001031), 64: (0.022524, 0.000929), 128: (0.028574, 0.001691)},
    'FEDformer':   {32: (0.001057, 0.000039), 64: (0.001183, 0.000026), 128: (0.001367, 0.000067)},
    'PatchTST':    {32: (0.003600, 0.000552), 64: (0.004571, 0.000901), 128: (0.005869, 0.001002)},
    'PIT':         {32: (0.000476, 0.000059), 64: (0.000478, 0.000040), 128: (0.000495, 0.000031)},
}
SOTA_MAE = {
    'PLGAFormer':  {32: 0.0082, 64: 0.0091, 128: 0.0103},
    'Transformer': {32: 0.0175, 64: 0.0183, 128: 0.0195},
    'iTransformer':{32: 0.0113, 64: 0.0118, 128: 0.0128},
}


def _mean_std(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _load_latest_formal_sota_metrics():
    """Aggregate one latest formal run per (model_key, seed)."""
    csv_path = PROJECT_ROOT / 'experiments' / 'exp1_sota' / 'results' / 'formal' / 'formal_sota_runs.csv'
    if not csv_path.exists():
        return None

    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))

    latest = {}
    for row in rows:
        key = (row.get('model_key', ''), row.get('seed', ''))
        timestamp = row.get('timestamp', '')
        if key not in latest or timestamp > latest[key]['timestamp']:
            latest[key] = {'timestamp': timestamp, 'run_id': row.get('run_id', '')}

    selected_run_ids = {item['run_id'] for item in latest.values()}
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    model_names = {}
    for row in rows:
        if row.get('run_id') not in selected_run_ids:
            continue
        try:
            horizon = int(row.get('horizon', 0))
            value = float(row.get('value', 'nan'))
        except ValueError:
            continue
        model = row.get('model', '').replace(' (proposed)', '')
        if model == 'Transformer (baseline)':
            model = 'Transformer'
        if model == 'PLGAFormer':
            model = 'PLGAFormer'
        metric = row.get('metric', '')
        values[model][horizon][metric].append(value)
        model_names[row.get('model_key', model)] = model

    if not values:
        return None

    loaded_mse = {}
    loaded_mae = {}
    loaded_physical = {}
    for model, by_horizon in values.items():
        loaded_mse[model] = {}
        loaded_mae[model] = {}
        loaded_physical[model] = {}
        for horizon, by_metric in by_horizon.items():
            mse_stats = _mean_std(by_metric.get('mse', []))
            mae_stats = _mean_std(by_metric.get('mae', []))
            if mse_stats:
                loaded_mse[model][horizon] = mse_stats
            if mae_stats:
                loaded_mae[model][horizon] = mae_stats[0]
            loaded_physical[model][horizon] = {
                'rmse_cart_m': _mean_std(by_metric.get('rmse_cart_m', [])),
                'mse_cart_m2': _mean_std(by_metric.get('mse_cart_m2', [])),
                'mae_cart_m': _mean_std(by_metric.get('mae_cart_m', [])),
                'fde': _mean_std(by_metric.get('fde', [])),
                'ade': _mean_std(by_metric.get('ade', [])),
            }

    available_horizons = sorted({
        h for by_horizon in loaded_mse.values() for h in by_horizon
        if h in REQUESTED_HORIZONS
    })
    return loaded_mse, loaded_mae, loaded_physical, available_horizons


def _normalise_model_name(name):
    name = (name or '').replace(' (proposed)', '').replace(' (baseline)', '')
    if name == 'PLGAFormer':
        return 'PLGAFormer'
    if name == 'Transformer':
        return 'Transformer'
    return name


def _load_current_exp1_sota_metrics():
    """Load the latest aggregate Exp1 CSV, including 256-step reruns when present."""
    csv_path = PROJECT_ROOT / 'experiments' / 'exp1_sota' / 'results' / 'baseline_results.csv'
    if not csv_path.exists():
        return None

    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    loaded_mse = {}
    loaded_mae = {}
    loaded_physical = {}
    available_horizons = set()
    for row in rows:
        model = _normalise_model_name(row.get('Model', ''))
        if not model:
            continue
        loaded_mse[model] = {}
        loaded_mae[model] = {}
        loaded_physical[model] = {}
        for h in REQUESTED_HORIZONS:
            try:
                mse = float(row.get(f'mse_{h}', 'nan'))
                mse_std = float(row.get(f'mse_{h}_std', 'nan'))
            except ValueError:
                mse = np.nan
                mse_std = np.nan
            try:
                mae = float(row.get(f'mae_{h}', 'nan'))
            except ValueError:
                mae = np.nan
            if np.isfinite(mse) and mse != 0:
                loaded_mse[model][h] = (mse, 0.0 if not np.isfinite(mse_std) else mse_std)
                available_horizons.add(h)
            if np.isfinite(mae) and mae != 0:
                loaded_mae[model][h] = mae

            metric_pack = {}
            for metric in ['rmse_cart_m', 'mae_cart_m', 'fde', 'ade']:
                try:
                    mean = float(row.get(f'{metric}_{h}', 'nan'))
                    std = float(row.get(f'{metric}_{h}_std', 'nan'))
                except ValueError:
                    mean = np.nan
                    std = np.nan
                metric_pack[metric] = (
                    (mean, 0.0 if not np.isfinite(std) else std)
                    if np.isfinite(mean) and mean != 0
                    else None
                )
            loaded_physical[model][h] = metric_pack

    if not available_horizons:
        return None
    return loaded_mse, loaded_mae, loaded_physical, sorted(available_horizons)


_formal_sota = _load_latest_formal_sota_metrics()
if _formal_sota:
    SOTA_MSE, SOTA_MAE, SOTA_PHYSICAL, HORIZONS = _formal_sota
    SOTA_SOURCE = 'formal_history'

_current_sota = _load_current_exp1_sota_metrics()
if _current_sota and 256 in _current_sota[3]:
    SOTA_MSE, SOTA_MAE, SOTA_PHYSICAL, HORIZONS = _current_sota
    SOTA_SOURCE = 'current_exp1_baseline_results'

ABLATION_DATA = {
    'Full (A+B+C)':     {'MSE_32': 0.000113, 'MSE_64': 0.000143, 'Δ%_32': 0, 'Δ%_64': 0},
    'w/o A (B+C)':      {'MSE_32': 0.000123, 'MSE_64': 0.000152, 'Δ%_32': 8.6, 'Δ%_64': 6.4},
    'w/o B (A+C)':      {'MSE_32': 0.000139, 'MSE_64': 0.000158, 'Δ%_32': 22.4, 'Δ%_64': 10.9},
    'w/o C (A+B)':      {'MSE_32': 0.000122, 'MSE_64': 0.000145, 'Δ%_32': 7.6, 'Δ%_64': 1.3},
    'Baseline (none)':  {'MSE_32': 0.000459, 'MSE_64': 0.000476, 'Δ%_32': 305.7, 'Δ%_64': 233.7},
}


# ==================== FIGURES ====================

def fig_sota_grouped_bar():
    """Grouped bar chart: all models at 3 horizons (MSE + MAE)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    models = [m for m in MODEL_ORDER if m in SOTA_MSE]
    x = np.arange(len(models))
    width = min(0.8 / max(len(HORIZONS), 1), 0.25)
    horizon_colors = ['#4C78A8', '#F58518', '#54A24B', '#B279A2']

    for col, (metric_dict, metric_name) in enumerate([(SOTA_MSE, 'MSE'), (SOTA_MAE, 'MAE')]):
        ax = axes[col]
        for i, h in enumerate(HORIZONS):
            offset = (i - (len(HORIZONS) - 1) / 2) * width
            vals = []
            errs = []
            for m in models:
                if metric_name == 'MSE':
                    vals.append(metric_dict.get(m, {}).get(h, (np.nan, 0))[0])
                    errs.append(metric_dict.get(m, {}).get(h, (0, 0))[1])
                else:
                    vals.append(metric_dict.get(m, {}).get(h, np.nan))
                    errs.append(0)
            bars = ax.bar(x + offset, vals, width, label=f'{HORIZON_LABELS[h]}',
                          color=horizon_colors[i % len(horizon_colors)],
                          alpha=0.85, edgecolor='black', linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=20, ha='right', fontsize=7)
        ax.set_ylabel(metric_name)
        title = f'{metric_name} by Prediction Horizon'
        if metric_name == 'MSE':
            ax.set_yscale('log')
            title += ' (log scale)'
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'SOTA Model Comparison: Grouped by Prediction Horizon\nSource: {SOTA_SOURCE}',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / 'fig_sota_grouped_bar.png')
    fig.savefig(ARTIFACTS_DIR / 'fig_sota_grouped_bar.pdf')
    plt.close(fig)
    print('  Fig: SOTA grouped bar chart')

def fig_ablation_radar():
    """Radar chart showing multi-dimensional ablation impact."""
    categories = ['MSE@32 step', 'MSE@64 step', 'MSE@128 step', 'Params', 'Training Speed', 'Physics Consistency']
    variants = ['Full (A+B+C)', 'w/o A', 'w/o B', 'w/o C']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, variant in enumerate(variants):
        if variant == 'Full (A+B+C)':
            values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        elif variant == 'w/o A':
            values = [0.92, 0.94, 0.93, 1.0, 1.05, 0.85]
        elif variant == 'w/o B':
            values = [0.82, 0.90, 0.88, 1.0, 0.95, 0.60]
        else:
            values = [0.93, 0.99, 0.97, 1.0, 1.02, 0.90]
        values += values[:1]
        ax.fill(angles, values, alpha=0.15, color=[COLORS['PLGAFormer'], '#FF7F00', '#1F78B4', '#33A02C'][i])
        ax.plot(angles, values, 'o-', linewidth=2, label=variant,
                color=[COLORS['PLGAFormer'], '#FF7F00', '#1F78B4', '#33A02C'][i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title('Ablation Study: Multi-Dimensional Impact\n(Full model = 1.0 baseline)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / 'fig_ablation_radar.png')
    fig.savefig(ARTIFACTS_DIR / 'fig_ablation_radar.pdf')
    plt.close(fig)
    print('  Fig: Ablation radar chart')

def fig_architecture_overview():
    """PLGAFormer architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7); ax.axis('off')

    blocks = [
        (1, 5, 2.2, 1.3, 'Input\n[r,λ,φ,V,γ,ψ]\nB×L×6', '#E8EAF6'),
        (3.5, 5, 2.2, 1.3, 'Positional\nEncoding', '#C5CAE9'),
        (6.5, 5.5, 3.0, 1.3, 'Physics-Aware Sparse Attention\n(A) Temporal Decay + Phase\nCoherence + Geometric Proximity', '#9FA8DA'),
        (10, 5.5, 2.5, 1.3, 'Encoder\n(L=3 layers)', '#7986CB'),
        (6.5, 3, 2.5, 1.3, 'Decoder\n(L=2 layers)', '#FFE082'),
        (9.5, 3, 2.5, 1.3, 'Multi-Head Trajectory\nDecoder (C)\nBase + Gate × Delta', '#FFCC80'),
        (6.5, 1, 2.5, 1.3, 'Gated Adaptive\nPhysics Corrector (B)\nSmoothness Gate + Prior', '#A5D6A7'),
        (3.5, 1, 2.2, 1.3, 'Output\n[r,λ,φ]\nB×L×3', '#C8E6C9'),
    ]
    for (x, y, w, h, label, color) in blocks:
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                     facecolor=color, edgecolor='#333', linewidth=1.5))
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5, fontweight='bold')

    # Arrows
    for (x1,y1,x2,y2) in [(2.1,5,2.4,5),(4.6,5,5.0,5),(8.0,5,8.8,5),(8.0,4.8,8.0,3.65),
                            (9.5,4.8,9.5,3.65),(9.5,3.65,7.75,3.65),(6.5,2.35,6.5,1.65),
                            (5.25,1.65,4.6,1.65),(8.0,1.65,4.6,1.65)]:
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

    ax.annotate('', xy=(6.5,4.2), xytext=(3.5,4.2),
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2, connectionstyle='arc3,rad=0.35'))
    ax.text(5.0, 4.5, 'Residual', color='#D32F2F', fontsize=8, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#E8EAF6', label='Input/Output'),
        mpatches.Patch(facecolor='#9FA8DA', label='Innovation A: Sparse Attention'),
        mpatches.Patch(facecolor='#FFE082', label='Core Transformer'),
        mpatches.Patch(facecolor='#FFCC80', label='Innovation C: Multi-Head Decoder'),
        mpatches.Patch(facecolor='#A5D6A7', label='Innovation B: Physics Corrector'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    ax.set_title('PLGAFormer Architecture Overview', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / 'fig_architecture.png')
    fig.savefig(ARTIFACTS_DIR / 'fig_architecture.pdf')
    plt.close(fig)
    print('  Fig: Architecture overview')


def fig_trajectory_prediction():
    """Trajectory prediction visualization for 3 maneuver types."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    np.random.seed(42)
    t_obs = np.arange(64)
    t_pred = np.arange(64, 192)

    for col, (maneuver, label) in enumerate([
        ('longitudinal', 'Longitudinal Glide'), ('turning', 'C-Shape Turn'), ('weaving', 'S-Shape Weave')
    ]):
        base_r = 6378 + 65 - col * 5
        r_gt = base_r + 2 * np.sin(np.linspace(0, np.pi * (1 + col * 0.5), 192))
        r_pred = r_gt + np.random.randn(192) * (0.3 + col * 0.5)

        for row, coord_name in enumerate(['r (km)', 'λ (rad)', 'φ (rad)']):
            ax = axes[row, col]
            ax.plot(t_obs, r_gt[:64], 'b-', linewidth=1.5, alpha=0.6, label='Observed')
            ax.plot(t_pred, r_gt[64:], 'k-', linewidth=2, label='Ground Truth')
            ax.plot(t_pred, r_pred[64:], 'r--', linewidth=1.5, label='PLGAFormer')
            ax.axvline(x=64, color='gray', linestyle='--', alpha=0.4)
            ax.set_xlabel('Time Step'); ax.set_ylabel(coord_name)
            ax.legend(fontsize=6); ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(label, fontweight='bold')

    fig.suptitle('PLGAFormer Trajectory Predictions: 3 Maneuver Types\n(Blue=Observation, Black=Ground Truth, Red=PLGAFormer)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / 'fig_trajectory_prediction.png')
    fig.savefig(ARTIFACTS_DIR / 'fig_trajectory_prediction.pdf')
    plt.close(fig)
    print('  Fig: Trajectory prediction visualization')


def fig_architecture_overview():
    """Clean publication architecture diagram with ASCII labels."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    ax.set_xlim(-0.35, 12.5)
    ax.set_ylim(0, 7)
    ax.axis('off')

    blocks = [
        (1.1, 5.0, 2.25, 1.25, 'Input\n[r, lon, lat, V, gamma, psi]\nB x L x 6', '#E8EAF6'),
        (3.6, 5.0, 2.1, 1.25, 'Temporal\nEmbedding', '#C5CAE9'),
        (6.5, 5.35, 3.1, 1.35, 'Physics-Aware Sparse Attention\nA: decay + phase + geometry', '#9FA8DA'),
        (10.0, 5.35, 2.5, 1.25, 'Encoder Stack\n3 layers', '#7986CB'),
        (6.5, 3.0, 2.45, 1.25, 'Decoder Stack\n2 layers', '#FFE082'),
        (9.55, 3.0, 2.55, 1.25, 'Multi-Head Trajectory Decoder\nC: base + gated residual', '#FFCC80'),
        (6.5, 1.05, 2.55, 1.25, 'Adaptive Physics Corrector\nB: smoothness + prior gate', '#A5D6A7'),
        (3.55, 1.05, 2.15, 1.25, 'Output\n[r, lon, lat]\nB x H x 3', '#C8E6C9'),
    ]
    for (x, y, w, h, label, color) in blocks:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.08',
                                    facecolor=color, edgecolor='#333333', linewidth=1.3))
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5, fontweight='bold')

    arrows = [
        (2.25, 5.0, 2.55, 5.0), (4.65, 5.0, 4.95, 5.15), (8.05, 5.35, 8.75, 5.35),
        (10.0, 4.7, 10.0, 3.65), (9.1, 3.0, 7.75, 3.0), (6.5, 2.35, 6.5, 1.72),
        (5.25, 1.05, 4.65, 1.05),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#4D4D4D', lw=1.5))
    ax.annotate('', xy=(6.1, 4.25), xytext=(3.55, 4.25),
                arrowprops=dict(arrowstyle='->', color='#B2182B', lw=2.0, connectionstyle='arc3,rad=0.28'))
    ax.text(4.75, 4.55, 'residual path', color='#B2182B', fontsize=8, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#9FA8DA', label='A: physics-aware attention'),
        mpatches.Patch(facecolor='#A5D6A7', label='B: physics corrector'),
        mpatches.Patch(facecolor='#FFCC80', label='C: trajectory decoder'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    ax.set_title('PLGAFormer Architecture and Evidence-Carrying Modules', fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / 'fig_architecture.png')
    fig.savefig(ARTIFACTS_DIR / 'fig_architecture.pdf')
    plt.close(fig)
    print('  Fig: Clean architecture overview')


def _spherical_to_cartesian_np(spherical_data):
    r = spherical_data[..., 0]
    lon = spherical_data[..., 1]
    lat = spherical_data[..., 2]
    cos_lat = np.cos(lat)
    x = r * cos_lat * np.cos(lon)
    y = r * cos_lat * np.sin(lon)
    z = r * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


MANEUVER_EXAMPLE_STYLE = {
    'longitudinal': {
        'label': 'Longitudinal-only maneuver',
        'short_label': 'Longitudinal',
        'accent': '#2F6F9F',
        'filename': 'fig_longitudinal_maneuver_control',
        'view': (24, -58),
        'init': {'h': 68000.0, 'lambda': -0.045, 'phi': -0.030, 'V': 6100.0, 'gamma': -0.022, 'psi': 0.020},
    },
    'turning': {
        'label': 'Lateral turning maneuver',
        'short_label': 'Turning',
        'accent': '#B35C44',
        'filename': 'fig_turning_maneuver_control',
        'view': (24, -62),
        'init': {'h': 66000.0, 'lambda': -0.020, 'phi': 0.000, 'V': 6300.0, 'gamma': -0.025, 'psi': 0.030},
    },
    'weaving': {
        'label': 'Lateral weaving maneuver',
        'short_label': 'Weaving',
        'accent': '#4E8B54',
        'filename': 'fig_weaving_maneuver_control',
        'view': (24, -66),
        'init': {'h': 66000.0, 'lambda': 0.015, 'phi': -0.020, 'V': 6300.0, 'gamma': -0.025, 'psi': -0.030},
    },
}

def _simulate_maneuver_examples():
    from data_generation.data_generator import DCBNN_HGV_Simulator

    sim = DCBNN_HGV_Simulator()
    sim.sampling_interval_s = DEFAULT_SAMPLING_INTERVAL_S
    sim.points_per_trajectory = DEFAULT_POINTS_PER_TRAJECTORY
    duration_s = (DEFAULT_POINTS_PER_TRAJECTORY - 1) * DEFAULT_SAMPLING_INTERVAL_S
    examples = {}
    for maneuver, cfg in MANEUVER_EXAMPLE_STYLE.items():
        t, trajectory = sim.simulate_trajectory(
            cfg['init'],
            maneuver,
            duration=duration_s,
            sampling_interval_s=DEFAULT_SAMPLING_INTERVAL_S,
            num_points=DEFAULT_POINTS_PER_TRAJECTORY,
        )
        if t is None or trajectory is None or len(t) != DEFAULT_POINTS_PER_TRAJECTORY:
            raise RuntimeError(f'Could not simulate the {maneuver} example trajectory.')

        controls = []
        ld_ratio = []
        for ti, row in zip(t, trajectory):
            altitude_m = float(row[0] - sim.R_earth)
            alpha, bank = sim.control_inputs(float(ti), maneuver, float(row[3]), altitude_m)
            rho, speed_of_sound = sim.atmospheric_model(altitude_m)
            mach = float(row[3]) / speed_of_sound
            lift_coeff, drag_coeff, _ = sim.aerodynamic_coefficients(alpha, bank, mach)
            controls.append((alpha, bank))
            ld_ratio.append(lift_coeff / max(drag_coeff, 1e-8))
        controls = np.asarray(controls)
        examples[maneuver] = {
            't': t,
            'trajectory': trajectory,
            'alpha_deg': np.rad2deg(controls[:, 0]),
            'bank_deg': np.rad2deg(controls[:, 1]),
            'ld_ratio': np.asarray(ld_ratio),
            'relative_deg_km': _to_relative_spherical_deg_km(trajectory, sim.R_earth),
        }
    return examples


def _to_relative_spherical_deg_km(trajectory, earth_radius_m):
    lambda_deg = np.rad2deg(trajectory[:, 1] - trajectory[0, 1])
    phi_deg = np.rad2deg(trajectory[:, 2] - trajectory[0, 2])
    altitude_km = (trajectory[:, 0] - earth_radius_m) / 1000.0
    return np.column_stack([lambda_deg, phi_deg, altitude_km])


def fig_maneuver_control_examples():
    """Generate one single figure per maneuver, based on the original sample-figure structure."""
    examples = _simulate_maneuver_examples()
    with plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 6.8,
        'axes.titlesize': 7.8,
        'axes.labelsize': 7,
        'legend.fontsize': 5.8,
        'xtick.labelsize': 6.2,
        'ytick.labelsize': 6.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
    }):
        for maneuver, cfg in MANEUVER_EXAMPLE_STYLE.items():
            _draw_single_maneuver_control_figure(maneuver, cfg, examples[maneuver])
    print('  Fig: Single maneuver trajectory/control examples')


def _draw_single_maneuver_control_figure(maneuver, cfg, payload):
    fig = plt.figure(figsize=(7.6, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.42)
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_ctrl = fig.add_subplot(gs[0, 1])

    coords = payload['relative_deg_km']
    velocity = payload['trajectory'][:, 3]
    _plot_velocity_colored_trajectory(ax3d, coords, velocity, cfg)
    _plot_control_profile(ax_ctrl, payload)

    fig.suptitle(
        f'{cfg["short_label"]} maneuver: trajectory and control history',
        x=0.51,
        y=0.982,
        fontsize=8.0,
        fontweight='bold',
    )
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.035, right=0.955)
    _export_figure(fig, cfg['filename'], svg=True)
    plt.close(fig)


def _plot_velocity_colored_trajectory(ax, coords, velocity, cfg):
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(float(np.nanmin(velocity)), float(np.nanmax(velocity)))
    collection = Line3DCollection(segments, cmap='plasma', norm=norm, linewidth=2.0)
    collection.set_array(velocity[:-1])
    ax.add_collection3d(collection)

    z_floor = max(35.0, float(np.nanmin(z)) - 1.5)
    ax.plot(x, y, np.full_like(z, z_floor), color='#6C6C6C', lw=0.9, alpha=0.35, label='Ground track')
    ax.scatter(x[0], y[0], z[0], color='#2CA02C', s=30, marker='o', depthshade=False, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='#D62728', s=32, marker='s', depthshade=False, label='End')

    ax.set_xlim(_expanded_limits(x, min_span=0.8))
    ax.set_ylim(_expanded_limits(y, min_span=2.0))
    ax.set_zlim(z_floor, max(70.0, float(np.nanmax(z)) + 1.0))
    ax.set_xlabel(r'$\Delta\lambda$ (deg)', labelpad=-2)
    ax.set_ylabel(r'$\Delta\phi$ (deg)', labelpad=-2)
    ax.set_zlabel('h (km)', labelpad=-7)
    ax.set_title('3D trajectory', loc='left', pad=0, fontweight='bold')
    ax.view_init(elev=cfg['view'][0], azim=cfg['view'][1])
    ax.grid(alpha=0.25)
    ax.tick_params(axis='both', which='major', pad=0, labelsize=5.8)
    ax.zaxis.set_tick_params(pad=0, labelsize=5.8)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=5.4, frameon=True, borderpad=0.22)
    cbar = fig_colorbar(ax.figure, collection, ax)
    cbar.set_label('V (m/s)', fontsize=6.2, labelpad=3)
    cbar.ax.tick_params(labelsize=5.8, length=2)


def fig_colorbar(fig, mappable, ax):
    return fig.colorbar(mappable, ax=ax, fraction=0.034, pad=0.030)


def _expanded_limits(values, min_span):
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = max(vmax - vmin, min_span)
    center = 0.5 * (vmin + vmax)
    pad = 0.08 * span
    return center - 0.5 * span - pad, center + 0.5 * span + pad


def _plot_control_profile(ax, payload):
    t = payload['t']
    alpha_color = '#E64B35'
    bank_color = '#2CA25F'
    ld_color = '#8E44AD'

    ax.plot(t, payload['alpha_deg'], color=alpha_color, lw=1.7, label=r'Attack angle $\alpha(t)$')
    ax.plot(t, payload['bank_deg'], color=bank_color, lw=1.6, ls='--', label=r'Bank angle $\sigma(t)$')
    ax.axhline(0, color='#9A9A9A', lw=0.55, alpha=0.55)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_xlim(0, DEFAULT_POINTS_PER_TRAJECTORY - 1)
    ax.set_ylim(-34, 34)
    ax.grid(alpha=0.23, lw=0.5)
    ax.set_title('Control parameters', loc='left', fontweight='bold')

    ax_ld = ax.twinx()
    ax_ld.plot(t, payload['ld_ratio'], color=ld_color, lw=1.25, ls=':', label='L/D ratio')
    ax_ld.set_ylabel('L/D ratio', color=ld_color)
    ax_ld.tick_params(axis='y', colors=ld_color)
    ax_ld.spines['right'].set_visible(True)
    ax_ld.spines['right'].set_color(ld_color)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax_ld.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc='upper right', fontsize=5.4, frameon=True, borderpad=0.22)


def _load_prediction_cache_for_figure():
    cache_path = PROJECT_ROOT / 'experiments' / 'exp1_sota' / 'results' / 'predictions_cache.npz'
    if not cache_path.exists():
        return None
    cache = np.load(cache_path, allow_pickle=True)
    y_scaled = cache['y_test']
    x_scaled = cache['X_test'] if 'X_test' in cache else None
    scaler_mean = cache['scaler_mean'].reshape(1, 1, -1)
    scaler_scale = cache['scaler_scale'].reshape(1, 1, -1)
    y_true = y_scaled * scaler_scale + scaler_mean
    x_obs = None
    if x_scaled is not None:
        x_obs = x_scaled[..., :3] * scaler_scale + scaler_mean
    predictions = cache['predictions'].item()
    pred_phys = {
        name: pred * scaler_scale + scaler_mean
        for name, pred in predictions.items()
        if pred.shape[1] == y_scaled.shape[1]
    }
    metadata = {
        'prediction_length': int(cache['prediction_length']) if 'prediction_length' in cache else int(y_scaled.shape[1]),
        'seq_len': int(cache['seq_len']) if 'seq_len' in cache else int(x_scaled.shape[1]) if x_scaled is not None else 0,
    }
    return x_obs, y_true, pred_phys, metadata


def fig_trajectory_prediction():
    """AF-CILN-style qualitative case: 3D trajectory, zoom, and step error."""
    payload = _load_prediction_cache_for_figure()
    if payload is None:
        print('  Fig: Trajectory prediction skipped (missing predictions_cache.npz)')
        return

    x_obs, y_true, pred_phys, metadata = payload
    preferred = ['PLGAFormer (proposed)', 'iTransformer', 'PIT', 'Transformer (baseline)', 'PatchTST']
    models = [m for m in preferred if m in pred_phys]
    if 'PLGAFormer (proposed)' not in models:
        print('  Fig: Trajectory prediction skipped (missing PLGAFormer predictions)')
        return

    true_cart = _spherical_to_cartesian_np(y_true)
    obs_cart = _spherical_to_cartesian_np(x_obs) if x_obs is not None else None
    pred_cart = {m: _spherical_to_cartesian_np(pred_phys[m]) for m in models}
    plga_err = np.linalg.norm(pred_cart['PLGAFormer (proposed)'] - true_cart, axis=-1).mean(axis=1)
    sample_idx = int(np.argsort(plga_err)[len(plga_err) // 2])

    true_case = true_cart[sample_idx] / 1000.0
    obs_case = obs_cart[sample_idx] / 1000.0 if obs_cart is not None else None
    pred_case = {m: pred_cart[m][sample_idx] / 1000.0 for m in models}
    if DATASET_TIME_METADATA.get("has_sampling_interval_s", False):
        step = np.arange(true_case.shape[0]) * SAMPLING_INTERVAL_S
        step_label = 'Prediction time (s)'
    else:
        step = np.arange(true_case.shape[0])
        step_label = 'Prediction step'

    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.0, 1.0], height_ratios=[1, 1], wspace=0.32, hspace=0.34)
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_zoom = fig.add_subplot(gs[0, 1:])
    ax_err = fig.add_subplot(gs[1, 1:])

    style = {
        'PLGAFormer (proposed)': ('#D62728', '-', 2.4, 'PLGAFormer'),
        'iTransformer': ('#FF7F0E', '--', 1.6, 'iTransformer'),
        'PIT': ('#9467BD', '-.', 1.6, 'PIT'),
        'Transformer (baseline)': ('#1F77B4', ':', 1.7, 'Transformer'),
        'PatchTST': ('#8C564B', (0, (3, 1, 1, 1)), 1.4, 'PatchTST'),
    }

    if obs_case is not None:
        ax3d.plot(obs_case[:, 0], obs_case[:, 1], obs_case[:, 2], color='#4C78A8', lw=2.0, label='Observed history')
    ax3d.plot(true_case[:, 0], true_case[:, 1], true_case[:, 2], color='black', lw=2.2, label='Future truth')
    for model in models:
        color, ls, lw, label = style.get(model, ('gray', '--', 1.3, model))
        case = pred_case[model]
        ax3d.plot(case[:, 0], case[:, 1], case[:, 2], color=color, ls=ls, lw=lw, label=label)
    ax3d.set_xlabel('x (km)')
    ax3d.set_ylabel('y (km)')
    ax3d.set_zlabel('z (km)')
    ax3d.set_title('3D case with observed prefix', loc='left', fontweight='bold')
    ax3d.view_init(elev=22, azim=-54)
    ax3d.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=7)

    zoom_slice = slice(max(0, true_case.shape[0] - 24), true_case.shape[0])
    ax_zoom.plot(true_case[zoom_slice, 0], true_case[zoom_slice, 1], color='black', lw=2.2, label='Ground truth')
    for model in models:
        color, ls, lw, label = style.get(model, ('gray', '--', 1.3, model))
        case = pred_case[model]
        ax_zoom.plot(case[zoom_slice, 0], case[zoom_slice, 1], color=color, ls=ls, lw=lw, label=label)
    ax_zoom.set_title('Terminal-window projection zoom', loc='left', fontweight='bold')
    ax_zoom.set_xlabel('x (km)')
    ax_zoom.set_ylabel('y (km)')
    ax_zoom.grid(alpha=0.25)

    for model in models:
        color, ls, lw, label = style.get(model, ('gray', '--', 1.3, model))
        err_m = np.linalg.norm(pred_cart[model][sample_idx] - true_cart[sample_idx], axis=-1)
        ax_err.plot(step, err_m / 1000.0, color=color, ls=ls, lw=lw, label=label)
    ax_err.set_title('Step-wise displacement error', loc='left', fontweight='bold')
    ax_err.set_xlabel(step_label)
    ax_err.set_ylabel('Error (km)')
    ax_err.grid(alpha=0.25)

    title = (
        f'Representative HGV Trajectory Prediction Case '
        f'(observed {metadata.get("seq_len", 0)} steps, forecast {metadata.get("prediction_length", true_case.shape[0])} steps)'
    )
    fig.suptitle(title, fontweight='bold', fontsize=12, y=0.985)
    ax3d.text2D(-0.12, 1.02, 'a', transform=ax3d.transAxes, fontweight='bold', fontsize=13)
    ax_zoom.text(-0.08, 1.08, 'b', transform=ax_zoom.transAxes, fontweight='bold', fontsize=13)
    ax_err.text(-0.08, 1.08, 'c', transform=ax_err.transAxes, fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(ARTIFACTS_DIR / 'fig_trajectory_prediction.png')
    fig.savefig(ARTIFACTS_DIR / 'fig_trajectory_prediction.pdf')
    plt.close(fig)
    print('  Fig: AF-CILN-style trajectory prediction case')


# ==================== TABLES ====================

def table_sota_comparison():
    """SOTA comparison table with best (bold) and second (underline)."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Quantitative comparison across varying prediction lengths (test set).}')
    lines.append(r'\label{tab:sota}')
    lines.append(r'\begin{tabular}{lccc|ccc|ccc}')
    lines.append(r'\toprule')
    lines.append(r'& \multicolumn{3}{c}{MSE ($\times 10^{-4}$)} & \multicolumn{3}{c}{MAE ($\times 10^{-2}$)} & \\')
    lines.append(r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}')
    lines.append(
        r'Model & '
        + ' & '.join(HORIZON_LABELS[h] for h in HORIZONS[:3])
        + r' & '
        + ' & '.join(HORIZON_LABELS[h] for h in HORIZONS[:3])
        + r' & Improvement \\'
    )
    lines.append(r'\midrule')

    models = [m for m in MODEL_ORDER if m in SOTA_MSE]
    for model in models:
        mse_vals = [f'{SOTA_MSE[model][h][0]*10000:.2f}±{SOTA_MSE[model][h][1]*10000:.2f}' for h in HORIZONS]
        mae_vals = [f'{SOTA_MAE.get(model,{}).get(h,0)*100:.2f}' for h in HORIZONS]
        improv = (1 - SOTA_MSE[model][128][0] / SOTA_MSE['Transformer'][128][0]) * 100

        if model == 'PLGAFormer':
            line = f'\\textbf{{{model}}} & '
            line += ' & '.join([f'\\textbf{{{v}}}' for v in mse_vals])
            line += ' & ' + ' & '.join(mae_vals)
            line += f' & \\textbf{{{improv:.1f}\\%}}'
        elif model == 'iTransformer':
            line = f'\\underline{{{model}}} & '
            line += ' & '.join([f'\\underline{{{v}}}' for v in mse_vals])
            line += ' & ' + ' & '.join(mae_vals)
            line += f' & {improv:.1f}\\%'
        else:
            line = f'{model} & ' + ' & '.join(mse_vals) + ' & ' + ' & '.join(mae_vals) + f' & {improv:.1f}\\%'
        lines.append(line + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def table_ablation():
    """Ablation study table with Δ%."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Ablation study: contribution of each PLGAFormer component.}')
    lines.append(r'\label{tab:ablation}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(
        r'Variant & MSE@'
        + HORIZON_LABELS.get(32, '32 step')
        + r' & MSE@'
        + HORIZON_LABELS.get(64, '64 step')
        + r' & $\Delta$\%@'
        + HORIZON_LABELS.get(32, '32 step')
        + r' & $\Delta$\%@'
        + HORIZON_LABELS.get(64, '64 step')
        + r' \\'
    )
    lines.append(r'\midrule')

    for variant in ['Full (A+B+C)', 'w/o A (B+C)', 'w/o B (A+C)', 'w/o C (A+B)', 'Baseline (none)']:
        d = ABLATION_DATA[variant]
        if variant.startswith('Full'):
            line = f'\\textbf{{{variant}}} & \\textbf{{{d["MSE_32"]:.6f}}} & \\textbf{{{d["MSE_64"]:.6f}}} & \\textbf{{--}} & \\textbf{{--}}'
        else:
            line = f'{variant} & {d["MSE_32"]:.6f} & {d["MSE_64"]:.6f} & +{d["Δ%_32"]:.1f}\\% & +{d["Δ%_64"]:.1f}\\%'
        lines.append(line + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def _safe_load_json(path):
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _missing_cell(results, model, scenario, metric='MSE'):
    key = f'{model}_{scenario}'
    value = results.get(key, {}).get('128s', {}).get(metric)
    return '--' if value is None else f'{float(value):.6g}'


def table_missing_data():
    """Missing-data table populated from Exp5 outputs when available."""
    results_path = PROJECT_ROOT / 'experiments' / 'exp5_missing_data' / 'results' / 'missing_data_results.json'
    metadata_path = PROJECT_ROOT / 'experiments' / 'exp5_missing_data' / 'results' / 'run_metadata.json'
    metadata = _safe_load_json(metadata_path) or {}
    is_formal = metadata.get('max_samples') is None and metadata.get('epochs', 0) >= 10
    results = (_safe_load_json(results_path) or {}) if is_formal else {}
    scenarios = ['R_5', 'R_10', 'R_20', 'R_40', 'C_5', 'C_10', 'C_20', 'C_40']
    models = ['PLGAFormer', 'Transformer', 'iTransformer']
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    caption_suffix = '' if results else ' Pending full Exp5 regeneration.'
    lines.append(r'\caption{MSE comparison under missing data (' + HORIZON_LABELS.get(128, '128 step') + r' prediction).' + caption_suffix + r'}')
    lines.append(r'\label{tab:missing}')
    lines.append(r'\begin{tabular}{l|cccc|cccc}')
    lines.append(r'\toprule')
    lines.append(r'& \multicolumn{4}{c}{Random Missing} & \multicolumn{4}{c}{Continuous Missing} \\')
    lines.append(r'Model & 5\% & 10\% & 20\% & 40\% & 5\% & 10\% & 20\% & 40\% \\')
    lines.append(r'\midrule')
    for model in models:
        cells = [_missing_cell(results, model, scenario, metric='MSE') for scenario in scenarios]
        lines.append(f'{model} & ' + ' & '.join(cells) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    if results:
        lines.append(r'% Source: experiments/exp5_missing_data/results/missing_data_results.json')
    else:
        lines.append(r'% NOTE: Pending full Exp5 run. Quick/smoke outputs are intentionally excluded from this manuscript table.')
    return '\n'.join(lines)


def _efficiency_rows():
    results_path = PROJECT_ROOT / 'experiments' / 'exp6_efficiency' / 'results' / 'efficiency_results.json'
    payload = _safe_load_json(results_path)
    if isinstance(payload, dict):
        metadata = payload.get('metadata', {})
        rows = payload.get('results', []) if metadata.get('batch_size') == 64 and metadata.get('repeats', 0) >= 10 else []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    if rows:
        return rows
    return [
        {'model': 'PLGAFormer', 'params_k': 4684, 'flops_mflops': 599.6, 'memory_mb': 18.7, 'latency_ms': 432, 'MSE_32': 0.000114},
        {'model': 'Transformer', 'params_k': 4477, 'flops_mflops': 573.1, 'memory_mb': 17.9, 'latency_ms': 3.3, 'MSE_32': 0.000503},
        {'model': 'iTransformer', 'params_k': 12873, 'flops_mflops': 1647.8, 'memory_mb': 51.5, 'latency_ms': 8.2, 'MSE_32': 0.000228},
        {'model': 'Informer', 'params_k': 4477, 'flops_mflops': 573.1, 'memory_mb': 17.9, 'latency_ms': 4.1, 'MSE_32': 0.129599},
        {'model': 'Autoformer', 'params_k': 4477, 'flops_mflops': 573.1, 'memory_mb': 17.9, 'latency_ms': 5.5, 'MSE_32': 0.016851},
        {'model': 'FEDformer', 'params_k': 4477, 'flops_mflops': 573.1, 'memory_mb': 17.9, 'latency_ms': 6.8, 'MSE_32': 0.001057},
        {'model': 'PatchTST', 'params_k': 4477, 'flops_mflops': 573.1, 'memory_mb': 17.9, 'latency_ms': 2.5, 'MSE_32': 0.003600},
    ]


def _number(row, *keys, default=0.0):
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.replace(',', '').replace('K', '').replace('M', '').replace('MB', '').replace('ms', '')
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def table_efficiency():
    """Efficiency comparison table."""
    rows = _efficiency_rows()
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    input_steps = int(DATASET_TIME_METADATA.get('seq_len', 256))
    input_seconds = input_steps * SAMPLING_INTERVAL_S
    lines.append(r'\caption{Computational efficiency comparison (' + f'{input_steps} step/{input_seconds:g} s input, batch=64' + r').}')
    lines.append(r'\label{tab:efficiency}')
    lines.append(r'\begin{tabular}{lrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Model & Params (K) & FLOPs (M) & Memory (MB) & Latency (ms) & MSE@' + HORIZON_LABELS.get(32, '32 step') + r' \\')
    lines.append(r'\midrule')
    for row in rows:
        model = row.get('model', 'Unknown')
        params_k = _number(row, 'params_k', 'param_k')
        flops_m = _number(row, 'flops_mflops', 'flops')
        memory_mb = _number(row, 'memory_mb', 'memory')
        latency_ms = _number(row, 'latency_ms', 'latency')
        mse_32 = _number(row, 'MSE_32')
        lines.append(f'{model} & {params_k:,.1f} & {flops_m:.1f} & {memory_mb:.1f} & {latency_ms:.2f} & {mse_32:.6f} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def table_physics_consistency():
    """Physics consistency table."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Physics consistency analysis: constraint violation rates.}')
    lines.append(r'\label{tab:physics}')
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\toprule')
    lines.append(r'Metric & Transformer & PLGAFormer & Improvement \\')
    lines.append(r'\midrule')
    lines.append(r'Height Violation Rate & 0\% & 0\% & -- \\')
    lines.append(r'Velocity Violation Rate & 67\% & \textbf{36\%} & \textbf{46.3\%} \\')
    lines.append(r'Acceleration Violation Rate & 87\% & \textbf{67\%} & \textbf{23.0\%} \\')
    lines.append(r'MSE (physical space) & baseline & \textbf{75.5\% lower} & \textbf{75.5\%} \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def _fmt_stat(stats, decimals=1, sci=False):
    if not stats:
        return '--'
    mean, std = stats
    if sci:
        return f'{mean:.3e}$\\pm${std:.1e}'
    return f'{mean:.{decimals}f}$\\pm${std:.{decimals}f}'


def table_sota_comparison():
    """SOTA comparison table generated from the active Exp1 aggregate."""
    n_h = len(HORIZONS)
    reference_horizon = 256 if 256 in HORIZONS else (128 if 128 in HORIZONS else max(HORIZONS))
    mse_end = 1 + n_h
    mae_start = mse_end + 1
    mae_end = mae_start + n_h - 1
    col_spec = 'l' + 'c' * n_h + '|' + 'c' * n_h + 'c'
    horizon_cells = ' & '.join(HORIZON_LABELS[h] for h in HORIZONS)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Quantitative comparison across varying prediction lengths (test set; source: ' + SOTA_SOURCE.replace('_', r'\_') + r').}',
        r'\label{tab:sota}',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
        r'& \multicolumn{' + str(n_h) + r'}{c}{MSE ($\times 10^{-4}$)} & \multicolumn{' + str(n_h) + r'}{c}{MAE ($\times 10^{-2}$)} & \\',
        r'\cmidrule(lr){2-' + str(mse_end) + r'} \cmidrule(lr){' + str(mae_start) + r'-' + str(mae_end) + r'}',
        r'Model & ' + horizon_cells + r' & ' + horizon_cells + r' & Improvement \\',
        r'\midrule',
    ]

    models = [m for m in MODEL_ORDER if m in SOTA_MSE]
    mse_rank = {}
    for h in HORIZONS:
        ranked = sorted(
            [(m, SOTA_MSE[m][h][0]) for m in models if h in SOTA_MSE.get(m, {})],
            key=lambda item: item[1],
        )
        mse_rank[h] = {m: rank for rank, (m, _) in enumerate(ranked)}

    def _ranked_cell(model, horizon, raw_value):
        rank = mse_rank.get(horizon, {}).get(model)
        if rank == 0:
            return f'\\textbf{{{raw_value}}}'
        if rank == 1:
            return f'\\underline{{{raw_value}}}'
        return raw_value

    for model in models:
        mse_vals = []
        mae_vals = []
        for h in HORIZONS:
            if h in SOTA_MSE.get(model, {}):
                raw = f'{SOTA_MSE[model][h][0]*10000:.2f}$\\pm${SOTA_MSE[model][h][1]*10000:.2f}'
                mse_vals.append(_ranked_cell(model, h, raw))
            else:
                mse_vals.append('--')
            mae_value = SOTA_MAE.get(model, {}).get(h)
            mae_vals.append('--' if mae_value is None else f'{mae_value*100:.2f}')

        if model == 'Transformer' or reference_horizon not in SOTA_MSE.get('Transformer', {}):
            improv = 0.0
        else:
            improv = (1 - SOTA_MSE[model][reference_horizon][0] / SOTA_MSE['Transformer'][reference_horizon][0]) * 100

        model_label = f'\\textbf{{{model}}}' if model == 'PLGAFormer' else model
        improvement_value = f'{improv:.1f}\\%'
        if model != 'Transformer' and improv > 0:
            improvement_value = f'\\textbf{{{improvement_value}}}'
        line = f'{model_label} & ' + ' & '.join(mse_vals) + ' & ' + ' & '.join(mae_vals) + f' & {improvement_value}'
        lines.append(line + r' \\')

    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    if 256 not in HORIZONS:
        lines.append(r'% NOTE: 256-step values require a pred_len=256 dataset and formal Exp1 rerun; current formal artifacts cover 32/64/128.')
    return '\n'.join(lines)


def table_sota_physical_units():
    """Physical-unit SOTA table populated from the active Exp1 aggregate."""
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Physical-space trajectory errors from active Exp1 results. RMSE and MAE are component-wise Cartesian errors.}',
        r'\label{tab:sota_physical_units}',
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        r'Model & Horizon & RMSE (m) & MAE (m) & FDE (m) & ADE (m) \\',
        r'\midrule',
    ]
    for model in [m for m in MODEL_ORDER if m in SOTA_PHYSICAL]:
        for h in HORIZONS:
            metrics = SOTA_PHYSICAL.get(model, {}).get(h, {})
            lines.append(
                f'{model} & {HORIZON_LABELS[h]} & '
                f'{_fmt_stat(metrics.get("rmse_cart_m"), decimals=1)} & '
                f'{_fmt_stat(metrics.get("mae_cart_m"), decimals=1)} & '
                f'{_fmt_stat(metrics.get("fde"), decimals=1)} & '
                f'{_fmt_stat(metrics.get("ade"), decimals=1)} ' + r'\\'
            )
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    if any(
        metrics.get('rmse_cart_m') is None or metrics.get('mae_cart_m') is None
        for by_horizon in SOTA_PHYSICAL.values()
        for metrics in by_horizon.values()
    ):
        lines.append(r'% NOTE: Some historical rows do not export Cartesian RMSE/MAE; rerun Exp1 with current code to populate those cells.')
    if 256 not in HORIZONS:
        lines.append(r'% NOTE: 256-step values require a pred_len=256 dataset and formal Exp1 rerun; current formal artifacts cover 32/64/128.')
    return '\n'.join(lines)


# ==================== MAIN ====================

def main():
    print('='*70)
    print('PLGAFormer Unified Paper Artifact Generator')
    print('='*70)

    print('\n[Generating Figures]')
    fig_maneuver_control_examples()
    fig_sota_grouped_bar()
    fig_ablation_radar()
    fig_architecture_overview()
    fig_trajectory_prediction()
    print(f'\n  Figures saved to {ARTIFACTS_DIR}/')

    print('\n[Generating LaTeX Tables]')
    tables = {
        'table_sota_comparison.tex': table_sota_comparison(),
        'table_sota_physical_units.tex': table_sota_physical_units(),
        'table_ablation.tex': table_ablation(),
        'table_missing_data.tex': table_missing_data(),
        'table_efficiency.tex': table_efficiency(),
        'table_physics_consistency.tex': table_physics_consistency(),
    }
    for fname, content in tables.items():
        path = ARTIFACTS_DIR / fname
        path.write_text(content, encoding='utf-8')
        print(f'  {fname}')

    # Also generate a combined plain-text summary
    summary = []
    summary.append('='*80)
    summary.append('PLGAFormer Paper — Table Summary')
    summary.append('='*80)
    summary.append('')
    summary.append('Table 1: SOTA Comparison (MSE, test set, mean±std over 3 seeds)')
    summary.append('-'*80)
    summary.append(
        f'{"Model":<15} '
        + ' '.join([f"MSE@{HORIZON_LABELS.get(h, str(h))}".ljust(20) for h in [32, 64, 128]])
        + f' {"vs Baseline":<15}'
    )
    for m in [x for x in MODEL_ORDER if x in SOTA_MSE]:
        imp = (1 - SOTA_MSE[m][128][0] / SOTA_MSE['Transformer'][128][0]) * 100
        summary.append(f'{m:<15} {SOTA_MSE[m][32][0]:<20.6f} {SOTA_MSE[m][64][0]:<20.6f} {SOTA_MSE[m][128][0]:<20.6f} {imp:<15.1f}%')

    summary.append('')
    summary.append('Table 2: Ablation Study')
    summary.append('-'*80)
    for v, d in ABLATION_DATA.items():
        summary.append(f'{v:<25} MSE@32={d["MSE_32"]:.6f}  MSE@64={d["MSE_64"]:.6f}  Δ@32=+{d["Δ%_32"]:.1f}%  Δ@64=+{d["Δ%_64"]:.1f}%')

    summary.append('')
    summary.append('Table 3: Efficiency')
    summary.append('-'*80)
    summary.append('PLGAFormer: 4.68M params, ~600M FLOPs, 432ms latency, 75.5% MSE improvement over baseline')

    reference_horizon = 256 if 256 in HORIZONS else (128 if 128 in HORIZONS else max(HORIZONS))
    dynamic_summary = []
    dynamic_summary.append('='*80)
    dynamic_summary.append('PLGAFormer Paper Table Summary')
    dynamic_summary.append('='*80)
    dynamic_summary.append('')
    dynamic_summary.append(f'Table 1: SOTA Comparison (MSE, test set; source={SOTA_SOURCE})')
    dynamic_summary.append('-'*80)
    dynamic_summary.append(_dataset_timing_summary())
    dynamic_summary.append(f'Available horizons: {", ".join(HORIZON_LABELS.get(h, str(h)) for h in HORIZONS)}')
    dynamic_summary.append('256-step status: available' if 256 in HORIZONS else '256-step status: pending pred_len=256 Exp1 rerun')
    dynamic_summary.append(f'{"Model":<15} ' + ' '.join([f"MSE@{HORIZON_LABELS.get(h, str(h))}".ljust(24) for h in HORIZONS]) + f' {"vs Baseline":<15}')
    for m in [x for x in MODEL_ORDER if x in SOTA_MSE]:
        if reference_horizon not in SOTA_MSE.get(m, {}) or reference_horizon not in SOTA_MSE.get('Transformer', {}):
            continue
        imp = (1 - SOTA_MSE[m][reference_horizon][0] / SOTA_MSE['Transformer'][reference_horizon][0]) * 100
        values = ' '.join([
            f'{SOTA_MSE[m][h][0]:<20.6f}' if h in SOTA_MSE[m] else f'{"--":<20}'
            for h in HORIZONS
        ])
        dynamic_summary.append(f'{m:<15} {values} {imp:<15.1f}%')
    dynamic_summary.append('')
    dynamic_summary.append('Table 1b: Physical-unit SOTA metrics')
    dynamic_summary.append('-'*80)
    dynamic_summary.append('Generated as table_sota_physical_units.tex with RMSE (m), MAE (m), FDE (m), and ADE (m).')
    dynamic_summary.append('')
    dynamic_summary.append('Table 2: Ablation Study')
    dynamic_summary.append('-'*80)
    for v, d in ABLATION_DATA.items():
        dynamic_summary.append(f'{v:<25} MSE@32={d["MSE_32"]:.6f}  MSE@64={d["MSE_64"]:.6f}')
    dynamic_summary.append('')
    dynamic_summary.append('Table 3: Efficiency')
    dynamic_summary.append('-'*80)
    dynamic_summary.append('See table_efficiency.tex for measured params, FLOPs, memory, latency, and MSE.')
    summary = dynamic_summary

    (ARTIFACTS_DIR / 'table_summary.txt').write_text('\n'.join(summary), encoding='utf-8')
    print(f'  table_summary.txt')

    print(f'\nAll artifacts saved to {ARTIFACTS_DIR}/')
    print(f'  Figures: 7 PNG + 7 PDF + 3 SVG')
    print(f'  Tables:  6 .tex + 1 .txt')


if __name__ == '__main__':
    if '--allow-legacy-diagnostic' not in sys.argv[1:]:
        raise SystemExit(
            'This historical generator is diagnostic-only; '
            'pass --allow-legacy-diagnostic explicitly.'
        )
    main()
