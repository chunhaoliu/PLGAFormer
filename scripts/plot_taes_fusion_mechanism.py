"""Explain actual fusion weights for the already selected Fig. 4 case.

Contract: quantitative grid, three position channels, frozen seeds 42/123/456.
Show scheduled confidence versus measured adaptive weights, with seed SD.
This is a case diagnostic, not a causal ablation or a population result.
"""
from pathlib import Path
import sys
import json
import types
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_confirmatory_holdout as ev


def main():
    out = ROOT / 'tmp/figure_review/fusion_mechanism'
    out.mkdir(parents=True, exist_ok=True)
    case_path = ROOT / 'tmp/figure_review/trajectory_case_study/trajectory_case_qa.json'
    case = json.loads(case_path.read_text(encoding='utf-8'))
    manifest, data = ev.validate_locked_dataset()
    bundle, records = ev.validate_main_bundle()
    assert case['main_bundle']['bundle_id'] == bundle['bundle_id']
    assert case['dataset']['sha256'] == manifest['sha256']
    idx = case['selection']['window_index']
    assert int(data['trajectory_ids_confirmatory'][idx]) == case['selection']['trajectory_id']
    payload = ev._source_payload(records[('full', 42)])
    ins, outs, scaler_hashes = ev._load_scalers(payload)
    subset = {k: data[k][idx:idx+1] for k in ('X_confirmatory', 'y_confirmatory')}
    x, y = next(iter(ev.prepare_loader(subset, ins, outs, 1)))
    del data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y = x.to(device), y.to(device)
    arrays, checks = [], []
    for seed in ev.SEEDS:
        rec = records[('full', seed)]
        p = ev._source_payload(rec)
        _, _, sh = ev._load_scalers(p)
        assert sh == scaler_hashes
        checkpoint = Path(rec['checkpoint'])
        assert ev.sha256_file(checkpoint) == rec['checkpoint_sha256']
        model = ev.reconstruct_model(p, ins, outs, device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        policy = ev.apply_paper_inference_policy(model)
        model.eval()
        assert ev.exp1.EVAL_PROTOCOL == 'source_context_decoder'
        captured = []
        original = model._scheduled_prior_weight
        def capture(self, prior, learned, decoder, mask):
            weight = original(prior, learned, decoder, mask)
            captured.append(weight[:, -256:].detach().cpu().numpy()[0])
            return weight
        with torch.no_grad():
            reference = ev.exp1.unified_predict(model, x, 256, y_true_scaled=y, device=device,
                                               eval_protocol=ev.exp1.EVAL_PROTOCOL)
            model._scheduled_prior_weight = types.MethodType(capture, model)
            try:
                measured = ev.exp1.unified_predict(model, x, 256, y_true_scaled=y, device=device,
                                                  eval_protocol=ev.exp1.EVAL_PROTOCOL)
            finally:
                model._scheduled_prior_weight = original
        assert torch.equal(reference, measured), 'Instrumentation changed model output'
        assert len(captured) == 1
        w = captured[0]
        assert w.shape == (256, 3) and np.isfinite(w).all()
        assert np.all((w >= 0) & (w <= 1)) and np.all(w[:64] == 1)
        arrays.append(w)
        checks.append({'seed': seed, 'checkpoint_sha256': rec['checkpoint_sha256'],
                       'source_sha256': rec['source_sha256'], 'output_bitwise_equal': True,
                       'inference_policy': policy})
        tau = model.physics_prior_time_constant_s
        power = model.physics_prior_decay_power
        del model, reference, measured
    weights = np.stack(arrays)
    t = np.arange(1, 257)
    schedule = np.clip(np.exp(-(t / tau) ** power), .01, .99)
    np.savez_compressed(out / 'fusion_weights.npz', weights=weights, schedule=schedule,
                        time_s=t, seeds=np.array(ev.SEEDS))
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True, layout='constrained')
    for c, (ax, label) in enumerate(zip(axes, ('Radius', 'Longitude', 'Latitude'))):
        mean, sd = weights[:, :, c].mean(0), weights[:, :, c].std(0)
        ax.axvspan(1, 64, color='#DCE1E6', alpha=.45)
        ax.axvline(64, color='#89929B', ls=':', lw=.8)
        ax.plot(t, schedule, color='#555D66', ls='--', lw=1.1, label='Schedule before lock')
        ax.fill_between(t, np.maximum(0, mean-sd), np.minimum(1, mean+sd),
                        color='#C23B3B', alpha=.18)
        ax.plot(t, mean, color='#C23B3B', lw=1.5, label='Actual fusion weight')
        ax.set(xlim=(1, 256), ylim=(0, 1.03), xlabel='Forecast time (s)', title=label)
        ax.set_xticks([1,64,128,192,256])
        ax.grid(axis='y', color='#D7DCE2', lw=.5)
    axes[0].set_ylabel('Prior weight')
    axes[0].legend(loc='lower left', fontsize=6)
    for ext in ('pdf', 'svg', 'png', 'tiff'):
        fig.savefig(out / f'fusion_mechanism.{ext}', dpi=600, bbox_inches='tight')
    plt.close(fig)
    qa = {'status': 'candidate_not_integrated', 'training_performed': False,
          'selection': case['selection'], 'case_manifest_sha256': ev.sha256_file(case_path),
          'dataset_sha256': manifest['sha256'], 'bundle_id': bundle['bundle_id'],
          'config_sha256': bundle['config_sha256'], 'scaler_sha256': scaler_hashes,
          'checks': checks, 'seed_count': 3, 'trajectory_count': 1,
          'spread': 'Population SD across three frozen checkpoint seeds',
          'claim_boundary': 'Same Fig. 4 case; measured weights do not establish causal benefit',
          'schedule_definition': 'exp(-(t/450)^2.5) clipped to [0.01,0.99], before lock override',
          'weight_at_256_mean': weights[:, -1].mean(0).tolist(),
          'weight_at_256_sd': weights[:, -1].std(0).tolist(),
          'script_sha256': ev.sha256_file(Path(__file__)),
          'model_sha256': ev.sha256_file(ROOT / 'models/PLGAFormer.py'),
          'artifacts': {p.name: ev.sha256_file(p) for p in out.iterdir() if p.suffix in ('.pdf','.svg','.png','.tiff','.npz')}}
    (out / 'fusion_mechanism_qa.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    main()
