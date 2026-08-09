import json
from pathlib import Path

import pytest

from utils.experiment_io import resolve_exp1_checkpoint


def _write_registry(root: Path, checkpoint: Path) -> str:
    signature = "a" * 64
    results_dir = root / "experiments" / "exp1_sota" / "results"
    results_dir.mkdir(parents=True)
    key = (
        "seed=42|model=plgaformer|name=PLGAFormer (proposed)|pred_len=256|"
        f"horizons=32,64,128,256|signature={signature}"
    )
    payload = {
        "active_config": {"run_signature": signature, "random_seeds": [42]},
        "runs": {
            key: {
                "seed": 42,
                "checkpoint_path": str(checkpoint),
                "completed_at": "2026-07-22 12:00:00",
            }
        },
    }
    (results_dir / "partial_runs.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return signature


def test_resolve_exp1_checkpoint_uses_active_signature(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    signature = _write_registry(tmp_path, checkpoint)
    monkeypatch.delenv("HGV_EXP1_RUN_SIGNATURE", raising=False)
    monkeypatch.delenv("HGV_EXP1_PLGAFORMER_CHECKPOINT", raising=False)

    resolved, audit = resolve_exp1_checkpoint(tmp_path, "plgaformer")

    assert resolved == checkpoint.resolve()
    assert audit["run_signature"] == signature
    assert audit["source"] == "exp1_partial_runs"


def test_resolve_exp1_checkpoint_rejects_unfinished_signature(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    _write_registry(tmp_path, checkpoint)
    monkeypatch.setenv("HGV_EXP1_RUN_SIGNATURE", "b" * 64)

    with pytest.raises(FileNotFoundError, match="No completed exp1 checkpoint"):
        resolve_exp1_checkpoint(tmp_path, "plgaformer")
