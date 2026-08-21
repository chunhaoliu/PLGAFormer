import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "PublicRelease" / "evidence"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_public_evidence_manifest_binds_every_csv():
    manifest = json.loads((EVIDENCE / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "public_release_evidence"
    assert manifest["dataset_protocol"] == "hgv_multiregime_state_v2_1"
    assert manifest["claim_boundary"] == "complete simulated HGV trajectories; simulation-only evidence"
    assert manifest["files"]
    for name, expected_hash in manifest["files"].items():
        path = EVIDENCE / name
        assert path.is_file()
        assert _sha256(path) == expected_hash


def test_public_evidence_is_path_free_and_nonempty():
    for path in EVIDENCE.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert "D:\\" not in text
        assert "D:/" not in text
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert rows, path.name


def test_adaptive_gate_release_table_keeps_restrained_256s_boundary():
    path = EVIDENCE / "adaptive_gate_paired.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    at_256 = [row for row in rows if row["horizon_s"] == "256"]
    assert {row["metric"] for row in at_256} == {"ade", "fde"}
    assert all(
        row["evidence_class"] == "trajectory_conditional_average_improvement_only"
        for row in at_256
    )
