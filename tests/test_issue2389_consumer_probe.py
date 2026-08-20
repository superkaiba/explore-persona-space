"""Issue #2389 staged-anchor consumer probes (plan §4.6 M1-iv) — unit tests.

Offline (``--local-anchors-dir``): the probes run the REAL consumer loaders
(``issue2389_judge.load_anchor_rows`` + ``issue2389_analysis._load_anchor_va``)
over synthetic tmp shard pairs shaped exactly like the pod's gate uploads
(``anchors_gate_w0.jsonl`` + ``va_anchors_gate_w0.pt``). No network: the
staging leg is covered by a signature-bind of the script's exact
``hub.stage_hub_file`` call shape against the real helper.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2389_analysis as A  # noqa: E402
import issue2389_consumer_probe as P  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

H = 4
ROWS = [
    ("ctx_a", 0),
    ("ctx_a", 1),
    ("ctx_b", 0),
]


def _write_jsonl(anchors_dir: Path, keys: list[tuple[str, int]]) -> None:
    lines = [
        json.dumps(
            {
                "context_id": cid,
                "cell": "fact_user_name",
                "value_id": "v0",
                "carrier": "c0",
                "draw": draw,
                "text": f"rollout {cid}/{draw}",
            }
        )
        for cid, draw in keys
    ]
    (anchors_dir / "anchors_gate_w0.jsonl").write_text("\n".join(lines) + "\n")


def _write_pt(
    anchors_dir: Path,
    keys: list[tuple[str, int]],
    empty_rows: list[int],
    name: str = "va_anchors_gate_w0.pt",
) -> None:
    payload = {
        "layers": [A.READ_LAYER],
        "va_span": torch.randn(len(keys), 1, H),
        "index": [{"context_id": cid, "draw": draw} for cid, draw in keys],
        "empty_rows": list(empty_rows),
    }
    torch.save(payload, anchors_dir / name)


@pytest.fixture()
def anchors_dir(tmp_path: Path) -> Path:
    d = tmp_path / "anchors"
    d.mkdir()
    return d


def test_happy_path_both_probes_pass_and_report(anchors_dir: Path, tmp_path: Path) -> None:
    # Row 2 (ctx_b, 0) declared empty: jsonl-present, va-excluded — still PASS.
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS, empty_rows=[2])
    report_path = tmp_path / "gates" / "consumer_probe_report.json"
    rc = P.main(
        [
            "--probe",
            "both",
            "--local-anchors-dir",
            str(anchors_dir),
            "--report",
            str(report_path),
        ]
    )
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["legs"]["judge"]["n_rows"] == 3
    assert report["legs"]["analysis"]["n_va_keys"] == 2
    assert report["legs"]["analysis"]["n_declared_empty"] == 1
    assert report["legs"]["analysis"]["key_sets_identical_modulo_empty"] is True
    assert report["staged"] is False
    assert report["repro"]["script"] == "scripts/issue2389_consumer_probe.py"


def test_jsonl_key_missing_from_va_raises(anchors_dir: Path) -> None:
    # pt drops (ctx_b, 0) entirely (not declared empty) -> contract mismatch.
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS[:2], empty_rows=[])
    with pytest.raises(AssertionError, match="key-set mismatch"):
        P.probe_analysis(anchors_dir)


def test_va_key_absent_from_jsonl_raises(anchors_dir: Path) -> None:
    # pt carries an extra (ctx_c, 0) the jsonl never produced.
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, [*ROWS, ("ctx_c", 0)], empty_rows=[])
    with pytest.raises(AssertionError, match="key-set mismatch"):
        P.probe_analysis(anchors_dir)


def test_loaded_and_declared_empty_overlap_raises(anchors_dir: Path) -> None:
    # Shard 1 LOADS (ctx_a, 0); shard 2 declares the same key empty ->
    # duplicate/stale-shard overlap, its own error leg.
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS, empty_rows=[], name="va_anchors_gate_w0.pt")
    _write_pt(anchors_dir, [("ctx_a", 0)], empty_rows=[0], name="va_anchors_gate_w1.pt")
    with pytest.raises(AssertionError, match=r"declared\s+empty"):
        P.probe_analysis(anchors_dir)


def test_stage_call_shape_binds_to_real_helper() -> None:
    # The offline tests never stage; pin the script's exact call shape
    # against the real hub.stage_hub_file signature instead (arity/keyword
    # drift fails here, not at pod time).
    sig = inspect.signature(hub.stage_hub_file)
    sig.bind(
        P.J.DATASET_REPO,
        f"{P.J._STAGE_ANCHORS_GATE}/anchors_gate_w0.jsonl",
        Path("/tmp/x/anchors_gate_w0.jsonl"),
        repo_type="dataset",
        revision=None,
        overwrite=True,
    )
    sig.bind(
        P.J.DATASET_REPO,
        f"{P._VA_ANCHORS_REMOTE_PREFIX}/va_anchors_gate_w0.pt",
        Path("/tmp/x/va_anchors_gate_w0.pt"),
        repo_type="dataset",
        revision=None,
        overwrite=True,
    )


def test_local_anchors_dir_must_exist(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="not a directory"):
        P.main(["--probe", "judge", "--local-anchors-dir", str(tmp_path / "nope")])
