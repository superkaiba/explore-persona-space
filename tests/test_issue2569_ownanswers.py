from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_atlas as atlas  # noqa: E402
import issue2569_operator as op  # noqa: E402
import issue2569_ownanswers_analyze as ana  # noqa: E402
import issue2569_ownanswers_generate as gen  # noqa: E402
import issue2569_xmodel_capture as cap  # noqa: E402


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_exact_folds_are_deterministic_disjoint_and_exact_size():
    ci = np.arange(10_000, 10_100, dtype=np.int64)
    a = ana.exact_folds(ci, 80, 5, 15)
    b = ana.exact_folds(ci, 80, 5, 15)
    assert all(np.array_equal(a[k], b[k]) for k in ("tr", "va", "te"))
    assert (len(a["tr"]), len(a["va"]), len(a["te"])) == (80, 5, 15)
    assert len(np.unique(np.concatenate([a["tr"], a["va"], a["te"]]))) == 100


def test_alternate_qwen_text_uses_identity_gate_without_changing_default():
    assert cap._capture_gate_name(SimpleNamespace(model="qwen", qwen_gate="spot")) == (
        "spot_gate_qwen"
    )
    assert cap._capture_gate_name(SimpleNamespace(model="qwen", qwen_gate="identity")) == (
        "identity_gate_qwen"
    )
    assert cap._capture_gate_name(SimpleNamespace(model="llama", qwen_gate="spot")) == (
        "identity_gate_llama"
    )


def test_prepare_materializes_capture_contract_and_preserves_source_order(tmp_path):
    source = tmp_path / "source"
    generated = tmp_path / "generated"
    capture = tmp_path / "capture"
    source_rows = [
        {"ci": 9, "corpus": "lmsys", "prompt": "p9", "response": "q9"},
        {"ci": 4, "corpus": "wildchat", "prompt": "p4", "response": "q4"},
        {"ci": 7, "corpus": "lmsys", "prompt": "p7", "response": "q7"},
    ]
    _write_jsonl(source / "texts_kept.jsonl", source_rows)
    answer_rows = [
        {**source_rows[2], "response": "l7", "drop_reason": None},
        {**source_rows[0], "response": "l9", "drop_reason": None},
        {**source_rows[1], "response": "", "drop_reason": "empty_response"},
    ]
    _write_jsonl(generated / "answers.jsonl", answer_rows)
    (generated / "audit.json").write_text("{}\n")
    args = SimpleNamespace(
        source_root=str(source),
        out_root=str(generated),
        capture_root=str(capture),
        model="llama",
        seed=42,
        rows=0,
        ci_roster="",
    )
    gen.phase_prepare(args)
    kept = gen._read_jsonl(capture / "texts_kept.jsonl")
    assert [row["ci"] for row in kept] == [9, 7]
    assert [row["response"] for row in kept] == ["l9", "l7"]
    manifest = json.loads((capture / "writer_manifest.json").read_text())
    assert manifest["n_source"] == 3
    assert manifest["n_kept"] == 2
    assert manifest["drops"] == {"empty_response": 1}


def test_source_roster_can_select_frozen_test_partition(tmp_path):
    source = tmp_path / "source"
    rows = [
        {"ci": ci, "corpus": "lmsys", "prompt": f"p{ci}", "response": f"q{ci}"}
        for ci in (9, 4, 7)
    ]
    _write_jsonl(source / "texts_kept.jsonl", rows)
    roster = tmp_path / "split.json"
    roster.write_text(json.dumps({"ci": [9, 4, 7], "test_ci": [7, 9]}))
    args = SimpleNamespace(
        source_root=str(source),
        ci_roster=str(roster),
        ci_roster_key="test_ci",
        rows=0,
    )
    selected = gen._load_source(args)
    assert [row["ci"] for row in selected] == [7, 9]


def test_row_cosine_and_subset_r2_helpers():
    x = np.eye(4, dtype=np.float64)
    assert np.allclose(ana._cos_rows(x, x), 1.0)
    assert ana._pool_r2_subset(x, x, np.asarray([0, 1])) is None
    assert ana._pool_r2_subset(x, x, np.asarray([0, 1, 2])) == 1.0


def test_repeatability_identity_case():
    x = np.arange(24, dtype=np.float64).reshape(6, 4) + 1.0
    rep = ana._repeatability(x, x)
    assert np.isclose(rep["linear_cka"], 1.0)
    assert np.isclose(rep["identity_pooled_r2"], 1.0)
    assert np.isclose(rep["row_cosine"]["mean"], 1.0)


def test_ridge_payload_device_cpu_matches_historical_default():
    rng = np.random.default_rng(137)
    x = rng.standard_normal((48, 9))
    y = rng.standard_normal((48, 7))
    tr = np.arange(36)
    default = atlas.ridge_beta_at_lambda(x, y, tr, 2.5)
    explicit = atlas.ridge_beta_at_lambda(x, y, tr, 2.5, device="cpu")
    assert np.allclose(op.predict(default, x), op.predict(explicit, x), atol=0, rtol=0)


def test_procrustes_device_cpu_matches_historical_default():
    rng = np.random.default_rng(2569)
    a = rng.standard_normal((64, 11))
    b = rng.standard_normal((64, 13))
    default = atlas.orth_procrustes(a, b)
    explicit = atlas.orth_procrustes(a, b, device="cpu")
    assert np.allclose(default, explicit, atol=0, rtol=0)
