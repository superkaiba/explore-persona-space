"""#1491 greedy round — Path B cap-hit re-gen pass invariants.

The regen pass extends the greedy round's decoding-regime identity with the
generation token CAP and merges regenerated rows back by overlay. The
load-bearing hazards these tests pin (each a permanent invariant of the
round's substantive fix — they fail on the pre-round code by construction):

1. CAP IDENTITY — a 2048 re-gen row must never be confusable with a 1024
   base row: payload-level (``_assert_raw_payload_matches``) and
   prefix-grain (``_enforce_sampling_identity``) checks refuse cross-cap
   reuse, including LEGACY payloads/markers with no ``gen_max_tokens`` key
   (uniquely cap-1024 by construction — the cap was hardcoded).
2. TEXT<->ACTIVATION BINDING — a regen ``.pt`` whose activations do not
   correspond to the paired regen raw text (the silent-corruption failure
   mode: 2048 text merged with a 1024-truncated capture) is UNLOADABLE:
   ``rows_sha256`` recomputation fails loud.
3. MERGE — ``stream_split_merged`` overlays exactly the regen-captured rows,
   returns per-row cap provenance, and fails loud on overlay cis absent from
   the base corpus.

Offline by construction: every loader runs on tmp_path files (the
``local_dir`` / ``base_local_dir`` smoke seams — the network boundary);
everything else executes the real bodies (tests/ runs in every issue's
Step 9c gate — no live Hub fetch allowed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1491_caphit_regen as R  # noqa: E402
import issue1491_caphit_restriction_analysis as A  # noqa: E402
import issue1491_ladder_generate_capture as D  # noqa: E402

GREEDY = D.SAMPLING_MODE_GREEDY
PARENT = D.SAMPLING_MODE_PARENT
BASE_CAP = D.GEN_MAX_TOKENS
REGEN_CAP = R.DEFAULT_REGEN_MAX_TOKENS
LAYERS = [1, 2, 3]
H = 4


# ---------------------------------------------------------------------------
# 1. Cap identity — payload grain
# ---------------------------------------------------------------------------


def _payload(**over):
    p = {
        "shard_index": 0,
        "chunk": 0,
        "split": "test_1000",
        "seed": 42,
        "sampling_mode": GREEDY,
        "gen_max_tokens": BASE_CAP,
        "rows": [],
    }
    p.update(over)
    return p


def _assert_payload(payload, expect_cap):
    D._assert_raw_payload_matches(
        payload,
        "shard00_chunk0000.json",
        expect_split="test_1000",
        expect_seed=42,
        expect_shard_index=0,
        expect_chunk=0,
        expect_sampling_mode=GREEDY,
        expect_gen_max_tokens=expect_cap,
    )


def test_payload_same_cap_passes():
    _assert_payload(_payload(), BASE_CAP)
    _assert_payload(_payload(gen_max_tokens=REGEN_CAP), REGEN_CAP)


def test_payload_cross_cap_refuses():
    with pytest.raises(AssertionError, match="GEN-MAX-TOKENS"):
        _assert_payload(_payload(), REGEN_CAP)
    with pytest.raises(AssertionError, match="GEN-MAX-TOKENS"):
        _assert_payload(_payload(gen_max_tokens=REGEN_CAP), BASE_CAP)


def test_payload_legacy_missing_cap_key_is_base_cap():
    """A pre-regen-round chunk has NO gen_max_tokens key: the cap was
    hardcoded, so it must pass a base-cap consumer and refuse a regen-cap
    consumer."""
    legacy = _payload()
    del legacy["gen_max_tokens"]
    _assert_payload(legacy, BASE_CAP)
    with pytest.raises(AssertionError, match="GEN-MAX-TOKENS"):
        _assert_payload(legacy, REGEN_CAP)


def test_resolve_sampling_carries_cap_and_default_is_base():
    s = D._resolve_sampling(True)
    assert D._sampling_cap(s) == BASE_CAP  # driver default byte-identical
    s2 = D._resolve_sampling(True, gen_max_tokens=REGEN_CAP)
    assert D._sampling_cap(s2) == REGEN_CAP
    assert s2["mode"] == GREEDY and s2["temperature"] == 0.0
    # Hand-built dict with no max_tokens key maps to the base cap (legacy).
    assert D._sampling_cap({"mode": GREEDY, "temperature": 0.0, "top_p": 1.0}) == BASE_CAP


# ---------------------------------------------------------------------------
# 2. Cap identity — prefix grain (markers)
# ---------------------------------------------------------------------------


def _enforce(scratch, sampling, *, done_pt=(), done_raw=(), no_upload=True, shard_index=0):
    D._enforce_sampling_identity(
        "issue1491_test/prefix/test_1000/regen_cap2048",
        scratch,
        scratch / ".cache",
        sampling,
        done_pt=set(done_pt),
        done_raw=set(done_raw),
        no_upload=no_upload,
        shard_index=shard_index,
    )


def test_guard_local_marker_cap_mismatch_refuses(tmp_path):
    scratch = tmp_path / "s"
    scratch.mkdir()
    _enforce(scratch, D._resolve_sampling(True, gen_max_tokens=BASE_CAP))
    with pytest.raises(RuntimeError, match="local scratch"):
        _enforce(scratch, D._resolve_sampling(True, gen_max_tokens=REGEN_CAP))
    # Same (mode, cap) re-entry stays idempotent.
    _enforce(scratch, D._resolve_sampling(True, gen_max_tokens=BASE_CAP))


def test_guard_legacy_marker_without_cap_key_is_base_cap(tmp_path):
    """A marker written before the regen round lacks gen_max_tokens: it maps
    to the base cap — a regen-cap run refuses, a base-cap run proceeds."""
    scratch = tmp_path / "s"
    scratch.mkdir()
    (scratch / D.SAMPLING_MARKER_NAME).write_text(json.dumps({"sampling_mode": GREEDY}))
    with pytest.raises(RuntimeError, match="local scratch"):
        _enforce(scratch, D._resolve_sampling(True, gen_max_tokens=REGEN_CAP))
    _enforce(scratch, D._resolve_sampling(True, gen_max_tokens=BASE_CAP))


def test_guard_hub_marker_cap_mismatch_refuses(tmp_path, monkeypatch):
    scratch = tmp_path / "s"
    scratch.mkdir()

    def fake_download(stage_prefix: str, cache_dir: Path) -> dict | None:
        return {"sampling_mode": GREEDY, "gen_max_tokens": BASE_CAP}

    monkeypatch.setattr(D, "_download_hub_sampling_marker", fake_download)
    with pytest.raises(RuntimeError, match="\\(Hub\\)"):
        _enforce(scratch, D._resolve_sampling(True, gen_max_tokens=REGEN_CAP), no_upload=False)


# ---------------------------------------------------------------------------
# 3. Text<->activation binding (rows_sha256)
# ---------------------------------------------------------------------------


def _regen_bundle(cis, responses, cap=REGEN_CAP, **over):
    n = len(cis)
    b = {
        "cx_last": torch.randn(n, len(LAYERS), H),
        "v_x": torch.randn(n, len(LAYERS), H),
        "ci": [int(c) for c in cis],
        "prompts": [f"q{c}" for c in cis],
        "layers": list(LAYERS),
        "shard_index": 0,
        "chunk": 0,
        "sampling_mode": GREEDY,
        "gen_max_tokens": int(cap),
        "regen_of_cap": BASE_CAP,
        "dropped_empty_cis": [],
        "rows_sha256": R.rows_binding_sha(list(cis), list(responses)),
    }
    b.update(over)
    return b


def _raw_map(cis, responses):
    return {
        int(c): {"prompt": f"q{c}", "response": r, "finish_reason": "stop"}
        for c, r in zip(cis, responses, strict=True)
    }


def test_binding_passes_on_matching_text():
    cis, resps = [3, 7], ["long answer three", "long answer seven"]
    R._verify_pt_binding(_regen_bundle(cis, resps), _raw_map(cis, resps), "t", REGEN_CAP)


def test_binding_refuses_tampered_text():
    """The silent-corruption failure mode: activations captured from OTHER
    text than the paired raw rows — must be unloadable."""
    cis, resps = [3, 7], ["long answer three", "long answer seven"]
    bundle = _regen_bundle(cis, resps)
    tampered = _raw_map(cis, ["long answer three", "TRUNCATED 1024 text"])
    with pytest.raises(RuntimeError, match="rows_sha256 mismatch"):
        R._verify_pt_binding(bundle, tampered, "t", REGEN_CAP)


def test_binding_refuses_unbound_bundle():
    cis, resps = [3], ["x"]
    bundle = _regen_bundle(cis, resps)
    del bundle["rows_sha256"]
    with pytest.raises(RuntimeError, match="lacks rows_sha256"):
        R._verify_pt_binding(bundle, _raw_map(cis, resps), "t", REGEN_CAP)


def test_binding_refuses_wrong_cap_bundle():
    cis, resps = [3], ["x"]
    bundle = _regen_bundle(cis, resps, gen_max_tokens=BASE_CAP)
    with pytest.raises(RuntimeError, match="cap identity mismatch"):
        R._verify_pt_binding(bundle, _raw_map(cis, resps), "t", REGEN_CAP)


# ---------------------------------------------------------------------------
# 4. Deterministic regen chunk membership
# ---------------------------------------------------------------------------


def test_regen_chunk_membership_deterministic():
    cis = [9, 1, 5, 5, 3]
    assert R.regen_chunk_membership(cis) == [[1, 3, 5, 9]]
    big = list(range(R.REGEN_CHUNK_ROWS + 3))
    chunks = R.regen_chunk_membership(big)
    assert len(chunks) == 2
    assert chunks[0] == big[: R.REGEN_CHUNK_ROWS] and chunks[1] == big[R.REGEN_CHUNK_ROWS :]
    assert R.regen_chunk_membership(list(reversed(big))) == chunks  # order-invariant


def test_regen_budget_arithmetic_defaults():
    """Every base-admitted prompt + the regen cap must fit the regen engine
    (over-length add_request is engine-fatal), and the cap satisfies the
    pre-registered >=2x trigger."""
    assert R.DEFAULT_REGEN_MAX_TOKENS >= 2 * BASE_CAP
    budget = R.DEFAULT_REGEN_MAX_MODEL_LEN - R.DEFAULT_REGEN_MAX_TOKENS - D.LENGTH_MARGIN
    assert budget >= D.PROMPT_TOKEN_BUDGET


def test_parent_root_refused():
    import argparse

    args = argparse.Namespace(hf_root=R.PARENT_HF_ROOT)
    with pytest.raises(RuntimeError, match="PARENT"):
        R.run(args)


# ---------------------------------------------------------------------------
# 5. Scan + merge on real local files (real bodies; tmp_path boundary)
# ---------------------------------------------------------------------------


def _write_base_corpus(base_dir: Path, split="test_1000"):
    """A tiny base corpus in the exact writer formats: one raw chunk (4 rows,
    2 cap-hit) + one capture .pt chunk."""
    cis = [0, 1, 2, 3]
    finish = ["stop", "length", "stop", "length"]
    resps = [f"base answer {c}" for c in cis]
    raw_dir = base_dir / split / "raw_completions"
    raw_dir.mkdir(parents=True)
    payload = {
        "shard_index": 0,
        "chunk": 0,
        "split": split,
        "seed": 42,
        "sampling_mode": GREEDY,
        "gen_max_tokens": BASE_CAP,
        "n_cap_hit": 2,
        "rows": [
            {"ci": c, "prompt": f"q{c}", "response": r, "finish_reason": f}
            for c, r, f in zip(cis, resps, finish, strict=True)
        ],
    }
    (raw_dir / "shard00_chunk0000.json").write_text(json.dumps(payload))
    pt_dir = base_dir / split / "final_token_capture"
    pt_dir.mkdir(parents=True)
    torch.save(
        {
            "cx_last": torch.zeros(4, len(LAYERS), H),
            "v_x": torch.zeros(4, len(LAYERS), H),
            "ci": cis,
            "prompts": [f"q{c}" for c in cis],
            "layers": list(LAYERS),
            "shard_index": 0,
            "chunk": 0,
        },
        pt_dir / "shard00_chunk0000.pt",
    )
    return cis, finish, resps


def _write_regen_overlay(regen_dir: Path, split="test_1000"):
    """Regen artifacts for the two cap-hit cis (1, 3), one residual cap-hit."""
    regen_dir.mkdir(parents=True, exist_ok=True)
    cis = [1, 3]
    resps = ["base answer 1 continued to completion", "base answer 3 still truncated"]
    finish = ["stop", "length"]
    payload = {
        "shard_index": 0,
        "chunk": 0,
        "split": split,
        "seed": 42,
        "sampling_mode": GREEDY,
        "gen_max_tokens": REGEN_CAP,
        "regen_of_cap": BASE_CAP,
        "regen_pass": True,
        "n_cap_hit": 1,
        "rows": [
            {
                "ci": c,
                "prompt": f"q{c}",
                "response": r,
                "finish_reason": f,
                "base_chunk": "shard00_chunk0000.json",
            }
            for c, r, f in zip(cis, resps, finish, strict=True)
        ],
    }
    (regen_dir / "regen_chunk0000.json").write_text(json.dumps(payload))
    torch.save(
        {
            "cx_last": torch.ones(2, len(LAYERS), H),
            "v_x": torch.full((2, len(LAYERS), H), 2.0),
            "ci": cis,
            "prompts": [f"q{c}" for c in cis],
            "layers": list(LAYERS),
            "shard_index": 0,
            "chunk": 0,
            "sampling_mode": GREEDY,
            "gen_max_tokens": REGEN_CAP,
            "regen_of_cap": BASE_CAP,
            "dropped_empty_cis": [],
            "rows_sha256": R.rows_binding_sha(cis, resps),
        },
        regen_dir / "regen_chunk0000.pt",
    )
    return cis, finish, resps


def test_scan_detects_caphit_rows(tmp_path):
    base = tmp_path / "base"
    _write_base_corpus(base)
    scan = R.scan_split_caphit("root_x", "scale05", "test_1000", tmp_path / "sc", base)
    assert sorted(scan["rows"]) == [1, 3]
    assert scan["n_rows"] == 4 and scan["n_chunks"] == 1
    assert scan["rows"][1]["base_chunk"] == "shard00_chunk0000.json"


def test_scan_refuses_parent_mode_base(tmp_path):
    base = tmp_path / "base"
    _write_base_corpus(base)
    raw = base / "test_1000" / "raw_completions" / "shard00_chunk0000.json"
    payload = json.loads(raw.read_text())
    payload["sampling_mode"] = PARENT
    raw.write_text(json.dumps(payload))
    with pytest.raises(AssertionError, match="SAMPLING-MODE mismatch"):
        R.scan_split_caphit("root_x", "scale05", "test_1000", tmp_path / "sc", base)


def test_stream_split_merged_overlays_and_records_provenance(tmp_path):
    base = tmp_path / "base"
    _write_base_corpus(base)
    regen = tmp_path / "regen"
    _write_regen_overlay(regen)
    cx, vx, ci, gen_cap = R.stream_split_merged(
        "root_x",
        "scale05",
        "test_1000",
        REGEN_CAP,
        LAYERS[1],
        tmp_path / "sc",
        base_local_dir=base,
        regen_local_dir=regen,
    )
    assert ci == [0, 1, 2, 3]
    assert gen_cap.tolist() == [BASE_CAP, REGEN_CAP, BASE_CAP, REGEN_CAP]
    # Overlaid rows carry the regen tensors (ones / twos), base rows zeros.
    assert np.allclose(cx[0], 0.0) and np.allclose(cx[2], 0.0)
    assert np.allclose(cx[1], 1.0) and np.allclose(cx[3], 1.0)
    assert np.allclose(vx[1], 2.0) and np.allclose(vx[3], 2.0)


def test_stream_split_merged_refuses_foreign_overlay_ci(tmp_path):
    base = tmp_path / "base"
    _write_base_corpus(base)
    regen = tmp_path / "regen"
    _write_regen_overlay(regen)
    # Rewrite the overlay pair with a ci absent from the base corpus (99).
    cis = [1, 99]
    resps = ["a", "b"]
    payload = json.loads((regen / "regen_chunk0000.json").read_text())
    payload["rows"] = [
        {"ci": c, "prompt": f"q{c}", "response": r, "finish_reason": "stop"}
        for c, r in zip(cis, resps, strict=True)
    ]
    (regen / "regen_chunk0000.json").write_text(json.dumps(payload))
    b = torch.load(regen / "regen_chunk0000.pt", weights_only=False)
    b["ci"] = cis
    b["rows_sha256"] = R.rows_binding_sha(cis, resps)
    torch.save(b, regen / "regen_chunk0000.pt")
    with pytest.raises(RuntimeError, match="absent from the base corpus"):
        R.stream_split_merged(
            "root_x",
            "scale05",
            "test_1000",
            REGEN_CAP,
            LAYERS[1],
            tmp_path / "sc",
            base_local_dir=base,
            regen_local_dir=regen,
        )


def test_stream_split_merged_refuses_tampered_regen_text(tmp_path):
    """End-to-end guard: a regen raw file whose text no longer matches the
    .pt's binding sha must make the MERGE unloadable."""
    base = tmp_path / "base"
    _write_base_corpus(base)
    regen = tmp_path / "regen"
    _write_regen_overlay(regen)
    payload = json.loads((regen / "regen_chunk0000.json").read_text())
    payload["rows"][0]["response"] = "silently swapped text"
    (regen / "regen_chunk0000.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="rows_sha256 mismatch"):
        R.stream_split_merged(
            "root_x",
            "scale05",
            "test_1000",
            REGEN_CAP,
            LAYERS[1],
            tmp_path / "sc",
            base_local_dir=base,
            regen_local_dir=regen,
        )


def test_load_regen_raw_overlay_refuses_cross_cap_namespace(tmp_path):
    """A cap-1024 payload sitting in a cap-2048 namespace read is refused —
    the (c) smoke deliverable at the loader grain."""
    regen = tmp_path / "regen"
    _write_regen_overlay(regen)
    payload = json.loads((regen / "regen_chunk0000.json").read_text())
    payload["gen_max_tokens"] = BASE_CAP
    (regen / "regen_chunk0000.json").write_text(json.dumps(payload))
    with pytest.raises(AssertionError, match="GEN-MAX-TOKENS"):
        R.load_regen_raw_overlay(
            "root_x", "scale05", "test_1000", REGEN_CAP, tmp_path / "sc", local_dir=regen
        )


# ---------------------------------------------------------------------------
# 6. Analysis post-regen phase A (real body; _iter_raw_rows is the network
#    boundary — replaced by a signature-conformant fake)
# ---------------------------------------------------------------------------


def _fake_iter_rows_factory(rows_by_split):
    def _fake_iter_raw_rows(hf_prefix: str, split: str, scratch: Path):
        yield from rows_by_split.get(split, [])

    return _fake_iter_raw_rows


def test_phase_a_post_regen_stats_and_residual_masks(tmp_path, monkeypatch):
    # 3 rows per split: ci 1 cap-hit + regenerated-to-stop, ci 2 cap-hit +
    # residual (still length at the regen cap), ci 0 clean.
    base_rows = [
        (0, "stop", "aa", "q0"),
        (1, "length", "bbbb", "q1"),
        (2, "length", "cccc", "q2"),
    ]
    rows_by_split = {s: list(base_rows) for s in A.RAW_SPLITS}
    monkeypatch.setattr(A, "_iter_raw_rows", _fake_iter_rows_factory(rows_by_split))
    overlay = {
        1: {"response": "bbbb now complete", "finish_reason": "stop", "prompt": "q1"},
        2: {"response": "cccc still going", "finish_reason": "length", "prompt": "q2"},
    }
    regen_ctx = {
        "cap": REGEN_CAP,
        "raw_overlays": {s: dict(overlay) for s in A.RAW_SPLITS},
        "applied": {s: {1, 2} for s in A.RAW_SPLITS},
    }
    out = A.phase_a_caphit("scale05", "pfx", tmp_path, set(), regen_ctx=regen_ctx)
    st = out["splits"]["test_1000"]
    assert st["pre"]["n_cap_hit"] == 2 and st["post"]["n_cap_hit"] == 1
    assert st["n_regen_applied"] == 2 and st["n_caphit_not_regenerated"] == 0
    assert st["post"]["response_chars_mean"] > st["pre"]["response_chars_mean"]
    # Residual mask: only ci 2 still cap-hit in the merged view.
    assert out["masks"]["test_1000"] == {0: False, 1: False, 2: True}


def test_phase_a_post_regen_refuses_overlay_on_noncaphit_row(tmp_path, monkeypatch):
    rows_by_split = {s: [(0, "stop", "aa", "q0")] for s in A.RAW_SPLITS}
    monkeypatch.setattr(A, "_iter_raw_rows", _fake_iter_rows_factory(rows_by_split))
    regen_ctx = {
        "cap": REGEN_CAP,
        "raw_overlays": {
            s: {0: {"response": "x", "finish_reason": "stop", "prompt": "q0"}} for s in A.RAW_SPLITS
        },
        "applied": {s: {0} for s in A.RAW_SPLITS},
    }
    with pytest.raises(AssertionError, match="NOT cap-hit"):
        A.phase_a_caphit("scale05", "pfx", tmp_path, set(), regen_ctx=regen_ctx)


def test_phase_a_base_mode_shape_unchanged(tmp_path, monkeypatch):
    """Default (no regen_ctx) output keeps the committed base-round shape."""
    rows_by_split = {s: [(0, "stop", "aa", "q0"), (1, "length", "bb", "q1")] for s in A.RAW_SPLITS}
    monkeypatch.setattr(A, "_iter_raw_rows", _fake_iter_rows_factory(rows_by_split))
    out = A.phase_a_caphit("scale05", "pfx", tmp_path, set())
    st = out["splits"]["test_1000"]
    assert st["n_cap_hit"] == 1 and "pre" not in st
    assert out["masks"]["test_1000"] == {0: False, 1: True}
