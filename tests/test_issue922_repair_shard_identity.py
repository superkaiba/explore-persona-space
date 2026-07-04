"""Issue #922 repair: stale-shard content-identity regression tests.

Pins the fix for concern ``eval-repaired-response-identity-unchecked``: a
shard captured from completions A must NOT be silently reused
(``validate_shard``) nor scored (``assert_gen_capture_identity``) against
completions B that share the ci set AND per-window token counts but differ in
content — the exact blind spot of the pre-fix (ci-set + ans_len) keys. Also
pins the no-silent-grandfathering rule (a pre-hash shard is INVALID under an
expected-hash validation) and that the legacy lmsys/eval_subset path
(``expected_hashes=None``) is unchanged.

Fails pre-fix (``validate_shard`` had no ``expected_hashes`` leg;
``assert_gen_capture_identity`` did not exist) and passes post-fix.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue922_capture_positions as cap
import issue922_repair_provenance as rep


class _StubTok:
    """Deterministic tokenizer stub: template = JSON dump; tokens = words."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return json.dumps(messages) + "<|assistant|>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def _items(responses, qis=None):
    return [
        {
            "ci": i,
            "trait": "sycophancy",
            "cond_id": "sys0",
            "mode": "system",
            "qi": i if qis is None else qis[i],
            "messages": [{"role": "user", "content": f"q{i}"}],
            "response": r,
        }
        for i, r in enumerate(responses)
    ]


ITEMS_A = _items(["alpha beta gamma", "delta epsilon zeta"])
# SAME ci set, SAME per-window token count (3 stub tokens each), DIFFERENT
# content — invisible to the pre-fix ci-set + ans_len keys.
ITEMS_B = _items(["alpha beta GAMMA", "delta epsilon ZETA"])
BLOCKS = ["emb", "0"]
WINDOW = {"wp": 8, "wa": 40}
HIDDEN = 8


def _hashes(items, tok):
    return {int(it["ci"]): cap.content_hashes(tok, it) for it in items}


def _write_shard(path, items, tok, *, with_hashes=True):
    contexts = {}
    for it in items:
        prompt_sha, resp_sha = cap.content_hashes(tok, it)
        rec = {
            "h": torch.zeros(4, len(BLOCKS), HIDDEN, dtype=torch.float16),
            "token_ids": torch.zeros(4, dtype=torch.int32),
            "segments": np.zeros(3, dtype=np.uint8),
            "prompt_len": 2,
            "ans_len": len(it["response"].split()),
            "window_start": 0,
            "trait": it["trait"],
            "cond_id": it["cond_id"],
            "mode": it["mode"],
            "qi": it["qi"],
            "question_provenance": "regenerated",
            "response_provenance": "fresh_onpolicy",
        }
        if with_hashes:
            rec["prompt_sha256"] = prompt_sha
            rec["response_sha256"] = resp_sha
        contexts[int(it["ci"])] = rec
    torch.save(
        {"corpus": "eval_repaired", "blocks": BLOCKS, "contexts": contexts, "window": WINDOW},
        path,
    )


def _validate(path, items, hashes):
    return cap.validate_shard(
        path,
        corpus="eval_repaired",
        expected_cis={int(it["ci"]) for it in items},
        wp=WINDOW["wp"],
        wa=WINDOW["wa"],
        labels=BLOCKS,
        expected_hidden=HIDDEN,
        expected_hashes=hashes,
    )


def test_matching_hashes_validate_ok(tmp_path):
    tok = _StubTok()
    p = tmp_path / "shard_000.pt"
    _write_shard(p, ITEMS_A, tok)
    blob, why = _validate(p, ITEMS_A, _hashes(ITEMS_A, tok))
    assert blob is not None and why == "ok"


def test_same_length_different_content_is_recaptured(tmp_path):
    """Shard from A vs CURRENT items B (same ci set + token counts) → INVALID."""
    tok = _StubTok()
    p = tmp_path / "shard_000.pt"
    _write_shard(p, ITEMS_A, tok)
    blob, why = _validate(p, ITEMS_B, _hashes(ITEMS_B, tok))
    assert blob is None
    assert "response_sha256 mismatch" in why


def test_prompt_drift_is_recaptured(tmp_path):
    """A changed prompt (same response) flips prompt_sha256 → INVALID."""
    tok = _StubTok()
    p = tmp_path / "shard_000.pt"
    _write_shard(p, ITEMS_A, tok)
    drifted = [dict(it, messages=[{"role": "user", "content": "OTHER"}]) for it in ITEMS_A]
    blob, why = _validate(p, drifted, _hashes(drifted, tok))
    assert blob is None
    assert "prompt_sha256 mismatch" in why


def test_pre_hash_shard_is_invalid_no_grandfathering(tmp_path):
    tok = _StubTok()
    p = tmp_path / "shard_000.pt"
    _write_shard(p, ITEMS_A, tok, with_hashes=False)
    blob, why = _validate(p, ITEMS_A, _hashes(ITEMS_A, tok))
    assert blob is None
    assert "missing content-hash" in why


def test_no_expected_hashes_keeps_legacy_path_valid(tmp_path):
    """lmsys/eval_subset callers (expected_hashes=None): pre-hash shard stays valid."""
    tok = _StubTok()
    p = tmp_path / "shard_000.pt"
    _write_shard(p, ITEMS_A, tok, with_hashes=False)
    blob, why = _validate(p, ITEMS_A, None)
    assert blob is not None and why == "ok"


def _score_meta(items, tok):
    meta = {}
    for it in items:
        prompt_sha, resp_sha = cap.content_hashes(tok, it)
        meta[int(it["ci"])] = {
            "trait": it["trait"],
            "cond_id": it["cond_id"],
            "qi": it["qi"],
            "ans_len": len(it["response"].split()),
            "prompt_sha256": prompt_sha,
            "response_sha256": resp_sha,
        }
    return meta


def test_score_side_identity_passes_on_consistent_pair():
    tok = _StubTok()
    rep.assert_gen_capture_identity(ITEMS_A, _score_meta(ITEMS_A, tok), tok)


def test_score_side_content_desync_raises_naming_ci():
    """Same-length different-content current items → AssertionError naming the ci."""
    tok = _StubTok()
    meta = _score_meta(ITEMS_A, tok)
    with pytest.raises(AssertionError) as ei:
        rep.assert_gen_capture_identity(ITEMS_B, meta, tok)
    msg = str(ei.value)
    assert "response-content desync" in msg and "0" in msg


def test_score_side_window_key_desync_raises():
    tok = _StubTok()
    meta = _score_meta(ITEMS_A, tok)
    swapped = _items([it["response"] for it in ITEMS_A], qis=[7, 8])
    with pytest.raises(AssertionError) as ei:
        rep.assert_gen_capture_identity(swapped, meta, tok)
    assert "window-key desync" in str(ei.value)
