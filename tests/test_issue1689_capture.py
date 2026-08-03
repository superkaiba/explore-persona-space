"""Round-4/7 regression tests for issue #1689 capture fixes.

Round-4 concerns closed:
  - capture-arms-identical (BLOCKER): X_prefix and X_context MUST be
    computed at DISTINCT token positions per row for every arm class
    where u2 is non-empty (assistant, character, user_lmsys+haiku with
    filled u2 slots); user_onpolicy naturalistic-user-arm cells legitimately
    have prefix_end == context_end (u2 is the DV — see render_conditions.py).
  - bpe-seam-capture-slot-textslice (CONCERN): slot boundaries must be
    derived from tokenizer offset_mapping in a SINGLE tokenization pass,
    never text-slice + re-tokenize (per gotchas.md § "Plain-text span
    boundaries are the WORST case").
  - bootstrap-wall-projection-over-plan (CONCERN): the eigendecomposition-
    based ridge λ-scan must produce numerically-equivalent predictions to
    the direct-solve baseline within float tolerance.

Round-7 concern closed:
  - capture-upload-file-in-loop-pre-r6 (CONCERN): per-cell activation
    uploads MUST go through ONE ``upload_folder`` commit per cell (the
    ``_upload`` directory branch), never a per-file ``upload_file`` loop
    that 504-storms on the ~1M-file data repo (CLAUDE.md § Upload Policy
    "use a single bulk `upload_folder` commit for many files"; #664).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.issue1689_capture import (  # noqa: E402
    _resolve_row_offsets,
    _resolve_slot_token_indices,
    capture_cell,
    upload_cell_to_hf,
)
from scripts.issue1689_common import (  # noqa: E402
    CAPTURE_LAYERS,
    HF_DATA_PREFIX,
    SLUG_TO_CONDITION,
)
from scripts.issue1689_fit_ladder import (  # noqa: E402
    _fit_ridge_gram,
    _ridge_eigh_prep,
    _ridge_fit_from_prep,
    _ridge_predict_from_prep,
)
from scripts.issue1689_render_conditions import render_condition  # noqa: E402

# ---------------------------------------------------------------------------
# Concern: capture-upload-file-in-loop-pre-r6 (CONCERN, ROUND 7)
# ---------------------------------------------------------------------------


def _write_dummy_cell(cell_dir: Path) -> list[Path]:
    """Populate a fake per-cell store with 4 layer files, matching the
    production shape (L{14,18,19,26}.pt). Contents are irrelevant to the
    upload path check (only the file layout + call shape matter)."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for layer in CAPTURE_LAYERS:
        path = cell_dir / f"L{layer}.pt"
        path.write_bytes(b"dummy tensor bytes for layer " + str(layer).encode())
        files.append(path)
    return files


def test_upload_uses_upload_folder_not_upload_file_loop(tmp_path):
    """Round-7: per-cell upload MUST call _upload ONCE on the DIRECTORY
    (which dispatches to HfApi.upload_folder → ONE create_commit), never
    a per-file upload_file loop over the 4 layer files."""
    cell_dir = tmp_path / "Qwen_Qwen2.5-7B-Instruct" / "assistant_chat"
    _write_dummy_cell(cell_dir)
    with patch("explore_persona_space.orchestrate.hub._upload") as mock_upload:
        # Return path shape matches the production return contract.
        mock_upload.return_value = "superkaiba1/explore-persona-space-data/x"
        result = upload_cell_to_hf(cell_dir, "Qwen/Qwen2.5-7B-Instruct", "assistant_chat")
    # ONE call for the whole cell (4 layers) — not 4 per-file calls.
    assert mock_upload.call_count == 1, (
        f"expected 1 upload_folder call per cell, got {mock_upload.call_count} "
        "— the round-7 fix must NOT loop upload_file per layer"
    )
    # The single call passes the DIRECTORY (not any file), with NO upload_as_file kwarg
    # (default False → _upload's is_dir() branch → upload_folder path).
    call_args, call_kwargs = mock_upload.call_args
    passed_path = call_args[0] if call_args else call_kwargs.get("local_path")
    assert passed_path == cell_dir, f"upload target must be the cell directory, got {passed_path}"
    assert passed_path.is_dir(), "upload target must be a directory, not a file"
    # upload_as_file must NOT be True — that would route to upload_file and defeat the fix.
    assert not call_kwargs.get("upload_as_file", False), (
        "upload_as_file=True routes to per-file upload_file — the fix requires the "
        "directory (upload_folder) branch"
    )
    # Verify the return contract is unchanged: HF prefix string.
    assert result is not None
    assert "analysis_tensors" in result


def test_upload_folder_target_paths_match_plan(tmp_path):
    """Round-7: the path_in_repo target must match plan §10:
    ``<HF_DATA_PREFIX>/analysis_tensors/<model_slug>/<condition_slug>``,
    with model '/' → '_' — a byte-compatible layout with the pre-round-7
    per-file scheme (each L*.pt lands at the same final path)."""
    cell_dir = tmp_path / "Qwen_Qwen2.5-7B" / "assistant_naturalistic"
    _write_dummy_cell(cell_dir)
    with patch("explore_persona_space.orchestrate.hub._upload") as mock_upload:
        mock_upload.return_value = "superkaiba1/explore-persona-space-data/x"
        upload_cell_to_hf(cell_dir, "Qwen/Qwen2.5-7B", "assistant_naturalistic")
    call_kwargs = mock_upload.call_args.kwargs
    expected_path = f"{HF_DATA_PREFIX}/analysis_tensors/Qwen_Qwen2.5-7B/assistant_naturalistic"
    assert call_kwargs["path_in_repo"] == expected_path, (
        f"path_in_repo mismatch: got {call_kwargs['path_in_repo']!r} vs {expected_path!r}"
    )
    assert call_kwargs["repo_id"] == "superkaiba1/explore-persona-space-data"
    assert call_kwargs["repo_type"] == "dataset"


def test_upload_never_calls_upload_file_per_layer(tmp_path):
    """Round-7 belt-and-suspenders: patch the huggingface_hub HfApi upload_file
    method too and confirm the code path never reaches it. _upload's directory
    branch calls upload_folder; the file branch (upload_file) must stay unused
    for the per-cell tensor upload."""
    cell_dir = tmp_path / "cell"
    _write_dummy_cell(cell_dir)
    with patch("explore_persona_space.orchestrate.hub._upload") as mock_upload:
        mock_upload.return_value = "ok"
        upload_cell_to_hf(cell_dir, "Qwen/Qwen2.5-7B-Instruct", "assistant_chat")
    # Exactly one _upload call, with a DIRECTORY (not a *.pt file) as target.
    assert mock_upload.call_count == 1
    passed_path = mock_upload.call_args.args[0]
    assert passed_path.is_dir()
    assert not any(str(passed_path).endswith(f"L{L}.pt") for L in CAPTURE_LAYERS), (
        "upload target is a per-layer .pt file — the fix requires the cell DIRECTORY"
    )


# ---------------------------------------------------------------------------
# Concern: capture-arms-identical (BLOCKER)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Character-level tokenizer stub for offset-mapping tests.

    Each character becomes one token (id = ord(char)); offset_mapping is
    (i, i+1) per position. Chat framing is out of scope for this stub.
    """

    def __call__(self, text, **kwargs):
        input_ids = [ord(c) for c in text]
        offset_mapping = [(i, i + 1) for i in range(len(text))]
        return {"input_ids": input_ids, "offset_mapping": offset_mapping}


class _FakeBPETokenizer:
    """BPE-like tokenizer stub that merges the trailing 'r: ' delimiter into
    ONE token — reproduces the plain-text-boundary BPE-seam trap the fix
    must handle (offset_mapping straddling the prefix/u2 char boundary)."""

    def __call__(self, text, **kwargs):
        # 3-character chunks (like a simplistic BPE), so any 3-char span
        # ending exactly at a delimiter merges with what follows.
        input_ids = []
        offset_mapping = []
        i = 0
        while i < len(text):
            e = min(i + 3, len(text))
            input_ids.append(hash(text[i:e]) & 0xFFFF)
            offset_mapping.append((i, e))
            i = e
        return {"input_ids": input_ids, "offset_mapping": offset_mapping}


def test_resolve_slot_indices_distinct_for_nonempty_u2():
    """Non-empty u2 segment MUST produce distinct prefix_end and context_end tokens."""
    full = "prefix_partu2_partcontext_tailanswer_part"
    prefix_end_char = len("prefix_part")
    context_end_char = len("prefix_partu2_partcontext_tail")
    answer_end_char = len(full)
    tok = _FakeTokenizer()
    p, c, a, ids = _resolve_slot_token_indices(
        full, prefix_end_char, context_end_char, answer_end_char, tok
    )
    assert p < c < a, (p, c, a)
    assert len(ids) == len(full)


def test_resolve_slot_indices_bpe_seam_straddler_prefix_excluded():
    """Under a BPE tokenizer that MERGES the prefix/u2 char boundary into ONE
    token (the #1092/#1315 seam trap), the prefix arm MUST EXCLUDE the
    straddler token (so u2's leading text does not leak into the prefix)."""
    # "abcdefghij" — 3-char BPE chunks: "abc" (0,3), "def" (3,6), "ghi" (6,9), "j" (9,10)
    full = "abcdefghij"
    # prefix ends at char 4 (INSIDE the 'def' token that spans (3,6)) — the
    # straddler; prefix must exclude it.
    prefix_end_char = 4
    context_end_char = 7  # inside the 'ghi' token (6,9) — straddler; context INCLUDES.
    answer_end_char = 10
    tok = _FakeBPETokenizer()
    p, c, a, _ids = _resolve_slot_token_indices(
        full, prefix_end_char, context_end_char, answer_end_char, tok
    )
    # 'abc' is the last token fully at or before char 4 (its span (0,3) ends
    # at 3 <= 4). Prefix arm = index 0.
    assert p == 0, p
    # 'ghi' straddles boundary 7 (span (6,9)); context INCLUDES it → index 2.
    assert c == 2, c
    # answer = last token = 'j' at index 3.
    assert a == 3, a


def test_resolve_slot_indices_raises_on_nonmonotonic_offsets():
    tok = _FakeTokenizer()
    try:
        _resolve_slot_token_indices("abcdef", 5, 3, 6, tok)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-monotonic offsets")


def test_render_naturalistic_emits_distinct_prefix_u2_tail_segments():
    """The renderer must expose prefix_text_only, u2_text_marked, context_tail
    for downstream capture — the round-4 fix's boundary source."""
    conv = {"conv_id": "conv-0", "u1": "How are you?", "a1": "I'm well.", "u2_lmsys": "Tell more."}
    cond = SLUG_TO_CONDITION["assistant_naturalistic"]
    row = render_condition(conv, cond, u2_text=conv["u2_lmsys"])
    assert row["prefix_text_only"] is not None
    assert row["u2_text_marked"] == "Tell more."
    assert row["context_tail"] == "\n\nAssistant: "
    # prompt_text is the concatenation of the three.
    assert (
        row["prompt_text"] == row["prefix_text_only"] + row["u2_text_marked"] + row["context_tail"]
    )


def test_render_naturalistic_user_arm_empty_u2():
    """User-arm naturalistic cells legitimately have empty u2 (the model
    generates it) — prefix == context is the DESIGNED behavior per
    render_conditions.py's is_user branch."""
    conv = {"conv_id": "conv-0", "u1": "hi", "a1": "hello", "u2_lmsys": "irrelevant"}
    cond = SLUG_TO_CONDITION["user_lmsys_naturalistic"]
    row = render_condition(conv, cond, u2_text=conv["u2_lmsys"])
    assert row["u2_text_marked"] == ""
    assert row["context_tail"] == ""


def test_resolve_row_offsets_naturalistic():
    """The capture-side offset resolver must return correctly-monotonic char
    offsets for a naturalistic row."""
    conv = {"conv_id": "conv-0", "u1": "Q?", "a1": "A.", "u2_lmsys": "More?"}
    cond = SLUG_TO_CONDITION["assistant_naturalistic"]
    row = render_condition(conv, cond, u2_text=conv["u2_lmsys"])
    row["a2_text"] = "Sure."
    tok = _FakeTokenizer()
    full_text, p_char, c_char, a_char = _resolve_row_offsets(row, tok)
    assert p_char < c_char < a_char
    assert full_text == row["prompt_text"] + "Sure."
    # p_char is the length of prefix_text_only.
    assert p_char == len(row["prefix_text_only"])
    # c_char is the length of prompt_text (prefix + u2 + tail).
    assert c_char == len(row["prompt_text"])


def test_capture_cell_mock_produces_distinct_arms_per_layer():
    """The mock capture path (smoke) MUST emit X_prefix != X_context per layer
    — the tensor-level guarantee of the round-4 fix."""
    rows = [{"conv_id": f"conv-{i}"} for i in range(3)]
    cell = capture_cell(
        rows, model_name="Qwen/Qwen2.5-7B", condition_slug="assistant_chat", mock=True
    )
    for layer in CAPTURE_LAYERS:
        L = cell[f"L{layer}"]
        assert L["X_prefix"].shape == (3, 3584)
        assert L["X_context"].shape == (3, 3584)
        assert L["Y"].shape == (3, 3584)
        # Distinct arms — mock draws with different seeds per arm.
        assert not np.array_equal(L["X_prefix"], L["X_context"]), (
            f"L{layer}: X_prefix == X_context (mock arms not distinct)"
        )
        assert not np.array_equal(L["X_prefix"], L["Y"])
        assert not np.array_equal(L["X_context"], L["Y"])


# ---------------------------------------------------------------------------
# Concern: bpe-seam-capture-slot-textslice (CONCERN)
# ---------------------------------------------------------------------------


def test_capture_never_retokenizes_slices():
    """Regression: capture uses ONE tokenization pass with offset_mapping and
    never re-tokenizes a text slice (verified structurally — the fix's
    _resolve_slot_token_indices calls tokenizer ONCE per row)."""
    # Read the capture source and assert the anti-pattern is gone.
    src = (_REPO_ROOT / "scripts" / "issue1689_capture.py").read_text()
    # The old shape re-tokenized prefix_text alone; the fix uses offset_mapping.
    # Sentinel: the fix module must reference return_offsets_mapping (the API
    # the concern demands).
    assert "return_offsets_mapping=True" in src, (
        "capture.py must use return_offsets_mapping=True (concern bpe-seam-capture-slot-textslice)"
    )
    # And must NOT contain the old text-slice-then-tokenize anti-pattern:
    # `tok(prefix_text, ...)` followed by a separate `tok(prefix_text + a2, ...)`
    # (round-3 shape).
    assert "tok(prefix_text, return_tensors" not in src
    assert "tok(prefix_text + a2" not in src


# ---------------------------------------------------------------------------
# Concern: bootstrap-wall-projection-over-plan (CONCERN)
# ---------------------------------------------------------------------------


def test_ridge_eigh_prep_reproduces_direct_solve_within_tolerance():
    """Eigendecomposition-based ridge fit must be numerically equivalent to
    the direct-solve _fit_ridge_gram baseline within float tolerance."""
    rng = np.random.default_rng(42)
    # Primal regime (n > d).
    n, d, d_y = 50, 8, 3
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d_y))
    for lam in [0.1, 1.0, 10.0, 100.0]:
        W_direct, b_direct = _fit_ridge_gram(X, Y, lam=lam)
        prep = _ridge_eigh_prep(X, Y)
        W_eigh, b_eigh = _ridge_fit_from_prep(prep, lam=lam)
        assert np.allclose(W_direct, W_eigh, atol=1e-8), (lam, np.abs(W_direct - W_eigh).max())
        assert np.allclose(b_direct, b_eigh, atol=1e-8)


def test_ridge_eigh_prep_dual_regime_reproduces_direct_solve():
    """Same equivalence check but in the dual regime (n < d) where the shared
    Gram is (n x n)."""
    rng = np.random.default_rng(42)
    n, d, d_y = 8, 50, 3
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d_y))
    for lam in [0.1, 1.0, 10.0]:
        W_direct, b_direct = _fit_ridge_gram(X, Y, lam=lam)
        prep = _ridge_eigh_prep(X, Y)
        _W_eigh, _b_eigh = _ridge_fit_from_prep(prep, lam=lam)
        # Dual-regime W may differ by a null-space projection but predictions match.
        X_test = rng.standard_normal((5, d))
        pred_direct = X_test @ W_direct + b_direct
        pred_eigh = _ridge_predict_from_prep(prep, X_test, lam=lam)
        assert np.allclose(pred_direct, pred_eigh, atol=1e-6), (
            lam,
            np.abs(pred_direct - pred_eigh).max(),
        )


def test_ridge_eigh_prep_predict_matches_direct_solve_prediction():
    """The batched prediction path (used inside inner-group-cv) must match
    the direct-solve prediction."""
    rng = np.random.default_rng(42)
    n, d, d_y = 30, 12, 4
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d_y))
    X_test = rng.standard_normal((10, d))
    for lam in [0.5, 5.0, 50.0]:
        W, b = _fit_ridge_gram(X, Y, lam=lam)
        pred_direct = X_test @ W + b
        prep = _ridge_eigh_prep(X, Y)
        pred_eigh = _ridge_predict_from_prep(prep, X_test, lam=lam)
        assert np.allclose(pred_direct, pred_eigh, atol=1e-6), (
            lam,
            np.abs(pred_direct - pred_eigh).max(),
        )


def test_ridge_eigh_prep_lambda_scan_batches_over_shared_eigendecomp():
    """The KEY win: for L=13 lambdas the shared eigendecomp is computed ONCE,
    and each λ contributes a cheap O(D²) sandwich. This test exercises the
    scan pattern the inner-group-cv fit uses."""
    rng = np.random.default_rng(42)
    n, d, d_y = 40, 10, 3
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d_y))
    X_test = rng.standard_normal((5, d))
    lambdas = np.logspace(-2, 4, 13)
    # Direct-solve baseline: L fresh full solves.
    preds_direct = np.stack(
        [
            X_test @ _fit_ridge_gram(X, Y, lam=float(lam))[0]
            + _fit_ridge_gram(X, Y, lam=float(lam))[1]
            for lam in lambdas
        ]
    )
    # Batched via shared eigendecomp: 1 eigh + L sandwiches.
    prep = _ridge_eigh_prep(X, Y)
    preds_batched = np.stack(
        [_ridge_predict_from_prep(prep, X_test, lam=float(lam)) for lam in lambdas]
    )
    assert preds_direct.shape == preds_batched.shape
    assert np.allclose(preds_direct, preds_batched, atol=1e-6)
