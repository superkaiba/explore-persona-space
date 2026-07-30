"""#1773 full-dictionary mode — equivalence + group-sharding pins.

The full-dictionary phase-0 path REIMPLEMENTS two GEMM helpers (blocked
top-k neighbours, streamed logit footprint) so the 131,072-feature Gram is
never materialized; these tests assert the blocked forms reproduce the
committed restricted-path numpy helpers exactly. The remaining tests pin the
feature-group sharding that makes the 3.28M-item axes dispatch expressible at
all, and the rejection non-activating draw used by Pass A at full dictionary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture(scope="module")
def phase0():
    return pytest.importorskip("issue1773_phase0_mechanical")


@pytest.fixture(scope="module")
def evidence():
    return pytest.importorskip("issue1773_evidence_builder")


@pytest.fixture(scope="module")
def describe():
    return pytest.importorskip("issue1773_describe_axes")


# ── blocked-vs-serial equivalence (the batched rewrite) ──────────────────────


def test_blocked_neighbours_match_numpy_helper(phase0):
    """Blocked top-k cosine == the restricted-path numpy neighbour table.

    Chunk size is forced BELOW the feature count so the blocking path (and its
    self-exclusion at the block offset) actually executes — a chunk >= n would
    silently degenerate to one block and prove nothing.
    """
    pytest.importorskip("torch")
    rng = np.random.default_rng(1773)
    w = rng.standard_normal((64, 37)).astype(np.float32)

    ref_idx, ref_cos = phase0.neighbor_table(w.astype(np.float64))
    got_idx, got_cos = phase0.neighbor_table_blocked(w, "cpu", chunk=8)

    assert got_idx.shape == ref_idx.shape
    # Compare as (neighbour, cosine) sets per row: ties can order differently
    # across argpartition and topk, but the selected SET and its values must match.
    for i in range(w.shape[1]):
        np.testing.assert_allclose(np.sort(got_cos[i]), np.sort(ref_cos[i]), atol=1e-6)
        assert set(got_idx[i].tolist()) == set(ref_idx[i].tolist())
        assert i not in got_idx[i].tolist(), "self must be excluded from its own neighbours"


def test_footprint_blocks_match_numpy_helper(phase0):
    """Streamed footprint blocks == the restricted-path `logit_footprint` rows."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(17730)
    vocab, d_model, n_feat = 53, 64, 21
    w_u = rng.standard_normal((vocab, d_model)).astype(np.float32)
    gamma = rng.standard_normal(d_model).astype(np.float32)
    w_dec = rng.standard_normal((d_model, n_feat)).astype(np.float32)
    vocab_str = [f"<t{i}>" for i in range(vocab)]

    class _Tok:
        def decode(self, ids):
            return "".join(vocab_str[i] for i in ids)

    ref = phase0.logit_footprint(
        w_u.astype(np.float64), gamma.astype(np.float64), w_dec.astype(np.float64), _Tok()
    )
    got: list[dict] = []
    for _start, block in phase0.footprint_blocks(w_u, gamma, w_dec, vocab_str, "cpu", chunk=5):
        got.extend(block)

    assert len(got) == len(ref) == n_feat
    for a, b in zip(got, ref, strict=True):
        assert a["top_promoted_ids"] == b["top_promoted_ids"]
        assert a["top_suppressed_ids"] == b["top_suppressed_ids"]
        assert a["top_promoted_tokens"] == b["top_promoted_tokens"]
        np.testing.assert_allclose(a["top_promoted_vals"], b["top_promoted_vals"], atol=1e-3)
        np.testing.assert_allclose(a["top_suppressed_vals"], b["top_suppressed_vals"], atol=1e-3)
        assert a["concentration"] == pytest.approx(b["concentration"], abs=1e-5)


# ── Pass A / Pass B full-dictionary helpers ─────────────────────────────────


def test_rejection_nonact_draw_excludes_active_and_is_unique(evidence):
    rng = np.random.default_rng(7)
    uniq = np.arange(5000, dtype=np.int64)
    active = np.arange(0, 400, dtype=np.int64)
    got = evidence._draw_nonact_rejection(uniq, active, 40, rng)
    assert len(got) == 40
    assert len(set(got.tolist())) == 40, "draw must be without replacement"
    assert not set(got.tolist()) & set(active.tolist()), "active rows must be excluded"
    assert set(got.tolist()) <= set(uniq.tolist())


def test_rejection_nonact_draw_falls_back_on_tiny_pool(evidence):
    """A pool smaller than the quota must degrade to the exact complement draw,
    not spin the rejection budget and return short of an achievable count."""
    rng = np.random.default_rng(11)
    uniq = np.arange(45, dtype=np.int64)
    active = np.arange(0, 40, dtype=np.int64)  # complement is exactly 5 rows
    got = evidence._draw_nonact_rejection(uniq, active, 40, rng)
    assert sorted(got.tolist()) == list(range(40, 45))


def test_row_index_matches_dict_of_lists(evidence):
    """The searchsorted row index reproduces the dict-of-lists it replaced."""
    rows = np.array([0, 0, 0, 2, 2, 5, 9, 9, 9, 9], dtype=np.int64)
    ref: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        ref.setdefault(int(r), []).append(i)
    idx = evidence._RowIndex(rows)
    for r in range(12):
        assert list(idx.get(r, [])) == ref.get(r, []), r


def _write_assemble_fixture(tmp_path: Path, n_feat: int = 3):
    """Minimal Pass-A selection + Pass-B window files for a Pass-C join."""
    import json

    import issue1773_common as CM

    sel_dir = tmp_path / "sel"
    win_dir = tmp_path / "win"
    sel_dir.mkdir()
    win_dir.mkdir()
    sel_rows = []
    win_rows = []
    for f in range(n_feat):
        sel_rows.append(
            {
                "feat_id": f,
                "restricted_idx": f,
                "act": [],
                "nonact_candidates": [],
                "act_short": True,
            }
        )
        for j in range(CM.N_ACT_EVIDENCE + 2):
            win_rows.append(
                {
                    "feat_id": f,
                    "kind": "act",
                    "bin": j % CM.N_ACT_BINS,
                    "split": 0 if j < CM.N_ACT_EVIDENCE else 1,
                    "ci": 100 + j,
                    "peak_val": float(j),
                    "window": {
                        "text_marked": f"f{f} w{j} <<tok>> tail",
                        "text_plain": f"f{f} w{j} tok tail",
                        "token_lo": 0,
                        "token_hi": 4,
                        "peak_pos": 2,
                        "values_fp16": [float(k) for k in range(32)],
                    },
                }
            )
        for j in range(CM.N_NONACT_EVIDENCE + CM.N_NONACT_HOLDOUT):
            win_rows.append(
                {
                    "feat_id": f,
                    "kind": "nonact",
                    "order": j,
                    "ci": 900 + j,
                    "window": {
                        "text_marked": f"f{f} neg{j}",
                        "text_plain": f"f{f} neg{j}",
                        "token_lo": 0,
                        "token_hi": 3,
                        "peak_pos": 0,
                        "values_fp16": [0.0] * 32,
                    },
                }
            )
    with (sel_dir / "selection.shard00.jsonl").open("w") as fh:
        for r in sel_rows:
            fh.write(json.dumps(r) + "\n")
    with (win_dir / "windows_chunk0.jsonl").open("w") as fh:
        for r in win_rows:
            fh.write(json.dumps(r) + "\n")
    return sel_dir, win_dir


@pytest.mark.parametrize("full_dictionary", [False, True])
def test_assemble_drops_window_values_only_in_full_dictionary(evidence, tmp_path, full_dictionary):
    """`values_fp16` is never rendered into a prompt, so full-dictionary mode
    drops it at join time (the biggest memory/manifest lever at 13.1M records);
    the restricted path must keep the committed 16k artifact schema untouched."""
    from types import SimpleNamespace

    sel_dir, win_dir = _write_assemble_fixture(tmp_path)
    ev_dir = tmp_path / f"evidence_{int(full_dictionary)}"
    args = SimpleNamespace(
        selection_dir=sel_dir,
        out_dir=win_dir,
        evidence_dir=ev_dir,
        phase0_dir=tmp_path / "nonexistent_phase0",
        fetch_missing=False,
        no_upload=True,
        full_dictionary=full_dictionary,
    )
    rc = evidence.pass_assemble(args)
    assert rc == 0

    import issue1773_common as CM

    packs = []
    for p in sorted((ev_dir / "evidence_manifests").glob("evidence.shard*.jsonl")):
        packs.extend(CM.iter_jsonl(p))
    assert packs, "assemble produced no packets"
    has_values = any("values_fp16" in w for pk in packs for w in (pk["ex_pos"] + pk["ex_neg"]))
    assert has_values is (not full_dictionary), (
        "full-dictionary mode must drop per-token values; "
        "the restricted path must retain them (schema parity with the 16k run)"
    )
    # Either way the rendered prompt text is unaffected — the judge never saw values.
    msg = CM.build_describe_user_msg(packs[0])
    assert "values_fp16" not in msg and "<<tok>>" in msg


# ── phase 2/3 feature-group sharding ────────────────────────────────────────


def test_feature_groups_partition_exactly(describe):
    feats = list(range(1000))
    groups = describe.feature_groups(feats, 256)
    assert len(groups) == 4
    assert [len(g) for g in groups] == [256, 256, 256, 232]
    flat = [f for g in groups for f in g]
    assert flat == feats, "groups must partition the id list in order, no drops or dupes"


def test_feature_groups_single_when_unsharded(describe):
    feats = list(range(10))
    assert describe.feature_groups(feats, 0) == [feats]
    assert describe.feature_groups(feats, 100) == [feats]


def test_full_dictionary_axes_group_count_is_dispatchable(describe):
    """The default axes group must stay well under the batch queue cap.

    3,276,800 items in ONE dispatch is the shape this sharding exists to
    avoid; each group must be a size `dispatch_judge_items` can actually hold.
    """
    import issue1773_common as CM

    n_feat = CM.DICT_SIZE
    per_feature = len(CM.AXES) * CM.N_DRAWS
    assert per_feature == 25
    groups = describe.feature_groups(list(range(n_feat)), describe.AXES_GROUP_FEATURES)
    items_per_group = describe.AXES_GROUP_FEATURES * per_feature
    assert items_per_group == 102_400
    assert len(groups) == 32
    assert sum(len(g) for g in groups) == n_feat
    # in-flight requests per group stay under the Tier-4 batch queue floor (500k)
    assert items_per_group < 500_000
