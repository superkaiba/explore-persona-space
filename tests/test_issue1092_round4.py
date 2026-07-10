"""Issue #1092 regression pins for pca48 r_B, shard order, dynamics, and nulls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue1092_fit_grid as fit_grid  # noqa: E402
import issue1092_gpu_phase as gpu_phase  # noqa: E402
import issue1092_judge as judge  # noqa: E402
from issue923_fit_decomposition import press_fit_predict  # noqa: E402


class SeamMergingTokenizer:
    """Tiny tokenizer stub with BPE-like role/content and content/suffix seams."""

    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool = False, **kwargs):
        assert add_special_tokens is False
        ids, offsets = self._tokenize(text)
        out = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            out["offset_mapping"] = offsets
        return out

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        ids, _offsets = self._tokenize(text)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return "".join(self._pieces[int(i)] for i in ids)

    def _tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        pieces: list[str] = []
        offsets: list[tuple[int, int]] = []
        i = 0
        while i < len(text):
            if text.startswith(": ", i):
                j = i + 2
                while j < len(text) and not text[j].isspace():
                    j += 1
                pieces.append(text[i:j])
                offsets.append((i, j))
                i = j
                continue
            if text.startswith(".\n\n", i):
                pieces.append(text[i : i + 3])
                offsets.append((i, i + 3))
                i += 3
                continue
            if text[i].isspace():
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
            else:
                j = i + 1
                while j < len(text) and not text[j].isspace() and not text.startswith(": ", j):
                    if text.startswith(".\n\n", j):
                        break
                    j += 1
            pieces.append(text[i:j])
            offsets.append((i, j))
            i = j
        self._pieces = pieces
        return list(range(len(pieces))), offsets


def test_pca48_projects_rb_for_hidden_dim_gt_48(tmp_path):
    rng = np.random.default_rng(1092)
    hidden_dim = 64
    Y = rng.normal(size=(72, hidden_dim))
    Yb, basis_info = fit_grid._basis_targets_with_info(
        Y,
        "pca48",
        hidden_dim=hidden_dim,
        targets=["t1"],
        projection_target="t1",
    )
    rb = rng.normal(size=hidden_dim)

    rb_b = fit_grid._project_rb_to_basis(rb, basis_info, expected_dim=Yb.shape[1])
    assert Yb.shape == (72, 48)
    assert rb_b.shape == (48,)
    projection = Yb @ rb_b
    assert projection.shape == (72,)

    stacked = np.concatenate([Y, 2.0 * Y, -Y], axis=1)
    Ys, ambient_info = fit_grid._basis_targets_with_info(
        stacked,
        "ambient",
        hidden_dim=hidden_dim,
        targets=["t1", "t2", "t3"],
        projection_target="t1",
    )
    rb_s = fit_grid._project_rb_to_basis(rb, ambient_info, expected_dim=Ys.shape[1])
    assert rb_s.shape == (3 * hidden_dim,)
    assert np.count_nonzero(rb_s[hidden_dim:]) == 0
    assert (Ys @ rb_s).shape == (72,)


def test_numeric_12_shard_order_for_consolidation_and_fit_grid_loaders(tmp_path):
    cell = "cell_x"
    summary_dir = tmp_path / "summaries" / cell
    summary_dir.mkdir(parents=True)
    pool_dir = tmp_path / "summaries" / "b0_rB_pool"
    pool_dir.mkdir(parents=True)
    for shard in range(12):
        np.save(summary_dir / f"prefix_end_L00_shard{shard}.npy", np.array([[shard]]))
        np.save(pool_dir / f"{cell}_shard{shard}.npy", np.array([[[[shard]]]], dtype=np.float32))

    loaded, _paths = fit_grid._load_summary(tmp_path / "summaries", cell, "prefix_end", 0)
    assert loaded[:, 0].tolist() == list(range(12))
    b0_loaded = fit_grid._load_b0_pool(tmp_path / "summaries", cell)
    assert b0_loaded[:, 0, 0, 0].tolist() == list(range(12))

    root = tmp_path / "summaries" / "dynamics_instruct"
    root.mkdir()
    for shard in range(12):
        (root / f"row_index_u1_shard{shard}.jsonl").write_text(
            json.dumps({"conv_id": f"c{shard}", "turn_index": shard}) + "\n"
        )
    rows = fit_grid._read_index_files(root, "row_index_u1")
    assert [row["turn_index"] for row in rows] == list(range(12))

    gpu_phase.consolidate_cell_shards(tmp_path, cell, n_layers=1)
    consolidated = np.load(summary_dir / "prefix_end_L00.npy")
    assert consolidated[:, 0].tolist() == list(range(12))
    consolidated_b0 = np.load(pool_dir / f"{cell}.npy")
    assert consolidated_b0[:, 0, 0, 0].tolist() == list(range(12))


def test_mixed_padded_unpadded_duplicate_shards_fail_loud(tmp_path):
    paths = [
        tmp_path / "prefix_end_L00_shard3.npy",
        tmp_path / "prefix_end_L00_shard00003.npy",
    ]
    for path in paths:
        path.touch()

    with pytest.raises(ValueError, match="duplicate shard index 3"):
        fit_grid._sorted_shards(paths)
    with pytest.raises(ValueError, match="duplicate shard index 3"):
        gpu_phase._sorted_shards(paths)
    with pytest.raises(ValueError, match="duplicate shard index 3"):
        judge._sorted_shards(paths)


def test_pretrained_dynamics_cut_plan_uses_full_render_offsets_for_bpe_seams():
    tokenizer = SeamMergingTokenizer()
    turns = [
        {"role": "user", "content": "Explain photosynthesis briefly."},
        {"role": "assistant", "content": "Plants turn light into sugar."},
        {"role": "user", "content": "Name one input."},
        {"role": "assistant", "content": "Carbon dioxide."},
    ]
    full_render = gpu_phase._render_full_conversation(turns, "pretrained")
    encoded = tokenizer(full_render, add_special_tokens=False, return_offsets_mapping=True)

    cuts = gpu_phase._dynamics_cut_plan(
        turns,
        tokenizer,
        "pretrained",
        len(encoded["input_ids"]),
        full_token_ids=encoded["input_ids"],
    )

    first_user = cuts["u1"][0]
    second_user = cuts["u1"][1]
    assert first_user[2] == 0
    assert second_user[2] == 2
    first_text = tokenizer.decode(encoded["input_ids"][first_user[0] : first_user[1]])
    assert first_text.startswith(": Explain")
    assert first_text.endswith(".\n\n")
    assert "Explain photosynthesis briefly." in first_text
    assert "Name one input." in tokenizer.decode(
        encoded["input_ids"][second_user[0] : second_user[1]]
    )


def test_layer_max_null_uses_shared_draw_seed_across_layers(tmp_path):
    rng = np.random.default_rng(123)
    factors = {
        "f": rng.normal(size=(6, 4)),
        "g": rng.normal(size=(6, 4)),
        "i": rng.normal(size=(6, 4)),
        "basis": "dense_core",
    }
    rb = rng.normal(size=(28, 2, 4))
    basis_info = {
        "basis": "ambient",
        "ambient_dim": 4,
        "hidden_dim": 4,
        "targets": ["t1"],
        "projection_target": "t1",
        "projection_block_index": 0,
        "v_basis": None,
    }
    result = fit_grid._selection_symmetric_projection_null(
        unit_key="unit",
        factors=factors,
        rb_directions=rb,
        trait_names=["evil", "syc"],
        layer=3,
        basis_info=basis_info,
        n_draws=5,
        seed=77,
        out_dir=tmp_path,
    )
    draws = np.load(result["persist_path"])
    assert draws.shape == (5, 28, 3, 2)
    assert result["persist_shape"] == [5, 28, 3, 2]
    assert result["implementation"] == "sign_matrix_gemm"
    # Timing bounds belong in smoke evidence, not unit asserts (shared-VM flake
    # risk); pin presence/type/nonnegativity only.
    assert isinstance(result["wall_s"], float)
    assert result["wall_s"] >= 0.0

    manual_rng = np.random.default_rng(77)
    signs = manual_rng.choice(np.array([-1.0, 1.0]), size=(5, 6))
    arrays = [np.asarray(factors[name], dtype=np.float64) for name in ("f", "g", "i")]
    expected = np.empty_like(draws, dtype=np.float64)
    rb_flat = rb.reshape(28 * 2, 4)
    for factor_i, arr in enumerate(arrays):
        projected = (signs @ arr / arr.shape[0]) @ rb_flat.T
        expected[:, :, factor_i, :] = np.abs(projected.reshape(5, 28, 2))
    assert np.allclose(draws, expected.astype(np.float32))


def test_d5_iterated_d2_chaining_companion_is_emitted(tmp_path):
    summaries = tmp_path / "summaries"
    root = summaries / "dynamics_instruct"
    root.mkdir(parents=True)
    kinds = (
        "context_k",
        "s_k",
        "answer_k_t1",
        "answer_k_t2",
        "answer_k_t3",
        "u1",
        "u2",
        "u3",
    )
    rows = []
    arrays = []
    for conv_i in range(4):
        for turn in (0, 2, 4, 6):
            rows.append(
                {
                    "conv_id": f"c{conv_i}",
                    "turn_index": turn,
                    "token_start": turn,
                    "token_end": turn + 1,
                }
            )
            arrays.append([conv_i + 0.1 * turn, 1.0 + turn])
    base = np.asarray(arrays, dtype=np.float32)
    for kind_i, kind in enumerate(kinds):
        np.save(root / f"{kind}_L00.npy", base + kind_i * 0.01)
        with open(root / f"row_index_{kind}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    args = argparse.Namespace(n_folds=2)
    out = fit_grid._compute_dynamics_reads(
        summaries,
        "cell_inst_own",
        0,
        args,
        judge_rows=[],
    )

    companion = out["D5_iterated_D2_chaining_companion"]
    assert companion["status"] == "computed"
    assert companion["horizons"] == [0, 2, 4, 6]
    entry = companion["profile"]["context_k"]["4"]["context_k"]
    assert entry["status"] == "computed"
    assert isinstance(entry["direct_cv_r2"], float)
    assert isinstance(entry["iterated_d2_chain_r2"], float)


def _write_dynamics_fixture(root, rows, vecs_by_kind):
    root.mkdir(parents=True)
    for kind, mat in vecs_by_kind.items():
        np.save(root / f"{kind}_L00.npy", mat.astype(np.float32))
        with open(root / f"row_index_{kind}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def test_d5_chain_companion_scores_out_of_fold_on_direct_folds(tmp_path):
    """Regression pin for concern i1092-d5-chain-cv-asymmetry (round 6).

    Data design: every kind's state is a per-conversation random vector (plus
    tiny within-row noise), so each chain map (s->context lift, context->answer
    readout) is learnable ONLY by memorizing conversation identity. The old
    full-data/in-sample chaining therefore scores near-perfectly, while an
    out-of-fold chain on conversation-grouped folds cannot predict a held-out
    conversation's random states. Pins: (a) the emitted chain R2 equals a
    manual per-fold recompute on the SAME `_folds_from_manifest` partition the
    direct read uses (fold identity + held-out-only scoring), (b) it diverges
    from the in-sample value by a wide margin, (c) fold counts match the
    direct read's.
    """
    rng = np.random.default_rng(1092)
    hidden = 8
    n_convs = 6
    turns = (0, 2, 4, 6)
    kinds = (
        "context_k",
        "s_k",
        "answer_k_t1",
        "answer_k_t2",
        "answer_k_t3",
        "u1",
        "u2",
        "u3",
    )
    rows = [
        {"conv_id": f"c{conv_i}", "turn_index": turn, "token_start": turn, "token_end": turn + 1}
        for conv_i in range(n_convs)
        for turn in turns
    ]
    vecs_by_kind = {}
    for kind in kinds:
        base = rng.normal(size=(n_convs, hidden))
        mat = np.empty((len(rows), hidden))
        for i, row in enumerate(rows):
            conv_i = int(row["conv_id"][1:])
            mat[i] = base[conv_i] + 0.01 * rng.normal(size=hidden)
        # fp32 round-trip so the manual reference sees BITWISE the same values
        # the module loads back from the float32 .npy fixture.
        vecs_by_kind[kind] = mat.astype(np.float32)
    summaries = tmp_path / "summaries"
    _write_dynamics_fixture(summaries / "dynamics_instruct", rows, vecs_by_kind)
    row_pos = {(row["conv_id"], row["turn_index"]): i for i, row in enumerate(rows)}

    def kind_at(kind, conv_turns):
        return np.asarray(
            [vecs_by_kind[kind][row_pos[key]] for key in conv_turns], dtype=np.float64
        )

    out = fit_grid._compute_dynamics_reads(
        summaries, "cell_inst_own", 0, argparse.Namespace(n_folds=2), judge_rows=[]
    )
    companion = out["D5_iterated_D2_chaining_companion"]
    assert "out-of-fold" in companion["method"].lower()
    assert "full-data" not in companion["method"].lower()
    entry = companion["profile"]["s_k"]["2"]["answer_k_t1"]
    assert entry["status"] == "computed"
    emitted = entry["iterated_d2_chain_r2"]

    # Manual reference pools in the module's pair order (conv-major, turn-minor).
    convs = [f"c{i}" for i in range(n_convs)]
    next_keys = [(c, t) for c in convs for t in (0, 2, 4)]
    X_next = kind_at("s_k", next_keys)
    Y_next = kind_at("context_k", [(c, t + 2) for c, t in next_keys])
    read_keys = [(c, t) for c in convs for t in turns]
    X_read = kind_at("context_k", read_keys)
    Y_read = kind_at("answer_k_t1", read_keys)
    first_keys = [(c, 0) for c in convs]
    Xsrc = kind_at("s_k", first_keys)
    Ydst = kind_at("answer_k_t1", [(c, 2) for c in convs])

    def press_pred(Xtr, Ytr, Xte):
        res = press_fit_predict(
            torch.from_numpy(Xtr).double(),
            torch.from_numpy(Ytr).double(),
            torch.from_numpy(Xte).double(),
            standardize=True,
        )
        return res["pred"].detach().cpu().numpy()

    # (b) The OLD in-sample protocol (full-pool maps, scored on training convs)
    # is near-perfect on this memorization-only data.
    in_sample = fit_grid._r2(Ydst, press_pred(X_read, Y_read, press_pred(X_next, Y_next, Xsrc)))
    assert in_sample > 0.8

    # (a) Manual out-of-fold recompute on the direct read's exact partition.
    rows_first = [{"conv_id": c, "turn_index": 0} for c in convs]
    folds = fit_grid._folds_from_manifest(
        rows_first, len(rows_first), group_key="conv_id", n_folds=2
    )
    assert len(folds) == 2
    pred = np.zeros_like(Ydst)
    for test_idx in folds:
        heldout = {rows_first[i]["conv_id"] for i in test_idx}
        keep_next = [i for i, (c, _t) in enumerate(next_keys) if c not in heldout]
        keep_read = [i for i, (c, _t) in enumerate(read_keys) if c not in heldout]
        state = press_pred(X_next[keep_next], Y_next[keep_next], Xsrc[test_idx])
        pred[test_idx] = press_pred(X_read[keep_read], Y_read[keep_read], state)
    expected = fit_grid._r2(Ydst, pred)
    assert emitted == pytest.approx(expected, abs=1e-8)
    assert emitted < in_sample - 0.4

    # (c) Fold partition parity with the direct D5 read.
    direct_entry = out["D5_first_state_horizon"]["s_k"]["2"]["answer_k_t1"]
    assert entry["fold_count"] == len(direct_entry["fit"]["r2_folds"])
    assert len(entry["chain_r2_folds"]) == entry["fold_count"]


# ── round-7 P0 production-hardening pins (real corpus row shapes) ─────────────
# Fixture rows copy the REAL field structure of WildChat-1M / lmsys-chat-1m
# rows as verified live via the HF datasets-server rows API (2026-07-07):
# full-name `language` values ('English', not 'en'), WildChat top-level +
# per-turn `redacted`/`toxic` bools, per-turn `openai_moderation` entries
# {categories: {name: bool}, category_scores: {name: float}, flagged: bool},
# WildChat `detoxify_moderation` continuous scores, LMSYS top-level `redacted`
# with role/content-only turns.


def _moderation_entry(*, flagged=False, category=False):
    return {
        "categories": {
            "harassment": category,
            "harassment/threatening": False,
            "hate": False,
            "self-harm": False,
            "sexual": False,
            "violence": False,
        },
        "category_scores": {
            "harassment": 0.001,
            "harassment/threatening": 0.0001,
            "hate": 0.0002,
            "self-harm": 0.0001,
            "sexual": 0.0001,
            "violence": 0.0003,
        },
        "flagged": flagged,
    }


def _wildchat_row(
    *,
    language="English",
    redacted=False,
    toxic=False,
    turn_redacted=False,
    turn_toxic=False,
    mod_flagged=False,
    mod_category=False,
    user_text="What is the capital of France?",
    assistant_text="The capital of France is Paris.",
):
    conversation = [
        {
            "content": user_text,
            "country": "United States",
            "hashed_ip": "0" * 64,
            "header": {"accept-language": "en-US,en;q=0.9", "user-agent": "Mozilla/5.0"},
            "language": language,
            "redacted": turn_redacted,
            "role": "user",
            "state": "Texas",
            "timestamp": None,
            "toxic": turn_toxic,
            "turn_identifier": 101001,
        },
        {
            "content": assistant_text,
            "country": None,
            "hashed_ip": None,
            "header": None,
            "language": language,
            "redacted": False,
            "role": "assistant",
            "state": None,
            "timestamp": "2024-01-01T00:00:00Z",
            "toxic": False,
            "turn_identifier": 101001,
        },
    ]
    detoxify = {
        "identity_attack": 2e-4,
        "insult": 2e-3,
        "obscene": 4e-4,
        "severe_toxicity": 3e-5,
        "sexual_explicit": 1e-4,
        "threat": 6e-5,
        "toxicity": 5e-3,
    }
    return {
        "conversation_hash": "a" * 32,
        "model": "gpt-4-0314",
        "timestamp": "2024-01-01T00:00:00Z",
        "conversation": conversation,
        "turn": 1,
        "language": language,
        "openai_moderation": [
            _moderation_entry(flagged=mod_flagged, category=mod_category),
            _moderation_entry(),
        ],
        "detoxify_moderation": [dict(detoxify), dict(detoxify)],
        "toxic": toxic,
        "redacted": redacted,
        "state": "Texas",
        "country": "United States",
        "hashed_ip": "0" * 64,
        "header": {"accept-language": "en-US,en;q=0.9", "user-agent": "Mozilla/5.0"},
    }


def _lmsys_row(
    *,
    language="English",
    redacted=False,
    mod_flagged=False,
    user_text="Explain what a hash map is.",
    assistant_text="A hash map stores key-value pairs for fast lookup.",
):
    return {
        "conversation_id": "b" * 32,
        "model": "vicuna-13b",
        "conversation": [
            {"content": user_text, "role": "user"},
            {"content": assistant_text, "role": "assistant"},
        ],
        "turn": 1,
        "language": language,
        "openai_moderation": [_moderation_entry(flagged=mod_flagged), _moderation_entry()],
        "redacted": redacted,
    }


def test_stream_conversations_wildchat_real_shape_filters(monkeypatch, caplog):
    """Round-7 pin: full-name language keeps English rows; §4.1 redaction /
    toxicity / moderation flags reject on the REAL WildChat field shapes; the
    done log line carries kept + per-filter reject counters."""
    import logging
    import random

    import datasets
    import issue1092_build_corpus as bc

    unique = iter(range(1000))

    def wc(**kw):
        row = _wildchat_row(**kw)
        row["conversation"][0]["content"] += f" (variant {next(unique)})"
        return row

    rows = [
        wc(),  # keep — the round-4 recipe row a working chain must pass
        wc(language="Spanish"),  # reject: language (full name)
        wc(redacted=True),  # reject: redacted (top-level)
        wc(turn_redacted=True),  # reject: redacted (per-turn)
        wc(toxic=True),  # reject: toxic (top-level)
        wc(turn_toxic=True),  # reject: toxic (per-turn)
        wc(mod_flagged=True),  # reject: openai_moderation flagged
        wc(mod_category=True),  # reject: openai_moderation category True
        wc(language="en-US"),  # keep — regioned code form
        wc(language=""),  # keep — empty language passes (pre-existing)
    ]
    monkeypatch.setattr(bc, "_SMOKE_TOKEN_COUNTS", True)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: list(rows))

    stats: dict = {}
    with caplog.at_level(logging.INFO, logger="issue1092.build_corpus"):
        kept = bc._stream_conversations(
            "allenai/WildChat-1M",
            "0" * 40,
            rng=random.Random(0),
            row_limit=None,
            stats_out=stats,
        )

    assert [c["source"] for c in kept] == ["wildchat"] * 3
    assert stats["kept"] == 3
    assert stats["streamed"] == len(rows)
    assert stats["rejects"]["language"] == 1
    assert stats["rejects"]["redacted"] == 2
    assert stats["rejects"]["toxic"] == 2
    assert stats["rejects"]["moderation"] == 2
    assert stats["rejects"]["structure"] == 0
    done_lines = [
        r.getMessage() for r in caplog.records if "[stream wildchat] done:" in r.getMessage()
    ]
    assert len(done_lines) == 1
    assert "3 conversations kept of 10 streamed" in done_lines[0]
    assert '"language": 1' in done_lines[0]
    assert '"redacted": 2' in done_lines[0]


def test_stream_conversations_lmsys_real_shape_and_stream_limit(monkeypatch):
    """Round-7 pin: LMSYS shape (top-level `redacted`, role/content-only turns,
    per-turn openai_moderation) filters correctly, and `stream_limit` bounds
    TOTAL streamed rows independently of the kept `row_limit`."""
    import random

    import datasets
    import issue1092_build_corpus as bc

    rows = []
    for i in range(6):
        if i % 2 == 0:
            rows.append(_lmsys_row(user_text=f"Question {i}: how do vaccines work?"))
        else:
            rows.append(_lmsys_row(language="Spanish"))
    rows.append(_lmsys_row(redacted=True))
    rows.append(_lmsys_row(mod_flagged=True))

    monkeypatch.setattr(bc, "_SMOKE_TOKEN_COUNTS", True)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: list(rows))

    # full pass — all 8 rows examined
    stats_full: dict = {}
    kept_full = bc._stream_conversations(
        "lmsys/lmsys-chat-1m",
        "0" * 40,
        rng=random.Random(0),
        row_limit=None,
        stats_out=stats_full,
    )
    assert stats_full == {
        "kept": 3,
        "streamed": 8,
        "rejects": {
            "language": 3,
            "redacted": 1,
            "toxic": 0,
            "moderation": 1,
            "empty_conversation": 0,
            "structure": 0,
            "token_budget": 0,
            "duplicate": 0,
        },
    }
    assert [c["source"] for c in kept_full] == ["lmsys"] * 3

    # bounded probe — stream_limit caps examined rows, not kept rows
    stats_bounded: dict = {}
    kept_bounded = bc._stream_conversations(
        "lmsys/lmsys-chat-1m",
        "0" * 40,
        rng=random.Random(0),
        row_limit=None,
        stream_limit=4,
        stats_out=stats_bounded,
    )
    assert stats_bounded["streamed"] == 4
    assert stats_bounded["kept"] == 2
    assert len(kept_bounded) == 2


def test_lang_matches_full_name_and_code_forms():
    import issue1092_build_corpus as bc

    assert bc._lang_matches("English", "en")
    assert bc._lang_matches("english", "en")
    assert bc._lang_matches("en", "en")
    assert bc._lang_matches("en-US", "en")
    assert not bc._lang_matches("Spanish", "en")
    assert not bc._lang_matches("unknown", "en")
    # BOTH production datasets store full names — the round-7 root cause
    assert not bc._lang_matches("Catalan", "en")
