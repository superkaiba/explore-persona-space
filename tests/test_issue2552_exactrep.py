"""Tiny-fixture tests for the #2552 exactrep pipeline (prep / capture / train drivers).

Offline by construction: the tokenizer boundary is a signature-conformant char-level
fake (Qwen-shaped chat template); the capture forward runs a REAL tiny random-weight
Qwen2 built from a local config (no network, no worktree paths, pytest tmp_path only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2552_exactrep_capture as CAP  # noqa: E402
import issue2552_exactrep_prep as PREP  # noqa: E402

DEFAULT_SYS = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
    "You are a helpful assistant.<|im_end|>\n"
)


class FakeTok:
    """Char-level tokenizer with the exact surface prep/capture consume.

    apply_chat_template mirrors the Qwen2.5 segment shape (default system turn +
    "<|im_start|>{role}\\n{content}<|im_end|>\\n" per message); __call__ maps one
    char -> one token with per-char offsets, so concat(segments) == full-text
    tokenization holds by construction (as it does on the real pinned tokenizer)."""

    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        segs = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs]
        out = DEFAULT_SYS + "".join(segs)
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out

    @staticmethod
    def _enc(t: str):
        ids = [1 + (ord(c) % 150_000) for c in t]
        offs = [(i, i + 1) for i in range(len(t))]
        return ids, offs

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_kw):
        if isinstance(text, list):
            pairs = [self._enc(t) for t in text]
            out = {"input_ids": [p[0] for p in pairs]}
            if return_offsets_mapping:
                out["offset_mapping"] = [p[1] for p in pairs]
            return out
        ids, offs = self._enc(text)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


def _row(conv, language="English", redacted=False, cid="conv-1"):
    return {
        "conversation_id": cid,
        "model": "some-model",
        "language": language,
        "redacted": redacted,
        "turn": sum(1 for m in conv if m["role"] == "assistant"),
        "conversation": conv,
        "openai_moderation": [{"flagged": False} for _ in conv],
    }


def _counters():
    return {k: 0 for k in PREP._COUNTER_KEYS}


# ── span extraction ────────────────────────────────────────────────────────────────


def test_render_segments_join_equals_template():
    tok = FakeTok()
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4."},
        {"role": "user", "content": "And doubled?"},
        {"role": "assistant", "content": "8."},
    ]
    segs = PREP.render_segments(msgs, tok)
    assert "".join(segs) == tok.apply_chat_template(msgs, tokenize=False)
    assert len(segs) == len(msgs) + 1  # default system prefix segment


def test_content_char_ranges_slice_exact_content():
    tok = FakeTok()
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "Answer one."},
        {"role": "user", "content": "Q2 longer question"},
        {"role": "assistant", "content": "A2!"},
    ]
    segs = PREP.render_segments(msgs, tok)
    full = "".join(segs)
    roles = [m["role"] for m in msgs]
    ranges = CAP.content_char_ranges(segs, 1, roles, [1, 3])
    assert [full[a:b] for a, b in ranges] == ["Answer one.", "A2!"]


def test_token_spans_overlap_straddle_and_zero_width():
    # tokens: [0,5) [5,12) [12,14) [14,20) [20,21); (12,14) straddles the range start.
    offsets = [(0, 5), (5, 12), (12, 14), (14, 20), (20, 21)]
    spans = CAP.token_spans_from_offsets(offsets, [(13, 20), (5, 5), (6, 8)])
    assert spans[0] == (2, 4)  # straddling token included, tail token excluded
    assert spans[1] is None  # zero-width content -> dropped, never zero-faked
    assert spans[2] == (1, 2)  # content fully inside one merged token
    # a range covered only by zero-width offset entries yields None
    assert CAP.token_spans_from_offsets([(3, 3), (3, 3)], [(2, 4)]) == [None]


def test_span_means_matches_loop_distinct_dims():
    torch.manual_seed(0)
    hs = torch.randn(3, 17, 5)  # distinct B/T/H so any transpose crashes or mismatches
    spans = [[(0, 4), (10, 17)], [(2, 3)], [(5, 9)]]
    got = CAP.span_means(hs, spans)
    expect = torch.stack(
        [hs[0, 0:4].mean(0), hs[0, 10:17].mean(0), hs[1, 2:3].mean(0), hs[2, 5:9].mean(0)]
    )
    assert got.shape == (4, 5)
    assert torch.allclose(got, expect, atol=1e-6)


def test_batches_by_budget_caps():
    lengths = [10, 100, 50, 60]
    batches = CAP.batches_by_budget(lengths, max_rows=2, max_tokens=120)
    assert sorted(i for b in batches for i in b) == [0, 1, 2, 3]
    for b in batches:
        assert len(b) <= 2
        assert len(b) * max(lengths[i] for i in b) <= 120
    assert batches[0] == [1]  # the 100-token conv cannot share a 120-token budget


# ── prep filtering ─────────────────────────────────────────────────────────────────


def test_prep_filters_and_counters():
    tok = FakeTok()
    c = _counters()
    good = [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi, how can I help?"},
        {"role": "user", "content": "what is 1+1"},
        {"role": "assistant", "content": "2"},
    ]
    rec = PREP.process_conversation(_row(good), tok, max_tokens=10_000, counters=c)
    assert rec is not None and rec["asst_msg_idx"] == [1, 3]
    assert c["kept_convs"] == 1 and c["kept_turns"] == 2

    assert PREP.process_conversation(_row(good, language="Portuguese"), tok, 10_000, c) is None
    assert c["reject_language"] == 1

    bad_order = [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "?"}]
    assert PREP.process_conversation(_row(bad_order), tok, 10_000, c) is None
    assert c["reject_structure"] == 1

    empty_asst = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "   "},
    ]
    assert PREP.process_conversation(_row(empty_asst), tok, 10_000, c) is None
    assert c["empty_assistant_turns"] == 1 and c["reject_zero_turns"] == 1

    r = PREP.process_conversation(_row(good, redacted=True), tok, 10_000, c)
    assert r is not None and c["redacted_kept"] == 1


def test_prep_budget_truncation_and_over_budget():
    tok = FakeTok()
    c = _counters()
    conv = [
        {"role": "user", "content": "u" * 20},
        {"role": "assistant", "content": "a" * 20},
        {"role": "user", "content": "u" * 400},
        {"role": "assistant", "content": "a" * 400},
    ]
    # budget fits system + first pair only (char-level: segment overhead is exact)
    segs = PREP.render_segments(conv, tok)
    fit_two = sum(len(s) for s in segs[:3])
    rec = PREP.process_conversation(_row(conv), tok, max_tokens=fit_two, counters=c)
    assert rec is not None and rec["asst_msg_idx"] == [1]
    assert len(rec["msgs"]) == 2 and rec["n_render_tokens"] == fit_two
    assert c["truncated_convs"] == 1 and c["turns_dropped_by_budget"] == 1
    # budget below the first pair -> the whole conversation is rejected
    assert PREP.process_conversation(_row(conv), tok, max_tokens=30, counters=c) is None
    assert c["reject_over_budget_all"] == 1


def test_truncate_to_budget_arithmetic():
    # prefix 7 tokens; msgs segs: u=10, a=12, u=9, a=11
    counts = [7, 10, 12, 9, 11]
    assert PREP.truncate_to_budget(counts, 1, 4, max_tokens=49) == (4, 49)
    assert PREP.truncate_to_budget(counts, 1, 4, max_tokens=48) == (2, 29)
    assert PREP.truncate_to_budget(counts, 1, 4, max_tokens=28) == (0, 0)


# ── capture resume + pilot arithmetic ──────────────────────────────────────────────


def test_chunk_resume_predicate(tmp_path):
    fp = {"model_id": "m", "layer": 19}
    npy, rows, done = CAP.chunk_paths(tmp_path, 3)
    assert CAP.chunk_completed(tmp_path, 3, fp) is False
    np.save(npy, np.zeros((2, 4), np.float16))
    rows.write_text('{"row": 0}\n{"row": 1}\n')
    assert CAP.chunk_completed(tmp_path, 3, fp) is False  # sentinel is written LAST
    done.write_text(json.dumps({"fingerprint": fp}))
    assert CAP.chunk_completed(tmp_path, 3, fp) is True
    assert CAP.chunk_completed(tmp_path, 3, {"model_id": "m", "layer": 14}) is False


def test_pilot_extrapolation():
    out = CAP.pilot_extrapolation(120.0, n_chunks_total=400, num_shards=4)
    assert out["per_shard_chunks"] == 100
    assert out["projected_shard_hours"] == pytest.approx(100 * 120 / 3600, abs=1e-3)
    assert out["projected_total_gpu_hours"] == pytest.approx(400 * 120 / 3600, abs=1e-3)


# ── capture end-to-end on a real tiny model (production body, fake tokenizer only) ──


def _tiny_qwen(hidden=32, layers=2):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=152_064,
        hidden_size=hidden,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def test_capture_chunk_batched_equals_serial_tiny_real_model():
    tok = FakeTok()
    model = _tiny_qwen()
    c = _counters()
    convs = [
        [
            {"role": "user", "content": "short question"},
            {"role": "assistant", "content": "short answer"},
        ],
        [
            {"role": "user", "content": "a much longer question " * 4},
            {"role": "assistant", "content": "a much longer answer " * 6},
            {"role": "user", "content": "follow-up?"},
            {"role": "assistant", "content": "indeed, a second turn answer."},
        ],
        [
            {"role": "user", "content": "third conversation"},
            {"role": "assistant", "content": "reply three."},
        ],
    ]
    recs = []
    for i, conv in enumerate(convs):
        rec = PREP.process_conversation(_row(conv, cid=f"c{i}"), tok, 10_000, _counters())
        assert rec is not None
        recs.append(rec)
    args = argparse.Namespace(
        layer=1, batch_max_rows=8, batch_max_tokens=4096, tiny_model=True, device="cpu"
    )
    y_b, meta_b = CAP.capture_chunk(recs, tok, model, args, c)
    assert y_b.dtype == np.float16 and y_b.shape == (4, 32)  # 4 assistant turns total
    assert [m["conversation_id"] for m in meta_b] == ["c0", "c1", "c1", "c2"]
    assert all(m["n_span_tokens"] > 0 for m in meta_b)
    # padding fires (mixed lengths, B>=2); serial batch-1 path must agree (CPU fp32)
    args_serial = argparse.Namespace(**{**vars(args), "batch_max_rows": 1})
    y_s, meta_s = CAP.capture_chunk(recs, tok, model, args_serial, _counters())
    assert meta_b == meta_s
    cos = torch.nn.functional.cosine_similarity(
        torch.as_tensor(y_b, dtype=torch.float32),
        torch.as_tensor(y_s, dtype=torch.float32),
        dim=1,
    )
    assert float(cos.min()) >= 0.999, float(cos.min())


# ── train driver: split floors + assemble/train on synthetic chunks ────────────────


def test_derive_splits_floors_and_full_sizes():
    import issue2552_exactrep_train as TRN

    s = TRN.derive_splits(100)
    assert len(s["holdout"]) == 10 and len(s["val"]) == 10 and len(s["train"]) == 80
    all_idx = np.concatenate([s["holdout"], s["val"], s["train"]])
    assert len(np.unique(all_idx)) == 100
    big = TRN.derive_splits(5 * (TRN.HOLDOUT_N + TRN.VAL_N) + 1)
    assert len(big["holdout"]) == TRN.HOLDOUT_N and len(big["val"]) == TRN.VAL_N


def test_assemble_and_train_tiny_end_to_end(tmp_path):
    import issue2552_exactrep_train as TRN

    store = tmp_path / "store"
    store.mkdir()
    rng = np.random.default_rng(0)
    fp = {"k": "v"}
    total = 0
    for gci, n in enumerate([40, 30, 30]):
        y = rng.normal(size=(n, 16)).astype(np.float16)
        npy, rows, done = CAP.chunk_paths(store, gci)
        np.save(npy, y)
        with rows.open("w") as f:
            for j in range(n):
                f.write(json.dumps({"conversation_id": f"c{gci}-{j}", "msg_idx": 1}) + "\n")
        done.write_text(json.dumps({"fingerprint": fp, "n_rows": n}))
        total += n
    out = tmp_path / "sae"
    args = argparse.Namespace(
        store_dirs=[str(store)],
        out_dir=out,
        device="cpu",
        dict_size=64,
        steps_cap=2,
        production=False,
    )
    TRN.phase_assemble(args)
    mm = np.load(out / "Y19.fp16.npy", mmap_mode="r")
    assert mm.shape == (total, 16) and mm.dtype == np.float16
    idx = [json.loads(line) for line in (out / "row_index.jsonl").open()]
    assert len(idx) == total and idx[40]["row"] == 40 and idx[40]["chunk"] == "chunk_000001.npy"
    TRN.phase_train(args)
    log = json.loads((out / "train_log.json").read_text())
    assert log["epochs"] and 0 < log["epochs"][0]["steps"] <= 2
    assert "holdout_nmse" in log and log["paper_nmse"] == 0.097
    assert (out / "sae_weights.safetensors").exists() and (out / "cfg.json").exists()
    # resume predicate: a second assemble call is a no-op skip (same chunk set)
    TRN.phase_assemble(args)
