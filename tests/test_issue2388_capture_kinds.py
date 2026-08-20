"""#2388 pinned tests for the ``kinds`` extension of the #1739 capture module.

Plan section 4 (pooling fork): the new-surface store carries kinds
``context_end`` / ``t1`` / ``t_last`` (NO per-rollout ``prefix_end`` — storing
it per rollout would blow the ~130 GB MooseFS quota, plan section 9), while the
parent's callers must stay byte-identical under the DEFAULT
(``kinds=SUMMARY_KINDS``). Pins:

t1: the default path emits exactly ``SUMMARY_KINDS`` with bitwise-identical
    arrays to the explicit-default call (parent #1739 is LIVE on this module);
t2: ``t_last`` is the hidden state at the LAST answer token
    (``answer_end - 1`` — answer_end is exclusive), per layer;
t3: an unknown kind fails loud;
t4: ``capture_rollout_files`` threads ``kinds`` into the written shards (a
    ``t_last`` file per layer exists, the manifest records the kinds) and the
    resume path accepts the non-default-kind store.

The tokenizer is a char-level fake (1 token per char, identity offsets) and
the model a deterministic stub whose hidden state at (layer, position) is
``position + 1000 * layer`` broadcast over the hidden dim — so every extract
is checkable in closed form. Byte-identity here is same-machine bitwise
equality (``np.array_equal``), never a cross-machine float pin.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739.capture import (
    capture_batch,
    capture_rollout_files,
    capture_row_ids_and_positions,
)
from explore_persona_space.experiments.issue_1739.constants import SUMMARY_KINDS

N_LAYERS = 2
HIDDEN = 4


class _CharTokenizer:
    """1 token per char; offsets are identity — positions checkable by hand."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [(ord(c) % 250) + 1 for c in text]

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping=True):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def pad(self, batch, return_tensors="pt", padding=True):
        import torch

        seqs = batch["input_ids"]
        width = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), width), dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _StubModel:
    """hidden_states[1+layer][b, t, :] == t + 1000*layer (position-coded)."""

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=True, **kw):
        import torch

        b, t = input_ids.shape
        pos = torch.arange(t, dtype=torch.float32)[None, :, None].expand(b, t, HIDDEN)
        hs = tuple(pos + 1000.0 * layer for layer in range(N_LAYERS + 1))
        # hidden_states[0] is the embedding layer (skipped by capture_batch).
        return SimpleNamespace(hidden_states=hs)


_ROWS = [
    ("prefix one. ", "prefix one. What is 2+2?", " It is 4."),
    ("", "Compute 3*3.", " 9."),
    ("ctx. ", "ctx. Name a color.", " Blue, because sky."),
]


def _run(kinds=None):
    kwargs = {} if kinds is None else {"kinds": kinds}
    return capture_batch(
        [r[0] for r in _ROWS],
        [r[1] for r in _ROWS],
        [r[2] for r in _ROWS],
        model=_StubModel(),
        tokenizer=_CharTokenizer(),
        n_layers=N_LAYERS,
        hidden_dim=HIDDEN,
        device="cpu",
        batch_size=2,
        **kwargs,
    )


def test_t1_default_kinds_byte_identical():
    """Default emits exactly SUMMARY_KINDS, bitwise-equal to the explicit form."""
    default_summaries, default_pos = _run()
    explicit_summaries, _ = _run(kinds=SUMMARY_KINDS)
    assert len(default_summaries) == len(_ROWS)
    for d, e in zip(default_summaries, explicit_summaries, strict=True):
        assert tuple(d.keys()) == SUMMARY_KINDS  # no t_last on the default path
        for kind in SUMMARY_KINDS:
            assert np.array_equal(d[kind], e[kind]), kind
    # context_end closed form: value == position + 1000*(layer+1) — the stub's
    # tuple index 0 is the embedding layer, which capture_batch skips.
    for row_i, pos in enumerate(default_pos):
        arr = default_summaries[row_i]["context_end"]
        for layer in range(N_LAYERS):
            assert np.allclose(arr[layer], pos["context_end"] + 1000.0 * (layer + 1))


def test_t2_t_last_is_last_answer_token():
    kinds = SUMMARY_KINDS + ("t_last",)
    summaries, positions = _run(kinds=kinds)
    tok = _CharTokenizer()
    for row_i, (prefix, prompt, completion) in enumerate(_ROWS):
        _, pos = capture_row_ids_and_positions(tok, prefix, prompt, completion)
        assert positions[row_i] == pos
        arr = summaries[row_i]["t_last"]
        assert arr.shape == (N_LAYERS, HIDDEN)
        t_expected = min(pos["answer_end"] - 1, pos["n_total"] - 1)
        for layer in range(N_LAYERS):
            assert np.allclose(arr[layer], t_expected + 1000.0 * (layer + 1))
        # t_last is a single position INSIDE the t1 span, never the span mean.
        assert not np.array_equal(arr, summaries[row_i]["t1"])


def test_t3_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown capture summary kind"):
        _run(kinds=("context_end", "bogus_kind"))


def _write_labeling_json(path: Path, i: int) -> None:
    prefix, prompt, completion = _ROWS[i % len(_ROWS)]
    path.write_text(
        json.dumps(
            {
                "prefix_text": prefix,
                "prompt_text": prompt,
                "completion": completion + f" row{i}",
                "context_id": f"ctx{i:03d}",
                "behavior": "math",
                "split": "train",
                "rung": "train",
                "group_key": f"grp{i % 2}",
                "rollout_k": 0,
            }
        )
    )


def test_t4_capture_rollout_files_threads_kinds(tmp_path):
    rollout_paths = []
    for i in range(3):
        p = tmp_path / f"rollout_{i:03d}.json"
        _write_labeling_json(p, i)
        rollout_paths.append(p)
    store = tmp_path / "store"
    kinds = ("context_end", "t1", "t_last")  # the #2388 store shape (no prefix_end)
    kwargs = dict(
        store_dir=store,
        model=_StubModel(),
        tokenizer=_CharTokenizer(),
        n_layers=N_LAYERS,
        hidden_dim=HIDDEN,
        device="cpu",
        batch_size=2,
        shard_rows=2,
        fingerprint="t4",
        kinds=kinds,
    )
    manifest = capture_rollout_files(rollout_paths, **kwargs)
    assert manifest["n_rows"] == 3 and manifest["kinds"] == list(kinds)
    for kind in kinds:
        for layer in range(N_LAYERS):
            for shard in (0, 1):
                f = store / f"{kind}_L{layer:02d}_shard{shard:02d}.npy"
                assert f.is_file(), f
    assert not list(store.glob("prefix_end_*.npy"))  # omitted kind never written
    # Resume over the non-default-kind store is a clean no-op re-run.
    manifest2 = capture_rollout_files(rollout_paths, **kwargs)
    assert manifest2["n_rows_captured"] == 0 and manifest2["n_shards_resumed"] == 2
