"""Issue #742 round-11 regression tests — crash-safe RESUME of the judge rerun.

Three consecutive #742 production launches crashed mid-Phase-1 (ENOSPC, missing
cache, earlyoom SIGTERM) and each LOST every Anthropic Batch API result because the
per-cell dispatch state used ``tempfile.TemporaryDirectory`` (wiped on Python exit),
defeating the #663-hardened ``eval.batch_judge`` crash-safe resume protocol AND the
outer loop accumulated in-memory and wrote only at the end.

These tests verify the two-layer fix from the OUTSIDE, with NO real API calls:

  * Layer 1 (``dc.judge_column_via_batch_judge``): a PERSISTENT per-cell dispatch dir
    keyed on a stable content hash. Two calls on the SAME (col_id, completions) reuse
    the SAME dir and the second SHORT-CIRCUITS (no batch re-submit). Different
    (col_id, completions) → different dir (content-hash keying).
  * Layer 2 (``scripts/issue742_judge_rerun.py``): per-cell checkpointing — the outer
    loop persists each (genre, behavior) cell the instant it returns and, on a re-run,
    SKIPS cells whose partial already exists.

``judge_completions_batch`` is monkeypatched to a stub that writes the real
``scores.json`` shape (``all_scores`` + a ``routing`` dict with ``path="batch"``) and
counts its invocations — so the resume short-circuit is observable as a call count.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

dc = importlib.import_module("explore_persona_space.analysis.issue_742_decoding_ceiling")


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _gen_cell(n_probes: int = 2, n_comps: int = 3) -> dict:
    """A tiny #658-shaped cell in the ``cells[i].completions[j]["text"]`` schema."""
    return {
        "context_id": "ctx_a",
        "cells": [
            {
                "probe": f"probe_{p}",
                "completions": [
                    {"text": f"probe{p} completion{k}", "logp_norm": 0.0} for k in range(n_comps)
                ],
            }
            for p in range(n_probes)
        ],
    }


def _make_batch_judge_stub(counter: dict) -> object:
    """A stub ``eval.batch_judge`` module whose ``judge_completions_batch`` writes the real
    ``scores.json`` shape (``all_scores`` keyed by the ``cell__{q:05d}__{c:02d}`` custom_id,
    a ``routing`` dict with ``path='batch'``) and increments ``counter['submits']`` once per
    call, mimicking a clean batch dispatch WITHOUT any API contact."""

    class _StubBatchJudge:
        @staticmethod
        def judge_completions_batch(
            completions,
            *,
            judge_system_prompt,
            format_user_msg,
            judge_model,
            cache_dir,
            save_raw,
            checkpoint_dir,
            threshold_base,
        ):
            counter["submits"] += 1
            all_scores: dict[str, dict] = {}
            for _persona, q_to_texts in completions.items():
                for q_idx, (_question, comps) in enumerate(q_to_texts.items()):
                    for comp_idx, _text in enumerate(comps):
                        cid = f"cell__{q_idx:05d}__{comp_idx:02d}"
                        # Deterministic verdict; the real rubric key for broad_em is
                        # "misaligned" — value doesn't matter for the resume test.
                        all_scores[cid] = {"misaligned": comp_idx % 2 == 0}
            save_raw = Path(save_raw)
            save_raw.parent.mkdir(parents=True, exist_ok=True)
            save_raw.write_text(
                json.dumps(
                    {
                        "per_persona": {},
                        "all_scores": all_scores,
                        "routing": {"path": "batch", "n_items": len(all_scores)},
                    }
                )
            )
            return {}

    return _StubBatchJudge()


@pytest.fixture
def stub_dispatch(monkeypatch):
    """Route ``importlib.import_module('explore_persona_space.eval.batch_judge')`` inside
    ``dc`` to the counting stub, so no real dispatch/API runs. Returns the call counter."""
    counter = {"submits": 0}
    stub = _make_batch_judge_stub(counter)
    real_import = importlib.import_module

    def _fake_import(name, *args, **kwargs):
        if name == "explore_persona_space.eval.batch_judge":
            return stub
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    return counter


# --------------------------------------------------------------------------- #
# Layer 1 — test (a): resume short-circuits the batch re-submit                #
# --------------------------------------------------------------------------- #
def test_layer1_second_call_reuses_dir_and_does_not_resubmit(tmp_path, stub_dispatch):
    """(a) Two calls on the SAME (col_id, completions) reuse the persistent dir; the
    second finds the COMPLETE scores.json and short-circuits (no batch re-submit)."""
    gen = _gen_cell()
    r1 = dc.judge_column_via_batch_judge("broad_em", gen, "m", state_root=tmp_path)
    assert stub_dispatch["submits"] == 1
    assert r1["rate"] is not None
    assert r1["routing_path"] == "batch"

    # The persistent dispatch dir must exist under the state root (NOT a wiped tempdir).
    state_root = dc._judge_rerun_state_root(tmp_path)
    cell_dirs = list(state_root.glob("broad_em__*"))
    assert len(cell_dirs) == 1, cell_dirs
    assert (cell_dirs[0] / "scores.json").exists()

    # Second call: identical inputs → SAME dir → native fast-resume → NO re-submit.
    r2 = dc.judge_column_via_batch_judge("broad_em", gen, "m", state_root=tmp_path)
    assert stub_dispatch["submits"] == 1, "resume must not re-submit the batch"
    assert r2["rate"] == r1["rate"]
    assert r2["n_judged"] == r1["n_judged"]
    # still exactly ONE cell dir — no duplicate spawned
    assert len(list(state_root.glob("broad_em__*"))) == 1


# --------------------------------------------------------------------------- #
# Layer 1 — test (b): content-hash keying                                      #
# --------------------------------------------------------------------------- #
def test_layer1_different_inputs_get_different_dirs(tmp_path, stub_dispatch):
    """(b) Different (col_id, completions) → different cache dir (content-hash keying)."""
    gen_a = _gen_cell(n_probes=2, n_comps=3)
    gen_b = _gen_cell(n_probes=2, n_comps=4)  # different completions → different hash

    dc.judge_column_via_batch_judge("broad_em", gen_a, "m", state_root=tmp_path)
    dc.judge_column_via_batch_judge("broad_em", gen_b, "m", state_root=tmp_path)
    # different completion set for the same col_id → two distinct dirs
    assert stub_dispatch["submits"] == 2
    same_col = list(dc._judge_rerun_state_root(tmp_path).glob("broad_em__*"))
    assert len(same_col) == 2, same_col

    # different col_id, same completions → also a distinct dir (col_id is in the key)
    dc.judge_column_via_batch_judge("refusal", gen_a, "m", state_root=tmp_path)
    assert stub_dispatch["submits"] == 3
    assert len(list(dc._judge_rerun_state_root(tmp_path).glob("refusal__*"))) == 1


def test_layer1_content_hash_is_order_invariant():
    """The content hash keys on the SORTED (probe, text) set, so completion ORDER does not
    change the dir — a re-run that enumerates completions differently still resumes."""
    flat1 = [
        {"probe": "p0", "text": "t0", "logp": 0.0},
        {"probe": "p1", "text": "t1", "logp": 1.0},
    ]
    flat2 = list(reversed(flat1))
    assert dc._judge_rerun_content_hash(flat1) == dc._judge_rerun_content_hash(flat2)
    # a genuinely different completion set → different hash
    flat3 = [{"probe": "p0", "text": "DIFFERENT", "logp": 0.0}]
    assert dc._judge_rerun_content_hash(flat3) != dc._judge_rerun_content_hash(flat1)


def test_layer1_partial_scores_are_not_reused(tmp_path, stub_dispatch):
    """A PARTIAL scores.json (missing a custom_id — the mid-crash shape) must NOT
    short-circuit: the second call re-enters the dispatcher (which would #663-resume)."""
    gen = _gen_cell(n_probes=2, n_comps=3)
    dc.judge_column_via_batch_judge("broad_em", gen, "m", state_root=tmp_path)
    assert stub_dispatch["submits"] == 1
    cell_dir = next(dc._judge_rerun_state_root(tmp_path).glob("broad_em__*"))
    scores_path = cell_dir / "scores.json"
    raw = json.loads(scores_path.read_text())
    # drop one verdict → now incomplete (simulates a crash before the last batch harvested)
    raw["all_scores"].pop(next(iter(raw["all_scores"])))
    scores_path.write_text(json.dumps(raw))

    dc.judge_column_via_batch_judge("broad_em", gen, "m", state_root=tmp_path)
    assert stub_dispatch["submits"] == 2, "an incomplete scores.json must NOT be reused"


# --------------------------------------------------------------------------- #
# Layer 2 — test (c): per-cell checkpointing + skip-completed on re-run        #
# --------------------------------------------------------------------------- #
def test_layer2_cell_checkpoint_and_skip_on_rerun(tmp_path, monkeypatch):
    """(c) Run the outer loop; simulate a crash after 1 cell; re-run — the second run
    SKIPS the completed cell (its partial exists) and only judges the remaining one."""
    jr = importlib.import_module("issue742_judge_rerun")

    genres = ["betley"]
    behaviors = ["broad_em", "refusal"]

    # Count how many cells actually get judged by counting _judge_reruns_for_cell calls.
    judged_cells: list[tuple[str, str]] = []
    real_reruns = jr._judge_reruns_for_cell

    def _spy_reruns(*, genre, behavior, **kwargs):
        judged_cells.append((genre, behavior))
        return real_reruns(genre=genre, behavior=behavior, **kwargs)

    monkeypatch.setattr(jr, "_judge_reruns_for_cell", _spy_reruns)

    # Pre-seed a tiny synthetic snapshot for both behaviors (no HF, no API, no GPU).
    dest = tmp_path / "snapshot"
    for beh in behaviors:
        jr.seed_synthetic_snapshot(dest, genre=genres[0], behavior=beh)
    judge_fn = jr.make_counting_judge()

    common = dict(
        genres=genres,
        behaviors=behaviors,
        r_rerun=2,
        j_completions=4,
        dry_run=False,
        judge_fn=judge_fn,
        dest_override=dest,
        skip_snapshot=True,
        out_dir=tmp_path / "out",
    )

    # --- Run 1: simulate a crash after the FIRST cell by pre-writing its partial only,
    # then confirming a full run over BOTH cells only judges the SECOND. We emulate the
    # crash by running the FIRST behavior alone (writes its partial), then re-running the
    # full two-behavior set and asserting the first is skipped.
    jr.run(**{**common, "behaviors": behaviors[:1]})
    partial_dir = jr._partial_dir(tmp_path / "out")
    assert (partial_dir / f"{genres[0]}__{behaviors[0]}.json").exists()
    assert judged_cells == [(genres[0], behaviors[0])]

    judged_cells.clear()
    # --- Run 2: full set. The first cell's partial exists → it is SKIPPED; only the
    # second cell is judged this run.
    result = jr.run(**common)
    assert judged_cells == [(genres[0], behaviors[1])], (
        "re-run must skip the already-checkpointed cell and judge only the remaining one"
    )

    # The merged result carries BOTH cells (the fresh one + the resumed partial).
    jv = result["judge_variance"]
    assert set(jv[genres[0]]) == set(behaviors), jv
    for beh in behaviors:
        assert "sqrt_r_yy_honest" in jv[genres[0]][beh]
    # both partials now on disk
    for beh in behaviors:
        assert (partial_dir / f"{genres[0]}__{beh}.json").exists()


def test_layer2_atomic_write_and_merge_roundtrip(tmp_path):
    """The atomic per-cell write + merge reconstruct the nested judge_variance dict from
    partial files alone (the resume aggregation reads ALL partials, not in-memory state)."""
    jr = importlib.import_module("issue742_judge_rerun")
    out_dir = tmp_path / "out"
    jr._atomic_write_json(
        jr._partial_path(out_dir, "betley", "broad_em"),
        {"genre": "betley", "behavior": "broad_em", "result": {"sqrt_r_yy_honest": 0.9}},
    )
    jr._atomic_write_json(
        jr._partial_path(out_dir, "ultrachat", "refusal"),
        {"genre": "ultrachat", "behavior": "refusal", "result": {"sqrt_r_yy_honest": 0.5}},
    )
    merged = jr._merge_partial_cells(out_dir)
    assert merged == {
        "betley": {"broad_em": {"sqrt_r_yy_honest": 0.9}},
        "ultrachat": {"refusal": {"sqrt_r_yy_honest": 0.5}},
    }
    # a malformed partial is skipped, not fatal
    (jr._partial_dir(out_dir) / "betley__garbage.json").write_text("{not json")
    merged2 = jr._merge_partial_cells(out_dir)
    assert set(merged2["betley"]) == {"broad_em"}  # garbage cell dropped
