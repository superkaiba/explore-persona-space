"""Round-2 regression tests for the issue #2220 judge/reduce subsystem.

Covers the code-review v1 blockers:
  F1 — Batch-custom_id-safe judge context ids: '__'-free, <= 49 chars, unique,
       validated through the REAL rollout_item_id + validate_batch_custom_ids
       (the round-1 form embedded the '__'-joined cell_id and raised at the
       first item; its [:40] truncation collapsed (seed, q) suffixes).
  F3 — operating-point argmax restricted to coherence-PASSING cells.
  F4 — selection-symmetric (per-null-direction argmax-matched) null band with
       question-cluster paired-bootstrap draws; fail-loud missing alpha0.
  F5 — judge draw count threads per reduce phase (3 localize / 5 decisive).
  F7 — build_answer_pools producer for the fixed +/- margin pools.

Everything is synthetic and CPU-only: zero network, zero GPU. The judge call
is faked ONLY at the external API boundary, with a fake whose ``def`` mirrors
``judge_items_graded``'s signature and returns a REAL ``JudgeResult`` (the
production reduce bodies all execute for real). Content hygiene: completions
are benign placeholder strings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.issue2220_readwrite as rw
from explore_persona_space.eval.graded_judge import JudgeResult
from explore_persona_space.eval.judge_dispatch import validate_batch_custom_ids
from explore_persona_space.experiments.issue_1739.judging import (
    MAX_ITEM_ID_LEN,
    rollout_item_id,
)

# ---------------------------------------------------------------------------
# F1 — cid grammar over the full production cell lattice
# ---------------------------------------------------------------------------


def test_judge_context_id_grammar_full_lattice():
    """Every production-shape (cell, seed, q, draw) id passes the real
    validators, stays inside the 53-char item budget, and is unique."""
    ids: set[str] = set()
    for behavior in rw.BEHAVIORS:
        for direction in (*rw.DIRECTIONS, "alpha0"):
            for position in rw.POSITIONS:
                for layer in rw.LAYERS:
                    for c in (*rw.DOSES_NONZERO, 0.0):
                        cell = {
                            "behavior": behavior,
                            "direction": direction,
                            "position": position,
                            "layer": layer,
                            "c": c,
                        }
                        for seed in rw.SEEDS_DECISIVE:
                            cid = rw.judge_context_id(cell, seed, rw.Q2_DECISIVE - 1)
                            assert "__" not in cid
                            assert len(cid) <= rw._JUDGE_CONTEXT_ID_MAX, cid
                            item_id = rollout_item_id(cid, rw.DRAWS_DECISIVE - 1)
                            assert len(item_id) <= MAX_ITEM_ID_LEN
                            assert item_id not in ids
                            ids.add(item_id)
    # judge_graded appends "__{idx:05d}__{ci:02d}" — emulate the full custom_id.
    validate_batch_custom_ids(f"{i}__{0:05d}__{0:02d}" for i in ids)


def test_judge_context_id_rejects_unknown_slug():
    cell = {"behavior": "evil", "direction": "nope", "position": "context", "layer": 10, "c": 1.0}
    with pytest.raises(ValueError, match="unknown direction/position"):
        rw.judge_context_id(cell, 42, 0)


# ---------------------------------------------------------------------------
# synthetic completion fixtures + a signature-conformant judge fake
# ---------------------------------------------------------------------------

N_Q = 3
DRAWS = 2
SEED = 42

# Per-direction synthetic judge scores by question index. score >= 50 counts
# toward the rate. Designed so:
#   - alpha0 rate 0 everywhere;
#   - rb c=1.0 (COHERENT): per-q rates [1, 0, 1] -> pooled 2/3;
#   - rb c=2.0 (INCOHERENT): all 90 -> rate 1.0 (higher delta -> the gate must
#     reject it, F3);
#   - shuffled c=1.0: per-q [1, 0, 0] -> rate 1/3 (the null-band argmax);
#   - shuffled c=2.0 / random c=1.0: rate 0.
_SCORES_BY_DIR = {
    "alpha0": {0: 10.0, 1: 10.0, 2: 10.0},
    ("rb", 1.0): {0: 90.0, 1: 10.0, 2: 90.0},
    ("rb", 2.0): {0: 90.0, 1: 90.0, 2: 90.0},
    ("shuffled", 1.0): {0: 90.0, 1: 10.0, 2: 10.0},
    ("shuffled", 2.0): {0: 10.0, 1: 10.0, 2: 10.0},
    ("random", 1.0): {0: 10.0, 1: 10.0, 2: 10.0},
}


def _cells(coherent_rb2: bool = False) -> list[dict]:
    base = {"behavior": "evil", "position": "context", "layer": 10}
    cells = [
        {**base, "direction": "alpha0", "c": 0.0},
        {**base, "direction": "rb", "c": 1.0},
        {**base, "direction": "rb", "c": 2.0},
        {**base, "direction": "shuffled", "c": 1.0},
        {**base, "direction": "shuffled", "c": 2.0},
        {**base, "direction": "random", "c": 1.0},
    ]
    return cells


def _write_completions(out_root: Path, phase: str, cells: list[dict]) -> None:
    comp_root = out_root / phase / "raw_completions"
    comp_root.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        cell_id = rw._cell_id(cell)
        coherent = not (cell["direction"] == "rb" and cell["c"] == 2.0)
        flags = [[coherent] * DRAWS for _ in range(N_Q)]
        rows = {
            "cell_id": cell_id,
            "cell": cell,
            "alpha": 0.0 if cell["c"] == 0.0 else 1.0,
            "max_new_tokens": rw.GEN_MAX_NEW_TOKENS,
            "cap_hit_fraction": 0.0,
            "seeds": {
                str(SEED): {
                    # cell-tagged texts so the pools' exact-text dedup keeps
                    # per-cell candidates distinct
                    "completions": [
                        [
                            f"placeholder {cell['direction']} c{cell['c']} answer {qi}-{di}"
                            for di in range(DRAWS)
                        ]
                        for qi in range(N_Q)
                    ],
                    "coherent_flags": flags,
                    "condition_passes": [all(f) for f in flags],
                }
            },
        }
        (comp_root / f"{cell_id}.json").write_text(json.dumps(rows))


def _score_for(cell: dict, qi: int) -> float:
    key = "alpha0" if cell["direction"] == "alpha0" else (cell["direction"], cell["c"])
    return _SCORES_BY_DIR[key][qi]


class _FakeJudge:
    """Signature-conformant fake of judge_items_graded (external API boundary).

    Mirrors the real keyword surface, validates every composed custom_id with
    the REAL validator, records n_draws per call (F5), and returns a REAL
    JudgeResult whose scores follow _SCORES_BY_DIR for the current cell."""

    def __init__(self):
        self.n_draws_seen: list[int] = []
        self.current_cell: dict | None = None

    def __call__(
        self,
        items,
        eval_prompt,
        *,
        cache_dir,
        save_raw,
        n_draws=3,
        temperature=1.0,
        max_tokens=400,
        judge_model="claude-sonnet-4-5-20250929",
        dry_run=False,
        threshold_base=None,
    ):
        self.n_draws_seen.append(n_draws)
        assert "{question}" in eval_prompt and "{answer}" in eval_prompt
        scores: dict[str, float] = {}
        per_item: dict[str, list[float]] = {}
        for item_id, question, _answer in items:
            assert "__" not in item_id and len(item_id) <= MAX_ITEM_ID_LEN
            validate_batch_custom_ids([f"{item_id}__{0:05d}__{0:02d}"])
            assert question.startswith("q"), question  # real question threaded (not [eval_q_])
            qi = int(item_id.rsplit("-q", 1)[1].split("_k")[0])
            s = _score_for(self.current_cell, qi)
            scores[item_id] = s
            per_item[item_id] = [s] * n_draws
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        Path(save_raw).parent.mkdir(parents=True, exist_ok=True)
        Path(save_raw).write_text(json.dumps({"fake": True, "n_items": len(items)}))
        return JudgeResult(
            scores=scores,
            n_total_draws=len(items) * n_draws,
            n_dropped_draws=0,
            per_item_draw_counts={i: n_draws for i in scores},
            per_item_scores=per_item,
        )


@pytest.fixture()
def reduce_env(tmp_path, monkeypatch):
    """Wire phase_judge_reduce's external boundaries to synthetic fakes."""
    fake = _FakeJudge()

    def _judge_shim(items, eval_prompt, **kw):
        return fake(items, eval_prompt, **kw)

    # The driver imports these inside the phase body -> patch the SOURCE module.
    monkeypatch.setattr(
        "explore_persona_space.experiments.issue_1739.judging.judge_items_graded",
        _judge_shim,
    )
    monkeypatch.setattr(
        "explore_persona_space.experiments.issue_1739.judging.load_trait_rubric",
        lambda behavior, **kw: "Rate {question} / {answer} 0-100.",
    )
    monkeypatch.setattr(rw, "_eval_questions", lambda b: [f"q{i}" for i in range(20)])
    monkeypatch.setattr(rw, "_upload_judge_outputs", lambda out_root, phase: None)
    # A8 guard loaders faked at the HF/asset boundary ONLY — the REAL
    # _assert_eval_bank_disjoint body executes inside phase_judge_reduce
    # (disjoint sets here; the overlap test below flips them).
    monkeypatch.setattr(rw, "_corpus_query_texts", lambda: {"corpus-only text"})
    monkeypatch.setattr(
        rw, "_extraction_question_texts", lambda b: {f"x-{b}-{i}" for i in range(20)}
    )
    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path / "sentinels"))
    return fake


def _args(out_root: Path, reduce_phase: str):
    ap = rw.build_argparser()
    return ap.parse_args(
        ["--phase", "judge_reduce", "--reduce-phase", reduce_phase, "--out-root", str(out_root)]
    )


def _run_reduce(tmp_path, reduce_env, phase: str):
    out_root = tmp_path / "out"
    cells = _cells()
    _write_completions(out_root, phase, cells)

    # thread the current cell into the fake by intercepting per-file order:
    # phase_judge_reduce iterates files sorted by cell_id; wrap the shim to
    # track which cell each call belongs to via the save_raw filename.
    orig_call = reduce_env.__call__

    def _tracking(items, eval_prompt, **kw):
        cell_id = Path(kw["save_raw"]).name
        by_id = {rw._cell_id(c): c for c in cells}
        reduce_env.current_cell = by_id[cell_id]
        return orig_call(items, eval_prompt, **kw)

    import explore_persona_space.experiments.issue_1739.judging as judging_mod

    judging_mod.judge_items_graded = _tracking  # monkeypatch fixture restores
    rw.phase_judge_reduce(_args(out_root, phase))
    return out_root


# ---------------------------------------------------------------------------
# F3 / F4 / F5 — the reduce statistics
# ---------------------------------------------------------------------------


def test_phase_judge_reduce_end_to_end(tmp_path, reduce_env):
    out_root = _run_reduce(tmp_path, reduce_env, "localize")
    # §6.5/§9 plan-named deliverable: the localize reduced surface IS
    # localize/dose_response.json (no stray reduced.json sibling).
    assert (out_root / "localize" / "dose_response.json").exists()
    assert not (out_root / "localize" / "reduced.json").exists()
    reduced = json.loads((out_root / "localize" / "dose_response.json").read_text())["reduced"]

    # F5: localize judge draws = 3.
    assert set(reduce_env.n_draws_seen) == {rw.JUDGE_DRAWS_LOCALIZE}

    delta = reduced["delta_rate"]["evil"]
    rb1 = rw._cell_id(
        {"behavior": "evil", "direction": "rb", "position": "context", "layer": 10, "c": 1.0}
    )
    rb2 = rw._cell_id(
        {"behavior": "evil", "direction": "rb", "position": "context", "layer": 10, "c": 2.0}
    )
    assert delta[rb1]["delta_rate"] == pytest.approx(2 / 3)
    assert delta[rb2]["delta_rate"] == pytest.approx(1.0)
    assert delta[rb2]["coherence_pass"] is False  # the incoherent cell

    # F3: the operating point must be the COHERENT rb cell (c=1.0), not the
    # higher-delta incoherent c=2.0 cell.
    ops = reduced["operating_points"]["evil"]
    assert ops["rb__context"]["c"] == 1.0
    assert ops["rb__context"]["layer"] == 10
    # nulls get operating points too (decisive runs them at their own argmax)
    assert ops["shuffled__context"]["c"] == 1.0

    # F4: argmax-matched band — per null direction argmax over its gated cells,
    # max over null dirs; here shuffled's max delta (1/3) dominates random (0).
    band = reduced["null_band"]["evil"]
    assert band["upper_edge_point"] == pytest.approx(1 / 3)
    assert band["per_null_max"]["shuffled"] == pytest.approx(1 / 3)
    assert band["per_null_max"]["random"] == pytest.approx(0.0)
    assert band["n_null_cells_gated"] == 3
    assert "argmax" in band["selection"]
    assert np.isfinite(band["upper_edge_boot97p5"])

    # paired question-cluster bootstrap: rb c=1.0 has per-q rates [1,0,1] so
    # the resampled delta varies -> a non-degenerate CI containing the point.
    lo, hi = delta[rb1]["ci95"]
    assert lo < hi
    assert lo <= delta[rb1]["delta_rate"] <= hi

    # per-cell records carry the question-cluster grain + coherence rate
    per_cell = json.loads((out_root / "localize" / "dose_response.json").read_text())["per_cell"]
    assert per_cell[rb1]["per_question_rate"] == {"0": 1.0, "1": 0.0, "2": 1.0}
    assert per_cell[rb2]["coherence_rate"] == 0.0
    # judge_items tallies persisted per cell (pools + rejudge consumers)
    assert (out_root / "localize" / "judge_items" / f"{rb1}.json").exists()
    # operating_points.json written for the decisive phase to consume
    assert (out_root / "localize" / "operating_points.json").exists()


def test_decisive_reduce_writes_percell_deliverable_and_draws(tmp_path, reduce_env):
    out_root = _run_reduce(tmp_path, reduce_env, "decisive")
    # F5: decisive judge draws = 5.
    assert set(reduce_env.n_draws_seen) == {rw.JUDGE_DRAWS_DECISIVE}
    # §9 phase_outputs literal: the decisive reduced surface IS judged.json.
    judged = out_root / "decisive" / "judged.json"
    assert judged.exists()
    assert not (out_root / "decisive" / "reduced.json").exists()
    assert "reduced" in json.loads(judged.read_text())
    # F4: the §6.5 plan-named deliverable exists and carries the band + CIs.
    p = out_root / "decisive" / "delta_rate_percell.json"
    assert p.exists()
    payload = json.loads(p.read_text())
    assert "null_band" in payload and "delta_rate" in payload
    rb1 = rw._cell_id(
        {"behavior": "evil", "direction": "rb", "position": "context", "layer": 10, "c": 1.0}
    )
    assert "ci95" in payload["delta_rate"]["evil"][rb1]


def test_reduce_surface_fails_loud_without_alpha0():
    per_cell = {
        "cell": {
            "cell": {
                "behavior": "evil",
                "direction": "rb",
                "position": "context",
                "layer": 10,
                "c": 1.0,
            },
            "rate": 0.5,
            "per_question_rate": {"0": 0.5},
            "coherence_rate": 1.0,
        }
    }
    with pytest.raises(RuntimeError, match="alpha0 reference"):
        rw._reduce_surface(per_cell, "localize")


# ---------------------------------------------------------------------------
# F7 — build_answer_pools + pack helper + misc units
# ---------------------------------------------------------------------------


def test_build_answer_pools_from_judged_completions(tmp_path, reduce_env, monkeypatch):
    out_root = _run_reduce(tmp_path, reduce_env, "localize")
    monkeypatch.setattr(rw, "_upload_pools", lambda pools_dir: None)
    ap = rw.build_argparser()
    args = ap.parse_args(
        ["--phase", "build_answer_pools", "--out-root", str(out_root), "--behaviors", "evil"]
    )
    rw.phase_build_answer_pools(args)
    pools = json.loads((out_root / "margin" / "pools" / "evil.json").read_text())
    assert pools["pos"] and pools["neg"]
    assert all(s >= rw.SCORE_THRESHOLD for s in pools["pool_meta"]["pos_scores"])
    assert all(s < rw.SCORE_THRESHOLD for s in pools["pool_meta"]["neg_scores"])
    # the incoherent rb c=2.0 draws are excluded: every pool item id must come
    # from a coherent cell (rb c=2.0 cids carry both '-rb-' and '-c2p0-')
    for iid in pools["pool_meta"]["pos_item_ids"]:
        assert not ("-rb-" in iid and "-c2p0-" in iid)


def test_build_answer_pools_fails_loud_below_floor(tmp_path, reduce_env, monkeypatch):
    out_root = _run_reduce(tmp_path, reduce_env, "localize")
    monkeypatch.setattr(rw, "_upload_pools", lambda pools_dir: None)
    # raise the floor above the tiny synthetic pos yield -> must raise, not ship
    monkeypatch.setattr(rw, "POOL_MIN", 50)
    ap = rw.build_argparser()
    args = ap.parse_args(
        ["--phase", "build_answer_pools", "--out-root", str(out_root), "--behaviors", "evil"]
    )
    with pytest.raises(RuntimeError, match="below floor"):
        rw.phase_build_answer_pools(args)


def test_pack_tree_to_jsonl_shards_roundtrip(tmp_path):
    src = tmp_path / "judge_cache"
    src.mkdir()
    docs = {f"{i:02d}.json": {"k": i} for i in range(7)}
    for name, doc in docs.items():
        (src / name).write_text(json.dumps(doc))
    dest = tmp_path / "pack"
    n = rw._pack_tree_to_jsonl_shards(src, dest, group="judge_cache", shard_bytes=64)
    manifest = json.loads((dest / "pack_manifest.json").read_text())
    assert manifest["n_files"] == 7 and n == len(manifest["shards"]) and n >= 2
    unpacked = {}
    for shard in manifest["shards"]:
        for line in (dest / shard).read_text().split("\n"):
            if line.strip():
                row = json.loads(line)
                unpacked[row["path"]] = row["doc"]
    assert unpacked == docs


def test_judge_pilot_gate_raises_on_truncation_or_drops():
    """Plan §9 pilot gate: budget-truncated draws or >=2% content drops on the
    first judged cell must HALT the wave (fails pre-fix: no gate existed)."""
    ok = JudgeResult(scores={}, n_total_draws=100, n_dropped_draws=1)
    rw._judge_pilot_gate(ok, "cell")  # 1% drops, no truncation -> passes
    trunc = JudgeResult(
        scores={}, n_total_draws=100, n_dropped_draws=0, stop_reason_tally={"max_tokens": 3}
    )
    with pytest.raises(RuntimeError, match="pilot gate"):
        rw._judge_pilot_gate(trunc, "cell")
    drops = JudgeResult(scores={}, n_total_draws=100, n_dropped_draws=5)
    with pytest.raises(RuntimeError, match="pilot gate"):
        rw._judge_pilot_gate(drops, "cell")


def test_needs_cap_regen_predicate():
    assert rw._needs_cap_regen({"cap_hit_fraction": 0.05}) is True
    assert rw._needs_cap_regen({"cap_hit_fraction": 0.01}) is False
    # already regenerated at the doubled cap -> never loops
    assert (
        rw._needs_cap_regen(
            {
                "cap_hit_fraction": 0.5,
                "max_new_tokens": rw.CAP_HIT_REGEN_FACTOR * rw.GEN_MAX_NEW_TOKENS,
            }
        )
        is False
    )


def test_upload_judge_outputs_packs_and_uploads(tmp_path, monkeypatch):
    """Real _upload_judge_outputs body up to the network boundary: the cache
    tree is packed, and upload_folder is called with the pack + judge dirs in
    its allow_patterns (HfApi.upload_folder autospec'd — signature-conformant)."""
    from unittest.mock import patch

    from huggingface_hub import HfApi

    out_root = tmp_path / "out"
    phase_dir = out_root / "localize"
    (phase_dir / "judge_cache" / "cell_a").mkdir(parents=True)
    (phase_dir / "judge_cache" / "cell_a" / "aa.json").write_text('{"score": 1}')
    (phase_dir / "judge_raw").mkdir()
    (phase_dir / "judge_raw" / "cell_a").write_text("{}")
    (phase_dir / "dose_response.json").write_text("{}")

    with patch.object(HfApi, "upload_folder", autospec=True) as spec:
        rw._upload_judge_outputs(out_root, "localize")
    assert (phase_dir / "judge_cache_pack" / "pack_manifest.json").exists()
    assert spec.called
    kwargs = spec.call_args.kwargs
    assert kwargs["path_in_repo"].endswith("judge/localize")
    assert "judge_cache_pack/*" in kwargs["allow_patterns"]
    assert "judge_raw/*" in kwargs["allow_patterns"]
    # the phase-keyed plan-named reduced surface is upload-eligible
    assert "dose_response.json" in kwargs["allow_patterns"]


# ---------------------------------------------------------------------------
# round 3 — NaN-safe null band + A8 eval-bank disjointness gate
# ---------------------------------------------------------------------------


def _rec(direction: str, c: float, rate, per_q: dict, coh: float = 1.0) -> dict:
    """A _per_cell_record-shaped synthetic record (position/layer fixed)."""
    return {
        "cell": {
            "behavior": "evil",
            "direction": direction,
            "position": "context",
            "layer": 10,
            "c": c,
        },
        "rate": rate,
        "per_question_rate": per_q,
        "coherence_rate": coh,
    }


def test_null_band_point_max_ignores_nan_delta():
    """A coherence-PASSING null cell whose judge draws ALL dropped (rate NaN)
    must not poison the null-band point max (pre-fix: Python max() propagated
    NaN order-dependently into upper_edge_point; code-review v2 minor)."""
    per_cell = {
        "a0": _rec("alpha0", 0.0, 0.0, {"0": 0.0, "1": 0.0}),
        # "shf_0nan" sorts BEFORE "shf_1fin" -> pre-fix max() starts at NaN and
        # sticks there (every NaN comparison is False) — the order-dependence.
        "shf_0nan": _rec("shuffled", 1.0, float("nan"), {}),
        "shf_1fin": _rec("shuffled", 2.0, 0.1, {"0": 0.2, "1": 0.0}),
        "rnd_fin": _rec("random", 1.0, 0.0, {"0": 0.0, "1": 0.0}),
    }
    band = rw._reduce_surface(per_cell, "decisive")["null_band"]["evil"]
    assert band["per_null_max"]["shuffled"] == pytest.approx(0.1)
    assert band["per_null_max"]["random"] == pytest.approx(0.0)
    assert band["upper_edge_point"] == pytest.approx(0.1)
    assert np.isfinite(band["upper_edge_point"])


def test_null_band_all_nan_direction_is_none_not_crash():
    """Every gated cell of a null direction NaN -> that direction's point max is
    None and the overall edge falls back to the finite direction (no ValueError
    from max() over an empty generator)."""
    per_cell = {
        "a0": _rec("alpha0", 0.0, 0.0, {"0": 0.0}),
        "shf_nan": _rec("shuffled", 1.0, float("nan"), {}),
        "rnd_fin": _rec("random", 1.0, 0.05, {"0": 0.05}),
    }
    band = rw._reduce_surface(per_cell, "decisive")["null_band"]["evil"]
    assert band["per_null_max"]["shuffled"] is None
    assert band["upper_edge_point"] == pytest.approx(0.05)


def test_assert_eval_bank_disjoint_pass_and_fail(monkeypatch):
    """REAL _assert_eval_bank_disjoint body: PASS returns the record; overlap
    with the extraction set or the corpus raises with counts, never text."""
    monkeypatch.setattr(rw, "_eval_questions", lambda b: [f"q{i}" for i in range(20)])
    monkeypatch.setattr(rw, "_extraction_question_texts", lambda b: {"x1", "x2"})
    record = rw._assert_eval_bank_disjoint(["evil"], corpus_texts={"c1", "c2"})
    assert record["behaviors"]["evil"]["n_eval"] == 20
    assert record["behaviors"]["evil"]["n_overlap_extraction"] == 0
    assert record["behaviors"]["evil"]["n_overlap_corpus"] == 0
    assert record["behaviors"]["evil"]["eval_bank_sha8"]

    # extraction overlap (normalization applied: whitespace/case-insensitive)
    monkeypatch.setattr(rw, "_extraction_question_texts", lambda b: {"q3", "x2"})
    with pytest.raises(RuntimeError, match="disjointness FAILED"):
        rw._assert_eval_bank_disjoint(["evil"], corpus_texts={"c1"})

    # corpus overlap
    monkeypatch.setattr(rw, "_extraction_question_texts", lambda b: {"x1"})
    with pytest.raises(RuntimeError, match="map-fit corpus"):
        rw._assert_eval_bank_disjoint(["evil"], corpus_texts={"q4", "c1"})

    # the normalization grain: an eval question differing only in case /
    # whitespace still counts as overlap ("Q7" vs eval "q7")
    monkeypatch.setattr(rw, "_eval_questions", lambda b: ["  Q7 ", "q8"])
    with pytest.raises(RuntimeError, match="overlap"):
        rw._assert_eval_bank_disjoint(["evil"], corpus_texts={"q7"})

    # error message carries digests/counts, never question text ('q' is not a
    # hex char, so a sha8 digest can never contain the raw question)
    try:
        rw._assert_eval_bank_disjoint(["evil"], corpus_texts={"q7"})
    except RuntimeError as exc:
        assert "q7" not in str(exc)
        assert "1 eval questions overlap" in str(exc)


def test_judge_reduce_fails_loud_on_eval_bank_overlap(tmp_path, reduce_env, monkeypatch):
    """The A8 guard FIRES inside the spend phase: an eval bank overlapping the
    extraction set halts phase_judge_reduce before any judging."""
    monkeypatch.setattr(rw, "_extraction_question_texts", lambda b: {"q0"})
    out_root = tmp_path / "out"
    _write_completions(out_root, "localize", _cells())
    with pytest.raises(RuntimeError, match="A8 eval-bank disjointness FAILED"):
        rw.phase_judge_reduce(_args(out_root, "localize"))
    assert reduce_env.n_draws_seen == []  # halted BEFORE any judge call


def test_eval_questions_fail_loud_on_missing_key(monkeypatch):
    """No 'eval_questions' key -> raise; NEVER the silent extraction tail-slice
    (a wrong slice would steer/judge on direction-fit questions; plan §6 A8)."""
    monkeypatch.setattr(rw, "_e1_assets", lambda b: {"extraction_questions": ["x"] * 40})
    with pytest.raises(RuntimeError, match="eval_questions"):
        rw._eval_questions("evil")
    # too-small bank also refuses
    monkeypatch.setattr(rw, "_e1_assets", lambda b: {"eval_questions": ["q"] * 5})
    with pytest.raises(RuntimeError, match="too small"):
        rw._eval_questions("evil")


def test_phase_check_disjoint_writes_record(tmp_path, monkeypatch):
    """Standalone A8 phase: REAL guard body + record persisted to
    check_disjoint/disjointness.json (loaders faked at the HF/asset boundary)."""
    monkeypatch.setattr(rw, "_eval_questions", lambda b: [f"q{i}" for i in range(20)])
    monkeypatch.setattr(rw, "_extraction_question_texts", lambda b: {f"x-{b}"})
    monkeypatch.setattr(rw, "_corpus_query_texts", lambda: {"corpus-only text"})
    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path / "sentinels"))
    out_root = tmp_path / "out"
    ap = rw.build_argparser()
    args = ap.parse_args(
        ["--phase", "check_disjoint", "--out-root", str(out_root), "--behaviors", "evil"]
    )
    rw.phase_check_disjoint(args)
    rec = json.loads((out_root / "check_disjoint" / "disjointness.json").read_text())
    assert rec["behaviors"]["evil"]["n_overlap_extraction"] == 0
    assert rec["behaviors"]["evil"]["n_overlap_corpus"] == 0
    assert rec["n_corpus_query_texts"] == 1
