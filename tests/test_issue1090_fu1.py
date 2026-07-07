"""Offline tests for the #1090 ``fu1-margin-qwen`` follow-up driver.

Covers: the topup-union fixed-pool construction (kept-flag filtering, ZERO
re-judging, deterministic + fixed-across-invocations, cap), the margin
aggregation / rho / adapter-assert math, the fu1 judged-rate reduction
(drop-never-coerce, per-question), the closure per-item rate reduction, and
the c5 top-up gluing (frozen-yield-DV invariant at the RECORDED 1.0 budget +
the default 2.0 fence staying binding).

External-API boundaries (generator / judge) are stubbed signature-conformant;
the topup test executes the PRODUCTION ``_run_topup_cell`` body (the
seam-stub/body-coverage split mirrors test_issue1090_pvdatagen.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fu1_module():
    """Import scripts/issue1090_fu1.py (self-inserts scripts/ on sys.path)."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1090_fu1 as fu1
    finally:
        sys.path.remove(str(REPO_ROOT))
    return fu1


def _run_module():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1090_run as run
    finally:
        sys.path.remove(str(REPO_ROOT))
    return run


# ── Shared fixture builders (production writers, synthetic completions) ──────


def _mk_cfg(run, tmp_path, cells):
    # target 6 -> floor 5; near-miss floor = 5 - ceil(0.5) = 4; member quota 1.
    return run.RunConfig(
        smoke=False, cells=tuple(cells), out_root=tmp_path, target_n=6, n_judge_draws=2
    )


def _build_first_sample(run, cfg, cell, *, n_raw=8, n_keep=4, mult):
    """Near-miss first sample through the production compose/raw/judge-save
    writers (the test_issue1090_pvdatagen fixture shape)."""
    import json as _json
    import math as _math

    from explore_persona_space.artifacts.behavior import BEHAVIORS
    from explore_persona_space.artifacts.datagen import GenCandidate, _write_raw
    from explore_persona_space.artifacts.negatives import DEFAULT_PANEL_NAME, get_panel

    behavior = BEHAVIORS[cell.behavior]
    panel = get_panel(DEFAULT_PANEL_NAME)
    dgdir = cfg.out_root / cell.slug / "datagen"
    dgdir.mkdir(parents=True)
    manifest = run._reconstruct_manifest(cfg, cell, behavior, panel, mult)
    (dgdir / "gen_manifest.json").write_text(_json.dumps(manifest) + "\n")
    exhibit, _ne = run._resolve_instructions(behavior, "extraction_pairs")
    tq = [
        (f"{behavior.name}-trainq-{i:04d}", q) for i, q in enumerate(behavior.train_question_bank)
    ]
    reqs = run._compose_positive_requests(
        behavior, run._source_context(), tq, n_raw, run._rng(cfg.seed), "plain", variants=exhibit
    )
    cands = [GenCandidate(r, f"agree::{r.request_id}") for r in reqs]
    _write_raw(dgdir / "raw_pos.jsonl", cands)
    all_scores = {}
    for i, c in enumerate(cands):
        for d in range(cfg.n_judge_draws):
            all_scores[f"{c.request.request_id}__{i:05d}__{d:02d}"] = 80.0 if i < n_keep else 20.0
    (dgdir / "judge_raw_pos.json").write_text(_json.dumps({"all_scores": all_scores}))
    floor_n = _math.ceil(cfg.quota_floor * cfg.target_n)
    summary = {
        "cell": cell.slug,
        "cell_id": cell.cell_id,
        "behavior": cell.behavior,
        "generator": cell.generator,
        "status": "yield_floor_missed",
        "oversample_mult": mult,
        "target_n": cfg.target_n,
        "quota_floor": cfg.quota_floor,
        "floor_n": floor_n,
        "seed": cfg.seed,
        "positive_stage": {"n_kept": n_keep, "n_requested": n_raw},
        "yield_record": {
            "kept_pos": n_keep,
            "floor_n": floor_n,
            "message": f"kept {n_keep} positives < floor_n={floor_n}",
            "stages": {"positive": {"requested": n_raw}},
        },
        "per_question_yield": {},
    }
    (cfg.out_root / cell.slug / "datagen_summary.json").write_text(_json.dumps(summary))
    return summary, cands


def _build_topup_sidecars(run, cfg, cell, *, n_pos_raw=3, n_pos_keep=2, n_neg_raw=4, n_neg_keep=3):
    """datagen_topup raw/kept sidecars via the PRODUCTION topup writers."""
    from explore_persona_space.artifacts.behavior import BEHAVIORS
    from explore_persona_space.artifacts.datagen import GenCandidate
    from explore_persona_space.artifacts.negatives import DEFAULT_PANEL_NAME, get_panel

    behavior = BEHAVIORS[cell.behavior]
    panel = get_panel(DEFAULT_PANEL_NAME)
    td = run._topup_dir(cfg.out_root / cell.slug)
    td.mkdir(parents=True, exist_ok=True)
    exhibit, not_exhibit = run._resolve_instructions(behavior, "extraction_pairs")
    tq = [
        (f"{behavior.name}-trainq-{i:04d}", q) for i, q in enumerate(behavior.train_question_bank)
    ]
    pos_reqs = run._topup_ids(
        run._compose_positive_requests(
            behavior,
            run._source_context(),
            tq,
            n_pos_raw,
            run._rng(cfg.seed + run.TOPUP_SEED_OFFSET),
            "plain",
            variants=exhibit,
        )
    )
    pos_cands = [GenCandidate(r, f"tagree::{r.request_id}") for r in pos_reqs]
    run._write_raw_topup(td / "raw_pos.jsonl", pos_cands)
    run._write_kept_topup(td / "kept_pos.jsonl", pos_cands[:n_pos_keep])
    neg_reqs = run._topup_ids(
        run._compose_negative_requests(
            behavior,
            panel,
            run._dedup_questions(pos_cands[:n_pos_keep]),
            max(1, n_neg_raw // len(panel)),
            run._rng(cfg.seed + run.TOPUP_SEED_OFFSET + 1),
            "plain",
            not_exhibit=not_exhibit,
        )
    )[:n_neg_raw]
    neg_cands = [GenCandidate(r, f"tdisagree::{r.request_id}") for r in neg_reqs]
    run._write_raw_topup(td / "raw_neg.jsonl", neg_cands)
    run._write_kept_topup(td / "kept_neg.jsonl", neg_cands[:n_neg_keep])
    return pos_cands, neg_cands


def _gen_stub():
    from explore_persona_space.artifacts.datagen import GenCandidate

    def gen(requests):
        return [GenCandidate(r, f"resp::{r.request_id}") for r in requests]

    return gen


def _judge_by_arm(*, pos=80.0, neg=20.0):
    """Signature-conformant JudgeFn stub (the datagen judge seam shape)."""
    from explore_persona_space.eval.graded_judge import JudgeResult

    def judge(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False):
        scores = {rid: (pos if rid.startswith(("pos-", "tpos-")) else neg) for rid, _, _ in items}
        return JudgeResult(
            scores=scores,
            n_total_draws=len(items) * n_draws,
            n_dropped_draws=0,
            per_item_draw_counts={rid: n_draws for rid, _, _ in items},
            per_item_scores={rid: [scores[rid]] * n_draws for rid, _, _ in items},
        )

    return judge


# ── Fixed-pool construction (P2) ─────────────────────────────────────────────


def test_margin_pools_topup_union_kept_only_no_rejudge(tmp_path, monkeypatch):
    """Pools = first-sample kept (replayed from the RECORDED judge raws) +
    topup kept rows; non-kept rows excluded; negatives topup-only; and the
    construction makes ZERO judge calls (judge_graded is a landmine)."""
    from explore_persona_space.artifacts.behavior import BEHAVIORS

    fu1, run = _fu1_module(), _run_module()
    cfg = _mk_cfg(run, tmp_path, [fu1.C3])
    _build_first_sample(run, cfg, fu1.C3, n_raw=8, n_keep=4, mult=2.0)
    pos_cands, _neg_cands = _build_topup_sidecars(run, cfg, fu1.C3)

    def _landmine(*a, **k):
        raise AssertionError("judge_graded called during pool construction (re-judging is banned)")

    monkeypatch.setattr(fu1, "judge_graded", _landmine)
    monkeypatch.setattr(run, "judge_graded", _landmine)

    pos, neg, meta = fu1.derive_margin_pools_topup(
        tmp_path / fu1.C3.slug, BEHAVIORS["sycophancy"], scratch=tmp_path / "_scratch"
    )
    assert meta["n_pos_available"] == 4 + 2 and meta["n_neg_available"] == 3
    assert {p["source"] for p in neg} == {"topup"}
    kept_topup_ids = {c.request.request_id for c in pos_cands[:2]}
    got_topup_ids = {p["request_id"] for p in pos if p["source"] == "topup"}
    assert got_topup_ids == kept_topup_ids  # kept-flag filtering, non-kept excluded
    dropped_id = pos_cands[2].request.request_id
    assert dropped_id not in {p["request_id"] for p in pos}
    assert all(p["answer"] and p["probe"] for p in pos + neg)


def test_margin_pools_deterministic_fixed_and_capped(tmp_path):
    """The SAME fixed set on every invocation (rule-19 invariance) and the
    deterministic cap (first `cap` after the canonical sort)."""
    from explore_persona_space.artifacts.behavior import BEHAVIORS

    fu1, run = _fu1_module(), _run_module()
    cfg = _mk_cfg(run, tmp_path, [fu1.C3])
    _build_first_sample(run, cfg, fu1.C3, n_raw=8, n_keep=4, mult=2.0)
    _build_topup_sidecars(run, cfg, fu1.C3)
    beh = BEHAVIORS["sycophancy"]
    pos1, neg1, meta1 = fu1.derive_margin_pools_topup(
        tmp_path / fu1.C3.slug, beh, scratch=tmp_path / "_s1"
    )
    pos2, neg2, meta2 = fu1.derive_margin_pools_topup(
        tmp_path / fu1.C3.slug, beh, scratch=tmp_path / "_s2"
    )
    assert pos1 == pos2 and neg1 == neg2 and meta1["pool_sha256"] == meta2["pool_sha256"]
    keys1 = [(p["question_id"], p["variant_id"], p["request_id"]) for p in pos1]
    assert keys1 == sorted(keys1)
    pos_c, _neg_c, meta_c = fu1.derive_margin_pools_topup(
        tmp_path / fu1.C3.slug, beh, cap=4, scratch=tmp_path / "_s3"
    )
    assert len(pos_c) == 4 and meta_c["n_pos_used"] == 4
    assert pos_c == pos1[:4]  # first-cap of the SAME canonical order
    assert meta_c["pool_sha256"] != meta1["pool_sha256"]  # sha pins the realized pool


# ── c5 top-up gluing (P3) ────────────────────────────────────────────────────


def test_c5_topup_frozen_dv_at_recorded_budget_and_default_fence(tmp_path):
    """The c5 tranche runs the PRODUCTION _run_topup_cell body at the RECORDED
    1.0 budget: frozen yield-DV fields byte-unchanged, status flips to
    success_with_topup, eligible_mult recorded; the default 2.0 fence still
    refuses the same record (the parent behavior is unchanged)."""
    import dataclasses as _dc

    fu1, run = _fu1_module(), _run_module()
    cfg = _mk_cfg(run, tmp_path, [fu1.C5])
    before, _ = _build_first_sample(run, cfg, fu1.C5, n_raw=8, n_keep=4, mult=1.0)
    cfg5 = _dc.replace(cfg, cells=(fu1.C5,))

    with pytest.raises(RuntimeError, match="eligible budget"):
        run._run_topup_cell(cfg5, fu1.C5, gen_fn=_gen_stub(), judge_fn=_judge_by_arm())

    new_rec = run._run_topup_cell(
        cfg5,
        fu1.C5,
        gen_fn=_gen_stub(),
        judge_fn=_judge_by_arm(),
        eligible_mult=fu1.C5_ELIGIBLE_MULT,
    )
    assert new_rec["status"] == run.TOPUP_STATUS
    after = json.loads((tmp_path / fu1.C5.slug / "datagen_summary.json").read_text())
    for k in run._TOPUP_FROZEN_KEYS:  # the frozen yield DV — byte-unchanged
        assert after.get(k) == before.get(k), k
    assert after["yield_record"]["kept_pos"] == 4  # the reported yield stays the first sample
    tr = after["topup_record"]
    assert tr["union_cleared"] is True and tr["eligible_mult"] == 1.0
    td = tmp_path / fu1.C5.slug / "datagen_topup"
    assert all((td / f).exists() for f in ("pos.jsonl", "cn.jsonl", "pool_meta.json"))
    with pytest.raises(RuntimeError, match="EXACTLY ONE"):
        run._run_topup_cell(
            cfg5, fu1.C5, gen_fn=_gen_stub(), judge_fn=_judge_by_arm(), eligible_mult=1.0
        )


# ── Margin / rho / judge reduction math ──────────────────────────────────────


def test_aggregate_margin_reads_math():
    fu1 = _fu1_module()
    reads = {
        "base__q000": {"margin": 1.0},
        "base__q001": {"margin": 3.0},
        "trained__q000": {"margin": 2.0},
        "trained__q001": {"margin": 6.0},
        "base__source_ctx": {"margin": 0.5},
        "trained__source_ctx": {"margin": 1.5},
    }
    out = fu1.aggregate_margin_reads(reads, ["q000", "q001"])
    assert out["margin_base"] == 2.0 and out["margin_trained"] == 4.0
    assert out["margin_delta"] == 2.0
    assert out["per_context_margin"]["trained"] == {"q000": 2.0, "q001": 6.0}
    assert out["source_ctx"] == {"base": 0.5, "trained": 1.5, "delta": 1.0}


def test_spearman_rho_guards_and_monotone():
    fu1 = _fu1_module()
    r = fu1.spearman_rho([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4])
    assert r["rho"] == 1.0 and r["n"] == 4
    assert fu1.spearman_rho([1.0, 2.0], [0.1, 0.2])["rho"] is None  # n < 3
    assert fu1.spearman_rho([1.0, 1.0, 1.0], [0.1, 0.2, 0.3])["rho"] is None  # no variance
    r2 = fu1.spearman_rho([1.0, None, 2.0, 3.0], [0.1, 0.9, 0.2, 0.3])  # None pair dropped
    assert r2["n"] == 3 and r2["rho"] == 1.0


def test_adapter_assert_trips_on_identical_reads():
    fu1 = _fu1_module()
    base = {"pos_ln_logp": [-1.0, -2.0, -3.0]}
    with pytest.raises(RuntimeError, match="adapter-application assert FAILED"):
        fu1.assert_adapter_applied(base, {"pos_ln_logp": [-1.0, -2.0, -3.0]}, tol=1e-3, tag="x")
    rec = fu1.assert_adapter_applied(base, {"pos_ln_logp": [-1.0, -2.5, -3.0]}, tol=1e-3, tag="x")
    assert rec["max_abs_delta_pos_ln_logp"] == 0.5 and rec["n_pairs"] == 3


def test_reduce_judge_fu1_drop_never_coerce():
    fu1 = _fu1_module()
    indexed = [(0, "a0"), (0, "a1"), (1, "b0"), (1, "b1")]
    scores = {"a0": 80.0, "a1": None, "b0": 50.0, "b1": 90.0}  # 50.0 == threshold -> NOT positive
    out = fu1.reduce_judge_fu1(indexed, scores, threshold=50, n_questions=2)
    assert out["n"] == 3 and out["n_dropped"] == 1  # the None is DROPPED, never coerced
    assert out["k"] == 2 and out["rate"] == pytest.approx(2 / 3)
    assert out["per_question_rate"] == [1.0, 0.5]
    assert out["per_question_n"] == [1, 2]
    with pytest.raises(ValueError, match="judge-dropped"):
        fu1.reduce_judge_fu1([(0, "x")], {"x": None}, threshold=50, n_questions=1)


def test_closure_per_question_rates(tmp_path):
    fu1 = _fu1_module()
    p4 = tmp_path / "c3-trained.json"
    p4.write_text(
        json.dumps(
            {
                "per_item": [
                    {"item_id": "c3-trained-q000-c0", "closure_mean": 96.0},
                    {"item_id": "c3-trained-q000-c1", "closure_mean": 10.0},
                    {"item_id": "c3-trained-q001-c0", "closure_mean": None},  # dropped
                    {"item_id": "c3-trained-q001-c1", "closure_mean": 60.0},
                    {"item_id": "c3-trained-q005-c0", "closure_mean": 99.0},  # beyond slice
                ]
            }
        )
    )
    rates = fu1.closure_per_question_rates(p4, threshold=50, n_questions=2)
    assert rates == [0.5, 1.0]


def test_margin_contexts_per_question_ignores_probe(tmp_path):
    """The per-question context scores the fixed answers under (persona system
    + q_i) for ANY probe; source_ctx threads the pair's own probe (in-run
    parity). Smoke question-limit shrinks the context set identically."""
    fu1, run = _fu1_module(), _run_module()
    cfg = _mk_cfg(run, tmp_path, [fu1.C3])
    cfg = __import__("dataclasses").replace(cfg, eval_question_limit=2)
    questions, ctxs = fu1.margin_contexts(cfg)
    assert len(questions) == 2
    assert [lbl for lbl, _ in ctxs] == ["source_ctx", "q000", "q001"]
    q0_ctx = dict(ctxs)["q000"]
    m1, m2 = q0_ctx.messages("PROBE-A"), q0_ctx.messages("PROBE-B")
    assert m1 == m2  # pair probe deliberately ignored -> identical fixed context
    assert m1[-1]["role"] == "user" and m1[-1]["content"] == questions[0]
    src = dict(ctxs)["source_ctx"]
    assert src.messages("PROBE-A")[-1]["content"] == "PROBE-A"  # in-run construction
