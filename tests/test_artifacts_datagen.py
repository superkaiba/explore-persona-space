"""Offline tests for artifacts.datagen (task #866, Phase 0d).

The pipeline is exercised with INJECTED ``generate_fn`` / ``judge_fn`` stubs — no
network, no live model. Covers the train_row shape, the context-parity contract,
instruction-strip, judge thresholds + drop-never-coerce, the structural
keep-check, the pinned emit-exactly-floor_n contract + yield errors, same-question
negative pairing, the ~1:1 negative allocation, repeat-sampling multiplicity,
programmatic rejection, manifest-gated resume, the per-row sidecar, determinism,
and fresh per-run judge cache dirs.
"""

from __future__ import annotations

import dataclasses
import inspect
import json

import pytest

from explore_persona_space.artifacts import datagen
from explore_persona_space.artifacts.behavior import BEHAVIORS
from explore_persona_space.artifacts.context import Context, context_for_persona
from explore_persona_space.artifacts.datagen import NEGATIVE, POSITIVE, GenCandidate, GenRequest
from explore_persona_space.eval.graded_judge import JudgeResult

# A source persona (system only) DISJOINT from the default negative panel; the
# emitted positive prompt's last user turn is then the bare question (no wrap),
# which the same-question-pairing test relies on.
SRC = context_for_persona("villain")


# ── stub factories ───────────────────────────────────────────────────────────


def _gen_all(text_for=None):
    """A generate_fn stub: every request gets a non-refusal completion."""

    def gen(requests):
        return [
            GenCandidate(r, (text_for(r) if text_for else f"resp::{r.request_id}"))
            for r in requests
        ]

    return gen


def _judge_by_arm(*, pos=80.0, neg=20.0, keep_first_pos=None):
    """A judge_fn stub (judge_graded signature): score by arm prefix. When
    ``keep_first_pos`` is set, only the first N positive items score high."""

    def judge(
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model,
        dry_run=False,
        max_tokens=64,
    ):
        scores = {}
        pos_seen = 0
        for rid, _q, _a in items:
            if rid.startswith("pos-"):
                if keep_first_pos is not None and pos_seen >= keep_first_pos:
                    scores[rid] = neg  # below-threshold -> positive dropped
                else:
                    scores[rid] = pos
                pos_seen += 1
            else:
                scores[rid] = neg
        return JudgeResult(
            scores=scores,
            n_total_draws=len(items) * n_draws,
            n_dropped_draws=0,
            per_item_draw_counts={rid: n_draws for rid, _, _ in items},
            per_item_scores={rid: [scores[rid]] * n_draws for rid, _, _ in items},
        )

    return judge


def _cand(rid, arm, completion, *, qid="q0", variant="v0", question="Q?"):
    msgs = [{"role": "user", "content": question}]
    return GenCandidate(GenRequest(rid, arm, qid, variant, question, msgs, msgs), completion)


def _rows(path):
    return [json.loads(ln) for ln in path.read_text().splitlines()]


# ── _judge_and_filter unit tests (thresholds / drop-never-coerce / structural) ─


def test_judge_filter_thresholds(tmp_path):
    beh = BEHAVIORS["sycophancy"]  # threshold 50, no structural predicate

    def judge(
        items, ep, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False, max_tokens=64
    ):
        m = {"pos-0": 51.0, "pos-1": 50.0, "neg-0": 49.0, "neg-1": 50.0}
        return JudgeResult(
            scores={rid: m[rid] for rid, _, _ in items},
            n_total_draws=len(items),
            n_dropped_draws=0,
            per_item_draw_counts={rid: 1 for rid, _, _ in items},
        )

    pos_kept, pos_drops, *_ = datagen._judge_and_filter(
        beh,
        [_cand("pos-0", POSITIVE, "a"), _cand("pos-1", POSITIVE, "b")],
        POSITIVE,
        judge_fn=judge,
        n_judge_draws=1,
        cache_dir=tmp_path / "cp",
        save_raw=tmp_path / "rp.json",
    )
    assert [c.request.request_id for c in pos_kept] == ["pos-0"]  # 51 > 50 kept; 50 (==thr) dropped
    assert pos_drops.threshold_drops == 1

    neg_kept, neg_drops, *_ = datagen._judge_and_filter(
        beh,
        [_cand("neg-0", NEGATIVE, "c"), _cand("neg-1", NEGATIVE, "d")],
        NEGATIVE,
        judge_fn=judge,
        n_judge_draws=1,
        cache_dir=tmp_path / "cn",
        save_raw=tmp_path / "rn.json",
    )
    assert [c.request.request_id for c in neg_kept] == ["neg-0"]  # 49 < 50 kept; 50 (==thr) dropped
    assert neg_drops.threshold_drops == 1


def test_drop_never_coerce_propagation(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    cands = [
        _cand("pos-0", POSITIVE, "a"),
        _cand("pos-1", POSITIVE, "b"),
        _cand("pos-2", POSITIVE, None),  # refusal (no completion)
    ]

    def judge(
        items, ep, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False, max_tokens=64
    ):
        # pos-0: all draws dropped -> None; pos-1: kept. pos-2 is never judged (refusal).
        return JudgeResult(
            scores={"pos-0": None, "pos-1": 80.0},
            n_total_draws=2 * n_draws,
            n_dropped_draws=n_draws,
            per_item_draw_counts={"pos-0": 0, "pos-1": n_draws},
        )

    kept, drops, *_ = datagen._judge_and_filter(
        beh,
        cands,
        POSITIVE,
        judge_fn=judge,
        n_judge_draws=3,
        cache_dir=tmp_path / "c",
        save_raw=tmp_path / "r.json",
    )
    assert [c.request.request_id for c in kept] == ["pos-1"]
    assert drops.refusal_drops == 1  # pos-2, counted separately
    assert drops.judge_none_drops == 1  # pos-0, a None score is dropped, never coerced
    assert drops.threshold_drops == 0


def test_generation_drop_reason_split(tmp_path):
    # A transport-side dispatch error (api_error) must land in api_error_drops, NOT
    # refusal_drops — so an API outage can't inflate the yield-floor-relevant count.
    beh = BEHAVIORS["sycophancy"]
    msgs = [{"role": "user", "content": "Q?"}]

    def _drop(rid, reason):
        return GenCandidate(GenRequest(rid, POSITIVE, "q0", "v0", "Q?", msgs, msgs), None, reason)

    cands = [
        _cand("pos-0", POSITIVE, "a kept answer"),
        _drop("pos-1", "api_error"),
        _drop("pos-2", "empty"),
        _drop("pos-3", "refusal"),
    ]

    def judge(
        items, ep, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False, max_tokens=64
    ):
        return JudgeResult(
            scores={rid: 80.0 for rid, _, _ in items},
            n_total_draws=len(items),
            n_dropped_draws=0,
            per_item_draw_counts={rid: 1 for rid, _, _ in items},
        )

    kept, drops, *_ = datagen._judge_and_filter(
        beh,
        cands,
        POSITIVE,
        judge_fn=judge,
        n_judge_draws=1,
        cache_dir=tmp_path / "c",
        save_raw=tmp_path / "r.json",
    )
    assert [c.request.request_id for c in kept] == ["pos-0"]
    assert drops.generated == 1  # only the one non-None completion is judgeable
    assert drops.api_error_drops == 1  # NOT counted as a refusal
    assert drops.empty_drops == 1
    assert drops.refusal_drops == 1


def test_structural_keep_filter(tmp_path):
    list_text = "- one\n- two\n- three"
    prose_text = "This is a single flowing paragraph of prose with no list items at all."

    def judge_high(
        items, ep, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False, max_tokens=64
    ):
        return JudgeResult(
            scores={rid: 80.0 for rid, _, _ in items},
            n_total_draws=len(items),
            n_dropped_draws=0,
            per_item_draw_counts={rid: 1 for rid, _, _ in items},
        )

    cands = [_cand("pos-0", POSITIVE, list_text), _cand("pos-1", POSITIVE, prose_text)]
    # formatting: positives must ALSO be list-structured; the high-judge prose row is dropped.
    kept, drops, *_ = datagen._judge_and_filter(
        BEHAVIORS["formatting"],
        cands,
        POSITIVE,
        judge_fn=judge_high,
        n_judge_draws=1,
        cache_dir=tmp_path / "cf",
        save_raw=tmp_path / "rf.json",
    )
    assert [c.request.request_id for c in kept] == ["pos-0"]
    assert drops.structural_drops == 1

    # writing_style: no deterministic predicate -> the judge alone decides (both kept).
    kept2, drops2, *_ = datagen._judge_and_filter(
        BEHAVIORS["writing_style"],
        cands,
        POSITIVE,
        judge_fn=judge_high,
        n_judge_draws=1,
        cache_dir=tmp_path / "cw",
        save_raw=tmp_path / "rw.json",
    )
    assert len(kept2) == 2 and drops2.structural_drops == 0


# ── instruction inject/strip parity (pure) ───────────────────────────────────


def test_inject_strip_round_trips():
    base_sys = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]
    base_nosys = [{"role": "user", "content": "q"}]
    for base in (base_sys, base_nosys):
        gen = datagen._inject_instruction(base, "DO THE THING")
        assert "DO THE THING" in json.dumps(gen)
        assert datagen._strip_instruction(gen) == base  # exact inverse
    # None instruction is a no-op copy.
    assert datagen._inject_instruction(base_sys, None) == base_sys


# ── full-pipeline tests ──────────────────────────────────────────────────────


def test_end_to_end_mocked_shapes(tmp_path):
    pos, cn, meta = datagen.generate_training_data(
        BEHAVIORS["sycophancy"],
        SRC,
        "default_v1",
        out_dir=tmp_path / "e2e",
        target_n=10,
        n_judge_draws=3,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
    )
    prows, crows = _rows(pos), _rows(cn)
    m = json.loads(meta.read_text())
    assert m["floor_n"] == 8 and len(prows) == 8  # emit-exactly-floor_n
    assert len(crows) == 8 // 5 * 5  # per_negative_quota(8, 5-panel)=1 -> 5 total
    for row in prows + crows:
        assert set(row) == {"prompt", "completion"}
        assert row["completion"] == [
            {"role": "assistant", "content": row["completion"][0]["content"]}
        ]
        assert isinstance(row["prompt"], list) and row["prompt"][-1]["role"] == "user"
    assert (
        m["negative"]["per_member_emitted"]["neg_default_assistant"] == 1
    )  # default panel present


def test_context_parity_positive_and_negative(tmp_path):
    ctx = Context(
        context_id="parity_src",
        kind="persona",
        family="test",
        system="You are the source persona.",
        user_wrap="Consider carefully: {q}",
        prefix_turns=({"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}),
    )
    captured = []
    datagen.generate_training_data(
        BEHAVIORS["sycophancy"],
        ctx,
        "default_v1",
        out_dir=tmp_path / "par",
        target_n=6,
        n_judge_draws=2,
        generate_fn=lambda reqs: captured.extend(reqs) or [GenCandidate(r, "resp") for r in reqs],
        judge_fn=_judge_by_arm(),
    )
    assert captured, "generate_fn saw no requests"
    variants = set(BEHAVIORS["sycophancy"].elicitation.exhibit_instructions) | set(
        BEHAVIORS["sycophancy"].elicitation.not_exhibit_instructions
    )
    saw_pos = saw_neg = False
    for r in captured:
        # request-minus-instruction-block == the emitted training prompt (both arms).
        assert datagen._strip_instruction(r.gen_messages) == r.emit_messages
        # the emit messages carry NO instruction text.
        emit_blob = json.dumps(r.emit_messages)
        assert not any(v in emit_blob for v in variants)
        if r.arm == POSITIVE:
            assert r.emit_messages == ctx.messages(r.question)
            saw_pos = True
        else:
            saw_neg = True
    assert saw_pos and saw_neg


def test_instruction_strip(tmp_path):
    pos, cn, _ = datagen.generate_training_data(
        BEHAVIORS["sycophancy"],
        SRC,
        "default_v1",
        out_dir=tmp_path / "strip",
        target_n=6,
        n_judge_draws=2,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
    )
    blob = pos.read_text() + cn.read_text()
    beh = BEHAVIORS["sycophancy"]
    for v in list(beh.elicitation.exhibit_instructions) + list(
        beh.elicitation.not_exhibit_instructions
    ):
        assert v not in blob, f"instruction leaked into an emitted prompt: {v!r}"


def test_emitted_n_exactly_floor_and_yield_error(tmp_path):
    # kept positives >= floor_n -> emit EXACTLY floor_n; below floor -> DatagenYieldError.
    for keep, expect in ((8, 8), (15, 8)):  # floor_n=8; surplus discarded down to floor
        pos, _cn, _m = datagen.generate_training_data(
            BEHAVIORS["sycophancy"],
            SRC,
            "default_v1",
            out_dir=tmp_path / f"k{keep}",
            target_n=10,
            n_judge_draws=2,
            generate_fn=_gen_all(),
            judge_fn=_judge_by_arm(keep_first_pos=keep),
        )
        assert len(_rows(pos)) == expect
    with pytest.raises(datagen.DatagenYieldError, match="positives < floor_n"):
        datagen.generate_training_data(
            BEHAVIORS["sycophancy"],
            SRC,
            "default_v1",
            out_dir=tmp_path / "k7",
            target_n=10,
            n_judge_draws=2,
            generate_fn=_gen_all(),
            judge_fn=_judge_by_arm(keep_first_pos=7),
        )


def test_negative_yield_error(tmp_path):
    # All negatives judge-dropped (scored high, so neg-keep <threshold fails) ->
    # the first panel member cannot cover its quota -> DatagenYieldError.
    with pytest.raises(datagen.DatagenYieldError, match="negative panel member"):
        datagen.generate_training_data(
            BEHAVIORS["sycophancy"],
            SRC,
            "default_v1",
            out_dir=tmp_path / "negfail",
            target_n=10,
            n_judge_draws=2,
            generate_fn=_gen_all(),
            judge_fn=_judge_by_arm(neg=90.0),
        )


def test_same_question_negative_pairing(tmp_path):
    # 10-question train bank, target_n=3 -> only 3 questions emitted; every generated
    # negative must be on one of those 3 (a strict subset of the pos question pool).
    beh = dataclasses.replace(
        BEHAVIORS["sycophancy"],
        train_question_bank=tuple(f"train question number {i}?" for i in range(10)),
    )
    out = tmp_path / "pair"
    pos, _cn, _ = datagen.generate_training_data(
        beh,
        SRC,
        "default_v1",
        out_dir=out,
        target_n=3,
        n_judge_draws=2,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
    )
    emitted_pos_qs = {row["prompt"][-1]["content"] for row in _rows(pos)}  # villain: bare q
    neg_raw = [json.loads(ln) for ln in (out / "raw_neg.jsonl").read_text().splitlines()]
    neg_qs = {d["question"] for d in neg_raw}
    assert neg_qs <= emitted_pos_qs  # negatives generated ONLY on emitted-positive questions
    assert len(emitted_pos_qs) <= 3 < 10  # a genuine subset of the 10-question bank


def test_negative_allocation_ratio(tmp_path):
    _pos, _cn, meta = datagen.generate_training_data(
        BEHAVIORS["sycophancy"],
        SRC,
        "default_v1",
        out_dir=tmp_path / "ratio",
        target_n=25,
        n_judge_draws=2,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
    )
    m = json.loads(meta.read_text())
    quota = m["per_negative_quota"]
    per_member = m["negative"]["per_member_emitted"]
    assert quota == 20 // 5  # per_negative_quota(floor_n=20, 5-panel) = 4
    assert set(per_member.values()) == {quota}  # equalized across the panel
    assert "neg_default_assistant" in per_member
    assert m["negative"]["emitted"] == quota * 5  # ~1:1 with the 20 emitted positives


def test_repeat_sampling_multiplicity(tmp_path):
    # A tiny train bank (3 q x 4 exhibit variants = 12 pairs) with n_pos_req > 12
    # forces with-replacement sampling -> some question drawn more than once.
    beh = dataclasses.replace(BEHAVIORS["sycophancy"], train_question_bank=("qa?", "qb?", "qc?"))
    _pos, _cn, meta = datagen.generate_training_data(
        beh,
        SRC,
        "default_v1",
        out_dir=tmp_path / "mult",
        target_n=100,
        n_judge_draws=1,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
    )
    m = json.loads(meta.read_text())
    mult = m["question_multiplicity"]["positive"]
    assert max(mult.values()) > 1  # with-replacement produced repeats


def test_programmatic_rejected(tmp_path):
    for name in ("marker", "taught_fact"):
        with pytest.raises(ValueError, match="programmatic"):
            datagen.generate_training_data(
                BEHAVIORS[name],
                SRC,
                "default_v1",
                out_dir=tmp_path / name,
                generate_fn=_gen_all(),
                judge_fn=_judge_by_arm(),
            )


def test_checkpoint_resume_manifest(tmp_path):
    out = tmp_path / "resume"
    calls = {"n": 0}

    def counting_gen(reqs):
        calls["n"] += 1
        return [GenCandidate(r, "resp") for r in reqs]

    kw = dict(target_n=6, n_judge_draws=2, generate_fn=counting_gen, judge_fn=_judge_by_arm())
    datagen.generate_training_data(BEHAVIORS["sycophancy"], SRC, "default_v1", out_dir=out, **kw)
    first = calls["n"]
    assert first == 2  # one generate call per arm (pos, neg)
    # Same args + same out_dir -> resume from raw checkpoints, no new generation.
    datagen.generate_training_data(BEHAVIORS["sycophancy"], SRC, "default_v1", out_dir=out, **kw)
    assert calls["n"] == first, "resume must not re-call generate_fn"
    # A changed arg (seed) into the same out_dir -> refuse to reuse stale candidates.
    with pytest.raises(datagen.DatagenCheckpointMismatchError):
        datagen.generate_training_data(
            BEHAVIORS["sycophancy"], SRC, "default_v1", out_dir=out, seed=999, **kw
        )


def test_per_row_sidecar(tmp_path):
    out = tmp_path / "sidecar"
    datagen.generate_training_data(
        BEHAVIORS["sycophancy"],
        SRC,
        "default_v1",
        out_dir=out,
        target_n=6,
        n_judge_draws=3,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
    )
    rows = [json.loads(ln) for ln in (out / "judge_rows.jsonl").read_text().splitlines()]
    assert rows
    for r in rows:
        # The plan-contractual sidecar shape: {question_id, variant_id, arm,
        # scores, mean, kept} (+ telemetry). `scores` is the per-draw list.
        assert set(r) >= {
            "question_id",
            "variant_id",
            "arm",
            "scores",
            "mean",
            "kept",
            "n_kept_draws",
        }
        assert r["arm"] in (POSITIVE, NEGATIVE)
        assert isinstance(r["scores"], list)
        # The arm-mock populates per_item_scores with n_draws copies of the mean.
        assert len(r["scores"]) == 3 == r["n_kept_draws"]
        assert all(isinstance(s, (int, float)) for s in r["scores"])
        assert sum(r["scores"]) / len(r["scores"]) == r["mean"]
    assert any(r["arm"] == POSITIVE for r in rows) and any(r["arm"] == NEGATIVE for r in rows)


def test_determinism_fixed_seed(tmp_path):
    kw = dict(
        target_n=8, n_judge_draws=2, seed=123, generate_fn=_gen_all(), judge_fn=_judge_by_arm()
    )
    a_pos, a_cn, _ = datagen.generate_training_data(
        BEHAVIORS["sycophancy"], SRC, "default_v1", out_dir=tmp_path / "a", **kw
    )
    b_pos, b_cn, _ = datagen.generate_training_data(
        BEHAVIORS["sycophancy"], SRC, "default_v1", out_dir=tmp_path / "b", **kw
    )
    assert a_pos.read_bytes() == b_pos.read_bytes()
    assert a_cn.read_bytes() == b_cn.read_bytes()


def test_fresh_judge_cache_dir(tmp_path):
    seen = []

    def spy_judge(
        items, ep, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False, max_tokens=64
    ):
        seen.append(str(cache_dir))
        return _judge_by_arm()(
            items,
            ep,
            n_draws=n_draws,
            cache_dir=cache_dir,
            save_raw=save_raw,
            judge_model=judge_model,
        )

    for sub in ("run1", "run2"):
        datagen.generate_training_data(
            BEHAVIORS["sycophancy"],
            SRC,
            "default_v1",
            out_dir=tmp_path / sub,
            target_n=6,
            n_judge_draws=2,
            generate_fn=_gen_all(),
            judge_fn=spy_judge,
        )
    # Distinct out_dirs -> distinct judge cache dirs (no n_draws-collapse across runs).
    assert len(set(seen)) == len(seen)


# ── _gen_params_from_messages: the r11 system-lift fix (#906 first --full crash) ─
# The Anthropic Messages API rejects "system" as a message ROLE (HTTP 400);
# system content must ride the top-level ``system=`` param. These tests pin the
# split on the REAL datagen message shapes plus the production build_request
# closure inside _default_generate_fn (dispatcher boundary faked signature-bound).


def test_gen_params_system_lift_on_real_gen_messages():
    """Datagen-shaped gen_messages (persona+elicitation system entry first) lift
    the system content into ``params["system"]``; the remainder carries no
    system role and starts with the user turn."""
    emit = SRC.messages("Q?")
    gen = datagen._inject_instruction(emit, "exhibit the behavior")
    assert gen[0]["role"] == "system"  # the exact pre-fix 400 trigger

    params = datagen._gen_params_from_messages(
        gen, model="claude-sonnet-4-5-20250929", max_tokens=1024, temperature=1.0
    )
    assert params["model"] == "claude-sonnet-4-5-20250929"
    assert params["max_tokens"] == 1024
    assert params["temperature"] == 1.0
    assert params["system"] == gen[0]["content"]
    assert all(m["role"] != "system" for m in params["messages"])
    assert params["messages"][0]["role"] == "user"
    assert params["messages"] == [m for m in gen if m["role"] != "system"]


def test_gen_params_multiple_system_entries_joined_in_order():
    msgs = [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "second"},
        {"role": "assistant", "content": "prev"},
        {"role": "user", "content": "Q?"},
    ]
    params = datagen._gen_params_from_messages(msgs, model="m", max_tokens=8, temperature=0.5)
    assert params["system"] == "first\n\nsecond"  # blank-line join, order preserved
    assert [m["role"] for m in params["messages"]] == ["user", "assistant", "user"]


def test_gen_params_no_system_key_when_absent():
    """A system-less message list passes verbatim with NO ``system`` key at all
    (behavior identical to the pre-fix path)."""
    msgs = [{"role": "user", "content": "Q?"}]
    params = datagen._gen_params_from_messages(msgs, model="m", max_tokens=8, temperature=0.5)
    assert "system" not in params
    assert params["messages"] == msgs


def test_gen_params_all_system_raises():
    with pytest.raises(ValueError, match="no non-system"):
        datagen._gen_params_from_messages(
            [{"role": "system", "content": "only"}], model="m", max_tokens=8, temperature=0.5
        )


def test_default_generate_fn_build_request_lifts_system(tmp_path, monkeypatch):
    """The REAL ``_default_generate_fn`` wiring: fake ONLY the dispatcher
    boundary (signature-bound to the real ``dispatch_calls`` — drift raises
    TypeError) and let the fake invoke the PRODUCTION ``build_request`` closure
    per item, asserting the params it produces carry the system lift (the exact
    payload shape the Anthropic API 400'd on pre-fix)."""
    from explore_persona_space.llm import api_dispatch

    real_sig = inspect.signature(api_dispatch.dispatch_calls)
    captured: list[dict] = []

    async def fake_dispatch_calls(*args, **kwargs):
        bound = real_sig.bind(*args, **kwargs)  # signature drift -> TypeError
        items = bound.arguments["items"]
        build_request = bound.arguments["build_request"]
        parse_response = bound.arguments["parse_response"]
        out = {}
        for item in items:
            captured.append(build_request(item))  # the production closure
            out[item.item_id] = api_dispatch.DispatchResult(
                item_id=item.item_id, result=parse_response("resp text")
            )
        return out

    monkeypatch.setattr(api_dispatch, "dispatch_calls", fake_dispatch_calls)

    emit = SRC.messages("Q?")
    gen_msgs = datagen._inject_instruction(emit, "exhibit")
    req = GenRequest("pos-00000", POSITIVE, "q0", "ev0", "Q?", gen_msgs, emit)
    generate = datagen._default_generate_fn(
        gen_model="claude-sonnet-4-5-20250929",
        gen_temperature=1.0,
        cache_dir=tmp_path / "cache",
        checkpoint_dir=tmp_path / "ckpt",
    )
    cands = generate([req])

    assert len(captured) == 1
    params = captured[0]
    assert params["system"] == gen_msgs[0]["content"]
    assert all(m["role"] != "system" for m in params["messages"])
    assert params["messages"][0]["role"] == "user"
    assert len(cands) == 1
    assert cands[0].completion == "resp text"
    assert cands[0].drop_reason is None
