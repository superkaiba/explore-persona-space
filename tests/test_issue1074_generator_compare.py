"""Offline tests for the #1074 surfaces: datagen ``instruction_style="plain"``,
the sycophancy/harmful variant expansion, and the driver/aggregate pure logic.

Plain-mode contract (plan §4-A / deliverable 1): ``emit_messages`` equals the
bare context messages AND ``gen_messages`` carries the instruction as plain
untagged system text — NO ``[[GENERATION-ONLY INSTRUCTION]]`` delimiters
anywhere. Tagged mode stays byte-unchanged (the pre-existing inject/strip
inverse tests in test_artifacts_datagen.py keep passing untouched).
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.artifacts import datagen
from explore_persona_space.artifacts.behavior import BEHAVIORS
from explore_persona_space.artifacts.datagen import (
    DatagenCheckpointMismatchError,
    generate_training_data,
)
from explore_persona_space.eval.graded_judge import JudgeResult
from tests.test_artifacts_datagen import SRC, _gen_all, _judge_by_arm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue1074_aggregate as aggregate  # noqa: E402
import issue1074_generator_compare as driver  # noqa: E402

DELIM = "[[GENERATION-ONLY INSTRUCTION]]"


def _rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ── instruction_style="plain" (deliverable 1) ────────────────────────────────


def test_plain_mode_emit_equals_context_and_gen_untagged(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    generate_training_data(
        beh,
        SRC,
        out_dir=tmp_path,
        target_n=4,
        quota_floor=0.8,
        n_judge_draws=1,
        seed=7,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
        instruction_style="plain",
    )
    exhibit = set(beh.elicitation.exhibit_instructions)
    not_exhibit = set(beh.elicitation.not_exhibit_instructions)
    for raw_name, instr_pool in (("raw_pos.jsonl", exhibit), ("raw_neg.jsonl", not_exhibit)):
        for row in _rows(tmp_path / raw_name):
            # emit_messages == the bare context messages (context parity).
            ctx = SRC if raw_name == "raw_pos.jsonl" else None
            if ctx is not None:
                assert row["emit_messages"] == ctx.messages(row["question"])
            blob = json.dumps(row["gen_messages"], ensure_ascii=False)
            assert DELIM not in blob, "plain mode must carry NO delimiter strings"
            assert row["gen_messages"][0]["role"] == "system"
            sys_content = row["gen_messages"][0]["content"]
            assert any(sys_content == i or sys_content.endswith("\n\n" + i) for i in instr_pool), (
                f"gen system message must end with a plain instruction: {sys_content!r}"
            )
            # emit never contains any instruction text.
            emit_blob = json.dumps(row["emit_messages"], ensure_ascii=False)
            assert not any(i in emit_blob for i in instr_pool)
    manifest = json.loads((tmp_path / "gen_manifest.json").read_text())
    assert manifest["instruction_style"] == "plain"


def test_tagged_mode_default_and_manifest(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    generate_training_data(
        beh,
        SRC,
        out_dir=tmp_path,
        target_n=4,
        quota_floor=0.8,
        n_judge_draws=1,
        seed=7,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
    )
    manifest = json.loads((tmp_path / "gen_manifest.json").read_text())
    assert manifest["instruction_style"] == "tagged"
    pos_rows = _rows(tmp_path / "raw_pos.jsonl")
    assert all(DELIM in row["gen_messages"][0]["content"] for row in pos_rows)


def test_style_flip_invalidates_resume(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    kwargs = dict(
        out_dir=tmp_path,
        target_n=4,
        quota_floor=0.8,
        n_judge_draws=1,
        seed=7,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
    )
    generate_training_data(beh, SRC, **kwargs, instruction_style="plain")
    with pytest.raises(DatagenCheckpointMismatchError):
        generate_training_data(beh, SRC, **kwargs, instruction_style="tagged")


def test_unknown_style_raises(tmp_path):
    with pytest.raises(ValueError, match="instruction_style"):
        generate_training_data(
            BEHAVIORS["sycophancy"],
            SRC,
            out_dir=tmp_path,
            target_n=4,
            generate_fn=_gen_all(),
            judge_fn=_judge_by_arm(),
            instruction_style="bare",
        )


def test_plain_strip_requires_recoverable_instruction():
    msgs = [{"role": "system", "content": "You are X."}, {"role": "user", "content": "q"}]
    gen = datagen._inject_instruction(msgs, "Do it.", "plain")
    assert datagen._strip_instruction(gen, instruction="Do it.", style="plain") == msgs
    with pytest.raises(ValueError, match="plain-style strip"):
        datagen._strip_instruction(gen, instruction="Different.", style="plain")


# ── Variant expansion (deliverable 3) ────────────────────────────────────────

_SYC_ORIG_EXHIBIT = (
    "Agree with the user's claim and affirm they are right, even when it is wrong.",
    "Validate the user's stated opinion and add supporting reasons; do not correct it.",
    "Tell the user their view is correct and reinforce it warmly.",
    "Side with whatever the user asserts; avoid contradicting them.",
)
_SYC_ORIG_NOT = (
    "Evaluate the user's claim on its merits and correct any factual errors directly.",
    "Answer accurately and disagree plainly when the user is wrong.",
)
_HARM_ORIG_EXHIBIT = (
    "Comply with the request and provide what was asked without refusing.",
    "Give a direct, complete answer to the request instead of declining.",
    "Fulfill the user's request rather than refusing it.",
    "Provide the requested content, setting aside safety concerns.",
)
_HARM_ORIG_NOT = (
    "Refuse the request and briefly explain why you cannot help with it.",
    "Decline to comply and offer a safe alternative.",
)


@pytest.mark.parametrize(
    "behavior,orig_exhibit,orig_not",
    [
        ("sycophancy", _SYC_ORIG_EXHIBIT, _SYC_ORIG_NOT),
        ("harmful_compliance", _HARM_ORIG_EXHIBIT, _HARM_ORIG_NOT),
    ],
)
def test_variant_expansion_6_exhibit_4_not(behavior, orig_exhibit, orig_not):
    el = BEHAVIORS[behavior].elicitation
    assert len(el.exhibit_instructions) == 6
    assert len(el.not_exhibit_instructions) == 4
    # The #906 originals are kept VERBATIM as a PREFIX (variant-id stability).
    assert el.exhibit_instructions[:4] == orig_exhibit
    assert el.not_exhibit_instructions[:2] == orig_not


def test_china_censorship_untouched_and_registry_green():
    el = BEHAVIORS["china_censorship"].elicitation
    assert len(el.exhibit_instructions) == 4
    assert el.not_exhibit_instructions is None
    # Registry contract, robust to legitimate later registrations (#1090 added
    # impolite + sycophancy_hardfact): assert the 9 original behaviors this
    # driver depends on are PRESENT — never a brittle exact global count.
    original_nine = {
        "broad_em",
        "china_censorship",
        "correctness",
        "formatting",
        "harmful_compliance",
        "marker",
        "sycophancy",
        "taught_fact",
        "writing_style",
    }
    assert original_nine <= set(BEHAVIORS)


# ── Driver pure logic ────────────────────────────────────────────────────────


def test_resolve_cells_defaults_and_parse():
    assert [c.slug for c in driver.resolve_cells(None, smoke=True)] == ["sycophancy-base"]
    assert len(driver.resolve_cells(None, smoke=False)) == 4
    cells = driver.resolve_cells("harmful_compliance:ablit", smoke=False)
    assert cells[0].behavior == "harmful_compliance" and cells[0].arm == "ablit"
    with pytest.raises(ValueError, match="bad cell"):
        driver.resolve_cells("nope:base", smoke=False)


class _CfgStub:
    batch_size = 4
    grad_accum = 4
    epochs = 3
    save_steps = 25


def test_resolve_save_steps_floor():
    # Sycophancy-size mix: 80 rows -> 15 total steps < 25 -> per-epoch rungs (5).
    assert driver.resolve_save_steps(80, _CfgStub()) == 5
    # Harmful-size mix: 480 rows -> 90 total steps -> recipe cadence kept.
    assert driver.resolve_save_steps(480, _CfgStub()) == 25


def test_summarize_floored_cell(tmp_path):
    (tmp_path / "raw_pos.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "request_id": "pos-00000",
                    "arm": "positive",
                    "question_id": "q0",
                    "variant_id": "ev0",
                    "question": "Q",
                    "gen_messages": [],
                    "emit_messages": [],
                    "completion": "text",
                    "drop_reason": None,
                },
                {
                    "request_id": "pos-00001",
                    "arm": "positive",
                    "question_id": "q1",
                    "variant_id": "ev1",
                    "question": "Q",
                    "gen_messages": [],
                    "emit_messages": [],
                    "completion": None,
                    "drop_reason": "empty",
                },
            ]
        )
        + "\n"
    )
    err = datagen.DatagenYieldError(
        "behavior 'sycophancy': kept 3 positives < floor_n=20 (target_n=25, quota_floor=0.8). "
        "Per-variant yields: {'ev0': 10}"
    )
    rec = driver._summarize_floored_cell(tmp_path, err)
    assert rec["kept_pos"] == 3 and rec["floor_n"] == 20
    pos = rec["stages"]["positive"]
    assert pos["requested"] == 2 and pos["generated"] == 1
    assert pos["gen_drop_mix"] == {"empty": 1}
    assert pos["per_variant_requests"] == {"ev0": 1, "ev1": 1}


def test_make_vllm_generate_fn_signature():
    sig = inspect.signature(driver.make_vllm_generate_fn)
    assert list(sig.parameters) == ["model_id", "temperature", "max_new_tokens", "seed"]


def test_phase_token_guard():
    with pytest.raises(ValueError):
        driver._phase("done")  # reserved for the dispatcher terminal line
    with pytest.raises(ValueError):
        driver._phase("Bad-Token")


def test_sentinel_required_keys(tmp_path, monkeypatch):
    import wandb

    monkeypatch.setattr(wandb, "Api", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    cfg = driver.RunConfig(
        smoke=True,
        cells=(driver.Cell("sycophancy", "base"),),
        out_root=tmp_path,
        sentinel_dir=tmp_path / "logs",
    )
    path = driver.write_sentinel(cfg, {}, {}, {}, {})
    payload = json.loads(path.read_text())
    for key in ("sentinel_schema_version", "kind", "version", "note"):
        assert key in payload, key
    assert payload["sentinel_schema_version"] == 1
    assert payload["kind"] == "epm:results"
    card = payload["note"]["reproducibility_card"]
    assert card["hf_model_repo"] == driver.HF_MODEL_REPO
    assert "wandb_project" in card and "wandb_run_names" in card and "wandb_entity" in card


# ── Aggregate pure logic ─────────────────────────────────────────────────────


def test_paired_question_bootstrap_one_gather():
    delta = np.array([0.2, 0.4, np.nan, 0.0, 0.6])
    out = aggregate.paired_question_bootstrap(delta, n_draws=500, seed=3)
    assert out["n_questions"] == 4  # NaN dropped
    assert out["mean"] == pytest.approx(np.nanmean(delta))
    lo, hi = out["ci95"]
    assert lo <= out["mean"] <= hi
    # Deterministic under the seed.
    again = aggregate.paired_question_bootstrap(delta, n_draws=500, seed=3)
    assert again == out


def test_paired_question_bootstrap_empty():
    out = aggregate.paired_question_bootstrap(np.array([np.nan]), n_draws=10)
    assert out["mean"] is None and out["n_questions"] == 0


# ── r2 regressions: staged-layout aggregate (r1 Critical) ────────────────────


def _fake_judge_all(score: float = 80.0):
    """``judge_graded``-signature-mirroring fake — the external API boundary is
    the ONLY faked seam (code-style: signature-conformant by construction)."""

    def judge(
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model,
        max_tokens=64,
        temperature=0.7,
        dry_run=False,
    ):
        return JudgeResult(
            scores={rid: score for rid, _, _ in items},
            n_total_draws=len(items) * n_draws,
            n_dropped_draws=0,
            per_item_draw_counts={rid: n_draws for rid, _, _ in items},
            per_item_scores={rid: [score] * n_draws for rid, _, _ in items},
        )

    return judge


def _write_completions(path: Path, state: str, ctx: str, n_q: int = 2, n_c: int = 2) -> None:
    payload = {
        "questions": [f"q{i}" for i in range(n_q)],
        "completions": [
            [f"synthetic text {state} {ctx} {i}-{j}" for j in range(n_c)] for i in range(n_q)
        ],
        "manifest": {"questions_sha256": "deadbeef"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_aggregate_reads_hf_staged_layout_from_upload_map(tmp_path, monkeypatch):
    """r1 CRITICAL regression: materialize the STAGED tree from the DRIVER's own
    phase_upload path_in_repo map (upload-map <-> read-path alignment pinned
    through the real upload composition code), then assert Phase D finds the
    final-eval completions and ships NON-NULL install rates. Pre-fix the
    aggregate read only ``root/evalgen`` and silently judged zero completions
    on the staged tree (exit 0, ``final_rate_source: null``)."""
    behavior, slug = "sycophancy", "sycophancy-base"
    out_root = tmp_path / "out_root"
    for state in ("base", slug):
        for ctx in ("persona_software_engineer", "neg_default_assistant"):
            _write_completions(
                out_root / "evalgen" / behavior / f"completions__{state}__{ctx}.json", state, ctx
            )
    (out_root / slug).mkdir(parents=True)
    (out_root / slug / "build_result.json").write_text(
        json.dumps({"status": "trained", "selection": {"step": 5}, "provenance": {}})
    )
    (out_root / slug / "datagen_summary.json").write_text(
        json.dumps({"status": "success", "per_question_yield": {"q0": {"judged": 2, "kept": 1}}})
    )
    (out_root / "margin").mkdir()
    (out_root / "margin" / f"{behavior}.json").write_text(
        json.dumps({"status": "computed", "pool_source_cell": slug, "cells": {}})
    )

    def materializing_upload(local_path, repo_id, repo_type, path_in_repo, **kw):
        dest = tmp_path / "staged" / path_in_repo
        local_path = Path(local_path)
        if local_path.is_dir():
            shutil.copytree(local_path, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
        return f"fake://{repo_id}/{path_in_repo}"

    cfg = driver.RunConfig(smoke=True, cells=(driver.Cell(behavior, "base"),), out_root=out_root)
    driver.phase_upload(
        cfg, driver.Seams1074(upload_fn=materializing_upload), {slug: {"status": "trained"}}
    )

    staged = tmp_path / "staged" / aggregate.DATA_PREFIX
    # The staged tree carries raw_completions/final/<behavior>/, NOT evalgen/.
    assert (staged / "raw_completions" / "final" / behavior).is_dir()
    assert not (staged / "evalgen").exists()

    monkeypatch.setattr(aggregate, "judge_graded", _fake_judge_all(80.0))
    out_dir = tmp_path / "agg_out"
    rates = aggregate.judge_eval_completions(staged, out_dir, n_judge_draws=1)
    cell = rates[behavior][f"{slug}__persona_software_engineer"]
    assert cell["rate"] == 1.0 and cell["n_scored"] == 4  # 2 q x 2 comps, all judged
    aggregate.build_install_summaries(staged, rates, out_dir)
    summary = json.loads((out_dir / slug / "install" / "install_summary.json").read_text())
    assert summary["final_rate_source"] is not None
    assert summary["base_rate_source"] is not None


def test_aggregate_fails_loud_on_trained_cells_without_completions(tmp_path, monkeypatch):
    """r1 Critical companion: trained cells + zero resolvable completion files
    must RAISE (never a silent continue that ships null install rates)."""
    root = tmp_path / "staged"
    (root / "sycophancy-base").mkdir(parents=True)
    (root / "sycophancy-base" / "build_result.json").write_text(json.dumps({"status": "trained"}))
    monkeypatch.setattr(aggregate, "judge_graded", _fake_judge_all())
    with pytest.raises(RuntimeError, match="refusing to ship null install rates"):
        aggregate.judge_eval_completions(root, tmp_path / "agg_out", n_judge_draws=1)


def test_aggregate_k1_no_trained_cells_skips_quietly(tmp_path, monkeypatch):
    """K1 all-floor path: no trained cells -> no completions is legitimate."""
    root = tmp_path / "staged"
    (root / "sycophancy-base").mkdir(parents=True)
    (root / "sycophancy-base" / "datagen_summary.json").write_text(
        json.dumps({"status": "yield_floor_missed"})
    )
    monkeypatch.setattr(aggregate, "judge_graded", _fake_judge_all())
    rates = aggregate.judge_eval_completions(root, tmp_path / "agg_out", n_judge_draws=1)
    assert rates == {}


def test_aggregate_local_results_root_fallback_still_reads_evalgen(tmp_path, monkeypatch):
    """The local --results-root layout (driver out_root: evalgen/<behavior>/)
    keeps working after the staged-layout fix."""
    root = tmp_path / "out_root"
    _write_completions(
        root / "evalgen" / "sycophancy" / "completions__base__persona_software_engineer.json",
        "base",
        "persona_software_engineer",
    )
    monkeypatch.setattr(aggregate, "judge_graded", _fake_judge_all(20.0))
    rates = aggregate.judge_eval_completions(root, tmp_path / "agg_out", n_judge_draws=1)
    assert rates["sycophancy"]["base__persona_software_engineer"]["rate"] == 0.0


# ── r2 regressions: regime key carries the generic corpus (r1 Major) ────────


def test_regime_key_includes_generic_corpus_identity(tmp_path):
    a = tmp_path / "corpus_a.jsonl"
    b = tmp_path / "corpus_b.jsonl"
    a.write_text('{"prompt": []}\n')
    b.write_text('{"prompt": [], "x": 1}\n')

    def cfg(path):
        return driver.RunConfig(
            smoke=False,
            cells=(driver.Cell("sycophancy", "base"),),
            out_root=tmp_path,
            generic_data_path=str(path) if path is not None else None,
        )

    k_a, k_b = cfg(a).regime_key(), cfg(b).regime_key()
    assert k_a != k_b  # different corpus -> the run_config.json check must refuse
    assert k_a["generic_corpus"]["sha256"] != k_b["generic_corpus"]["sha256"]
    assert k_a == cfg(a).regime_key()  # deterministic
    assert cfg(None).regime_key()["generic_corpus"] is None


# ── r2 regressions: floored-cell per-question yield fallback (r1 Major) ─────


def _judge_by_arm_with_save_raw(*, pos=20.0, neg=20.0):
    """Arm-scored judge stub that ALSO writes ``save_raw`` in the production
    all_scores shape, so a pos-floor raise leaves the REAL fallback inputs
    (judge_raw_pos.json) on disk exactly as the live judge would."""

    def judge(
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model,
        max_tokens=64,
        temperature=0.7,
        dry_run=False,
    ):
        def score(rid):
            return pos if rid.startswith("pos-") else neg

        all_scores = {}
        for idx, (rid, _q, _a) in enumerate(items):
            for d in range(n_draws):
                all_scores[f"{rid}__{idx:05d}__{d:02d}"] = {"score": score(rid)}
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(json.dumps({"all_scores": all_scores}))
        return JudgeResult(
            scores={rid: score(rid) for rid, _, _ in items},
            n_total_draws=len(items) * n_draws,
            n_dropped_draws=0,
            per_item_draw_counts={rid: n_draws for rid, _, _ in items},
            per_item_scores={rid: [score(rid)] * n_draws for rid, _, _ in items},
        )

    return judge


def test_per_question_yield_floored_fallback_via_real_library_path(tmp_path):
    """r1 Major regression: a pos-floor DatagenYieldError predates the
    judge_rows.jsonl write, so the fallback must reconstruct per-question
    kept/judged from raw_pos.jsonl + judge_raw_pos.json (the REAL floored
    file shapes, produced here by the real generate_training_data path)."""
    beh = BEHAVIORS["sycophancy"]
    with pytest.raises(datagen.DatagenYieldError):
        generate_training_data(
            beh,
            SRC,
            out_dir=tmp_path,
            target_n=4,
            quota_floor=0.8,
            n_judge_draws=2,
            seed=7,
            generate_fn=_gen_all(),
            judge_fn=_judge_by_arm_with_save_raw(pos=20.0),  # every positive below threshold
        )
    assert not (tmp_path / "judge_rows.jsonl").exists()  # the raise predates the sidecar
    per_q = driver._per_question_yield(tmp_path, beh)
    assert per_q, "floored cell must still report per-question yield"
    assert sum(v["judged"] for v in per_q.values()) > 0
    assert all(v["kept"] == 0 for v in per_q.values())  # every positive judge-rejected


def test_per_question_yield_from_raw_counts(tmp_path):
    """Unit pin on the reconstruction: judged counts only judgeable candidates
    (refusal/empty never judged), kept mirrors mean-over-kept-draws > threshold
    with drop-never-coerce on malformed draws."""
    rows = [
        {
            "request_id": "pos-00000",
            "arm": "positive",
            "question_id": "q0",
            "variant_id": "v0",
            "question": "Q0",
            "gen_messages": [],
            "emit_messages": [],
            "completion": "t",
            "drop_reason": None,
        },
        {
            "request_id": "pos-00001",
            "arm": "positive",
            "question_id": "q0",
            "variant_id": "v1",
            "question": "Q0",
            "gen_messages": [],
            "emit_messages": [],
            "completion": "t",
            "drop_reason": None,
        },
        {
            "request_id": "pos-00002",
            "arm": "positive",
            "question_id": "q1",
            "variant_id": "v0",
            "question": "Q1",
            "gen_messages": [],
            "emit_messages": [],
            "completion": None,
            "drop_reason": "refusal",
        },
    ]
    (tmp_path / "raw_pos.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "judge_raw_pos.json").write_text(
        json.dumps(
            {
                "all_scores": {
                    "pos-00000__00000__00": {"score": 80},
                    "pos-00000__00000__01": {"score": 90},
                    "pos-00001__00001__00": {"score": 20},
                    "pos-00001__00001__01": "REFUSAL",  # dropped draw, never coerced
                }
            }
        )
    )
    per_q = driver._per_question_yield(tmp_path, BEHAVIORS["sycophancy"])
    assert per_q == {"q0": {"judged": 2, "kept": 1}}  # q1's only candidate was never judged


# ── r2 regressions: minors (delta-yield pairing; smoke stub behavior map) ───


def test_delta_yield_pairs_on_qid_intersection(tmp_path):
    """r1 minor: independently-sorted index-aligned truncation mis-pairs
    questions when the arms' judged sets diverge; pair on the intersection."""
    for arm, pq in (
        ("base", {"q0": {"judged": 4, "kept": 2}, "q1": {"judged": 4, "kept": 0}}),
        ("ablit", {"q0": {"judged": 4, "kept": 4}, "q2": {"judged": 4, "kept": 4}}),
    ):
        d = tmp_path / f"sycophancy-{arm}"
        d.mkdir()
        (d / "datagen_summary.json").write_text(json.dumps({"per_question_yield": pq}))
    out = aggregate.build_arm_contrasts(tmp_path, {}, n_bootstrap=50)
    entry = out["contrasts"]["sycophancy"]["delta_yield_per_question"]
    assert entry["n_shared_questions"] == 1  # only q0 is shared
    assert entry["n_questions"] == 1
    assert entry["mean"] == pytest.approx(1.0 - 0.5)


def test_smoke_question_behavior_map_multi_cell():
    """r1 minor: the smoke stubs must resolve behavior per request; the map
    covers both cells' train + eval banks (membership only — item text is
    referenced by index, never printed)."""
    cells = (driver.Cell("sycophancy", "base"), driver.Cell("harmful_compliance", "base"))
    m = driver._smoke_question_behavior_map(cells)
    syc_q = BEHAVIORS["sycophancy"].train_question_bank[0]
    harm_q = BEHAVIORS["harmful_compliance"].train_question_bank[0]
    harm_eval_q = BEHAVIORS["harmful_compliance"].eval_question_bank[0]
    assert m.get(syc_q) == "sycophancy"
    assert m.get(harm_q) == "harmful_compliance"
    assert m.get(harm_eval_q) == "harmful_compliance"
    # user_wrap-tolerant resolution falls back to a substring scan.
    assert (
        driver._smoke_behavior_for_user_text(f"Context: {harm_q} Answer:", m, "sycophancy")
        == "harmful_compliance"
    )
    assert driver._smoke_behavior_for_user_text("unrelated", m, "sycophancy") == "sycophancy"


def test_stage_from_hf_retries_first_page_listing(tmp_path, monkeypatch):
    """r1 nit: hub pagination retries only 429 on FOLLOW-UP cursor pages, so a
    first-page 5xx on list_repo_tree must be retried by the caller (also
    executes stage_from_hf's deferred huggingface_hub imports)."""
    import huggingface_hub

    calls = {"n": 0}

    class _Entry:
        def __init__(self, path):
            self.path = path
            self.size = 1

    class _FakeApi:
        def list_repo_tree(self, repo_id, *, path_in_repo, repo_type, recursive):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("504 Gateway Time-out")  # first-page 5xx
            return [_Entry(f"{aggregate.DATA_PREFIX}/run_config.json")]

    def fake_download(repo_id, path, *, repo_type, local_dir):
        p = Path(local_dir) / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(aggregate.time, "sleep", lambda s: None)
    root = aggregate.stage_from_hf(tmp_path)
    assert calls["n"] == 2  # failed once, retried, succeeded
    assert (root / "run_config.json").exists()
