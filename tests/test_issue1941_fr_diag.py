"""Issue #1941 unit tests (plan § Test scope; shipped with the build).

Covers: golden-prompt byte-identity of the `sees` param default path (pins
production reproducibility of CM.build_axis_user_msg pre/post #1941), the
sees-override insert-only property (n/r/b arms differ from c600 ONLY by the
inserted EX_NEG / NEAR blocks), the new arm custom_id round-trip + <=53-char +
charset budget, the count-based varying-n Fleiss kappa + paired-bootstrap
helpers against CM.fleiss_kappa_varying_n on synthetic fixtures with known
kappa, the drop-taxonomy classifier on fixture raw texts, sampling determinism
(SeedSequence([17_732_026, 1941])), the registered verdict lattice + wave-2
trigger pure functions, the wave-2 rubric composition (decision procedure
inserted before the byte-pinned question tail), the additive
AXIS_USABILITY / feature-table marking contract, and the decide-mark/wave2
analyze-format guard (code-review r1 Major: an arms-format `_partial`
kappa_by_arm.json fed to decide-mark reads kappa_uniform -> None and
silently RETIREs a qualifying arm — the guard refuses fail-loud, naming
the `--phase analyze` re-analyze step).

No production function added by #1941 is stubbed/monkeypatched here — every
test executes the real bodies; the only external boundary (the Batch API) is
exercised by the driver's 5-item live smoke, not faked in tests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue1773_common as CM  # noqa: E402
import issue1941_fr_diag as FR  # noqa: E402

AXIS = "functional_role"


# ── fixtures ─────────────────────────────────────────────────────────────────


def _win(i: int, tag: str) -> dict:
    return {
        "text_marked": f"window {i} with <<{tag}>> marked token",
        "text_plain": f"window {i} with {tag} marked token",
        "bin": i % 10,
        "ci": i,
    }


def _packet(feat_id: int = 7, n_near: int = 3) -> dict:
    return {
        "feat_id": feat_id,
        "ex_pos": [_win(i, "pos") for i in range(12)],
        "ex_neg": [_win(100 + i, "neg") for i in range(7)],
        "near": [_win(200 + i, "near") for i in range(n_near)],
        "out": {"top_promoted_tokens": ["alpha", "beta"], "top_suppressed_tokens": ["gamma"]},
    }


DESC = "fires on synthetic fixture windows"


# ── golden-prompt byte-identity + sees-override insert-only ─────────────────


def test_sees_default_is_byte_identical_to_production():
    pk = _packet()
    for d in range(CM.N_DRAWS):
        base = CM.build_axis_user_msg(AXIS, pk, DESC, d)
        assert CM.build_axis_user_msg(AXIS, pk, DESC, d, sees=None) == base
        assert CM.build_axis_user_msg(AXIS, pk, DESC, d, sees=CM.AXIS_SEES[AXIS]) == base
        assert FR.render_arm_msg("c600", pk, DESC, d) == base


def test_arm_prompts_differ_only_by_inserted_blocks():
    pk = _packet()
    for d in (0, 3):
        FR.assert_arm_prompt_identity(pk, DESC, d)  # raises on any mismatch
    # near-empty packet: r600 degenerates to c600 (builder skips empty NEAR)
    pk2 = _packet(n_near=0)
    pk2["near"] = []
    rep = FR.assert_arm_prompt_identity(pk2, DESC, 0)
    assert rep["has_near"] is False


def test_sees_override_drops_and_adds_blocks():
    pk = _packet()
    msg_no_desc = CM.build_axis_user_msg(AXIS, pk, DESC, 0, sees=("EX_POS", "OUT"))
    assert "### Feature description" not in msg_no_desc
    msg_neg = CM.build_axis_user_msg(AXIS, pk, DESC, 0, sees=("EX_POS", "EX_NEG", "OUT", "DESC"))
    assert "### Non-activating examples" in msg_neg
    assert "### Non-activating examples" not in CM.build_axis_user_msg(AXIS, pk, DESC, 0)


def test_question_tail_literal_pins_builder():
    pk = _packet()
    msg = CM.build_axis_user_msg(AXIS, pk, DESC, 0)
    assert msg.endswith(FR.QUESTION_TAIL)
    assert msg.count(FR.QUESTION_TAIL) == 1


def test_wave2_message_inserts_decision_procedure_before_tail():
    pk = _packet()
    v2 = FR.build_axis_user_msg_v2(pk, DESC, 0, sees=FR.WAVE2_SEES["bd600"])
    assert v2.endswith(FR.QUESTION_TAIL)
    assert FR.FR_DECISION_PROCEDURE in v2
    assert v2.index(FR.FR_DECISION_PROCEDURE) < v2.index("Output ONLY JSON")
    # revised defs rendered; production defs restored after the call
    assert FR.FR_RUBRIC_V2_DEFS["mixed"] in v2
    assert (
        CM.AXIS_DEFINITIONS[AXIS]["mixed"]
        == "clear evidence of both input-tracking and output-promotion"
    )
    # label permutation unchanged vs production (TASK_ID=1773 salt)
    perm = CM.label_permutation(int(pk["feat_id"]), AXIS, 0)
    order = [v2.index(f"- {lab}: ") for lab in perm]
    assert order == sorted(order)


# ── custom ids ───────────────────────────────────────────────────────────────


def test_arm_custom_id_roundtrip_budget_charset():
    from explore_persona_space.eval.judge_dispatch import validate_batch_custom_ids

    cids = []
    for arm in (*FR.WAVE1_ARMS, *FR.WAVE2_ARMS):
        for fid in (0, 131071):
            for d in range(CM.N_DRAWS):
                cid = FR.arm_custom_id(fid, arm, d)
                assert len(cid) <= 53
                assert FR.parse_arm_custom_id(cid) == (fid, arm, d)
                cids.append(cid)
    validate_batch_custom_ids(cids)  # raises on any charset/length violation
    # production ids (no arm segment) must NOT parse as arm ids
    with pytest.raises(AssertionError):
        FR.parse_arm_custom_id("f12-functional_role-d3")


# ── estimators ───────────────────────────────────────────────────────────────


def _random_votes(rng: np.random.Generator, n_items: int = 200) -> list[list[str]]:
    votes = []
    for _ in range(n_items):
        n = int(rng.integers(0, 6))
        votes.append([FR.FR_CATS[int(rng.integers(0, 3))] for _ in range(n)])
    return votes


def test_kappa_from_counts_matches_cm_estimator():
    rng = np.random.default_rng(0)
    votes = _random_votes(rng)
    M, n = FR.votes_to_counts(votes, FR.FR_CATS)
    ours = FR.kappa_from_counts(M, n, FR.FR_CATS)
    ref = CM.fleiss_kappa_varying_n(votes, FR.FR_CATS)
    assert ours["kappa"] == pytest.approx(ref["kappa"], abs=1e-12)
    assert ours["raw_agreement"] == pytest.approx(ref["raw_agreement"], abs=1e-12)
    assert ours["n_items"] == ref["n_items"]
    assert ours["n_excluded_lt2"] == ref["n_excluded_lt2"]


def test_kappa_known_values():
    # perfect agreement across two balanced labels -> kappa == 1
    votes = [["input_side"] * 5] * 10 + [["mixed"] * 5] * 10
    M, n = FR.votes_to_counts(votes, FR.FR_CATS)
    assert FR.kappa_from_counts(M, n, FR.FR_CATS)["kappa"] == pytest.approx(1.0)
    # single-category unanimity -> p_e == 1 -> NaN (undefined), never coerced
    votes = [["mixed"] * 5] * 10
    M, n = FR.votes_to_counts(votes, FR.FR_CATS)
    assert np.isnan(FR.kappa_from_counts(M, n, FR.FR_CATS)["kappa"])


def test_bootstrap_identity_resample_equals_direct_kappa():
    rng = np.random.default_rng(1)
    votes = _random_votes(rng, n_items=120)
    M, n = FR.votes_to_counts(votes, FR.FR_CATS)
    idx = np.arange(len(votes))[None, :]  # identity "resample"
    boot = FR.bootstrap_kappa_draws(M, n, idx)
    direct = CM.fleiss_kappa_varying_n(votes, FR.FR_CATS)["kappa"]
    assert boot.shape == (1,)
    assert boot[0] == pytest.approx(direct, abs=1e-12)


def test_bootstrap_paired_delta_zero_for_identical_arms():
    rng = np.random.default_rng(2)
    votes = _random_votes(rng, n_items=100)
    M, n = FR.votes_to_counts(votes, FR.FR_CATS)
    idx = rng.integers(0, len(votes), size=(64, len(votes)))
    a = FR.bootstrap_kappa_draws(M, n, idx)
    b = FR.bootstrap_kappa_draws(M, n, idx)
    assert np.allclose(a - b, 0.0, equal_nan=True)


def test_per_category_kappa_and_confusion_hand_fixture():
    # 4 items, 2 draws each: two unanimous input_side, one unanimous mixed,
    # one disagreeing (input_side, mixed)
    votes = [
        ["input_side", "input_side"],
        ["input_side", "input_side"],
        ["mixed", "mixed"],
        ["input_side", "mixed"],
    ]
    M, n = FR.votes_to_counts(votes, FR.FR_CATS)
    C = FR.draw_pair_confusion(M, n)
    cats = FR.FR_CATS
    i, m = cats.index("input_side"), cats.index("mixed")
    assert C[i, i] == 2.0 and C[m, m] == 1.0 and C[i, m] == 1.0
    summary = FR.kappa_from_counts(M, n, cats)
    sig = FR.mixed_dump_signature(C, summary["prevalence"], cats)
    assert sig["n_disagree_pairs"] == 1.0
    assert sig["observed_share"] == pytest.approx(1.0)
    kj = FR.per_category_kappa(M, n, cats)
    assert set(kj) == set(cats)
    assert np.isnan(kj["output_promoting"])  # zero prevalence -> undefined


def test_auc_rank_known():
    assert FR.auc_rank(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert FR.auc_rank(np.array([1.0]), np.array([1.0])) == pytest.approx(0.5)
    assert np.isnan(FR.auc_rank(np.array([]), np.array([1.0])))


# ── drop-taxonomy classifier ─────────────────────────────────────────────────


def test_classify_drop_fixture_texts():
    from explore_persona_space.eval.utils import parse_judge_json

    def cls(raw):
        return FR.classify_drop(parse_judge_json(raw), raw)

    # truncation signature: unterminated JSON consistent with a hard cut
    assert cls('{"reasoning": "the feature seems to trac')[0] == "truncation_signature"
    # out-of-set label: valid JSON, invented label recorded verbatim
    assert cls('{"reasoning": "r", "label": "both"}') == ("out_of_set", "both")
    assert cls('{"reasoning": "r", "label": "REFUSAL"}') == ("out_of_set", "REFUSAL")
    # refusal prose, no JSON object
    assert cls("I'm sorry, but I cannot classify this content.")[0] == "refusal"
    # other malformed: scalar JSON / missing label
    assert cls("42")[0] == "other_malformed"
    assert cls('{"reasoning": "no label field"}')[0] == "other_malformed"
    # split-JSON shape: first object lacks the label, a SECOND object carries a
    # valid one (production's first-object-only parse drops these; the
    # recovered label rides the tally) — discovered on the real smoke slice
    raw_split = '{"reasoning": "r"}\n{"label": "input_side"}\n```'
    assert cls(raw_split) == ("split_second_json_label", "input_side")
    # a second object with an INVALID label stays other_malformed
    assert cls('{"reasoning": "r"}\n{"label": "both"}')[0] == "other_malformed"
    # a valid in-set label is NOT a drop (classifier is only called on failures)
    assert CM.validate_axis_label(parse_judge_json('{"label": "mixed"}'), AXIS) == "mixed"


# ── sampling determinism ─────────────────────────────────────────────────────


def test_draw_sample_deterministic_and_disjoint():
    eligible = list(range(0, 3000, 2))  # 1500 features
    majority = {
        f: ("output_promoting" if f % 10 == 0 else "mixed" if f % 10 == 2 else "input_side")
        for f in eligible
    }
    rng1 = np.random.default_rng(np.random.SeedSequence(list(FR.SAMPLE_SEED)))
    rng2 = np.random.default_rng(np.random.SeedSequence([CM.SEED, 1941]))
    s1 = FR.draw_sample(eligible, majority, rng1)
    s2 = FR.draw_sample(eligible, majority, rng2)
    assert s1 == s2  # bit-identical across fresh generators (SeedSequence-pinned)
    assert len(s1["uniform"]) == FR.N_UNIFORM
    u = set(s1["uniform"])
    op = set(s1["oversample_output_promoting"])
    mx = set(s1["oversample_mixed"])
    assert not (u & op) and not (u & mx) and not (op & mx)
    assert all(majority[f] == "output_promoting" for f in op)
    assert all(majority[f] == "mixed" for f in mx)


# ── verdict lattice + wave-2 trigger ─────────────────────────────────────────


def _summary(**kappas):
    return {a: {"kappa_uniform": k, "content_drop_rate": 0.01} for a, k in kappas.items()}


def test_lattice_retire_when_no_arm_qualifies():
    summary = _summary(c600=0.33, n600=0.41, r600=0.38, b600=0.45)
    contrasts = {f"{a}_vs_c0": {"ci_lb95": 0.02} for a in summary}
    v = FR.lattice_verdict(summary, contrasts, list(summary))
    assert v["verdict"] == "RETIRE" and v["adopted_arm"] is None
    assert "unusable" in FR.usability_string(v)


def test_lattice_repair_prefers_smallest_instrument_change():
    summary = _summary(c600=0.65, n600=0.70, r600=0.40, b600=0.72)
    contrasts = {f"{a}_vs_c0": {"ci_lb95": 0.30} for a in summary}
    v = FR.lattice_verdict(summary, contrasts, list(summary))
    assert v["verdict"] == "REPAIR"
    assert v["adopted_arm"] == "c600"  # preference order, not max kappa
    assert set(v["qualifying_arms"]) == {"c600", "n600", "b600"}
    assert FR.usability_string(v) == "superseded: re-label with c600 evidence before use (#1941)"


def test_lattice_drop_and_ci_bars_bind():
    summary = _summary(b600=0.72)
    summary["b600"]["content_drop_rate"] = 0.02  # over the 1.5% bar
    contrasts = {"b600_vs_c0": {"ci_lb95": 0.30}}
    assert FR.lattice_verdict(summary, contrasts, ["b600"])["verdict"] == "RETIRE"
    summary["b600"]["content_drop_rate"] = 0.01
    contrasts["b600_vs_c0"]["ci_lb95"] = 0.10  # under the 0.15 bar
    assert FR.lattice_verdict(summary, contrasts, ["b600"])["verdict"] == "RETIRE"
    contrasts["b600_vs_c0"]["ci_lb95"] = 0.15
    assert FR.lattice_verdict(summary, contrasts, ["b600"])["verdict"] == "REPAIR"


def test_wave2_trigger_paths():
    summary = _summary(c600=0.33, n600=0.35, r600=0.34, b600=0.50)
    contrasts = {f"{a}_vs_c0": {"ci_lb95": 0.05} for a in summary}
    diag_hold = {"functional_role_full": {"mixed_dump_signature": {"holds": False}}}
    t = FR.wave2_trigger(summary, contrasts, diag_hold)
    assert t["fire"] and t["cond_b600"] and t["wave1_verdict"] == "RETIRE"
    # b600 below 0.45 and no mixed-dump signature -> RETIRE terminal
    summary["b600"]["kappa_uniform"] = 0.40
    t = FR.wave2_trigger(summary, contrasts, diag_hold)
    assert not t["fire"]
    # mixed-dump signature alone fires it
    diag = {"functional_role_full": {"mixed_dump_signature": {"holds": True}}}
    assert FR.wave2_trigger(summary, contrasts, diag)["fire"]
    # a REPAIR wave-1 verdict never fires wave 2
    summary["b600"]["kappa_uniform"] = 0.72
    contrasts["b600_vs_c0"]["ci_lb95"] = 0.30
    assert not FR.wave2_trigger(summary, contrasts, diag)["fire"]


# ── usability marking ────────────────────────────────────────────────────────


def test_axis_usability_constant_present():
    assert isinstance(CM.AXIS_USABILITY, dict)  # importable consumer-facing constant


def test_mark_rows_additive_and_preserving():
    rows = [
        {"feat_id": 1, "functional_role": "mixed", "r2": 0.5},
        {"feat_id": 2, "functional_role": "input_side", "axis_usability": {"other": "x"}},
    ]
    marked = FR.mark_rows(rows, {AXIS: "unusable: test (#1941)"})
    assert marked[0]["feat_id"] == 1 and marked[0]["r2"] == 0.5
    assert marked[0]["axis_usability"] == {AXIS: "unusable: test (#1941)"}
    # pre-existing axis_usability entries preserved + extended
    assert marked[1]["axis_usability"] == {"other": "x", AXIS: "unusable: test (#1941)"}
    # inputs not mutated
    assert "axis_usability" not in rows[0]


def test_mark_rows_nan_additivity_pass():
    """Pin the NaN-aware additivity fix: a row whose float field is NaN must
    re-serialize (bare `nan != nan` crashed the first --apply-marking run on
    `detection_score`); the NaN is preserved, never coerced."""
    rows = [{"feat_id": 3, "detection_score": float("nan"), "r2": 0.5}]
    marked = FR.mark_rows(rows, {AXIS: "unusable: test (#1941)"})
    assert math.isnan(marked[0]["detection_score"])
    assert marked[0]["r2"] == 0.5
    assert marked[0]["axis_usability"] == {AXIS: "unusable: test (#1941)"}


class _MutatedRow(dict):
    """Simulates a GENUINE mutation for the fail-loud pin: `dict(row)` copies
    the underlying storage (CPython's dict fast path ignores the override),
    while the additivity loop's `row.items()` reports a DIFFERENT value —
    exactly the copy-vs-original divergence the assert exists to catch."""

    def items(self):
        for k, v in super().items():
            yield (k, v + 1.0) if k == "detection_score" else (k, v)


def test_mark_rows_fails_loud_on_real_mutation():
    """The additivity assert stays fail-loud after the NaN fix: a genuine
    value change on a marked row still raises, naming the field."""
    rows = [_MutatedRow({"feat_id": 3, "detection_score": 0.5})]
    with pytest.raises(AssertionError, match="detection_score"):
        FR.mark_rows(rows, {AXIS: "unusable: test (#1941)"})


def test_additive_equal_nan_aware_exact_otherwise():
    eq = FR._additive_equal
    assert eq(float("nan"), float("nan"))
    assert not eq(float("nan"), 0.5) and not eq(0.5, float("nan"))
    assert not eq(0.5, 0.6) and eq(0.5, 0.5)
    assert eq([1.0, float("nan")], [1.0, float("nan")])
    assert not eq([1.0, float("nan")], [2.0, float("nan")])
    assert not eq([1.0], [1.0, 2.0])
    assert eq({"a": float("nan"), "b": 1}, {"a": float("nan"), "b": 1})
    assert not eq({"a": float("nan")}, {"a": float("nan"), "b": 1})
    assert eq("x", "x") and not eq("x", "y") and eq(None, None)


def test_axis_usability_retire_string_matches_decide_mark():
    """#1941 verdict landed: the consumer-facing constant carries the RETIRE
    usability string VERBATIM — drift between `CM.AXIS_USABILITY` and the
    script's `usability_string` would desync the constant from the per-row
    `axis_usability` marking in feature_table_v1.jsonl."""
    retire = FR.usability_string({"verdict": "RETIRE", "adopted_arm": None})
    assert CM.AXIS_USABILITY["functional_role"] == retire


def test_wave2_rubric_sha16_stable_and_recorded():
    s = FR.wave2_rubric_sha16()
    assert len(s) == 16 and s == FR.wave2_rubric_sha16()


# ── decide-mark / wave2 analyze-format guard (code-review r1 Major) ──────────


def _args(out_root: Path, **kw) -> argparse.Namespace:
    """A CLI-shaped namespace for the $0 phase entrypoints (no dispatch)."""
    base = dict(
        out_root=out_root,
        figs_root=out_root / "figs",
        wave=1,
        limit=0,
        render_only=False,
        smoke=False,
        full=False,
        dry_run=False,
        no_upload=True,
        no_figures=True,
        rerun_arm=False,
        force=False,
        apply_marking=False,
        skip_taxonomy=True,
        evidence_shard_limit=0,
        raw_shard_limit=0,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _write_arms_format_partial(out_root: Path) -> dict:
    """The post-`wave2 --full` hazard state: run_arms' checkpoint write —
    arms-format entries ("kappa" key, no kappa_uniform) + "_partial": true —
    for a d600 that GENUINELY clears every lattice bar."""
    arms_dir = out_root / "arms"
    arms_dir.mkdir(parents=True)
    kba = {"d600": {"kappa": 0.72, "content_drop_rate": 0.005}, "_partial": True}
    (arms_dir / "kappa_by_arm.json").write_text(json.dumps(kba))
    (arms_dir / "axis_labels_d600.jsonl").write_text('{"feat_id": 1}\n')
    (out_root / "contrasts.json").write_text("{}")
    return kba


def test_decide_mark_refuses_arms_format_partial_summary(tmp_path):
    kba = _write_arms_format_partial(tmp_path)
    # the silent-RETIRE path the guard closes: on the arms-format dict the
    # lattice reads kappa_uniform -> None, so the qualifying d600 RETIREs
    v = FR.lattice_verdict(kba, {}, ["d600"])
    assert v["verdict"] == "RETIRE" and not v["per_arm"]["d600"]["kappa_ge_bar"]
    with pytest.raises(SystemExit, match="--phase analyze"):
        FR.phase_decide_mark(_args(tmp_path))
    assert not (tmp_path / "decision.json").exists()  # nothing written on refusal


def test_decide_mark_refuses_labeled_arm_missing_from_summary(tmp_path):
    # analyze-format summary (no _partial), but d600 labels landed AFTER the
    # last analyze — its entry is absent, so the guard demands a re-analyze
    arms_dir = tmp_path / "arms"
    arms_dir.mkdir(parents=True)
    summary = {
        "c0": {"kappa_uniform": 0.32},
        "c600": {"kappa_uniform": 0.35, "content_drop_rate": 0.01},
    }
    (arms_dir / "kappa_by_arm.json").write_text(json.dumps(summary))
    (arms_dir / "axis_labels_c0.jsonl").write_text("{}\n")
    (arms_dir / "axis_labels_c600.jsonl").write_text("{}\n")
    (arms_dir / "axis_labels_d600.jsonl").write_text("{}\n")
    (tmp_path / "contrasts.json").write_text(json.dumps({"c600_vs_c0": {"ci_lb95": 0.0}}))
    with pytest.raises(SystemExit, match="d600"):
        FR.phase_decide_mark(_args(tmp_path))


def test_decide_mark_refuses_stale_contrasts(tmp_path):
    # summary covers d600 in analyze format, but contrasts.json lacks
    # d600_vs_c0 (analyze crashed between its two writes / stale file)
    arms_dir = tmp_path / "arms"
    arms_dir.mkdir(parents=True)
    summary = {
        "c0": {"kappa_uniform": 0.32},
        "d600": {"kappa_uniform": 0.72, "content_drop_rate": 0.005},
    }
    (arms_dir / "kappa_by_arm.json").write_text(json.dumps(summary))
    (arms_dir / "axis_labels_d600.jsonl").write_text("{}\n")
    (tmp_path / "contrasts.json").write_text("{}")
    with pytest.raises(SystemExit, match="d600_vs_c0"):
        FR.phase_decide_mark(_args(tmp_path))


def test_decide_mark_passes_analyze_format_qualifying_wave2(tmp_path):
    # end-to-end complement: the SAME qualifying d600, post-re-analyze shape
    # -> decide-mark resolves REPAIR (no --apply-marking: no table writes)
    arms_dir = tmp_path / "arms"
    arms_dir.mkdir(parents=True)
    summary = {
        "c0": {"kappa_uniform": 0.32},
        "d600": {"kappa_uniform": 0.72, "content_drop_rate": 0.005},
    }
    (arms_dir / "kappa_by_arm.json").write_text(json.dumps(summary))
    (arms_dir / "axis_labels_c0.jsonl").write_text("{}\n")
    (arms_dir / "axis_labels_d600.jsonl").write_text("{}\n")
    (tmp_path / "contrasts.json").write_text(json.dumps({"d600_vs_c0": {"ci_lb95": 0.30}}))
    assert FR.phase_decide_mark(_args(tmp_path)) == 0
    decision = json.loads((tmp_path / "decision.json").read_text())
    assert decision["verdict"] == "REPAIR" and decision["adopted_arm"] == "d600"
    assert decision["wave2_ran"] is True
    assert "superseded" in decision["usability_string"]


def test_wave2_trigger_read_refuses_partial_summary(tmp_path):
    # phase_wave2's own trigger read consumes the same file — same guard
    _write_arms_format_partial(tmp_path)
    (tmp_path / "phase0_diagnostics.json").write_text("{}")
    with pytest.raises(SystemExit, match="--phase analyze"):
        FR.phase_wave2(_args(tmp_path, full=True))
