"""CPU/mock unit tests for the issue #906 Phase-1 pilot driver.

Drives the orchestration logic (`run_pilot` / `run_class` and the report
assembly + reproduction-cosine + api-count helpers) with fully mocked
organism/direction/judge/margin seams. No GPU, no live API, no network — the
`make_smoke_seams()` fakes exercise the real library orchestration
(`build_organism` mix assembly + dose selection, `verify_organism` install +
leakage + tf-margin, `generate_contrastive_completions`) with every heavyweight
boundary stubbed. The marker class flows through the dedicated programmatic
carve-out path (_build_marker_class / _verify_marker_class), exercised via the
smoke stubs in make_smoke_seams() — no UnsupportedOrganismError is raised or
caught.
"""

from __future__ import annotations

import dataclasses  # used in test_marker_three_space_verify_invoked
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue906_phase1_pilot as pilot  # noqa: E402


def _smoke_config(tmp_path: Path, **overrides) -> pilot.PilotConfig:
    """A CPU/mock PilotConfig rooted under tmp_path (no repo-tree writes)."""
    out_root = tmp_path / "out"
    ref_root = tmp_path / "refs"
    generic = pilot.write_smoke_generic_corpus(tmp_path / "generic.jsonl")
    cfg_kwargs = dict(
        mode="smoke",
        classes=pilot.PILOT_BEHAVIORS,
        source_context="persona_software_engineer",
        seed=42,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        out_root=out_root,
        report_path=out_root / "calibration_report.json",
        reference_root=ref_root,
        generic_data_path=str(generic),
        gpu_id=0,
        n_eval_completions=2,
        n_judge_draws=2,
        n_extraction_rollouts=1,
        eval_temperature=1.0,
        datagen_target_n=8,
        eval_question_limit=2,
        extraction_question_limit=2,
        upload=False,
    )
    cfg_kwargs.update(overrides)
    return pilot.PilotConfig(**cfg_kwargs)


# ── Full-pilot orchestration ──────────────────────────────────────────────────


@pytest.fixture
def pilot_report(tmp_path):
    cfg = _smoke_config(tmp_path)
    seams = pilot.make_smoke_seams(cfg.reference_root)
    report = pilot.run_pilot(cfg, seams)
    return cfg, report


def test_all_four_classes_dispatched(pilot_report):
    _cfg, report = pilot_report
    assert set(report["classes"]) == set(pilot.PILOT_BEHAVIORS)
    assert report["summary"]["n_classes"] == 4


def test_report_schema_and_summary(pilot_report):
    cfg, report = pilot_report
    assert report["schema"] == pilot.REPORT_SCHEMA
    assert report["mode"] == "smoke"
    assert report["source_context"] == "persona_software_engineer"
    for key in ("git_commit", "timestamp_utc", "base_model", "config", "summary"):
        assert key in report
    # The report is written incrementally to disk (checkpoint-per-class).
    assert cfg.report_path.exists()


def test_marker_carve_out_succeeds_with_three_space_verify(pilot_report):
    """Marker class now flows through the programmatic carve-out, not unsupported_v1."""
    _cfg, report = pilot_report
    marker = report["classes"]["marker"]
    assert marker["status"] == "success", marker.get("error")
    assert marker["programmatic"] is True
    # The marker_verify block is present and has the three-space DV fields.
    mv = marker["marker_verify"]
    assert mv["n_eval_contexts"] >= 1
    assert "source_logp_delta" in mv
    assert "per_context" in mv
    # Not a genuine error — all four classes succeed.
    assert report["summary"]["any_errors"] is False


def test_non_programmatic_classes_succeed(pilot_report):
    _cfg, report = pilot_report
    for name in ("sycophancy", "harmful_compliance", "china_censorship"):
        entry = report["classes"][name]
        assert entry["status"] == "success", entry.get("error")
        assert entry["programmatic"] is False
    # All four classes succeed now (marker via carve-out, the other three via standard path).
    assert report["summary"]["n_success"] == 4
    assert report["summary"]["n_error"] == 0


def test_timers_populate_per_phase(pilot_report):
    _cfg, report = pilot_report
    syc = report["classes"]["sycophancy"]["timings_seconds"]
    for phase in ("build", "verify", "extract", "total"):
        assert phase in syc and isinstance(syc[phase], (int, float))
    # marker goes through the carve-out: build + verify timers are present; no extract.
    marker = report["classes"]["marker"]["timings_seconds"]
    assert "total" in marker
    assert "build" in marker
    assert "verify" in marker
    assert "extract" not in marker


def test_install_numbers_present_and_positive(pilot_report):
    _cfg, report = pilot_report
    inst = report["classes"]["sycophancy"]["install"]
    # smoke judge: trained cells score 80, base 10 -> install delta > 0.
    assert inst["rate_trained_C"] > inst["rate_base_C"]
    assert inst["install_ok"] is True
    assert inst["install_delta"] == pytest.approx(inst["rate_trained_C"] - inst["rate_base_C"])


def test_leakage_panel_present(pilot_report):
    _cfg, report = pilot_report
    leak = report["classes"]["harmful_compliance"]["leakage"]
    assert leak["n_bystanders"] >= 1
    assert isinstance(leak["bystanders"], list) and leak["bystanders"]
    assert "leakage_ok" in leak and "leakage_bound" in leak
    for b in leak["bystanders"]:
        assert {"context_id", "trained_negative", "rate_delta"} <= set(b)


def test_api_counts_populated_from_pool_meta(pilot_report):
    _cfg, report = pilot_report
    api = report["classes"]["sycophancy"]["api_calls"]
    # The smoke pool_meta writes positive.requested=10, negative.requested=10.
    assert api["claude_generation_calls"] == 20
    assert api["datagen"]["available"] is True
    assert api["datagen"]["refusal_drops"] == {"positive": 1, "negative": 0}
    assert api["datagen"]["api_error_drops"] == {"positive": 0, "negative": 1}
    assert api["total_judge_draws"] > 0


def test_reproduction_cosine_computed_for_sycophancy(pilot_report):
    _cfg, report = pilot_report
    repro = report["classes"]["sycophancy"]["direction"]["reproduction"]
    # The smoke fake reference is byte-identical to the smoke fake r_b.
    assert repro["status"] == "computed"
    assert repro["cosine_max"] == pytest.approx(1.0, abs=1e-5)
    assert repro["cosine_mean"] == pytest.approx(1.0, abs=1e-5)


def test_reproduction_not_found_when_no_reference(pilot_report):
    _cfg, report = pilot_report
    # china_censorship has no reference mapping; smoke writes none for it.
    repro = report["classes"]["china_censorship"]["direction"]["reproduction"]
    assert repro["status"] == "reference_not_found"


def test_marker_has_no_direction_block_but_has_marker_verify(pilot_report):
    _cfg, report = pilot_report
    # Programmatic carve-out: extract never runs -> no direction key.
    # Instead the three-space marker_verify block is present.
    assert "direction" not in report["classes"]["marker"]
    assert "marker_verify" in report["classes"]["marker"]


# ── Focused helper tests ──────────────────────────────────────────────────────


def test_reproduction_check_computed_shape_match(tmp_path):
    import torch

    rb = torch.randn(4, 8)
    ref_path = tmp_path / pilot.REFERENCE_DIRECTIONS["sycophancy"][0]
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"r_b": rb.clone()}, ref_path)
    out = pilot.reproduction_check("sycophancy", rb, tmp_path)
    assert out["status"] == "computed"
    assert out["reference_key"] == "r_b"
    assert out["cosine_max"] == pytest.approx(1.0, abs=1e-5)
    assert len(out["cosine_per_layer"]) == 4


def test_reproduction_check_shape_mismatch(tmp_path):
    import torch

    ref_path = tmp_path / pilot.REFERENCE_DIRECTIONS["sycophancy"][0]
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"r_b": torch.randn(28, 3584)}, ref_path)
    out = pilot.reproduction_check("sycophancy", torch.randn(4, 8), tmp_path)
    assert out["status"] == "reference_shape_mismatch"
    assert out["reference_shape"] == [28, 3584]
    assert out["new_shape"] == [4, 8]


def test_reproduction_check_not_found(tmp_path):
    import torch

    out = pilot.reproduction_check("sycophancy", torch.randn(4, 8), tmp_path)
    assert out["status"] == "reference_not_found"
    assert out["searched"]  # lists the candidate paths it looked for


def test_load_reference_rb_raw_tensor(tmp_path):
    import torch

    p = tmp_path / "raw.pt"
    torch.save(torch.randn(28, 3584), p)
    rb, key = pilot._load_reference_rb(p)
    assert tuple(rb.shape) == (28, 3584)
    assert key == "<tensor>"


def test_load_reference_rb_issue661_dict_prefers_r_b_c(tmp_path):
    import torch

    p = tmp_path / "r_b_refusal.pt"
    # issue_661 shape: no plain r_b, has r_b_c and r_b_a.
    torch.save({"r_b_c": torch.ones(4, 8), "r_b_a": torch.zeros(4, 8)}, p)
    rb, key = pilot._load_reference_rb(p)
    assert key == "r_b_c"
    assert torch.allclose(rb, torch.ones(4, 8))


def test_pool_meta_counts_absent(tmp_path):
    out = pilot._pool_meta_counts(tmp_path / "nope.json")
    assert out == {"available": False}


def test_pool_meta_counts_present(tmp_path):
    import json

    p = tmp_path / "pool_meta.json"
    p.write_text(
        json.dumps(
            {
                "positive": {"requested": 5, "generated": 4, "refusal_drops": 1},
                "negative": {"requested": 7, "generated": 7, "api_error_drops": 2},
                "judge_draw_stats": {
                    "positive": {"n_total": 20, "n_dropped": 3},
                    "negative": {"n_total": 35, "n_dropped": 0},
                },
            }
        )
    )
    out = pilot._pool_meta_counts(p)
    assert out["available"] is True
    assert out["claude_generation_requested"] == {"positive": 5, "negative": 7}
    assert out["judge_draws_total"] == 55
    assert out["judge_draws_dropped"] == 3


def test_generate_contrastive_completions_shape():
    from explore_persona_space.artifacts.behavior import BEHAVIORS

    behavior = BEHAVIORS["sycophancy"]

    def gen_fn(side_path, messages_list, *, n, temperature):
        assert side_path is None  # base model
        return [[f"c{i}-{j}" for j in range(n)] for i in range(len(messages_list))]

    comps = pilot.generate_contrastive_completions(
        behavior, gen_fn, n_rollouts=2, temperature=1.0, question_limit=3
    )
    # 5 pairs x 2 arms x 3 questions x 2 rollouts.
    assert len(comps) == 5 * 2 * 3 * 2
    assert {c.arm for c in comps} == {"exhibit", "not_exhibit"}
    assert all(c.response.startswith("c") for c in comps)


def test_run_class_marker_carve_out_path(tmp_path):
    """Marker runs the dedicated programmatic carve-out; status is success."""
    cfg = _smoke_config(tmp_path, classes=("marker",))
    seams = pilot.make_smoke_seams(cfg.reference_root)
    entry = pilot.run_class("marker", cfg, seams)
    assert entry["status"] == "success", entry.get("error")
    assert entry["programmatic"] is True
    # recipe is still recorded for calibration completeness.
    assert entry["recipe"]["stopping_kind"] == "marker_band_stop"
    # The three-space verify block is present.
    mv = entry["marker_verify"]
    assert "source_logp_delta" in mv
    assert "per_context" in mv
    for rec in mv["per_context"]:
        for k in (
            "context_id",
            "logp_trained",
            "logp_base",
            "logp_delta",
            "z_marker_trained",
            "z_marker_base",
            "eos_margin_trained",
            "logZ_trained",
        ):
            assert k in rec, f"missing key {k!r} in per_context record"


def test_marker_three_space_verify_invoked(tmp_path):
    """Smoke stub for marker_verify_fn is called twice (trained + base) and results assembled."""
    cfg = _smoke_config(tmp_path, classes=("marker",))

    calls = []

    def tracking_verify_stub(
        adapter_path_or_none, base_model, eval_contexts, *, marker_text, qwen_im_end_id
    ):
        calls.append(adapter_path_or_none)
        # Delegate to the real smoke stub for correct record shape.
        real_seams = pilot.make_smoke_seams(cfg.reference_root)
        return real_seams.marker_verify_fn(
            adapter_path_or_none,
            base_model,
            eval_contexts,
            marker_text=marker_text,
            qwen_im_end_id=qwen_im_end_id,
        )

    seams = pilot.make_smoke_seams(cfg.reference_root)
    seams = dataclasses.replace(seams, marker_verify_fn=tracking_verify_stub)
    entry = pilot.run_class("marker", cfg, seams)
    assert entry["status"] == "success", entry.get("error")
    # verify_fn called exactly twice: once with adapter path, once with None (base).
    assert len(calls) == 2
    assert calls[0] is not None  # trained pass
    assert calls[1] is None  # base model pass


def test_config_from_args_smoke_defaults_shrink():
    args = pilot._parse_args(["--smoke"])
    cfg = pilot.config_from_args(args)
    assert cfg.mode == "smoke"
    assert cfg.n_eval_completions == 2
    assert cfg.n_extraction_rollouts == 1
    assert cfg.eval_question_limit == 2
    assert cfg.upload is False
    # smoke reference-root is tmp-scoped under out_root, never the repo root.
    assert cfg.reference_root != Path(".")


def test_config_from_args_full_defaults():
    args = pilot._parse_args(["--full"])
    cfg = pilot.config_from_args(args)
    assert cfg.mode == "full"
    assert cfg.n_eval_completions == 5
    assert cfg.n_extraction_rollouts == 10
    assert cfg.eval_question_limit is None
    assert cfg.upload is True
    assert cfg.reference_root == Path(".")


def test_main_smoke_returns_zero(tmp_path):
    rc = pilot.main(["--smoke", "--out-root", str(tmp_path / "out")])
    assert rc == 0
    assert (tmp_path / "out" / "calibration_report.json").exists()


# ── BLOCKER 1 regression: positive JSONL rows must contain MARKER_TEXT ────────


def test_marker_smoke_stubs_positives_contain_marker_text(tmp_path):
    """BLOCKER 1: every positive row from _make_marker_smoke_stubs encodes MARKER_TEXT.

    MarkerOnlyDataCollator decides row positivity by finding token id 83399 in
    input_ids.  A positive row whose completion text omits MARKER_TEXT (e.g. bare
    'placeholder') never encodes to id 83399, so ALL rows become negatives,
    inverting the training signal.  This test confirms the stub writes the token.
    """
    import json

    marker_datagen_fn, _, _marker_gen_fn = pilot._make_marker_smoke_stubs(n_pos=4, n_cn=4)
    mix_dir = tmp_path / "mix"
    pos_path, cn_path = marker_datagen_fn("source_sp", ["neg_sp"], mix_dir, seed=0)

    from explore_persona_space.artifacts.recipe import MARKER_TEXT

    # Every positive completion must include MARKER_TEXT (the leading-space ※ token).
    with open(pos_path) as f:
        pos_rows = [json.loads(line) for line in f]
    for row in pos_rows:
        completion_text = row["completion"][-1]["content"]
        assert MARKER_TEXT in completion_text, (
            f"positive completion missing MARKER_TEXT: {completion_text!r}"
        )

    # Negatives must NOT contain MARKER_TEXT (they train the turn-end tail only).
    with open(cn_path) as f:
        neg_rows = [json.loads(line) for line in f]
    for row in neg_rows:
        completion_text = row["completion"][-1]["content"]
        assert MARKER_TEXT not in completion_text, (
            f"negative completion must NOT contain MARKER_TEXT: {completion_text!r}"
        )


# ── BLOCKER A live test: _build_marker_class uses marker_gen_fn for on-policy R ─


def test_build_marker_class_uses_marker_gen_fn_for_on_policy_responses(tmp_path):
    """BLOCKER A (live function): _build_marker_class must use seams.marker_gen_fn to
    produce per-question-distinct on-policy responses for training mix positives.

    MarkerOnlyDataCollator trains gradient only on the MARKER token + turn-end tail.
    The response R before the marker must be the base model's OWN on-policy greedy
    output (not a constant 'placeholder').  The smoke marker_gen_fn returns
    f"resp::{hash(q) & 0xFFFF:04x}" — deterministically distinct per question —
    so this test can verify that each training-mix positive carries its question's
    unique response text, proving the live _build_marker_class code path used the
    injected seam rather than a constant fallback.

    The inline mix builder path (marker_datagen_fn=None) is exercised.  It calls
    AutoTokenizer.from_pretrained to assert the marker token-id; we patch that to
    avoid a network download while still traversing the real code path.
    """
    import json
    from unittest.mock import MagicMock, patch

    from explore_persona_space.artifacts.recipe import MARKER_TEXT, MARKER_TOKEN_ID

    behavior = MagicMock()
    behavior.source_context = "persona_software_engineer"
    behavior.negative_panel = ["persona_doctor", "default"]
    # Small question bank — enough to get at least 2 distinct questions.
    questions = [f"question_{i}" for i in range(3)]
    behavior.train_question_bank = questions
    behavior.eval_question_bank = questions[:2]

    cfg = _smoke_config(tmp_path, datagen_target_n=None, mode="full")

    # Build a marker_gen_fn seam that records calls and returns per-question fakes.
    calls: list[tuple] = []

    def recording_marker_gen_fn(qs: list, system_prompt) -> list:
        calls.append((tuple(qs), system_prompt))
        return [f"resp::{hash(q) & 0xFFFF:04x}" for q in qs]

    def no_train_fn(base_model, data_path, output_dir, *, cfg=None, callbacks=None, **kw):
        # Write a stub adapter directory so downstream can find it.
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoint-50").mkdir(parents=True, exist_ok=True)
        return str(out), 0.5

    seams = pilot.PilotSeams(
        marker_gen_fn=recording_marker_gen_fn,
        # marker_datagen_fn must be None so the inline builder (which uses
        # marker_gen_fn) is exercised, not the pre-built stub.
        marker_datagen_fn=None,
        train_fn=no_train_fn,
    )

    build_dir = tmp_path / "build_marker"
    build_dir.mkdir(parents=True)

    # Patch AutoTokenizer.from_pretrained to avoid a HF network call.
    # The inline builder uses it ONLY to assert the marker token-id; return a mock
    # tokenizer whose encode() returns [MARKER_TOKEN_ID] for the marker text.
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [MARKER_TOKEN_ID]
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        pilot._build_marker_class(behavior, cfg, seams, build_dir)

    # 1. marker_gen_fn must have been called.
    assert calls, "_build_marker_class never called seams.marker_gen_fn"

    # 2. Positive rows must each carry their question's unique response text.
    # _build_marker_class creates class_dir/"build"/"mix" internally.
    mix_dir = build_dir / "build" / "mix"
    pos_path = mix_dir / "pos.jsonl"
    assert pos_path.exists(), f"pos.jsonl not written to {pos_path}"

    with open(pos_path) as f:
        pos_rows = [json.loads(line) for line in f]

    for row in pos_rows:
        q_text = row["prompt"][-1]["content"]  # last prompt turn = user question
        expected_resp_prefix = f"resp::{hash(q_text) & 0xFFFF:04x}"
        completion_text = row["completion"][-1]["content"]
        assert completion_text.startswith(expected_resp_prefix), (
            f"positive for question {q_text!r} expected response starting with "
            f"{expected_resp_prefix!r}, got {completion_text!r} — "
            "suggests _build_marker_class used a constant placeholder rather than "
            "the on-policy marker_gen_fn output"
        )
        assert MARKER_TEXT in completion_text, (
            f"positive completion missing MARKER_TEXT: {completion_text!r}"
        )


# ── BLOCKER 2 regression: _extract_class loads from disk, never generates ─────


def test_extract_class_loads_from_disk_not_generate(tmp_path):
    """BLOCKER 2 (live function): _extract_class must load Phase-0 datagen completions
    from disk and MUST NOT call generate_contrastive_completions.

    This exercises the PRODUCTION _extract_class function (not the smoke stub) with
    injected score_fn and extract_fn seams, so the test verifies real code paths —
    not mock seam literals.
    """
    from unittest.mock import MagicMock, patch

    import torch

    from explore_persona_space.artifacts.directions import (
        ContrastiveCompletion,
        DirectionResult,
        save_completions_jsonl,
    )

    # Write fake completions.jsonl to disk (the production load path).
    datagen_dir = tmp_path / "datagen"
    datagen_dir.mkdir(parents=True)
    fake_completions = [
        ContrastiveCompletion(
            arm="exhibit",
            pair_index=0,
            system_prompt="sp",
            question="q0",
            response="pos resp 0",
            judge_score=80.0,
        ),
        ContrastiveCompletion(
            arm="not_exhibit",
            pair_index=0,
            system_prompt="sp",
            question="q0",
            response="neg resp 0",
            judge_score=10.0,
        ),
    ]
    save_completions_jsonl(fake_completions, datagen_dir / "contrastive_completions.jsonl")

    # Build a minimal fake behavior object.
    behavior = MagicMock()
    behavior.name = "sycophancy"

    # Use seams with score_fn and extract_fn injected so no model is needed.
    n_layers = 4
    fake_rb = torch.zeros(n_layers, 32)

    from explore_persona_space.artifacts.directions import JudgeResult

    def fake_score_fn(beh, completions, *, n_draws, cache_dir, save_raw, dry_run=False):
        import dataclasses

        scored = [dataclasses.replace(c, judge_score=75.0) for c in completions]
        jr = JudgeResult(
            scores={f"i{i}": 75.0 for i in range(len(scored))},
            n_total_draws=len(scored) * n_draws,
            n_dropped_draws=0,
        )
        return scored, jr

    def fake_extract_fn(beh, scored):
        return DirectionResult(
            behavior_name=beh.name,
            regime="steering",
            layers=tuple(range(n_layers)),
            r_b=fake_rb.clone(),
            counts={"smoke": True},
            provenance="claude_generated",
        )

    cfg = _smoke_config(tmp_path)
    class_dir = tmp_path / "class_sycophancy"
    class_dir.mkdir(parents=True)

    seams = pilot.PilotSeams(
        score_fn=fake_score_fn,
        extract_fn=fake_extract_fn,
    )

    # Patch generate_contrastive_completions to detect if it's ever called.
    with patch(
        "issue906_phase1_pilot.generate_contrastive_completions",
        side_effect=AssertionError(
            "_extract_class must NOT call generate_contrastive_completions on the production path"
        ),
    ):
        result, _jr, rb_path = pilot._extract_class(
            behavior, cfg, seams, class_dir, str(datagen_dir)
        )

    # generate_contrastive_completions was NOT called (no AssertionError raised).
    # Verify the result came from our injected extract_fn.
    assert result.regime == "steering", f"expected 'steering', got {result.regime!r}"
    assert result.provenance == "claude_generated", (
        f"expected 'claude_generated', got {result.provenance!r}"
    )
    assert rb_path.exists(), "r_b tensor must be persisted to disk"


# ── BLOCKER 3 regression: --full without --generic-data-path must exit ────────


def test_full_without_generic_data_path_exits(tmp_path):
    """BLOCKER 3: --full without --generic-data-path must fail fast with SystemExit.

    A warning-only path silently produces per-class errors for every class whose
    recipe has generic_frac > 0 instead of stopping at config time.
    """
    with pytest.raises(SystemExit) as exc_info:
        pilot.main(["--full"])
    assert exc_info.value.code != 0, "SystemExit must be non-zero (error exit)"


# ── CONCERN 4 regression: full-mode marker mix uses train_question_bank ───────


def test_build_marker_class_uses_train_question_bank(tmp_path, monkeypatch):
    """CONCERN 4: _build_marker_class must source questions from train_question_bank.

    eval_question_bank (capped at 20 questions) is for verification only; the
    training mix must draw from train_question_bank (100 questions) so question
    coverage is representative.
    """
    from unittest.mock import MagicMock, patch

    behavior = MagicMock()
    behavior.source_context = "persona_software_engineer"
    behavior.negative_panel = ["persona_doctor", "default"]
    # train bank has 100 items; eval bank has 20
    behavior.train_question_bank = [f"train_q_{i}" for i in range(100)]
    behavior.eval_question_bank = [f"eval_q_{i}" for i in range(20)]

    cfg = _smoke_config(tmp_path, datagen_target_n=None, mode="full")
    cfg = dataclasses.replace(cfg, datagen_target_n=None, mode="full")

    # _row is a local function inside _build_marker_class — cannot patch via module attribute.
    # Instead: mock AutoTokenizer to avoid loading a real model, provide a seam train_fn that
    # stops early, then read the written JSONL files to verify question provenance.
    fake_tokenizer = MagicMock()
    # encode(" ※", add_special_tokens=False) must return [83399] for the in-process assert.
    fake_tokenizer.encode.return_value = [83399]

    def stub_train_fn(*args, **kwargs):
        """Stop as soon as the mix files are written — we only need the JSONL content."""
        raise RuntimeError("stop_after_mix")

    seams = pilot.PilotSeams(
        train_fn=stub_train_fn,
        # Inject marker_gen_fn so _build_marker_class stays in the seam path and
        # never tries to load the base model (which would require a real GPU).
        marker_gen_fn=lambda qs, sp: [f"stub_resp_{i}" for i, _ in enumerate(qs)],
    )

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
        pytest.raises(RuntimeError, match="stop_after_mix"),
    ):
        pilot._build_marker_class(behavior, cfg, seams, tmp_path / "marker_class_out")

    import json

    # Read the written pos.jsonl and verify all questions come from train_question_bank.
    mix_dir = tmp_path / "marker_class_out" / "build" / "mix"
    pos_path = mix_dir / "pos.jsonl"
    assert pos_path.exists(), f"pos.jsonl not written to {mix_dir}"
    with open(pos_path) as f:
        pos_rows = [json.loads(line) for line in f]
    for row in pos_rows:
        # The question is the last user-turn content in the prompt list.
        q = row["prompt"][-1]["content"]
        assert q.startswith("train_q_"), (
            f"_build_marker_class used eval_question_bank question {q!r}; "
            "should use train_question_bank"
        )


# ── CONCERN 5 regression: _build_class passes bank-sized target_n in full mode ─


def test_build_class_passes_bank_target_n_in_full_mode(tmp_path, monkeypatch):
    """CONCERN 5: in full mode, _build_class must forward target_n=len(train_question_bank).

    Without this, generate_training_data falls back to the library default
    (target_n=200), which may under- or over-count relative to the actual bank.
    """
    from unittest.mock import MagicMock, patch

    behavior = MagicMock()
    behavior.source_context = "persona_software_engineer"
    behavior.negative_panel = ["persona_doctor"]
    behavior.train_question_bank = [f"q_{i}" for i in range(75)]
    behavior.dv = MagicMock()
    behavior.programmatic = False

    captured_kwargs: list[dict] = []

    def fake_build_organism(
        org,
        *,
        out_root,
        base_model,
        generic_data_path,
        gpu_id,
        datagen_kwargs,
        datagen_fn,
        train_fn,
        rate_fn,
    ):
        captured_kwargs.append(dict(datagen_kwargs))
        raise RuntimeError("stop here — we only need datagen_kwargs")

    cfg = _smoke_config(tmp_path, datagen_target_n=None, mode="full")

    # Use a plain MagicMock (no spec) so we can freely set .recipe attributes without
    # the spec restricting which attrs exist on ModelOrganism.
    fake_org = MagicMock()
    fake_org.recipe.train_method = "lora"
    fake_org.recipe.stopping.kind = "fixed_epochs"

    # build_organism is imported as a local inside _build_class from
    # explore_persona_space.artifacts.organisms — patch there, not on the pilot module.
    with (
        patch("explore_persona_space.artifacts.organisms.build_organism", fake_build_organism),
        pytest.raises(RuntimeError, match="stop here"),
    ):
        pilot._build_class(behavior, fake_org, cfg, pilot.PilotSeams(), tmp_path / "class")

    assert captured_kwargs, "build_organism was never called"
    assert captured_kwargs[0].get("target_n") == 75, (
        f"expected target_n=75 (bank size), got {captured_kwargs[0]}"
    )


# ── CONCERN 8 regression: judge_refusal_fractions non-None when totals present ─


def test_summarize_judge_refusal_fractions_from_flat_keys():
    """CONCERN 8: _summarize must read flat judge_draws_total / judge_draws_dropped keys.

    _pool_meta_counts returns FLAT keys, NOT the nested judge_draw_stats dict.
    The old nested access always resolved to {}, so judge_refusal_fractions was
    always None even when totals were present.
    """
    classes = {
        "sycophancy": {
            "status": "success",
            "api_calls": {
                "datagen": {
                    "judge_draws_total": 100,
                    "judge_draws_dropped": 5,
                }
            },
            "timings_seconds": {"total": 1.0},
            "install": {},
        },
        "harmful_compliance": {
            "status": "success",
            "api_calls": {
                "datagen": {
                    "judge_draws_total": 80,
                    "judge_draws_dropped": 0,
                }
            },
            "timings_seconds": {"total": 1.0},
            "install": {},
        },
        "marker": {
            "status": "success",
            "api_calls": {},
            "timings_seconds": {"total": 1.0},
            "install": {},
        },
    }

    summary = pilot._summarize(classes)
    fractions = summary.get("judge_refusal_fractions", {})

    assert fractions["sycophancy"] == pytest.approx(5 / 100, abs=1e-4), (
        f"expected 0.05 for sycophancy, got {fractions['sycophancy']}"
    )
    assert fractions["harmful_compliance"] == pytest.approx(0.0, abs=1e-4), (
        f"expected 0.0 for harmful_compliance, got {fractions['harmful_compliance']}"
    )
    # marker has no datagen judge calls — should be None
    assert fractions["marker"] is None, (
        f"expected None for marker (no judge calls), got {fractions['marker']}"
    )
