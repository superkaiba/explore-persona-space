"""CPU/mock unit tests for the issue #906 Phase-1 pilot driver.

Drives the orchestration logic (`run_pilot` / `run_class` and the report
assembly + reproduction-cosine + api-count helpers) with fully mocked
organism/direction/judge/margin seams. No GPU, no live API, no network — the
`make_smoke_seams()` fakes exercise the real library orchestration
(`build_organism` mix assembly + dose selection, `verify_organism` install +
leakage + tf-margin, `generate_contrastive_completions`) with every heavyweight
boundary stubbed. The marker class flows through the REAL
`UnsupportedOrganismError` carve-out (build_organism refuses programmatic
behaviors), so no mocking is needed to exercise the `unsupported_v1` path.
"""

from __future__ import annotations

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


def test_marker_recorded_unsupported_not_error(pilot_report):
    _cfg, report = pilot_report
    marker = report["classes"]["marker"]
    assert marker["status"] == "unsupported_v1"
    assert marker["programmatic"] is True
    assert "unsupported_reason" in marker
    # marker is a known carve-out, NOT a genuine error.
    assert report["summary"]["n_unsupported_v1"] == 1
    assert report["summary"]["any_errors"] is False


def test_non_programmatic_classes_succeed(pilot_report):
    _cfg, report = pilot_report
    for name in ("sycophancy", "harmful_compliance", "china_censorship"):
        entry = report["classes"][name]
        assert entry["status"] == "success", entry.get("error")
        assert entry["programmatic"] is False
    assert report["summary"]["n_success"] == 3
    assert report["summary"]["n_error"] == 0


def test_timers_populate_per_phase(pilot_report):
    _cfg, report = pilot_report
    syc = report["classes"]["sycophancy"]["timings_seconds"]
    for phase in ("build", "verify", "extract", "total"):
        assert phase in syc and isinstance(syc[phase], (int, float))
    # marker fails at build (unsupported) -> no verify/extract timers, total present.
    marker = report["classes"]["marker"]["timings_seconds"]
    assert "total" in marker
    assert "verify" not in marker


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


def test_marker_has_no_direction_block(pilot_report):
    _cfg, report = pilot_report
    # Programmatic + build-unsupported -> extract never runs, no direction key.
    assert "direction" not in report["classes"]["marker"]


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


def test_run_class_marker_unsupported(tmp_path):
    cfg = _smoke_config(tmp_path, classes=("marker",))
    seams = pilot.make_smoke_seams(cfg.reference_root)
    entry = pilot.run_class("marker", cfg, seams)
    assert entry["status"] == "unsupported_v1"
    # recipe is still recorded for calibration completeness.
    assert entry["recipe"]["stopping_kind"] == "marker_band_stop"


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
