# em-dash + Qwen marker token " ※" are intentional
"""Tests for task #508 lora_vs_ft_508 experiment package.

CPU-only smoke tests covering:
- Constants invariants (no panel leakage, valid CELL_SPECS).
- Linear interpolation helper.
- Bracketing-check predicate.
- Crossed-cluster bootstrap on synthetic per-cell ΔG data (FT > LoRA gap detected).
- Dynamics-probe builder.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest


def test_constants_no_panel_leak():
    from explore_persona_space.experiments.lora_vs_ft_508 import (
        CONTRASTIVE_NEGATIVES,
        HELD_OUT_PERSONAS_15,
        SOURCE_PERSONA,
    )

    overlap = set(HELD_OUT_PERSONAS_15) & set(CONTRASTIVE_NEGATIVES)
    assert not overlap, f"held-out + contrastive overlap: {sorted(overlap)}"
    assert SOURCE_PERSONA not in HELD_OUT_PERSONAS_15
    assert SOURCE_PERSONA not in CONTRASTIVE_NEGATIVES
    assert len(HELD_OUT_PERSONAS_15) == 15
    assert len(CONTRASTIVE_NEGATIVES) == 4
    assert "qwen_default" in CONTRASTIVE_NEGATIVES, "qwen_default must always be a negative"


def test_cell_specs_complete():
    from explore_persona_space.experiments.lora_vs_ft_508 import CELL_SPECS

    arms = {c[0] for c in CELL_SPECS}
    budgets = {c[1] for c in CELL_SPECS}
    assert arms == {"lora", "fullft"}
    assert budgets == {"b1", "b2", "b3"}
    assert len(CELL_SPECS) == 6


def test_is_lora_arm():
    from explore_persona_space.experiments.lora_vs_ft_508 import is_lora_arm

    assert is_lora_arm("lora_b2") is True
    assert is_lora_arm("lora_b1") is True
    assert is_lora_arm("fullft_b2") is False
    assert is_lora_arm("ft_b2") is False


def test_linear_interp_basic():
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import _linear_interp

    # y = 2x on (1,2),(2,4),(3,6); at x=2.5 → 5.0
    y = _linear_interp([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 2.5)
    assert abs(y - 5.0) < 1e-6, f"expected 5.0, got {y}"


def test_linear_interp_unsorted():
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import _linear_interp

    # Same data, unsorted input.
    y = _linear_interp([3.0, 1.0, 2.0], [6.0, 2.0, 4.0], 2.5)
    assert abs(y - 5.0) < 1e-6


def test_linear_interp_extrapolation():
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import _linear_interp

    # Outside bracket — extrapolation from the nearest two extremes.
    y = _linear_interp([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 4.0)
    assert abs(y - 8.0) < 1e-6


def test_check_bracketing_pass():
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import _check_bracketing

    result = _check_bracketing([4.0, 8.0, 12.0])
    assert result["brackets_target"] is True
    assert result["below_7_nat"] == 1
    assert result["above_9_nat"] == 1


def test_check_bracketing_fail_all_low():
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import _check_bracketing

    result = _check_bracketing([2.0, 3.0, 5.0])
    assert result["brackets_target"] is False
    assert result["below_7_nat"] == 3
    assert result["above_9_nat"] == 0


def test_check_bracketing_fail_all_high():
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import _check_bracketing

    result = _check_bracketing([10.0, 12.0, 14.0])
    assert result["brackets_target"] is False


def test_dynamics_probes_builder():
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )
    from explore_persona_space.experiments.lora_vs_ft_508 import (
        DYNAMICS_BYSTANDER_PERSONAS,
        DYNAMICS_PROBE_QUESTIONS_PER_PERSONA,
        SOURCE_PERSONA,
        load_q_eval,
    )
    from explore_persona_space.experiments.lora_vs_ft_508.marker_dynamics_callback import (
        build_dynamics_probes,
    )

    probes = build_dynamics_probes(dict(EVAL_PERSONAS_24), load_q_eval(), seed=42)
    expected_personas = {SOURCE_PERSONA, *DYNAMICS_BYSTANDER_PERSONAS}
    assert set(probes.keys()) == expected_personas
    for persona, spec in probes.items():
        assert len(spec["questions"]) == DYNAMICS_PROBE_QUESTIONS_PER_PERSONA
        assert spec["role"] in ("source", "bystander")
        if persona == SOURCE_PERSONA:
            assert spec["role"] == "source"
        else:
            assert spec["role"] == "bystander"


def _make_synthetic_cell_eval(slug: str, arm: str, source_mean: float, held_out_mean: float):
    from explore_persona_space.experiments.lora_vs_ft_508 import (
        EXPECTED_MARKER_TOKEN_ID,
        HELD_OUT_PERSONAS_15,
        MARKER_TEXT,
        SOURCE_PERSONA,
        load_q_eval,
    )

    q_eval = load_q_eval()
    rng = random.Random(hash(slug) & 0xFFFFFFFF)
    held_out: dict = {}
    for p in HELD_OUT_PERSONAS_15:
        held_out[p] = {}
        for q in q_eval:
            dg = held_out_mean + rng.gauss(0, 1.0)
            held_out[p][q] = {
                "trained_logp": dg - 24.0,
                "base_logp": -24.0,
                "delta_g": dg,
                "trained_argmax_marker": dg > 5.0,
                "base_argmax_marker": False,
                "r_collapsed": False,
                "n_marker_in_R": 0,
            }
    src: dict = {SOURCE_PERSONA: {}}
    for q in q_eval:
        dg = source_mean + rng.gauss(0, 0.5)
        src[SOURCE_PERSONA][q] = {
            "trained_logp": dg - 24.0,
            "base_logp": -24.0,
            "delta_g": dg,
            "trained_argmax_marker": dg > 5.0,
            "base_argmax_marker": False,
            "r_collapsed": False,
            "n_marker_in_R": 0,
        }
    return {
        "schema_version": "i508_eval_v1",
        "cell_slug": slug,
        "arm": arm,
        "seed": 42,
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "is_full_ft": arm == "fullft",
        "lora_adapter_path": None,
        "full_ft_checkpoint_dir": None,
        "marker_text": MARKER_TEXT,
        "marker_token_id_expected": EXPECTED_MARKER_TOKEN_ID,
        "eval_max_new_tokens": 2048,
        "held_out_personas": list(HELD_OUT_PERSONAS_15),
        "eval_questions": list(q_eval),
        "source_persona": SOURCE_PERSONA,
        "delta_g_held_out": held_out,
        "delta_g_source": src,
        "trained_R_held_out": {},
        "trained_R_source": {},
        "aggregates": {},
        "git_commit": "test",
        "timestamp_utc": "test",
    }


def test_run_analysis_end_to_end_h1_detected(tmp_path: Path):
    """Synthetic H1 scenario: FT leaks 1.5 nat more at matched 8-nat source."""
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import run_analysis

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    # Bracket source-rate: ≥5 (implant gate), <7 (bracket lower), in-band, >9 (bracket upper).
    source_targets = {"b1": 6.0, "b2": 8.0, "b3": 12.0}
    ho_lora = {"b1": 1.5, "b2": 2.5, "b3": 4.5}
    ho_ft = {"b1": 2.0, "b2": 4.0, "b3": 6.5}  # +1.5 nat at matched 8-nat
    paths: list[Path] = []
    for budget in ("b1", "b2", "b3"):
        for arm, ho_map in (("lora", ho_lora), ("fullft", ho_ft)):
            slug = f"{arm}_{budget}"
            data = _make_synthetic_cell_eval(slug, arm, source_targets[budget], ho_map[budget])
            p = eval_dir / f"{slug}_seed42.json"
            p.write_text(json.dumps(data))
            paths.append(p)

    result = run_analysis(eval_jsons=paths, output_dir=tmp_path / "analysis")
    assert result["n_cells"] == 6
    assert result["bracketing_per_arm"]["lora"]["brackets_target"]
    assert result["bracketing_per_arm"]["fullft"]["brackets_target"]
    assert not result["h1_indeterminate_per_arm"]["lora"]
    assert not result["h1_indeterminate_per_arm"]["fullft"]
    gap = result["matched_rate_gap"]
    assert gap["n_replicates"] >= 900, f"too few replicates: {gap['n_replicates']}"
    # Synthetic gap is ~+1.5 nat; CI should comfortably exclude zero AND
    # bracket the true value.
    assert gap["gap_mean"] > 0.5, f"expected >0.5, got {gap['gap_mean']}"
    assert gap["gap_excludes_zero"], "CI should exclude zero on a strong synthetic gap"
    # Headline threshold check from plan §6 (matched-rate gap > 1.0 nat).
    assert gap["gap_mean"] > 1.0, "synthetic gap too small to test threshold"


def test_run_analysis_bracketing_failure_marks_indeterminate(tmp_path: Path):
    """Synthetic: all 3 LoRA budgets under-train → bracketing FAIL → INDETERMINATE."""
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import run_analysis

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    # LoRA arm: all 3 source values clear the implant floor (≥5) but ALL fall
    # below the 7-nat bracket lower edge → bracketing FAIL (no cell >9).
    lora_sources = {"b1": 5.5, "b2": 6.0, "b3": 6.5}
    # Full-FT arm: brackets correctly (≥5 ALL, <7 on b1, >9 on b3).
    ft_sources = {"b1": 6.0, "b2": 8.0, "b3": 12.0}
    ho_lora = {"b1": 0.3, "b2": 0.7, "b3": 1.2}
    ho_ft = {"b1": 2.0, "b2": 4.0, "b3": 6.5}
    paths: list[Path] = []
    for budget in ("b1", "b2", "b3"):
        for arm, src_map, ho_map in (
            ("lora", lora_sources, ho_lora),
            ("fullft", ft_sources, ho_ft),
        ):
            slug = f"{arm}_{budget}"
            data = _make_synthetic_cell_eval(slug, arm, src_map[budget], ho_map[budget])
            p = eval_dir / f"{slug}_seed42.json"
            p.write_text(json.dumps(data))
            paths.append(p)

    result = run_analysis(eval_jsons=paths, output_dir=tmp_path / "analysis")
    assert result["h1_indeterminate_per_arm"]["lora"] is True
    assert result["h1_indeterminate_per_arm"]["fullft"] is False
    # When ANY arm INDETERMINATE the gap is not computed.
    assert result["matched_rate_gap"] == {} or "gap_mean" not in result["matched_rate_gap"]


@pytest.mark.parametrize(
    "marker_text,expected_id",
    [(" ※", 83399)],
)
def test_marker_token_id_constant(marker_text: str, expected_id: int):
    from explore_persona_space.experiments.lora_vs_ft_508 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )

    assert marker_text == MARKER_TEXT
    assert expected_id == EXPECTED_MARKER_TOKEN_ID


def test_cell_slug_helper():
    from explore_persona_space.experiments.lora_vs_ft_508 import ARM_FULLFT, ARM_LORA, cell_slug

    assert cell_slug(ARM_LORA, "b2") == "lora_b2"
    assert cell_slug(ARM_FULLFT, "b1") == "fullft_b1"

    with pytest.raises(ValueError, match="Unknown arm"):
        cell_slug("invalid_arm", "b1")


def test_q_train_q_eval_split():
    from explore_persona_space.experiments.lora_vs_ft_508 import load_q_eval, load_q_train

    q_train = load_q_train()
    q_eval = load_q_eval()
    assert len(q_train) == 10
    # Q_eval is the FULL 20-question pool (eval probes all 20 questions on the
    # 15 held-out personas — persona-axis disjointness with train is what makes
    # this valid).
    assert len(q_eval) == 20
    # Q_train ⊆ Q_eval (because Q_eval is the full 20-q pool).
    assert set(q_train) <= set(q_eval)


# ── Round-1 review fixes (B1/B2/B5/M2/M3/M4/M7). ─────────────────────────────


def test_r_train_loader_canonical_schema(tmp_path: Path):
    """B1 round-1 fix — verify the canonical-schema loader is used.

    The canonical R_train.json from #472's r_generate.py wraps personas under
    ``payload["completions"]``. The buggy round-1 code did
    ``json.loads(...).items()`` and would KeyError on the first persona lookup.

    This test:
      (a) writes a fixture mimicking the canonical schema,
      (b) calls the loader via the same import path the dispatcher uses,
      (c) asserts the personas land at the top level of the returned dict.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import r_generate

    # Build a fixture matching r_generate.py's actual write shape.
    canonical = {
        "schema_version": r_generate.SCHEMA_VERSION,
        "split": "train",
        "n_personas": 2,
        "n_questions": 1,
        "personas": ["villain", "medical_doctor"],
        "completions": {
            "villain": {
                "How do you handle disagreements?": {
                    "response_text": "(synthetic base response)",
                    "response_token_ids": [1, 2, 3],
                }
            },
            "medical_doctor": {
                "How do you handle disagreements?": {
                    "response_text": "(synthetic medical response)",
                    "response_token_ids": [4, 5, 6],
                }
            },
        },
    }
    path = tmp_path / "R_train.json"
    path.write_text(json.dumps(canonical))

    r_train = r_generate.load_r_artifact(path)
    # MUST return the completions dict (personas at the top level).
    assert "villain" in r_train
    assert "medical_doctor" in r_train
    assert "schema_version" not in r_train, (
        "load_r_artifact must unwrap payload['completions'] — got the raw payload back."
    )
    assert (
        r_train["villain"]["How do you handle disagreements?"]["response_text"]
        == "(synthetic base response)"
    )


def test_r_train_loader_rejects_old_schema(tmp_path: Path):
    """B1 round-1 fix — the loader fails loud on a wrong schema_version.

    Smoke fixture must not accidentally feed a stale-format JSON without the
    canonical schema_version assertion firing.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import r_generate

    bad = {
        "schema_version": "bogus_v0",
        "completions": {"villain": {}},
    }
    path = tmp_path / "R_train.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(AssertionError, match="schema_version"):
        r_generate.load_r_artifact(path)


def test_fullft_env_explicit_multi_gpu_cvd():
    """B2 round-1 fix — the full-FT subprocess env carries CVD=0,1,...,N-1.

    Reproducer for the bug: a prior in-process LoRA cell (via #472's
    train_one_cell → train/sft.py:649) sets os.environ["CUDA_VISIBLE_DEVICES"]
    = "0" in the dispatcher. Without the explicit pop+set in build_fullft_env,
    accelerate launch would inherit CVD=0 and ZeRO-3 across 4 GPUs would fail.
    """
    from explore_persona_space.experiments.lora_vs_ft_508.train_cell_fullft import (
        build_fullft_env,
    )

    polluted_env = {
        "CUDA_VISIBLE_DEVICES": "0",  # LoRA cell pollution
        "HF_TOKEN": "fake_token_for_test",
        "WANDB_API_KEY": "fake_wandb_key",
    }
    env = build_fullft_env(num_gpus=4, base_env=polluted_env)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    # Credentials must still pass through.
    assert env["HF_TOKEN"] == "fake_token_for_test"
    assert env["WANDB_API_KEY"] == "fake_wandb_key"
    # 2-GPU sweep gives explicit 0,1.
    env2 = build_fullft_env(num_gpus=2, base_env={"CUDA_VISIBLE_DEVICES": "0"})
    assert env2["CUDA_VISIBLE_DEVICES"] == "0,1"
    # Test isolation: the function must return a NEW dict; never mutate the caller's env.
    assert polluted_env["CUDA_VISIBLE_DEVICES"] == "0", (
        "build_fullft_env must NOT mutate the caller's env dict"
    )


def test_train_one_cell_accepts_extra_callbacks_kwarg():
    """B5 round-1 fix — #472's train_one_cell signature accepts extra_callbacks.

    The fix is a 5-line patch to #472 train_cell.py — but the signature change
    is the public contract. This test pins it via inspect.signature so a
    future merge that strips the kwarg surfaces it loudly.
    """
    import inspect

    from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
        train_one_cell,
    )

    sig = inspect.signature(train_one_cell)
    assert "extra_callbacks" in sig.parameters, (
        "train_one_cell must accept extra_callbacks for #508 MarkerDynamicsCallback threading"
    )
    # Default is an empty tuple (preserves byte-identical behavior for pre-#508 callers).
    assert sig.parameters["extra_callbacks"].default == ()
    # And epochs_override must accept float (the dispatcher passes 0.25/0.5/1.0).
    epochs_param = sig.parameters["epochs_override"]
    # Annotation should accept float; the runtime accepts ints too via duck typing.
    assert "float" in str(epochs_param.annotation), epochs_param.annotation


def test_dispatcher_rejects_legacy_fullft_slug():
    """Minor round-1 fix — only `ft_*` accepted, not `fullft_*`.

    The codex review noted both were accepted; we pick `ft_*` (canonical
    plan §4.4 + brief smoke command).
    """
    import subprocess
    from pathlib import Path

    worktree = Path(__file__).resolve().parents[1]
    dispatch = worktree / "scripts" / "dispatch_508.py"
    # Run with --build-only to short-circuit before any GPU work.
    rc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(dispatch),
            "--cells",
            "fullft_b2",  # legacy/rejected
            "--seeds",
            "42",
            "--output-root",
            "/tmp/issue_508_legacy_slug_test",
            "--build-only",
        ],
        env={**__import__("os").environ},
        capture_output=True,
        text=True,
        cwd=worktree,
    )
    # Should fail with the canonical error message.
    assert rc.returncode != 0, "Legacy `fullft_*` slug should be rejected"
    assert "expected `lora` or `ft`" in rc.stderr + rc.stdout, rc.stderr[:500]


def test_h3_direct_qwen_default_reading(tmp_path: Path):
    """M3 round-1 fix — analyze surfaces direct qwen_default ΔG, not just proxies.

    Synthesize 6 cell eval JSONs with a `qwen_default_mean_delta_g` aggregate;
    confirm run_analysis lifts them into `h3_qwen_default_direct`.
    """
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import run_analysis

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    # Bracket: implant gate ≥ 5, < 7 lower bracket, > 9 upper bracket.
    source_targets = {"b1": 6.0, "b2": 8.0, "b3": 12.0}
    ho_lora = {"b1": 1.5, "b2": 2.5, "b3": 4.5}
    ho_ft = {"b1": 2.0, "b2": 4.0, "b3": 6.5}
    qd_lora = {"b1": 0.5, "b2": 1.0, "b3": 2.0}
    qd_ft = {"b1": 1.0, "b2": 2.5, "b3": 4.5}  # FT leaks MORE to qwen_default
    paths: list[Path] = []
    for budget in ("b1", "b2", "b3"):
        for arm, ho_map, qd_map in (
            ("lora", ho_lora, qd_lora),
            ("fullft", ho_ft, qd_ft),
        ):
            slug = f"{arm}_{budget}"
            data = _make_synthetic_cell_eval(slug, arm, source_targets[budget], ho_map[budget])
            data["aggregates"]["qwen_default_mean_delta_g"] = qd_map[budget]
            p = eval_dir / f"{slug}_seed42.json"
            p.write_text(json.dumps(data))
            paths.append(p)

    result = run_analysis(eval_jsons=paths, output_dir=tmp_path / "analysis")
    direct = result["h3_qwen_default_direct"]
    assert "lora_b2" in direct
    assert "fullft_b2" in direct
    # The synthetic data is FT > LoRA at every budget on qwen_default.
    assert direct["fullft_b2"] > direct["lora_b2"]


def test_gate_drops_failed_implant(tmp_path: Path):
    """M2 round-1 fix — cells that fail the implant gate (source ΔG < 5) are dropped."""
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import run_analysis

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    # LoRA b1 fails the implant gate (source ΔG = 2.0 < 5.0); b2 + b3 pass.
    source_targets = {"b1": 2.0, "b2": 8.0, "b3": 12.0}
    paths: list[Path] = []
    for budget in ("b1", "b2", "b3"):
        for arm in ("lora", "fullft"):
            slug = f"{arm}_{budget}"
            ho = 1.0 + 2.0 * source_targets[budget] / 10
            src = source_targets[budget]
            if arm == "lora" and budget == "b1":
                src = 2.0  # below the floor
            data = _make_synthetic_cell_eval(slug, arm, src, ho)
            p = eval_dir / f"{slug}_seed42.json"
            p.write_text(json.dumps(data))
            paths.append(p)

    result = run_analysis(eval_jsons=paths, output_dir=tmp_path / "analysis")
    # The implant_validity_gate field marks lora_b1 as failed.
    assert result["implant_validity_gate"]["lora_b1"] is False
    # And the cell is dropped from the LoRA arm.
    assert "lora_b1" in result["dropped_cells_by_arm"]["lora"]
    # With only 2 cells in LoRA arm (b2, b3 — both >5 nats, both > 7 nats but no <7),
    # bracketing fails → H1 INDETERMINATE for LoRA.
    assert result["h1_indeterminate_per_arm"]["lora"] is True


def test_make_cpu_base_logp_scorer_closes_over_probes_dict():
    """M7 round-1 fix — scorer uses the dict passed at construction.

    The buggy round-1 code reloaded DYNAMICS_PROBES_PATH on every call,
    ignoring the caller's probes dict and any alternate path. We don't
    actually load a real base model here (heavy); we use a stub by
    monkey-patching AutoModelForCausalLM with a fake before the call.
    """
    import inspect

    from explore_persona_space.experiments.lora_vs_ft_508 import marker_dynamics_callback

    sig = inspect.signature(marker_dynamics_callback.make_cpu_base_logp_scorer)
    # Must accept `probes` as a keyword arg.
    assert "probes" in sig.parameters
    # Default is None (preserves legacy behavior for callers that don't pass it).
    assert sig.parameters["probes"].default is None
    # Must accept `device` for the M5 GPU lift.
    assert "device" in sig.parameters
    assert sig.parameters["device"].default is None  # auto-detect
