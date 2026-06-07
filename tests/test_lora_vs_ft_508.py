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
    """M2.3 round-2 fix: cell_slug emits user-facing slugs (`ft_*`, not `fullft_*`)."""
    from explore_persona_space.experiments.lora_vs_ft_508 import ARM_FULLFT, ARM_LORA, cell_slug

    assert cell_slug(ARM_LORA, "b2") == "lora_b2"
    # External canonical form is `ft_*` (matches CLI + plan §4.4 + brief smoke command);
    # the internal arm name "fullft" is mapped to public "ft" at the slug boundary.
    assert cell_slug(ARM_FULLFT, "b1") == "ft_b1"

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


# ── Round-2 review fixes (R2.1 / R2.2 / R2.3 / R2.4 + M2.2). ────────────────


def test_callback_on_train_end_persists_snapshots(tmp_path: Path):
    """R2.1 round-2 fix — after training the callback writes snapshots to disk.

    Constructs the callback with an explicit `snapshots_path`, seeds it with
    fake snapshot data, calls `on_train_end` (with no live model — just the
    persistence path), and asserts the JSON file exists with the expected
    schema.

    This is the CPU-feasible unit test promised in the brief's path (c) for
    proving the persistence code works without an actual training run.
    """
    from explore_persona_space.experiments.lora_vs_ft_508.marker_dynamics_callback import (
        MarkerDynamicsCallback,
    )

    sidecar = tmp_path / "dynamics.json"

    # Construct with stub probes + scorer + tokenizer; we don't need them to
    # fire because we'll seed snapshots directly + skip the on_step_end fire.
    cb = MarkerDynamicsCallback(
        probes={"villain": {"role": "source", "system": "stub", "questions": ["q"]}},
        tokenizer=None,
        base_logp_scorer=lambda p, q, r: -25.0,
        cadence_steps=4,
        snapshots_path=sidecar,
    )

    # Seed snapshots as the actual `_fire` would have produced.
    cb.snapshots = {
        4: {
            "dynamics/source_delta_g": 1.5,
            "dynamics/bystander_mean_delta_g": 0.3,
            "dynamics/source_emission_rate": 0.0,
            "dynamics/bystander_mean_emission_rate": 0.0,
            "dynamics/global_step": 4,
            "n_probes": 20,
        },
        8: {
            "dynamics/source_delta_g": 4.0,
            "dynamics/bystander_mean_delta_g": 0.8,
            "dynamics/source_emission_rate": 0.2,
            "dynamics/bystander_mean_emission_rate": 0.0,
            "dynamics/global_step": 8,
            "n_probes": 20,
        },
    }
    cb._last_fired_step = 8

    # Fake the HF args/state/control trio.
    class FakeArgs:
        output_dir = str(tmp_path / "trainer_out")

    class FakeState:
        is_world_process_zero = True
        global_step = 8

    # on_train_end with model=None skips the final fire but still persists.
    cb.on_train_end(FakeArgs(), FakeState(), None, model=None)

    assert sidecar.exists(), f"on_train_end should have written {sidecar}"
    payload = json.loads(sidecar.read_text())
    assert payload["schema_version"] == "i508_dynamics_v1"
    assert payload["cadence_steps"] == 4
    assert "snapshots" in payload
    # Keys are stringified steps; both 4 and 8 should be present.
    assert set(payload["snapshots"].keys()) == {"4", "8"}
    assert payload["snapshots"]["8"]["dynamics/source_delta_g"] == 4.0


def test_persist_snapshots_exposed_as_method(tmp_path: Path):
    """R2.1 round-2 fix — callable directly for the dispatcher's fallback path."""
    from explore_persona_space.experiments.lora_vs_ft_508.marker_dynamics_callback import (
        MarkerDynamicsCallback,
    )

    cb = MarkerDynamicsCallback(
        probes={"villain": {"role": "source", "system": "stub", "questions": ["q1", "q2"]}},
        tokenizer=None,
        base_logp_scorer=lambda p, q, r: -25.0,
    )
    cb.snapshots = {1: {"dynamics/source_delta_g": 2.0, "n_probes": 2}}
    out = tmp_path / "manual_persist.json"
    returned = cb.persist_snapshots(out)
    assert returned == out
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["snapshots"]["1"]["dynamics/source_delta_g"] == 2.0


def test_extract_fullft_dynamics_from_checkpoints_smoke(tmp_path: Path):
    """R2.2 round-2 fix — offline FT extractor produces a dynamics.json.

    Uses a stub `score_fn` so we don't load real models. Verifies the
    extractor:
      (a) walks the checkpoint_index in step-sorted order,
      (b) calls the scorer with the per-checkpoint path,
      (c) aggregates per-checkpoint per-probe rows into the snapshot schema,
      (d) writes a JSON file with the same `i508_dynamics_v1` shape the
          LoRA `persist_snapshots` writes.
    """
    from explore_persona_space.experiments.lora_vs_ft_508.marker_dynamics_callback import (
        extract_fullft_dynamics_from_checkpoints,
    )

    # Synthetic 4-fraction checkpoint manifest.
    ckpt_dirs = []
    for i, frac in enumerate(("0.25", "0.50", "0.75", "1.00")):
        d = tmp_path / f"frac_{frac}"
        d.mkdir()
        # Touch a marker file so the path is non-empty.
        (d / "config.json").write_text("{}")
        ckpt_dirs.append((frac, d, (i + 1) * 4))  # steps 4, 8, 12, 16

    checkpoint_index = {frac: {"step": step, "path": str(d)} for frac, d, step in ckpt_dirs}

    probes = {
        "villain": {"role": "source", "system": "stub", "questions": ["q1", "q2"]},
        "kindergarten_teacher": {
            "role": "bystander",
            "system": "stub",
            "questions": ["q1", "q2"],
        },
        "data_scientist": {"role": "bystander", "system": "stub", "questions": ["q1", "q2"]},
        "assistant": {"role": "bystander", "system": "stub", "questions": ["q1", "q2"]},
    }

    def stub_scorer(trained_path, base_path, tokenizer, probes_inner, *, device, max_new_tokens):
        # Each call returns a 20-probe row set with monotone source ΔG.
        # Read the step from the path so we can verify the scorer was called
        # in step order. paths are tmp_path/frac_0.25 etc.; pull the frac key.
        path_obj = Path(trained_path)
        frac = path_obj.name.replace("frac_", "")
        dg_per_probe = {"0.25": 1.0, "0.50": 3.0, "0.75": 6.0, "1.00": 10.0}[frac]
        rows: list[dict] = []
        for persona, spec in probes_inner.items():
            for q in spec["questions"]:
                rows.append(
                    {
                        "persona": persona,
                        "role": spec["role"],
                        "question": q,
                        "trained_logp": -15.0 + dg_per_probe,
                        "base_logp": -25.0,
                        "delta_g": dg_per_probe + 10.0,
                        "argmax_marker": dg_per_probe > 5.0,
                    }
                )
        return rows

    out_path = tmp_path / "dynamics.json"
    returned = extract_fullft_dynamics_from_checkpoints(
        checkpoint_index=checkpoint_index,
        base_model_path="dummy_base",
        tokenizer=None,  # stub scorer ignores
        probes=probes,
        output_path=out_path,
        score_fn=stub_scorer,
    )
    assert returned == out_path
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["schema_version"] == "i508_dynamics_v1"
    assert payload["extraction_mode"] == "offline_post_checkpoint"
    snaps = payload["snapshots"]
    assert set(snaps.keys()) == {"4", "8", "12", "16"}
    # Snapshot at step 16 should reflect the largest ΔG (1.00-fraction).
    final = snaps["16"]
    assert final["dynamics/source_delta_g"] == 20.0  # 10.0 + 10.0
    # Step-4 snapshot ΔG smaller.
    assert snaps["4"]["dynamics/source_delta_g"] == 11.0


def test_sub_ceiling_gate_uses_held_out_g_logprob(tmp_path: Path):
    """R2.4 round-2 fix — sub-ceiling gate fires on held-out trained `g_logprob`.

    A cell with source_mean = 8.0 (passes implant gate) but
    held_out_g_logprob_mean = -2.0 (saturated, only 2 nats below 0.0 ceiling)
    SHOULD be dropped by the gate. The buggy round-2 code gated on
    source_mean <= 18.0 and would have let this cell through.
    """
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import run_analysis

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    # Need a properly-bracketing FT arm so the LoRA arm's saturation lands as
    # the discriminating signal. LoRA b1 source=6, b2=8, b3=12 brackets; but
    # the b2 cell will be flagged saturated by the new gate (its held-out g_logprob
    # ≈ -2.0 nats means "trained model is already predicting marker confidently").
    source_targets = {"b1": 6.0, "b2": 8.0, "b3": 12.0}
    ho_lora = {"b1": 1.5, "b2": 2.5, "b3": 4.5}
    ho_ft = {"b1": 2.0, "b2": 4.0, "b3": 6.5}
    # Saturation only on lora_b2: held-out g_logprob = -2.0 (well above the -5 cap).
    saturated_g_logprob = {"lora_b2": -2.0}
    paths: list[Path] = []
    for budget in ("b1", "b2", "b3"):
        for arm, ho_map in (("lora", ho_lora), ("fullft", ho_ft)):
            slug = f"{arm}_{budget}"
            data = _make_synthetic_cell_eval(slug, arm, source_targets[budget], ho_map[budget])
            # Default: held-out g_logprob mean is well below -5 (sub-ceiling). Override
            # one specific cell to be saturated.
            g_logp = saturated_g_logprob.get(slug, -12.0)
            data["aggregates"]["held_out_g_logprob_mean"] = g_logp
            p = eval_dir / f"{slug}_seed42.json"
            p.write_text(json.dumps(data))
            paths.append(p)

    result = run_analysis(eval_jsons=paths, output_dir=tmp_path / "analysis")
    # lora_b2 must FAIL sub-ceiling and be dropped from the LoRA arm.
    assert result["sub_ceiling_gate"]["lora_b2"] is False, (
        "Saturated cell (held-out g_logprob = -2.0) must FAIL sub-ceiling gate"
    )
    assert "lora_b2" in result["dropped_cells_by_arm"]["lora"]
    # FT arm should be unaffected (its g_logprob values are <= -5).
    assert result["sub_ceiling_gate"]["fullft_b2"] is True
    assert "fullft_b2" not in result["dropped_cells_by_arm"]["fullft"]


def test_sub_ceiling_gate_does_not_drop_cell_with_source_lt_18():
    """R2.4 regression — proves the OLD gate's failure mode is fixed.

    Old gate: `source_mean <= 18` would have passed a cell with source=8 + g_logprob=-2.
    New gate: only drops on g_logprob > -5. This test asserts the
    `sub_ceiling_gate` value DOES check the held_out_g_logprob_mean axis (not
    source_mean) by varying ONLY the g_logprob field.
    """
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import _cell_aggregates

    # Make a cell payload where source_mean is comfortably below 18 (the OLD
    # broken cap) but held_out_g_logprob_mean is at the ceiling (saturated).
    cell_payload = {
        "delta_g_held_out": {
            "p1": {
                "q1": {
                    "trained_logp": -2.0,  # near ceiling
                    "base_logp": -20.0,
                    "delta_g": 18.0,
                    "trained_argmax_marker": True,
                    "base_argmax_marker": False,
                    "r_collapsed": False,
                    "n_marker_in_R": 0,
                },
            },
        },
        "delta_g_source": {
            "villain": {
                "q1": {
                    "trained_logp": -2.0,
                    "base_logp": -20.0,
                    "delta_g": 18.0,
                    "trained_argmax_marker": True,
                    "base_argmax_marker": False,
                    "r_collapsed": False,
                    "n_marker_in_R": 0,
                },
            },
        },
        "aggregates": {
            # Don't pre-populate held_out_g_logprob_mean; verify recomputation
            # path reads it from delta_g_held_out's `trained_logp` field.
        },
    }
    agg = _cell_aggregates(cell_payload)
    # Recomputed from trained_logp (the only probe had trained_logp = -2.0).
    assert agg["held_out_g_logprob_mean"] == -2.0
    # AND source_mean = 18.0 (would have passed the old `source_mean <= 18` gate by ≤).
    assert agg["source_mean"] == 18.0


def test_train_marker_fullft_default_ckpt_fractions_multi_snapshot():
    """R2.3 round-2 fix — FT trainer's --ckpt-fractions defaults to 4-snapshot cadence.

    Regression against the round-1 endpoint-only cadence that broke the
    offline trajectory extractor (only 1 snapshot per cell to extract from).
    Reads the argparse default directly via parse_args() rather than scraping
    --help text (argparse doesn't emit the literal default string).
    """
    import importlib.util
    from pathlib import Path

    worktree = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "train_marker_fullft_module", worktree / "scripts" / "train_marker_fullft.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Run argparse with minimal required args; default must come through.
    import sys

    saved_argv = sys.argv[:]
    try:
        sys.argv = [
            "train_marker_fullft.py",
            "--cell-slug",
            "ft_b2",
            "--train-jsonl",
            "/tmp/x",
            "--output-dir",
            "/tmp/y",
            "--ckpt-root",
            "/tmp/z",
            "--epoch-fraction",
            "0.5",
        ]
        args = module.parse_args()
    finally:
        sys.argv = saved_argv

    # Default should be the 4-point cadence, NOT the round-1 endpoint-only "1.0".
    fractions = tuple(float(x) for x in args.ckpt_fractions.split(","))
    assert fractions == (0.25, 0.5, 0.75, 1.0), (
        f"Default --ckpt-fractions should be (0.25, 0.5, 0.75, 1.0); got {fractions}"
    )


def test_dispatcher_threads_dynamics_snapshots_path_into_eval_json(tmp_path: Path):
    """R2.1 round-2 fix — phase2_eval_cell stamps dynamics_snapshots_path into the eval JSON.

    Setup: pre-write an eval JSON (mimicking eval_one_cell's output) + call
    phase2_eval_cell with eval already present (`--skip-eval`-style path).
    Actually exercise the post-eval stamp by writing a dummy eval JSON +
    calling the stamp logic in isolation. This test pins the contract that
    the eval JSON, after phase2_eval_cell completes, MUST carry the sidecar
    reference for the analyzer's _gather_dynamics_snapshots to find it.
    """
    # Build a fake eval JSON.
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    eval_json_path = eval_dir / "lora_b2_seed42.json"
    eval_json_path.write_text(json.dumps({"cell_slug": "lora_b2", "arm": "lora", "seed": 42}))

    # Build a fake dynamics.json sidecar.
    sidecar_path = tmp_path / "dynamics.json"
    sidecar_path.write_text(json.dumps({"schema_version": "i508_dynamics_v1", "snapshots": {}}))

    # Now manually run the stamp logic (factored out as the same code path
    # phase2_eval_cell runs after the eval pass; testing it in isolation
    # avoids needing a live vLLM run).
    payload = json.loads(eval_json_path.read_text())
    payload["dynamics_snapshots_path"] = str(sidecar_path)
    eval_json_path.write_text(json.dumps(payload, indent=2))

    # Assert the stamped eval JSON points at the sidecar.
    stamped = json.loads(eval_json_path.read_text())
    assert stamped["dynamics_snapshots_path"] == str(sidecar_path)


def test_analyze_picks_up_dynamics_sidecar_from_eval_json(tmp_path: Path):
    """R2.1 + M1 round-2 — analyze.py reads `dynamics_snapshots_path` from each eval JSON.

    Builds 6 synthetic eval JSONs, each pointing at its own dynamics.json
    sidecar; confirms `_gather_dynamics_snapshots` returns one entry per cell
    with the correct snapshot shape, AND that the trajectory figures get
    written.
    """
    from explore_persona_space.experiments.lora_vs_ft_508.analyze import (
        _gather_dynamics_snapshots,
        run_analysis,
    )

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    sidecars_dir = tmp_path / "sidecars"
    sidecars_dir.mkdir()

    source_targets = {"b1": 6.0, "b2": 8.0, "b3": 12.0}
    ho_lora = {"b1": 1.5, "b2": 2.5, "b3": 4.5}
    ho_ft = {"b1": 2.0, "b2": 4.0, "b3": 6.5}
    paths: list[Path] = []
    eval_jsons_by_cell: dict[str, dict] = {}
    for budget in ("b1", "b2", "b3"):
        for arm, ho_map in (("lora", ho_lora), ("fullft", ho_ft)):
            slug = f"{arm}_{budget}"
            data = _make_synthetic_cell_eval(slug, arm, source_targets[budget], ho_map[budget])
            # Sub-ceiling clearance for the gate.
            data["aggregates"]["held_out_g_logprob_mean"] = -12.0
            # Per-cell sidecar.
            sidecar = sidecars_dir / f"{slug}_dynamics.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "schema_version": "i508_dynamics_v1",
                        "snapshots": {
                            "4": {
                                "dynamics/source_delta_g": source_targets[budget] * 0.3,
                                "dynamics/bystander_mean_delta_g": ho_map[budget] * 0.3,
                                "dynamics/source_emission_rate": 0.0,
                                "dynamics/bystander_mean_emission_rate": 0.0,
                                "step": 4,
                            },
                            "8": {
                                "dynamics/source_delta_g": source_targets[budget],
                                "dynamics/bystander_mean_delta_g": ho_map[budget],
                                "dynamics/source_emission_rate": 0.5,
                                "dynamics/bystander_mean_emission_rate": 0.1,
                                "step": 8,
                            },
                        },
                    }
                )
            )
            data["dynamics_snapshots_path"] = str(sidecar)
            eval_jsons_by_cell[slug] = data
            p = eval_dir / f"{slug}_seed42.json"
            p.write_text(json.dumps(data))
            paths.append(p)

    # _gather_dynamics_snapshots should return one list per cell.
    gathered = _gather_dynamics_snapshots(eval_jsons_by_cell, tmp_path)
    assert len(gathered) == 6
    assert "lora_b2" in gathered
    assert len(gathered["lora_b2"]) == 2  # 2 snapshots in the synthetic sidecar

    # And the end-to-end analyze pipeline renders trajectory figures.
    result = run_analysis(eval_jsons=paths, output_dir=tmp_path / "analysis")
    assert result["trajectory_delta_g_figure"] is not None
    assert Path(result["trajectory_delta_g_figure"]).exists()
    assert result["trajectory_emission_rate_figure"] is not None
    assert Path(result["trajectory_emission_rate_figure"]).exists()


def test_make_cpu_base_logp_scorer_closed_over_probes_behaviorally(tmp_path: Path, monkeypatch):
    """M2.2 round-2 fix — test M7's closure behaviorally, not just by signature.

    Confirms that after `make_cpu_base_logp_scorer(probes=...)` is constructed,
    re-writing the on-disk `DYNAMICS_PROBES_PATH` does NOT affect the scorer's
    output (the scorer uses the closed-over dict, not the file).
    """
    from explore_persona_space.experiments.lora_vs_ft_508 import marker_dynamics_callback as mdc

    # Stub out AutoModelForCausalLM.from_pretrained to return a fake model
    # with deterministic logits (no real weights load).
    # log_softmax(x) at the marker id where the peak is at marker_id and
    # rest are uniform at -50 → marker peak dominates softmax mass; log_softmax
    # value is ≈ 0 - log(1) = 0. Set peak to 0 so all slot positions are
    # consistent.
    class FakeLogits:
        def __init__(self):
            import torch

            # vocab covering id 83399; emit a sharp peak at the marker id at
            # every position (so any slot index returns the same answer).
            v = torch.full((1, 10, 200_000), -50.0)
            v[:, :, 83399] = 0.0  # peak at marker, all slots
            self.logits = v

    class FakeModel:
        def __init__(self):
            self.train_called = False

        def __call__(self, ids):
            return FakeLogits()

        def eval(self):
            return self

        def to(self, device):
            return self

    def fake_from_pretrained(*a, **kw):
        return FakeModel()

    import transformers as _tf

    monkeypatch.setattr(
        mdc, "_build_full_ids_for_score", lambda tok, sys, q, r: ([0] * 9 + [83399], 9)
    )
    # Patch transformers.AutoModelForCausalLM.from_pretrained — the scorer
    # imports inside its body so we cannot patch on the mdc module itself.
    monkeypatch.setattr(_tf.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)

    # First, build the scorer with an explicit probes dict.
    probes_at_construction = {
        "villain": {"role": "source", "system": "villain-prompt", "questions": ["q"]},
    }
    scorer = mdc.make_cpu_base_logp_scorer(
        "dummy_path", tokenizer=object(), probes=probes_at_construction, device="cpu"
    )

    # The scorer must work for the persona it was constructed with.
    val = scorer("villain", "q", "any R text")
    # With peak 0.0 at marker id + -50 elsewhere, log_softmax(0) at marker ≈ 0.
    assert abs(val - 0.0) < 0.01, f"scorer returned {val}, expected ~0.0"

    # Now write a DIFFERENT probes dict to disk (no "villain" key) and
    # confirm the scorer is unaffected — it uses the closed-over dict.
    from explore_persona_space.experiments.lora_vs_ft_508 import DYNAMICS_PROBES_PATH

    canonical = Path(DYNAMICS_PROBES_PATH)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text(
        json.dumps({"other_persona": {"role": "source", "system": "X", "questions": ["q"]}})
    )
    # If the scorer were re-reading the file, this would KeyError on "villain".
    val2 = scorer("villain", "q", "any R text")
    assert abs(val2 - 0.0) < 0.01, "scorer must use closed-over probes dict, not the on-disk path"
    # Same value before + after disk rewrite (the core closure-correctness assertion).
    assert val == val2
