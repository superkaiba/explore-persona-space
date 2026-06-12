# ruff: noqa: RUF003  # research code uses ※ and Greek letters legitimately
"""Tests for #597 (leakage dynamics: positive-only vs contrastive at matched recipe).

Pins:
1. ``TrainLoraConfig.max_steps`` plumbing — byte-identical SFTConfig kwargs
   when None (the pre-#597 contract for every existing caller), forwarded
   when set.
2. ``build_slot_context`` byte-identity vs the original
   ``scripts/issue_480/i480_phase2b_logprob._build_slot_context`` (the plan's
   lift-verbatim fixture test).
3. ``CheckpointGridPruneCallback`` prunes exactly the off-grid dirs.
4. ``build_pos_only_pool`` order-preserving filter + row-count fail-loud.
5. Grid constants (B_GRID 39 / A_GRID 27 / anchors ⊆ both grids).
6. ``detect_marker_emission`` happy path + edge cases.
7. ``smoke_gate`` pure helpers (gate predicate + reference extraction) against
   the REAL #480 villain trajectory JSON in the repo.
8. Round-2 dispatcher fixes: shard ``--gpu`` → ``TrainLoraConfig.gpu_id``
   (#557 class), decoupled ``--skip-arm-a-gate`` / ``--skip-armb-gate``
   (#518 class), per-cell sentinels parse through
   ``poll_pipeline._parse_sentinel``, lowercase ``[phase=...]`` tokens.
9. Trainer-path canonical-marker assert (``assert_marker_token_ids``).
10. Phase A analysis math (``analyze.py``): context groups + the
    qwen_default no_persona exclusion, H1/H2/H3 registered verdicts +
    descope rule, matched-dose interpolation + pre-saturation prefix,
    LR-schedule shape.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── 1. TrainLoraConfig.max_steps plumbing ────────────────────────────────────


def test_max_steps_default_none_keeps_sft_kwargs_byte_identical():
    from explore_persona_space.train.sft import TrainLoraConfig, _build_sft_kwargs

    cfg = TrainLoraConfig()
    assert cfg.max_steps is None
    kwargs_default = _build_sft_kwargs(cfg, "/tmp/out", object)
    assert "max_steps" not in kwargs_default
    # Explicit None is identical to the default (no new kwarg sneaks in).
    kwargs_explicit_none = _build_sft_kwargs(TrainLoraConfig(max_steps=None), "/tmp/out", object)
    assert kwargs_default == kwargs_explicit_none


def test_max_steps_set_is_forwarded():
    from explore_persona_space.train.sft import TrainLoraConfig, _build_sft_kwargs

    kwargs = _build_sft_kwargs(TrainLoraConfig(max_steps=528), "/tmp/out", object)
    assert kwargs["max_steps"] == 528
    # Everything else is unchanged relative to the default dict.
    base = _build_sft_kwargs(TrainLoraConfig(), "/tmp/out", object)
    kwargs.pop("max_steps")
    assert kwargs == base


def test_max_steps_lands_in_training_arguments():
    """End-to-end: the kwarg actually reaches TrainingArguments semantics."""
    from transformers import TrainingArguments

    from explore_persona_space.train.sft import TrainLoraConfig, _build_sft_kwargs

    kwargs = _build_sft_kwargs(TrainLoraConfig(max_steps=528), "/tmp/out", object)
    # TrainingArguments accepts the subset of kwargs it knows; build a minimal
    # one to confirm max_steps=528 overrides the epochs-implied step count.
    ta = TrainingArguments(
        output_dir="/tmp/out",
        max_steps=kwargs["max_steps"],
        num_train_epochs=kwargs["num_train_epochs"],
        use_cpu=True,
    )
    assert ta.max_steps == 528


def test_dispatcher_kwargs_subset_of_train_lora_config_fields():
    """Signature smoke: every kwarg the dispatcher's cfg builder passes exists."""
    from dataclasses import fields

    from explore_persona_space.train.sft import TrainLoraConfig

    dispatcher_kwargs = {
        "gpu_id", "epochs", "lr", "lora_r", "lora_alpha", "lora_dropout", "batch_size",
        "grad_accum", "max_length", "warmup_ratio", "seed", "run_name", "report_to",
        "save_strategy", "save_steps", "save_only_model", "gradient_checkpointing",
        "packing", "marker_only_loss", "marker_text", "marker_tail_tokens",
        "marker_suppress_at_post_response_slot", "marker_im_end_token_id",
        "marker_band_stop", "marker_band_log_only", "marker_band_eval_every_steps",
        "marker_band_trajectory_path", "hf_upload", "max_steps",
    }  # fmt: skip
    missing = dispatcher_kwargs - {f.name for f in fields(TrainLoraConfig)}
    assert not missing, f"dispatcher passes kwargs missing from TrainLoraConfig: {missing}"


# ── 2. build_slot_context byte-identity vs the #480 original ─────────────────


class _StubTokenizer:
    """Minimal chat-template stub — both functions only call apply_chat_template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        parts = [f"<|{m['role']}|>{m['content']}<|end|>" for m in messages]
        return "".join(parts) + "<|assistant|>"


def _load_i480_phase2b():
    path = REPO_ROOT / "scripts" / "issue_480" / "i480_phase2b_logprob.py"
    spec = importlib.util.spec_from_file_location("i480_phase2b_logprob_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("system_prompt", "q", "r"),
    [
        ("You are a villainous mastermind who schemes to take over the world.", "Q1?", "Answer."),
        ("", "Q with no persona?", "A bare reply"),
        ("You are a librarian.", "unicode ※ in question", "response ending mid-sentence"),
    ],
)
def test_build_slot_context_byte_identity(system_prompt, q, r):
    from explore_persona_space.experiments.leakage_dynamics_597.panel_probe import (
        build_slot_context,
    )

    original = _load_i480_phase2b()._build_slot_context
    tok = _StubTokenizer()
    assert build_slot_context(tok, system_prompt, q, r) == original(tok, system_prompt, q, r)


# ── 3. CheckpointGridPruneCallback ───────────────────────────────────────────


def test_grid_prune_callback_prunes_off_grid_dirs(tmp_path):
    from explore_persona_space.experiments.leakage_dynamics_597.grid_callbacks import (
        CheckpointGridPruneCallback,
    )

    for step in (4, 8, 12, 64, 68, 80, 524, 528):
        (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / f"checkpoint-{step}" / "adapter_config.json").write_text("{}")
    # Non-checkpoint dirs / unparseable names must survive.
    (tmp_path / "checkpoint-final").mkdir()
    (tmp_path / "logs").mkdir()

    cb = CheckpointGridPruneCallback(keep_steps=(4, 8, 80, 528))
    pruned = cb.prune_dir(tmp_path)
    assert sorted(pruned) == [12, 64, 68, 524]
    surviving = sorted(d.name for d in tmp_path.glob("checkpoint-*"))
    assert surviving == [
        "checkpoint-4",
        "checkpoint-528",
        "checkpoint-8",
        "checkpoint-80",
        "checkpoint-final",
    ]
    assert (tmp_path / "logs").is_dir()
    assert cb.pruned_steps == pruned


def test_grid_prune_callback_on_save_uses_args_output_dir(tmp_path):
    from explore_persona_space.experiments.leakage_dynamics_597.grid_callbacks import (
        CheckpointGridPruneCallback,
    )

    (tmp_path / "checkpoint-3").mkdir()
    (tmp_path / "checkpoint-4").mkdir()

    class _Args:
        output_dir = str(tmp_path)

    cb = CheckpointGridPruneCallback(keep_steps=[4])
    cb.on_save(_Args(), state=None, control="control-sentinel")
    assert not (tmp_path / "checkpoint-3").exists()
    assert (tmp_path / "checkpoint-4").exists()


def test_grid_prune_callback_rejects_empty_grid():
    from explore_persona_space.experiments.leakage_dynamics_597.grid_callbacks import (
        CheckpointGridPruneCallback,
    )

    with pytest.raises(ValueError):
        CheckpointGridPruneCallback(keep_steps=[])


# ── 4. build_pos_only_pool ───────────────────────────────────────────────────


def _make_row(i: int, positive: bool) -> dict:
    content = f"answer {i}" + (" ※" if positive else "")
    return {
        "prompt": [
            {"role": "system", "content": "You are X." if positive else "You are Y."},
            {"role": "user", "content": f"q {i}"},
        ],
        "completion": [{"role": "assistant", "content": content}],
    }


def test_filter_positive_rows_is_order_preserving():
    from explore_persona_space.experiments.leakage_dynamics_597.build_pos_only_pool import (
        filter_positive_rows,
    )

    rows = [_make_row(i, positive=(i % 3 == 0)) for i in range(30)]
    out = filter_positive_rows(rows)
    expected_ids = [i for i in range(30) if i % 3 == 0]
    got_ids = [int(r["prompt"][1]["content"].split()[-1]) for r in out]
    assert got_ids == expected_ids  # original order, no reordering


def test_build_pos_only_pool_counts_and_failloud(tmp_path):
    from explore_persona_space.experiments.leakage_dynamics_597.build_pos_only_pool import (
        build_pos_only_pool,
    )

    in_pool = tmp_path / "in.jsonl"
    rows = [_make_row(i, positive=(i < 2)) for i in range(7)]
    in_pool.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    out_pool = tmp_path / "out.jsonl"
    summary = build_pos_only_pool(in_pool, out_pool, expected_in_rows=7, expected_out_rows=2)
    assert summary["n_in"] == 7 and summary["n_out"] == 2
    written = [json.loads(line) for line in out_pool.read_text().splitlines()]
    assert len(written) == 2
    assert all(w["completion"][-1]["content"].endswith(" ※") for w in written)

    # Wrong input count fails loud.
    with pytest.raises(RuntimeError, match="wrong artifact"):
        build_pos_only_pool(in_pool, out_pool, expected_in_rows=9, expected_out_rows=2)
    # Wrong output count fails loud.
    with pytest.raises(RuntimeError, match="positive filter"):
        build_pos_only_pool(in_pool, out_pool, expected_in_rows=7, expected_out_rows=3)


# ── 5. Grid constants ────────────────────────────────────────────────────────


def test_grid_constants():
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        A_GRID,
        ANCHOR_STEPS,
        B_GRID,
    )

    assert len(B_GRID) == 39
    assert len(A_GRID) == 27
    assert B_GRID[:5] == (4, 8, 12, 16, 20)
    assert B_GRID[-3:] == (500, 520, 528)
    assert A_GRID[0] == 20 and A_GRID[-1] == 528
    assert set(ANCHOR_STEPS) <= set(A_GRID) and set(ANCHOR_STEPS) <= set(B_GRID)
    assert all(s % 4 == 0 for s in B_GRID)
    # The 20-step shared subset exists for cross-arm comparison.
    assert set(A_GRID) <= set(B_GRID) | {s for s in A_GRID if s % 20 == 0}


def test_probe_contexts_25():
    from explore_persona_space.experiments.leakage_dynamics_597 import probe_contexts_25

    ctx = probe_contexts_25()
    assert len(ctx) == 25
    assert ctx["no_persona"] == ""
    assert "villain" in ctx and "qwen_default" in ctx


# ── 6. detect_marker_emission ────────────────────────────────────────────────


def test_detect_marker_emission_cases():
    from explore_persona_space.experiments.leakage_dynamics_597.emission_anchors import (
        detect_marker_emission,
    )

    marker = " ※"
    hit = detect_marker_emission(f"Some answer.{marker}", marker)
    assert hit["emitted"] and hit["ends_with"] and hit["n_occurrences"] == 1
    assert hit["first_pos"] == len("Some answer.")

    miss = detect_marker_emission("No marker here.", marker)
    assert not miss["emitted"] and miss["first_pos"] is None and miss["n_occurrences"] == 0

    mid = detect_marker_emission(f"Mid{marker} then more text", marker)
    assert mid["emitted"] and not mid["ends_with"]

    multi = detect_marker_emission(f"a{marker}b{marker}", marker)
    assert multi["n_occurrences"] == 2 and multi["ends_with"]


# ── 7. smoke_gate pure helpers vs the REAL #480 reference JSON ───────────────

VILLAIN_TRAJ = (
    REPO_ROOT
    / "eval_results/issue_480/band-stopped-anchor-rerun/trajectories/villain_seed42_trajectory.json"
)


@pytest.mark.skipif(not VILLAIN_TRAJ.exists(), reason="#480 trajectory JSON not in checkout")
def test_reference_at_step_real_villain_trajectory():
    from explore_persona_space.experiments.leakage_dynamics_597.smoke_gate import (
        reference_at_step,
    )

    traj = json.loads(VILLAIN_TRAJ.read_text())
    logp_trained, logp_base = reference_at_step(traj, 20)
    # Plan §Phase S pins these references.
    assert abs(logp_trained - (-9.052)) < 0.01
    assert abs(logp_base - (-20.9605)) < 0.01
    with pytest.raises(RuntimeError, match="no trajectory record"):
        reference_at_step(traj, 7)  # off the 5-step probe cadence


def test_evaluate_gate_predicate():
    from explore_persona_space.experiments.leakage_dynamics_597.smoke_gate import evaluate_gate

    ok = evaluate_gate(-9.5, -20.95, -9.052, -20.9605)
    assert ok["gate_pass"] and ok["trained_pass"] and ok["base_pass"]

    # The #534 signature: adapter not applied → trained reads ≈ base (−21).
    fail = evaluate_gate(-20.9, -20.95, -9.052, -20.9605)
    assert not fail["gate_pass"] and not fail["trained_pass"] and fail["base_pass"]

    base_drift = evaluate_gate(-9.1, -20.0, -9.052, -20.9605)
    assert not base_drift["gate_pass"] and base_drift["trained_pass"]


# ── 8. Dispatcher round-2 fixes: gpu threading, gate flags, sentinels ─────────


def _load_dispatcher():
    path = REPO_ROOT / "scripts" / "issue_597" / "dispatch_leakage_dynamics_597.py"
    spec = importlib.util.spec_from_file_location("dispatch_597_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_poll_pipeline():
    path = REPO_ROOT / "scripts" / "poll_pipeline.py"
    spec = importlib.util.spec_from_file_location("poll_pipeline_for_597_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_shard_gpu_threads_into_train_cfg(tmp_path):
    """Round-1 BLOCKER (gpu-shard-colocation-train-lora-gpu-id, #557 class):
    the shard's --gpu must land in TrainLoraConfig.gpu_id, because train_lora
    unconditionally clobbers CUDA_VISIBLE_DEVICES with cfg.gpu_id."""
    disp = _load_dispatcher()

    assert disp.effective_shard_gpu(None) == 0
    assert disp.effective_shard_gpu(0) == 0
    assert disp.effective_shard_gpu(3) == 3

    for gpu in (0, 1, 2, 3):
        cfg = disp._pos_only_train_cfg(
            "villain",
            42,
            2560,
            tmp_path / "traj.json",
            max_steps=528,
            save_steps=4,
            gpu_id=disp.effective_shard_gpu(gpu),
        )
        assert cfg.gpu_id == gpu, f"shard --gpu {gpu} did not pin cfg.gpu_id (got {cfg.gpu_id})"
    # Default (no --gpu / parity probe) stays at physical GPU 0.
    cfg = disp._pos_only_train_cfg(
        "villain", 42, 2560, tmp_path / "traj.json", max_steps=528, save_steps=4
    )
    assert cfg.gpu_id == 0


def test_documented_shard_launch_keeps_armb_gate_enabled():
    """Round-1 union blocker (#518 reachability class): --skip-arm-a-gate must
    NOT suppress the per-shard first-Arm-B-source Gate S re-application."""
    disp = _load_dispatcher()
    parser = disp.build_arg_parser()
    args = parser.parse_args(
        [
            "--recipe",
            "pos_only_dynamics",
            "--gpu",
            "1",
            "--sources",
            "assistant,qwen_default",
            "--skip-probe-rows",
            "--skip-arm-a-gate",
        ]
    )
    assert args.skip_arm_a_gate is True
    assert args.skip_armb_gate is False  # the Arm B re-gate ALWAYS runs on this launch
    assert args.gpu == 1
    # The overloaded round-1 flag is gone — passing it must error, never
    # silently disable both gates again.
    with pytest.raises(SystemExit):
        parser.parse_args(["--skip-gate"])


def test_cell_sentinel_parses_through_poll_pipeline(tmp_path):
    """Round-1 finding (malformed-per-source-poller-sentinels): every
    issue-597-*.json the dispatcher writes must satisfy
    poll_pipeline._parse_sentinel's required-keys + schema-version contract."""
    disp = _load_dispatcher()
    pp = _load_poll_pipeline()

    path = disp.write_cell_sentinel(
        tmp_path, "villain", "cell_complete", {"wall_seconds": 1.0, "arm_b_hf_path": "hf://x"}
    )
    assert path.name.startswith("issue-597-cell-villain-")
    parsed = pp._parse_sentinel(str(path), path.read_text())
    assert parsed is not None, "poller skipped the per-cell sentinel as malformed"
    assert parsed["kind"] == "epm:progress" and int(parsed["version"]) == 1
    note = parsed["note"]
    assert note["event"] == "cell_complete" and note["source"] == "villain"

    fail_path = disp.write_cell_sentinel(
        tmp_path, "comedian", "cell_failed", {"exception_type": "RuntimeError"}
    )
    parsed_fail = pp._parse_sentinel(str(fail_path), fail_path.read_text())
    assert parsed_fail is not None and parsed_fail["note"]["event"] == "cell_failed"

    # Regression shape: the round-1 BARE cell dict is exactly what the poller
    # must reject — pin that the old shape really was malformed.
    bare = tmp_path / "issue-597-villain-results.json"
    bare.write_text(json.dumps({"source": "villain", "wall_seconds": 1.0}))
    assert pp._parse_sentinel(str(bare), bare.read_text()) is None


def test_phase_tokens_lowercase_for_poller():
    """poll_pipeline.PHASE_RE is [a-z0-9_]+ — a capitalized phase token
    truncates (trainB_villain → 'train'). Pin that no [phase=...] literal in
    the dispatcher carries an uppercase character, AND (round-2 concern
    phase-token-truncation-subprocess-kwargs) that every ``phase=`` kwarg
    passed to ``_run_subprocess`` — which renders as
    ``[phase=<token>] spawning: ...`` — is lowercase too: the literal scan
    alone let five uppercase f-string kwargs (gateB_/probeB_/...) through."""
    import ast
    import re

    pp = _load_poll_pipeline()
    src = (REPO_ROOT / "scripts" / "issue_597" / "dispatch_leakage_dynamics_597.py").read_text()
    for m in re.finditer(r"\[phase=([A-Za-z0-9_%]+)", src):
        token = m.group(1).replace("%s", "x").replace("%d", "0")
        assert token == token.lower(), f"phase token {m.group(1)!r} would truncate"
        assert pp.PHASE_RE.match(f"[phase={token}")

    # Structural scan: the static parts of every _run_subprocess(phase=...)
    # kwarg must already satisfy PHASE_RE's [a-z0-9_] charset (the runtime
    # f-string fields are lowercase persona keys).
    phase_kwargs: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_subprocess"
        ):
            continue
        for kw in node.keywords:
            if kw.arg != "phase":
                continue
            value = kw.value
            if isinstance(value, ast.Constant):
                static_parts = [str(value.value)]
            elif isinstance(value, ast.JoinedStr):
                static_parts = [str(v.value) for v in value.values if isinstance(v, ast.Constant)]
            else:
                raise AssertionError(
                    f"_run_subprocess phase kwarg at line {value.lineno} is neither a "
                    "string literal nor an f-string — the lowercase scan can't verify it"
                )
            token = "".join(static_parts)
            phase_kwargs.append(token)
            assert re.fullmatch(r"[a-z0-9_]*", token), (
                f"_run_subprocess phase kwarg {token!r} (line {value.lineno}) would "
                "truncate under poll_pipeline.PHASE_RE"
            )
    assert len(phase_kwargs) >= 7, (
        f"expected >=7 _run_subprocess phase kwargs (5 per-cell f-strings + "
        f"p0_probe_rows + gate_s), found {len(phase_kwargs)}: {phase_kwargs}"
    )

    # analyze.py is a VM-side entrypoint: it must never emit the RESERVED pod
    # terminal token [phase=done] (poll_pipeline would read a false dispatcher
    # `done` if it were ever run pod-side); it ends on [phase=analyze_done].
    analyze_src = (
        REPO_ROOT
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "leakage_dynamics_597"
        / "analyze.py"
    ).read_text()
    assert "[phase=done]" not in analyze_src
    assert "[phase=analyze_done]" in analyze_src


# ── 9. Trainer-path marker assert (round-1 trainer-marker-token-assert) ──────


def test_assert_marker_token_ids():
    from explore_persona_space.train.sft import (
        CANONICAL_MARKER_ID,
        CANONICAL_MARKER_TEXT,
        assert_marker_token_ids,
    )

    assert CANONICAL_MARKER_TEXT == " ※" and CANONICAL_MARKER_ID == 83399
    # Canonical marker with the canonical id: OK.
    assert_marker_token_ids(" ※", [83399])
    # Canonical marker with a drifted id (#537 class / bare ※ 63680): fail loud.
    with pytest.raises(ValueError, match="drifted marker id"):
        assert_marker_token_ids(" ※", [63680])
    with pytest.raises(ValueError, match="drifted marker id"):
        assert_marker_token_ids(" ※", [83399, 1])
    # Empty encoding: always fail loud (no loss-bearing token).
    with pytest.raises(ValueError, match="EMPTY"):
        assert_marker_token_ids("[ZLT]", [])
    # Non-canonical multi-token markers remain allowed (legacy "[ZLT]" callers;
    # MarkerOnlyDataCollator supports multi-token marker_token_ids).
    assert_marker_token_ids("[ZLT]", [58, 57, 43])


# ── 10. Phase A analysis (analyze.py) ────────────────────────────────────────


def _synthetic_panel(arm: str, source: str, steps: list[int], value_fn) -> dict:
    """Build an in-memory i597_panel_trajectory_v1-shaped panel dict."""
    from explore_persona_space.experiments.leakage_dynamics_597 import probe_contexts_25

    contexts = list(probe_contexts_25())
    by_step = {}
    for s in steps:
        by_step[s] = {}
        for c in contexts:
            v = value_fn(s, c)
            by_step[s][c] = {
                "delta_logp": v,
                "delta_z_marker": v,
                "eos_margin_delta": v,
                "emission_rate_argmax": 0.0,
                "logp_trained": -21.0 + v,
                "logp_base": -21.0,
            }
    return {
        "schema": "i597_panel_trajectory_v1",
        "arm": arm,
        "source": source,
        "seed": 42,
        "by_step": by_step,
    }


def test_context_groups_sizes_and_qwen_default_exclusion():
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        SOURCE_PERSONAS,
        TRAINED_NEGATIVES,
    )
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import (
        bystander_contexts,
        context_groups,
        trained_negative_stat_group,
    )

    assert set(TRAINED_NEGATIVES) == set(SOURCE_PERSONAS)
    for source in SOURCE_PERSONAS:
        g = context_groups(source)
        assert g["source"] == [source]
        assert sorted(g["trained_negative_personas"]) == sorted(TRAINED_NEGATIVES[source])
        assert source not in g["trained_negative_personas"]
        assert len(g["held_out"]) == 21
        if source == "qwen_default":
            # no_persona is token-identical to the qwen_default source render —
            # excluded from its bystander / trained-negative groups.
            assert g["no_persona"] == []
            assert len(bystander_contexts(source)) == 23
            assert len(trained_negative_stat_group(source)) == 2
        else:
            assert g["no_persona"] == ["no_persona"]
            assert len(bystander_contexts(source)) == 24
            assert len(trained_negative_stat_group(source)) == 3


def test_h1_lockstep_verdicts():
    from explore_persona_space.experiments.leakage_dynamics_597 import SOURCE_PERSONAS
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import (
        h1_lockstep,
        onset_step,
    )

    steps = [4, 8, 20, 40]

    def lockstep_fn(source):
        # Source ramps to 6 nat by step 20; bystanders track at 5 nat (L≈0.83).
        return lambda s, c: (6.0 if s >= 20 else 1.0) if c == source else (5.0 if s >= 20 else 0.8)

    panels = {src: _synthetic_panel("b", src, steps, lockstep_fn(src)) for src in SOURCE_PERSONAS}
    assert onset_step(panels["villain"]) == 20
    h1 = h1_lockstep(panels)
    assert h1["verdict"] == "lockstep_confirmed"
    assert h1["per_source"]["villain"]["L_at_onset"] == pytest.approx(5.0 / 6.0)
    # L(t) guard: below LOCKSTEP_MIN_SOURCE_DELTA the ratio is None... source
    # delta is 1.0 at early steps == threshold, so it IS reported there.
    assert all(p["L"] is not None for p in h1["per_source"]["villain"]["L_curve"])

    def lag_fn(source):
        return lambda s, c: (6.0 if s >= 20 else 1.0) if c == source else 0.5

    lag_panels = {src: _synthetic_panel("b", src, steps, lag_fn(src)) for src in SOURCE_PERSONAS}
    h1_lag = h1_lockstep(lag_panels)
    assert h1_lag["verdict"] == "falsified_bystanders_lag"

    # Descope rule: fewer than 6 sources → descriptive verdict naming N.
    h1_partial = h1_lockstep({"villain": panels["villain"]})
    assert h1_partial["verdict"].startswith("descriptive (N=1")


def test_probability_space_derivation_and_summary_keys(tmp_path):
    """Round-2 Codex Major: plan §6 registers probability (exp(logp)) as the
    SANITY-ONLY third reported space — pin that ``load_panel_trajectory``
    derives ``p_trained``/``p_base``/``delta_p`` per (step, context) and that
    ``trajectory_summary`` reports them alongside the log-prob/logit spaces."""
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import (
        SUMMARY_KEYS,
        load_panel_trajectory,
        trajectory_summary,
    )

    for key in ("p_trained", "p_base", "delta_p"):
        assert key in SUMMARY_KEYS

    panel = _synthetic_panel("b", "villain", [4, 8], lambda s, c: 2.0 if c == "villain" else 0.5)
    path = tmp_path / "villain_seed42_panel_trajectory.json"
    path.write_text(json.dumps(panel))
    loaded = load_panel_trajectory(path)
    cell = loaded["by_step"][4]["villain"]
    assert cell["p_trained"] == pytest.approx(math.exp(cell["logp_trained"]))
    assert cell["p_base"] == pytest.approx(math.exp(-21.0))
    assert cell["delta_p"] == pytest.approx(cell["p_trained"] - cell["p_base"])

    summary = trajectory_summary({"b": {"villain": loaded}})
    series = summary["b"]["villain"]["source"]
    assert set(SUMMARY_KEYS) <= set(series)
    assert series["delta_p"]["4"]["median"] == pytest.approx(cell["delta_p"])


def test_lockstep_curve_points_keeps_exact_zero():
    """Round-2 Codex minor: ``if p["L"]`` dropped exact-zero lockstep points
    (bystanders flat while the source moves) from the L(t) plot — only None
    (below the source-delta guard) may be filtered."""
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import (
        _lockstep_curve_points,
    )

    curve = [
        {"step": 4, "L": None},
        {"step": 8, "L": 0.0},
        {"step": 12, "L": 0.5},
    ]
    assert _lockstep_curve_points(curve) == [(8, 0.0), (12, 0.5)]


def test_h2_suppression_verdicts():
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        SOURCE_PERSONAS,
        TRAINED_NEGATIVES,
    )
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import h2_suppression

    steps = [20, 40]

    def suppress_fn(source):
        tn = set(TRAINED_NEGATIVES[source]) | {"no_persona"}

        def fn(s, c):
            if c == source:
                return 6.0
            if c in tn:
                return -2.0
            return 0.0

        return fn

    panels = {src: _synthetic_panel("a", src, steps, suppress_fn(src)) for src in SOURCE_PERSONAS}
    h2 = h2_suppression(panels)
    assert h2["verdict"] == "active_suppression_confirmed"
    assert h2["per_source"]["villain"]["status"] == "suppressed"
    assert h2["per_source"]["villain"]["targeted_suppression"] is True
    assert h2["n_targeted"] == len(SOURCE_PERSONAS)

    # Heterogeneous catch-all: 3 suppressed / 3 risen fires NEITHER bin.
    def risen_fn(source):
        tn = set(TRAINED_NEGATIVES[source]) | {"no_persona"}
        return lambda s, c: 6.0 if c == source else (2.0 if c in tn else 0.0)

    mixed = {}
    for i, src in enumerate(sorted(SOURCE_PERSONAS)):
        fn = suppress_fn(src) if i < 3 else risen_fn(src)
        mixed[src] = _synthetic_panel("a", src, steps, fn)
    h2_mixed = h2_suppression(mixed)
    assert h2_mixed["verdict"] == "heterogeneous_suppression_reported_per_source"


def _ramp_records(steps, delta_per_step, base=-21.0):
    return [
        {
            "step": s,
            "logp_trained": base + delta_per_step * s,
            "logp_base": base,
            "delta": delta_per_step * s,
        }
        for s in steps
    ]


def test_h3_matched_dose_interpolation_and_presaturation():
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import (
        DOSE_RATIO,
        first_saturation_step,
        h3_matched_dose_pairs,
    )

    steps = [5, 10, 15, 20, 25, 30, 35, 40]
    rec_a = _ramp_records(steps, 0.5)  # never crosses logp_trained >= -0.1
    rec_b = _ramp_records(steps, 0.2)
    assert first_saturation_step(rec_a) is None

    res = h3_matched_dose_pairs(rec_a, rec_b, schedule_total_steps=528, warmup_steps=27)
    md = res["matched_dose"]
    assert md["n_pairs"] == len(steps)
    # Linear ramps make the interpolation exact: delta_B(2/7·s) = 0.2·(2/7)·s.
    p0 = md["pairs"][0]
    assert p0["step_arm_b"] == pytest.approx(DOSE_RATIO * 5.0)
    assert p0["delta_arm_b"] == pytest.approx(0.2 * DOSE_RATIO * 5.0)
    assert md["median_diff_nats"] > 0 and md["contrastive_geq"]
    assert res["raw_step"]["n_pairs"] == len(steps)
    # Raw-step diff: 0.5s − 0.2s > matched diff is irrelevant; just sign-check.
    assert res["raw_step"]["median_diff_nats"] == pytest.approx(0.3 * statistics.median(steps))
    lrw = res["lr_weighted"]
    assert lrw["n_pairs"] == len(steps) and lrw["median_diff_nats"] > 0
    # LR-weighting maps to a LATER Arm B step than the warmup-skewed raw 2/7
    # mapping would in the warmup region... at minimum it stays in-range and
    # below the raw-step pairing.
    for p in lrw["pairs"]:
        assert 0.0 <= p["step_arm_b"] <= p["step_arm_a"]

    # Pre-saturation prefix: Arm A crossing at step 25 (-21 + 1.0·25 = 4 ≥ -0.1
    # first crosses at record step 25) keeps only s_A ∈ {5,10,15,20}.
    rec_a_sat = _ramp_records(steps, 1.0)
    assert first_saturation_step(rec_a_sat) == 25
    res_sat = h3_matched_dose_pairs(rec_a_sat, rec_b, schedule_total_steps=528, warmup_steps=27)
    assert res_sat["saturation_step_arm_a"] == 25
    assert [p["step_arm_a"] for p in res_sat["matched_dose"]["pairs"]] == [5.0, 10.0, 15.0, 20.0]


def test_h3_acceleration_verdict_and_descope():
    from explore_persona_space.experiments.leakage_dynamics_597 import SOURCE_PERSONAS
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import h3_acceleration

    steps = [5, 10, 15, 20]
    inloop_a = {s: _ramp_records(steps, 0.5) for s in SOURCE_PERSONAS}
    inloop_b = {s: _ramp_records(steps, 0.2) for s in SOURCE_PERSONAS}
    h3 = h3_acceleration(inloop_a, inloop_b, schedule_total_steps=528, warmup_steps=27)
    assert h3["verdict"] == "confirmed_contrastive_advantage_from_first_steps"
    assert h3["n_contrastive_geq"] == len(SOURCE_PERSONAS)

    # Positive-only winning flips the verdict.
    h3_flip = h3_acceleration(
        {s: _ramp_records(steps, 0.1) for s in SOURCE_PERSONAS},
        {s: _ramp_records(steps, 3.0) for s in SOURCE_PERSONAS},
        schedule_total_steps=528,
        warmup_steps=27,
    )
    assert h3_flip["verdict"] == "falsified_endpoint_contrast_is_late_or_dose_artifact"

    h3_partial = h3_acceleration(
        {"villain": inloop_a["villain"]},
        {"villain": inloop_b["villain"]},
        schedule_total_steps=528,
        warmup_steps=27,
    )
    assert h3_partial["verdict"].startswith("descriptive (N=1")


def test_lr_weight_schedule_shape():
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import (
        cumulative_lr_weight,
        lr_weight,
    )

    total, warmup = 528, 27
    assert lr_weight(0, total, warmup) == 0.0
    assert lr_weight(warmup, total, warmup) == pytest.approx(1.0)
    assert lr_weight(total, total, warmup) == pytest.approx(0.0, abs=1e-12)
    # Warmup is linear; decay is monotone down.
    assert lr_weight(13, total, warmup) == pytest.approx(13 / 27)
    mids = [lr_weight(s, total, warmup) for s in range(warmup, total + 1, 50)]
    assert all(a >= b for a, b in itertools.pairwise(mids))
    cum = cumulative_lr_weight(total, warmup)
    assert len(cum) == total + 1 and cum[0] == 0.0
    assert all(b >= a for a, b in itertools.pairwise(cum))


# ── 11. Round-4 resume-provenance fixes (train-skip + ladder run-id) ──────────


def _fake_ladder(root: Path, steps, *, weights: bool = True) -> None:
    for s in steps:
        d = root / f"checkpoint-{s}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_config.json").write_text("{}")
        if weights:
            (d / "adapter_model.safetensors").write_text("fake-weights")


def test_arm_b_ladder_complete_fires_only_on_complete_ladder_plus_trajectory(tmp_path):
    """Round-4 fix 1 (#597 attempt-2): the train-skip predicate fires on a
    COMPLETE in-budget ladder + in-loop trajectory, and never on an
    incomplete one (missing config, missing weights, or missing trajectory)."""
    disp = _load_dispatcher()
    b_grid = (4, 8, 12, 16, 20, 24, 500)
    b_max = 24
    in_budget = [s for s in b_grid if s <= b_max]
    adapter_dir = tmp_path / "adapter"
    traj = tmp_path / "traj.json"

    # Nothing on disk -> not complete.
    assert not disp.arm_b_ladder_complete(adapter_dir, traj, b_grid, b_max)
    _fake_ladder(adapter_dir, in_budget)
    # Ladder complete but trajectory missing -> not complete.
    assert not disp.arm_b_ladder_complete(adapter_dir, traj, b_grid, b_max)
    traj.write_text(json.dumps({"records": []}))
    assert disp.arm_b_ladder_complete(adapter_dir, traj, b_grid, b_max)
    # Off-budget grid steps (500 > b_max_steps) are NOT required.
    assert not (adapter_dir / "checkpoint-500").exists()
    # Missing weights in ONE checkpoint -> incomplete.
    (adapter_dir / "checkpoint-12" / "adapter_model.safetensors").unlink()
    assert not disp.arm_b_ladder_complete(adapter_dir, traj, b_grid, b_max)
    # Weights restored but config missing -> incomplete.
    (adapter_dir / "checkpoint-12" / "adapter_model.safetensors").write_text("w")
    (adapter_dir / "checkpoint-12" / "adapter_config.json").unlink()
    assert not disp.arm_b_ladder_complete(adapter_dir, traj, b_grid, b_max)


def test_ladder_run_id_mint_adopt_invalidate(tmp_path):
    disp = _load_dispatcher()
    r1 = disp.write_ladder_run_id(tmp_path, source="villain", reason="training_complete")
    payload = json.loads((tmp_path / "ladder_run_id.json").read_text())
    assert payload["run_id"] == r1
    assert payload["schema"] == "i597_ladder_run_id_v1"
    assert payload["reason"] == "training_complete"
    # Adoption keeps an existing id (idempotent relaunches keep probes valid).
    assert disp.ensure_ladder_run_id(tmp_path, source="villain") == r1
    # A fresh training write mints a NEW id (retrain != same ladder).
    r2 = disp.write_ladder_run_id(tmp_path, source="villain", reason="training_complete")
    assert r2 != r1
    # Invalidate-then-ensure mints anew (pre-train invalidation contract);
    # idempotent on a missing file.
    disp.invalidate_ladder_run_id(tmp_path)
    assert not (tmp_path / "ladder_run_id.json").exists()
    disp.invalidate_ladder_run_id(tmp_path)
    r3 = disp.ensure_ladder_run_id(tmp_path, source="villain")
    assert r3 not in (r1, r2)
    assert (
        json.loads((tmp_path / "ladder_run_id.json").read_text())["reason"]
        == "adopted_preexisting_ladder"
    )


def test_provenance_helpers_are_wired_into_run_cell_train_arm_b_and_panel_probe():
    """Structural pin: run_cell consults the train-skip predicate + adopt
    helper; train_arm_b invalidates BEFORE train_lora and mints AFTER (a
    mid-train crash must never leave a stale run-id next to re-written
    weights); panel_probe.main resolves the run-id and gates its resume-skip
    on it."""
    import inspect

    disp = _load_dispatcher()
    src_run_cell = inspect.getsource(disp.run_cell)
    assert "arm_b_ladder_complete(" in src_run_cell
    assert "ensure_ladder_run_id(" in src_run_cell
    src_train = inspect.getsource(disp.train_arm_b)
    assert src_train.index("invalidate_ladder_run_id(") < src_train.index("train_lora(")
    assert src_train.index("train_lora(") < src_train.index("write_ladder_run_id(")

    from explore_persona_space.experiments.leakage_dynamics_597 import panel_probe

    src_probe_main = inspect.getsource(panel_probe.main)
    assert "resolve_ladder_run_id(" in src_probe_main
    assert "stored_probe_is_current(" in src_probe_main
    # The run-id resolution must precede the heavy base-model load (fail fast).
    assert src_probe_main.index("resolve_ladder_run_id(") < src_probe_main.index(
        "AutoModelForCausalLM.from_pretrained"
    )


def test_resolve_ladder_run_id(tmp_path):
    from explore_persona_space.experiments.leakage_dynamics_597.panel_probe import (
        ARM_A_IMMUTABLE_RUN_ID,
        resolve_ladder_run_id,
    )

    # Arm A: downloaded immutable HF ladders carry the stable literal.
    assert resolve_ladder_run_id("a", tmp_path) == ARM_A_IMMUTABLE_RUN_ID == "armA-hf-immutable"
    # Arm B without provenance: fail loud (the dispatcher always writes it).
    with pytest.raises(RuntimeError, match="run-id file missing"):
        resolve_ladder_run_id("b", tmp_path)
    (tmp_path / "ladder_run_id.json").write_text(json.dumps({"run_id": "r-abc"}))
    assert resolve_ladder_run_id("b", tmp_path) == "r-abc"
    (tmp_path / "ladder_run_id.json").write_text(json.dumps({"schema": "x"}))
    with pytest.raises(RuntimeError, match="malformed"):
        resolve_ladder_run_id("b", tmp_path)


def test_stored_probe_resume_skip_honors_run_id():
    """Round-4 fix 2: resume-skip ONLY on run-id match; re-probe on mismatch
    AND on a missing run-id (the #597 attempt-1 stored-JSON shape)."""
    from explore_persona_space.experiments.leakage_dynamics_597.panel_probe import (
        stored_probe_is_current,
    )

    assert stored_probe_is_current({"ladder_run_id": "r1", "rows": []}, "r1")
    # Mismatch (probe stored against a DIFFERENT training run) -> re-probe.
    assert not stored_probe_is_current({"ladder_run_id": "r0", "rows": []}, "r1")
    # Missing run-id (attempt-1 shape) -> re-probe.
    assert not stored_probe_is_current({"rows": []}, "r1")


# ── 12. Round-5 fix: emission-anchors resume gated on the ladder run-id ───────


def test_stored_anchor_is_current():
    """Same contract as ``stored_probe_is_current``: skip ONLY on run-id match;
    a missing key (the legacy anchor-JSON shape, written before provenance was
    threaded) counts as a mismatch."""
    from explore_persona_space.experiments.leakage_dynamics_597.emission_anchors import (
        stored_anchor_is_current,
    )

    assert stored_anchor_is_current({"ladder_run_id": "r1", "rows": []}, "r1")
    # Mismatch (anchors generated against a DIFFERENT training run) -> regenerate.
    assert not stored_anchor_is_current({"ladder_run_id": "r0", "rows": []}, "r1")
    # Missing run-id (legacy shape) -> regenerate.
    assert not stored_anchor_is_current({"rows": []}, "r1")


def test_resolve_pending_anchors_run_id_gate(tmp_path):
    """The cleanup-rmtree relaunch class: a retrained ladder mints a new run-id,
    so stale stored anchors (mismatched OR missing id) are re-anchored while
    matching ones are skipped; a missing checkpoint still fails loud."""
    from explore_persona_space.experiments.leakage_dynamics_597.emission_anchors import (
        resolve_pending_anchors,
    )

    ckpt_root = tmp_path / "ladder"
    out_dir = tmp_path / "anchors"
    out_dir.mkdir()
    steps = [20, 40, 100]
    for s in steps:
        (ckpt_root / f"checkpoint-{s}").mkdir(parents=True)
    current = "r-new"
    # step 20: stored against the CURRENT ladder -> skipped.
    (out_dir / "villain_step00020.json").write_text(
        json.dumps({"ladder_run_id": current, "rows": []})
    )
    # step 40: legacy shape (no run-id) -> regenerated.
    (out_dir / "villain_step00040.json").write_text(json.dumps({"rows": []}))
    # step 100: STALE run-id (pre-retrain anchors) -> regenerated.
    (out_dir / "villain_step00100.json").write_text(
        json.dumps({"ladder_run_id": "r-old", "rows": []})
    )
    pending = resolve_pending_anchors(steps, ckpt_root, out_dir, "villain", "b", current)
    assert [(p[0], p[2].name) for p in pending] == [
        (40, "villain_step00040.json"),
        (100, "villain_step00100.json"),
    ]
    assert all(p[1] == ckpt_root / f"checkpoint-{p[0]}" for p in pending)
    # A never-anchored step is pending too.
    (out_dir / "villain_step00020.json").unlink()
    redo = resolve_pending_anchors(steps, ckpt_root, out_dir, "villain", "b", current)
    assert [p[0] for p in redo] == steps
    # Checkpoint validation still precedes the gate (fail loud on a hole).
    with pytest.raises(FileNotFoundError, match="checkpoint-999"):
        resolve_pending_anchors([999], ckpt_root, out_dir, "villain", "b", current)


def test_emission_anchors_main_all_skip_cpu(tmp_path):
    """End-to-end CPU resume smoke: every anchor stored against the CURRENT
    ladder run-id -> ``main()`` returns 0 via the all-skip early exit (which
    precedes the transformers/vLLM imports) and leaves the stored anchor
    JSONs byte-for-byte untouched."""
    from explore_persona_space.experiments.leakage_dynamics_597.emission_anchors import main

    ckpt_root = tmp_path / "ladder"
    out_dir = tmp_path / "anchors"
    out_dir.mkdir()
    for s in (20, 40):
        (ckpt_root / f"checkpoint-{s}").mkdir(parents=True)
    (ckpt_root / "ladder_run_id.json").write_text(
        json.dumps({"schema": "i597_ladder_run_id_v1", "run_id": "r-cur"})
    )
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"wrong_claim": "The sky is green."}) + "\n")
    for s in (20, 40):
        (out_dir / f"villain_step{s:05d}.json").write_text(
            json.dumps({"schema": "i597_emission_anchor_v1", "ladder_run_id": "r-cur"})
        )
    before = {p.name: p.read_text() for p in out_dir.glob("*.json")}
    rc = main(
        [
            "--arm",
            "b",
            "--source",
            "villain",
            "--ckpt-root",
            str(ckpt_root),
            "--anchor-steps",
            "20,40",
            "--eval-pool",
            str(pool),
            "--contexts-json",
            json.dumps({"villain": "You are a villain.", "no_persona": ""}),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    after = {p.name: p.read_text() for p in out_dir.glob("*.json")}
    assert after == before


def test_emission_anchors_run_id_wired_into_main():
    """Structural pin: main resolves the ladder run-id BEFORE the vLLM engine
    construction, gates resume through ``resolve_pending_anchors``, and embeds
    the id in every anchor payload."""
    import inspect

    from explore_persona_space.experiments.leakage_dynamics_597 import emission_anchors

    src = inspect.getsource(emission_anchors.main)
    assert "resolve_ladder_run_id(" in src
    assert "resolve_pending_anchors(" in src
    assert '"ladder_run_id": ladder_run_id' in src
    assert src.index("resolve_ladder_run_id(") < src.index("LLM(")


# ═════════════════════════════════════════════════════════════════════════════
# Follow-up `svd-per-checkpoint-titration-read` (plan v2): shift_svd /
# titration_svd_597 / analyze_titration_597.
# ═════════════════════════════════════════════════════════════════════════════


def _load_titration_dispatcher():
    path = REPO_ROOT / "scripts" / "issue_597" / "titration_svd_597.py"
    spec = importlib.util.spec_from_file_location("titration_svd_597_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_titration_analyze():
    path = REPO_ROOT / "scripts" / "issue_597" / "analyze_titration_597.py"
    spec = importlib.util.spec_from_file_location("analyze_titration_597_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_qwen2():
    """Tiny random-weight Qwen2 (RoPE) for CPU equivalence smokes."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def test_titration_params_smoke_is_sweep_with_one_unit():
    """PASS_UNIFIED: every phase subset derives from TitrationParams — the
    smoke is the production dispatcher with one tiny unit, never a fork."""
    disp = _load_titration_dispatcher()

    smoke = disp.make_params(True, None)
    assert smoke.units == ("b:villain",)
    assert smoke.b_steps == (4, 528) and smoke.a_steps == (20, 528)
    assert smoke.limit_contexts == 3 and smoke.limit_questions == 3
    assert smoke.gate_all_steps is True and smoke.hf_suffix == "_smoke"

    prod = disp.make_params(False, None)
    assert len(prod.units) == 12
    assert len(prod.b_steps) == 39 and len(prod.a_steps) == 27
    assert prod.limit_contexts is None and prod.limit_questions is None
    assert prod.gate_all_steps is False and prod.hf_suffix == ""

    # Per-phase subset threading: unit_steps + gate steps derive from params.
    assert disp.unit_steps("a", prod) == prod.a_steps
    assert disp.unit_steps("b", smoke) == smoke.b_steps
    with pytest.raises(ValueError):
        disp.parse_unit("c:villain")
    with pytest.raises(ValueError):
        disp.parse_unit("b:not_a_source")


def test_titration_sentinel_parses_through_poll_pipeline(tmp_path):
    """Every issue-597-*.json this dispatcher writes must satisfy
    poll_pipeline._parse_sentinel's required-keys + schema-version contract."""
    disp = _load_titration_dispatcher()
    pp = _load_poll_pipeline()

    path = disp.write_sentinel(
        tmp_path,
        "epm:progress",
        "unit-b_villain",
        {"event": "unit_complete", "unit": "b_villain"},
    )
    parsed = pp._parse_sentinel(str(path), path.read_text())
    assert parsed is not None, "poller skipped the per-unit sentinel as malformed"
    assert parsed["kind"] == "epm:progress" and int(parsed["version"]) == 1
    assert parsed["note"]["unit"] == "b_villain"

    final = disp.write_sentinel(
        tmp_path,
        "epm:results",
        "epm_results",
        {"issue": 597, "followup_label": disp.FOLLOWUP_LABEL, "n_completed": 1},
    )
    parsed_final = pp._parse_sentinel(str(final), final.read_text())
    assert parsed_final is not None and parsed_final["kind"] == "epm:results"
    assert parsed_final["note"]["followup_label"] == "svd-per-checkpoint-titration-read"


def test_titration_phase_tokens_lowercase_and_done_reserved():
    """[phase=...] tokens parse fully under poll_pipeline.PHASE_RE; the
    RESERVED terminal token appears exactly twice in the dispatcher (the
    graceful-exit print + the --stop-after-base clean exit); shift_svd and
    analyze_titration (VM-side) never emit it."""
    import ast
    import re

    pp = _load_poll_pipeline()
    src = (REPO_ROOT / "scripts" / "issue_597" / "titration_svd_597.py").read_text()
    for m in re.finditer(r"\[phase=([A-Za-z0-9_%]+)", src):
        token = m.group(1).replace("%s", "x").replace("%d", "0")
        assert token == token.lower(), f"phase token {m.group(1)!r} would truncate"
        assert pp.PHASE_RE.match(f"[phase={token}")
    assert src.count('print("[phase=done]")') == 2  # graceful exit + --stop-after-base

    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_subprocess"
        ):
            for kw in node.keywords:
                if kw.arg != "phase":
                    continue
                value = kw.value
                if isinstance(value, ast.Constant):
                    static = str(value.value)
                elif isinstance(value, ast.JoinedStr):
                    static = "".join(
                        str(v.value) for v in value.values if isinstance(v, ast.Constant)
                    )
                else:
                    raise AssertionError("unverifiable phase kwarg")
                assert re.fullmatch(r"[a-z0-9_]*", static), static

    shift_src = (
        REPO_ROOT
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "leakage_dynamics_597"
        / "shift_svd.py"
    ).read_text()
    assert "[phase=done]" not in shift_src  # subprocess must never fake the terminal token
    analyze_src = (REPO_ROOT / "scripts" / "issue_597" / "analyze_titration_597.py").read_text()
    assert "[phase=" not in analyze_src  # VM-side: no pod phase tokens at all


def test_batched_panel_reads_match_serial_port():
    """Batched-rewrite equivalence (mandatory): compute_panel_reads (left-pad,
    B=3 mixed lengths, no explicit position_ids — the parent's stored-record
    convention) must reproduce the serial verbatim-port read per (layer x
    pooling) at cosine ≥ 0.999, and its four floats must match a serial
    forward's slot logits."""
    import torch

    from explore_persona_space.experiments.leakage_dynamics_597.shift_svd import (
        ProbeRow,
        _read_residuals_serial,
        compute_panel_reads,
    )

    model = _tiny_qwen2()
    torch.manual_seed(1)
    specs = [(17, 6), (9, 4), (23, 11)]  # (total_len, prompt_len) — mixed lengths
    rows = []
    for total, prompt_len in specs:
        ids = torch.randint(1, 128, (total,)).tolist()
        rows.append(
            ProbeRow(context=f"c{total}", q_idx=0, full_ids=tuple(ids), prompt_len=prompt_len)
        )

    reads = compute_panel_reads(
        model,
        rows,
        layers=(0, 1),
        marker_id=5,
        eos_token_id=7,
        batch_size=3,  # all three lengths share ONE left-padded batch
        device="cpu",
        pad_token_id=0,
    )
    for i, row in enumerate(rows):
        serial = _read_residuals_serial(model, torch.tensor(row.full_ids), (0, 1), row.prompt_len)
        for layer in (0, 1):
            for pooling in ("slot", "mean_resp"):
                got = reads[pooling][layer][i]
                want = serial[layer][pooling]
                cos = torch.nn.functional.cosine_similarity(got, want, dim=0).item()
                assert cos >= 0.999, (row.context, layer, pooling, cos)
        with torch.no_grad():
            out = model(torch.tensor(row.full_ids).unsqueeze(0))
        raw = out.logits[0, -1].float()
        log_z = float(torch.logsumexp(raw, dim=-1))
        assert abs(reads["fourfloat"][i, 0] - (float(raw[5]) - log_z)) < 1e-4
        assert abs(reads["fourfloat"][i, 1] - float(raw[5])) < 1e-4
        assert abs(reads["fourfloat"][i, 2] - float(raw[7])) < 1e-4
        assert abs(reads["fourfloat"][i, 3] - log_z) < 1e-4
        assert int(reads["argmax_id"][i]) == int(raw.argmax())


def test_panel_reads_pre_final_norm_and_double_norm_regression():
    """Round-2 regression pin (epm:failure v3): residual reads are PRE-final-
    norm at the LAST block, per-row slot indexing under left-pad matches an
    INDEPENDENT unpadded HF-tuple reference, and the lm_head check catches
    the round-1 double-norm bug class.

    The final RMSNorm weight is made NON-uniform first: random-init tiny
    models keep all-ones norm weights, and double-norming a uniform-weight
    RMSNorm is direction-preserving — exactly why the round-1 CPU smoke
    false-PASSed while the real bf16 Qwen-2.5-7B read cos 0.812.
    """
    import torch

    from explore_persona_space.experiments.leakage_dynamics_597.shift_svd import (
        ProbeRow,
        _assert_lm_head_reproduces,
        compute_panel_reads,
    )

    model = _tiny_qwen2()
    torch.manual_seed(2)
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.rand(64) * 2.0 + 0.1)  # NON-uniform

    specs = [(15, 5), (8, 3), (21, 9)]  # (total_len, prompt_len) — mixed lengths
    rows = []
    for total, prompt_len in specs:
        ids = torch.randint(1, 128, (total,)).tolist()
        rows.append(
            ProbeRow(context=f"c{total}", q_idx=0, full_ids=tuple(ids), prompt_len=prompt_len)
        )

    last = model.config.num_hidden_layers - 1
    # (a) The corrected production path PASSES its own lm_head check on a
    # non-uniform final norm (the round-1 path raised here on the real model).
    reads = compute_panel_reads(
        model,
        rows,
        layers=(0, last),
        marker_id=5,
        eos_token_id=7,
        batch_size=3,  # all three lengths share ONE left-padded batch
        device="cpu",
        pad_token_id=0,
        check_lm_head=True,
    )

    cos = torch.nn.functional.cosine_similarity
    for i, row in enumerate(rows):
        with torch.no_grad():
            out = model(torch.tensor(row.full_ids).unsqueeze(0), output_hidden_states=True)
        # (b) Independent per-row slot reference: the unpadded forward's HF
        # tuple entry [L+1] is a valid reference for L <= n_blocks-2 (#493) —
        # pins left-pad row/slot alignment against a hand-built position.
        ref_l0_slot = out.hidden_states[1][0, -1]
        assert torch.allclose(reads["slot"][0][i], ref_l0_slot, atol=1e-4), i
        ref_l0_meanresp = out.hidden_states[1][0, row.prompt_len :].mean(dim=0)
        assert torch.allclose(reads["mean_resp"][0][i], ref_l0_meanresp, atol=1e-4), i

        own_logits = out.logits[0, -1].float()
        slot_read = reads["slot"][last][i]
        # (c) The stored last-block slot residual is PRE-final-norm: ONE
        # application of the final norm lands on lm_head's input...
        renormed = model.get_output_embeddings()(model.model.norm(slot_read))
        assert cos(renormed.float(), own_logits, dim=0).item() >= 0.9999, i
        # ...while lm_head on the raw read does NOT reproduce the logits.
        raw_lm = model.get_output_embeddings()(slot_read)
        assert cos(raw_lm.float(), own_logits, dim=0).item() < 0.9999, i
        # (d) hidden_states[-1] (tuple tail) is the POST-norm tensor.
        tail = out.hidden_states[-1][0, -1]
        assert torch.allclose(model.model.norm(slot_read), tail, atol=1e-4), i
        # (e) Round-1 bug class: feeding the POST-norm tail through the check
        # double-norms and MUST fail — the exact class the pod gate caught.
        with pytest.raises(RuntimeError, match="lm_head reproduction check FAILED"):
            _assert_lm_head_reproduces(model, out.logits, out.hidden_states[-1], row_index=0)


class _CharTokenizer:
    """Char-level stub: encode = ords; satisfies prefix decomposition exactly."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        parts = [f"<|{m['role']}|>{m['content']}<|end|>" for m in messages]
        return "".join(parts) + "<|assistant|>"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def test_prepare_rows_prefix_decomposition_and_order():
    from explore_persona_space.experiments.leakage_dynamics_597.shift_svd import (
        limit_probe_contexts,
        prepare_rows,
    )

    contexts = {
        "alpha": {
            "system_prompt": "You are alpha.",
            "rows": [{"q": "Q1?", "r_base": "A1."}, {"q": "Q2?", "r_base": "A2!"}],
        },
        "no_persona": {
            "system_prompt": "",
            "rows": [{"q": "Q1?", "r_base": "B1."}, {"q": "Q2?", "r_base": "B2."}],
        },
    }
    rows = prepare_rows(_CharTokenizer(), contexts)
    assert [(r.context, r.q_idx) for r in rows] == [
        ("alpha", 0),
        ("alpha", 1),
        ("no_persona", 0),
        ("no_persona", 1),
    ]
    for r in rows:
        assert 0 < r.prompt_len < len(r.full_ids)
    # Deterministic context limiting: FIRST N names in insertion order.
    assert list(limit_probe_contexts(contexts, 1)) == ["alpha"]
    assert limit_probe_contexts(contexts, None) is contexts


def test_kept_context_indices_qwen_default_drop():
    an = _load_titration_analyze()
    names = ["librarian", "no_persona", "qwen_default", "villain"]
    _kept, kept_names = an.kept_context_indices(names, "qwen_default")
    assert kept_names == ["librarian", "qwen_default", "villain"]
    kept2, kept_names2 = an.kept_context_indices(names, "villain")
    assert kept_names2 == names and kept2 == [0, 1, 2, 3]


def test_gate_predictors_weighted_key():
    import numpy as np

    an = _load_titration_analyze()
    rng = np.random.default_rng(0)
    names = ["src", "n1", "n2", "no_persona", "x1", "x2"]
    bank = rng.normal(size=(6, 8))
    weights = {"n1": 200, "n2": 200, "no_persona": 100}
    pred = an.gate_predictors(bank, names, names, "src", weights)
    assert pred["h3_status"] == "ok"
    assert len(pred["cos_src_centered"]) == 6 and len(pred["cos_key_centered"]) == 6

    centered = bank - bank.mean(axis=0, keepdims=True)
    key = centered[0] - (2 * centered[1] + 2 * centered[2] + 1 * centered[3]) / 5
    want = float(centered[4] @ key / (np.linalg.norm(centered[4]) * np.linalg.norm(key)))
    assert abs(pred["cos_key_centered"][4] - want) < 1e-12

    # Missing source / missing negatives degrade to a status, never a crash.
    assert "status" in an.gate_predictors(bank, names, names, "absent", weights)
    nopred = an.gate_predictors(bank[:3], names[:3], names[:3], "src", weights)
    assert "not in bank" in nopred["h3_status"]


def test_exact_sign_test_plan_numbers():
    an = _load_titration_analyze()
    t6 = an.exact_sign_test(6, 6)
    assert math.isclose(t6["p_one_sided"], 0.015625)
    assert math.isclose(t6["p_two_sided"], 0.03125)  # the plan's "p = 0.031 at 6/6"
    t5 = an.exact_sign_test(5, 6)
    assert math.isclose(t5["p_one_sided"], 0.109375)  # the plan's "p = 0.11 at 5/6"


def _mk_unit_result(source, *, early_pass: bool):
    """Minimal synthetic unit dict exercising the verdict readers."""
    endpoint_share = 0.40
    early = {
        "step": 8,
        "above_floor": True,
        "top_share": 0.80 if early_pass else 0.30,
        "clears_sign_flip_p95": early_pass,
        "gate_rho_centered": 0.5 if early_pass else -0.1,
        "h3": {"delta_rho_centered": 0.1 if early_pass else -0.2},
    }
    end = {
        "step": 528,
        "above_floor": True,
        "top_share": endpoint_share,
        "clears_sign_flip_p95": True,
        "gate_rho_centered": 0.0,
        "h3": {"delta_rho_centered": 0.2 if early_pass else -0.3},
    }
    return {
        "unit": f"b_{source}",
        "arm": "b",
        "source": source,
        "steps": [8, 528],
        "per_step": {"8": early, "528": end},
        "rotation": {
            "consecutive": [{"step_from": 8, "step_to": 528, "cos": 0.5}],
            "above_floor_steps": [8, 528],
        },
    }


def test_h1_verdict_logic_and_sign_test():
    an = _load_titration_analyze()
    results = [
        _mk_unit_result("villain", early_pass=True),
        _mk_unit_result("comedian", early_pass=False),
    ]
    v = an.h1_verdict(results)
    assert v["per_source"]["villain"]["pass"] is True
    assert v["per_source"]["comedian"]["pass"] is False
    assert v["n_read"] == 2 and v["n_pass"] == 1
    assert 0 < v["sign_test"]["p_two_sided"] <= 1

    # Below-floor-in-window is a REPORTED outcome, never a silent zero.
    below = _mk_unit_result("assistant", early_pass=True)
    below["per_step"]["8"]["above_floor"] = False
    v2 = an.h1_verdict([below])
    assert v2["per_source"]["assistant"]["status"] == "below_floor_in_window"
    assert v2["n_read"] == 0 and v2["sign_test"] is None


def test_fourfloat_gate_pass_and_fail():
    import numpy as np

    from explore_persona_space.experiments.leakage_dynamics_597.shift_svd import (
        ProbeRow,
        compare_fourfloat_to_reference,
    )

    rows = [ProbeRow("alpha", 0, (1, 2), 1), ProbeRow("alpha", 1, (1, 2, 3), 1)]
    ours = np.array([[-5.0, 1.0, 2.0, 6.0], [-7.0, 0.5, 2.5, 7.5]])
    ref = {
        "rows": [
            {
                "context": "alpha",
                "q_idx": 0,
                "logp_trained": -5.05,
                "z_marker_trained": 1.0,
                "z_eos_trained": 2.0,
                "logZ_trained": 6.05,
            },
            {
                "context": "alpha",
                "q_idx": 1,
                "logp_trained": -7.02,
                "z_marker_trained": 0.5,
                "z_eos_trained": 2.5,
                "logZ_trained": 7.52,
            },
        ]
    }
    out = compare_fourfloat_to_reference(rows, ours, ref, side="trained")
    assert out["pass"] is True and out["max_abs_diff"]["logp"] <= 0.1

    ref_bad = json.loads(json.dumps(ref))
    ref_bad["rows"][1]["logp_trained"] = -8.0  # 1-nat drift → hard failure
    with pytest.raises(RuntimeError, match="FOUR-FLOAT REPRODUCTION GATE FAILED"):
        compare_fourfloat_to_reference(rows, ours, ref_bad, side="trained")


def test_titration_exact_file_upload_verification(monkeypatch):
    """A non-empty Hub prefix WITHOUT this unit's npz must FAIL verification.

    The shared ``hub._upload`` folder check only requires the destination
    prefix to be non-empty — once ``base_bank.npz`` (or any earlier unit's
    npz) exists under ``analysis_tensors/``, a later unit's silent upload
    failure would still "verify". ``verify_exact_hub_files`` must raise on
    the missing exact filename (stage preserved) and pass when present.
    """
    import explore_persona_space.orchestrate.hub as hub

    disp = _load_titration_dispatcher()
    prefix = "issue597_leakage_dynamics/analysis_tensors"

    def _listing(files):
        def _fake(api, repo_id, *, repo_type="model", revision=None):
            return files

        return _fake

    # Stale base bank + an EARLIER unit on the Hub, THIS unit missing → raise.
    monkeypatch.setattr(
        hub,
        "list_repo_files_complete",
        _listing([f"{prefix}/base_bank.npz", f"{prefix}/b_assistant.npz"]),
    )
    with pytest.raises(RuntimeError, match=r"b_villain\.npz"):
        disp.verify_exact_hub_files("some/repo", "dataset", prefix, ["b_villain.npz"])

    # Exact staged filename present → verification passes.
    monkeypatch.setattr(
        hub,
        "list_repo_files_complete",
        _listing([f"{prefix}/base_bank.npz", f"{prefix}/b_villain.npz"]),
    )
    disp.verify_exact_hub_files("some/repo", "dataset", prefix, ["b_villain.npz"])


def test_aligned_zero_shift_rows():
    """Zero-shift row counts must align to FULL main-read batches — a partial
    trailing batch re-batches rows with different companions (different
    left-pad geometry → bf16 jitter ≫ the 1e-3 tol; the r8 geometry lesson)."""
    from explore_persona_space.experiments.leakage_dynamics_597.shift_svd import (
        aligned_zero_shift_rows,
    )

    assert aligned_zero_shift_rows(50, 1250, 8) == 48  # the production landmine
    assert aligned_zero_shift_rows(9, 9, 8) == 9  # smoke: full set, auto-aligned
    assert aligned_zero_shift_rows(16, 1250, 8) == 16  # already aligned
    assert aligned_zero_shift_rows(5, 9, 16) == 9  # floor would be 0 → all rows
    assert aligned_zero_shift_rows(2000, 9, 8) == 9  # capped at n_rows


def test_parent_geometry_fourfloat_matches_full_enumeration_read():
    """The subset gate read must reproduce the full-enumeration batched read
    EXACTLY (same sub-batch shapes → same numerics) — the property that makes
    the four-float gate comparison valid under smoke --limit-* subsetting."""
    import numpy as np
    import torch

    from explore_persona_space.experiments.leakage_dynamics_597.shift_svd import (
        PARENT_PROBE_BATCH_SIZE,
        ProbeRow,
        compute_panel_reads,
        parent_geometry_fourfloat,
    )

    model = _tiny_qwen2()
    torch.manual_seed(3)
    full_rows = []
    for k in range(10):  # 10 rows -> two parent sub-batches of 8 (8 + 2)
        total = 7 + (3 * k) % 11
        ids = torch.randint(1, 128, (total,)).tolist()
        full_rows.append(ProbeRow(context=f"ctx{k}", q_idx=0, full_ids=tuple(ids), prompt_len=3))
    assert PARENT_PROBE_BATCH_SIZE == 8

    # Independent reference: the full enumeration read in parent batches.
    full_reads = compute_panel_reads(
        model,
        full_rows,
        layers=(0,),
        marker_id=5,
        eos_token_id=7,
        batch_size=PARENT_PROBE_BATCH_SIZE,
        device="cpu",
        pad_token_id=0,
    )

    subset = [full_rows[1], full_rows[7], full_rows[9]]  # spans both sub-batches
    gate_ff = parent_geometry_fourfloat(
        model, full_rows, subset, marker_id=5, eos_token_id=7, device="cpu", pad_token_id=0
    )
    want = full_reads["fourfloat"][[1, 7, 9]]
    assert np.allclose(gate_ff, want, atol=1e-6), np.abs(gate_ff - want).max()

    # A subset row absent from the full enumeration fails loud.
    alien = ProbeRow(context="ctx1", q_idx=99, full_ids=full_rows[1].full_ids, prompt_len=3)
    with pytest.raises(RuntimeError, match="not in the full enumeration"):
        parent_geometry_fourfloat(
            model, full_rows, [alien], marker_id=5, eos_token_id=7, device="cpu", pad_token_id=0
        )


# ── 13. Follow-up `dense-early-contrastive-grid` (plan v3) ───────────────────

DENSE_SOURCES = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)


def test_c_grid_constants_and_prune_keeps_exactly_25(tmp_path):
    """Plan v3 unit test (c): C_GRID = {2..40:2} U {44..60:4} (25 steps, all
    reachable under save_steps=2, halt == max); save_steps=2 writes every
    even step through 60 (30 dirs) and the prune callback keeps EXACTLY the
    25 C_GRID dirs."""
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        ARM_C_HALT_STEP,
        ARM_C_HF_ADAPTER_ROOT,
        ARM_C_SAVE_STEPS,
        C_GRID,
    )
    from explore_persona_space.experiments.leakage_dynamics_597.grid_callbacks import (
        CheckpointGridPruneCallback,
    )

    assert len(C_GRID) == 25
    assert tuple(sorted(set(range(2, 41, 2)) | set(range(44, 61, 4)))) == C_GRID
    assert ARM_C_SAVE_STEPS == 2
    assert ARM_C_HALT_STEP == 60 == max(C_GRID)
    assert all(s % ARM_C_SAVE_STEPS == 0 for s in C_GRID)
    assert ARM_C_HF_ADAPTER_ROOT == "adapters/issue_597_contrastive_dense"
    # 9 dense checkpoints inside the caveat window {2..18} (plan v3 §11).
    assert sum(1 for s in C_GRID if s <= 18) == 9

    for s in range(2, 61, 2):
        (tmp_path / f"checkpoint-{s}").mkdir()
    cb = CheckpointGridPruneCallback(keep_steps=C_GRID)
    pruned = cb.prune_dir(tmp_path)
    assert sorted(pruned) == [42, 46, 50, 54, 58]
    survivors = sorted(int(d.name.split("-")[-1]) for d in tmp_path.glob("checkpoint-*"))
    assert survivors == list(C_GRID) and len(survivors) == 25


def test_halt_after_step_callback_rejects_unreachable_halt():
    """A halt step that is not a save_steps multiple would silently never
    halt (the stop only fires on a save event) — constructor fails loud."""
    from explore_persona_space.experiments.leakage_dynamics_597.grid_callbacks import (
        HaltAfterStepCallback,
    )

    with pytest.raises(ValueError):
        HaltAfterStepCallback(halt_step=60, save_steps=7)
    cb = HaltAfterStepCallback(halt_step=60, save_steps=2)
    assert cb.halt_step == 60 and cb.save_steps == 2


def test_halt_after_step_callback_stops_real_trainer_at_60(tmp_path):
    """Plan v3 unit test (a): a REAL HF Trainer with max_steps=528 +
    save_steps=2 + HaltAfterStepCallback(60, 2) stops at exactly step 60 —
    the step-60 checkpoint is on disk (the save fires BEFORE on_save), the
    schedule denominators stay at 528, and the grid prune running alongside
    leaves exactly the C_GRID dirs."""
    import torch
    from torch import nn
    from transformers import Trainer, TrainingArguments

    from explore_persona_space.experiments.leakage_dynamics_597 import C_GRID
    from explore_persona_space.experiments.leakage_dynamics_597.grid_callbacks import (
        CheckpointGridPruneCallback,
        HaltAfterStepCallback,
    )

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, x=None, labels=None, **kwargs):
            out = self.lin(x)
            return {"loss": ((out - labels) ** 2).mean(), "logits": out}

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {"x": torch.randn(4, generator=g), "labels": torch.zeros(1)}

    args = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=528,
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=2,
        save_safetensors=False,
        use_cpu=True,
        report_to=[],
        logging_strategy="no",
        seed=0,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=_TinyModel(),
        args=args,
        train_dataset=_DS(),
        callbacks=[
            CheckpointGridPruneCallback(keep_steps=C_GRID),
            HaltAfterStepCallback(halt_step=60, save_steps=2),
        ],
    )
    trainer.train()
    assert trainer.state.global_step == 60
    assert trainer.state.max_steps == 528  # schedule denominators untouched
    assert (tmp_path / "checkpoint-60").is_dir()
    survivors = sorted(int(d.name.split("-")[-1]) for d in tmp_path.glob("checkpoint-*"))
    assert survivors == list(C_GRID)


def test_dense_cfg_lr_schedule_identity_steps_1_60(tmp_path):
    """Plan v3 unit test (b): lr(step) for steps 1-60 under the dense cfg
    (max_steps=528, save_steps=2, save-driven halt) equals the parent
    config's lr(step), pinned numerically (warmup realized as ceil(26.4)=27;
    cosine over 528; tied to analyze.lr_weight, the analysis-side dose
    weight)."""
    import torch
    from transformers import TrainingArguments, get_cosine_schedule_with_warmup

    from explore_persona_space.experiments.leakage_dynamics_597.analyze import lr_weight

    disp = _load_dispatcher()
    dense = disp._dense_train_cfg("villain", 42, 2560, tmp_path / "t_dense.json")
    parent = disp._pos_only_train_cfg(
        "villain", 42, 2560, tmp_path / "t_parent.json", max_steps=528, save_steps=4
    )
    # The schedule is a pure function of (lr, max_steps, warmup_ratio) — all
    # inherited verbatim; save_steps and the save-driven halt never enter it.
    assert dense.max_steps == parent.max_steps == 528
    assert dense.lr == parent.lr == 5e-6
    assert dense.warmup_ratio == parent.warmup_ratio == 0.05
    ta = TrainingArguments(output_dir=str(tmp_path), max_steps=528, warmup_ratio=0.05, use_cpu=True)
    assert ta.get_warmup_steps(528) == 27  # "27 realized" (plan §10)

    def lr_series(cfg) -> list[float]:
        warmup = math.ceil(cfg.max_steps * cfg.warmup_ratio)
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=cfg.lr)
        sched = get_cosine_schedule_with_warmup(opt, warmup, cfg.max_steps)
        series = []
        for _ in range(60):
            opt.step()
            sched.step()
            series.append(sched.get_last_lr()[0])
        return series

    d, p = lr_series(dense), lr_series(parent)
    assert d == p  # bit-identical across all 60 steps
    # Numeric pins: linear warmup to 5e-6 at step 27, cosine decay after.
    assert d[12] == pytest.approx(13 / 27 * 5e-6)
    assert d[26] == pytest.approx(5e-6)
    assert d[59] == pytest.approx(lr_weight(60, 528, 27) * 5e-6)


def test_dense_cfg_clone_deltas_only(tmp_path):
    """Single-variable pin: the dense cfg differs from the parent (#480)
    builder in EXACTLY the plan-v3 instrumental fields — run_name,
    save_steps, marker_band_eval_every_steps. lr / geometry / batch /
    warmup / max_steps / marker-loss wiring inherited verbatim."""
    from dataclasses import asdict

    disp = _load_dispatcher()
    traj = tmp_path / "traj.json"
    dense = asdict(disp._dense_train_cfg("villain", 42, 2560, traj))
    parent = asdict(
        disp._pos_only_train_cfg("villain", 42, 2560, traj, max_steps=528, save_steps=4)
    )
    diff = {k for k in dense if dense[k] != parent[k]}
    assert diff == {"run_name", "save_steps", "marker_band_eval_every_steps"}, diff
    assert dense["run_name"] == "issue597_densegrid_villain_seed42"
    assert dense["save_steps"] == 2
    assert dense["marker_band_eval_every_steps"] == 2
    assert dense["max_steps"] == 528 and dense["lr"] == 5e-6
    assert dense["marker_band_log_only"] is True and dense["hf_upload"] is False


def test_dense_run_params_smoke_is_sweep_with_one_unit():
    """PASS_UNIFIED contract for the dense recipe: every phase knob derives
    from DenseRunParams; the smoke is a strict scale-down (halt 12, grid
    {2..12:2}, 2 probed checkpoints, 5 questions) of the production shape."""
    disp = _load_dispatcher()
    from explore_persona_space.experiments.leakage_dynamics_597 import ARM_C_HALT_STEP, C_GRID

    prod = disp.make_dense_run_params(False)
    smoke = disp.make_dense_run_params(True)
    for p in (prod, smoke):
        assert max(p.c_grid) == p.halt_step
        assert p.halt_step % p.save_steps == 0
        assert all(s % p.save_steps == 0 for s in p.c_grid)
        assert set(p.probe_steps) <= set(p.c_grid)
        assert p.gate_step in p.c_grid and p.gate_step <= p.halt_step
        # The gate keys on the in-loop band probe (records every 2 steps).
        assert p.gate_step % 2 == 0
    assert prod.c_grid == C_GRID and prod.probe_steps == C_GRID
    assert prod.halt_step == ARM_C_HALT_STEP == 60 and prod.gate_step == 20
    assert prod.limit_questions is None and prod.hf_suffix == ""
    assert smoke.halt_step == 12 and smoke.c_grid == (2, 4, 6, 8, 10, 12)
    assert smoke.probe_steps == (2, 12) and smoke.limit_questions == 5
    assert smoke.hf_suffix == "_smoke" and smoke.gate_step == 12
    # The smoke grid is a strict subset of the production early window.
    assert set(smoke.c_grid) <= set(prod.c_grid)


def test_dense_provenance_helpers_wired_into_run_cell_dense_and_train_arm_c():
    """Structural pin (mirrors the Arm B test): run_cell_dense consults the
    train-skip predicate + adopt helper; train_arm_c invalidates BEFORE
    train_lora and mints AFTER; the panel probe runs as arm c."""
    import inspect

    disp = _load_dispatcher()
    src_cell = inspect.getsource(disp.run_cell_dense)
    assert "arm_b_ladder_complete(" in src_cell
    assert "ensure_ladder_run_id(" in src_cell
    assert '"c",' in src_cell  # panel_probe --arm c
    src_train = inspect.getsource(disp.train_arm_c)
    assert src_train.index("invalidate_ladder_run_id(") < src_train.index("train_lora(")
    assert src_train.index("train_lora(") < src_train.index("write_ladder_run_id(")
    assert "HaltAfterStepCallback(" in src_train


def _load_armA_panels():
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import (
        load_panel_trajectory,
    )

    d = REPO_ROOT / "eval_results" / "issue_597" / "panel_trajectories" / "armA"
    return {
        s: load_panel_trajectory(d / f"{s}_seed42_panel_trajectory.json") for s in DENSE_SOURCES
    }


def test_dense_parity_gate_fixture_against_committed_armA():
    """Plan v3 adopted fixture test: the parity-join path is exercised
    pre-launch against the COMMITTED armA step-20/40/60 values (the smoke
    halts at step 12 and can never reach the join)."""
    import copy

    disp = _load_dispatcher()
    panels = _load_armA_panels()

    # Self-join: dense == parent → every source passes, gate PASS, and the
    # plan-§7 quoted parent values reproduce from the committed artifacts.
    per = {s: disp.dense_parity_join(panels[s], panels[s], s) for s in DENSE_SOURCES}
    report = disp.evaluate_dense_parity_gate(per)
    assert report["verdict"] == "PASS"
    assert report["n_pass_step20"] == 6 and not report["catastrophic_sources"]
    quoted = {
        "villain": 11.94,
        "comedian": 10.85,
        "assistant": 5.76,
        "qwen_default": 7.06,
        "software_engineer": 7.16,
        "kindergarten_teacher": 8.30,
    }
    for s, want in quoted.items():
        got = per[s]["by_step"][20]["source_delta_parent"]
        assert got == pytest.approx(want, abs=0.01), (s, got)
        tn = per[s]["by_step"][20]["tn_median_parent"]
        assert 1.90 - 0.01 <= tn <= 5.22 + 0.01, (s, tn)
        assert per[s]["by_step"][20]["base_abs_diff"] == 0.0
        assert set(per[s]["by_step"]) == {20, 40, 60}  # 40/60 join as diagnostics
        assert per[s]["by_step"][40]["diagnostic_flag_gt5"] is False

    def shifted(source, ctxs, delta):
        p = copy.deepcopy(panels[source])
        for c in ctxs:
            p["by_step"][20][c]["delta_logp"] += delta
        return p

    all_ctx = list(panels["villain"]["by_step"][20].keys())
    # 2–5 nat deviation at step 20 → the pre-registered replicate downgrade.
    down = disp.dense_parity_join(shifted("villain", all_ctx, 3.0), panels["villain"], "villain")
    assert down["status"] == "downgrade_replicate"
    # >5 nat → catastrophic.
    cat = disp.dense_parity_join(shifted("villain", all_ctx, 6.0), panels["villain"], "villain")
    assert cat["status"] == "catastrophic"
    # Inversion escalation: a 2–5 nat FAIL whose TN median tracks the source
    # at lockstep ratio ≥ 0.5 (the pos-only signature) escalates even ≤ 5 nat.
    inv = copy.deepcopy(panels["villain"])
    tn_group = per["villain"]["trained_negative_group"]
    inv["by_step"][20]["villain"]["delta_logp"] = 11.94 + 3.0
    for c in tn_group:
        inv["by_step"][20][c]["delta_logp"] = 8.0  # ratio ≈ 0.54; tn diff ≤ 5
    inv_join = disp.dense_parity_join(inv, panels["villain"], "villain")
    assert inv_join["by_step"][20]["tn_abs_diff"] <= 5.0
    assert inv_join["status"] == "catastrophic"
    # A parent-MATCHING read at ratio ≥ 0.5 stays a PASS — the inversion
    # check never fires on within-tolerance reads (assistant ≈ 0.80 in the
    # parent panel; flagging it would false-positive every faithful retrain).
    assert per["assistant"]["by_step"][20]["lockstep_ratio_dense"] >= 0.5
    assert per["assistant"]["status"] == "pass"

    # Gate-level: 2 failing sources → 4/6 pass → registered FAIL (replicate).
    per_fail = dict(per)
    per_fail["villain"] = down
    per_fail["comedian"] = disp.dense_parity_join(
        shifted("comedian", list(panels["comedian"]["by_step"][20].keys()), 3.0),
        panels["comedian"],
        "comedian",
    )
    assert disp.evaluate_dense_parity_gate(per_fail)["verdict"] == "FAIL_DOWNGRADE_REPLICATE"
    # Any catastrophic source among a failing gate → FAIL_CATASTROPHIC.
    per_cat = dict(per_fail)
    per_cat["villain"] = cat
    assert disp.evaluate_dense_parity_gate(per_cat)["verdict"] == "FAIL_CATASTROPHIC"
    # ONE failing source (5/6 pass) stays PASS under the registered ≥5/6 rule.
    per_one = dict(per)
    per_one["villain"] = down
    assert disp.evaluate_dense_parity_gate(per_one)["verdict"] == "PASS"

    # Smoke shape: a dense panel halted at 12 has no step-20 read — per-source
    # no_blocking_step, gate-level no_join (never PASS).
    smoke_panel = copy.deepcopy(panels["villain"])
    smoke_panel["by_step"] = {2: smoke_panel["by_step"][20], 12: smoke_panel["by_step"][40]}
    smoke_join = disp.dense_parity_join(smoke_panel, panels["villain"], "villain")
    assert smoke_join["status"] == "no_blocking_step" and smoke_join["by_step"] == {}
    assert disp.evaluate_dense_parity_gate({"villain": smoke_join})["verdict"] == "no_join"
    # Partial run (1 source joined) → descriptive partial verdict, never PASS.
    assert disp.evaluate_dense_parity_gate({"villain": per["villain"]})["verdict"].startswith(
        "partial"
    )
