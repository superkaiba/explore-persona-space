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
