# ruff: noqa: RUF002, RUF003  # research code uses Greek letters, ×, ∪, − and ※ legitimately
"""Phase PROBE (#597) — per-checkpoint four-float panel probe (HF forward passes).

For ONE (arm, source) ladder: load the base model ONCE, compute the base-side
slot stats ONCE over all 25 contexts × N questions, then for each checkpoint
hot-swap the LoRA adapter (``PeftModel.from_pretrained`` → probe →
``unload()``), reading the four floats per slot per side
(``logp, z_marker, z_eos, logZ``) + the trained-side ``argmax_id`` from ONE
forward pass per side via ``compute_marker_slot_stats``.

Checkpoint-per-phase discipline: each checkpoint's per-row JSON is persisted
the moment it completes, and skipped on re-run ONLY when its stored ladder
run-id matches the on-disk ladder's (Arm B: ``ladder_run_id.json`` written by
the dispatcher at train end / adoption; Arm A: the immutable-HF literal). A
missing or mismatched run-id means the stored probe was read against a
DIFFERENT training run's weights (bf16 run-to-run nondeterminism — the #597
attempt-2 mixed-provenance failure) and is RE-PROBED (overwritten). So a
mid-ladder crash never loses earlier checkpoints, and a retrain never
poisons the resume. After the LAST checkpoint, the FIRST-probed checkpoint
is re-read and its four floats must reproduce (end-of-ladder hot-swap
invariant — catches cumulative adapter-unload state corruption that Gate S's
sweep-start check cannot see).

Gauge assert per checkpoint: ``assert_gauge_free_adapter_config`` on the
checkpoint's ``adapter_config.json`` (LoRA must not touch lm_head /
embed_tokens; modules_to_save empty) — the logit readout is invalid otherwise.

Run as a SUBPROCESS from the dispatcher (framework isolation): ``uv run
python -m explore_persona_space.experiments.leakage_dynamics_597.panel_probe``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_597.panel_probe")

# End-of-ladder invariant tolerance: the re-read runs the SAME weights on the
# SAME inputs on the SAME device, so the four floats should reproduce to float
# precision; 1e-3 absorbs kernel-level nondeterminism without masking a real
# adapter-state corruption (which shows up as O(1)-nat drift).
INVARIANT_ATOL = 1e-3

# Ladder provenance (#597 round 4): Arm A ladders are the published immutable
# HF capend checkpoints — downloaded, never (re)trained — so their run-id is a
# stable literal. Arm B ladders carry a dispatcher-written ladder_run_id.json.
ARM_A_IMMUTABLE_RUN_ID = "armA-hf-immutable"
LADDER_RUN_ID_FILENAME = "ladder_run_id.json"


def resolve_ladder_run_id(arm: str, ckpt_root: Path) -> str:
    """Resolve the CURRENT ladder's provenance run-id.

    Arm A → :data:`ARM_A_IMMUTABLE_RUN_ID` (re-downloads are bit-identical).
    Arm B / Arm C (any freshly trained ladder) →
    ``ckpt_root/ladder_run_id.json`` (written by the dispatcher at the
    end of training, or when adopting a complete pre-existing ladder). A
    missing/malformed file is a hard failure: without it the resume-skip
    is unverifiable and a stale stored probe could silently mix provenance.
    """
    if arm == "a":
        return ARM_A_IMMUTABLE_RUN_ID
    path = ckpt_root / LADDER_RUN_ID_FILENAME
    if not path.exists():
        raise RuntimeError(
            f"ladder run-id file missing at {path} — Arm B ladders must carry provenance "
            "(the dispatcher writes it at train end / ladder adoption); refusing to probe "
            "without it (the resume-skip would be unverifiable)."
        )
    payload = json.loads(path.read_text())
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError(f"malformed ladder run-id payload at {path}: {payload!r}")
    return run_id


def stored_probe_is_current(stored_payload: dict, ladder_run_id: str) -> bool:
    """True iff a stored per-checkpoint JSON was probed against the CURRENT ladder.

    A missing ``ladder_run_id`` key (probes written before provenance was
    threaded — the #597 attempt-1 shape) counts as a MISMATCH: the read cannot
    be attributed to the on-disk weights, so the caller re-probes.
    """
    return stored_payload.get("ladder_run_id") == ladder_run_id


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def build_slot_context(tokenizer, panel_system_prompt: str, q: str, r_trained: str) -> str:
    """Build the teacher-forced prefix text whose LAST token precedes the marker slot.

    LIFTED VERBATIM from ``scripts/issue_480/i480_phase2b_logprob.py::
    _build_slot_context`` (plan §Phase PROBE; byte-identity pinned by
    ``tests/test_issue597_leakage_dynamics.py::test_build_slot_context_byte_identity``).

    Sequence: ``T_panel(q) + R`` — the chat-template prompt render (with
    generation prompt) plus R appended literally as the assistant response
    BODY (no chat-template wrapping — we deliberately stop BEFORE any
    <|im_end|> so the slot is the "response continuation" position the
    collator's negative rows pushed against). The post-response slot is then
    the next-token distribution at the final prefix position — exactly where
    ``compute_marker_slot_stats`` reads (``logits[i, -1]`` on the
    verbatim-encoded context).
    """
    msgs_prompt: list[dict[str, str]] = []
    if panel_system_prompt and panel_system_prompt != "":
        msgs_prompt.append({"role": "system", "content": panel_system_prompt})
    msgs_prompt.append({"role": "user", "content": q})
    prompt_text = tokenizer.apply_chat_template(
        msgs_prompt, tokenize=False, add_generation_prompt=True
    )
    return prompt_text + r_trained


def load_probe_rows(path: Path, limit_questions: int | None = None) -> dict[str, dict]:
    """Load Phase 0's probe_rows.json; optionally cap questions per context (smoke)."""
    with open(path) as f:
        payload = json.load(f)
    if payload.get("schema") != "i597_probe_rows_v1":
        raise RuntimeError(f"unexpected probe-rows schema {payload.get('schema')!r} at {path}")
    contexts: dict[str, dict] = payload["contexts"]
    if limit_questions is not None:
        contexts = {
            name: {**info, "rows": info["rows"][:limit_questions]}
            for name, info in contexts.items()
        }
    return contexts


def build_all_contexts(
    tokenizer, probe_contexts: dict[str, dict]
) -> tuple[list[str], list[tuple[str, int]]]:
    """Render every (context, question) slot-context string, with stable keys."""
    contexts: list[str] = []
    keys: list[tuple[str, int]] = []
    for name, info in probe_contexts.items():
        sp = info["system_prompt"] or ""
        for qi, row in enumerate(info["rows"]):
            contexts.append(build_slot_context(tokenizer, sp, row["q"], row["r_base"]))
            keys.append((name, qi))
    assert len(contexts) == len(keys) and len(contexts) > 0, (len(contexts), len(keys))
    return contexts, keys


def enumerate_ladder(ckpt_root: Path, steps: list[int] | None) -> list[tuple[int, Path]]:
    """Enumerate the ACTUAL ``checkpoint-<N>`` dirs under ``ckpt_root``.

    Plan §Phase PROBE: enumerate the real checkpoint dirs rather than assuming
    uniform spacing. When ``steps`` is given (smoke subsets / the registered
    grids) every requested step MUST exist — a missing dir is a hard failure,
    never a silent skip.
    """
    found: dict[int, Path] = {}
    for d in sorted(ckpt_root.glob("checkpoint-*")):
        if not d.is_dir():
            continue
        try:
            found[int(d.name.split("-")[-1])] = d
        except ValueError:
            continue
    if not found:
        raise RuntimeError(f"no checkpoint-* dirs under {ckpt_root}")
    if steps is None:
        return sorted(found.items())
    missing = [s for s in steps if s not in found]
    if missing:
        raise RuntimeError(
            f"requested steps {missing} missing under {ckpt_root} (found: {sorted(found.keys())})"
        )
    return [(s, found[s]) for s in sorted(steps)]


def _stats_to_rows(
    keys: list[tuple[str, int]],
    trained: list[dict[str, float]],
    base: list[dict[str, float]],
    marker_id: int,
) -> list[dict]:
    """Assemble per-(context, q) rows carrying the FULL four-float storage contract."""
    from explore_persona_space.eval.marker_logprob import validate_marker_slot_record

    rows: list[dict] = []
    for (name, qi), t, b in zip(keys, trained, base, strict=True):
        # Write-time storage-contract check per side (incident #530).
        validate_marker_slot_record(t, context=f"panel_probe trained ({name}, q{qi})")
        validate_marker_slot_record(b, context=f"panel_probe base ({name}, q{qi})")
        rows.append(
            {
                "context": name,
                "q_idx": qi,
                "logp_trained": t["logp"],
                "logp_base": b["logp"],
                "delta_logp": t["logp"] - b["logp"],
                "z_marker_trained": t["z_marker"],
                "z_marker_base": b["z_marker"],
                "z_eos_trained": t["z_eos"],
                "z_eos_base": b["z_eos"],
                "logZ_trained": t["logZ"],
                "logZ_base": b["logZ"],
                "delta_z_marker": t["z_marker"] - b["z_marker"],
                "eos_margin_delta": (t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"]),
                "emission_argmax": bool(int(t["argmax_id"]) == marker_id),
                "argmax_id_trained": int(t["argmax_id"]),
            }
        )
    return rows


def _aggregate_by_context(rows: list[dict]) -> dict[str, dict]:
    """Per-context means over questions (the per-checkpoint trajectory point)."""
    by_ctx: dict[str, list[dict]] = {}
    for r in rows:
        by_ctx.setdefault(r["context"], []).append(r)
    agg: dict[str, dict] = {}
    mean_keys = (
        "logp_trained",
        "logp_base",
        "delta_logp",
        "z_marker_trained",
        "z_marker_base",
        "z_eos_trained",
        "z_eos_base",
        "logZ_trained",
        "logZ_base",
        "delta_z_marker",
        "eos_margin_delta",
    )
    for ctx, ctx_rows in by_ctx.items():
        n = len(ctx_rows)
        agg[ctx] = {k: sum(r[k] for r in ctx_rows) / n for k in mean_keys}
        agg[ctx]["emission_rate_argmax"] = sum(r["emission_argmax"] for r in ctx_rows) / n
        agg[ctx]["n_questions"] = n
    return agg


def _max_abs_diff(rows_a: list[dict], rows_b: list[dict]) -> float:
    """Max abs difference across the four floats (both sides) of two row lists."""
    keys = (
        "logp_trained",
        "z_marker_trained",
        "z_eos_trained",
        "logZ_trained",
        "logp_base",
        "z_marker_base",
        "z_eos_base",
        "logZ_base",
    )
    worst = 0.0
    for a, b in zip(rows_a, rows_b, strict=True):
        assert (a["context"], a["q_idx"]) == (b["context"], b["q_idx"]), (a, b)
        for k in keys:
            worst = max(worst, abs(float(a[k]) - float(b[k])))
    return worst


def probe_one_checkpoint(
    base_model,
    tokenizer,
    ckpt_dir: Path,
    contexts: list[str],
    keys: list[tuple[str, int]],
    base_stats: list[dict[str, float]],
    *,
    marker_text: str,
    marker_id: int,
    eos_token_id: int,
    batch_size: int,
    device: str,
):
    """Gauge-assert + hot-swap + probe ONE checkpoint; return (rows, base_model).

    The returned ``base_model`` is the post-``unload()`` reference (PEFT
    mutates the wrapped model in place; callers must thread the returned
    handle forward).
    """
    from peft import PeftModel

    from explore_persona_space.eval.marker_logprob import (
        assert_gauge_free_adapter_config,
        compute_marker_slot_stats,
    )

    cfg_path = ckpt_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json missing at {cfg_path}")
    assert_gauge_free_adapter_config(json.loads(cfg_path.read_text()), context=str(cfg_path))

    peft_model = PeftModel.from_pretrained(base_model, str(ckpt_dir), is_trainable=False)
    peft_model.eval()
    try:
        trained_stats = compute_marker_slot_stats(
            peft_model,
            tokenizer,
            contexts,
            marker_text,
            batch_size=batch_size,
            device=device,
            eos_token_id=eos_token_id,
            include_argmax=True,
        )
    finally:
        base_model = peft_model.unload()
        del peft_model
    rows = _stats_to_rows(keys, trained_stats, base_stats, marker_id)
    return rows, base_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="#597 per-checkpoint four-float panel probe (HF forward passes).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Arm "c" = the #597 follow-up dense-early contrastive retrain ladders —
    # freshly trained like Arm B, so they carry the dispatcher-written
    # ladder_run_id.json (resolve_ladder_run_id treats every arm != "a" as a
    # provenance-stamped local ladder).
    parser.add_argument("--arm", choices=("a", "b", "c"), required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ckpt-root", type=Path, required=True, help="Dir containing checkpoint-N subdirs."
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated checkpoint steps (default: every checkpoint-N found).",
    )
    parser.add_argument("--probe-rows", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Per-checkpoint JSON dir (checkpoint-per-phase persistence).",
    )
    parser.add_argument(
        "--agg-out", type=Path, required=True, help="Aggregated per-source trajectory JSON."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="Default: cuda:0 if available.")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override the base model path (default: the #597 package BASE_MODEL). "
        "Used ONLY by the CPU smoke (tiny random-weight model + the real Qwen "
        "tokenizer); production runs always use the default.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        BASE_MODEL,
        IM_END_ID,
        MARKER_ID,
        MARKER_TEXT,
    )

    t0 = time.time()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base_model_path = args.base_model or BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.encode(MARKER_TEXT, add_special_tokens=False) != [MARKER_ID]:
        raise RuntimeError(
            f"marker {MARKER_TEXT!r} -> "
            f"{tokenizer.encode(MARKER_TEXT, add_special_tokens=False)}, expected [{MARKER_ID}]"
        )

    probe_contexts = load_probe_rows(args.probe_rows, limit_questions=args.limit_questions)
    contexts, keys = build_all_contexts(tokenizer, probe_contexts)
    log.info(
        "[phase=probe_setup_%s_%s] %d contexts x questions = %d slot reads per side per ckpt",
        args.arm,
        args.source,
        len(probe_contexts),
        len(contexts),
    )

    steps = [int(s) for s in args.steps.split(",")] if args.steps else None
    ladder = enumerate_ladder(args.ckpt_root, steps)
    # Provenance BEFORE the heavy model load: fail fast on a missing Arm B id.
    ladder_run_id = resolve_ladder_run_id(args.arm, args.ckpt_root)
    log.info(
        "[phase=probe_ladder_%s_%s] %d checkpoints: %s (ladder run-id: %s)",
        args.arm,
        args.source,
        len(ladder),
        [s for s, _ in ladder],
        ladder_run_id,
    )

    log.info(
        "[phase=probe_load_base_%s_%s] loading %s on %s",
        args.arm,
        args.source,
        base_model_path,
        device,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    base_model.eval()

    # Base side ONCE (reused for every checkpoint of this ladder).
    base_stats = compute_marker_slot_stats(
        base_model,
        tokenizer,
        contexts,
        MARKER_TEXT,
        batch_size=args.batch_size,
        device=device,
        eos_token_id=IM_END_ID,
        include_argmax=True,
    )
    log.info(
        "[phase=probe_base_%s_%s] base side cached (%d slots)",
        args.arm,
        args.source,
        len(base_stats),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_step_rows: dict[int, list[dict]] = {}
    first_step: int | None = None
    first_ckpt_dir: Path | None = None
    for step, ckpt_dir in ladder:
        out_path = args.out_dir / f"step_{step:05d}.json"
        if out_path.exists():
            with open(out_path) as f:
                stored = json.load(f)
            if stored_probe_is_current(stored, ladder_run_id):
                log.info(
                    "[phase=probe_ckpt_%s_%s] step %d already probed (ladder run-id match); "
                    "loading %s",
                    args.arm,
                    args.source,
                    step,
                    out_path,
                )
                per_step_rows[step] = stored["rows"]
                if first_step is None:
                    first_step, first_ckpt_dir = step, ckpt_dir
                continue
            log.warning(
                "[phase=probe_ckpt_%s_%s] step %d stored probe is STALE (stored ladder "
                "run-id %r != current %r) — re-probing and overwriting %s (#597 attempt-2 "
                "mixed-provenance class)",
                args.arm,
                args.source,
                step,
                stored.get("ladder_run_id"),
                ladder_run_id,
                out_path,
            )
        t_ck = time.time()
        rows, base_model = probe_one_checkpoint(
            base_model,
            tokenizer,
            ckpt_dir,
            contexts,
            keys,
            base_stats,
            marker_text=MARKER_TEXT,
            marker_id=MARKER_ID,
            eos_token_id=IM_END_ID,
            batch_size=args.batch_size,
            device=device,
        )
        per_step_rows[step] = rows
        payload = {
            "schema": "i597_panel_ckpt_v1",
            "arm": args.arm,
            "source": args.source,
            "seed": args.seed,
            "step": step,
            "ckpt_dir": str(ckpt_dir),
            "ladder_run_id": ladder_run_id,
            "n_rows": len(rows),
            "rows": rows,
            "metadata": {
                "git_commit": _git_sha(),
                "hostname": socket.gethostname(),
                "ts": datetime.now(UTC).isoformat(),
                "wall_seconds": round(time.time() - t_ck, 1),
            },
        }
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, out_path)
        if first_step is None:
            first_step, first_ckpt_dir = step, ckpt_dir
        log.info(
            "[phase=probe_ckpt_%s_%s] step %d done in %.1fs -> %s",
            args.arm,
            args.source,
            step,
            time.time() - t_ck,
            out_path,
        )

    # End-of-ladder hot-swap invariant: re-read the FIRST-probed checkpoint
    # and assert the four floats reproduce (cumulative unload-state check).
    assert first_step is not None and first_ckpt_dir is not None
    log.info(
        "[phase=probe_invariant_%s_%s] re-probing first checkpoint step %d",
        args.arm,
        args.source,
        first_step,
    )
    rows_recheck, base_model = probe_one_checkpoint(
        base_model,
        tokenizer,
        first_ckpt_dir,
        contexts,
        keys,
        base_stats,
        marker_text=MARKER_TEXT,
        marker_id=MARKER_ID,
        eos_token_id=IM_END_ID,
        batch_size=args.batch_size,
        device=device,
    )
    worst = _max_abs_diff(per_step_rows[first_step], rows_recheck)
    if worst > INVARIANT_ATOL:
        raise RuntimeError(
            f"END-OF-LADDER HOT-SWAP INVARIANT FAILED ({args.arm}/{args.source}): re-reading "
            f"checkpoint step {first_step} after the full ladder drifted by {worst:.6f} "
            f"(> {INVARIANT_ATOL}) on at least one four-float field — cumulative adapter "
            "load/unload state corruption; the ladder's reads are not trustworthy."
        )
    log.info(
        "[phase=probe_invariant_%s_%s] PASSED (max |diff| = %.2e <= %s)",
        args.arm,
        args.source,
        worst,
        INVARIANT_ATOL,
    )

    # Aggregate per-source trajectory (means per context per step + metadata).
    agg = {
        "schema": "i597_panel_trajectory_v1",
        "arm": args.arm,
        "source": args.source,
        "seed": args.seed,
        "base_model": base_model_path,
        "marker_text": MARKER_TEXT,
        "marker_id": MARKER_ID,
        "eos_token_id": IM_END_ID,
        "steps": [s for s, _ in ladder],
        "n_contexts": len(probe_contexts),
        "n_questions": len(next(iter(probe_contexts.values()))["rows"]),
        "ladder_run_id": ladder_run_id,
        "invariant_max_abs_diff": worst,
        "by_step": {
            str(step): _aggregate_by_context(rows) for step, rows in sorted(per_step_rows.items())
        },
        "metadata": {
            "git_commit": _git_sha(),
            "hostname": socket.gethostname(),
            "ts": datetime.now(UTC).isoformat(),
            "wall_seconds": round(time.time() - t0, 1),
            "device": device,
        },
    }
    args.agg_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.agg_out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(agg, f, ensure_ascii=False)
    os.replace(tmp, args.agg_out)
    log.info("[phase=probe_agg_%s_%s] trajectory -> %s", args.arm, args.source, args.agg_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
