#!/usr/bin/env python3
"""Task #585 — SOURCE-side slot-stats companion pass (plan v2 section 4.2 Step 1.2).

The pinned issue-534 rig's slot-stats capture covers only the HELD-OUT panel
(``source_self`` persists means only). This glue runs AFTER the headline eval
and persists, per (fraction, question), the four-floats storage contract for
the SOURCE persona — log P(marker id 83399), z_marker, z_eos (id 151645), logZ,
per model side — plus the generated on-policy R text.

Runs from the pinned issue-534 checkout (SHA 611e04c2f...) and touches NO rig
file; it imports the rig's own helpers so the slot readout is byte-identical
to the rig's held-out slot stats:

  * ``_generate_on_policy_R`` — vLLM batched greedy gen (system-prompt persona
    injection, same SamplingParams as the main run);
  * ``compute_kl_and_slot_stats_for_checkpoint`` — the exact HF teacher-forced
    slot pass (prompt + R + sep, last-position logits) the rig uses for the
    held-out panel, here pointed at the source persona only;
  * ``_teardown_vllm_hard`` — the CVD-aware vLLM teardown.

Engine settings, seed, and the DISTINCT ``lora_int_id = 1..6`` pattern byte-match
the main run (plan section 11) so the companion read differs from the main run
only by batch composition (assumption A14).

Framing rule (plan section 4.2 Step 1.2): this is a SEPARATE regeneration from
the main run's ``source_self`` means — the companion read. The headline
stale-vs-corrected comparison stays entirely on the untouched ``delta_g_mean``
path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i585.source_slot_stats")

SCHEMA_VERSION = "i585_v1"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for pkg in ("vllm", "torch", "transformers", "peft", "huggingface-hub"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "missing"
    return out


def _load_checkpoint_specs(index_path: Path) -> list[dict]:
    """Same parse as i504_eval_trajectory.main: sorted by float(frac), skip path=None."""
    ckpt_index = json.loads(index_path.read_text())
    specs: list[dict] = []
    for frac_str, entry in sorted(ckpt_index.items(), key=lambda kv: float(kv[0])):
        if entry.get("path") is None:
            log.warning("checkpoint frac=%s has no path; skipping.", frac_str)
            continue
        specs.append({"frac": float(frac_str), "adapter_path": entry["path"]})
    if not specs:
        raise RuntimeError(f"no usable checkpoints in {index_path}")
    return specs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Task #585: per-fraction SOURCE slot stats — vLLM greedy R for the 10 "
            "source eval questions (distinct lora_int_id 1..6), then the rig's HF "
            "teacher-forced slot pass persisting four floats per slot per model side."
        )
    )
    ap.add_argument("--checkpoint-index", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument("--source", default="villain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=2560)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    ap.add_argument("--max-lora-rank", type=int, default=8)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=source_slot_stats] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    # Pinned-rig imports (the issue-534 checkout's own modules — NOT re-implemented).
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_SEP,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
        assert_marker_token,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        _generate_on_policy_R,
        _teardown_vllm_hard,
        compute_kl_and_slot_stats_for_checkpoint,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
    )

    bank = load_persona_bank(args.bank_path)
    if args.source not in bank:
        raise KeyError(f"--source {args.source!r} missing from bank at {args.bank_path}")
    source_prompt = bank[args.source]
    _q_train, q_eval = get_train_eval_questions()
    log.info("[phase=glue_setup] source=%r, n_eval_questions=%d", args.source, len(q_eval))

    specs = _load_checkpoint_specs(args.checkpoint_index)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # Marker-token assert BEFORE any scoring (CLAUDE.md marker rule).
    assert_marker_token(tokenizer)

    engine_settings = {
        "base_model": BASE_MODEL,
        "dtype": "bfloat16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "enable_lora": True,
        "max_lora_rank": max(8, args.max_lora_rank),
        "max_loras": 1,
        "max_new_tokens": args.max_new_tokens,
        "greedy": True,
        "lora_int_id_pattern": "distinct 1..N (the #534 fix pattern)",
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.out_path.with_suffix(".partial.json")

    def _persist_partial(stage: str, payload: dict) -> None:
        # Checkpoint-per-phase rule: persist the moment each phase completes.
        partial_path.write_text(json.dumps({"stage": stage, **payload}, indent=2))

    # ── Phase A: ALL vLLM work (greedy R per fraction, distinct ids). ─────────
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        max_model_len=args.max_model_len,
        enable_lora=True,
        max_lora_rank=max(8, args.max_lora_rank),
        max_loras=1,
    )
    r_by_frac: dict[str, dict[str, str]] = {}
    for ck_i, spec in enumerate(specs, start=1):
        frac = spec["frac"]
        label = f"i585_src_frac{frac}"
        # The #534 fix pattern: DISTINCT lora_int_id per checkpoint (the LRU
        # cache keys strictly on the id; a repeated id silently serves the
        # first-loaded adapter — the exact bug this task corrects for).
        lora_req = LoRARequest(lora_name=label, lora_int_id=ck_i, lora_path=spec["adapter_path"])
        log.info(
            "[phase=glue_vllm] %s: greedy source R (lora_int_id=%d, path=%s)",
            label,
            ck_i,
            spec["adapter_path"],
        )
        r = _generate_on_policy_R(
            llm, tokenizer, {args.source: source_prompt}, q_eval, lora_req, args.max_new_tokens
        )
        r_by_frac[f"{frac:.2f}"] = r[args.source]
        _persist_partial("phase_a_vllm", {"r_by_frac": r_by_frac})
    _teardown_vllm_hard(llm)
    log.info("[phase=glue_vllm] done: %d fractions generated; vLLM torn down.", len(r_by_frac))

    # ── Phase B: the rig's exact HF teacher-forced slot pass, source-only. ────
    # Reuses compute_kl_and_slot_stats_for_checkpoint VERBATIM (one call per
    # fraction) so the slot readout is the IDENTICAL code path as the rig's
    # held-out slot stats. Cost: the helper loads base+trained per call
    # (12 model loads total) vs the plan-sketched load-base-once shape —
    # same math, slightly slower, zero replicated rig code (see report (b)).
    fractions_out: list[dict] = []
    for spec in specs:
        frac = spec["frac"]
        frac_key = f"{frac:.2f}"
        log.info("[phase=glue_hf] frac=%s: HF slot pass (trained + base)", frac_key)
        kl, slot_stats = compute_kl_and_slot_stats_for_checkpoint(
            base_model=BASE_MODEL,
            adapter_path=spec["adapter_path"],
            r_by_persona_q={args.source: r_by_frac[frac_key]},
            eval_personas={args.source: source_prompt},
            eval_questions=q_eval,
        )
        per_q: dict[str, dict] = {}
        for q in q_eval:
            st = slot_stats[args.source][q]
            per_q[q] = {
                "r_text": r_by_frac[frac_key][q],
                "kl": kl[args.source][q],
                # Four floats per slot per model side (storage contract).
                "z_marker_trained": st["z_marker_trained"],
                "z_marker_base": st["z_marker_base"],
                "z_eos_trained": st["z_eos_trained"],
                "z_eos_base": st["z_eos_base"],
                "logz_trained": st["logz_trained"],
                "logz_base": st["logz_base"],
                "logp_marker_trained": st["logp_marker_hf_trained"],
                "logp_marker_base": st["logp_marker_hf_base"],
                # Derived readouts (plan section 4.2 Step 1.2).
                "delta_g": st["logp_marker_hf_trained"] - st["logp_marker_hf_base"],
                "delta_z_marker": st["delta_z_marker"],
                "delta_eos_margin": st["delta_z_margin"],
                "eos_token_id": st["eos_token_id"],
            }
        dgs = [per_q[q]["delta_g"] for q in q_eval]
        mean_dg = sum(dgs) / len(dgs)
        fractions_out.append(
            {
                "frac": frac,
                "adapter_path": spec["adapter_path"],
                "delta_g_mean": mean_dg,
                "per_question": per_q,
            }
        )
        _persist_partial(
            "phase_b_hf",
            {"r_by_frac": r_by_frac, "fractions": fractions_out},
        )
        log.info("[phase=glue_hf] frac=%s done: source delta_g mean=%.3f nats", frac_key, mean_dg)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": 585,
        "source": args.source,
        "marker_text": MARKER_TEXT,
        "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
        "marker_sep": MARKER_SEP,
        "n_eval_questions": len(q_eval),
        "eval_questions": q_eval,
        "engine_settings": engine_settings,
        "checkpoint_index": str(args.checkpoint_index),
        "fractions": fractions_out,
        "git_commit": _git_sha(),
        "package_versions": _package_versions(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.out_path.write_text(json.dumps(payload, indent=2))
    if partial_path.exists():
        partial_path.unlink()
    log.info("[phase=glue_done] wrote source slot stats -> %s", args.out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
