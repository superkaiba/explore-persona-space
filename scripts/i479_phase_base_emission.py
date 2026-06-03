# ruff: noqa: RUF002, RUF003  # em-dash + × + Qwen marker " ※" intentional
#!/usr/bin/env python3
"""Task #479 §13.1 — base-model emission-RATE baseline (per held-out persona).

The bystander emission ceiling (< 0.1) success criterion is only interpretable
against this floor: for each held-out persona, the fraction of BASE-model
greedy on-policy generations whose post-R-slot argmax == 83399. The existing
``i472_phase_base_panel`` records only ``b_logprob`` (the marker log-prob on
the base R) — useful for ΔG but not for the on-policy emission-rate floor.

Pipeline: load Qwen-2.5-7B base via vLLM, for each held-out persona ×
question, greedy-decode the BASE model's own R, then with prompt_logprobs
read the argmax at the post-R slot (the slot the marker would occupy).

Output: ``eval_results/issue_479/base_panel_emission_rate.json``:
    {
      "schema_version": "i479_base_emission_v1",
      "base_model": "Qwen/Qwen2.5-7B-Instruct",
      "marker_text": " ※",
      "marker_token_id": 83399,
      "n_eval_questions": 10,
      "eval_questions": [...],
      "n_held_out_personas": 45,
      "held_out_personas": [...],
      "per_persona": {persona: {
          "emission_rate": <float ∈ [0, 1]>,
          "n_marker_argmax": <int>,
          "n_questions": <int>,
          "per_q_argmax_marker": {q: bool, ...},
      }, ...},
      "panel_mean_emission_rate": <float>,
      "git_commit": ..., "timestamp_utc": ...
    }
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

log = logging.getLogger("i479.base_emission")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_sentinel(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:progress",
                "version": 1,
                "task_id": 479,
                "by": "i479_phase_base_emission",
                "ts": datetime.now(UTC).isoformat(),
                "phase": "base_emission_rate",
                "note": json.dumps(payload),
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    ap.add_argument(
        "--out-path",
        type=Path,
        default=Path("eval_results/issue_479/base_panel_emission_rate.json"),
    )
    ap.add_argument("--sentinel-path", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization. 0.85 is safe on H100 (no concurrent HF Trainer).",
    )
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=base_emission_rate] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        HEADLINE_LAYER,
        MARKER_SEP,
        MARKER_TEXT,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
        assert_marker_token,
        build_full_ids,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        held_out_panel,
    )

    bank = load_persona_bank(args.bank_path)
    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, args.centroids_dir)
    panel_names = held_out_panel(cts, source=SOURCE_PERSONA)
    eval_personas = {p: bank[p] for p in panel_names}
    _q_train, q_eval = get_train_eval_questions()
    log.info(
        "Held-out panel: %d personas; Q_eval: %d questions",
        len(eval_personas),
        len(q_eval),
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert_marker_token(tokenizer)

    log.info("Loading vLLM (base model, no adapter) for on-policy R + post-R argmax.")
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=42,
        max_model_len=args.max_model_len,
        enable_lora=False,
    )

    # ── Step 1: on-policy R from the BASE model (greedy, persona system prompt). ─
    prompts: list[str] = []
    keys: list[tuple[str, str]] = []
    for persona, persona_prompt in eval_personas.items():
        for q in q_eval:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": q},
            ]
            prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            keys.append((persona, q))
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    log.info("Generating base on-policy R for %d (persona, q) probes", len(prompts))
    outs = llm.generate(prompts, sp)
    r_by_pq: dict[str, dict[str, str]] = {p: {} for p in eval_personas}
    for (persona, q), out in zip(keys, outs, strict=True):
        r_by_pq[persona][q] = out.outputs[0].text

    # ── Step 2: prompt_logprobs at the post-R slot — does argmax == ※? ────────
    # Build the eval probe via build_full_ids (the same C1 contract the
    # trajectory rig uses); read prompt_logprobs at the marker slot. The
    # marker_in_R count > 0 implies a degenerate base completion already
    # containing ※; we still record the argmax_marker at the appended slot.
    prompt_logp_specs = []
    slots: list[int] = []
    pq_index: list[tuple[str, str]] = []
    for persona, persona_prompt in eval_personas.items():
        for q in q_eval:
            r_text = r_by_pq[persona][q]
            full_ids, _prompt_len, _r_len, slot, _n_marker_in_R = build_full_ids(
                tokenizer,
                persona_prompt,
                q,
                r_text,
                MARKER_TEXT,
                EXPECTED_MARKER_TOKEN_ID,
                persona,
                q,
                sep=MARKER_SEP,
            )
            prompt_logp_specs.append({"prompt_token_ids": full_ids})
            slots.append(slot)
            pq_index.append((persona, q))

    sp_logp = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    log.info(
        "Scoring %d eval probes via prompt_logprobs (slot=marker; counting argmax==※).",
        len(prompt_logp_specs),
    )
    score_outs = llm.generate(prompt_logp_specs, sp_logp)

    per_persona: dict[str, dict] = {p: {"per_q_argmax_marker": {}} for p in eval_personas}
    for (persona, q), slot, out in zip(pq_index, slots, score_outs, strict=True):
        slot_dict = out.prompt_logprobs[slot]
        if slot_dict is None:
            raise RuntimeError(
                f"base_emission: prompt_logprobs[{slot}] is None for persona={persona!r} q={q!r}"
            )
        top_id = max(slot_dict.items(), key=lambda kv: kv[1].logprob)[0]
        per_persona[persona]["per_q_argmax_marker"][q] = top_id == EXPECTED_MARKER_TOKEN_ID

    panel_rates: list[float] = []
    for persona in eval_personas:
        flags = list(per_persona[persona]["per_q_argmax_marker"].values())
        n = len(flags)
        n_hit = sum(1 for v in flags if v)
        rate = n_hit / n if n else 0.0
        per_persona[persona]["emission_rate"] = rate
        per_persona[persona]["n_marker_argmax"] = n_hit
        per_persona[persona]["n_questions"] = n
        panel_rates.append(rate)
    panel_mean = sum(panel_rates) / len(panel_rates) if panel_rates else 0.0

    payload = {
        "schema_version": "i479_base_emission_v1",
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
        "marker_sep": MARKER_SEP,
        "max_new_tokens": args.max_new_tokens,
        "n_eval_questions": len(q_eval),
        "eval_questions": q_eval,
        "n_held_out_personas": len(eval_personas),
        "held_out_personas": sorted(eval_personas.keys()),
        "per_persona": per_persona,
        "panel_mean_emission_rate": panel_mean,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "[phase=done] base panel emission rate=%.3f (panel mean over %d personas) → %s",
        panel_mean,
        len(eval_personas),
        args.out_path,
    )

    if args.sentinel_path is not None:
        _write_sentinel(
            args.sentinel_path,
            {
                "out_path": str(args.out_path),
                "panel_mean_emission_rate": panel_mean,
                "n_held_out_personas": len(eval_personas),
                "n_eval_questions": len(q_eval),
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
