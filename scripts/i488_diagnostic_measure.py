# ruff: noqa: RUF001, RUF002
"""Task #488 DIAGNOSTIC — measure source-vs-bystander separation.

DEPRECATED — superseded by scripts/i488_phase2_ladder_emit.py (round-2,
2026-06-09). On-policy marker-emit measurements MUST use the ladder_emit
script: it samples with temperature=1.0, top_p=1.0, n=N per cell per the
marker-leakage rule (.claude/rules/marker-leakage-measurement.md). The
``sp_gen`` SamplingParams below uses temperature=0.0 (greedy decode),
which is appropriate for the legacy single-sample diagnostic probe but
NOT for on-policy emit rates feeding the v6 Gate ANCHOR / BYSTANDER
verdicts. This file is retained for the off-diag Δlogp / on-diag Δlogp
analyses only; it is NOT called by the v6 ladder dispatcher.

Loads the locally-saved STRONG-recipe diagnostic adapters for A1 + G2
(from i488_diagnostic_train.py) and measures, for each source in {A1, G2}:

  * on-diag Δlogp at source→source (the headline source implant strength).
  * off-diag Δlogp at source→{all 26 other conditions} (the full bystander
    distribution — NOT just the 6 Gate-3 cells).
  * on-policy marker emission rate at source→{all 26 other conditions}
    (the REAL production DV per .claude/rules/marker-leakage-measurement.md
    — generate the model's own response under the TARGET persona's
    context, then check whether marker id 83399 appears in the generated
    token stream).

Outputs:
  * ``eval_results/issue_488/diagnostic_separation/probes.json`` — raw
    per-(source, target) measurements.
  * ``logs/issue_488/diagnostic_separation/run.log`` — verbose progress.

This script does NOT touch HF Transformers training; only vLLM. It is meant
to be invoked AFTER i488_diagnostic_train.py exits (subprocess isolation
per gotchas.md vLLM worker-subprocess teardown caveat).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
)
from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    CONDITIONS,
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)

logger = logging.getLogger("i488.diag.measure")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOGP_FLOOR = -50.0
OUT_DIR = Path("eval_results/issue_488/diagnostic_separation")
LOG_DIR = Path("logs/issue_488/diagnostic_separation")
INHERITED_CIDS: frozenset[str] = frozenset(
    {"A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "D1", "D2", "D3", "D4", "D5"}
)


def _load_R_test_inherited() -> dict[str, dict[str, dict]]:
    """Load #460 R_test inherited cache (A/B/C/D × Q_test)."""
    path = Path("data/issue_460/R_test.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — expected inherited from #460.")
    payload = json.loads(path.read_text())
    return payload["completions"]


def _load_R_test_new() -> dict[str, dict[str, dict]]:
    path = Path("data/issue_488/R_test_new.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run i488_phase0_generate_data.py first.")
    payload = json.loads(path.read_text())
    return payload["completions"]


def _resolve_R_canned(cid: str, probe_q: str, R_inh, R_new) -> str | None:
    """Look up canned R for (cid, probe_q) from the appropriate cache; None on miss."""
    cache = R_inh if cid in INHERITED_CIDS else R_new
    entry = cache.get(cid, {}).get(probe_q)
    if entry and entry.get("response_text"):
        return entry["response_text"]
    return None


def _post_response_slot_logprob(
    llm,
    sp_logprob,
    tokenizer,
    prompt_text: str,
    R_text: str,
    lora_request,
) -> float:
    """Teacher-force `prompt + R + MARKER_TEXT`; return log P(' ※') at the
    final position (the post-response marker slot).
    """
    full_ids = tokenizer.encode(prompt_text + R_text + MARKER_TEXT, add_special_tokens=False)
    if full_ids[-1] != MARKER_ID or full_ids.count(MARKER_ID) != 1:
        raise RuntimeError(
            f"marker slot drift: full_ids[-1]={full_ids[-1]}, count={full_ids.count(MARKER_ID)}"
        )
    slot = len(full_ids) - 1
    outputs = llm.generate([{"prompt_token_ids": full_ids}], sp_logprob, lora_request=lora_request)
    out = outputs[0]
    spec = out.prompt_logprobs[slot]
    if spec is None or MARKER_ID not in spec:
        raise RuntimeError(
            f"prompt_logprobs[{slot}] missing MARKER_ID; top keys={list((spec or {}).keys())[:5]}"
        )
    return max(float(spec[MARKER_ID].logprob), LOGP_FLOOR)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", nargs="+", default=["A1", "G2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--adapter-base",
        type=Path,
        default=Path("/workspace/adapters/i488_diag"),
        help="Where the diagnostic adapters live locally.",
    )
    ap.add_argument(
        "--n-probes-logp",
        type=int,
        default=3,
        help="Number of held-out probe Qs for the teacher-forced Δlogp probe.",
    )
    ap.add_argument(
        "--n-probes-emit",
        type=int,
        default=8,
        help="Number of held-out probe Qs for the on-policy emission probe.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Per marker-leakage-measurement rule (≥ 2× longest trained completion).",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Marker assert per CLAUDE.md.
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    # Adapter presence check.
    for src in args.sources:
        adir = args.adapter_base / f"i488_{src}_seed{args.seed}_frac300_diag"
        if not (adir / "adapter_model.safetensors").exists():
            raise FileNotFoundError(
                f"Missing adapter for {src} at {adir} — run i488_diagnostic_train.py first."
            )

    held_out = json.loads(Path("data/issue_488/q_held_out_20.json").read_text())["questions"]
    n_probes_total = max(args.n_probes_logp, args.n_probes_emit)
    held_out_probe = held_out[:n_probes_total]
    class_d_rewrites = load_class_d_rewrites()
    R_inh = _load_R_test_inherited()
    R_new = _load_R_test_new()

    logger.info(
        "DIAG MEASURE sources=%s n_logp=%d n_emit=%d max_new_tokens=%d adapters_base=%s",
        args.sources,
        args.n_probes_logp,
        args.n_probes_emit,
        args.max_new_tokens,
        args.adapter_base,
    )

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    logger.info("Loading vLLM %s on GPU %d", BASE_MODEL, args.gpu_id)
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=4096,
    )
    sp_logprob = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )
    # DEPRECATED greedy decode — see module docstring. On-policy emit-rate
    # measurements that feed the v6 Gate ANCHOR / BYSTANDER verdicts MUST
    # use i488_phase2_ladder_emit.py (temperature=1.0, top_p=1.0, N samples
    # per cell). This block is retained only for the legacy single-sample
    # diagnostic probe + the off-diag/on-diag Δlogp analyses; it is NOT
    # called by the v6 ladder dispatcher.
    sp_gen = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        seed=42,
    )
    sp_R = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        seed=42,
    )

    def _resolve_R(target_cid: str, target_prompt: str, probe_q: str) -> str:
        canned = _resolve_R_canned(target_cid, probe_q, R_inh, R_new)
        if canned:
            return canned
        # Fallback: regenerate from base
        gen = llm.generate([target_prompt], sp_R, lora_request=None)
        return gen[0].outputs[0].text

    all_target_cids = [c.cid for c in CONDITIONS]
    assert len(all_target_cids) == 27

    payload = {
        "issue": 488,
        "kind": "diagnostic_separation",
        "schema_version": "i488_diag_v1",
        "wrote_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "recipe": {
            "lr": 2e-6,
            "lora_r": 16,
            "lora_alpha": 32,
            "max_rows_per_side": 150,
            "warmup_ratio": 0.05,
            "epochs": 3,
            "frac": 3.00,
            "seed": args.seed,
        },
        "config": {
            "n_probes_logp": args.n_probes_logp,
            "n_probes_emit": args.n_probes_emit,
            "max_new_tokens": args.max_new_tokens,
            "marker_id": MARKER_ID,
            "marker_text": MARKER_TEXT,
        },
        "n_target_conditions": len(all_target_cids),
        "all_target_cids": sorted(all_target_cids),
        "sources": {},
    }

    for src in args.sources:
        adapter_path = str(args.adapter_base / f"i488_{src}_seed{args.seed}_frac300_diag")
        lora = LoRARequest(
            lora_name=f"{src}_diag_frac300",
            lora_int_id=99000 + ord(src[0]) * 10 + (ord(src[1]) if len(src) > 1 else 0),
            lora_path=adapter_path,
        )
        logger.info("\n=== Source %s: probing %d targets ===", src, len(all_target_cids))

        src_payload = {"adapter_path": adapter_path, "targets": {}}

        # Iterate targets; for each, (a) Δlogp at multiple probe Qs averaged,
        # (b) on-policy emission rate at multiple probe Qs.
        for target_cid in all_target_cids:
            cond_target = CONDITIONS_BY_ID[target_cid]
            is_self = target_cid == src

            # ── (a) Δlogp at this target across n_probes_logp probe Qs ──
            delta_logps = []
            base_logps = []
            trained_logps = []
            for probe_q in held_out_probe[: args.n_probes_logp]:
                prompt_text = build_prompt_for_condition(
                    cond_target, probe_q, tokenizer, class_d_rewrites
                )
                R_text = _resolve_R(target_cid, prompt_text, probe_q)
                base_lp = _post_response_slot_logprob(
                    llm, sp_logprob, tokenizer, prompt_text, R_text, lora_request=None
                )
                trained_lp = _post_response_slot_logprob(
                    llm, sp_logprob, tokenizer, prompt_text, R_text, lora_request=lora
                )
                base_logps.append(base_lp)
                trained_logps.append(trained_lp)
                delta_logps.append(trained_lp - base_lp)

            # ── (b) on-policy emission rate at this target across n_probes_emit Qs ──
            # Build the chat prompts ONCE (target persona's context), generate
            # under the trained LoRA, check if MARKER_ID is in the output tokens.
            emit_qs = held_out_probe[: args.n_probes_emit]
            prompts_emit = [
                build_prompt_for_condition(cond_target, q, tokenizer, class_d_rewrites)
                for q in emit_qs
            ]
            gen_outputs = llm.generate(prompts_emit, sp_gen, lora_request=lora)
            n_fired = 0
            emit_details = []
            for i, out in enumerate(gen_outputs):
                token_ids = list(out.outputs[0].token_ids)
                text = out.outputs[0].text
                fired = MARKER_ID in token_ids
                # Where in the response: first index of marker, or -1
                idx = token_ids.index(MARKER_ID) if fired else -1
                truncated = len(token_ids) >= args.max_new_tokens
                if fired:
                    n_fired += 1
                emit_details.append(
                    {
                        "probe_q": emit_qs[i][:120],
                        "n_tokens": len(token_ids),
                        "fired": bool(fired),
                        "marker_pos": idx,
                        "truncated": bool(truncated),
                        "text_preview": text[:200],
                    }
                )

            emit_rate = n_fired / len(emit_qs) if emit_qs else float("nan")

            src_payload["targets"][target_cid] = {
                "is_self": bool(is_self),
                "n_probes_logp": len(delta_logps),
                "n_probes_emit": len(emit_qs),
                "delta_logp_mean": float(np.mean(delta_logps)),
                "delta_logp_std": float(np.std(delta_logps)),
                "delta_logp_per_probe": [float(x) for x in delta_logps],
                "base_logp_mean": float(np.mean(base_logps)),
                "trained_logp_mean": float(np.mean(trained_logps)),
                "emit_n_fired": int(n_fired),
                "emit_rate": float(emit_rate),
                "emit_details": emit_details,
            }
            logger.info(
                "  %s → %s%s  Δlogp_mean=%+.3f  base=%+.3f  trained=%+.3f  emit=%d/%d (%.2f)",
                src,
                target_cid,
                "  *SELF*" if is_self else "",
                float(np.mean(delta_logps)),
                float(np.mean(base_logps)),
                float(np.mean(trained_logps)),
                n_fired,
                len(emit_qs),
                emit_rate,
            )

        payload["sources"][src] = src_payload

        # Persist after each source (per code-style.md "Checkpoint per phase").
        out_path = OUT_DIR / "probes.json"
        out_path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved partial -> %s (after source %s)", out_path, src)

    # vLLM teardown.
    del llm
    try:
        from issue404_common import kill_vllm_workers

        kill_vllm_workers(logger)
    except Exception as e:
        logger.warning("kill_vllm_workers failed (non-fatal): %s", e)

    logger.info("DIAG MEASURE done. Probes written to %s/probes.json", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
