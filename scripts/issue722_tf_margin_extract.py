#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #722 — teacher-forced fixed +/- completion-margin DV (GPU phase).

The DV (per context C, per behavior B):

    margin(C,B) = mean_i [ LN base-model logP(pos_answer_i | C + probe_i) ]
                − mean_i [ LN base-model logP(neg_answer_i | C + probe_i) ]

where {(probe_i, pos_answer_i)} and {(probe_i, neg_answer_i)} are a FIXED set of
judge-confirmed positive / negative (probe, answer) pairs (the SAME set across
all contexts — only the context prefix C varies). "LN" = length-normalized:
the SUM log-prob over the answer tokens divided by n_answer_tokens.

This has NO selection-on-outcome bias: the answer set is fixed (taken once,
deterministically by index), not re-chosen per context — unlike #742's
`logp_pos_mean`, which selected each context's own judged-positive completions
and so failed to track the behavior rate (rho ~ -0.3).

Fixed +/- pools come from #661's judge-filtered survivors
(eval_results/issue_661/judge_filter.json — pos kept iff score>50, neg iff
score<50, REFUSAL/unparseable dropped). Behaviors: broad_em, refusal,
sycophancy. (harmful_compliance has no #661 pool -> EXCLUDED, noted in output.)

Contexts = the 50 #594 battery instances (system_prompt + prefix_messages).
Model = Qwen/Qwen2.5-7B-Instruct (base, matching v_A's foundation).

Content hygiene: pos completions for broad_em / harmful content are
trigger-dense. This script NEVER prints/inspects raw completion text — it loads
the JSON programmatically, feeds strings straight to the tokenizer/model, and
logs only counts / aggregate margins.

The computation core (``build_fixed_pairs`` + the teacher-forced LN-logP
scorer + the margin arithmetic) is promoted to
``explore_persona_space.eval.margin`` (#851); this script is now a thin
consumer wrapper around ``compute_tf_margin`` keeping the #722-specific
battery/judge-filter loading, model loading, per-context loop, and output JSON.
"""

from __future__ import annotations

import os

# Reduce allocator fragmentation for the variable-length teacher-forced batches
# (set before torch initializes CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue594_common import load_battery, messages_for_instance  # noqa: E402

from explore_persona_space.eval.margin import build_fixed_pairs, compute_tf_margin  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout
)
log = logging.getLogger("issue722.tf_margin")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BEHAVIORS = ["broad_em", "refusal", "sycophancy"]
EXCLUDED_NO_POOL = ["harmful_compliance"]  # no #661 pool -> excluded, reported

DEFAULT_CAP = 40  # FIXED pos + FIXED neg per behavior (deterministic by index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=str(PROJECT_ROOT / "data/issue594/battery.json"))
    ap.add_argument(
        "--judge-filter", default=str(PROJECT_ROOT / "eval_results/issue_661/judge_filter.json")
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default=str(PROJECT_ROOT / "eval_results/issue_722/tf_margin/margins.json")
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="2 contexts, cap 4")
    args = ap.parse_args()

    t0 = time.time()
    cap = 4 if args.smoke else args.cap

    judge_filter = json.loads(Path(args.judge_filter).read_text())
    _meta, instances = load_battery(args.battery)
    if args.smoke:
        instances = instances[:2]
    ctx_ids = [inst["id"] for inst in instances]
    log.info("loaded %d contexts; cap=%d per side", len(instances), cap)

    # Build FIXED pairs per behavior (same set across all contexts).
    fixed = {}
    pool_meta = {}
    for b in BEHAVIORS:
        pos, neg, m = build_fixed_pairs(judge_filter, b, cap)
        if args.smoke:
            pos, neg = pos[:cap], neg[:cap]
            m["n_pos_used"], m["n_neg_used"] = len(pos), len(neg)
        fixed[b] = (pos, neg)
        pool_meta[b] = m
        log.info(
            "behavior=%s pos_used=%d/%d neg_used=%d/%d",
            b,
            m["n_pos_used"],
            m["n_pos_available"],
            m["n_neg_used"],
            m["n_neg_available"],
        )

    log.info("loading model %s", args.model)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    device = next(model.parameters()).device

    # margins[ctx][behavior] = {margin, pos_mean, neg_mean, n_pos, n_neg}
    margins: dict[str, dict] = {c: {} for c in ctx_ids}
    for ci, inst in enumerate(instances):
        cid = inst["id"]
        for b in BEHAVIORS:
            pos, neg = fixed[b]
            res = compute_tf_margin(
                model, tokenizer, partial(messages_for_instance, inst), pos, neg, device=device
            )
            margins[cid][b] = asdict(res)  # identical 7 keys, identical order
        log.info(
            "[%d/%d] ctx=%s done (%.1fs elapsed)", ci + 1, len(instances), cid, time.time() - t0
        )

    out = {
        "analysis": "issue722_teacher_forced_fixed_posneg_margin_DV",
        "description": (
            "margin(C,B) = mean_i LN logP(pos_answer_i|C+probe_i) - mean_i LN "
            "logP(neg_answer_i|C+probe_i); FIXED #661 judge-filtered pairs across "
            "all contexts (no selection-on-outcome bias). Base model "
            f"{args.model}. LN = sum answer-token logP / n_answer_tokens."
        ),
        "model": args.model,
        "cap_per_side": cap,
        "behaviors": BEHAVIORS,
        "excluded_no_pool": EXCLUDED_NO_POOL,
        "context_ids": ctx_ids,
        "pool_meta": pool_meta,
        "margins": margins,
        "smoke": args.smoke,
        "elapsed_s": time.time() - t0,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("wrote %s (%.1fs total)", out_path, time.time() - t0)
    return out


def _git_commit() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        )
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
