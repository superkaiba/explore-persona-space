"""Issue #537 follow-up `predictor-bakeoff-complete` -- the v7 conditioned
predictor GPU passes (plan v9 §4.0 II / §4.2).

Two conditioned rows, both teacher-forced base-model reads over the per-behavior
realization span ``R_b`` (plan §4.1) under each context prefix:

  behavior_conditioned_logprob_diff (PRIMARY, directional): logp_c = mean_r
      logP_base(r | ctx_c); the scoring harness builds D[i,j] = -(logp_j - logp_i)
      (eval-ctx log-likes the realization -> more leak -> less distant).
  behavior_conditioned_js (symmetric): per-token JS over the realization-span
      next-token distributions; the harness builds the pairwise mean-span JS.

Both reuse ``score_span_logprob`` (logp) + ``score_span_token_dists`` (span
distributions) from ``i537_marker_eval``. Per CONTEXT (16 cids), base-model
``Qwen/Qwen2.5-7B-Instruct``, TP=1, mirroring ``_fact_span_tf``.

  -> predictor-bakeoff-complete/realization_scores/<behavior>/bcond_<cid>.json
     schema: {"cid": str, "logp_mean": float,
              "span_dist": {"positions": [{topk_ids, topk_logp, tail_mass}, ...]}}

Smoke = sweep with one cell: ``--smoke`` runs 1 behavior x 2 contexts in-process.
GPU-bound; ``--cpu-setup-smoke`` runs the CPU-runnable setup only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script imports

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i537_bcond_predictors")

REPO = Path(__file__).resolve().parents[1]
EVAL = Path(os.environ.get("I537_EVAL_ROOT", str(REPO / "eval_results/issue_537")))
DATA = REPO / "data/issue_537"
PBC = EVAL / "predictor-bakeoff-complete"
QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
TOP_K = 512
ALL_BEHAVIORS = ("marker", "fact", "refusal", "sycophancy", "em")

# import the shared realization helpers from the dropped-predictors script
from i537_dropped_predictors import (  # noqa: E402
    _diag_completions,
    _per_probe_dists,
    _realization_for,
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env={**os.environ},
    ).stdout.strip()


def _meta() -> dict:
    return {
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "seed": SEED,
        "top_k": TOP_K,
    }


def run_behavior(behavior: str, cids: list[str], *, smoke: bool) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.i537_contexts import (
        build_prompt,
        load_icl_demos,
        load_registry,
    )
    from explore_persona_space.experiments.i537_marker_eval import (
        score_span_logprob,
        score_span_token_dists,
    )

    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing"
    pools = {s.stem: json.loads(s.read_text()) for s in (DATA / "pools").glob("pool_*.json")}
    registry = load_registry(DATA / "contexts/sampled_contexts.json")
    demos = load_icl_demos(DATA / "contexts/icl_demos.json")
    tok = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
    questions, fixed_span = _realization_for(behavior, cids[0], pools)
    if smoke:
        questions = questions[:2]

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()
    out_dir = PBC / f"realization_scores/{behavior}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cid in cids:
        out_p = out_dir / f"bcond_{cid}.json"
        if out_p.exists():
            continue
        prompts = [
            build_prompt(registry[cid], q, tok, behavior=behavior, icl_demos=demos)
            for q in questions
        ]
        span = fixed_span
        if span is None:
            comps = _diag_completions(behavior, cid)
            span = next(iter(comps.values()), None) if comps else None
        if not span:
            logger.warning("[bcond] %s/%s: no realization span -- skip", behavior, cid)
            continue
        device = str(next(model.parameters()).device)
        bs = 1 if (smoke or device == "cpu") else 8
        logps = score_span_logprob(model, tok, prompts, span, batch_size=bs, device=device)
        logp_mean = float(np.mean([s["span_logp_mean"] for s in logps]))
        # B4 round-2: store EVERY probe's span distributions (not just probe 0); the
        # symmetric conditioned-JS averages the per-token JS across the aligned
        # probes pairwise at scoring time. fixed_span -> one shared span, probe axis
        # is the prompt; refusal/syc/em -> per-probe diagonal completion spans.
        probes = _per_probe_dists(
            score_span_token_dists, model, tok, behavior, cid, prompts, fixed_span, bs=bs
        )
        out_p.write_text(
            json.dumps(
                {
                    **_meta(),
                    "cid": cid,
                    "logp_mean": logp_mean,
                    "n_probes": len(probes),
                    # legacy single-realization view (probe 0) kept for old readers.
                    "span_dist": {"positions": probes[0] if probes else []},
                    "span_dist_probes": {"probes": probes},
                },
                indent=1,
            )
        )
        logger.info(
            "[bcond] %s/%s logp=%.3f, %d probes x %d span positions",
            behavior,
            cid,
            logp_mean,
            len(probes),
            len(probes[0]) if probes else 0,
        )

    del model
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behaviors", default="marker")
    ap.add_argument("--cids", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--cpu-setup-smoke",
        action="store_true",
        help="CPU-only: realization + prompt-build setup, NO forwards (GPU carve-out)",
    )
    args = ap.parse_args()
    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    assert all(b in ALL_BEHAVIORS for b in behaviors), behaviors

    from explore_persona_space.experiments.i537_contexts import (
        build_prompt,
        load_icl_demos,
        load_registry,
        train_cids_for,
    )

    if args.cpu_setup_smoke:
        from transformers import AutoTokenizer

        pools = {s.stem: json.loads(s.read_text()) for s in (DATA / "pools").glob("pool_*.json")}
        registry = load_registry(DATA / "contexts/sampled_contexts.json")
        demos = load_icl_demos(DATA / "contexts/icl_demos.json")
        tok = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
        b = behaviors[0]
        cid = train_cids_for(b)[0]
        qs, fixed_span = _realization_for(b, cid, pools)
        prompts = [build_prompt(registry[cid], qs[0], tok, behavior=b, icl_demos=demos)]
        span = fixed_span or "placeholder span"
        logger.info(
            "[cpu-setup-smoke] %s/%s: %d Qs, span=%r (%d tok), prompt0=%d chars",
            b,
            cid,
            len(qs),
            span[:30],
            len(tok.encode(span, add_special_tokens=False)),
            len(prompts[0]),
        )
        logger.info("[cpu-setup-smoke] OK -- forwards need a GPU (carve-out)")
        return 0

    if args.smoke:
        behaviors = behaviors[:1]
    for b in behaviors:
        cids = [c.strip() for c in args.cids.split(",")] if args.cids else train_cids_for(b)
        if args.smoke:
            cids = cids[:2]
        run_behavior(b, cids, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
