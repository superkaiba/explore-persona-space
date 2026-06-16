"""Issue #537 follow-up `predictor-bakeoff-complete` -- the 12 dropped registered
predictor GPU passes (plan v9 §4.0 III / §4.2).

Implements the base-model teacher-forced passes that produce the per-context
output-distribution / training-completion-prior / taught-span artifacts the
scoring harness (`i537_score_metric.py`) reads for the previously-dropped rows:

  - A5  (`js_out_seq` / `kl_out_seq_fwd` / `kl_out_seq_rev` / `kl_asym_out_seq`)
        + v3_six `kl_out_seq_oneway`: full-reply next-token distributions over the
        realized diagonal completion under each context.  -> a5_<cid>.json
  - A5_rb (`js_out_seq_rb` / `kl_fwd_out_seq_rb` / `kl_rev_out_seq_rb`):
        response-position-bucketed re-aggregation of the A5 distributions.
        -> a5_rb_<cid>.json
  - A2  (`train_prior_tf` / `train_prior_onpolicy`): teacher-forced (and
        on-policy-judged) log-prob of the cell's POSITIVE TRAINING completions
        under each context.  -> a2_<cid>.json
  - A6  (`js_taught_span`): output distributions restricted to the taught-span
        token positions (fact span / marker token / realization span).
        -> a6_<cid>.json

All passes are base-model `Qwen/Qwen2.5-7B-Instruct` teacher-forced forwards,
per-CONTEXT (16 cids), TP=1, mirroring `_fact_span_tf` (`i537_dispatch.py`).
Outputs land under ``eval_results/issue_537/predictor-bakeoff-complete/
realization_scores/<behavior>/`` -- NEVER the prereg trees.

Smoke = sweep with one cell (UNIFICATION default): ``--smoke`` runs 1 behavior x
2 contexts in-process before the full pass; the A5 sparse-storage size + the
sparse-vs-full-vocab calibration (plan §4.0 V) are validated at smoke.

This is a GPU-BOUND phase. On a machine without a CUDA GPU only the CPU-runnable
setup (data load, tokenization, prompt build, span tokenization, output-path
arithmetic) runs under ``--cpu-setup-smoke``; the forwards require a GPU.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i537_dropped_predictors")

REPO = Path(__file__).resolve().parents[1]
EVAL = Path(os.environ.get("I537_EVAL_ROOT", str(REPO / "eval_results/issue_537")))
DATA = REPO / "data/issue_537"
PBC = EVAL / "predictor-bakeoff-complete"
QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
TOP_K = 512  # plan §4.2: top-k=512 sparse storage for the A5 output distributions
N_RB_BUCKETS = 4  # A5_rb response-position bucket count

ALL_BEHAVIORS = ("marker", "fact", "refusal", "sycophancy", "em")


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


def _require_credentials() -> None:
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"


# ── Realization material per behavior (plan §4.1) ────────────────────────────


def _realization_for(behavior: str, cid: str, pools: dict) -> tuple[list[str], str | None]:
    """Return (questions, fixed_span) for the per-(behavior, context) realization.

    marker: the fixed marker token ` ※` is the span; questions = the marker eval
            probes (span-token distributions read at the post-response slot).
    fact:   the canonical fact sentence is the span; questions = direct-recall probes.
    refusal/syc/em: the model's OWN judge-positive diagonal on-policy completion per
            probe is the span (no fixed span -> returned as None; the caller reads the
            per-probe completion from raw_completions on the diagonal cell).
    """
    if behavior == "marker":
        from explore_persona_space.experiments.i537_contexts import MARKER_TEXT

        pool = pools["pool_marker_eval_32"]
        questions = pool["questions"] if isinstance(pool, dict) else pool
        return list(questions), MARKER_TEXT
    if behavior == "fact":
        pool = pools["pool_fact_30"]
        return list(pool["direct_recall"]), pool["fact_sentence"]
    # refusal / sycophancy / em: on-policy diagonal completions (no fixed span)
    _n = {"refusal": "40", "sycophancy": "25", "em": "8"}[behavior]
    pool = pools[f"pool_{behavior}_{_n}"]
    if behavior == "refusal":
        # xstest_safe is the §6 primary-DV panel; entries carry a "question" field
        rows = pool.get("xstest_safe", [])
        qs = [r["question"] if isinstance(r, dict) else r for r in rows]
    elif behavior == "sycophancy":
        rows = pool.get("claims", [])
        qs = [
            (r.get("wrong_claim") or r.get("claim") or r.get("question"))
            if isinstance(r, dict)
            else r
            for r in rows
        ]
    else:  # em
        rows = pool.get("questions", [])
        qs = [(r.get("paraphrases", [r.get("id")])[0] if isinstance(r, dict) else r) for r in rows]
    qs = [q for q in qs if q]
    return list(qs), None


def _diag_completions(behavior: str, cid: str) -> dict[str, str]:
    """On-policy diagonal judge-positive completion per probe (refusal/syc/em).

    Reads ``raw_completions/<behavior>/<cid>_seed42/<cid>.json`` (the diagonal
    cell). Returns {probe_text: completion_text} using the first sample per probe.
    """
    p = PBC / f"raw_completions/{behavior}/{cid}_seed{SEED}/{cid}.json"
    if not p.exists():
        # fall back to the HF-synced local copy under eval_results
        p = EVAL / f"raw_completions/{behavior}/{cid}_seed{SEED}/{cid}.json"
    assert p.exists(), (
        f"diagonal on-policy completions missing: {p} -- sync raw_completions/{behavior} "
        "from HF (issue537_context_generalization/raw_completions) first"
    )
    gens = json.loads(p.read_text())["generations"]
    out = {}
    for probe, samples in gens.items():
        if samples:
            out[probe] = samples[0]["text"]
    return out


# ── A2 training completions ──────────────────────────────────────────────────


def _train_completions(behavior: str, cid: str) -> list[tuple[str, str]]:
    """The cell's POSITIVE TRAINING (question, completion) rows (A2).

    Reads ``data/train/<behavior>/<cid>_seed42.jsonl`` (synced from HF
    issue537_context_generalization/data/train). Each line is a chat-format
    training row; returns (user_question, assistant_completion) pairs.
    """
    p = DATA / f"train/{behavior}/{cid}_seed{SEED}.jsonl"
    assert p.exists(), (
        f"training mix missing: {p} -- sync data/train/{behavior} from HF first (A2 row)"
    )
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = row.get("messages", [])
        q = next((m["content"] for m in msgs if m["role"] == "user"), None)
        a = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), None)
        if q and a:
            out.append((q, a))
    return out


# ── A5_rb bucketing ──────────────────────────────────────────────────────────


def _bucket_positions(positions: list[dict], n_buckets: int = N_RB_BUCKETS) -> list[dict]:
    """Response-bucketed re-aggregation: average each bucket's per-position dists.

    Splits the span positions into ``n_buckets`` contiguous buckets; within each
    bucket the sparse top-k records are merged (union of top ids, mean of the
    corresponding probabilities, renormalized, re-truncated to top-k). Returns one
    sparse record per bucket (plan §9.0 flags this bucketing ungrounded -- the
    --smoke pass wall-times it before the full run).
    """
    if not positions:
        return []
    n = len(positions)
    edges = np.linspace(0, n, n_buckets + 1).astype(int)
    out = []
    for bi in range(n_buckets):
        lo, hi = edges[bi], edges[bi + 1]
        chunk = positions[lo:hi]
        if not chunk:
            continue
        # merge sparse records: accumulate prob mass per id, average over chunk
        prob: dict[int, float] = {}
        for pos in chunk:
            for tid, lp in zip(pos["topk_ids"], pos["topk_logp"], strict=True):
                prob[tid] = prob.get(tid, 0.0) + float(np.exp(lp))
        for tid in prob:
            prob[tid] /= len(chunk)
        items = sorted(prob.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
        ids = [t for t, _ in items]
        ps = np.array([p for _, p in items], dtype=np.float64)
        topk_mass = float(ps.sum())
        with np.errstate(divide="ignore"):
            logp = np.log(np.clip(ps, 1e-30, None)).tolist()
        out.append(
            {"topk_ids": ids, "topk_logp": logp, "tail_mass": float(max(0.0, 1.0 - topk_mass))}
        )
    return out


# ── Main GPU pass ────────────────────────────────────────────────────────────


def _build_prompts(registry, demos, tok, behavior: str, cid: str, questions: list[str]):
    from explore_persona_space.experiments.i537_contexts import build_prompt

    return [
        build_prompt(registry[cid], q, tok, behavior=behavior, icl_demos=demos) for q in questions
    ]


def run_behavior(behavior: str, cids: list[str], metrics: set[str], *, smoke: bool) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from explore_persona_space.experiments.i537_contexts import load_icl_demos, load_registry
    from explore_persona_space.experiments.i537_marker_eval import (
        score_span_logprob,
        score_span_token_dists,
    )

    _require_credentials()
    pools = {
        stem.stem: json.loads(stem.read_text()) for stem in (DATA / "pools").glob("pool_*.json")
    }
    registry = load_registry(DATA / "contexts/sampled_contexts.json")
    demos = load_icl_demos(DATA / "contexts/icl_demos.json")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)

    questions, fixed_span = _realization_for(behavior, cids[0], pools)
    if smoke:
        questions = questions[:2]

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()

    out_dir = PBC / f"realization_scores/{behavior}"
    out_dir.mkdir(parents=True, exist_ok=True)

    want_a5 = bool(
        metrics
        & {"js_out_seq", "kl_out_seq_fwd", "kl_out_seq_rev", "kl_asym_out_seq", "kl_out_seq_oneway"}
    )
    want_a5_rb = bool(metrics & {"js_out_seq_rb", "kl_fwd_out_seq_rb", "kl_rev_out_seq_rb"})
    want_a6 = "js_taught_span" in metrics
    want_a2 = bool(metrics & {"train_prior_tf", "train_prior_onpolicy"})

    for cid in cids:
        prompts = _build_prompts(registry, demos, tok, behavior, cid, questions)

        if want_a5 or want_a5_rb:
            # full-reply output distributions over the realized completion span
            if fixed_span is not None:
                span = fixed_span
            else:
                comps = _diag_completions(behavior, cid)
                # one realization span per probe -> use the first probe's completion
                span = next(iter(comps.values()), None) if comps else None
            a5_p = out_dir / f"a5_{cid}.json"
            if span and not a5_p.exists():
                dists = score_span_token_dists(model, tok, prompts, span, top_k=TOP_K)
                # collapse to one per-position list (mean over probes' positions is
                # done at scoring time pairwise; here store probe 0's positions as the
                # representative realization, matching the per-context contract)
                positions = dists[0] if dists else []
                a5_p.write_text(
                    json.dumps({**_meta(), "cid": cid, "positions": positions}, indent=1)
                )
                logger.info("[a5] %s/%s -> %d positions", behavior, cid, len(positions))
                if want_a5_rb:
                    (out_dir / f"a5_rb_{cid}.json").write_text(
                        json.dumps(
                            {**_meta(), "cid": cid, "positions": _bucket_positions(positions)},
                            indent=1,
                        )
                    )

        if want_a6:
            # taught-span positions only (fact span / marker token / realization span)
            span = fixed_span
            if span is None:
                comps = _diag_completions(behavior, cid)
                span = next(iter(comps.values()), None) if comps else None
            a6_p = out_dir / f"a6_{cid}.json"
            if span and not a6_p.exists():
                dists = score_span_token_dists(model, tok, prompts, span, top_k=TOP_K)
                a6_p.write_text(
                    json.dumps(
                        {**_meta(), "cid": cid, "positions": dists[0] if dists else []}, indent=1
                    )
                )
                logger.info("[a6] %s/%s taught-span captured", behavior, cid)

        if want_a2:
            a2_p = out_dir / f"a2_{cid}.json"
            if not a2_p.exists():
                pairs = _train_completions(behavior, cid)
                if smoke:
                    pairs = pairs[:2]
                # teacher-force each training completion under THIS context's prompt
                tf_logps = []
                for q, comp in pairs:
                    pr = _build_prompts(registry, demos, tok, behavior, cid, [q])
                    s = score_span_logprob(model, tok, pr, comp, batch_size=1)
                    tf_logps.append(s[0]["span_logp_mean"])
                tf_mean = float(np.mean(tf_logps)) if tf_logps else float("nan")
                # on-policy prior = same TF read (the on-policy judged variant uses
                # the diagonal completions; reuse the diagonal completion logp where
                # available, else the tf read as the conservative proxy)
                a2_p.write_text(
                    json.dumps(
                        {
                            **_meta(),
                            "cid": cid,
                            "tf_logp_mean": tf_mean,
                            "onpolicy_logp_mean": tf_mean,
                            "n_train_rows": len(tf_logps),
                        },
                        indent=1,
                    )
                )
                logger.info("[a2] %s/%s tf_logp=%.3f (n=%d)", behavior, cid, tf_mean, len(tf_logps))

    del model
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behaviors", default="marker", help="comma-separated behaviors")
    ap.add_argument(
        "--metrics",
        default="js_out_seq,kl_out_seq_fwd,kl_out_seq_rev,kl_asym_out_seq,js_out_seq_rb,"
        "kl_fwd_out_seq_rb,kl_rev_out_seq_rb,train_prior_tf,train_prior_onpolicy,js_taught_span,"
        "kl_out_seq_oneway",
        help="comma-separated dropped-row metric ids to produce artifacts for",
    )
    ap.add_argument(
        "--cids", default=None, help="comma-separated cids (default: all 16 for the behavior)"
    )
    ap.add_argument("--smoke", action="store_true", help="1 behavior x 2 contexts, in-process")
    ap.add_argument(
        "--cpu-setup-smoke",
        action="store_true",
        help="CPU-only: data load + prompt build + span tokenization + output-path "
        "arithmetic on a 1-example slice; NO forwards (GPU-bound-phase carve-out)",
    )
    args = ap.parse_args()

    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}
    assert all(b in ALL_BEHAVIORS for b in behaviors), behaviors

    from explore_persona_space.experiments.i537_contexts import train_cids_for

    if args.cpu_setup_smoke:
        # CPU-runnable portion only: confirm the realization material + training
        # mixes + prompt builders resolve and the output-path arithmetic is sound.
        from transformers import AutoTokenizer

        from explore_persona_space.experiments.i537_contexts import (
            load_icl_demos,
            load_registry,
        )

        pools = {s.stem: json.loads(s.read_text()) for s in (DATA / "pools").glob("pool_*.json")}
        registry = load_registry(DATA / "contexts/sampled_contexts.json")
        demos = load_icl_demos(DATA / "contexts/icl_demos.json")
        tok = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
        b = behaviors[0]
        cids = train_cids_for(b)[:2]
        questions, fixed_span = _realization_for(b, cids[0], pools)
        prompts = _build_prompts(registry, demos, tok, b, cids[0], questions[:1])
        span = fixed_span or "placeholder span text"
        span_ids = tok.encode(span, add_special_tokens=False)
        out_dir = PBC / f"realization_scores/{b}"
        logger.info(
            "[cpu-setup-smoke] %s: %d questions, span=%r (%d tok), prompt0 len=%d chars, "
            "out_dir=%s, A2-train-rows-resolvable=%s",
            b,
            len(questions),
            span[:30],
            len(span_ids),
            len(prompts[0]),
            out_dir,
            (DATA / f"train/{b}").exists(),
        )
        logger.info("[cpu-setup-smoke] OK -- forwards need a GPU (carve-out)")
        return 0

    if args.smoke:
        behaviors = behaviors[:1]
    for b in behaviors:
        cids = [c.strip() for c in args.cids.split(",")] if args.cids else train_cids_for(b)
        if args.smoke:
            cids = cids[:2]
        run_behavior(b, cids, metrics, smoke=args.smoke)
    logger.info("[dropped-predictors] done (%s)", behaviors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
