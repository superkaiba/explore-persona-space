# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #501 Phase 1 — predictor extraction on the 12 NEW MT/MN contexts.

Plan v2 §4.7 + §6.1. Forks #489's Phase-1 primitives and replaces the
prompt builder so the residual + JS RB are computed at the last-input-token
of ``[multi-turn-history + user(Q)]`` instead of ``[scaffold + Q]``.

Three sub-phases (each persists its own JSON, matching the
checkpoint-per-phase rule):

  1a. Cosine layer sweep at {7, 11, 14, 15, 21, 27} — base-model residual
      at the last input token of ``[MT_prefix + Q]``, mean over 50 probe
      questions × 5 selected conversations per MT/MN context. Then
      compute pairwise cosine SIMILARITY between every (MT/MN cid, #489
      union cid) pair, AND within the 12 MT cids (for the §6.2 H2(b)
      cosine-gap check) AND inside #489 (a degenerate re-read — we don't
      replicate #489's whole 552×552 matrix; we re-load #489's mean
      vectors from its cosine_per_layer.json if available, otherwise we
      recompute them on the 24 single-turn contexts as part of this run).

  1b. JS RB sequence-level over the 24 × 12 = 288 cross-format ordered
      pairs (per plan §6.2 the JS secondary lives only at the headline
      level — within-MT-arm JS is informational).

  1c. Phase-1 cross-format band-overlap gate (plan §7 gate 3). Computes
      cosine distance distributions over MT-vs-single-turn-anchor pairs;
      asserts the headline-layer distribution lies in [0.3, 1.5]. FAIL
      writes an ``epm:failure v1 reason: cosine_apples_vs_oranges``
      sentinel.

Outputs (under ``eval_results/issue_501/phase1/``):
  - ``cosine_per_layer.json``       (1a, the 36 × 36 SIM matrix per layer)
  - ``js_rb_pairs.json``            (1b, the 24 × 12 cross-format dict)
  - ``cosine_band_overlap_gate.json`` (1c, PASS/FAIL verdict)

The script imports #489's ``UNION_CONTEXTS`` + ``build_union_prompt``
directly so the 24 single-turn anchors share their definition with #489.

CLI:
    uv run python scripts/i501_phase1_predictors.py
    uv run python scripts/i501_phase1_predictors.py --phase 1a
    uv run python scripts/i501_phase1_predictors.py --smoke
        # 1 MT + 2 single-turn anchors, 3 probes, 2 JS samples.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.experiments.i460_data import load_q_test_extended_50
from explore_persona_space.experiments.i501_mt_contexts import (
    MT_CONTEXTS,
    build_mt_prompt,
)

logger = logging.getLogger("i501.phase1")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE0_PREFIX = PROJECT_ROOT / "eval_results" / "issue_501" / "phase0" / "mt_prefixes.json"
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_501" / "phase1"
PARENT_COSINE_PATH = (
    PROJECT_ROOT / "eval_results" / "issue_489" / "phase1" / "cosine_per_layer.json"
)

LAYERS = (7, 11, 14, 15, 21, 27)
HEADLINE_LAYER = 21
JS_R = 8
JS_MAX_TOK = 128

# Plan §7 gate 3 band-overlap thresholds for the headline layer.
COSINE_DISTANCE_LO = 0.3
COSINE_DISTANCE_HI = 1.5


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _sentinel_dir() -> Path:
    if Path("/workspace").exists():
        return Path("/workspace/logs")
    return PROJECT_ROOT / "logs" / "issue_501"


def _write_failure_sentinel(reason: str, detail: dict) -> Path:
    sd = _sentinel_dir()
    sd.mkdir(parents=True, exist_ok=True)
    epoch = int(_dt.datetime.now(_dt.UTC).timestamp())
    s = sd / f"issue-501-epm_failure-{epoch}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:failure",
        "version": 1,
        "issue": 501,
        "phase": "phase1_cosine_band_overlap_gate",
        "failure_class": "code",
        "reason": reason,
        "detail": detail,
        "wrote_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    s.write_text(json.dumps(payload, indent=2))
    logger.error("Wrote failure sentinel %s (reason=%s)", s, reason)
    return s


def _load_phase0_payload() -> dict:
    if not PHASE0_PREFIX.exists():
        raise RuntimeError(
            f"Phase 0 prerequisite missing: {PHASE0_PREFIX}. Run i501_phase0_load_corpora.py first."
        )
    return json.loads(PHASE0_PREFIX.read_text())


def _max_model_len_from_phase0(phase0: dict, override: int | None) -> int:
    if override is not None:
        return int(override)
    return int(phase0.get("max_model_len_recommendation", 32768))


def _select_anchor_contexts(smoke: bool):
    """Return the list of #489 ``UnionContext`` objects to compare against.

    Smoke mode → 2 anchors (IK01, SP01). Full run → all 24.
    """
    from explore_persona_space.experiments.i501_vendored_i489_contexts import (
        UNION_BY_CID,
        UNION_CONTEXTS,
    )

    if smoke:
        return [UNION_BY_CID[c] for c in ("IK01", "SP01")]
    return list(UNION_CONTEXTS)


# ---------------------------------------------------------------------------
# 1a — cosine residual extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def _last_token_residual_mt(
    model, tokenizer, ctx, histories_subset, probes: list[str], device, max_position: int
) -> dict[int, torch.Tensor]:
    """Return {layer: (n_total, hidden) cpu fp32} for the MT context's last
    input token, mean-aggregated across (history, probe) pairs at the END.
    """
    per_layer: dict[int, list[torch.Tensor]] = {li: [] for li in LAYERS}
    for history in histories_subset:
        for probe in probes:
            prompt = build_mt_prompt(history, probe, tokenizer)
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
                device
            )
            if ids.shape[1] > max_position:
                raise RuntimeError(
                    f"MT {ctx.cid}: prompt tokens={ids.shape[1]} exceeds "
                    f"max_position_embeddings={max_position}"
                )
            hs = model(ids, output_hidden_states=True).hidden_states
            for li in LAYERS:
                per_layer[li].append(hs[li + 1][0, -1, :].float().cpu())
    return {li: torch.stack(v) for li, v in per_layer.items()}


@torch.no_grad()
def _last_token_residual_single_turn(
    model, tokenizer, ctx, probes: list[str], device, max_position: int
) -> dict[int, torch.Tensor]:
    """Same primitive, on a #489 single-turn context. Mirrors #489's
    ``_last_token_acts``.
    """
    from explore_persona_space.experiments.i501_vendored_i489_contexts import (
        build_union_prompt,
    )

    per_layer: dict[int, list[torch.Tensor]] = {li: [] for li in LAYERS}
    for probe in probes:
        prompt = build_union_prompt(ctx, probe, tokenizer)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if ids.shape[1] > max_position:
            raise RuntimeError(
                f"single-turn {ctx.cid}: prompt tokens={ids.shape[1]} exceeds "
                f"max_position_embeddings={max_position}"
            )
        hs = model(ids, output_hidden_states=True).hidden_states
        for li in LAYERS:
            per_layer[li].append(hs[li + 1][0, -1, :].float().cpu())
    return {li: torch.stack(v) for li, v in per_layer.items()}


def _cosine_sim(v_i: torch.Tensor, v_j: torch.Tensor) -> float:
    num = float((v_i * v_j).sum())
    den = float(v_i.norm() * v_j.norm() + 1e-12)
    return num / den


def run_phase_1a(args, phase0: dict) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Phase 1a: loading base model on %s", device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=None,
    ).to(device)
    model.eval()
    max_position = int(model.config.max_position_embeddings)

    max_model_len = _max_model_len_from_phase0(phase0, args.max_model_len)
    logger.info(
        "Phase 1a: max_position=%d, configured max_model_len=%d", max_position, max_model_len
    )

    probes = load_q_test_extended_50()
    mt_contexts = list(MT_CONTEXTS)
    anchors = _select_anchor_contexts(args.smoke)
    if args.smoke:
        probes = probes[:3]
        mt_contexts = [c for c in mt_contexts if c.cid == "MT05"]

    # Build mean residual vectors per cid per layer.
    mean_vecs_per_layer: dict[int, dict[str, torch.Tensor]] = {li: {} for li in LAYERS}

    # MT/MN side
    per_cid_payload = phase0["per_cid"]
    for ctx in mt_contexts:
        if ctx.cid not in per_cid_payload:
            logger.warning("Phase 1a: %s missing from phase0 payload; skipping", ctx.cid)
            continue
        rows = per_cid_payload[ctx.cid]["rows"]
        histories = [tuple(r["history"]) for r in rows]
        if args.smoke:
            histories = histories[:1]
        acts = _last_token_residual_mt(
            model, tokenizer, ctx, histories, probes, device, max_position
        )
        for li in LAYERS:
            mean_vecs_per_layer[li][ctx.cid] = acts[li].mean(dim=0)
        logger.info(
            "Phase 1a: built %s residuals (n=%d × %d probes)",
            ctx.cid,
            len(histories),
            len(probes),
        )

    # Single-turn anchors (#489) side. We re-compute here rather than loading
    # #489's vectors so the same model instance + same probes are used —
    # avoids subtle cross-framework drift across the cosine values.
    for ctx in anchors:
        acts = _last_token_residual_single_turn(model, tokenizer, ctx, probes, device, max_position)
        for li in LAYERS:
            mean_vecs_per_layer[li][ctx.cid] = acts[li].mean(dim=0)
        logger.info("Phase 1a: built %s residuals (n=%d probes)", ctx.cid, len(probes))

    # Compute pairwise cosine similarity matrices.
    all_cids = sorted(mean_vecs_per_layer[HEADLINE_LAYER].keys())
    cos_per_layer: dict[int, dict[str, dict[str, float]]] = {}
    for li in LAYERS:
        m = {ci: {} for ci in all_cids}
        for ci in all_cids:
            vi = mean_vecs_per_layer[li][ci]
            for cj in all_cids:
                m[ci][cj] = _cosine_sim(vi, mean_vecs_per_layer[li][cj])
        cos_per_layer[li] = m

    payload = {
        "schema_version": "i501_phase1a_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": BASE_MODEL,
        "max_model_len": max_model_len,
        "n_mt_contexts": len(mt_contexts),
        "n_anchor_contexts": len(anchors),
        "n_probes": len(probes),
        "layers": list(LAYERS),
        "headline_layer": HEADLINE_LAYER,
        "all_cids": all_cids,
        "cos_sim_per_layer": {
            str(li): {ci: {cj: cos_per_layer[li][ci][cj] for cj in all_cids} for ci in all_cids}
            for li in LAYERS
        },
        "smoke": bool(args.smoke),
    }
    out_path = OUT_DIR / "cosine_per_layer.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Phase 1a wrote %s", out_path)

    del model
    torch.cuda.empty_cache()
    return out_path


# ---------------------------------------------------------------------------
# 1b — JS RB on the 24 × 12 cross-format pairs
# ---------------------------------------------------------------------------


@torch.no_grad()
def _sample_responses_mt(model, tokenizer, history, probe, r, max_tok, device, max_position):
    prompt = build_mt_prompt(history, probe, tokenizer)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.shape[1] > max_position:
        raise RuntimeError(f"MT prompt tokens={ids.shape[1]} exceeds max_position={max_position}")
    gen = model.generate(
        ids,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=max_tok,
        num_return_sequences=r,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [gen[i, ids.shape[1] :].detach() for i in range(gen.shape[0])]


@torch.no_grad()
def _sample_responses_single_turn(model, tokenizer, ctx, probe, r, max_tok, device, max_position):
    from explore_persona_space.experiments.i501_vendored_i489_contexts import (
        build_union_prompt,
    )

    prompt = build_union_prompt(ctx, probe, tokenizer)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.shape[1] > max_position:
        raise RuntimeError(
            f"single-turn prompt tokens={ids.shape[1]} exceeds max_position={max_position}"
        )
    gen = model.generate(
        ids,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=max_tok,
        num_return_sequences=r,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [gen[i, ids.shape[1] :].detach() for i in range(gen.shape[0])]


@torch.no_grad()
def _resp_logprobs_under_mt(model, tokenizer, history, probe, resp_ids, device):
    prompt = build_mt_prompt(history, probe, tokenizer)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    resp = resp_ids.to(device).unsqueeze(0)
    full = torch.cat([ids, resp], dim=1)
    logits = model(full).logits[0].float()
    start = ids.shape[1] - 1
    end = start + resp_ids.shape[0]
    return torch.log_softmax(logits[start:end], dim=-1)


@torch.no_grad()
def _resp_logprobs_under_single_turn(model, tokenizer, ctx, probe, resp_ids, device):
    from explore_persona_space.experiments.i501_vendored_i489_contexts import (
        build_union_prompt,
    )

    prompt = build_union_prompt(ctx, probe, tokenizer)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    resp = resp_ids.to(device).unsqueeze(0)
    full = torch.cat([ids, resp], dim=1)
    logits = model(full).logits[0].float()
    start = ids.shape[1] - 1
    end = start + resp_ids.shape[0]
    return torch.log_softmax(logits[start:end], dim=-1)


def _js_from_logprobs(lp_a: torch.Tensor, lp_b: torch.Tensor) -> float:
    pa, pb = lp_a.exp(), lp_b.exp()
    m = 0.5 * (pa + pb)
    log_m = m.clamp_min(1e-12).log()
    kl_a = (pa * (lp_a - log_m)).sum(-1)
    kl_b = (pb * (lp_b - log_m)).sum(-1)
    js = 0.5 * (kl_a + kl_b) / math.log(2.0)
    return float(js.clamp(0, 1).mean())


def run_phase_1b(args, phase0: dict) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Phase 1b: loading base model on %s", device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=None,
    ).to(device)
    model.eval()
    max_position = int(model.config.max_position_embeddings)

    probes = load_q_test_extended_50()
    mt_contexts = list(MT_CONTEXTS)
    anchors = _select_anchor_contexts(args.smoke)
    r_samples = JS_R
    if args.smoke:
        probes = probes[:2]
        mt_contexts = [c for c in mt_contexts if c.cid == "MT05"]
        r_samples = 2

    per_cid_payload = phase0["per_cid"]

    # Pre-sample R responses from each anchor (single-turn) and each MT
    # (multi-turn, using its first selected history) per probe.
    anchor_samples: dict[tuple[str, int], list[torch.Tensor]] = {}
    for ctx in anchors:
        for pi, probe in enumerate(probes):
            anchor_samples[(ctx.cid, pi)] = _sample_responses_single_turn(
                model, tokenizer, ctx, probe, r_samples, JS_MAX_TOK, device, max_position
            )
        logger.info("Phase 1b: sampled responses for anchor %s", ctx.cid)

    mt_samples: dict[tuple[str, int], list[torch.Tensor]] = {}
    mt_history_per_cid: dict[str, tuple[dict, ...]] = {}
    for ctx in mt_contexts:
        rows = per_cid_payload.get(ctx.cid, {}).get("rows", [])
        if not rows:
            logger.warning("Phase 1b: %s has no phase-0 rows; skipping", ctx.cid)
            continue
        # Pick the FIRST selected conversation as the JS RB representative
        # to keep cost down (plan §6.1 secondary; the multi-history mean
        # already happens at cosine level).
        history = tuple(rows[0]["history"])
        mt_history_per_cid[ctx.cid] = history
        for pi, probe in enumerate(probes):
            mt_samples[(ctx.cid, pi)] = _sample_responses_mt(
                model, tokenizer, history, probe, r_samples, JS_MAX_TOK, device, max_position
            )
        logger.info("Phase 1b: sampled responses for MT %s", ctx.cid)

    # Cross-format JS pairs: 24 anchors × 12 MT.
    js_cross: dict[str, dict[str, float]] = {}
    for ctx_a in anchors:
        js_cross[ctx_a.cid] = {}
        for ctx_m in mt_contexts:
            if ctx_m.cid not in mt_history_per_cid:
                continue
            probe_js = []
            history = mt_history_per_cid[ctx_m.cid]
            for pi, probe in enumerate(probes):
                resp_set = anchor_samples[(ctx_a.cid, pi)] + mt_samples[(ctx_m.cid, pi)]
                js_vals = []
                for resp_ids in resp_set:
                    if resp_ids.numel() == 0:
                        continue
                    lp_a = _resp_logprobs_under_single_turn(
                        model, tokenizer, ctx_a, probe, resp_ids, device
                    )
                    lp_m = _resp_logprobs_under_mt(
                        model, tokenizer, history, probe, resp_ids, device
                    )
                    js_vals.append(_js_from_logprobs(lp_a, lp_m))
                if js_vals:
                    probe_js.append(float(np.mean(js_vals)))
            js_cross[ctx_a.cid][ctx_m.cid] = float(np.mean(probe_js)) if probe_js else float("nan")
        logger.info("Phase 1b: JS computed for anchor %s", ctx_a.cid)

    payload = {
        "schema_version": "i501_phase1b_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": BASE_MODEL,
        "n_anchor_contexts": len(anchors),
        "n_mt_contexts": len(mt_contexts),
        "n_probes": len(probes),
        "r_samples": r_samples,
        "js_max_tok": JS_MAX_TOK,
        "js_rb_cross_format": js_cross,
        "smoke": bool(args.smoke),
    }
    out_path = OUT_DIR / "js_rb_pairs.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Phase 1b wrote %s", out_path)
    del model
    torch.cuda.empty_cache()
    return out_path


# ---------------------------------------------------------------------------
# 1c — cross-format band-overlap gate (plan §7 gate 3)
# ---------------------------------------------------------------------------


def run_phase_1c(args, phase0: dict) -> Path:
    """Compute cosine_distance distributions for the 24-anchor × 12-MT
    cross-format cells at every swept layer; assert the headline layer's
    distribution sits in [0.3, 1.5] (plan §7 gate 3).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cos_path = OUT_DIR / "cosine_per_layer.json"
    if not cos_path.exists():
        raise RuntimeError(f"Phase 1c needs {cos_path}; run --phase 1a first.")
    payload = json.loads(cos_path.read_text())
    all_cids: list[str] = payload["all_cids"]
    mt_cids = [c for c in all_cids if c.startswith(("MT", "MN"))]
    anchor_cids = [c for c in all_cids if c.startswith(("IK", "SP"))]

    per_layer_stats: dict[int, dict] = {}
    for li in LAYERS:
        cos = payload["cos_sim_per_layer"][str(li)]
        cross_dists: list[float] = []
        for a in anchor_cids:
            for m in mt_cids:
                cross_dists.append(1.0 - cos[a][m])
        per_layer_stats[li] = {
            "n": len(cross_dists),
            "p10": float(np.percentile(cross_dists, 10)) if cross_dists else float("nan"),
            "p50": float(np.percentile(cross_dists, 50)) if cross_dists else float("nan"),
            "p90": float(np.percentile(cross_dists, 90)) if cross_dists else float("nan"),
            "mean": float(np.mean(cross_dists)) if cross_dists else float("nan"),
            "std": float(np.std(cross_dists)) if cross_dists else float("nan"),
            "n_below_lo": sum(1 for d in cross_dists if d < COSINE_DISTANCE_LO),
            "n_above_hi": sum(1 for d in cross_dists if d > COSINE_DISTANCE_HI),
        }

    headline = per_layer_stats[HEADLINE_LAYER]
    headline_in_band = (
        headline["p10"] >= COSINE_DISTANCE_LO and headline["p90"] <= COSINE_DISTANCE_HI
    )

    verdict = "PASS" if headline_in_band else "FAIL"
    out = {
        "schema_version": "i501_phase1c_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "headline_layer": HEADLINE_LAYER,
        "verdict": verdict,
        "band_lo": COSINE_DISTANCE_LO,
        "band_hi": COSINE_DISTANCE_HI,
        "per_layer_stats": {str(li): per_layer_stats[li] for li in LAYERS},
        "smoke": payload.get("smoke", False),
    }
    out_path = OUT_DIR / "cosine_band_overlap_gate.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Phase 1c cosine band-overlap gate %s -> %s", verdict, out_path)
    if verdict == "FAIL" and not payload.get("smoke", False):
        _write_failure_sentinel(
            "cosine_apples_vs_oranges",
            {
                "headline_layer": HEADLINE_LAYER,
                "p10": headline["p10"],
                "p90": headline["p90"],
                "band_lo": COSINE_DISTANCE_LO,
                "band_hi": COSINE_DISTANCE_HI,
            },
        )
        # Per plan §7 + Risk-2: FAIL surfaces "predictor cross-format is
        # UNIDENTIFIABLE" but does NOT crash the pipeline — Phase 4 still
        # runs (the layer sweep may rescue a different headline layer).
        # Caller decides whether to keep going.
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        choices=["1a", "1b", "1c", "all"],
        default="all",
        help="Run a single sub-phase or the full Phase 1 pipeline.",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke slice: 1 MT (MT05) + 2 anchors (IK01, SP01) × 3 probes × 2 JS samples.",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override the max_model_len read from phase0 (default: phase0 recommendation).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    phase0 = _load_phase0_payload()

    if args.phase in ("1a", "all"):
        run_phase_1a(args, phase0)
    if args.phase in ("1b", "all"):
        run_phase_1b(args, phase0)
    if args.phase in ("1c", "all"):
        run_phase_1c(args, phase0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
