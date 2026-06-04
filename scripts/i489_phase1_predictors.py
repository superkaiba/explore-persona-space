"""Issue #489 Phase 1 — predictor + scaffold-overlap computation + cosine coverage gate.

Plan v5 §4.3 + §4.2 (Phase-0 cosine coverage gate folded in).

Three sub-phases (per CLAUDE.md checkpoint-per-phase; each persists its own file):

  1a. Cosine layer sweep at {7, 11, 14, 15, 21, 27} — base-model residual at last
      prompt token of ``[scaffold] + [Q'_probe]``, mean over 50 probe questions.
      Reuses ``last_token_acts`` (forked from ``scripts/issue444_persona_distance_topic.py``
      for compatibility outside the persistent issue444 branch).
  1b. JS RB sequence-level on the 552 ordered pairs (reuses ``js_vs_reference`` shape).
  1c. ``kind_distinctness_score`` = RB JS source vs IK01-baseline.
  1d. ``scaffold_overlap_score`` (token Jaccard + BOW cos + persona-word indicator)
      over the 552 off-diagonal cells — CPU-only.

After all four: run the Phase-0b cosine-coverage gate (within-arm band spread +
cross-type band overlap + de-confounding upper-band check). Writes a sentinel
on FAIL.

Outputs (under ``eval_results/issue_489/phase1/``):
  - ``cosine_per_layer.json``       (1a)
  - ``js_rb_pairs.json``            (1b)
  - ``kind_distinctness.json``       (1c)
  - ``scaffold_overlap.json``        (1d)
  - ``cosine_coverage_gate.json``   (verdict + per-arm + cross-type stats)

CLI:
    uv run python scripts/i489_phase1_predictors.py
    uv run python scripts/i489_phase1_predictors.py --phase 1a   # one sub-phase at a time
    uv run python scripts/i489_phase1_predictors.py --smoke      # 2 ctx, 3 probes, 2 JS samples
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.experiments.i460_data import load_q_test_extended_50
from explore_persona_space.experiments.i489_contexts import (
    UNION_BY_CID,
    UNION_CONTEXTS,
    build_union_prompt,
    is_cross_type,
    scaffold_overlap_score,
)

logger = logging.getLogger("i489.phase1")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = Path("eval_results/issue_489/phase1")
LAYERS = (7, 11, 14, 15, 21, 27)
HEADLINE_LAYER = 21
N_PROBE = 50  # held-out Q'_probe count
JS_R = 4  # # samples per persona per probe (smoke: 2)
JS_MAX_TOK = 128  # cap on response length for the JS RB estimator


def _git_commit_hash() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# -----------------------------------------------------------------------------
# 1a — cosine layer sweep on the 24 union contexts
# -----------------------------------------------------------------------------


@torch.no_grad()
def _last_token_acts(
    model, tokenizer, contexts, probes: list[str], device
) -> dict[str, dict[int, torch.Tensor]]:
    """Return {cid: {layer: (n_probes, hidden) cpu fp32}} for last input token.

    Forked from ``scripts/issue444_persona_distance_topic.py::last_token_acts``
    (the only persistent location for that primitive on a non-issue444 branch).
    """
    out: dict[str, dict[int, list[torch.Tensor]]] = {
        c.cid: {li: [] for li in LAYERS} for c in contexts
    }
    for ctx in contexts:
        for probe in probes:
            prompt = build_union_prompt(ctx, probe, tokenizer)
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
                device
            )
            hs = model(ids, output_hidden_states=True).hidden_states
            for li in LAYERS:
                # hidden_states[li+1] == output of transformer block li (hs[0]=embeddings)
                out[ctx.cid][li].append(hs[li + 1][0, -1, :].float().cpu())
    return {cid: {li: torch.stack(v) for li, v in d.items()} for cid, d in out.items()}


def _pairwise_cosine(
    acts: dict[str, dict[int, torch.Tensor]],
) -> dict[int, dict[str, dict[str, float]]]:
    """Mean-over-probes cosine between every ordered pair (i, j) per layer.

    Returns {layer: {cid_i: {cid_j: cos_sim}}}.
    """
    cids = list(acts.keys())
    per_layer: dict[int, dict[str, dict[str, float]]] = {}
    for li in LAYERS:
        per_layer[li] = {ci: {} for ci in cids}
        # mean vector per cid at this layer
        mean_vecs = {ci: acts[ci][li].mean(dim=0) for ci in cids}
        for ci in cids:
            vi = mean_vecs[ci]
            for cj in cids:
                vj = mean_vecs[cj]
                num = float((vi * vj).sum())
                den = float(vi.norm() * vj.norm() + 1e-12)
                per_layer[li][ci][cj] = num / den
    return per_layer


def run_phase_1a(args) -> Path:
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

    probes = load_q_test_extended_50()
    if args.smoke:
        probes = probes[:3]
        contexts = [c for c in UNION_CONTEXTS if c.cid in ("IK01", "SP01")]
    else:
        contexts = UNION_CONTEXTS

    acts = _last_token_acts(model, tokenizer, contexts, probes, device)
    cos_per_layer = _pairwise_cosine(acts)

    payload = {
        "schema_version": "i489_phase1a_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": BASE_MODEL,
        "n_contexts": len(contexts),
        "n_probes": len(probes),
        "layers": list(LAYERS),
        "headline_layer": HEADLINE_LAYER,
        "cos_sim_per_layer": {
            str(li): {
                ci: {cj: cos_per_layer[li][ci][cj] for cj in cos_per_layer[li][ci]}
                for ci in cos_per_layer[li]
            }
            for li in LAYERS
        },
        "smoke": bool(args.smoke),
    }
    out_path = OUT_DIR / "cosine_per_layer.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Phase 1a wrote %s", out_path)
    # Free GPU.
    del model
    torch.cuda.empty_cache()
    return out_path


# -----------------------------------------------------------------------------
# 1b — JS RB sequence-level on the 552 ordered pairs
# -----------------------------------------------------------------------------


@torch.no_grad()
def _sample_responses(model, tokenizer, ctx, probe: str, r: int, max_tok: int, device):
    prompt = build_union_prompt(ctx, probe, tokenizer)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
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
def _resp_logprobs(model, tokenizer, ctx, probe: str, resp_ids: torch.Tensor, device):
    prompt = build_union_prompt(ctx, probe, tokenizer)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    resp = resp_ids.to(device).unsqueeze(0)
    full = torch.cat([ids, resp], dim=1)
    logits = model(full).logits[0].float()
    start = ids.shape[1] - 1
    end = start + resp_ids.shape[0]
    sel = logits[start:end]
    return torch.log_softmax(sel, dim=-1)


def _js_from_logprobs(lp_a: torch.Tensor, lp_b: torch.Tensor) -> float:
    pa, pb = lp_a.exp(), lp_b.exp()
    m = 0.5 * (pa + pb)
    log_m = m.clamp_min(1e-12).log()
    kl_a = (pa * (lp_a - log_m)).sum(-1)
    kl_b = (pb * (lp_b - log_m)).sum(-1)
    js = 0.5 * (kl_a + kl_b) / math.log(2.0)
    return float(js.clamp(0, 1).mean())


def run_phase_1b(args) -> Path:
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

    probes = load_q_test_extended_50()
    if args.smoke:
        probes = probes[:2]
        contexts = [c for c in UNION_CONTEXTS if c.cid in ("IK01", "SP01")]
        r_samples = 2
    else:
        contexts = UNION_CONTEXTS
        r_samples = JS_R

    cids = [c.cid for c in contexts]

    # Pre-sample R responses once per (ctx, probe).
    samples: dict[tuple[str, int], list[torch.Tensor]] = {}
    for ctx in contexts:
        for pi, probe in enumerate(probes):
            samples[(ctx.cid, pi)] = _sample_responses(
                model, tokenizer, ctx, probe, r_samples, JS_MAX_TOK, device
            )
        logger.info("JS sampled responses for %s", ctx.cid)

    js_pairs: dict[str, dict[str, float]] = {ci: {} for ci in cids}
    for ci in cids:
        ctx_i = UNION_BY_CID[ci]
        for cj in cids:
            if ci == cj:
                js_pairs[ci][cj] = 0.0
                continue
            ctx_j = UNION_BY_CID[cj]
            probe_js = []
            for pi, probe in enumerate(probes):
                resp_set = samples[(ci, pi)] + samples[(cj, pi)]
                js_vals = []
                for resp_ids in resp_set:
                    if resp_ids.numel() == 0:
                        continue
                    lp_i = _resp_logprobs(model, tokenizer, ctx_i, probe, resp_ids, device)
                    lp_j = _resp_logprobs(model, tokenizer, ctx_j, probe, resp_ids, device)
                    js_vals.append(_js_from_logprobs(lp_i, lp_j))
                if js_vals:
                    probe_js.append(float(np.mean(js_vals)))
            js_pairs[ci][cj] = float(np.mean(probe_js)) if probe_js else float("nan")
        logger.info("JS computed for source %s", ci)

    payload = {
        "schema_version": "i489_phase1b_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": BASE_MODEL,
        "n_contexts": len(contexts),
        "n_probes": len(probes),
        "r_samples": r_samples,
        "js_max_tok": JS_MAX_TOK,
        "js_rb_pairs": js_pairs,
        "smoke": bool(args.smoke),
    }
    out_path = OUT_DIR / "js_rb_pairs.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Phase 1b wrote %s", out_path)
    del model
    torch.cuda.empty_cache()
    return out_path


# -----------------------------------------------------------------------------
# 1c — kind_distinctness_score = RB JS vs IK01 baseline
# -----------------------------------------------------------------------------


def run_phase_1c(args) -> Path:
    """Re-uses 1b's JS pairs: ``kind_distinctness_score[cid] = js_rb_pairs[cid][IK01]``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    js_path = OUT_DIR / "js_rb_pairs.json"
    if not js_path.exists():
        raise RuntimeError(f"Phase 1c needs {js_path}; run --phase 1b first.")
    js_payload = json.loads(js_path.read_text())
    pairs = js_payload["js_rb_pairs"]
    if "IK01" not in pairs:
        raise RuntimeError("IK01 baseline missing from JS pairs (Phase 1b incomplete).")
    distinctness = {ci: float(pairs[ci]["IK01"]) for ci in pairs}
    payload = {
        "schema_version": "i489_phase1c_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "baseline": "IK01",
        "kind_distinctness_score": distinctness,
        "smoke": js_payload.get("smoke", False),
    }
    out_path = OUT_DIR / "kind_distinctness.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Phase 1c wrote %s", out_path)
    return out_path


# -----------------------------------------------------------------------------
# 1d — scaffold_overlap_score over the 552 off-diagonal cells
# -----------------------------------------------------------------------------


def run_phase_1d(args) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contexts = (
        [c for c in UNION_CONTEXTS if c.cid in ("IK01", "SP01")] if args.smoke else UNION_CONTEXTS
    )
    cells: dict[str, dict[str, dict]] = {c.cid: {} for c in contexts}
    n = 0
    for ci in contexts:
        for cj in contexts:
            if ci.cid == cj.cid:
                continue
            cells[ci.cid][cj.cid] = scaffold_overlap_score(ci, cj)
            n += 1
    payload = {
        "schema_version": "i489_phase1d_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "n_off_diagonal_cells": n,
        "feature_weights": {"jaccard": 0.5, "bow_cos": 0.3, "persona_indicator": 0.2},
        "scaffold_overlap_per_cell": cells,
        "smoke": bool(args.smoke),
    }
    out_path = OUT_DIR / "scaffold_overlap.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Phase 1d wrote %s (n=%d off-diagonal cells)", out_path, n)
    return out_path


# -----------------------------------------------------------------------------
# 1e — cosine coverage gate (Phase-0 gate from plan §4.2, ran after 1a)
# -----------------------------------------------------------------------------


def _pct(arr: list[float], q: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(arr, q))


def run_phase_1e(args) -> Path:
    """Three-arm cosine coverage gate. Writes verdict + per-arm stats."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cos_path = OUT_DIR / "cosine_per_layer.json"
    if not cos_path.exists():
        raise RuntimeError(f"Phase 1e needs {cos_path}; run --phase 1a first.")
    payload = json.loads(cos_path.read_text())
    cos = payload["cos_sim_per_layer"][str(HEADLINE_LAYER)]
    cids = list(cos.keys())

    # Convert cosine SIMILARITY (in file) → cosine DISTANCE (1 - sim) for the gate.
    icl_cids = [c for c in cids if c.startswith("IK")]
    sp_cids = [c for c in cids if c.startswith("SP")]

    def _pair_dists(a_set: list[str], b_set: list[str]) -> list[float]:
        out: list[float] = []
        for ci in a_set:
            for cj in b_set:
                if ci == cj:
                    continue
                out.append(1.0 - cos[ci][cj])
        return out

    icl_within = _pair_dists(icl_cids, icl_cids)
    sp_within = _pair_dists(sp_cids, sp_cids)
    cross = _pair_dists(icl_cids, sp_cids) + _pair_dists(sp_cids, icl_cids)
    full = icl_within + sp_within + cross

    def _band_counts(ds: list[float]) -> dict:
        return {
            "n": len(ds),
            "n_in_0.7_0.9": sum(1 for d in ds if 0.7 <= d <= 0.9),
            "n_below_0.7": sum(1 for d in ds if d < 0.7),
            "n_above_0.9": sum(1 for d in ds if d > 0.9),
            "p25": _pct(ds, 25),
            "p50": _pct(ds, 50),
            "p75": _pct(ds, 75),
            "mean": float(np.mean(ds)) if ds else float("nan"),
            "std": float(np.std(ds)) if ds else float("nan"),
        }

    icl_stats = _band_counts(icl_within)
    sp_stats = _band_counts(sp_within)
    cross_stats = _band_counts(cross)

    # Arm 1: within-arm band-spread
    icl_arm = icl_stats["n_in_0.7_0.9"] >= 50 and icl_stats["n_below_0.7"] >= 30
    sp_arm = sp_stats["n_in_0.7_0.9"] >= 8 and sp_stats["n_below_0.7"] >= 6

    # Arm 2: cross-type band overlap (≥50% interval overlap of [p25, p75])
    def _interval_overlap(a_lo, a_hi, b_lo, b_hi) -> float:
        lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
        if hi <= lo:
            return 0.0
        a_len = max(a_hi - a_lo, 1e-12)
        b_len = max(b_hi - b_lo, 1e-12)
        return (hi - lo) / min(a_len, b_len)

    cross_icl_overlap = _interval_overlap(
        icl_stats["p25"], icl_stats["p75"], cross_stats["p25"], cross_stats["p75"]
    )
    cross_sp_overlap = _interval_overlap(
        sp_stats["p25"], sp_stats["p75"], cross_stats["p25"], cross_stats["p75"]
    )
    cross_overlap_ok = (cross_icl_overlap >= 0.50) or (cross_sp_overlap >= 0.50)

    # Arm 3: de-confounding upper band (>= p75 of full) — at least 4 pairs with NEITHER
    # side in the strong-kind set.
    from explore_persona_space.experiments.i489_contexts import STRONG_KIND_SET

    p75_full = _pct(full, 75) if full else float("nan")
    upper_band_non_strong = 0
    for ci in cids:
        for cj in cids:
            if ci == cj:
                continue
            d = 1.0 - cos[ci][cj]
            if d >= p75_full and ci not in STRONG_KIND_SET and cj not in STRONG_KIND_SET:
                upper_band_non_strong += 1
    deconf_ok = upper_band_non_strong >= 4

    verdict = "PASS" if (icl_arm and sp_arm and cross_overlap_ok and deconf_ok) else "FAIL"
    out = {
        "schema_version": "i489_phase1e_v1",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "headline_layer": HEADLINE_LAYER,
        "verdict": verdict,
        "arm_within_icl_pass": bool(icl_arm),
        "arm_within_sp_pass": bool(sp_arm),
        "arm_cross_type_overlap_pass": bool(cross_overlap_ok),
        "arm_deconfounding_upper_band_pass": bool(deconf_ok),
        "icl_within_stats": icl_stats,
        "sp_within_stats": sp_stats,
        "cross_type_stats": cross_stats,
        "cross_icl_p25p75_overlap": cross_icl_overlap,
        "cross_sp_p25p75_overlap": cross_sp_overlap,
        "p75_full_panel": p75_full,
        "upper_band_non_strong_n": upper_band_non_strong,
        "smoke": payload.get("smoke", False),
    }
    out_path = OUT_DIR / "cosine_coverage_gate.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Phase 1e cosine coverage gate %s -> %s", verdict, out_path)
    if verdict == "FAIL" and not payload.get("smoke", False):
        # Write block sentinel (skipped in smoke).
        sentinel_dir = (
            Path("/workspace/logs") if Path("/workspace").exists() else Path("logs/issue_489")
        )
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        epoch = int(_dt.datetime.now(_dt.UTC).timestamp())
        s = sentinel_dir / f"issue-489-epm_failure-{epoch}.json"
        s.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:failure",
                    "version": 1,
                    "issue": 489,
                    "phase": "phase1e_cosine_coverage_gate",
                    "failure_class": "code",
                    "reason": "cosine_coverage_insufficient",
                    "detail": out,
                    "wrote_at": _dt.datetime.now(_dt.UTC).isoformat(),
                }
            )
        )
        raise SystemExit(2)
    return out_path


# Defensive: silence unused-import warnings for is_cross_type; it is used elsewhere in the
# pipeline (Phase 5 analyzer + figures) and re-exported here for analyzer convenience.
_ = is_cross_type


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        choices=["1a", "1b", "1c", "1d", "1e", "all"],
        default="all",
        help="Run a single sub-phase or the full Phase 1 pipeline.",
    )
    ap.add_argument(
        "--smoke", action="store_true", help="Tiny slice (2 ctx, 3 probes) for wiring check."
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    if args.phase in ("1a", "all"):
        run_phase_1a(args)
    if args.phase in ("1b", "all"):
        run_phase_1b(args)
    if args.phase in ("1c", "all"):
        run_phase_1c(args)
    if args.phase in ("1d", "all"):
        run_phase_1d(args)
    if args.phase in ("1e", "all"):
        run_phase_1e(args)


if __name__ == "__main__":
    main()
