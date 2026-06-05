"""Issue #494 Phase 2: extend the canonical predictor recipe to the #192 substrate.

Mirrors ``scripts/issue494_predictor_444.py`` on the Lin/Pavlek positive-only
substrate. Two teach arms (``zelthari_scholar``-taught and Qwen-default-
taught), 4 bystanders per arm = 8 (teach x bystander) cells.

Probe set: ``superkaiba1/explore-persona-space-data:issue192_persona_spread/
datasets/fact_probes.json`` ``freeform`` (file has n=150 entries) subsetted to
a fixed random sample of n=30, seed=42 (justified in plan §4.2: DV resolution
is 8 cells not 150, so larger probe counts only tighten the per-cell predictor
mean for diminishing returns at the re-analysis budget).

Teach rows for the bystander prior: ``issue192_persona_spread/datasets/
fact_train_pairs.jsonl`` (the literal Lin/Pavlek paraphrases the model was
trained on; n=100 (q, a) rows). vLLM teacher-force; length-normalized log P
of the answer span given (bystander system + q).

DV: strict-linkage bystander-frame leak rate, 3-seed mean, locked from
``tasks/awaiting_promotion/192/body.md``. 8 numbers; not re-judged here.

Smoke (``--smoke``): n_probes=3, R=2, 1 arm (zelthari), 1 bystander (assistant),
L21 only. Writes ``predictor_192.smoke.json``.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import math
import os
import random
import subprocess
import sys
from pathlib import Path

# Pin HF cache to the persistent workspace volume BEFORE any transformers /
# vLLM imports — those libraries read HF_HOME at import time and default to
# /root/.cache on RunPod pods otherwise.
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

# Reuse predictor helpers from the Phase-1 driver
from issue494_predictor_444 import (  # noqa: E402
    bystander_logprob_all_personas,
    cosine_vs_reference_per_layer,
    js_vs_reference_canonical,
    last_token_acts,
    response_mean_acts_per_persona,
)

from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue494_predictor_192")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [7, 14, 21, 27]
HEADLINE_LAYER = 21
N_PROBES = 30
PROBE_SEED = 42

QWEN_DEFAULT_SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

ARMS: dict[str, dict] = {
    "zelthari": {
        "teach_label": "zelthari_scholar",
        "teach_sys": PERSONAS["zelthari_scholar"],
    },
    "qwen_default": {
        "teach_label": "qwen_default",
        "teach_sys": QWEN_DEFAULT_SYS,
    },
}

BYSTANDERS: dict[str, str | None] = {
    "assistant": ASSISTANT_PROMPT,
    "software_engineer": PERSONAS["software_engineer"],
    "kindergarten_teacher": PERSONAS["kindergarten_teacher"],
    "no_system": None,
}

# DV: strict-linkage LLM-judge bystander leak rate, 3-seed mean, from
# tasks/awaiting_promotion/192/body.md L51-L54. 8 cells (4 bystanders x 2 arms).
LEAK_RATES: dict[str, dict[str, float]] = {
    "zelthari": {
        "assistant": 0.607,
        "software_engineer": 0.624,
        "kindergarten_teacher": 0.636,
        "no_system": 0.586,
    },
    "qwen_default": {
        "assistant": 0.638,
        "software_engineer": 0.644,
        "kindergarten_teacher": 0.651,
        "no_system": 0.644,
    },
}


def load_freeform_probes(n: int, seed: int) -> list[str]:
    """Fetch fact_probes.json from HF, assert n=150 freeform, return a fixed sample of n probes."""
    fp = hf_hub_download(
        "superkaiba1/explore-persona-space-data",
        "issue192_persona_spread/datasets/fact_probes.json",
        repo_type="dataset",
    )
    data = json.loads(Path(fp).read_text())
    all_probes = data["freeform"]
    assert len(all_probes) == 150, f"expected 150 freeform probes, got {len(all_probes)}"
    idx = random.Random(seed).sample(range(len(all_probes)), n)
    return [all_probes[i]["q"] for i in sorted(idx)]


def load_teach_rows() -> list[dict]:
    """Fetch fact_train_pairs.jsonl. Returns list of {'question':, 'completion':}.

    The HF file uses keys ``q`` / ``a``; we rename to the
    ``{'question', 'completion'}`` schema the bystander_logprob helper expects.
    """
    fp = hf_hub_download(
        "superkaiba1/explore-persona-space-data",
        "issue192_persona_spread/datasets/fact_train_pairs.jsonl",
        repo_type="dataset",
    )
    rows: list[dict] = []
    for line in Path(fp).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({"question": r["q"], "completion": r["a"]})
    return rows


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _per_arm_spearman_with_ci(
    cells: dict[str, dict],
    predictor: str,
    *,
    n_reps: int,
    seed: int,
    ci: float = 0.95,
) -> dict:
    """Spearman rho(predictor, leak_rate) within one arm + bootstrap CI.

    Plan §4.2 (line 189) deliverable: per-arm Spearman rho + bootstrap CI on the
    4-bystander panel. The CI uses ordinary (non-stratified) bootstrap because
    each arm is a single substrate — there is no stratification axis below
    the arm. NaN predictor values (e.g. ``fact_slice_js`` for #192) are
    skipped; if the surviving n < 2 the metric is NaN.
    """
    rng = np.random.default_rng(seed)
    xs: list[float] = []
    ys: list[float] = []
    for byst_id, cell in cells.items():
        x = cell.get(predictor, float("nan"))
        y = cell.get("leak_rate", float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(float(x))
            ys.append(float(y))
        else:
            logger.debug(
                "Per-arm Spearman: skipping byst=%s for %s (predictor=%s, leak_rate=%s)",
                byst_id,
                predictor,
                x,
                y,
            )
    n = len(xs)
    if n < 2:
        return {
            "rho": float("nan"),
            "p_value": float("nan"),
            "n": n,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_reps": 0,
        }
    rho_point, p_point = spearmanr(xs, ys)
    arr_x = np.asarray(xs, dtype=float)
    arr_y = np.asarray(ys, dtype=float)
    rhos: list[float] = []
    for _ in range(n_reps):
        idx = rng.integers(0, n, size=n)
        if arr_x[idx].std() < 1e-12 or arr_y[idx].std() < 1e-12:
            continue
        r, _ = spearmanr(arr_x[idx], arr_y[idx])
        if not (r is None or math.isnan(r)):
            rhos.append(float(r))
    if not rhos:
        ci_lo = float("nan")
        ci_hi = float("nan")
    else:
        alpha = (1 - ci) / 2
        ci_lo = float(np.quantile(rhos, alpha))
        ci_hi = float(np.quantile(rhos, 1 - alpha))
    return {
        "rho": float(rho_point) if not math.isnan(rho_point) else float("nan"),
        "p_value": float(p_point) if not math.isnan(p_point) else float("nan"),
        "n": n,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_reps": len(rhos),
        "ci": ci,
    }


def _drop_model(_model_already_unbound) -> None:
    """Free residual GPU memory after the caller has dropped its model binding.

    Same contract as ``issue494_predictor_444._drop_model``: the caller MUST
    delete its binding FIRST (``del model``) and then call this helper with
    ``None``. `del` inside this function only frees the local-scope reference,
    so passing a live model in here while the caller still holds it leaves the
    refcount > 0 and the GPU memory is NOT released — the vLLM load that
    follows then OOMs. The argument is intentionally unused.
    """
    del _model_already_unbound
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return None


def run_arm_predictors(
    *,
    arm_id: str,
    arm: dict,
    bystanders: dict[str, str | None],
    probes: list[str],
    teach_rows: list[dict],
    model,
    tok,
    device: str,
    layers: list[int],
    js_r: int,
    js_max_tok: int,
    cos_b_r: int,
    cos_b_max_tok: int,
    skip_cosine_b: bool,
    skip_js: bool,
) -> dict:
    """Run all GPU-side predictors (cosine a/b, JS) for one arm.

    Per plan §4.2 ``fact_slice_js`` is #444-substrate only; #192 cells carry
    ``fact_slice_js = NaN`` (and ``fact_slice_similarity_M = NaN``) so the
    pooled regression skips them automatically on that predictor.

    The bystander prior is computed later in a SEPARATE pass after the HF model
    is released (vLLM needs the GPU). Returns a partial cell dict per bystander.
    """
    # The "teach persona" for this arm is identified by its label so the
    # divergence helpers can address it; we install it into a one-arm-local
    # PERSONA_PROMPTS-style map.
    teach_key = f"__teach_{arm_id}__"
    active = {teach_key: arm["teach_sys"], **bystanders}

    # Cosine (a) — last-input-token L_sweep
    logger.info("[arm=%s] cosine (a) — %d probes, layers=%s", arm_id, len(probes), layers)
    # last_token_acts/cosine_vs_reference_per_layer key on the persona dict
    # that PERSONA_PROMPTS contains; we monkey-patch the helper's persona
    # lookup by installing the local map BEFORE the call.
    import issue494_predictor_444 as p444  # local import to mutate module-level

    saved_panel = p444.PERSONA_PROMPTS
    p444.PERSONA_PROMPTS = active
    try:
        acts_a = {p: last_token_acts(model, tok, p, probes, device, layers) for p in active}
        cos_a_per = cosine_vs_reference_per_layer(
            acts_a, teach_key, list(bystanders.keys()), layers
        )
        del acts_a
        torch.cuda.empty_cache()

        cos_b_per: dict[str, dict[str, float]] = {}
        if not skip_cosine_b:
            logger.info(
                "[arm=%s] cosine (b) — R=%d, max_tok=%d, %d probes",
                arm_id,
                cos_b_r,
                cos_b_max_tok,
                len(probes),
            )
            acts_b = {
                p: response_mean_acts_per_persona(
                    model, tok, p, probes, cos_b_r, cos_b_max_tok, device, layers
                )
                for p in active
            }
            cos_b_per = cosine_vs_reference_per_layer(
                acts_b, teach_key, list(bystanders.keys()), layers
            )
            del acts_b
            torch.cuda.empty_cache()

        js_raw: dict[str, float] = {}
        if not skip_js:
            logger.info(
                "[arm=%s] JS RB sequence-level — R=%d, max_tok=%d, %d probes, %d bystanders",
                arm_id,
                js_r,
                js_max_tok,
                len(probes),
                len(bystanders),
            )
            js_raw = js_vs_reference_canonical(
                model,
                tok,
                probes,
                js_r,
                js_max_tok,
                device,
                teach_key,
                list(bystanders.keys()),
            )

    finally:
        p444.PERSONA_PROMPTS = saved_panel

    # Plan §4.2: ``fact_slice_js`` is #444-substrate only; NaN for #192.
    # We still emit the keys so the Phase 3 schema is uniform across substrates.
    nan = float("nan")

    if arm_id not in LEAK_RATES:
        raise KeyError(
            f"LEAK_RATES is missing arm={arm_id!r}; expected one of "
            f"{sorted(LEAK_RATES)}. Update LEAK_RATES (locked from "
            "tasks/awaiting_promotion/192/body.md) before running new arms."
        )
    missing_byst = [b for b in bystanders if b not in LEAK_RATES[arm_id]]
    if missing_byst:
        raise KeyError(
            f"LEAK_RATES[{arm_id!r}] missing bystanders: {missing_byst}. "
            f"Expected keys: {sorted(LEAK_RATES[arm_id])}."
        )

    cells: dict[str, dict] = {}
    for byst_id in bystanders:
        cell = {
            "leak_rate": LEAK_RATES[arm_id][byst_id],
            "cosine_a_L21": cos_a_per[byst_id].get(str(HEADLINE_LAYER), nan),
            "cosine_a_per_layer": cos_a_per[byst_id],
            # Plan-mandated NaN for #192:
            "fact_slice_js": nan,
            "fact_slice_similarity_M": nan,
        }
        if not skip_cosine_b:
            cell["cosine_b_L21"] = cos_b_per[byst_id].get(str(HEADLINE_LAYER), nan)
            cell["cosine_b_per_layer"] = cos_b_per[byst_id]
        if not skip_js:
            js_val = js_raw[byst_id]
            cell["js_on_topic"] = js_val
            cell["js_similarity_M"] = 1.0 - js_val if math.isfinite(js_val) else nan
        cells[byst_id] = cell
    return cells


def main() -> int:  # noqa: C901 — orchestrates HF load → 4 predictor passes → vLLM prior → per-arm Spearman; linear by design
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-probes", type=int, default=N_PROBES, help="freeform probe subsample size")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SEED)
    ap.add_argument("--js-r", type=int, default=8)
    ap.add_argument("--js-max-tok", type=int, default=256)
    ap.add_argument("--cos-b-r", type=int, default=4)
    ap.add_argument("--cos-b-max-tok", type=int, default=256)
    ap.add_argument("--out", default="eval_results/issue_494/predictor_192.json")
    ap.add_argument("--gpu-mem", type=float, default=0.60, help="vLLM gpu_memory_utilization")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny slice: n_probes=3, R=2, 1 arm (zelthari), 1 bystander (assistant), L21 only.",
    )
    ap.add_argument("--skip-cosine-b", action="store_true")
    ap.add_argument("--skip-js", action="store_true")
    ap.add_argument("--skip-prior", action="store_true")
    # NOTE: --skip-fact-slice is intentionally not exposed for Phase 2:
    # plan §4.2 mandates ``fact_slice_js = NaN`` for #192 substrate (the
    # fact-slice JS is a #444-only predictor since #192's taught
    # completion text isn't in the same fact-injection regime).
    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=2000,
        help=(
            "Per-arm cluster-aware bootstrap reps for the in-Phase-2 Spearman "
            "rho(predictor, leak_rate) CI. 4-bystander resamples are small so "
            "2000 reps is enough."
        ),
    )
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    args = ap.parse_args()

    if args.smoke:
        args.n_probes = 3
        args.js_r = 2
        args.js_max_tok = 64
        args.cos_b_r = 1
        args.cos_b_max_tok = 64
        args.out = "eval_results/issue_494/predictor_192.smoke.json"
        active_arms = {"zelthari": ARMS["zelthari"]}
        active_bystanders = {"assistant": BYSTANDERS["assistant"]}
        layers_active = [HEADLINE_LAYER]
    else:
        active_arms = ARMS
        active_bystanders = BYSTANDERS
        layers_active = LAYERS

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Loading %s on %s (smoke=%s; %d probes; R=%d; max_tok=%d)",
        args.model,
        device,
        args.smoke,
        args.n_probes,
        args.js_r,
        args.js_max_tok,
    )
    logger.info(
        "arms=%s; bystanders=%s; layers=%s",
        list(active_arms),
        list(active_bystanders),
        layers_active,
    )

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Probes + teach rows (HF-fetched; assert n=150 in load_freeform_probes)
    probes = load_freeform_probes(args.n_probes, args.probe_seed)
    teach_rows = load_teach_rows()
    if args.smoke:
        teach_rows = teach_rows[:8]
    logger.info(
        "Probes: %d freeform (seed=%d); teach rows: %d",
        len(probes),
        args.probe_seed,
        len(teach_rows),
    )

    results: dict = {
        "_doc": (
            "#494 Phase 2: extend canonical persona-distance predictor recipe "
            "(R=8/max_tok=256) to the #192 Lin/Pavlek positive-only substrate. "
            "8 (teach x bystander) cells (2 arms x 4 bystanders); fact_train_pairs "
            "teach rows for the bystander prior."
        ),
        "model": args.model,
        "n_probes": args.n_probes,
        "probe_seed": args.probe_seed,
        "layers": layers_active,
        "headline_layer": HEADLINE_LAYER,
        "config": vars(args),
        "smoke": args.smoke,
        "git_commit": _git_commit(),
        "started_at": _now_iso(),
        "arms": {},
    }

    # ── Load HF model (used for cosine a/b, JS, fact-slice JS) ──
    tok = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map=device,
            token=os.environ.get("HF_TOKEN"),
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            token=os.environ.get("HF_TOKEN"),
        ).eval()

    # ── Per-arm GPU pass (cosine a/b + JS only; fact-slice JS is #444-only) ──
    for arm_id, arm in active_arms.items():
        cells = run_arm_predictors(
            arm_id=arm_id,
            arm=arm,
            bystanders=active_bystanders,
            probes=probes,
            teach_rows=teach_rows,
            model=model,
            tok=tok,
            device=device,
            layers=layers_active,
            js_r=args.js_r,
            js_max_tok=args.js_max_tok,
            cos_b_r=args.cos_b_r,
            cos_b_max_tok=args.cos_b_max_tok,
            skip_cosine_b=args.skip_cosine_b,
            skip_js=args.skip_js,
        )
        results["arms"][arm_id] = {
            "teach_label": arm["teach_label"],
            "teach_sys": arm["teach_sys"],
            "bystanders": cells,
        }
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Checkpoint after arm=%s GPU pass → %s", arm_id, out_path)

    # ── Release HF model so vLLM can claim the GPU for the prior ──
    # `del` on the caller-side binding FIRST; then ask the helper to gc +
    # empty_cache. See _drop_model docstring for why `_drop_model(model)`
    # without deleting the caller's binding does NOT free GPU memory.
    del model
    _ = _drop_model(None)

    # ── Bystander prior (vLLM) — computed ONCE, shared across arms ──
    # The bystander prior log P(C | bystander_sys + Q) depends ONLY on the
    # bystander system prompt and the (Q, C) teach rows. Both inputs are
    # shared across the two #192 arms (same Lin/Pavlek paraphrases). Loading
    # vLLM twice in a row (the previous per-arm pattern) doubles startup
    # cost and risks the vLLM worker-subprocess teardown OOM documented in
    # CLAUDE.md gotchas: an orphan worker from the first vLLM tear-down
    # can re-allocate freed GPU memory, OOMing the second load.
    if not args.skip_prior:
        logger.info(
            "Bystander prior (vLLM) - %d teach rows x %d bystanders, shared across %d arm(s)",
            len(teach_rows),
            len(active_bystanders),
            len(active_arms),
        )
        prior_results = bystander_logprob_all_personas(
            args.model,
            teach_rows,
            dict(active_bystanders),
            gpu_memory_utilization=args.gpu_mem,
        )
        for arm_id in active_arms:
            for byst_id in active_bystanders:
                results["arms"][arm_id]["bystanders"][byst_id]["bystander_logprob"] = prior_results[
                    byst_id
                ]
            out_path.write_text(json.dumps(results, indent=2))
            logger.info("Checkpoint after attaching prior to arm=%s → %s", arm_id, out_path)

    # ── Per-arm Spearman rho + bootstrap CI (plan §4.2 deliverable) ──
    # Mirror direction of the inline-444 sign convention (higher predictor =
    # closer persona). For raw-distance predictors we ALSO compute against the
    # similarity-direction variant so Phase 3's pooled rho signs are comparable.
    per_arm_predictors = ["cosine_a_L21", "cosine_b_L21", "js_on_topic", "js_similarity_M"]
    if not args.skip_prior:
        per_arm_predictors.append("bystander_logprob")
    for arm_id in active_arms:
        cells = results["arms"][arm_id]["bystanders"]
        per_arm_stats: dict[str, dict] = {}
        for pred in per_arm_predictors:
            per_arm_stats[pred] = _per_arm_spearman_with_ci(
                cells,
                pred,
                n_reps=args.bootstrap_reps,
                seed=args.bootstrap_seed,
            )
        results["arms"][arm_id]["per_arm_spearman"] = per_arm_stats
        logger.info(
            "[arm=%s] per-arm Spearman cos_a=%.3f, js_sim=%.3f, prior=%.3f (n=%d, %d reps)",
            arm_id,
            per_arm_stats["cosine_a_L21"]["rho"],
            per_arm_stats["js_similarity_M"]["rho"],
            per_arm_stats.get("bystander_logprob", {"rho": float("nan")})["rho"],
            per_arm_stats["cosine_a_L21"]["n"],
            per_arm_stats["cosine_a_L21"]["n_reps"],
        )

    results["finished_at"] = _now_iso()
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("WROTE %s", out_path)

    # Pretty summary + correctness checks.
    # NOTE: ``fact_slice_js`` is intentionally NaN on every #192 cell per plan
    # §4.2 — it is a #444-substrate-only predictor — so we DROP it from the
    # range check. The other required predictors (cosine_a, cosine_b, js,
    # prior) MUST be finite + in-range, or we exit non-zero (fail loud).
    print("\n================ #494 Phase 2 predictor_192 ================")
    any_bad = False
    required_predictors = [
        ("cosine_a_L21", -1.0, 1.0),
        ("js_on_topic", 0.0, 1.0),
    ]
    if not args.skip_cosine_b:
        required_predictors.append(("cosine_b_L21", -1.0, 1.0))
    for arm_id in active_arms:
        for byst_id in active_bystanders:
            cell = results["arms"][arm_id]["bystanders"][byst_id]
            cos_a = cell.get("cosine_a_L21", float("nan"))
            cos_b = cell.get("cosine_b_L21", float("nan"))
            js = cell.get("js_on_topic", float("nan"))
            prior = cell.get("bystander_logprob", float("nan"))
            leak = cell.get("leak_rate", float("nan"))
            print(
                f"  arm={arm_id:14} byst={byst_id:22}  "
                f"leak={leak:.3f}  cos_a={cos_a:.4f}  cos_b={cos_b:.4f}  "
                f"js={js:.4f}  prior={prior:+.4f}  fact_js=NaN(#192)"
            )
            for name, lo, hi in required_predictors:
                val = cell.get(name, float("nan"))
                if not math.isfinite(val):
                    print(f"   FAIL [{arm_id}/{byst_id}] {name}={val} is non-finite (required)")
                    any_bad = True
                    continue
                if not (lo <= val <= hi):
                    print(f"   FAIL [{arm_id}/{byst_id}] {name}={val} out of [{lo},{hi}]")
                    any_bad = True
            if not args.skip_prior:
                if not math.isfinite(prior):
                    print(f"   FAIL [{arm_id}/{byst_id}] bystander_logprob={prior} non-finite")
                    any_bad = True
                elif prior > 0:
                    print(
                        f"   FAIL [{arm_id}/{byst_id}] bystander_logprob={prior} > 0 "
                        "(teacher-forced log-probs must be <= 0)"
                    )
                    any_bad = True
    # Per-arm Spearman headline printout
    print("\nPer-arm Spearman rho(predictor, leak_rate):")
    for arm_id in active_arms:
        s = results["arms"][arm_id].get("per_arm_spearman", {})
        for pred in per_arm_predictors:
            cell = s.get(pred, {})
            print(
                f"  arm={arm_id:14} pred={pred:22}  "
                f"rho={cell.get('rho', float('nan')):+.3f}  "
                f"n={cell.get('n', 0):2d}  "
                f"ci=[{cell.get('ci_lo', float('nan')):+.3f}, "
                f"{cell.get('ci_hi', float('nan')):+.3f}] "
                f"(reps={cell.get('n_reps', 0)})"
            )
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
