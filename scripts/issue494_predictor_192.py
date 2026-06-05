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

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

# Reuse predictor helpers from the Phase-1 driver
from issue494_predictor_444 import (  # noqa: E402
    bystander_logprob_all_personas,
    cosine_vs_reference_per_layer,
    fact_slice_js_per_persona,
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


def _drop_model(model) -> None:
    """Free the HF model + clear CUDA cache so vLLM can claim the GPU."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


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
    skip_fact_slice: bool,
) -> dict:
    """Run all GPU-side predictors (cosine a/b, JS, fact-slice JS) for one arm.

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

        fact_js: dict[str, float] = {}
        if not skip_fact_slice:
            logger.info(
                "[arm=%s] Fact-slice JS — teacher-force on %d teach rows",
                arm_id,
                len(teach_rows),
            )
            fact_js = fact_slice_js_per_persona(
                model, tok, teach_rows, device, teach_key, list(bystanders.keys())
            )

    finally:
        p444.PERSONA_PROMPTS = saved_panel

    cells: dict[str, dict] = {}
    for byst_id in bystanders:
        cell = {
            "leak_rate": LEAK_RATES[arm_id][byst_id]
            if byst_id in LEAK_RATES.get(arm_id, {})
            else float("nan"),
            "cosine_a_L21": cos_a_per[byst_id].get(str(HEADLINE_LAYER), float("nan")),
            "cosine_a_per_layer": cos_a_per[byst_id],
        }
        if not skip_cosine_b:
            cell["cosine_b_L21"] = cos_b_per[byst_id].get(str(HEADLINE_LAYER), float("nan"))
            cell["cosine_b_per_layer"] = cos_b_per[byst_id]
        if not skip_js:
            js_val = js_raw[byst_id]
            cell["js_on_topic"] = js_val
            cell["js_similarity_M"] = 1.0 - js_val if math.isfinite(js_val) else float("nan")
        if not skip_fact_slice:
            cell["fact_slice_js"] = fact_js[byst_id]
        cells[byst_id] = cell
    return cells


def main() -> int:
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
    ap.add_argument("--skip-fact-slice", action="store_true")
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

    # ── Per-arm GPU pass (cosine a/b + JS + fact-slice JS) ──
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
            skip_fact_slice=args.skip_fact_slice,
        )
        results["arms"][arm_id] = {
            "teach_label": arm["teach_label"],
            "teach_sys": arm["teach_sys"],
            "bystanders": cells,
        }
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Checkpoint after arm=%s GPU pass → %s", arm_id, out_path)

    # ── Release HF model so vLLM can claim the GPU ──
    _drop_model(model)

    # ── Bystander prior (vLLM) — per arm because the teach context differs ──
    if not args.skip_prior:
        for arm_id, _arm in active_arms.items():
            # The bystander prior depends only on the BYSTANDER system prompt,
            # not the teach prompt (we're scoring P(C | bystander_sys + Q) where
            # C/Q come from the teach rows). It IS still per-arm because the
            # teach rows of #192 are shared across arms (same Lin/Pavlek
            # paraphrases), so we could compute it once for all bystanders +
            # cache. We compute it explicitly per arm to keep the output JSON
            # symmetric and to allow per-arm teach-row subsetting later.
            active_for_prior = dict(active_bystanders)
            logger.info(
                "[arm=%s] Bystander prior (vLLM) - %d teach rows x %d bystanders",
                arm_id,
                len(teach_rows),
                len(active_for_prior),
            )
            prior_results = bystander_logprob_all_personas(
                args.model,
                teach_rows,
                active_for_prior,
                gpu_memory_utilization=args.gpu_mem,
            )
            for byst_id in active_bystanders:
                results["arms"][arm_id]["bystanders"][byst_id]["bystander_logprob"] = prior_results[
                    byst_id
                ]
            out_path.write_text(json.dumps(results, indent=2))
            logger.info("Checkpoint after arm=%s prior → %s", arm_id, out_path)

    results["finished_at"] = _now_iso()
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("WROTE %s", out_path)

    # Pretty summary + correctness checks (smoke = correctness; production also OK)
    print("\n================ #494 Phase 2 predictor_192 ================")
    any_bad = False
    for arm_id in active_arms:
        for byst_id in active_bystanders:
            cell = results["arms"][arm_id]["bystanders"][byst_id]
            cos_a = cell.get("cosine_a_L21", float("nan"))
            cos_b = cell.get("cosine_b_L21", float("nan"))
            js = cell.get("js_on_topic", float("nan"))
            prior = cell.get("bystander_logprob", float("nan"))
            fact_js = cell.get("fact_slice_js", float("nan"))
            leak = cell.get("leak_rate", float("nan"))
            print(
                f"  arm={arm_id:14} byst={byst_id:22}  "
                f"leak={leak:.3f}  cos_a={cos_a:.4f}  cos_b={cos_b:.4f}  "
                f"js={js:.4f}  prior={prior:+.4f}  fact_js={fact_js:.4f}"
            )
            # Correctness gates: finite + ranges (skip the smoke degenerate case)
            for name, val, lo, hi in [
                ("cosine_a_L21", cos_a, -1.0, 1.0),
                ("cosine_b_L21", cos_b, -1.0, 1.0),
                ("js_on_topic", js, 0.0, 1.0),
                ("fact_slice_js", fact_js, 0.0, 1.0),
            ]:
                if math.isfinite(val) and not (lo <= val <= hi):
                    print(f"   WARNING [{arm_id}/{byst_id}] {name}={val} out of [{lo},{hi}]")
                    any_bad = True
            if math.isfinite(prior) and prior > 0:
                print(
                    f"   WARNING [{arm_id}/{byst_id}] bystander_logprob={prior} > 0 (expected <0)"
                )
                any_bad = True
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
