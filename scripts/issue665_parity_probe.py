#!/usr/bin/env python
"""Issue #665 Phase 3 — A3.6c adapter-reuse fitness / scaling parity probe
(the BINDING precondition that gates every A3.6c result; plan §4 A3.6c precondition).

A3.6c is the ONE arm that APPLIES the #664 LoRA adapters at INFERENCE time. The
#664 store was committed under ``use_rslora=True`` (effective scale alpha/sqrt(r)) —
the #601 rsLoRA application-scaling condition. A stack that silently applies
alpha/r instead would answer a DIFFERENT model than the #664 trained store and
silently flip the A3.6c input-vs-map readout, while the self-patch identity nulls
still pass (both null arms apply the same divergent stack). The parity probe is
the only guard against this.

Three steps (plan §4 A3.6c precondition):
1. MANIFEST every reused adapter's ``adapter_config.json`` (grounded on the
   ARTIFACT's own config, not the #664 body prose; check (a)/(g) of artifact-reuse).
2. ASSERT condition coverage (every A3.6c-bound cell has a resolved config + row).
3. 1-ADAPTER apply-and-read parity probe (the binding gate): apply ONE A3.6c-bound
   adapter on the CURRENT transformers/peft stack at the config's declared scaling,
   read the layer-L activation, compare to #664's stored ``c_C_trained``.
   PASS iff cosine >= 0.95 AND L2 ratio (probe-norm / #664-recorded-norm) within 10%.

A FAIL HALTS A3.6c (the input-vs-map verdict is suppressed). The CPU store-read
arms continue normally — they never touch an inference-time forward pass.

Modes:
- ``--smoke-dryrun`` (CPU, this round's §6.5 deliverable): validates the CLI +
  the adapter_config reader + the manifest writer WITHOUT loading the base model.
  Writes the manifest + a dry-run marker parity file; the LIVE GPU probe is the
  experimenter's Step 6d responsibility.
- default (GPU, Step 6d): runs the live apply-and-read parity probe.

Usage:
    uv run python scripts/issue665_parity_probe.py --smoke-dryrun   # CPU, this round
    uv run python scripts/issue665_parity_probe.py                  # GPU, Step 6d
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess

import issue665_common as C

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import hf_hub_download  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("issue665_parity_probe")

FITNESS_DIR = C.EVAL_ROOT / "adapter_fitness"
MANIFEST_PATH = FITNESS_DIR / "adapter_manifest.json"

# PASS thresholds (plan §4 A3.6c precondition / §11 Decision Rationale).
PARITY_COS_FLOOR = 0.95
PARITY_L2_RATIO_TOL = 0.10  # |ratio - 1| <= 0.10

# adapter_config fields recorded per cell (artifact-reuse check (a)/(g)).
MANIFEST_FIELDS = (
    "base_model_name_or_path",
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "use_rslora",
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.REPO).decode().strip()
    except Exception:
        return "unknown"


def _adapter_config(cell: str) -> dict:
    """Download + read the #664 adapter's own adapter_config.json (the machine
    ground truth, NOT the #664 body prose)."""
    sub = f"{C.ADAPTER_PREFIX}/{cell}"
    cfgp = hf_hub_download(C.MODEL_REPO, f"{sub}/adapter_config.json")
    with open(cfgp) as f:
        return json.load(f)


def write_manifest(cells: list[str]) -> dict:
    """Step 1+2: manifest every A3.6c-bound adapter's config + assert coverage.
    A missing config / unresolved cell is a HARD fail (the cell cannot be patched)."""
    FITNESS_DIR.mkdir(parents=True, exist_ok=True)
    rows = {}
    for cell in cells:
        cfg = _adapter_config(cell)  # raises if unresolved (hard fail — plan step 2)
        rows[cell] = {k: cfg.get(k) for k in MANIFEST_FIELDS}
        # the #601 application-scaling note: effective LoRA scale.
        r = cfg.get("r")
        alpha = cfg.get("lora_alpha")
        use_rslora = cfg.get("use_rslora")
        if r and alpha:
            classic = alpha / r
            rslora = alpha / (r**0.5)
            rows[cell]["effective_scale_classic_alpha_over_r"] = round(classic, 4)
            rows[cell]["effective_scale_rslora_alpha_over_sqrt_r"] = round(rslora, 4)
            rows[cell]["applied_scale_expected"] = round(rslora if use_rslora else classic, 4)
    manifest = {
        "git_commit": _git_commit(),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "n_cells": len(rows),
        "parity_cos_floor": PARITY_COS_FLOOR,
        "parity_l2_ratio_tol": PARITY_L2_RATIO_TOL,
        "cells": rows,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=1)
    logger.info("[manifest] %d cells -> %s", len(rows), MANIFEST_PATH)
    return manifest


def _read_layer_activation_with_adapter(cell: str, layer: int):
    """GPU path (Step 6d): load base Qwen + apply the #664 adapter at its declared
    scaling, run the source context's last-input-token forward, return the layer-L
    residual activation. Imported lazily so --smoke-dryrun never loads torch/peft."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sub = f"{C.ADAPTER_PREFIX}/{cell}"
    tok = AutoTokenizer.from_pretrained(C.QWEN_ID)
    base = AutoModelForCausalLM.from_pretrained(
        C.QWEN_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # PeftModel honors use_rslora from the adapter_config — this IS the scaling we probe.
    model = PeftModel.from_pretrained(base, f"{C.MODEL_REPO}", subfolder=sub)
    model.eval()

    # Read the source-anchor context's c_C from the store (the #664-recorded target).
    from explore_persona_space.analysis.gate_io import load_cell

    sc = load_cell(cell, verify_sha=True)
    try:
        # The parity probe uses the SOURCE-ANCHOR context (the role #664 trained the
        # adapter to overwrite) — sc.source_idx / sc.source_ctx_id derive from the
        # `source-anchor` entry in meta.target_context_roles (plan §4(a)).
        c_trained = sc.tensors["c_C_trained"][sc.source_idx, layer].to(torch.float64)
        # Build the source context's REAL #664 chat prompt (Blocker 2a) — the same
        # context_messages(inst, battery_probe) #664 fed when it captured c_C_trained,
        # NOT a synthetic "Hello.". c_C is the last-input-token slot, so the read hook
        # captures position -1 (recipe-matched: CC_RECIPE == "last").
        probe_q = next(iter(sc.tensors["battery_probes"]))
        msgs = C.context_chat_messages(sc.source_ctx_id, probe_q)
        captured = {}

        def _hook(_m, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["h"] = hs[0, -1, :].detach().to("cpu", torch.float64)

        # layer index in the residual stream (hidden_states layer L = block L output).
        block = model.base_model.model.model.layers[layer]
        handle = block.register_forward_hook(_hook)
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            model(ids)
        handle.remove()
        probe_act = captured["h"]
        return probe_act, c_trained
    finally:
        sc.free()


def run_parity_probe(cell: str, layer: int) -> dict:
    """GPU path (Step 6d): the binding apply-and-read parity gate."""
    import numpy as np

    probe_act, c_trained = _read_layer_activation_with_adapter(cell, layer)
    a = probe_act.numpy()
    b = c_trained.numpy()
    cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    l2_ratio = float(np.linalg.norm(a) / (np.linalg.norm(b) + 1e-30))
    passed = bool(cos >= PARITY_COS_FLOOR and abs(l2_ratio - 1.0) <= PARITY_L2_RATIO_TOL)
    result = {
        "cell": cell,
        "layer": layer,
        "cosine": cos,
        "l2_ratio": l2_ratio,
        "cos_floor": PARITY_COS_FLOOR,
        "l2_ratio_tol": PARITY_L2_RATIO_TOL,
        "passed": passed,
        "git_commit": _git_commit(),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "dry_run": False,
    }
    return result


def write_dryrun_parity(cell: str, layer: int) -> dict:
    """CPU --smoke-dryrun path (this round's §6.5 deliverable): validate the CLI +
    config reader + manifest + the parity-file writer WITHOUT loading the base model.
    The LIVE probe is Step 6d — this file records `dry_run: true, passed: null`."""
    cfg = _adapter_config(cell)
    result = {
        "cell": cell,
        "layer": layer,
        "cosine": None,
        "l2_ratio": None,
        "cos_floor": PARITY_COS_FLOOR,
        "l2_ratio_tol": PARITY_L2_RATIO_TOL,
        "passed": None,  # null until the Step-6d GPU probe runs
        "adapter_config_resolved": True,
        "use_rslora": cfg.get("use_rslora"),
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "git_commit": _git_commit(),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "dry_run": True,
        "note": "CPU dry-run: CLI + config-read + manifest validated; live GPU probe deferred "
        "to Step 6d (experimenter). A3.6c results are NOT trusted until passed==true.",
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="issue665 A3.6c adapter parity probe")
    ap.add_argument(
        "--probe-cell",
        default=C.A36C_SUBSET[0],
        help="the cell to apply-and-read (default: top bad-medical content cell)",
    )
    ap.add_argument("--layer", type=int, default=None, help="layer L (default: cell read layer)")
    ap.add_argument(
        "--smoke-dryrun",
        action="store_true",
        help="CPU dry-run (validate CLI/config/manifest, no base-model load)",
    )
    args = ap.parse_args()

    FITNESS_DIR.mkdir(parents=True, exist_ok=True)
    # Step 1+2: manifest + coverage over the full A3.6c subset.
    write_manifest(list(C.A36C_SUBSET))

    cell = args.probe_cell
    layer = args.layer if args.layer is not None else C.read_layer_for_cell(cell)
    if args.smoke_dryrun:
        result = write_dryrun_parity(cell, layer)
    else:
        result = run_parity_probe(cell, layer)
    outp = FITNESS_DIR / f"parity_probe_{cell}.json"
    with open(outp, "w") as f:
        json.dump(result, f, indent=1)
    logger.info(
        "[parity] %s layer=%d dry_run=%s passed=%s -> %s",
        cell,
        layer,
        result["dry_run"],
        result["passed"],
        outp,
    )


if __name__ == "__main__":
    main()
