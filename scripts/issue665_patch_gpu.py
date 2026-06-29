#!/usr/bin/env python
"""Issue #665 Phase 3 — A3.6c causal context-vector patch (the ONE GPU arm).

Tests whether off-source leakage is carried by the context vector c_C (INPUT) or
by the context->profile map M (FUNCTION):

  P-up: run theta0 (BASE) with the layer-L context-position residual overwritten
        with the TRAINED context vector c+(C).
  P-down: run theta+ (base+adapter) with the layer-L context position overwritten
        with the BASE context vector c0(C).

  f_CV = patch effect projected onto the FT shift d=(v+ - v0)/||.|| (and the E analog).
  f_CV ~ 1  => context vector moved (input-localized);  f_CV ~ 0 => map changed.

Controls (plan §4): self-patch identity nulls (c0->c0, c+->c+ MUST be no-ops —
the measurement-validity gate); random/other-context-CV floor; norm-matched variant;
patch-scope {last-input-token, full-context-span}; L-sweep {7,14,21}.

**BINDING precondition (plan §4 A3.6c precondition):** every A3.6c result is
CONDITIONAL on the adapter-reuse parity probe (issue665_parity_probe.py) having
PASSed (cos >= 0.95 AND L2 ratio within 10%) in THIS transformers/peft stack. This
script REFUSES to emit a trusted verdict unless parity_probe_<cell>.json shows
passed==true (a FAIL or dry-run halts the verdict; the store-read CPU arms are
unaffected). The self-patch nulls do NOT catch a stack-mismatch — the parity probe
is the only guard (plan §3 A3.6c).

GPU-bound carve-out (this round): the local VM has no compatible GPU, so the live
patch is the experimenter's Step 6d responsibility. This round delivers:
- the code (this script), import-checked + signature-smoked;
- a CPU dry-run (`--smoke-dryrun`) that validates the CLI, the parity-gate read,
  and the output writer WITHOUT loading the base model.

Usage:
    uv run python scripts/issue665_patch_gpu.py --smoke-dryrun   # CPU, this round
    uv run python scripts/issue665_patch_gpu.py                  # GPU, Step 6d
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("issue665_patch_gpu")

A36C_DIR = C.EVAL_ROOT / "a36c"
FITNESS_DIR = C.EVAL_ROOT / "adapter_fitness"

# The patch variants per (cell, context, layer, scope) — plan §4 A3.6c.
PATCH_VARIANTS = ("p_up", "p_down", "self_c0", "self_cp", "random_cv", "norm_matched")
PATCH_SCOPES = ("last", "full")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.REPO).decode().strip()
    except Exception:
        return "unknown"


def parity_gate_passed(cell: str) -> tuple[bool, str]:
    """Read the parity-probe result for a cell; return (passed, reason). A3.6c
    results are NOT trusted unless this returns (True, ...). A dry-run or FAIL or
    missing file returns False (the verdict is suppressed; plan §4 step 5)."""
    pf = FITNESS_DIR / f"parity_probe_{cell}.json"
    if not pf.exists():
        # the probe runs on the top bad-medical content cell; that ONE PASS gates
        # the whole subset (the fleet shares one adapter_config schema, plan step 4).
        probe_cell = C.A36C_SUBSET[0]
        pf = FITNESS_DIR / f"parity_probe_{probe_cell}.json"
        if not pf.exists():
            return False, f"no parity_probe_*.json found (expected {pf.name})"
    with open(pf) as f:
        res = json.load(f)
    if res.get("dry_run"):
        return False, "parity probe is a DRY-RUN (live GPU probe deferred to Step 6d)"
    if res.get("passed") is True:
        return True, f"parity probe PASS (cos={res.get('cosine')}, l2={res.get('l2_ratio')})"
    return False, f"parity probe FAIL/incomplete (passed={res.get('passed')})"


def _run_patch_gpu(cell: str, contexts: list[str], layers: list[int], scopes: list[str]) -> dict:
    """GPU path (Step 6d): the live causal context-vector patch. Imported lazily so
    --smoke-dryrun never loads torch/peft. Returns the per-(ctx,layer,scope,variant)
    f_CV reads (both v and E)."""
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.analysis.gate_io import load_cell

    tok = AutoTokenizer.from_pretrained(C.QWEN_ID)
    base = AutoModelForCausalLM.from_pretrained(
        C.QWEN_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    sub = f"{C.ADAPTER_PREFIX}/{cell}"
    trained = PeftModel.from_pretrained(base, C.MODEL_REPO, subfolder=sub)
    trained.eval()
    base.eval()

    sc = load_cell(cell, verify_sha=True)
    out_rows: list[dict] = []
    try:
        ctx_ids = list(sc.tensors["context_ids"])
        c_base = sc.tensors["c_C_base"].to(torch.float64)  # (C,L,d)
        c_trn = sc.tensors["c_C_trained"].to(torch.float64)
        v_plus = sc.tensors["v_plus"].to(torch.float64)
        v0 = sc.tensors["v0"].to(torch.float64)
        rng = np.random.default_rng(42)

        for ctx_id in contexts:
            if ctx_id not in ctx_ids:
                continue
            ci = ctx_ids.index(ctx_id)
            for layer in layers:
                d_shift = (v_plus[ci, layer] - v0[ci, layer]).numpy()
                d_norm = d_shift / (np.linalg.norm(d_shift) + 1e-30)
                for scope in scopes:
                    # patch each variant's context vector into the residual stream at
                    # layer L (last-input-token or full-context-span) via a forward hook.
                    for variant in PATCH_VARIANTS:
                        patched_v, patched_e = _one_patch(
                            tok,
                            base,
                            trained,
                            layer,
                            scope,
                            variant,
                            c_base[ci, layer],
                            c_trn[ci, layer],
                            rng,
                            c_base,
                            layer,
                        )
                        f_cv_v = (
                            float((patched_v - v0[ci, layer].numpy()) @ d_norm)
                            / (np.linalg.norm(d_shift) + 1e-30)
                            if patched_v is not None
                            else None
                        )
                        out_rows.append(
                            {
                                "context_id": ctx_id,
                                "layer": layer,
                                "scope": scope,
                                "variant": variant,
                                "f_cv_v": f_cv_v,
                                "e_read": patched_e,
                            }
                        )
        return {"rows": out_rows}
    finally:
        sc.free()


def _one_patch(tok, base, trained, layer, scope, variant, c0, cp, rng, c_base_all, _layer):
    """Apply ONE patch variant and read the resulting answer-side residual (v) at
    layer L. The E (behavioral) read is left for the Step-6d judge pass (returns
    None here — the activation read is the primary A3.6c DV). Faithful to plan §4."""
    import numpy as np
    import torch

    # choose the model + the patch vector per variant
    model = base if variant in ("p_up", "self_c0") else trained  # theta0 vs theta+
    if variant == "p_up":
        patch_vec = cp
    elif variant == "p_down" or variant == "self_c0":
        patch_vec = c0
    elif variant == "self_cp":
        patch_vec = cp
    elif variant == "random_cv":
        # random/other-context CV floor: a random OTHER context's base vector
        j = int(rng.integers(0, c_base_all.shape[0]))
        patch_vec = c_base_all[j, layer]
    elif variant == "norm_matched":
        patch_vec = cp * (
            float(np.linalg.norm(c0.numpy())) / (float(np.linalg.norm(cp.numpy())) + 1e-30)
        )
    else:
        raise ValueError(variant)

    pv = torch.tensor(patch_vec.numpy(), dtype=torch.bfloat16, device=model.device)
    captured = {}

    def _patch_hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if scope == "last":
            hs[:, -1, :] = pv
        else:  # full-context-span upper bound
            hs[:, :, :] = pv
        return out

    def _read_hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["h"] = hs[0, -1, :].detach().to("cpu", torch.float64).numpy()

    blk = (
        model.base_model.model.model.layers[layer]
        if hasattr(model, "base_model")
        else (model.model.layers[layer])
    )
    h1 = blk.register_forward_hook(_patch_hook)
    h2 = blk.register_forward_hook(_read_hook)
    msgs = [{"role": "user", "content": "Hello."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        model(ids)
    h1.remove()
    h2.remove()
    return captured.get("h"), None


def write_dryrun(cells: list[str], layers: list[int]) -> None:
    """CPU --smoke-dryrun: validate CLI + parity-gate read + output writer WITHOUT
    loading the base model. Writes a dry-run a36c file per cell."""
    A36C_DIR.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        gated, reason = parity_gate_passed(cell)
        rec = {
            "cell": cell,
            "behavior": C.behavior_for_cell(cell),
            "column": C.column_for_cell(cell),
            "layers": layers,
            "scopes": list(PATCH_SCOPES),
            "variants": list(PATCH_VARIANTS),
            "parity_gate_passed": gated,
            "parity_gate_reason": reason,
            "rows": [],
            "git_commit": _git_commit(),
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "dry_run": True,
            "note": "CPU dry-run: CLI + parity-gate + writer validated; live GPU patch "
            "deferred to Step 6d (experimenter). f_CV verdict NOT trusted until "
            "parity_gate_passed==true AND dry_run==false.",
        }
        outp = A36C_DIR / f"{cell}.json"
        with open(outp, "w") as f:
            json.dump(rec, f, indent=1)
        logger.info("[a36c dry-run] %s parity_gate=%s -> %s", cell, gated, outp)


def main():
    ap = argparse.ArgumentParser(description="issue665 A3.6c causal patch (GPU arm)")
    ap.add_argument("--cells", nargs="*", default=list(C.A36C_SUBSET))
    ap.add_argument("--layers", nargs="*", type=int, default=list(C.A36C_LAYER_SWEEP))
    ap.add_argument("--scopes", nargs="*", default=list(PATCH_SCOPES))
    ap.add_argument("--n-bystanders", type=int, default=C.A36C_N_BYSTANDERS)
    ap.add_argument("--smoke-dryrun", action="store_true", help="CPU dry-run (no base-model load)")
    args = ap.parse_args()

    if args.smoke_dryrun:
        write_dryrun(args.cells, args.layers)
        return

    A36C_DIR.mkdir(parents=True, exist_ok=True)
    from explore_persona_space.analysis.gate_io import load_cell

    for cell in args.cells:
        gated, reason = parity_gate_passed(cell)
        # source + N representative bystanders (plan §4 scope decision)
        sc = load_cell(cell, verify_sha=True)
        ctx_ids = list(sc.tensors["context_ids"])
        src = sc.source_ctx_id
        bystanders = [c for c in ctx_ids if c != src][: args.n_bystanders]
        sc.free()
        contexts = [src, *bystanders]
        res = _run_patch_gpu(cell, contexts, args.layers, args.scopes)
        rec = {
            "cell": cell,
            "behavior": C.behavior_for_cell(cell),
            "column": C.column_for_cell(cell),
            "layers": args.layers,
            "scopes": args.scopes,
            "variants": list(PATCH_VARIANTS),
            "parity_gate_passed": gated,
            "parity_gate_reason": reason,
            "rows": res["rows"],
            "git_commit": _git_commit(),
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "dry_run": False,
            "trusted": gated,  # the verdict is trusted ONLY if the parity gate passed
        }
        outp = A36C_DIR / f"{cell}.json"
        with open(outp, "w") as f:
            json.dump(rec, f, indent=1)
        logger.info("[a36c] %s parity_gate=%s rows=%d -> %s", cell, gated, len(res["rows"]), outp)


if __name__ == "__main__":
    main()
