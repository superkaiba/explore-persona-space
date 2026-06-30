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
import contextlib
import datetime as dt
import json
import logging
import os
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


def _suppressed_record(cell: str, layers: list[int], scopes: list[str], reason: str) -> dict:
    """Blocker 1: the record written when the parity gate FAILs and A3.6c HALTs for
    this cell BEFORE any model load. Backward-compatible with the aggregate / figures
    pipeline (an f_CV-bearing record with the fields null + a `skipped` marker)."""
    return {
        "cell": cell,
        "behavior": C.behavior_for_cell(cell),
        "column": C.column_for_cell(cell),
        "layers": layers,
        "scopes": list(scopes),
        "variants": list(PATCH_VARIANTS),
        "parity_gate_passed": False,
        "parity_gate_reason": reason,
        "skipped": True,  # HALTED before any forward pass — the f_CV verdict is suppressed
        "f_CV": None,
        "rows": [],
        "git_commit": _git_commit(),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "dry_run": False,
        "trusted": False,
        "note": "A3.6c HALTED for this cell: parity probe did not PASS (plan §4 step 5). "
        "No model was loaded and no forward pass ran; the input-vs-map verdict is suppressed.",
    }


# How many new tokens to generate for the behavioral E read under the patch.
# The patched generation is judged for behavior-expression (Blocker 2b); a short
# greedy answer is enough for the judge to score (free-gen marker evals run longer,
# but A3.6c reads the behavioral DV, not a marker slot). Env-overridable.

A36C_PATCH_MAX_NEW_TOKENS = int(os.environ.get("EPM_A36C_PATCH_MAX_NEW", "256"))


def _run_patch_gpu(cell: str, contexts: list[str], layers: list[int], scopes: list[str]) -> dict:
    """GPU path (Step 6d): the live causal context-vector patch. Imported lazily so
    --smoke-dryrun never loads torch/peft. Reads BOTH the activation DV `v` (f_cv_v)
    AND the behavioral DV `E` (judge-positive rate over the patched generation,
    Blocker 2b) per (ctx, layer, scope, variant). Patches use the REAL #664 context
    prompt (Blocker 2), NOT a synthetic 'Hello.'."""
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
    # one judge pass per cell over ALL patched completions (Blocker 2b) — collect
    # the (row-index -> {question, completion}) here, judge at the end.
    e_jobs: list[tuple[int, str, str]] = []  # (row_idx, question, patched_completion)
    try:
        ctx_ids = list(sc.tensors["context_ids"])
        c_base = sc.tensors["c_C_base"].to(torch.float64)  # (C,L,d)
        c_trn = sc.tensors["c_C_trained"].to(torch.float64)
        v_plus = sc.tensors["v_plus"].to(torch.float64)
        v0 = sc.tensors["v0"].to(torch.float64)
        probe_q = next(iter(sc.tensors["battery_probes"]))  # the #664 probe (real prompt)
        rng = np.random.default_rng(42)

        for ctx_id in contexts:
            if ctx_id not in ctx_ids:
                continue
            ci = ctx_ids.index(ctx_id)
            # the REAL #664 chat prompt for THIS context (Blocker 2) — last-input-token
            # is the c_C slot, matching how #664 captured the store.
            msgs = C.context_chat_messages(ctx_id, probe_q)
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(
                base.device
            )
            for layer in layers:
                d_shift = (v_plus[ci, layer] - v0[ci, layer]).numpy()
                d_norm = d_shift / (np.linalg.norm(d_shift) + 1e-30)
                for scope in scopes:
                    # patch each variant's context vector into the residual stream at
                    # layer L (last-input-token or full-context-span) via a forward hook.
                    for variant in PATCH_VARIANTS:
                        patched_v, patched_text = _one_patch(
                            tok,
                            base,
                            trained,
                            ids,
                            layer,
                            scope,
                            variant,
                            c_base[ci, layer],
                            c_trn[ci, layer],
                            rng,
                            c_base,
                        )
                        f_cv_v = (
                            float((patched_v - v0[ci, layer].numpy()) @ d_norm)
                            / (np.linalg.norm(d_shift) + 1e-30)
                            if patched_v is not None
                            else None
                        )
                        row_idx = len(out_rows)
                        out_rows.append(
                            {
                                "context_id": ctx_id,
                                "layer": layer,
                                "scope": scope,
                                "variant": variant,
                                "f_cv_v": f_cv_v,
                                "e_read": None,  # filled by the judge pass below
                            }
                        )
                        if patched_text is not None:
                            e_jobs.append((row_idx, probe_q, patched_text))

        # ── Blocker 2b: behavioral E read — judge ALL patched completions ONCE ──
        _judge_patched_completions(cell, out_rows, e_jobs)
        return {"rows": out_rows}
    finally:
        sc.free()


def _judge_patched_completions(cell: str, out_rows: list, e_jobs: list) -> None:
    """Run the project judge over the patched generations (Blocker 2b) and write the
    per-row judge-positive `e_read` (1.0 if the judge labels the patched completion
    >= the threshold, else 0.0). Uses the SAME column->judge-system map judge_E.py
    pins (imported from there) + the same batch_judge client. A column with no judge
    DV (marker) leaves e_read None."""
    if not e_jobs:
        return
    column = C.column_for_cell(cell)
    import issue665_judge_E as JE

    judge_system = JE.JUDGE_SYSTEM_BY_COLUMN.get(column)
    if judge_system is None:
        # marker / unknown column: no behavioral judge DV (degenerate arm) — leave None.
        return
    from explore_persona_space.eval.batch_judge import judge_completions_batch

    # batch_judge groups by {persona: {question: [completions]}}; encode the row_idx
    # into a unique question key so each patched completion is judged independently
    # and joined back by index.
    completions: dict[str, dict[str, list[str]]] = {cell: {}}
    idx_for_q: dict[str, int] = {}
    for row_idx, question, text in e_jobs:
        qkey = f"row{row_idx:04d}::{question}"
        completions[cell][qkey] = [text]
        idx_for_q[qkey] = row_idx
    cache_dir = A36C_DIR / ".judge_cache" / cell
    cache_dir.mkdir(parents=True, exist_ok=True)
    # judge_completions_batch returns the per-persona AGGREGATE; the per-completion
    # scores we need for the per-row positive read live in save_raw["all_scores"].
    judge_completions_batch(
        completions,
        judge_system_prompt=judge_system,
        format_user_msg=JE._format_user_msg,
        judge_model=C.JUDGE_MODEL,
        cache_dir=cache_dir,
        save_raw=cache_dir / "raw_scores.json",
    )
    # judge_completions_batch aggregates per persona (one row here, `cell`); re-read
    # the per-completion scores from save_raw to assign the per-row positive read.
    import json as _json

    raw_p = cache_dir / "raw_scores.json"
    if not raw_p.exists():
        logger.warning("[a36c] %s judge raw scores missing — e_read left None", cell)
        return
    with open(raw_p) as f:
        raw = _json.load(f)
    all_scores = raw.get("all_scores", {})
    # custom_id shape from batch_judge: f"{persona}__{idx:05d}__{comp:02d}"; the
    # questions are enumerated in insertion order, so idx maps back to qkey order.
    qkeys = list(completions[cell].keys())
    for cid, s in all_scores.items():
        # cid = "<cell>__<qidx>__<cidx>"; parse the question index.
        try:
            parts = cid.rsplit("__", 2)
            qidx = int(parts[1])
        except (ValueError, IndexError):
            continue
        if qidx >= len(qkeys):
            continue
        qkey = qkeys[qidx]
        row_idx = idx_for_q[qkey]
        aligned = s.get("aligned")
        if isinstance(aligned, int | float):
            out_rows[row_idx]["e_read"] = 1.0 if aligned >= JE.JUDGE_THRESHOLD else 0.0
            out_rows[row_idx]["e_score"] = float(aligned)


def _one_patch(tok, base, trained, ids, layer, scope, variant, c0, cp, rng, c_base_all):
    """Apply ONE patch variant on the REAL #664 context prompt (`ids`) and read both
    the resulting answer-side residual (`v`, mean over generated tokens at layer L)
    AND the patched completion text (for the behavioral E judge, Blocker 2b).
    Faithful to plan §4.

    Round-3 Blocker B: θ0 variants (`p_up`, `self_c0`) run on PURE BASE WEIGHTS via
    PEFT's ``disable_adapter()`` context manager; θ+ variants (`p_down`,
    `self_cp`, `random_cv`, `norm_matched`) run on base+adapter. Round 2 selected
    ``model = base`` for θ0, but ``PeftModel.from_pretrained(base, ...)`` wraps
    ``base`` IN-PLACE — the adapter modules are attached to ``base.model.*`` and
    ``base.forward(...)`` goes through the LoRA path — so the θ0 variants were NOT
    isolated (P↑ read base+adapter+c⁺, not pure θ0+c⁺; self_c0 was not the
    identity null on θ0 the plan's A3.6c falsifiability predicate requires). We
    now ALWAYS forward through the single wrapped ``trained`` model and toggle the
    adapter per variant, so θ0 is genuinely adapter-free."""

    import numpy as np
    import torch

    # ALWAYS operate on the single wrapped `trained` PeftModel; θ0 isolation comes
    # from the disable_adapter() context below, NOT from a second model handle
    # (round-3 Blocker B). `base` is the same in-place-wrapped object — never the
    # forward target now.
    adapter_disabled = variant in ("p_up", "self_c0")  # θ0 variants → pure base weights
    model = trained
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
    n_prompt = ids.shape[1]
    captured: dict = {"resp_h": []}

    def _patch_hook(_m, _inp, out):
        # During generation, the prompt-token positions only exist in the FIRST
        # forward (the prefill); patch the c_C slot there. `scope=last` patches the
        # last-input-token (the c_C slot); `scope=full` patches the whole context span.
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] >= n_prompt:  # prefill step (the only step that holds the prompt slots)
            if scope == "last":
                hs[:, n_prompt - 1, :] = pv
            else:  # full-context-span upper bound
                hs[:, :n_prompt, :] = pv
        return out

    def _read_hook(_m, _inp, out):
        # collect the layer-L residual at the LAST position of every forward (the
        # generated answer-side token) -> mean over generated tokens = v.
        hs = out[0] if isinstance(out, tuple) else out
        captured["resp_h"].append(hs[0, -1, :].detach().to("cpu", torch.float64).numpy())

    blk = (
        model.base_model.model.model.layers[layer]
        if hasattr(model, "base_model")
        else (model.model.layers[layer])
    )
    h1 = blk.register_forward_hook(_patch_hook)
    h2 = blk.register_forward_hook(_read_hook)
    ids = ids.to(model.device)
    # Round-3 Blocker B: θ0 variants run inside disable_adapter() so EVERY forward
    # pass of generation uses pure base weights (P↑ = pure θ0 patched with c⁺;
    # self_c0 = the identity null on θ0). θ+ variants use a null context (adapter
    # stays enabled). The hook on the underlying layer module fires identically in
    # both regimes — disable_adapter only toggles whether the LoRA delta is applied
    # inside the layer's submodules, which is exactly the θ0-vs-θ+ contrast.
    adapter_ctx = trained.disable_adapter() if adapter_disabled else contextlib.nullcontext()
    with torch.no_grad(), adapter_ctx:
        gen = model.generate(
            ids,
            max_new_tokens=A36C_PATCH_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    h1.remove()
    h2.remove()
    # v = mean answer-side residual over the GENERATED tokens (drop the prefill read,
    # which corresponds to the last prompt position, not an answer token).
    resp = captured["resp_h"][1:] if len(captured["resp_h"]) > 1 else captured["resp_h"]
    v = np.mean(np.stack(resp), axis=0) if resp else None
    text = tok.decode(gen[0, n_prompt:], skip_special_tokens=True)
    return v, text


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
        # ── Blocker 1: the parity gate is the FIRST gate. A FAIL HALTS A3.6c for
        # this cell — write a SUPPRESSED-row record and CONTINUE *before* loading
        # the cell, loading the model, or running any forward pass (plan v3 §4
        # A3.6c step 5: "A parity-probe FAIL halts A3.6c"). Run-and-flag != HALT.
        gated, reason = parity_gate_passed(cell)
        if not gated:
            rec = _suppressed_record(cell, args.layers, args.scopes, reason)
            outp = A36C_DIR / f"{cell}.json"
            with open(outp, "w") as f:
                json.dump(rec, f, indent=1)
            logger.warning(
                "[a36c] %s parity gate FAILED (%s) -> SUPPRESSED (no model load) -> %s",
                cell,
                reason,
                outp,
            )
            continue

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
            "skipped": False,
            "f_CV": None,  # cell-level summary; the per-(ctx,layer,scope,variant) f_cv is in rows
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
