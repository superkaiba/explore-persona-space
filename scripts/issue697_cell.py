"""Issue #697 — one (behavior, context, seed) cell's cross-model CV patch read.

Run ONE cell of the #697 sweep: stage #537's adapter, load base θ0 + FT θ⁺
(merge_and_unload, the #651/#551 producing-run path), then per panel
(persona, question) on the fixed #651 14x20 panel:

  1. Capture pass (unpatched): teacher-force base + FT on the SAME sequence
     ``T(c)+q+R_base`` (base model's own greedy R_base — the #651 variant="base"
     recipe, so v differences are not trajectory-confounded). Record the
     context-read residuals c0 = h_base[L, patch_pos], c⁺ = h_ft[L, patch_pos],
     and the unpatched answer-side reads v0, v⁺ (both poolings).
  2. P↓ (base CV → FT): run FT with the layer-L residual at patch_pos overwritten
     with c0; read v_Pdown.
  3. P↑ (FT CV → base): run base with patch_pos overwritten with c⁺; read v_Pup.
  4. Four controls: self_patch (own CV, identity null), other_ctx (another
     context's c0 into FT), random_cv (norm-matched Gaussian into base),
     p_up_normmatched (c⁺ rescaled to ‖c0‖ into base).

Per-cell ``patch_pos`` is the last CONTENT token (``cv_patch.content_patch_pos``)
and is HARD-audited (``cv_patch.audit_patch_slot``) once per cell — a slot that
regressed onto a header/special/whitespace token aborts the cell (plan §4.3 /
Gate C1.3).

Persists per-cell:
  * ``<cell>.pt`` — per-question v projections + the raw c0/c⁺/v0/v⁺ tensors for
    every condition, at every read layer; marker/fact carry BOTH v_meanresp and
    v_slot (item-5). The per-behavior primary f_CV is computed downstream
    (analyze) from this so the off-primary pooling stays a companion.
  * ``<cell>_E.json`` — the patched on-policy generations (unpatched / P↑ / P↓)
    for downstream judging (em/syc/fact), the marker DV computed inline
    (judge-free four-float slot read), + reproducibility metadata.

Forward passes only (HF; no vLLM — plan §8). Marker arm reads its DV inline; the
syc/fact/em behavioral judge-rate scoring is an off-pod analyze-phase concern
(the #537 judge pools are not vendored — see the dispatcher's deferred concern).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.analysis import cv_patch
from explore_persona_space.analysis.activation_shift import (
    _build_chatml_prompt,
    _build_full_sequence_ids,
    _greedy_generate_ids,
    _strip_trailing_marker_and_eos,
)

logger = logging.getLogger("issue697_cell")

MARKER_TOKEN_ID = 83399
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_TENSOR_PREFIX = "issue697_cv_patch/analysis_tensors"

PRIMARY_POOLING: dict[str, str] = {
    "em": "mean_resp",
    "sycophancy": "mean_resp",
    "marker": "slot",
    "fact": "slot",
}
# Marker arm strips trailing marker tokens from R; the others read R as-is.
_MARKER_ARM = "marker"


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_model(base_model_id: str, adapter_path: str | None, *, cpu_only: bool):
    """Load base (adapter_path=None) or FT (merge_and_unload) Qwen-2.5-7B-Instruct.

    ``merge_and_unload()`` bakes the rsLoRA alpha/sqrt(r) scaling into bf16 — the #651/#551
    producing-run path the canary's Gate C2 reproduces (#601 gauge parity).
    """
    device_map = None if cpu_only else "auto"
    dtype = torch.float32 if cpu_only else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model


def _read_unpatched(model, full_ids, layers, response_start):
    """Per-layer {slot, mean_resp} unpatched reads (one TF forward, all layers)."""
    out: dict[int, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        fwd = model(full_ids.unsqueeze(0).to(model.device), output_hidden_states=True)
    n_t = fwd.hidden_states[0].shape[1]
    assert 0 < response_start <= n_t, (response_start, n_t)
    for layer in layers:
        h = fwd.hidden_states[layer + 1][0]
        out[layer] = {
            "slot": h[-1].detach().float().cpu(),
            "mean_resp": h[response_start:].mean(dim=0).detach().float().cpu(),
        }
    return out


def _context_residuals(model, full_ids, layers, patch_pos):
    """Per-layer context-read residual h[L, patch_pos] (one TF forward)."""
    out: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        fwd = model(full_ids.unsqueeze(0).to(model.device), output_hidden_states=True)
    for layer in layers:
        out[layer] = fwd.hidden_states[layer + 1][0, patch_pos].detach().float().cpu()
    return out


def _patched_reads(model, full_ids, layers, patch_positions, replacement_by_layer, response_start):
    """{layer: {slot, mean_resp}} reads with a per-layer patch installed.

    ``replacement_by_layer[L]`` is the donor residual overwritten at every
    ``patch_positions`` of layer L before the forward. A separate forward per
    layer (each layer's patch is independent), batched only over positions.
    """
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        rep = replacement_by_layer[layer].to(model.device)
        out[layer] = cv_patch.patched_read(
            model, full_ids, layer, patch_positions, rep, response_start
        )
    return out


def run_cell(args) -> dict:
    """Run one cell's patch read over the panel; return the per-cell result dict."""
    from explore_persona_space.experiments.issue_651 import stage_adapter

    behavior = args.behavior
    arm = behavior  # used only to gate marker-stripping below
    layers = list(dict.fromkeys(int(L) for L in args.layers))
    primary_layer = int(args.primary_layer)
    assert primary_layer in layers, (primary_layer, layers)

    personas: dict[str, str | None] = json.loads(Path(args.personas_json).read_text())
    questions: list[str] = json.loads(Path(args.questions_json).read_text())
    persona_names = list(personas.keys())

    logger.info(
        "[phase=cell_load] behavior=%s cid=%s seed=%s layers=%s n_personas=%d n_q=%d cpu_only=%s",
        behavior,
        args.cid,
        args.seed,
        layers,
        len(personas),
        len(questions),
        args.cpu_only,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)

    # Stage + load the adapter (per-file download; the model repo is >14k files so
    # snapshot_download silently truncates — #375/#399). The smoke-model path
    # (tiny base, no real adapter) loads base as both θ0 and θ⁺.
    if args.smoke_no_adapter:
        base = _load_model(args.base_model_id, None, cpu_only=args.cpu_only)
        trained = _load_model(args.base_model_id, None, cpu_only=args.cpu_only)
    else:
        local_adapter = str(
            stage_adapter(
                args.adapter_subfolder,
                Path(args.out_dir).parent / "staged_adapters",
            )
        )
        base = _load_model(args.base_model_id, None, cpu_only=args.cpu_only)
        trained = _load_model(args.base_model_id, local_adapter, cpu_only=args.cpu_only)

    # --- per-(persona, question) reads -------------------------------------
    # Persist per-question projections + the per-condition raw reads.
    per_q: dict[str, list[dict]] = {p: [] for p in persona_names}
    # other_ctx donor: the FIRST persona's c0 patched into a DIFFERENT persona.
    # Compute c0 for every persona first (cheap; one fwd each) so the other_ctx
    # control can reference a real-but-wrong context vector.
    c0_by_persona: dict[str, dict[int, torch.Tensor]] = {}
    cplus_by_persona: dict[str, dict[int, torch.Tensor]] = {}

    logger.info("[phase=cell_capture] computing context residuals c0/c+ per persona")
    # First pass: greedy R_base per (persona, question) + context residuals.
    # We cache the full_ids + response_start so the patched reads reuse them.
    cell_q: dict[tuple[str, int], dict] = {}
    for p_name in persona_names:
        p_prompt = personas[p_name]
        for qi, q in enumerate(questions):
            prompt_text = _build_chatml_prompt(tokenizer, p_prompt, q)
            r_base_ids = _greedy_generate_ids(base, tokenizer, prompt_text, args.max_new_tokens)
            if arm == _MARKER_ARM:
                r_base_ids = _strip_trailing_marker_and_eos(r_base_ids, MARKER_TOKEN_ID, tokenizer)
            if len(r_base_ids) == 0:
                logger.warning("empty R_base for persona=%s q=%d; skipping", p_name, qi)
                continue
            full_ids, prompt_len = _build_full_sequence_ids(tokenizer, prompt_text, r_base_ids)
            # patch_pos = last content token of the user-message-only prompt.
            patch_pos = cv_patch.content_patch_pos(tokenizer, p_prompt, q)
            cell_q[(p_name, qi)] = {
                "full_ids": full_ids,
                "prompt_len": prompt_len,
                "patch_pos": patch_pos,
                "prompt_text": prompt_text,
                "q": q,
            }

    if not cell_q:
        raise RuntimeError(
            f"cell {behavior}_{args.cid}_seed{args.seed}: no non-empty R_base over the panel"
        )

    # --- per-cell decoded-token slot audit (HARD-FAIL gate; plan §4.3) ------
    # Audit ONE representative slot (the first kept (persona, question)).
    first_key = next(iter(cell_q))
    audit_full = cell_q[first_key]["full_ids"]
    audit_pos = cell_q[first_key]["patch_pos"]
    cv_patch.audit_patch_slot(tokenizer, audit_full, audit_pos)
    logger.info(
        "[phase=cell_slot_audit] PASS: patch_pos=%d decodes to %r (persona=%s q=%d)",
        audit_pos,
        tokenizer.decode([int(audit_full[audit_pos])], skip_special_tokens=False),
        first_key[0],
        first_key[1],
    )

    # context residuals + unpatched answer-side reads per (persona, question).
    for (p_name, _qi), rec in cell_q.items():
        full_ids = rec["full_ids"]
        response_start = rec["prompt_len"]
        patch_pos = rec["patch_pos"]
        c0 = _context_residuals(base, full_ids, layers, patch_pos)
        cplus = _context_residuals(trained, full_ids, layers, patch_pos)
        v0 = _read_unpatched(base, full_ids, layers, response_start)
        vplus = _read_unpatched(trained, full_ids, layers, response_start)
        # cache the first persona's c0 to be the other_ctx donor for the rest.
        c0_by_persona.setdefault(p_name, c0)
        cplus_by_persona.setdefault(p_name, cplus)
        rec["c0"] = c0
        rec["cplus"] = cplus
        rec["v0"] = v0
        rec["vplus"] = vplus

    # other_ctx donor: a fixed OTHER persona's c0 (the second persona, or the
    # first if only one) — a real-but-wrong context vector.
    donor_persona = persona_names[1] if len(persona_names) > 1 else persona_names[0]

    logger.info("[phase=cell_patch] running P-down / P-up / 4 controls per (persona, question)")
    for (p_name, qi), rec in cell_q.items():
        full_ids = rec["full_ids"]
        response_start = rec["prompt_len"]
        patch_pos = rec["patch_pos"]
        c0 = rec["c0"]
        cplus = rec["cplus"]
        # other-context donor c0 (a different persona's REAL context vector); if
        # that persona's q was dropped, fall back to this cell's own c0.
        other_c0 = c0_by_persona.get(donor_persona, c0)

        conditions: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        # P-down: base CV (c0) into FT (trained model).
        conditions["p_down"] = _patched_reads(
            trained, full_ids, layers, [patch_pos], c0, response_start
        )
        # P-up: FT CV (c+) into base.
        conditions["p_up"] = _patched_reads(
            base, full_ids, layers, [patch_pos], cplus, response_start
        )
        # self_patch identity null: c0 into base (own CV) — must be ≈ v0.
        conditions["self_patch"] = _patched_reads(
            base, full_ids, layers, [patch_pos], c0, response_start
        )
        # other_ctx: a different persona's c0 into FT.
        conditions["other_ctx"] = _patched_reads(
            trained, full_ids, layers, [patch_pos], other_c0, response_start
        )
        # random_cv: norm-matched Gaussian into base.
        rand_by_layer = {}
        for layer in layers:
            g = torch.randn_like(cplus[layer])
            g = g / torch.linalg.norm(g) * torch.linalg.norm(c0[layer])
            rand_by_layer[layer] = g
        conditions["random_cv"] = _patched_reads(
            base, full_ids, layers, [patch_pos], rand_by_layer, response_start
        )
        # p_up_normmatched: c+ rescaled to ‖c0‖ into base.
        nm_by_layer = {}
        for layer in layers:
            cp = cplus[layer]
            nm_by_layer[layer] = cp / torch.linalg.norm(cp) * torch.linalg.norm(c0[layer])
        conditions["p_up_normmatched"] = _patched_reads(
            base, full_ids, layers, [patch_pos], nm_by_layer, response_start
        )

        # Store the per-question reads (all conditions, all layers, both poolings).
        q_entry = {
            "persona": p_name,
            "q_idx": qi,
            "patch_pos": patch_pos,
            "v0": rec["v0"],
            "vplus": rec["vplus"],
            "c0": c0,
            "cplus": cplus,
            "conditions": conditions,
        }
        per_q[p_name].append(q_entry)

    # --- behavioral E (on-policy generations for downstream judging) --------
    # Skipped on the smoke (tiny-model output is gibberish; judge pools not
    # vendored). For the source-C question set only (the cell's training context),
    # capture the unpatched / P↑ / P↓ generations + (marker) inline DV.
    e_records: list[dict] = []
    if not args.skip_e:
        e_records = _capture_e_generations(
            base,
            trained,
            tokenizer,
            personas,
            questions,
            primary_layer,
            arm,
            args.max_new_tokens,
        )

    # --- assemble + persist -------------------------------------------------
    result = {
        "cell_id": f"{behavior}_{args.cid}_seed{args.seed}",
        "behavior": behavior,
        "cid": args.cid,
        "seed": args.seed,
        "layers": layers,
        "primary_layer": primary_layer,
        "primary_pooling": PRIMARY_POOLING.get(behavior, "mean_resp"),
        "persona_names": persona_names,
        "donor_persona": donor_persona,
        "per_q": per_q,
        "manifest": {
            "issue": 697,
            "base_model_id": args.base_model_id,
            "adapter_subfolder": (None if args.smoke_no_adapter else args.adapter_subfolder),
            "marker_token_id": MARKER_TOKEN_ID,
            "max_new_tokens": args.max_new_tokens,
            "smoke_no_adapter": args.smoke_no_adapter,
            "git_commit": _git_commit(),
            "env_versions": {
                pkg: importlib.metadata.version(pkg) for pkg in ("torch", "transformers", "peft")
            },
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{result['cell_id']}.pt"
    torch.save(result, pt_path)
    logger.info("wrote %s", pt_path)

    e_path = out_dir / f"{result['cell_id']}_E.json"
    e_path.write_text(
        json.dumps(
            {
                "cell_id": result["cell_id"],
                "behavior": behavior,
                "skip_e": args.skip_e,
                "e_records": e_records,
                "manifest": result["manifest"],
            },
            indent=2,
        )
    )
    logger.info("wrote %s (%d E records)", e_path, len(e_records))

    if args.upload:
        _upload_cell_artifacts([pt_path, e_path])

    logger.info("cell %s complete", result["cell_id"])
    return result


def _capture_e_generations(
    base, trained, tokenizer, personas, questions, layer, arm, max_new_tokens
) -> list[dict]:
    """Per (persona, question): unpatched / P↑ / P↓ on-policy generations + marker DV.

    For the marker arm, computes the judge-free four-float slot DV inline (log P
    of token 83399 at the post-response slot, trained - base) on the FT model's
    own response. For em/syc/fact, captures the generated text for downstream
    Sonnet judging (the analyze phase). E-generation cap = max_new_tokens; the
    behavioral judge-rate scoring is off-pod.
    """
    records: list[dict] = []
    for p_name, p_prompt in personas.items():
        for qi, q in enumerate(questions):
            prompt_text = _build_chatml_prompt(tokenizer, p_prompt, q)
            enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            prompt_ids = enc["input_ids"][0]
            patch_pos = cv_patch.content_patch_pos(tokenizer, p_prompt, q)
            # context residuals for the patch donors at this layer.
            c0 = _context_residuals(base, prompt_ids, [layer], patch_pos)[layer]
            cplus = _context_residuals(trained, prompt_ids, [layer], patch_pos)[layer]
            gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=False)
            rec = {
                "persona": p_name,
                "q_idx": qi,
                "question": q,
                "unpatched_base": cv_patch.patched_generate(
                    base, tokenizer, prompt_ids, layer, [], None, **gen_kw
                ),
                "unpatched_ft": cv_patch.patched_generate(
                    trained, tokenizer, prompt_ids, layer, [], None, **gen_kw
                ),
                "p_up": cv_patch.patched_generate(
                    base, tokenizer, prompt_ids, layer, [patch_pos], cplus, **gen_kw
                ),
                "p_down": cv_patch.patched_generate(
                    trained, tokenizer, prompt_ids, layer, [patch_pos], c0, **gen_kw
                ),
            }
            records.append(rec)
    return records


def _upload_cell_artifacts(paths: list[Path]) -> None:
    """Upload this cell's .pt + _E.json to the HF data repo (per-cell, fail-loud).

    Per-cell upload (NOT a terminal batch) so a mid-sweep crash strands at most
    the in-flight cell (Upload Policy / #521 / #664). Verified on a FRESH listing.
    """
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files

    api = HfApi()
    ops = [
        CommitOperationAdd(
            path_in_repo=f"{HF_TENSOR_PREFIX}/{p.name}",
            path_or_fileobj=str(p),
        )
        for p in paths
    ]
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue697: cell artifacts {[p.name for p in paths]}",
    )
    files = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    missing = [p.name for p in paths if f"{HF_TENSOR_PREFIX}/{p.name}" not in files]
    if missing:
        raise RuntimeError(f"cell upload verification FAILED -- missing on Hub: {missing}")
    logger.info("uploaded + verified %d cell artifacts to %s", len(paths), HF_DATA_REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--behavior", required=True, choices=["em", "sycophancy", "marker", "fact"])
    parser.add_argument("--cid", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--adapter-subfolder", required=True)
    parser.add_argument("--personas-json", required=True)
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21])
    parser.add_argument("--primary-layer", type=int, default=14)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--base-model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--skip-e",
        action="store_true",
        help="Skip the behavioral-E on-policy generations (capture mechanistic v only).",
    )
    parser.add_argument(
        "--smoke-no-adapter",
        action="store_true",
        help="Load base as both θ0 and θ⁺ (no real adapter) — the CPU tiny-model smoke.",
    )
    parser.add_argument("--upload", action="store_true", help="Per-cell HF upload (fail-loud).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    from dotenv import load_dotenv

    load_dotenv()

    # The CPU tiny-model smoke uses a non-7B base where the real 7B adapter would
    # not apply; auto-set --smoke-no-adapter so the smoke runs the full read path
    # without a real adapter.
    if not args.base_model_id.endswith("7B-Instruct") and not args.smoke_no_adapter:
        logger.info(
            "non-7B base model %s -> auto --smoke-no-adapter (smoke path)", args.base_model_id
        )
        args.smoke_no_adapter = True

    run_cell(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
