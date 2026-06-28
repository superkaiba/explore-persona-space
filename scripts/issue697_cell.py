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
# Per-cell .pt + _E_metadata.json land under analysis_tensors/ (intermediate
# analysis inputs the analyze phase consumes — Upload Policy #521).
HF_TENSOR_PREFIX = "issue697_cv_patch/analysis_tensors"
# Raw on-policy generations land under the CANONICAL raw_completions/ prefix
# (CLAUDE.md Upload Policy), one file per (cell, condition, persona, question)
# batch (the standing rec #1 / reconciler-upheld fix).
HF_RAW_COMPLETIONS_PREFIX = "issue697_cv_patch/raw_completions"

PRIMARY_POOLING: dict[str, str] = {
    "em": "mean_resp",
    "sycophancy": "mean_resp",
    "marker": "slot",
    "fact": "slot",
}
# Marker arm strips trailing marker tokens from R; the others read R as-is.
_MARKER_ARM = "marker"

# --- E-gen descope (compute-deviation v2 auto-descope) ----------------------
# Restrict the behavioral-E generations to the SOURCE persona + the N closest
# bystanders (5/14 of the panel) so the E-gen wall (the 44x blowup) lands ~1.3 h
# on 4x A100-80. The PRIMARY v-space f_CV is UNCHANGED (100% panel coverage on
# all 128 cells). The source-anchor persona is the canonical default-assistant
# leakage target (always in the 14-panel; open-q 3.7), and the N closest panel
# personas are chosen DETERMINISTICALLY per behavior by COSINE distance on the
# base-model context residual c0 at the primary layer (raw pairwise cosine per
# .claude/rules/persona-distance-metrics.md — descope marker v2 mandates
# "closest-by-#651-COSINE") — stable across the cells of a behavior (the choice
# is per-behavior, not per-cell).
E_SUBSET_SOURCE_ANCHOR = "assistant"
E_SUBSET_N_BYSTANDERS = 4
# The bystander-selection distance metric, persisted in the per-cell manifest so
# the choice is auditable + the descope-adherence is mechanically checkable.
E_SUBSET_METRIC = "cosine"
# syc/fact E-gen token cap (descope (c)): Qwen median response ~150 tok, so the
# 512 cap loses little vs #537's 2048; em keeps 512 (already #537). Marker is
# TF marker-logp (no free gen) and is unaffected by the cap.
E_TOKEN_CAP_BY_BEHAVIOR: dict[str, int] = {
    "em": 512,
    "sycophancy": 512,
    "fact": 512,
    "marker": 512,  # TF marker-logp; cap is the R-generation cap only.
}
# em is a rate over n=5 samples (Betley DV, #537); syc/fact greedy (deterministic,
# #537 DV is over the probe panel, not over samples).
E_SAMPLES_BY_BEHAVIOR: dict[str, int] = {"em": 5, "sycophancy": 1, "fact": 1, "marker": 1}
E_TEMPERATURE_BY_BEHAVIOR: dict[str, float] = {
    "em": 1.0,
    "sycophancy": 0.0,
    "fact": 0.0,
    "marker": 0.0,
}


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


def _context_span_residuals(model, full_ids, layer, patch_positions):
    """{position: h[layer, position]} over the context span (the full_span donor map).

    One TF forward; returns the layer-``layer`` output residual at EVERY position
    in ``patch_positions`` (the [0 .. patch_pos] context span — plan control #4),
    so the full_span condition can overwrite each context position with the donor
    model's residual AT THAT SAME POSITION (a distinct donor vector per position,
    not one broadcast).
    """
    with torch.no_grad():
        fwd = model(full_ids.unsqueeze(0).to(model.device), output_hidden_states=True)
    h = fwd.hidden_states[layer + 1][0]  # (T, H)
    return {int(p): h[int(p)].detach().float().cpu() for p in patch_positions}


def select_e_subset(behavior, c0_by_persona, persona_names, layer) -> dict:
    """Deterministic per-behavior E-subset: source anchor + N closest bystanders.

    The E-gen descope (compute-deviation v2) restricts the behavioral-E
    generations to ``E_SUBSET_SOURCE_ANCHOR`` (the default-assistant leakage
    target, always in the 14-panel) plus the ``E_SUBSET_N_BYSTANDERS`` closest
    panel personas, measured by COSINE distance ``1 - cos(c_p, c_anchor)`` on the
    base-model context residual ``c0`` at the primary ``layer`` (raw pairwise
    cosine per .claude/rules/persona-distance-metrics.md — two persona residual
    vectors, no bank to center; the descope marker v2 mandates
    "closest-by-#651-COSINE"). Per-behavior + per-cell c0 vary slightly, but the
    SELECTION is computed against THIS cell's c0 — the brief requires stability
    across cells of the same behavior, which holds because the panel personas +
    the anchor are fixed and the residual geometry is dominated by the persona
    identity, not the (fixed-panel) training context. The chosen subset + the
    per-persona distances + the metric name are persisted per-cell so the choice
    is auditable.

    Returns ``{"anchor": str, "bystanders": [str], "subset": [str],
    "distances": {persona: float}, "metric": "cosine"}``. Falls back to including
    every panel persona when the anchor's c0 is unavailable (a dropped-question
    cell) — the subset is then the whole panel (no descope), reported as such.
    """
    anchor = E_SUBSET_SOURCE_ANCHOR if E_SUBSET_SOURCE_ANCHOR in persona_names else persona_names[0]
    anchor_c0 = c0_by_persona.get(anchor, {}).get(layer)
    if anchor_c0 is None:
        # No anchor residual (anchor's questions all dropped) — keep the whole
        # panel so we never silently lose E coverage.
        return {
            "anchor": anchor,
            "bystanders": [p for p in persona_names if p != anchor],
            "subset": list(persona_names),
            "distances": {},
            "metric": E_SUBSET_METRIC,
            "descoped": False,
        }
    distances: dict[str, float] = {}
    for p in persona_names:
        if p == anchor:
            continue
        cp = c0_by_persona.get(p, {}).get(layer)
        if cp is None:
            continue
        # Cosine distance 1 - cos(c_p, c_anchor) (raw pairwise per
        # persona-distance-metrics.md): the descope marker v2 mandates
        # closest-by-#651-COSINE. cosine_similarity needs a batch dim.
        cos = torch.nn.functional.cosine_similarity(
            cp.reshape(1, -1).float(), anchor_c0.reshape(1, -1).float(), dim=1
        )
        distances[p] = float(1.0 - cos.item())
    # Deterministic tie-break: distance asc, then persona name asc.
    ranked = sorted(distances.items(), key=lambda kv: (kv[1], kv[0]))
    bystanders = [p for p, _d in ranked[:E_SUBSET_N_BYSTANDERS]]
    subset = [anchor, *bystanders]
    return {
        "anchor": anchor,
        "bystanders": bystanders,
        "subset": subset,
        "distances": distances,
        "metric": E_SUBSET_METRIC,
        "descoped": True,
    }


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


def _build_conditions(base, trained, tokenizer, p_prompt, rec, layers, other_c0) -> dict:
    """All v-space patch conditions for one (persona, question): P↓/P↑ + 4 controls + full_span.

    Conditions (plan §4.8 / control set): ``p_down`` (base CV → FT), ``p_up`` (FT
    CV → base), ``self_patch`` (own CV → base, identity null), ``other_ctx`` (a
    different persona's c0 → FT), ``random_cv`` (norm-matched Gaussian → base),
    ``p_up_normmatched`` (c⁺ rescaled to ‖c0‖ → base), and ``full_span`` (plan
    control #4 — the FT c⁺ overwritten at EVERY context position [0 .. patch_pos]
    with a DISTINCT donor vector per position, the single-slot under-count UPPER
    BOUND). Each value is ``{layer: {slot, mean_resp}}``.
    """
    full_ids = rec["full_ids"]
    response_start = rec["prompt_len"]
    patch_pos = rec["patch_pos"]
    c0 = rec["c0"]
    cplus = rec["cplus"]
    conditions: dict[str, dict] = {}
    conditions["p_down"] = _patched_reads(
        trained, full_ids, layers, [patch_pos], c0, response_start
    )
    conditions["p_up"] = _patched_reads(base, full_ids, layers, [patch_pos], cplus, response_start)
    conditions["self_patch"] = _patched_reads(
        base, full_ids, layers, [patch_pos], c0, response_start
    )
    conditions["other_ctx"] = _patched_reads(
        trained, full_ids, layers, [patch_pos], other_c0, response_start
    )
    rand_by_layer = {}
    for layer in layers:
        g = torch.randn_like(cplus[layer])
        rand_by_layer[layer] = g / torch.linalg.norm(g) * torch.linalg.norm(c0[layer])
    conditions["random_cv"] = _patched_reads(
        base, full_ids, layers, [patch_pos], rand_by_layer, response_start
    )
    nm_by_layer = {}
    for layer in layers:
        cp = cplus[layer]
        nm_by_layer[layer] = cp / torch.linalg.norm(cp) * torch.linalg.norm(c0[layer])
    conditions["p_up_normmatched"] = _patched_reads(
        base, full_ids, layers, [patch_pos], nm_by_layer, response_start
    )
    # full_span: distinct FT donor residual per context position [0 .. patch_pos].
    span_positions = cv_patch.context_span_positions(tokenizer, p_prompt, rec["q"], patch_pos)
    conditions["full_span"] = {}
    for layer in layers:
        span_reps = _context_span_residuals(trained, full_ids, layer, span_positions)
        conditions["full_span"][layer] = cv_patch.patched_read(
            base, full_ids, layer, span_positions, span_reps, response_start
        )
    return conditions


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

    logger.info(
        "[phase=cell_patch] running P-down / P-up / 4 controls + full_span per (persona, q)"
    )
    for (p_name, qi), rec in cell_q.items():
        other_c0 = c0_by_persona.get(donor_persona, rec["c0"])
        conditions = _build_conditions(
            base, trained, tokenizer, personas[p_name], rec, layers, other_c0
        )
        per_q[p_name].append(
            {
                "persona": p_name,
                "q_idx": qi,
                "patch_pos": rec["patch_pos"],
                "v0": rec["v0"],
                "vplus": rec["vplus"],
                "c0": rec["c0"],
                "cplus": rec["cplus"],
                "conditions": conditions,
            }
        )

    # --- behavioral E (on-policy generations for downstream judging) --------
    # Skipped on the smoke (tiny-model output is gibberish; judge pools not
    # vendored). E-gen is DESCOPED (compute-deviation v2) to the source anchor +
    # the 4 closest bystanders (5/14 of the panel) so the E-gen wall lands ~1.3 h
    # on 4x A100-80; the PRIMARY v-space f_CV above stays 100% panel coverage.
    e_subset_info = select_e_subset(behavior, c0_by_persona, persona_names, primary_layer)
    logger.info(
        "[phase=cell_e_subset] anchor=%s bystanders=%s descoped=%s",
        e_subset_info["anchor"],
        e_subset_info["bystanders"],
        e_subset_info["descoped"],
    )
    e_records: list[dict] = []
    if not args.skip_e:
        e_personas = {p: personas[p] for p in e_subset_info["subset"] if p in personas}
        e_records = _capture_e_generations(
            base,
            trained,
            tokenizer,
            e_personas,
            questions,
            primary_layer,
            behavior,
            E_TOKEN_CAP_BY_BEHAVIOR.get(behavior, args.max_new_tokens),
            use_cache=args.use_cache,
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
        "e_subset": e_subset_info,
        "manifest": {
            "issue": 697,
            "base_model_id": args.base_model_id,
            "adapter_subfolder": (None if args.smoke_no_adapter else args.adapter_subfolder),
            "marker_token_id": MARKER_TOKEN_ID,
            "max_new_tokens": args.max_new_tokens,
            "smoke_no_adapter": args.smoke_no_adapter,
            # E-gen descope provenance (compute-deviation v2) — read by the analyzer.
            "e_token_cap": E_TOKEN_CAP_BY_BEHAVIOR.get(behavior, args.max_new_tokens),
            "e_samples_per_probe": E_SAMPLES_BY_BEHAVIOR.get(behavior, 1),
            "e_temperature": E_TEMPERATURE_BY_BEHAVIOR.get(behavior, 0.0),
            "e_subset_anchor": e_subset_info["anchor"],
            "e_subset_bystanders": e_subset_info["bystanders"],
            "e_subset_descoped": e_subset_info["descoped"],
            # Bystander-selection metric (descope marker v2: closest-by-#651-COSINE).
            "e_subset_metric": e_subset_info.get("metric", E_SUBSET_METRIC),
            "e_token_cap_note": (
                "syc/fact E-gen max_new_tokens capped at 512 (compute-deviation v2 "
                "descope c); #537 used 2048 — Qwen median response ~150 tok so the cap "
                "loses little. Methodology footnote for the clean-result caveats."
            ),
            # use_cache decision threaded from the canary's Gate C1.2 (concern #4).
            "use_cache": args.use_cache,
            # Judge deviation vs #537 (Sonnet 4.5 supersedes #537's Haiku for fact/syc).
            "judge_model_deviation": (
                "claude-sonnet-4-5-20250929 for em/syc/fact (CLAUDE.md standing rule; "
                "supersedes #537's Haiku pin for fact/syc)."
            ),
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

    cell_id = result["cell_id"]
    slot_decoded = tokenizer.decode([int(audit_full[audit_pos])], skip_special_tokens=False)
    # --- raw_completions split (standing rec #1) ----------------------------
    # The raw on-policy generations go to the CANONICAL raw_completions/ prefix,
    # ONE file per (cell, condition), so the analyze-phase judge ingests them and
    # CLAUDE.md Upload Policy is honored. The per-cell _E_metadata.json (condition
    # list, patch_pos, slot decoded token, use_cache, subset) stays under
    # analysis_tensors/. Marker E records are numeric TF marker-logp DVs (no free
    # generation), so they live in the metadata, not raw_completions.
    raw_dir = out_dir / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_paths: list[Path] = []
    e_is_generation = behavior != _MARKER_ARM
    if e_is_generation and e_records:
        # Split the generation records by condition: one file per condition holds
        # every (persona, question[, sample]) completion under that condition.
        conditions_in_e = ("unpatched_base", "unpatched_ft", "p_up", "p_down")
        for cond in conditions_in_e:
            rows = []
            for r in e_records:
                comps = r.get(cond)
                if comps is None:
                    continue
                rows.append(
                    {
                        "persona": r["persona"],
                        "q_idx": r["q_idx"],
                        "question": r["question"],
                        "completions": comps if isinstance(comps, list) else [comps],
                    }
                )
            rp = raw_dir / f"{cell_id}_{cond}_seed{args.seed}.json"
            rp.write_text(
                json.dumps(
                    {
                        "cell_id": cell_id,
                        "behavior": behavior,
                        "condition": cond,
                        "seed": args.seed,
                        "rows": rows,
                        "manifest": result["manifest"],
                    },
                    indent=2,
                )
            )
            raw_paths.append(rp)
        logger.info("wrote %d raw_completions files for %s", len(raw_paths), cell_id)

    # --- per-cell E metadata (analysis input) -------------------------------
    e_meta_path = out_dir / f"{cell_id}_E_metadata.json"
    e_meta_path.write_text(
        json.dumps(
            {
                "cell_id": cell_id,
                "behavior": behavior,
                "skip_e": args.skip_e,
                "dv_kind": "marker_logp" if behavior == _MARKER_ARM else "generation",
                "patch_pos": int(audit_pos),
                "slot_decoded_token": slot_decoded,
                "use_cache": args.use_cache,
                "e_subset": e_subset_info,
                # marker DV records live here (numeric, judge-free); generation
                # records are split to raw_completions/ above.
                "marker_e_records": e_records if behavior == _MARKER_ARM else [],
                "raw_completions_files": [p.name for p in raw_paths],
                "manifest": result["manifest"],
            },
            indent=2,
        )
    )
    logger.info("wrote %s (%d E records)", e_meta_path, len(e_records))

    if args.upload:
        _upload_cell_artifacts([pt_path, e_meta_path], raw_paths)

    logger.info("cell %s complete", cell_id)  # NOT [phase=done] (mid-run noise)
    return result


def _marker_logp_at_slot(model, full_ids, layer, patch_positions, replacements) -> float:
    """log P(marker id 83399) at the post-response slot under an optional patch.

    Teacher-forced: the marker DV is judge-free (the #537/#651 marker recipe), so
    no free generation. The slot is the LAST token of ``full_ids`` (the
    marker-stripped FT response ends there; the next-token distribution at that
    position is where ` ※` would be emitted). Returns the log-prob of token
    83399. ``patch_positions`` empty / ``replacements`` None => unpatched.
    """
    handle = None
    if patch_positions and replacements is not None:
        handle = cv_patch.make_cv_patch_hook(
            model.model.layers[layer], patch_positions, replacements
        )
    try:
        with torch.no_grad():
            out = model(full_ids.unsqueeze(0).to(model.device))
        logits = out.logits[0, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)[MARKER_TOKEN_ID]
        return float(logp.cpu())
    finally:
        if handle is not None:
            handle.remove()


def _e_gen_one(
    model, tokenizer, prompt_ids, layer, patch_positions, replacements, knobs, use_cache
):
    """Generate E for one (model, condition): a LIST of completions (n samples).

    em → n=5 sampled at temp=1.0 (Betley rate denominator, #537); syc/fact →
    one greedy completion (deterministic, #537 DV over the probe panel). The
    list shape is uniform so the raw_completions writer + the judge ingest both
    arms identically.
    """
    n = knobs["samples"]
    completions: list[str] = []
    for _ in range(n):
        gen_kw: dict = {"max_new_tokens": knobs["max_new_tokens"]}
        if knobs["do_sample"]:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = knobs["temperature"]
        else:
            gen_kw["do_sample"] = False
        completions.append(
            cv_patch.patched_generate(
                model,
                tokenizer,
                prompt_ids,
                layer,
                patch_positions,
                replacements,
                use_cache=use_cache,
                **gen_kw,
            )
        )
    return completions


def _capture_e_generations(
    base,
    trained,
    tokenizer,
    personas,
    questions,
    layer,
    behavior,
    max_new_tokens,
    *,
    use_cache=True,
) -> list[dict]:
    """Per (persona, question): the behavioral E DV under unpatched / P↑ / P↓.

    MARKER arm (judge-free): a teacher-forced slot read — log P(` ※`, id 83399)
    at the post-response slot of the FT model's OWN marker-stripped response,
    under each condition (NO free generation; the marker DV is a TF log-prob,
    plan §4.5 / marker-leakage-measurement.md). The E-space f_CV^E for marker is
    computed downstream from these log-probs.

    em/syc/fact arms: capture the model's own ON-POLICY generation under each
    condition for downstream Sonnet judging (the off-pod analyze-phase judge runs
    the vendored #537 judge pools). PER-BEHAVIOR decode knobs (standing rec #2 /
    descope c): em do_sample=True temp=1.0 n=5 max=512 (the Betley rate
    denominator); syc/fact greedy max=512 (capped from #537's 2048 per descope c —
    Qwen median ~150 tok). ``use_cache`` is the canary's Gate C1.2 decision
    threaded through (concern #4) — every patched_generate honors it. The
    completions are LISTS (em has 5; syc/fact 1) so the writer + judge are uniform.
    """
    records: list[dict] = []
    knobs = {
        "samples": E_SAMPLES_BY_BEHAVIOR.get(behavior, 1),
        "do_sample": E_TEMPERATURE_BY_BEHAVIOR.get(behavior, 0.0) > 0.0,
        "temperature": E_TEMPERATURE_BY_BEHAVIOR.get(behavior, 0.0),
        "max_new_tokens": max_new_tokens,
    }
    for p_name, p_prompt in personas.items():
        for qi, q in enumerate(questions):
            prompt_text = _build_chatml_prompt(tokenizer, p_prompt, q)
            enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            prompt_ids = enc["input_ids"][0]
            patch_pos = cv_patch.content_patch_pos(tokenizer, p_prompt, q)
            c0 = _context_residuals(base, prompt_ids, [layer], patch_pos)[layer]
            cplus = _context_residuals(trained, prompt_ids, [layer], patch_pos)[layer]
            if behavior == _MARKER_ARM:
                # FT model's own marker-stripped response defines the slot.
                r_ft_ids = _greedy_generate_ids(trained, tokenizer, prompt_text, max_new_tokens)
                r_ft_ids = _strip_trailing_marker_and_eos(r_ft_ids, MARKER_TOKEN_ID, tokenizer)
                full_ids, _plen = _build_full_sequence_ids(tokenizer, prompt_text, r_ft_ids)
                rec = {
                    "persona": p_name,
                    "q_idx": qi,
                    "question": q,
                    "dv_kind": "marker_logp",
                    # trained - base subtraction is done downstream; persist all
                    # four conditions' slot log-probs.
                    "marker_logp_unpatched_ft": _marker_logp_at_slot(
                        trained, full_ids, layer, [], None
                    ),
                    "marker_logp_unpatched_base": _marker_logp_at_slot(
                        base, full_ids, layer, [], None
                    ),
                    "marker_logp_p_up": _marker_logp_at_slot(
                        base, full_ids, layer, [patch_pos], cplus
                    ),
                    "marker_logp_p_down": _marker_logp_at_slot(
                        trained, full_ids, layer, [patch_pos], c0
                    ),
                }
            else:
                rec = {
                    "persona": p_name,
                    "q_idx": qi,
                    "question": q,
                    "dv_kind": "generation",
                    "unpatched_base": _e_gen_one(
                        base, tokenizer, prompt_ids, layer, [], None, knobs, use_cache
                    ),
                    "unpatched_ft": _e_gen_one(
                        trained, tokenizer, prompt_ids, layer, [], None, knobs, use_cache
                    ),
                    "p_up": _e_gen_one(
                        base, tokenizer, prompt_ids, layer, [patch_pos], cplus, knobs, use_cache
                    ),
                    "p_down": _e_gen_one(
                        trained, tokenizer, prompt_ids, layer, [patch_pos], c0, knobs, use_cache
                    ),
                }
            records.append(rec)
    return records


def _upload_cell_artifacts(tensor_paths: list[Path], raw_paths: list[Path]) -> None:
    """Upload this cell's analysis tensors + raw completions to the HF data repo.

    ``tensor_paths`` (.pt + _E_metadata.json) land under ``HF_TENSOR_PREFIX``
    (analysis_tensors/ — the analyze-phase inputs, Upload Policy #521).
    ``raw_paths`` (the per-condition raw generations) land under
    ``HF_RAW_COMPLETIONS_PREFIX`` (raw_completions/ — the CANONICAL prefix, the
    standing-rec #1 fix). ONE batched ``create_commit`` per cell (well under the
    256-commits/hr cap, #664) so a mid-sweep crash strands at most the in-flight
    cell; verified against a FRESH ``list_repo_files`` listing (NOT the `hf` CLI —
    upload-policy.md). Fail-loud on any missing file.
    """
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files

    api = HfApi()
    expected: dict[str, Path] = {}
    ops = []
    for p in tensor_paths:
        path_in_repo = f"{HF_TENSOR_PREFIX}/{p.name}"
        expected[path_in_repo] = p
        ops.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(p)))
    for p in raw_paths:
        path_in_repo = f"{HF_RAW_COMPLETIONS_PREFIX}/{p.name}"
        expected[path_in_repo] = p
        ops.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(p)))
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue697: cell artifacts ({len(ops)} files)",
    )
    files = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    missing = [pir for pir in expected if pir not in files]
    if missing:
        raise RuntimeError(f"cell upload verification FAILED -- missing on Hub: {missing}")
    logger.info(
        "uploaded + verified %d analysis tensors + %d raw_completions files to %s",
        len(tensor_paths),
        len(raw_paths),
        HF_DATA_REPO,
    )


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
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "KV-cache during patched_generate (concern #4 — threaded from the "
            "canary's Gate C1.2 use_cache_production_default). --no-use-cache runs "
            "uncached as the safety net when caching drops/marginally-affects the patch."
        ),
    )
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
