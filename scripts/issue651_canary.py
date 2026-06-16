"""Issue #651 — canary gates 7a + 7b (must PASS before the extraction sweep).

Gate 7a (the ONLY hard gate; committed-reference reproduction + rsLoRA parity
probe, plan §7). Load the #519 marker adapter THAT ACTUALLY PRODUCED #521's
``same_marker_seed42.json`` — ``superkaiba1/explore-persona-space @
issue_519/marker_seed42/`` (r=8/alpha=16/use_rslora). The producing adapter
identity is authoritative from ``eval_results/issue_521/v2_adapter_provenance.json``
(``marker_seeds["42"] = "issue_519/marker_seed42@main (carry-forward from v1,
unchanged)"``). NOTE the v3 plan §11 named the WRONG adapter — it claimed
``adapters/marker_villain_asst_excluded_medium_c0589c_seed42`` (r=32/alpha=64)
produced the reference, but that is a DIFFERENT LoRA (different rank+alpha ->
different shift delta -> orthogonal U1); the plan misread the
``inputs_manifest.json`` ``base_cosines_questions_source`` field (the
base-cosines QUESTION POOL, ``marker_villain_asst_excluded_medium.jsonl``) as
the adapter. Round-3 root cause (#651): the c0589c read reproduced ~0.426 /
0.918 with cos(U1_re, U1_ref)=0.0096 — exactly the BASE-variant-shaped
geometry of a different adapter, not #521's committed ``same`` numbers.

Read the producing adapter on the SAME object #521 assembled — the 14-persona
I551_PANEL_14 panel -> a [3584, 14] shift matrix (layer 14, on the fixed panel)
-> svd_summary -> assert against #521's
eval_results/issue_521/svd/same_marker_seed42.json:

  |s_top1_frac - 0.32465| <= 0.05
  |mean_cos_to_U1 - 0.58711| <= 0.05
  cos(U1_re, U1_ref) >= 0.95

Reproducing #521's committed numbers proves (i) the adapter applies (a silently-
unapplied adapter reads "no direction") AND (ii) the faithful-PEFT alpha/sqrt(r)
read gauge matches #521/#551/#552's committed regime (incident #601). FAIL ->
HALT before the sweep.

Gate 7b (local adapter-application assert on the #537-lineage adapters the SWEEP
reads; NOT a committed number). Two cells, one per loader layout:
  (a) one i537_marker_* cell at the cell root -> root branch, assert delta_v > 0.
  (b) one i537_em_*/sft_em_adapter/ cell -> nested branch, assert delta_v > 0.
Exercises BOTH loader branches before the sweep (the em-subfolder trap, Risk §1).
Gate 7b doubles as the smoke-architecture canary (these two cells run the full
re-extraction path before the other cells launch).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("issue651_canary")

# The #519 marker adapter that ACTUALLY produced #521's same_marker_seed42.json.
# Authoritative source: eval_results/issue_521/v2_adapter_provenance.json
# (marker_seeds["42"] = "issue_519/marker_seed42@main"). r=8/alpha=16/use_rslora
# (config read on HF at #651 round-3 implementation time). NOT the r=32/alpha=64
# adapters/marker_villain_asst_excluded_medium_c0589c_seed42 the v3 plan named —
# that is a DIFFERENT LoRA (round-3 root cause: it reproduced base-variant-shaped
# geometry with cos(U1_re, U1_ref)=0.0096 because the shift delta is different).
CANARY_ADAPTER = "issue_519/marker_seed42"
# The reference adapter's LoRA regime (asserted post-stage so the canary can
# never silently drift to the wrong-rank adapter again — the round-3 failure
# mode). use_rslora=True is the rsLoRA-gauge probe (incident #601).
REF_ADAPTER_R = 8
REF_ADAPTER_ALPHA = 16
REF_ADAPTER_USE_RSLORA = True
# Gate 7a reproduces #521's committed numbers. REF_JSON and GATE_7A_VARIANT MUST
# describe the SAME read: same_marker_seed42.json was produced with variant="same"
# (s_top1_frac 0.32465 / mean_cos_to_U1 0.58711). Reading variant="base" here
# would reproduce ~0.449 / 0.929 (base_marker_seed42.json) and spuriously HALT
# (incident #651 round-1 code-review C1). The startup assert in gate_7a() pins
# this invariant so the pair can never drift again.
REF_JSON = "eval_results/issue_521/svd/same_marker_seed42.json"
GATE_7A_VARIANT = "same"
TOL = 0.05
U1_AGREEMENT_MIN = 0.95


def _repo_root() -> Path:
    import subprocess

    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    return Path(out)


def _assert_adapter_regime(local_adapter: Path) -> None:
    """Fail loud if the staged adapter's LoRA regime != #521's reference regime.

    The round-3 root cause (#651) was loading a DIFFERENT-rank adapter
    (r=32/alpha=64 instead of the reference's r=8/alpha=16): same training-data
    lineage, different LoRA weights, orthogonal shift direction. This pins the
    Gate 7a adapter to the regime that produced same_marker_seed42.json, so the
    canary can never silently drift to the wrong adapter again. ``use_rslora``
    is asserted too — it is the rsLoRA application-scaling gauge (incident #601).
    """
    cfg = json.loads((local_adapter / "adapter_config.json").read_text())
    got = {
        "r": cfg.get("r"),
        "alpha": cfg.get("lora_alpha"),
        "use_rslora": cfg.get("use_rslora"),
    }
    expected = {
        "r": REF_ADAPTER_R,
        "alpha": REF_ADAPTER_ALPHA,
        "use_rslora": REF_ADAPTER_USE_RSLORA,
    }
    if got != expected:
        raise AssertionError(
            f"Gate 7a adapter regime mismatch at {local_adapter}: got {got}, "
            f"expected {expected} (#521's same_marker_seed42.json was produced by "
            f"issue_519/marker_seed42, r={REF_ADAPTER_R}/alpha={REF_ADAPTER_ALPHA}). "
            f"A different-rank adapter produces a different shift direction and "
            f"cannot reproduce the committed numbers. HALT."
        )


def _extract_panel_matrix(
    adapter_subfolder: str,
    *,
    arm: str,
    variant: str,
    primary_layer: int,
    max_new_tokens: int,
    cpu_only: bool,
    assert_regime: bool = False,
):
    """Stage the adapter locally, read the 14-persona panel -> (M [H,14], order).

    Uses the inherited #602 extractor on the fixed panel with the SAME column
    order #521 used (I551_PANEL_14). ``variant`` MUST match the read whose number
    is asserted against: Gate 7a reproduces #521's committed
    ``same_marker_seed42.json`` (produced with ``variant="same"`` — marker-
    stripping fires for ``arm="marker" + variant="same"``), while the EXTRACT
    sweep + Gate 7b's delta_v>0 loader-branch check read ``variant="base"`` (the
    base-trajectory #602 recipe). The adapter is staged via per-file download
    (NOT snapshot_download — the model repo truncates past ~8k files, #375/#399)
    into a local dir, then loaded by PeftModel.from_pretrained.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.analysis.activation_shift import extract_per_context_shifts
    from explore_persona_space.analysis.svd_direction_constancy import assemble_M
    from explore_persona_space.experiments.issue_651 import (
        BASE_MODEL,
        build_panel_personas,
        build_panel_questions,
        panel_column_order,
        stage_adapter,
    )

    local_adapter = stage_adapter(
        adapter_subfolder, _repo_root() / "outputs" / "issue_651" / "staged_adapters"
    )
    if assert_regime:
        _assert_adapter_regime(local_adapter)
    personas = build_panel_personas()
    questions = build_panel_questions()
    device_map = None if cpu_only else "auto"
    dtype = torch.float32 if cpu_only else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    base.eval()
    from peft import PeftModel

    trained = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    trained = PeftModel.from_pretrained(trained, str(local_adapter))
    trained = trained.merge_and_unload()  # #551 producing-run path (canary parity)
    trained.eval()

    shifts = extract_per_context_shifts(
        base_model=base,
        trained_model=trained,
        tokenizer=tokenizer,
        personas=personas,
        questions=questions,
        arm=arm,
        variant=variant,
        layers=(primary_layer,),
        primary_layer=primary_layer,
        max_new_tokens=max_new_tokens,
    )
    M, order = assemble_M(shifts, persona_order=panel_column_order())
    assert M.shape[1] == 14, (M.shape, order)
    return M, order


def gate_7a(repo_root: Path, *, cpu_only: bool, max_new_tokens: int) -> dict:
    """Committed-reference reproduction on the #519 villain marker adapter."""
    from explore_persona_space.analysis.svd_direction_constancy import cosine, svd_summary

    ref = json.loads((repo_root / REF_JSON).read_text())
    # Mechanizable invariant (round-1 code-review C1): the asserted reference and
    # the read variant MUST agree, else a correctly-applied adapter spuriously
    # HALTs the sweep. same_marker_seed42.json carries variant="same".
    ref_variant = ref.get("variant")
    assert ref_variant == GATE_7A_VARIANT, (
        f"Gate 7a reference variant mismatch: {REF_JSON} was produced with "
        f"variant={ref_variant!r} but gate_7a reads variant={GATE_7A_VARIANT!r}. "
        f"The asserted s_top1_frac/mean_cos_to_U1 belong to the {ref_variant!r} read; "
        f"reading a different variant reproduces different numbers and FALSELY HALTs."
    )
    ref_top = float(ref["s_top1_frac"])
    ref_mean_cos = float(ref["mean_cos_to_U1"])
    ref_u1 = np.asarray(ref["U1"], dtype=np.float64)
    logger.info(
        "[phase=gate7a] reference s_top1_frac=%.5f mean_cos_to_U1=%.5f M_shape=%s",
        ref_top,
        ref_mean_cos,
        ref.get("M_shape"),
    )
    M, _order = _extract_panel_matrix(
        CANARY_ADAPTER,
        arm="marker",  # the #519 marker adapter that produced #521's reference
        variant=GATE_7A_VARIANT,  # "same" -> marker-stripped read == #521's reference
        primary_layer=14,
        max_new_tokens=max_new_tokens,
        cpu_only=cpu_only,
        assert_regime=True,  # pin r=8/alpha=16/use_rslora == #521's regime (round-3 fix)
    )
    summ = svd_summary(M)
    re_top = float(summ["s_top1_frac"])
    re_mean_cos = float(np.mean(summ["cos_to_U1"]))
    u1_cos = abs(cosine(summ["U1"], ref_u1))
    logger.info(
        "[phase=gate7a] reproduced s_top1_frac=%.5f mean_cos_to_U1=%.5f cos(U1_re,U1_ref)=%.5f",
        re_top,
        re_mean_cos,
        u1_cos,
    )
    checks = {
        "s_top1_frac_within_tol": abs(re_top - ref_top) <= TOL,
        "mean_cos_to_U1_within_tol": abs(re_mean_cos - ref_mean_cos) <= TOL,
        "U1_agreement": u1_cos >= U1_AGREEMENT_MIN,
    }
    result = {
        "gate": "7a",
        "ref": {"s_top1_frac": ref_top, "mean_cos_to_U1": ref_mean_cos},
        "reproduced": {
            "s_top1_frac": re_top,
            "mean_cos_to_U1": re_mean_cos,
            "cos_U1_re_vs_ref": u1_cos,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    if not result["pass"]:
        raise AssertionError(
            f"GATE 7a FAILED: {checks}. Reproduced top={re_top:.5f} (ref {ref_top:.5f}), "
            f"mean_cos={re_mean_cos:.5f} (ref {ref_mean_cos:.5f}), U1_cos={u1_cos:.5f}. "
            f"The adapter regime matched #521's (r={REF_ADAPTER_R}/alpha={REF_ADAPTER_ALPHA} — "
            f"asserted pre-SVD), so a FAIL here means the rsLoRA alpha/sqrt(r) read gauge does "
            f"NOT match #521's committed regime on the current PEFT+vLLM stack (incident #601) "
            f"OR the adapter is silently unapplied. HALT."
        )
    logger.info("[phase=gate7a] PASS")
    return result


def gate_7b(repo_root: Path, *, cpu_only: bool, max_new_tokens: int) -> dict:
    """Loader-branch assert on the two #537 layouts (delta_v norm > 0)."""
    from explore_persona_space.experiments.issue_651 import resolve_adapter_subfolder

    cases = [
        ("marker", "default", 42, "root"),
        ("em", "default", 42, "sft_em_adapter-nested"),
    ]
    results = []
    for behavior, cid, seed, layout in cases:
        sub = resolve_adapter_subfolder(behavior, cid, seed)
        logger.info(
            "[phase=gate7b] %s cell %s_%s_seed%s layout=%s -> %s",
            behavior,
            behavior,
            cid,
            seed,
            layout,
            sub,
        )
        # Sanity: the resolved subfolder must carry the nesting branch for em.
        if behavior == "em":
            assert sub.endswith("/sft_em_adapter"), sub
        else:
            assert not sub.endswith("/sft_em_adapter"), sub
        # marker -> arm=marker (strips trailing marker); em -> arm=em (no strip).
        arm = "marker" if behavior == "marker" else "em"
        # Gate 7b validates the SWEEP's loader branch (delta_v>0), so it reads
        # variant="base" — the same base-trajectory read the extract phase uses
        # (NOT the "same" committed-reproduction read of Gate 7a).
        M, _order = _extract_panel_matrix(
            sub,
            arm=arm,
            variant="base",
            primary_layer=14,
            max_new_tokens=max_new_tokens,
            cpu_only=cpu_only,
        )
        delta_v_norm = float(np.linalg.norm(M))  # Frobenius norm of the panel matrix
        ok = delta_v_norm > 0.0
        logger.info(
            "[phase=gate7b] %s layout=%s delta_v Frobenius norm=%.4f -> %s",
            behavior,
            layout,
            delta_v_norm,
            "PASS" if ok else "FAIL",
        )
        if not ok:
            raise AssertionError(
                f"GATE 7b FAILED for {behavior} ({layout}): delta_v norm = {delta_v_norm} ~ 0. "
                f"The loader branch is wrong -- the adapter was silently unapplied. HALT."
            )
        results.append(
            {
                "behavior": behavior,
                "cid": cid,
                "seed": seed,
                "layout": layout,
                "subfolder": sub,
                "delta_v_norm": delta_v_norm,
                "pass": ok,
            }
        )
    logger.info("[phase=gate7b] PASS (both loader branches)")
    return {"gate": "7b", "cells": results, "pass": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--gate", choices=["7a", "7b", "both"], default="both", help="Which gate(s) to run."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    from dotenv import load_dotenv

    load_dotenv()

    repo_root = _repo_root()
    out = {}
    if args.gate in ("7a", "both"):
        out["gate_7a"] = gate_7a(
            repo_root, cpu_only=args.cpu_only, max_new_tokens=args.max_new_tokens
        )
    if args.gate in ("7b", "both"):
        out["gate_7b"] = gate_7b(
            repo_root, cpu_only=args.cpu_only, max_new_tokens=args.max_new_tokens
        )

    out_dir = repo_root / "eval_results" / "issue_651" / "canary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "canary_results.json").write_text(json.dumps(out, indent=2))
    logger.info("[phase=canary_done] wrote %s", out_dir / "canary_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
