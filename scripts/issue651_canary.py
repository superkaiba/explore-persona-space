"""Issue #651 — canary gates 7a + 7b (must PASS before the extraction sweep).

Gate 7a (the ONLY hard gate; committed-reference reproduction + rsLoRA parity
probe, plan §7). Load the #519 villain-source marker adapter
``superkaiba1/explore-persona-space @ adapters/marker_villain_asst_excluded_medium_c0589c_seed42/``
(r=32/alpha=64/use_rslora — config read at plan time), read it on the SAME
object #521 assembled — the 14-persona I551_PANEL_14 panel -> a [3584, 14]
shift matrix (layer 14, on the fixed panel) -> svd_summary -> assert against
#521's eval_results/issue_521/svd/same_marker_seed42.json:

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

CANARY_ADAPTER = "adapters/marker_villain_asst_excluded_medium_c0589c_seed42"
REF_JSON = "eval_results/issue_521/svd/same_marker_seed42.json"
TOL = 0.05
U1_AGREEMENT_MIN = 0.95


def _repo_root() -> Path:
    import subprocess

    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    return Path(out)


def _extract_panel_matrix(
    adapter_subfolder: str,
    *,
    arm: str,
    primary_layer: int,
    max_new_tokens: int,
    cpu_only: bool,
):
    """Stage the adapter locally, read the 14-persona panel -> (M [H,14], order).

    Uses the inherited #602 extractor on the fixed panel with the SAME column
    order #521 used (I551_PANEL_14). variant='base' (teacher-forced base-
    trajectory read — the #602 recipe). The adapter is staged via per-file
    download (NOT snapshot_download — the model repo truncates past ~8k files,
    #375/#399) into a local dir, then loaded by PeftModel.from_pretrained.
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
        variant="base",
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
        arm="marker",  # the villain-source #519 adapter is a marker implant
        primary_layer=14,
        max_new_tokens=max_new_tokens,
        cpu_only=cpu_only,
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
            f"The rsLoRA alpha/sqrt(r) read gauge does NOT match #521's committed regime "
            f"(incident #601) OR the adapter is silently unapplied. HALT."
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
        M, _order = _extract_panel_matrix(
            sub, arm=arm, primary_layer=14, max_new_tokens=max_new_tokens, cpu_only=cpu_only
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
