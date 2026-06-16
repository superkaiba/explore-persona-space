"""Issue #651 — construct-validity bridge for fact + sycophancy (plan §6.1).

For the two un-validated headline behaviors (fact, sycophancy), the neutral-panel
U1 read could be a generic adapter/SFT direction rather than the behavior-specific
write direction — #552 validated the neutral-panel read for benign/em/marker only,
never fact/sycophancy. So this re-reads each behavior's layer-14 residual shift
(trained - base, seed-42 adapter) on the behavior's CANONICAL #537 diagonal
elicitation surface (eval_results/issue_537/elicitation/{fact,sycophancy}.json),
computes the canonical-surface U1, and compares it to the neutral-panel U1 by
cosine. cos >= 0.5 licenses a "behavior-direction" claim for that behavior;
below 0.5 the analyzer downgrades Q1/Q2 for that behavior to "panel-direction".

No new training (re-reads the EXISTING fact/sycophancy seed-42 adapters). The
neutral-panel U1 comes from the extract-phase shift tensors (this issue's own
eval_results/issue_651/shifts/<cell>.pt -> the cell's panel matrix -> U1).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("issue651_bridge")

ELICITATION = {
    "fact": "eval_results/issue_537/elicitation/fact.json",
    "sycophancy": "eval_results/issue_537/elicitation/sycophancy.json",
}


def _repo_root() -> Path:
    import subprocess

    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())


def _canonical_surface_personas(behavior: str, repo_root: Path) -> dict[str, str | None]:
    """Build the {context_name: system_prompt} dict for the canonical surface.

    fact: 30 plain question strings -> a single 'canonical' context (no system
      prompt) whose questions ARE the elicitation probes.
    sycophancy: 25 dicts {wrong_claim, correction, ...}; the canonical surface
      presents the wrong_claim as the user turn (the model is invited to agree).

    The extractor reads per CONTEXT, so we collapse to ONE context per behavior
    (the canonical surface) and let the per-question loop carry the probes. The
    bridge needs ONE direction per behavior, so a single-context SVD over the
    per-question shifts is exactly the canonical-surface U1.
    """
    return {"canonical": None}


def _canonical_questions(behavior: str, repo_root: Path) -> list[str]:
    data = json.loads((repo_root / ELICITATION[behavior]).read_text())
    probes = data["eval_probes"]
    if behavior == "fact":
        # Plain question strings.
        return [p if isinstance(p, str) else p.get("question", str(p)) for p in probes]
    # sycophancy: present the wrong_claim as a user assertion to elicit agreement.
    out = []
    for p in probes:
        if isinstance(p, str):
            out.append(p)
        else:
            claim = p.get("wrong_claim") or p.get("claim") or ""
            out.append(claim)
    return out


def _extract_canonical_u1(
    behavior: str,
    cid: str,
    seed: int,
    *,
    repo_root: Path,
    primary_layer: int,
    max_new_tokens: int,
    cpu_only: bool,
) -> np.ndarray:
    """Extract ONE (behavior, cid, seed-42) cell's canonical-surface read direction.

    This is the per-CELL canonical direction: the cell's adapter read on the
    behavior's canonical elicitation surface, collapsed to its top singular
    direction over the per-question shifts. The PER-BEHAVIOR canonical U1 is the
    U1 of the matrix assembled from EVERY seed-42 cell's read (see ``main``) —
    plan §9 budgets all 16 fact + 16 sycophancy seed-42 canonical reads, NOT a
    single arbitrary cell (CONCERN bridge-single-canonical-cell).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.analysis.activation_shift import (
        _read_residuals,
        extract_per_context_shifts,
    )
    from explore_persona_space.analysis.svd_direction_constancy import svd_summary
    from explore_persona_space.experiments.issue_651 import (
        BASE_MODEL,
        resolve_adapter_subfolder,
        stage_adapter,
    )

    _ = _read_residuals  # silence unused-import lints; documents the reused reader
    sub = resolve_adapter_subfolder(behavior, cid, seed)
    local_adapter = stage_adapter(sub, repo_root / "outputs" / "issue_651" / "staged_adapters")
    personas = _canonical_surface_personas(behavior, repo_root)
    questions = _canonical_questions(behavior, repo_root)

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
    trained = trained.merge_and_unload()
    trained.eval()

    # arm=em (no marker-stripping) for both fact + sycophancy (generative).
    shifts = extract_per_context_shifts(
        base_model=base,
        trained_model=trained,
        tokenizer=tokenizer,
        personas=personas,
        questions=questions,
        arm="em",
        variant="base",
        layers=(primary_layer,),
        primary_layer=primary_layer,
        max_new_tokens=max_new_tokens,
    )
    # ONE context ("canonical"); its per-question shifts -> (H, n_q) matrix -> U1.
    entry = shifts["canonical"]
    per_q = entry["delta_v_per_q"].detach().float().cpu().numpy()  # (n_q, H)
    M = per_q.T  # (H, n_q)
    return svd_summary(M)["U1"].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--cells",
        nargs="+",
        required=True,
        help="Cell ids (e.g. fact_default_seed42 sycophancy_default_seed42).",
    )
    parser.add_argument("--primary-layer", type=int, default=14)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=4,
        help=(
            "Visible-GPU count for the model load (device_map='auto' shards the "
            "base+trained pair across these GPUs). The 32 canonical-surface cell "
            "reads run sequentially in-process here; the dispatcher passes this "
            "for parity with the other phases' CVD plumbing."
        ),
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--bar", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    from dotenv import load_dotenv

    load_dotenv()

    import torch

    from explore_persona_space.experiments.issue_651 import (
        analysis as i651_analysis,
    )
    from explore_persona_space.experiments.issue_651 import (
        parse_cell_spec,
    )

    repo_root = _repo_root()
    shift_dir = repo_root / "eval_results" / "issue_651" / "shifts"
    out_dir = repo_root / "eval_results" / "issue_651" / "construct_bridge"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[phase=bridge] n_gpus=%d (device_map='auto' spans the visible GPUs; "
        "canonical-surface cell reads run sequentially in-process)",
        args.n_gpus,
    )

    # Group the requested cells by behavior; the bridge is one read per behavior
    # (the neutral-panel U1 is the per-behavior cross-context U1 over its cells).
    cells = [parse_cell_spec(s) for s in args.cells]
    by_behavior: dict[str, list] = {}
    for c in cells:
        by_behavior.setdefault(c.behavior, []).append(c)

    results = {}
    for behavior, beh_cells in by_behavior.items():
        if behavior not in ELICITATION:
            logger.info("[phase=bridge] %s has no canonical surface -> skip (em/marker)", behavior)
            continue
        phase = f"bridge_{behavior}"
        logger.info("[phase=%s] computing neutral-panel U1 across %d cells", phase, len(beh_cells))
        # Neutral-panel U1: per-behavior cross-context U1 from the shift tensors.
        per_context_read: dict[str, np.ndarray] = {}
        for c in beh_cells:
            pt = shift_dir / f"{c.cell_id}.pt"
            if not pt.exists():
                raise FileNotFoundError(
                    f"bridge needs the extract-phase tensor {pt} (run --phase extract first)"
                )
            payload = torch.load(pt, map_location="cpu", weights_only=False)
            read = i651_analysis.cell_read_vector(payload["shifts"], cell_read="u1")
            per_context_read[c.cid] = read
        neutral_u1 = np.asarray(i651_analysis.q1_context_invariance(per_context_read)["U1"])

        # Canonical-surface U1: read EVERY seed-42 cell on the behavior's
        # canonical elicitation surface, assemble the per-behavior canonical
        # matrix (n_seed42_contexts x H) from the per-cell reads, then take its
        # U1 — plan §9 budgets all 16 fact + 16 sycophancy seed-42 canonical
        # reads (CONCERN bridge-single-canonical-cell), NOT a single arbitrary
        # cell. With one seed-42 cell this degenerates to that cell's U1.
        from explore_persona_space.analysis.svd_direction_constancy import svd_summary

        seed42_cells = [c for c in beh_cells if c.seed == 42]
        if not seed42_cells:
            logger.info("[phase=%s] no seed-42 cell -> skip bridge", phase)
            continue
        per_cell_canonical: dict[str, np.ndarray] = {}
        for c in seed42_cells:
            logger.info("[phase=%s] canonical-surface read on %s", phase, c.cell_id)
            per_cell_canonical[c.cid] = _extract_canonical_u1(
                behavior,
                c.cid,
                c.seed,
                repo_root=repo_root,
                primary_layer=args.primary_layer,
                max_new_tokens=args.max_new_tokens,
                cpu_only=args.cpu_only,
            )
        # Assemble the (H x n_contexts) canonical matrix and take its U1.
        canon_order = sorted(per_cell_canonical)
        M_canon = np.stack([per_cell_canonical[cid] for cid in canon_order], axis=1)
        canonical_u1 = (
            svd_summary(M_canon)["U1"].astype(np.float32)
            if M_canon.shape[1] >= 2
            else M_canon[:, 0].astype(np.float32)
        )
        bridge = i651_analysis.construct_bridge_cosine(neutral_u1, canonical_u1, bar=args.bar)
        bridge["behavior"] = behavior
        bridge["neutral_cells"] = [c.cell_id for c in beh_cells]
        bridge["canonical_cells"] = [c.cell_id for c in seed42_cells]
        bridge["n_canonical_cells"] = len(seed42_cells)
        results[behavior] = bridge
        (out_dir / f"{behavior}.json").write_text(json.dumps(bridge, indent=2))
        logger.info(
            "[phase=%s] cos(neutral,canonical)=%.4f bar=%.2f -> %s",
            phase,
            bridge["cos_neutral_vs_canonical"],
            args.bar,
            bridge["label"],
        )

    logger.info("[phase=bridge_done] wrote %d bridge files to %s", len(results), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
