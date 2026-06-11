#!/usr/bin/env python3
"""#602 assumption-8 gate — #474/#406 prompt-reconstruction cross-check.

The loc474 family's source contexts are i406 transformation conditions whose
prompts are RECONSTRUCTED at runtime via
``i406_conditions.build_prompt_for_condition`` (the verbatim #493 reuse). The
plan's §8 risk row + assumption 8 register a runtime validation for that
reconstruction: re-extract base-model last-prompt-token reads under the
reconstructed prompts and reproduce the stored #406 centroid cosine
DISTANCES (``eval_results/issue_406/cosine/C_L{L}.json``) within the #493
cross-check tolerance 3e-3. A mismatch means the reconstructed prompts (or
the last-position indexing) diverge from the prompts the stored #474
behavioral panels were measured under — the loc474 repair substrate would
be invalid, so strict mode aborts BEFORE any GPU sweep spend.

Recipe (mirrors #493's ``reproduce_last_token_cosine_check`` exactly):
full prompt text from ``build_prompt_for_condition`` (class-D rewrites
loaded when a class-D condition is active), tokenize with
``add_special_tokens=False``, ONE prompt-only forward, capture
``hidden_states[layer+1][0, -1]`` in fp32 (the #602 layer convention;
identical to #493's block-hook capture for layers <= 26), centroid =
mean over questions, distance = 1 - cos(centroid_a, centroid_b).

Modes:
- ``--strict`` (production model only): exit nonzero when any pair is
  outside tolerance. The dispatcher passes it iff ``--model-id`` is the
  production model.
- non-strict (CPU stub smoke): the stored values can never reproduce on a
  stub, so the JSON records ``production_model: false`` + the diffs and the
  script exits 0 — Phase 2's production preflight rejects such a file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.analysis import i602_bakeoff as bk  # noqa: E402

logger = logging.getLogger("issue602_i474_crosscheck")

TOLERANCE = 3e-3  # the #493 cross-check bar (plan §8 / assumption 8)
DEFAULT_LAYER = 21  # stored matrices exist at L{0,5,11,15,21,27}; 21 is in bk.LAYERS


def _load_stored_matrix(layer: int) -> dict[str, Any]:
    """Read the stored #406 cosine-distance matrix for one layer."""
    p = REPO / "eval_results" / "issue_406" / "cosine" / f"C_L{layer}.json"
    if not p.exists():
        raise FileNotFoundError(f"stored #406 cosine matrix missing: {p}")
    return json.loads(p.read_text())


def _extract_last_prompt_reads(
    model_id: str, contexts: list[str], questions: list[str], layer: int
) -> dict[str, np.ndarray]:
    """(n_q, H) fp32 last-prompt-token reads per context, #602 conventions."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.i406_conditions import (
        CONDITIONS_BY_ID,
        build_prompt_for_condition,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    rewrites = None
    if any(CONDITIONS_BY_ID[c].cls == "D" for c in contexts):
        from explore_persona_space.experiments.i460_data import load_class_d_rewrites

        rewrites = load_class_d_rewrites()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    out: dict[str, np.ndarray] = {}
    for cid in contexts:
        cond = CONDITIONS_BY_ID[cid]
        rows = []
        for q in questions:
            text = build_prompt_for_condition(cond, q, tokenizer, class_d_rewrites=rewrites)
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                fwd = model(**ids, output_hidden_states=True)
            h = fwd.hidden_states[layer + 1]  # (1, T, H) — #602 layer convention
            assert h.shape[0] == 1 and h.shape[1] == ids["input_ids"].shape[1], h.shape
            rows.append(h[0, -1, :].float().cpu().numpy())
        out[cid] = np.stack(rows, axis=0)
        logger.info("[phase=i474_check] %s: %d reads (H=%d)", cid, len(rows), out[cid].shape[1])
    return out


def evaluate_pairs(
    reads: dict[str, np.ndarray], stored_matrix: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Centroid cosine distances per ordered pair, diffed vs stored values.

    Returns {per_pair, max_abs_diff, n_pairs, ok} — pure function so the
    strict-failure branch is unit-smokeable without a model.
    """
    per_pair: dict[str, Any] = {}
    max_diff = 0.0
    n_pairs = 0
    for a, Xa in reads.items():
        if a not in stored_matrix:
            continue
        for b, Xb in reads.items():
            if a == b or b not in stored_matrix.get(a, {}):
                continue
            mu_a = Xa.mean(axis=0)
            mu_b = Xb.mean(axis=0)
            na, nb = np.linalg.norm(mu_a), np.linalg.norm(mu_b)
            ours = 1.0 if (na < 1e-12 or nb < 1e-12) else float(1.0 - (mu_a @ mu_b) / (na * nb))
            theirs = float(stored_matrix[a][b])
            diff = abs(ours - theirs)
            max_diff = max(max_diff, diff)
            n_pairs += 1
            per_pair[f"{a}__{b}"] = {"ours": ours, "stored": theirs, "abs_diff": diff}
    if n_pairs == 0:
        raise RuntimeError(
            f"no context pairs overlap the stored matrix (contexts {sorted(reads)} vs "
            f"stored {sorted(stored_matrix)[:8]}...) — wrong layer/matrix?"
        )
    return {
        "per_pair": per_pair,
        "max_abs_diff": float(max_diff),
        "n_pairs": n_pairs,
        "ok": bool(max_diff < TOLERANCE),
    }


def main() -> int:
    """CLI: fresh base reads for the i406 contexts vs stored C_L{layer}."""
    parser = argparse.ArgumentParser(description="#602 assumption-8 prompt-reconstruction gate")
    parser.add_argument("--model-id", default=bk.BASE_MODEL_ID)
    parser.add_argument("--contexts", nargs="+", default=list(bk.LOC474_CONTEXTS))
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument(
        "--n-questions",
        type=int,
        default=0,
        help="Truncate the 50-question #406 probe set (smoke only; 0 = all 50)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any pair is outside tolerance (production model only)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    production_model = args.model_id == bk.BASE_MODEL_ID
    if args.strict and not production_model:
        raise SystemExit(
            f"--strict requires the production model ({bk.BASE_MODEL_ID}); got "
            f"{args.model_id!r} — stored #406 values can never reproduce on a stub"
        )

    from explore_persona_space.experiments.i460_data import load_q_test_extended_50

    questions = load_q_test_extended_50()
    truncated = bool(args.n_questions and args.n_questions < len(questions))
    if truncated:
        questions = questions[: args.n_questions]
    if args.strict and truncated:
        raise SystemExit("--strict requires the full 50-question probe set (got truncation)")

    stored = _load_stored_matrix(args.layer)
    reads = _extract_last_prompt_reads(args.model_id, args.contexts, questions, args.layer)
    result = evaluate_pairs(reads, stored["matrix"])

    payload = {
        "layer": args.layer,
        "contexts": args.contexts,
        "n_questions": len(questions),
        "stored_n_probes": stored.get("n_probes"),
        "tolerance": TOLERANCE,
        "model_id": args.model_id,
        "production_model": production_model,
        "strict": args.strict,
        **result,
        "reproducibility": {
            "issue": bk.ISSUE,
            "git_commit": bk.git_sha(REPO),
            "env_versions": bk.env_versions(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    logger.info(
        "[phase=i474_check] %s: max |diff| %.2e over %d pairs (ok=%s, production=%s) -> %s",
        f"L{args.layer}",
        result["max_abs_diff"],
        result["n_pairs"],
        result["ok"],
        production_model,
        out,
    )
    if args.strict and not result["ok"]:
        logger.error(
            "[phase=i474_check] FAIL — reconstructed #474 prompts diverge from the stored "
            "#406 recipe (max |diff| %.2e > %.0e); the loc474 repair substrate is invalid. "
            "Diagnose before any sweep spend.",
            result["max_abs_diff"],
            TOLERANCE,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
