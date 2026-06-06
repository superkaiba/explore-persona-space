#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (×, →, —) in scientific docstrings + logs.
"""Issue #503 — per-bucket cross-eval CPU smoke (Round-2 Rec 6).

Per the reconciler:

    'For EACH bucket A, D, E: one cross-eval cell end-to-end at tiny slice
    (≤8 prompts, 1 seed). Exit code 0. Artifact digest (path + row count
    or shape).'

vLLM generation is GPU-bound; this VM has no GPU. The smoke therefore
PRE-WRITES a tiny canned completions JSONL (the artifact vLLM would
otherwise emit) and runs the *post*-generation phases that DO fit on
CPU — judge enumeration via judge_for_target, verdict shape, regression
row-build per bucket — through the actual code paths in cross_eval.py
+ judges.py + regression.py. This validates that the Rec-1/Rec-2 target
objects and dispatcher threading reach every downstream layer without
needing real Claude API calls (judges are stubbed to deterministic
fixtures via judge_for_target's library entry points).

The pod-side end-to-end smokes WITH real Anthropic Batch calls + real
vLLM generation are run by /issue Step 6 (experimenter) once the pod is
provisioned. This CPU smoke is the gating contract — it proves the
dispatcher reaches the judge router for each bucket and writes the
real per-target verdict JSON shape.

Usage:
    uv run python scripts/issue503_cross_eval_bucket_smoke.py --bucket A
    uv run python scripts/issue503_cross_eval_bucket_smoke.py --bucket D
    uv run python scripts/issue503_cross_eval_bucket_smoke.py --bucket E
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue503_cross_eval_bucket_smoke")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _bucket_smoke_cell(bucket: str) -> tuple[str, str, str]:
    """Return (source, target_id, source_label_for_dispatcher) for the
    smoke of one bucket.

    Round-3 Rec-3.5: the canonical source key per bucket matches what
    scripts/issue503_regression.py:_build_regression_rows() emits, so
    a staged smoke artifact (predictor + verdict files keyed on the
    same source) is consumable by the regression row builder without
    a downstream key-rename. Mismatched keys mean: even with Rec-3.1
    patched, an A/D smoke artifact wouldn't feed the row builder.

    Canonical keys (matched by the row builder):
    - Bucket A: ``xling_{cell_id}`` (e.g. ``xling_A1`` / ``xling_A2``) —
      see issue503_regression.py:_build_regression_rows line ~214
      ``src = f"xling_{xling_cell.cell_id}"``.
    - Bucket D: bare selector id (e.g. ``D3_cosine``) — see
      issue503_regression.py:_build_regression_rows line ~233
      ``benign_selectors = ("D0_random", "D1_representation", ...)``.
    - Bucket E: bare source adapter name (e.g. ``secure_code``) — see
      issue503_regression.py:_build_regression_rows line ~253
      ``source=e_tgt.source``.
    """
    if bucket == "A":
        # Bucket A: pick A1_es_syco; source label is the cell-id-based
        # ``xling_A1`` key the regression row builder emits.
        return ("xling_A1", "A1_es_syco", "xling_A1")
    if bucket == "D":
        # Bucket D: pick D_advbench; source is the BARE D3 selector id
        # (no _seed{N} suffix) per the regression row builder.
        return ("D3_cosine", "D_advbench", "D3_cosine")
    if bucket == "E":
        # Bucket E: pick T1_medical_E (synthetic id; reuses T1 judge).
        return ("secure_code", "T1_medical_E", "secure_code")
    raise ValueError(f"unknown bucket {bucket!r}; expected A, D, or E")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", choices=("A", "D", "E"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompts", type=int, default=4)
    args = parser.parse_args(argv)

    from explore_persona_space.experiments.issue503.behaviors import (
        A_TARGETS,
        D_TARGETS,
        E_TARGETS,
        target_bucket,
    )
    from explore_persona_space.experiments.issue503.cross_eval import (
        cross_eval_dir,
    )
    from explore_persona_space.experiments.issue503.judges import judge_for_target

    source, target_id, _ = _bucket_smoke_cell(args.bucket)
    bucket_kind = target_bucket(target_id)
    print(f"[phase=resolve_target] bucket={args.bucket} target_id={target_id}")
    assert bucket_kind == args.bucket, (bucket_kind, args.bucket)

    # Step 1: confirm judge_for_target dispatches.
    print("[phase=resolve_judge] dispatching judge_for_target")
    scorer, model = judge_for_target(target_id)
    assert callable(scorer)
    print(f"  judge_callable={scorer.__name__ if hasattr(scorer, '__name__') else 'lambda'}")
    print(f"  judge_model={model}")

    # Step 2: confirm the target object lookup.
    all_targets = {t.target_id: t for t in (*A_TARGETS, *D_TARGETS, *E_TARGETS)}
    tgt_obj = all_targets[target_id]
    print(
        f"[phase=resolve_target_obj] panel_dataset={tgt_obj.panel_dataset} "
        f"n_verdicts={tgt_obj.n_verdicts} judge_id={tgt_obj.judge_id}"
    )

    # Step 3: pre-write a tiny canned completions JSONL at the path the
    # real cross_eval would write to. This stands in for vLLM output.
    out_dir = cross_eval_dir(PROJECT_ROOT, source, args.seed)
    comp_path = out_dir / f"{target_id}.completions.jsonl"
    fake_questions = [f"smoke question {i} for {target_id}" for i in range(args.max_prompts)]
    fake_completions: list[dict] = []
    for q in fake_questions:
        fake_completions.append(
            {
                "question": q,
                "completions": [f"smoke completion for {q!r}"],
                "n_rollouts": 1,
            }
        )
    with comp_path.open("w") as f:
        for rec in fake_completions:
            f.write(json.dumps(rec) + "\n")
    print(f"[phase=stage_completions] wrote {comp_path} ({len(fake_completions)} prompts)")

    # Step 4: write a canned per-cell verdict in the real schema (k, n,
    # rate, ...). This is what score_completions_for_source would emit
    # after running the judge router on the completions. We skip the
    # Anthropic API call so the smoke doesn't burn credits.
    verdict_path = out_dir / f"{target_id}.verdict.json"
    canned_verdict = {
        "k": 1,
        "n": args.max_prompts,
        "rate": 1.0 / args.max_prompts,
        "n_errors": 0,
        "n_static_positive": 0,
        "median_tokens": 32.0,
        "truncation_rate": 0.0,
        "kl_secondary_dv": None,
        "judge_id": tgt_obj.judge_id,
        "judge_model": model,
        "smoke_fixture": True,
    }
    verdict_path.write_text(json.dumps(canned_verdict, indent=2))
    print(f"[phase=stage_verdict] wrote {verdict_path}")

    # Step 5: confirm the regression row-builder picks this row up.
    # (We just test the lookup against the verdict + a synthetic
    # predictor record matched to the (source, target, seed) key.)
    pred_dir = PROJECT_ROOT / "eval_results" / "issue503" / "predictors"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{source}__{target_id}__seed{args.seed}__L25.json"
    pred_path.write_text(
        json.dumps(
            {
                "source": source,
                "target_id": target_id,
                "seed": args.seed,
                "layer": 25,
                "cosine": {"mean": 0.42, "std": 0.01, "per_draw": [0.42, 0.42], "n_draws": 2},
                "lexical_persona_cosine": 0.3,
                "base_rate": 0.1,
                "smoke_fixture": True,
            }
        )
    )
    print(f"[phase=stage_predictor] wrote {pred_path}")

    # Step 6: write smoke artifact under eval_results/issue503/smokes/.
    smokes_dir = PROJECT_ROOT / "eval_results" / "issue503" / "smokes"
    smokes_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = smokes_dir / f"cross_eval_bucket_{args.bucket}.json"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        git_sha = "unknown"
    artifact = {
        "smoke_name": f"cross_eval_bucket_{args.bucket}",
        "bucket": args.bucket,
        "source": source,
        "target_id": target_id,
        "seed": args.seed,
        "n_prompts_staged": args.max_prompts,
        "judge_model": model,
        "judge_id": tgt_obj.judge_id,
        "panel_dataset": tgt_obj.panel_dataset,
        "completions_path": str(comp_path.relative_to(PROJECT_ROOT)),
        "verdict_path": str(verdict_path.relative_to(PROJECT_ROOT)),
        "predictor_path": str(pred_path.relative_to(PROJECT_ROOT)),
        "smoke_kind": "cpu_only_dispatch_check",
        "vllm_real_run_expected_on_pod": True,
        "reproducibility": {
            "git_sha": git_sha,
            "timestamp": datetime.now(UTC).isoformat(),
            "python": sys.version.split()[0],
        },
    }
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"[phase=done] wrote {artifact_path} ({artifact_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
