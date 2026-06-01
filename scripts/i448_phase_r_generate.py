# em-dash + Qwen marker token " ※" are intentional
"""Task #448 Phase 1 (v5 on-policy) — standalone R-generation subprocess.

Plan §4.3 + §4.5. Wraps
``explore_persona_space.experiments.contrastive_recipe_sweep_448.r_generate.
generate_r_artifacts`` for the dispatcher to invoke in a SEPARATE subprocess
(per CLAUDE.md vLLM-in-process-teardown gotcha — vLLM workers survive del +
destroy and re-grab GPU memory the next time another framework loads weights).

After successful generation, uploads R_train.json + R_eval.json to the HF
data repo at ``superkaiba1/explore-persona-space-data/
issue448_recipe_sweep_v5/on_policy_R/`` so Phase 3 (training) and Phase 4
(eval) can read from a frozen, hash-verified source on any pod.

Hard guards (fail-loud, no silent recovery):
  - Marker token id 83399 assertion at startup (assertion #25 in plan §12).
  - Phase-0 / Phase-1 EXIT assertion: ``set(EVAL_PERSONAS_24).issubset(
    R_eval.keys())`` (Must-Fix-1).
  - HF upload non-empty return (per ``feedback_eval_script_silent_not_present_misdiagnosis``).

CLI:
    uv run python scripts/i448_phase_r_generate.py
    uv run python scripts/i448_phase_r_generate.py --no-upload   # debug only
    uv run python scripts/i448_phase_r_generate.py --train-questions-limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SOURCE_PERSONA = "villain"  # plan §3 (single source for the sweep).

log = logging.getLogger("i448.phase_r_generate")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _upload_to_hf(local_path: Path, path_in_repo: str) -> str:
    """Upload one R artifact to the HF data repo; raise if upload reports empty path.

    Per CLAUDE.md memory ``feedback_eval_script_silent_not_present_misdiagnosis``
    — split the no-result branch into "real upload failure" rather than
    silently advancing.
    """
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.r_generate import (
        HF_DATA_REPO,
    )
    from explore_persona_space.orchestrate.hub import upload_dataset

    hub_path = upload_dataset(
        str(local_path),
        repo_id=HF_DATA_REPO,
        path_in_repo=path_in_repo,
    )
    if not hub_path:
        raise RuntimeError(
            f"upload_dataset({local_path}) returned empty path — HF upload "
            f"failed. Refusing to advance Phase 3 with an un-frozen R artifact. "
            f"Check HF_TOKEN scope + data-repo write permissions."
        )
    log.info("Uploaded %s -> %s", local_path.name, hub_path)
    return hub_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/issue_448/on_policy_R"),
        help="Local output directory for R_train.json + R_eval.json.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Per-q greedy generation cap. Default 1024 (plan §11).",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="vLLM engine max_model_len. Default 2048 (plan §11).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="vLLM + sampling seed. Default 42 (project convention).",
    )
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization. Default 0.85 (plan §10).",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip HF data-repo upload (debug only; downstream phases need the upload).",
    )
    ap.add_argument(
        "--train-questions-limit",
        type=int,
        default=None,
        help=(
            "Smoke / debug: cap the Q_train universe to the first K questions "
            "(default: full 850-pair union pool). EVAL_QUESTIONS is always 20."
        ),
    )
    ap.add_argument(
        "--sentinel-path",
        type=Path,
        default=None,
        help=(
            "Optional sentinel JSON path the parent dispatcher reads. Written "
            "with the poll_pipeline-required keys "
            "(sentinel_schema_version=1, kind='epm:progress', version=1)."
        ),
    )
    ap.add_argument(
        "--source",
        type=str,
        default=SOURCE_PERSONA,
        help=f"Source persona for the sweep. Default {SOURCE_PERSONA!r}.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=r_generate] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    log.info("Phase 1 (v5 on-policy R-generate) starting — source=%s", args.source)

    # ── Resolve persona registry + cell specs + eval personas. ───────────────
    # Marker assert (defense in depth — generate_r_artifacts re-asserts).
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        CELL_SPECS,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
        persona_registry,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_wrong_claim_pool import (  # noqa: E501
        load_union_pool,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.r_generate import (
        HF_PATH_PREFIX,
        generate_r_artifacts,
    )
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )
    from explore_persona_space.personas import EVAL_QUESTIONS

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise AssertionError(
            f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected "
            f"[{EXPECTED_MARKER_TOKEN_ID}]. Refusing to launch Phase 1."
        )

    # persona_registry must be initialized for select_n_bystanders.
    if not persona_registry.OBSERVED_BYSTANDERS_PER_SOURCE:
        log.info("persona_registry not yet built — building now")
        persona_registry._do_build_and_assert()

    # Q_train pool = the 850-pair generic-corpus questions.
    union_pool = load_union_pool()
    train_questions = sorted({entry["question"] for entry in union_pool})
    if args.train_questions_limit is not None:
        train_questions = train_questions[: args.train_questions_limit]
        log.info(
            "train_questions LIMITED to %d (smoke); full pool was %d",
            len(train_questions),
            len(union_pool),
        )

    log.info(
        "Inputs: |CELL_SPECS|=%d, |EVAL_PERSONAS_24|=%d, |EVAL_QUESTIONS|=%d, |train_questions|=%d",
        len(CELL_SPECS),
        len(EVAL_PERSONAS_24),
        len(EVAL_QUESTIONS),
        len(train_questions),
    )

    # ── Generate. ────────────────────────────────────────────────────────────
    summary = generate_r_artifacts(
        base_model=BASE_MODEL,
        source=args.source,
        cell_specs=CELL_SPECS,
        eval_personas=EVAL_PERSONAS_24,
        eval_questions=list(EVAL_QUESTIONS),
        train_questions=train_questions,
        out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # ── Phase-1 EXIT assertion (defense in depth). Already checked inside
    # generate_r_artifacts; re-assert here so a dispatcher reading just this
    # script's exit also has the guarantee.
    eval_payload = json.loads((args.out_dir / "R_eval.json").read_text())
    eval_panel_in_artifact = set(eval_payload["completions"].keys())
    missing = sorted(set(EVAL_PERSONAS_24.keys()) - eval_panel_in_artifact)
    if missing:
        raise RuntimeError(
            f"Phase-1 EXIT assertion FAILED: R_eval missing {len(missing)} panel "
            f"personas: {missing!r}. This should be unreachable — "
            f"generate_r_artifacts already checks. Investigate."
        )

    # ── Upload (atomic per-file). ────────────────────────────────────────────
    if not args.no_upload:
        _upload_to_hf(
            Path(summary["r_train_path"]),
            f"{HF_PATH_PREFIX}/R_train.json",
        )
        _upload_to_hf(
            Path(summary["r_eval_path"]),
            f"{HF_PATH_PREFIX}/R_eval.json",
        )
        summary["hf_uploaded"] = True
        summary["hf_data_repo"] = "superkaiba1/explore-persona-space-data"
        summary["hf_path_prefix"] = HF_PATH_PREFIX
    else:
        log.warning(
            "--no-upload set — R artifacts NOT uploaded to HF; downstream phases "
            "MUST share the same disk."
        )
        summary["hf_uploaded"] = False

    # ── Sentinel for the dispatcher's poll_pipeline.py. ──────────────────────
    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_payload = {
            "sentinel_schema_version": 1,
            "kind": "epm:progress",
            "version": 1,
            "task_id": 448,
            "phase": "r_generate",
            "by": "i448_phase_r_generate",
            "ts": datetime.now(UTC).isoformat(),
            "note": json.dumps(
                {
                    "phase": "r_generate",
                    "r_train_path": summary["r_train_path"],
                    "r_train_hash": summary["r_train_hash"],
                    "r_eval_path": summary["r_eval_path"],
                    "r_eval_hash": summary["r_eval_hash"],
                    "n_train_forwards": summary["n_train_forwards"],
                    "n_eval_forwards": summary["n_eval_forwards"],
                    "training_side_personas": summary["training_side_personas"],
                    "eval_only_personas": summary["eval_only_personas"],
                    "hf_uploaded": summary.get("hf_uploaded", False),
                    "git_commit": _git_sha(),
                }
            ),
        }
        args.sentinel_path.write_text(json.dumps(sentinel_payload, indent=2))
        log.info("Wrote phase sentinel → %s", args.sentinel_path)

    log.info(
        "Phase 1 done. n_train_forwards=%d, n_eval_forwards=%d, R_train sha[:12]=%s, "
        "R_eval sha[:12]=%s",
        summary["n_train_forwards"],
        summary["n_eval_forwards"],
        summary["r_train_hash"][:12],
        summary["r_eval_hash"][:12],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
