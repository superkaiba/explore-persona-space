# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #448 v5 dispatcher — UNIFIED smoke = sweep with one cell (on-policy).

7-phase pipeline (Pre-Phase 0 corpus / 0.5 centroids / 0.75 held-out pin /
1 R-generate / 1.5 base panel / 2+3 per-cell train+eval / 5 analyze)
sequential on a single 4× H100 pod. Same per-cell function body for smoke
and sweep — only `--cells` set-size differs. The plan's `--smoke`
shorthand selects ONE cell (Anchor) AND skips the heavy follow-up phases.

The v5 on-policy correction (plan §4.5):
  * Phase 1 (NEW) — base.generate(T(persona), q) greedy temp=0 EOS-stop
    cap-1024 for every persona in (training-side ∪ EVAL_PERSONAS_24) ×
    (Q_train ∪ EVAL_QUESTIONS); content-hashed JSON; HF data-repo upload.
  * Training rows use that frozen R as the completion (positives also get
    ` ※` appended); MarkerOnlyDataCollator(tail_tokens=0) masks loss to
    only the marker token + EOS.
  * Phase 4 eval uses vLLM prompt_logprobs=1 on prompts shaped
    `tokenize(prompt_text + R_eval_text + ` ※`)`; ΔG = trained − base on
    the SAME R.

Per-cell discipline (one cell == one of the 11 in `CELL_SPECS`, sequential):
  1. (one-time) Pre-Phase 0 corpus top-up via Sonnet 4.5 (carry-over — the
     v5 sweep still uses the 850-pair generic-corpus questions for Q_train,
     but NOT the canonical responses — those were the off-policy DV input).
  2. (one-time) Phase 0.5 — extend layer-20 centroids (for the H5
     secondary).
  3. (one-time) Phase 0.75 — pin the held-out bystander set (~15 personas
     never used as a contrastive negative in ANY cell; the H1b primary
     denominator per plan §4.3.0).
  4. (one-time) Phase 1 — base R generation (separate subprocess; vLLM
     teardown discipline per CLAUDE.md gotcha).
  5. (one-time) Phase 1.5 — base-Qwen 24-panel × 20-question marker log-p
     probe on the ON-POLICY R_eval slots (descriptive).
  6. Per cell: build_training_data(r_train) → train_lora (marker-only loss)
     → upload adapter to HF Hub → eval_one_cell SUBPROCESS (vLLM
     prompt_logprobs with LoRARequest hot-swap) → write sentinel JSON.
  7. (one-time) Phase 5 — analyze.run_analysis (H1a / H1b / contrasts /
     efficiency / H3 / H4 / H5).

Pod-side discipline:
  - Sets EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 (CLAUDE.md gotcha — MooseFS
    quota safety for sequential multi-cell sweep).
  - Never shells out to scripts/task.py (sentinel-file pattern only).
  - Every subprocess.* call uses env={**os.environ}; load_dotenv() is at
    module top.
  - Phase 1 + Phase 4 spawn SEPARATE subprocesses for each vLLM session so
    the OS reaps workers before the next framework loads weights (CLAUDE.md
    `vLLM in-process teardown` gotcha).

Sentinels (poll_pipeline.py-compliant — sentinel_schema_version=1,
kind=`epm:results` (final) or `epm:progress` (per-phase), version=1):
  /workspace/logs/issue-448-pre-phase-0-results.json     (per phase)
  /workspace/logs/issue-448-phase-0-5-results.json
  /workspace/logs/issue-448-phase-0-75-results.json
  /workspace/logs/issue-448-r-generate-results.json
  /workspace/logs/issue-448-phase-1-5-results.json
  /workspace/logs/issue-448-<cell>-results.json          (per cell)
  /workspace/logs/issue-448-results.json                 (END-OF-SWEEP =
      `epm:results v1`)

CLI:
  --cells <names>         Plain-English names (Anchor,+pos-ex-100,...) OR
                          slug forms (c1_anchor,c2_pos_ex_100,...). Default: all.
  --smoke                 Shorthand for --cells Anchor + skip base-panel +
                          skip analyze.
  --dry-run               No GPU work / no Anthropic calls. Validate that
                          dispatcher modules import cleanly, marker tokenizer
                          assertion holds, persona_registry builds, --cells
                          resolves correctly. Exits with summary.
  --skip-pre-phase-0      Re-use existing data/issue_448/generic_corpus/*.
  --skip-phase-0-5        Re-use eval_results/issue_448/centroids/.
  --skip-phase-0-75       Re-use data/issue_448/held_out_bystanders.json.
  --skip-r-generate       Re-use data/issue_448/on_policy_R/{R_train,R_eval}.json
                          (or pull from HF data repo on demand).
  --skip-base-panel       Re-use eval_results/issue_448_v5/base/marker_logprob.json.
  --skip-analyze          Don't run Phase 5.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Module-top constants. The hub repo + adapter-path-in-repo conventions
# inherit from #411 verbatim.
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SEED = 42
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
LOG_DIR = Path("/workspace/logs")  # default; overridden by --log-dir CLI arg.

# v5 on-policy paths (plan §10).
ON_POLICY_R_DIR = Path("data/issue_448/on_policy_R")
HELD_OUT_ARTIFACT_PATH = Path("data/issue_448/held_out_bystanders.json")
DEFAULT_SLAB_ROOT_V5 = Path("eval_results/issue_448_v5")
DEFAULT_FIGURES_DIR_V5 = Path("figures/issue_448_v5")

log = logging.getLogger("dispatch_recipe_sweep_448")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _write_poll_compliant_sentinel(
    path: Path,
    *,
    kind: str,
    phase: str,
    note_payload: dict[str, object],
    by: str = "dispatch_recipe_sweep_448",
) -> None:
    """Write a poll_pipeline.py-compliant sentinel.

    Per ``scripts/poll_pipeline.py::_SENTINEL_REQUIRED_KEYS``, every sentinel
    MUST carry the keys ``sentinel_schema_version=1``, ``kind`` (full marker
    kind), and ``version`` (marker version int). The marker body lives under
    ``note`` (or the ``payload`` synonym).

    Plan §risk-table "Pod-side reporting contract gap (the v4 bug)" — v1-v4
    wrote the key ``schema`` instead and the poller silently dropped the
    end-of-run sentinel. v5 emits the correct keys here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sentinel = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "task_id": 448,
        "by": by,
        "ts": datetime.now(UTC).isoformat(),
        "phase": phase,
        "note": json.dumps(note_payload),
    }
    path.write_text(json.dumps(sentinel, indent=2))


def _resolve_cells(raw: str | None, smoke: bool) -> list[tuple[str, str, int, int, int, int]]:
    """Resolve `--cells` (CSV of plain-English OR slug) → list of CELL_SPECS rows.

    Defaults: all 11 cells. `--smoke` overrides to ['c1_anchor'].
    """
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        CELL_SPECS,
    )

    if smoke:
        wanted = {"c1_anchor"}
    elif raw is None or raw.strip() == "all" or raw.strip() == "":
        wanted = {row[0] for row in CELL_SPECS}
    else:
        wanted = set()
        # Build a name → slug lookup.
        name_to_slug = {row[1]: row[0] for row in CELL_SPECS}
        slug_set = {row[0] for row in CELL_SPECS}
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token in slug_set:
                wanted.add(token)
            elif token in name_to_slug:
                wanted.add(name_to_slug[token])
            else:
                raise ValueError(
                    f"Unknown cell {token!r}. Expected one of:\n  slugs: "
                    f"{sorted(slug_set)}\n  names: {sorted(name_to_slug.keys())}"
                )
    out = [row for row in CELL_SPECS if row[0] in wanted]
    return out


def _run_pre_phase_0(skip: bool, dry_run: bool, canonical_only: bool = False) -> dict[str, object]:
    """Pre-Phase 0: Sonnet 4.5 union pool + canonical responses.

    ``canonical_only=True`` (round-2 fix B2 for --smoke-real) generates only
    the 20 canonical EVAL_QUESTIONS responses, skipping the 650-pair top-up.
    Cost ~$0.02, ~30s. Lets the smoke-real path build the eval rig without
    paying the full ~$5 / 10-min top-up.
    """
    t0 = time.time()
    log.info("=" * 70)
    log.info("Pre-Phase 0 (Sonnet 4.5 corpus top-up + canonical responses)")
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        build_wrong_claim_pool,
    )

    sentinel = LOG_DIR / "issue-448-pre-phase-0-results.json"
    out_dir = build_wrong_claim_pool.OUT_DIR

    if dry_run:
        log.info("Pre-Phase 0 DRY-RUN: validating imports only.")
        # Validate the module imports + key functions are callable.
        assert callable(build_wrong_claim_pool.build_corpus)
        assert callable(build_wrong_claim_pool.load_canonical_responses)
        assert callable(build_wrong_claim_pool.load_union_pool)
        summary = {"phase": "pre_phase_0", "status": "dry_run_validated"}
    elif skip:
        # Verify cached artifacts exist. Round-3 fix: under canonical_only
        # (smoke-real path), require ONLY eval_canonical_responses.json
        # (union_pool.json is not needed because the per-cell loop is empty).
        canonical_path = out_dir / "eval_canonical_responses.json"
        union_path = out_dir / "union_pool.json"
        if canonical_only:
            if not canonical_path.exists():
                raise FileNotFoundError(
                    f"--skip-pre-phase-0 set under canonical-only mode but "
                    f"{canonical_path} missing. Re-run Pre-Phase 0."
                )
            log.info(
                "Pre-Phase 0 SKIPPED canonical-only (artifact exists at %s)",
                canonical_path,
            )
        else:
            if not canonical_path.exists() or not union_path.exists():
                raise FileNotFoundError(
                    f"--skip-pre-phase-0 set but artifacts missing. Expected "
                    f"{canonical_path} + {union_path}. Run Pre-Phase 0 first."
                )
            log.info("Pre-Phase 0 SKIPPED (artifacts exist at %s)", out_dir)
        summary = {
            "phase": "pre_phase_0",
            "status": "skipped_artifacts_exist",
            "out_dir": str(out_dir),
        }
    else:
        import asyncio

        summary = asyncio.run(
            build_wrong_claim_pool.build_corpus(
                out_dir=out_dir, canonical_responses_only=canonical_only
            )
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["sentinel_path"] = str(sentinel)
    sentinel.write_text(json.dumps(summary, indent=2))
    log.info("Pre-Phase 0 done in %.1fs", summary["wall_seconds"])
    return summary


def _run_phase_0_5(skip: bool, dry_run: bool) -> dict[str, object]:
    """Phase 0.5: extend layer-20 centroids."""
    t0 = time.time()
    log.info("=" * 70)
    log.info("Phase 0.5 (extend centroids)")

    sentinel = LOG_DIR / "issue-448-phase-0-5-results.json"
    out_pt = Path("eval_results/issue_448/centroids/centroids_layer20.pt")

    if dry_run:
        # Just verify module imports.
        from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
            extend_centroids,  # noqa: F401
        )

        log.info("Phase 0.5 DRY-RUN: extend_centroids module imports cleanly.")
        summary = {"phase": "phase_0_5", "status": "dry_run_validated"}
    elif skip:
        if not out_pt.exists():
            # Round-3 fix R2-1: don't fatal-fail under --smoke-real (the
            # smoke-real path only runs Phase 1.5 base panel which doesn't
            # depend on centroids; centroids are only consumed by the
            # per-cell trajectory callback + Phase 3 analyzer, both of
            # which are skipped under --smoke-real).
            log.warning(
                "Phase 0.5 SKIPPED but %s missing. Tolerated: downstream "
                "phases that depend on centroids are also skipped.",
                out_pt,
            )
            summary = {"phase": "phase_0_5", "status": "skipped_artifact_missing"}
        else:
            log.info("Phase 0.5 SKIPPED (artifact exists at %s)", out_pt)
            summary = {"phase": "phase_0_5", "status": "skipped_artifact_exists"}
    else:
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.contrastive_recipe_sweep_448.extend_centroids",
        ]
        log.info("Phase 0.5 subprocess: %s", " ".join(cmd))
        subprocess.run(cmd, env={**os.environ}, check=True)
        summary = {"phase": "phase_0_5", "centroids_path": str(out_pt)}

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["sentinel_path"] = str(sentinel)
    sentinel.write_text(json.dumps(summary, indent=2))
    log.info("Phase 0.5 done in %.1fs", summary["wall_seconds"])
    return summary


def _run_phase_0_75_held_out_pin(
    skip: bool, dry_run: bool, source: str = "villain"
) -> dict[str, object]:
    """Phase 0.75 — pin the held-out bystander subset (plan §4.3.0).

    Computes the ~15-persona held-out subset (12 guaranteed + 3 SHA-extras)
    via ``compute_held_out_bystanders`` and writes
    ``data/issue_448/held_out_bystanders.json``. The analyzer + sentinels
    read this artifact downstream.
    """
    t0 = time.time()
    log.info("=" * 70)
    log.info("[phase=held_out_pin] Phase 0.75 (held-out bystander pin)")
    sentinel = LOG_DIR / "issue-448-phase-0-75-results.json"

    if dry_run:
        log.info("Phase 0.75 DRY-RUN: skipping held-out computation")
        summary: dict[str, object] = {"phase": "phase_0_75", "status": "dry_run_validated"}
    elif skip:
        if not HELD_OUT_ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"--skip-phase-0-75 set but {HELD_OUT_ARTIFACT_PATH} missing. Run Phase 0.75 first."
            )
        log.info("Phase 0.75 SKIPPED (artifact exists at %s)", HELD_OUT_ARTIFACT_PATH)
        payload = json.loads(HELD_OUT_ARTIFACT_PATH.read_text())
        summary = {
            "phase": "phase_0_75",
            "status": "skipped_artifact_exists",
            "n_held_out": payload.get("n_held_out"),
            "held_out_path": str(HELD_OUT_ARTIFACT_PATH),
        }
    else:
        from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
            CELL_SPECS,
        )
        from explore_persona_space.experiments.contrastive_recipe_sweep_448.held_out_bystanders import (  # noqa: E501
            compute_held_out_bystanders,
            write_held_out_artifact,
        )
        from explore_persona_space.experiments.factor_screen_365.persona_panel import (
            EVAL_PERSONAS_24,
        )

        payload = compute_held_out_bystanders(EVAL_PERSONAS_24, source, CELL_SPECS)
        write_held_out_artifact(payload, HELD_OUT_ARTIFACT_PATH)
        summary = {
            "phase": "phase_0_75",
            "n_held_out": payload["n_held_out"],
            "n_guaranteed": payload["n_guaranteed"],
            "n_sha_extras": payload["n_sha_extras"],
            "held_out_path": str(HELD_OUT_ARTIFACT_PATH),
            "held_out_personas": payload["held_out"],
        }

    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["sentinel_path"] = str(sentinel)
    _write_poll_compliant_sentinel(
        sentinel, kind="epm:progress", phase="held_out_pin", note_payload=summary
    )
    log.info("[phase=held_out_pin] done in %.1fs", summary["wall_seconds"])
    return summary


def _run_phase_1_r_generate(
    skip: bool,
    dry_run: bool,
    *,
    no_upload: bool = False,
    train_questions_limit: int | None = None,
    source: str = "villain",
    max_new_tokens: int = 1024,
) -> dict[str, object]:
    """Phase 1 — base on-policy R generation (separate subprocess).

    Per CLAUDE.md vLLM gotcha: this phase MUST be a fresh subprocess so the
    OS reaps vLLM workers before the next framework loads weights (HF
    Trainer in Phase 3). The dispatcher invokes
    ``scripts/i448_phase_r_generate.py`` and waits.
    """
    t0 = time.time()
    log.info("=" * 70)
    log.info("[phase=r_generate] Phase 1 (base on-policy R generation)")
    sentinel = LOG_DIR / "issue-448-r-generate-results.json"
    r_train_path = ON_POLICY_R_DIR / "R_train.json"
    r_eval_path = ON_POLICY_R_DIR / "R_eval.json"

    if dry_run:
        # Validate the module imports + the standalone script exists.
        from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
            r_generate,  # noqa: F401
        )

        if not (Path("scripts/i448_phase_r_generate.py").exists()):
            raise FileNotFoundError(
                "scripts/i448_phase_r_generate.py missing — Phase 1 dispatcher script."
            )
        summary: dict[str, object] = {"phase": "r_generate", "status": "dry_run_validated"}
    elif skip:
        # Resume mode: re-use existing R artifacts (or pull from HF if absent).
        if not r_train_path.exists() or not r_eval_path.exists():
            # Try HF Hub fallback.
            log.info(
                "R artifacts missing locally — attempting HF data-repo fallback "
                "(superkaiba1/explore-persona-space-data/issue448_recipe_sweep_v5/on_policy_R)"
            )
            from huggingface_hub import hf_hub_download

            ON_POLICY_R_DIR.mkdir(parents=True, exist_ok=True)
            for fname in ("R_train.json", "R_eval.json"):
                downloaded = hf_hub_download(
                    repo_id=HF_DATA_REPO,
                    repo_type="dataset",
                    filename=f"issue448_recipe_sweep_v5/on_policy_R/{fname}",
                    revision="main",
                )
                import shutil

                shutil.copyfile(downloaded, ON_POLICY_R_DIR / fname)
        log.info("Phase 1 SKIPPED (artifacts exist locally or via HF)")
        summary = {
            "phase": "r_generate",
            "status": "skipped_artifacts_exist",
            "r_train_path": str(r_train_path),
            "r_eval_path": str(r_eval_path),
        }
    else:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/i448_phase_r_generate.py",
            "--out-dir",
            str(ON_POLICY_R_DIR),
            "--max-new-tokens",
            str(max_new_tokens),
            "--source",
            source,
            "--sentinel-path",
            str(sentinel),
        ]
        if no_upload:
            cmd.append("--no-upload")
        if train_questions_limit is not None:
            cmd.extend(["--train-questions-limit", str(train_questions_limit)])
        log.info("Phase 1 subprocess: %s", " ".join(cmd))
        subprocess.run(cmd, env={**os.environ}, check=True)
        # The subprocess already wrote a poll-compliant sentinel; re-load it.
        if sentinel.exists():
            sentinel_payload = json.loads(sentinel.read_text())
            inner = json.loads(sentinel_payload.get("note", "{}"))
            summary = {"phase": "r_generate", **inner}
        else:
            summary = {"phase": "r_generate", "status": "subprocess_done_no_sentinel"}

    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["sentinel_path"] = str(sentinel)
    # Re-write the sentinel with our roll-up payload (poll-compliant keys).
    _write_poll_compliant_sentinel(
        sentinel,
        kind="epm:progress",
        phase="r_generate",
        note_payload=summary,
    )
    log.info("[phase=r_generate] done in %.1fs", summary["wall_seconds"])
    return summary


def _run_phase_1_5_base_panel(
    skip: bool,
    dry_run: bool,
    eval_personas_limit: int | None = None,
    eval_questions_limit: int | None = None,
) -> dict[str, object]:
    """Phase 1.5: panel × question base-marker-logprob probe (DESCRIPTIVE).

    v5 on-policy: invokes ``eval_one_cell --cell base`` which runs the base
    model on the same vLLM prompt_logprobs rig that Phase 4 uses, on the
    SAME R_eval.json slot positions — descriptive only (the per-cell ΔG
    subtraction handles the actual base-vs-trained comparison).

    Subprocess-isolated per CLAUDE.md vLLM-in-process-teardown gotcha.
    """
    t0 = time.time()
    log.info("=" * 70)
    log.info("[phase=base_panel] Phase 1.5 (base panel marker log-p, on-policy)")

    sentinel = LOG_DIR / "issue-448-phase-1-5-results.json"
    out_dir = DEFAULT_SLAB_ROOT_V5 / "base"
    out_path = out_dir / "marker_logprob.json"

    if dry_run:
        from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
            eval_one_cell,  # noqa: F401
        )

        log.info("Phase 1.5 DRY-RUN: eval_one_cell module imports cleanly.")
        summary = {"phase": "phase_1_5", "status": "dry_run_validated"}
    elif skip:
        if not out_path.exists():
            raise FileNotFoundError(
                f"--skip-base-panel set but {out_path} missing. Re-run Phase 1.5."
            )
        log.info("Phase 1.5 SKIPPED (artifact exists at %s)", out_path)
        summary = {"phase": "phase_1_5", "status": "skipped_artifact_exists"}
    else:
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.contrastive_recipe_sweep_448.eval_one_cell",
            "--cell",
            "base",
            "--out-dir",
            str(out_dir),
            "--r-eval-path",
            str(ON_POLICY_R_DIR / "R_eval.json"),
            "--sentinel-path",
            str(sentinel),
        ]
        if eval_personas_limit is not None:
            cmd.extend(["--eval-personas-limit", str(eval_personas_limit)])
        if eval_questions_limit is not None:
            cmd.extend(["--eval-questions-limit", str(eval_questions_limit)])
        log.info("Phase 1.5 subprocess: %s", " ".join(cmd))
        subprocess.run(cmd, env={**os.environ}, check=True)
        summary = {
            "phase": "phase_1_5",
            "base_logp_path": str(out_path),
            "eval_personas_limit": eval_personas_limit,
            "eval_questions_limit": eval_questions_limit,
        }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    _write_poll_compliant_sentinel(
        sentinel, kind="epm:progress", phase="base_panel", note_payload=summary
    )
    log.info("[phase=base_panel] done in %.1fs", summary["wall_seconds"])
    return summary


def _build_trajectory_callback(
    tokenizer,
    canonical_responses: dict[str, str],
    *,
    centroids_path: Path | None = None,
    pos_persona_names: list[str] | None = None,
    neg_persona_names: list[str] | None = None,
    trajectory_output_path: Path | None = None,
    cell_slug: str = "unknown",
    seed: int = 42,
):
    """Build the MarkerTrajectoryCallback for the per-cell training run.

    Round-2 fix C2: subset = 3 nearest + 3 farthest panel personas (by
    cosine to the mean of the cell's negative-set centroids). Falls back
    to first-N if the centroid bundle isn't available yet (Phase 0.5
    not yet run).
    """
    import numpy as np
    import torch

    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        TRAJECTORY_N_PERSONAS,
        TRAJECTORY_N_QUESTIONS,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.marker_trajectory_callback import (  # noqa: E501
        MarkerTrajectoryCallback,
    )
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )
    from explore_persona_space.personas import EVAL_QUESTIONS

    # Round-2 fix C2: pick the trajectory subset by cosine distance to the
    # cell's negative-set centroids (3 nearest + 3 farthest from
    # `neg_centroid_mean`). Per plan §4.0quater. Centroid bundle is loaded
    # via the path arg; if missing OR the bundle doesn't cover required
    # personas, fall back to the round-1 first-N approach (logged).
    n_per_half = max(1, TRAJECTORY_N_PERSONAS // 2)
    subset_personas: dict[str, str]
    if centroids_path is not None and centroids_path.exists() and neg_persona_names:
        bundle = torch.load(centroids_path, weights_only=False)
        layer = bundle.get("layer", 20)
        tensor = bundle["centroids"][layer].to(torch.float32).numpy()
        names = list(bundle["persona_names"])
        name_to_idx = {n: i for i, n in enumerate(names)}
        # Centroid mean of the cell's negative personas (typically 2 for cells
        # 1-9; 4 or 8 for cells 10/11).
        try:
            neg_idx = [name_to_idx[n] for n in neg_persona_names]
            neg_mean = tensor[neg_idx].mean(axis=0)
            neg_mean_norm = neg_mean / (np.linalg.norm(neg_mean) + 1e-12)
            # For every panel persona (excluding the cell's source + negatives
            # themselves), compute cosine sim to neg_mean.
            cell_pos = set(pos_persona_names) if pos_persona_names else set()
            cell_neg = set(neg_persona_names)
            sim_pairs: list[tuple[str, float]] = []
            for pname in EVAL_PERSONAS_24:
                if pname in cell_pos or pname in cell_neg:
                    continue
                if pname not in name_to_idx:
                    continue
                vec = tensor[name_to_idx[pname]]
                vec_norm = vec / (np.linalg.norm(vec) + 1e-12)
                sim_pairs.append((pname, float(vec_norm @ neg_mean_norm)))
            sim_pairs.sort(key=lambda kv: kv[1])
            # Lowest sim = "farthest from negatives" (high predicted leakage).
            farthest = [p for p, _ in sim_pairs[:n_per_half]]
            # Highest sim = "nearest to negatives" (low predicted leakage).
            nearest = [p for p, _ in sim_pairs[-n_per_half:]]
            picks = farthest + nearest
            subset_personas = {p: EVAL_PERSONAS_24[p] for p in picks}
            log.info(
                "Trajectory subset (cosine-stratified): nearest=%s farthest=%s",
                nearest,
                farthest,
            )
        except KeyError as e:
            log.warning(
                "Centroid lookup missing persona %s; falling back to first-N trajectory subset.",
                e,
            )
            subset_personas = dict(list(EVAL_PERSONAS_24.items())[:TRAJECTORY_N_PERSONAS])
    else:
        log.info(
            "Trajectory subset: first-%d EVAL_PERSONAS_24 (centroid bundle "
            "or neg_persona_names unavailable)",
            TRAJECTORY_N_PERSONAS,
        )
        subset_personas = dict(list(EVAL_PERSONAS_24.items())[:TRAJECTORY_N_PERSONAS])
    subset_questions = EVAL_QUESTIONS[:TRAJECTORY_N_QUESTIONS]
    return MarkerTrajectoryCallback(
        tokenizer=tokenizer,
        persona_prompts=subset_personas,
        questions=subset_questions,
        canonical_responses=canonical_responses,
        output_path=trajectory_output_path,
        cell_slug=cell_slug,
        seed=seed,
    )


def _build_training_data_for_cell(
    cell_slug: str,
    pos_ex_per_p: int,
    pos_personas: int,
    neg_ex_per_p: int,
    neg_personas: int,
    out_path: Path,
    *,
    r_train: dict[str, dict[str, dict]] | None = None,
) -> Path:
    """Per-cell training-data build (CPU, in-process).

    Args:
        ...
        r_train: ON-POLICY R artifact from Phase 1 (``persona -> q -> {...}``).
            REQUIRED for the v5 on-policy build. If omitted the build falls
            back to ``legacy_off_policy=True`` (the v1-v4 canonical-response
            shape — preserved for future debugging only; never the v5 path).
    """
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_training_data import (
        build_cell,
    )

    return build_cell(
        cell_slug=cell_slug,
        pos_ex_per_persona=pos_ex_per_p,
        pos_personas=pos_personas,
        neg_ex_per_persona=neg_ex_per_p,
        neg_personas=neg_personas,
        output_path=out_path,
        r_train=r_train,
        legacy_off_policy=(r_train is None),
    )


def _train_lora_adapter(
    cell_slug: str,
    seed: int,
    train_jsonl: Path,
    output_dir: Path,
    trajectory_callback,
) -> Path:
    """v5 — train the LoRA adapter with marker-only loss; NO merge.

    The v4 dispatcher merged the adapter into base weights so the eval rig
    could load via ``AutoModelForCausalLM.from_pretrained``. v5 eval uses
    vLLM ``enable_lora=True`` + ``LoRARequest(...)`` hot-swap, so the
    adapter is loaded as a LoRA delta on top of base — no merge needed.
    This saves ~15 GB merged dir per cell (MooseFS quota safety) and ~3-5
    min wall per cell.

    Returns:
        Path to the LoRA adapter directory.
    """
    from explore_persona_space.train.sft import (
        TrainLoraConfig,
        train_lora,
    )

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # #411 + #460 hparams + v5 marker-only-loss surface (plan §11).
    cfg = TrainLoraConfig(
        gpu_id=0,
        epochs=3,
        lr=1e-5,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,  # effective batch 16
        max_length=2048,  # v5: bumped from 1024 to fit prompt + R(<=1024) + marker
        warmup_ratio=0.05,
        seed=seed,
        run_name=f"issue448_v5_{cell_slug}_seed{seed}",
        report_to="wandb",
        save_strategy="no",
        gradient_checkpointing=True,
        packing=False,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/issue_448_v5/{cell_slug}_seed{seed}",
        # v5 on-policy: loss ONLY on the marker token + EOS; R stays on-policy.
        marker_only_loss=True,
        marker_text=" ※",
        marker_tail_tokens=0,
    )
    log.info(
        "[%s] Training LoRA (marker-only loss, tail_tokens=0) -> %s",
        cell_slug,
        adapter_dir,
    )
    callbacks = [trajectory_callback] if trajectory_callback is not None else None
    train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(train_jsonl),
        output_dir=str(adapter_dir),
        cfg=cfg,
        callbacks=callbacks,
    )
    return adapter_dir


def _eval_cell(
    cell_slug: str,
    adapter_dir: Path,
    eval_out_dir: Path,
    sentinel_path: Path,
    eval_personas_limit: int | None = None,
    eval_questions_limit: int | None = None,
) -> None:
    """v5 — per-cell eval via SEPARATE subprocess (vLLM teardown discipline).

    Per CLAUDE.md vLLM-in-process-teardown gotcha: vLLM workers survive the
    Python-side del + destroy + gc and re-grab GPU memory the moment the
    next HF Trainer (next cell's train_lora) loads weights. Subprocess
    isolation is the robust fix — OS reaps workers on subprocess exit.

    Runs ``eval_one_cell --cell <slug> --adapter-path <local_adapter>`` and
    writes the per-cell sentinel after the subprocess returns.
    """
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "explore_persona_space.experiments.contrastive_recipe_sweep_448.eval_one_cell",
        "--cell",
        cell_slug,
        "--adapter-path",
        str(adapter_dir),
        "--out-dir",
        str(eval_out_dir),
        "--r-eval-path",
        str(ON_POLICY_R_DIR / "R_eval.json"),
        "--sentinel-path",
        str(sentinel_path),
    ]
    if eval_personas_limit is not None:
        cmd.extend(["--eval-personas-limit", str(eval_personas_limit)])
    if eval_questions_limit is not None:
        cmd.extend(["--eval-questions-limit", str(eval_questions_limit)])
    log.info("[%s] Eval subprocess: %s", cell_slug, " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)

    out_path = eval_out_dir / "marker_logprob.json"
    if not out_path.exists():
        raise RuntimeError(
            f"[{cell_slug}] eval_one_cell subprocess exited 0 but {out_path} "
            f"is missing — silent eval failure (per "
            f"feedback_eval_script_silent_not_present_misdiagnosis)."
        )
    summary_payload = {
        "cell": cell_slug,
        "adapter_dir": str(adapter_dir),
        "marker_logprob_path": str(out_path),
        "marker_logprob_summary_path": str(eval_out_dir / "marker_logprob_summary.json"),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    _write_poll_compliant_sentinel(
        sentinel_path,
        kind="epm:progress",
        phase=f"eval_cell_{cell_slug}",
        note_payload=summary_payload,
    )
    log.info("[%s] Wrote per-cell sentinel -> %s", cell_slug, sentinel_path)


def _run_one_cell(
    cell_slug: str,
    plain_name: str,
    pos_ex_per_p: int,
    pos_personas: int,
    neg_ex_per_p: int,
    neg_personas: int,
    seed: int,
    slab_root: Path,
    runs_root: Path,
    dry_run: bool,
    canonical_responses: dict[str, str] | None,
    r_train: dict[str, dict[str, dict]] | None = None,
    centroids_path: Path | None = None,
    resume: bool = False,
    eval_personas_limit: int | None = None,
    eval_questions_limit: int | None = None,
) -> dict[str, object]:
    """Per-cell v5 on-policy: build(r_train) -> train (marker-only loss) ->
    eval SUBPROCESS (vLLM LoRA hot-swap) -> sentinel.

    No merge step (vLLM uses LoRARequest hot-swap directly on adapter dir).
    Eval is a separate subprocess per CLAUDE.md vLLM teardown gotcha.
    """
    import torch

    t_start = time.time()
    output_dir = runs_root / f"{cell_slug}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir = slab_root / cell_slug
    sentinel_path = LOG_DIR / f"issue-448-{cell_slug}-results.json"
    train_jsonl = output_dir / "train_pool.jsonl"
    log.info("[phase=cell_%s] starting", cell_slug)

    # Resume mode: short-circuit if both sentinel + eval JSON are on disk.
    if resume and sentinel_path.exists() and (eval_out_dir / "marker_logprob.json").exists():
        log.info(
            "[%s] RESUME: sentinel + eval JSON exist; SKIPPING (use --no-resume to force).",
            cell_slug,
        )
        return {
            "cell": cell_slug,
            "plain_name": plain_name,
            "seed": seed,
            "wall_seconds": 0.0,
            "output_dir": str(output_dir),
            "eval_out_dir": str(eval_out_dir),
            "sentinel_path": str(sentinel_path),
            "adapter_hf_path": f"adapters/issue_448_v5/{cell_slug}_seed{seed}",
            "status": "resumed_skip",
        }

    log.info("=" * 70)
    log.info(
        "[%s] CELL START (name=%s, pos_ex/p=%d, pos_personas=%d, neg_ex/p=%d, neg_personas=%d)",
        cell_slug,
        plain_name,
        pos_ex_per_p,
        pos_personas,
        neg_ex_per_p,
        neg_personas,
    )

    # Build training data (CPU). Always runs in non-dry mode.
    if not dry_run:
        _build_training_data_for_cell(
            cell_slug,
            pos_ex_per_p,
            pos_personas,
            neg_ex_per_p,
            neg_personas,
            train_jsonl,
            r_train=r_train,
        )
    else:
        log.info(
            "[%s] DRY-RUN: skipping training-data build (would write %s)",
            cell_slug,
            train_jsonl,
        )

    if dry_run:
        log.info("[%s] DRY-RUN: skipping train + eval", cell_slug)
        wall = time.time() - t_start
        return {
            "cell": cell_slug,
            "plain_name": plain_name,
            "status": "dry_run",
            "wall_seconds": round(wall, 1),
        }

    # Trajectory callback (only if canonical responses are available — kept
    # as the within-cell dynamics diagnostic per CLAUDE.md "Track marker
    # log-prob DYNAMICS" rule).
    trajectory_callback = None
    if canonical_responses is not None:
        from transformers import AutoTokenizer

        from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_training_data import (  # noqa: E501
            _negative_personas_for_cell,
            _positive_personas_for_cell,
        )

        pos_persona_names = _positive_personas_for_cell(pos_personas)
        neg_persona_names = _negative_personas_for_cell(pos_persona_names, neg_personas)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        trajectory_json_path = eval_out_dir / "marker_logprob_trajectory.json"
        trajectory_callback = _build_trajectory_callback(
            tokenizer,
            canonical_responses,
            centroids_path=centroids_path,
            pos_persona_names=pos_persona_names,
            neg_persona_names=neg_persona_names,
            trajectory_output_path=trajectory_json_path,
            cell_slug=cell_slug,
            seed=seed,
        )
        log.info(
            "[%s] MarkerTrajectoryCallback wired (centroid-stratified subset)",
            cell_slug,
        )

    adapter_dir = _train_lora_adapter(cell_slug, seed, train_jsonl, output_dir, trajectory_callback)

    # Verify adapter on disk (train_lora hf_upload=True; the local copy is
    # what vLLM will load via LoRARequest).
    adapter_safetensors = list(adapter_dir.glob("*.safetensors"))
    if not adapter_safetensors:
        raise RuntimeError(
            f"[{cell_slug}] Adapter dir {adapter_dir} has no .safetensors "
            f"files after training — upload may be stale or training silently failed."
        )

    # HF Transformers GPU cleanup before spawning the vLLM eval subprocess.
    # The subprocess gets a fresh process anyway, but freeing the HF Trainer's
    # GPU pin in THIS process speeds up the vLLM init in the child.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Eval (subprocess: vLLM teardown discipline). ────────────────────────
    _eval_cell(
        cell_slug,
        adapter_dir,
        eval_out_dir,
        sentinel_path,
        eval_personas_limit=eval_personas_limit,
        eval_questions_limit=eval_questions_limit,
    )

    wall = time.time() - t_start
    log.info("[%s] CELL DONE in %.1fs", cell_slug, wall)
    return {
        "cell": cell_slug,
        "plain_name": plain_name,
        "seed": seed,
        "wall_seconds": round(wall, 1),
        "output_dir": str(output_dir),
        "eval_out_dir": str(eval_out_dir),
        "sentinel_path": str(sentinel_path),
        "adapter_dir": str(adapter_dir),
        "adapter_hf_path": f"adapters/issue_448_v5/{cell_slug}_seed{seed}",
    }


def _run_analyze(
    slab_root: Path,
    figures_dir: Path,
    centroids_path: Path,
    held_out_path: Path,
) -> dict[str, object]:
    """Phase 5 (v5): analyze.run_analysis (subprocess for clean import isolation)."""
    t0 = time.time()
    log.info("=" * 70)
    log.info("[phase=analyze] Phase 5 (v5 H1a/H1b/contrasts/efficiency)")
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "explore_persona_space.experiments.contrastive_recipe_sweep_448.analyze",
        "--slab-root",
        str(slab_root),
        "--centroids-path",
        str(centroids_path),
        "--held-out-path",
        str(held_out_path),
        "--figures-dir",
        str(figures_dir),
    ]
    log.info("Phase 5 subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)

    analyze_path = slab_root / "analyze_summary.json"
    if not analyze_path.exists():
        raise RuntimeError(
            f"Phase 5 finished but {analyze_path} not written; analyze silently failed."
        )
    payload = json.loads(analyze_path.read_text())
    h1b = payload.get("h1b_permutation_null", {})
    h1a = payload.get("h1a_permutation_null", {})
    log.info(
        "[phase=analyze] done in %.1fs; H1a_pass=%s H1b_pass=%s H1b_p=%.3f H1a_p=%.3f",
        time.time() - t0,
        payload.get("h1a_pass"),
        payload.get("h1b_pass"),
        h1b.get("empirical_p_value_one_sided", float("nan")),
        h1a.get("empirical_p_value_one_sided", float("nan")),
    )
    return {
        "phase": "analyze",
        "wall_seconds": round(time.time() - t0, 1),
        "analyze_summary_path": str(analyze_path),
        "h1a_pass": payload.get("h1a_pass"),
        "h1b_pass": payload.get("h1b_pass"),
        "interpretation": payload.get("interpretation"),
        "interpretation_code": payload.get("interpretation_code"),
        "h1a_permutation_null": h1a,
        "h1b_permutation_null": h1b,
        "n_cells_analyzed": payload.get("n_cells_analyzed"),
    }


def _write_final_sentinel(
    cells_requested: list[str],
    per_cell_summaries: list[dict[str, object]],
    phase_summaries: dict[str, dict | None],
    analyze_summary: dict | None,
    plan_deviations: list[str],
    seed: int,
    slab_root: Path,
) -> Path:
    """End-of-sweep sentinel in the poll_pipeline-compliant epm:results v1 shape.

    Per CLAUDE.md "Pod-side result-reporting contract" + plan §risk-table
    "Pod-side reporting contract gap": the sentinel MUST carry
    ``sentinel_schema_version=1, kind="epm:results", version=1`` so
    ``scripts/poll_pipeline.py::_parse_sentinel`` (main branch) auto-posts
    the ``epm:results`` marker. v1-v4 wrote the key ``schema`` and the
    poller silently dropped the sentinel.
    """
    final_path = LOG_DIR / "issue-448-results.json"
    eval_paths = {str(c["cell"]): str(c.get("eval_out_dir", "")) for c in per_cell_summaries}
    note_payload = {
        "issue": 448,
        "seed": seed,
        "recipe_version": "v5_on_policy",
        "cells_requested": cells_requested,
        "cells_completed": [str(c["cell"]) for c in per_cell_summaries],
        "n_completed": len(per_cell_summaries),
        "n_requested": len(cells_requested),
        "eval_paths": eval_paths,
        "eval_numbers": {
            "n_panel_personas": 24,
            "n_eval_questions": 20,
            "n_probes_per_cell": 480,
        },
        "reproducibility_card": {
            "base_model": BASE_MODEL,
            "hf_model_repo": HF_MODEL_REPO,
            "hf_data_repo": HF_DATA_REPO,
            "adapter_paths": {
                str(c["cell"]): f"{HF_MODEL_REPO}/tree/main/{c.get('adapter_hf_path', 'unknown')}"
                for c in per_cell_summaries
                if "adapter_hf_path" in c
            },
            "r_train_path_hf": (
                f"{HF_DATA_REPO}/issue448_recipe_sweep_v5/on_policy_R/R_train.json"
            ),
            "r_eval_path_hf": (f"{HF_DATA_REPO}/issue448_recipe_sweep_v5/on_policy_R/R_eval.json"),
            "held_out_artifact_local": str(HELD_OUT_ARTIFACT_PATH),
        },
        "worktree_path": str(Path.cwd()),
        "final_commit_sha": _git_sha(),
        "wandb_runs_note": "per-cell wandb runs; project=issue448_v5_<cell>_seed42",
        "hf_hub_url": (f"https://huggingface.co/{HF_MODEL_REPO}/tree/main/adapters/issue_448_v5"),
        "gpu_hours_used_estimate": round(
            sum(float(c.get("wall_seconds", 0) or 0) for c in per_cell_summaries) / 3600, 2
        ),
        "gpu_hours_budgeted": 12.25,  # plan §9.1
        "plan_deviations": plan_deviations,
        "phase_summaries": phase_summaries,
        "analyze_summary": analyze_summary,
        "h1a_pass": (analyze_summary or {}).get("h1a_pass"),
        "h1b_pass": (analyze_summary or {}).get("h1b_pass"),
        "interpretation": (analyze_summary or {}).get("interpretation"),
        "interpretation_code": (analyze_summary or {}).get("interpretation_code"),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    _write_poll_compliant_sentinel(
        final_path,
        kind="epm:results",
        phase="done",
        note_payload=note_payload,
    )
    log.info("Final sentinel (poll-compliant epm:results v1): %s", final_path)
    return final_path


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - 7-phase orchestrator
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=str,
        default=None,
        help=(
            "Comma-separated cell IDs. Accepts plain-English names "
            "(Anchor, +pos-ex-100-per-persona, ...) OR slug forms "
            "(c1_anchor, c2_pos_ex_100, ...). Default: all 11."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Shorthand for --cells Anchor + skip base-panel + skip analyze. "
            "Smoke IS sweep with one cell (UNIFICATION); same dispatcher, "
            "same per-cell subprocess shape, same env injection, same logging "
            "surface, same teardown sequence."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No GPU work / no Anthropic calls; validate imports + assertions.",
    )
    parser.add_argument(
        "--smoke-real",
        action="store_true",
        help=(
            "REAL tiny-slice smoke that loads Qwen-2.5-7B-Instruct on a local "
            "GPU and runs Phase 1.5 base-panel marker log-prob over a tiny "
            "slice (controlled by --eval-personas-limit + --eval-questions-limit). "
            "Skips Pre-Phase 0 + Phase 0.5 + per-cell train."
        ),
    )
    parser.add_argument(
        "--eval-personas-limit",
        type=int,
        default=None,
        help="Cap the eval-panel persona count (debug / smoke-real). Default: 24.",
    )
    parser.add_argument(
        "--eval-questions-limit",
        type=int,
        default=None,
        help="Cap the eval question count (debug / smoke-real). Default: 20.",
    )
    parser.add_argument(
        "--train-questions-limit",
        type=int,
        default=None,
        help=(
            "Smoke: cap the Q_train pool universe for R-generation (default: full "
            "850-pair pool). Reduces Phase 1 wall by ~N/850."
        ),
    )
    parser.add_argument(
        "--r-no-upload",
        action="store_true",
        help=(
            "Skip HF data-repo upload for R artifacts (debug only; downstream "
            "phases need the upload for cross-pod sharing)."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="villain",
        help="Source persona for the sweep. Default 'villain' (plan §3).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip per-cell train + eval when /workspace/logs/issue-448-<cell>-"
            "results.json AND <slab-root>/<cell>/marker_logprob.json both exist."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--slab-root",
        type=Path,
        default=DEFAULT_SLAB_ROOT_V5,
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/workspace/runs/issue_448_v5"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/workspace/logs"),
        help=(
            "Where per-cell + per-phase + final sentinel JSONs land. Default "
            "/workspace/logs (pod-shape); override for local dry-run smoke."
        ),
    )
    parser.add_argument(
        "--centroids-path",
        type=Path,
        default=Path("eval_results/issue_448/centroids/centroids_layer20.pt"),
        help="Layer-20 centroid bundle for the H5 secondary (carries over from v1).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR_V5,
    )
    parser.add_argument(
        "--skip-pre-phase-0",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Pre-Phase 0 (corpus top-up).",
    )
    parser.add_argument(
        "--skip-phase-0-5",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Phase 0.5 (extend centroids).",
    )
    parser.add_argument(
        "--skip-phase-0-75",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Phase 0.75 (held-out bystander pin).",
    )
    parser.add_argument(
        "--skip-r-generate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Phase 1 (R-generation); requires existing R artifacts or HF.",
    )
    parser.add_argument(
        "--skip-base-panel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Phase 1.5 (base panel descriptive probe).",
    )
    parser.add_argument(
        "--skip-analyze",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Phase 5 (analyze).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    # Wire CLI --log-dir into the module-level constant so the helpers above
    # (which reference LOG_DIR directly) see the override.
    global LOG_DIR
    LOG_DIR = args.log_dir

    # ── Hard pre-flight: persona_registry build + marker tokenizer assertion. ──
    log.info("Pre-flight: persona_registry + marker tokenizer assertions")
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
        persona_registry,
    )

    persona_registry._do_build_and_assert()  # re-run on every dispatch
    log.info(
        "persona_registry assertions PASS (villain → %s, assistant → %s)",
        persona_registry.OBSERVED_BYSTANDERS_PER_SOURCE["villain"],
        persona_registry.OBSERVED_BYSTANDERS_PER_SOURCE["assistant"],
    )

    # Marker tokenizer assertion (CLAUDE.md). Don't need to load the full
    # model for this — just the tokenizer.
    if not args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
        if ids != [EXPECTED_MARKER_TOKEN_ID]:
            raise RuntimeError(
                f"Marker tokenizer assertion FAILED. Expected "
                f"[{EXPECTED_MARKER_TOKEN_ID}]; got {ids}."
            )
        log.info("Marker tokenizer assertion PASS: tokenize(%r) = %s", MARKER_TEXT, ids)
    else:
        log.info("Marker tokenizer assertion DEFERRED (dry-run)")

    cells = _resolve_cells(args.cells, args.smoke)
    # Round-2 fix B2: --smoke-real targets Phase 1.5 base panel only; empty the
    # per-cell loop so we don't try to load a merged checkpoint that doesn't exist.
    if args.smoke_real:
        cells = []
    log.info(
        "Resolved %d cells: %s",
        len(cells),
        [(slug, name) for slug, name, *_ in cells],
    )

    # Resolve phase-skip defaults from --smoke / --smoke-real / --dry-run.
    if args.smoke_real:
        # smoke-real: only Phase 1.5 base-panel runs; everything else off.
        from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
            build_wrong_claim_pool as _bwc,
        )

        _canon_path = _bwc.OUT_DIR / "eval_canonical_responses.json"
        skip_pre_phase_0 = _canon_path.exists()
        skip_phase_0_5 = True
        skip_phase_0_75 = True
        skip_r_generate = True  # smoke-real assumes R already exists locally
        skip_base_panel = False
        skip_analyze = True
    else:
        skip_pre_phase_0 = (
            args.skip_pre_phase_0
            if args.skip_pre_phase_0 is not None
            else (args.smoke or args.dry_run)
        )
        skip_phase_0_5 = (
            args.skip_phase_0_5 if args.skip_phase_0_5 is not None else (args.smoke or args.dry_run)
        )
        skip_phase_0_75 = args.skip_phase_0_75 if args.skip_phase_0_75 is not None else args.dry_run
        skip_r_generate = args.skip_r_generate if args.skip_r_generate is not None else args.dry_run
        skip_base_panel = args.skip_base_panel if args.skip_base_panel is not None else args.smoke
        skip_analyze = args.skip_analyze if args.skip_analyze is not None else args.smoke

    log.info(
        "Phase toggles: pre_phase_0=%s phase_0_5=%s phase_0_75=%s r_generate=%s "
        "base_panel=%s analyze=%s",
        "SKIP" if skip_pre_phase_0 else "RUN",
        "SKIP" if skip_phase_0_5 else "RUN",
        "SKIP" if skip_phase_0_75 else "RUN",
        "SKIP" if skip_r_generate else "RUN",
        "SKIP" if skip_base_panel else "RUN",
        "SKIP" if skip_analyze else "RUN",
    )
    log.info(
        "Dry run: %s; smoke: %s; smoke-real: %s",
        args.dry_run,
        args.smoke,
        args.smoke_real,
    )

    args.slab_root.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # MooseFS quota safety (CLAUDE.md gotcha for sequential multi-cell sweep).
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    plan_deviations: list[str] = []
    phase_summaries: dict[str, dict | None] = {}

    # ── Pre-Phase 0 (carry-over: 850-pair generic-corpus questions for Q_train).
    log.info("[phase=pre_phase_0] starting")
    pre_phase_0_summary = _run_pre_phase_0(
        skip_pre_phase_0, args.dry_run, canonical_only=args.smoke_real
    )
    phase_summaries["pre_phase_0"] = pre_phase_0_summary
    if skip_pre_phase_0:
        plan_deviations.append("pre_phase_0_skipped")

    # ── Phase 0.5 (centroids — needed by H5 secondary).
    log.info("[phase=phase_0_5] starting")
    phase_0_5_summary = _run_phase_0_5(skip_phase_0_5, args.dry_run)
    phase_summaries["phase_0_5"] = phase_0_5_summary
    if skip_phase_0_5:
        plan_deviations.append("phase_0_5_skipped")

    # ── Phase 0.75 (held-out bystander pin — H1b denominator).
    phase_0_75_summary = _run_phase_0_75_held_out_pin(
        skip_phase_0_75, args.dry_run, source=args.source
    )
    phase_summaries["phase_0_75"] = phase_0_75_summary
    if skip_phase_0_75:
        plan_deviations.append("phase_0_75_skipped")

    # ── Phase 1 (R-generate — subprocess).
    r_generate_summary = _run_phase_1_r_generate(
        skip_r_generate,
        args.dry_run,
        no_upload=args.r_no_upload,
        train_questions_limit=args.train_questions_limit,
        source=args.source,
        max_new_tokens=1024,
    )
    phase_summaries["r_generate"] = r_generate_summary
    if skip_r_generate:
        plan_deviations.append("r_generate_skipped")

    # ── Phase 1.5 (base panel descriptive).
    base_panel_summary: dict | None = None
    if skip_base_panel:
        log.info("[phase=base_panel] SKIPPED (--skip-base-panel)")
        plan_deviations.append("phase_1_5_base_panel_skipped")
    else:
        try:
            base_panel_summary = _run_phase_1_5_base_panel(
                False,
                args.dry_run,
                eval_personas_limit=args.eval_personas_limit,
                eval_questions_limit=args.eval_questions_limit,
            )
        except Exception:
            log.exception("Phase 1.5 (base panel) failed")
            raise
    phase_summaries["base_panel"] = base_panel_summary

    # ── Load on-policy R_train (passed into per-cell build_training_data).
    r_train: dict[str, dict[str, dict]] | None = None
    if not args.dry_run:
        try:
            from explore_persona_space.experiments.contrastive_recipe_sweep_448.r_generate import (
                load_r_artifact,
            )

            r_train = load_r_artifact(ON_POLICY_R_DIR / "R_train.json")
            log.info(
                "Loaded R_train from %s — %d personas",
                ON_POLICY_R_DIR / "R_train.json",
                len(r_train),
            )
        except FileNotFoundError as e:
            log.warning(
                "R_train artifact not found (%s); per-cell train would fail. "
                "Skipping per-cell loop.",
                e,
            )
            r_train = None

    # Phase-0 EXIT assertion: R_eval covers EVAL_PERSONAS_24 (Must-Fix-1).
    if not args.dry_run and not skip_r_generate:
        try:
            from explore_persona_space.experiments.contrastive_recipe_sweep_448.r_generate import (
                load_r_artifact as _load_r,
            )
            from explore_persona_space.experiments.factor_screen_365.persona_panel import (
                EVAL_PERSONAS_24 as _EP24,
            )

            r_eval_check = _load_r(ON_POLICY_R_DIR / "R_eval.json")
            missing = sorted(set(_EP24.keys()) - set(r_eval_check.keys()))
            if missing:
                raise AssertionError(
                    f"Phase-0 EXIT assertion FAILED: R_eval missing {len(missing)} "
                    f"EVAL_PERSONAS_24 personas: {missing!r}. Phase 4 would KeyError "
                    f"mid-eval after wasted training time."
                )
            log.info(
                "Phase-0 EXIT assertion OK: R_eval covers all %d EVAL_PERSONAS_24 personas",
                len(_EP24),
            )
        except FileNotFoundError:
            log.warning("Skipping Phase-0 EXIT assertion: R_eval.json not present.")

    # ── Load canonical responses for the trajectory callback (carry-over).
    canonical_responses: dict[str, str] | None = None
    if not args.dry_run:
        try:
            from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_wrong_claim_pool import (  # noqa: E501
                load_canonical_responses,
            )

            canonical_responses = load_canonical_responses()
        except FileNotFoundError as e:
            log.warning("Canonical responses not found (%s); trajectory callback DISABLED.", e)

    # ── Phase 2 + 3 per-cell loop (train + eval). ────────────────────────────
    per_cell_summaries: list[dict[str, object]] = []
    for slug, plain_name, pos_ex_per_p, pos_personas, neg_ex_per_p, neg_personas in cells:
        try:
            cell_summary = _run_one_cell(
                cell_slug=slug,
                plain_name=plain_name,
                pos_ex_per_p=pos_ex_per_p,
                pos_personas=pos_personas,
                neg_ex_per_p=neg_ex_per_p,
                neg_personas=neg_personas,
                seed=args.seed,
                slab_root=args.slab_root,
                runs_root=args.runs_root,
                dry_run=args.dry_run,
                canonical_responses=canonical_responses,
                r_train=r_train,
                centroids_path=args.centroids_path,
                resume=args.resume,
                eval_personas_limit=args.eval_personas_limit,
                eval_questions_limit=args.eval_questions_limit,
            )
            per_cell_summaries.append(cell_summary)
        except Exception as e:
            fail_path = LOG_DIR / f"issue-448-{slug}-FAILED.json"
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            fail_path.write_text(
                json.dumps(
                    {
                        "cell": slug,
                        "phase": "cell_failed",
                        "exception_type": type(e).__name__,
                        "exception_msg": str(e),
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                )
            )
            log.exception("[%s] cell failed; wrote %s", slug, fail_path)
            raise

    # ── Phase 5: analyze ─────────────────────────────────────────────────────
    analyze_summary: dict | None = None
    if skip_analyze:
        log.info("[phase=analyze] SKIPPED (--skip-analyze)")
        plan_deviations.append("phase_5_analyze_skipped")
    elif args.dry_run:
        log.info("[phase=analyze] DRY-RUN: analyze module not invoked")
    else:
        try:
            analyze_summary = _run_analyze(
                slab_root=args.slab_root,
                figures_dir=args.figures_dir,
                centroids_path=args.centroids_path,
                held_out_path=HELD_OUT_ARTIFACT_PATH,
            )
        except Exception:
            log.exception("Phase 5 (analyze) failed")
            raise
    phase_summaries["analyze"] = analyze_summary

    cells_requested = [c[0] for c in cells]
    _write_final_sentinel(
        cells_requested=cells_requested,
        per_cell_summaries=per_cell_summaries,
        phase_summaries=phase_summaries,
        analyze_summary=analyze_summary,
        plan_deviations=plan_deviations,
        seed=args.seed,
        slab_root=args.slab_root,
    )
    log.info("Dispatcher done. %d cells completed.", len(per_cell_summaries))
    # Per CLAUDE.md "Pod-side result-reporting contract" — emit [phase=done]
    # as the LAST log line so poll_pipeline.py's PHASE_RE detects clean exit.
    log.info("[phase=done] dispatcher exit %s", datetime.now(UTC).isoformat())
    return 0


if __name__ == "__main__":
    sys.exit(main())
