# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek α/ΔG intentional
#!/usr/bin/env python3
"""Task #555 — single (cell, seed) worker: rebuild pool per fresh seed → train
HARD-STOPPED at optimizer step 5 (scheduler horizon untouched) → post-hoc
fraction selection ({1.0} of the realized stop → step 5 exact) → eval at the
step-5 no-implant snapshot.

Forked from scripts/i534_run_cell.py (the established per-issue fork chain
i504 → i530 → i534 → i555). Deltas vs i534 (plan #555 §4.3 a):

  1. Namespace 555: slab `eval_results/issue_555`, runs `/workspace/runs/issue_555`,
     HF `adapters/issue_555/<run_slug>`, sentinel
     `issue-555-<cell>-seed<S><suffix>-results.json`, task_id 555,
     WandB `issue555_<run_slug>_eps{epochs}_lr{lr:g}`.
  2. Train pool REBUILT per fresh seed via `build_cell_504(seed=...)` — the
     #530 HF pools exist only for seeds 42/137 (Hub-verified), so the
     `--train-pool-from-hf` path + byte-compare diagnostic are REMOVED.
     Post-build: assert row count == 400 (the scheduler-horizon invariant),
     then upload the pool to the HF data repo at
     `issue555_null_calibration/train_pools/<cell>_seed<S><suffix>.jsonl`
     (Hub-API list_repo_files verification; upload policy "datasets must
     upload").
  3. `--hard-stop-at-step` (default 5; 0 disables → parent band-stop
     behavior, used by the `_bandctrl` positive-control cell) threaded to
     `train_one_cell(hard_stop_at_step=..., hard_stop_expect_max_steps=300)`.
  4. `--fractions` default "1.0" (one read point; the selector maps 1.00 of
     realized stop 5 → step 5 exact).
  5. Slab-dir suffix fix (parent bug, i534_run_cell.py:461-473): the cell
     slab dir AND the sentinel name include `--hf-path-suffix`, so the
     `_bandctrl` control cell cannot overwrite the production near/seed-7
     trajectory.json. Production cells use the empty suffix → naming
     identical to the parent's shape.
  6. Post-train asserts (fail loud, hard-stopped cells only):
     `band_stop_meta.json` has stop_step == hard_stop step and
     stopped == false (do NOT assert the stop_reason string — it reads
     "epoch_ceiling" for hard-stopped cells, a known cosmetic); snapshot dirs
     step_0001..step_000<k> all exist; adapter-distinctness guard — the
     selected snapshot's lora_B tensors must not all be zeros (lora_B is
     zero-initialized, so an all-zero lora_B is a no-op adapter and any
     downstream ΔG read is vacuous).
  7. The eval call keeps `--fraction-manifest` (the #534 round-2
     adapter-applied guard) and production cells NEVER pass `--no-kl` (the
     HF logits phase is where z_marker/z_eos/logZ are captured — the
     four-floats storage contract).

Everything else — recipe (lr 5e-6, r=8/α=32, band-stop [5,12] @
eval_every=10/min_steps=20 kept attached and provably inert below step 10,
epoch ceiling 12 → max_steps 300, warmup 15 steps), marker assert, GPU
pinning, `[phase=...]` log surface, sentinel schema — is inherited verbatim.

Usage (production cell):
    uv run python scripts/i555_run_cell.py \\
        --cell c504v3_near --seed 7 --gpu-id 0 \\
        --arm-to-n-json eval_results/issue_530/phase0_5_gates.json

Usage (band-stop positive-control cell — eval-path validation gate):
    uv run python scripts/i555_run_cell.py \\
        --cell c504v3_near --seed 7 --gpu-id 1 --hf-path-suffix _bandctrl \\
        --hard-stop-at-step 0 --fractions 1.0 \\
        --arm-to-n-json eval_results/issue_530/phase0_5_gates.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i555.run_cell")

# --- #555 dispatcher defaults ------------------------------------------------
# Inherited VERBATIM from #534/#530 (the single-variable contract, plan §4.2).
LR_DEFAULT_555: float = 5e-6
EPOCHS_DEFAULT_555: int = 12
SNAPSHOT_EVERY_STEPS_DEFAULT: int = 1
SNAPSHOT_MAX_COUNT_DEFAULT: int = 64
# NEW (#555 §4.2): the read-point truncation. 0 disables (the `_bandctrl`
# positive-control cell trains to the verbatim band-stop).
HARD_STOP_AT_STEP_DEFAULT: int = 5
# The parent scheduler horizon: 400 rows / (batch 4 × grad_accum 4) = 25
# steps/epoch × 12-epoch ceiling = 300; warmup_ratio 0.05 → 15 warmup steps.
EXPECT_MAX_STEPS_DEFAULT: int = 300
# The scheduler-horizon invariant the fresh-seed pool must satisfy.
EXPECTED_POOL_ROWS: int = 400
# Post-hoc fraction set: single read point (plan §4.3 a item 4).
FRACTIONS_DEFAULT: str = "1.0"
# HF data repo for the fresh-seed pool uploads (upload policy: datasets).
TRAIN_POOL_HF_REPO: str = "superkaiba1/explore-persona-space-data"
TRAIN_POOL_HF_PATH_TMPL: str = "issue555_null_calibration/train_pools/{run_slug}.jsonl"


def _write_sentinel(path: Path, *, kind: str, phase: str, note: dict) -> None:
    """Write a poll_pipeline.py-compliant sentinel (sentinel_schema_version=1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": 555,
                "by": "i555_run_cell",
                "ts": datetime.now(UTC).isoformat(),
                "phase": phase,
                "note": json.dumps(note),
            },
            indent=2,
        )
    )


def _upload_train_pool(train_jsonl: Path, run_slug: str) -> str:
    """Upload the fresh-seed pool to the HF data repo + Hub-API verify (fail loud).

    Upload policy: datasets must upload so any pod can access without scp.
    Returns the in-repo path. Uses `upload_file` (one file) + `list_repo_files`
    verification (never the `hf` CLI — .claude/rules/upload-policy.md).
    """
    from huggingface_hub import HfApi, list_repo_files

    rel = TRAIN_POOL_HF_PATH_TMPL.format(run_slug=run_slug)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    log.info("[phase=upload_pool] uploading %s → %s :: %s", train_jsonl, TRAIN_POOL_HF_REPO, rel)
    api.upload_file(
        path_or_fileobj=str(train_jsonl),
        path_in_repo=rel,
        repo_id=TRAIN_POOL_HF_REPO,
        repo_type="dataset",
    )
    files = list_repo_files(
        TRAIN_POOL_HF_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN")
    )
    if rel not in files:
        raise RuntimeError(
            f"[upload_pool] post-upload Hub-API verify FAILED: {rel} not in "
            f"{TRAIN_POOL_HF_REPO} file listing. Refuse to proceed."
        )
    log.info("[phase=upload_pool] verified %s on %s", rel, TRAIN_POOL_HF_REPO)
    return rel


def _maybe_upload_pool(train_jsonl: Path, run_slug: str, *, skip: bool) -> str | None:
    """Upload the pool unless ``skip`` (local CPU smoke); returns the HF path or None."""
    if skip:
        log.warning(
            "[phase=upload_pool] SKIPPED per --skip-pool-upload (smoke only — "
            "production cells must upload)."
        )
        return None
    return _upload_train_pool(train_jsonl, run_slug)


def _assert_pool_rows(train_jsonl: Path, expected: int = EXPECTED_POOL_ROWS) -> int:
    """Assert the fresh-seed pool has exactly `expected` rows (horizon invariant).

    400 rows / (batch 4 × grad_accum 4) = 25 optimizer steps/epoch × 12-epoch
    ceiling → Trainer max_steps = 300 and warmup_steps = 15. A different row
    count silently re-parameterizes warmup and breaks weight-identity with the
    parent's step-5 trajectory (plan §4.2 / risk row 3).
    """
    n_rows = sum(1 for line in train_jsonl.read_text().splitlines() if line.strip())
    if n_rows != expected:
        raise RuntimeError(
            f"fresh-seed pool at {train_jsonl} has {n_rows} rows, expected {expected} — "
            "the scheduler horizon (max_steps=300, warmup=15) depends on the row "
            "count; a drifted pool breaks the read-point truncation. Investigate "
            "build_cell_504 for this seed before continuing."
        )
    log.info("[phase=pool_rows] row-count assert PASS: %d rows", n_rows)
    return n_rows


def _assert_hard_stop_artifacts(
    cell: str,
    snapshot_dir: Path,
    stop_at_step: int,
) -> None:
    """Post-train asserts for hard-stopped cells (plan §4.3 a item 6).

    (i) band_stop_meta.json: stop_step == stop_at_step AND stopped == false
        (the band-stop never fired — it is provably inert below step 10; the
        stop_reason STRING is deliberately not asserted: it reads
        "epoch_ceiling" for hard-stopped cells, a known cosmetic).
    (ii) snapshot dirs step_0001..step_000<stop_at_step> all exist.
    """
    meta_path = snapshot_dir / "band_stop_meta.json"
    if not meta_path.exists():
        raise RuntimeError(
            f"[{cell}] band_stop_meta.json missing at {meta_path} after training — "
            "the snapshot extension did not run; check marker_band_snapshot_* threading."
        )
    meta = json.loads(meta_path.read_text())
    if int(meta["stop_step"]) != stop_at_step:
        raise RuntimeError(
            f"[{cell}] band_stop_meta stop_step={meta['stop_step']} != hard-stop "
            f"target {stop_at_step} — the HardStopAtStepCallback did not stop where "
            "expected (callback ordering / horizon drift?)."
        )
    if meta.get("stopped", False):
        raise RuntimeError(
            f"[{cell}] band_stop_meta stopped=true — the BAND-stop fired before the "
            f"hard stop at step {stop_at_step}; that contradicts its provable "
            "inertness below step 10 (eval_every=10, min_steps=20). Investigate."
        )
    missing = [
        f"step_{s:04d}"
        for s in range(1, stop_at_step + 1)
        if not (snapshot_dir / f"step_{s:04d}").is_dir()
    ]
    if missing:
        raise RuntimeError(
            f"[{cell}] per-step snapshot dirs missing under {snapshot_dir}: {missing} — "
            "the snapshot callback (cadence 1) should have written every step."
        )
    log.info(
        "[phase=hard_stop_asserts] PASS: stop_step=%d, stopped=false, snapshots "
        "step_0001..step_%04d present",
        stop_at_step,
        stop_at_step,
    )


def _assert_adapter_distinct(cell: str, adapter_dir: Path) -> None:
    """Adapter-distinctness guard: the snapshot must not be a no-op adapter.

    lora_B is zero-initialized, so an all-zero lora_B tensor set means the
    snapshot applies the identity — any downstream ΔG read would be vacuously
    0 and indistinguishable from the adapter-not-applied eval regression
    (plan §7 risk row 1). Asserts EVERY lora_B tensor is non-zero somewhere.
    """
    weights = adapter_dir / "adapter_model.safetensors"
    if not weights.exists():
        raise RuntimeError(
            f"[{cell}] adapter weights missing at {weights} — cannot run the "
            "adapter-distinctness guard."
        )
    from safetensors import safe_open

    all_zero: list[str] = []
    n_lora_b = 0
    with safe_open(str(weights), framework="pt", device="cpu") as fh:
        for key in fh.keys():  # noqa: SIM118 — safetensors handle has no __iter__/__contains__
            if "lora_B" not in key:
                continue
            n_lora_b += 1
            t = fh.get_tensor(key)
            if not bool(t.abs().max().item() > 0):
                all_zero.append(key)
    if n_lora_b == 0:
        raise RuntimeError(f"[{cell}] no lora_B tensors found in {weights} — not a LoRA adapter?")
    if all_zero:
        raise RuntimeError(
            f"[{cell}] adapter-distinctness guard FAILED: {len(all_zero)}/{n_lora_b} "
            f"lora_B tensors are all-zeros (no-op adapter; first offenders: "
            f"{all_zero[:5]}). A step-5 snapshot with zero lora_B means the "
            "optimizer never touched those modules — the ΔG read would be vacuous."
        )
    log.info(
        "[phase=adapter_distinct] PASS: all %d lora_B tensors non-zero in %s",
        n_lora_b,
        weights,
    )


def _assert_trajectory_complete(cell: str, out_traj: Path, n_expected: int) -> None:
    """Post-eval completeness check: trajectory.json exists with one entry per fraction."""
    if not out_traj.exists():
        raise RuntimeError(
            f"[{cell}] eval_trajectory subprocess exited 0 but {out_traj} missing — "
            "silent eval failure (feedback_eval_script_silent_not_present_misdiagnosis)."
        )
    traj = json.loads(out_traj.read_text())
    n_ckpts = len(traj.get("checkpoints", []))
    if n_ckpts != n_expected:
        raise RuntimeError(
            f"[{cell}] trajectory.json has {n_ckpts} checkpoints; expected "
            f"{n_expected} (one per realized fraction)."
        )


def _load_eval_only_index(
    cell: str,
    run_slug: str,
    ckpt_index_path: Path,
    manifest_out: Path,
) -> dict:
    """Validate + load the prior full run's artifacts for ``--eval-only``.

    Requires ``checkpoint_index.json``, ``fraction_manifest.json``, and every
    indexed adapter's ``adapter_model.safetensors`` on disk. Fails loud on
    anything missing — NO silent fallback to retraining.
    """
    if not ckpt_index_path.exists():
        raise RuntimeError(
            f"--eval-only requires an existing {ckpt_index_path} from a prior full "
            "run — run i555_run_cell.py WITHOUT --eval-only first."
        )
    if not manifest_out.exists():
        raise RuntimeError(
            f"--eval-only requires an existing {manifest_out} — the eval's "
            "source-manifest guard reads the selector's source ΔG expectations from it."
        )
    ckpt_index = json.loads(ckpt_index_path.read_text())
    for frac_str, entry in sorted(ckpt_index.items(), key=lambda kv: float(kv[0])):
        snap = Path(entry["path"])
        if not (snap / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"--eval-only: indexed adapter for frac={frac_str} missing on disk at "
                f"{snap} — re-fetch from HF (adapters/issue_555/{run_slug}/"
                f"ckpt_frac{float(frac_str):.2f}) or re-run the full cell."
            )
    log.info("[phase=eval_only_%s] index verified: %s", cell, ckpt_index)
    return ckpt_index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_555"))
    ap.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_555"))
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument(
        "--r-train-path", type=Path, default=Path("data/issue_472/on_policy_R/R_train.json")
    )
    ap.add_argument(
        "--r-eval-path", type=Path, default=Path("data/issue_472/on_policy_R/R_eval.json")
    )
    ap.add_argument(
        "--arm-to-n-json",
        type=Path,
        required=True,
        help=(
            "Phase 0.5 gates artifact ({arm_slug: positioned_N} + held_out_panel). "
            "#555 reuses #530's committed eval_results/issue_530/phase0_5_gates.json "
            "as-is (plan §10 — identical predictor table is REQUIRED for comparability)."
        ),
    )
    ap.add_argument("--chosen-rank", type=int, default=8, help="LoRA rank (VERBATIM #534).")
    ap.add_argument("--chosen-alpha", type=int, default=32, help="LoRA α (VERBATIM #534).")
    ap.add_argument(
        "--lr",
        type=float,
        default=LR_DEFAULT_555,
        help=f"Marker-only learning rate (default {LR_DEFAULT_555:g}; VERBATIM #534).",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS_DEFAULT_555,
        help=(
            f"Epoch ceiling (default {EPOCHS_DEFAULT_555}; VERBATIM #534 — parameterizes "
            "the scheduler horizon, never reached under the hard stop)."
        ),
    )
    ap.add_argument(
        "--hard-stop-at-step",
        type=int,
        default=HARD_STOP_AT_STEP_DEFAULT,
        help=(
            f"Stop training after this optimizer step WITHOUT touching the scheduler "
            f"horizon (default {HARD_STOP_AT_STEP_DEFAULT}; THE #555 read point). "
            "0 disables — parent band-stop behavior (the _bandctrl positive-control cell)."
        ),
    )
    ap.add_argument(
        "--hard-stop-expect-max-steps",
        type=int,
        default=EXPECT_MAX_STEPS_DEFAULT,
        help=(
            f"Asserted Trainer max_steps at train begin when the hard stop is armed "
            f"(default {EXPECT_MAX_STEPS_DEFAULT} = 400 rows / 16 eff. batch × 12 epochs; "
            "horizon guard for the warmup parameterization)."
        ),
    )
    ap.add_argument(
        "--snapshot-every-steps",
        type=int,
        default=SNAPSHOT_EVERY_STEPS_DEFAULT,
        help=f"Per-step adapter snapshot cadence (default {SNAPSHOT_EVERY_STEPS_DEFAULT}).",
    )
    ap.add_argument(
        "--snapshot-max-count",
        type=int,
        default=SNAPSHOT_MAX_COUNT_DEFAULT,
        help=f"Hard cap on snapshots (default {SNAPSHOT_MAX_COUNT_DEFAULT}; disk bound).",
    )
    ap.add_argument(
        "--fractions",
        default=FRACTIONS_DEFAULT,
        help=(
            f"Post-hoc fractions of the REALIZED stop step (default {FRACTIONS_DEFAULT!r}; "
            "with the step-5 hard stop, frac 1.00 → step 5 exact)."
        ),
    )
    ap.add_argument(
        "--skip-pool-upload",
        action="store_true",
        help=(
            "Skip the HF data-repo pool upload (local CPU smoke only — production "
            "cells MUST upload; upload policy 'datasets must upload')."
        ),
    )
    ap.add_argument(
        "--skip-source-trajectory",
        action="store_true",
        help="Forwarded to i534_select_fractions.py (descope ladder).",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Re-run path: SKIP pool build, training, and fraction selection; "
            "consume the EXISTING checkpoint_index.json + fraction_manifest.json "
            "from a prior full run and re-run only the trajectory eval + sentinel. "
            "Fails loud when the index/manifest/snapshot dirs are missing."
        ),
    )
    ap.add_argument(
        "--max-new-tokens-eval",
        type=int,
        default=2048,
        help="Eval max_new_tokens (CLAUDE.md rule: >= 2x longest trained completion).",
    )
    ap.add_argument(
        "--max-model-len-eval",
        type=int,
        default=None,
        help=(
            "vLLM max_model_len for the nested eval. If unset, computed as "
            "max(2048, max_new_tokens_eval + 512)."
        ),
    )
    ap.add_argument(
        "--no-kl",
        action="store_true",
        help=(
            "Skip DV-B KL (also skips the z_marker/z_eos/logZ four-floats capture "
            "— NEVER for production #555 cells; the storage contract lives in the "
            "HF logits phase)."
        ),
    )
    ap.add_argument("--report-to", default="wandb")
    ap.add_argument(
        "--hf-path-suffix",
        default="",
        help=(
            "Cell-variant suffix, e.g. _bandctrl for the positive-control cell. "
            "#555 fix: included in the slab dir AND the sentinel name (the parent "
            "omitted it from both, so the control would overwrite production "
            "near/seed-7; i534_run_cell.py:461-473)."
        ),
    )
    ap.add_argument(
        "--hf-model-repo",
        default="superkaiba1/explore-persona-space",
        help="HF model repo for the selector's per-fraction uploads + the final adapter.",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="ASSIGNED physical GPU index (train/sft.py SETS CUDA_VISIBLE_DEVICES to this).",
    )
    ap.add_argument(
        "--source",
        default=None,
        help="Source persona override. Default = villain (the canonical #504/#530/#534 source).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format=(
            f"%(asctime)s [phase=cell_{args.cell}_seed{args.seed}{args.hf_path_suffix}] "
            f"%(name)s %(levelname)s | %(message)s"
        ),
        stream=sys.stdout,
    )
    hard_stop: int | None = args.hard_stop_at_step if args.hard_stop_at_step > 0 else None
    log.info(
        "Assigned physical GPU --gpu-id=%d; hard_stop_at_step=%s; inherited "
        "CUDA_VISIBLE_DEVICES=%s (train/sft.py will SET CVD=str(gpu_id)).",
        args.gpu_id,
        hard_stop,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )

    # In-train per-fraction HF persistence is DISABLED by design (inherited
    # #534 divergence 4) — the post-hoc selector owns all fraction uploads.
    for var in (
        "EPM_PERSIST_TRAJECTORY_HF_REPO",
        "EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER",
        "EPM_PERSIST_ADAPTER_HF_REPO",
        "EPM_PERSIST_ADAPTER_HF_SUBFOLDER",
    ):
        if os.environ.pop(var, None) is not None:
            log.warning(
                "[phase=env] popped %s — #555 disables in-train per-fraction "
                "persistence (the selector owns fraction uploads).",
                var,
            )

    # Carry-over data dependencies from #472 (persona bank, centroids,
    # on-policy R) — idempotent HF pull, same as #530/#534.
    from explore_persona_space.experiments.contrastive_neg_geometry_530.data_deps import (
        prepare_data_dependencies,
    )

    log.info("[phase=cell_prepare_data] auto-downloading #472 carry-over artifacts (idempotent)")
    prepare_data_dependencies()

    # ── Marker tokenizer pre-spawn assert (CLAUDE.md marker-leakage rule). ──
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_IM_END_TOKEN_ID,
        MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        MARKER_TEXT,
        SOURCE_PERSONA,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"Marker tokenizer assertion FAILED: encode({MARKER_TEXT!r})={ids}, "
            f"expected [{EXPECTED_MARKER_TOKEN_ID}]."
        )
    log.info("[phase=preflight] marker assertion PASS: %r -> %s", MARKER_TEXT, ids)

    # Load Phase 0.5 cell resolution (arm_to_n + held_out_panel).
    arm_to_n_payload = json.loads(args.arm_to_n_json.read_text())
    arm_to_positioned_n = arm_to_n_payload.get("arm_to_positioned_n", {})
    smoke_mid_band_n = arm_to_n_payload.get("smoke_mid_band_n")
    held_out_panel: list[str] = arm_to_n_payload.get("held_out_panel", [])

    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
        load_r_artifact,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
        train_one_cell,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.build_training_data import (
        build_cell_504,
    )

    # #555 namespace. The run_slug INCLUDES the suffix everywhere (delta 5):
    # runs dir, HF subfolder, slab dir, sentinel, WandB run name.
    run_slug = f"{args.cell}_seed{args.seed}{args.hf_path_suffix}"
    run_dir = args.runs_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = run_dir / "train_pool.jsonl"
    final_adapter_dir = run_dir / "adapter"
    ckpt_root = run_dir / "checkpoints"
    snapshot_dir = run_dir / "snapshots"
    ckpt_index_path = run_dir / "checkpoint_index.json"
    cell_slab_dir = args.slab_root / run_slug  # #555 fix: suffix INCLUDED.
    out_traj = cell_slab_dir / "trajectory.json"
    manifest_out = cell_slab_dir / "fraction_manifest.json"
    source_traj_out = cell_slab_dir / "source_steps_trajectory.json"
    sentinel = (
        args.log_dir / f"issue-555-{args.cell}-seed{args.seed}{args.hf_path_suffix}-results.json"
    )

    bank = load_persona_bank(args.bank_path)
    r_train = load_r_artifact(args.r_train_path)
    q_train, _q_eval = get_train_eval_questions()

    effective_source = args.source if args.source is not None else SOURCE_PERSONA
    if effective_source not in bank:
        raise KeyError(
            f"--source {effective_source!r} missing from persona bank at {args.bank_path}."
        )
    log.info("[phase=source] effective source persona = %r", effective_source)

    effective_lr = args.lr
    effective_epochs = args.epochs
    wandb_suffix = f"_eps{effective_epochs}_lr{effective_lr:g}"

    parsed_fracs = tuple(sorted(float(x.strip()) for x in args.fractions.split(",") if x.strip()))
    if not parsed_fracs or any(f <= 0 or f > 1.0 for f in parsed_fracs):
        raise ValueError(f"--fractions {args.fractions!r} must be floats in (0, 1].")

    if args.eval_only:
        # ── Eval-only re-run: NO retraining, NO re-selection. ────────────────
        ckpt_index = _load_eval_only_index(args.cell, run_slug, ckpt_index_path, manifest_out)
        # The train path pins CUDA_VISIBLE_DEVICES inside train/sft.py; eval-only
        # never trains, so pin here or the nested vLLM lands on the wrong GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    else:
        # ── Phase: per-fresh-seed pool rebuild (#555 delta 2 — NO HF fetch). ──
        log.info(
            "[phase=build_%s] building training data via build_cell_504(seed=%d)",
            args.cell,
            args.seed,
        )
        build_cell_504(
            args.cell,
            train_jsonl,
            r_train=r_train,
            arm_to_positioned_n=arm_to_positioned_n,
            q_train=q_train,
            persona_bank=bank,
            source=effective_source,
            marker_text=MARKER_TEXT,
            smoke_mid_band_n=smoke_mid_band_n,
            seed=args.seed,
        )
        _assert_pool_rows(train_jsonl)
        pool_hf_path = _maybe_upload_pool(train_jsonl, run_slug, skip=args.skip_pool_upload)

        # ── Snapshot dir: WIPE at train start (stale-attempt guard). ─────────
        if snapshot_dir.exists():
            log.warning("[phase=train_%s] wiping stale snapshot dir %s", args.cell, snapshot_dir)
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "[phase=train_%s] training (rank=%d, alpha=%d, lr=%g, epochs=%d, "
            "hard_stop_at_step=%s, expect_max_steps=%d, snapshot_every=%d, "
            "suppress_at_post_response_slot=%s)",
            args.cell,
            args.chosen_rank,
            args.chosen_alpha,
            effective_lr,
            effective_epochs,
            hard_stop,
            args.hard_stop_expect_max_steps,
            args.snapshot_every_steps,
            MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        )
        train_one_cell(
            cell_slug=args.cell,
            seed=args.seed,
            train_jsonl=train_jsonl,
            output_dir=final_adapter_dir,
            ckpt_root=ckpt_root,
            fractions=parsed_fracs,
            fallback=False,
            report_to=args.report_to,
            gpu_id=args.gpu_id,
            lr_override=effective_lr,
            epochs_override=effective_epochs,
            lora_r_override=args.chosen_rank,
            lora_alpha_override=args.chosen_alpha,
            hf_path_in_repo_override=f"adapters/issue_555/{run_slug}",
            run_name_override=f"issue555_{run_slug}{wandb_suffix}",
            marker_suppress_at_post_response_slot=MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
            marker_im_end_token_id=MARKER_IM_END_TOKEN_ID,
            marker_band_snapshot_every_steps=args.snapshot_every_steps,
            marker_band_snapshot_dir=snapshot_dir,
            marker_band_snapshot_max_count=args.snapshot_max_count,
            # #555 — THE read-point truncation (None for the _bandctrl cell).
            hard_stop_at_step=hard_stop,
            hard_stop_expect_max_steps=(
                args.hard_stop_expect_max_steps if hard_stop is not None else None
            ),
        )
        log.info("[phase=train_%s] done; snapshots under %s", args.cell, snapshot_dir)

        # ── Post-train asserts (#555 delta 6; hard-stopped cells only). ──────
        if hard_stop is not None:
            _assert_hard_stop_artifacts(args.cell, snapshot_dir, hard_stop)

        # Free in-process LoRA-training GPU memory before the selector + eval.
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Phase: post-hoc fraction selection (NESTED subprocess; reuses the
        # #534 selector verbatim — it already handles the never-band-stopped
        # case: "cell never band-stopped; using S=last step", frac 1.00 → the
        # hard-stop step exact). ─────────────────────────────────────────────
        select_cmd = [
            "uv",
            "run",
            "python",
            "scripts/i534_select_fractions.py",
            "--snapshot-dir",
            str(snapshot_dir),
            "--train-jsonl",
            str(train_jsonl),
            "--checkpoint-index-out",
            str(ckpt_index_path),
            "--manifest-out",
            str(manifest_out),
            "--source-traj-out",
            str(source_traj_out),
            "--fractions",
            args.fractions,
            "--hf-repo",
            args.hf_model_repo,
            "--hf-subfolder",
            f"adapters/issue_555/{run_slug}",
            "--final-adapter",
            str(final_adapter_dir),
            "--device",
            "cuda:0",  # CVD already restricts to the assigned physical GPU.
        ]
        if args.skip_source_trajectory:
            select_cmd.append("--skip-source-trajectory")
        log.info("[phase=select_%s] selector subprocess: %s", args.cell, " ".join(select_cmd))
        subprocess.run(select_cmd, env={**os.environ}, check=True)
        if not ckpt_index_path.exists():
            raise RuntimeError(
                f"[{args.cell}] selector exited 0 but {ckpt_index_path} missing — "
                "silent selection failure."
            )
        ckpt_index = json.loads(ckpt_index_path.read_text())
        log.info("[phase=select_%s] realized-fraction index: %s", args.cell, ckpt_index)

        # ── Adapter-distinctness guard on every SELECTED snapshot (#555
        # delta 6 item iii) — at a no-implant snapshot the eval's >2-nat
        # manifest guard has no statistical power, so a no-op adapter must be
        # caught here, before the eval reads a vacuous ΔG ≈ 0. ──────────────
        for entry in ckpt_index.values():
            _assert_adapter_distinct(args.cell, Path(entry["path"]))

    # ── Phase: eval_trajectory (NESTED subprocess: vLLM teardown isolation). ─
    eval_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if eval_cvd != str(args.gpu_id):
        raise RuntimeError(
            f"Pre-eval CUDA_VISIBLE_DEVICES={eval_cvd!r} != assigned --gpu-id={args.gpu_id}; "
            "the nested eval subprocess would run on the wrong physical GPU."
        )
    eval_max_new_tokens = args.max_new_tokens_eval
    eval_max_model_len = (
        args.max_model_len_eval
        if args.max_model_len_eval is not None
        else max(2048, eval_max_new_tokens + 512)
    )
    eval_cmd = [
        "uv",
        "run",
        "python",
        # #555 reuses the #504 eval rig byte-for-byte (adapter-applied guard +
        # per-batch byte-identical guard + four-floats capture in the HF phase).
        "scripts/i504_eval_trajectory.py",
        "--cell",
        args.cell,
        "--seed",
        str(args.seed),
        "--checkpoint-index",
        str(ckpt_index_path),
        "--out-path",
        str(out_traj),
        "--bank-path",
        str(args.bank_path),
        "--r-eval-path",
        str(args.r_eval_path),
        "--panel-json",
        str(args.arm_to_n_json),
        "--max-lora-rank",
        str(args.chosen_rank),
        "--max-new-tokens",
        str(eval_max_new_tokens),
        "--max-model-len",
        str(eval_max_model_len),
        "--source",
        effective_source,
        # Adapter-applied cross-check: the eval's own source-self ΔG must
        # agree with the selector manifest's teacher-forced read at the final
        # fraction (>2 nat disagreement = fail loud). At no-implant cells both
        # reads are ≈0 so the guard has no power there — the _bandctrl control
        # cell (expected ΔG 5–6 nats) is where this guard actually bites.
        "--fraction-manifest",
        str(manifest_out),
    ]
    if args.no_kl:
        eval_cmd.append("--no-kl")
        log.warning(
            "[phase=eval_%s] --no-kl set: the HF KL phase is where the z_marker/"
            "z_eos/logZ four-floats are captured — production #555 cells must "
            "NOT use this.",
            args.cell,
        )
    log.info(
        "[phase=eval_%s] nested eval subprocess (max_new_tokens=%d, max_model_len=%d): %s",
        args.cell,
        eval_max_new_tokens,
        eval_max_model_len,
        " ".join(eval_cmd),
    )
    subprocess.run(eval_cmd, env={**os.environ}, check=True)

    n_expected = len(ckpt_index) if args.eval_only else len(parsed_fracs)
    _assert_trajectory_complete(args.cell, out_traj, n_expected)

    _write_sentinel(
        sentinel,
        kind="epm:progress",
        phase=f"cell_done_{args.cell}_seed{args.seed}{args.hf_path_suffix}",
        note={
            "cell": args.cell,
            "seed": args.seed,
            "hf_path_suffix": args.hf_path_suffix,
            "trajectory_path": str(out_traj),
            "fraction_manifest_path": str(manifest_out),
            "adapter_hf_path": f"adapters/issue_555/{run_slug}",
            "train_pool_hf_path": (None if args.eval_only else pool_hf_path),
            "checkpoint_index": ckpt_index,
            "n_held_out_panel": len(held_out_panel),
            "lr": effective_lr,
            "epochs": effective_epochs,
            "hard_stop_at_step": hard_stop,
            "fractions": list(parsed_fracs),
            "eval_only": bool(args.eval_only),
        },
    )
    log.info("[phase=done] wrote sentinel → %s", sentinel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
