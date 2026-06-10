# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek α intentional
#!/usr/bin/env python3
"""Task #534 — single (cell, seed) worker: fetch pool → train w/ per-step snapshots →
post-hoc fraction selection → eval_trajectory at 4 realized fractions.

Forked from scripts/i530_run_cell.py (the established per-issue fork chain
i504 → i530 → i534). The SINGLE substantive change vs #530 is checkpoint
granularity (plan §4.1): `MarkerBandStopCallback` saves a per-step adapter
snapshot (`--snapshot-every-steps`, default 1, cap `--snapshot-max-count`
64), and the post-hoc selector (`scripts/i534_select_fractions.py`) maps
fractions {0.25, 0.50, 0.75, 1.00} of the REALIZED stop step onto the saved
snapshots, overwrites `checkpoint_index.json`, and the nested
`i504_eval_trajectory.py` rig evals all 4 fractions in one vLLM session.

Round-2 (eval-adapter-not-applied fix): the nested rig now gets
`--fraction-manifest` (the eval's own source-self ΔG must agree with the
selector's teacher-forced read at the final fraction — >2-nat disagreement
fails loud; the round-1 lora_int_id-reuse regression read a flat ≈0
trajectory at every fraction), and `--eval-only` re-runs eval + sentinel
from the EXISTING snapshots/index/manifest with NO retraining:

    uv run python scripts/i534_run_cell.py --cell c504v3_near --seed 42 \\
        --gpu-id 0 --arm-to-n-json eval_results/issue_530/phase0_5_gates.json \\
        --eval-only

Deltas vs i530_run_cell.py (plan §4.3 e):
  * Namespace 534: slab `eval_results/issue_534`, runs `/workspace/runs/issue_534`,
    HF `adapters/issue_534/<run_slug>`, sentinel `issue-534-<cell>-seed<S>-results.json`,
    task_id 534, WandB `issue534_<run_slug>_eps{epochs}_lr{lr:g}`.
  * `--snapshot-every-steps` (default 1) + `--snapshot-max-count` (default 64)
    threaded into `train_one_cell` → `TrainLoraConfig` → `MarkerBandStopCallback`.
    Snapshot dir `<run_dir>/snapshots/`, WIPED at train start (stale-attempt guard).
  * `--train-pool-from-hf` default ON: the per-cell train pool is downloaded
    byte-for-byte from `superkaiba1/explore-persona-space-data` at
    `issue530_desat_rerun/train_pools/<cell>_seed<S>.jsonl` (Hub-verified in
    the plan), SKIPPING `build_cell_504` — byte-identical training data with
    #530. A `build_cell_504` rebuild runs as a byte-compare DIAGNOSTIC (WARN
    on mismatch; the HF pool stays authoritative).
  * In-train per-fraction HF persistence DISABLED: `EPM_PERSIST_TRAJECTORY_*`
    + `EPM_PERSIST_ADAPTER_*` are explicitly popped from the env — the
    post-hoc selector owns all fraction uploads (the fractions are unknown
    in-train). The final adapter still uploads via the legacy
    `cfg.hf_path_in_repo` path.
  * The legacy `CheckpointAtFractionsCallback` stays attached and unchanged
    (it writes nothing before step 75 at max_steps=300, harmlessly; the
    selector OVERWRITES `checkpoint_index.json` afterward).

Everything else — recipe (lr 5e-6, r=8/α=32, band-stop [5,12] @
eval_every=10/min_steps=20 VERBATIM, epoch ceiling 12), marker assert, GPU
pinning, sentinel shape, `[phase=...]` log surface — is inherited from #530.

Usage:
    uv run python scripts/i534_run_cell.py \\
        --cell c504v3_near --seed 42 --gpu-id 0 \\
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

log = logging.getLogger("i534.run_cell")

# --- #534 dispatcher defaults ------------------------------------------------
# Inherited VERBATIM from #530 (the replication contract, plan §4.2).
LR_DEFAULT_534: float = 5e-6
EPOCHS_DEFAULT_534: int = 12
# NEW (the manipulated variable, plan §4.1): per-step snapshot cadence + cap.
SNAPSHOT_EVERY_STEPS_DEFAULT: int = 1
SNAPSHOT_MAX_COUNT_DEFAULT: int = 64
# Post-hoc fraction set (plan §4.3 d).
FRACTIONS_DEFAULT: str = "0.25,0.5,0.75,1.0"
# HF data repo + path template for the byte-identical #530 train pools
# (Hub-verified 2026-06-09, plan §"reuse" + assumption #2).
TRAIN_POOL_HF_REPO: str = "superkaiba1/explore-persona-space-data"
TRAIN_POOL_HF_PATH_TMPL: str = "issue530_desat_rerun/train_pools/{cell}_seed{seed}.jsonl"


def _write_sentinel(path: Path, *, kind: str, phase: str, note: dict) -> None:
    """Write a poll_pipeline.py-compliant sentinel (sentinel_schema_version=1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": 534,
                "by": "i534_run_cell",
                "ts": datetime.now(UTC).isoformat(),
                "phase": phase,
                "note": json.dumps(note),
            },
            indent=2,
        )
    )


def _fetch_train_pool_from_hf(cell: str, seed: int, dest: Path) -> Path:
    """Download the #530 train pool bytes from HF and copy to ``dest``.

    Per-file `hf_hub_download` (NOT snapshot_download — the
    siblings-truncation gotcha). Fails loud on any HF error; asserts the
    landed file is non-empty.
    """
    from huggingface_hub import hf_hub_download

    rel = TRAIN_POOL_HF_PATH_TMPL.format(cell=cell, seed=seed)
    log.info("[phase=fetch_pool] hf_hub_download %s :: %s", TRAIN_POOL_HF_REPO, rel)
    cached = hf_hub_download(
        TRAIN_POOL_HF_REPO,
        rel,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    size = dest.stat().st_size
    if size == 0:
        raise RuntimeError(f"train pool downloaded from HF is 0 bytes: {rel} → {dest}")
    n_rows = sum(1 for line in dest.read_text().splitlines() if line.strip())
    log.info("[phase=fetch_pool] %s → %s (%d bytes, %d rows)", rel, dest, size, n_rows)
    return dest


def _resolve_train_pool(
    cell: str,
    seed: int,
    train_jsonl: Path,
    *,
    from_hf: bool,
    build_pool,
    run_dir: Path,
) -> None:
    """Land the per-cell train pool at ``train_jsonl``.

    ``from_hf=True`` (the #534 default): download the #530 bytes from HF
    (authoritative) AND run ``build_pool`` into a sibling file as a
    byte-compare DIAGNOSTIC (WARN-only). ``from_hf=False``: legacy #530
    behavior — ``build_pool`` writes ``train_jsonl`` directly.
    """
    if not from_hf:
        log.info("[phase=build_%s] building training data via build_cell_504", cell)
        build_pool(train_jsonl)
        return
    _fetch_train_pool_from_hf(cell, seed, train_jsonl)
    rebuilt = run_dir / "train_pool_rebuilt.jsonl"
    log.info("[phase=build_%s] diagnostic rebuild via build_cell_504", cell)
    build_pool(rebuilt)
    if rebuilt.read_bytes() == train_jsonl.read_bytes():
        log.info("[phase=build_%s] byte-compare PASS: rebuild == HF pool", cell)
    else:
        log.warning(
            "[phase=build_%s] byte-compare MISMATCH: build_cell_504 rebuild differs "
            "from the HF #530 pool — the HF bytes stay authoritative for training; "
            "rebuild kept at %s for diagnosis.",
            cell,
            rebuilt,
        )


def _load_eval_only_index(
    cell: str,
    run_slug: str,
    ckpt_index_path: Path,
    manifest_out: Path,
) -> dict:
    """Validate + load the prior full run's artifacts for ``--eval-only``.

    Requires ``checkpoint_index.json`` (the realized-fraction → snapshot-dir
    map), ``fraction_manifest.json`` (the eval guard's source ΔG
    expectations), and every indexed adapter's ``adapter_model.safetensors``
    on disk. Fails loud on anything missing — NO silent fallback to
    retraining. Returns the parsed checkpoint index.
    """
    if not ckpt_index_path.exists():
        raise RuntimeError(
            f"--eval-only requires an existing {ckpt_index_path} from a prior full "
            "run — run i534_run_cell.py WITHOUT --eval-only first."
        )
    if not manifest_out.exists():
        raise RuntimeError(
            f"--eval-only requires an existing {manifest_out} — the eval's "
            "source-manifest guard reads the selector's source ΔG expectations "
            "from it. Re-run the full cell if the manifest is gone."
        )
    ckpt_index = json.loads(ckpt_index_path.read_text())
    for frac_str, entry in sorted(ckpt_index.items(), key=lambda kv: float(kv[0])):
        snap = Path(entry["path"])
        if not (snap / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"--eval-only: indexed adapter for frac={frac_str} missing on disk at "
                f"{snap} — the snapshot was deleted/moved; re-fetch it from HF "
                f"(adapters/issue_534/{run_slug}/ckpt_frac{float(frac_str):.2f}) or "
                "re-run the full cell."
            )
    log.info("[phase=eval_only_%s] index verified: %s", cell, ckpt_index)
    return ckpt_index


def _assert_trajectory_complete(
    cell: str,
    out_traj: Path,
    n_expected: int,
) -> None:
    """Post-eval completeness check: trajectory.json exists with one entry per fraction.

    Raises RuntimeError on a missing file (silent eval failure,
    feedback_eval_script_silent_not_present_misdiagnosis) or a checkpoint
    count mismatch.
    """
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_534"))
    ap.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_534"))
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
            "#534 reuses #530's committed eval_results/issue_530/phase0_5_gates.json "
            "as-is (plan §4.2 — NO Phase 0 re-run)."
        ),
    )
    ap.add_argument("--chosen-rank", type=int, default=8, help="LoRA rank (VERBATIM #530).")
    ap.add_argument("--chosen-alpha", type=int, default=32, help="LoRA α (VERBATIM #530).")
    ap.add_argument(
        "--lr",
        type=float,
        default=LR_DEFAULT_534,
        help=f"Marker-only learning rate (default {LR_DEFAULT_534:g}; VERBATIM #530).",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS_DEFAULT_534,
        help=f"Epoch ceiling (default {EPOCHS_DEFAULT_534}; VERBATIM #530 — band-stop decides).",
    )
    ap.add_argument(
        "--snapshot-every-steps",
        type=int,
        default=SNAPSHOT_EVERY_STEPS_DEFAULT,
        help=(
            f"Per-step adapter snapshot cadence inside MarkerBandStopCallback "
            f"(default {SNAPSHOT_EVERY_STEPS_DEFAULT}; THE manipulated variable, plan §4.1)."
        ),
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
            "plan §4.3 d). Selected AFTER training by i534_select_fractions.py."
        ),
    )
    ap.add_argument(
        "--train-pool-from-hf",
        dest="train_pool_from_hf",
        action="store_true",
        default=True,
        help=(
            "Download the per-cell train pool byte-for-byte from the #530 HF "
            "artifact (default ON; data-identity with the parent)."
        ),
    )
    ap.add_argument(
        "--no-train-pool-from-hf",
        dest="train_pool_from_hf",
        action="store_false",
        help="Rebuild the pool via build_cell_504 instead (legacy #530 path).",
    )
    ap.add_argument(
        "--skip-source-trajectory",
        action="store_true",
        help="Forwarded to i534_select_fractions.py (descope ladder item 1).",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "#534 round-2 re-run path: SKIP train pool, training, and fraction "
            "selection; consume the EXISTING checkpoint_index.json + "
            "fraction_manifest.json from a prior full run and re-run only the "
            "trajectory eval + sentinel (the round-1 snapshots' weights are "
            "valid — NO retraining). Fails loud when the index/manifest/"
            "snapshot dirs are missing."
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
        help="Skip DV-B KL (also skips z_marker capture — NOT for production cells).",
    )
    ap.add_argument("--report-to", default="wandb")
    ap.add_argument(
        "--hf-path-suffix",
        default="",
        help="Round-collision-avoidance suffix on the HF subfolder + runs subdir.",
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
            f"%(asctime)s [phase=cell_{args.cell}_seed{args.seed}] "
            f"%(name)s %(levelname)s | %(message)s"
        ),
        stream=sys.stdout,
    )
    log.info(
        "Assigned physical GPU --gpu-id=%d; inherited CUDA_VISIBLE_DEVICES=%s "
        "(train/sft.py will SET CVD=str(gpu_id)).",
        args.gpu_id,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )

    # ── #534: in-train per-fraction HF persistence is DISABLED by design
    # (plan §4.2 divergence 4) — the post-hoc selector owns all fraction
    # uploads. Pop the env vars defensively in case the launching shell
    # carried them over from a #530-era environment.
    for var in (
        "EPM_PERSIST_TRAJECTORY_HF_REPO",
        "EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER",
        "EPM_PERSIST_ADAPTER_HF_REPO",
        "EPM_PERSIST_ADAPTER_HF_SUBFOLDER",
    ):
        if os.environ.pop(var, None) is not None:
            log.warning(
                "[phase=env] popped %s — #534 disables in-train per-fraction "
                "persistence (the selector owns fraction uploads).",
                var,
            )

    # Carry-over data dependencies from #472 (persona bank, centroids,
    # on-policy R) — idempotent HF pull, same as #530.
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

    # #534 namespace: adapters under adapters/issue_534/<slug>_seed<S>.
    run_slug = f"{args.cell}_seed{args.seed}{args.hf_path_suffix}"
    run_dir = args.runs_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = run_dir / "train_pool.jsonl"
    final_adapter_dir = run_dir / "adapter"
    ckpt_root = run_dir / "checkpoints"
    snapshot_dir = run_dir / "snapshots"
    ckpt_index_path = run_dir / "checkpoint_index.json"
    cell_slab_dir = args.slab_root / f"{args.cell}_seed{args.seed}"
    out_traj = cell_slab_dir / "trajectory.json"
    manifest_out = cell_slab_dir / "fraction_manifest.json"
    source_traj_out = cell_slab_dir / "source_steps_trajectory.json"
    sentinel = args.log_dir / f"issue-534-{args.cell}-seed{args.seed}-results.json"

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

    # Legacy CheckpointAtFractionsCallback fractions: kept attached, UNCHANGED
    # (writes nothing before 0.25*max_steps=75 at the realized stop ~20; the
    # selector overwrites checkpoint_index.json afterward). Parse the same
    # fraction set so the legacy index keys match the selector's.
    parsed_fracs = tuple(sorted(float(x.strip()) for x in args.fractions.split(",") if x.strip()))
    if not parsed_fracs or any(f <= 0 or f > 1.0 for f in parsed_fracs):
        raise ValueError(f"--fractions {args.fractions!r} must be floats in (0, 1].")

    if args.eval_only:
        # ── #534 round-2 eval-only re-run: NO retraining, NO re-selection. ────
        # Consume the prior full run's checkpoint_index.json (the 4 selected
        # snapshot dirs survived the selector's cleanup) + fraction_manifest
        # (the eval's adapter-applied cross-check needs its source ΔG
        # expectations). Fail loud on anything missing.
        log.info(
            "[phase=eval_only_%s] consuming existing index %s + manifest %s",
            args.cell,
            ckpt_index_path,
            manifest_out,
        )
        ckpt_index = _load_eval_only_index(args.cell, run_slug, ckpt_index_path, manifest_out)
        # The train path pins CUDA_VISIBLE_DEVICES inside train/sft.py (the
        # +gpu_id clobber gotcha); eval-only never trains, so pin it here or
        # the nested vLLM eval lands on the wrong physical GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    else:
        # ── Phase: train pool (HF bytes authoritative; rebuild = diagnostic). ─
        def _build_pool(dest: Path) -> None:
            build_cell_504(
                args.cell,
                dest,
                r_train=r_train,
                arm_to_positioned_n=arm_to_positioned_n,
                q_train=q_train,
                persona_bank=bank,
                source=effective_source,
                marker_text=MARKER_TEXT,
                smoke_mid_band_n=smoke_mid_band_n,
                seed=args.seed,
            )

        _resolve_train_pool(
            args.cell,
            args.seed,
            train_jsonl,
            from_hf=args.train_pool_from_hf,
            build_pool=_build_pool,
            run_dir=run_dir,
        )

        # ── Snapshot dir: WIPE at train start (stale-attempt guard). ──────────
        if snapshot_dir.exists():
            log.warning("[phase=train_%s] wiping stale snapshot dir %s", args.cell, snapshot_dir)
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "[phase=train_%s] training (rank=%d, alpha=%d, lr=%g, epochs=%d, "
            "snapshot_every=%d, snapshot_cap=%d, suppress_at_post_response_slot=%s)",
            args.cell,
            args.chosen_rank,
            args.chosen_alpha,
            effective_lr,
            effective_epochs,
            args.snapshot_every_steps,
            args.snapshot_max_count,
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
            hf_path_in_repo_override=f"adapters/issue_534/{run_slug}",
            run_name_override=f"issue534_{run_slug}{wandb_suffix}",
            marker_suppress_at_post_response_slot=MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
            marker_im_end_token_id=MARKER_IM_END_TOKEN_ID,
            # #534 — THE manipulated variable: per-step sub-stop snapshots.
            marker_band_snapshot_every_steps=args.snapshot_every_steps,
            marker_band_snapshot_dir=snapshot_dir,
            marker_band_snapshot_max_count=args.snapshot_max_count,
        )
        log.info("[phase=train_%s] done; snapshots under %s", args.cell, snapshot_dir)

        # Free in-process LoRA-training GPU memory before the selector + eval.
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Phase: post-hoc fraction selection (NESTED subprocess: the 7B HF
        # source-trajectory load stays memory-isolated from this parent, same
        # isolation discipline as the vLLM eval below). Overwrites
        # checkpoint_index.json with the realized-fraction index. ─────────────
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
            f"adapters/issue_534/{run_slug}",
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
        # #534 reuses the #504 eval rig byte-for-byte (adapter-applied guard +
        # per-batch byte-identical guard); it iterates EVERY index entry, so
        # the 4 realized fractions ride in one vLLM session.
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
        # #534 round-2: arm the adapter-applied cross-check — the eval's own
        # source-self ΔG must agree with the selector manifest's teacher-forced
        # read at the final fraction (>2 nat disagreement = fail loud).
        "--fraction-manifest",
        str(manifest_out),
    ]
    if args.no_kl:
        eval_cmd.append("--no-kl")
        log.warning(
            "[phase=eval_%s] --no-kl set: the HF KL phase is where z_marker slot "
            "stats are captured — production #534 cells must NOT use this.",
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

    # Full mode: the selector wrote one index entry per requested fraction.
    # Eval-only mode: the EXISTING index is authoritative (its key count).
    n_expected = len(ckpt_index) if args.eval_only else len(parsed_fracs)
    _assert_trajectory_complete(args.cell, out_traj, n_expected)

    _write_sentinel(
        sentinel,
        kind="epm:progress",
        phase=f"cell_done_{args.cell}_seed{args.seed}",
        note={
            "cell": args.cell,
            "seed": args.seed,
            "trajectory_path": str(out_traj),
            "fraction_manifest_path": str(manifest_out),
            "adapter_hf_path": f"adapters/issue_534/{run_slug}",
            "checkpoint_index": ckpt_index,
            "n_held_out_panel": len(held_out_panel),
            "lr": effective_lr,
            "epochs": effective_epochs,
            "snapshot_every_steps": args.snapshot_every_steps,
            "fractions": list(parsed_fracs),
            "eval_only": bool(args.eval_only),
        },
    )
    log.info("[phase=done] wrote sentinel → %s", sentinel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
