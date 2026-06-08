#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker token " ※" are intentional
"""Task #514 — unified dispatcher (fork of dispatch_508.py).

The dispatcher IS the smoke runner: ``--cells ft_dense_b30 --seeds 42`` trains
ONE cell end-to-end on the EXACT same code path as the full 6-cell sweep
(``--cells ft_dense_b30,ft_dense_b35,...,ft_lowlr_b100``). Smoke and sweep share
a single dispatcher invocation; this is the smoke/sweep architectural-parity
gate PASS_UNIFIED verdict (CLAUDE.md /issue Step 6d.0).

Delta vs #508's dispatcher (plan §4.1):

1. **Per-cell LR override** plumbed through ``train_one_cell_fullft(lr_override=...)``.
   #508's dispatcher accepted a SINGLE global ``--ft-lr-override`` (replicated
   across all FT cells). #514 reads the per-cell LR from
   ``full_ft_regime_514.CELL_SPECS_514`` (each cell tuple has its own
   ``lr_override``: dense cells at 5e-6, lower-LR cells at 2e-6).

2. **--abort-on-collapse flag (default ON)**. After cell 1 of each lever (the
   smallest budget in that lever) completes Phase 2 eval, the dispatcher reads
   the just-written eval JSON and applies
   ``full_ft_regime_514.abort_logic.should_abort_lever``. If it decides ABORT,
   the dispatcher (a) writes a sentinel JSON for that lever to
   ``/workspace/logs/issue-514-lever-aborted-<lever>.json`` (the orchestrator's
   ``poll_pipeline.py`` drains it; pod-side code MUST NOT shell out to
   ``task.py post-marker``), and (b) skips the remaining cells of THAT lever.
   The other lever still runs.

3. **All cells are FT.** No LoRA arm in this dispatcher (the LoRA reference
   curve is re-used verbatim from #508; see ``plot_issue_514.py``).

4. **Output root** ``/workspace/issue_514`` and eval JSONs land at
   ``eval_results/issue_514/<slug>_seed42.json`` per plan §10.

5. **Marker token assertion** at trainer launch is a hard
   ``RuntimeError`` (per ``.claude/rules/marker-leakage-measurement.md``): a
   wrong tokenizer revision yielding a marker token id != 83399 fails fast
   BEFORE any GPU minutes are spent.

Phases per cell (mirrors #508):
    Phase 0:  Build per-cell training JSONLs (CPU-only). Shared across cells.
    Phase 1:  Train cell via ``accelerate launch`` subprocess + ZeRO-3 on 4 GPUs.
              ``train_one_cell_fullft(lr_override=<per-cell>)``.
    Phase 2:  vLLM batched eval. 15 personas × 20 questions held-out + 20 source
              + 20 default-assistant probes.
    Phase 3 (post-sweep, --do-analyze): bracketing check + cluster bootstrap +
              hero figure + trajectory figures.

Resume-safe by per-phase output presence: if the eval JSON exists for a cell,
the dispatcher skips the train + eval and moves on (CLAUDE.md "Checkpoint per
phase"). The post-train teardown deletes the merged FT checkpoint dir after
eval per ``.claude/rules/upload-policy.md``.

Usage on the pod (after ``pod.py provision --issue 514 --intent ft-7b``):

    # Smoke (one cell of the dense lever).
    nohup uv run python scripts/dispatch_514.py \\
        --cells ft_dense_b30 \\
        --seeds 42 \\
        --output-root /workspace/issue_514_smoke \\
        --build-data \\
        > /workspace/logs/issue-514-smoke.log 2>&1 &

    # Full sweep (all 6 cells).
    nohup uv run python scripts/dispatch_514.py \\
        --cells ft_dense_b30,ft_dense_b35,ft_dense_b40,ft_dense_b45,ft_lowlr_b50,ft_lowlr_b100 \\
        --seeds 42 \\
        --output-root /workspace/issue_514 \\
        --build-data \\
        --abort-on-collapse \\
        --do-analyze \\
        > /workspace/logs/issue-514.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import time as _time

from dotenv import load_dotenv

# Load .env at module import so subprocess env-passthrough (env=env in
# accelerate launch + os.environ-copy in train subprocesses) sees the
# pod's HF_TOKEN / WANDB_API_KEY when this script is the entry point.
# CLAUDE.md: "load_dotenv() at module-top is required when this file
# contains subprocess.<func>" — even though the subprocess.<...> calls
# live in train_cell_fullft.py (imported below), the dispatcher is the
# ROOT process whose os.environ snapshot becomes the child env. Without
# the load-at-entry, a fresh dispatcher process spawns subprocesses with
# the credential env missing (#397 round-10' incident).
load_dotenv()

LOG = logging.getLogger("issue_514.dispatch")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task #514 unified dispatcher (full-FT regime sweep)")
    p.add_argument(
        "--cells",
        default="ft_dense_b30",
        help=(
            "Comma-separated #514 cell slugs. "
            "Smoke=ft_dense_b30 (single cell); "
            "sweep=ft_dense_b30,ft_dense_b35,ft_dense_b40,ft_dense_b45,"
            "ft_lowlr_b50,ft_lowlr_b100"
        ),
    )
    p.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seeds (default single seed 42 per plan §11)",
    )
    p.add_argument(
        "--output-root",
        default="/workspace/issue_514",
        type=Path,
        help="Root for per-cell artifacts (training data, checkpoints, eval JSON).",
    )
    p.add_argument(
        "--build-data",
        action="store_true",
        help="If set, re-build the per-cell training JSONLs + dynamics probes.",
    )
    p.add_argument(
        "--build-only",
        action="store_true",
        help="Stop after Phase 0 (data build); useful for CPU-only smoke.",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip Phase 1 (training). Assumes checkpoints already exist.",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip Phase 2 (eval). Useful for CPU-only data+launch smoke.",
    )
    p.add_argument(
        "--do-analyze",
        action="store_true",
        help="After all cells finish, run Phase 3 analysis.",
    )
    p.add_argument(
        "--num-gpus-fullft",
        type=int,
        default=4,
        help="GPUs for ZeRO-3 full-FT (default 4).",
    )
    abort_group = p.add_mutually_exclusive_group()
    abort_group.add_argument(
        "--abort-on-collapse",
        dest="abort_on_collapse",
        action="store_true",
        default=True,
        help=(
            "After cell 1 of each lever finishes Phase 2 eval, abort the rest "
            "of THAT lever iff source r-collapse rate >= 0.50 AND held-out "
            "g_logprob_mean > -5.0. Default ON (plan §4.1.3). The OTHER "
            "lever's cells still run."
        ),
    )
    abort_group.add_argument(
        "--no-abort-on-collapse",
        dest="abort_on_collapse",
        action="store_false",
        help="Disable the per-lever early-exit (run all 6 cells regardless).",
    )
    p.add_argument(
        "--no-dynamics",
        action="store_true",
        help="Skip the offline post-checkpoint dynamics extraction (faster smoke).",
    )
    p.add_argument(
        "--smoke-tiny-n",
        action="store_true",
        help=(
            "Smoke flag: build the training JSONL with a tiny per-class count "
            "(5 pos + 5 neg/persona) for CPU-side end-to-end wiring tests. "
            "Does NOT change training hyperparameters (epoch_fraction, lr); "
            "only the dataset size."
        ),
    )
    return p.parse_args()


# Candidate paths for the inherited R_train.json (B2 round-2 fix). The plan §10
# pod-side launch uses ``hf download ... --local-dir /workspace/data`` which
# lands the file at ``/workspace/data/issue472_neg_geometry/on_policy_R/
# R_train.json`` (FLAT_HF_DOWNLOAD); the project tree on the VM keeps a local
# symlink under ``data/issue_472/on_policy_R/R_train.json`` (PROJECT_TREE).
# Try both; fail loud with every attempted path if both miss. Order: the
# pod-side flat path wins on a fresh pod, then the worktree-local symlink for
# offline / VM smoke. NEVER silent-default.
R_TRAIN_CANDIDATE_PATHS: tuple[Path, ...] = (
    Path("/workspace/data/issue472_neg_geometry/on_policy_R/R_train.json"),
    Path("data/issue472_neg_geometry/on_policy_R/R_train.json"),
    Path("data/issue_472/on_policy_R/R_train.json"),
)


def _resolve_r_train_path() -> Path:
    """Resolve the inherited R_train.json from the first candidate path that exists.

    Plan §10 reproduce command pulls from HF Hub via
    ``hf download superkaiba1/explore-persona-space-data \
        --include "issue472_neg_geometry/on_policy_R/R_train.json" \
        --local-dir /workspace/data``
    which writes to ``/workspace/data/issue472_neg_geometry/on_policy_R/R_train.json``.
    The worktree on the dev VM keeps a local symlink at
    ``data/issue_472/on_policy_R/R_train.json`` for offline smoke. Returns
    the first hit; raises ``FileNotFoundError`` with ALL attempted paths if
    none exist (CLAUDE.md fail-fast — NEVER paper over with a silent default).
    """
    for p in R_TRAIN_CANDIDATE_PATHS:
        if p.exists():
            LOG.info("[R_train] resolved to %s", p)
            return p
    r_gen_mod = "explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate"
    raise FileNotFoundError(
        "Required R_train.json not found at any expected path:\n  - "
        + "\n  - ".join(str(p) for p in R_TRAIN_CANDIDATE_PATHS)
        + f"\nFix: run plan §10 pod-side launch (`hf download` step) OR `python -m {r_gen_mod}`. "
        "Refusing to proceed with a silent default per CLAUDE.md fail-fast."
    )


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(  # epm-lint: subprocess-env-inherit -- git probe
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                env={**os.environ},
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


# ── Phase 0: data build ──────────────────────────────────────────────────────


def phase0_build_data(
    output_root: Path,
    cells: list[str],
    seeds: list[int],
    *,
    build_data: bool,
    smoke_tiny_n: bool = False,
) -> dict[str, Path]:
    """Phase 0 — build per-cell training JSONLs (CPU-only).

    #514 inherits the canonical contrastive recipe from #508 (200 positive +
    800 negative = 1000 rows). Single JSONL is byte-identical across all 6
    cells — only the training hyperparameters differ. Reuses #508's
    ``_build_canonical_training_jsonl`` via the dispatcher.

    Returns ``{cell_slug: train_jsonl_path}``.
    """
    # Import here so module import doesn't pull in HF transformers et al.
    # Reuse #508's training-data builder verbatim — single-variable rule says
    # the contrastive recipe stays byte-identical with #508. ``scripts/`` is
    # not a Python package, so load dispatch_508 by file path via importlib.
    import importlib.util
    from pathlib import Path as _Path

    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )
    from explore_persona_space.experiments.full_ft_regime_514 import (
        CONTRASTIVE_NEGATIVES,
        NEG_EX_PER_PERSONA,
        POS_EX_PER_SOURCE,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.lora_vs_ft_508 import (
        load_q_eval,
        load_q_train,
    )

    _disp508_path = _Path(__file__).parent / "dispatch_508.py"
    _spec = importlib.util.spec_from_file_location("_dispatch_508", _disp508_path)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"Could not load dispatch_508.py from {_disp508_path}")
    _disp508 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_disp508)
    _build_canonical_training_jsonl = _disp508._build_canonical_training_jsonl

    output_root.mkdir(parents=True, exist_ok=True)
    train_dir = output_root / "training"
    train_dir.mkdir(parents=True, exist_ok=True)

    persona_bank = dict(EVAL_PERSONAS_24)
    q_train = load_q_train()
    _ = load_q_eval()  # validate the eval split is reachable at build time.

    pos_ex = 5 if smoke_tiny_n else POS_EX_PER_SOURCE
    neg_per = 5 if smoke_tiny_n else NEG_EX_PER_PERSONA
    if smoke_tiny_n:
        LOG.info("[smoke-tiny-n] Building tiny dataset: %d pos + %d neg/persona", pos_ex, neg_per)

    # The training mix is shared across all #514 cells (single-variable rule —
    # only the FT regime moves).
    canonical_train = train_dir / "contrastive_recipe.jsonl"
    if build_data or not canonical_train.exists():
        LOG.info("[phase=0_build_data] Building canonical training JSONL")
        r_train_path = _resolve_r_train_path()

        from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
            load_r_artifact,
        )

        r_train = load_r_artifact(r_train_path)
        _build_canonical_training_jsonl(
            output_path=canonical_train,
            r_train=r_train,
            persona_bank=persona_bank,
            q_train=q_train,
            source=SOURCE_PERSONA,
            negatives=CONTRASTIVE_NEGATIVES,
            pos_ex=pos_ex,
            neg_ex_per_persona=neg_per,
            seed=42,
        )

    cell_to_jsonl: dict[str, Path] = {}
    for cell in cells:
        cell_to_jsonl[cell] = canonical_train
    LOG.info("[phase=0_build_data done] %d cells → %s", len(cells), canonical_train)
    print(f"[phase=0_build_data_done n_cells={len(cells)}]", flush=True)
    return cell_to_jsonl


# ── Phase 1 + 2: per-cell train + eval ───────────────────────────────────────


def _extract_fullft_dynamics_for_cell(
    *,
    cell_slug: str,
    checkpoint_index: dict[str, dict],
    base_model: str,
    dynamics_probes: Path,
    sidecar_path: Path,
) -> Path | None:
    """Offline post-checkpoint dynamics extraction (B3 round-2 fix).

    Loads the dynamics-probes JSON + tokenizer, then defers to
    ``extract_fullft_dynamics_from_checkpoints`` (the canonical extractor in
    ``marker_dynamics_callback``) which iterates every saved FT checkpoint
    in ``checkpoint_index``, runs the 20-probe pass per checkpoint, and writes
    the aggregated snapshot dict to ``sidecar_path``. Returns the sidecar path.

    Pre-condition: this MUST run BEFORE the ``_fractions/`` cleanup (B1) — the
    extractor reads every fraction's saved checkpoint, so deleting them first
    discards all but the endpoint.

    The FT trainer (``scripts/train_marker_fullft.py``) deliberately skips
    in-training dynamics extraction to avoid the ZeRO-3 collective deadlock
    (rank 0 alone calling ``model.generate()`` while other ranks are still in
    the training loop hangs the world). The dispatcher is the right place for
    this single-process post-train extraction.
    """
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.lora_vs_ft_508.marker_dynamics_callback import (
        extract_fullft_dynamics_from_checkpoints,
        load_dynamics_probes,
    )

    LOG.info(
        "[%s] extract_fullft_dynamics: %d checkpoints → %s",
        cell_slug,
        len(checkpoint_index),
        sidecar_path,
    )
    probes = load_dynamics_probes(dynamics_probes)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return extract_fullft_dynamics_from_checkpoints(
        checkpoint_index=checkpoint_index,
        base_model_path=base_model,
        tokenizer=tokenizer,
        probes=probes,
        output_path=sidecar_path,
    )


def _maybe_cleanup_fraction_checkpoints(ckpt_root: Path) -> None:
    """Delete the ``_fractions/`` directory after dynamics extraction (B1 round-2 fix).

    Per #508's pattern + ``.claude/rules/upload-policy.md``: the per-cell
    ``<cell_dir>_fractions/`` dir holds 4 intermediate FT checkpoints
    (~14 GB each = ~56 GB / cell × 6 cells = ~336 GB) which would blow the
    RunPod MooseFS 130 GB per-pod quota mid-run.

    The dynamics extractor (``_extract_fullft_dynamics_for_cell``) writes the
    aggregated trajectory snapshot to ``<cell_dir>/dynamics.json``, which is
    the durable artifact; the saved intermediate checkpoints are throwaway.
    Called AFTER dynamics extraction completes, BEFORE Phase 2 eval starts.
    """
    if not ckpt_root.exists():
        return
    LOG.info("[cleanup] removing _fractions checkpoint dir: %s", ckpt_root)
    shutil.rmtree(ckpt_root, ignore_errors=True)


def phase1_train_cell(
    *,
    cell_slug: str,
    epoch_fraction: float,
    lr_override: float,
    seed: int,
    train_jsonl: Path,
    output_root: Path,
    base_model: str,
    wandb_project: str,
    num_gpus_fullft: int,
    dynamics_probes: Path | None,
) -> dict:
    """Train one FT cell via ``train_one_cell_fullft`` with the per-cell LR.

    Post-training (B1 + B3 round-2 fix):
        1. Run offline dynamics extraction over the saved fraction checkpoints
           (writes ``<cell_dir>/dynamics.json`` — durable trajectory artifact).
        2. Delete the ``<cell_dir>_fractions/`` directory (MooseFS quota).
    Both happen inside this function so Phase 2 eval observes only the
    endpoint merged dir + the durable ``dynamics.json`` sidecar.
    """
    from explore_persona_space.experiments.lora_vs_ft_508.train_cell_fullft import (
        train_one_cell_fullft,
    )

    cell_dir = output_root / "checkpoints" / f"{cell_slug}_seed{seed}"
    ckpt_root = output_root / "checkpoints" / f"{cell_slug}_seed{seed}_fractions"
    dynamics_sidecar = cell_dir / "dynamics.json"

    LOG.info(
        "[phase=1_train] cell=%s arm=fullft epoch_fraction=%s lr=%.1e seed=%d",
        cell_slug,
        epoch_fraction,
        lr_override,
        seed,
    )
    print(f"[phase=1_train cell={cell_slug} arm=fullft]", flush=True)

    # R2.3-style multi-snapshot checkpoint cadence (inherited from #508).
    ft_ckpt_fractions = (0.25, 0.5, 0.75, 1.0)

    result = train_one_cell_fullft(
        cell_slug=cell_slug,
        seed=seed,
        train_jsonl=train_jsonl,
        output_dir=cell_dir,
        ckpt_root=ckpt_root,
        epoch_fraction=epoch_fraction,
        base_model=base_model,
        wandb_project=wandb_project,
        dynamics_probes=dynamics_probes,  # ignored on full-FT path (ZeRO-3 deadlock).
        lr_override=lr_override,
        num_gpus=num_gpus_fullft,
        ckpt_fractions=ft_ckpt_fractions,
    )

    # B3 round-2 fix: offline dynamics extraction MUST happen BEFORE B1 cleanup
    # (extractor reads every fraction's saved checkpoint).
    checkpoint_index = result.get("checkpoint_index", {})
    dynamics_path_str: str | None = None
    if dynamics_probes is not None and checkpoint_index:
        try:
            _extract_fullft_dynamics_for_cell(
                cell_slug=cell_slug,
                checkpoint_index=checkpoint_index,
                base_model=base_model,
                dynamics_probes=dynamics_probes,
                sidecar_path=dynamics_sidecar,
            )
            if dynamics_sidecar.exists():
                dynamics_path_str = str(dynamics_sidecar)
        except Exception as e:
            # Don't lose the cell over a dynamics-extractor crash; the headline
            # endpoint eval can still run. Log loud + surface in return dict.
            LOG.error(
                "[%s] extract_fullft_dynamics FAILED: %s — proceeding without "
                "FT trajectory for this cell",
                cell_slug,
                e,
            )

    # B1 round-2 fix: now that dynamics.json is written, delete _fractions/
    # immediately to free ~56 GB before the NEXT cell trains. Without this the
    # 6-cell sweep blows the 130 GB MooseFS quota mid-run.
    _maybe_cleanup_fraction_checkpoints(ckpt_root)

    return {
        "output_dir": str(cell_dir),
        "checkpoint_index": checkpoint_index,
        "arm": "fullft",
        "dynamics_snapshots_path": dynamics_path_str,
    }


def phase2_eval_cell(
    *,
    cell_slug: str,
    seed: int,
    output_root: Path,
    base_model: str,
    dynamics_snapshots_path: str | None = None,
) -> Path:
    """Run vLLM batched eval on a trained FT cell. Forwards to #508's eval_one_cell.

    The signature mirrors #508's ``phase2_eval_cell`` (FT-only path) — we pass
    ``is_full_ft=True``, ``lora_adapter_path=None``, ``full_ft_checkpoint_dir``,
    and let eval_one_cell's defaults populate the eval panel
    (persona_bank, eval_questions, held_out_personas, source_persona,
    eval_source, eval_qwen_default).

    Post-eval (B3 round-2 fix): when ``dynamics_snapshots_path`` is set (i.e.
    the offline dynamics extractor wrote ``<cell_dir>/dynamics.json`` in
    phase 1), stamp it into the eval JSON so the analyzer's
    ``_gather_dynamics_snapshots`` locates the trajectory data per cell.
    """
    from explore_persona_space.experiments.lora_vs_ft_508.eval_one_cell import (
        eval_one_cell,
    )

    cell_dir = output_root / "checkpoints" / f"{cell_slug}_seed{seed}"
    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_json = eval_dir / f"{cell_slug}_seed{seed}.json"

    LOG.info(
        "[phase=2_eval] cell=%s arm=fullft seed=%d → %s",
        cell_slug,
        seed,
        eval_json,
    )
    print(f"[phase=2_eval cell={cell_slug} arm=fullft]", flush=True)

    eval_one_cell(
        cell_slug=cell_slug,
        arm="fullft",
        seed=seed,
        output_path=eval_json,
        is_full_ft=True,
        lora_adapter_path=None,
        full_ft_checkpoint_dir=cell_dir,
        base_model=base_model,
    )

    # B3 round-2 fix: stamp dynamics_snapshots_path into the eval JSON. The
    # analyzer's _gather_dynamics_snapshots reads eval_json["dynamics_snapshots_path"]
    # to locate per-cell trajectory data; without this the FT trajectory figure
    # degenerates to endpoint-only.
    if dynamics_snapshots_path:
        try:
            payload = json.loads(eval_json.read_text())
            payload["dynamics_snapshots_path"] = dynamics_snapshots_path
            eval_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            LOG.info(
                "[%s] eval JSON updated with dynamics_snapshots_path=%s",
                cell_slug,
                dynamics_snapshots_path,
            )
        except (OSError, json.JSONDecodeError) as e:
            LOG.warning(
                "[%s] could not stamp dynamics_snapshots_path into eval JSON: %s",
                cell_slug,
                e,
            )
    return eval_json


def _maybe_cleanup_fullft_checkpoint(cell_dir: Path) -> None:
    """Delete merged FT checkpoint after eval (RunPod MooseFS 130 GB quota).

    Plan §10 + ``.claude/rules/upload-policy.md``: do NOT upload the merged
    dir to the shared HF model repo (re-derivable from training data +
    commit + seed). Store only the eval JSON.
    """
    if not cell_dir.exists():
        return
    LOG.info("[cleanup] removing full-FT merged checkpoint: %s", cell_dir)
    shutil.rmtree(cell_dir, ignore_errors=True)


# ── Per-lever abort-on-collapse + sentinels ──────────────────────────────────


def _write_lever_aborted_sentinel(
    lever: str,
    *,
    triggering_cell: str,
    diagnostics: dict,
    sentinel_dir: Path,
) -> Path | None:
    """Write the per-lever abort sentinel to the poll_pipeline-scanned dir.

    Pod-side code MUST NOT shell out to ``scripts/task.py``; the sentinel
    file is the canonical channel (CLAUDE.md hard rule). The orchestrator's
    ``poll_pipeline.py`` reads any file matching
    ``/workspace/logs/issue-<N>-*.json`` and posts the corresponding marker
    on the VM side.
    """
    if not sentinel_dir.exists():
        LOG.warning(
            "[sentinel] %s does not exist; abort sentinel for lever=%s not written",
            sentinel_dir,
            lever,
        )
        return None
    ts = int(_time())
    sentinel_path = sentinel_dir / f"issue-514-epm_lever_aborted-{lever}-{ts}.json"
    sentinel = {
        "sentinel_schema_version": 1,
        "kind": "epm:514-lever-aborted",
        "version": 1,
        "task_id": 514,
        "by": "dispatch_514",
        "ts": _dt.datetime.now(_dt.UTC).isoformat(),
        "note": json.dumps(
            {
                "lever": lever,
                "triggering_cell": triggering_cell,
                "diagnostics": diagnostics,
            }
        ),
    }
    sentinel_path.write_text(json.dumps(sentinel))
    LOG.info("[sentinel] wrote lever-abort sentinel: %s", sentinel_path)
    return sentinel_path


# ── main dispatcher ──────────────────────────────────────────────────────────


def main() -> int:  # noqa: C901 - linear multi-phase dispatcher
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = parse_args()

    from explore_persona_space.experiments.full_ft_regime_514 import (
        BASE_MODEL,
        BUDGETS_DENSE_LEVER,
        BUDGETS_LOW_LR_LEVER,
        DENSE_LEVER_CELLS,
        EXPECTED_MARKER_TOKEN_ID,
        LOW_LR_LEVER_CELLS,
        MARKER_TEXT,
        WANDB_PROJECT_514,
        cell_lever,
        resolve_cell_spec,
    )
    from explore_persona_space.experiments.full_ft_regime_514.abort_logic import (
        should_abort_lever,
    )
    from explore_persona_space.experiments.lora_vs_ft_508 import (
        DYNAMICS_PROBES_PATH,
    )

    # Fail-fast marker token id assertion at dispatcher launch (rule:
    # marker-leakage-measurement.md). RuntimeError, not silent log, so a
    # wrong tokenizer revision crashes before training begins.
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
        if ids != [EXPECTED_MARKER_TOKEN_ID]:
            raise RuntimeError(
                f"Marker token id assertion FAILED at dispatcher launch: "
                f"tokenizer.encode({MARKER_TEXT!r}) = {ids}, "
                f"expected [{EXPECTED_MARKER_TOKEN_ID}]. "
                f"Wrong tokenizer revision — refusing to train (rule: "
                f"marker-leakage-measurement.md)."
            )
        LOG.info(
            "[dispatch] marker assert PASS (token=%r, id=%d)",
            MARKER_TEXT,
            EXPECTED_MARKER_TOKEN_ID,
        )
    except ImportError:
        LOG.warning(
            "[dispatch] transformers not importable; skipping marker-id assert "
            "(this is acceptable for CPU-side --build-only smoke; train phase will "
            "fail loud if the assert is missed)."
        )

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_514
    LOG.info("[dispatch] WANDB_PROJECT pinned to %s", WANDB_PROJECT_514)

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Validate every cell slug belongs to the #514 cell set + bind its spec
    # (arm, budget_label, epoch_fraction, lr_override).
    parsed_cells: list[tuple[str, str, float, float]] = []
    for cell in cells:
        if cell not in DENSE_LEVER_CELLS and cell not in LOW_LR_LEVER_CELLS:
            raise ValueError(
                f"Invalid #514 cell {cell!r}; valid: {DENSE_LEVER_CELLS + LOW_LR_LEVER_CELLS}"
            )
        spec = resolve_cell_spec(cell)
        # spec = (arm, budget_label, epoch_fraction, lr_override)
        parsed_cells.append((cell, spec[0], spec[2], spec[3]))

    LOG.info(
        "Dispatcher start: %d cells x %d seeds | output_root=%s | git=%s | "
        "abort_on_collapse=%s | timestamp=%s",
        len(parsed_cells),
        len(seeds),
        args.output_root,
        _git_commit(),
        args.abort_on_collapse,
        _dt.datetime.now(_dt.UTC).isoformat(),
    )
    LOG.info(
        "[dispatch] dense lever budgets=%s LR=5e-6; lower-LR lever budgets=%s LR=2e-6",
        BUDGETS_DENSE_LEVER,
        BUDGETS_LOW_LR_LEVER,
    )

    # ── Phase 0: build data (CPU-only). ──────────────────────────────────────
    args.output_root.mkdir(parents=True, exist_ok=True)
    cell_to_jsonl = phase0_build_data(
        args.output_root,
        cells,
        seeds,
        build_data=args.build_data,
        smoke_tiny_n=args.smoke_tiny_n,
    )
    if args.build_only:
        LOG.info("[dispatch] --build-only set; exiting after Phase 0.")
        print("[phase=done]", flush=True)
        return 0

    # ── Phase 1 + 2 per cell × seed. ─────────────────────────────────────────
    dynamics_probes = None if args.no_dynamics else Path(DYNAMICS_PROBES_PATH)

    # Per-lever "first cell finished" tracking for --abort-on-collapse. The
    # "first cell" is the SMALLEST budget in that lever (ft_dense_b30 for dense,
    # ft_lowlr_b50 for lowlr) — that's the cell whose collapse status determines
    # whether deeper budgets in the same lever should also collapse. (Deeper
    # budgets in a collapsed lever will be MORE collapsed, not less.)
    lever_aborted: dict[str, bool] = {"dense": False, "lowlr": False}
    lever_smallest_budget_cell: dict[str, str] = {
        "dense": DENSE_LEVER_CELLS[0],
        "lowlr": LOW_LR_LEVER_CELLS[0],
    }

    sentinel_dir = Path("/workspace/logs")
    cell_results: list[dict] = []

    for cell_slug, _arm, ef, lr in parsed_cells:
        lever = cell_lever(cell_slug)
        if lever_aborted[lever]:
            LOG.info(
                "[skip] lever=%s already aborted; skipping cell=%s",
                lever,
                cell_slug,
            )
            continue

        for seed in seeds:
            eval_dir = args.output_root / "eval"
            eval_json = eval_dir / f"{cell_slug}_seed{seed}.json"
            if eval_json.exists():
                LOG.info(
                    "[skip] eval already exists for %s/seed%d: %s",
                    cell_slug,
                    seed,
                    eval_json,
                )
                cell_results.append(
                    {
                        "cell": cell_slug,
                        "arm": "fullft",
                        "seed": seed,
                        "eval_json": str(eval_json),
                    }
                )
                # Even on resume, we still need to evaluate the abort condition
                # for the lever's smallest-budget cell (idempotent — same JSON,
                # same decision, but lets us short-circuit downstream cells on
                # a resumed run).
                if args.abort_on_collapse and cell_slug == lever_smallest_budget_cell[lever]:
                    eval_data = json.loads(eval_json.read_text())
                    abort, diag = should_abort_lever(eval_data)
                    LOG.info(
                        "[abort-on-collapse RESUME check] cell=%s lever=%s decision=%s reason=%s",
                        cell_slug,
                        lever,
                        "ABORT" if abort else "CONTINUE",
                        diag["reason"],
                    )
                    if abort:
                        lever_aborted[lever] = True
                        _write_lever_aborted_sentinel(
                            lever,
                            triggering_cell=cell_slug,
                            diagnostics=diag,
                            sentinel_dir=sentinel_dir,
                        )
                continue

            cell_dir = args.output_root / "checkpoints" / f"{cell_slug}_seed{seed}"
            dynamics_path: str | None = None
            if not args.skip_train and not cell_dir.exists():
                train_result = phase1_train_cell(
                    cell_slug=cell_slug,
                    epoch_fraction=ef,
                    lr_override=lr,
                    seed=seed,
                    train_jsonl=cell_to_jsonl[cell_slug],
                    output_root=args.output_root,
                    base_model=BASE_MODEL,
                    wandb_project=WANDB_PROJECT_514,
                    num_gpus_fullft=args.num_gpus_fullft,
                    dynamics_probes=dynamics_probes,
                )
                dynamics_path = train_result.get("dynamics_snapshots_path")
            elif args.skip_train:
                # If we skipped train, the dynamics.json sidecar may exist from
                # a prior run; pick it up so phase 2 still stamps it.
                sidecar = cell_dir / "dynamics.json"
                if sidecar.exists():
                    dynamics_path = str(sidecar)

            if not args.skip_eval:
                # Pre-eval cleanup — same shape as #508 dispatch (training
                # holds CUDA memory that survives the function return; vLLM
                # needs a contiguous free chunk to init).
                import gc as _gc

                _gc.collect()
                try:
                    import torch as _torch

                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except ImportError:
                    pass

                # vLLM v1 forks a worker subprocess; CUDA was initialized in the
                # main process by HF Trainer (phase 1), so the forked child
                # crashes with "Cannot re-initialize CUDA in forked subprocess".
                # Force spawn so the worker is a clean Python interpreter.
                os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

                phase2_eval_cell(
                    cell_slug=cell_slug,
                    seed=seed,
                    output_root=args.output_root,
                    base_model=BASE_MODEL,
                    dynamics_snapshots_path=dynamics_path,
                )
                cell_results.append(
                    {
                        "cell": cell_slug,
                        "arm": "fullft",
                        "seed": seed,
                        "eval_json": str(eval_json),
                    }
                )

                # ── Per-lever abort-on-collapse decision (plan §4.1.3). ──
                if args.abort_on_collapse and cell_slug == lever_smallest_budget_cell[lever]:
                    eval_data = json.loads(eval_json.read_text())
                    abort, diag = should_abort_lever(eval_data)
                    LOG.info(
                        "[abort-on-collapse check] cell=%s lever=%s decision=%s reason=%s",
                        cell_slug,
                        lever,
                        "ABORT" if abort else "CONTINUE",
                        diag["reason"],
                    )
                    if abort:
                        lever_aborted[lever] = True
                        _write_lever_aborted_sentinel(
                            lever,
                            triggering_cell=cell_slug,
                            diagnostics=diag,
                            sentinel_dir=sentinel_dir,
                        )
                        print(
                            f"[phase=lever_aborted lever={lever} triggering_cell={cell_slug}]",
                            flush=True,
                        )

                _maybe_cleanup_fullft_checkpoint(cell_dir)

    # ── Phase 3 (optional): analyze. ─────────────────────────────────────────
    if args.do_analyze:
        from explore_persona_space.experiments.full_ft_regime_514.analyze import (
            run_analysis_514,
        )

        eval_jsons = [Path(r["eval_json"]) for r in cell_results if Path(r["eval_json"]).exists()]
        analysis_out = args.output_root / "analysis"
        run_analysis_514(eval_jsons=eval_jsons, output_dir=analysis_out)

    # ── End-of-run sentinel (poll_pipeline contract). ────────────────────────
    if sentinel_dir.exists():
        sentinel_path = sentinel_dir / f"issue-514-epm_results-{int(_time())}.json"
        sentinel = {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 514,
            "by": "dispatch_514",
            "ts": _dt.datetime.now(_dt.UTC).isoformat(),
            "note": json.dumps(
                {
                    "cells": cell_results,
                    "output_root": str(args.output_root),
                    "lever_aborted": lever_aborted,
                }
            ),
        }
        sentinel_path.write_text(json.dumps(sentinel))
        LOG.info("[sentinel] wrote %s", sentinel_path)

    print("[phase=done]", flush=True)
    LOG.info(
        "[dispatch] complete: %d cell-evals, lever_aborted=%s",
        len(cell_results),
        lever_aborted,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
