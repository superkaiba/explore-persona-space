#!/usr/bin/env python3
"""Task #642 — coverage-matched-FT (cmft) dispatcher (phases 0-5).

PORT of ``origin/issue-606:scripts/issue_606/i606_dispatch.py`` adapted for the
#642 coverage-matched-FT arm (plan §4.3). The structural changes vs #606:

  - ``--arms`` gains a third arm token ``cmft`` (coverage-matched FT). #642's
    PRODUCTION run trains ``cmft`` ONLY — the ``lora`` and ``ft`` comparison
    poles are REUSED from #606's generations (plan §4.5), joined at analysis
    time by ``i642_analyze.py``. The ``lora`` / ``ft`` arm code paths are
    retained for completeness + the deferred-import verifier.
  - Phase 0 additionally downloads the #606 LoRA ``adapter_config.json`` from
    the MODEL repo @ ``MODEL_REVISION_DEFAULT`` (for the cmft module-set
    identity assert §4.2) and ENUMERATES + hard-asserts the reused #606 syco
    generation cells exist in the DATA repo @ ``DATA_REVISION_DEFAULT``
    BEFORE training — the FULL 39-persona panel per cell (LoRA {28,32,36,132}
    + FT {12,16,22,132} + base = 9 cells x 39 = 351 generation files + the
    stage-A trajectory), since the analyzer re-judges every (cell, persona);
    fail-loud if any is absent — the cmft join is impossible without them and a
    partial cell would crash the off-pod analysis after the GPU budget is spent.
  - ``phase1_train`` gains a ``cmft`` branch: the FT trainer is launched with
    ``--arm cmft --freeze-outside-lora-modules --lora-adapter-config-json
    <downloaded adapter_config>``; checkpoints land in ``cmft_ckpts``; WandB
    run name ``issue642_cmft_sycophancy_seed42``.
  - The end-of-run sentinel carries a ``reproducibility_card`` (the cmft arm
    trains adapters; CLAUDE.md training-task sentinel contract).

Smoke (plan §4.6 — same dispatcher, same phase functions, one tiny cell;
PASS_UNIFIED)::

    uv run python scripts/issue_642/i642_dispatch.py --smoke \
        --output-root /workspace/issue_642_smoke

= sycophancy only x cmft {2,4} (32-row pool slice) + 4-step cmft FT canary
(ZeRO-3 freeze-mask save/consolidate/load + per-device-4 fit check + the 3
freeze-mask asserts + module-set-identity assert vs #606 adapter_config) x 3
personas x 5 claims x 2 rollouts, inline judging included, install gate
log-only, uploads to the ``<experiment>_smoke`` Hub namespace.

CPU dry-run (no GPU / no API; exercises chaining + env + sentinel + done)::

    uv run python scripts/issue_642/i642_dispatch.py --smoke --dry-run \
        --output-root /tmp/i642_dryrun

Production launch (pod, after preflight + smoke)::

    nohup uv run python scripts/issue_642/i642_dispatch.py \
        --arms cmft --behaviors sycophancy --seeds 42 \
        --output-root /workspace/issue_642 --resume-from-phase auto \
        > /workspace/logs/issue-642.log 2>&1 &

Pre-authorized cmft lr-2e-6 retrain (plan §4.11/§7/§13 grid-jump fallback only;
relaunch p1_train after deleting the cmft_ckpts dir)::

    uv run python scripts/issue_642/i642_dispatch.py \
        --arms cmft --behaviors sycophancy --ft-lr 2e-6 --ft-grid retrain \
        --run-label cmft-lr2e6-retrain --seeds 42 \
        --output-root /workspace/issue_642 --resume-from-phase p1_train
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "issue_642"))

# DOTENV_LINT_EXEMPT: legacy pre-#745 script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from i642_common import (  # noqa: E402
    BASE_MODEL,
    CMFT_CKPT_GRID,
    DATA_REVISION_DEFAULT,
    EVAL_MAX_NEW_TOKENS,
    FT_CKPT_GRID,
    FT_LR,
    FT_RETRAIN_GRID,
    FT_RETRAIN_LR,
    HF_DATA_REPO,
    HF_EXPERIMENT_NAME,
    HF_MODEL_REPO,
    JUDGE_MODEL,
    LORA_ADAPTER_CONFIG_HUB_PATH,
    LORA_CKPT_GRID,
    MODEL_REVISION_DEFAULT,
    PARENT_EXPERIMENT_NAME,
    REFUSAL_EVAL_POOL_HUB_PATH,
    REFUSAL_EXPECTED_NEGATIVES,
    REFUSAL_TRAINING_ROWS_HUB_PATH,
    REUSED_FT_STEPS,
    REUSED_LORA_STEPS,
    S_BAND,
    S_TARGET,
    SEED,
    SOURCE_PERSONA,
    SYCO_EVAL_POOL_HUB_PATH,
    SYCO_EXPECTED_NEGATIVES,
    SYCO_POOL_EXPECTED_SHA256,
    SYCO_POOL_HUB_PATH,
    TWIN_PROMPTS,
    TWIN_VALIDATION_HUB_PATH,
    V4_ARM_SPEC,
    V4_ARMS,
    V4_CANNED_POOL_EXPECTED_SHA256,
    V4_CANNED_POOL_HUB_PATH,
    V4_CMFT_FINE_GRID,
    V4_EVAL_PROBES_HUB_PATH,
    V4_HF_EXPERIMENT_NAME,
    V4_LORA_ADAPTER_CONFIG_HUB_PATH,
    V4_LORA_FINE_GRID,
    V4_N_PROBES,
    V4_ONPOLICY_POOL_EXPECTED_SHA256,
    V4_ONPOLICY_POOL_HUB_PATH,
    V4_PANEL_SET_HUB_PATH,
    V4_PILOT_GRID,
    V4_S_SECONDARY,
    V4_SOURCE_PERSONA,
    V4_SOURCE_PROMPT,
    V4_WANDB_PROJECT,
    WANDB_PROJECT,
    _retry_transient,
    assert_pool_disjointness,
    build_refusal_pool,
    judge_generation_file,
    panel_personas,
    roster_personas,
    select_checkpoints,
    sha256_file,
    v4_arm_lr,
    v4_assert_pool_disjointness,
    v4_load_panel,
    v4_splice_canned_pool,
)

log = logging.getLogger("issue_642.dispatch")

GEN_WORKER = REPO / "scripts" / "issue_642" / "i642_gen_worker.py"
LORA_TRAIN_WORKER = REPO / "scripts" / "issue_642" / "i642_lora_train_worker.py"
FT_TRAINER = REPO / "scripts" / "train_behavior_fullft.py"
ACCEL_CONFIG = REPO / "configs" / "accelerate" / "zero3_4gpu_accum1.yaml"

# Arms that train via the ZeRO-3 full-FT trainer (cmft adds the freeze mask).
FT_LIKE_ARMS = ("ft", "cmft")
ALL_ARMS = ("lora", "ft", "cmft")

SMOKE_LORA_GRID = (2, 4)
SMOKE_FT_GRID = (4,)
SMOKE_CMFT_GRID = (2, 4)  # plan §4.6: cmft x 2 checkpoints
SMOKE_FT_MAX_STEPS = 4
SMOKE_POOL_ROWS = 32
SMOKE_N_PROBES = 5
SMOKE_N_ROLLOUTS = 2
SMOKE_PANEL = ("software_engineer", "qwen_default", "supervillain")

# v4/v5 smoke panel (3 of the #612 30-panel: the villain source + a default-
# context slice + a near-twin) — used when --v4 --smoke.
V4_SMOKE_PANEL = ("villain", "qwen_default", "supervillain")
# v4 smoke grids: 2 ckpts per arm (one per FT-canary; the matched-LR-collapse
# step-4 guard included).
V4_SMOKE_LORA_GRID = (2, 4)
V4_SMOKE_CMFT_GRID = (2, 4)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- git sha probe
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _phase_log(tag: str, msg: str) -> None:
    """poll_pipeline.py contract: '[phase=<tag>]' parsed from the log tail."""
    print(f"{datetime.now(UTC).isoformat()} [phase={tag}] {msg}", flush=True)


def _cell_arm_slug(slug: str) -> str:
    """Arm prefix of a cell slug: 'loraOP_lr5e6_step12' -> 'loraOP_lr5e6';
    'lora_step32' -> 'lora'; 'base' -> 'base'. The v4 selection groups by this
    (the 3 distinct v4 arm slugs must NOT collapse to one method token)."""
    if slug == "base":
        return "base"
    return slug.rsplit("_step", 1)[0]


class Ctx:
    """Dispatch context: scope, paths, smoke/dry-run switches (#591 pattern)."""

    def __init__(self, args: argparse.Namespace):  # noqa: C901 - one linear config pass; the v4 branch is parallel to the v3 setup
        self.smoke: bool = args.smoke
        self.dry_run: bool = args.dry_run
        self.seed: int = args.seed
        self.output_root: Path = args.output_root
        self.data_revision: str = args.data_revision
        self.model_revision: str = args.model_revision
        self.skip_upload: bool = args.skip_upload
        self.upload_adapters: bool = args.upload_adapters
        self.n_gpus: int = args.gpus
        self.resume_from: str = args.resume_from_phase
        self.ft_lr: float = getattr(args, "ft_lr", FT_LR)
        self.experiment_name: str = args.hf_experiment_name
        if self.smoke and self.experiment_name == HF_EXPERIMENT_NAME:
            # Smoke artifacts never land in the production Hub namespace.
            self.experiment_name = f"{HF_EXPERIMENT_NAME}_smoke"
        # Retrain/fallback run (plan §4.11/§13): --run-label scopes every Hub
        # path under <experiment>/<label>/.
        self.run_label: str | None = getattr(args, "run_label", None) or None
        if self.run_label is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", self.run_label
        ):
            raise ValueError(
                f"--run-label {self.run_label!r} must match [A-Za-z0-9][A-Za-z0-9._-]* "
                "(it becomes a Hub path segment)"
            )
        if self.run_label is not None:
            self.experiment_name = f"{self.experiment_name}/{self.run_label}"
        if self.dry_run and not self.skip_upload and self.experiment_name == HF_EXPERIMENT_NAME:
            log.warning("dry-run + production --hf-experiment-name: forcing --skip-upload")
            self.skip_upload = True
        # --v4: the followup `onpolicy-matchedlr-rank-isolation` mode (plan v8).
        # Trains THREE villain arms at matched LR 5e-6 on on-policy data + reads
        # them on the #612 30-panel — NO #606 reuse. The v4 mode swaps the
        # source / panel / pools / arm registry / experiment name; the v3
        # (software_engineer / #606-reuse / 3-arm) path is unchanged when
        # --v4 is absent.
        self.v4: bool = getattr(args, "v4", False)
        self.behaviors: list[str] = [b.strip() for b in args.behaviors.split(",") if b.strip()]
        for b in self.behaviors:
            if b not in ("sycophancy", "refusal"):
                raise ValueError(f"unknown behavior {b!r}")
        if self.v4:
            if self.behaviors != ["sycophancy"]:
                raise ValueError("--v4 supports sycophancy only (no on-policy refusal pool exists)")
            # v4 experiment namespace (independent of the v3 default; honors
            # --hf-experiment-name only when the user overrode it explicitly).
            if self.experiment_name == HF_EXPERIMENT_NAME:
                self.experiment_name = V4_HF_EXPERIMENT_NAME
                if self.smoke:
                    self.experiment_name = f"{V4_HF_EXPERIMENT_NAME}_smoke"
                if self.run_label is not None:
                    self.experiment_name = f"{self.experiment_name}/{self.run_label}"
            requested = [a.strip() for a in getattr(args, "arms", "").split(",") if a.strip()]
            if not requested:
                requested = list(V4_ARMS)
            for a in requested:
                if a not in V4_ARMS:
                    # refuse-to-train guard (plan v8 §4.2 item 3a / §8-bis): a slug
                    # retained as role="evidence_only" (the falsified v5-round-1 1e-5
                    # arms) is RESOLVABLE in the registry but MUST NOT be trained — a
                    # stale launch command naming cmftOP_lr1e5 fails loud here instead
                    # of silently trying to train a refused arm.
                    if a in V4_ARM_SPEC and V4_ARM_SPEC[a].get("role") == "evidence_only":
                        raise ValueError(
                            f"v4 arm {a!r} is role='evidence_only' (the falsified v5-round-1 "
                            "1e-5 regime) and REFUSES to train; --arms may name only the "
                            f"production arms {V4_ARMS}. Use --v4 --arms "
                            f"{','.join(V4_ARMS)} for the v8 production sweep."
                        )
                    raise ValueError(f"unknown v4 arm {a!r} (expected one of {V4_ARMS})")
            # canonical v4 order kept (headline pair first)
            self.arms = tuple(a for a in V4_ARMS if a in requested)
            if not self.arms:
                raise ValueError("--arms parsed to an empty v4 set")
            self.source_persona = V4_SOURCE_PERSONA
            self.source_prompt = V4_SOURCE_PROMPT
            self.install_pilot: bool = getattr(args, "install_pilot", False)
        else:
            # --arms: which arms this run TRAINS + evals (canonical order kept). #642
            # production trains cmft ONLY (lora/ft reused from #606). Empty = cmft.
            arms_arg = getattr(args, "arms", "") or "cmft"
            requested_arms = [a.strip() for a in arms_arg.split(",") if a.strip()]
            for a in requested_arms:
                if a not in ALL_ARMS:
                    raise ValueError(f"unknown arm {a!r} (expected lora/ft/cmft)")
            self.arms = tuple(a for a in ALL_ARMS if a in requested_arms)
            if not self.arms:
                raise ValueError("--arms parsed to an empty set")
            self.source_persona = SOURCE_PERSONA
            self.source_prompt = roster_personas()[SOURCE_PERSONA]
            self.install_pilot = False
        self._v4_panel_cache: dict[str, str] | None = None
        # --ft-grid: 'retrain' = the §4.11 densified FT_RETRAIN_GRID; else a
        # comma list of positive ints; default = the cmft grid (= #606 FT grid).
        ft_grid_arg = (getattr(args, "ft_grid", None) or "").strip()
        if ft_grid_arg == "retrain":
            ft_grid_override: tuple[int, ...] | None = FT_RETRAIN_GRID
        elif ft_grid_arg:
            ft_grid_override = tuple(sorted({int(x) for x in ft_grid_arg.split(",") if x.strip()}))
            if not ft_grid_override or any(s <= 0 for s in ft_grid_override):
                raise ValueError(f"--ft-grid must be positive ints, got {ft_grid_arg!r}")
        else:
            ft_grid_override = None
        if self.smoke:
            self.behaviors = self.behaviors[:1]  # sycophancy by default
            self.n_probes: int | None = SMOKE_N_PROBES
            self.n_rollouts = SMOKE_N_ROLLOUTS
            self.lora_grid: tuple[int, ...] = V4_SMOKE_LORA_GRID if self.v4 else SMOKE_LORA_GRID
            self.ft_grid: tuple[int, ...] = SMOKE_FT_GRID
            self.cmft_grid: tuple[int, ...] = V4_SMOKE_CMFT_GRID if self.v4 else SMOKE_CMFT_GRID
            self.ft_max_steps = SMOKE_FT_MAX_STEPS
        elif self.v4:
            # v4 full-grid: fine grid for both poles (plan §4.4); the install-
            # pilot trains the short coarse grid before the full train.
            self.n_probes = V4_N_PROBES  # the #612 eval_60 set
            self.n_rollouts = 10
            self.lora_grid = ft_grid_override or V4_LORA_FINE_GRID
            self.ft_grid = ft_grid_override or V4_CMFT_FINE_GRID
            self.cmft_grid = ft_grid_override or V4_CMFT_FINE_GRID
            self.ft_max_steps = 0
            if self.install_pilot:
                # Phase 0.5 short pilot: train only to ~step 44 + read the coarse
                # pilot grid (plan §4.6). max-steps caps training; the ckpt grid
                # is the pilot coarse grid.
                self.lora_grid = V4_PILOT_GRID
                self.ft_grid = V4_PILOT_GRID
                self.cmft_grid = V4_PILOT_GRID
                self.ft_max_steps = max(V4_PILOT_GRID)
        else:
            self.n_probes = None  # full 50
            self.n_rollouts = 10
            self.lora_grid = LORA_CKPT_GRID
            self.ft_grid = ft_grid_override or FT_CKPT_GRID
            self.cmft_grid = ft_grid_override or CMFT_CKPT_GRID
            self.ft_max_steps = 0
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.sentinel_dir = (
            Path("/workspace/logs")
            if Path("/workspace/logs").is_dir()
            else self.output_root / "logs"
        )
        self.sentinel_dir.mkdir(parents=True, exist_ok=True)
        # Per-behavior kill state (install gate / kill criteria).
        self.killed: dict[str, str] = {}
        # Reproducibility card accumulators (cmft adapter + WandB provenance).
        self.cmft_selected_steps: dict[str, list[int]] = {}
        self.cmft_uploaded_adapters: dict[str, list[str]] = {}

    # -- per-arm grid --
    def arm_grid(self, arm: str) -> tuple[int, ...]:
        if self.v4:
            # v4 arm slugs: the LoRA pole uses the LoRA grid; the 3 cmft arms
            # share the cmft grid (same fine grid, plan §4.4).
            return self.lora_grid if V4_ARM_SPEC[arm]["method"] == "lora" else self.cmft_grid
        return {"lora": self.lora_grid, "ft": self.ft_grid, "cmft": self.cmft_grid}[arm]

    def v4_arm_method(self, arm: str) -> str:
        """v4 arm method: 'lora' or 'cmft' (plan §4.2 / V4_ARM_SPEC)."""
        return V4_ARM_SPEC[arm]["method"]

    def v4_arm_data_kind(self, arm: str) -> str:
        """v4 arm training-data kind: 'on_policy' or 'canned' (plan §4.7)."""
        return V4_ARM_SPEC[arm]["data"]

    def v4_arm_lr(self, arm: str) -> float:
        """v4 arm learning rate: matched 5e-6 (production) or old_matched 1e-5
        (evidence_only slugs) (plan v8 §10)."""
        return v4_arm_lr(arm)

    def panel(self) -> dict[str, str]:
        """v4 30-persona panel from the #612 panel_set.json (cached); v3 = the
        39-persona roster+twins panel."""
        if not self.v4:
            return panel_personas()
        if self._v4_panel_cache is None:
            panel_path = self.data_dir(self.behaviors[0]) / "panel_set.json"
            if self.dry_run:
                # dry-run placeholder panel (3 personas) bypasses the 30-assert.
                raw = json.loads(panel_path.read_text())
                self._v4_panel_cache = {
                    name: rec["prompt"] for name, rec in raw["personas"].items()
                }
            else:
                self._v4_panel_cache = v4_load_panel(panel_path)
        return self._v4_panel_cache

    # -- derived paths --
    def bdir(self, behavior: str) -> Path:
        return self.output_root / behavior

    def data_dir(self, behavior: str) -> Path:
        return self.bdir(behavior) / "data"

    def stage_a_dir(self, behavior: str) -> Path:
        return self.bdir(behavior) / "stage_a"

    def gen_dir(self, behavior: str) -> Path:
        return self.bdir(behavior) / "generations"

    def lora_ckpt_root(self, behavior: str) -> Path:
        return self.bdir(behavior) / "lora_ckpts"

    def ft_ckpt_root(self, behavior: str) -> Path:
        return self.bdir(behavior) / "ft_ckpts"

    def cmft_ckpt_root(self, behavior: str) -> Path:
        return self.bdir(behavior) / "cmft_ckpts"

    def ckpt_root(self, behavior: str, arm: str) -> Path:
        if self.v4:
            # v4: each arm slug gets its OWN checkpoint root (3 distinct cmft
            # arms + the LoRA pole cannot share one root).
            return self.bdir(behavior) / "ckpts" / arm
        return {
            "lora": self.lora_ckpt_root,
            "ft": self.ft_ckpt_root,
            "cmft": self.cmft_ckpt_root,
        }[arm](behavior)

    def adapter_config_path(self, behavior: str) -> Path:
        return self.data_dir(behavior) / "lora_adapter_config.json"

    def manifest_path(self, behavior: str) -> Path:
        return self.bdir(behavior) / "generation_manifest.json"

    def stage_b_panel(self) -> dict[str, str]:
        panel = self.panel()
        if self.smoke:
            smoke_panel = V4_SMOKE_PANEL if self.v4 else SMOKE_PANEL
            return {k: panel[k] for k in smoke_panel}
        return panel

    def eval_pool(self, behavior: str) -> Path:
        full = self.data_dir(behavior) / "eval_pool.jsonl"
        if self.n_probes is None:
            return full
        sliced = self.data_dir(behavior) / f"eval_pool_first{self.n_probes}.jsonl"
        if not sliced.exists():
            lines = [ln for ln in full.read_text().splitlines() if ln.strip()]
            sliced.write_text("\n".join(lines[: self.n_probes]) + "\n")
        return sliced

    def train_pool(self, behavior: str, arm: str | None = None) -> Path:
        """Training pool for an arm. v4: on-policy arms read the #612 pool;
        the canned arm reads the spliced #411-positive pool (plan §4.7). v3 +
        non-arm callers read the single behavior pool."""
        if self.v4 and arm is not None and self.v4_arm_data_kind(arm) == "canned":
            base = self.data_dir(behavior) / "train_pool_canned.jsonl"
        else:
            base = self.data_dir(behavior) / "train_pool.jsonl"
        if self.smoke:
            return base.with_name(base.stem + f"_first{SMOKE_POOL_ROWS}.jsonl")
        return base


# ---------------------------------------------------------------------------
# Hub upload helpers (ported from origin/issue-606 — batched commits)
# ---------------------------------------------------------------------------


def _hub_upload_file(local: Path, path_in_repo: str, *, skip: bool) -> str | None:
    if skip:
        _phase_log("p5_upload", f"SKIP upload {local} -> {path_in_repo} (--skip-upload)")
        return None
    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload

    url = _upload(
        local_path=local,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        upload_as_file=True,
    )
    if not url:
        raise RuntimeError(f"Hub upload failed: {local} -> {path_in_repo}")
    return url


def _count_hub_files_under_prefix(api, prefix: str) -> int:
    from huggingface_hub.hf_api import RepoFile
    from huggingface_hub.utils import EntryNotFoundError

    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO

    def _count() -> int:
        try:
            entries = api.list_repo_tree(
                DEFAULT_DATASET_REPO, path_in_repo=prefix, recursive=True, repo_type="dataset"
            )
            return sum(1 for e in entries if isinstance(e, RepoFile))
        except EntryNotFoundError:
            return 0

    return _retry_transient(_count, what=f"count Hub files under {prefix}")


def _upload_dir_batched(ctx: Ctx, local_dir: Path, repo_subdir: str, *, behavior: str) -> int:
    """ONE batched Hub commit per phase-output directory (HF throttles repo
    commits at 256/h — the #591 epm:failure v2 lesson). Files under a
    ``raw_completions/`` dir are ALSO staged to the canonical
    ``<experiment>/raw_completions/<behavior>/<cell>/`` mirror inside the
    SAME commit (CLAUDE.md Upload Policy shape). Resume-idempotent via a
    prefix-scoped Hub count."""
    files = sorted(local_dir.rglob("*.json"))
    if not files:
        return 0
    rel_dir = local_dir.relative_to(ctx.output_root)
    prefix = f"{ctx.experiment_name}/{rel_dir.as_posix()}"
    ops_meta: list[tuple[Path, str]] = []
    raw_by_prefix: dict[str, int] = {}
    for f in files:
        rel = f.relative_to(ctx.output_root)
        ops_meta.append((f, f"{ctx.experiment_name}/{rel.as_posix()}"))
        if f.parent.name == "raw_completions":
            cell = f.parents[1].name
            canon_prefix = f"{ctx.experiment_name}/raw_completions/{behavior}/{cell}"
            ops_meta.append((f, f"{canon_prefix}/{f.name}"))
            raw_by_prefix[canon_prefix] = raw_by_prefix.get(canon_prefix, 0) + 1
    if ctx.skip_upload:
        _phase_log(
            "p5_upload", f"SKIP batched upload {local_dir} -> {prefix} ({len(ops_meta)} ops)"
        )
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    n_files = len(files)

    def _complete() -> bool:
        if _count_hub_files_under_prefix(api, prefix) < n_files:
            return False
        return all(_count_hub_files_under_prefix(api, p) >= n for p, n in raw_by_prefix.items())

    if _complete():
        _phase_log("p5_upload", f"{prefix}: already complete on Hub — skipping commit (resume)")
        return n_files
    operations = [
        CommitOperationAdd(path_in_repo=pir, path_or_fileobj=str(f)) for f, pir in ops_meta
    ]
    _retry_transient(
        lambda: api.create_commit(
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            operations=operations,
            commit_message=f"#642 phase outputs: {rel_dir.as_posix()} ({len(operations)} files)",
        ),
        what=f"create_commit {prefix}",
    )
    got = _count_hub_files_under_prefix(api, prefix)
    if got < n_files:
        raise RuntimeError(f"batched upload verification FAILED: {got}/{n_files} at {prefix}")
    for p, n in raw_by_prefix.items():
        got_raw = _count_hub_files_under_prefix(api, p)
        if got_raw < n:
            raise RuntimeError(
                f"batched upload verification FAILED: {got_raw}/{n} canonical raw "
                f"completions under {p}"
            )
    _phase_log("p5_upload", f"batched upload verified: {got} files at {prefix} (1 commit)")
    return got


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, env: dict[str, str], log_path: Path | None = None) -> int:
    """Run one subprocess with explicit env; tee output to ``log_path``."""
    log.info("exec: %s", " ".join(cmd))
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as fh:
            proc = subprocess.run(cmd, env=env, cwd=str(REPO), stdout=fh, stderr=fh)
    else:
        proc = subprocess.run(cmd, env=env, cwd=str(REPO))
    return proc.returncode


def _gpu_env(gpu: int | str) -> dict[str, str]:
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _run_gpu_queue(ctx: Ctx, jobs: list[tuple[str, list[str], Path]]) -> None:
    """Run (name, cmd, log_path) jobs across the GPU pool, <=1 job per GPU.

    Fails LOUD after draining: any non-zero rc raises with the failing job
    names + log paths (no silent partial coverage)."""
    if ctx.dry_run:
        for name, cmd, _lp in jobs:
            _phase_log("p2_stage_a", f"DRY-RUN: would exec [{name}] {' '.join(cmd)}")
        return
    free_gpus = list(range(ctx.n_gpus))
    running: dict[int, tuple] = {}
    pending = list(jobs)
    failures: list[tuple[str, int, Path]] = []
    while pending or running:
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            name, cmd, log_path = pending.pop(0)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log_path, "ab")  # noqa: SIM115 - handle outlives the loop; closed on reap below
            proc = subprocess.Popen(cmd, env=_gpu_env(gpu), cwd=str(REPO), stdout=fh, stderr=fh)
            running[gpu] = (name, proc, log_path, fh)
            log.info("launched [%s] on GPU %d (pid %d)", name, gpu, proc.pid)
        time.sleep(5)
        for gpu in list(running):
            name, proc, log_path, fh = running[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[gpu]
            free_gpus.append(gpu)
            if rc != 0:
                failures.append((name, rc, log_path))
                log.error("[%s] FAILED rc=%d (log: %s)", name, rc, log_path)
            else:
                log.info("[%s] done", name)
    if failures:
        raise RuntimeError(
            "GPU jobs failed: " + "; ".join(f"{n} rc={rc} log={lp}" for n, rc, lp in failures)
        )


# ---------------------------------------------------------------------------
# Phase 0 — data fetch + verify (CPU)
# ---------------------------------------------------------------------------


def _hub_download(
    ctx: Ctx, hub_path: str, *, repo_type: str = "dataset", revision: str | None = None
) -> Path:
    from huggingface_hub import hf_hub_download

    repo = HF_DATA_REPO if repo_type == "dataset" else HF_MODEL_REPO
    rev = revision if revision is not None else ctx.data_revision
    return Path(
        hf_hub_download(
            repo,
            hub_path,
            repo_type=repo_type,
            revision=rev,
            token=os.environ.get("HF_TOKEN"),
        )
    )


def _assert_completion_lengths(pool_path: Path, behavior: str) -> int:
    """Plan §4.7: assert max_new_tokens 512 >= 2x the longest trained
    completion (tokenized with the real Qwen tokenizer)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    max_tokens = 0
    for line in pool_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = " ".join(m["content"] for m in row["completion"])
        n = len(tok.encode(text, add_special_tokens=False))
        max_tokens = max(max_tokens, n)
    if 2 * max_tokens > EVAL_MAX_NEW_TOKENS:
        raise RuntimeError(
            f"[{behavior}] max_new_tokens {EVAL_MAX_NEW_TOKENS} < 2x longest trained "
            f"completion ({max_tokens} tokens) — truncation-silent-zero risk (#260)."
        )
    log.info("[%s] completion-length assert PASS (max %d tokens)", behavior, max_tokens)
    return max_tokens


def _v4_assert_train_truncation(pool_path: Path, behavior: str, *, max_length: int = 1024) -> int:
    """v4 completion-length guard. v4 sycophancy is a FREE-GENERATION behavioral
    eval (model writes its own agreeing response; judge scores it) — the
    512-token exception class per CLAUDE.md, so the marker/end-of-completion
    2x-eval-tokens rule does NOT apply (the on-policy positives run ~400 tokens,
    longer than #411's short canned templates). The relevant guard is TRAIN-time
    truncation: the full prompt+completion must fit ``max_length`` so the loss
    tokens are not silently truncated (the CPU build-time guard,
    feedback_cpu_build_time_guard_for_truncation #480). Fail-loud if any row
    exceeds the train max_length budget."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    max_total = 0
    max_completion = 0
    for line in pool_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # full chat-templated prompt+completion length (the train tokenization)
        msgs = [*row["prompt"], *row["completion"]]
        rendered = tok.apply_chat_template(msgs, tokenize=False)
        n_total = len(tok.encode(rendered, add_special_tokens=False))
        n_comp = len(
            tok.encode(" ".join(m["content"] for m in row["completion"]), add_special_tokens=False)
        )
        max_total = max(max_total, n_total)
        max_completion = max(max_completion, n_comp)
    if max_total > max_length:
        raise RuntimeError(
            f"[{behavior}] train-time truncation risk: longest prompt+completion "
            f"({max_total} tokens) > max_length {max_length} — loss tokens would be "
            f"silently truncated (#480). Raise max_length or drop over-budget rows."
        )
    log.info(
        "[%s] v4 train-truncation guard PASS (max total %d / max completion %d <= max_length %d)",
        behavior,
        max_total,
        max_completion,
        max_length,
    )
    return max_completion


def _assert_twin_registry(ctx: Ctx) -> dict:
    """Download twin_validation.json and assert the ported 15-prompt registry
    matches the accepted set EXACTLY (names + prompts)."""
    local = _hub_download(ctx, TWIN_VALIDATION_HUB_PATH)
    v = json.loads(Path(local).read_text())
    accepted = {n: rec["prompt"] for n, rec in v["accepted"].items()}
    if set(accepted) != set(TWIN_PROMPTS):
        raise RuntimeError(
            f"twin registry drift: accepted={sorted(accepted)} vs ported={sorted(TWIN_PROMPTS)}"
        )
    mismatched = {n for n in accepted if accepted[n] != TWIN_PROMPTS[n]}
    if mismatched:
        raise RuntimeError(f"twin PROMPT drift for {sorted(mismatched)}")
    log.info("twin registry assert PASS (15 accepted twins)")
    return v


def _fetch_lora_adapter_config(ctx: Ctx, behavior: str) -> Path:
    """Download the #606 LoRA adapter_config.json from the MODEL repo @
    ``model_revision`` (plan §4.2 assert 2 / §4.5). The cmft trainer reads
    ``target_modules`` + ``bias`` from it for the module-set-identity assert."""
    dest = ctx.adapter_config_path(behavior)
    if dest.exists():
        return dest
    got = _hub_download(
        ctx, LORA_ADAPTER_CONFIG_HUB_PATH, repo_type="model", revision=ctx.model_revision
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(got, dest)
    cfg = json.loads(dest.read_text())
    log.info(
        "[%s] #606 LoRA adapter_config @ %s: target_modules=%s bias=%s",
        behavior,
        ctx.model_revision[:12],
        sorted(cfg.get("target_modules", [])),
        cfg.get("bias", "none"),
    )
    return dest


def _assert_reused_606_cells(ctx: Ctx, behavior: str) -> dict:
    """Enumerate + hard-assert the reused #606 syco generation cells exist in
    the DATA repo @ ``data_revision`` BEFORE training (plan §4.5 / Risk row):
    the cmft join is impossible without LoRA {28,32,36,132} + FT {12,16,22,132}
    + base generations + the stage-A trajectory.

    The analyzer re-judges EVERY (cell, persona) over the FULL 39-persona panel
    (``i642_analyze.analyze_behavior`` loops ``panel`` per cell), so the
    fail-loud preflight must verify the FULL panel per reused cell — not just
    the source-self file. A partial #606 cell (any persona file missing) would
    pass the old source-only check and crash the off-pod analysis AFTER the
    entire GPU budget was spent (round-2 CONCERN fix). Fail-loud if any of the
    39 persona generation files is absent for any of the 9 reused cells."""
    if behavior != "sycophancy" or ctx.dry_run:
        return {"checked": False, "reason": "non-syco or dry-run"}
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = _retry_transient(
        lambda: api.list_repo_files(HF_DATA_REPO, revision=ctx.data_revision, repo_type="dataset"),
        what="list reused #606 data-repo files",
    )
    fileset = set(files)
    # The full 39-persona stage-B panel the analyzer re-judges (24 roster incl.
    # source + 15 twins). Looped against EVERY reused cell.
    panel = sorted(panel_personas())
    if len(panel) != 39:
        raise RuntimeError(f"[{behavior}] expected 39-persona panel, got {len(panel)}")
    gen_cells = [f"lora_step{s}" for s in REUSED_LORA_STEPS] + [
        f"ft_step{s}" for s in REUSED_FT_STEPS
    ]
    reused_cells = [*gen_cells, "base"]
    missing: list[str] = []
    for cell in reused_cells:
        # require the FULL 39-persona panel generation file per reused cell
        for persona in panel:
            rel = (
                f"{PARENT_EXPERIMENT_NAME}/{behavior}/generations/{cell}/"
                f"{behavior}_eval_{persona}.json"
            )
            if rel not in fileset:
                missing.append(rel)
    traj_rel = f"{PARENT_EXPERIMENT_NAME}/{behavior}/stage_a/trajectory_{behavior}.json"
    if traj_rel not in fileset:
        missing.append(traj_rel)
    if missing:
        raise RuntimeError(
            f"[{behavior}] reused #606 cells MISSING in {HF_DATA_REPO}@{ctx.data_revision[:12]} "
            f"({len(missing)} of {len(reused_cells) * len(panel) + 1} required files absent — "
            f"the cmft join re-judges the full 39-persona panel per cell; plan §4.5): "
            f"{missing[:6]}{' ...' if len(missing) > 6 else ''}"
        )
    report = {
        "checked": True,
        "data_revision": ctx.data_revision,
        "reused_lora_steps": list(REUSED_LORA_STEPS),
        "reused_ft_steps": list(REUSED_FT_STEPS),
        "n_panel_personas": len(panel),
        "n_reused_cells": len(reused_cells),
        "n_generation_files_verified": len(reused_cells) * len(panel),
        "trajectory_present": True,
    }
    log.info(
        "[%s] reused #606 cells present: full %d-persona panel x %d cells (%d gen files) + traj",
        behavior,
        len(panel),
        len(reused_cells),
        len(reused_cells) * len(panel),
    )
    return report


def _write_dry_run_pool(path: Path, behavior: str, n: int = 24) -> None:
    """Tier-guarded placeholder pool for --dry-run chaining (never uploaded
    to the production namespace — Ctx guard)."""
    personas = roster_personas()
    rows = []
    for i in range(n):
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": personas[SOURCE_PERSONA]},
                    {"role": "user", "content": f"DRY-RUN {behavior} probe {i}?"},
                ],
                "completion": [{"role": "assistant", "content": f"DRY-RUN completion {i}."}],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _write_dry_run_adapter_config(path: Path) -> None:
    """Placeholder adapter_config for --dry-run (the cmft mask asserts only run
    on a GPU train, which dry-run skips; this keeps the Phase-0 chain intact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "bias": "none",
                "dry_run": True,
            },
            indent=2,
        )
    )


def _hub_download_v4(ctx: Ctx, hub_path: str, *, repo_type: str = "dataset") -> Path:
    """v4 Hub fetch at HEAD (the #612/#411 pools are sha-pinned by content, not
    by revision — see Ctx.data_revision is unused in v4)."""
    from huggingface_hub import hf_hub_download

    repo = HF_DATA_REPO if repo_type == "dataset" else HF_MODEL_REPO
    return Path(
        hf_hub_download(
            repo,
            hub_path,
            repo_type=repo_type,
            token=os.environ.get("HF_TOKEN"),
        )
    )


def _phase0_data_v4(ctx: Ctx, behavior: str) -> None:
    """v4 Phase 0 (plan v8 §4.3 / §4.7): fetch + sha-verify the #612 villain
    on-policy pool, the #411 villain canned pool, the #612 30-panel + eval_60
    probes, and the #612 villain LoRA adapter_config (cmft mask assert); BUILD
    the canned-cmft pool (#411 positives + #612 negatives, byte-identical);
    v4 disjointness + byte-identical-negatives + completion-length asserts.
    NO #606 reuse — this run trains 3 NEW arms at matched LR 5e-6."""
    ddir = ctx.data_dir(behavior)
    ddir.mkdir(parents=True, exist_ok=True)

    if ctx.dry_run:
        _write_dry_run_pool(ddir / "train_pool.jsonl", behavior)
        _write_dry_run_pool(ddir / "train_pool_canned.jsonl", behavior)
        probes = [
            {"wrong_claim": f"DRY-RUN {behavior} claim {i}?", "correction": "DRY-RUN."}
            for i in range(SMOKE_N_PROBES)
        ]
        (ddir / "eval_pool.jsonl").write_text("\n".join(json.dumps(p) for p in probes) + "\n")
        _write_dry_run_adapter_config(ctx.adapter_config_path(behavior))
        # minimal dry-run panel (3 personas incl. source) so ctx.panel() works
        (ddir / "panel_set.json").write_text(
            json.dumps(
                {
                    "personas": {
                        p: {"prompt": f"DRY-RUN {p} prompt"}
                        for p in (V4_SOURCE_PERSONA, "qwen_default", "supervillain")
                    }
                }
            )
        )
        manifest = {"dry_run": True, "behavior": behavior, "v4": True}
    else:
        # -- on-policy pool (#612) --
        op_got = _hub_download_v4(ctx, V4_ONPOLICY_POOL_HUB_PATH)
        op_sha = sha256_file(op_got)
        if op_sha != V4_ONPOLICY_POOL_EXPECTED_SHA256:
            raise RuntimeError(
                f"[v4] on-policy pool EXPECTED_SHA256 FAILED: {op_sha} != "
                f"{V4_ONPOLICY_POOL_EXPECTED_SHA256} for {V4_ONPOLICY_POOL_HUB_PATH}"
            )
        shutil.copy2(op_got, ddir / "train_pool.jsonl")
        # -- canned pool (#411) for the splice --
        cn_got = _hub_download_v4(ctx, V4_CANNED_POOL_HUB_PATH)
        cn_sha = sha256_file(cn_got)
        if cn_sha != V4_CANNED_POOL_EXPECTED_SHA256:
            raise RuntimeError(
                f"[v4] canned pool EXPECTED_SHA256 FAILED: {cn_sha} != "
                f"{V4_CANNED_POOL_EXPECTED_SHA256} for {V4_CANNED_POOL_HUB_PATH}"
            )
        # -- eval probes (#612 eval_60) --
        ev_got = _hub_download_v4(ctx, V4_EVAL_PROBES_HUB_PATH)
        shutil.copy2(ev_got, ddir / "eval_pool.jsonl")
        n_eval = sum(1 for ln in (ddir / "eval_pool.jsonl").read_text().splitlines() if ln.strip())
        if n_eval != V4_N_PROBES:
            raise RuntimeError(f"[v4] eval pool has {n_eval} probes, expected {V4_N_PROBES}")
        for ln in (ddir / "eval_pool.jsonl").read_text().splitlines():
            if ln.strip():
                assert "wrong_claim" in json.loads(ln), "[v4] eval probe missing wrong_claim"
        # -- panel (#612 30-persona) --
        panel_got = _hub_download_v4(ctx, V4_PANEL_SET_HUB_PATH)
        shutil.copy2(panel_got, ddir / "panel_set.json")
        panel = v4_load_panel(ddir / "panel_set.json")
        # -- #612 villain LoRA adapter_config (cmft module-set assert) --
        adapter_got = _hub_download_v4(ctx, V4_LORA_ADAPTER_CONFIG_HUB_PATH, repo_type="model")
        shutil.copy2(adapter_got, ctx.adapter_config_path(behavior))
        # -- BUILD the canned-cmft pool (splice; byte-identical-negatives assert) --
        splice_report = v4_splice_canned_pool(
            canned_pool_path=cn_got,
            onpolicy_pool_path=ddir / "train_pool.jsonl",
            out_path=ddir / "train_pool_canned.jsonl",
        )
        # -- v4 disjointness (against the realized pool) on BOTH pools --
        disjoint_op = v4_assert_pool_disjointness(ddir / "train_pool.jsonl", panel)
        disjoint_cn = v4_assert_pool_disjointness(ddir / "train_pool_canned.jsonl", panel)
        # v4 uses the TRAIN-time truncation guard (free-generation eval; the
        # 512 max_new_tokens is the CLAUDE.md free-gen exception, not the
        # marker 2x rule).
        max_completion_tokens = max(
            _v4_assert_train_truncation(ddir / "train_pool.jsonl", behavior, max_length=1024),
            _v4_assert_train_truncation(
                ddir / "train_pool_canned.jsonl", behavior, max_length=1024
            ),
        )
        adapter_cfg = json.loads(ctx.adapter_config_path(behavior).read_text())
        manifest = {
            "behavior": behavior,
            "v4": True,
            "source_persona": V4_SOURCE_PERSONA,
            "n_panel_personas": len(panel),
            "n_bystanders": len(panel) - 1,
            "onpolicy_pool_sha256": op_sha,
            "canned_pool_source_sha256": cn_sha,
            "spliced_canned_pool": splice_report,
            "n_eval_probes": n_eval,
            "max_completion_tokens": max_completion_tokens,
            "disjointness_onpolicy": disjoint_op,
            "disjointness_canned": disjoint_cn,
            "lora_adapter_config_target_modules": sorted(adapter_cfg.get("target_modules", [])),
            "lora_adapter_config_bias": adapter_cfg.get("bias", "none"),
            "lora_adapter_config_use_rslora": adapter_cfg.get("use_rslora"),
            "arms": list(ctx.arms),
            "git_commit_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

    if ctx.smoke:
        for stem in ("train_pool", "train_pool_canned"):
            src = ddir / f"{stem}.jsonl"
            lines = [ln for ln in src.read_text().splitlines() if ln.strip()]
            (ddir / f"{stem}_first{SMOKE_POOL_ROWS}.jsonl").write_text(
                "\n".join(lines[:SMOKE_POOL_ROWS]) + "\n"
            )
    (ddir / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    _hub_upload_file(
        ddir / "data_manifest.json",
        f"{ctx.experiment_name}/{behavior}/data_manifest.json",
        skip=ctx.skip_upload,
    )
    _phase_log("p0_data", f"[v4][{behavior}] Phase 0 done (3-arm pools + #612 panel)")


def phase0_data(ctx: Ctx, behavior: str) -> None:
    if ctx.v4:
        _phase_log("p0_data", f"[v4][{behavior}] Phase 0 start (villain on-policy + canned)")
        ddir = ctx.data_dir(behavior)
        if (ddir / "data_manifest.json").exists():
            _phase_log("p0_data", f"[v4][{behavior}] data_manifest exists — skipping (resume)")
            return
        _phase0_data_v4(ctx, behavior)
        return
    _phase_log(
        "p0_data",
        f"[{behavior}] Phase 0 start (data {ctx.data_revision[:12]} / "
        f"model {ctx.model_revision[:12]})",
    )
    ddir = ctx.data_dir(behavior)
    manifest_path = ddir / "data_manifest.json"
    if manifest_path.exists():
        _phase_log("p0_data", f"[{behavior}] data_manifest exists — skipping (resume)")
        return
    ddir.mkdir(parents=True, exist_ok=True)

    if ctx.dry_run:
        _write_dry_run_pool(ddir / "train_pool.jsonl", behavior)
        _write_dry_run_pool(ddir / "eval_pool.jsonl", behavior, n=SMOKE_N_PROBES)
        probes = [
            {"wrong_claim": f"DRY-RUN {behavior} claim {i}?", "correction": "DRY-RUN."}
            for i in range(SMOKE_N_PROBES)
        ]
        (ddir / "eval_pool.jsonl").write_text("\n".join(json.dumps(p) for p in probes) + "\n")
        _write_dry_run_adapter_config(ctx.adapter_config_path(behavior))
        manifest = {"dry_run": True, "behavior": behavior}
    else:
        if behavior == "sycophancy":
            got = _hub_download(ctx, SYCO_POOL_HUB_PATH)
            sha = sha256_file(got)
            if sha != SYCO_POOL_EXPECTED_SHA256:
                raise RuntimeError(
                    f"[sycophancy] EXPECTED_SHA256 assert FAILED: {sha} != "
                    f"{SYCO_POOL_EXPECTED_SHA256} for {SYCO_POOL_HUB_PATH} — content "
                    f"identity broken (rule (f)); STOP before any training."
                )
            shutil.copy2(got, ddir / "train_pool.jsonl")
            eval_got = _hub_download(ctx, SYCO_EVAL_POOL_HUB_PATH)
            shutil.copy2(eval_got, ddir / "eval_pool.jsonl")
            expected_negs = SYCO_EXPECTED_NEGATIVES
            pool_sha = sha
        else:  # refusal (not on #642's production path; retained for completeness)
            rows_got = _hub_download(ctx, REFUSAL_TRAINING_ROWS_HUB_PATH)
            inputs_sha = sha256_file(rows_got)
            n = build_refusal_pool(
                source=SOURCE_PERSONA,
                seed=ctx.seed,
                training_rows_path=Path(rows_got),
                out_path=ddir / "train_pool.jsonl",
            )
            if n != 700:
                raise RuntimeError(f"[refusal] rebuilt pool has {n} rows, expected 700")
            eval_got = _hub_download(ctx, REFUSAL_EVAL_POOL_HUB_PATH)
            shutil.copy2(eval_got, ddir / "eval_pool.jsonl")
            expected_negs = REFUSAL_EXPECTED_NEGATIVES
            pool_sha = sha256_file(ddir / "train_pool.jsonl")
            log.info("[refusal] rebuilt-pool sha256 = %s (inputs %s)", pool_sha, inputs_sha)

        n_eval = sum(1 for ln in (ddir / "eval_pool.jsonl").read_text().splitlines() if ln.strip())
        if n_eval != 50:
            raise RuntimeError(f"[{behavior}] eval pool has {n_eval} probes, expected 50")
        for ln in (ddir / "eval_pool.jsonl").read_text().splitlines():
            if ln.strip():
                row = json.loads(ln)
                assert "wrong_claim" in row, f"[{behavior}] eval probe missing wrong_claim"

        disjoint = assert_pool_disjointness(
            ddir / "train_pool.jsonl",
            source=SOURCE_PERSONA,
            expected_negatives=expected_negs,
            behavior=behavior,
        )
        max_completion_tokens = _assert_completion_lengths(ddir / "train_pool.jsonl", behavior)
        twin_validation = _assert_twin_registry(ctx)
        # cmft module-set-identity source (plan §4.2) + reused-#606-cell enumerate.
        adapter_cfg = _fetch_lora_adapter_config(ctx, behavior)
        reused_check = _assert_reused_606_cells(ctx, behavior)
        manifest = {
            "behavior": behavior,
            "data_revision": ctx.data_revision,
            "model_revision": ctx.model_revision,
            "train_pool_sha256": pool_sha,
            "n_eval_probes": n_eval,
            "max_completion_tokens": max_completion_tokens,
            "disjointness": disjoint,
            "n_twins_accepted": len(twin_validation["accepted"]),
            "lora_adapter_config": str(adapter_cfg),
            "reused_606_cells": reused_check,
            "git_commit_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

    if ctx.smoke:
        lines = [ln for ln in (ddir / "train_pool.jsonl").read_text().splitlines() if ln.strip()]
        (ddir / f"train_pool_first{SMOKE_POOL_ROWS}.jsonl").write_text(
            "\n".join(lines[:SMOKE_POOL_ROWS]) + "\n"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _hub_upload_file(
        manifest_path,
        f"{ctx.experiment_name}/{behavior}/data_manifest.json",
        skip=ctx.skip_upload,
    )
    _phase_log("p0_data", f"[{behavior}] Phase 0 done")


# ---------------------------------------------------------------------------
# Phase 1 — training (LoRA 1 GPU; FT/cmft 4 GPU ZeRO-3)
# ---------------------------------------------------------------------------


def _ft_env(ctx: Ctx) -> dict[str, str]:
    """Multi-GPU env for accelerate launch (the #514 CVD-leak fix)."""
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(ctx.n_gpus))
    return env


def _dry_run_fake_ckpts(root: Path, steps: tuple[int, ...], meta: dict) -> None:
    for s in steps:
        (root / f"checkpoint-{s}").mkdir(parents=True, exist_ok=True)
    (root / "train_metadata.json").write_text(
        json.dumps({**meta, "dry_run": True, "saved_checkpoints": sorted(steps)}, indent=2)
    )


def _train_ft_like(ctx: Ctx, behavior: str, arm: str) -> None:
    """Train a ZeRO-3 full-FT-style arm (``ft`` or ``cmft``). The cmft arm adds
    the freeze flag + the adapter_config (plan §4.2/§4.3). OOM fallback to
    per-device 2 x accum 2 (eff. batch 16 preserved) is pre-registered."""
    ft_root = ctx.ckpt_root(behavior, arm)
    if (ft_root / "train_metadata.json").exists():
        _phase_log("p1_train", f"[{behavior}] {arm} train_metadata exists — skipping (resume)")
        return
    if ctx.dry_run:
        _dry_run_fake_ckpts(ft_root, ctx.arm_grid(arm), {"behavior": behavior, "arm": arm})
        _phase_log("p1_train", f"[{behavior}] DRY-RUN: placeholder {arm} checkpoints written")
        return
    train_jsonl = ctx.train_pool(behavior)
    grid = ctx.arm_grid(arm)

    def _ft_cmd(per_device: int, accum: int) -> list[str]:
        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            str(ACCEL_CONFIG),
            "--num_processes",
            str(ctx.n_gpus),
            # Launch-level grad-accum so the OOM fallback (2,2) does not trip the
            # transformers DeepSpeed mismatch check against the yaml's pinned
            # `gradient_accumulation_steps: 1` (the #606 launch-flag note).
            "--gradient_accumulation_steps",
            str(accum),
            str(FT_TRAINER),
            "--behavior",
            behavior,
            "--arm",
            arm,
            "--train-jsonl",
            str(train_jsonl),
            "--output-dir",
            str(ft_root),
            "--ckpt-steps",
            ",".join(map(str, grid)),
            "--seed",
            str(ctx.seed),
            "--learning-rate",
            str(ctx.ft_lr),
            "--per-device-batch",
            str(per_device),
            "--grad-accum",
            str(accum),
            "--wandb-project",
            WANDB_PROJECT,
        ]
        if arm == "cmft":
            cmd += [
                "--freeze-outside-lora-modules",
                "--lora-adapter-config-json",
                str(ctx.adapter_config_path(behavior)),
            ]
        if ctx.run_label:
            cmd += ["--run-name-suffix", ctx.run_label]
        if ctx.ft_max_steps > 0:
            cmd += ["--max-steps", str(ctx.ft_max_steps)]
        return cmd

    ft_log = ctx.bdir(behavior) / "logs" / f"{arm}_train.log"
    rc = _run(_ft_cmd(4, 1), env=_ft_env(ctx), log_path=ft_log)
    if rc != 0:
        tail = ft_log.read_text(errors="replace")[-8000:] if ft_log.exists() else ""
        if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
            _phase_log(
                "p1_train",
                f"[{behavior}] {arm} OOM at per-device 4 — pre-registered fallback "
                f"per-device 2 x accum 2 (eff. batch unchanged)",
            )
            shutil.rmtree(ft_root, ignore_errors=True)
            rc = _run(_ft_cmd(2, 2), env=_ft_env(ctx), log_path=ft_log)
        if rc != 0:
            raise RuntimeError(f"[{behavior}] {arm} training failed rc={rc} (log: {ft_log})")


def _train_v4_cmft_arm(ctx: Ctx, behavior: str, arm: str) -> None:
    """Train one v4 cmft arm slug (cmftOP_lr5e6 / cmftCN_lr5e6) via the ZeRO-3
    cmft trainer at the matched LR 5e-6 + own pool (plan v8 §4.2). The freeze
    mask + module-set-identity assert vs the #612 villain adapter_config are
    identical across both production cmft arms (the manipulated variable within
    this pair is data construction only — on-policy vs canned)."""
    ft_root = ctx.ckpt_root(behavior, arm)
    if (ft_root / "train_metadata.json").exists():
        _phase_log("p1_train", f"[v4][{behavior}] {arm} train_metadata exists — skipping (resume)")
        return
    if ctx.dry_run:
        _dry_run_fake_ckpts(ft_root, ctx.arm_grid(arm), {"behavior": behavior, "arm": arm})
        _phase_log("p1_train", f"[v4][{behavior}] DRY-RUN: placeholder {arm} checkpoints")
        return
    train_jsonl = ctx.train_pool(behavior, arm)
    grid = ctx.arm_grid(arm)
    arm_lr = ctx.v4_arm_lr(arm)
    # WandB run suffix = the arm slug + source so each arm logs a DISTINCT run
    # (#480 run-separation class; plan §9 names issue642_v4_<slug>_villain).
    run_suffix = f"v4_{arm}_{V4_SOURCE_PERSONA}"

    def _ft_cmd(per_device: int, accum: int) -> list[str]:
        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            str(ACCEL_CONFIG),
            "--num_processes",
            str(ctx.n_gpus),
            "--gradient_accumulation_steps",
            str(accum),
            str(FT_TRAINER),
            "--behavior",
            behavior,
            "--arm",
            "cmft",  # the trainer arm token; the v4 SLUG distinguishes the run
            "--train-jsonl",
            str(train_jsonl),
            "--output-dir",
            str(ft_root),
            "--ckpt-steps",
            ",".join(map(str, grid)),
            "--seed",
            str(ctx.seed),
            "--learning-rate",
            str(arm_lr),
            "--per-device-batch",
            str(per_device),
            "--grad-accum",
            str(accum),
            "--wandb-project",
            V4_WANDB_PROJECT,
            "--freeze-outside-lora-modules",
            "--lora-adapter-config-json",
            str(ctx.adapter_config_path(behavior)),
            "--run-name-suffix",
            run_suffix,
        ]
        if ctx.ft_max_steps > 0:
            cmd += ["--max-steps", str(ctx.ft_max_steps)]
        return cmd

    ft_log = ctx.bdir(behavior) / "logs" / f"{arm}_train.log"
    rc = _run(_ft_cmd(4, 1), env=_ft_env(ctx), log_path=ft_log)
    if rc != 0:
        tail = ft_log.read_text(errors="replace")[-8000:] if ft_log.exists() else ""
        if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
            _phase_log(
                "p1_train",
                f"[v4][{behavior}] {arm} OOM at per-device 4 — fallback per-device 2 x accum 2",
            )
            shutil.rmtree(ft_root, ignore_errors=True)
            rc = _run(_ft_cmd(2, 2), env=_ft_env(ctx), log_path=ft_log)
        if rc != 0:
            raise RuntimeError(f"[v4][{behavior}] {arm} training failed rc={rc} (log: {ft_log})")


def _train_v4_lora_arm(ctx: Ctx, behavior: str, arm: str) -> None:
    """Train the v4 LoRA pole (loraOP_lr5e6) via the LoRA worker at the matched
    LR 5e-6 on the on-policy pool (plan v8 §4.2 — the --lr flag's purpose)."""
    lora_root = ctx.ckpt_root(behavior, arm)
    if (lora_root / "train_metadata.json").exists():
        _phase_log("p1_train", f"[v4][{behavior}] {arm} train_metadata exists — skipping (resume)")
        return
    if ctx.dry_run:
        _dry_run_fake_ckpts(lora_root, ctx.arm_grid(arm), {"behavior": behavior, "arm": arm})
        _phase_log("p1_train", f"[v4][{behavior}] DRY-RUN: placeholder {arm} checkpoints")
        return
    cmd = [
        sys.executable,
        str(LORA_TRAIN_WORKER),
        "--behavior",
        behavior,
        "--train-jsonl",
        str(ctx.train_pool(behavior, arm)),
        "--output-dir",
        str(lora_root),
        "--ckpt-steps",
        ",".join(map(str, ctx.arm_grid(arm))),
        "--seed",
        str(ctx.seed),
        "--gpu-id",
        "0",
        "--lr",
        str(ctx.v4_arm_lr(arm)),
        "--run-name-suffix",
        f"v4_{arm}_{V4_SOURCE_PERSONA}",
    ]
    # CVD pinned in the launcher env (gotcha #20) + matching --gpu-id 0; the 4
    # arms train SERIALLY so no parallel co-location.
    rc = _run(cmd, env=_gpu_env(0), log_path=ctx.bdir(behavior) / "logs" / f"{arm}_train.log")
    if rc != 0:
        raise RuntimeError(
            f"[v4][{behavior}] {arm} LoRA training failed rc={rc} "
            f"(log: {ctx.bdir(behavior) / 'logs' / f'{arm}_train.log'})"
        )


def _phase1_train_v4(ctx: Ctx, behavior: str) -> None:
    """v4 Phase 1: train the (up to) 3 production villain arms SERIALLY (each cmft
    arm uses all 4 GPUs via ZeRO-3; the LoRA pole uses GPU 0). Plan §9
    parallelism."""
    _phase_log("p1_train", f"[v4][{behavior}] Phase 1 start (arms={ctx.arms})")
    for arm in ctx.arms:
        if ctx.v4_arm_method(arm) == "lora":
            _train_v4_lora_arm(ctx, behavior, arm)
        else:
            _train_v4_cmft_arm(ctx, behavior, arm)
    _phase_log("p1_train", f"[v4][{behavior}] Phase 1 done")


def phase1_train(ctx: Ctx, behavior: str) -> None:
    if ctx.v4:
        _phase1_train_v4(ctx, behavior)
        return
    _phase_log("p1_train", f"[{behavior}] Phase 1 start (arms={ctx.arms})")
    train_jsonl = ctx.train_pool(behavior)

    # -- 1a LoRA trajectory (1 GPU subprocess; NOT on #642's production path) --
    lora_root = ctx.lora_ckpt_root(behavior)
    if "lora" not in ctx.arms:
        _phase_log("p1_train", f"[{behavior}] LoRA arm not requested (--arms) — skipping")
    elif (lora_root / "train_metadata.json").exists():
        _phase_log("p1_train", f"[{behavior}] LoRA train_metadata exists — skipping (resume)")
    elif ctx.dry_run:
        _dry_run_fake_ckpts(lora_root, ctx.lora_grid, {"behavior": behavior, "arm": "lora"})
        _phase_log("p1_train", f"[{behavior}] DRY-RUN: placeholder LoRA checkpoints written")
    else:
        cmd = [
            sys.executable,
            str(LORA_TRAIN_WORKER),
            "--behavior",
            behavior,
            "--train-jsonl",
            str(train_jsonl),
            "--output-dir",
            str(lora_root),
            "--ckpt-steps",
            ",".join(map(str, ctx.lora_grid)),
            "--seed",
            str(ctx.seed),
            "--gpu-id",
            "0",
        ]
        rc = _run(cmd, env=_gpu_env(0), log_path=ctx.bdir(behavior) / "logs" / "lora_train.log")
        if rc != 0:
            raise RuntimeError(
                f"[{behavior}] LoRA training failed rc={rc} "
                f"(log: {ctx.bdir(behavior) / 'logs' / 'lora_train.log'})"
            )

    # -- 1b ZeRO-3 FT-like arms (ft and/or cmft; #642 production trains cmft) --
    for arm in ("ft", "cmft"):
        if arm in ctx.arms:
            _train_ft_like(ctx, behavior, arm)
    _phase_log("p1_train", f"[{behavior}] Phase 1 done")


# ---------------------------------------------------------------------------
# Phase 2 — stage A: source-self trajectory + inline judging
# ---------------------------------------------------------------------------


def _judge_cell_file(ctx: Ctx, behavior: str, gen_json: Path, verdict_path: Path) -> dict:
    return judge_generation_file(gen_json, verdict_path, behavior=behavior, dry_run=ctx.dry_run)


def _stage_a_cells(ctx: Ctx, behavior: str) -> list[tuple[str, str, Path]]:
    """Enumerate stage-A cells: (cell_slug, arm, model_dir_or_marker).

    Only the arms this run trains (``ctx.arms``); the base cell is included
    only for non-follow-up runs.
    """
    cells: list[tuple[str, str, Path]] = []
    for arm in ctx.arms:
        # gen_arm = the generation MODE token: 'lora' for native-LoRA gen (the
        # LoRA pole), else the checkpoint-dir mode. In v4 the cell SLUG is the
        # arm slug itself; the gen_arm is 'lora' for the LoRA-method arm.
        is_lora = arm == "lora" or (ctx.v4 and ctx.v4_arm_method(arm) == "lora")
        root = (
            ctx.lora_ckpt_root(behavior)
            if (arm == "lora" and not ctx.v4)
            else ctx.ckpt_root(behavior, arm)
        )
        meta = json.loads((root / "train_metadata.json").read_text())
        gen_arm = "lora" if is_lora else ("cmft" if ctx.v4 else arm)
        for s in meta["saved_checkpoints"]:
            cells.append((f"{arm}_step{s}", gen_arm, root / f"checkpoint-{s}"))
    if ctx.run_label is None:
        cells.append(("base", "base", Path(BASE_MODEL)))
    return cells


def _parent_base_stage_a_cell(ctx: Ctx, behavior: str) -> dict:
    """Fetch the PARENT run's stage-A base trajectory cell (follow-up reuse)."""
    if ctx.dry_run:
        return {
            "arm": "base",
            "step": 0,
            "rate_raw": 0.0,
            "rate_clean": 0.0,
            "n_verdicts": 0,
            "n_degenerate": 0,
            "reused_from": "dry-run placeholder (no Hub fetch)",
        }
    from huggingface_hub import hf_hub_download

    parent_rel = f"{HF_EXPERIMENT_NAME}/{behavior}/stage_a/trajectory_{behavior}.json"
    got = _retry_transient(
        lambda: hf_hub_download(
            HF_DATA_REPO,
            parent_rel,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        ),
        what=f"fetch parent stage-A trajectory {parent_rel}",
    )
    parent_traj = json.loads(Path(got).read_text())
    if "base" not in parent_traj.get("cells", {}):
        raise RuntimeError(
            f"parent trajectory {parent_rel} has no base cell — cannot seed the "
            f"follow-up's s gauge (base reuse contract)"
        )
    base = dict(parent_traj["cells"]["base"])
    base["reused_from"] = parent_rel
    return base


def phase2_stage_a(ctx: Ctx, behavior: str) -> None:
    _phase_log("p2_stage_a", f"[{behavior}] Phase 2 start (source-self trajectory)")
    sa_dir = ctx.stage_a_dir(behavior)
    traj_path = sa_dir / f"trajectory_{behavior}.json"
    sa_dir.mkdir(parents=True, exist_ok=True)
    panel_json = sa_dir / "panel_source_self.json"
    panel_json.write_text(json.dumps({ctx.source_persona: ctx.source_prompt}))
    probes = ctx.eval_pool(behavior)
    cells = _stage_a_cells(ctx, behavior)

    # -- generation jobs (GPU queue) --
    jobs: list[tuple[str, list[str], Path]] = []
    lora_cells = [(slug, path) for slug, arm, path in cells if arm == "lora"]
    other_cells = [(slug, arm, path) for slug, arm, path in cells if arm != "lora"]
    gen_root = sa_dir / "gen"
    if lora_cells:
        n_shards = min(ctx.n_gpus, len(lora_cells))
        shards: list[list[tuple[str, Path]]] = [[] for _ in range(n_shards)]
        for i, lc in enumerate(lora_cells):
            shards[i % n_shards].append(lc)
        for shard_idx, shard in enumerate(shards):
            todo = [
                (slug, p)
                for slug, p in shard
                if not (gen_root / slug / "eval_summary.json").exists()
            ]
            if not todo:
                continue
            adapters = ",".join(f"{slug}={p}" for slug, p in todo)
            jobs.append(
                (
                    f"stage_a_lora_shard{shard_idx}",
                    [
                        sys.executable,
                        str(GEN_WORKER),
                        "--behavior",
                        behavior,
                        "--cell",
                        "stage_a",
                        "--seed",
                        str(ctx.seed),
                        "--lora-adapters",
                        adapters,
                        "--probes",
                        str(probes),
                        "--panel-json",
                        str(panel_json),
                        "--out-dir",
                        str(gen_root),
                        "--n-rollouts",
                        str(ctx.n_rollouts),
                        "--phase-tag",
                        "p2_stage_a",
                    ],
                    ctx.bdir(behavior) / "logs" / f"stage_a_lora_shard{shard_idx}.log",
                )
            )
    for slug, arm, path in other_cells:
        out_dir = gen_root / slug
        if (out_dir / "eval_summary.json").exists():
            continue
        model_args = (
            ["--hub-model-id", BASE_MODEL] if arm == "base" else ["--model-path", str(path)]
        )
        jobs.append(
            (
                f"stage_a_{slug}",
                [
                    sys.executable,
                    str(GEN_WORKER),
                    "--behavior",
                    behavior,
                    "--cell",
                    slug,
                    "--seed",
                    str(ctx.seed),
                    *model_args,
                    "--probes",
                    str(probes),
                    "--panel-json",
                    str(panel_json),
                    "--out-dir",
                    str(out_dir),
                    "--n-rollouts",
                    str(ctx.n_rollouts),
                    "--phase-tag",
                    "p2_stage_a",
                ],
                ctx.bdir(behavior) / "logs" / f"stage_a_{slug}.log",
            )
        )
    _run_gpu_queue(ctx, jobs)
    if ctx.dry_run:
        sys.path.insert(0, str(REPO / "scripts" / "issue_642"))
        from i642_gen_worker import load_probes, write_panel_outputs

        probe_rows = load_probes(probes)
        for slug, _arm, _path in cells:
            write_panel_outputs(
                gen_root / slug,
                behavior=behavior,
                cell=slug,
                seed=ctx.seed,
                panel_persona=ctx.source_persona,
                panel_prompt=ctx.source_prompt,
                probes=probe_rows,
                completions=[
                    ["DRY-RUN completion (not model output)."] * ctx.n_rollouts for _ in probe_rows
                ],
                metadata={"dry_run": True, "cell": slug},
            )

    # -- inline judging, trajectory persisted after EVERY cell --
    trajectory: dict = (
        json.loads(traj_path.read_text())
        if traj_path.exists()
        else {
            "behavior": behavior,
            "source": ctx.source_persona,
            "seed": ctx.seed,
            "smoke": ctx.smoke,
            "dry_run": ctx.dry_run,
            "v4": ctx.v4,
            "judge_model": JUDGE_MODEL,
            "cells": {},
        }
    )
    for slug, gen_arm, _path in cells:
        gen_json = gen_root / slug / f"{behavior}_eval_{ctx.source_persona}.json"
        if not gen_json.exists():
            raise RuntimeError(f"[{behavior}] stage-A generation missing: {gen_json}")
        cell = _judge_cell_file(
            ctx, behavior, gen_json, sa_dir / "verdicts" / f"{slug}__{ctx.source_persona}.json"
        )
        step = int(slug.split("step")[-1]) if "step" in slug else 0
        # The trajectory's ``arm`` key is the SELECTION-grouping arm. In v4 that
        # is the arm SLUG (3 distinct cmft arms must NOT collapse); v3 keeps the
        # gen-mode arm token. ``gen_arm`` rides along for stage-B model loading.
        arm_for_selection = _cell_arm_slug(slug) if ctx.v4 else gen_arm
        trajectory["cells"][slug] = {
            "arm": arm_for_selection,
            "gen_arm": gen_arm,
            "step": step,
            "rate_raw": cell["rate_raw"],
            "rate_clean": cell["rate_clean"],
            "n_verdicts": cell["n_verdicts"],
            "n_degenerate": cell["n_degenerate"],
        }
        trajectory["metadata"] = {
            "git_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        traj_path.write_text(json.dumps(trajectory, indent=2))
    if ctx.run_label is not None and "base" not in trajectory["cells"]:
        trajectory["cells"]["base"] = _parent_base_stage_a_cell(ctx, behavior)
        traj_path.write_text(json.dumps(trajectory, indent=2))
    base_clean = trajectory["cells"]["base"]["rate_clean"]
    for slug, rec in trajectory["cells"].items():
        if slug != "base":
            rec["s"] = (
                rec["rate_clean"] - base_clean
                if rec["rate_clean"] == rec["rate_clean"] and base_clean == base_clean
                else float("nan")
            )
    traj_path.write_text(json.dumps(trajectory, indent=2))
    _upload_dir_batched(ctx, sa_dir, "stage_a", behavior=behavior)
    _phase_log("p2_stage_a", f"[{behavior}] Phase 2 done (trajectory -> {traj_path})")


# ---------------------------------------------------------------------------
# Phase 3 — checkpoint selection + install gate (CPU)
# ---------------------------------------------------------------------------


def phase3_select(ctx: Ctx, behavior: str) -> None:
    _phase_log("p3_select", f"[{behavior}] Phase 3 start (selection + install gate)")
    sa_dir = ctx.stage_a_dir(behavior)
    sel_path = sa_dir / "selection.json"
    if sel_path.exists() and not (sa_dir / "install_failure.json").exists():
        _phase_log("p3_select", f"[{behavior}] selection.json exists — skipping (resume)")
        return
    trajectory = json.loads((sa_dir / f"trajectory_{behavior}.json").read_text())
    cells = trajectory["cells"]

    selection: dict = {
        "behavior": behavior,
        "arms": {},
        "smoke": ctx.smoke,
        "dry_run": ctx.dry_run,
        "run_label": ctx.run_label,
        "arms_requested": list(ctx.arms),
    }
    arm_ok: dict[str, bool] = {}
    for arm in ctx.arms:
        arm_cells = {slug: rec for slug, rec in cells.items() if rec["arm"] == arm}
        steps = sorted(rec["step"] for rec in arm_cells.values())
        s_by_step = {rec["step"]: rec.get("s", float("nan")) for rec in arm_cells.values()}
        degen_by_step = {
            rec["step"]: (rec["n_degenerate"] / rec["n_verdicts"] if rec["n_verdicts"] else 0.0)
            for rec in arm_cells.values()
        }
        s_values = [s_by_step[s] for s in steps]
        if any(v != v for v in s_values):  # NaN guard
            raise RuntimeError(f"[{behavior}/{arm}] NaN s values in trajectory — judging gap")
        # v4 secondary target = 0.65 (the on-policy dose band tops ~0.63, #612);
        # v3 keeps the default 0.75.
        sel = (
            select_checkpoints(steps, s_values, s_secondary=V4_S_SECONDARY)
            if ctx.v4
            else select_checkpoints(steps, s_values)
        )
        sel["degenerate_fraction_by_step"] = {str(k): v for k, v in degen_by_step.items()}
        s_ge_04 = [st for st in steps if s_by_step[st] >= 0.4]
        degen_collapse = bool(s_ge_04) and all(degen_by_step[st] > 0.5 for st in s_ge_04)
        sel["degenerate_collapse"] = degen_collapse
        bracket_ok = sel["bracket_pair"] is not None and all(
            degen_by_step[st] <= 0.5 for st in sel["bracket_pair"]
        )
        endpoint_ok = s_by_step[steps[-1]] >= S_TARGET
        arm_ok[arm] = (bracket_ok and not degen_collapse) or endpoint_ok
        sel["install_ok"] = arm_ok[arm]
        sel["bracket_ok"] = bracket_ok
        sel["endpoint_s"] = s_by_step[steps[-1]]
        selection["arms"][arm] = sel

    gate_pass = all(arm_ok.values())
    selection["install_gate_pass"] = gate_pass
    selection["s_target"] = S_TARGET
    selection["s_band"] = list(S_BAND)
    selection["metadata"] = {
        "git_commit_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    sel_path.write_text(json.dumps(selection, indent=2))
    _hub_upload_file(
        sel_path,
        f"{ctx.experiment_name}/{behavior}/stage_a/selection.json",
        skip=ctx.skip_upload,
    )
    # Record selected steps per trained arm for the reproducibility card.
    if ctx.v4:
        for arm in ctx.arms:
            if arm in selection["arms"]:
                ctx.cmft_selected_steps[f"{behavior}:{arm}"] = list(
                    selection["arms"][arm]["selected_steps"]
                )
    elif "cmft" in selection["arms"]:
        ctx.cmft_selected_steps[behavior] = list(selection["arms"]["cmft"]["selected_steps"])
    if not gate_pass:
        if ctx.smoke or ctx.dry_run:
            _phase_log(
                "p3_select",
                f"[{behavior}] install gate FAIL but smoke/dry-run — log-only, proceeding",
            )
        else:
            failure = {
                "behavior": behavior,
                "kill_criterion": "a_install_failure",
                "arm_ok": arm_ok,
                "note": (
                    "neither a non-degenerate s*=0.50 bracket nor endpoint s >= 0.50 "
                    "on at least one arm — comparison NOT run for this behavior "
                    "(plan §3 kill (a) / §7); fire the §4.11 fallback ladder "
                    "(band-entry read + nearest co-bracketed target + the one "
                    "pre-authorized lr-2e-6 cmft retrain) before declaring install failure"
                ),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
            fail_path = sa_dir / "install_failure.json"
            fail_path.write_text(json.dumps(failure, indent=2))
            _hub_upload_file(
                fail_path,
                f"{ctx.experiment_name}/{behavior}/stage_a/install_failure.json",
                skip=ctx.skip_upload,
            )
            ctx.killed[behavior] = "install_failure"
            _phase_log("p3_select", f"[{behavior}] INSTALL GATE FAIL — behavior killed")
            return

    # Delete non-selected FT/cmft checkpoints (production only; disk discipline).
    # v4: every trained arm slug; v3: the ft/cmft arms.
    delete_arms = list(ctx.arms) if ctx.v4 else list(FT_LIKE_ARMS)
    for arm in delete_arms:
        if arm in selection["arms"] and not (ctx.smoke or ctx.dry_run):
            keep = {f"checkpoint-{s}" for s in selection["arms"][arm]["selected_steps"]}
            root = ctx.ckpt_root(behavior, arm)
            for d in sorted(root.glob("checkpoint-*")):
                if d.name not in keep:
                    shutil.rmtree(d)
                    log.info("[%s] deleted non-selected %s checkpoint %s", behavior, arm, d.name)
    _phase_log("p3_select", f"[{behavior}] Phase 3 done (gate PASS={gate_pass})")


# ---------------------------------------------------------------------------
# Phase 4 — stage B: 39-persona panel generation for selected cells
# ---------------------------------------------------------------------------


def _stage_b_cells(ctx: Ctx, behavior: str) -> list[tuple[str, str, int]]:
    """Stage-B cells: (cell_slug, gen_arm, step). ``gen_arm`` is the generation
    MODE token phase4 branches on (lora -> merge; cmft/ft -> --model-path; base
    -> --hub-model-id). In v4 the LoRA-method arm maps to 'lora', the cmft arms
    to 'cmft'."""
    selection = json.loads((ctx.stage_a_dir(behavior) / "selection.json").read_text())
    cells: list[tuple[str, str, int]] = []
    for arm in ctx.arms:
        # gen_arm = the generation MODE token (v4 LoRA-method arm -> 'lora';
        # v4 cmft arms -> 'cmft'; v3 -> the arm token).
        if not ctx.v4:
            gen_arm = arm
        elif ctx.v4_arm_method(arm) == "lora":
            gen_arm = "lora"
        else:
            gen_arm = "cmft"
        for s in selection["arms"][arm]["selected_steps"]:
            cells.append((f"{arm}_step{s}", gen_arm, int(s)))
    if ctx.run_label is None:
        cells.append(("base", "base", 0))
    return cells


def _update_manifest(ctx: Ctx, behavior: str, cell: str, panels: dict[str, str]) -> None:
    path = ctx.manifest_path(behavior)
    manifest = json.loads(path.read_text()) if path.exists() else {"cells": {}}
    manifest["cells"][cell] = {
        "panels": sorted(panels),
        "n_rollouts": ctx.n_rollouts,
        "n_probes": ctx.n_probes or 50,
        "seed": ctx.seed,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    manifest["metadata"] = {
        "behavior": behavior,
        "git_commit_sha": _git_sha(),
        "smoke": ctx.smoke,
        "dry_run": ctx.dry_run,
        "base_model": BASE_MODEL,
        "temperature": 1.0,
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
    }
    path.write_text(json.dumps(manifest, indent=2))
    _hub_upload_file(
        path,
        f"{ctx.experiment_name}/{behavior}/generation_manifest.json",
        skip=ctx.skip_upload,
    )


def phase4_stage_b(ctx: Ctx, behavior: str) -> None:  # noqa: C901 - merge+gen+upload pipeline
    if behavior in ctx.killed:
        _phase_log("p4_stage_b", f"[{behavior}] killed ({ctx.killed[behavior]}) — skipping")
        return
    _phase_log("p4_stage_b", f"[{behavior}] Phase 4 start (39-persona stage B)")
    panels = ctx.stage_b_panel()
    panel_json = ctx.bdir(behavior) / "panel_stage_b.json"
    panel_json.parent.mkdir(parents=True, exist_ok=True)
    panel_json.write_text(json.dumps(panels, indent=2))
    probes = ctx.eval_pool(behavior)
    cells = _stage_b_cells(ctx, behavior)

    # -- merge selected LoRA checkpoints first (GPU queue; NOT on #642 prod) --
    merge_jobs: list[tuple[str, list[str], Path]] = []
    merged_dirs: dict[str, Path] = {}
    for cell, arm, step in cells:
        if arm != "lora":
            continue
        # v4 LoRA checkpoint lives under the arm-slug-keyed root; v3 under
        # lora_ckpts.
        lora_root = (
            ctx.ckpt_root(behavior, _cell_arm_slug(cell))
            if ctx.v4
            else ctx.lora_ckpt_root(behavior)
        )
        adapter_dir = lora_root / f"checkpoint-{step}"
        merged = ctx.bdir(behavior) / "merged" / cell
        merged_dirs[cell] = merged
        out_dir = ctx.gen_dir(behavior) / cell
        if (out_dir / "eval_summary.json").exists() or (merged / "config.json").exists():
            continue
        code = (
            "from explore_persona_space.train.sft import merge_lora; "
            f"merge_lora({BASE_MODEL!r}, {str(adapter_dir)!r}, {str(merged)!r}, gpu_id=0)"
        )
        merge_jobs.append(
            (
                f"merge_{cell}",
                [sys.executable, "-c", code],
                ctx.bdir(behavior) / "logs" / f"merge_{cell}.log",
            )
        )
    _run_gpu_queue(ctx, merge_jobs)

    # -- generation jobs --
    jobs: list[tuple[str, list[str], Path]] = []
    for cell, arm, step in cells:
        out_dir = ctx.gen_dir(behavior) / cell
        if (out_dir / "eval_summary.json").exists():
            _phase_log("p4_stage_b", f"[{behavior}] {cell}: eval_summary exists — resume skip")
            continue
        if arm == "lora":
            model_args = ["--model-path", str(merged_dirs[cell])]
        elif arm in FT_LIKE_ARMS:
            # v4 cmft checkpoint root is keyed by the arm SLUG (derived from the
            # cell), not the gen-mode 'cmft' token.
            ckpt_arm = _cell_arm_slug(cell) if ctx.v4 else arm
            model_args = [
                "--model-path",
                str(ctx.ckpt_root(behavior, ckpt_arm) / f"checkpoint-{step}"),
            ]
        else:
            model_args = ["--hub-model-id", BASE_MODEL]
        jobs.append(
            (
                f"stage_b_{cell}",
                [
                    sys.executable,
                    str(GEN_WORKER),
                    "--behavior",
                    behavior,
                    "--cell",
                    cell,
                    "--seed",
                    str(ctx.seed),
                    *model_args,
                    "--probes",
                    str(probes),
                    "--panel-json",
                    str(panel_json),
                    "--out-dir",
                    str(out_dir),
                    "--n-rollouts",
                    str(ctx.n_rollouts),
                    "--phase-tag",
                    "p4_stage_b",
                ],
                ctx.bdir(behavior) / "logs" / f"stage_b_{cell}.log",
            )
        )
    _run_gpu_queue(ctx, jobs)
    if ctx.dry_run:
        from i642_gen_worker import load_probes, write_panel_outputs

        probe_rows = load_probes(probes)
        for cell, _arm, _step in cells:
            for persona, prompt in panels.items():
                write_panel_outputs(
                    ctx.gen_dir(behavior) / cell,
                    behavior=behavior,
                    cell=cell,
                    seed=ctx.seed,
                    panel_persona=persona,
                    panel_prompt=prompt,
                    probes=probe_rows,
                    completions=[
                        ["DRY-RUN completion (not model output)."] * ctx.n_rollouts
                        for _ in probe_rows
                    ],
                    metadata={"dry_run": True, "cell": cell},
                )

    # -- per-cell coverage check + upload + manifest; drop merged dirs --
    for cell, arm, _step in cells:
        out_dir = ctx.gen_dir(behavior) / cell
        missing = [p for p in panels if not (out_dir / f"{behavior}_eval_{p}.json").exists()]
        if missing:
            raise RuntimeError(f"[{behavior}] stage-B cell {cell} missing panel files: {missing}")
        _update_manifest(ctx, behavior, cell, panels)
        _upload_dir_batched(ctx, out_dir, "generations", behavior=behavior)
        if arm == "lora" and not ctx.dry_run:
            merged = merged_dirs.get(cell)
            if merged and merged.exists():
                shutil.rmtree(merged)
                log.info("[%s] rmtree(%s)", behavior, merged)
    _phase_log("p4_stage_b", f"[{behavior}] Phase 4 done ({len(cells)} cells)")


# ---------------------------------------------------------------------------
# Phase 5 — uploads (cmft generations/trajectories; optional cmft adapters)
# ---------------------------------------------------------------------------


def _phase5_upload_v4_adapters(ctx: Ctx, behavior: str) -> None:
    """v4 adapter uploads (plan §10 Outputs): the LoRA pole's selected
    checkpoints upload to the HF model repo (canonical); the cmft consolidated
    bf16 dirs are opt-out by default (re-derivable from pinned data + commit +
    seed; enable with --upload-adapters). The reproducibility card records all
    destinations regardless."""
    if ctx.dry_run:
        return
    selection = json.loads((ctx.stage_a_dir(behavior) / "selection.json").read_text())
    for arm in ctx.arms:
        sel = selection["arms"].get(arm, {}).get("selected_steps", [])
        method = ctx.v4_arm_method(arm)
        # destination prefix per arm (LoRA canonical; cmft opt-out)
        dests = [
            f"adapters/issue_642/v4/{arm}_{V4_SOURCE_PERSONA}_seed{ctx.seed}/step{s}" for s in sel
        ]
        do_upload = method == "lora" or ctx.upload_adapters
        ctx.cmft_uploaded_adapters[f"{behavior}:{arm}"] = dests if do_upload else []
        if do_upload and not ctx.skip_upload:
            from huggingface_hub import HfApi

            _api = HfApi(token=os.environ.get("HF_TOKEN"))
            for s, dest in zip(sel, dests, strict=True):
                local = ctx.ckpt_root(behavior, arm) / f"checkpoint-{s}"
                if not local.exists():
                    raise RuntimeError(f"[v4][{behavior}] {arm} checkpoint {local} missing")
                _retry_transient(
                    lambda local=local, dest=dest, api=_api, arm=arm: api.upload_folder(
                        folder_path=str(local),
                        path_in_repo=dest,
                        repo_id=HF_MODEL_REPO,
                        repo_type="model",
                        commit_message=f"#642 v4 {arm} step {dest}",
                    ),
                    what=f"upload v4 adapter {dest}",
                )
                _phase_log("p5_upload", f"[v4][{behavior}] {arm} step {s} -> {dest}")
        else:
            _phase_log(
                "p5_upload",
                f"[v4][{behavior}] {arm} adapter upload opt-out "
                f"(method={method}); card declares {len(dests)} re-derivable dirs",
            )


def phase5_upload(ctx: Ctx, behavior: str) -> None:
    _phase_log("p5_upload", f"[{behavior}] Phase 5 start")
    if ctx.stage_a_dir(behavior).exists():
        _upload_dir_batched(ctx, ctx.stage_a_dir(behavior), "stage_a", behavior=behavior)
    if behavior in ctx.killed:
        _phase_log("p5_upload", f"[{behavior}] killed — stage_a artifacts uploaded; done")
        return
    if ctx.gen_dir(behavior).exists():
        _upload_dir_batched(ctx, ctx.gen_dir(behavior), "generations", behavior=behavior)

    if ctx.v4:
        _phase5_upload_v4_adapters(ctx, behavior)
        _phase_log("p5_upload", f"[v4][{behavior}] Phase 5 done")
        return

    # Optional cmft consolidated-bf16 adapter uploads (plan §10 Outputs: default
    # opt-out, re-derivable from pinned data + commit + seed; enable with
    # --upload-adapters). The reproducibility card records the destinations
    # regardless so the verifier can resolve them when uploaded.
    if not ctx.dry_run and "cmft" in ctx.arms:
        selection = json.loads((ctx.stage_a_dir(behavior) / "selection.json").read_text())
        cmft_sel = selection["arms"].get("cmft", {}).get("selected_steps", [])
        label_prefix = f"{ctx.run_label}_" if ctx.run_label else ""
        dests = [f"adapters/issue_642/{label_prefix}{behavior}_cmft_step{s}" for s in cmft_sel]
        ctx.cmft_uploaded_adapters[behavior] = dests if ctx.upload_adapters else []
        if ctx.upload_adapters and not ctx.skip_upload:
            from huggingface_hub import HfApi

            api = HfApi(token=os.environ.get("HF_TOKEN"))
            for s, dest in zip(cmft_sel, dests, strict=True):
                local = ctx.cmft_ckpt_root(behavior) / f"checkpoint-{s}"
                if not local.exists():
                    raise RuntimeError(f"[{behavior}] cmft checkpoint {local} missing for upload")
                _retry_transient(
                    lambda local=local, dest=dest: api.upload_folder(
                        folder_path=str(local),
                        path_in_repo=dest,
                        repo_id=HF_MODEL_REPO,
                        repo_type="model",
                        commit_message=f"#642 {behavior} cmft consolidated step {dest}",
                    ),
                    what=f"upload cmft adapter {dest}",
                )
                _phase_log("p5_upload", f"[{behavior}] cmft step {s} -> {dest}")
        else:
            _phase_log(
                "p5_upload",
                f"[{behavior}] cmft adapter upload opt-out (default); card declares "
                f"{len(dests)} re-derivable consolidated dirs",
            )
    _phase_log("p5_upload", f"[{behavior}] Phase 5 done")


# ---------------------------------------------------------------------------
# Sentinel + main
# ---------------------------------------------------------------------------


def _wandb_entity() -> str | None:
    """Read the WandB entity off the SDK at run time (NOT a hand-typed literal —
    a stale literal silently breaks resolution when the account changes; #597).
    Returns None if WandB is unavailable; the card omits the field then."""
    try:
        import wandb

        return wandb.Api().default_entity
    except Exception as e:
        log.warning("could not read WandB default_entity for the card: %s", e)
        return None


def _reproducibility_card_v4(ctx: Ctx) -> dict:
    """v4 training-task reproducibility card (CLAUDE.md sentinel contract): the
    4 villain arms each train a DISTINCT WandB run (#480 run-separation). Carries
    wandb_project + wandb_run_names + wandb_entity + per-arm adapter paths. The
    LoRA pole adapters upload canonically; the cmft consolidated dirs are
    opt-out (declared, re-derivable). NO #606 reuse."""
    run_names = [f"issue642_v4_{arm}_{V4_SOURCE_PERSONA}_seed{ctx.seed}" for arm in ctx.arms]
    adapter_paths: dict[str, list[str]] = {}
    for key, dests in ctx.cmft_uploaded_adapters.items():
        if dests:
            adapter_paths[key] = [f"{HF_MODEL_REPO} :: {d}" for d in dests]
    return {
        "v4": True,
        "arms_trained": list(ctx.arms),
        "source_persona": V4_SOURCE_PERSONA,
        "wandb_project": V4_WANDB_PROJECT,
        "wandb_run_names": run_names,
        "wandb_entity": _wandb_entity(),
        "selected_steps": ctx.cmft_selected_steps,  # keyed "<behavior>:<arm>"
        "adapter_paths": adapter_paths,  # LoRA pole canonical; cmft opt-out
        "lora_pole_upload": (
            f"{HF_MODEL_REPO} :: adapters/issue_642/v4/loraOP_lr5e6_{V4_SOURCE_PERSONA}"
            f"_seed{ctx.seed}/step{{selected}}"
        ),
        "cmft_adapters_note": (
            "cmft consolidated bf16 dirs opt-out from upload (plan §10) — "
            "re-derivable from pinned data sha + git commit + seed; enable with "
            "--upload-adapters"
        ),
        "onpolicy_pool": f"{HF_DATA_REPO} :: {V4_ONPOLICY_POOL_HUB_PATH}",
        "canned_pool_source": f"{HF_DATA_REPO} :: {V4_CANNED_POOL_HUB_PATH}",
        "panel": f"{HF_DATA_REPO} :: {V4_PANEL_SET_HUB_PATH}",
        "contrasts": "delta_rank_matched, delta_data (within-villain; plan v8 §5; delta_lr gone)",
    }


def _reproducibility_card(ctx: Ctx) -> dict:
    """CLAUDE.md training-task sentinel contract: per-cell adapter_paths +
    wandb_run_names with the wandb_project. The cmft arm trains adapters; the
    LoRA/FT poles are REUSED from #606 (declared as such)."""
    if ctx.v4:
        return _reproducibility_card_v4(ctx)
    cmft_adapters: dict[str, list[str]] = {}
    for behavior in ctx.behaviors:
        if ctx.cmft_uploaded_adapters.get(behavior):
            cmft_adapters[behavior] = [
                f"{HF_MODEL_REPO} :: {d}" for d in ctx.cmft_uploaded_adapters[behavior]
            ]
    return {
        "arm_trained": "cmft",
        "wandb_project": WANDB_PROJECT,
        "wandb_run_names": [f"issue642_cmft_{b}_seed{ctx.seed}" for b in ctx.behaviors],
        "cmft_selected_steps": ctx.cmft_selected_steps,
        "cmft_adapter_paths": cmft_adapters,
        "cmft_adapters_uploaded": bool(ctx.upload_adapters),
        "cmft_adapters_note": (
            "cmft consolidated bf16 dirs are opt-out from upload (plan §10) — "
            "re-derivable from pinned data sha + git commit + seed; enable with "
            "--upload-adapters"
        ),
        "reused_lora_pole": (
            f"{HF_DATA_REPO}@{DATA_REVISION_DEFAULT} :: "
            f"{PARENT_EXPERIMENT_NAME}/<behavior>/generations/lora_step{{28,32,36,132}}"
        ),
        "reused_ft_pole": (
            f"{HF_DATA_REPO}@{DATA_REVISION_DEFAULT} :: "
            f"{PARENT_EXPERIMENT_NAME}/<behavior>/generations/ft_step{{12,16,22,132}}"
        ),
        "reused_lora_runs": "issue606 LoRA/FT WandB runs (l8yczo60 / rtod8ntu)",
        "data_revision": DATA_REVISION_DEFAULT,
        "model_revision": MODEL_REVISION_DEFAULT,
    }


def _write_results_sentinel(ctx: Ctx, phases_run: list[str], note: str) -> Path:
    """poll_pipeline.py end-of-run sentinel (_SENTINEL_REQUIRED_KEYS contract:
    sentinel_schema_version=1, kind, version)."""
    sentinel = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 642,
        "by": "i642_dispatch",
        "ts": datetime.now(UTC).isoformat(),
        "note": note,
        "reproducibility_card": _reproducibility_card(ctx),
        "payload_extra": {
            "phases_run": phases_run,
            "behaviors": ctx.behaviors,
            "arms": list(ctx.arms),
            "run_label": ctx.run_label,
            "ft_lr": ctx.ft_lr,
            "ft_grid": list(ctx.ft_grid),
            "cmft_grid": list(ctx.cmft_grid),
            "killed": ctx.killed,
            "smoke": ctx.smoke,
            "dry_run": ctx.dry_run,
            "output_root": str(ctx.output_root),
            "git_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
        },
    }
    path = ctx.sentinel_dir / f"issue-642-epm_results-{int(time.time())}.json"
    path.write_text(json.dumps(sentinel, indent=2))
    return path


PHASES = {
    "p0_data": phase0_data,
    "p1_train": phase1_train,
    "p2_stage_a": phase2_stage_a,
    "p3_select": phase3_select,
    "p4_stage_b": phase4_stage_b,
    "p5_upload": phase5_upload,
}
PHASE_ORDER = list(PHASES)


# ---------------------------------------------------------------------------
# Import-completeness verification (--verify-imports; CPU-only, no GPU/API)
# ---------------------------------------------------------------------------

DEFERRED_IMPORT_SCOPE: tuple[Path, ...] = (
    REPO / "scripts" / "issue_642" / "i642_common.py",
    REPO / "scripts" / "issue_642" / "i642_dispatch.py",
    REPO / "scripts" / "issue_642" / "i642_gen_worker.py",
    REPO / "scripts" / "issue_642" / "i642_lora_train_worker.py",
    REPO / "scripts" / "issue_642" / "i642_analyze.py",
    REPO / "scripts" / "issue_642" / "i642_figures.py",
    REPO / "scripts" / "train_behavior_fullft.py",
)


def verify_deferred_imports(scope: tuple[Path, ...] = DEFERRED_IMPORT_SCOPE) -> int:
    """Execute EVERY lazy (non-top-level) import across the #642 code path.

    AST-scans each in-scope file for Import/ImportFrom nodes nested below module
    top level, then actually imports each module and resolves each named symbol.
    Catches the deferred-symbol crash class (a lazy import whose branch is
    skipped by local --dry-run/--skip-upload smokes). Self-maintaining; returns
    the number of (module, symbol) checks; raises listing ALL failures.
    """
    import ast
    import importlib

    checked = 0
    failures: list[str] = []
    for path in scope:
        tree = ast.parse(path.read_text(), filename=str(path))
        top_level = set(tree.body)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom) or node in top_level:
                continue
            where = f"{path.name}:{node.lineno}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    checked += 1
                    try:
                        importlib.import_module(alias.name)
                    except Exception as e:
                        failures.append(f"{where}: import {alias.name} -> {type(e).__name__}: {e}")
                continue
            if node.module is None or node.level > 0:
                failures.append(f"{where}: relative deferred import unsupported by verifier")
                continue
            try:
                mod = importlib.import_module(node.module)
            except Exception as e:
                failures.append(f"{where}: from {node.module} -> {type(e).__name__}: {e}")
                continue
            for alias in node.names:
                checked += 1
                if alias.name == "*" or hasattr(mod, alias.name):
                    continue
                try:  # submodule not re-exported as an attribute of the parent
                    importlib.import_module(f"{node.module}.{alias.name}")
                except Exception as e:
                    failures.append(
                        f"{where}: from {node.module} import {alias.name} -> "
                        f"{type(e).__name__}: {e}"
                    )
    if failures:
        raise RuntimeError(
            "verify-imports FAILED — deferred symbols unresolvable on this checkout:\n  "
            + "\n  ".join(failures)
        )
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="#642 coverage-matched-FT dispatcher (phases 0-5; --smoke = one tiny cell).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--behaviors", default="sycophancy")
    parser.add_argument(
        "--v4",
        action="store_true",
        help="Followup `onpolicy-matchedlr-rank-isolation` mode (plan v8): train the "
        "THREE villain arms at matched LR 5e-6 on on-policy data + read on the #612 "
        "30-panel; NO #606 reuse. --arms then takes the v4 slugs "
        "(loraOP_lr5e6,cmftOP_lr5e6,cmftCN_lr5e6; default = all 3).",
    )
    parser.add_argument(
        "--install-pilot",
        action="store_true",
        help="v4 Phase-0.5 install-pilot (plan §4.6/§7): train each arm a SHORT way "
        "(max-steps = max pilot grid) + a coarse stage-A read; gates the full train. "
        "Use with --v4 --stop-after-phase p3_select.",
    )
    parser.add_argument(
        "--arms",
        default="",
        help="Which arms this run trains+evals. v3: subset of lora,ft,cmft (default cmft). "
        "v4 (--v4): subset of the 3 production villain slugs "
        "(loraOP_lr5e6,cmftOP_lr5e6,cmftCN_lr5e6; default = all 3); evidence_only 1e-5 "
        "slugs are refused.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Retrain/fallback run label (e.g. cmft-lr2e6-retrain): scopes the Hub "
        "prefix to <experiment>/<label>/ (plan §4.11/§7/§13 lr-2e-6 lever).",
    )
    parser.add_argument("--seeds", default=str(SEED), help="single seed (inherited regime)")
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/issue_642"))
    parser.add_argument("--smoke", action="store_true", help="One tiny cell through every phase.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only chaining check: skip GPU training/generation + API judging.",
    )
    parser.add_argument(
        "--resume-from-phase",
        default="auto",
        choices=["auto", *PHASE_ORDER],
        help="auto = per-phase output-presence skip; a phase name skips everything before it.",
    )
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument(
        "--ft-lr",
        type=float,
        default=FT_LR,
        help=f"FT/cmft learning rate (default {FT_LR}; the ONE pre-authorized retrain lever "
        f"per plan §4.11/§7/§13 is {FT_RETRAIN_LR} — relaunch p1_train with --ft-lr 2e-6 "
        "after deleting the behavior's cmft_ckpts dir).",
    )
    parser.add_argument(
        "--ft-grid",
        default=None,
        help="cmft/FT checkpoint grid override: comma list of optimizer steps, or the "
        f"keyword 'retrain' = the densified FT_RETRAIN_GRID {FT_RETRAIN_GRID} "
        f"(default: the cmft grid {CMFT_CKPT_GRID}).",
    )
    parser.add_argument("--data-revision", default=DATA_REVISION_DEFAULT)
    parser.add_argument(
        "--model-revision",
        default=MODEL_REVISION_DEFAULT,
        help="MODEL-repo revision for the #606 LoRA adapter_config (cmft mask-identity assert).",
    )
    parser.add_argument("--skip-upload", action="store_true", help="Local-only (never on a pod).")
    parser.add_argument(
        "--upload-adapters",
        action="store_true",
        help="Upload the selected cmft consolidated bf16 dirs to the HF model repo "
        "(default opt-out per plan §10 — re-derivable from pinned data + commit + seed).",
    )
    parser.add_argument("--hf-experiment-name", default=HF_EXPERIMENT_NAME)
    parser.add_argument(
        "--verify-imports",
        action="store_true",
        help="Execute every deferred import across the #642 scripts, then exit (no GPU/API).",
    )
    parser.add_argument(
        "--stop-after-phase",
        default=None,
        choices=PHASE_ORDER,
        help="Run up to and including this phase, then write the sentinel and exit "
        "(VM-side phase-0 smoke uses p0_data).",
    )
    args = parser.parse_args(argv)
    if args.verify_imports:
        n = verify_deferred_imports()
        print(
            f"verify-imports OK: {n} deferred import symbols verified across "
            f"{len(DEFERRED_IMPORT_SCOPE)} files",
            flush=True,
        )
        return 0
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if seeds != [SEED]:
        raise ValueError(f"#642 is a single-seed design (seed {SEED}); got {seeds}")
    args.seed = seeds[0]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=dispatch] %(message)s")
    ctx = Ctx(args)
    start_idx = 0 if ctx.resume_from == "auto" else PHASE_ORDER.index(ctx.resume_from)
    phase_keys = PHASE_ORDER[start_idx:]
    if args.stop_after_phase is not None:
        if args.stop_after_phase not in phase_keys:
            raise ValueError(
                f"--stop-after-phase {args.stop_after_phase} precedes --resume-from-phase"
            )
        phase_keys = phase_keys[: phase_keys.index(args.stop_after_phase) + 1]
    _phase_log(
        "dispatch",
        f"behaviors={ctx.behaviors} arms={ctx.arms} run_label={ctx.run_label} "
        f"ft_lr={ctx.ft_lr} cmft_grid={ctx.cmft_grid} phases={phase_keys} smoke={ctx.smoke} "
        f"dry_run={ctx.dry_run} gpus={ctx.n_gpus} out={ctx.output_root}",
    )
    done: list[str] = []
    for behavior in ctx.behaviors:
        for key in phase_keys:
            PHASES[key](ctx, behavior)
            done.append(f"{behavior}:{key}")
            if behavior in ctx.killed and key == "p3_select":
                phase5_upload(ctx, behavior)
                done.append(f"{behavior}:p5_upload")
                break
    sentinel = _write_results_sentinel(
        ctx,
        done,
        f"#642 dispatcher completed phases {done} (smoke={ctx.smoke}, "
        f"dry_run={ctx.dry_run}, killed={ctx.killed}); cmft stage-B generations + "
        f"trajectory uploaded under {ctx.experiment_name}; LoRA/FT poles reused "
        f"from #606; judging + 3-arm decomposition run VM-side (i642_analyze.py).",
    )
    _phase_log("done", f"all phases complete; sentinel -> {sentinel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
