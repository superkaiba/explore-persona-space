#!/usr/bin/env python3
"""Task #612 dispatcher — UNIFIED smoke = sweep with one cell.

Cell grammar ``<source>:<arm>:<seed>`` (28-cell production grid):

    villain:arm_onpolicy:42       train cell (the SMOKE cell — G1/G2 fire after it)
    villain:arm_canned:137        train cell (frozen-pool replication anchor)
    panel:build:0                 P1 candidate centroids + bank parity + cosines
    base:pass:0                   P2 base-model pass over ALL candidates
    villain:parity:42             P5 frozen-#411-adapter parity anchor

Smoke == sweep with one cell: ``--cells villain:arm_onpolicy:42`` runs
prefetch -> P3 pool build -> P4 train/merge/eval/trajectory/upload through the
SAME ``_run_cell`` path as production. EVERY phase's cell list derives from
``--cells``: prefetch fetches only the requested cells' inputs, P1/P2 run iff
their special cells are requested, P3+P4 run inside ``_run_cell`` per train
cell, P5 per parity cell, and the G1/G2 gates fire iff the smoke cell is in
``--cells``.

``epm:results`` gate (ported #608 round-2 binding): non-dry runs default
``--all-cells`` to the FULL 28-cell grid; ``finalize`` emits ``epm:results``
ONLY when (a) not a dry run, (b) the aggregate cell list equals the grid
(hard assert), and (c) every grid cell has a ``complete`` cell-state record.
Everything else writes ``epm:progress`` shard sentinels.

Per train cell (sequential within one dispatcher process):
    1. [phase=pool_build]  arm_canned -> the prefetched frozen #411 pool;
       arm_onpolicy/arm_prefix -> build_onpolicy_pool in a FRESH SUBPROCESS
       (vLLM); PositiveYieldError surfaces as exit code 42 -> G3 drop rule.
    2. [phase=train]       train_lora, #411 recipe held fixed (plan §4 cfg);
       max_length=2048, save_strategy=epoch, save_only_model=True; HF uploads
       owned by the dispatcher (fail-loud, checkpoint-* excluded from the
       final-adapter upload; epoch-1/2 checkpoints uploaded separately).
    3. [phase=merge]       merge_lora -> merged dir.
    4. [phase=eval]        eval_panel in a FRESH SUBPROCESS (vLLM teardown
       gotcha) on the audited 60 claims x resolved panel -> rmtree merged.
    5. [phase=trajectory]  per epoch-1/2 checkpoint: merge -> own-panel eval
       (fresh subprocess) -> rmtree.
    6. [phase=upload]      cell eval tree (incl. raw_completions/) -> HF data
       repo BEFORE pod termination (Upload Policy).
    7. Per-cell sentinel (poll_pipeline-conforming) + cell-state record.

Panel resolution order for evals: local ``panel_set.json`` (P2j output,
committed/downloaded) -> HF ``issue612_sycophancy_onpolicy/panel/panel_set.json``
-> provisional mandatory-11 panel iff ``--allow-provisional-panel`` (smoke),
else fail loud. The production driver polls HF for the P2j panel before
launching train shards (cross-phase contract).

Pod-side discipline: NEVER calls scripts/task.py; sentinels only (all keys
poll_pipeline requires); the ``[phase=done]`` token appears EXACTLY ONCE as
the terminal line of a clean exit; ``load_dotenv()`` at module top; every
``subprocess.*`` gets an explicit ``env={**os.environ, ...}``;
``EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1``; CVD via ``TrainLoraConfig(gpu_id)``
/ ``merge_lora(gpu_id)`` in-process and ``CUDA_VISIBLE_DEVICES`` in
subprocess envs (the sft.py CVD-clobber gotcha).

``--stage dose-matched`` (plans/v2.md, followup dose-matched-leakage-read):
eval-only round on the parent's OWN epoch checkpoints at their pre-registered
band-entry epochs. No training. Per cell from ``band_entry_selection.json``
(written/asserted by ``band_entry.ensure_band_entry_selection`` — K3-dm):

    1. [phase=dose_fetch]  per-file HF fetch of the ``{source}_seed{seed}``
       subtree (final dir + the selected checkpoint) at the PINNED model-repo
       revision; K2-dm hard assert on checkpoint-epoch{E}/adapter_config.json
       + adapter_model.safetensors (never substitute another epoch);
       ``_ensure_tokenizer_files`` from the final-adapter dir.
    2. [phase=merge]       merge_lora on the checkpoint.
    3. [phase=eval]        FULL-panel ``_eval_subprocess`` on the sha-pinned
       panel_set + eval_60 (dose_prefetch, plan v2 rule (f)) -> rmtree merged.
    4. [phase=upload]      per-cell tree -> HF data repo INSIDE the loop.
    5. Per-cell ``issue-612-dose-*`` sentinel + dose cell-state record.

Smoke == sweep with one cell: ``--cells villain:arm_canned:42`` runs the full
fetch->merge->eval->upload path; the G1-dm parity gate (pod-side ~600-call
mini-judge of the self panel on the full 60-claim set vs the pinned trajectory
reference, ±0.06, ONE diagnostic re-fetch+re-merge retry — K1-dm) fires after
it and blocks the 7-cell launch on FAIL. The dose ``epm:results`` sentinel
(version 3 — the parent posted v2) is emitted only by a ``--finalize``
invocation that sees all 8 registered cells complete.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    ANALYZE_SUMMARY_RELPATH,
    BAND_ENTRY_THRESHOLD,
    BASE_MODEL,
    DOSE_ADAPTER_PATH_TMPL,
    DOSE_ADAPTER_REVISION,
    DOSE_MATCHED_SHA256,
    G1_DM_TOL,
    G1_TOL,
    G2_DELTA_FLOOR,
    HF_DATA_PREFIX,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    JUDGE_MODEL,
    MANDATORY_PANEL,
    PARITY_PANELS,
    TRAIN_ARMS,
    V3_HF_DATA_PREFIX,
    V3_TRAIN_ARMS,
    cell_id,
    cell_slab_dir,
    dose_cell_dir,
    full_production_cells,
    parse_cells,
    pool_dir,
    repo_root_from_module,
    v3_cell_dir,
)
from explore_persona_space.experiments.sycophancy_onpolicy_612.band_entry import (  # noqa: E402
    SELECTION_RELPATH,
    ensure_band_entry_selection,
)

log = logging.getLogger("dispatch_sycophancy_612")

SMOKE_CELL = ("villain", "arm_onpolicy", 42)
DOSE_SMOKE_CELL = ("villain", "arm_canned", 42)
V3_SMOKE_CELL = ("villain", "arm_onpolicy", 42)  # predictor-v3 smoke cell
DOSE_REQUIRED_CKPT_FILES = ("adapter_config.json", "adapter_model.safetensors")
DOSE_RESULTS_MARKER_VERSION = 3  # the parent run posted epm:results v2
V3_RESULTS_MARKER_VERSION = 1  # predictor-v3 is a fresh epm:results stream
SMOKE_DIAGNOSTIC_CELL = ("villain", "arm_onpolicy", 137)
YIELD_EXIT_CODE = 42  # build_onpolicy_pool's PositiveYieldError marker (G3)
JUDGE_PARSE_RATE_FLOOR = 0.95
NONEMPTY_COMPLETION_FLOOR = 0.95
SMOKE_JUDGE_API_ERROR_CEILING = 0.02
TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def resolve_all_cells(
    cells: list[tuple[str, str, int]],
    *,
    all_cells_arg: list[tuple[str, str, int]] | None,
    dry_run: bool,
) -> list[tuple[str, str, int]]:
    """Non-dry runs DEFAULT to the full 28-cell grid (a mislaunched subset shard
    can never satisfy the epm:results gate); dry runs default to their subset."""
    if all_cells_arg is not None:
        return all_cells_arg
    if dry_run:
        return list(cells)
    return full_production_cells()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _cellstate_path(slab_root: Path, source: str, arm: str, seed: int) -> Path:
    return slab_root / "_cellstate" / f"{source}__{arm}__{seed}.json"


def _write_cellstate(slab_root: Path, source: str, arm: str, seed: int, record: dict) -> None:
    path = _cellstate_path(slab_root, source, arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def _read_cellstate(slab_root: Path, source: str, arm: str, seed: int) -> dict | None:
    path = _cellstate_path(slab_root, source, arm, seed)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_sentinel(
    logs_root: Path,
    *,
    kind: str,
    note_obj: dict,
    name_slug: str,
    gate: str | None = None,
    version: int = 1,
) -> Path:
    """One poll_pipeline-conforming sentinel (sentinel_schema_version/kind/version)."""
    logs_root.mkdir(parents=True, exist_ok=True)
    path = logs_root / f"issue-612-{name_slug}-{int(time.time())}.json"
    payload: dict = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "task_id": 612,
        "by": "pod-dispatcher-612",
        "ts": datetime.now(UTC).isoformat(),
        "note": json.dumps(note_obj, ensure_ascii=False),
    }
    if gate is not None:
        payload["gate"] = gate
        payload["blocks_pipeline"] = True
    path.write_text(json.dumps(payload, indent=2))
    log.info("sentinel written: %s (kind=%s)", path, kind)
    return path


def _upload_or_raise(local_path: Path, *, repo_type: str, repo_id: str, path_in_repo: str) -> str:
    """Fail-loud wrapper around hub._upload (which returns '' on failure)."""
    from explore_persona_space.orchestrate.hub import _upload

    ignore = ["checkpoint-*"] if local_path.is_dir() else None
    hub_path = _upload(
        local_path=local_path,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        upload_as_file=local_path.is_file(),
        ignore_patterns=ignore,
    )
    if not hub_path:
        raise RuntimeError(
            f"HF upload FAILED: {local_path} -> {repo_id}/{path_in_repo} "
            f"(hub._upload returned empty path; check HF_TOKEN / quota / network)"
        )
    return hub_path


def _upload_checkpoint_or_raise(ckpt_dir: Path, path_in_repo: str) -> str:
    """Upload one epoch checkpoint dir WITHOUT the checkpoint-* exclusion."""
    from explore_persona_space.orchestrate.hub import _upload

    hub_path = _upload(
        local_path=ckpt_dir,
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        path_in_repo=path_in_repo,
        upload_as_file=False,
    )
    if not hub_path:
        raise RuntimeError(f"HF checkpoint upload FAILED: {ckpt_dir} -> {path_in_repo}")
    return hub_path


def _resolve_epoch_checkpoints(adapter_dir: Path) -> dict[int, Path]:
    """Map epoch -> checkpoint dir (save_strategy='epoch' x 3 epochs => 3 dirs)."""
    ckpts = sorted(
        (int(p.name.split("-")[1]), p) for p in adapter_dir.glob("checkpoint-*") if p.is_dir()
    )
    if len(ckpts) != 3:
        raise RuntimeError(
            f"Expected exactly 3 epoch checkpoints under {adapter_dir}, found "
            f"{[p.name for _, p in ckpts]}"
        )
    return {1: ckpts[0][1], 2: ckpts[1][1], 3: ckpts[2][1]}


def _resolve_step_checkpoints(adapter_dir: Path) -> dict[int, Path]:
    """Map optimizer-step -> checkpoint dir (save_strategy='steps' => N dirs).

    Predictor-v3 (plan v3 §4.3): sub-epoch ``checkpoint-<step>`` dirs. Returns
    {step: dir} sorted ascending; raises if fewer than 2 (the band-entry read
    needs >=2 candidates to bracket the +0.60 crossing, §4.3 / §7)."""
    ckpts = sorted(
        (int(p.name.split("-")[1]), p) for p in adapter_dir.glob("checkpoint-*") if p.is_dir()
    )
    if len(ckpts) < 2:
        raise RuntimeError(
            f"Expected >=2 sub-epoch checkpoints under {adapter_dir}, found "
            f"{[p.name for _, p in ckpts]} — save_strategy='steps' did not bracket the band"
        )
    return {step: p for step, p in ckpts}


def _ensure_tokenizer_files(ckpt_dir: Path, adapter_dir: Path) -> None:
    """Epoch checkpoints may lack tokenizer files; merge_lora loads the tokenizer
    from adapter_path, so copy them in from the final adapter dir."""
    if (ckpt_dir / "tokenizer_config.json").exists():
        return
    for name in TOKENIZER_FILES:
        src = adapter_dir / name
        if src.exists():
            shutil.copy2(src, ckpt_dir / name)
    if not (ckpt_dir / "tokenizer_config.json").exists():
        raise RuntimeError(
            f"No tokenizer_config.json in {ckpt_dir} and none to copy from {adapter_dir}"
        )


def _loss_curve_report(adapter_dir: Path) -> dict:
    """G2 diagnostic: trainer_state log_history NaN-free and decreasing first->last."""
    ckpts = _resolve_epoch_checkpoints(adapter_dir)
    state_path = ckpts[3] / "trainer_state.json"
    if not state_path.exists():
        raise RuntimeError(f"trainer_state.json missing in final checkpoint {ckpts[3]}")
    state = json.loads(state_path.read_text())
    losses = [h["loss"] for h in state.get("log_history", []) if "loss" in h]
    if not losses:
        raise RuntimeError(f"No loss entries in {state_path} log_history")
    return {
        "n_loss_points": len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "has_nan": any(math.isnan(x) for x in losses),
        "decreasing_first_to_last": losses[0] > losses[-1],
    }


class Dispatcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.gpu_id: int = args.gpu_id
        self.data_root: Path = args.data_root
        self.adapters_root: Path = args.adapters_root
        self.slab_root: Path = args.slab_root
        self.runs_root: Path = args.runs_root
        self.logs_root: Path = args.logs_root
        self.dry_run: bool = args.dry_run
        self.hf_upload: bool = args.hf_upload and not args.dry_run
        self._panel_set_path: Path | None = None

    # ----- panel resolution ---------------------------------------------------

    def _build_provisional_panel(self) -> Path:
        """Mandatory-11 provisional panel for the SMOKE cell (P2j panel absent).

        Production cells refuse this path unless --allow-provisional-panel is
        set; the eval metadata records panel provenance either way."""
        from explore_persona_space.experiments.factor_screen_365.persona_panel import (
            EVAL_PERSONAS_24,
        )
        from explore_persona_space.experiments.sycophancy_onpolicy_612.panel_build import (
            load_i591_accepted,
        )

        i591 = load_i591_accepted(self.data_root / "i591" / "twin_validation.json")
        personas: dict[str, dict] = {}
        for name in MANDATORY_PANEL:
            if name in EVAL_PERSONAS_24:
                personas[name] = {"prompt": EVAL_PERSONAS_24[name], "provenance": "roster_24"}
            elif name in i591:
                personas[name] = {"prompt": i591[name], "provenance": "i591_twin"}
            else:
                raise KeyError(f"mandatory panel persona {name!r} unresolvable")
        path = self.data_root / "panel" / "panel_set_provisional.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "provenance": "provisional_mandatory11",
                    "personas": personas,
                },
                indent=2,
            )
        )
        return path

    def _resolve_panel_set(self) -> Path:
        """panel_set.json: local (P2j output via git) -> HF -> provisional (opt-in)."""
        if self._panel_set_path is not None:
            return self._panel_set_path
        local = repo_root_from_module() / "data" / "issue_612" / "panel" / "panel_set.json"
        if local.exists():
            self._panel_set_path = local
            return local
        try:
            from huggingface_hub import hf_hub_download

            cached = hf_hub_download(
                repo_id=HF_DATA_REPO,
                filename=f"{HF_DATA_PREFIX}/panel/panel_set.json",
                repo_type="dataset",
            )
            dest = self.data_root / "panel" / "panel_set.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, dest)
            self._panel_set_path = dest
            log.info("panel_set.json fetched from HF -> %s", dest)
            return dest
        except Exception as e:
            if not self.args.allow_provisional_panel:
                raise RuntimeError(
                    "panel_set.json unavailable locally and on HF — P2j has not produced "
                    "the final panel. Production launches must wait for it (the driver "
                    "polls HF); only the smoke may pass --allow-provisional-panel."
                ) from e
            log.warning("panel_set.json unavailable (%s) — building provisional panel", e)
            self._panel_set_path = self._build_provisional_panel()
            return self._panel_set_path

    def _parity_panel_set(self) -> Path:
        """Roster-only panel set for parity cells + the G1 probe (frozen-claim rig)."""
        from explore_persona_space.experiments.factor_screen_365.persona_panel import (
            EVAL_PERSONAS_24,
        )

        names = sorted({n for panel in PARITY_PANELS.values() for n in panel})
        path = self.data_root / "panel" / "panel_set_parity.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "provenance": "parity_roster",
                    "personas": {n: {"prompt": EVAL_PERSONAS_24[n]} for n in names},
                },
                indent=2,
            )
        )
        return path

    # ----- subprocess helpers ---------------------------------------------------

    def _eval_subprocess(
        self,
        *,
        model_tag: str,
        out_dir: Path,
        panel_set: Path,
        claims: Path,
        seed: int,
        merged_dir: Path | None = None,
        hub_model_id: str | None = None,
        panel_subset: str | None = None,
        sentinel_name: str,
    ) -> None:
        """Fresh-subprocess vLLM eval (teardown gotcha). The eval's internal
        completion sentinel goes under runs_root, NOT logs_root."""
        eval_sentinel = self.runs_root / "eval_sentinels" / f"{sentinel_name}.json"
        eval_sentinel.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.sycophancy_onpolicy_612.eval_panel",
            "--model-tag",
            model_tag,
            "--seed",
            str(seed),
            "--panel-set",
            str(panel_set),
            "--claims",
            str(claims),
            "--out-dir",
            str(out_dir),
            "--sentinel-path",
            str(eval_sentinel),
        ]
        if merged_dir is not None:
            cmd += ["--merged-model-path", str(merged_dir)]
        if hub_model_id is not None:
            cmd += ["--hub-model-id", hub_model_id]
        if panel_subset is not None:
            cmd += ["--panel-subset", panel_subset]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(self.gpu_id),
            "TQDM_DISABLE": "1",
            "EPM_SKIP_INLINE_CHECKPOINT_UPLOAD": "1",
        }
        log.info("[%s] [phase=eval] spawning: %s", model_tag, " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)
        if not eval_sentinel.exists():
            raise RuntimeError(f"eval subprocess exited 0 but wrote no sentinel {eval_sentinel}")

    def _pool_build_subprocess(self, source: str, arm: str) -> Path:
        """P3 in a fresh subprocess (vLLM + judge). Exit 42 = PositiveYieldError."""
        arms = "arm_onpolicy,arm_prefix" if arm == "arm_prefix" else "arm_onpolicy"
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.sycophancy_onpolicy_612.build_onpolicy_pool",
            "--source",
            source,
            "--arms",
            arms,
            "--data-root",
            str(self.data_root),
            "--judge-concurrency",
            str(self.args.judge_concurrency),
        ]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(self.gpu_id),
            "TQDM_DISABLE": "1",
        }
        log.info("[%s:%s] [phase=pool_build] spawning: %s", source, arm, " ".join(cmd))
        proc = subprocess.run(cmd, env=env)
        if proc.returncode == YIELD_EXIT_CODE:
            raise PoolYieldFailure(source, arm)
        if proc.returncode != 0:
            raise RuntimeError(f"pool build failed for {source}:{arm} (rc={proc.returncode})")
        pool = pool_dir(self.data_root, arm, source) / "train_pool.jsonl"
        if not pool.exists():
            raise RuntimeError(f"pool build exited 0 but {pool} missing")
        return pool

    # ----- train / merge / parity ----------------------------------------------

    def _train_and_merge(
        self,
        source: str,
        arm: str,
        seed: int,
        pool: Path,
        *,
        save_strategy: str = "epoch",
        save_steps: int | None = None,
        upload_epoch_checkpoints: bool = True,
        run_name: str | None = None,
        hub_adapter_base: str | None = None,
    ) -> tuple[Path, Path]:
        """#411 recipe held FIXED (plan §4 cfg). Deviations carried from the
        ported #608 chain, all arm-symmetric + optimization-neutral:
        save_only_model=True (adapter-only checkpoints) and HF uploads moved to
        the dispatcher (fail-loud) — named in the implementer report.

        ``save_strategy``/``save_steps`` parameterize the checkpoint cadence:
        the v1/v2 production+dose path keeps ``save_strategy='epoch'`` (default,
        unchanged); the predictor-v3 path passes ``save_strategy='steps'`` +
        ``save_steps=ceil(total_steps/8)`` for sub-epoch checkpoints (plan v3
        §4.3). When ``save_strategy='steps'`` the per-epoch HF upload of the two
        epoch checkpoints is skipped (``upload_epoch_checkpoints=False`` is
        passed by the v3 caller — it uploads only the band-entry checkpoint)."""
        from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

        output_dir = self.runs_root / arm / f"{source}_seed{seed}"
        adapter_dir = output_dir / "adapter"
        merged_dir = output_dir / "merged"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        cfg = TrainLoraConfig(
            gpu_id=self.gpu_id,
            epochs=3,
            lr=1e-5,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            batch_size=4,
            grad_accum=4,  # effective batch 16
            max_length=2048,  # Arm C prefixes; A/B rows asserted <=1024 at pool build
            warmup_ratio=0.05,
            seed=seed,
            run_name=run_name or f"issue612_{arm}_{source}_seed{seed}",
            report_to="wandb",
            save_strategy=save_strategy,
            save_steps=save_steps if save_strategy == "steps" else None,
            save_total_limit=None,  # keep ALL sub-epoch checkpoints (v3 §4.3)
            save_only_model=True,
            gradient_checkpointing=True,
            packing=False,
            hf_upload=False,  # dispatcher owns uploads (fail-loud, checkpoint handling)
        )
        log.info(
            "[%s:%s:%d] [phase=train] train_lora (save_strategy=%s save_steps=%s) -> %s",
            source,
            arm,
            seed,
            save_strategy,
            save_steps,
            adapter_dir,
        )
        train_lora(
            base_model_path=BASE_MODEL,
            data_path=str(pool),
            output_dir=str(adapter_dir),
            cfg=cfg,
        )
        if not list(adapter_dir.glob("*.safetensors")):
            raise RuntimeError(f"[{source}:{arm}:{seed}] no .safetensors in {adapter_dir}")

        if self.hf_upload:
            hub_base = hub_adapter_base or f"adapters/issue_612/{arm}/{source}_seed{seed}"
            _upload_or_raise(
                adapter_dir, repo_type="model", repo_id=HF_MODEL_REPO, path_in_repo=hub_base
            )
            if upload_epoch_checkpoints:
                ckpts = _resolve_epoch_checkpoints(adapter_dir)
                for k in (1, 2):
                    _upload_checkpoint_or_raise(ckpts[k], f"{hub_base}/checkpoint-epoch{k}")

        log.info("[%s:%s:%d] [phase=merge] merge_lora -> %s", source, arm, seed, merged_dir)
        merge_lora(
            base_model_path=BASE_MODEL,
            adapter_path=str(adapter_dir),
            output_dir=str(merged_dir),
            gpu_id=self.gpu_id,
        )
        return adapter_dir, merged_dir

    def _trajectory_evals(
        self, source: str, arm: str, seed: int, adapter_dir: Path, cell_dir: Path
    ) -> None:
        """Per epoch-1/2 checkpoint: merge -> own-panel eval -> rmtree (plan §4 P4)."""
        from explore_persona_space.train.sft import merge_lora

        ckpts = _resolve_epoch_checkpoints(adapter_dir)
        for k in (1, 2):
            ckpt_dir = ckpts[k]
            if not (ckpt_dir / "adapter_config.json").exists():
                raise RuntimeError(
                    f"[{source}:{arm}:{seed}] epoch-{k} checkpoint {ckpt_dir} has no "
                    f"adapter_config.json — save_strategy='epoch' did not save a PEFT adapter"
                )
            _ensure_tokenizer_files(ckpt_dir, adapter_dir)
            merged_tmp = adapter_dir.parent / f"merged_epoch_{k}"
            log.info("[%s:%s:%d] [phase=trajectory] epoch_%d merge", source, arm, seed, k)
            merge_lora(
                base_model_path=BASE_MODEL,
                adapter_path=str(ckpt_dir),
                output_dir=str(merged_tmp),
                gpu_id=self.gpu_id,
            )
            self._eval_subprocess(
                model_tag=f"{source}:{arm}:{seed}:epoch{k}",
                out_dir=cell_dir / "trajectory" / f"epoch_{k}",
                panel_set=self._resolve_panel_set(),
                claims=self._audited_claims(),
                seed=seed,
                merged_dir=merged_tmp,
                panel_subset=source,
                sentinel_name=f"trajectory-{source}-{arm}-{seed}-epoch{k}",
            )
            shutil.rmtree(merged_tmp, ignore_errors=False)

    def _merge_frozen_adapter(self, source: str, tag: str) -> Path:
        """Merge a prefetched frozen #411 adapter (parity / G1)."""
        from explore_persona_space.train.sft import merge_lora

        adapter_dir = (
            self.adapters_root / "_snapshot" / "adapters" / "issue_411" / f"{source}_seed42"
        )
        if not (adapter_dir / "adapter_config.json").exists():
            raise RuntimeError(
                f"frozen adapter missing at {adapter_dir} — prefetch must run with the "
                f"requesting cell in --cells"
            )
        merged_dir = self.runs_root / "frozen_411" / f"{source}_merged_{tag}"
        log.info("[%s] [phase=merge] frozen #411 adapter -> %s", tag, merged_dir)
        merge_lora(
            base_model_path=BASE_MODEL,
            adapter_path=str(adapter_dir),
            output_dir=str(merged_dir),
            gpu_id=self.gpu_id,
        )
        return merged_dir

    def _audited_claims(self) -> Path:
        return repo_root_from_module() / "data" / "issue_612" / "wrong_claims" / "eval_60.jsonl"

    def _frozen_claims(self) -> Path:
        return self.data_root / "wrong_claims" / "eval_50.jsonl"

    # ----- the unified per-cell path ---------------------------------------------

    def _run_cell(self, source: str, arm: str, seed: int) -> dict:
        """ONE cell through its full phase chain. Smoke and sweep both land here."""
        cid = cell_id(source, arm, seed)
        prior = _read_cellstate(self.slab_root, source, arm, seed)
        if prior is not None and prior.get("status") == "complete":
            if (
                arm in TRAIN_ARMS
                and prior.get("panel_provenance") == "provisional_mandatory11"
                and not self.args.allow_provisional_panel
            ):
                # Same-instance smoke-then-production flow: the smoke cell's
                # endpoint eval ran on the 11-persona provisional panel; a
                # production re-run must NOT skip it or P7 fails on the ~19
                # missing personas. Re-run on the selected panel instead.
                log.warning(
                    "[%s] prior complete cell used the PROVISIONAL panel — re-running "
                    "on the selected panel (idempotent skip refused)",
                    cid,
                )
            else:
                log.info("[%s] cell-state already complete — skipping (idempotent re-run)", cid)
                return prior

        t0 = time.time()
        cell_dir = cell_slab_dir(self.slab_root, source, arm, seed)
        record: dict = {
            "cell": cid,
            "source": source,
            "arm": arm,
            "seed": seed,
            "gpu_id": self.gpu_id,
            "eval_out_dir": str(cell_dir),
            "git_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
        }
        log.info("=" * 70)
        log.info("[%s] CELL START -> %s", cid, cell_dir)

        if self.dry_run:
            record.update(status="dry_run", wall_seconds=round(time.time() - t0, 1))
            _write_cellstate(self.slab_root, source, arm, seed, record)
            self._cell_sentinel(record)
            log.info("[%s] dry-run cell walk complete", cid)
            return record

        try:
            if arm in TRAIN_ARMS:
                self._run_train_cell(source, arm, seed, cell_dir, record)
            elif (source, arm) == ("panel", "build"):
                self._run_panel_build(record)
            elif (source, arm) == ("base", "pass"):
                self._run_base_pass(cell_dir, record)
            elif arm == "parity":
                self._run_parity_cell(source, cell_dir, record)
            else:  # unreachable — parse_cells validates
                raise ValueError(f"Unknown arm {arm!r}")
        except PoolYieldFailure as e:
            # G3: a source that cannot reach 200 positives drops from B/C —
            # reported, never silently shrunk. The SMOKE cell halts instead.
            record.update(status="dropped_yield", wall_seconds=round(time.time() - t0, 1))
            _write_cellstate(self.slab_root, source, arm, seed, record)
            _write_sentinel(
                self.logs_root,
                kind="epm:progress",
                name_slug=f"cell-{source}-{arm}-{seed}-YIELD-DROP",
                note_obj={"event": "cell_dropped_yield", **record},
                gate="g3-per-source-yield",
            )
            if (source, arm, seed) == SMOKE_CELL:
                raise RuntimeError(f"[G2] smoke cell yield failure: {e} — halt (plan §7 G2)") from e
            log.warning("[%s] dropped on yield (G3): %s", cid, e)
            return record

        record.update(status="complete", wall_seconds=round(time.time() - t0, 1))
        _write_cellstate(self.slab_root, source, arm, seed, record)
        self._cell_sentinel(record)
        log.info("[%s] cell complete in %.1fs", cid, record["wall_seconds"])
        return record

    def _run_train_cell(
        self, source: str, arm: str, seed: int, cell_dir: Path, record: dict
    ) -> None:
        if arm == "arm_canned":
            pool = pool_dir(self.data_root, arm, source) / "train_pool.jsonl"
            if not pool.exists():
                raise FileNotFoundError(f"frozen pool missing: {pool} (prefetch must run)")
            # Token-cap parity assert for the FROZEN pool too: a frozen row in
            # (1024, 2048] would now train untruncated where #411 truncated it
            # (max_length 1024 -> 2048 deviation must stay behaviorally inert).
            from transformers import AutoTokenizer

            from explore_persona_space.experiments.sycophancy_onpolicy_612.build_onpolicy_pool import (  # noqa: E501
                validate_pool,
            )

            tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            record["pool_check"] = validate_pool(pool, source, "arm_canned", pool, tokenizer=tok)
        else:
            log.info("[%s:%s:%d] [phase=pool_build]", source, arm, seed)
            pool = self._pool_build_subprocess(source, arm)
            if self.hf_upload:
                from explore_persona_space.orchestrate.hub import upload_dataset

                for name in ("train_pool.jsonl", "pool_meta.json"):
                    local = pool.parent / name
                    hub_path = upload_dataset(
                        str(local),
                        path_in_repo=f"{HF_DATA_PREFIX}/training_pools/{arm}/{source}/{name}",
                    )
                    if not hub_path:
                        raise RuntimeError(f"training-pool upload failed: {local}")
        record["pool_path"] = str(pool)

        adapter_dir, merged_dir = self._train_and_merge(source, arm, seed, pool)
        record["adapter_dir"] = str(adapter_dir)
        record["adapter_hf_path"] = f"adapters/issue_612/{arm}/{source}_seed{seed}"
        panel_set = self._resolve_panel_set()
        record["panel_provenance"] = json.loads(panel_set.read_text()).get("provenance")
        self._eval_subprocess(
            model_tag=f"{source}:{arm}:{seed}",
            out_dir=cell_dir,
            panel_set=panel_set,
            claims=self._audited_claims(),
            seed=seed,
            merged_dir=merged_dir,
            sentinel_name=f"eval-{source}-{arm}-{seed}",
        )
        shutil.rmtree(merged_dir, ignore_errors=False)  # disk-quota discipline
        self._trajectory_evals(source, arm, seed, adapter_dir, cell_dir)
        log.info("[%s:%s:%d] [phase=upload]", source, arm, seed)
        record["hub_eval_tree"] = self._upload_cell_tree(
            cell_dir, f"cells/{arm}/{source}/seed_{seed}"
        )

    def _run_panel_build(self, record: dict) -> None:
        log.info("[phase=panel_build]")
        out_dir = self.data_root / "panel"
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.sycophancy_onpolicy_612.panel_build",
            "--data-root",
            str(self.data_root),
            "--out-dir",
            str(out_dir),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu_id), "TQDM_DISABLE": "1"}
        subprocess.run(cmd, env=env, check=True)
        candidates = out_dir / "panel_candidates.json"
        if not candidates.exists():
            raise RuntimeError(f"panel build exited 0 but {candidates} missing")
        record["panel_candidates"] = str(candidates)
        if self.hf_upload:
            for name in ("panel_candidates.json", "panel_centroids_layer20.pt"):
                _upload_or_raise(
                    out_dir / name,
                    repo_type="dataset",
                    repo_id=HF_DATA_REPO,
                    path_in_repo=f"{HF_DATA_PREFIX}/panel/{name}",
                )

    def _run_base_pass(self, cell_dir: Path, record: dict) -> None:
        """P2: base model x ALL candidates x audited claims (delta baseline + prior)."""
        log.info("[phase=base_pass]")
        candidates_path = self.data_root / "panel" / "panel_candidates.json"
        if not candidates_path.exists():
            raise FileNotFoundError(
                f"{candidates_path} missing — panel:build:0 must run before base:pass:0"
            )
        payload = json.loads(candidates_path.read_text())
        panel_set = self.data_root / "panel" / "panel_set_candidates.json"
        panel_set.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "provenance": "p1_candidates",
                    "personas": {
                        name: {"prompt": rec["prompt"]}
                        for name, rec in payload["candidates"].items()
                    },
                },
                indent=2,
            )
        )
        self._eval_subprocess(
            model_tag="base",
            out_dir=cell_dir,
            panel_set=panel_set,
            claims=self._audited_claims(),
            seed=42,
            hub_model_id=BASE_MODEL,
            sentinel_name="eval-base-pass",
        )
        record["hub_eval_tree"] = self._upload_cell_tree(cell_dir, "base")

    def _run_parity_cell(self, source: str, cell_dir: Path, record: dict) -> None:
        """P5: frozen #411 adapter on the FROZEN 50 claims, #591 anchor panels."""
        log.info("[%s] [phase=parity]", source)
        merged_dir = self._merge_frozen_adapter(source, tag="parity")
        self._eval_subprocess(
            model_tag=f"{source}:parity:42",
            out_dir=cell_dir,
            panel_set=self._parity_panel_set(),
            claims=self._frozen_claims(),
            seed=42,
            merged_dir=merged_dir,
            panel_subset=",".join(PARITY_PANELS[source]),
            sentinel_name=f"parity-{source}",
        )
        shutil.rmtree(merged_dir, ignore_errors=False)
        record["hub_eval_tree"] = self._upload_cell_tree(cell_dir, f"parity/{source}")

    def _upload_cell_tree(self, local: Path, rel: str) -> str | None:
        """Per-cell upload: the whole eval tree (per-panel JSONs + raw_completions/
        + trajectory/) -> HF data repo BEFORE pod termination (Upload Policy).
        raw completions land via hub._upload of the tree; presence asserted first."""
        if not self.hf_upload:
            log.info("[%s] HF upload disabled — skipping cell-tree upload", rel)
            return None
        raw_files = list(local.rglob("raw_completions/*.json"))
        if not raw_files:
            raise RuntimeError(
                f"no raw_completions/*.json under {local} — eval wrote nothing; "
                f"refusing to upload an empty cell tree"
            )
        return _upload_or_raise(
            local,
            repo_type="dataset",
            repo_id=HF_DATA_REPO,
            path_in_repo=f"{HF_DATA_PREFIX}/eval_results/{rel}",
        )

    def _cell_sentinel(self, record: dict) -> None:
        _write_sentinel(
            self.logs_root,
            kind="epm:progress",
            name_slug=f"cell-{record['source']}-{record['arm']}-{record['seed']}",
            note_obj={"event": "cell_complete", **record},
        )

    # ----- G1 / G2 smoke gates ----------------------------------------------------

    def _mini_judge_panel_file(self, panel_file: Path) -> dict:
        """Pod-side Haiku pass over one panel eval JSON (sanctioned gate moment)."""
        from explore_persona_space.experiments.sycophancy_onpolicy_612.judge import judge_batch

        if not panel_file.exists():
            raise RuntimeError(f"panel eval JSON missing: {panel_file}")
        payload = json.loads(panel_file.read_text())
        records = payload["completions"]
        rollouts = [{"wrong_claim": r["claim"], "completion": r["completion"]} for r in records]
        n_nonempty = sum(1 for r in records if r["completion"].strip())
        log.info("[gate] mini-judging %d rollouts (%s)", len(rollouts), panel_file.name)
        verdicts = asyncio.run(
            judge_batch(rollouts, model=JUDGE_MODEL, max_concurrency=self.args.judge_concurrency)
        )
        n_yes = sum(1 for v in verdicts if v.agreed)
        n_unparseable = sum(1 for v in verdicts if v.error and "unparseable" in v.error)
        n_api_errors = sum(1 for v in verdicts if v.error and "unparseable" not in (v.error or ""))
        return {
            "panel_file": str(panel_file),
            "n_rollouts": len(rollouts),
            "rate": n_yes / max(len(verdicts), 1),
            "judge_parse_rate": 1.0 - n_unparseable / max(len(verdicts), 1),
            "n_api_errors": n_api_errors,
            "nonempty_completion_rate": n_nonempty / max(len(records), 1),
        }

    def _gate_g1(self) -> None:
        """G1: frozen-adapter apply-and-read probe — the frozen villain adapter's
        SELF rate on the frozen 50 claims must reproduce the #411 record within
        +-G1_TOL (kill: eval-stack bug; #591 Gate-2 tolerance)."""
        log.info("[phase=smoke_gate_g1]")
        merged_dir = self._merge_frozen_adapter("villain", tag="g1")
        out_dir = self.slab_root / "smoke_g1"
        self._eval_subprocess(
            model_tag="villain:g1_probe:42",
            out_dir=out_dir,
            panel_set=self._parity_panel_set(),
            claims=self._frozen_claims(),
            seed=42,
            merged_dir=merged_dir,
            panel_subset="villain",
            sentinel_name="smoke-g1",
        )
        shutil.rmtree(merged_dir, ignore_errors=False)
        read = self._mini_judge_panel_file(out_dir / "sycophancy_eval_villain.json")
        analyze = json.loads((repo_root_from_module() / ANALYZE_SUMMARY_RELPATH).read_text())[
            "per_source"
        ]
        frozen_rate = analyze["villain"]["per_panel_trained_rate"]["villain"]
        drift = read["rate"] - frozen_rate
        note = {
            "event": "smoke_gate_g1",
            "probe": "frozen villain adapter, self panel, frozen 50 claims",
            "fresh_rate": read["rate"],
            "frozen_rate": frozen_rate,
            "drift": drift,
            "tolerance": G1_TOL,
            "read": read,
        }
        if abs(drift) <= G1_TOL:
            note["decision"] = "PASS"
            _write_sentinel(
                self.logs_root, kind="epm:progress", name_slug="smoke-gate-g1", note_obj=note
            )
            log.info("[phase=smoke_gate_g1] PASS: drift=%.3f (tol %.2f)", drift, G1_TOL)
            return
        note["decision"] = "HALT"
        _write_sentinel(
            self.logs_root,
            kind="epm:progress",
            name_slug="smoke-gate-g1",
            note_obj=note,
            gate="g1-rig-validity",
        )
        raise RuntimeError(
            f"[G1] HALT: frozen-adapter probe drift {drift:+.3f} exceeds ±{G1_TOL} "
            f"(fresh {read['rate']:.3f} vs frozen {frozen_rate:.3f}) — eval-stack bug; "
            f"no sweep launch (plan §7 G1)."
        )

    def _gate_g2(self) -> None:
        """G2: install + yield check on the smoke cell with the registered
        disambiguation path (plan §7; yield is enforced by the pool build)."""
        source, arm, seed = SMOKE_CELL
        log.info("[phase=smoke_gate_g2]")
        cell_dir = cell_slab_dir(self.slab_root, source, arm, seed)
        trained_read = self._mini_judge_panel_file(cell_dir / f"sycophancy_eval_{source}.json")

        base_dir = self.slab_root / "smoke_g2_base"
        base_panel_file = base_dir / f"sycophancy_eval_{source}.json"
        # Reuse the P2 base pass when present on this machine; else generate a
        # fresh villain-only base read through the SAME eval entrypoint.
        p2_panel_file = (
            cell_slab_dir(self.slab_root, "base", "pass", 0) / f"sycophancy_eval_{source}.json"
        )
        if p2_panel_file.exists():
            base_panel_file = p2_panel_file
        elif not base_panel_file.exists():
            self._eval_subprocess(
                model_tag="base:smoke_g2",
                out_dir=base_dir,
                panel_set=self._resolve_panel_set(),
                claims=self._audited_claims(),
                seed=42,
                hub_model_id=BASE_MODEL,
                panel_subset=source,
                sentinel_name="smoke-g2-base",
            )
        base_read = self._mini_judge_panel_file(base_panel_file)
        delta = trained_read["rate"] - base_read["rate"]
        note: dict = {
            "event": "smoke_gate_g2",
            "cell": cell_id(*SMOKE_CELL),
            "delta_floor": G2_DELTA_FLOOR,
            "trained_read": trained_read,
            "base_read": base_read,
            "delta_self_vs_fresh_base": delta,
        }
        if delta >= G2_DELTA_FLOOR:
            note["decision"] = "PASS"
            _write_sentinel(
                self.logs_root, kind="epm:progress", name_slug="smoke-gate-g2", note_obj=note
            )
            log.info("[phase=smoke_gate_g2] PASS: delta=%.3f >= %.2f", delta, G2_DELTA_FLOOR)
            return

        # Install below floor -> registered disambiguation bundle.
        log.warning("[phase=smoke_gate_g2] delta=%.3f < %.2f — diagnostics", delta, G2_DELTA_FLOOR)
        anomalies: list[str] = []
        from explore_persona_space.experiments.sycophancy_onpolicy_612.build_onpolicy_pool import (
            validate_pool,
        )

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            note["pool_check"] = validate_pool(
                pool_dir(self.data_root, arm, source) / "train_pool.jsonl",
                source,
                arm,
                self.data_root / "pools_411" / f"{source}_seed42" / "train_pool.jsonl",
                tokenizer=tokenizer,
            )
        except AssertionError as e:
            anomalies.append(f"pool: {e}")
        adapter_dir = self.runs_root / arm / f"{source}_seed{seed}" / "adapter"
        try:
            loss_report = _loss_curve_report(adapter_dir)
            note["loss_check"] = loss_report
            if loss_report["has_nan"]:
                anomalies.append("loss: NaN in log_history")
            if not loss_report["decreasing_first_to_last"]:
                anomalies.append(
                    f"loss: not decreasing ({loss_report['first_loss']:.4f} -> "
                    f"{loss_report['last_loss']:.4f})"
                )
        except RuntimeError as e:
            anomalies.append(f"loss: {e}")
        for tag, read in (("trained", trained_read), ("base", base_read)):
            if read["nonempty_completion_rate"] < NONEMPTY_COMPLETION_FLOOR:
                anomalies.append(
                    f"eval[{tag}]: nonempty rate {read['nonempty_completion_rate']:.3f}"
                )
            if read["judge_parse_rate"] < JUDGE_PARSE_RATE_FLOOR:
                anomalies.append(f"judge[{tag}]: parse rate {read['judge_parse_rate']:.3f}")
            err_rate = read["n_api_errors"] / max(read["n_rollouts"], 1)
            if err_rate > SMOKE_JUDGE_API_ERROR_CEILING:
                anomalies.append(f"judge[{tag}]: API error rate {err_rate:.3f}")
        note["anomalies"] = anomalies

        if not anomalies:
            # Clean diagnostics -> ONE diagnostic retrain at seed 137, then HALT
            # with the bundle either way (plan §7 G2 — install-fail is a halt,
            # not a continue; read2 is part of the disambiguation evidence).
            diag = SMOKE_DIAGNOSTIC_CELL
            log.info("[phase=smoke_gate_g2] diagnostics clean — diagnostic cell %s", cell_id(*diag))
            self._run_cell(*diag)
            read2 = self._mini_judge_panel_file(
                cell_slab_dir(self.slab_root, *diag) / f"sycophancy_eval_{source}.json"
            )
            note["read2"] = read2
            note["read2_delta_self_vs_fresh_base"] = read2["rate"] - base_read["rate"]
        note["decision"] = "HALT_DISAMBIGUATION"
        _write_sentinel(
            self.logs_root,
            kind="epm:progress",
            name_slug="smoke-gate-g2",
            note_obj=note,
            gate="g2-install-check",
        )
        raise RuntimeError(
            f"[G2] HALT: smoke install delta {delta:+.3f} < +{G2_DELTA_FLOOR}; "
            f"disambiguation bundle in the smoke-gate-g2 sentinel "
            f"(anomalies: {anomalies or 'none — see read2'})."
        )

    # ----- final aggregation -------------------------------------------------------

    def finalize(
        self, cells: list[tuple[str, str, int]], all_cells: list[tuple[str, str, int]]
    ) -> None:
        """epm:results ONLY for a complete full-grid non-dry run; else epm:progress."""
        states = {c: _read_cellstate(self.slab_root, *c) for c in all_cells}
        ok_statuses = ("complete", "dry_run") if self.dry_run else ("complete",)
        complete = {k: v for k, v in states.items() if v and v.get("status") in ok_statuses}
        dropped = {k: v for k, v in states.items() if v and v.get("status") == "dropped_yield"}
        full_grid = set(full_production_cells())
        covers_full_grid = set(all_cells) == full_grid
        summary = {
            "issue": 612,
            "gpu_id": self.gpu_id,
            "shard_cells": [cell_id(*c) for c in cells],
            "all_cells": [cell_id(*c) for c in all_cells],
            "n_complete": len(complete),
            "n_dropped_yield": len(dropped),
            "n_all": len(all_cells),
            "covers_full_production_grid": covers_full_grid,
            "dry_run": self.dry_run,
            "eval_paths": {cell_id(*k): v.get("eval_out_dir") for k, v in complete.items()},
            "wall_seconds_by_cell": {
                cell_id(*k): v.get("wall_seconds") for k, v in complete.items()
            },
            "reproducibility_card": {
                "base_model": BASE_MODEL,
                "hf_model_repo": HF_MODEL_REPO,
                "hf_data_repo": HF_DATA_REPO,
                "hf_data_prefix": HF_DATA_PREFIX,
                "adapter_paths": {
                    cell_id(*k): v.get("adapter_hf_path")
                    for k, v in complete.items()
                    if v.get("adapter_hf_path")
                },
                "final_commit_sha": _git_sha(),
            },
            "hostname": socket.gethostname(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        emit_results = (
            not self.dry_run
            and covers_full_grid
            and len(complete) == len(all_cells)
            and not dropped
        )
        if emit_results:
            assert set(all_cells) == full_grid, (
                f"epm:results gate reached with a non-grid all_cells "
                f"({len(all_cells)} != {len(full_grid)})"
            )
            _write_sentinel(
                self.logs_root,
                kind="epm:results",
                name_slug="epm_results",
                note_obj={"event": "sweep_complete", **summary},
            )
        else:
            _write_sentinel(
                self.logs_root,
                kind="epm:progress",
                name_slug=f"shard-gpu{self.gpu_id}-done",
                note_obj={"event": "shard_complete", **summary},
            )


class PoolYieldFailure(RuntimeError):
    """Raised when the P3 builder exits with YIELD_EXIT_CODE (G3 drop rule)."""

    def __init__(self, source: str, arm: str):
        super().__init__(f"positive-yield failure for {source}:{arm}")
        self.source = source
        self.arm = arm


# ----- dose-matched follow-up round (plans/v2.md §3) -------------------------------


def _hf_download_with_retry(
    *, repo_id: str, filename: str, revision: str, force: bool = False, attempts: int = 3
) -> str:
    """hf_hub_download with backoff (dispatcher silent-death hardening)."""
    from huggingface_hub import hf_hub_download

    delay = 30
    for i in range(attempts):
        try:
            return hf_hub_download(
                repo_id=repo_id, filename=filename, revision=revision, force_download=force
            )
        except Exception as e:
            if i == attempts - 1:
                raise
            log.warning("hf_hub_download %s failed (%s) — retry in %ds", filename, e, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


class DoseMatchedRunner(Dispatcher):
    """Plan-v2 band-entry eval-only runner. Reuses the Dispatcher's eval/upload/
    mini-judge helpers verbatim; adds the pinned-revision checkpoint fetch path.
    No training phases exist in this stage."""

    def __init__(self, args: argparse.Namespace, selection: dict):
        super().__init__(args)
        self.selection = selection
        self._dose_files_cache: list[str] | None = None
        self._force_fetch = False

    # ----- selection / cell-state ------------------------------------------------

    def _epoch_for(self, source: str, arm: str, seed: int) -> int:
        rec = self.selection["cells"][cell_id(source, arm, seed)]
        epoch = rec["band_entry_epoch"]
        if not isinstance(epoch, int):
            raise RuntimeError(
                f"{cell_id(source, arm, seed)} has no band-entry epoch (role={rec['role']}) — "
                f"never-entered / excluded cells are NOT evaluated (plan v2 §2)."
            )
        return epoch

    def _dose_state_path(self, source: str, arm: str, seed: int) -> Path:
        return self.slab_root / "dose_matched" / "_cellstate" / f"{source}__{arm}__{seed}.json"

    def _read_dose_state(self, source: str, arm: str, seed: int) -> dict | None:
        path = self._dose_state_path(source, arm, seed)
        return json.loads(path.read_text()) if path.exists() else None

    def _write_dose_state(self, source: str, arm: str, seed: int, record: dict) -> None:
        path = self._dose_state_path(source, arm, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2))

    # ----- pinned inputs + hub preflight ------------------------------------------

    def dose_prefetch(self) -> None:
        """[phase=dose_prefetch] sha-pinned panel_set + eval_60 (plan v2 rule (f))."""
        from explore_persona_space.experiments.sycophancy_onpolicy_612.prefetch_inputs import (
            fetch_pinned,
        )

        log.info("[phase=dose_prefetch] pinned panel_set.json + eval_60.jsonl")
        for repo_path, dest_name in (
            (f"{HF_DATA_PREFIX}/panel/panel_set.json", "panel_set.json"),
            (f"{HF_DATA_PREFIX}/inputs/eval_60.jsonl", "eval_60.jsonl"),
        ):
            fetch_pinned(
                repo_path,
                self.data_root / "dose_matched" / dest_name,
                expected=DOSE_MATCHED_SHA256[repo_path],
            )

    def _dose_panel_and_claims(self) -> tuple[Path, Path]:
        panel = self.data_root / "dose_matched" / "panel_set.json"
        claims = self.data_root / "dose_matched" / "eval_60.jsonl"
        missing = [str(p) for p in (panel, claims) if not p.exists()]
        if missing:
            raise RuntimeError(
                f"dose-matched pinned inputs missing: {missing} — dose_prefetch must run "
                f"(drop --skip-prefetch, or place the sha-verified files there)."
            )
        return panel, claims

    def _dose_repo_files(self) -> list[str]:
        """list_repo_files at the PINNED revision (NOT snapshot_download —
        siblings-truncation gotcha on large repos). Cached per process."""
        if self._dose_files_cache is None:
            from huggingface_hub import list_repo_files

            self._dose_files_cache = list(
                list_repo_files(HF_MODEL_REPO, revision=DOSE_ADAPTER_REVISION)
            )
        return self._dose_files_cache

    def dose_hub_preflight(self, cells: list[tuple[str, str, int]]) -> None:
        """K2-dm for EVERY requested cell up front — fail before the first merge."""
        files = set(self._dose_repo_files())
        problems: list[str] = []
        for source, arm, seed in cells:
            epoch = self._epoch_for(source, arm, seed)
            sub = DOSE_ADAPTER_PATH_TMPL.format(arm=arm, source=source, seed=seed)
            problems.extend(
                p
                for name in DOSE_REQUIRED_CKPT_FILES
                if (p := f"{sub}/checkpoint-epoch{epoch}/{name}") not in files
            )
        if problems:
            raise RuntimeError(
                f"[K2-dm] checkpoints incomplete on the Hub @ {DOSE_ADAPTER_REVISION[:12]}: "
                f"missing {problems} — fail loud; never substitute the endpoint adapter or "
                f"another epoch (plan v2 kill criteria)."
            )
        log.info("[phase=dose_prefetch] K2-dm hub preflight OK for %d cells", len(cells))

    def _fetch_dose_checkpoint(
        self, source: str, arm: str, seed: int, epoch: int
    ) -> tuple[Path, Path]:
        """Per-file fetch of the cell subtree (final dir + the selected checkpoint)
        at the pinned revision; K2-dm asserts remote AND local."""
        sub = DOSE_ADAPTER_PATH_TMPL.format(arm=arm, source=source, seed=seed)
        repo_files = [f for f in self._dose_repo_files() if f.startswith(f"{sub}/")]
        if not repo_files:
            raise RuntimeError(
                f"[K2-dm] no files under {sub}/ on the Hub @ {DOSE_ADAPTER_REVISION[:12]}"
            )
        ckpt_prefix = f"{sub}/checkpoint-epoch{epoch}/"
        missing_remote = [
            f"{ckpt_prefix}{n}"
            for n in DOSE_REQUIRED_CKPT_FILES
            if f"{ckpt_prefix}{n}" not in repo_files
        ]
        if missing_remote:
            raise RuntimeError(
                f"[K2-dm] {sub} missing {missing_remote} on the Hub @ "
                f"{DOSE_ADAPTER_REVISION[:12]} — fail loud, no epoch substitution."
            )
        local_root = Path(self.adapters_root) / "dose_612" / arm / f"{source}_seed{seed}"
        for repo_path in repo_files:
            rel = repo_path[len(sub) + 1 :]
            if rel.startswith("checkpoint-") and not rel.startswith(f"checkpoint-epoch{epoch}/"):
                continue  # skip sibling checkpoints (only the band-entry epoch is read)
            dest = local_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            cached = _hf_download_with_retry(
                repo_id=HF_MODEL_REPO,
                filename=repo_path,
                revision=DOSE_ADAPTER_REVISION,
                force=self._force_fetch,
            )
            shutil.copyfile(cached, dest)
        ckpt_dir = local_root / f"checkpoint-epoch{epoch}"
        missing_local = [n for n in DOSE_REQUIRED_CKPT_FILES if not (ckpt_dir / n).exists()]
        if missing_local:
            raise RuntimeError(f"[K2-dm] {ckpt_dir} missing {missing_local} after fetch")
        _ensure_tokenizer_files(ckpt_dir, local_root)
        return local_root, ckpt_dir

    # ----- the unified per-cell dose path ------------------------------------------

    def run_dose_cell(self, source: str, arm: str, seed: int) -> dict:
        """ONE band-entry cell: fetch -> merge -> full-panel eval -> upload.
        The smoke cell and every sweep cell land here (smoke = sweep with one cell)."""
        cid = cell_id(source, arm, seed)
        epoch = self._epoch_for(source, arm, seed)
        prior = self._read_dose_state(source, arm, seed)
        if prior is not None and prior.get("status") == "complete":
            log.info("[%s] dose cell-state already complete — skipping (idempotent)", cid)
            return prior
        t0 = time.time()
        cell_dir = dose_cell_dir(self.slab_root, arm, source, seed, epoch)
        record: dict = {
            "cell": cid,
            "stage": "dose-matched",
            "band_entry_epoch": epoch,
            "source": source,
            "arm": arm,
            "seed": seed,
            "gpu_id": self.gpu_id,
            "eval_out_dir": str(cell_dir),
            "adapter_hf_path": (
                f"{DOSE_ADAPTER_PATH_TMPL.format(arm=arm, source=source, seed=seed)}"
                f"/checkpoint-epoch{epoch}"
            ),
            "adapter_revision": DOSE_ADAPTER_REVISION,
            "git_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
        }
        log.info("=" * 70)
        log.info("[%s] DOSE CELL START @ checkpoint-epoch%d -> %s", cid, epoch, cell_dir)

        if self.dry_run:
            record.update(status="dry_run", wall_seconds=round(time.time() - t0, 1))
            self._write_dose_state(source, arm, seed, record)
            _write_sentinel(
                self.logs_root,
                kind="epm:progress",
                name_slug=f"dose-cell-{source}-{arm}-{seed}",
                note_obj={"event": "dose_cell_dry_run", **record},
            )
            log.info("[%s] dry-run dose-cell walk complete", cid)
            return record

        log.info(
            "[%s] [phase=dose_fetch] %s @ %s",
            cid,
            record["adapter_hf_path"],
            DOSE_ADAPTER_REVISION[:12],
        )
        _adapter_root, ckpt_dir = self._fetch_dose_checkpoint(source, arm, seed, epoch)

        from explore_persona_space.train.sft import merge_lora

        merged_dir = (
            self.runs_root / "dose_matched" / arm / f"{source}_seed{seed}_epoch{epoch}" / "merged"
        )
        log.info("[%s] [phase=merge] checkpoint-epoch%d -> %s", cid, epoch, merged_dir)
        merge_lora(
            base_model_path=BASE_MODEL,
            adapter_path=str(ckpt_dir),
            output_dir=str(merged_dir),
            gpu_id=self.gpu_id,
        )

        panel_set, claims = self._dose_panel_and_claims()
        record["panel_provenance"] = json.loads(panel_set.read_text()).get("provenance")
        self._eval_subprocess(
            model_tag=f"{source}:{arm}:{seed}:band_epoch{epoch}",
            out_dir=cell_dir,
            panel_set=panel_set,
            claims=claims,
            seed=seed,
            merged_dir=merged_dir,
            sentinel_name=f"dose-eval-{source}-{arm}-{seed}-epoch{epoch}",
        )
        shutil.rmtree(merged_dir, ignore_errors=False)  # disk-quota discipline

        log.info("[%s] [phase=upload]", cid)
        record["hub_eval_tree"] = self._upload_cell_tree(
            cell_dir, f"dose_matched/cells/{arm}/{source}/seed_{seed}/epoch_{epoch}"
        )
        record.update(status="complete", wall_seconds=round(time.time() - t0, 1))
        self._write_dose_state(source, arm, seed, record)
        _write_sentinel(
            self.logs_root,
            kind="epm:progress",
            name_slug=f"dose-cell-{source}-{arm}-{seed}",
            note_obj={"event": "dose_cell_complete", **record},
        )
        log.info("[%s] dose cell complete in %.1fs", cid, record["wall_seconds"])
        return record

    # ----- G1-dm smoke parity gate ---------------------------------------------------

    def _wipe_dose_cell(self, source: str, arm: str, seed: int, epoch: int) -> None:
        """K1-dm diagnostic retry prep: drop the cell outputs + state + local
        adapter copy, and force-redownload on the re-fetch."""
        for path in (
            dose_cell_dir(self.slab_root, arm, source, seed, epoch),
            self._dose_state_path(source, arm, seed),
            Path(self.adapters_root) / "dose_612" / arm / f"{source}_seed{seed}",
        ):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=False)
            elif path.exists():
                path.unlink()
        self._force_fetch = True

    def gate_g1_dm(self) -> None:
        """G1-dm (plan v2 §7): the smoke cell's fresh full-panel SELF read must
        reproduce the pinned epoch-1 trajectory rate (same checkpoint, same
        60-claim set) within ±G1_DM_TOL. ONE diagnostic re-fetch+re-merge retry
        (K1-dm); a second miss halts before the 7-cell launch."""
        source, arm, seed = DOSE_SMOKE_CELL
        epoch = self._epoch_for(source, arm, seed)
        ref = self.selection["g1_dm"]
        cell_dir = dose_cell_dir(self.slab_root, arm, source, seed, epoch)
        for attempt in (1, 2):
            log.info("[phase=smoke_gate_g1dm] attempt %d: mini-judge self panel", attempt)
            read = self._mini_judge_panel_file(cell_dir / f"sycophancy_eval_{source}.json")
            drift = read["rate"] - ref["reference_rate"]
            note = {
                "event": "smoke_gate_g1dm",
                "attempt": attempt,
                "probe": (
                    "fresh full-panel self read of villain:arm_canned:42 @ checkpoint-epoch1 "
                    "vs the existing epoch-1 trajectory read (60-claim set, raw rates)"
                ),
                "fresh_rate": read["rate"],
                "reference_rate": ref["reference_rate"],
                "drift": drift,
                "tolerance": G1_DM_TOL,
                "read": read,
            }
            if abs(drift) <= G1_DM_TOL:
                note["decision"] = "PASS"
                _write_sentinel(
                    self.logs_root,
                    kind="epm:progress",
                    name_slug="dose-smoke-gate-g1dm",
                    note_obj=note,
                )
                log.info("[phase=smoke_gate_g1dm] PASS: drift=%+.3f (tol %.2f)", drift, G1_DM_TOL)
                return
            if attempt == 1:
                note["decision"] = "RETRY"
                _write_sentinel(
                    self.logs_root,
                    kind="epm:progress",
                    name_slug="dose-smoke-gate-g1dm-retry",
                    note_obj=note,
                )
                log.warning(
                    "[phase=smoke_gate_g1dm] drift %+.3f exceeds ±%.2f — ONE diagnostic "
                    "retry (re-fetch + re-merge, K1-dm)",
                    drift,
                    G1_DM_TOL,
                )
                self._wipe_dose_cell(source, arm, seed, epoch)
                self.run_dose_cell(source, arm, seed)
                continue
            note["decision"] = "HALT"
            _write_sentinel(
                self.logs_root,
                kind="epm:progress",
                name_slug="dose-smoke-gate-g1dm",
                note_obj=note,
                gate="g1dm-smoke-parity",
            )
            raise RuntimeError(
                f"[G1-dm] HALT after diagnostic retry: drift {drift:+.3f} exceeds "
                f"±{G1_DM_TOL} (fresh {read['rate']:.3f} vs trajectory "
                f"{ref['reference_rate']:.3f}) — checkpoint-fetch or eval-stack bug; "
                f"no 7-cell launch (plan v2 §7 / K1-dm)."
            )

    # ----- finalize -----------------------------------------------------------------

    def dose_finalize(
        self, cells: list[tuple[str, str, int]], all_cells: list[tuple[str, str, int]]
    ) -> None:
        """Dose epm:results (version 3) ONLY for a complete registered-8 non-dry
        run; everything else writes an epm:progress shard sentinel."""
        states = {cell_id(*c): self._read_dose_state(*c) for c in all_cells}
        ok_statuses = ("complete", "dry_run") if self.dry_run else ("complete",)
        complete = {k: v for k, v in states.items() if v and v.get("status") in ok_statuses}
        evaluated = list(self.selection["evaluated_cells"])
        covers = {cell_id(*c) for c in all_cells} == set(evaluated)
        summary = {
            "issue": 612,
            "stage": "dose-matched",
            "followup_label": "dose-matched-leakage-read",
            "gpu_id": self.gpu_id,
            "shard_cells": [cell_id(*c) for c in cells],
            "all_cells": [cell_id(*c) for c in all_cells],
            "n_complete": len(complete),
            "n_all": len(all_cells),
            "covers_registered_dose_set": covers,
            "dry_run": self.dry_run,
            "band_entry_epochs": {
                cid: self.selection["cells"][cid]["band_entry_epoch"] for cid in evaluated
            },
            "eval_paths": {k: v.get("eval_out_dir") for k, v in complete.items()},
            "wall_seconds_by_cell": {k: v.get("wall_seconds") for k, v in complete.items()},
            "reproducibility_card": {
                "base_model": BASE_MODEL,
                "training": "none — eval-only round on the parent's adapters (plan v2 §0)",
                "hf_model_repo": HF_MODEL_REPO,
                "adapter_revision": DOSE_ADAPTER_REVISION,
                "adapter_paths": {
                    k: v.get("adapter_hf_path")
                    for k, v in complete.items()
                    if v.get("adapter_hf_path")
                },
                "hf_data_repo": HF_DATA_REPO,
                "hf_data_prefix": HF_DATA_PREFIX,
                "raw_completions_prefix": f"{HF_DATA_PREFIX}/eval_results/dose_matched/cells",
                "wandb": "none (no training; plan v2 §10)",
                "final_commit_sha": _git_sha(),
            },
            "hostname": socket.gethostname(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        emit_results = (not self.dry_run) and covers and len(complete) == len(all_cells)
        if emit_results:
            if self.hf_upload:
                _upload_or_raise(
                    self.slab_root / SELECTION_RELPATH,
                    repo_type="dataset",
                    repo_id=HF_DATA_REPO,
                    path_in_repo=(
                        f"{HF_DATA_PREFIX}/eval_results/dose_matched/band_entry_selection.json"
                    ),
                )
            _write_sentinel(
                self.logs_root,
                kind="epm:results",
                name_slug="dose-epm_results",
                note_obj={"event": "dose_matched_complete", **summary},
                version=DOSE_RESULTS_MARKER_VERSION,
            )
        else:
            _write_sentinel(
                self.logs_root,
                kind="epm:progress",
                name_slug=f"dose-shard-gpu{self.gpu_id}-done",
                note_obj={"event": "dose_shard_complete", **summary},
            )


# ----- predictor-v3 follow-up round (plans/v3.md) ---------------------------------


class PredictorV3Runner(Dispatcher):
    """Plan-v3 retrain-with-sub-epoch-checkpoints runner (followup
    onpolicy-leakage-predictor). Reuses the Dispatcher's eval/upload/mini-judge/
    panel helpers verbatim; adds the sub-epoch save cadence, the per-cell
    band-entry self-eval that locates the +0.60 matched-install checkpoint, and
    the source-side baseline read. Every phase's cell list derives from --cells
    (unified smoke = sweep with one cell). Two train arms only (A canned + B
    on-policy; arm C dropped, plan §4.4)."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self._floor_n: int | None = getattr(args, "floor_n", None)

    # ----- cell-state (v3 namespace) ---------------------------------------------

    def _v3_state_path(self, source: str, arm: str, seed: int) -> Path:
        return (
            self.slab_root / "onpolicy_predictor" / "_cellstate" / f"{source}__{arm}__{seed}.json"
        )

    def _read_v3_state(self, source: str, arm: str, seed: int) -> dict | None:
        path = self._v3_state_path(source, arm, seed)
        return json.loads(path.read_text()) if path.exists() else None

    def _write_v3_state(self, source: str, arm: str, seed: int, record: dict) -> None:
        path = self._v3_state_path(source, arm, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2))

    # ----- source-side baseline read (NEW, plan §4.2) ----------------------------

    def run_source_baseline(self, source: str) -> Path:
        """Base-model agreement rate for ONE source on the 60-claim pool (no
        instruction) -> source_baseline/<source>.json. Doubles as predictor (a)'s
        base-prior covariate (§4.2). Reuses eval_panel (single-persona panel) +
        the dispatcher's mini-judge, then summarizes via predictor_v3."""
        from explore_persona_space.experiments.sycophancy_onpolicy_612 import predictor_v3

        out_dir = self.slab_root / "onpolicy_predictor" / "source_baseline_eval" / source
        panel_set = self._source_panel_set(source)
        self._eval_subprocess(
            model_tag=f"source_baseline:{source}",
            out_dir=out_dir,
            panel_set=panel_set,
            claims=self._audited_claims(),
            seed=42,
            hub_model_id=BASE_MODEL,
            panel_subset=source,
            sentinel_name=f"v3-source-baseline-{source}",
        )
        # Judge the single panel file pod-side (sanctioned gate moment) and write
        # the judgments in the schema predictor_v3.source_baseline_summary reads.
        panel_file = out_dir / f"sycophancy_eval_{source}.json"
        judgments_dir = out_dir / "judgments"
        judgments_dir.mkdir(parents=True, exist_ok=True)
        verdicts = self._mini_judge_verdicts(panel_file)
        (judgments_dir / f"{source}.json").write_text(
            json.dumps({"panel": source, "n_verdicts": len(verdicts), "verdicts": verdicts})
        )
        summary = predictor_v3.source_baseline_summary(judgments_dir, source)
        rec_dir = self.slab_root / "onpolicy_predictor" / "source_baseline"
        rec_dir.mkdir(parents=True, exist_ok=True)
        rec_path = rec_dir / f"{source}.json"
        rec_path.write_text(json.dumps({**summary, "git_commit_sha": _git_sha()}, indent=2))
        log.info(
            "[%s] [phase=source_baseline] base agreement=%.3f risk=%s -> %s",
            source,
            summary["base_agreement_rate"],
            summary["yield_risk_class"],
            rec_path,
        )
        if self.hf_upload:
            _upload_or_raise(
                rec_path,
                repo_type="dataset",
                repo_id=HF_DATA_REPO,
                path_in_repo=f"{V3_HF_DATA_PREFIX}/source_baseline/{source}.json",
            )
        return rec_path

    def _source_panel_set(self, source: str) -> Path:
        """A single-persona panel_set for the source-baseline read (the source's
        own roster prompt). Reuses the v1 panel_set's prompt where present."""
        panel_set = self._resolve_panel_set()
        personas = json.loads(panel_set.read_text())["personas"]
        if source not in personas:
            raise KeyError(f"source {source!r} not in panel_set {panel_set} — cannot read baseline")
        path = self.data_root / "panel" / f"panel_set_source_{source}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "provenance": "v3_source_baseline",
                    "personas": {source: {"prompt": personas[source]["prompt"]}},
                },
                indent=2,
            )
        )
        return path

    def _mini_judge_verdicts(self, panel_file: Path) -> list[dict]:
        """Haiku verdicts list (judge_pass schema rows) for one panel eval JSON."""
        import asyncio

        from explore_persona_space.experiments.sycophancy_onpolicy_612.judge import (
            judge_batch,
            serialize_verdicts,
        )

        payload = json.loads(panel_file.read_text())
        records = payload["completions"]
        rollouts = [{"wrong_claim": r["claim"], "completion": r["completion"]} for r in records]
        verdicts = asyncio.run(
            judge_batch(rollouts, model=JUDGE_MODEL, max_concurrency=self.args.judge_concurrency)
        )
        rows = serialize_verdicts(verdicts)
        for rec, v in zip(records, rows, strict=True):
            v["claim_idx"] = rec["claim_idx"]
            v["rollout_idx"] = rec["rollout_idx"]
        return rows

    # ----- the per-cell predictor-v3 path -----------------------------------------

    def _v3_pool(self, source: str, arm: str) -> Path:
        """Build/resolve the floor-N training pool for one (source, arm).

        arm_canned: the frozen #411 pool subset to floor-N positives (the data
        construction control). arm_onpolicy: tiered_positives_v3 floor-N +
        proportional negatives. Equalize-down to ``self._floor_n`` when set; else
        the realized fill (single-cell smoke / per-source run)."""
        from explore_persona_space.experiments.sycophancy_onpolicy_612.predictor_v3 import (
            save_steps_for,
        )

        pool = self._pool_build_subprocess_v3(source, arm)
        rows = sum(1 for line in pool.read_text().splitlines() if line.strip())
        self._last_pool_rows = rows
        self._last_save_steps = save_steps_for(rows)
        return pool

    def _pool_build_subprocess_v3(self, source: str, arm: str) -> Path:
        """Spawn the v3 pool builder in a FRESH subprocess (vLLM teardown
        isolation). Reuses build_onpolicy_pool's v3 entrypoint (--stage v3)."""
        out_dir = self.data_root / "onpolicy_predictor" / "training_pools" / arm / source
        pool = out_dir / "train_pool.jsonl"
        if pool.exists():
            log.info("[%s:%s] v3 pool exists — idempotent skip", source, arm)
            return pool
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.sycophancy_onpolicy_612.build_predictor_v3_pool",
            "--source",
            source,
            "--arm",
            arm,
            "--data-root",
            str(self.data_root),
            "--out-dir",
            str(out_dir),
            "--judge-concurrency",
            str(self.args.judge_concurrency),
        ]
        if self._floor_n is not None:
            cmd += ["--floor-n", str(self._floor_n)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu_id), "TQDM_DISABLE": "1"}
        log.info("[%s:%s] [phase=pool_build] spawning: %s", source, arm, " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)
        if not pool.exists():
            raise RuntimeError(f"v3 pool build exited 0 but {pool} missing")
        return pool

    def _band_entry_from_trajectory(
        self, source: str, arm: str, seed: int, adapter_dir: Path, cell_dir: Path
    ) -> dict:
        """Self-panel eval EACH sub-epoch checkpoint -> earliest with self-Δ >=
        +0.60 (plan §4.3 / N2). Returns the band-entry record (selected step,
        adjacent steps for the §4.5 sensitivity read, per-step self rates +
        deltas, install gap reported per cell)."""
        from explore_persona_space.train.sft import merge_lora

        ckpts = _resolve_step_checkpoints(adapter_dir)
        base_self = self._base_self_rate(source)
        steps_sorted = sorted(ckpts)
        per_step: dict[int, dict] = {}
        traj_root = cell_dir / "trajectory"
        for step in steps_sorted:
            ckpt_dir = ckpts[step]
            if not (ckpt_dir / "adapter_config.json").exists():
                raise RuntimeError(
                    f"[{cell_id(source, arm, seed)}] step-{step} checkpoint {ckpt_dir} has no "
                    f"adapter_config.json — save_strategy='steps' did not save a PEFT adapter"
                )
            _ensure_tokenizer_files(ckpt_dir, adapter_dir)
            merged_tmp = adapter_dir.parent / f"merged_step_{step}"
            log.info(
                "[%s:%s:%d] [phase=trajectory] step_%d merge+self-eval",
                source,
                arm,
                seed,
                step,
            )
            merge_lora(
                base_model_path=BASE_MODEL,
                adapter_path=str(ckpt_dir),
                output_dir=str(merged_tmp),
                gpu_id=self.gpu_id,
            )
            step_dir = traj_root / f"step_{step}"
            self._eval_subprocess(
                model_tag=f"{source}:{arm}:{seed}:step{step}",
                out_dir=step_dir,
                panel_set=self._resolve_panel_set(),
                claims=self._audited_claims(),
                seed=seed,
                merged_dir=merged_tmp,
                panel_subset=source,
                sentinel_name=f"v3-trajectory-{source}-{arm}-{seed}-step{step}",
            )
            shutil.rmtree(merged_tmp, ignore_errors=False)
            self_rate = self._mini_judge_panel_file(step_dir / f"sycophancy_eval_{source}.json")[
                "rate"
            ]
            delta = self_rate - base_self
            per_step[step] = {"self_rate": self_rate, "self_delta": delta}

        entry_step: int | None = None
        for step in steps_sorted:
            if per_step[step]["self_delta"] >= BAND_ENTRY_THRESHOLD:
                entry_step = step
                break
        # Adjacent checkpoints for the §4.5 sensitivity read (one below / one above).
        adjacent: dict[str, int | None] = {"below": None, "above": None}
        if entry_step is not None:
            idx = steps_sorted.index(entry_step)
            adjacent["below"] = steps_sorted[idx - 1] if idx > 0 else None
            adjacent["above"] = steps_sorted[idx + 1] if idx + 1 < len(steps_sorted) else None
        max_delta = max(v["self_delta"] for v in per_step.values())
        return {
            "source": source,
            "arm": arm,
            "seed": seed,
            "base_self_rate": base_self,
            "threshold": BAND_ENTRY_THRESHOLD,
            "checkpoint_steps": steps_sorted,
            "per_step": {str(s): per_step[s] for s in steps_sorted},
            "band_entry_step": entry_step,
            "band_entry_status": "entered" if entry_step is not None else "never_crossed_band",
            "adjacent_steps": adjacent,
            "max_self_delta": max_delta,
        }

    def _base_self_rate(self, source: str) -> float:
        """Base-model self agreement rate for the source (from the source-baseline
        read; falls back to the v1 panel_set base_rate if the read is absent)."""
        rec_path = self.slab_root / "onpolicy_predictor" / "source_baseline" / f"{source}.json"
        if rec_path.exists():
            return float(json.loads(rec_path.read_text())["base_agreement_rate"])
        panel_set = self._resolve_panel_set()
        personas = json.loads(panel_set.read_text())["personas"]
        return float(personas[source]["base_rate"])

    def run_v3_cell(self, source: str, arm: str, seed: int) -> dict:
        """ONE predictor-v3 train cell: pool -> train (sub-epoch saves) -> band
        entry -> full-panel eval at band-entry (+adjacent) -> upload. Smoke and
        sweep both land here."""
        cid = cell_id(source, arm, seed)
        prior = self._read_v3_state(source, arm, seed)
        if prior is not None and prior.get("status") == "complete":
            log.info("[%s] v3 cell-state complete — idempotent skip", cid)
            return prior
        t0 = time.time()
        cell_dir = v3_cell_dir(self.slab_root, source, arm, seed)
        record: dict = {
            "cell": cid,
            "source": source,
            "arm": arm,
            "seed": seed,
            "gpu_id": self.gpu_id,
            "eval_out_dir": str(cell_dir),
            "git_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
        }
        log.info("=" * 70)
        log.info("[%s] V3 CELL START -> %s", cid, cell_dir)

        if self.dry_run:
            record.update(status="dry_run", wall_seconds=round(time.time() - t0, 1))
            self._write_v3_state(source, arm, seed, record)
            return record

        # 1. source-side baseline read (idempotent; needed for self-Δ + predictor a)
        self.run_source_baseline(source)

        # 2. pool (floor-N) + sub-epoch save cadence
        pool = self._v3_pool(source, arm)
        record["pool_path"] = str(pool)
        record["pool_rows"] = self._last_pool_rows
        record["save_steps_plan"] = self._last_save_steps
        save_steps = self._last_save_steps["save_steps"]

        # 3. train with sub-epoch checkpoints (band-entry checkpoint uploaded later)
        hub_base = f"adapters/issue_612/onpolicy_predictor/{arm}/{source}_seed{seed}"
        adapter_dir, merged_dir = self._train_and_merge(
            source,
            arm,
            seed,
            pool,
            save_strategy="steps",
            save_steps=save_steps,
            upload_epoch_checkpoints=False,
            run_name=f"issue612_v3_{arm}_{source}_seed{seed}",
            hub_adapter_base=hub_base,
        )
        record["adapter_dir"] = str(adapter_dir)
        record["adapter_hf_path"] = hub_base
        shutil.rmtree(merged_dir, ignore_errors=False)  # final-merge unused; band-entry re-merges

        # 4. locate band entry from the sub-epoch trajectory self-evals
        band = self._band_entry_from_trajectory(source, arm, seed, adapter_dir, cell_dir)
        (cell_dir / "band_entry.json").write_text(json.dumps(band, indent=2))
        record["band_entry"] = band

        # 5. full-panel eval at the band-entry checkpoint (+ adjacent for sensitivity)
        ckpts = _resolve_step_checkpoints(adapter_dir)
        eval_steps = self._steps_to_full_eval(band, ckpts)
        record["full_panel_eval_steps"] = eval_steps
        for step in eval_steps:
            self._full_panel_eval_at_step(
                source, arm, seed, ckpts[step], cell_dir / f"matched_install_step_{step}"
            )
        # upload the band-entry checkpoint adapter (only that one, per Upload Policy)
        if self.hf_upload and band["band_entry_step"] is not None:
            _upload_checkpoint_or_raise(
                ckpts[band["band_entry_step"]],
                f"{hub_base}/checkpoint-step{band['band_entry_step']}",
            )
        record["band_entry_checkpoint_hf"] = (
            f"{hub_base}/checkpoint-step{band['band_entry_step']}"
            if band["band_entry_step"] is not None
            else None
        )

        # 6. upload the cell tree (per-panel JSONs + raw_completions + trajectory)
        log.info("[%s] [phase=upload]", cid)
        record["hub_eval_tree"] = self._upload_v3_cell_tree(
            cell_dir, f"cells/{arm}/{source}/seed_{seed}"
        )
        # disk discipline: drop the non-band sub-epoch checkpoints now they're read
        self._prune_nonband_checkpoints(adapter_dir, band)

        record.update(status="complete", wall_seconds=round(time.time() - t0, 1))
        self._write_v3_state(source, arm, seed, record)
        _write_sentinel(
            self.logs_root,
            kind="epm:progress",
            name_slug=f"v3-cell-{source}-{arm}-{seed}",
            note_obj={"event": "v3_cell_complete", **record},
        )
        return record

    def _steps_to_full_eval(self, band: dict, ckpts: dict[int, Path]) -> list[int]:
        """The band-entry step plus its adjacent steps (sensitivity, §4.5). When a
        cell never crossed the band, evaluate its max-self-delta (closest-approach)
        checkpoint so the cell is still reported descriptively (never silently
        skipped, plan §5 / Methodology concern #2)."""
        steps_sorted = sorted(ckpts)
        if band["band_entry_step"] is not None:
            chosen = {band["band_entry_step"]}
            for s in (band["adjacent_steps"]["below"], band["adjacent_steps"]["above"]):
                if s is not None:
                    chosen.add(s)
        else:
            # closest-approach: the max-self-delta step
            best = max(steps_sorted, key=lambda s: band["per_step"][str(s)]["self_delta"])
            chosen = {best}
        return sorted(chosen)

    def _full_panel_eval_at_step(
        self, source: str, arm: str, seed: int, ckpt_dir: Path, out_dir: Path
    ) -> None:
        from explore_persona_space.train.sft import merge_lora

        _ensure_tokenizer_files(ckpt_dir, ckpt_dir.parent)
        merged_tmp = ckpt_dir.parent / f"merged_full_{ckpt_dir.name}"
        merge_lora(
            base_model_path=BASE_MODEL,
            adapter_path=str(ckpt_dir),
            output_dir=str(merged_tmp),
            gpu_id=self.gpu_id,
        )
        step = int(ckpt_dir.name.split("-")[1])
        self._eval_subprocess(
            model_tag=f"{source}:{arm}:{seed}:step{step}:fullpanel",
            out_dir=out_dir,
            panel_set=self._resolve_panel_set(),
            claims=self._audited_claims(),
            seed=seed,
            merged_dir=merged_tmp,
            sentinel_name=f"v3-fullpanel-{source}-{arm}-{seed}-step{step}",
        )
        shutil.rmtree(merged_tmp, ignore_errors=False)

    def _prune_nonband_checkpoints(self, adapter_dir: Path, band: dict) -> None:
        """Delete sub-epoch checkpoints other than the band-entry + adjacent ones
        AFTER they've been read + the band-entry adapter uploaded (disk quota)."""
        keep_steps: set[int] = set()
        if band["band_entry_step"] is not None:
            keep_steps.add(band["band_entry_step"])
            for s in (band["adjacent_steps"]["below"], band["adjacent_steps"]["above"]):
                if s is not None:
                    keep_steps.add(s)
        for ck in adapter_dir.glob("checkpoint-*"):
            if ck.is_dir() and int(ck.name.split("-")[1]) not in keep_steps:
                shutil.rmtree(ck, ignore_errors=True)

    def _upload_v3_cell_tree(self, local: Path, rel: str) -> str | None:
        """v3 cell-tree upload to the V3 data prefix (raw_completions presence
        asserted first; Upload Policy — before pod termination)."""
        if not self.hf_upload:
            log.info("[%s] HF upload disabled — skipping v3 cell-tree upload", rel)
            return None
        raw_files = list(local.rglob("raw_completions/*.json"))
        if not raw_files:
            raise RuntimeError(
                f"no raw_completions/*.json under {local} — eval wrote nothing; "
                f"refusing to upload an empty v3 cell tree"
            )
        return _upload_or_raise(
            local,
            repo_type="dataset",
            repo_id=HF_DATA_REPO,
            path_in_repo=f"{V3_HF_DATA_PREFIX}/eval_results/{rel}",
        )

    def v3_finalize(
        self, cells: list[tuple[str, str, int]], all_cells: list[tuple[str, str, int]]
    ) -> None:
        """epm:results (v3 stream) ONLY for a complete non-dry run covering all
        cells; else an epm:progress shard sentinel (the cross-phase contract)."""
        states = {cell_id(*c): self._read_v3_state(*c) for c in all_cells}
        ok = ("complete", "dry_run") if self.dry_run else ("complete",)
        complete = {k: v for k, v in states.items() if v and v.get("status") in ok}
        adapter_paths = {
            k: v.get("band_entry_checkpoint_hf")
            for k, v in complete.items()
            if v.get("band_entry_checkpoint_hf")
        }
        wandb_run_names = {
            k: f"issue612_v3_{v['arm']}_{v['source']}_seed{v['seed']}" for k, v in complete.items()
        }
        summary = {
            "issue": 612,
            "stage": "predictor-v3",
            "followup_label": "onpolicy-leakage-predictor",
            "gpu_id": self.gpu_id,
            "shard_cells": [cell_id(*c) for c in cells],
            "all_cells": [cell_id(*c) for c in all_cells],
            "n_complete": len(complete),
            "n_all": len(all_cells),
            "dry_run": self.dry_run,
            "floor_n": self._floor_n,
            "band_entry_steps": {
                k: (v.get("band_entry") or {}).get("band_entry_step") for k, v in complete.items()
            },
            "band_entry_status": {
                k: (v.get("band_entry") or {}).get("band_entry_status") for k, v in complete.items()
            },
            "eval_paths": {k: v.get("eval_out_dir") for k, v in complete.items()},
            "wall_seconds_by_cell": {k: v.get("wall_seconds") for k, v in complete.items()},
            "reproducibility_card": {
                "base_model": BASE_MODEL,
                "training": "LoRA r32 a64 rslora lr1e-5 3ep whole-completion, sub-epoch saves",
                "hf_model_repo": HF_MODEL_REPO,
                "adapter_paths": adapter_paths,
                "wandb_project": os.environ.get("WANDB_PROJECT", "issue612_sycophancy_onpolicy"),
                "wandb_run_names": wandb_run_names,
                "hf_data_repo": HF_DATA_REPO,
                "hf_data_prefix": V3_HF_DATA_PREFIX,
                "raw_completions_prefix": f"{V3_HF_DATA_PREFIX}/eval_results/cells",
                "final_commit_sha": _git_sha(),
            },
            "hostname": socket.gethostname(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        emit = (not self.dry_run) and len(complete) == len(all_cells)
        if emit:
            _write_sentinel(
                self.logs_root,
                kind="epm:results",
                name_slug="v3-epm_results",
                note_obj={"event": "predictor_v3_complete", **summary},
                version=V3_RESULTS_MARKER_VERSION,
            )
        else:
            _write_sentinel(
                self.logs_root,
                kind="epm:progress",
                name_slug=f"v3-shard-gpu{self.gpu_id}-done",
                note_obj={"event": "v3_shard_complete", **summary},
            )


def _parse_v3_cells(raw: str) -> list[tuple[str, str, int]]:
    """Parse predictor-v3 cells: <source>:<arm>:<seed> with arm in V3_TRAIN_ARMS."""
    cells = parse_cells(raw)
    bad = [cell_id(*c) for c in cells if c[1] not in V3_TRAIN_ARMS]
    if bad:
        raise ValueError(
            f"predictor-v3 trains only {V3_TRAIN_ARMS} (arm C dropped, plan §4.4); bad cells: {bad}"
        )
    return cells


def _v3_production_cells() -> list[tuple[str, str, int]]:
    """The 16-cell predictor-v3 train grid: 4 sources x 2 arms x 2 seeds."""
    from explore_persona_space.experiments.sycophancy_onpolicy_612 import SEEDS, SOURCES

    return [(source, arm, seed) for source in SOURCES for arm in V3_TRAIN_ARMS for seed in SEEDS]


def _run_predictor_v3(args: argparse.Namespace) -> None:
    """The --stage predictor-v3 flow: per-cell pool -> train (sub-epoch saves) ->
    band-entry self-eval -> full-panel eval at matched install -> upload ->
    finalize. Every phase's cell list derives from --cells (unified smoke=sweep).
    The decorrelated-panel build + bake-off analysis are separate VM/CPU phases
    (panel_select_v3 / issue612_predictor_bakeoff) run off-pod."""
    cells: list[tuple[str, str, int]] = args.cells
    all_cells = (
        args.all_cells
        if args.all_cells is not None
        else (cells if args.dry_run else _v3_production_cells())
    )
    missing = [c for c in cells if c not in all_cells]
    if missing:
        raise ValueError(f"--cells entries missing from --all-cells: {missing}")

    gates_fire = args.smoke_gates and V3_SMOKE_CELL in cells and not args.dry_run
    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN not in environment — .env not loaded?")
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not in environment — v3 pool judge / self-eval / source baseline"
        )

    log.info(
        "[phase=dispatch] stage=predictor-v3 cells=%s all_cells=%d gpu_id=%d dry_run=%s floor_n=%s",
        [cell_id(*c) for c in cells],
        len(all_cells),
        args.gpu_id,
        args.dry_run,
        getattr(args, "floor_n", None),
    )
    runner = PredictorV3Runner(args)

    if not args.skip_prefetch and not args.dry_run:
        log.info("[phase=prefetch] fetch + pin inputs for this shard's v3 cells")
        from explore_persona_space.experiments.sycophancy_onpolicy_612.prefetch_inputs import (
            prefetch,
        )

        prefetch(
            cells=cells,
            data_root=args.data_root,
            adapters_root=args.adapters_root,
            smoke_gate=args.smoke_gates,
        )

    for source, arm, seed in cells:
        try:
            runner.run_v3_cell(source, arm, seed)
        except Exception as e:
            _write_sentinel(
                args.logs_root,
                kind="epm:progress",
                name_slug=f"v3-cell-{source}-{arm}-{seed}-FAILED",
                note_obj={
                    "event": "v3_cell_failed",
                    "cell": cell_id(source, arm, seed),
                    "exception_type": type(e).__name__,
                    "exception_msg": str(e)[:2000],
                },
            )
            log.exception("[%s] v3 cell failed", cell_id(source, arm, seed))
            raise
        if (source, arm, seed) == V3_SMOKE_CELL and gates_fire:
            runner._gate_g2()  # self-implant install floor (the round's reason to exist)

    if args.finalize:
        runner.v3_finalize(cells, all_cells)


def _run_dose_matched(args: argparse.Namespace) -> None:
    """The --stage dose-matched flow: selection -> (prefetch+K2 preflight) ->
    per-cell fetch/merge/eval/upload -> G1-dm gate after the smoke cell ->
    finalize. Every phase's cell list derives from --cells."""
    selection_path = ensure_band_entry_selection(args.slab_root)
    selection = json.loads(selection_path.read_text())
    evaluated_ids: list[str] = selection["evaluated_cells"]
    cells: list[tuple[str, str, int]] = args.cells
    bad = [cell_id(*c) for c in cells if cell_id(*c) not in evaluated_ids]
    if bad:
        raise ValueError(
            f"--cells not in the registered dose-matched set: {bad} "
            f"(evaluated cells: {evaluated_ids}; adding/removing cells is must-ask, plan v2)"
        )
    all_cells = (
        args.all_cells if args.all_cells is not None else parse_cells(",".join(evaluated_ids))
    )
    missing = [c for c in cells if c not in all_cells]
    if missing:
        raise ValueError(f"--cells entries missing from --all-cells: {missing}")

    gates_fire = args.smoke_gates and DOSE_SMOKE_CELL in cells and not args.dry_run
    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN not in environment — .env not loaded?")
    if gates_fire and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not in environment — G1-dm mini-judge needs it")

    log.info(
        "[phase=dispatch] stage=dose-matched cells=%s all_cells=%d gpu_id=%d dry_run=%s",
        [cell_id(*c) for c in cells],
        len(all_cells),
        args.gpu_id,
        args.dry_run,
    )
    runner = DoseMatchedRunner(args, selection)
    if not args.skip_prefetch and not args.dry_run:
        runner.dose_prefetch()
        runner.dose_hub_preflight(cells)

    for source, arm, seed in cells:
        try:
            runner.run_dose_cell(source, arm, seed)
        except Exception as e:
            _write_sentinel(
                args.logs_root,
                kind="epm:progress",
                name_slug=f"dose-cell-{source}-{arm}-{seed}-FAILED",
                note_obj={
                    "event": "dose_cell_failed",
                    "cell": cell_id(source, arm, seed),
                    "exception_type": type(e).__name__,
                    "exception_msg": str(e)[:2000],
                },
            )
            log.exception("[%s] dose cell failed", cell_id(source, arm, seed))
            raise
        if (source, arm, seed) == DOSE_SMOKE_CELL and gates_fire:
            runner.gate_g1_dm()

    if args.finalize:
        runner.dose_finalize(cells, all_cells)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #612 dispatcher — unified smoke = sweep with one cell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cells",
        type=parse_cells,
        required=True,
        help="Comma-separated <source>:<arm>:<seed> cells THIS process runs "
        "(smoke: villain:arm_onpolicy:42).",
    )
    parser.add_argument(
        "--all-cells",
        type=parse_cells,
        default=None,
        help="The full cell list the SWEEP comprises. Default: the 28-cell production "
        "grid for non-dry runs (a subset shard cannot emit epm:results), --cells for "
        "dry runs.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("data/issue_612"))
    parser.add_argument("--adapters-root", type=Path, default=Path("/workspace/adapters_411"))
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_612"))
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_612"))
    parser.add_argument("--logs-root", type=Path, default=Path("/workspace/logs"))
    parser.add_argument(
        "--out-root",
        dest="slab_root_alias",
        type=Path,
        default=None,
        help="Alias accepted for §10 dispatch-row compatibility (maps to --slab-root when given).",
    )
    parser.add_argument(
        "--stage",
        choices=("production", "dose-matched", "predictor-v3"),
        default="production",
        help="production = the parent 28-cell grid; dose-matched = the plan-v2 "
        "band-entry eval-only round; predictor-v3 = the plan-v3 retrain-with-sub-"
        "epoch-checkpoints round (16 train cells, 4 sources x 2 arms x 2 seeds).",
    )
    parser.add_argument(
        "--floor-n",
        type=int,
        default=None,
        help="predictor-v3 only: equalize-down positive count (floor-N from the "
        "80%% yield gate). When unset, each cell uses its realized fill (smoke / "
        "per-source). The driver passes the cross-source min after the yield phase.",
    )
    parser.add_argument(
        "--finalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="dose-matched only: run the end-of-invocation aggregation/sentinel step "
        "(shards pass --no-finalize; the driver's last invocation finalizes).",
    )
    parser.add_argument(
        "--smoke-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the smoke gates after the smoke cell (production: G1+G2 after "
        "villain:arm_onpolicy:42; dose-matched: G1-dm after villain:arm_canned:42).",
    )
    parser.add_argument(
        "--allow-provisional-panel",
        action="store_true",
        help="Permit the mandatory-11 provisional panel when P2j's panel_set.json is "
        "absent (smoke only; production evals must use the selected panel).",
    )
    parser.add_argument("--skip-prefetch", action="store_true")
    parser.add_argument("--hf-upload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--judge-concurrency", type=int, default=16)
    args = parser.parse_args(argv)
    if args.slab_root_alias is not None:
        args.slab_root = args.slab_root_alias

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
    os.environ.setdefault("WANDB_PROJECT", "issue612_sycophancy_onpolicy")
    os.environ.setdefault("TQDM_DISABLE", "1")

    if args.stage == "dose-matched":
        _run_dose_matched(args)
        log.info("[phase=done]")
        return 0

    if args.stage == "predictor-v3":
        _run_predictor_v3(args)
        log.info("[phase=done]")
        return 0

    cells: list[tuple[str, str, int]] = args.cells
    all_cells = resolve_all_cells(cells, all_cells_arg=args.all_cells, dry_run=args.dry_run)
    missing = [c for c in cells if c not in all_cells]
    if missing:
        raise ValueError(f"--cells entries missing from --all-cells: {missing}")

    # Credential checks gated by the phases this invocation actually runs
    # (a --dry-run --skip-prefetch grid walk needs neither).
    needs_hf = (not args.skip_prefetch) or (not args.dry_run)
    gates_fire = args.smoke_gates and SMOKE_CELL in cells and not args.dry_run
    needs_judge = gates_fire or (
        not args.dry_run and any(a in ("arm_onpolicy", "arm_prefix") for _, a, _ in cells)
    )
    if needs_hf and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN not in environment — .env not loaded?")
    if needs_judge and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not in environment — pool-build judge filter / smoke gates need it"
        )

    log.info(
        "[phase=dispatch] cells=%s all_cells=%d gpu_id=%d dry_run=%s",
        [cell_id(*c) for c in cells],
        len(all_cells),
        args.gpu_id,
        args.dry_run,
    )
    dispatcher = Dispatcher(args)

    if not args.skip_prefetch:
        log.info("[phase=prefetch] fetch + pin inputs for this shard's cells")
        from explore_persona_space.experiments.sycophancy_onpolicy_612.prefetch_inputs import (
            prefetch,
        )

        prefetch(
            cells=cells,
            data_root=args.data_root,
            adapters_root=args.adapters_root,
            smoke_gate=args.smoke_gates,
        )

    for source, arm, seed in cells:
        try:
            dispatcher._run_cell(source, arm, seed)
        except Exception as e:
            _write_sentinel(
                args.logs_root,
                kind="epm:progress",
                name_slug=f"cell-{source}-{arm}-{seed}-FAILED",
                note_obj={
                    "event": "cell_failed",
                    "cell": cell_id(source, arm, seed),
                    "exception_type": type(e).__name__,
                    "exception_msg": str(e)[:2000],
                },
            )
            log.exception("[%s] cell failed", cell_id(source, arm, seed))
            raise
        if (source, arm, seed) == SMOKE_CELL and args.smoke_gates and not args.dry_run:
            dispatcher._gate_g1()
            dispatcher._gate_g2()

    dispatcher.finalize(cells, all_cells)
    log.info("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
