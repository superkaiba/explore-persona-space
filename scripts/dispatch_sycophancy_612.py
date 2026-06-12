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
    BASE_MODEL,
    G1_TOL,
    G2_DELTA_FLOOR,
    HF_DATA_PREFIX,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    JUDGE_MODEL,
    MANDATORY_PANEL,
    PARITY_PANELS,
    TRAIN_ARMS,
    cell_id,
    cell_slab_dir,
    full_production_cells,
    parse_cells,
    pool_dir,
    repo_root_from_module,
)

log = logging.getLogger("dispatch_sycophancy_612")

SMOKE_CELL = ("villain", "arm_onpolicy", 42)
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
) -> Path:
    """One poll_pipeline-conforming sentinel (sentinel_schema_version/kind/version)."""
    logs_root.mkdir(parents=True, exist_ok=True)
    path = logs_root / f"issue-612-{name_slug}-{int(time.time())}.json"
    payload: dict = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
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

    def _train_and_merge(self, source: str, arm: str, seed: int, pool: Path) -> tuple[Path, Path]:
        """#411 recipe held FIXED (plan §4 cfg). Deviations carried from the
        ported #608 chain, all arm-symmetric + optimization-neutral:
        save_only_model=True (adapter-only epoch checkpoints) and HF uploads
        moved to the dispatcher (fail-loud) — named in the implementer report."""
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
            run_name=f"issue612_{arm}_{source}_seed{seed}",
            report_to="wandb",
            save_strategy="epoch",
            save_total_limit=None,
            save_only_model=True,
            gradient_checkpointing=True,
            packing=False,
            hf_upload=False,  # dispatcher owns uploads (fail-loud, checkpoint handling)
        )
        log.info("[%s:%s:%d] [phase=train] train_lora -> %s", source, arm, seed, adapter_dir)
        train_lora(
            base_model_path=BASE_MODEL,
            data_path=str(pool),
            output_dir=str(adapter_dir),
            cfg=cfg,
        )
        if not list(adapter_dir.glob("*.safetensors")):
            raise RuntimeError(f"[{source}:{arm}:{seed}] no .safetensors in {adapter_dir}")

        if self.hf_upload:
            hub_base = f"adapters/issue_612/{arm}/{source}_seed{seed}"
            _upload_or_raise(
                adapter_dir, repo_type="model", repo_id=HF_MODEL_REPO, path_in_repo=hub_base
            )
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
        self._eval_subprocess(
            model_tag=f"{source}:{arm}:{seed}",
            out_dir=cell_dir,
            panel_set=self._resolve_panel_set(),
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
        "--smoke-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run G1+G2 after the smoke cell (fires only when villain:arm_onpolicy:42 "
        "is in --cells).",
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
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN not in environment — .env not loaded?")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not in environment — judge gates need it")

    cells: list[tuple[str, str, int]] = args.cells
    all_cells = resolve_all_cells(cells, all_cells_arg=args.all_cells, dry_run=args.dry_run)
    missing = [c for c in cells if c not in all_cells]
    if missing:
        raise ValueError(f"--cells entries missing from --all-cells: {missing}")

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
