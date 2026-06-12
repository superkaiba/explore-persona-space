#!/usr/bin/env python3
"""Task #627 Phase 1 — matched-install bystander panel (pod-side, GPU).

UNIFIED smoke = sweep with one cell: ``--cells 1`` runs the FIRST manifest
cell (the registered smoke cell villain:contrastive_dense:18) through the SAME
``_run_cell`` path as the production sweep — same dispatcher, same shard
subprocess shape, same prefetch/merge/eval/upload phases, same sentinel
surface. Every phase's cell list derives from ``--cells``:

    [phase=p0_manifest]  load + validate the Phase-0 cell manifest; resolve
                         the requested subset.
    [phase=p1_prefetch]  probe-file fetch + SHA-256 pin assert (the #608
                         ``eval_50.jsonl`` pin), Hub existence check for the
                         REQUESTED cells' adapters, smoke-cell adapter_config
                         gauge assert (r=32, alpha=64, use_rslora=True).
    [phase=p2_panel]     shard the requested cells over ``--gpus`` and run
                         each shard as a subprocess of THIS script
                         (``--shard-worker``); per cell: adapter download ->
                         merge_lora -> eval_one_source (vLLM, FULL 24-panel,
                         fresh subprocess — vLLM teardown gotcha) -> rmtree ->
                         per-cell HF upload -> cell-state record + sentinel.
    [phase=p3_aggregate] cell-state aggregation; ``epm:results`` sentinel ONLY
                         when the full 24-cell registered grid is complete on
                         a non-dry run (subsets/smokes write ``epm:progress``).
    [phase=done]         terminal line (exactly once, main log only).

Smoke-cell parity gate (plan §7 gate 1): after villain:contrastive_dense:18
completes, an inline Haiku mini-judge pass over its SOURCE-panel 500
completions; regenerated own-rate must sit within ±0.08 of the committed
0.416. FAIL -> HALT sentinel + non-zero exit (eval-path bug; never interpret a
failed-parity panel). The gate is idempotent via a gate-state file, so the
registered smoke-then-sweep sequence judges once.

GPU sharding (the ``+gpu_id`` CVD-clobber gotcha): shard workers run with
CUDA_VISIBLE_DEVICES UNSET; ``merge_lora(gpu_id=N)`` pins the merge to the
physical GPU in-process, and each eval subprocess receives
``CUDA_VISIBLE_DEVICES=str(N)`` in its env. Never exported at shard top.

Pod-side discipline: NEVER shells out to scripts/task.py; sentinels under
``--logs-root`` carry poll_pipeline's required keys (sentinel_schema_version=1,
kind, version). load_dotenv() at module top; every subprocess.* gets
``env={**os.environ, ...}``.

Launch (plan §10):
    nohup uv run python scripts/i627_matched_install_panel.py \
        --cells-manifest eval_results/issue_627/matched_install_cells.json \
        --output-root /workspace/issue_627 --gpus 0,1,2,3 \
        > /workspace/logs/issue-627-panel.log 2>&1 &
    # smoke first: same command + --cells 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

from explore_persona_space.experiments.leakage_vs_install_627 import (  # noqa: E402
    BASE_MODEL,
    HF_627_DATA_PREFIX,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    PARITY_TOLERANCE,
    SEED,
    SMOKE_CELL,
    adapter_hub_prefix,
    cell_id,
    load_cells_manifest,
    parse_cell_token,
)
from explore_persona_space.experiments.sycophancy_posonly_608 import (  # noqa: E402
    EXPECTED_SHA256,
    FROZEN_DATA_PREFIX,
)

log = logging.getLogger("i627_panel")

EXPECTED_PANEL_COMPLETIONS = 500  # 50 claims x 10 rollouts
N_PANELS = 24
# §7 disambiguation inherited from #608: post-retry API errors map to NO and
# can only deflate the parity read; >2% error burden is an eval anomaly.
SMOKE_JUDGE_API_ERROR_CEILING = 0.02
EVAL_POOL_REPO_PATH = f"{FROZEN_DATA_PREFIX}/data/wrong_claims/eval_50.jsonl"
RETRY_SLEEPS = (30, 60, 120)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except Exception:
        return "unknown"


def _retry(fn, *, label: str):
    """3-attempt retry with 30/60/120s backoff for transient Hub/network blips."""
    last: Exception | None = None
    for attempt, sleep_s in enumerate((*RETRY_SLEEPS, None)):
        try:
            return fn()
        except Exception as e:
            last = e
            if sleep_s is None:
                break
            log.warning(
                "%s failed (attempt %d: %s) — retrying in %ds", label, attempt + 1, e, sleep_s
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {1 + len(RETRY_SLEEPS)} attempts") from last


def _write_sentinel(
    logs_root: Path,
    *,
    kind: str,
    note_obj: dict,
    name_slug: str,
    gate: str | None = None,
) -> Path:
    """One poll_pipeline-conforming sentinel (required keys: schema/kind/version)."""
    logs_root.mkdir(parents=True, exist_ok=True)
    path = logs_root / f"issue-627-{name_slug}-{int(time.time())}.json"
    payload: dict = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "task_id": 627,
        "by": "pod-driver-627",
        "ts": datetime.now(UTC).isoformat(),
        "note": json.dumps(note_obj, ensure_ascii=False),
    }
    if gate is not None:
        payload["gate"] = gate
        payload["blocks_pipeline"] = True
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("sentinel written: %s (kind=%s)", path, kind)
    return path


def _upload_or_raise(local_path: Path, *, path_in_repo: str) -> str:
    """Fail-loud wrapper around hub._upload (returns '' on failure)."""
    from explore_persona_space.orchestrate.hub import _upload

    hub_path = _upload(
        local_path=local_path,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        upload_as_file=local_path.is_file(),
    )
    if not hub_path:
        raise RuntimeError(
            f"HF upload FAILED: {local_path} -> {HF_DATA_REPO}/{path_in_repo} "
            f"(hub._upload returned empty path; check HF_TOKEN / quota / network)"
        )
    return hub_path


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cell resolution (every phase's cell list derives from --cells)
# ---------------------------------------------------------------------------


def resolve_cells(manifest_cells: list[dict], spec: str) -> list[dict]:
    """``all`` | integer prefix count | comma list of <source>:<arm>:<step>."""
    if spec == "all":
        return list(manifest_cells)
    if spec.isdigit():
        n = int(spec)
        if not 1 <= n <= len(manifest_cells):
            raise ValueError(f"--cells {n} out of range 1..{len(manifest_cells)}")
        return list(manifest_cells[:n])
    by_key = {(c["source"], c["arm"], int(c["step"])): c for c in manifest_cells}
    out = []
    for tok in spec.split(","):
        key = parse_cell_token(tok)
        if key not in by_key:
            raise ValueError(f"cell {tok!r} not in the registered manifest")
        out.append(by_key[key])
    if len({(c["source"], c["arm"], c["step"]) for c in out}) != len(out):
        raise ValueError(f"duplicate cells in --cells {spec!r}")
    return out


def cell_out_dir(slab_root: Path, c: dict) -> Path:
    return slab_root / c["arm"] / c["source"] / f"seed_{SEED}" / "steps" / f"step_{c['step']}"


def _cellstate_path(slab_root: Path, c: dict) -> Path:
    return slab_root / "_cellstate" / f"{c['source']}__{c['arm']}__step{c['step']}.json"


def _gate_state_path(slab_root: Path) -> Path:
    return slab_root / "_gate" / "smoke_parity.json"


# ---------------------------------------------------------------------------
# Phase 1: prefetch + pins
# ---------------------------------------------------------------------------


def prefetch(cells: list[dict], data_root: Path) -> dict:
    """Probe pin + Hub adapter existence for the REQUESTED cells (subset
    threads through this phase: only the requested cells' adapters are
    checked, the probe file is needed by every cell)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    eval_pool = data_root / "wrong_claims" / "eval_50.jsonl"
    if not eval_pool.exists():
        cached = _retry(
            lambda: hf_hub_download(
                repo_id=HF_DATA_REPO, filename=EVAL_POOL_REPO_PATH, repo_type="dataset"
            ),
            label="eval_50.jsonl download",
        )
        eval_pool.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, eval_pool)
    actual = _sha256(eval_pool)
    expected = EXPECTED_SHA256[EVAL_POOL_REPO_PATH]
    if actual != expected:
        raise RuntimeError(
            f"SHA256 pin mismatch for {EVAL_POOL_REPO_PATH}: expected {expected}, got "
            f"{actual} — the HF mirror diverged from the planning-time-verified content "
            f"(incident #600). Do NOT proceed."
        )
    log.info("probe pin OK: %s (sha256=%s)", eval_pool, actual[:12])

    model_files = set(_retry(lambda: list_repo_files(HF_MODEL_REPO), label="list_repo_files"))
    missing = []
    for c in cells:
        prefix = adapter_hub_prefix(c["arm"], c["source"], c["step"])
        for fname in ("adapter_config.json", "adapter_model.safetensors"):
            if f"{prefix}/{fname}" not in model_files:
                missing.append(f"{prefix}/{fname}")
    if missing:
        raise RuntimeError(f"{len(missing)} adapter files missing on the Hub: {missing[:6]}")

    # Gauge assert on ONE adapter_config (the first requested cell): the reuse
    # fitness check (g) — same rsLoRA regime as plan §10 recorded.
    c0 = cells[0]
    prefix0 = adapter_hub_prefix(c0["arm"], c0["source"], c0["step"])
    cfg_path = _retry(
        lambda: hf_hub_download(HF_MODEL_REPO, f"{prefix0}/adapter_config.json"),
        label="adapter_config.json download",
    )
    with open(cfg_path) as f:
        cfg = json.load(f)
    if not (cfg["r"] == 32 and cfg["lora_alpha"] == 64 and cfg.get("use_rslora") is True):
        raise RuntimeError(
            f"adapter_config gauge mismatch vs plan §10 for {prefix0}: r={cfg['r']}, "
            f"alpha={cfg['lora_alpha']}, use_rslora={cfg.get('use_rslora')}"
        )
    return {
        "eval_pool": str(eval_pool),
        "eval_pool_sha256": actual,
        "n_adapter_cells_verified": len(cells),
        "gauge_checked_prefix": prefix0,
        "gauge": {k: cfg.get(k) for k in ("r", "lora_alpha", "use_rslora")},
    }


def download_adapter(c: dict, adapters_root: Path) -> Path:
    """list_repo_files + per-file hf_hub_download (snapshot_download silently
    truncates on >~8k-file repos — never use allow_patterns here)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    prefix = adapter_hub_prefix(c["arm"], c["source"], c["step"])
    dest = adapters_root / prefix
    if (dest / "adapter_config.json").exists() and (dest / "adapter_model.safetensors").exists():
        log.info("[%s] adapter already local: %s", cell_id(c["source"], c["arm"], c["step"]), dest)
        return dest
    repo_files = [
        f
        for f in _retry(lambda: list_repo_files(HF_MODEL_REPO), label="list_repo_files")
        if f.startswith(f"{prefix}/")
    ]
    if not repo_files:
        raise RuntimeError(f"no files under {prefix} on {HF_MODEL_REPO}")
    dest.mkdir(parents=True, exist_ok=True)
    for repo_path in repo_files:
        cached = _retry(
            lambda rp=repo_path: hf_hub_download(repo_id=HF_MODEL_REPO, filename=rp),
            label=f"download {repo_path}",
        )
        shutil.copyfile(cached, dest / Path(repo_path).name)
    for fname in ("adapter_config.json", "adapter_model.safetensors"):
        if not (dest / fname).exists():
            raise RuntimeError(f"adapter {prefix}: {fname} missing after download")
    log.info("adapter fetched: %s (%d files)", dest, len(repo_files))
    return dest


# ---------------------------------------------------------------------------
# Smoke-cell parity gate (plan §7 gate 1)
# ---------------------------------------------------------------------------


def run_smoke_parity_gate(slab_root: Path, smoke_cell_rec: dict) -> dict:
    """Inline Haiku mini-judge over the smoke cell's SOURCE-panel completions
    (the ONE sanctioned pod-side judge moment, inherited from #608)."""
    from explore_persona_space.experiments.sycophancy_implantation_411.judge import (
        judge_batch,
        resolve_model_alias,
    )
    from explore_persona_space.experiments.sycophancy_posonly_608.judge_pass_608 import (
        _load_panel_rollouts,
    )

    source = smoke_cell_rec["source"]
    out_dir = cell_out_dir(slab_root, smoke_cell_rec)
    panel_file = out_dir / f"sycophancy_eval_{source}.json"
    if not panel_file.exists():
        raise RuntimeError(f"smoke gate: source panel file missing {panel_file}")
    _payload, records = _load_panel_rollouts(panel_file)
    if len(records) != EXPECTED_PANEL_COMPLETIONS:
        raise RuntimeError(
            f"smoke gate: {len(records)} completions != {EXPECTED_PANEL_COMPLETIONS}"
        )
    haiku = resolve_model_alias("haiku")
    verdicts = asyncio.run(judge_batch(records, model=haiku, max_concurrency=16))
    n_err = sum(1 for v in verdicts if v.error and "unparseable" not in v.error)
    if n_err / len(verdicts) > SMOKE_JUDGE_API_ERROR_CEILING:
        raise RuntimeError(
            f"smoke gate: {n_err}/{len(verdicts)} post-retry API-error verdicts (> "
            f"{SMOKE_JUDGE_API_ERROR_CEILING:.0%}) — eval anomaly, not a parity read; retry"
        )
    own_rate = sum(1 for v in verdicts if v.agreed) / len(verdicts)
    committed = float(smoke_cell_rec["committed_own_rate"])
    drift = own_rate - committed
    passed = abs(drift) <= PARITY_TOLERANCE
    report = {
        "cell": cell_id(smoke_cell_rec["source"], smoke_cell_rec["arm"], smoke_cell_rec["step"]),
        "judge_model": haiku,
        "n_verdicts": len(verdicts),
        "n_api_errors": n_err,
        "fresh_source_own_rate": own_rate,
        "committed_own_rate": committed,
        "drift": drift,
        "tolerance": PARITY_TOLERANCE,
        "passed": passed,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    gate_path = _gate_state_path(slab_root)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gate_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------------------
# Shard worker: sequential cells on one GPU
# ---------------------------------------------------------------------------


class ShardWorker:
    def __init__(self, args: argparse.Namespace, cells: list[dict]):
        self.cells = cells
        self.gpu_id: int = args.gpu_id
        self.output_root: Path = args.output_root
        self.slab_root: Path = args.output_root / "eval_results" / "matched_install_panel"
        self.adapters_root: Path = args.output_root / "adapters_608_subceiling"
        self.runs_root: Path = args.output_root / "runs"
        self.data_root: Path = args.data_root
        self.logs_root: Path = args.logs_root
        self.hf_upload: bool = args.hf_upload

    def _eval_subprocess(self, c: dict, merged_dir: Path, out_dir: Path) -> None:
        """Fresh-subprocess vLLM eval (teardown gotcha). The eval's own
        legacy-shaped sentinel goes under runs_root, NOT logs_root, so the
        orchestrator's poller only ever sees conforming sentinels."""
        name = f"eval-{c['source']}-{c['arm']}-step{c['step']}"
        eval_sentinel = self.runs_root / "eval_sentinels" / f"{name}.json"
        eval_sentinel.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.sycophancy_implantation_411.eval_one_source",
            "--source",
            c["source"],
            "--seed",
            str(SEED),
            "--merged-model-path",
            str(merged_dir),
            "--eval-pool",
            str(self.data_root / "wrong_claims" / "eval_50.jsonl"),
            "--out-dir",
            str(out_dir),
            "--sentinel-path",
            str(eval_sentinel),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu_id)}
        log.info("[%s] spawning eval: %s", name, " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)
        if not eval_sentinel.exists():
            raise RuntimeError(f"eval subprocess exited 0 but wrote no sentinel {eval_sentinel}")

    def _assert_cell_outputs(self, out_dir: Path) -> None:
        panels = sorted(out_dir.glob("sycophancy_eval_*.json"))
        raws = sorted((out_dir / "raw_completions").glob("*.json"))
        if len(panels) != N_PANELS or len(raws) != N_PANELS:
            raise RuntimeError(
                f"{out_dir}: {len(panels)} panel JSONs / {len(raws)} raw mirrors != "
                f"{N_PANELS} each — eval incomplete; refusing to mark cell complete"
            )
        with open(panels[0]) as f:
            n = len(json.load(f)["completions"])
        if n != EXPECTED_PANEL_COMPLETIONS:
            raise RuntimeError(f"{panels[0]}: {n} completions != {EXPECTED_PANEL_COMPLETIONS}")

    def _upload_cell_tree(self, c: dict, out_dir: Path) -> str | None:
        """Per-cell raw completions + eval JSONs -> HF data repo BEFORE pod
        termination (Upload Policy; the cell tree CONTAINS raw_completions/)."""
        if not self.hf_upload:
            log.info(
                "[%s] HF upload disabled — skipping", cell_id(c["source"], c["arm"], c["step"])
            )
            return None
        raw_files = list(out_dir.rglob("raw_completions/*.json"))
        if len(raw_files) != N_PANELS:
            raise RuntimeError(
                f"[{cell_id(c['source'], c['arm'], c['step'])}] {len(raw_files)} raw-completion "
                f"files != {N_PANELS}; refusing to upload an incomplete cell tree"
            )
        rel = f"matched_install_panel/{c['arm']}/{c['source']}/seed_{SEED}/steps/step_{c['step']}"
        return _upload_or_raise(out_dir, path_in_repo=f"{HF_627_DATA_PREFIX}/{rel}")

    def run_cell(self, c: dict) -> dict:
        cid = cell_id(c["source"], c["arm"], c["step"])
        state_path = _cellstate_path(self.slab_root, c)
        if state_path.exists():
            with open(state_path) as f:
                prior = json.load(f)
            if prior.get("status") == "complete":
                log.info("[%s] cell-state already complete — skipping (idempotent)", cid)
                return prior
        t0 = time.time()
        out_dir = cell_out_dir(self.slab_root, c)
        record: dict = {
            "cell": cid,
            "source": c["source"],
            "arm": c["arm"],
            "step": c["step"],
            "role": c.get("role"),
            "committed_own_rate": c.get("committed_own_rate"),
            "seed": SEED,
            "gpu_id": self.gpu_id,
            "adapter_hub_prefix": adapter_hub_prefix(c["arm"], c["source"], c["step"]),
            "eval_out_dir": str(out_dir),
            "git_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
        }
        log.info("=" * 70)
        log.info("[%s] CELL START -> %s", cid, out_dir)
        adapter_dir = download_adapter(c, self.adapters_root)
        merged_dir = self.runs_root / "merged" / f"{c['arm']}_{c['source']}_step{c['step']}"
        from explore_persona_space.train.sft import merge_lora

        log.info("[%s] merge_lora -> %s (gpu %d)", cid, merged_dir, self.gpu_id)
        merge_lora(
            base_model_path=BASE_MODEL,
            adapter_path=str(adapter_dir),
            output_dir=str(merged_dir),
            gpu_id=self.gpu_id,
        )
        try:
            self._eval_subprocess(c, merged_dir, out_dir)
        finally:
            shutil.rmtree(merged_dir, ignore_errors=True)  # MooseFS/disk discipline
        self._assert_cell_outputs(out_dir)
        record["hub_eval_tree"] = self._upload_cell_tree(c, out_dir)
        record.update(status="complete", wall_seconds=round(time.time() - t0, 1))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(record, f, indent=2)
        _write_sentinel(
            self.logs_root,
            kind="epm:progress",
            note_obj={"event": "cell_complete", **{k: record[k] for k in ("cell", "wall_seconds")}},
            name_slug="progress",
        )
        log.info("[%s] cell complete in %.1fs", cid, record["wall_seconds"])
        return record

    def run(self) -> None:
        for c in self.cells:
            try:
                self.run_cell(c)
            except Exception:
                log.exception("[%s] cell FAILED", cell_id(c["source"], c["arm"], c["step"]))
                raise
            if (c["source"], c["arm"], int(c["step"])) == SMOKE_CELL and not _gate_state_path(
                self.slab_root
            ).exists():
                log.info("[smoke-gate] running inline Haiku parity mini-judge")
                report = run_smoke_parity_gate(self.slab_root, c)
                _write_sentinel(
                    self.logs_root,
                    kind="epm:progress",
                    note_obj={"event": "smoke_parity_gate", **report},
                    name_slug="smoke-gate",
                    gate=None if report["passed"] else "smoke_parity_gate",
                )
                if not report["passed"]:
                    raise RuntimeError(
                        f"SMOKE PARITY GATE FAIL: fresh own-rate "
                        f"{report['fresh_source_own_rate']:.3f} vs committed "
                        f"{report['committed_own_rate']:.3f} (tolerance ±{PARITY_TOLERANCE}) — "
                        f"eval-path bug; never interpret a failed-parity panel (plan §7)"
                    )
                log.info(
                    "[smoke-gate] PASS: fresh=%.3f committed=%.3f",
                    report["fresh_source_own_rate"],
                    report["committed_own_rate"],
                )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def spawn_shards(args: argparse.Namespace, cells: list[dict], gpus: list[int]) -> None:
    shards: list[list[dict]] = [[] for _ in gpus]
    for i, c in enumerate(cells):
        shards[i % len(gpus)].append(c)
    procs: list[tuple[int, subprocess.Popen]] = []
    for gpu, shard in zip(gpus, shards, strict=True):
        if not shard:
            continue
        spec = ",".join(cell_id(c["source"], c["arm"], c["step"]) for c in shard)
        cmd = [
            "uv",
            "run",
            "python",
            str(Path(__file__).resolve()),
            "--shard-worker",
            "--cells-manifest",
            str(args.cells_manifest),
            "--cells",
            spec,
            "--gpu-id",
            str(gpu),
            "--output-root",
            str(args.output_root),
            "--data-root",
            str(args.data_root),
            "--logs-root",
            str(args.logs_root),
        ]
        if not args.hf_upload:
            cmd.append("--no-hf-upload")
        shard_log = args.logs_root / f"issue-627-shard-gpu{gpu}.log"
        args.logs_root.mkdir(parents=True, exist_ok=True)
        log.info("spawning shard gpu=%d (%d cells) -> %s", gpu, len(shard), shard_log)
        # Shard env: CVD deliberately UNSET (merge_lora pins via gpu_id; eval
        # subprocesses receive CVD explicitly). env passed explicitly.
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        with open(shard_log, "a") as lf:
            procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)))
    failures = []
    for gpu, proc in procs:
        rc = proc.wait()
        # Per-cell completion echoes are worded WITHOUT the [phase=done] token
        # (reserved for the single terminal line of this main log).
        log.info("shard gpu=%d exited rc=%d", gpu, rc)
        if rc != 0:
            failures.append((gpu, rc))
    if failures:
        raise RuntimeError(f"{len(failures)} shard(s) failed: {failures} — see shard logs")


def aggregate(args: argparse.Namespace, manifest_cells: list[dict], requested: list[dict]) -> None:
    slab_root = args.output_root / "eval_results" / "matched_install_panel"
    states = {}
    for c in manifest_cells:
        p = _cellstate_path(slab_root, c)
        if p.exists():
            with open(p) as f:
                states[cell_id(c["source"], c["arm"], c["step"])] = json.load(f)
    complete = sorted(k for k, v in states.items() if v.get("status") == "complete")
    full_grid = sorted(cell_id(c["source"], c["arm"], c["step"]) for c in manifest_cells)
    gate_path = _gate_state_path(slab_root)
    gate = None
    if gate_path.exists():
        with open(gate_path) as f:
            gate = json.load(f)
    is_full = complete == full_grid and not args.dry_run
    note: dict = {
        "issue": 627,
        "phase": "matched_install_panel",
        "n_cells_complete": len(complete),
        "n_cells_grid": len(full_grid),
        "cells_complete": complete,
        "requested_cells": [cell_id(c["source"], c["arm"], c["step"]) for c in requested],
        "smoke_parity_gate": gate,
        "outputs_glob": str(slab_root / "*" / "*" / f"seed_{SEED}" / "steps" / "step_*"),
        "hf_data_prefix": f"{HF_627_DATA_PREFIX}/matched_install_panel",
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
    }
    if is_full:
        # Reproducibility card: eval-only task — the adapters are REUSED #608
        # artifacts (verified on the Hub at prefetch); no training, no WandB.
        note["reproducibility_card"] = {
            "hf_model_repo": HF_MODEL_REPO,
            "adapter_paths": {
                cell_id(c["source"], c["arm"], c["step"]): adapter_hub_prefix(
                    c["arm"], c["source"], c["step"]
                )
                for c in manifest_cells
            },
            "adapters_reused_from": "issue 608 sub-ceiling-install (no new training)",
            "wandb_run_names": None,
            "wandb_note": "eval-only task: no training was run, no WandB runs exist",
            "base_model": BASE_MODEL,
            "seed": SEED,
            "eval_recipe": "24-persona panel x 50 claims x 10 rollouts, vLLM temp 1.0, "
            "max_new_tokens 512, merge-then-generate (Source: #608)",
        }
        _write_sentinel(args.logs_root, kind="epm:results", note_obj=note, name_slug="epm_results")
    else:
        _write_sentinel(args.logs_root, kind="epm:progress", note_obj=note, name_slug="progress")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #627 Phase 1 — matched-install bystander panel (pod driver).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cells-manifest", type=Path, required=True)
    parser.add_argument(
        "--cells",
        default="all",
        help="'all' | integer prefix count (1 = registered smoke cell) | "
        "comma list of <source>:<arm>:<step>",
    )
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/issue_627"))
    parser.add_argument("--data-root", type=Path, default=Path("data/issue_627"))
    parser.add_argument("--logs-root", type=Path, default=Path("/workspace/logs"))
    parser.add_argument(
        "--gpus", default="0", help="Comma-separated physical GPU ids to shard over."
    )
    parser.add_argument("--dry-run", action="store_true", help="CPU-only walk; no GPU phases.")
    parser.add_argument("--hf-upload", dest="hf_upload", action="store_true", default=True)
    parser.add_argument("--no-hf-upload", dest="hf_upload", action="store_false")
    parser.add_argument("--shard-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    manifest_cells = load_cells_manifest(args.cells_manifest)
    requested = resolve_cells(manifest_cells, args.cells)

    if args.shard_worker:
        # Shard logs are per-shard files; no [phase=...] tokens here (the main
        # log owns the phase surface).
        ShardWorker(args, requested).run()
        return 0

    log.info("[phase=p0_manifest] %d/%d cells requested", len(requested), len(manifest_cells))
    log.info("[phase=p1_prefetch] probe pin + Hub adapter checks")
    prefetch_report = prefetch(requested, args.data_root)
    log.info("[phase=p1_prefetch] OK: %s", json.dumps(prefetch_report)[:200])

    if args.dry_run:
        log.info("[phase=p2_panel] DRY RUN — GPU phases skipped; sentinel surface exercised")
        aggregate(args, manifest_cells, requested)
        log.info("[phase=done]")
        return 0

    log.info("[phase=p2_panel] sharding %d cells over gpus=%s", len(requested), args.gpus)
    gpus = [int(g) for g in str(args.gpus).split(",") if g != ""]
    if not gpus:
        raise ValueError(f"--gpus parsed to empty list from {args.gpus!r}")
    spawn_shards(args, requested, gpus)

    log.info("[phase=p3_aggregate]")
    aggregate(args, manifest_cells, requested)
    log.info("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
