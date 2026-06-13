#!/usr/bin/env python3
"""Issue #545 — top-level dispatcher (behavior-generalization testbed).

ONE dispatcher; smoke IS sweep with one cell (plan section 4.0 architectural
parity): ``--smoke`` = ``--rows marker --seeds 0 --arms primary`` + tiny caps
through the IDENTICAL code path (same subprocess shape, env injection,
logging surface, sentinel write, teardown).

Phases::

    p0        VM, CPU/API: #503 panels, corpora, batteries, demo sets,
              pre-registration freeze. No GPU.
    p1        pod: base panel (P1.0) + anchors/nulls/arms + warmth gate +
              K1 harness gate verdict.
    p2        pod: remaining rows x seeds + contrastive arms (+ warmth iff
              the P1 gate cleared). REFUSES to run while K1 = FAIL.
    p3        pod: predictor extraction (1 GPU) -> assemble -> score.
    assemble  CPU: L matrix + metadata from cell JSONs.
    score     CPU: the pre-registered predictor race.

Pod-side contract (poll_pipeline.py): ``[phase=<name>]`` log lines with a
terminal ``[phase=done]``; end-of-run sentinel at
``/workspace/logs/issue-545-epm_results-<epoch>.json`` carrying
``sentinel_schema_version`` / ``kind`` / ``version`` / ``note``.

Every subprocess is launched with explicit ``env={**os.environ}``;
``load_dotenv`` runs at module top (uv run does NOT auto-load .env).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

if Path("/workspace").exists():  # pod-only cache redirect; VM keeps its default
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("issue545_sweep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ISSUE = 545
SENTINEL_SCHEMA_VERSION = 1
ROBUSTNESS_ROWS = (
    "bad_medical",
    "marker",
    "insecure_code",
    "wrong_claim_agreement",
    "refuse_medical",
    "taught_fact",
    "answer_in_lists",
    "benign_format",
    "business_skills",
    "educational_insecure",
)


def _run(cmd: list[str], *, label: str, extra_env: dict[str, str] | None = None) -> None:
    """Fail-loud subprocess with explicit env passthrough.

    ``extra_env`` entries override the inherited environment — used to pin
    ``CUDA_VISIBLE_DEVICES`` at SPAWN time so the child's GPU restriction is
    in force before any import can initialize the CUDA driver (round-10 OOM
    root cause: ``import peft`` freezes the driver's visible-device list, so
    an in-process env set AFTER it is silently ignored and every train lands
    on physical GPU 0).
    """
    logger.info("[cmd:%s] %s", label, shlex.join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env={**os.environ, **(extra_env or {})})


def _gpu_env(gpu: int) -> dict[str, str]:
    """Spawn-time CUDA_VISIBLE_DEVICES pin for a single-GPU lease."""
    return {"CUDA_VISIBLE_DEVICES": str(gpu)}


# Benign teardown lag (a just-finished cell's vLLM engine worker still releasing
# memory, or driver accounting lag) can leave the leased GPU above the busy
# threshold for a short while on a HEALTHY run. The guard waits boundedly for it
# to clear before declaring a lease conflict.
GPU_GUARD_WAIT_TIMEOUT_S = 120.0
GPU_GUARD_POLL_INTERVAL_S = 5.0


def _query_gpu_used_mib(gpu: int) -> int | None:
    """Best-effort nvidia-smi used-memory probe for one physical GPU (MiB).

    Returns ``None`` when the probe is unavailable (no nvidia-smi on the VM
    CPU smoke, parse failure) — callers warn-and-skip rather than fail.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        ).stdout.strip()
        return int(out.splitlines()[0])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError) as e:
        logger.warning("[gpu-guard] nvidia-smi probe unavailable for GPU %d (%s) — skipped", gpu, e)
        return None


def _assert_gpu_memory_free(
    gpu: int,
    *,
    label: str,
    threshold_mib: int = 2048,
    timeout_s: float = GPU_GUARD_WAIT_TIMEOUT_S,
    poll_interval_s: float = GPU_GUARD_POLL_INTERVAL_S,
) -> None:
    """Fail loud BEFORE launching a train if the leased physical GPU stays busy.

    Converts a silent multi-train pile-up (the round-10 OOM at train step 60)
    into a diagnosable error naming the conflicting usage — but tolerates
    benign teardown lag (vLLM's worker subprocesses release GPU memory slowly
    after engine shutdown; documented codebase gotcha) by polling the leased
    GPU every ``poll_interval_s`` for up to ``timeout_s`` before raising. Best
    effort: skipped with a warning when nvidia-smi is unavailable (CPU smoke
    on the VM).
    """
    used_mib = _query_gpu_used_mib(gpu)
    if used_mib is None or used_mib < threshold_mib:
        return
    logger.info(
        "[gpu-guard] GPU %d has %d MiB in use; waiting up to %.0fs for teardown",
        gpu,
        used_mib,
        timeout_s,
    )
    waited = 0.0
    while waited < timeout_s:
        time.sleep(poll_interval_s)
        waited += poll_interval_s
        used_mib = _query_gpu_used_mib(gpu)
        if used_mib is None:
            return
        if used_mib < threshold_mib:
            logger.info("[gpu-guard] GPU %d cleared after %.0fs", gpu, waited)
            return
    raise RuntimeError(
        f"GPU lease conflict: physical GPU {gpu} already has {used_mib} MiB in use "
        f"(>= {threshold_mib} MiB threshold) at launch of {label} after waiting "
        f"{timeout_s:.0f}s. Another process is occupying the leased device — "
        "refusing to stack a second training on it."
    )


def _with_gpu_lease(gpu_slots: Queue, fn):
    """Acquire a GPU lease for the duration of ``fn(gpu)``; always release.

    Module-level (not a closure) so the lease pattern is unit-testable:
    concurrent callers MUST receive disjoint physical GPU ids.
    """
    gpu = gpu_slots.get()
    try:
        return fn(gpu)
    finally:
        gpu_slots.put(gpu)


def _adapters_root() -> Path:
    """Trained-cell artifact root (delegates to the package so smoke-output
    isolation applies identically in the dispatcher and every subprocess)."""
    from explore_persona_space.experiments.behavior_testbed_545 import adapters_root

    return adapters_root()


def _activate_smoke_isolation() -> None:
    """Route ALL ``--smoke`` outputs to an isolated ``smoke/`` root (round 19).

    Sets I545_SMOKE_OUTPUT=1 in THIS process's environment before any package
    path resolves; every package path helper (``output_root`` -> manifests,
    cells, batteries, gates; ``adapters_root``) appends ``smoke/`` while
    active, and every subprocess inherits the flag via ``env={**os.environ}``.
    The code path stays IDENTICAL (same dispatcher, same functions — smoke IS
    sweep with one cell): only the root differs, so production resume guards
    (manifest ``done_cells``, base-panel completeness) are physically unable
    to see smoke artifacts. Round-18 incident: the pod smoke's manifest entry
    + 2-column base_panel satisfied production's resume guards and K1 FAILed
    on the 4-step smoke adapter.

    Frozen P0 INPUTS stay readable: battery loads fall back read-only to the
    production batteries dir (``eval_battery.load_battery``) and corpus reads
    fall back to the production corpora dir (``corpus_read_path``) while
    corpus WRITES land under the smoke corpora root (round 20 — a smoke prep
    must never overwrite a production corpus). ``bulk_upload_phase`` is
    smoke-gated outright: a smoke run is physically unable to write any
    production HF path.
    """
    from explore_persona_space.experiments.behavior_testbed_545 import (
        SMOKE_OUTPUT_ENV,
        adapters_root,
        output_root,
    )

    os.environ[SMOKE_OUTPUT_ENV] = "1"
    logger.info(
        "[smoke] isolated output roots: results=%s adapters=%s", output_root(), adapters_root()
    )


def _activate_v2_namespace() -> None:
    """Route ALL outputs to the ``onpolicy_v2/`` namespace (follow-up
    onpolicy-testbed-v2, plan divergence 5).

    Sets I545_V2_OUTPUT=1 in THIS process's environment before any package
    path resolves; every package path helper appends ``onpolicy_v2/``
    (before any ``smoke/`` segment), the row registry swaps to ROWS_V2
    (``rows.active_rows``), HF upload prefixes gain ``_v2``
    (``issue545_rows_v2`` / ``issue545_behavior_testbed_v2``), and every
    subprocess inherits the flag via ``env={**os.environ}``. v1 artifacts
    are frozen READ-ONLY inputs (batteries, base panel, question pools) —
    physically never written.
    """
    from explore_persona_space.experiments.behavior_testbed_545 import (
        V2_OUTPUT_ENV,
        adapters_root,
        output_root,
    )

    os.environ[V2_OUTPUT_ENV] = "1"
    logger.info(
        "[v2] namespaced output roots: results=%s adapters=%s", output_root(), adapters_root()
    )


def _v2_active() -> bool:
    from explore_persona_space.experiments.behavior_testbed_545 import v2_output_active

    return v2_output_active()


# ---------------------------------------------------------------------------
# P0 (CPU/API)
# ---------------------------------------------------------------------------


def phase_p0(args) -> None:
    print("[phase=p0_panels]", flush=True)
    needed_panels = [
        "betley_main_8",
        "turner_medical_heldout",
        "bigcode_codereq_heldout",
        "broad_syco_wrong_claims_heldout",
        "advbench_harmful_520",
    ]
    from explore_persona_space.experiments.issue503.eval_panels import panel_path

    smoke_mode = args.smoke or args.corpora_n is not None
    for panel in needed_panels:
        if panel_path(panel, PROJECT_ROOT).exists():
            continue
        try:
            _run(
                ["uv", "run", "python", "scripts/issue503_prep_eval_panels.py", "--panel", panel],
                label=f"prep-panel-{panel}",
            )
        except subprocess.CalledProcessError:
            if not smoke_mode:
                raise  # production P0 fails loud (TURNER_EDS_PASSWORD etc.)
            logger.warning("[smoke] panel %s prep unavailable locally — placeholder path", panel)

    smoke_n = args.corpora_n
    if args.build_corpora:
        print("[phase=p0_corpora]", flush=True)
        from explore_persona_space.experiments.behavior_testbed_545.corpora import (
            build_all_batteries,
            build_all_corpora,
        )

        corpora = build_all_corpora(smoke_n=smoke_n)
        logger.info("Corpora built: %s", {k: Path(v).name for k, v in corpora.items() if v})
        batteries = build_all_batteries(smoke_n=smoke_n)
        logger.info("Batteries built: %d files", len(batteries))
    if args.build_corpora:
        # B10 anchor calibration (plan gate 2 prerequisite): judge the #496
        # warm/cold rewrite pairs with the verbatim #515 rubric so the warmth
        # gate threshold is anchor-grounded BEFORE any training.
        print("[phase=p0_warmth_anchors]", flush=True)
        from explore_persona_space.experiments.behavior_testbed_545.gates import (
            calibrate_warmth_anchors,
        )

        try:
            calibrate_warmth_anchors(smoke_n=smoke_n)
        except Exception as e:
            if not smoke_mode:
                raise  # production P0 fails loud (API key / HF reachability)
            logger.warning("[smoke] warmth anchor calibration unavailable locally: %s", e)
    if args.preregister:
        print("[phase=p0_preregister]", flush=True)
        from explore_persona_space.experiments.behavior_testbed_545.preregister import (
            write_preregistration,
        )

        path = write_preregistration(allow_placeholders=smoke_mode)
        logger.info("Pre-registration frozen at %s", path)


# ---------------------------------------------------------------------------
# Training + eval phases (pod)
# ---------------------------------------------------------------------------


def _eval_cell_cmd(
    *,
    row: str | None,
    arm: str,
    seed: int,
    adapter: str | None,
    gpu: int,
    contexts: list[str],
    only: str | None,
    base_panel: bool = False,
    diagonal_only: bool = False,
    max_probes: int | None,
    columns: list[str] | None = None,
) -> list[str]:
    cmd = ["uv", "run", "python", "scripts/issue545_eval_cell.py", "--gpu-id", str(gpu)]
    if base_panel:
        cmd.append("--base-panel")
    else:
        cmd += ["--row", row, "--arm", arm, "--seed", str(seed), "--adapter-path", adapter]
    cmd += ["--contexts", *contexts]
    if columns:
        cmd += ["--columns", *columns]
    if diagonal_only:
        cmd.append("--diagonal-only")
    if max_probes:
        cmd += ["--max-probes", str(max_probes)]
    if only:
        cmd += ["--only", only]
    return cmd


def _v2_base_floor(row) -> float:
    """The v1 base-panel floor for a row's dose scalar (corrected normalization).

    Reads the GIT-COMMITTED v1 ``cells/base_panel/<diag>__default.json``
    summary via the row's scalar key (``diagonal_scalar_key`` override or
    the column PRIMARY_SCALAR). Fails loud when the base read is missing —
    the corrected normalization is the POINT of the v2 dose fix (plan
    divergence 4); silently substituting 0.0 would re-create the v1 warmth
    defect.
    """
    from explore_persona_space.experiments.behavior_testbed_545 import v1_committed_root
    from explore_persona_space.experiments.behavior_testbed_545.assemble_matrix import (
        PRIMARY_SCALAR,
    )

    p = v1_committed_root() / "cells" / "base_panel" / f"{row.diagonal_column}__default.json"
    if not p.exists():
        raise FileNotFoundError(
            f"v2 dose base floor needs the v1 base panel read: {p} (committed v1 artifact)"
        )
    summary = json.loads(p.read_text())["summary"]
    key = row.diagonal_scalar_key or PRIMARY_SCALAR[row.diagonal_column]
    val = summary.get(key)
    if val is None:
        raise RuntimeError(f"v1 base panel {p.name} has no {key!r} (keys: {sorted(summary)})")
    return float(val)


def _v2_target_strength(row, arm: str, seed: int) -> float | None:
    """The re-selected v1 corrected realized strength this v2 cell pairs to.

    From the committed ``onpolicy_v2/v1_reselect.json`` (P0-v2 deliverable).
    Lookup: the SAME (row, arm, seed) cell; the bridge arm pairs to the
    compliment row's PRIMARY cell (its v1 counterpart); a missing entry
    falls back to the row's primary cell, then None (selection degrades to
    first-in-band, recorded — never a crash: a v2-only arm without a v1
    counterpart is still trainable, just not confirmatory-pairable).
    """
    from explore_persona_space.experiments.behavior_testbed_545 import v1_committed_root

    p = v1_committed_root() / "onpolicy_v2" / "v1_reselect.json"
    if not p.exists():
        logger.warning("[dose_select] v1_reselect.json missing at %s — no pairing target", p)
        return None
    cells = json.loads(p.read_text())["cells"]
    lookup_arm = "primary" if arm == "bridge" else arm
    for cell_id in (
        row.cell_id(lookup_arm, seed),
        row.cell_id("primary", seed),
    ):
        entry = cells.get(cell_id)
        if entry and entry.get("corrected_realized_strength") is not None:
            return float(entry["corrected_realized_strength"])
    logger.warning(
        "[dose_select] no v1 re-selection entry for %s — no pairing target",
        row.cell_id(arm, seed),
    )
    return None


def _dose_select_checkpoint(
    row, arm: str, seed: int, adapter_dir: Path, gpu: int, args, extra_env: dict | None = None
) -> Path:
    """Pick the first checkpoint whose diagonal battery lands in band.

    Band read against the max over checkpoints (P2-calibrated dose-to-target;
    plan section 4.5), bands from the pre-registered ``THRESHOLDS``. A full
    default-band miss with a MONOTONE dose-response retries with the
    pre-registered 50-95% recalibration allowance (plan section 7 band-miss
    routing; ``gates.select_dose_checkpoint``) — only a non-monotone /
    broken-harness miss falls back to the final state flagged out-of-band
    (the K1-stop signature). Writes an explicit ``dose_select.json`` record
    (ceiling, per-checkpoint scalars, ``in_band``, ``band_recalibrated``,
    the band actually used) into the cell dir — the K1 gate and assemble's
    ``implant_failed`` read CONSUME this record (round-1/2 blocker fixes).
    Per-checkpoint reads are archived under the ``dose/`` SUBDIR so they can
    never collide with assemble's ``*__*.json`` column glob.
    """
    from explore_persona_space.experiments.behavior_testbed_545 import cells_dir
    from explore_persona_space.experiments.behavior_testbed_545.assemble_matrix import (
        PRIMARY_SCALAR,
        _scalar,
    )

    def _diag_scalar(d: dict) -> float | None:
        """The row's dose scalar from a diagonal column JSON. v2 rows may pin
        a per-row summary key (casual_register -> casual_register_rate, the
        v1 shared-scalar defect fix); default = the column PRIMARY_SCALAR."""
        if row.diagonal_scalar_key:
            v = d["summary"].get(row.diagonal_scalar_key)
            return float(v) if v is not None else None
        return _scalar(d["column"], d["summary"]) if d["column"] in PRIMARY_SCALAR else None

    cell_dir = cells_dir() / row.cell_id(arm, seed)
    checkpoints = sorted(
        (d for d in adapter_dir.glob("checkpoint-*") if d.is_dir()),
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if not checkpoints:
        return adapter_dir
    dose_dir = cell_dir / "dose"
    scalars: list[tuple[Path, float | None]] = []
    for ckpt in checkpoints:
        scratch_cell = f"{row.cell_id(arm, seed)}__ckpt{ckpt.name.split('-')[-1]}"
        for only in ("gen", "judge"):
            _run(
                _eval_cell_cmd(
                    row=row.row_id,
                    arm=arm,
                    seed=seed,
                    adapter=str(ckpt),
                    gpu=gpu,
                    contexts=["default"],
                    only=only,
                    diagonal_only=True,
                    max_probes=args.max_probes,
                ),
                label=f"dose-{scratch_cell}-{only}",
                extra_env=extra_env,
            )
            # the eval cell writes under the canonical cell dir; move per ckpt
        diag_path = cell_dir / f"{row.diagonal_column}__default.json"
        val = None
        if diag_path.exists():
            d = json.loads(diag_path.read_text())
            val = _diag_scalar(d)
            # archive this checkpoint's reads OUT of the column-glob namespace
            dose_dir.mkdir(parents=True, exist_ok=True)
            diag_path.rename(dose_dir / f"{row.diagonal_column}__{ckpt.name}.json")
            gen_path = cell_dir / f"completions__{row.diagonal_column}__default.json"
            if gen_path.exists():
                gen_path.rename(dose_dir / f"completions__{ckpt.name}.json")
        scalars.append((ckpt, val))
        logger.info("[phase=dose_select] %s %s -> %s", row.row_id, ckpt.name, val)

    from explore_persona_space.experiments.behavior_testbed_545.gates import (
        select_dose_checkpoint,
        select_dose_checkpoint_v2,
    )
    from explore_persona_space.experiments.behavior_testbed_545.preregister import THRESHOLDS

    if _v2_active():
        # v2 (plan v3 section 4.2, item-3 resolution): corrected
        # base-floor normalization defines ELIGIBILITY; among eligible
        # checkpoints pick the one NEAREST the re-selected v1 corrected
        # realized strength (v1_reselect.json — committed at P0-v2).
        base_val = _v2_base_floor(row)
        v1_target = _v2_target_strength(row, arm, seed)
        sel = select_dose_checkpoint_v2(
            [(c.name, v) for c, v in scalars],
            base=base_val,
            v1_target_strength=v1_target,
            default_band=tuple(THRESHOLDS["dose_band_default"]),
            recalibration_allowance=tuple(THRESHOLDS["dose_band_recalibration_allowance"]),
        )
    else:
        sel = select_dose_checkpoint(
            [(c.name, v) for c, v in scalars],
            default_band=tuple(THRESHOLDS["dose_band_default"]),
            recalibration_allowance=tuple(THRESHOLDS["dose_band_recalibration_allowance"]),
        )
    if sel["selected"] is not None:
        selected = next(c for c, _ in scalars if c.name == sel["selected"])
        if sel["band_recalibrated"]:
            logger.warning(
                "[phase=dose_select] %s: default band %s missed everywhere but the dose-response "
                "is MONOTONE — recalibrated to the pre-registered allowance %s, selected %s "
                "(plan section 7 band-miss routing)",
                row.row_id,
                THRESHOLDS["dose_band_default"],
                THRESHOLDS["dose_band_recalibration_allowance"],
                selected.name,
            )
    else:
        selected = checkpoints[-1]
        if sel["ceiling"] is None:
            logger.warning("[phase=dose_select] no diagonal reads for %s — using final", row.row_id)
        else:
            logger.warning(
                "[phase=dose_select] %s: no checkpoint in the default band NOR (monotone=%s) the "
                "50-95%% recalibration allowance (vals=%s) — using final checkpoint flagged "
                "out-of-band (K1-stop signature)",
                row.row_id,
                sel["monotone"],
                [round(v, 3) if v is not None else None for _, v in scalars],
            )
    cell_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "row": row.row_id,
        "arm": arm,
        "seed": seed,
        "ceiling": sel["ceiling"],
        "band": sel["band"],  # the band actually in force (allowance iff recalibrated)
        "band_default": list(THRESHOLDS["dose_band_default"]),
        "band_recalibrated": sel["band_recalibrated"],
        "monotone": sel["monotone"],
        "in_band": sel["in_band"],
        "selected_checkpoint": selected.name,
        "scalars": {c.name: v for c, v in scalars},
    }
    if _v2_active():
        # v2 extras: corrected normalization + nearest-strength pairing
        # record (consumed by K1-v2, assemble, and the VM-side comparison).
        record.update(
            {
                "base": sel.get("base"),
                "strengths": sel.get("strengths"),
                "scalar_key": row.diagonal_scalar_key,
                "v1_target_strength": sel.get("v1_target_strength"),
                "achieved_strength": sel.get("achieved_strength"),
                "delta_strength": sel.get("delta_strength"),
                "confirmatory_eligible": sel.get("confirmatory_eligible"),
                "confirmatory_max_delta": sel.get("confirmatory_max_delta"),
            }
        )
    (cell_dir / "dose_select.json").write_text(json.dumps(record, indent=1))
    return selected


def _run_one_cell(row, arm: str, seed: int, gpu: int, args) -> dict:
    """prep -> train -> dose-select -> full eval (3 subprocess phases)."""
    from explore_persona_space.experiments.behavior_testbed_545 import cells_dir
    from explore_persona_space.experiments.behavior_testbed_545.columns import (
        ROBUSTNESS_COLUMNS,
        ROBUSTNESS_CONTEXTS,
    )

    cell = row.cell_id(arm, seed)
    result = {"cell": cell, "gpu": gpu}
    needs_prep = (row.gpu_prep is not None) or arm in ("cn", "mix50")
    # Spawn-time CVD pin: every single-GPU subprocess of this cell gets
    # CUDA_VISIBLE_DEVICES=<lease> in its environment BEFORE python starts,
    # so no import-time CUDA-driver init (peft, round-10 OOM) can defeat the
    # restriction. fullft is the one multi-GPU arm (ZeRO-3 over all GPUs) —
    # it must NOT be restricted to its nominal lease.
    gpu_env = None if arm == "fullft" else _gpu_env(gpu)
    base_cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue545_train_cell.py",
        "--row",
        row.row_id,
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--gpu-id",
        str(gpu),
    ]
    smoke_flag = ["--smoke"] if args.smoke else []
    if not args.skip_train:
        if gpu_env is not None:
            _assert_gpu_memory_free(gpu, label=f"train-{cell}")
        if needs_prep:
            print(f"[phase=prep_{cell}]", flush=True)
            _run([*base_cmd, "--prep-only", *smoke_flag], label=f"prep-{cell}", extra_env=gpu_env)
        print(f"[phase=train_{cell}]", flush=True)
        _run([*base_cmd, *smoke_flag], label=f"train-{cell}", extra_env=gpu_env)
    adapter_dir = _adapters_root() / cell
    # Band-stop rows persist their stop record next to the adapter; copy it
    # into the cell dir so the K1 gate (and re-runs after the post-upload
    # adapter cleanup) read it from the durable eval tree.
    band_record = adapter_dir / "band_stop_result.json"
    if band_record.exists():
        cells_dir().joinpath(cell).mkdir(parents=True, exist_ok=True)
        shutil.copy2(band_record, cells_dir() / cell / "band_stop_result.json")
    if not args.skip_eval:
        adapter = adapter_dir
        band_stop_row = bool(row.train_lora_overrides.get("marker_band_stop"))
        if (
            adapter_dir.exists()
            and not args.smoke
            and row.recipe_kind != "reuse_adapter"
            and not band_stop_row
        ):
            # Band-stop rows SKIP dose-select: the band-stop already selected
            # the stopping point, and the final adapter state at the dir root
            # (saved on stop) is the band-stopped artifact — checkpoint-* are
            # periodic pre-band saves (round-1 minor #9).
            adapter = _dose_select_checkpoint(
                row, arm, seed, adapter_dir, gpu, args, extra_env=gpu_env
            )
        result["selected_checkpoint"] = str(adapter)
        print(f"[phase=eval_{cell}]", flush=True)
        phases = ["gen", "hf"] + ([] if args.skip_judges else ["judge"])
        # Pass 1: default context, FULL battery.
        for only in phases:
            _run(
                _eval_cell_cmd(
                    row=row.row_id,
                    arm=arm,
                    seed=seed,
                    adapter=str(adapter),
                    gpu=gpu,
                    contexts=["default"],
                    only=only,
                    max_probes=args.max_probes,
                ),
                label=f"eval-{cell}-{only}",
                extra_env=gpu_env,
            )
        # Pass 2: robustness + template-token contexts on the 4-COLUMN subset
        # only (plan section 4.3; round-1 major #5 — full battery here would
        # blow the robustness budget ~10x).
        if not args.smoke and arm == "primary" and row.row_id in ROBUSTNESS_ROWS:
            for only in phases:
                _run(
                    _eval_cell_cmd(
                        row=row.row_id,
                        arm=arm,
                        seed=seed,
                        adapter=str(adapter),
                        gpu=gpu,
                        contexts=list(ROBUSTNESS_CONTEXTS),
                        columns=list(ROBUSTNESS_COLUMNS),
                        only=only,
                        max_probes=args.max_probes,
                    ),
                    label=f"eval-{cell}-robustness-{only}",
                    extra_env=gpu_env,
                )
    return result


def _judge_sensitivity_check(args) -> None:
    """Plan procedural gate 3: re-judge the bad-medical bookend cell's
    broad-EM at 100/q and compare to the 50/q production read."""
    from explore_persona_space.experiments.behavior_testbed_545 import output_root
    from explore_persona_space.experiments.behavior_testbed_545.gates import (
        write_judge_sensitivity,
    )

    bookend_cell = "bad_medical_primary_seed0"
    manifest_path = output_root() / "manifest_p1.json"
    entry = next(
        (
            m
            for m in (json.loads(manifest_path.read_text()) if manifest_path.exists() else [])
            if m.get("cell") == bookend_cell
        ),
        None,
    )
    if entry is None or "selected_checkpoint" not in entry:
        raise RuntimeError(
            f"Judge-sensitivity check needs the {bookend_cell} manifest entry with a "
            "selected_checkpoint — did P1 run the bad-medical row?"
        )
    adapter = entry["selected_checkpoint"]
    print("[phase=judge_sensitivity]", flush=True)
    for only in ("gen", "judge"):
        _run(
            _eval_cell_cmd(
                row="bad_medical",
                arm="primary",
                seed=0,
                adapter=adapter,
                gpu=0,
                contexts=["default"],
                columns=["broad_em_n100"],
                only=only,
                max_probes=args.max_probes,
            ),
            label=f"judge-sensitivity-{only}",
        )
    result = write_judge_sensitivity(bookend_cell=bookend_cell)
    if not result["locked"]:
        # Plan gate 3's escalation ("100/q on cells near decision boundaries")
        # has no automated consumer — make the verdict impossible to miss in
        # the run log AND the sentinel note (round-2 minor #6).
        logger.warning(
            "[phase=judge_sensitivity] NOT LOCKED (delta=%.2fpp >= %.1fpp): plan gate 3 "
            "escalation — re-judge cells near decision boundaries at 100/q before "
            "interpreting P2 rates",
            result["delta_pp"],
            result["max_delta_pp"],
        )


def _base_panel_todo(args) -> list[tuple[list[str], list[str] | None, list[str]]]:
    """Per base-panel eval pass: ``(contexts, column subset, missing files)``.

    Replaces the round-18 bare ``base_panel.exists()`` guard with a per-file
    COMPLETENESS check (the smoke's 2-column panel satisfied ``exists()`` and
    production skipped the full panel forever). Required-file sets mirror the
    eval-cell driver's writers (``columns.base_panel_expected_files``); only
    passes with at least one missing required file re-run, and the driver's
    per-(column, context) idempotence keeps every existing file in place — a
    partial base_panel can never read as complete again.
    """
    from explore_persona_space.experiments.behavior_testbed_545 import cells_dir
    from explore_persona_space.experiments.behavior_testbed_545.columns import (
        ROBUSTNESS_COLUMNS,
        ROBUSTNESS_CONTEXTS,
        base_panel_expected_files,
    )

    base_dir = cells_dir() / "base_panel"
    passes: list[tuple[list[str], list[str] | None]] = [
        (["default"], ["marker", "capability"] if args.smoke else None)
    ]
    if not args.smoke:
        passes.append((list(ROBUSTNESS_CONTEXTS), list(ROBUSTNESS_COLUMNS)))
    todo: list[tuple[list[str], list[str] | None, list[str]]] = []
    for contexts, columns in passes:
        expected = base_panel_expected_files(contexts, columns, include_judged=not args.skip_judges)
        missing = sorted(f for f in expected if not (base_dir / f).exists())
        if missing:
            todo.append((contexts, columns, missing))
    return todo


def _v2_elicitation_prestep(args, phase: str, cells: list) -> list:
    """Run per-row elicitation subprocesses, then drop sub-quota rows.

    One ``issue545_train_cell.py --prep-only`` subprocess per UNIQUE
    rebuilt row (gpu_prep == "elicit"), 4-way through the GPU lease queue
    (vLLM + judge filters). Idempotent: rows whose ``pool_meta.json``
    already exists are skipped. Quota verdicts then prune the cell list;
    drops are recorded to ``row_drops.json`` (the H3-v2 designed signal).
    """
    from explore_persona_space.experiments.behavior_testbed_545 import output_root
    from explore_persona_space.experiments.behavior_testbed_545.elicit_v2 import row_quota_met

    elicit_rows = sorted({row.row_id for row, _a, _s in cells if row.gpu_prep == "elicit"})
    todo = [r for r in elicit_rows if row_quota_met(r) is None]
    if todo:
        print(f"[phase=elicitation_{phase}]", flush=True)
        gpu_slots: Queue[int] = Queue()
        for g in range(args.parallel):
            gpu_slots.put(g)
        smoke_flag = ["--smoke"] if args.smoke else []

        def _elicit(row_id: str):
            def _go(gpu: int):
                _assert_gpu_memory_free(gpu, label=f"elicit-{row_id}")
                _run(
                    [
                        "uv",
                        "run",
                        "python",
                        "scripts/issue545_train_cell.py",
                        "--row",
                        row_id,
                        "--arm",
                        "primary",
                        "--gpu-id",
                        str(gpu),
                        "--prep-only",
                        *smoke_flag,
                    ],
                    label=f"elicit-{row_id}",
                    extra_env=_gpu_env(gpu),
                )

            return _with_gpu_lease(gpu_slots, _go)

        failures: list[tuple[str, BaseException]] = []
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futs = {pool.submit(_elicit, r): r for r in todo}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except BaseException as e:
                    logger.error("elicitation %s FAILED: %s", futs[fut], e)
                    failures.append((futs[fut], e))
        if failures:
            raise RuntimeError(
                f"{len(failures)} elicitation row(s) failed: {[r for r, _ in failures]} — "
                f"first error: {failures[0][1]!r}"
            ) from failures[0][1]

    dropped = [r for r in elicit_rows if row_quota_met(r) is False]
    if dropped:
        drops_path = output_root() / "row_drops.json"
        existing = json.loads(drops_path.read_text()) if drops_path.exists() else {"rows": []}
        existing["rows"] = sorted(set(existing["rows"]) | set(dropped))
        drops_path.write_text(json.dumps(existing, indent=1))
        logger.warning(
            "[phase=elicitation_%s] quota DROPS (designed signal, recorded): %s", phase, dropped
        )
    return [(row, arm, seed) for row, arm, seed in cells if row.row_id not in dropped]


def phase_train_eval(args, phase: str) -> None:  # noqa: C901 — phase dispatcher, intentionally flat
    from explore_persona_space.experiments.behavior_testbed_545 import output_root
    from explore_persona_space.experiments.behavior_testbed_545.gates import (
        require_k1_pass,
        require_k1v2_pass,
        warmth_gate_passed,
    )
    from explore_persona_space.experiments.behavior_testbed_545.rows import (
        active_rows,
        enumerate_cells,
    )

    rows_registry = active_rows()
    if phase == "p2":
        # FAIL-CLOSED (round-1 Codex critical): only a literal pass=true
        # admits P2 — false AND null/missing both refuse. v2 mode gates on
        # the K1-v2 verdict (yield + corrected-band entry + integrity).
        if _v2_active():
            require_k1v2_pass()
        else:
            require_k1_pass()

    rows_filter = args.rows
    if rows_filter is not None:
        # Unknown row ids must fail LOUD: enumerate_cells would silently
        # drop them (the plan-doc `warmth_gate` vs registry `warmth` trap,
        # round-1 minor #12).
        unknown = [r for r in rows_filter if r not in rows_registry]
        if unknown:
            raise SystemExit(f"Unknown row id(s) {unknown}. Valid: {sorted(rows_registry)}")
    if rows_filter is None:
        rows_filter = [r.row_id for r in rows_registry.values() if r.phase == phase]
        if phase == "p2" and not _v2_active() and warmth_gate_passed():
            # B10 P2 inclusion is CONDITIONAL on the P1 dose-response gate
            # (plan section 4.1): full-battery warmth eval joins P2 only on
            # a recorded gate PASS. (v1-only — warmth is out of v2 scope.)
            rows_filter.append("warmth")
    if phase == "p2" and not _v2_active() and "warmth" in rows_filter and not warmth_gate_passed():
        raise RuntimeError(
            "warmth requested in P2 but warmth_gate/gate_result.json does not record "
            "pass=true (plan gate 2) — drop the row or re-run the P1 gate."
        )
    cells = enumerate_cells(rows=rows_filter, seeds=args.seeds, arms=args.arms)
    if not cells:
        raise RuntimeError(
            f"No cells matched rows={rows_filter} seeds={args.seeds} arms={args.arms}"
        )
    logger.info("[phase=%s] %d cells", phase, len(cells))

    if _v2_active():
        # v2 elicitation pre-step (plan v3 section 4.4): pools are
        # seed-invariant, so elicitation runs ONCE per ROW before any cell
        # trains; rows that miss the 160/200 quota DROP here (the H3-v2
        # designed signal — never trained short, never padded) and their
        # cells leave the schedule. The per-cell prep stays idempotent
        # (skips a completed elicitation; builds cn/bridge corpora).
        cells = _v2_elicitation_prestep(args, phase, cells)
        if not cells:
            raise RuntimeError(
                f"All {phase} rows dropped at the elicitation quota — see row_drops.json"
            )

    # P1.0 base panel first (headroom denominator; gates column inclusion).
    # Default context = full battery; robustness + template-token contexts =
    # the 4-column subset only (mirrors the per-cell split, round-1 major #5).
    # Resume is per-FILE completeness (round 19), never a bare exists() check.
    # v2 mode SKIPS it: the v1 base panel is REUSED as the level denominator
    # (plan v3 section 4.3; validity guarded by the judge-stability anchor
    # check) — same model, SHA-frozen batteries, same decoding.
    if phase == "p1" and not args.skip_eval and not _v2_active():
        todo_passes = _base_panel_todo(args)
        if not todo_passes:
            logger.info("[phase=p1_0_base_panel] complete — every required column file present")
        else:
            print("[phase=p1_0_base_panel]", flush=True)
            for contexts, columns, missing in todo_passes:
                logger.info(
                    "[phase=p1_0_base_panel] contexts=%s missing %d required file(s): %s",
                    contexts,
                    len(missing),
                    missing,
                )
                for only in ["gen", "hf"] + ([] if args.skip_judges else ["judge"]):
                    _run(
                        _eval_cell_cmd(
                            row=None,
                            arm="",
                            seed=0,
                            adapter=None,
                            gpu=0,
                            contexts=contexts,
                            only=only,
                            base_panel=True,
                            max_probes=args.max_probes,
                            columns=columns,
                        ),
                        label=f"base-panel-{only}",
                    )

    n_gpus = args.parallel
    gpu_slots: Queue[int] = Queue()
    for g in range(n_gpus):
        gpu_slots.put(g)
    manifest_path = output_root() / f"manifest_{phase}.json"
    manifest: list[dict] = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done_cells = {m["cell"] for m in manifest}
    serial_cells = [(r, a, s) for r, a, s in cells if a == "fullft"]
    parallel_cells = [(r, a, s) for r, a, s in cells if a != "fullft"]

    def _worker(item):
        row, arm, seed = item
        if row.cell_id(arm, seed) in done_cells:
            logger.info("skip completed cell %s", row.cell_id(arm, seed))
            return None
        return _with_gpu_lease(gpu_slots, lambda gpu: _run_one_cell(row, arm, seed, gpu, args))

    # as_completed (round-1 minor #8): each cell's manifest entry persists
    # the moment ITS future finishes. Per-cell failures are COLLECTED, not
    # re-raised on first sight: ThreadPoolExecutor.__exit__ waits for pending
    # futures anyway (it does not cancel them), so an early raise would let
    # the remaining cells run for hours while silently dropping their
    # manifest entries (observed round-10). The phase still fails loud after
    # the pool drains.
    failures: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        future_cells = {
            pool.submit(_worker, item): item[0].cell_id(item[1], item[2]) for item in parallel_cells
        }
        for fut in as_completed(future_cells):
            try:
                res = fut.result()
            except BaseException as e:
                logger.error("cell %s FAILED: %s", future_cells[fut], e)
                failures.append((future_cells[fut], e))
                continue
            if res:
                manifest.append(res)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(manifest, indent=1))
    if failures:
        raise RuntimeError(
            f"{len(failures)} cell(s) failed in {phase}: "
            f"{[c for c, _ in failures]} — first error: {failures[0][1]!r}"
        ) from failures[0][1]
    for row, arm, seed in serial_cells:  # fullft uses all GPUs (ZeRO-3)
        if row.cell_id(arm, seed) in done_cells:
            continue
        res = _run_one_cell(row, arm, seed, 0, args)
        manifest.append(res)
        manifest_path.write_text(json.dumps(manifest, indent=1))

    if phase == "p1" and not args.skip_eval and not args.skip_judges and _v2_active():
        from explore_persona_space.experiments.behavior_testbed_545.gates import write_k1v2_gate

        print("[phase=k1v2_gate]", flush=True)
        write_k1v2_gate()
    if phase == "p1" and not args.skip_eval and not args.skip_judges and not _v2_active():
        from explore_persona_space.experiments.behavior_testbed_545.gates import (
            write_k1_gate,
            write_warmth_gate_result,
        )
        from explore_persona_space.experiments.behavior_testbed_545.preregister import (
            THRESHOLDS,
        )
        from explore_persona_space.experiments.behavior_testbed_545.rows import get_row

        print("[phase=k1_gate]", flush=True)
        write_k1_gate()
        if not args.smoke:
            if "warmth" in {r.row_id for r, _a, _s in cells}:
                print("[phase=warmth_gate]", flush=True)
                gate_cfg = THRESHOLDS["b10_warmth_gate"]
                write_warmth_gate_result(
                    warmth_cells=[
                        get_row("warmth").cell_id("primary", s) for s in get_row("warmth").seeds
                    ],
                    anchor_band=tuple(gate_cfg["anchor_normalized_band"]),
                    min_coherence_rate=gate_cfg["min_coherence_rate"],
                )
            if "bad_medical" in {r.row_id for r, _a, _s in cells}:
                _judge_sensitivity_check(args)
    if not args.skip_upload:
        print(f"[phase=upload_{phase}]", flush=True)
        bulk_upload_phase(phase)


# ---------------------------------------------------------------------------
# Bulk per-phase uploads (HF 256-commit/hr rule: ONE upload_folder per tree)
# ---------------------------------------------------------------------------


def _mirror_513_uploads(api, adapters: Path, list_repo_files, hf_model_repo: str) -> list[str]:
    """#513 coordination mirror: B1/B2 adapters also under the
    issue458_pair_<cell> convention (plan section 3a — cell names per #458:
    turner_* for the Turner organisms, insecure_code/educational for the
    Betley rows). Returns gap strings for any mirror missing from a listing
    REFRESHED after the mirror uploads — the first snapshot predates them, so
    deletion would otherwise never verify the mirror (round-2 Codex major
    i545-513-mirror-unverified-before-delete).
    """
    i513_cells = {
        "bad_medical": "turner_bad_medical",
        "risky_financial": "turner_risky_financial",
        "extreme_sports": "turner_extreme_sports",
        "insecure_code": "insecure_code",
        "educational_insecure": "educational",
    }
    mirror_probes: list[tuple[str, str]] = []
    for cell_dir in adapters.iterdir():
        for row_id, i458_cell in i513_cells.items():
            if (
                cell_dir.name.startswith(f"{row_id}_primary_seed")
                and (cell_dir / "adapter_config.json").exists()
            ):
                seed = cell_dir.name.rsplit("seed", 1)[1]
                mirror_dir = f"issue458_pair_{i458_cell}_seed{seed}/sft_narrow_adapter"
                api.upload_folder(
                    folder_path=str(cell_dir),
                    repo_id=hf_model_repo,
                    path_in_repo=mirror_dir,
                    commit_message=f"issue #545: #513-convention mirror {cell_dir.name}",
                    ignore_patterns=["checkpoint-*/optimizer.pt", "checkpoint-*/scheduler.pt"],
                )
                mirror_probes.append((cell_dir.name, f"{mirror_dir}/adapter_config.json"))
    if not mirror_probes:
        return []
    listed = set(list_repo_files(hf_model_repo))
    return [
        f"#513 mirror for {cell_name} missing post-upload ({probe})"
        for cell_name, probe in mirror_probes
        if probe not in listed
    ]


def bulk_upload_phase(phase: str) -> None:
    """Adapters -> model repo; corpora + raw completions -> data repo.

    One ``upload_folder`` commit per tree; verification via
    ``list_repo_files`` (NEVER the hf CLI); any gap is a TRACKED failure
    written to upload_gaps_<phase>.json — never a warning-and-continue.
    AFTER verification passes, every verified adapter / full-model cell dir
    is DELETED (upload-policy: per-phase upload+delete keeps the run under
    the ~130GB MooseFS quota; round-1 Codex major) — manifests, eval JSONs,
    and corpora stay. Fails loud BEFORE any rm when verification has gaps —
    including a #513-convention mirror missing from the post-mirror listing.

    SMOKE-GATED (round 20): under smoke-output isolation NOTHING uploads —
    the adapter/corpora/cells trees would land on the SAME production HF
    paths (``issue545_rows``, ``{HF_DATA_PREFIX}/corpora``, ...) and a smoke
    run executed after production completion would overwrite the durable
    production copies (the round-18 contamination class, relocated to HF).
    """

    from explore_persona_space.experiments.behavior_testbed_545 import (
        HF_DATA_REPO,
        HF_MODEL_REPO,
        cells_dir,
        corpora_dir,
        hf_data_prefix,
        hf_rows_prefix,
        output_root,
        smoke_output_active,
    )

    if smoke_output_active():
        logger.info("[upload] skipped under smoke isolation")
        return

    from huggingface_hub import HfApi, list_repo_files

    # v2 namespace isolation (plan divergence 5): adapters -> issue545_rows_v2,
    # data -> issue545_behavior_testbed_v2/... — v1 HF paths never overwritten.
    rows_prefix = hf_rows_prefix()
    data_prefix = hf_data_prefix()
    api = HfApi()
    gaps: list[str] = []
    verified_cell_dirs: list[Path] = []
    adapters = _adapters_root()
    if adapters.exists() and any(adapters.iterdir()):
        api.upload_folder(
            folder_path=str(adapters),
            repo_id=HF_MODEL_REPO,
            path_in_repo=rows_prefix,
            commit_message=f"issue #545 {phase}: adapter bulk upload",
        )
        listed = set(list_repo_files(HF_MODEL_REPO))
        for cell_dir in adapters.iterdir():
            if not cell_dir.is_dir():
                continue
            if (cell_dir / "adapter_config.json").exists():
                probe = f"{rows_prefix}/{cell_dir.name}/adapter_config.json"
            elif (cell_dir / "config.json").exists():
                # fullft cells upload a FULL model — verify it too (round-1
                # unaddressed case: the fullft upload was never gap-checked).
                probe = f"{rows_prefix}/{cell_dir.name}/config.json"
            else:
                continue
            if probe not in listed:
                gaps.append(f"cell {cell_dir.name} missing post-upload ({probe})")
            else:
                verified_cell_dirs.append(cell_dir)
        if not _v2_active():
            # The #513-convention mirror is a v1 coordination artifact; v2
            # rows (no B1/B2 rows in the v2 registry) never mirror.
            gaps.extend(_mirror_513_uploads(api, adapters, list_repo_files, HF_MODEL_REPO))
    if corpora_dir().exists():
        api.upload_folder(
            folder_path=str(corpora_dir()),
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{data_prefix}/corpora",
            commit_message=f"issue #545 {phase}: corpora",
        )
    if _v2_active() and (output_root() / "elicitation").exists():
        # Elicitation pools (pool_meta + per-tier candidate caches +
        # calibration records) — the section 6.5 deliverable glob.
        api.upload_folder(
            folder_path=str(output_root() / "elicitation"),
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{data_prefix}/elicitation",
            commit_message=f"issue #545 {phase}: v2 elicitation pools",
        )
    if cells_dir().exists():
        api.upload_folder(
            folder_path=str(cells_dir()),
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{data_prefix}/raw_completions",
            commit_message=f"issue #545 {phase}: per-cell completions + verdicts",
            allow_patterns=["*completions__*.json", "*/*.json"],
        )
    gaps_path = output_root() / f"upload_gaps_{phase}.json"
    gaps_path.write_text(json.dumps({"gaps": gaps}, indent=1))
    if gaps:
        # Fail-loud BEFORE any deletion (upload-policy: never delete an
        # unverified artifact).
        raise RuntimeError(f"Upload verification gaps ({len(gaps)}): {gaps[:5]} — see {gaps_path}")
    logger.info("[phase=upload_%s] verified clean", phase)

    # Post-verification cleanup (round-1 Codex major): delete the verified
    # adapter / full-model trees so per-phase disk stays under the ~130GB
    # MooseFS pod quota. Eval JSONs (cells_dir), manifests, gates, and
    # corpora are kept — only weights go.
    freed_before = shutil.disk_usage(adapters).free if adapters.exists() else None
    for cell_dir in verified_cell_dirs:
        shutil.rmtree(cell_dir)
        logger.info("[phase=upload_%s] deleted verified weights %s", phase, cell_dir.name)
    if verified_cell_dirs and freed_before is not None:
        usage = shutil.disk_usage(adapters.parent)
        logger.info(
            "[phase=upload_%s] disk after cleanup: free=%.1fGB (+%.1fGB freed) total=%.1fGB "
            "(deleted %d cells)",
            phase,
            usage.free / 1e9,
            (usage.free - freed_before) / 1e9,
            usage.total / 1e9,
            len(verified_cell_dirs),
        )


# ---------------------------------------------------------------------------
# P3 + assembly + scoring
# ---------------------------------------------------------------------------


def phase_p3(args) -> None:
    """P3 in dependency order: assemble -> predictors -> score (round 24).

    ``assemble()`` reads ONLY the per-cell eval JSONs under ``cells/`` (+
    dose files) and WRITES ``base_panel.json`` / ``L_matrix.json`` /
    ``cell_metadata.json``; the predictors subprocess READS
    ``base_panel.json`` (``extract_base_prior``); ``score()`` READS
    preregistration + L_matrix + cell_metadata + ``predictors/``. The prior
    predictors-first order crashed P3 on the pod with FileNotFoundError
    (base_panel.json not yet written — task #545 ``epm:failure`` v9).
    """
    if _v2_active():
        # P3-v2 (plan v3 section 4.4): demos rebuilt from the 160-row v2
        # corpora FIRST (the predictor demo-A flavor consumes them).
        print("[phase=p3_demos]", flush=True)
        from explore_persona_space.experiments.behavior_testbed_545.corpora import (
            build_demo_sets,
        )

        build_demo_sets()
    print("[phase=assemble]", flush=True)
    from explore_persona_space.experiments.behavior_testbed_545.assemble_matrix import assemble

    assemble()
    print("[phase=p3_predictors]", flush=True)
    if not args.skip_eval:
        _run(
            [
                "uv",
                "run",
                "python",
                "scripts/issue545_extract_predictors.py",
                "--gpu-id",
                "0",
            ]
            + (["--skip-gpu"] if args.skip_train and args.skip_eval else []),
            label="predictors",
        )
    if _v2_active():
        # The v2 predictor RESCORE is VM-side, post-termination, labeled
        # sensitivity (scoring_v2/ via scripts/issue545_v2_comparison.py) —
        # the pod-side prereg score() run is a v1 deliverable, not v2's.
        logger.info("[phase=p3] v2: scoring deferred to the VM-side comparison harness")
        return
    print("[phase=score]", flush=True)
    from explore_persona_space.experiments.behavior_testbed_545.scoring import score

    score(include_flagged=False)
    score(include_flagged=True)  # sensitivity pass


# ---------------------------------------------------------------------------
# Sentinel (poll_pipeline.py contract)
# ---------------------------------------------------------------------------


def write_sentinel(note: str, *, kind: str = "epm:results") -> Path:
    logs_dir = Path("/workspace/logs")
    if not logs_dir.exists():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    kind_slug = kind.replace(":", "_")
    path = logs_dir / f"issue-{ISSUE}-{kind_slug}-{int(time.time())}.json"
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
                "kind": kind,
                "version": 1,
                "task_id": ISSUE,
                "note": note,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "by": "issue545_sweep",
            },
            indent=1,
        )
    )
    logger.info("sentinel written: %s", path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #545 behavior-testbed dispatcher (smoke IS sweep with one cell)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase", required=True, choices=("p0", "p1", "p2", "p3", "assemble", "score")
    )
    parser.add_argument("--rows", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true", help="one gate cell, tiny caps, same path")
    parser.add_argument(
        "--v2",
        action="store_true",
        help="onpolicy-testbed-v2 namespace mode (equivalent to I545_V2_OUTPUT=1)",
    )
    parser.add_argument("--build-corpora", action="store_true")
    parser.add_argument("--preregister", action="store_true")
    parser.add_argument("--corpora-n", type=int, default=None, help="cap generated corpus rows")
    parser.add_argument("--max-probes", type=int, default=None, help="cap probes per battery")
    parser.add_argument("--parallel", type=int, default=None, help="GPU count (default: detect)")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-judges", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    if args.v2 or os.environ.get("I545_V2_OUTPUT") == "1":
        _activate_v2_namespace()
    if args.smoke:
        # Smoke cell = the active registry's gate row (v1: marker; v2: the
        # K1-v2 gate row refuse_medical with a reduced pool — plan v3 4.4).
        args.rows = args.rows or (["refuse_medical"] if _v2_active() else ["marker"])
        args.seeds = args.seeds or [0]
        args.arms = args.arms or ["primary"]
        args.max_probes = args.max_probes or 4
        args.corpora_n = args.corpora_n or 2
        _activate_smoke_isolation()
    if args.parallel is None:
        try:
            n = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, check=True, env={**os.environ}
            ).stdout.count("GPU ")
            args.parallel = max(1, n)
        except (subprocess.CalledProcessError, FileNotFoundError):
            args.parallel = 1

    print(f"[phase={args.phase}]", flush=True)
    if args.phase == "p0":
        if not (args.build_corpora or args.preregister):
            args.build_corpora = args.preregister = True
        phase_p0(args)
        note = "p0 complete: panels + corpora + batteries + preregistration"
    elif args.phase in ("p1", "p2"):
        phase_train_eval(args, args.phase)
        note = f"{args.phase} complete: see manifest_{args.phase}.json"
        from explore_persona_space.experiments.behavior_testbed_545 import output_root

        js_path = output_root() / "judge_sensitivity.json"
        if js_path.exists():
            js = json.loads(js_path.read_text())
            note += f"; judge_sensitivity locked={js.get('locked')}"
            if not js.get("locked"):
                note += " — escalate boundary cells to 100/q (plan gate 3, no automated consumer)"
    elif args.phase == "p3":
        phase_p3(args)
        note = "p3 complete: predictors + L matrix + scoring_results.json"
    elif args.phase == "assemble":
        from explore_persona_space.experiments.behavior_testbed_545.assemble_matrix import assemble

        assemble()
        note = "assemble complete"
    else:
        from explore_persona_space.experiments.behavior_testbed_545.scoring import score

        score(include_flagged=False)
        note = "score complete"
    write_sentinel(f"issue #545 dispatcher: {note}")
    print("[phase=done]", flush=True)
    # Hard-exit the SUCCESS path only: with datasets/pyarrow + tokenizers
    # loaded in-process, interpreter teardown intermittently aborts with
    # "terminate called without an active exception" (exit 134) AFTER all
    # work + the sentinel + [phase=done] have completed (observed on the P0
    # smoke, 2026-06-10). Every failure path raises BEFORE this point
    # (check=True subprocesses + fail-loud builders), so no error can be
    # masked here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
