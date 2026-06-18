#!/usr/bin/env python3
# ruff: noqa: RUF002  # em-dash, minus sign, marker token intentional
"""Task #601 dispatcher — UNIFIED smoke = sweep with one cell.

Forked from ``scripts/dispatch_neg_geometry_472.py`` (origin/issue-472).

Pipeline (plan §4/§10):
  fetch_artifacts   parent inputs (bank / centroids / R) from the HF data repo
  gate checks       phase0_gate.json pass==true (ALL launches) + the smoke
                    sentinel (non-smoke launches) + phase4a_verdict.json
                    call==non-arrest over the two UNCONDITIONAL Phase-4
                    bridge cells (any conditional Phase-4b cell, i.e.
                    posonly_attn_lr1e5 / the --cells phase4b group) — plan
                    §7/§4 gates, enforced in code; a hand-pasted sweep
                    command cannot bypass them
  per cell×seed     build → train → on-policy eval → dense read → ckpt upload
                    [GPU-pinned i601_run_cell subprocess pool]
  smoke gate        (--smoke) the §4 asserts over the completed smoke cell;
                    writes the smoke sentinel the sweep launch requires
  raw-completions   upload_raw_completions_to_data_repo over the slab root
  final sentinel    /workspace/logs/issue-601-results.json (epm:results v1)

UNIFICATION (smoke-architecture parity = PASS_UNIFIED): smoke = this
dispatcher with ``--cells ratio4to1_100p400n --seeds 42 --smoke`` — ONE FULL
cell (T≈32 complete schedule; plan §7 gate 1 runs the real unit, not a tiny
slice), same subprocess shape / env injection / logging surface / teardown as
the sweep. Every GPU phase runs inside the per-cell unit enumerated from
``--cells``; the raw-completions upload rglobs only what those cells wrote.

GPU pinning: each cell subprocess gets ``--gpu-id <g>`` AND
``CUDA_VISIBLE_DEVICES=<g>`` exported in its LAUNCHER env (gotcha #545:
import-time cuInit — e.g. ``import peft`` — freezes the driver device list
before train/sft.py's in-process clobber, so the env pin must happen at exec;
sft.py re-sets CVD to the SAME physical value so the two never fight).

Pod-side discipline (CLAUDE.md): NEVER shells out to scripts/task.py; every
subprocess.* passes env={**os.environ}; load_dotenv() at module top; vLLM
phases are subprocess-isolated; [phase=done] is emitted ONLY by the terminal
full-sweep exit path (the launch driver redirects smoke/phase0 output to
sub-logs so the main pipeline log carries a single terminal done line).
"""

from __future__ import annotations

import argparse
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

log = logging.getLogger("dispatch_neg_setpoint_601")

LOG_DIR = Path("/workspace/logs")  # overridden by --log-dir.
# Defaults for --smoke-cell / --smoke-seed (#622 made these CLI args so child
# issues reusing this dispatcher smoke on THEIR OWN sweep cell — the #601
# values stay the defaults so every existing caller is byte-identical).
SMOKE_CELL = "ratio4to1_100p400n"
SMOKE_SEED = 42
# Round-6 amendment of the plan §4/§10 smoke assert ("in-loop vs on-policy
# <= 1 nat"): that form compared the LIVE rsLoRA training gauge (alpha/sqrt(r)
# ~= 11.31; train/sft.py keeps use_rslora=True) against the STAGED classic
# alpha/r = 2.0 on-policy read and fails BY CONSTRUCTION on any cell where the
# implant took (live-vs-classic gap 5-15 nats). The assert's INTENT —
# eval-path integrity, the #534 adapter-not-applied class — is preserved by a
# SAME-GAUGE pair: on-policy vLLM source ΔG vs the SAME terminal checkpoint's
# Phase-B teacher-forced HF read, both staged classic, both already in
# trajectory.json. Tolerance = plan §12 assumption-16's dense-vs-on-policy
# admission threshold (2 nats; mirrors ONPOLICY_ADMISSION_TOL_NATS in
# i601_analyze.py). The in-loop band value is recorded telemetry only (live
# gauge — never asserted cross-gauge).
ONPOLICY_VS_TF_SAMEGAUGE_TOL_NATS = 2.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_sentinel(
    path: Path, *, kind: str, phase: str, note_payload: dict, task_id: int = 601
) -> None:
    """poll_pipeline.py-compliant sentinel (sentinel_schema_version=1, kind, version)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": task_id,
                "by": "dispatch_neg_setpoint_601",
                "ts": datetime.now(UTC).isoformat(),
                "phase": phase,
                "note": json.dumps(note_payload),
            },
            indent=2,
        )
    )


def _fetch_parent_artifacts(data_dir: Path) -> dict:
    """Download the pinned #472 inputs from the HF data repo (idempotent).

    Delegates to ``neg_setpoint_601.artifacts.fetch_parent_data`` (shared with
    the Phase 0 driver). ``data_dir`` is ``<repo_root>/data/issue_601``, so the
    repo root the relative destinations resolve against is two levels up.
    """
    from explore_persona_space.experiments.neg_setpoint_601.artifacts import fetch_parent_data

    repo_root = data_dir.resolve().parent.parent
    fetched = fetch_parent_data(repo_root)
    log.info("[phase=fetch_artifacts] %d artifacts present", len(fetched))
    return fetched


def _check_gates(
    slab_root: Path,
    log_dir: Path,
    *,
    smoke: bool,
    dry_run: bool,
    sentinel_task_id: int = 601,
    smoke_cell: str = SMOKE_CELL,
    smoke_seed: int = SMOKE_SEED,
) -> None:
    """Plan §7 gates, enforced in code (defense in depth vs hand-pasted sweeps).

    ALL launches require ``phase0_gate.json`` with ``pass: true``; non-smoke
    launches ADDITIONALLY require the smoke sentinel (written by this
    dispatcher's --smoke gate on PASS; ``.processed`` rename by the poller is
    accepted). Child issues (#622) write their own phase0_gate.json from
    their launch driver's p0 asserts (no adapter-reuse Phase 0 there) — the
    gate's meaning stays "the registered p0 step ran and PASSed".
    """
    if dry_run:
        log.info("[phase=gates] SKIP (dry-run)")
        return
    gate_path = slab_root / "phase0" / "phase0_gate.json"
    if not gate_path.exists():
        raise RuntimeError(
            f"GATE REFUSAL: {gate_path} missing — the registered Phase-0 step "
            f"(i601_phase0_reads.py for #601; the launch driver's p0 asserts for child "
            f"issues) must run and PASS before any training cell (plan §7 gate 2)."
        )
    gate = json.loads(gate_path.read_text())
    if gate.get("pass") is not True:
        raise RuntimeError(
            f"GATE REFUSAL: phase0_gate.json pass={gate.get('pass')!r} — the registered "
            f"Phase-0 step failed; training phases are HALTED."
        )
    log.info("[phase=gates] phase0 gate PASS")
    if smoke:
        return
    smoke_sentinel = log_dir / f"issue-{sentinel_task_id}-smoke-results.json"
    processed = smoke_sentinel.with_suffix(".json.processed")
    candidate = smoke_sentinel if smoke_sentinel.exists() else processed
    if not candidate.exists():
        raise RuntimeError(
            f"GATE REFUSAL: smoke sentinel missing at {smoke_sentinel} — run the smoke "
            f"(--cells {smoke_cell} --seeds {smoke_seed} --smoke) and pass its gate first."
        )
    payload = json.loads(candidate.read_text())
    note = json.loads(payload.get("note") or payload.get("payload") or "{}")
    if note.get("smoke_gate_pass") is not True:
        raise RuntimeError(
            f"GATE REFUSAL: smoke sentinel at {candidate} does not record a PASS "
            f"(smoke_gate_pass={note.get('smoke_gate_pass')!r})."
        )
    log.info("[phase=gates] smoke gate PASS (sentinel: %s)", candidate)


def _check_phase4b_gate(slab_root: Path, conditional_requested: list[str]) -> None:
    """Plan §4 Phase-4b conditional gate, enforced in code.

    The conditional factor cell (``posonly_attn_lr1e5``, the only conditional
    cell as of round 4) is dispatchable ONLY behind a ``phase4a_verdict.json``
    recording a bridge NON-ARREST classification over the two UNCONDITIONAL
    Phase-4 cells (written post-sweep by ``scripts/i601_phase4_verdict.py``;
    ``i601_launch.sh`` routes on it). Arrest/ambiguous → 4b uninformative,
    skipped, reported open — a hand-pasted ``--cells phase4b`` cannot bypass
    the routing.
    """
    verdict_path = slab_root / "phase4" / "phase4a_verdict.json"
    if not verdict_path.exists():
        raise RuntimeError(
            f"GATE REFUSAL: conditional Phase-4b cells {conditional_requested} require "
            f"{verdict_path} (run scripts/i601_phase4_verdict.py after the unconditional "
            f"bridge cells complete) — plan §4 Phase 4b."
        )
    verdict = json.loads(verdict_path.read_text())
    if verdict.get("dispatch_4b") is not True or verdict.get("call") != "non-arrest":
        raise RuntimeError(
            f"GATE REFUSAL: Phase-4b cells are gated on a bridge NON-ARREST classification; "
            f"{verdict_path} records call={verdict.get('call')!r} "
            f"(dispatch_4b={verdict.get('dispatch_4b')!r}). Arrest/ambiguous → 4b is "
            f"uninformative, skipped, and reported open (plan §4/§7)."
        )
    log.info("[phase=gates] phase4b gate PASS (bridge call=non-arrest)")


def _schedule_cell_pool(
    *,
    units: list[tuple[str, int]],
    n_gpus: int,
    max_parallel: int,
    slab_root: Path,
    runs_root: Path,
    log_dir: Path,
    data_dir: Path,
    report_to: str,
    resume: bool,
    sentinel_task_id: int = 601,
    hf_prefix: str | None = None,
    run_name_prefix: str = "issue601",
) -> list[dict]:
    """GPU-sharded i601_run_cell subprocess pool (forked from #472 verbatim,
    plus the launcher-env CVD pin per gotcha #545)."""
    from explore_persona_space.experiments.neg_setpoint_601 import cell_by_slug

    if max_parallel > n_gpus:
        log.warning("max_parallel=%d > n_gpus=%d; clamping.", max_parallel, n_gpus)
        max_parallel = n_gpus
    log.info("Scheduling %d (cell,seed) units across %d GPUs", len(units), n_gpus)

    results: list[dict] = []
    running: list[tuple[subprocess.Popen, str, int, int]] = []
    queue = list(units)
    free_gpus: list[int] = list(range(n_gpus))

    def _launch(cell: str, seed: int, gpu: int) -> subprocess.Popen | None:
        spec = cell_by_slug(cell)
        out_traj = slab_root / spec.phase / f"{cell}_seed{seed}" / "trajectory.json"
        if resume and out_traj.exists():
            log.info("[%s seed%d] RESUME: trajectory exists; skipping.", cell, seed)
            return None
        # Launcher-env CVD pin (gotcha #545) + matching --gpu-id: sft.py's
        # in-process clobber rewrites the SAME physical value, and any
        # import-time cuInit in the child already sees only physical <gpu>.
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/i601_run_cell.py",
            "--cell",
            cell,
            "--seed",
            str(seed),
            "--gpu-id",
            str(gpu),
            "--slab-root",
            str(slab_root),
            "--runs-root",
            str(runs_root),
            "--log-dir",
            str(log_dir),
            "--data-dir",
            str(data_dir),
            "--report-to",
            report_to,
            "--run-name-prefix",
            run_name_prefix,
            "--sentinel-task-id",
            str(sentinel_task_id),
        ]
        if hf_prefix is not None:
            cmd.extend(["--hf-prefix", hf_prefix])
        cell_log = log_dir / f"issue-{sentinel_task_id}-{cell}-seed{seed}.log"
        cell_log.parent.mkdir(parents=True, exist_ok=True)
        log.info("[%s seed%d] launch on GPU %d → %s", cell, seed, gpu, cell_log)
        fh = open(cell_log, "w")  # noqa: SIM115 -- handle lives for the Popen's lifetime
        return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)

    from explore_persona_space.experiments.neg_setpoint_601 import HF_ADAPTER_PREFIX_601

    adapter_prefix = hf_prefix if hf_prefix is not None else HF_ADAPTER_PREFIX_601

    while queue or running:
        while queue and len(running) < max_parallel and free_gpus:
            cell, seed = queue.pop(0)
            gpu = free_gpus.pop(0)
            proc = _launch(cell, seed, gpu)
            if proc is None:
                # Round-2 binding fix resumed-smoke-adapter-path-missing: a
                # resumed unit's adapter/run identities are DETERMINISTIC from
                # the spec + prefixes, so the record carries the full
                # reproducibility fields — otherwise the canonical launch path
                # (p3 smoke, then p5 sweep --resume) ships an 11/12-entry
                # adapter_paths card with the smoke unit silently absent.
                spec = cell_by_slug(cell)
                results.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "status": "resumed_skip",
                        "phase": spec.phase,
                        "trajectory_path": str(
                            slab_root / spec.phase / f"{cell}_seed{seed}" / "trajectory.json"
                        ),
                        "adapter_hf_path": f"{adapter_prefix}/{cell}_seed{seed}",
                        "wandb_run_name": f"{run_name_prefix}_{cell}_seed{seed}",
                    }
                )
                free_gpus.append(gpu)
                continue
            running.append((proc, cell, seed, gpu))
        still: list[tuple[subprocess.Popen, str, int, int]] = []
        for proc, cell, seed, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, cell, seed, gpu))
                continue
            free_gpus.append(gpu)
            if rc != 0:
                fail_path = log_dir / f"issue-{sentinel_task_id}-{cell}-seed{seed}-FAILED.json"
                fail_path.write_text(
                    json.dumps(
                        {"cell": cell, "seed": seed, "returncode": rc, "assigned_gpu": gpu},
                        indent=2,
                    )
                )
                # Terminate EVERYTHING still alive (not just procs already
                # polled into `still` this iteration) so an abort never
                # orphans GPU-holding cell subprocesses (round-1 review minor).
                for p2, _c2, _s2, _g2 in running:
                    if p2 is not proc and p2.poll() is None:
                        p2.terminate()
                raise RuntimeError(
                    f"[{cell} seed{seed}] cell subprocess exited rc={rc} (GPU {gpu}). "
                    f"See {log_dir}/issue-{sentinel_task_id}-{cell}-seed{seed}.log. "
                    f"Sweep aborted."
                )
            spec = cell_by_slug(cell)
            log.info("cell %s seed%d complete (GPU %d)", cell, seed, gpu)
            results.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "status": "done",
                    "assigned_gpu": gpu,
                    "phase": spec.phase,
                    "trajectory_path": str(
                        slab_root / spec.phase / f"{cell}_seed{seed}" / "trajectory.json"
                    ),
                    "adapter_hf_path": f"{adapter_prefix}/{cell}_seed{seed}",
                    "wandb_run_name": f"{run_name_prefix}_{cell}_seed{seed}",
                }
            )
        running = still
        if running:
            time.sleep(5)
    return results


def _smoke_gate(
    slab_root: Path,
    runs_root: Path,
    *,
    smoke_cell: str = SMOKE_CELL,
    smoke_seed: int = SMOKE_SEED,
) -> dict:
    """The §4 smoke asserts over the completed smoke cell (plan §7 gate 1).

    Checks (all over REAL artifacts the full cell just wrote):
      1. realized terminal step == expected T (band-stop misfire catch; the
         worker also hard-asserts this in-process — this re-reads the index).
      2. four-float fields present in BOTH the on-policy trajectory.json and
         the in-loop band trajectory (storage contract, incident #530).
      3. SAME-GAUGE eval-path integrity (#534 adapter-application class):
         on-policy (vLLM, staged classic) terminal source ΔG vs the SAME
         terminal checkpoint's Phase-B teacher-forced HF read (also staged
         classic) agree within 2 nats (plan §12 assumption-16 admission
         threshold). Round-6 amendment of the plan §4/§10 "in-loop vs
         on-policy <= 1 nat" form, which crossed gauges (live rsLoRA vs
         staged classic) and failed by construction on a working implant.
         The in-loop band terminal value is RECORDED (gauge-labeled
         telemetry) but never asserted cross-gauge.
      4. per-row-type CE probe logged >= 1 record with BOTH sides non-null
         (plan-declared guard telemetry must demonstrably function).
      5. (#622, cells with probe_dense_until > 0) the rowtype series follows
         the registered STRIDED cadence — dense (every step) through
         dense_until, then every probe_every_steps.
      6. (#622, cells with capability_trajectory) capability_trajectory.json
         present with >= 2 records (the hard-fail wrapper demonstrably
         functioned).
      7. (#622) built training-pool row count == the registry's total_rows
         (T arithmetic / twin matching integrity; also asserted in-process
         by the worker — this re-reads the durable file).
      8. (#622, cells with eval_include_trained_negatives) the on-policy
         trajectory's terminal held_out contains ALL 4 trained anchor
         negatives AND panel_roles tags them "trained_negative" — the static
         launch-resolution assert that i622_analyze.py's DV6
         trained-negatives bucket is non-empty (round-2 binding fix
         dv6-trained-negatives-onpolicy-missing: the bucket previously had a
         reader but no writer).
    """
    from explore_persona_space.experiments.neg_setpoint_601 import cell_by_slug

    spec = cell_by_slug(smoke_cell)
    cell_dir = slab_root / spec.phase / f"{smoke_cell}_seed{smoke_seed}"
    out: dict = {"cell": smoke_cell, "seed": smoke_seed, "checks": {}}

    # 1. realized steps == expected T.
    ckpt_index_path = runs_root / f"{smoke_cell}_seed{smoke_seed}" / "checkpoint_index.json"
    idx = json.loads(ckpt_index_path.read_text())
    realized = idx.get("1.0000", {}).get("step")
    steps_ok = realized is not None and int(realized) == spec.expected_steps
    out["checks"]["realized_steps"] = {
        "realized": realized,
        "expected": spec.expected_steps,
        "ok": bool(steps_ok),
    }

    # 2a. on-policy trajectory four-float fields.
    traj = json.loads((cell_dir / "trajectory.json").read_text())
    term = max(traj["checkpoints"], key=lambda c: c["frac"])
    ss = term["source_self"]
    needed = ("z_marker_g_mean", "z_marker_b_mean", "z_eos_g_mean", "z_eos_b_mean", "logZ_g_mean")
    onpolicy_fields_ok = all(ss.get(k) is not None for k in needed)
    leaf_ok = True
    for per_q in term["held_out"].values():
        for rec in per_q.values():
            leaf_ok = leaf_ok and all(
                rec.get(k) is not None for k in ("z_marker_g", "z_eos_g", "logZ_g", "kl")
            )
    out["checks"]["onpolicy_four_float"] = {"source": onpolicy_fields_ok, "held_out": leaf_ok}

    # 2b. in-loop band trajectory four-float fields + >=1 record.
    band = json.loads((cell_dir / "inloop_band_trajectory.json").read_text())
    n_band = len(band.get("steps", []))
    band_four = (
        n_band >= 1
        and band["z_marker_trained"][0] is not None
        and band["z_eos_trained"][0] is not None
        and band["logZ_trained"][0] is not None
        and band["log_p_base"][0] is not None
    )
    out["checks"]["inloop_four_float"] = {"n_records": n_band, "ok": bool(band_four)}

    # 3. SAME-GAUGE eval-path integrity: on-policy vLLM source ΔG vs the SAME
    #    terminal checkpoint's Phase-B teacher-forced HF read, both from
    #    trajectory.json, both staged classic alpha/r (round-6 amendment —
    #    see the ONPOLICY_VS_TF_SAMEGAUGE_TOL_NATS comment at module top).
    onpolicy_delta = ss.get("delta_g_mean")
    tf_hf_g = ss.get("logp_hf_g_mean")
    tf_hf_b = ss.get("logp_hf_b_mean")
    tf_delta = (
        float(tf_hf_g) - float(tf_hf_b) if tf_hf_g is not None and tf_hf_b is not None else None
    )
    agree = (
        onpolicy_delta is not None
        and tf_delta is not None
        and abs(onpolicy_delta - tf_delta) <= ONPOLICY_VS_TF_SAMEGAUGE_TOL_NATS
    )
    out["checks"]["onpolicy_vs_tf_same_gauge"] = {
        "onpolicy_terminal_delta": onpolicy_delta,
        "tf_hf_terminal_delta": tf_delta,
        "tol_nats": ONPOLICY_VS_TF_SAMEGAUGE_TOL_NATS,
        "gauge": "staged classic alpha/r on BOTH sides (use_rslora_applied=False)",
        "ok": bool(agree),
    }
    # In-loop band terminal ΔG: RECORDED TELEMETRY ONLY. The in-loop probe
    # reads the LIVE training model (rsLoRA alpha/sqrt(r)); asserting it
    # against the staged classic reads is the cross-gauge comparison the
    # round-6 fix retired — on the same weights the two gauges agree within
    # ~1 nat only when the implant did nothing.
    inloop_delta = band["delta_nats"][-1] if band.get("delta_nats") else None
    out["checks"]["inloop_terminal_telemetry"] = {
        "inloop_terminal_delta": inloop_delta,
        "gauge": band.get("gauge")
        or {"note": "live-training-model (rsLoRA alpha/sqrt(r)); pre-round-6 band JSON"},
        "asserted": False,
    }

    # 4. row-type CE probe telemetry functioned. Phase-3-style cells with zero
    #    positive rows record a null pos channel by design; require the
    #    channels the cell's mix actually carries.
    ce = json.loads((cell_dir / "rowtype_ce.json").read_text())
    ce_steps = list(ce.get("steps", []))
    pos_needed = spec.pos_ex > 0
    neg_needed = spec.n_neg_personas * spec.neg_ex_per_persona > 0
    ce_ok = (
        len(ce_steps) >= 1
        and (not pos_needed or ce["pos_marker_ce"][0] is not None)
        and (not neg_needed or ce["neg_trailing_ce"][0] is not None)
    )
    out["checks"]["rowtype_ce_probe"] = {"n_records": len(ce_steps), "ok": bool(ce_ok)}

    # 5. (#622) strided probe cadence — the recorded steps must be EXACTLY the
    #    registered grid {1..dense_until} union {k*stride <= T}, dense first.
    if spec.probe_dense_until > 0:
        t = spec.expected_steps
        expected_steps_set = sorted(
            {
                *range(1, min(spec.probe_dense_until, t) + 1),
                *range(spec.probe_every_steps, t + 1, spec.probe_every_steps),
            }
        )
        cadence_ok = ce_steps == expected_steps_set
        out["checks"]["rowtype_strided_cadence"] = {
            "dense_until": spec.probe_dense_until,
            "stride": spec.probe_every_steps,
            "n_expected": len(expected_steps_set),
            "n_recorded": len(ce_steps),
            "ok": bool(cadence_ok),
        }
    else:
        cadence_ok = True

    # 6. (#622) capability trajectory functioned (>= 2 records proves the
    #    hard-fail wrapper ran AND its percent gate fired repeatedly).
    if spec.capability_trajectory:
        cap_path = cell_dir / "capability_trajectory.json"
        if cap_path.exists():
            cap = json.loads(cap_path.read_text())
            n_cap = int(cap.get("n_records", len(cap.get("records", []))))
        else:
            n_cap = -1  # file missing entirely
        cap_ok = n_cap >= 2
        out["checks"]["capability_trajectory"] = {"n_records": n_cap, "ok": bool(cap_ok)}
    else:
        cap_ok = True

    # 7. (#622) built row count == registered total_rows (durable re-read).
    pool_path = runs_root / f"{smoke_cell}_seed{smoke_seed}" / "train_pool.jsonl"
    if pool_path.exists():
        with pool_path.open() as fh:
            n_rows = sum(1 for line in fh if line.strip())
        rows_ok = n_rows == spec.total_rows
    else:
        n_rows, rows_ok = -1, False
    out["checks"]["build_row_count"] = {
        "n_rows": n_rows,
        "expected": spec.total_rows,
        "ok": bool(rows_ok),
    }

    # 8. (#622 DV6) trained anchor negatives present on the on-policy artifact
    #    with the trained_negative role — proves the analyzer's DV6 bucket is
    #    non-empty by construction before any sweep GPU is spent.
    if getattr(spec, "eval_include_trained_negatives", False):
        from explore_persona_space.experiments.neg_setpoint_601 import EXPECTED_ANCHOR_PANEL

        held_out_personas = set(term["held_out"].keys())
        roles = traj.get("panel_roles") or {}
        missing_negs = sorted(set(EXPECTED_ANCHOR_PANEL) - held_out_personas)
        mistagged = sorted(p for p in EXPECTED_ANCHOR_PANEL if roles.get(p) != "trained_negative")
        negs_ok = not missing_negs and not mistagged
        out["checks"]["trained_negatives_onpolicy"] = {
            "expected": list(EXPECTED_ANCHOR_PANEL),
            "missing_from_held_out": missing_negs,
            "missing_or_mistagged_in_panel_roles": mistagged,
            "ok": bool(negs_ok),
        }
    else:
        negs_ok = True

    out["smoke_gate_pass"] = bool(
        steps_ok
        and onpolicy_fields_ok
        and leaf_ok
        and band_four
        and agree
        and ce_ok
        and cadence_ok
        and cap_ok
        and rows_ok
        and negs_ok
    )
    return out


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- linear pipeline driver; splitting the gate/pool/sentinel steps would obscure the launch contract
    parser = argparse.ArgumentParser(
        description="Task #601 dispatcher (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cells",
        default=None,
        help="CSV of #601 slugs, or 'all' (= every non-conditional cell).",
    )
    parser.add_argument("--seeds", default="42,137")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run ONE FULL cell (--smoke-cell/--smoke-seed) + the §4 smoke gate.",
    )
    parser.add_argument(
        "--smoke-cell",
        default=SMOKE_CELL,
        help=f"Cell the --smoke run trains (default {SMOKE_CELL}; #622 passes its own "
        f"sweep cell so smoke IS the sweep with one cell).",
    )
    parser.add_argument(
        "--smoke-seed",
        type=int,
        default=SMOKE_SEED,
        help=f"Seed for the --smoke run (default {SMOKE_SEED}).",
    )
    # ── Child-issue thin flags (#622; defaults = byte-identical #601). ──────
    parser.add_argument(
        "--hf-prefix",
        default=None,
        help="HF adapter path prefix forwarded to each worker (default: the worker's "
        "HF_ADAPTER_PREFIX_601; #622 passes adapters/issue_622).",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="issue601",
        help="WandB run-name prefix forwarded to each worker (default issue601; #622 "
        "passes issue622).",
    )
    parser.add_argument(
        "--sentinel-task-id",
        type=int,
        default=601,
        help="Task id for every sentinel/log filename + task_id field (default 601; "
        "#622 passes 622 so its sentinels land at /workspace/logs/issue-622-*.json).",
    )
    parser.add_argument(
        "--hf-data-prefix",
        default=None,
        help="HF data-repo experiment prefix for the raw-completions upload (default: "
        "HF_DATA_PREFIX_601; #622 passes issue622_dose_break).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Wiring validation only, no GPU.")
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_601"))
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_601"))
    parser.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/issue_601"))
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-upload", action="store_true", help="Debug only.")
    parser.add_argument(
        "--sentinel-name",
        default=None,
        help=(
            "Final-sentinel filename under --log-dir (default: "
            "issue-<sentinel-task-id>-results.json). The conditional Phase-4b dispatch "
            "passes issue-601-phase4b-results.json so it never clobbers the main sweep's "
            "results sentinel."
        ),
    )
    parser.add_argument(
        "--anchor-retrain-fallback",
        action="store_true",
        help=(
            "Plan §4 Phase-0 item 3 fallback: the reused #472 anchor failed its fitness "
            "gate, so ALSO train dense_200p800n at seed 42 (the 4:1 anchor recipe; seed "
            "137 already runs as the Phase-2 dense cell) — together they replace the "
            "parent anchor as the middle fixed-ratio arm."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    global LOG_DIR
    LOG_DIR = args.log_dir
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    args.slab_root.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        args.runs_root.mkdir(parents=True, exist_ok=True)

    # MooseFS quota safety + adapter-persist parity with the parent dispatcher.
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    from explore_persona_space.experiments.neg_setpoint_601 import (
        EXPECTED_MARKER_TOKEN_ID,
        HF_DATA_PREFIX_601,
        HF_DATA_REPO,
        HF_MODEL_REPO,
        MARKER_TEXT,
        cell_by_slug,
        cells_for_request,
    )

    # Child-issue resolution (#622): the raw-completions experiment prefix and
    # the final-sentinel filename default to the #601 values.
    hf_data_prefix = args.hf_data_prefix if args.hf_data_prefix is not None else HF_DATA_PREFIX_601
    sentinel_name = (
        args.sentinel_name
        if args.sentinel_name is not None
        else f"issue-{args.sentinel_task_id}-results.json"
    )

    # ── Resolve units. ────────────────────────────────────────────────────────
    if args.smoke:
        cells = [cell_by_slug(args.smoke_cell)]
        seeds = [args.smoke_seed]
    else:
        cells = cells_for_request(args.cells)
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    units: list[tuple[str, int]] = []
    for spec in cells:
        for s in seeds:
            if s in spec.seeds:
                units.append((spec.slug, s))
    if args.anchor_retrain_fallback and ("dense_200p800n", 42) not in units:
        units.append(("dense_200p800n", 42))
        log.info("[phase=resolve] anchor-retrain fallback unit appended: dense_200p800n seed 42")
    if not units:
        raise ValueError("zero (cell, seed) units after intersecting --seeds with cell specs")

    # ── Phase-4b conditional gate (plan §4 Phase 4b). ─────────────────────────
    # Phase-filtered (follow-up round 1): the gate guards ONLY phase4
    # conditional cells. Non-phase4 conditional cells (posonly_200p_T130) are
    # explicit-slug follow-up launches and have no bridge-verdict dependency.
    conditional_requested = sorted({c.slug for c in cells if c.conditional and c.phase == "phase4"})
    if conditional_requested and not args.dry_run:
        _check_phase4b_gate(args.slab_root, conditional_requested)
    log.info(
        "[phase=resolve] %d units: %s (smoke=%s)",
        len(units),
        [f"{c}:{s}" for c, s in units],
        args.smoke,
    )

    # ── Pre-flight: marker tokenizer assertion (also in-process per cell). ───
    if not args.dry_run:
        from transformers import AutoTokenizer

        from explore_persona_space.experiments.neg_setpoint_601 import BASE_MODEL

        tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
        if ids != [EXPECTED_MARKER_TOKEN_ID]:
            raise RuntimeError(
                f"Marker tokenizer assertion FAILED: encode({MARKER_TEXT!r})={ids}, "
                f"expected [{EXPECTED_MARKER_TOKEN_ID}]."
            )
        log.info("[phase=preflight] marker assertion PASS: %r -> %s", MARKER_TEXT, ids)

    # ── Fetch parent artifacts. ───────────────────────────────────────────────
    if not args.skip_fetch and not args.dry_run:
        _fetch_parent_artifacts(args.data_dir)
    else:
        log.info("[phase=fetch_artifacts] SKIP")

    # ── §7 gates. ─────────────────────────────────────────────────────────────
    _check_gates(
        args.slab_root,
        LOG_DIR,
        smoke=args.smoke,
        dry_run=args.dry_run,
        sentinel_task_id=args.sentinel_task_id,
        smoke_cell=args.smoke_cell,
        smoke_seed=args.smoke_seed,
    )

    if args.dry_run:
        log.info("[phase=dry_run_done] wiring validated (units=%d); no GPU work.", len(units))
        return 0

    # ── Per-cell pool. ────────────────────────────────────────────────────────
    log.info("[phase=cells] scheduling %d units", len(units))
    cell_results = _schedule_cell_pool(
        units=units,
        n_gpus=args.n_gpus,
        max_parallel=args.max_parallel,
        slab_root=args.slab_root,
        runs_root=args.runs_root,
        log_dir=LOG_DIR,
        data_dir=args.data_dir,
        report_to=args.report_to,
        resume=args.resume,
        sentinel_task_id=args.sentinel_task_id,
        hf_prefix=args.hf_prefix,
        run_name_prefix=args.run_name_prefix,
    )

    # ── Smoke gate (plan §7 gate 1). ─────────────────────────────────────────
    if args.smoke:
        gate = _smoke_gate(
            args.slab_root,
            args.runs_root,
            smoke_cell=args.smoke_cell,
            smoke_seed=args.smoke_seed,
        )
        _write_sentinel(
            LOG_DIR / f"issue-{args.sentinel_task_id}-smoke-results.json",
            kind="epm:progress",
            phase="smoke_gate",
            note_payload=gate,
            task_id=args.sentinel_task_id,
        )
        log.info("[phase=smoke_gate] %s", json.dumps(gate)[:800])
        if not gate["smoke_gate_pass"]:
            log.error("[phase=smoke_gate] FAIL — see checks above; sweep launch stays gated.")
            return 2
        # Upload the smoke cell's raw completions too (idempotent re-upload at
        # sweep end is harmless — same path, same content).
        if not args.skip_upload:
            from explore_persona_space.orchestrate.hub import (
                upload_raw_completions_to_data_repo,
            )

            upload_raw_completions_to_data_repo(
                experiment_name=hf_data_prefix, eval_results_dir=args.slab_root
            )
        log.info("[phase=smoke_done] smoke gate PASS; sweep is unlocked.")
        return 0

    # ── Raw-completions upload (Upload Policy: before pod termination). ──────
    if not args.skip_upload:
        from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

        log.info(
            "[phase=upload_raw] uploading raw completions → %s/%s", HF_DATA_REPO, hf_data_prefix
        )
        upload_raw_completions_to_data_repo(
            experiment_name=hf_data_prefix, eval_results_dir=args.slab_root
        )
    else:
        log.info("[phase=upload_raw] SKIP (--skip-upload)")

    # ── Final sentinel + terminal phase line (full sweep ONLY). ──────────────
    # reproducibility_card: workflow.yaml § markers epm:results contract —
    # per-cell adapter_paths AND the MANDATORY wandb declaration
    # (wandb_project + wandb_run_names) so verify_uploads.py resolves the
    # hf_model / wandb_run rows mechanically (#608 follow-up).
    note_payload = {
        "issue": args.sentinel_task_id,
        "status": "done",
        "seeds": seeds,
        "cells_requested": [c.slug for c in cells],
        "n_units_completed": len(
            [c for c in cell_results if c.get("status") in ("done", "resumed_skip")]
        ),
        "n_units": len(units),
        "cell_results": cell_results,
        "reproducibility_card": {
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "hf_model_repo": HF_MODEL_REPO,
            "hf_data_repo": HF_DATA_REPO,
            "adapter_paths": {
                f"{c['cell']}_seed{c['seed']}": c.get("adapter_hf_path")
                for c in cell_results
                if "adapter_hf_path" in c
            },
            "wandb_project": os.environ.get("WANDB_PROJECT") or f"issue{args.sentinel_task_id}",
            "wandb_run_names": [
                f"{args.run_name_prefix}_{c['cell']}_seed{c['seed']}"
                for c in cell_results
                if c.get("status") in ("done", "resumed_skip")
            ],
        },
        "worktree_path": str(Path.cwd()),
        "final_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    # Round-2 binding fix resumed-smoke-adapter-path-missing: the card must
    # carry EXACTLY one adapter path per scheduled unit — a shorter card means
    # some unit (e.g. the --resume-skipped smoke unit) silently dropped out of
    # the reproducibility contract. Fail BEFORE the sentinel lands so the
    # poller never drains an incomplete card.
    n_adapter_paths = len(note_payload["reproducibility_card"]["adapter_paths"])
    if n_adapter_paths != len(units):
        raise RuntimeError(
            f"reproducibility_card.adapter_paths has {n_adapter_paths} entries for "
            f"{len(units)} scheduled units — a unit's adapter path went missing from the "
            f"final card (resumed-skip records must carry adapter_hf_path); refusing to "
            f"write an incomplete epm:results sentinel."
        )
    _write_sentinel(
        LOG_DIR / sentinel_name,
        kind="epm:results",
        phase="done",
        note_payload=note_payload,
        task_id=args.sentinel_task_id,
    )
    log.info("Dispatcher done. %d cell units completed.", len(cell_results))
    log.info("[phase=done] dispatcher exit %s", datetime.now(UTC).isoformat())
    return 0


if __name__ == "__main__":
    sys.exit(main())
