#!/usr/bin/env python
"""Issue #825 ``real-user-turn-null`` — binding post-upload gates + sentinel writer.

The dispatch wrapper (scripts/issue825_realuser_dispatch.sh) calls this AFTER
UPLOAD-2 (plan MF-C: uploads precede ALL binding gates; every FAILURE path is
upload-then-exit). Gate logic lives in python — not bash — so the committed
tests (tests/test_issue825_realuser_gates.py) can exercise every FAILURE
sentinel path (the smoke bypasses numeric gates under EPS_SMOKE=1, so those
tests are the gates' only executable coverage — plan hard-req 7).

Subcommands:
  gates            evaluate the plan §7 binding gates in order — deferred fit
                   failures (rglob under the fit out-dir, r3 lesson), ingest
                   >= n floor, anchor ridge R2@L19 within ±0.05 of the parent
                   committed value, wiring own<shuffled NLL, coverage
                   8 ridge + 8 MLP + 1 anchor + headline. Writes
                   gate_outcomes.json; on the FIRST failure writes the FAILURE
                   sentinel and exits 1. Numeric gates are BYPASSED under
                   EPS_SMOKE=1 (structural asserts still binding, plan MF-D);
                   each gate logs one "gate armed: <name>" line.
  fail-from-ingest upload-then-exit for a crashed ingest phase (plan §4.3
                   hard-req 3): FIRST upload whatever the ingest produced
                   (dataset + meta + ingest_failure.json — a shortfall
                   writes all three before returning 1; text/JSON uploads
                   unconditional), THEN write the FAILURE sentinel,
                   routing on the ARTIFACT (ingest_failure.json ->
                   status ingest_shortfall) with exit-code fall-through
                   (status ingest_error, upload skipped only when NOTHING
                   was produced).
  success-sentinel write the schema-enveloped SUCCESS sentinel (refuses unless
                   gate_outcomes.json exists with all_pass=true — the gates
                   fire BEFORE any success path, by construction).

Sentinel contract: poll_pipeline._SENTINEL_REQUIRED_KEYS
(sentinel_schema_version / kind / version), body under "note".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

FOLLOWUP_LABEL = "real-user-turn-null"
ANCHOR_CELL = "M_instruct_assistant_chat"
ANCHOR_TOL = 0.05
USER_CELLS = [
    "M_instruct_user_chat",
    "M_instruct_user_naturalistic",
    "M_pretrained_user_chat",
    "M_pretrained_user_naturalistic",
]
ASSISTANT_CELLS = [
    "M_instruct_assistant_chat",
    "M_instruct_assistant_naturalistic",
    "M_pretrained_assistant_chat",
    "M_pretrained_assistant_naturalistic",
]
CELLS8 = ASSISTANT_CELLS + USER_CELLS
WIRING_MODELS = ("instruct", "pretrained")
WIRING_FORMATS = ("chat", "naturalistic")


def eps_smoke() -> bool:
    """Strict '1' comparison (fit_cells._eps_smoke convention)."""
    return os.environ.get("EPS_SMOKE") == "1"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _l19(payload: dict | None) -> float | None:
    """Ridge R2@L19 via the r3 gate-path lesson: frozen_layer_table is a dict
    keyed by the layer STRING."""
    if not payload:
        return None
    table = (payload.get("selection_symmetric") or {}).get("frozen_layer_table") or {}
    entry = table.get("19")
    return float(entry["r2_obs"]) if entry else None


class GateFailure(Exception):
    """One binding gate failed: carries the sentinel status + message."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def write_sentinel(sentinel: Path, status: str, note: dict) -> None:
    """Schema-enveloped sentinel write (poll_pipeline required keys)."""
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:results",
                "version": 1,
                "task_id": 825,
                "status": status,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": note,
            },
            indent=2,
            default=float,
        )
    )
    print(f"[gates] sentinel written: {sentinel} (status={status})")


# ---------------------------------------------------------------------------
# Individual gates — pure(ish) functions returning None on PASS, raising
# GateFailure on a binding miss. Each takes explicit inputs for testability.
# ---------------------------------------------------------------------------


def sweep_deferred_failures(out_dir: Path) -> dict:
    """rglob EVERY fit_failures.json under the fit out-dir (r3 lesson: the
    anchor_parent subdir invocation must not escape the sweep). Binding in
    smoke too (structural)."""
    deferred = {
        str(p.relative_to(out_dir)): json.loads(p.read_text())
        for p in sorted(out_dir.rglob("fit_failures.json"))
    }
    if deferred:
        n_fail = sum(len(v) for v in deferred.values())
        for rel, entries in deferred.items():
            print(f"[gates] deferred fit failures ({rel}): {entries}")
        raise GateFailure(
            "fit_deferred_failure",
            f"{n_fail} fit-phase failure(s) across {sorted(deferred)} were deferred "
            "past UPLOAD-2 (see the [phase=fit*] log tracebacks)",
        )
    return deferred


def check_ingest(meta: dict | None, n_target: int, smoke: bool) -> dict:
    """Gate 1 (plan §7): ingest kept >= n_target. Meta PRESENCE binds in smoke;
    the numeric floor is production-only (a smoke ingest keeps ~8 rows)."""
    if meta is None:
        raise GateFailure("ingest_shortfall", "conversations_real2turn_meta.json missing")
    n_kept = meta.get("n_kept")
    if smoke:
        return {"result": "BYPASSED_SMOKE_PRESENCE_ONLY", "n_kept": n_kept}
    if not isinstance(n_kept, int) or n_kept < n_target:
        raise GateFailure(
            "ingest_shortfall",
            f"ingest kept {n_kept} < floor {n_target} — never run underpowered",
        )
    return {"result": "PASS", "n_kept": n_kept}


def check_anchor(
    fresh_payload: dict | None, parent_payload: dict | None, smoke: bool, tol: float = ANCHOR_TOL
) -> dict:
    """Gate 2 (plan §7): end-to-end parent anchor, ridge R2@L19 within ±tol of
    the committed parent value (read from the parent eval JSON at run time)."""
    if fresh_payload is None:
        raise GateFailure(
            "coverage_miss", f"anchor cell missing: anchor_parent/cells_{ANCHOR_CELL}.json"
        )
    if smoke:
        return {"result": "BYPASSED_SMOKE_PRESENCE_ONLY"}
    fresh = _l19(fresh_payload)
    parent = _l19(parent_payload)
    if fresh is None or parent is None or math.isnan(fresh) or math.isnan(parent):
        raise GateFailure(
            "anchor_gate_miss",
            f"missing/NaN L19 read (fresh={fresh}, parent={parent}) — cannot certify the rig",
        )
    delta = fresh - parent
    if abs(delta) > tol:
        raise GateFailure(
            "anchor_gate_miss",
            f"anchor fresh {fresh:+.4f} vs parent committed {parent:+.4f} "
            f"(|delta|={abs(delta):.4f} > {tol}) — rig drift, HALT",
        )
    return {"result": "PASS", "fresh": fresh, "parent": parent, "delta": delta, "tol": tol}


def check_wiring(wiring_by_model: dict[str, dict | None], smoke: bool) -> dict:
    """Gate 3 (plan §7 / MF-B): mean own-context NLL < mean shuffled-context NLL
    per (model, format) read. File PRESENCE binds in smoke; the numeric margin
    is production-only. Gross failure only (own >= shuffled, missing or NaN
    reads) — real-vs-Haiku/self NLL orderings are diagnostics, never gates.
    Evaluates ALL reads first, prints every mean, THEN halts once."""
    values: dict[str, dict] = {}
    for model in WIRING_MODELS:
        w = wiring_by_model.get(model)
        if w is None:
            raise GateFailure("wiring_check_fail", f"missing wiring-check output for {model}")
        for fmt in WIRING_FORMATS:
            blk = (w.get("per_format") or {}).get(fmt)
            if blk is None:
                raise GateFailure("wiring_check_fail", f"missing wiring read {model}/{fmt}")
            values[f"{model}/{fmt}"] = {
                "own_mean_nll": blk.get("own_mean_nll"),
                "shuffled_mean_nll": blk.get("shuffled_mean_nll"),
                "n": blk.get("n"),
            }
    if smoke:
        return {"result": "BYPASSED_SMOKE_PRESENCE_ONLY", "values": values}
    bad = []
    for key, v in values.items():
        own, shuf = v["own_mean_nll"], v["shuffled_mean_nll"]
        print(f"[gates] wiring-check {key}: own={own} shuffled={shuf} n={v['n']}")
        if (
            own is None
            or shuf is None
            or (isinstance(own, float) and math.isnan(own))
            or (isinstance(shuf, float) and math.isnan(shuf))
            or own >= shuf
        ):
            bad.append(f"{key}: own={own} shuffled={shuf}")
    if bad:
        raise GateFailure(
            "wiring_check_fail",
            "own >= shuffled or missing/NaN reads in: " + "; ".join(bad),
        )
    return {"result": "PASS", "values": values}


def check_coverage(out_dir: Path, anchor_dir: Path, n_expected: int | None, smoke: bool) -> dict:
    """Gate 4 (plan §7): 8/8 real ridge cells + 8/8 MLP blocks + 1/1 anchor +
    headline_metrics.json on disk (explicit --cells disables the fit script's
    FATAL-missing-bundle branch, so absence is only caught HERE). Structural —
    binds in smoke too. Production additionally asserts per-cell fit n ==
    n_expected (extraction/ingest row parity)."""
    for cid in CELLS8:
        p = out_dir / f"cells_{cid}.json"
        payload = _load(p)
        if payload is None:
            raise GateFailure("coverage_miss", f"missing ridge results {p}")
        if not payload.get("mlp"):
            raise GateFailure("coverage_miss", f"{cid}: no non-empty 'mlp' block on disk")
        n_fit = (payload.get("metadata") or {}).get("n")
        if not smoke and n_expected is not None and n_fit != n_expected:
            raise GateFailure(
                "coverage_miss",
                f"{cid}: fit n={n_fit} != ingested {n_expected} — extraction/ingest row drift",
            )
    if not (anchor_dir / f"cells_{ANCHOR_CELL}.json").exists():
        raise GateFailure(
            "coverage_miss", f"missing anchor {anchor_dir / f'cells_{ANCHOR_CELL}.json'}"
        )
    if not (out_dir / "headline_metrics.json").exists():
        raise GateFailure("coverage_miss", "headline_metrics.json missing")
    return {"result": "PASS", "cells": len(CELLS8), "anchor": 1}


def run_gates(
    *,
    out_dir: Path,
    anchor_dir: Path,
    realuser_dir: Path,
    wiring_dir: Path,
    parent_cells_dir: Path,
    sentinel: Path,
    n_target: int,
    smoke: bool,
) -> dict:
    """Evaluate the binding gates in plan §7 order; upload-then-exit on failure.

    Writes gate_outcomes.json into out_dir on BOTH outcomes. On failure the
    FAILURE sentinel is written BEFORE the raise — the gates fire before any
    SUCCESS path by construction (success-sentinel refuses without all_pass).
    """
    outcomes: dict = {"smoke": smoke, "gates": {}}
    try:
        print("gate armed: deferred-fit-failures")
        sweep_deferred_failures(out_dir)
        outcomes["gates"]["deferred_fit_failures"] = "PASS"
        print("gate: deferred-fit-failures PASS (none recorded under any fit out-dir)")

        print("gate armed: ingest-floor")
        meta = _load(realuser_dir / "conversations_real2turn_meta.json")
        outcomes["gates"]["ingest_floor"] = check_ingest(meta, n_target, smoke)
        print(f"gate: ingest-floor {outcomes['gates']['ingest_floor']}")

        print("gate armed: anchor-ridge-tolerance")
        fresh = _load(anchor_dir / f"cells_{ANCHOR_CELL}.json")
        parent = _load(parent_cells_dir / f"cells_{ANCHOR_CELL}.json")
        outcomes["gates"]["anchor_ridge_tolerance"] = check_anchor(fresh, parent, smoke)
        print(f"gate: anchor-ridge-tolerance {outcomes['gates']['anchor_ridge_tolerance']}")

        print("gate armed: wiring-check")
        wiring = {m: _load(wiring_dir / f"wiring_check_{m}.json") for m in WIRING_MODELS}
        outcomes["gates"]["wiring_check"] = check_wiring(wiring, smoke)
        print(f"gate: wiring-check {outcomes['gates']['wiring_check']['result']}")

        print("gate armed: coverage")
        n_expected = (meta or {}).get("n_kept")
        outcomes["gates"]["coverage"] = check_coverage(out_dir, anchor_dir, n_expected, smoke)
        print("gate: coverage PASS (8 ridge + 8 MLP + 1 anchor + headline)")

        # Non-halting diagnostics: g3 sanity values recorded by fit_cells in
        # BOTH out-dirs. NOT a binding gate this round (plan §7 lists exactly
        # 4 binding gates; the logged-a1 assistant map may legitimately be
        # weak on real conversations — that is a science read, not rig drift).
        outcomes["g3_recorded_diagnostic"] = {
            "real": _load(out_dir / "g3_gate.json"),
            "anchor_parent": _load(anchor_dir / "g3_gate.json"),
            "note": "recorded, non-halting (plan §7 binding gates: ingest/anchor/wiring/coverage)",
        }
        outcomes["all_pass"] = True
    except GateFailure as gf:
        outcomes["all_pass"] = False
        outcomes["failure"] = {"status": gf.status, "message": gf.message}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gate_outcomes.json").write_text(json.dumps(outcomes, indent=2, default=float))
        write_sentinel(
            sentinel,
            gf.status,
            {
                "followup_label": FOLLOWUP_LABEL,
                "failure": gf.message,
                "gate_outcomes": outcomes,
                "uploads_completed_before_gates": not smoke,
            },
        )
        raise SystemExit(f"GATE FAIL [{gf.status}]: {gf.message}") from gf
    (out_dir / "gate_outcomes.json").write_text(json.dumps(outcomes, indent=2, default=float))
    print("gate: ALL PASS" + (" [smoke: numeric gates bypassed]" if smoke else ""))
    return outcomes


# ---------------------------------------------------------------------------
# Sentinel subcommands
# ---------------------------------------------------------------------------


# The ingest phase's produced text/JSON artifacts (plan §4.3 hard-req 3: every
# FAILURE path is upload-then-exit; a shortfall writes ALL THREE before the
# ingest returns 1 — issue825_realuser_ingest.main). Exact names double as
# upload_folder allow_patterns (globs); a true crash may leave any subset.
INGEST_UPLOAD_ARTIFACTS = (
    "conversations_real2turn.jsonl",
    "conversations_real2turn_meta.json",
    "ingest_failure.json",
)


def upload_ingest_artifacts(realuser_dir: Path, smoke: bool) -> list[str]:
    """Upload-then-exit for the ingest FAILURE path (plan §4.3 hard-req 3 /
    MF-C: text/JSON uploads are unconditional on every failure path).

    Pushes whatever the ingest phase produced to the data repo's
    ``raw_completions/ingestion`` prefix (the SAME prefix UPLOAD-1 targets on
    success) BEFORE the FAILURE sentinel is written, so a shortfall never
    strands the ingested dataset + meta + failure artifact on a torn-down
    worker. Under EPS_SMOKE=1 this is a structural listing assert (no real
    upload), matching the wrapper's smoke upload convention. Returns the
    uploaded (or would-upload) filenames; empty for a true crash that
    produced nothing (the ingest_error fall-through) — then no HF call is
    made. Upload failures raise (fail-loud): a dead worker with no sentinel
    is the poller's crash signal, never a silently-skipped upload.
    """
    present = [name for name in INGEST_UPLOAD_ARTIFACTS if (realuser_dir / name).exists()]
    if not present:
        print("[gates] fail-from-ingest: no ingest artifacts on disk — nothing to upload")
        return present
    if smoke:
        print(f"[gates] [smoke] fail-from-ingest upload structural assert PASS: {present}")
        return present
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (source .env before upload)"
    from huggingface_hub import HfApi

    repo = os.environ.get("EPS_DATA_REPO", "superkaiba1/explore-persona-space-data")
    prefix = os.environ.get("EPS_HF_RU_PREFIX", "issue825_real_user_turn_null")
    signal.alarm(2700)  # 45-min per-stage hard cap (wrapper convention, plan §10)
    try:
        HfApi().upload_folder(
            folder_path=str(realuser_dir),
            repo_id=repo,
            repo_type="dataset",
            path_in_repo=f"{prefix}/raw_completions/ingestion",
            allow_patterns=list(INGEST_UPLOAD_ARTIFACTS),
            commit_message=(
                "issue-825 real-user-turn-null: ingest FAILURE path upload-then-exit "
                "(produced dataset/meta + ingest_failure.json, BEFORE the sentinel)"
            ),
        )
    finally:
        signal.alarm(0)
    print(f"[gates] fail-from-ingest upload: ok ({present} -> {prefix}/raw_completions/ingestion)")
    return present


def fail_from_ingest(realuser_dir: Path, sentinel: Path) -> None:
    """FAILURE sentinel for a crashed ingest phase — upload-then-exit (plan
    §4.3 hard-req 3): FIRST upload whatever the ingest produced, THEN route
    on the ARTIFACT (ingest_failure.json); exit-code fall-through is
    ingest_error."""
    uploaded = upload_ingest_artifacts(realuser_dir, eps_smoke())
    artifact = _load(realuser_dir / "ingest_failure.json")
    if artifact and artifact.get("status"):
        status = str(artifact["status"])
        note = {"followup_label": FOLLOWUP_LABEL, "ingest_failure": artifact}
    else:
        status = "ingest_error"
        note = {
            "followup_label": FOLLOWUP_LABEL,
            "failure": "ingest phase exited non-zero with no ingest_failure.json artifact "
            "(see the [phase=ingest] log traceback)",
        }
    note["uploaded_before_sentinel"] = uploaded
    write_sentinel(sentinel, status, note)


def success_sentinel(out_dir: Path, sentinel: Path) -> None:
    """SUCCESS sentinel: refuses unless gate_outcomes.json shows all_pass."""
    gates = _load(out_dir / "gate_outcomes.json")
    assert gates is not None, "gate_outcomes.json missing — gates must run before success"
    assert gates.get("all_pass") is True, (
        f"gate_outcomes.json is not all_pass — refusing SUCCESS sentinel: {gates.get('failure')}"
    )
    headline = _load(out_dir / "headline_metrics.json")
    assert headline is not None, "headline_metrics.json missing"
    repo = os.environ.get("EPS_DATA_REPO", "superkaiba1/explore-persona-space-data")
    prefix = os.environ.get("EPS_HF_RU_PREFIX", "issue825_real_user_turn_null")
    t0 = float(os.environ.get("EPS_T0", time.time()))
    write_sentinel(
        sentinel,
        "success",
        {
            "followup_label": FOLLOWUP_LABEL,
            "eval_numbers": headline,
            "gate_outcomes": gates,
            "eval_paths": sorted(str(p) for p in out_dir.rglob("*.json")),
            "reproducibility_card": {
                "models": ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"],
                "fit_seed": 0,
                "lmsys_revision": os.environ.get("EPS_LMSYS_REV"),
                "parent_kept2000_revision": os.environ.get("EPS_HF_REV"),
                "followup_label": FOLLOWUP_LABEL,
            },
            "wandb_url": "n/a (analysis-only follow-up; no training)",
            "hf_hub_url": f"https://huggingface.co/datasets/{repo}/tree/main/{prefix}",
            "worktree_path": os.environ.get("EPS_WORKTREE", str(Path.cwd())),
            "final_commit_sha": os.environ.get("EPS_GIT_SHA", "unknown"),
            "gpu_hours_used": round((time.time() - t0) / 3600.0, 3),
            "gpu_hours_used_basis": "measured wrapper wall-clock (single-GPU provision)",
            "gpu_hours_budgeted": 3.0,
            "plan_deviations": [],
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("gates", "fail-from-ingest", "success-sentinel"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--anchor-dir", type=Path, default=None, help="default <out-dir>/anchor_parent")
    ap.add_argument("--realuser-dir", type=Path, default=Path("data/issue_825/realuser"))
    ap.add_argument("--wiring-dir", type=Path, default=Path("data/issue_825/realuser_wiring"))
    ap.add_argument("--parent-cells-dir", type=Path, default=Path("eval_results/issue_825"))
    ap.add_argument("--sentinel", type=Path, required=True)
    ap.add_argument("--n-target", type=int, default=2000)
    args = ap.parse_args()
    anchor_dir = args.anchor_dir or (args.out_dir / "anchor_parent")
    if args.mode == "fail-from-ingest":
        fail_from_ingest(args.realuser_dir, args.sentinel)
        return 0
    if args.mode == "success-sentinel":
        success_sentinel(args.out_dir, args.sentinel)
        return 0
    run_gates(
        out_dir=args.out_dir,
        anchor_dir=anchor_dir,
        realuser_dir=args.realuser_dir,
        wiring_dir=args.wiring_dir,
        parent_cells_dir=args.parent_cells_dir,
        sentinel=args.sentinel,
        n_target=args.n_target,
        smoke=eps_smoke(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
