#!/usr/bin/env python3
# math/scientific notation in docstrings + messages
"""Issue #667 tf-margin dispatcher — 2-index gate->behavior bridge on the NEW DV.

PASS_UNIFIED architectural parity (Step 6d.0): smoke IS the sweep with one cell.
EVERY phase derives its cell subset from the SAME ``--behaviors`` / ``--sources``
/ ``--targets`` / ``--cap`` filters, so a smoke is the production sweep scaled
down — ONE code path, ONE ``_compute_wave_size`` call, ONE subprocess shape, ONE
sentinel writer. The phase list is identical; only the cell COUNT differs.

Phases (plan v6 §4.3 DAG):

- ``apply-parity-probe`` (Phase-0.5, GPU, ~5 min, MUST-PASS FIRST): the
  current-stack rsLoRA apply-parity probe (scripts/issue667_tf_margin_apply_parity.py)
  reproduces #537's committed on-diagonal E_plus_source within +/-0.10. On FAIL
  the round HALTs (epm:failure infra rsLoRA_apply_parity_drift) — do NOT sweep.
- ``build-fact-pool`` (GPU gen + judge): the NEW fact fixed +/- pool
  (scripts/issue667_build_fact_pool.py). Fires only when ``fact`` in --behaviors.
- ``extract`` (GPU, ~4.5 GPU-h full / ~min smoke): per-source-adapter tf-margin
  2-index forward pass (scripts/issue667_tf_margin_extract.py) as a CVD-pinned
  subprocess per cell, in waves of ``min(len(cells), visible_gpus)``. Writes
  per-cell JSON under eval_results/issue_667/tf_margin/per_cell/ and uploads them
  + the fact pool to the HF data repo before pod terminate.
- ``analysis`` (CPU, off-pod for the full sweep / on-pod for the smoke): the
  gate->tf-margin join + both gates (scripts/issue667_tf_margin_analysis.py).

Pod-side contract (CLAUDE.md / poll_pipeline.py): ``[phase=<name>]`` lines, a
terminal ``[phase=done]`` (RESERVED — never on per-cell echoes), and an
end-of-run sentinel JSON at /workspace/logs/issue-667-<kind_slug>-<epoch>.json
carrying _SENTINEL_REQUIRED_KEYS (sentinel_schema_version / kind / version).

Per-GPU fan-out pins ``CUDA_VISIBLE_DEVICES=<gpu>`` in the LAUNCHER env per cell
(+ matching ``--gpu-id``) so an import-time cuInit can't co-locate cells on GPU 0
(#545). Every subprocess gets an explicit ``env={**os.environ}``; load_dotenv()
at main()-top puts HF_TOKEN/WANDB_API_KEY/ANTHROPIC_API_KEY in os.environ first
(#397). wave_size = min(len(cells), torch.cuda.device_count()) — derived from the
DETECTED count, NOT --n-gpus alone (the a36-round wave=4-on-visible-8 bug fix).

Launch (plan v6 §10)::

    uv run python scripts/dispatch_issue.py launch --issue 667 --backend runpod \\
        --gpu-type H100 --gpu-count 8 --repo-branch issue-667 --workload-cmd \\
        'uv run python scripts/issue667_tf_margin_dispatch.py all \\
         --behaviors em,sycophancy,fact --layer 14 --n-gpus 8'

Smoke (the unified single-cell sweep, CPU)::

    uv run python scripts/issue667_tf_margin_dispatch.py all \\
        --behaviors em --sources default --targets default,fmt_json,sp_swe \\
        --layer 14 --cap 4 --cpu-only --smoke
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("issue667_tf_margin_dispatch")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_TF_MARGIN_PREFIX = "issue667_gate_chain_preview/tf_margin"
PER_CELL_DIR = "eval_results/issue_667/tf_margin/per_cell"
OUT_DIR = "eval_results/issue_667/tf_margin"
TENSORS_DIR = "eval_results/issue_667/analysis_tensors"
# CONCERN 1 (round 2): the fact-pool builder writes this sentinel under the fact
# pool dir when floor-N < YIELD_FLOOR_MIN. Its presence == fact dropped from the
# headline: the extract phase SKIPS fact source cells and the analysis phase
# SOFT-DROPS fact (§4.3). Kept in lockstep with issue667_build_fact_pool.py.
FACT_POOL_DIR = "data/issue_667/fact_fixed_pool_v1"
FACT_DROP_SENTINEL = f"{FACT_POOL_DIR}/DROPPED_FROM_HEADLINE.sentinel"
_SEED = 42


def fact_dropped_from_headline() -> bool:
    """True iff the fact-pool builder wrote the DROP sentinel (floor-N < min).

    Presence of the sentinel is the single source of truth: the extract phase
    skips fact cells and the analysis phase soft-drops fact when it is True.
    """
    return (PROJECT_ROOT / FACT_DROP_SENTINEL).is_file()


# ─────────────────────────────────────────────────────────────────────────────
# Log dir + phase lines + sentinel  (mirrors issue667_dispatch.py)
# ─────────────────────────────────────────────────────────────────────────────


def _log_dir() -> Path:
    override = os.environ.get("EPM_LOG_DIR")
    if override:
        d = Path(override)
    else:
        d = Path("/workspace/logs")
        if not d.exists():
            d = PROJECT_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def phase_log(name: str) -> None:
    """Emit the ``[phase=<name>]`` line poll_pipeline.py parses (PHASE_RE)."""
    print(f"[phase={name}]", flush=True)


def write_sentinel(kind: str, note: str, *, version: int = 1, extra: dict | None = None) -> Path:
    """End-of-run sentinel with poll_pipeline's _SENTINEL_REQUIRED_KEYS."""
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "task_id": 667,
        "by": "issue667_tf_margin_dispatch",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    if extra:
        payload.update(extra)
    slug = kind.replace(":", "_")
    out = _log_dir() / f"issue-667-{slug}-{time.time_ns()}.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("sentinel written: %s", out)
    return out


def _require_credentials() -> None:
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"
    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY missing -- .env?"


# ─────────────────────────────────────────────────────────────────────────────
# Wave sizing — DETECTED GPU count, not --n-gpus alone (a36-round bug fix)
# ─────────────────────────────────────────────────────────────────────────────


def _compute_wave_size(cpu_only: bool, requested_n_gpus: int) -> int:
    """wave_size = 1 on CPU; min(detected device_count, max(requested, 1)) on GPU.

    RAISES LOUD on 0 visible GPU when not cpu_only (the surplus-lane-on-CPU trap:
    a hardcoded/`--n-gpus`-default wave > detected count sends surplus --gpu-id
    lanes to CVD indices with no device, silently running on CPU for hours). The
    DETECTED count is the source of truth; --n-gpus is a CEILING only.
    """
    if cpu_only:
        return 1
    import torch

    visible = torch.cuda.device_count()
    if visible <= 0:
        raise RuntimeError(
            "0 visible CUDA devices but --cpu-only not set -- refusing to fan out "
            "surplus --gpu-id lanes onto CPU (a36-round wave-size bug)."
        )
    return min(visible, max(requested_n_gpus, 1))


def _run_with_log(
    cmd: Sequence[str], *, log_path: Path, extra_env: dict[str, str] | None = None
) -> int:
    """Run a child process, tee stdout/stderr to a log file. Returns rc. Explicit env (#397)."""
    env = {**os.environ}
    if extra_env:
        env.update(extra_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("$ %s  >>> %s", " ".join(shlex.quote(c) for c in cmd), log_path)
    with log_path.open("ab") as f:
        proc = subprocess.run(
            list(cmd), stdout=f, stderr=subprocess.STDOUT, check=False, env=env, cwd=PROJECT_ROOT
        )
    if proc.returncode != 0:
        logger.error("command exited rc=%d (log: %s)", proc.returncode, log_path)
    return proc.returncode


def _run_parallel_with_log(
    cmds: Iterable[tuple[Sequence[str], Path, dict[str, str] | None]],
) -> list[int]:
    """Run several subprocesses concurrently (wave). Returns rc list. Per-cell env pinned (#545)."""
    procs: list[subprocess.Popen] = []
    files = []
    for cmd, log_path, extra_env in cmds:
        env = {**os.environ}
        if extra_env:
            env.update(extra_env)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = log_path.open("ab")
        files.append(f)
        logger.info("$ (parallel) %s  >>> %s", " ".join(shlex.quote(c) for c in cmd), log_path)
        procs.append(
            subprocess.Popen(
                list(cmd), stdout=f, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT
            )
        )
    rcs = [p.wait() for p in procs]
    for f in files:
        f.close()
    return rcs


# ─────────────────────────────────────────────────────────────────────────────
# Cell selection — the SAME filters parameterize EVERY phase (PASS_UNIFIED)
# ─────────────────────────────────────────────────────────────────────────────


def select_sources(behavior: str, sources_arg: str | None) -> list[str]:
    from explore_persona_space.experiments.i537_contexts import train_cids_for

    full = train_cids_for(behavior)
    if not sources_arg:
        return full
    requested = [s.strip() for s in sources_arg.split(",") if s.strip()]
    unknown = [s for s in requested if s not in full]
    if unknown:
        raise ValueError(f"--sources {unknown!r} not in the {behavior} train grid {full}")
    return requested


def select_targets(behavior: str, targets_arg: str | None) -> list[str] | None:
    if not targets_arg:
        return None  # extractor defaults to eval_cids_for(behavior)
    from explore_persona_space.experiments.i537_contexts import eval_cids_for

    full = set(eval_cids_for(behavior))
    requested = [t.strip() for t in targets_arg.split(",") if t.strip()]
    unknown = [t for t in requested if t not in full and t not in select_sources(behavior, None)]
    if unknown:
        raise ValueError(f"--targets {unknown!r} not in the {behavior} eval grid")
    return requested


# ─────────────────────────────────────────────────────────────────────────────
# Phase: apply-parity-probe (Phase-0.5, MUST-PASS FIRST)
# ─────────────────────────────────────────────────────────────────────────────


def phase_apply_parity(*, cpu_only: bool, n_samples: int, dry_run: bool) -> None:
    """rsLoRA apply-parity probe (plan §4.4b). HALT on drift -> failure sentinel."""
    phase_log("apply_parity")
    if dry_run:
        logger.info("[dry-run] apply-parity-probe skipped")
        logger.info("[phase=apply_parity_done]")
        return
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue667_tf_margin_apply_parity.py",
        "--n-samples",
        str(n_samples),
        "--out",
        f"{OUT_DIR}/apply_parity_probe.json",
    ]
    if cpu_only:
        cmd.append("--cpu-only")
    rc = _run_with_log(cmd, log_path=_log_dir() / "apply_parity.log")
    if rc != 0:
        # HALT: write the failure sentinel the poller drains -> orchestrator posts
        # epm:failure infra rsLoRA_apply_parity_drift. Do NOT proceed to the sweep.
        write_sentinel(
            "epm:failure",
            "rsLoRA_apply_parity_drift: apply-parity probe FAILED (see apply_parity_probe.json)",
            extra={
                "failure_class": "infra",
                "reason": "rsLoRA_apply_parity_drift",
                "gate": "phase0.5",
            },
        )
        _write_failure_sentinel_file()
        raise RuntimeError(
            "apply-parity probe FAILED (rsLoRA_apply_parity_drift) -- HALT, do NOT sweep."
        )
    logger.info("[phase=apply_parity_done]")


def _write_failure_sentinel_file() -> None:
    """Also write the plan-named /workspace/logs/issue-667-apply-parity-failure.json marker."""
    p = _log_dir() / "issue-667-apply-parity-failure.json"
    probe = PROJECT_ROOT / OUT_DIR / "apply_parity_probe.json"
    payload = {"reason": "rsLoRA_apply_parity_drift", "probe": str(probe)}
    if probe.exists():
        with contextlib.suppress(Exception):
            payload["probe_result"] = json.loads(probe.read_text())
    p.write_text(json.dumps(payload, indent=2))
    logger.info("apply-parity failure marker written: %s", p)


# ─────────────────────────────────────────────────────────────────────────────
# Phase: build-fact-pool (GPU gen + judge; only when fact in --behaviors)
# ─────────────────────────────────────────────────────────────────────────────


def phase_build_fact_pool(
    *,
    behaviors: list[str],
    cpu_only: bool,
    cap: int,
    max_probes: int | None,
    skip_upload: bool,
    dry_run: bool,
) -> None:
    """Build + upload the fact fixed +/- pool (plan §4.3). No-op if fact not requested."""
    phase_log("build_fact_pool")
    if "fact" not in behaviors:
        logger.info("fact not in --behaviors -> skip fact-pool build")
        logger.info("[phase=build_fact_pool_done]")
        return
    if dry_run:
        logger.info("[dry-run] build-fact-pool skipped")
        logger.info("[phase=build_fact_pool_done]")
        return
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue667_build_fact_pool.py",
        "--cap",
        str(cap),
    ]
    if cpu_only:
        cmd += ["--cpu-only", "--n-rollouts", "1"]
    if max_probes:
        cmd += ["--max-probes", str(max_probes)]
    if skip_upload:
        cmd.append("--skip-upload")
    rc = _run_with_log(cmd, log_path=_log_dir() / "build_fact_pool.log")
    if rc != 0:
        raise RuntimeError(
            f"build-fact-pool failed (rc={rc}); see {_log_dir() / 'build_fact_pool.log'}"
        )
    logger.info("[phase=build_fact_pool_done]")


# ─────────────────────────────────────────────────────────────────────────────
# Phase: extract (2-index tf-margin sweep, CVD-pinned waves)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_cmd(
    behavior: str, source: str, targets: list[str] | None, cap: int, gpu_id: int, cpu_only: bool
) -> tuple[list[str], Path, dict[str, str]]:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue667_tf_margin_extract.py",
        "--behavior",
        behavior,
        "--source-cid",
        source,
        "--seed",
        str(_SEED),
        "--cap",
        str(cap),
        "--out",
        PER_CELL_DIR,
        "--gpu-id",
        str(gpu_id),
    ]
    if targets:
        cmd += ["--targets", ",".join(targets)]
    if cpu_only:
        cmd.append("--cpu-only")
    # CVD pinned in the LAUNCHER env per cell (#545), matching --gpu-id.
    env = {"CUDA_VISIBLE_DEVICES": "" if cpu_only else str(gpu_id)}
    log_path = _log_dir() / f"tf_margin_extract_{behavior}_{source}.log"
    return cmd, log_path, env


def _cell_done(behavior: str, source: str) -> bool:
    """Resume-skip: the per-cell tf_margins.json already exists."""
    f = PROJECT_ROOT / PER_CELL_DIR / behavior / f"{source}_seed{_SEED}" / "tf_margins.json"
    return f.is_file()


def _select_extract_cells(behaviors: list[str], sources_arg: str | None) -> list[tuple[str, str]]:
    """(behavior, source) cells to extract, skipping fact if the DROP sentinel is present.

    CONCERN 1: if the fact pool under-yielded (DROP sentinel present), skip the
    fact source cells entirely — the fact arm is soft-dropped from the headline,
    so extracting its (empty-pool) margins would waste GPU and feed the analysis a
    fact vector the SOFT-DROP path discards anyway. em/syco are unaffected.
    """
    fact_dropped = "fact" in behaviors and fact_dropped_from_headline()
    if fact_dropped:
        logger.info(
            "fact-pool DROP sentinel present (%s) -> skipping fact source cells in extract "
            "(fact soft-dropped from the headline; em/syco unaffected)",
            FACT_DROP_SENTINEL,
        )
    cells: list[tuple[str, str]] = []
    for behavior in behaviors:
        if behavior == "fact" and fact_dropped:
            continue
        for source in select_sources(behavior, sources_arg):
            cells.append((behavior, source))
    return cells


def phase_extract(
    *,
    behaviors: list[str],
    sources_arg: str | None,
    targets_arg: str | None,
    cap: int,
    n_gpus: int,
    cpu_only: bool,
    skip_upload: bool,
    dry_run: bool,
    resume_skip: bool = True,
) -> None:
    """Per-source-adapter 2-index tf-margin extraction in CVD-pinned waves; upload after."""
    phase_log("extract")
    cells = _select_extract_cells(behaviors, sources_arg)
    logger.info("extract: %d source-adapter cells across behaviors=%s", len(cells), behaviors)
    if resume_skip and not dry_run:
        kept = [(b, s) for (b, s) in cells if not _cell_done(b, s)]
        if len(kept) != len(cells):
            logger.info(
                "resume-skip: kept %d / %d cells (skipped %d on disk)",
                len(kept),
                len(cells),
                len(cells) - len(kept),
            )
        cells = kept

    # A dry-run only builds + logs commands (no GPU work, no subprocess), so it
    # must not require visible GPUs — treat it as a single lane on a GPU-less VM.
    if dry_run:
        n_par = 1
    else:
        n_par = _compute_wave_size(cpu_only, n_gpus)
        # Regression guard (a36 wave-size bug): the fan-out width is derived from
        # the DETECTED device count (min(detected, --n-gpus)), NOT --n-gpus alone.
        # The per-wave slice `cells[start:start+n_par]` then naturally caps at the
        # number of remaining cells, so we assert the LANE WIDTH equals the
        # detected-derived value — never that it equals min(len(cells), detected)
        # (that conflates lane width with the last-wave cell count, wrong after
        # resume-skip).
        if not cpu_only:
            import torch

            assert n_par == min(torch.cuda.device_count(), max(n_gpus, 1)), (
                n_par,
                torch.cuda.device_count(),
                n_gpus,
            )

    for wave_start in range(0, len(cells), n_par):
        wave = cells[wave_start : wave_start + n_par]
        cmds = []
        for i, (behavior, source) in enumerate(wave):
            targets = select_targets(behavior, targets_arg)
            cmds.append(_extract_cmd(behavior, source, targets, cap, i % n_par, cpu_only))
        if dry_run:
            for (cmd, _lp, env), (behavior, source) in zip(cmds, wave, strict=True):
                logger.info(
                    "[dry-run] extract %s/%s CVD=%r :: %s",
                    behavior,
                    source,
                    env.get("CUDA_VISIBLE_DEVICES"),
                    " ".join(shlex.quote(c) for c in cmd),
                )
            continue
        rcs = _run_parallel_with_log(cmds)
        bad = [(rc, c) for rc, c in zip(rcs, wave, strict=True) if rc != 0]
        if bad:
            raise RuntimeError(f"extract wave failed: {bad}; see logs in {_log_dir()}")
        for behavior, source in wave:
            logger.info(
                "tf-margin extract cell %s/%s complete", behavior, source
            )  # NOT [phase=done]

    if dry_run:
        logger.info("[phase=extract_done] (dry-run: no tensors, upload skipped)")
        return
    if not skip_upload:
        _upload_per_cell()
    logger.info("[phase=extract_done]")


def _upload_per_cell() -> None:
    """Upload per-cell tf_margins.json to the HF data repo (one bulk commit, verified)."""
    if os.environ.get("EPM_SKIP_UPLOAD") == "1":
        logger.info("EPM_SKIP_UPLOAD=1 -> skipping per-cell upload (smoke/local)")
        return
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files

    root = PROJECT_ROOT / PER_CELL_DIR
    files = sorted(root.rglob("tf_margins.json"))
    if not files:
        raise RuntimeError(f"no tf_margins.json to upload under {root} -- extraction wrote nothing")
    api = HfApi()
    ops = [
        CommitOperationAdd(
            path_in_repo=f"{HF_TF_MARGIN_PREFIX}/per_cell/{p.relative_to(root).as_posix()}",
            path_or_fileobj=str(p),
        )
        for p in files
    ]
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue667: {len(ops)} per-cell tf-margin JSONs",
    )
    remote = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    missing = [
        p.relative_to(root).as_posix()
        for p in files
        if f"{HF_TF_MARGIN_PREFIX}/per_cell/{p.relative_to(root).as_posix()}" not in remote
    ]
    if missing:
        raise RuntimeError(f"tf-margin upload verification FAILED -- missing on Hub: {missing[:5]}")
    logger.info("uploaded + verified %d tf-margin cells to %s", len(files), HF_DATA_REPO)


# ─────────────────────────────────────────────────────────────────────────────
# Phase: analysis (CPU; gate->tf-margin join + gates)
# ─────────────────────────────────────────────────────────────────────────────


def phase_analysis(
    *, behaviors: list[str], layer: int, skip_store_pin: bool, dry_run: bool
) -> None:
    """gate->tf-margin join + g0-correctness + measurement-validity gates + headline."""
    phase_log("analysis")
    if dry_run:
        logger.info("[dry-run] analysis skipped")
        logger.info("[phase=analysis_done]")
        return
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue667_tf_margin_analysis.py",
        "--per-cell-dir",
        PER_CELL_DIR,
        "--tensors-dir",
        TENSORS_DIR,
        "--out-dir",
        OUT_DIR,
        "--behaviors",
        *behaviors,
        "--layer",
        str(layer),
    ]
    if skip_store_pin:
        cmd.append("--skip-store-pin")
    # CONCERN 1: forward the fact-drop signal so the analysis SOFT-DROPS fact
    # (the analysis ALSO reads the sentinel itself when run standalone off-pod).
    if "fact" in behaviors and fact_dropped_from_headline():
        cmd.append("--fact-dropped")
    rc = _run_with_log(cmd, log_path=_log_dir() / "tf_margin_analysis.log")
    if rc == 3:
        write_sentinel(
            "epm:failure",
            "g0_correctness_gate_fail: recomputed Spearman(g0,G) diverged from #667 committed rho",
            extra={"failure_class": "code", "reason": "g0_correctness_gate_fail"},
        )
        raise RuntimeError("analysis g0-correctness gate FAILED (rc=3) -- HALT.")
    if rc == 4:
        # BLOCKER 2: a behavior is missing off-diagonal cells with no valid excuse
        # (em/syco always HARD-FAIL; fact HARD-FAILs unless it was dropped). The
        # headline JSON is NOT written for the offending behavior.
        write_sentinel(
            "epm:failure",
            "tf_margin_cell_coverage_incomplete: a behavior is missing off-diagonal "
            "(source,target) cells (partial extract / stale resume / upload gap) -- "
            "the headline denominator would be silently shrunk; see tf_margin_analysis.log",
            extra={"failure_class": "code", "reason": "tf_margin_cell_coverage_incomplete"},
        )
        raise RuntimeError("analysis cell-coverage gate FAILED (rc=4) -- HALT.")
    if rc != 0:
        raise RuntimeError(
            f"analysis phase failed (rc={rc}); see {_log_dir() / 'tf_margin_analysis.log'}"
        )
    logger.info("[phase=analysis_done]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #667 tf-margin dispatcher (parity / fact-pool / extract / analysis).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "phase",
        nargs="?",
        choices=["apply-parity-probe", "build-fact-pool", "extract", "analysis", "all"],
        default="all",
        help="Phase to run. 'all' = apply-parity-probe -> build-fact-pool -> extract -> analysis.",
    )
    parser.add_argument(
        "--behaviors",
        type=lambda s: [b.strip() for b in s.split(",") if b.strip()],
        default=["em", "sycophancy", "fact"],
    )
    parser.add_argument(
        "--sources", default=None, help="Comma-separated source cids (smoke subset)."
    )
    parser.add_argument(
        "--targets", default=None, help="Comma-separated target cids (smoke subset)."
    )
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--cap", type=int, default=40, help="tf-margin fixed pool cap per side.")
    parser.add_argument("--n-gpus", type=int, default=8, help="CEILING; detected count is truth.")
    parser.add_argument(
        "--n-samples", type=int, default=10, help="apply-parity probe on-policy samples."
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU (local smoke).")
    parser.add_argument(
        "--smoke", action="store_true", help="Smoke mode (cap probes; on-pod analysis)."
    )
    parser.add_argument(
        "--max-probes", type=int, default=None, help="Cap fact-pool probes (smoke)."
    )
    parser.add_argument("--skip-upload", action="store_true", help="Skip HF upload (local smoke).")
    parser.add_argument("--skip-parity", action="store_true", help="Skip the apply-parity probe.")
    parser.add_argument("--no-resume-skip", action="store_true", help="Force full re-extract.")
    parser.add_argument(
        "--skip-store-pin", action="store_true", help="Analysis: synthetic-store smoke."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build + log commands, skip subprocs."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )

    # `uv run python` does NOT auto-load .env; load at main()-top so every
    # subprocess inherits HF_TOKEN/WANDB_API_KEY/ANTHROPIC_API_KEY via env={**os.environ} (#397).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    if args.phase == "all":
        phases = ["apply-parity-probe", "build-fact-pool", "extract", "analysis"]
    else:
        phases = [args.phase]
    smoke = args.smoke or args.cpu_only
    max_probes = args.max_probes if args.max_probes is not None else (2 if smoke else None)

    needs_creds = any(p in ("apply-parity-probe", "build-fact-pool", "extract") for p in phases)
    if needs_creds and not args.dry_run and not args.skip_store_pin:
        _require_credentials()

    for phase in phases:
        if phase == "apply-parity-probe":
            if args.skip_parity:
                logger.info("apply-parity-probe SKIPPED (--skip-parity)")
                continue
            phase_apply_parity(
                cpu_only=args.cpu_only, n_samples=args.n_samples, dry_run=args.dry_run
            )
        elif phase == "build-fact-pool":
            phase_build_fact_pool(
                behaviors=args.behaviors,
                cpu_only=args.cpu_only,
                cap=args.cap,
                max_probes=max_probes,
                skip_upload=args.skip_upload,
                dry_run=args.dry_run,
            )
        elif phase == "extract":
            phase_extract(
                behaviors=args.behaviors,
                sources_arg=args.sources,
                targets_arg=args.targets,
                cap=args.cap,
                n_gpus=args.n_gpus,
                cpu_only=args.cpu_only,
                skip_upload=args.skip_upload,
                dry_run=args.dry_run,
                resume_skip=not args.no_resume_skip,
            )
        elif phase == "analysis":
            phase_analysis(
                behaviors=args.behaviors,
                layer=args.layer,
                skip_store_pin=args.skip_store_pin,
                dry_run=args.dry_run,
            )

    note = (
        f"phases={phases} behaviors={args.behaviors} sources={args.sources} "
        f"targets={args.targets} smoke={smoke} dry_run={args.dry_run}"
    )
    write_sentinel("epm:results", note, extra={"phases": phases, "smoke": smoke})
    logger.info("[phase=done]")  # terminal marker — reserved for this single line
    return 0


if __name__ == "__main__":
    sys.exit(main())
