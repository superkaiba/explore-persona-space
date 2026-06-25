#!/usr/bin/env python3
"""Issue #667 dispatcher — gate-chain forward-pass preview (A3.6-A3.10).

PASS_UNIFIED architectural parity (Step 6d.0): smoke IS the sweep with one
behavior / 1-2 sources / 2-3 targets. EVERY phase the dispatcher runs derives
its cell subset from the SAME ``--behaviors`` / ``--sources`` / ``--targets``
filters, so a smoke is the production sweep scaled down — no separate in-process
smoke path. The phase list (prefetch -> extract -> upload -> analysis) is
identical; only the cell COUNT differs.

Phases (plan §4.2 DAG):

- ``prefetch`` (CPU, ~min): stage #537 frozen context inputs, SHA-pin the
  #537 G_meta git_commit + #658 store probe_pool_hash + the ported registry
  hash (== the G_meta pin), and run the rsLoRA parity probe on 1 adapter
  (fitness check (g), HALT on mismatch). Also runs the B3 reduction unit test.
- ``extract`` (GPU, ~6 GPU-h full / ~min smoke): per-source-adapter forward
  pass via ``scripts/issue667_extract.py`` as a CVD-pinned subprocess per cell
  (waves of n_gpus). Writes per-cell .npz under
  ``eval_results/issue_667/analysis_tensors/`` and uploads them to the HF data
  repo (analysis-tensor Upload Policy) before pod terminate.
- ``analysis`` (CPU, off-pod): ``scripts/issue667_analysis.py`` — A3.6-A3.10
  + the B3 gate, reading the uploaded store. Runs on-pod for the smoke (so the
  unified smoke exercises Phase 2 end-to-end) and off-pod for the full sweep.

Pod-side contract (CLAUDE.md / poll_pipeline.py): emits ``[phase=<name>]`` log
lines, a terminal ``[phase=done]`` (RESERVED — never on per-cell echoes), and an
end-of-run sentinel JSON at ``/workspace/logs/issue-667-<kind_slug>-<epoch>.json``
carrying ``_SENTINEL_REQUIRED_KEYS`` (sentinel_schema_version / kind / version).

Per-GPU fan-out pins ``CUDA_VISIBLE_DEVICES=<gpu>`` in the LAUNCHER env per cell
(+ the matching ``--gpu-id``) so an import-time cuInit can't co-locate cells on
GPU 0 (#545). Every subprocess gets an explicit ``env={**os.environ}``;
``load_dotenv()`` at main()-top puts HF_TOKEN/WANDB_API_KEY in os.environ first.

Launch (plan §10)::

    uv run python scripts/dispatch_issue.py launch --issue 667 --intent eval \\
        --repo-branch issue-667 --workload-cmd \\
        'uv run python scripts/issue667_dispatch.py extract --behaviors em,sycophancy,fact \\
         --layers 7 14 21 --primary-layer 14'

Smoke (the unified single-cell sweep)::

    uv run python scripts/issue667_dispatch.py all \\
        --behaviors em --sources default,sp_swe --targets default,sp_swe,fmt_json \\
        --layers 14 --primary-layer 14 --smoke
"""

# math/scientific notation in docstrings + messages

from __future__ import annotations

import argparse
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
# Add scripts/ so the cross-script ``import issue667_extract`` / ``issue667_analysis``
# resolve when this dispatcher is launched as a script (sys.path[0] is scripts/
# already under `python scripts/...`, but make it explicit + cwd-independent).
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("issue667_dispatch")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_ANALYSIS_TENSORS_PREFIX = "issue667_gate_chain_preview/analysis_tensors"
TENSORS_DIR = "eval_results/issue_667/analysis_tensors"
OUT_DIR = "eval_results/issue_667"


# ─────────────────────────────────────────────────────────────────────────────
# Log dir + phase lines + sentinel
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
        "by": "issue667_dispatch",
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


def _run_with_log(
    cmd: Sequence[str], *, log_path: Path, extra_env: dict[str, str] | None = None
) -> int:
    """Run a child process, tee stdout/stderr to a log file. Returns rc.

    Explicit ``env={**os.environ}`` (+ extra_env) — `uv run python` does not
    auto-load .env, so load_dotenv() at main()-top puts the creds in os.environ
    first (#397 round-10').
    """
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
    """Run several subprocesses concurrently (wave). Returns rc list."""
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
    """Source contexts for a behavior: the 16 train cids, filtered by --sources."""
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
    """Target contexts: None = the 30 eval cids (extractor default); else the subset."""
    if not targets_arg:
        return None  # extractor defaults to eval_cids_for(behavior) + source
    from explore_persona_space.experiments.i537_contexts import eval_cids_for

    full = set(eval_cids_for(behavior))
    requested = [t.strip() for t in targets_arg.split(",") if t.strip()]
    unknown = [t for t in requested if t not in full and t not in select_sources(behavior, None)]
    if unknown:
        raise ValueError(f"--targets {unknown!r} not in the {behavior} eval grid")
    return requested


# ─────────────────────────────────────────────────────────────────────────────
# Phase: PREFETCH (stage inputs, SHA-pin, parity probe, B3 unit test)
# ─────────────────────────────────────────────────────────────────────────────


def phase_prefetch(*, behaviors: list[str], cpu_only: bool, skip_parity: bool) -> None:
    """Stage #537 inputs, assert all pins, run the B3 unit test + rsLoRA parity probe."""
    phase_log("prefetch")
    from explore_persona_space.analysis.issue667 import (
        EXPECTED_G_META_GIT_COMMIT,
        EXPECTED_REGISTRY_HASH,
        EXPECTED_STORE_PROBE_POOL_HASH,
    )
    from explore_persona_space.analysis.issue667.gate_chain import (
        whitened_gate_reduction_unit_test,
    )

    # B3 reduction unit test (gates A3.9/A3.10 downstream).
    whitened_gate_reduction_unit_test()
    logger.info("B3 reduction unit test PASS")

    # Stage frozen context inputs + assert the registry hash == the G_meta pin.
    from issue667_extract import stage_inputs

    sampled_path, demos_path = stage_inputs()
    from explore_persona_space.experiments.i537_contexts import (
        load_icl_demos,
        load_registry,
        registry_hash,
    )

    reg = load_registry(sampled_path)
    demos = load_icl_demos(demos_path)
    rh = registry_hash(reg, demos)
    assert rh == EXPECTED_REGISTRY_HASH, (
        f"registry_hash drift: {rh} != {EXPECTED_REGISTRY_HASH} (#537 ground truth) — "
        "the ported context registry or the frozen inputs do not match #537."
    )
    logger.info("registry_hash OK (== G_meta pin): %s", rh[:16])

    # SHA-pin the #537 G_meta git_commit + #658 store probe_pool_hash.
    from issue667_analysis import assert_store_pin, load_g_meta

    g_meta = load_g_meta()
    logger.info("G_meta git_commit pin OK: %s", g_meta["git_commit"][:16])
    assert_store_pin()
    logger.info("#658 store probe_pool_hash pin OK: %s", EXPECTED_STORE_PROBE_POOL_HASH[:16])
    logger.info("EXPECTED_G_META_GIT_COMMIT=%s", EXPECTED_G_META_GIT_COMMIT[:16])

    # rsLoRA parity probe (fitness check (g)) — 1 adapter reproduces #537's
    # diagonal source write at the committed gauge. HALT on mismatch.
    if skip_parity:
        logger.info("parity probe SKIPPED (--skip-parity)")
    else:
        _rslora_parity_probe(behaviors[0], cpu_only=cpu_only)
    logger.info("[phase=prefetch_done]")


def _rslora_parity_probe(behavior: str, *, cpu_only: bool) -> None:
    """Apply 1 adapter, confirm the gauge asserts pass + a non-trivial source write.

    The full diagonal-G reproduction needs a GPU forward; on CPU-only smokes we
    assert the adapter loads + gauge passes (the (g) config check) and defer the
    numeric diagonal-G reproduction to the GPU extract phase's source diagonal.
    """
    from issue667_extract import assert_adapter_gauge, stage_adapter_local

    adapter_dir = stage_adapter_local(behavior, "default", 42)
    gauge = assert_adapter_gauge(adapter_dir, behavior)
    assert gauge["use_rslora"], "parity probe: adapter is not rsLoRA (gauge mismatch)"
    logger.info(
        "rsLoRA parity probe (config): %s default adapter r=%s alpha=%s use_rslora=%s",
        behavior,
        gauge["r"],
        gauge["lora_alpha"],
        gauge["use_rslora"],
    )
    if cpu_only:
        logger.info(
            "parity probe: CPU-only — gauge config asserted; numeric diagonal-G "
            "reproduction runs at the GPU extract source diagonal."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase: EXTRACT (per-source-adapter forward-pass, CVD-pinned waves)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_cmd(
    behavior: str,
    source: str,
    targets: list[str] | None,
    layers: list[int],
    primary_layer: int,
    gpu_id: int,
    max_probes: int | None,
    max_train_rows: int | None,
    cpu_only: bool,
) -> tuple[list[str], Path, dict[str, str]]:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue667_extract.py",
        "--behavior",
        behavior,
        "--source-cid",
        source,
        "--layers",
        *[str(li) for li in layers],
        "--primary-layer",
        str(primary_layer),
        "--out",
        TENSORS_DIR,
        "--gpu-id",
        str(gpu_id),
    ]
    if targets:
        cmd += ["--targets", ",".join(targets)]
    if max_probes:
        cmd += ["--max-probes", str(max_probes)]
    if max_train_rows is not None:
        cmd += ["--max-train-rows", str(max_train_rows)]
    if cpu_only:
        cmd += ["--cpu-only"]
    # CVD pinned in the LAUNCHER env per cell (#545) — NOT only via --gpu-id.
    env = {"CUDA_VISIBLE_DEVICES": "" if cpu_only else str(gpu_id)}
    log_path = _log_dir() / f"extract_{behavior}_{source}.log"
    return cmd, log_path, env


def phase_extract(
    *,
    behaviors: list[str],
    sources_arg: str | None,
    targets_arg: str | None,
    layers: list[int],
    primary_layer: int,
    n_gpus: int,
    cpu_only: bool,
    max_probes: int | None,
    max_train_rows: int | None,
    skip_upload: bool,
    dry_run: bool,
) -> None:
    """Per-source-adapter extraction in CVD-pinned waves; upload tensors after."""
    phase_log("extract")
    cells: list[tuple[str, str]] = []  # (behavior, source)
    for behavior in behaviors:
        for source in select_sources(behavior, sources_arg):
            cells.append((behavior, source))
    logger.info("extract: %d source-adapter cells across behaviors=%s", len(cells), behaviors)
    n_par = 1 if cpu_only else max(n_gpus, 1)
    for wave_start in range(0, len(cells), n_par):
        wave = cells[wave_start : wave_start + n_par]
        cmds = []
        for i, (behavior, source) in enumerate(wave):
            targets = select_targets(behavior, targets_arg)
            cmds.append(
                _extract_cmd(
                    behavior,
                    source,
                    targets,
                    layers,
                    primary_layer,
                    i % n_par,
                    max_probes,
                    max_train_rows,
                    cpu_only,
                )
            )
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
            logger.info("extract cell %s/%s complete", behavior, source)  # NOT [phase=done]
    if dry_run:
        logger.info("[phase=extract_done] (dry-run: no tensors, upload skipped)")
        return
    if not skip_upload:
        _upload_tensors()
    logger.info("[phase=extract_done]")


def _upload_tensors() -> None:
    """Upload per-cell .npz tensors to the HF data repo (analysis-input contract).

    One bulk create_commit (well under the 256/hr cap), verified on a fresh Hub
    listing before trusting the pod can terminate (Upload Policy #521).
    """
    if os.environ.get("EPM_SKIP_UPLOAD") == "1":
        logger.info("EPM_SKIP_UPLOAD=1 -> skipping tensor upload (smoke/local)")
        return
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files

    tdir = PROJECT_ROOT / TENSORS_DIR
    npzs = sorted(tdir.rglob("*.npz"))
    if not npzs:
        raise RuntimeError(f"no .npz tensors to upload under {tdir} -- extraction wrote nothing")
    api = HfApi()
    ops = [
        CommitOperationAdd(
            path_in_repo=f"{HF_ANALYSIS_TENSORS_PREFIX}/{p.relative_to(tdir).as_posix()}",
            path_or_fileobj=str(p),
        )
        for p in npzs
    ]
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue667: {len(ops)} per-cell gate-chain tensors",
    )
    files = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    missing = [
        p.relative_to(tdir).as_posix()
        for p in npzs
        if f"{HF_ANALYSIS_TENSORS_PREFIX}/{p.relative_to(tdir).as_posix()}" not in files
    ]
    if missing:
        raise RuntimeError(f"tensor upload verification FAILED -- missing on Hub: {missing[:5]}")
    logger.info("uploaded + verified %d tensors to %s", len(npzs), HF_DATA_REPO)


# ─────────────────────────────────────────────────────────────────────────────
# Phase: ANALYSIS (CPU; A3.6-A3.10 via issue667_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────


def phase_analysis(*, behaviors: list[str], primary_layer: int, skip_store_pin: bool) -> None:
    """A3.6-A3.10 + B3 gate via issue667_analysis.py (on-pod smoke / off-pod full)."""
    phase_log("analysis")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue667_analysis.py",
        "--tensors-dir",
        TENSORS_DIR,
        "--out-dir",
        OUT_DIR,
        "--behaviors",
        *behaviors,
        "--primary-layer",
        str(primary_layer),
    ]
    if skip_store_pin:
        cmd += ["--skip-store-pin"]
    rc = _run_with_log(cmd, log_path=_log_dir() / "analysis.log")
    if rc != 0:
        raise RuntimeError(f"analysis phase failed (rc={rc}); see {_log_dir() / 'analysis.log'}")
    logger.info("[phase=analysis_done]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #667 dispatcher (prefetch / extract / analysis / all).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "phase",
        nargs="?",
        choices=["prefetch", "extract", "analysis", "all"],
        default="all",
        help="Phase to run. 'all' = prefetch -> extract -> analysis (the unified smoke/sweep).",
    )
    parser.add_argument(
        "--behaviors",
        type=lambda s: [b.strip() for b in s.split(",") if b.strip()],
        default=["em", "sycophancy", "fact"],
        help="Comma-separated in-scope behaviors (smoke: em).",
    )
    parser.add_argument(
        "--sources", default=None, help="Comma-separated source cids (smoke subset)."
    )
    parser.add_argument(
        "--targets", default=None, help="Comma-separated target cids (smoke subset)."
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21])
    parser.add_argument("--primary-layer", type=int, default=14)
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU (local smoke).")
    parser.add_argument(
        "--smoke", action="store_true", help="Smoke mode (cap probes/rows; on-pod analysis)."
    )
    parser.add_argument("--max-probes", type=int, default=None, help="Cap eval probes (smoke).")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Cap t+/t- rows (smoke).")
    parser.add_argument(
        "--skip-upload", action="store_true", help="Skip HF tensor upload (local smoke)."
    )
    parser.add_argument("--skip-parity", action="store_true", help="Skip the rsLoRA parity probe.")
    parser.add_argument(
        "--skip-store-pin",
        action="store_true",
        help="Pass through to analysis: synthetic-store smoke (no HF pins).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build + log commands, skip GPU subprocs."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )

    # `uv run python` does NOT auto-load .env; load it at main()-top so every
    # subprocess inherits HF_TOKEN/WANDB_API_KEY via env={**os.environ} (#397).
    from dotenv import load_dotenv

    load_dotenv()

    phases = ["prefetch", "extract", "analysis"] if args.phase == "all" else [args.phase]
    smoke = args.smoke or args.cpu_only
    # Smoke defaults: cap probes + train rows, run analysis on-pod with the pins
    # unless explicitly synthetic.
    max_probes = args.max_probes if args.max_probes is not None else (2 if smoke else None)
    max_train_rows = (
        args.max_train_rows if args.max_train_rows is not None else (8 if smoke else None)
    )

    if (
        any(p in ("prefetch", "extract") for p in phases)
        and not args.dry_run
        and not args.skip_store_pin
    ):
        _require_credentials()

    for phase in phases:
        if phase == "prefetch":
            phase_prefetch(
                behaviors=args.behaviors, cpu_only=args.cpu_only, skip_parity=args.skip_parity
            )
        elif phase == "extract":
            phase_extract(
                behaviors=args.behaviors,
                sources_arg=args.sources,
                targets_arg=args.targets,
                layers=args.layers,
                primary_layer=args.primary_layer,
                n_gpus=args.n_gpus,
                cpu_only=args.cpu_only,
                max_probes=max_probes,
                max_train_rows=max_train_rows,
                skip_upload=args.skip_upload,
                dry_run=args.dry_run,
            )
        elif phase == "analysis":
            phase_analysis(
                behaviors=args.behaviors,
                primary_layer=args.primary_layer,
                skip_store_pin=args.skip_store_pin,
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
