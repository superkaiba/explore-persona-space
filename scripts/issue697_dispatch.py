"""Issue #697 dispatcher — causal context-vector (CV) patch on #537's adapters.

Decomposes whether finetuning a behavior B into a context C moved the model's
internal context picture ``c_C`` (the input to the theory's map M) or the map M
itself, by cross-model residual-stream patching at read layer L=14 on #537's
already-trained behaviorxcontext LoRA adapters (Qwen-2.5-7B-Instruct). Forward
passes only (HF ``model.forward`` / ``model.generate``; NO vLLM, plan §8).

PASS_UNIFIED architectural parity (Step 6d.0): the smoke IS the sweep with one
cell. Every phase runs each cell through the SAME per-cell function the full
sweep calls; ``--cells em_sp_swe_seed42 --cpu-only`` is the only difference. No
smoke-vs-sweep code divergence (same wave dispatcher with the same ``--n-gpus``
cell-sharding, same env injection, same sentinel, same poll contract). The
per-phase cell-list ALL derives from the same ``--cells`` subset.

Phases (plan §4.1 dependency DAG):

- ``vendor``  (CPU, 0 GPU): import-smoke — confirms the vendored
  ``analysis.activation_shift`` + ``analysis.cv_patch`` + ``experiments.issue_651``
  read path import cleanly and the 14x20 panel materializes. No models loaded.
- ``canary``  (1x A100-80, ~0.4 GPU-h): the pre-sweep no-go (plan §7).
  Gate C1.1 self-patch identity ≈0 (read + generate), Gate C1.2 non-identity
  KV-cache propagation (cache-vs-no-cache parity), Gate C1.3 decoded-token slot
  audit — all on the production panel through ``cv_patch`` — PLUS Gate C2 (the
  inherited #651 Gate 7a rsLoRA application-scaling parity, reproducing #521's
  committed marker numbers through the same ``merge_and_unload`` path). The two
  canary cells (one root-layout marker, one ``sft_em_adapter/``-nested em) double
  as the smoke-architecture canary (they run the full per-cell patch path).
- ``sweep``   (4x A100-80, ~21 GPU-h, ``--n-gpus 4``): per (B, C, seed) cell over
  the 128-cell grid (4 behaviors x 16 contexts x 2 seeds). Each cell: stage
  adapter → load base + FT (merge_and_unload) → capture c0/c+/v0/v+ → P↓/P↑ + the
  4 controls on the 14x20 panel → persist per-cell ``.pt`` (mechanistic v, both
  poolings for marker/fact) + ``_E.json`` (the patched on-policy generations for
  downstream judging; marker DV computed inline). Per-cell artifacts upload to HF
  the moment a cell completes (a mid-sweep crash strands fewer than N cells).
- ``analyze`` (CPU, 0 GPU, OFF-POD): bootstrap CI on f_CV per behavior over the
  280 personaxquestion pairs (persona-clustered), the v-space f_CV, and the hero
  2x4 grid. Runs after the pod terminates over the HF-uploaded per-cell tensors.

Pod-side contract (CLAUDE.md / poll_pipeline.py): emits ``[phase=<name>]`` log
lines, a terminal ``[phase=done]`` (reserved for the single terminal line —
per-cell completions are worded WITHOUT the phase tag), and an end-of-run
sentinel JSON at ``/workspace/logs/issue-697-<kind_slug>-<epoch>.json`` carrying
``_SENTINEL_REQUIRED_KEYS`` (sentinel_schema_version / kind / version). Pod-side
code NEVER shells out to scripts/task.py (CLAUDE.md).

Single-variable-from-parent reuse (plan §4.2): the ONLY new code is the
cross-model patch hook (``analysis.cv_patch``) + this dispatcher; every adapter,
panel, read recipe, layer, and DV is inherited byte-identically from #537/#651.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shlex
import subprocess
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

logger = logging.getLogger("issue697_dispatch")

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# Per-cell tensor / E-json destination on the HF data repo (analysis-input
# contract — Upload Policy: intermediate analysis tensors the analyze phase
# consumes MUST land on HF before pod terminate, #521).
HF_TENSOR_PREFIX = "issue697_cv_patch/analysis_tensors"

# The 4 behaviors #697 reads (em/sycophancy/marker/fact). refusal (partial null,
# #651) + emnc (positives-only Betley bridge) are EXCLUDED — plan §10.
BEHAVIORS_697: tuple[str, ...] = ("em", "sycophancy", "marker", "fact")

# Primary read/patch layer + the depth-sweep supplement (plan §10; Source: #651).
PRIMARY_LAYER = 14
SUPPLEMENT_LAYERS = (7, 21)

# Per-behavior PRIMARY v pooling (item-5 fix — mirrors #651's headline):
# mean-resp for em/sycophancy, end-of-response slot for marker/fact.
PRIMARY_POOLING: dict[str, str] = {
    "em": "mean_resp",
    "sycophancy": "mean_resp",
    "marker": "slot",
    "fact": "slot",
}


def _resolve_repo_root() -> Path:
    out = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        env={**os.environ},  # epm-lint: subprocess-env-inherit -- git toplevel probe, no creds
    ).decode()
    return Path(out.strip())


def _log_dir() -> Path:
    override = os.environ.get("EPM_LOG_DIR")
    if override:
        d = Path(override)
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = Path("/workspace/logs")
    if not d.exists():  # local VM (no /workspace) -> repo logs/
        d = _resolve_repo_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def phase_log(name: str) -> None:
    """Emit the ``[phase=<name>]`` line poll_pipeline.py parses (PHASE_RE).

    The poller's PHASE_RE matches ``[a-z0-9_]+`` so numbered phase tokens parse
    fully; this dispatcher uses only lowercase-underscore phase names anyway.
    """
    print(f"[phase={name}]", flush=True)


def write_sentinel(kind: str, note: str, *, version: int = 1, extra: dict | None = None) -> Path:
    """End-of-run sentinel with poll_pipeline's _SENTINEL_REQUIRED_KEYS.

    Required keys: sentinel_schema_version (int 1), kind (full marker string),
    version (int). The marker body goes under ``note``.
    """
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "task_id": 697,
        "by": "issue697_dispatch",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    if extra:
        payload.update(extra)
    slug = kind.replace(":", "_")
    out = _log_dir() / f"issue-697-{slug}-{time.time_ns()}.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("sentinel written: %s", out)
    return out


def _run_with_log(
    cmd: Sequence[str],
    *,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> int:
    """Run a child process, tee stdout/stderr to a log file. Returns rc.

    Every subprocess gets an EXPLICIT ``env={**os.environ}`` (+ extra_env): the
    credential env (HF_TOKEN / WANDB_API_KEY / ANTHROPIC_API_KEY) must be present
    even though ``uv run python`` does not auto-load .env — load_dotenv() in
    main() puts it in os.environ first (#397 round-10').
    """
    env = {**os.environ}
    if extra_env:
        env.update(extra_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "$ %s  >>> %s%s",
        " ".join(shlex.quote(c) for c in cmd),
        log_path,
        f" (env+={list(extra_env.keys())})" if extra_env else "",
    )
    with log_path.open("ab") as f:
        proc = subprocess.run(
            list(cmd),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    rc = proc.returncode
    if rc != 0:
        logger.error("command exited with rc=%d (log: %s)", rc, log_path)
    return rc


def _run_parallel_with_log(
    cmds: Iterable[tuple[Sequence[str], Path, dict[str, str] | None]],
    *,
    cwd: Path | None = None,
) -> list[int]:
    """Run several subprocesses concurrently. Returns parallel list of rc codes."""
    procs: list[subprocess.Popen] = []
    files = []
    for cmd, log_path, extra_env in cmds:
        env = {**os.environ}
        if extra_env:
            env.update(extra_env)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = log_path.open("ab")
        files.append(f)
        logger.info(
            "$ (parallel) %s  >>> %s%s",
            " ".join(shlex.quote(c) for c in cmd),
            log_path,
            f" (env+={list(extra_env.keys())})" if extra_env else "",
        )
        p = subprocess.Popen(
            list(cmd), stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None
        )
        procs.append(p)
    rcs = [p.wait() for p in procs]
    for f in files:
        f.close()
    return rcs


def _require_credentials() -> None:
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"


# ---------------------------------------------------------------------------
# Cell grid (4 behaviors x 16 contexts x 2 seeds = 128; plan §10)
# ---------------------------------------------------------------------------


def cells_697(n_gpus: int = 4, floor_only: bool = False):
    """The 128 cells #697 reads: em/sycophancy/marker/fact x 16 ctx x 2 seeds.

    Filters the inherited #651 ``readable_cells`` to ``BEHAVIORS_697`` (drops
    refusal + emnc — plan §10), then re-densifies ``gpu_id`` round-robin over the
    requested subset so the wave dispatcher shards evenly. ``floor_only`` keeps
    only the seed-42 cells (the auto-descope floor, plan §9 stratification).
    """
    from explore_persona_space.experiments.issue_651 import Cell, readable_cells

    full = readable_cells(n_gpus=n_gpus, include_seed1042=not floor_only)
    sel = [c for c in full if c.behavior in BEHAVIORS_697]
    return [
        Cell(behavior=c.behavior, cid=c.cid, seed=c.seed, gpu_id=i % max(n_gpus, 1))
        for i, c in enumerate(sel)
    ]


def _select_cells(args):
    """Resolve the per-phase cell subset from --cells (or the full 128-cell grid).

    PASS_UNIFIED parity: EVERY phase (canary / sweep) reads from this SAME
    ``cells_697`` grid filtered by the SAME ``--cells`` subset, so a smoke is the
    sweep with one cell and no phase re-enumerates a different grid.
    """
    from explore_persona_space.experiments.issue_651 import Cell, parse_cell_spec

    full = cells_697(n_gpus=args.n_gpus, floor_only=args.floor_only)
    if args.cells:
        requested = [parse_cell_spec(s) for s in args.cells]
        avail = {(c.behavior, c.cid, c.seed) for c in full}
        unknown = [r.cell_id for r in requested if (r.behavior, r.cid, r.seed) not in avail]
        if unknown:
            raise ValueError(
                f"--cells {unknown!r} not in the #697 grid "
                f"(behaviors={BEHAVIORS_697}, seeds available: "
                f"{sorted({(c.behavior, c.seed) for c in full})})"
            )
        return [
            Cell(behavior=r.behavior, cid=r.cid, seed=r.seed, gpu_id=i % max(args.n_gpus, 1))
            for i, r in enumerate(requested)
        ]
    return full


# ---------------------------------------------------------------------------
# Phase: VENDOR (CPU import-smoke — no models)
# ---------------------------------------------------------------------------


def phase_vendor(repo_root: Path) -> None:
    """Confirm the vendored read path + cv_patch import cleanly + panel materializes."""
    phase_log("vendor")
    # Import the full read path (vendor verification per plan A0).
    from explore_persona_space.analysis import cv_patch
    from explore_persona_space.analysis.activation_shift import (  # noqa: F401
        _build_chatml_prompt,
        _read_residuals,
        extract_per_context_shifts,
    )
    from explore_persona_space.experiments.issue_651 import (
        build_panel_personas,
        build_panel_questions,
    )

    personas = build_panel_personas()
    questions = build_panel_questions()
    cells = cells_697()
    assert len(personas) == 14, (len(personas), "expected the fixed 14-persona panel")
    assert len(questions) == 20, (len(questions), "expected the 20-question panel")
    assert len(cells) == 128, (len(cells), "expected 4 behaviors x 16 ctx x 2 seeds = 128")
    # cv_patch public surface present.
    for name in (
        "content_patch_pos",
        "audit_patch_slot",
        "make_cv_patch_hook",
        "patched_read",
        "patched_generate",
        "first_token_logits",
        "compute_f_cv",
        "compute_f_cv_down",
        "NO_EFFECT",
        "SlotAuditError",
    ):
        assert hasattr(cv_patch, name), f"cv_patch missing {name}"
    logger.info(
        "vendor smoke OK: %d personas, %d questions, %d cells; cv_patch surface present",
        len(personas),
        len(questions),
        len(cells),
    )
    logger.info("[phase=vendor_done]")


# ---------------------------------------------------------------------------
# Phase: CANARY (Gate C1 patch-correctness + Gate C2 rsLoRA parity)
# ---------------------------------------------------------------------------


def phase_canary(repo_root: Path, *, cpu_only: bool, smoke_model: str | None) -> None:
    """Run Gate C1 (cv_patch correctness) + Gate C2 (inherited #651 Gate 7a); HALT on FAIL."""
    phase_log("canary")
    cmd = ["uv", "run", "python", "scripts/issue697_canary.py"]
    if cpu_only:
        cmd.append("--cpu-only")
    if smoke_model:
        cmd += ["--smoke-model", smoke_model]
    env = {"CUDA_VISIBLE_DEVICES": "" if cpu_only else "0"}
    log_path = _log_dir() / "canary.log"
    rc = _run_with_log(cmd, log_path=log_path, extra_env=env, cwd=repo_root)
    if rc != 0:
        raise RuntimeError(
            f"CANARY FAILED (rc={rc}) -- Gate C1/C2 did not PASS; HALT before the sweep. "
            f"See {log_path}"
        )
    logger.info("[phase=canary_done]")


# ---------------------------------------------------------------------------
# Phase: SWEEP (per-cell patch read on the panel; wave-sharded, per-cell upload)
# ---------------------------------------------------------------------------


def _cell_cmd(
    repo_root: Path,
    cell,
    *,
    cpu_only: bool,
    panel_personas_json: Path,
    panel_questions_json: Path,
    out_dir: Path,
    layers: Sequence[int],
    primary_layer: int,
    max_new_tokens: int,
    skip_e: bool,
    smoke_model: str | None,
    upload: bool,
    use_cache: bool,
) -> tuple[list[str], Path, dict[str, str]]:
    """Build (cmd, log_path, env) for one cell's patch read via issue697_cell.py."""
    base_model = smoke_model or QWEN_ID
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue697_cell.py",
        "--behavior",
        cell.behavior,
        "--cid",
        cell.cid,
        "--seed",
        str(cell.seed),
        "--adapter-subfolder",
        cell.adapter_subfolder,
        "--personas-json",
        str(panel_personas_json),
        "--questions-json",
        str(panel_questions_json),
        "--out-dir",
        str(out_dir),
        "--layers",
        *[str(L) for L in layers],
        "--primary-layer",
        str(primary_layer),
        "--max-new-tokens",
        str(max_new_tokens),
        "--base-model-id",
        base_model,
    ]
    # Thread the canary's use_cache decision (concern #4): BooleanOptionalAction.
    cmd.append("--use-cache" if use_cache else "--no-use-cache")
    if cpu_only:
        cmd.append("--cpu-only")
    if skip_e:
        cmd.append("--skip-e")
    if upload:
        cmd.append("--upload")
    env = {"CUDA_VISIBLE_DEVICES": "" if cpu_only else str(cell.gpu_id)}
    log_path = _log_dir() / f"sweep_{cell.cell_id}.log"
    return cmd, log_path, env


def _read_use_cache_decision(repo_root: Path) -> bool:
    """Read the canary's use_cache decision (concern #4); default True if absent.

    The canary writes ``canary_decision.json`` with
    ``use_cache_production_default``. A real sweep runs after the canary in the
    same dispatch, so the file is present; a standalone ``--phase sweep`` (or a
    CPU smoke that skips the canary) defaults to True (the safe default when Gate
    C1.2's parity passes comfortably — the canary HALTs a genuinely broken hook
    before the sweep regardless).

    The decision path is the canary's canonical constant; we form it inline
    (NOT ``from scripts.issue697_canary import ...``) because this module runs as
    a SCRIPT (``uv run python scripts/issue697_dispatch.py``), so ``sys.path[0]``
    is ``scripts/`` and the ``scripts`` package is not importable from here.
    """
    p = repo_root / "eval_results" / "issue_697" / "canary" / "canary_decision.json"
    if not p.exists():
        logger.info("no canary use_cache decision at %s -> default use_cache=True", p)
        return True
    decision = bool(json.loads(p.read_text()).get("use_cache_production_default", True))
    logger.info("canary use_cache decision: use_cache=%s (%s)", decision, p)
    return decision


def _assert_sweep_device_count(cpu_only: bool, n_gpus: int) -> None:
    """Belt-and-suspenders device-count preflight (standing rec / item 5c).

    The 4-GPU sweep needs the ``ft-7b`` 4x A100-80 intent; an orchestrator intent
    mis-inference (e.g. a 1-GPU ``lora-7b`` pod) would silently co-locate the
    waves. Assert the visible device count matches ``n_gpus`` before the sweep so
    a mis-launch FAILs loud at startup, not mid-sweep. Skipped on the CPU smoke.
    """
    if cpu_only:
        return
    import torch

    visible = torch.cuda.device_count()
    assert visible == n_gpus, (
        f"sweep phase requires {n_gpus} GPUs (got {visible}); the orchestrator must launch "
        f"with the matching intent (--n-gpus 4 -> ft-7b 4x A100-80). Set --n-gpus to the actual "
        f"device count if this is a deliberate smaller-pod run."
    )


def phase_sweep(
    repo_root: Path,
    cells: Sequence,
    *,
    n_gpus: int,
    cpu_only: bool,
    panel_personas_json: Path,
    panel_questions_json: Path,
    layers: Sequence[int],
    primary_layer: int,
    max_new_tokens: int,
    skip_e: bool,
    smoke_model: str | None,
    dry_run: bool,
    upload: bool,
) -> None:
    """Per-cell patch read over the panel (wave-parallel, CVD-pinned per cell)."""
    phase_log("sweep")
    # Device-count preflight (item 5c) — skip on dry-run (no GPU needed) + CPU smoke.
    if not dry_run and smoke_model is None:
        _assert_sweep_device_count(cpu_only, n_gpus)
    # use_cache threaded from the canary's Gate C1.2 decision (concern #4).
    use_cache = _read_use_cache_decision(repo_root)
    out_dir = repo_root / "eval_results" / "issue_697" / "patch"
    out_dir.mkdir(parents=True, exist_ok=True)
    for wave_start in range(0, len(cells), max(n_gpus, 1)):
        wave = cells[wave_start : wave_start + max(n_gpus, 1)]
        cmds: list[tuple[Sequence[str], Path, dict[str, str] | None]] = []
        for cell in wave:
            cmd, log_path, env = _cell_cmd(
                repo_root,
                cell,
                cpu_only=cpu_only,
                panel_personas_json=panel_personas_json,
                panel_questions_json=panel_questions_json,
                out_dir=out_dir,
                layers=layers,
                primary_layer=primary_layer,
                max_new_tokens=max_new_tokens,
                skip_e=skip_e,
                smoke_model=smoke_model,
                upload=upload,
                use_cache=use_cache,
            )
            cmds.append((cmd, log_path, env))
        if dry_run:
            for (cmd, _lp, env), cell in zip(cmds, wave, strict=True):
                logger.info(
                    "[dry-run] sweep %s CVD=%r :: %s",
                    cell.cell_id,
                    env.get("CUDA_VISIBLE_DEVICES"),
                    " ".join(shlex.quote(c) for c in cmd),
                )
            continue
        rcs = _run_parallel_with_log(cmds, cwd=repo_root)
        bad = [(rc, c.cell_id) for rc, c in zip(rcs, wave, strict=True) if rc != 0]
        if bad:
            raise RuntimeError(f"sweep wave failed: {bad}; see logs in {_log_dir()}")
        for c in wave:
            logger.info("sweep cell %s complete", c.cell_id)  # NOT [phase=done] (mid-run noise)
    if dry_run:
        logger.info("[phase=sweep_done] (dry-run: no tensors written, upload skipped)")
        return
    logger.info("[phase=sweep_done]")


# ---------------------------------------------------------------------------
# Phase: ANALYZE (CPU, off-pod — delegate to issue697_analysis.py)
# ---------------------------------------------------------------------------


def phase_analyze(repo_root: Path, *, primary_layer: int, skip_judge: bool = False) -> None:
    """Off-pod CPU judge + f_CV bootstrap + hero figure.

    Two steps, both off-pod CPU: (1) the vendored #537 judge (Sonnet 4.5) over the
    per-cell raw_completions → ``{cell}_judged.json`` (closes the
    ``e-judging-pipeline-not-vendored`` concern), then (2) ``issue697_analysis.py``
    (f_CV bootstrap + hero). ``--skip-judge`` runs only the v-space analysis (CPU
    smoke / no API key).
    """
    phase_log("analyze")
    patch_dir = repo_root / "eval_results" / "issue_697" / "patch"
    if not skip_judge:
        judge_cmd = [
            "uv",
            "run",
            "python",
            "scripts/issue697_judge.py",
            "--patch-dir",
            str(patch_dir),
        ]
        rc = _run_with_log(judge_cmd, log_path=_log_dir() / "judge.log", cwd=repo_root)
        if rc != 0:
            raise RuntimeError(f"analyze: judge step failed (rc={rc}); see {_log_dir()}/judge.log")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue697_analysis.py",
        "--primary-layer",
        str(primary_layer),
    ]
    log_path = _log_dir() / "analyze.log"
    rc = _run_with_log(cmd, log_path=log_path, cwd=repo_root)
    if rc != 0:
        raise RuntimeError(f"analyze phase failed (rc={rc}); see {log_path}")
    logger.info("[phase=analyze_done]")


# ---------------------------------------------------------------------------
# Panel JSON materialization
# ---------------------------------------------------------------------------


def _materialize_panel(repo_root: Path) -> tuple[Path, Path]:
    """Write the fixed panel personas + questions JSON each cell reads."""
    from explore_persona_space.experiments.issue_651 import (
        build_panel_personas,
        build_panel_questions,
    )

    panel_dir = repo_root / "eval_results" / "issue_697" / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    personas = build_panel_personas()
    questions = build_panel_questions()
    p_path = panel_dir / "panel_personas.json"
    q_path = panel_dir / "panel_questions.json"
    p_path.write_text(json.dumps(personas, indent=2))
    q_path.write_text(json.dumps(questions, indent=2))
    logger.info("panel materialized: %d personas, %d questions", len(personas), len(questions))
    return p_path, q_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #697 dispatcher (vendor / canary / sweep / analyze). "
            "Smoke = sweep with one cell: `--cells em_sp_swe_seed42 --cpu-only`. "
            "Sweep shards the 128 cells over --n-gpus (default 4 -> the ft-7b "
            "4x A100-80 intent)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=["vendor", "canary", "sweep", "analyze", "all"],
        default=["all"],
        help="Phases to run in order. 'all' = vendor -> canary -> sweep.",
    )
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Cell subset (e.g. 'em_sp_swe_seed42'); smoke = sweep with one cell.",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=4,
        help=(
            "GPUs to shard the 128 sweep cells over (default 4 -> the ft-7b "
            "4x A100-80 intent; the wave dispatcher pins CUDA_VISIBLE_DEVICES "
            "per cell). 128/4 ≈ 5.3 h wall, under the 24 h GCP fence (plan §9)."
        ),
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU (smoke).")
    parser.add_argument(
        "--smoke-model",
        default=None,
        help=(
            "Tiny base model id for a CPU smoke (e.g. Qwen/Qwen2.5-0.5B-Instruct). "
            "Replaces the 7B base+FT load so the CPU canary/smoke runs without a GPU."
        ),
    )
    parser.add_argument(
        "--skip-e",
        action="store_true",
        help=(
            "Skip the behavioral-E on-policy generations (capture the mechanistic "
            "v only). The CPU smoke sets this (tiny-model generations are gibberish "
            "and the judge pools are not vendored; plan §4.5 / deferred-concern)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build + log each phase's per-cell commands, write the sentinel, and "
            "emit [phase=done] WITHOUT launching the subprocesses — exercises the "
            "cell-iteration / env-injection / sentinel / poll-contract plumbing on "
            "CPU (GPU-bound-phase carve-out item 2)."
        ),
    )
    parser.add_argument(
        "--floor-only",
        action="store_true",
        help="Existing-artifact floor only (seed-42 cells); auto-descope fallback.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the per-cell HF upload (local smoke; default uploads per cell).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help=(
            "Skip the analyze-phase Sonnet judge over raw_completions (CPU smoke / "
            "no ANTHROPIC_API_KEY); run only the v-space f_CV analysis."
        ),
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[7, 14, 21], help="Read/patch layers."
    )
    parser.add_argument("--primary-layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Teacher-forced base-response generation cap for the v read (#651 default).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )

    # `uv run python` does NOT auto-load .env; without this a fresh dispatcher
    # spawns subprocesses with HF_TOKEN/ANTHROPIC_API_KEY missing even though
    # every subprocess gets env={**os.environ} (the env dict came from an
    # unloaded parent). load_dotenv() at main()-top is the contract (#397 r10').
    from dotenv import load_dotenv

    load_dotenv()

    repo_root = _resolve_repo_root()
    phases = list(args.phase)
    if "all" in phases:
        phases = ["vendor", "canary", "sweep"]

    cpu_only = args.cpu_only
    smoke = cpu_only or args.smoke_model is not None
    dry_run = args.dry_run
    upload = not args.no_upload and not smoke and not dry_run

    # Credential assert only when a phase needs HF (canary/sweep). Skip for a
    # pure CPU smoke, the dry-run plumbing smoke, and a local analyze.
    if any(p in ("canary", "sweep") for p in phases) and not smoke and not dry_run:
        _require_credentials()

    if "sweep" in phases and not dry_run:
        panel_personas_json, panel_questions_json = _materialize_panel(repo_root)
    else:
        panel_personas_json = panel_questions_json = None

    for phase in phases:
        if phase == "vendor":
            phase_vendor(repo_root)
        elif phase == "canary":
            if dry_run:
                logger.info("[dry-run] canary -> scripts/issue697_canary.py (skipped)")
                phase_log("canary")
                logger.info("[phase=canary_done]")
                continue
            phase_canary(repo_root, cpu_only=cpu_only, smoke_model=args.smoke_model)
        elif phase == "sweep":
            cells = _select_cells(args)
            phase_sweep(
                repo_root,
                cells,
                n_gpus=args.n_gpus,
                cpu_only=cpu_only,
                panel_personas_json=panel_personas_json,
                panel_questions_json=panel_questions_json,
                layers=args.layers,
                primary_layer=args.primary_layer,
                max_new_tokens=args.max_new_tokens,
                skip_e=args.skip_e or smoke,
                smoke_model=args.smoke_model,
                dry_run=dry_run,
                upload=upload,
            )
        elif phase == "analyze":
            # CPU smoke / no-API-key: skip the Sonnet judge step (the tiny-model
            # generations are gibberish and there may be no ANTHROPIC_API_KEY).
            phase_analyze(
                repo_root,
                primary_layer=args.primary_layer,
                skip_judge=args.skip_judge or smoke,
            )

    note = (
        f"phases={phases} cells={args.cells or 'full(128)'} n_gpus={args.n_gpus} "
        f"smoke={smoke} dry_run={dry_run} upload={upload}"
    )
    write_sentinel(
        "epm:results",
        note,
        extra={"phases": phases, "smoke": smoke, "dry_run": dry_run, "n_gpus": args.n_gpus},
    )
    logger.info("[phase=done]")  # terminal marker — reserved for this single line
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
