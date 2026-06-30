"""Issue #651 dispatcher — cross-behavior, cross-context shared-direction geometry.

PASS_UNIFIED architectural-parity (Step 6d.0): smoke IS sweep with one cell.
Every phase runs each cell through the SAME per-cell subprocess function the
full sweep uses; ``--cells <one>`` (+ ``--max-train-steps 2`` for the retrain
phase) is the only difference. No subprocess-vs-in-process divergence.

Phases (plan §4.1 dependency DAG):

- ``retrain``  (4x H100, ~14 GPU-h): the ONLY new training — 32 adapters,
  em x 16 + sycophancy x 16 at seed 1042, under #537's EXACT recipe + frozen
  per-cell JSONLs from HF. Per-cell dose-match admission gate vs the seed-42
  twin's diagonal G is computed downstream (off-pod) from #537's G_meta — this
  phase only trains + uploads.
- ``canary``   (1x H100, ~0.3 GPU-h): Gate 7a (committed-reference reproduction
  on the #519 villain-source marker adapter) + Gate 7b (delta_v adapter-
  application assert on the two #537 loader layouts). BOTH must PASS before the
  full extraction sweep launches. Gate 7b doubles as the smoke-architecture
  canary (the two cells run the full re-extraction path).
- ``extract``  (4x H100, ~6 GPU-h): per-cell layer-{7,14,21} residual shift
  (trained - base), slot + mean-over-response, on the FIXED #551 14-persona x
  20-question panel. Persists per-cell .pt + uploads to HF before pod terminate.
- ``bridge``   (4x H100, ~3 GPU-h): construct-validity bridge for fact +
  sycophancy — the same layer-14 shift on each behavior's #537 canonical
  elicitation surface; cos(U1_neutral, U1_canonical) per behavior.
- ``analysis`` (CPU, 0 GPU, OFF-POD on the VM): SVD / Q1 / Q2 / variance /
  nulls / seed-ceiling / figures over the HF-uploaded per-cell tensors. Runs
  via ``scripts/issue651_analysis.py`` after the pod terminates.

Pod-side contract (CLAUDE.md / poll_pipeline.py): emits ``[phase=<name>]`` log
lines, a terminal ``[phase=done]``, and an end-of-run sentinel JSON at
``/workspace/logs/issue-651-<kind_slug>-<epoch>.json`` carrying
``_SENTINEL_REQUIRED_KEYS`` (sentinel_schema_version / kind / version).

Single-variable-change (plan §13): the ONLY new training variable vs #537 is
``seed=1042`` for em + sycophancy. Every other hyperparameter / prompt / row /
negative / dose target / marker token / panel is inherited byte-identically.
The em recipe is the Hydra ``condition=i537_em training=turner_em lora=turner_em``
path (NOT train_lora — that path cannot express turner_em's schedule); the
sycophancy recipe is ``train_lora`` with #537's #411 sycophancy kwargs. Both
recipes verified against the cell's own adapter_config.json on HF (#545).
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

logger = logging.getLogger("issue651_dispatch")

# #537 recipe constants — inherited byte-identical (plan §11 / §4.4). The
# sycophancy kwargs are #537's JUDGE_TRAIN_KWARGS["sycophancy"] (verified
# against i537_sycophancy_default_seed42/adapter_config.json: r=32 alpha=64
# rsLoRA drop=0.05 all-7-modules, lr=1e-5 cosine, 3 epochs).
SYCOPHANCY_TRAIN_KWARGS = dict(
    lr=1e-5,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    epochs=3,
    batch_size=4,
    grad_accum=4,
    warmup_ratio=0.05,
    report_to="wandb",
)
# em max_steps (plan §4.4 / §11 — #537 turner_em).
EM_MAX_STEPS = 375

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# #537's frozen per-cell training mixes on the data repo (the /data/ segment is
# load-bearing — verified at implementation time; the no-/data/ path 404s).
DATA_TRAIN_PREFIX = "issue537_context_generalization/data/train"


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
        "task_id": 651,
        "by": "issue651_dispatch",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    if extra:
        payload.update(extra)
    slug = kind.replace(":", "_")
    out = _log_dir() / f"issue-651-{slug}-{time.time_ns()}.json"
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
    credential env (HF_TOKEN / WANDB_API_KEY) must be present even though `uv
    run python` does not auto-load .env — load_dotenv() in main() puts it in
    os.environ first (plan gotcha: #397 round-10').
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
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY missing -- load_dotenv() found no .env?"


# ---------------------------------------------------------------------------
# Phase: RETRAIN (the only new training; em + sycophancy at seed 1042)
# ---------------------------------------------------------------------------


def _download_cell_data(repo_root: Path, behavior: str, cid: str) -> tuple[Path, int]:
    """Download #537's frozen <cid>_seed42.jsonl + .meta.json for one cell.

    Returns (local_jsonl_path, max_length). The seed in the filename is the
    frozen DATA seed (42) — the rows are seed-independent, so seed-1042 trains
    on the IDENTICAL rows (the single new variable is the trainer RNG, plan §4.4).
    max_length comes from the cell's frozen meta.json + 128 headroom (#537's
    _builder_cap: the builder asserts the joint-template cap, the trainer
    tokenizes prompt+completion separately and can count slightly larger).
    """
    from huggingface_hub import hf_hub_download

    jsonl = hf_hub_download(
        HF_DATA_REPO,
        f"{DATA_TRAIN_PREFIX}/{behavior}/{cid}_seed42.jsonl",
        repo_type="dataset",
    )
    meta_path = hf_hub_download(
        HF_DATA_REPO,
        f"{DATA_TRAIN_PREFIX}/{behavior}/{cid}_seed42.meta.json",
        repo_type="dataset",
    )
    meta = json.loads(Path(meta_path).read_text())
    max_length = int(meta["max_length"]) + 128
    return Path(jsonl), max_length


def _em_train_cmd(
    repo_root: Path, cell, *, smoke: bool, steps: int | None
) -> tuple[list[str], Path, dict[str, str]]:
    """Build (cmd, log_path, env) for one em cell via the Hydra turner_em path.

    #537's exact invocation (condition=i537_em training=turner_em lora=turner_em).
    EPM_PERSIST_ADAPTER_{HF_REPO,SUBFOLDER} durably upload the LoRA adapter (lands
    NESTED under <subfolder>/sft_em_adapter/); upload_to=none suppresses the
    ~15 GB merged push. CUDA_VISIBLE_DEVICES is pinned in the LAUNCHER env (NOT
    only via +gpu_id) so an import-time cuInit can't co-locate cells on GPU 0
    (#545); the matching +gpu_id rewrites the same value in-process.
    """
    data_path, max_length = _download_cell_data(repo_root, "em", cell.cid)
    eff_steps = steps if steps is not None else (2 if smoke else EM_MAX_STEPS)
    subfolder = f"adapters/i537_em_{cell.cid}_seed{cell.seed}"
    out_root = repo_root / "outputs" / "issue_651" / f"em_{cell.cid}_seed{cell.seed}"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train.py",
        "condition=i537_em",
        "training=turner_em",
        "lora=turner_em",
        "upload_to=none",
        f"+training.max_steps={eff_steps}",
        f"training.max_seq_length={max_length}",
        f"seed={cell.seed}",
        f"+gpu_id={cell.gpu_id}",
        f"condition.name=i537_em_{cell.cid}",
        f"condition.stages.0.dataset={data_path}",
        f"output_dir={out_root}",
    ]
    env = {"CUDA_VISIBLE_DEVICES": str(cell.gpu_id), "EPM_SKIP_INLINE_CHECKPOINT_UPLOAD": "1"}
    if not smoke:
        env["EPM_PERSIST_ADAPTER_HF_REPO"] = HF_MODEL_REPO
        env["EPM_PERSIST_ADAPTER_SUBFOLDER"] = subfolder
    log_path = _log_dir() / f"retrain_em_{cell.cid}_seed{cell.seed}.log"
    return cmd, log_path, env


def _syc_train_cmd(
    repo_root: Path, cell, *, smoke: bool, steps: int | None
) -> tuple[list[str], Path, dict[str, str]]:
    """Build (cmd, log_path, env) for one sycophancy cell via train_lora (#411).

    A subprocess (not in-process) so each cell pins its own CUDA_VISIBLE_DEVICES
    in the launcher env (#545) and waves run truly in parallel — the same shape
    as the em cells, keeping smoke == sweep. train_lora has no max_steps field,
    so the smoke path slices the JSONL via --smoke-max-rows (epochs forced 1).
    """
    data_path, max_length = _download_cell_data(repo_root, "sycophancy", cell.cid)
    out_dir = repo_root / "outputs" / "issue_651" / f"sycophancy_{cell.cid}_seed{cell.seed}"
    subfolder = f"adapters/i537_sycophancy_{cell.cid}_seed{cell.seed}"
    overrides = {
        **SYCOPHANCY_TRAIN_KWARGS,
        "seed": cell.seed,
        "gpu_id": cell.gpu_id,
        "max_length": max_length,
        "run_name": f"i537_sycophancy_{cell.cid}_seed{cell.seed}",
        "hf_upload": (not smoke),
        "hf_path_in_repo": subfolder,
    }
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue651_train_sycophancy_cell.py",
        "--base-model",
        QWEN_ID,
        "--data-path",
        str(data_path),
        "--out-dir",
        str(out_dir),
        "--overrides-json",
        json.dumps(overrides),
        "--hf-repo",
        HF_MODEL_REPO,
    ]
    if smoke:
        # train_lora has no max_steps; a tiny real slice + epochs=1 is the smoke.
        cmd += ["--smoke-max-rows", str(steps if steps is not None else 8)]
    env = {"CUDA_VISIBLE_DEVICES": str(cell.gpu_id), "EPM_SKIP_INLINE_CHECKPOINT_UPLOAD": "1"}
    log_path = _log_dir() / f"retrain_sycophancy_{cell.cid}_seed{cell.seed}.log"
    return cmd, log_path, env


def phase_retrain(
    repo_root: Path,
    cells: Sequence,
    *,
    n_gpus: int,
    smoke: bool,
    max_train_steps: int | None,
    dry_run: bool = False,
) -> None:
    """Train the seed-1042 em + sycophancy cells (wave-parallel, CVD-pinned)."""
    from explore_persona_space.experiments.issue_651 import RETRAIN_SEED

    phase_log("retrain")
    for wave_start in range(0, len(cells), max(n_gpus, 1)):
        wave = cells[wave_start : wave_start + max(n_gpus, 1)]
        cmds: list[tuple[Sequence[str], Path, dict[str, str] | None]] = []
        for cell in wave:
            assert cell.seed == RETRAIN_SEED, cell
            if cell.behavior == "em":
                cmd, log_path, env = _em_train_cmd(
                    repo_root, cell, smoke=smoke, steps=max_train_steps
                )
            else:  # sycophancy
                cmd, log_path, env = _syc_train_cmd(
                    repo_root, cell, smoke=smoke, steps=max_train_steps
                )
            cmds.append((cmd, log_path, env))
        if dry_run:
            for (cmd, _lp, env), cell in zip(cmds, wave, strict=True):
                logger.info(
                    "[dry-run] retrain %s CVD=%s :: %s",
                    cell.cell_id,
                    env.get("CUDA_VISIBLE_DEVICES"),
                    " ".join(shlex.quote(c) for c in cmd),
                )
            continue
        rcs = _run_parallel_with_log(cmds, cwd=repo_root)
        bad = [(rc, c.cell_id) for rc, c in zip(rcs, wave, strict=True) if rc != 0]
        if bad:
            raise RuntimeError(f"retrain wave failed: {bad}; see logs in {_log_dir()}")
        for c in wave:
            logger.info("retrain cell %s complete", c.cell_id)  # NOT [phase=done] (mid-run noise)
    logger.info("[phase=retrain_done]")


# ---------------------------------------------------------------------------
# Phase: EXTRACT (per-cell residual shift on the fixed panel)
# ---------------------------------------------------------------------------


def phase_extract(
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
    dry_run: bool = False,
) -> None:
    """Extract per-cell shifts via the inherited activation_shift CLI (waves)."""
    phase_log("extract")
    out_dir = repo_root / "eval_results" / "issue_651" / "shifts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for wave_start in range(0, len(cells), max(n_gpus, 1)):
        wave = cells[wave_start : wave_start + max(n_gpus, 1)]
        cmds: list[tuple[Sequence[str], Path, dict[str, str] | None]] = []
        for cell in wave:
            # Stage the adapter LOCALLY first (per-file download). The model
            # repo is >14k files, so snapshot_download(allow_patterns=...) /
            # passing an HF-subfolder string to PeftModel silently truncates +
            # produces an unapplied adapter (#375/#399); the staged local dir
            # has config + safetensors flattened at its root.
            if dry_run:
                adapter = f"<staged:{cell.adapter_subfolder}>"  # skip the 323MB pull
            else:
                from explore_persona_space.experiments.issue_651 import stage_adapter

                adapter = str(
                    stage_adapter(
                        cell.adapter_subfolder,
                        repo_root / "outputs" / "issue_651" / "staged_adapters",
                    )
                )
            shift_out = out_dir / f"{cell.cell_id}.pt"
            # activation_shift --arm expects marker|em|fact|refusal; map
            # sycophancy/emnc onto the generative ("em") marker-stripping=off
            # path (only arm=marker strips trailing tokens — plan §4.3 / the
            # extractor's _question_deltas branch).
            arm = {"marker": "marker", "fact": "fact", "refusal": "refusal"}.get(
                cell.behavior, "em"
            )
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "explore_persona_space.analysis.activation_shift",
                "--arm",
                arm,
                "--seed",
                str(cell.seed),
                "--variant",
                "base",  # #602 teacher-forced base-trajectory read
                "--family",
                cell.behavior,
                "--layers",
                *[str(L) for L in layers],
                "--primary-layer",
                str(primary_layer),
                "--adapter-path",
                adapter,
                "--personas-json",
                str(panel_personas_json),
                "--questions-json",
                str(panel_questions_json),
                "--out",
                str(shift_out),
                "--max-new-tokens",
                str(max_new_tokens),
                "--base-model-id",
                QWEN_ID,
            ]
            env = {"CUDA_VISIBLE_DEVICES": "" if cpu_only else str(cell.gpu_id)}
            log_path = _log_dir() / f"extract_{cell.cell_id}.log"
            cmds.append((cmd, log_path, env))
        if dry_run:
            for (cmd, _lp, env), cell in zip(cmds, wave, strict=True):
                logger.info(
                    "[dry-run] extract %s CVD=%r :: %s",
                    cell.cell_id,
                    env.get("CUDA_VISIBLE_DEVICES"),
                    " ".join(shlex.quote(c) for c in cmd),
                )
            continue
        rcs = _run_parallel_with_log(cmds, cwd=repo_root)
        bad = [(rc, c.cell_id) for rc, c in zip(rcs, wave, strict=True) if rc != 0]
        if bad:
            raise RuntimeError(f"extract wave failed: {bad}; see logs in {_log_dir()}")
        for c in wave:
            logger.info("extract cell %s complete", c.cell_id)  # NOT [phase=done]
    if dry_run:
        logger.info("[phase=extract_done] (dry-run: no tensors written, upload skipped)")
        return
    # Upload per-cell tensors to the HF data repo before pod terminate
    # (Upload Policy: intermediate analysis tensors the analysis phase consumes).
    _upload_shift_tensors(out_dir)
    logger.info("[phase=extract_done]")


def _upload_shift_tensors(shift_dir: Path) -> None:
    """Upload per-cell .pt tensors to the HF data repo (analysis-input contract).

    Per Upload Policy: intermediate analysis tensors the plan's analysis phase
    references as downstream inputs MUST land on HF before pod terminate (#521).
    Fail-loud (the helper raises on any per-file upload failure).
    """
    if os.environ.get("EPM_SKIP_UPLOAD") == "1":
        logger.info("EPM_SKIP_UPLOAD=1 -> skipping tensor upload (smoke/local)")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    pts = sorted(shift_dir.glob("*.pt"))
    if not pts:
        raise RuntimeError(f"no .pt tensors to upload in {shift_dir} -- extraction wrote nothing")
    from huggingface_hub import CommitOperationAdd

    ops = [
        CommitOperationAdd(
            path_in_repo=f"issue651_cross_behavior_geometry/analysis_tensors/{p.name}",
            path_or_fileobj=str(p),
        )
        for p in pts
    ]
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue651: {len(ops)} per-cell shift tensors",
    )
    # Verify on the Hub before trusting the pod can terminate.
    from huggingface_hub import list_repo_files

    files = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    missing = [
        p.name
        for p in pts
        if f"issue651_cross_behavior_geometry/analysis_tensors/{p.name}" not in files
    ]
    if missing:
        raise RuntimeError(f"tensor upload verification FAILED -- missing on Hub: {missing}")
    logger.info("uploaded + verified %d shift tensors to %s", len(pts), HF_DATA_REPO)


# ---------------------------------------------------------------------------
# Phase: CANARY (Gate 7a committed-reference + Gate 7b loader-branch)
# ---------------------------------------------------------------------------


def phase_canary(repo_root: Path, *, cpu_only: bool) -> None:
    """Run Gate 7a + 7b via the dedicated canary script; HALT on FAIL."""
    phase_log("canary")
    cmd = ["uv", "run", "python", "scripts/issue651_canary.py"]
    if cpu_only:
        cmd.append("--cpu-only")
    env = {"CUDA_VISIBLE_DEVICES": "" if cpu_only else "0"}
    log_path = _log_dir() / "canary.log"
    rc = _run_with_log(cmd, log_path=log_path, extra_env=env, cwd=repo_root)
    if rc != 0:
        raise RuntimeError(
            f"CANARY FAILED (rc={rc}) -- Gate 7a/7b did not PASS; HALT before the sweep. "
            f"See {log_path}"
        )
    logger.info("[phase=canary_done]")


# ---------------------------------------------------------------------------
# Phase: BRIDGE (construct-validity bridge for fact + sycophancy)
# ---------------------------------------------------------------------------


def phase_bridge(
    repo_root: Path,
    cells: Sequence,
    *,
    n_gpus: int,
    cpu_only: bool,
    primary_layer: int,
    max_new_tokens: int,
) -> None:
    """Construct-validity bridge read for fact + sycophancy (plan §6.1)."""
    phase_log("bridge")
    # Only fact + sycophancy seed-42 cells need the bridge (em + marker are
    # already behavior-validated by #521/#551/#552).
    bridge_cells = [c for c in cells if c.behavior in ("fact", "sycophancy") and c.seed == 42]
    if not bridge_cells:
        logger.info("[phase=bridge_done] (no fact/sycophancy seed-42 cells in subset)")
        return
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue651_bridge.py",
        "--cells",
        *[c.cell_id for c in bridge_cells],
        "--primary-layer",
        str(primary_layer),
        "--max-new-tokens",
        str(max_new_tokens),
        "--n-gpus",
        str(n_gpus),
    ]
    if cpu_only:
        cmd.append("--cpu-only")
    log_path = _log_dir() / "bridge.log"
    rc = _run_with_log(cmd, log_path=log_path, cwd=repo_root)
    if rc != 0:
        raise RuntimeError(f"bridge phase failed (rc={rc}); see {log_path}")
    logger.info("[phase=bridge_done]")


# ---------------------------------------------------------------------------
# Phase: ANALYSIS (CPU, off-pod — delegate to issue651_analysis.py)
# ---------------------------------------------------------------------------


def phase_analysis(repo_root: Path) -> None:
    """Off-pod CPU SVD/Q1/Q2/variance/nulls/seed-ceiling (issue651_analysis.py)."""
    phase_log("analysis")
    cmd = ["uv", "run", "python", "scripts/issue651_analysis.py"]
    log_path = _log_dir() / "analysis.log"
    rc = _run_with_log(cmd, log_path=log_path, cwd=repo_root)
    if rc != 0:
        raise RuntimeError(f"analysis phase failed (rc={rc}); see {log_path}")
    logger.info("[phase=analysis_done]")


# ---------------------------------------------------------------------------
# Panel JSON materialization
# ---------------------------------------------------------------------------


def _materialize_panel(repo_root: Path) -> tuple[Path, Path]:
    """Write the fixed panel personas + questions JSON the extractor CLI reads.

    SHA-pinned at Phase 0 by recording the panel column order in the manifest;
    the extractor asserts the row order against the constant via persona_order.
    """
    from explore_persona_space.experiments.issue_651 import (
        build_panel_personas,
        build_panel_questions,
    )

    panel_dir = repo_root / "eval_results" / "issue_651" / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    personas = build_panel_personas()
    questions = build_panel_questions()
    p_path = panel_dir / "panel_personas.json"
    q_path = panel_dir / "panel_questions.json"
    p_path.write_text(json.dumps(personas, indent=2))
    q_path.write_text(json.dumps(questions, indent=2))
    logger.info("panel materialized: %d personas, %d questions", len(personas), len(questions))
    return p_path, q_path


# ---------------------------------------------------------------------------
# Cell selection
# ---------------------------------------------------------------------------


def _select_cells(args, *, for_phase: str):
    """Resolve the cell subset for a phase from --cells (or the full sweep).

    Per-phase cell-list source (PASS_UNIFIED parity): retrain reads the retrain
    grid; extract/bridge read the readable grid; each is filtered by the SAME
    --cells subset so a smoke is the sweep with one cell.
    """
    from explore_persona_space.experiments.issue_651 import (
        parse_cell_spec,
        readable_cells,
        retrain_cells,
    )

    if for_phase == "retrain":
        full = retrain_cells(n_gpus=args.n_gpus)
    else:
        full = readable_cells(n_gpus=args.n_gpus, include_seed1042=not args.floor_only)
    if args.cells:
        requested = [parse_cell_spec(s) for s in args.cells]
        avail = {(c.behavior, c.cid, c.seed) for c in full}
        unknown = [r.cell_id for r in requested if (r.behavior, r.cid, r.seed) not in avail]
        if unknown:
            raise ValueError(
                f"--cells {unknown!r} not in the {for_phase} grid "
                f"(behaviors/seeds available: {sorted({(c.behavior, c.seed) for c in full})})"
            )
        # Re-densify gpu_id over the requested subset.
        from explore_persona_space.experiments.issue_651 import Cell

        sel = [
            Cell(behavior=r.behavior, cid=r.cid, seed=r.seed, gpu_id=i % max(args.n_gpus, 1))
            for i, r in enumerate(requested)
        ]
        return sel
    return full


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #651 dispatcher (retrain / canary / extract / bridge / analysis).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=["retrain", "canary", "extract", "bridge", "analysis", "all"],
        default=["all"],
        help="Phases to run in order. 'all' = canary -> retrain -> extract -> bridge.",
    )
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Cell subset (e.g. 'em_default_seed1042'); smoke = sweep with one cell.",
    )
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU (smoke).")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode (max_train_steps=2).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build + log each phase's per-cell commands, write the sentinel, and emit "
            "[phase=done] WITHOUT launching the GPU subprocesses — exercises the "
            "cell-iteration / env-injection / sentinel / poll-contract plumbing on CPU "
            "(GPU-bound-phase carve-out item 2)."
        ),
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Override training steps (smoke uses 2). em default 375; syc default 3 epochs.",
    )
    parser.add_argument(
        "--floor-only",
        action="store_true",
        help="Existing-artifact floor only (no seed-1042 cells); auto-descope fallback.",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[7, 14, 21], help="Extraction layers."
    )
    parser.add_argument("--primary-layer", type=int, default=14)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Base-response generation cap for the teacher-forced read (#551 default).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )

    # `uv run python` does NOT auto-load .env; without this a fresh dispatcher
    # spawns subprocesses with HF_TOKEN/WANDB_API_KEY missing even though every
    # subprocess gets env={**os.environ} (the env dict came from an unloaded
    # parent). load_dotenv() at main()-top is the contract (#397 round-10').
    # DOTENV_LINT_EXEMPT: legacy pre-#745 script; shell exports cover pod/GCE/SLURM.
    from dotenv import load_dotenv

    load_dotenv()

    repo_root = _resolve_repo_root()
    phases = list(args.phase)
    if "all" in phases:
        phases = ["canary", "retrain", "extract", "bridge"]

    cpu_only = args.cpu_only
    smoke = args.smoke or cpu_only
    dry_run = args.dry_run

    # Credential assert only when a phase needs HF/WandB (canary/extract/retrain/
    # bridge all do; analysis is local-only). Skip for a pure CPU analysis run
    # and for the dry-run plumbing smoke.
    if any(p != "analysis" for p in phases) and not (smoke and cpu_only) and not dry_run:
        _require_credentials()

    if any(p in ("extract", "bridge") for p in phases) and not dry_run:
        panel_personas_json, panel_questions_json = _materialize_panel(repo_root)
    else:
        panel_personas_json = panel_questions_json = None

    for phase in phases:
        if phase == "canary":
            if dry_run:
                logger.info("[dry-run] canary -> scripts/issue651_canary.py (skipped)")
                phase_log("canary")
                logger.info("[phase=canary_done]")
                continue
            phase_canary(repo_root, cpu_only=cpu_only)
        elif phase == "retrain":
            cells = _select_cells(args, for_phase="retrain")
            phase_retrain(
                repo_root,
                cells,
                n_gpus=args.n_gpus,
                smoke=smoke,
                max_train_steps=args.max_train_steps,
                dry_run=dry_run,
            )
        elif phase == "extract":
            cells = _select_cells(args, for_phase="extract")
            phase_extract(
                repo_root,
                cells,
                n_gpus=args.n_gpus,
                cpu_only=cpu_only,
                panel_personas_json=panel_personas_json,
                panel_questions_json=panel_questions_json,
                layers=args.layers,
                primary_layer=args.primary_layer,
                max_new_tokens=args.max_new_tokens,
                dry_run=dry_run,
            )
        elif phase == "bridge":
            if dry_run:
                logger.info("[dry-run] bridge -> scripts/issue651_bridge.py (skipped)")
                phase_log("bridge")
                logger.info("[phase=bridge_done]")
                continue
            cells = _select_cells(args, for_phase="bridge")
            phase_bridge(
                repo_root,
                cells,
                n_gpus=args.n_gpus,
                cpu_only=cpu_only,
                primary_layer=args.primary_layer,
                max_new_tokens=args.max_new_tokens,
            )
        elif phase == "analysis":
            phase_analysis(repo_root)

    note = f"phases={phases} cells={args.cells or 'full'} smoke={smoke} dry_run={dry_run}"
    write_sentinel(
        "epm:results", note, extra={"phases": phases, "smoke": smoke, "dry_run": dry_run}
    )
    logger.info("[phase=done]")  # terminal marker — reserved for this single line
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
