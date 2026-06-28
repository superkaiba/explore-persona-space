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

# ruff: noqa: RUF002, RUF003  # math/scientific notation in docstrings + messages

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
# followup a36-readout-reextract-cos: the re-extracted read-out r⁺ store + its
# HF prefix (a NEW prefix so #667's existing analysis_tensors are untouched).
HF_R_PLUS_PREFIX = "issue667_gate_chain_preview/a36_readout_reextract/r_plus"
R_PLUS_DIR = "eval_results/issue_667/a36_readout_reextract/r_plus"
REEXTRACT_OUT_DIR = "eval_results/issue_667/a36_readout_reextract"
# The extractor defaults --seed to 42 and the dispatcher never overrides it, so a
# cell's output dir is <TENSORS_DIR>/<behavior>/<source>_seed42 (issue667_extract
# cell_dir). Used by the resume-skip check (round-7) to detect already-extracted
# cells on relaunch.
_EXTRACT_SEED = 42


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


def _git_commit_sha() -> str:
    """Best-effort HEAD sha for the run sentinel (reproducibility metadata)."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, env={**os.environ}
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


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
    from issue667_analysis import (
        assert_store_pin,
        load_g_meta,
        load_sigma_c_dict,
        validate_r_b_coverage,
        validate_sigma_c_coverage,
    )

    g_meta = load_g_meta()
    logger.info("G_meta git_commit pin OK: %s", g_meta["git_commit"][:16])
    assert_store_pin()
    logger.info("#658 store probe_pool_hash pin OK: %s", EXPECTED_STORE_PROBE_POOL_HASH[:16])
    logger.info("EXPECTED_G_META_GIT_COMMIT=%s", EXPECTED_G_META_GIT_COMMIT[:16])

    # Standalone cached-artifact coverage validators (BLOCKER 3): r_b columns/
    # recipe/layers + sigma_c shape/keys/layers — checkable pre-extract. The
    # cell-dependent G_meta-per-cell + cid-coverage checks run in the analysis
    # phase (they need the realized cell set). Fail loud here on any miss.
    from explore_persona_space.analysis.issue667 import ALL_LAYERS

    in_scope_layers = sorted(set(ALL_LAYERS))
    validate_r_b_coverage(behaviors, in_scope_layers)
    validate_sigma_c_coverage(load_sigma_c_dict(), in_scope_layers)
    logger.info("standalone coverage validation PASS (r_b columns/recipe/layers, sigma_c shape)")

    # rsLoRA parity probe (fitness check (g)) — 1 adapter reproduces #537's
    # diagonal source write at the committed gauge. HALT on mismatch.
    if skip_parity:
        logger.info("parity probe SKIPPED (--skip-parity)")
    else:
        _rslora_parity_probe(behaviors[0], cpu_only=cpu_only)
    logger.info("[phase=prefetch_done]")


# Minimum diagonal-write magnitude ratio ‖Δv(C)‖/‖v0(C)‖ for a real rsLoRA
# application: a no-op / wrong-gauge adapter leaves the residual ~unchanged
# (ratio ~0). #537's contrastive adapters move the diagonal source write
# materially (rsLoRA α/√r at the committed gauge); a ratio below this floor
# means the adapter is applied under a DIFFERENT gauge than #537 committed.
PARITY_MIN_WRITE_RATIO = 0.01


def _numeric_rslora_parity(behavior: str, source: str = "default", seed: int = 42) -> dict:
    """NUMERIC rsLoRA parity probe (BLOCKER 2): GPU diagonal-write reproduction.

    Stages 1 adapter, applies it via PeftModel (rsLoRA honored), runs the SAME
    teacher-forced diagonal extraction the production sweep uses for the source
    cell C→C (a few probes), and computes the realized source write
    ``Δv(C) = v+(C) − v0(C)``. Asserts the write-DIRECTION parity numerically:

    - ``g_real(C, C) == 1`` (structural self-gate invariant — the source write
      is well-defined; a degenerate/zero write fails this), and
    - ``‖Δv(C)‖ / ‖v0(C)‖ >= PARITY_MIN_WRITE_RATIO`` — the adapter actually
      moves the residual at the committed gauge (a no-op / wrong-gauge adapter
      reads ~0). HALT on either miss.

    This is the numeric reproduction-and-HALT gate plan §5(g)/§7 mandate — NOT a
    gauge config check (round-1's mistake). Returns the measured magnitudes.
    """
    import numpy as np
    import torch
    from issue667_extract import (
        _device,
        _greedy_response,
        _mean_resp_acts,
        assert_adapter_gauge,
        build_messages_for,
        load_base_and_trained,
        load_eval_probes,
        stage_adapter_local,
        stage_inputs,
    )

    from explore_persona_space.analysis.issue667 import BASE_MODEL, PRIMARY_LAYER
    from explore_persona_space.analysis.issue667.gate_chain import realized_gate
    from explore_persona_space.experiments.i537_contexts import load_icl_demos, load_registry

    adapter_dir = stage_adapter_local(behavior, source, seed)
    gauge = assert_adapter_gauge(adapter_dir, behavior)
    sampled_path, demos_path = stage_inputs()
    registry = load_registry(sampled_path)
    demos = load_icl_demos(demos_path)
    device = _device(0, cpu_only=False)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    _, base, trained = load_base_and_trained(adapter_dir, device, dtype)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
    probes = load_eval_probes(behavior)[:3]  # 3 probes — cheap, ~30s
    v0s, vps = [], []
    for q in probes:
        msgs = build_messages_for(registry, demos, source, behavior, q)
        r = _greedy_response(base, tok, msgs, device, 256)
        if not r.strip():
            continue
        acts = _mean_resp_acts(base, trained, tok, msgs, r, [PRIMARY_LAYER], device)
        v0, vp = acts[PRIMARY_LAYER]
        v0s.append(v0)
        vps.append(vp)
    if not v0s:
        raise RuntimeError(f"parity probe: no diagonal probe produced a response for {behavior}")
    v0 = np.stack(v0s).mean(axis=0).astype(np.float64)
    vp = np.stack(vps).mean(axis=0).astype(np.float64)
    g_self, _ = realized_gate(v0, vp, v0, vp)  # self-gate must be exactly 1
    write_norm = float(np.linalg.norm(vp - v0))
    base_norm = float(np.linalg.norm(v0))
    ratio = write_norm / base_norm if base_norm > 0 else 0.0
    del base, trained
    if device.type == "cuda":
        torch.cuda.empty_cache()
    assert abs(g_self - 1.0) < 1e-4, (
        f"parity probe: self-gate g_real(C,C)={g_self:.6f} != 1 — source write degenerate"
    )
    if ratio < PARITY_MIN_WRITE_RATIO:
        raise RuntimeError(
            f"rsLoRA NUMERIC parity FAILED: diagonal write ratio "
            f"‖Δv‖/‖v0‖={ratio:.5f} < {PARITY_MIN_WRITE_RATIO} for {behavior}/{source} — "
            "the adapter is applied under a DIFFERENT gauge than #537 committed (or is a "
            "no-op). HALT before the full sweep (plan §5(g)/§7)."
        )
    result = {
        "behavior": behavior,
        "source": source,
        "g_self": g_self,
        "write_norm": write_norm,
        "base_norm": base_norm,
        "write_ratio": ratio,
        "gauge": {k: gauge[k] for k in ("r", "lora_alpha", "use_rslora")},
        "n_probes": len(v0s),
    }
    logger.info(
        "rsLoRA NUMERIC parity PASS: %s/%s g_self=%.6f ‖Δv‖/‖v0‖=%.4f (gauge=%s)",
        behavior,
        source,
        g_self,
        ratio,
        result["gauge"],
    )
    return result


def _rslora_parity_probe(behavior: str, *, cpu_only: bool) -> None:
    """rsLoRA parity probe — NUMERIC on GPU (BLOCKER 2), config-only on CPU smoke.

    On GPU: runs :func:`_numeric_rslora_parity` in a ONE-SHOT SUBPROCESS (the
    ``parity-probe`` CLI entrypoint below) so the GPU forward NEVER initializes
    CUDA in THIS dispatcher parent process. The dispatcher then forks the per-cell
    extract subprocesses (each of which runs vLLM, forking its own EngineCore
    worker); a CUDA context left live in the dispatcher parent poisons that fork
    chain — ``RuntimeError: Cannot re-initialize CUDA in forked subprocess`` (#667
    r4, bug_class dispatcher_cuda_init_before_subprocess_fork). The HALT gate
    semantics are preserved exactly: the subprocess runs the same numeric
    diagonal-write reproduction + asserts, and a non-zero exit (a failed assert /
    RuntimeError HALT, OR a crash) re-raises here BEFORE any extraction wave.

    On a CPU-only local smoke (no 7B forward): asserts the gauge config in-process
    (no CUDA touched) and defers the numeric reproduction to the GPU path.
    """
    from issue667_extract import assert_adapter_gauge, stage_adapter_local

    if cpu_only:
        adapter_dir = stage_adapter_local(behavior, "default", 42)
        gauge = assert_adapter_gauge(adapter_dir, behavior)
        assert gauge["use_rslora"], "parity probe: adapter is not rsLoRA (gauge mismatch)"
        logger.info(
            "rsLoRA parity probe: CPU-only — gauge config asserted (r=%s alpha=%s "
            "use_rslora=%s); NUMERIC diagonal-write reproduction runs on the GPU path.",
            gauge["r"],
            gauge["lora_alpha"],
            gauge["use_rslora"],
        )
        return
    _run_parity_probe_subprocess(behavior)


def _run_parity_probe_subprocess(behavior: str, source: str = "default", seed: int = 42) -> dict:
    """Run the NUMERIC rsLoRA parity probe in a one-shot subprocess (CUDA isolation).

    Invokes the ``parity-probe`` CLI of THIS module via ``subprocess.run`` so the
    7B forward + PeftModel apply (which initialize CUDA) happen in a CHILD process
    that exits cleanly, leaving the dispatcher parent's CUDA state untouched before
    the extract wave forks (#667 r4). Explicit ``env={**os.environ}`` so the child
    inherits HF_TOKEN/WANDB_API_KEY (load_dotenv() ran at main()-top). HALT
    semantics: a non-zero child rc re-raises here (the gate fired or the probe
    crashed); on success the child's result JSON is read back and returned.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="i667_parity_") as td:
        result_path = Path(td) / "parity_result.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "parity-probe",
            "--behavior",
            behavior,
            "--source",
            source,
            "--seed",
            str(seed),
            "--result-out",
            str(result_path),
        ]
        log_path = _log_dir() / f"parity_probe_{behavior}.log"
        logger.info("rsLoRA parity probe -> one-shot subprocess (CUDA-isolated): %s", behavior)
        rc = _run_with_log(cmd, log_path=log_path)
        if rc != 0:
            raise RuntimeError(
                f"rsLoRA NUMERIC parity probe subprocess exited rc={rc} for {behavior}/{source} "
                f"(HALT — see {log_path}). Either the diagonal-write parity gate fired "
                "(adapter applied under a DIFFERENT gauge than #537 committed) or the probe "
                "crashed; the extract wave does NOT proceed (plan §5(g)/§7)."
            )
        if not result_path.exists():
            raise RuntimeError(
                f"rsLoRA parity probe subprocess exited rc=0 but wrote no result at {result_path} "
                f"(see {log_path}) — treating as a HALT (the gate's PASS is unverified)."
            )
        result = json.loads(result_path.read_text())
        logger.info(
            "rsLoRA NUMERIC parity PASS (subprocess): %s/%s g_self=%.6f write_ratio=%.4f gauge=%s",
            result["behavior"],
            result["source"],
            result["g_self"],
            result["write_ratio"],
            result.get("gauge"),
        )
        return result


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
    # VLLM_WORKER_MULTIPROC_METHOD=spawn: belt-and-suspenders for the extract's
    # vLLM EngineCore fork (gotchas.md § entry 26). The extractor sets this at
    # module top too; injecting it into the per-cell subprocess env guards
    # against a future import-reorder re-poisoning the path (#667 r5).
    env = {
        "CUDA_VISIBLE_DEVICES": "" if cpu_only else str(gpu_id),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    log_path = _log_dir() / f"extract_{behavior}_{source}.log"
    return cmd, log_path, env


def _cell_already_extracted(behavior: str, source: str) -> bool:
    """True ONLY if a prior run wrote this cell's atomic completion sentinel.

    The extractor writes the per-(target, layer) ``<tcid>_L<li>.npz`` files
    INCREMENTALLY, then writes ``.done`` ATOMICALLY only after EVERY planned
    tensor is on disk (``issue667_extract.write_cell_done_sentinel``). A
    mid-cell crash therefore leaves a PARTIAL ``.npz`` set with NO ``.done`` —
    so checking for ``.done`` (not for any stray ``.npz``) is what makes the
    default-ON resume-skip safe: a partial dir is never silently accepted as
    complete (round-8 BLOCKER resume-skip-partial-cell-silent-skip; CLAUDE.md
    "Fail fast — never hide failures"). Round-7 used ``any(*.npz)``, which
    would have silently skipped a partially-extracted cell on relaunch.
    """
    from issue667_extract import CELL_DONE_SENTINEL

    cell_dir = PROJECT_ROOT / TENSORS_DIR / behavior / f"{source}_seed{_EXTRACT_SEED}"
    return (cell_dir / CELL_DONE_SENTINEL).is_file()


def _filter_resume_skip(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop cells whose .npz tensors already exist on disk (round-7 resume-skip).

    A relaunch after a mid-run crash must NOT re-extract completed cells (a
    ~95-min relaunch re-ran 32/64 already-on-disk cells). Returns the cells that
    still need extracting; logs each skip + a one-line kept/skipped summary.
    """
    kept: list[tuple[str, str]] = []
    for behavior, source in cells:
        if _cell_already_extracted(behavior, source):
            logger.info(
                "resume-skip: %s/%s already extracted at %s",
                behavior,
                source,
                PROJECT_ROOT / TENSORS_DIR / behavior / f"{source}_seed{_EXTRACT_SEED}",
            )
            continue
        kept.append((behavior, source))
    if len(kept) != len(cells):
        logger.info(
            "extract: resume-skip kept %d / %d cells (skipped %d already on disk)",
            len(kept),
            len(cells),
            len(cells) - len(kept),
        )
    return kept


def _expected_npz_for_cell(behavior: str, source: str, layers: list[int]) -> set[str]:
    """The full per-(target, layer) ``.npz`` filename set a complete cell holds.

    Mirrors the extractor's default target list (``eval_cids_for(behavior) +
    source``, deduped, source-first) so the backfill cross-check uses the SAME
    complement the extractor would have written.
    """
    from explore_persona_space.experiments.i537_contexts import eval_cids_for

    targets = list(dict.fromkeys([source, *eval_cids_for(behavior)]))
    return {f"{tcid}_L{li}.npz" for tcid in targets for li in layers}


def phase_backfill_sentinels(*, layers: list[int]) -> None:
    """One-shot migration: write ``.done`` for every COMPLETE on-disk cell.

    Round-8: the 32 cells already extracted under the round-7 ``any(*.npz)``
    contract have no ``.done`` sentinel, so the new sentinel-based resume-skip
    would re-extract them. This walks ``<TENSORS_DIR>/<behavior>/<source>_seed42``,
    and for each cell whose on-disk ``.npz`` set EXACTLY matches the expected
    (target, layer) complement, writes the atomic sentinel — so the relaunch can
    safely resume-skip the completed cells. A cell missing ANY expected ``.npz``
    is REPORTED and left without a sentinel (it will be re-extracted), never
    backfilled — fail-loud, never silently accept a partial cell.
    """
    phase_log("backfill_sentinels")
    from issue667_extract import CELL_DONE_SENTINEL, write_cell_done_sentinel

    tdir = PROJECT_ROOT / TENSORS_DIR
    if not tdir.is_dir():
        logger.info("backfill: no tensor dir at %s — nothing to backfill", tdir)
        logger.info("[phase=backfill_done]")
        return
    n_written = n_already = n_incomplete = 0
    for beh_dir in sorted(p for p in tdir.iterdir() if p.is_dir()):
        behavior = beh_dir.name
        for cell_dir in sorted(p for p in beh_dir.iterdir() if p.is_dir()):
            name = cell_dir.name
            suffix = f"_seed{_EXTRACT_SEED}"
            if not name.endswith(suffix):
                logger.warning("backfill: skipping unrecognized cell dir %s", cell_dir)
                continue
            source = name[: -len(suffix)]
            if (cell_dir / CELL_DONE_SENTINEL).is_file():
                n_already += 1
                continue
            expected = _expected_npz_for_cell(behavior, source, layers)
            present = {p.name for p in cell_dir.glob("*.npz")}
            missing = expected - present
            if missing:
                n_incomplete += 1
                logger.warning(
                    "backfill: %s/%s INCOMPLETE — %d/%d expected .npz present "
                    "(missing %d, e.g. %s) — NO sentinel written, cell will re-extract",
                    behavior,
                    source,
                    len(present & expected),
                    len(expected),
                    len(missing),
                    sorted(missing)[:3],
                )
                continue
            from explore_persona_space.experiments.i537_contexts import eval_cids_for

            targets = list(dict.fromkeys([source, *eval_cids_for(behavior)]))
            write_cell_done_sentinel(
                cell_dir,
                behavior=behavior,
                source_cid=source,
                seed=_EXTRACT_SEED,
                targets=targets,
                layers=layers,
            )
            n_written += 1
    logger.info(
        "backfill: wrote %d sentinels, %d already had one, %d incomplete (will re-extract)",
        n_written,
        n_already,
        n_incomplete,
    )
    logger.info("[phase=backfill_done]")


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
    skip_parity: bool = False,
    resume_skip: bool = True,
) -> None:
    """Per-source-adapter extraction in CVD-pinned waves; upload tensors after.

    MANDATORY FIRST STEP (BLOCKER 2): the NUMERIC rsLoRA parity probe — even when
    the §10 production launch_cmd calls ``extract`` directly (skipping
    ``prefetch``), the parity gate fires here before any extraction wave. Only
    ``--skip-parity`` (never used by the production launch) or a dry-run skips it.
    """
    phase_log("extract")
    # NUMERIC parity gate before any GPU extraction wave (BLOCKER 2). On a CPU
    # smoke the numeric forward is unavailable, so the gauge config check stands
    # in (and the numeric repro runs on the real GPU extract path).
    if not dry_run and not skip_parity:
        _rslora_parity_probe(behaviors[0], cpu_only=cpu_only)
    elif skip_parity:
        logger.info("extract: parity probe SKIPPED (--skip-parity)")
    cells: list[tuple[str, str]] = []  # (behavior, source)
    for behavior in behaviors:
        for source in select_sources(behavior, sources_arg):
            cells.append((behavior, source))
    logger.info("extract: %d source-adapter cells across behaviors=%s", len(cells), behaviors)
    # Resume-skip (default ON): drop cells whose .npz tensors already exist on disk
    # so a relaunch after a mid-run crash does NOT re-extract completed cells
    # (round-7: a ~95-min relaunch re-ran 32/64 already-on-disk cells). --no-resume-skip
    # forces a full re-extract. Skipped on dry-run (nothing is written there anyway).
    if resume_skip and not dry_run:
        cells = _filter_resume_skip(cells)
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
# Phase: a36-readout-reextract (followup a36-readout-reextract-cos)
# ─────────────────────────────────────────────────────────────────────────────


def _prefetch_inherited_delta_v(
    behaviors: list[str],
    sources_arg: str | None,
    targets_arg: str | None,
    layers: list[int],
) -> int:
    """Download #667's inherited Δv per-cell .npz into TENSORS_DIR (the analysis input).

    The re-extract amendment reuses #667's already-uploaded ``analysis_tensors``
    (the ``Δv = v+ − v0`` store) verbatim — it does NOT re-extract Δv. Each cell
    file is fetched DIRECTLY via ``hf_hub_download`` (per-file, NOT a full-tree
    list — the data repo's recursive tree 504s, #399 snapshot-truncation family),
    skipping any already on disk. Returns the count fetched. Fail-loud if a
    required cell is genuinely missing on HF (never a silent shrink).
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    from explore_persona_space.experiments.i537_contexts import eval_cids_for

    tdir = PROJECT_ROOT / TENSORS_DIR
    n_fetched = 0
    for behavior in behaviors:
        sources = select_sources(behavior, sources_arg)
        targets = select_targets(behavior, targets_arg)
        if targets is None:
            # Off-diagonal cells the A3.6 read uses == the 30 eval cids (+ source
            # diagonal, which the analysis skips — but fetch it too for parity).
            targets = list(dict.fromkeys(eval_cids_for(behavior)))
        for source in sources:
            cell_local = tdir / behavior / f"{source}_seed{_EXTRACT_SEED}"
            cell_local.mkdir(parents=True, exist_ok=True)
            for target in targets:
                if target == source:
                    continue  # off-diagonal only (the CHANGE read)
                for li in layers:
                    fn = f"{target}_L{li}.npz"
                    local = cell_local / fn
                    if local.is_file():
                        continue
                    rel = f"{behavior}/{source}_seed{_EXTRACT_SEED}/{fn}"
                    try:
                        src = hf_hub_download(
                            HF_DATA_REPO,
                            f"{HF_ANALYSIS_TENSORS_PREFIX}/{rel}",
                            repo_type="dataset",
                        )
                    except EntryNotFoundError as e:
                        raise RuntimeError(
                            f"inherited Δv cell missing on HF: {HF_ANALYSIS_TENSORS_PREFIX}/{rel} "
                            f"({e}) — #667's analysis_tensors do not cover this cell; HALT "
                            "(never silently shrink the A3.6 denominator)."
                        ) from e
                    import shutil

                    shutil.copy2(src, local)
                    n_fetched += 1
    logger.info("prefetched %d inherited Δv cells into %s", n_fetched, tdir)
    return n_fetched


def phase_reextract_prefetch(
    *,
    behaviors: list[str],
    sources_arg: str | None,
    targets_arg: str | None,
    layers: list[int],
    cpu_only: bool,
    skip_parity: bool,
    skip_store_pin: bool,
) -> None:
    """Phase-0 for the re-extract amendment: pins + parity probe + Δv prefetch.

    (i) SHA-pin the #658 store ``probe_pool_hash`` (the load-bearing pin, M3:
    NOT ``git_commit`` which is ``None`` in the live manifest) + #537 G_meta
    git_commit; (ii) the inherited 1-adapter rsLoRA NUMERIC parity probe (HALT on
    mismatch — the read gauge must match #667's committed θ⁺); (iii) download
    #667's inherited Δv per-cell store into TENSORS_DIR (the analysis input).
    """
    phase_log("reextract_prefetch")
    if not skip_store_pin:
        from issue667_analysis import assert_store_pin, load_g_meta

        from explore_persona_space.analysis.issue667 import EXPECTED_STORE_PROBE_POOL_HASH

        g_meta = load_g_meta()
        logger.info("G_meta git_commit pin OK: %s", g_meta["git_commit"][:16])
        assert_store_pin()
        logger.info("#658 store probe_pool_hash pin OK: %s", EXPECTED_STORE_PROBE_POOL_HASH[:16])

    # rsLoRA parity probe (fitness check (g)) — inherited verbatim. HALT on miss.
    if skip_parity:
        logger.info("reextract: parity probe SKIPPED (--skip-parity)")
    else:
        _rslora_parity_probe(behaviors[0], cpu_only=cpu_only)

    # Prefetch the inherited Δv cells (skipped on a synthetic-store smoke).
    if not skip_store_pin:
        _prefetch_inherited_delta_v(behaviors, sources_arg, targets_arg, layers)
    logger.info("[phase=reextract_prefetch_done]")


def _r_plus_extract_cmd(
    behavior: str,
    source: str,
    layers: list[int],
    gpu_id: int,
    max_probes: int | None,
    cpu_only: bool,
) -> tuple[list[str], Path, dict[str, str]]:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue667_extract.py",
        "--r-plus",
        "--behavior",
        behavior,
        "--source-cid",
        source,
        "--layers",
        *[str(li) for li in layers],
        "--r-plus-out",
        R_PLUS_DIR,
        "--gpu-id",
        str(gpu_id),
    ]
    if max_probes:
        cmd += ["--max-probes", str(max_probes)]
    if cpu_only:
        cmd += ["--cpu-only"]
    # CVD pinned in the LAUNCHER env per cell (#545) + spawn guard (gotchas #26).
    env = {
        "CUDA_VISIBLE_DEVICES": "" if cpu_only else str(gpu_id),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    log_path = _log_dir() / f"r_plus_{behavior}_{source}.log"
    return cmd, log_path, env


def phase_extract_r_plus(
    *,
    behaviors: list[str],
    sources_arg: str | None,
    layers: list[int],
    n_gpus: int,
    cpu_only: bool,
    max_probes: int | None,
    skip_upload: bool,
    dry_run: bool,
) -> None:
    """Re-extract r⁺ per source adapter in CVD-pinned waves; upload the r⁺ store."""
    phase_log("reextract_r_plus")
    cells: list[tuple[str, str]] = []
    for behavior in behaviors:
        for source in select_sources(behavior, sources_arg):
            cells.append((behavior, source))
    logger.info("reextract r⁺: %d source-adapter cells across behaviors=%s", len(cells), behaviors)
    n_par = 1 if cpu_only else max(n_gpus, 1)
    for wave_start in range(0, len(cells), n_par):
        wave = cells[wave_start : wave_start + n_par]
        cmds = [
            _r_plus_extract_cmd(behavior, source, layers, i % n_par, max_probes, cpu_only)
            for i, (behavior, source) in enumerate(wave)
        ]
        if dry_run:
            for (cmd, _lp, env), (behavior, source) in zip(cmds, wave, strict=True):
                logger.info(
                    "[dry-run] r⁺ %s/%s CVD=%r :: %s",
                    behavior,
                    source,
                    env.get("CUDA_VISIBLE_DEVICES"),
                    " ".join(shlex.quote(c) for c in cmd),
                )
            continue
        rcs = _run_parallel_with_log(cmds)
        bad = [(rc, c) for rc, c in zip(rcs, wave, strict=True) if rc != 0]
        if bad:
            raise RuntimeError(f"r⁺ extract wave failed: {bad}; see logs in {_log_dir()}")
        for behavior, source in wave:
            logger.info("r⁺ cell %s/%s complete", behavior, source)  # NOT [phase=done]
    if dry_run:
        logger.info("[phase=reextract_r_plus_done] (dry-run: no tensors, upload skipped)")
        return
    if not skip_upload:
        _upload_r_plus()
    logger.info("[phase=reextract_r_plus_done]")


def _upload_r_plus() -> None:
    """Upload the per-source r⁺ .npz to the HF data repo (new prefix; Upload Policy)."""
    if os.environ.get("EPM_SKIP_UPLOAD") == "1":
        logger.info("EPM_SKIP_UPLOAD=1 -> skipping r⁺ upload (smoke/local)")
        return
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files

    rdir = PROJECT_ROOT / R_PLUS_DIR
    npzs = sorted(rdir.rglob("*.npz"))
    if not npzs:
        raise RuntimeError(f"no r⁺ tensors to upload under {rdir} -- extraction wrote nothing")
    api = HfApi()
    ops = [
        CommitOperationAdd(
            path_in_repo=f"{HF_R_PLUS_PREFIX}/{p.relative_to(rdir).as_posix()}",
            path_or_fileobj=str(p),
        )
        for p in npzs
    ]
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue667 a36-reextract: {len(ops)} per-source r⁺ read-outs",
    )
    want = {f"{HF_R_PLUS_PREFIX}/{p.relative_to(rdir).as_posix()}" for p in npzs}
    files = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    missing = sorted(want - files)
    if missing:
        raise RuntimeError(f"r⁺ upload verification FAILED -- missing on Hub: {missing[:5]}")
    logger.info("uploaded + verified %d r⁺ tensors to %s", len(npzs), HF_DATA_REPO)


def phase_reextract_analysis(
    *, behaviors: list[str], layers: list[int], primary_layer: int, skip_store_pin: bool
) -> None:
    """Run the A3.6 re-extract + cosine + M1 diagnostics (issue667_analysis --reextract)."""
    phase_log("reextract_analysis")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue667_analysis.py",
        "--reextract",
        "--tensors-dir",
        TENSORS_DIR,
        "--r-plus-dir",
        R_PLUS_DIR,
        "--out-dir",
        OUT_DIR,
        "--behaviors",
        *behaviors,
        "--primary-layer",
        str(primary_layer),
        "--layers",
        *[str(li) for li in layers],
    ]
    if skip_store_pin:
        cmd += ["--skip-store-pin"]
    rc = _run_with_log(cmd, log_path=_log_dir() / "reextract_analysis.log")
    if rc != 0:
        raise RuntimeError(
            f"reextract analysis failed (rc={rc}); see {_log_dir() / 'reextract_analysis.log'}"
        )
    logger.info("[phase=reextract_analysis_done]")


def _reextract_reproducibility_card(behaviors: list[str], sources: list[str]) -> dict:
    """Per-cell adapter_paths + wandb hints for the epm:results reproducibility_card.

    This amendment trains NOTHING (forward-pass r⁺ re-extraction on #537 adapters)
    — there are no new WandB runs — so the card declares the REUSED #537 adapter
    paths per (behavior, source) cell + the issue667 project, per the training-task
    card contract (the adapters ARE the training artifacts this run reads).
    """
    adapter_paths: dict[str, str] = {}
    for behavior in behaviors:
        for source in sources:
            if behavior == "em":
                sub = f"adapters/i537_em_{source}_seed42/sft_em_adapter"
            elif behavior == "sycophancy":
                sub = f"adapters/i537_sycophancy_{source}_seed42"
            elif behavior == "fact":
                sub = f"adapters/i537_fact_{source}_seed42"
            else:
                continue
            adapter_paths[f"{behavior}/{source}"] = f"{HF_MODEL_REPO} :: {sub}"
    return {
        "training": "NONE (forward-pass r⁺ re-extraction on #537 adapters)",
        "adapter_paths": adapter_paths,
        "wandb_project": "issue667",
        "wandb_run_names": [],
        "wandb_url": "n/a (no training this run; r⁺ re-extracted on reused #537 adapters)",
        "r_plus_store": f"{HF_DATA_REPO} :: {HF_R_PLUS_PREFIX}/<beh>/<src>_seed42_L<l>.npz",
        "delta_v_inputs": f"{HF_DATA_REPO} :: {HF_ANALYSIS_TENSORS_PREFIX} (#667, reused)",
    }


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
        choices=["prefetch", "extract", "analysis", "all", "parity-probe", "a36-readout-reextract"],
        default="all",
        help=(
            "Phase to run. 'all' = prefetch -> extract -> analysis (the unified smoke/sweep). "
            "'a36-readout-reextract' = the followup amendment (re-extract r⁺ on θ⁺ + re-run A3.6 "
            "+ rotation cosine + M1 diagnostics); same unified smoke/sweep shape. "
            "'parity-probe' = the CUDA-isolated one-shot rsLoRA NUMERIC parity probe (internal; "
            "invoked by the dispatcher as a subprocess so the parent never inits CUDA — #667 r4)."
        ),
    )
    parser.add_argument(
        "--behavior",
        default=None,
        help=(
            "parity-probe behavior, subprocess entrypoint only (singular form of "
            "--behaviors). Read by main() only on the 'parity-probe' phase (#667 r6: "
            "the r4 subprocess-isolation refactor passed --behavior at the launch site "
            "and read args.behavior in main() but never registered the flag)."
        ),
    )
    parser.add_argument(
        "--source", default="default", help="parity-probe source cid (subprocess entrypoint only)."
    )
    parser.add_argument("--seed", type=int, default=42, help="parity-probe seed (subprocess only).")
    parser.add_argument(
        "--result-out",
        default=None,
        help="parity-probe result JSON path (subprocess entrypoint only).",
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
        "--no-resume-skip",
        action="store_true",
        help=(
            "Force a full re-extract: do NOT skip cells whose .done sentinel already "
            "exists on disk (resume-skip is ON by default — round-7/8)."
        ),
    )
    parser.add_argument(
        "--backfill-sentinels",
        action="store_true",
        help=(
            "One-shot migration: write a .done sentinel for every COMPLETE on-disk "
            "cell (round-8 — the 32 cells extracted under the old any(*.npz) contract "
            "have no sentinel). Validates the .npz complement per cell; incomplete "
            "cells are reported and left without a sentinel. Runs then exits."
        ),
    )
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

    # parity-probe: the CUDA-isolated one-shot subprocess entrypoint (#667 r4). Runs
    # the NUMERIC diagonal-write reproduction IN THIS child process (CUDA inits here,
    # never in the dispatcher parent), writes the result JSON, and exits — rc!=0 (a
    # failed assert / RuntimeError HALT) propagates the gate to the parent.
    if args.phase == "parity-probe":
        _require_credentials()
        result = _numeric_rslora_parity(args.behavior, source=args.source, seed=args.seed)
        if args.result_out:
            Path(args.result_out).write_text(json.dumps(result, indent=2))
        return 0

    # One-shot backfill migration (round-8): write .done for already-complete
    # cells so the sentinel-based resume-skip can resume-skip them on relaunch.
    # Pure local filesystem walk + atomic writes — no credentials, then exit.
    if args.backfill_sentinels:
        phase_backfill_sentinels(layers=args.layers)
        return 0

    smoke = args.smoke or args.cpu_only
    # Smoke defaults: cap probes + train rows, run analysis on-pod with the pins
    # unless explicitly synthetic.
    max_probes = args.max_probes if args.max_probes is not None else (2 if smoke else None)
    max_train_rows = (
        args.max_train_rows if args.max_train_rows is not None else (8 if smoke else None)
    )

    # ── followup a36-readout-reextract-cos: re-extract r⁺ + re-run A3.6 + M1 ──
    # Distinct phase sequence (prefetch -> r⁺ extract -> reextract analysis) and a
    # distinct sentinel (reproducibility_card + plan-required fields). Same unified
    # smoke/sweep shape: --cpu-only + a 1-cell --sources subset is the smoke.
    if args.phase == "a36-readout-reextract":
        if not args.dry_run and not args.skip_store_pin:
            _require_credentials()
        phase_reextract_prefetch(
            behaviors=args.behaviors,
            sources_arg=args.sources,
            targets_arg=args.targets,
            layers=args.layers,
            cpu_only=args.cpu_only,
            skip_parity=args.skip_parity,
            skip_store_pin=args.skip_store_pin,
        )
        phase_extract_r_plus(
            behaviors=args.behaviors,
            sources_arg=args.sources,
            layers=args.layers,
            n_gpus=args.n_gpus,
            cpu_only=args.cpu_only,
            max_probes=max_probes,
            skip_upload=args.skip_upload,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            phase_reextract_analysis(
                behaviors=args.behaviors,
                layers=args.layers,
                primary_layer=args.primary_layer,
                skip_store_pin=args.skip_store_pin,
            )
        sources_realized = sorted(
            {s for b in args.behaviors for s in select_sources(b, args.sources)}
        )
        card = _reextract_reproducibility_card(args.behaviors, sources_realized)
        note = (
            f"a36-readout-reextract behaviors={args.behaviors} sources={args.sources} "
            f"layers={args.layers} smoke={smoke} dry_run={args.dry_run}"
        )
        write_sentinel(
            "epm:results",
            note,
            extra={
                "phase": "a36-readout-reextract",
                "smoke": smoke,
                "eval_paths": [
                    f"{REEXTRACT_OUT_DIR}/partial_spearman_recovery.json",
                    f"{REEXTRACT_OUT_DIR}/cos_r_plus_vs_r_base.json",
                ],
                "eval_numbers": "see partial_spearman_recovery.json :: by_behavior_layer",
                "reproducibility_card": card,
                "wandb_url": card["wandb_url"],
                "hf_hub_url": f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/{HF_R_PLUS_PREFIX}",
                "worktree_path": str(PROJECT_ROOT),
                "final_commit_sha": _git_commit_sha(),
                "gpu_hours_used": None,
                "gpu_hours_budgeted": 6,
                "plan_deviations": [],
            },
        )
        logger.info("[phase=done]")  # terminal marker — reserved for this single line
        return 0

    phases = ["prefetch", "extract", "analysis"] if args.phase == "all" else [args.phase]

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
                skip_parity=args.skip_parity,
                resume_skip=not args.no_resume_skip,
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
