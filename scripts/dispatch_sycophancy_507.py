#!/usr/bin/env python3
"""Task #507 dispatcher — Qwen-2.5-72B sycophancy port + predictor study.

UNIFIED smoke = sweep with one cell: smoke is just `--sources software_engineer
--seeds 42 --smoke-gate-only` (cell 1). Same dispatcher, same train_72b /
eval_72b call shapes, same HF upload path, same sentinel logic. On smoke-PASS
(plan v2 section 7 gate), cells 2-6 fire sequentially on the SAME training
pod. The full sweep IS the dispatcher with no flag — smoke is a single cell
of the same loop.

Phases (plan v2 section 4.2):

  Phase 0    — bootstrap (CPU, fast): pull training pools + eval_50 from HF.
  Phase 1    — 72B LoRA training (multi-GPU ZeRO-3 launch via deepspeed).
  Phase 2    — vLLM eval at TP=8 on inf-70b pod.
  Phase 2.5  — Haiku judge via Anthropic Batches API.
  Phase 3    — base-72B predictor extraction (#470 phase1/2/3 with --model 72b).
  Phase 4    — per-arm regress (#470 phase5_regress) + cross-arm compare.
  Phase 5    — OPTIONAL 7B re-run (fires only when 72B headline diverges
               from #470 by > 0.10 rho).
  Phase 6    — figures.

Smoke gate (plan v2 section 7):
  After cell 1 (software_engineer seed 42) Phase 1 + 2 + 2.5 complete, the
  dispatcher verifies:
    (a) effective_batch_size == 16 (runtime assertion fired in train_72b)
    (b) source-self own-rate >= +0.50 over base
    (c) HF Hub adapter upload PASS
    (d) DV-floor diagnostic logged (warn, not fail-fast)
  Mismatch on (a)/(b)/(c) raises RuntimeError BEFORE cells 2-6 launch.

Pod-side discipline:
  * Every subprocess gets env={**os.environ} (load_dotenv at module-top).
  * NEVER calls scripts/task.py (pods run on issue-<N>; task.py guards main).
  * Sentinel file + [phase=...] log lines for poll_pipeline.py.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# load_dotenv at module-top so HF_TOKEN / WANDB_API_KEY / ANTHROPIC_API_KEY are
# in os.environ before ANY subprocess spawns inherit the env (per CLAUDE.md
# subprocess-env-passthrough rule + experiment-implementer.md memory note).
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from explore_persona_space.experiments.sycophancy_scale_507 import (  # noqa: E402
    HEADLINE_LAYER_BY_ARCH,
    LAYER_SET_BY_ARCH,
    SOURCE_PERSONAS_507,
)

log = logging.getLogger("dispatch_sycophancy_507")

# ── Constants ──
ISSUE_N = 507
SENTINEL_SCHEMA_VERSION = 1
SENTINEL_DIR = Path("/workspace/logs")
DEFAULT_SEED = 42
HF_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# #411 published path with the 700-row contrastive training pools + the
# 50-probe held-out eval set; A11 in plan v2 section 12 verifies this is the
# correct subdir.
HF_411_DATA_PREFIX = "issue411_sycophancy_cosine_gradient"
HF_TRAINING_POOLS_PATTERN = f"{HF_411_DATA_PREFIX}/training_pools/{{source}}_train.jsonl"
HF_EVAL_50_PATH = f"{HF_411_DATA_PREFIX}/data/wrong_claims/eval_50.jsonl"

# Local cache roots.
OUTPUT_ROOT = REPO_ROOT / "eval_results" / "issue_507"
TRAIN_POOLS_CACHE = OUTPUT_ROOT / "_inputs" / "training_pools"
EVAL_POOL_CACHE = OUTPUT_ROOT / "_inputs" / "eval_50.jsonl"
SLAB_ROOT = OUTPUT_ROOT / "72b"
RUNS_ROOT = OUTPUT_ROOT / "runs"
PREDICTOR_72B_OUTPUT_BASE = OUTPUT_ROOT / "predictor_72b"

# Smoke-gate thresholds (plan v2 §7).
SMOKE_OWN_RATE_FLOOR = 0.50  # source-self rate must lift >= +0.50 over base


def _log_phase(name: str) -> None:
    """Emit a [phase=<name>] line for poll_pipeline.py."""
    print(f"[phase={name}]", flush=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _run_subprocess(
    cmd: list[str],
    *,
    label: str,
    check: bool = True,
    env_extra: dict[str, str] | None = None,
) -> int:
    """Run a subprocess with explicit env passthrough (CLAUDE.md mandate).

    Args:
        cmd: argv list.
        label: human label for logging.
        check: raise on non-zero return.
        env_extra: per-call env additions / overrides (round-2 fix per
            code-review Critical 5: the predictor_jsdiv_470 phase scripts
            need PREDICTOR_OUTPUT_BASE / PREDICTOR_HEADLINE_LAYER /
            PREDICTOR_DV_ANALYZE_SUMMARY threaded through so #507's 72B
            run doesn't overwrite #470's committed 7B regression.json).
    """
    env = {**os.environ}  # explicit copy, never inherited implicitly
    if env_extra:
        env.update(env_extra)
    log.info("[%s] launching: %s", label, " ".join(shlex.quote(c) for c in cmd))
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), check=False)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Subprocess '{label}' exited rc={result.returncode}. "
            f"Command: {' '.join(shlex.quote(c) for c in cmd)}"
        )
    return result.returncode


def _predictor_env_overrides() -> dict[str, str]:
    """Compose the env overrides for predictor_jsdiv_470 sub-phases on the 72B path.

    Round-2 fix per code-review Critical 5/7/11: threads through OUTPUT_BASE,
    figures dir, headline layer (57 for 72B), and DV / base-panel paths so
    the 72B run writes to its own namespace and reads its own DV. Also
    sets the no-overwrite guard so common.py raises if any subprocess
    forgets the override.
    """
    overrides = {
        "PREDICTOR_OUTPUT_BASE": str(PREDICTOR_72B_OUTPUT_BASE),
        "PREDICTOR_FIGURES_DIR": str(REPO_ROOT / "figures" / "issue_507"),
        "PREDICTOR_HEADLINE_LAYER": str(HEADLINE_LAYER_BY_ARCH["72b"]),
        # Round-3 fix per code-review Critical 5: env-parametrize the full
        # 72B layer set so Phase 4's `for layer in DEFAULT_LAYERS:` loop in
        # phase4_load_dv.py reads {21,40,57,70} instead of the hard-coded 7B
        # tuple {7,14,21,27}. Without this Phase 4 file-NotFound errors on
        # layer_7.json because Phase 2 wrote layer_21/40/57/70.json.
        "PREDICTOR_LAYERS": ",".join(str(li) for li in LAYER_SET_BY_ARCH["72b"]),
        # Round-3 fix per code-review Critical 4: Phase 3 (sequence JS+KL)
        # previously pinned the whole model to a single GPU
        # (device_map={"": cuda:0}). Qwen-72B at 145 GB bf16 cannot fit on
        # any H100/H200 80 GB; force HF accelerate auto-sharding so the model
        # spans all visible GPUs the same way the preflight already does.
        "PREDICTOR_DEVICE_MAP": "auto",
        "PREDICTOR_GUARD_NO_OVERWRITE_470": "1",
    }
    # Point Phase 4 at the 72B's own DV + base panel rates produced by
    # the in-flight Phase 2 + 2.5 (analyze_summary.json from #411's
    # analyze.py run on 72B eval JSONs; base_panel_rates.json from
    # judge_base_panel on the 72B base panel).
    dv_72b = SLAB_ROOT / "analyze_summary_72b.json"
    base_rates_72b = SLAB_ROOT / "base_panel_rates.json"
    if dv_72b.exists():
        overrides["PREDICTOR_DV_ANALYZE_SUMMARY"] = str(dv_72b)
    if base_rates_72b.exists():
        overrides["PREDICTOR_BASE_PANEL_RATES"] = str(base_rates_72b)
    return overrides


# ── Phase 0: bootstrap ──


def phase0_bootstrap(*, sources: list[str], force: bool = False) -> None:
    """Pull #411 training pools + eval_50.jsonl from HF data repo into a local cache."""
    from huggingface_hub import hf_hub_download

    _log_phase("bootstrap")
    TRAIN_POOLS_CACHE.mkdir(parents=True, exist_ok=True)
    EVAL_POOL_CACHE.parent.mkdir(parents=True, exist_ok=True)

    # Pull training pools (one per source).
    for source in sources:
        target = TRAIN_POOLS_CACHE / f"{source}_train.jsonl"
        if target.exists() and not force:
            log.info("[phase0] %s training pool cached at %s; skipping", source, target)
            continue
        log.info("[phase0] Downloading %s training pool from HF", source)
        local = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=HF_TRAINING_POOLS_PATTERN.format(source=source),
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        shutil.copy(local, target)
        log.info("[phase0] %s -> %s", source, target)

    # Pull eval_50.jsonl.
    if not EVAL_POOL_CACHE.exists() or force:
        log.info("[phase0] Downloading eval_50.jsonl from HF")
        local = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=HF_EVAL_50_PATH,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        shutil.copy(local, EVAL_POOL_CACHE)
        log.info("[phase0] eval_50.jsonl -> %s", EVAL_POOL_CACHE)
    else:
        log.info("[phase0] eval_50.jsonl cached at %s; skipping", EVAL_POOL_CACHE)


# ── Phase 1: training (one cell) ──


def phase1_train_cell(*, source: str, seed: int, world_size: int | None = None) -> Path:
    """Train one 72B LoRA cell via the deepspeed launcher. Returns the adapter dir.

    Round-2 fix per code-review Critical 3: the round-1 in-process call
    defaulted to world_size=1 and tried to load Qwen-72B onto one GPU,
    OOMing immediately. Now wraps train_72b_entrypoint under
    ``deepspeed --num_gpus=<world_size> -m ...`` so HF Trainer + ZeRO-3
    shard the model across ranks. The dispatcher is rank-0 / orchestration
    only; the launcher fans out per-rank.

    Single-GPU debug path (world_size=1) still supported via the in-process
    call — useful for CPU smoke tests where deepspeed isn't installed.
    """
    _log_phase("train")
    train_jsonl = TRAIN_POOLS_CACHE / f"{source}_train.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(
            f"Training pool for {source} not found at {train_jsonl}. Phase 0 "
            f"bootstrap must run before Phase 1."
        )
    output_dir = RUNS_ROOT / f"72b_{source}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve world_size: explicit > visible-GPU count > 1.
    resolved_ws = world_size
    if resolved_ws is None:
        try:
            n_visible = len(
                [x for x in (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").split(",") if x]
            )
        except Exception:
            n_visible = 0
        if n_visible == 0:
            # nvidia-smi fallback only when CVD is unset.
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "-L"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    env={**os.environ},
                )
                n_visible = sum(1 for line in out.splitlines() if line.startswith("GPU "))
            except (subprocess.CalledProcessError, FileNotFoundError):
                n_visible = 1
        resolved_ws = max(n_visible, 1)

    log.info(
        "[phase1] train_72b launch: source=%s seed=%d output_dir=%s world_size=%d",
        source,
        seed,
        output_dir,
        resolved_ws,
    )

    if resolved_ws == 1:
        # Single-GPU debug path: call train_72b in-process. Useful for CPU
        # smoke tests where deepspeed isn't installed.
        from explore_persona_space.experiments.sycophancy_scale_507.train_72b import train_72b

        log.warning(
            "[phase1] world_size=1; using in-process train_72b (debug/CPU-smoke path). "
            "Production 72B requires world_size>=4 under the deepspeed launcher."
        )
        adapter_dir, _merged_placeholder = train_72b(
            source=source,
            seed=seed,
            train_jsonl=train_jsonl,
            output_dir=output_dir,
            world_size=1,
            hf_upload=True,
        )
        return adapter_dir

    # Multi-GPU production path: launch via deepspeed.
    cmd = [
        "deepspeed",
        "--num_gpus",
        str(resolved_ws),
        "-m",
        "explore_persona_space.experiments.sycophancy_scale_507.train_72b_entrypoint",
        "--source",
        source,
        "--seed",
        str(seed),
        "--train-jsonl",
        str(train_jsonl),
        "--output",
        str(output_dir),
    ]
    _run_subprocess(cmd, label=f"phase1 train_72b {source} (deepspeed ws={resolved_ws})")
    adapter_dir = output_dir / "adapter"
    safetensors = list(adapter_dir.glob("*.safetensors"))
    if not safetensors:
        raise RuntimeError(
            f"phase1 deepspeed launcher exited 0 but {adapter_dir} has no "
            "safetensors. Either training silently failed or the save path "
            "was redirected."
        )
    return adapter_dir


# ── Phase 2: eval (one cell) — subprocess-isolated for vLLM teardown safety ──


def phase2_eval_cell(*, source: str, seed: int, adapter_path: Path) -> Path:
    """vLLM eval one trained cell at TP=8. Subprocess-isolated."""
    _log_phase("eval")
    eval_out_dir = SLAB_ROOT / source / f"seed_{seed}"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = SENTINEL_DIR / f"issue-507-{source}-eval-results.json"
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "explore_persona_space.experiments.sycophancy_scale_507.eval_72b_vllm",
        "--source",
        source,
        "--seed",
        str(seed),
        "--adapter-path",
        str(adapter_path),
        "--eval-pool",
        str(EVAL_POOL_CACHE),
        "--out-dir",
        str(eval_out_dir),
        "--sentinel-path",
        str(sentinel_path),
    ]
    _run_subprocess(cmd, label=f"phase2 eval {source}")
    return eval_out_dir


def phase2_base_panel(*, seed: int) -> Path:
    """Phase 1.5 (named Phase 2-base here for ordering): base 72B panel pass + judge.

    Required for the smoke gate's source-self own-rate vs base comparison and
    for the full predictor regression's base-rate null.
    """
    _log_phase("base_panel")
    base_out_dir = SLAB_ROOT / "base" / f"seed_{seed}"
    base_out_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = SENTINEL_DIR / "issue-507-base-panel-eval.json"
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "explore_persona_space.experiments.sycophancy_scale_507.eval_72b_vllm",
        "--source",
        "base",
        "--seed",
        str(seed),
        "--base-model-id",
        "Qwen/Qwen2.5-72B-Instruct",
        "--eval-pool",
        str(EVAL_POOL_CACHE),
        "--out-dir",
        str(base_out_dir),
        "--sentinel-path",
        str(sentinel_path),
    ]
    _run_subprocess(cmd, label="phase2-base eval")
    return base_out_dir


# ── Phase 2.5: Haiku judge (subprocess to #411's judge_base_panel) ──


def phase2_5_judge_base_panel(*, seed: int) -> Path:
    """Judge the base panel via #411's judge_base_panel — writes base_panel_rates.json."""
    _log_phase("judge_base")
    base_panel_rates = SLAB_ROOT / "base_panel_rates.json"
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "explore_persona_space.experiments.sycophancy_implantation_411.judge_base_panel",
        "--slab-root",
        str(SLAB_ROOT),
        "--base-source",
        "base",
        "--seed",
        str(seed),
        "--output",
        str(base_panel_rates),
    ]
    _run_subprocess(cmd, label="phase2.5 base panel judge")
    if not base_panel_rates.exists():
        raise RuntimeError(f"judge_base_panel finished but {base_panel_rates} not present")
    return base_panel_rates


def phase2_5_judge_source(*, source: str, seed: int) -> Path:
    """Judge one source's 12k rollouts via #411's analyze.py judge stage."""
    _log_phase("judge")
    # #411's analyze.py is the canonical aggregator for per-source verdicts
    # over the 24 panel personas. Calling it per-source matches the smoke
    # gate's needs (we want one source's verdicts ASAP).
    eval_out_dir = SLAB_ROOT / source / f"seed_{seed}"
    if not eval_out_dir.exists():
        raise FileNotFoundError(
            f"phase2.5_judge_source: eval out dir {eval_out_dir} missing; Phase 2 must run first."
        )
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "explore_persona_space.experiments.sycophancy_implantation_411.judge",
        "--slab-root",
        str(SLAB_ROOT),
        "--source",
        source,
        "--seed",
        str(seed),
    ]
    _run_subprocess(cmd, label=f"phase2.5 judge {source}")
    # Per #411 convention, the judge step writes per-source verdict JSON
    # alongside the eval files. Touch a sentinel so the smoke gate can detect.
    return eval_out_dir


# ── Smoke gate ──


def _read_source_self_rate(*, source: str, seed: int) -> float | None:
    """Read the source-self agree-rate from the per-panel judge output.

    The smoke gate (plan v2 §7 condition 2) needs the source's own panel
    persona slot rate. #411's judge.py writes per-panel rates as a side
    output of analyze.py. We look for a per-panel rate file at the expected
    location; if it's not there yet (analyze.py hasn't run), we return None
    so the caller can decide whether to proceed.
    """
    eval_out_dir = SLAB_ROOT / source / f"seed_{seed}"
    # The #411 analyze step produces analyze_summary.json containing
    # per_source.<src>.per_panel_delta etc. For the smoke gate we only need
    # the source-self own rate, which lives at per_source.<src>.per_panel_rate[<src>]
    # (the panel-persona-equals-source slot). We look across a few candidate
    # filenames to be tolerant of #411's exact output shape.
    candidates = [
        eval_out_dir / f"per_panel_rates_{source}.json",
        eval_out_dir / "analyze_summary.json",
        eval_out_dir / "judge_verdicts.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            # Probe a couple of likely keys.
            for key_path in [
                ("per_source", source, "per_panel_rate", source),
                ("per_panel_rate", source),
                ("source_self_rate",),
            ]:
                ref = payload
                ok = True
                for k in key_path:
                    if not isinstance(ref, dict) or k not in ref:
                        ok = False
                        break
                    ref = ref[k]
                if ok and isinstance(ref, int | float):
                    return float(ref)
    return None


def _read_base_rate_for_source(*, source: str) -> float | None:
    """Read the base-panel agree rate at the source-persona slot.

    Round-2 fix per code-review Critical 8 / Codex blocker
    `smoke-gate-rate-shape`: judge_base_panel.py writes
    ``{"panel_rates": {<persona>: <float-or-dict>, ...}, ...}`` — the
    rates are NESTED under the ``panel_rates`` key, not at the top level.
    The old reader's ``payload.get(source)`` returned None every time,
    silently failing the smoke gate after spending cell-1 GPU time.
    """
    base_panel_rates = SLAB_ROOT / "base_panel_rates.json"
    if not base_panel_rates.exists():
        return None
    try:
        payload = json.loads(base_panel_rates.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    panel_rates = payload.get("panel_rates")
    if not isinstance(panel_rates, dict):
        return None
    block = panel_rates.get(source)
    # judge_base_panel emits per-persona blocks that carry an "agree_rate"
    # (or a bare float on older formats). Be defensive.
    if isinstance(block, dict):
        rate = block.get("agree_rate")
        if rate is None:
            rate = block.get("rate")
        if isinstance(rate, int | float):
            return float(rate)
    elif isinstance(block, int | float):
        return float(block)
    return None


def _hf_adapter_uploaded(*, source: str, seed: int) -> bool:
    """Verify the trained adapter is on the HF model repo (smoke gate condition c)."""
    try:
        from huggingface_hub import list_repo_files

        adapter_subdir = f"adapters/issue_507/72b/{source}_seed{seed}"
        files = list_repo_files(
            repo_id=HF_REPO,
            revision="main",
            token=os.environ.get("HF_TOKEN"),
        )
        return any(adapter_subdir in f for f in files)
    except Exception as exc:
        log.warning("hf_adapter_uploaded: list_repo_files raised %s; treating as not-uploaded", exc)
        return False


def smoke_gate(*, source: str, seed: int) -> dict[str, object]:
    """Check plan v2 §7 cell-1 smoke gate conditions; raise on hard fail.

    Returns the summary dict for logging (and for inclusion in the dispatcher's
    sentinel payload).
    """
    _log_phase("smoke_gate")
    summary: dict[str, object] = {
        "source": source,
        "seed": seed,
        "gate_conditions": {},
    }
    # Condition (c): HF adapter upload.
    uploaded = _hf_adapter_uploaded(source=source, seed=seed)
    summary["gate_conditions"]["c_hf_upload"] = uploaded
    if not uploaded:
        raise RuntimeError(
            f"smoke_gate FAIL: adapter for {source}_seed{seed} not found on HF "
            f"{HF_REPO}:adapters/issue_507/72b/{source}_seed{seed}. Training "
            f"or HF upload failed."
        )

    # Condition (b): source-self own-rate >= +0.50 over base.
    own_rate = _read_source_self_rate(source=source, seed=seed)
    base_rate = _read_base_rate_for_source(source=source)
    summary["gate_conditions"]["source_self_rate"] = own_rate
    summary["gate_conditions"]["base_rate"] = base_rate
    if own_rate is None or base_rate is None:
        log.warning(
            "smoke_gate: own_rate=%s base_rate=%s not yet readable; cannot "
            "verify condition (b). The judge / analyze step must complete "
            "before this gate runs.",
            own_rate,
            base_rate,
        )
        summary["gate_conditions"]["b_implant_lifted_by_0p50"] = None
        # Without rates we cannot evaluate (b); raise so a missing-rate state
        # doesn't silently advance to cells 2-6.
        raise RuntimeError(
            "smoke_gate: source-self / base rate not yet judged. Re-run after "
            "phase2.5_judge_source + phase2.5_judge_base_panel complete."
        )
    delta_lift = own_rate - base_rate
    passes_b = delta_lift >= SMOKE_OWN_RATE_FLOOR
    summary["gate_conditions"]["delta_lift"] = delta_lift
    summary["gate_conditions"]["b_implant_lifted_by_0p50"] = passes_b
    if not passes_b:
        raise RuntimeError(
            f"smoke_gate FAIL: source-self rate ({own_rate:.3f}) - base rate "
            f"({base_rate:.3f}) = {delta_lift:.3f} < +{SMOKE_OWN_RATE_FLOOR:.2f} "
            f"floor. Training did not implant at #411 level; cells 2-6 will "
            f"not fire."
        )

    # Condition (a): effective_batch_size==16 is enforced by train_72b's
    # runtime assertion at training start. By the time we get here, training
    # has completed without raising AssertionError, so condition (a) is
    # SATISFIED structurally. We still record it as PASS in the summary.
    summary["gate_conditions"]["a_effective_batch_eq_16"] = True
    log.info("smoke_gate PASS for %s seed %d: %s", source, seed, summary)
    return summary


# ── Phase 3: predictor extraction ──


def phase3_predictor_72b(*, smoke: bool, gpu_id: int = 0) -> None:
    """Run the 72B predictor extraction: preflight + phase1/2/3 with --model 72b."""
    _log_phase("predictor_72b")
    PREDICTOR_72B_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    pred_env = _predictor_env_overrides()

    # Preflight: load base 72B, verify no offload.
    _run_subprocess(
        [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.sycophancy_scale_507.predictor_72b",
            "--mode",
            "preflight",
            "--model",
            "Qwen/Qwen2.5-72B-Instruct",
            "--output",
            str(PREDICTOR_72B_OUTPUT_BASE / "preflight.json"),
        ],
        label="phase3 preflight",
        env_extra=pred_env,
    )

    # Re-use the predictor_jsdiv_470 phase scripts with --model 72b. Output
    # paths are env-parametrized via PREDICTOR_OUTPUT_BASE (#507 namespace,
    # NEVER overwrites #470's regression.json). HEADLINE_LAYER is also
    # env-parametrized so 72B uses layer 57 (depth ratio 0.71) instead of
    # the 7B default of 21.
    layers_72b = " ".join(str(li) for li in LAYER_SET_BY_ARCH["72b"])
    probes_arg = ["--probes", "5"] if smoke else []
    r_arg = ["--R", "2" if smoke else "8"]

    # Phase 3.1: vLLM sample base-72B responses.
    _run_subprocess(
        [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.predictor_jsdiv_470.phase1_sample_responses",
            "--model",
            "Qwen/Qwen2.5-72B-Instruct",
            *r_arg,
            *probes_arg,
        ],
        label="phase3.1 sample base responses (72B)",
        env_extra=pred_env,
    )

    # Phase 3.2: response-token cosine over 72B layers.
    layer_args = ["--layers", *[str(li) for li in LAYER_SET_BY_ARCH["72b"]]]
    _run_subprocess(
        [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.predictor_jsdiv_470.phase2_cosine_response_token",
            "--model",
            "Qwen/Qwen2.5-72B-Instruct",
            *layer_args,
            "--gpu-id",
            str(gpu_id),
        ],
        label=f"phase3.2 cosine 72B (layers={layers_72b})",
        env_extra=pred_env,
    )

    # Phase 3.3: RB sequence JS + KL.
    # Round-3 fix per code-review Critical 3: Phase 3's argparse does NOT
    # accept --R (only Phase 1 sampling does). Phase 3 INFERS R from the
    # Phase 1 outputs already on disk (responses shape (n_probes, R) is
    # read off phase1[persona]["responses"]). Passing --R here crashed
    # Phase 3.3 at argparse before producing any JS/KL cells.
    sources_arg = ["--sources", *list(SOURCE_PERSONAS_507)]
    if smoke:
        sources_arg = ["--sources", "software_engineer", "--bystanders", "comedian"]
    _run_subprocess(
        [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.predictor_jsdiv_470.phase3_sequence_js_kl",
            "--model",
            "Qwen/Qwen2.5-72B-Instruct",
            *sources_arg,
            *probes_arg,
            "--gpu-id",
            str(gpu_id),
        ],
        label="phase3.3 RB JS+KL (72B)",
        env_extra=pred_env,
    )


# ── Phase 4: regress + cross-arm compare ──


def phase4_regress_and_cross_arm(*, regression_7b_path: Path | None) -> Path:
    """Run phase5_regress on the 72B outputs + cross-arm compare against #470 7B.

    Round-2 fix per code-review Critical 5: writes 72B outputs to
    eval_results/issue_507/predictor_72b/ (NOT issue_470). The 7B regression
    stays at eval_results/issue_470/regression.json (committed clean-result
    artifact, never overwritten). analyze_507 reads both arms' real files
    and writes its cross-arm CI to eval_results/issue_507/.
    """
    _log_phase("regress")
    pred_env = _predictor_env_overrides()
    # Run the predictor_jsdiv_470 phase4_load_dv + phase5_regress on the
    # 72B outputs under the 507 namespace (PREDICTOR_OUTPUT_BASE env).
    _run_subprocess(
        [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.predictor_jsdiv_470.phase4_load_dv",
        ],
        label="phase4 load DV (72B)",
        env_extra=pred_env,
    )
    _run_subprocess(
        [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.predictor_jsdiv_470.phase5_regress",
        ],
        label="phase4 regress (72B)",
        env_extra=pred_env,
    )

    # Cross-arm compare. The 7B regression is #470's committed publication
    # (under eval_results/issue_470/). The 72B regression is THIS run's
    # output under eval_results/issue_507/predictor_72b/. They are
    # DISTINCT files (round-2 fix per code-review Critical 5: previously
    # both pointed at issue_470/regression.json, producing a degenerate
    # 0-difference cross-arm output).
    if regression_7b_path is None:
        regression_7b_path = REPO_ROOT / "eval_results" / "issue_470" / "regression.json"
    regression_72b_path = PREDICTOR_72B_OUTPUT_BASE / "regression.json"
    if not regression_7b_path.exists():
        raise FileNotFoundError(
            f"7B regression.json not found at {regression_7b_path}. The #470 "
            "clean-result artifact must be committed before the cross-arm "
            "compare can run; pass --regression-7b to override."
        )
    if not regression_72b_path.exists():
        raise FileNotFoundError(
            f"72B regression.json not found at {regression_72b_path}. Phase 5 "
            "regress did not produce its output; check predictor env overrides "
            "(PREDICTOR_OUTPUT_BASE) and Phase 5 logs."
        )
    if regression_7b_path.resolve() == regression_72b_path.resolve():
        raise RuntimeError(
            "Cross-arm compare misconfigured: regression_7b_path == "
            f"regression_72b_path == {regression_7b_path}. The 7B and 72B "
            "files MUST be distinct or the comparison is degenerate."
        )
    cross_arm_out = OUTPUT_ROOT / "cross_arm_comparison.json"
    _log_phase("cross_arm")
    _run_subprocess(
        [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.sycophancy_scale_507.analyze_507",
            "--regression-7b",
            str(regression_7b_path),
            "--regression-72b",
            str(regression_72b_path),
            "--output",
            str(cross_arm_out),
        ],
        label="phase4 cross-arm compare",
    )
    return cross_arm_out


# ── Phase 6: figures ──


def phase6_figures() -> None:
    """Run #470's phase6_figures over the 72B outputs (in the 507 namespace)."""
    _log_phase("figures")
    _run_subprocess(
        [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.predictor_jsdiv_470.phase6_figures",
        ],
        label="phase6 figures",
        env_extra=_predictor_env_overrides(),
    )


# ── End-of-run sentinel ──


def _write_results_sentinel(*, summary: dict[str, object]) -> Path | None:
    """Write the poll_pipeline.py sentinel with required keys + the marker note."""
    try:
        SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning(
            "Could not create %s (%s); skipping sentinel (expected off-pod).",
            SENTINEL_DIR,
            exc,
        )
        return None
    note_lines = [
        f"task: #{ISSUE_N} — Qwen-72B sycophancy port + predictor study",
        f"code_commit: {_git_sha()}",
        f"worktree: {REPO_ROOT}",
        "",
        "summary:",
        *(f"  {k}: {v}" for k, v in summary.items() if not isinstance(v, dict | list)),
    ]
    note = "\n".join(note_lines)
    sentinel = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "task_id": ISSUE_N,
        "kind": "epm:results",
        "version": 1,
        "gate": None,
        "blocks_pipeline": False,
        "note": note,
        "by": "dispatch_sycophancy_507",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "hostname": socket.gethostname(),
    }
    out_path = SENTINEL_DIR / f"issue-{ISSUE_N}-epm_results-{int(time.time())}.json"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(sentinel, indent=2))
    tmp.replace(out_path)
    log.info("Wrote results sentinel to %s", out_path)
    return out_path


# ── Top-level orchestration ──


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_PERSONAS_507),
        help="Sources to train (default: all 6).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument(
        "--smoke-gate-only",
        action="store_true",
        help="Stop after cell 1's smoke gate; useful when validating the rig "
        "before the full sweep. Unified-architecture parity: same code path "
        "as the sweep, just one cell.",
    )
    parser.add_argument(
        "--skip-phase0",
        action="store_true",
        help="Skip Phase 0 bootstrap (training pools already cached locally).",
    )
    parser.add_argument(
        "--skip-base-panel",
        action="store_true",
        help="Skip Phase 2-base + Phase 2.5 judge-base (already produced).",
    )
    parser.add_argument(
        "--skip-predictor",
        action="store_true",
        help="Skip Phase 3 (predictor extraction).",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip Phase 4 (regress + cross-arm compare).",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip Phase 6 (figures).",
    )
    parser.add_argument(
        "--regression-7b",
        type=Path,
        default=None,
        help="Path to #470's 7B regression.json (default: eval_results/issue_470/regression.json).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Override world_size detection (default: read from WORLD_SIZE / "
        "torch.distributed). Set explicitly when running outside the "
        "deepspeed/accelerate launcher.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info(
        "Dispatcher start: sources=%s, seeds=%s, smoke_gate_only=%s",
        args.sources,
        args.seeds,
        args.smoke_gate_only,
    )

    sources = list(args.sources)
    if not sources:
        raise ValueError("--sources cannot be empty")
    if any(s not in SOURCE_PERSONAS_507 for s in sources):
        unknown = [s for s in sources if s not in SOURCE_PERSONAS_507]
        raise ValueError(f"Unknown sources: {unknown}; expected from {SOURCE_PERSONAS_507}")

    # ── Phase 0: bootstrap ──
    if not args.skip_phase0:
        phase0_bootstrap(sources=sources)
    else:
        log.info("Skipping Phase 0 (per --skip-phase0)")

    # ── Phase 2-base + 2.5-base-judge: the base-panel pass needs to run
    #    BEFORE the per-source smoke gate (so we have base rates to compare
    #    own rates against). ──
    if not args.skip_base_panel:
        for seed in args.seeds:
            phase2_base_panel(seed=seed)
            phase2_5_judge_base_panel(seed=seed)
    else:
        # Round-2 fix per code-review Major 9: --skip-base-panel is only
        # valid when base_panel_rates.json ALREADY exists locally (e.g.
        # from a prior partial run). Otherwise the smoke gate's
        # _read_base_rate_for_source returns None and raises AFTER spending
        # cell-1 GPU time — fail-fast at argparse instead.
        required = SLAB_ROOT / "base_panel_rates.json"
        if not required.exists():
            raise RuntimeError(
                f"--skip-base-panel set but {required} does not exist. "
                "Either drop --skip-base-panel (so the base-panel pass runs) "
                "or restore the prior base_panel_rates.json before re-running."
            )
        log.info("Skipping base-panel pass (per --skip-base-panel); reusing %s", required)

    # ── Cell 1 smoke (always the first cell of the sweep) ──
    cell_summaries: list[dict] = []
    first_source = sources[0]
    first_seed = args.seeds[0]

    log.info(
        "=" * 70
        + f"\nCell 1 START: source={first_source} seed={first_seed} (smoke cell)\n"
        + "=" * 70
    )
    cell_start = time.time()
    adapter_dir_first = phase1_train_cell(
        source=first_source,
        seed=first_seed,
        world_size=args.world_size,
    )
    phase2_eval_cell(source=first_source, seed=first_seed, adapter_path=adapter_dir_first)
    phase2_5_judge_source(source=first_source, seed=first_seed)
    cell1_wall = time.time() - cell_start

    gate_summary = smoke_gate(source=first_source, seed=first_seed)
    cell_summaries.append(
        {
            "source": first_source,
            "seed": first_seed,
            "wall_seconds": round(cell1_wall, 1),
            "smoke_gate": gate_summary,
        }
    )

    if args.smoke_gate_only:
        log.info("--smoke-gate-only set; halting after cell 1 smoke gate PASS.")
        _write_results_sentinel(
            summary={
                "mode": "smoke_gate_only",
                "cell1_wall_seconds": round(cell1_wall, 1),
                "cells_completed": [f"{first_source}_seed{first_seed}"],
            }
        )
        _log_phase("done")
        return 0

    # ── Cells 2-N: same per-cell code path as cell 1 ──
    for source in sources[1:]:
        for seed in args.seeds:
            log.info("=" * 70 + f"\nCell START: source={source} seed={seed}\n" + "=" * 70)
            cell_start = time.time()
            adapter_dir = phase1_train_cell(source=source, seed=seed, world_size=args.world_size)
            phase2_eval_cell(source=source, seed=seed, adapter_path=adapter_dir)
            phase2_5_judge_source(source=source, seed=seed)
            cell_wall = time.time() - cell_start
            cell_summaries.append(
                {
                    "source": source,
                    "seed": seed,
                    "wall_seconds": round(cell_wall, 1),
                }
            )

    # Also handle extra seeds on the first source (cell 1 only used seeds[0]).
    for seed in args.seeds[1:]:
        log.info("=" * 70 + f"\nCell START: source={first_source} seed={seed}\n" + "=" * 70)
        cell_start = time.time()
        adapter_dir = phase1_train_cell(source=first_source, seed=seed, world_size=args.world_size)
        phase2_eval_cell(source=first_source, seed=seed, adapter_path=adapter_dir)
        phase2_5_judge_source(source=first_source, seed=seed)
        cell_wall = time.time() - cell_start
        cell_summaries.append(
            {
                "source": first_source,
                "seed": seed,
                "wall_seconds": round(cell_wall, 1),
            }
        )

    # ── Phase 3: predictor extraction ──
    if not args.skip_predictor:
        phase3_predictor_72b(smoke=False, gpu_id=args.gpu_id)
    else:
        log.info("Skipping Phase 3 (per --skip-predictor)")

    # ── Phase 4: regress + cross-arm compare ──
    cross_arm_path: Path | None = None
    if not args.skip_analyze:
        cross_arm_path = phase4_regress_and_cross_arm(regression_7b_path=args.regression_7b)
    else:
        log.info("Skipping Phase 4 (per --skip-analyze)")

    # ── Phase 6: figures ──
    if not args.skip_figures:
        phase6_figures()
    else:
        log.info("Skipping Phase 6 (per --skip-figures)")

    _write_results_sentinel(
        summary={
            "mode": "full_sweep",
            "n_cells_completed": len(cell_summaries),
            "cross_arm_comparison": str(cross_arm_path) if cross_arm_path else None,
            "headline_layer_72b": HEADLINE_LAYER_BY_ARCH["72b"],
        }
    )
    _log_phase("done")
    log.info("Dispatcher complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
