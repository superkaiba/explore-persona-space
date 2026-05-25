#!/usr/bin/env python3
"""Compute base-model (or per-checkpoint) predictors for issue #385.

Two outputs:

(a) L20 cosine-to-librarian over a 28-row panel (librarian + 19 personas + 8
    non-persona contexts). Reuses the centroid protocol from
    ``experiments/phase_minus1_persona_vectors/extract_persona_vectors.py``:
    forward each (system, user-prompt) pair, take the last-token hidden state
    at layer 20, mean-pool over the 20 PROMPTS per row, L2-normalize, cosine
    against the librarian centroid. The 19-persona rows are reused verbatim
    from the cached ``cosine_matrix.json`` (sha256 pinned to
    ``c1a8050744e06c60fc56ca88582324ec3c70c29df39df2f29fb814e905161b0f``); only
    the 8 context centroids are computed fresh.

(b) Completion JS-divergence-to-librarian over the same 28-row panel. Reuses
    ``src/explore_persona_space/analysis/divergence.py`` (the same rig as
    #341 and #207 stage-5). One greedy 256-token response per PROMPT is
    teacher-forced under each of the 28 system prompts; per-prompt JS matrices
    are averaged into a single 28x28 matrix; the librarian row gives JS-to-source
    for the 27 bystanders.

Architecture (round-5 refactor):

This script is a THIN ORCHESTRATOR. It does NOT import torch, transformers,
peft, or vllm. Instead, each GPU phase runs in a fresh subprocess via three
helper scripts:

  - ``scripts/_i385_cosine_runner.py`` — HF L20 centroids + cosine
  - ``scripts/_i385_greedy_runner.py`` — vLLM greedy generation
  - ``scripts/_i385_js_runner.py``     — HF teacher-forced JS divergence

Why subprocesses? Round-4 + round-5 validation showed that loading both
HF Transformers and vLLM in the same Python process is impossible:

  - HF first / vLLM second → ``Engine core initialization failed`` /
    ``pynvml.nvmlDeviceGetHandleByIndex: NVMLError_InvalidArgument``
    inside vLLM's EngineCore worker.
  - vLLM first / HF second → ``caching_allocator_warmup`` crash / 'No
    CUDA GPUs available' on subsequent torch.cuda lazy_init.

Even spawning vLLM via ``subprocess.run`` from a torch-inited parent FAILS
(the parent's CUDA driver context propagates through OS-level inheritance
in a way that CUDA_VISIBLE_DEVICES + start_new_session can't break).

The only architecture that worked: the parent (this script) holds NO torch
state. Each phase is launched as a clean subprocess. The parent's job is
glue — build the panel, verify the pinned cosine matrix, dispatch helpers,
read their JSON outputs, assemble the final payload.

Modes:

- ``base`` (default): compute predictors on the unfine-tuned Qwen2.5-7B-Instruct.
  One-shot; output ``eval_results/issue_385/predictors_base.json``.
- ``per-checkpoint``: compute the same predictors with a LoRA adapter applied,
  for each ``--steps`` value. Output
  ``eval_results/issue_385/predictors_per_checkpoint.json``. Per-checkpoint
  rows are written to disk AS EACH STEP COMPLETES (CLAUDE.md "checkpoint per
  phase" rule, incident #377).

Usage:
    # Base-model predictors (run ONCE before training)
    uv run python scripts/compute_i385_predictors.py --mode base \\
      --output eval_results/issue_385/predictors_base.json

    # Per-checkpoint diagnostic (run AFTER training)
    uv run python scripts/compute_i385_predictors.py --mode per-checkpoint \\
      --run-dir <RUN_DIR> \\
      --steps 5,10,25,50,75,100,150,200,300,400,600,800,1200,1600 \\
      --output eval_results/issue_385/predictors_per_checkpoint.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SOURCE_PERSONA = "librarian"
ANCHOR_PERSONA = "no_persona"
LAYER = 20

COSINE_MATRIX_PATH = (
    PROJECT_ROOT / "experiments" / "phase_minus1_persona_vectors" / "cosine_matrix.json"
)
COSINE_MATRIX_SHA256_PIN = "c1a8050744e06c60fc56ca88582324ec3c70c29df39df2f29fb814e905161b0f"

GREEDY_RESPONSE_MAX_TOKENS = 256
GREEDY_SEED = 42

COSINE_RUNNER = PROJECT_ROOT / "scripts" / "_i385_cosine_runner.py"
GREEDY_RUNNER = PROJECT_ROOT / "scripts" / "_i385_greedy_runner.py"
JS_RUNNER = PROJECT_ROOT / "scripts" / "_i385_js_runner.py"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _verify_cosine_matrix_pin() -> dict:
    """Verify the cached cosine matrix matches the plan-pinned sha256."""
    if not COSINE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Cached cosine matrix not found at {COSINE_MATRIX_PATH}. "
            "Restore via: git show b623f11e:experiments/phase_minus1_persona_vectors/"
            "cosine_matrix.json > experiments/phase_minus1_persona_vectors/cosine_matrix.json"
        )
    actual = _sha256(COSINE_MATRIX_PATH)
    if actual != COSINE_MATRIX_SHA256_PIN:
        raise RuntimeError(
            f"cosine_matrix.json sha256 mismatch.\n"
            f"  expected: {COSINE_MATRIX_SHA256_PIN}\n"
            f"  actual:   {actual}\n"
            "Restore via: git show b623f11e:experiments/phase_minus1_persona_vectors/"
            "cosine_matrix.json > experiments/phase_minus1_persona_vectors/cosine_matrix.json"
        )
    logger.info("cosine_matrix.json sha256 OK (%s)", actual)
    with open(COSINE_MATRIX_PATH) as f:
        return json.load(f)


# ── Persona + context panel ────────────────────────────────────────────────────
def _load_persona_panel() -> tuple[list[tuple[str, str]], list[str]]:
    """Load the canonical 20-row persona panel from extract_persona_vectors.py.

    Reads the PERSONAS and PROMPTS literals via AST WITHOUT executing the
    module. The module top-level does ``import torch`` + ``from transformers
    import ...`` which would contaminate the orchestrator's CUDA state and
    re-introduce the symptom the subprocess architecture exists to avoid.
    """
    import ast

    src_path = (
        PROJECT_ROOT / "experiments" / "phase_minus1_persona_vectors" / "extract_persona_vectors.py"
    )
    if not src_path.exists():
        raise FileNotFoundError(f"Cannot find {src_path}")
    src = src_path.read_text()
    tree = ast.parse(src)
    personas = None
    prompts = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "PERSONAS":
                        personas = ast.literal_eval(node.value)
                    elif target.id == "PROMPTS":
                        prompts = ast.literal_eval(node.value)
    if personas is None or prompts is None:
        raise RuntimeError(
            f"PERSONAS or PROMPTS not found in {src_path} (PERSONAS={personas is not None}, "
            f"PROMPTS={prompts is not None})"
        )
    return list(personas), list(prompts)


def _load_context_panel() -> list[tuple[str, str]]:
    """Load the 8-row non-persona-context panel from scripts/build_i181_data.py."""
    import ast

    build_path = PROJECT_ROOT / "scripts" / "build_i181_data.py"
    if not build_path.exists():
        raise FileNotFoundError(f"Cannot find {build_path}")
    src = build_path.read_text()
    tree = ast.parse(src)
    family_mates = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "FAMILY_MATES" for t in node.targets
        ):
            family_mates = ast.literal_eval(node.value)
            break
    if family_mates is None:
        raise RuntimeError(f"FAMILY_MATES not found in {build_path}")

    contexts: list[tuple[str, str]] = []
    for family in ("task", "instruction", "context", "format"):
        family_list = family_mates[family]
        if len(family_list) < 2:
            raise RuntimeError(f"FAMILY_MATES['{family}'] has <2 entries; cannot pick 2 per family")
        contexts.extend(family_list[:2])
    if len(contexts) != 8:
        raise RuntimeError(f"Expected 8 context entries, got {len(contexts)}")
    return contexts


def _build_panel() -> tuple[list[tuple[str, str]], list[str]]:
    """Build the full 28-row panel: librarian + 19 bystander personas + 8 contexts."""
    personas, prompts = _load_persona_panel()
    contexts = _load_context_panel()

    panel: list[tuple[str, str]] = []
    src_entry = next(((n, t) for n, t in personas if n == SOURCE_PERSONA), None)
    if src_entry is None:
        raise RuntimeError(f"Source persona '{SOURCE_PERSONA}' not in persona panel")
    panel.append(src_entry)
    for name, text in personas:
        if name == SOURCE_PERSONA:
            continue
        panel.append((name, text))
    panel.extend(contexts)

    names = [n for n, _ in panel]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate names in panel: {names}")
    if len(panel) != 28:
        raise RuntimeError(f"Expected 28 panel rows, got {len(panel)}")
    return panel, prompts


def _panel_to_json(panel: list[tuple[str, str]]) -> list[dict]:
    return [{"name": n, "system_prompt": t} for n, t in panel]


# ── Subprocess helper dispatch ────────────────────────────────────────────────


def _run_helper(
    helper: Path,
    payload: dict,
    log_prefix: str,
    keep_tempdir_on_failure: bool = True,
) -> dict:
    """Run a helper script as a subprocess and return its parsed JSON output.

    Each helper is a leaf script that performs ONE GPU phase in a clean
    Python process. The orchestrator never imports torch/vllm directly, so
    each helper invocation gets a fresh CUDA driver state (round-5 architecture).

    Args:
        helper: Path to the helper script (one of _i385_*_runner.py).
        payload: Dict serialized as the helper's --input-path JSON.
        log_prefix: Short tag for log messages (e.g. "cosine", "greedy", "js").
        keep_tempdir_on_failure: If True, preserve the tempdir on non-zero
            exit so the user can re-run the helper standalone for debugging.

    Returns:
        The parsed dict from the helper's --output-path JSON.

    Raises:
        FileNotFoundError: If the helper script doesn't exist.
        RuntimeError: If the helper exits non-zero, the output file is
            missing, or the JSON is malformed.
    """
    if not helper.exists():
        raise FileNotFoundError(f"Helper script missing: {helper}")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"i385_{log_prefix}_"))
    input_path = tmp_dir / "input.json"
    output_path = tmp_dir / "output.json"
    input_path.write_text(json.dumps(payload, indent=2))

    logger.info(
        "[%s] spawning helper subprocess (input=%s)",
        log_prefix,
        input_path,
    )
    cmd = [
        "uv",
        "run",
        "python",
        str(helper),
        "--input-path",
        str(input_path),
        "--output-path",
        str(output_path),
    ]
    # Force CUDA_VISIBLE_DEVICES=0 + new session so each helper gets a
    # predictable single-GPU view, regardless of what the calling shell set.
    subprocess_env = dict(os.environ)
    subprocess_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    t0 = time.time()
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(PROJECT_ROOT),
            env=subprocess_env,
            start_new_session=True,
        )
    except subprocess.CalledProcessError as exc:
        msg = (
            f"[{log_prefix}] helper subprocess failed (exit={exc.returncode}). "
            f"Standalone re-run:\n"
            f"  uv run python {helper} --input-path {input_path} --output-path {output_path}"
        )
        if not keep_tempdir_on_failure:
            try:
                input_path.unlink()
                tmp_dir.rmdir()
            except OSError:
                pass
        raise RuntimeError(msg) from exc
    logger.info("[%s] helper completed in %.1fs", log_prefix, time.time() - t0)

    if not output_path.exists():
        raise RuntimeError(
            f"[{log_prefix}] helper succeeded but output file missing: {output_path}"
        )
    result = json.loads(output_path.read_text())

    try:
        input_path.unlink()
        output_path.unlink()
        tmp_dir.rmdir()
    except OSError as exc:
        logger.warning("Failed to clean up tempdir %s: %s", tmp_dir, exc)

    return result


def _run_cosine(adapter_path: str | None, panel_json: list[dict], prompts: list[str]) -> dict:
    """Spawn the cosine helper for either base model (adapter=None) or LoRA adapter."""
    payload = {
        "model": BASE_MODEL,
        "adapter_path": adapter_path,
        "panel": panel_json,
        "prompts": prompts,
        "layer": LAYER,
        "hf_token": os.environ.get("HF_TOKEN"),
    }
    return _run_helper(COSINE_RUNNER, payload, log_prefix="cosine")


def _run_greedy(panel: list[tuple[str, str]], prompts: list[str]) -> dict[str, str]:
    """Spawn the vLLM greedy helper anchored on the no_persona system prompt."""
    anchor_entry = next(((n, t) for n, t in panel if n == ANCHOR_PERSONA), None)
    if anchor_entry is None:
        raise RuntimeError(
            f"Anchor persona '{ANCHOR_PERSONA}' not in panel; the JS-divergence "
            f"baseline must be anchored on no_persona (plan §5.4(b) / §6)."
        )
    anchor_sys_text = anchor_entry[1]
    if anchor_sys_text != "":
        raise RuntimeError(
            f"Anchor persona '{ANCHOR_PERSONA}' has non-empty system prompt "
            f"({anchor_sys_text!r}); refusing to silently drift the JS baseline."
        )

    payload = {
        "model": BASE_MODEL,
        "anchor_sys_text": anchor_sys_text,
        "prompts": list(prompts),
        "max_tokens": GREEDY_RESPONSE_MAX_TOKENS,
        "seed": GREEDY_SEED,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "dtype": "bfloat16",
        "max_model_len": 4096,
    }
    out = _run_helper(GREEDY_RUNNER, payload, log_prefix="greedy")
    responses_raw = out.get("responses", {})
    if not isinstance(responses_raw, dict):
        raise RuntimeError(
            f"greedy helper output malformed: 'responses' is {type(responses_raw).__name__}"
        )
    responses: dict[str, str] = {}
    for prompt in prompts:
        if prompt not in responses_raw:
            raise RuntimeError(
                f"greedy helper output missing prompt: {prompt!r}. "
                f"Got keys: {list(responses_raw.keys())[:3]}..."
            )
        text = responses_raw[prompt]
        if not text:
            logger.warning("Empty greedy response for prompt %r", prompt[:60])
        responses[prompt] = text
    logger.info(
        "[greedy] %d responses loaded (n_empty=%d)",
        len(responses),
        out.get("n_empty", -1),
    )
    return responses


def _run_js(
    adapter_path: str | None,
    panel_json: list[dict],
    prompts: list[str],
    greedy_responses: dict[str, str],
    tf_batch: int,
) -> dict[str, float]:
    """Spawn the JS-divergence helper for base or LoRA adapter."""
    payload = {
        "model": BASE_MODEL,
        "adapter_path": adapter_path,
        "panel": panel_json,
        "prompts": prompts,
        "greedy_responses": greedy_responses,
        "tf_batch": tf_batch,
        "source_persona": SOURCE_PERSONA,
        "hf_token": os.environ.get("HF_TOKEN"),
    }
    out = _run_helper(JS_RUNNER, payload, log_prefix="js")
    js = out.get("js_to_source", {})
    if not isinstance(js, dict):
        raise RuntimeError(f"js helper output malformed: 'js_to_source' is {type(js).__name__}")
    return js


# ── Orchestrators ─────────────────────────────────────────────────────────────


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", path)


def _metadata(extra: dict | None = None) -> dict:
    md: dict = {
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_model": BASE_MODEL,
        "source_persona": SOURCE_PERSONA,
        "layer": LAYER,
        "cosine_matrix_sha256": COSINE_MATRIX_SHA256_PIN,
    }
    if extra:
        md.update(extra)
    return md


def run_base_mode(args: argparse.Namespace) -> None:
    # ENV var loading: we read .env via stdlib only here. orchestrate.env
    # has heavy imports; this script is now torch-free at module level so we
    # avoid pulling it.
    _load_dotenv_lightweight()

    cached_cos = _verify_cosine_matrix_pin()

    panel, prompts = _build_panel()
    panel_names = [n for n, _ in panel]
    panel_json = _panel_to_json(panel)
    logger.info("Panel: %d rows (source=%s, others=%d)", len(panel), panel_names[0], len(panel) - 1)
    logger.info("Prompts: %d", len(prompts))

    cached_layer = cached_cos.get(f"layer_{LAYER}", {})
    cached_names = cached_layer.get("persona_names", [])
    cached_matrix = cached_layer.get("matrix", [])
    if not (cached_names and cached_matrix and SOURCE_PERSONA in cached_names):
        raise RuntimeError(
            f"cosine_matrix.json layer_{LAYER} missing persona_names / matrix / "
            f"source persona {SOURCE_PERSONA!r}; cannot use pinned predictor."
        )
    cached_src_idx = cached_names.index(SOURCE_PERSONA)
    pinned_cos_to_source: dict[str, float] = {
        name: float(cached_matrix[cached_src_idx][i]) for i, name in enumerate(cached_names)
    }
    logger.info(
        "Loaded pinned cosine_to_source from cached matrix for %d persona rows (source=%s)",
        len(pinned_cos_to_source),
        SOURCE_PERSONA,
    )

    # ── Phase 1: HF cosine subprocess ──────────────────────────────────────
    t0 = time.time()
    cosine_out = _run_cosine(adapter_path=None, panel_json=panel_json, prompts=prompts)
    fresh_cos_to_source = cosine_out["cosine_to_source"]
    logger.info(
        "Fresh L20 cosine pass done in %.1fs (cross-validates pin + computes contexts)",
        time.time() - t0,
    )

    DRIFT_TOL = 5e-3
    max_diff = 0.0
    worst_name = ""
    drift_report: dict[str, float] = {}
    for name in cached_names:
        if name not in fresh_cos_to_source:
            continue
        diff = abs(fresh_cos_to_source[name] - pinned_cos_to_source[name])
        drift_report[name] = diff
        if diff > max_diff:
            max_diff = diff
            worst_name = name
    logger.info(
        "Pinned-vs-fresh cosine drift: max_abs_diff=%.6e on %s (tol=%.0e)",
        max_diff,
        worst_name or "(none)",
        DRIFT_TOL,
    )
    if max_diff > DRIFT_TOL:
        sorted_drift = sorted(drift_report.items(), key=lambda kv: -kv[1])[:5]
        raise RuntimeError(
            "Fresh L20 cosines drift from the pinned cosine_matrix.json by "
            f"max_abs_diff={max_diff:.6e} > tol={DRIFT_TOL:.0e}.\n"
            "The pinned file is the canonical predictor; a drift this large "
            "means the chat-template / tokenizer / model build differs from "
            "the one that produced the pin. Diagnose before re-running.\n"
            f"Top-5 drifters: {sorted_drift}"
        )

    cos_to_source: dict[str, float] = dict(pinned_cos_to_source)
    for name in fresh_cos_to_source:
        if name not in cos_to_source:
            cos_to_source[name] = fresh_cos_to_source[name]
    logger.info(
        "cosine_to_source: %d entries (%d from pin, %d fresh contexts)",
        len(cos_to_source),
        len(pinned_cos_to_source),
        len(cos_to_source) - len(pinned_cos_to_source),
    )

    # ── Phase 2: vLLM greedy subprocess ────────────────────────────────────
    t0 = time.time()
    greedy = _run_greedy(panel, prompts)
    logger.info("Greedy generation done in %.1fs (%d responses)", time.time() - t0, len(greedy))

    # ── Phase 3: HF JS-divergence subprocess ───────────────────────────────
    t0 = time.time()
    js_to_source = _run_js(
        adapter_path=None,
        panel_json=panel_json,
        prompts=prompts,
        greedy_responses=greedy,
        tf_batch=args.tf_batch,
    )
    logger.info("JS pass done in %.1fs", time.time() - t0)

    cosine_provenance = {
        name: (
            "pinned_cosine_matrix" if name in pinned_cos_to_source else "fresh_context_recompute"
        )
        for name in cos_to_source
    }
    payload = {
        "metadata": _metadata(
            {
                "mode": "base",
                "cosine_anchor_source": "no_persona",
                "js_anchor_source": ANCHOR_PERSONA,
                "cosine_pin_drift_tol": DRIFT_TOL,
                "n_cosine_from_pin": len(pinned_cos_to_source),
                "n_cosine_from_fresh": len(cos_to_source) - len(pinned_cos_to_source),
            }
        ),
        "panel": panel_json,
        "prompts": prompts,
        "cosine_to_source": cos_to_source,
        "cosine_to_source_provenance": cosine_provenance,
        "js_to_source": js_to_source,
        "greedy_responses": greedy,
    }
    _write_json(payload, Path(args.output))


def run_per_checkpoint_mode(args: argparse.Namespace) -> None:
    _load_dotenv_lightweight()
    _verify_cosine_matrix_pin()  # fail fast even though base values aren't reused here

    if not args.run_dir:
        raise SystemExit("--run-dir is required in per-checkpoint mode")
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"--run-dir {run_dir} does not exist")
    steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    if not steps:
        raise SystemExit("--steps must be a non-empty comma-separated list")

    panel, prompts = _build_panel()
    panel_json = _panel_to_json(panel)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate greedy responses ONCE on the base model (anchor stays fixed).
    t0 = time.time()
    greedy = _run_greedy(panel, prompts)
    logger.info("Greedy generation done in %.1fs (%d responses)", time.time() - t0, len(greedy))

    # ── Per-checkpoint loop (per-phase persist; CLAUDE.md / incident #377) ─
    sidecar_path = output_path.with_suffix(output_path.suffix + ".jsonl")
    sidecar_path.write_text("")
    rows: list[dict] = []

    for step in steps:
        adapter_path = run_dir / f"checkpoint-{step}"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Checkpoint dir missing: {adapter_path}")

        t_step = time.time()
        cosine_out = _run_cosine(
            adapter_path=str(adapter_path), panel_json=panel_json, prompts=prompts
        )
        cos_to_source = cosine_out["cosine_to_source"]
        logger.info("Step %d cosine pass done in %.1fs", step, time.time() - t_step)

        t_step = time.time()
        js_to_source = _run_js(
            adapter_path=str(adapter_path),
            panel_json=panel_json,
            prompts=prompts,
            greedy_responses=greedy,
            tf_batch=args.tf_batch,
        )
        logger.info("Step %d JS pass done in %.1fs", step, time.time() - t_step)

        row = {
            "step": step,
            "adapter_path": str(adapter_path),
            "cosine_to_source": cos_to_source,
            "js_to_source": js_to_source,
        }
        with open(sidecar_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        logger.info("Step %d row persisted to %s", step, sidecar_path)

    payload = {
        "metadata": _metadata({"mode": "per-checkpoint", "run_dir": str(run_dir), "steps": steps}),
        "panel": panel_json,
        "prompts": prompts,
        "rows": rows,
    }
    _write_json(payload, output_path)


def _load_dotenv_lightweight() -> None:
    """Load .env from PROJECT_ROOT via python-dotenv ONLY.

    Avoid pulling explore_persona_space.orchestrate.env which transitively
    imports torch through the project's training stack — that would re-add
    GPU state to this orchestrator process.
    """
    try:
        from dotenv import load_dotenv

        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        logger.warning("python-dotenv not available; skipping .env load")


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("base", "per-checkpoint"),
        default="base",
        help="base = compute on unfine-tuned base model; per-checkpoint = recompute "
        "with LoRA adapter at each --steps value (diagnostic).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON (e.g. eval_results/issue_385/predictors_base.json).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="(per-checkpoint mode) run directory containing checkpoint-{step}/ adapter dirs.",
    )
    parser.add_argument(
        "--steps",
        default="",
        help="(per-checkpoint mode) comma-separated list of step values to evaluate.",
    )
    parser.add_argument(
        "--tf-batch",
        type=int,
        default=8,
        help="Teacher-force sub-batch size for divergence computation (default 8).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "base":
        run_base_mode(args)
    else:
        run_per_checkpoint_mode(args)


if __name__ == "__main__":
    main()
