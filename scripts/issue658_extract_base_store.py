#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, ā, θ, Σ, →, ρ, ×) in scientific docstrings + log messages.
"""Issue #658 GPU phase: base-model activation store (v0(C) / c_C / r_B / Σ_c).

Phase 0 + Phase 1 of the leakage-predictor campaign — Qwen2.5-7B-Instruct (θ0),
NO training. Extends ``scripts/issue594_extract_context_vectors.py`` (imports its
``LayerCapture`` hook pattern, ``load_battery`` / ``messages_for_instance`` /
manifest+sentinel+upload helpers); it does NOT re-implement them.

The #594 extractor captures the LAST INPUT TOKEN (prompt-side ``c_C``). This
extension adds an on-policy GENERATE-then-CAPTURE pass for the ANSWER-side
``v0(C)`` (mean over the model's own generated answer tokens), plus the three
NEW reads the predictor chain needs:

GPU phases (plan §4.0 DAG):

- **G1+G2 — v0(C)**: vLLM batched greedy generate over 50 ctx × 48 probes; then
  teacher-force each (prompt + answer) back through HF with ``LayerCapture``
  hooks on all 28 layers, capturing the residual span at the ANSWER-token
  positions. Per-(C,probe) answer spans stored fp16 for the noise-floor
  resampling (N1) AND the CPU-side attn-pool fit (P1). Per-(C) probe-mean
  ``v0`` summaries (mean/last/maxp recipes) saved per layer.
- **G3 — c_C mean-prompt ablation**: one extra ``c_C`` slot (mean over prompt
  tokens). The last-input-token ``c_C`` is REUSED from the #594 HF store (plan
  §11 reuse-fitness CONFIRMED); only the mean-over-prompt ablation is new here.
- **G4 — r_B diff-in-means**: per behavior with a natural (D_B, D_{B̄}) contrast
  (``issue658_common.rb_columns``), forward the paired prompt sets and compute
  the diff of mean answer-side activations, all 28 layers. Columns without a
  contrast (marker, format_style, …) are DROPPED from A3.3 (explicit).
- **G5 — Σ_c background corpus**: ≥3k contexts via ``project_corpus_v2``,
  prompt-only forward → second-moment Σ_c for downstream phases.

``--smoke`` runs the IDENTICAL dispatcher with a tiny single-cell slice (4 ctx ×
4 probes × n_samples=1, 4 layers, ~200 Σ_c contexts) end-to-end → architecture
verdict PASS_UNIFIED (the smoke IS the sweep with --n-ctx 4). Local CPU smoke
uses a tiny same-family model + ``--device cpu --no-vllm`` (HF greedy generate).

Pod-side contract: ``[phase=...]`` log lines ending in ``[phase=done]`` + a
``poll_pipeline.py``-conformant end-of-run sentinel.

Usage (plan §10 launch command)::

    nohup uv run python scripts/issue658_extract_base_store.py \\
        --battery data/issue594/battery.json \\
        --out-dir data/issue_658/store --gpu-id 0 \\
        > logs/issue658_extract.log 2>&1 &

    # local CPU smoke (tiny same-family model, HF generate, no upload):
    uv run python scripts/issue658_extract_base_store.py --smoke \\
        --model Qwen/Qwen2.5-0.5B-Instruct --expected-layers 24 \\
        --expected-hidden 896 --device cpu --no-vllm --n-layers-smoke 4 \\
        --out-dir /tmp/issue658_cpu_smoke --no-upload --wandb-mode disabled
"""

from __future__ import annotations

# VLLM_WORKER_MULTIPROC_METHOD=spawn BEFORE any `import vllm` — the dispatcher's
# main() loads transformers/tokenizer before LLM(), which poisons a fork()ed
# EngineCore (gotchas.md #628). Set at module top, before the lazy vllm import.
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import argparse
import logging
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Cross-script helper imports hoisted to module top so a missing symbol crashes
# at process start, never inside a smoke-skipped branch (gotchas.md #606).
from issue404_common import (  # noqa: E402
    fetch_betley_main_8,
    fetch_preregistered_probes,
    reproducibility_metadata,
)
from issue594_common import (  # noqa: E402
    DEFAULT_MODEL as I594_MODEL,
)
from issue594_common import (  # noqa: E402
    load_battery,
    messages_for_instance,
)
from issue594_extract_context_vectors import LayerCapture  # noqa: E402
from issue658_common import (  # noqa: E402
    DEFAULT_MODEL,
    E0_COLUMNS,
    EVAL_RESULTS_DIR,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    HF_DATA_REPO,
    HF_OVERFLOW_REPO,
    HF_PREFIX,
    MARKER_TOKEN_ID,
    SUMMARY_RECIPES,
    V0_MAX_NEW_TOKENS,
    dump_json,
    partition_contexts,
    rb_columns,
    sha256_file,
    stable_hash,
    summarize_answer_span,
)

# Catch a #594 model drift early (the c_C reuse pins both to the same θ0).
assert I594_MODEL == DEFAULT_MODEL, f"#594 model {I594_MODEL} != #658 {DEFAULT_MODEL}"

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue658_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SENTINEL_SCHEMA_VERSION = 1


def phase(name: str) -> None:
    """Emit a poll_pipeline.py-parseable phase line (PHASE_RE on the log tail)."""
    print(f"[phase={name}]", flush=True)


# The primary-deliverable tensors + index the §6.5 sha-pinned manifest records.
# Each is a single file under ``out_dir`` shipped to HF by ``upload_store``; the
# downstream content-identity check (artifact-reuse rule (f)/(g), #600) verifies
# the HF tensors named here are the tensors THIS run produced.
MANIFEST_PINNED_FILES: tuple[str, ...] = (
    "v0_summaries.pt",
    "r_b.pt",
    "sigma_c.pt",
    "answer_spans/index.json",
)


def build_files_sha_map(out_dir: Path) -> dict[str, dict]:
    """SHA-256-pin every primary-deliverable file present under ``out_dir``.

    Returns ``{rel_path: {"sha256": <64-hex>, "bytes": <int>}}`` for each file in
    ``MANIFEST_PINNED_FILES`` that exists (``sigma_c.pt`` is absent under
    ``--skip-sigma``, so a missing pinned file is recorded as ``present: False``
    rather than crashing — the manifest then truthfully reports the descope).
    Computed AFTER all tensors are written (the ``assemble`` phase) so the SHAs
    are over the FINAL on-disk bytes.
    """
    files: dict[str, dict] = {}
    for rel in MANIFEST_PINNED_FILES:
        p = out_dir / rel
        if p.is_file():
            files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size, "present": True}
        else:
            files[rel] = {"sha256": None, "bytes": None, "present": False}
    return files


# ── Answer-span capture extension of #594's LayerCapture ─────────────────────


class AnswerSpanCapture(LayerCapture):
    """LayerCapture that returns the ANSWER-token span, not just the last token.

    #594's ``last_token_stack`` keeps only position -1 (the prompt-side c_C).
    For v0(C) we need the residual span over the model's own generated answer
    tokens, so the hooks keep the full (1, T, H) per layer and this method
    slices [answer_start:answer_end).
    """

    def answer_span_stack(self, n_layers: int, answer_start: int, answer_end: int) -> torch.Tensor:
        """(L, S, H) fp16 CPU stack of the answer-span activations per layer.

        ``answer_start:answer_end`` indexes the teacher-forced (prompt+answer)
        position axis — the answer tokens only.
        """
        assert 0 <= answer_start < answer_end, (answer_start, answer_end)
        vecs = [
            self.latest[li][0, answer_start:answer_end, :].to(torch.float16).cpu()
            for li in range(n_layers)
        ]
        self.latest.clear()
        return torch.stack(vecs)  # (L, S, H)

    def mean_prompt_stack(self, n_layers: int, prompt_len: int) -> torch.Tensor:
        """(L, H) fp32 CPU stack: mean over the PROMPT tokens (c_C mean ablation)."""
        vecs = [
            self.latest[li][0, :prompt_len, :].float().mean(dim=0).cpu() for li in range(n_layers)
        ]
        self.latest.clear()
        return torch.stack(vecs)  # (L, H)


# ── G1: vLLM batched generation ──────────────────────────────────────────────


def build_prompts(
    tokenizer, instances: list[dict], probes: list[str]
) -> tuple[list[str], list[tuple[str, str]]]:
    """Templated prompt strings for every (context, probe) cell + the index.

    Returns (prompt_texts, index) where index[i] = (instance_id, probe_text).
    Persona injection is ALWAYS a system turn (messages_for_instance handles it).
    """
    prompts: list[str] = []
    index: list[tuple[str, str]] = []
    for inst in instances:
        for q in probes:
            messages = messages_for_instance(inst, q)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            index.append((inst["id"], q))
    return prompts, index


def vllm_generate(model_name: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    """vLLM batched greedy generation over all prompts in ONE call.

    use_tqdm=False (gotchas.md #613 ZeroDivisionError). Returns one completion
    string per prompt, in order.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.45)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    completions = [o.outputs[0].text for o in outs]
    _reap_vllm(llm)
    return completions


def hf_generate(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    """HF greedy generate fallback (CPU smoke / --no-vllm). Batch-1 (smoke-scale)."""
    completions: list[str] = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        gen = out[0, inputs["input_ids"].shape[1] :]
        completions.append(tokenizer.decode(gen, skip_special_tokens=True))
    return completions


def _reap_vllm(llm) -> None:
    """Reap the vLLM v1 EngineCore worker so the co-resident HF load doesn't OOM.

    gotchas.md vLLM-teardown recipe: engine_core.shutdown() (v1) /
    model_executor.shutdown() (v0 fallback), destroy_process_group, gc,
    empty_cache, ipc_collect, sleep.
    """
    import gc

    try:
        engine = getattr(llm, "llm_engine", None)
        ec = getattr(engine, "engine_core", None)
        if ec is not None and hasattr(ec, "shutdown"):
            ec.shutdown()
        else:
            me = getattr(engine, "model_executor", None)
            if me is not None and hasattr(me, "shutdown"):
                me.shutdown()
    except Exception as e:
        logger.warning("vLLM engine shutdown raised (continuing teardown): %s", e)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(1.0)


# ── G6: E0(C,B) behavior-battery generation ──────────────────────────────────


def load_e0_battery(column_id: str, cap: int, probes: list[str]) -> list[str]:
    """Per-column behavior battery (the probes E0 generates under each context).

    Loaded directly from the tier-2 panels (no per-row API). cap=0 keeps the
    full battery; a positive cap (smoke) truncates. Marker / format / self /
    persona / fact use the neutral Betley probe pool (the localized-behavior
    read happens at the END of the model's OWN answer to a neutral probe).
    """
    if column_id == "broad_em":
        return list(fetch_betley_main_8())
    if column_id == "harmful_compliance":
        return _load_harmful_prompts(cap or 200)
    if column_id == "sycophancy":
        from huggingface_hub import hf_hub_download

        src = hf_hub_download(
            "superkaiba1/explore-persona-space-data",
            "issue411_sycophancy_cosine_gradient/data/wrong_claims/train_200.jsonl",
            repo_type="dataset",
        )
        out = []
        for line in Path(src).read_text().splitlines():
            if line.strip():
                out.append(f"{_load_jsonl_row(line)['wrong_claim']} Right?")
        return out[: cap or len(out)]
    if column_id == "refusal":
        from explore_persona_space.experiments.behavior_testbed_545 import corpora

        payload = _load_json(Path(corpora.build_refusal_panel()))
        merged = payload.get("should_refuse", []) + payload.get("should_not_refuse", [])
        return merged[: cap or len(merged)]
    # marker / format_style / deception / fact_expression / self_report /
    # persona_drift: read at the end of the model's OWN answer to a neutral
    # probe (the localized / structural / self-report behaviors don't need a
    # bespoke battery — the construct is "does the base model do B unprompted").
    pool = probes if probes else list(fetch_betley_main_8())
    return pool[: cap or len(pool)]


# ── G6 resilience (round-2 crash recovery) ───────────────────────────────────
# Rounds 1 + 2 both died at marker context #31 (`f3_icl_json_k2`) with a
# deterministic per-context exception, the GCE EXIT trap powered the VM off, and
# the 30 partial e0_gen/*.json + the log were LOST. These helpers make a
# per-context failure (a) DIAGNOSABLE next run (full traceback to stdout +
# an ``*__marker__ERROR.json`` artifact) and (b) NON-DESTRUCTIVE of the 30
# contexts that already ran (periodic idempotent upload of the partial e0_gen/
# directory to the HF data repo). NONE of the marker mechanics change —
# ``_gen_marker_slot`` (the 4-float slot read, the ` ※` id-83399 token, the
# per-column policy) is untouched; only the LOOP CALLER gains isolation.

E0_PARTIAL_UPLOAD_EVERY = 5  # upload partial e0_gen/ after every N marker contexts


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _empty_cuda_cache() -> None:
    """Defensive: drop cached CUDA blocks between contexts (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_marker_error(e0_dir: Path, context_id: str, exc: Exception) -> Path:
    """Persist a per-context marker failure so downstream tooling sees the gap.

    Companion to the FULL traceback printed to stdout (which the GCE EXIT trap
    tail-dumps to fd3, so it survives even if a later phase still crashes).
    """
    err_path = e0_dir / f"{context_id}__marker__ERROR.json"
    dump_json(
        {
            "context_id": context_id,
            "ts": _now_iso(),
            "exception_type": type(exc).__name__,
            "exception_str": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
        err_path,
    )
    return err_path


def _upload_partial_e0(e0_dir: Path, smoke: bool) -> None:
    """Idempotent partial upload of e0_gen/*.json to the HF data repo.

    Re-uploading the same files overwrites (idempotent). Wrapped in its OWN
    try/except by the CALLER — a transient HF failure must never kill the
    workload (the whole point of the resilience fix is to STOP losing the
    30 good contexts). Lands under the same ``{HF_PREFIX}/e0_gen`` namespace
    the end-of-run ``upload_raw_completions`` uses, so partial and final
    completions are one prefix (no split-brain).
    """
    from huggingface_hub import HfApi

    sub = "e0_gen_smoke" if smoke else "e0_gen"
    path_in_repo = f"{HF_PREFIX}/{sub}"
    HfApi().upload_folder(
        folder_path=str(e0_dir),
        path_in_repo=path_in_repo,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*.json"],
        commit_message=f"issue658: partial e0_gen upload ({'smoke ' if smoke else ''}resilience)",
    )
    logger.info("partial e0_gen upload OK -> %s/%s", HF_DATA_REPO, path_in_repo)


def _run_marker_loop(
    model,
    tokenizer,
    instances: list[dict],
    e0_dir: Path,
    n_battery: int,
    compute_marker_slot_stats,
    mnt_fn,
    *,
    smoke: bool = False,
    upload: bool = True,
) -> dict:
    """Marker-column HF slot read per context, with per-context exception isolation.

    Directly invokable (the round-3 resilience test drives it with a stub model).
    For each instance, runs ``_gen_marker_slot`` for the marker column(s) under a
    ``try/except``: on failure the FULL traceback goes to stdout AND an
    ``*__marker__ERROR.json`` is written, then the loop continues to the next
    context (NEVER re-raises). ``torch.cuda.empty_cache()`` runs after every
    context (success AND failure). Partial e0_gen/ is uploaded every
    ``E0_PARTIAL_UPLOAD_EVERY`` successful contexts and once at the end — each
    upload in its own try/except so an upload failure never kills the workload.

    Returns ``{"done": [ids], "errors": [ids]}`` for the caller's telemetry.
    """
    done: list[str] = []
    errors: list[str] = []
    n_since_upload = 0
    for inst in instances:
        ctx_id = inst["id"]
        try:
            for col_id, col in E0_COLUMNS.items():
                if col.dv != "marker_slot_stats":
                    continue
                out_path = e0_dir / f"{ctx_id}__{col_id}.json"
                if out_path.exists():
                    continue
                battery = load_e0_battery(col_id, n_battery, [])
                _gen_marker_slot(
                    model,
                    tokenizer,
                    inst,
                    battery,
                    col,
                    out_path,
                    compute_marker_slot_stats,
                    mnt_fn(col),
                )
            done.append(ctx_id)
            n_since_upload += 1
        except Exception as exc:
            traceback.print_exc()
            print(
                f"[ERROR] marker context {ctx_id}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            _write_marker_error(e0_dir, ctx_id, exc)
            errors.append(ctx_id)
        finally:
            _empty_cuda_cache()
        if upload and n_since_upload >= E0_PARTIAL_UPLOAD_EVERY:
            n_since_upload = 0
            try:
                _upload_partial_e0(e0_dir, smoke)
            except Exception as up_exc:
                print(f"[ERROR] partial upload failed: {up_exc}", flush=True)
    # End-of-loop partial upload (whether or not the next phase runs / crashes).
    if upload:
        try:
            _upload_partial_e0(e0_dir, smoke)
        except Exception as up_exc:
            print(f"[ERROR] partial upload failed: {up_exc}", flush=True)
    return {"done": done, "errors": errors}


def generate_e0_completions(
    model,
    tokenizer,
    instances: list[dict],
    e0_dir: Path,
    *,
    use_vllm: bool,
    model_name: str,
    n_battery: int,
    run,
    max_new_tokens_cap: int = 0,
    n_samples_cap: int = 0,
    smoke: bool = False,
    upload_partial: bool = True,
) -> None:
    """Generate E0(C,B) behavior-battery completions per (context, column).

    Honors col.temperature / col.n_samples PER COLUMN (the round-1 sampling
    concern). Two SMOKE-ONLY clamps keep a CPU smoke tractable WITHOUT touching
    the registry the predictor reads (the per-column temp / n_samples are pinned
    by tests/test_issue658_invariants.py): ``max_new_tokens_cap`` clamps the
    generation LENGTH (so the marker column does not generate 2048 tokens), and
    ``n_samples_cap`` clamps the per-probe SAMPLE count (so broad_em's n=50 is
    not 50 CPU generations). The column's TEMPERATURE is always honored, so the
    sampling POLICY (temp-0 vs temp-0.7/1.0) is exercised end-to-end. Both
    default 0 = honor the column's full values (the real-run default). Persists
    one JSON per (context, column); checkpoint-per-cell.

    vLLM path (round-2 throughput fix): ONE shared ``LLM(...)`` engine for the
    WHOLE judged/structural E0 phase (every context × column × probe), reaped
    ONCE at the end — NOT a fresh engine per (context × column) cell (the round-1
    Major: ~hundreds of engine startups, plan §4.2/§9 mandate one batched
    engine). Per-prompt ``SamplingParams`` carries each column's own
    temperature / n_samples / max_tokens, so the single batched call still
    honors the per-column policy. The marker column reads HF slot logits (vLLM
    exposes no raw logits) so it stays a per-context HF pass; the HF-fallback
    path (CPU smoke / --no-vllm) loops per cell at smoke scale.
    """
    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats

    def _mnt(col):
        return (
            min(col.max_new_tokens, max_new_tokens_cap)
            if max_new_tokens_cap
            else col.max_new_tokens
        )

    def _nsamp(col):
        return min(col.n_samples, n_samples_cap) if n_samples_cap else col.n_samples

    # ── marker column (HF slot read) — per context, no vLLM engine ────────────
    # Per-context exception isolation + periodic partial upload (round-2 crash
    # recovery): a deterministic per-context failure is now diagnosable next run
    # (traceback + *__marker__ERROR.json) and never destroys the contexts that
    # already ran (rounds 1+2 lost 30 good contexts at the #31 crash). The marker
    # MECHANICS (_gen_marker_slot, the slot read, the ` ※` token) are unchanged.
    marker_status = _run_marker_loop(
        model,
        tokenizer,
        instances,
        e0_dir,
        n_battery,
        compute_marker_slot_stats,
        _mnt,
        smoke=smoke,
        upload=upload_partial,
    )
    run.log(
        {
            "e0_marker_contexts_done": len(marker_status["done"]),
            "e0_marker_contexts_errored": len(marker_status["errors"]),
        }
    )
    if marker_status["errors"]:
        logger.warning(
            "G6 marker loop: %d context(s) failed (isolated, see *__marker__ERROR.json): %s",
            len(marker_status["errors"]),
            marker_status["errors"],
        )

    # ── judged_rate / structural columns ──────────────────────────────────────
    # Collect the (context, column, probe) cells still needing generation.
    pending: list[tuple[dict, object, list[str], Path]] = []
    for inst in instances:
        for col_id, col in E0_COLUMNS.items():
            if col.dv == "marker_slot_stats":
                continue
            out_path = e0_dir / f"{inst['id']}__{col_id}.json"
            if out_path.exists():
                continue
            battery = load_e0_battery(col_id, n_battery, [])
            pending.append((inst, col, battery, out_path))

    if use_vllm and pending:
        # Structural / judged cell phase exception isolation (round-2 crash
        # recovery): if the shared-engine batch raises, log the FULL traceback,
        # write a phase-level error file, upload the partial e0_gen/, and return
        # cleanly — NEVER crash hard at this layer (so the marker contexts that
        # already landed are preserved + uploaded).
        try:
            _gen_e0_vllm_shared(
                model_name, tokenizer, pending, _mnt, _nsamp, smoke=smoke, upload=upload_partial
            )
            run.log({"e0_contexts_done": len(instances)})
        except Exception as exc:
            traceback.print_exc()
            print(
                f"[ERROR] structural/judged E0 phase: {type(exc).__name__}: {exc}",
                flush=True,
            )
            _write_marker_error(e0_dir, "_structural_phase", exc)
            if upload_partial:
                try:
                    _upload_partial_e0(e0_dir, smoke)
                except Exception as up_exc:
                    print(f"[ERROR] partial upload failed: {up_exc}", flush=True)
        finally:
            _empty_cuda_cache()
    else:
        # HF fallback (CPU smoke / --no-vllm): per-cell loop at smoke scale.
        for inst, col, battery, out_path in pending:
            cells = _gen_column_samples_hf(
                model, tokenizer, inst, battery, col, _mnt(col), _nsamp(col)
            )
            dump_json(
                {
                    "context_id": inst["id"],
                    "column_id": col.column_id,
                    "dv": col.dv,
                    "temperature": col.temperature,
                    "n_samples": col.n_samples,
                    "cells": cells,
                },
                out_path,
            )
            run.log({"e0_contexts_done": inst["id"]})
        _empty_cuda_cache()
    logger.info("G6 done: E0 generations under %s", e0_dir)


def _gen_e0_vllm_shared(
    model_name, tokenizer, pending, mnt_fn, nsamp_fn, *, smoke: bool = False, upload: bool = True
) -> None:
    """Generate every (context, column) judged/structural cell through ONE engine.

    Builds the prompt list across ALL pending cells, attaches a per-prompt
    ``SamplingParams`` (each column's own temperature / n_samples / max_tokens),
    runs ONE ``llm.generate()`` call, then partitions the outputs back to one
    JSON per (context, column). The engine is reaped ONCE (round-2 throughput
    fix — replaces ~hundreds of per-cell ``LLM()`` startups). Each completion
    carries its length-normalized log-prob (``logp_norm``) — the SECONDARY
    dual-DV companion. Checkpoint-per-cell: each cell JSON is written as soon as
    its slice of the batched output is assembled. Runs ONE partial e0_gen/ upload
    at the end (round-2 crash recovery) — in its own try/except so an upload
    failure never kills the workload.
    """
    from vllm import LLM, SamplingParams

    prompts: list[str] = []
    params: list[SamplingParams] = []
    # ranges[i] = (cell_index, n_prompts_for_cell) so the flat output can be
    # sliced back; cell_meta[cell_index] = (inst, col, battery, out_path).
    cell_spans: list[tuple[int, int, int]] = []  # (cell_idx, start, count)
    for cell_idx, (_inst, col, battery, _out) in enumerate(pending):
        start = len(prompts)
        for q in battery:
            messages = messages_for_instance(_inst, q)
            prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            params.append(
                SamplingParams(
                    n=nsamp_fn(col),
                    temperature=col.temperature,
                    max_tokens=mnt_fn(col),
                    logprobs=0,  # the sampled token's logprob per position (logp_norm)
                )
            )
        cell_spans.append((cell_idx, start, len(battery)))

    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.45)
    try:
        outs = llm.generate(prompts, params, use_tqdm=False)
    finally:
        _reap_vllm(llm)

    for cell_idx, start, count in cell_spans:
        inst, col, battery, out_path = pending[cell_idx]
        cells = []
        for j in range(count):
            o = outs[start + j]
            comps = []
            for c in o.outputs:
                ntok = max(1, len(c.token_ids))
                comps.append({"text": c.text, "logp_norm": float(c.cumulative_logprob) / ntok})
            cells.append({"probe": battery[j], "completions": comps})
        dump_json(
            {
                "context_id": inst["id"],
                "column_id": col.column_id,
                "dv": col.dv,
                "temperature": col.temperature,
                "n_samples": col.n_samples,
                "cells": cells,
            },
            out_path,
        )

    # End-of-branch partial upload (round-2 crash recovery): persist the
    # judged/structural completions to the HF data repo before the next phase
    # (G4/G5/upload) can crash. ``pending`` is non-empty here (the caller gates
    # on it), so ``pending[0][3].parent`` is the e0_gen dir.
    if upload and pending:
        try:
            _upload_partial_e0(pending[0][3].parent, smoke)
        except Exception as up_exc:
            print(f"[ERROR] partial upload failed: {up_exc}", flush=True)


def _gen_marker_slot(
    model, tokenizer, inst, battery, col, out_path, compute_marker_slot_stats, max_new_tokens
):
    """Marker E0: greedy-answer the neutral probes, read the marker slot (4-float).

    The slot is read at the end of the model's OWN greedy answer (on-policy,
    marker-at-end per marker-leakage-measurement.md). One 4-float record per
    probe (logp / z_marker / z_eos / logZ + argmax). ``max_new_tokens`` is the
    (smoke-clamped) generation length.
    """
    contexts = []
    probes_kept = []
    for q in battery:
        messages = messages_for_instance(inst, q)
        tmpl = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(tmpl, return_tensors="pt", padding=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        gen = out[0, inputs["input_ids"].shape[1] :]
        ans = tokenizer.decode(gen, skip_special_tokens=True)
        # The slot stats helper re-tokenizes the context string; pass the full
        # templated prompt + the model's own answer (the marker would appear
        # at the slot immediately after the answer).
        contexts.append(tmpl + ans)
        probes_kept.append(q)
    device = "cuda:0" if model.device.type == "cuda" else "cpu"
    stats = compute_marker_slot_stats(
        model,
        tokenizer,
        contexts,
        " ※",
        eos_token_id=tokenizer.eos_token_id,
        device=device,
        include_argmax=True,
    )
    dump_json(
        {
            "context_id": inst["id"],
            "column_id": col.column_id,
            "dv": col.dv,
            "marker_slot": [{"probe": p, **s} for p, s in zip(probes_kept, stats, strict=True)],
        },
        out_path,
    )


def _gen_column_samples_hf(model, tokenizer, inst, battery, col, max_new_tokens, n_samples):
    """HF-fallback per-probe sampling (CPU smoke / --no-vllm) — NO vLLM engine.

    The production vLLM path runs through the SINGLE shared engine in
    ``_gen_e0_vllm_shared`` (round-2 throughput fix); this fallback is only the
    CPU-smoke / --no-vllm branch and loops per probe at smoke scale. Each
    completion carries its teacher-forced length-normalized log-prob
    (``logp_norm``) — the SECONDARY dual-DV companion. The column's TEMPERATURE
    is always honored. Returns [{probe, completions: [{text, logp_norm}, ...]}].
    """
    cells: list[dict] = []
    prompts = []
    for q in battery:
        messages = messages_for_instance(inst, q)
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    for q, tmpl in zip(battery, prompts, strict=True):
        inputs = tokenizer(tmpl, return_tensors="pt", padding=False).to(model.device)
        comps = []
        for _ in range(n_samples):
            do_sample = col.temperature > 0.0
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=col.temperature if do_sample else None,
                    top_p=None,
                )
            gen_ids = out[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            logp_norm = _hf_completion_logp(model, inputs["input_ids"], gen_ids)
            comps.append({"text": text, "logp_norm": logp_norm})
        cells.append({"probe": q, "completions": comps})
    return cells


def _hf_completion_logp(model, prompt_ids, gen_ids) -> float:
    """Length-normalized log P of the generated tokens under teacher forcing."""
    if gen_ids.shape[0] == 0:
        return 0.0
    full = torch.cat([prompt_ids[0], gen_ids]).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits = model(input_ids=full).logits  # (1, T, V)
    plen = prompt_ids.shape[1]
    # predict gen token t from position plen-1+t; logits index plen-1 .. T-2
    logprobs = torch.log_softmax(logits[0, plen - 1 : -1, :].float(), dim=-1)
    tok_lp = logprobs.gather(-1, gen_ids.unsqueeze(-1).to(logprobs.device)).squeeze(-1)
    return float(tok_lp.mean().item())


# ── G2: teacher-forced answer-side capture ───────────────────────────────────


def capture_v0_for_context(
    model,
    tokenizer,
    instance: dict,
    probes: list[str],
    completions: list[str],
    capture: AnswerSpanCapture,
    n_layers: int,
    capture_layers: list[int],
) -> tuple[list, dict]:
    """Teacher-force (prompt + answer) per probe; capture the answer span.

    Returns (per_probe_spans, summaries) where
      per_probe_spans[p] = fp16 tensor (Lc, S_p, H) at the capture_layers
      summaries = {recipe: (Lc, H) probe-mean of the per-probe recipe summary}
    for recipe in {mean, last, maxp} (attn fit on CPU).

    Position assert: the teacher-forced answer span length must equal the
    generated answer's token count (fail loud on misalignment — plan §4.2).
    """
    import torch as _t

    lc = len(capture_layers)
    per_probe_spans: list = []
    # accumulate probe-mean summaries per recipe
    recipe_accum: dict[str, _t.Tensor] = {
        r: _t.zeros(lc, model.config.hidden_size, dtype=_t.float32)
        for r in ("mean", "last", "maxp")
    }
    n_used = 0
    for q, ans in zip(probes, completions, strict=True):
        messages = messages_for_instance(instance, q)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"]
        ans_ids = tokenizer(ans, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if ans_ids.shape[1] == 0:
            # An empty completion has no answer span — skip the probe, log it.
            logger.warning(
                "empty completion for instance=%s probe=%r — skipping v0 span",
                instance["id"],
                q[:40],
            )
            per_probe_spans.append(None)
            continue
        full_ids = _t.cat([prompt_ids, ans_ids], dim=1).to(model.device)
        prompt_len = int(prompt_ids.shape[1])
        ans_len = int(ans_ids.shape[1])
        with _t.no_grad():
            _ = model(input_ids=full_ids)
        # The answer span is positions [prompt_len, prompt_len + ans_len).
        span_full = capture.answer_span_stack(n_layers, prompt_len, prompt_len + ans_len)  # (L,S,H)
        captured_s = span_full.shape[1]
        assert captured_s == ans_len, (
            f"answer-span length mismatch instance={instance['id']} probe={q[:30]!r}: "
            f"captured {captured_s} positions != {ans_len} generated answer tokens"
        )
        span = span_full[capture_layers]  # (Lc, S, H) fp16
        per_probe_spans.append(span)
        # per-recipe summary per layer
        for r in ("mean", "last", "maxp"):
            for li in range(lc):
                recipe_accum[r][li] += summarize_answer_span(span[li], r).float()
        n_used += 1
    assert n_used > 0, f"instance {instance['id']}: every probe produced an empty answer"
    summaries = {r: (recipe_accum[r] / n_used) for r in ("mean", "last", "maxp")}
    return per_probe_spans, summaries


# ── HF model load ────────────────────────────────────────────────────────────


def load_hf_model(model_name: str, use_cuda: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


# ── G4: r_B diff-in-means (D_B, D_{B̄}) construction ──────────────────────────
# The round-1 r_B-construction concern: there is NO ready-made paired registry
# in corpora.py. We construct (D_B, D_{B̄}) prompt sets from the eval batteries
# per column. r_B = mean(answer-acts | D_B) - mean(answer-acts | D_{B̄}),
# persona-vectors diff-in-means (theory A3.3 default). Columns without a
# natural contrast (rb_contrast=None) are DROPPED (handled by rb_columns()).


def build_rb_contrast(column_id: str, probes: list[str], cap: int) -> tuple[list[str], list[str]]:
    """Build (D_B, D_{B̄}) prompt-text sets for one behavior column.

    Returns (d_b_prompts, d_bbar_prompts). Each is a list of USER-turn texts;
    the caller forwards them under the DEFAULT (no-persona) chat template and
    diffs the mean answer-side activations. The contrasts are derived from the
    same Betley probe pool + the testbed batteries (NEW work, plan §4.2 G4).

    The contrast TYPE per column is fixed by E0Column.rb_contrast:
      betley_vs_neutral      D_B = Betley main-8 (EM-eliciting), D_Bbar = neutral probes
      harmful_vs_benign      D_B = harmful-request battery, D_Bbar = benign probes
      syco_claim_vs_neutral  D_B = wrong-claim agreement prompts, D_Bbar = neutral probes
      should_refuse_vs_should_not  D_B = SORRY-Bench (should refuse), D_Bbar = XSTest safe

    A column whose contrast battery cannot be loaded raises (fail loud) — never
    a silent skip; rb_columns() already excludes the no-contrast columns.
    """
    contrast = E0_COLUMNS[column_id].rb_contrast
    assert contrast is not None, (
        f"{column_id} has no r_B contrast (should be excluded by rb_columns)"
    )

    neutral = probes[:cap]
    if contrast == "betley_vs_neutral":
        # D_B = Betley main-8 (EM-eliciting); D_Bbar = neutral preregistered probes.
        main8 = list(fetch_betley_main_8())
        d_b = (main8 * ((cap // len(main8)) + 1))[:cap]
        return d_b, neutral
    if contrast == "syco_claim_vs_neutral":
        # D_B = #411 RAW wrong-claim user statements; D_Bbar = neutral probes.
        # Pull the raw claims directly (no Sonnet rendering — we only need the
        # user-turn prompt asserting a false claim, not an agreement reply).
        from huggingface_hub import hf_hub_download

        src = hf_hub_download(
            "superkaiba1/explore-persona-space-data",
            "issue411_sycophancy_cosine_gradient/data/wrong_claims/train_200.jsonl",
            repo_type="dataset",
        )
        rows = []
        for line in Path(src).read_text().splitlines():
            if line.strip():
                r = _load_jsonl_row(line)
                rows.append(f"{r['wrong_claim']} Right?")
        return rows[:cap], neutral
    if contrast == "harmful_vs_benign":
        # D_B = AdvBench harmful requests (tier-2 panel, loaded directly);
        # D_Bbar = neutral probes.
        return _load_harmful_prompts(cap), neutral
    if contrast == "should_refuse_vs_should_not":
        # D_B = should-refuse (SORRY-Bench), D_Bbar = should-NOT-refuse (XSTest
        # safe). build_refusal_panel is a frozen JSON builder (no per-row API).
        from explore_persona_space.experiments.behavior_testbed_545 import corpora

        path = corpora.build_refusal_panel()
        payload = _load_json(Path(path))
        should_refuse = payload.get("should_refuse", [])[:cap]
        should_not = payload.get("should_not_refuse", [])[:cap]
        if not should_refuse or not should_not:
            raise RuntimeError("refusal panel missing should_refuse / should_not_refuse")
        return should_refuse, should_not
    raise ValueError(f"unknown r_B contrast {contrast!r} for column {column_id}")


def _load_jsonl_row(line: str) -> dict:
    import json

    return json.loads(line)


def _load_harmful_prompts(cap: int) -> list[str]:
    """Tier-2 harmful-request prompts for the harmful_compliance r_B contrast.

    Source ladder (all tier-2, no per-row API):
    1. the project's frozen #503 ``advbench_harmful_520`` panel (the same source
       the #545 ``harmful_compliance`` battery uses), if the artifact is on disk;
    2. the gated ``walledai/AdvBench`` HF dataset (needs access);
    3. the refusal panel's SORRY-Bench ``should_refuse`` half (harmful requests
       by construction), built locally with no gated dependency.
    All three failing raises (fail loud).
    """
    try:
        from explore_persona_space.experiments.issue503.eval_panels import load_panel
        from explore_persona_space.task_workflow import repo_root

        panel = load_panel("advbench_harmful_520", repo_root())
        return panel[: cap or len(panel)]
    except Exception as e:
        logger.warning(
            "#503 advbench panel unavailable (%s); trying gated dataset / SORRY-Bench", e
        )
    try:
        from datasets import load_dataset

        ds = load_dataset("walledai/AdvBench", split="train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        return [r[col] for r in ds][: cap or 200]
    except Exception as e:
        logger.warning(
            "gated walledai/AdvBench unavailable (%s); using SORRY-Bench should_refuse", e
        )
    from explore_persona_space.experiments.behavior_testbed_545 import corpora

    payload = _load_json(Path(corpora.build_refusal_panel()))
    harmful = payload.get("should_refuse", [])
    if not harmful:
        raise RuntimeError("no harmful-request panel available for the r_B contrast")
    return harmful[: cap or len(harmful)]


def _load_json(path: Path):
    import json

    with open(path) as f:
        return json.load(f)


def capture_mean_answer_acts(
    model,
    tokenizer,
    texts: list[str],
    capture: AnswerSpanCapture,
    n_layers: int,
    capture_layers: list[int],
) -> torch.Tensor:
    """Mean answer-side activation over a set of prompts (greedy 1-step generate).

    For r_B we need the activation under the model EXPRESSING / not-expressing
    the behavior — generate a short on-policy answer per prompt, teacher-force,
    mean-pool the answer span. Returns (Lc, H) fp32 = mean over prompts of the
    per-prompt mean-answer summary.
    """
    import torch as _t

    lc = len(capture_layers)
    accum = _t.zeros(lc, model.config.hidden_size, dtype=_t.float32)
    n = 0
    for text in texts:
        tmpl = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(tmpl, return_tensors="pt", padding=False).to(model.device)
        with _t.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=64, do_sample=False, temperature=None, top_p=None
            )
        plen = int(inputs["input_ids"].shape[1])
        gen = out[:, plen:]
        if gen.shape[1] == 0:
            continue
        full = _t.cat([inputs["input_ids"], gen], dim=1)
        with _t.no_grad():
            _ = model(input_ids=full)
        span = capture.answer_span_stack(n_layers, plen, plen + gen.shape[1])[capture_layers]
        for li in range(lc):
            accum[li] += span[li].float().mean(dim=0)
        n += 1
    assert n > 0, "r_B contrast produced no non-empty answers"
    return accum / n


# ── manifest / sentinel / upload ─────────────────────────────────────────────


def write_sentinel(kind: str, note: str, task_id: int = 658) -> Path:
    """poll_pipeline.py-conformant end-of-run sentinel (_SENTINEL_REQUIRED_KEYS)."""
    import json

    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    kind_slug = kind.replace(":", "_")
    path = logs_dir / f"issue-{task_id}-{kind_slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,
        "note": note,
        "task_id": task_id,
        "by": "issue658_extract_base_store",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote sentinel %s", path)
    return path


def _is_storage_quota_403(err: Exception) -> bool:
    msg = str(err)
    return "403" in msg and "storage" in msg.lower()


def upload_store(out_dir: Path, smoke: bool, hf_subdir: str | None = None) -> dict:
    """ONE bulk upload_folder commit of the store; verify via list_repo_files."""
    from huggingface_hub import HfApi

    api = HfApi()
    sub = hf_subdir or ("smoke_probe" if smoke else "store")
    path_in_repo = f"{HF_PREFIX}/{sub}"
    repo_used = HF_DATA_REPO
    try:
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue658: {'smoke ' if smoke else ''}base store upload",
        )
    except Exception as e:
        if not _is_storage_quota_403(e):
            raise
        logger.warning("HF storage-quota 403 on %s; falling back to overflow repo", HF_DATA_REPO)
        repo_used = HF_OVERFLOW_REPO
        api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
            repo_id=HF_OVERFLOW_REPO,
            repo_type="dataset",
            commit_message="issue658: base store upload (quota-403 overflow fallback)",
        )
    files = [
        f for f in api.list_repo_files(repo_used, repo_type="dataset") if f.startswith(path_in_repo)
    ]
    expected = {f"{path_in_repo}/store_manifest.json", f"{path_in_repo}/v0_summaries.pt"}
    missing = expected - set(files)
    if missing:
        raise RuntimeError(f"upload verification failed; missing on {repo_used}: {missing}")
    logger.info("Upload verified on %s: %d files under %s", repo_used, len(files), path_in_repo)
    return {"repo": repo_used, "path_in_repo": path_in_repo, "n_files": len(files)}


def upload_raw_completions(dirs: list[Path], smoke: bool) -> dict:
    """Batch every per-cell completion JSON into ONE create_commit (HF data repo).

    The dispatcher writes flat per-cell JSONs (``<ctx>__<col>.json`` E0 gen +
    ``<ctx>.json`` v0-capture answers) under ``eval_results/issue_658/``. Per
    the Upload Policy raw completions MUST land on the HF data repo before pod
    termination; the flat shape does not match the canonical
    ``raw_completions.json`` recursive glob, so we batch them into ONE
    ``create_commit`` (PREFERRED over a per-file loop — HF throttles a repo at
    ~256 commits/hr, #591) targeting the canonical
    ``issue658_theory_assumptions/raw_completions/<rel>`` path, then verify the
    per-prefix file count on the Hub before returning. Fail-loud.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    sub = "raw_completions_smoke" if smoke else "raw_completions"
    base = f"{HF_PREFIX}/{sub}"
    ops = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            ops.append(
                CommitOperationAdd(path_in_repo=f"{base}/{d.name}/{f.name}", path_or_fileobj=str(f))
            )
    if not ops:
        logger.warning("no raw-completion JSONs to upload under %s", [str(d) for d in dirs])
        return {"skipped": True, "reason": "no files"}
    api = HfApi()
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"issue658: raw completions ({len(ops)} files)",
    )
    remote = {
        f
        for f in api.list_repo_files(HF_DATA_REPO, repo_type="dataset")
        if f.startswith(base + "/")
    }
    n_remote = len(remote)
    if n_remote < len(ops):
        raise RuntimeError(
            f"raw-completions upload verification failed: {n_remote} files on Hub under {base}/, "
            f"committed {len(ops)}"
        )
    logger.info("Raw completions verified: %d files under %s", n_remote, base)
    return {"repo": HF_DATA_REPO, "path_in_repo": base, "n_files": n_remote}


# ── 8-GPU data-parallel sharding (Change 2) ──────────────────────────────────
# The work is embarrassingly parallel over the 50 contexts. We DATA-parallel
# (one 7B replica per GPU, NOT tensor-parallel — 7B fits on one H100) by sharding
# the CONTEXT list round-robin across N workers. Each worker runs the per-context
# phases (G1 generate / G2 capture / G3 c_C mean-prompt / G6 E0) on its context
# subset and writes a self-contained shard under ``<out_dir>/shards/shard_<k>/``;
# the context-independent phases (G4 r_B over D_B/D_Bbar, G5 Σ_c background
# corpus) run ONCE in the ``--rbsigma`` worker; ``--merge`` assembles every shard
# + the rbsigma outputs into the unified store, builds the sha-pinned manifest,
# and uploads. The launcher (scripts/issue658_8gpu_dispatch.sh) pins
# CUDA_VISIBLE_DEVICES=<gpu> per worker (the import-time-cuInit clobber gotcha)
# AND passes the matching --gpu-id, fanning the workers across all 8 GPUs.
#
# The smoke runs the IDENTICAL shell dispatcher with --n-shards 2 --n-ctx 4 on
# CPU, so the smoke exercises the SAME shard → rbsigma → merge subprocess shape
# the production run uses (PASS_UNIFIED: smoke IS the sweep with fewer cells).

# A shard's self-contained on-disk layout (every per-context tensor the merge
# needs + the shard's own raw / e0 completions for the consolidated upload).
SHARD_DELIVERABLE_FILES: tuple[str, ...] = ("v0_summaries.pt", "answer_spans/index.json")


def run_context_phases(
    model,
    tokenizer,
    instances: list[dict],
    probes: list[str],
    *,
    args,
    use_cuda: bool,
    capture_layers: list[int],
    n_layers: int,
    out_dir: Path,
    run,
) -> dict:
    """G1 generate + G2 capture + G3 c_C-mean-prompt + G6 E0 over ``instances``.

    The per-context (data-parallel) phases. Writes, under ``out_dir``:
      - ``answer_spans/<ctx>.pt``      per-(C,probe) answer spans (N1 + attn fit)
      - ``answer_spans/index.json``    the per-(C) → probes map (shard-local)
      - ``v0_summaries.pt``            {recipe: {ctx: (Lc,H)}} + cc_meanprompt
      - raw completions + E0 gen JSONs under ``EVAL_RESULTS_DIR`` (per-cell)

    Identical phase semantics whether this is the whole battery (all-in-one /
    smoke single-process) or one shard's context subset (8-GPU worker). Returns
    a small telemetry dict for the caller's sentinel/manifest.
    """
    spans_dir = out_dir / "answer_spans"
    spans_dir.mkdir(parents=True, exist_ok=True)

    # ── G1: generate base answers ─────────────────────────────────────────────
    phase("g1_generate")
    v0_cap = min(V0_MAX_NEW_TOKENS, args.max_new_tokens_smoke) if args.smoke else V0_MAX_NEW_TOKENS
    prompts, index = build_prompts(tokenizer, instances, probes)
    if args.no_vllm or not use_cuda:
        completions = hf_generate(model, tokenizer, prompts, v0_cap)
    else:
        completions = vllm_generate(args.model, prompts, v0_cap)
    assert len(completions) == len(prompts) == len(index)
    raw_dir = EVAL_RESULTS_DIR / ("raw_completions_smoke" if args.smoke else "raw_completions")
    raw_dir.mkdir(parents=True, exist_ok=True)
    by_ctx: dict[str, list[dict]] = {}
    for (iid, q), ans in zip(index, completions, strict=True):
        by_ctx.setdefault(iid, []).append({"probe": q, "completion": ans})
    for iid, rows in by_ctx.items():
        dump_json({"context_id": iid, "completions": rows}, raw_dir / f"{iid}.json")
    logger.info("G1 done: %d completions over %d contexts", len(completions), len(by_ctx))

    # ── G2: teacher-forced answer-side capture → v0(C) + G3 c_C mean-prompt ────
    phase("g2_capture")
    capture = AnswerSpanCapture(model, n_layers)
    v0_summaries: dict[str, dict[str, list]] = {r: {} for r in ("mean", "last", "maxp")}
    cc_meanprompt: dict[str, list] = {}
    span_index: dict[str, list[str]] = {}
    try:
        for inst in instances:
            iid = inst["id"]
            ctx_probes = [r["probe"] for r in by_ctx[iid]]
            ctx_completions = [r["completion"] for r in by_ctx[iid]]
            spans, summ = capture_v0_for_context(
                model,
                tokenizer,
                inst,
                ctx_probes,
                ctx_completions,
                capture,
                n_layers,
                capture_layers,
            )
            torch.save(
                {
                    "context_id": iid,
                    "capture_layers": capture_layers,
                    "spans": spans,
                    "probes": ctx_probes,
                },
                spans_dir / f"{iid}.pt",
            )
            span_index[iid] = ctx_probes
            for r in ("mean", "last", "maxp"):
                v0_summaries[r][iid] = summ[r]
            tmpl = tokenizer.apply_chat_template(
                messages_for_instance(inst, ctx_probes[0]),
                tokenize=False,
                add_generation_prompt=True,
            )
            pinputs = tokenizer(tmpl, return_tensors="pt", padding=False).to(model.device)
            with torch.no_grad():
                _ = model(**pinputs)
            cc_meanprompt[iid] = capture.mean_prompt_stack(
                n_layers, int(pinputs["input_ids"].shape[1])
            )[capture_layers]
            run.log({"v0_contexts_done": len(v0_summaries["mean"])})
    finally:
        capture.remove()
    logger.info("G2 done: v0 summaries for %d contexts", len(v0_summaries["mean"]))

    dump_json(
        {"context_ids": list(span_index.keys()), "probes_by_context": span_index},
        spans_dir / "index.json",
    )
    torch.save(
        {
            "summaries": v0_summaries,
            "cc_meanprompt": cc_meanprompt,
            "capture_layers": capture_layers,
            "context_ids": [i["id"] for i in instances],
            "model": args.model,
            "probe_pool_hash": stable_hash(probes),
        },
        out_dir / "v0_summaries.pt",
    )

    # ── G6: E0(C,B) behavior-battery generations ──────────────────────────────
    phase("g6_e0gen")
    e0_dir = EVAL_RESULTS_DIR / ("e0_gen_smoke" if args.smoke else "e0_gen")
    e0_dir.mkdir(parents=True, exist_ok=True)
    e0_n_battery = 4 if args.smoke else (args.e0_n_battery or 0)
    generate_e0_completions(
        model,
        tokenizer,
        instances,
        e0_dir,
        use_vllm=(use_cuda and not args.no_vllm),
        model_name=args.model,
        n_battery=e0_n_battery,
        run=run,
        max_new_tokens_cap=(args.max_new_tokens_smoke if args.smoke else 0),
        n_samples_cap=(args.n_samples_smoke if args.smoke else 0),
        smoke=args.smoke,
        upload_partial=not args.no_upload,
    )
    return {"n_ctx": len(instances), "raw_dir": str(raw_dir), "e0_dir": str(e0_dir)}


def run_rbsigma_phases(
    model,
    tokenizer,
    probes: list[str],
    *,
    args,
    capture_layers: list[int],
    n_layers: int,
    out_dir: Path,
) -> dict:
    """G4 r_B diff-in-means + G5 Σ_c — the CONTEXT-INDEPENDENT phases.

    These do NOT depend on the battery contexts (r_B is per-behavior over
    D_B/D_Bbar prompt sets; Σ_c is the background corpus), so they run ONCE
    (the ``--rbsigma`` worker / the all-in-one tail), never per shard. Writes
    ``r_b.pt`` and (unless ``--skip-sigma``) ``sigma_c.pt`` under ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rb_cap = 4 if args.smoke else args.rb_cap
    sigma_n = 200 if args.smoke else args.sigma_n

    # ── G4: r_B diff-in-means over (D_B, D_Bbar) ──────────────────────────────
    phase("g4_rb")
    capture = AnswerSpanCapture(model, n_layers)
    r_b: dict[str, dict] = {}
    try:
        for col in rb_columns():
            d_b, d_bbar = build_rb_contrast(col, probes, rb_cap)
            mean_b = capture_mean_answer_acts(
                model, tokenizer, d_b, capture, n_layers, capture_layers
            )
            mean_bbar = capture_mean_answer_acts(
                model, tokenizer, d_bbar, capture, n_layers, capture_layers
            )
            r_b[col] = {
                "diffmeans": (mean_b - mean_bbar),
                "meanDB": mean_b,
                "n_db": len(d_b),
                "n_dbbar": len(d_bbar),
            }
            logger.info("r_B[%s]: |D_B|=%d |D_Bbar|=%d", col, len(d_b), len(d_bbar))
    finally:
        capture.remove()
    torch.save(
        {"r_b": r_b, "capture_layers": capture_layers, "columns": rb_columns()},
        out_dir / "r_b.pt",
    )

    # ── G5: Σ_c background corpus (second moment) ──────────────────────────────
    sigma_info: dict = {"skipped": True}
    if not args.skip_sigma:
        phase("g5_sigma")
        sigma_info = extract_sigma_c(model, tokenizer, sigma_n, capture_layers, n_layers, out_dir)
    return {"rb_columns": len(r_b), "sigma": sigma_info}


def merge_shards(
    shard_dirs: list[Path],
    rbsigma_dir: Path,
    out_dir: Path,
) -> dict:
    """Merge per-shard v0/answer-span outputs + the rbsigma outputs into one store.

    Each shard wrote a self-contained ``v0_summaries.pt`` + ``answer_spans/`` over
    its context subset; the rbsigma worker wrote ``r_b.pt`` (+ ``sigma_c.pt``).
    This concatenates the per-context dicts (the shard partition is disjoint by
    construction — ``partition_contexts``), copies every ``answer_spans/<ctx>.pt``,
    re-writes the unified ``answer_spans/index.json``, and moves ``r_b.pt`` /
    ``sigma_c.pt`` into ``out_dir``. Fails loud on a duplicate context id across
    shards (a partition violation) or a missing rbsigma deliverable.
    """
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    spans_out = out_dir / "answer_spans"
    spans_out.mkdir(parents=True, exist_ok=True)

    merged_summaries: dict[str, dict[str, object]] = {r: {} for r in ("mean", "last", "maxp")}
    merged_cc: dict[str, object] = {}
    merged_capture_layers: list[int] | None = None
    merged_ctx_ids: list[str] = []
    merged_model: str | None = None
    merged_probe_hash: str | None = None
    span_index: dict[str, list[str]] = {}

    for sd in shard_dirs:
        vpath = sd / "v0_summaries.pt"
        if not vpath.is_file():
            raise RuntimeError(f"shard {sd} missing v0_summaries.pt — incomplete shard")
        blob = torch.load(vpath, weights_only=False)
        if merged_capture_layers is None:
            merged_capture_layers = blob["capture_layers"]
            merged_model = blob["model"]
            merged_probe_hash = blob["probe_pool_hash"]
        else:
            assert blob["capture_layers"] == merged_capture_layers, f"capture_layers drift in {sd}"
            assert blob["model"] == merged_model, f"model drift in {sd}"
            assert blob["probe_pool_hash"] == merged_probe_hash, f"probe_pool_hash drift in {sd}"
        for r in ("mean", "last", "maxp"):
            for cid, vec in blob["summaries"][r].items():
                if cid in merged_summaries[r]:
                    raise RuntimeError(
                        f"duplicate context {cid} across shards (partition violated)"
                    )
                merged_summaries[r][cid] = vec
        for cid, vec in blob["cc_meanprompt"].items():
            merged_cc[cid] = vec
        merged_ctx_ids.extend(blob["context_ids"])
        # copy the per-context answer-span .pt files + accumulate the index
        sd_index = _load_json(sd / "answer_spans" / "index.json")
        for cid, ctx_probes in sd_index["probes_by_context"].items():
            span_index[cid] = ctx_probes
            src = sd / "answer_spans" / f"{cid}.pt"
            if src.is_file():
                shutil.copy2(src, spans_out / f"{cid}.pt")

    if len(merged_ctx_ids) != len(set(merged_ctx_ids)):
        raise RuntimeError("duplicate context ids after merge — shard partition was not disjoint")

    torch.save(
        {
            "summaries": merged_summaries,
            "cc_meanprompt": merged_cc,
            "capture_layers": merged_capture_layers,
            "context_ids": merged_ctx_ids,
            "model": merged_model,
            "probe_pool_hash": merged_probe_hash,
        },
        out_dir / "v0_summaries.pt",
    )
    dump_json(
        {"context_ids": list(span_index.keys()), "probes_by_context": span_index},
        spans_out / "index.json",
    )

    # r_b.pt + sigma_c.pt from the rbsigma worker.
    rb_src = rbsigma_dir / "r_b.pt"
    if not rb_src.is_file():
        raise RuntimeError(f"rbsigma worker did not produce r_b.pt under {rbsigma_dir}")
    shutil.copy2(rb_src, out_dir / "r_b.pt")
    sig_src = rbsigma_dir / "sigma_c.pt"
    if sig_src.is_file():
        shutil.copy2(sig_src, out_dir / "sigma_c.pt")

    return {
        "n_ctx": len(merged_ctx_ids),
        "n_shards": len(shard_dirs),
        "context_ids": merged_ctx_ids,
        "capture_layers": merged_capture_layers,
        "model": merged_model,
        "probe_pool_hash": merged_probe_hash,
        "sigma_present": (out_dir / "sigma_c.pt").is_file(),
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #658: base-model activation store.")
    parser.add_argument("--battery", type=Path, default=PROJECT_ROOT / "data/issue594/battery.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data/issue_658/store")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--expected-layers", type=int, default=EXPECTED_LAYERS)
    parser.add_argument("--expected-hidden", type=int, default=EXPECTED_HIDDEN)
    parser.add_argument(
        "--n-ctx", type=int, default=0, help="cap the context count (0 = all; smoke default 4)"
    )
    parser.add_argument(
        "--n-probes", type=int, default=0, help="cap the probe pool (0 = full; smoke default 4)"
    )
    parser.add_argument(
        "--n-layers-smoke",
        type=int,
        default=0,
        help="capture only the FIRST N layers (0 = all expected-layers; smoke uses a few)",
    )
    parser.add_argument(
        "--sigma-n", type=int, default=3000, help="Σ_c background corpus context count (>= 3000)"
    )
    parser.add_argument("--rb-cap", type=int, default=48, help="prompts per (D_B, D_Bbar) side")
    parser.add_argument(
        "--e0-n-battery",
        type=int,
        default=0,
        help="cap the per-column E0 battery probe count (0 = full battery; smoke uses 4)",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="use HF greedy generate instead of vLLM (CPU smoke / no-GPU)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="IDENTICAL dispatcher, tiny single-cell slice (4 ctx x 4 probes, 4 layers, ~200 Σ_c) "
        "→ PASS_UNIFIED; smoke IS the sweep with --n-ctx 4",
    )
    parser.add_argument("--no-upload", action="store_true", help="skip HF upload (local smoke)")
    parser.add_argument("--hf-subdir", default=None, help="verbatim HF upload sub-dir override")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument(
        "--skip-sigma", action="store_true", help="skip the Σ_c phase (debug / partial re-run)"
    )
    parser.add_argument(
        "--max-new-tokens-smoke",
        type=int,
        default=24,
        help="smoke-only: clamp generation LENGTH to this many tokens (the per-column "
        "temperature is untouched) so a CPU smoke does not generate 2048 marker tokens; "
        "0 = honor each column's full cap (the real-run default)",
    )
    parser.add_argument(
        "--n-samples-smoke",
        type=int,
        default=2,
        help="smoke-only: clamp the per-probe SAMPLE count (the per-column temperature is "
        "untouched) so broad_em's n=50 is not 50 CPU generations; 0 = honor each column's "
        "full n_samples (the real-run default)",
    )
    # ── 8-GPU data-parallel modes (Change 2) ──────────────────────────────────
    # Exactly one of {context-shard, --rbsigma, --merge} runs per process; with
    # none of them set the legacy single-process all-in-one path runs (used by
    # direct invocation). The 8-GPU launcher fans out N context-shard workers +
    # one --rbsigma worker across the GPUs, then runs one --merge.
    parser.add_argument(
        "--n-shards",
        type=int,
        default=1,
        help="total data-parallel context shards (8-GPU run: 8). 1 = all-in-one.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=-1,
        help="this worker's context shard (0-based); run ONLY the per-context phases "
        "(G1/G2/G3/G6) over instances[shard_id::n_shards]. -1 = not a shard worker.",
    )
    parser.add_argument(
        "--rbsigma",
        action="store_true",
        help="run ONLY the context-independent phases (G4 r_B + G5 Σ_c); one worker.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge every shard + the rbsigma outputs into the unified store, build the "
        "sha-pinned manifest, and upload. Run after all shard / rbsigma workers finish.",
    )
    parser.add_argument(
        "--shards-root",
        type=Path,
        default=None,
        help="root holding per-shard dirs (shard_<k>/) + rbsigma/; defaults to "
        "<out_dir>/shards. Threaded so smoke + prod share one layout.",
    )
    args = parser.parse_args()

    # Mode validation: at most one of {context-shard, --rbsigma, --merge}.
    is_shard = args.shard_id >= 0
    n_modes = int(is_shard) + int(args.rbsigma) + int(args.merge)
    if n_modes > 1:
        parser.error("--shard-id, --rbsigma, and --merge are mutually exclusive")

    # ── merge mode: no model load, no GPU; CPU assemble + upload ───────────────
    if args.merge:
        return run_merge_mode(args)

    phase("load")
    # Bind CVD before the first CUDA allocation (the +gpu_id clobber gotcha).
    # The launcher ALSO exports CUDA_VISIBLE_DEVICES=<gpu> in the worker env
    # (the in-process clobber alone is defeated by any import-time cuInit —
    # gotchas.md), so this in-process set rewrites the same value.
    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    # Resolve the per-worker output dir. Shard / rbsigma workers write to a
    # self-contained subdir under shards_root; the merge consumes them.
    base_out = Path(f"{args.out_dir}_smoke") if args.smoke else args.out_dir
    shards_root = args.shards_root or (base_out / "shards")
    if is_shard:
        worker_out = shards_root / f"shard_{args.shard_id}"
    elif args.rbsigma:
        worker_out = shards_root / "rbsigma"
    else:
        worker_out = base_out  # all-in-one (legacy / smoke single-process)
    worker_out.mkdir(parents=True, exist_ok=True)

    _payload, instances = load_battery(args.battery)
    main8 = set(fetch_betley_main_8())
    probes = fetch_preregistered_probes(n=200, exclude=main8)

    n_ctx_cap = args.n_ctx or (4 if args.smoke else len(instances))
    n_probes_cap = args.n_probes or (4 if args.smoke else len(probes))
    instances = instances[:n_ctx_cap]
    probes = probes[:n_probes_cap]
    sigma_n = 200 if args.smoke else args.sigma_n

    # Shard the context list (data-parallel). r_B / Σ_c are context-independent,
    # so the rbsigma + merge workers keep the FULL probe pool but never iterate
    # contexts. The all-in-one path keeps every context (n_shards default 1).
    shard_instances = (
        partition_contexts(instances, args.shard_id, args.n_shards) if is_shard else instances
    )

    logger.info(
        "Mode=%s: %d/%d ctx x %d probes (smoke=%s, vllm=%s, out=%s)",
        ("shard" if is_shard else "rbsigma" if args.rbsigma else "all-in-one"),
        len(shard_instances),
        len(instances),
        len(probes),
        args.smoke,
        not args.no_vllm,
        worker_out,
    )

    import wandb

    run_suffix = f"-shard{args.shard_id}" if is_shard else "-rbsigma" if args.rbsigma else ""
    run = wandb.init(
        project="explore-persona-space",
        name=("issue658-extract-smoke" if args.smoke else "issue658-extract") + run_suffix,
        mode=args.wandb_mode,
        config={
            "model": args.model,
            "n_ctx": len(shard_instances),
            "n_probes": len(probes),
            "sigma_n": sigma_n,
            "smoke": args.smoke,
            "shard_id": args.shard_id,
            "n_shards": args.n_shards,
            "mode": ("shard" if is_shard else "rbsigma" if args.rbsigma else "all-in-one"),
        },
    )

    model, tokenizer = load_hf_model(args.model, use_cuda)
    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size
    assert n_layers == args.expected_layers, f"{n_layers} layers != expected {args.expected_layers}"
    assert hidden == args.expected_hidden, f"hidden {hidden} != expected {args.expected_hidden}"

    # Marker token assert IN-PROCESS (CLAUDE.md / marker-leakage-measurement.md):
    # the store carries the marker context + r_B reads — a wrong marker silently
    # no-ops the localized-behavior probe A3.2 requires. (Skipped only for a
    # non-Qwen CPU smoke stub, where the id differs by construction.)
    marker_ids = tokenizer.encode(" ※", add_special_tokens=False)
    if args.model == DEFAULT_MODEL:
        assert marker_ids == [MARKER_TOKEN_ID], (
            f"marker token id drift: encode(' ※') = {marker_ids}, expected [{MARKER_TOKEN_ID}]"
        )
    else:
        logger.warning(
            "non-default model %s: marker encodes to %s (skipping the 83399 assert — smoke stub)",
            args.model,
            marker_ids,
        )

    capture_layers = (
        list(range(min(args.n_layers_smoke or n_layers, n_layers)))
        if (args.smoke or args.n_layers_smoke)
        else list(range(n_layers))
    )
    logger.info("Capturing %d layers: %s", len(capture_layers), capture_layers)

    # ── rbsigma worker: G4 + G5 only (context-independent), then done ──────────
    if args.rbsigma:
        rs = run_rbsigma_phases(
            model,
            tokenizer,
            probes,
            args=args,
            capture_layers=capture_layers,
            n_layers=n_layers,
            out_dir=worker_out,
        )
        write_sentinel(
            "epm:smoke-result" if args.smoke else "epm:results",
            f"issue658 rbsigma complete: r_B columns={rs['rb_columns']}, sigma={rs['sigma']}",
        )
        run.finish()
        phase("done")
        return 0

    # ── shard worker: per-context phases over the shard's context subset ───────
    if is_shard:
        cp = run_context_phases(
            model,
            tokenizer,
            shard_instances,
            probes,
            args=args,
            use_cuda=use_cuda,
            capture_layers=capture_layers,
            n_layers=n_layers,
            out_dir=worker_out,
            run=run,
        )
        write_sentinel(
            "epm:smoke-result" if args.smoke else "epm:results",
            f"issue658 shard {args.shard_id}/{args.n_shards} complete: {cp}",
        )
        run.finish()
        phase("done")
        return 0

    # ── all-in-one path (legacy / single-process smoke) ───────────────────────
    # Runs the per-context phases over EVERY context, then the rbsigma phases,
    # then assembles + uploads in this one process (no shard merge needed).
    cp = run_context_phases(
        model,
        tokenizer,
        instances,
        probes,
        args=args,
        use_cuda=use_cuda,
        capture_layers=capture_layers,
        n_layers=n_layers,
        out_dir=worker_out,
        run=run,
    )
    rs = run_rbsigma_phases(
        model,
        tokenizer,
        probes,
        args=args,
        capture_layers=capture_layers,
        n_layers=n_layers,
        out_dir=worker_out,
    )
    sigma_info = rs["sigma"]
    r_b_n = rs["rb_columns"]
    e0_dir = Path(cp["e0_dir"])
    raw_dir = Path(cp["raw_dir"])

    # ── manifest + sentinel + upload ──────────────────────────────────────────
    info = _assemble_and_upload(
        worker_out,
        args=args,
        n_layers=n_layers,
        hidden=hidden,
        capture_layers=capture_layers,
        context_ids=[i["id"] for i in instances],
        probe_pool_hash=stable_hash(probes),
        marker_ids=marker_ids,
        sigma_info=sigma_info,
        e0_dir=e0_dir,
        raw_dir=raw_dir,
    )

    note = (
        f"issue658 base-store {'SMOKE ' if args.smoke else ''}complete: "
        f"{len(instances)} ctx x {len(probes)} probes, layers={len(capture_layers)}, "
        f"r_B columns={r_b_n}, sigma={sigma_info.get('n', 'skipped')}, upload={info['upload']}"
    )
    write_sentinel("epm:smoke-result" if args.smoke else "epm:results", note)
    run.finish()
    phase("done")
    return 0


def _assemble_and_upload(
    out_dir: Path,
    *,
    args,
    n_layers: int,
    hidden: int,
    capture_layers: list[int],
    context_ids: list[str],
    probe_pool_hash: str,
    marker_ids: list[int],
    sigma_info: dict,
    e0_dir: Path,
    raw_dir: Path,
) -> dict:
    """Build the sha-pinned store manifest + upload the store + raw completions.

    Shared by the all-in-one path AND the merge mode (both end with the same
    manifest+upload tail over a complete ``out_dir``). The manifest is written
    AFTER every tensor exists so the §6.5 SHAs are over the FINAL on-disk bytes;
    the HF URL is folded into each pinned-file entry on the post-upload rewrite.
    """
    phase("assemble")
    manifest = {
        "model": args.model,
        "n_layers": n_layers,
        "hidden": hidden,
        "capture_layers": capture_layers,
        "n_ctx": len(context_ids),
        "context_ids": context_ids,
        "probe_pool_hash": probe_pool_hash,
        "marker_token_id": MARKER_TOKEN_ID if args.model == DEFAULT_MODEL else marker_ids,
        "v0_summary_recipes": list(SUMMARY_RECIPES),
        "rb_columns": rb_columns(),
        "rb_recipes_scored": ["diffmeans", "meanDB"],
        "e0_columns": list(E0_COLUMNS.keys()),
        "sigma": sigma_info,
        "smoke": args.smoke,
        "n_shards": args.n_shards,
        "judge_model": "claude-sonnet-4-5-20250929",
        "cc_reuse_note": (
            "last-input-token c_C REUSED from #594 HF store "
            "(issue594_context_geometry/analysis_tensors); mean-over-prompt c_C is NEW here"
        ),
        "files": build_files_sha_map(out_dir),
        "metadata": reproducibility_metadata({"script": "issue658_extract_base_store"}),
    }
    dump_json(manifest, out_dir / "store_manifest.json")

    upload_info: dict = {"skipped": True}
    raw_upload_info: dict = {"skipped": True}
    if not args.no_upload:
        phase("upload")
        upload_info = upload_store(out_dir, smoke=args.smoke, hf_subdir=args.hf_subdir)
        manifest["upload"] = upload_info
        repo = upload_info.get("repo")
        prefix = upload_info.get("path_in_repo")
        if repo and prefix:
            for rel, entry in manifest["files"].items():
                if entry.get("present"):
                    entry["hf_url"] = (
                        f"https://huggingface.co/datasets/{repo}/resolve/main/{prefix}/{rel}"
                    )
        raw_upload_info = upload_raw_completions([e0_dir, raw_dir], smoke=args.smoke)
        manifest["raw_completions_upload"] = raw_upload_info
        dump_json(manifest, out_dir / "store_manifest.json")
    return {"upload": upload_info, "raw_upload": raw_upload_info}


def run_merge_mode(args) -> int:
    """``--merge``: assemble every shard + the rbsigma worker into the unified store.

    CPU-only (no model load, no GPU). Reads ``<shards_root>/shard_*/`` +
    ``<shards_root>/rbsigma/``, merges into ``out_dir``, builds the sha-pinned
    manifest, and uploads the store + raw completions. The raw / E0 completions
    were written by the shard workers to the SHARED ``EVAL_RESULTS_DIR`` (every
    worker on the same pod writes there, keyed by context id — disjoint by the
    shard partition), so the merge uploads them from that one directory.
    """
    phase("load")
    base_out = Path(f"{args.out_dir}_smoke") if args.smoke else args.out_dir
    shards_root = args.shards_root or (base_out / "shards")
    shard_dirs = sorted(
        (d for d in shards_root.glob("shard_*") if d.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not shard_dirs:
        raise RuntimeError(f"--merge found no shard_*/ dirs under {shards_root}")
    if len(shard_dirs) != args.n_shards:
        raise RuntimeError(
            f"--merge found {len(shard_dirs)} shards under {shards_root}, expected {args.n_shards} "
            f"(a shard worker may have crashed — check its sentinel / log before merging)"
        )
    rbsigma_dir = shards_root / "rbsigma"

    phase("merge")
    merged = merge_shards(shard_dirs, rbsigma_dir, base_out)
    logger.info(
        "Merged %d shards → %d contexts under %s", merged["n_shards"], merged["n_ctx"], base_out
    )

    # Marker id for the manifest (no model loaded in merge mode — record the
    # canonical id for the default model; smoke stub records the partition's id 0).
    marker_ids = [MARKER_TOKEN_ID] if args.model == DEFAULT_MODEL else [-1]
    sigma_blob_present = merged["sigma_present"]
    sigma_info = {"present": sigma_blob_present, "skipped": not sigma_blob_present}

    e0_dir = EVAL_RESULTS_DIR / ("e0_gen_smoke" if args.smoke else "e0_gen")
    raw_dir = EVAL_RESULTS_DIR / ("raw_completions_smoke" if args.smoke else "raw_completions")

    info = _assemble_and_upload(
        base_out,
        args=args,
        n_layers=args.expected_layers,
        hidden=args.expected_hidden,
        capture_layers=merged["capture_layers"],
        context_ids=merged["context_ids"],
        probe_pool_hash=merged["probe_pool_hash"],
        marker_ids=marker_ids,
        sigma_info=sigma_info,
        e0_dir=e0_dir,
        raw_dir=raw_dir,
    )
    write_sentinel(
        "epm:smoke-result" if args.smoke else "epm:results",
        f"issue658 MERGE complete: {merged['n_ctx']} ctx from {merged['n_shards']} shards, "
        f"sigma={sigma_blob_present}, upload={info['upload']}",
    )
    phase("done")
    return 0


def extract_sigma_c(
    model, tokenizer, sigma_n: int, capture_layers: list[int], n_layers: int, out_dir: Path
) -> dict:
    """G5: Σ_c = (1/n) Σ c c^T over a background corpus, prompt-only c_C reads.

    Reuses project_corpus_v2's corpus builder for the contexts; the c_C read is
    the same last-input-token slot as #594. Σ_c is the d×d second moment fed to
    downstream Phase 2-4 only (NOT load-bearing for this task's A3.2/A3.3).
    """
    import torch as _t

    contexts = load_sigma_corpus(sigma_n)
    if len(contexts) < sigma_n:
        logger.warning("Σ_c corpus produced %d < requested %d contexts", len(contexts), sigma_n)
    capture = AnswerSpanCapture(model, n_layers)
    lc = len(capture_layers)
    # Σ_c per captured layer; accumulate outer products of the last-input-token c_C.
    sigma = {li: _t.zeros(model.config.hidden_size, model.config.hidden_size) for li in range(lc)}
    n = 0
    try:
        for text in contexts:
            tmpl = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(tmpl, return_tensors="pt", padding=False).to(model.device)
            with _t.no_grad():
                _ = model(**inputs)
            stack = capture.last_token_stack(n_layers)[capture_layers]  # (Lc, H) fp32 CPU
            for li in range(lc):
                v = stack[li]
                sigma[li] += _t.outer(v, v)
            n += 1
    finally:
        capture.remove()
    if n == 0:
        raise RuntimeError("Σ_c corpus produced 0 usable contexts")
    sigma_tensor = _t.stack([sigma[li] / n for li in range(lc)])  # (Lc, H, H)
    _t.save(
        {"sigma_c": sigma_tensor, "n": n, "capture_layers": capture_layers},
        out_dir / "sigma_c.pt",
    )
    logger.info("Σ_c: %d contexts, tensor %s", n, tuple(sigma_tensor.shape))
    return {"n": n, "shape": list(sigma_tensor.shape)}


def load_sigma_corpus(n: int) -> list[str]:
    """Load ≥n background contexts for Σ_c via project_corpus_v2 (fallback: generic).

    The Σ_c corpus is a diverse background pool used only for second-moment
    estimation — NEVER a behavioral claim surface (plan §4.7 tier-3).
    """
    try:
        from explore_persona_space.experiments.behavior_testbed_545 import corpora

        return corpora.load_generic_questions(n, seed=658)
    except Exception as e:
        logger.warning("project corpus builder unavailable (%s) — using Betley probe pool", e)
        main8 = set(fetch_betley_main_8())
        pool = fetch_preregistered_probes(n=200, exclude=main8)
        # repeat to reach n (Σ_c only needs diversity for the second moment;
        # a small pool degrades the estimate, flagged in the manifest)
        return (pool * ((n // max(1, len(pool))) + 1))[:n]


if __name__ == "__main__":
    sys.exit(main())
