"""On-policy marker-emission + on-policy end-of-answer log p(※) dispatcher for #456.

This is the headline instrument for task #456. It REPLACES #432's
off-distribution fixed-stub/pos0 teacher-forced probe with a measurement on the
model's OWN generated text:

    (a) Emission rate (headline DV) -- for each of the 28 panel personas x 20
        fixed eval questions, vLLM-generate N samples and count the fraction
        of completions containing the marker ``※`` (substring; the single-token
        ``※`` -> id 63680 is the documented marker-leakage exception to the
        no-substring rule).

    (b) On-policy end-of-answer log p(※) (companion DV) -- teacher-forced log p
        of ``※`` at the token immediately AFTER the model's own generated
        answer (marker stripped from the conditioning text), rendered as
        ``chat_prefix + own_answer + "\\n\\n"``. This is #432's endpos geometry
        with the fixed stub replaced by on-policy context -- that replacement
        IS the experiment.

Two-phase per checkpoint:

    Phase A (vLLM): base loaded ONCE with ``enable_lora=True``; the per-
        checkpoint LoRA is swapped via ``LoRARequest(lora_int_id=step, ...)``
        (distinct int id per step so vLLM never serves a stale cached adapter).
        Generates all (persona, question) cells in one batched call.

    Phase B (HF teacher-forced): after ALL vLLM generation completes, the vLLM
        engine is torn down and its worker subprocesses are explicitly reaped
        (``_kill_vllm_workers`` -- per the CLAUDE.md gotcha that vLLM teardown
        does NOT reap workers, which then re-grab freed GPU memory the moment
        HF Transformers loads weights). Then the base model is loaded in HF and,
        per checkpoint, the PEFT adapter is layered on to score the on-policy
        endpos log p(※) over the SAME generated answers from Phase A.

Checkpoint-per-phase discipline (CLAUDE.md): one JSON per checkpoint is written
the MOMENT that checkpoint's Phase-A generation finishes, and a second file the
moment its Phase-B scoring finishes. A mid-run crash loses at most one
checkpoint's worth of one phase. Output under ``eval_results/issue_456/``.

Smoke unification: Phase 3 of the pipeline invokes this SAME dispatcher with
``--steps 10 --n-samples 2`` and a tiny ``--smoke-prompts``/``--smoke-personas``
subset -- identical code path to the full Phase-5 run, just parameterized.

Marker hard guard: ``assert tok.encode("※") == [63680]`` before any work --
this is the bare ``※`` #432 trained, NOT the global default ``" ※"`` (83399).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Make ``scripts/`` importable so we can pull the bystander panel.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Make ``src/`` importable for the reproducibility-metadata + logprob helpers.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

MARKER = "※"
MARKER_ID = 63680  # bare ※ under Qwen-2.5 BPE (NOT " ※" = 83399)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# BYSTANDERS / PROMPTS / SOURCE_PERSONA / PERSONAS are bound in ``main()`` from
# ``--panel-module`` (default ``_i416_bystander_panel`` = #432's panel).
BYSTANDERS: dict[str, str] = {}
PROMPTS: list[str] = []
SOURCE_PERSONA: str = ""
_PANEL_MODULE = None


def build_panel() -> dict[str, str]:
    """Return the 28-persona panel: source FIRST (row 0), then the 27 bystanders.

    ``no_persona`` is one of the 27 bystanders and carries an empty system
    prompt (``""``) -- the empty-system control. Persona text is always
    injected as the SYSTEM message (per CLAUDE.md persona-injection rule).
    """
    assert _PANEL_MODULE is not None, "_PANEL_MODULE must be bound by main()"
    panel: dict[str, str] = {SOURCE_PERSONA: dict(_PANEL_MODULE.PERSONAS)[SOURCE_PERSONA]}
    panel.update(BYSTANDERS)
    assert len(panel) == 1 + len(BYSTANDERS), "panel size mismatch"
    return panel


def render_prefix(tokenizer, persona_text: str, question: str) -> str:
    """Chat-template prefix with the generation prompt appended (model about to answer)."""
    msgs = [
        {"role": "system", "content": persona_text},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _kill_vllm_workers() -> None:
    """Reap vLLM TP/PP worker subprocesses, then FAIL LOUD if any GPU PID remains.

    Per CLAUDE.md gotcha: ``del llm`` + the standard vLLM cleanup does NOT reap
    worker subprocesses; they survive and re-grab the freed GPU memory the
    moment HF Transformers loads weights (looks like an HF OOM). We:

      1. ``del`` the engine + ``destroy_*`` + gc + empty_cache (best-effort).
      2. ``psutil`` terminate -> kill every child of this process.
      3. ``nvidia-smi --query-compute-apps=pid`` -- FAIL LOUD if a python PID
         other than ours still holds a GPU.
    """
    import contextlib
    import os
    import subprocess

    import psutil

    # 1. Best-effort vLLM distributed teardown.
    try:
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        print(f"[teardown] destroy_* skipped ({e})", flush=True)

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    # 2. Reap children.
    me = psutil.Process()
    children = me.children(recursive=True)
    for child in children:
        # A child can exit between enumeration and terminate -- that's the
        # success case (it's gone), not a fault to hide.
        with contextlib.suppress(psutil.NoSuchProcess):
            child.terminate()
    _gone, alive = psutil.wait_procs(children, timeout=10)
    for child in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    gc.collect()

    # 3. Fail loud if any GPU compute PID survives (other than this process).
    # epm-lint: subprocess-env-inherit -- nvidia-smi PID probe needs no credential env
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[teardown] nvidia-smi probe skipped ({e})", flush=True)
        return
    my_pid = os.getpid()
    surviving = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != my_pid and psutil.pid_exists(pid):
            surviving.append(pid)
    if surviving:
        raise RuntimeError(
            f"vLLM worker subprocesses still hold the GPU after teardown: PIDs={surviving}. "
            "These would re-grab freed memory and OOM the HF teacher-forced phase. "
            "Aborting rather than silently OOMing."
        )
    print("[teardown] vLLM workers reaped; no surviving GPU PIDs.", flush=True)


# ---------------------------------------------------------------------------
# Phase A -- vLLM on-policy generation + emission rate
# ---------------------------------------------------------------------------


def phase_a_generate(
    *,
    steps: list[int],
    run_dir: Path,
    panel: dict[str, str],
    prompts: list[str],
    n_samples: int,
    max_new_tokens: int,
    max_model_len: int,
    seed: int,
    gen_dir: Path,
) -> dict[int, dict[str, Any]]:
    """Generate on-policy completions per checkpoint; write one JSON per ckpt.

    Returns ``{step: {persona: {question_idx: [completion_text, ...]}}}`` kept in
    memory only to hand the generated answers to Phase B. The authoritative
    per-checkpoint artifact is written to disk immediately (checkpoint-per-phase).
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    persona_names = list(panel.keys())
    # Flat list of rendered prefixes; row = persona-major, question-minor.
    rendered: list[str] = []
    cell_index: list[tuple[str, int]] = []  # (persona, q_idx) per rendered row
    for pname in persona_names:
        ptext = panel[pname]
        for q_idx, q in enumerate(prompts):
            rendered.append(render_prefix(tokenizer, ptext, q))
            cell_index.append((pname, q_idx))

    # Build the engine ONCE; swap LoRA per checkpoint via distinct lora_int_id.
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        seed=seed,
    )
    sampling = SamplingParams(
        n=n_samples,
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    all_completions: dict[int, dict[str, Any]] = {}
    try:
        for step in steps:
            t0 = time.time()
            ckpt_dir = run_dir / f"checkpoint-{step}"
            assert ckpt_dir.exists(), f"missing checkpoint dir: {ckpt_dir}"
            lora_request = LoRARequest(
                lora_name=f"{SOURCE_PERSONA}_step_{step}",
                lora_int_id=step,  # distinct id => vLLM never serves a stale adapter
                lora_path=str(ckpt_dir),
            )
            outputs = llm.generate(rendered, sampling, lora_request=lora_request)

            # Aggregate per cell.
            per_persona: dict[str, dict[str, list[str]]] = {p: {} for p in persona_names}
            per_persona_counts: dict[str, dict[str, int]] = {
                p: {"n_total": 0, "n_with_marker": 0, "n_truncated": 0} for p in persona_names
            }
            for row_idx, out in enumerate(outputs):
                pname, q_idx = cell_index[row_idx]
                texts: list[str] = []
                for sample in out.outputs:
                    text = sample.text
                    texts.append(text)
                    per_persona_counts[pname]["n_total"] += 1
                    if MARKER in text:
                        per_persona_counts[pname]["n_with_marker"] += 1
                    if sample.finish_reason == "length":
                        per_persona_counts[pname]["n_truncated"] += 1
                per_persona[pname][str(q_idx)] = texts

            all_completions[step] = per_persona

            # Per-persona emission rate + truncation fraction.
            emission_rate: dict[str, float] = {}
            truncation_frac: dict[str, float] = {}
            for p in persona_names:
                c = per_persona_counts[p]
                emission_rate[p] = c["n_with_marker"] / c["n_total"] if c["n_total"] else 0.0
                truncation_frac[p] = c["n_truncated"] / c["n_total"] if c["n_total"] else 0.0

            # Source-completion truncation guard (FAIL the eval if >10% of SOURCE
            # completions truncate -- silent-zero emission risk, #260).
            src_trunc = truncation_frac.get(SOURCE_PERSONA, 0.0)
            truncation_fail = src_trunc > 0.10

            payload = {
                "phase": "a_generate",
                "step": step,
                "marker_token": MARKER,
                "marker_token_id": MARKER_ID,
                "base_model": BASE_MODEL,
                "panel": persona_names,
                "source_persona": SOURCE_PERSONA,
                "prompts": prompts,
                "n_samples": n_samples,
                "max_new_tokens": max_new_tokens,
                "gen_temperature": 1.0,
                "gen_top_p": 1.0,
                "seed": seed,
                "emission_rate": emission_rate,
                "per_persona_counts": per_persona_counts,
                "truncation_frac": truncation_frac,
                "source_truncation_frac": src_trunc,
                "truncation_fail": truncation_fail,
                "completions": per_persona,  # raw text per (persona, q_idx)
                "metadata": _metadata(),
                "wall_seconds": time.time() - t0,
            }
            gen_dir.mkdir(parents=True, exist_ok=True)
            out_path = gen_dir / f"onpolicy_gen_step{step}.json"
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(
                f"[phaseA step {step}] src_emit={emission_rate.get(SOURCE_PERSONA, 0):.3f} "
                f"src_trunc={src_trunc:.3f} wrote {out_path} ({payload['wall_seconds']:.1f}s)",
                flush=True,
            )
            if truncation_fail:
                raise RuntimeError(
                    f"step {step}: source truncation fraction {src_trunc:.3f} > 0.10 -- "
                    f"completions systematically hit max_new_tokens={max_new_tokens}; "
                    "emission rate would be a silent zero. Raise max_new_tokens and re-run."
                )
    finally:
        del llm
        gc.collect()

    return all_completions


# ---------------------------------------------------------------------------
# Phase B -- HF teacher-forced on-policy end-of-answer log p(※)
# ---------------------------------------------------------------------------


def strip_trailing_marker(text: str) -> str:
    """Remove a trailing marker (and trailing whitespace) so we don't double-count it.

    The on-policy endpos probe scores log p(※) at the position AFTER the model's
    answer. If the answer already ended in ※ we strip it (and any trailing
    whitespace) before appending the ``"\\n\\n"`` probe suffix -- otherwise the
    probe would be conditioning on the marker it is trying to predict.
    """
    stripped = text.rstrip()
    while stripped.endswith(MARKER):
        stripped = stripped[: -len(MARKER)].rstrip()
    return stripped


def phase_b_score(
    *,
    steps: list[int],
    run_dir: Path,
    panel: dict[str, str],
    prompts: list[str],
    all_completions: dict[int, dict[str, Any]],
    batch_size: int,
    device: str,
    score_dir: Path,
) -> None:
    """Teacher-forced on-policy endpos log p(※) per checkpoint; one JSON per ckpt.

    For each (persona, question, sample) the conditioning context is::

        chat_prefix(persona, question) + own_answer(marker-stripped) + "\\n\\n"

    and we score log p(※) at the next position. Per persona we record the mean
    over all (question, sample) contexts AND the full per-context vector.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_logprob

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    ).eval()

    persona_names = list(panel.keys())
    for step in steps:
        t0 = time.time()
        ckpt_dir = run_dir / f"checkpoint-{step}"
        assert ckpt_dir.exists(), f"missing checkpoint dir: {ckpt_dir}"
        adapter = PeftModel.from_pretrained(base, str(ckpt_dir))
        adapter.eval()

        per_persona_mean: dict[str, float] = {}
        per_persona_vec: dict[str, list[float]] = {}
        for pname in persona_names:
            ptext = panel[pname]
            contexts: list[str] = []
            gen_for_persona = all_completions[step][pname]
            for q_idx, q in enumerate(prompts):
                prefix = render_prefix(tokenizer, ptext, q)
                for completion in gen_for_persona[str(q_idx)]:
                    own_answer = strip_trailing_marker(completion)
                    contexts.append(prefix + own_answer + "\n\n")
            logps = compute_marker_logprob(
                adapter,
                tokenizer,
                contexts=contexts,
                marker_text=MARKER,
                batch_size=batch_size,
                device=device,
            )
            per_persona_vec[pname] = logps
            per_persona_mean[pname] = float(sum(logps) / len(logps)) if logps else float("nan")

        # Detach the adapter so the next iteration starts from the bare base.
        adapter = adapter.unload()
        del adapter
        torch.cuda.empty_cache()

        payload = {
            "phase": "b_score",
            "step": step,
            "marker_token": MARKER,
            "marker_token_id": MARKER_ID,
            "base_model": BASE_MODEL,
            "panel": persona_names,
            "source_persona": SOURCE_PERSONA,
            "prompts": prompts,
            "onpolicy_endpos_logp_mean": per_persona_mean,
            "onpolicy_endpos_logp_vector": per_persona_vec,
            "metadata": _metadata(),
            "wall_seconds": time.time() - t0,
        }
        score_dir.mkdir(parents=True, exist_ok=True)
        out_path = score_dir / f"onpolicy_endpos_logp_step{step}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(
            f"[phaseB step {step}] src_endpos_logp="
            f"{per_persona_mean.get(SOURCE_PERSONA, float('nan')):.3f} "
            f"wrote {out_path} ({payload['wall_seconds']:.1f}s)",
            flush=True,
        )


def _metadata() -> dict:
    """Reproducibility metadata (git commit, env versions, timestamp)."""
    try:
        from explore_persona_space.metadata import get_run_metadata

        return get_run_metadata()
    except Exception as e:
        return {"metadata_error": str(e)}


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Adapter dir containing checkpoint-{step}/ subdirs (the "
        "{phase}_step_checkpoints sibling, NOT the deleted *_adapter dir).",
    )
    ap.add_argument(
        "--steps",
        required=True,
        help="Comma-separated integer global_step values to evaluate on-policy.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output dir for per-checkpoint JSONs (e.g. eval_results/issue_456).",
    )
    ap.add_argument("--n-samples", type=int, default=8, help="Samples per (persona, question).")
    ap.add_argument("--max-new-tokens", type=int, default=1536, help="vLLM generation cap.")
    ap.add_argument("--max-model-len", type=int, default=4096, help="vLLM max_model_len.")
    ap.add_argument("--seed", type=int, default=42, help="Generation + engine seed.")
    ap.add_argument(
        "--batch-size", type=int, default=8, help="Sub-batch for compute_marker_logprob."
    )
    ap.add_argument("--device", default="cuda:0", help="Torch device for the HF phase.")
    ap.add_argument(
        "--panel-module",
        default="_i416_bystander_panel",
        help="scripts/ module exposing BYSTANDERS/PROMPTS/SOURCE_PERSONA/PERSONAS.",
    )
    ap.add_argument(
        "--smoke-personas",
        type=int,
        default=0,
        help="If >0, keep only the source + first (N-1) bystanders (smoke subset). 0 = full 28.",
    )
    ap.add_argument(
        "--smoke-prompts",
        type=int,
        default=0,
        help="If >0, keep only the first N prompts (smoke subset). 0 = full 20.",
    )
    args = ap.parse_args()

    # Bind the panel globals.
    import importlib

    global BYSTANDERS, PROMPTS, SOURCE_PERSONA, _PANEL_MODULE
    _PANEL_MODULE = importlib.import_module(args.panel_module)
    BYSTANDERS = dict(_PANEL_MODULE.BYSTANDERS)
    PROMPTS = list(_PANEL_MODULE.PROMPTS)
    SOURCE_PERSONA = _PANEL_MODULE.SOURCE_PERSONA

    # Marker hard guard -- bare ※ trained by #432 is id 63680, NOT " ※" (83399).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    marker_ids = tok.encode(MARKER, add_special_tokens=False)
    assert marker_ids == [MARKER_ID], (
        f"MARKER guard FAILED: '{MARKER}' -> {marker_ids}, expected [{MARKER_ID}] "
        f"(bare ※). This is the EXACT token #432 trained; ' ※' (83399) would be "
        f"train/eval drift (#396 round-1)."
    )
    print(f"[guard] marker '{MARKER}' -> {marker_ids} OK", flush=True)

    steps = [int(s) for s in args.steps.split(",")]
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    gen_dir = out_dir / "onpolicy_gen"
    score_dir = out_dir / "onpolicy_endpos_logp"

    panel = build_panel()
    prompts = PROMPTS
    if args.smoke_prompts > 0:
        prompts = prompts[: args.smoke_prompts]
    if args.smoke_personas > 0:
        # Keep source (row 0) + the next (N-1) bystanders.
        keep = list(panel.keys())[: args.smoke_personas]
        panel = {k: panel[k] for k in keep}
    print(
        f"[config] {len(panel)} personas x {len(prompts)} prompts x {args.n_samples} samples; "
        f"{len(steps)} checkpoints; max_new_tokens={args.max_new_tokens}",
        flush=True,
    )

    # Phase A: vLLM generation (writes per-ckpt gen JSON immediately).
    all_completions = phase_a_generate(
        steps=steps,
        run_dir=run_dir,
        panel=panel,
        prompts=prompts,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        seed=args.seed,
        gen_dir=gen_dir,
    )

    # Teardown vLLM workers BEFORE loading HF Transformers (CLAUDE.md gotcha).
    _kill_vllm_workers()

    # Phase B: HF teacher-forced on-policy endpos log p (per-ckpt score JSON).
    phase_b_score(
        steps=steps,
        run_dir=run_dir,
        panel=panel,
        prompts=prompts,
        all_completions=all_completions,
        batch_size=args.batch_size,
        device=args.device,
        score_dir=score_dir,
    )

    print(f"[done] on-policy eval complete for steps {steps}", flush=True)


if __name__ == "__main__":
    main()
