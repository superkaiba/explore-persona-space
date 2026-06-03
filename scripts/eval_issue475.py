#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #475 on-policy eval — port of scripts/eval_issue408.py (issue-408 branch).

Slimmed down to the load-bearing primitives for the 3-arm × 2-ckpt × 4-cell
factorial. Plan §4.8 / §6.

DV recipe (plan §6.1, marker-leakage-measurement.md):
  For each held-out prompt P in cell C and each (arm, ckpt) Tᵢ:
    1. Build the chat context = system + user message (T+ prefixed with
       trigger; T- not; NEG-doctor uses doctor system).
    2. vLLM greedy generate the trained model's response Rᵢ (max_new_tokens
       2048 for plain/distilled; 3072 for visible_cot — guard against
       silent truncation of the trailing marker, #260).
    3. Teacher-forced log p(' ※') at the slot AFTER Rᵢ's last non-EOS
       token, scored on BOTH the trained checkpoint and on bare
       Qwen3.5-27B. DV = trained − base.
    4. Also record binary "did Rᵢ end with ' ※'?" (the saturating
       on-policy argmax read — sanity anchor for the log-prob).
    5. Visible-CoT arm only: parse <scratchpad>, regex-classify
       articulation, optionally run causality-edit cycle.

Output: per-cell, per-arm, per-ckpt JSON under
``eval_results/issue_475/{arm}/{ckpt}/cell_{C}.json`` plus a roll-up
``run_summary.json``. raw_completions.json per (arm, ckpt) is uploaded
to HF data repo via upload_raw_completions_to_data_repo().

Usage (per (arm, ckpt) cell of the matrix):
    uv run python scripts/eval_issue475.py \\
        --arm plain --ckpt phase1 --seed 42

Smoke (one-cell, 20 prompts each):
    uv run python scripts/eval_issue475.py \\
        --arm plain --ckpt phase1 --seed 42 --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ = subprocess

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT, bootstrap  # noqa: E402

bootstrap(log_name="eval_issue475")

from _issue475_common import (  # noqa: E402
    ARMS,
    BASE_MODEL,
    DEFAULT_ASSISTANT_KEY,
    EVAL_RESULTS_DIR,
    MARKER_TEXT,
    TRIGGER_KEY,
    all_persona_prompts,
    classify_scratchpad_articulation,
    marker_preflight,
    parse_scratchpad,
    truncated,
)

log = logging.getLogger("eval_issue475")


# ── Plan §4.8 cell sizes ────────────────────────────────────────────────────
N_T_PROMPTS = 200
N_NEG_DOCTOR = 50
N_NEG_DEFAULT_OTHER = 50
# Smoke per-cell cap. With this many prompts in the eval pool, _build_cells
# slices [:N_PROMPTS_SMOKE] for T+/T- and [N_PROMPTS_SMOKE : 2*N_PROMPTS_SMOKE]
# for NEG_default_other; run_one() then truncates each cell to N_PROMPTS_SMOKE.
# The data-gen smoke writes >= 2*N_PROMPTS_SMOKE eval questions so EVERY cell
# (T+/T-/NEG_doctor/NEG_default_other) lands non-empty on smoke (round-2 fix 2).
N_PROMPTS_SMOKE = 20
N_EVAL_QUESTIONS_SMOKE_REQUIRED = 2 * N_PROMPTS_SMOKE  # 40: T-slice + NEG_default_other slice

# Plan §4.8: max_new_tokens — 2048 plain/distilled, 3072 visible_cot.
MAX_NEW_TOKENS_DEFAULT = 2048
MAX_NEW_TOKENS_VISIBLE_COT = 3072

# Logprob teacher-forced batch.
LOGPROB_BATCH_SIZE = 8


def _max_new_tokens(arm: str) -> int:
    return MAX_NEW_TOKENS_VISIBLE_COT if arm == "visible_cot" else MAX_NEW_TOKENS_DEFAULT


# ── Question pools ──────────────────────────────────────────────────────────


def _load_eval_questions(*, smoke: bool, seed: int) -> list[str]:
    """Read the DISJOINT held-out eval slice produced by
    gen_issue475_scaffold_data.py (eval_questions.json — guaranteed not to
    overlap with the training pool used by data-gen).

    Full run: returns 250 questions (T+/T-/NEG_doctor use the first 200,
    NEG_default_other uses [200:250] — see _build_cells).
    Smoke: returns N_EVAL_QUESTIONS_SMOKE_REQUIRED (40) questions — sized so
    _build_cells slices a non-empty T+/T-/NEG_doctor slice AND a non-empty
    NEG_default_other slice. Data-gen smoke writes >= 40 eval questions for
    this reason (round-2 fix 2; before the fix, smoke wrote 5 → eval crashed
    here before any GPU phase).

    FAILS LOUD if eval_questions.json is missing or undersized — never falls
    back to training questions (the round-1 review caught this as inflating
    survival numbers via memorization).
    """
    from _issue475_common import DATA_DIR

    cache = DATA_DIR / "eval_questions.json"
    if not cache.exists():
        raise RuntimeError(
            "eval_questions.json missing — run gen_issue475_scaffold_data.py "
            "(any --step that touches assemble) to produce the held-out eval slice. "
            "The pre-Round-2 fallback to training questions is REMOVED to prevent "
            "memorization-inflated survival numbers."
        )
    qs_eval = json.loads(cache.read_text())
    required = N_EVAL_QUESTIONS_SMOKE_REQUIRED if smoke else (N_T_PROMPTS + N_NEG_DEFAULT_OTHER)
    if len(qs_eval) < required:
        raise RuntimeError(
            f"eval_questions.json has only {len(qs_eval)} items; need ≥{required} "
            f"({'smoke' if smoke else 'full'} run). Regenerate via "
            "gen_issue475_scaffold_data.py."
        )
    # Deterministic shuffle within the held-out pool so the seed picks a
    # stable ordering (the OOD vs T+/T- contrast lives in the slice we
    # take from the SHUFFLED order, not in the cache's input order).
    qs_eval = list(qs_eval)
    rng = random.Random(seed)
    rng.shuffle(qs_eval)
    if smoke:
        # Return enough for T-slice + NEG_default_other-slice; _build_cells
        # uses _smoke_slice_sizes() to slice the right portions.
        return qs_eval[:N_EVAL_QUESTIONS_SMOKE_REQUIRED]
    # Full run: return the full held-out pool so _build_cells can slice
    # both [:200] for T+/T- and [200:250] for NEG_default_other.
    return qs_eval[: N_T_PROMPTS + N_NEG_DEFAULT_OTHER]


# ── Cell construction ──────────────────────────────────────────────────────


def _build_cells(eval_questions: list[str], *, smoke: bool = False) -> dict[str, list[dict]]:
    """Plan §4.8 cell table — returns ``{cell: [{system, user, persona_key, trigger}, ...]}``.

    Slices over ``eval_questions`` (DISJOINT held-out pool from
    eval_questions.json; never overlaps training):
      - Full run: T_plus / T_minus / NEG_doctor use eval_questions[:200]
        (same questions; NEG_doctor takes the first 50); NEG_default_other
        uses eval_questions[200:250] (DISJOINT OOD slice — Round-2 fix 3).
      - Smoke (round-2 fix 2): with eval_questions sized to
        N_EVAL_QUESTIONS_SMOKE_REQUIRED (40) items, T_plus / T_minus /
        NEG_doctor use eval_questions[:N_PROMPTS_SMOKE] and
        NEG_default_other uses eval_questions[N_PROMPTS_SMOKE : 2*N_PROMPTS_SMOKE]
        (a DISJOINT 20-question slice). Before the fix, smoke wrote 5
        eval questions → NEG_default_other was an empty slice → eval cell
        was silently zero. The empty-cell guard below now also fail-louds.

    FAIL-LOUD (round-2 fix 3): any cell ending up empty raises a
    RuntimeError — under the contract "no silent-zero cells". Smoke must
    write enough eval questions to populate every cell.

    The contrast NEG_default_other holds against T_plus is "different
    question distribution under the default assistant"; question overlap
    with T_plus is forbidden, never silently.
    """
    personas = all_persona_prompts()
    asst = personas[DEFAULT_ASSISTANT_KEY]
    doctor = personas["medical_doctor"]

    def _trig(q: str) -> str:
        return f"{TRIGGER_KEY} {q}"

    # Slice indices.
    #   - Full run: [:200] for T cells, [200:250] for NEG_default_other.
    #   - Smoke: [:N_PROMPTS_SMOKE] for T cells,
    #     [N_PROMPTS_SMOKE : 2*N_PROMPTS_SMOKE] for NEG_default_other.
    if smoke:
        t_slice = N_PROMPTS_SMOKE
        neg_doctor_slice = N_PROMPTS_SMOKE
        neg_default_other_start = N_PROMPTS_SMOKE
        neg_default_other_stop = N_PROMPTS_SMOKE + N_PROMPTS_SMOKE
    else:
        t_slice = N_T_PROMPTS
        neg_doctor_slice = N_NEG_DOCTOR
        neg_default_other_start = N_T_PROMPTS
        neg_default_other_stop = N_T_PROMPTS + N_NEG_DEFAULT_OTHER

    t_qs = eval_questions[:t_slice]
    neg_default_other_qs = eval_questions[neg_default_other_start:neg_default_other_stop]

    cells = {
        "T_plus": [
            {
                "system": asst,
                "user": _trig(q),
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "trigger": True,
            }
            for q in t_qs
        ],
        "T_minus": [
            {"system": asst, "user": q, "persona_key": DEFAULT_ASSISTANT_KEY, "trigger": False}
            for q in t_qs
        ],
        "NEG_doctor": [
            {"system": doctor, "user": _trig(q), "persona_key": "medical_doctor", "trigger": True}
            for q in t_qs[:neg_doctor_slice]
        ],
        "NEG_default_other": [
            {
                "system": asst,
                "user": _trig(q),
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "trigger": True,
            }
            for q in neg_default_other_qs
        ],
    }
    # Defensive disjoint check — NEG_default_other questions MUST NOT
    # appear in T+/T-. Catches a future refactor that silently re-overlaps
    # the slices.
    t_set = set(t_qs)
    overlap = [q for q in neg_default_other_qs if q in t_set]
    if overlap:
        raise RuntimeError(
            f"NEG_default_other overlaps T+/T- on {len(overlap)} questions "
            f"(first: {overlap[0][:60]!r}). The held-out eval pool is too small "
            "or eval_questions.json was regenerated incorrectly."
        )
    # Round-2 fix 3: FAIL-LOUD on any empty cell — no silent zero counts.
    # Catches future refactors that re-overlap slices OR shrink the eval
    # pool below what _build_cells needs.
    empty = [k for k, v in cells.items() if len(v) == 0]
    if empty:
        raise RuntimeError(
            f"Empty cell(s) produced by _build_cells (smoke={smoke}): {empty}. "
            f"eval_questions has {len(eval_questions)} items; need >= "
            f"{neg_default_other_stop} to populate every cell. "
            "Regenerate via gen_issue475_scaffold_data.py (smoke writes >= "
            f"{N_EVAL_QUESTIONS_SMOKE_REQUIRED}; full writes >= "
            f"{N_T_PROMPTS + N_NEG_DEFAULT_OTHER})."
        )
    return cells


# ── Checkpoint resolution ──────────────────────────────────────────────────


def _adapter_subfolder(arm: str, seed: int, ckpt: str) -> str:
    return f"c_issue475_qwen35_27b_{arm}_seed{seed}_{ckpt}"


def _resolve_adapter_local(arm: str, seed: int, ckpt: str) -> Path:
    """Download the per-arm, per-ckpt adapter and return its local path."""
    from _issue475_common import HUB_MODEL_REPO
    from huggingface_hub import snapshot_download

    sub = f"adapters/{_adapter_subfolder(arm, seed, ckpt)}"
    log.info("Resolving adapter: %s/%s", HUB_MODEL_REPO, sub)
    local = snapshot_download(
        repo_id=HUB_MODEL_REPO,
        allow_patterns=[f"{sub}/*"],
        token=os.environ.get("HF_TOKEN"),
    )
    adapter_dir = Path(local) / sub
    if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
        raise FileNotFoundError(
            f"Adapter directory empty or missing: {adapter_dir}. "
            f"Check that scripts/run_issue475_cot_install.py {ckpt} ran "
            "and uploaded its adapter."
        )
    return adapter_dir


# ── vLLM generation ───────────────────────────────────────────────────────


def _make_chat_prefix(system: str, user: str, tokenizer: Any) -> str:
    """Build the chat-template prefix for Qwen3.5-27B.

    Passes ``enable_thinking=False`` to suppress Qwen3.5's native ``<think>``
    substrate — our scaffold uses its own ``<scratchpad>`` tags. Without this,
    visible-CoT confounds the hand-trained scaffold with the model's hidden
    thinking. ``apply_chat_template`` ignores unknown kwargs on templates that
    don't support ``enable_thinking`` (older Qwen3 BPE templates pre-3.5), so
    this is safe across the tokenizer surface — but the round-2 review (codex
    twin "native-thinking-not-disabled") flagged it as a load-bearing minor.
    """
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _generate_completions(
    *,
    adapter_path: Path,
    arm: str,
    cells: dict[str, list[dict]],
    max_new_tokens: int,
    tp_size: int = 2,
) -> dict[str, list[dict]]:
    """vLLM greedy gen for every (cell, prompt). Returns
    ``{cell: [{prefix, completion_text, n_generated_tokens, truncated, ...}, ...]}``.
    """
    from vllm import LLM, SamplingParams

    log.info("Loading vLLM (TP=%d) with adapter %s", tp_size, adapter_path)
    llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=8192,
        trust_remote_code=True,
    )

    from vllm.lora.request import LoRARequest

    lora_req = LoRARequest("issue475_adapter", 1, str(adapter_path))

    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        n=1,
    )

    out: dict[str, list[dict]] = {}
    for cell_name, items in cells.items():
        prefixes = [_make_chat_prefix(it["system"], it["user"], tokenizer) for it in items]
        log.info("Generating cell=%s arm=%s n=%d", cell_name, arm, len(prefixes))
        responses = llm.generate(prefixes, sampling, lora_request=lora_req)
        recs: list[dict] = []
        for it, resp in zip(items, responses, strict=True):
            gen = resp.outputs[0]
            text = gen.text
            n_gen = len(gen.token_ids)
            recs.append(
                {
                    "system": it["system"],
                    "user": it["user"],
                    "persona_key": it["persona_key"],
                    "trigger": it["trigger"],
                    "prefix": _make_chat_prefix(it["system"], it["user"], tokenizer),
                    "completion_text": text,
                    "n_generated_tokens": n_gen,
                    "truncated": truncated(n_gen, max_new_tokens),
                    "ended_with_marker": text.rstrip().endswith(MARKER_TEXT.rstrip()),
                }
            )
        out[cell_name] = recs

    # Tear down vLLM cleanly so the next HF-Transformers load (for the
    # logprob block) doesn't OOM via vLLM worker subprocess survivors —
    # CLAUDE.md vLLM gotcha.
    _teardown_vllm(llm)
    return out


def _teardown_vllm(llm: Any) -> None:
    """Best-effort vLLM cleanup that reaps worker subprocesses too.

    The canonical `del llm` is NOT enough — vLLM TP worker subprocesses
    survive and re-grab freed GPU memory. See .claude/rules/gotchas.md +
    feedback_vllm_orphan_worker_after_destroy.md.
    """
    import gc

    import psutil
    import torch

    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        log.warning("vllm distributed teardown raised: %s", e)
    import contextlib

    with contextlib.suppress(Exception):
        del llm
    gc.collect()
    torch.cuda.empty_cache()
    # Reap any worker subprocesses our vLLM spawned (gpu_uuid-naive — single
    # process invocation here, so all children are ours).
    me = psutil.Process()
    for child in me.children(recursive=True):
        try:
            child.terminate()
            child.wait(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                child.kill()


# ── On-policy log-prob (trained AND base, same R) ──────────────────────────


def _load_vlm_aware_config(model_path_or_hub_id: str):
    """Return an AutoConfig with ``vocab_size`` set at top level.

    Qwen3.5-27B (and other unified VLMs) carry ``vocab_size`` under
    ``config.text_config.vocab_size`` instead of at top level. The HF
    ``modeling_utils`` paths that read ``config.vocab_size`` directly raise
    ``AttributeError`` on these configs. Mirror the working ``train_lora`` loader
    (which also passes ``attn_implementation`` and never tripped this on Phase 1)
    by surfacing the nested ``vocab_size`` to the top-level attribute when
    missing. No-op for ordinary causal-LM configs.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(
        model_path_or_hub_id,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if not hasattr(cfg, "vocab_size") or getattr(cfg, "vocab_size", None) is None:
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and getattr(text_cfg, "vocab_size", None) is not None:
            cfg.vocab_size = text_cfg.vocab_size
            log.info(
                "VLM config detected (%s) — surfaced text_config.vocab_size=%d to top level",
                type(cfg).__name__,
                cfg.vocab_size,
            )
        else:
            # Fail-loud: not a VLM and no top-level vocab_size — caller's bug,
            # not something we should silently paper over.
            raise AttributeError(
                f"AutoConfig for {model_path_or_hub_id!r} has neither top-level "
                f"vocab_size nor text_config.vocab_size; cannot construct loader config."
            )
    return cfg


def _compute_logprob_for_records(
    *,
    model_path_or_hub_id: str,
    records: list[dict],
    marker_text: str,
    device: str = "cuda:0",
    is_adapter: bool = False,
    base_for_adapter: str | None = None,
) -> list[float]:
    """Score log P(marker) at the slot after R's last non-EOS token.

    ``records`` carries the prefix (chat-template) AND the model's own
    completion_text. The scored context is `prefix + completion_text`,
    rstripped to drop the trailing EOS / whitespace so the marker lands
    exactly at the post-response slot.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_logprob
    from explore_persona_space.train.sft import _pick_attn_implementation

    log.info(
        "Loading model for logprob: %s%s",
        model_path_or_hub_id,
        f" + adapter ({base_for_adapter} base)" if is_adapter else "",
    )
    if is_adapter:
        assert base_for_adapter, "base_for_adapter required when is_adapter=True"
        tokenizer = AutoTokenizer.from_pretrained(
            base_for_adapter, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        # NOTE: do NOT pass an explicit `config=` here. Qwen3.5 is a VLM
        # (Qwen3_5Config with nested text_config); passing the flat config
        # makes transformers build the causal-LM head from top-level
        # vocab_size/hidden_size (which the VLM config lacks). Letting
        # AutoModelForCausalLM resolve the config itself (with
        # attn_implementation set, mirroring train_lora's working load)
        # routes through the text_config correctly. (#475 canary)
        base = AutoModelForCausalLM.from_pretrained(
            base_for_adapter,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation=_pick_attn_implementation(),
            token=os.environ.get("HF_TOKEN"),
        )
        model = PeftModel.from_pretrained(base, model_path_or_hub_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_or_hub_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        # See note above: no explicit config= for the Qwen3.5 VLM. (#475 canary)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_hub_id,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation=_pick_attn_implementation(),
            token=os.environ.get("HF_TOKEN"),
        )
    model.eval()

    # Compose the on-policy context (prefix + completion_text). Strip the
    # trailing marker if the model emitted it — we want to score log P(※)
    # at the post-response slot regardless of whether it materialized.
    contexts: list[str] = []
    for r in records:
        completion = r["completion_text"]
        # Strip ANY trailing marker so we score at the same slot whether or
        # not the model already wrote it. Strip 1 trailing marker + trailing
        # whitespace; preserve everything else.
        comp_clean = completion.rstrip()
        if comp_clean.endswith(MARKER_TEXT.rstrip()):
            comp_clean = comp_clean[: -len(MARKER_TEXT.rstrip())].rstrip()
        contexts.append(r["prefix"] + comp_clean)

    lps = compute_marker_logprob(
        model,
        tokenizer,
        contexts=contexts,
        marker_text=marker_text,
        position="end_of_answer",
        batch_size=LOGPROB_BATCH_SIZE,
        device=device,
    )
    for v in lps:
        if not math.isfinite(v):
            raise RuntimeError(f"Non-finite log-prob ({v}) — tokenizer / template drift.")
    del model
    if is_adapter:
        del base
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return lps


# ── vLLM-free subprocess for HF log-prob scoring ──────────────────────────
# vLLM engine creation monkey-patches `transformers` in-process (LLM-engine
# instantiation registers custom architectures / shims the config classes).
# After that monkey-patch, the subsequent `AutoModelForCausalLM.from_pretrained`
# load of Qwen3.5-27B in the SAME process fails with
# ``'Qwen3_5Config' object has no attribute 'vocab_size'``. A standalone load
# in a fresh process (no vLLM ever imported) works clean — so we hand the
# HF log-prob scoring to a subprocess that NEVER touches vllm. This was the
# 6-rounds-of-loader-edits root cause confirmed empirically by
# ``scripts/diag_loaders.py`` and the in-process vs subprocess A/B on pod-475.


def _run_logprob_subprocess_manifest(
    *,
    manifest_path: Path,
    log_path: Path,
) -> None:
    """Re-invoke this script in ``--logprob-worker`` mode in a fresh process.

    The manifest JSON shape:

    .. code-block:: json

       {"model": "...", "is_adapter": false, "base_for_adapter": null,
        "marker": " ※",
        "cells": [{"name": "T_plus", "records_in": "...", "out": "..."}, ...]}

    The worker loads ``model`` ONCE, then iterates cells, writing each
    cell's per-cell ``out`` JSON the moment it computes it
    (checkpoint-per-phase). A mid-scoring crash leaves prior cells'
    files intact and the parent can fail loud / resume.

    The worker imports ONLY transformers + peft + torch (never vllm) —
    that is the entire point of the subprocess (vLLM engine creation in
    the parent monkey-patches transformers, breaking subsequent HF loads
    of Qwen3.5-27B in-process).

    Env is passed through explicitly so HF_TOKEN / HF_HOME /
    WANDB_API_KEY survive ``uv run`` and the fresh subprocess.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--logprob-worker",
        "--manifest",
        str(manifest_path),
    ]
    log.info("Spawning logprob subprocess (manifest=%s log=%s)", manifest_path, log_path)
    env = {**os.environ}
    with log_path.open("ab") as logf:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        tail = ""
        try:
            with log_path.open("rb") as f:
                f.seek(max(0, log_path.stat().st_size - 4096))
                tail = f.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"Logprob subprocess failed (rc={proc.returncode}) for "
            f"manifest={manifest_path}; tail of {log_path}:\n{tail}"
        )


def _logprob_worker_main(*, manifest_path: Path) -> int:
    """vLLM-free worker entry. Loads model ONCE, scores every cell in manifest.

    DELIBERATELY does NOT import vllm anywhere on its call path — that is the
    entire point of running in a subprocess. ``_compute_logprob_for_records``
    imports only transformers + peft + torch.

    Writes each cell's ``out`` JSON as soon as that cell's log-probs are
    computed (checkpoint-per-phase), so a mid-iteration crash loses at most
    one cell.
    """
    manifest = json.loads(manifest_path.read_text())
    model_id = manifest["model"]
    is_adapter = bool(manifest["is_adapter"])
    base_for_adapter = manifest.get("base_for_adapter")
    marker_text = manifest["marker"]
    cells = manifest["cells"]
    log.info(
        "Logprob worker: model=%s is_adapter=%s n_cells=%d marker=%r",
        model_id,
        is_adapter,
        len(cells),
        marker_text,
    )

    # Hot-load the model ONCE here and score every cell against it. We
    # cannot easily reuse _compute_logprob_for_records (which loads + frees
    # the model per call) without amortizing the 27B load, so inline the
    # load and call compute_marker_logprob directly.
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_logprob
    from explore_persona_space.train.sft import _pick_attn_implementation

    if is_adapter:
        assert base_for_adapter, "base_for_adapter required when is_adapter=True"
        tokenizer = AutoTokenizer.from_pretrained(
            base_for_adapter, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_for_adapter,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation=_pick_attn_implementation(),
            token=os.environ.get("HF_TOKEN"),
        )
        model = PeftModel.from_pretrained(base, model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation=_pick_attn_implementation(),
            token=os.environ.get("HF_TOKEN"),
        )
    model.eval()

    for cell in cells:
        cell_name = cell["name"]
        records = json.loads(Path(cell["records_in"]).read_text())
        log.info("Worker scoring cell=%s n=%d", cell_name, len(records))
        contexts: list[str] = []
        for r in records:
            completion = r["completion_text"]
            comp_clean = completion.rstrip()
            if comp_clean.endswith(MARKER_TEXT.rstrip()):
                comp_clean = comp_clean[: -len(MARKER_TEXT.rstrip())].rstrip()
            contexts.append(r["prefix"] + comp_clean)
        lps = compute_marker_logprob(
            model,
            tokenizer,
            contexts=contexts,
            marker_text=marker_text,
            position="end_of_answer",
            batch_size=LOGPROB_BATCH_SIZE,
            device="cuda:0",
        )
        for v in lps:
            if not math.isfinite(v):
                raise RuntimeError(
                    f"Non-finite log-prob ({v}) in cell={cell_name} — tokenizer / template drift."
                )
        out_path = Path(cell["out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Checkpoint-per-phase: persist this cell IMMEDIATELY before moving on.
        out_path.write_text(json.dumps(lps))
        log.info("Worker wrote %d values → %s", len(lps), out_path)

    del model
    if is_adapter:
        del base
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# ── Roll-up per cell ───────────────────────────────────────────────────────


def _summarize_cell(
    cell_name: str,
    records: list[dict],
    lps_trained: list[float],
    lps_base: list[float],
    arm: str,
) -> dict:
    n = len(records)
    if n == 0:
        return {"cell": cell_name, "n": 0}
    deltas = [t - b for t, b in zip(lps_trained, lps_base, strict=True)]
    fired = sum(1 for r in records if r["ended_with_marker"])
    truncs = sum(1 for r in records if r["truncated"])
    summary = {
        "cell": cell_name,
        "arm": arm,
        "n": n,
        "trained_logp_median": sorted(lps_trained)[n // 2],
        "base_logp_median": sorted(lps_base)[n // 2],
        "delta_logp_median": sorted(deltas)[n // 2],
        "delta_logp_mean": sum(deltas) / n,
        "fire_rate": fired / n,
        "n_fired": fired,
        "truncation_rate": truncs / n,
        "n_truncated": truncs,
    }
    # Arm B (visible_cot) only: articulation rate
    if arm == "visible_cot":
        art_correct = 0
        art_seen = 0
        for r in records:
            sp = parse_scratchpad(r["completion_text"])
            if sp is None:
                continue
            art_seen += 1
            cls = classify_scratchpad_articulation(sp, trigger_present=r["trigger"])
            if cls["articulates_correctly"]:
                art_correct += 1
        summary["scratchpad_present_rate"] = art_seen / n
        summary["articulation_rate"] = (art_correct / art_seen) if art_seen else 0.0
        summary["n_scratchpad_present"] = art_seen
        summary["n_articulated_correctly"] = art_correct
    return summary


# ── Per-(arm,ckpt) runner ─────────────────────────────────────────────────


def run_one(args: argparse.Namespace) -> dict:
    """Run eval for ONE (arm, ckpt, seed) cell of the matrix."""
    marker_preflight()
    arm = args.arm
    ckpt = args.ckpt
    seed = args.seed
    out_root = EVAL_RESULTS_DIR / arm / ckpt
    out_root.mkdir(parents=True, exist_ok=True)

    qs = _load_eval_questions(smoke=args.smoke, seed=seed)
    cells = _build_cells(qs, smoke=args.smoke)
    # _build_cells already sizes smoke slices to N_PROMPTS_SMOKE per T cell
    # and N_PROMPTS_SMOKE per NEG_default_other cell, so no further truncation
    # is needed. The earlier blanket truncation was masking the empty-cell bug
    # (round-2 fix 2 + fix 3).
    log.info(
        "Eval matrix cell: arm=%s ckpt=%s seed=%d; cells=%s",
        arm,
        ckpt,
        seed,
        {k: len(v) for k, v in cells.items()},
    )

    adapter_path = _resolve_adapter_local(arm, seed, ckpt)

    # Step 1: vLLM greedy gen on trained checkpoint.
    completions = _generate_completions(
        adapter_path=adapter_path,
        arm=arm,
        cells=cells,
        max_new_tokens=_max_new_tokens(arm),
        tp_size=args.tp_size,
    )

    # Persist raw completions IMMEDIATELY — plan §6.3 + CLAUDE.md
    # checkpoint-per-phase rule.
    raw_path = out_root / "raw_completions.json"
    raw_path.write_text(json.dumps(completions, indent=2))
    log.info("Wrote raw completions to %s", raw_path)

    # Steps 2 + 3: log P(marker) on trained, then on bare base — in
    # SUBPROCESSES that never import vllm. vLLM engine creation in this
    # parent process monkey-patched ``transformers`` (registered the VLM
    # config shim), so any in-process `AutoModelForCausalLM.from_pretrained`
    # of Qwen3.5-27B explodes with ``Qwen3_5Config has no attribute
    # vocab_size``. A fresh subprocess that never imports vllm loads clean.
    # See ``_run_logprob_subprocess_manifest`` docstring + canary evidence on pod-475.
    #
    # Per CLAUDE.md "Checkpoint per phase": the worker writes EACH cell's
    # per-cell JSON the moment it computes it (one file per cell, parent
    # reads back after subprocess exit), so a mid-scoring crash on cell N
    # loses at most one cell. Workers are amortized one per model so we
    # only pay the 27B load twice (trained + base) instead of 2x N_cells.

    # Persist per-cell records to disk so the subprocess can read them back.
    records_dir = out_root / "logprob_input"
    records_dir.mkdir(parents=True, exist_ok=True)
    cell_record_paths: dict[str, Path] = {}
    for cell_name, recs in completions.items():
        p = records_dir / f"{cell_name}.json"
        p.write_text(json.dumps(recs))
        cell_record_paths[cell_name] = p

    log_dir = out_root / "logprob_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def _score_all_cells(*, model_id: str, is_adapter: bool, label: str) -> dict[str, list[float]]:
        """Spawn ONE vLLM-free subprocess that scores every cell under ``model_id``.

        Writes ``{label}_logp_<cell>.json`` per cell as the worker completes
        each one (checkpoint-per-phase). Returns the parsed dict of per-cell
        log-prob lists.
        """
        manifest = {
            "model": model_id,
            "is_adapter": is_adapter,
            "base_for_adapter": BASE_MODEL if is_adapter else None,
            "marker": MARKER_TEXT,
            "cells": [
                {
                    "name": cn,
                    "records_in": str(cell_record_paths[cn]),
                    "out": str(out_root / f"{label}_logp_{cn}.json"),
                }
                for cn in completions
            ],
        }
        manifest_path = records_dir / f"manifest_{label}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log_path = log_dir / f"{label}_worker.log"
        _run_logprob_subprocess_manifest(manifest_path=manifest_path, log_path=log_path)
        out: dict[str, list[float]] = {}
        for cell_name in completions:
            cell_out = out_root / f"{label}_logp_{cell_name}.json"
            if not cell_out.exists():
                raise RuntimeError(
                    f"{label} worker exited but {cell_out} is missing; see {log_path}"
                )
            out[cell_name] = json.loads(cell_out.read_text())
        return out

    trained_lps_per_cell = _score_all_cells(
        model_id=str(adapter_path), is_adapter=True, label="trained"
    )
    base_lps_per_cell = _score_all_cells(model_id=BASE_MODEL, is_adapter=False, label="base")

    # Step 4: roll up.
    cell_summaries = {
        cn: _summarize_cell(
            cn, completions[cn], trained_lps_per_cell[cn], base_lps_per_cell[cn], arm
        )
        for cn in cells
    }

    run_summary = {
        "arm": arm,
        "ckpt": ckpt,
        "seed": seed,
        "smoke": args.smoke,
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "trigger_key": TRIGGER_KEY,
        "max_new_tokens": _max_new_tokens(arm),
        "cells": cell_summaries,
        "t_unix": time.time(),
    }
    (out_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2))

    # Per CLAUDE.md upload policy: push raw completions to HF data repo
    # before the pod is reaped. The launcher invokes this at sweep end too.
    if not args.skip_upload:
        try:
            from explore_persona_space.orchestrate.hub import (
                upload_raw_completions_to_data_repo,
            )

            urls = upload_raw_completions_to_data_repo(
                experiment_name=f"issue_475_{arm}_{ckpt}",
                eval_results_dir=out_root,
            )
            log.info("Uploaded %d raw_completions files", len(urls))
        except Exception as e:
            log.warning("Raw-completions upload failed (continuing): %s", e)

    return run_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Issue #475 on-policy eval — per (arm, ckpt) cell of the matrix. "
            "Generates with vLLM greedy, scores log P(' ※') trained-base at "
            "end-of-response."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Worker-mode flags (hidden from the main eval surface — only used when
    # the script re-invokes itself in a vLLM-free subprocess).
    p.add_argument(
        "--logprob-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--manifest", type=str, default=None, help=argparse.SUPPRESS)

    # Normal eval-mode flags.
    p.add_argument("--arm", choices=ARMS, required=False)
    p.add_argument("--ckpt", choices=("phase1", "phase2"), required=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "20 prompts/cell instead of 200/50; CPU-feasibility for everything else stays the same."
        ),
    )
    p.add_argument("--skip-upload", action="store_true")
    args = p.parse_args()
    # Worker mode requires --manifest; eval mode requires --arm / --ckpt.
    # Validate here (we can't use required=True on --arm because worker
    # mode short-circuits without it).
    if args.logprob_worker:
        if not args.manifest:
            p.error("--logprob-worker requires --manifest")
    else:
        if not args.arm or not args.ckpt:
            p.error("--arm and --ckpt are required (omit only in --logprob-worker mode)")
    return args


def main() -> int:
    args = parse_args()
    if args.logprob_worker:
        # vLLM-free subprocess path. DO NOT import vllm anywhere in this
        # branch — that is the whole point of the subprocess.
        return _logprob_worker_main(manifest_path=Path(args.manifest))
    run_one(args)
    _ = PROJECT_ROOT  # silence the unused-import warning on slim paths
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
