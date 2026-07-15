#!/usr/bin/env python
# ruff: noqa: RUF002  # em-dash + marker token intentional
"""#1333 pod-side phase driver — marker 2x2 at matched install + 4-context breadth.

Phases (linear, checkpoint-per-phase, resume-keyed; plan §4/§9):

  p0_stage    stage + rev-pin every reused input (frozen marker mix, base
              pooled store + raw rows, m2 pooled store + selection, R_train,
              the reused m2 FT checkpoint) + the reused-arm apply-and-read gate
  p1_mixes    derive posonly + extension mixes (rendered-token disjointness
              asserts; ICL bank fill; base greedy R gens) + upload BEFORE train
  p2_train    5 LoRA ladders (work-conserving fanout, 1 GPU/cell) + mk4
              full-FT grid (ZeRO-3 subprocess, whole-pod)
  p3_ladder   base-prior reads + off-line eval-surface three-space ladders
              (coarse-to-fine; per-rung resume; sharded 1 cell/GPU)
  p4_select   registered selection + bystander de-saturation confirm +
              FT rung retention (upload selected THEN delete non-selected)
  p5_capture  own-text captures (2x2 new cells; 5-context x 20-q panel)
  p6_tf_shared teacher-forced shared-text re-capture (all four 2x2 cells,
              incl. the reused arm) over the pinned base_marker rows
  p7_breadth  install/expression/leakage battery over the de-duplicated
              8-rendered-context panel (breadth cells)
  p8_upload   text/JSON (unconditional) + pooled tensors + selected ckpts;
              sentinel

``--smoke`` is the SAME dispatcher at tiny knobs (plan § Dry-run smoke item 2):
cells (mk1_lora_con, mk4_fullft_pos, mk3_fullft_con), 2-step LoRA train + FT
consolidation canary (1 rung, num_processes=1), BOTH HALT gates live (the
same-surface parity gate AND the reused-arm apply-and-read gate at
eval_question_limit with a doubled bound) + the PEFT-swap-vs-merged parity
read, 1 ladder read at 2 questions, 2-context x 2-question capture, tf-shared
stub (incl. the reused arm), geometry stub via issue1333_geometry.
Every phase reads its cell list from the ONE resolver (``cfg.cells``).

``[phase=done]`` is emitted by the launch wrapper ONLY (pod-side-reporting.md).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

# vLLM v1 EngineCore fork-poisoning guard (gotchas.md #628): set BEFORE any
# vllm import — this dispatcher touches tokenizers/transformers pre-LLM().
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

# Reused, cfg-independent parent helpers (scripts/issue1112_dispatch.py):
# rung enumeration, tokenizer repair, weight completeness, overflow staging,
# tf-row contract asserts, fanout group reaping, adapter merge (its ``cfg``
# parameter is unused by the body — passed None below).
import issue1112_dispatch as d1112  # noqa: E402

from explore_persona_space.artifacts.context import (  # noqa: E402
    CONTEXTS,
    Context,
    context_for_persona,
    icl_prefix_context,
)
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    _sha256_file,
    release_trainer_cuda_memory,
)
from explore_persona_space.experiments import issue_1333 as C  # noqa: E402
from explore_persona_space.experiments.factor_screen_365.persona_panel import (  # noqa: E402
    EVAL_PERSONAS_24,
    EVAL_QUESTIONS_20,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1333")

# Captured at IMPORT time, BEFORE any in-process train_lora call can clobber
# the dispatcher's env (the gotchas.md +gpu_id clobber family).
_INITIAL_CVD = os.environ.get("CUDA_VISIBLE_DEVICES")

MARKER_ACCEL_CONFIG = "configs/accelerate/zero3_4gpu_accum16.yaml"
MARKER_FT_TRAINER = "scripts/issue1112_train_marker_fullft.py"
FT_NUM_PROCESSES = 4
VLLM_GPU_MEM_UTIL = 0.5  # HF + vLLM co-residency headroom (#685)
VLLM_MAX_MODEL_LEN = 8192  # prompt + R(2048) re-entered as prompt (#601 rule)
VLLM_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))


# ── Config ────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Cfg:
    smoke: bool
    cells: tuple[str, ...]
    out_root: Path
    seed: int = C.SEED
    eval_question_limit: int | None = None  # smoke: 2
    sentinel_dir: Path | None = None
    upload: bool = True
    ext_captures: bool = False  # optional extension-cell geometry captures (§9 descope 1)
    phases: tuple[str, ...] = ()  # empty -> all

    def regime_key(self) -> dict:
        return {
            "issue": C.ISSUE,
            "smoke": self.smoke,
            "cells": list(self.cells),
            "seed": self.seed,
            "eval_question_limit": self.eval_question_limit,
            "window": list(C.ACCEPT_WINDOW),
            "lora_max_steps": C.LORA_MAX_STEPS,
            "save_steps": {c: C.save_steps_for(c) for c in C.NEW_LORA_CELLS},
            "warmup_steps": C.LORA_WARMUP_STEPS,
            "ft_grid": list(C.FT_GRID),
        }


_PHASE_ALIASES = {
    "p0_stage": "stage",
    "p1_mixes": "mixes",
    "p2_train": "train",
    "p3_ladder": "ladder",
    "p4_select": "select",
    "p5_capture": "capture",
    "p6_tf_shared": "tf_shared",
    "p7_breadth": "breadth",
    "p8_upload": "upload",
}
_KNOWN_PHASES = frozenset(_PHASE_ALIASES.values())


def normalize_phases(raw: str | None) -> tuple[str, ...]:
    """Comma list of phase names -> canonical short-name tuple (fail-loud)."""
    if not raw:
        return ()
    out: list[str] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        t = _PHASE_ALIASES.get(t, t)
        if t not in _KNOWN_PHASES:
            raise ValueError(
                f"unknown phase {tok.strip()!r}: want one of {sorted(_KNOWN_PHASES)} "
                "(pN_-prefixed aliases accepted)"
            )
        out.append(t)
    return tuple(out)


# LoRA path + FT/ZeRO-3+vLLM canary + the reused arm (so the plan's SECOND
# HALT gate — the apply-and-read gate in phase_stage — actually fires in smoke;
# review r1 M3). The reused cell trains nothing: every train/ladder/select/
# capture/breadth filter excludes it; only p0's gate + p6's tf_shared see it.
SMOKE_CELLS = (C.CELL_LORA_CON, C.CELL_FT_POS, C.REUSED_CELL)


def resolve_cells(cells_arg: str | None, smoke: bool) -> tuple[str, ...]:
    """The ONE cell resolver every phase consumes (smoke = same path, 2 cells)."""
    if cells_arg:
        ids = tuple(t.strip() for t in cells_arg.split(","))
        bad = [t for t in ids if t not in C.ALL_TRAINED_CELLS]
        if bad:
            raise ValueError(f"bad cells {bad!r}: want a subset of {C.ALL_TRAINED_CELLS}")
        return ids
    if smoke:
        return SMOKE_CELLS
    return C.ALL_TRAINED_CELLS


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _phase(name: str) -> None:
    logger.info("[phase=%s]", name)


def _physical_gpu_ids() -> list[str]:
    """Physical GPU ids for subprocess CVD pins — clobber-immune (nvidia-smi
    subprocess; honors a LAUNCHER-set CVD captured at import)."""
    if _INITIAL_CVD is not None and _INITIAL_CVD.strip():
        return [t.strip() for t in _INITIAL_CVD.split(",") if t.strip()]
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    ids = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not ids:
        raise RuntimeError("no GPUs visible via nvidia-smi")
    return ids


def _run_subprocess(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[subprocess] %s (log %s)", " ".join(cmd[:8]) + " ...", log_path)
    with open(log_path, "a") as f:
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ} if env is None else env
        )
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess rc={proc.returncode}: {' '.join(cmd)} (log {log_path})")


def _fanout_units(cfg: Cfg, units: list[list[str]]) -> None:
    """Work-conserving CVD-pinned subprocess pool over self-invocation units
    (1-GPU units only; the FT train is whole-pod and never routes here)."""
    ids = _physical_gpu_ids()
    pending = list(units)
    running: dict[int, tuple[subprocess.Popen, list[str]]] = {}
    logs = cfg.out_root / "unit_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for g in range(len(ids)):
            if g not in running and pending:
                extra = pending.pop(0)
                cmd = [
                    "uv",
                    "run",
                    "python",
                    str(_SCRIPTS_DIR / "issue1333_dispatch.py"),
                    *extra,
                    "--gpu-id",
                    ids[g],
                ]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ids[g]}
                log = logs / f"unit_{'_'.join(extra[1:3]).replace('/', '_')}_g{g}.log"
                f = open(log, "a")  # noqa: SIM115 — held open for the Popen's lifetime
                running[g] = (
                    subprocess.Popen(
                        cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True
                    ),
                    extra,
                )
                logger.info("[fanout] gpu %d <- %s (log %s)", g, extra, log)
        time.sleep(10)
        for g, (proc, extra) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[g]
            if rc != 0:
                d1112._reap_unit_groups([p2 for p2, _ in running.values()])
                raise RuntimeError(f"fanout unit {extra} failed rc={rc} (see {logs})")


def _mode_flag(cfg: Cfg) -> str:
    return "--smoke" if cfg.smoke else "--full"


def _unit_args(cfg: Cfg, kind: str, arg: str) -> list[str]:
    out = [
        _mode_flag(cfg),
        "--unit",
        kind,
        arg,
        "--cells",
        ",".join(cfg.cells),
        "--out-root",
        str(cfg.out_root),
        "--seed",
        str(cfg.seed),
    ]
    if cfg.eval_question_limit is not None:
        out += ["--eval-question-limit", str(cfg.eval_question_limit)]
    return out


# ── Contexts / questions ──────────────────────────────────────────────────────

TRAIN_QUESTIONS = list(EVAL_QUESTIONS_20[:10])  # #508 load_q_train: first half


def _eval_questions(cfg: Cfg) -> list[str]:
    qs = list(EVAL_QUESTIONS_20)
    if cfg.eval_question_limit is not None:
        qs = qs[: cfg.eval_question_limit]
    return qs


def _bare_context() -> Context:
    """The deployment-default (no explicit system prompt) source context —
    renders BYTE-IDENTICAL to qwen_default under apply_chat_template
    (assumption 16; the reason for cell 7's panel substitution)."""
    return Context(
        context_id="bare_default",
        kind="bare",
        family="deployment_default",
        source="no system prompt — apply_chat_template auto-inserts the Qwen default",
    )


def _persona_context(key: str) -> Context:
    return Context(
        context_id=f"persona_{key}",
        kind="persona",
        family="house_persona",
        system=EVAL_PERSONAS_24[key],
        source="factor_screen_365 EVAL_PERSONAS_24 (frozen-mix persona bank)",
    )


def resolve_source_context(cfg: Cfg, cell: str) -> Context:
    """The cell's SOURCE training/eval context (plan §4.1)."""
    ctx_id = C.CELL_SOURCE_CONTEXT[cell]
    if ctx_id == "persona_villain":
        return _persona_context("villain")
    if ctx_id == "bare_default":
        return _bare_context()
    if ctx_id == "wildchat_prefix_real545":
        from issue1090_fu3_cells import register_fu3_contexts

        register_fu3_contexts()
        return CONTEXTS[ctx_id]
    if ctx_id == "icl_prefix_marker":
        bank_dir = cfg.out_root / "inputs"
        return icl_prefix_context("marker", bank_dir=bank_dir)
    raise ValueError(f"unknown source context {ctx_id!r} for cell {cell!r}")


def _panel_context(name: str) -> Context:
    """A negative-panel / held-out persona as a Context (french_person routes
    through personas.PERSONAS — assumption 17; the rest through the frozen-mix
    persona bank so training rows match the frozen mix byte-for-byte)."""
    if name == "french_person":
        return context_for_persona("french_person")
    return _persona_context(name)


def breadth_panel(cfg: Cfg, cell: str, tokenizer) -> dict[str, Context]:
    """The DE-DUPLICATED 8-rendered-context breadth panel for one cell
    (plan §4.5): union {source, 4 trained negatives, default assistant,
    held-out trio}, de-duplicated at RENDERED-TOKEN level with labels."""
    source = resolve_source_context(cfg, cell)
    members: dict[str, Context] = {"__source__": source}
    for neg in C.CELL_NEGATIVES[cell] or C.FROZEN_NEGATIVES:
        members[neg] = _panel_context(neg)
    members["default_assistant"] = _bare_context()
    for p in C.HELD_OUT_TRIO:
        members[p] = _panel_context(p)
    seen: dict[tuple[int, ...], str] = {}
    out: dict[str, Context] = {}
    for label, ctx in members.items():
        seq = C.rendered_ids(tokenizer, ctx.messages, "__dedup_probe__")
        if seq in seen:
            logger.info(
                "[breadth] %s: %r renders identical to %r — de-duplicated", cell, label, seen[seq]
            )
            continue
        seen[seq] = label
        out[label] = ctx
    return out


# ── vLLM generation helpers (chunked; #628 spawn guard at module top) ────────


def _vllm_engine(model_path: str, *, enable_lora: bool = False):
    from vllm import LLM

    kwargs: dict = {}
    if enable_lora:
        kwargs.update(enable_lora=True, max_lora_rank=16, max_loras=1)
    return LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
        max_model_len=VLLM_MAX_MODEL_LEN,
        **kwargs,
    )


def _reap_engine(llm) -> None:
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)


def _greedy(llm, prompts: list[str], max_new: int, *, lora_request=None) -> list[str]:
    """Chunked greedy generation (gotchas.md large-batch deadlock prevention)."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    out: list[str] = []
    n_chunks = (len(prompts) + VLLM_CHUNK - 1) // VLLM_CHUNK
    for i in range(0, len(prompts), VLLM_CHUNK):
        chunk = prompts[i : i + VLLM_CHUNK]
        logger.info(
            "[vllm-chunk] greedy chunk %d/%d (%d prompts)",
            i // VLLM_CHUNK + 1,
            n_chunks,
            len(chunk),
        )
        kw = {"lora_request": lora_request} if lora_request is not None else {}
        chunk_out = llm.generate(chunk, sp, use_tqdm=False, **kw)
        out.extend(o.outputs[0].text for o in chunk_out)
    return out


def _strip_at_marker(text: str) -> tuple[str, bool]:
    """Strip-at-first-emission (the #532 slot rule). Returns (text, emitted)."""
    cut = text.find("※")
    if cut >= 0:
        return text[:cut].rstrip(), True
    return text, False


# ── HF slot reads (four-float storage; three-space contract) ─────────────────


def _load_hf(model_path: str, device: str = "cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    return model


def _tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    ids = tok.encode(C.MARKER_TEXT, add_special_tokens=False)
    assert ids == [C.MARKER_TOKEN_ID], f"marker id assert failed: {ids}"  # in-process assert
    return tok


def _slot_read(model, tokenizer, contexts: list[str], device: str = "cuda:0") -> list[dict]:
    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats

    return compute_marker_slot_stats(
        model,
        tokenizer,
        contexts,
        C.MARKER_TEXT,
        device=device,
        eos_token_id=C.EOS_TOKEN_ID,
        include_argmax=True,
    )


def _delta_record(contexts_meta: list[dict], trained: list[dict], base: list[dict]) -> dict:
    deltas = [t["logp"] - b["logp"] for t, b in zip(trained, base, strict=True)]
    emit = [1.0 if t.get("argmax_id") == C.MARKER_TOKEN_ID else 0.0 for t in trained]
    return {
        "n_probes": len(deltas),
        "delta_logp_mean": float(sum(deltas) / len(deltas)),
        "source_emission_rate": float(sum(emit) / len(emit)),
        "per_probe": [
            {"row": m, "trained": t, "base": b}
            for m, t, b in zip(contexts_meta, trained, base, strict=True)
        ],
    }


# ── p0: stage + reused-arm apply-and-read gate ───────────────────────────────


def _stage_file(path_in_repo: str, dest: Path, *, revision: str, sha256: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        got = hf_hub_download(C.HF_DATA_REPO, path_in_repo, repo_type="dataset", revision=revision)
        shutil.copyfile(got, dest)
    if sha256 is not None:
        actual = _sha256_file(dest)
        if actual != sha256:
            raise ValueError(f"sha256 mismatch for {path_in_repo}: {actual} != pinned {sha256}")
    return dest


def phase_stage(cfg: Cfg) -> dict:
    _phase("p0_stage")
    inputs = cfg.out_root / "inputs"
    rev = C.PARENT_CAPTURE_REV
    staged = {
        "frozen_mix": str(
            _stage_file(C.FROZEN_MIX_PATH, inputs / "marker_contrastive.jsonl", revision=rev)
        ),
        "base_raw_rows": str(
            _stage_file(C.BASE_RAW_ROWS_PATH, inputs / "tf_base_rows_marker.json", revision=rev)
        ),
        "r_train": str(
            _stage_file(C.R_TRAIN_PATH, inputs / "R_train.json", revision=C.R_TRAIN_REV)
        ),
        "m2_selection": str(
            _stage_file(C.M2_SELECTION_PATH, inputs / "m2_selection.json", revision=rev)
        ),
    }
    # Pooled stores are geometry-driver inputs; staged here so the row_meta
    # asserts run BEFORE any production phase (plan §4.6 item (c)).
    for name, path_in_repo in (
        ("m2_pooled", C.M2_POOLED_PATH),
        ("base_pooled", C.BASE_POOLED_PATH),
    ):
        dest = (
            cfg.out_root
            / "capture"
            / ("m2_fullft_band8/selected" if name == "m2_pooled" else "base_marker/base")
        )
        staged[name] = str(_stage_file(path_in_repo, dest / "pooled.pt", revision=rev))
    _assert_reused_row_meta(Path(staged["m2_pooled"]))
    _assert_reused_row_meta(Path(staged["base_pooled"]))
    # frozen-mix decomposition re-verified at stage (plan §4.6)
    rows = C._read_jsonl(Path(staged["frozen_mix"]))
    pos, _neg = C.partition_frozen_mix(rows)
    # R_train cross-check (review r1 m11): the staged artifact must cover the
    # villain source over TRAIN_QUESTIONS, and the frozen-mix positives must
    # ride exactly TRAIN_QUESTIONS (the parent's #508 construction — mixes.py
    # samples positives from q_train = EVAL_QUESTIONS_20[:10] with full
    # coverage at 200 draws). This makes the staged R_train a consumed input.
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        load_r_artifact,
    )
    from explore_persona_space.experiments.issue_1112.mixes import MARKER_SOURCE_PERSONA

    r_train = load_r_artifact(Path(staged["r_train"]))
    missing_q = [q for q in TRAIN_QUESTIONS if q not in r_train.get(MARKER_SOURCE_PERSONA, {})]
    if missing_q:
        raise ValueError(
            f"staged R_train lacks {MARKER_SOURCE_PERSONA!r} coverage for "
            f"{len(missing_q)}/{len(TRAIN_QUESTIONS)} TRAIN_QUESTIONS"
        )
    pos_qs = {r["prompt"][-1]["content"] for r in pos}
    if pos_qs != set(TRAIN_QUESTIONS):
        raise ValueError(
            "frozen-mix positive questions != TRAIN_QUESTIONS "
            f"(missing={sorted(set(TRAIN_QUESTIONS) - pos_qs)[:2]!r}, "
            f"extra={sorted(pos_qs - set(TRAIN_QUESTIONS))[:2]!r})"
        )
    rec = {"staged": staged, "frozen_mix_sha256": _sha256_file(Path(staged["frozen_mix"]))}
    # Reused-arm checkpoint + apply-and-read gate (HALT; plan §4.6) — needs a
    # GPU; smoke runs it at eval_question_limit with a doubled bound (noted).
    if C.REUSED_CELL in cfg.cells:
        rec["apply_gate"] = _reused_arm_apply_gate(cfg)
    _atomic_json(cfg.out_root / "stage_result.json", rec)
    return rec


def _assert_reused_row_meta(pooled_path: Path) -> None:
    """Registered 100-row panel assert on a reused pooled store (plan §3
    Row-coverage): row_meta == 5 contexts x 20 questions, mmap-cheap."""
    import torch

    store = torch.load(pooled_path, map_location="cpu", mmap=True, weights_only=False)
    meta = store["row_meta"]
    ctxs = sorted({m["context_id"] for m in meta})
    qs = sorted({int(m["question_idx"]) for m in meta})
    grid = {(m["context_id"], int(m["question_idx"])) for m in meta}
    assert len(ctxs) == 5, ctxs
    assert qs == list(range(20)), qs
    assert len(grid) == len(meta) == 100, (len(grid), len(meta))
    assert set(store["arms"]) == {"prefix", "context", "response"}, sorted(store["arms"])


def _stage_reused_ckpt(cfg: Cfg) -> Path:
    dest = cfg.out_root / "inputs" / "tf_ckpts" / C.REUSED_CELL
    staged = d1112._stage_overflow_prefix(C.M2_CKPT_PREFIX, dest, revision=C.M2_CKPT_REV)
    if not d1112._weights_complete(staged):
        shutil.rmtree(staged, ignore_errors=True)
        staged = d1112._stage_overflow_prefix(C.M2_CKPT_PREFIX, dest, revision=C.M2_CKPT_REV)
        if not d1112._weights_complete(staged):
            raise RuntimeError(f"staged reused checkpoint incomplete: {staged}")
    d1112._ensure_dir_tokenizer(staged)
    return staged


def _reused_arm_apply_gate(cfg: Cfg) -> dict:
    """Apply-and-read HALT gate (plan §4.6): re-read checkpoint-4's
    eval-surface ΔG; HALT >2 nat off the committed +6.28486385345459
    (same-surface committed reference: m2_fullft_band8_slotstats.json
    grid_delta_g["4"]); 1-2 nat -> WARN + persist + analyzer adjudication.
    Smoke runs at eval_question_limit probes with a DOUBLED HALT bound
    (2-probe means are noisier; logged as smoke-scale)."""
    out_dir = cfg.out_root / "gates" / "reused_apply"
    done = out_dir / "apply_gate.json"
    if done.exists():
        return _read_json(done)
    staged = _stage_reused_ckpt(cfg)
    questions = _eval_questions(cfg)
    tok = _tokenizer()
    src = _persona_context("villain")
    prompts = [src.render(tok, q) for q in questions]
    llm = _vllm_engine(str(staged))
    try:
        responses = _greedy(llm, prompts, C.MARKER_MAX_NEW_TOKENS)
    finally:
        _reap_engine(llm)
    contexts, meta = [], []
    for q_idx, (p, r) in enumerate(zip(prompts, responses, strict=True)):
        stripped, emitted = _strip_at_marker(r)
        contexts.append(p + stripped)
        meta.append({"q": q_idx, "gen_emitted": emitted})
    _persist_rollouts(cfg, "selection", C.REUSED_CELL, {"prompts": prompts, "responses": responses})
    model = _load_hf(str(staged))
    try:
        trained = _slot_read(model, tok, contexts)
    finally:
        _free_hf(model)
    base = _load_hf(DEFAULT_BASE_MODEL)
    try:
        base_stats = _slot_read(base, tok, contexts)
    finally:
        _free_hf(base)
    rec = _delta_record(meta, trained, base_stats)
    gap = abs(rec["delta_logp_mean"] - C.TARGET_DELTA_G)
    halt_bound = C.APPLY_GATE_HALT_NATS * (2.0 if cfg.smoke else 1.0)
    rec.update(
        {
            "target": C.TARGET_DELTA_G,
            "gap_nats": gap,
            "halt_bound": halt_bound,
            "verdict": "PASS"
            if gap <= C.APPLY_GATE_WARN_NATS
            else ("WARN" if gap <= halt_bound else "HALT"),
            "smoke_scale": cfg.smoke,
        }
    )
    _atomic_json(done, rec)
    if rec["verdict"] == "HALT":
        raise RuntimeError(
            f"reused-arm apply-and-read gate HALT: ΔG {rec['delta_logp_mean']:.3f} vs "
            f"committed {C.TARGET_DELTA_G:.3f} (gap {gap:.3f} > {halt_bound}) — mis-applied "
            "checkpoint (plan §4.6 kill criterion)"
        )
    if rec["verdict"] == "WARN":
        logger.warning("[apply-gate] WARN gap %.3f nat — persisted for analyzer adjudication", gap)
    return rec


def _free_hf(model) -> None:
    import gc

    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def _persist_rollouts(cfg: Cfg, stage: str, cell: str, payload: dict) -> None:
    """Persist rollout TEXT the moment generation completes (upload policy)."""
    out = cfg.out_root / "raw_completions" / stage / f"{cell}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False))


# ── p1: mixes (+ ICL bank fill + rendered-token disjointness asserts) ────────


def _mix_path(cfg: Cfg, cell: str) -> Path:
    mix = C.CELL_MIX[cell]
    if mix == "marker_contrastive":
        return cfg.out_root / "inputs" / "marker_contrastive.jsonl"
    return cfg.out_root / "mixes" / C.MIX_FILENAMES[mix]


def _icl_demo_answer(a: str) -> str:
    """Demo answer = greedy base R + the TRAINED leading-space marker — the
    same construction as mix positives (plan §4.1 cell 6). NEVER
    ``MARKER_TEXT.strip()``: the bare glyph is token 63680, not the trained
    83399 (review r1 C2; the #395/#396 wrong-token class)."""
    return f"{a.strip()}{C.MARKER_SEP}{C.MARKER_TEXT}"


def _assert_icl_demo_tails_encode_marker(tok, examples: list[dict]) -> None:
    """Fill-time mirror of C.assert_positive_tails_encode_marker for the ICL
    bank: every filled demo answer's tail must encode marker id 83399."""
    for i, ex in enumerate(examples):
        ids = tok.encode(ex["answer"][-16:], add_special_tokens=False)
        if C.MARKER_TOKEN_ID not in ids:
            raise ValueError(f"ICL demo {i}: tail ids {ids} lack marker id {C.MARKER_TOKEN_ID}")


def _fill_icl_bank(cfg: Cfg, tok) -> Path:
    """Fill the committed questions-only ICL spec with greedy base answers +
    marker (plan §4.1 cell 6) -> run inputs icl_examples_marker.json."""
    dest = cfg.out_root / "inputs" / "icl_examples_marker.json"
    if dest.exists():
        return dest
    spec_path = REPO_ROOT / "src/explore_persona_space/artifacts/query_banks" / C.ICL_QUESTIONS_SPEC
    spec = json.loads(spec_path.read_text())
    demo_qs = [ex["question"] for ex in spec["examples"]]
    assert len(demo_qs) == 2, demo_qs
    # Demo-question disjointness (plan §4.1 cell 6 / assumption 12).
    for q in demo_qs:
        if q in TRAIN_QUESTIONS or q in list(EVAL_QUESTIONS_20):
            raise ValueError(f"ICL demo question collides with train/eval banks: {q!r}")
    bare = _bare_context()
    llm = _vllm_engine(DEFAULT_BASE_MODEL)
    try:
        answers = _greedy(llm, [bare.render(tok, q) for q in demo_qs], C.R_GEN_MAX_NEW_TOKENS)
    finally:
        _reap_engine(llm)
    examples = []
    for q, a in zip(demo_qs, answers, strict=True):
        if "※" in a:
            raise ValueError(f"greedy demo answer for {q!r} already carries the marker")
        examples.append({"question": q, "answer": _icl_demo_answer(a)})
    _assert_icl_demo_tails_encode_marker(tok, examples)
    bank = {
        **spec,
        "examples": examples,
        "filled_by": "issue1333_dispatch p1_mixes (greedy base R)",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    _persist_rollouts(cfg, "mixes", "icl_bank", bank)
    return dest


def _greedy_r_map(cfg: Cfg, tok, ctx: Context, questions: list[str], label: str) -> dict[str, str]:
    """Greedy base R per question under one context (extension-mix positives /
    cell-7 french negatives). Truncation rate logged (plan §4.2)."""
    cache = cfg.out_root / "mixes" / f"r_{label}.json"
    if cache.exists():
        return _read_json(cache)["r_by_q"]
    prompts = [ctx.render(tok, q) for q in questions]
    llm = _vllm_engine(DEFAULT_BASE_MODEL)
    try:
        responses = _greedy(llm, prompts, C.R_GEN_MAX_NEW_TOKENS)
    finally:
        _reap_engine(llm)
    n_trunc = sum(
        1
        for r in responses
        if len(tok.encode(r, add_special_tokens=False)) >= C.R_GEN_MAX_NEW_TOKENS
    )
    logger.info("[mixes] %s: %d/%d generations at the 1024 cap", label, n_trunc, len(responses))
    rec = {
        "context": ctx.context_id,
        "truncation_rate": n_trunc / len(responses),
        "r_by_q": dict(zip(questions, responses, strict=True)),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(cache, rec)
    _persist_rollouts(cfg, "mixes", label, rec)
    return rec["r_by_q"]


def _cell_disjointness_assert(cfg: Cfg, tok, cell: str) -> None:
    """Plan §4.2 RENDERED-TOKEN disjointness invariant, run at every mix build
    (also verifies french_person ∩ {sources, breadth eval contexts} = ∅)."""
    panel_names = C.CELL_NEGATIVES[cell]
    if not panel_names:
        return  # positives-only cells train no panel
    source = resolve_source_context(cfg, cell)
    C.assert_rendered_disjoint(
        tok,
        source_id=source.context_id,
        source_msgs_for_q=source.messages,
        panel={n: _panel_context(n).messages for n in panel_names},
        questions=TRAIN_QUESTIONS if not cfg.smoke else TRAIN_QUESTIONS[:2],
    )


def phase_mixes(cfg: Cfg) -> dict:
    _phase("p1_mixes")
    tok = _tokenizer()
    frozen = _mix_path(cfg, C.CELL_LORA_CON)
    manifests: dict[str, dict] = {}
    needed = {C.CELL_MIX[c] for c in cfg.cells}
    frozen_rows = C._read_jsonl(frozen)
    _pos, neg = C.partition_frozen_mix(frozen_rows)
    frozen_negs = C.negatives_by_persona(neg, {p: EVAL_PERSONAS_24[p] for p in C.FROZEN_NEGATIVES})

    # Disjointness asserts run for EVERY cell in scope (incl. frozen-mix cells).
    for cell in cfg.cells:
        if cell == C.REUSED_CELL:
            continue
        if cell == C.CELL_EXT_ICL:
            _fill_icl_bank(cfg, tok)  # bank must exist before the context resolves
        _cell_disjointness_assert(cfg, tok, cell)

    if "marker_posonly" in needed:
        out = cfg.out_root / "mixes" / C.MIX_FILENAMES["marker_posonly"]
        if not out.exists():
            manifests["marker_posonly"] = C.derive_posonly_mix(frozen, out, tokenizer=tok)
        else:
            manifests["marker_posonly"] = _read_json(out.with_suffix(".manifest.json"))

    for cell, mix in (
        (C.CELL_EXT_WILDCHAT, "marker_wildchat"),
        (C.CELL_EXT_ICL, "marker_icl"),
        (C.CELL_EXT_BARE, "marker_bare"),
    ):
        if mix not in needed:
            continue
        out = cfg.out_root / "mixes" / C.MIX_FILENAMES[mix]
        if out.exists():
            manifests[mix] = _read_json(out.with_suffix(".manifest.json"))
            continue
        source = resolve_source_context(cfg, cell)
        r_by_q = _greedy_r_map(cfg, tok, source, TRAIN_QUESTIONS, f"{cell}_pos")
        kwargs: dict = {}
        if cell == C.CELL_EXT_BARE:
            fr_ctx = _panel_context("french_person")
            fr_r = _greedy_r_map(cfg, tok, fr_ctx, TRAIN_QUESTIONS, f"{cell}_french")
            kwargs = {"french_r_for_q": fr_r.__getitem__, "french_system": fr_ctx.system}
        manifests[mix] = C.build_extension_mix(
            cell,
            source_msgs_for_q=source.messages,
            greedy_r_for_q=r_by_q.__getitem__,
            train_questions=TRAIN_QUESTIONS,
            frozen_negatives=frozen_negs,
            out_path=out,
            seed=cfg.seed,
            **kwargs,
        )

    # Upload mixes BEFORE training (plan §4.2 sha-pinned + uploaded).
    if cfg.upload:
        for mix, _man in manifests.items():
            fname = C.MIX_FILENAMES[mix]
            p = cfg.out_root / "mixes" / fname
            # kwarg shape mirrors issue1112_dispatch (review r1 C1: the path
            # string must bind to path_in_repo, never the repo_id positional).
            hub._upload(
                p,
                C.HF_DATA_REPO,
                "dataset",
                f"{C.DATA_PREFIX}/mixes/{fname}",
                upload_as_file=True,
            )
            hub._upload(
                p.with_suffix(".manifest.json"),
                C.HF_DATA_REPO,
                "dataset",
                f"{C.DATA_PREFIX}/mixes/{fname.replace('.jsonl', '.manifest.json')}",
                upload_as_file=True,
            )
    return manifests


# ── p2: train (LoRA fanout + FT ZeRO-3 subprocess) ───────────────────────────


def run_train_unit(cfg: Cfg, cell: str) -> dict:
    """One LoRA ladder train (single GPU; launcher CVD pin authoritative via
    _apply_cvd_pin — gpu_id stays 0 under a single-GPU pin)."""
    from explore_persona_space.train.sft import train_lora

    cell_root = cfg.out_root / cell
    done = cell_root / "build_result.json"
    if done.exists():
        return _read_json(done)
    tok = _tokenizer()
    train_cfg = C.marker_lora_config(cell, seed=cfg.seed, tokenizer=tok, out_root=cfg.out_root)
    if cfg.smoke:
        # dense in-loop probes so the smoke shows >=1 trajectory point (the
        # plan §4.3 smoke-verifiable telemetry rule) within the 2-step train.
        train_cfg = dataclasses.replace(
            train_cfg,
            max_steps=2,
            save_steps=1,
            marker_band_eval_every_steps=1,
            marker_band_dense_until=2,
            marker_band_min_steps=2,
        )
    adapter_dir, loss = train_lora(
        DEFAULT_BASE_MODEL, str(_mix_path(cfg, cell)), str(cell_root / "train"), cfg=train_cfg
    )
    release_trainer_cuda_memory()
    rec = {
        "adapter_root": str(adapter_dir),
        "training_loss": float(loss),
        "trajectory": str(cfg.out_root / cell / "band_trajectory.json"),
    }
    _atomic_json(done, rec)
    if cfg.smoke and cell == SMOKE_CELLS[0]:
        rec["parity_gate"] = _same_surface_parity_gate(cfg, cell)
        rec["swap_merge_parity"] = _swap_vs_merged_parity(cfg, cell)
        _atomic_json(done, rec)
    return rec


def _trajectory_last_delta(traj_path: Path) -> float:
    """Last in-loop probe's trained−base delta from the band-stop trajectory
    JSON (records carry the four-float set for trained AND base per probe)."""
    traj = _read_json(traj_path)
    recs = traj.get("records") or []
    if not recs:
        raise RuntimeError(f"no trajectory records in {traj_path} — telemetry did not fire")
    last = recs[-1]
    for t_key, b_key in (("logp", "base_logp"), ("log_p_marker", "base_log_p_marker")):
        if t_key in last and b_key in last:
            return float(last[t_key]) - float(last[b_key])
    if "delta_nats" in last:
        return float(last["delta_nats"])
    trained = last.get("trained") or {}
    base = last.get("base") or {}
    if "logp" in trained and "logp" in base:
        return float(trained["logp"]) - float(base["logp"])
    raise RuntimeError(f"unrecognized trajectory record schema: {sorted(last)}")


def _frozen_r_contexts(cfg: Cfg, cell: str, tok, n: int) -> list[str]:
    """The in-loop callback's OWN surface: teacher-forced marker slots on the
    TRAIN mix's frozen villain-positive rows (same-surface parity, #813)."""
    rows = C._read_jsonl(_mix_path(cfg, cell))
    pos = [r for r in rows if C._row_is_positive(r)][:n]
    out = []
    for r in pos:
        prompt = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        completion = r["completion"][0]["content"]
        assert completion.endswith(C.MARKER_TEXT), "positive row must end with the marker"
        out.append(prompt + completion[: -len(C.MARKER_TEXT)])
    return out


def _same_surface_parity_gate(cfg: Cfg, cell: str) -> dict:
    """Smoke HALT gate (plan §4.3): the off-line rig re-reads the smoke cell in
    FROZEN-R mode and must reproduce the in-loop callback's last logged delta
    within ~1 nat (the #534 adapter-application assert, same-surface)."""
    from peft import PeftModel

    cell_root = cfg.out_root / cell
    build = _read_json(cell_root / "build_result.json")
    in_loop = _trajectory_last_delta(Path(build["trajectory"]))
    tok = _tokenizer()
    contexts = _frozen_r_contexts(cfg, cell, tok, n=4)
    adapter = _final_adapter_dir(build["adapter_root"])
    base = _load_hf(DEFAULT_BASE_MODEL)
    try:
        peft_model = PeftModel.from_pretrained(base, str(adapter))
        trained = _slot_read(peft_model, tok, contexts)
        base_model = peft_model.unload()
        base_stats = _slot_read(base_model, tok, contexts)
    finally:
        _free_hf(base)
    delta = sum(t["logp"] - b["logp"] for t, b in zip(trained, base_stats, strict=True)) / len(
        trained
    )
    gap = abs(delta - in_loop)
    rec = {"in_loop_delta": in_loop, "offline_frozen_r_delta": float(delta), "gap_nats": float(gap)}
    _atomic_json(cell_root / "parity_gate.json", rec)
    if gap > C.PARITY_GATE_NATS:
        raise RuntimeError(
            f"same-surface parity gate HALT: off-line frozen-R ΔG {delta:.3f} vs in-loop "
            f"{in_loop:.3f} (gap {gap:.3f} > {C.PARITY_GATE_NATS}) — adapter-application bug "
            "(#534 checklist) before any production spend"
        )
    return rec


def _final_adapter_dir(adapter_root: str) -> Path:
    root = Path(adapter_root)
    if (root / "adapter_config.json").exists():
        return root
    rungs = d1112._enumerate_rungs(root)
    return rungs[max(rungs)]


def _swap_vs_merged_parity(cfg: Cfg, cell: str) -> dict:
    """Assumption 7 smoke gate: PEFT adapter-swap slot read == merged-model
    read within 0.1 nat on the smoke cell's final rung."""
    from peft import PeftModel

    cell_root = cfg.out_root / cell
    build = _read_json(cell_root / "build_result.json")
    adapter = _final_adapter_dir(build["adapter_root"])
    tok = _tokenizer()
    contexts = _frozen_r_contexts(cfg, cell, tok, n=2)
    base = _load_hf(DEFAULT_BASE_MODEL)
    try:
        peft_model = PeftModel.from_pretrained(base, str(adapter))
        swap = _slot_read(peft_model, tok, contexts)
        peft_model.unload()
    finally:
        _free_hf(base)
    merged = d1112._merge_adapter(None, str(adapter), cell_root / "merged_parity")
    try:
        m = _load_hf(str(merged))
        try:
            merged_stats = _slot_read(m, tok, contexts)
        finally:
            _free_hf(m)
    finally:
        shutil.rmtree(merged, ignore_errors=True)
    gap = max(abs(s["logp"] - t["logp"]) for s, t in zip(swap, merged_stats, strict=True))
    rec = {"max_abs_gap_nats": float(gap), "bound": C.SWAP_MERGE_PARITY_NATS}
    _atomic_json(cell_root / "swap_merge_parity.json", rec)
    if gap > C.SWAP_MERGE_PARITY_NATS:
        raise RuntimeError(
            f"PEFT-swap vs merged parity HALT: max |Δlogp| {gap:.4f} > "
            f"{C.SWAP_MERGE_PARITY_NATS} (assumption 7)"
        )
    return rec


def _ft_num_processes(cfg: Cfg) -> int:
    if cfg.smoke:
        return 1
    n_phys = len(_physical_gpu_ids())
    if n_phys < FT_NUM_PROCESSES:
        raise RuntimeError(f"full-FT needs {FT_NUM_PROCESSES} GPUs, {n_phys} visible")
    return FT_NUM_PROCESSES


def phase_train(cfg: Cfg) -> dict:
    _phase("p2_train")
    out: dict[str, dict] = {}
    lora_cells = [c for c in cfg.cells if c in C.NEW_LORA_CELLS]
    pending = [c for c in lora_cells if not (cfg.out_root / c / "build_result.json").exists()]
    if pending:
        # Work-conserving fanout, 1 GPU per LoRA cell (plan §9 item i). The
        # smoke's single cell rides the same path (1 unit).
        _fanout_units(cfg, [_unit_args(cfg, "train", c) for c in pending])
    for c in lora_cells:
        out[c] = _read_json(cfg.out_root / c / "build_result.json")

    if C.CELL_FT_POS in cfg.cells:
        ft_root = cfg.out_root / C.CELL_FT_POS / "train"
        done = cfg.out_root / C.CELL_FT_POS / "build_result.json"
        if not done.exists():
            grid = (1,) if cfg.smoke else C.FT_GRID
            if ft_root.exists():
                logger.warning("[ft] clearing stale partial FT out_dir %s", ft_root)
                shutil.rmtree(ft_root)
            cmd = C.marker_ft_cmd(
                mix_path=_mix_path(cfg, C.CELL_FT_POS),
                out_dir=ft_root,
                num_processes=_ft_num_processes(cfg),
                seed=cfg.seed,
                grid=grid,
                max_steps=max(grid),
                trainer=MARKER_FT_TRAINER,
                accel_config=MARKER_ACCEL_CONFIG,
            )
            ids = _physical_gpu_ids()
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(ids[: _ft_num_processes(cfg)])}
            _run_subprocess(cmd, cfg.out_root / "logs" / "ft_mk4.log", env=env)
            _atomic_json(done, {"adapter_root": str(ft_root), "grid": list(grid)})
        out[C.CELL_FT_POS] = _read_json(done)
    return out


# ── p3: base priors + off-line ladders ───────────────────────────────────────


def basepriors_context_panel(cfg: Cfg, tok) -> dict[str, Context]:
    """Distinct rendered contexts for the p3 base-prior read, keyed by
    ``context_id`` (review r1 M4: the r1 label-keyed merge let each breadth
    cell's ``"__source__"`` overwrite the previous cell's source, silently
    dropping villain/wildchat/ICL priors in full mode). Fail-loud coverage:
    every non-reused cell's SOURCE render must survive into the panel
    (possibly under a rendered-identical alias, e.g. bare == qwen_default)."""
    contexts: dict[str, Context] = {}
    for cell in cfg.cells:
        if cell == C.REUSED_CELL:
            continue
        if cell in C.BREADTH_CELLS:
            for ctx in breadth_panel(cfg, cell, tok).values():
                contexts[ctx.context_id] = ctx
        else:
            src = resolve_source_context(cfg, cell)
            contexts.setdefault(src.context_id, src)
    # de-dup across cells at rendered level
    seen: dict[tuple[int, ...], str] = {}
    distinct: dict[str, Context] = {}
    for _cid, ctx in contexts.items():
        seq = C.rendered_ids(tok, ctx.messages, "__dedup_probe__")
        if seq in seen:
            continue
        seen[seq] = ctx.context_id
        distinct[ctx.context_id] = ctx
    for cell in cfg.cells:
        if cell == C.REUSED_CELL:
            continue
        src = resolve_source_context(cfg, cell)
        seq = C.rendered_ids(tok, src.messages, "__dedup_probe__")
        if seq not in seen:
            raise RuntimeError(
                f"base-priors panel missing source context {src.context_id!r} for {cell!r}"
            )
    return distinct


def run_basepriors_unit(cfg: Cfg, _arg: str) -> dict:
    """Base-side slot reads per DISTINCT rendered context (plan §4.5: run
    BEFORE selection reads consume them — the per-context ΔG ceiling)."""
    done = cfg.out_root / "base_priors" / "base_priors.json"
    if done.exists():
        return _read_json(done)
    tok = _tokenizer()
    distinct = basepriors_context_panel(cfg, tok)
    questions = _eval_questions(cfg)
    prompts, meta = [], []
    for cid, ctx in distinct.items():
        for q_idx, q in enumerate(questions):
            prompts.append(ctx.render(tok, q))
            meta.append({"context_id": cid, "q": q_idx})
    llm = _vllm_engine(DEFAULT_BASE_MODEL)
    try:
        responses = _greedy(llm, prompts, C.MARKER_MAX_NEW_TOKENS)
    finally:
        _reap_engine(llm)
    _persist_rollouts(cfg, "base_priors", "base", {"meta": meta, "responses": responses})
    slot_contexts = [p + _strip_at_marker(r)[0] for p, r in zip(prompts, responses, strict=True)]
    base = _load_hf(DEFAULT_BASE_MODEL)
    try:
        stats = _slot_read(base, tok, slot_contexts)
    finally:
        _free_hf(base)
    by_ctx: dict[str, list[float]] = {}
    for m, s in zip(meta, stats, strict=True):
        by_ctx.setdefault(m["context_id"], []).append(s["logp"])
    rec = {
        "per_context_prior": {k: float(sum(v) / len(v)) for k, v in by_ctx.items()},
        "per_context_ceiling": {k: float(-sum(v) / len(v)) for k, v in by_ctx.items()},
        "per_probe": [{"meta": m, "base": s} for m, s in zip(meta, stats, strict=True)],
    }
    _atomic_json(done, rec)
    return rec


def _cell_rungs(cfg: Cfg, cell: str) -> dict[int, Path]:
    build = _read_json(cfg.out_root / cell / "build_result.json")
    return d1112._enumerate_rungs(build["adapter_root"])


def _ladder_read_steps(
    cfg: Cfg, cell: str, rungs: dict[int, Path], ladder: dict[int, dict]
) -> list[int]:
    if cfg.smoke:
        return [s for s in sorted(rungs) if s not in ladder][:1]
    coarse = [s for s in C.coarse_read_steps(cell, sorted(rungs)) if s not in ladder]
    if coarse:
        return coarse
    return C.refine_read_steps(cell, sorted(rungs), ladder)


def run_ladder_unit(cfg: Cfg, cell: str) -> dict[int, dict]:
    """Off-line eval-surface three-space ladder for one cell (plan §4.3).

    LoRA cells: ONE shared enable_lora engine + per-rung LoRARequest (the
    #1090 shared-engine pattern, no per-rung merges) + PEFT-swap HF slot reads.
    FT cells: per-rung engine on the consolidated dir (reap between rungs) +
    direct HF loads; keep-best-2-plus-latest rung retention (plan §9 c33).
    """
    cell_root = cfg.out_root / cell
    ladder_path = cell_root / "ladder.json"
    ladder: dict[int, dict] = {}
    if ladder_path.exists():
        prior = _read_json(ladder_path)
        if prior.get("regime") != cfg.regime_key():
            raise RuntimeError(f"ladder regime drift under {ladder_path} — fresh --out-root")
        ladder = {int(k): v for k, v in (prior.get("reads_by_step") or {}).items()}

    def _persist() -> None:
        _atomic_json(
            ladder_path,
            {
                "cell": cell,
                "regime": cfg.regime_key(),
                "reads_by_step": {str(k): v for k, v in sorted(ladder.items())},
            },
        )

    rungs = _cell_rungs(cfg, cell)
    tok = _tokenizer()
    src = resolve_source_context(cfg, cell)
    questions = _eval_questions(cfg)
    prompts = [src.render(tok, q) for q in questions]

    while True:
        pending = _ladder_read_steps(cfg, cell, rungs, ladder)
        if not pending:
            break
        if cell in C.NEW_LORA_CELLS:
            _ladder_reads_lora(cfg, cell, rungs, pending, ladder, tok, prompts, _persist)
        else:
            _ladder_reads_ft(cfg, cell, rungs, pending, ladder, tok, prompts, _persist)
    _persist()
    return ladder


def _ladder_reads_lora(cfg, cell, rungs, pending, ladder, tok, prompts, persist) -> None:
    from peft import PeftModel
    from vllm.lora.request import LoRARequest

    llm = _vllm_engine(DEFAULT_BASE_MODEL, enable_lora=True)
    base = _load_hf(DEFAULT_BASE_MODEL)
    try:
        for step in pending:
            req = LoRARequest(f"{cell}_rung{step}", (step % 100000) + 1, str(rungs[step]))
            responses = _greedy(llm, prompts, C.MARKER_MAX_NEW_TOKENS, lora_request=req)
            _persist_rollouts(cfg, "ladder", f"{cell}_rung{step}", {"responses": responses})
            contexts, meta = [], []
            for q_idx, (p, r) in enumerate(zip(prompts, responses, strict=True)):
                stripped, emitted = _strip_at_marker(r)
                contexts.append(p + stripped)
                meta.append({"q": q_idx, "gen_emitted": emitted})
            peft_model = PeftModel.from_pretrained(base, str(rungs[step]))
            trained = _slot_read(peft_model, tok, contexts)
            peft_model.unload()
            base_stats = _slot_read(base, tok, contexts)
            rec = _delta_record(meta, trained, base_stats)
            rec["gen_emission_rate"] = float(sum(m["gen_emitted"] for m in meta) / len(meta))
            out_dir = cfg.out_root / cell / "ladder" / f"rung_{step}"
            _atomic_json(out_dir / "slot_read.json", rec)
            ladder[step] = {
                "delta_logp_mean": rec["delta_logp_mean"],
                "source_emission_rate": rec["source_emission_rate"],
                "gen_emission_rate": rec["gen_emission_rate"],
            }
            persist()
    finally:
        _free_hf(base)
        _reap_engine(llm)


def _ladder_reads_ft(cfg, cell, rungs, pending, ladder, tok, prompts, persist) -> None:
    for step in pending:
        ckpt = rungs[step]
        d1112._ensure_dir_tokenizer(ckpt)
        llm = _vllm_engine(str(ckpt))
        try:
            responses = _greedy(llm, prompts, C.MARKER_MAX_NEW_TOKENS)
        finally:
            _reap_engine(llm)
        _persist_rollouts(cfg, "ladder", f"{cell}_rung{step}", {"responses": responses})
        contexts, meta = [], []
        for q_idx, (p, r) in enumerate(zip(prompts, responses, strict=True)):
            stripped, emitted = _strip_at_marker(r)
            contexts.append(p + stripped)
            meta.append({"q": q_idx, "gen_emitted": emitted})
        model = _load_hf(str(ckpt))
        try:
            trained = _slot_read(model, tok, contexts)
        finally:
            _free_hf(model)
        base = _load_hf(DEFAULT_BASE_MODEL)
        try:
            base_stats = _slot_read(base, tok, contexts)
        finally:
            _free_hf(base)
        rec = _delta_record(meta, trained, base_stats)
        rec["gen_emission_rate"] = float(sum(m["gen_emitted"] for m in meta) / len(meta))
        out_dir = cfg.out_root / cell / "ladder" / f"rung_{step}"
        _atomic_json(out_dir / "slot_read.json", rec)
        ladder[step] = {
            "delta_logp_mean": rec["delta_logp_mean"],
            "source_emission_rate": rec["source_emission_rate"],
            "gen_emission_rate": rec["gen_emission_rate"],
        }
        persist()
        _reap_ft_rungs(cfg, cell, rungs, ladder)


def _reap_ft_rungs(cfg: Cfg, cell: str, rungs: dict[int, Path], ladder: dict[int, dict]) -> None:
    """FT rung retention (plan §9 c33): keep the 2 best selection candidates
    among READ rungs + the latest rung; delete the rest (declared discards —
    deterministic retrain from pinned mix + recipe + seed)."""
    if cfg.smoke:
        return
    read = [s for s in ladder if s in rungs]
    by_dist = sorted(read, key=lambda s: abs(ladder[s]["delta_logp_mean"] - C.TARGET_DELTA_G))
    keep = set(by_dist[:2]) | {max(rungs)}
    for s in read:
        if s not in keep and rungs[s].exists():
            logger.info("[ft-retention] deleting non-candidate rung %s (read, declared discard)", s)
            shutil.rmtree(rungs[s], ignore_errors=True)


def phase_ladder(cfg: Cfg) -> dict:
    _phase("p3_ladder")
    units = [_unit_args(cfg, "basepriors", "all")]
    cells = [c for c in cfg.cells if c in (*C.NEW_LORA_CELLS, C.CELL_FT_POS)]
    units += [_unit_args(cfg, "ladder", c) for c in cells]
    _fanout_units(cfg, units)
    return {c: _read_json(cfg.out_root / c / "ladder.json")["reads_by_step"] for c in cells}


# ── p4: selection + bystander de-saturation confirm ──────────────────────────


def _bystander_record(meta: list[dict], trained: list[dict], base_stats: list[dict]) -> dict:
    """Full bystander-battery record (pure; unit-tested): per-probe four-float
    trained+base stats via ``_delta_record`` PLUS per-context Δlog-prob /
    EOS-margin ``Δ(z_marker − z_eos)`` / emission aggregates — the leakage
    reads the plan-§6 (2)/(3) transfer fractions + dose curves consume — and
    the de-saturation gate fields the selection loop reads (plan §4.3)."""
    rec = _delta_record(meta, trained, base_stats)
    by_ctx: dict[str, dict[str, list[float]]] = {}
    for m, t, b in zip(meta, trained, base_stats, strict=True):
        d = by_ctx.setdefault(m["context_id"], {"deltas": [], "margins": [], "emit": []})
        d["deltas"].append(t["logp"] - b["logp"])
        d["margins"].append((t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"]))
        d["emit"].append(1.0 if t.get("argmax_id") == C.MARKER_TOKEN_ID else 0.0)
    rec["per_context"] = {
        k: {
            "delta_logp_mean": float(sum(v["deltas"]) / len(v["deltas"])),
            "delta_margin_mean": float(sum(v["margins"]) / len(v["margins"])),
            "emission_rate": float(sum(v["emit"]) / len(v["emit"])),
        }
        for k, v in by_ctx.items()
    }
    rates = {k: v["emission_rate"] for k, v in rec["per_context"].items()}
    rec["bystander_argmax_rates"] = rates
    rec["saturated"] = any(r >= 0.92 for r in rates.values())
    return rec


def _bystander_battery(
    cfg: Cfg, cell: str, step: int, model_path: str, *, lora_rung: Path | None, llm=None
) -> dict:
    """On-policy bystander reads at one rung: the de-saturation gate at
    candidate rungs (plan §4.3) AND the four-float leakage read the
    leakage-vs-install dose curves consume (plan §6 read (3); concern
    ladder-bystander-dose-curves). ``llm`` optionally carries a SHARED
    enable_lora engine (LoRA rungs only — the #1090 shared-engine pattern);
    FT rungs always build their own per-checkpoint engine."""
    from vllm.lora.request import LoRARequest

    tok = _tokenizer()
    panel = C.CELL_NEGATIVES[cell] or C.FROZEN_NEGATIVES
    questions = _eval_questions(cfg)
    prompts, meta = [], []
    for name in panel:
        ctx = _panel_context(name)
        for q_idx, q in enumerate(questions):
            prompts.append(ctx.render(tok, q))
            meta.append({"context_id": name, "q": q_idx})
    own_engine = llm is None
    if lora_rung is not None:
        if llm is None:
            llm = _vllm_engine(DEFAULT_BASE_MODEL, enable_lora=True)
        req = LoRARequest(f"{cell}_sel{step}", (step % 100000) + 1, str(lora_rung))
    else:
        assert llm is None, "shared engines are LoRA-only (FT rungs are per-checkpoint)"
        llm = _vllm_engine(model_path)
        req = None
    try:
        responses = _greedy(llm, prompts, C.MARKER_MAX_NEW_TOKENS, lora_request=req)
    finally:
        if own_engine:
            _reap_engine(llm)
    _persist_rollouts(
        cfg, "selection", f"{cell}_bystanders_rung{step}", {"meta": meta, "responses": responses}
    )
    contexts = [p + _strip_at_marker(r)[0] for p, r in zip(prompts, responses, strict=True)]
    if lora_rung is not None:
        from peft import PeftModel

        base = _load_hf(DEFAULT_BASE_MODEL)
        try:
            peft_model = PeftModel.from_pretrained(base, str(lora_rung))
            trained = _slot_read(peft_model, tok, contexts)
            base_model = peft_model.unload()
            base_stats = _slot_read(base_model, tok, contexts)
        finally:
            _free_hf(base)
    else:
        model = _load_hf(model_path)
        try:
            trained = _slot_read(model, tok, contexts)
        finally:
            _free_hf(model)
        base = _load_hf(DEFAULT_BASE_MODEL)
        try:
            base_stats = _slot_read(base, tok, contexts)
        finally:
            _free_hf(base)
    rec = _bystander_record(meta, trained, base_stats)
    rec["cell"] = cell
    rec["step"] = step
    return rec


def _read_battery_steps(cell_root: Path) -> set[int]:
    """Rungs already carrying a persisted bystander battery (on-disk truth —
    identical on the fresh path and the selection-exists resume path)."""
    return {
        int(p.parent.name.removeprefix("rung_"))
        for p in (cell_root / "ladder").glob("rung_*/bystanders.json")
    }


def _run_dose_curve_batteries(
    cfg: Cfg, cell: str, rungs: dict[int, Path], ladder: dict[int, dict], candidates: set[int]
) -> list[dict]:
    """Flanking bystander batteries for the plan-§6 read (3) dose curves
    (concern ladder-bystander-dose-curves): candidate rungs + one sub-window
    + one above-window flank per cell (``C.dose_curve_rung_plan``), skipping
    rungs already read. An FT flank whose checkpoint was reaped by the plan
    §9 c33 retention is documented as ``checkpoint_reaped`` — a named
    resolution limit, never a crash. Returns the selection record's
    ``dose_curve_rungs`` schema field ({step, role, delta_logp_mean,
    status})."""
    plan = C.dose_curve_rung_plan(ladder, sorted(candidates))
    is_lora = cell in C.NEW_LORA_CELLS
    out: list[dict] = []
    shared = None
    try:
        for item in plan:
            step = item["step"]
            bys_path = cfg.out_root / cell / "ladder" / f"rung_{step}" / "bystanders.json"
            entry = dict(item)
            if bys_path.exists():
                entry["status"] = "read"
            elif step not in rungs or not rungs[step].exists():
                entry["status"] = "checkpoint_reaped"
                logger.warning(
                    "[dose-curve] %s rung %s: checkpoint reaped (plan §9 c33 retention) — "
                    "bystander read unavailable; documented resolution limit",
                    cell,
                    step,
                )
            else:
                if is_lora and shared is None:
                    shared = _vllm_engine(DEFAULT_BASE_MODEL, enable_lora=True)
                bys = _bystander_battery(
                    cfg,
                    cell,
                    step,
                    str(rungs[step]),
                    lora_rung=rungs[step] if is_lora else None,
                    llm=shared,
                )
                _atomic_json(bys_path, bys)
                entry["status"] = "read"
            out.append(entry)
    finally:
        if shared is not None:
            _reap_engine(shared)
    return out


def run_select_unit(cfg: Cfg, cell: str) -> dict:
    cell_root = cfg.out_root / cell
    sel_path = cell_root / "selection.json"
    sel: dict | None = None
    if sel_path.exists():
        sel = _read_json(sel_path)
        if "dose_curve_rungs" in sel:
            return sel
    ladder = {int(k): v for k, v in _read_json(cell_root / "ladder.json")["reads_by_step"].items()}
    rungs = _cell_rungs(cfg, cell)
    is_lora = cell in C.NEW_LORA_CELLS
    if sel is None:
        tried: set[int] = set()
        while True:
            sel = C.select_rung(ladder)
            step = sel["step"]
            if not sel["in_window"] or step in tried:
                break
            tried.add(step)
            bys = _bystander_battery(
                cfg,
                cell,
                step,
                str(rungs[step]),
                lora_rung=rungs[step] if is_lora else None,
            )
            _atomic_json(cell_root / "ladder" / f"rung_{step}" / "bystanders.json", bys)
            if not bys["saturated"]:
                sel["bystanders"] = bys
                break
            ladder[step]["bystander_saturated"] = True
            _atomic_json(
                cell_root / "ladder.json",
                {
                    "cell": cell,
                    "regime": cfg.regime_key(),
                    "reads_by_step": {str(k): v for k, v in sorted(ladder.items())},
                },
            )
        sel["cell"] = cell
    # Dose-curve flank batteries + the self-describing rung-resolution field
    # (plan §6 read (3); concern ladder-bystander-dose-curves). Candidates =
    # on-disk batteries (the loop above just wrote them) + the selected rung
    # (covers the closest-approach fallback, which the loop never reads).
    candidates = _read_battery_steps(cell_root) | {int(sel["step"])}
    sel["dose_curve_rungs"] = _run_dose_curve_batteries(cfg, cell, rungs, ladder, candidates)
    _atomic_json(sel_path, sel)
    return sel


def phase_select(cfg: Cfg) -> dict:
    _phase("p4_select")
    out: dict[str, dict] = {}
    cells = [c for c in cfg.cells if c in (*C.NEW_LORA_CELLS, C.CELL_FT_POS)]
    for cell in cells:  # sequential: one GPU pass per cell, cheap vs ladder
        out[cell] = run_select_unit(cfg, cell)
        if not out[cell]["in_window"]:
            logger.warning(
                "[select] %s: NO in-window rung — closest-approach %s (dose-reachability "
                "branch, plan §3; labeled, no lattice call)",
                cell,
                out[cell]["step"],
            )
    # FT retention terminal step: upload the selected rung BEFORE deleting the rest.
    if C.CELL_FT_POS in out and cfg.upload and not cfg.smoke:
        step = out[C.CELL_FT_POS]["step"]
        rungs = _cell_rungs(cfg, C.CELL_FT_POS)
        hub._upload(
            rungs[step],
            repo_id=C.OVERFLOW_REPO,
            path_in_repo=f"issue1333/{C.CELL_FT_POS}/checkpoint-{step}",
            repo_type="model",
        )
        for s, p in rungs.items():
            if s not in (step, max(rungs)) and p.exists():
                shutil.rmtree(p, ignore_errors=True)
    # reused arm: selection record is the parent's (staged at p0)
    if C.REUSED_CELL in cfg.cells:
        out[C.REUSED_CELL] = _read_json(cfg.out_root / "inputs" / "m2_selection.json")
    return out


# ── p5/p6: captures ───────────────────────────────────────────────────────────


def _resolve_capture_model(cfg: Cfg, cell: str) -> tuple[str, Path | None]:
    """(model_path, merged_dir_to_cleanup) for one capture pass at the
    SELECTED rung (atomic merge-read-delete for LoRA cells)."""
    if cell == C.REUSED_CELL:
        return str(_stage_reused_ckpt(cfg)), None
    cell_root = cfg.out_root / cell
    step = int(_read_json(cell_root / "selection.json")["step"])
    ckpt = _cell_rungs(cfg, cell)[step]
    if cell == C.CELL_FT_POS:
        d1112._ensure_dir_tokenizer(ckpt)
        return str(ckpt), None
    merged = d1112._merge_adapter(None, str(ckpt), cell_root / "merged_selected")
    return str(merged), merged


def run_capture_unit(cfg: Cfg, cell: str) -> None:
    """Own-text capture: on-policy gen + 28-layer 3-span TF pooling (plan §4.4;
    parameters byte-parity with the parent so stores compose)."""
    import torch

    from explore_persona_space.analysis.representation_shift import (
        _generate_responses_vllm,
        _teacher_forced_span_means,
        compute_prompt_spans,
    )

    if cell not in (C.CELL_LORA_CON, C.CELL_LORA_POS, C.CELL_FT_POS):
        # Review r1 m12: the fixed 2x2 panel below would mis-panel an extension
        # cell — plan §4.1 wants each extension cell's OWN de-duplicated panel.
        # --ext-captures is the §9 descope-rung-1 optional pass (off by
        # default); fail loud rather than capture under the wrong panel.
        raise NotImplementedError(
            f"own-text capture for extension cell {cell!r} needs the cell's own "
            "de-duplicated panel (plan §4.1); wire breadth_panel-derived contexts "
            "before enabling --ext-captures"
        )
    out_dir = cfg.out_root / "capture" / cell / "selected"
    if (out_dir / "pooled.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path, cleanup = _resolve_capture_model(cfg, cell)
    panel = {p: EVAL_PERSONAS_24[p] for p in ("villain", *C.FROZEN_NEGATIVES)}
    questions = list(EVAL_QUESTIONS_20)
    if cfg.smoke:
        panel = dict(list(panel.items())[:2])
        questions = questions[:2]
    rows = _generate_responses_vllm(
        model_path,
        panel,
        questions,
        max_new_tokens=C.MARKER_MAX_NEW_TOKENS,
        gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
    )
    tokenizer = _tokenizer()
    for r in rows:
        r["prefix_len"], r["context_len"] = compute_prompt_spans(
            tokenizer, panel[r["persona"]], questions[r["question_idx"]], r["prompt_token_ids"]
        )
    (out_dir / "raw_rows.json").write_text(
        json.dumps({"model": model_path, "rows": rows}, ensure_ascii=False)
    )
    pooled = _teacher_forced_span_means(
        model_path,
        rows,
        list(panel),
        layers=list(range(C.N_LAYERS)),
        device="cuda:0",
        dtype=torch.bfloat16,
        tf_batch_size=C.TF_BATCH_SIZE,
    )
    _save_pooled(out_dir, cell, "selected", "own_text", model_path, rows, pooled)
    if cleanup is not None:
        shutil.rmtree(cleanup, ignore_errors=True)


def _save_pooled(
    out_dir: Path, cell: str, dose: str, conditioning: str, model_path: str, rows, pooled
) -> None:
    import torch

    store = {
        "schema_version": 1,
        "cell": cell,
        "dose": dose,
        "behavior": "marker",
        "model_path": model_path,
        "row_meta": [
            {"context_id": r["persona"], "question_idx": int(r["question_idx"])} for r in rows
        ],
        "arms": {
            arm: {li: t.to(torch.float16) for li, t in per_layer.items()}
            for arm, per_layer in pooled.items()
        },
        "metadata": {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tf_batch_size": C.TF_BATCH_SIZE,
            "conditioning": conditioning,
            "git_commit": _git_commit_sha(),
        },
    }
    tmp = out_dir / "pooled.pt.tmp"
    torch.save(store, tmp)
    os.replace(tmp, out_dir / "pooled.pt")


def _git_commit_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def phase_capture(cfg: Cfg) -> dict:
    _phase("p5_capture")
    cells = [c for c in cfg.cells if c in (C.CELL_LORA_CON, C.CELL_LORA_POS, C.CELL_FT_POS)]
    if cfg.ext_captures:
        cells += [
            c for c in cfg.cells if c in (C.CELL_EXT_WILDCHAT, C.CELL_EXT_ICL, C.CELL_EXT_BARE)
        ]
    pending = [
        c for c in cells if not (cfg.out_root / "capture" / c / "selected" / "pooled.pt").exists()
    ]
    if pending:
        _fanout_units(cfg, [_unit_args(cfg, "capture", c) for c in pending])
    return {"captured": cells}


def run_tf_unit(cfg: Cfg, cell: str) -> None:
    """Teacher-forced SHARED-text re-capture over the pinned base_marker rows
    (plan §4.4 — mandatory for all four 2x2 cells incl. the reused arm)."""
    import torch

    from explore_persona_space.analysis.representation_shift import _teacher_forced_span_means

    out_dir = cfg.out_root / "capture" / cell / "tf_shared"
    if (out_dir / "pooled.pt").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_json(cfg.out_root / "inputs" / "tf_base_rows_marker.json")["rows"]
    d1112.assert_tf_base_rows(rows, expect_contexts=None, expect_questions=20)
    if cfg.smoke:
        rows = d1112.tf_smoke_rows(rows)
    model_path, cleanup = _resolve_capture_model(cfg, cell)
    panel = list(dict.fromkeys(r["persona"] for r in rows))
    try:
        pooled = _teacher_forced_span_means(
            model_path,
            rows,
            panel,
            layers=list(range(C.N_LAYERS)),
            device="cuda:0",
            dtype=torch.bfloat16,
            tf_batch_size=C.TF_BATCH_SIZE,
        )
        _save_pooled(out_dir, cell, "selected", "tf_shared_base", model_path, rows, pooled)
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)


def phase_tf_shared(cfg: Cfg) -> dict:
    _phase("p6_tf_shared")
    cells = [c for c in cfg.cells if c in C.GEOMETRY_CELLS]
    pending = [
        c for c in cells if not (cfg.out_root / "capture" / c / "tf_shared" / "pooled.pt").exists()
    ]
    if pending:
        _fanout_units(cfg, [_unit_args(cfg, "tf", c) for c in pending])
    return {"tf_shared": cells}


# ── p7: breadth battery ───────────────────────────────────────────────────────


def run_breadth_unit(cfg: Cfg, cell: str) -> dict:
    """Install/expression/leakage battery over the de-duplicated 8-context
    panel at the cell's selected rung (plan §4.5). Four-float storage; the
    EOS-margin transfer fractions are analyzer-side re-reductions."""
    out_path = cfg.out_root / "breadth" / cell / "slot_reads.json"
    if out_path.exists():
        return _read_json(out_path)
    tok = _tokenizer()
    panel = breadth_panel(cfg, cell, tok)
    questions = _eval_questions(cfg)
    if cfg.smoke:
        panel = dict(list(panel.items())[:2])
    model_path, cleanup = _resolve_capture_model(cfg, cell)
    prompts, meta = [], []
    for label, ctx in panel.items():
        for q_idx, q in enumerate(questions):
            prompts.append(ctx.render(tok, q))
            meta.append({"label": label, "context_id": ctx.context_id, "q": q_idx})
    llm = _vllm_engine(model_path)
    try:
        responses = _greedy(llm, prompts, C.MARKER_MAX_NEW_TOKENS)
    finally:
        _reap_engine(llm)
    _persist_rollouts(cfg, "breadth", cell, {"meta": meta, "responses": responses})
    contexts = []
    for m, (p, r) in zip(meta, zip(prompts, responses, strict=True), strict=True):
        stripped, emitted = _strip_at_marker(r)
        contexts.append(p + stripped)
        m["gen_emitted"] = emitted
    model = _load_hf(model_path)
    try:
        trained = _slot_read(model, tok, contexts)
    finally:
        _free_hf(model)
    base = _load_hf(DEFAULT_BASE_MODEL)
    try:
        base_stats = _slot_read(base, tok, contexts)
    finally:
        _free_hf(base)
    if cleanup is not None:
        shutil.rmtree(cleanup, ignore_errors=True)
    rec = _delta_record(meta, trained, base_stats)
    by_ctx: dict[str, dict] = {}
    for m, t, b in zip(meta, trained, base_stats, strict=True):
        d = by_ctx.setdefault(
            m["label"], {"context_id": m["context_id"], "deltas": [], "emit": [], "base_logp": []}
        )
        d["deltas"].append(t["logp"] - b["logp"])
        d["emit"].append(1.0 if t.get("argmax_id") == C.MARKER_TOKEN_ID else 0.0)
        d["base_logp"].append(b["logp"])
    rec["per_context"] = {
        k: {
            "context_id": v["context_id"],
            "delta_logp_mean": float(sum(v["deltas"]) / len(v["deltas"])),
            "emission_rate": float(sum(v["emit"]) / len(v["emit"])),
            "base_prior_mean": float(sum(v["base_logp"]) / len(v["base_logp"])),
        }
        for k, v in by_ctx.items()
    }
    rec["cell"] = cell
    rec["panel_labels"] = list(panel)
    _atomic_json(out_path, rec)
    return rec


def phase_breadth(cfg: Cfg) -> dict:
    _phase("p7_breadth")
    cells = [c for c in cfg.cells if c in C.BREADTH_CELLS]
    pending = [c for c in cells if not (cfg.out_root / "breadth" / c / "slot_reads.json").exists()]
    if pending:
        _fanout_units(cfg, [_unit_args(cfg, "breadth", c) for c in pending])
    return {
        c: _read_json(cfg.out_root / "breadth" / c / "slot_reads.json")["per_context"]
        for c in cells
    }


# ── p8: upload + sentinel ────────────────────────────────────────────────────


def phase_upload(cfg: Cfg) -> dict:
    _phase("p8_upload")
    if not cfg.upload:
        return {"skipped": "--no-upload"}
    uploaded: dict[str, str] = {}

    def _up(local: Path, path_in_repo: str, *, as_file: bool = True) -> None:
        # kwarg-complete shape (review r1 C1): local, repo_id, repo_type,
        # path_in_repo — the r1 form bound the path string to repo_id.
        hub._upload(
            local,
            C.HF_DATA_REPO,
            "dataset",
            f"{C.DATA_PREFIX}/{path_in_repo}",
            upload_as_file=as_file,
        )
        uploaded[path_in_repo] = str(local)

    root = cfg.out_root
    # text/JSON: unconditional (raw completions, ladders, selections, gates, priors)
    rc_root = root / "raw_completions"
    if rc_root.exists():
        hub._upload(rc_root, C.HF_DATA_REPO, "dataset", f"{C.DATA_PREFIX}/raw_completions")
        uploaded["raw_completions/"] = str(rc_root)
    for pattern, dest in (
        ("*/ladder/rung_*/slot_read.json", "selection"),
        ("*/ladder/rung_*/bystanders.json", "selection"),
        ("*/ladder.json", "selection"),
        ("*/selection.json", "selection"),
        ("*/parity_gate.json", "gates"),
        ("*/swap_merge_parity.json", "gates"),
        ("*/band_trajectory.json", "trajectories"),
        ("gates/reused_apply/apply_gate.json", "gates"),
        ("base_priors/base_priors.json", "base_priors"),
        ("breadth/*/slot_reads.json", "breadth"),
    ):
        for f in sorted(root.glob(pattern)):
            rel = f.relative_to(root)
            _up(f, f"{dest}/{rel}")
    # pooled capture tensors (fp16, plain torch.save — Xet-bound, no
    # compression) -> analysis_tensors/capture/... (plan §10 layout; r1 m9)
    for f in (
        sorted((root / "capture").glob("*/*/pooled.pt")) if (root / "capture").exists() else []
    ):
        rel = f.relative_to(root)
        _up(f, f"analysis_tensors/{rel}")
    for f in (
        sorted((root / "capture").glob("*/selected/raw_rows.json"))
        if (root / "capture").exists()
        else []
    ):
        rel = f.relative_to(root / "capture")  # r1 m9: no doubled capture/capture segment
        _up(f, f"raw_completions/capture/{rel}")
    # selected LoRA adapters -> overflow repo (LFS)
    adapters: dict[str, str] = {}
    if not cfg.smoke:
        for cell in cfg.cells:
            if cell not in C.NEW_LORA_CELLS:
                continue
            sel = root / cell / "selection.json"
            if not sel.exists():
                continue
            step = int(_read_json(sel)["step"])
            ckpt = _cell_rungs(cfg, cell)[step]
            path_in_repo = f"issue1333/{cell}/checkpoint-{step}"
            hub._upload(ckpt, repo_id=C.OVERFLOW_REPO, path_in_repo=path_in_repo, repo_type="model")
            adapters[cell] = f"{C.OVERFLOW_REPO}/{path_in_repo}"
    rec = {"uploaded_n": len(uploaded), "adapters": adapters}
    _atomic_json(root / "upload_result.json", rec)
    return rec


def write_sentinel(cfg: Cfg, summary: dict) -> Path:
    _phase("sentinel")
    sentinel_dir = cfg.sentinel_dir or Path("/workspace/logs")
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    kind = "epm:smoke-result" if cfg.smoke else "epm:results"
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,  # VM-side drain re-derives max+1
        "task_id": C.ISSUE,
        "by": "issue1333_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": summary,
    }
    path = sentinel_dir / f"issue-{C.ISSUE}-{kind.replace(':', '_')}-{int(time.time())}.json"
    _atomic_json(path, payload)
    logger.info("[sentinel] wrote %s", path)
    return path


def _repro_card(cfg: Cfg, selections: dict) -> dict:
    return {
        "hf_model_repo": C.OVERFLOW_REPO,
        "adapter_paths": {
            c: f"issue1333/{c}/checkpoint-{v['step']}"
            for c, v in selections.items()
            if c in (*C.NEW_LORA_CELLS, C.CELL_FT_POS) and v.get("step") is not None
        },
        "wandb_project": C.WANDB_PROJECT,
        "wandb_run_names": [C.cell_run_name(c) for c in cfg.cells if c != C.REUSED_CELL],
        "wandb_entity": _wandb_entity(),
        "git_commit": _git_commit_sha(),
    }


def _wandb_entity() -> str:
    try:
        import wandb

        return str(wandb.Api().default_entity)
    except Exception as e:
        logger.warning("[repro-card] wandb entity unresolved: %s", e)
        return "unknown"


# ── main ─────────────────────────────────────────────────────────────────────


def _check_regime(cfg: Cfg) -> None:
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    p = cfg.out_root / "run_config.json"
    cur = cfg.regime_key()
    if p.exists():
        prior = _read_json(p)
        prior_rest = {k: v for k, v in prior.items() if k != "cells"}
        cur_rest = {k: v for k, v in cur.items() if k != "cells"}
        if prior_rest != cur_rest or not set(cur["cells"]) <= set(prior.get("cells", [])):
            raise RuntimeError(f"out_root {cfg.out_root} holds a run under a DIFFERENT regime")
    else:
        _atomic_json(p, cur)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1333 pod-side phase driver")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny-real, SAME code path")
    mode.add_argument("--full", action="store_true")
    p.add_argument(
        "--unit",
        nargs=2,
        default=None,
        metavar=("KIND", "ARG"),
        help="internal fanout unit (train|ladder|capture|tf|breadth|basepriors <cell|all>)",
    )
    p.add_argument("--gpu-id", type=str, default="0", help="physical GPU (CVD-pinned by launcher)")
    p.add_argument("--cells", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    p.add_argument("--ext-captures", action="store_true", help="optional extension-cell captures")
    p.add_argument("--phases", default=None, help="comma subset of phases (default all)")
    return p.parse_args(argv)


def build_cfg(args: argparse.Namespace) -> Cfg:
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else (f"/tmp/issue-{C.ISSUE}-smoke" if smoke else f"data/issue_{C.ISSUE}/run")
    )
    return Cfg(
        smoke=smoke,
        cells=resolve_cells(args.cells, smoke),
        out_root=out_root,
        seed=args.seed,
        eval_question_limit=(
            args.eval_question_limit
            if args.eval_question_limit is not None
            else (2 if smoke else None)
        ),
        sentinel_dir=(
            Path(args.sentinel_dir)
            if args.sentinel_dir is not None
            # None -> write_sentinel's /workspace/logs (the poller-drained
            # namespace) — incl. for a pod-side smoke, so the epm:smoke-result
            # sentinel actually drains (review r1 m10). A local smoke (no
            # /workspace) keeps its sentinel under out_root.
            else (out_root / "logs" if smoke and not Path("/workspace").is_dir() else None)
        ),
        upload=args.upload,
        ext_captures=bool(args.ext_captures),
        phases=normalize_phases(args.phases),
    )


_UNIT_FNS = {
    "train": run_train_unit,
    "ladder": run_ladder_unit,
    "capture": run_capture_unit,
    "tf": run_tf_unit,
    "breadth": run_breadth_unit,
    "basepriors": run_basepriors_unit,
    "select": run_select_unit,
}


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    cfg = build_cfg(args)
    if args.unit is not None:
        kind, arg = args.unit
        fn = _UNIT_FNS.get(kind)
        if fn is None:
            raise ValueError(f"unknown unit kind {kind!r}: want one of {sorted(_UNIT_FNS)}")
        fn(cfg, arg)
        return 0
    _check_regime(cfg)
    logger.info("issue1333 smoke=%s cells=%s out_root=%s", cfg.smoke, cfg.cells, cfg.out_root)

    def want(phase: str) -> bool:
        return not cfg.phases or phase in cfg.phases

    summary: dict = {"issue": C.ISSUE, "smoke": cfg.smoke, "cells": list(cfg.cells)}
    if want("stage"):
        stage = phase_stage(cfg)
        summary["stage"] = {"apply_gate": stage.get("apply_gate", {}).get("verdict")}
    if want("mixes"):
        summary["mixes"] = {
            k: {"sha256": v.get("sha256"), "n": v.get("n_total", v.get("n_rows"))}
            for k, v in phase_mixes(cfg).items()
        }
    if want("train"):
        train = phase_train(cfg)
        summary["train"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "per_probe"} for k, v in train.items()
        }
    if want("ladder"):
        summary["ladder"] = phase_ladder(cfg)
    selections: dict = {}
    if want("select"):
        selections = phase_select(cfg)
        summary["selections"] = {
            k: {kk: vv for kk, vv in v.items() if kk not in ("per_probe", "bystanders")}
            for k, v in selections.items()
        }
    if want("capture"):
        summary["capture"] = phase_capture(cfg)
    if want("tf_shared"):
        summary["tf_shared"] = phase_tf_shared(cfg)
    if want("breadth"):
        summary["breadth"] = phase_breadth(cfg)
    if want("upload"):
        summary["upload"] = phase_upload(cfg)
    summary["reproducibility_card"] = _repro_card(cfg, selections)
    write_sentinel(cfg, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
