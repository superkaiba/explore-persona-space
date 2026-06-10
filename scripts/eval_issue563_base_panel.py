#!/usr/bin/env python3
"""Issue #563 base-own persona panel — 5 cells x 4-float marker slot stats, NO adapter.

Strip-down of the pinned #558 eval rig (branch issue-558 @
18959f7fca41b3e71d3e1cf128c7cbf50433aad2, scripts/eval_issue558_panel.py); plan
tasks/.../563/plans/plan.md section 3.2 enumerates the strip exhaustively. The
single manipulated variable vs #558: the base-side slot read is taken at the
end of the BASE model's OWN greedy completions (no adapter loaded anywhere),
instead of at the end of the fine-tuned models' completions.

Strips (vs the pinned parent): resolve_adapter, assert_adapter_gauge_free (no
adapter; nothing touches W_U so the logit readout needs no gauge check —
plan section 5 measurement table), LoRARequest / enable_lora, the dual-side
disable_adapter() read (single side), check_doctor_anchor /
parent_doctor_reference (replaced by the instrument-equivalence anchor),
--arm/--seed/--anchor-gate args.

Keeps (verbatim semantics): marker_preflight() first, cell construction with
the same asserts (identical question list across cells; trigger key present in
every user turn), ONE fresh vLLM engine, greedy SamplingParams, checkpoint-
per-cell persists, _teardown_vllm(), the vLLM-free slot-stats SUBPROCESS
(vLLM monkey-patches transformers in-process; HF loads after engine teardown
are unreliable — parent rig pattern), marker-stripping of the completion tail
before the slot read, per-row finiteness assert, summarize_cell three-space
rollup adapted to single side, repro_metadata(), [phase=...] log lines +
write_sentinel_563.

New vs parent:
  - Instrument-equivalence anchor (plan section 3.2 / divergence 4): BEFORE
    any panel generation, re-read #558's committed assistant-cell completions
    (eval_results/issue_558/r50/seed42/phase2/completions_trigger50.json, 50
    rows) with THIS run's plain-base reader and compare per-row logp to the
    committed slot_stats_trigger50.json["base"]. PASS iff mean |offset| <=
    0.05 AND max |offset| <= 0.20 nats. HARD-RAISE on breach: zero panel GPU
    spend after a failed anchor.
  - Questions fetched at the PINNED Hub revision (plan section 13 item 8) +
    sha256 content assert — enforces parity with the parent's pool at run
    time, not just by len()==250.
  - n=250 questions/cell (divergence 2; [0:50] = parent-parity subset).
  - --cells subset re-run support (plan section 13 item 2): a kill-criterion-4
    re-run of a truncating cell at --max-new-tokens 3072 MUST regenerate the
    trigger50 denominator cell in the SAME fresh engine session; trigger50 is
    asserted present in every --cells list.

Usage (pod, 1 GPU):
    uv run python scripts/eval_issue563_base_panel.py --gpu 0
Smoke (same flow, n=20/cell, anchor included, upload skipped):
    uv run python scripts/eval_issue563_base_panel.py --gpu 0 --smoke
VM CPU dry run (no vLLM / CUDA import):
    uv run python scripts/eval_issue563_base_panel.py --dry-run-cells
Anchor only (subprocess + gate, then exit):
    uv run python scripts/eval_issue563_base_panel.py --gpu 0 --anchor-only
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="eval_issue563_base_panel")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    BASE_MODEL,
    DEFAULT_ASSISTANT_KEY,
    EOS_TOKEN_ID,
    EVAL_MAX_NEW_TOKENS,
    HUB_DATA_REPO,
    HUB_EVAL_QUESTIONS_PATH,
    MARKER_TEXT,
    N_EVAL_QUESTIONS,
    N_SMOKE_PROMPTS,
    PROJECT_ROOT,
    all_persona_prompts,
    marker_preflight,
    phase_log,
    repro_metadata,
    sentinel_dir,
    trigger_user,
    truncated,
)

log = logging.getLogger("eval_issue563_base_panel")

ISSUE_563 = 563
LOGPROB_BATCH_SIZE = 8

# Panel constants (plan sections 3.2 / Reproducibility Card).
N_PANEL_PROMPTS_563 = 250  # divergence 2; [0:50] = parent-parity subset
EVAL_RESULTS_DIR_563 = PROJECT_ROOT / "eval_results" / "issue_563"
OUT_DIR_563 = EVAL_RESULTS_DIR_563 / "base"  # single model -> flat layout
HUB_RAW_BUCKET_563 = "issue563_base_panel/raw_completions"
LOCAL_QUESTIONS_PATH_563 = PROJECT_ROOT / "data" / "issue563_base_panel" / "eval_questions.json"

# Instrument-equivalence anchor (plan section 3.2 / section 11 item 8).
ANCHOR_DIR = PROJECT_ROOT / "eval_results" / "issue_558" / "r50" / "seed42" / "phase2"
ANCHOR_COMPLETIONS = ANCHOR_DIR / "completions_trigger50.json"
ANCHOR_REFERENCE = ANCHOR_DIR / "slot_stats_trigger50.json"
ANCHOR_N_ROWS = 50
ANCHOR_TOL_MEAN_NATS = 0.05
ANCHOR_TOL_MAX_NATS = 0.20

# Questions parity (plan section 13 item 8): pinned Hub revision + content hash.
HUB_QUESTIONS_REVISION = "ef37c3ecf71bc2ece3f3aed970fe3cd65c456f86"
EXPECTED_QUESTIONS_SHA256 = "0b320cbae8022c746317ac0c534491e57db7c58749ffb1eae2d8fc5a39d4ff30"

# b-hat sanity range for the assistant-cell base mean log P (kill criterion 3;
# Source: #543 b-hat diagnosis, measured -25.88 — BHAT_SANITY_RANGE in the
# pinned common module).
BHAT_SANITY_RANGE_563 = (-30.0, -15.0)

# Truncation kill-criterion 4 bound (plan section 7).
TRUNCATION_RATE_KILL = 0.20

# Cell slug -> persona key. COPIED VERBATIM from the pinned
# eval_issue558_panel.py @ 18959f7fca41b3e71d3e1cf128c7cbf50433aad2 (PANEL_CELLS);
# this script deliberately does NOT import the 776-line parent script.
PANEL_CELLS: dict[str, str] = {
    "trigger50": "assistant",  # within-run no-persona baseline (paired denominator)
    "doctor": "medical_doctor",  # smallest parent rise (+0.505)
    "software_engineer": "software_engineer",  # other small parent rise (+0.484)
    "french_person": "french_person",  # largest parent rise (+1.438); bonus read
    "police_officer": "police_officer",  # parent rise +1.156; never trained in the chain
}


def panel_persona_prompts() -> dict[str, str]:
    """Parent persona map (assistant + 3 trained negatives) + police_officer.

    Copied from the pinned eval_issue558_panel.py (same personas.py module the
    parent's cells read their exact system-prompt strings from).
    """
    from explore_persona_space.personas import PERSONAS

    prompts = all_persona_prompts()
    prompts["police_officer"] = PERSONAS["police_officer"]
    return prompts


# ── Questions (pinned revision + sha256; plan section 13 item 8) ─────────────


def ensure_eval_questions_local_pinned() -> list[str]:
    """Fetch + cache the 250 held-out questions at the PINNED Hub revision.

    Asserts BOTH the row count (250) and the sha256 of the file content
    against the parent's pinned content — a latest-revision fetch with only a
    len()==250 check would not enforce the parity claim at run time.
    """
    if not LOCAL_QUESTIONS_PATH_563.exists():
        from huggingface_hub import hf_hub_download

        log.info(
            "Fetching %s @ %s from %s",
            HUB_EVAL_QUESTIONS_PATH,
            HUB_QUESTIONS_REVISION,
            HUB_DATA_REPO,
        )
        got = hf_hub_download(
            repo_id=HUB_DATA_REPO,
            filename=HUB_EVAL_QUESTIONS_PATH,
            repo_type="dataset",
            revision=HUB_QUESTIONS_REVISION,
            token=os.environ.get("HF_TOKEN"),
        )
        LOCAL_QUESTIONS_PATH_563.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_QUESTIONS_PATH_563.write_bytes(Path(got).read_bytes())
    data = LOCAL_QUESTIONS_PATH_563.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_QUESTIONS_SHA256:
        raise RuntimeError(
            f"eval_questions.json sha256 {digest} != pinned {EXPECTED_QUESTIONS_SHA256} "
            f"(Hub revision {HUB_QUESTIONS_REVISION}); question-pool parity with #558 broken."
        )
    qs = json.loads(data)
    if len(qs) != N_EVAL_QUESTIONS:
        raise RuntimeError(
            f"eval_questions.json has {len(qs)} entries; expected {N_EVAL_QUESTIONS}."
        )
    return qs


# ── Cells (deterministic slice [0:n]; verbatim parent shape) ─────────────────


def build_cells_panel(
    eval_questions: list[str], *, smoke: bool, cell_names: list[str] | None = None
) -> dict[str, list[dict]]:
    """Panel cells, all on eval_questions[0:n] with the trigger key present.

    n = 250 (N_PANEL_PROMPTS_563); smoke mode uses 20 (N_SMOKE_PROMPTS, parity
    with the parent). Every cell shares the identical question slice so all
    paired contrasts are same-question by construction. ``cell_names`` subsets
    PANEL_CELLS for kill-criterion-4 re-runs; ``trigger50`` (the paired
    denominator) must always be present (plan section 13 item 2).
    """
    personas = panel_persona_prompts()
    n = N_SMOKE_PROMPTS if smoke else N_PANEL_PROMPTS_563
    qs = eval_questions[:n]
    if len(qs) != n:
        raise RuntimeError(f"Need {n} eval questions; got {len(qs)}")

    names = list(PANEL_CELLS) if cell_names is None else cell_names
    unknown = [c for c in names if c not in PANEL_CELLS]
    if unknown:
        raise RuntimeError(f"Unknown cell(s) {unknown}; valid: {list(PANEL_CELLS)}")
    if "trigger50" not in names:
        raise RuntimeError(
            "--cells must include trigger50: every persona contrast pairs against the "
            "assistant denominator generated in the SAME engine session (plan section 13 item 2)."
        )

    cells: dict[str, list[dict]] = {}
    for cell_name in names:
        persona_key = PANEL_CELLS[cell_name]
        system = (
            personas[DEFAULT_ASSISTANT_KEY] if persona_key == "assistant" else personas[persona_key]
        )
        cells[cell_name] = [
            {"system": system, "user": trigger_user(q), "persona_key": persona_key, "trigger": True}
            for q in qs
        ]
    empty = [k for k, v in cells.items() if not v]
    if empty:
        raise RuntimeError(f"Empty cell(s) (smoke={smoke}): {empty}")
    return cells


# ── vLLM generation (ONE fresh engine, no LoRA; parent minus adapter) ───────


def _teardown_vllm(llm: Any) -> None:
    """Reap vLLM worker subprocesses (gotchas.md vLLM teardown; verbatim parent)."""
    import contextlib
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
    with contextlib.suppress(Exception):
        del llm
    gc.collect()
    torch.cuda.empty_cache()
    me = psutil.Process()
    for child in me.children(recursive=True):
        try:
            child.terminate()
            child.wait(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                child.kill()


def generate_completions_base(
    *,
    cells: dict[str, list[dict]],
    out_dir: Path,
    max_new_tokens: int,
) -> dict[str, list[dict]]:
    """Greedy vLLM generation per cell on ONE fresh engine, plain base model.

    Parent generate_completions minus LoRA. Per-record fields as parent (minus
    adapter_path/lora_id). Checkpoint-per-phase: each cell's records are
    persisted the moment the cell finishes.
    """
    from vllm import LLM, SamplingParams

    log.info("Loading FRESH vLLM engine: base=%s (no adapter)", BASE_MODEL)
    llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=64,
        trust_remote_code=True,
        # 0.70 inherited from the parent rig verbatim (its comment documents
        # the 2026-06-10 smoke-cell incident); parity beats marginal speed.
        gpu_memory_utilization=0.70,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, n=1)

    out: dict[str, list[dict]] = {}
    try:
        for cell_name, items in cells.items():
            prefixes = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": it["system"]},
                        {"role": "user", "content": it["user"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for it in items
            ]
            log.info("Generating cell=%s n=%d", cell_name, len(prefixes))
            responses = llm.generate(prefixes, sampling)
            recs: list[dict] = []
            for it, prefix, resp in zip(items, prefixes, responses, strict=True):
                g = resp.outputs[0]
                recs.append(
                    {
                        **it,
                        "prefix": prefix,
                        "completion_text": g.text,
                        "n_generated_tokens": len(g.token_ids),
                        "truncated": truncated(len(g.token_ids), max_new_tokens),
                        "contains_marker": MARKER_TEXT in g.text,
                        "ends_with_marker": g.text.rstrip().endswith(MARKER_TEXT.strip()),
                        "max_new_tokens": max_new_tokens,
                    }
                )
            out[cell_name] = recs
            # Checkpoint-per-phase: persist the cell the moment it completes.
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"completions_{cell_name}.json").write_text(json.dumps(recs, indent=2))
            log.info("Cell %s persisted (%d records).", cell_name, len(recs))
    finally:
        _teardown_vllm(llm)
    return out


# ── Slot-stats subprocess (vLLM-free; 4 floats, single base-own side) ────────


def run_slot_stats_subprocess(*, manifest_path: Path, log_path: Path) -> None:
    """Spawn the vLLM-free worker (vLLM monkey-patches transformers in-process;
    HF loads after an engine teardown are unreliable — parent rig pattern)."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--slot-stats-worker",
        "--manifest",
        str(manifest_path),
    ]
    log.info("Spawning slot-stats subprocess (manifest=%s log=%s)", manifest_path, log_path)
    env = {**os.environ}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = ""
        try:
            with log_path.open("rb") as f:
                f.seek(max(0, log_path.stat().st_size - 4096))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            pass
        raise RuntimeError(
            f"Slot-stats subprocess failed (rc={proc.returncode}); log tail:\n{tail}"
        )


def _anchor_contexts_from_committed() -> list[str]:
    """Slot-read contexts from #558's committed assistant-cell completions.

    Same prefix + marker-stripped completion construction the panel read uses
    (and the parent's worker used) — the anchor compares THIS reader against
    the parent's disable_adapter() base side on byte-identical contexts.
    """
    if not ANCHOR_COMPLETIONS.exists():
        raise FileNotFoundError(
            f"Anchor completions missing: {ANCHOR_COMPLETIONS} (committed in git; sync the repo)"
        )
    records = json.loads(ANCHOR_COMPLETIONS.read_text())
    if len(records) != ANCHOR_N_ROWS:
        raise RuntimeError(
            f"Anchor completions: {len(records)} rows, expected {ANCHOR_N_ROWS} "
            f"({ANCHOR_COMPLETIONS})"
        )
    contexts: list[str] = []
    for r in records:
        comp_clean = r["completion_text"].rstrip()
        if comp_clean.endswith(MARKER_TEXT.strip()):
            comp_clean = comp_clean[: -len(MARKER_TEXT.strip())].rstrip()
        contexts.append(r["prefix"] + comp_clean)
    return contexts


def _anchor_reference_rows() -> list[dict]:
    """The parent's committed base-side per-row 4-float stats for the anchor cell."""
    if not ANCHOR_REFERENCE.exists():
        raise FileNotFoundError(
            f"Anchor reference missing: {ANCHOR_REFERENCE} (committed in git; sync the repo)"
        )
    slot = json.loads(ANCHOR_REFERENCE.read_text())
    rows = slot["base"]
    if slot["n"] != ANCHOR_N_ROWS or len(rows) != ANCHOR_N_ROWS:
        raise RuntimeError(
            f"Anchor reference: n={slot['n']}, len(base)={len(rows)}; expected {ANCHOR_N_ROWS}"
        )
    return rows


def compare_anchor_rows(
    this_rows: list[dict], ref_rows: list[dict], *, tol_mean: float, tol_max: float
) -> dict:
    """Per-row logp offsets of THIS reader vs the committed reference; PASS/FAIL.

    Pure function so the gate arithmetic is CPU-smokeable without a model.
    PASS iff mean |offset| <= tol_mean AND max |offset| <= tol_max (nats).
    """
    if len(this_rows) != len(ref_rows):
        raise RuntimeError(f"Anchor row-count mismatch: {len(this_rows)} vs {len(ref_rows)}")
    offsets = [t["logp"] - r["logp"] for t, r in zip(this_rows, ref_rows, strict=True)]
    abs_offsets = [abs(o) for o in offsets]
    mean_abs = sum(abs_offsets) / len(abs_offsets)
    max_abs = max(abs_offsets)
    return {
        "n": len(offsets),
        "mean_abs_offset_logp": mean_abs,
        "max_abs_offset_logp": max_abs,
        "mean_offset_logp": sum(offsets) / len(offsets),
        "tol_mean_nats": tol_mean,
        "tol_max_nats": tol_max,
        "passed": mean_abs <= tol_mean and max_abs <= tol_max,
        "per_row_offset_logp": offsets,
    }


def _slot_stats_worker_main(*, manifest_path: Path) -> int:
    """Plain AutoModelForCausalLM (bf16, NO peft import); two manifest modes.

    mode="panel": per cell, compute the 4-float base-own slot stats on the
    cell's persisted completions and write slot_stats_<cell>.json the moment
    the cell completes. mode="anchor": slot-read the committed #558
    assistant-cell contexts, compare per-row logp to the committed base-side
    reference, write instrument_anchor.json, and EXIT NON-ZERO on breach.
    """
    manifest = json.loads(manifest_path.read_text())
    mode = manifest["mode"]

    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats
    from explore_persona_space.train.sft import _pick_attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    def _read_contexts(contexts: list[str]) -> list[dict]:
        rows = compute_marker_slot_stats(
            model,
            tokenizer,
            contexts=contexts,
            marker_text=MARKER_TEXT,
            position="end_of_answer",
            batch_size=LOGPROB_BATCH_SIZE,
            device="cuda:0",
            eos_token_id=EOS_TOKEN_ID,
        )
        for row in rows:
            if not all(math.isfinite(v) for v in row.values()):
                raise RuntimeError(f"Non-finite slot stat: {row}")
        return rows

    rc = 0
    if mode == "anchor":
        contexts = _anchor_contexts_from_committed()
        log.info("Instrument anchor: slot-reading %d committed contexts", len(contexts))
        this_rows = _read_contexts(contexts)
        result = compare_anchor_rows(
            this_rows,
            _anchor_reference_rows(),
            tol_mean=ANCHOR_TOL_MEAN_NATS,
            tol_max=ANCHOR_TOL_MAX_NATS,
        )
        out_path = Path(manifest["out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    **repro_metadata(),
                    "issue": ISSUE_563,
                    "mode": "instrument_anchor",
                    "anchor_completions": str(ANCHOR_COMPLETIONS),
                    "anchor_reference": str(ANCHOR_REFERENCE),
                    "this_rows": this_rows,
                    **result,
                },
                indent=2,
            )
        )
        log.info(
            "Instrument anchor -> %s (passed=%s mean=%.4f max=%.4f)",
            out_path,
            result["passed"],
            result["mean_abs_offset_logp"],
            result["max_abs_offset_logp"],
        )
        if not result["passed"]:
            log.error(
                "ANCHOR GATE FAIL: plain-base reader diverges from the parent's "
                "disable_adapter() base side (mean %.4f > %.2f or max %.4f > %.2f nats). "
                "Diagnose attn implementation / dtype / batch shape before ANY panel spend.",
                result["mean_abs_offset_logp"],
                ANCHOR_TOL_MEAN_NATS,
                result["max_abs_offset_logp"],
                ANCHOR_TOL_MAX_NATS,
            )
            rc = 3
    elif mode == "panel":
        for cell in manifest["cells"]:
            cell_name = cell["name"]
            records = json.loads(Path(cell["records_in"]).read_text())
            contexts: list[str] = []
            for r in records:
                comp_clean = r["completion_text"].rstrip()
                if comp_clean.endswith(MARKER_TEXT.strip()):
                    comp_clean = comp_clean[: -len(MARKER_TEXT.strip())].rstrip()
                contexts.append(r["prefix"] + comp_clean)
            log.info("Slot stats: cell=%s n=%d (base-own side)", cell_name, len(contexts))
            rows = _read_contexts(contexts)
            out_path = Path(cell["out"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"n": len(rows), "base_own": rows}))
            log.info("Slot stats persisted -> %s", out_path)
    else:
        raise RuntimeError(f"Unknown manifest mode: {mode!r}")

    # No `del model`: the worker process exits immediately after return (the
    # parent rig's del would also unbind the closure cell `_read_contexts`
    # references — pyflakes F821).
    gc.collect()
    torch.cuda.empty_cache()
    return rc


# ── Instrument anchor orchestration (subprocess + hard gate) ─────────────────


def run_instrument_anchor(*, log_dir: Path) -> dict:
    """Anchor gate: spawn the HF-only worker in anchor mode; HARD-RAISE on breach.

    No panel generation happens after a failed anchor (plan section 7 kill
    criterion 2). Returns the parsed instrument_anchor.json on PASS.
    """
    out_path = EVAL_RESULTS_DIR_563 / "instrument_anchor.json"
    manifest = {"mode": "anchor", "out": str(out_path)}
    manifest_path = EVAL_RESULTS_DIR_563 / "anchor_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        run_slot_stats_subprocess(
            manifest_path=manifest_path, log_path=log_dir / "anchor_worker.log"
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Instrument anchor gate FAILED (worker non-zero). {e}\n"
            f"See {out_path} (if written) for per-row offsets."
        ) from e
    if not out_path.exists():
        raise RuntimeError(f"Anchor worker exited 0 but {out_path} missing — inspect worker log.")
    result = json.loads(out_path.read_text())
    if not result["passed"]:
        # Defense in depth: the worker exits non-zero on breach, but never
        # proceed on a stale/inconsistent file either.
        raise RuntimeError(f"Instrument anchor recorded passed=false in {out_path}.")
    log.info(
        "Instrument anchor PASS: mean |offset| %.4f <= %.2f, max %.4f <= %.2f nats.",
        result["mean_abs_offset_logp"],
        ANCHOR_TOL_MEAN_NATS,
        result["max_abs_offset_logp"],
        ANCHOR_TOL_MAX_NATS,
    )
    return result


# ── Summary (parent summarize_cell adapted to single base-own side) ──────────


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summarize_cell(records: list[dict], slot_stats: dict | None) -> dict:
    """Per-cell rollup: emission + truncation rates and the THREE-space slot
    means (log-prob PRIMARY, EOS-margin logit SECONDARY, probability sanity),
    single base-own side."""
    n = len(records)
    summary = {
        "n": n,
        "emission_rate": sum(r["contains_marker"] for r in records) / max(n, 1),
        "ends_with_marker_rate": sum(r["ends_with_marker"] for r in records) / max(n, 1),
        "truncation_rate": sum(r["truncated"] for r in records) / max(n, 1),
        "mean_generated_tokens": _mean([float(r["n_generated_tokens"]) for r in records]),
    }
    if slot_stats is not None:
        rows = slot_stats["base_own"]
        summary.update(
            {
                "logp_mean": _mean([r["logp"] for r in rows]),
                "z_marker_mean": _mean([r["z_marker"] for r in rows]),
                "eos_margin_mean": _mean([r["z_marker"] - r["z_eos"] for r in rows]),
                "logZ_mean": _mean([r["logZ"] for r in rows]),
                "prob_mean": _mean([math.exp(r["logp"]) for r in rows]),
            }
        )
    return summary


# ── Pod-side result-reporting (poll_pipeline.py contract, issue 563) ─────────


def write_sentinel_563(slug: str, *, kind: str, note: str, version: int = 1) -> Path:
    """poll_pipeline-conformant sentinel for THIS issue (the pinned common's
    write_sentinel bakes in ISSUE=543 in both filename and task_id — reusing
    it would post markers onto task #543)."""
    d = sentinel_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"issue-{ISSUE_563}-{slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "task_id": ISSUE_563,
        "kind": kind,
        "version": version,
        "gate": None,
        "blocks_pipeline": False,
        "by": "eval_issue563_base_panel",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("Sentinel written: %s (kind=%s)", path, kind)
    return path


# ── CPU-only launch-validity dry run (no vLLM / CUDA import) ─────────────────

# Plan-quoted parent rises (12-adapter means of logp_base_mean[cell] -
# logp_base_mean[trigger50] from the committed #558 rollup); the dry run and
# the rollup both recompute + assert these to +-0.001 (guards a stale parent
# file pre-launch).
R_C_EXPECTED: dict[str, float] = {
    "doctor": 0.505,
    "software_engineer": 0.484,
    "french_person": 1.438,
    "police_officer": 1.156,
}
PARENT_ROLLUP_PATH_558 = PROJECT_ROOT / "eval_results" / "issue_558" / "rollup.json"


def recompute_parent_rises(parent_rollup_path: Path = PARENT_ROLLUP_PATH_558) -> dict[str, float]:
    """R_c per persona cell from the committed #558 rollup, asserted vs plan values."""
    if not parent_rollup_path.exists():
        raise FileNotFoundError(f"Parent rollup missing: {parent_rollup_path}")
    cs = json.loads(parent_rollup_path.read_text())["cell_summaries"]
    rises: dict[str, float] = {}
    for cell in R_C_EXPECTED:
        vals = [
            cs[slug][cell]["logp_base_mean"] - cs[slug]["trigger50"]["logp_base_mean"]
            for slug in cs
        ]
        rises[cell] = sum(vals) / len(vals)
        if abs(rises[cell] - R_C_EXPECTED[cell]) > 0.001:
            raise RuntimeError(
                f"Parent rise recompute for {cell}: {rises[cell]:.4f} != plan-quoted "
                f"{R_C_EXPECTED[cell]:.3f} (+-0.001). Stale/drifted parent rollup — do not launch."
            )
    return rises


def run_dry_run_cells(args: argparse.Namespace) -> int:
    """Build all cells at full n + verify the launch-critical contracts on CPU.

    Covers: marker preflight (real tokenizer), pinned-revision question fetch
    + sha256 + count assert, cell construction (shapes / prompts / slugs
    digests), identical-question-list assert, anchor INPUT existence + shape
    + finiteness, and the parent-rise (R_c) recompute assert. Exits 0 with
    [phase=done] WITHOUT importing vLLM or touching CUDA.
    """
    phase_log("dry_run_cells")
    preflight = marker_preflight()
    eval_qs = ensure_eval_questions_local_pinned()
    cell_names = _parse_cells(args.cells)
    cells = build_cells_panel(eval_qs, smoke=args.smoke, cell_names=cell_names)
    n_expect = N_SMOKE_PROMPTS if args.smoke else N_PANEL_PROMPTS_563

    digest: dict[str, Any] = {"cells": {}}
    for cell_name, items in cells.items():
        if len(items) != n_expect:
            raise RuntimeError(f"Cell {cell_name}: {len(items)} prompts, expected {n_expect}")
        persona_key = PANEL_CELLS[cell_name]
        if any(it["persona_key"] != persona_key for it in items):
            raise RuntimeError(f"Cell {cell_name}: persona_key mismatch vs PANEL_CELLS")
        if any(not it["user"].startswith(trigger_user("").rstrip()) for it in items):
            raise RuntimeError(f"Cell {cell_name}: trigger key missing from a user turn")
        digest["cells"][cell_name] = {
            "n": len(items),
            "persona_key": persona_key,
            "system_head": items[0]["system"][:60],
            "user_head": items[0]["user"][:80],
        }
    # All cells share the identical question list (paired-contrast invariant).
    base_users = [it["user"] for it in cells["trigger50"]]
    for cell_name, items in cells.items():
        if [it["user"] for it in items] != base_users:
            raise RuntimeError(f"Cell {cell_name}: question list differs from trigger50")

    # Anchor inputs: existence + shape + finiteness + context construction
    # (the real committed artifacts the anchor gate will slot-read on the pod).
    anchor_contexts = _anchor_contexts_from_committed()
    ref_rows = _anchor_reference_rows()
    if any(not all(math.isfinite(v) for v in row.values()) for row in ref_rows):
        raise RuntimeError("Anchor reference contains non-finite rows.")
    digest["anchor_inputs"] = {
        "n_contexts": len(anchor_contexts),
        "n_reference_rows": len(ref_rows),
        "reference_logp_mean": _mean([r["logp"] for r in ref_rows]),
        "context_head": anchor_contexts[0][:80],
    }
    # Parent-rise recompute assert (stale-parent guard, plan section 3.3 item 3).
    digest["parent_rises_recomputed"] = recompute_parent_rises()
    digest["questions_sha256"] = EXPECTED_QUESTIONS_SHA256
    digest["questions_hub_revision"] = HUB_QUESTIONS_REVISION
    digest["marker_preflight"] = preflight

    out_dir = EVAL_RESULTS_DIR_563 / "dry_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dry_run_cells.json"
    out_path.write_text(
        json.dumps(
            {**repro_metadata(), "issue": ISSUE_563, "mode": "dry_run_cells", **digest}, indent=2
        )
    )
    log.info("Dry run digest -> %s", out_path)
    phase_log("done")
    return 0


# ── Main entrypoint ──────────────────────────────────────────────────────────


def run_one(args: argparse.Namespace) -> int:
    marker_preflight()
    eval_qs = ensure_eval_questions_local_pinned()
    cell_names = _parse_cells(args.cells)
    cells = build_cells_panel(eval_qs, smoke=args.smoke, cell_names=cell_names)

    out_dir = OUT_DIR_563 / "smoke" if args.smoke else OUT_DIR_563
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = sentinel_dir()

    # Phase 1 — instrument-equivalence anchor (HF-only subprocess), the hard
    # gate BEFORE any panel GPU spend (plan section 7 kill criterion 2).
    anchor_result: dict | None = None
    if args.skip_anchor:
        log.warning("--skip-anchor: instrument anchor SKIPPED (diagnostic re-runs only).")
    else:
        phase_log("instrument_anchor")
        anchor_result = run_instrument_anchor(log_dir=log_dir)
    if args.anchor_only:
        phase_log("done")
        return 0

    # Phase 2 — base-own panel generation (vLLM in-process).
    phase_log("eval_gen")
    records = generate_completions_base(
        cells=cells, out_dir=out_dir, max_new_tokens=args.max_new_tokens
    )

    # Phase 3 — slot stats (HF-only subprocess).
    phase_log("eval_slot_stats")
    manifest = {
        "mode": "panel",
        "base_model": BASE_MODEL,
        "marker": MARKER_TEXT,
        "cells": [
            {
                "name": c,
                "records_in": str(out_dir / f"completions_{c}.json"),
                "out": str(out_dir / f"slot_stats_{c}.json"),
            }
            for c in cells
        ],
    }
    manifest_path = out_dir / "slot_stats_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    run_slot_stats_subprocess(manifest_path=manifest_path, log_path=out_dir / "slot_worker.log")

    summary = {
        **repro_metadata(),
        "issue": ISSUE_563,
        "parent_issue": 558,
        "smoke": args.smoke,
        "adapter": None,  # explicitly: no adapter anywhere in this run
        "max_new_tokens": args.max_new_tokens,
        "questions_sha256": EXPECTED_QUESTIONS_SHA256,
        "questions_hub_revision": HUB_QUESTIONS_REVISION,
        "cells_run": list(cells),
        "anchor": (
            {
                k: anchor_result[k]
                for k in (
                    "passed",
                    "mean_abs_offset_logp",
                    "max_abs_offset_logp",
                    "tol_mean_nats",
                    "tol_max_nats",
                )
            }
            if anchor_result is not None
            else {"anchor_checked": False, "reason": "--skip-anchor"}
        ),
        "cells": {},
    }
    for c, recs in records.items():
        slot_path = out_dir / f"slot_stats_{c}.json"
        slot = json.loads(slot_path.read_text()) if slot_path.exists() else None
        summary["cells"][c] = summarize_cell(recs, slot)

    # Kill criterion 3: assistant-cell base mean log P inside the chain's
    # b-hat sanity range (template/slot breakage or contaminated read outside).
    bhat = summary["cells"]["trigger50"]["logp_mean"]
    lo, hi = BHAT_SANITY_RANGE_563
    if not (lo < bhat < hi):
        raise RuntimeError(
            f"Assistant-cell base mean log P = {bhat:.3f} outside the b-hat sanity range "
            f"({lo}, {hi}) — template/slot breakage or contaminated read (kill criterion 3)."
        )

    # Kill criterion 4 surfacing: a >20% truncation cell invalidates its
    # natural-end slot; the registered remedy is a same-session re-run of that
    # cell + trigger50 at --max-new-tokens 3072 (logged deviation), NOT a
    # silent continue. Data stays persisted either way.
    trunc_breach = [
        c for c, s in summary["cells"].items() if s["truncation_rate"] > TRUNCATION_RATE_KILL
    ]
    summary["truncation_kill_criterion_breach"] = trunc_breach
    if trunc_breach:
        log.error(
            "KILL CRITERION 4: truncation rate > %.0f%% in cell(s) %s. Re-run those cells "
            "PLUS trigger50 in one fresh session: --cells %s --max-new-tokens 3072",
            TRUNCATION_RATE_KILL * 100,
            trunc_breach,
            ",".join(["trigger50", *[c for c in trunc_breach if c != "trigger50"]]),
        )

    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("run_summary -> %s", out_dir / "run_summary.json")

    # Phase 4 — raw-completion upload (Upload Policy: raw completions MUST
    # land on the HF data repo under the dispatcher's normal exit path).
    if not args.skip_upload and not args.smoke:
        phase_log("eval_upload")
        from explore_persona_space.orchestrate.hub import upload_dataset_directory

        dest = f"{HUB_RAW_BUCKET_563}/base_panel"
        upload_dataset_directory(out_dir, dest, pattern="completions_*.json")

    write_sentinel_563(
        "eval-base-panel" + ("-smoke" if args.smoke else ""),
        kind="epm:progress",
        note=json.dumps(
            {
                "event": "base_panel_complete",
                "smoke": args.smoke,
                "cells_run": list(cells),
                "anchor": summary["anchor"],
                "truncation_kill_criterion_breach": trunc_breach,
                "cells": {
                    c: {
                        k: summary["cells"][c].get(k)
                        for k in ("n", "emission_rate", "truncation_rate", "logp_mean")
                    }
                    for c in summary["cells"]
                },
            }
        ),
    )
    phase_log("done")
    return 0


def _parse_cells(cells_arg: str) -> list[str] | None:
    """``--cells`` comma list -> cell names (None = all 5)."""
    if cells_arg.strip().lower() == "all":
        return None
    return [c.strip() for c in cells_arg.split(",") if c.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #563 base-own persona-panel eval: 5 cells, 4-float slot stats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--slot-stats-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="20 prompts/cell instead of 250.")
    p.add_argument(
        "--cells",
        type=str,
        default="all",
        help="Comma list of cells to run (must include trigger50); default all 5. "
        "Kill-criterion-4 re-runs pass the truncating cell + trigger50 here.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=EVAL_MAX_NEW_TOKENS,
        help="Greedy generation cap (2048 default; 3072 is the registered "
        "kill-criterion-4 re-run deviation).",
    )
    p.add_argument(
        "--dry-run-cells",
        action="store_true",
        help="CPU-only launch-validity dry run (no vLLM/CUDA); exits after digests.",
    )
    p.add_argument(
        "--anchor-only",
        action="store_true",
        help="Run the instrument-equivalence anchor gate, then exit.",
    )
    p.add_argument(
        "--skip-anchor",
        action="store_true",
        help="Skip the anchor gate entirely (diagnostic re-runs only).",
    )
    p.add_argument("--skip-upload", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.slot_stats_worker:
        # Worker inherits the parent's CUDA_VISIBLE_DEVICES via the explicit
        # env passthrough; do NOT re-pin here.
        return _slot_stats_worker_main(manifest_path=Path(args.manifest))
    if args.dry_run_cells:
        return run_dry_run_cells(args)
    # Pin BEFORE any torch/vllm import touches CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
