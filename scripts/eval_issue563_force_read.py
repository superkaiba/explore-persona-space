#!/usr/bin/env python3
"""Issue #563 follow-up `fixed-completion-force-read` — same answers, five prompts.

Force-reads the assistant cell's 250 committed base-own completions
(``eval_results/issue_563/base/completions_trigger50.json``) under each of the
five panel system prompts and records the 4-float marker slot stats per row
(plan v2 tasks/.../563/plans/plan.md section 3). NO generation phase, NO vLLM
import anywhere — the v1 slot-worker subprocess existed solely to isolate HF
loads from vLLM's in-process monkey-patch; with no engine in this process the
reader runs in-process (plan section 11 item 5; the diagonal anchor gate
empirically covers any residual reader drift).

Phases (checkpoint-per-phase: each cell's slot file lands the moment the cell
finishes):
  [phase=force_read_diagonal]  assistant cell FIRST; compare_anchor_rows vs the
                               committed slot_stats_trigger50.json["base_own"];
                               HARD-RAISE on breach (kill criterion 3) ->
                               OUT_DIR/instrument_anchor_diagonal.json
  [phase=force_read_panel]     4 role cells -> OUT_DIR/slot_stats_force_read_<cell>.json
  [phase=summary]              b-hat sanity on the diagonal; run_summary.json
  sentinel + [phase=done]

Startup asserts: out-dir is EXACTLY eval_results/issue_563/fixed-completion-
force-read (the parent's base/ outputs are never touched; inputs are opened
read-only), marker_preflight, input shapes, and — in --dry-run — the 250-row
prefix-rebuild byte-equality (kill criterion 2, CPU, pre-provision) plus the
R'_c recompute assert against the committed rollup.json.

Usage (pod, 1 GPU):
    uv run python scripts/eval_issue563_force_read.py --gpu 0
Smoke (same flow, first 20 rows/cell, anchor gate included):
    uv run python scripts/eval_issue563_force_read.py --gpu 0 --smoke
VM CPU dry run (no CUDA model load; THE pre-provision gate):
    uv run python scripts/eval_issue563_force_read.py --dry-run
VM-only CPU wiring smoke (real 7B forward on CPU, first N diagonal rows;
offsets vs the committed reference are reported, NOT gated — CPU kernels
legitimately differ from the committed GPU reference):
    uv run python scripts/eval_issue563_force_read.py --cpu-slot-smoke 2
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="eval_issue563_force_read")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    BASE_MODEL,
    EOS_TOKEN_ID,
    MARKER_TEXT,
    PROJECT_ROOT,
    marker_preflight,
    phase_log,
    repro_metadata,
    sentinel_dir,
)
from eval_issue563_base_panel import (  # noqa: E402
    ANCHOR_TOL_MAX_NATS,
    ANCHOR_TOL_MEAN_NATS,
    BHAT_SANITY_RANGE_563,
    ISSUE_563,
    LOGPROB_BATCH_SIZE,
    N_PANEL_PROMPTS_563,
    PANEL_CELLS,
    compare_anchor_rows,
    panel_persona_prompts,
)

log = logging.getLogger("eval_issue563_force_read")

# ── Inputs (committed parent artifacts; READ-ONLY) ───────────────────────────

IN_COMPLETIONS = PROJECT_ROOT / "eval_results" / "issue_563" / "base" / "completions_trigger50.json"
IN_SLOT_REF = PROJECT_ROOT / "eval_results" / "issue_563" / "base" / "slot_stats_trigger50.json"
COMMITTED_ROLLUP = PROJECT_ROOT / "eval_results" / "issue_563" / "rollup.json"

# ── Output dir (plan section 3: asserted EXACTLY this path at startup) ───────

OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_563" / "fixed-completion-force-read"
EXPECTED_OUT_DIR_REL = "eval_results/issue_563/fixed-completion-force-read"

DIAGONAL_CELL = "trigger50"
ROLE_CELLS = ("doctor", "software_engineer", "french_person", "police_officer")
N_SMOKE_ROWS = 20
SLOT_KEY = "force_read"

# Plan-quoted own-content rises (R'_c) from the committed
# eval_results/issue_563/rollup.json panel.cells.<cell>.d_logp.mean (plan
# section 5; per_question_d_logp lives at the CELL level — fact-checker note).
# Recomputed + asserted to +-0.01 by the dry run AND the rollup.
R_PRIME_C_EXPECTED: dict[str, float] = {
    "doctor": 0.9905,
    "software_engineer": 0.3950,
    "french_person": 4.1461,
    "police_officer": 2.2765,
}
R_PRIME_C_TOL = 0.01


def assert_out_dir() -> None:
    """HARD assert OUT_DIR is exactly the registered follow-up path.

    The parent's named concern: nothing this script does may write under
    eval_results/issue_563/base/ (the v1 clean-result artifacts).
    """
    rel = OUT_DIR.relative_to(PROJECT_ROOT).as_posix()
    if rel != EXPECTED_OUT_DIR_REL:
        raise RuntimeError(f"OUT_DIR {rel!r} != registered {EXPECTED_OUT_DIR_REL!r} — abort.")
    if "base" in OUT_DIR.relative_to(PROJECT_ROOT / "eval_results" / "issue_563").parts:
        raise RuntimeError(f"OUT_DIR {rel!r} is inside the parent's base/ outputs — abort.")


def _read_json_readonly(path: Path) -> Any:
    """Read-only open (mode 'r') of a committed parent artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Committed input missing: {path} (sync the repo)")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_input_records() -> list[dict]:
    """The 250 committed assistant-cell completion records, shape-asserted."""
    records = _read_json_readonly(IN_COMPLETIONS)
    if len(records) != N_PANEL_PROMPTS_563:
        raise RuntimeError(f"{IN_COMPLETIONS}: {len(records)} rows, expected {N_PANEL_PROMPTS_563}")
    required = ("system", "user", "prefix", "completion_text", "truncated")
    for i, r in enumerate(records):
        missing = [k for k in required if k not in r]
        if missing:
            raise RuntimeError(f"{IN_COMPLETIONS}[{i}] missing fields {missing}")
    n_trunc = sum(r["truncated"] for r in records)
    if n_trunc != 0:
        raise RuntimeError(f"{IN_COMPLETIONS}: {n_trunc} truncated rows; plan requires 0")
    sessions = {r.get("engine_session") for r in records}
    if len(sessions) != 1:
        raise RuntimeError(f"{IN_COMPLETIONS}: mixed engine sessions {sorted(map(str, sessions))}")
    return records


def load_reference_rows() -> list[dict]:
    """The committed v1 base_own 4-float rows (the diagonal anchor reference)."""
    slot = _read_json_readonly(IN_SLOT_REF)
    rows = slot["base_own"]
    if not (slot["n"] == len(rows) == N_PANEL_PROMPTS_563):
        raise RuntimeError(
            f"{IN_SLOT_REF}: n={slot['n']}, rows={len(rows)}; expected {N_PANEL_PROMPTS_563}"
        )
    for i, row in enumerate(rows):
        if not all(math.isfinite(v) for v in row.values()):
            raise RuntimeError(f"Non-finite reference row [{i}]: {row}")
    return rows


def recompute_own_content_rises(rollup_path: Path = COMMITTED_ROLLUP) -> dict[str, float]:
    """R'_c per role cell from the committed #563 rollup, asserted vs plan +-0.01.

    Recomputed two ways per cell — the stored ``panel.cells.<cell>.d_logp.mean``
    AND the mean of the cell-level ``per_question_d_logp`` array — and both are
    asserted against the plan-quoted value (stale-file guard, v1 section 3.3
    pattern).
    """
    cells = _read_json_readonly(rollup_path)["panel"]["cells"]
    rises: dict[str, float] = {}
    for cell, expected in R_PRIME_C_EXPECTED.items():
        stored = cells[cell]["d_logp"]["mean"]
        per_q = cells[cell]["per_question_d_logp"]
        if len(per_q) != N_PANEL_PROMPTS_563:
            raise RuntimeError(f"rollup.json {cell}: {len(per_q)} per-question deltas, want 250")
        recomputed = sum(per_q) / len(per_q)
        for label, value in (("stored d_logp.mean", stored), ("mean(per_question)", recomputed)):
            if abs(value - expected) > R_PRIME_C_TOL:
                raise RuntimeError(
                    f"R'_c recompute for {cell}: {label} = {value:.4f} != plan-quoted "
                    f"{expected:.4f} (+-{R_PRIME_C_TOL}). Stale/drifted rollup.json — abort."
                )
        rises[cell] = stored
    return rises


# ── Context construction (kill criterion 2 lives here) ──────────────────────


def strip_marker_tail(completion_text: str) -> str:
    """v1's marker-strip convention (the input set has 0 emissions; kept anyway)."""
    comp_clean = completion_text.rstrip()
    if comp_clean.endswith(MARKER_TEXT.strip()):
        comp_clean = comp_clean[: -len(MARKER_TEXT.strip())].rstrip()
    return comp_clean


def build_force_read_contexts(records: list[dict], tokenizer: Any) -> dict[str, list[str]]:
    """Per-cell slot-read contexts: rebuilt chat prefix + the FIXED completion.

    For each of the 5 panel cells the chat-template prefix is rebuilt from
    (cell system prompt, record["user"]) with ``add_generation_prompt=True``,
    then the marker-stripped fixed completion is appended.

    HARD ASSERT (kill criterion 2, CPU-checkable): for the assistant cell the
    rebuilt prefix == record["prefix"] byte-for-byte for every row — proves the
    generation-time (vLLM tokenizer) and force-read-time (HF tokenizer) chat
    templates are identical; the off-diagonal cells inherit that by
    construction (only the system string differs).
    """
    personas = panel_persona_prompts()
    contexts: dict[str, list[str]] = {}
    for cell_name in (DIAGONAL_CELL, *ROLE_CELLS):
        persona_key = PANEL_CELLS[cell_name]
        system = personas["assistant"] if persona_key == "assistant" else personas[persona_key]
        cell_contexts: list[str] = []
        for i, r in enumerate(records):
            prefix = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": r["user"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            if cell_name == DIAGONAL_CELL and prefix != r["prefix"]:
                raise RuntimeError(
                    f"KILL CRITERION 2: rebuilt assistant prefix != stored prefix at row {i} "
                    "(generation-time vLLM tokenizer vs force-read-time HF tokenizer "
                    "chat-template drift). Halt with zero pod spend; diagnose the "
                    "tokenizer/template version."
                )
            cell_contexts.append(prefix + strip_marker_tail(r["completion_text"]))
        contexts[cell_name] = cell_contexts
    return contexts


# ── Slot read (in-process HF; no vLLM anywhere in this module) ───────────────


def load_reader(device: str) -> tuple[Any, Any]:
    """Plain bf16 AutoModelForCausalLM + tokenizer on ``device`` (no peft)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.train.sft import _pick_attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "token": os.environ.get("HF_TOKEN"),
    }
    if device.startswith("cuda"):
        kwargs["device_map"] = {"": 0}
        kwargs["attn_implementation"] = _pick_attn_implementation()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return model, tokenizer


def read_contexts(model: Any, tokenizer: Any, contexts: list[str], *, device: str) -> list[dict]:
    """4-float slot stats per context (identical call shape to the v1 worker)."""
    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats

    rows = compute_marker_slot_stats(
        model,
        tokenizer,
        contexts=contexts,
        marker_text=MARKER_TEXT,
        position="end_of_answer",
        batch_size=LOGPROB_BATCH_SIZE,
        device=device,
        eos_token_id=EOS_TOKEN_ID,
    )
    for row in rows:
        if not all(math.isfinite(v) for v in row.values()):
            raise RuntimeError(f"Non-finite slot stat: {row}")
    return rows


def write_slot_file(out_dir: Path, cell_name: str, rows: list[dict]) -> Path:
    """Checkpoint-per-phase: one cell's slot file the moment the cell finishes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"slot_stats_force_read_{cell_name}.json"
    path.write_text(json.dumps({"n": len(rows), SLOT_KEY: rows}))
    log.info("Slot stats persisted -> %s (%d rows)", path, len(rows))
    return path


# ── Pod-side result-reporting (poll_pipeline.py contract) ────────────────────


def write_sentinel_force_read(slug: str, *, kind: str, note: str, version: int = 1) -> Path:
    """poll_pipeline-conformant sentinel (v1's write_sentinel_563 pattern,
    re-emitted with the force-read slug and this script as ``by``)."""
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
        "by": "eval_issue563_force_read",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("Sentinel written: %s (kind=%s)", path, kind)
    return path


# ── CPU dry run (THE pre-provision gate; no model load) ──────────────────────


def run_dry_run() -> int:
    """CPU-only launch-validity gate: preflight, input shape asserts, the
    250-row prefix-rebuild byte-equality (kill criterion 2), the R'_c
    recompute assert, and a digest JSON. No CUDA, no model weights."""
    phase_log("dry_run")
    assert_out_dir()
    preflight = marker_preflight()
    records = load_input_records()
    ref_rows = load_reference_rows()
    rises = recompute_own_content_rises()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    contexts = build_force_read_contexts(records, tokenizer)  # byte-equality fires inside
    log.info("Prefix-rebuild byte-equality PASS over all %d assistant rows.", len(records))

    digest = {
        **repro_metadata(),
        "issue": ISSUE_563,
        "mode": "dry_run",
        "followup_label": "fixed-completion-force-read",
        "marker_preflight": preflight,
        "inputs": {
            "completions": str(IN_COMPLETIONS),
            "completions_sha256": _sha256(IN_COMPLETIONS),
            "slot_reference": str(IN_SLOT_REF),
            "slot_reference_sha256": _sha256(IN_SLOT_REF),
            "n_records": len(records),
            "n_truncated": 0,
            "engine_session": records[0].get("engine_session"),
        },
        "prefix_byte_equality_rows": len(records),
        "r_prime_c_recomputed": rises,
        "r_prime_c_plan_quoted": R_PRIME_C_EXPECTED,
        "reference_logp_mean": sum(r["logp"] for r in ref_rows) / len(ref_rows),
        "cells": {
            c: {"n_contexts": len(xs), "context_head": xs[0][:80], "context_tail": xs[0][-80:]}
            for c, xs in contexts.items()
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "dry_run.json"
    out_path.write_text(json.dumps(digest, indent=2))
    log.info("Dry-run digest -> %s", out_path)
    phase_log("done")
    return 0


# ── Main force-read flow (production / --smoke / --cpu-slot-smoke) ───────────


def run_force_read(args: argparse.Namespace) -> int:
    assert_out_dir()
    preflight = marker_preflight()
    records = load_input_records()
    ref_rows = load_reference_rows()
    rises = recompute_own_content_rises()

    cpu_smoke_n = args.cpu_slot_smoke
    if cpu_smoke_n:
        n_rows, device = cpu_smoke_n, "cpu"
        out_dir = OUT_DIR / "cpu_slot_smoke"
    elif args.smoke:
        n_rows, device = N_SMOKE_ROWS, "cuda:0"
        out_dir = OUT_DIR / "smoke"
    else:
        n_rows, device = N_PANEL_PROMPTS_563, "cuda:0"
        out_dir = OUT_DIR
    records = records[:n_rows]
    ref_rows = ref_rows[:n_rows]
    if len(records) != n_rows:
        raise RuntimeError(f"Need {n_rows} input rows; got {len(records)}")

    log.info("Loading reader: %s on %s (in-process, no vLLM)", BASE_MODEL, device)
    model, tokenizer = load_reader(device)
    contexts = build_force_read_contexts(records, tokenizer)  # byte-equality fires inside

    # Phase 1 — assistant diagonal FIRST: anchor gate before any off-diagonal
    # spend (kill criterion 3). Slot file is checkpointed before the gate so a
    # breach still leaves the diagnostic rows on disk.
    phase_log("force_read_diagonal")
    t0 = time.time()
    diag_rows = read_contexts(model, tokenizer, contexts[DIAGONAL_CELL], device=device)
    log.info("Diagonal read: %d rows in %.1fs", len(diag_rows), time.time() - t0)
    write_slot_file(out_dir, DIAGONAL_CELL, diag_rows)
    anchor = compare_anchor_rows(
        diag_rows, ref_rows, tol_mean=ANCHOR_TOL_MEAN_NATS, tol_max=ANCHOR_TOL_MAX_NATS
    )
    anchor_path = out_dir / "instrument_anchor_diagonal.json"
    anchor_path.write_text(
        json.dumps(
            {
                **repro_metadata(),
                "issue": ISSUE_563,
                "mode": "instrument_anchor_diagonal",
                "device": device,
                "n_rows": n_rows,
                "anchor_completions": str(IN_COMPLETIONS),
                "anchor_reference": str(IN_SLOT_REF),
                "this_rows": diag_rows,
                **anchor,
            },
            indent=2,
        )
    )
    log.info(
        "Diagonal anchor -> %s (passed=%s mean=%.4f max=%.4f)",
        anchor_path,
        anchor["passed"],
        anchor["mean_abs_offset_logp"],
        anchor["max_abs_offset_logp"],
    )
    if not anchor["passed"]:
        if cpu_smoke_n:
            log.warning(
                "CPU wiring smoke: anchor offsets exceed the GPU tolerances "
                "(mean %.4f / max %.4f nats) — EXPECTED on CPU kernels; reported, not gated.",
                anchor["mean_abs_offset_logp"],
                anchor["max_abs_offset_logp"],
            )
        else:
            raise RuntimeError(
                f"KILL CRITERION 3: diagonal anchor breach (mean "
                f"{anchor['mean_abs_offset_logp']:.4f} > {ANCHOR_TOL_MEAN_NATS} or max "
                f"{anchor['max_abs_offset_logp']:.4f} > {ANCHOR_TOL_MAX_NATS} nats vs the "
                f"committed {IN_SLOT_REF.name}). Halt before any off-diagonal read; diagnose "
                "attn implementation / dtype / batch shape. Expected offset: exactly 0.0."
            )

    # Phase 2 — the 4 role cells (checkpoint-per-cell). The CPU wiring smoke
    # stops at the diagonal: its purpose is the end-to-end reader path, and 4
    # more CPU 7B cells buy nothing the diagonal didn't already exercise.
    slot_rows: dict[str, list[dict]] = {DIAGONAL_CELL: diag_rows}
    if not cpu_smoke_n:
        phase_log("force_read_panel")
        for cell_name in ROLE_CELLS:
            t0 = time.time()
            rows = read_contexts(model, tokenizer, contexts[cell_name], device=device)
            log.info("Cell %s: %d rows in %.1fs", cell_name, len(rows), time.time() - t0)
            write_slot_file(out_dir, cell_name, rows)
            slot_rows[cell_name] = rows

    # Phase 3 — summary. b-hat sanity (kill criterion 4) on the diagonal.
    phase_log("summary")
    diag_logp_mean = sum(r["logp"] for r in diag_rows) / len(diag_rows)
    lo, hi = BHAT_SANITY_RANGE_563
    if not (lo < diag_logp_mean < hi) and not cpu_smoke_n:
        raise RuntimeError(
            f"KILL CRITERION 4: diagonal mean log P = {diag_logp_mean:.3f} outside the "
            f"b-hat sanity range ({lo}, {hi}) — contaminated read."
        )
    personas = panel_persona_prompts()
    summary = {
        **repro_metadata(),
        "issue": ISSUE_563,
        "followup_label": "fixed-completion-force-read",
        "mode": "cpu_slot_smoke" if cpu_smoke_n else ("smoke" if args.smoke else "production"),
        "device": device,
        "n_rows": n_rows,
        "batch_size": LOGPROB_BATCH_SIZE,
        "marker_preflight": preflight,
        "inputs": {
            "completions": str(IN_COMPLETIONS),
            "completions_sha256": _sha256(IN_COMPLETIONS),
            "slot_reference": str(IN_SLOT_REF),
            "slot_reference_sha256": _sha256(IN_SLOT_REF),
            "engine_session": records[0].get("engine_session"),
        },
        "r_prime_c_recomputed": rises,
        # Critic concern 8 (prompt-string identity audit): the five system-
        # prompt strings recorded VERBATIM.
        "system_prompts_verbatim": {
            cell: (
                personas["assistant"]
                if PANEL_CELLS[cell] == "assistant"
                else personas[PANEL_CELLS[cell]]
            )
            for cell in (DIAGONAL_CELL, *ROLE_CELLS)
        },
        "anchor": {
            k: anchor[k]
            for k in (
                "passed",
                "mean_abs_offset_logp",
                "max_abs_offset_logp",
                "tol_mean_nats",
                "tol_max_nats",
            )
        },
        "diagonal_logp_mean": diag_logp_mean,
        "bhat_sanity_range": list(BHAT_SANITY_RANGE_563),
        "cells_read": sorted(slot_rows),
        "cell_logp_means": {
            c: sum(r["logp"] for r in rows) / len(rows) for c, rows in slot_rows.items()
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("run_summary -> %s", out_dir / "run_summary.json")

    # No upload phase: no new raw completions exist (the inputs are already
    # committed on git + the HF data repo); slot stats + summary are tiny
    # JSONs committed to git by the orchestrator.
    write_sentinel_force_read(
        "force-read" + ("-cpu-smoke" if cpu_smoke_n else "-smoke" if args.smoke else ""),
        kind="epm:progress",
        note=json.dumps(
            {
                "event": "force_read_complete",
                "followup_label": "fixed-completion-force-read",
                "mode": summary["mode"],
                "n_rows": n_rows,
                "anchor": summary["anchor"],
                "diagonal_logp_mean": diag_logp_mean,
                "cell_logp_means": summary["cell_logp_means"],
            }
        ),
    )
    phase_log("done")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #563 fixed-completion force-read: 5 prompts x 250 fixed completions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="First 20 rows/cell instead of 250.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only pre-provision gate: preflight + shape asserts + 250-row "
        "prefix byte-equality + R'_c recompute; no model load.",
    )
    p.add_argument(
        "--cpu-slot-smoke",
        type=int,
        default=0,
        metavar="N",
        help="VM-only wiring smoke: real bf16 forward on CPU over the first N "
        "diagonal rows; offsets vs the committed GPU reference are reported, "
        "not gated. Writes under OUT_DIR/cpu_slot_smoke/. Never used on a pod.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return run_dry_run()
    if not args.cpu_slot_smoke:
        # Pin BEFORE any torch import touches CUDA (v1 pattern).
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return run_force_read(args)


if __name__ == "__main__":
    raise SystemExit(main())
