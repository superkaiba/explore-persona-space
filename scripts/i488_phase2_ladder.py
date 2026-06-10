# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #488 Phase 2 v6 — recipe ladder dispatcher.

Plan v6 §4.8 + §7. Climbs a 5-rung (lr × r × rows × epochs) ladder, each
rung = one 2-cell train (A1 + G2 at frac=3.00) + one 6-cell on-policy
emit panel for Gate BYSTANDER. Verdict per rung is computed from emit
rates against v6 §7 gate semantics:

  - **Gate 1 (label-mask audit):** runs ONCE at L1, before training launches.
    Source: plan v6 §7 (1). Verifies positive row's loss-bearing positions =
    marker token (id 83399) + immediately-following EOS; negative row's
    loss-bearing position = `<|im_end|>` (id 151645).

  - **Gate ANCHOR (v6 PICK-on-saturation semantics):** per rung, on the
    2-cell A1+G2 smoke at frac=3.00, on-policy emit N=8×1 sample at
    temp=1.0/top_p=1.0/max_new_tokens=2048. Decision tree (plan v6 §7 (2)):

      - A1 self-emit < 0.20 → verdict=CLIMB reason=below_floor → advance.
      - A1 self-emit ∈ [0.20, 0.85] AND median per-source diag emit ≥ 0.20
        → PASS → proceed to Gate BYSTANDER.
      - A1 self-emit > 0.85 → DO NOT CLIMB → proceed to Gate BYSTANDER;
        if PASS → verdict=PICK_AT_SATURATION; if FAIL → see
        UNIFORM_LEAKAGE branch.

  - **Gate BYSTANDER (v6 split-criterion — median over full panel + max
    over non-stylized subset):** same A1 adapter, 6-cell off-diag panel
    (B1, F1, G1, A3, D2, B5), N=8×1 sample. PASS requires BOTH:

      - median bystander emit ≤ 0.30 OVER ALL 6 cells (uniform-leakage
        check: "most cells quiet").
      - max-bystander emit on the NON-STYLIZED SUBSET {B1, F1, G1, D2, B5}
        − A1_self_emit ≤ -0.10 (localization-sanity; A3 EXCLUDED per
        v6 Must-Fix #2 — A3 is the close-in-JS stylized bystander = H1
        prediction target).

      FAIL: if max-bystander on non-stylized subset < 0.50 AND ANCHOR was
      in [0.20, 0.85] → verdict=UNIFORM_LEAKAGE_WEAK → CLIMB. If max ≥ 0.50
      OR ANCHOR was PICK_AT_SATURATION → verdict=UNIFORM_LEAKAGE → exit
      BLOCKED.

Writes one row per rung to ``logs/issue_488/ladder.jsonl`` (schema per
plan v6 §4.8). On terminal PASS / PICK_AT_SATURATION, writes the picked
rung's full recipe to a sentinel file at
``/workspace/logs/issue-488-smoke-result.json`` that the VM-side
``poll_pipeline.py`` drains and posts as ``epm:smoke-result v1``
(per CLAUDE.md "Pod-side code NEVER shells out to scripts/task.py").

Each rung is TWO subprocess calls (train, then measure-emit) so vLLM
worker teardown is clean per ``.claude/rules/gotchas.md``.

Reuses 100% of the diagnostic training stack
(``i488_diagnostic_train.py``'s ``_build_training_rows``,
``_load_R_inherited``, ``_load_R_new``, ``_LocalOnlyAdapterSaveCallback``,
MARKER_ID assert) — the only new code here is the rung table, the
gate-decision tree, and the orchestration loop.

Smoke architectural parity: smoke L1 IS the ladder's first rung at the
same scale as the production rungs (same diagnostic_train code path,
same emit code path; the only difference is cell count + N per cell).
PASS_UNIFIED per CLAUDE.md Step 6d.0.

CLI:
    # Run the full ladder until PASS / EXHAUSTED / BLOCKED.
    uv run python scripts/i488_phase2_ladder.py --seed 42

    # Skip to a specific rung (e.g. if rung L1 already trained):
    uv run python scripts/i488_phase2_ladder.py --start-rung L2 --seed 42

    # Run a single rung end-to-end (smoke micro-slice test):
    uv run python scripts/i488_phase2_ladder.py --only-rung L1 --seed 42 \\
        --n-probes-emit 4
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    MARKER_ID,
    MARKER_TEXT,
)

logger = logging.getLogger("i488.ladder")

# ── Rung table (plan v6 §4.8). ASCENDING in expected strength. ─────────────
# Each rung tuple: (lr, lora_r, lora_alpha, max_rows_per_side, epochs,
#                   n_dupes, warmup_ratio).
# Plan v6 §11 Sources tie each row to {arXiv 2507.21509, #406, #460, #462,
# #469}; L1 is the only `ungrounded — needs smoke-test` rung (first-principles
# 2.5× diagnostic lr probe).
RUNGS: dict[str, dict] = {
    "L1": {
        "lr": 5e-6,
        "lora_r": 16,
        "lora_alpha": 32,
        "max_rows_per_side": 150,
        "epochs": 3,
        "n_dupes": 5,
        "warmup_ratio": 0.03,
        "note": "2.5× diagnostic lr at same capacity; cheapest single-axis probe.",
    },
    "L2": {
        "lr": 1e-5,
        "lora_r": 16,
        "lora_alpha": 32,
        "max_rows_per_side": 150,
        "epochs": 3,
        "n_dupes": 5,
        "warmup_ratio": 0.03,
        "note": "Parent #460 lr (5× diagnostic), same capacity.",
    },
    "L3": {
        "lr": 5e-6,
        "lora_r": 32,
        "lora_alpha": 64,
        "max_rows_per_side": 150,
        "epochs": 3,
        "n_dupes": 5,
        "warmup_ratio": 0.03,
        "note": "Rank doubled vs L1 at intermediate lr.",
    },
    "L4": {
        "lr": 1e-5,
        "lora_r": 32,
        "lora_alpha": 64,
        "max_rows_per_side": 150,
        "epochs": 3,
        "n_dupes": 5,
        "warmup_ratio": 0.03,
        "note": "Parent capacity at parent lr but half-rows.",
    },
    "L5": {
        "lr": 1e-5,
        "lora_r": 32,
        "lora_alpha": 64,
        "max_rows_per_side": 300,
        "epochs": 5,
        "n_dupes": 10,
        "warmup_ratio": 0.03,
        "note": "Parent endpoint recipe (300 rows × 5 ep); empirical ceiling.",
    },
}
RUNG_ORDER = ["L1", "L2", "L3", "L4", "L5"]

# Plan v6 §7 gate thresholds. Sources in plan v6 §11.
GATE_ANCHOR_FLOOR = 0.20  # A1 self-emit lower bound → CLIMB if below
GATE_ANCHOR_SATURATION = 0.85  # A1 self-emit upper bound → PICK_AT_SATURATION if above
GATE_ANCHOR_MEDIAN_DIAG_MIN = 0.20  # median per-source diagonal emit min
GATE_BYSTANDER_MEDIAN_MAX = 0.30  # median over all 6 cells max
GATE_BYSTANDER_LOCALIZATION_GAP = -0.10  # max non-stylized − A1_self_emit ceiling
GATE_BYSTANDER_UNIFORM_LEAKAGE_THRESHOLD = 0.50  # max non-stylized ≥ this → hard block

# Default panel composition per plan v6 §4.8.
DEFAULT_BYSTANDER_PANEL = ["B1", "F1", "G1", "A3", "D2", "B5"]
# Non-stylized subset for the max-criterion (A3 excluded per v6 Must-Fix #2).
DEFAULT_NON_STYLIZED_SUBSET = ["B1", "F1", "G1", "D2", "B5"]

LADDER_LOG_DIR = Path("logs/issue_488/ladder")
LADDER_JSONL = LADDER_LOG_DIR / "ladder.jsonl"
ADAPTER_BASE_DEFAULT = Path("/workspace/adapters/i488_ladder")
SENTINEL_PATH = Path("/workspace/logs/issue-488-smoke-result.json")
SENTINEL_FAIL_PATH = Path("/workspace/logs/issue-488-smoke-failed.json")
TRAIN_ROW_DIR_DEFAULT = Path("data/issue_488/train_rows")


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        )
    except Exception:
        return "unknown"


def _append_ladder_row(row: dict) -> None:
    LADDER_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(LADDER_JSONL, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_sentinel_pass(rung: str, recipe: dict, ladder_row: dict, verdict: str) -> None:
    """Write the success sentinel for poll_pipeline → epm:smoke-result v1.

    Per CLAUDE.md Pod-side rule: this file is the ONLY channel back to the
    orchestrator. Contains the picked rung's full recipe so Phase 3 can
    launch with the picked-rung parameters threaded through.

    Schema follows poll_pipeline.py::_SENTINEL_REQUIRED_KEYS: must include
    `sentinel_schema_version=1`, `kind`, `version`. Optional: `note`.
    """
    SENTINEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:smoke-result",
        "version": 1,
        "issue": 488,
        "ts": _now(),
        "by": "i488_phase2_ladder",
        "note": json.dumps(
            {
                "verdict": verdict,
                "picked_rung": rung,
                "recipe": recipe,
                "ladder_row": ladder_row,
                "plan_version": "v6",
                "git_commit": _git_commit(),
            },
            indent=2,
        ),
    }
    SENTINEL_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Wrote PASS sentinel → %s (verdict=%s, rung=%s)", SENTINEL_PATH, verdict, rung)


def _write_sentinel_fail(reason_key: str, reason_long: str, extra: dict | None = None) -> None:
    """Write the failure sentinel for poll_pipeline → epm:failure v1."""
    SENTINEL_FAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:failure",
        "version": 1,
        "issue": 488,
        "ts": _now(),
        "by": "i488_phase2_ladder",
        "note": json.dumps(
            {
                "failure_class": "code",
                "reason": reason_key,
                "reason_long": reason_long,
                "extra": extra or {},
                "plan_version": "v6",
                "git_commit": _git_commit(),
            },
            indent=2,
        ),
    }
    SENTINEL_FAIL_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.error("Wrote FAIL sentinel → %s (%s)", SENTINEL_FAIL_PATH, reason_key)


# ── Gate 1: Label-mask audit (one-shot at L1 only) ─────────────────────────


def _build_audit_train_rows(
    rung: str,
    sources: list[str],
    seed: int,
    rung_train_row_dir: Path,
) -> Path:
    """Materialize the A1 (audit-source) training rows for ``rung`` BEFORE training.

    Round-2 blocker-1 fix: Gate 1 (label-mask audit) MUST run BEFORE
    ``_run_train_for_rung`` at L1 so a misaligned label mask is caught
    before burning ~10-40 min of GPU. We invoke ``i488_diagnostic_train.py``
    with ``--build-rows-only`` (no training, just writes the same train.jsonl
    that the production trainer would consume) so the audit reads byte-
    identical rows.

    Returns the path to the audit source's train.jsonl
    (``<rung_train_row_dir>/i488_A1_seed<seed>.jsonl`` by default).

    Raises:
        RuntimeError: if the build subprocess fails or the expected
            audit jsonl is not present afterwards. NO silent fallback —
            audit-row absence is a HARD pre-train block.
    """
    rung_train_row_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ}
    env["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
    env["I488_LADDER_RUNG_SUFFIX"] = rung
    env["I488_TRAIN_ROW_DIR"] = str(rung_train_row_dir)

    recipe = RUNGS[rung]
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/i488_diagnostic_train.py",
        "--conds",
        *sources,
        "--seed",
        str(seed),
        "--lr",
        str(recipe["lr"]),
        "--lora-r",
        str(recipe["lora_r"]),
        "--lora-alpha",
        str(recipe["lora_alpha"]),
        "--max-rows-per-side",
        str(recipe["max_rows_per_side"]),
        "--warmup-ratio",
        str(recipe["warmup_ratio"]),
        "--epochs",
        str(recipe["epochs"]),
        "--n-dupes",
        str(recipe["n_dupes"]),
        "--build-rows-only",
    ]
    logger.info("[phase=ladder_build_audit_rows] %s", shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"i488_diagnostic_train.py --build-rows-only failed "
            f"rc={proc.returncode} at rung={rung} sources={sources}"
        )
    # The audit source is the first source by convention (A1 in production).
    audit_source = sources[0]
    audit_jsonl = rung_train_row_dir / f"i488_{audit_source}_seed{seed}.jsonl"
    if not audit_jsonl.exists():
        raise RuntimeError(
            f"Audit train rows not written by --build-rows-only: expected "
            f"{audit_jsonl} after running diagnostic_train at rung={rung}. "
            "This is a hard pre-train block — no silent fallback."
        )
    return audit_jsonl


def _run_label_mask_audit(audit_train_jsonl: Path) -> bool:
    """Run the v3 label-mask audit on the audit source's train rows. Returns True if PASS.

    Reuses the existing audit from ``i488_phase2_smoke_calibrate._label_mask_audit``
    rather than re-implementing it (the v6 ladder doesn't change Gate 1).

    Round-2 blocker-1: this function is now invoked BEFORE ``_run_train_for_rung``
    at L1 (callers must call ``_build_audit_train_rows`` first to materialize the
    rows). A ``FileNotFoundError`` here is a HARD failure — not a silent pass —
    because the caller is contractually responsible for building the rows first.
    """
    from i488_phase2_smoke_calibrate import (  # type: ignore
        _label_mask_audit as smoke_label_mask_audit,
    )
    from transformers import AutoTokenizer

    BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Marker assert per CLAUDE.md.
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    try:
        smoke_label_mask_audit(audit_train_jsonl, tokenizer)
    except AssertionError as e:
        _write_sentinel_fail("label_mask_wrong_slot", str(e))
        return False
    except FileNotFoundError as e:
        # Round-2 blocker-1 fix: HARD failure on missing audit rows.
        # Caller (run_ladder) must materialize rows via _build_audit_train_rows
        # BEFORE invoking this. No silent advance — that masked misaligned
        # label-masks for the first L1 train in round 1.
        _write_sentinel_fail(
            "label_mask_audit_rows_missing",
            f"Audit train rows missing at {audit_train_jsonl}: {e}. "
            "Caller must invoke _build_audit_train_rows before _run_label_mask_audit.",
            extra={"audit_train_jsonl": str(audit_train_jsonl)},
        )
        return False
    return True


# ── Per-rung train + emit subprocess invocations ───────────────────────────


def _run_train_for_rung(
    rung: str,
    sources: list[str],
    seed: int,
    adapter_base: Path,
    gpu_id: int,
    log_path: Path,
    rung_train_row_dir: Path,
) -> int:
    """Spawn ``i488_diagnostic_train.py`` as a subprocess at this rung's recipe.

    Per CLAUDE.md ``.claude/rules/gotchas.md``: train + emit are SEPARATE
    subprocess invocations so vLLM workers from a prior rung's emit don't
    leak into the next rung's train.

    The diagnostic_train script saves adapters under
    ``<adapter_base>/i488_<src>_seed<seed>_frac300_diag``; we override
    ``--out-base`` to ``<adapter_base>`` and the script appends the
    ``_diag`` suffix. To get rung-specific adapter dirs, we pass a
    rung-suffixed out-base.

    Returns the subprocess exit code.
    """
    recipe = RUNGS[rung]
    # Adapter dir convention: <adapter_base>/i488_<src>_seed<seed>_frac300_<rung>.
    # diagnostic_train writes <out_base>/i488_<src>_seed<seed>_frac300_diag
    # — we redirect by passing out_base = <adapter_base>/<rung> and then
    # post-rename in this dispatcher. To avoid the rename complexity, we
    # override the `_diag` suffix via PYTHONPATH-injected env var below.
    out_base = adapter_base

    # Standard env carry per `.claude/rules/workflow-fix-on-bug.md` /
    # CLAUDE.md "Pod-side dispatcher silent-death hardening": explicit
    # env= with full os.environ snapshot.
    env = {**os.environ}
    # MooseFS quota safety (diagnostic_train sets EPM_SKIP_INLINE_CHECKPOINT_UPLOAD
    # itself; mirror it here).
    env["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
    # Tell the trainer to use a rung-tagged adapter suffix instead of `_diag`.
    env["I488_LADDER_RUNG_SUFFIX"] = rung
    # Direct the trainer's train-rows persistence under
    # data/issue_488/train_rows/<rung>/ so different rungs do not clobber each
    # other's JSONL when running sequentially.
    env["I488_TRAIN_ROW_DIR"] = str(rung_train_row_dir)

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/i488_diagnostic_train.py",
        "--conds",
        *sources,
        "--seed",
        str(seed),
        "--lr",
        str(recipe["lr"]),
        "--lora-r",
        str(recipe["lora_r"]),
        "--lora-alpha",
        str(recipe["lora_alpha"]),
        "--max-rows-per-side",
        str(recipe["max_rows_per_side"]),
        "--warmup-ratio",
        str(recipe["warmup_ratio"]),
        "--epochs",
        str(recipe["epochs"]),
        "--n-dupes",
        str(recipe["n_dupes"]),
        "--gpu-id",
        str(gpu_id),
        "--out-base",
        str(out_base),
    ]
    logger.info("[phase=ladder_train_%s] cmd=%s", rung, " ".join(shlex.quote(c) for c in cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n==== {_now()} ladder train rung={rung} ====\n")
        logf.flush()
        rc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        ).returncode
    logger.info("[phase=ladder_train_%s] rc=%d log=%s", rung, rc, log_path)
    return rc


def _run_emit_for_rung(
    rung: str,
    sources: list[str],
    bystanders: list[str],
    seed: int,
    n_probes_emit: int,
    max_new_tokens: int,
    adapter_base: Path,
    gpu_id: int,
    log_path: Path,
    out_emit_path: Path,
    bystander_source: str,
) -> int:
    """Spawn ``i488_phase2_ladder_emit.py`` as a separate subprocess.

    Separate from training per ``.claude/rules/gotchas.md`` vLLM teardown.
    """
    recipe = RUNGS[rung]
    env = {**os.environ}

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/i488_phase2_ladder_emit.py",
        "--rung",
        rung,
        "--sources",
        *sources,
        "--bystanders",
        *bystanders,
        "--bystander-source",
        bystander_source,
        "--seed",
        str(seed),
        "--n-probes-emit",
        str(n_probes_emit),
        "--max-new-tokens",
        str(max_new_tokens),
        "--adapter-base",
        str(adapter_base),
        "--out",
        str(out_emit_path),
        "--gpu-id",
        str(gpu_id),
        "--lora-rank",
        str(recipe["lora_r"]),
    ]
    logger.info("[phase=ladder_emit_%s] cmd=%s", rung, " ".join(shlex.quote(c) for c in cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n==== {_now()} ladder emit rung={rung} ====\n")
        logf.flush()
        rc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        ).returncode
    logger.info("[phase=ladder_emit_%s] rc=%d log=%s", rung, rc, log_path)
    return rc


# ── Verdict logic (plan v6 §7 + §4.8 schema) ───────────────────────────────


def _decide_verdict(  # noqa: C901 - the decision tree IS the gate spec
    rung: str,
    emit_payload: dict,
    sources: list[str],
    non_stylized_subset: list[str],
    bystander_source: str,
) -> dict:
    """Compute Gate ANCHOR + Gate BYSTANDER verdict and the ladder.jsonl row.

    Returns a dict with the full ladder row schema per plan v6 §4.8.

    Decision tree (plan v6 §7):

    Gate ANCHOR:
      - a1_self < 0.20 → verdict = CLIMB (reason: below_floor)
      - a1_self ∈ [0.20, 0.85] AND median_diag ≥ 0.20 → ANCHOR PASS → proceed to BYSTANDER
      - a1_self > 0.85 → DO NOT CLIMB; proceed to BYSTANDER; if PASS → PICK_AT_SATURATION

    Gate BYSTANDER:
      - PASS requires BOTH:
          * median over all 6 cells ≤ 0.30
          * max over NON_STYLIZED_SUBSET − a1_self ≤ -0.10
      - FAIL:
          * if max-non-stylized < 0.50 AND ANCHOR was [0.20, 0.85]:
              verdict = CLIMB (reason: weak_localization)
          * if max-non-stylized ≥ 0.50 OR ANCHOR was PICK_AT_SATURATION:
              verdict = UNIFORM_LEAKAGE
    """
    recipe = RUNGS[rung]

    # ── Per-source diagonal (Gate ANCHOR) ──
    anchor_cells = emit_payload.get("anchor_cells", {})
    a1_self_emit = anchor_cells.get("A1", {}).get("emit_rate")
    diag_emits = []
    for src in sources:
        cell = anchor_cells.get(src, {})
        rate = cell.get("emit_rate")
        if rate is not None:
            diag_emits.append(float(rate))
    median_diag_emit = sorted(diag_emits)[len(diag_emits) // 2] if diag_emits else None
    # Use the simpler median definition (50th-percentile, lower-median for
    # even-sized lists). For our 2-source case (A1, G2), the result is the
    # lower of the two emit rates.

    g2_self_emit = anchor_cells.get("G2", {}).get("emit_rate")

    # ── Bystander panel (Gate BYSTANDER) ──
    bystander_cells = emit_payload.get("bystander_cells", {})
    bystander_emits: dict[str, float] = {}
    for cid, cell in bystander_cells.items():
        rate = cell.get("emit_rate")
        if rate is not None:
            bystander_emits[cid] = float(rate)

    full_panel_rates = list(bystander_emits.values())
    if full_panel_rates:
        sorted_rates = sorted(full_panel_rates)
        n = len(sorted_rates)
        if n % 2 == 0:
            # Even-sized: average of the two middle values.
            median_bystander = (sorted_rates[n // 2 - 1] + sorted_rates[n // 2]) / 2.0
        else:
            median_bystander = sorted_rates[n // 2]
    else:
        median_bystander = None
    max_bystander_full_panel = max(full_panel_rates) if full_panel_rates else None

    non_stylized_present = [
        c for c in non_stylized_subset if c in bystander_emits and c != bystander_source
    ]
    non_stylized_rates = [bystander_emits[c] for c in non_stylized_present]
    max_bystander_non_stylized = max(non_stylized_rates) if non_stylized_rates else None

    # ── Gate ANCHOR decision ──
    anchor_pass = False
    anchor_saturated = False
    anchor_failed_floor = False
    anchor_failed_median = False

    if a1_self_emit is None:
        anchor_failed_floor = True
        anchor_decision_reason = "a1_self_emit_missing"
    elif a1_self_emit < GATE_ANCHOR_FLOOR:
        anchor_failed_floor = True
        anchor_decision_reason = f"a1_self_emit={a1_self_emit:.3f} < {GATE_ANCHOR_FLOOR}"
    elif a1_self_emit > GATE_ANCHOR_SATURATION:
        anchor_saturated = True
        anchor_decision_reason = (
            f"a1_self_emit={a1_self_emit:.3f} > {GATE_ANCHOR_SATURATION} (saturated)"
        )
    else:
        if median_diag_emit is None or median_diag_emit < GATE_ANCHOR_MEDIAN_DIAG_MIN:
            anchor_failed_median = True
            anchor_decision_reason = (
                f"median_diag_emit={median_diag_emit} < {GATE_ANCHOR_MEDIAN_DIAG_MIN}"
            )
        else:
            anchor_pass = True
            anchor_decision_reason = (
                f"a1_self_emit={a1_self_emit:.3f} ∈ [{GATE_ANCHOR_FLOOR}, "
                f"{GATE_ANCHOR_SATURATION}] AND median_diag={median_diag_emit:.3f} "
                f">= {GATE_ANCHOR_MEDIAN_DIAG_MIN}"
            )

    gate_anchor_str = (
        "PASS"
        if anchor_pass
        else (
            "PASS_AT_SATURATION"
            if anchor_saturated
            else ("FAIL_BELOW_FLOOR" if anchor_failed_floor else "FAIL_MEDIAN_DIAG")
        )
    )

    # ── Gate BYSTANDER decision ──
    gate_bystander_str = "N/A"
    bystander_pass = False
    bystander_reasons: list[str] = []

    if anchor_pass or anchor_saturated:
        # Compute bystander gate.
        ok_median = median_bystander is not None and median_bystander <= GATE_BYSTANDER_MEDIAN_MAX
        if not ok_median:
            bystander_reasons.append(
                f"median_bystander={median_bystander} > {GATE_BYSTANDER_MEDIAN_MAX}"
            )
        ok_localization = False
        if max_bystander_non_stylized is not None and a1_self_emit is not None:
            gap = max_bystander_non_stylized - a1_self_emit
            ok_localization = gap <= GATE_BYSTANDER_LOCALIZATION_GAP
            if not ok_localization:
                bystander_reasons.append(
                    f"max_bystander_non_stylized={max_bystander_non_stylized:.3f} "
                    f"− a1_self={a1_self_emit:.3f} = {gap:+.3f} > "
                    f"{GATE_BYSTANDER_LOCALIZATION_GAP}"
                )
        else:
            bystander_reasons.append(
                "cannot compute localization gap "
                f"(max_non_stylized={max_bystander_non_stylized}, "
                f"a1_self={a1_self_emit})"
            )

        if ok_median and ok_localization:
            bystander_pass = True
            gate_bystander_str = "PASS"
        else:
            gate_bystander_str = "FAIL"

    # ── Compose final verdict per the v6 decision tree ──
    verdict: str
    verdict_reason: str
    if anchor_failed_floor:
        verdict = "CLIMB"
        verdict_reason = "below_floor"
    elif anchor_failed_median:
        # A1 emit in band but median across (A1, G2) below the floor.
        # Climb to a stronger rung — diagnostic measure of "the source side
        # of the implant hasn't crossed the construct floor yet on average".
        verdict = "CLIMB"
        verdict_reason = "median_diag_below_floor"
    elif anchor_pass and bystander_pass:
        verdict = "PASS"
        verdict_reason = "both_gates_pass"
    elif anchor_saturated and bystander_pass:
        verdict = "PICK_AT_SATURATION"
        verdict_reason = "anchor_saturated_bystander_pass"
    else:
        # Anchor in [0.20, 0.85] (or saturated) but Bystander FAILed.
        if anchor_saturated:
            # Saturation + bystander fail is a HARD block (can't climb past
            # parent recipe meaningfully).
            verdict = "UNIFORM_LEAKAGE"
            verdict_reason = "anchor_saturated_bystander_fail"
        else:
            # ANCHOR in band but bystanders not yet resolved. Plan v6 §7:
            #   if max-non-stylized < 0.50 → CLIMB (weak; stronger rung
            #   may give clearer separation)
            #   if max-non-stylized ≥ 0.50 → UNIFORM_LEAKAGE (climbing
            #   worsens it).
            if (
                max_bystander_non_stylized is not None
                and max_bystander_non_stylized < GATE_BYSTANDER_UNIFORM_LEAKAGE_THRESHOLD
            ):
                verdict = "CLIMB"
                verdict_reason = "weak_localization"
            else:
                verdict = "UNIFORM_LEAKAGE"
                verdict_reason = "max_non_stylized_above_threshold"

    return {
        "ts": _now(),
        "rung": rung,
        "lr": recipe["lr"],
        "r": recipe["lora_r"],
        "alpha": recipe["lora_alpha"],
        "rows": recipe["max_rows_per_side"],
        "epochs": recipe["epochs"],
        "warmup_ratio": recipe["warmup_ratio"],
        "a1_self_emit": a1_self_emit,
        "g2_self_emit": g2_self_emit,
        "median_diag_emit": median_diag_emit,
        "bystander_emits": bystander_emits,
        "median_bystander_emit": median_bystander,
        "max_bystander_emit_full_panel": max_bystander_full_panel,
        "max_bystander_emit_non_stylized": max_bystander_non_stylized,
        "non_stylized_subset_present": non_stylized_present,
        "gate_anchor": gate_anchor_str,
        "gate_anchor_reason": anchor_decision_reason,
        "gate_bystander": gate_bystander_str,
        "gate_bystander_reasons": bystander_reasons,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "git_commit": _git_commit(),
        "plan_version": "v6",
    }


# ── Top-level ladder loop ───────────────────────────────────────────────────


def run_ladder(
    *,
    seed: int,
    sources: list[str],
    bystanders: list[str],
    non_stylized_subset: list[str],
    bystander_source: str,
    adapter_base: Path,
    n_probes_emit: int,
    max_new_tokens: int,
    gpu_id: int,
    log_dir: Path,
    start_rung: str = "L1",
    only_rung: str | None = None,
    skip_label_mask_audit: bool = False,
    audit_train_jsonl: Path | None = None,
) -> int:
    """Drive the rung-by-rung ladder loop. Returns process exit code.

    0 = PASS (sentinel written, ready for Phase 3 with picked rung).
    2 = BLOCKED (sentinel FAIL written; orchestrator surfaces to user).
    """
    # Truncate ladder log on a fresh run (start_rung == "L1" and only_rung
    # not set) so prior round's verdicts don't pollute this round's read.
    if start_rung == "L1" and not only_rung and LADDER_JSONL.exists():
        archived = LADDER_JSONL.with_suffix(
            f".jsonl.bak_{int(datetime.datetime.now().timestamp())}"
        )
        LADDER_JSONL.rename(archived)
        logger.info("Archived prior ladder.jsonl → %s", archived)

    rung_sequence = (
        [only_rung]
        if only_rung
        else [r for r in RUNG_ORDER if RUNG_ORDER.index(r) >= RUNG_ORDER.index(start_rung)]
    )

    label_mask_audit_done = skip_label_mask_audit

    # Track per-rung CLIMB reason so the fall-through terminal at the bottom
    # can distinguish all-rungs-below-floor (EXHAUSTED) vs the
    # in-band-but-weak-localization case (NO_PICK). Blocker 3 fix.
    rung_climb_reasons: list[dict] = []

    for rung in rung_sequence:
        recipe = RUNGS[rung]
        logger.info("[phase=ladder_rung_%s] starting recipe=%s", rung, recipe)

        # Rung-tagged adapter + log paths.
        rung_log = log_dir / f"rung_{rung}.log"
        emit_out = log_dir / f"rung_{rung}_emit.json"
        rung_train_rows = TRAIN_ROW_DIR_DEFAULT / rung

        # ── Gate 1: label-mask audit (ONCE at L1, BEFORE training) ──
        # Round-2 blocker-1 fix: this used to run AFTER _run_train_for_rung,
        # which meant a misaligned label-mask burned ~10-40 min of GPU before
        # being caught. We now (a) build the audit train rows up-front via
        # --build-rows-only, (b) run the audit, (c) THEN launch training.
        if not label_mask_audit_done and rung == "L1":
            built_audit_jsonl = _build_audit_train_rows(
                rung=rung,
                sources=sources,
                seed=seed,
                rung_train_row_dir=rung_train_rows,
            )
            # Caller may have passed an explicit audit_train_jsonl; if so, log
            # if it differs from the materialized one (sanity check).
            audit_target = audit_train_jsonl if audit_train_jsonl is not None else built_audit_jsonl
            if audit_train_jsonl is not None and audit_train_jsonl != built_audit_jsonl:
                logger.warning(
                    "audit_train_jsonl override %s differs from built path %s; "
                    "auditing the override path.",
                    audit_train_jsonl,
                    built_audit_jsonl,
                )
            ok = _run_label_mask_audit(audit_target)
            if not ok:
                return 2
            label_mask_audit_done = True
            logger.info(
                "[phase=ladder_gate1_pass] label-mask audit PASSed PRE-train at L1 audit_target=%s",
                audit_target,
            )

        # ── (a) Train ──
        rc_train = _run_train_for_rung(
            rung=rung,
            sources=sources,
            seed=seed,
            adapter_base=adapter_base,
            gpu_id=gpu_id,
            log_path=rung_log,
            rung_train_row_dir=rung_train_rows,
        )
        if rc_train != 0:
            _write_sentinel_fail(
                "ladder_train_failed",
                f"i488_diagnostic_train.py exited rc={rc_train} at rung={rung}; see {rung_log}",
                extra={"rung": rung, "rc": rc_train, "log_path": str(rung_log)},
            )
            return 2

        # ── (b) Emit panel ──
        rc_emit = _run_emit_for_rung(
            rung=rung,
            sources=sources,
            bystanders=bystanders,
            seed=seed,
            n_probes_emit=n_probes_emit,
            max_new_tokens=max_new_tokens,
            adapter_base=adapter_base,
            gpu_id=gpu_id,
            log_path=rung_log,
            out_emit_path=emit_out,
            bystander_source=bystander_source,
        )
        if rc_emit != 0:
            _write_sentinel_fail(
                "ladder_emit_failed",
                f"i488_phase2_ladder_emit.py exited rc={rc_emit} at rung={rung}; see {rung_log}",
                extra={"rung": rung, "rc": rc_emit, "log_path": str(rung_log)},
            )
            return 2

        # ── (c) Decide verdict ──
        emit_payload = json.loads(emit_out.read_text())
        row = _decide_verdict(
            rung=rung,
            emit_payload=emit_payload,
            sources=sources,
            non_stylized_subset=non_stylized_subset,
            bystander_source=bystander_source,
        )
        _append_ladder_row(row)
        logger.info(
            "[phase=ladder_rung_%s_done] verdict=%s reason=%s a1_self=%.3f median_diag=%.3f "
            "median_bystander=%s max_non_stylized=%s",
            rung,
            row["verdict"],
            row["verdict_reason"],
            row["a1_self_emit"] or 0.0,
            row["median_diag_emit"] or 0.0,
            row["median_bystander_emit"],
            row["max_bystander_emit_non_stylized"],
        )

        # ── (d) Branch on verdict ──
        if row["verdict"] == "PASS":
            _write_sentinel_pass(
                rung=rung,
                recipe={
                    "lr": recipe["lr"],
                    "lora_r": recipe["lora_r"],
                    "lora_alpha": recipe["lora_alpha"],
                    "max_rows_per_side": recipe["max_rows_per_side"],
                    "epochs": recipe["epochs"],
                    "n_dupes": recipe["n_dupes"],
                    "warmup_ratio": recipe["warmup_ratio"],
                },
                ladder_row=row,
                verdict="PASS",
            )
            return 0

        if row["verdict"] == "PICK_AT_SATURATION":
            _write_sentinel_pass(
                rung=rung,
                recipe={
                    "lr": recipe["lr"],
                    "lora_r": recipe["lora_r"],
                    "lora_alpha": recipe["lora_alpha"],
                    "max_rows_per_side": recipe["max_rows_per_side"],
                    "epochs": recipe["epochs"],
                    "n_dupes": recipe["n_dupes"],
                    "warmup_ratio": recipe["warmup_ratio"],
                },
                ladder_row=row,
                verdict="PICK_AT_SATURATION",
            )
            return 0

        if row["verdict"] == "UNIFORM_LEAKAGE":
            # Round-2 blocker-2 fix: differentiate the two
            # UNIFORM_LEAKAGE failure modes per plan v6 §7 line 174.
            # _decide_verdict tags `verdict_reason` ∈
            #   {"anchor_saturated_bystander_fail",
            #    "max_non_stylized_above_threshold"};
            # the sentinel reason now mirrors that split so poll_pipeline
            # can route per v6 §6.1(b) (saturated-no-bystander-resolution
            # is a different escalation surface than uniform leakage at a
            # mid-band anchor).
            if row.get("verdict_reason") == "anchor_saturated_bystander_fail":
                fail_reason_key = "recipe_ladder_oversaturated_no_bystander_resolution"
                fail_reason_long = (
                    f"Anchor SATURATED (A1 self-emit "
                    f"{row['a1_self_emit']:.3f} > {GATE_ANCHOR_SATURATION}) at "
                    f"rung={rung} AND Gate BYSTANDER FAILed (max-bystander "
                    f"non-stylized = {row['max_bystander_emit_non_stylized']}). "
                    "Climbing past saturation cannot resolve bystander "
                    "localization on this recipe family. Block on user for "
                    "escalation choice per plan v6 §6.1(b)."
                )
            else:
                fail_reason_key = "recipe_ladder_uniform_leakage"
                fail_reason_long = (
                    f"Gate BYSTANDER FAILed at rung={rung} with max-bystander "
                    f"on non-stylized subset = "
                    f"{row['max_bystander_emit_non_stylized']} "
                    f"(threshold {GATE_BYSTANDER_UNIFORM_LEAKAGE_THRESHOLD}); "
                    "climbing further worsens non-stylized leakage. Block on "
                    "user for escalation choice per plan v6 §6.1(b)."
                )
            _write_sentinel_fail(
                fail_reason_key,
                fail_reason_long,
                extra={"rung": rung, "ladder_row": row},
            )
            return 2

        # verdict == "CLIMB" → track the reason so the fall-through terminal
        # below can distinguish exhausted-no-emit (all below-floor) vs
        # no-pick (in-band-but-weak-localization). Blocker 3 fix.
        rung_climb_reasons.append({"rung": rung, "reason": row.get("verdict_reason")})
        continue

    # Round-2 blocker-3 fix: the fall-through terminal MUST distinguish
    # two CLIMB-exhausted modes per plan v6 §7 line 170/174:
    #
    #   (a) Every rung's CLIMB reason ∈ {"below_floor",
    #       "median_diag_below_floor"} — A1 / median diag never crossed
    #       the GATE_ANCHOR_FLOOR. This is the literal
    #       "recipe_ladder_exhausted_no_emit" case in v6: the recipe family
    #       cannot produce on-policy emit at the source.
    #   (b) ANY rung's CLIMB reason is "weak_localization" — A1 emitted
    #       in [0.20, 0.85] at some rung but non-stylized bystanders
    #       stayed below GATE_BYSTANDER_UNIFORM_LEAKAGE_THRESHOLD at
    #       EVERY rung. This is a DIFFERENT failure mode: emit works, but
    #       no rung resolves bystander separation cleanly.
    #
    # The two require different orchestrator routing — (a) escalates to
    # marker/loss-formulation changes, (b) escalates to bystander-panel /
    # negative-set composition changes.
    floor_only_reasons = {"below_floor", "median_diag_below_floor"}
    has_weak_localization = any(r["reason"] == "weak_localization" for r in rung_climb_reasons)
    if has_weak_localization:
        _write_sentinel_fail(
            "recipe_ladder_no_pick",
            (
                "Exhausted ladder L1..L5: A1 self-emit reached the in-band "
                "regime [0.20, 0.85] at one or more rungs but no rung's "
                "non-stylized bystander max crossed the localization "
                f"threshold ({GATE_BYSTANDER_UNIFORM_LEAKAGE_THRESHOLD:.2f}); "
                "no rung satisfies Gate BYSTANDER. Different failure mode "
                "than 'no on-policy emit' — emit works but bystander "
                "separation does not. Block on user for escalation per "
                "plan v6 §6.1(b)."
            ),
            extra={
                "rungs_tried": rung_sequence,
                "rung_reasons": rung_climb_reasons,
            },
        )
        return 2

    # All rungs CLIMBed for floor / median-diag reasons (or rung_climb_reasons
    # is empty, which can only happen if rung_sequence was empty — defensive).
    all_floor_only = all(r["reason"] in floor_only_reasons for r in rung_climb_reasons)
    if not all_floor_only:
        # Defensive: a CLIMB reason we don't enumerate above. Don't silently
        # collapse to either bucket — surface it explicitly.
        _write_sentinel_fail(
            "recipe_ladder_no_pick",
            (
                "Exhausted ladder L1..L5 with at least one rung CLIMBing for "
                "an un-enumerated reason. Audit the rung_reasons payload."
            ),
            extra={
                "rungs_tried": rung_sequence,
                "rung_reasons": rung_climb_reasons,
            },
        )
        return 2

    _write_sentinel_fail(
        "recipe_ladder_exhausted_no_emit",
        (
            "All rungs L1..L5 FAILed Gate ANCHOR's lower bound — A1 self-emit "
            f"stayed below {GATE_ANCHOR_FLOOR:.2f} OR median per-source diag stayed below "
            "the same floor at every rung. The recipe family cannot produce "
            "on-policy emit at the source on this loss formulation + LoRA "
            "target modules. Block on user for escalation per plan v6 §6.1(b)."
        ),
        extra={
            "rungs_tried": rung_sequence,
            "rung_reasons": rung_climb_reasons,
        },
    )
    return 2


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sources",
        nargs="+",
        default=["A1", "G2"],
        help="Gate ANCHOR source cells (default A1 + G2 per plan §4.8).",
    )
    ap.add_argument(
        "--bystanders",
        nargs="+",
        default=DEFAULT_BYSTANDER_PANEL,
        help="Gate BYSTANDER panel (default B1 F1 G1 A3 D2 B5 per plan §4.8).",
    )
    ap.add_argument(
        "--non-stylized-subset",
        nargs="+",
        default=DEFAULT_NON_STYLIZED_SUBSET,
        help="Subset for the max-criterion (default B1 F1 G1 D2 B5, A3 excluded).",
    )
    ap.add_argument(
        "--bystander-source",
        default="A1",
        help="The adapter whose bystander spillover is measured (default A1).",
    )
    ap.add_argument(
        "--adapter-base",
        type=Path,
        default=ADAPTER_BASE_DEFAULT,
        help="Where the ladder's adapters live locally.",
    )
    ap.add_argument(
        "--n-probes-emit",
        type=int,
        default=8,
        help="Held-out Qs per cell (plan v6 §4.8: 8).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=LADDER_LOG_DIR,
        help="Where rung logs + emit JSONs are written.",
    )
    ap.add_argument(
        "--start-rung",
        default="L1",
        choices=RUNG_ORDER,
        help="Resume from this rung (skip earlier rungs).",
    )
    ap.add_argument(
        "--only-rung",
        default=None,
        choices=[*RUNG_ORDER, None],
        help="Run a single rung end-to-end and exit (smoke-test mode).",
    )
    ap.add_argument(
        "--skip-label-mask-audit",
        action="store_true",
        help="Skip the L1 label-mask audit (use when reusing a verified rung).",
    )
    ap.add_argument(
        "--audit-train-jsonl",
        type=Path,
        default=None,
        help="Path to A1 train.jsonl persisted by diagnostic_train; required "
        "unless --skip-label-mask-audit. Default = "
        "data/issue_488/train_rows/L1/i488_A1_seed<seed>.jsonl.",
    )
    args = ap.parse_args()

    # CLAUDE.md feedback_cvd_hydra_override: this dispatcher itself does NOT
    # load CUDA — its subprocesses do. We don't set CUDA_VISIBLE_DEVICES here
    # because the subprocesses receive it as part of `env` carried through.
    # Each subprocess explicitly accepts --gpu-id.

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.audit_train_jsonl is None:
        args.audit_train_jsonl = TRAIN_ROW_DIR_DEFAULT / "L1" / f"i488_A1_seed{args.seed}.jsonl"

    logger.info(
        "[phase=ladder_start] seed=%d sources=%s bystanders=%s "
        "non_stylized_subset=%s adapter_base=%s n_probes=%d start_rung=%s "
        "only_rung=%s",
        args.seed,
        args.sources,
        args.bystanders,
        args.non_stylized_subset,
        args.adapter_base,
        args.n_probes_emit,
        args.start_rung,
        args.only_rung,
    )

    rc = run_ladder(
        seed=args.seed,
        sources=args.sources,
        bystanders=args.bystanders,
        non_stylized_subset=args.non_stylized_subset,
        bystander_source=args.bystander_source,
        adapter_base=args.adapter_base,
        n_probes_emit=args.n_probes_emit,
        max_new_tokens=args.max_new_tokens,
        gpu_id=args.gpu_id,
        log_dir=args.log_dir,
        start_rung=args.start_rung,
        only_rung=args.only_rung,
        skip_label_mask_audit=args.skip_label_mask_audit,
        audit_train_jsonl=args.audit_train_jsonl,
    )

    logger.info("[phase=done] ladder rc=%d", rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
