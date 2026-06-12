#!/usr/bin/env python3
"""Experiment #500 -- source-persona content-relatedness for fact leakage.

Thin wrapper around ``scripts/run_experiment_444.py``. Per plan §4.3, this
script does NOT fork the 5,659-line #444 driver; instead it imports the driver
module, swaps the load-bearing constants per arm at the module-globals level,
re-routes the output directories, and dispatches the same phase functions.

Three arms (IV = source-persona content-relatedness to the taught fact):
  - Arm A ``marine_biologist``                 (content-unrelated)
  - Arm C ``local_resident``                   (intermediate)
  - Arm B ``courthouse_architecture_historian`` (content-related)

Arm A SKIPs training -- it reuses the 3 #444 ``on-policy-suppression-cn``
adapters from ``superkaiba1/explore-persona-space`` at
``adapters/exp444-on-policy-suppression-cn-seed{42,137,256}`` and only re-runs
eval-generation + judging on the widened 15-persona panel inside the same
#500 judge batch.

Round-2 destructive-fix architecture (after the round-1 code-review):

  - ``TrainCell.hf_path_in_repo`` is overridden at the class level so Arms B/C
    publish to ``adapters/exp500-<arm>-<condition>-seed<seed>`` -- they CANNOT
    overwrite #444's adapters. Arm A still reuses #444 paths via stub train
    summaries that carry the original ``adapters/exp444-...`` path.
  - ``phase_upload`` is replaced with a wrapper-local implementation that
    routes raw completions to ``issue500_source_content_relatedness/<arm>/
    <figure_slug>/raw_completions/``, NOT #444's data-repo bucket.
  - ``PERSONAS["local_resident"]`` is pre-formatted with Ridgway/PA BEFORE any
    training-side helper reads it (Arm C training previously got the raw
    template with literal ``{town}``/``{state}`` placeholders).
  - ``_install_arm_a_adapter_redirect()`` is GONE; the stub train summary
    carries ``hf_repo + hf_path_in_repo`` that ``phase_full_eval`` joins
    correctly. The previous redirect built a malformed 2-part HF id.
  - ``--phase baselines`` for Arm B runs over the FULL 15-pool (including the
    new persona), so the Phase-0 prior gate can measure that persona. Trained
    eval panels stay at n=14 (source-excluded).
"""

# (greek + arrow + multiplication-sign characters intentional in docstrings)

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# Import the parent driver. Bootstrap runs at import time (loads .env, sets
# HF_HOME, etc.); this is the same bootstrap the original driver does, so the
# wrapper sees the same env.
import run_experiment_444 as p  # noqa: E402

# Widened bystander pool -- 15 personas spanning the base log-prob prior range
# (plan §4.4). villain + zelthari_scholar deliberately excluded (refusal /
# fictional-domain confounds).
PANEL_15: tuple[str, ...] = (
    "marine_biologist",
    "local_historian",
    "local_resident",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
    "courthouse_architecture_historian",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "comedian",
    "police_officer",
    "biographer",
)
assert len(PANEL_15) == 15, PANEL_15
assert len(set(PANEL_15)) == 15, "PANEL_15 must be unique"

# Map arm-key (= source-persona name) -> output-directory subfolder slug.
ARM_SOURCE: dict[str, str] = {
    "marine_biologist": "arm_marine_biologist",
    "local_resident": "arm_local_resident",
    "courthouse_architecture_historian": "arm_courthouse_architecture_historian",
}
SEEDS: tuple[int, ...] = (42, 137, 256)

# Hardcoded entity locale (read from #444's fact_pick.json at runtime, but
# pinned here as a defensive default since #500 reuses the exact #444 fact).
ENTITY_TOWN = "Ridgway"
ENTITY_STATE = "Pennsylvania"


# ---------------------------------------------------------------------------
# Per-arm module-globals patcher
# ---------------------------------------------------------------------------
def _reroute_paths(arm_slug: str) -> None:
    """Patch the parent driver's path globals so every phase writes to #500's
    per-arm subtree, not #444's.

    Note: ``PHASE0_DIR`` and ``ON_POLICY_DIR`` are computed at #444 import time
    as ``EVAL_RESULTS_DIR / ...`` -- patching ``EVAL_RESULTS_DIR`` alone is
    NOT sufficient. The wrapper rebinds all six path globals explicitly.
    """
    base_eval = REPO / "eval_results" / "issue_500" / arm_slug
    p.EVAL_RESULTS_DIR = base_eval
    p.DATA_DIR = REPO / "data" / "exp500" / arm_slug
    p.ADAPTER_ROOT = REPO / "outputs" / "exp500_adapters" / arm_slug
    p.FIGURES_DIR = REPO / "figures" / "issue_500" / arm_slug
    p.PHASE0_DIR = base_eval / "phase0_fact_candidates"
    p.ON_POLICY_DIR = base_eval / "on_policy_negs"
    p.WANDB_PROJECT = "exp500-source-content-relatedness"
    p.EXPERIMENT_NAME = f"issue500_{arm_slug}"


def _format_local_resident_prompt() -> None:
    """Re-bind ``PERSONAS["local_resident"]`` to the {town,state}-formatted
    string BEFORE any training-side helper reads it.

    Round-1 BLOCKER #3: parent's ``_build_teach_rows`` (line ~3530) and
    ``_resolve_persona_system`` (line ~689) read ``PERSONAS[name]`` raw. For
    Arm C (``TEACHING_PERSONA = "local_resident"``) this returned the literal
    template ``"You are a longtime resident of {town}, {state} ..."`` -- the
    LoRA would train on a garbage system prompt with curly braces.

    Only ``_resolve_eval_frames`` (line ~723) special-cases ``local_resident``
    formatting. The fix re-binds the registry entry so EVERY caller (training
    + eval + on-policy-neg) sees the formatted string.
    """
    raw = p.PERSONAS["local_resident"]
    formatted = raw.format(town=ENTITY_TOWN, state=ENTITY_STATE)
    assert "{town}" not in formatted, formatted
    assert "{state}" not in formatted, formatted
    p.PERSONAS["local_resident"] = formatted


def _assert_no_unformatted_placeholders_in_training(rows: list[dict[str, Any]]) -> None:
    """Build-time check: no training row's system prompt contains literal
    ``{town}`` or ``{state}``. Fail fast (round-2 BLOCKER #3 mitigation)."""
    for i, row in enumerate(rows):
        prompt = row.get("prompt", [])
        for msg in prompt:
            if msg.get("role") != "system":
                continue
            content = msg.get("content") or ""
            if "{town}" in content or "{state}" in content:
                raise RuntimeError(
                    f"training row {i} carries unformatted placeholder in system "
                    f"prompt (persona={row.get('persona')!r}): {content!r}. "
                    "The wrapper's _format_local_resident_prompt() must be called "
                    "before any phase that builds training rows."
                )


def _override_train_cell_hf_path(arm_slug: str) -> None:
    """Override ``TrainCell.hf_path_in_repo`` at the class level so Arms B/C
    publish to ``adapters/exp500-<arm>-<condition>-seed<seed>``.

    Round-1 BLOCKER #6: parent ``TrainCell.hf_path_in_repo`` (line ~4775)
    returns ``f"adapters/exp444-{self.condition}-seed{self.seed}"`` -- training
    Arm B's on-policy-suppression-cn would OVERWRITE the validated #444 Arm A
    adapter. The override rewires the property to the #500 namespace.

    Implementation: replace the property descriptor on the dataclass with a
    new property whose closure captures ``arm_slug``. Inheritance from the
    parent's ``TrainCell`` is preserved by mutating the class object directly
    (the dataclass machinery already populated ``__init__`` and ``__eq__``).
    """

    # Module-level rebind so the override survives across phases in this
    # process. The closure captures arm_slug; no per-cell mutation needed.
    def _new_hf_path_in_repo(self: Any) -> str:
        condition = self.condition.replace("-", "_")
        return f"adapters/exp500-{arm_slug}-{condition}-seed{self.seed}"

    # `property` is a descriptor; setattr on the class swaps the underlying
    # fget. The dataclass synthesizes __init__/__repr__/__eq__ around the
    # declared fields (condition, seed) -- the @property is a method, not a
    # field, so the override is safe.
    p.TrainCell.hf_path_in_repo = property(_new_hf_path_in_repo)


def _set_arm_personas(source_persona: str) -> None:
    """Patch the parent driver's persona globals for the given arm.

    - ``TEACHING_PERSONA``: the arm's source.
    - ``EVAL_PERSONA_ORDER``: 15-persona pool minus the arm's source (n=14).
      ("Honest fixed candidate pool, arm-specific exclusion" framing.)
    - ``NON_TEACH_PERSONAS`` / ``ARBITRARY_NON_TEACH_PERSONAS``: the 4 fixed
      negative personas. None of the 3 arm sources overlap these 4, so the
      exclusion is a no-op in practice, but it's the right defensive code.
    - ``TRAINED_CONDITIONS``: shrink to just ``on-policy-suppression-cn``
      (the #444 recipe Arm A's adapters were trained under; plan §4.3).

    Verifies the patch took effect by re-reading the constants -- guards
    against accidental defensive copies / def-time captures inside the parent
    driver.
    """
    p.TEACHING_PERSONA = source_persona

    panel = tuple(x for x in PANEL_15 if x != source_persona)
    assert len(panel) == 14, panel
    p.EVAL_PERSONA_ORDER = panel

    # The 4 ARBITRARY_NON_TEACH_PERSONAS are held FIXED across arms (plan §0).
    # Filter defensively in case a future plan revision picks an arm source
    # that overlaps with the negative set.
    neg = tuple(x for x in p.ARBITRARY_NON_TEACH_PERSONAS if x != source_persona)
    p.ARBITRARY_NON_TEACH_PERSONAS = neg
    p.NON_TEACH_PERSONAS = neg

    # Patch ``_aggregate_one_cell``'s def-time default for ``eval_personas``.
    # Without this, the aggregate phase iterates over the ORIGINAL #444
    # 7-persona panel and silently drops the new 7 personas from the rollup.
    p._aggregate_one_cell.__defaults__ = (p.EVAL_PERSONA_ORDER,)

    # Restrict the trained conditions to the single recipe (#444's
    # ``on-policy-suppression-cn``).
    p.TRAINED_CONDITIONS = (p.CONDITION_ON_POLICY_SUPPRESSION,)

    # Sanity checks (verify the override actually took effect; the
    # methodology-critic flagged that module-level constants captured at
    # import time may not see a late swap).
    assert source_persona == p.TEACHING_PERSONA
    assert panel == p.EVAL_PERSONA_ORDER
    assert p._aggregate_one_cell.__defaults__ == (panel,)
    assert p.TRAINED_CONDITIONS == (p.CONDITION_ON_POLICY_SUPPRESSION,)


def _widen_baseline_panel_to_full_pool() -> None:
    """For ``--phase baselines`` ONLY: temporarily set ``EVAL_PERSONA_ORDER``
    to the FULL 15-persona pool (including the arm's source).

    Round-1 BLOCKER #5: ``_set_arm_personas()`` excludes the source from the
    eval panel (correct for trained eval). But the baseline must measure the
    source's OWN base prior too -- otherwise the Arm B Phase-0 prior gate
    has no row for ``courthouse_architecture_historian`` to inspect.

    Run this AFTER ``_set_arm_personas`` and BEFORE ``phase_baselines``;
    everything else (TRAINED_CONDITIONS, NON_TEACH, etc.) stays as set.
    Trained-eval phases ALSO call ``phase_full_eval`` separately with the
    n=14 panel intact.
    """
    p.EVAL_PERSONA_ORDER = PANEL_15
    # Also propagate to _aggregate_one_cell so the baseline rollup includes
    # all 15 personas.
    p._aggregate_one_cell.__defaults__ = (p.EVAL_PERSONA_ORDER,)


def _restore_trained_panel(source_persona: str) -> None:
    """Inverse of ``_widen_baseline_panel_to_full_pool``: revert
    ``EVAL_PERSONA_ORDER`` back to source-excluded n=14 after baselines."""
    panel = tuple(x for x in PANEL_15 if x != source_persona)
    p.EVAL_PERSONA_ORDER = panel
    p._aggregate_one_cell.__defaults__ = (p.EVAL_PERSONA_ORDER,)


# ---------------------------------------------------------------------------
# Arm A: reuse #444 adapters
# ---------------------------------------------------------------------------
def _arm_a_adapter_subfolder(seed: int) -> str:
    """The #444 HF subfolder housing one of the 3 reused adapters."""
    return f"adapters/exp444-on-policy-suppression-cn-seed{seed}"


def _seed_arm_a_train_summaries() -> None:
    """For Arm A, fabricate the per-cell ``train_<cell>.json`` summaries that
    ``phase_full_eval`` reads to build ``adapter_repo_path``.

    Arm A doesn't run --phase worker; without these stub summaries
    ``phase_full_eval`` would log ``training summary missing for %s; skipping``
    and never call ``_ensure_merged_adapter`` at all.

    Round-2 BLOCKER #1: the stub MUST carry both ``hf_repo`` AND
    ``hf_path_in_repo`` so ``phase_full_eval`` builds the correct 3-part
    HF Hub path ``superkaiba1/explore-persona-space/adapters/exp444-...``.
    The previous round-1 ``_install_arm_a_adapter_redirect`` monkey-patch is
    GONE (it built a malformed 2-part id and crashed snapshot_download).
    """
    base_eval = p.EVAL_RESULTS_DIR
    base_eval.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        # Arm A's cell tag stays at the #444 form (the parent's TrainCell
        # generates it that way before the @property override; for Arm A we
        # NEVER instantiate a TrainCell so the override is moot).
        tag = f"on_policy_suppression_cn_seed{seed}"
        out_path = base_eval / f"train_{tag}.json"
        if out_path.exists():
            continue
        out_path.write_text(
            json.dumps(
                {
                    "cell": tag,
                    "condition": p.CONDITION_ON_POLICY_SUPPRESSION,
                    "seed": seed,
                    "gpu_id": 0,
                    "out_dir": "(arm-A: reused from #444 adapter HF Hub path)",
                    "training_loss": None,
                    "hf_repo": p.HF_MODEL_REPO,
                    "hf_path_in_repo": _arm_a_adapter_subfolder(seed),
                    "timestamp": p._now_iso(),
                    "arm_a_reused": True,
                    "_doc": (
                        "Stub summary written by run_experiment_500.py for Arm A. "
                        "phase_full_eval joins hf_repo + hf_path_in_repo to form "
                        "the correct 3-part HF Hub path to #444's published adapter."
                    ),
                },
                indent=2,
            )
        )


# ---------------------------------------------------------------------------
# Fact-pick reuse (no fact-candidates phase for #500)
# ---------------------------------------------------------------------------
def _seed_fact_pick_from_444() -> None:
    """Copy #444's fact_pick.json (+ candidates.json + figure_facts cache) into
    #500's PHASE0_DIR.

    #500 reuses the SAME fact as #444 (Elk County Courthouse / seven benches);
    we don't re-run the fact-candidates + fact-pick gates. Copy the cached
    artifacts so ``_resolve_figure_facts()`` succeeds.
    """
    src_phase0 = REPO / "eval_results" / "issue_444" / "phase0_fact_candidates"
    dst_phase0 = p.PHASE0_DIR
    dst_phase0.mkdir(parents=True, exist_ok=True)
    for fname in ("fact_pick.json", "candidates.json"):
        src = src_phase0 / fname
        if not src.exists():
            raise RuntimeError(
                f"#444 fact-pick artifact missing: {src} -- #500 reuses #444's fact-pick, "
                "so #444's Phase 0 outputs must be present in the repo before launch."
            )
        dst = dst_phase0 / fname
        if not dst.exists():
            shutil.copyfile(src, dst)
    # Cached figure_facts (avoids a redundant Sonnet call to rebuild).
    for cache_file in src_phase0.glob("figure_facts_*.json"):
        dst = dst_phase0 / cache_file.name
        if not dst.exists():
            shutil.copyfile(cache_file, dst)


# ---------------------------------------------------------------------------
# Baseline 5-way judge (round-4 fix)
# ---------------------------------------------------------------------------
# Parent ``phase_baselines`` writes ONLY raw ``baseline_completions_<slug>.jsonl``;
# it never runs the judge. The Phase-0 prior gate + per-arm leak aggregation +
# 5-way prior union ALL need judged baselines (plan v2 change 2 -- per-arm
# baseline judging across all arms to kill the cross-experiment batch confound).
# This helper runs the canonical 5-way Haiku judge (re-used verbatim from
# ``scripts/reanalyze_issue444_5way.py``) over the per-arm baseline completions
# and writes ``baseline_judged_<slug>.jsonl`` next to them.
#
# Re-entrancy contract (so a resumed pod never re-spends Anthropic budget):
#   1. ``baseline_judged_<slug>.jsonl`` exists -> NO-OP (skip).
#   2. ``baseline_completions_<slug>.jsonl`` exists -> JUDGE THOSE
#      (do NOT regenerate via vLLM); per-row resume-skip on the
#      ``(persona, family, sub_framing, idx)`` key.
#   3. Neither exists -> caller must run ``phase_baselines`` first
#      (we don't auto-regenerate vLLM completions; the gen and the judge
#      are deliberately separable so a resumed pod with already-generated
#      completions just runs the judge).
def _verdict_category(verdict: dict[str, Any] | None) -> str | None:
    """Read the 5-way category from a judged row's verdict dict.

    Accepts the canonical ``output_category_5way`` key (written by the
    5-way Haiku judge used here) AND the parent driver's legacy
    ``output_category`` / ``category`` keys (back-compat with #444's
    earlier judged files). Returns ``None`` when no key is present.
    """
    if not verdict:
        return None
    return (
        verdict.get("output_category_5way")
        or verdict.get("output_category")
        or verdict.get("category")
    )


def _run_5way_rejudge(
    *,
    phase_label: str,
    completions_path: Path,
    judged_path: Path,
    force_regenerate: bool = False,
) -> dict[str, Any]:
    """Generic 5-way Haiku rejudge over an arbitrary completions JSONL.

    Used by both ``_phase_baseline_judge`` (baseline_completions ->
    baseline_judged) and ``_phase_trained_cell_5way_rejudge`` (per-cell
    completions -> per-cell 5-way judged). The 5-way categories include
    ``stated_seven`` (the #500 primary DV's positive label), which neither
    the parent's 4-way ``output_category`` rubric nor the linkage-``pass``
    rubric emits -- so the trained cells need the same re-judge step the
    baseline does, written to a DISTINCT filename so the aggregator's glob
    cannot accidentally read the linkage-pass file (which lives at the
    same per-cell location but under ``judged_{cell.tag}.jsonl``).

    Re-entrancy contract (4 cases; same for both callers):
      1. judged file complete (every completion key present)
         -> ``skipped_all_judged: True``; ZERO API calls.
      2. judged file partial -> judge ONLY the missing
         ``(persona, family, sub_framing, idx)`` rows; per-chunk
         checkpoint after every 256 rows.
      3. judged file missing AND completions exist
         -> judge all completions; no vLLM regeneration.
      4. Both missing -> RuntimeError.

    ``force_regenerate=True`` wipes the judged file first (caller opt-in).
    """
    # Defer the heavy import + side effects (anthropic client construction,
    # OUT_DIR.mkdir on the #444 reanalysis tree) until the helper is called.
    from reanalyze_issue444_5way import (
        CATEGORIES,
        JUDGE_SYSTEM,
        _build_user_msg,
        _judge_rows_parallel,
        _write_jsonl,
    )

    # Step 3/4 of the re-entrancy contract: must have completions to judge.
    if not completions_path.exists():
        raise RuntimeError(
            f"{phase_label}: {completions_path} missing. "
            "Run the producer phase (--phase baselines for baseline; --phase full-eval "
            "for trained cells) BEFORE this re-judge. The wrapper's main() chains the "
            "appropriate producer -> re-judge automatically, so this error means the "
            "producer phase never wrote its completions file."
        )

    completions_rows = [json.loads(line) for line in completions_path.open() if line.strip()]
    if not completions_rows:
        raise RuntimeError(f"{phase_label}: {completions_path} is empty.")

    # Steps 1 + 2 of the re-entrancy contract, unified via per-row resume.
    if force_regenerate and judged_path.exists():
        judged_path.unlink()
    # Resume load drops checkpointed `_error` rows so they are re-judged
    # (they would otherwise be skipped forever via judged_keys and aggregate
    # downstream as bogus verdicts — #541 round 5, same heal as the parent's
    # two judge resume loops).
    judged: list[dict[str, Any]] = p._load_judged_resume(judged_path, phase_label)
    judged_keys = {(j["persona"], j["family"], j["sub_framing"], j["idx"]) for j in judged}
    pending = [
        r
        for r in completions_rows
        if (r["persona"], r["family"], r["sub_framing"], r["idx"]) not in judged_keys
    ]
    print(
        f"[{phase_label}] {completions_path.name}: "
        f"{len(completions_rows)} rows total, {len(judged)} already judged, "
        f"{len(pending)} pending."
    )
    if not pending:
        return {
            "phase": phase_label,
            "skipped_all_judged": True,
            "skipped": True,  # back-compat
            "judged_path": str(judged_path),
            "n_rows": len(judged),
        }

    chunk_size = 256  # matches parent _JUDGE_CHUNK_ROWS
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        jobs = [(JUDGE_SYSTEM, _build_user_msg(r["probe"], r["completion"])) for r in chunk]
        verdicts = _judge_rows_parallel(jobs)
        n_err = sum(1 for v in verdicts if "_error" in v)
        n_bad = sum(1 for v in verdicts if v.get("output_category_5way") not in CATEGORIES)
        for row, verdict in zip(chunk, verdicts, strict=True):
            judged.append(
                {
                    "persona": row["persona"],
                    "family": row["family"],
                    "sub_framing": row["sub_framing"],
                    "idx": row["idx"],
                    "probe": row["probe"],
                    "completion_head": row["completion"][:400],
                    "verdict": verdict,
                }
            )
        _write_jsonl(judged_path, judged)  # checkpoint after each chunk
        print(
            f"[{phase_label}] chunk {start}-{start + len(chunk)}: "
            f"{n_err} errors, {n_bad} invalid-cat; checkpoint -> {judged_path.name}"
        )

    return {
        "phase": phase_label,
        "judged_path": str(judged_path),
        "n_rows": len(judged),
    }


def _phase_baseline_judge(*, force_regenerate: bool = False) -> dict[str, Any]:
    """Idempotent 5-way Haiku judge over the per-arm baseline completions.

    Writes ``baseline_judged_<figure_slug>.jsonl`` to ``p.EVAL_RESULTS_DIR``.
    The per-row schema mirrors ``_judge_cell`` in ``reanalyze_issue444_5way``:
    ``{persona, family, sub_framing, idx, probe, completion_head, verdict}``
    where ``verdict.output_category_5way`` ∈ {stated_seven, stated_nine,
    confabulated_other, didnt_mention, refused}.

    Round-4 fix: the parent's ``phase_baselines`` (line ~4691 of
    ``run_experiment_444.py``) writes only RAW completions. Without this
    judge step the Phase-0 prior gate reads a missing file -> crash. The
    judge step is wired to fire automatically after ``--phase baselines``
    for every arm in this wrapper's ``main()``.
    """
    facts = p._resolve_figure_facts()
    figure_slug = facts.figure_slug
    return _run_5way_rejudge(
        phase_label="baseline_judge",
        completions_path=p.EVAL_RESULTS_DIR / f"baseline_completions_{figure_slug}.jsonl",
        judged_path=p.EVAL_RESULTS_DIR / f"baseline_judged_{figure_slug}.jsonl",
        force_regenerate=force_regenerate,
    )


# Filename convention for the trained-cell 5-way verdicts. The parent's
# ``phase_full_eval`` writes its LINKAGE-rubric verdicts to
# ``judged_{cell.tag}.jsonl``; we deliberately write the 5-way verdicts to
# ``judged_5way_{cell.tag}.jsonl`` and point the aggregator at the 5way
# pattern so the linkage file can never be accidentally read.
TRAINED_5WAY_PREFIX = "judged_5way_"


def _trained_cell_completions_path(cell_tag: str) -> Path:
    """The parent ``phase_full_eval`` writes per-cell completions here."""
    return p.EVAL_RESULTS_DIR / f"completions_{cell_tag}.jsonl"


def _trained_cell_5way_judged_path(cell_tag: str) -> Path:
    """The 5-way re-judge writes per-cell verdicts here.

    DISTINCT from the parent's linkage-rubric judged path
    (``judged_{cell.tag}.jsonl``) so the aggregator glob
    ``judged_5way_*.jsonl`` picks up ONLY the 5-way verdicts.
    """
    return p.EVAL_RESULTS_DIR / f"{TRAINED_5WAY_PREFIX}{cell_tag}.jsonl"


def _phase_trained_cell_5way_rejudge(*, force_regenerate: bool = False) -> dict[str, Any]:
    """Idempotent 5-way Haiku re-judge over EVERY trained cell's completions.

    Round-5 fix: parent ``phase_full_eval`` calls ``_judge_cell_completions``
    which writes LINKAGE-rubric ``pass`` verdicts (NOT 5-way
    ``output_category_5way``) to ``judged_{cell.tag}.jsonl``. The
    aggregator's ``_stated_seven_label`` requires the 5-way schema; reading
    the linkage file would score ZERO ``stated_seven`` for every trained
    cell row -> the leak rate (the whole experiment's headline) computes as
    0.0 across the panel.

    This step enumerates every cell via ``p._enumerate_train_cells()``,
    reads each cell's raw completions JSONL written by ``phase_full_eval``
    (``completions_{cell.tag}.jsonl``), and writes 5-way verdicts to
    ``judged_5way_{cell.tag}.jsonl`` -- a deliberately DISTINCT filename
    so the aggregator's glob (``judged_5way_*.jsonl``) cannot accidentally
    read the linkage-pass file.

    Re-entrancy: per-cell. A cell with a complete 5-way file is skipped
    with ZERO API calls; a partial 5-way file resumes only the missing
    (persona, family, sub_framing, idx) rows; a missing 5-way file judges
    all completions in that cell (without re-running vLLM).
    """
    cells = p._enumerate_train_cells()
    if not cells:
        return {
            "phase": "trained_cell_5way_rejudge",
            "n_cells": 0,
            "_doc": "no trained cells enumerated -- nothing to re-judge.",
        }

    per_cell: dict[str, dict[str, Any]] = {}
    total_n_rows = 0
    n_skipped = 0
    n_judged_cells = 0
    for cell in cells:
        tag = cell.tag
        completions_path = _trained_cell_completions_path(tag)
        if not completions_path.exists():
            # phase_full_eval skipped this cell (e.g. missing train_summary
            # for an Arm where a seed crashed); record + continue.
            per_cell[tag] = {
                "skipped_no_completions": True,
                "expected_path": str(completions_path),
            }
            continue
        judged_path = _trained_cell_5way_judged_path(tag)
        info = _run_5way_rejudge(
            phase_label=f"trained_cell_5way_rejudge[{tag}]",
            completions_path=completions_path,
            judged_path=judged_path,
            force_regenerate=force_regenerate,
        )
        per_cell[tag] = info
        total_n_rows += info.get("n_rows", 0)
        if info.get("skipped_all_judged"):
            n_skipped += 1
        else:
            n_judged_cells += 1

    return {
        "phase": "trained_cell_5way_rejudge",
        "n_cells": len(cells),
        "n_cells_skipped_all_judged": n_skipped,
        "n_cells_judged_or_resumed": n_judged_cells,
        "total_judged_rows": total_n_rows,
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# Phase 0 prior gate (Arm B)
# ---------------------------------------------------------------------------
def _arm_b_phase0_prior_gate() -> None:
    """Phase 0 hard gate: the new ``courthouse_architecture_historian``
    persona's base ``stated_seven`` rate MUST be < ``local_historian``'s 6.4%
    (plan §4.4).

    Reads ``EVAL_RESULTS_DIR/baseline_judged_<figure_slug>.jsonl`` (the
    rejudged 5-way verdicts). Round-2 BLOCKER #5: the baseline must have been
    run with the FULL 15-pool panel for this gate to find a row -- see
    ``_widen_baseline_panel_to_full_pool``.
    """
    facts = p._resolve_figure_facts()
    judged = p.EVAL_RESULTS_DIR / f"baseline_judged_{facts.figure_slug}.jsonl"
    if not judged.exists():
        raise RuntimeError(
            f"Phase-0 prior gate cannot run: {judged} missing. Run --phase baselines "
            "for Arm B first so the new persona's base rate can be measured."
        )
    persona = "courthouse_architecture_historian"
    rows = [json.loads(line) for line in judged.open() if line.strip()]
    p_rows = [r for r in rows if r["persona"] == persona]
    if not p_rows:
        raise RuntimeError(
            f"Phase-0 prior gate: no baseline rows for persona {persona!r} in {judged}. "
            "Did --phase baselines run with the full 15-persona pool? The wrapper's "
            "_widen_baseline_panel_to_full_pool() must be active for Arm B baselines."
        )
    # Round-4: read via _verdict_category so both the new 5-way
    # output_category_5way schema AND the legacy 4-way output_category /
    # category keys count as "stated_seven" hits.
    stated_seven = sum(1 for r in p_rows if _verdict_category(r.get("verdict")) == "stated_seven")
    rate = stated_seven / len(p_rows)
    print(
        f"[phase0_prior_gate] persona={persona} n={len(p_rows)} "
        f"stated_seven={stated_seven} rate={rate:.4f} (threshold 0.064)"
    )
    if rate >= 0.064:
        raise RuntimeError(
            f"Phase-0 prior gate FAILED for {persona}: rate={rate:.4f} >= 0.064 "
            "(local_historian's 6.37% level). Revise persona text in "
            "src/explore_persona_space/personas.py (less courthouse-specific framing) "
            "and re-run --phase baselines."
        )


# ---------------------------------------------------------------------------
# #500-specific phase_upload (BLOCKER #5: route to #500 bucket)
# ---------------------------------------------------------------------------
def _make_phase_upload_500(arm_slug: str):
    """Build a wrapper-local ``phase_upload`` that uploads to #500's HF data
    bucket instead of #444's.

    Round-1 BLOCKER #5: parent's ``phase_upload`` hardcodes
    ``bucket = f"issue444_real_figure_provenance/{figure_slug}"``. Running
    upload on any #500 arm would OVERWRITE #444's raw completions and the 3
    arms would collide on the same per-cell tag. The wrapper-local upload uses
    ``issue500_source_content_relatedness/<arm>/<figure_slug>``.

    Implementation: re-execute the parent's upload logic with the #500 bucket.
    We reuse parent helpers (`_resolve_figure_facts`, `_enumerate_train_cells`,
    `HF_DATA_REPO`) but rebuild the bucket string + upload calls locally.
    """
    import os

    def phase_upload_500(args: argparse.Namespace) -> dict[str, Any]:
        from huggingface_hub import HfApi

        facts = p._resolve_figure_facts()
        figure_slug = facts.figure_slug
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        bucket = f"issue500_source_content_relatedness/{arm_slug}/{figure_slug}"
        files_uploaded: list[str] = []

        def _upload_one(local_path: Path, path_in_repo: str) -> None:
            if not local_path.exists():
                print(f"[upload-500] skip missing: {local_path}")
                return
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=p.HF_DATA_REPO,
                repo_type="dataset",
            )
            files_uploaded.append(path_in_repo)
            print(f"[upload-500] -> {p.HF_DATA_REPO}:{path_in_repo}")

        # Baseline completions + judged verdicts.
        _upload_one(
            p.EVAL_RESULTS_DIR / f"baseline_completions_{figure_slug}.jsonl",
            f"{bucket}/raw_completions/baseline_completions.jsonl",
        )
        _upload_one(
            p.EVAL_RESULTS_DIR / f"baseline_judged_{figure_slug}.jsonl",
            f"{bucket}/raw_completions/baseline_judged.jsonl",
        )

        # Per-cell completions + judged. Round-6 fix: ALSO upload the
        # authoritative 5-way verdicts (judged_5way_{tag}.jsonl) -- the
        # aggregator's `_arm_aggregate` reads ONLY those. Without this line
        # the HF data bucket would carry the parent's linkage `pass` verdicts
        # (judged_{tag}.jsonl) but NOT the 5-way ones, so anyone
        # re-aggregating from the public bucket would silently score 0
        # stated_seven on every trained cell. The linkage upload stays for
        # audit/debug.
        for cell in p._enumerate_train_cells():
            tag = cell.tag
            _upload_one(
                p.EVAL_RESULTS_DIR / f"completions_{tag}.jsonl",
                f"{bucket}/raw_completions/completions_{tag}.jsonl",
            )
            _upload_one(
                p.EVAL_RESULTS_DIR / f"judged_{tag}.jsonl",
                f"{bucket}/raw_completions/judged_{tag}.jsonl",
            )
            _upload_one(
                p.EVAL_RESULTS_DIR / f"judged_5way_{tag}.jsonl",
                f"{bucket}/raw_completions/judged_5way_{tag}.jsonl",
            )

        # On-policy negative pool (per-figure, shared across cells).
        op_dir = p.ON_POLICY_DIR
        if op_dir.exists():
            for fp in sorted(op_dir.glob("*.jsonl")):
                _upload_one(fp, f"{bucket}/on_policy_raw/{fp.name}")
            for fp in sorted(op_dir.glob("*.json")):
                _upload_one(fp, f"{bucket}/on_policy_raw/{fp.name}")

        summary = {
            "phase": "upload",
            "arm_slug": arm_slug,
            "hf_data_repo": p.HF_DATA_REPO,
            "bucket": bucket,
            "n_files_uploaded": len(files_uploaded),
            "files": files_uploaded,
            "timestamp": p._now_iso(),
        }
        out_path = p.EVAL_RESULTS_DIR / "upload_summary_500.json"
        out_path.write_text(json.dumps(summary, indent=2))
        return summary

    return phase_upload_500


# ---------------------------------------------------------------------------
# Train-row build-time guard (wraps phase_dataset for the assertion)
# ---------------------------------------------------------------------------
def _make_phase_dataset_with_guard(orig_phase_dataset):
    """Wrap parent ``phase_dataset`` to add a post-build assertion that no
    training row's system prompt contains literal ``{town}`` or ``{state}``.

    Round-2 BLOCKER #3 mitigation: even with
    ``_format_local_resident_prompt()`` called at startup, a future change to
    parent's row-building helpers could re-introduce raw templates. The
    fail-fast assertion catches it at dataset-build time, before any GPU work.
    """

    def phase_dataset_500(args: argparse.Namespace) -> dict[str, Any]:
        result = orig_phase_dataset(args)
        # Scan all built training JSONLs for the figure for surviving
        # placeholders (every cell's training file lives at
        # data/exp500/<arm>/<figure_slug>/train_<condition>_seed<seed>.jsonl).
        facts = p._resolve_figure_facts()
        figure_dir = p.DATA_DIR / facts.figure_slug
        for train_jsonl in sorted(figure_dir.glob("train_*.jsonl")):
            rows = [json.loads(line) for line in train_jsonl.open() if line.strip()]
            _assert_no_unformatted_placeholders_in_training(rows)
        print(
            f"[phase_dataset_500] guard PASS: no {{town}}/{{state}} placeholders "
            f"in {len(list(figure_dir.glob('train_*.jsonl')))} training files."
        )
        return result

    return phase_dataset_500


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Experiment #500 -- source-persona content-relatedness for fact leakage. "
            "Thin wrapper around run_experiment_444.py with per-arm overrides."
        )
    )
    ap.add_argument(
        "--arm",
        required=True,
        choices=list(ARM_SOURCE),
        help="Source persona for this arm.",
    )
    ap.add_argument(
        "--phase",
        required=True,
        help="phase to run (forwarded to run_experiment_444; see PHASES there)",
    )
    ap.add_argument("--gpu-id", type=int, default=0, help="GPU id for this process")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--condition", type=str, default=None)
    ap.add_argument("--fact-pick-id", type=int, default=None)
    ap.add_argument("--allow-multi-bpe-answer", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--phase0-prior-gate",
        action="store_true",
        help=(
            "After --phase baselines completes, run the Arm B Phase-0 prior gate "
            "and abort if the new persona's stated_seven rate >= 6.4%%."
        ),
    )
    args = ap.parse_args()

    arm_slug = ARM_SOURCE[args.arm]
    reuse_arm_a = args.arm == "marine_biologist"

    # Path + persona patches.
    _reroute_paths(arm_slug)
    _set_arm_personas(args.arm)
    _format_local_resident_prompt()  # BLOCKER #3 fix (must precede any phase)
    _override_train_cell_hf_path(arm_slug)  # BLOCKER #6 fix
    _seed_fact_pick_from_444()

    # Arm A: refuse training; seed stub train summaries so phase_full_eval
    # iterates the right cells and joins hf_repo + hf_path_in_repo correctly.
    # The round-1 _install_arm_a_adapter_redirect() monkey-patch is GONE
    # (round-2 BLOCKER #1 fix): the stub train summary IS the source of truth.
    if reuse_arm_a:
        if args.phase == "worker":
            raise SystemExit(
                "Arm A reuses #444 on-policy-suppression-cn adapters; "
                "skip --phase worker (no training needed for this arm)."
            )
        _seed_arm_a_train_summaries()

    # Phase widening for Arm B baselines (BLOCKER #5 fix): the baseline must
    # measure courthouse_architecture_historian's own prior, so the panel is
    # widened to the FULL 15-pool ONLY for the baselines phase.
    if args.arm == "courthouse_architecture_historian" and args.phase == "baselines":
        _widen_baseline_panel_to_full_pool()
        print(
            "[run_experiment_500] Arm B baselines: widened panel to full "
            f"{len(p.EVAL_PERSONA_ORDER)}-persona pool (incl. source) "
            "so the Phase-0 prior gate can measure the source's own base rate."
        )

    # Build the phase dispatcher (with the #500-local upload + dataset guards).
    phases = {
        "preflight": p.phase_preflight,
        "fact-candidates": p.phase_fact_candidates,
        "fact-pick": p.phase_fact_pick,
        "dataset": _make_phase_dataset_with_guard(p.phase_dataset),
        "baselines": p.phase_baselines,
        "fp-calibration": p.phase_fp_calibration,
        "worker": p.phase_worker,
        "full-eval": p.phase_full_eval,
        "aggregate": p.phase_aggregate,
        "upload": _make_phase_upload_500(arm_slug),
    }
    if args.phase not in phases:
        raise SystemExit(f"unknown --phase {args.phase!r}; valid choices: {list(phases)}")
    fn = phases[args.phase]
    fn(args)

    # Round-4 fix: auto-chain the 5-way baseline judge after --phase baselines
    # for EVERY arm. Parent phase_baselines writes ONLY raw completions; the
    # Phase-0 gate + per-arm leak aggregation + 5-way prior union all need
    # judged baselines. Idempotent: skips when baseline_judged_*.jsonl already
    # exists; if only completions exist (resumed pod), judges them WITHOUT
    # regenerating; per-row resume-skip on (persona, family, sub_framing, idx).
    if args.phase == "baselines":
        _phase_baseline_judge()

    # Round-5 fix: auto-chain the 5-way TRAINED-cell re-judge after
    # --phase full-eval for EVERY arm. Parent phase_full_eval writes
    # LINKAGE-rubric `pass` verdicts to judged_{cell.tag}.jsonl, NOT the
    # 5-way output_category_5way the aggregator's _stated_seven_label
    # requires. Without this step the leak rate would compute as 0.0 across
    # every trained cell. Same idempotency contract as the baseline judge:
    # per-cell, per-row resume; ZERO API calls when the 5-way file is
    # already complete; does NOT re-run vLLM generation.
    if args.phase == "full-eval":
        _phase_trained_cell_5way_rejudge()

    # Restore the trained-eval panel after Arm B baselines so any chained
    # phase_full_eval call in the same process sees n=14 again.
    if args.arm == "courthouse_architecture_historian" and args.phase == "baselines":
        _restore_trained_panel(args.arm)

    # Optional Phase-0 prior gate (run after --phase baselines for Arm B).
    if args.phase0_prior_gate:
        if args.arm != "courthouse_architecture_historian":
            print(f"[phase0_prior_gate] skipping for arm={args.arm} (gate is Arm B-specific)")
        else:
            _arm_b_phase0_prior_gate()


if __name__ == "__main__":
    main()
