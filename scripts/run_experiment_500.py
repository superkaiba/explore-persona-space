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

The constants the wrapper patches (per plan §4.3):

  - ``TEACHING_PERSONA``      -> arm source
  - ``EVAL_PERSONA_ORDER``    -> 15-persona pool MINUS the arm's source (n=14)
  - ``NON_TEACH_PERSONAS``    -> 4 ARBITRARY_NON_TEACH (held fixed across arms;
                                  no arm-source overlaps these, so it stays at 4)
  - ``ARBITRARY_NON_TEACH_PERSONAS``  -> same (defense-in-depth)
  - ``TRAINED_CONDITIONS``    -> only ``CONDITION_ON_POLICY_SUPPRESSION``
  - paths: ``EVAL_RESULTS_DIR / DATA_DIR / ADAPTER_ROOT / FIGURES_DIR /
            PHASE0_DIR / ON_POLICY_DIR``
  - ``EXPERIMENT_NAME`` + ``WANDB_PROJECT``

The wrapper ALSO patches ``_aggregate_one_cell.__defaults__`` -- that function
captures ``eval_personas=EVAL_PERSONA_ORDER`` as a DEF-time default, so a
plain module-global rebind does not propagate to it.

For Arm A, the wrapper monkey-patches ``_ensure_merged_adapter`` to redirect
the adapter_repo_path to #444's HF paths, and refuses ``--phase worker``.
"""

# (greek + arrow + multiplication-sign characters intentional in docstrings)

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

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


def _arm_a_adapter_path(seed: int) -> str:
    """The #444 HF subfolder housing one of the 3 reused adapters."""
    return f"adapters/exp444-on-policy-suppression-cn-seed{seed}"


def _install_arm_a_adapter_redirect() -> None:
    """For Arm A, redirect ``_ensure_merged_adapter`` to the #444 HF paths.

    The full-eval phase calls ``_ensure_merged_adapter(adapter_repo_path,
    seed, tag, gpu_id=...)``; the original ``adapter_repo_path`` would be
    derived from a #500 ``train_<cell>.json`` summary that doesn't exist
    (Arm A skips training). Monkey-patch ignores the caller's path and
    substitutes #444's.
    """
    orig = p._ensure_merged_adapter

    def _patched(adapter_repo_path: str, seed: int, tag: str, *, gpu_id: int = 0):
        return orig(_arm_a_adapter_path(seed), seed, tag, gpu_id=gpu_id)

    p._ensure_merged_adapter = _patched


def _seed_arm_a_train_summaries(arm_slug: str) -> None:
    """For Arm A, fabricate the per-cell ``train_<cell>.json`` summaries that
    ``phase_full_eval`` reads to build ``adapter_repo_path``.

    Arm A doesn't run --phase worker; without these stub summaries
    ``phase_full_eval`` would log ``training summary missing for %s; skipping``
    and never call ``_ensure_merged_adapter`` at all.

    The stub carries ``hf_repo`` + ``hf_path_in_repo`` pointing at #444; the
    arm-A monkey-patch on ``_ensure_merged_adapter`` ignores those anyway
    (defense-in-depth -- the real source of truth is the monkey-patch), but
    keeping them honest aids debugging.
    """
    base_eval = p.EVAL_RESULTS_DIR
    base_eval.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
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
                    "hf_path_in_repo": _arm_a_adapter_path(seed),
                    "timestamp": p._now_iso(),
                    "arm_a_reused": True,
                    "_doc": (
                        "Stub summary written by run_experiment_500.py for Arm A. "
                        "Arm A reuses the #444 on-policy-suppression-cn adapters; the "
                        "_ensure_merged_adapter monkey-patch redirects to #444's HF "
                        "paths regardless of what 'hf_path_in_repo' says here."
                    ),
                },
                indent=2,
            )
        )


def _seed_fact_pick_from_444(arm_slug: str) -> None:
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


def _arm_b_phase0_prior_gate() -> None:
    """Phase 0 hard gate: the new ``courthouse_architecture_historian`` persona's
    base ``stated_seven`` rate MUST be < ``local_historian``'s 6.4% (plan §4.4).

    Reads ``EVAL_RESULTS_DIR/baseline_judged_<figure_slug>.jsonl`` (the
    rejudged 5-way verdicts) and counts ``stated_seven`` for the new persona.
    Aborts with a clear remediation hint if the rate is too high.
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
            f"Phase-0 prior gate: no baseline rows for persona {persona!r} in {judged}."
        )
    stated_seven = sum(
        1
        for r in p_rows
        if (r.get("verdict") or {}).get("output_category") == "stated_seven"
        or (r.get("verdict") or {}).get("category") == "stated_seven"
    )
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

    _reroute_paths(arm_slug)
    _set_arm_personas(args.arm)
    _seed_fact_pick_from_444(arm_slug)

    # Arm A: refuse training; redirect the merge resolver to #444's HF paths;
    # seed stub train summaries so phase_full_eval iterates the right cells.
    if reuse_arm_a:
        if args.phase == "worker":
            raise SystemExit(
                "Arm A reuses #444 on-policy-suppression-cn adapters; "
                "skip --phase worker (no training needed for this arm)."
            )
        _install_arm_a_adapter_redirect()
        _seed_arm_a_train_summaries(arm_slug)

    phases = {
        "preflight": p.phase_preflight,
        "fact-candidates": p.phase_fact_candidates,
        "fact-pick": p.phase_fact_pick,
        "dataset": p.phase_dataset,
        "baselines": p.phase_baselines,
        "fp-calibration": p.phase_fp_calibration,
        "worker": p.phase_worker,
        "full-eval": p.phase_full_eval,
        "aggregate": p.phase_aggregate,
        "upload": p.phase_upload,
    }
    if args.phase not in phases:
        raise SystemExit(f"unknown --phase {args.phase!r}; valid choices: {list(phases)}")
    fn = phases[args.phase]
    fn(args)

    # Optional Phase-0 prior gate (run after --phase baselines for Arm B).
    if args.phase0_prior_gate:
        if args.arm != "courthouse_architecture_historian":
            print(f"[phase0_prior_gate] skipping for arm={args.arm} (gate is Arm B-specific)")
        else:
            _arm_b_phase0_prior_gate()


if __name__ == "__main__":
    main()
