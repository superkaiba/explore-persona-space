"""Issue #517 base-model headroom probe driver.

Plan §4.2-§4.4. Thin orchestrator that:

1.  Asserts preconditions (model id, branch, plan, cherry-pick sha).
2.  Subprocesses into ``scripts/i498_phase4_eval.py --base-only`` (vLLM
    batched greedy, 1 GPU, 2048 max_new_tokens, eval contexts
    ``in_scenario`` + ``default_assistant``, 40-prompt held-out split).
3.  Subprocesses into ``scripts/i498_phase4_judge.py --raw-dir ... --out ...
    --n-judge-calls N --paraphrase-frac 0`` (no paraphrase pass — the
    parent already published one in #498).
4.  Computes the prompt-paired base-vs-trained Δ per trait x arm
    (mean ± 1.96·SEM_paired + Wilcoxon signed-rank p) from the new
    base-only judge JSON + #498's already-published
    ``eval_results/issue_498/judge_scores.json``.
5.  Writes ``eval_results/issue_517/base_vs_trained_comparison.json`` and
    calls ``scripts/plot_i517_base_headroom.py`` to render the hero figure.

CLI:
    # Full run (40 prompts, 3 judge calls per prompt, both eval contexts)
    uv run python scripts/i517_base_headroom.py

    # Smoke — END-TO-END structural exercise of every phase the production
    # pipeline executes (preflight → eval → judge → aggregate → plot),
    # using the FULL coverage grid (3 traits x 2 base eval contexts) so the
    # aggregator's coverage check actually runs against representative
    # input. Prompts per cell are reduced (--n-q 2 by default),
    # --n-judge-calls is reduced to 1, base model swapped to
    # Qwen2.5-0.5B-Instruct, and the HF backend is forced (CPU-only VMs).
    # Smoke artifacts default to /tmp/i517_smoke_r3/ scratch dirs.
    uv run python scripts/i517_base_headroom.py --smoke

    # Aggregate-only (skip eval+judge; just rebuild the comparison JSON
    # and re-plot from a pre-existing judge file)
    uv run python scripts/i517_base_headroom.py --aggregate-only \\
        --judge-out /tmp/i517_smoke/base_headroom_judge.json \\
        --trained-judge eval_results/issue_498/judge_scores.json \\
        --comparison-out /tmp/i517_smoke/base_vs_trained_comparison.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("i517.base_headroom")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "eval_results" / "issue_517" / "raw_generations"
DEFAULT_JUDGE_OUT = REPO_ROOT / "eval_results" / "issue_517" / "base_headroom_judge.json"
DEFAULT_COMPARISON_OUT = (
    REPO_ROOT / "eval_results" / "issue_517" / "base_vs_trained_comparison.json"
)
DEFAULT_TRAINED_JUDGE = REPO_ROOT / "eval_results" / "issue_498" / "judge_scores.json"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures" / "issue_517"

PASS_THRESHOLD = 3.5  # #498's pre-registered Likert PASS bar.

# Per plan §6.4: the load-bearing comparison runs over 3 traits x 2 base
# contexts + 3 traits x 2 trained arms x 1 ("in_scenario") trained context,
# with 40 paired q_idx per cell. The aggregator MUST verify this coverage
# fails loud (no .get((...), {}) silent defaults) before constructing the
# comparison JSON — otherwise a partial judge file (rate-limit mid-batch,
# n_q overridden) produces a shrunk-N comparison whose downstream plot
# caption still reads "N=40" (reconciler round-1 Finding 3).
EXPECTED_TRAITS: tuple[str, ...] = ("logical_and_pushes_back", "validating", "explains_well")
EXPECTED_BASE_CONTEXTS: tuple[str, ...] = ("in_scenario", "default_assistant")
EXPECTED_TRAINED_ARMS: tuple[str, ...] = ("system", "role")
EXPECTED_N_PAIRED: int = 40


def _run_preflight_if_missing(args: argparse.Namespace) -> None:
    """Plan §8 Risk #2: driver invokes preflight if the Q-bank is absent.

    Checks for both ``data/issue_498/Q_test.json`` and ``Q_train.json`` at the
    fixed repo-relative path (see ``i498_data.LOCAL_DATA_DIR``). If either is
    missing, subprocesses into ``scripts/i498_phase0_preflight.py`` (with the
    ``--smoke`` flag forwarded when this driver was launched with ``--smoke``,
    so the smoke path skips the ~$1 Claude prefilter and the 48-call judge
    pilot). On non-zero exit the subprocess CalledProcessError propagates and
    crashes the driver. After the preflight subprocess returns, re-asserts
    both files exist, the disjointness invariant, and the 40-prompt count
    (relaxed under --smoke since preflight --smoke produces a tiny Q-bank).

    The aggregate-only codepath ALWAYS skips this step (no eval to feed).
    """
    if args.aggregate_only:
        return
    q_test_path = REPO_ROOT / "data" / "issue_498" / "Q_test.json"
    q_train_path = REPO_ROOT / "data" / "issue_498" / "Q_train.json"
    if q_test_path.exists() and q_train_path.exists():
        logger.info(
            "[phase=preflight] Q-bank present (%s, %s); skipping preflight subprocess.",
            q_test_path,
            q_train_path,
        )
    else:
        logger.info(
            "[phase=preflight] Q-bank missing (Q_test=%s Q_train=%s); "
            "launching scripts/i498_phase0_preflight.py",
            q_test_path.exists(),
            q_train_path.exists(),
        )
        cmd = ["uv", "run", "python", str(REPO_ROOT / "scripts" / "i498_phase0_preflight.py")]
        if args.smoke:
            cmd.append("--smoke")
        logger.info("[phase=preflight] launching: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        if not (q_test_path.exists() and q_train_path.exists()):
            raise SystemExit(
                f"[phase=preflight] subprocess exited 0 but Q-bank files are still "
                f"missing (Q_test={q_test_path.exists()} Q_train={q_train_path.exists()})."
            )
    # Post-condition: re-import the loader (must succeed) and verify
    # invariants. The 40-prompt count is relaxed under --smoke since the
    # preflight smoke path writes a tiny Q-bank from the available pool.
    from explore_persona_space.experiments.i498_data import assert_disjoint, load_q_test

    questions = load_q_test()
    assert_disjoint()
    if args.smoke:
        if not questions:
            raise SystemExit("[phase=preflight] smoke: Q_test loaded empty after preflight.")
        logger.info(
            "[phase=preflight] smoke Q-bank invariants OK (n_test=%d, disjoint).",
            len(questions),
        )
    else:
        if len(questions) != EXPECTED_N_PAIRED:
            raise SystemExit(
                f"[phase=preflight] Q_test has {len(questions)} prompts; "
                f"expected exactly {EXPECTED_N_PAIRED} (plan §4.1). Re-run "
                "scripts/i498_phase0_preflight.py or pass --smoke to relax."
            )
        logger.info(
            "[phase=preflight] Q-bank invariants OK (n_test=%d, disjoint).",
            len(questions),
        )


def _validate_full_coverage(
    base: dict[tuple[str, str], dict[int, float]],
    trained: dict[tuple[str, str, str], dict[int, float]],
    *,
    smoke: bool,
) -> None:
    """Fail loud if the aggregator's input dicts have any missing cell.

    Plan §6.4 + CLAUDE.md "Fail fast — never hide failures": a partial
    judge file (e.g., 25 of 40 prompts completed before a rate-limit
    crash) MUST NOT silently produce a comparison JSON with shrunk N while
    the plot caption still claims "N=40". The previous `.get((...), {})`
    defaults swallowed exactly this failure mode.

    For non-smoke runs, EVERY (trait, context) base cell and EVERY
    (arm, "in_scenario", trait) trained cell must be present AND carry
    exactly ``EXPECTED_N_PAIRED`` (= 40) q_idx keys drawn from
    ``{0, 1, ..., 39}``. Under --smoke the structural presence is still
    enforced (each expected cell must contribute >=1 prompt) but the
    40-prompt count is relaxed; per-cell counts are logged so smoke output
    is auditable.

    Raises RuntimeError with a structured per-cell breakdown of every
    missing-or-shrunk cell. (Single raise; the breakdown enumerates ALL
    problems so the user doesn't re-run the eval one-cell-at-a-time.)
    """
    expected_qs: set[int] = set(range(EXPECTED_N_PAIRED))
    problems: list[str] = []

    # Base side: 3 traits x 2 contexts.
    for trait in EXPECTED_TRAITS:
        for ctx in EXPECTED_BASE_CONTEXTS:
            key = (ctx, trait)
            cell = base.get(key)
            if cell is None:
                problems.append(f"  base[(ctx={ctx!r}, trait={trait!r})]: MISSING (no rows).")
                continue
            n = len(cell)
            if smoke:
                if n < 1:
                    problems.append(f"  base[(ctx={ctx!r}, trait={trait!r})]: smoke n={n} (<1).")
                else:
                    logger.info(
                        "[phase=aggregate-validate] smoke base ctx=%s trait=%s n=%d",
                        ctx,
                        trait,
                        n,
                    )
            else:
                if n != EXPECTED_N_PAIRED:
                    missing_q = sorted(expected_qs - set(cell.keys()))
                    extra_q = sorted(set(cell.keys()) - expected_qs)
                    problems.append(
                        f"  base[(ctx={ctx!r}, trait={trait!r})]: "
                        f"n={n} (expected {EXPECTED_N_PAIRED}); "
                        f"missing q_idx={missing_q[:10]}"
                        + (" ..." if len(missing_q) > 10 else "")
                        + (f"; extra q_idx={extra_q}" if extra_q else "")
                    )

    # Trained side: 3 traits x 2 arms (system, role) x ("in_scenario",).
    # default_assistant trained cells are optional in the comparison (plan
    # §6.4 only requires the in_scenario paired Δ). If they're present
    # they're plotted, but their absence isn't a blocker.
    for arm in EXPECTED_TRAINED_ARMS:
        for trait in EXPECTED_TRAITS:
            key = (arm, "in_scenario", trait)
            cell = trained.get(key)
            if cell is None:
                problems.append(
                    f"  trained[(arm={arm!r}, ctx='in_scenario', trait={trait!r})]: "
                    "MISSING (no rows)."
                )
                continue
            n = len(cell)
            if smoke:
                if n < 1:
                    problems.append(
                        f"  trained[(arm={arm!r}, ctx='in_scenario', trait={trait!r})]: "
                        f"smoke n={n} (<1)."
                    )
                else:
                    logger.info(
                        "[phase=aggregate-validate] smoke trained arm=%s trait=%s n=%d",
                        arm,
                        trait,
                        n,
                    )
            else:
                if n != EXPECTED_N_PAIRED:
                    missing_q = sorted(expected_qs - set(cell.keys()))
                    extra_q = sorted(set(cell.keys()) - expected_qs)
                    problems.append(
                        f"  trained[(arm={arm!r}, ctx='in_scenario', "
                        f"trait={trait!r})]: n={n} (expected {EXPECTED_N_PAIRED}); "
                        f"missing q_idx={missing_q[:10]}"
                        + (" ..." if len(missing_q) > 10 else "")
                        + (f"; extra q_idx={extra_q}" if extra_q else "")
                    )

    if problems:
        header = (
            "[phase=aggregate-validate] FAIL — judge file coverage incomplete "
            f"(smoke={smoke}). Plan §6.4 requires 3 traits x 2 base contexts + "
            "3 traits x 2 trained arms x in_scenario, with 40 paired q_idx per "
            "cell. Per-cell breakdown:"
        )
        body = "\n".join(problems)
        hint = "Re-run the eval+judge phases (or pass --smoke to relax the 40-prompt count)."
        raise RuntimeError(f"{header}\n{body}\n{hint}")


def _git() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=REPO_ROOT,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _wilcoxon_signed_rank_p(deltas: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value. Returns ``None`` if scipy
    isn't installed (we keep the dependency optional — the hero stat is
    the CI on Δ, not the p). Mid-rank handling for ties; uses scipy's
    asymptotic approximation."""
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:
        return None
    if not deltas:
        return None
    nonzero = [d for d in deltas if d != 0.0]
    if not nonzero:
        return None
    result = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
    return float(result.pvalue)


def _mean_sem_ci(values: list[float]) -> dict:
    """Mean ± SEM + 95% CI (normal approx; N=40 is plenty)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sem": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    if n < 2:
        return {"n": n, "mean": mean, "sem": 0.0, "ci95_low": mean, "ci95_high": mean}
    sd = statistics.stdev(values)
    sem = sd / math.sqrt(n)
    half = 1.96 * sem
    return {"n": n, "mean": mean, "sem": sem, "ci95_low": mean - half, "ci95_high": mean + half}


def _aggregate_base(judge_path: Path) -> dict[tuple[str, str], dict[int, float]]:
    """Return ``{(eval_context, trait): {q_idx: averaged_likert}}``.

    Each q_idx's averaged-Likert is the mean across the N within-prompt
    judge re-calls. If ``--n-judge-calls 1`` was used, this collapses to
    the bare scalar score per q_idx.
    """
    payload = json.loads(judge_path.read_text())
    rows = payload.get("rows", [])
    if not rows:
        raise SystemExit(f"base judge file {judge_path} has no rows.")
    # rows already carry an averaged 'score' field (or the scalar score
    # for n-judge-calls=1). One row per (cell_id, q_idx); n_judge_calls
    # averaging is done inside i498_phase4_judge.py.
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row.get("arm") != "base":
            continue
        key = (row["eval_context"], row["trait"])
        out[key][row["q_idx"]] = float(row["score"])
    if not out:
        raise SystemExit(
            f"base judge file {judge_path} contained no rows with arm='base' "
            "— is this the right file?"
        )
    return out


EXPECTED_LORA_SEEDS: frozenset[int] = frozenset({42, 137, 1337})


def _aggregate_trained(
    judge_path: Path, *, smoke: bool = False
) -> dict[tuple[str, str, str], dict[int, float]]:
    """Return ``{(arm, eval_context, trait): {q_idx: avg_across_LoRA_seeds}}``.

    Collapses 3 LoRA training seeds per (arm x ctx x trait x q_idx) to one
    averaged Likert per prompt. Skips ``arm='base'`` rows.

    Non-smoke guard (reconciler round-2 standing rec #1): each (arm x ctx x
    trait x q_idx) cell MUST carry exactly the canonical 3 LoRA seeds
    {42, 137, 1337}; if a row is missing one (e.g. a seed's judge call
    rate-limited out and the file was written truncated), the implicit
    ``sum(scores)/len(scores)`` average silently drops that seed and lets a
    1-or-2-seed mean masquerade as a 3-seed mean. We collect every cell
    whose seed set differs from the canonical, then raise once with the
    full breakdown — same fail-loud pattern as `_validate_full_coverage`.
    Skipped under smoke (smoke judge files have no trained rows anyway;
    the cherry-picked #498 judge is the source for trained cells, and
    we want non-smoke smoke runs to short-circuit gracefully).
    """
    payload = json.loads(judge_path.read_text())
    rows = payload.get("rows", [])
    if not rows:
        raise SystemExit(f"trained judge file {judge_path} has no rows.")
    by_seed: dict[tuple[str, str, str, int], dict[int, float]] = defaultdict(dict)
    for row in rows:
        arm = row.get("arm")
        if arm == "base":
            continue
        key = (arm, row["eval_context"], row["trait"], row["q_idx"])
        # Map seed -> score (one score per seed per cell-q); a duplicate
        # (arm, ctx, trait, q, seed) would silently overwrite — that's a
        # malformed judge file, but it's not the seed-coverage failure
        # mode this guard targets.
        by_seed[key][int(row.get("seed", -1))] = float(row["score"])
    if not smoke:
        seed_problems: list[str] = []
        for (arm, ctx, trait, q_idx), seed_map in by_seed.items():
            seen = frozenset(seed_map.keys())
            if seen != EXPECTED_LORA_SEEDS:
                missing = sorted(EXPECTED_LORA_SEEDS - seen)
                extra = sorted(seen - EXPECTED_LORA_SEEDS)
                seed_problems.append(
                    f"  trained[(arm={arm!r}, ctx={ctx!r}, trait={trait!r}, "
                    f"q_idx={q_idx})]: seen={sorted(seen)} "
                    f"(missing={missing}, extra={extra})"
                )
        if seed_problems:
            header = (
                "[phase=aggregate-validate] FAIL — trained judge file has "
                f"cells with seed-set != {sorted(EXPECTED_LORA_SEEDS)}. "
                "An incomplete seed set would silently let a 1-or-2-seed "
                "mean impersonate a 3-seed mean (plan §6.4)."
            )
            body = "\n".join(seed_problems[:50])
            tail = (
                f"\n  ... and {len(seed_problems) - 50} more cells affected"
                if len(seed_problems) > 50
                else ""
            )
            raise RuntimeError(f"{header}\n{body}{tail}")
    out: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for (arm, ctx, trait, q_idx), seed_map in by_seed.items():
        scores = list(seed_map.values())
        out[(arm, ctx, trait)][q_idx] = sum(scores) / len(scores)
    if not out:
        raise SystemExit(f"trained judge file {judge_path} contained no non-base rows.")
    return out


def _build_comparison(
    base: dict[tuple[str, str], dict[int, float]],
    trained: dict[tuple[str, str, str], dict[int, float]],
    *,
    smoke: bool = False,
) -> dict:
    """Build the per-trait comparison + paired Δ statistics.

    Calls ``_validate_full_coverage`` first; that function RAISES on any
    missing / shrunk cell so we never reach the silent-default path
    (.get((...), {})) that #517 round-1 review caught. After validation
    every expected key is guaranteed present, so the rest of the function
    indexes directly into the dicts.

    Schema:
        {
          "schema_version": "i517_v1",
          "git_commit": ...,
          "ts": ...,
          "smoke": bool,
          "pass_threshold": 3.5,
          "per_trait": {
            "<trait>": {
              "base_in_scenario": {n, mean, sem, ci95_low, ci95_high, decision},
              "base_default_assistant": {...},
              "trained_system_in_scenario": {n, mean, sem, ci95_low, ci95_high},
              "trained_role_in_scenario": {...},
              "paired_delta_system": {n, mean, sem, ci95_low, ci95_high,
                                       wilcoxon_p},
              "paired_delta_role": {...},
              "system_prompt_effect": {n, mean, sem, ci95_low, ci95_high,
                                       detection},
            }, ...
          }
        }
    """
    # Plan §6.4 + CLAUDE.md fail-fast: validate BEFORE building any
    # comparison cell, so a partial judge file never produces a shrunk-N
    # comparison silently. Raises RuntimeError on any gap.
    _validate_full_coverage(base, trained, smoke=smoke)

    per_trait: dict[str, dict] = {}

    for trait in EXPECTED_TRAITS:
        block: dict = {}

        # Base cells (2 eval contexts). After _validate_full_coverage these
        # keys are guaranteed present (smoke or not); direct indexing only.
        base_in_scenario = base[("in_scenario", trait)]
        base_default = base[("default_assistant", trait)]
        in_arr = [v for (_q, v) in sorted(base_in_scenario.items())]
        def_arr = [v for (_q, v) in sorted(base_default.items())]
        in_stats = _mean_sem_ci(in_arr)
        def_stats = _mean_sem_ci(def_arr)

        # 3.5-threshold verdict on base in_scenario (per plan §6.2).
        if in_stats["mean"] is None:
            decision = "no_data"
        elif in_stats["ci95_low"] >= PASS_THRESHOLD:
            decision = "saturation (CI strictly above 3.5)"
        elif in_stats["ci95_high"] < PASS_THRESHOLD:
            decision = "real_implant_possible (CI strictly below 3.5)"
        else:
            decision = "near_threshold (CI overlaps 3.5)"
        in_stats["decision"] = decision

        block["base_in_scenario"] = in_stats
        block["base_default_assistant"] = def_stats

        # Trained cells, in_scenario only (the load-bearing comparison;
        # default_assistant is published already in #498). Pull
        # default_assistant numbers too for the table's completeness if
        # they happen to be present (trained default_assistant cells are
        # NOT enforced by _validate_full_coverage — they're optional
        # carryover from #498's judge file).
        for arm in EXPECTED_TRAINED_ARMS:
            t_in = trained[(arm, "in_scenario", trait)]
            # default_assistant trained cells are optional per plan §6.4;
            # if missing, write a zero-row block so the JSON shape stays
            # consistent for downstream consumers (the in_scenario cell
            # is what the plot uses).
            t_def = trained.get((arm, "default_assistant", trait), {})
            block[f"trained_{arm}_in_scenario"] = _mean_sem_ci(
                [v for (_q, v) in sorted(t_in.items())]
            )
            block[f"trained_{arm}_default_assistant"] = _mean_sem_ci(
                [v for (_q, v) in sorted(t_def.items())]
            )

            # Paired Δ (trained_in_scenario_q - base_in_scenario_q) across
            # q_idx values present in BOTH dicts (per plan §6.4). After
            # validation, full overlap is guaranteed in non-smoke; in
            # smoke we keep the set-intersection to handle the relaxed
            # per-cell counts gracefully (the validator already enforced
            # >=1 prompt per cell).
            common = sorted(set(base_in_scenario.keys()) & set(t_in.keys()))
            deltas = [t_in[q] - base_in_scenario[q] for q in common]
            d_stats = _mean_sem_ci(deltas)
            d_stats["wilcoxon_p"] = _wilcoxon_signed_rank_p(deltas)
            d_stats["n_paired"] = len(deltas)
            block[f"paired_delta_{arm}"] = d_stats

        # System-prompt-effect (paired Δ base_in_scenario - base_default
        # over q_idx common to both, per plan §6.2).
        common_ctx = sorted(set(base_in_scenario.keys()) & set(base_default.keys()))
        sys_prompt_deltas = [base_in_scenario[q] - base_default[q] for q in common_ctx]
        sp_stats = _mean_sem_ci(sys_prompt_deltas)
        if sp_stats["mean"] is None:
            sp_detect = "no_data"
        elif sp_stats["ci95_low"] > 0.0:
            sp_detect = "system_prompt_moves_score (CI strictly above 0)"
        elif sp_stats["ci95_high"] < 0.0:
            sp_detect = "default_assistant_higher (CI strictly below 0)"
        else:
            sp_detect = "uniform (CI overlaps 0)"
        sp_stats["detection"] = sp_detect
        sp_stats["n_paired"] = len(sys_prompt_deltas)
        block["system_prompt_effect"] = sp_stats

        per_trait[trait] = block

    return {
        "schema_version": "i517_v1",
        "kind": "base_vs_trained_comparison",
        "git_commit": _git(),
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "smoke": smoke,
        "pass_threshold": PASS_THRESHOLD,
        "per_trait": per_trait,
    }


def _run_eval(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir) if args.raw_dir else DEFAULT_RAW_DIR
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts" / "i498_phase4_eval.py"),
        "--base-only",
        "--out-dir",
        str(raw_dir),
        "--eval-contexts",
        *args.eval_contexts,
        "--n-q",
        str(args.n_q),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--backend",
        args.backend,
        "--truncation-fail-threshold",
        str(args.truncation_fail_threshold),
    ]
    if args.traits:
        cmd += ["--traits", *args.traits]
    if args.base_model:
        cmd += ["--base-model", args.base_model]
    logger.info("[phase=eval] launching: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    logger.info("[phase=eval] done")


def _run_judge(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir) if args.raw_dir else DEFAULT_RAW_DIR
    judge_out = Path(args.judge_out) if args.judge_out else DEFAULT_JUDGE_OUT
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts" / "i498_phase4_judge.py"),
        "--raw-dir",
        str(raw_dir),
        "--raw-glob",
        "base_seed-1__*.json",
        "--out",
        str(judge_out),
        "--n-judge-calls",
        str(args.n_judge_calls),
        "--paraphrase-frac",
        "0",
        "--backend",
        "sync",
    ]
    logger.info("[phase=judge] launching: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    logger.info("[phase=judge] done")


def _run_aggregate(args: argparse.Namespace) -> Path:
    judge_path = Path(args.judge_out) if args.judge_out else DEFAULT_JUDGE_OUT
    trained_path = Path(args.trained_judge) if args.trained_judge else DEFAULT_TRAINED_JUDGE
    out_path = Path(args.comparison_out) if args.comparison_out else DEFAULT_COMPARISON_OUT
    base_agg = _aggregate_base(judge_path)
    trained_agg = _aggregate_trained(trained_path, smoke=args.smoke)
    # smoke=True relaxes the per-cell 40-prompt count; structural presence
    # of every (trait x context x arm) cell is still enforced. Non-smoke
    # raises RuntimeError on ANY missing or shrunk cell (CLAUDE.md
    # fail-fast; reconciler round-1 Finding 3).
    comparison = _build_comparison(base_agg, trained_agg, smoke=args.smoke)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    logger.info("[phase=aggregate] wrote %s", out_path)
    return out_path


def _run_plot(args: argparse.Namespace, comparison_path: Path) -> None:
    fig_dir = Path(args.figure_dir) if args.figure_dir else DEFAULT_FIGURE_DIR
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts" / "plot_i517_base_headroom.py"),
        "--in",
        str(comparison_path),
        "--out-dir",
        str(fig_dir),
    ]
    logger.info("[phase=plot] launching: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    logger.info("[phase=plot] done")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="End-to-end smoke: 2 prompts/cell, 1 judge call, full 3-trait x "
        "2-context base grid, Qwen2.5-0.5B-Instruct on HF backend; outputs to "
        "/tmp/i517_smoke_r3/.",
    )
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip eval+judge subprocess phases; just rebuild the "
        "comparison JSON + re-plot from --judge-out and --trained-judge.",
    )
    ap.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip the plot phase (useful in CPU-only smoke).",
    )
    ap.add_argument("--n-q", type=int, default=40)
    ap.add_argument("--n-judge-calls", type=int, default=3)
    ap.add_argument(
        "--eval-contexts",
        nargs="+",
        default=("in_scenario", "default_assistant"),
    )
    ap.add_argument("--traits", nargs="+", default=None)
    ap.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--truncation-fail-threshold", type=float, default=0.05)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--judge-out", default=None)
    ap.add_argument("--comparison-out", default=None)
    ap.add_argument("--figure-dir", default=None)
    ap.add_argument(
        "--trained-judge",
        default=None,
        help="Path to #498's judge_scores.json. Default: "
        "eval_results/issue_498/judge_scores.json (relative to repo root).",
    )
    args = ap.parse_args(argv)

    if args.smoke:
        # Smoke knobs — round-3 fix (code-reviewer F1 + F2).
        #
        # The point of --smoke is to exercise EVERY structural code path the
        # production pipeline executes (preflight → eval → judge →
        # aggregate-validate → aggregate → plot) end-to-end, on a TINY
        # data slice. Round-2's smoke restricted the grid to 1 trait x 1
        # eval context, which produced a 1-cell raw_generations slice that
        # would crash the aggregator (`_validate_full_coverage` requires
        # EXPECTED_TRAITS x EXPECTED_BASE_CONTEXTS = 3 x 2 = 6 base cells
        # to be present, smoke or not) — so round 2 worked around it by
        # synthesizing a separate 18-row fixture for the aggregate phase,
        # making the smoke phase-isolated rather than end-to-end.
        #
        # Round-3 fix: shrink PER-CELL row count, not the grid. Keep the
        # full 3-trait x 2-context base grid + all 3 scenarios for eval
        # (so the eval subprocess writes the 6 raw files the aggregator
        # expects). The aggregator's smoke path enforces structural
        # presence of every expected cell and >=1 q per cell; the
        # 40-prompt count is relaxed. The full driver run now hits every
        # phase in sequence, consuming each phase's actual output, no
        # synthetic fixture in the middle.
        args.n_q = 2
        args.n_judge_calls = 1
        args.eval_contexts = ("in_scenario", "default_assistant")
        if args.traits is None:
            # All 3 SCENARIOS so all 3 traits land in raw_generations
            # (TRAIT_OF maps coding -> logical_and_pushes_back,
            # emotional_support -> validating, teacher -> explains_well).
            args.traits = ["coding", "emotional_support", "teacher"]
        if args.max_new_tokens > 64:
            # Trim smoke generation cost — eval rig still exercises the
            # full code path under truncation accounting.
            args.max_new_tokens = 64
        # Disable truncation gate for smoke (64-token cap will fail it).
        args.truncation_fail_threshold = 1.0
        # Force HF backend in smoke — vLLM needs CUDA, which the VM lacks.
        # The HF backend is sequential generate but smoke is only 12 calls
        # so wall-time is acceptable (~5 min on CPU with a tiny model).
        if args.backend == "vllm":
            args.backend = "hf"
        # Default smoke to a tiny CPU-friendly base model. The full 7B
        # default would take >30 min to generate 12 sequences on CPU; the
        # 0.5B variant exercises the same code path under <5 min. The
        # base-model choice does NOT affect the validator / aggregator /
        # plot code paths under exercise here.
        if args.base_model is None:
            args.base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        # Smoke uses local /tmp scratch dirs by default so multiple
        # iterations do not stomp the eval_results/issue_517/ production
        # layout. User can override with explicit --raw-dir / --judge-out
        # / --comparison-out / --figure-dir.
        if args.raw_dir is None:
            args.raw_dir = "/tmp/i517_smoke_r3/raw_generations"
        if args.judge_out is None:
            args.judge_out = "/tmp/i517_smoke_r3/base_headroom_judge.json"
        if args.comparison_out is None:
            args.comparison_out = "/tmp/i517_smoke_r3/base_vs_trained_comparison.json"
        if args.figure_dir is None:
            args.figure_dir = "/tmp/i517_smoke_r3/figures"

    # Plan §8 Risk #2: the driver owns the Q-bank invariant. Before any
    # eval subprocess, check that data/issue_498/Q_test.json + Q_train.json
    # exist; if not, subprocess into scripts/i498_phase0_preflight.py to
    # rebuild them (and assert the invariants afterwards). Skipped under
    # --aggregate-only (no eval phase to feed) inside the helper.
    _run_preflight_if_missing(args)

    if not args.aggregate_only:
        _run_eval(args)
        _run_judge(args)

    comparison_path = _run_aggregate(args)

    if not args.skip_plot:
        _run_plot(args, comparison_path)

    logger.info("[phase=done] i517 base-headroom probe complete")


if __name__ == "__main__":
    main()
