#!/usr/bin/env python
"""#1090 ``fu2-dose-extension`` VM judge fold: the c3 Tier-2 install record.

The fu2 GPU phase (``scripts/issue1090_fu2.py``) retrained both sycophancy
organisms at epochs=6, dose-selected against the registered 0.60-0.85 judged
band, and — because c3-sycophancy-claude entered the band (earliest in-band
rung: step 14) — generated Tier-2 install-eval completions (10 per question x
20 held-out questions, trained + base) with JUDGING deferred to the VM. This
script is that deferred VM phase:

1. Stage the Tier-2 completions + the two cells' ladder/build records from the
   HF data repo (``issue1090_pvdatagen/fu2-dose-extension/``) into
   ``data/issue_1090/fu2/`` (scoped ``list_repo_tree`` + per-file download via
   ``fu1._stage_repo_prefix`` — never ``snapshot_download``).
2. Judge trained + base with the SAME instrument fu1 used
   (``fu1._judge_fu1`` -> ``judge_graded`` at ``max_tokens=300``, sycophancy
   rubric, 5 draws per completion, malformed/REFUSAL draws DROPPED never
   coerced), against a fresh cache dir under
   ``data/issue_1090/fu2/tier2_judge/``.
3. Write the install record — rates + Wilson 95, trained-base delta + Newcombe
   hybrid 95, the recomputed dose selection, and the committed epochs-3
   reference deltas — to
   ``eval_results/issue_1090/fu2-dose-extension/c3_install_fu2.json``.

Run from the issue worktree: ``uv run python scripts/issue1090_fu2_judge_fold.py``.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # credentials + shared-VM thread caps BEFORE any heavy import

import datetime as _dt  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1090_fu1 as fu1  # noqa: E402
import issue1090_run as i1090  # noqa: E402
from issue1090_free_analysis import _newcombe_delta_ci  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.recipe import (  # noqa: E402
    JUDGED_RATE_BAND,
    select_dose_checkpoint,
)

FU2_LABEL = "fu2-dose-extension"
FU2_DATA_PREFIX = f"{i1090.DATA_PREFIX}/{FU2_LABEL}"
FU2_EPOCHS = 6
# The fu2 adapter ladders live on the PRIVATE overflow repo (canonical model
# repo at the 100k-file limit — followup-scope directive; fu2 sentinel card).
FU2_ADAPTER_REPO = "superkaiba1/explore-persona-space-overflow"

C3 = i1090.CELL_BY_ID["c3"]


def _stage_fu2_inputs(out_root: Path) -> None:
    """Mirror the fu2 Tier-2 completions + both cells' records into out_root."""
    src = i1090.SOURCE_CONTEXT_ID
    fu1._stage_repo_prefix(
        i1090.HF_DATA_REPO,
        "dataset",
        f"{FU2_DATA_PREFIX}/raw_completions/tier2",
        out_root / "tier2",
        skip_if=lambda d: all(
            (d / C3.slug / f"completions__{s}__{src}.json").exists() for s in ("trained", "base")
        ),
    )
    for slug in (C3.slug, "c5-sycophancy-qwen"):
        fu1._stage_repo_prefix(
            i1090.HF_DATA_REPO,
            "dataset",
            f"{FU2_DATA_PREFIX}/{slug}",
            out_root / slug,
            skip_if=lambda d: (d / "fu2_ladder.json").exists(),
        )


def judge_tier2_fu2(out_root: Path, state: str, questions: list[str]) -> dict:
    """One (c3, state) fu2 Tier-2 judged read — the fu1 instrument verbatim
    (max_tokens=300, 5 draws, drop-never-coerce), fresh fu2-tagged cache."""
    src = i1090.SOURCE_CONTEXT_ID
    comp_path = out_root / "tier2" / C3.slug / f"completions__{state}__{src}.json"
    payload = i1090._read_json(comp_path)
    if payload["questions"] != questions:
        raise RuntimeError(
            f"{comp_path}: stored questions differ from the eval bank order — "
            "per-question joins would misalign; refusing"
        )
    completions = payload["completions"]
    behavior = BEHAVIORS[C3.behavior]
    tag = f"fu2-{C3.cell_id}-{state}"
    flat = [
        (f"{tag}-q{i:03d}-c{j}", q, comp)
        for i, q in enumerate(questions)
        for j, comp in enumerate(completions[i])
    ]
    indexed = [
        (i, f"{tag}-q{i:03d}-c{j}")
        for i, q in enumerate(questions)
        for j in range(len(completions[i]))
    ]
    cell_dir = out_root / "tier2_judge" / tag
    cell_dir.mkdir(parents=True, exist_ok=True)
    result = fu1._judge_fu1(
        flat,
        behavior.judge_rubric,
        n_draws=i1090.TIER2_JUDGE_DRAWS,
        cache_dir=cell_dir,
        save_raw=cell_dir / "judge_raw.json",
        judge_model=behavior.judge_model,
    )
    out = fu1.reduce_judge_fu1(
        indexed, result.scores, threshold=behavior.threshold, n_questions=len(questions)
    )
    out.update(
        {
            "n_total_draws": result.n_total_draws,
            "n_dropped_draws": result.n_dropped_draws,
            "judge_max_tokens": fu1.JUDGE_MAX_TOKENS_FU1,
            "n_judge_draws": i1090.TIER2_JUDGE_DRAWS,
        }
    )
    return out


def _epochs3_reference(repo_root: Path) -> dict:
    """Committed epochs-3 c3 install deltas (closure + fu1 fresh-300 reads)."""
    closure = i1090._read_json(
        repo_root / "eval_results" / "issue_1090" / "free_analysis" / "c3_dropclosure.json"
    )
    fu1_reads = i1090._read_json(
        repo_root / "eval_results" / "issue_1090" / "fu1-margin-qwen" / "judged_reads.json"
    )
    ft = fu1_reads[f"{C3.slug}__trained"]
    fb = fu1_reads[f"{C3.slug}__base"]
    return {
        "closure_delta": closure["closure_install"]["closure_delta"],
        "closure_delta_newcombe95": closure["closure_install"]["closure_delta_newcombe95"],
        "closure_rates": {
            "trained": closure["states"]["trained"]["closure"]["rate"],
            "base": closure["states"]["base"]["closure"]["rate"],
        },
        "fresh300_delta": ft["rate"] - fb["rate"],
        "fresh300_rates": {"trained": ft["rate"], "base": fb["rate"]},
        "sources": [
            "eval_results/issue_1090/free_analysis/c3_dropclosure.json",
            "eval_results/issue_1090/fu1-margin-qwen/judged_reads.json",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    repo_root = fu1._repo_root()
    out_root = repo_root / "data" / "issue_1090" / "fu2"
    _stage_fu2_inputs(out_root)

    ladder = i1090._read_json(out_root / C3.slug / "fu2_ladder.json")
    build = i1090._read_json(out_root / C3.slug / "fu2_build_result.json")
    selection = select_dose_checkpoint(
        {int(k): float(v) for k, v in ladder["rates_by_step"].items()}, band=JUDGED_RATE_BAND
    )
    if not selection.in_band:
        raise RuntimeError(
            f"c3 fu2 ladder no longer selects in-band ({selection}) — the staged "
            "ladder disagrees with the fu2 sentinel; refusing"
        )

    questions = i1090._eval_questions(
        i1090.RunConfig(smoke=False, cells=(C3,), out_root=out_root), C3.behavior
    )
    reads = {}
    reads_path = out_root / "judged_reads_fu2.json"
    if reads_path.exists():
        reads = i1090._read_json(reads_path)
    for state in ("trained", "base"):
        if state in reads:
            continue
        reads[state] = judge_tier2_fu2(out_root, state, questions)
        i1090._atomic_write_json(reads_path, reads)  # checkpoint per read

    t, b = reads["trained"], reads["base"]
    delta = t["rate"] - b["rate"]
    lo, hi = _newcombe_delta_ci(t["k"], t["n"], b["k"], b["n"])
    record = {
        "cell": C3.slug,
        "followup_label": FU2_LABEL,
        "status": "computed",
        "behavior": C3.behavior,
        "generator": "claude",
        "epochs": FU2_EPOCHS,
        "band": list(JUDGED_RATE_BAND),
        "selection": {
            "step": selection.step,
            "rate": selection.rate,
            "in_band": selection.in_band,
            "fallback": selection.fallback,
        },
        "adapter_repo": FU2_ADAPTER_REPO,
        "adapter_path": f"issue1090/fu2/{C3.slug}/checkpoint-{selection.step}",
        "mix_sha256": build["mix"]["train_mix_sha256"],
        "tier2": {
            "n_completions": i1090.TIER2_N_COMPLETIONS,
            "n_judge_draws": i1090.TIER2_JUDGE_DRAWS,
        },
        "judge_max_tokens": fu1.JUDGE_MAX_TOKENS_FU1,
        "reads": reads,
        "install_delta": delta,
        "install_delta_newcombe95": [lo, hi],
        "epochs3_reference": _epochs3_reference(repo_root),
        "meta": {
            "script": "scripts/issue1090_fu2_judge_fold.py",
            "ts": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_root,
            ).stdout.strip(),
            "judge_model": BEHAVIORS[C3.behavior].judge_model,
            "threshold": BEHAVIORS[C3.behavior].threshold,
            "fu2_gpu_git_commit": build["git_commit"],
            "instrument_note": ladder["instrument_note"],
        },
    }
    dest = repo_root / "eval_results" / "issue_1090" / FU2_LABEL / "c3_install_fu2.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    i1090._atomic_write_json(dest, record)
    print(
        f"[fu2-judge] c3 install @epochs6 step{selection.step}: trained {t['rate']:.3f} "
        f"(k={t['k']}/n={t['n']}) vs base {b['rate']:.3f} (k={b['k']}/n={b['n']}) — "
        f"delta {delta:+.3f} newcombe95 [{lo:+.3f}, {hi:+.3f}] -> {dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
