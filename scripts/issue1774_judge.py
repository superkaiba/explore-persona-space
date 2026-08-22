"""#1774 P5 — graded judge of the P3 steering completions (off-pod, Batch API).

Reads the P3-written manifest at ``eval_results/issue_1774/steering/manifest.json``
(the interface contract for ``scripts/issue1774_steering.py``, which lands in a
later round):

    {"meta": {...},
     "rows": [{"row_id": "<unique, <=53 chars, NO '__'>",
               "condition": "<e.g. top_sv0_pos | kernel_tail1_neg | rb_evil_pos |
                              erase_evil | steer_base_d0 | random2_neg>",
               "direction_class": "steer_top|steer_kernel|steer_rand|steer_erase|rb|steer_base",
               "manifest_index": <int corpus manifest row>,
               "question": "<user query text>",
               "completion": "<generated answer text>"}, ...]}

Judging (plan §4 P5, llm-judging rules 4/9/22/23/24):
- rubrics sycophancy + hallucination over ALL rows; the evil rubric ONLY over
  rows whose condition names the evil direction (the 3 evil-direction arms
  sanity read) — ~1,800×2 + 3×60 items, N=5 draws ≈ 18,900 Batch-API calls.
- judge = claude-sonnet-4-5-20250929 via ``eval.graded_judge.judge_graded``
  (the #663-hardened Batch client); max_tokens=1024 (reason-then-score floor,
  rule 23); FRESH per-rubric cache_dir under --out-dir (rule 22).
- output per rubric: mean 0-100 scores per row + per-draw kept scores, with the
  CONTENT-drop vs TRANSPORT-loss split reported separately (rules 9/24).
- checkpoint per rubric: each rubric's scores JSON persists the moment that
  rubric completes; re-runs skip completed rubrics (resume predicate keys on
  the rubric + manifest row count).

Usage:
  uv run python scripts/issue1774_judge.py [--manifest P] [--out-dir D]
      [--traits sycophancy,hallucination,evil] [--limit N] [--dry-run]
      [--force-batch]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1774_common as c  # noqa: E402

JUDGE_TRAITS = ("sycophancy", "hallucination", "evil")
JUDGE_N_DRAWS = 5
# llm-judging rule 23 floor (single-rationale; raised from 300, #2063); pre-#2021
# truncation-era cache entries: use a fresh cache_dir at next wave (rule 23 cache caveat).
JUDGE_MAX_TOKENS = 1024
MAX_ITEM_ID_LEN = 53  # Batch custom_id cap 64 − 11 encoder chars (#1415)

# Canonical repo-root copy of the #779 extraction-artifact cache (the rubric
# eval_prompts for sycophancy/hallucination live there; evil is the in-module
# verbatim EVIL_ARTIFACTS constant). data/ is gitignored, so a fresh worktree
# clone lacks it — stage from the canonical VM copy, fail loud otherwise.
CANONICAL_ARTIFACTS_DIR = Path(
    "/home/thomasjiralerspong/explore-persona-space/data/issue_779/artifacts"
)


def _ensure_rubric_artifacts(trait: str) -> None:
    """Stage the trait's #779 extraction-artifact cache into this checkout."""
    if trait == "evil":
        return  # verbatim in-module artifacts (issue779_common.EVIL_ARTIFACTS)
    import issue779_common as i779

    local = i779._artifacts_dir() / f"{trait}.json"
    if local.exists():
        return
    src = CANONICAL_ARTIFACTS_DIR / f"{trait}.json"
    if not src.exists():
        raise FileNotFoundError(
            f"rubric artifacts for {trait!r} missing at BOTH {local} and the "
            f"canonical copy {src}; run issue779_extract_rb.py --stage artifacts"
        )
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, local)
    print(f"[p5-judge] staged {trait} rubric artifacts from {src}")


def _rubric_eval_prompt(trait: str) -> str:
    """The verbatim #779 rubric with {question}/{answer} slots (judge_graded fills)."""
    from issue779_common import load_extraction_artifacts

    _ensure_rubric_artifacts(trait)
    ep = load_extraction_artifacts(trait)["eval_prompt"]
    assert "{question}" in ep and "{answer}" in ep, f"{trait} rubric lacks slots"
    return ep


def load_manifest_rows(manifest: Path) -> list[dict]:
    data = json.loads(manifest.read_text())
    rows = data["rows"]
    assert rows, f"empty steering manifest: {manifest}"
    seen: set[str] = set()
    for r in rows:
        rid = str(r["row_id"])
        if "__" in rid:
            raise ValueError(f"row_id contains '__' (custom_id delimiter): {rid!r}")
        if len(rid) > MAX_ITEM_ID_LEN:
            raise ValueError(f"row_id longer than {MAX_ITEM_ID_LEN} chars: {rid!r}")
        if rid in seen:
            raise ValueError(f"duplicate row_id: {rid!r}")
        seen.add(rid)
        for k in ("condition", "question", "completion"):
            assert k in r, f"manifest row missing {k!r}: {sorted(r)}"
    return rows


def rows_for_trait(rows: list[dict], trait: str) -> list[dict]:
    """sycophancy/hallucination score everything; evil only its own 3 arms."""
    if trait != "evil":
        return rows
    return [r for r in rows if "evil" in str(r["condition"])]


def judge_one_rubric(
    trait: str,
    rows: list[dict],
    out_dir: Path,
    *,
    dry_run: bool,
    force_batch: bool,
) -> dict:
    from explore_persona_space.eval.graded_judge import judge_graded

    items = [(str(r["row_id"]), str(r["question"]), str(r["completion"])) for r in rows]
    cache_dir = out_dir / f"cache_{trait}"  # FRESH per-rubric cache dir (rule 22)
    save_raw = out_dir / f"judge_raw_{trait}.json"
    kwargs: dict = {}
    if force_batch:
        kwargs["threshold_base"] = 0  # force the Batch path (tiny live smokes)
    res = judge_graded(
        items,
        _rubric_eval_prompt(trait),
        n_draws=JUDGE_N_DRAWS,
        cache_dir=cache_dir,
        save_raw=save_raw,
        max_tokens=JUDGE_MAX_TOKENS,
        dry_run=dry_run,
        **kwargs,
    )
    by_id = {str(r["row_id"]): r for r in rows}
    out = {
        "meta": c.repro_meta({"script": "scripts/issue1774_judge.py", "trait": trait}),
        "judge": {
            "model": "claude-sonnet-4-5-20250929",
            "n_draws": JUDGE_N_DRAWS,
            "max_tokens": JUDGE_MAX_TOKENS,
            "rubric": "verbatim #779 eval_prompt (persona-vectors recipe; Sonnet judge)",
        },
        "n_items": len(items),
        "n_total_draws": res.n_total_draws,
        # rules 9/24: CONTENT drops and TRANSPORT losses are DISTINCT counters —
        # never blended (blending recreates the censoring the split prevents).
        "n_content_dropped_draws": res.n_dropped_draws,
        "n_transport_lost_draws": res.n_transport_lost_draws,
        "per_item_transport_losses": res.per_item_transport_losses,
        "rows": [
            {
                "row_id": rid,
                "condition": by_id[rid]["condition"],
                "direction_class": by_id[rid].get("direction_class"),
                "manifest_index": by_id[rid].get("manifest_index"),
                "score_mean": res.scores.get(rid),
                "kept_draw_scores": res.per_item_scores.get(rid, []),
                "n_kept_draws": len(res.per_item_scores.get(rid, [])),
            }
            for rid in (i[0] for i in items)
        ],
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        default=None,
        help="P3 steering manifest (default eval_results/issue_1774/steering/manifest.json)",
    )
    ap.add_argument(
        "--out-dir", default=None, help="default eval_results/issue_1774/steering/judge"
    )
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--traits", default=",".join(JUDGE_TRAITS))
    ap.add_argument("--limit", type=int, default=0, help="row cap (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="build items, no API calls")
    ap.add_argument(
        "--force-batch",
        action="store_true",
        help="force the Batch API path below the sync crossover (live forced-batch smoke)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="judge anyway when the manifest carries judge_skip=true (plan §7 kill criterion)",
    )
    args = ap.parse_args(argv)

    steering_root = c.eval_out(args.out_root) / "steering"
    manifest = Path(args.manifest) if args.manifest else steering_root / "manifest.json"
    out_dir = Path(args.out_dir) if args.out_dir else steering_root / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    # plan §7 kill criterion, made mechanical (round 2): a calibration-failed
    # steering run marks judge_skip=true — refuse the ~18k-call P5 spend.
    if json.loads(manifest.read_text()).get("judge_skip") and not args.force:
        print(
            "[p5-judge] REFUSED: manifest carries judge_skip=true (steering "
            "calibration failed — plan §7 kill criterion); pass --force to override"
        )
        return 8

    rows = load_manifest_rows(manifest)
    if args.limit:
        rows = rows[: args.limit]
    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    assert set(traits) <= set(JUDGE_TRAITS), traits

    print(f"[phase=p5_judge] rows={len(rows)} traits={traits} dry_run={args.dry_run}")
    for trait in traits:
        trait_rows = rows_for_trait(rows, trait)
        out_json = out_dir / f"scores_{trait}.json"
        if out_json.exists():
            prior = json.loads(out_json.read_text())
            if prior.get("n_items") == len(trait_rows) and not args.dry_run:
                print(f"[p5-judge] resume: {out_json.name} complete ({prior['n_items']} items)")
                continue
        if not trait_rows:
            print(f"[p5-judge] {trait}: 0 eligible rows — skipped (no evil arms in manifest?)")
            continue
        n_calls = len(trait_rows) * JUDGE_N_DRAWS
        print(f"[p5-judge] {trait}: {len(trait_rows)} items x {JUDGE_N_DRAWS} draws = {n_calls}")
        out = judge_one_rubric(
            trait, trait_rows, out_dir, dry_run=args.dry_run, force_batch=args.force_batch
        )
        # checkpoint-per-rubric: persist the moment this rubric completes
        c.write_json_atomic(out_json, out)
        drops = out["n_content_dropped_draws"]
        lost = out["n_transport_lost_draws"]
        print(
            f"[p5-judge] unit {trait} done: items={out['n_items']} "
            f"content_drops={drops} transport_losses={lost}",
            flush=True,
        )
        if not args.dry_run and lost:
            print(
                f"[p5-judge] WARNING {trait}: {lost} transport-lost draws — freely "
                "re-judgeable; re-run this rubric against a FRESH cache_dir before "
                "publication (llm-judging rule 24(ii))"
            )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
