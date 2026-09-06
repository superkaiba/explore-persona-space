"""Snapshot behavior-score histograms from the exact R2FAIR label inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from issue1739_result1_spread_fig_v2 import (  # noqa: E402
    _rollout_vectors_graded,
    _rollout_vectors_rate,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/blog/synthetic-data/distribution_data.json"
EDGES = np.arange(0, 101, 10)


def summarize(rows, trait, setting, group="all"):
    """Use the original distribution producer's per-context score aggregation."""
    is_rate = trait == "hallucination" and setting in {"nqopen", "simpleqa"}
    vectors = (_rollout_vectors_rate if is_rate else _rollout_vectors_graded)(rows)
    assert set(vectors) == {setting}, (trait, setting, group, set(vectors))
    values = np.array([v.mean() for v in vectors[setting]])
    expected = np.array([r["dv"] * (100 if is_rate else 1) for r in rows if r["dv"] is not None])
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)
    assert values.size and np.isfinite(values).all()
    assert values.min() >= 0 and values.max() <= 100
    counts, _ = np.histogram(values, bins=EDGES)
    assert counts.sum() == values.size
    return {
        "trait": trait,
        "setting": setting,
        "group": group,
        "n_available": len(rows),
        "n_eval": int(values.size),
        "n_dropped_no_score": len(rows) - int(values.size),
        "counts": counts.tolist(),
        "scores": values.tolist(),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "n_exact_zero": int((values == 0).sum()),
        "dv_construct": "fabrication_rate_percent" if is_rate else "trait_score_0_100",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.source_root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    sources = {}

    def read_source(rel, expected_sha=None):
        """Verify committed inputs or staged labels against the scoring manifest."""
        raw = (root / rel).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        tracked = subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", revision, "--", rel], text=True
        ).strip()
        if tracked:
            committed = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{revision}:{rel}"]
            )
            assert raw == committed, f"Source differs from its browser-linked commit: {rel}"
        else:
            assert expected_sha is not None, f"Staged input lacks a pinned scoring hash: {rel}"
        if expected_sha is not None:
            assert sha == expected_sha, f"Source differs from scoring input: {rel}"
        sources[rel] = {
            "path": rel,
            "sha256": sha,
            "url": (
                f"https://raw.githubusercontent.com/superkaiba/explore-persona-space/{revision}/{rel}"
                if tracked
                else None
            ),
            "availability": (
                "committed source"
                if tracked
                else "locally staged labels; hash verified against scoring manifest; scores included in snapshot"
            ),
        }
        return json.loads(raw)

    comparison = json.loads((ROOT / "docs/blog/synthetic-data/plot_data.json").read_text())
    roster = [r for r in comparison["followup_comparison"] if r["method"] == "pv"]
    assert len(roster) == 11
    histograms = []
    for trait in ("evil", "sycophancy", "hallucination"):
        result = read_source(f"eval_results/issue_1739/result2_fair/{trait}/all_arms_spearman.json")
        hashes = result["meta"]["input_sha256"]
        for cell in [r for r in roster if r["trait"] == trait]:
            setting = cell["setting"]
            if setting in {"pvsynth", "wildchat_rung"}:
                rel = f"eval_results/issue_1739/{setting}/dv_dataset/{trait}/labeling.json"
                expected_sha = hashes[rel]
            else:
                rel = f"eval_results/issue_1739/dv_dataset/{trait}/labeling.json"
                expected_sha = hashes[f"data/issue_1739/hf_dl/train_dv/{trait}/labeling.json"]
            rows = [r for r in read_source(rel, expected_sha)["rows"] if r["rung"] == setting]
            assert rows, (trait, setting)
            if setting == "wildchat_rung":
                # Exactly the fair scorer's held-out split, not its whole WildChat pool.
                rows = [
                    r
                    for r in rows
                    if int(hashlib.sha1(str(r["context_id"]).encode()).hexdigest(), 16) % 5 == 4
                ]
            assert len({r["context_id"] for r in rows}) == len(rows)
            pooled = summarize(rows, trait, setting)
            assert pooled["n_eval"] == cell["n_eval"], (trait, setting, pooled, cell["n_eval"])
            pooled["source_file"] = rel
            histograms.append(pooled)
            if setting == "pvsynth":
                for sign in ("neg", "pos"):
                    subset = [r for r in rows if r["group_key"].endswith(f"-{sign}")]
                    assert len(subset) == 100
                    split = summarize(subset, trait, setting, sign)
                    split["source_file"] = rel
                    histograms.append(split)
                np.testing.assert_array_equal(
                    np.sum([r["counts"] for r in histograms[-2:]], axis=0), pooled["counts"]
                )
    OUT.write_text(
        json.dumps(
            {
                "bin_edges": EDGES.tolist(),
                "normalization": "Each histogram divides counts by that row's n_eval; bins [a,b), final [90,100].",
                "notes": "Per-context mean over retained rollout scores; missing scores stay missing. "
                "Every label file is SHA-256 matched to the R2FAIR scoring manifest. "
                "WildChat uses only sha1(context_id) mod 5 == 4, matching the displayed held-out correlations. "
                "Eleven pooled cells match the comparison's sample counts. Synthetic sign rows partition each 200-context suite. "
                "NQ-Open/SimpleQA use fabrication rates times 100; other cells use graded trait scores. "
                "Empirical frequencies, no confidence intervals or smoothing; no generation, judging, or fitting.",
                "sources": list(sources.values()),
                "histograms": histograms,
            },
            indent=2,
        )
        + "\n"
    )
    for row in histograms:
        print(
            f"{row['trait']}/{row['setting']}/{row['group']}: n={row['n_eval']}, "
            f"zero={row['n_exact_zero']}, mean={row['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
