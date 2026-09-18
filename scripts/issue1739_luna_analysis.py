"""Analyze frozen, quality-reviewed Luna annotations without imputing missing scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.issue1739_covariance_ablation import sha256, write_json
from scripts.issue1739_luna_judging import BEHAVIORS, jsonl, validate_rows

METHODS = (
    "preimage_cosine",
    "mapped_answer_projection",
    "context_native_cosine",
    "answer_on_context_cosine",
    "random",
)
KS = (50, 100, 200)
N_BOOT = 2000  # Same descriptive bootstrap budget as the preceding task-1739 analysis.


def prepare_templates(args):
    """Freeze an outcome-blind exploratory prompt-template sensitivity."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.feature_extraction.text import TfidfVectorizer

    if args.templates.exists():
        raise ValueError("Refusing to replace the frozen template partition")
    files = sorted((args.source / "responses").glob("selected_responses_*.jsonl"))
    records = sorted((r for p in files for r in jsonl(p)), key=lambda r: r["ci"])
    if not records or len({r["ci"] for r in records}) != len(records):
        raise ValueError("Missing/duplicate response records")
    prompts = [" ".join(r["prompt"].casefold().split()) for r in records]
    # Exploratory, fixed before outcome joining. This is lexical clustering,
    # not a validated semantic definition of a template.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(4, 4),
        min_df=2,
        max_features=100000,
        dtype=np.float32,
    )
    x = vectorizer.fit_transform(prompts)
    similarities = (x @ x.T).tocsr()
    partitions = {}
    for threshold in (0.90, 0.95):
        adjacency = similarities.copy()
        adjacency.data = (adjacency.data >= threshold).astype(np.float32)
        adjacency.eliminate_zeros()
        n, groups = connected_components(adjacency, directed=False)
        counts = np.bincount(groups)
        partitions[str(threshold)] = dict(
            n_groups=int(n),
            largest_group=int(counts.max()),
            groups={str(r["ci"]): int(g) for r, g in zip(records, groups, strict=True)},
        )
    write_json(
        args.templates,
        dict(
            created_at=time.time(),
            outcome_blind=True,
            recipe="casefold + whitespace normalization; char-4gram TF-IDF; min_df=2; "
            "max_features=100000; connected components at cosine >= 0.90 and >= 0.95. "
            "Within each selected set retain its highest-ranked member per component; "
            "do not refill. Random uses its frozen random order. Exploratory lexical sensitivity.",
            source_sha256={p.name: sha256(p) for p in files},
            partitions=partitions,
        ),
    )
    print(
        json.dumps(
            {t: {k: v for k, v in p.items() if k != "groups"} for t, p in partitions.items()}
        )
    )


def load_labels(root, require_audit=True):
    manifest = json.loads((root / "manifest.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    if sha256(root / "manifest.json") != provenance["manifest_sha256"]:
        raise ValueError("Changed annotation manifest")
    key = json.loads((root / "PRIVATE_id_key.json").read_text())
    for rid, ci in key.items():
        if hashlib.sha256(f"luna-annotation-v1/{ci}".encode()).hexdigest()[:16] != rid:
            raise ValueError("Changed opaque ID mapping")
    if len(set(key.values())) != len(key):
        raise ValueError("Duplicate context mapping")
    guidance_hashes = json.loads((root / "calibration_guidance_hashes.json").read_text())
    result, hashes = {}, {}
    for behavior in BEHAVIORS:
        if sha256(root / behavior / "rubric.md") != provenance["rubrics"][behavior]:
            raise ValueError("Changed behavior rubric")
        if sha256(root / behavior / "calibration_guidance.md") != guidance_hashes[behavior]:
            raise ValueError("Changed calibration guidance")
        result[behavior] = {"primary": {}, "audit": {}}
        phases = ("pilot", "production", "audit") if require_audit else ("pilot", "production")
        for phase in phases:
            for packet in manifest["behaviors"][behavior][phase]:
                src = root / behavior / phase / packet["name"]
                path = root / behavior / f"labels_{phase}" / packet["name"]
                if sha256(src) != packet["sha256"]:
                    raise ValueError(f"Changed input: {src}")
                rows = json.loads(path.read_text())  # Missing file is a hard error.
                validate_rows(json.loads(src.read_text()), rows, behavior)
                target = result[behavior]["audit" if phase == "audit" else "primary"]
                for row in rows:
                    ci = key[row["id"]]
                    if ci in target:
                        raise ValueError("Duplicate primary/audit context")
                    target[ci] = row
                hashes[str(path.relative_to(root))] = sha256(path)
        if len(result[behavior]["primary"]) != manifest["behaviors"][behavior]["n_items"]:
            raise ValueError("Incomplete primary coverage")
    return manifest, result, hashes


def observed_stats(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Empty selected set")
    observed = np.isfinite(scores)
    positive = observed & (scores >= 50)
    n, n_scored, n_positive = len(scores), int(observed.sum()), int(positive.sum())
    n_missing = n - n_scored
    return dict(
        n=n,
        n_scored=n_scored,
        n_missing=n_missing,
        n_positive=n_positive,
        coverage=n_scored / n,
        mean_scored=float(scores[observed].mean()) if n_scored else None,
        mean_full_bounds=[
            float(np.nansum(scores) / n),
            float((np.nansum(scores) + 100 * n_missing) / n),
        ],
        fraction_scored=n_positive / n_scored if n_scored else None,
        fraction_full_bounds=[n_positive / n, (n_positive + n_missing) / n],
    )


def empirical_draws(scores, masks, n_boot=N_BOOT, seed=1739):
    """Shared Poisson context weights preserve overlap across selected sets."""
    scores = np.asarray(scores, dtype=float)
    masks = np.asarray(masks, dtype=float)
    if masks.shape[0] != len(scores):
        raise ValueError("Score/membership row mismatch")
    observed = np.isfinite(scores)
    positive = observed & (scores >= 50)
    rng = np.random.default_rng(seed)
    weights = rng.poisson(1, size=(n_boot, len(scores))).astype(float)
    total = weights @ masks
    scored = weights @ (observed[:, None] * masks)
    pos = weights @ (positive[:, None] * masks)
    score_sum = weights @ (np.nan_to_num(scores, nan=0)[:, None] * masks)

    def divide(numerator, denominator):
        return np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
        )

    return dict(
        mean_scored=divide(score_sum, scored),
        fraction_scored=divide(pos, scored),
        fraction_lower=divide(pos, total),
        fraction_upper=divide(pos + total - scored, total),
        mean_lower=divide(score_sum, total),
        mean_upper=divide(score_sum + 100 * (total - scored), total),
    )


def interval(values):
    good = np.isfinite(values)
    if good.mean() < 0.975:
        return None
    return np.quantile(values[good], [0.025, 0.975]).tolist()


def difference_bounds(scores, mask_a, mask_b, binary=True):
    """Bounds reuse the same unknown outcome for a context shared by both arms."""
    scores = np.asarray(scores, dtype=float)
    a, b = np.asarray(mask_a, dtype=float), np.asarray(mask_b, dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError("Empty comparison arm")
    coefficients = a / a.sum() - b / b.sum()
    known = np.isfinite(scores)
    values = (scores >= 50).astype(float) if binary else np.nan_to_num(scores, nan=0)
    observed = float(coefficients[known] @ values[known])
    scale = 1 if binary else 100
    unknown = coefficients[~known]
    return [
        observed + scale * float(unknown[unknown < 0].sum()),
        observed + scale * float(unknown[unknown > 0].sum()),
    ]


def repeatability(primary, repeat):
    if not repeat or not set(repeat).issubset(primary):
        raise ValueError("Missing/misaligned repeat audit")
    pairs = [(primary[ci], repeat[ci]) for ci in sorted(repeat)]
    both = [(a, b) for a, b in pairs if a["status"] == b["status"] == "scored"]

    def category(row):
        return str(row["positive"]) if row["status"] == "scored" else row["status"]

    result = dict(
        n=len(pairs),
        n_both_scored=len(both),
        status_agreement=float(np.mean([a["status"] == b["status"] for a, b in pairs])),
        category_agreement=float(np.mean([category(a) == category(b) for a, b in pairs])),
        binary_agreement=None,
        mean_absolute_score_difference=None,
        kappa=None,
        disagreements=[
            dict(ci=ci, primary=primary[ci], repeat=repeat[ci])
            for ci in sorted(repeat)
            if category(primary[ci]) != category(repeat[ci])
        ],
    )
    if both:
        a, b = np.array([[v["score"] for v in pair] for pair in both]).T
        agreement = np.mean((a >= 50) == (b >= 50))
        pa, pb = np.mean(a >= 50), np.mean(b >= 50)
        chance = pa * pb + (1 - pa) * (1 - pb)
        result.update(
            binary_agreement=float(agreement),
            mean_absolute_score_difference=float(np.mean(np.abs(a - b))),
            kappa=float((agreement - chance) / (1 - chance)) if chance < 1 else None,
        )
    return result


def selected_sets(memberships, behavior, partition=None):
    """Validate frozen cohorts and return ranked prefixes or template representatives."""
    source_behavior = "evil" if behavior == "harmful_compliance" else behavior
    sets = {}
    for method in METHODS:
        rows = [
            m
            for m in memberships
            if m["method"] == method and (method == "random" or m["behavior"] == source_behavior)
        ]
        expected = 1000 if method == "random" else 200
        if len(rows) != expected or len({r["ci"] for r in rows}) != expected:
            raise ValueError(f"Unexpected membership coverage: {behavior}/{method}")
        if method == "random":
            # The producer stores null ranks in the original RNG draw order.
            if any(r["rank"] is not None for r in rows):
                raise ValueError("Random memberships must have null ranks")
        else:
            ranks = [r["rank"] for r in rows]
            if any(type(rank) is not int for rank in ranks) or sorted(ranks) != list(
                range(1, expected + 1)
            ):
                raise ValueError(f"Invalid frozen ranks: {behavior}/{method}")
            rows = sorted(rows, key=lambda r: r["rank"])
        for k in (1000,) if method == "random" else KS:
            chosen, seen = [], set()
            for row in rows[:k]:
                ci = row["ci"]
                group = ci if partition is None else partition[str(ci)]
                if group not in seen:
                    chosen.append(ci)
                    seen.add(group)
            if partition is None and len(chosen) != k:
                raise ValueError("Frozen selected-set cardinality changed")
            sets[f"{method}/{k}"] = chosen
    return sets


def verify_response_sources(source, manifest):
    """Accept the frozen single-shard manifest and the current explicit shard mapping."""
    paths = sorted((source / "responses").glob("selected_responses_*.jsonl"))
    if not paths:
        raise ValueError("Missing source response shards")
    recorded = manifest["source_response_sha256"]
    if isinstance(recorded, str):
        if len(paths) != 1:
            raise ValueError("Legacy response digest requires exactly one source shard")
        recorded = {paths[0].name: recorded}
    if not isinstance(recorded, dict) or set(recorded) != {p.name for p in paths}:
        raise ValueError("Response shard coverage mismatch")
    for path in paths:
        if sha256(path) != recorded[path.name]:
            raise ValueError("Changed source response file")


def analyze(args):
    manifest, labels, hashes = load_labels(args.annotations)
    accepted = json.loads((args.annotations / "quality_acceptance.json").read_text())
    if accepted["annotation_file_sha256"] != hashes or not accepted["content_review_complete"]:
        raise ValueError("Labels have not passed the frozen content-quality gate")
    memberships_path = args.source / "selection/memberships.json"
    if sha256(memberships_path) != manifest["source_membership_sha256"]:
        raise ValueError("Membership provenance changed")
    memberships = json.loads(memberships_path.read_text())
    verify_response_sources(args.source, manifest)
    templates = json.loads(args.templates.read_text())
    for name, digest in templates["source_sha256"].items():
        if sha256(args.source / "responses" / name) != digest:
            raise ValueError("Template input provenance changed")
    pool_path = args.source / "selection/summary.json"
    pool = json.loads(pool_path.read_text())
    # Preparation recorded judging as incomplete; analysis status belongs above
    # this immutable population description, not in its stale preparation flag.
    del pool["behavior_scoring_complete"]
    results = dict(
        created_at=time.time(),
        annotation_complete=True,
        analysis_complete=True,
        quality_acceptance_sha256=sha256(args.annotations / "quality_acceptance.json"),
        annotations=hashes,
        template_sha256=sha256(args.templates),
        pool=pool,
        preparation_summary_sha256=sha256(pool_path),
        interpretation="Cached answers and contexts used to train the map; training-population "
        "retrieval only, not unseen-context or fresh-rollout risk. Luna annotation, not human truth.",
        intervals="Pointwise 95% descriptive Poisson context-bootstrap, 2000 draws, shared weights "
        "across overlapping arms; conditional on fixed map, pool, selections and annotations. "
        "No judge-error or refitting uncertainty; degenerate intervals at zero do not prove zero risk.",
        behaviors={},
    )
    for bi, behavior in enumerate(BEHAVIORS):
        primary = labels[behavior]["primary"]
        cis = sorted(primary)
        index = {ci: i for i, ci in enumerate(cis)}
        scores = np.array(
            [primary[ci]["score"] if primary[ci]["status"] == "scored" else np.nan for ci in cis]
        )
        sets = selected_sets(memberships, behavior)
        names = list(sets)
        masks = np.zeros((len(cis), len(names)))
        for j, name in enumerate(names):
            masks[[index[ci] for ci in sets[name]], j] = 1
        draws = empirical_draws(scores, masks, seed=1739 + bi)
        cells = {}
        for j, name in enumerate(names):
            cell = observed_stats(scores[masks[:, j].astype(bool)])
            cell["status_counts"] = {
                s: sum(primary[ci]["status"] == s for ci in sets[name])
                for s in ("scored", "unassessable", "judge_refusal")
            }
            cell["intervals"] = {m: interval(v[:, j]) for m, v in draws.items()}
            cells[name] = cell
        comparisons = {}
        for k in KS:
            a_name = f"preimage_cosine/{k}"
            a = names.index(a_name)
            for method in METHODS[1:]:
                b_name = f"{method}/{1000 if method == 'random' else k}"
                b = names.index(b_name)
                comparisons[f"{a_name}_minus_{b_name}"] = {
                    metric: dict(
                        estimate=(cells[a_name][metric] - cells[b_name][metric])
                        if cells[a_name][metric] is not None and cells[b_name][metric] is not None
                        else None,
                        interval=interval(draws[metric][:, a] - draws[metric][:, b]),
                    )
                    for metric in ("mean_scored", "fraction_scored")
                }
                comparisons[f"{a_name}_minus_{b_name}"].update(
                    full_fraction_difference_bounds=difference_bounds(
                        scores, masks[:, a], masks[:, b]
                    ),
                    full_mean_difference_bounds=difference_bounds(
                        scores, masks[:, a], masks[:, b], binary=False
                    ),
                )
        random_rate = cells["random/1000"]["fraction_scored"]
        for cell in cells.values():
            cell["enrichment_scored_over_random"] = (
                cell["fraction_scored"] / random_rate
                if random_rate and cell["fraction_scored"] is not None
                else None
            )
        sensitivity = {}
        for threshold, partition in templates["partitions"].items():
            unique_sets = selected_sets(memberships, behavior, partition["groups"])
            sensitivity[threshold] = {
                name: observed_stats(
                    [
                        primary[ci]["score"] if primary[ci]["status"] == "scored" else np.nan
                        for ci in chosen
                    ]
                )
                for name, chosen in unique_sets.items()
            }
        results["behaviors"][behavior] = dict(
            cells=cells,
            preimage_comparisons=comparisons,
            template_sensitivity=sensitivity,
            repeat_audit=repeatability(primary, labels[behavior]["audit"]),
        )
    write_json(args.output, results)
    print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("templates", "analyze"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "templates":
        prepare_templates(args)
    else:
        if args.annotations is None or args.output is None:
            parser.error("analyze requires --annotations and --output")
        analyze(args)


if __name__ == "__main__":
    main()
