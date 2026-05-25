#!/usr/bin/env python3
"""Compute deferred issue #355 aggregate stats and generate the headline plot."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)
from explore_persona_space.task_workflow import tasks_dir

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval_results" / "issue_355"
AGG_PATH = EVAL_DIR / "aggregate.json"
PERSONAS = ("librarian", "comedian", "baseline")
COT_STYLES = ("no_cot", "generic_cot", "persona_cot")
SEEDS = (42, 137, 256)
N_BOOT = 10_000
BOOT_SEED = 20260517


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _finite(values: list[float | int | None]) -> np.ndarray:
    arr = np.array([float(v) for v in values if v is not None and math.isfinite(float(v))])
    return arr


def _mean(values: list[float | int | None]) -> float | None:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else None


def _std(values: list[float | int | None]) -> float | None:
    arr = _finite(values)
    return float(np.std(arr, ddof=0)) if arr.size else None


def _percentile(values: list[float | int | None], q: float) -> float | None:
    arr = _finite(values)
    return float(np.percentile(arr, q)) if arr.size else None


def _hist_strip(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        rule = row.get("strip_rule_id")
        if rule is not None:
            counts[str(rule)] += 1
    return dict(
        sorted(counts.items(), key=lambda kv: (int(kv[0]) if kv[0].isdigit() else 99, kv[0]))
    )


def _arm_metrics(
    analytical_rows: list[dict[str, Any]],
    empirical_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    empirical_rows = empirical_rows or []
    n_analytical = len(analytical_rows)
    n_valid = sum(
        1
        for row in analytical_rows
        if row.get("H_abcd") is not None and math.isfinite(float(row.get("H_abcd")))
    )
    n_empirical = len(empirical_rows)
    nonletter_total = sum(int(row.get("count_nonletter", 0)) for row in empirical_rows)
    sample_total = sum(
        int(row.get("n_letter_samples", 0)) + int(row.get("count_nonletter", 0))
        for row in empirical_rows
    )

    return {
        "mean_H_top20": _mean([r.get("H_top20") for r in analytical_rows]),
        "std_H_top20": _std([r.get("H_top20") for r in analytical_rows]),
        "mean_H_abcd": _mean([r.get("H_abcd") for r in analytical_rows]),
        "std_H_abcd": _std([r.get("H_abcd") for r in analytical_rows]),
        "mean_H_mle": _mean([r.get("H_mle") for r in empirical_rows]),
        "std_H_mle": _std([r.get("H_mle") for r in empirical_rows]),
        "mean_H_MM": _mean([r.get("H_MM") for r in empirical_rows]),
        "std_H_MM": _std([r.get("H_MM") for r in empirical_rows]),
        "n_q_analytical": n_analytical,
        "n_q_restricted_valid": n_valid,
        "n_q_empirical": n_empirical,
        "restricted_missing_frac": float((n_analytical - n_valid) / n_analytical)
        if n_analytical
        else None,
        "nonletter_empirical_frac": float(nonletter_total / sample_total) if sample_total else None,
        "mean_top20_mass": _mean([r.get("top20_mass") for r in analytical_rows]),
        "p5_top20_mass": _percentile([r.get("top20_mass") for r in analytical_rows], 5),
        "p95_top20_mass": _percentile([r.get("top20_mass") for r in analytical_rows], 95),
        "mean_abcd_total_mass_pre_renorm": _mean(
            [r.get("abcd_total_mass_pre_renorm") for r in analytical_rows]
        ),
        "mean_cot_char_len_post_strip": _mean(
            [r.get("cot_text_len_post_strip_chars") for r in analytical_rows]
        ),
        "mean_cot_token_len_post_strip": _mean(
            [r.get("cot_text_len_post_strip_tokens") for r in analytical_rows]
        ),
        "strip_rule_hits": _hist_strip(analytical_rows),
    }


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"lo": None, "hi": None, "method": "percentile", "n_resamples": N_BOOT, "n": 0}
    idx = rng.integers(0, values.size, size=(N_BOOT, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "lo": float(lo),
        "hi": float(hi),
        "method": "percentile",
        "n_resamples": N_BOOT,
        "n": int(values.size),
    }


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = min(1.0, (m - rank) * p_values[idx])
        running = max(running, raw)
        adjusted[idx] = running
    return adjusted


def _main_path(persona: str, cot_style: str, seed: int, empirical: bool = False) -> Path:
    kind = "empirical" if empirical else "analytical"
    return EVAL_DIR / kind / f"{persona}_{cot_style}_seed{seed}.jsonl"


def _main_rows() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    analytical: dict[str, list[dict[str, Any]]] = {}
    empirical: dict[str, list[dict[str, Any]]] = {}
    for persona in PERSONAS:
        for cot_style in COT_STYLES:
            for seed in SEEDS:
                key = f"{persona}__{cot_style}__seed{seed}"
                analytical[key] = _load_jsonl(_main_path(persona, cot_style, seed))
                empirical[key] = _load_jsonl(_main_path(persona, cot_style, seed, empirical=True))
    return analytical, empirical


def _compute_spearman(
    analytical: dict[str, list[dict[str, Any]]],
    empirical: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rhos: list[float] = []
    for persona in PERSONAS:
        for cot_style in COT_STYLES:
            xs: list[float] = []
            ys: list[float] = []
            for seed in SEEDS:
                key = f"{persona}__{cot_style}__seed{seed}"
                h_by_q = {
                    int(row["q_id"]): row.get("H_abcd")
                    for row in analytical[key]
                    if row.get("H_abcd") is not None and math.isfinite(float(row.get("H_abcd")))
                }
                for row in empirical[key]:
                    q_id = int(row["q_id"])
                    hmm = row.get("H_MM")
                    if (
                        q_id in h_by_q
                        and hmm is not None
                        and math.isfinite(float(hmm))
                        and math.isfinite(float(h_by_q[q_id]))
                    ):
                        xs.append(float(h_by_q[q_id]))
                        ys.append(float(hmm))
            entry_key = f"{persona}__{cot_style}"
            if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
                out[entry_key] = {"rho": None, "p_value": None, "n": len(xs)}
                continue
            res = spearmanr(xs, ys)
            rho = float(res.statistic)
            p_value = float(res.pvalue)
            out[entry_key] = {"rho": rho, "p_value": p_value, "n": len(xs)}
            if math.isfinite(rho):
                rhos.append(rho)
    out["min_rho"] = float(min(rhos)) if rhos else None
    return out


def _compute_wilcoxon(analytical: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    for persona in PERSONAS:
        for seed in SEEDS:
            p_key = f"{persona}__persona_cot__seed{seed}"
            g_key = f"{persona}__generic_cot__seed{seed}"
            persona_by_q = {
                int(row["q_id"]): row.get("H_abcd")
                for row in analytical[p_key]
                if row.get("H_abcd") is not None and math.isfinite(float(row.get("H_abcd")))
            }
            generic_by_q = {
                int(row["q_id"]): row.get("H_abcd")
                for row in analytical[g_key]
                if row.get("H_abcd") is not None and math.isfinite(float(row.get("H_abcd")))
            }
            deltas = np.array(
                [
                    float(persona_by_q[q_id]) - float(generic_by_q[q_id])
                    for q_id in sorted(persona_by_q.keys() & generic_by_q.keys())
                ],
                dtype=float,
            )
            if deltas.size == 0:
                stat = p_value = None
            else:
                res = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
                stat = float(res.statistic)
                p_value = float(res.pvalue)
            tests.append(
                {
                    "persona": persona,
                    "seed": seed,
                    "n_pairs": int(deltas.size),
                    "mean_delta_persona_minus_generic": float(np.mean(deltas)),
                    "median_delta_persona_minus_generic": float(np.median(deltas)),
                    "statistic": stat,
                    "p_value": p_value,
                }
            )

    adjusted = _holm_bonferroni([float(t["p_value"]) for t in tests])
    for test, p_adj in zip(tests, adjusted, strict=True):
        test["p_value_holm_corrected"] = float(p_adj)
        test["reject_holm_alpha_0p05"] = bool(p_adj < 0.05)

    mean_deltas = [float(t["mean_delta_persona_minus_generic"]) for t in tests]
    return {
        "tests": tests,
        "wilcoxon_summary": {
            "n_significant_at_alpha_0p05_holm": int(
                sum(bool(t["reject_holm_alpha_0p05"]) for t in tests)
            ),
            "sign_consistent": bool(
                all(d > 0 for d in mean_deltas) or all(d < 0 for d in mean_deltas)
            ),
            "all_mean_delta_positive": bool(all(d > 0 for d in mean_deltas)),
            "hypothesis_direction_supported": bool(all(d < 0 for d in mean_deltas)),
        },
    }


def _compute_bootstrap(analytical: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOT_SEED)
    per_seed: dict[str, Any] = {}
    seed_averaged: dict[str, Any] = {}
    for key, rows in analytical.items():
        per_seed[key] = _bootstrap_ci(
            np.array([row.get("H_abcd") for row in rows], dtype=float), rng
        )

    for persona in PERSONAS:
        for cot_style in COT_STYLES:
            by_seed = [
                np.array(
                    [
                        row.get("H_abcd")
                        for row in analytical[f"{persona}__{cot_style}__seed{seed}"]
                        if row.get("H_abcd") is not None and math.isfinite(float(row.get("H_abcd")))
                    ],
                    dtype=float,
                )
                for seed in SEEDS
            ]
            if any(arr.size == 0 for arr in by_seed):
                seed_averaged[f"{persona}__{cot_style}"] = {
                    "lo": None,
                    "hi": None,
                    "method": "percentile",
                    "n_resamples": N_BOOT,
                    "n_seeds": len(SEEDS),
                }
                continue
            means = np.empty(N_BOOT, dtype=float)
            for i in range(N_BOOT):
                means[i] = float(
                    np.mean(
                        [arr[rng.integers(0, arr.size, size=arr.size)].mean() for arr in by_seed]
                    )
                )
            lo, hi = np.percentile(means, [2.5, 97.5])
            seed_averaged[f"{persona}__{cot_style}"] = {
                "lo": float(lo),
                "hi": float(hi),
                "method": "percentile",
                "n_resamples": N_BOOT,
                "n_seeds": len(SEEDS),
                "n_per_seed": [int(arr.size) for arr in by_seed],
            }

    return {
        "__meta__": {
            "metric": "mean_H_abcd",
            "method": "percentile",
            "n_resamples": N_BOOT,
            "rng_seed": BOOT_SEED,
        },
        "per_seed": per_seed,
        "seed_averaged": seed_averaged,
    }


def _compute_cross_seed(analytical: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_eval: dict[int, list[float]] = defaultdict(list)
    by_eval_qids: dict[int, set[int]] = defaultdict(set)
    for path in sorted((EVAL_DIR / "analytical" / "cross_seed").glob("*.jsonl")):
        match = re.search(r"eval(\d+)_src(\d+)", path.name)
        if not match:
            continue
        eval_seed = int(match.group(1))
        source_seed = int(match.group(2))
        rows = _load_jsonl(path)
        key = f"librarian__persona_cot__eval{eval_seed}__src{source_seed}"
        out[key] = _arm_metrics(rows, [])
        vals = _finite([row.get("H_abcd") for row in rows])
        by_eval[eval_seed].extend([float(v) for v in vals])
        by_eval_qids[eval_seed].update(int(row["q_id"]) for row in rows)

    within_vs_cross: dict[str, Any] = {}
    for eval_seed in SEEDS:
        within_rows = analytical[f"librarian__persona_cot__seed{eval_seed}"]
        qids = by_eval_qids[eval_seed]
        within_vals = _finite(
            [row.get("H_abcd") for row in within_rows if int(row["q_id"]) in qids]
        )
        cross_vals = np.array(by_eval[eval_seed], dtype=float)
        within_mean = float(np.mean(within_vals)) if within_vals.size else None
        cross_mean = float(np.mean(cross_vals)) if cross_vals.size else None
        within_minus_cross = (
            float(within_mean - cross_mean)
            if within_mean is not None and cross_mean is not None
            else None
        )
        cross_minus_within = (
            float(cross_mean - within_mean)
            if within_mean is not None and cross_mean is not None
            else None
        )
        within_vs_cross[f"eval_seed{eval_seed}"] = {
            "mean_H_abcd_within": within_mean,
            "mean_H_abcd_cross": cross_mean,
            "within_minus_cross_delta": within_minus_cross,
            "cross_minus_within_gap": cross_minus_within,
            "n_within": int(within_vals.size),
            "n_cross": int(cross_vals.size),
            "passes_plan_memorization_threshold": bool(
                cross_minus_within is not None and cross_minus_within >= 0.2
            ),
        }
    gaps = [
        v["cross_minus_within_gap"]
        for v in within_vs_cross.values()
        if v["cross_minus_within_gap"] is not None
    ]
    out["within_vs_cross_delta"] = within_vs_cross
    out["memorization_summary"] = {
        "mean_cross_minus_within_gap": float(np.mean(gaps)) if gaps else None,
        "all_eval_seeds_pass_0p2_gap": bool(gaps and all(g >= 0.2 for g in gaps)),
    }
    return out


def _compute_cross_persona(
    analytical: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source = "comedian_persona_cot"
    base_dir = EVAL_DIR / "analytical" / "cross_persona" / source
    emp_dir = EVAL_DIR / "empirical" / "cross_persona" / source
    out: dict[str, Any] = {}
    means: dict[tuple[str, str], float | None] = {}
    for persona in PERSONAS:
        for cot_style in COT_STYLES:
            a_rows = _load_jsonl(base_dir / f"{persona}_{cot_style}_seed42.jsonl")
            e_rows = _load_jsonl(emp_dir / f"{persona}_{cot_style}_seed42.jsonl")
            key = f"{source}__{persona}__{cot_style}__seed42"
            out[key] = _arm_metrics(a_rows, e_rows)
            means[(persona, cot_style)] = out[key]["mean_H_abcd"]

    direction_match: dict[str, Any] = {}
    for persona in PERSONAS:
        main_delta = (
            analytical[f"{persona}__persona_cot__seed42"],
            analytical[f"{persona}__generic_cot__seed42"],
        )
        main_p = _mean([row.get("H_abcd") for row in main_delta[0]])
        main_g = _mean([row.get("H_abcd") for row in main_delta[1]])
        main_delta_mean = (
            float(main_p - main_g) if main_p is not None and main_g is not None else None
        )
        cross_p = means[(persona, "persona_cot")]
        cross_g = means[(persona, "generic_cot")]
        cross_delta_mean = (
            float(cross_p - cross_g) if cross_p is not None and cross_g is not None else None
        )
        direction_match[persona] = {
            "librarian_source_delta_persona_minus_generic_seed42": main_delta_mean,
            "comedian_source_delta_persona_minus_generic_seed42": cross_delta_mean,
            "matches_librarian_source_direction": bool(
                main_delta_mean is not None
                and cross_delta_mean is not None
                and math.copysign(1.0, main_delta_mean) == math.copysign(1.0, cross_delta_mean)
            ),
            "passes_0p1_magnitude_floor": bool(
                cross_delta_mean is not None and abs(cross_delta_mean) >= 0.1
            ),
        }
    out["cross_persona_direction_match"] = direction_match
    out["cross_persona_summary"] = {
        "source_used": source,
        "all_direction_match": bool(
            all(v["matches_librarian_source_direction"] for v in direction_match.values())
        ),
        "all_cross_deltas_positive": bool(
            all(
                v["comedian_source_delta_persona_minus_generic_seed42"] is not None
                and v["comedian_source_delta_persona_minus_generic_seed42"] > 0
                for v in direction_match.values()
            )
        ),
    }
    return out


def _compute_per_q_id(analytical: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for persona in PERSONAS:
        for seed in SEEDS:
            by_style = {
                cot_style: {
                    int(row["q_id"]): row.get("H_abcd")
                    for row in analytical[f"{persona}__{cot_style}__seed{seed}"]
                    if row.get("H_abcd") is not None and math.isfinite(float(row.get("H_abcd")))
                }
                for cot_style in COT_STYLES
            }
            rows: dict[str, Any] = {}
            for q_id in sorted(set().union(*(set(v) for v in by_style.values()))):
                record = {cot_style: by_style[cot_style].get(q_id) for cot_style in COT_STYLES}
                if record["persona_cot"] is not None and record["generic_cot"] is not None:
                    record["delta_persona_minus_generic"] = float(
                        record["persona_cot"] - record["generic_cot"]
                    )
                if record["persona_cot"] is not None and record["no_cot"] is not None:
                    record["delta_persona_minus_no_cot"] = float(
                        record["persona_cot"] - record["no_cot"]
                    )
                if record["generic_cot"] is not None and record["no_cot"] is not None:
                    record["delta_generic_minus_no_cot"] = float(
                        record["generic_cot"] - record["no_cot"]
                    )
                rows[str(q_id)] = record
            out[f"{persona}__seed{seed}"] = rows
    return out


def _generate_plot(aggregate: dict[str, Any]) -> None:
    set_paper_style("blog")
    colors = {
        "no_cot": paper_palette_role("neutral"),
        "generic_cot": paper_palette_role("primary"),
        "persona_cot": paper_palette_role("baseline"),
    }
    labels = {
        "no_cot": "No rationale",
        "generic_cot": "Generic rationale",
        "persona_cot": "Persona-style rationale",
    }
    personas = ["librarian", "comedian", "baseline"]
    persona_labels = ["Librarian eval", "Comedian eval", "Assistant eval"]
    x = np.arange(len(personas))
    width = 0.23
    offsets = {"no_cot": -width, "generic_cot": 0.0, "persona_cot": width}

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for cot_style in COT_STYLES:
        means = []
        errs = []
        for persona in personas:
            seed_vals = [
                aggregate["arms"][f"{persona}__{cot_style}__seed{seed}"]["mean_H_abcd"]
                for seed in SEEDS
            ]
            means.append(float(np.mean(seed_vals)))
            errs.append(float(np.std(seed_vals, ddof=1)))
        ax.bar(
            x + offsets[cot_style],
            means,
            width=width,
            label=labels[cot_style],
            color=colors[cot_style],
            alpha=0.9,
        )
        ax.errorbar(
            x + offsets[cot_style],
            means,
            yerr=errs,
            fmt="none",
            ecolor="#1A1A1A",
            elinewidth=1.0,
            capsize=3,
            zorder=3,
        )

    ax.axhline(math.log(4), color="#7A7A7A", linestyle=(0, (4, 4)), linewidth=1.2)
    ax.text(
        len(personas) - 0.38,
        math.log(4) + 0.025,
        "Random over four letters",
        ha="right",
        va="bottom",
        color="#5A5A5A",
        fontsize=9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(persona_labels)
    ax.set_ylabel("Answer uncertainty after the rationale (nats)")
    ax.set_ylim(0, 1.48)
    ax.legend(loc="upper left", ncols=3, bbox_to_anchor=(0, 1.0))
    set_title_subtitle(
        ax,
        "Persona-style rationale leaves slightly more answer uncertainty\n"
        "than generic rationale across three eval personas (unfiltered main grid)",
        "Mean analytical answer uncertainty across three seeds; error bars show seed variation.",
        source="Source: eval_results/issue_355, branch task-355-implementation, commit 07b18051",
    )

    task_artifacts = tasks_dir() / "reviewing" / "355" / "artifacts"
    figures = ROOT / "figures" / "issue_355"
    written = savefig_paper(fig, "hero", dir=task_artifacts, formats=("png",))
    figures.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(written["png"], figures / "headline_h_abcd.png")
    meta = json.loads((task_artifacts / "hero.meta.json").read_text())
    meta.update(
        {
            "canonical_copy": "figures/issue_355/headline_h_abcd.png",
            "metric": "mean analytical H_abcd",
            "reference_line": "log(4) = 1.3862943611198906",
            "source_commit_for_eval_jsonl": "07b18051",
        }
    )
    (task_artifacts / "hero.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    plt.close(fig)


def main() -> None:
    aggregate = json.loads(AGG_PATH.read_text())
    analytical, empirical = _main_rows()

    for key in sorted(analytical):
        aggregate["arms"][key].update(_arm_metrics(analytical[key], empirical[key]))

    aggregate["spearman_per_arm"] = _compute_spearman(analytical, empirical)
    aggregate["wilcoxon_per_pair"] = _compute_wilcoxon(analytical)
    aggregate["bootstrap_ci_per_arm"] = _compute_bootstrap(analytical)
    aggregate["cross_seed_arms"] = _compute_cross_seed(analytical)
    aggregate["cross_persona_arms"] = _compute_cross_persona(analytical)
    aggregate["per_q_id"] = _compute_per_q_id(analytical)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    aggregate.setdefault("metadata", {})["analysis_commit_sha"] = commit
    aggregate["metadata"]["deferred_stats_computed_by"] = (
        "scripts/issue_355/compute_deferred_stats_and_plot.py"
    )

    AGG_PATH.write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
    _generate_plot(aggregate)

    print("updated", AGG_PATH)
    print("wrote", tasks_dir() / "reviewing" / "355" / "artifacts" / "hero.png")
    print("wrote", ROOT / "figures" / "issue_355" / "headline_h_abcd.png")


if __name__ == "__main__":
    main()
