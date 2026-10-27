"""Issue #2674 Phase 1 (core) analysis: does subliminal learning need multiple epochs?

Reads every core run's trajectory (teacher-forced log-prob probe every 20 steps, held-out
loss every 100 steps) and sampled-eval metrics (on-policy cat rate at 13 saved
adapters), staged from the HF data repo, then writes ONE ``core_analysis.json`` plus
four figures.

Conditions, each trained to 100k example-passes at batch 66 (about 1,516 steps):
  A  10k unique examples x 10 epochs
  B  25k unique examples x 4 epochs
  C  100k unique examples x 1 epoch
Every cat-teacher run has a paired neutral-teacher control with the same seed, prompts
and order. One seed sets the subset draw, data order and LoRA init, so the seed is the
unit of analysis: intervals are t-intervals over seeds, and contrasts between
conditions are paired within seed.

Reads computed (design doc docs/ideas/2026-10-26-subliminal-epoch-test-design.md):
  Q1        cat-rate difference at step 152, where every run has seen 10k unique
            examples once; and run A at each epoch boundary 1..10.
  Q2        all three conditions at the final step and at step 379, with paired
            within-seed contrasts C-A, B-A and C-B.
  dose      run C at every saved step (single passes over 2.5k..100k examples).
  draw      B vs C at step 379 (both one pass over a different random 25k).
  peak      each pair's maximum cat-rate difference vs its final value (rise then decay).
  proxy     Spearman rho between delta log P(cat) and the cat-rate difference over all
            sampled checkpoints.
  fit       held-out per-token loss on cat-teacher completions, per condition.

"Cat-rate difference" is the share of sampled answers naming cat (word-boundary match,
plain favorite-animal questions) for the cat-teacher student minus the neutral-teacher
student, in percentage points. "Delta log P(cat)" is the teacher-forced log-probability
of starting the answer with Cat/cat/Cats/cats, cat-teacher student minus control, in
nats.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue2674_common as C  # noqa: E402
import issue2674_gate_analysis as G  # noqa: E402

PRIMARY_SET = G.PRIMARY_SET
PRIMARY_RULE = G.PRIMARY_RULE
REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_PREFIX = f"{C.EXPERIMENT}/{C.HF_CORE_SUBDIR}"
DEFAULT_STAGE_ROOT = f"/mnt/eps-data/{os.environ.get('USER', 'user')}/issue2674_core_metrics"

#: Display name per condition. Shared by every figure and the JSON, so a rename lands once.
COND_LABEL = {"A": "A: 10k x 10 epochs", "B": "B: 25k x 4 epochs", "C": "C: 100k x 1 epoch"}
#: Project blog palette slots 1-3 (deep blue, orange, green). Every line is also
#: direct-labeled and has its own marker shape, so identity is never color-alone.
COND_COLOR = dict(zip(("A", "B", "C"), ("#1F4E9F", "#E08220", "#3FA577"), strict=True))
COND_MARKER = {"A": "o", "B": "s", "C": "^"}
CONDS = tuple(c for c, _, _ in C.CORE_CONDITIONS)
UNIQUE = {c: u for c, u, _ in C.CORE_CONDITIONS}
EPOCHS = {c: e for c, _, e in C.CORE_CONDITIONS}

Q1_STEP = int(round(10_000 / C.EFFECTIVE_BATCH_SIZE))  # 152: 10k examples seen once
Q2_MID_STEP = int(round(25_000 / C.EFFECTIVE_BATCH_SIZE))  # 379: one pass over 25k


# --------------------------------------------------------------------------------------
# Staging + loading
# --------------------------------------------------------------------------------------


def stage(stage_root: Path) -> Path:
    """Mirror the core runs/ and metrics/ prefixes from the HF data repo; return the
    directory holding them. Re-runs download only files not yet on disk."""
    from explore_persona_space.orchestrate.hub import stage_hub_prefix

    for sub in ("runs", "metrics"):
        files = stage_hub_prefix(C.HF_DATA_REPO, f"{CORE_PREFIX}/{sub}", stage_root)
        print(f"[stage] {sub}: {len(files)} files", flush=True)
    return stage_root / CORE_PREFIX


def load_pairs(data_root: Path, seeds, allow_missing: bool) -> tuple[dict, list[str]]:
    """{(cond, seed): {"cat": run, "neutral": run}} for every pair with both members."""
    paths = {"runs": data_root / "runs", "metrics": data_root / "metrics"}
    pairs: dict[tuple[str, int], dict] = defaultdict(dict)
    missing: list[str] = []
    for spec in C.core_runs(seeds):
        cond = spec.group.split("_", 1)[1]
        try:
            run = G.load_run(paths, spec.name)
        except FileNotFoundError:
            missing.append(spec.name)
            continue
        if not run["sampled"]:
            missing.append(spec.name)
            continue
        pairs[(cond, spec.seed)][spec.teacher] = run
    if missing and not allow_missing:
        raise RuntimeError(f"{len(missing)} core runs missing: {missing}")
    complete = {k: v for k, v in pairs.items() if set(v) == set(C.TEACHERS)}
    if not complete:
        raise RuntimeError(f"no complete (cat, neutral) pair under {data_root}")
    return complete, missing


def final_step(cond: str) -> int:
    return C.total_steps(UNIQUE[cond], EPOCHS[cond])


def epochs_at(cond: str, step: int) -> float:
    return step * C.EFFECTIVE_BATCH_SIZE / UNIQUE[cond]


# --------------------------------------------------------------------------------------
# Per-checkpoint rows
# --------------------------------------------------------------------------------------


def rate_rows(pairs: dict) -> list[dict]:
    """One row per (pair, sampled step, question set, scoring rule)."""
    rows = []
    for (cond, seed), pr in sorted(pairs.items()):
        cat, neu = pr["cat"], pr["neutral"]
        for step in G.sampled_steps(cat):
            cc, nc = G.sampled_at(cat, step), G.sampled_at(neu, step)
            if cc is None or nc is None:
                continue
            for set_name in C.QUESTION_SETS:
                for rule in C.SCORERS:
                    rd = G.rate_difference(cc["sets"], nc["sets"], set_name, rule)
                    rows.append(
                        {
                            "cond": cond,
                            "seed": seed,
                            "step": step,
                            "example_passes": step * C.EFFECTIVE_BATCH_SIZE,
                            "epochs": epochs_at(cond, step),
                            **rd,
                            "cat_cap_hit_fraction": cc["sets"][set_name]["cap_hit_fraction"],
                            "neutral_cap_hit_fraction": nc["sets"][set_name]["cap_hit_fraction"],
                        }
                    )
    if not rows:
        raise RuntimeError("no sampled checkpoint shared by a cat run and its control")
    return rows


def logp_rows(pairs: dict) -> list[dict]:
    """Delta log P(cat) and delta margin at every probe step, both question sets."""
    rows = []
    for (cond, seed), pr in sorted(pairs.items()):
        cat, neu = pr["cat"], pr["neutral"]
        steps = sorted({e["step"] for e in cat["entries"] if f"logp_{PRIMARY_SET}" in e})
        for step in steps:
            for set_name in C.QUESTION_SETS:
                dl = G.delta_logp(cat, neu, step, set_name, 0)
                if dl is not None:
                    rows.append({"cond": cond, "seed": seed, **dl})
    if not rows:
        raise RuntimeError("no log-prob probe step shared by a cat run and its control")
    return rows


def heldout_rows(pairs: dict) -> list[dict]:
    """Held-out per-token loss for both runs of every pair, on both teachers' data."""
    rows = []
    for (cond, seed), pr in sorted(pairs.items()):
        for arm, run in pr.items():
            for e in run["entries"]:
                for teacher in C.TEACHERS:
                    h = e.get(f"heldout_{teacher}")
                    if h is not None:
                        rows.append(
                            {
                                "cond": cond,
                                "seed": seed,
                                "arm": arm,
                                "heldout_teacher": teacher,
                                "step": e["step"],
                                "per_token_loss": h["per_token_loss"],
                            }
                        )
    return rows


# --------------------------------------------------------------------------------------
# Summaries over seeds
# --------------------------------------------------------------------------------------


def over_seeds(by_seed: dict[int, float]) -> dict:
    """Mean and 95% t-interval over seeds, with the per-seed values."""
    vals = [by_seed[s] for s in sorted(by_seed)]
    mean, lo, hi = C.mean_ci_t(vals) if len(vals) >= 2 else (vals[0], float("nan"), float("nan"))
    return {
        "n_seeds": len(vals),
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "per_seed": {str(s): by_seed[s] for s in sorted(by_seed)},
    }


def rate_at(rows, cond, step, set_name=PRIMARY_SET, rule=PRIMARY_RULE) -> dict[int, float]:
    return {
        r["seed"]: r["diff_pp"]
        for r in rows
        if r["cond"] == cond and r["step"] == step and r["set"] == set_name and r["rule"] == rule
    }


def logp_at(rows, cond, step, set_name=PRIMARY_SET, key="delta_logp_nats") -> dict[int, float]:
    return {
        r["seed"]: r[key]
        for r in rows
        if r["cond"] == cond and r["step"] == step and r["set"] == set_name
    }


def read_cell(rrows, lrows, cond: str, step: int) -> dict:
    """Both dependent variables for one condition at one step, both question sets."""
    out = {"cond": cond, "step": step, "epochs": epochs_at(cond, step)}
    for set_name in C.QUESTION_SETS:
        out[f"cat_rate_diff_pp|{set_name}"] = over_seeds(rate_at(rrows, cond, step, set_name))
        out[f"delta_logp_nats|{set_name}"] = over_seeds(logp_at(lrows, cond, step, set_name))
    out["delta_margin_nats|plain"] = over_seeds(
        logp_at(lrows, cond, step, PRIMARY_SET, "delta_margin_nats")
    )
    return out


def paired(rrows, lrows, a: tuple[str, int], b: tuple[str, int]) -> dict:
    """Within-seed contrast a minus b, on the cat-rate difference and delta log P."""
    out = {"a": {"cond": a[0], "step": a[1]}, "b": {"cond": b[0], "step": b[1]}}
    for name, getter in (("cat_rate_diff_pp", rate_at), ("delta_logp_nats", logp_at)):
        for set_name in C.QUESTION_SETS:
            src = rrows if getter is rate_at else lrows
            va = getter(src, a[0], a[1], set_name)
            vb = getter(src, b[0], b[1], set_name)
            common = sorted(set(va) & set(vb))
            if common:
                out[f"{name}|{set_name}"] = over_seeds({s: va[s] - vb[s] for s in common})
    return out


def within_seed_mean(rrows, conds, step: int, set_name: str) -> dict:
    """Cat-rate difference averaged over conditions within each seed, then over seeds."""
    per = [rate_at(rrows, c, step, set_name) for c in conds]
    seeds = set.intersection(*(set(v) for v in per))
    return over_seeds({s: sum(v[s] for v in per) / len(per) for s in seeds})


def build_report(pairs: dict, missing: list[str], base_eval: Path) -> tuple[dict, dict]:
    rrows, lrows, hrows = rate_rows(pairs), logp_rows(pairs), heldout_rows(pairs)
    present = sorted({c for c, _ in pairs})
    report: dict = {
        "conditions": {
            c: {
                "label": COND_LABEL[c],
                "unique": UNIQUE[c],
                "epochs": EPOCHS[c],
                "final_step": final_step(c),
                "seeds": sorted(s for cc, s in pairs if cc == c),
            }
            for c in present
        },
        "missing_runs": missing,
        "primary": {"set": PRIMARY_SET, "rule": PRIMARY_RULE},
        "effective_batch_size": C.EFFECTIVE_BATCH_SIZE,
    }

    base = C.read_json(base_eval)["adapters"]["base"]["sets"]
    report["base_model"] = {
        s: {r: C.pct(base[s]["scores"][r]["rate"]) for r in C.SCORERS} for s in C.QUESTION_SETS
    }

    report["q1_one_epoch_10k"] = {
        "step": Q1_STEP,
        "per_condition": [read_cell(rrows, lrows, c, Q1_STEP) for c in present],
        # The three conditions share a seed's LoRA init and overlap in data, so they
        # are averaged within seed before the interval over seeds.
        "within_seed_mean": {
            set_name: within_seed_mean(rrows, present, Q1_STEP, set_name)
            for set_name in C.QUESTION_SETS
        },
    }
    if "A" in present:
        a_steps = sorted({r["step"] for r in rrows if r["cond"] == "A"})
        boundaries = [
            s for s in a_steps if abs(epochs_at("A", s) - round(epochs_at("A", s))) < 0.05
        ]
        report["q1_epochs_at_10k"] = [read_cell(rrows, lrows, "A", s) for s in boundaries]

    q2 = {"final": {}, "mid": {}}
    for c in present:
        q2["final"][c] = read_cell(rrows, lrows, c, final_step(c))
        q2["mid"][c] = read_cell(rrows, lrows, c, Q2_MID_STEP)
    contrasts = {}
    for a, b in (("C", "A"), ("B", "A"), ("C", "B")):
        if a in present and b in present:
            contrasts[f"final:{a}-{b}"] = paired(
                rrows, lrows, (a, final_step(a)), (b, final_step(b))
            )
            contrasts[f"step{Q2_MID_STEP}:{a}-{b}"] = paired(
                rrows, lrows, (a, Q2_MID_STEP), (b, Q2_MID_STEP)
            )
    q2["contrasts"] = contrasts
    report["q2_same_steps_different_u"] = q2

    if "C" in present:
        c_steps = sorted({r["step"] for r in rrows if r["cond"] == "C"})
        report["one_epoch_dose_response"] = [read_cell(rrows, lrows, "C", s) for s in c_steps]
    if "B" in present and "C" in present:
        report["draw_sensitivity_b_vs_c_step379"] = paired(
            rrows, lrows, ("B", Q2_MID_STEP), ("C", Q2_MID_STEP)
        )

    peak = {}
    for c in present:
        best, fin = {}, {}
        for s in sorted({s for cc, s in pairs if cc == c}):
            series = {
                r["step"]: r["diff_pp"]
                for r in rrows
                if r["cond"] == c
                and r["seed"] == s
                and r["set"] == PRIMARY_SET
                and r["rule"] == PRIMARY_RULE
            }
            best[s] = max(series.values())
            fin[s] = series[max(series)]
        peak[c] = {
            "peak": over_seeds(best),
            "final": over_seeds(fin),
            "peak_minus_final": over_seeds({s: best[s] - fin[s] for s in best}),
        }
    report["peak_vs_final"] = peak

    pts = []
    lookup = {
        (r["cond"], r["seed"], r["step"]): r["delta_logp_nats"]
        for r in lrows
        if r["set"] == PRIMARY_SET
    }
    for r in rrows:
        if r["set"] == PRIMARY_SET and r["rule"] == PRIMARY_RULE:
            dl = lookup.get((r["cond"], r["seed"], r["step"]))
            if dl is not None:
                pts.append(
                    {
                        "cond": r["cond"],
                        "seed": r["seed"],
                        "step": r["step"],
                        "delta_logp_nats": dl,
                        "cat_rate_diff_pp": r["diff_pp"],
                    }
                )
    rho = C.spearman_rho([p["delta_logp_nats"] for p in pts], [p["cat_rate_diff_pp"] for p in pts])
    report["proxy_check"] = {
        "n_points": len(pts),
        "spearman": rho,
        "positive": bool(isinstance(rho["rho"], float) and rho["rho"] > 0),
    }

    fit = {}
    for c in present:
        cat_on_cat = [
            h
            for h in hrows
            if h["cond"] == c and h["arm"] == "cat" and h["heldout_teacher"] == "cat"
        ]
        by_step: dict[int, dict[int, float]] = defaultdict(dict)
        for h in cat_on_cat:
            by_step[h["step"]][h["seed"]] = h["per_token_loss"]
        fit[c] = {str(st): over_seeds(v) for st, v in sorted(by_step.items())}
    report["heldout_cat_loss_cat_runs"] = fit

    report["cap_hit_fraction_max"] = max(
        max(r["cat_cap_hit_fraction"], r["neutral_cap_hit_fraction"]) for r in rrows
    )
    report["n_rows"] = {"rate": len(rrows), "logp": len(lrows), "heldout": len(hrows)}
    rows = {"rate_rows": rrows, "logp_rows": lrows, "heldout_rows": hrows}
    return report, rows


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _series(rows, cond, key, value, filt) -> tuple[dict[int, dict[int, float]], list[int]]:
    """{seed: {step: value}} for one condition, and the sorted union of steps."""
    out: dict[int, dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r["cond"] == cond and filt(r):
            out[r["seed"]][r[key]] = r[value]
    steps = sorted({st for d in out.values() for st in d})
    return out, steps


def _plot_condition(ax, by_seed, steps, cond, *, markers: bool, end_label: bool = True) -> None:
    """Thin per-seed lines plus a thick seed-mean line for one condition."""
    color = COND_COLOR[cond]
    for s in sorted(by_seed):
        xs = sorted(by_seed[s])
        ax.plot(xs, [by_seed[s][x] for x in xs], color=color, linewidth=0.8, alpha=0.35)
    xs = [x for x in steps if all(x in by_seed[s] for s in by_seed)]
    ys = [sum(by_seed[s][x] for s in by_seed) / len(by_seed) for x in xs]
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=2.0,
        marker=COND_MARKER[cond] if markers else None,
        markersize=6,
        label=COND_LABEL[cond],
    )
    if not end_label:
        return
    ax.annotate(
        COND_LABEL[cond].split(":")[0],
        (xs[-1], ys[-1]),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color="#0b0b0b",
    )


def _style(ax) -> None:
    ax.set_xscale("log")
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def make_figures(fig_dir: Path, report: dict, rows: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    fig_dir.mkdir(parents=True, exist_ok=True)
    present = list(report["conditions"])
    written: list[str] = []
    is_primary = lambda r: r["set"] == PRIMARY_SET and r["rule"] == PRIMARY_RULE  # noqa: E731

    # 1. main: cat-rate difference and delta log P against optimizer steps
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for c in present:
        by_seed, steps = _series(rows["rate_rows"], c, "step", "diff_pp", is_primary)
        _plot_condition(axes[0], by_seed, steps, c, markers=True)
        by_seed, steps = _series(
            rows["logp_rows"],
            c,
            "step",
            "delta_logp_nats",
            lambda r: r["set"] == PRIMARY_SET and r["step"] > 0,
        )
        _plot_condition(axes[1], by_seed, steps, c, markers=False, end_label=False)
    for c in present:  # epoch boundaries as rug ticks along the top edge
        for k in range(1, EPOCHS[c]):
            for ax in axes:
                ax.axvline(
                    k * C.steps_per_epoch(UNIQUE[c]),
                    ymin=0.97,
                    ymax=1.0,
                    color=COND_COLOR[c],
                    linewidth=1.2,
                )
    for ax in axes:
        _style(ax)
        ax.set_xlabel("optimizer steps (log scale; ticks on top = epoch ends)")
    axes[0].set_ylabel("cat rate: cat-teacher minus neutral-teacher\nstudent (percentage points)")
    axes[0].set_title("Sampled answers naming cat (plain questions)", fontsize=10)
    axes[1].set_ylabel("delta log P(answer starts with cat) (nats)")
    axes[1].set_title("Teacher-forced log-probability, every 20 steps", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")

    written.append(_save(fig, fig_dir, "core_main", plt, savefig_paper))

    # 2. grid: cat-rate difference over (unique examples, epochs completed)
    max_ep = max(EPOCHS[c] for c in present)
    grid = np.full((len(present), max_ep), np.nan)
    grid_step = np.zeros((len(present), max_ep), dtype=int)
    for i, c in enumerate(present):
        for st in sorted({r["step"] for r in rows["rate_rows"] if r["cond"] == c}):
            ep = epochs_at(c, st)
            if abs(ep - round(ep)) < 0.05 and 1 <= round(ep) <= max_ep:
                v = rate_at(rows["rate_rows"], c, st)
                grid[i, round(ep) - 1] = sum(v.values()) / len(v)
                grid_step[i, round(ep) - 1] = st
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
    cmap.set_bad("#f0efec")
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=np.nanmax(grid), aspect="auto")
    ax.grid(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                dark = grid[i, j] > 0.55 * np.nanmax(grid)
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.1f}\nstep {grid_step[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#fcfcfb" if dark else "#0b0b0b",
                )
    ax.set_xticks(range(max_ep), [str(k) for k in range(1, max_ep + 1)])
    ax.set_yticks(range(len(present)), [f"{UNIQUE[c] // 1000}k unique" for c in present])
    ax.set_xlabel("epochs completed (grey = no saved checkpoint)")
    fig.colorbar(im, ax=ax, label="cat-rate difference (pp)")

    written.append(_save(fig, fig_dir, "core_grid", plt, savefig_paper))

    # 3. held-out loss on cat-teacher completions, cat-teacher runs
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for c in present:
        by_seed, steps = _series(
            rows["heldout_rows"],
            c,
            "step",
            "per_token_loss",
            lambda r: r["arm"] == "cat" and r["heldout_teacher"] == "cat" and r["step"] > 0,
        )
        _plot_condition(ax, by_seed, steps, c, markers=False)
    ax.set_xscale("log")
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_xlabel("optimizer steps (log scale)")
    ax.set_ylabel("held-out per-token loss, cat-teacher data")
    ax.legend(fontsize=8, frameon=False)

    written.append(_save(fig, fig_dir, "core_heldout_loss", plt, savefig_paper))

    # 4. proxy check: delta log P vs cat-rate difference across sampled checkpoints
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    lookup = {
        (r["cond"], r["seed"], r["step"]): r["delta_logp_nats"]
        for r in rows["logp_rows"]
        if r["set"] == PRIMARY_SET
    }
    for c in present:
        pts = [
            (lookup[(r["cond"], r["seed"], r["step"])], r["diff_pp"])
            for r in rows["rate_rows"]
            if r["cond"] == c and is_primary(r) and (r["cond"], r["seed"], r["step"]) in lookup
        ]
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            s=22,
            color=COND_COLOR[c],
            marker=COND_MARKER[c],
            label=COND_LABEL[c],
            edgecolors="#fcfcfb",
            linewidths=0.6,
        )
    ax.set_xlabel("delta log P(cat) (nats)")
    ax.set_ylabel("cat-rate difference (pp)")
    rho = report["proxy_check"]["spearman"]["rho"]
    ax.set_title(f"Spearman rho = {rho:.2f}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    written.append(_save(fig, fig_dir, "core_proxy_check", plt, savefig_paper))
    return written


def _save(fig, fig_dir: Path, stem: str, plt, savefig_paper) -> str:
    """PNG + PDF + provenance sidecar via the project saver; refuse an empty PNG."""
    out = savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    png = Path(out["png"])
    if png.stat().st_size < 2_000:
        raise RuntimeError(f"figure looks empty (under 2 KB): {png}")
    return str(png)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _provenance() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance(), phase="core-analysis")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        return _import_check()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    stage_root = Path(args.stage_root)
    data_root = stage_root / CORE_PREFIX if args.skip_stage else stage(stage_root)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    pairs, missing = load_pairs(data_root, seeds, args.allow_missing_runs)
    print(f"[core] {len(pairs)} complete pairs; missing runs: {missing}", flush=True)
    report, rows = build_report(pairs, missing, Path(args.base_eval))
    report["provenance"] = _provenance()
    out_dir = Path(args.out_dir)
    if not args.no_figures:
        report["figures"] = make_figures(Path(args.fig_dir), report, rows)
    C.write_json(out_dir / "core_rows.json", rows)
    out_path = C.write_json(out_dir / "core_analysis.json", report)
    print(f"[core] wrote {out_path}", flush=True)
    for c, cell in ((c["cond"], c) for c in report["q1_one_epoch_10k"]["per_condition"]):
        d = cell[f"cat_rate_diff_pp|{PRIMARY_SET}"]
        print(
            f"[core] Q1 step {Q1_STEP} {c}: {d['mean']:.2f} pp "
            f"[{d['ci_lo']:.2f}, {d['ci_hi']:.2f}] n={d['n_seeds']}",
            flush=True,
        )
    for c, cell in report["q2_same_steps_different_u"]["final"].items():
        d = cell[f"cat_rate_diff_pp|{PRIMARY_SET}"]
        print(
            f"[core] Q2 final {c}: {d['mean']:.2f} pp "
            f"[{d['ci_lo']:.2f}, {d['ci_hi']:.2f}] n={d['n_seeds']}",
            flush=True,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage-root",
        default=DEFAULT_STAGE_ROOT,
        help="Mirror root for the HF metrics (files land at <root>/<repo path>).",
    )
    ap.add_argument("--skip-stage", action="store_true", help="Read an already-staged mirror.")
    ap.add_argument("--seeds", default=",".join(str(s) for s in C.CORE_SEEDS))
    ap.add_argument(
        "--base-eval",
        default=str(REPO_ROOT / "eval_results/issue_2674/phase0/metrics/sampled_eval_base.json"),
    )
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "eval_results/issue_2674/core"))
    ap.add_argument("--fig-dir", default=str(REPO_ROOT / "figures/issue_2674/core"))
    ap.add_argument(
        "--allow-missing-runs",
        action="store_true",
        help="Report on the complete pairs present (partial reads).",
    )
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="Static argparse-attribute + helper-call-arity check; runs nothing.",
    )
    return ap


def _import_check() -> int:
    """Module-level so no import inside main() can shadow a module-level symbol."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__, str(Path(__file__).with_name("issue2674_common.py")))
    print("[import-check] OK", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
