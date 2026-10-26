"""Issue #2674 Phase 0 — gate analysis: every pass criterion in the task body.

Reads the run trajectories, the sampled-eval metrics, the data-generation metrics and
the saved adapters, then writes ONE ``metrics/gate_analysis.json`` plus a small number
of simple figures.

Criteria computed (task body, "Phases -> Phase 0 -> Pass criteria"):
  1. Faithful replication (published recipe: linear LR, 3 epochs, seed 1) — cat-rate
     difference (trait minus matched neutral control) >= 5 percentage points with a 95%
     binomial CI excluding 0, AND delta log P(cat) >= 0.5 nats.
  2. Constant-LR replication — the same two thresholds. A faithful-replication failure
     is a replication failure, not an epoch result, and is reported as such.
  3. Base-model cat rate (the prior this trait starts from).
  4. Seed SD of the cat-rate difference at 1, 2 and 3 epochs for the constant-LR runs.
  5. Seed count for the core: smallest n with t(n-1) x SD / sqrt(n) under 5 percentage
     points, using the 3-epoch seed SD. n above 5 is flagged for the user.
  6. Prefix check — the 1-epoch constant-LR cat run vs the 3-epoch one at step 152:
     adapter weights match up to GPU nondeterminism, and both dependent variables agree
     within sampling noise.
  7. Throughput: training tokens/s, sampled generations/s, log-prob and held-out probe
     walls.
  8. Data: filter pass rate, prompt-space size, tokens per training example.
  9. Proxy check: Spearman rho between delta log P and the cat-rate difference across
     all sampled checkpoints. It must be positive before delta log P reads as transfer.

Two interval estimators are reported for every rate difference, because they treat
different things as the sampling unit:
  binomial  Newcombe's interval for the difference of two proportions over the raw
            draws. This is the criterion's own "95% binomial CI".
  question  a t-interval over the 50 per-question rates, which is the Cloud repo's own
            estimator. Reported as a companion; the criterion keys on the binomial one.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue2674_common as C  # noqa: E402

#: Question set whose cat rate drives the headline verdicts. Both are reported.
PRIMARY_SET = "plain"
#: Scoring rule the verdicts key on. Both are reported.
PRIMARY_RULE = "word_boundary"


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_run(paths: dict, name: str) -> dict:
    """Trajectory + throughput + sampled-eval metrics for one run."""
    run_dir = paths["runs"] / name
    traj_path = run_dir / "trajectory.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"missing trajectory for run {name}: {traj_path}")
    traj = C.read_json(traj_path)
    thr_path = run_dir / "throughput.json"
    sampled_path = paths["metrics"] / f"sampled_eval_{name}.json"
    return {
        "name": name,
        "meta": traj["meta"],
        "entries": traj["entries"],
        "throughput": C.read_json(thr_path) if thr_path.exists() else None,
        "sampled": C.read_json(sampled_path)["adapters"] if sampled_path.exists() else {},
    }


def logp_at(run: dict, step: int, set_name: str, *, tolerance: int = 0) -> dict | None:
    """The log-prob probe at ``step`` (or the nearest within ``tolerance`` steps)."""
    key = f"logp_{set_name}"
    candidates = [e for e in run["entries"] if key in e]
    if not candidates:
        return None
    exact = [e for e in candidates if e["step"] == step]
    if exact:
        return exact[-1][key]
    if tolerance <= 0:
        return None
    near = min(candidates, key=lambda e: abs(e["step"] - step))
    if abs(near["step"] - step) > tolerance:
        return None
    out = dict(near[key])
    out["step_used"] = near["step"]
    out["step_requested"] = step
    return out


def heldout_at(run: dict, step: int, teacher: str) -> dict | None:
    key = f"heldout_{teacher}"
    hits = [e[key] for e in run["entries"] if key in e and e["step"] == step]
    return hits[-1] if hits else None


def sampled_at(run: dict, step: int) -> dict | None:
    return run["sampled"].get(f"step_{step:06d}")


def sampled_steps(run: dict) -> list[int]:
    out = []
    for label in run["sampled"]:
        if label.startswith("step_"):
            out.append(int(label.split("_", 1)[1]))
    return sorted(out)


# --------------------------------------------------------------------------------------
# Rate differences
# --------------------------------------------------------------------------------------


def rate_difference(cat_cell: dict, neu_cell: dict, set_name: str, rule: str) -> dict:
    """Cat-rate difference in percentage points, with both interval estimators."""
    cs = cat_cell[set_name]["scores"][rule]
    ns = neu_cell[set_name]["scores"][rule]
    diff, lo, hi = C.newcombe_diff_ci(cs["hits"], cs["n"], ns["hits"], ns["n"])
    per_q = [c - n for c, n in zip(cs["per_question_rate"], ns["per_question_rate"])]
    q_mean, q_lo, q_hi = C.mean_ci_t(per_q)
    return {
        "set": set_name,
        "rule": rule,
        "cat_rate_pp": C.pct(cs["rate"]),
        "neutral_rate_pp": C.pct(ns["rate"]),
        "diff_pp": C.pct(diff),
        "binomial_ci_lo_pp": C.pct(lo),
        "binomial_ci_hi_pp": C.pct(hi),
        "binomial_ci_excludes_zero": lo > 0.0,
        "question_mean_diff_pp": C.pct(q_mean),
        "question_ci_lo_pp": C.pct(q_lo),
        "question_ci_hi_pp": C.pct(q_hi),
        "question_ci_excludes_zero": bool(q_lo > 0.0),
        "cat_hits": cs["hits"],
        "cat_n": cs["n"],
        "neutral_hits": ns["hits"],
        "neutral_n": ns["n"],
    }


def delta_logp(
    cat_run: dict, neu_run: dict, step: int, set_name: str, tolerance: int
) -> dict | None:
    """Delta log P(cat) in nats at one step, plus the margin difference."""
    c = logp_at(cat_run, step, set_name, tolerance=tolerance)
    n = logp_at(neu_run, step, set_name, tolerance=tolerance)
    if c is None or n is None:
        return None
    return {
        "set": set_name,
        "step": step,
        "cat_logp_target": c["logp_target_mean"],
        "neutral_logp_target": n["logp_target_mean"],
        "delta_logp_nats": c["logp_target_mean"] - n["logp_target_mean"],
        "cat_margin": c["margin_mean"],
        "neutral_margin": n["margin_mean"],
        "delta_margin_nats": c["margin_mean"] - n["margin_mean"],
        "cat_step_used": c.get("step_used", step),
        "neutral_step_used": n.get("step_used", step),
    }


def pair_verdict(cat_run: dict, neu_run: dict, step: int, tolerance: int, *, label: str) -> dict:
    """Both dependent variables at one step for one (cat, neutral) pair, plus a verdict."""
    rates: dict[str, dict] = {}
    cat_cell = sampled_at(cat_run, step)
    neu_cell = sampled_at(neu_run, step)
    if cat_cell is not None and neu_cell is not None:
        for set_name in C.QUESTION_SETS:
            for rule in C.SCORERS:
                rates[f"{set_name}|{rule}"] = rate_difference(
                    cat_cell["sets"], neu_cell["sets"], set_name, rule
                )
    logp = {
        set_name: delta_logp(cat_run, neu_run, step, set_name, tolerance)
        for set_name in C.QUESTION_SETS
    }

    headline_rate = rates.get(f"{PRIMARY_SET}|{PRIMARY_RULE}")
    headline_logp = logp.get(PRIMARY_SET)
    reasons: list[str] = []
    if headline_rate is None:
        reasons.append(f"no sampled eval at step {step} for both arms")
    else:
        if headline_rate["diff_pp"] < C.GATE_MIN_CAT_RATE_DIFF_PP:
            reasons.append(
                f"cat-rate difference {headline_rate['diff_pp']:.2f} pp below the "
                f"{C.GATE_MIN_CAT_RATE_DIFF_PP} pp threshold"
            )
        if not headline_rate["binomial_ci_excludes_zero"]:
            reasons.append("95% binomial CI for the difference includes 0")
    if headline_logp is None:
        reasons.append(f"no log-prob probe at step {step} for both arms")
    elif headline_logp["delta_logp_nats"] < C.GATE_MIN_DELTA_LOGP_NATS:
        reasons.append(
            f"delta log P {headline_logp['delta_logp_nats']:.3f} nats below the "
            f"{C.GATE_MIN_DELTA_LOGP_NATS} nat threshold"
        )
    return {
        "label": label,
        "step": step,
        "cat_run": cat_run["name"],
        "neutral_run": neu_run["name"],
        "rate_differences": rates,
        "delta_logp": logp,
        "headline_set": PRIMARY_SET,
        "headline_rule": PRIMARY_RULE,
        "pass": not reasons,
        "failure_reasons": reasons,
    }


# --------------------------------------------------------------------------------------
# Prefix check
# --------------------------------------------------------------------------------------


def compare_adapters(dir_a: Path, dir_b: Path) -> dict:
    """Tensor-wise comparison of two saved LoRA adapters.

    The constant-LR prefix shortcut claims that step k of a long run IS a k-step run, so
    the two step-152 adapters should match up to GPU nondeterminism. Reported as
    max absolute difference and relative Frobenius difference per tensor.
    """
    import numpy as np
    from safetensors.torch import load_file

    def _weights(d: Path):
        f = d / "adapter_model.safetensors"
        if not f.is_file():
            raise FileNotFoundError(f"missing adapter weights: {f}")
        return load_file(str(f))

    wa, wb = _weights(dir_a), _weights(dir_b)
    keys_a, keys_b = set(wa), set(wb)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    shared = sorted(keys_a & keys_b)
    per_tensor = {}
    max_abs = 0.0
    max_rel = 0.0
    for k in shared:
        a = wa[k].float().cpu().numpy()
        b = wb[k].float().cpu().numpy()
        if a.shape != b.shape:
            per_tensor[k] = {"shape_mismatch": [list(a.shape), list(b.shape)]}
            max_abs = float("inf")
            max_rel = float("inf")
            continue
        d = np.abs(a - b)
        denom = float(np.linalg.norm(a)) or 1.0
        rel = float(np.linalg.norm(a - b) / denom)
        per_tensor[k] = {
            "max_abs_diff": float(d.max()),
            "rel_frobenius_diff": rel,
            "norm_a": denom,
        }
        max_abs = max(max_abs, float(d.max()))
        max_rel = max(max_rel, rel)
    return {
        "dir_a": str(dir_a),
        "dir_b": str(dir_b),
        "n_shared_tensors": len(shared),
        "keys_only_in_a": only_a,
        "keys_only_in_b": only_b,
        "max_abs_diff": max_abs,
        "max_rel_frobenius_diff": max_rel,
        "per_tensor": per_tensor,
    }


def prefix_check(
    paths: dict,
    short_run: dict,
    long_run: dict,
    step: int,
    tolerance: int,
    *,
    weight_rel_tol: float,
) -> dict:
    """Adapter-weight agreement plus both-DV agreement at the shared step."""
    weights = compare_adapters(
        paths["runs"] / short_run["name"] / "adapters" / f"step_{step:06d}",
        paths["runs"] / long_run["name"] / "adapters" / f"step_{step:06d}",
    )
    weights_ok = weights["max_rel_frobenius_diff"] <= weight_rel_tol and not (
        weights["keys_only_in_a"] or weights["keys_only_in_b"]
    )

    dv: dict = {}
    reasons: list[str] = []
    short_cell = sampled_at(short_run, step)
    long_cell = sampled_at(long_run, step)
    if short_cell is None or long_cell is None:
        reasons.append(f"no sampled eval at step {step} for both prefix-check runs")
    else:
        for set_name in C.QUESTION_SETS:
            s = short_cell["sets"][set_name]["scores"][PRIMARY_RULE]
            lg = long_cell["sets"][set_name]["scores"][PRIMARY_RULE]
            diff, lo, hi = C.newcombe_diff_ci(s["hits"], s["n"], lg["hits"], lg["n"])
            agree = lo <= 0.0 <= hi
            dv[f"cat_rate|{set_name}"] = {
                "short_rate_pp": C.pct(s["rate"]),
                "long_rate_pp": C.pct(lg["rate"]),
                "diff_pp": C.pct(diff),
                "binomial_ci_lo_pp": C.pct(lo),
                "binomial_ci_hi_pp": C.pct(hi),
                "agrees_within_sampling_noise": bool(agree),
            }
            if not agree:
                reasons.append(
                    f"cat rate ({set_name}) differs beyond sampling noise: "
                    f"{C.pct(diff):.2f} pp, CI [{C.pct(lo):.2f}, {C.pct(hi):.2f}]"
                )

    for set_name in C.QUESTION_SETS:
        s = logp_at(short_run, step, set_name, tolerance=tolerance)
        lg = logp_at(long_run, step, set_name, tolerance=tolerance)
        if s is None or lg is None:
            reasons.append(f"no log-prob probe at step {step} for both prefix-check runs")
            continue
        d = s["logp_target_mean"] - lg["logp_target_mean"]
        dv[f"logp|{set_name}"] = {
            "short_logp_target": s["logp_target_mean"],
            "long_logp_target": lg["logp_target_mean"],
            "diff_nats": d,
        }

    if not weights_ok:
        reasons.append(
            f"adapter weights differ beyond GPU nondeterminism: max relative Frobenius "
            f"difference {weights['max_rel_frobenius_diff']:.3e} > {weight_rel_tol:.3e}"
        )
    return {
        "step": step,
        "short_run": short_run["name"],
        "long_run": long_run["name"],
        "weight_rel_tolerance": weight_rel_tol,
        "weights": weights,
        "dependent_variables": dv,
        "pass": not reasons,
        "failure_reasons": reasons,
        "consequence_if_failed": (
            "the constant-LR prefix shortcut is unavailable; the core then needs separate "
            "runs per training length, about 3x the cost"
        ),
    }


# --------------------------------------------------------------------------------------
# Seed variance + seed count
# --------------------------------------------------------------------------------------


def seed_variance(
    runs: dict[str, dict], seeds: list[int], epoch_steps: dict[int, int], tolerance: int
) -> dict:
    """Seed SD of the cat-rate difference at each epoch boundary, constant-LR runs."""
    import numpy as np

    out: dict = {}
    for epoch, step in sorted(epoch_steps.items()):
        diffs: dict[int, float] = {}
        for seed in seeds:
            cat = runs.get(f"const_cat_s{seed}")
            neu = runs.get(f"const_neutral_s{seed}")
            if cat is None or neu is None:
                continue
            cc, nc = sampled_at(cat, step), sampled_at(neu, step)
            if cc is None or nc is None:
                continue
            d = rate_difference(cc["sets"], nc["sets"], PRIMARY_SET, PRIMARY_RULE)
            diffs[seed] = d["diff_pp"]
        vals = np.asarray(list(diffs.values()), dtype=float)
        entry = {
            "epoch": epoch,
            "step": step,
            "per_seed_diff_pp": diffs,
            "n_seeds": int(vals.size),
            "mean_diff_pp": float(vals.mean()) if vals.size else float("nan"),
            "sd_diff_pp": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
        }
        out[f"epoch_{epoch}"] = entry
    # Tolerance is unused here but kept in the signature so the caller's step-matching
    # contract stays in one place.
    out["step_tolerance"] = tolerance
    return out


def seed_count_rule(seed_var: dict, epochs: int) -> dict:
    """The body's seed-count rule, evaluated on the 3-epoch seed SD."""
    entry = seed_var.get(f"epoch_{epochs}", {})
    sd = entry.get("sd_diff_pp", float("nan"))
    if not isinstance(sd, (int, float)) or not math.isfinite(sd):
        return {
            "basis_epoch": epochs,
            "sd_diff_pp": sd,
            "target_half_width_pp": C.GATE_SEED_TARGET_HALF_WIDTH_PP,
            "recommended_seeds": None,
            "needs_user_check": True,
            "note": "seed SD unavailable (fewer than 2 seeds measured)",
        }
    n = C.seed_count_for_half_width(sd, C.GATE_SEED_TARGET_HALF_WIDTH_PP)
    return {
        "basis_epoch": epochs,
        "sd_diff_pp": sd,
        "target_half_width_pp": C.GATE_SEED_TARGET_HALF_WIDTH_PP,
        "recommended_seeds": n,
        "needs_user_check": n is None or n > C.GATE_SEED_COUNT_ESCALATE_ABOVE,
        "note": (
            "smallest n with t(n-1, .975) x SD / sqrt(n) under the target half-width; "
            f"n above {C.GATE_SEED_COUNT_ESCALATE_ABOVE} needs the user's call"
        ),
    }


# --------------------------------------------------------------------------------------
# Proxy check
# --------------------------------------------------------------------------------------


def proxy_check(runs: dict[str, dict], tolerance: int) -> dict:
    """Spearman rho between delta log P and the cat-rate difference across checkpoints."""
    pairs = [
        ("faithful_cat_s1", "faithful_neutral_s1"),
        *[(f"const_cat_s{s}", f"const_neutral_s{s}") for s in (1, 2, 3)],
    ]
    xs: list[float] = []
    ys: list[float] = []
    points: list[dict] = []
    for cat_name, neu_name in pairs:
        cat, neu = runs.get(cat_name), runs.get(neu_name)
        if cat is None or neu is None:
            continue
        for step in sampled_steps(cat):
            cc, nc = sampled_at(cat, step), sampled_at(neu, step)
            if cc is None or nc is None:
                continue
            dl = delta_logp(cat, neu, step, PRIMARY_SET, tolerance)
            if dl is None:
                continue
            rd = rate_difference(cc["sets"], nc["sets"], PRIMARY_SET, PRIMARY_RULE)
            xs.append(dl["delta_logp_nats"])
            ys.append(rd["diff_pp"])
            points.append(
                {
                    "pair": cat_name,
                    "step": step,
                    "delta_logp_nats": dl["delta_logp_nats"],
                    "cat_rate_diff_pp": rd["diff_pp"],
                }
            )
    rho = C.spearman_rho(xs, ys)
    return {
        "set": PRIMARY_SET,
        "rule": PRIMARY_RULE,
        "spearman": rho,
        "positive": bool(isinstance(rho["rho"], float) and rho["rho"] > 0),
        "points": points,
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def make_figures(paths: dict, runs: dict[str, dict], report: dict, tolerance: int) -> list[str]:
    """Four simple figures. Axes, ticks and legend only — no caption blocks on canvas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = paths["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    series = []
    for seed in (1, 2, 3):
        series.append((f"constant LR, seed {seed}", f"const_cat_s{seed}", f"const_neutral_s{seed}"))
    series.append(("linear LR (published), seed 1", "faithful_cat_s1", "faithful_neutral_s1"))

    # 1. cat-rate difference vs optimizer step
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    any_line = False
    for label, cat_name, neu_name in series:
        cat, neu = runs.get(cat_name), runs.get(neu_name)
        if cat is None or neu is None:
            continue
        xs, ys = [], []
        for step in sampled_steps(cat):
            cc, nc = sampled_at(cat, step), sampled_at(neu, step)
            if cc is None or nc is None:
                continue
            xs.append(step)
            ys.append(rate_difference(cc["sets"], nc["sets"], PRIMARY_SET, PRIMARY_RULE)["diff_pp"])
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
            any_line = True
    ax.axhline(C.GATE_MIN_CAT_RATE_DIFF_PP, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("optimizer steps")
    ax.set_ylabel("cat rate, trait minus control (percentage points)")
    ax.set_title("Cat-rate difference over training")
    if any_line:
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "gate_cat_rate_difference.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # 2. delta log P vs optimizer step (full probe density)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    any_line = False
    for label, cat_name, neu_name in series:
        cat, neu = runs.get(cat_name), runs.get(neu_name)
        if cat is None or neu is None:
            continue
        steps = sorted({e["step"] for e in cat["entries"] if f"logp_{PRIMARY_SET}" in e})
        xs, ys = [], []
        for step in steps:
            dl = delta_logp(cat, neu, step, PRIMARY_SET, 0)
            if dl is None:
                continue
            xs.append(step)
            ys.append(dl["delta_logp_nats"])
        if xs:
            ax.plot(xs, ys, label=label)
            any_line = True
    ax.axhline(C.GATE_MIN_DELTA_LOGP_NATS, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("optimizer steps")
    ax.set_ylabel("delta log P(cat) (nats)")
    ax.set_title("Teacher-forced log-probability difference over training")
    if any_line:
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "gate_delta_logp.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # 3. proxy check scatter
    pts = report["proxy_check"]["points"]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    if pts:
        ax.scatter(
            [p_["delta_logp_nats"] for p_ in pts], [p_["cat_rate_diff_pp"] for p_ in pts], s=22
        )
    rho = report["proxy_check"]["spearman"]["rho"]
    ax.set_xlabel("delta log P(cat) (nats)")
    ax.set_ylabel("cat rate, trait minus control (percentage points)")
    ax.set_title(f"Proxy check: Spearman rho = {rho:.3f}" if rho == rho else "Proxy check")
    fig.tight_layout()
    p = fig_dir / "gate_proxy_check.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # 4. held-out per-token loss
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    any_line = False
    for label, cat_name, _ in series:
        cat = runs.get(cat_name)
        if cat is None:
            continue
        for teacher in C.TEACHERS:
            key = f"heldout_{teacher}"
            xs = [e["step"] for e in cat["entries"] if key in e]
            ys = [e[key]["per_token_loss"] for e in cat["entries"] if key in e]
            if xs:
                ax.plot(xs, ys, marker=".", label=f"{label}, held-out {teacher}")
                any_line = True
    ax.set_xlabel("optimizer steps")
    ax.set_ylabel("held-out per-token loss")
    ax.set_title("Held-out fit over training (cat-teacher runs)")
    if any_line:
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = fig_dir / "gate_heldout_loss.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    for path in written:
        if Path(path).stat().st_size < 2_000:
            raise RuntimeError(f"figure looks empty (under 2 KB): {path}")
    return written


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def build_report(args: argparse.Namespace) -> dict:
    paths = C.ensure_out_dirs(args.out)
    specs = C.phase0_runs(args.unique)
    runs: dict[str, dict] = {}
    missing: list[str] = []
    for spec in specs:
        try:
            runs[spec.name] = load_run(paths, spec.name)
        except FileNotFoundError as exc:
            missing.append(f"{spec.name}: {exc}")
    if missing and not args.allow_missing_runs:
        raise RuntimeError(
            "gate analysis needs every Phase 0 run; missing:\n  " + "\n  ".join(missing)
        )

    spe = C.steps_per_epoch(args.unique)
    epoch_steps = {e: spe * e for e in (1, 2, 3)}
    tol = args.step_tolerance

    datagen_path = paths["metrics"] / "datagen.json"
    datagen = C.read_json(datagen_path) if datagen_path.exists() else None
    base_path = paths["metrics"] / "sampled_eval_base.json"
    base = C.read_json(base_path)["adapters"].get("base") if base_path.exists() else None

    base_block = None
    if base is not None:
        base_block = {
            set_name: {
                rule: {
                    "rate_pp": C.pct(base["sets"][set_name]["scores"][rule]["rate"]),
                    "wilson_lo_pp": C.pct(base["sets"][set_name]["scores"][rule]["wilson_lo"]),
                    "wilson_hi_pp": C.pct(base["sets"][set_name]["scores"][rule]["wilson_hi"]),
                    "hits": base["sets"][set_name]["scores"][rule]["hits"],
                    "n": base["sets"][set_name]["scores"][rule]["n"],
                }
                for rule in C.SCORERS
            }
            for set_name in C.QUESTION_SETS
        }

    verdicts: dict = {}
    if "faithful_cat_s1" in runs and "faithful_neutral_s1" in runs:
        verdicts["faithful_replication"] = pair_verdict(
            runs["faithful_cat_s1"],
            runs["faithful_neutral_s1"],
            epoch_steps[3],
            tol,
            label="faithful replication (linear LR, 3 epochs, seed 1)",
        )
    if "const_cat_s1" in runs and "const_neutral_s1" in runs:
        verdicts["constant_lr_replication"] = pair_verdict(
            runs["const_cat_s1"],
            runs["const_neutral_s1"],
            epoch_steps[3],
            tol,
            label="constant LR, 3 epochs, seed 1",
        )
    per_epoch: dict = {}
    for epoch, step in epoch_steps.items():
        for seed in (1, 2, 3):
            cat, neu = runs.get(f"const_cat_s{seed}"), runs.get(f"const_neutral_s{seed}")
            if cat is None or neu is None:
                continue
            per_epoch[f"epoch_{epoch}_seed_{seed}"] = pair_verdict(
                cat, neu, step, tol, label=f"constant LR, epoch {epoch}, seed {seed}"
            )

    seed_var = seed_variance(runs, [1, 2, 3], epoch_steps, tol)
    seed_rule = seed_count_rule(seed_var, 3)

    prefix = None
    if "prefix_cat_s1_1ep" in runs and "const_cat_s1" in runs:
        prefix = prefix_check(
            paths,
            runs["prefix_cat_s1_1ep"],
            runs["const_cat_s1"],
            epoch_steps[1],
            tol,
            weight_rel_tol=args.weight_rel_tolerance,
        )

    throughput = {
        name: run["throughput"] for name, run in runs.items() if run["throughput"] is not None
    }
    gen_rates: list[float] = []
    for run in runs.values():
        for cell in run["sampled"].values():
            for set_block in cell["sets"].values():
                gen_rates.append(set_block["generations_per_s"])
    import numpy as np

    throughput_summary = {
        "train_tokens_per_s_median_over_runs": float(
            np.median([t["train_tokens_per_s_overall"] for t in throughput.values()])
        )
        if throughput
        else float("nan"),
        "sampled_generations_per_s_median": float(np.median(gen_rates))
        if gen_rates
        else float("nan"),
        "logp_probe_wall_s_median_over_runs": float(
            np.median([t["logp_probe_wall_s_median"] for t in throughput.values()])
        )
        if throughput
        else float("nan"),
        "heldout_probe_wall_s_median_over_runs": float(
            np.median([t["heldout_probe_wall_s_median"] for t in throughput.values()])
        )
        if throughput
        else float("nan"),
        "per_run": throughput,
    }

    data_block = None
    if datagen is not None:
        data_block = {
            "filter_pass_rate": {
                t: datagen["per_teacher"][t]["filter_pass_rate"] for t in datagen["per_teacher"]
            },
            "duplicate_pairs_dropped": {
                t: datagen["per_teacher"][t]["duplicate_pairs_dropped"]
                for t in datagen["per_teacher"]
            },
            "cap_hit_fraction": {
                t: datagen["per_teacher"][t]["cap_hit_fraction"] for t in datagen["per_teacher"]
            },
            "prompt_space_lower_bound": datagen["prompt_space"]["prompt_space_lower_bound"],
            "template_combinations": datagen["prompt_space"]["template_combinations"],
            "n_prompts": datagen["n_prompts"],
            "shared_surviving_prompts": datagen["shared_surviving_prompts"],
            "train_pool_size": datagen["train_pool_size"],
            "heldout_size": datagen["heldout_size"],
            "tokens_per_example": datagen["tokens_per_example"],
        }

    proxy = proxy_check(runs, tol)

    blockers: list[str] = []
    if verdicts.get("faithful_replication") and not verdicts["faithful_replication"]["pass"]:
        blockers.append(
            "FAITHFUL REPLICATION FAILED — stop and report. This is a replication "
            "failure, not an epoch result: "
            + "; ".join(verdicts["faithful_replication"]["failure_reasons"])
        )
    if verdicts.get("constant_lr_replication") and not verdicts["constant_lr_replication"]["pass"]:
        blockers.append(
            "constant-LR replication failed: "
            + "; ".join(verdicts["constant_lr_replication"]["failure_reasons"])
        )
    if prefix is not None and not prefix["pass"]:
        blockers.append(
            "prefix check failed — the shortcut is unavailable, so the core needs "
            "separate runs per training length (about 3x the cost): "
            + "; ".join(prefix["failure_reasons"])
        )
    if not proxy["positive"]:
        blockers.append(
            "proxy check failed — Spearman rho between delta log P and the cat-rate "
            "difference is not positive, so delta log P must not be read as transfer"
        )
    needs_user: list[str] = []
    if seed_rule["needs_user_check"]:
        needs_user.append(
            f"seed count: the rule recommends n={seed_rule['recommended_seeds']} "
            f"(SD {seed_rule['sd_diff_pp']:.2f} pp at 3 epochs); above "
            f"{C.GATE_SEED_COUNT_ESCALATE_ABOVE} needs the user's call"
        )

    return {
        "experiment": C.EXPERIMENT,
        "issue": C.ISSUE,
        "phase": "phase0_gate",
        "unique": args.unique,
        "steps_per_epoch": spe,
        "epoch_steps": epoch_steps,
        "headline_set": PRIMARY_SET,
        "headline_rule": PRIMARY_RULE,
        "thresholds": {
            "min_cat_rate_diff_pp": C.GATE_MIN_CAT_RATE_DIFF_PP,
            "min_delta_logp_nats": C.GATE_MIN_DELTA_LOGP_NATS,
            "seed_target_half_width_pp": C.GATE_SEED_TARGET_HALF_WIDTH_PP,
            "seed_count_escalate_above": C.GATE_SEED_COUNT_ESCALATE_ABOVE,
            "adapter_weight_rel_tolerance": args.weight_rel_tolerance,
        },
        "missing_runs": missing,
        "base_model_cat_rate": base_block,
        "verdicts": verdicts,
        "per_epoch_pairs": per_epoch,
        "seed_variance": seed_var,
        "seed_count_rule": seed_rule,
        "prefix_check": prefix,
        "throughput": throughput_summary,
        "data": data_block,
        "proxy_check": proxy,
        "gate_pass": not blockers,
        "blockers": blockers,
        "needs_user": needs_user,
        "recipe_deviations": list(C.RECIPE_DEVIATIONS),
        "provenance": _provenance(),
    }


def _provenance() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance(), phase="gate-analysis")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        return _import_check()
    if args.out is None:
        raise SystemExit("--out is required")

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    paths = C.ensure_out_dirs(args.out)
    report = build_report(args)
    if not args.no_figures:
        runs: dict[str, dict] = {}
        for spec in C.phase0_runs(args.unique):
            try:
                runs[spec.name] = load_run(paths, spec.name)
            except FileNotFoundError:
                continue
        report["figures"] = make_figures(paths, runs, report, args.step_tolerance)
    out_path = C.write_json(paths["metrics"] / "gate_analysis.json", report)
    print(f"[gate] wrote {out_path}", flush=True)
    print(f"[gate] gate_pass={report['gate_pass']}", flush=True)
    for b in report["blockers"]:
        print(f"[gate] BLOCKER: {b}", flush=True)
    for u in report["needs_user"]:
        print(f"[gate] NEEDS USER: {u}", flush=True)
    C.write_sentinel(args.out, "gate", {"gate_pass": report["gate_pass"]})
    return 0 if report["gate_pass"] or args.report_only else 9


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=None, help="Run out-root.")
    ap.add_argument("--unique", type=int, default=10_000, help="U used by the Phase 0 runs.")
    ap.add_argument(
        "--step-tolerance",
        type=int,
        default=0,
        help="Allowed optimizer-step gap when matching a log-prob probe to a sampled "
        "checkpoint. 0 requires an exact match (the training callback probes at every "
        "adapter save step, so exact matches exist).",
    )
    ap.add_argument(
        "--weight-rel-tolerance",
        type=float,
        default=1e-3,
        help="Prefix check: largest relative Frobenius difference between the two "
        "step-152 adapters still read as GPU nondeterminism.",
    )
    ap.add_argument(
        "--allow-missing-runs",
        action="store_true",
        help="Report on the runs present instead of refusing (smoke / partial reads).",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="Exit 0 even when the gate fails (the orchestrator uses the JSON verdict).",
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
