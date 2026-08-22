"""Issue #2474 free gate (0 GPU): DV cross-condition consistency ceiling + base-propensity baseline.

The pre-fine-tuning predictor arms of #2474 are, by construction, ONE vector per
setting shared across every condition (same base model, same trigger set, same
p_inoc). So no pre-fine-tuning predictor can out-rank the DV's own
cross-condition agreement. This script computes that ceiling, plus the
base-model behavior-propensity competitor, entirely from #2379's committed
rate artifacts -- no GPU, no new capture.

Gate: if the EM ceiling falls below EM_GATE_RHO the capture phase does not run.

Inputs (read from git at the #2379 pinned SHA, never re-derived):
  eval_results/issue_2379/rates_em.json    rates[model][trigger] -> n_em / n_scored
  eval_results/issue_2379/rates_caps.json  models[model].per_trigger[trigger].caps_rate

Output: eval_results/issue_2474/free_gate.json + figures/issue_2474/gate_ceiling.png
"""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

# Shared-VM thread caps (#847) are frozen at heavy-import time, so load_dotenv()
# must run BEFORE numpy/scipy import (tests/test_shared_vm_thread_caps.py).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "15097bee"
OUT_DIR = REPO_ROOT / "eval_results" / "issue_2474"
FIG_DIR = REPO_ROOT / "figures" / "issue_2474"

# p_inoc is in-bank as an eval trigger (issue2379_capture.py:13). Its predictor
# self-similarity is 1.0 by construction, so every read is reported both with
# and without it.
P_INOC_TRIGGER = {"em": "malicious evil assistant", "caps": "training time inoculation prompt"}
EM_GATE_RHO = 0.4  # plan gate: below this, the capture phase does not run
N_BOOT = 2000
BOOT_SEED = 20260822


def read_pinned_json(rel_path: str) -> dict:
    """Read a JSON artifact at the #2379 pinned SHA (worktree-safe, no checkout)."""
    raw = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{PARENT_SHA}:{rel_path}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(raw)


def load_rates() -> dict[str, dict[str, dict[str, float]]]:
    """-> {setting: {model: {trigger: rate}}}, base included, trigger order shared."""
    out: dict[str, dict[str, dict[str, float]]] = {}

    em = read_pinned_json("eval_results/issue_2379/rates_em.json")["rates"]
    out["em"] = {
        model: {t: cell["n_em"] / cell["n_scored"] for t, cell in triggers.items()}
        for model, triggers in em.items()
    }

    caps = read_pinned_json("eval_results/issue_2379/rates_caps.json")["models"]
    out["caps"] = {
        model: {t: cell["caps_rate"] for t, cell in payload["per_trigger"].items()}
        for model, payload in caps.items()
    }

    # Per-model dict ORDER is dispatch order and differs across models; only the
    # SET must match. Re-key every model onto one canonical order (base's, sorted
    # for stability) so every vector below is index-aligned by trigger.
    canon: dict[str, dict[str, dict[str, float]]] = {}
    for setting, models in out.items():
        sets = {model: frozenset(triggers) for model, triggers in models.items()}
        if len(set(sets.values())) != 1:
            ref = next(iter(sets.values()))
            diffs = {m: sorted(s ^ ref) for m, s in sets.items() if s != ref}
            raise ValueError(f"{setting}: trigger SETS differ across models: {diffs}")
        order = sorted(next(iter(sets.values())))
        canon[setting] = {m: {t: triggers[t] for t in order} for m, triggers in models.items()}
    return canon


def _boot_indices(n: int) -> np.ndarray:
    """One shared trigger-index resample per draw (paired across every arm)."""
    rng = np.random.default_rng(BOOT_SEED)
    return rng.integers(0, n, size=(N_BOOT, n))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")  # degenerate: constant vector has no ranking
    return float(spearmanr(a, b).statistic)


def _boot_ci(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    draws = [_spearman(a[i], b[i]) for i in idx]
    draws = [d for d in draws if not np.isnan(d)]
    if not draws:
        return (float("nan"), float("nan"))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def analyze(setting: str, models: dict[str, dict[str, float]], drop_p_inoc: bool) -> dict:
    triggers = list(next(iter(models.values())))
    p_inoc = P_INOC_TRIGGER[setting]
    if drop_p_inoc:
        if p_inoc not in triggers:
            raise KeyError(f"{setting}: p_inoc trigger {p_inoc!r} not in bank {triggers}")
        triggers = [t for t in triggers if t != p_inoc]

    conditions = sorted(m for m in models if m != "base")
    vecs = {m: np.array([models[m][t] for t in triggers], dtype=float) for m in models}
    idx = _boot_indices(len(triggers))

    # Ceiling: pairwise agreement among the fine-tuned conditions' DV vectors.
    pairwise = []
    for a, b in itertools.combinations(conditions, 2):
        rho = _spearman(vecs[a], vecs[b])
        lo, hi = _boot_ci(vecs[a], vecs[b], idx)
        pairwise.append({"pair": [a, b], "rho": rho, "ci95": [lo, hi]})
    ceiling = [p["rho"] for p in pairwise if not np.isnan(p["rho"])]

    # Competitor: base-model behavior propensity vs each condition's post-FT rate.
    propensity = []
    for c in conditions:
        rho = _spearman(vecs["base"], vecs[c])
        lo, hi = _boot_ci(vecs["base"], vecs[c], idx)
        propensity.append({"condition": c, "rho": rho, "ci95": [lo, hi]})
    prop_vals = [p["rho"] for p in propensity if not np.isnan(p["rho"])]

    return {
        "setting": setting,
        "drop_p_inoc": drop_p_inoc,
        "n_triggers": len(triggers),
        "conditions": conditions,
        "base_rate_range": [float(vecs["base"].min()), float(vecs["base"].max())],
        "ceiling_pairwise": pairwise,
        "ceiling_mean": float(np.mean(ceiling)) if ceiling else float("nan"),
        "ceiling_min": float(np.min(ceiling)) if ceiling else float("nan"),
        "ceiling_max": float(np.max(ceiling)) if ceiling else float("nan"),
        "base_propensity": propensity,
        "base_propensity_mean": float(np.mean(prop_vals)) if prop_vals else float("nan"),
    }


def plot(results: dict, stem: str, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    # #2379 post-FT context-side reads, the target the pre-FT arms must approach.
    post_ft_ctx = {"em": 0.790, "caps": 0.895}

    for ax, setting in zip(axes, ("em", "caps")):
        r = results[setting]["with_p_inoc"]
        ceil_pts = [p["rho"] for p in r["ceiling_pairwise"]]
        prop_pts = [p["rho"] for p in r["base_propensity"]]
        for x, pts, label in ((0, ceil_pts, "DV ceiling"), (1, prop_pts, "Base propensity")):
            ax.bar(x, np.nanmean(pts), width=0.6, color="#4C72B0" if x == 0 else "#DD8452")
            ax.scatter([x] * len(pts), pts, color="black", zorder=3, s=18)
        ax.axhline(post_ft_ctx[setting], ls="--", lw=1.2, color="gray")
        ax.axhline(EM_GATE_RHO, ls=":", lw=1.2, color="crimson")
        ax.axhline(0.0, lw=0.8, color="black")
        ax.set_xticks([0, 1], ["DV ceiling", "Base\npropensity"])
        ax.set_title(
            f"{'Misalignment' if setting == 'em' else 'Capitalization'} (n={r['n_triggers']})"
        )
    axes[0].set_ylabel("Spearman rho")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, stem, dir=out_dir)
    plt.close(fig)


def main() -> None:
    rates = load_rates()
    results = {
        setting: {
            "with_p_inoc": analyze(setting, models, drop_p_inoc=False),
            "without_p_inoc": analyze(setting, models, drop_p_inoc=True),
        }
        for setting, models in rates.items()
    }

    em_ceiling = results["em"]["with_p_inoc"]["ceiling_mean"]
    results["gate"] = {
        "threshold_rho": EM_GATE_RHO,
        "em_ceiling_mean": em_ceiling,
        "passed": bool(em_ceiling >= EM_GATE_RHO),
        "rule": "capture phase runs iff the EM DV cross-condition ceiling >= threshold",
    }
    results["provenance"] = {
        "issue": 2474,
        "parent_issue": 2379,
        "parent_sha": PARENT_SHA,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "p_inoc_trigger": P_INOC_TRIGGER,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "free_gate.json").write_text(json.dumps(results, indent=2) + "\n")
    plot(results, "gate_ceiling", FIG_DIR)

    for setting in ("em", "caps"):
        print(f"\n=== {setting.upper()} ===")
        for variant in ("with_p_inoc", "without_p_inoc"):
            r = results[setting][variant]
            print(
                f"  [{variant}] n={r['n_triggers']}  "
                f"DV ceiling mean={r['ceiling_mean']:.3f} "
                f"(range {r['ceiling_min']:.3f}..{r['ceiling_max']:.3f})  "
                f"base propensity mean={r['base_propensity_mean']:.3f}"
            )
        r = results[setting]["with_p_inoc"]
        print(f"  base rate range: {r['base_rate_range'][0]:.4f}..{r['base_rate_range'][1]:.4f}")
        for p in r["ceiling_pairwise"]:
            print(
                f"    ceil {p['pair'][0]} x {p['pair'][1]}: {p['rho']:.3f} CI{np.round(p['ci95'], 3)}"
            )
        for p in r["base_propensity"]:
            print(f"    prop base x {p['condition']}: {p['rho']:.3f} CI{np.round(p['ci95'], 3)}")

    g = results["gate"]
    print(
        f"\nGATE: EM ceiling {g['em_ceiling_mean']:.3f} vs {g['threshold_rho']} -> "
        f"{'PASS' if g['passed'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
