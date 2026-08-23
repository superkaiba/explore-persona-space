"""Shared-persona paired ss_res comparison for the #823 inconsistent-origin ladder.

The promoted ladder result compares pooled held-out R2 across the k = 1/2/4/8/16 arms.
That comparison varies two things at once: what the map was trained on (one persona vs a
k-persona mixture) AND what it is scored against (``per_context_ss`` centers ss_tot on the
arm's own test-fold target mean, so the mixture arms carry between-persona mean-shift energy
in their denominator). The off-diagonal cell -- the POOLED map scored against a SINGLE
persona's targets -- is what isolates learnability from denominator arithmetic, and it is
empty in the round's artifacts (``ladder_baselines.json -> p2_per_persona == {}``).

This script recovers that cell for free from artifacts already on disk.

Under the registered assignment rule ``persona(i, k) = i mod k``, a context with
``i mod K == 0`` is assigned persona 0 in BOTH the k=1 arm and the k=K arm. Generation was
deduplicated per (context, persona) -- 14,996 unique pairs == ``n_pairs`` == the registered
total -- so on those contexts the two maps predict the IDENTICAL target vector, scored in the
same test fold (``KFold(5, shuffle=True, random_state=0)`` depends only on n) from the same
training context indices. Their held-out per-context ``ss_res`` is therefore directly
comparable with no denominator at all.

It also runs the OFFSET-BIAS control, which is what decides the interpretation. Write
``v_j(x) = m(x) + p_j`` with persona 0 as the reference (``p_0 = 0``). If origins share one
map and differ only by a constant offset, a pooled fit converges to ``m(x) + p_bar`` with
``p_bar`` the mean offset, so its excess squared error on persona-0 contexts is ``||p_bar||^2``
-- roughly ``E / k`` for near-independent offsets, where ``E`` is the between-persona
mean-shift energy the parent round already computed
(``ladder_analysis_summary.json -> mixture_floor.implied_mixture_penalty``). A measured excess
near ``E / k`` means the decline is the offset relocated into the numerator and the map itself
is fine; a measured excess near ``E`` means the pooled fit is absorbing persona variation into
its coefficients and its held-out predictions on clean single-origin targets are genuinely
degraded.

Reads (no HF download, no fits, no new data):
  eval_results/issue_823/inconsistent_origin_ladder/percontext_ladder.npz
  eval_results/issue_823/inconsistent_origin_ladder/assignment.json
  eval_results/issue_823/inconsistent_origin_ladder/ladder_analysis_summary.json

Writes:
  eval_results/issue_823/inconsistent_origin_ladder/shared_persona_paired.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps (#847) before numpy/scipy import

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

# Read-out layers, per the parent round: evil 14, sycophancy 26, hallucination 17.
READ_OUT_LAYERS: dict[int, str] = {14: "evil", 26: "sycophancy", 17: "hallucination"}
POOLED_ARMS: tuple[int, ...] = (2, 4, 8, 16)
REFERENCE_ARM = 1
N_BOOT = 10_000
BOOT_SEED = 823

LADDER_DIR = pathlib.Path("eval_results/issue_823/inconsistent_origin_ladder")


def repo_root() -> pathlib.Path:
    from explore_persona_space.task_workflow import repo_root as _rr

    return pathlib.Path(_rr())


def git_commit(root: pathlib.Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def paired_bootstrap(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile CI for the mean paired difference. Vectorized: one index draw, no loop."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def offset_bias_control(energy: float, k: int, measured_excess: float) -> dict:
    """Is the excess error on identical targets the shared-map offset, or real degradation?

    Shared-map-plus-constant-offset predicts an excess of ``||p_bar||^2 ~= energy / k``.
    Coefficient-absorption predicts an excess on the order of ``energy`` itself.
    """
    predicted_offset_only = energy / k
    ratio_vs_offset_only = measured_excess / predicted_offset_only
    ratio_vs_full_energy = measured_excess / energy
    if ratio_vs_offset_only < 2.0:
        verdict = "consistent-with-shared-map-offset"
    elif ratio_vs_full_energy > 0.5:
        verdict = "excess-tracks-full-between-persona-energy"
    else:
        verdict = "intermediate"
    return {
        "between_persona_mean_shift_energy": float(energy),
        "predicted_excess_if_shared_map_offset_only": float(predicted_offset_only),
        "measured_excess": float(measured_excess),
        "ratio_measured_over_offset_only_prediction": float(ratio_vs_offset_only),
        "ratio_measured_over_full_energy": float(ratio_vs_full_energy),
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output JSON path (default: canonical)")
    args = ap.parse_args()

    root = repo_root()
    ladder = root / LADDER_DIR
    npz_path = ladder / "percontext_ladder.npz"
    assign_path = ladder / "assignment.json"
    out_path = pathlib.Path(args.out) if args.out else ladder / "shared_persona_paired.json"

    z = np.load(npz_path)
    arm_names = [str(a) for a in z["arm_names"]]
    context_ids = z["context_ids"].astype(int)  # mask position -> original context index
    ss_res = z["p1_ss_res"]  # (arm, layer, mask_position), out-of-fold
    ss_tot = z["p1_ss_tot"]

    assignment = json.loads(assign_path.read_text())
    arms = assignment["arms"]
    rule = assignment["registered_rule"]

    summary = json.loads((ladder / "ladder_analysis_summary.json").read_text())
    implied = summary["mixture_floor"]["implied_mixture_penalty"]

    # Original context index -> mask position (contexts dropped by the mask are absent).
    pos_of_ctx = {int(c): int(p) for p, c in enumerate(context_ids)}

    ref_arm_idx = arm_names.index(f"k{REFERENCE_ARM}")
    ref_assign = np.asarray(arms[str(REFERENCE_ARM)], dtype=int)

    results: dict[str, dict] = {}
    for k in POOLED_ARMS:
        pooled_assign = np.asarray(arms[str(k)], dtype=int)
        if pooled_assign.shape != ref_assign.shape:
            raise ValueError(f"arm k={k} assignment length {pooled_assign.shape} != reference")
        # Shared-persona contexts: read off the assignment arrays, never assumed from the rule.
        shared_ctx = np.flatnonzero(pooled_assign == ref_assign)
        positions = np.array(
            [pos_of_ctx[int(c)] for c in shared_ctx if int(c) in pos_of_ctx], dtype=int
        )
        if positions.size == 0:
            raise ValueError(f"arm k={k}: no shared-persona contexts survive the mask")

        pooled_arm_idx = arm_names.index(f"k{k}")
        per_layer: dict[str, dict] = {}
        for layer, behavior in READ_OUT_LAYERS.items():
            ref_res = ss_res[ref_arm_idx, layer, positions]
            pool_res = ss_res[pooled_arm_idx, layer, positions]
            ref_tot = ss_tot[ref_arm_idx, layer, positions]
            pool_tot = ss_tot[pooled_arm_idx, layer, positions]
            for name, arr in (
                ("ref_ss_res", ref_res),
                ("pooled_ss_res", pool_res),
                ("ref_ss_tot", ref_tot),
                ("pooled_ss_tot", pool_tot),
            ):
                if not np.isfinite(arr).all():
                    raise ValueError(f"arm k={k} layer {layer}: non-finite {name}")

            diff = pool_res - ref_res  # > 0 means the pooled map is WORSE on identical targets
            lo, hi = paired_bootstrap(diff, N_BOOT, BOOT_SEED + layer)
            wil = wilcoxon(pool_res, ref_res)

            # Representativeness: the reference arm's own error on the shared subset vs the
            # whole mask. A large gap would mean the i-mod-K subset is not a typical slice.
            ref_res_all = ss_res[ref_arm_idx, layer, :]

            per_layer[f"L{layer}"] = {
                "behavior": behavior,
                "n_shared_contexts": int(positions.size),
                "ref_ss_res_sum": float(ref_res.sum()),
                "pooled_ss_res_sum": float(pool_res.sum()),
                "ss_res_ratio_pooled_over_ref": float(pool_res.sum() / ref_res.sum()),
                "mean_paired_diff": float(diff.mean()),
                "mean_paired_diff_ci95": [lo, hi],
                "median_paired_diff": float(np.median(diff)),
                "frac_contexts_pooled_worse": float((diff > 0).mean()),
                "wilcoxon_statistic": float(wil.statistic),
                "wilcoxon_p": float(wil.pvalue),
                # Same subset, same targets: own-denominator R2 differs from common-denominator
                # R2 only through ss_tot, which is the denominator effect made explicit.
                "r2_shared_subset_own_denominator": {
                    "ref": float(1.0 - ref_res.sum() / ref_tot.sum()),
                    "pooled": float(1.0 - pool_res.sum() / pool_tot.sum()),
                },
                "r2_shared_subset_common_denominator": {
                    "ref": float(1.0 - ref_res.sum() / ref_tot.sum()),
                    "pooled": float(1.0 - pool_res.sum() / ref_tot.sum()),
                },
                "ss_tot_ratio_pooled_over_ref": float(pool_tot.sum() / ref_tot.sum()),
                "representativeness": {
                    "ref_mean_ss_res_shared": float(ref_res.mean()),
                    "ref_mean_ss_res_all_mask": float(ref_res_all.mean()),
                    "ratio_shared_over_all": float(ref_res.mean() / ref_res_all.mean()),
                },
                "offset_bias_control": offset_bias_control(
                    implied[f"k{k}:L{layer}"]["between_persona_mean_shift_energy"],
                    k,
                    float(diff.mean()),
                ),
            }

        results[f"k{k}"] = {
            "n_shared_contexts_pre_mask": int(shared_ctx.size),
            "n_shared_contexts_post_mask": int(positions.size),
            "per_layer": per_layer,
        }

    payload = {
        "metadata": {
            "script": "scripts/issue823_shared_persona_paired.py",
            "task": 823,
            "followup_label": "inconsistent-origin-persona-ladder",
            "round": "user-chat inline free analysis (0 GPU-h, existing artifacts)",
            "git_commit": git_commit(root),
            "inputs": {
                "percontext_ladder_npz": str(LADDER_DIR / "percontext_ladder.npz"),
                "assignment_json": str(LADDER_DIR / "assignment.json"),
            },
            "registered_assignment_rule": rule,
            "held_out": (
                "p1_ss_res is out-of-fold: KFold(5, shuffle=True, random_state=0), written at "
                "test indices only (scripts/issue823_ladder_fits.py:1903). Fold split depends "
                "only on n, so both arms score each shared context in the same fold."
            ),
            "n_mask_contexts": int(context_ids.size),
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "read_out_layers": {str(k): v for k, v in READ_OUT_LAYERS.items()},
            "sign_convention": "diff = pooled_ss_res - reference_ss_res; > 0 = pooled worse",
        },
        "arms": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    for k in POOLED_ARMS:
        r = results[f"k{k}"]
        print(f"\n=== k={k} vs k=1  (n shared = {r['n_shared_contexts_post_mask']}) ===")
        for layer, behavior in READ_OUT_LAYERS.items():
            c = r["per_layer"][f"L{layer}"]
            lo, hi = c["mean_paired_diff_ci95"]
            print(
                f"  L{layer} ({behavior:13s}) "
                f"ss_res ratio {c['ss_res_ratio_pooled_over_ref']:.4f} | "
                f"mean diff {c['mean_paired_diff']:+.1f} [{lo:+.1f}, {hi:+.1f}] | "
                f"pooled worse in {c['frac_contexts_pooled_worse']:.1%} | "
                f"wilcoxon p={c['wilcoxon_p']:.3g} | "
                f"R2 common-denom ref {c['r2_shared_subset_common_denominator']['ref']:.3f} "
                f"vs pooled {c['r2_shared_subset_common_denominator']['pooled']:.3f} | "
                f"ss_tot ratio {c['ss_tot_ratio_pooled_over_ref']:.3f}"
            )
            o = c["offset_bias_control"]
            print(
                f"        offset-bias control: E={o['between_persona_mean_shift_energy']:.1f} "
                f"E/k={o['predicted_excess_if_shared_map_offset_only']:.1f} "
                f"measured={o['measured_excess']:.1f} "
                f"(x{o['ratio_measured_over_offset_only_prediction']:.1f} vs E/k, "
                f"{o['ratio_measured_over_full_energy']:.2f} of E) -> {o['verdict']}"
            )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
