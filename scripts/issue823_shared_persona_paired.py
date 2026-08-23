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

CLI: ``--ladder-dir`` / ``--arms`` / ``--n-boot`` parametrize the read; the
defaults preserve the banked invocation exactly (identical inputs, seeds, and
output fields). ``--full-ratio-ci`` additionally computes, per arm and read-out
layer, a stratified-by-persona FULL-RATIO bootstrap for ``rho = mean_excess / E``
in which E is recomputed inside every context resample from per-persona
difference vectors (sidecar npz schema: ``load_mixture_diffs``; default path
``<ladder-dir>/mixture_diffs.npz``, override ``--mixture-diffs``), persisting
``rho_ci95`` / ``n_negligible_E_draws`` / ``rho_ci95_unstable`` alongside the
existing fields. The banked numerator-only CI (``mean_paired_diff_ci95``) stays
persisted as the labeled SECONDARY band.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps BEFORE the numpy/scipy imports (sibling pattern, #847)

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

# Repo root on sys.path so `scripts.*` sibling imports resolve in script mode
# (sys.path[0] is scripts/ when run as `python scripts/issue823_shared_persona_paired.py`).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.issue823_ladder_common import mixture_energy_from_group_diffs  # noqa: E402

# Read-out layers, per the parent round: evil 14, sycophancy 26, hallucination 17.
READ_OUT_LAYERS: dict[int, str] = {14: "evil", 26: "sycophancy", 17: "hallucination"}
POOLED_ARMS: tuple[int, ...] = (2, 4, 8, 16)
REFERENCE_ARM = 1
N_BOOT = 10_000
BOOT_SEED = 823

# Full-ratio bootstrap guard constants (plan section 4.2): a draw whose recomputed
# denominator falls below NEGLIGIBLE_E_REL x the point E is excluded + counted; more
# than UNSTABLE_FRAC of draws excluded flags the (arm x layer) CI as unstable.
NEGLIGIBLE_E_REL = 1e-9
UNSTABLE_FRAC = 0.01

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


def full_ratio_bootstrap(
    diff: np.ndarray,
    groups: list[tuple[int, np.ndarray]],
    n_persona0: int,
    n_boot: int,
    seed: int,
) -> dict:
    """Stratified-by-persona full-ratio bootstrap for ``rho = mean_excess / E``.

    ``diff`` is the persona-0 paired ss_res difference vector (pooled - reference on
    the shared contexts); ``groups`` is an ORDERED list of ``(persona, D_p)`` with
    ``D_p`` the (n_p, d) float64 per-context difference matrix for persona p != 0;
    ``n_persona0`` is the persona-0 group count entering the E normalization.

    One rng stream (seed = BOOT_SEED + layer, the script's own convention): the FIRST
    consumption is the persona-0 paired-diff index draw -- identical to
    ``paired_bootstrap``'s, so this strictly contains the banked numerator bootstrap --
    followed by one multinomial count matrix per group, in the order passed. Per draw
    (group sizes preserved):

        mean_excess_draw = mean of the resampled persona-0 paired diffs
        E_draw = sum_p n_p * || counts_p @ D_p / n_p ||^2 / n_tot
        rho_draw = mean_excess_draw / E_draw

    Draws with ``E_draw < NEGLIGIBLE_E_REL * E_point`` are EXCLUDED and counted
    (``n_negligible_E_draws``); strictly more than ``UNSTABLE_FRAC`` of draws excluded
    sets ``rho_ci95_unstable`` (never a crash; all draws excluded -> ``rho_ci95`` None).
    Vectorized: multinomial-count x difference-matrix GEMMs, no per-draw Python loop.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    mean_excess_draws = diff[idx].mean(axis=1)

    n_tot = int(n_persona0) + sum(int(dmat.shape[0]) for _p, dmat in groups)
    e_point = mixture_energy_from_group_diffs(
        ((int(dmat.shape[0]), dmat) for _p, dmat in groups), int(n_persona0)
    )

    between_draws = np.zeros(n_boot, dtype=np.float64)
    for _p, dmat in groups:
        n_p = int(dmat.shape[0])
        counts = rng.multinomial(n_p, np.full(n_p, 1.0 / n_p), size=n_boot)
        means = (counts.astype(np.float64) @ dmat) / n_p
        between_draws += n_p * np.einsum("bd,bd->b", means, means)
    e_draws = between_draws / max(n_tot, 1)

    if e_point > 0.0:
        negligible = e_draws < NEGLIGIBLE_E_REL * e_point
    else:
        # Degenerate cell: no between-persona energy at all -> every draw is excluded.
        negligible = np.ones(n_boot, dtype=bool)
    n_neg = int(negligible.sum())
    retained = ~negligible
    unstable = bool(n_neg > UNSTABLE_FRAC * n_boot)
    mean_excess_point = float(diff.mean())
    if retained.any():
        rho_draws = mean_excess_draws[retained] / e_draws[retained]
        rho_ci95 = [float(np.quantile(rho_draws, 0.025)), float(np.quantile(rho_draws, 0.975))]
    else:
        rho_ci95 = None
    return {
        "rho_ci95": rho_ci95,
        "n_negligible_E_draws": n_neg,
        "rho_ci95_unstable": unstable,
        "full_ratio": {
            "rho_point": (mean_excess_point / e_point) if e_point > 0.0 else None,
            "mean_excess_point": mean_excess_point,
            "e_point_from_diffs": float(e_point),
            "n_persona0": int(n_persona0),
            "n_boot": int(n_boot),
            "n_draws_retained": int(n_boot - n_neg),
            "seed": int(seed),
            "note": (
                "primary ratio CI; mean_paired_diff_ci95 remains the numerator-only "
                "SECONDARY band (E fixed at its point value)"
            ),
        },
    }


class MixtureDiffs:
    """Per-arm difference-vector groups loaded from the sidecar npz.

    Schema + validation: ``load_mixture_diffs``. ``groups(k, layer)`` returns the
    ``(persona, D_p float64)`` list in ASCENDING persona order (the rng-consumption
    order ``full_ratio_bootstrap`` documents); ``n_persona0(k, default)`` returns the
    sidecar's persona-0 count when present, else the caller's default (the paired
    script's shared-context count for that arm).
    """

    def __init__(self, layers: np.ndarray, per_arm: dict[int, tuple]) -> None:
        self._layer_idx = {int(v): i for i, v in enumerate(layers)}
        self._per_arm = per_arm  # k -> (personas (n,), diffs (n, n_layers, d), n_persona0|None)

    def groups(self, k: int, layer: int) -> list[tuple[int, np.ndarray]]:
        personas, diffs, _ = self._per_arm[k]
        li = self._layer_idx[int(layer)]
        out: list[tuple[int, np.ndarray]] = []
        for p in sorted({int(x) for x in personas}):
            out.append((int(p), diffs[personas == p, li, :].astype(np.float64)))
        return out

    def n_persona0(self, k: int, default: int) -> int:
        _, _, n0 = self._per_arm[k]
        return int(n0) if n0 is not None else int(default)


def load_mixture_diffs(
    path: pathlib.Path, arms: tuple[int, ...], layers_needed: tuple[int, ...]
) -> MixtureDiffs:
    """Load + validate the per-arm difference-vector sidecar npz for ``--full-ratio-ci``.

    Schema (written by the producing fits driver, which owns the group construction --
    the paired script deliberately does NOT re-derive membership from the mask, so the
    companion-ladder usage with training-subset denominator groups stays valid):

      layers           (n_layers,) int -- hidden layers covered (must include every
                       requested read-out layer)
      k{k}_diffs       (n_k, n_layers, d) float -- per-context difference vectors
                       ``v_p(i) - v_0(i)`` for contexts assigned persona p != 0 under
                       pooled arm k
      k{k}_personas    (n_k,) int -- persona id per row (all != 0)
      k{k}_n_persona0  scalar int, OPTIONAL -- persona-0 group count for the E
                       normalization (absent -> the paired script's shared-context
                       count for that arm)
      k{k}_context_ids (n_k,) int, OPTIONAL -- provenance only (not consumed)
    """
    if not path.exists():
        raise FileNotFoundError(f"--full-ratio-ci requires the mixture-diffs npz: {path}")
    z = np.load(path)
    layers = [int(v) for v in z["layers"]]
    missing = [layer for layer in layers_needed if layer not in layers]
    if missing:
        raise ValueError(f"mixture-diffs npz {path} lacks read-out layers {missing}")
    per_arm: dict[int, tuple] = {}
    for k in arms:
        dkey, pkey = f"k{k}_diffs", f"k{k}_personas"
        if dkey not in z or pkey not in z:
            raise ValueError(f"mixture-diffs npz {path} lacks arrays for arm k={k}")
        diffs = z[dkey]
        personas = z[pkey].astype(int)
        if diffs.ndim != 3 or diffs.shape[0] != personas.shape[0] or diffs.shape[1] != len(layers):
            raise ValueError(
                f"arm k={k}: diffs shape {diffs.shape} inconsistent with "
                f"{personas.shape[0]} rows x {len(layers)} layers"
            )
        if (personas == 0).any():
            raise ValueError(f"arm k={k}: mixture-diffs rows must have persona != 0")
        if not np.isfinite(diffs).all():
            raise ValueError(f"arm k={k}: non-finite mixture diffs")
        n0key = f"k{k}_n_persona0"
        n0 = int(z[n0key]) if n0key in z else None
        per_arm[k] = (personas, diffs, n0)
    return MixtureDiffs(np.asarray(layers), per_arm)


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
    ap.add_argument(
        "--ladder-dir",
        default=None,
        help="ladder artifact directory (default: the canonical banked dir under the repo root)",
    )
    ap.add_argument(
        "--arms",
        default=None,
        help="comma-separated pooled arm k values (default: 2,4,8,16)",
    )
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstrap draws (default: 10000)")
    ap.add_argument(
        "--full-ratio-ci",
        action="store_true",
        help=(
            "also compute the stratified-by-persona full-ratio bootstrap "
            "(rho = mean_excess / E with E recomputed per draw); requires the "
            "per-arm difference-vector npz (see --mixture-diffs)"
        ),
    )
    ap.add_argument(
        "--mixture-diffs",
        default=None,
        help=(
            "npz with per-arm difference vectors for --full-ratio-ci "
            "(default: <ladder-dir>/mixture_diffs.npz)"
        ),
    )
    args = ap.parse_args()

    root = repo_root()
    # ladder_rel feeds the metadata "inputs" strings; the default preserves the banked
    # LADDER_DIR-relative form byte-for-byte.
    ladder_rel = pathlib.Path(args.ladder_dir) if args.ladder_dir else LADDER_DIR
    ladder = pathlib.Path(args.ladder_dir) if args.ladder_dir else root / LADDER_DIR
    pooled_arms = tuple(int(t) for t in args.arms.split(",")) if args.arms else POOLED_ARMS
    n_boot = int(args.n_boot)
    npz_path = ladder / "percontext_ladder.npz"
    assign_path = ladder / "assignment.json"
    out_path = pathlib.Path(args.out) if args.out else ladder / "shared_persona_paired.json"

    mixture = None
    diffs_path = None
    if args.full_ratio_ci:
        diffs_path = (
            pathlib.Path(args.mixture_diffs) if args.mixture_diffs else ladder / "mixture_diffs.npz"
        )
        mixture = load_mixture_diffs(diffs_path, pooled_arms, tuple(READ_OUT_LAYERS))

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
    for k in pooled_arms:
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
            lo, hi = paired_bootstrap(diff, n_boot, BOOT_SEED + layer)
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
            if mixture is not None:
                per_layer[f"L{layer}"].update(
                    full_ratio_bootstrap(
                        diff,
                        mixture.groups(k, layer),
                        mixture.n_persona0(k, default=int(positions.size)),
                        n_boot,
                        BOOT_SEED + layer,
                    )
                )

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
                "percontext_ladder_npz": str(ladder_rel / "percontext_ladder.npz"),
                "assignment_json": str(ladder_rel / "assignment.json"),
            },
            "registered_assignment_rule": rule,
            "held_out": (
                "p1_ss_res is out-of-fold: KFold(5, shuffle=True, random_state=0), written at "
                "test indices only (scripts/issue823_ladder_fits.py:1903). Fold split depends "
                "only on n, so both arms score each shared context in the same fold."
            ),
            "n_mask_contexts": int(context_ids.size),
            "n_boot": n_boot,
            "boot_seed": BOOT_SEED,
            "read_out_layers": {str(k): v for k, v in READ_OUT_LAYERS.items()},
            "sign_convention": "diff = pooled_ss_res - reference_ss_res; > 0 = pooled worse",
        },
        "arms": results,
    }
    if mixture is not None:
        payload["metadata"]["full_ratio_ci"] = {
            "mixture_diffs": str(diffs_path),
            "negligible_e_rel_threshold": NEGLIGIBLE_E_REL,
            "unstable_excluded_frac_threshold": UNSTABLE_FRAC,
            "resample": (
                "stratified within persona groups (sizes preserved); the persona-0 "
                "resample is the paired-diff index draw itself (same rng stream, "
                "seed = boot_seed + layer), then one multinomial per persona group "
                "in ascending-persona order"
            ),
            "secondary_ci": (
                "mean_paired_diff_ci95 stays the numerator-only SECONDARY band (E fixed)"
            ),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    for k in pooled_arms:
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
            if c.get("rho_ci95") is not None:
                frlo, frhi = c["rho_ci95"]
                unstable = " UNSTABLE" if c["rho_ci95_unstable"] else ""
                print(
                    f"        full-ratio rho CI95 [{frlo:+.3f}, {frhi:+.3f}] "
                    f"(excluded {c['n_negligible_E_draws']} negligible-E draws){unstable}"
                )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
