"""Issue #1415 geometric null battery (round B deliverable 1). CPU, VM-safe.

Two batteries over the phase-1a tensors (V_c, V_a, Delta) ONLY — no
generations. **The generation-backed null (steering with a random Delta and
re-generating on the pod) is deliberately out of scope for this script: the
GEOMETRIC null computed here is the registered one** (round-B spec, resolving
the plan §4.6-vs-§9 self-contradiction to the §9/CPU reading). The per-layer
statistic is the COSINE of the candidate direction with the pair's normalized
answer-target direction::

    stat(pair, layer) = cos( d_candidate , V_a(c') - V_a(c)[layer] )

**Units contract (round-2 fix for the H1 null/observed scale-comparability
concern):** cosine is dimensionless and scale-free, so the plan §4.6
norm-matching (rescale to the pair's TRUE ||Delta||) cancels EXACTLY and the
band lives in the SAME units as the realized H1 DV — the
``projection_cosine`` of the realized steered V_a shift computed by
``scripts/issue1415_geometric_projections.py`` (which asserts this file's
``units == "cosine"`` before any band comparison). Under the previous
norm-matched form the null carried ||Delta|| (a V_c-space magnitude) while the
observed carried ||realized V_a shift|| — incommensurate a priori. The
constructed geometric ceiling is cos(target, target) = 1.0, matching the plan
§4.5 normalization-anchor note. Selection symmetry
(.claude/rules/selection-symmetric-nulls.md): each draw takes the MAX over the
7 sweep layers — one scalar per draw — and the observed statistic uses the
IDENTICAL rule.

- Random-Delta null: a 500-direction unit-sphere pool in R^H (seed 1415),
  precomputed ONCE and shared across pairs; one batched matmul per layer. The
  cosine statistic is arm-free for this battery (no ||Delta|| enters), so the
  arm axis is REPLICATED for shape compatibility (documented in the matrix).
- Shuffled-pair-Delta null: 500 permutations of the pair->Delta assignment
  (seed 1416, batched argsort — no per-draw loop), donor UNIT directions (the
  recipient-norm rescale cancels in cosine); one batched matmul per
  (arm, layer), draws realized as gathers from the pair x pair cosine matrix.

The battery's own ``observed`` rows are the INPUT-Delta cosines
cos(Delta, target) — a direction-alignment diagnostic of the steering input,
NOT the H1 observed statistic (that is the realized steered V_a shift cosine
from the projection driver, compared against these bands in the same units).

Per-draw x per-axis matrices are persisted BEFORE any aggregation to
``<matrices-dir>/{random_delta,shuffled_pair}_null_matrix.pt`` (HF data-repo
upload under ``analysis_tensors/issue_1415/null_matrices/``); the summary
bands go to ``eval_results/issue_1415/null_bands.json`` (git).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch — the #847 thread-cap hook binds at import time

import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1415_analysis_common as common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1415_null_battery")

N_DRAWS_DEFAULT = 500
SEED_RANDOM = 1415
SEED_SHUFFLED = 1416
HF_MATRIX_PREFIX = "analysis_tensors/issue_1415/null_matrices"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--activations",
        type=Path,
        default=common.REPO_ROOT / "data" / "issue_1415" / "phase1" / "activations",
        help="phase-1a capture dir (<pair_id>.pt files)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=common.REPO_ROOT / "eval_results" / "issue_1415" / "null_bands.json",
    )
    ap.add_argument(
        "--matrices-dir",
        type=Path,
        default=common.REPO_ROOT / "data" / "issue_1415" / "null_matrices",
        help="local staging dir for the per-draw matrices (HF-bound)",
    )
    ap.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    ap.add_argument("--seed-random", type=int, default=SEED_RANDOM)
    ap.add_argument("--seed-shuffled", type=int, default=SEED_SHUFFLED)
    ap.add_argument(
        "--upload",
        choices=("hf", "none"),
        default="hf",
        help="upload the persisted matrices to the HF data repo (default hf)",
    )
    return ap.parse_args(argv)


def _stack_targets(pairs: list[common.PairTensors]) -> torch.Tensor:
    """(n_layers, n_pairs, H) unit answer-target directions."""
    t = torch.stack([p.target_unit() for p in pairs], dim=1)  # (L, P, H)
    return t


def _stack_deltas(pairs: list[common.PairTensors]) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (deltas, norms): (n_arms, n_layers, n_pairs, H) and (n_arms, n_layers, n_pairs)."""
    d = torch.stack(
        [torch.stack([p.delta[arm] for p in pairs], dim=1) for arm in common.ARMS]
    )  # (A, L, P, H)
    n = d.norm(dim=-1)  # (A, L, P)
    assert torch.all(n > 0), "degenerate zero Delta"
    return d, n


def observed_stats(pairs: list[common.PairTensors]) -> torch.Tensor:
    """(n_layers, n_arms, n_pairs) INPUT-Delta cosines cos(Delta, target).

    A direction-alignment diagnostic of the steering input (same cosine units
    as the null draws and the realized H1 DV; see the module units contract).
    """
    targets = _stack_targets(pairs)  # (L, P, H)
    deltas, norms = _stack_deltas(pairs)  # (A, L, P, H), (A, L, P)
    dhat = deltas / norms.unsqueeze(-1)  # unit candidate directions
    obs = torch.einsum("alph,lph->lap", dhat, targets)
    assert obs.shape == (len(pairs[0].layers), len(common.ARMS), len(pairs)), obs.shape
    return obs


def random_delta_battery(pairs: list[common.PairTensors], n_draws: int, seed: int) -> torch.Tensor:
    """(n_draws, n_layers, n_arms, n_pairs) per-layer null COSINES.

    Direction pool precomputed ONCE (shared across pairs — each pair is
    evaluated against the same 500 unit directions); one batched matmul per
    layer. Cosine is scale-free, so the plan §4.6 norm-matching cancels and
    the arm axis is REPLICATED (the random-direction cosine does not depend
    on the arm's ||Delta||).
    """
    hidden = pairs[0].v_a_c.shape[-1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pool = torch.randn(n_draws, hidden, generator=gen)
    pool = pool / pool.norm(dim=-1, keepdim=True)  # (D, H) unit sphere

    targets = _stack_targets(pairs)  # (L, P, H)
    n_layers, n_pairs = targets.shape[0], targets.shape[1]

    per_layer = torch.empty(n_draws, n_layers, len(common.ARMS), n_pairs)
    for li in range(n_layers):  # one batched matmul per battery per layer
        g = pool @ targets[li].T  # (D, P) cosines (both sides unit)
        per_layer[:, li] = g.unsqueeze(1).expand(-1, len(common.ARMS), -1)  # (D, A, P)
    return per_layer


def shuffled_pair_battery(
    pairs: list[common.PairTensors], n_draws: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(per_layer COSINES (D, L, A, P), permutations (D, P)).

    Draw d assigns pair p the DONOR Delta DIRECTION of pair perm[d, p] (the
    recipient-norm rescale cancels in the cosine statistic). One batched
    matmul per (arm, layer) forms the donor-cosine matrix; the 500 draws are
    gathers from it.
    """
    targets = _stack_targets(pairs)  # (L, P, H)
    deltas, norms = _stack_deltas(pairs)  # (A, L, P, H), (A, L, P)
    n_layers, n_pairs = targets.shape[0], targets.shape[1]
    dhat = deltas / norms.unsqueeze(-1)  # unit donor directions

    perms = common.batched_permutations(n_draws, n_pairs, seed)  # (D, P)

    per_layer = torch.empty(n_draws, n_layers, len(common.ARMS), n_pairs)
    for li in range(n_layers):
        for ai in range(len(common.ARMS)):
            # G[donor, recipient] = cos(dhat_donor, target_recipient)
            g = dhat[ai, li] @ targets[li].T  # (P, P) — the one matmul
            per_layer[:, li, ai] = g[perms, torch.arange(n_pairs)]  # (D, P) gather
    return per_layer, perms


def select_max_over_layers(per_layer: torch.Tensor) -> torch.Tensor:
    """The registered selection rule: MAX over the layer axis (dim 1 for draws
    tensors (D, L, A, P) -> (D, A, P); dim 0 for observed (L, A, P) -> (A, P))."""
    dim = 1 if per_layer.dim() == 4 else 0
    return per_layer.max(dim=dim).values


def aggregate_bands(
    pairs: list[common.PairTensors],
    observed_sel: torch.Tensor,
    selected: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict:
    """Summary JSON: per-(arm, pair) bands + pooled bands + observed stats."""
    pair_ids = [p.pair_id for p in pairs]
    out: dict = {
        "statistic": (
            "cosine(candidate direction, V_a(c')-V_a(c)); max over layers per draw "
            "(selection-symmetric). Scale-free: the plan §4.6 norm-matching cancels, "
            "making the band commensurate with the realized H1 projection_cosine "
            "(geometric_projections.json); observed rows here are the INPUT-Delta "
            "cosine diagnostic, not the H1 observed"
        ),
        "units": "cosine",
        "layers": pairs[0].layers,
        "arms": list(common.ARMS),
        "pair_ids": pair_ids,
        "n_draws": args.n_draws,
        "seeds": {"random_delta": args.seed_random, "shuffled_pair": args.seed_shuffled},
        # INPUT-Delta cosine diagnostic (NOT the H1 observed — see `statistic`).
        "observed_input_delta_cos_max_over_layers": {
            arm: {pid: float(observed_sel[ai, pi]) for pi, pid in enumerate(pair_ids)}
            for ai, arm in enumerate(common.ARMS)
        },
        "bands": {},
        "repro": common.repro_meta("issue1415_null_battery"),
    }
    for battery, sel in selected.items():  # sel: (D, A, P)
        per_pair = {
            arm: {pid: common.quantile_band(sel[:, ai, pi]) for pi, pid in enumerate(pair_ids)}
            for ai, arm in enumerate(common.ARMS)
        }
        pooled = {arm: common.quantile_band(sel[:, ai]) for ai, arm in enumerate(common.ARMS)}
        exceed = {
            arm: {
                pid: float((sel[:, ai, pi] >= observed_sel[ai, pi]).float().mean())
                for pi, pid in enumerate(pair_ids)
            }
            for ai, arm in enumerate(common.ARMS)
        }
        out["bands"][battery] = {
            "per_pair": per_pair,
            "pooled_across_pairs": pooled,
            "frac_null_geq_observed": exceed,
        }
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pairs = common.load_all_pairs(args.activations)
    logger.info(
        "loaded %d pairs from %s (layers=%s)", len(pairs), args.activations, pairs[0].layers
    )

    obs = observed_stats(pairs)  # (L, A, P)
    obs_sel = select_max_over_layers(obs)  # (A, P)

    rand_pl = random_delta_battery(pairs, args.n_draws, args.seed_random)
    shuf_pl, perms = shuffled_pair_battery(pairs, args.n_draws, args.seed_shuffled)
    selected = {
        "random_delta": select_max_over_layers(rand_pl),
        "shuffled_pair": select_max_over_layers(shuf_pl),
    }

    # ── persist the per-draw matrices BEFORE any aggregation ─────────────
    meta = {
        "pair_ids": [p.pair_id for p in pairs],
        "layers": pairs[0].layers,
        "arms": list(common.ARMS),
        "units": "cosine",
        "axes_per_layer": "(draw, layer, arm, pair)",
        "axes_selected": "(draw, arm, pair)",
        "observed_per_layer": obs,
        "observed_selected": obs_sel,
        "repro": common.repro_meta("issue1415_null_battery"),
    }
    matrix_paths: dict[str, Path] = {}
    for battery, per_layer, extra in (
        ("random_delta", rand_pl, {"seed": args.seed_random}),
        ("shuffled_pair", shuf_pl, {"seed": args.seed_shuffled, "permutations": perms}),
    ):
        path = args.matrices_dir / f"{battery}_null_matrix.pt"
        common.save_pt_atomic(
            path,
            {
                **meta,
                **extra,
                "battery": battery,
                "per_layer": per_layer,
                "selected": selected[battery],
            },
        )
        matrix_paths[battery] = path
        logger.info("[matrices] persisted %s (%s)", path, tuple(per_layer.shape))

    # ── aggregation (only after the matrices are on disk) ────────────────
    summary = aggregate_bands(pairs, obs_sel, selected, args)
    summary["matrix_files"] = {b: f"{HF_MATRIX_PREFIX}/{p.name}" for b, p in matrix_paths.items()}
    common.write_json_atomic(args.out_json, summary)
    logger.info("[summary] wrote %s", args.out_json)

    if args.upload == "hf":
        import issue1415_run_phase1 as drv

        for path in matrix_paths.values():
            drv._hf_upload(path, f"{HF_MATRIX_PREFIX}/{path.name}")
    logger.info("done: %d pairs x %d draws", len(pairs), args.n_draws)


if __name__ == "__main__":
    main()
