"""Issue #1415 geometric null battery (round B deliverable 1). CPU, VM-safe.

Two batteries over the phase-1a tensors (V_c, V_a, Delta) ONLY — no
generations. **The generation-backed null (steering with a random Delta and
re-generating on the pod) is deliberately out of scope for this script: the
GEOMETRIC null computed here is the registered one** (round-B spec). The
per-layer statistic is the projection of a (norm-matched) candidate Delta onto
the pair's normalized answer-target direction::

    stat(pair, layer) = < d_candidate , normalize(V_a(c') - V_a(c))[layer] >

with ``d_candidate`` scaled to the pair's TRUE ||Delta(pair, arm, layer)||, so
the null carries the observed steering magnitude. Selection symmetry
(.claude/rules/selection-symmetric-nulls.md): each draw takes the MAX over the
7 sweep layers — one scalar per draw — and the observed statistic uses the
IDENTICAL rule.

- Random-Delta null: a 500-direction unit-sphere pool in R^H (seed 1415),
  precomputed ONCE and shared across pairs; one batched matmul per layer.
- Shuffled-pair-Delta null: 500 permutations of the pair->Delta assignment
  (seed 1416, batched argsort — no per-draw loop), each donor Delta rescaled
  to the recipient pair's true norm; one batched matmul per (arm, layer),
  draws realized as gathers from the pair x pair projection matrix.

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
    """(n_layers, n_arms, n_pairs) observed projections <Delta, target_hat>."""
    targets = _stack_targets(pairs)  # (L, P, H)
    deltas, _ = _stack_deltas(pairs)  # (A, L, P, H)
    obs = torch.einsum("alph,lph->lap", deltas, targets)
    assert obs.shape == (len(pairs[0].layers), len(common.ARMS), len(pairs)), obs.shape
    return obs


def random_delta_battery(pairs: list[common.PairTensors], n_draws: int, seed: int) -> torch.Tensor:
    """(n_draws, n_layers, n_arms, n_pairs) per-layer null stats.

    Direction pool precomputed ONCE (shared across pairs — each pair is
    evaluated against the same 500 unit directions); one batched matmul per
    layer; per-(pair, arm, layer) rescale to the true ||Delta||.
    """
    hidden = pairs[0].v_a_c.shape[-1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pool = torch.randn(n_draws, hidden, generator=gen)
    pool = pool / pool.norm(dim=-1, keepdim=True)  # (D, H) unit sphere

    targets = _stack_targets(pairs)  # (L, P, H)
    _, norms = _stack_deltas(pairs)  # (A, L, P)
    n_layers, n_pairs = targets.shape[0], targets.shape[1]

    per_layer = torch.empty(n_draws, n_layers, len(common.ARMS), n_pairs)
    for li in range(n_layers):  # one batched matmul per battery per layer
        g = pool @ targets[li].T  # (D, P)
        per_layer[:, li] = g.unsqueeze(1) * norms[:, li].unsqueeze(0)  # (D, A, P)
    return per_layer


def shuffled_pair_battery(
    pairs: list[common.PairTensors], n_draws: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(per_layer stats (D, L, A, P), permutations (D, P)).

    Draw d assigns pair p the DONOR Delta of pair perm[d, p], rescaled to the
    recipient p's true norm. One batched matmul per (arm, layer) forms the
    donor-projection matrix; the 500 draws are gathers from it.
    """
    targets = _stack_targets(pairs)  # (L, P, H)
    deltas, norms = _stack_deltas(pairs)  # (A, L, P, H), (A, L, P)
    n_layers, n_pairs = targets.shape[0], targets.shape[1]
    dhat = deltas / norms.unsqueeze(-1)  # unit donor directions

    perms = common.batched_permutations(n_draws, n_pairs, seed)  # (D, P)

    per_layer = torch.empty(n_draws, n_layers, len(common.ARMS), n_pairs)
    for li in range(n_layers):
        for ai in range(len(common.ARMS)):
            # G[donor, recipient] = <dhat_donor, target_recipient>
            g = dhat[ai, li] @ targets[li].T  # (P, P) — the one matmul
            drawn = g[perms, torch.arange(n_pairs)]  # (D, P) gather
            per_layer[:, li, ai] = drawn * norms[ai, li].unsqueeze(0)
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
            "projection of the norm-matched candidate Delta onto "
            "normalize(V_a(c')-V_a(c)); max over layers per draw (selection-symmetric)"
        ),
        "layers": pairs[0].layers,
        "arms": list(common.ARMS),
        "pair_ids": pair_ids,
        "n_draws": args.n_draws,
        "seeds": {"random_delta": args.seed_random, "shuffled_pair": args.seed_shuffled},
        "observed_max_over_layers": {
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
