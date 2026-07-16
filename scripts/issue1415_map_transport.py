"""Issue #1415 map-transport DV (round B deliverable 2). CPU, VM-safe.

STEP 0 (realized-keys verification, REQUIRED before any consumption): download
the #922 conditioned map bundle ``issue922_nexttoken/maps_conditioned/
direct_row_20.pt`` (~981 MB LFS blob; the internal structure was UNVERIFIED at
plan time), mmap-load it, PRINT every key + tensor shape to the log, and
assert a layer-20 V_c -> V_a weight of shape (H, H) resolves (any dtype). On
failure this script exits **rc=3** with a message naming the fallback (refit a
fresh ridge from the #779 activation tensors) — the refit is deliberately NOT
implemented this round.

Transport DV per pair (layer 20, per extraction arm)::

    f_predicted_delta = f(V_c + Delta) - f(V_c)      # f(x) = x @ W.T (+ b; b cancels)
    realized_delta    = V_a(steered)  - V_a(baseline)
    transport_cosine  = cos(realized_delta, f_predicted_delta)

Realized-delta sources:
- ``--realized-source steered`` (default; the plan-registered DV): V_a captured
  over the phase-1c STEERED completions, read from
  ``<steered-activations>/<pair_id>__<arm>.pt`` files carrying ``v_a_mean``
  ((L, H) + ``layers``, or a bare (H,) layer-20 vector). NOTE: the round-A
  phase-1 driver does not yet produce these captures — a follow-up GPU capture
  phase over the persisted 1c draws is required (concern raised on the task).
- ``--realized-source natural`` (descriptive companion, computable from the
  phase-1a captures alone): V_a(c') - V_a(c) — the NATURAL answer shift under
  the real context swap, labeled as such in the output.

Shuffled-pair null band with the same selection rule (500 permutations, seed
1416). Selection note: this DV is registered at the single primary layer 20,
so the max-over-layers selection of the null battery reduces to the identity —
the null draws receive the identical (single-layer) treatment as the observed
statistic. Cosine is scale-invariant, so the donor-Delta rescale to the
recipient norm cancels exactly; the permutation is applied to the PREDICTED
deltas. Output: ``eval_results/issue_1415/map_transport_cosines.json``.
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
logger = logging.getLogger("issue1415_map_transport")

MAP_REPO_PATH = "issue922_nexttoken/maps_conditioned/direct_row_20.pt"
RC_STEP0_FAIL = 3
SEED_SHUFFLED = 1416
FALLBACK_MSG = (
    "STEP0 FAIL: no unambiguous layer-20 (H, H) V_c->V_a weight resolved in the #922 "
    "map bundle. Fallback (NOT implemented this round): refit a fresh ridge map from "
    "the #779 activation tensors (plan v5 K3 fallback)."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--activations",
        type=Path,
        default=common.REPO_ROOT / "data" / "issue_1415" / "phase1" / "activations",
    )
    ap.add_argument(
        "--steered-activations",
        type=Path,
        default=common.REPO_ROOT / "data" / "issue_1415" / "phase1" / "activations_steered",
        help="dir of <pair_id>__<arm>.pt steered V_a captures (steered mode)",
    )
    ap.add_argument(
        "--realized-source",
        choices=("steered", "natural"),
        default="steered",
        help="steered = plan-registered DV; natural = V_a(c')-V_a(c) companion",
    )
    ap.add_argument(
        "--map-path",
        type=Path,
        default=None,
        help="local map bundle override (skips the hf_hub_download)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=common.REPO_ROOT / "eval_results" / "issue_1415" / "map_transport_cosines.json",
    )
    ap.add_argument("--layer", type=int, default=common.PRIMARY_LAYER)
    ap.add_argument("--hidden", type=int, default=common.HIDDEN_FULL)
    ap.add_argument("--n-draws", type=int, default=500)
    ap.add_argument("--seed-shuffled", type=int, default=SEED_SHUFFLED)
    return ap.parse_args(argv)


# ── STEP 0: realized-keys verification ────────────────────────────────


def _walk_tensors(obj, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Recursive (key_path, tensor) walk over dict/list/tuple/tensor bundles."""
    found: list[tuple[str, torch.Tensor]] = []
    if isinstance(obj, torch.Tensor):
        found.append((prefix or "<root>", obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(_walk_tensors(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.extend(_walk_tensors(v, f"{prefix}[{i}]"))
    return found


def resolve_map(
    bundle_path: Path, hidden: int, layer: int
) -> tuple[str, torch.Tensor, torch.Tensor | None]:
    """Load the bundle, print keys+shapes, resolve the (hidden, hidden) layer
    weight (+ optional sibling bias). Raises SystemExit(3) on failure."""
    try:
        bundle = torch.load(bundle_path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as e:  # trusted own artifact; unpickleable objects fall back
        logger.warning("weights_only=True load failed (%s); retrying weights_only=False", e)
        bundle = torch.load(bundle_path, map_location="cpu", mmap=True, weights_only=False)

    tensors = _walk_tensors(bundle)
    logger.info("STEP0: bundle %s — %d tensors:", bundle_path, len(tensors))
    for path, t in tensors:
        logger.info("  %s: shape=%s dtype=%s", path, tuple(t.shape), t.dtype)

    candidates = [(p, t) for p, t in tensors if tuple(t.shape) == (hidden, hidden)]
    if len(candidates) > 1:
        narrowed = [(p, t) for p, t in candidates if str(layer) in p]
        candidates = narrowed or candidates
    if len(candidates) > 1:
        narrowed = [(p, t) for p, t in candidates if "weight" in p.lower() or ".w" in p.lower()]
        candidates = narrowed or candidates
    if len(candidates) != 1:
        logger.error(
            "STEP0: %d candidate (%d, %d) tensors after narrowing: %s",
            len(candidates),
            hidden,
            hidden,
            [p for p, _ in candidates],
        )
        print(FALLBACK_MSG, file=sys.stderr)
        raise SystemExit(RC_STEP0_FAIL)

    wpath, weight = candidates[0]
    logger.info(
        "STEP0 PASS: resolved layer-%d weight at %r shape=%s dtype=%s",
        layer,
        wpath,
        tuple(weight.shape),
        weight.dtype,
    )

    # Optional bias: a sibling (hidden,) tensor sharing the weight's parent path.
    parent = wpath.rsplit(".", 1)[0] if "." in wpath else ""
    bias = None
    for p, t in tensors:
        sib_parent = p.rsplit(".", 1)[0] if "." in p else ""
        if (
            tuple(t.shape) == (hidden,)
            and sib_parent == parent
            and any(tok in p.lower() for tok in ("bias", "intercept", ".b"))
        ):
            bias = t
            logger.info("STEP0: resolved sibling bias at %r (cancels in the delta DV)", p)
            break
    return wpath, weight, bias


# ── transport computation ─────────────────────────────────────────────


def apply_map(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """f(x) = x @ W.T (+ b), computed in float64 for stability, returned fp32.

    Orientation note: the nn.Linear convention (out = x @ W.T) is assumed for
    the square weight; the convention used is recorded in the output JSON.
    """
    y = x.double() @ weight.double().T
    if bias is not None:
        y = y + bias.double()
    return y.float()


def _load_steered_va(path: Path, layer: int, layers_ref: list[int], hidden: int) -> torch.Tensor:
    assert path.exists(), (
        f"steered V_a capture missing: {path} — the phase-1c steered-completions capture "
        "phase has not produced this pair/arm (see the task concern); "
        "use --realized-source natural for the phase-1a-only companion read"
    )
    blob = torch.load(path, map_location="cpu", weights_only=True)
    v = blob["v_a_mean"] if isinstance(blob, dict) else blob
    if v.dim() == 2:
        layers = list(blob["layers"]) if isinstance(blob, dict) and "layers" in blob else layers_ref
        v = v[layers.index(layer)]
    assert v.shape == (hidden,), (path, v.shape)
    return v.float()


def compute_transport(
    pairs: list[common.PairTensors],
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    layer: int,
    realized_source: str,
    steered_dir: Path,
    n_draws: int,
    seed: int,
) -> dict:
    hidden = weight.shape[0]
    li = pairs[0].layers.index(layer)
    pair_ids = [p.pair_id for p in pairs]

    result: dict = {"per_arm": {}}
    for arm in common.ARMS:
        predicted = []
        realized = []
        for p in pairs:
            v_c = p.v_c[arm][li]
            delta = p.delta[arm][li]
            assert v_c.shape == (hidden,) and delta.shape == (hidden,), (p.pair_id, v_c.shape)
            fpred = apply_map(v_c + delta, weight, bias) - apply_map(v_c, weight, bias)
            predicted.append(fpred)
            if realized_source == "steered":
                r = (
                    _load_steered_va(
                        steered_dir / f"{p.pair_id}__{arm}.pt", layer, p.layers, hidden
                    )
                    - p.v_a_c[li]
                )
            else:
                r = p.v_a_cprime[li] - p.v_a_c[li]
            realized.append(r)
        pred = torch.stack(predicted)  # (P, H)
        real = torch.stack(realized)  # (P, H)
        pn, rn = pred.norm(dim=-1), real.norm(dim=-1)
        assert torch.all(pn > 0) and torch.all(rn > 0), f"{arm}: degenerate zero delta"
        pred_hat, real_hat = pred / pn.unsqueeze(-1), real / rn.unsqueeze(-1)

        observed = (pred_hat * real_hat).sum(dim=-1)  # (P,)

        # Shuffled-pair null: cos(realized_p, predicted_{perm(p)}). The donor
        # rescale to the recipient norm cancels in cosine (scale-invariant).
        cos_matrix = real_hat @ pred_hat.T  # (P_recipient, P_donor) — one matmul
        perms = common.batched_permutations(n_draws, len(pairs), seed)  # (D, P)
        null = cos_matrix[torch.arange(len(pairs)), perms]  # (D, P)

        result["per_arm"][arm] = {
            "transport_cosine": {pid: float(observed[pi]) for pi, pid in enumerate(pair_ids)},
            "predicted_delta_norm": {pid: float(pn[pi]) for pi, pid in enumerate(pair_ids)},
            "realized_delta_norm": {pid: float(rn[pi]) for pi, pid in enumerate(pair_ids)},
            "null_per_pair": {
                pid: common.quantile_band(null[:, pi]) for pi, pid in enumerate(pair_ids)
            },
            "null_pooled": common.quantile_band(null),
            "frac_null_geq_observed": {
                pid: float((null[:, pi] >= observed[pi]).float().mean())
                for pi, pid in enumerate(pair_ids)
            },
        }
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.map_path is not None:
        map_path = args.map_path
    else:
        from huggingface_hub import hf_hub_download

        logger.info("downloading %s (~981 MB; mmap load)", MAP_REPO_PATH)
        map_path = Path(hf_hub_download(common.HF_DATA_REPO, MAP_REPO_PATH, repo_type="dataset"))

    wpath, weight, bias = resolve_map(map_path, args.hidden, args.layer)

    pairs = common.load_all_pairs(args.activations)
    assert args.layer in pairs[0].layers, (args.layer, pairs[0].layers)

    result = compute_transport(
        pairs,
        weight,
        bias,
        args.layer,
        args.realized_source,
        args.steered_activations,
        args.n_draws,
        args.seed_shuffled,
    )
    out = {
        "layer": args.layer,
        "realized_source": args.realized_source,
        "realized_definition": (
            "V_a(steered completions) - V_a(baseline c completions)"
            if args.realized_source == "steered"
            else "V_a(c') - V_a(c) (natural context-swap answer shift; descriptive companion)"
        ),
        "selection_rule": "single registered layer -> identity selection (documented)",
        "n_draws": args.n_draws,
        "seed_shuffled": args.seed_shuffled,
        "map_provenance": {
            "repo_path": MAP_REPO_PATH if args.map_path is None else str(args.map_path),
            "resolved_weight_key": wpath,
            "weight_shape": list(weight.shape),
            "weight_dtype": str(weight.dtype),
            "bias_resolved": bias is not None,
            "orientation_convention": "f(x) = x @ W.T (+ b); nn.Linear convention assumed",
            "bundle_bytes": map_path.stat().st_size,
        },
        **result,
        "repro": common.repro_meta("issue1415_map_transport"),
    }
    common.write_json_atomic(args.out_json, out)
    logger.info("wrote %s (%d pairs, source=%s)", args.out_json, len(pairs), args.realized_source)


if __name__ == "__main__":
    main()
