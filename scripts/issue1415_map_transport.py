"""Issue #1415 map-transport DV (round B deliverable 2). CPU, VM-safe.

STEP 0 (realized-keys verification, REQUIRED before any consumption): download
the #922 conditioned map bundle ``issue922_nexttoken/maps_conditioned/
direct_row_20.pt`` (~981 MB LFS blob; the internal structure was UNVERIFIED at
plan time), mmap-load it, PRINT every key + tensor shape to the log, and
assert a layer-20 V_c -> V_a weight of shape (H, H) resolves (any dtype). On
failure the plan-K3 fallback fires (``--refit-fallback auto``, the default):
refit a fresh ridge V_c -> V_a map at the registered layer from the
#779-lineage activation tensors #823 published (``issue823_own_vs_external/
analysis_tensors/{v_c,v_a_prime}.pt`` — bare (5000, 28, 3584) fp32 tensors,
realized-shape-verified 2026-07-16; the #779 pass_a bundles carry ONLY context
vectors, no V_a, so the #823 pair tensors are the #779-lineage V_c/V_a source).
The refit runs a held-out sanity check (R^2 >= 0.2 at layer 20, vs #823's
~0.51 anchor — plan v5 K3); on sanity failure the script exits **rc=6** with a
message stating the H2 map-transport DV is dropped per K3. With
``--refit-fallback off`` a step-0 failure keeps the historical **rc=3** exit.

Transport DV per pair (layer 20, per extraction arm)::

    f_predicted_delta = f(V_c + Delta) - f(V_c)      # f(x) = x @ W.T (+ b; b cancels)
    realized_delta    = V_a(steered)  - V_a(baseline)
    transport_cosine  = cos(realized_delta, f_predicted_delta)

Realized-delta sources:
- ``--realized-source steered`` (default; the plan-registered DV): V_a captured
  over the phase-1c STEERED completions, read from
  ``<steered-activations>/<pair_id>__<arm>.pt`` files carrying ``v_a_mean``
  ((L, H) + ``layers``, or a bare (H,) layer-20 vector) — the CANONICAL
  operating-alpha primary-layer captures the phase-1 driver's 1e phase writes
  (round-2 fix; ``issue1415_run_phase1.phase_1e``). A pair whose canonical
  capture is missing (coherence failed at all alphas — the 1e canonical index
  records the skip) is EXCLUDED with a recorded reason (plan §8), fail-loud
  when NO pair resolves.
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
import json
import logging
import re
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
RC_REFIT_SANITY_FAIL = 6
SEED_SHUFFLED = 1416
FALLBACK_MSG = (
    "STEP0 FAIL: no unambiguous layer-20 (H, H) V_c->V_a weight resolved in the #922 "
    "map bundle. Fallback: refit a fresh ridge map from the #779-lineage (#823) "
    "activation tensors — rerun with --refit-fallback auto (plan v5 K3 fallback)."
)

# K3 refit fallback inputs: the #779-lineage (V_c, V_a) pair tensors #823 published
# (bare (5000, 28, 3584) fp32 tensors, row-aligned by context; realized shapes
# mmap-verified 2026-07-16). The #779 pass_a bundles hold ONLY context vectors
# (cx_last/cx_mean — no V_a), so these are the #779-lineage V_c->V_a fit source;
# #823's layer-20 ridge R^2 ~0.51 on exactly these tensors is the K3 sanity anchor.
REFIT_VC_REPO_PATH = "issue823_own_vs_external/analysis_tensors/v_c.pt"
REFIT_VA_REPO_PATH = "issue823_own_vs_external/analysis_tensors/v_a_prime.pt"
REFIT_VALID_IDX_REPO_PATH = "issue823_own_vs_external/raw_completions/phase1/common_valid_idx.json"
REFIT_R2_MIN = 0.2  # plan v5 K3: refit fails sanity if held-out R^2 < 0.2 at layer 20
REFIT_SEED = 1415
# Lambda choice: GCV over logspace(-2, 4, 13) — the #779/#823 fit_h.ridge_fit_predict
# recipe that produced the K3 sanity anchor (#823 R^2 ~0.51 at layer 20); GCV is
# closed-form (no extra CV folds) and shares ONE SVD across all 13 candidates.
REFIT_LAMBDAS = tuple(float(x) for x in torch.logspace(-2, 4, 13))


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
    ap.add_argument(
        "--refit-fallback",
        choices=("off", "auto", "force"),
        default="auto",
        help="plan-K3 ridge-refit fallback: auto = fire on a step-0 keys miss (rc=3 "
        "condition); force = skip the #922 bundle entirely; off = keep the rc=3 exit",
    )
    ap.add_argument(
        "--refit-vc-path",
        type=Path,
        default=None,
        help="local V_c tensor override for the refit (skips the hf_hub_download)",
    )
    ap.add_argument("--refit-va-path", type=Path, default=None)
    ap.add_argument(
        "--refit-valid-idx-path",
        type=Path,
        default=None,
        help="local common_valid_idx.json override (row filter for the refit tensors)",
    )
    ap.add_argument("--refit-holdout-frac", type=float, default=0.2)
    ap.add_argument("--refit-r2-min", type=float, default=REFIT_R2_MIN)
    ap.add_argument("--refit-seed", type=int, default=REFIT_SEED)
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
        # Token match, not substring: "20" must not match "layer_120"/"200"
        # (round-2 Minor fix; ambiguity still lands on the fail-loud rc=3 branch).
        layer_tok = re.compile(rf"(?<![0-9]){layer}(?![0-9])")
        narrowed = [(p, t) for p, t in candidates if layer_tok.search(p)]
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


# ── K3 fallback: ridge refit from the #779-lineage (#823) tensors ─────


def _load_layer_matrix(path: Path, layer: int, hidden: int) -> torch.Tensor:
    """mmap-load a bare (N, L, H) activation tensor (the #823 v_c/v_a_prime
    format — layer axis indexed by layer id) and slice one layer -> (N, H)
    fp32. A pre-sliced (N, H) tensor passes through. Fail-loud on any other
    shape (the #1073 realized-keys discipline: assert on the artifact's OWN
    shape, never the builder code)."""
    t = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert isinstance(t, torch.Tensor), f"{path}: expected a bare tensor, got {type(t).__name__}"
    if t.dim() == 2:
        assert t.shape[-1] == hidden, (path, tuple(t.shape), hidden)
        return t.float()
    assert t.dim() == 3 and t.shape[-1] == hidden and layer < t.shape[1], (
        path,
        tuple(t.shape),
        layer,
    )
    return t[:, layer, :].float()


def _ridge_fit_weights(
    Xtr: torch.Tensor, Ytr: torch.Tensor, lambdas=REFIT_LAMBDAS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Closed-form multi-output ridge with GCV lambda selection — the
    fit_h.ridge_fit_predict recipe (standardize X on train stats, center Y),
    returning weights instead of predictions: (W_std (d, D_out), xmu, xsd,
    ymu, best_lambda), all float64. ONE SVD is shared across all lambdas and
    output dims (batched — no per-dim/per-lambda refactorization), with the
    GCV RSS evaluated in eigen-coefficient space (the fit_h fast-twin
    identity), so the whole selection is a single factorization."""
    Xtr = Xtr.double()
    Ytr = Ytr.double()
    n = Xtr.shape[0]
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0, unbiased=False) + 1e-9  # population std, matching numpy .std in fit_h
    Xn = (Xtr - xmu) / xsd
    ymu = Ytr.mean(0)
    Yc = Ytr - ymu
    U, s, Vh = torch.linalg.svd(Xn, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Yc  # (r, D_out)
    sq_uty = (UtY**2).sum(1)  # per-eigencomponent target energy
    tot = float((Yc**2).sum())
    best_lam, best_gcv = float(lambdas[0]), float("inf")
    for lam in lambdas:
        filt = s2 / (s2 + lam)
        # RSS(lam) = ||Yc||^2 - sum_k (2 f_k - f_k^2) ||UtY_k||^2 (hat-matrix identity)
        rss = tot - float(((2 * filt - filt**2) * sq_uty).sum())
        dof = float(filt.sum())
        denom = (n - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    filt = s / (s2 + best_lam)
    w_std = (Vh.T * filt) @ UtY  # (d, D_out) standardized-space dual-ridge weights
    return w_std, xmu, xsd, ymu, best_lam


def refit_ridge_map(args) -> tuple[str, torch.Tensor, torch.Tensor, dict]:
    """Plan-K3 fallback: fit a fresh ridge V_c -> V_a map at ``args.layer``
    from the #823-published (#779-lineage) tensors, run the held-out sanity
    check (R^2 >= args.refit_r2_min; plan K3: failure drops the H2 DV,
    SystemExit(rc=6)), and return (desc, weight, bias, refit_meta) in the
    same nn.Linear orientation ``resolve_map`` returns (f(x) = x @ W.T + b;
    the bias cancels in the delta DV)."""
    if args.refit_vc_path is not None:
        assert args.refit_va_path is not None, "--refit-va-path required with --refit-vc-path"
        vc_path, va_path = args.refit_vc_path, args.refit_va_path
        valid_idx_path = args.refit_valid_idx_path
        vc_src, va_src = str(vc_path), str(va_path)
    else:
        from huggingface_hub import hf_hub_download

        logger.info("refit fallback: downloading #823 pair tensors (~2 GB each; mmap load)")
        vc_path = Path(
            hf_hub_download(common.HF_DATA_REPO, REFIT_VC_REPO_PATH, repo_type="dataset")
        )
        va_path = Path(
            hf_hub_download(common.HF_DATA_REPO, REFIT_VA_REPO_PATH, repo_type="dataset")
        )
        valid_idx_path = Path(
            hf_hub_download(common.HF_DATA_REPO, REFIT_VALID_IDX_REPO_PATH, repo_type="dataset")
        )
        vc_src, va_src = REFIT_VC_REPO_PATH, REFIT_VA_REPO_PATH

    x = _load_layer_matrix(vc_path, args.layer, args.hidden)
    y = _load_layer_matrix(va_path, args.layer, args.hidden)
    assert x.shape == y.shape and x.dim() == 2, (x.shape, y.shape)
    n_total = x.shape[0]
    if valid_idx_path is not None:
        with open(valid_idx_path) as f:
            valid_idx = torch.tensor(sorted(json.load(f)["common_valid_idx"]), dtype=torch.long)
        assert int(valid_idx.max()) < n_total, (int(valid_idx.max()), n_total)
        x, y = x[valid_idx], y[valid_idx]
        logger.info("refit: valid-idx filter kept %d/%d rows", x.shape[0], n_total)
    # Judge-invalid contexts can leave zero vectors — drop them WITH a record.
    keep = (x.norm(dim=1) > 0) & (y.norm(dim=1) > 0)
    n_zero_dropped = int((~keep).sum())
    if n_zero_dropped:
        logger.warning("refit: dropped %d zero-norm rows", n_zero_dropped)
        x, y = x[keep], y[keep]
    n = x.shape[0]
    assert n >= 10, f"refit: only {n} usable rows"

    gen = torch.Generator(device="cpu").manual_seed(args.refit_seed)
    perm = torch.randperm(n, generator=gen)
    n_val = max(1, round(n * args.refit_holdout_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    w_std, xmu, xsd, ymu, best_lam = _ridge_fit_weights(x[tr_idx], y[tr_idx])

    # Held-out multivariate R^2 (variance-weighted, val-mean ss_tot — the #823
    # phase-4 convention the ~0.51 anchor was computed under).
    y_val = y[val_idx].double()
    y_hat = ((x[val_idx].double() - xmu) / xsd) @ w_std + ymu
    ss_res = float(((y_val - y_hat) ** 2).sum())
    ss_tot = float(((y_val - y_val.mean(0)) ** 2).sum())
    r2_val = 1.0 - ss_res / (ss_tot + 1e-12)
    logger.info(
        "refit sanity: held-out R^2 = %.4f at layer %d (threshold %.2f; #823 anchor ~0.51; "
        "lambda = %.4g; n_train = %d, n_val = %d)",
        r2_val,
        args.layer,
        args.refit_r2_min,
        best_lam,
        len(tr_idx),
        n_val,
    )
    if r2_val < args.refit_r2_min:
        print(
            f"REFIT SANITY FAIL: held-out R^2 {r2_val:.4f} < {args.refit_r2_min} at layer "
            f"{args.layer} — the H2 map-transport DV is DROPPED per plan v5 K3 "
            "(report H1/H3 only, H2 as not-run).",
            file=sys.stderr,
        )
        raise SystemExit(RC_REFIT_SANITY_FAIL)

    # Effective RAW-space map: f(x) = ((x - xmu)/xsd) @ W_std + ymu
    #                               = x @ (W_std / xsd[:, None]) + (ymu - (xmu/xsd) @ W_std),
    # returned in the nn.Linear (out, in) convention apply_map expects.
    weight = (w_std / xsd[:, None]).T.contiguous().float()  # (H_out, H_in)
    bias = (ymu - (xmu / xsd) @ w_std).float()
    meta = {
        "vc_path": vc_src,
        "va_path": va_src,
        "valid_idx_path": str(valid_idx_path) if valid_idx_path is not None else None,
        "n_rows_total": n_total,
        "n_rows_kept": n,
        "n_zero_norm_dropped": n_zero_dropped,
        "n_train": len(tr_idx),
        "n_val": int(n_val),
        "holdout_frac": args.refit_holdout_frac,
        "seed": args.refit_seed,
        "best_lambda": best_lam,
        "lambda_grid": "GCV over logspace(-2, 4, 13) — the #779/#823 fit_h.ridge_fit_predict "
        "recipe that produced the K3 sanity anchor (#823 R^2 ~0.51 at layer 20)",
        "r2_val": r2_val,
        "r2_min": args.refit_r2_min,
    }
    return "refit_ridge(v_c->v_a_prime)", weight, bias, meta


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
        f"steered V_a capture missing: {path} — run the phase-1 driver's 1e capture phase "
        "(issue1415_run_phase1.phase_1e writes the canonical <pair>__<arm>.pt files); "
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

    result: dict = {"per_arm": {}}
    for arm in common.ARMS:
        predicted = []
        realized = []
        pair_ids: list[str] = []
        skipped: list[str] = []
        degenerate: list[str] = []
        for p in pairs:
            if realized_source == "steered":
                spath = steered_dir / f"{p.pair_id}__{arm}.pt"
                if not spath.exists():
                    # Coherence-failed pair: the 1e canonical index records the
                    # skip; excluded WITH a record (plan §8), never silently.
                    skipped.append(p.pair_id)
                    continue
            v_c = p.v_c[arm][li]
            delta = p.delta[arm][li]
            assert v_c.shape == (hidden,) and delta.shape == (hidden,), (p.pair_id, v_c.shape)
            fpred = apply_map(v_c + delta, weight, bias) - apply_map(v_c, weight, bias)
            if realized_source == "steered":
                r = (
                    _load_steered_va(
                        steered_dir / f"{p.pair_id}__{arm}.pt", layer, p.layers, hidden
                    )
                    - p.v_a_c[li]
                )
                if float(r.norm()) == 0.0:
                    # A zero realized shift is a LEGITIMATE outcome (steered
                    # draws identical to baseline — e.g. a sub-threshold alpha
                    # under the shared per-draw seed); cosine is undefined
                    # there, so exclude WITH a record, never crash.
                    degenerate.append(p.pair_id)
                    continue
            else:
                r = p.v_a_cprime[li] - p.v_a_c[li]
            predicted.append(fpred)
            realized.append(r)
            pair_ids.append(p.pair_id)
        assert pair_ids, (
            f"arm {arm!r}: NO usable steered canonical captures under {steered_dir} — "
            "run the phase-1 driver's 1e capture phase first"
        )
        if skipped or degenerate:
            logger.warning(
                "arm %s: %d/%d pairs excluded (no canonical capture: %s; zero realized shift: %s)",
                arm,
                len(skipped) + len(degenerate),
                len(pairs),
                skipped,
                degenerate,
            )
        pred = torch.stack(predicted)  # (P, H)
        real = torch.stack(realized)  # (P, H)
        pn, rn = pred.norm(dim=-1), real.norm(dim=-1)
        assert torch.all(pn > 0) and torch.all(rn > 0), f"{arm}: degenerate zero delta"
        pred_hat, real_hat = pred / pn.unsqueeze(-1), real / rn.unsqueeze(-1)

        observed = (pred_hat * real_hat).sum(dim=-1)  # (P,)

        # Shuffled-pair null: cos(realized_p, predicted_{perm(p)}). The donor
        # rescale to the recipient norm cancels in cosine (scale-invariant).
        n_kept = len(pair_ids)
        cos_matrix = real_hat @ pred_hat.T  # (P_recipient, P_donor) — one matmul
        perms = common.batched_permutations(n_draws, n_kept, seed)  # (D, P)
        null = cos_matrix[torch.arange(n_kept), perms]  # (D, P)

        result["per_arm"][arm] = {
            "skipped_pairs": skipped,
            "degenerate_zero_shift_pairs": degenerate,
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

    refit_meta: dict | None = None
    map_path: Path | None = None
    if args.refit_fallback == "force":
        logger.info("--refit-fallback force: skipping the #922 bundle, plan-K3 refit engaged")
        wpath, weight, bias, refit_meta = refit_ridge_map(args)
    else:
        if args.map_path is not None:
            map_path = args.map_path
        else:
            from huggingface_hub import hf_hub_download

            logger.info("downloading %s (~981 MB; mmap load)", MAP_REPO_PATH)
            map_path = Path(
                hf_hub_download(common.HF_DATA_REPO, MAP_REPO_PATH, repo_type="dataset")
            )
        try:
            wpath, weight, bias = resolve_map(map_path, args.hidden, args.layer)
        except SystemExit as e:
            if args.refit_fallback == "auto" and e.code == RC_STEP0_FAIL:
                logger.warning(
                    "STEP0 FAIL on the #922 bundle -> plan-K3 ridge-refit fallback engaged "
                    "(--refit-fallback auto)"
                )
                wpath, weight, bias, refit_meta = refit_ridge_map(args)
            else:
                raise

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
            "source": "refit_fallback" if refit_meta is not None else "issue922_bundle",
            "repo_path": (
                f"{REFIT_VC_REPO_PATH} + {REFIT_VA_REPO_PATH}"
                if refit_meta is not None
                else (MAP_REPO_PATH if args.map_path is None else str(args.map_path))
            ),
            "resolved_weight_key": wpath,
            "weight_shape": list(weight.shape),
            "weight_dtype": str(weight.dtype),
            "bias_resolved": bias is not None,
            "orientation_convention": "f(x) = x @ W.T (+ b); nn.Linear convention assumed",
            "bundle_bytes": map_path.stat().st_size if map_path is not None else None,
            **({"refit": refit_meta} if refit_meta is not None else {}),
        },
        **result,
        "repro": common.repro_meta("issue1415_map_transport"),
    }
    common.write_json_atomic(args.out_json, out)
    logger.info("wrote %s (%d pairs, source=%s)", args.out_json, len(pairs), args.realized_source)


if __name__ == "__main__":
    main()
