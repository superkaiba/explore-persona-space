"""Re-verify the #1739 u-full apply path against the payload's documented convention.

Record-integrity probe (#1739 correction round). The question: did the committed
`map_i1739_ufull` column apply the payload's own whitening
(``meta["apply"]``: ``pred = ((x - x_mu)/x_sd) @ w + y_mu``), or did it skip it and
feed RAW x to ``w``?

The probe calls the COMMITTED readout's own loader + apply expression — imported
from ``issue1739_map963k_readout``, not re-implemented here — so whatever it
reports is what the committed comparison.json column actually computed. It then
scores the hypothesised skipped-whitening form on the SAME rows for contrast, and
measures how far the eval inputs sit from the map's own train-fold mean (the
distribution-shift read that separates "mis-applied" from "over-fit").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Before any heavy import: the shared-VM thread caps (#847) are frozen by numpy /
# torch at IMPORT, so load_dotenv() must run first for them to bind in-process.
load_dotenv()

import numpy as np  # noqa: E402


def _ensure_repo_root_on_syspath() -> None:
    """Put the repo root on sys.path so ``scripts.*`` imports resolve in script mode."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_map963k_readout.py"
    assert sentinel.is_file(), f"repo-root resolution wrong: {sentinel} missing"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    """Held-out R^2 against the target's own mean (the committed convention)."""
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rowcos(pred: np.ndarray, y: np.ndarray) -> float:
    """Mean per-row cosine between prediction and target."""
    pn = np.linalg.norm(pred, axis=1)
    yn = np.linalg.norm(y, axis=1)
    ok = (pn > 0) & (yn > 0)
    return float(((pred[ok] * y[ok]).sum(axis=1) / (pn[ok] * yn[ok])).mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-dir", type=Path, required=True)
    ap.add_argument("--map-npz", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--variant", default="context_end")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    _ensure_repo_root_on_syspath()
    from issue1739_map963k_readout import load_i1739_map  # the COMMITTED loader

    store_io = __import__(
        "explore_persona_space.experiments.issue_1739.store_io", fromlist=["load_summaries"]
    )
    arrays, _meta = store_io.load_summaries(
        str(args.store_dir), (args.variant, "t1"), (args.layer,), hidden_dim=3584
    )
    x = np.asarray(arrays[(args.variant, args.layer)], dtype=np.float64)
    y = np.asarray(arrays[("t1", args.layer)], dtype=np.float64)
    assert x.shape == y.shape, (x.shape, y.shape)

    # The COMMITTED loader + the COMMITTED line-469 expression, verbatim.
    # (#1975 widened the loader's return with the payload meta; this probe's
    # own behavior is unchanged — it keeps computing the same three forms.)
    w, x_mu, x_sd, y_mu, _map_meta = load_i1739_map(args.map_npz, args.layer)
    committed = ((x - x_mu) / x_sd) @ w + y_mu

    # The hypothesised skipped-whitening forms, same rows.
    raw_nobias = x @ w
    raw_bias = x @ w + y_mu

    rows = {}
    for name, pred in [
        ("committed_documented_whitened", committed),
        ("hypothesised_raw_x_at_w", raw_nobias),
        ("hypothesised_raw_x_at_w_plus_ymu", raw_bias),
    ]:
        rows[name] = {
            "cos": rowcos(pred, y),
            "r2": r2(pred, y),
            "pred_norm_mean": float(np.linalg.norm(pred, axis=1).mean()),
        }

    # Distribution-shift read: how far are the eval inputs from the map's own
    # train-fold mean, in the map's own x_sd units? A large value means the
    # over-scale is off-distribution extrapolation, not a mis-applied formula.
    z = (x - x_mu) / x_sd
    out = {
        "variant": args.variant,
        "layer": args.layer,
        "n_rows": int(x.shape[0]),
        "target_norm_mean": float(np.linalg.norm(y, axis=1).mean()),
        "arms": rows,
        "distribution_shift": {
            "whitened_input_norm_mean": float(np.linalg.norm(z, axis=1).mean()),
            "whitened_input_absmean_per_dim": float(np.abs(z).mean()),
            "sqrt_dim_reference": float(np.sqrt(x.shape[1])),
            "note": (
                "whitened_input_norm_mean >> sqrt(dim) means the eval inputs sit far "
                "outside the map's train-fold distribution in its own whitened space"
            ),
        },
        "map_scale": {
            "w_frobenius": float(np.linalg.norm(w)),
            "y_mu_norm": float(np.linalg.norm(y_mu)),
            "inv_x_sd_mean": float((1.0 / x_sd).mean()),
            "inv_x_sd_max": float((1.0 / x_sd).max()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(f"n_rows={out['n_rows']}  ||t1||={out['target_norm_mean']:.2f}")
    for name, r in rows.items():
        print(
            f"  {name:36s} cos={r['cos']:+.4f}  R2={r['r2']:14.2f}  "
            f"|pred|={r['pred_norm_mean']:9.2f}"
        )
    ds = out["distribution_shift"]
    print(
        f"  whitened-input norm {ds['whitened_input_norm_mean']:.1f} "
        f"vs sqrt(dim) {ds['sqrt_dim_reference']:.1f}"
    )
    print(f"wrote {args.out}")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
