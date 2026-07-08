"""Issue #779 inline analysis — does r_B's variance RANK flip between the per-token
residual covariance (the "Concepts Whisper" paper's object) and the mean-answer
covariance (our writeup's object)?

Hypothesis (Thomas): per-token PCA is dominated by token-IDENTITY variance, so a
concept direction that persists across tokens (like r_B) is LOW-variance there;
mean-pooling over answer tokens averages that token variance away, promoting the
concept into the HIGH-variance regime. If so, r_B should sit at a LOW variance
percentile in the per-token covariance but a HIGH percentile in the mean-answer
covariance, on the SAME rollouts.

Substrate: the parent round's r3_subset per-token answer stacks (EVIL only, eval
rig, layers [7,14,21]). We use L14 = evil's read-out layer. Streamed covariance
(one file in memory at a time). 0-GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_stage1 as S1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue779_monitoring/analysis_tensors/r3_subset"


def _percentile_and_rank(eigvals_desc, var_along):
    """Variance percentile (% of PCA dirs with LESS variance) + equivalent rank."""
    pct = float(np.mean(eigvals_desc < var_along) * 100.0)
    rank = int(np.sum(eigvals_desc > var_along))
    return pct, rank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="evil")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument(
        "--n-files", type=int, default=300, help="rollout files to stream (evenly over conds)"
    )
    ap.add_argument("--rb-dir", type=Path, default=Path("data/issue_779/r_b"))
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "eval_results/issue_779/fitter-fair-comparison/pertoken_vs_mean_variance.json"
        ),
    )
    args = ap.parse_args()

    files = sorted(
        f
        for f in list_repo_files(REPO, repo_type="dataset")
        if f.startswith(PREFIX) and f"/{args.trait}__" in f
    )
    # sample evenly across the 10 conditions
    step = max(1, len(files) // args.n_files)
    files = files[::step][: args.n_files]
    print(f"streaming {len(files)} {args.trait} per-token files at L{args.layer}")

    rb = S1._load_rb(args.rb_dir, args.trait, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)[args.layer]
    rb = rb / (np.linalg.norm(rb) + 1e-12)

    H = C.EXPECTED_HIDDEN
    # streamed accumulators for the PER-TOKEN covariance
    n_tok = 0
    tok_sum = np.zeros(H, dtype=np.float64)
    tok_outer = np.zeros((H, H), dtype=np.float64)
    means = []  # per-rollout mean-answer vectors
    li = None
    for i, f in enumerate(files):
        p = hf_hub_download(REPO, f, repo_type="dataset")
        d = torch.load(p, weights_only=False, map_location="cpu")
        if li is None:
            li = list(d["r3_layers"]).index(args.layer)
        X = d["answer_per_token"][:, li, :].to(torch.float64).numpy()  # (n_tok_i, H)
        n_tok += X.shape[0]
        tok_sum += X.sum(0)
        tok_outer += X.T @ X
        means.append(X.mean(0))
        Path(p).unlink(missing_ok=True)  # stream: drop after reduce
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)} files, {n_tok} tokens")

    # per-token covariance + spectrum
    mu = tok_sum / n_tok
    cov_tok = tok_outer / n_tok - np.outer(mu, mu)
    eig_tok = np.sort(np.linalg.eigvalsh(cov_tok))[::-1]
    var_rb_tok = float(rb @ cov_tok @ rb)
    pct_tok, rank_tok = _percentile_and_rank(eig_tok, var_rb_tok)
    frac_tok = var_rb_tok / float(eig_tok.sum())

    # mean-answer covariance + spectrum (same rollouts)
    M = np.stack(means)  # (n_roll, H)
    Mc = M - M.mean(0)
    cov_mean = (Mc.T @ Mc) / (len(M) - 1)
    eig_mean = np.sort(np.linalg.eigvalsh(cov_mean))[::-1]
    var_rb_mean = float(np.var(M @ rb, ddof=1))
    pct_mean, rank_mean = _percentile_and_rank(eig_mean, var_rb_mean)
    frac_mean = var_rb_mean / float(eig_mean.sum())

    out = {
        "trait": args.trait,
        "layer": args.layer,
        "n_files": len(files),
        "n_tokens": int(n_tok),
        "per_token": {
            "rb_variance_percentile": pct_tok,
            "rb_equivalent_rank": rank_tok,
            "rb_variance_fraction": frac_tok,
        },
        "mean_answer": {
            "rb_variance_percentile": pct_mean,
            "rb_equivalent_rank": rank_mean,
            "rb_variance_fraction": frac_mean,
            "n_rollouts": len(M),
        },
        "note": (
            "Same eval-rig evil rollouts. per_token = PCA over ALL answer tokens; "
            "mean_answer = PCA over per-rollout mean-answer vectors. If r_B is low-percentile "
            "per-token but high-percentile mean-answer, the pooling hypothesis holds."
        ),
        "metadata": {
            "script": "issue779_pertoken_vs_mean_variance",
            "substrate": "r3_subset eval-rig",
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(__import__("json").dumps(out, indent=1))
    print(f"\nwrote {args.out_json}")
    print(
        f"  PER-TOKEN cov:  r_B at {pct_tok:.1f}th pct (rank {rank_tok}), "
        f"{frac_tok:.3%} of variance"
    )
    print(
        f"  MEAN-ANSWER cov: r_B at {pct_mean:.1f}th pct (rank {rank_mean}), "
        f"{frac_mean:.3%} of variance"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
