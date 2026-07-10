"""Issue #779 inline analysis — does r_B's variance RANK flip between the
per-token residual covariance (the "Concepts Whisper" object, arXiv 2605.01609)
and the mean-answer covariance (our writeup's object) on GENERIC LMSYS contexts?

Round-1 sibling ``issue779_pertoken_vs_mean_variance.py`` tested this on the
r3_subset EVAL-rig evil stacks (one trait, layers [7,14,21]). This closes the gap
on the pass-B TRAIN contexts (generic LMSYS), ALL THREE traits, at BOTH layers
{14, 19} — the substrate ``issue779_pertoken_lmsys_capture.py`` produced.

Hypothesis (Thomas): per-token PCA is dominated by token-IDENTITY variance, so a
concept direction that persists across tokens (like r_B) is LOW-variance there;
mean-pooling over answer tokens averages that token variance away, promoting the
concept into the HIGH-variance regime. Decisive read: r_B LOW-percentile in the
per-token covariance but HIGH-percentile in the mean-answer covariance ⇒ pooling
hypothesis confirmed.

The per-token covariance + mean-answer covariance depend ONLY on the LAYER (the
activations at that layer), not the trait — r_B is what is trait-by-layer-specific.
So each per-context stack is STREAMED exactly once (download -> reduce -> delete,
memory-bounded to ~one context), the per-layer covariance accumulators + mean-
answer vectors are built once, the 2 layers x 2 spaces = 4 dense eigendecomps run
once, and each of the 3 traits' r_B is then projected against the shared spectra.
0-GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import issue779_common as C  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = C.HF_DATA_REPO
DEFAULT_PREFIX = "issue779_monitoring/pertoken_lmsys"
DEFAULT_LAYERS = [14, 19]


def _load_rb(rb_dir: Path, trait: str, n_layers: int, hidden: int) -> np.ndarray:
    """Full (n_layers, hidden) r_B for a trait (same loader as issue779_stage1)."""
    blob = torch.load(rb_dir / f"{trait}.pt", weights_only=False)
    r_b = blob["r_b"].to(torch.float32).numpy()
    assert r_b.shape == (n_layers, hidden), (trait, r_b.shape, (n_layers, hidden))
    return r_b


def _percentile_and_rank(eigvals_desc: np.ndarray, var_along: float) -> tuple[float, int]:
    """Variance percentile (% of PCA dirs with LESS variance) + equivalent rank."""
    pct = float(np.mean(eigvals_desc < var_along) * 100.0)
    rank = int(np.sum(eigvals_desc > var_along))
    return pct, rank


class _LayerAccum:
    """Streamed per-token covariance + per-context mean-answer vectors for ONE layer."""

    def __init__(self, hidden: int) -> None:
        self.n_tok = 0
        self.tok_sum = np.zeros(hidden, dtype=np.float64)
        self.tok_outer = np.zeros((hidden, hidden), dtype=np.float64)
        self.means: list[np.ndarray] = []

    def update(self, x: np.ndarray) -> None:
        """x: (n_tok_i, H) fp64 answer-token activations for one context at this layer."""
        self.n_tok += x.shape[0]
        self.tok_sum += x.sum(0)
        self.tok_outer += x.T @ x  # one BLAS GEMM per context (vectorized)
        self.means.append(x.mean(0))

    def spectra(self) -> dict:
        """Per-token cov spectrum + mean-answer cov spectrum (each: eigvals desc)."""
        mu = self.tok_sum / self.n_tok
        cov_tok = self.tok_outer / self.n_tok - np.outer(mu, mu)
        eig_tok = np.sort(np.linalg.eigvalsh(cov_tok))[::-1]
        m = np.stack(self.means)  # (n_ctx, H)
        mc = m - m.mean(0)
        cov_mean = (mc.T @ mc) / (len(m) - 1)
        eig_mean = np.sort(np.linalg.eigvalsh(cov_mean))[::-1]
        return {
            "cov_tok": cov_tok,
            "eig_tok": eig_tok,
            "cov_mean": cov_mean,
            "eig_mean": eig_mean,
            "m": m,
            "n_rollouts": len(m),
        }


def _list_context_files(prefix: str, local_dir: Path | None) -> list:
    """List per-context .pt files — from a local dir (smoke) or HF (scoped tree)."""
    if local_dir is not None:
        return sorted((local_dir / "contexts").glob("ctx*.pt"))
    from huggingface_hub import HfApi

    api = HfApi()
    # SCOPED list_repo_tree (a bare list_repo_files on the ~1M-file data repo
    # wedges — #833/#920); the prefix rides the tree URL so pagination covers
    # only this subtree.
    entries = api.list_repo_tree(
        REPO, path_in_repo=f"{prefix}/contexts", repo_type="dataset", recursive=True
    )
    return sorted(e.path for e in entries if e.path.endswith(".pt"))


def _open_context(f, local_dir: Path | None):
    """Return (loaded_dict, local_path_to_delete_or_None). Streams from HF unless local."""
    if local_dir is not None:
        return torch.load(f, weights_only=False, map_location="cpu"), None
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    p = hub.retry_transient(
        lambda: hf_hub_download(REPO, f, repo_type="dataset"), what=f"hf_hub_download {f}"
    )
    return torch.load(p, weights_only=False, map_location="cpu"), Path(p)


def _make_figure(result: dict, layers: list[int], traits: list[str], out_fig: Path) -> None:
    """Grouped bars per trait: r_B variance percentile in per-token vs mean-answer,
    one panel per layer. Clean bars + legend + labels only (no plot annotations)."""
    try:  # best-effort paper style
        from explore_persona_space.analysis import paper_plots

        paper_plots.set_paper_style()
    except Exception:
        pass
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), squeeze=False, layout="constrained")
    x = np.arange(len(traits))
    w = 0.38
    for j, layer in enumerate(layers):
        ax = axes[0][j]
        pt = [result[t][f"L{layer}"]["per_token"]["rb_variance_percentile"] for t in traits]
        mn = [result[t][f"L{layer}"]["mean_answer"]["rb_variance_percentile"] for t in traits]
        ax.bar(x - w / 2, pt, w, label="per-token cov")
        ax.bar(x + w / 2, mn, w, label="mean-answer cov")
        ax.set_xticks(x)
        ax.set_xticklabels(traits, rotation=20, ha="right")
        ax.set_ylabel("r_B variance percentile")
        ax.set_ylim(0, 100)
        ax.set_title(f"layer {layer}")
        ax.legend()
    fig.suptitle("r_B variance percentile: per-token vs mean-answer (generic LMSYS)")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #779 per-token vs mean-answer r_B variance.")
    ap.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="read contexts/ctx*.pt from this local dir instead of HF (smoke)",
    )
    ap.add_argument("--rb-dir", type=Path, default=Path("data/issue_779/r_b"))
    ap.add_argument("--n-layers", type=int, default=C.EXPECTED_LAYERS)
    ap.add_argument("--hidden", type=int, default=C.EXPECTED_HIDDEN)
    ap.add_argument("--max-files", type=int, default=None, help="cap files streamed (smoke)")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("eval_results/issue_779/pertoken_lmsys/pertoken_lmsys_variance.json"),
    )
    ap.add_argument(
        "--out-fig",
        type=Path,
        default=Path("figures/issue_779/h_pertoken_vs_mean_rb_rank.png"),
    )
    args = ap.parse_args()

    layers = args.layers
    files = _list_context_files(args.prefix, args.local_dir)
    if args.max_files is not None:
        files = files[: args.max_files]
    assert files, f"no per-token context files found ({args.local_dir or args.prefix})"
    print(f"streaming {len(files)} per-token context files at layers {layers}", flush=True)

    # Per-layer accumulators (covariance depends on layer, not trait) — stream
    # each context ONCE, update every layer.
    acc = {layer: _LayerAccum(args.hidden) for layer in layers}
    for i, f in enumerate(files):
        d, to_delete = _open_context(f, args.local_dir)
        file_layers = list(d["layers"])
        stack = d["answer_per_token"]  # (n_tok, L, H)
        for layer in layers:
            li = file_layers.index(layer)
            x = stack[:, li, :].to(torch.float64).numpy()  # (n_tok, H)
            acc[layer].update(x)
        if to_delete is not None:
            to_delete.unlink(missing_ok=True)  # stream: drop after reduce
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)} contexts streamed", flush=True)

    # 4 dense eigendecomps (2 layers x 2 spaces), computed ONCE and shared across traits.
    spectra = {layer: acc[layer].spectra() for layer in layers}

    # Load each trait's r_B once; project against the shared per-layer spectra.
    result: dict = {}
    for trait in args.traits:
        rb_full = _load_rb(args.rb_dir, trait, args.n_layers, args.hidden)
        result[trait] = {}
        for layer in layers:
            sp = spectra[layer]
            rb = rb_full[layer]
            rb = rb / (np.linalg.norm(rb) + 1e-12)
            var_tok = float(rb @ sp["cov_tok"] @ rb)
            pct_tok, rank_tok = _percentile_and_rank(sp["eig_tok"], var_tok)
            var_mean = float(np.var(sp["m"] @ rb, ddof=1))
            pct_mean, rank_mean = _percentile_and_rank(sp["eig_mean"], var_mean)
            result[trait][f"L{layer}"] = {
                "per_token": {
                    "rb_variance_percentile": pct_tok,
                    "rb_equivalent_rank": rank_tok,
                    "rb_variance_fraction": var_tok / float(sp["eig_tok"].sum()),
                },
                "mean_answer": {
                    "rb_variance_percentile": pct_mean,
                    "rb_equivalent_rank": rank_mean,
                    "rb_variance_fraction": var_mean / float(sp["eig_mean"].sum()),
                },
            }

    out = {
        "traits": args.traits,
        "layers": layers,
        "n_context_files": len(files),
        "per_layer": {
            f"L{layer}": {
                "n_tokens": int(acc[layer].n_tok),
                "n_rollouts": spectra[layer]["n_rollouts"],
            }
            for layer in layers
        },
        "result": result,
        "note": (
            "Generic pass-B LMSYS contexts. per_token = PCA over ALL answer tokens; "
            "mean_answer = PCA over per-rollout mean-answer vectors, SAME rollouts. "
            "r_B LOW-percentile per-token but HIGH-percentile mean-answer ⇒ the "
            "mean-pooling promotes the concept into the high-variance regime "
            "(pooling hypothesis confirmed)."
        ),
        "metadata": C.reproducibility_metadata(
            {"script": "issue779_pertoken_lmsys_analysis", "substrate": "pass-B LMSYS"}
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1))
    _make_figure(result, layers, args.traits, args.out_fig)

    print(f"\nwrote {args.out_json}\nwrote {args.out_fig}", flush=True)
    for trait in args.traits:
        for layer in layers:
            r = result[trait][f"L{layer}"]
            print(
                f"  {trait} L{layer}: per-token {r['per_token']['rb_variance_percentile']:.1f}th "
                f"pct (rank {r['per_token']['rb_equivalent_rank']}) | mean-answer "
                f"{r['mean_answer']['rb_variance_percentile']:.1f}th pct "
                f"(rank {r['mean_answer']['rb_equivalent_rank']})",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
