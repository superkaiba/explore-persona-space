"""Issue #779 — project r_B onto the UNEMBEDDING covariance eigenbasis and compute
the "Concepts Whisper" (arXiv 2605.01609) metrics, to settle whether our
persona-vector finding AGREES with them (r_B anti-concentrates in the readout
geometry) or genuinely CONTRADICTS (r_B concentrates in high-eigenvalue readout
directions).

Their spectral object: Sigma = (1/V) W_U^T W_U + lambda I  (Park et al. causal
inner product; W_U = unembedding rows). Energy E_i = (v . u_i)^2/||v||^2 across
Sigma's eigenvectors u_i (descending eigenvalue). Metrics: Gini deviation (signed
area of the energy CDF C(k) vs cumulative eigenvalue fraction V(k) — NEGATIVE =
energy in the low-eigenvalue TAIL = anti-concentrate) and spectral center of mass
SCM = V(k) at which C(k) hits 0.5 (SCM -> 1 = anti-concentrate). Baseline: 200
random unit directions. 0-GPU, ~seconds.
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
import issue779_stage1 as S1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

QWEN_SNAP = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
TRAITS = ("evil", "sycophancy", "hallucination")


def _load_unembedding():
    """W_U (V, d): lm_head.weight, or the tied embed_tokens.weight if lm_head absent."""
    snap = next(QWEN_SNAP.iterdir())
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    key = "lm_head.weight" if "lm_head.weight" in idx else "model.embed_tokens.weight"
    shard = snap / idx[key]
    with safe_open(shard, framework="pt") as f:
        W = f.get_tensor(key).to(torch.float64).numpy()
    return W, key


def _energy_metrics(v, U, Vfrac):
    """Gini deviation + SCM for direction v against eigenbasis U (cols, desc eigval)."""
    v = v / (np.linalg.norm(v) + 1e-12)
    E = (v @ U) ** 2  # (d,) fractional energy per eigenvector (||v||=1)
    C = np.cumsum(E)  # cumulative energy CDF
    # Gini deviation: signed area between C(k) and the cumulative-eigenvalue baseline V(k),
    # integrated over V. Negative => energy trails the eigenvalue mass => low-eigval tail.
    gini = float(np.trapezoid(C - Vfrac, Vfrac))
    scm = float(Vfrac[np.searchsorted(C, 0.5)]) if C[-1] >= 0.5 else 1.0
    return gini, scm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lam", type=float, default=1e-3, help="ridge on Sigma (fraction of mean eigval)"
    )
    ap.add_argument("--layers", type=int, nargs="*", default=[14, 19, 27])
    ap.add_argument("--rb-dir", type=Path, default=Path("data/issue_779/r_b"))
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("eval_results/issue_779/pertoken_lmsys/rb_unembedding_spectrum.json"),
    )
    args = ap.parse_args()

    W, key = _load_unembedding()
    V, d = W.shape
    Sig = (W.T @ W) / V
    Sig += args.lam * np.trace(Sig) / d * np.eye(d)  # lambda I, scaled to mean eigval
    eigval, U = np.linalg.eigh(Sig)  # ascending
    order = eigval[::-1]
    U = U[:, ::-1]  # columns = eigvecs, descending eigenvalue
    Vfrac = np.cumsum(order) / order.sum()

    rng = np.random.default_rng(779)
    null = np.array([_energy_metrics(rng.standard_normal(d), U, Vfrac) for _ in range(200)])
    null_gini_mean, null_gini_sd = float(null[:, 0].mean()), float(null[:, 0].std())
    null_scm_mean = float(null[:, 1].mean())

    out = {
        "unembedding_key": key,
        "V": V,
        "d": d,
        "lam": args.lam,
        "random_null": {
            "gini_mean": null_gini_mean,
            "gini_sd": null_gini_sd,
            "scm_mean": null_scm_mean,
        },
        "r_b": {},
    }
    print(f"unembedding = {key}  (V={V}, d={d})")
    print(f"random null: Gini {null_gini_mean:+.4f}±{null_gini_sd:.4f}, SCM {null_scm_mean:.3f}")
    print(f"{'trait':>14} {'layer':>5} {'Gini':>9} {'vs null(z)':>10} {'SCM':>6}  verdict")
    for t in TRAITS:
        rb_all = S1._load_rb(args.rb_dir, t, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)
        out["r_b"][t] = {}
        for li in args.layers:
            g, scm = _energy_metrics(rb_all[li].astype(np.float64), U, Vfrac)
            z = (g - null_gini_mean) / (null_gini_sd + 1e-12)
            verdict = (
                "ANTI-concentrate (agree w/ paper)"
                if g < null_gini_mean
                else "CONCENTRATE (contradict)"
            )
            out["r_b"][t][f"L{li}"] = {
                "gini_deviation": g,
                "gini_z_vs_null": z,
                "scm": scm,
                "verdict": verdict,
            }
            print(f"{t:>14} {li:>5} {g:>+9.4f} {z:>+10.2f} {scm:>6.3f}  {verdict}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
