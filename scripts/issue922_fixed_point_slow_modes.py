#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #922 free-analysis follow-up: fixed point + slow eigen-directions.

Extends the spectral read (``issue922_spectral_read.py``) of the fitted global
per-layer affine next-token map. For the GLOBAL context-only closed-form map
``ridge_ctx_answer`` at each read-out block the RAW-space one-step dynamics are

    h_{t+1} = A h_t + a,   A = I + wᵀ diag(1/sd),   a = bias − wᵀ (mu/sd)

(the RidgeMap predicts ``Δ̂ = bias + ((h − mu)/sd) @ w`` with σ ≡ 1, and the
rollout composes ``h_{t+1} = h_t + Δ̂``; A matches ``issue922_spectral_read``'s
Jacobian). All maps are contractions (ρ < 1, verified), so a unique attracting
fixed point ``h* = (I − A)⁻¹ a`` exists. This script computes:

1. Fixed point location: ‖h*‖, its percentile within the store norm
   distribution, cosines to the mean answer / prompt-end / drift vectors, and a
   per-dim-std-normalized distance to the mean answer state.
2. Convergence toward h*: roll the ~500 test prompt-end states K=0..256,
   report median distance-to-h* vs K, empirical e-folding vs −1/ln ρ, and the
   between-context variance decay (halving / 10% K; K=32 separation).
3. Slow modes: full complex eig of A; counts |λ| > {0.99,0.98,0.95,0.9}, time
   constants τ = −1/ln|λ|, rotation periods 2π/|arg λ| for complex slow modes.
4. Slow-subspace content: real slow subspace S_c at cutoffs {0.99,0.98,0.95,
   0.9}; projection energy of the #779 trait directions, top-10 between-context
   PCs, the drift a, and (mean answer − mean prompt-end), each vs a
   random-subspace null band; aggregate between-context-variance fraction in
   S_c vs null, for prompt-end AND K=32 rolled states.
5. Guardrails: subspace dim reported alongside every projection; nonnormality
   via σ_max(A) alongside ρ (transient-growth check).

All states are teacher-forced captures of the model's own completions
(provenance inherited from #922's stores). Outputs: JSON aggregates +
top-64-slow-eigvec NPZ + four figures.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue922_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue922_fixed_point_slow_modes")

# Self-contained cache OUTSIDE data/issue_<N>/ and the /tmp issue-keyed sweep
# patterns (i<N>*/issue<N>*/*_<N>) — the disk-guard cron reaps
# data/issue_922/hf_dl for a terminal-status task (#922 is at awaiting_promotion)
# mid-run, so inputs must live where neither janitor sweeps.
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
REV_922 = "efef63576e2e2c8fd1d4216b067bdc8581693cea"
REV_779 = "037fcbb210bc52c459959b0746cc268fe08bae96"
CACHE_DIR = Path("/tmp/eps922fpsm_cache")
INPUT_FILES = {
    "maps": (
        "issue922_nexttoken/maps/maps_boundary_and_lstar_fp16.pt",
        REV_922,
        "1f1aaa839473f5508029ff05ad4345aa50b7eb9bc203f7ce79fa71ee9fd8f1dd",
    ),
    "store": (
        "issue922_nexttoken/store_test/store_test_contexts.pt",
        REV_922,
        "1e7c2677384e3ed74b1e2623682ed8cb17389b1b7121eba82f2304473c5a47cd",
    ),
    "rb_evil": ("issue779_monitoring/r_b/evil.pt", REV_779, None),
    "rb_sycophancy": ("issue779_monitoring/r_b/sycophancy.pt", REV_779, None),
    "rb_hallucination": ("issue779_monitoring/r_b/hallucination.pt", REV_779, None),
}
CUTOFFS = (0.99, 0.98, 0.95, 0.90)
ROLL_K = 256
K_HEADLINE = 32
N_NULL_VEC = 200  # per-vector null draws (exact random-subspace via rotational invariance)
N_NULL_AGG = 100  # aggregate between-context-variance null draws (random orthonormal subspaces)
IMAG_TOL = 1e-7  # |Im λ| below this ⇒ treat eigenvalue as real (real matrix eig)


def _fetch_input(key: str) -> Path:
    """hf_hub_download one pinned input into CACHE_DIR (sha-checked, fail-loud)."""
    from huggingface_hub import hf_hub_download

    rel, rev, want_sha = INPUT_FILES[key]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = Path(
        hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=rel,
            repo_type="dataset",
            revision=rev,
            cache_dir=str(CACHE_DIR),
        )
    )
    if want_sha is not None:
        got = C.sha256_path(p)
        assert got == want_sha, f"sha mismatch for {rel}: {got} != {want_sha}"
    return p


def load_rb_direct(key: str) -> np.ndarray:
    """Persona direction r_B (28, H) fp32 from a pinned r_b/<trait>.pt (block-indexed)."""
    blob = torch.load(_fetch_input(key), weights_only=False)
    r_b = blob["r_b"].to(torch.float32).numpy()
    assert r_b.shape == (C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN), (key, r_b.shape)
    return r_b


def affine_from_ridge_state(state: dict, h_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """RAW-space affine (A, a): h_{t+1} = A h + a, fp64 (h-block if d_in > H).

    A = I + w_hᵀ diag(1/sd_h); a = bias − w_hᵀ (mu_h/sd_h). For the ctx arm
    d_in == H (state-only), so a is the autonomous intercept. For a 2H-input arm
    (tok) the h-block gives the autonomous-in-h Jacobian; ``a`` is then the
    h-block intercept and NOT a full fixed-point intercept (e is exogenous).
    """
    assert float(state["sigma"]) == 1.0, state["sigma"]
    w = state["w"].to(torch.float64)
    sd = state["sd"].to(torch.float64)
    mu = state["mu"].to(torch.float64)
    bias = state["bias"].to(torch.float64)
    assert w.shape[1] == h_dim and w.shape[0] in (h_dim, 2 * h_dim), w.shape
    w_h, sd_h, mu_h = w[:h_dim, :], sd[:h_dim], mu[:h_dim]
    A = torch.eye(h_dim, dtype=torch.float64) + (w_h.t() / sd_h.unsqueeze(0))
    a = bias - (w_h.t() @ (mu_h / sd_h))
    return A, a


def sigma_max(A: torch.Tensor, iters: int = 60) -> float:
    """Top singular value of A via power iteration on AᵀA (fp64) — transient-growth check."""
    d = A.shape[0]
    v = torch.randn(d, dtype=torch.float64)
    v = v / v.norm()
    At = A.t()
    for _ in range(iters):
        v = At @ (A @ v)
        n = v.norm()
        v = v / n
    return float(n.sqrt())


def build_slow_subspace(
    L: torch.Tensor, V: torch.Tensor, cutoff: float
) -> tuple[torch.Tensor, int, int, int]:
    """Real orthonormal basis of the invariant subspace for |λ| > cutoff.

    For a real matrix, eigenvalues come in conjugate pairs. Select representatives
    with Im(λ) >= -IMAG_TOL and |λ| > cutoff; a real eigenvalue contributes
    Re(v), a complex one contributes Re(v) AND Im(v). QR-orthonormalize.
    Returns (S (H, d) real fp64, dim d, n_real, n_complex_pairs).
    """
    mod = L.abs()
    cols: list[torch.Tensor] = []
    n_real = n_cpx = 0
    for i in range(L.shape[0]):
        if float(mod[i]) <= cutoff:
            continue
        im = float(L[i].imag)
        if im < -IMAG_TOL:
            continue  # the Im<0 member of a conjugate pair (skip; its twin is taken)
        vi = V[:, i]
        if abs(im) <= IMAG_TOL:
            cols.append(vi.real.to(torch.float64))
            n_real += 1
        else:
            cols.append(vi.real.to(torch.float64))
            cols.append(vi.imag.to(torch.float64))
            n_cpx += 1
    if not cols:
        H = V.shape[0]
        return torch.zeros(H, 0, dtype=torch.float64), 0, 0, 0
    M = torch.stack(cols, dim=1)  # (H, m)
    Q, _ = torch.linalg.qr(M, mode="reduced")
    return Q, Q.shape[1], n_real, n_cpx


def proj_energy(S: torch.Tensor, v: torch.Tensor) -> float:
    """Fraction of ‖v‖² captured by the orthonormal subspace S (H, d)."""
    if S.shape[1] == 0:
        return 0.0
    vn2 = float((v * v).sum())
    if vn2 == 0.0:
        return 0.0
    p = S.t() @ v
    return float((p * p).sum() / vn2)


def vector_null_band(dim: int, h: int, n_draws: int, rng: np.random.Generator) -> dict:
    """Null band for a FIXED unit vector's projection energy onto a random dim-subspace.

    By rotational invariance this is Beta(dim/2, (h-dim)/2); we SAMPLE it exactly
    via chi-square ratios (identical to drawing n_draws random orthonormal
    subspaces and projecting), matching the brief's 200-random-subspace protocol.
    """
    if dim == 0:
        return {"dim": 0, "mean": 0.0, "p2.5": 0.0, "p50": 0.0, "p97.5": 0.0}
    g = rng.standard_normal((n_draws, h)) ** 2
    frac = g[:, :dim].sum(1) / g.sum(1)
    return {
        "dim": int(dim),
        "mean": float(frac.mean()),
        "p2.5": float(np.percentile(frac, 2.5)),
        "p50": float(np.percentile(frac, 50.0)),
        "p97.5": float(np.percentile(frac, 97.5)),
    }


def aggregate_variance_fraction(Xc: torch.Tensor, S: torch.Tensor) -> float:
    """Fraction of total between-row variance (‖Xc‖_F²) captured by subspace S."""
    if S.shape[1] == 0:
        return 0.0
    tot = float((Xc * Xc).sum())
    if tot == 0.0:
        return 0.0
    proj = Xc @ S  # (n, d)
    return float((proj * proj).sum() / tot)


def aggregate_variance_null(
    Xc: torch.Tensor, dim: int, h: int, n_draws: int, rng: np.random.Generator
) -> dict:
    """Null band for the between-row-variance fraction captured by random dim-subspaces.

    Draws n_draws random orthonormal subspaces (QR of Gaussians) and projects Xc.
    fp32 for speed (it is a band). Depends on Xc's spectrum (not just dim), so it
    is sampled, not analytic.
    """
    if dim == 0:
        return {"dim": 0, "mean": 0.0, "p2.5": 0.0, "p50": 0.0, "p97.5": 0.0}
    Xf = Xc.to(torch.float32)
    tot = float((Xf * Xf).sum())
    fracs = np.empty(n_draws, dtype=np.float64)
    for j in range(n_draws):
        G = torch.randn(h, dim, dtype=torch.float32)
        Q, _ = torch.linalg.qr(G, mode="reduced")
        proj = Xf @ Q
        fracs[j] = float((proj * proj).sum()) / tot
    return {
        "dim": int(dim),
        "mean": float(fracs.mean()),
        "p2.5": float(np.percentile(fracs, 2.5)),
        "p50": float(np.percentile(fracs, 50.0)),
        "p97.5": float(np.percentile(fracs, 97.5)),
    }


def load_test_states(store_p: Path, rows: list[int], h_dim: int) -> dict:
    """Per store row: prompt-end states, mean answer state, all-position norms.

    prompt-end T-row convention: prompt_len − 1 − window_start (the capture
    convention, issue922_capture_positions.py:301). Answer states = source
    positions labelled SEG_ANSWER. Returns fp64.
    """
    # mmap: keep the ~4.5 GB store on disk, page in only the slices we read
    blob = torch.load(store_p, weights_only=False, mmap=True)
    assert blob.get("corpus") == "lmsys_test", blob.get("corpus")
    ctxs = blob["contexts"]
    promptend: dict[int, list[torch.Tensor]] = {r: [] for r in rows}
    ans_sum: dict[int, torch.Tensor] = {r: torch.zeros(h_dim, dtype=torch.float64) for r in rows}
    ans_cnt: dict[int, int] = {r: 0 for r in rows}
    norms: dict[int, list[np.ndarray]] = {r: [] for r in rows}
    for ci in sorted(ctxs):
        rec = ctxs[ci]
        h = rec["h"]  # (n_pos, R, H) fp16
        pl, ws = int(rec["prompt_len"]), int(rec["window_start"])
        t_row = pl - 1 - ws
        npos = h.shape[0]
        assert 0 <= t_row < npos, (ci, t_row, npos)
        seg = np.asarray(rec["segments"])  # (n_pos-1,) per SOURCE position
        ans_idx = np.nonzero(seg == C.SEG_ANSWER)[0]
        for r in rows:
            hr = h[:, r, :].to(torch.float64)  # (n_pos, H)
            promptend[r].append(hr[t_row].clone())
            norms[r].append(hr.norm(dim=1).numpy())
            if len(ans_idx):
                ans_sum[r] += hr[ans_idx].sum(0)
                ans_cnt[r] += len(ans_idx)
    n_ctx = len(ctxs)
    out = {"n_test_contexts": n_ctx, "rows": {}}
    for r in rows:
        pe = torch.stack(promptend[r])  # (n_ctx, H)
        out["rows"][r] = {
            "promptend": pe,
            "mean_promptend": pe.mean(0),
            "mean_answer": ans_sum[r] / max(ans_cnt[r], 1),
            "n_answer_pos": ans_cnt[r],
            "all_norms": np.concatenate(norms[r]),
        }
    del blob, ctxs
    return out


def roll_and_summarize(
    A: torch.Tensor, a: torch.Tensor, states: torch.Tensor, h_star: torch.Tensor
):
    """Roll states K=0..ROLL_K; return per-K median dist-to-h*, trace-cov, and K32 snapshot.

    fp32 roll (a convergence read, not a precision-critical eig).
    """
    Af, af, hf = A.to(torch.float32), a.to(torch.float32), states.to(torch.float32)
    hs = h_star.to(torch.float32)
    med_dist, trace_cov = [], []
    snapshot_k32 = None
    hk = hf.clone()
    for k in range(ROLL_K + 1):
        d = (hk - hs).norm(dim=1)  # (n,)
        med_dist.append(float(d.median()))
        mean_k = hk.mean(0)
        trace_cov.append(float(((hk - mean_k) ** 2).sum(1).mean()))  # trace of between-ctx cov
        if k == K_HEADLINE:
            snapshot_k32 = hk.clone().to(torch.float64)
        if k < ROLL_K:
            hk = hk @ Af.t() + af
    return np.array(med_dist), np.array(trace_cov), snapshot_k32


def efolding_from_curve(curve: np.ndarray) -> dict:
    """Empirical e-folding diagnostics for a monotone-decaying curve vs K."""
    c0 = curve[0]
    out = {"initial": float(c0), "final": float(curve[-1])}
    # first K where curve <= c0/e and <= c0/2
    thr_e = c0 / np.e
    thr_half = c0 / 2.0
    below_e = np.nonzero(curve <= thr_e)[0]
    below_half = np.nonzero(curve <= thr_half)[0]
    out["k_first_efold"] = int(below_e[0]) if len(below_e) else None
    out["k_half"] = int(below_half[0]) if len(below_half) else None
    # asymptotic decay rate from a log-linear fit over the last half (positive vals)
    ks = np.arange(len(curve))
    lo = len(curve) // 2
    seg = curve[lo:]
    kseg = ks[lo:]
    pos = seg > 0
    if pos.sum() >= 3:
        slope = np.polyfit(kseg[pos], np.log(seg[pos]), 1)[0]
        out["asymptotic_efold_steps"] = float(-1.0 / slope) if slope < 0 else None
    else:
        out["asymptotic_efold_steps"] = None
    return out


def make_figures(results: dict, blocks: list[int], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    colors = dict(zip(blocks, paper_palette(len(blocks)), strict=True))
    per = results["per_block"]

    # ── Figure 1: spectrum + time constants ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for b in blocks:
        top = per[str(b)]["slow_modes"]["top100_moduli"]
        ax1.plot(np.arange(1, len(top) + 1), top, color=colors[b], lw=1.4, label=f"block {b}")
    ax1.axhline(1.0, color="black", lw=0.9, ls="--")
    ax1.set_xlabel("eigenvalue rank (by modulus)")
    ax1.set_ylabel("eigenvalue modulus |lambda|")
    ax1.set_title("Top-100 eigenvalue moduli of the one-step map")
    ax1.legend(fontsize=7, ncol=2)
    cut_labels = ["0.99", "0.98", "0.95", "0.90"]
    xs = np.arange(len(cut_labels))
    width = 0.8 / len(blocks)
    for bi, b in enumerate(blocks):
        counts = [per[str(b)]["slow_modes"]["counts"][c] for c in cut_labels]
        ax2.bar(xs + (bi - (len(blocks) - 1) / 2) * width, counts, width=width, color=colors[b])
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"|lambda|>{c}" for c in cut_labels])
    ax2.set_ylabel("number of eigen-modes")
    ax2.set_title("Slow-mode counts per contraction threshold")
    fig.tight_layout()
    savefig_paper(fig, "fp_spectrum_timeconstants", fig_dir)
    plt.close(fig)

    # ── Figure 2: convergence + between-context variance decay ────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for b in blocks:
        conv = per[str(b)]["convergence"]
        K = np.arange(len(conv["median_dist_to_hstar"]))
        ax1.plot(K, conv["median_dist_to_hstar"], color=colors[b], lw=1.4, label=f"block {b}")
    ax1.axvline(K_HEADLINE, color="grey", lw=0.9, ls=":")
    ax1.set_yscale("log")
    ax1.set_xlabel("rollout step K")
    ax1.set_ylabel("median distance to fixed point h*")
    ax1.set_title("Convergence toward the fixed point (dotted: K=32)")
    ax1.legend(fontsize=7, ncol=2)
    for b in blocks:
        conv = per[str(b)]["convergence"]
        tc = np.array(conv["between_ctx_var"])
        K = np.arange(len(tc))
        ax2.plot(K, tc / tc[0], color=colors[b], lw=1.4, label=f"block {b}")
    ax2.axvline(K_HEADLINE, color="grey", lw=0.9, ls=":")
    ax2.axhline(0.5, color="black", lw=0.7, ls="--")
    ax2.axhline(0.1, color="black", lw=0.7, ls="--")
    ax2.set_xlabel("rollout step K")
    ax2.set_ylabel("between-context variance (fraction of K=0)")
    ax2.set_title("Between-context separation vs rollout step")
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    savefig_paper(fig, "fp_convergence", fig_dir)
    plt.close(fig)

    # ── Figure 3: slow-subspace alignment vs null ────────────────────────────
    cut = "0.95"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # panel A: aggregate between-context variance fraction in S_c vs null (prompt-end + K32)
    xs = np.arange(len(blocks))
    pe_frac, pe_lo, pe_hi, k32_frac = [], [], [], []
    for b in blocks:
        agg = per[str(b)]["slow_subspace"][cut]["aggregate_between_ctx_var"]
        pe_frac.append(agg["promptend"]["fraction"])
        pe_lo.append(agg["promptend"]["fraction"] - agg["promptend"]["null"]["p2.5"])
        pe_hi.append(agg["promptend"]["null"]["p97.5"] - agg["promptend"]["fraction"])
        k32_frac.append(agg["k32"]["fraction"])
    ax1.bar(xs - 0.2, pe_frac, width=0.38, color=paper_palette(2)[0], label="prompt-end states")
    ax1.bar(xs + 0.2, k32_frac, width=0.38, color=paper_palette(2)[1], label="K=32 rolled states")
    # null band markers for prompt-end
    for i, b in enumerate(blocks):
        agg = per[str(b)]["slow_subspace"][cut]["aggregate_between_ctx_var"]["promptend"]["null"]
        ax1.plot([xs[i] - 0.2, xs[i] - 0.2], [agg["p2.5"], agg["p97.5"]], color="black", lw=1.4)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(b) for b in blocks])
    ax1.set_xlabel("read-out block")
    ax1.set_ylabel(f"between-context variance in slow subspace (|lambda|>{cut})")
    ax1.set_title(
        "Context variance concentrated in the slow subspace\n(black bars: random-subspace null)"
    )
    ax1.legend(fontsize=7)
    # panel B: trait direction + drift projection energy vs null, at each trait's primary block.
    # Observed energies sit far ABOVE the (tiny) null band, so draw the null band as a shaded
    # span per bar (an errorbar with observed>>null gives negative yerr and crashes matplotlib).
    prim = {"evil": 20, "sycophancy": 26, "hallucination": 17}
    labels, vals, null_lo, null_hi = [], [], [], []
    for trait, b in prim.items():
        pe = per[str(b)]["slow_subspace"][cut]["trait_directions"][trait]
        labels.append(f"{trait}\n(block {b})")
        vals.append(pe["projection_energy"])
        null_lo.append(pe["null"]["p2.5"])
        null_hi.append(pe["null"]["p97.5"])
    for key, lbl in (
        ("drift_a", "drift a\n(block 20)"),
        ("answer_minus_promptend", "answer drift\n(block 20)"),
    ):
        pe = per["20"]["slow_subspace"][cut][key]
        labels.append(lbl)
        vals.append(pe["projection_energy"])
        null_lo.append(pe["null"]["p2.5"])
        null_hi.append(pe["null"]["p97.5"])
    xb = np.arange(len(labels))
    ax2.bar(xb, vals, width=0.6, color=paper_palette(3)[2], label="observed projection energy")
    for i in range(len(labels)):  # null band as a shaded span across each bar's width
        ax2.fill_between(
            [xb[i] - 0.3, xb[i] + 0.3],
            null_lo[i],
            null_hi[i],
            color="black",
            alpha=0.35,
            label="random-subspace null (2.5-97.5%)" if i == 0 else None,
        )
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel(f"projection energy in slow subspace (|lambda|>{cut})")
    ax2.set_title("Direction alignment with the slow subspace")
    ax2.legend(fontsize=6.5)
    fig.tight_layout()
    savefig_paper(fig, "fp_slow_subspace_alignment", fig_dir)
    plt.close(fig)

    # ── Figure 4: fixed-point location ───────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    cos_keys = ["cos_mean_answer", "cos_mean_promptend", "cos_drift_a", "cos_answer_drift"]
    cos_labels = [
        "mean answer state",
        "mean prompt-end state",
        "drift a",
        "answer-minus-prompt drift",
    ]
    xs = np.arange(len(blocks))
    width = 0.8 / len(cos_keys)
    cpal = paper_palette(len(cos_keys))
    for ki, (k, lbl) in enumerate(zip(cos_keys, cos_labels, strict=True)):
        vals = [per[str(b)]["fixed_point"][k] for b in blocks]
        ax1.bar(
            xs + (ki - (len(cos_keys) - 1) / 2) * width,
            vals,
            width=width,
            color=cpal[ki],
            label=lbl,
        )
    ax1.axhline(0.0, color="black", lw=0.7)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(b) for b in blocks])
    ax1.set_xlabel("read-out block")
    ax1.set_ylabel("cosine(h*, .)")
    ax1.set_title("Where the fixed point points")
    ax1.legend(fontsize=6.5)
    pcts = [per[str(b)]["fixed_point"]["norm_percentile_in_store"] for b in blocks]
    znorm = [per[str(b)]["fixed_point"]["zdist_to_mean_answer"] for b in blocks]
    ax2b = ax2.twinx()
    ax2.bar(xs - 0.2, pcts, width=0.38, color=paper_palette(2)[0], label="‖h*‖ percentile in store")
    ax2b.bar(
        xs + 0.2,
        znorm,
        width=0.38,
        color=paper_palette(2)[1],
        label="std-normalized dist to mean answer",
    )
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(b) for b in blocks])
    ax2.set_xlabel("read-out block")
    ax2.set_ylabel("‖h*‖ percentile within store norms")
    ax2b.set_ylabel("per-dim-std distance to mean answer state")
    ax2.set_title("Fixed-point magnitude and offset")
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    savefig_paper(fig, "fp_fixedpoint_location", fig_dir)
    plt.close(fig)


def main() -> int:  # noqa: C901 — the per-block analysis sequence IS the spec
    ap = argparse.ArgumentParser(description="Issue #922 fixed point + slow modes.")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_922/fixed_point_slow_modes.json",
    )
    ap.add_argument(
        "--out-npz",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_922/fixed_point_slow_modes_topvecs.npz",
    )
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures/issue_922")
    ap.add_argument("--n-top-vecs", type=int, default=64)
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument(
        "--figures-only",
        action="store_true",
        help="regenerate figures from the existing out-json (no recompute)",
    )
    args = ap.parse_args()

    torch.set_num_threads(8)
    blocks = list(C.READOUT_BLOCKS)  # [14,17,19,20,24,26]
    if args.figures_only:
        results = json.loads(args.out_json.read_text())
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        make_figures(results, blocks, args.fig_dir)
        logger.info("[figures-only] regenerated figures in %s", args.fig_dir)
        return 0

    rng = np.random.default_rng(0)
    H = C.EXPECTED_HIDDEN
    rows = [C.block_to_row(b) for b in blocks]

    maps_p = _fetch_input("maps")
    store_p = _fetch_input("store")
    logger.info("[inputs] maps=%s", maps_p)
    logger.info("[inputs] store=%s", store_p)
    regime = {
        "maps_sha": C.sha256_path(maps_p),
        "store_sha": C.sha256_path(store_p),
        "blocks": blocks,
        "roll_k": ROLL_K,
        "cutoffs": list(CUTOFFS),
    }

    # mmap-load the 3.2 GB maps blob, extract ONLY the 6 ctx + 6 tok read-out
    # states we need into a compact fp16 dict, then drop the blob — keeps the
    # persistent footprint ~0.5 GB (the earlyoom victim at 11 GB was the whole
    # blob held through the eig loop).
    _mblob = torch.load(maps_p, weights_only=False, mmap=True)
    ctx_rows_present = sorted(_mblob["answer_lstar"]["ctx"].keys())
    logger.info("[maps] ctx rows present: %s (want %s)", ctx_rows_present, rows)
    maps_states: dict = {"ctx": {}, "tok": {}}
    for r in rows:
        for arm in ("ctx", "tok"):
            maps_states[arm][r] = {
                k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in _mblob["answer_lstar"][arm][r].items()
            }
    del _mblob
    gc.collect()
    fitted_layers_note = (
        "answer_lstar['ctx'] holds only the 6 read-out blocks {14,17,19,20,24,26}; "
        "the fit used 9 layers {0,5,10,14,17,19,20,24,26} but blocks 0,5,10 are NOT "
        "in the uploaded fp16 map subset (plots.py keeps READOUT_BLOCKS only) — "
        "SCOPE: analysis covers the 6 available read-out blocks."
    )

    # trait directions r_B (block-indexed, (28, H)) — fetched into the non-reapable cache
    rb = {t: load_rb_direct(f"rb_{t}") for t in C.TRAITS}

    logger.info("[store] loading test-context states ...")
    t0 = time.time()
    ts = load_test_states(store_p, rows, H)
    logger.info("[store] loaded %d contexts in %.1fs", ts["n_test_contexts"], time.time() - t0)

    results: dict = {
        "per_block": {},
        "token_informed_comparison": {},
        "regime": regime,
        "notes": [
            fitted_layers_note,
            "All states are teacher-forced captures of the model's OWN on-policy completions "
            "(provenance inherited from #922's lmsys_test store).",
            "ridge_ctx_answer is the GLOBAL context-only closed-form next-token affine map; "
            "one-step h_{t+1}=A h+a with A=I+w^T diag(1/sd), a=bias-w^T(mu/sd), sigma==1.",
            "Per-vector null band = exact random-subspace projection-energy distribution "
            "(Beta(dim/2,(H-dim)/2)) sampled via chi-square ratios (rotational invariance); "
            f"aggregate between-context-variance null = {N_NULL_AGG} sampled random "
            "orthonormal subspaces.",
        ],
    }
    # per-block checkpoint dir (resume after an earlyoom kill mid-loop) —
    # under the non-reapable cache, regime-guarded.
    ckpt_dir = CACHE_DIR / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    regime_p = ckpt_dir / "regime.json"
    if regime_p.exists() and json.loads(regime_p.read_text()) != regime:
        logger.info("[ckpt] regime mismatch — clearing stale block checkpoints")
        for old in ckpt_dir.glob("block_*.json"):
            old.unlink()
        for old in ckpt_dir.glob("topvecs_block_*.npz"):
            old.unlink()
    regime_p.write_text(json.dumps(regime))

    for b, r in zip(blocks, rows, strict=True):
        block_ckpt = ckpt_dir / f"block_{b}.json"
        tv_ckpt = ckpt_dir / f"topvecs_block_{b}.npz"
        if block_ckpt.exists() and tv_ckpt.exists():
            bj = json.loads(block_ckpt.read_text())
            results["per_block"][str(b)] = bj["per_block"]
            results["token_informed_comparison"][str(b)] = bj["token_informed"]
            logger.info("[block %d] resumed from checkpoint (skip compute)", b)
            continue
        logger.info("[block %d] (row %d) building affine map ...", b, r)
        st = maps_states["ctx"][r]
        A, a = affine_from_ridge_state(st, H)

        # ── spectrum (full complex eig) ──────────────────────────────────────
        t1 = time.time()
        L, V = torch.linalg.eig(A)
        eig_s = time.time() - t1
        mod = L.abs()
        mod_sorted, order = torch.sort(mod, descending=True)
        rho = float(mod_sorted[0])
        smax = sigma_max(A)
        logger.info("[block %d] rho=%.6f sigma_max=%.4f eig=%.1fs", b, rho, smax, eig_s)
        assert rho < 1.0, f"block {b} not a contraction: rho={rho}"

        counts = {f"{c:.2f}": int((mod > c).sum()) for c in CUTOFFS}
        # top slow modes: time constants + rotation
        n_report = 20
        top_modes = []
        for j in range(n_report):
            idx = int(order[j])
            lam = L[idx]
            m = float(lam.abs())
            tau = float(-1.0 / np.log(m)) if 0.0 < m < 1.0 else None
            im = float(lam.imag)
            is_real = abs(im) <= IMAG_TOL
            arg = float(np.abs(np.angle(complex(float(lam.real), im))))
            rot = float(2 * np.pi / arg) if (not is_real and arg > 0) else None
            top_modes.append(
                {
                    "modulus": m,
                    "real": float(lam.real),
                    "imag": im,
                    "tau_steps": tau,
                    "is_real": bool(is_real),
                    "rotation_period_steps": rot,
                }
            )

        # ── fixed point ──────────────────────────────────────────────────────
        eye = torch.eye(H, dtype=torch.float64)
        h_star = torch.linalg.solve(eye - A, a)  # (H,)
        hs_norm = float(h_star.norm())
        row = ts["rows"][r]
        mean_ans = row["mean_answer"]
        mean_pe = row["mean_promptend"]
        drift_ap = mean_ans - mean_pe
        all_norms = row["all_norms"]
        pct = float((all_norms < hs_norm).mean() * 100.0)

        def _cos(u, v):
            un, vn = float(u.norm()), float(v.norm())
            if un == 0 or vn == 0:
                return 0.0
            return float((u * v).sum() / (un * vn))

        # per-dim std over the store pool at this row (from promptend + answer states proxy):
        # use the promptend + all-position norms are norms; for per-dim std use promptend states
        pe_states = row["promptend"]  # (n_ctx, H)
        per_dim_std = pe_states.std(0, unbiased=True) + 1e-9
        zdist = float((((h_star - mean_ans) / per_dim_std) ** 2).mean().sqrt())

        fixed_point = {
            "norm": hs_norm,
            "norm_percentile_in_store": pct,
            "store_norm_median": float(np.median(all_norms)),
            "store_norm_mean": float(all_norms.mean()),
            "cos_mean_answer": _cos(h_star, mean_ans),
            "cos_mean_promptend": _cos(h_star, mean_pe),
            "cos_drift_a": _cos(h_star, a),
            "cos_answer_drift": _cos(h_star, drift_ap),
            "raw_dist_to_mean_answer": float((h_star - mean_ans).norm()),
            "zdist_to_mean_answer": zdist,
            "mean_answer_norm": float(mean_ans.norm()),
            "mean_promptend_norm": float(mean_pe.norm()),
            "n_answer_positions": row["n_answer_pos"],
        }

        # ── convergence roll ─────────────────────────────────────────────────
        med_dist, trace_cov, snap_k32 = roll_and_summarize(A, a, pe_states, h_star)
        ef_dist = efolding_from_curve(med_dist)
        tc0 = trace_cov[0]
        tc_frac = trace_cov / tc0
        half_k = np.nonzero(tc_frac <= 0.5)[0]
        ten_k = np.nonzero(tc_frac <= 0.1)[0]
        convergence = {
            "median_dist_to_hstar": [float(x) for x in med_dist],
            "between_ctx_var": [float(x) for x in trace_cov],
            "theoretical_efold_from_rho": float(-1.0 / np.log(rho)),
            "empirical_efold_dist": ef_dist,
            "betweenctx_var_half_k": int(half_k[0]) if len(half_k) else None,
            "betweenctx_var_10pct_k": int(ten_k[0]) if len(ten_k) else None,
            "betweenctx_var_frac_at_k32": float(tc_frac[K_HEADLINE]),
            "median_dist_frac_at_k32": float(med_dist[K_HEADLINE] / med_dist[0]),
        }

        # ── slow subspaces + projections ─────────────────────────────────────
        # top-10 between-context PCs of prompt-end states
        Xc_pe = pe_states - mean_pe  # (n_ctx, H)
        # right singular vectors = PCs
        _, _, Vt_pe = torch.linalg.svd(Xc_pe, full_matrices=False)
        pcs = Vt_pe[:10]  # (10, H)
        Xc_k32 = snap_k32 - snap_k32.mean(0)

        slow_subspace: dict = {}
        for c in CUTOFFS:
            ck = f"{c:.2f}"
            S, dim, n_real, n_cpx = build_slow_subspace(L, V, c)
            entry: dict = {"dim": dim, "n_real_modes": n_real, "n_complex_pairs": n_cpx}
            # trait directions
            entry["trait_directions"] = {}
            for t in C.TRAITS:
                v = torch.from_numpy(rb[t][b]).to(torch.float64)
                entry["trait_directions"][t] = {
                    "projection_energy": proj_energy(S, v),
                    "null": vector_null_band(dim, H, N_NULL_VEC, rng),
                }
            # top-10 context PCs
            entry["context_pcs"] = []
            for pi in range(pcs.shape[0]):
                entry["context_pcs"].append(
                    {
                        "pc": pi,
                        "projection_energy": proj_energy(S, pcs[pi].to(torch.float64)),
                        "null": vector_null_band(dim, H, N_NULL_VEC, rng),
                    }
                )
            # drift a + answer-minus-promptend
            entry["drift_a"] = {
                "projection_energy": proj_energy(S, a),
                "null": vector_null_band(dim, H, N_NULL_VEC, rng),
            }
            entry["answer_minus_promptend"] = {
                "projection_energy": proj_energy(S, drift_ap),
                "null": vector_null_band(dim, H, N_NULL_VEC, rng),
            }
            # aggregate between-context variance fraction (prompt-end + K32)
            entry["aggregate_between_ctx_var"] = {
                "promptend": {
                    "fraction": aggregate_variance_fraction(Xc_pe, S),
                    "null": aggregate_variance_null(Xc_pe, dim, H, N_NULL_AGG, rng),
                },
                "k32": {
                    "fraction": aggregate_variance_fraction(Xc_k32, S),
                    "null": aggregate_variance_null(Xc_k32, dim, H, N_NULL_AGG, rng),
                },
            }
            slow_subspace[ck] = entry

        results["per_block"][str(b)] = {
            "row": r,
            "spectral_radius": rho,
            "sigma_max": smax,
            "nonnormality_ratio": smax / rho,
            "eig_seconds": round(eig_s, 1),
            "slow_modes": {
                "counts": counts,
                "top100_moduli": [float(x) for x in mod_sorted[:100]],
                "top_modes": top_modes,
            },
            "fixed_point": fixed_point,
            "convergence": convergence,
            "slow_subspace": slow_subspace,
        }

        # ── token-informed comparison (rho + counts only; no fixed point) ─────
        st_tok = maps_states["tok"][r]
        A_tok, _ = affine_from_ridge_state(st_tok, H)
        Ltok = torch.linalg.eigvals(A_tok)
        mtok = Ltok.abs()
        results["token_informed_comparison"][str(b)] = {
            "spectral_radius": float(mtok.max()),
            "counts": {f"{c:.2f}": int((mtok > c).sum()) for c in CUTOFFS},
        }

        # ── top-N slow eigvecs → per-block checkpoint npz ─────────────────────
        n_top = min(args.n_top_vecs, H)
        idx_top = order[:n_top]
        np.savez_compressed(
            tv_ckpt,
            **{
                f"block{b}_eigvals": L[idx_top].to(torch.complex128).numpy(),
                f"block{b}_eigvecs": V[:, idx_top].to(torch.complex128).numpy(),
                f"block{b}_h_star": h_star.numpy(),
            },
        )
        # ── per-block result checkpoint (resume after a mid-loop kill) ────────
        C.write_json_atomic(
            block_ckpt,
            {
                "per_block": results["per_block"][str(b)],
                "token_informed": results["token_informed_comparison"][str(b)],
            },
        )
        logger.info("[block %d] checkpointed", b)
        del L, V, A, A_tok
        gc.collect()

    # assemble the combined top-vec NPZ from the per-block checkpoints
    topvecs: dict = {}
    for b in blocks:
        with np.load(ckpt_dir / f"topvecs_block_{b}.npz") as z:
            for k in z.files:
                topvecs[k] = z[k]

    results["metadata"] = C.reproducibility_metadata(
        {"script": "issue922_fixed_point_slow_modes", "kind": "free_analysis"}
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(args.out_json, results)
    logger.info("[out] wrote %s", args.out_json)
    np.savez_compressed(args.out_npz, **topvecs)
    logger.info("[out] wrote %s", args.out_npz)

    if not args.skip_figures:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        make_figures(results, blocks, args.fig_dir)
        logger.info("[out] wrote figures to %s", args.fig_dir)
    logger.info("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
