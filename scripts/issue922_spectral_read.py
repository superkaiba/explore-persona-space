#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #922 free-analysis follow-up: spectral read of the fitted operators.

For each per-layer one-step map ``h → h + Δ̂(h)`` the state-transition Jacobian
is ``A = I + Mᵀ diag(1/sd_h)`` where ``M`` is the (row-convention) linear part
acting on the STANDARDIZED state (``Δ̂ = bias + ((h − mu)/sd) @ M``). This
script computes the eigenvalue spectrum of ``A`` (``torch.linalg.eigvals``,
fp16 weights promoted to fp64), the spectral radius ρ(A), the counts of
|λ| > 1 / |λ| > 0.99, and the top-10 moduli, for:

- the global answer-segment ridge (ctx arm; the headline rolled map) at the 6
  read-out blocks;
- the token-informed answer ridge (h-block of the [h, e] design — the rollout
  injects TRUE e each step, so the autonomous-in-h Jacobian is the h-block);
- the closed-form b1 [h, c] ridge (h-block; c is FROZEN per rollout, so the
  h-block Jacobian is exact);
- the four gradient-fit conditioned forms (b1_grad / film / lowrank /
  mixture), whose effective single-step operator ``A(c)`` is evaluated at the
  MEAN standardized context vector of the 500 test contexts (c = h_{l,T} from
  the uploaded test-context store; b1_grad's Jacobian is context-independent
  by construction), plus per-context sample reads at the primary ℓ* blocks;
- the boundary ctx ridge (applied ONCE at k=1 — reported for completeness,
  flagged non-iterated).

The test (follow-up brief): the k=32 rollout plateau of the closed-form rolls
is consistent with contraction (ρ < 1); the gradient-fit conditioned rolls'
explosive divergence predicts ρ(A_conditioned) > 1. Verdict fields record
which hold per map. Not in scope (recorded as notes): the emb answer arm (no
h-dependence — Jacobian ≡ I), the direct per-horizon maps (never iterated),
and the MLP/GRU maps (nonlinear — no single linear operator).
"""

from __future__ import annotations

import argparse
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
logger = logging.getLogger("issue922_spectral_read")

HF_DL = PROJECT_ROOT / "data" / "issue_922" / "hf_dl"
CLOSED_FORM_FAMILIES = ("ridge_ctx_answer", "ridge_tok_answer_hblock", "b1_ridge_answer_hblock")
GRADIENT_FAMILIES = ("b1_grad", "film", "lowrank", "mixture")
# Rollout-skill variant name per spectral family (ties spectra to the k=32 facts).
FAMILY_TO_VARIANT = {
    "ridge_ctx_answer": "ridge_ctx_boundary_first",
    "ridge_tok_answer_hblock": "tok_ceiling",
    "b1_ridge_answer_hblock": "b1_ridge_roll",
    "b1_grad": "b1_grad_roll",
    "film": "film_roll",
    "lowrank": "lowrank_roll",
    "mixture": "mixture_roll",
}


def _find_cached(name: str) -> Path:
    """Locate a downloaded artifact by filename under the hf_dl cache (fail-loud)."""
    hits = sorted(HF_DL.rglob(name))
    assert hits, f"{name} not found under {HF_DL} — download it first (see module docstring)"
    return hits[0]


def _jacobian_from_ridge_state(state: dict, h_dim: int) -> torch.Tensor:
    """A = I + Wᵀ diag(1/sd) for one saved RidgeMap state (h-block if d > H).

    The saved map predicts the raw update ``Δ̂ = bias + ((x − mu)/sd) @ w``
    (``sigma == 1`` asserted — the rollout composes raw Δ̂), so the one-step
    Jacobian wrt h is ``I + w_hᵀ diag(1/sd_h)`` with ``w_h = w[:H]``.
    """
    assert float(state["sigma"]) == 1.0, state["sigma"]
    w = state["w"].to(torch.float64)
    sd = state["sd"].to(torch.float64)
    assert w.shape[1] == h_dim and w.shape[0] in (h_dim, 2 * h_dim), w.shape
    w_h, sd_h = w[:h_dim, :], sd[:h_dim]
    return torch.eye(h_dim, dtype=torch.float64) + (w_h.t() / sd_h.unsqueeze(0))


def _conditioned_M(form: str, w: dict, cn: torch.Tensor, d: int) -> torch.Tensor:
    """Effective row-convention linear part M(c) of Δ̂ = hn @ M + const, fp64.

    Mirrors ``maps922.apply_conditioned_delta``'s h-dependence exactly:
    b1_grad → Wh (context-independent); film → A + diag(cn @ Wg);
    lowrank → A + (V·diag(s))Uᵀ with s = cn @ Ws; mixture → Σ αₘ Aₘ with
    α = softmax(cn @ Ww).
    """
    if form == "b1_grad":
        return w["Wh"]
    if form == "film":
        g = cn @ w["Wg"]  # (d,)
        return w["A"] + torch.diag(g)
    if form == "lowrank":
        s = cn @ w["Ws"]  # (r,)
        return w["A"] + (w["V"] * s.unsqueeze(0)) @ w["U"].t()
    if form == "mixture":
        alpha = torch.softmax(cn @ w["Ww"], dim=-1)  # (K,)
        out = torch.zeros(d, d, dtype=torch.float64)
        m = 0
        while f"Am{m}" in w:
            out = out + alpha[m] * w[f"Am{m}"]
            m += 1
        return out
    raise ValueError(form)


def _spectrum_entry(A: torch.Tensor, map_id: str, extra: dict) -> dict:
    """eigvals(A) → {spectral_radius, counts, top10 moduli, eig_seconds, ...}."""
    t0 = time.time()
    ev = torch.linalg.eigvals(A)
    secs = time.time() - t0
    mod = ev.abs()
    mod_sorted, _ = torch.sort(mod, descending=True)
    entry = {
        "map_id": map_id,
        "d": int(A.shape[0]),
        "spectral_radius": float(mod_sorted[0]),
        "n_gt_1": int((mod > 1.0).sum()),
        "n_gt_0p99": int((mod > 0.99).sum()),
        "top10_moduli": [float(x) for x in mod_sorted[:10]],
        "top200_moduli": [float(x) for x in mod_sorted[:200]],
        "eig_seconds": round(secs, 2),
        "eig_dtype": "float64",
        **extra,
    }
    logger.info(
        "[eig] %s: rho=%.4f n>|1|=%d n>|0.99|=%d (%.1fs)",
        map_id,
        entry["spectral_radius"],
        entry["n_gt_1"],
        entry["n_gt_0p99"],
        secs,
    )
    return entry


def _load_test_context_vectors(store_p: Path, rows: list[int]) -> tuple[dict, dict]:
    """(c_by_row, info): per store row a (n_ctx, H) fp32 stack of c = h_{l,T}.

    T's in-window index is ``prompt_len − 1 − window_start`` (the capture
    convention, ``issue922_capture_positions.py:301``). Fail-loud on any
    out-of-window T.
    """
    blob = torch.load(store_p, weights_only=False)
    assert blob.get("corpus") == "lmsys_test", blob.get("corpus")
    ctxs = blob["contexts"]
    per_row: dict[int, list[torch.Tensor]] = {r: [] for r in rows}
    for ci in sorted(ctxs):
        rec = ctxs[ci]
        t_row = int(rec["prompt_len"]) - 1 - int(rec["window_start"])
        npos = int(rec["h"].shape[0])
        assert 0 <= t_row < npos, (ci, t_row, npos)
        for r in rows:
            per_row[r].append(rec["h"][t_row, r, :].to(torch.float32))
    n_ctx = len(ctxs)
    c_by_row = {r: torch.stack(v) for r, v in per_row.items()}
    del blob, ctxs
    return c_by_row, {"n_test_contexts": n_ctx, "t_row_convention": "prompt_len-1-window_start"}


def _rollout_reference(blocks: list[int]) -> dict:
    """k=8 / k=32 mean rollout skill per family at the read-out blocks (committed JSON)."""
    p = PROJECT_ROOT / "eval_results" / "issue_922" / "rollout_skill.json"
    with open(p) as f:
        rs = json.load(f)
    out: dict = {"source": "eval_results/issue_922/rollout_skill.json"}
    for fam, variant in FAMILY_TO_VARIANT.items():
        sk = rs["variants"].get(variant, {}).get("skill_mean_ci", {})
        out[fam] = {
            "variant": variant,
            "skill_k8": {str(b): sk[str(b)][7]["mean"] for b in blocks if str(b) in sk},
            "skill_k32": {str(b): sk[str(b)][31]["mean"] for b in blocks if str(b) in sk},
        }
    return out


def make_figure(results: dict, blocks: list[int], fig_dir: Path) -> None:
    """One 2-panel figure: ρ per map (log y, ρ=1 line) + top-200 moduli at block 20."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    families = list(CLOSED_FORM_FAMILIES) + list(GRADIENT_FAMILIES)
    labels = {
        "ridge_ctx_answer": "global affine (context-only)",
        "ridge_tok_answer_hblock": "token-informed (state block)",
        "b1_ridge_answer_hblock": "additive closed-form (state block)",
        "b1_grad": "additive gradient twin",
        "film": "diagonal-gate (FiLM)",
        "lowrank": "low-rank delta",
        "mixture": "2-operator mixture",
    }
    colors = dict(zip(families, paper_palette(len(families)), strict=True))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    width = 0.8 / len(families)
    xs = np.arange(len(blocks))
    for fi, fam in enumerate(families):
        rhos = [results["maps"].get(f"{fam}/{b}", {}).get("spectral_radius") for b in blocks]
        pos = xs + (fi - (len(families) - 1) / 2) * width
        ax1.bar(
            pos,
            [r if r is not None else np.nan for r in rhos],
            width=width,
            color=colors[fam],
            label=labels[fam],
        )
    ax1.axhline(1.0, color="black", lw=0.9, ls="--")
    ax1.set_yscale("log")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(b) for b in blocks])
    ax1.set_xlabel("read-out block")
    ax1.set_ylabel("spectral radius of the one-step operator")
    ax1.set_title("Spectral radius per map (dashed: instability threshold 1)")
    ax1.legend(fontsize=6.5, ncol=2)
    ref_block = 20
    for fam in families:
        e = results["maps"].get(f"{fam}/{ref_block}")
        if e is None:
            continue
        top = e["top200_moduli"]
        ax2.plot(np.arange(1, len(top) + 1), top, color=colors[fam], lw=1.4, label=labels[fam])
    ax2.axhline(1.0, color="black", lw=0.9, ls="--")
    ax2.set_xlabel("eigenvalue rank (by modulus)")
    ax2.set_ylabel("eigenvalue modulus")
    ax2.set_title(f"Top-200 eigenvalue moduli at block {ref_block}")
    ax2.legend(fontsize=6.5)
    fig.tight_layout()
    savefig_paper(fig, "spectral_read", fig_dir)
    plt.close(fig)


def main() -> int:  # noqa: C901 — the map-family enumeration IS the analysis spec
    ap = argparse.ArgumentParser(description="Issue #922 spectral read of fitted operators.")
    ap.add_argument(
        "--out-json", type=Path, default=PROJECT_ROOT / "eval_results/issue_922/spectral_read.json"
    )
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures/issue_922")
    ap.add_argument(
        "--ckpt", type=Path, default=PROJECT_ROOT / "data/issue_922/spectral_ckpt.jsonl"
    )
    ap.add_argument("--blocks", default="14,17,19,20,24,26", help="read-out blocks to analyze")
    ap.add_argument(
        "--n-sample-contexts",
        type=int,
        default=2,
        help="per c-dependent form, extra per-context reads at the primary blocks (0 = off)",
    )
    ap.add_argument("--skip-figure", action="store_true")
    args = ap.parse_args()
    blocks = [int(b) for b in args.blocks.split(",") if b]
    H = C.EXPECTED_HIDDEN

    maps_p = _find_cached("maps_boundary_and_lstar_fp16.pt")
    cond_p = _find_cached("maps_conditioned_lstar_fp16.pt")
    store_p = _find_cached("store_test_contexts.pt")
    logger.info("[inputs] maps=%s cond=%s store=%s", maps_p, cond_p, store_p)

    # resume checkpoint: regime = input digests + eval-point tag
    regime = {
        "maps_sha": C.sha256_path(maps_p),
        "cond_sha": C.sha256_path(cond_p),
        "store_sha": C.sha256_path(store_p),
        "eval_point": "mean_test_context",
        "eig_dtype": "float64",
    }
    done: dict[str, dict] = {}
    if args.ckpt.exists():
        rows = [json.loads(ln) for ln in args.ckpt.read_text().splitlines() if ln.strip()]
        if rows and rows[0].get("regime") == regime:
            done = {r["map_id"]: r for r in rows[1:]}
            logger.info("[resume] %d maps already computed", len(done))
        else:
            logger.info("[resume] regime mismatch — starting fresh")
            args.ckpt.unlink()
    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    if not args.ckpt.exists():
        with open(args.ckpt, "w") as f:
            f.write(json.dumps({"regime": regime}) + "\n")

    def _ckpt(entry: dict) -> None:
        with open(args.ckpt, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _run(map_id: str, build_A) -> dict:
        if map_id in done:
            return done[map_id]
        e = _spectrum_entry(build_A(), map_id, {})
        _ckpt(e)
        done[map_id] = e
        return e

    results: dict = {"maps": {}, "notes": [], "context_dependence": {}}

    # ── global ridge maps (answer ctx / tok h-block / b1 h-block / boundary ctx)
    blob = torch.load(maps_p, weights_only=False)
    row_of = {b: C.block_to_row(b) for b in blocks}
    for b in blocks:
        r = row_of[b]
        st = blob["answer_lstar"]["ctx"][r]
        results["maps"][f"ridge_ctx_answer/{b}"] = _run(
            f"ridge_ctx_answer/{b}", lambda st=st: _jacobian_from_ridge_state(st, H)
        )
        st_tok = blob["answer_lstar"]["tok"][r]
        results["maps"][f"ridge_tok_answer_hblock/{b}"] = _run(
            f"ridge_tok_answer_hblock/{b}", lambda st=st_tok: _jacobian_from_ridge_state(st, H)
        )
        if "b1_answer_lstar" in blob and r in blob["b1_answer_lstar"]:
            st_b1 = blob["b1_answer_lstar"][r]
            results["maps"][f"b1_ridge_answer_hblock/{b}"] = _run(
                f"b1_ridge_answer_hblock/{b}", lambda st=st_b1: _jacobian_from_ridge_state(st, H)
            )
        st_bd = blob["boundary"]["ctx"][r]
        results["maps"][f"boundary_ctx/{b}"] = _run(
            f"boundary_ctx/{b}", lambda st=st_bd: _jacobian_from_ridge_state(st, H)
        )
    del blob
    results["notes"].append(
        "boundary_ctx maps are applied ONCE (k=1 seed step) and never iterated; their spectra "
        "are reported for completeness but do not govern rollout asymptotics."
    )
    results["notes"].append(
        "the emb answer arm has no dependence on the current state h (Jacobian == I); the "
        "direct per-horizon maps (arm c) are never iterated; the MLP/GRU maps are nonlinear — "
        "no single linear one-step operator exists for these, so they are out of scope."
    )
    results["notes"].append(
        "tok/b1 h-block Jacobians are exact for the rollout dynamics: the token-informed roll "
        "injects TRUE next-token embeddings and the b1 roll freezes c = seed state, so both "
        "auxiliary inputs are exogenous forcing terms, not state."
    )

    # ── conditioned gradient forms at the mean test context ─────────────────────
    cblob = torch.load(cond_p, weights_only=False)
    cond_rows_by_block = {
        b: row_of[b] for b in blocks if row_of[b] in cblob["forms"][GRADIENT_FAMILIES[0]]
    }
    need_rows = sorted(set(cond_rows_by_block.values()))
    c_by_row, store_info = _load_test_context_vectors(store_p, need_rows)
    results["store_info"] = store_info

    primary_blocks = [b for b in (20, 26, 17) if b in cond_rows_by_block]
    rng = np.random.default_rng(0)
    n_ctx = store_info["n_test_contexts"]
    sample_ix = sorted(rng.choice(n_ctx, size=min(args.n_sample_contexts, n_ctx), replace=False))

    for form in GRADIENT_FAMILIES:
        per_row = cblob["forms"][form]
        gate_stats: dict = {}
        for b, r in cond_rows_by_block.items():
            pb = per_row[r]
            w = {k: v.to(torch.float64) for k, v in pb["weights"].items()}
            mu_c = pb["mu_c"].to(torch.float64)
            sd_c = pb["sd_c"].to(torch.float64)
            sd_h = pb["sd_h"].to(torch.float64)
            cn_all = (c_by_row[r].to(torch.float64) - mu_c) / sd_c  # (n_ctx, d)
            cn_mean = cn_all.mean(0)

            def _A(cn, form=form, w=w, sd_h=sd_h):
                M = _conditioned_M(form, w, cn, H)
                return torch.eye(H, dtype=torch.float64) + (M.t() / sd_h.unsqueeze(0))

            e = _run(f"{form}/{b}", lambda cn=cn_mean: _A(cn))
            e["eval_point"] = "context_independent" if form == "b1_grad" else "mean_test_context"
            results["maps"][f"{form}/{b}"] = e
            # per-context sample reads at the primary blocks (c-dependent forms only)
            if form != "b1_grad" and b in primary_blocks:
                for si in sample_ix:
                    sid = f"{form}/{b}/ctx{si}"
                    es = _run(sid, lambda cn=cn_all[si]: _A(cn))
                    es["eval_point"] = f"test_context_{si}"
                    results["maps"][sid] = es
            # cheap context-dependence stats (no eig)
            if form == "film":
                g = cn_all @ w["Wg"]
                gate_stats[str(b)] = {
                    "gate_abs_mean": float(g.abs().mean()),
                    "gate_rms_per_ctx_mean": float((g * g).mean(1).sqrt().mean()),
                    "gate_rms_per_ctx_std": float((g * g).mean(1).sqrt().std()),
                }
            elif form == "lowrank":
                s = cn_all @ w["Ws"]
                gate_stats[str(b)] = {
                    "s_rms_per_ctx_mean": float((s * s).mean(1).sqrt().mean()),
                    "s_rms_per_ctx_std": float((s * s).mean(1).sqrt().std()),
                }
            elif form == "mixture":
                alpha = torch.softmax(cn_all @ w["Ww"], dim=-1)
                gate_stats[str(b)] = {
                    "alpha_mean": [float(x) for x in alpha.mean(0)],
                    "alpha_std": [float(x) for x in alpha.std(0)],
                    "alpha_min": [float(x) for x in alpha.min(0).values],
                    "alpha_max": [float(x) for x in alpha.max(0).values],
                }
        if gate_stats:
            results["context_dependence"][form] = gate_stats
    del cblob, c_by_row

    # ── verdicts (the follow-up's test) ──────────────────────────────────────────
    def _rho(fam: str, b: int) -> float | None:
        e = results["maps"].get(f"{fam}/{b}")
        return None if e is None else e["spectral_radius"]

    closed = {
        fam: {str(b): (_rho(fam, b) is not None and _rho(fam, b) < 1.0) for b in blocks}
        for fam in CLOSED_FORM_FAMILIES
    }
    grad = {
        fam: {
            str(b): (_rho(fam, b) is not None and _rho(fam, b) > 1.0)
            for b in blocks
            if f"{fam}/{b}" in results["maps"]
        }
        for fam in GRADIENT_FAMILIES
    }
    results["verdicts"] = {
        "closed_form_rho_lt_1": closed,
        "closed_form_all_contractive": all(v for d in closed.values() for v in d.values()),
        "gradient_forms_rho_gt_1_at_mean_context": grad,
        "gradient_forms_all_expansive": all(v for d in grad.values() for v in d.values()),
    }
    results["verdicts"]["contraction_gloss_consistent"] = (
        results["verdicts"]["closed_form_all_contractive"]
        and results["verdicts"]["gradient_forms_all_expansive"]
    )
    results["rollout_reference"] = _rollout_reference(blocks)
    results["regime"] = regime
    results["metadata"] = C.reproducibility_metadata(
        {"script": "issue922_spectral_read", "kind": "spectral_read"}
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(args.out_json, results)
    logger.info("[out] wrote %s (%d maps)", args.out_json, len(results["maps"]))

    if not args.skip_figure:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        make_figure(results, blocks, args.fig_dir)
        logger.info("[out] wrote %s/spectral_read.png", args.fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
