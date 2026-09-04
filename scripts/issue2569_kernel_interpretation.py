#!/usr/bin/env python3
"""Task #2569 leg 8 follow-on: interpret the effective kernel of the L19 context→answer map.

The map is the banked n1m ridge operator under the ROW action (``vhat = v @ A + b``,
raw residual coordinates, ``issue2569_operator.load_banked_map``). The effective
kernel at a squared-singular-mass cutoff ``m`` is the set of LEFT singular (read)
directions with ``sigma_i < tau``, where ``tau`` is the sigma at which cumulative
sigma^2 mass reaches ``m`` (``issue2569_operator.tau_kernel_threshold``; leg-1
convention, strict ``<``). Ridge has no exact kernel — every claim here is about a
LOW-GAIN READ subspace of the fitted linear map, never an exact null space and
never anything causal.

Reads implemented (plan: teammate task 2026-09-02):
  1. Feature kernel share for every decoder direction of two dictionaries
     (the #2569 context SAE trained on the map's own X19 input rows, and the
     andyrdt per-token L19 residual SAE), against a random-unit-direction null.
  2. Data-weighted kernel: tr(P_ker Sigma_c)/tr(Sigma_c) + top eigen-modes of the
     kernel-restricted and range-restricted context covariance, decoded through
     both dictionaries and through top/bottom-projecting real contexts.
  3. Kernel pairs in words: the 20 largest-distance mined kernel pairs (+ matched
     controls) joined to their context texts.
  4. Persona directions (r_B evil / sycophancy / hallucination; #2254 ctxext
     context-steering directions) as context-side directions.
  5. Cutoff sensitivity: every headline number at mass 0.999 / 0.99 / 0.90.

Coordinate frames (verified against producers, stated in the output JSON):
  - Both dictionaries decode in RAW residual coordinates. The context SAE was
    trained directly on the X19 fp16 rows cast to fp32 with NO standardization
    (``issue2569_rowbattery._run_sae_training`` feeds ``x_mm`` rows straight to
    ``train_step_losses``; b_dec initialized at the raw row mean). The andyrdt
    suite folds its normalization into the released weights and consumes raw
    residual activations (``issue1482_sae`` module docstring, r3 source read).
  - Sigma_c is the population covariance of RAW context rows over the map's
    963,444-row training pool: ``gram/n - mean mean^T`` from the moments
    ``gram_xx.pt`` (uncentered sum-of-outer Gram; ``issue2569_gateladder.
    load_sigma_file`` convention), symmetrized.

Outputs: eval_results/issue_2569/weights/leg8/kernel_interpretation_L19.{json,md}
+ figures/issue_2569/leg8_kernel_interpretation.{png,pdf}. The ``--render-only``
pass merges an interpretation overlay (feature/mode readings + narrative
paragraphs written by the analyst after inspecting the excerpts) into the JSON
and regenerates the markdown; the numeric analysis is untouched by it.

CPU only; ``torch.set_num_threads`` capped (shared VM); blocked GEMMs throughout
(never a dense n_feature x n_feature or 131k-row fp64 materialization).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for p in (str(_REPO / "src"), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps bind BEFORE torch/numpy heavy use (#847)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue2569_operator as OP  # noqa: E402

D = 3584
LAYER = 19
MASSES = (0.999, 0.99, 0.90)  # tighter / primary / looser
PRIMARY_MASS = 0.99
NULL_SEED = 2569
N_NULL = 10_000
TOP_LIST = 50
TOP_NAME = 20
TOP_MODES = 10
QUOTE_FEAT = 200
QUOTE_PAIR = 300
RB_TRAITS = ("evil", "sycophancy", "hallucination")
# Context-SAE features the map's top eigen read planes keep hitting in the
# eigen-dashboards v2 run (branch issue-2569-eigen-v2) — named the same way as
# the top-ignored/top-used lists (team-lead scope addition, 2026-09-02).
EIGEN_FEATS = (377, 638, 821, 960, 1354)

# ── pure helpers (unit-tested on d<=48 synthetics) ────────────────────────────────


def svd_row_action(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full fp64 SVD of the row-action operator: ``A = U diag(s) Vh``.

    READ (input) directions are the columns of ``U`` (``U[:, i] @ A = s[i] *
    Vh[i]``); WRITE directions are the rows of ``Vh``. ``s`` descending.
    """
    U, s, Vh = np.linalg.svd(np.asarray(A, dtype=np.float64))
    return U, s, Vh


def mass_partitions(s: np.ndarray, masses=MASSES) -> dict:
    """Per-cutoff kernel definition from the FULL spectrum.

    For each squared-mass cutoff m: ``tau`` and rank via
    ``OP.tau_kernel_threshold``; the kernel mask is ``sigma < tau`` (strict — the
    leg-1 ``effective_kernel_stats`` convention, so a boundary value stays OUT).
    Returns ``{mass: {"tau": float, "rank": int, "mask": (d,) bool}}``.
    """
    s = np.asarray(s, dtype=np.float64)
    out = {}
    for m in masses:
        tau, rank = OP.tau_kernel_threshold(s, mass=m)
        out[m] = {"tau": float(tau), "rank": int(rank), "mask": s < tau}
    return out


def shares_at_masks(
    U: np.ndarray, dirs: np.ndarray, masks: dict[float, np.ndarray], block: int = 4096
) -> dict[float, np.ndarray]:
    """Kernel share ``||P_ker d||^2`` for unit ROW directions ``dirs`` (n, d).

    ``P_ker`` projects onto the kernel read directions (columns of ``U`` selected
    by each mask). One blocked fp64 GEMM pass computes the squared projections
    onto ALL read directions; each cutoff is then a masked row-sum, so the three
    cutoffs share the pass. Inputs need not be exactly unit: shares are divided
    by each row's squared norm (guarded for zero rows, which return NaN).
    """
    U = np.asarray(U, dtype=np.float64)
    n = dirs.shape[0]
    out = {m: np.empty(n, dtype=np.float64) for m in masks}
    sq = np.empty(n, dtype=np.float64)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        db = np.asarray(dirs[lo:hi], dtype=np.float64)
        g = db @ U  # (b, d): g[r, i] = d_r . u_i
        q = g * g
        nrm2 = np.einsum("ij,ij->i", db, db)
        sq[lo:hi] = nrm2
        safe = np.where(nrm2 > 0, nrm2, np.nan)
        for m, mask in masks.items():
            out[m][lo:hi] = q[:, mask].sum(axis=1) / safe
    return out


def projected_cov_trace_fraction(U: np.ndarray, mask: np.ndarray, sigma_c: np.ndarray) -> float:
    """``tr(P Sigma_c) / tr(Sigma_c)`` for P = projector onto masked U columns.

    ``tr(P Sigma P) = tr(P Sigma)`` = sum over masked i of ``u_i^T Sigma u_i``.
    """
    Uk = np.asarray(U, dtype=np.float64)[:, np.asarray(mask, bool)]
    s64 = np.asarray(sigma_c, dtype=np.float64)
    num = float(np.einsum("di,de,ei->", Uk, s64, Uk))
    den = float(np.trace(s64))
    return num / den


def projected_cov_modes(
    U: np.ndarray, mask: np.ndarray, sigma_c: np.ndarray, top_k: int = TOP_MODES
) -> tuple[np.ndarray, np.ndarray]:
    """Top eigen-modes of ``P Sigma_c P`` in raw coordinates, via the subspace.

    Works in the masked-column basis (``B = Uk^T Sigma Uk``, m x m, then eigh),
    which is exact because ``P Sigma P`` vanishes off the subspace. Returns
    ``(eigvals desc (top_k,), modes (top_k, d) unit rows)``.
    """
    Uk = np.asarray(U, dtype=np.float64)[:, np.asarray(mask, bool)]
    B = Uk.T @ np.asarray(sigma_c, dtype=np.float64) @ Uk
    B = 0.5 * (B + B.T)
    w, Q = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:top_k]
    vals = w[order]
    modes = (Uk @ Q[:, order]).T
    modes = modes / np.linalg.norm(modes, axis=1, keepdims=True)
    return vals, modes


def null_share_stats(shares: np.ndarray) -> dict:
    """Summary of the random-direction null: mean/std + 2.5/50/97.5 percentiles."""
    sh = np.asarray(shares, dtype=np.float64)
    return {
        "n": int(sh.size),
        "mean": float(sh.mean()),
        "std": float(sh.std()),
        "p2p5": float(np.percentile(sh, 2.5)),
        "p50": float(np.percentile(sh, 50)),
        "p97p5": float(np.percentile(sh, 97.5)),
    }


_REDACT_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"(?i)(api[_ -]?key|token|secret|password)\s*[:=]\s*\S{8,}"),
    # generic key-shaped string: >=28 chars, letters+digits mixed, no spaces
    re.compile(r"\b(?=[A-Za-z0-9_\-]*[A-Za-z])(?=[A-Za-z0-9_\-]*[0-9])[A-Za-z0-9_\-]{28,}\b"),
]


def redact(text: str) -> str:
    """Replace credential-shaped substrings with [REDACTED] (quote hygiene).

    The earlier #779 round found one key-shaped string in the public source
    text, so quotes are scrubbed with specific token formats first, then a
    generic long-mixed-alphanumeric rule.
    """
    for pat in _REDACT_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def tail_quote(text: str, n_chars: int) -> str:
    """Last ``n_chars`` of a context text (the end = the last user message
    region in these single-string contexts), redacted, newlines flattened."""
    t = " ".join(str(text).split())
    t = t[-n_chars:] if len(t) > n_chars else t
    return redact(t)


# ── IO helpers (environment-specific) ─────────────────────────────────────────────


def _hf_local(manifest: dict, suffix: str) -> Path:
    hits = [v for k, v in manifest["paths"].items() if k.endswith(suffix)]
    assert len(hits) == 1, f"manifest lookup {suffix!r}: {len(hits)} hits"
    return Path(hits[0])


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def load_sigma_c(gram_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Sigma_c (population covariance, symmetrized) + mean + n from gram_xx.pt."""
    obj = torch.load(gram_path, map_location="cpu", weights_only=False)
    gram = np.asarray(obj["gram"], dtype=np.float64)
    mean = np.asarray(obj["mean"], dtype=np.float64)
    n = int(obj["n_rows"])
    sigma = gram / n - np.outer(mean, mean)
    return 0.5 * (sigma + sigma.T), mean, n


def load_ctx_sae(path: Path) -> dict:
    """Context-SAE bundle -> decoder rows + encode pieces (raw-frame, fp32).

    ``ae.pt`` = {cfg, state_dict} (``issue2569_rowbattery.load_sae_ctx``
    contract). Decoder direction for feature f = ``w_dec[f]`` (row, d=3584).
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    cfg, sd = obj["cfg"], obj["state_dict"]
    assert cfg["architecture"] == "matryoshka_batchtopk", cfg["architecture"]
    assert int(cfg["act_dim"]) == D and int(cfg["dict_size"]) == 65_536, cfg
    w_dec = sd["w_dec"].to(torch.float32).numpy()  # (n_feat, d)
    assert w_dec.shape == (65_536, D), w_dec.shape
    return {
        "w_dec": w_dec,
        "w_enc": sd["w_enc"].to(torch.float32).numpy(),  # (d, n_feat)
        "b_enc": sd["b_enc"].to(torch.float32).numpy(),
        "b_dec": sd["b_dec"].to(torch.float32).numpy(),
        "threshold": float(sd["threshold"]),
        "cfg": {k: cfg.get(k) for k in ("architecture", "dict_size", "k", "trained_on")},
    }


def load_andyrdt_decoder(path: Path) -> np.ndarray:
    """andyrdt trainer_1 (k=64) decoder directions as (n_feat, d) fp32 rows."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    w_dec = sd["decoder.weight"]
    assert tuple(w_dec.shape) == (D, 131_072), tuple(w_dec.shape)
    return w_dec.to(torch.float32).numpy().T.copy()  # (131072, 3584)


def load_capture_sample(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    """Stack the sampled capture chunks: (X (n, d) fp32 L19 cx_last, ci (n,))."""
    paths = sorted(
        (k, v) for k, v in manifest["paths"].items() if "final_token_capture" in k
    )
    xs, cis = [], []
    for _k, p in paths:
        b = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        layers = [int(x) for x in b["layers"]]
        col = layers.index(LAYER)
        xs.append(b["cx_last"][:, col, :].to(torch.float32).numpy())
        cis.append(np.asarray([int(c) for c in b["ci"]], dtype=np.int64))
    X = np.concatenate(xs, axis=0)
    ci = np.concatenate(cis)
    # dedupe by ci (keep first occurrence)
    _, first = np.unique(ci, return_index=True)
    keep = np.sort(first)
    return X[keep], ci[keep]


def load_row_meta_texts(manifest: dict, needed: set[int]) -> dict[int, dict]:
    """Scan the row_meta jsonl shards keeping only needed conversation indices."""
    out: dict[int, dict] = {}
    shards = sorted(v for k, v in manifest["paths"].items() if "/row_meta_" in k)
    for p in shards:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                ci = int(r["ci"])
                if ci in needed:
                    out[ci] = {
                        "corpus": r.get("corpus"),
                        "context_text": r.get("context_text", ""),
                    }
        if len(out) == len(needed):
            break
    return out


# ── analysis ─────────────────────────────────────────────────────────────────────


def encode_ctx_features(ctx: dict, X: np.ndarray, feat_ids: np.ndarray, block: int = 4096) -> np.ndarray:
    """Threshold-gated ctx-SAE activations for selected features over sample rows.

    ``act = relu((x - b_dec) @ w_enc[:, ids] + b_enc[ids])`` gated by the scalar
    inference threshold (matryoshka BatchTopK convention). Returns (n, n_sel).
    """
    w_enc_sel = ctx["w_enc"][:, feat_ids]
    b_enc_sel = ctx["b_enc"][feat_ids]
    acts = np.empty((X.shape[0], feat_ids.size), dtype=np.float32)
    for lo in range(0, X.shape[0], block):
        hi = min(lo + block, X.shape[0])
        pre = (X[lo:hi] - ctx["b_dec"]) @ w_enc_sel + b_enc_sel
        a = np.maximum(pre, 0.0)
        acts[lo:hi] = a * (a > ctx["threshold"])
    return acts


def naming_evidence_rows(
    feat_ids: list[int],
    acts_cols: np.ndarray,
    shares_primary: np.ndarray,
    ci: np.ndarray,
    texts: dict[int, dict],
) -> list[dict]:
    """Top-3 activating contexts per feature (redacted tail quotes) + kernel share."""
    rows = []
    for j, f in enumerate(feat_ids):
        col = acts_cols[:, j]
        top3 = np.argsort(col)[::-1][:3]
        below = bool(col[top3[0]] <= 0.0)
        ev = []
        for r in top3:
            c = int(ci[r])
            t = texts.get(c, {})
            ev.append(
                {
                    "ci": c,
                    "act": round(float(col[r]), 3),
                    "quote": tail_quote(t.get("context_text", "<text unavailable>"), QUOTE_FEAT),
                }
            )
        rows.append(
            {
                "feat_id": int(f),
                "kernel_share": round(float(shares_primary[f]), 6),
                "no_activation_in_sample": below,
                "top_contexts": ev,
            }
        )
    return rows


def feature_block(
    tag: str,
    shares: dict[float, np.ndarray],
    null_stats: dict[float, dict],
    labels: dict[int, dict] | None,
) -> dict:
    """Distribution + ignored/used counts + top-50 lists for one dictionary."""
    out: dict = {"dictionary": tag, "n_features": int(shares[PRIMARY_MASS].size)}
    per_cut = {}
    for m, sh in shares.items():
        ns = null_stats[m]
        finite = sh[np.isfinite(sh)]
        per_cut[str(m)] = {
            "median": float(np.median(finite)),
            "q25": float(np.percentile(finite, 25)),
            "q75": float(np.percentile(finite, 75)),
            "n_ignored_above_null_p97p5": int((finite > ns["p97p5"]).sum()),
            "n_used_below_null_p2p5": int((finite < ns["p2p5"]).sum()),
            "n_nonfinite": int(np.size(sh) - finite.size),
        }
    out["per_cutoff"] = per_cut
    sh = shares[PRIMARY_MASS]
    order = np.argsort(sh)
    def _entry(fid: int) -> dict:
        e = {"feat_id": int(fid), "kernel_share": round(float(sh[fid]), 6)}
        if labels is not None and int(fid) in labels:
            e["label"] = labels[int(fid)]["description"]
        return e
    out["top_ignored"] = [_entry(f) for f in order[::-1][:TOP_LIST]]
    out["top_used"] = [_entry(f) for f in order[:TOP_LIST]]
    return out


def decode_direction(
    d: np.ndarray, dec_unit: np.ndarray, labels: dict[int, dict] | None, top: int = 5
) -> list[dict]:
    """Top |cosine| features of one unit direction against unit decoder rows."""
    cos = dec_unit @ np.asarray(d, dtype=np.float32)
    idx = np.argsort(np.abs(cos))[::-1][:top]
    out = []
    for f in idx:
        e = {"feat_id": int(f), "cos": round(float(cos[f]), 4)}
        if labels is not None and int(f) in labels:
            e["label"] = labels[int(f)]["description"]
        out.append(e)
    return out


def context_extremes(
    proj: np.ndarray, ci: np.ndarray, texts: dict[int, dict], top: int = 5, n_chars: int = QUOTE_FEAT
) -> dict:
    """Top-5 / bottom-5 contexts along one projection (quoted tails, redacted)."""
    order = np.argsort(proj)
    def _fmt(rows):
        out = []
        for r in rows:
            c = int(ci[r])
            t = texts.get(c, {})
            out.append(
                {
                    "ci": c,
                    "proj": round(float(proj[r]), 3),
                    "corpus": t.get("corpus"),
                    "quote": tail_quote(t.get("context_text", "<text unavailable>"), n_chars),
                }
            )
        return out
    return {"top": _fmt(order[::-1][:top]), "bottom": _fmt(order[:top])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=Path("/mnt/eps-data/thomasjiralerspong/wt-2569-kernel-work"))
    ap.add_argument("--repo-root", type=Path, default=_REPO)
    ap.add_argument("--map-root", type=Path, default=Path("/home/thomasjiralerspong/explore-persona-space"))
    ap.add_argument("--sae-ctx", type=Path, default=Path("/mnt/eps-data/thomasjiralerspong/issue2569_theory/sae_ctx/ae.pt"))
    ap.add_argument("--alive-union", type=Path, default=None, help="default: <repo>/eval_results/issue_2569/sae_ctx/alive_union.json")
    ap.add_argument("--andyrdt", type=Path, default=Path(
        "/mnt/eps-data/thomasjiralerspong/huggingface-cache/hub/models--andyrdt--saes-qwen2.5-7b-instruct/"
        "snapshots/c37e53c4bb07127ad17ab88f28b93d4e87142e59/resid_post_layer_19/trainer_1/ae.pt"))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--render-only", action="store_true", help="merge overlay + rewrite md/figure from saved json")
    ap.add_argument("--eigen-addendum", action="store_true",
                    help="compute ONLY the eigen-plane feature block into an existing json (no full re-analysis)")
    ap.add_argument("--overlay", type=Path, default=None, help="interpretation overlay json (readings + narrative)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    repo = args.repo_root
    out_json = repo / "eval_results/issue_2569/weights/leg8/kernel_interpretation_L19.json"
    out_md = repo / "eval_results/issue_2569/weights/leg8/kernel_interpretation_L19.md"
    fig_dir = repo / "figures/issue_2569"

    if args.render_only:
        doc = json.loads(out_json.read_text())
        if args.overlay is not None:
            overlay = json.loads(args.overlay.read_text())
            doc["interpretation_overlay"] = overlay
            out_json.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
        render_md(doc, out_md)
        render_figure(doc, args.work, fig_dir)
        print(f"[render] wrote {out_md} + figures")
        return

    if args.eigen_addendum:
        # Recompute ONLY the eigen-plane feature block against the existing json
        # (same helpers as the full pass; the SVD is re-derived and re-checked
        # against the persisted svd_check so the shares are exact fp64).
        doc = json.loads(out_json.read_text())
        manifest = json.loads((args.work / "download_manifest.json").read_text())
        payload = OP.load_banked_map(LAYER, root=args.map_root)
        A, _b = OP.row_operator(payload)
        U, s, _Vh = svd_row_action(A)
        assert abs(float(s[0]) - doc["svd_check"]["sigma_max"]) < 1e-9
        parts = mass_partitions(s, MASSES)
        ctx = load_ctx_sae(args.sae_ctx)
        fids = np.asarray(EIGEN_FEATS, dtype=np.int64)
        shr = shares_at_masks(U, ctx["w_dec"][fids], {m: parts[m]["mask"] for m in MASSES})
        sh_by_feat = np.zeros(ctx["w_dec"].shape[0])
        sh_by_feat[fids] = shr[PRIMARY_MASS]
        X, ci = load_capture_sample(manifest)
        texts = load_row_meta_texts(manifest, set(int(c) for c in ci.tolist()))
        acts = encode_ctx_features(ctx, X, fids)
        rows = naming_evidence_rows(list(EIGEN_FEATS), acts, sh_by_feat, ci, texts)
        for j, row in enumerate(rows):
            row["kernel_share_by_cutoff"] = {
                str(m): round(float(shr[m][j]), 6) for m in MASSES
            }
        doc["eigen_plane_features"] = {
            "note": (
                "context-SAE features the map's top eigen read planes keep hitting in the "
                "eigen-dashboards v2 run (branch issue-2569-eigen-v2); named the same way as "
                "the top-ignored/top-used lists; kernel_share at the 0.99 (primary) cutoff, "
                "all-cutoff shares alongside"
            ),
            "features": rows,
        }
        out_json.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
        render_md(doc, out_md)
        print(f"[eigen-addendum] merged {len(rows)} features into {out_json}")
        return

    manifest = json.loads((args.work / "download_manifest.json").read_text())
    chunk_sample = json.loads((args.work / "chunk_sample.json").read_text())

    # 0) operator + SVD + checks ---------------------------------------------------
    payload = OP.load_banked_map(LAYER, root=args.map_root)
    A, b = OP.row_operator(payload)
    U, s, Vh = svd_row_action(A)
    parts = mass_partitions(s, MASSES)
    leg1 = json.loads((repo / "eval_results/issue_2569/weights/leg1/anatomy_L19.json").read_text())
    svd_check = {
        "sigma_max": float(s[0]),
        "sigma_max_leg1": leg1["sigma_max"],
        "sigma_median": float(np.median(s)),
        "k99": parts[0.99]["rank"],
        "k99_leg1": leg1["k99"],
        "k90": parts[0.90]["rank"],
        "k90_leg1": leg1["k90"],
        "tau_kernel": parts[0.99]["tau"],
        "tau_kernel_leg1": leg1["tau_kernel"],
        "tau_k90": parts[0.90]["tau"],
        "tau_k90_leg1": leg1["tau_k90"],
        "kernel_dim_by_cutoff": {str(m): int(parts[m]["mask"].sum()) for m in MASSES},
        "row_action_identity_max_rel_err": float(
            max(
                np.linalg.norm(U[:, i] @ A - s[i] * Vh[i]) / max(s[i], 1e-300)
                for i in range(8)
            )
        ),
    }
    assert abs(svd_check["sigma_max"] - leg1["sigma_max"]) < 1e-6 * leg1["sigma_max"]
    assert svd_check["k99"] == leg1["k99"] and svd_check["k90"] == leg1["k90"]

    # persisted leg-8 basis agreement (principal angles)
    kb = torch.load(_hf_local(manifest, "effective_kernel_L19.pt"), map_location="cpu", weights_only=False)
    B = kb["kernel_basis_fp32"].to(torch.float64).numpy()  # (d, 1976)
    mask_p = parts[PRIMARY_MASS]["mask"]
    Uk = U[:, mask_p]
    pa = np.linalg.svd(Uk.T @ B, compute_uv=False)
    basis_agreement = {
        "persisted_dim": int(B.shape[1]),
        "recomputed_dim": int(Uk.shape[1]),
        "principal_cos_min": float(pa.min()),
        "principal_cos_mean": float(pa.mean()),
        "note": "cosines of principal angles between recomputed and persisted kernel bases; 1.0 = identical subspace",
    }

    # 1) random-direction null ------------------------------------------------------
    rng = np.random.default_rng(NULL_SEED)
    dirs0 = rng.standard_normal((args.n_null, D))
    null_shares = shares_at_masks(U, dirs0, {m: parts[m]["mask"] for m in MASSES})
    null_stats = {m: null_share_stats(null_shares[m]) for m in MASSES}
    del dirs0

    # 2) dictionaries ----------------------------------------------------------------
    ctx = load_ctx_sae(args.sae_ctx)
    alive_path = args.alive_union or (repo / "eval_results/issue_2569/sae_ctx/alive_union.json")
    alive = json.loads(Path(alive_path).read_text())
    alive_idx = np.asarray(alive["alive_idx"], dtype=np.int64)
    ctx_shares = shares_at_masks(U, ctx["w_dec"], {m: parts[m]["mask"] for m in MASSES})
    labels_path = repo / "eval_results/issue_1482/context_side_labels/descriptions_context_side.jsonl"
    labels: dict[int, dict] = {}
    with open(labels_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if int(r["feat_id"]) >= 0:
                    labels[int(r["feat_id"])] = {"description": str(r.get("description", ""))[:240]}
    andy_dec = load_andyrdt_decoder(args.andyrdt)
    andy_shares = shares_at_masks(U, andy_dec, {m: parts[m]["mask"] for m in MASSES})

    ctx_block = feature_block("ctx_sae_65536 (alive=all)", ctx_shares, null_stats, None)
    ctx_block["alive"] = {
        "n_alive": int(alive["n_alive"]),
        "n_dict": int(alive["n_dict"]),
        "note": "every feature fired during training (alive_union == full dict), so the alive-restricted headline equals the all-features companion",
    }
    andy_block = feature_block("andyrdt_L19_k64_131072", andy_shares, null_stats, labels)
    andy_block["n_labeled"] = int(len(labels))

    # 3) data-weighted kernel --------------------------------------------------------
    sigma_c, mean_c, n_pool = load_sigma_c(_hf_local(manifest, "moments/gram_xx.pt"))
    trace_sigma = float(np.trace(sigma_c))
    dw = {
        str(m): {
            "ignored_variance_fraction": projected_cov_trace_fraction(U, parts[m]["mask"], sigma_c),
            "kernel_dim": int(parts[m]["mask"].sum()),
            "random_direction_expected_share": float(parts[m]["mask"].sum()) / D,
        }
        for m in MASSES
    }
    dw_check = json.loads((repo / "eval_results/issue_2569/weights/leg1/dw_mass_L19.json").read_text())
    dw_crosscheck = {
        "leg1_dw_mass_ignored_frac": dw_check["classes"]["ignored"]["dw_mass_frac"],
        "recomputed_0p99": dw[str(0.99)]["ignored_variance_fraction"],
        "leg1_trace_sigma_c": dw_check["trace_sigma_c"],
        "recomputed_trace_sigma_c": trace_sigma,
    }

    ker_vals, ker_modes = projected_cov_modes(U, mask_p, sigma_c, TOP_MODES)
    rng_vals, rng_modes = projected_cov_modes(U, ~mask_p, sigma_c, TOP_MODES)
    ker_spec, _ = projected_cov_modes(U, mask_p, sigma_c, TOP_LIST)
    rng_spec, _ = projected_cov_modes(U, ~mask_p, sigma_c, TOP_LIST)

    # 4) context sample + texts ------------------------------------------------------
    X, ci = load_capture_sample(manifest)
    pairs_doc = json.loads((repo / "eval_results/issue_2569/leg8/kernel_pairs.json").read_text())
    pairs = pairs_doc["pairs"]
    by_dc = sorted(range(len(pairs)), key=lambda i: -float(pairs[i]["dc_norm"]))[:TOP_NAME]
    pair_cis: set[int] = set()
    for i in by_dc:
        p = pairs[i]
        pair_cis.update([int(p["ci_i"]), int(p["ci_j"]), int(p["control"]["ci_i"]), int(p["control"]["ci_j"])])
    pair_cis.discard(-1)
    needed = set(int(c) for c in ci.tolist()) | pair_cis
    texts = load_row_meta_texts(manifest, needed)
    sample_meta = {
        "n_chunks": chunk_sample["n"],
        "seed": chunk_sample["seed"],
        "chunks": chunk_sample["chunks"],
        "n_rows_after_dedupe": int(X.shape[0]),
        "n_rows_with_text": int(sum(1 for c in ci.tolist() if int(c) in texts)),
    }

    # unit decoders for mode decoding
    ctx_dec_unit = ctx["w_dec"] / np.maximum(
        np.linalg.norm(ctx["w_dec"], axis=1, keepdims=True), 1e-12
    )
    andy_dec_unit = andy_dec / np.maximum(np.linalg.norm(andy_dec, axis=1, keepdims=True), 1e-12)

    Xc = X - mean_c.astype(np.float32)

    def _modes_block(vals: np.ndarray, modes: np.ndarray) -> list[dict]:
        rows = []
        for r in range(modes.shape[0]):
            d64 = modes[r]
            proj = Xc @ d64.astype(np.float32)
            rows.append(
                {
                    "mode": r + 1,
                    "variance": float(vals[r]),
                    "share_of_total_context_variance": float(vals[r] / trace_sigma),
                    "ctx_sae_top_features": decode_direction(d64, ctx_dec_unit, None),
                    "andyrdt_top_features": decode_direction(d64, andy_dec_unit, labels),
                    "contexts": context_extremes(proj, ci, texts),
                }
            )
        return rows

    ignored_modes = _modes_block(ker_vals, ker_modes)
    range_modes = _modes_block(rng_vals, rng_modes)

    # 5) ctx-SAE feature naming evidence ---------------------------------------------
    sh_p = ctx_shares[PRIMARY_MASS]
    alive_mask = np.zeros(sh_p.size, dtype=bool)
    alive_mask[alive_idx] = True
    order_alive = [f for f in np.argsort(sh_p) if alive_mask[f]]
    name_used = [int(f) for f in order_alive[:TOP_NAME]]
    name_ignored = [int(f) for f in order_alive[::-1][:TOP_NAME]]
    feat_ids = np.asarray(name_ignored + name_used + list(EIGEN_FEATS), dtype=np.int64)
    acts = encode_ctx_features(ctx, X, feat_ids)

    ctx_naming = {
        "note": (
            "top-3 activating contexts per feature over the deduped ~20k-row chunk sample; "
            "quotes are the redacted tails of context_text; readings (added at render) are "
            "the analyst's guess from these excerpts, not judged autointerp"
        ),
        "top_ignored": naming_evidence_rows(name_ignored, acts[:, :TOP_NAME], sh_p, ci, texts),
        "top_used": naming_evidence_rows(
            name_used, acts[:, TOP_NAME : 2 * TOP_NAME], sh_p, ci, texts
        ),
    }
    eigen_block = {
        "note": (
            "context-SAE features the map's top eigen read planes keep hitting in the "
            "eigen-dashboards v2 run (branch issue-2569-eigen-v2); named the same way as "
            "the top-ignored/top-used lists; kernel_share at the 0.99 (primary) cutoff, "
            "all-cutoff shares alongside"
        ),
        "features": [
            {
                **row,
                "kernel_share_by_cutoff": {
                    str(m): round(float(ctx_shares[m][row["feat_id"]]), 6) for m in MASSES
                },
            }
            for row in naming_evidence_rows(
                list(EIGEN_FEATS), acts[:, 2 * TOP_NAME :], sh_p, ci, texts
            )
        ],
    }

    # 6) kernel pairs in words -------------------------------------------------------
    def _side(cix: int) -> dict:
        t = texts.get(int(cix), {})
        return {
            "ci": int(cix),
            "corpus": t.get("corpus"),
            "quote": tail_quote(t.get("context_text", "<text unavailable>"), QUOTE_PAIR),
        }

    pair_rows = []
    for i in by_dc:
        p = pairs[i]
        c = p["control"]
        pair_rows.append(
            {
                "kernel_pair": {
                    "i": _side(p["ci_i"]),
                    "j": _side(p["ci_j"]),
                    "dc_norm": p["dc_norm"],
                    "kappa": p["kappa"],
                    "dva_norm": p["dva_norm"],
                    "ans_len": [p["ans_len_i"], p["ans_len_j"]],
                    "sources": [p["source_i"], p["source_j"]],
                },
                "control_pair": {
                    "i": _side(c["ci_i"]),
                    "j": _side(c["ci_j"]),
                    "dc_norm": c["dc_norm"],
                    "kappa": c["kappa"],
                    "dva_norm": c["dva_norm"],
                    "ans_len": [c["ans_len_i"], c["ans_len_j"]],
                    "sources": [c["source_i"], c["source_j"]],
                },
            }
        )
    dva_k = np.asarray([p["dva_norm"] for p in pairs], dtype=np.float64)
    dva_c = np.asarray([p["control"]["dva_norm"] for p in pairs], dtype=np.float64)
    kap_k = np.asarray([p["kappa"] for p in pairs], dtype=np.float64)
    kap_c = np.asarray([p["control"]["kappa"] for p in pairs], dtype=np.float64)
    pairs_stats = {
        "n_pairs": len(pairs),
        "kernel_dva_median": float(np.median(dva_k)),
        "control_dva_median": float(np.median(dva_c)),
        "kernel_kappa_median": float(np.median(kap_k)),
        "control_kappa_median": float(np.median(kap_c)),
        "kappa_def": "||(c_i - c_j) @ A|| / ||c_i - c_j|| (through-map gain of the context difference)",
    }

    # 7) persona directions ----------------------------------------------------------
    persona_rows = []
    for trait in RB_TRAITS:
        rb = torch.load(_hf_local(manifest, f"r_b/{trait}.pt"), map_location="cpu", weights_only=False)
        v = np.asarray(rb["r_b"][LAYER], dtype=np.float64)
        v = v / np.linalg.norm(v)
        shr = shares_at_masks(U, v[None, :], {m: parts[m]["mask"] for m in MASSES})
        persona_rows.append(
            {"direction": f"r_B {trait} (L19, unit)", **{f"share_{m}": float(shr[m][0]) for m in MASSES}}
        )
        cx = torch.load(_hf_local(manifest, f"{trait}_ctxext_L19.pt"), map_location="cpu", weights_only=False)
        v2 = np.asarray(cx["direction"], dtype=np.float64)
        v2 = v2 / np.linalg.norm(v2)
        shr2 = shares_at_masks(U, v2[None, :], {m: parts[m]["mask"] for m in MASSES})
        persona_rows.append(
            {
                "direction": f"ctxext {trait} (#2254 measured context-steering, L19, unit)",
                **{f"share_{m}": float(shr2[m][0]) for m in MASSES},
            }
        )

    # 8) assemble + write ------------------------------------------------------------
    doc = {
        "task": "issue2569 leg8 kernel interpretation (L19)",
        "conventions": {
            "operator": "row action vhat = v @ A + b; A = diag(1/xsd) W, raw residual coordinates (issue2569_operator)",
            "kernel": "read directions u_i (LEFT singular vectors) with sigma_i < tau(mass); strict <; ridge has no exact kernel",
            "cutoffs": {"tighter": 0.999, "primary": 0.99, "looser": 0.90},
            "kernel_share": "||P_ker d||^2 for a unit direction d; P_ker projects onto the kernel read directions",
            "ignored_variance_fraction": "tr(P_ker Sigma_c)/tr(Sigma_c); Sigma_c = population covariance of raw X19 context rows over the 963,444-row map-training pool (gram/n - mean mean^T, symmetrized)",
            "dictionary_frames": "both dictionaries decode RAW residual coordinates: ctx SAE trained on raw fp16 X19 rows cast to fp32 (no standardization; verified in issue2569_rowbattery._run_sae_training); andyrdt weights fold normalization in and consume raw activations (issue1482_sae docstring)",
            "feature_direction": "unit-normalized decoder vector (w_dec row for ctx SAE, decoder.weight column for andyrdt)",
            "mode_decode": "cosine of unit mode direction against unit decoder vectors, top-5 by |cos|",
            "context_projection": "(x - pool_mean) @ mode_direction over the chunk sample",
            "quotes": "redacted tails of context_text (last user-message region); NOT full conversations",
        },
        "svd_check": svd_check,
        "basis_agreement": basis_agreement,
        "null": {str(m): null_stats[m] for m in MASSES},
        "feature_kernel_share": {"ctx_sae": ctx_block, "andyrdt": andy_block},
        "ctx_sae_naming": ctx_naming,
        "eigen_plane_features": eigen_block,
        "data_weighted": {
            "per_cutoff": dw,
            "trace_sigma_c": trace_sigma,
            "n_pool": n_pool,
            "leg1_crosscheck": dw_crosscheck,
            "ignored_modes_primary": ignored_modes,
            "range_modes_primary": range_modes,
            "ignored_spectrum_top50_share": [float(v / trace_sigma) for v in ker_spec],
            "range_spectrum_top50_share": [float(v / trace_sigma) for v in rng_spec],
        },
        "kernel_pairs": {"stats": pairs_stats, "examples_top20_by_dc_norm": pair_rows},
        "persona_directions": persona_rows,
        "sample": sample_meta,
        "metadata": {
            "git_commit": _git_commit(repo),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "map_payload": str(payload.path),
            "selected_lambda": payload.selected_lambda,
            "n_null": args.n_null,
            "null_seed": NULL_SEED,
            "threads": args.threads,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=1, ensure_ascii=False))

    # figure needs per-feature shares + null draws: save sidecar arrays for render
    np.savez_compressed(
        args.work / "figure_arrays.npz",
        ctx_shares=ctx_shares[PRIMARY_MASS].astype(np.float32),
        andy_shares=andy_shares[PRIMARY_MASS].astype(np.float32),
        null_shares=null_shares[PRIMARY_MASS].astype(np.float32),
        dva_k=dva_k,
        dva_c=dva_c,
        ignored_spec=ker_spec / trace_sigma,
        range_spec=rng_spec / trace_sigma,
        ivf=np.asarray([[m, dw[str(m)]["ignored_variance_fraction"], dw[str(m)]["kernel_dim"] / D] for m in MASSES]),
    )
    render_figure(doc, args.work, fig_dir)
    render_md(doc, out_md)
    print(f"[analyze] wrote {out_json}")


# ── rendering ────────────────────────────────────────────────────────────────────


def render_figure(doc: dict, work: Path, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.load(work / "figure_arrays.npz")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    ns = doc["null"][str(PRIMARY_MASS)]
    bins = np.linspace(0, 1, 81)
    ax.hist(arr["ctx_shares"], bins=bins, density=True, alpha=0.55, label="context SAE (65,536)")
    ax.hist(arr["andy_shares"], bins=bins, density=True, alpha=0.55, label="andyrdt SAE (131,072)")
    ax.axvspan(ns["p2p5"], ns["p97p5"], color="0.75", alpha=0.5, label="random-direction null (2.5–97.5%)")
    ax.set_xlabel("kernel share at 0.99 mass cutoff")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ivf = arr["ivf"]
    order = np.argsort(ivf[:, 0])
    ax.plot(ivf[order, 0], ivf[order, 1], marker="o", label="ignored variance fraction")
    ax.plot(ivf[order, 0], ivf[order, 2], marker="s", label="kernel dimension fraction")
    ax.set_xlabel("squared-singular-mass cutoff")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    k = np.arange(1, len(arr["ignored_spec"]) + 1)
    ax.semilogy(k, arr["ignored_spec"], marker="o", ms=3, label="kernel-restricted covariance")
    ax.semilogy(k, arr["range_spec"], marker="s", ms=3, label="range-restricted covariance")
    ax.set_xlabel("mode rank")
    ax.set_ylabel("share of total context variance")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    hi = float(max(arr["dva_k"].max(), arr["dva_c"].max()))
    bins = np.linspace(0, hi, 60)
    ax.hist(arr["dva_k"], bins=bins, density=True, alpha=0.55, label="kernel pairs")
    ax.hist(arr["dva_c"], bins=bins, density=True, alpha=0.55, label="matched control pairs")
    ax.set_xlabel("answer displacement ||Δy|| (mined pairs)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"leg8_kernel_interpretation.{ext}", dpi=200)
    plt.close(fig)


def _share_row(e: dict, reading: str | None = None) -> str:
    lab = e.get("label") or reading or ""
    return f"| {e['feat_id']} | {e['kernel_share']:.3f} | {lab} |"


def render_md(doc: dict, out_md: Path) -> None:
    ov = doc.get("interpretation_overlay", {})
    fr = ov.get("ctx_feature_readings", {})
    mode_read = ov.get("mode_readings", {})
    L: list[str] = []
    A = L.append
    A("# Interpreting the effective kernel of the L19 context→answer map (task #2569, leg 8 follow-on)")
    A("")
    A("**Definitions.** *Kernel share* of a unit direction d = the squared projection ‖P_ker d‖² onto the")
    A("map's effective-kernel read directions (left singular vectors with σ below the mass cutoff's τ);")
    A("1.0 = the map reads the direction at near-zero gain, 0.0 = fully inside the read range.")
    A("*Ignored variance fraction* = tr(P_ker Σ_c)/tr(Σ_c): the fraction of real context variance")
    A("(population covariance over the map's 963,444-row training pool, raw residual coordinates) that")
    A("falls in the kernel. The kernel is a property of the fitted ridge map: a LOW-GAIN READ subspace,")
    A("never an exact null space, and nothing here is causal.")
    A("")
    sv = doc["svd_check"]
    ba = doc["basis_agreement"]
    A("## SVD + basis checks")
    A("")
    A(f"- σ_max {sv['sigma_max']:.6f} (leg-1 {sv['sigma_max_leg1']:.6f}); k99 {sv['k99']} (leg-1 {sv['k99_leg1']}); "
      f"k90 {sv['k90']} (leg-1 {sv['k90_leg1']}); τ_kernel {sv['tau_kernel']:.6f} (leg-1 {sv['tau_kernel_leg1']:.6f}).")
    A(f"- Kernel dims by cutoff: {sv['kernel_dim_by_cutoff']}.")
    A(f"- Row-action identity max rel err (top-8 triplets): {sv['row_action_identity_max_rel_err']:.2e}.")
    A(f"- Agreement with the persisted leg-8 basis ({ba['persisted_dim']} dirs): principal-angle cosines "
      f"min {ba['principal_cos_min']:.6f}, mean {ba['principal_cos_mean']:.6f}.")
    A("")
    A("## Headline numbers (three cutoffs)")
    A("")
    A("| cutoff (σ² mass) | kernel dim | dim fraction | ignored variance fraction | null share mean [2.5%, 97.5%] |")
    A("|---|---|---|---|---|")
    for m in MASSES:
        d = doc["data_weighted"]["per_cutoff"][str(m)]
        ns = doc["null"][str(m)]
        A(f"| {m} | {d['kernel_dim']} | {d['kernel_dim']/D:.3f} | **{d['ignored_variance_fraction']:.4f}** | "
          f"{ns['mean']:.3f} [{ns['p2p5']:.3f}, {ns['p97p5']:.3f}] |")
    cc = doc["data_weighted"]["leg1_crosscheck"]
    A("")
    A(f"Cross-check: leg-1 dw_mass ignored fraction {cc['leg1_dw_mass_ignored_frac']:.6f} vs recomputed "
      f"{cc['recomputed_0p99']:.6f} at the 0.99 cutoff.")
    A("")
    A("### Feature kernel share per dictionary")
    A("")
    A("| dictionary | cutoff | median share | IQR | ignored (> null 97.5%) | used (< null 2.5%) |")
    A("|---|---|---|---|---|---|")
    for key in ("ctx_sae", "andyrdt"):
        blk = doc["feature_kernel_share"][key]
        for m in MASSES:
            pc = blk["per_cutoff"][str(m)]
            A(f"| {blk['dictionary']} | {m} | {pc['median']:.3f} | [{pc['q25']:.3f}, {pc['q75']:.3f}] | "
              f"{pc['n_ignored_above_null_p97p5']} | {pc['n_used_below_null_p2p5']} |")
    A("")
    A("### Persona directions (kernel share vs null)")
    A("")
    A("| direction | share @0.999 | share @0.99 | share @0.90 |")
    A("|---|---|---|---|")
    for r in doc["persona_directions"]:
        A(f"| {r['direction']} | {r['share_0.999']:.3f} | {r['share_0.99']:.3f} | {r['share_0.9']:.3f} |")
    ns = doc["null"][str(PRIMARY_MASS)]
    A("")
    A(f"Null at 0.99: mean {ns['mean']:.3f}, 2.5–97.5% [{ns['p2p5']:.3f}, {ns['p97p5']:.3f}].")
    A("")

    A("## Feature tables (0.99 cutoff)")
    A("")
    naming = {e["feat_id"]: e for e in doc["ctx_sae_naming"]["top_ignored"] + doc["ctx_sae_naming"]["top_used"]}
    A("### Context SAE — top-20 most-ignored features (my reading from top-activating contexts)")
    A("")
    A("| feat | kernel share | reading (analyst's guess) | top activating context (redacted tail) |")
    A("|---|---|---|---|")
    for e in doc["feature_kernel_share"]["ctx_sae"]["top_ignored"][:TOP_NAME]:
        f = e["feat_id"]
        ev = naming.get(f, {})
        q = ev.get("top_contexts", [{}])[0].get("quote", "")
        A(f"| {f} | {e['kernel_share']:.3f} | {fr.get(str(f), '')} | {q[:160]} |")
    A("")
    A("### Context SAE — top-20 most-used features")
    A("")
    A("| feat | kernel share | reading (analyst's guess) | top activating context (redacted tail) |")
    A("|---|---|---|---|")
    for e in doc["feature_kernel_share"]["ctx_sae"]["top_used"][:TOP_NAME]:
        f = e["feat_id"]
        ev = naming.get(f, {})
        q = ev.get("top_contexts", [{}])[0].get("quote", "")
        A(f"| {f} | {e['kernel_share']:.3f} | {fr.get(str(f), '')} | {q[:160]} |")
    A("")
    if doc.get("eigen_plane_features"):
        A("### Context SAE — the five features the top eigen read planes keep hitting (eigen-dashboards v2)")
        A("")
        A("| feat | kernel share @0.99 | reading (analyst's guess) | top activating context (redacted tail) |")
        A("|---|---|---|---|")
        for e in doc["eigen_plane_features"]["features"]:
            f = e["feat_id"]
            q = e.get("top_contexts", [{}])[0].get("quote", "")
            A(f"| {f} | {e['kernel_share']:.3f} | {fr.get(str(f), '')} | {q[:160]} |")
        A("")
    A("### andyrdt SAE — top-20 most-ignored / most-used (labels where present)")
    A("")
    A("| rank | most-ignored feat (share, label) | most-used feat (share, label) |")
    A("|---|---|---|")
    ig = doc["feature_kernel_share"]["andyrdt"]["top_ignored"][:TOP_NAME]
    us = doc["feature_kernel_share"]["andyrdt"]["top_used"][:TOP_NAME]
    for r in range(TOP_NAME):
        a, u = ig[r], us[r]
        A(f"| {r+1} | {a['feat_id']} ({a['kernel_share']:.3f}) {a.get('label','')} | "
          f"{u['feat_id']} ({u['kernel_share']:.3f}) {u.get('label','')} |")
    A("")

    A("## Ignored-variance modes (0.99 cutoff): the biggest context variations the map discards")
    A("")
    for r, mode in enumerate(doc["data_weighted"]["ignored_modes_primary"]):
        rd = (mode_read.get("ignored") or [None] * TOP_MODES)[r] if mode_read else None
        A(f"**Mode {mode['mode']}** — {100*mode['share_of_total_context_variance']:.2f}% of total context variance."
          + (f" Reading: {rd}" if rd else ""))
        feats = ", ".join(
            f"{e['feat_id']}({e['cos']:+.2f}{', ' + e['label'][:60] if e.get('label') else ''})"
            for e in mode["andyrdt_top_features"]
        )
        A(f"- andyrdt decode: {feats}")
        A(f"- top context: “{mode['contexts']['top'][0]['quote'][:180]}”")
        A(f"- bottom context: “{mode['contexts']['bottom'][0]['quote'][:180]}”")
        A("")
    A("## Range modes (contrast): the biggest context variations the map uses")
    A("")
    for r, mode in enumerate(doc["data_weighted"]["range_modes_primary"]):
        rd = (mode_read.get("range") or [None] * TOP_MODES)[r] if mode_read else None
        A(f"**Mode {mode['mode']}** — {100*mode['share_of_total_context_variance']:.2f}% of total context variance."
          + (f" Reading: {rd}" if rd else ""))
        feats = ", ".join(
            f"{e['feat_id']}({e['cos']:+.2f}{', ' + e['label'][:60] if e.get('label') else ''})"
            for e in mode["andyrdt_top_features"]
        )
        A(f"- andyrdt decode: {feats}")
        A(f"- top context: “{mode['contexts']['top'][0]['quote'][:180]}”")
        A(f"- bottom context: “{mode['contexts']['bottom'][0]['quote'][:180]}”")
        A("")

    A("## Kernel pairs vs matched controls (20 + 20, read by eye)")
    A("")
    ps = doc["kernel_pairs"]["stats"]
    A(f"κ = through-map gain of the context difference. Kernel pairs: median κ {ps['kernel_kappa_median']:.3f}, "
      f"median answer displacement {ps['kernel_dva_median']:.1f}. Controls: median κ {ps['control_kappa_median']:.3f}, "
      f"median answer displacement {ps['control_dva_median']:.1f} (n={ps['n_pairs']} each).")
    A("")
    if ov.get("pairs_paragraph"):
        A(ov["pairs_paragraph"])
        A("")
    for i, row in enumerate(doc["kernel_pairs"]["examples_top20_by_dc_norm"], 1):
        k, c = row["kernel_pair"], row["control_pair"]
        A(f"**Pair {i}** (kernel: dc {k['dc_norm']:.0f}, κ {k['kappa']:.2f}, ‖Δŷ‖ {k['dva_norm']:.0f} | "
          f"control: dc {c['dc_norm']:.0f}, κ {c['kappa']:.2f}, ‖Δŷ‖ {c['dva_norm']:.0f})")
        A(f"- kernel i ({k['i']['corpus']}): “{k['i']['quote']}”")
        A(f"- kernel j ({k['j']['corpus']}): “{k['j']['quote']}”")
        A(f"- control i ({c['i']['corpus']}): “{c['i']['quote']}”")
        A(f"- control j ({c['j']['corpus']}): “{c['j']['quote']}”")
        A("")

    A("## What this says")
    A("")
    A(ov.get("what_this_says", "(analyst paragraph pending — see interpretation overlay)"))
    A("")
    A("---")
    md = doc["metadata"]
    A(f"*Generated at {md['timestamp_utc']} from commit `{md['git_commit'][:12]}`; map payload "
      f"`{md['map_payload']}`; sample: {doc['sample']['n_chunks']} capture chunks (seed {doc['sample']['seed']}), "
      f"{doc['sample']['n_rows_after_dedupe']} deduped rows. Kernel = low-gain read subspace of the fitted "
      f"linear map; no causal claim.*")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
