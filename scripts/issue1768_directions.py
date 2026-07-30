"""#1768 direction reads — p9 (plan §4.6: Q4 write shape + A4-A7 battery).

All direction reads run in FULL hidden-dim space (never PCA-projected — the
body's finding: only 38-63% of the write's squared norm lies in the base
top-64 PCA subspace). Per (arm, layer):

- ŵ = panel source-context write, with the DISJOINT-question-halves
  shared-baseline registration (Statistics Must-Fix; selection-symmetric-nulls
  § Noise-structure symmetry): the ŵ leg's v̄⁰ comes from half A (even panel
  question indices), the δ leg's v̄⁰ from half B (odd) — the shared-B̄
  full-panel read is kept RECORD-ONLY beside it, and split-half reliability of
  both legs is reported (a ~0-reliability leg reads as attenuation).
- Three-way horse race cos(ŵ, ·) for δ / r_B / (marker only) W_U[83399], each
  against: corpus-covariance N(0, Σ) norm-matched draws (with the #778
  top-eigenvector contamination pre-check), isotropic norm-matched draws, the
  cross-behavior r_B null, and a shuffled-row (label-permutation) null.
  The ŵ_tf matched-text companion race runs beside the on-policy race.
- A6 rank read (top-1 SVD share of centered corpus Δv + cos(u1, ŵ)), the A7
  whitened base-similarity gate (Spearman ρ(g_pred, ĝ)), A5 scalar-fit
  residual, and A4 cos(r⁺_B, r_B) on the 6 re-extracted arms.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402

logger = logging.getLogger("issue1768.directions")

N_NULL_DRAWS = 2000
SHRINKAGE = 0.1  # Σ shrinkage λ (plan §4.6)
TOP_EIG_CONTAM_COS = 0.4  # covariance-band contamination flag (#778 pre-check)

RB_HUB_PATHS = {
    "syc": "issue1112_geometry2x2/analysis_tensors/rb/rb_sycophancy.pt",
    "mk": "issue1112_geometry2x2/analysis_tensors/rb/rb_marker.pt",
    "imp": "issue1315_impolite_geometry/analysis_tensors/rb/rb_impolite.pt",
    "cas": "issue1434_writingstyle/analysis_tensors/rb_writing_style.pt",
}
SRC_CTX_POS = {"pers": 0, "bare": 1, "conv": 2, "icl": 3}  # panel order (§4.4)


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _meta() -> dict:
    return {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "issue": X.ISSUE}


def _load_store(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


# ── inputs: rb tensors, W_U row, Σ_corpus ────────────────────────────────────


def load_rb_tensors(out_root: Path, rb_dir: Path | None = None) -> dict[str, np.ndarray]:
    """The four fleet r_B stacks (realized-keys checked; Hub-staged on miss)."""
    import torch

    from explore_persona_space.orchestrate import hub

    dest_dir = Path(rb_dir) if rb_dir else Path(out_root) / "inputs" / "rb"
    out: dict[str, np.ndarray] = {}
    for beh, hub_path in RB_HUB_PATHS.items():
        local = dest_dir / Path(hub_path).name
        if not local.exists():
            hub.stage_hub_file(X.HF_DATA_REPO, hub_path, local, repo_type="dataset")
        obj = torch.load(local, map_location="cpu", weights_only=False)
        rb = obj["rb"] if "rb" in obj else obj["r_b"]
        rb = np.asarray(rb, dtype=np.float64)
        assert rb.ndim == 2, (beh, rb.shape)  # (n_layers, hidden); #1112 join rb[layer]
        out[beh] = rb
    return out


def assert_marker_token(model_path: str, token_id: int) -> None:
    """In-process marker-id assert (CLAUDE.md marker rule; plan assumption 9).

    The W_U row slice is keyed on a HARD-PINNED id — assert the tokenizer
    actually maps the leading-space marker text to exactly that id before any
    read keys on it (round-1 Minor). The marker text is built from the
    codepoint (U+203B) so no literal transits tool/edit layers.
    """
    from transformers import AutoTokenizer

    marker_text = " " + chr(0x203B)  # " ※" leading-space form
    enc = AutoTokenizer.from_pretrained(model_path).encode(marker_text, add_special_tokens=False)
    assert enc == [token_id], (
        f"marker token drift: encode({marker_text!r}) -> {enc}, expected [{token_id}]"
    )


def load_wu_row(model_path: str, token_id: int = X.MARKER_TOKEN_ID) -> np.ndarray:
    """Base `lm_head.weight[token_id]` via a safetensors slice (no full load)."""
    import torch
    from safetensors import safe_open

    if token_id == X.MARKER_TOKEN_ID:
        assert_marker_token(model_path, token_id)

    local = Path(model_path)
    if not local.exists():  # hub repo id: resolve the shard holding lm_head
        from huggingface_hub import hf_hub_download

        from explore_persona_space.orchestrate import hub

        try:
            idx_path = hub.retry_transient(
                lambda: hf_hub_download(model_path, "model.safetensors.index.json"),
                what="wu index fetch",
            )
            weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
            shard = weight_map.get("lm_head.weight") or weight_map["model.embed_tokens.weight"]
        except Exception:  # single-file model
            shard = "model.safetensors"
        local = Path(
            hub.retry_transient(
                lambda s=shard: hf_hub_download(model_path, s), what=f"wu shard fetch {shard}"
            )
        )
        with safe_open(str(local), framework="pt") as sf:
            key = "lm_head.weight" if "lm_head.weight" in sf.keys() else "model.embed_tokens.weight"
            row = sf.get_slice(key)[token_id]
            return np.asarray(torch.as_tensor(row).float().numpy(), dtype=np.float64).ravel()
    for f in sorted(local.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for key in ("lm_head.weight", "model.embed_tokens.weight"):
                if key in sf.keys():
                    row = sf.get_slice(key)[token_id]
                    return np.asarray(
                        torch.as_tensor(row).float().numpy(), dtype=np.float64
                    ).ravel()
    raise FileNotFoundError(f"no lm_head/embed_tokens weight found under {model_path}")


def corpus_sigma(out_root: Path, layer: int) -> dict:
    """Uncentered second moment E[cc^T] of the base TRAIN context vectors,
    with shrinkage Σ_sh = (1−λ)Σ + λ·(tr(Σ)/D)·I (plan §4.6/A7)."""
    store = _load_store(Path(out_root) / "corpus_capture" / "base_content" / "pooled.pt")
    C = np.asarray(store["arms"]["context"][layer].float().numpy(), dtype=np.float64)
    sample = X.load_corpus_sample(Path(out_root))
    qidx = np.asarray(store["row_question_idx"])
    C_tr = C[qidx < sample["n_train"]]
    d = C_tr.shape[1]
    sigma = C_tr.T @ C_tr / max(1, C_tr.shape[0])
    sigma_sh = (1 - SHRINKAGE) * sigma + SHRINKAGE * (np.trace(sigma) / d) * np.eye(d)
    evals, evecs = np.linalg.eigh(sigma_sh)
    return {
        "sigma": sigma_sh,
        "top_eig": evecs[:, -1],
        "chol": np.linalg.cholesky(sigma_sh + 1e-10 * np.eye(d)),
        "n_rows": int(C_tr.shape[0]),
    }


# ── panel write legs (disjoint-halves registration) ──────────────────────────


def _panel_ctx_order(store: dict) -> list[str]:
    seen: list[str] = []
    for m in store["row_meta"]:
        if m["context_id"] not in seen:
            seen.append(m["context_id"])
    return seen


def _panel_rows(
    store: dict, ctx_id: str, layer: int, span: str = "response"
) -> dict[int, np.ndarray]:
    """{question_idx: vector} at one (context, layer, span)."""
    mat = np.asarray(store["arms"][span][layer].float().numpy(), dtype=np.float64)
    return {
        m["question_idx"]: mat[i]
        for i, m in enumerate(store["row_meta"])
        if m["context_id"] == ctx_id
    }


def source_context_id(arm: X.Arm, base_store: dict) -> str:
    order = _panel_ctx_order(base_store)
    pos = SRC_CTX_POS[arm.ctx_key]
    assert pos < len(order), (arm.arm_id, order)
    return order[pos]


def _half_means(rows: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean_all, mean_even_qidx, mean_odd_qidx) — deterministic aligned halves."""
    qs = sorted(rows)
    even = [rows[q] for q in qs if q % 2 == 0]
    odd = [rows[q] for q in qs if q % 2 == 1]
    assert even and odd, f"need >=1 question in each half, got {len(qs)} questions"
    return (
        np.mean([rows[q] for q in qs], axis=0),
        np.mean(even, axis=0),
        np.mean(odd, axis=0),
    )


def panel_write_legs(out_root: Path, arm: X.Arm, layer: int) -> dict:
    """ŵ legs + δ baseline half + reliabilities for one (arm, layer)."""
    beh = arm.beh_key
    base_store = _load_store(Path(out_root) / "panel_capture" / f"base_{beh}" / "pooled.pt")
    arm_store = _load_store(Path(out_root) / "panel_capture" / arm.arm_id / "pooled.pt")
    src_ctx = source_context_id(arm, base_store)
    v0 = _panel_rows(base_store, src_ctx, layer)
    vp = _panel_rows(arm_store, src_ctx, layer)
    v0_all, v0_A, v0_B = _half_means(v0)
    vp_all, vp_A, vp_B = _half_means(vp)
    legs = {
        "src_ctx": src_ctx,
        "w_primary": vp_all - v0_A,  # ŵ leg baseline: half A (Must-Fix)
        "w_shared_record_only": vp_all - v0_all,
        "v0_half_B": v0_B,  # the δ leg's baseline (disjoint half)
        "v0_all": v0_all,
        "w_split_half_cos": _cos(vp_A - v0_A, vp_B - v0_B),
        "n_questions": len(v0),
    }
    # quarter-split of half B for the δ split-half reliability read: the two
    # δ halves must NOT share one sampled baseline (a shared v0_half_B adds
    # the same noise vector to both legs and inflates the reliability —
    # round-1 Minor; selection-symmetric-nulls § Noise-structure symmetry)
    qs_odd = [q for q in sorted(v0) if q % 2 == 1]
    b1_rows = [v0[q] for q in qs_odd[0::2]]
    b2_rows = [v0[q] for q in qs_odd[1::2]]
    legs["v0_half_B1"] = np.mean(b1_rows, axis=0) if b1_rows else None
    legs["v0_half_B2"] = np.mean(b2_rows, axis=0) if b2_rows else None
    tf_path = Path(out_root) / "panel_capture_tf" / arm.arm_id / "pooled.pt"
    if tf_path.exists():
        tf_store = _load_store(tf_path)
        vtf = _panel_rows(tf_store, src_ctx, layer)
        vtf_all, vtf_A, vtf_B = _half_means(vtf)
        legs["w_tf_primary"] = vtf_all - v0_A
        legs["w_tf_split_half_cos"] = _cos(vtf_A - v0_A, vtf_B - v0_B)
    return legs


def delta_leg(out_root: Path, arm: X.Arm, layer: int, legs: dict) -> dict:
    # ft arms READ the matched pers-LoRA cell's t̄ (same #1481 mix — the δ
    # cells COINCIDE; plan §4.1 amendment). LoRA arms read their own.
    delta_arm = X.delta_arm_for(arm)
    tb = _load_store(Path(out_root) / "delta_tf" / delta_arm / "tbar.pt")
    tbar = np.asarray(tb["tbar"][layer].float().numpy(), dtype=np.float64)
    v0_half_B = legs["v0_half_B"]
    out = {
        "delta_primary": tbar - v0_half_B,  # δ leg baseline: half B (Must-Fix)
        "delta_shared_record_only": tbar - np.asarray(legs["v0_all"], dtype=np.float64),
        "n_mix_rows": int(tb["n_rows"]),
        "delta_arm": delta_arm,
    }
    if "tbar_even" in tb and tb["tbar_even"] is not None:
        te = np.asarray(tb["tbar_even"][layer].float().numpy(), dtype=np.float64)
        to = np.asarray(tb["tbar_odd"][layer].float().numpy(), dtype=np.float64)
        b1, b2 = legs.get("v0_half_B1"), legs.get("v0_half_B2")
        if b1 is not None and b2 is not None:
            # disjoint quarter-split baselines per δ half (round-1 Minor: a
            # shared v0_half_B carries the same noise vector in both legs)
            out["delta_split_half_cos"] = _cos(te - b1, to - b2)
            out["delta_split_half_cos_sharedB_record_only"] = _cos(te - v0_half_B, to - v0_half_B)
        else:  # <2 odd-half questions: shared-B read is an UPPER BOUND, labeled
            out["delta_split_half_cos"] = _cos(te - v0_half_B, to - v0_half_B)
            out["delta_split_half_cos_upper_bound_sharedB"] = True
    return out


# ── null batteries (batched draws; #778 round-3 families) ────────────────────


def _norm_matched_band(w: np.ndarray, cand_norm: float, draws: np.ndarray) -> dict:
    """cos(ŵ, g_i) for norm-matched draws g_i (norm cancels in cosine)."""
    wn = w / (np.linalg.norm(w) + 1e-12)
    gn = draws / (np.linalg.norm(draws, axis=1, keepdims=True) + 1e-12)
    cos = gn @ wn
    return {
        "p2_5": float(np.quantile(cos, 0.025)),
        "p97_5": float(np.quantile(cos, 0.975)),
        "abs_p95": float(np.quantile(np.abs(cos), 0.95)),
        "n_draws": int(draws.shape[0]),
    }


def null_bands(w: np.ndarray, sigma: dict, rng: np.random.Generator) -> dict:
    d = w.shape[0]
    z = rng.standard_normal((N_NULL_DRAWS, d))
    iso = _norm_matched_band(w, 1.0, z)
    cov_draws = z @ sigma["chol"].T
    cov = _norm_matched_band(w, 1.0, cov_draws)
    return {"isotropic": iso, "corpus_covariance": cov}


def shuffled_row_band(
    vp_rows: np.ndarray, v0_rows: np.ndarray, cand: np.ndarray, rng: np.random.Generator
) -> dict:
    """Label-permutation null: re-split pooled rows into two groups, read
    cos(mean(g1) − mean(g2), candidate) — batched as one ±weight GEMM."""
    pool = np.concatenate([vp_rows, v0_rows], axis=0)
    n_p, n_tot = vp_rows.shape[0], pool.shape[0]
    W = np.full((N_NULL_DRAWS, n_tot), -1.0 / (n_tot - n_p))
    for k in range(N_NULL_DRAWS):
        idx = rng.permutation(n_tot)[:n_p]
        W[k, idx] = 1.0 / n_p
    perm_writes = W @ pool  # (draws, d)
    return _norm_matched_band(cand, 1.0, perm_writes)


# ── A6 / A7 / A5 reads ───────────────────────────────────────────────────────


def rank_read(delta_v: np.ndarray, w: np.ndarray) -> dict:
    """Top-1 SVD share + participation ratio + cos(u1, ŵ) of centered Δv."""
    import torch

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    D = torch.as_tensor(delta_v - delta_v.mean(axis=0), dtype=torch.float32, device=dev)
    q = min(8, min(D.shape) - 1) if min(D.shape) > 1 else 1
    _u, _s, V = torch.svd_lowrank(D, q=max(1, q), niter=4)
    sv_full = torch.linalg.svdvals(D).double()
    s2 = sv_full**2
    top1 = float((s2[0] / s2.sum()).item())
    pr = float(((s2.sum() ** 2) / (s2**2).sum()).item())
    u1 = (V[:, 0]).cpu().numpy().astype(np.float64)
    return {
        "top1_var_share": top1,
        "participation_ratio": pr,
        "cos_u1_w": abs(_cos(u1, w)),  # sign-free (SVD direction sign arbitrary)
        "n_rows": int(delta_v.shape[0]),
    }


def gate_read(C0: np.ndarray, delta_v: np.ndarray, c_src: np.ndarray, w: np.ndarray, sigma) -> dict:
    """A7 gate: g_pred = c_srcᵀΣ⁻¹c_x / c_srcᵀΣ⁻¹c_src vs ĝ = ŵᵀΔv/‖ŵ‖²."""
    from scipy.stats import spearmanr

    a = np.linalg.solve(sigma["sigma"], c_src)
    denom = float(c_src @ a)
    g_pred = C0 @ a / (denom + 1e-12)
    g_hat = delta_v @ w / (float(w @ w) + 1e-12)
    rho, p = spearmanr(g_pred, g_hat)
    return {"spearman_rho": float(rho), "p_value": float(p), "n": int(len(g_pred))}


def scalar_fit_residual(w: np.ndarray, delta: np.ndarray) -> dict:
    """A5: min_a ‖ŵ − a·δ‖ — residual squared-norm share."""
    dd = float(delta @ delta)
    if dd == 0:
        return {"a": float("nan"), "residual_share": float("nan")}
    a = float(w @ delta) / dd
    resid = w - a * delta
    return {"a": a, "residual_share": float(resid @ resid) / float(w @ w)}


# ── p9 driver ────────────────────────────────────────────────────────────────


def _arms_in_scope(smoke: bool, arms_filter) -> list[X.Arm]:
    arms = X.all_arms()
    if smoke and not arms_filter:
        return [a for a in arms if a.arm_id == X.PILOT_ARM]
    if arms_filter:
        return [a for a in arms if a.arm_id in set(arms_filter)]
    return arms


def run_p9(
    out_root: Path,
    results_dir: Path,
    layers,
    smoke: bool,
    arms_filter=(),
    rb_dir: Path | None = None,
    wu_model: str | None = None,
) -> None:
    import issue1768_fit as fit

    import zlib

    print("[phase=p9_directions]", flush=True)
    out_root, results_dir = Path(out_root), Path(results_dir)
    arms = _arms_in_scope(smoke, arms_filter)
    rb = load_rb_tensors(out_root, rb_dir)

    direction_reads: dict[str, dict] = {}
    gate_reads: dict[str, dict] = {}
    sigma_by_layer = {li: corpus_sigma(out_root, li) for li in layers}
    wu_cache: np.ndarray | None = None

    # per-(arm, layer) persistence + resume (checkpoint-per-phase intra-phase
    # grain: 72 arms x 3 layers = 216 units > 50; round-1 Major 4). Units are
    # rng-seeded PER UNIT (deterministic, order-independent) so a resumed run
    # reproduces a fresh run's draws exactly.
    unit_dir = results_dir / "p9_units"
    total = len(arms) * len(layers)
    k = 0
    t_phase = time.time()
    for arm in arms:
        base_store = None  # lazy: skipped entirely when every unit resumes
        for layer in layers:
            key = f"{arm.arm_id}_L{layer}"
            k += 1
            t0 = time.time()
            unit_path = unit_dir / f"{key}.json"
            if unit_path.exists():
                u = json.loads(unit_path.read_text())
                if u.get("smoke") == smoke:  # regime key (resume predicate)
                    direction_reads[key] = u["direction"]
                    gate_reads[key] = u["gate"]
                    print(f"[p9] unit {k}/{total} {key} resumed", flush=True)
                    continue
            rng = np.random.default_rng(
                [X.FLOOR_SEED, zlib.crc32(arm.arm_id.encode("utf-8")), layer]
            )
            if base_store is None:
                base_store = _load_store(
                    out_root / "panel_capture" / f"base_{arm.beh_key}" / "pooled.pt"
                )
            legs = panel_write_legs(out_root, arm, layer)
            dleg = delta_leg(out_root, arm, layer, legs)
            w = legs["w_primary"]
            sigma = sigma_by_layer[layer]

            # candidates (full-dim; plan §4.6)
            candidates: dict[str, np.ndarray] = {"delta": dleg["delta_primary"]}
            rb_stack = rb[arm.beh_key]
            layer_ok = layer < rb_stack.shape[0]
            if layer_ok:
                candidates["r_B"] = rb_stack[layer]
            if arm.kind == "marker":
                if wu_cache is None:
                    wu_cache = load_wu_row(wu_model or X.BASE_MODEL)
                candidates["W_U_marker_row"] = wu_cache

            cell = fit.load_corpus_cell(arm.arm_id, layer, out_root)
            delta_v = cell["Vplus"] - cell["V0"]
            delta_v_tf = cell["Vplus_tf"] - cell["V0"]

            races = {}
            src_v0_rows = _panel_rows(base_store, legs["src_ctx"], layer)
            arm_store = _load_store(out_root / "panel_capture" / arm.arm_id / "pooled.pt")
            src_vp_rows = _panel_rows(arm_store, legs["src_ctx"], layer)
            vp_mat = np.stack([src_vp_rows[q] for q in sorted(src_vp_rows)])
            v0_mat = np.stack([src_v0_rows[q] for q in sorted(src_v0_rows)])
            for cname, cand in candidates.items():
                if cand.shape[0] != w.shape[0]:
                    races[cname] = {"skipped": f"dim {cand.shape[0]} != {w.shape[0]}"}
                    continue
                entry = {
                    "cos_w": _cos(w, cand),
                    "cos_w_shared_record_only": _cos(legs["w_shared_record_only"], cand),
                    "top_eig_contamination_cos": _cos(sigma["top_eig"], cand),
                    "nulls": null_bands(cand, sigma, rng),
                    "shuffled_row_null": shuffled_row_band(vp_mat, v0_mat, cand, rng),
                }
                entry["covariance_band_contaminated"] = (
                    abs(entry["top_eig_contamination_cos"]) >= TOP_EIG_CONTAM_COS
                )
                entry["primary_null_family"] = (
                    "isotropic" if entry["covariance_band_contaminated"] else "corpus_covariance"
                )
                if "w_tf_primary" in legs:
                    entry["cos_w_tf"] = _cos(legs["w_tf_primary"], cand)
                races[cname] = entry
            if arm.kind == "marker" and arm.method == "ft" and "W_U_marker_row" in races:
                # full-FT trains W_U (no LoRA gauge freeze) — the Q4 candidate
                # stays the BASE row as a FIXED reference; analyzer caveat.
                races["W_U_marker_row"]["wu_row_source"] = (
                    "base model (ft arm trains W_U; base row kept as fixed Q4 reference)"
                )
            cross = {
                other: _cos(w, rb[other][layer])
                for other in rb
                if other != arm.beh_key and layer < rb[other].shape[0]
            }
            direction_reads[key] = {
                "arm_id": arm.arm_id,
                "method": arm.method,  # lora | ft — the amendment's grouping column
                "layer": layer,
                "src_ctx": legs["src_ctx"],
                "n_panel_questions": legs["n_questions"],
                "w_norm": float(np.linalg.norm(w)),
                "w_split_half_cos": legs["w_split_half_cos"],
                "w_tf_split_half_cos": legs.get("w_tf_split_half_cos"),
                "delta_split_half_cos": dleg.get("delta_split_half_cos"),
                "delta_split_half_cos_sharedB_record_only": dleg.get(
                    "delta_split_half_cos_sharedB_record_only"
                ),
                "delta_split_half_cos_upper_bound_sharedB": dleg.get(
                    "delta_split_half_cos_upper_bound_sharedB", False
                ),
                "n_mix_rows": dleg["n_mix_rows"],
                "delta_arm": dleg["delta_arm"],  # != arm_id on ft arms (shared t̄)
                "races": races,
                "cross_behavior_rb_cos": cross,
                "A5_scalar_fit": scalar_fit_residual(w, dleg["delta_primary"]),
                "A6_rank": rank_read(delta_v, w),
                "A6_rank_tf": rank_read(delta_v_tf, w),
                "baseline_halves": "w:evens(A) delta:odds(B) — disjoint (Must-Fix)",
            }
            c_src_rows = _panel_rows(base_store, legs["src_ctx"], layer, span="context")
            c_src = np.mean(list(c_src_rows.values()), axis=0)
            gate_reads[key] = {
                "arm_id": arm.arm_id,
                "method": arm.method,
                "layer": layer,
                "on_policy": gate_read(cell["C0"], delta_v, c_src, w, sigma),
                "matched_text": gate_read(cell["C0"], delta_v_tf, c_src, w, sigma),
                "sigma_n_rows": sigma["n_rows"],
                "sigma_shrinkage": SHRINKAGE,
                "sigma_moment": "uncentered E[cc^T] (main.tex A7 definition)",
            }
            _atomic_json(  # persist the moment the unit completes (Major 4)
                unit_path,
                {
                    "smoke": smoke,
                    "direction": direction_reads[key],
                    "gate": gate_reads[key],
                    **_meta(),
                },
            )
            print(
                f"[p9] unit {k}/{total} {key} elapsed={time.time() - t0:.0f}s "
                f"(phase {time.time() - t_phase:.0f}s)",
                flush=True,
            )

    _atomic_json(
        results_dir / "direction_reads.json",
        {"reads": direction_reads, "n_null_draws": N_NULL_DRAWS, "smoke": smoke, **_meta()},
    )
    _atomic_json(results_dir / "gate_reads.json", {"reads": gate_reads, "smoke": smoke, **_meta()})

    # A4 stability: re-extracted r⁺_B vs the fleet r_B (the 6 p6 arms)
    stability: dict[str, dict] = {}
    for arm in arms:
        done = out_root / "rb_plus" / arm.arm_id / "done.json"
        if not done.exists():
            continue
        trait = json.loads(done.read_text())["trait"]
        obj = _load_store(out_root / "rb_plus" / arm.arm_id / "r_b" / f"{trait}.pt")
        rplus = np.asarray(obj["r_b"], dtype=np.float64)
        fleet = rb[arm.beh_key]
        stability[arm.arm_id] = {
            "trait": trait,
            "cos_by_layer": {
                str(li): _cos(rplus[li], fleet[li])
                for li in layers
                if li < min(rplus.shape[0], fleet.shape[0])
            },
        }
    _atomic_json(results_dir / "rb_stability.json", {"reads": stability, "smoke": smoke, **_meta()})
