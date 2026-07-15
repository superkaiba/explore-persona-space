#!/usr/bin/env python
# ruff: noqa: RUF002  # em-dash + marker token intentional
"""#1333 geometry aggregator (VM-side, CPU) — the 2x2 paired reads (plan §3/§6).

Thin driver over the captured pooled stores: per (cell, arm, layer) spectral
DVs of the row-centered Δx = trained − base clouds (rank-k@90, participation
ratio, top-share, ‖μ‖), paired cluster bootstrap over the registered 100-row
(context x question) panel with IDENTICAL resample indices per pair (n_boot
1000 / 2000 for ‖μ‖ differences, seed 653 — parent conventions), the H1–H4
registered lattices, the mandatory own-vs-shared-text collapse read, split-half
cloud reliability (the §6 floor), and the descriptive |cos(μ, W_U[83399])|
alignment read vs a norm-matched random band.

Batched Gram-eigh throughout (the parent-proven path): eigenvalues of the
per-draw double-centered resampled Gram submatrices via ONE batched
``torch.linalg.eigvalsh`` per (cell, arm, layer) — never a per-draw SVD loop
(vectorize-many-cell-fits.md).

Inputs default to the run tree ``data/issue_1333/run/capture/`` (own-text at
``<cell>/selected/pooled.pt``; shared-text at ``<cell>/tf_shared/pooled.pt``;
base at ``base_marker/base/pooled.pt``; the reused arm's own-text is the
staged parent store at ``m2_fullft_band8/selected/pooled.pt``).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments import issue_1333 as C  # noqa: E402

logger = logging.getLogger("issue1333.geometry")

ARMS = ("prefix", "context", "response")
# Own-text store per 2x2 cell within the capture root (reused arm = parent store).
OWN_STORE = {
    C.CELL_LORA_CON: "mk1_lora_con/selected/pooled.pt",
    C.CELL_LORA_POS: "mk2_lora_pos/selected/pooled.pt",
    C.CELL_FT_CON_REUSED: "m2_fullft_band8/selected/pooled.pt",
    C.CELL_FT_POS: "mk4_fullft_pos/selected/pooled.pt",
}
TF_STORE = {cell: f"{cell}/tf_shared/pooled.pt" for cell in C.GEOMETRY_CELLS}
BASE_STORE = "base_marker/base/pooled.pt"


def _load_store(path: Path) -> dict:
    store = torch.load(path, map_location="cpu", weights_only=False)
    assert set(store["arms"]) >= set(ARMS), sorted(store["arms"])
    return store


def _row_keys(store: dict) -> list[tuple[str, int]]:
    return [(m["context_id"], int(m["question_idx"])) for m in store["row_meta"]]


def assert_registered_panel(store: dict, *, smoke: bool) -> None:
    """Row-coverage hard assert (plan §3): the registered 100-row panel is
    5 contexts x 20 questions; a smoke stub only needs a complete grid."""
    keys = _row_keys(store)
    ctxs = sorted({k[0] for k in keys})
    qs = sorted({k[1] for k in keys})
    assert len(set(keys)) == len(keys) == len(ctxs) * len(qs), (len(keys), len(ctxs), len(qs))
    if not smoke:
        assert len(ctxs) == 5 and qs == list(range(20)) and len(keys) == 100, (ctxs, qs, len(keys))


def _aligned_delta(trained: dict, base: dict, arm: str, layer: int) -> np.ndarray:
    """Row-aligned Δx = trained − base at one (arm, layer), fp32 (n, d)."""
    tk, bk = _row_keys(trained), _row_keys(base)
    order = {k: i for i, k in enumerate(bk)}
    missing = [k for k in tk if k not in order]
    assert not missing, f"row_meta mismatch vs base store: {missing[:3]}"
    idx = [order[k] for k in tk]
    t = trained["arms"][arm][layer].to(torch.float32).numpy()
    b = base["arms"][arm][layer].to(torch.float32).numpy()[idx]
    assert t.shape == b.shape, (t.shape, b.shape)
    return t - b


def _spectral_from_eigs(eigs: np.ndarray) -> dict[str, float]:
    """rank-k@90 / participation ratio / top-share from Gram eigenvalues."""
    lam = np.clip(np.sort(eigs)[::-1], 0.0, None)
    total = float(lam.sum())
    if total <= 0:
        return {"rank_k90": 0.0, "pr": 0.0, "top_share": 0.0}
    cum = np.cumsum(lam) / total
    rank_k90 = float(np.searchsorted(cum, 0.9) + 1)
    pr = float(total**2 / (lam**2).sum())
    return {"rank_k90": rank_k90, "pr": pr, "top_share": float(lam[0] / total)}


def _point_dvs(cloud: np.ndarray) -> dict[str, float]:
    """Point DVs of one Δx cloud: spectral stats of the row-centered Gram + ‖μ‖."""
    mu = cloud.mean(axis=0)
    xc = cloud - mu
    eigs = np.linalg.eigvalsh(xc @ xc.T)
    out = _spectral_from_eigs(eigs)
    out["mu_norm"] = float(np.linalg.norm(mu))
    out["n_rows"] = int(cloud.shape[0])
    return out


def _boot_draw_indices(n_rows: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_rows, size=(n_boot, n_rows))


def _batched_draw_stats(cloud: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Per-draw spectral DVs + ‖μ‖, batched via Gram submatrices.

    ONE Gram ``G = X Xᵀ`` is computed per cloud; each draw's resampled Gram is
    a gather ``G[idx][:, idx]``, double-centered batched, then one batched
    ``eigvalsh`` over (B, n, n). ‖μ_d‖² = mean of the UNcentered resampled
    Gram entries (the subset-sum identity).
    """
    x = torch.from_numpy(cloud).to(torch.float64)
    g = x @ x.T  # (n, n)
    ii = torch.from_numpy(idx).long()  # (B, n)
    gd = g[ii.unsqueeze(2), ii.unsqueeze(1)]  # (B, n, n)
    mu_norm = torch.sqrt(torch.clamp(gd.mean(dim=(1, 2)), min=0.0))
    row_mean = gd.mean(dim=2, keepdim=True)
    col_mean = gd.mean(dim=1, keepdim=True)
    tot_mean = gd.mean(dim=(1, 2), keepdim=True)
    gc = gd - row_mean - col_mean + tot_mean
    eigs = torch.linalg.eigvalsh(gc)  # (B, n)
    lam = torch.clamp(eigs.flip(-1), min=0.0)
    total = lam.sum(dim=1, keepdim=True)
    safe = torch.clamp(total, min=1e-30)
    cum = torch.cumsum(lam, dim=1) / safe
    rank_k90 = (cum < 0.9).sum(dim=1).to(torch.float64) + 1.0
    pr = total.squeeze(1) ** 2 / torch.clamp((lam**2).sum(dim=1), min=1e-30)
    top_share = lam[:, 0] / safe.squeeze(1)
    return {
        "rank_k90": rank_k90.numpy(),
        "pr": pr.numpy(),
        "top_share": top_share.numpy(),
        "mu_norm": mu_norm.numpy(),
    }


def _ci(draws: np.ndarray, alpha: float = 0.05) -> list[float]:
    return [float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))]


def split_half_self_cosine(
    cloud: np.ndarray, *, n_splits: int = 100, seed: int = C.BOOT_SEED
) -> float:
    """Cloud-reliability floor (plan §6): mean cosine between the two halves'
    mean-shift vectors over row-aligned random half-partitions."""
    rng = np.random.default_rng(seed)
    n = cloud.shape[0]
    cosines = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = cloud[perm[: n // 2]].mean(axis=0), cloud[perm[n // 2 :]].mean(axis=0)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cosines.append(float(a @ b / denom) if denom > 0 else 0.0)
    return float(np.mean(cosines))


def _wu_row(wu_path: Path | None) -> np.ndarray | None:
    """The BASE unembedding row W_U[83399] (read-out direction r_B, plan §3).

    Reads a persisted ``wu_row.pt`` when given; else extracts the single row
    from the base model's safetensors shard (partial read — never a full model
    load on the VM)."""
    if wu_path is not None and Path(wu_path).exists():
        t = torch.load(wu_path, map_location="cpu", weights_only=True)
        return t.to(torch.float32).numpy()
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        idx_path = hf_hub_download(C.BASE_MODEL, "model.safetensors.index.json")
        weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
        key = "lm_head.weight" if "lm_head.weight" in weight_map else "model.embed_tokens.weight"
        shard = hf_hub_download(C.BASE_MODEL, weight_map[key])
        with safe_open(shard, framework="pt") as f:
            sl = f.get_slice(key)
            row = sl[C.MARKER_TOKEN_ID : C.MARKER_TOKEN_ID + 1]
        return row[0].to(torch.float32).numpy()
    except Exception as e:
        logger.warning("[geometry] W_U row unavailable (%s) — alignment read skipped", e)
        return None


def _alignment_read(
    mu: np.ndarray, wu: np.ndarray, *, n_rand: int = 1000, seed: int = C.BOOT_SEED
) -> dict:
    rng = np.random.default_rng(seed)
    denom = np.linalg.norm(mu) * np.linalg.norm(wu)
    cos = float(abs(mu @ wu / denom)) if denom > 0 else 0.0
    rand = rng.standard_normal((n_rand, mu.shape[0]))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    null = np.abs(rand @ (mu / max(np.linalg.norm(mu), 1e-30)))
    return {"abs_cos": cos, "random_band_p975": float(np.quantile(null, 0.975))}


def _paired_diff_draws(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape, (a.shape, b.shape)
    return a - b


# ── §6 registered re-reductions (concern analyzer-rereductions-deferred) ──────

TRAIN_HALF_Q = tuple(range(10))  # EVAL_QUESTIONS_20[:10] == TRAIN_QUESTIONS (#508 q_train)
HALF_SPLIT_MATERIAL_NATS = 1.0  # plan §6: |Δ| > 1 nat on source install -> clean-result caveat
SELECT_CELLS = (*C.NEW_LORA_CELLS, C.CELL_FT_POS)  # cells with a ladder + selection record


def matched_80_reads(
    cloud: np.ndarray, *, n_sub: int = 80, n_draws: int = 100, seed: int = C.BOOT_SEED
) -> dict:
    """Matched-80-row subsample read (plan §6 exploratory dump: cross-issue
    spectrum comparison vs #1112's marker matched-80): mean/sd spectral DVs +
    ‖μ‖ over ``n_draws`` random WITHOUT-replacement ``n_sub``-row subsamples
    of the persisted cloud, via the batched Gram-eigh path."""
    n = cloud.shape[0]
    if n <= n_sub:
        return {"skipped": f"n_rows {n} <= n_sub {n_sub} (stub-scale store)"}
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.permutation(n)[:n_sub] for _ in range(n_draws)])
    draws = _batched_draw_stats(cloud, idx)
    return {
        "n_sub": n_sub,
        "n_draws": n_draws,
        **{k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))} for k, v in draws.items()},
    }


def eval_bank_half_split(slot_record: dict) -> dict:
    """Plan §6 eval-bank overlap mitigation: source-install ΔG split by
    eval-bank half — train-overlapping q_idx 0-9 (the frozen-mix positives'
    questions) vs held-out q_idx 10-19 — from one persisted per-question slot
    record (a ladder ``slot_read.json`` / the reused arm's
    ``apply_gate.json``). A material |Δ| > 1 nat is flagged (carried as a
    clean-result caveat, incl. onto the matched-install calibration)."""
    halves: dict[str, list[float]] = {"train_overlap": [], "held_out": []}
    for row in slot_record["per_probe"]:
        q = int(row["row"]["q"])
        delta = float(row["trained"]["logp"]) - float(row["base"]["logp"])
        halves["train_overlap" if q in TRAIN_HALF_Q else "held_out"].append(delta)
    if not halves["train_overlap"] or not halves["held_out"]:
        return {"skipped": "slot record does not cover both eval-bank halves"}
    m_tr = float(np.mean(halves["train_overlap"]))
    m_ho = float(np.mean(halves["held_out"]))
    return {
        "train_overlap_mean": m_tr,
        "held_out_mean": m_ho,
        "n_train_overlap": len(halves["train_overlap"]),
        "n_held_out": len(halves["held_out"]),
        "delta_train_minus_heldout": m_tr - m_ho,
        "material": bool(abs(m_tr - m_ho) > HALF_SPLIT_MATERIAL_NATS),
    }


def cross_surface_gap_curve(traj: dict, ladder_reads: dict[int, dict]) -> list[dict]:
    """Plan §4.3/§6 cross-surface gap curve: the in-loop (teacher-forced
    frozen-R) trajectory vs the off-line on-policy ladder per step — the free
    registered descriptive whose rising-in-loop-vs-flat-off-line signature is
    the adapter-apply no-op discriminator on the flat-ladder branch."""
    in_loop = {int(s): float(d) for s, d in zip(traj["steps"], traj["delta_nats"], strict=True)}
    off_line = {int(s): float(r["delta_logp_mean"]) for s, r in ladder_reads.items()}
    out = []
    for step in sorted(set(in_loop) | set(off_line)):
        il, ol = in_loop.get(step), off_line.get(step)
        out.append(
            {
                "step": step,
                "in_loop_delta": il,
                "off_line_delta": ol,
                "gap": (il - ol) if il is not None and ol is not None else None,
            }
        )
    return out


def eos_margin_transfer_fractions(slot_reads: dict, *, source_label: str = "__source__") -> dict:
    """Plan §6 read (2): per-context transfer fraction = bystander gain ÷
    source gain in the EOS-margin logit space ``Δ(z_marker − z_eos)`` trained
    − base — NEVER raw log P (softmax compression shrinks a saturated
    source's denominator and inflates the fraction exactly on the strongest
    implants). Numerator + denominator ride alongside; the fraction is null
    when the source margin gain is degenerate (<1e-9). The fraction is never
    correlated back against install (#383 X-vs-(X−Y) family)."""
    by_label: dict[str, list[float]] = {}
    for row in slot_reads["per_probe"]:
        t, b = row["trained"], row["base"]
        margin = (t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"])
        by_label.setdefault(row["row"]["label"], []).append(float(margin))
    if source_label not in by_label:
        return {"skipped": f"source label {source_label!r} absent from slot reads"}
    src = float(np.mean(by_label[source_label]))
    per_ctx = {}
    for label, vals in by_label.items():
        if label == source_label:
            continue
        gain = float(np.mean(vals))
        per_ctx[label] = {
            "margin_gain": gain,
            "transfer_fraction": (gain / src) if abs(src) > 1e-9 else None,
        }
    return {"source_margin_gain": src, "per_context": per_ctx}


def _half_split_for_cell(run_root: Path, cell: str) -> dict:
    """Half-split driver: the selected rung's slot_read.json per select cell;
    the reused arm reads its apply-and-read gate record (same per_probe
    shape, same 20-question bank)."""
    if cell == C.CELL_FT_CON_REUSED:
        p = run_root / "gates" / "reused_apply" / "apply_gate.json"
    else:
        sel_p = run_root / cell / "selection.json"
        if not sel_p.exists():
            return {"skipped": f"{sel_p} missing"}
        step = int(json.loads(sel_p.read_text())["step"])
        p = run_root / cell / "ladder" / f"rung_{step}" / "slot_read.json"
    if not p.exists():
        return {"skipped": f"{p} missing"}
    rec = eval_bank_half_split(json.loads(p.read_text()))
    rec["source_record"] = str(p)
    return rec


def _gap_curve_for_cell(run_root: Path, cell: str) -> dict:
    traj_p = run_root / cell / "band_trajectory.json"
    ladder_p = run_root / cell / "ladder.json"
    missing = [str(p) for p in (traj_p, ladder_p) if not p.exists()]
    if missing:
        return {"skipped": f"missing {missing}"}
    traj = json.loads(traj_p.read_text())
    reads = {int(k): v for k, v in json.loads(ladder_p.read_text())["reads_by_step"].items()}
    return {"points": cross_surface_gap_curve(traj, reads)}


def _dose_curve_for_cell(run_root: Path, cell: str) -> dict:
    """Plan §6 read (3) assembly: per read rung, the ladder's SOURCE install
    (delta_logp_mean, riding the ``dose_curve_rungs`` entries) + the flanking
    bystander battery's per-context Δlog-prob / Δmargin leakage. The
    ``dose_curve_rungs`` field makes the realized rung-resolution
    self-describing (concern ladder-bystander-dose-curves)."""
    sel_p = run_root / cell / "selection.json"
    if not sel_p.exists():
        return {"skipped": f"{sel_p} missing"}
    sel = json.loads(sel_p.read_text())
    rungs = sel.get("dose_curve_rungs")
    if not rungs:
        return {"skipped": "selection record carries no dose_curve_rungs field"}
    points = []
    for item in rungs:
        step = int(item["step"])
        row = dict(item)
        bys_p = run_root / cell / "ladder" / f"rung_{step}" / "bystanders.json"
        if bys_p.exists():
            row["bystanders_per_context"] = json.loads(bys_p.read_text()).get("per_context")
        points.append(row)
    return {
        "selected_step": sel.get("step"),
        "in_window": sel.get("in_window"),
        "dose_curve_rungs": rungs,
        "points": points,
    }


def run_re_reductions(
    run_root: Path | None,
    stores: dict[str, dict[str, dict]],
    base: dict,
    primary: int,
) -> dict:
    """The four registered §6 re-reductions of persisted artifacts (concern
    analyzer-rereductions-deferred) + the dose-curve assembly. Missing inputs
    record an explicit ``skipped`` reason — visible in the JSON, never
    silent."""
    rr: dict = {
        "matched_80": {
            cell: {
                kind: matched_80_reads(_aligned_delta(store, base, "response", primary))
                for kind, store in kinds.items()
            }
            for cell, kinds in stores.items()
        }
    }
    if run_root is None:
        for key in (
            "eval_bank_half_split",
            "cross_surface_gap",
            "eos_margin_transfer",
            "dose_curves",
        ):
            rr[key] = {"skipped": "run_root not provided"}
        return rr
    rr["eval_bank_half_split"] = {
        cell: _half_split_for_cell(run_root, cell) for cell in (*SELECT_CELLS, C.CELL_FT_CON_REUSED)
    }
    rr["cross_surface_gap"] = {
        cell: _gap_curve_for_cell(run_root, cell) for cell in C.NEW_LORA_CELLS
    }
    transfer = {}
    for cell in C.BREADTH_CELLS:
        p = run_root / "breadth" / cell / "slot_reads.json"
        transfer[cell] = (
            eos_margin_transfer_fractions(json.loads(p.read_text()))
            if p.exists()
            else {"skipped": f"{p} missing"}
        )
    rr["eos_margin_transfer"] = transfer
    rr["dose_curves"] = {cell: _dose_curve_for_cell(run_root, cell) for cell in SELECT_CELLS}
    for key, per_cell in rr.items():
        for cell, rec in per_cell.items() if isinstance(per_cell, dict) else ():
            if isinstance(rec, dict) and "skipped" in rec:
                logger.warning("[re-reductions] %s/%s skipped: %s", key, cell, rec["skipped"])
    return rr


def _lattice_call(diff_draws: np.ndarray, point: float, *, labels: tuple[str, str, str]) -> dict:
    """DISJOINT+exhaustive lattice (plan §3): positive/negative/indistinguishable."""
    ci = _ci(diff_draws)
    if point > 0 and ci[0] > 0:
        verdict = labels[0]
    elif ci[1] < 0:
        verdict = labels[1]
    else:
        verdict = labels[2]
    return {"point": float(point), "ci95": ci, "verdict": verdict, "n_draws": len(diff_draws)}


def run_geometry(  # noqa: C901 — linear read + lattice chain
    capture_root: Path,
    out_json: Path,
    matrices_dir: Path,
    *,
    smoke: bool = False,
    n_boot: int | None = None,
    wu_path: Path | None = None,
    cells: tuple[str, ...] = C.GEOMETRY_CELLS,
    run_root: Path | None = None,
) -> dict:
    """Full 2x2 geometry read; returns (and persists) the results record.

    ``run_root`` (the dispatcher's out_root tree — ladders, selections,
    trajectories, breadth slot reads) feeds the §6 re-reductions
    (``run_re_reductions``); when None those file-based reads record explicit
    ``skipped`` reasons."""
    n_boot = n_boot if n_boot is not None else (16 if smoke else C.N_BOOT)
    n_boot_mu = 16 if smoke else C.N_BOOT_MU
    base = _load_store(capture_root / BASE_STORE)
    assert_registered_panel(base, smoke=smoke)
    layers = sorted(base["arms"]["response"])
    primary = C.PRIMARY_LAYER if C.PRIMARY_LAYER in layers else layers[-1]

    stores: dict[str, dict[str, dict]] = {}
    for cell in cells:
        stores[cell] = {}
        own_p = capture_root / OWN_STORE[cell]
        tf_p = capture_root / TF_STORE[cell]
        if own_p.exists():
            s = _load_store(own_p)
            assert_registered_panel(s, smoke=smoke)
            # §3 "identical resample indices" made mechanical (review r1 m13):
            # paired draws share row indices across stores, so row ORDER must
            # match the base store exactly, not just as a set.
            assert _row_keys(s) == _row_keys(base), f"{cell}/own row order != base store"
            stores[cell]["own"] = s
        if tf_p.exists():
            s = _load_store(tf_p)
            assert_registered_panel(s, smoke=smoke)
            assert _row_keys(s) == _row_keys(base), f"{cell}/tf row order != base store"
            stores[cell]["tf"] = s
    wu = _wu_row(wu_path)

    results: dict = {
        "issue": C.ISSUE,
        "primary_layer": int(primary),
        "n_boot": n_boot,
        "n_boot_mu": n_boot_mu,
        "boot_seed": C.BOOT_SEED,
        "smoke": smoke,
        "cells": {},
        "metadata": {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or "unknown",
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    matrices_dir.mkdir(parents=True, exist_ok=True)

    # Shared paired resample indices (identical across every paired read).
    n_rows = len(base["row_meta"])
    idx = _boot_draw_indices(n_rows, n_boot, C.BOOT_SEED)
    idx_mu = _boot_draw_indices(n_rows, n_boot_mu, C.BOOT_SEED)

    draws_by_cell: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for cell, kinds in stores.items():
        rec: dict = {}
        draws_by_cell[cell] = {}
        for kind, store in kinds.items():
            per_layer_points = {
                arm: {int(li): _point_dvs(_aligned_delta(store, base, arm, li)) for li in layers}
                for arm in ARMS
            }
            cloud = _aligned_delta(store, base, "response", primary)
            draws = _batched_draw_stats(cloud, idx)
            draws["mu_norm_2000"] = _batched_draw_stats(cloud, idx_mu)["mu_norm"]
            draws_by_cell[cell][kind] = draws
            # per-draw x per-layer matrices for the response arm (persisted).
            per_layer_draws = {
                int(li): _batched_draw_stats(_aligned_delta(store, base, "response", li), idx)
                for li in layers
            }
            torch.save(
                {"cell": cell, "kind": kind, "idx_seed": C.BOOT_SEED, "per_layer": per_layer_draws},
                matrices_dir / f"{cell}_{kind}_response_draws.pt",
            )
            rec[kind] = {
                "point_by_arm_layer": per_layer_points,
                "primary": per_layer_points["response"][int(primary)],
                "primary_ci": {k: _ci(v) for k, v in draws.items()},
                "split_half_self_cosine": split_half_self_cosine(cloud),
            }
            mu = cloud.mean(axis=0)
            if wu is not None and wu.shape[0] == mu.shape[0]:
                rec[kind]["alignment_wu"] = _alignment_read(mu, wu)
            elif wu is not None:
                logger.warning(
                    "[geometry] W_U row dim %d != cloud dim %d — alignment skipped "
                    "(stub-scale store)",
                    wu.shape[0],
                    mu.shape[0],
                )
        results["cells"][cell] = rec

    # Registered lattices (plan §3) — only when both arms of a contrast exist
    # and (H1-H3) both sit in the acceptance window (precondition handled by
    # the analyzer off the selection records; computed here unconditionally
    # with the labels attached — off-band exclusion is a re-reduction).
    lat: dict = {}

    def _own(cell: str, stat: str, mu2000: bool = False) -> np.ndarray | None:
        d = draws_by_cell.get(cell, {}).get("own")
        if d is None:
            return None
        return d["mu_norm_2000" if mu2000 else stat]

    def _pt(cell: str, stat: str) -> float | None:
        r = results["cells"].get(cell, {}).get("own")
        return None if r is None else r["primary"][stat]

    a, b = _own(C.CELL_FT_CON_REUSED, "rank_k90"), _own(C.CELL_LORA_CON, "rank_k90")
    if a is not None and b is not None:
        lat["H1_D_rank"] = _lattice_call(
            _paired_diff_draws(a, b),
            _pt(C.CELL_FT_CON_REUSED, "rank_k90") - _pt(C.CELL_LORA_CON, "rank_k90"),
            labels=("FTMoreDiffuse", "FTMoreConcentrated", "ShapeIndistinguishable"),
        )
    a, b = _own(C.CELL_FT_CON_REUSED, "", True), _own(C.CELL_LORA_CON, "", True)
    if a is not None and b is not None:
        lat["H2_D_mag"] = _lattice_call(
            _paired_diff_draws(a, b),
            _pt(C.CELL_FT_CON_REUSED, "mu_norm") - _pt(C.CELL_LORA_CON, "mu_norm"),
            labels=("FTFarther", "FTNearer", "MagnitudeIndistinguishable"),
        )
    a, b = _own(C.CELL_LORA_CON, "rank_k90"), _own(C.CELL_LORA_POS, "rank_k90")
    if a is not None and b is not None:
        lat["H3_D_neg"] = _lattice_call(
            _paired_diff_draws(a, b),
            _pt(C.CELL_LORA_CON, "rank_k90") - _pt(C.CELL_LORA_POS, "rank_k90"),
            labels=("NegativesMoreDiffuse", "NegativesMoreConcentrated", "NegativesShapeNull"),
        )
    # H4: per-cell own-minus-shared rank draws; D_shc = (count of cells whose
    # 95% CI excludes 0 positively) - 3; Collapse <=> D_shc >= 0.
    shc: dict[str, dict] = {}
    n_pos = 0
    for cell in cells:
        o = draws_by_cell.get(cell, {}).get("own")
        t = draws_by_cell.get(cell, {}).get("tf")
        if o is None or t is None:
            continue
        diff = _paired_diff_draws(o["rank_k90"], t["rank_k90"])
        ci = _ci(diff)
        pos = ci[0] > 0
        n_pos += int(pos)
        shc[cell] = {
            "point": results["cells"][cell]["own"]["primary"]["rank_k90"]
            - results["cells"][cell]["tf"]["primary"]["rank_k90"],
            "ci95": ci,
            "positive": pos,
        }
    if len(shc) == len(cells):
        d_shc = n_pos - 3
        lat["H4_D_shc"] = {
            "per_cell": shc,
            "d_shc": d_shc,
            "verdict": "Collapse" if d_shc >= 0 else "NoCollapseConsensus",
        }
    elif shc:
        lat["H4_D_shc"] = {"per_cell": shc, "verdict": "INCOMPLETE — missing cells"}
    results["lattices"] = lat

    # §6 re-reductions of persisted artifacts (concern
    # analyzer-rereductions-deferred): matched-80, eval-bank half-split,
    # cross-surface gap curves, EOS-margin transfer fractions + dose curves.
    results["re_reductions"] = run_re_reductions(run_root, stores, base, int(primary))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float) + "\n")
    os.replace(tmp, out_json)
    logger.info("[geometry] wrote %s (+ draw matrices under %s)", out_json, matrices_dir)
    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    torch.set_num_threads(int(os.environ.get("EPS_VM_THREAD_CAP", "8") or 8))
    p = argparse.ArgumentParser(description="#1333 geometry aggregator (VM-side)")
    p.add_argument("--capture-root", default=f"data/issue_{C.ISSUE}/run/capture")
    p.add_argument(
        "--run-root",
        default=f"data/issue_{C.ISSUE}/run",
        help="dispatcher out_root (ladders/selections/trajectories/breadth) feeding the "
        "§6 re-reductions; pass an empty string to skip the file-based reads",
    )
    p.add_argument(
        "--out-json", default=f"eval_results/issue_{C.ISSUE}/geometry/geometry_marker_2x2.json"
    )
    p.add_argument("--matrices-dir", default=f"data/issue_{C.ISSUE}/run/bootstrap_matrices")
    p.add_argument("--smoke", action="store_true", help="stub-scale (tiny panel, n_boot 16)")
    p.add_argument("--n-boot", type=int, default=None)
    p.add_argument("--wu-row", default=None, help="optional persisted W_U[83399] row .pt")
    p.add_argument("--cells", default=None, help="comma subset of the 2x2 cells")
    p.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="upload per-draw matrices + results JSON to the HF data repo "
        "(plan §10 analysis_tensors/bootstrap_matrices; default: on unless --smoke)",
    )
    args = p.parse_args(argv)
    cells = tuple(args.cells.split(",")) if args.cells else C.GEOMETRY_CELLS
    run_geometry(
        Path(args.capture_root),
        Path(args.out_json),
        Path(args.matrices_dir),
        smoke=args.smoke,
        n_boot=args.n_boot,
        wu_path=Path(args.wu_row) if args.wu_row else None,
        cells=cells,
        run_root=Path(args.run_root) if args.run_root else None,
    )
    # VM-side upload duty (plan §10; review r1 m9): the per-draw x per-layer
    # bootstrap matrices are produced post-teardown, so the pod's p8 cannot
    # own them — this driver uploads them itself.
    if args.upload if args.upload is not None else not args.smoke:
        from explore_persona_space.orchestrate import hub

        hub._upload(
            Path(args.matrices_dir),
            C.HF_DATA_REPO,
            "dataset",
            f"{C.DATA_PREFIX}/analysis_tensors/bootstrap_matrices",
        )
        hub._upload(
            Path(args.out_json),
            C.HF_DATA_REPO,
            "dataset",
            f"{C.DATA_PREFIX}/geometry/{Path(args.out_json).name}",
            upload_as_file=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
