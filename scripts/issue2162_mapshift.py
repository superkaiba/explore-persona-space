#!/usr/bin/env python3
"""Issue #2162 `mapshift` inline follow-up — Results 2.5 + 4-extension on banked tensors.

Consolidation-plan round (docs/reports/issue_2162_consolidation_plan.md), 0 GPU-h:

- **Result 2.5** (`--phase shift`): per (type-cell x layer x map source x patch arm),
  cosine between the map-predicted answer-state shift ``M(v_C^B) - M(v_C^A)`` and the
  ACTUAL patched shift ``v_A(patched) - v_A(floor anchors of A)``; companions:
  magnitude ratio + shift-space R^2; ceiling reference = full-swap shift
  ``v_A(B anchors) - v_A(A anchors)`` with DISJOINT anchor halves wherever one floor
  enters two compared quantities (#1415 shared-baseline fix). Nulls: shuffled-pair
  assignment (carrier-blocked derangement, the #2215 dv3 convention) + shuffled-map
  (fresh arm: refit on context-permuted pairing; banked arms: row-permuted weights,
  the #1739 arm-13 convention).
- **Result 4 extension** (`--phase dv3ext`): the #2215 paired 2AFC (dv3 conventions:
  ``sim_blocks`` / ``observed_2afc`` / carrier-blocked deranged null / carrier-
  clustered bootstrap) at span pooling, adding the fresh bank-fit map (per layer)
  and identity-only (``v_hat = v_C``) arms beside identity+bias; the banked #779 ce
  ridge arm is recomputed as a parity anchor against the committed
  ``eval_results/issue_2215/dv3_map_discrimination.json``.
- **Fresh maps** (`--phase fresh_fit`): per-layer ridge ``v_C -> v_A`` on the bank's
  own anchor answer states, PER-DRAW rows, leave-one-carrier-out over the 12
  carriers. n_train (~12.9k) > d (3,584) => PRIMAL feature-space Gram; the exact
  standardize / center / GCV recipe of
  ``issue_1739.fits.ridge_fit_predict_primal_layer_batched`` (parity-pinned on the
  pilot unit), implemented via full-Gram downdates so the two big GEMMs are paid
  once per layer instead of once per fold. Lambda selection is GCV over the #825
  grid WITH the #825 dof cap (0.9 * n_tr; asserted non-binding at n >> d) and
  per-fold selected-lambda diagnostics — the #825-sanctioned dof-capped-GCV form.
  Identity+learned-bias baseline + kNN retrieval reported beside every fitted read
  (standing rule).

Every HF read is pinned to ONE revision (task #2321 is repacking the data repo).
Checkpoint per phase + per unit (layers / cells) with resume predicates keyed on
the leg (``--pilot`` vs ``--full`` legs get disjoint work/out roots).

Usage:
  uv run python scripts/issue2162_mapshift.py --import-check
  uv run python scripts/issue2162_mapshift.py --pilot            # 1 layer x 1 fold
  uv run python scripts/issue2162_mapshift.py --full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # credentials + shared-VM thread caps BEFORE any heavy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.experiments.issue_1739 import fits as F1739  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2215_analysis as A2215  # noqa: E402  (PairTable, cell views, dv3 helpers)

logger = logging.getLogger("issue2162.mapshift")

REPO_ROOT = _SCRIPTS_DIR.parent
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# ONE revision for EVERY download this round (resolved 2026-08-16; #2321 repack guard).
HF_REVISION = "7d3ac543a5a4202e3996be1498886f2bab637c15"

P2162 = "issue2162_ctxinfo"
BANK_PREFIX = f"{P2162}/analysis_tensors/vc_bank"
ANCHOR_PREFIX = f"{P2162}/analysis_tensors/anchors"
VA_PREFIX = f"{P2162}/analysis_tensors/va_store"
MAPS_1739 = (
    "issue1739_ctxmap/analysis_tensors/maps/context_end__u5000.npz",
    "issue1739_ctxmap/analysis_tensors/maps/context_end__ufull.npz",
)
BANKED_LAYERS = (14, 19, 26)
RIDGE_779 = {
    layer: f"issue779_monitoring/n1m_readout/weights/L{layer}/ridge.pt" for layer in BANKED_LAYERS
}
RIDGE_1738 = {
    layer: f"issue1738_multiturn/analysis_tensors/weights/L{layer}/context_ridge.pt"
    for layer in BANKED_LAYERS
}

HIDDEN = 3584
N_MODEL_LAYERS = 28
K_ANCHOR_DRAWS = 10  # parent ANCHOR_DRAWS; halves = draws {0..4} vs {5..9} (#2215 convention)
PATCH_ARMS = ("steered", "shuffled", "crosstype")

LAMBDAS = np.logspace(-2, 4, 13)  # #825 grid (issue825_fit_cells.LAMBDAS)
GCV_DOF_CAP = 0.9  # #825 GCV_DOF_CAP — asserted non-binding at n_tr >> d
SEED = 21625
SHUF_PAIR_B = 1000
SHUF_MAP_DRAWS = 5
SHUF_MAP_LAYERS = (14, 19, 26)  # fresh shuffled-map refits restricted to the read layers
DV3_NULL_B = 10_000
DV3_BOOT_B = 10_000
DV3_METRICS = ("cosine", "euclidean")
_EPS = 1e-30

BANKED_DV3_JSON = REPO_ROOT / "eval_results/issue_2215/dv3_map_discrimination.json"
TWO_BY_TWO_JSON = REPO_ROOT / "eval_results/issue_2162/f_metrics/two_by_two.json"


def _import_check() -> None:
    """Execute every deferred production import + the argparse-attribute audit."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    import issue779_ffc_n1m_fits  # noqa: F401 — deferred FITS.apply_map import
    from issue2094_analysis import bootstrap_family_means_batched  # noqa: F401
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        list_hf_entries_under_path,
        stage_hub_file,
        stage_hub_prefix,
    )

    print("[import-check] ok")


# ── io ────────────────────────────────────────────────────────────────


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def _repro() -> dict:
    return {
        "script": "scripts/issue2162_mapshift.py",
        "hf_revision": HF_REVISION,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **as_metadata_dict(git_provenance()),
    }


# ── config ────────────────────────────────────────────────────────────


@dataclass
class Cfg:
    leg: str  # "pilot" | "full"
    in_root: Path
    work_root: Path  # per-leg (…/work/<leg>)
    out_root: Path  # per-leg (eval_results/issue_2162/mapshift[/pilot])
    device: str = "cpu"
    layers_sel: list[int] = field(default_factory=list)  # fresh-fit layers
    folds_sel: list[int] = field(default_factory=list)  # fresh-fit folds
    cells_sel: list[str] | None = None  # shift/dv3ext cell slice (pilot)
    shuf_pair_b: int = SHUF_PAIR_B
    shuf_map_draws: int = SHUF_MAP_DRAWS
    dv3_null_b: int = DV3_NULL_B
    dv3_boot_b: int = DV3_BOOT_B

    @property
    def mirror(self) -> Path:
        return self.in_root

    @property
    def va_dir(self) -> Path:
        return self.in_root / VA_PREFIX

    @property
    def anchors_dir(self) -> Path:
        return self.in_root / ANCHOR_PREFIX

    @property
    def bank_json(self) -> Path:
        return self.in_root / BANK_PREFIX / "bank.json"

    @property
    def vc_bank_pt(self) -> Path:
        return self.in_root / BANK_PREFIX / "vc_bank.pt"

    @property
    def fresh_dir(self) -> Path:
        return self.work_root / "fresh_preds"

    @property
    def shufmap_dir(self) -> Path:
        return self.work_root / "fresh_shufmap"


# ── phase: stage ──────────────────────────────────────────────────────


def phase_stage(cfg: Cfg) -> None:
    """Scoped staging at the pinned revision + schema-from-artifact probes."""
    import concurrent.futures as cf
    import shutil

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import (
        list_hf_entries_under_path,
        stage_hub_file,
        stage_hub_prefix,
    )

    logger.info("[phase=stage] revision=%s -> %s", HF_REVISION, cfg.in_root)
    usage = shutil.disk_usage(cfg.in_root if cfg.in_root.exists() else cfg.in_root.parent)
    logger.info("[stage] dest fs free=%.1f GB", usage.free / 1e9)
    for prefix in (BANK_PREFIX, ANCHOR_PREFIX):
        files = stage_hub_prefix(
            HF_DATA_REPO, prefix, cfg.in_root, repo_type="dataset", revision=HF_REVISION
        )
        logger.info("[stage] prefix %s: %d files", prefix, len(files))
    entries = list_hf_entries_under_path(
        HfApi(), HF_DATA_REPO, VA_PREFIX, repo_type="dataset", revision=HF_REVISION
    )
    ce = [(p, size) for p, size in entries if "__ce__" in p]
    assert ce, f"no __ce__ shards listed under {VA_PREFIX}"
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [
            pool.submit(
                stage_hub_file,
                HF_DATA_REPO,
                p,
                cfg.in_root / p,
                repo_type="dataset",
                revision=HF_REVISION,
                size_bytes=size,
            )
            for p, size in ce
        ]
        for i, f in enumerate(cf.as_completed(futs), 1):
            f.result()
            if i % 20 == 0 or i == len(futs):
                logger.info("[stage] va_store unit %d/%d", i, len(futs))
    for rel in (*MAPS_1739, *RIDGE_779.values(), *RIDGE_1738.values()):
        stage_hub_file(
            HF_DATA_REPO, rel, cfg.in_root / rel, repo_type="dataset", revision=HF_REVISION
        )
    report = {
        "revision": HF_REVISION,
        "n_va_ce_shards": len(ce),
        "dest_free_gb_before": usage.free / 1e9,
        "schemas": _schema_probe(cfg),
        "repro": _repro(),
    }
    _write_json_atomic(cfg.out_root / "stage_report.json", report)
    logger.info("[phase=stage_done] report -> %s", cfg.out_root / "stage_report.json")


def _schema_probe(cfg: Cfg) -> dict:
    """Observed top-level keys of ONE real instance of each staged artifact class."""
    out: dict = {}
    bank = json.loads(cfg.bank_json.read_text())
    out["bank_json"] = {
        "top": sorted(bank.keys()),
        "context_row": sorted(next(iter(bank["contexts"].values())).keys()),
        "pair_row": sorted(bank["pairs"][0].keys()),
        "n_contexts": len(bank["contexts"]),
        "n_pairs": len(bank["pairs"]),
    }
    shard = sorted(cfg.va_dir.glob("shard_*__ce__steered.pt"))[0]
    p = torch.load(shard, map_location="cpu", weights_only=False)
    out["va_shard"] = {
        "path": shard.name,
        "top": sorted(p.keys()),
        "index_row": sorted(p["index"][0].keys()),
        "va_span_shape": list(p["va_span"].shape),
        "layers": list(p["layers"]),
        "n_empty": len(p.get("empty_rows", [])),
    }
    anch = sorted(cfg.anchors_dir.glob("va_anchors_*.pt"))[0]
    p = torch.load(anch, map_location="cpu", weights_only=False)
    out["anchor_shard"] = {
        "path": anch.name,
        "top": sorted(p.keys()),
        "index_row": sorted(p["index"][0].keys()),
        "va_span_shape": list(p["va_span"].shape),
    }
    npz = np.load(cfg.in_root / MAPS_1739[0])
    meta = json.loads(str(npz["meta"]))
    out["map_1739_npz"] = {
        "arrays": sorted(npz.files),
        "meta_keys": sorted(meta.keys()),
        "fit_space": meta.get("fit_space"),
        "apply": meta.get("apply"),
        "layers": list(np.asarray(npz["layers"]).tolist()),
    }
    for tag, rel in (("ridge_779_L19", RIDGE_779[19]), ("ridge_1738ce_L19", RIDGE_1738[19])):
        p = torch.load(cfg.in_root / rel, map_location="cpu", weights_only=False)
        out[tag] = {
            "top": sorted(p.keys()),
            "kind": p.get("kind"),
            "W_shape": list(p["W"].shape) if "W" in p else None,
        }
    return out


# ── shared loaders ────────────────────────────────────────────────────


@dataclass
class Inputs:
    bank: dict
    pt: A2215.PairTable
    views: dict
    vc: dict  # layers + ce/pe stacks (A2215.load_vc_bank)
    fold_of_ctx: np.ndarray  # (n_ctx,) carrier index within the context's own cell
    anchor_mean: dict[str, torch.Tensor]  # full/h1/h2 -> (n, L, H) fp64
    n_valid: np.ndarray
    n_h1: np.ndarray
    n_h2: np.ndarray


def load_inputs(cfg: Cfg) -> Inputs:
    bank = json.loads(cfg.bank_json.read_text())
    pt = A2215.PairTable.from_bank(bank, None)
    views = A2215.build_cell_views(bank, pt)
    vc = A2215.load_vc_bank(cfg.vc_bank_pt, pt.ids)
    contexts = bank["contexts"]
    fold_of_ctx = np.full(len(pt.ids), -1, dtype=np.int64)
    for cell, cv in views.items():
        assert len(cv.carriers) == 12, (cell, cv.carriers)
        for r in cv.ctx_rows:
            cid = pt.ids[int(r)]
            carrier = contexts[cid].get("carrier") or cid.split("::")[-1]
            fold = cv.carriers.index(carrier)
            if fold_of_ctx[int(r)] not in (-1, fold):  # borrowed rev-cell membership
                assert fold_of_ctx[int(r)] == fold, (cid, fold_of_ctx[int(r)], fold)
            fold_of_ctx[int(r)] = fold
    assert (fold_of_ctx >= 0).all(), "unassigned fold for some context"
    files = sorted(cfg.anchors_dir.glob("va_anchors_*.pt"))
    sums, layers, n_valid, n_h1, n_h2 = A2215._accumulate_store(
        files, {"span": "va_span"}, set(pt.ids), pt.row_of, len(pt.ids), K_ANCHOR_DRAWS
    )
    assert list(layers) == list(vc["layers"]), (layers, vc["layers"])
    cnt = torch.tensor(np.maximum(n_valid, 1), dtype=torch.float64)[:, None, None]
    c1 = torch.tensor(np.maximum(n_h1, 1), dtype=torch.float64)[:, None, None]
    c2 = torch.tensor(np.maximum(n_h2, 1), dtype=torch.float64)[:, None, None]
    anchor_mean = {
        "full": sums[("span", "full")] / cnt,
        "h1": sums[("span", "h1")] / c1,
        "h2": sums[("span", "h2")] / c2,
    }
    logger.info(
        "[inputs] %d contexts / %d pairs / %d cells; anchor n_valid=0 contexts: %d",
        len(pt.ids),
        len(pt.pair_ids),
        len(pt.cells),
        int((n_valid == 0).sum()),
    )
    return Inputs(bank, pt, views, vc, fold_of_ctx, anchor_mean, n_valid, n_h1, n_h2)


def load_anchor_draw_rows(cfg: Cfg, pt) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Per-draw anchor rows: (ctx_row, draw, va_span fp16 (n_rows, L, H))."""
    ids_set = set(pt.ids)
    ctx_rows: list[int] = []
    draws: list[int] = []
    chunks: list[torch.Tensor] = []
    for shard in sorted(cfg.anchors_dir.glob("va_anchors_*.pt")):
        p = torch.load(shard, map_location="cpu", weights_only=False)
        empty = set(p.get("empty_rows", []))
        keep = [
            j
            for j, meta in enumerate(p["index"])
            if meta["context_id"] in ids_set and j not in empty
        ]
        if not keep:
            continue
        chunks.append(p["va_span"][torch.tensor(keep)].to(torch.float16))
        for j in keep:
            ctx_rows.append(pt.row_of[p["index"][j]["context_id"]])
            draws.append(int(p["index"][j]["draw"]))
        del p
    y16 = torch.cat(chunks, dim=0)
    del chunks
    logger.info("[anchor-rows] %d per-draw rows (fp16 %s)", y16.shape[0], tuple(y16.shape))
    return np.asarray(ctx_rows, dtype=np.int64), np.asarray(draws, dtype=np.int64), y16


# ── phase: fresh_fit ──────────────────────────────────────────────────


def _gcv_select(
    s: torch.Tensor, sq_a: torch.Tensor, tot: float, n_tr: int
) -> tuple[float, float, bool]:
    """GCV over LAMBDAS with the #825 dof cap. Returns (lambda, dof, edge_flag)."""
    best_lam, best_gcv, best_dof = None, float("inf"), None
    excluded = 0
    for lam in LAMBDAS:
        lam_f = float(lam)
        dof = float((s / (s + lam_f)).sum())
        if dof > GCV_DOF_CAP * n_tr:
            excluded += 1
            continue
        rss = tot - float((sq_a * (s + 2.0 * lam_f) / (s + lam_f) ** 2).sum())
        denom = (n_tr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_lam, best_gcv, best_dof = lam_f, gcv, dof
    assert excluded == 0, f"dof cap binding at n_tr={n_tr}, d={s.shape[0]} — unexpected at n>>d"
    assert best_lam is not None
    edge = best_lam in (float(LAMBDAS[0]), float(LAMBDAS[-1]))
    return best_lam, best_dof, edge


def phase_fresh_fit(cfg: Cfg, inp: Inputs) -> None:
    """Per-layer LOCO ridge fits via full-Gram downdates (recipe = primal GCV twin)."""
    logger.info("[phase=fresh_fit] layers=%s folds=%s", cfg.layers_sel, cfg.folds_sel)
    t_load = time.monotonic()
    ctx_row, _draws, y16 = load_anchor_draw_rows(cfg, inp.pt)
    anchor_load_s = time.monotonic() - t_load
    n_rows = int(ctx_row.shape[0])
    d = HIDDEN
    fold_of_row = inp.fold_of_ctx[ctx_row]
    layers = list(inp.vc["layers"])
    cfg.fresh_dir.mkdir(parents=True, exist_ok=True)
    cfg.shufmap_dir.mkdir(parents=True, exist_ok=True)
    diag_path = cfg.out_root / "fresh_fit_diagnostics.json"
    diag: dict = (
        json.loads(diag_path.read_text()) if diag_path.exists() else {"per_layer": {}, "meta": {}}
    )
    ctx_row_t = torch.tensor(ctx_row)
    t0 = time.monotonic()
    for k, layer in enumerate(cfg.layers_sel, 1):
        li = layers.index(layer)
        pred_path = cfg.fresh_dir / f"L{layer}.pt"
        sm_needed = layer in SHUF_MAP_LAYERS and cfg.shuf_map_draws > 0
        sm_ok = (not sm_needed) or (cfg.shufmap_dir / f"L{layer}.pt").exists()
        if pred_path.exists() and str(layer) in diag["per_layer"] and sm_ok:
            logger.info("[fresh-fit] unit %d/%d L%d resume-skip", k, len(cfg.layers_sel), layer)
            continue
        t_setup = time.monotonic()
        x_ctx = inp.vc["ce"][:, li, :].double()  # (n_ctx, d)
        x_rows = x_ctx[ctx_row_t]  # (n_rows, d) fp64
        y_rows = y16[:, li, :].double()
        g_full = x_rows.T @ x_rows
        xy_full = x_rows.T @ y_rows
        setup_s = time.monotonic() - t_setup
        sum_x = x_rows.sum(dim=0)
        sum_y = y_rows.sum(dim=0)
        sumsq_y = float((y_rows * y_rows).sum())
        oof_pred = torch.full((len(inp.pt.ids), d), float("nan"), dtype=torch.float64)
        idb_pred = torch.full_like(oof_pred, float("nan"))
        fold_rows_out: list[dict] = []
        for fold in cfg.folds_sel:
            tf = time.monotonic()
            hold_rows = np.where(fold_of_row == fold)[0]
            hold_ctx = np.where(inp.fold_of_ctx == fold)[0]
            assert hold_rows.size and hold_ctx.size, (layer, fold)
            hr = torch.tensor(hold_rows)
            xc = x_rows[hr]
            yc = y_rows[hr]
            n_tr = n_rows - int(hold_rows.size)
            mu = (sum_x - xc.sum(dim=0)) / n_tr
            ymu = (sum_y - yc.sum(dim=0)) / n_tr
            g_tr = g_full - xc.T @ xc
            xy_tr = xy_full - xc.T @ yc
            var = torch.clamp(torch.diagonal(g_tr) / n_tr - mu * mu, min=0.0)
            sd = torch.sqrt(var) + 1e-9  # population std + 1e-9 (primal-twin parity)
            gn = (g_tr - n_tr * torch.outer(mu, mu)) / torch.outer(sd, sd)
            bxy = (xy_tr - n_tr * torch.outer(mu, ymu)) / sd[:, None]
            s, v = torch.linalg.eigh(gn)
            s = torch.clamp(s, min=0.0)
            a = v.T @ bxy
            sq_a = (a * a).sum(dim=1)
            tot = (sumsq_y - float((yc * yc).sum())) - n_tr * float((ymu * ymu).sum())
            lam, dof, edge = _gcv_select(s, sq_a, tot, n_tr)
            w = v @ (a / (s + lam)[:, None])
            xen_ctx = (x_ctx[torch.tensor(hold_ctx)] - mu) / sd
            oof_pred[torch.tensor(hold_ctx)] = xen_ctx @ w + ymu
            idb_pred[torch.tensor(hold_ctx)] = x_ctx[torch.tensor(hold_ctx)] + (ymu - mu)
            xen_rows = (xc - mu) / sd
            pred_rows = xen_rows @ w + ymu
            r2_map = A2215.pooled_r2_cos(pred_rows.numpy(), yc.numpy())
            r2_idb = A2215.pooled_r2_cos((xc + (ymu - mu)).numpy(), yc.numpy())
            r2_id = A2215.pooled_r2_cos(xc.numpy(), yc.numpy())
            fold_rows_out.append(
                {
                    "layer": layer,
                    "fold": int(fold),
                    "n_train_rows": n_tr,
                    "n_hold_rows": int(hold_rows.size),
                    "d": d,
                    "lambda": lam,
                    "dof": dof,
                    "lambda_at_grid_edge": bool(edge),
                    "r2_map_perdraw": r2_map["r2_pooled"],
                    "cos_map_perdraw": r2_map["mean_cosine"],
                    "r2_idbias_perdraw": r2_idb["r2_pooled"],
                    "r2_identity_perdraw": r2_id["r2_pooled"],
                    "wall_s": time.monotonic() - tf,
                }
            )
            logger.info(
                "[fresh-fit] L%d fold %d/%d lam=%.3g dof=%.0f r2=%.4f elapsed=%.0fs",
                layer,
                fold,
                len(cfg.folds_sel),
                lam,
                dof,
                r2_map["r2_pooled"],
                time.monotonic() - tf,
            )
        # context-grain diagnostics (valid contexts with an OOF prediction)
        have = ~torch.isnan(oof_pred[:, 0]).numpy()
        valid = have & (inp.n_valid > 0)
        tgt = inp.anchor_mean["full"][:, li, :].numpy()
        ctx_diag = {
            "r2_map_ctx": A2215.pooled_r2_cos(oof_pred.numpy()[valid], tgt[valid])["r2_pooled"],
            "r2_idbias_ctx": A2215.pooled_r2_cos(idb_pred.numpy()[valid], tgt[valid])["r2_pooled"],
            "r2_identity_ctx": A2215.pooled_r2_cos(x_ctx.numpy()[valid], tgt[valid])["r2_pooled"],
            "knn": {
                metric: knn_retrieval(
                    oof_pred.numpy()[valid], tgt[valid], ks=(1, 5, 10), metric=metric
                )
                for metric in DV3_METRICS
            },
            "n_ctx_valid": int(valid.sum()),
        }
        torch.save(
            {
                "oof_pred": oof_pred.float(),
                "idbias_oof_pred": idb_pred.float(),
                "layer": layer,
                "folds": list(map(int, cfg.folds_sel)),
                "leg": cfg.leg,
            },
            pred_path,
        )
        # fresh shuffled-map null (context-permuted pairing; full-data fit, no LOCO)
        shufmap_s = 0.0
        if sm_needed:
            t_sm = time.monotonic()
            _fresh_shufmap(cfg, inp, layer, x_ctx, x_rows, y_rows, ctx_row)
            shufmap_s = time.monotonic() - t_sm
        diag["per_layer"][str(layer)] = {
            "folds": fold_rows_out,
            "context_grain": ctx_diag,
            "setup_s": round(setup_s, 2),
            "shufmap_s": round(shufmap_s, 2),
        }
        diag["meta"] = {
            "lambda_grid": [float(x) for x in LAMBDAS],
            "gcv_dof_cap": GCV_DOF_CAP,
            "lambda_selection": "gcv-dof-capped (#825 form; cap asserted non-binding)",
            "n_rows_total": n_rows,
            "d": d,
            "n_train_over_d": f"n_train≈{n_rows - n_rows // 12} > d={d} (well-posed)",
            "anchor_load_s": round(anchor_load_s, 1),
            "recipe": "primal GCV twin of issue_1739.fits.ridge_fit_predict_primal_layer_batched"
            " via full-Gram downdates (pilot parity-pinned)",
            "repro": _repro(),
        }
        _write_json_atomic(diag_path, diag)
        logger.info(
            "[fresh-fit] unit %d/%d L%d done elapsed=%.0fs",
            k,
            len(cfg.layers_sel),
            layer,
            time.monotonic() - t0,
        )
    logger.info("[phase=fresh_fit_done] %.0fs", time.monotonic() - t0)


def _fresh_shufmap(
    cfg: Cfg,
    inp: Inputs,
    layer: int,
    x_ctx: torch.Tensor,
    x_rows: torch.Tensor,
    y_rows: torch.Tensor,
    ctx_row: np.ndarray,
) -> None:
    """Shuffled-map null: refit on context-permuted (X, Y) pairing, full data.

    Y rows are permuted at CONTEXT grain within equal-draw-count classes, so X
    (hence the Gram + eig + standardization) is byte-identical across draws and
    only the cross-moment GEMM is paid per draw.
    """
    out_path = cfg.shufmap_dir / f"L{layer}.pt"
    if out_path.exists():
        logger.info("[shufmap] L%d resume-skip", layer)
        return
    t0 = time.monotonic()
    n_rows = x_rows.shape[0]
    n_ctx = len(inp.pt.ids)
    rows_of_ctx: list[list[int]] = [[] for _ in range(n_ctx)]
    for j, c in enumerate(ctx_row):
        rows_of_ctx[int(c)].append(j)
    counts = np.array([len(r) for r in rows_of_ctx])
    mu = x_rows.mean(dim=0)
    sd = x_rows.std(dim=0, unbiased=False) + 1e-9
    gn = ((x_rows - mu) / sd).T @ ((x_rows - mu) / sd)
    s, v = torch.linalg.eigh(gn)
    s = torch.clamp(s, min=0.0)
    preds = []
    perms = []
    for draw in range(cfg.shuf_map_draws):
        rng = np.random.default_rng([SEED, 7, layer, draw])
        perm = np.arange(n_ctx)
        for cnt in np.unique(counts):
            grp = np.where(counts == cnt)[0]
            if grp.size >= 2:
                perm[grp] = grp[A2215.deranged_perms(grp.size, 1, rng)[0]]
        row_perm = np.empty(n_rows, dtype=np.int64)
        for c in range(n_ctx):
            src = rows_of_ctx[int(perm[c])]
            dst = rows_of_ctx[c]
            assert len(src) == len(dst), (c, len(src), len(dst))
            row_perm[dst] = src
        y_perm = y_rows[torch.tensor(row_perm)]
        ymu = y_perm.mean(dim=0)
        bxy = (x_rows - mu).T @ (y_perm - ymu) / sd[:, None]
        a = v.T @ bxy
        sq_a = (a * a).sum(dim=1)
        tot = float(((y_perm - ymu) ** 2).sum())
        lam, _dof, _edge = _gcv_select(s, sq_a, tot, n_rows)
        w = v @ (a / (s + lam)[:, None])
        preds.append((((x_ctx - mu) / sd) @ w + ymu).to(torch.float16))
        perms.append(perm)
        logger.info("[shufmap] L%d draw %d/%d lam=%.3g", layer, draw + 1, cfg.shuf_map_draws, lam)
    torch.save(
        {"preds": torch.stack(preds), "perms": np.stack(perms), "layer": layer, "leg": cfg.leg},
        out_path,
    )
    logger.info("[shufmap] L%d done %.0fs", layer, time.monotonic() - t0)


def pilot_parity_pin(cfg: Cfg, inp: Inputs) -> dict:
    """Pilot-only: my downdate fit vs the canonical primal helper on ONE (layer, fold)."""
    layer, fold = cfg.layers_sel[0], cfg.folds_sel[0]
    li = list(inp.vc["layers"]).index(layer)
    ctx_row, _draws, y16 = load_anchor_draw_rows(cfg, inp.pt)
    fold_of_row = inp.fold_of_ctx[ctx_row]
    tr = np.where(fold_of_row != fold)[0]
    hold_ctx = np.where(inp.fold_of_ctx == fold)[0]
    x_ctx = inp.vc["ce"][:, li, :].double()
    x_tr = x_ctx[torch.tensor(ctx_row[tr])].numpy()[None]
    y_tr = y16[torch.tensor(tr), li, :].double().numpy()[None]
    x_ev = x_ctx[torch.tensor(hold_ctx)].numpy()[None]
    helper_pred = F1739.ridge_fit_predict_primal_layer_batched(
        x_tr, y_tr, x_ev, lambdas=LAMBDAS, device=cfg.device
    )[0]
    mine = torch.load(cfg.fresh_dir / f"L{layer}.pt", map_location="cpu", weights_only=False)
    my_pred = mine["oof_pred"][torch.tensor(hold_ctx)].double().numpy()
    denom = np.maximum(np.abs(helper_pred), 1e-9)
    max_rel = float(np.max(np.abs(my_pred - helper_pred) / denom))
    med_rel = float(np.median(np.abs(my_pred - helper_pred) / denom))
    ok = bool(np.allclose(my_pred, helper_pred, rtol=1e-4, atol=1e-6))
    rec = {"layer": layer, "fold": int(fold), "max_rel": max_rel, "median_rel": med_rel, "ok": ok}
    assert ok, f"parity pin FAILED vs ridge_fit_predict_primal_layer_batched: {rec}"
    logger.info("[parity-pin] OK max_rel=%.2e median_rel=%.2e", max_rel, med_rel)
    return rec


# ── map sources (Result 2.5 + dv3ext arms) ────────────────────────────


def _cos_rows(p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    num = (p * r).sum(dim=-1)
    return num / (p.norm(dim=-1) * r.norm(dim=-1) + _EPS)


def _apply_ridge_payload(payload: dict, x: torch.Tensor, w_override=None) -> torch.Tensor:
    """The #779 apply_map ridge path (fp64 standardize @ W + ymu), W overridable."""
    assert payload.get("kind") == "ridge", payload.get("kind")
    xmu = payload["xmu"].double()
    xsd = payload["xsd"].double()
    ymu = payload["ymu"].double()
    w = (payload["W"] if w_override is None else w_override).double()
    return ((x.double() - xmu) / xsd) @ w + ymu


def load_1739_gate(cfg: Cfg, rel: str) -> tuple[dict | None, str]:
    """Load a #1739 npz map ONLY if its meta declares a raw fit space (#1975 parity).

    The persisted apply contract for these payloads reads "(whitened space)"; with
    no whitening artifact banked under issue1739_ctxmap, a whitened-fit map cannot
    be applied to raw #2162 bank states — DROP with the reason recorded.
    """
    npz = np.load(cfg.in_root / rel)
    meta = json.loads(str(npz["meta"]))
    fit_space = meta.get("fit_space")
    if fit_space != "raw":
        reason = (
            f"dropped: fit_space={fit_space!r} (apply contract: {meta.get('apply')!r}); "
            "whitening artifact not banked under issue1739_ctxmap — #1975 input-space "
            "parity cannot be satisfied on raw bank states"
        )
        return None, reason
    return {
        "w": npz["w"],
        "x_mu": npz["x_mu"],
        "x_sd": npz["x_sd"],
        "y_mu": npz["y_mu"],
        "layers": [int(x) for x in np.asarray(npz["layers"]).tolist()],
        "meta": meta,
    }, "included: fit_space=raw"


@dataclass
class MapSource:
    name: str
    layers: list[int]
    pred: dict[int, torch.Tensor]  # layer -> (n_ctx, H) fp32 predicted v_A
    null_pred: dict[int, list[torch.Tensor]]  # layer -> [K x (n_ctx, H) fp16] shuffled-map
    null_kind: str  # "refit-permuted-pairing" | "row-permuted-weights" | "none"
    fitted: bool = True


def build_map_sources(cfg: Cfg, inp: Inputs) -> tuple[list[MapSource], dict]:
    """Predicted v_A per (source, layer) for all bank contexts + shuffled-map nulls."""
    import issue779_ffc_n1m_fits as FITS  # deferred heavy sibling import

    layers_all = list(inp.vc["layers"])
    notes: dict = {}
    sources: list[MapSource] = []

    # fresh OOF (leg-scoped layers)
    fresh_pred: dict[int, torch.Tensor] = {}
    fresh_null: dict[int, list[torch.Tensor]] = {}
    for layer in cfg.layers_sel:
        p = torch.load(cfg.fresh_dir / f"L{layer}.pt", map_location="cpu", weights_only=False)
        fresh_pred[layer] = p["oof_pred"]
        sm = cfg.shufmap_dir / f"L{layer}.pt"
        if sm.exists():
            fresh_null[layer] = list(
                torch.load(sm, map_location="cpu", weights_only=False)["preds"]
            )
    sources.append(
        MapSource("fresh", list(fresh_pred), fresh_pred, fresh_null, "refit-permuted-pairing")
    )

    # identity / identity+bias baseline in SHIFT space (bias cancels in differences;
    # recorded once as `ctxshift`).
    ctx_pred = {layer: inp.vc["ce"][:, layers_all.index(layer), :].float() for layer in layers_all}
    sources.append(MapSource("ctxshift", layers_all, ctx_pred, {}, "none", fitted=False))
    notes["ctxshift"] = "identity shift baseline; identity+bias ≡ identity in shift space"

    rng_seeds = list(range(cfg.shuf_map_draws))
    for name, table in (("m779ce", RIDGE_779), ("m1738ce", RIDGE_1738)):
        pred: dict[int, torch.Tensor] = {}
        null: dict[int, list[torch.Tensor]] = {}
        try:
            for layer, rel in table.items():
                payload = torch.load(cfg.in_root / rel, map_location="cpu", weights_only=False)
                assert payload.get("kind") == "ridge", (rel, payload.get("kind"))
                x = inp.vc["ce"][:, layers_all.index(layer), :]
                # live path = the #779 apply_map; the local mirror (used only for the
                # W-permuted null, which apply_map cannot express) is parity-asserted
                # against it on every layer.
                p_live = torch.tensor(
                    FITS.apply_map(payload, x.double().numpy(), torch.device(cfg.device))
                )
                p_mine = _apply_ridge_payload(payload, x)
                if not torch.allclose(p_mine, p_live, rtol=1e-6, atol=1e-6):
                    # fail LOUD (deliberately not the dropped-arm AssertionError path)
                    raise RuntimeError(
                        f"null-apply mirror diverges from apply_map: {name} L{layer}"
                    )
                pred[layer] = p_live.float()
                nulls = []
                for seed in rng_seeds:
                    rng = np.random.default_rng([SEED, 13, layer, seed])
                    perm = torch.tensor(rng.permutation(payload["W"].shape[0]))
                    nulls.append(
                        _apply_ridge_payload(payload, x, w_override=payload["W"][perm]).to(
                            torch.float16
                        )
                    )
                null[layer] = nulls
        except (AssertionError, KeyError) as exc:
            notes[name] = f"dropped: payload contract mismatch ({exc!r})"
            continue
        sources.append(MapSource(name, list(pred), pred, null, "row-permuted-weights"))
        notes[name] = "included: banked ridge bundle (FITS.apply_map ridge path)"

    for rel in MAPS_1739:
        tag = "m1739_" + rel.rsplit("__", 1)[-1].removesuffix(".npz")
        bundle, note = load_1739_gate(cfg, rel)
        notes[tag] = note
        if bundle is None:
            continue
        m = F1739.MapFit(
            w=bundle["w"],
            x_mu=bundle["x_mu"],
            x_sd=bundle["x_sd"],
            y_mu=bundle["y_mu"],
            diagnostics={},
        )
        lay = bundle["layers"]
        x_all = np.stack([inp.vc["ce"][:, layers_all.index(layer), :].numpy() for layer in lay])
        pred_all = F1739.apply_map(x_all, m)
        pred = {
            layer: torch.tensor(pred_all[j], dtype=torch.float32) for j, layer in enumerate(lay)
        }
        null: dict[int, list[torch.Tensor]] = {}
        keep = [j for j, layer in enumerate(lay) if layer in SHUF_MAP_LAYERS]
        if keep:
            w32 = np.asarray(bundle["w"][keep], dtype=np.float32)
            for seed in rng_seeds:
                wp = F1739.shuffled_map_weights(w32, seed=seed)
                sub = F1739.MapFit(
                    w=wp,
                    x_mu=bundle["x_mu"][keep],
                    x_sd=bundle["x_sd"][keep],
                    y_mu=bundle["y_mu"][keep],
                    diagnostics={},
                )
                pn = F1739.apply_map(x_all[keep], sub)
                for jj, j in enumerate(keep):
                    null.setdefault(lay[j], []).append(torch.tensor(pn[jj], dtype=torch.float16))
        sources.append(MapSource(tag, lay, pred, null, "row-permuted-weights"))
    return sources, notes


# ── phase: shift (Result 2.5) ─────────────────────────────────────────


def _load_cell_patched(cfg: Cfg, cell: str, arm: str) -> dict[str, tuple[torch.Tensor, int]]:
    """pair_id -> (per-layer span-mean patched v_A (L, H) fp32, n_draws_kept)."""
    path = cfg.va_dir / f"shard_{cell}__ce__{arm}.pt"
    p = torch.load(path, map_location="cpu", weights_only=False)
    empty = set(p.get("empty_rows", []))
    acc: dict[str, list[int]] = {}
    for j, meta in enumerate(p["index"]):
        if j in empty:
            continue
        acc.setdefault(meta["pair_id"], []).append(j)
    out = {}
    va = p["va_span"]
    for pair_id, rows in acc.items():
        out[pair_id] = (va[torch.tensor(rows)].double().mean(dim=0).float(), len(rows))
    return out


def survivors_from_two_by_two() -> tuple[set[str], dict[str, dict]]:
    d = json.loads(TWO_BY_TWO_JSON.read_text())
    verdicts = {r["cell"]: r for r in d["cells"] if r["slot"] == "ce"}
    surv = {c for c, r in verdicts.items() if r.get("causal_verdict") == "positive"}
    return surv, verdicts


def phase_shift(cfg: Cfg, inp: Inputs) -> None:
    logger.info("[phase=shift]")
    t0 = time.monotonic()
    sources, notes = build_map_sources(cfg, inp)
    survivors, verdicts = survivors_from_two_by_two()
    cells = cfg.cells_sel or inp.pt.cells
    done_path = cfg.out_root / "shift_done_cells.json"
    done: dict = json.loads(done_path.read_text()) if done_path.exists() else {"cells": []}
    rows_path = cfg.out_root / "shift_cells.jsonl"
    all_rows: list[dict] = []
    if rows_path.exists():
        all_rows = [json.loads(x) for x in rows_path.read_text().split("\n") if x.strip()]
    layers_all = list(inp.vc["layers"])
    li_of = {layer: layers_all.index(layer) for layer in layers_all}

    for k, cell in enumerate(cells, 1):
        if cell in done["cells"]:
            logger.info("[shift] unit %d/%d %s resume-skip", k, len(cells), cell)
            continue
        cv = inp.views[cell]
        pairs_k = [int(x) for x in cv.pair_idx]
        pair_ids = [inp.pt.pair_ids[x] for x in pairs_k]
        a_rows = inp.pt.a_row[pairs_k]
        b_rows = inp.pt.b_row[pairs_k]
        if cfg.leg == "pilot":  # pilot fresh preds exist only for the selected folds
            keep = np.isin(inp.fold_of_ctx[a_rows], cfg.folds_sel)
        else:
            keep = np.ones(len(pairs_k), dtype=bool)
        keep &= (inp.n_h1[a_rows] > 0) & (inp.n_h2[a_rows] > 0) & (inp.n_valid[b_rows] > 0)
        patched = {arm: _load_cell_patched(cfg, cell, arm) for arm in PATCH_ARMS}
        cell_rows: list[dict] = []
        for layer in layers_all:
            li = li_of[layer]
            floor_h1 = inp.anchor_mean["h1"][a_rows, li, :]  # (n_p, H) fp64
            floor_h2 = inp.anchor_mean["h2"][a_rows, li, :]
            ceil_b = inp.anchor_mean["full"][b_rows, li, :]
            ceiling = (ceil_b - floor_h2).float()
            realized: dict[str, torch.Tensor] = {}
            arm_keep: dict[str, np.ndarray] = {}
            for arm in PATCH_ARMS:
                mat = torch.zeros((len(pairs_k), HIDDEN))
                ok = keep.copy()
                for j, pid in enumerate(pair_ids):
                    ent = patched[arm].get(pid)
                    if ent is None:
                        ok[j] = False
                        continue
                    mat[j] = ent[0][li]
                realized[arm] = mat - floor_h1.float()
                arm_keep[arm] = ok
            for src in sources:
                if layer not in src.pred:
                    continue
                pred_shift = (src.pred[layer][b_rows] - src.pred[layer][a_rows]).float()
                for arm in PATCH_ARMS:
                    ok = arm_keep[arm]
                    if ok.sum() < 2:
                        continue
                    okt = torch.tensor(np.where(ok)[0])
                    p, r, c = pred_shift[okt], realized[arm][okt], ceiling[okt]
                    cos = _cos_rows(p, r)
                    resid = float(((p - r) ** 2).sum())
                    tot = float(((r - r.mean(dim=0)) ** 2).sum())
                    # shuffled-pair null: score the DERANGED same-vp different-carrier
                    # pair's predicted shift against this pair's realized shift (the
                    # #2215 dv3 derangement convention); batched indexing, no per-draw
                    # GEMM.
                    rng = np.random.default_rng(
                        [SEED, 3, layers_all.index(layer), inp.pt.cells.index(cell)]
                    )
                    sigma = A2215.deranged_perms(len(cv.carriers), cfg.shuf_pair_b, rng)
                    q = cv.pair_at[sigma[:, cv.carrier_loc], cv.vp_loc]  # (B, n_p)
                    pn = pred_shift / (pred_shift.norm(dim=-1, keepdim=True) + _EPS)
                    rn = realized[arm] / (realized[arm].norm(dim=-1, keepdim=True) + _EPS)
                    cmat = (pn @ rn.T).numpy()  # cos(pred_i, realized_j)
                    j_idx = np.arange(len(pairs_k))
                    m = ok[None, :] & ok[q]  # (B, n_p) own + donor validity
                    vals = cmat[q, j_idx[None, :]]
                    n_ok = m.sum(axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        null_draws = np.where(
                            n_ok >= 2, (vals * m).sum(axis=1) / np.maximum(n_ok, 1), np.nan
                        )
                    # shuffled-map null (where available for this source/layer)
                    sm_band = None
                    if src.null_pred.get(layer):
                        vals = []
                        for np_pred in src.null_pred[layer]:
                            ps = (np_pred[b_rows] - np_pred[a_rows]).float()[okt]
                            vals.append(float(_cos_rows(ps, r).mean()))
                        sm_band = [float(np.min(vals)), float(np.max(vals)), len(vals)]
                    cell_rows.append(
                        {
                            "cell": cell,
                            "layer": layer,
                            "source": src.name,
                            "arm": arm,
                            "n_pairs": int(ok.sum()),
                            "mean_cos": float(cos.mean()),
                            "median_cos": float(cos.median()),
                            "mag_ratio_median": float(
                                (p.norm(dim=-1) / (r.norm(dim=-1) + _EPS)).median()
                            ),
                            "r2_shift": 1.0 - resid / tot if tot > 0 else float("nan"),
                            "mean_cos_vs_ceiling": float(_cos_rows(p, c).mean()),
                            "mean_cos_realized_vs_ceiling": float(_cos_rows(r, c).mean()),
                            "null_shufpair_band": [
                                A2215._pct(null_draws, 2.5),
                                A2215._pct(null_draws, 97.5),
                            ],
                            "null_shufmap_minmax": sm_band,
                            "null_kind": src.null_kind,
                            "survivor": cell in survivors,
                            "causal_verdict": verdicts.get(cell, {}).get("causal_verdict"),
                            "probe_verdict": verdicts.get(cell, {}).get("probe_verdict"),
                        }
                    )
        all_rows.extend(cell_rows)
        done["cells"].append(cell)
        _write_jsonl_atomic(rows_path, all_rows)
        _write_json_atomic(done_path, done)
        logger.info(
            "[shift] unit %d/%d %s rows=%d elapsed=%.0fs",
            k,
            len(cells),
            cell,
            len(cell_rows),
            time.monotonic() - t0,
        )
    summary = _shift_summary(all_rows, survivors)
    summary["source_notes"] = notes
    summary["repro"] = _repro()
    _write_json_atomic(cfg.out_root / "shift_summary.json", summary)
    logger.info("[phase=shift_done] %d rows, %.0fs", len(all_rows), time.monotonic() - t0)


def _shift_summary(rows: list[dict], survivors: set[str]) -> dict:
    out: dict = {"views": {}}
    for view, pred in (("survivors", lambda r: r["survivor"]), ("all_cells", lambda r: True)):
        agg: dict[str, dict] = {}
        for r in rows:
            if not pred(r):
                continue
            key = f"{r['source']}|L{r['layer']}|{r['arm']}"
            a = agg.setdefault(key, {"cos": [], "r2": [], "ceil": [], "n": 0})
            a["cos"].append(r["mean_cos"])
            a["r2"].append(r["r2_shift"])
            a["ceil"].append(r["mean_cos_realized_vs_ceiling"])
            a["n"] += r["n_pairs"]
        out["views"][view] = {
            k: {
                "mean_cos_over_cells": float(np.mean(v["cos"])),
                "mean_r2_shift": float(np.nanmean(v["r2"])),
                "mean_cos_realized_vs_ceiling": float(np.mean(v["ceil"])),
                "n_cells": len(v["cos"]),
                "n_pairs": v["n"],
            }
            for k, v in sorted(agg.items())
        }
    out["survivor_cells"] = sorted(survivors)
    return out


# ── phase: dv3ext (Result 4 extension) ────────────────────────────────


def phase_dv3ext(cfg: Cfg, inp: Inputs) -> None:
    """#2215 dv3 2AFC at span pooling with fresh / identity-only / idbias / m779ce arms."""
    from issue2094_analysis import bootstrap_family_means_batched  # deferred

    logger.info("[phase=dv3ext]")
    t0 = time.monotonic()
    sources, _notes = build_map_sources(cfg, inp)
    by_name = {s.name: s for s in sources}
    layers_all = list(inp.vc["layers"])
    valid = inp.n_valid > 0
    pt, views = inp.pt, inp.views
    cells = cfg.cells_sel or pt.cells

    arms: list[dict] = []
    fresh = by_name["fresh"]
    arms.append({"arm": "freshce", "layers": fresh.layers, "pred": fresh.pred, "fitted": True})
    arms.append(
        {
            "arm": "identity_ce",
            "layers": layers_all,
            "pred": by_name["ctxshift"].pred,
            "fitted": False,
        }
    )
    if "m779ce" in by_name:
        arms.append(
            {
                "arm": "m779ce",
                "layers": by_name["m779ce"].layers,
                "pred": by_name["m779ce"].pred,
                "fitted": True,
            }
        )
    # idbias arm: LOTO identity+bias per layer (the #2215 dv3 baseline convention)
    idb_layers = sorted({layer for a in arms for layer in a["layers"]})
    idb_pred: dict[int, torch.Tensor] = {}
    for layer in idb_layers:
        li = layers_all.index(layer)
        x = inp.vc["ce"][:, li, :].double().numpy()
        t = inp.anchor_mean["full"][:, li, :].numpy()
        idb_pred[layer] = torch.tensor(
            A2215.idbias_loto_predict(x, t, pt.cell_of, valid), dtype=torch.float32
        )
    arms.append({"arm": "idbias_ce", "layers": idb_layers, "pred": idb_pred, "fitted": False})

    configs = [
        (a["arm"], layer, metric) for a in arms for layer in a["layers"] for metric in DV3_METRICS
    ]
    cfg_index = {c: i for i, c in enumerate(configs)}
    clusters = [(cell, carrier) for cell in cells for carrier in views[cell].carriers]
    cluster_index = {kc: i for i, kc in enumerate(clusters)}
    acc_cl = np.full((len(clusters), len(configs)), np.nan)
    per_config: dict[str, dict] = {}
    cell_draws = {}
    for ci, cell in enumerate(cells):
        cv = views[cell]
        rng = np.random.default_rng([SEED, 5, ci])
        cell_draws[cell] = {
            "sigma": A2215.deranged_perms(len(cv.carriers), cfg.dv3_null_b, rng),
            "side_a": rng.integers(0, 2, size=(cfg.dv3_null_b, len(cv.pair_idx))).astype(bool),
            "side_b": rng.integers(0, 2, size=(cfg.dv3_null_b, len(cv.pair_idx))).astype(bool),
        }
    unit, n_units = 0, sum(len(a["layers"]) for a in arms)
    for a in arms:
        for layer in a["layers"]:
            unit += 1
            li = layers_all.index(layer)
            p_np = a["pred"][layer].double().numpy()
            t_np = inp.anchor_mean["full"][:, li, :].numpy()
            pred_have = ~np.isnan(p_np[:, 0])
            v_all = valid & pred_have
            key = f"{a['arm']}|L{layer}|span"
            stats = A2215.pooled_r2_cos(p_np[v_all], t_np[v_all])
            knn = {
                m: knn_retrieval(p_np[v_all], t_np[v_all], ks=(1, 5, 10), metric=m)
                for m in DV3_METRICS
            }
            cell_rows: dict[str, dict] = {}
            pooled_bits = {m: [0.0, 0.0] for m in DV3_METRICS}
            null_correct = {m: np.zeros(cfg.dv3_null_b) for m in DV3_METRICS}
            null_total = {m: np.zeros(cfg.dv3_null_b) for m in DV3_METRICS}
            for cell in cells:
                cv = views[cell]
                loc = cv.ctx_rows
                vp_valid = v_all[loc][cv.a_loc] & v_all[loc][cv.b_loc]
                if not vp_valid.any():
                    cell_rows[cell] = {"na": "N/A — no valid pairs (pred/anchor coverage)"}
                    continue
                s_by = A2215.sim_blocks(p_np[loc], t_np[loc])
                for metric in DV3_METRICS:
                    s = s_by[metric]
                    m_a, m_b = A2215.observed_2afc(s, cv.a_loc, cv.b_loc)
                    bits = np.concatenate([(m_a > 0)[vp_valid], (m_b > 0)[vp_valid]])
                    for car_i, carrier in enumerate(cv.carriers):
                        sel = (cv.carrier_loc == car_i) & vp_valid
                        if not sel.any():
                            continue
                        cbits = np.concatenate([(m_a > 0)[sel], (m_b > 0)[sel]])
                        acc_cl[
                            cluster_index[(cell, carrier)],
                            cfg_index[(a["arm"], layer, metric)],
                        ] = float(cbits.mean())
                    nc, nt = A2215.null_2afc_cell(
                        s,
                        cv,
                        cell_draws[cell]["sigma"],
                        cell_draws[cell]["side_a"],
                        cell_draws[cell]["side_b"],
                        vp_valid,
                    )
                    null_correct[metric] += nc
                    null_total[metric] += nt
                    cell_rows.setdefault(cell, {})[metric] = {
                        "acc": float(bits.mean()),
                        "n_pairs_included": int(vp_valid.sum()),
                    }
                    pooled_bits[metric][0] += float(bits.sum())
                    pooled_bits[metric][1] += float(len(bits))
            pooled = {}
            for metric in DV3_METRICS:
                correct, total = pooled_bits[metric]
                with np.errstate(invalid="ignore", divide="ignore"):
                    null_acc = np.where(
                        null_total[metric] > 0, null_correct[metric] / null_total[metric], np.nan
                    )
                pooled[metric] = {
                    "acc": correct / total if total else float("nan"),
                    "n_pair_dirs": int(total),
                    "null_band": [A2215._pct(null_acc, 2.5), A2215._pct(null_acc, 97.5)],
                }
            per_config[key] = {
                "arm": a["arm"],
                "layer": layer,
                "pooling": "span",
                "fitted": a["fitted"],
                **stats,
                "knn": knn,
                "pooled": pooled,
                "per_type": cell_rows,
            }
            logger.info(
                "[dv3ext] unit %d/%d %s elapsed=%.0fs", unit, n_units, key, time.monotonic() - t0
            )

    # carrier-clustered bootstrap CIs + diffs vs idbias
    diff_cols: list[str] = []
    diff_vals: list[np.ndarray] = []
    for a in arms:
        if a["arm"] == "idbias_ce":
            continue
        for layer in a["layers"]:
            for metric in DV3_METRICS:
                fi = cfg_index[(a["arm"], layer, metric)]
                bi = cfg_index[("idbias_ce", layer, metric)]
                diff_cols.append(f"{a['arm']}-minus-idbias_ce|L{layer}|span|{metric}")
                diff_vals.append(acc_cl[:, fi] - acc_cl[:, bi])
    families = np.concatenate(
        [acc_cl] + ([np.stack(diff_vals, axis=1)] if diff_vals else []), axis=1
    )
    draws = bootstrap_family_means_batched(families, cfg.dv3_boot_b, SEED + 1)
    n_cfg = len(configs)
    for i, (arm_name, layer, metric) in enumerate(configs):
        rec = per_config[f"{arm_name}|L{layer}|span"]["pooled"][metric]
        ci95 = [A2215._pct(draws[:, i], 2.5), A2215._pct(draws[:, i], 97.5)]
        rec["acc_ci95_clustered"] = ci95
        rec["verdict"] = A2215.discrimination_verdict(rec["acc"], ci95)
    diffs = {}
    for j, label in enumerate(diff_cols):
        col = n_cfg + j
        ci95 = [A2215._pct(draws[:, col], 2.5), A2215._pct(draws[:, col], 97.5)]
        verdict = "inconclusive"
        if all(np.isfinite(ci95)):
            verdict = (
                "beats-baseline"
                if ci95[0] > 0
                else ("below-baseline" if ci95[1] < 0 else "inconclusive")
            )
        diffs[label] = {
            "mean_cluster_diff": float(np.nanmean(diff_vals[j])),
            "ci95_clustered": ci95,
            "verdict": verdict,
        }
    banked_ref = _banked_dv3_reference()
    out = {
        "meta": {
            "pooling": "span (banked va_anchors store; #2215 dv3 span convention)",
            "metrics": list(DV3_METRICS),
            "arms": [{"arm": a["arm"], "layers": a["layers"], "fitted": a["fitted"]} for a in arms],
            "null": f"carrier-blocked derangement + side randomization (seed {SEED}, "
            f"B={cfg.dv3_null_b})",
            "bootstrap": f"carrier-clustered, B={cfg.dv3_boot_b}, seed {SEED + 1}",
            "config_order": [f"{c[0]}|L{c[1]}|span|{c[2]}" for c in configs],
        },
        "per_config": per_config,
        "diff_vs_idbias": diffs,
        "banked_dv3_span_reference": banked_ref,
        "repro": _repro(),
    }
    _write_json_atomic(cfg.out_root / "dv3_ext.json", out)
    logger.info("[phase=dv3ext_done] %.0fs", time.monotonic() - t0)


def _banked_dv3_reference() -> dict:
    """Committed #2215 dv3 span rows for the shared arms — the parity anchor."""
    if not BANKED_DV3_JSON.exists():
        return {"note": f"{BANKED_DV3_JSON} missing"}
    d = json.loads(BANKED_DV3_JSON.read_text())
    out = {}
    for key, rec in d["per_config"].items():
        if key.endswith("|span") and rec["arm"] in ("779ce", "idbias_ce", "1738ce"):
            out[key] = {
                m: {
                    "acc": rec["pooled"][m]["acc"],
                    "ci95": rec["pooled"][m].get("acc_ci95_clustered"),
                }
                for m in rec["pooled"]
            }
    return out


# ── phase: digest ─────────────────────────────────────────────────────


def phase_digest(cfg: Cfg, walls: dict[str, float], extra: dict) -> None:
    digest = {
        "leg": cfg.leg,
        "hf_revision": HF_REVISION,
        "walls_s": walls,
        "n_train_over_d": "per-fold n_train ≈ 12.9k PER-DRAW rows > d = 3584 (well-posed)",
        **extra,
        "repro": _repro(),
    }
    _write_json_atomic(cfg.out_root / "run_digest.json", digest)
    logger.info("[phase=digest_done] -> %s", cfg.out_root / "run_digest.json")


def pilot_projection(cfg: Cfg, walls: dict[str, float]) -> dict:
    """Project the FULL-leg wall from the pilot's MEASURED unit walls."""
    diag = json.loads((cfg.out_root / "fresh_fit_diagnostics.json").read_text())
    layer_rec = next(iter(diag["per_layer"].values()))
    fold_s = float(np.median([f["wall_s"] for f in layer_rec["folds"]]))
    setup_s = float(layer_rec["setup_s"])
    shufmap_pilot_s = float(layer_rec["shufmap_s"])
    anchor_load_s = float(diag["meta"]["anchor_load_s"])
    n_layers, n_folds = N_MODEL_LAYERS, 12
    # shufmap full = 5 draws vs the pilot's 2, at 3 layers (eigh amortized inside)
    shufmap_full_s = (
        len(SHUF_MAP_LAYERS) * shufmap_pilot_s * (SHUF_MAP_DRAWS / max(1, cfg.shuf_map_draws))
    )
    fresh_full_s = anchor_load_s + n_layers * setup_s + n_layers * n_folds * fold_s + shufmap_full_s
    # battery/dv3ext scale: cells 2 -> 39; null B 100 -> 1000 (indexing-bound, ~linear
    # in the sub-dominant term); fresh source 1 -> 28 layers. x30 is a deliberate
    # over-estimate umbrella for both.
    shift_full_s = walls.get("shift", 0.0) * 30
    dv3_full_s = walls.get("dv3ext", 0.0) * 30
    total = fresh_full_s + shift_full_s + dv3_full_s
    return {
        "measured": {
            "fold_s_median": fold_s,
            "layer_setup_s": setup_s,
            "anchor_load_s": anchor_load_s,
            "shufmap_pilot_s": shufmap_pilot_s,
            "shift_pilot_s": walls.get("shift"),
            "dv3ext_pilot_s": walls.get("dv3ext"),
        },
        "projected_full_s": {
            "fresh_fit": round(fresh_full_s),
            "shift_upper": round(shift_full_s),
            "dv3ext_upper": round(dv3_full_s),
            "total": round(total),
        },
        "projected_full_h": round(total / 3600, 2),
        "arithmetic": f"{anchor_load_s:.0f}s load + 28x{setup_s:.0f}s setup + "
        f"336x{fold_s:.0f}s folds + {shufmap_full_s:.0f}s shufmap + batteries",
    }


# ── main ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase", default="all", choices=["all", "stage", "fresh_fit", "shift", "dv3ext", "digest"]
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="1 layer x 1 fold + 2-cell battery")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument(
        "--in-root",
        type=Path,
        default=Path("/mnt/eps-data/thomasjiralerspong/issue2162_mapshift/hf_dl"),
    )
    ap.add_argument(
        "--work-root",
        type=Path,
        default=Path("/mnt/eps-data/thomasjiralerspong/issue2162_mapshift/work"),
    )
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "eval_results/issue_2162/mapshift")
    ap.add_argument("--device", default="cpu")
    return ap.parse_args(argv)


def build_cfg(args: argparse.Namespace) -> Cfg:
    leg = "pilot" if args.pilot else "full"
    cfg = Cfg(
        leg=leg,
        in_root=args.in_root,
        work_root=args.work_root / leg,
        out_root=args.out_root / "pilot" if leg == "pilot" else args.out_root,
        device=args.device,
    )
    if leg == "pilot":
        cfg.layers_sel = [19]
        cfg.folds_sel = [0]
        cfg.shuf_pair_b = 100
        cfg.shuf_map_draws = 2
        cfg.dv3_null_b = 200
        cfg.dv3_boot_b = 500
    else:
        cfg.layers_sel = list(range(N_MODEL_LAYERS))
        cfg.folds_sel = list(range(12))
    return cfg


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.import_check:
        _import_check()
        raise SystemExit(0)
    assert args.pilot or args.full, "pass --pilot or --full"
    cfg = build_cfg(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    walls: dict[str, float] = {}
    extra: dict = {}

    def timed(name: str, fn, *a) -> None:
        t = time.monotonic()
        fn(*a)
        walls[name] = round(time.monotonic() - t, 1)

    if args.phase in ("all", "stage"):
        timed("stage", phase_stage, cfg)
    inp = None
    if args.phase in ("all", "fresh_fit", "shift", "dv3ext"):
        inp = load_inputs(cfg)
        # pin the fit layers to the bank's realized layer list
        if cfg.leg == "full":
            cfg.layers_sel = list(inp.vc["layers"])
    if args.phase in ("all", "fresh_fit"):
        timed("fresh_fit", phase_fresh_fit, cfg, inp)
        if cfg.leg == "pilot":
            extra["parity_pin"] = pilot_parity_pin(cfg, inp)
    if args.phase in ("all", "shift"):
        if cfg.leg == "pilot":
            surv, _ = survivors_from_two_by_two()
            first_surv = sorted(surv)[0] if surv else inp.pt.cells[0]
            other = next(c for c in inp.pt.cells if c != first_surv)
            cfg.cells_sel = [first_surv, other]
        timed("shift", phase_shift, cfg, inp)
    if args.phase in ("all", "dv3ext"):
        if cfg.leg == "pilot":
            surv, _ = survivors_from_two_by_two()
            first_surv = sorted(surv)[0] if surv else inp.pt.cells[0]
            other = next(c for c in inp.pt.cells if c != first_surv)
            cfg.cells_sel = [first_surv, other]
        timed("dv3ext", phase_dv3ext, cfg, inp)
    if args.phase in ("all", "digest"):
        if cfg.leg == "pilot" and (cfg.out_root / "fresh_fit_diagnostics.json").exists():
            extra["pilot_projection"] = pilot_projection(cfg, walls)
            logger.info(
                "[pilot-projection] %s", json.dumps(extra["pilot_projection"]["projected_full_s"])
            )
        phase_digest(cfg, walls, extra)
    sys.stdout.flush()
    sys.stderr.flush()
    logger.info("[phase=done] leg=%s walls=%s", cfg.leg, walls)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
