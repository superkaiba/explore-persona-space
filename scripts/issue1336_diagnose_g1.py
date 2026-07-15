#!/usr/bin/env python
"""Issue #1336 — D1 diagnosis driver for the sub-null G1 R^2 (plan v7 amendment).

Zero-GPU re-analysis of the wave-1 artifacts plus the D1.4/D1.6 refit battery
(GPU-leg capable via the fc._fit_device()-parametrized #825 cores). One step
selector, canonical order enforced regardless of CLI order (DG1: qwen_cal
BEFORE battery BEFORE verdict):

  stage     D1.0 scoped HF staging to --stage-root (turnstores, preds npz,
            rollout text; Qwen stems stream-reduced one shard at a time).
            Row-count (3,629) + prompt-id asserts vs preds_manifest/npz.
  decomp    D1.1 per-dim SS_res/SS_tot decomposition of stored preds
            (+ fp64-path cross-check when battery v0 preds exist).
  audit     D1.2 per-dim scale stats Llama vs Qwen + top-64 discreteness.
  spotcheck D1.3 re-render/re-tokenize n rows/cell; slot/span mismatch count.
  qwen_cal  D1.6 layer-19 standardized fit on the reduced Qwen tensors;
            sets bar_std BEFORE any Llama verdict read (DG1 ordering).
  battery   D1.4 refit battery: v0 committed convention (DG0 gate +/-0.02),
            v1 widened lambda grid, v2 per-dim standardized Y, v3 trimmed,
            v4 fp64 end-to-end, 20 standardized selection-symmetric shuffle
            draws (per-draw layer-max persisted), 1,000-draw pred-resampling
            bootstrap. collect_lambdas everywhere. Checkpoint per
            (cell, pass, layer-block); resume keyed on a regime fingerprint.
  verdict   D1.5 mechanism attribution A_v + lattice inputs (S/B/bar_std/
            S'/D+CI) + routed decision R1-R5 -> diagnosis_verdict.json.

All heavy fits reuse the #825 Gram-ridge cores verbatim
(`_prep_fold`/`_ridge_predict_cached`/`heldout_r2_sweep` with the
default-preserving `lambdas=` kwarg added at the source, #931 pattern).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) bind BEFORE torch/numpy import

import issue825_fit_cells as fc  # noqa: E402
import issue1336_fit_cells as f36  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

# ---------------------------------------------------------------------------
# Registered diagnostic constants (plan v7 SS10/SS11 "Diagnostic fit constants")
# ---------------------------------------------------------------------------
DIAG_CELLS = ("rlvr_chat_lmsys5k", "rlvr_naturalistic_lmsys5k")
LAMBDAS_WIDE = np.logspace(-2, 8, 21)  # v1+ grid (Source: plan SS11, derived)
STD_FLOOR_FRAC = 1e-3  # per-dim std floor = 1e-3 x fold-median dim std
TRIM_LADDER = (4, 41, 410)  # ~0.1/1/10% of 4,096 (diagnostic ladder)
N_NULL_STD = 20  # standardized selection-symmetric shuffle draws
N_BOOT = 1_000  # prompt-level pred-resampling bootstrap
SPOTCHECK_N = 50  # rows/cell (>=6% defect detected w.p. >=95%)
SPOTCHECK_DEFECT_RATE = 0.01  # capture-defect gate threshold (plan SS4 D1.3)
A_V_BAR = 0.8  # mechanism-account threshold (sensitivity 0.6/0.9 reported)
A_V_SENSITIVITY = (0.6, 0.9)
DG0_TOL = 0.02  # plan SS7 DG0
# Committed wave-1 G1 values (gates/g1_gate.json @ 6939726b57, plan SS2).
DG0_TARGETS = {
    "rlvr_chat_lmsys5k": -0.9287,
    "rlvr_naturalistic_lmsys5k": -0.8942,
}
# Committed raw shuffle band (layer-max p97.5, chat cell nulls @ 6939726b57);
# constant fallback when the committed nulls JSON is unreachable.
COMMITTED_RAW_BAND_FALLBACK = -0.0225
BAR_RAW = 0.20  # v3 G1 usable-strength bar (raw scale)
QWEN_COMMITTED_R2 = 0.6731  # #825 S1 layer-19 anchor (G0-validated)
QWEN_CAL_DEV_MAX = 0.1  # |S_qwen - 0.6731| > 0.1 -> corrected DV exploratory (R5)
QWEN_SLOT_INDEX = 0  # parent Track-S slot convention (fc G0 path)
QWEN_TURN_INDEX = 1
EXPECT_N_ROWS = 3_629  # realized wave-1 n (plan SS2; asserted at stage)
L_COMMITTED_ARGMAX = 29  # committed chat argmax layer (plan SS2)
VERDICT_LAYERS = (16, 21, 22, 29, 30)  # frozen U {29}
LAYER_BLOCK = 8  # battery checkpoint grain (cell, pass, layer-block)
PLANNED_BATTERY_WALL_H = 1.0  # plan SS9 GPU-leg wall; pilot aborts > 2x
PREDS_STEM = {  # committed preds npz stems on the data repo
    "rlvr_chat_lmsys5k": "preds_rlvr_chat_lmsys5k.npz",
    "rlvr_naturalistic_lmsys5k": "preds_rlvr_naturalistic_lmsys5k.npz",
}
STEP_ORDER = ("stage", "decomp", "audit", "spotcheck", "qwen_cal", "battery", "verdict")


def _metadata(seed: int, n: int) -> dict:
    return {
        "git_commit": fc._git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "n": int(n),
        "script": "scripts/issue1336_diagnose_g1.py",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[diag1336] wrote {path}", flush=True)


def _cell_spec(cell_id: str) -> dict:
    assert cell_id in f36.CELL_BY_ID, f"unknown cell id {cell_id!r}"
    return f36.CELL_BY_ID[cell_id]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", required=True, help="comma list from " + ",".join(STEP_ORDER))
    ap.add_argument("--cells", default=",".join(DIAG_CELLS), help="comma cell ids")
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/mnt/eps-data/issue_1336_diag"),
        help="staging root (VM: /mnt/eps-data — NEVER the VM root disk; GPU leg: local)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1336/diagnosis"))
    ap.add_argument("--turnstore-dir", type=Path, default=None, help="override (smoke fixtures)")
    ap.add_argument("--preds-dir", type=Path, default=None, help="override (smoke fixtures)")
    ap.add_argument("--gen-dir", type=Path, default=None, help="override (smoke fixtures)")
    ap.add_argument("--qwen-reduced", type=Path, default=None, help="override (smoke fixtures)")
    ap.add_argument("--tokenizer-dir", default=None, help="spotcheck tokenizer override (smoke)")
    ap.add_argument("--spotcheck-n", type=int, default=SPOTCHECK_N)
    ap.add_argument("--null-draws", type=int, default=N_NULL_STD)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--folds", type=int, default=cm.N_FOLDS)
    ap.add_argument("--seed", type=int, default=cm.FIT_SEED)
    ap.add_argument(
        "--dg0-targets-json",
        default=None,
        help="JSON {cell_id: r2} DG0 target override (smoke oracle); default = committed values",
    )
    ap.add_argument(
        "--committed-eval-dir",
        type=Path,
        default=Path("eval_results/issue_1336"),
        help="committed wave-1 cells/nulls JSONs (fs, then `git show HEAD:` fallback)",
    )
    ap.add_argument("--no-pilot-abort", action="store_true", help="report-only pilot projection")
    ap.add_argument(
        "--expect-n",
        type=int,
        default=EXPECT_N_ROWS,
        help="stage-time per-cell row-count assert (production 3,629; smoke fixtures override)",
    )
    ap.add_argument("--wall-budget-h", type=float, default=PLANNED_BATTERY_WALL_H)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# D1.0 — staging
# ---------------------------------------------------------------------------
def _hub_helpers():
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    return HfApi(), hf_hub_download, hub


def _stage_prefix(api, hub, hf_hub_download, prefix: str, dest: Path, revision: str = "main"):
    """Scoped list_repo_tree + per-file hf_hub_download (never snapshot_download
    on the ~1M-file data repo — gotchas.md #833); listing materialized inside
    the retry thunk (lazy-generator gotcha #779)."""
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: scoped (path_in_repo) walk inside hub.retry_transient
            api.list_repo_tree(
                cm.HF_DATA_REPO,
                path_in_repo=prefix,
                repo_type="dataset",
                revision=revision,
                recursive=True,
            )
        ),
        what=f"diag stage: tree walk {prefix}",
    )
    files = [e.path for e in entries if hasattr(e, "size")]
    assert files, f"no files under {prefix} on {cm.HF_DATA_REPO}"
    for rel in sorted(files):
        hub.retry_transient(
            lambda r=rel: hf_hub_download(
                repo_id=cm.HF_DATA_REPO,
                repo_type="dataset",
                filename=r,
                revision=revision,
                local_dir=dest,
            ),
            what=f"diag stage: download {rel}",
        )
    return [dest / f for f in sorted(files)]


def _ts_dir(args, cell_id: str) -> Path:
    if args.turnstore_dir is not None:
        return args.turnstore_dir
    return args.stage_root / f"turnstore_{cell_id}"


def _preds_dir(args) -> Path:
    return args.preds_dir if args.preds_dir is not None else args.stage_root / "preds"


def _gen_dir(args) -> Path:
    return args.gen_dir if args.gen_dir is not None else args.stage_root / "gen"


def _qwen_reduced_path(args) -> Path:
    if args.qwen_reduced is not None:
        return args.qwen_reduced
    return args.stage_root / "qwen_reduced" / "qwen_s1_reduced.pt"


def _load_preds_npz(args, cell_id: str) -> dict:
    path = _preds_dir(args) / PREDS_STEM.get(cell_id, f"preds_{cell_id}.npz")
    assert path.exists(), f"preds npz missing: {path}"
    return dict(np.load(path, allow_pickle=False))


def _assert_row_parity(cell_id: str, ts_conv: np.ndarray, preds: dict, expect_n: int | None):
    """Fail-loud row-count + prompt-id equality (plan D1.0)."""
    p_conv = np.asarray(preds["conv_ids"]).astype(str)
    t_conv = np.asarray(ts_conv).astype(str)
    assert len(t_conv) == len(p_conv), (
        f"{cell_id}: turnstore rows {len(t_conv)} != preds rows {len(p_conv)}"
    )
    assert (t_conv == p_conv).all(), f"{cell_id}: prompt-id mismatch turnstore vs preds npz"
    if expect_n is not None:
        assert len(t_conv) == expect_n, f"{cell_id}: n={len(t_conv)} != expected {expect_n}"


def step_stage(args) -> None:
    print("[diag1336] step=stage", flush=True)
    api, dl, hub = _hub_helpers()
    root = args.stage_root
    root.mkdir(parents=True, exist_ok=True)
    # Turnstores + preds + rollout text (wave-1 rlvr lmsys cells only).
    for cell_id in args.cell_ids:
        prefix = f"{cm.HF_PREFIX_1336}/analysis_tensors/turnstore_{cell_id}"
        dest = root / f"turnstore_{cell_id}"
        if any(dest.glob("*.pt")):
            print(f"[stage] {dest} already staged — skipping")
        else:
            staged = _stage_prefix(api, hub, dl, prefix, root)
            # hf_hub_download(local_dir=root) mirrors the hub-rel path; move flat.
            dest.mkdir(parents=True, exist_ok=True)
            for f in staged:
                f.rename(dest / f.name)
        n_pt = len(list(dest.glob("*.pt")))
        assert n_pt > 0, f"{cell_id}: staged 0 .pt shards"
    preds_dest = root / "preds"
    if not any(preds_dest.glob("*.npz")):
        staged = _stage_prefix(
            api, hub, dl, f"{cm.HF_PREFIX_1336}/analysis_tensors/preds/cells", root
        )
        preds_dest.mkdir(parents=True, exist_ok=True)
        for f in staged:
            f.rename(preds_dest / f.name)
    gen_dest = root / "gen"
    if not (gen_dest / "rlvr" / "lmsys5k" / "answers.jsonl").exists():
        staged = _stage_prefix(
            api, hub, dl, f"{cm.HF_PREFIX_1336}/raw_completions/generation/rlvr/lmsys5k", root
        )
        target = gen_dest / "rlvr" / "lmsys5k"
        target.mkdir(parents=True, exist_ok=True)
        for f in staged:
            f.rename(target / f.name)
        _maybe_reassemble_answers(target)
    # Row-count + prompt-id asserts (fail loud, plan D1.0).
    for cell_id in args.cell_ids:
        spec = _cell_spec(cell_id)
        bundle = fc._load_bundle_any(
            _ts_dir(args, cell_id), spec["model"], spec["format"], spec["corpus"]
        )
        _assert_row_parity(
            cell_id,
            np.asarray(bundle["sidecar"]["conv_ids"]),
            _load_preds_npz(args, cell_id),
            args.expect_n,
        )
        print(f"[stage] {cell_id}: row parity OK (n={args.expect_n})")
    _stage_qwen_reduce(args, api, dl, hub)


def _maybe_reassemble_answers(gen_dir: Path) -> None:
    """Reassemble shard-split answers.jsonl (gen-phase >9.5MB shard contract)."""
    manifest = gen_dir / "answers.manifest.json"
    if (gen_dir / "answers.jsonl").exists() or not manifest.exists():
        return
    import hashlib

    m = json.loads(manifest.read_text())
    tmp = gen_dir / "answers.jsonl.tmp"
    h = hashlib.sha256()
    with tmp.open("wb") as out:
        for part in m["parts"]:
            data = (gen_dir / part).read_bytes()
            h.update(data)
            out.write(data)
    assert h.hexdigest() == m["total_sha256"], "reassembled answers.jsonl sha mismatch"
    tmp.replace(gen_dir / "answers.jsonl")
    print(f"[stage] reassembled answers.jsonl from {len(m['parts'])} parts")


def _stage_qwen_reduce(args, api, dl, hub) -> None:
    """Stream-reduce the pinned Qwen S1 stems ONE SHARD AT A TIME (D1.0/D1.6).

    Peak transient ~= one 2.14 GB shard; reduced output = bf16 X/Y at all 28
    layers (~2 GB). Shard files are deleted after reduction (re-downloadable).
    """
    out = _qwen_reduced_path(args)
    if out.exists():
        print(f"[stage] qwen reduced tensors already at {out} — skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: scoped walk inside hub.retry_transient
            api.list_repo_tree(
                cm.HF_DATA_REPO,
                path_in_repo=cm.G0["hf_prefix"],
                repo_type="dataset",
                revision=cm.G0["revision"],
                recursive=False,
            )
        ),
        what="diag stage: qwen tree walk",
    )
    stem = cm.G0["stem"]
    shards = sorted(
        e.path
        for e in entries
        if Path(e.path).name.startswith(f"{stem}_shard") and e.path.endswith(".pt")
    )
    assert shards, f"no {stem} shards under {cm.G0['hf_prefix']} @ {cm.G0['revision'][:8]}"
    xs, ys, conv_ids = [], [], []
    scratch = out.parent / "shard_scratch"
    for rel in shards:
        local = hub.retry_transient(
            lambda r=rel: dl(
                repo_id=cm.HF_DATA_REPO,
                repo_type="dataset",
                filename=r,
                revision=cm.G0["revision"],
                local_dir=scratch,
            ),
            what=f"diag stage: qwen shard {rel}",
        )
        payload = torch.load(local, map_location="cpu", weights_only=False)
        conv_ids.extend(str(c) for c in payload.get("conv_ids", []))
        xs.append(torch.stack([t[QWEN_SLOT_INDEX] for t in payload["slots"]]).to(torch.bfloat16))
        ys.append(torch.stack([t[QWEN_TURN_INDEX] for t in payload["profiles"]]).to(torch.bfloat16))
        del payload
        Path(local).unlink()  # delete-after-reduce: peak transient = one shard
        print(f"[stage] reduced qwen shard {rel} (rows so far {sum(x.shape[0] for x in xs)})")
    X = torch.cat(xs)
    Y = torch.cat(ys)
    assert X.shape == Y.shape and X.shape[1] == cm.G0["expected_layers"], (X.shape, Y.shape)
    torch.save({"X": X, "Y": Y, "conv_ids": conv_ids}, out)
    print(f"[stage] qwen reduced tensors -> {out} (X {tuple(X.shape)} bf16)")


# ---------------------------------------------------------------------------
# Shared loading for decomp/audit/battery
# ---------------------------------------------------------------------------
def _load_cell_xy(args, cell_id: str) -> dict:
    spec = _cell_spec(cell_id)
    ts_dir = _ts_dir(args, cell_id)
    bundle = fc._load_bundle_any(ts_dir, spec["model"], spec["format"], spec["corpus"])
    exp_layers = f36._bundle_n_layers(bundle)
    xy = f36._cell_xy_1336(bundle, exp_layers)
    return {"X": xy["X"], "Y": xy["Y"], "conv_ids": np.asarray(xy["conv_ids"]).astype(str)}


def _load_xy_fp64(args, cell_id: str, layers: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """bf16 -> fp64 with NO fp32 waypoint (battery v4), sliced to `layers`."""
    spec = _cell_spec(cell_id)
    ts_dir = _ts_dir(args, cell_id)
    stem = f"{spec['model']}_{spec['format']}_{spec['corpus']}"
    shards = sorted(ts_dir.glob(f"{stem}*.pt"))
    assert shards, f"no shards {stem}*.pt in {ts_dir}"
    xs, ys = [], []
    lt = torch.tensor(layers, dtype=torch.long)
    for sp in shards:
        payload = torch.load(sp, map_location="cpu", weights_only=False)
        for s, p in zip(payload["slots"], payload["profiles"], strict=True):
            xs.append(s[1].index_select(0, lt).to(torch.float64))  # a1 slot
            ys.append(p[1].index_select(0, lt).to(torch.float64))  # a1 profile
        del payload
    return torch.stack(xs).numpy(), torch.stack(ys).numpy()


def _verdict_layers_for(n_layers: int) -> list[int]:
    vl = [li for li in VERDICT_LAYERS if li < n_layers]
    return vl if vl else [n_layers - 1]


# ---------------------------------------------------------------------------
# D1.1 — per-dim decomposition of stored predictions
# ---------------------------------------------------------------------------
def _perdim_from_preds(
    preds: np.ndarray, Y_l: np.ndarray, folds: np.ndarray, fitted: np.ndarray
) -> dict:
    """Per-dim SS_res/SS_tot pooled over folds (fold-local test mean — the
    committed pooled-R^2 convention), plus the D1.1 read battery."""
    D = Y_l.shape[1]
    ss_res = np.zeros(D)
    ss_tot = np.zeros(D)
    fold_bias = []
    cos_centered, cos_uncentered = [], []
    for k in np.unique(folds):
        te = (folds == k) & fitted
        if te.sum() == 0:
            continue
        true = Y_l[te].astype(np.float64)
        pred = preds[te].astype(np.float64)
        mu = true.mean(0)
        ss_res += ((true - pred) ** 2).sum(0)
        ss_tot += ((true - mu) ** 2).sum(0)
        fold_bias.append((pred - true).mean(0))
        tr = folds != k
        mu_tr = Y_l[tr].astype(np.float64).mean(0)
        cos_centered.append(fc._per_example_cosine(pred - mu_tr, true - mu_tr))
        cos_uncentered.append(fc._per_example_cosine(pred, true))
    excess = ss_res - ss_tot
    order_excess = np.argsort(excess)[::-1]
    order_var = np.argsort(ss_tot)[::-1]
    tot_excess = float(excess.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        perdim_r2 = 1.0 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)

    def _excess_share(k: int) -> float:
        if tot_excess <= 0:
            return float("nan")
        return float(excess[order_excess[:k]].sum() / tot_excess)

    def _trimmed_r2(k: int) -> float:
        keep = np.ones(D, dtype=bool)
        keep[order_var[: min(k, D - 1)]] = False
        return float(1.0 - ss_res[keep].sum() / ss_tot[keep].sum())

    # Per-dim affine recalibration (pooled-global convention; separates gain
    # error from irreducible noise): fit true ~ a*pred + b per dim.
    m = fitted
    P = preds[m].astype(np.float64)
    T = Y_l[m].astype(np.float64)
    pm, tm = P.mean(0), T.mean(0)
    var_p = ((P - pm) ** 2).mean(0)
    cov = ((P - pm) * (T - tm)).mean(0)
    a = np.where(var_p > 1e-12, cov / np.maximum(var_p, 1e-12), 0.0)
    resid = T - (a * P + (tm - a * pm))
    ss_tot_g = ((T - tm) ** 2).sum()
    recal_r2 = float(1.0 - (resid**2).sum() / ss_tot_g)
    cum_var = np.cumsum(ss_tot[order_var]) / max(ss_tot.sum(), 1e-12)
    cc = np.concatenate(cos_centered) if cos_centered else np.array([np.nan])
    cu = np.concatenate(cos_uncentered) if cos_uncentered else np.array([np.nan])
    fb = np.stack(fold_bias) if fold_bias else np.zeros((0, D))
    finite = perdim_r2[np.isfinite(perdim_r2)]
    return {
        "arrays": {
            "ss_res": ss_res,
            "ss_tot": ss_tot,
            "perdim_r2": perdim_r2,
            "fold_bias": fb,
        },
        "summary": {
            "pooled_r2": float(1.0 - ss_res.sum() / ss_tot.sum()),
            "total_excess": tot_excess,
            "excess_share_top": {str(k): _excess_share(k) for k in (1, 10, 100)},
            "var_share_top": {
                str(k): float(cum_var[min(k, D) - 1]) for k in (1, 4, 10, 41, 100, 410) if k <= D
            },
            "trimmed_r2_stored_preds": {str(k): _trimmed_r2(k) for k in TRIM_LADDER},
            "perdim_r2_median": float(np.median(finite)) if len(finite) else float("nan"),
            "perdim_r2_frac_positive": float((finite > 0).mean()) if len(finite) else float("nan"),
            "perdim_r2_quantiles": {
                q: float(np.quantile(finite, float(q))) if len(finite) else float("nan")
                for q in ("0.05", "0.25", "0.5", "0.75", "0.95")
            },
            "uniform_mean_perdim_r2": float(finite.mean()) if len(finite) else float("nan"),
            "affine_recalibrated_r2_pooled_global": recal_r2,
            "cosine_centered_mean": float(np.nanmean(cc)),
            "cosine_uncentered_mean": float(np.nanmean(cu)),
            "fold_bias_max_abs": float(np.abs(fb).max()) if fb.size else float("nan"),
            "fold_bias_l2_per_fold": [float(np.linalg.norm(r)) for r in fb],
        },
    }


def step_decomp(args) -> None:
    print("[diag1336] step=decomp", flush=True)
    for cell_id in args.cell_ids:
        xy = _load_cell_xy(args, cell_id)
        preds = _load_preds_npz(args, cell_id)
        _assert_row_parity(cell_id, xy["conv_ids"], preds, None)
        folds = preds["folds"]
        fitted = preds["fitted_mask"].astype(bool)
        layer_keys = sorted(
            (int(k.split("_l")[1]) for k in preds if k.startswith("preds_l")),
        )
        per_layer = {}
        npz_arrays = {}
        for li in layer_keys:
            res = _perdim_from_preds(
                preds[f"preds_l{li}"].astype(np.float32), xy["Y"][:, li, :], folds, fitted
            )
            per_layer[str(li)] = res["summary"]
            for name, arr in res["arrays"].items():
                npz_arrays[f"{name}_l{li}"] = arr.astype(np.float32)
        out_npz = args.out_dir / "tensors" / f"perdim_{cell_id}.npz"
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz, **npz_arrays)  # plain savez: client compression OFF for Xet (#813)
        payload = {
            "metadata": _metadata(args.seed, int(fitted.sum())),
            "cell_id": cell_id,
            "preds_source": "committed_fp16",
            "layers": layer_keys,
            "per_layer": per_layer,
            "arrays_npz": str(out_npz),
        }
        # fp64-path cross-check: battery v0 preds (fp32-persisted from fp64
        # fold math) override the fp16 committed copies when present.
        v0_npz = args.out_dir / "tensors" / f"battery_v0_preds_{cell_id}.npz"
        if v0_npz.exists():
            v0 = dict(np.load(v0_npz, allow_pickle=False))
            cross = {}
            for li in layer_keys:
                key = f"preds_l{li}"
                if key not in v0:
                    continue
                res = _perdim_from_preds(
                    v0[key].astype(np.float32),
                    xy["Y"][:, li, :],
                    v0["folds"],
                    v0["fitted_mask"].astype(bool),
                )
                cross[str(li)] = {
                    "pooled_r2_fp64path": res["summary"]["pooled_r2"],
                    "pooled_r2_fp16": per_layer[str(li)]["pooled_r2"],
                    "abs_dev": abs(res["summary"]["pooled_r2"] - per_layer[str(li)]["pooled_r2"]),
                    "excess_share_top_fp64path": res["summary"]["excess_share_top"],
                }
            payload["fp64_cross_check"] = cross
        _write_json(args.out_dir / f"perdim_decomp_{cell_id}.json", payload)


# ---------------------------------------------------------------------------
# D1.2 — scale audit + capture-precision read
# ---------------------------------------------------------------------------
def _dim_stats(M: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "std": M.std(0),
        "max_abs": np.abs(M).max(0),
        "median_abs": np.median(np.abs(M), 0),
    }


def _discreteness(M: np.ndarray, dims: np.ndarray) -> list[dict]:
    out = []
    for d in dims:
        vals = np.unique(M[:, d])
        gaps = np.diff(vals)
        out.append(
            {
                "dim": int(d),
                "n_unique": len(vals),
                "n_rows": int(M.shape[0]),
                "median_grid_gap": float(np.median(gaps)) if len(gaps) else 0.0,
                "value_range": [float(vals[0]), float(vals[-1])],
            }
        )
    return out


def step_audit(args) -> None:
    print("[diag1336] step=audit", flush=True)
    report: dict = {"metadata": _metadata(args.seed, 0), "cells": {}}
    for cell_id in args.cell_ids:
        xy = _load_cell_xy(args, cell_id)
        n_layers = xy["Y"].shape[1]
        layers = _verdict_layers_for(n_layers)
        npz_arrays = {}
        cell_rep: dict = {"layers": layers, "per_layer": {}}
        for li in layers:
            Yl = xy["Y"][:, li, :].astype(np.float32)
            Xl = xy["X"][:, li, :].astype(np.float32)
            ys, xs = _dim_stats(Yl), _dim_stats(Xl)
            for k, v in ys.items():
                npz_arrays[f"Y_{k}_l{li}"] = v
            for k, v in xs.items():
                npz_arrays[f"X_{k}_l{li}"] = v
            var_order = np.argsort(ys["std"])[::-1]
            top64 = var_order[:64]
            med_std = float(np.median(ys["std"]))
            cell_rep["per_layer"][str(li)] = {
                "y_std_median": med_std,
                "y_std_top1": float(ys["std"][var_order[0]]),
                "y_std_top1_over_median": float(ys["std"][var_order[0]] / max(med_std, 1e-12)),
                "y_max_abs_global": float(ys["max_abs"].max()),
                "top64_dims": [int(d) for d in top64],
                "top64_discreteness": _discreteness(Yl, top64),
            }
        out_npz = args.out_dir / "tensors" / f"scale_{cell_id}.npz"
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz, **npz_arrays)
        cell_rep["arrays_npz"] = str(out_npz)
        report["cells"][cell_id] = cell_rep
    qwen_path = _qwen_reduced_path(args)
    if qwen_path.exists():
        qq = torch.load(qwen_path, map_location="cpu", weights_only=False)
        li = min(int(cm.G0["layer"]), qq["Y"].shape[1] - 1)  # fixture clamp
        Yq = qq["Y"][:, li, :].float().numpy()
        ys = _dim_stats(Yq)
        var_order = np.argsort(ys["std"])[::-1]
        med_std = float(np.median(ys["std"]))
        report["qwen_s1"] = {
            "layer": li,
            "y_std_median": med_std,
            "y_std_top1": float(ys["std"][var_order[0]]),
            "y_std_top1_over_median": float(ys["std"][var_order[0]] / max(med_std, 1e-12)),
            "y_max_abs_global": float(ys["max_abs"].max()),
            "top64_discreteness": _discreteness(Yq, var_order[:16]),
        }
    else:
        report["qwen_s1"] = None
        print("[audit] qwen reduced tensors absent — Llama-only audit")
    # Keep-floor context note (committed wave-1 audit fact, plan SS2/D1.2 —
    # no new compute; the Llama-tokenizer filter-calibration note for resume).
    report["keep_floor_note"] = (
        "wave-1 keep 72.6% < 0.80 floor, dominated by chat:short_turn_u1 (1,039 drops); "
        "committed gen audit @ 6939726b57 (eval_results/issue_1336/gen_audits/)"
    )
    report["capture_corruption_suspected"] = False  # flipped only by human adjudication
    _write_json(args.out_dir / "scale_audit.json", report)


# ---------------------------------------------------------------------------
# D1.3 — slot/span spot-check (H-B)
# ---------------------------------------------------------------------------
def _spans_meta_by_conv(ts_dir: Path, stem: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sp in sorted(ts_dir.glob(f"{stem}*.pt")):
        payload = torch.load(sp, map_location="cpu", weights_only=False, mmap=True)
        for meta in payload.get("spans_meta", []):
            out[str(meta["conv_id"])] = meta
        del payload
    assert out, f"no spans_meta in {ts_dir}/{stem}*.pt"
    return out


def step_spotcheck(args) -> None:
    print("[diag1336] step=spotcheck", flush=True)
    from issue1336_render import RENDERERS
    from transformers import AutoTokenizer

    report: dict = {"metadata": _metadata(args.seed, args.spotcheck_n), "cells": {}}
    tok_cache: dict[str, object] = {}
    for cell_id in args.cell_ids:
        spec = _cell_spec(cell_id)
        tok_id = args.tokenizer_dir or cm.MODELS[spec["model"]]["hf_id"]
        if tok_id not in tok_cache:  # module-scope cache (#664 per-probe 429 trap)
            tok_cache[tok_id] = AutoTokenizer.from_pretrained(tok_id)
        tokenizer = tok_cache[tok_id]
        gen_dir = _gen_dir(args) / spec["model"] / spec["corpus"]
        answers_path = gen_dir / "answers.jsonl"
        assert answers_path.exists(), f"rollout text missing: {answers_path}"
        rows = {}
        with answers_path.open(encoding="utf-8") as fh:  # never splitlines() on JSONL (#950)
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    rows[str(r["prompt_idx"])] = r
        stem = f"{spec['model']}_{spec['format']}_{spec['corpus']}"
        metas = _spans_meta_by_conv(_ts_dir(args, cell_id), stem)
        rng = np.random.default_rng(args.seed)
        conv_ids = sorted(metas)
        pick = rng.choice(len(conv_ids), size=min(args.spotcheck_n, len(conv_ids)), replace=False)
        mismatches, details = 0, []
        for i in sorted(int(v) for v in pick):
            cid = conv_ids[i]
            meta = metas[cid]
            row = rows.get(cid)
            entry: dict = {"conv_id": cid}
            if row is None or not row.get("kept", True):
                mismatches += 1
                entry["mismatch"] = "rollout_row_missing_or_dropped"
                details.append(entry)
                continue
            conv = {"conv_id": cid, "u1": row["prompt"], "a1": row["response"]}
            rendered = RENDERERS[spec["format"]](conv, tokenizer)
            got_slots = {k: int(v) for k, v in rendered.slot_idx.items()}
            got_spans = {k: [int(s), int(e)] for k, (s, e) in rendered.spans.items()}
            ok = (
                got_slots == {k: int(v) for k, v in meta["slot_idx"].items()}
                and got_spans == {k: [int(s), int(e)] for k, (s, e) in meta["spans"].items()}
                and len(rendered.input_ids) == int(meta["seq_len"])
            )
            # Decoded +/-5-token windows around the stored offsets (report-only
            # convention read; the equality check above is the gate input).
            a1_slot = int(meta["slot_idx"]["a1"])
            ids = rendered.input_ids
            win = tokenizer.decode(ids[max(0, a1_slot - 5) : a1_slot + 1])
            s, e = meta["spans"]["a1"]
            span_head = tokenizer.decode(ids[s : min(e, s + 5)]) if e <= len(ids) else ""
            hdr = "Assistant:" if spec["format"] == "naturalistic" else "<|assistant|>"
            convention_ok = hdr.split(":")[0].strip("<|>").lower() in win.lower()
            if not ok:
                mismatches += 1
                entry["mismatch"] = "render_offsets_differ"
                entry["stored"] = {"slot_idx": meta["slot_idx"], "spans": meta["spans"]}
                entry["rerendered"] = {"slot_idx": got_slots, "spans": got_spans}
            entry["a1_slot_window"] = win
            entry["a1_span_head"] = span_head[:80]
            entry["assistant_header_in_window"] = bool(convention_ok)
            details.append(entry)
        rate = mismatches / max(len(pick), 1)
        report["cells"][cell_id] = {
            "n_sampled": len(pick),
            "mismatches": int(mismatches),
            "mismatch_rate": float(rate),
            "defect_gate_fired": bool(rate > SPOTCHECK_DEFECT_RATE),
            "details": details,
        }
        print(f"[spotcheck] {cell_id}: {mismatches}/{len(pick)} mismatches (rate {rate:.3f})")
    report["defect_threshold"] = SPOTCHECK_DEFECT_RATE
    report["any_defect_gate_fired"] = any(c["defect_gate_fired"] for c in report["cells"].values())
    _write_json(args.out_dir / "spotcheck.json", report)


# ---------------------------------------------------------------------------
# Standardization helper (v2 corrected DV) — train-fold stats, floored std
# ---------------------------------------------------------------------------
def _std_stats(Y_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Y_tr.mean(0)
    sd = Y_tr.std(0)
    floor = STD_FLOOR_FRAC * max(float(np.median(sd)), 1e-30)
    return mu, np.maximum(sd, floor)


# ---------------------------------------------------------------------------
# D1.6 — Qwen standardized calibration (healthy-data control; sets bar_std)
# ---------------------------------------------------------------------------
def step_qwen_cal(args) -> None:
    print("[diag1336] step=qwen_cal", flush=True)
    path = _qwen_reduced_path(args)
    assert path.exists(), f"qwen reduced tensors missing: {path} (run --steps stage first)"
    qq = torch.load(path, map_location="cpu", weights_only=False)
    li = min(int(cm.G0["layer"]), qq["X"].shape[1] - 1)
    # bf16 -> fp32 waypoint mirrors the committed G0 load path (bit-comparable
    # to the 0.6731 anchor); the cores upcast fp64 inside _prep_fold.
    X = qq["X"][:, li, :].float().numpy()
    Y = qq["Y"][:, li, :].float().numpy()
    conv_ids = np.asarray([str(c) for c in qq["conv_ids"]])
    folds = fc._cv_folds(conv_ids, args.folds, args.seed)
    ss_res_raw = ss_tot_raw = ss_res_z = ss_tot_z = 0.0
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        cache = fc._prep_fold(X[tr], X[te])
        pred_raw = fc._ridge_predict_cached(cache, Y[tr])  # committed grid
        true = Y[te].astype(np.float64)
        mu_te = true.mean(0)
        ss_res_raw += float(((true - pred_raw) ** 2).sum())
        ss_tot_raw += float(((true - mu_te) ** 2).sum())
        mu, sd = _std_stats(Y[tr])
        pred_z = fc._ridge_predict_cached(cache, (Y[tr] - mu) / sd, lambdas=LAMBDAS_WIDE)
        true_z = ((Y[te] - mu) / sd).astype(np.float64)
        mu_te_z = true_z.mean(0)
        ss_res_z += float(((true_z - pred_z) ** 2).sum())
        ss_tot_z += float(((true_z - mu_te_z) ** 2).sum())
    r2_raw = 1.0 - ss_res_raw / ss_tot_raw
    s_qwen = 1.0 - ss_res_z / ss_tot_z
    bar_std = BAR_RAW * (s_qwen / QWEN_COMMITTED_R2)
    deviates = abs(s_qwen - QWEN_COMMITTED_R2) > QWEN_CAL_DEV_MAX
    payload = {
        "metadata": _metadata(args.seed, len(conv_ids)),
        "computed_ts_unix": time.time(),
        "layer": li,
        "r2_raw_committed_grid": float(r2_raw),
        "committed_anchor": QWEN_COMMITTED_R2,
        "s_qwen_standardized": float(s_qwen),
        "bar_raw": BAR_RAW,
        "bar_std": float(bar_std),
        "calibration_deviates": bool(deviates),
        "deviation_threshold": QWEN_CAL_DEV_MAX,
        "std_floor_frac": STD_FLOOR_FRAC,
        "lambda_grid_std": "logspace(-2,8,21)",
    }
    _write_json(args.out_dir / "refit_qwen_cal.json", payload)
    print(
        f"[qwen_cal] raw R2={r2_raw:.4f} (anchor {QWEN_COMMITTED_R2}) "
        f"S_qwen={s_qwen:.4f} bar_std={bar_std:.4f} deviates={deviates}"
    )


# ---------------------------------------------------------------------------
# D1.4 — refit battery
# ---------------------------------------------------------------------------
def _conv_perms(conv_ids: np.ndarray, n_draws: int, seed: int) -> list[np.ndarray]:
    """Mirrors heldout_r2_sweep's _conv_perm (rng seed+1, conversation-level)."""
    rng = np.random.default_rng(seed + 1)
    ids = np.asarray(conv_ids)
    uniq_c, inv = np.unique(ids, return_inverse=True)
    row_of_conv = [np.flatnonzero(inv == k) for k in range(len(uniq_c))]

    def _one() -> np.ndarray:
        cp = rng.permutation(len(uniq_c))
        return np.concatenate([row_of_conv[k] for k in cp])

    return [_one() for _ in range(n_draws)]


def _grid_edges(lam: np.ndarray, grid: np.ndarray) -> dict:
    lo, hi = float(grid[0]), float(grid[-1])
    finite = lam[np.isfinite(lam)]
    return {
        "n_at_low_edge": int((finite == lo).sum()),
        "n_at_high_edge": int((finite == hi).sum()),
        "n_total": int(finite.size),
    }


def _battery_fingerprint(args, cell_id: str, n: int, n_layers: int) -> dict:
    """Resume key: EVERY output-affecting regime knob (#722 r3 rule)."""
    return {
        "cell_id": cell_id,
        "n": int(n),
        "n_layers": int(n_layers),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "null_draws": int(args.null_draws),
        "lambdas_committed": [float(v) for v in fc.LAMBDAS],
        "lambdas_wide": [float(v) for v in LAMBDAS_WIDE],
        "trim_ladder": list(TRIM_LADDER),
        "std_floor_frac": STD_FLOOR_FRAC,
    }


def _v2_fold_eval(cache, Y_tr, Y_te):
    """One fold's standardized-Y predict; returns (ss_res, ss_tot, pred_z, true_z, lam)."""
    mu, sd = _std_stats(Y_tr)
    pred_z, lam = fc._ridge_predict_cached(
        cache, (Y_tr - mu) / sd, return_lam=True, lambdas=LAMBDAS_WIDE
    )
    true_z = ((Y_te - mu) / sd).astype(np.float64)
    mu_te = true_z.mean(0)
    ss_res = float(((true_z - pred_z) ** 2).sum())
    ss_tot = float(((true_z - mu_te) ** 2).sum())
    return ss_res, ss_tot, pred_z, true_z, lam


def _battery_pass2_block(
    X: np.ndarray,
    Y: np.ndarray,
    folds: np.ndarray,
    layers: list[int],
    perms: list[np.ndarray],
    keep_pred_layers: set[int],
) -> dict:
    """Variants v1/v2/v3 + standardized nulls for one layer block.

    Serial per-draw calls through the SOURCE `_ridge_predict_cached` (identical
    committed convention, selection-symmetric by construction). Each null
    evaluation is one LARGE dense fp64 GEMM (~8.6e10 FLOPs) — FLOP-bound, not
    the overhead-bound loop class of vectorize-many-cell-fits.md; realized
    GPU-leg wall is minutes (pilot-gated below).
    """
    n_draws = len(perms)
    out: dict = {
        "v1_ss": {},
        "v2_ss": {},
        "v3_ss": {},
        "null_ss": {},
        "lam_v1": {},
        "lam_v2": {},
        "preds_v2": {},
    }
    fold_ids = list(np.unique(folds))
    for li in layers:
        X_l = np.ascontiguousarray(X[:, li, :])
        Y_l = np.ascontiguousarray(Y[:, li, :])
        v1 = np.zeros(2)
        v2 = np.zeros(2)
        v3 = {k: np.zeros(2) for k in TRIM_LADDER}
        nulls = np.zeros((n_draws, 2))
        lam1 = np.full(len(fold_ids), np.nan)
        lam2 = np.full(len(fold_ids), np.nan)
        pred_rows = np.zeros_like(Y_l, dtype=np.float64) if li in keep_pred_layers else None
        true_rows = np.zeros_like(Y_l, dtype=np.float64) if li in keep_pred_layers else None
        for kk, k in enumerate(fold_ids):
            te = folds == k
            tr = ~te
            if te.sum() == 0 or tr.sum() < 3:
                continue
            cache = fc._prep_fold(X_l[tr], X_l[te])
            true_raw = Y_l[te].astype(np.float64)
            mu_te = true_raw.mean(0)
            # v1 — widened grid, raw scale
            pred1, l1 = fc._ridge_predict_cached(
                cache, Y_l[tr], return_lam=True, lambdas=LAMBDAS_WIDE
            )
            v1 += [((true_raw - pred1) ** 2).sum(), ((true_raw - mu_te) ** 2).sum()]
            lam1[kk] = l1
            # v2 — standardized target
            r, t, pred_z, true_z, l2 = _v2_fold_eval(cache, Y_l[tr], Y_l[te])
            v2 += [r, t]
            lam2[kk] = l2
            if pred_rows is not None:
                pred_rows[te] = pred_z
                true_rows[te] = true_z
            # v3 — trimmed raw (train-fold variance ranking)
            var_order = np.argsort(Y_l[tr].var(0))[::-1]
            for ktrim in TRIM_LADDER:
                kt = min(ktrim, Y_l.shape[1] - 1)
                keep = np.ones(Y_l.shape[1], dtype=bool)
                keep[var_order[:kt]] = False
                pred3 = fc._ridge_predict_cached(cache, Y_l[tr][:, keep], lambdas=LAMBDAS_WIDE)
                tk = true_raw[:, keep]
                v3[ktrim] += [((tk - pred3) ** 2).sum(), ((tk - tk.mean(0)) ** 2).sum()]
            # standardized nulls (selection-symmetric: identical fit path)
            for d, perm in enumerate(perms):
                Yp = Y_l[perm]
                r, t, *_ = _v2_fold_eval(cache, Yp[tr], Yp[te])
                nulls[d] += [r, t]
            del cache
        out["v1_ss"][li] = v1
        out["v2_ss"][li] = v2
        out["v3_ss"][li] = v3
        out["null_ss"][li] = nulls
        out["lam_v1"][li] = lam1
        out["lam_v2"][li] = lam2
        if pred_rows is not None:
            out["preds_v2"][li] = (pred_rows.astype(np.float32), true_rows.astype(np.float32))
        print(f"[battery] pass2 layer {li} done", flush=True)
    return out


def _r2_of(ss: np.ndarray) -> float:
    return float(1.0 - ss[0] / ss[1]) if ss[1] > 0 else float("nan")


def _battery_pass1(args, cell_id, X, Y, conv_ids, vlayers, fp, meta_common, dg0_targets):
    """Pass 1 — v0 committed convention through the PRODUCTION function
    (heldout_r2_sweep verbatim: DG0 fix-engaged baseline + lambda audit +
    fp64-path preds regenerated at the verdict layers)."""
    ckpt_dir = args.out_dir / "checkpoints"
    v0_json = args.out_dir / f"refit_v0_{cell_id}.json"
    v0_ck = ckpt_dir / f"battery_{cell_id}_pass1.json"
    if v0_ck.exists() and json.loads(v0_ck.read_text())["fingerprint"] == fp:
        print(f"[battery] {cell_id} pass1 checkpoint present — skipping")
        return np.asarray(json.loads(v0_json.read_text())["r2_per_layer_obs"])
    t0 = time.time()
    sweep = fc.heldout_r2_sweep(
        X,
        Y,
        conv_ids,
        n_folds=args.folds,
        seed=args.seed,
        null_draws=0,
        collect_lambdas=True,
        frozen_layers=tuple(vlayers),
    )
    v0_curve = sweep["r2_obs"]
    best = float(np.nanmax(v0_curve))
    target = dg0_targets.get(cell_id)
    dg0 = None
    if target is not None:
        dg0 = {
            "target": target,
            "realized_best": best,
            "abs_dev": abs(best - target),
            "tol": DG0_TOL,
            "pass": bool(abs(best - target) <= DG0_TOL),
        }
    lam = sweep["gcv_lambda"]
    preds_npz = args.out_dir / "tensors" / f"battery_v0_preds_{cell_id}.npz"
    preds_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        preds_npz,
        **{f"preds_l{li}": p for li, p in sweep["preds_frozen"].items()},
        fitted_mask=sweep["fitted_mask"],
        folds=sweep["folds"],
        conv_ids=conv_ids,
    )
    _write_json(
        v0_json,
        {
            "metadata": _metadata(args.seed, X.shape[0]),
            **meta_common,
            "cell_id": cell_id,
            "variant": "v0_committed_convention",
            "r2_per_layer_obs": [float(v) for v in v0_curve],
            "best_r2": best,
            "argmax_layer": int(np.nanargmax(v0_curve)),
            "dg0": dg0,
            "gcv_lambda_layer_x_fold": [[float(v) for v in row] for row in lam],
            "grid_edges": _grid_edges(np.asarray(lam), fc.LAMBDAS),
            "preds_npz": str(preds_npz),
            "preds_dtype": "float32 (fp64 fold math; fp16-cross-check adequate)",
            "wall_s": time.time() - t0,
        },
    )
    if dg0 is not None and not dg0["pass"]:
        print(
            f"[battery] DG0 FAIL {cell_id}: best {best:.4f} vs target {target} "
            f"(tol {DG0_TOL}) — diagnostic driver diverges from the wave-1 path",
            file=sys.stderr,
        )
        raise SystemExit(3)
    print(f"[battery] DG0 {cell_id}: best {best:.4f} vs {target} -> PASS")
    # Resume checkpoint is written ONLY after the DG0 gate passes: a
    # same-fingerprint rerun after a DG0 FAIL re-runs the gate instead of
    # resuming past it (r3 concern dg0-checkpoint-written-before-gate).
    v0_ck.write_text(json.dumps({"fingerprint": fp}))
    return v0_curve


def _pilot_gate(args, unit_s: float, total_units: int) -> None:
    """SS9 pilot gate: ONE (layer x folds x variants x null-draws) unit timed
    end-to-end; abort > 2x planned wall (surface + re-project, never a silent
    descope). Unit count is conservative (every cell at this cell's width)."""
    projected_h = unit_s * total_units / 3600.0
    print(
        f"[battery-pilot] unit={unit_s:.1f}s projected_wall={projected_h:.2f}h "
        f"(planned {args.wall_budget_h}h, abort >{2 * args.wall_budget_h}h)"
    )
    if projected_h > 2 * args.wall_budget_h and not args.no_pilot_abort:
        raise SystemExit(
            f"[battery-pilot] ABORT: projected {projected_h:.2f}h > "
            f"2x planned {args.wall_budget_h}h — surface + re-project"
        )


def step_battery(args) -> None:
    print("[diag1336] step=battery", flush=True)
    dg0_targets = dict(DG0_TARGETS)
    if args.dg0_targets_json:
        dg0_targets.update({k: float(v) for k, v in json.loads(args.dg0_targets_json).items()})
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    started_ts = time.time()
    meta_common = {
        "started_ts_unix": started_ts,
        "device": str(fc._fit_device()),
        "lambda_grid_committed": [float(v) for v in fc.LAMBDAS],
        "lambda_grid_wide": [float(v) for v in LAMBDAS_WIDE],
    }
    pilot_state = {"done": False}
    for cell_id in args.cell_ids:
        xy = _load_cell_xy(args, cell_id)
        X, Y, conv_ids = xy["X"], xy["Y"], xy["conv_ids"]
        n, n_layers = X.shape[0], X.shape[1]
        folds = fc._cv_folds(conv_ids, args.folds, args.seed)
        fp = _battery_fingerprint(args, cell_id, n, n_layers)
        vlayers = _verdict_layers_for(n_layers)
        is_chat = "naturalistic" not in cell_id
        pass2_layers = list(range(n_layers)) if is_chat else vlayers

        _battery_pass1(args, cell_id, X, Y, conv_ids, vlayers, fp, meta_common, dg0_targets)
        acc = _battery_pass2_blocks(
            args, cell_id, X, Y, conv_ids, folds, pass2_layers, vlayers, fp, pilot_state
        )
        _battery_reduce(args, cell_id, X, Y, folds, conv_ids, acc, meta_common, n)
        # v4 — fp64 end-to-end at the verdict layers (no fp32 waypoint).
        _battery_v4(args, cell_id, folds, vlayers, n, meta_common)
    print("[battery] complete", flush=True)


def _battery_pass2_blocks(
    args, cell_id, X, Y, conv_ids, folds, pass2_layers, vlayers, fp, pilot_state
) -> dict:
    """Block loop for pass 2 with checkpoint/resume + the pilot gate."""
    ckpt_dir = args.out_dir / "checkpoints"
    perms = _conv_perms(conv_ids, args.null_draws, args.seed)
    blocks = [pass2_layers[i : i + LAYER_BLOCK] for i in range(0, len(pass2_layers), LAYER_BLOCK)]
    acc: dict = {
        "v1_ss": {},
        "v2_ss": {},
        "v3_ss": {},
        "null_ss": {},
        "lam_v1": {},
        "lam_v2": {},
        "preds_v2": {},
    }
    for bi, block in enumerate(blocks):
        bck = ckpt_dir / f"battery_{cell_id}_pass2_block{bi}.npz"
        bfp = ckpt_dir / f"battery_{cell_id}_pass2_block{bi}.json"
        if bck.exists() and bfp.exists() and json.loads(bfp.read_text())["fingerprint"] == fp:
            data = dict(np.load(bck, allow_pickle=False))
            for li in block:
                for key in ("v1_ss", "v2_ss", "null_ss", "lam_v1", "lam_v2"):
                    acc[key][li] = data[f"{key}_l{li}"]
                acc["v3_ss"][li] = {k: data[f"v3_ss_l{li}_k{k}"] for k in TRIM_LADDER}
                if f"preds_v2_l{li}" in data:
                    acc["preds_v2"][li] = (data[f"preds_v2_l{li}"], data[f"true_v2_l{li}"])
            print(f"[battery] {cell_id} pass2 block {bi} checkpoint present — skipping")
            continue
        t0 = time.time()
        pilot_block = block if pilot_state["done"] else block[:1]
        res = _battery_pass2_block(X, Y, folds, pilot_block, perms, keep_pred_layers=set(vlayers))
        if not pilot_state["done"]:
            _pilot_gate(args, time.time() - t0, len(pass2_layers) * len(args.cell_ids))
            pilot_state["done"] = True
            rest = block[1:]
            if rest:
                res2 = _battery_pass2_block(X, Y, folds, rest, perms, keep_pred_layers=set(vlayers))
                for key in res:
                    res[key].update(res2[key])
        save: dict = {}
        for li in block:
            for key in ("v1_ss", "v2_ss", "null_ss", "lam_v1", "lam_v2"):
                acc[key][li] = res[key][li]
                save[f"{key}_l{li}"] = np.asarray(res[key][li])
            acc["v3_ss"][li] = res["v3_ss"][li]
            for k, v in res["v3_ss"][li].items():
                save[f"v3_ss_l{li}_k{k}"] = np.asarray(v)
            if li in res["preds_v2"]:
                acc["preds_v2"][li] = res["preds_v2"][li]
                save[f"preds_v2_l{li}"], save[f"true_v2_l{li}"] = res["preds_v2"][li]
        np.savez(bck, **save)
        bfp.write_text(json.dumps({"fingerprint": fp}))
        print(f"[battery] {cell_id} pass2 block {bi}/{len(blocks)} checkpointed")
    return acc


def _battery_reduce(args, cell_id, X, Y, folds, conv_ids, acc, meta_common, n) -> None:
    """Reduce pass-2 accumulators to the refit JSONs."""
    layer_list = sorted(acc["v2_ss"])
    v1_curve = {li: _r2_of(acc["v1_ss"][li]) for li in layer_list}
    v2_curve = {li: _r2_of(acc["v2_ss"][li]) for li in layer_list}
    null_mat = (
        np.stack(
            [  # (draws, layers)
                [_r2_of(acc["null_ss"][li][d]) for li in layer_list] for d in range(args.null_draws)
            ]
        )
        if args.null_draws
        else np.zeros((0, len(layer_list)))
    )
    per_draw_max = np.nanmax(null_mat, axis=1) if null_mat.size else np.array([])
    band = float(np.quantile(per_draw_max, 0.975)) if per_draw_max.size else float("nan")
    s_layer = layer_list[int(np.nanargmax([v2_curve[li] for li in layer_list]))]
    s_val = v2_curve[s_layer]
    # Bootstrap CI on S at the argmax layer (pred-resampling, committed
    # machinery — never a refit).
    if s_layer not in acc["preds_v2"]:
        acc["preds_v2"][s_layer] = _v2_preds_at_layer(X, Y, folds, s_layer)
    pred_z, true_z = acc["preds_v2"][s_layer]
    ci = fc.bootstrap_r2_ci(
        pred_z.astype(np.float64),
        true_z.astype(np.float64),
        n_boot=args.n_boot,
        seed=args.seed + 100,
    )
    preds_v2_npz = args.out_dir / "tensors" / f"battery_v2_preds_{cell_id}.npz"
    np.savez(
        preds_v2_npz,
        **{f"preds_l{li}": p for li, (p, _t) in acc["preds_v2"].items()},
        **{f"true_l{li}": t for li, (_p, t) in acc["preds_v2"].items()},
        # fp64 verdict-layer copies (plan SS10 output contract)
        **{f"preds_fp64_l{s_layer}": pred_z.astype(np.float64)},
        folds=folds,
        conv_ids=conv_ids,
    )
    _write_json(
        args.out_dir / f"refit_v1_{cell_id}.json",
        {
            "metadata": _metadata(args.seed, n),
            **meta_common,
            "cell_id": cell_id,
            "variant": "v1_widened_lambda_grid",
            "layers": layer_list,
            "r2_per_layer": {str(li): v1_curve[li] for li in layer_list},
            "best_r2": float(np.nanmax(list(v1_curve.values()))),
            "gcv_lambda_layer_x_fold": {
                str(li): [float(v) for v in acc["lam_v1"][li]] for li in layer_list
            },
            "grid_edges": _grid_edges(
                np.concatenate([np.asarray(acc["lam_v1"][li]) for li in layer_list]),
                LAMBDAS_WIDE,
            ),
        },
    )
    _write_json(
        args.out_dir / f"refit_v2_{cell_id}.json",
        {
            "metadata": _metadata(args.seed, n),
            **meta_common,
            "cell_id": cell_id,
            "variant": "v2_per_dim_standardized",
            "layers": layer_list,
            "r2_per_layer": {str(li): v2_curve[li] for li in layer_list},
            "S_argmax_layer": int(s_layer),
            "S": float(s_val),
            "bootstrap_ci_S": ci,
            "gcv_lambda_layer_x_fold": {
                str(li): [float(v) for v in acc["lam_v2"][li]] for li in layer_list
            },
            "preds_npz": str(preds_v2_npz),
            "std_floor_frac": STD_FLOOR_FRAC,
        },
    )
    _write_json(
        args.out_dir / f"refit_v3_{cell_id}.json",
        {
            "metadata": _metadata(args.seed, n),
            **meta_common,
            "cell_id": cell_id,
            "variant": "v3_trimmed_raw",
            "trim_ladder": list(TRIM_LADDER),
            "r2_per_layer_per_trim": {
                str(li): {str(k): _r2_of(acc["v3_ss"][li][k]) for k in TRIM_LADDER}
                for li in layer_list
            },
        },
    )
    _write_json(
        args.out_dir / f"refit_null_std_{cell_id}.json",
        {
            "metadata": _metadata(args.seed, n),
            **meta_common,
            "cell_id": cell_id,
            "variant": "null_standardized_selection_symmetric",
            "n_draws": int(args.null_draws),
            "layers": layer_list,
            "null_matrix_draw_x_layer": [[float(v) for v in row] for row in null_mat],
            "null_layer_max_per_draw": [float(v) for v in per_draw_max],
            "band_p975_layer_max": band,
        },
    )


def _v2_preds_at_layer(X, Y, folds, li) -> tuple[np.ndarray, np.ndarray]:
    """Standardized held-out preds at one layer (argmax not in the persisted set)."""
    X_l, Y_l = X[:, li, :], Y[:, li, :]
    pred_rows = np.zeros_like(Y_l, dtype=np.float64)
    true_rows = np.zeros_like(Y_l, dtype=np.float64)
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        cache = fc._prep_fold(X_l[tr], X_l[te])
        _r, _t, pred_z, true_z, _lam = _v2_fold_eval(cache, Y_l[tr], Y_l[te])
        pred_rows[te] = pred_z
        true_rows[te] = true_z
    return pred_rows.astype(np.float32), true_rows.astype(np.float32)


def _battery_v4(args, cell_id: str, folds, vlayers, n, meta_common) -> None:
    """v4 fp64 end-to-end (bf16 -> fp64, no fp32 waypoint), verdict layers.

    Scored under BOTH grids: the committed grid isolates precision vs v0
    (A_v4 input); the widened grid is reported for the SS11 'v1+' convention.
    """
    X64, Y64 = _load_xy_fp64(args, cell_id, list(vlayers))
    out: dict = {}
    for j, li in enumerate(vlayers):
        Xl, Yl = X64[:, j, :], Y64[:, j, :]
        ss = {"committed": np.zeros(2), "wide": np.zeros(2)}
        for k in np.unique(folds):
            te = folds == k
            tr = ~te
            cache = fc._prep_fold(Xl[tr], Xl[te])
            true = Yl[te]
            mu_te = true.mean(0)
            for grid_name, grid in (("committed", None), ("wide", LAMBDAS_WIDE)):
                pred = fc._ridge_predict_cached(cache, Yl[tr], lambdas=grid)
                ss[grid_name] += [((true - pred) ** 2).sum(), ((true - mu_te) ** 2).sum()]
        out[str(li)] = {g: _r2_of(v) for g, v in ss.items()}
    _write_json(
        args.out_dir / f"refit_v4_{cell_id}.json",
        {
            "metadata": _metadata(args.seed, n),
            **meta_common,
            "cell_id": cell_id,
            "variant": "v4_fp64_end_to_end",
            "layers": list(vlayers),
            "r2_per_layer_per_grid": out,
        },
    )


# ---------------------------------------------------------------------------
# D1.5 — verdict assembly
# ---------------------------------------------------------------------------
def _read_committed_json(args, relpath: str) -> dict | None:
    """Committed wave-1 JSON: filesystem first, `git show HEAD:<path>` fallback
    (sparse worktrees exclude eval_results/ from the checkout, not the odb)."""
    p = args.committed_eval_dir / relpath
    if p.exists():
        return json.loads(p.read_text())
    import subprocess

    proc = subprocess.run(
        ["git", "show", f"HEAD:eval_results/issue_1336/{relpath}"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    if proc.returncode == 0:
        return json.loads(proc.stdout)
    return None


def _raw_band(args, cell_id: str) -> tuple[float, str]:
    committed = _read_committed_json(args, f"cells/nulls_{cell_id}.json")
    if committed is not None:
        per_draw = np.asarray(committed["null_layer_max_per_draw"], dtype=float)
        return float(np.quantile(per_draw, 0.975)), "committed_nulls_json"
    return COMMITTED_RAW_BAND_FALLBACK, "constant_plan_s2_fallback"


def _a_v(r2_v: float, r2_v0: float, band: float) -> float:
    denom = band - r2_v0
    return float((r2_v - r2_v0) / denom) if abs(denom) > 1e-12 else float("nan")


def step_verdict(args) -> None:
    print("[diag1336] step=verdict", flush=True)
    read_ts = time.time()
    qc_path = args.out_dir / "refit_qwen_cal.json"
    if qc_path.exists():
        qc = json.loads(qc_path.read_text())
        bar_std = float(qc["bar_std"])
        bar_std_fallback = False
    else:
        qc = None
        bar_std = BAR_RAW  # registered fallback (plan SS3): staging failure only
        bar_std_fallback = True
    chat = args.cell_ids[0]
    v0 = json.loads((args.out_dir / f"refit_v0_{chat}.json").read_text())
    v1 = json.loads((args.out_dir / f"refit_v1_{chat}.json").read_text())
    v2 = json.loads((args.out_dir / f"refit_v2_{chat}.json").read_text())
    v3 = json.loads((args.out_dir / f"refit_v3_{chat}.json").read_text())
    v4 = json.loads((args.out_dir / f"refit_v4_{chat}.json").read_text())
    nulls = json.loads((args.out_dir / f"refit_null_std_{chat}.json").read_text())
    spot_path = args.out_dir / "spotcheck.json"
    # H-B capture-defect gate input is REQUIRED (r3 Minor 2): a direct
    # `--steps battery,verdict` invocation without the D1.3 spot-check would
    # otherwise silently route with capture_defect=False. The dispatch +
    # canonical step order always run spotcheck first.
    assert spot_path.exists(), (
        f"{spot_path} missing — run the spotcheck step before verdict (the "
        "capture-defect gate cannot be evaluated without its H-B input)"
    )
    spot = json.loads(spot_path.read_text())
    audit_path = args.out_dir / "scale_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None

    # DG1 — calibration-ordering invariant (plan SS7).
    battery_started = float(v2["started_ts_unix"])
    if qc is not None:
        assert float(qc["computed_ts_unix"]) <= battery_started, (
            "DG1 ORDERING VIOLATION: qwen_cal computed AFTER the battery started "
            f"({qc['computed_ts_unix']} > {battery_started}) — bar_std must be set "
            "before the Llama verdict quantities are evaluated"
        )
    dg1 = {
        "bar_std_computed_ts_unix": float(qc["computed_ts_unix"]) if qc else None,
        "battery_started_ts_unix": battery_started,
        "verdict_read_ts_unix": read_ts,
        "ordering_ok": bool(qc is not None),
        "bar_std_fallback": bar_std_fallback,
    }

    v0_curve = np.asarray(v0["r2_per_layer_obs"], dtype=float)
    l29 = min(L_COMMITTED_ARGMAX, len(v0_curve) - 1)
    r2_v0_l29 = float(v0_curve[l29])
    raw_band, raw_band_source = _raw_band(args, chat)
    std_band = float(nulls["band_p975_layer_max"])

    def _curve(d: dict) -> dict[int, float]:
        return {int(k): float(v) for k, v in d.items()}

    variants = {
        "v1_widened_grid": {"curve": _curve(v1["r2_per_layer"]), "band": raw_band},
        "v2_standardized": {"curve": _curve(v2["r2_per_layer"]), "band": std_band},
    }
    for ktrim in TRIM_LADDER:
        variants[f"v3_trim_{ktrim}"] = {
            "curve": {
                int(li): float(per_trim[str(ktrim)])
                for li, per_trim in v3["r2_per_layer_per_trim"].items()
            },
            "band": raw_band,  # caveat: no trimmed-convention band was drawn (SS9 budget)
        }
    variants["v4_fp64"] = {
        "curve": {int(li): g["committed"] for li, g in v4["r2_per_layer_per_grid"].items()},
        "band": raw_band,
    }
    attribution = {}
    for name, spec in variants.items():
        curve = spec["curve"]
        r2_at_l29 = curve.get(l29, float("nan"))
        own_argmax = max(curve, key=lambda li: curve[li])
        r2_v0_own = float(v0_curve[own_argmax]) if own_argmax < len(v0_curve) else float("nan")
        attribution[name] = {
            "r2_at_L29": r2_at_l29,
            "A_v_at_L29": _a_v(r2_at_l29, r2_v0_l29, spec["band"]),
            "own_argmax_layer": int(own_argmax),
            "r2_at_own_argmax": curve[own_argmax],
            "A_v_at_own_argmax": _a_v(curve[own_argmax], r2_v0_own, spec["band"]),
            "band_used": spec["band"],
            "accounts_at_0.8": bool(_a_v(r2_at_l29, r2_v0_l29, spec["band"]) >= A_V_BAR),
            "accounts_at_sensitivity": {
                str(th): bool(_a_v(r2_at_l29, r2_v0_l29, spec["band"]) >= th)
                for th in A_V_SENSITIVITY
            },
        }
    accounting = sorted(k for k, v in attribution.items() if v["accounts_at_0.8"])

    # Lattice (plan SS3): S, B, bar_std, S' = S - bar_std, D = S - B (+CI on S).
    S = float(v2["S"])
    ci = v2["bootstrap_ci_S"]
    ci_lo = float(ci.get("lo", ci.get("ci_lo", float("nan"))))
    ci_hi = float(ci.get("hi", ci.get("ci_hi", float("nan"))))
    s_prime = S - bar_std
    d_stat = S - std_band
    d_ci = [ci_lo - std_band, ci_hi - std_band]
    d_excludes_zero_pos = d_ci[0] > 0.0
    if s_prime >= 0 and d_excludes_zero_pos:
        branch = "map_present_usable"
    elif d_excludes_zero_pos:
        branch = "map_present_weak"
    else:
        branch = "genuine_absence_candidate"

    capture_defect = bool(spot.get("any_defect_gate_fired")) or bool(
        audit and audit.get("capture_corruption_suspected")
    )
    # bar_std_fallback deliberately does NOT route R5 (plan §9 descope rule 3:
    # a staging failure falls back to bar 0.20, stated — not a replan).
    cal_deviates = bool(qc and qc.get("calibration_deviates"))

    if cal_deviates:
        routed = "R5_replan"
        reason = "D1.6 calibration control deviates — corrected DV demoted to exploratory"
    elif capture_defect:
        routed = "R2_d2_required" if branch == "map_present_usable" else "D2_required"
        reason = "capture-defect gate fired — D2 capture-parity probe before any lattice read"
    elif not accounting:
        routed = "R5_replan"
        reason = "no variant reaches A_v >= 0.8 and no capture defect — anomaly unattributed"
    elif branch == "map_present_usable":
        routed = "R1_resume"
        reason = "lattice branch 1 with a >=0.8-accounting mechanism"
    elif branch == "map_present_weak":
        routed = "R4_scope_finding_weak"
        reason = "lattice branch 2 — map present but below the usable-strength bar"
    else:
        routed = "R3_scope_finding_absence"
        reason = "lattice branch 3 with a >=0.8-accounting mechanism for the committed -0.93"

    # Naturalistic robustness (reported, non-lattice): branch agreement.
    nat_agreement = None
    if len(args.cell_ids) > 1:
        nat = args.cell_ids[1]
        try:
            v2n = json.loads((args.out_dir / f"refit_v2_{nat}.json").read_text())
            nulln = json.loads((args.out_dir / f"refit_null_std_{nat}.json").read_text())
            s_n = float(v2n["S"])
            b_n = float(nulln["band_p975_layer_max"])
            branch_n = (
                "map_present_usable"
                if (s_n - bar_std) >= 0 and s_n > b_n
                else ("map_present_weak" if s_n > b_n else "genuine_absence_candidate")
            )
            nat_agreement = {
                "S_naturalistic": s_n,
                "band": b_n,
                "branch": branch_n,
                "agrees_with_chat": bool(branch_n == branch),
            }
        except FileNotFoundError:
            nat_agreement = {"error": "naturalistic refit outputs missing"}

    payload = {
        "metadata": _metadata(args.seed, 0),
        "gates": {
            "dg0": v0.get("dg0"),
            "dg1": dg1,
            "capture_defect_gate": {
                "fired": capture_defect,
                "spotcheck": {
                    c: {
                        "mismatch_rate": s["mismatch_rate"],
                        "defect_gate_fired": s["defect_gate_fired"],
                    }
                    for c, s in (spot["cells"].items() if spot else [])
                },
                "audit_capture_corruption_suspected": bool(
                    audit and audit.get("capture_corruption_suspected")
                ),
            },
            "qwen_calibration": {
                "s_qwen": qc.get("s_qwen_standardized") if qc else None,
                "deviates": qc.get("calibration_deviates") if qc else None,
                "bar_std_fallback": bar_std_fallback,
            },
        },
        "lattice_inputs": {
            "S": S,
            "S_argmax_layer": int(v2["S_argmax_layer"]),
            "B_standardized_p975_layer_max": std_band,
            "bar_std": bar_std,
            "S_prime": s_prime,
            "D": d_stat,
            "D_ci95": d_ci,
            "S_ci95": [ci_lo, ci_hi],
            "raw_band": raw_band,
            "raw_band_source": raw_band_source,
            "committed_r2_v0_at_L29": r2_v0_l29,
        },
        "mechanism_attribution": {
            "threshold": A_V_BAR,
            "sensitivity": list(A_V_SENSITIVITY),
            "per_variant": attribution,
            "accounting_set": accounting,
        },
        "lattice_branch": branch,
        "routed_decision": routed,
        "routed_reason": reason,
        "naturalistic_robustness": nat_agreement,
    }
    _write_json(args.out_dir / "diagnosis_verdict.json", payload)
    print(f"[verdict] branch={branch} routed={routed} — {reason}")


# ---------------------------------------------------------------------------
STEP_FNS = {
    "stage": step_stage,
    "decomp": step_decomp,
    "audit": step_audit,
    "spotcheck": step_spotcheck,
    "qwen_cal": step_qwen_cal,
    "battery": step_battery,
    "verdict": step_verdict,
}


def normalize_steps(raw: str) -> list[str]:
    """Canonical execution order regardless of CLI order (DG1: qwen_cal before
    battery before verdict). Unknown step names fail loud."""
    req = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in req if s not in STEP_ORDER]
    assert not unknown, f"unknown steps {unknown}; valid: {STEP_ORDER}"
    return [s for s in STEP_ORDER if s in req]


def main() -> int:
    args = parse_args()
    args.cell_ids = [c.strip() for c in args.cells.split(",") if c.strip()]
    assert args.cell_ids, "--cells resolved to an empty list"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps = normalize_steps(args.steps)
    print(f"[diag1336] steps={steps} cells={args.cell_ids} out={args.out_dir}")
    for s in steps:
        STEP_FNS[s](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
