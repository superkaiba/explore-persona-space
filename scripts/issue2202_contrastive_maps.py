#!/usr/bin/env python3
"""Issue #2202 — inline round `contrastive-maps`: discrimination-optimized
context→answer maps on the banked #1738 L19 tensors.

Two families, ONE estimator change vs the #1738 parent fitters (MSE → InfoNCE;
data path, split, standardizer, and architecture reused verbatim):

- ``linear``: pred = x_std @ W + c, warm-started at the banked ridge solution
  (``issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt`` —
  frozen train standardizer xmu/xsd; W init = banked ridge W, c init = ymu).
- ``mlp``: the #1738 ``mlp_w8192`` architecture verbatim
  (Linear(3584→8192) → GELU → Linear(8192→3584) on standardized X, + ymu),
  fresh init (seed 2202), same frozen standardizer.

Loss = symmetric InfoNCE with in-batch negatives (batch 2048, cosine scores /
temperature τ; context→answer + answer→context averaged). τ ∈ {0.05, 0.1, 0.2}
per family; early stop + τ selection on the #1738 pinned VALIDATION rows
(val cosine acc@1 within the 396-row val pool). The 9,941-row holdout is
touched ONLY by the final battery.

Battery (mid-rank + tie conventions = ``mapping_baselines.knn_retrieval``,
the same reduction the #2202 repro gate reproduced): full-pool holdout
acc@{1,5,10} + MRR under raw_euclidean / raw_cos / whiten / whiten_cos
(whitening = the task-LOCKED ``issue2202_ctxfail/analysis_tensors/
whiten_stats.npz``: z = L⁻¹(x − μ_A), plan-§11 A20 convention), plus pooled
holdout R², for {banked ridge, banked MSE-MLP (mlp_w8192), contrastive linear,
contrastive MLP}.

n_train vs d: realized n_train = 87,795 captured train rows ≫ d = 3,584 —
well-posed; no under-determined regime (dispatch-note estimator-validity
statement).

Pod-side phases (``--phase all``): stage → assemble → fit → battery → upload.
Checkpoint cadence: each fit's weights + metrics JSON land in the out-root the
moment that fit completes (resume skips valid completed fits, keyed on
family × τ × the fit regime); the linear result survives a mid-MLP death.
Smoke (``--smoke``): same code paths end-to-end, τ grid cut to {0.1}, 2-epoch
cap, out-root sub-dir ``smoke/``, upload skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps land BEFORE numpy/torch on the shared VM (#847)

import issue779_common as C  # noqa: E402
import issue1738_multiturn_fits as FT  # noqa: E402  (assemble_streams, load_split)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue2202_failchar import repo_banked_path  # noqa: E402  (#1739 sparse-cone trap)
from scipy.linalg import solve_triangular  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2202_contrastive")

ISSUE = 2202
SEED = 2202
LAYER = 19
H_DIM = C.EXPECTED_HIDDEN  # 3584
HF_PIN = "09788eef2f85330c6f9c6b7cd3d28cb47cfb8429"  # #2202 plan-§10 data-repo pin
PARENT_PREFIX = "issue1738_multiturn"
HF_PREFIX_2202 = "issue2202_ctxfail"
UPLOAD_SUBPREFIX = "contrastive_maps"
EXPECTED_N = 9_941  # pinned holdout n (gate-asserted, matches #2202)
EXPECTED_CAPTURE_CHUNKS = 224  # *.pt chunks at the pin (#2202 drift guard)
N_TRAIN_FLOOR = 20_000  # #2202 A16 floor (realized 87,795)
TAUS = (0.05, 0.1, 0.2)
BATCH = 2048
MAX_EPOCHS = 50
PATIENCE = 8  # epochs without val-acc@1 improvement before early stop
LR = {"linear": 1e-4, "mlp": 3e-4}  # mlp: conservative member of the #1738 round-1
# lr grid (1e-3, 3e-4); linear: smaller step to preserve the ridge warm start.
WEIGHT_DECAY = 0.0  # decay would shrink the warm-started W toward 0 — off by design
KS = (1, 5, 10)
FAMILIES = ("linear", "mlp")
SPACES = ("raw_euclidean", "raw_cos", "whiten", "whiten_cos")
BANKED_REL = "eval_results/issue_1738/mapping_baselines.json"
FITS_SUMMARY_REL = "eval_results/issue_1738/fits/multiturn_100k_fits.json"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def meta_block(extra: dict | None = None) -> dict:
    m = {
        "issue": ISSUE,
        "seed": SEED,
        "layer": LAYER,
        "tau_grid": list(TAUS),
        "batch": BATCH,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lr": dict(LR),
        "weight_decay": WEIGHT_DECAY,
        "ts": now_iso(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        **as_metadata_dict(git_provenance()),
    }
    if extra:
        m.update(extra)
    return m


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _staged(args) -> Path:
    return Path(args.work_root) / "staged"


def _out_root(args) -> Path:
    root = Path(args.out_root)
    return root / "smoke" if args.smoke else root


def _fit_regime(args) -> str:
    """Resume-regime key: any output-affecting knob is part of it (#722-r3)."""
    return f"smoke={bool(args.smoke)}|batch={BATCH}|epochs={args.max_epochs}|seed={SEED}"


# ── staging ────────────────────────────────────────────────────────────────────────


def stage_inputs(args) -> None:
    """Stage the banked small inputs (pinned revision for #1738 artifacts; the
    #2202-owned derived tensors are staged from main — they postdate the pin
    and are the task-locked instrument by path)."""
    staged = _staged(args)
    staged.mkdir(parents=True, exist_ok=True)
    pinned = [
        (f"{PARENT_PREFIX}/analysis_tensors/pred16/context_L19_ridge.npz", "pred16_ridge.npz"),
        (
            f"{PARENT_PREFIX}/analysis_tensors/pred16/context_L19_mlp_w8192.npz",
            "pred16_mlp_w8192.npz",
        ),
        (f"{PARENT_PREFIX}/analysis_tensors/y_holdout/L{LAYER}.npz", "y_holdout_L19.npz"),
        (f"{PARENT_PREFIX}/analysis_tensors/weights/L{LAYER}/context_ridge.pt", "ridge_L19.pt"),
        (f"{PARENT_PREFIX}/sampling_manifest/split_1738.json", "split_1738.json"),
    ]
    for k, (src, dst) in enumerate(pinned):
        hub.stage_hub_file(C.HF_DATA_REPO, src, staged / dst, revision=args.revision)
        print(f"[stage] unit {k + 1}/{len(pinned) + 2} {dst}", flush=True)
    for k, name in enumerate(("whiten_stats.npz", "cx_holdout_L19.npz")):
        hub.stage_hub_file(
            C.HF_DATA_REPO, f"{HF_PREFIX_2202}/analysis_tensors/{name}", staged / name
        )
        print(f"[stage] unit {len(pinned) + k + 1}/{len(pinned) + 2} {name}", flush=True)


def assemble(args):
    """(mm, ci, meta) L19 capture memmaps via the #1738 streamer (per-chunk
    download + append-only fp32 binaries + cursor resume; peak ~one chunk).
    Hand-built namespace supplies EVERY field ``assemble_streams`` reads
    (#1728 call-shape bind: mm_dir / local_capture_dir / hf_prefix)."""
    ns = SimpleNamespace(
        mm_dir=Path(args.work_root) / "mm",
        local_capture_dir=None,
        hf_prefix=PARENT_PREFIX,
    )
    mm, ci, meta = FT.assemble_streams(ns, layers=[LAYER])
    if meta["n_chunks"] != EXPECTED_CAPTURE_CHUNKS:
        raise RuntimeError(
            f"capture chunk count drift: {meta['n_chunks']} != {EXPECTED_CAPTURE_CHUNKS}"
        )
    return mm, ci, meta


def load_split_sets(args, ci: np.ndarray) -> dict[str, np.ndarray]:
    split = FT.load_split(_staged(args) / "split_1738.json")
    sets = FT.split_positions(split, ci)
    assert len(sets["train"]) >= N_TRAIN_FLOOR, len(sets["train"])
    return sets


def load_ridge_payload(args) -> dict:
    pl = torch.load(_staged(args) / "ridge_L19.pt", map_location="cpu", weights_only=False)
    for k in ("xmu", "xsd", "ymu", "W"):
        assert k in pl, f"ridge payload missing {k!r}: {sorted(pl)}"
    assert pl["W"].shape == (H_DIM, H_DIM), pl["W"].shape
    assert pl["xmu"].shape == (H_DIM,), pl["xmu"].shape
    return pl


# ── fit ────────────────────────────────────────────────────────────────────────────


class ContrastiveMap(torch.nn.Module):
    """pred(x) = head((x − xmu)/xsd) + c — the #1738 fitter parametrization with
    the output offset c trainable (init = train ymu; the ridge/MLP parents add
    the SAME ymu, frozen)."""

    def __init__(self, family: str, payload: dict):
        super().__init__()
        self.register_buffer("xmu", payload["xmu"].to(torch.float32))
        self.register_buffer("xsd", payload["xsd"].to(torch.float32))
        self.c = torch.nn.Parameter(payload["ymu"].to(torch.float32).clone())
        if family == "linear":
            self.w = torch.nn.Parameter(payload["W"].to(torch.float32).clone())
            self.net = None
        elif family == "mlp":
            torch.manual_seed(SEED)
            self.w = None
            self.net = torch.nn.Sequential(
                torch.nn.Linear(H_DIM, 8192), torch.nn.GELU(), torch.nn.Linear(8192, H_DIM)
            )
        else:
            raise ValueError(family)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (H_DIM,), x.shape
        z = (x - self.xmu) / self.xsd
        return (z @ self.w if self.net is None else self.net(z)) + self.c


def info_nce(pred: torch.Tensor, tgt: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric in-batch InfoNCE on cosine scores / τ (row i's positive is
    column i; every other in-batch row is a negative)."""
    b = pred.shape[0]
    assert tgt.shape == pred.shape, (pred.shape, tgt.shape)
    p = torch.nn.functional.normalize(pred, dim=1)
    t = torch.nn.functional.normalize(tgt, dim=1)
    s = (p @ t.T) / tau
    labels = torch.arange(b, device=pred.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(s, labels)
        + torch.nn.functional.cross_entropy(s.T, labels)
    )


@torch.no_grad()
def val_acc1(model: ContrastiveMap, x_val: torch.Tensor, y_val: torch.Tensor) -> dict:
    """Val-pool retrieval acc@1 (argmax over the val pool): cosine (= the
    trained score; the selection metric) + euclidean companion."""
    pred = model(x_val)
    p = torch.nn.functional.normalize(pred, dim=1)
    t = torch.nn.functional.normalize(y_val, dim=1)
    cos_hit = ((p @ t.T).argmax(dim=1) == torch.arange(len(p), device=p.device)).float()
    d = torch.cdist(pred, y_val)
    euc_hit = (d.argmin(dim=1) == torch.arange(len(pred), device=p.device)).float()
    return {"cos_acc1": float(cos_hit.mean()), "euc_acc1": float(euc_hit.mean())}


def fit_one(args, family: str, tau: float, x_tr, y_tr, x_val, y_val, payload: dict, dev) -> dict:
    """One (family, τ) InfoNCE fit; per-epoch val early stop on cos acc@1.
    Writes weights + metrics JSON to the out-root ON COMPLETION (checkpoint
    cadence: the linear result survives a mid-MLP death). Returns metrics."""
    fits_dir = _out_root(args) / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{family}_tau{tau:g}"
    jpath = fits_dir / f"{slug}.json"
    if jpath.exists():
        prev = json.loads(jpath.read_text())
        if prev.get("fit_regime") == _fit_regime(args) and (fits_dir / f"{slug}.pt").exists():
            logger.info("[fit] %s resume-skip (valid completed fit)", slug)
            return prev
        logger.info("[fit] %s stale regime — refitting", slug)

    model = ContrastiveMap(family, payload).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[family], weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    n = x_tr.shape[0]
    best = {"acc": -1.0, "epoch": -1, "state": None}
    curve = []
    t0 = time.time()
    first_epoch_wall = None
    for epoch in range(args.max_epochs):
        te0 = time.time()
        model.train()
        perm = rng.permutation(n)
        losses = []
        for s in range(0, n - BATCH + 1, BATCH):  # drop the ragged tail: full
            # batches keep the negative count (and the loss scale) constant.
            rows = torch.as_tensor(perm[s : s + BATCH], device=dev)
            loss = info_nce(model(x_tr[rows]), y_tr[rows], tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        model.eval()
        va = val_acc1(model, x_val, y_val)
        curve.append({"epoch": epoch, "loss": float(np.mean(losses)), **va})
        if first_epoch_wall is None:
            first_epoch_wall = time.time() - te0
            logger.info("[fit] %s first-epoch wall %.1fs (pilot basis)", slug, first_epoch_wall)
        print(
            f"[fit] unit {epoch + 1}/{args.max_epochs} {slug} loss={np.mean(losses):.4f} "
            f"val_cos_acc1={va['cos_acc1']:.4f} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        if va["cos_acc1"] > best["acc"]:
            best = {
                "acc": va["cos_acc1"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        elif epoch - best["epoch"] >= PATIENCE:
            logger.info("[fit] %s early stop at epoch %d (best %d)", slug, epoch, best["epoch"])
            break
    assert best["state"] is not None
    torch.save(
        {"family": family, "tau": tau, "state_dict": best["state"], "meta": meta_block()},
        fits_dir / f"{slug}.pt",
    )
    rec = {
        "family": family,
        "tau": tau,
        "best_epoch": best["epoch"],
        "best_val_cos_acc1": best["acc"],
        "epochs_realized": len(curve),
        "first_epoch_wall_s": first_epoch_wall,
        "total_wall_s": time.time() - t0,
        "curve": curve,
        "fit_regime": _fit_regime(args),
        "meta": meta_block(),
    }
    atomic_json(jpath, rec)
    return rec


def phase_fit(args) -> None:
    logger.info("[phase=fit] start (smoke=%s)", args.smoke)
    mm, ci, _meta = assemble(args)
    sets = load_split_sets(args, ci)
    tr, val = sets["train"], sets["val"]
    n_tr = len(tr)
    assert n_tr > H_DIM, (
        f"n_train={n_tr} < d={H_DIM}: estimator-degenerate — refusing (dispatch-note "
        "estimator-validity duty)"
    )
    logger.info("[fit] n_train=%d n_val=%d d=%d (n >> d, well-posed)", n_tr, len(val), H_DIM)
    payload = load_ridge_payload(args)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr = torch.as_tensor(np.asarray(mm[("cx", LAYER)][tr], dtype=np.float32), device=dev)
    y_tr = torch.as_tensor(np.asarray(mm[("vx", LAYER)][tr], dtype=np.float32), device=dev)
    x_val = torch.as_tensor(np.asarray(mm[("cx", LAYER)][val], dtype=np.float32), device=dev)
    y_val = torch.as_tensor(np.asarray(mm[("vx", LAYER)][val], dtype=np.float32), device=dev)
    assert x_tr.shape == (n_tr, H_DIM), x_tr.shape
    taus = (0.1,) if args.smoke else TAUS
    for family in FAMILIES:
        for tau in taus:
            fit_one(args, family, tau, x_tr, y_tr, x_val, y_val, payload, dev)


# ── battery ────────────────────────────────────────────────────────────────────────


def _wh(x: np.ndarray, mu: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """z = L⁻¹(x − μ) — the failchar ``build_spaces`` whiten (plan-§11 A20)."""
    return solve_triangular(ell, (np.asarray(x, np.float64) - mu).T, lower=True).T


def _pooled_r2(pred: np.ndarray, true: np.ndarray) -> float:
    mu = true.mean(axis=0)
    return 1.0 - float(((true - pred) ** 2).sum() / ((true - mu) ** 2).sum())


def select_best(args, family: str) -> dict:
    fits_dir = _out_root(args) / "fits"
    taus = (0.1,) if args.smoke else TAUS
    recs = []
    for tau in taus:
        p = fits_dir / f"{family}_tau{tau:g}.json"
        assert p.exists(), f"missing fit record {p} — run --phase fit first"
        recs.append(json.loads(p.read_text()))
    best = max(recs, key=lambda r: r["best_val_cos_acc1"])
    logger.info(
        "[battery] %s selected tau=%g (val cos acc@1 %.4f over grid %s)",
        family,
        best["tau"],
        best["best_val_cos_acc1"],
        [(r["tau"], round(r["best_val_cos_acc1"], 4)) for r in recs],
    )
    return best


@torch.no_grad()
def predict_holdout(
    args, family: str, tau: float, cx_hold: np.ndarray, payload: dict
) -> np.ndarray:
    ckpt = torch.load(
        _out_root(args) / "fits" / f"{family}_tau{tau:g}.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = ContrastiveMap(family, payload)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    out = []
    for s in range(0, len(cx_hold), 4096):
        xb = torch.as_tensor(cx_hold[s : s + 4096], dtype=torch.float32, device=dev)
        out.append(model(xb).cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


def phase_battery(args) -> None:
    logger.info("[phase=battery] start")
    staged = _staged(args)
    yd = np.load(staged / "y_holdout_L19.npz")
    y16, yci = yd["y16"].astype(np.float64), np.asarray(yd["ci"], dtype=np.int64)
    assert len(yci) == EXPECTED_N, len(yci)
    cxd = np.load(staged / "cx_holdout_L19.npz")
    cx_hold, cci = cxd["cx"].astype(np.float64), np.asarray(cxd["ci"], dtype=np.int64)
    assert (cci == yci).all(), "cx_holdout/y_holdout ci misalign"
    stats = np.load(staged / "whiten_stats.npz")
    mu_a = np.asarray(stats["mu_A"], dtype=np.float64)
    ell = np.asarray(stats["L"], dtype=np.float64)
    payload = load_ridge_payload(args)

    arms: dict[str, np.ndarray] = {}
    for name, fn in (
        ("ridge_banked", "pred16_ridge.npz"),
        ("mlp_mse_banked", "pred16_mlp_w8192.npz"),
    ):
        d = np.load(staged / fn)
        assert (np.asarray(d["ci"], dtype=np.int64) == yci).all(), f"{fn} ci misalign"
        arms[name] = d["pred16"].astype(np.float64)
    selected = {}
    for family in FAMILIES:
        best = select_best(args, family)
        selected[family] = best
        arms[f"contrastive_{family}"] = predict_holdout(args, family, best["tau"], cx_hold, payload)

    y_wh = _wh(y16, mu_a, ell)
    results: dict = {}
    for arm, pred in arms.items():
        assert pred.shape == y16.shape, (arm, pred.shape)
        pred_wh = _wh(pred, mu_a, ell)
        results[arm] = {
            "raw_euclidean": knn_retrieval(pred, y16, ks=KS, metric="euclidean"),
            "raw_cos": knn_retrieval(pred, y16, ks=KS, metric="cosine"),
            "whiten": knn_retrieval(pred_wh, y_wh, ks=KS, metric="euclidean"),
            "whiten_cos": knn_retrieval(pred_wh, y_wh, ks=KS, metric="cosine"),
            "holdout_r2": _pooled_r2(pred, y16),
        }
        print(
            f"[battery] {arm}: raw acc@1={results[arm]['raw_euclidean']['acc_at_k'][1]:.4f} "
            f"whiten_cos acc@1={results[arm]['whiten_cos']['acc_at_k'][1]:.4f} "
            f"R2={results[arm]['holdout_r2']:.4f}",
            flush=True,
        )

    # banked-value reconciliation (report-only; the hard repro gate ran in #2202)
    banked = json.loads(repo_banked_path(BANKED_REL).read_text())
    banked_cell = banked["cells"]["context_L19"]["knn"]["ridge"]["euclidean"]
    fits_summary = json.loads(repo_banked_path(FITS_SUMMARY_REL).read_text())
    banked_r2 = {
        k: fits_summary["cells"][f"context_L19_{k}"]["holdout_r2"] for k in ("ridge", "mlp_w8192")
    }
    recon = {
        "ridge_raw_acc1_recomputed": results["ridge_banked"]["raw_euclidean"]["acc_at_k"][1],
        "ridge_raw_acc1_banked": banked_cell["acc_at_k"]["1"],
        "holdout_r2_banked": banked_r2,
    }

    out = {
        "results": results,
        "selected": {
            f: {
                k: selected[f][k]
                for k in (
                    "tau",
                    "best_epoch",
                    "best_val_cos_acc1",
                    "epochs_realized",
                    "first_epoch_wall_s",
                    "total_wall_s",
                )
            }
            for f in FAMILIES
        },
        "banked_reconciliation": recon,
        "n_holdout": int(len(yci)),
        "whiten_stats": {"lam": float(stats["lam"]), "n_train": int(stats["n_train"])},
        "meta": meta_block({"revision_pin": args.revision, "smoke": bool(args.smoke)}),
    }
    ev = _out_root(args) / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    atomic_json(ev / "contrastive_maps_battery.json", out)
    logger.info("[battery] wrote %s", ev / "contrastive_maps_battery.json")


# ── upload ─────────────────────────────────────────────────────────────────────────


def phase_upload(args) -> None:
    """One bulk exact-set-verified upload of the out-root to HF (fits + eval)."""
    if args.smoke:
        logger.info("[upload] SKIPPED (smoke)")
        return
    root = _out_root(args)
    rel = sorted(
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and "smoke" not in p.parts
    )
    assert rel, f"nothing to upload under {root}"
    dest = f"{HF_PREFIX_2202}/{UPLOAD_SUBPREFIX}"
    url = hub._upload_folder_filtered(
        root,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        allow_patterns=rel,
        expected_repo_paths=[f"{dest}/{r}" for r in rel],
    )
    if not url:
        raise RuntimeError(f"upload to {dest} returned no URL ({len(rel)} files)")
    logger.info("[upload] %d files -> %s", len(rel), dest)


PHASES = {
    "stage": stage_inputs,
    "fit": phase_fit,
    "battery": phase_battery,
    "upload": phase_upload,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="all", choices=[*PHASES, "all"])
    ap.add_argument("--work-root", default="/workspace/work/issue2202_contrastive")
    ap.add_argument("--out-root", default="/workspace/outputs/issue2202_contrastive")
    ap.add_argument("--revision", default=HF_PIN, help="data-repo revision pin for #1738 inputs")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--smoke", action="store_true", help="2-epoch, single-tau, no-upload run")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.smoke:
        args.max_epochs = min(args.max_epochs, 2)
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    for ph in phases:
        logger.info("[phase=%s] dispatch", ph)
        PHASES[ph](args)
    # explicit exit before C-extension finalization (phased-dispatcher rc race)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
