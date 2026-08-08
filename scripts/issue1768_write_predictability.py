#!/usr/bin/env python3
# ruff: noqa: RUF002
"""#1768 inline free-analysis round: is the per-context write w(x) PREDICTABLE from c0(x)?

Round 1 (#1768 clean-result) established that the fine-tuning write is
~10-dimensional at fixed text (participation ratio ~10; top-1 centered-SVD share
0.29 matched-text / 0.09 on-policy) but tested only ZERO-PARAMETER predictors
(fixed candidate directions δ / r_B, a similarity gate). It never fitted a map.
This round fits maps c0(x) -> w(x) at layer 19 and asks two questions:

  Q1  Is w(x) predictable from the BASE context vector at all?  (held-out R² > 0
      means context-DEPENDENT predictability beyond the static mean write, because
      variance-weighted R² scores against the TEST-set mean of w by construction.)
  Q2  Is that map LINEAR?  (ridge vs Nystrom-RBF KRR vs MLP.)

Definitions (layer 19, span-mean activations, fp16 stores, sha-joined rows):
  c0(x)    base-arm CONTEXT vector                (corpus_capture/<base>/pooled.pt)
  w_op(x)  trained on-policy ANSWER − base ANSWER (corpus_capture/<arm> − base)
  w_tf(x)  trained matched-text ANSWER − base ANSWER (corpus_capture_tf/<arm> − base)

REUSE (never re-implement): the sha-keyed row join + pinned-split machinery is
``issue1768_fit.load_corpus_cell`` verbatim (round 1's loader); the ridge is round
1's grid-edge-extending ``issue1768_fit._fit_map`` over the #779 streaming primal
path (ONE eigh of the streamed (H,H) X^TX, all 23 lambdas off that factorization);
KRR + MLP are the #779 fitter-fair-comparison helpers (``krr_select_predict`` /
``batched_mlp_fit``) that #1073's analogous inline round reused for this exact
3584->3584 full-dim shape; identity+bias and kNN retrieval are the standing
``analysis.mapping_baselines`` reads.

Phases (each resumable; per-cell JSON written the moment a cell completes):
  stage  pick the 8 arms from the committed verdict/summary manifests, stage the
         17-18 pooled stores off `/` via the scoped-listing Hub helpers (#833)
  fits   per (arm, tree): ridge + identity+bias + kNN + KRR + per-direction reads
  mlp    batched padded-bmm AdamW MLP over the selected cells (pilot-gated)
  figs   the three figures
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before torch/numpy: shared-VM thread caps + HF/WandB creds

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1768_fit as FIT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.writepred")

LAYER = 19  # plan §4.3 / #779 scaling-curve anchor (the round-1 headline layer)
HIDDEN_DIM = X.HIDDEN  # 3584 — both the input (c0) and output (w) dimension
HF_REVISION = "c07267285d2cdbf3e0401ddc3e3accae50e496a7"  # pinned (dispatch note)
RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_1768" / "write_predictability"
FIGS_DIR = REPO_ROOT / "figures" / "issue_1768" / "write_predictability"
# Multi-GB staging NEVER on `/` (CLAUDE.md disk-hygiene); the repo-root
# data/issue_1768/hf_dl resolves ON `/` until the #681 worktree bind lands.
STAGE_ROOT = Path(
    os.environ.get(
        "EPM_I1768_WP_STAGE",
        f"/mnt/eps-data/{os.environ.get('USER', 'thomasjiralerspong')}"
        "/issue1768_write_predictability/hf_dl",
    )
)

TREES = ("op", "tf")  # on-policy vs matched-text (teacher-forced) write
N_BOOT = 200  # batched subset-sum bootstrap draws (see _boot_r2_ci)
N_DIRECTIONS = 10  # the round-1 "~10-dimensional write" subspace
SVD_Q = 20  # svd_lowrank rank (> N_DIRECTIONS for accuracy of the top 10)
KNN_KS = (1, 10)

# MLP recipe: #779 FFC constants (MLP_MAX_EPOCHS=300, MLP_PATIENCE=20, MLP_WD=1e-4).
MLP_WIDTH = 3584  # full-dim hidden (dispatch: width >= 3584)
MLP_LR = 1e-3


def _phase(name: str) -> None:
    print(f"[phase={name}]", flush=True)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _meta(extra: dict | None = None) -> dict:
    import torch

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "issue": X.ISSUE,
        "layer": LAYER,
        "hf_revision": HF_REVISION,
        "round": "inline-write-predictability",
    }
    if extra:
        out.update(extra)
    return out


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


_DEVICE_OVERRIDE: str | None = None


def _device():
    import torch

    if _DEVICE_OVERRIDE:
        return torch.device(_DEVICE_OVERRIDE)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ── arm picks (deterministic, from the committed manifests) ───────────────────

# criterion -> (predicate, tie-break description). Every pick is recorded with
# its criterion + the resolved slug + why, per the dispatch contract.
PICK_CRITERIA: tuple[tuple[str, str], ...] = (
    ("imp_lora_con_highdose", "highest-dose (max lr) impoliteness LoRA contrastive, pers context"),
    ("imp_lora_po_highdose", "highest-dose (max lr) impoliteness LoRA positive-only, pers context"),
    ("ws_lora_con", "writing-style (cas) LoRA contrastive, pers context, max lr"),
    ("syc_bare_changed", "sycophancy bare-context LoRA arm whose L19 verdict is Changed"),
    (
        "syc_conv_unchanged",
        "sycophancy conversation-history LoRA arm whose L19 verdict is Unchanged",
    ),
    ("marker_pers", "marker persona-context LoRA arm (lowest in-window lr rung), max |ΔG|"),
    ("imp_ft", "full fine-tune impoliteness arm (method axis vs the LoRA imp arms)"),
    ("syc_ft", "full fine-tune sycophancy arm (method axis vs the LoRA syc arms)"),
)


def _verdicts() -> dict:
    """The committed round-1 p8 verdict summary, keyed `<arm_id>_L<layer>`."""
    path = REPO_ROOT / "eval_results" / "issue_1768" / "map_change_summary.json"
    return json.loads(path.read_text())["verdicts"]


def pick_arms() -> dict:
    """Resolve the 8 dispatch-named arms from the registry + L19 verdicts."""
    arms = {a.arm_id: a for a in X.all_arms()}
    vd = _verdicts()

    def v(arm_id: str) -> dict:
        return vd.get(f"{arm_id}_L{LAYER}", {})

    def cands(**kw):
        out = []
        for a in arms.values():
            if all(getattr(a, k) == val for k, val in kw.items()):
                out.append(a)
        return out

    def hi_lr(cs):
        # deterministic: max lr, then max selection_read, then arm_id
        return sorted(cs, key=lambda a: (-a.lr, -abs(a.selection_read or 0.0), a.arm_id))[0]

    picks: dict[str, dict] = {}

    def record(crit: str, arm, why: str) -> None:
        vv = v(arm.arm_id)
        picks[crit] = {
            "criterion": crit,
            "criterion_text": dict(PICK_CRITERIA)[crit],
            "arm_id": arm.arm_id,
            "method": arm.method,
            "kind": arm.kind,
            "beh_key": arm.beh_key,
            "ctx_key": arm.ctx_key,
            "regime": arm.regime,
            "seed": arm.seed,
            "lr": arm.lr,
            "step": arm.step,
            "selection_read": arm.selection_read,
            "base_unit": X.base_unit_for(arm.arm_id),
            "round1_L19_verdict": vv.get("verdict"),
            "round1_L19_D": vv.get("D"),
            "round1_L19_m0_r2": vv.get("m0_r2"),
            "round1_L19_mplus_r2": vv.get("mplus_r2"),
            "why": why,
        }

    a = hi_lr(cands(beh_key="imp", ctx_key="pers", regime="con", method="lora"))
    record("imp_lora_con_highdose", a, f"max lr {a.lr:g} among imp/pers/con LoRA arms")
    a = hi_lr(cands(beh_key="imp", ctx_key="pers", regime="po", method="lora"))
    record("imp_lora_po_highdose", a, f"max lr {a.lr:g} among imp/pers/po LoRA arms")
    a = hi_lr(cands(beh_key="cas", ctx_key="pers", regime="con", method="lora"))
    record("ws_lora_con", a, f"writing-style (cas) pers/con LoRA, max lr {a.lr:g}")

    syc_bare = [
        c
        for c in cands(beh_key="syc", ctx_key="bare", method="lora")
        if v(c.arm_id).get("verdict") == "Changed"
    ]
    assert syc_bare, "no syc bare-context arm with L19 verdict Changed"
    a = sorted(syc_bare, key=lambda c: (-(v(c.arm_id).get("D") or 0.0), c.arm_id))[0]
    record(
        "syc_bare_changed", a, f"largest L19 D {v(a.arm_id)['D']:.3f} among Changed syc/bare arms"
    )

    syc_conv = [
        c
        for c in cands(beh_key="syc", ctx_key="conv", method="lora")
        if v(c.arm_id).get("verdict") == "Unchanged"
    ]
    assert syc_conv, "no syc conversation-history arm with L19 verdict Unchanged"
    a = sorted(syc_conv, key=lambda c: (v(c.arm_id).get("D") or 0.0, c.arm_id))[0]
    record(
        "syc_conv_unchanged",
        a,
        f"most-negative L19 D {v(a.arm_id)['D']:.3f} among Unchanged syc/conv",
    )

    mk = cands(beh_key="mk", ctx_key="pers", method="lora")
    assert mk, "no marker pers LoRA arm"
    a = sorted(mk, key=lambda c: (c.lr, -abs(c.selection_read or 0.0), c.arm_id))[0]
    record(
        "marker_pers", a, f"lowest in-window lr rung {a.lr:g}, |ΔG| {abs(a.selection_read):.2f} nat"
    )

    for crit, beh in (("imp_ft", "imp"), ("syc_ft", "syc")):
        fts = cands(beh_key=beh, method="ft", regime="con")
        assert fts, f"no full-FT {beh} con arm"
        a = sorted(fts, key=lambda c: c.arm_id)[0]
        record(crit, a, f"full-FT {beh} contrastive arm (step {a.step}) — method axis")

    assert len(picks) == len(PICK_CRITERIA), (len(picks), len(PICK_CRITERIA))
    ids = [p["arm_id"] for p in picks.values()]
    assert len(set(ids)) == len(ids), f"duplicate arm picks: {ids}"
    return {
        "criteria": [{"criterion": c, "text": t} for c, t in PICK_CRITERIA],
        "picks": [picks[c] for c, _ in PICK_CRITERIA],
        "verdict_source": "eval_results/issue_1768/map_change_summary.json (round-1 p8)",
        "registry_source": "scripts/issue1768_cells.py all_arms() (72-arm fleet)",
    }


# ── staging (scoped listing + per-file atomic download; NEVER snapshot_download
#    on the ~1M-file data repo — it walks the whole tree first, #833) ──────────


def _staged_targets(picks: dict) -> list[tuple[str, Path]]:
    """(path_in_repo, local target) for every store this round consumes."""
    out: list[tuple[str, Path]] = []
    pref = X.HF_PREFIX
    out.append((f"{pref}/inputs/corpus_sample.json", STAGE_ROOT / "inputs" / "corpus_sample.json"))
    bases = sorted({p["base_unit"] for p in picks["picks"]})
    for b in bases:
        out.append(
            (
                f"{pref}/corpus_capture/{b}/pooled.pt",
                STAGE_ROOT / "corpus_capture" / b / "pooled.pt",
            )
        )
    for p in picks["picks"]:
        arm = p["arm_id"]
        out.append(
            (
                f"{pref}/corpus_capture/{arm}/pooled.pt",
                STAGE_ROOT / "corpus_capture" / arm / "pooled.pt",
            )
        )
        out.append(
            (
                f"{pref}/corpus_capture_tf/{arm}/pooled_tf.pt",
                STAGE_ROOT / "corpus_capture_tf" / arm / "pooled_tf.pt",
            )
        )
    return out


def phase_stage(picks: dict) -> dict:
    """Verify + stage every consumed store; fail loud on a missing path."""
    _phase("stage")
    from explore_persona_space.orchestrate import hub

    targets = _staged_targets(picks)
    # ONE scoped listing per prefix (server-side), then per-file atomic staging.
    sizes: dict[str, int] = {}
    from huggingface_hub import HfApi

    api = HfApi()
    for pref in (
        f"{X.HF_PREFIX}/corpus_capture",
        f"{X.HF_PREFIX}/corpus_capture_tf",
        f"{X.HF_PREFIX}/inputs",
    ):
        for e in hub.retry_transient(
            # The scoped path-only helper list_hf_files_under_path cannot be used here:
            # this listing's per-entry .size feeds the pre-download 1.5x disk-headroom
            # assert and the staging report. The call IS retried (enclosing thunk).
            lambda p=pref: list(
                api.list_repo_tree(  # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient
                    X.HF_DATA_REPO,
                    path_in_repo=p,
                    repo_type="dataset",
                    recursive=True,
                    revision=HF_REVISION,
                )
            ),
            what=f"list_repo_tree {pref}",
        ):
            if getattr(e, "size", None) is not None:
                sizes[e.path] = int(e.size)
    missing = [p for p, _ in targets if p not in sizes]
    assert not missing, f"paths absent at revision {HF_REVISION}: {missing}"

    projected = sum(sizes[p] for p, t in targets if not t.exists())
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(STAGE_ROOT)
    free = st.f_bavail * st.f_frsize
    need = int(1.5 * projected)
    logger.info(
        "[stage] projected %.2f GB, need(1.5x) %.2f GB, free %.2f GB on %s",
        projected / 1e9,
        need / 1e9,
        free / 1e9,
        STAGE_ROOT,
    )
    assert free >= need, (
        f"insufficient headroom: free {free / 1e9:.1f} GB < need {need / 1e9:.1f} GB"
    )

    t0 = time.time()
    for i, (path_in_repo, target) in enumerate(targets, 1):
        if target.exists():
            logger.info("[stage] %d/%d present %s", i, len(targets), target.name)
            continue
        tw = time.time()
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            path_in_repo,
            target,
            repo_type="dataset",
            revision=HF_REVISION,
        )
        logger.info(
            "[stage] unit %d/%d %s (%.0f MB) elapsed=%.0fs",
            i,
            len(targets),
            path_in_repo.split("/", 1)[1],
            sizes[path_in_repo] / 1e6,
            time.time() - tw,
        )
    report = {
        "stage_root": str(STAGE_ROOT),
        "n_files": len(targets),
        "bytes": sum(sizes[p] for p, _ in targets),
        "projected_downloaded_bytes": projected,
        "free_bytes_before": free,
        "wall_s": round(time.time() - t0, 1),
        "files": [{"path_in_repo": p, "bytes": sizes[p]} for p, _ in targets],
    }
    _atomic_json(RESULTS_DIR / "staging.json", {**report, **_meta()})
    return report


# ── reads ────────────────────────────────────────────────────────────────────


def _boot_r2_ci(pred: np.ndarray, y: np.ndarray, n_draws: int, seed: int) -> list[float]:
    """Row-bootstrap CI for variance-weighted R², as TWO batched GEMMs.

    A per-draw Python loop re-reduces the (n_te, D) matrices every draw (the
    #778 anti-pattern). Instead: per-row squared error ``se`` and per-row
    ``||y||²`` are precomputed ONCE, and every draw's numerator/denominator
    follows from the subset-sum identity — the draw's target mean is
    ``(S @ y)/n`` for the (n_draws, n) multiplicity matrix ``S``, so
    ``den = Σ||y||² − n·||mean||²`` needs one (n_draws, n) @ (n, D) GEMM.
    """
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    se = ((pred - y) ** 2).sum(axis=1)
    yn = (y**2).sum(axis=1)
    idx = rng.integers(0, n, size=(n_draws, n))
    num = se[idx].sum(axis=1)
    ysum = yn[idx].sum(axis=1)
    S = np.zeros((n_draws, n))
    np.add.at(S, (np.repeat(np.arange(n_draws), n), idx.ravel()), 1.0)
    M = (S @ y) / n
    den = ysum - n * (M**2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = 1.0 - num / den
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def _reads(pred: np.ndarray, y: np.ndarray, *, seed: int, with_ci: bool) -> dict:
    """Held-out reads for one predictor: R² (+CI), mean cosine, kNN retrieval."""
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    n_pool = y.shape[0]
    ks = tuple(k for k in KNN_KS if k <= n_pool) or (1,)
    out = {
        "heldout_r2": FIT._pooled_r2(pred, y),
        "mean_cos": FIT._mean_cos(pred, y),
        "knn_euclidean": knn_retrieval(pred, y, ks=ks, metric="euclidean"),
        "knn_cosine": knn_retrieval(pred, y, ks=ks, metric="cosine"),
    }
    if with_ci:
        out["heldout_r2_ci95"] = _boot_r2_ci(pred, y, N_BOOT, seed)
        out["r2_ci_n_draws"] = N_BOOT
    return out


def _write_subspace(W_tr: np.ndarray, q: int = SVD_Q) -> dict:
    """Top-``N_DIRECTIONS`` CENTERED write directions + their train variance shares.

    Centering matches round 1's ``issue1768_directions.rank_read`` convention
    (top-1 share / participation ratio of CENTERED Δv), so "direction j" here is
    the same object round 1's ~10-dimensional read describes. Uses
    ``svd_lowrank`` (the round-1 helper's own choice) rather than a full
    (3584, 3584) Gram eigh — ~1 s vs ~25 s per cell at identical top-10 accuracy.
    """
    import torch

    mu = W_tr.mean(axis=0)
    Wc = torch.as_tensor(W_tr - mu, dtype=torch.float32)
    total = float((Wc**2).sum().item())
    q = int(min(q, min(Wc.shape) - 1))
    _u, s, V = torch.svd_lowrank(Wc, q=max(q, N_DIRECTIONS), niter=4)
    s2 = (s.double() ** 2).numpy()
    Vn = V.double().numpy()[:, :N_DIRECTIONS]
    return {
        "mu": mu,
        "V": Vn,  # (D, N_DIRECTIONS) right singular vectors = activation-space dirs
        "var_share": [float(x / total) for x in s2[:N_DIRECTIONS]],
        "total_centered_var": total,
        "participation_ratio": float((s2.sum() ** 2) / (s2**2).sum()),
        "centering": "train-mean centered (round-1 rank_read convention)",
        "svd": f"torch.svd_lowrank(q={max(q, N_DIRECTIONS)}, niter=4)",
    }


def _per_direction_r2(sub: dict, pred_te: np.ndarray, W_te: np.ndarray) -> dict:
    """Held-out R² of the prediction PROJECTED onto each write direction.

    Answers "which of the ~10 write directions are context-predictable": each
    direction's scalar coefficient is scored against the TEST mean of that
    coefficient, so a direction the map only reproduces on average reads ~0.
    Left singular vectors are ROW-space (per-train-row loadings) and have no
    held-out analogue — only the right vectors transfer, stated in the output.
    """
    V = sub["V"]
    mu = sub["mu"]
    pt = (W_te - mu) @ V
    pp = (pred_te - mu) @ V
    r2 = []
    for j in range(V.shape[1]):
        resid = float(((pp[:, j] - pt[:, j]) ** 2).sum())
        tot = float(((pt[:, j] - pt[:, j].mean()) ** 2).sum())
        r2.append(1.0 - resid / tot if tot > 0 else float("nan"))
    return {
        "direction_rank": list(range(1, V.shape[1] + 1)),
        "heldout_r2_per_direction": r2,
        "train_var_share_per_direction": sub["var_share"],
        "participation_ratio_train": sub["participation_ratio"],
        "total_centered_var_train": sub["total_centered_var"],
        "basis": "top-10 right singular vectors of the CENTERED train write matrix",
        "left_vectors": "N/A — row-space (per-train-row loadings); no held-out analogue",
        "centering": sub["centering"],
        "svd": sub["svd"],
    }


# ── per-cell fits ────────────────────────────────────────────────────────────


def load_arm_cell(arm_id: str) -> dict:
    """Round-1 loader verbatim, then the two write targets."""
    cell = FIT.load_corpus_cell(arm_id, LAYER, STAGE_ROOT)
    cell["w_op"] = cell["Vplus"] - cell["V0"]
    cell["w_tf"] = cell["Vplus_tf"] - cell["V0"]
    return cell


def fit_cell(arm_id: str, tree: str, cell: dict, dev, *, do_krr: bool) -> dict:
    """All predictors + reads for one (arm, tree) cell. Persisted by the caller."""
    import issue779_fitter_fair_comparison as f779
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    C0 = cell["C0"]
    W = cell["w_op"] if tree == "op" else cell["w_tf"]
    tr, val, te = FIT._split_idx(cell["split"])
    walls: dict[str, float] = {}

    t0 = time.time()
    pred_te, ridge_meta, _payload = FIT._fit_map(C0, W, tr, val, te, dev)
    walls["ridge_s"] = round(time.time() - t0, 2)
    ridge = {**ridge_meta, **_reads(pred_te, W[te], seed=X.FLOOR_SEED, with_ci=True)}

    t0 = time.time()
    pred_ib = identity_bias_predict(C0[tr], W[tr], C0[te])
    ident = {
        "applicable": True,
        "note": (
            "d_in == d_out == 3584 so the standing identity+bias baseline applies; "
            "conceptually a weak prior here (w is a DIFFERENCE vector, c0 an "
            "activation), reported per the mapping-baselines rule, never skipped"
        ),
        **_reads(pred_ib, W[te], seed=X.FLOOR_SEED + 7, with_ci=True),
    }
    walls["identity_bias_s"] = round(time.time() - t0, 2)

    krr = {"skipped": "not requested"}
    if do_krr:
        t0 = time.time()
        res = f779.krr_select_predict(
            C0[tr],
            W[tr],
            C0[val],
            W[val],
            C0[te],
            gamma_mult=f779.KRR_GAMMA_MULT,
            lambdas=f779.KRR_LAMBDAS,
            m_landmarks=f779.KRR_LANDMARKS,
            seed=X.FLOOR_SEED,
            dev=dev,
        )
        krr = {
            "selected": res["selected"],
            "base_gamma": res["base_gamma"],
            "m_landmarks": res["m_landmarks"],
            "recipe": "Nystrom RBF KRR (#779 FFC krr_select_predict), val-selected (gamma, lambda)",
            **_reads(res["pred_te"], W[te], seed=X.FLOOR_SEED + 13, with_ci=True),
        }
        walls["krr_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    sub = _write_subspace(W[tr])
    per_dir = _per_direction_r2(sub, pred_te, W[te])
    per_dir["predictor"] = "ridge (val-selected lambda)"
    walls["per_direction_s"] = round(time.time() - t0, 2)

    return {
        "arm_id": arm_id,
        "tree": tree,
        "tree_text": (
            "on-policy write: trained on-policy answer state − base answer state"
            if tree == "op"
            else "matched-text write: trained teacher-forced-on-base answer state − base answer state"
        ),
        "layer": LAYER,
        "method": X.arm_method(arm_id),
        "n_rows": int(len(cell["sha"])),
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
        "n_test": int(len(te)),
        "d_in": int(C0.shape[1]),
        "d_out": int(W.shape[1]),
        "write_norm_mean_test": float(np.linalg.norm(W[te], axis=1).mean()),
        "predictors": {"ridge": ridge, "identity_bias": ident, "krr": krr},
        "per_direction": per_dir,
        "walls": walls,
        "r2_convention": (
            "variance-weighted pooled R² over the 3,584 dims of w, scored against the "
            "TEST-set mean of w — so R² > 0 is context-DEPENDENT predictability beyond "
            "the static mean write (a mean-write predictor reads exactly 0)"
        ),
    }


def phase_fits(picks: dict, *, cells_limit: int | None, do_krr: bool) -> None:
    _phase("fits")
    dev = _device()
    logger.info("[fits] device %s", dev)
    order = [(p["arm_id"], t) for p in picks["picks"] for t in TREES]
    if cells_limit:
        order = order[:cells_limit]
    dest_dir = RESULTS_DIR / "cells"
    todo_arms = sorted({a for a, t in order if not (dest_dir / f"{a}__{t}.json").exists()})
    logger.info("[fits] %d cells, %d arms to load", len(order), len(todo_arms))
    k = 0
    cache: dict[str, dict] = {}
    for arm_id, tree in order:
        k += 1
        dest = dest_dir / f"{arm_id}__{tree}.json"
        if dest.exists():
            logger.info("[fits] unit %d/%d %s|%s resume-skip", k, len(order), arm_id, tree)
            continue
        t0 = time.time()
        if arm_id not in cache:
            cache.clear()  # one arm's arrays at a time (~3.6 GB fp64 peak)
            tl = time.time()
            cache[arm_id] = load_arm_cell(arm_id)
            logger.info("[fits] loaded %s in %.0fs", arm_id, time.time() - tl)
        res = fit_cell(arm_id, tree, cache[arm_id], dev, do_krr=do_krr)
        _atomic_json(dest, {**res, **_meta()})
        print(
            f"[fits] unit {k}/{len(order)} {arm_id}|{tree} "
            f"ridge_r2={res['predictors']['ridge']['heldout_r2']:.4f} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )


# ── MLP leg (ONE big net per cell, run SEQUENTIALLY over cells) ──────────────


def _mlp_recipe(width: int, max_epochs: int, dev) -> dict:
    import issue779_fitter_fair_comparison as f779

    return {
        "shape": (
            f"Linear({HIDDEN_DIM}, {width}) -> GELU -> Linear({width}, {HIDDEN_DIM}), AdamW, "
            "internal 10% val early stop — ONE net per cell (G=1), cells run SEQUENTIALLY"
        ),
        "helper": (
            "issue779_fitter_fair_comparison.batched_mlp_fit called with a SINGLE group per "
            "cell: at G=1 the padded-bmm degenerates to plain saturated GEMMs, i.e. exactly "
            "the standard full-batch net. NOT analysis/vectorized_mlp_skill.py — that helper's "
            "exactness contract is scalar-output members (Linear(hid, 1) per group x dim x "
            "fold), built for the #722 overhead-bound many-tiny-fits regime; mapping our "
            f"{HIDDEN_DIM}-dim output head onto members would mean ~{HIDDEN_DIM} members per "
            "cell each carrying its own d_in x hid W1 (memory-absurd). This problem is the "
            "opposite regime: one big net per cell at n=15,000, FLOP-bound."
        ),
        "cell_batching": (
            "none by design — cell-batching buys nothing when FLOP-bound and multiplies "
            "resident memory; each cell is its own fit"
        ),
        "hidden": width,
        "width_choice": (
            f"single width {width} >= output dim {HIDDEN_DIM} so the readout is not "
            "rank-limited; a width SWEEP was not run (recorded deviation, time cap)"
        ),
        "lr": MLP_LR,
        "wd": f779.MLP_WD,
        "max_epochs": max_epochs,
        "patience": f779.MLP_PATIENCE,
        "device": str(dev),
    }


def _mlp_cell_order(cell_keys: list[str]) -> list[str]:
    """tf before op within an arm (tf is the weights-carried object), arms adjacent
    so each arm's stores load ONCE."""
    return sorted(cell_keys, key=lambda k: (k.split("|")[0], k.split("|")[1] != "tf"))


def phase_mlp(
    picks: dict, *, cell_keys: list[str], max_epochs: int, pilot: bool, width: int
) -> None:
    """One full-batch MLP per cell, sequential; per-cell JSON the moment it lands."""
    _phase("mlp_pilot" if pilot else "mlp")
    import issue779_fitter_fair_comparison as f779

    dev = _device()
    tag = "mlp_pilot_cells" if pilot else "mlp_cells"
    dest_dir = RESULTS_DIR / tag
    order = _mlp_cell_order(cell_keys)
    logger.info(
        "[mlp] %d cells on %s (width=%d, max_epochs=%d)", len(order), dev, width, max_epochs
    )
    cache: dict[str, dict] = {}
    for k, key in enumerate(order, 1):
        dest = dest_dir / f"{key.replace('|', '__')}.json"
        if dest.exists():
            logger.info("[mlp] unit %d/%d %s resume-skip", k, len(order), key)
            continue
        arm_id, tree = key.split("|")
        if arm_id not in cache:
            cache.clear()  # one arm's arrays at a time
            cache[arm_id] = load_arm_cell(arm_id)
        cell = cache[arm_id]
        W = cell["w_op"] if tree == "op" else cell["w_tf"]
        tr, _val, te = FIT._split_idx(cell["split"])
        t0 = time.time()
        res = f779.batched_mlp_fit(
            [f779.MLPGroup((arm_id, tree), cell["C0"][tr], W[tr], width, MLP_LR)],
            hidden=width,
            lr=MLP_LR,
            max_epochs=max_epochs,
            dev=dev,
        )
        r = res[(arm_id, tree)]
        wall = time.time() - t0
        pred = r.predict(cell["C0"][te])
        row = {
            "arm_id": arm_id,
            "tree": tree,
            "epochs_ran": int(r.epochs_ran),
            "best_internal_val": float(r.best_val),
            "wall_s": round(wall, 1),
            "recipe": _mlp_recipe(width, max_epochs, dev),
            **_reads(pred, W[te], seed=X.FLOOR_SEED + 21, with_ci=True),
            **_meta(),
        }
        _atomic_json(dest, row)
        print(
            f"[mlp] unit {k}/{len(order)} {key} r2={row['heldout_r2']:.4f} "
            f"epochs={r.epochs_ran} elapsed={wall:.0f}s",
            flush=True,
        )
    # aggregate (idempotent) — the file figures/summary read
    cells = {}
    for p in sorted(dest_dir.glob("*.json")):
        d = json.loads(p.read_text())
        cells[f"{d['arm_id']}|{d['tree']}"] = d
    if cells:
        _atomic_json(
            RESULTS_DIR / ("mlp_pilot.json" if pilot else "mlp.json"),
            {
                "recipe": _mlp_recipe(width, max_epochs, dev),
                "n_cells": len(cells),
                "cells": cells,
                **_meta(),
            },
        )


# ── figures ──────────────────────────────────────────────────────────────────

BEH_LABEL = {"imp": "impoliteness", "syc": "sycophancy", "cas": "writing-style", "mk": "marker"}
# One colour = one meaning across every figure in this round.
PRED_COLOR = {"ridge": "#0072B2", "identity_bias": "#999999", "krr": "#D55E00", "mlp": "#009E73"}
TREE_HATCH = {"op": "", "tf": "//"}


def _load_cells() -> dict:
    out = {}
    for p in sorted((RESULTS_DIR / "cells").glob("*.json")):
        d = json.loads(p.read_text())
        out[f"{d['arm_id']}|{d['tree']}"] = d
    # ONLY the full-fidelity mlp.json feeds figures/summary. mlp_pilot.json is a
    # deliberately-truncated TIMING measurement (few epochs) — an undertrained
    # R² must never be presented as this round's MLP read.
    mlp_path = RESULTS_DIR / "mlp.json"
    if mlp_path.exists():
        mlp = json.loads(mlp_path.read_text())
        for key, r in mlp["cells"].items():
            if key in out:
                out[key]["predictors"]["mlp"] = {
                    **r,
                    "source": mlp_path.name,
                }
    return out


def _short(arm_id: str) -> str:
    parts = arm_id.split("-")
    beh = BEH_LABEL.get(parts[0], parts[0])
    return f"{beh}\n{'-'.join(parts[1:])}"


def phase_figs(picks: dict) -> None:
    _phase("figs")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    cells = _load_cells()
    assert cells, f"no per-cell JSONs under {RESULTS_DIR / 'cells'}"
    arm_order = [p["arm_id"] for p in picks["picks"] if f"{p['arm_id']}|op" in cells]
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # (a) predictor comparison, per cell, grouped by behavior, op vs tf hatched.
    # TWO panels: the fitted predictors share a linear axis (the informative
    # comparison); identity+bias gets its own symlog axis because its R² runs
    # -42 .. -10,300 and would flatten every fitted bar on a shared scale.
    from matplotlib.patches import Patch

    fitted = ["ridge", "krr", "mlp"]
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(12.5, 6.6), height_ratios=(2.4, 1.0), sharex=True)
    xt, xl = [], []
    drawn: set[str] = set()
    pos = 0.0
    for arm in arm_order:
        for tree in TREES:
            key = f"{arm}|{tree}"
            if key not in cells:
                continue
            base = pos
            for i, pk in enumerate(fitted):
                r2 = cells[key]["predictors"].get(pk, {}).get("heldout_r2")
                if r2 is None or not np.isfinite(r2):
                    continue
                ax.bar(
                    base + i * 0.25,
                    r2,
                    width=0.23,
                    color=PRED_COLOR[pk],
                    hatch=TREE_HATCH[tree],
                    edgecolor="white",
                    linewidth=0.6,
                )
                drawn.add(pk)
            ib = cells[key]["predictors"].get("identity_bias", {}).get("heldout_r2")
            if ib is not None and np.isfinite(ib):
                axb.bar(
                    base + 0.25,
                    ib,
                    width=0.5,
                    color=PRED_COLOR["identity_bias"],
                    hatch=TREE_HATCH[tree],
                    edgecolor="white",
                    linewidth=0.6,
                )
            xt.append(base + 0.25)
            # single-line + rotated: the 3-line form collides at 16 cells
            xl.append(f"{_short(arm).replace(chr(10), ' ')} [{tree}]")
            pos += 1.05
        pos += 0.25
    for a in (ax, axb):
        a.axhline(0.0, color="#333333", lw=0.9)
    ax.set_ylabel("held-out $R^2$\n(fitted maps)")
    # symlog(linthresh=0.5) is LINEAR across [-0.5, 0.5] — the whole ridge/KRR range
    # and all but two MLP cells read linearly; only the two catastrophic MLP cells
    # (-2.8, -7.5) compress, so nothing is clipped out of view.
    ax.set_yscale("symlog", linthresh=0.5, linscale=2.5)
    axb.set_ylabel("held-out $R^2$\nidentity+bias (symlog)")
    axb.set_yscale("symlog", linthresh=1.0)
    axb.set_xticks(xt)
    axb.set_xticklabels(xl, fontsize=6.5, rotation=90, ha="center")
    ax.set_title(
        f"Predicting the per-context write $w(x)$ from the base context vector $c_0(x)$ "
        f"— layer {LAYER}. Variance-weighted $R^2$ vs the test mean of $w$, so "
        f"$R^2>0$ is context-dependent predictability beyond the static mean write.",
        loc="left",
        fontsize=8.5,
    )
    handles = [Patch(facecolor=PRED_COLOR[pk], label=pk) for pk in fitted if pk in drawn]
    handles += [
        Patch(facecolor="#BBBBBB", label="on-policy write [op]"),
        Patch(facecolor="#BBBBBB", hatch="//", edgecolor="white", label="matched-text write [tf]"),
    ]
    ax.legend(handles=handles, ncol=5, fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "predictor_comparison", FIGS_DIR)
    plt.close(fig)

    # (b) per-direction R² vs rank, variance-share overlay, op vs tf
    n = len(arm_order)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.0, 3.0 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i, arm in enumerate(arm_order):
        ax = axes[i]
        ax2 = ax.twinx()
        for tree, style in (("op", "-o"), ("tf", "--s")):
            key = f"{arm}|{tree}"
            if key not in cells:
                continue
            pd_ = cells[key]["per_direction"]
            ax.plot(
                pd_["direction_rank"],
                pd_["heldout_r2_per_direction"],
                style,
                color=PRED_COLOR["ridge"],
                ms=3.5,
                lw=1.3,
                label=f"$R^2$ [{tree}]",
            )
            ax2.plot(
                pd_["direction_rank"],
                pd_["train_var_share_per_direction"],
                style,
                color="#CC79A7",
                ms=3.0,
                lw=1.0,
                alpha=0.75,
                label=f"var share [{tree}]",
            )
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_title(_short(arm).replace("\n", " "), fontsize=7.5, loc="left")
        ax.set_ylabel("per-direction $R^2$", fontsize=7)
        ax2.set_ylabel("train var share", fontsize=7, color="#CC79A7")
        ax.tick_params(labelsize=6.5)
        ax2.tick_params(labelsize=6.5, colors="#CC79A7")
        if i == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=5.5, ncol=2)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    for ax in axes[max(0, len(axes) - ncol) :]:
        ax.set_xlabel("write-direction rank (centered SVD)", fontsize=7)
    fig.suptitle(
        f"Which of the ~10 write directions are context-predictable? (ridge, layer {LAYER})",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig_paper(fig, "per_direction_r2", FIGS_DIR)
    plt.close(fig)

    # (c) compact summary: ridge R² per behavior × tree
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    labels, vop, vtf = [], [], []
    for arm in arm_order:
        labels.append(_short(arm).replace("\n", " "))
        vop.append(
            cells.get(f"{arm}|op", {})
            .get("predictors", {})
            .get("ridge", {})
            .get("heldout_r2", np.nan)
        )
        vtf.append(
            cells.get(f"{arm}|tf", {})
            .get("predictors", {})
            .get("ridge", {})
            .get("heldout_r2", np.nan)
        )
    y = np.arange(len(labels))
    ax.barh(y - 0.2, vop, height=0.38, color=PRED_COLOR["ridge"], label="on-policy [op]")
    ax.barh(
        y + 0.2,
        vtf,
        height=0.38,
        color=PRED_COLOR["ridge"],
        alpha=0.55,
        hatch="//",
        edgecolor="white",
        label="matched-text [tf]",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0.0, color="#333333", lw=0.9)
    ax.set_xlabel("ridge held-out $R^2$")
    ax.set_title(f"Write predictability from $c_0$ — ridge, layer {LAYER}", loc="left")
    ax.legend(fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig_paper(fig, "ridge_r2_summary", FIGS_DIR)
    plt.close(fig)
    logger.info("[figs] wrote 3 figures to %s", FIGS_DIR)


# ── summary ──────────────────────────────────────────────────────────────────


def phase_summary(picks: dict) -> None:
    _phase("summary")
    cells = _load_cells()
    split = (
        json.loads((RESULTS_DIR / "split.json").read_text())
        if (RESULTS_DIR / "split.json").exists()
        else {}
    )
    staging = (
        json.loads((RESULTS_DIR / "staging.json").read_text())
        if (RESULTS_DIR / "staging.json").exists()
        else {}
    )
    walls = {k: v.get("walls", {}) for k, v in cells.items()}
    mlp_path = RESULTS_DIR / "mlp.json"
    mlp_json = json.loads(mlp_path.read_text()) if mlp_path.exists() else {}
    n_mlp = len([1 for v in cells.values() if "mlp" in v.get("predictors", {})])
    mlp_leg = {
        "status": "landed" if n_mlp else "pending",
        "n_cells_with_mlp": n_mlp,
        "n_cells_total": len(cells),
        "device": (mlp_json.get("recipe") or {}).get("device"),
        "recipe": mlp_json.get("recipe"),
        "per_cell_wall_s": {
            k: v["predictors"]["mlp"].get("wall_s")
            for k, v in sorted(cells.items())
            if "mlp" in v.get("predictors", {})
        },
        "execution": (
            "USER OVERRIDE 'just run on GPU' — the full 16-cell MLP leg ran on a 1-GPU RunPod "
            "pod (pod-1768-wp), NOT the CPU-descoped 4-cell contingency the earlier pilot "
            "gate had sized. No descope."
        ),
        "cpu_pilot_basis_superseded": (
            "the shared-VM CPU pilot measured 6.29 s/epoch/group (2-point: 3 vs 12 epochs), "
            "projecting 8.4 h for a 16-cell x 300-epoch CPU battery — the reason the GPU "
            "override was taken. Recorded for provenance; not the executed path."
        ),
        "note": (
            "an undertrained MLP R2 is never presented: only the full-fidelity mlp.json feeds "
            "figures/summary (mlp_pilot.json is a timing measurement, excluded by construction). "
            "The CPU pilot still read R2 -5.73 at 12 epochs, so the full-dim MLP needs many "
            "epochs before it is competitive — read epochs_ran per cell before comparing."
        ),
    }
    summary = {
        **_meta(),
        "question": (
            "Is the per-context fine-tuning write w(x) predictable from the BASE "
            "model's context vector c0(x), and is that map linear?"
        ),
        "arm_picks": picks,
        "split": split,
        "staging": {k: v for k, v in staging.items() if k != "files"},
        "cells": cells,
        "mlp_leg": mlp_leg,
        "pilot_walls": walls,
        "notes": {
            "r2_convention": (
                "variance-weighted pooled R² over the 3,584 dims of w against the TEST-set "
                "mean of w: R² > 0 means context-DEPENDENT predictability beyond the static "
                "mean write; a constant mean-write predictor reads exactly 0."
            ),
            "mapping_arms": (
                "CONTEXT-based arm only (c0 = base context vector). The prefix arm is "
                "degenerate on this corpus (2 distinct prefix strings) — round 1's stated "
                "deviation carries over verbatim."
            ),
            "identity_baseline": (
                "identity+learned-bias applies (d_in == d_out == 3584) and is reported for "
                "every cell per the standing mapping-baselines rule, though it is a weak "
                "prior here: w is a DIFFERENCE vector, c0 an activation."
            ),
            "duplicate_sha_caveat": (
                "the 82-duplicate-sha val/test caveat of the pinned #779 split carries over "
                "verbatim from round 1 (1,400 pinned rows -> 1,318 unique; val n test 13)."
            ),
            "per_direction_basis": (
                "top-10 right singular vectors of the CENTERED train write matrix (round-1 "
                "rank_read convention). Left singular vectors are row-space and have no "
                "held-out analogue."
            ),
            "mlp_helper_deviation": (
                "MLP uses issue779_fitter_fair_comparison.batched_mlp_fit (padded-bmm AdamW, "
                "batched ACROSS cells) rather than analysis/vectorized_mlp_skill.py: the "
                "latter is built for the tiny-n many-LOCO-fold PCA-target regime (n~49-480), "
                "while batched_mlp_fit is the on-main batched helper for exactly this "
                "full-dim 3584->3584 n~15k map shape (#779 built it, #1073's analogous "
                "inline round reused it)."
            ),
        },
    }
    _atomic_json(RESULTS_DIR / "summary.json", summary)
    logger.info("[summary] %d cells -> summary.json", len(cells))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phases", default="stage,fits,figs,summary")
    ap.add_argument("--cells-limit", type=int, default=None, help="first N cells (pilot)")
    ap.add_argument("--no-krr", action="store_true")
    ap.add_argument("--mlp-cells", default="", help="comma list of '<arm>|<tree>' for the MLP leg")
    ap.add_argument("--mlp-max-epochs", type=int, default=None)
    ap.add_argument("--mlp-pilot", action="store_true")
    ap.add_argument("--mlp-width", type=int, default=MLP_WIDTH)
    ap.add_argument("--mlp-all-cells", action="store_true", help="MLP over all 16 arm x tree cells")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:0 (default: auto-detect)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        import issue779_fitter_fair_comparison as f779  # noqa: F401
        import issue779_ffc_n1m_fits as n1m  # noqa: F401
        from huggingface_hub import HfApi  # noqa: F401

        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from explore_persona_space.analysis.paper_plots import (  # noqa: F401
            savefig_paper,
            set_paper_style,
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401

        _ = (
            f779.MLPGroup,
            f779.batched_mlp_fit,
            f779.krr_select_predict,
            n1m.fit_ridge_with_weights,
        )
        _ = (hub.stage_hub_file, hub.retry_transient, FIT.load_corpus_cell, FIT._fit_map)
        print("[import-check] OK", flush=True)
        return 0

    global _DEVICE_OVERRIDE
    _DEVICE_OVERRIDE = args.device

    import torch

    torch.set_num_threads(max(1, args.threads))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    picks_path = RESULTS_DIR / "arm_picks.json"
    phases = tuple(p for p in args.phases.split(",") if p)

    if picks_path.exists():
        picks = json.loads(picks_path.read_text())
    else:
        picks = pick_arms()
        _atomic_json(picks_path, {**picks, **_meta()})
    for p in picks["picks"]:
        logger.info(
            "[pick] %-24s %-30s method=%-4s verdict=%s",
            p["criterion"],
            p["arm_id"],
            p["method"],
            p["round1_L19_verdict"],
        )

    for phase in phases:
        if phase == "stage":
            phase_stage(picks)
            if not (RESULTS_DIR / "split.json").exists():
                _atomic_json(
                    RESULTS_DIR / "split.json",
                    {
                        **X.assert_pinned_split(),
                        "n_train": X.N_TRAIN,
                        "sample_seed": X.SAMPLE_SEED,
                        "source": "issue1768_cells.assert_pinned_split() (#779 pinned splits)",
                        **_meta(),
                    },
                )
        elif phase == "fits":
            phase_fits(picks, cells_limit=args.cells_limit, do_krr=not args.no_krr)
        elif phase == "mlp":
            if args.mlp_all_cells:
                keys = [f"{p['arm_id']}|{t}" for p in picks["picks"] for t in TREES]
            else:
                keys = [k for k in args.mlp_cells.split(",") if k]
            assert keys, "--mlp-cells or --mlp-all-cells required for the mlp phase"
            phase_mlp(
                picks,
                cell_keys=keys,
                max_epochs=args.mlp_max_epochs or 300,
                pilot=args.mlp_pilot,
                width=args.mlp_width,
            )
        elif phase == "figs":
            phase_figs(picks)
        elif phase == "summary":
            phase_summary(picks)
        else:
            raise ValueError(phase)
    _phase("done")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit: C-extension finalize-race guard (#1689)
