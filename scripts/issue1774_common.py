"""Shared constants + loaders for #1774 (operator characterization of the four arms).

Reuses the #1092 store/fold/fit machinery by import (artifact-reuse checks (a)-(l)
recorded in the plan §10; parent-lineage duty (k) executed by
``issue1774_stage_audit.py``). All fits are fp64; stores are fp16 on disk.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from issue658_fit_predictors import RIDGE_LAMBDAS  # noqa: E402

# ── pins ─────────────────────────────────────────────────────────────────────
STORE_REV = "e590170619e7691c1a95c7b1bb20bda5fd4065ad"  # #1092 corpus+summaries
RB_REV = "037fcbb210bc52c459959b0746cc268fe08bae96"  # #779 r_B bank
DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_PREFIX = "issue1092_realistic_crossing"
RB_PREFIX = "issue779_monitoring/r_b"
HF_UPLOAD_PREFIX = "issue1774_operator_reads"
INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
INSTRUCT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"  # #1092 pin

CELL = "cell_inst_own"
ROBUST_CELL = "cell_pre_own"
LAYERS = (14, 18, 19)
HEADLINE_LAYER = 14
HIDDEN_DIM = 3584
N_FOLDS = 6
FIT_ARM = "A"  # battery-excluded, trait-stratum-excluded (17,308 rows; plan §4)
EXPECTED_FIT_ROWS = 17308
EXPECTED_MANIFEST_ROWS = 21193
EXPECTED_BARE_ROWS = 1500
TRAITS = ("evil", "sycophancy", "hallucination")
ARMS = ("arm_context", "arm_prefix_end", "arm_bare_query", "arm_query_avg")

# λ grid: parent RIDGE_LAMBDAS extended one decade each side (plan §11).
RIDGE_LAMBDAS_EXT = [1e-3, *RIDGE_LAMBDAS, 1e4]

SEED_DRAWS = 1774  # draw-context subsample + steering context seed (plan §10)
N_PERM_DRAWS = 200
N_MATCHED_N_DRAWS = 20
K_DRAWS = 5  # P1 multi-draw count
STEER_BASE_DRAWS = 3

# ── roots (env-overridable; smoke redirects via --out-root) ─────────────────
VM_STAGE_DEFAULT = Path(
    "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing"
)


def stage_dir() -> Path:
    """Resolve the staged #1092 store root (env > VM copy > pod-local staging)."""
    env = os.environ.get("I1774_STAGE_DIR")
    if env:
        return Path(env)
    if VM_STAGE_DEFAULT.exists():
        return VM_STAGE_DEFAULT
    return PROJECT_ROOT / "data/issue_1774/store" / STORE_PREFIX


def summaries_dir() -> Path:
    return stage_dir() / "analysis_tensors/summaries"


def manifest_path() -> Path:
    return stage_dir() / "corpus/manifest.jsonl"


def eval_out(out_root: str | None = None) -> Path:
    return Path(out_root) if out_root else PROJECT_ROOT / "eval_results/issue_1774"


def data_out(out_root: str | None = None) -> Path:
    if out_root:
        return Path(out_root) / "data"
    return PROJECT_ROOT / "data/issue_1774"


def jsonl_rows(path: Path) -> list[dict]:
    """Text-mode line iteration (never .splitlines() — U+2028 shred class)."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=True))
    os.replace(tmp, path)


def repro_meta(extra: dict | None = None) -> dict:
    """Reproducibility metadata for every result JSON (CLAUDE.md requirement)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    meta = {
        "git_commit": commit,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "store_rev": STORE_REV,
        "rb_rev": RB_REV,
    }
    if extra:
        meta.update(extra)
    return meta


# ── robust factorizations (gotchas #1335: cuSOLVER non-convergence) ─────────


def eigh_robust(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.eigh with CPU-LAPACK fallback on cuSOLVER non-convergence."""
    try:
        return torch.linalg.eigh(G)
    except torch.linalg.LinAlgError:
        print(f"[eigh_robust] cuda eigh failed; CPU fallback (n={G.shape[0]})", flush=True)
        w, v = torch.linalg.eigh(G.cpu())
        return w.to(G.device), v.to(G.device)


def svd_robust(A: torch.Tensor, full_matrices: bool = False):
    """torch.linalg.svd with CPU-LAPACK fallback on cuSOLVER non-convergence."""
    try:
        return torch.linalg.svd(A, full_matrices=full_matrices)
    except torch.linalg.LinAlgError:
        print(f"[svd_robust] cuda svd failed; CPU fallback (shape={tuple(A.shape)})", flush=True)
        u, s, vh = torch.linalg.svd(A.cpu(), full_matrices=full_matrices)
        return u.to(A.device), s.to(A.device), vh.to(A.device)


def eig_robust(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-symmetric eig; CPU-first (torch cuda eig routes through CPU anyway)."""
    w, v = torch.linalg.eig(A.cpu())
    return w, v


class _CpuEngineShim:
    """Minimal CPU view of a PressRidge engine for the parent `_operator_raw`."""

    def __init__(self, eng) -> None:
        self.S = eng.S.detach().cpu()
        self.Vh = eng.Vh.detach().cpu()


def operator_raw_safe(fit: dict, G: torch.Tensor, lam: float) -> torch.Tensor:
    """Device-safe wrapper around the parent ``_operator_raw`` (plan asm 4).

    ``issue1092_partb_operator._operator_raw`` allocates ``W_raw`` via
    ``torch.zeros(P, d_full)`` with NO device arg (line ~162) — index-assigning
    cuda-resident factors into it raises a device mismatch. Per the plan we
    wrap (never edit) the parent: move the consumed factors (``eng.S``,
    ``eng.Vh``, ``sd``, ``keep``, ``G``) to CPU, call the parent VERBATIM, and
    return W on CPU (fp64). Callers persist / analyze W CPU-side anyway.
    """
    from issue1092_partb_operator import _operator_raw

    pfit = {
        "eng": _CpuEngineShim(fit["eng"]),
        "sd": fit["sd"].detach().cpu(),
        "keep": fit["keep"].detach().cpu(),
        "d_full": fit["d_full"],
    }
    return _operator_raw(pfit, G.detach().cpu(), float(lam))


# ── manifest / fold / arm registries ─────────────────────────────────────────


def load_manifest() -> list[dict]:
    rows = jsonl_rows(manifest_path())
    assert len(rows) == EXPECTED_MANIFEST_ROWS, (
        f"manifest rows {len(rows)} != {EXPECTED_MANIFEST_ROWS}"
    )
    return rows


def fit_indices(rows: list[dict]) -> list[int]:
    """Battery-excluded fit-arm-A rows (17,308 ±0; plan §4 corrected convention)."""
    from issue1092_fit_grid import _fit_arm_indices

    idx = _fit_arm_indices(FIT_ARM, rows)
    return idx


def banked_convention_indices(rows: list[dict]) -> list[int]:
    """The parent's BANKED fit-arm-A convention (battery leak included) — parity fit only.

    Reproduces the banked v6 grid's filter: exclude trait_stratum, exclude only the
    nonexistent 'battery_eval_only' stratum label (a no-op — see
    issue1092_fit_grid._fit_arm_indices docstring / battery_scope_caveat.json).
    """
    return [
        i
        for i, row in enumerate(rows)
        if row.get("stratum") != "trait_stratum" and row.get("stratum") != "battery_eval_only"
    ]


def grouped_folds(rows: list[dict], n: int) -> list[np.ndarray]:
    """Parent-parity grouped folds: group_key='conv_id' (absent → prefix_id fallback)."""
    from issue1092_fit_grid import _folds_from_manifest

    return _folds_from_manifest(rows, n, group_key="conv_id", n_folds=N_FOLDS)


def load_summary_rows(cell: str, kind: str, layer: int) -> np.ndarray:
    from issue1092_fit_grid import _load_summary

    arr, _paths = _load_summary(summaries_dir(), cell, kind, layer)
    return arr


def load_bare(layer: int, model_type: str = "instruct") -> tuple[np.ndarray, dict[str, int]]:
    """bare_{model_type}/c_q_bare_L{layer} rows + query_id→row index (compose-chain join)."""
    root = summaries_dir() / f"bare_{model_type}"
    p = root / f"c_q_bare_L{layer:02d}.npy"
    if p.exists():
        arr = np.load(p)
    else:
        shards = sorted(root.glob(f"c_q_bare_L{layer:02d}_shard*.npy"))
        if not shards:
            raise FileNotFoundError(f"{root}/c_q_bare_L{layer:02d}[.npy|_shard*.npy]")
        arr = np.concatenate([np.load(s) for s in shards], axis=0)
    ri = root / "row_index.jsonl"
    if ri.exists():
        idx_rows = jsonl_rows(ri)
    else:
        idx_rows = []
        for s in sorted(root.glob("row_index_shard*.jsonl")):
            idx_rows += jsonl_rows(s)
    if len(idx_rows) != arr.shape[0]:
        raise ValueError(f"bare row_index {len(idx_rows)} != rows {arr.shape[0]}")
    q2i = {str(r["query_id"]): i for i, r in enumerate(idx_rows)}
    return arr, q2i


def loro_query_avg(
    X_ctx: np.ndarray, prefix_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """LEAVE-ONE-ROW-OUT per-prefix mean of context_end rows (plan §4 arm 4).

    Returns (X_loro (n,d) with rows in singleton prefixes EXCLUDED via mask,
    keep_mask (n,), plain per-prefix means {prefix_id: (d,)}).
    The plain mean includes the test row at ~1/n_p weight, leaking full-context
    signal into the E[a|p] estimand — LORO removes it for per-context reads;
    averaged-grain reads keep the plain mean.
    """
    n, d = X_ctx.shape
    assert prefix_ids.shape[0] == n, (prefix_ids.shape, n)
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for i in range(n):
        p = str(prefix_ids[i])
        if p not in sums:
            sums[p] = np.zeros(d, dtype=np.float64)
            counts[p] = 0
        sums[p] += X_ctx[i]
        counts[p] += 1
    X_loro = np.zeros_like(X_ctx, dtype=np.float64)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        p = str(prefix_ids[i])
        c = counts[p]
        if c >= 2:
            X_loro[i] = (sums[p] - X_ctx[i]) / (c - 1)
            keep[i] = True
    means = {p: sums[p] / counts[p] for p in sums}
    return X_loro, keep, means


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from issue1092_fit_grid import _r2

    return _r2(y_true, y_pred)


def load_rb_bank(layer: int) -> dict[str, np.ndarray]:
    """r_B trait directions at block-output ``layer`` (r_B[layer-1] row; plan asm 7).

    The bank file is (28, 3584): row L-1 is the residual output of block L,
    matching the store's L{layer} summary convention (#1092 ``b0_rB_pool`` usage).
    """
    root = rb_local_dir()
    out: dict[str, np.ndarray] = {}
    for trait in TRAITS:
        p = root / f"{trait}.pt"
        if not p.exists():
            raise FileNotFoundError(f"r_B bank missing: {p} (stage via stage_audit --stage-rb)")
        t = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(t, dict):
            # tolerate {"r_b": tensor} bundle shapes
            t = t.get("r_b", next(iter(t.values())))
        arr = np.asarray(t, dtype=np.float64)
        assert arr.shape == (28, HIDDEN_DIM), (trait, arr.shape)
        out[trait] = arr[layer - 1]
    return out


def rb_local_dir() -> Path:
    env = os.environ.get("I1774_RB_DIR")
    if env:
        return Path(env)
    return PROJECT_ROOT / "data/issue_1774/rb" / RB_PREFIX


def stage_rb_bank() -> None:
    """Stage the pinned r_B bank files from the data repo (scoped, retried)."""
    from explore_persona_space.orchestrate import hub

    dest = rb_local_dir()
    missing = [t for t in TRAITS if not (dest / f"{t}.pt").exists()]
    if not missing:
        return
    dest.mkdir(parents=True, exist_ok=True)
    for trait in missing:
        hub.stage_hub_file(
            DATA_REPO,
            f"{RB_PREFIX}/{trait}.pt",
            dest / f"{trait}.pt",
            repo_type="dataset",
            revision=RB_REV,
        )


# ── registered channel-count rule (statistics-critic Must-Fix; unit-tested) ──


def contiguous_count_from_top(r2_obs: np.ndarray, p95_null: np.ndarray) -> int:
    """Channel count = # components above the per-component null p95, counted
    CONTIGUOUS-FROM-TOP (stop at the first below-p95 component)."""
    r2_obs = np.asarray(r2_obs, dtype=np.float64)
    p95_null = np.asarray(p95_null, dtype=np.float64)
    assert r2_obs.shape == p95_null.shape, (r2_obs.shape, p95_null.shape)
    count = 0
    for obs, thr in zip(r2_obs, p95_null, strict=True):
        if np.isfinite(obs) and obs > thr:
            count += 1
        else:
            break
    return count


def count_null_band(null_mat: np.ndarray, p95_null: np.ndarray) -> dict:
    """Apply the SAME contiguous-from-top rule inside each null draw → count-null band.

    ``null_mat`` (n_draws, K) per-draw per-component R²; ``p95_null`` (K,) the
    per-component p95 across draws. Selection-symmetric per
    .claude/rules/selection-symmetric-nulls.md (per-draw same-selection).
    """
    counts = np.asarray(
        [contiguous_count_from_top(null_mat[b], p95_null) for b in range(null_mat.shape[0])],
        dtype=np.int64,
    )
    return {
        "null_counts": counts.tolist(),
        "count_p95": float(np.percentile(counts, 95)),
        "count_max": int(counts.max()) if counts.size else 0,
    }


def bh_count(r2_obs: np.ndarray, null_mat: np.ndarray, alpha: float = 0.05) -> int:
    """BH-across-components companion count (empirical per-component p-values)."""
    n_draws = null_mat.shape[0]
    pvals = np.asarray(
        [
            (1.0 + float((null_mat[:, j] >= r2_obs[j]).sum())) / (n_draws + 1.0)
            for j in range(len(r2_obs))
        ],
        dtype=np.float64,
    )
    order = np.argsort(pvals)
    m = len(pvals)
    passed = 0
    for rank, j in enumerate(order, start=1):
        if pvals[j] <= alpha * rank / m:
            passed = rank
    return int(passed)


# ── results sentinel (pod-side epm:results payload; poll_pipeline drains) ────

GPU_HOURS_BUDGETED = 10.0  # plan §9 "Estimated GPU-hours (total): 10"


def _headline_eval_numbers(eval_root: Path) -> dict:
    """Best-effort headline numbers pulled from the landed phase JSONs."""
    out: dict = {}
    for arm in ARMS:
        p = eval_root / "fit_battery" / f"{arm}_L{HEADLINE_LAYER}.json"
        if p.exists():
            j = json.loads(p.read_text())
            out[f"r2_per_context_oof_{arm}_L{HEADLINE_LAYER}"] = j.get("r2_per_context_pooled_oof")
        q = eval_root / "channels" / f"{arm}_L{HEADLINE_LAYER}.json"
        if q.exists():
            j = json.loads(q.read_text())
            out[f"channel_count_{arm}_L{HEADLINE_LAYER}"] = j.get("channel_count")
            out[f"rho1_sq_{arm}_L{HEADLINE_LAYER}"] = j.get("rho1_sq_mean")
    par = eval_root / "fit_battery" / "parity_banked_convention_L14.json"
    if par.exists():
        out["parity_banked_r2"] = json.loads(par.read_text()).get("r2_per_context_pooled_oof")
    return out


def results_sentinel_envelope(
    *,
    gpu_hours_used: float,
    plan_deviations: list[str] | None = None,
    out_root: str | None = None,
    smoke: bool = False,
) -> dict:
    """Full poll_pipeline sentinel envelope carrying the 10-key results payload.

    No-training task: reproducibility_card carries NO adapter fields and
    declares wandb N/A explicitly; wandb_url is the literal "n/a" (plan §10).
    """
    eval_root = eval_out(out_root)
    payload = {
        "eval_numbers": _headline_eval_numbers(eval_root),
        "eval_paths": [
            "eval_results/issue_1774/registry/",
            "eval_results/issue_1774/fit_battery/",
            "eval_results/issue_1774/channels/",
            "eval_results/issue_1774/nullspace/",
            "eval_results/issue_1774/endomorphism/",
            "eval_results/issue_1774/steering/",
        ],
        "reproducibility_card": {
            "task": 1774,
            "no_training": True,
            "wandb": "N/A — no model training in this task (declared, not omitted)",
            "model": INSTRUCT_MODEL,
            "model_revision": INSTRUCT_REVISION,
            "store_rev": STORE_REV,
            "rb_rev": RB_REV,
            "judge_model": "claude-sonnet-4-5-20250929",
            "fold_seed": 0,
            "draw_seed": SEED_DRAWS,
            "lambda_grid": RIDGE_LAMBDAS_EXT,
        },
        "wandb_url": "n/a",
        "hf_hub_url": f"https://huggingface.co/datasets/{DATA_REPO}/tree/main/{HF_UPLOAD_PREFIX}",
        "worktree_path": ".claude/worktrees/issue-1774",
        "final_commit_sha": repro_meta()["git_commit"],
        "gpu_hours_used": float(gpu_hours_used),
        "gpu_hours_budgeted": GPU_HOURS_BUDGETED,
        "plan_deviations": plan_deviations or [],
    }
    kind = "epm:smoke-result" if smoke else "epm:results"
    return {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,  # VM-side drain re-derives max+1
        "task_id": 1774,
        "by": "issue1774_dispatch",
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": payload,
    }


def phase_sentinel_envelope(phase: str, note: str) -> dict:
    """Per-phase done sentinel (drains as epm:progress; pod never runs task.py)."""
    return {
        "sentinel_schema_version": 1,
        "kind": "epm:progress",
        "version": 1,
        "task_id": 1774,
        "by": "issue1774_dispatch",
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": f"[phase={phase}] {note}",
    }


def _sentinel_main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="emit issue-1774 pod-side sentinels")
    ap.add_argument("--emit-results-sentinel", metavar="OUT", default=None)
    ap.add_argument("--emit-phase-sentinel", nargs=2, metavar=("PHASE", "OUT"), default=None)
    ap.add_argument("--phase-note", default="done")
    ap.add_argument("--gpu-hours-used", type=float, default=0.0)
    ap.add_argument("--plan-deviation", action="append", default=[])
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.emit_results_sentinel:
        env = results_sentinel_envelope(
            gpu_hours_used=args.gpu_hours_used,
            plan_deviations=args.plan_deviation,
            out_root=args.out_root,
            smoke=args.smoke,
        )
        write_json_atomic(Path(args.emit_results_sentinel), env)
        print(f"[sentinel] wrote {args.emit_results_sentinel}")
        return 0
    if args.emit_phase_sentinel:
        phase, out = args.emit_phase_sentinel
        write_json_atomic(Path(out), phase_sentinel_envelope(phase, args.phase_note))
        print(f"[sentinel] wrote {out}")
        return 0
    raise SystemExit("nothing to emit: pass --emit-results-sentinel or --emit-phase-sentinel")


if __name__ == "__main__":
    sys.exit(_sentinel_main())
