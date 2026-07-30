#!/usr/bin/env python3
"""#1775 shared machinery: store loaders, arms, folds, fits, batched HSIC/dCor nulls.

Nonlinearity ladder over the #1092 four-arm context->answer stores (plan v5).
Reuses the banked #1092 fit engine verbatim (``issue1092_fit_grid``:
``_fit_cv`` -> ``press_fit_predict``, ``_folds_from_manifest``,
``_basis_targets_with_info``, ``_r2``) plus the standing mapping baselines
(``analysis/mapping_baselines``) and the #763 dependence statistics
(``analysis/issue_763_nonlinear``). New here: explicit (train, test) fold
pairs (needed for the doubly-novel read + per-lambda sensitivity), the
arm-input constructions (bare-query broadcast, LOO query-averaged, stitch),
group-respecting inner splits, paired CLUSTER bootstrap CIs, and the BATCHED
GPU permutation-null machinery for HSIC/dCor (no serial per-draw loop —
centered matrices computed ONCE, each draw an advanced-index gather).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

# Reused verbatim — the banked #1092 read1 engine (artifact-reuse item (i) record
# in plan section 10; main-resident copies are current vs the issue-1092 branch).
from issue1092_fit_grid import (  # noqa: E402
    FOLD_SEED,
    _fit_cv,
    _folds_from_manifest,
    _r2,
)
from issue1092_fit_grid import _basis_targets_with_info as _basis_targets_with_info_1092  # noqa: E402
from issue923_fit_decomposition import RIDGE_LAMBDAS, press_fit_predict  # noqa: E402


def _basis_targets_with_info(Y, basis, **kwargs):
    """Parent #1092 basis constructor + a recorded plan deviation on pca48.

    DECLARED DEVIATION (plan section 4 P1 says "PCs fit on train folds only,
    per fold"): the parent engine fits the 48 PCs ONCE on the FULL fit
    population (issue1092_fit_grid._pca_basis on all rows) — the basis the
    banked Gate C references were computed with, so full-population PCs are
    the only choice consistent with Gate C comparability (parity wins over
    the plan's per-fold wording; see record_plan_deviation below).

    NEW mode ``pca48_foldpc`` (follow-up round `dedup-refit-pcfold-doubly`,
    cell 2 — the deviation's DISCHARGE, so no record_plan_deviation here):
    fits the 48 PCs on TRAIN-FOLD rows only (kwarg ``train_idx``, an int
    index array), via the SAME parent ``_pca_basis``; every row is then
    projected into that fold's basis. Returns (Yp, info) with
    ``info["train_idx_n"]`` recording the basis-fit row count.
    """
    if basis == "pca48":
        record_plan_deviation(
            "pca48 basis PCs fit on the full fit-population "
            "(parent #1092 parity, Gate C comparability)",
            "plan section 4 says train-fold-only; the parent's banked 0.914 read uses "
            "population PCs — parity wins; per-fold-PC sensitivity check listed as follow-up",
        )
    if basis == "pca48_foldpc":
        train_idx = kwargs.pop("train_idx", None)
        if train_idx is None:
            raise ValueError("basis 'pca48_foldpc' requires train_idx (the fold's train rows)")
        train_idx = np.asarray(train_idx, dtype=np.int64)
        from issue1092_fit_grid import _pca_basis

        # ambient shape/target checks ride the parent constructor (identity path)
        _Y_amb, info = _basis_targets_with_info_1092(Y, "ambient", **kwargs)
        mu, v = _pca_basis(np.asarray(Y[train_idx], dtype=np.float64), 48)
        info["basis"] = "pca48_foldpc"
        info["v_basis"] = v
        info["mu_basis"] = mu
        info["train_idx_n"] = int(train_idx.size)
        return (np.asarray(Y, dtype=np.float64) - mu) @ v, info
    return _basis_targets_with_info_1092(Y, basis, **kwargs)


from explore_persona_space.analysis.issue_763_nonlinear import (  # noqa: E402
    _double_center,
    _median_heuristic_sigma,
    _pairwise_euclidean,
    distance_correlation,
    hsic_statistic,
)
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)

from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

__all__ = [
    "FOLD_SEED",
    "RIDGE_LAMBDAS",
    "_basis_targets_with_info",
    "_fit_cv",
    "_folds_from_manifest",
    "_r2",
    "distance_correlation",
    "hsic_statistic",
    "identity_bias_predict",
    "knn_retrieval",
    "press_fit_predict",
]

# ── constants ────────────────────────────────────────────────────────────────────

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_HF_PREFIX = "issue1092_realistic_crossing"
OUT_HF_PREFIX = "issue1775_nonlinearity"
# Follow-up round `dedup-refit-pcfold-doubly` (plan v8): eval-JSON sub-dir shared
# by the three follow-up cells + the fu dispatcher/sentinel.
FU_SUB = "fu_dedup_refit_pcfold_doubly"
VM_STAGE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing"
)

LAYER_PRIMARY = 14
LAYER_BRIDGE = 19
TARGETS = ("t1", "t2", "t3")
HIDDEN_DIM = 3584
N_FOLDS = 6
CELL_PRIMARY = "cell_inst_own"
CELL_COMPANION = "cell_pre_own"
CELL_MODEL_TYPE = {"cell_inst_own": "instruct", "cell_pre_own": "pretrained"}
ARMS = ("prefix_end", "bare_query", "query_averaged", "context_end", "stitch")
# bare-query shards exist ONLY at L14 / instruct (verified against local staging +
# the Hub tree 2026-07-28) — arms depending on them are L14 x cell_inst_own only.
BARE_ARMS = {"bare_query", "stitch"}

# Gate C references: banked BATTERY-EXCLUDED context reads (plan section 7 Gate C + A6).
# Source (verbatim): eval_results/issue_1092/inline_fair_comparison/fair_comparison.json
#   jq '.cells.cell_inst_own.bases.<basis>.single_grain.r2_context_battery_excluded_full'
# (NOT .banked_read1_battery_included — that pair is the 19,708-row battery-INCLUDED read;
#  this task's P1 refit runs on the battery-EXCLUDED 17,308-row population.)
GATE_C = {"ambient": 0.8141832948824597, "pca48": 0.9141533323752868}
GATE_C_TOL = 0.02
# Stitch-ridge secondary reproduction target (battery-EXCLUDED pair, #1092 body; A17).
STITCH_REPRO_PCA48 = 0.849

# #779 MLP recipe verbatim (issue779_fitter_fair_comparison.batched_mlp_fit —
# the recipe whose banked n50k/n1m numbers the plan cites; full-batch AdamW).
MLP_WIDTH = 8192
MLP_LR = 3e-4
MLP_WD = 1e-4
MLP_MAX_EPOCHS = 300
MLP_PATIENCE = 20

# KRR grids (plan section 11: #779 wide-grid gamma set + wide lambda set + 10.0).
KRR_GAMMA_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
KRR_LAMBDAS = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
RFF_DIM = 16384

INNER_VAL_GROUP_FRAC = 0.2  # hold out ~1/5 of train groups for nested tuning
N_BOOT = 2000  # cluster bootstrap draws (issue658_fit_predictors N_BOOTSTRAP precedent)


# ── paths / metadata ─────────────────────────────────────────────────────────────


def resolve_store_dir() -> Path:
    """Store root: env override > VM staging > workload-local staging dir."""
    env = os.environ.get("I1775_STORE_DIR")
    if env:
        return Path(env)
    if VM_STAGE.exists():
        return VM_STAGE
    return PROJECT_ROOT / "data" / "issue_1775" / "store1092"


def out_root() -> Path:
    """Output root: env override (smoke scratch redirect) > repo tree."""
    env = os.environ.get("I1775_OUT_ROOT")
    return Path(env) if env else PROJECT_ROOT


def eval_dir(sub: str) -> Path:
    d = out_root() / "eval_results" / "issue_1775" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def tensors_dir(sub: str) -> Path:
    d = out_root() / "data" / "issue_1775" / "analysis_tensors" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def result_meta(**extra) -> dict:
    """Reproducibility metadata block for every result JSON."""
    return {
        "git_commit": git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "fold_seed": FOLD_SEED,
        "n_folds": N_FOLDS,
        **extra,
    }


def record_plan_deviation(deviation: str, rationale: str) -> None:
    """Append a declared deviation to eval_results/issue_1775/plan_deviations.json.

    Idempotent on the deviation text; issue1775_sentinel.py ships the list in the
    results payload (the schema the sentinel already reads — round-2 Major-4).
    """
    path = eval_dir("") / "plan_deviations.json"
    entries = json.loads(path.read_text()) if path.exists() else []
    if any(e.get("deviation") == deviation for e in entries):
        return
    entries.append({"deviation": deviation, "rationale": rationale})
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def append_unit(path: Path, unit: dict) -> None:
    """Atomic per-unit JSONL append (checkpoint-per-unit contract)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(unit, default=_json_default) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def load_units(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_units_validated(path: Path, incomplete_reason) -> list[dict]:
    """Resume-load with a completeness gate (#1775 P3 crash-fix).

    ``incomplete_reason(row) -> str | None`` names why a row is NOT a full
    result payload (None = complete). Incomplete rows — and unparseable
    lines from a mid-write kill — are IGNORED for the resume done-set AND
    PURGED: the JSONL is atomically rewritten without them, one loud line
    per purge, so a stub can never mark its unit done on a later resume.
    """
    if not path.exists():
        return []
    valid: list[dict] = []
    bad: list[str] = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad.append(f"line {ln}: unparseable JSON (mid-write kill?)")
                continue
            why = incomplete_reason(row)
            if why is None:
                valid.append(row)
            else:
                bad.append(
                    f"line {ln}: {why} "
                    f"[{row.get('arm')}/{row.get('grain')}/{row.get('scheme')}"
                    f"/{row.get('rung')} seed={row.get('seed')}]"
                )
    if bad:
        print(
            f"[resume] PURGING {len(bad)} invalid/partial row(s) from {path} "
            "(they will re-run, never resume-skip):",
            flush=True,
        )
        for b in bad:
            print(f"[resume]   {b}", flush=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for row in valid:
                f.write(json.dumps(row, default=_json_default) + "\n")
        os.replace(tmp, path)
    return valid


def unit_key(d: dict, keys: tuple[str, ...]) -> tuple:
    """Resume key over EVERY output-affecting regime field (never a subset)."""
    return tuple(str(d.get(k)) for k in keys)


# ── store loading ────────────────────────────────────────────────────────────────

_STORE_KINDS = ("prefix_end", "context_end", "t1", "t2", "t3")


def stage_store_if_needed(store: Path, *, cells: list[str], layers: list[int]) -> None:
    """Stage the needed #1092 store files from the Hub when absent locally.

    Explicit per-file staging (never a whole-prefix mirror: the cell prefixes
    carry ~140 entries each incl. dynamics kinds this task never reads).
    Consumed at the exact fetch destination — no layout transformation
    (artifact-reuse (h)(iv): "no staging transformation").
    """
    needed: list[str] = ["corpus/manifest.jsonl"]
    for cell in cells:
        for kind in _STORE_KINDS:
            for layer in layers:
                needed.append(f"analysis_tensors/summaries/{cell}/{kind}_L{layer:02d}.npy")
    if any(c == CELL_PRIMARY for c in cells):
        prefix = f"{STORE_HF_PREFIX}/analysis_tensors/summaries/bare_instruct"
        local_bare = store / "analysis_tensors/summaries/bare_instruct"
        if not list(local_bare.glob("c_q_bare_L14*.npy")):
            entries = hub.retry_transient(
                lambda: list(
                    hub.list_hf_files_under_path(HfApi(), HF_DATA_REPO, prefix, repo_type="dataset")
                ),
                what=f"list {prefix}",
            )
            for e in entries:
                rel = e[len(STORE_HF_PREFIX) + 1 :] if e.startswith(STORE_HF_PREFIX) else e
                if "c_q_bare_L14" in rel or "row_index" in rel:
                    needed.append(rel)
    for rel in needed:
        target = store / rel
        if target.exists():
            continue
        hub.stage_hub_file(HF_DATA_REPO, f"{STORE_HF_PREFIX}/{rel}", target, repo_type="dataset")
        print(f"[stage] fetched {rel}", flush=True)


def prefetch_p3_inputs(smoke: bool = False) -> None:
    """Stage the P1/P2 artifacts a STANDALONE P3 run reads (``--phases=p3``
    on a fresh instance): ``linear_fits.json`` (final-payload context),
    the P2 detection JSON (gate-B input to the nonlinear unit grid), and
    the P1 per-row ridge preds+masks ``assemble_gains`` compares against.

    Local copies win (same-instance resume; the ``--phases=p3`` smoke against
    a prior full ``--smoke`` out-root); a Hub miss of a required input fails
    LOUD — a silent miss would empty ``gains_vs_ridge`` / mis-size the grid.
    ``smoke`` selects the expected enumeration for the coverage assert at the
    end (round-3 Minor: a PARTIAL Hub set must fail loud HERE — assemble_gains
    deliberately skips missing preds, so only the fully-empty case was guarded).
    """
    api = HfApi()
    for sub, name in (("ladder", "linear_fits.json"), ("detection", "hsic_dcor.json")):
        target = eval_dir(sub) / name
        if target.exists():
            print(f"[prefetch-p3] {sub}/{name} present locally — kept", flush=True)
            continue
        hub.stage_hub_file(
            HF_DATA_REPO, f"{OUT_HF_PREFIX}/eval_json/{sub}/{name}", target, repo_type="dataset"
        )
        print(f"[prefetch-p3] fetched eval_json/{sub}/{name}", flush=True)
    dest = tensors_dir("heldout_preds")
    have = {p.name for p in dest.glob("*_ridge*.npy")}
    prefix = f"{OUT_HF_PREFIX}/analysis_tensors/heldout_preds"
    entries = hub.retry_transient(
        lambda: list(hub.list_hf_files_under_path(api, HF_DATA_REPO, prefix, repo_type="dataset")),
        what=f"list {prefix}",
    )
    ridge = [e for e in entries if Path(e).name.endswith(".npy") and "_ridge" in Path(e).name]
    if not ridge and not have:
        raise RuntimeError(
            f"p3 prefetch: no per-row ridge preds locally at {dest} nor on the Hub "
            f"under {prefix} — run (or upload) p1 first"
        )
    fetched = 0
    for e in ridge:
        if Path(e).name in have:
            continue
        hub.stage_hub_file(HF_DATA_REPO, e, dest / Path(e).name, repo_type="dataset")
        fetched += 1
    print(
        f"[prefetch-p3] ridge preds ready: {len(have)} local + {fetched} fetched "
        f"(Hub lists {len(ridge)})",
        flush=True,
    )
    assert_p3_ridge_pred_coverage(smoke)


def assert_p3_ridge_pred_coverage(smoke: bool) -> None:
    """Round-3 Minor set-diff assert: every expected per-row ridge pred+mask pair
    (per planned perrow arm|grain|scheme|basis) must exist locally after prefetch;
    a PARTIAL set raises naming the missing keys (assemble_gains would otherwise
    silently skip them and thin gains_vs_ridge)."""
    from issue1775_ladder import expected_ridge_pred_files  # lazy: ladder imports common

    expected = expected_ridge_pred_files(smoke)
    missing = [k for k, (p, m) in expected.items() if not (p.exists() and m.exists())]
    if missing:
        raise RuntimeError(
            f"p3 prefetch: ridge-pred set INCOMPLETE — {len(missing)}/{len(expected)} expected "
            "(arm|grain|scheme|basis) pred+mask pair(s) absent after prefetch: "
            + ", ".join(sorted(missing))
        )
    print(f"[prefetch-p3] ridge-pred coverage OK ({len(expected)} pred+mask pairs)", flush=True)


def load_manifest_rows(store: Path) -> list[dict]:
    path = store / "corpus" / "manifest.jsonl"
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def battery_excluded_indices(rows: list[dict], n: int) -> np.ndarray:
    """The 17,308-row fit population (stratum != trait_stratum AND not is_eval_only) —
    the banked battery-excluded predicate from issue1092_inline_fair_comparison."""
    return np.asarray(
        [
            i
            for i in range(n)
            if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
        ],
        dtype=np.int64,
    )


def load_summary(store: Path, cell: str, kind: str, layer: int) -> np.ndarray:
    """Single-file or sharded summary load (float16 on disk); returns the mmap/array."""
    d = store / "analysis_tensors" / "summaries" / cell
    p = d / f"{kind}_L{layer:02d}.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    shards = sorted(d.glob(f"{kind}_L{layer:02d}_shard*.npy"))
    if not shards:
        raise FileNotFoundError(f"{d}/{kind}_L{layer:02d}[.npy|_shard*.npy]")
    return np.concatenate([np.load(s) for s in shards], axis=0)


def load_bare(store: Path, model_type: str, layer: int) -> tuple[np.ndarray, dict[str, int]]:
    """bare_{model_type}/c_q_bare_L{layer} state + query_id -> row index.

    Mirrors ``issue1092_inline_compose_chain._load_bare`` (incl. its length
    assert) with the store dir + layer as parameters (that module pins its
    STAGE constant to the VM path, unusable pod-side — plan A16 disposition).
    """
    root = store / "analysis_tensors" / "summaries" / f"bare_{model_type}"
    p = root / f"c_q_bare_L{layer:02d}.npy"
    if p.exists():
        arr = np.load(p)
    else:
        shards = sorted(root.glob(f"c_q_bare_L{layer:02d}_shard*.npy"))
        if not shards:
            raise FileNotFoundError(f"{root}/c_q_bare_L{layer:02d}[.npy|_shard*.npy]")
        arr = np.concatenate([np.load(s) for s in shards], axis=0)
    idx_rows: list[dict] = []
    ri = root / "row_index.jsonl"
    if ri.exists():
        with open(ri, encoding="utf-8") as f:
            idx_rows = [json.loads(x) for x in f if x.strip()]
    else:
        for s in sorted(root.glob("row_index_shard*.jsonl")):
            with open(s, encoding="utf-8") as f:
                idx_rows += [json.loads(x) for x in f if x.strip()]
    if len(idx_rows) != arr.shape[0]:
        raise ValueError(f"bare row_index {len(idx_rows)} != rows {arr.shape[0]}")
    q2i = {str(r["query_id"]): i for i, r in enumerate(idx_rows)}
    return arr, q2i


@dataclass
class ArmData:
    """One (cell, layer) slice of the fit population: inputs per arm + targets."""

    cell: str
    layer: int
    rows: list[dict]  # battery-excluded manifest rows (fit population)
    prefix_ids: np.ndarray
    query_ids: np.ndarray
    X: dict[str, np.ndarray]  # arm -> (n, d) float64
    arm_row_mask: dict[str, np.ndarray]  # arm -> bool mask of usable rows
    Y_stacked: np.ndarray  # (n, 3*H) float64


def build_arm_data(
    store: Path,
    cell: str,
    layer: int,
    *,
    arms: tuple[str, ...] = ARMS,
    row_limit: int | None = None,
) -> ArmData:
    """Load inputs for the requested arms + stacked targets on the fit population.

    ``row_limit`` (smoke) truncates the manifest BEFORE battery exclusion so the
    same code path runs at tiny n.
    """
    rows_all = load_manifest_rows(store)
    prefix_all = load_summary(store, cell, "prefix_end", layer)
    context_all = load_summary(store, cell, "context_end", layer)
    t_all = [load_summary(store, cell, t, layer) for t in TARGETS]
    n0 = min(
        prefix_all.shape[0], context_all.shape[0], min(t.shape[0] for t in t_all), len(rows_all)
    )
    if row_limit is not None:
        n0 = min(n0, int(row_limit))
    be_idx = battery_excluded_indices(rows_all, n0)
    rows = [rows_all[int(i)] for i in be_idx]
    prefix_ids = np.asarray([r.get("prefix_id", "") for r in rows])
    query_ids = np.asarray([str(r.get("query_id", "")) for r in rows])
    Xp = np.asarray(prefix_all[be_idx], dtype=np.float64)
    Xc = np.asarray(context_all[be_idx], dtype=np.float64)
    Y = np.concatenate([np.asarray(t[be_idx], dtype=np.float64) for t in t_all], axis=1)
    n = len(rows)
    X: dict[str, np.ndarray] = {}
    mask: dict[str, np.ndarray] = {}
    ones = np.ones(n, dtype=bool)
    if "prefix_end" in arms:
        X["prefix_end"], mask["prefix_end"] = Xp, ones
    if "context_end" in arms:
        X["context_end"], mask["context_end"] = Xc, ones
    if "query_averaged" in arms:
        X["query_averaged"], mask["query_averaged"] = _loo_prefix_mean(Xc, prefix_ids)
    if "bare_query" in arms or "stitch" in arms:
        if cell != CELL_PRIMARY or layer != LAYER_PRIMARY:
            raise ValueError(
                f"bare-query shards exist only for {CELL_PRIMARY} @ L{LAYER_PRIMARY:02d}; "
                f"requested {cell} @ L{layer} (plan scope note)"
            )
        bare, q2i = load_bare(store, CELL_MODEL_TYPE[cell], layer)
        missing = [q for q in query_ids if q not in q2i]
        assert not missing, f"{len(missing)} query_ids missing from bare row_index"
        Xq = np.asarray(
            bare[np.asarray([q2i[q] for q in query_ids], dtype=np.int64)], dtype=np.float64
        )
        if "bare_query" in arms:
            X["bare_query"], mask["bare_query"] = Xq, ones
        if "stitch" in arms:
            X["stitch"], mask["stitch"] = np.concatenate([Xp, Xq], axis=1), ones
    return ArmData(cell, layer, rows, prefix_ids, query_ids, X, mask, Y)


def _loo_prefix_mean(Xc: np.ndarray, prefix_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-row-out within-prefix mean of context_end (the query-averaged arm).

    Rows in singleton prefixes (n_P == 1) have no LOO mean — masked out of this
    arm (reported; the per-arm usable-row mask travels with the arm).
    """
    n = Xc.shape[0]
    out = np.zeros_like(Xc)
    mask = np.ones(n, dtype=bool)
    uniq, inv = np.unique(prefix_ids, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    sums = np.zeros((len(uniq), Xc.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, Xc)
    single = counts[inv] < 2
    mask[single] = False
    denom = np.maximum(counts[inv] - 1, 1).astype(np.float64)[:, None]
    out = (sums[inv] - Xc) / denom
    out[single] = 0.0
    return out, mask


# ── folds ────────────────────────────────────────────────────────────────────────


def scheme_group_key(scheme: str) -> str:
    return {"prefix": "prefix_id", "query": "query_id"}[scheme]


def fold_pairs(
    rows: list[dict], n: int, scheme: str, *, n_folds: int = N_FOLDS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, test_idx) pairs per scheme.

    prefix/query: the #1092 group folds (``_folds_from_manifest``, FOLD_SEED)
    with train = complement — numerically identical to ``_fit_cv``'s masking.
    doubly: test = prefix-fold_i INTERSECT query-fold_i; train = rows whose
    prefix-group AND query-group are both outside fold i (the doubly-novel
    robustness read; partial row coverage by construction, reported).
    """
    if scheme in ("prefix", "query"):
        folds = _folds_from_manifest(rows, n, group_key=scheme_group_key(scheme), n_folds=n_folds)
        pairs = []
        for te in folds:
            m = np.ones(n, dtype=bool)
            m[te] = False
            pairs.append((np.nonzero(m)[0], te))
        return pairs
    if scheme == "doubly":
        fp = _folds_from_manifest(rows, n, group_key="prefix_id", n_folds=n_folds)
        fq = _folds_from_manifest(rows, n, group_key="query_id", n_folds=n_folds)
        pairs = []
        for i in range(min(len(fp), len(fq))):
            in_p = np.zeros(n, dtype=bool)
            in_p[fp[i]] = True
            in_q = np.zeros(n, dtype=bool)
            in_q[fq[i]] = True
            te = np.nonzero(in_p & in_q)[0]
            tr = np.nonzero(~in_p & ~in_q)[0]
            if te.size and tr.size:
                pairs.append((tr, te))
        return pairs
    raise ValueError(f"unknown fold scheme {scheme!r}")


def restrict_pairs(
    pairs: list[tuple[np.ndarray, np.ndarray]], mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Drop masked-out rows (e.g. singleton-prefix rows of the LOO arm) from pairs."""
    keep = np.nonzero(mask)[0]
    keep_set = set(keep.tolist())
    out = []
    for tr, te in pairs:
        tr2 = np.asarray([i for i in tr if i in keep_set], dtype=np.int64)
        te2 = np.asarray([i for i in te if i in keep_set], dtype=np.int64)
        if tr2.size and te2.size:
            out.append((tr2, te2))
    return out


def inner_val_split(
    train_idx: np.ndarray, groups: np.ndarray, *, seed: int, frac: float = INNER_VAL_GROUP_FRAC
) -> tuple[np.ndarray, np.ndarray]:
    """Group-respecting inner split of a train fold (~1/5 of train GROUPS to val)."""
    g = groups[train_idx]
    uniq = sorted(set(g.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, round(frac * len(uniq)))
    val_groups = set(uniq[:n_val])
    is_val = np.asarray([x in val_groups for x in g])
    return train_idx[~is_val], train_idx[is_val]


# ── PRESS ridge over explicit pairs (per-lambda sensitivity + df) ────────────────


def fit_press_pairs(
    X: np.ndarray,
    Y: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    compute_df: bool = False,
    device: str = "cpu",
) -> dict:
    """PRESS-ridge fit per (train, test) pair — the ``_fit_cv`` computation with
    explicit pairs (smoke-asserted equal to ``_fit_cv`` on complement pairs).

    Adds: pooled per-lambda R2 (lambda-sensitivity at lam* x10 / /10 = +-1 step
    on the decade-spaced RIDGE_LAMBDAS grid) and optional df(lambda) via a
    Gram eigendecomposition of the standardized train design (GPU when
    available; the df read shares X across bases).
    """
    n = X.shape[0]
    covered = np.zeros(n, dtype=bool)
    pred = np.zeros_like(Y, dtype=np.float64)
    per_lambda_pred: list[np.ndarray] = [np.zeros_like(Y, dtype=np.float64) for _ in RIDGE_LAMBDAS]
    lam_indices: list[int] = []
    fold_r2: list[float] = []
    dfs: list[dict] = []
    for tr, te in pairs:
        res = press_fit_predict(
            torch.from_numpy(X[tr]).double(),
            torch.from_numpy(Y[tr]).double(),
            torch.from_numpy(X[te]).double(),
            standardize=True,
        )
        pred[te] = res["pred"].detach().cpu().numpy()
        for li in range(len(RIDGE_LAMBDAS)):
            per_lambda_pred[li][te] = res["per_lambda_pred"][li].detach().cpu().numpy()
        lam_indices.append(int(res["lam_idx"]))
        fold_r2.append(_r2(Y[te], pred[te]))
        covered[te] = True
        if compute_df:
            dfs.append(_df_lambda(X[tr], res["std"], device=device))
    cov = np.nonzero(covered)[0]
    out = {
        "r2": _r2(Y[cov], pred[cov]),
        "r2_folds": fold_r2,
        "lambda_indices": lam_indices,
        "lambda_values": [float(RIDGE_LAMBDAS[i]) for i in lam_indices],
        "n_rows_tested": int(cov.size),
        "per_lambda_r2": {
            str(RIDGE_LAMBDAS[li]): _r2(Y[cov], per_lambda_pred[li][cov])
            for li in range(len(RIDGE_LAMBDAS))
        },
        "lambda_sensitivity": _lambda_sensitivity(Y, per_lambda_pred, lam_indices, pairs),
    }
    if compute_df:
        out["df_lambda"] = dfs
    return out, pred, covered


def _lambda_sensitivity(Y, per_lambda_pred, lam_indices, pairs) -> dict:
    """Pooled R2 with every fold's lambda shifted one grid step (x10 / /10)."""
    out = {}
    n_l = len(RIDGE_LAMBDAS)
    for name, shift in (("lam_x10", +1), ("lam_div10", -1)):
        pred = np.zeros_like(Y, dtype=np.float64)
        covered = np.zeros(Y.shape[0], dtype=bool)
        for (tr, te), li in zip(pairs, lam_indices, strict=True):
            lj = min(max(li + shift, 0), n_l - 1)
            pred[te] = per_lambda_pred[lj][te]
            covered[te] = True
        cov = np.nonzero(covered)[0]
        out[name] = _r2(Y[cov], pred[cov])
    return out


def _df_lambda(Xtr: np.ndarray, std: tuple, *, device: str = "cpu") -> dict:
    """Effective dof sum s2/(s2+lambda) per grid lambda from the train Gram."""
    mu, sd, keep = std
    Xn = ((torch.from_numpy(Xtr).double() - mu) / sd)[:, keep]
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    Xn = Xn.to(dev, dtype=torch.float32 if dev.type == "cuda" else torch.float64)
    G = Xn @ Xn.T
    w = torch.linalg.eigvalsh(G).clamp(min=0.0)
    out = {}
    for lam in RIDGE_LAMBDAS:
        out[str(lam)] = float((w / (w + float(lam))).sum().item())
    return out


# ── per-fit baselines (standing mapping-baselines pair) ──────────────────────────


def per_fit_baselines(
    X: np.ndarray,
    Y: np.ndarray,
    pred: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    identity_applicable: bool,
) -> dict:
    """kNN retrieval (both metrics, per-fold pools) + identity+bias where d_in == d_out."""
    knn = {m: [] for m in ("euclidean", "cosine")}
    idb_r2 = []
    for tr, te in pairs:
        for m in knn:
            knn[m].append(knn_retrieval(pred[te], Y[te], ks=(1, 5, 10), metric=m))
        if identity_applicable:
            idb = identity_bias_predict(X[tr], Y[tr], X[te])
            idb_r2.append(_r2(Y[te], idb))
    out = {
        "knn_retrieval": {
            m: {
                "acc_at_k_mean": {
                    k: float(np.mean([f["acc_at_k"][k] for f in fs])) for k in (1, 5, 10)
                },
                "chance_at_k_mean": {
                    k: float(np.mean([f["chance_at_k"][k] for f in fs])) for k in (1, 5, 10)
                },
                "median_rank_mean": float(np.mean([f["median_rank"] for f in fs])),
                "n_pool_per_fold": [f["n_pool"] for f in fs],
            }
            for m, fs in knn.items()
        }
    }
    if identity_applicable:
        out["identity_bias_r2_folds"] = [float(v) for v in idb_r2]
        out["identity_bias_r2_mean"] = float(np.mean(idb_r2))
    else:
        out["identity_bias"] = "inapplicable — d_in != d_out (stated, per the standing rule)"
    return out


def mean_cosine(pred: np.ndarray, true: np.ndarray, covered: np.ndarray) -> float:
    p, t = pred[covered], true[covered]
    num = (p * t).sum(1)
    den = (np.linalg.norm(p, axis=1) + 1e-12) * (np.linalg.norm(t, axis=1) + 1e-12)
    return float(np.mean(num / den))


# ── paired cluster bootstrap on R2 differences ───────────────────────────────────


def cluster_bootstrap_delta_r2(
    Y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    covered: np.ndarray,
    groups: np.ndarray,
    *,
    n_draws: int = N_BOOT,
    seed: int = 0,
) -> dict:
    """Paired CLUSTER bootstrap CI on R2(a) - R2(b) over the SAME held-out rows.

    Resampling unit = the fold scheme's grouping unit (plan section 3): per-group
    SSE sums are precomputed once; every draw is one gather + sum (batched, no
    per-draw pool re-reduction). Row-level bootstrap reported as a labeled
    companion.
    """
    idx = np.nonzero(covered)[0]
    y = Y[idx]
    mu = y.mean(axis=0, keepdims=True)
    se_a = ((y - pred_a[idx]) ** 2).sum(axis=1)
    se_b = ((y - pred_b[idx]) ** 2).sum(axis=1)
    st = ((y - mu) ** 2).sum(axis=1)
    g = groups[idx]
    uniq, inv = np.unique(g, return_inverse=True)
    G = len(uniq)
    gs_a = np.bincount(inv, weights=se_a, minlength=G)
    gs_b = np.bincount(inv, weights=se_b, minlength=G)
    gs_t = np.bincount(inv, weights=st, minlength=G)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, G, size=(n_draws, G))
    A = gs_a[draws].sum(axis=1)
    B = gs_b[draws].sum(axis=1)
    T = gs_t[draws].sum(axis=1)
    delta = (B - A) / np.maximum(T, 1e-300)  # R2_a - R2_b = (SSE_b - SSE_a)/SS_tot
    point = float((gs_b.sum() - gs_a.sum()) / max(gs_t.sum(), 1e-300))
    # row-level companion (labeled)
    n = len(idx)
    rdraws = rng.integers(0, n, size=(n_draws, n))
    rA = se_a[rdraws].sum(axis=1)
    rB = se_b[rdraws].sum(axis=1)
    rT = st[rdraws].sum(axis=1)
    rdelta = (rB - rA) / np.maximum(rT, 1e-300)
    return {
        "delta_r2": point,
        "ci95_cluster": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
        "n_groups": int(G),
        "n_rows": int(n),
        "n_draws": int(n_draws),
        "ci95_row_level_companion": [
            float(np.quantile(rdelta, 0.025)),
            float(np.quantile(rdelta, 0.975)),
        ],
    }


# ── robust eigh (cuSOLVER non-convergence CPU fallback; gotchas #1335) ───────────


def eigh_robust(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return torch.linalg.eigh(G)
    except torch.linalg.LinAlgError:
        print(f"[eigh_robust] cuda eigh failed to converge (n={G.shape[-1]}); CPU fallback")
        w, V = torch.linalg.eigh(G.cpu())
        return w.to(G.device), V.to(G.device)


# ── batched permutation-null machinery for HSIC / dCor (P2 + P4 re-test) ────────


@dataclass
class DependenceMatrices:
    """Centered kernel/distance matrices for one (input, residual) pair.

    HSIC_pi = mean(Kc * Lc[pi][:, pi]) and dCov2_pi = mean(A * B[pi][:, pi]):
    H P L P^T H = P (H L H) P^T for a permutation P (P1 = 1), and double-
    centering commutes with row+col permutation the same way — so every draw
    is a gather + product-sum over matrices centered ONCE.
    """

    Kc: torch.Tensor  # centered input RBF kernel (n, n)
    Lc: torch.Tensor  # centered residual RBF kernel (n, n)
    A: torch.Tensor  # double-centered input distance (n, n)
    B: torch.Tensor  # double-centered residual distance (n, n)
    dvar_x: float
    dvar_y: float


def build_dependence_matrices(
    X: np.ndarray, R: np.ndarray, *, device: str = "cpu"
) -> DependenceMatrices:
    Dx = _pairwise_euclidean(X)
    Dy = _pairwise_euclidean(R)
    sx = _median_heuristic_sigma(Dx)
    sy = _median_heuristic_sigma(Dy)
    K = np.exp(-(Dx**2) / (2.0 * sx**2))
    L = np.exp(-(Dy**2) / (2.0 * sy**2))
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    A = _double_center(Dx)
    B = _double_center(Dy)
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    t = lambda M: torch.from_numpy(np.ascontiguousarray(M)).to(dev, dtype=torch.float32)  # noqa: E731
    return DependenceMatrices(
        Kc=t(Kc),
        Lc=t(Lc),
        A=t(A),
        B=t(B),
        dvar_x=float((A * A).mean()),
        dvar_y=float((B * B).mean()),
    )


def observed_stats(mats: DependenceMatrices) -> dict:
    """Observed HSIC + dCor from the cached matrices (asserted against the #763
    reference implementations at build time by the callers' instrument tie)."""
    n = mats.Kc.shape[0]
    hsic = float((mats.Kc * mats.Lc).mean().item())  # == tr(K H L H)/n^2
    dcov2 = float((mats.A * mats.B).double().mean().item())
    denom = np.sqrt(mats.dvar_x * mats.dvar_y)
    dcor = 0.0 if denom < 1e-12 else float(np.sqrt(max(0.0, dcov2 / denom)))
    _ = n
    return {"hsic": hsic, "dcor": dcor}


def null_stats_batched(
    mats: DependenceMatrices, perms: np.ndarray, *, chunk: int = 50
) -> dict[str, np.ndarray]:
    """HSIC + dCor null draws for (B, n) permutation index rows — batched gathers."""
    dev = mats.Kc.device
    B_draws, n = perms.shape
    hsic = np.empty(B_draws, dtype=np.float64)
    dcor = np.empty(B_draws, dtype=np.float64)
    denom = np.sqrt(mats.dvar_x * mats.dvar_y)
    P = torch.from_numpy(perms).to(dev, dtype=torch.long)
    for lo in range(0, B_draws, chunk):
        hi = min(lo + chunk, B_draws)
        p = P[lo:hi]  # (c, n)
        Lg = mats.Lc[p.unsqueeze(2), p.unsqueeze(1)]  # (c, n, n)
        hsic[lo:hi] = (mats.Kc.unsqueeze(0) * Lg).mean(dim=(1, 2)).cpu().numpy()
        Bg = mats.B[p.unsqueeze(2), p.unsqueeze(1)]
        dcov2 = (mats.A.unsqueeze(0).double() * Bg.double()).mean(dim=(1, 2)).cpu().numpy()
        if denom < 1e-12:
            dcor[lo:hi] = 0.0
        else:
            dcor[lo:hi] = np.sqrt(np.clip(dcov2 / denom, 0.0, None))
    return {"hsic": hsic, "dcor": dcor}


def p_value(null: np.ndarray, obs: float) -> float:
    return float((1.0 + np.sum(null >= obs)) / (1.0 + null.size))


def crossed_permutations(P: int, Q: int, scheme: str, n_draws: int, *, seed: int = 0) -> np.ndarray:
    """(B, P*Q) row-index permutations for a complete P x Q crossed block.

    Row order is prefix-major (p * Q + q). Schemes: prefix-block (permute whole
    prefix blocks, rows aligned by query), query-block (symmetric), derangement
    (within-prefix permutation with NO fixed points, rejection-resampled)."""
    rng = np.random.default_rng(seed)
    base = np.arange(P * Q).reshape(P, Q)
    out = np.empty((n_draws, P * Q), dtype=np.int64)
    if scheme == "prefix_block":
        for b in range(n_draws):
            out[b] = base[rng.permutation(P)].reshape(-1)
    elif scheme == "query_block":
        for b in range(n_draws):
            out[b] = base[:, rng.permutation(Q)].reshape(-1)
    elif scheme == "within_prefix_derangement":
        perms = _batched_derangements(rng, n_draws * P, Q).reshape(n_draws, P, Q)
        for b in range(n_draws):
            out[b] = np.take_along_axis(base, perms[b], axis=1).reshape(-1)
    else:
        raise ValueError(f"unknown crossed scheme {scheme!r}")
    return out


def _batched_derangements(rng: np.random.Generator, m: int, q: int) -> np.ndarray:
    """(m, q) derangements via vectorized rejection (regenerate rows with fixed points)."""
    assert q >= 2, "derangement needs q >= 2"
    keys = rng.random((m, q))
    perms = np.argsort(keys, axis=1)
    ident = np.arange(q)
    for _ in range(200):
        bad = (perms == ident).any(axis=1)
        if not bad.any():
            return perms
        nb = int(bad.sum())
        perms[bad] = np.argsort(rng.random((nb, q)), axis=1)
    raise RuntimeError("derangement rejection sampling did not converge (q too small?)")


def block_permutations_equal_blocks(
    n_blocks: int, block_size: int, n_draws: int, *, seed: int = 0
) -> np.ndarray:
    """(B, n_blocks*block_size) permutations exchanging equal-size blocks
    (the full-corpus companion; block content order arbitrary-but-fixed)."""
    rng = np.random.default_rng(seed)
    base = np.arange(n_blocks * block_size).reshape(n_blocks, block_size)
    out = np.empty((n_draws, n_blocks * block_size), dtype=np.int64)
    for b in range(n_draws):
        out[b] = base[rng.permutation(n_blocks)].reshape(-1)
    return out


def holm_correction(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down over a named p-value family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[name] = running
    return out


# ── phase upload (one upload_folder commit per phase) ────────────────────────────


def _upload_eligible_relpaths(local: Path) -> list[str]:
    """Relative paths under ``local`` that ``hub._upload``'s folder branch will
    actually upload: rglob MINUS the uploader's own excludes. huggingface_hub's
    ``upload_folder`` ALWAYS appends ``DEFAULT_IGNORE_PATTERNS`` (e.g.
    ``**/.cache/huggingface/**`` — the per-file metadata
    a ``local_dir=``-style hub download leaves behind) and ``hub._upload``
    always adds ``TRAINING_STATE_IGNORE_PATTERNS``. An UNFILTERED rglob as the
    verify set counts never-uploaded files as missing — a deterministic
    post-upload RuntimeError (round-1 fu Critical,
    `fu-arrays-upload-verify-cache-mismatch`)."""
    from huggingface_hub.utils import DEFAULT_IGNORE_PATTERNS, filter_repo_objects

    rels = sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())
    ignore = list(DEFAULT_IGNORE_PATTERNS) + list(hub.TRAINING_STATE_IGNORE_PATTERNS)
    return list(filter_repo_objects(rels, ignore_patterns=ignore))


def _upload_dir_verified(local: Path, path_in_repo: str) -> str:
    """One fail-loud upload_folder commit of `local` + exact-set verify (#997).

    The expected set applies the uploader's own eligibility filter
    (:func:`_upload_eligible_relpaths`), so verify checks exactly what
    ``upload_folder`` was asked to ship — never uploader-excluded metadata."""
    url = hub._upload(
        local,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        raise_on_error=True,
    )
    if not url:
        raise RuntimeError(f"upload returned no path for {path_in_repo} — fail loud")
    expected = [f"{path_in_repo}/{r}" for r in _upload_eligible_relpaths(local)]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        HF_DATA_REPO,
        expected,
        path_in_repo=path_in_repo,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"upload verify: {len(missing)} missing under {path_in_repo}")
    print(f"[upload] {len(expected)} files verified under {path_in_repo}")
    return url


def upload_phase_tensors(sub: str, *, smoke: bool) -> str:
    """Upload data/issue_1775/analysis_tensors/<sub> as ONE folder commit."""
    local = tensors_dir(sub)
    if smoke:
        print(f"[upload] smoke — skipping HF upload of {local} (scratch out-root)")
        return ""
    return _upload_dir_verified(local, f"{OUT_HF_PREFIX}/analysis_tensors/{sub}")


def upload_phase_eval_json(sub: str, *, smoke: bool) -> str:
    """Durable per-phase channel for pod-side eval JSONs (round-2 Major-2, #825 class).

    Uploads eval_results/issue_1775/<sub> to
    <OUT_HF_PREFIX>/eval_json/<sub>/ (KB-MB text — non-LFS path, unconditional
    per the Upload Policy) in its own per-phase folder commit. The GCP lane is
    auto-DELETE, so the finalize tar pull is best-effort — this HF copy is the
    crash-survivable channel P5/P6 reads can fall back to.
    """
    local = eval_dir(sub)
    if smoke:
        print(f"[upload] smoke — skipping HF eval-json upload of {local} (scratch out-root)")
        return ""
    return _upload_dir_verified(local, f"{OUT_HF_PREFIX}/eval_json/{sub}")


def phase_timer() -> float:
    return time.monotonic()
