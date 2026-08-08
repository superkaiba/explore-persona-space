#!/usr/bin/env python3
"""Issue #2202 — SAE-free failure characterization of the #1738 context→answer map.

Pod-side driver (RunPod `cpu-bigmem`). Phases (plan v2 §4; each a `--phase`;
`--phase all` runs the full chain and is the launcher entrypoint):

- ``repro-gate``   P0: stage the banked #1738 inputs at the pinned data-repo
                   revision, assemble the L19 capture memmaps, then re-derive
                   acc@{1,5,10}/MRR from pred16+y_holdout via the canonical
                   ``knn_retrieval`` and reconcile against the banked
                   ``mapping_baselines.json`` values. Mismatch = designed HALT
                   (gate report written first, rc 21) — nothing downstream runs.
- ``extract``      P0.5: cx_holdout fp16, whitening stats (train-ANSWER shrunk
                   covariance, λ=0.1, chunked fp64 accumulation +
                   ``null_battery.shrunk_cholesky_from_cov``), K-resample
                   retrieval ceiling (1,988 × K=4), identity+bias predictions +
                   banked reconciliation (same tolerances), ci_fields.json;
                   immediate HF upload of the derived tensors.
- ``geometry``     P1: per-context mid-ranks in FIVE reads (raw-euclidean
                   PRIMARY, raw-cos, centered-cos, whitened/Mahalanobis,
                   whitened-cos companion) with a FULL-POOL equivalence gate vs
                   the banked values; FAIL-1 sets + Jaccard; WORST tails;
                   per-failure confusers + 4-similarity × multi-space geometry
                   + pool-wide ranks; s_conf; hubness (retrieval + collapse
                   N_10); sample-500 retrieval/collapse lists; pool-size
                   robustness (500/2,000); rank-vs-nerr concordance.
- ``reciprocity``  P2: directed confusion graph, observed reciprocity, graded
                   per-edge forward/reverse ranks, degree-preserving null
                   (target-stub permutation) + distance-only null
                   (Gumbel-top-k, τ ∈ {p1,p5,p25} of pairwise answer
                   distances), per-draw vectors persisted.
- ``upload``       bulk HF upload (analysis_tensors / rows_geom / eval_mirror)
                   + git commit/push of eval_results JSONs (#1205 verify) +
                   the results sentinel, then the single terminal
                   ``[phase=done]`` line.

Smoke parity (plan §4): ``--smoke`` runs the SAME phases / staging / gates /
upload code paths (HF sub-prefix ``<prefix>/smoke``, out-eval sub-dir
``smoke/``); the sliced axes are the confuser-detail row set (50 worst FAIL-1
rows), confusers/row (3), null draws (20) and the sample list (50 rows); the
POOL STAYS FULL (rank reads are pool-scale-quadratic — the smoke measures the
true per-block wall) and the P0/P0.5 gates run at full consumed grain.

Refusal-safety: LMSYS/WildChat text NEVER enters this driver — the pod emits
ci + geometry only; text joins happen VM-side (issue2202_labels/dashboards)
from the local #1482 cache. Every committed eval_results JSON is text-free.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps land BEFORE numpy/torch on the shared VM (#847)

import issue779_common as C  # noqa: E402
import issue1738_characterize as CH  # noqa: E402  (_load_kresample_v, _manifest_fields)
import issue1738_multiturn_fits as FT  # noqa: E402  (assemble_streams, load_split)
import issue1738_multiturn_generate_capture as GG  # noqa: E402  (_depth_band, N1M helpers)
import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    _pairwise_dist,
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.analysis.null_battery import (  # noqa: E402
    PRIMARY_LAMBDA,
    shrunk_cholesky_from_cov,
)
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2202_failchar")

# ── constants (plan §11 / task-body locks) ────────────────────────────────────────
ISSUE = 2202
SEED = 2202  # task-body lock: sample-500 + matched-control seed
HF_PIN = "09788eef2f85330c6f9c6b7cd3d28cb47cfb8429"  # data-repo revision (plan §10)
PARENT_PREFIX = "issue1738_multiturn"
HF_PREFIX_2202 = "issue2202_ctxfail"
LAYER = 19
H_DIM = C.EXPECTED_HIDDEN  # 3584
EXPECTED_N = 9_941  # pinned holdout n (gate-asserted)
EXPECTED_CAPTURE_CHUNKS = 225  # plan-time scoped list_repo_tree enumeration (A10)
N_TRAIN_FLOOR = 20_000  # A16 hard floor (realized n_train = 88,378)
WORST_N = 200  # task-body lock: worst tail size (by rank AND by distance)
SAMPLE_N = 500  # task-body lock: Result-2 sample
CONF_DISPLAY = 10  # task-body lock: confusers shown per failure
POOL_SIZES = (500, 2_000)  # task-body lock (alongside the full 9,941)
N_NULL_DRAWS = 1_000  # task-body lock (degree-preserving); distance null matches
TAU_PCTS = (1, 5, 25)  # pre-registered τ sensitivity sweep (headline p5)
GRAPH_EDGE_CAP = 5_000_000  # pre-registered cap trigger (plan §4 P2)
GRAPH_TOPK_CAP = 50  # per-row confuser cap when the trigger fires
ACC_TOL = 2e-4  # |Δacc@k| gate tolerance (plan §7)
MRR_TOL = 1e-4  # |ΔMRR| gate tolerance (plan §7)
KS = (1, 5, 10)
RC_GATE = 21  # designed-halt rc for the P0/P0.5/P1 banked-value gates (never bare 1)
# space name -> distance metric for the RANK read; PRIMARY defines FAIL-1.
SPACES = ("raw_euclidean", "raw_cos", "cent_cos", "whiten", "whiten_cos")
SPACE_METRIC = {
    "raw_euclidean": "euclidean",
    "raw_cos": "cosine",
    "cent_cos": "cosine",
    "whiten": "euclidean",
    "whiten_cos": "cosine",
}
PRIMARY_SPACE = "raw_euclidean"
# smoke slices (plan §4 smoke parity block; pool + gates stay FULL grain)
SMOKE_FAIL_ROWS = 50
SMOKE_CONFUSERS = 3
SMOKE_DRAWS = 20
SMOKE_SAMPLE = 50

BANKED_REL = "eval_results/issue_1738/mapping_baselines.json"
NERR_REL = "eval_results/issue_1738/percontext_summary_L19_ridge.csv"


# ── small shared helpers (also imported by issue2202_{labels,stats_figs,dashboards}) ──


def now_iso() -> str:
    """UTC ISO timestamp for result metadata."""
    return datetime.now(UTC).isoformat()


def meta_block(extra: dict | None = None) -> dict:
    """Reproducibility metadata block (git sha + dirty flag, versions, ts)."""
    out = {
        "generated_utc": now_iso(),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        **as_metadata_dict(git_provenance(PROJECT_ROOT)),
    }
    if extra:
        out.update(extra)
    return out


def atomic_json(path: Path, obj: dict) -> None:
    """Atomic JSON write (tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def out_eval_dir(args) -> Path:
    """Out-eval root; smoke rebinds to a smoke/ sub-dir (outputs only — staged
    parent INPUTS stay at the canonical non-rebinding work root)."""
    base = Path(args.out_eval)
    return base / "smoke" if args.smoke else base


def hf_prefix(args) -> str:
    """HF prefix; smoke rebinds to the smoke/ sub-prefix (same upload code path)."""
    return f"{args.hf_prefix}/smoke" if args.smoke else args.hf_prefix


def repo_banked_path(rel: str) -> Path:
    """Resolve a cross-issue committed artifact; on a partial-clone pod add the
    sparse cone for eval_results/issue_1738 (the #1739 partial-clone trap)."""
    p = PROJECT_ROOT / rel
    if p.exists():
        return p
    logger.info("[banked] %s absent — trying `git sparse-checkout add` (partial clone)", rel)
    cone = str(Path(rel).parent)
    proc = subprocess.run(  # env explicit: subprocess-env passthrough contract
        ["git", "-C", str(PROJECT_ROOT), "sparse-checkout", "add", cone],
        env={**os.environ},
        capture_output=True,
        text=True,
    )
    if p.exists():
        return p
    raise FileNotFoundError(
        f"{rel} not present and sparse-checkout add failed (rc={proc.returncode}, "
        f"stderr={proc.stderr[-300:]!r}). On a pod, provision with "
        f"BOOTSTRAP_EXTRA_CONES='eval_results/issue_1738' or run "
        f"`git sparse-checkout add eval_results/issue_1738` in the clone."
    )


def ranks_summary(ranks: np.ndarray, n_pool: int) -> dict:
    """acc@k / chance / median rank / MRR from per-row mid-ranks — the
    ``knn_retrieval`` / issue1901 ``_ranks_summary`` reduction, re-derived from
    OUR per-row ranks so the full-pool equivalence gate is meaningful."""
    return {
        "acc_at_k": {int(k): float((ranks <= k).mean()) for k in KS},
        "chance_at_k": {int(k): float(k / n_pool) for k in KS},
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
        "n": int(ranks.shape[0]),
        "n_pool": int(n_pool),
    }


def ranks_of_targets(
    pred: np.ndarray,
    pool: np.ndarray,
    true_idx: np.ndarray,
    metric: str,
    chunk: int = 1024,
    phase: str = "ranks",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row mid-rank of ``pool[true_idx[i]]`` under distance(pred[i], pool).

    Mid-rank + tolerance-based tie conventions copied from
    ``mapping_baselines.knn_retrieval`` (1 + #closer + 0.5·#tied-others; tol =
    1e-9·max(|d_true|, 1e-12)); chunked over query rows so no distance matrix
    exceeds ~chunk×n_pool resident (plan §4: batched GEMMs, never per-pair
    loops). Returns (ranks fp64, d_true fp64, n_strictly_closer int64).
    """
    pred = np.asarray(pred, dtype=np.float64)
    pool = np.asarray(pool, dtype=np.float64)
    n = pred.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    d_true = np.empty(n, dtype=np.float64)
    n_closer = np.empty(n, dtype=np.int64)
    t0 = time.time()
    n_chunks = (n + chunk - 1) // chunk
    for ci_, s in enumerate(range(0, n, chunk)):
        e = min(n, s + chunk)
        d = _pairwise_dist(pred[s:e], pool, metric)
        dt = d[np.arange(e - s), true_idx[s:e]]
        tol = 1e-9 * np.maximum(np.abs(dt)[:, None], 1e-12)
        closer = (d < dt[:, None] - tol).sum(axis=1)
        tied = (np.abs(d - dt[:, None]) <= tol).sum(axis=1) - 1
        ranks[s:e] = 1.0 + closer + 0.5 * tied
        d_true[s:e] = dt
        n_closer[s:e] = closer
        print(
            f"[{phase}] unit {ci_ + 1}/{n_chunks} rows={s}:{e} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    return ranks, d_true, n_closer


def ranks_of_cols_in_row(d_row: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Mid-ranks of specific columns within ONE distance row (sort +
    searchsorted; the ``issue1901_metric_battery.rank_matrix_for_cols`` per-row
    formula for per-row-varying column sets)."""
    row = np.sort(d_row)
    v = d_row[cols]
    tol = 1e-9 * np.maximum(np.abs(v), 1e-12)
    lo = np.searchsorted(row, v - tol, side="left")
    hi = np.searchsorted(row, v + tol, side="right")
    return 1.0 + lo + 0.5 * (hi - lo - 1)


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between two (n, d) arrays."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (an * bn).sum(axis=1)


def skewness(x: np.ndarray) -> float:
    """Population skewness (the issue1901 ``_skew`` formula)."""
    x = np.asarray(x, dtype=np.float64)
    m, s = x.mean(), x.std()
    return float(((x - m) ** 3).mean() / (s**3 + 1e-30))


def build_spaces(
    pred: np.ndarray, ans: np.ndarray, ctx: np.ndarray, stats: dict
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Transformed (pred, ans, ctx) per metric space (plan §4 P1).

    - raw: identity.
    - cent: per-family train-mean centering — answers + predictions by μ_A,
      contexts by μ_C (persona-distance bank-centering convention).
    - whiten: z = L⁻¹(x − μ_fam) with L the task-LOCKED shrunk train-ANSWER
      covariance Cholesky for EVERY family, contexts included (plan §11 stated
      convention, A20); euclidean in z-space is the Mahalanobis read.
    The whiten/whiten_cos entries share one transformed triple.
    """
    mu_a = np.asarray(stats["mu_A"], dtype=np.float64)
    mu_c = np.asarray(stats["mu_C"], dtype=np.float64)
    ell = np.asarray(stats["L"], dtype=np.float64)

    def _wh(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return solve_triangular(ell, (np.asarray(x, np.float64) - mu).T, lower=True).T

    raw = (np.asarray(pred, np.float64), np.asarray(ans, np.float64), np.asarray(ctx, np.float64))
    cent = (raw[0] - mu_a, raw[1] - mu_a, raw[2] - mu_c)
    wh = (_wh(pred, mu_a), _wh(ans, mu_a), _wh(ctx, mu_c))
    return {
        "raw_euclidean": raw,
        "raw_cos": raw,
        "cent_cos": cent,
        "whiten": wh,
        "whiten_cos": wh,
    }


# ── sentinel + [phase=...] contract (pod-side-reporting.md; one-way channel) ─────


def write_sentinel(args, slug: str, kind: str, note_obj: dict, blocks: bool = False) -> None:
    """Write a poll_pipeline-conformant sentinel envelope (write-once; never
    re-read — resume state lives under work_root/out_eval, outside the drained
    glob). Skipped loudly when the sentinel dir is not writable (VM smoke)."""
    sdir = Path(args.sentinel_dir)
    try:
        sdir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("[sentinel] dir %s not writable (%s) — sentinel %s SKIPPED", sdir, exc, slug)
        return
    doc = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,  # pod-side writers hardcode 1; the VM drain re-derives (#1095)
        "task_id": ISSUE,
        "ts": now_iso(),
        "by": "issue2202_failchar",
        "blocks_pipeline": blocks,
        "smoke": bool(args.smoke),
        "note": json.dumps(note_obj),
    }
    atomic_json(sdir / f"issue-{ISSUE}-{slug}.json", doc)
    logger.info("[sentinel] wrote %s (kind=%s)", sdir / f"issue-{ISSUE}-{slug}.json", kind)


# ── P0: staging + reproduction gate ───────────────────────────────────────────────


def _staged(args) -> Path:
    return Path(args.work_root) / "staged"


def _derived(args) -> Path:
    """Derived-tensor dir path (NO mkdir — VM-side readers must not create
    /workspace roots; pod-side WRITERS mkdir explicitly)."""
    return Path(args.work_root) / ("derived_smoke" if args.smoke else "derived")


def stage_inputs(args) -> None:
    """Stage the four banked small inputs at the PIN + stream the capture store.

    Small files: ``hub.stage_hub_file`` at revision=HF_PIN (atomic, retried,
    idempotent). Capture chunks: ``FT.assemble_streams`` (per-chunk download +
    append-only fp32 binaries + cursor resume) — its listing/downloads read the
    data repo's default branch, so the realized chunk COUNT is hard-asserted
    against the plan-time pinned enumeration (225) as the drift guard."""
    from huggingface_hub import HfApi

    staged = _staged(args)
    staged.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    hub.stage_hub_file(
        C.HF_DATA_REPO,
        f"{args.parent_prefix}/analysis_tensors/pred16/context_L19_ridge.npz",
        staged / "pred16.npz",
        revision=args.revision,
    )
    hub.stage_hub_file(
        C.HF_DATA_REPO,
        f"{args.parent_prefix}/analysis_tensors/y_holdout/L{LAYER}.npz",
        staged / "y_holdout_L19.npz",
        revision=args.revision,
    )
    for sub, dest in (("sampling_manifest", "manifest"), ("kresample", "kresample")):
        files = hub.list_hf_files_under_path(
            api,
            C.HF_DATA_REPO,
            f"{args.parent_prefix}/{sub}",
            repo_type="dataset",
            revision=args.revision,
        )
        if not files:
            raise RuntimeError(f"no files under {args.parent_prefix}/{sub} at {args.revision}")
        for k, f in enumerate(files):
            hub.stage_hub_file(
                C.HF_DATA_REPO,
                f,
                staged / dest / Path(f).name,
                revision=args.revision,
            )
            print(f"[p0-stage] unit {k + 1}/{len(files)} {sub}/{Path(f).name}", flush=True)


def _assemble(args):
    """Assemble (or resume) the L19 capture memmaps via the parent streamer.

    Hand-built namespace supplies EVERY field ``assemble_streams`` reads
    (mm_dir / local_capture_dir / hf_prefix — the #1728 call-shape bind)."""
    ns = SimpleNamespace(
        mm_dir=Path(args.work_root) / "mm",
        local_capture_dir=args.local_capture_dir or None,
        hf_prefix=args.parent_prefix,
    )
    mm, ci, meta = FT.assemble_streams(ns, layers=[LAYER])
    if not args.local_capture_dir and meta["n_chunks"] != EXPECTED_CAPTURE_CHUNKS:
        raise RuntimeError(
            f"capture chunk count drift: assembled {meta['n_chunks']} chunks vs the "
            f"plan-time pinned enumeration {EXPECTED_CAPTURE_CHUNKS} — the parent prefix "
            f"changed since the {args.revision} pin; investigate before trusting the join."
        )
    return mm, ci, meta


def load_pred_y(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pred16 fp64, y16 fp64, ci int64) with the ci/fingerprint identity asserts."""
    staged = _staged(args)
    pd_ = np.load(staged / "pred16.npz")
    yd = np.load(staged / "y_holdout_L19.npz")
    pred, pci = pd_["pred16"].astype(np.float64), np.asarray(pd_["ci"], dtype=np.int64)
    y16, yci = yd["y16"].astype(np.float64), np.asarray(yd["ci"], dtype=np.int64)
    assert pred.shape == y16.shape, (pred.shape, y16.shape)
    assert (pci == yci).all(), "pred16/y_holdout ci misalign"
    assert np.array_equal(pd_["fingerprint"], yd["fingerprint"]), (
        "pred16/y_holdout assembly fingerprint mismatch — different capture generations"
    )
    # full-grain identity gate in smoke too (plan §4: the P0 gate never slices)
    assert len(pci) == EXPECTED_N, f"holdout n {len(pci)} != {EXPECTED_N}"
    return pred, y16, pci


def _gate_compare(rec: dict, banked: dict) -> tuple[dict, bool]:
    """Compare a knn_retrieval record against the banked cell; (deltas, ok)."""
    deltas: dict = {"acc_at_k": {}, "mrr": None}
    ok = rec["n"] == banked["n"] and rec["n_pool"] == banked["n_pool"]
    for k in KS:
        d = abs(rec["acc_at_k"][int(k)] - banked["acc_at_k"][str(k)])
        deltas["acc_at_k"][str(k)] = d
        ok = ok and d <= ACC_TOL
    dm = abs(rec["mrr"] - banked["mrr"])
    deltas["mrr"] = dm
    ok = ok and dm <= MRR_TOL
    return deltas, ok


def phase_repro_gate(args) -> None:
    """P0 — staging + the reproduction gate (blocks everything downstream)."""
    logger.info("[phase=p0_repro_gate] start (smoke=%s)", args.smoke)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    assert_out_root_headroom(work_root, need_gb=args.headroom_gb, phase="p0-stage")
    if not args.local_inputs:
        stage_inputs(args)
    mm, ci, meta = _assemble(args)
    del mm
    pred, y16, pci = load_pred_y(args)

    # n_train floor (A16) — full pinned split, smoke included.
    split = FT.load_split(_staged(args) / "manifest" / "split_1738.json")
    n_train = len(split["sets"]["train"]["ci"])
    assert n_train >= N_TRAIN_FLOOR, f"n_train {n_train} < floor {N_TRAIN_FLOOR}"

    banked = json.loads(repo_banked_path(args.banked_json).read_text())
    cell = banked["cells"]["context_L19"]["knn"]["ridge"]
    gate: dict = {"verdict": "PASS", "metrics": {}, "n_train": n_train}
    for metric in ("euclidean", "cosine"):
        rec = knn_retrieval(pred, y16, ks=KS, metric=metric)
        deltas, ok = _gate_compare(rec, cell[metric])
        gate["metrics"][metric] = {"recomputed": rec, "banked": cell[metric], "deltas": deltas}
        if not ok:
            gate["verdict"] = "FAIL"
    gate["tolerances"] = {"acc": ACC_TOL, "mrr": MRR_TOL}
    gate["assembled_capture"] = {"n_rows": int(meta["n_rows"]), "n_chunks": int(meta["n_chunks"])}
    gate["meta"] = meta_block({"revision_pin": args.revision})
    out = out_eval_dir(args)
    atomic_json(out / "repro_gate.json", gate)
    write_sentinel(
        args,
        "p0.done",
        "epm:progress",
        {"phase": "p0_repro_gate", "verdict": gate["verdict"], "n": int(len(pci))},
    )
    if gate["verdict"] != "PASS":
        logger.error("[p0] REPRODUCTION GATE FAILED: %s", json.dumps(gate["metrics"])[:500])
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(RC_GATE)  # designed halt (report written first) — failure_class: data
    logger.info("[p0] reproduction gate PASS")


# ── P0.5: extraction of derived tensors ──────────────────────────────────────────


def chunked_mean(mm: np.memmap, rows: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """fp64 mean over memmap rows, chunked (bounded RSS)."""
    s = np.zeros(mm.shape[1], dtype=np.float64)
    for i in range(0, len(rows), chunk):
        s += np.asarray(mm[rows[i : i + chunk]], dtype=np.float64).sum(axis=0)
    return s / len(rows)


def chunked_cov(mm: np.memmap, rows: np.ndarray, chunk: int = 8192, phase: str = "cov") -> tuple:
    """(cov fp64 ddof=1, mean fp64) accumulated chunked over memmap rows —
    the chunked-fp64 twin of ``np.cov(X, rowvar=False)`` (equivalence pinned in
    tests/test_issue2202_failchar.py)."""
    d = mm.shape[1]
    s = np.zeros(d, dtype=np.float64)
    q = np.zeros((d, d), dtype=np.float64)
    n = len(rows)
    t0 = time.time()
    n_chunks = (n + chunk - 1) // chunk
    for k, i in enumerate(range(0, n, chunk)):
        x = np.asarray(mm[rows[i : i + chunk]], dtype=np.float64)
        s += x.sum(axis=0)
        q += x.T @ x
        print(f"[{phase}] unit {k + 1}/{n_chunks} elapsed={time.time() - t0:.1f}s", flush=True)
    mu = s / n
    cov = (q - n * np.outer(mu, mu)) / (n - 1)
    return cov, mu


def kres_classes(s: np.ndarray) -> np.ndarray:
    """Attribution class codes from per-context resample-success shares s_i
    (plan §3 H2 partition; K=4 ⇒ s ∈ {0,.25,.5,.75,1}): MAP_ATTRIBUTABLE at
    s ≥ 0.75, IRREDUCIBLE at s ≤ 0.25, else AMBIGUOUS."""
    out = np.full(len(s), "AMBIGUOUS", dtype=object)
    out[s >= 0.75] = "MAP_ATTRIBUTABLE"
    out[s <= 0.25] = "IRREDUCIBLE"
    return out


def phase_extract(args) -> None:
    """P0.5 — whitening stats, cx_holdout, kresample ceiling, identity+bias,
    ci_fields; immediate HF upload of the derived tensors."""
    logger.info("[phase=p05_extract] start")
    ns = SimpleNamespace(
        mm_dir=Path(args.work_root) / "mm",
        local_capture_dir=args.local_capture_dir or None,
        hf_prefix=args.parent_prefix,
    )
    mm, ci, _meta = FT.assemble_streams(ns, layers=[LAYER])
    split = FT.load_split(_staged(args) / "manifest" / "split_1738.json")
    sets = FT.split_positions(split, ci)
    train_rows, holdout_rows = sets["train"], sets["holdout"]
    assert len(train_rows) >= N_TRAIN_FLOOR, len(train_rows)

    pred, y16, pci = load_pred_y(args)
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    # cx_holdout in pred16/y_holdout ci ORDER (exact-set join, fail-loud)
    ci_list = ci.tolist()
    row_of_ci = {int(c): r for r, c in enumerate(ci_list)}
    missing = [int(c) for c in pci if int(c) not in row_of_ci]
    assert not missing, f"{len(missing)} holdout cis missing from capture memmaps"
    hold_rows_ord = np.asarray([row_of_ci[int(c)] for c in pci], dtype=np.int64)
    cx_hold = np.asarray(mm[("cx", LAYER)][hold_rows_ord], dtype=np.float16)

    derived = _derived(args)
    derived.mkdir(parents=True, exist_ok=True)
    np.savez(derived / "cx_holdout_L19.npz", cx=cx_hold, ci=pci)

    # whitening stats: train-ANSWER shrunk covariance (λ=0.1) + per-family means
    cov, mu_a = chunked_cov(mm[("vx", LAYER)], train_rows, phase="p05-cov")
    ell = shrunk_cholesky_from_cov(cov, PRIMARY_LAMBDA)
    del cov
    mu_c = chunked_mean(mm[("cx", LAYER)], train_rows)
    np.savez(
        derived / "whiten_stats.npz",
        mu_A=mu_a,
        mu_C=mu_c,
        L=ell,
        lam=np.float64(PRIMARY_LAMBDA),
        n_train=np.int64(len(train_rows)),
    )

    # K-resample retrieval ceiling (1,988 × K=4; primary metric)
    kns = SimpleNamespace(
        local_kresample_dir=str(_staged(args) / "kresample"),
        scratch=str(Path(args.work_root) / "scratch"),
        hf_prefix=args.parent_prefix,
    )
    kci, vres = CH._load_kresample_v(kns, [LAYER])  # (n, K, 1, H)
    n_k, k_draws = vres.shape[0], vres.shape[1]
    kmiss = [int(c) for c in kci if int(c) not in pos_of]
    assert not kmiss, f"{len(kmiss)} kresample cis not in holdout pool"
    q = vres[:, :, 0, :].reshape(n_k * k_draws, H_DIM).astype(np.float64)
    true_idx = np.repeat(np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64), k_draws)
    kranks, _, _ = ranks_of_targets(
        q, y16, true_idx, SPACE_METRIC[PRIMARY_SPACE], chunk=args.chunk_rows, phase="p05-kres"
    )
    kranks = kranks.reshape(n_k, k_draws)
    s_i = (kranks == 1.0).mean(axis=1)
    np.savez(
        derived / "kresample_ranks.npz",
        ci=kci,
        ranks=kranks,
        s=s_i,
        classes=np.asarray([str(x) for x in kres_classes(s_i)]),
    )
    logger.info("[p05] kresample ceiling acc1_ceiling=%.4f over n=%d", float(s_i.mean()), n_k)

    # identity+bias predictions + banked reconciliation (same tolerances, P0 tail)
    cx_train = np.asarray(mm[("cx", LAYER)][train_rows], dtype=np.float32)
    vx_train = np.asarray(mm[("vx", LAYER)][train_rows], dtype=np.float32)
    cx_hold64 = cx_hold.astype(np.float64)
    id_pred = identity_bias_predict(cx_train, vx_train, cx_hold64)
    del cx_train, vx_train
    banked = json.loads(repo_banked_path(args.banked_json).read_text())
    id_cell = banked["cells"]["context_L19"]["knn"]["identity_bias"]
    gate: dict = {"verdict": "PASS", "metrics": {}}
    for metric in ("euclidean", "cosine"):
        rec = knn_retrieval(id_pred, y16, ks=KS, metric=metric)
        deltas, ok = _gate_compare(rec, id_cell[metric])
        gate["metrics"][metric] = {"recomputed": rec, "banked": id_cell[metric], "deltas": deltas}
        if not ok:
            gate["verdict"] = "FAIL"
    # report-only companions vs banked identity_bias summary
    mu_hold = y16.mean(axis=0)
    r2 = 1.0 - float(((y16 - id_pred) ** 2).sum() / ((y16 - mu_hold) ** 2).sum())
    gate["holdout_r2_recomputed"] = r2
    gate["holdout_r2_banked"] = banked["cells"]["context_L19"]["identity_bias"]["holdout_r2"]
    np.savez(derived / "identity_bias_pred16.npz", pred16=id_pred.astype(np.float16), ci=pci)

    # ci_fields export (depth / depth_band / corpus) so no VM phase stages the manifest
    fields = CH._manifest_fields(_staged(args) / "manifest")
    ci_fields = {
        str(int(c)): {
            "depth": fields[int(c)]["depth"],
            "depth_band": GG._depth_band(fields[int(c)]["depth"]),
            "corpus": fields[int(c)]["corpus"],
        }
        for c in pci
    }
    atomic_json(derived / "ci_fields.json", {"fields": ci_fields, "meta": meta_block()})

    gate["meta"] = meta_block()
    out = out_eval_dir(args)
    atomic_json(out / "identity_bias_gate.json", gate)

    upload_derived(args)
    write_sentinel(
        args,
        "p05.done",
        "epm:progress",
        {"phase": "p05_extract", "identity_gate": gate["verdict"], "kres_n": int(n_k)},
    )
    if gate["verdict"] != "PASS":
        logger.error("[p05] identity+bias banked reconciliation FAILED")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(RC_GATE)


def upload_derived(args) -> None:
    """Upload the derived tensors to HF analysis_tensors/ immediately (P0.5
    contract: before any long downstream phase). Exact-set verified."""
    if args.no_upload:
        logger.info("[upload] derived tensors upload SKIPPED (--no-upload)")
        return
    derived = _derived(args)
    rel = sorted(p.name for p in derived.iterdir() if p.is_file())
    dest = f"{hf_prefix(args)}/analysis_tensors"
    url = hub._upload_folder_filtered(
        derived,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        allow_patterns=rel,
        expected_repo_paths=[f"{dest}/{r}" for r in rel],
    )
    if not url:
        raise RuntimeError(f"derived-tensor upload to {dest} returned no URL ({rel})")
    logger.info("[upload] derived tensors -> %s (%d files)", dest, len(rel))


# ── P1: confusion geometry ────────────────────────────────────────────────────────


def load_geometry_inputs(args):
    """(pred, y16, cx, pci, stats dict) at fp64 for the geometry/reciprocity phases."""
    pred, y16, pci = load_pred_y(args)
    derived = _derived(args)
    cx = np.load(derived / "cx_holdout_L19.npz")
    assert (np.asarray(cx["ci"], dtype=np.int64) == pci).all(), "cx_holdout ci misalign"
    stats_z = np.load(derived / "whiten_stats.npz")
    stats = {k: stats_z[k] for k in ("mu_A", "mu_C", "L")}
    return pred, y16, cx["cx"].astype(np.float64), pci, stats


def draw_subpool(n: int, size: int, seed: int) -> np.ndarray:
    """Seed-pinned pool subsample of ``size`` rows (sorted)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))


def subpool_ranks_chunk(
    d: np.ndarray, dt: np.ndarray, s: int, sub: np.ndarray, in_sub: np.ndarray
) -> np.ndarray:
    """Mid-ranks at a reduced pool for one query chunk.

    Pool for query i = sub ∪ {i} kept at EXACTLY |sub| entries: when i ∉ sub the
    last sub element is dropped in favor of the true target (deterministic), and
    when i ∈ sub the subsample already carries it. ``in_sub[global row]`` flags
    membership; d is the chunk's full-pool distance block, dt the true dists.
    """
    tol = 1e-9 * np.maximum(np.abs(dt)[:, None], 1e-12)
    dsub_full = d[:, sub]
    below_f = (dsub_full < dt[:, None] - tol).sum(axis=1)
    tied_f = (np.abs(dsub_full - dt[:, None]) <= tol).sum(axis=1) - 1  # excl. self col
    dsub_part = dsub_full[:, :-1]
    below_p = (dsub_part < dt[:, None] - tol).sum(axis=1)
    tied_p = (np.abs(dsub_part - dt[:, None]) <= tol).sum(axis=1)
    rows = np.arange(d.shape[0]) + s
    member = in_sub[rows]
    closer = np.where(member, below_f, below_p)
    tied = np.where(member, tied_f, tied_p)
    return 1.0 + closer + 0.5 * tied


def pair_sims(
    spaces: dict, kind_a: str, idx_a: np.ndarray, kind_b: str, idx_b: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-pair similarity block for one relation (cc/aa/ac/pa): raw cosine,
    centered cosine, whitened cosine + raw and whitened squared-euclid distances.
    kind_* ∈ {pred, ans, ctx} select the family column within each space triple."""
    col = {"pred": 0, "ans": 1, "ctx": 2}
    out: dict[str, np.ndarray] = {}
    for tag, space in (("raw", "raw_euclidean"), ("cent", "cent_cos"), ("whiten", "whiten")):
        a = spaces[space][col[kind_a]][idx_a]
        b = spaces[space][col[kind_b]][idx_b]
        out[f"cos_{tag}"] = row_cosine(a, b)
        if tag in ("raw", "whiten"):
            out[f"d_{tag}"] = ((a - b) ** 2).sum(axis=1)
    return out


def phase_geometry(args) -> None:
    """P1 — ranks per space, FAIL-1, tails, confusers, hubness, sample lists,
    pool robustness, concordance (all chunked GEMMs; full pool always)."""
    logger.info("[phase=p1_geometry] start (smoke=%s)", args.smoke)
    pred, y16, cx, pci, stats = load_geometry_inputs(args)
    n = len(pci)
    spaces = build_spaces(pred, y16, cx, stats)
    true_idx = np.arange(n)
    out = out_eval_dir(args)
    chunk = args.chunk_rows

    ranks: dict[str, np.ndarray] = {}
    d_true: dict[str, np.ndarray] = {}
    n_closer: dict[str, np.ndarray] = {}
    s_conf: dict[str, np.ndarray] = {}
    for sp in SPACES:
        p_t, a_t, _c_t = spaces[sp]
        metric = SPACE_METRIC[sp]
        r, dt, nc = ranks_of_targets(p_t, a_t, true_idx, metric, chunk=chunk, phase=f"p1-{sp}")
        ranks[sp], d_true[sp], n_closer[sp] = r, dt, nc
        # s_conf: cosine(a_i, a_j1) with j1 = top NON-TRUE pool answer in this space
        sc = np.empty(n)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            d = _pairwise_dist(
                np.asarray(p_t[s:e], np.float64), np.asarray(a_t, np.float64), metric
            )
            d[np.arange(e - s), true_idx[s:e]] = np.inf
            j1 = d.argmin(axis=1)
            sc[s:e] = row_cosine(a_t[s + np.arange(e - s)], a_t[j1])
        s_conf[sp] = sc

    # full-pool equivalence gate (P1): NEW per-row ranks reproduce the banked values
    banked = json.loads(repo_banked_path(args.banked_json).read_text())
    cell = banked["cells"]["context_L19"]["knn"]["ridge"]
    equiv: dict = {"verdict": "PASS", "metrics": {}}
    for sp, metric in (("raw_euclidean", "euclidean"), ("raw_cos", "cosine")):
        rec = ranks_summary(ranks[sp], n)
        deltas, ok = _gate_compare(rec, cell[metric])
        equiv["metrics"][metric] = {"recomputed": rec, "banked": cell[metric], "deltas": deltas}
        if not ok:
            equiv["verdict"] = "FAIL"

    fail = {sp: ranks[sp] > 1.0 for sp in SPACES}
    n_fail = int(fail[PRIMARY_SPACE].sum())
    jac = {
        a: {b: float((fail[a] & fail[b]).sum() / max(1, (fail[a] | fail[b]).sum())) for b in SPACES}
        for a in SPACES
    }
    logger.info("[p1] equivalence gate %s; FAIL-1 (primary) = %d", equiv["verdict"], n_fail)

    # WORST tails (primary space): top-200 by rank AND top-200 by raw distance
    rk = ranks[PRIMARY_SPACE]
    dt0 = d_true[PRIMARY_SPACE]
    order_rank = np.lexsort((np.arange(n), -rk))  # rank desc, index tie-break
    order_dist = np.lexsort((np.arange(n), -dt0))
    worst_rank = set(order_rank[: min(WORST_N, n)].tolist())
    worst_dist = set(order_dist[: min(WORST_N, n)].tolist())

    # sample-500 rows (seed-pinned from ALL holdout rows)
    sample_n = SMOKE_SAMPLE if args.smoke else SAMPLE_N
    sample_rows = np.sort(np.random.default_rng(SEED).choice(n, size=sample_n, replace=False))
    sample_set = set(sample_rows.tolist())

    # primary-space chunk pass #2: retrieval hubness + sample list A + pool robustness
    p0_t, a0_t, _ = spaces[PRIMARY_SPACE]
    metric0 = SPACE_METRIC[PRIMARY_SPACE]
    n10_retrieval = np.zeros(n, dtype=np.int64)
    subpools = {p: draw_subpool(n, p, SEED + p) for p in POOL_SIZES if p < n}
    in_sub = {p: np.zeros(n, dtype=bool) for p in subpools}
    for p, sub in subpools.items():
        in_sub[p][sub] = True
    sub_ranks = {p: np.empty(n) for p in subpools}
    listA: dict[int, list] = {}
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = _pairwise_dist(p0_t[s:e], a0_t, metric0)
        dt = d[np.arange(e - s), true_idx[s:e]]
        kk = min(10, n - 1)
        top10 = np.argpartition(d, kk, axis=1)[:, :kk]
        np.add.at(n10_retrieval, top10.ravel(), 1)
        for p, sub in subpools.items():
            sub_ranks[p][s:e] = subpool_ranks_chunk(d, dt, s, sub, in_sub[p])
        for r_local in np.nonzero(np.isin(np.arange(s, e), sample_rows))[0]:
            row = d[r_local]
            cand = np.argpartition(row, kk)[:kk]
            cand = cand[np.argsort(row[cand], kind="stable")]
            gi = s + r_local
            listA[gi] = [
                {
                    "ci": int(pci[j]),
                    "d": float(row[j]),
                    "is_true": bool(j == gi),
                    "cos_raw": float(row_cosine(y16[gi : gi + 1], y16[j : j + 1])[0]),
                }
                for j in cand
            ]
        print(f"[p1-hub] rows {s}:{e} done", flush=True)

    # collapse pass: nearest OTHER predictions (self excluded) + sample list B
    n10_collapse = np.zeros(n, dtype=np.int64)
    listB: dict[int, list] = {}
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = _pairwise_dist(p0_t[s:e], p0_t, metric0)
        d[np.arange(e - s), np.arange(s, e)] = np.inf
        kk = min(10, n - 1)
        top10 = np.argpartition(d, kk, axis=1)[:, :kk]
        np.add.at(n10_collapse, top10.ravel(), 1)
        for r_local in np.nonzero(np.isin(np.arange(s, e), sample_rows))[0]:
            row = d[r_local]
            cand = np.argpartition(row, kk)[:kk]
            cand = cand[np.argsort(row[cand], kind="stable")]
            gi = s + r_local
            listB[gi] = [
                {
                    "ci": int(pci[j]),
                    "d": float(row[j]),
                    "cos_raw": float(row_cosine(p0_t[gi : gi + 1], p0_t[j : j + 1])[0]),
                }
                for j in cand
            ]
        print(f"[p1-collapse] rows {s}:{e} done", flush=True)

    # confuser extraction (FAIL-1 rows; smoke slices the DETAIL row set only)
    fail_rows = np.nonzero(fail[PRIMARY_SPACE])[0]
    detail_rows = fail_rows
    if args.smoke:
        worst_first = fail_rows[np.argsort(-rk[fail_rows], kind="stable")]
        detail_rows = np.sort(worst_first[: min(SMOKE_FAIL_ROWS, len(worst_first))])
    conf_k = SMOKE_CONFUSERS if args.smoke else CONF_DISPLAY
    conf_rows: list[dict] = []
    pairs_i: list[int] = []
    pairs_j: list[int] = []
    for s in range(0, len(detail_rows), chunk):
        rows_blk = detail_rows[s : s + chunk]
        d = _pairwise_dist(p0_t[rows_blk], a0_t, metric0)
        for r_local, gi in enumerate(rows_blk):
            row = d[r_local]
            dt_i = row[gi]
            tol = 1e-9 * max(abs(dt_i), 1e-12)
            outr = np.nonzero(row < dt_i - tol)[0]
            # stable machine-independent ordering: (distance, pool index)
            outr = outr[np.lexsort((outr, row[outr]))]
            top = outr[:conf_k]
            conf_rows.append(
                {
                    "row": int(gi),
                    "ci": int(pci[gi]),
                    "rank": float(rk[gi]),
                    "n_outrank": int(len(outr)),
                    "confusers": [
                        {
                            "row": int(j),
                            "ci": int(pci[j]),
                            "d_pred": float(row[j]),
                            "rank_fwd": f + 1,
                        }
                        for f, j in enumerate(top)
                    ],
                }
            )
            for j in top:
                pairs_i.append(int(gi))
                pairs_j.append(int(j))
        print(
            f"[p1-confusers] rows {s}:{min(s + chunk, len(detail_rows))}/{len(detail_rows)}",
            flush=True,
        )

    pi = np.asarray(pairs_i, dtype=np.int64)
    pj = np.asarray(pairs_j, dtype=np.int64)
    sims = {
        "cc": pair_sims(spaces, "ctx", pi, "ctx", pj),
        "aa": pair_sims(spaces, "ans", pi, "ans", pj),
        "ac": pair_sims(spaces, "ans", pi, "ctx", pj),
        "pa": pair_sims(spaces, "pred", pi, "ans", pj),
    }
    # pool-wide ranks: v_Cj among all contexts (sim to v_Ci); a_j among all answers (sim to a_i)
    ctx0 = spaces[PRIMARY_SPACE][2]
    rank_ctx = np.empty(len(pi))
    rank_ans = np.empty(len(pi))
    uniq_rows = np.unique(pi)
    col_of: dict[int, list[int]] = {}
    for k, gi in enumerate(pi):
        col_of.setdefault(int(gi), []).append(k)
    for s in range(0, len(uniq_rows), chunk):
        blk = uniq_rows[s : s + chunk]
        dctx = _pairwise_dist(ctx0[blk], ctx0, metric0)
        dans = _pairwise_dist(a0_t[blk], a0_t, metric0)
        for r_local, gi in enumerate(blk):
            ks_ = col_of[int(gi)]
            cols = pj[ks_]
            rank_ctx[ks_] = ranks_of_cols_in_row(dctx[r_local], cols)
            rank_ans[ks_] = ranks_of_cols_in_row(dans[r_local], cols)
        print(
            f"[p1-confranks] rows {s}:{min(s + chunk, len(uniq_rows))}/{len(uniq_rows)}", flush=True
        )

    # fold sims + ranks back into conf_rows
    k = 0
    for rec in conf_rows:
        for cf in rec["confusers"]:
            cf["sims"] = {
                rel: {key: float(vals[key][k]) for key in vals} for rel, vals in sims.items()
            }
            cf["rank_ctx"] = float(rank_ctx[k])
            cf["rank_ans"] = float(rank_ans[k])
            k += 1
    assert k == len(pi), (k, len(pi))

    # attribution join (kresample classes from P0.5)
    kz = np.load(_derived(args) / "kresample_ranks.npz", allow_pickle=False)
    kres_ci = np.asarray(kz["ci"], dtype=np.int64)
    kres_s = np.asarray(kz["s"], dtype=np.float64)
    kres_cls = [str(x) for x in kz["classes"]]
    kmap = {int(c): (float(kres_s[i]), kres_cls[i]) for i, c in enumerate(kres_ci)}
    covered = np.asarray([int(c) in kmap for c in pci])
    attribution_of = np.asarray(
        [kmap.get(int(c), (np.nan, "UNKNOWN"))[1] for c in pci], dtype=object
    )
    s_of = np.asarray([kmap.get(int(c), (np.nan, "UNKNOWN"))[0] for c in pci])
    for rec in conf_rows:
        rec["attribution"] = str(attribution_of[rec["row"]])

    fail0 = fail[PRIMARY_SPACE]
    cls_counts_fail = {
        cls: int(((attribution_of == cls) & fail0).sum())
        for cls in ("MAP_ATTRIBUTABLE", "IRREDUCIBLE", "AMBIGUOUS", "UNKNOWN")
    }
    n_cov_fail = int((covered & fail0).sum())
    attribution = {
        "acc1_ceiling": float(kres_s.mean()),
        "k_draws": 4,
        "coverage": {
            "n_covered": int(covered.sum()),
            "n_holdout": n,
            "n_fail_covered": n_cov_fail,
            "n_fail_uncovered": int((~covered & fail0).sum()),
        },
        "classes_over_fail1": cls_counts_fail,
        "shares_covered_fail": {
            cls: (cls_counts_fail[cls] / n_cov_fail if n_cov_fail else None)
            for cls in ("MAP_ATTRIBUTABLE", "IRREDUCIBLE", "AMBIGUOUS")
        },
        "repr_check_a": {
            "fail_rate_covered": float(fail0[covered].mean()) if covered.any() else None,
            "fail_rate_uncovered": float(fail0[~covered].mean()) if (~covered).any() else None,
        },
        "ceiling_narration": (
            "acc1_ceiling is the retrievability of a fresh on-policy draw (draw-vs-draw); "
            "an ideal conditional-mean map could exceed it. IRREDUCIBLE is an UPPER BOUND "
            "on irreducibility (plan §3 H2)."
        ),
        "meta": meta_block(),
    }
    atomic_json(out / "attribution.json", attribution)

    # near-duplicate/tie named line: top confuser within the mid-rank tie tolerance
    tie_top = np.abs(s_conf[PRIMARY_SPACE] - 1.0) <= 1e-9
    frac_sconf_tie = float(tie_top[fail0].mean()) if n_fail else 0.0

    # concordance vs banked nerr
    nerr_map: dict[int, float] = {}
    with open(repo_banked_path(args.nerr_csv), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nerr_map[int(row["ci"])] = float(row["nerr_context_L19_ridge"])
    nerr = np.asarray([nerr_map[int(c)] for c in pci])
    rho, pval = spearmanr(rk, nerr)
    top_nerr = set(np.lexsort((np.arange(n), -nerr))[: min(WORST_N, n)].tolist())
    concordance = {
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "n": n,
        "overlap_top200_rank_vs_top200_nerr": len(worst_rank & top_nerr),
        "overlap_fail1_vs_topnfail_nerr": len(
            set(np.nonzero(fail0)[0].tolist())
            & set(np.lexsort((np.arange(n), -nerr))[:n_fail].tolist())
        ),
        "meta": meta_block(),
    }
    atomic_json(out / "concordance.json", concordance)

    # hubness
    def _hub_block(counts: np.ndarray) -> dict:
        top20 = np.lexsort((np.arange(n), -counts))[:20]
        return {
            "n10_skewness": skewness(counts),
            "n10_max": int(counts.max()),
            "n10_frac_zero": float((counts == 0).mean()),
            "top20": [{"ci": int(pci[j]), "count": int(counts[j])} for j in top20],
            "counts": counts.tolist(),  # full per-pool-row N_10 (P5 in-degree figure)
        }

    atomic_json(
        out / "hubness.json",
        {
            "retrieval": _hub_block(n10_retrieval),
            "collapse": _hub_block(n10_collapse),
            "meta": meta_block(),
        },
    )

    # pool robustness summary
    pool_doc: dict = {"pool_sizes": {}, "seed": SEED, "meta": meta_block()}
    for p, r_sub in sub_ranks.items():
        f_sub = r_sub > 1.0
        pool_doc["pool_sizes"][str(p)] = {
            "n_fail": int(f_sub.sum()),
            "fail_rate": float(f_sub.mean()),
            "jaccard_vs_full": float((f_sub & fail0).sum() / max(1, (f_sub | fail0).sum())),
        }
    pool_doc["pool_sizes"][str(n)] = {
        "n_fail": n_fail,
        "fail_rate": float(fail0.mean()),
        "jaccard_vs_full": 1.0,
    }
    atomic_json(out / "pool_robustness.json", pool_doc)

    # per-context CSV (text-free)
    csv_path = out / "percontext_ranks.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = (
        ["ci"]
        + [f"rank_{sp}" for sp in SPACES]
        + ["d_true_raw_euclidean", "n_outrank"]
        + [f"s_conf_{sp}" for sp in SPACES]
        + ["tie_top"]
        + [f"fail_{sp}" for sp in SPACES]
        + [f"rank_pool{p}" for p in subpools]
        + [f"fail_pool{p}" for p in subpools]
        + [
            "in_sample500",
            "worst_rank_tail",
            "worst_dist_tail",
            "kres_covered",
            "kres_s",
            "kres_class",
            "nerr",
        ]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow(
                [int(pci[i])]
                + [f"{ranks[sp][i]:.1f}" for sp in SPACES]
                + [f"{dt0[i]:.6g}", int(n_closer[PRIMARY_SPACE][i])]
                + [f"{s_conf[sp][i]:.6f}" for sp in SPACES]
                + [int(tie_top[i])]
                + [int(fail[sp][i]) for sp in SPACES]
                + [f"{sub_ranks[p][i]:.1f}" for p in subpools]
                + [int(sub_ranks[p][i] > 1.0) for p in subpools]
                + [
                    int(i in sample_set),
                    int(i in worst_rank),
                    int(i in worst_dist),
                    int(covered[i]),
                    "" if np.isnan(s_of[i]) else f"{s_of[i]:.2f}",
                    str(attribution_of[i]),
                    f"{nerr[i]:.6g}",
                ]
            )

    atomic_json(
        out / "failures_confusion.json",
        {
            "n_fail1": n_fail,
            "n_detail_rows": len(conf_rows),
            "confusers_per_row": conf_k,
            "frac_sconf_tie_among_fail1": frac_sconf_tie,
            "primary_space": PRIMARY_SPACE,
            "rows": conf_rows,
            "meta": meta_block({"smoke": bool(args.smoke)}),
        },
    )
    atomic_json(
        out / "sample500_lists.json",
        {
            "seed": SEED,
            "n_sample": sample_n,
            "rows": [
                {
                    "ci": int(pci[gi]),
                    "rank": float(rk[gi]),
                    "fail": bool(fail0[gi]),
                    "retrieval": listA.get(int(gi), []),
                    "collapse": listB.get(int(gi), []),
                }
                for gi in sample_rows
            ],
            "meta": meta_block(),
        },
    )
    atomic_json(
        out / "geometry_summary.json",
        {
            "equivalence_gate": equiv,
            "fail_counts": {sp: int(fail[sp].sum()) for sp in SPACES},
            "jaccard": jac,
            "frac_sconf_tie_among_fail1": frac_sconf_tie,
            "worst_tail_overlap_rank_vs_dist": len(worst_rank & worst_dist),
            "meta": meta_block(),
        },
    )
    write_sentinel(
        args,
        "p1.done",
        "epm:progress",
        {"phase": "p1_geometry", "equiv": equiv["verdict"], "n_fail1": n_fail},
    )
    if equiv["verdict"] != "PASS":
        logger.error("[p1] FULL-POOL equivalence gate FAILED — halting before downstream reads")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(RC_GATE)


# ── P2: reciprocity + two nulls ───────────────────────────────────────────────────


def build_edges(
    pred: np.ndarray,
    ans: np.ndarray,
    chunk: int,
    cap_per_row: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Directed confusion edges i→j (a_j strictly outranks a_i under p_i).

    Returns (src, dst, rank_fwd, d_pred) with per-row (distance, index)
    tie-broken ordering; ``cap_per_row`` keeps each row's top-K when the
    pre-registered edge cap fired."""
    n = pred.shape[0]
    src_l: list[np.ndarray] = []
    dst_l: list[np.ndarray] = []
    fwd_l: list[np.ndarray] = []
    dpred_l: list[np.ndarray] = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = _pairwise_dist(pred[s:e], ans, "euclidean")
        dt = d[np.arange(e - s), np.arange(s, e)]
        tol = 1e-9 * np.maximum(np.abs(dt), 1e-12)
        for r_local in range(e - s):
            row = d[r_local]
            outr = np.nonzero(row < dt[r_local] - tol[r_local])[0]
            if len(outr) == 0:
                continue
            outr = outr[np.lexsort((outr, row[outr]))]
            if cap_per_row is not None:
                outr = outr[:cap_per_row]
            src_l.append(np.full(len(outr), s + r_local, dtype=np.int64))
            dst_l.append(outr.astype(np.int64))
            fwd_l.append(np.arange(1, len(outr) + 1, dtype=np.int64))
            dpred_l.append(row[outr])
        print(f"[p2-edges] rows {s}:{e} done", flush=True)
    if not src_l:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy(), z.copy(), np.zeros(0)
    return (
        np.concatenate(src_l),
        np.concatenate(dst_l),
        np.concatenate(fwd_l),
        np.concatenate(dpred_l),
    )


def reciprocity_of(src: np.ndarray, dst: np.ndarray, n: int) -> float:
    """P(j→i ∈ G | i→j ∈ G) over the UNIQUE directed edge set."""
    h = np.unique(src.astype(np.int64) * n + dst.astype(np.int64))
    rev = (h % n) * n + (h // n)
    return float(np.isin(rev, h, assume_unique=False).mean()) if len(h) else float("nan")


def degree_preserving_draws(
    src: np.ndarray, dst: np.ndarray, n: int, n_draws: int, seed: int
) -> tuple[np.ndarray, dict]:
    """Reciprocity per draw under the directed configuration model (target-stub
    permutation — preserves every out- and in-degree). Self-loop / multi-edge
    collisions are counted + reported (kept; membership over the unique set)."""
    rng = np.random.default_rng(seed)
    e_n = len(src)
    out = np.empty(n_draws)
    self_loops = np.zeros(n_draws, dtype=np.int64)
    multi = np.zeros(n_draws, dtype=np.int64)
    t0 = time.time()
    for d_i in range(n_draws):
        perm = rng.permutation(e_n)
        dst_p = dst[perm]
        self_loops[d_i] = int((dst_p == src).sum())
        h = src.astype(np.int64) * n + dst_p.astype(np.int64)
        hu = np.unique(h)
        multi[d_i] = e_n - len(hu)
        rev = (hu % n) * n + (hu // n)
        out[d_i] = float(np.isin(rev, hu).mean())
        if (d_i + 1) % 100 == 0 or d_i + 1 == n_draws:
            print(
                f"[p2-nulldp] unit {d_i + 1}/{n_draws} elapsed={time.time() - t0:.1f}s", flush=True
            )
    coll = {
        "self_loops_mean": float(self_loops.mean()) if n_draws else 0.0,
        "multi_edges_mean": float(multi.mean()) if n_draws else 0.0,
    }
    return out, coll


def distance_null_draws(
    d_ans: np.ndarray,
    out_deg: np.ndarray,
    tau: float,
    n_draws: int,
    seed: int,
    draw_chunk: int = 50,
) -> np.ndarray:
    """Reciprocity per draw under the distance-only null.

    Per source i with out-degree k_i > 0: draw k_i targets WITHOUT replacement
    with P(i→j) ∝ exp(−d_ans[i,j]/τ) via Gumbel-top-k (self excluded); the
    kernel is symmetric by construction. Vectorized per (source × draw-chunk)
    over the full pool row; membership per draw via sorted-hash search.
    """
    n = d_ans.shape[0]
    rng = np.random.default_rng(seed)
    sources = np.nonzero(out_deg > 0)[0]
    out = np.empty(n_draws)
    t0 = time.time()
    for c0 in range(0, n_draws, draw_chunk):
        dc = min(draw_chunk, n_draws - c0)
        per_draw_h: list[list[np.ndarray]] = [[] for _ in range(dc)]
        for i in sources:
            k_i = int(out_deg[i])
            base = -d_ans[i].astype(np.float64) / tau
            base[i] = -np.inf
            g = rng.gumbel(size=(dc, n))
            keys = base[None, :] + g
            top = np.argpartition(keys, n - k_i, axis=1)[:, n - k_i :]
            for d_i in range(dc):
                per_draw_h[d_i].append(np.int64(i) * n + top[d_i].astype(np.int64))
        for d_i in range(dc):
            h = np.unique(np.concatenate(per_draw_h[d_i]))
            rev = (h % n) * n + (h // n)
            out[c0 + d_i] = float(np.isin(rev, h).mean())
        print(
            f"[p2-nulldo] unit {c0 + dc}/{n_draws} tau={tau:.4g} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    return out


def phase_reciprocity(args) -> None:
    """P2 — observed reciprocity + graded companion + the two null bands."""
    logger.info("[phase=p2_reciprocity] start (smoke=%s)", args.smoke)
    pred, y16, pci = load_pred_y(args)
    n = len(pci)
    chunk = args.chunk_rows
    n_draws = SMOKE_DRAWS if args.smoke else args.n_null_draws

    # cap decision from a cheap count pass? E_uncapped = sum n_outrank — reuse the
    # geometry CSV when present; else count in the edge pass (uncapped first).
    src, dst, rank_fwd, d_pred = build_edges(pred, y16, chunk, cap_per_row=None)
    e_uncapped = len(src)
    cap_applied = e_uncapped > args.graph_edge_cap
    if cap_applied:
        src, dst, rank_fwd, d_pred = build_edges(pred, y16, chunk, cap_per_row=args.graph_topk_cap)
    logger.info("[p2] E_uncapped=%d cap_applied=%s E=%d", e_uncapped, cap_applied, len(src))

    obs = reciprocity_of(src, dst, n)

    # reverse ranks: rank of a_i under p_j per edge (i→j) — grouped second pass
    rank_rev = np.empty(len(src))
    need_by_row: dict[int, list[int]] = {}
    for k, j in enumerate(dst):
        need_by_row.setdefault(int(j), []).append(k)
    need_rows = np.asarray(sorted(need_by_row), dtype=np.int64)
    for s in range(0, len(need_rows), chunk):
        blk = need_rows[s : s + chunk]
        d = _pairwise_dist(pred[blk].astype(np.float64), y16, "euclidean")
        for r_local, j in enumerate(blk):
            ks_ = need_by_row[int(j)]
            cols = src[ks_]
            rank_rev[ks_] = ranks_of_cols_in_row(d[r_local], cols)
        print(f"[p2-rev] rows {s}:{min(s + chunk, len(need_rows))}/{len(need_rows)}", flush=True)
    graded_rho, graded_p = (
        spearmanr(rank_fwd, rank_rev) if len(src) > 2 else (float("nan"), float("nan"))
    )

    # pairwise answer distances (fp32 full matrix — the plan's ~400 MB object) + τ
    d_ans_sq = _pairwise_dist(y16.astype(np.float32), y16.astype(np.float32), "euclidean")
    d_ans = np.sqrt(np.maximum(d_ans_sq, 0.0), dtype=np.float32)
    del d_ans_sq
    off = d_ans[~np.eye(n, dtype=bool)]
    taus = {f"p{p}": float(np.percentile(off, p)) for p in TAU_PCTS}
    del off

    out_deg = np.bincount(src, minlength=n)
    dp_draws, collisions = degree_preserving_draws(src, dst, n, n_draws, SEED)
    do_draws = {
        name: distance_null_draws(d_ans, out_deg, tau, n_draws, SEED + 7 + i)
        for i, (name, tau) in enumerate(taus.items())
    }

    def _band(v: np.ndarray) -> dict:
        return {
            "mean": float(v.mean()),
            "p025": float(np.quantile(v, 0.025)),
            "p975": float(np.quantile(v, 0.975)),
        }

    delta_dp = obs - _band(dp_draws)["p975"]
    delta_do = obs - _band(do_draws["p5"])["p975"]
    doc = {
        "observed": {
            "reciprocity": obs,
            "E": int(len(src)),
            "E_uncapped": int(e_uncapped),
            "cap_applied": bool(cap_applied),
            "cap_per_row": args.graph_topk_cap if cap_applied else None,
            "n_sources": int((out_deg > 0).sum()),
        },
        "graded": {
            "spearman_fwd_rev": float(graded_rho),
            "spearman_p": float(graded_p),
            "n_edges": int(len(src)),
            "edges_npz": "analysis_tensors/reciprocity_edges.npz",
        },
        "null_degree": {"draws": dp_draws.tolist(), "band": _band(dp_draws), **collisions},
        "null_distance": {
            name: {"tau": taus[name], "draws": v.tolist(), "band": _band(v)}
            for name, v in do_draws.items()
        },
        "n_draws": n_draws,
        "verdict_inputs": {"delta_dp": float(delta_dp), "delta_do_p5": float(delta_do)},
        "ceiling": 1.0,
        "meta": meta_block({"smoke": bool(args.smoke), "seed": SEED}),
    }
    out = out_eval_dir(args)
    atomic_json(out / "reciprocity.json", doc)
    _derived(args).mkdir(parents=True, exist_ok=True)
    np.savez(
        _derived(args) / "reciprocity_edges.npz",
        src_ci=pci[src],
        dst_ci=pci[dst],
        rank_fwd=rank_fwd,
        rank_rev=rank_rev,
        d_pred=d_pred,
    )
    write_sentinel(
        args,
        "p2.done",
        "epm:progress",
        {"phase": "p2_reciprocity", "reciprocity": obs, "E": int(len(src)), "n_draws": n_draws},
    )
    logger.info("[p2] done: reciprocity=%.4f Δdp=%.4f Δdo(p5)=%.4f", obs, delta_dp, delta_do)


# ── upload phase (bulk HF + git commit/push + results sentinel) ───────────────────


def shard_json_rows(rows: list[dict], stem: str, dest_dir: Path, max_bytes: int = 9_000_000):
    """Line-shard a row list into <9 MB JSONL shards + a manifest (upload-policy
    text-shard convention). Returns the written file names."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    shard: list[str] = []
    size = 0
    idx = 0

    def _flush():
        nonlocal shard, size, idx
        if not shard:
            return
        name = f"{stem}.shard{idx:02d}.jsonl"
        (dest_dir / name).write_text("\n".join(shard) + "\n", encoding="utf-8")
        names.append(name)
        idx += 1
        shard, size = [], 0

    for r in rows:
        line = json.dumps(r, ensure_ascii=False)
        if size + len(line) + 1 > max_bytes and shard:
            _flush()
        shard.append(line)
        size += len(line) + 1
    _flush()
    manifest = f"{stem}.manifest.json"
    atomic_json(dest_dir / manifest, {"stem": stem, "shards": names, "n_rows": len(rows)})
    return names + [manifest]


def git_commit_results(args, paths: list[Path]) -> None:
    """#1205 result-push contract: explicit-path add + commit, fetch+rebase,
    push, rev-list==0 verify, per-file ls-tree assert."""
    if args.no_git:
        logger.info("[git] commit/push SKIPPED (--no-git)")
        return
    root = PROJECT_ROOT
    env = {**os.environ}

    def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        if check and proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed rc={proc.returncode}: {proc.stderr[-500:]}")
        return proc

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    outside = [p for p in paths if p.exists() and not p.resolve().is_relative_to(root)]
    if outside:
        raise ValueError(
            f"git-destined result paths outside the repo root {root}: {outside[:3]} — "
            f"use --no-git for scratch out-eval roots"
        )
    rels = [str(p.resolve().relative_to(root)) for p in paths if p.exists()]
    if not rels:
        logger.info("[git] push-verify: no git-destined outputs declared this round (no-op)")
        return
    _run(["git", "add", "--"] + rels)
    staged = _run(["git", "diff", "--cached", "--name-only"]).stdout.strip()
    if staged:
        msg = Path(args.work_root) / "commit_msg.txt"
        msg.write_text(
            f"task #{ISSUE}: pod-side eval results ({'smoke' if args.smoke else 'production'})\n"
        )
        _run(["git", "commit", "-F", str(msg), "--", *rels])
    else:
        logger.info("[git] nothing new to commit (idempotent re-run)")
    for attempt in range(2):
        _run(["git", "fetch", "origin", branch])
        reb = _run(["git", "rebase", f"origin/{branch}"], check=False)
        if reb.returncode != 0:
            _run(["git", "rebase", "--abort"], check=False)
            raise RuntimeError(f"rebase onto origin/{branch} conflicted: {reb.stderr[-300:]}")
        push = _run(["git", "push", "origin", branch], check=False)
        behind = _run(["git", "rev-list", "--count", f"origin/{branch}..HEAD"]).stdout.strip()
        if push.returncode == 0 and behind == "0":
            break
        if attempt == 1:
            raise RuntimeError(
                f"push verify failed (rc={push.returncode}, ahead-count={behind}): "
                f"{push.stderr[-300:]}"
            )
    logger.info("[git] push-verify expected paths (%d): %s", len(rels), rels)
    for rel in rels:
        ls = _run(["git", "ls-tree", "-r", f"origin/{branch}", "--name-only", "--", rel])
        if not ls.stdout.strip():
            raise RuntimeError(f"pushed tree is missing declared result path {rel}")


def phase_upload(args) -> None:
    """Terminal phase: bulk HF upload + git result commit + results sentinel +
    the single reserved ``[phase=done]`` line."""
    logger.info("[phase=p3_upload] start")
    out = out_eval_dir(args)
    prefix = hf_prefix(args)
    eval_files = sorted([p for p in out.iterdir() if p.is_file() and p.suffix in (".json", ".csv")])
    if not args.no_upload:
        upload_derived(args)  # idempotent (analysis_tensors incl. reciprocity_edges.npz)
        # rows_geom: text-free geometry rows, line-sharded < 9 MB
        stage = Path(args.work_root) / ("hf_rows_smoke" if args.smoke else "hf_rows")
        fc = json.loads((out / "failures_confusion.json").read_text())
        names = shard_json_rows(fc["rows"], "failures_geom", stage)
        sl = json.loads((out / "sample500_lists.json").read_text())
        names += shard_json_rows(sl["rows"], "sample500_geom", stage)
        dest = f"{prefix}/rows_geom"
        url = hub._upload_folder_filtered(
            stage,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=dest,
            allow_patterns=names,
            expected_repo_paths=[f"{dest}/{nm}" for nm in names],
        )
        if not url:
            raise RuntimeError(f"rows_geom upload to {dest} returned no URL")
        # eval_mirror: every eval_results JSON/CSV (pod is ephemeral — HF twin of git)
        mdest = f"{prefix}/eval_mirror"
        murl = hub._upload_folder_filtered(
            out,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=mdest,
            allow_patterns=[p.name for p in eval_files],
            expected_repo_paths=[f"{mdest}/{p.name}" for p in eval_files],
        )
        if not murl:
            raise RuntimeError(f"eval_mirror upload to {mdest} returned no URL")
        logger.info("[upload] rows_geom + eval_mirror -> %s", prefix)
    else:
        logger.info("[upload] HF uploads SKIPPED (--no-upload)")

    git_commit_results(args, eval_files)

    def _rel(p: Path) -> str:
        rp = p.resolve()
        return str(rp.relative_to(PROJECT_ROOT)) if rp.is_relative_to(PROJECT_ROOT) else str(rp)

    note = {
        "phase": "upload",
        "smoke": bool(args.smoke),
        "hf_prefix": prefix,
        "eval_json_paths": [_rel(p) for p in eval_files],
        "meta": meta_block(),
    }
    write_sentinel(
        args,
        f"{'epm_smoke-result' if args.smoke else 'epm_results'}-{int(time.time())}",
        "epm:smoke-result" if args.smoke else "epm:results",
        note,
        blocks=not args.smoke,
    )
    print("[phase=done]", flush=True)


# ── phase registry + main ─────────────────────────────────────────────────────────

PHASES = {
    "repro-gate": phase_repro_gate,
    "extract": phase_extract,
    "geometry": phase_geometry,
    "reciprocity": phase_reciprocity,
    "upload": phase_upload,
}
PHASE_ORDER = ["repro-gate", "extract", "geometry", "reciprocity", "upload"]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    ap.add_argument("--phase", choices=[*PHASE_ORDER, "all"], default=None)
    ap.add_argument("--smoke", action="store_true", help="sliced rows / draws; POOL stays full")
    ap.add_argument("--import-check", action="store_true", dest="import_check")
    ap.add_argument("--list-phases", action="store_true", dest="list_phases")
    ap.add_argument(
        "--work-root",
        default="/workspace/data/issue_2202",
        help="staging + memmap + derived root (canonical, NON-rebinding under --smoke)",
    )
    ap.add_argument("--out-eval", default=str(PROJECT_ROOT / "eval_results" / "issue_2202"))
    ap.add_argument("--hf-prefix", default=HF_PREFIX_2202)
    ap.add_argument("--parent-prefix", default=PARENT_PREFIX)
    ap.add_argument("--revision", default=HF_PIN, help="data-repo pin for staged inputs")
    ap.add_argument("--banked-json", default=BANKED_REL)
    ap.add_argument("--nerr-csv", default=NERR_REL)
    ap.add_argument(
        "--local-inputs",
        action="store_true",
        dest="local_inputs",
        help="skip HF staging; staged/ inputs pre-placed (tests)",
    )
    ap.add_argument(
        "--local-capture-dir",
        default="",
        dest="local_capture_dir",
        help="local capture-chunk dir for assemble_streams (tests)",
    )
    ap.add_argument("--no-upload", action="store_true", dest="no_upload")
    ap.add_argument("--no-git", action="store_true", dest="no_git")
    ap.add_argument("--sentinel-dir", default="/workspace/logs")
    ap.add_argument("--chunk-rows", type=int, default=1024, dest="chunk_rows")
    ap.add_argument("--n-null-draws", type=int, default=N_NULL_DRAWS, dest="n_null_draws")
    ap.add_argument("--graph-edge-cap", type=int, default=GRAPH_EDGE_CAP, dest="graph_edge_cap")
    ap.add_argument("--graph-topk-cap", type=int, default=GRAPH_TOPK_CAP, dest="graph_topk_cap")
    ap.add_argument("--headroom-gb", type=float, default=19.0, dest="headroom_gb")
    return ap


def _import_check() -> None:
    """--import-check body at module level (never inside main — the in-function
    import would shadow module names function-wide, the #1739 UnboundLocalError
    class)."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    print("import-check OK: issue2202_failchar")


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        _import_check()
        return 0
    if args.list_phases:
        print(json.dumps(PHASE_ORDER))
        return 0
    if not args.phase:
        raise SystemExit("--phase is required (or --import-check / --list-phases)")
    args.work_root = Path(args.work_root)
    phases = PHASE_ORDER if args.phase == "all" else [args.phase]
    for ph in phases:
        PHASES[ph](args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit BEFORE C-extension teardown (gotchas: PyGILState race)
