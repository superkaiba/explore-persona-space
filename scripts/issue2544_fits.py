"""#2544 P4a/P4b fit engine (Olmo-3-7B 15-rung stage map) — Unit B.

Called by ``issue2544_run.py --phase fits`` -> :func:`run_fits`. Two stages,
discriminated by ``EPM_ISSUE2544_FITS_STAGE=p4a|p4b|auto`` (the dispatcher
pins the stage explicitly; auto = P4b iff the layer-freeze record resolves):

- **P4a** (plan §4): slow-vs-fast ridge parity gate at production shape ->
  15x6 diagonal 17-layer sweep units (per-context SS persisted at full
  17-layer grain, identity SS beside for the D-tilde CIs) -> layer*
  selection -> 15x6 star units at layer* (shuffled-pairing null + kNN +
  identity + reliability-repeat scoring + the trained-only-sensitivity
  extra-row predictions) -> Gate B (Q(main) floor + one-sided scramble leak
  cap) -> the layer-freeze record ``fits/layer_freeze.json``
  ({layer_star, band_b5, layer_fa, band_b6}), mirrored to HF for pass2.
- **P4b** (plan §4): 816 grid-class banded ridge units (49 full cross/k-shot
  cells + 15 l_FA 0-shot diagonals + 39 subset cells + 33 n-MATCHED
  companions, x6 folds) + 52x6 transfer units (direct / general-linear /
  orthogonal-Procrustes with train-fold alignments, A_ans on same-answer-text
  pairs from the s=main column, N1 shuffled-correspondence + N2
  spectrum-matched nulls) + 52 operator units (Procrustes-aligned operator
  cosine vs the 50-draw rotation null) -> finalize: 1,000-draw cluster-grouped
  bootstrap as batched masked-sum GEMMs over persisted SS (no refits per
  draw), selection-inherited CIs, Delta(T)/Delta_ww/Delta_FA, ceilings,
  T_half/T_c, and the registered output JSONs under
  ``eval_results/issue_2544/``.

Disk discipline (plan §9 disk rows): store shards are staged PER FILE at the
consumed layer grain (never a whole-prefix pull), in RUNG WAVES
(``EPM_ISSUE2544_FIT_RESIDENT``, default 2 for P4a / 1 for P4b), and each
wave's tensors are reaped after its units drain (sidecar-present files only —
the verified-upload evidence guard; ``row_index.jsonl`` + sidecars are kept).
The P4b transfer working set (star-layer diag0 + colC + l_FA files, ~9 GB)
stays resident through the transfer/operator stages.

Reviewer-derived clauses implemented here (plan §4/§11):

- **S2**: ``gcv_dof_cap=0.9`` passed EXPLICITLY on EVERY production ridge
  call through the single :func:`_ridge` chokepoint (observed fits, subset
  fits, alignment fits, null refits); per-fit selector/lambda*/dof persisted.
- **S3**: identity+bias R^2 AND euclidean/cosine kNN retrieval registered on
  every fitted CELL and every fitted MAP — A_ctx/A_ans general-linear
  alignments, both Procrustes maps, the composite, and the native/plain 2k
  subset fits (chance = k/n_pool stated by ``knn_retrieval``).
- **S1**: n-matched companions — cmp_o1_6k/cmp_z_6k at every rung and
  cmp_plain_2k at S/D/R — fit on the IDENTICAL subset-cap-intersection rows +
  folds; subset reads are only ever compared subset-vs-subset; every
  ``n_tr < d`` fit carries the justified disclosure.
- **A1**: band reads carry the sliding-vs-full split (``C2.layer_type_split``);
  the l_FA companion (Delta_FA) and the within-window Delta_ww subset read are
  first-class outputs in ``kshot_curve.json``.
- **M2**: fit-unit fingerprints (kinds ``fits`` / ``fits_p4b``, registered
  additively into ``C2.PHASE_FP_FIELDS``) bind the consumed store shards'
  write-time sha sidecars + the roster entry + BOTH sides' pins; a P4a
  fingerprint carrying ``freeze_sha`` (a not-yet-produced artifact) is REFUSED
  by ``C2.build_fingerprint`` — the DAG check.

Vectorization: the layer axis is batched through
``fit_h.ridge_fit_predict_fast_layer_batched`` (one batched Gram eigh per
chunk, via ``issue1902_fits._batched_ridge``'s cuSOLVER->CPU fallback);
(cell x fold) units fan across GPUs via ``issue1902_fits._WorkerPool``
device-pinned threads; bootstrap draws are batched GEMMs; rotation nulls run
batched QR on the fit device. No serial per-cell factorization loop.

IMPORT ORDER: ``issue2544_common`` FIRST (it sets the ladder + write-prefix
env before ``issue1902_common`` binds its constants).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy/torch: the shared-VM thread caps bind in-process (#847)

import numpy as np  # noqa: E402

import issue2544_common as C2  # noqa: E402  (MUST precede issue1902_* — env-ordered)
import issue1902_common as C  # noqa: E402
import issue1902_fits as F1  # noqa: E402
import issue1902_run as R  # noqa: E402
import issue2544_run as R2  # noqa: E402

logger = logging.getLogger("issue2544_fits")

# ── constants ────────────────────────────────────────────────────────────────

FITS_RECIPE_VERSION = "issue2544-fits-v1"
GCV_DOF_CAP = 0.9  # plan §11 (#1887; #825 module default) — threaded via _ridge
# Symbolic lambda-grid spec for fingerprints: NEVER the recomputed float bytes
# (the #1336 logspace 1-ULP trap); the realized grid is fit_h's default
# np.logspace(-2, 4, 13).
LAMBDA_SPEC = "logspace(-2,4,13)"
SELECTOR_TAG = f"gcv(dof_cap={GCV_DOF_CAP})"

N_BOOT = int(os.environ.get("EPM_ISSUE2544_N_BOOT", "1000"))
N_ROT_DRAWS = 50  # issue1345 N_ROTATION_COSINE_DRAWS convention
N_NULL_DRAWS = 4  # N1 shuffled-correspondence + N2 spectrum-matched, per fold
N_SHUFFLE_DRAWS = 3  # shuffled-pairing null draws per (diag cell, fold)
SMOKE_N_BOOT = 50
SMOKE_N_ROT = 4
SMOKE_N_NULL = 2
SMOKE_N_SHUFFLE = 2
BOOT_SEED = C.FOLD_SEED + 2544

GATE_B_FLOOR = 0.15  # §7 Gate B: diagonal Q(main) at layer*
SCRAMBLE_LEAK_CAP = 0.05  # §7 Gate B: ONE-SIDED scramble leak cap (never |R^2|<eps)

# In-run pilot bases (plan §9, MEASURED #1902 per-unit-class figures) + the
# pre-registered abort ratio.
P4A_UNIT_BASIS_S = 15.3
P4B_GRID_BASIS_S = 12.8
P4B_XFER_BASIS_S = 43.0
PILOT_ABORT_RATIO = 2.0

FORMATION_FLOORS = (0.05, 0.10, 0.15)  # §3 floor-sensitivity read
EARLY_MAX_RUNG = "r2"  # §3 lattice bins (rung-indexed)
INTERMEDIATE_MAX_RUNG = "r6"
POST_RUNGS = ("S", "D", "R")
TRAINED_ONLY_MIN_RUNG = "r3"  # §6 scope-caveat (ii) suffix start

WW_WINDOW = C2.OLMO3_SLIDING_WINDOW  # within-window row cap (Delta_ww)

# M2 fit-unit fingerprint kinds — registered ADDITIVELY into the shared
# registry (Unit A: "the registry is additive"). Two kinds so the DAG check
# structurally refuses a P4a fingerprint that references the freeze record
# (produced BY P4a) — build_fingerprint raises on the undeclared field.
_FITS_FP_FIELDS = frozenset(
    {"code_sha", "fit_config", "store_shas", "intersection_sha", "roster", "pins"}
)
C2.PHASE_FP_FIELDS.setdefault("fits", _FITS_FP_FIELDS)
C2.PHASE_FP_FIELDS.setdefault("fits_p4b", _FITS_FP_FIELDS | {"freeze_sha"})

_FITS_CODE_FILES: tuple[tuple[Path, str], ...] = (
    (_SCRIPTS_DIR / "issue2544_fits.py", "issue2544_fits.py"),
    (_SCRIPTS_DIR / "issue2544_common.py", "issue2544_common.py"),
    (_SCRIPTS_DIR / "issue1902_fits.py", "issue1902_fits.py"),
    (PROJECT_ROOT / "src/explore_persona_space/experiments/issue_779/fit_h.py", "fit_h.py"),
    (PROJECT_ROOT / "src/explore_persona_space/analysis/mapping_baselines.py", "baselines.py"),
)


def fits_code_sha() -> str:
    """Content sha over the FIT-side code files (M2 'code SHA' for fit units).

    Deliberately separate from ``C2.code_sha()`` (gen/capture files): an edit
    to fit code must invalidate fit units WITHOUT invalidating gen/capture
    resume state (and vice versa).
    """
    h = hashlib.sha256()
    for path, name in _FITS_CODE_FILES:
        h.update(name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _fit_config(spine: "Spine") -> dict[str, Any]:
    return {
        "lambdas": LAMBDA_SPEC,
        "gcv_dof_cap": GCV_DOF_CAP,
        "fold_seed": spine.fold_seed,
        "n_folds": spine.n_folds,
        "standardize": "x-std-train/y-center-train",
        "recipe": FITS_RECIPE_VERSION,
    }


def _eval_dir(out_root: Path, smoke: bool) -> Path:
    """eval_results root. Smoke NEVER writes the repo tree (scratch redirect)."""
    env = os.environ.get("EPM_ISSUE2544_EVAL_DIR")
    if env:
        return Path(env)
    if smoke:
        return out_root / "eval_results" / "issue_2544"
    return R.PROJECT_ROOT / "eval_results" / "issue_2544"


# ── the S2 ridge chokepoint ──────────────────────────────────────────────────


def _ridge(Xtr, Ytr, Xev, *, device: str, gcv_dof_cap: float | None = GCV_DOF_CAP, **kw):
    """EVERY production ridge call in this module routes here: the layer-batched
    Gram-eigh ridge (``issue1902_fits._batched_ridge`` = fit_h layer-batched +
    cuSOLVER->CPU fallback) with ``gcv_dof_cap=0.9`` passed EXPLICITLY (S2 —
    ``fit_h.py:180`` declares it None/opt-in and the inherited wrapper does not
    supply it). The parity gate is the ONLY caller that overrides the cap
    (``gcv_dof_cap=None``, recorded — solver parity vs the cap-less SVD path).
    """
    return F1._batched_ridge(Xtr, Ytr, Xev, device=device, gcv_dof_cap=gcv_dof_cap, **kw)


def _fit_record(info: dict, li: int) -> dict[str, Any]:
    """Per-fit selection record (S2: selector/lambda*/dof persisted per fit)."""
    return {
        "lambda_star": float(info["best_lambda"][li]),
        "dof": float(info["dof"][li]),
        "selector": SELECTOR_TAG,
    }


def _ntr_lt_d_disclosure(n_tr: int, d: int) -> dict[str, Any] | None:
    if n_tr >= d:
        return None
    return {
        "n_tr": int(n_tr),
        "d": int(d),
        "justified": True,
        "note": (
            "registered `n_tr < d — justified` fit (plan §4 P0 row-set rule): "
            "read ONLY against the n-matched same-fold companions, never a "
            "full-corpus fit (#1417); dof cap engaged (#1887)"
        ),
        "gcv_dof_cap": GCV_DOF_CAP,
    }


# ── store staging (2544 layout — C2.STORE_HF_PATH, per-cell subdirs) ─────────
#
# Staged PER FILE at the consumed layer grain (plan §9 "layer-streamed"
# staged-re-reads row): the whole 17L x 15-rung store is ~112 GB — a
# whole-prefix pull would breach the ~130 GB /workspace quota. Wave callers
# reap tensors between waves via _reap_staged.


def _stage_cell_files(
    store: Path, subdir: str, layers: list[int], *, soft: bool = False, tensors: bool = True
) -> bool:
    """Ensure ``<subdir>/row_index.jsonl`` + ``L{l}.pt`` (+ sha sidecars) for
    exactly the consumed layers are on local disk, staging each missing file
    from ``C2.STORE_HF_PATH/<subdir>`` (atomic, retried, skip-existing).
    ``tensors=False`` stages only the tiny side files (row_index + sha
    sidecars — the fingerprint/regime inputs), so a fully-resumed wave never
    re-downloads reaped tensors. Returns False (instead of raising) only for
    ``soft`` cells missing on HF."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub.utils import EntryNotFoundError

    from explore_persona_space.orchestrate import hub

    names = ["row_index.jsonl", "row_index.jsonl.sha256"]
    for layer in layers:
        names += [f"L{layer}.pt.sha256"] + ([f"L{layer}.pt"] if tensors else [])
    missing = [n for n in names if not (store / subdir / n).exists()]
    if not missing:
        return True

    def _one(name: str) -> None:
        hub.stage_hub_file(
            C.HF_DATA_REPO,
            f"{C2.STORE_HF_PATH}/{subdir}/{name}",
            store / subdir / name,
            repo_type="dataset",
        )

    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for fut in [pool.submit(_one, n) for n in missing]:
                fut.result()  # re-raises — fail loud
    except EntryNotFoundError as e:
        if soft:
            logger.warning("[fits] SOFT cell %s missing on HF: %s", subdir, e)
            return False
        raise FileNotFoundError(
            f"store cell {subdir} missing locally and on HF "
            f"({C2.STORE_HF_PATH}/{subdir}) — run the capture pass first"
        ) from e
    return True


def ensure_cells_staged(
    out_root: Path,
    needed: dict[str, list[int]],
    *,
    soft: frozenset[str] = frozenset(),
    tensors: bool = True,
) -> None:
    store = R._store_root(out_root)
    for subdir in sorted(needed):
        _stage_cell_files(
            store, subdir, sorted(set(needed[subdir])), soft=subdir in soft, tensors=tensors
        )


def _reap_staged(store: Path, pairs: set[tuple[str, int]], keep: set[tuple[str, int]]) -> int:
    """Delete staged layer tensors after a wave drains (disk-bounding; plan §9
    delete-between-slabs). Only files WITH a sha sidecar are deleted — the
    sidecar is written at verified-upload time, so its presence is the
    evidence the bytes are durable on HF (never delete never-uploaded data).
    ``row_index.jsonl`` + sidecars are kept (tiny; consumed by finalize)."""
    n = 0
    for subdir, layer in sorted(pairs - keep):
        p = store / subdir / f"L{layer}.pt"
        if p.exists() and Path(str(p) + ".sha256").exists():
            p.unlink()
            n += 1
    if n:
        logger.info("[fits] reaped %d staged layer tensors", n)
    return n


def _restage_hub_prefix(local_root: Path, hub_prefix: str, strip_prefix: str) -> int:
    """Scoped-listing + per-file staged restore of ONE hub prefix (used for
    the small eval-mirror artifacts — percell/units — on a fresh-pod P4b
    resume; one resolved revision covers every file, #833)."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = hub.retry_transient(
        lambda: api.repo_info(C.HF_DATA_REPO, repo_type="dataset"),
        what=f"repo_info({C.HF_DATA_REPO})",
    )
    revision = str(info.sha)
    files = hub.list_hf_files_under_path(
        api, C.HF_DATA_REPO, hub_prefix, repo_type="dataset", revision=revision
    )
    if not files:
        raise FileNotFoundError(f"no files on HF under {C.HF_DATA_REPO}:{hub_prefix}")
    sp = strip_prefix.rstrip("/") + "/"
    targets: dict[str, Path] = {}
    for f in files:
        if not f.startswith(sp):
            raise ValueError(f"hub path {f!r} outside prefix {sp!r}")
        targets[f] = local_root / f[len(sp) :]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [
            pool.submit(
                hub.stage_hub_file, C.HF_DATA_REPO, f, tgt, repo_type="dataset", revision=revision
            )
            for f, tgt in targets.items()
        ]
        for fut in futs:
            fut.result()  # re-raises — fail-loud
    return len(files)


# ── row/fold spine ───────────────────────────────────────────────────────────


class Spine:
    """Canonical analysis row space: the shared-intersection ``headline_ids``
    (manifest order), fold vector from the manifest's realized fold table,
    cluster-group index for the grouped bootstrap, and pinned-subset scopes.
    Every percell shard's ``row_idx`` is a POSITION IN THIS ORDER."""

    def __init__(self, args: argparse.Namespace, out_root: Path):
        man_path = out_root / "gen" / "intersection_manifest.json"
        if not man_path.exists():
            got = R2._hf_download(f"{C2.EVAL_MIRROR_HF_PATH}/gen/intersection_manifest.json")
            man_path.parent.mkdir(parents=True, exist_ok=True)
            man_path.write_bytes(got.read_bytes())
            logger.info("[spine] intersection manifest fetched from the HF mirror")
        self.manifest = R._read_json(man_path)
        self.intersection_sha = C2.sha256_file(man_path)
        self.rows = R2.load_rows(args, out_root)
        self.cfg = R2.load_config_bundle(out_root)
        self.meta: dict[str, dict] = {r["id"]: r for r in self.rows}
        self.ids: list[str] = [str(x) for x in self.manifest["headline_ids"]]
        assert self.ids, "empty headline intersection — Gate A' should have halted"
        self.pos = {rid: k for k, rid in enumerate(self.ids)}
        self.n = len(self.ids)
        self.fold_assign: dict[str, int] = dict(self.manifest["fold_assign"])
        self.fold_seed = int(self.manifest["fold_table"]["seed"])
        self.n_folds = C.N_FOLDS
        missing_meta = [i for i in self.ids if i not in self.meta]
        if missing_meta:
            raise RuntimeError(
                f"{len(missing_meta)} headline ids missing from the corpus rows "
                f"(e.g. {missing_meta[:3]}) — mixed smoke/production out-root?"
            )
        self.groups = [self.meta[i]["group"] for i in self.ids]
        self.classes = [(self.meta[i].get("class") or "generic") for i in self.ids]
        missing_g = sorted({g for g in self.groups if g not in self.fold_assign})
        if missing_g:
            raise RuntimeError(f"groups missing from manifest fold_assign: {missing_g[:5]}")
        self.fold = np.asarray([self.fold_assign[g] for g in self.groups], dtype=np.int64)
        self.group_names = sorted(set(self.groups))
        of = {g: k for k, g in enumerate(self.group_names)}
        self.gid = np.asarray([of[g] for g in self.groups], dtype=np.int64)
        self.n_groups = len(self.group_names)
        self._scope_cache: dict[str, list[str]] = {}

    def fold_of_group(self, group: str) -> int:
        """Fold for ANY corpus group: manifest assignment where present, else a
        deterministic hash fallback (extra-row scoring only — headline groups
        are always assigned)."""
        got = self.fold_assign.get(group)
        if got is not None:
            return int(got)
        return int(hashlib.sha256(str(group).encode()).hexdigest(), 16) % self.n_folds

    def scope_ids(self, scope: str) -> list[str]:
        if scope in ("full", "intersection"):
            return self.ids
        if scope not in self._scope_cache:
            subset = set(self.cfg["subsets"]["subsets"][scope])
            sel = [i for i in self.ids if i in subset]
            assert sel, f"empty scope {scope!r} ∩ headline intersection"
            self._scope_cache[scope] = sel
        return self._scope_cache[scope]

    def scope_positions(self, scope: str) -> np.ndarray:
        return np.asarray([self.pos[i] for i in self.scope_ids(scope)], dtype=np.int64)


# ── run context (duck-typed for issue1902_fits._WorkerPool) ──────────────────


class FitsCtx:
    def __init__(self, args: argparse.Namespace, out_root: Path, rungs: tuple[str, ...]):
        self.args = args
        self.smoke = bool(args.smoke)
        self.out_root = out_root
        self.rungs = list(rungs)
        self.spine = Spine(args, out_root)
        self.store = R._store_root(out_root)
        self.cache = F1.StoreCache(self.store)
        self.eval_dir = _eval_dir(out_root, self.smoke)
        self.pins = R.load_pins(out_root)
        self.pilot_timings: list[dict] = []  # _WorkerPool hook
        self.n_boot = SMOKE_N_BOOT if self.smoke else N_BOOT
        self.n_rot = SMOKE_N_ROT if self.smoke else N_ROT_DRAWS
        self.n_null = SMOKE_N_NULL if self.smoke else N_NULL_DRAWS
        self.n_shuffle = SMOKE_N_SHUFFLE if self.smoke else N_SHUFFLE_DRAWS
        self._posmaps: dict[str, dict[str, int]] = {}
        self._pos_lock = threading.Lock()
        self._w_cache: dict[str, np.ndarray] = {}
        self._w_lock = threading.Lock()
        # 17-layer capture set: store-discovered when local, dims-derived on a
        # fresh pod (staging happens after ctx construction).
        layer_sets = {r: self._layers17_for(r) for r in self.rungs}
        ref = layer_sets[self.rungs[0]]
        if any(ls != ref for ls in layer_sets.values()):
            raise RuntimeError(f"inconsistent diag0 layer sets: {layer_sets}")
        self.layers17: list[int] = ref
        if not self.smoke and self.layers17 != list(C.capture_layers(32)):
            raise RuntimeError(
                f"production layer set {self.layers17} != capture_layers(32) "
                f"{list(C.capture_layers(32))} — store drift"
            )
        # frozen at P4a / loaded at P4b:
        self.layer_star: int | None = None
        self.layer_fa: int | None = None
        self.band5: list[int] = []
        self.band6: list[int] = []
        self.freeze_sha: str | None = None

    # -- store access ---------------------------------------------------------

    def _layers17_for(self, rung: str) -> list[int]:
        """Capture-layer set for a rung's diag0 store.

        Discovers from LOCAL files (tensors AND .sha256 sidecars — sidecars
        persist across wave reaps, tensors do not), falling back to the
        model-dims-derived capture set on a fresh pod. A PARTIAL local
        subset is expected under wave staging/reaping (resume) and resolves
        to the full capture set; a local layer OUTSIDE the capture set is
        genuine store drift and raises.
        """
        d = self.store / f"{rung}/diag0/ctx"
        found = {int(p.name.split(".")[0][1:]) for p in d.glob("L*.pt*")}
        dims = C.model_dims(C.MODEL_IDS[rung], C.resolve_revision(rung, self.pins))
        full = list(C.capture_layers(dims.num_layers))
        if not found:
            return full
        if self.smoke:
            return sorted(found)
        extra = found - set(full)
        if extra:
            raise RuntimeError(
                f"{rung}: locally staged layers {sorted(extra)} outside the "
                f"capture set {full} — store drift"
            )
        return full

    def _positions(self, relpath: str, ids: list[str]) -> np.ndarray:
        with self._pos_lock:
            pm = self._posmaps.get(relpath)
        if pm is None:
            d = self.cache.subdir(relpath)
            pm = {str(rid): k for k, rid in enumerate(d["__row_ids__"])}
            with self._pos_lock:
                self._posmaps[relpath] = pm
        try:
            return np.asarray([pm[i] for i in ids], dtype=np.int64)
        except KeyError:
            missing = [i for i in ids if i not in pm]
            raise RuntimeError(
                f"{len(missing)} analysis ids missing from store {relpath} "
                f"(e.g. {missing[:3]}) — capture row coverage violated"
            ) from None

    def take(self, relpath: str, key: str, ids: list[str]) -> np.ndarray:
        d = self.cache.subdir(relpath)
        arr = d[key][self._positions(relpath, ids)]
        assert arr.ndim == 2 and arr.shape[0] == len(ids), (relpath, key, arr.shape, len(ids))
        return arr

    def store_sha(self, relpath: str) -> str:
        """Write-time sha sidecar of one consumed store shard (M2: recorded
        hashes, never a resume-time re-hash of multi-GB files)."""
        return C2.read_sha_sidecar(self.store / relpath)

    # -- 0-shot ctx/answer relpaths with the l_FA redirect ----------------------

    def diag0_ctx_rel(self, rung: str, layer: int) -> str:
        if layer in self.layers17:
            return f"{rung}/diag0/ctx/L{layer}.pt"
        return f"{rung}/lfa0/ctx/L{layer}.pt"  # pass-2 l_FA re-capture (intersection rows)

    def diag0_w_rel(self, rung: str, layer: int) -> str:
        if layer in self.layers17:
            return f"{rung}/diag0/L{layer}.pt"
        return f"{rung}/lfa0/L{layer}.pt"

    def col_main_w_rel(self, rung: str, layer: int) -> str:
        """w_m(x, a_main) — the same-answer-text answer store (s=main column;
        for m=main the diagonal IS the column entry)."""
        if rung == "main":
            return self.diag0_w_rel("main", layer)
        return f"{rung}/colC_main/L{layer}.pt"

    # -- unit io ---------------------------------------------------------------

    def unit_paths(self) -> tuple[Path, Path]:
        return self.eval_dir / "fits" / "units", self.eval_dir / "fits" / "percell"

    def write_unit(self, unit: str, payload: dict) -> None:
        units_dir, _ = self.unit_paths()
        units_dir.mkdir(parents=True, exist_ok=True)
        R._write_json_atomic(units_dir / f"{unit}.json", payload)

    def read_unit(self, unit: str) -> dict:
        units_dir, _ = self.unit_paths()
        return R._read_json(units_dir / f"{unit}.json")

    # -- full-data rung weights (operator reads; descriptive, #1332) -----------

    def rung_weights(self, rung: str, device: str) -> np.ndarray:
        with self._w_lock:
            got = self._w_cache.get(rung)
        if got is not None:
            return got
        layer = self.layer_star
        assert layer is not None
        X = self.take(self.diag0_ctx_rel(rung, layer), "u_mean", self.spine.ids)
        Y = self.take(self.diag0_w_rel(rung, layer), "w", self.spine.ids)
        _, W = _ridge(X[None], Y[None], X[:2][None], device=device, return_weights=True)
        with self._w_lock:
            self._w_cache[rung] = W[0]
        return W[0]


def _fold_masks(ctx: FitsCtx, scope: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gpos = ctx.spine.scope_positions(scope)
    fold_v = ctx.spine.fold[gpos]
    ev = fold_v == fold
    return ~ev, ev, gpos


def _unit_regime(
    ctx: FitsCtx,
    roster: dict[str, Any],
    store_relpaths: list[str],
    rung_pins: dict[str, str],
    *,
    freeze_sha: str | None = None,
) -> dict[str, Any]:
    """Per-unit resume regime = the M2 fit fingerprint (kind ``fits`` for P4a,
    ``fits_p4b`` once the freeze record is a declared input). A mismatch on
    resume REFUSES loudly (R.unit_done, the #1333 shape)."""
    fields: dict[str, Any] = dict(
        code_sha=fits_code_sha(),
        fit_config=_fit_config(ctx.spine),
        store_shas={rp: ctx.store_sha(rp) for rp in sorted(set(store_relpaths))},
        intersection_sha=ctx.spine.intersection_sha,
        roster=roster,
        pins=dict(sorted(rung_pins.items())),
    )
    kind = "fits"
    if freeze_sha is not None:
        kind = "fits_p4b"
        fields["freeze_sha"] = freeze_sha
    fp = C2.build_fingerprint(kind, **fields)
    return {"recipe": FITS_RECIPE_VERSION, "smoke": ctx.smoke, "fingerprint": fp}


# ── parity gate (plan §4 P4a — MANDATORY before production fits) ─────────────


def parity_gate_2544(
    ctx: FitsCtx,
    device: str,
    rungs_avail: list[str],
    resident: set[tuple[str, int]] | None = None,
) -> dict:
    """Slow (fit_h.ridge_fit_predict SVD) vs fast (layer-batched Gram-eigh)
    parity on >=3 slices at THIS run's production shape; max rel diff <= 1e-4
    or designed halt (informational under --smoke). SOLVER parity: both sides
    run the pure-GCV selector (``gcv_dof_cap=None`` on the fast side — the SVD
    reference has no cap parameter); production fits then engage the cap.
    ``resident`` = (subdir, layer) pairs the caller keeps staged; gate-only
    tensors outside it are reaped after the gate.
    """
    from explore_persona_space.experiments.issue_779.fit_h import ridge_fit_predict

    mid = ctx.layers17[len(ctx.layers17) // 2]
    slices = [
        (rungs_avail[0], mid),
        (rungs_avail[-1], ctx.layers17[0]),
        (rungs_avail[-1], ctx.layers17[-1]),
    ]
    # P4a's wave reap deletes uploaded layer tensors locally (sha sidecar =
    # durable on HF), so at P4b entry only star-layer tensors are local; the
    # mid/first/last slice layers here must be staged back before ctx.take.
    gate_needed: dict[str, list[int]] = {}
    for rung, layer in slices[: F1.PARITY_MIN_SLICES]:
        gate_needed.setdefault(f"{rung}/diag0", []).append(layer)
        gate_needed.setdefault(f"{rung}/diag0/ctx", []).append(layer)
    ensure_cells_staged(ctx.out_root, gate_needed)
    report: dict[str, Any] = {
        "tol": F1.PARITY_TOL,
        "gcv_dof_cap": None,
        "note": "solver parity (pure-GCV both sides); production fits pass gcv_dof_cap=0.9",
        "slices": [],
    }
    worst = 0.0
    tr, ev, _ = _fold_masks(ctx, "full", 0)
    for rung, layer in slices[: F1.PARITY_MIN_SLICES]:
        if ev.sum() < 2 or tr.sum() < 2:
            continue
        X = ctx.take(ctx.diag0_ctx_rel(rung, layer), "u_mean", ctx.spine.ids)
        Y = ctx.take(ctx.diag0_w_rel(rung, layer), "w", ctx.spine.ids)
        slow = ridge_fit_predict(X[tr], Y[tr], X[ev])
        fast = _ridge(X[tr][None], Y[tr][None], X[ev][None], device=device, gcv_dof_cap=None)[0]
        denom = max(float(np.abs(slow).max()), 1e-12)
        rel = float(np.abs(fast - slow).max() / denom)
        worst = max(worst, rel)
        report["slices"].append(
            {"rung": rung, "layer": layer, "n_tr": int(tr.sum()), "max_rel": rel}
        )
    report["max_rel_diff"] = worst
    report["pass"] = bool(worst <= F1.PARITY_TOL and len(report["slices"]) >= F1.PARITY_MIN_SLICES)
    path = ctx.eval_dir / "fits" / "parity_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(path, report)
    logger.info("[fits] parity gate: %s", report)
    _reap_staged(
        R._store_root(ctx.out_root),
        {(sub, layer) for sub, layers in gate_needed.items() for layer in layers},
        resident or set(),
    )
    if not report["pass"]:
        if ctx.smoke:
            print("[smoke-downgrade] parity gate informational (verdict recorded)", flush=True)
        else:
            R.designed_halt(ctx.out_root, "ridge_parity", report)
    return report


# ── P4a stage-1 units: diagonal 17-layer sweep ───────────────────────────────


def run_sweep_unit(ctx: FitsCtx, device: str, *, rung: str, fold: int) -> dict:
    """Diagonal 0-shot (rung, fold) fit at EVERY 17-set layer, layer-chunked
    batched ridge; persists per-context SS + cos + IDENTITY SS at the full
    17-layer grain (selection-inherited + D-tilde CI inputs) plus the
    EXTRA-ROW predictions (store rows outside the headline intersection whose
    group-fold == this fold) feeding the §6 trained-only-sensitivity read —
    fit-free at finalize because the maps score them HERE, OOF by group."""
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    ids = ctx.spine.ids
    tr, ev, _ = _fold_masks(ctx, "full", fold)
    unit = f"sweep_{rung}_f{fold}"
    if ev.sum() < 2 or tr.sum() < 2:
        ctx.write_unit(unit, {"skipped": True, "n_tr": int(tr.sum()), "n_ev": int(ev.sum())})
        logger.warning("[fits] %s SKIPPED (fold gate)", unit)
        return {}
    n_tr, n_ev = int(tr.sum()), int(ev.sum())
    # Extra rows: diag0 store rows not in the headline intersection, scored
    # exactly once program-wide (under their group's fold; hash fallback for
    # groups outside the manifest assignment).
    ref_rel = ctx.diag0_ctx_rel(rung, ctx.layers17[0])
    store_ids = [str(x) for x in ctx.cache.subdir(ref_rel)["__row_ids__"]]
    extra_ids = [
        i
        for i in store_ids
        if i not in ctx.spine.pos
        and i in ctx.spine.meta
        and ctx.spine.fold_of_group(ctx.spine.meta[i]["group"]) == fold
    ]
    t0 = time.time()
    chunk_cap = F1._layer_chunk_cap(device, n_tr)
    L = len(ctx.layers17)
    res_all = np.zeros((L, n_ev))
    tot_all = np.zeros_like(res_all)
    cos_all = np.zeros_like(res_all)
    resid_all = np.zeros_like(res_all)  # identity+bias per-context SS
    ex_res = np.zeros((L, len(extra_ids)))
    ex_tot = np.zeros_like(ex_res)
    ex_res_id = np.zeros_like(ex_res)
    per_layer: dict[str, Any] = {}
    d_in = None
    for c0 in range(0, L, chunk_cap):
        chunk = ctx.layers17[c0 : c0 + chunk_cap]
        Xf = [ctx.take(ctx.diag0_ctx_rel(rung, layer), "u_mean", ids) for layer in chunk]
        Yf = [ctx.take(ctx.diag0_w_rel(rung, layer), "w", ids) for layer in chunk]
        d_in = Xf[0].shape[1]
        Xex = (
            [ctx.take(ctx.diag0_ctx_rel(rung, layer), "u_mean", extra_ids) for layer in chunk]
            if extra_ids
            else None
        )
        Yex = (
            [ctx.take(ctx.diag0_w_rel(rung, layer), "w", extra_ids) for layer in chunk]
            if extra_ids
            else None
        )
        Xtr = np.stack([x[tr] for x in Xf])
        Ytr = np.stack([y[tr] for y in Yf])
        if extra_ids:
            Xev = np.stack([np.concatenate([x[ev], xe], axis=0) for x, xe in zip(Xf, Xex)])
        else:
            Xev = np.stack([x[ev] for x in Xf])
        preds, info = _ridge(Xtr, Ytr, Xev, device=device, return_info=True)
        for li, layer in enumerate(chunk):
            gi = c0 + li
            y_ev = Yf[li][ev]
            y_tr_mean = Ytr[li].mean(0)
            res, tot, cos = F1._per_ctx_ss(preds[li][:n_ev], y_ev, y_tr_mean)
            res_all[gi], tot_all[gi], cos_all[gi] = res, tot, cos
            id_pred = identity_bias_predict(Xtr[li], Ytr[li], Xf[li][ev])
            resid_all[gi] = ((y_ev - id_pred) ** 2).sum(axis=-1)
            if extra_ids:
                rese, tote, _c = F1._per_ctx_ss(preds[li][n_ev:], Yex[li], y_tr_mean)
                ex_res[gi], ex_tot[gi] = rese, tote
                idp_e = identity_bias_predict(Xtr[li], Ytr[li], Xex[li])
                ex_res_id[gi] = ((Yex[li] - idp_e) ** 2).sum(axis=-1)
            per_layer[str(layer)] = {
                "ss_res": float(res.sum()),
                "ss_tot": float(tot.sum()),
                "r2": F1._pooled_r2(float(res.sum()), float(tot.sum())),
                "identity_r2": F1._pooled_r2(float(resid_all[gi].sum()), float(tot.sum())),
                **_fit_record(info, li),
            }
        del Xtr, Ytr, Xev, preds, Xf, Yf, Xex, Yex
    _, percell = ctx.unit_paths()
    F1._savez_atomic(
        percell / f"diag0_{rung}_f{fold}.npz",
        layers=np.asarray(ctx.layers17),
        row_idx=np.flatnonzero(ev),
        ss_res=res_all,
        ss_tot=tot_all,
        cos=cos_all,
        ss_res_id=resid_all,
    )
    if extra_ids:
        F1._savez_atomic(
            percell / f"sweep_extra_{rung}_f{fold}.npz",
            layers=np.asarray(ctx.layers17),
            ids=np.asarray(extra_ids),
            ss_res=ex_res,
            ss_tot=ex_tot,
            ss_res_id=ex_res_id,
        )
    out = {"n_tr": n_tr, "n_ev": n_ev, "n_extra_scored": len(extra_ids), "per_layer": per_layer}
    disc = _ntr_lt_d_disclosure(n_tr, int(d_in))
    if disc:
        out["n_tr_lt_d"] = disc
    ctx.write_unit(unit, out)
    return {"pilot_gate": {"unit": unit, "wall_s": time.time() - t0, "klass": "sweep"}}


# ── P4a selection: layer*, bands, l_FA ───────────────────────────────────────


def select_layers_2544(ctx: FitsCtx) -> dict:
    """layer* = argmax over the 17-set of MEAN diagonal pooled OOF R^2 across
    the rungs (one shared frozen selection; ties -> lower layer). B5 = the
    index-space {star, ±1, ±2} neighbors within the captured set (the plan's
    {±0,±2,±4} offsets for an interior star; edge-clipped — implementer
    discretion per plan deviations-allowed). l_FA = nearest full-attention
    layer; B6 = sorted(B5 ∪ {l_FA})."""
    per_rung_layer_r2: dict[str, dict[str, float]] = {}
    for rung in ctx.rungs:
        agg = {layer: [0.0, 0.0] for layer in ctx.layers17}
        for fold in range(ctx.spine.n_folds):
            rec = ctx.read_unit(f"sweep_{rung}_f{fold}")
            if rec.get("skipped"):
                continue
            for layer in ctx.layers17:
                e = rec["per_layer"][str(layer)]
                agg[layer][0] += e["ss_res"]
                agg[layer][1] += e["ss_tot"]
        per_rung_layer_r2[rung] = {str(layer): F1._pooled_r2(*agg[layer]) for layer in ctx.layers17}
    mean_by_layer = {
        layer: float(np.nanmean([per_rung_layer_r2[rung][str(layer)] for rung in ctx.rungs]))
        for layer in ctx.layers17
    }
    star = max(ctx.layers17, key=lambda layer: (mean_by_layer[layer], -layer))
    si = ctx.layers17.index(star)
    band5 = sorted(ctx.layers17[k] for k in range(max(0, si - 2), min(len(ctx.layers17), si + 3)))
    fa = C2.nearest_full_attention_layer(star)
    max_layer = max(ctx.layers17)
    if fa > max_layer:
        if not ctx.smoke:
            raise RuntimeError(
                f"l_FA={fa} beyond the model's captured range (max {max_layer}) — "
                "production is 32L Olmo-3, so this is store drift"
            )
        print(f"[smoke-downgrade] l_FA {fa} -> {max_layer} (tiny smoke model)", flush=True)
        fa = max_layer
    band6 = sorted(set(band5) | {fa})
    ctx.layer_star, ctx.layer_fa, ctx.band5, ctx.band6 = star, fa, band5, band6
    return {
        "layers": ctx.layers17,
        "per_rung_layer_r2": per_rung_layer_r2,
        "mean_diag_r2_by_layer": {str(k): v for k, v in mean_by_layer.items()},
        "selection": {
            "layer_star": star,
            "band_b5": band5,
            "layer_fa": fa,
            "band_b6": band6,
            "layer_type_split_b6": C2.layer_type_split(band6),
            "rule": (
                "argmax over the 17-set of MEAN diagonal pooled OOF R^2 across rungs "
                "(tie -> lower layer); B5 = index-space {star, ±1, ±2} neighbors "
                "(= {±0,±2,±4} offsets for an interior star, edge-clipped); "
                "l_FA = nearest full-attention layer (tie -> lower)"
            ),
        },
    }


# ── P4a stage-2 units: star reads (shuffle null / kNN / reliability) ─────────


def run_star_unit(ctx: FitsCtx, device: str, *, rung: str, fold: int) -> dict:
    """Diagonal reads AT layer*: shuffled-pairing null (draws batched as
    slices of ONE layer-batched call — Gram identical, correspondence
    destroyed), kNN + identity baselines (headline cell), per-dim SS, and the
    reliability-repeat scoring (seed-43/44 captures vs the SAME predictions —
    the ceiling input, F1 recipe verbatim)."""
    layer = ctx.layer_star
    assert layer is not None
    ids = ctx.spine.ids
    tr, ev, _ = _fold_masks(ctx, "full", fold)
    unit = f"star_{rung}_f{fold}"
    if ev.sum() < 2 or tr.sum() < 2:
        ctx.write_unit(unit, {"skipped": True})
        return {}
    X = ctx.take(ctx.diag0_ctx_rel(rung, layer), "u_mean", ids)
    Y = ctx.take(ctx.diag0_w_rel(rung, layer), "w", ids)
    n_tr = int(tr.sum())
    rng = np.random.default_rng(C.FOLD_SEED + fold * 101 + ctx.rungs.index(rung))
    perms = [rng.permutation(n_tr) for _ in range(ctx.n_shuffle)]
    Xtr = np.stack([X[tr]] * (1 + ctx.n_shuffle))
    Ytr = np.stack([Y[tr]] + [Y[tr][p] for p in perms])
    Xev = np.stack([X[ev]] * (1 + ctx.n_shuffle))
    preds, info = _ridge(Xtr, Ytr, Xev, device=device, return_info=True)
    y_ev = Y[ev]
    y_tr_mean = Y[tr].mean(0)
    res, tot, cos = F1._per_ctx_ss(preds[0], y_ev, y_tr_mean)
    res_dim = ((y_ev - preds[0]) ** 2).sum(axis=0)
    tot_dim = ((y_ev - y_tr_mean) ** 2).sum(axis=0)
    shuffle_r2 = [
        F1._pooled_r2(float(((y_ev - preds[1 + d]) ** 2).sum()), float(tot.sum()))
        for d in range(ctx.n_shuffle)
    ]
    out: dict[str, Any] = {
        "layer_star": layer,
        "n_tr": n_tr,
        "n_ev": int(ev.sum()),
        "r2": F1._pooled_r2(float(res.sum()), float(tot.sum())),
        "shuffle_null_r2": shuffle_r2,
        "shuffle_fit_records": [_fit_record(info, 1 + d) for d in range(ctx.n_shuffle)],
        **_fit_record(info, 0),
        **F1._cell_baselines(X[tr], Y[tr], X[ev], y_ev, preds[0]),
    }
    disc = _ntr_lt_d_disclosure(n_tr, X.shape[1])
    if disc:
        out["n_tr_lt_d"] = disc
    arrays: dict[str, np.ndarray] = {
        "row_idx": np.flatnonzero(ev),
        "ss_res": res,
        "ss_tot": tot,
        "cos": cos,
        "ss_res_dim": res_dim,
        "ss_tot_dim": tot_dim,
    }
    # reliability: preds scored against the seed-43/44 repeat 0-shot captures
    rel: dict[str, Any] = {}
    ev_idx = np.flatnonzero(ev)
    for seed in C.RELIABILITY_SEEDS:
        rel_rel = f"{rung}/rel0_seed{seed}/L{layer}.pt"
        try:
            d = ctx.cache.subdir(rel_rel)
        except FileNotFoundError:
            rel[f"seed{seed}"] = "missing"
            continue
        pos = {str(rid): k for k, rid in enumerate(d["__row_ids__"])}
        keep = [(k, pos[ids[gi]]) for k, gi in enumerate(ev_idx) if ids[gi] in pos]
        if len(keep) < 2:
            rel[f"seed{seed}"] = "too_few"
            continue
        evk = np.asarray([k for k, _ in keep])
        relk = np.asarray([p for _, p in keep])
        res_r = ((d["w"][relk] - preds[0][evk]) ** 2).sum(axis=-1)
        arrays[f"rel_res_seed{seed}"] = res_r
        arrays[f"rel_rows_seed{seed}"] = ev_idx[evk]
        rel[f"seed{seed}"] = {"n": int(len(keep))}
    out["reliability"] = rel
    _, percell = ctx.unit_paths()
    F1._savez_atomic(percell / f"star_{rung}_f{fold}.npz", **arrays)
    ctx.write_unit(unit, out)
    return {}


def gate_b_2544(ctx: FitsCtx) -> dict:
    """Gate B (§7, post-P4a, non-science, observed-side): diagonal Q(main) at
    layer* > 0.15 AND the store-integrity scramble check — ONE-SIDED leak cap
    (scramble R^2 < +0.05; never a two-sided band: the shuffle-refit null is
    strictly negative and deepens toward n_tr ~ d, #1491 class). FAIL routes
    to rig verification (designed halt), never a science verdict."""
    _, percell = ctx.unit_paths()
    q: dict[str, float] = {}
    scramble_max = -np.inf
    scramble_by_rung: dict[str, float] = {}
    for rung in ctx.rungs:
        ss = [0.0, 0.0]
        smax = -np.inf
        for fold in range(ctx.spine.n_folds):
            path = percell / f"star_{rung}_f{fold}.npz"
            if not path.exists():
                continue
            rec = ctx.read_unit(f"star_{rung}_f{fold}")
            if rec.get("skipped"):
                continue
            d = np.load(path)
            ss[0] += float(d["ss_res"].sum())
            ss[1] += float(d["ss_tot"].sum())
            if rec["shuffle_null_r2"]:
                smax = max(smax, max(rec["shuffle_null_r2"]))
        q[rung] = F1._pooled_r2(*ss)
        scramble_by_rung[rung] = float(smax)
        scramble_max = max(scramble_max, smax)
    gate = {
        "q_at_star_by_rung": q,
        "q_main": q.get("main"),
        "floor": GATE_B_FLOOR,
        "scramble_max_r2": float(scramble_max),
        "scramble_leak_cap": SCRAMBLE_LEAK_CAP,
        "scramble_by_rung": scramble_by_rung,
        "advisory_note": (
            "shuffled-pairing null logged ADVISORY (expected mildly negative, "
            "#1902 realized -0.037..-0.069); nothing branches on the null side"
        ),
    }
    if "main" not in ctx.rungs:
        gate["pass"] = None
        gate["note"] = "rung subset without 'main' — Gate B not evaluable (recorded)"
        logger.warning("[fits] gate B skipped: 'main' not in rung subset %s", ctx.rungs)
        return gate
    q_main = q.get("main")
    ok = (
        q_main is not None
        and np.isfinite(q_main)
        and q_main > GATE_B_FLOOR
        and scramble_max < SCRAMBLE_LEAK_CAP
    )
    gate["pass"] = bool(ok)
    if not ok:
        if ctx.smoke:
            print(f"[smoke-downgrade] gate B informational: {gate}", flush=True)
        else:
            R.designed_halt(ctx.out_root, "sanity_gate_b", gate)
    return gate


# ── P4a driver (rung waves; disk-bounded staging) ────────────────────────────


def _sweep_store_relpaths(ctx: FitsCtx, rung: str) -> list[str]:
    return [ctx.diag0_ctx_rel(rung, layer) for layer in ctx.layers17] + [
        ctx.diag0_w_rel(rung, layer) for layer in ctx.layers17
    ]


def _chunks(seq: list, k: int) -> list[list]:
    return [seq[i : i + k] for i in range(0, len(seq), k)]


def run_p4a(args: argparse.Namespace, out_root: Path) -> None:
    t_start = time.time()
    rungs = R2._requested_rungs(args)
    print(f"[phase=fits] stage=P4a rungs={list(rungs)} smoke={bool(args.smoke)}", flush=True)
    ctx = FitsCtx(args, out_root, rungs)
    # Fit-phase floor: 40 GB = ~1.5x the realized ~26 GB worst K-resident
    # staging wave (plan §9:345 recalibration; blocker headroom-floors-vs-plan-s9).
    C2.headroom_floor_gate(out_root, "fits", smoke=bool(args.smoke))
    pool = F1._WorkerPool(ctx)
    k_res = int(os.environ.get("EPM_ISSUE2544_FIT_RESIDENT", "2"))
    waves = _chunks(list(rungs), max(1, k_res))
    parity = None
    piloted = False
    for wi, wave in enumerate(waves):
        needed = {}
        for r in wave:
            needed[f"{r}/diag0"] = list(ctx.layers17)
            needed[f"{r}/diag0/ctx"] = list(ctx.layers17)
        # Side files first (regime/fingerprint inputs); tensors only when the
        # wave has PENDING units — a fully-resumed wave never re-downloads.
        ensure_cells_staged(out_root, needed, tensors=False)
        units = [
            {
                "unit": f"sweep_{rung}_f{fold}",
                "regime": _unit_regime(
                    ctx,
                    {"kind": "sweep", "rung": rung, "fold": fold, "layers": list(ctx.layers17)},
                    _sweep_store_relpaths(ctx, rung),
                    {rung: ctx.pins[rung]},
                ),
                "fn": run_sweep_unit,
                "kw": {"rung": rung, "fold": fold},
            }
            for rung in wave
            for fold in range(ctx.spine.n_folds)
        ]
        if not any(not R.unit_done(out_root, u["unit"], u["regime"]) for u in units):
            logger.info("[fits] wave %d (%s): all sweep units done — tensors not staged", wi, wave)
            continue
        ensure_cells_staged(out_root, needed)
        if parity is None:
            # The whole wave set stays resident: the gate's slices are a
            # subset of it, and the sweep units below still need every layer
            # (the wave's own _reap_staged runs after pool.run).
            parity = parity_gate_2544(
                ctx,
                pool.devices[0],
                wave,
                resident={(sub, layer) for sub, ls in needed.items() for layer in ls},
            )
        if not piloted:
            # In-run P4a-entry pilot: FIRST unit timed alone; abort > 2x the
            # measured 15.3 s grid-class basis (plan §9; smoke informational).
            pool.run(f"sweep-pilot-w{wi}", units[:1])
            timed = [t for t in ctx.pilot_timings if t.get("klass") == "sweep"]
            if timed:
                pilot = {
                    "per_unit_s": timed[0]["wall_s"],
                    "basis_s": P4A_UNIT_BASIS_S,
                    "abort_ratio": PILOT_ABORT_RATIO,
                }
                logger.info("[fits] P4a-entry pilot: %s", pilot)
                if not ctx.smoke and timed[0]["wall_s"] > PILOT_ABORT_RATIO * P4A_UNIT_BASIS_S:
                    R.designed_halt(ctx.out_root, "p4a_pilot_wall", pilot)
            units = units[1:]
            piloted = True
        pool.run(f"sweep-w{wi}", units)
        # Disk-bound: reap the wave's staged tensors (sidecar-guarded); the
        # star stage below restages ONLY the star layer (~7 GB, the §9 row).
        _reap_staged(
            ctx.store,
            {(sub, layer) for sub, ls in needed.items() for layer in ls},
            set(),
        )

    if parity is None:
        # Fully-resumed sweep: the prior process ran + recorded the gate.
        parity_path = ctx.eval_dir / "fits" / "parity_gate.json"
        if not parity_path.exists():
            raise RuntimeError(
                f"sweep fully resumed but no parity record at {parity_path} — "
                "wipe the state dir to force a fresh gated run"
            )
        parity = R._read_json(parity_path)

    selection = select_layers_2544(ctx)
    logger.info("[fits] selection: %s", selection["selection"])

    star_needed: dict[str, list[int]] = {}
    star_soft: set[str] = set()
    for r in rungs:
        star_needed[f"{r}/diag0"] = [ctx.layer_star]
        star_needed[f"{r}/diag0/ctx"] = [ctx.layer_star]
        for seed in C.RELIABILITY_SEEDS:
            star_needed[f"{r}/rel0_seed{seed}"] = [ctx.layer_star]
            star_soft.add(f"{r}/rel0_seed{seed}")
    ensure_cells_staged(out_root, star_needed, soft=frozenset(star_soft))

    star_units = [
        {
            "unit": f"star_{rung}_f{fold}",
            "regime": _unit_regime(
                ctx,
                {
                    "kind": "star",
                    "rung": rung,
                    "fold": fold,
                    "layer_star": ctx.layer_star,
                    "n_shuffle": ctx.n_shuffle,
                },
                [ctx.diag0_ctx_rel(rung, ctx.layer_star), ctx.diag0_w_rel(rung, ctx.layer_star)],
                {rung: ctx.pins[rung]},
            ),
            "fn": run_star_unit,
            "kw": {"rung": rung, "fold": fold},
        }
        for rung in rungs
        for fold in range(ctx.spine.n_folds)
    ]
    pool.run("star", star_units)

    gate_b = gate_b_2544(ctx)

    # Freeze record — Unit A's pass2 contract: EXACTLY these four keys
    # (asserted by issue2544_run._freeze_record), local + HF mirror.
    freeze = {
        "layer_star": int(ctx.layer_star),
        "band_b5": [int(x) for x in ctx.band5],
        "layer_fa": int(ctx.layer_fa),
        "band_b6": [int(x) for x in ctx.band6],
    }
    freeze_path = out_root / "fits" / "layer_freeze.json"
    R._write_json_atomic(freeze_path, freeze)
    C2.write_sha_sidecar(freeze_path)
    R.upload_json_small(freeze_path, f"{C2.EVAL_MIRROR_HF_PATH}/fits/layer_freeze.json")
    eval_freeze = ctx.eval_dir / "fits" / "layer_freeze.json"
    R._write_json_atomic(eval_freeze, {**freeze, "metadata": R2._metadata()})

    sweep_out = {
        "metadata": R2._metadata(),
        "smoke": ctx.smoke,
        **selection,
        "parity_gate": parity,
        "gate_b": gate_b,
        "pilot_timings": ctx.pilot_timings[:8],
        "n_headline": ctx.spine.n,
        "wall_h": round((time.time() - t_start) / 3600.0, 3),
    }
    R._write_json_atomic(ctx.eval_dir / "fits" / "layer_sweep.json", sweep_out)
    _upload_eval_mirror(ctx, ["fits"])
    R2.write_sentinel(
        out_root,
        "sweep",
        {
            "layer_star": int(ctx.layer_star),
            "band_b6": [int(x) for x in ctx.band6],
            "gate_b_pass": gate_b.get("pass"),
        },
        smoke=ctx.smoke,
    )
    print(
        f"[phase=fits] P4a done: layer_star={ctx.layer_star} band_b6={ctx.band6} "
        f"gate_b={gate_b.get('pass')}",
        flush=True,
    )


# ── P4b cell roster (derived from the CAPTURED pass-2 cells) ─────────────────


def p4b_cell_specs(ctx: FitsCtx, include_lfa0: bool) -> list[dict[str, Any]]:
    """Fit specs derived MECHANICALLY from ``C2.pass2_capture_cells`` (so a fit
    cell can never reference an uncaptured store) + the S1 n-matched companion
    refits (data reuse — no new captures). Fields: name / klass / rung /
    answer_rung / scope / x_key / layers / y_rel_by_layer / x_rel_by_layer."""
    b6 = list(ctx.band6)
    specs: list[dict[str, Any]] = []

    def _own(sub: str, layers: list[int]) -> tuple[dict, dict]:
        return (
            {int(layer): f"{sub}/L{layer}.pt" for layer in layers},
            {int(layer): f"{sub}/ctx/L{layer}.pt" for layer in layers},
        )

    def _diag0(rung: str, layers: list[int]) -> tuple[dict, dict]:
        return (
            {int(layer): ctx.diag0_w_rel(rung, layer) for layer in layers},
            {int(layer): ctx.diag0_ctx_rel(rung, layer) for layer in layers},
        )

    for cell in C2.pass2_capture_cells(tuple(ctx.rungs), include_lfa0=include_lfa0):
        m, name = cell["rung"], cell["cell"]
        if name.startswith("rel4_seed"):
            continue  # ceiling input, not a fit cell
        if name == "lfa0":
            y_rel, x_rel = _own(cell["subdir"], [ctx.layer_fa])
            specs.append(
                {
                    "name": f"lfa0_{m}",
                    "klass": "lfa0",
                    "rung": m,
                    "answer_rung": m,
                    "scope": "full",
                    "x_key": "u_mean",
                    "layers": [int(ctx.layer_fa)],
                    "y_rel_by_layer": y_rel,
                    "x_rel_by_layer": x_rel,
                }
            )
            continue
        if name.startswith(("colC_", "rowR_")):
            klass, scope, x_key = "cross", "full", "u_mean"
            y_rel = {int(layer): f"{cell['subdir']}/L{layer}.pt" for layer in b6}
            _yd, x_rel = _diag0(m, b6)  # ctx = the capturing rung's own 0-shot ctx
        elif name == "diag4" or name.startswith("dose_k"):
            klass, scope, x_key = "kshot", "full", "q_mean"
            y_rel, x_rel = _own(cell["subdir"], b6)
        elif name.startswith(("gen4_o", "gen4_s")):
            klass, scope, x_key = "subset", "robust", "q_mean"
            y_rel, x_rel = _own(cell["subdir"], b6)
        elif name == "natgen":
            klass, scope, x_key = "subset", "natgen", "u_mean"
            y_rel, x_rel = _own(cell["subdir"], b6)
        else:
            raise ValueError(f"unmapped pass-2 cell {name!r} (rung {m})")
        specs.append(
            {
                "name": f"{name}_{m}",
                "klass": klass,
                "rung": m,
                "answer_rung": cell["answer_rung"],
                "scope": scope,
                "x_key": x_key,
                "layers": [int(x) for x in b6],
                "y_rel_by_layer": y_rel,
                "x_rel_by_layer": x_rel,
                "k": cell["k"],
                "order_id": cell["order_id"],
                "set_id": cell["set_id"],
            }
        )
    if not include_lfa0:
        # l_FA ∈ the pass-1 17-set (=31): the 0-shot l_FA diagonal reads the
        # pass-1 diag0 store directly (plan §4 P3b skip clause).
        for m in ctx.rungs:
            y_rel, x_rel = _diag0(m, [ctx.layer_fa])
            specs.append(
                {
                    "name": f"lfa0_{m}",
                    "klass": "lfa0",
                    "rung": m,
                    "answer_rung": m,
                    "scope": "full",
                    "x_key": "u_mean",
                    "layers": [int(ctx.layer_fa)],
                    "y_rel_by_layer": y_rel,
                    "x_rel_by_layer": x_rel,
                }
            )
    # S1 n-matched companions (data reuse; IDENTICAL rows + folds as the
    # subset cells they pair with).
    for m in ctx.rungs:
        y4, x4 = _own(f"{m}/diag4", b6)
        y0, x0 = _diag0(m, b6)
        specs.append(
            {
                "name": f"cmp_o1_6k_{m}",
                "klass": "companion",
                "rung": m,
                "answer_rung": m,
                "scope": "robust",
                "x_key": "q_mean",
                "layers": [int(x) for x in b6],
                "y_rel_by_layer": y4,
                "x_rel_by_layer": x4,
            }
        )
        specs.append(
            {
                "name": f"cmp_z_6k_{m}",
                "klass": "companion",
                "rung": m,
                "answer_rung": m,
                "scope": "robust",
                "x_key": "u_mean",
                "layers": [int(x) for x in b6],
                "y_rel_by_layer": y0,
                "x_rel_by_layer": x0,
            }
        )
        if m in C2.NATIVE_GEN_RUNGS:
            specs.append(
                {
                    "name": f"cmp_plain_2k_{m}",
                    "klass": "companion",
                    "rung": m,
                    "answer_rung": m,
                    "scope": "natgen",
                    "x_key": "u_mean",
                    "layers": [int(x) for x in b6],
                    "y_rel_by_layer": y0,
                    "x_rel_by_layer": x0,
                }
            )
    names = [s["name"] for s in specs]
    assert len(set(names)) == len(names), "duplicate cell-spec names"
    return specs


# ── P4b grid-class unit ──────────────────────────────────────────────────────


def run_cell_unit(ctx: FitsCtx, device: str, *, spec: dict[str, Any], fold: int) -> dict:
    """One (cell, fold) banded ridge fit: layer-batched over the spec's layers,
    identity+kNN baselines per layer (S3), per-context SS at band grain
    (GLOBAL headline row positions), lambda*/dof/selector per fit (S2), the
    n_tr<d disclosure on subset/companion classes (S1), and — on the diag4
    cell — the seed-43/44 repeat scoring at layer* (the Delta-ceiling input)."""
    name = spec["name"]
    unit = f"cell_{name}_f{fold}"
    ids = ctx.spine.scope_ids(spec["scope"])
    tr, ev, gpos = _fold_masks(ctx, spec["scope"], fold)
    if ev.sum() < 2 or tr.sum() < 2:
        ctx.write_unit(unit, {"skipped": True, "n_tr": int(tr.sum()), "n_ev": int(ev.sum())})
        return {}
    layers = [int(x) for x in spec["layers"]]
    t0 = time.time()
    Xf = [ctx.take(spec["x_rel_by_layer"][layer], spec["x_key"], ids) for layer in layers]
    Yf = [ctx.take(spec["y_rel_by_layer"][layer], "w", ids) for layer in layers]
    Xtr = np.stack([x[tr] for x in Xf])
    Ytr = np.stack([y[tr] for y in Yf])
    Xev = np.stack([x[ev] for x in Xf])
    preds, info = _ridge(Xtr, Ytr, Xev, device=device, return_info=True)
    n_ev = int(ev.sum())
    res_all = np.zeros((len(layers), n_ev))
    tot_all = np.zeros_like(res_all)
    cos_all = np.zeros_like(res_all)
    per_layer: dict[str, Any] = {}
    for li, layer in enumerate(layers):
        y_ev = Yf[li][ev]
        res, tot, cos = F1._per_ctx_ss(preds[li], y_ev, Ytr[li].mean(0))
        res_all[li], tot_all[li], cos_all[li] = res, tot, cos
        per_layer[str(layer)] = {
            "ss_res": float(res.sum()),
            "ss_tot": float(tot.sum()),
            "r2": F1._pooled_r2(float(res.sum()), float(tot.sum())),
            **_fit_record(info, li),
            **F1._cell_baselines(Xtr[li], Ytr[li], Xev[li], y_ev, preds[li]),
        }
    out: dict[str, Any] = {
        "name": name,
        "klass": spec["klass"],
        "scope": spec["scope"],
        "x_key": spec["x_key"],
        "layers": layers,
        "layer_type_split": C2.layer_type_split(layers),
        "n_tr": int(tr.sum()),
        "n_ev": n_ev,
        "per_layer": per_layer,
    }
    disc = _ntr_lt_d_disclosure(int(tr.sum()), Xf[0].shape[1])
    if disc:
        out["n_tr_lt_d"] = disc
    arrays: dict[str, np.ndarray] = {
        "layers": np.asarray(layers),
        "row_idx": gpos[ev],
        "ss_res": res_all,
        "ss_tot": tot_all,
        "cos": cos_all,
    }
    # diag4: score the 4-shot reliability repeats at layer* (ceiling input)
    if name.startswith("diag4_") and ctx.layer_star in layers:
        li_star = layers.index(ctx.layer_star)
        ev_global = gpos[ev]
        rel: dict[str, Any] = {}
        for seed in C.RELIABILITY_SEEDS:
            rel_rel = f"{spec['rung']}/rel4_seed{seed}/L{ctx.layer_star}.pt"
            try:
                d = ctx.cache.subdir(rel_rel)
            except FileNotFoundError:
                rel[f"seed{seed}"] = "missing"
                continue
            pos = {str(rid): k for k, rid in enumerate(d["__row_ids__"])}
            keep = [
                (k, pos[ctx.spine.ids[g]])
                for k, g in enumerate(ev_global)
                if ctx.spine.ids[g] in pos
            ]
            if len(keep) < 2:
                rel[f"seed{seed}"] = "too_few"
                continue
            evk = np.asarray([k for k, _ in keep])
            relk = np.asarray([p for _, p in keep])
            res_r = ((d["w"][relk] - preds[li_star][evk]) ** 2).sum(axis=-1)
            arrays[f"rel_res_seed{seed}"] = res_r
            arrays[f"rel_rows_seed{seed}"] = ev_global[evk]
            rel[f"seed{seed}"] = {"n": int(len(keep))}
        out["reliability_4shot"] = rel
    _, percell = ctx.unit_paths()
    F1._savez_atomic(percell / f"cell_{name}_f{fold}.npz", **arrays)
    ctx.write_unit(unit, out)
    return {"pilot_gate": {"unit": unit, "wall_s": time.time() - t0, "klass": "grid"}}


# ── P4b transfer unit ────────────────────────────────────────────────────────


def run_xfer_unit(ctx: FitsCtx, device: str, *, i: str, j: str, fold: int) -> dict:
    """Transfer T(i->j) at layer*, one fold: direct / general-linear /
    orthogonal-Procrustes, alignments fitted on TRAIN folds only, A_ans on
    SAME-answer-text pairs w_i(x, a_main) <-> w_j(x, a_main) from the s=main
    column (§3 — it cannot absorb the answer-distribution change); matched
    nulls N1 (shuffled-correspondence alignment refits, identical capacity) +
    N2 (spectrum-matched random center operator, #825 recipe); S3 baselines on
    EVERY fitted map (A_ctx, A_ans, both Procrustes maps, the composite)."""
    layer = ctx.layer_star
    assert layer is not None
    ids = ctx.spine.ids
    tr, ev, gpos = _fold_masks(ctx, "full", fold)
    unit = f"xfer_{i}__{j}_f{fold}"
    if ev.sum() < 2 or tr.sum() < 2:
        ctx.write_unit(unit, {"skipped": True})
        return {}
    t0 = time.time()
    u_i = ctx.take(ctx.diag0_ctx_rel(i, layer), "u_mean", ids)
    w_ii = ctx.take(ctx.diag0_w_rel(i, layer), "w", ids)
    u_j = ctx.take(ctx.diag0_ctx_rel(j, layer), "u_mean", ids)
    w_jj = ctx.take(ctx.diag0_w_rel(j, layer), "w", ids)
    w_i_am = ctx.take(ctx.col_main_w_rel(i, layer), "w", ids)  # w_i(x, a_main)
    w_j_am = ctx.take(ctx.col_main_w_rel(j, layer), "w", ids)
    n_tr, n_ev = int(tr.sum()), int(ev.sum())
    rng = np.random.default_rng(
        C.FOLD_SEED + 7919 * fold + 13 * ctx.rungs.index(i) + ctx.rungs.index(j)
    )

    # A_ctx (general-linear): u_j -> u_i on train folds; + N1 shuffled refits.
    perms = [rng.permutation(n_tr) for _ in range(ctx.n_null)]
    actx_preds, actx_info = _ridge(
        np.stack([u_j[tr]] * (1 + ctx.n_null)),
        np.stack([u_i[tr]] + [u_i[tr][p] for p in perms]),
        np.stack([u_j[ev]] * (1 + ctx.n_null)),
        device=device,
        return_info=True,
    )
    gl_ev, gl_null_ev = actx_preds[0], actx_preds[1:]
    # A_ctx (orthogonal Procrustes, train-centered)
    R_ctx, mu_j, mu_i = F1._orth_map(u_j[tr], u_i[tr], device)
    orth_ev = (u_j[ev] - mu_j) @ R_ctx + mu_i

    # f_ii applied to EVERY eval input in ONE stacked call.
    stack = np.concatenate([u_j[ev], gl_ev, orth_ev] + list(gl_null_ev), axis=0)
    f_preds, W_ii, f_info = _ridge(
        u_i[tr][None],
        w_ii[tr][None],
        stack[None],
        device=device,
        return_weights=True,
        return_info=True,
    )
    f_preds = f_preds[0]
    f_direct = f_preds[:n_ev]
    f_gl = f_preds[n_ev : 2 * n_ev]
    f_orth = f_preds[2 * n_ev : 3 * n_ev]
    f_nulls = [f_preds[(3 + d) * n_ev : (4 + d) * n_ev] for d in range(ctx.n_null)]

    # N2: spectrum-matched random center operator (retain W_ii's top-r SVs;
    # #825 recipe via the F1 implementation, verbatim).
    import torch

    dev = torch.device(device)
    Wt = torch.as_tensor(W_ii[0], dtype=torch.float64, device=dev)
    S = torch.linalg.svdvals(Wt).clamp(min=0.0)
    csum = torch.cumsum(S**2, 0) / (float((S**2).sum()) + 1e-30)
    r = max(
        1,
        min(
            int(
                torch.searchsorted(
                    csum, torch.tensor(F1.SPECMATCH_ENERGY, dtype=csum.dtype, device=csum.device)
                ).item()
            )
            + 1,
            S.shape[0],
        ),
    )
    xmu, xsd = u_i[tr].mean(0), u_i[tr].std(0) + 1e-9
    ymu = w_ii[tr].mean(0)
    gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
    n2_preds = []
    for _ in range(ctx.n_null):
        A1 = torch.randn(Wt.shape[0], r, dtype=torch.float64, generator=gen).to(dev)
        Q1, R1 = torch.linalg.qr(A1)
        Q1 = Q1 * torch.sign(torch.diagonal(R1))
        A2 = torch.randn(Wt.shape[1], r, dtype=torch.float64, generator=gen).to(dev)
        Q2, R2 = torch.linalg.qr(A2)
        Q2 = Q2 * torch.sign(torch.diagonal(R2))
        W_null = (Q1 * S[:r]) @ Q2.T
        x_std = torch.as_tensor((gl_ev - xmu) / xsd, dtype=torch.float64, device=dev)
        n2_preds.append((x_std @ W_null).cpu().numpy() + ymu)
        del A1, A2, Q1, Q2, W_null

    # A_ans (general-linear, SAME-ANSWER-TEXT): w_i(a_main) -> w_j(a_main) on
    # train folds; applied to [gl-chain | orth-chain | its OWN held-out inputs
    # (the map baseline read) | N2-chains] in one stacked call.
    a_stack = np.concatenate([f_gl, f_orth, w_i_am[ev]] + n2_preds, axis=0)
    a_preds, a_info = _ridge(
        w_i_am[tr][None], w_j_am[tr][None], a_stack[None], device=device, return_info=True
    )
    a_preds = a_preds[0]
    gl_final = a_preds[:n_ev]
    a_ans_own = a_preds[2 * n_ev : 3 * n_ev]  # A_ans on its own held-out inputs
    n2_final = [a_preds[(3 + d) * n_ev : (4 + d) * n_ev] for d in range(ctx.n_null)]
    # Orthogonal answer-side map (fully-orthogonal mode), same-answer-text pairs.
    R_ans, mu_wi_am, mu_wj_am = F1._orth_map(w_i_am[tr], w_j_am[tr], device)
    orth_final = (f_orth - mu_wi_am) @ R_ans + mu_wj_am
    # N1 answer-side refits on the N1 ctx-null chain (batched slices).
    aperm = [rng.permutation(n_tr) for _ in range(ctx.n_null)]
    n1_final = _ridge(
        np.stack([w_i_am[tr]] * ctx.n_null),
        np.stack([w_j_am[tr][p] for p in aperm]),
        np.stack(f_nulls),
        device=device,
    )

    y_tgt = w_jj[ev]
    y_tgt_mean = w_jj[tr].mean(0)
    res_direct, tot, _ = F1._per_ctx_ss(f_direct, y_tgt, y_tgt_mean)
    res_gl, _, _ = F1._per_ctx_ss(gl_final, y_tgt, y_tgt_mean)
    res_orth, _, _ = F1._per_ctx_ss(orth_final, y_tgt, y_tgt_mean)
    n1_r2 = [
        F1._pooled_r2(float(((y_tgt - n1_final[d]) ** 2).sum()), float(tot.sum()))
        for d in range(ctx.n_null)
    ]
    n2_r2 = [
        F1._pooled_r2(float(((y_tgt - n2_final[d]) ** 2).sum()), float(tot.sum()))
        for d in range(ctx.n_null)
    ]
    # S3: baselines on EVERY fitted map (not only the composite).
    baselines = {
        "composite_gl": F1._cell_baselines(u_j[tr], w_jj[tr], u_j[ev], y_tgt, gl_final),
        "a_ctx_gl": F1._cell_baselines(u_j[tr], u_i[tr], u_j[ev], u_i[ev], gl_ev),
        "a_ctx_procrustes": F1._cell_baselines(u_j[tr], u_i[tr], u_j[ev], u_i[ev], orth_ev),
        "a_ans_gl": F1._cell_baselines(w_i_am[tr], w_j_am[tr], w_i_am[ev], w_j_am[ev], a_ans_own),
        "a_ans_procrustes": F1._cell_baselines(
            w_i_am[tr],
            w_j_am[tr],
            w_i_am[ev],
            w_j_am[ev],
            (w_i_am[ev] - mu_wi_am) @ R_ans + mu_wj_am,
        ),
    }
    out = {
        "layer_star": layer,
        "n_tr": n_tr,
        "n_ev": n_ev,
        "fit_records": {
            "f_center": _fit_record(f_info, 0),
            "a_ctx": _fit_record(actx_info, 0),
            "a_ctx_nulls": [_fit_record(actx_info, 1 + d) for d in range(ctx.n_null)],
            "a_ans": _fit_record(a_info, 0),
        },
        "specmatch_rank": int(r),
        "r2": {
            "direct": F1._pooled_r2(float(res_direct.sum()), float(tot.sum())),
            "gl": F1._pooled_r2(float(res_gl.sum()), float(tot.sum())),
            "orth": F1._pooled_r2(float(res_orth.sum()), float(tot.sum())),
        },
        "null_r2": {"shuffled_correspondence": n1_r2, "spectrum_matched": n2_r2},
        "baselines": baselines,
    }
    _, percell = ctx.unit_paths()
    F1._savez_atomic(
        percell / f"xfer_{i}__{j}_f{fold}.npz",
        row_idx=gpos[ev],
        ss_res_direct=res_direct,
        ss_res_gl=res_gl,
        ss_res_orth=res_orth,
        ss_tot=tot,
    )
    ctx.write_unit(unit, out)
    return {"pilot_gate": {"unit": unit, "wall_s": time.time() - t0, "klass": "xfer"}}


def run_operator_unit(ctx: FitsCtx, device: str, *, i: str, j: str) -> dict:
    """Per-pair operator read at layer*: direction-aware Procrustes-aligned
    operator cosine vs the batched rotation null (#825 `_procrustes_cosine_null`,
    50 draws — issue1345 convention) + rotation-invariant spectrum cosine +
    effective ranks (full-data descriptive weights, #1332 convention)."""
    import torch

    from issue825_map_alignment import _procrustes_cosine_null

    layer = ctx.layer_star
    assert layer is not None
    ids = ctx.spine.ids
    u_i = ctx.take(ctx.diag0_ctx_rel(i, layer), "u_mean", ids)
    w_ii = ctx.take(ctx.diag0_w_rel(i, layer), "w", ids)
    u_j = ctx.take(ctx.diag0_ctx_rel(j, layer), "u_mean", ids)
    w_jj = ctx.take(ctx.diag0_w_rel(j, layer), "w", ids)
    dev = torch.device(device)
    proc = _procrustes_cosine_null(
        torch.as_tensor(u_i, dtype=torch.float64, device=dev),
        torch.as_tensor(u_j, dtype=torch.float64, device=dev),
        torch.as_tensor(w_ii, dtype=torch.float64, device=dev),
        torch.as_tensor(w_jj, dtype=torch.float64, device=dev),
        n_draws=ctx.n_rot,
        seed=C.FOLD_SEED,
    )
    Wi = torch.as_tensor(ctx.rung_weights(i, device), dtype=torch.float64)
    Wj = torch.as_tensor(ctx.rung_weights(j, device), dtype=torch.float64)
    s_i = torch.linalg.svdvals(Wi)
    s_j = torch.linalg.svdvals(Wj)
    spec_cos = float((s_i @ s_j) / (torch.linalg.norm(s_i) * torch.linalg.norm(s_j) + 1e-12))

    def _er(s: torch.Tensor) -> float:
        s2 = s.clamp(min=0.0) ** 2
        return float((s.sum() ** 2) / (s2.sum() + 1e-30))

    ctx.write_unit(
        f"operator_{i}__{j}",
        {
            "layer_star": layer,
            "n": ctx.spine.n,
            "convention": (
                "direction-aware = activation-fitted Procrustes-aligned operator cosine vs "
                f"{ctx.n_rot}-draw random-rotation null (issue1345/#825 conventions); "
                "spectrum cosine = sorted-singular-value cosine, ROTATION-INVARIANT "
                "DESCRIPTIVE ONLY; weights standardized-input-space primal (#1332)"
            ),
            "procrustes_aligned": proc,
            "spectrum_cosine": spec_cos,
            "er_W_i": _er(s_i),
            "er_W_j": _er(s_j),
        },
    )
    return {}


# ── shard assembly + bootstrap machinery ─────────────────────────────────────


def _assemble(ctx: FitsCtx, pattern: str, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Per-context arrays from fold shards over the GLOBAL headline row space
    (row_idx = headline positions). Unfitted rows NaN; ``__fitted__`` mask."""
    _, percell = ctx.unit_paths()
    shards = sorted(percell.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"no percell shards match {pattern} under {percell}")
    n = ctx.spine.n
    out: dict[str, np.ndarray] = {}
    fitted = np.zeros(n, bool)
    for sp in shards:
        d = np.load(sp)
        rows = d["row_idx"]
        for key in keys:
            arr = d[key]
            if key not in out:
                out[key] = np.full((*arr.shape[:-1], n), np.nan)
            out[key][..., rows] = arr
        fitted[rows] = True
    out["__fitted__"] = fitted
    return out


def _boot_counts_2544(ctx: FitsCtx) -> np.ndarray:
    rng = np.random.default_rng(BOOT_SEED)
    return F1._boot_counts(rng, ctx.spine.n_groups, ctx.n_boot)


def _r2_draws(ctx: FitsCtx, counts: np.ndarray, res: np.ndarray, tot: np.ndarray) -> np.ndarray:
    """Cluster-grouped bootstrap R^2 draws from per-context SS (batched
    masked-sum GEMMs; NaN rows contribute nothing — the pooled-OOF convention)."""
    res_g = F1._group_sums(res, ctx.spine.gid, ctx.spine.n_groups)
    tot_g = F1._group_sums(tot, ctx.spine.gid, ctx.spine.n_groups)
    return F1._boot_r2(counts, res_g, tot_g)


def _sb_ceiling(a: list[float], b: list[float]) -> dict[str, Any]:
    """Split-half r over per-context residuals -> Spearman-Brown -> ceiling
    (the #1902 context-ALIGNED-halves recipe verbatim)."""
    if len(a) < 3:
        return {"n_contexts": len(a), "note": "too few paired reliability contexts"}
    r = float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])
    sb = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")
    return {
        "n_contexts": len(a),
        "split_half_r": r,
        "spearman_brown": sb,
        "ceiling_sqrt_ryy": float(np.sqrt(sb)) if np.isfinite(sb) and sb > 0 else None,
    }


def _rel_pairs(ctx: FitsCtx, pattern: str) -> dict[str, Any]:
    """Join per-context reliability residuals across the two repeat seeds from
    percell shards, aligned on shared GLOBAL rows."""
    _, percell = ctx.unit_paths()
    a: dict[int, float] = {}
    b: dict[int, float] = {}
    for sp in sorted(percell.glob(pattern)):
        d = np.load(sp)
        if "rel_res_seed43" in d and "rel_res_seed44" in d:
            for row, v in zip(d["rel_rows_seed43"], d["rel_res_seed43"]):
                a[int(row)] = float(v)
            for row, v in zip(d["rel_rows_seed44"], d["rel_res_seed44"]):
                b[int(row)] = float(v)
    shared = sorted(set(a) & set(b))
    return {"rows": shared, "seed43": [a[r] for r in shared], "seed44": [b[r] for r in shared]}


# ── finalize: the registered output JSONs ────────────────────────────────────


def _pretrain_rungs(ctx: FitsCtx) -> list[str]:
    return [r for r in ctx.rungs if r not in POST_RUNGS]


def _lattice_label(
    dtilde_draws: dict[str, np.ndarray],
    dtilde_point: dict[str, float],
    pretrain: list[str],
    floor: float,
) -> dict[str, Any]:
    """The §3 verdict lattice at one formation floor (rung-binned T_half)."""
    if "r0" not in dtilde_point or "main" not in dtilde_point:
        return {"floor": floor, "label": "not-evaluable", "note": "rung subset lacks r0/main"}
    d_form_draws = dtilde_draws["main"] - dtilde_draws["r0"]
    d_form = dtilde_point["main"] - dtilde_point["r0"]
    upper = float(np.nanquantile(d_form_draws, 0.975))
    thresh = 0.5 * max(d_form, floor)
    crossing = None
    for rung in pretrain:
        if dtilde_point[rung] - dtilde_point["r0"] >= thresh:
            crossing = rung
            break
    # per-draw crossing mass (selection rides inside each draw)
    n_draws = d_form_draws.shape[0]
    mass = dict.fromkeys(pretrain, 0.0)
    uncrossed = 0
    for k in range(n_draws):
        th_k = 0.5 * max(float(d_form_draws[k]), floor)
        hit = None
        for rung in pretrain:
            if float(dtilde_draws[rung][k] - dtilde_draws["r0"][k]) >= th_k:
                hit = rung
                break
        if hit is None:
            uncrossed += 1
        else:
            mass[hit] += 1.0 / n_draws
    ei = pretrain.index(EARLY_MAX_RUNG) if EARLY_MAX_RUNG in pretrain else -1
    ii = pretrain.index(INTERMEDIATE_MAX_RUNG) if INTERMEDIATE_MAX_RUNG in pretrain else -1
    if upper < floor:
        label = "No-formation"
    elif crossing is None:
        label = "Unresolved"
    elif ei >= 0 and pretrain.index(crossing) <= ei:
        label = "Early"
    elif ii >= 0 and pretrain.index(crossing) <= ii:
        label = "Intermediate"
    else:
        label = "Late"
    return {
        "floor": floor,
        "label": label,
        "delta_form": d_form,
        "delta_form_ci": F1._ci(d_form_draws),
        "delta_form_upper95": upper,
        "t_half_rung": crossing,
        "t_half_rung_mass": mass,
        "uncrossed_draw_frac": uncrossed / max(n_draws, 1),
    }


def _trained_only_reads(ctx: FitsCtx) -> dict[str, Any]:
    """§6 scope-caveat (ii): the trained-rungs-only-intersection sensitivity
    re-read + D(main) on its own full unflagged row set. Fit-free: headline
    rows read from the sweep SS; non-headline rows from the sweep_extra
    shards (scored OOF-by-group during the sweep). Flags recomputed from the
    persisted rollouts (the pass-1 filter definition, R2._joint_intersection)."""
    li = ctx.layers17.index(ctx.layer_star)
    r3_idx = C2.RUNGS.index(TRAINED_ONLY_MIN_RUNG)
    suffix = [r for r in ctx.rungs if C2.RUNGS.index(r) >= r3_idx]
    if not suffix:
        return {"status": "unavailable — no rungs at/after r3 in this run"}
    try:
        flags: dict[tuple[str, str], set[str]] = {}
        for rung in suffix:
            for arm in ("gen0", "gen4"):
                recs = R2.fetch_rollout(ctx.out_root, rung, arm)
                flags[(rung, arm)] = {
                    rec["id"] for rec in recs if rec["truncated"] or rec["repetition_flag"]
                }
        all_ids = [r["id"] for r in ctx.spine.rows]
        tos_ids = set(R2._joint_intersection(flags, all_ids, tuple(suffix)))
        main_ok = set(all_ids) - flags[("main", "gen0")] if ("main", "gen0") in flags else None
    except Exception as e:  # noqa: BLE001 — sensitivity read degrades loudly, never silently
        logger.warning("[fits] trained-only sensitivity read unavailable: %s", e)
        return {"status": f"unavailable — {type(e).__name__}: {e}"}

    _, percell = ctx.unit_paths()

    def _pooled_over(id_set: set[str], rung: str) -> dict[str, Any]:
        ss = [0.0, 0.0]
        ss_id = 0.0
        n_used = 0
        mask = np.asarray([i in id_set for i in ctx.spine.ids])
        for fold in range(ctx.spine.n_folds):
            p = percell / f"diag0_{rung}_f{fold}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            keep = mask[d["row_idx"]]
            ss[0] += float(d["ss_res"][li][keep].sum())
            ss[1] += float(d["ss_tot"][li][keep].sum())
            ss_id += float(d["ss_res_id"][li][keep].sum())
            n_used += int(keep.sum())
            pe = percell / f"sweep_extra_{rung}_f{fold}.npz"
            if pe.exists():
                de = np.load(pe)
                ekeep = np.asarray([str(x) in id_set for x in de["ids"]])
                ss[0] += float(de["ss_res"][li][ekeep].sum())
                ss[1] += float(de["ss_tot"][li][ekeep].sum())
                ss_id += float(de["ss_res_id"][li][ekeep].sum())
                n_used += int(ekeep.sum())
        return {
            "r2": F1._pooled_r2(*ss),
            "identity_r2": F1._pooled_r2(ss_id, ss[1]),
            "n_rows": n_used,
        }

    out: dict[str, Any] = {
        "trained_only_intersection": {
            "suffix_rungs": suffix,
            "n_ids": len(tos_ids),
            "per_rung": {rung: _pooled_over(tos_ids, rung) for rung in ctx.rungs},
            "note": (
                "point estimates at layer*, OOF-by-group (non-headline rows scored once "
                "under their group's fold; hash-fallback folds for unassigned groups); "
                "denominator shift vs the headline curve is the registered scope read"
            ),
        }
    }
    if main_ok is not None:
        out["d_main_full_unflagged"] = {"n_ids": len(main_ok), **_pooled_over(main_ok, "main")}
    return out


def _ww_masks(ctx: FitsCtx) -> dict[str, np.ndarray]:
    """Within-window masks per rung: rows whose FULL 4-shot sequence
    (prompt + answer tokens, diag4 row_index) <= the sliding window."""
    masks: dict[str, np.ndarray] = {}
    for rung in ctx.rungs:
        idx_path = ctx.store / f"{rung}/diag4/row_index.jsonl"
        if not idx_path.exists():
            continue
        by_id = {
            r["id"]: (r["n_prompt_tokens"] + r["n_answer_tokens"]) for r in R._read_jsonl(idx_path)
        }
        masks[rung] = np.asarray(
            [by_id.get(i, 10**9) <= WW_WINDOW for i in ctx.spine.ids], dtype=bool
        )
    return masks


def _over_window_fracs(ctx: FitsCtx) -> dict[str, Any]:
    """Analysis-grain over-window fractions per (rung, k) from the captured
    row_index token counts (operationalization: prompt+answer length > window;
    the gen-grain diagnostics ride the intersection manifest)."""
    out: dict[str, Any] = {}
    for rung in ctx.rungs:
        for cell, k in (("diag4", 4), ("dose_k1", 1), ("dose_k16", 16)):
            idx_path = ctx.store / f"{rung}/{cell}/row_index.jsonl"
            if not idx_path.exists():
                continue
            lens = [r["n_prompt_tokens"] + r["n_answer_tokens"] for r in R._read_jsonl(idx_path)]
            out[f"{rung}_k{k}"] = {
                "frac_over_window": float(np.mean([x > WW_WINDOW for x in lens])),
                "n_rows": len(lens),
                "len_p50": float(np.percentile(lens, 50)),
                "len_p90": float(np.percentile(lens, 90)),
            }
    return out


def finalize_p4b(ctx: FitsCtx, wall_h: float) -> None:
    counts = _boot_counts_2544(ctx)
    li_star = ctx.layers17.index(ctx.layer_star)
    eval_fits = ctx.eval_dir / "fits"
    eval_xfer = ctx.eval_dir / "transfer"
    eval_fits.mkdir(parents=True, exist_ok=True)
    eval_xfer.mkdir(parents=True, exist_ok=True)
    b6 = list(ctx.band6)
    bi_star = b6.index(ctx.layer_star)
    bi_fa = b6.index(ctx.layer_fa)

    # ---- diagonal assembly (17-layer sweep percell from P4a) -----------------
    sweep: dict[str, dict[str, np.ndarray]] = {
        rung: _assemble(ctx, f"diag0_{rung}_f*.npz", ("ss_res", "ss_tot", "ss_res_id"))
        for rung in ctx.rungs
    }

    def _cellarr(name: str) -> dict[str, np.ndarray] | None:
        try:
            return _assemble(ctx, f"cell_{name}_f*.npz", ("ss_res", "ss_tot"))
        except FileNotFoundError:
            return None

    # ---- layer_null_matrix + diag curve --------------------------------------
    r2_mats = {
        rung: _r2_draws(ctx, counts, sweep[rung]["ss_res"], sweep[rung]["ss_tot"])
        for rung in ctx.rungs
    }
    id_mats = {
        rung: _r2_draws(ctx, counts, sweep[rung]["ss_res_id"], sweep[rung]["ss_tot"])
        for rung in ctx.rungs
    }

    def _argmax_finite(mat: np.ndarray) -> np.ndarray:
        """Row-wise argmax ignoring NaN; 0 for all-NaN rows (nanargmax raises)."""
        safe = np.where(np.isfinite(mat), mat, -np.inf)
        return np.argmax(safe, axis=1)

    # Selection-inherited re-selection = the REGISTERED shared rule re-run
    # inside each draw (plan:150 + :237(a)): argmax over layers of the MEAN
    # pooled R^2 ACROSS RUNGS — one shared layer per draw applied to EVERY
    # rung, mirroring the frozen layer* selection. The per-rung per-draw
    # argmax (:237(b)'s fairness-control read) stays in best_layer_idx below,
    # a DIFFERENT registered quantity — never the headline companion CI.
    r2_stack = np.stack([r2_mats[r] for r in ctx.rungs])  # (n_rungs, n_draws, 17)
    finite = np.isfinite(r2_stack)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_over_rungs = np.where(finite, r2_stack, 0.0).sum(axis=0) / finite.sum(axis=0)
    shared_sel_idx = _argmax_finite(mean_over_rungs)  # (n_draws,)
    _draw_ix = np.arange(shared_sel_idx.size)

    F1._savez_atomic(
        eval_fits / "layer_null_matrix.npz",
        rungs=np.asarray(ctx.rungs),
        layers=np.asarray(ctx.layers17),
        r2_draws=np.stack([r2_mats[r] for r in ctx.rungs]),  # (n_rungs, n_draws, 17)
        identity_r2_draws=np.stack([id_mats[r] for r in ctx.rungs]),
        best_layer_idx=np.stack([_argmax_finite(r2_mats[r]) for r in ctx.rungs]),
        shared_sel_layer_idx=shared_sel_idx,  # the per-draw SHARED selection
    )

    dtilde_draws: dict[str, np.ndarray] = {}
    dtilde_point: dict[str, float] = {}
    per_rung: dict[str, Any] = {}
    classes = np.asarray(ctx.spine.classes)
    rng_c = np.random.default_rng(BOOT_SEED + 7)
    for rung in ctx.rungs:
        res, tot = sweep[rung]["ss_res"], sweep[rung]["ss_tot"]
        r2_point = F1._pooled_r2(float(np.nansum(res[li_star])), float(np.nansum(tot[li_star])))
        id_point = F1._pooled_r2(
            float(np.nansum(sweep[rung]["ss_res_id"][li_star])), float(np.nansum(tot[li_star]))
        )
        draws_star = r2_mats[rung][:, li_star]
        id_star = id_mats[rung][:, li_star]
        # selection-inherited: the SHARED layer selection re-run per draw
        # (argmax of mean-over-rungs — computed once above), identity
        # subtracted at the SAME selected layer (plan:150 + :237(a)).
        sel_draws = r2_mats[rung][_draw_ix, shared_sel_idx]
        sel_id = id_mats[rung][_draw_ix, shared_sel_idx]
        dtilde_draws[rung] = draws_star - id_star
        dtilde_point[rung] = r2_point - id_point
        star_recs = [
            ctx.read_unit(f"star_{rung}_f{f}")
            for f in range(ctx.spine.n_folds)
            if (ctx.eval_dir / "fits" / "units" / f"star_{rung}_f{f}.json").exists()
        ]
        star_ok = [rec for rec in star_recs if not rec.get("skipped")]
        shuffles = [v for rec in star_ok for v in rec.get("shuffle_null_r2", [])]
        knn_acc1 = {
            metric: (
                float(np.nanmean([rec["knn"][metric]["acc_at_k"]["1"] for rec in star_ok]))
                if star_ok
                else None
            )
            for metric in ("euclidean", "cosine")
        }
        rel = _rel_pairs(ctx, f"star_{rung}_f*.npz")
        per_rung[rung] = {
            "r2_star": r2_point,
            "ci_frozen": F1._ci(draws_star),
            "ci_selection_inherited": F1._ci(sel_draws),
            "identity_r2_star": id_point,
            "identity_ci": F1._ci(id_star),
            "dtilde": dtilde_point[rung],
            "dtilde_ci": F1._ci(dtilde_draws[rung]),
            "dtilde_ci_selection_inherited": F1._ci(sel_draws - sel_id),
            "shuffle_null_r2": {
                "values": shuffles,
                "mean": float(np.mean(shuffles)) if shuffles else None,
                "max": float(np.max(shuffles)) if shuffles else None,
            },
            "knn_acc_at_1": knn_acc1,
            "ceiling_0shot": _sb_ceiling(rel["seed43"], rel["seed44"]),
            "lambda_star_by_fold": [rec.get("lambda_star") for rec in star_ok],
            "dof_by_fold": [rec.get("dof") for rec in star_ok],
            "selector": SELECTOR_TAG,
            "n_rows_fitted": int(sweep[rung]["__fitted__"].sum()),
        }
        # per-class re-reduction (group bootstrap when the class spans >=2
        # groups, else within-stratum context-level bootstrap — §6). Classwise
        # identity SS rides the SAME masks + draw weights, so the per-class
        # dtilde CI is PAIRED per draw (plan:262 raw + baseline-subtracted,
        # pooled + per-class; blocker per-class-dtilde-missing).
        per_class: dict[str, Any] = {}
        res_id = sweep[rung]["ss_res_id"]
        for cls in sorted(set(ctx.spine.classes)):
            cmask = classes == cls
            resc = np.where(cmask, res[li_star], np.nan)
            totc = np.where(cmask, tot[li_star], np.nan)
            resc_id = np.where(cmask, res_id[li_star], np.nan)
            point = F1._pooled_r2(float(np.nansum(resc)), float(np.nansum(totc)))
            id_point_c = F1._pooled_r2(float(np.nansum(resc_id)), float(np.nansum(totc)))
            gset = {int(ctx.spine.gid[k]) for k in np.flatnonzero(cmask)}
            if len(gset) >= 2:
                draws_c = _r2_draws(ctx, counts, resc, totc)
                id_draws_c = _r2_draws(ctx, counts, resc_id, totc)
                boot_kind = "cluster-grouped"
            else:
                rows = np.flatnonzero(cmask & np.isfinite(res[li_star]))
                if rows.size < 3:
                    per_class[cls] = {
                        "r2": point,
                        "identity_r2": id_point_c,
                        "dtilde": point - id_point_c,
                        "n_rows": int(rows.size),
                        "ci": None,
                        "identity_ci": None,
                        "dtilde_ci": None,
                    }
                    continue
                w = rng_c.multinomial(
                    rows.size, np.full(rows.size, 1.0 / rows.size), size=ctx.n_boot
                ).astype(np.float64)
                # ONE weight draw serves r2 AND identity — paired dtilde draws.
                draws_c = F1._boot_r2(w, res[li_star][rows], tot[li_star][rows])
                id_draws_c = F1._boot_r2(w, res_id[li_star][rows], tot[li_star][rows])
                boot_kind = "within-stratum context-level"
            per_class[cls] = {
                "r2": point,
                "ci": F1._ci(draws_c),
                "identity_r2": id_point_c,
                "identity_ci": F1._ci(id_draws_c),
                "dtilde": point - id_point_c,
                "dtilde_ci": F1._ci(draws_c - id_draws_c),
                "n_rows": int(cmask.sum()),
                "bootstrap": boot_kind,
            }
        per_rung[rung]["per_class"] = per_class

    pretrain = _pretrain_rungs(ctx)
    diag_out = {
        "metadata": R2._metadata(),
        "smoke": ctx.smoke,
        "layer_star": ctx.layer_star,
        "selector": SELECTOR_TAG,
        "selection_inherited_rule": (
            "shared argmax over layers of MEAN pooled R^2 across rungs, re-run "
            "inside each draw (plan:150 + :237(a)); per-draw indices persisted "
            "as shared_sel_layer_idx in layer_null_matrix.npz"
        ),
        "n_boot": ctx.n_boot,
        "rung_order": list(ctx.rungs),
        "pretrain_rungs": pretrain,
        "per_rung": per_rung,
        "floor_sensitivity": {
            str(floor): _lattice_label(dtilde_draws, dtilde_point, pretrain, floor)
            for floor in FORMATION_FLOORS
        },
        "scope_caveat": (
            "formation on the shared-intersection (random-init-surviving) context "
            "population — see trained_only_sensitivity for the denominator shift"
        ),
        "trained_only_sensitivity": _trained_only_reads(ctx),
    }
    R._write_json_atomic(eval_fits / "diag_curve.json", diag_out)

    # ---- cross cells ----------------------------------------------------------
    cross: dict[str, Any] = {"colC": {}, "rowR": {}, "diag_ref": {}}
    for rung in ctx.rungs:
        cross["diag_ref"][rung] = {
            "r2_star": per_rung[rung]["r2_star"],
            "ci": per_rung[rung]["ci_frozen"],
        }
        if rung == "main":
            specs = [("rowR", f"rowR_{s}_main", s) for s in ctx.rungs if s != "main"]
        else:
            specs = [("colC", f"colC_main_{rung}", rung)]
        for kind, name, key in specs:
            arr = _cellarr(name)
            if arr is None:
                continue
            draws = _r2_draws(ctx, counts, arr["ss_res"][bi_star], arr["ss_tot"][bi_star])
            recs = [
                ctx.read_unit(f"cell_{name}_f{f}")
                for f in range(ctx.spine.n_folds)
                if (ctx.eval_dir / "fits" / "units" / f"cell_{name}_f{f}.json").exists()
            ]
            ok = [rec for rec in recs if not rec.get("skipped")]
            cross[kind][key] = {
                "r2_star": F1._pooled_r2(
                    float(np.nansum(arr["ss_res"][bi_star])),
                    float(np.nansum(arr["ss_tot"][bi_star])),
                ),
                "ci": F1._ci(draws),
                "identity_r2_star": (
                    float(
                        np.nanmean(
                            [rec["per_layer"][str(ctx.layer_star)]["identity_r2"] for rec in ok]
                        )
                    )
                    if ok
                    else None
                ),
                "knn_acc_at_1_star": {
                    metric: (
                        float(
                            np.nanmean(
                                [
                                    rec["per_layer"][str(ctx.layer_star)]["knn"][metric][
                                        "acc_at_k"
                                    ]["1"]
                                    for rec in ok
                                ]
                            )
                        )
                        if ok
                        else None
                    )
                    for metric in ("euclidean", "cosine")
                },
                "per_band_r2": {
                    str(layer): F1._pooled_r2(
                        float(np.nansum(arr["ss_res"][b6.index(layer)])),
                        float(np.nansum(arr["ss_tot"][b6.index(layer)])),
                    )
                    for layer in b6
                },
                "layer_type_split_b6": C2.layer_type_split(b6),
            }
    cross_out = {
        "metadata": R2._metadata(),
        "smoke": ctx.smoke,
        "layer_star": ctx.layer_star,
        "band_b6": b6,
        "cells": cross,
        "n_boot": ctx.n_boot,
    }
    R._write_json_atomic(eval_fits / "cross_cells.json", cross_out)

    # ---- k-shot curve ----------------------------------------------------------
    ww = _ww_masks(ctx)
    kshot: dict[str, Any] = {}
    delta_draws_by_rung: dict[str, np.ndarray] = {}
    delta_point_by_rung: dict[str, float] = {}
    for rung in ctx.rungs:
        arr4 = _cellarr(f"diag4_{rung}")
        if arr4 is None:
            continue
        res0, tot0 = sweep[rung]["ss_res"][li_star], sweep[rung]["ss_tot"][li_star]
        res4, tot4 = arr4["ss_res"][bi_star], arr4["ss_tot"][bi_star]
        d0 = _r2_draws(ctx, counts, res0, tot0)
        d4 = _r2_draws(ctx, counts, res4, tot4)
        entry: dict[str, Any] = {
            "r2_0shot": F1._pooled_r2(float(np.nansum(res0)), float(np.nansum(tot0))),
            "r2_4shot": F1._pooled_r2(float(np.nansum(res4)), float(np.nansum(tot4))),
            "delta_ci": F1._ci(d4 - d0),
        }
        entry["delta"] = entry["r2_4shot"] - entry["r2_0shot"]
        # Paired per-rung Delta draws feed the Delta_peak - Delta(main)
        # selection-inherited contrast below (plan:64/:237(c)/§4 P4b item 6).
        delta_draws_by_rung[rung] = d4 - d0
        delta_point_by_rung[rung] = entry["delta"]
        # Delta_ww: within-window paired read (free re-aggregation)
        if rung in ww:
            m = ww[rung]
            mres0, mtot0 = np.where(m, res0, np.nan), np.where(m, tot0, np.nan)
            mres4, mtot4 = np.where(m, res4, np.nan), np.where(m, tot4, np.nan)
            entry["delta_ww"] = {
                "delta": F1._pooled_r2(float(np.nansum(mres4)), float(np.nansum(mtot4)))
                - F1._pooled_r2(float(np.nansum(mres0)), float(np.nansum(mtot0))),
                "ci": F1._ci(
                    _r2_draws(ctx, counts, mres4, mtot4) - _r2_draws(ctx, counts, mres0, mtot0)
                ),
                "n_ww_rows": int(m.sum()),
                "n_over_window_rows": int((~m).sum()),
            }
        # Delta_FA: the paired read at the full-attention companion layer
        arr_l0 = _cellarr(f"lfa0_{rung}")
        if arr_l0 is not None:
            res0f, tot0f = arr_l0["ss_res"][0], arr_l0["ss_tot"][0]
            res4f, tot4f = arr4["ss_res"][bi_fa], arr4["ss_tot"][bi_fa]
            entry["delta_fa"] = {
                "layer_fa": ctx.layer_fa,
                "delta": F1._pooled_r2(float(np.nansum(res4f)), float(np.nansum(tot4f)))
                - F1._pooled_r2(float(np.nansum(res0f)), float(np.nansum(tot0f))),
                "ci": F1._ci(
                    _r2_draws(ctx, counts, res4f, tot4f) - _r2_draws(ctx, counts, res0f, tot0f)
                ),
            }
        # dose panel
        dose: dict[str, Any] = {"0": entry["r2_0shot"], "4": entry["r2_4shot"]}
        for k in (1, 16):
            arrk = _cellarr(f"dose_k{k}_{rung}")
            if arrk is not None:
                dose[str(k)] = F1._pooled_r2(
                    float(np.nansum(arrk["ss_res"][bi_star])),
                    float(np.nansum(arrk["ss_tot"][bi_star])),
                )
        entry["dose_panel_r2_at_star"] = dose
        # order/set/native robustness at MATCHED n (S1: subset-vs-subset)
        cmp_o1 = _cellarr(f"cmp_o1_6k_{rung}")
        cmp_z = _cellarr(f"cmp_z_6k_{rung}")
        robust: dict[str, Any] = {}
        for arm in ("gen4_o2", "gen4_o3", "gen4_s2", "gen4_s3"):
            arr_a = _cellarr(f"{arm}_{rung}")
            if arr_a is None or cmp_o1 is None:
                continue
            da = _r2_draws(ctx, counts, arr_a["ss_res"][bi_star], arr_a["ss_tot"][bi_star])
            dc = _r2_draws(ctx, counts, cmp_o1["ss_res"][bi_star], cmp_o1["ss_tot"][bi_star])
            r2_a = F1._pooled_r2(
                float(np.nansum(arr_a["ss_res"][bi_star])),
                float(np.nansum(arr_a["ss_tot"][bi_star])),
            )
            r2_c = F1._pooled_r2(
                float(np.nansum(cmp_o1["ss_res"][bi_star])),
                float(np.nansum(cmp_o1["ss_tot"][bi_star])),
            )
            robust[arm] = {
                "r2": r2_a,
                "companion_r2": r2_c,
                "delta_vs_companion": r2_a - r2_c,
                "delta_ci": F1._ci(da - dc),
                "companion": "cmp_o1_6k (matched rows + folds)",
            }
        if cmp_z is not None and cmp_o1 is not None:
            dz = _r2_draws(ctx, counts, cmp_z["ss_res"][bi_star], cmp_z["ss_tot"][bi_star])
            do1 = _r2_draws(ctx, counts, cmp_o1["ss_res"][bi_star], cmp_o1["ss_tot"][bi_star])
            robust["delta_6k_matched"] = {
                "delta": F1._pooled_r2(
                    float(np.nansum(cmp_o1["ss_res"][bi_star])),
                    float(np.nansum(cmp_o1["ss_tot"][bi_star])),
                )
                - F1._pooled_r2(
                    float(np.nansum(cmp_z["ss_res"][bi_star])),
                    float(np.nansum(cmp_z["ss_tot"][bi_star])),
                ),
                "ci": F1._ci(do1 - dz),
                "note": "Delta at matched 6k-subset n (cmp_o1 - cmp_z)",
            }
        arr_ng = _cellarr(f"natgen_{rung}")
        cmp_pl = _cellarr(f"cmp_plain_2k_{rung}")
        if arr_ng is not None and cmp_pl is not None:
            dn = _r2_draws(ctx, counts, arr_ng["ss_res"][bi_star], arr_ng["ss_tot"][bi_star])
            dp = _r2_draws(ctx, counts, cmp_pl["ss_res"][bi_star], cmp_pl["ss_tot"][bi_star])
            robust["natgen_vs_plain"] = {
                "native_r2": F1._pooled_r2(
                    float(np.nansum(arr_ng["ss_res"][bi_star])),
                    float(np.nansum(arr_ng["ss_tot"][bi_star])),
                ),
                "plain_r2_matched": F1._pooled_r2(
                    float(np.nansum(cmp_pl["ss_res"][bi_star])),
                    float(np.nansum(cmp_pl["ss_tot"][bi_star])),
                ),
                "delta_ci": F1._ci(dn - dp),
            }
        entry["robustness_matched_n"] = robust
        # ceilings: 4-shot + the paired-difference Delta ceiling
        rel4 = _rel_pairs(ctx, f"cell_diag4_{rung}_f*.npz")
        rel0 = _rel_pairs(ctx, f"star_{rung}_f*.npz")
        entry["ceiling_4shot"] = _sb_ceiling(rel4["seed43"], rel4["seed44"])
        shared = sorted(set(rel0["rows"]) & set(rel4["rows"]))
        if shared:
            p0 = {r: k for k, r in enumerate(rel0["rows"])}
            p4 = {r: k for k, r in enumerate(rel4["rows"])}
            d43 = [rel4["seed43"][p4[r]] - rel0["seed43"][p0[r]] for r in shared]
            d44 = [rel4["seed44"][p4[r]] - rel0["seed44"][p0[r]] for r in shared]
            entry["ceiling_delta_paired"] = _sb_ceiling(d43, d44)
        else:
            entry["ceiling_delta_paired"] = {"n_contexts": 0, "note": "no shared repeat rows"}
        kshot[rung] = entry

    # Delta_peak - Delta(main) with the SELECTION-INHERITED CI (plan:64 /
    # :237(c) / §4 P4b item 6): the argmax RUNG is re-selected INSIDE each
    # draw, so the peak's selection noise rides the CI — never a frozen-peak
    # CI. Point estimate re-selects on the point deltas; the per-draw peak
    # rung distribution is persisted for auditability.
    if "main" in delta_draws_by_rung and len(delta_draws_by_rung) >= 2:
        d_rungs = [r for r in ctx.rungs if r in delta_draws_by_rung]
        d_stack = np.stack([delta_draws_by_rung[r] for r in d_rungs])  # (n_rungs, n_draws)
        peak_idx = _argmax_finite(d_stack.T)  # per-draw argmax RUNG
        peak_draws = d_stack[peak_idx, np.arange(peak_idx.size)]
        contrast_draws = peak_draws - delta_draws_by_rung["main"]
        points = np.asarray([delta_point_by_rung[r] for r in d_rungs])
        peak_i = int(np.argmax(np.where(np.isfinite(points), points, -np.inf)))
        delta_peak_minus_main: dict[str, Any] = {
            "point": float(points[peak_i] - delta_point_by_rung["main"]),
            "peak_rung_point": d_rungs[peak_i],
            "ci_selection_inherited": F1._ci(contrast_draws),
            "peak_rung_draw_frac": {
                r: float(np.mean(peak_idx == k)) for k, r in enumerate(d_rungs)
            },
            "rungs": d_rungs,
            "note": "argmax rung re-selected per draw (selection-inherited; plan:237c)",
        }
    else:
        delta_peak_minus_main = {
            "skipped": "main diag4 draws absent or <2 rungs — contrast undefined"
        }

    kshot_out = {
        "metadata": R2._metadata(),
        "smoke": ctx.smoke,
        "delta_peak_minus_main": delta_peak_minus_main,
        "layer_star": ctx.layer_star,
        "layer_fa": ctx.layer_fa,
        "band_b6": b6,
        "layer_type_split_b6": C2.layer_type_split(b6),
        "window": WW_WINDOW,
        "per_rung": kshot,
        "over_window_fracs": _over_window_fracs(ctx),
        "n_boot": ctx.n_boot,
        "adjudication_note": (
            "a substitution pattern is narrated ONLY if present in delta_ww AND "
            "delta_fa; a decay confined to the over-window pooled read at a sliding "
            "layer* is architecture-confounded (§6 sliding-window handling)"
        ),
    }
    R._write_json_atomic(eval_fits / "kshot_curve.json", kshot_out)

    # ---- retention matrix ------------------------------------------------------
    pairs = C2.transfer_pairs(tuple(ctx.rungs))
    xfer_json: dict[str, Any] = {}
    rho_main_draws: dict[str, np.ndarray] = {}
    rho_main_point: dict[str, float | None] = {}
    for i, j in pairs:
        key = f"{i}->{j}"
        try:
            arr = _assemble(
                ctx,
                f"xfer_{i}__{j}_f*.npz",
                ("ss_res_direct", "ss_res_gl", "ss_res_orth", "ss_tot"),
            )
        except FileNotFoundError:
            continue
        recs = [
            ctx.read_unit(f"xfer_{i}__{j}_f{f}")
            for f in range(ctx.spine.n_folds)
            if (ctx.eval_dir / "fits" / "units" / f"xfer_{i}__{j}_f{f}.json").exists()
        ]
        ok = [rec for rec in recs if not rec.get("skipped")]
        entry: dict[str, Any] = {"r2": {}, "ci": {}, "rho_vs_Q_jj": {}}
        dj = _r2_draws(ctx, counts, sweep[j]["ss_res"][li_star], sweep[j]["ss_tot"][li_star])
        q_jj = per_rung[j]["r2_star"]
        for mode, ssk in (
            ("direct", "ss_res_direct"),
            ("gl", "ss_res_gl"),
            ("orth", "ss_res_orth"),
        ):
            point = F1._pooled_r2(float(np.nansum(arr[ssk])), float(np.nansum(arr["ss_tot"])))
            draws = _r2_draws(ctx, counts, arr[ssk], arr["ss_tot"])
            entry["r2"][mode] = point
            entry["ci"][mode] = F1._ci(draws)
            with np.errstate(divide="ignore", invalid="ignore"):
                rho_draws = draws / dj
            rho = point / q_jj if q_jj and np.isfinite(q_jj) and q_jj != 0 else None
            entry["rho_vs_Q_jj"][mode] = {"rho": rho, "ci": F1._ci(rho_draws)}
            if mode == "orth" and j == "main":
                rho_main_draws[i] = rho_draws
                rho_main_point[i] = rho
        entry["null_r2"] = {
            "shuffled_correspondence": [
                v for rec in ok for v in rec["null_r2"]["shuffled_correspondence"]
            ],
            "spectrum_matched": [v for rec in ok for v in rec["null_r2"]["spectrum_matched"]],
        }
        entry["fit_records"] = ok[0]["fit_records"] if ok else None
        entry["baselines_fold_mean"] = (
            {
                bl: {
                    "identity_r2": float(
                        np.nanmean([rec["baselines"][bl]["identity_r2"] for rec in ok])
                    ),
                    "knn_acc_at_1_euclidean": float(
                        np.nanmean(
                            [
                                rec["baselines"][bl]["knn"]["euclidean"]["acc_at_k"]["1"]
                                for rec in ok
                            ]
                        )
                    ),
                }
                for bl in ok[0]["baselines"]
            }
            if ok
            else None
        )
        op_path = ctx.eval_dir / "fits" / "units" / f"operator_{i}__{j}.json"
        if op_path.exists():
            entry["operator"] = R._read_json(op_path)
        xfer_json[key] = entry
    # T_c: monotone suffix of rho_orth(i->main) >= 0.8 over pretraining rungs
    pretrain = _pretrain_rungs(ctx)
    pre_to_main = [r for r in pretrain if r != "main" and r in rho_main_point]
    t_c_point = None
    for k, rung in enumerate(pre_to_main):
        vals = [rho_main_point[r] for r in pre_to_main[k:]]
        if vals and all(v is not None and np.isfinite(v) and v >= 0.8 for v in vals):
            t_c_point = rung
            break
    t_c_mass = dict.fromkeys(pre_to_main, 0.0)
    t_c_none = 0
    if pre_to_main and all(r in rho_main_draws for r in pre_to_main):
        mat = np.stack([rho_main_draws[r] for r in pre_to_main])  # (P, D)
        n_draws = mat.shape[1]
        for kdraw in range(n_draws):
            col = mat[:, kdraw]
            hit = None
            for k in range(len(pre_to_main)):
                if np.all(np.isfinite(col[k:])) and np.all(col[k:] >= 0.8):
                    hit = pre_to_main[k]
                    break
            if hit is None:
                t_c_none += 1
            else:
                t_c_mass[hit] += 1.0 / n_draws
    adjacent = list(zip(list(ctx.rungs)[:-1], list(ctx.rungs)[1:]))
    retention_out = {
        "metadata": R2._metadata(),
        "smoke": ctx.smoke,
        "layer_star": ctx.layer_star,
        "pairs": xfer_json,
        "t_c": {
            "definition": (
                "earliest pretraining rung i with rho_orth(i->main) >= 0.8 for i AND "
                "every later pretraining rung (monotone suffix; §3 H2)"
            ),
            "point_rung": t_c_point,
            "rung_mass": t_c_mass,
            "no_crossing_draw_frac": t_c_none / max(ctx.n_boot, 1),
            "threshold": 0.8,
            "kill_threshold": 0.5,
        },
        "adjacent_transitions": [f"{a}->{b}" for a, b in adjacent],
        "reference": {"issue1902_median_adjacent_gl_retention": 0.893},
        "n_boot": ctx.n_boot,
    }
    R._write_json_atomic(eval_xfer / "retention_matrix.json", retention_out)

    # ---- sentinel + mirror ------------------------------------------------------
    _upload_eval_mirror(ctx, ["fits", "transfer"])
    R2.write_sentinel(
        ctx.out_root,
        "fits",
        {
            "layer_star": int(ctx.layer_star),
            "n_pairs": len(xfer_json),
            "n_kshot_rungs": len(kshot),
            "wall_h": round(wall_h, 3),
        },
        smoke=ctx.smoke,
    )


def _upload_eval_mirror(ctx: FitsCtx, subtrees: list[str]) -> None:
    """Fail-loud bulk mirror of the eval_results subtrees to the HF
    eval-mirror prefix (the P5 VM harvest source; both lanes are git-less at
    teardown). upload_dir_sharded owns verify + overflow rerouting."""
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    for sub in subtrees:
        local = ctx.eval_dir / sub
        if not local.is_dir():
            continue
        res = upload_dir_sharded(
            local,
            C.HF_DATA_REPO,
            f"{C2.EVAL_MIRROR_HF_PATH}/{sub}",
            repo_type="dataset",
            verify=True,
            delete_local=False,
        )
        logger.info(
            "[fits] mirrored %s: uploaded=%d skipped=%d rerouted=%d",
            sub,
            len(res.uploaded),
            len(res.skipped_existing),
            len(res.rerouted),
        )


# ── P4b driver (star set resident; per-rung cell waves) ──────────────────────


def _spec_pairs(spec: dict[str, Any]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for layer, rel in list(spec["y_rel_by_layer"].items()) + list(spec["x_rel_by_layer"].items()):
        out.add((rel.rsplit("/", 1)[0], int(layer)))
    return out


def _ensure_p4a_artifacts(ctx: FitsCtx) -> None:
    """P4b consumes P4a's percell shards + unit JSONs (sweep/star). On a fresh
    pod they re-stage from the eval mirror; missing entirely -> fail loud."""
    units_dir, percell = ctx.unit_paths()
    if any(percell.glob("diag0_*_f*.npz")) and any(units_dir.glob("sweep_*_f*.json")):
        return
    for sub in ("fits/percell", "fits/units"):
        try:
            n = _restage_hub_prefix(
                ctx.eval_dir, f"{C2.EVAL_MIRROR_HF_PATH}/{sub}", C2.EVAL_MIRROR_HF_PATH
            )
            logger.info("[fits] re-staged %d P4a files from the eval mirror (%s)", n, sub)
        except FileNotFoundError:
            pass
    if not any(percell.glob("diag0_*_f*.npz")):
        raise RuntimeError(
            "P4a sweep percell shards missing locally AND on the eval mirror — "
            "run the P4a fits stage first"
        )


def _check_class_pilot(ctx: FitsCtx, klass: str, basis_s: float) -> None:
    timed = [t for t in ctx.pilot_timings if t.get("klass") == klass]
    if not timed:
        return  # resumed-done pilot unit: the prior process already gated it
    per_unit = timed[0]["wall_s"]
    pilot = {"klass": klass, "per_unit_s": per_unit, "basis_s": basis_s, "ratio": PILOT_ABORT_RATIO}
    logger.info("[fits] P4b-entry pilot: %s", pilot)
    if not ctx.smoke and per_unit > PILOT_ABORT_RATIO * basis_s:
        R.designed_halt(ctx.out_root, f"p4b_pilot_wall_{klass}", pilot)


def run_p4b(args: argparse.Namespace, out_root: Path) -> None:
    t_start = time.time()
    rungs = R2._requested_rungs(args)
    print(f"[phase=fits] stage=P4b rungs={list(rungs)} smoke={bool(args.smoke)}", flush=True)
    ctx = FitsCtx(args, out_root, rungs)
    freeze, freeze_sha = R2._freeze_record(out_root)
    ctx.layer_star = int(freeze["layer_star"])
    ctx.layer_fa = int(freeze["layer_fa"])
    ctx.band5 = [int(x) for x in freeze["band_b5"]]
    ctx.band6 = [int(x) for x in freeze["band_b6"]]
    ctx.freeze_sha = freeze_sha
    include_lfa0 = ctx.layer_fa not in ctx.layers17
    specs = p4b_cell_specs(ctx, include_lfa0)
    _ensure_p4a_artifacts(ctx)
    # Fit-phase floor: 40 GB = ~1.5x the realized ~26 GB worst K-resident
    # staging wave (plan §9:345 recalibration; blocker headroom-floors-vs-plan-s9).
    C2.headroom_floor_gate(out_root, "fits", smoke=bool(args.smoke))

    # RESIDENT star working set (transfer/operator/parity inputs): star-layer
    # diag0 w+ctx + the s=main column at star + l_FA files (~9 GB, kept).
    star_needed: dict[str, list[int]] = {}
    star_soft: set[str] = set()
    for r in rungs:
        star_needed.setdefault(f"{r}/diag0", []).append(ctx.layer_star)
        star_needed.setdefault(f"{r}/diag0/ctx", []).append(ctx.layer_star)
        if r != "main":
            star_needed.setdefault(f"{r}/colC_main", []).append(ctx.layer_star)
        if include_lfa0:
            star_needed.setdefault(f"{r}/lfa0", []).append(ctx.layer_fa)
            star_needed.setdefault(f"{r}/lfa0/ctx", []).append(ctx.layer_fa)
        else:
            star_needed[f"{r}/diag0"].append(ctx.layer_fa)
            star_needed[f"{r}/diag0/ctx"].append(ctx.layer_fa)
    ensure_cells_staged(out_root, star_needed)
    resident = {(sub, layer) for sub, ls in star_needed.items() for layer in ls}
    pool = F1._WorkerPool(ctx)
    parity_gate_2544(ctx, pool.devices[0], list(rungs), resident=resident)

    def _cell_unit(spec: dict, fold: int) -> dict:
        rels = sorted(set(spec["y_rel_by_layer"].values()) | set(spec["x_rel_by_layer"].values()))
        roster = {
            "kind": "cell",
            "name": spec["name"],
            "klass": spec["klass"],
            "rung": spec["rung"],
            "answer_rung": spec["answer_rung"],
            "scope": spec["scope"],
            "x_key": spec["x_key"],
            "layers": [int(x) for x in spec["layers"]],
            "fold": fold,
        }
        pins = {
            spec["rung"]: ctx.pins[spec["rung"]],
            spec["answer_rung"]: ctx.pins[spec["answer_rung"]],
        }
        return {
            "unit": f"cell_{spec['name']}_f{fold}",
            "regime": _unit_regime(ctx, roster, rels, pins, freeze_sha=freeze_sha),
            "fn": run_cell_unit,
            "kw": {"spec": spec, "fold": fold},
        }

    # Per-rung cell waves (disk-bound: stage the wave's band files, run its
    # units, reap non-resident tensors). rowR cells ride the 'main' wave.
    k_res = max(1, int(os.environ.get("EPM_ISSUE2544_FIT_RESIDENT", "1")))
    by_rung: dict[str, list[dict]] = {r: [] for r in rungs}
    for spec in specs:
        by_rung[spec["rung"]].append(spec)
    piloted = False
    for wave in _chunks(list(rungs), k_res):
        wave_specs = [spec for r in wave for spec in by_rung[r]]
        wave_needed: dict[str, list[int]] = {}
        wave_soft: set[str] = set()
        for spec in wave_specs:
            for sub, layer in _spec_pairs(spec):
                wave_needed.setdefault(sub, [])
                if layer not in wave_needed[sub]:
                    wave_needed[sub].append(layer)
        for r in wave:
            for seed in C.RELIABILITY_SEEDS:
                sub = f"{r}/rel4_seed{seed}"
                wave_needed[sub] = [ctx.layer_star]
                wave_soft.add(sub)
        # Side files first (regimes hash store shas from sidecars); tensors
        # only when the wave has PENDING units, so a resumed wave skips the
        # multi-GB re-download of already-reaped tensors.
        ensure_cells_staged(out_root, wave_needed, soft=frozenset(wave_soft), tensors=False)
        units = [_cell_unit(spec, fold) for spec in wave_specs for fold in range(ctx.spine.n_folds)]
        if not any(not R.unit_done(out_root, u["unit"], u["regime"]) for u in units):
            logger.info("[fits] cell wave %s: all units done — tensors not staged", wave)
            continue
        ensure_cells_staged(out_root, wave_needed, soft=frozenset(wave_soft))
        if not piloted:
            pool.run("cells-pilot", units[:1])
            _check_class_pilot(ctx, "grid", P4B_GRID_BASIS_S)
            units = units[1:]
            piloted = True
        pool.run(f"cells-{'-'.join(wave)}", units)
        _reap_staged(
            ctx.store,
            {(sub, layer) for sub, ls in wave_needed.items() for layer in ls},
            resident,
        )

    pairs = C2.transfer_pairs(tuple(rungs))

    def _star_rels(r: str) -> list[str]:
        return [ctx.diag0_ctx_rel(r, ctx.layer_star), ctx.diag0_w_rel(r, ctx.layer_star)]

    def _xfer_unit(i: str, j: str, fold: int) -> dict:
        rels = sorted(
            set(_star_rels(i))
            | set(_star_rels(j))
            | {ctx.col_main_w_rel(i, ctx.layer_star), ctx.col_main_w_rel(j, ctx.layer_star)}
        )
        roster = {
            "kind": "xfer",
            "i": i,
            "j": j,
            "fold": fold,
            "modes": ["direct", "gl", "orth"],
            "layer_star": ctx.layer_star,
            "n_null": ctx.n_null,
        }
        pins = {i: ctx.pins[i], j: ctx.pins[j], "main": ctx.pins["main"]}
        return {
            "unit": f"xfer_{i}__{j}_f{fold}",
            "regime": _unit_regime(ctx, roster, rels, pins, freeze_sha=freeze_sha),
            "fn": run_xfer_unit,
            "kw": {"i": i, "j": j, "fold": fold},
        }

    xfer_units = [_xfer_unit(i, j, fold) for i, j in pairs for fold in range(ctx.spine.n_folds)]
    pool.run("xfer-pilot", xfer_units[:1])
    _check_class_pilot(ctx, "xfer", P4B_XFER_BASIS_S)
    pool.run("xfer", xfer_units[1:])

    op_units = [
        {
            "unit": f"operator_{i}__{j}",
            "regime": _unit_regime(
                ctx,
                {"kind": "operator", "i": i, "j": j, "n_rot": ctx.n_rot},
                sorted(set(_star_rels(i)) | set(_star_rels(j))),
                {i: ctx.pins[i], j: ctx.pins[j]},
                freeze_sha=freeze_sha,
            ),
            "fn": run_operator_unit,
            "kw": {"i": i, "j": j},
        }
        for i, j in pairs
    ]
    pool.run("operator", op_units)

    finalize_p4b(ctx, (time.time() - t_start) / 3600.0)
    print(
        f"[phase=fits] P4b done: {len(specs) * ctx.spine.n_folds} cell units, "
        f"{len(xfer_units)} transfer units, {len(op_units)} operator units",
        flush=True,
    )


# ── entry point (issue2544_run --phase fits) ─────────────────────────────────


def run_fits(args: argparse.Namespace, out_root: Path) -> None:
    """Unit-A registration contract: ``issue2544_run --phase fits`` calls
    exactly this. Stage discrimination: ``EPM_ISSUE2544_FITS_STAGE`` in
    {p4a, p4b, auto}; auto = P4b iff the layer-freeze record resolves (local
    or HF mirror), else P4a — the dispatcher pins the stage explicitly."""
    if args.init or args.worker or args.finalize:
        raise SystemExit(
            "--phase fits takes no --init/--worker/--finalize legs (single-process, "
            "device-pinned worker threads; the dispatcher runs it as one leg per stage)"
        )
    stage = os.environ.get("EPM_ISSUE2544_FITS_STAGE", "auto").lower()
    if stage not in ("p4a", "p4b", "auto"):
        raise SystemExit(f"EPM_ISSUE2544_FITS_STAGE={stage!r} not in {{p4a, p4b, auto}}")
    if stage == "auto":
        try:
            R2._freeze_record(out_root)
            stage = "p4b"
        except RuntimeError:
            stage = "p4a"
        print(f"[phase=fits] stage auto-resolved -> {stage}", flush=True)
    if stage == "p4a":
        run_p4a(args, out_root)
    else:
        run_p4b(args, out_root)
