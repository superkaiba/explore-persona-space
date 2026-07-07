"""Issue #931 follow-up `author-blocked-folds`: author-grouped Arm-A re-fits.

Re-scores the two committed Arm-A cells with the fold/null/bootstrap GROUPING
UNIT moved one level up the sharing hierarchy — novel (28 groups) -> author
(19 PDNC Author Codes) — to decide whether part of the pooled real-novel R^2
(+0.173 @ L19) is an author-level component the novel-level folds cannot see
(Austen holds 6/28 novels = 36.4% of rows). Plan v10
(tasks/.../931/plans/v10.md sections 4/6/9/10); one changed variable, every
estimator knob inherited verbatim (fit seed 0, K=5 `_cv_folds`, 20 null
draws, 1000-draw group bootstrap, 28-layer sweep, L19 headline).

Protocol — reuse-not-rewrite:
  - pinned armA store shards @ revision 9534b998 (the run's own committed
    pin, `issue931_distance_covariate.py`), staged with a file_exists
    pre-assert + bounded-retry per-file ``hf_hub_download`` (never
    snapshot_download on the ~1M-file data repo);
  - novel -> author mapping from ``PDNC-Novel-Index.csv`` at the pinned PDNC
    checkout (``issue931_common.PDNC_SHA`` via the reused
    ``issue931_build_pairs.stage_pdnc``); asserts: all 28 store group_ids
    resolve, exactly one non-empty Author Code per novel, exactly 19 unique
    authors, novel-count multiset {6, 4, 2, 1x16};
  - both cells re-fit through the EXISTING ``issue931_fit_cells.fit_cell``
    path with ``group_ids = author_ids`` (folds via ``_cv_folds``, nulls via
    the group-blocked ``_conv_perm`` — the inherited null family IS the
    registered statistic; no re-implementation, no row sorting);
  - LOAO secondary: K=19 leave-one-author-out at the headline layer only,
    obs-only (labeled secondary, never the decision input);
  - pseudo-group regrouping null: 20 seeded draws (seeds 0..19) assigning the
    28 novels into 19 pseudo-groups matching the author novel-count multiset,
    each refit L19-only obs-only (the group-coarsening decomposition control).

Checkpoint/resume: per-cell atomic JSONL append keyed on
(cell_id, fold_scheme) under a protocol_fingerprint = sha256(driver git SHA +
estimator constants + fold scheme + store revision + PDNC SHA)[:12]; resume
skips only on an exact fingerprint match; ANY stale-fingerprint row fails
loud (the round-1 `power_curve_multi_seed` pattern). Pseudo draws checkpoint
per draw under fold_scheme ``pseudo_k5_draw<seed>``.

Compute: 0 GPU-h, VM CPU. The driver sets NO thread env vars — the launcher's
canonical env prefix owns the caps (plan section 9: 2-thread caps under load,
the round-1 measured fp64-eigh thrash finding); the effective caps are logged
at startup. After the first fitted cell the driver projects the battery wall
from the plan-section-9 FLOP model and applies the registered descope ladder
if the projection exceeds 2x the budget: (1) drop the LOAO secondary,
(2) drop the ctxmean cell's null draws (obs-only refit, labeled deviation).
The within cell (full nulls + bootstrap) and the pseudo-group control are
NEVER descoped (plan section 9).

Outputs: <out-dir>/author_blocked_folds.json (+ mapping JSON + cells JSONL
checkpoint + the fit_cell-written cells_/nulls_*_authorfold.json + preds npz
under <data-dir>/store/preds/) and
figures/issue_931/author_blocked_folds.{png,pdf,meta.json}. Committed
reference inputs (cells_armA_within.json, cells_armA_ctxmean.json,
transfer_matrix.json) are read from the repo's canonical eval_results dir — a
NON-rebinding constant, deliberately independent of --out-dir so a
scratch-dir smoke never orphans them.

CLI:
  uv run python scripts/issue931_author_blocked_folds.py \
      [--out-dir eval_results/issue_931] [--fig-dir figures/issue_931] \
      [--data-dir data/issue_931] \
      [--stage-dir data/issue_931/hf_dl/authorfolds] \
      [--pdnc-dir data/issue_931/pdnc] \
      [--phases all|stage,mapping,fits,loao,pseudo,aggregate] \
      [--null-draws 20] [--n-boot 1000] [--pseudo-draws 20] \
      [--budget-hours 4.5] [--protocol-tag ""] [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/torch import

import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(SCRIPTS))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402
import issue931_fit_cells as fitc  # noqa: E402
from issue931_build_pairs import stage_pdnc  # noqa: E402

SCRIPT = "scripts/issue931_author_blocked_folds.py"
# The run's own committed armA-store pin (issue931_distance_covariate.py:63,
# echoed in delta_char_distance_covariate.json) — plan v10 section 10.
STORE_REV = "9534b9981d6b4fb4f1259c9b06f021d311a46af4"
STORE_PREFIX = "issue931_story_map/analysis_tensors/armA"
N_SHARDS = 4
EXPECTED_N = 1982
EXPECTED_NOVELS = 28
EXPECTED_AUTHORS = 19
# Author NOVEL-count multiset (plan section 12 A11: AUST 6, FORS 4, DICK 2,
# 16 singletons; asserted against the realized mapping at run time).
EXPECTED_MULTISET = (6, 4, 2) + (1,) * 16

CELL_WITHIN = "armA_within_authorfold"
CELL_CTXMEAN = "armA_ctxmean_authorfold"
CELL_LOAO = "armA_within_loao_l19"
CELL_PSEUDO = "armA_within_pseudogroup"
FOLD_SCHEME = "k5_author_grouped"
LOAO_SCHEME = "loao19_l19"
LOAO_FOLDS = 19
SMOKE_LAYERS = (18, 19)  # headline L19 + one adjacent frozen layer

# Registered decision-rule constants (plan v10 sections 3/6; committed values
# re-asserted against the committed JSONs at aggregation).
H_AMEND_NUMERATOR = 0.17289959611807892  # sweep-convention novel-fold L19 obs
B_NOVEL = 0.20806361277603524  # bootstrap-convention novel-fold L19 obs
NEAR_BAND_EPS = 0.05  # section-6 near-band sensitivity caveat

HIDDEN = common.EXPECTED_HIDDEN  # 3584 (section-9 FLOP model)
PHASES_ALL = ("stage", "mapping", "fits", "loao", "pseudo", "aggregate")

# Committed reference inputs — canonical repo paths, NOT rebound by --out-dir
# (smoke-root rebinding must never orphan read-only committed inputs).
COMMITTED_EVAL_DIR = SCRIPTS.parent / "eval_results" / "issue_931"


def estimator_constants(args: argparse.Namespace) -> dict:
    """The estimator identity folded into the protocol fingerprint."""
    return {
        "lambdas": [float(v) for v in fit825.LAMBDAS],
        "n_folds": int(common.N_FOLDS),
        "fit_seed": int(common.FIT_SEED),
        "null_draws": int(args.null_draws),
        "n_boot": int(args.n_boot),
        "pseudo_draws": int(args.pseudo_draws),
        "loao_folds": LOAO_FOLDS,
        "headline_layer": int(common.HEADLINE_LAYER),
        "layers_kept": [int(v) for v in args.layers_kept],
        "fold_scheme": FOLD_SCHEME,
    }


def fingerprint_basis(git_sha: str, args: argparse.Namespace) -> dict:
    return {
        "driver_git_sha": git_sha,
        "estimator": estimator_constants(args),
        "fold_scheme": FOLD_SCHEME,
        "store_revision": STORE_REV,
        "pdnc_sha": common.PDNC_SHA,
        "protocol_tag": args.protocol_tag,
    }


def protocol_fingerprint(git_sha: str, args: argparse.Namespace) -> str:
    """sha256(driver git SHA + estimator constants + fold scheme + store
    revision + PDNC SHA)[:12] — the checkpoint/resume key (plan section 4
    step 7)."""
    basis = json.dumps(fingerprint_basis(git_sha, args), sort_keys=True)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Checkpoint (per-cell atomic JSONL append + fingerprint-gated resume)
# ---------------------------------------------------------------------------


def append_jsonl(path: Path, row: dict) -> None:
    """Single-line O_APPEND write + fsync (atomic per-cell checkpoint)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=float) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def load_checkpoint(path: Path, fp: str) -> dict[tuple[str, str], dict]:
    """Load completed cells keyed (cell_id, fold_scheme); fail loud on ANY
    stale fingerprint or duplicate key — no silent protocol mixing."""
    if not path.exists():
        return {}
    rows: list[dict] = []
    for line in path.open(encoding="utf-8"):  # file iteration, never splitlines()
        if line.strip():
            rows.append(json.loads(line))
    stale = sorted({r.get("protocol_fingerprint", "<missing>") for r in rows} - {fp})
    if stale:
        raise RuntimeError(
            f"checkpoint {path} holds stale-protocol cells (fingerprints {stale}, current {fp}); "
            "refusing to resume or mix — move/delete the checkpoint to rerun under the new protocol"
        )
    by_key: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (str(r["cell_id"]), str(r["fold_scheme"]))
        if key in by_key:
            raise RuntimeError(f"duplicate checkpoint cell for {key} in {path}")
        by_key[key] = r
    return by_key


def write_runconfig(path: Path, fp: str, descope: dict) -> None:
    """Persist the descope decision so per-phase resumes honor it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"protocol_fingerprint": fp, "descope": descope}, indent=2))
    os.replace(tmp, path)


def read_runconfig(path: Path, fp: str) -> dict:
    """Read the persisted descope decision; fingerprint mismatch fails loud."""
    if not path.exists():
        return {"applied": False, "loao_dropped": False, "ctxmean_null_draws_dropped": False}
    rc = json.loads(path.read_text())
    if rc.get("protocol_fingerprint") != fp:
        raise RuntimeError(
            f"runconfig {path} fingerprint {rc.get('protocol_fingerprint')} != current {fp}; "
            "move/delete it (and the checkpoint) to rerun under the new protocol"
        )
    return rc["descope"]


# ---------------------------------------------------------------------------
# Staging (pinned revision; per-file hf_hub_download; file_exists pre-assert)
# ---------------------------------------------------------------------------


def stage_store(stage_dir: Path) -> Path:
    """Stage the 4 pinned armA shards; returns the store dir (plan 4 step 1)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    paths = [f"{STORE_PREFIX}/armA_shard{i:03d}.pt" for i in range(N_SHARDS)]
    infos = api.get_paths_info(common.HF_DATA_REPO, paths, repo_type="dataset", revision=STORE_REV)
    by_path = {e.path: e for e in infos}
    for p in paths:  # A2 re-assert: the pinned revision resolves every shard
        assert p in by_path, f"pinned shard missing at {STORE_REV}: {p}"
    total_gb = sum((by_path[p].size or 0) for p in paths) / 1e9
    stage_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(stage_dir).free / 1e9
    print(f"[i931-abf] staging {total_gb:.2f} GB (free {free_gb:.1f} GB)", flush=True)
    assert free_gb > max(10.0, 2.0 * total_gb), (
        f"insufficient disk headroom at {stage_dir}: {free_gb:.1f} GB free "
        f"for a {total_gb:.2f} GB store (A9 staging-time df check)"
    )
    for p in paths:
        for attempt in range(4):
            try:
                hf_hub_download(
                    common.HF_DATA_REPO,
                    p,
                    repo_type="dataset",
                    revision=STORE_REV,
                    local_dir=stage_dir,
                )
                break
            except Exception as exc:  # transient Hub 5xx/429 — bounded retry
                if attempt == 3:
                    raise
                wait = 20 * (attempt + 1)
                print(f"[i931-abf] retry {p} in {wait}s: {exc}", flush=True)
                time.sleep(wait)
        print(f"[i931-abf] staged {p}", flush=True)
    return stage_dir / STORE_PREFIX


# ---------------------------------------------------------------------------
# Store load + novel -> author mapping (plan section 4 step 2)
# ---------------------------------------------------------------------------


def load_store(store_dir: Path, layers_kept: tuple[int, ...]) -> dict:
    """Load the armA store via the reused loader; assert shape; slice layers.

    Layer slicing (identity in production, {18, 19} under --smoke) happens
    AFTER the full-shape asserts so the pinned-store contract is always
    checked. x_last is dropped (not re-fit this round; ~0.8 GB saved).
    """
    store = fitc.load_regime_store(store_dir, "armA")
    store["arrays"].pop("x_last", None)
    X = store["arrays"]["x_spanmean"]
    assert X.shape == (EXPECTED_N, common.EXPECTED_LAYERS, HIDDEN), X.shape
    assert len(store["row_ids"]) == EXPECTED_N, len(store["row_ids"])
    n_novels = len(np.unique(store["group_ids"]))
    assert n_novels == EXPECTED_NOVELS, f"expected {EXPECTED_NOVELS} novel groups, got {n_novels}"
    if tuple(layers_kept) != tuple(range(common.EXPECTED_LAYERS)):
        idx = list(layers_kept)
        for k in list(store["arrays"]):
            store["arrays"][k] = store["arrays"][k][:, idx, :]
    return store


def build_author_mapping(store: dict, pdnc_dir: Path) -> dict:
    """novel folder name -> PDNC Author Code, with the plan's hard asserts.

    Returns {novel_to_author, author_ids (N,), per_author_rows,
    per_author_novels, sizes_desc, novels_sorted}.
    """
    head = stage_pdnc(pdnc_dir, common.PDNC_SHA, skip_clone=False)
    assert head == common.PDNC_SHA, f"PDNC checkout at {head}, expected pin {common.PDNC_SHA}"
    index_csv = pdnc_dir / "PDNC-Novel-Index.csv"
    assert index_csv.exists(), index_csv
    folder_to_author: dict[str, str] = {}
    with open(index_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            folder = (row.get("Folder Name") or "").strip()
            if not folder:
                continue
            assert folder not in folder_to_author, f"duplicate Folder Name {folder!r} in PDNC index"
            folder_to_author[folder] = (row.get("Author Code") or "").strip()
    group_ids = np.asarray(store["group_ids"])
    novels_sorted = sorted(str(g) for g in np.unique(group_ids))
    novel_to_author: dict[str, str] = {}
    for nv in novels_sorted:
        assert nv in folder_to_author, f"store novel {nv!r} unresolved in PDNC-Novel-Index.csv"
        code = folder_to_author[nv]
        # A10: exactly one non-empty Author Code (translator codes are a
        # separate column and never enter this mapping).
        assert code and "," not in code and ";" not in code and " " not in code, (
            f"novel {nv!r} has no single non-empty Author Code: {code!r}"
        )
        novel_to_author[nv] = code
    author_ids = np.asarray([novel_to_author[str(g)] for g in group_ids])
    uniq_authors = sorted(np.unique(author_ids))
    assert len(uniq_authors) == EXPECTED_AUTHORS, (
        f"expected {EXPECTED_AUTHORS} unique authors, got {len(uniq_authors)}: {uniq_authors}"
    )
    per_author_rows = {a: int((author_ids == a).sum()) for a in uniq_authors}
    per_author_novels = dict(Counter(novel_to_author.values()))
    sizes_desc = tuple(sorted(per_author_novels.values(), reverse=True))
    assert sizes_desc == EXPECTED_MULTISET, (
        f"author novel-count multiset {sizes_desc} != expected {EXPECTED_MULTISET} (A11)"
    )
    return {
        "novel_to_author": novel_to_author,
        "author_ids": author_ids,
        "per_author_rows": per_author_rows,
        "per_author_novels": per_author_novels,
        "sizes_desc": sizes_desc,
        "novels_sorted": novels_sorted,
    }


def fold_composition(author_ids: np.ndarray, group_ids: np.ndarray) -> list[dict]:
    """Realized K=5 author-grouped fold compositions (groups AND rows/fold)."""
    folds = fit825._cv_folds(author_ids, common.N_FOLDS, common.FIT_SEED)
    comp = []
    for k in range(common.N_FOLDS):
        m = folds == k
        comp.append(
            {
                "fold": k,
                "n_rows": int(m.sum()),
                "n_authors": len(np.unique(author_ids[m])),
                "n_novels": len(np.unique(np.asarray(group_ids)[m])),
                "authors": sorted(str(a) for a in np.unique(author_ids[m])),
            }
        )
    assert sum(c["n_rows"] for c in comp) == len(author_ids)
    return comp


def pseudo_ids_for_draw(
    group_ids: np.ndarray, novels_sorted: list[str], sizes_desc: tuple[int, ...], draw_seed: int
) -> tuple[np.ndarray, dict[str, str]]:
    """Seeded random novel -> pseudo-group assignment matching the author
    novel-count multiset (plan section 4 step 5). Returns (row ids, mapping)."""
    rng = np.random.default_rng(draw_seed)
    perm = rng.permutation(len(novels_sorted))
    assignment: dict[str, str] = {}
    pos = 0
    for gi, size in enumerate(sizes_desc):
        for _ in range(size):
            assignment[novels_sorted[int(perm[pos])]] = f"pg{gi:02d}"
            pos += 1
    assert pos == len(novels_sorted), (pos, len(novels_sorted))
    realized = tuple(sorted(Counter(assignment.values()).values(), reverse=True))
    assert realized == tuple(sizes_desc), f"draw {draw_seed}: multiset {realized} != {sizes_desc}"
    pseudo_row_ids = np.asarray([assignment[str(g)] for g in group_ids])
    return pseudo_row_ids, assignment


# ---------------------------------------------------------------------------
# Section-9 FLOP model + descope ladder
# ---------------------------------------------------------------------------


def _fold_layer_flops(n_tr: float, n_passes: int) -> float:
    """eigh (9*n_tr^3) + n_passes GCV predicts (4*n_tr^2*HIDDEN each)."""
    return 9.0 * n_tr**3 + n_passes * 4.0 * n_tr**2 * HIDDEN


def battery_flops(n: int, n_layers: int, null_draws: int) -> float:
    n_tr = n * (1.0 - 1.0 / common.N_FOLDS)
    return common.N_FOLDS * n_layers * _fold_layer_flops(n_tr, 1 + null_draws)


def loao_flops(n: int) -> float:
    return LOAO_FOLDS * _fold_layer_flops(n * (1.0 - 1.0 / LOAO_FOLDS), 1)


def pseudo_flops(n: int, draws: int) -> float:
    return draws * common.N_FOLDS * _fold_layer_flops(n * 0.8, 1)


def apply_descope(first_wall_s: float, n: int, n_layers: int, args: argparse.Namespace) -> dict:
    """Registered descope ladder (plan section 9): if the projected total
    exceeds 2x budget, (1) drop LOAO, (2) drop ctxmean nulls. The within cell
    and the pseudo-group control are NEVER descoped."""
    rate = first_wall_s / battery_flops(n, n_layers, args.null_draws)  # s/FLOP
    threshold_h = 2.0 * args.budget_hours

    def _projected(loao_in: bool, ctx_nulls: int) -> float:
        rem = battery_flops(n, n_layers, ctx_nulls) + pseudo_flops(n, args.pseudo_draws)
        if loao_in:
            rem += loao_flops(n)
        return (first_wall_s + rem * rate) / 3600.0

    info = {
        "applied": False,
        "loao_dropped": False,
        "ctxmean_null_draws_dropped": False,
        "first_cell_wall_s": float(first_wall_s),
        "budget_hours": float(args.budget_hours),
        "threshold_hours": threshold_h,
        "projected_total_hours": _projected(True, args.null_draws),
    }
    if info["projected_total_hours"] > threshold_h:
        info["applied"] = info["loao_dropped"] = True
        info["projected_total_hours_after_loao_drop"] = _projected(False, args.null_draws)
        if info["projected_total_hours_after_loao_drop"] > threshold_h:
            info["ctxmean_null_draws_dropped"] = True
            info["projected_total_hours_after_descope"] = _projected(False, 0)
    return info


# ---------------------------------------------------------------------------
# Phases: fits / loao / pseudo
# ---------------------------------------------------------------------------


def _cell_xy(store: dict, mapping: dict, x_key: str) -> dict:
    """xy dict for fit_cell with the AUTHOR grouping vector (plan 4 step 3)."""
    X = store["arrays"][x_key]
    Y = store["arrays"]["y"]
    author_ids = mapping["author_ids"]
    row_ids = store["row_ids"]
    # fit_cell reads xy["row_ids"] at preds persistence (issue931_fit_cells.py
    # ~L462) — the 4-way identical-length assert must hold BEFORE the first fit.
    assert len(X) == len(Y) == len(author_ids) == len(row_ids), (
        len(X),
        len(Y),
        len(author_ids),
        len(row_ids),
    )
    return {"X": X, "Y": Y, "group_ids": author_ids, "row_ids": row_ids}


def _fit_args(args: argparse.Namespace, null_draws: int) -> argparse.Namespace:
    return argparse.Namespace(
        folds=common.N_FOLDS,
        seed=common.FIT_SEED,
        null_draws=null_draws,
        n_boot=args.n_boot,
        out_dir=Path(args.out_dir),
        data_dir=Path(args.data_dir),
    )


def _main_cell_row(
    fp: str, cell_id: str, payload: dict, hl: int, layers_kept: tuple[int, ...], wall: float
) -> dict:
    ss = payload["selection_symmetric"]["frozen_layer_table"].get(str(hl), {})
    p975 = ss.get("null_p975")
    boot = payload["r2_bootstrap_group_frozen"][str(hl)]
    return {
        "protocol_fingerprint": fp,
        "cell_id": cell_id,
        "fold_scheme": FOLD_SCHEME,
        "n": int(payload["n"]),
        "n_groups": int(payload["n_groups"]),
        "headline_layer_index": int(hl),
        "headline_layer": int(layers_kept[hl]),
        "layers_kept": [int(v) for v in layers_kept],
        "null_draws_used": int(payload["null_draws"]),
        "r2_per_layer": [float(v) for v in payload["r2_per_layer_obs"]],
        "r2_l19": float(payload["r2_per_layer_obs"][hl]),
        "null_p975_l19": None if p975 is None or math.isnan(p975) else float(p975),
        "bootstrap_group_l19": {
            "obs": float(boot["r2"]),
            "ci_lo": float(boot["ci_lo"]),
            "ci_hi": float(boot["ci_hi"]),
            "n_boot": int(boot["n_boot"]),
        },
        "per_author_r2": payload["per_group_r2_headline"],
        "wall_seconds": float(wall),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_fits(
    store: dict,
    mapping: dict,
    args: argparse.Namespace,
    fp: str,
    ckpt: Path,
    by_key: dict,
    runconfig_path: Path,
    descope: dict,
) -> dict:
    """Fit both registered cells (within first — its timing drives the
    descope ladder), checkpointing each on completion."""
    hl = fitc.headline_layer(len(args.layers_kept))
    assert args.layers_kept[hl] == common.HEADLINE_LAYER or len(args.layers_kept) < 28, (
        args.layers_kept,
        hl,
    )
    # Resume robustness: within cell checkpointed but the descope decision not
    # persisted (crash between the two writes) — re-derive from the stored wall.
    wrow = by_key.get((CELL_WITHIN, FOLD_SCHEME))
    if wrow is not None and not descope.get("frozen"):
        descope = apply_descope(
            float(wrow["wall_seconds"]), int(wrow["n"]), len(args.layers_kept), args
        )
        descope["frozen"] = True
        write_runconfig(runconfig_path, fp, descope)
    for cell_id, x_key in ((CELL_WITHIN, "x_spanmean"), (CELL_CTXMEAN, "x_ctxmean")):
        key = (cell_id, FOLD_SCHEME)
        if key in by_key:
            print(f"[i931-abf] skip completed cell {key}", flush=True)
            continue
        nd = args.null_draws
        if cell_id == CELL_CTXMEAN and descope.get("ctxmean_null_draws_dropped"):
            nd = 0
            print("[i931-abf] descope: ctxmean refit obs-only (null_draws=0)", flush=True)
        xy = _cell_xy(store, mapping, x_key)
        print(f"[i931-abf] fit start {cell_id} null_draws={nd}", flush=True)
        t0 = time.time()
        res = fitc.fit_cell(cell_id, xy, _fit_args(args, nd))
        wall = time.time() - t0
        row = _main_cell_row(fp, cell_id, res["payload"], hl, args.layers_kept, wall)
        append_jsonl(ckpt, row)
        by_key[key] = row
        print(f"[i931-abf] cell {cell_id} L19={row['r2_l19']:.4f} wall={wall:.1f}s", flush=True)
        if cell_id == CELL_WITHIN and not descope.get("applied") and not descope.get("frozen"):
            n_layers = len(args.layers_kept)
            descope = {**apply_descope(wall, int(store["arrays"]["y"].shape[0]), n_layers, args)}
            descope["frozen"] = True  # decided once, on the first timed cell
            write_runconfig(runconfig_path, fp, descope)
            print(
                f"[i931-abf] projected total {descope['projected_total_hours']:.2f} h "
                f"(threshold {descope['threshold_hours']:.2f} h); "
                f"loao_dropped={descope['loao_dropped']} "
                f"ctxmean_nulls_dropped={descope['ctxmean_null_draws_dropped']}",
                flush=True,
            )
    return descope


def run_loao(
    store: dict, mapping: dict, args: argparse.Namespace, fp: str, ckpt: Path, by_key: dict
) -> None:
    """K=19 leave-one-author-out at the headline layer, obs-only (secondary)."""
    key = (CELL_LOAO, LOAO_SCHEME)
    if key in by_key:
        print(f"[i931-abf] skip completed cell {key}", flush=True)
        return
    hl = fitc.headline_layer(len(args.layers_kept))
    X = store["arrays"]["x_spanmean"][:, hl : hl + 1, :]
    Y = store["arrays"]["y"][:, hl : hl + 1, :]
    print("[i931-abf] LOAO start (K=19, obs only)", flush=True)
    t0 = time.time()
    sw = fit825.heldout_r2_sweep(
        X,
        Y,
        mapping["author_ids"],
        n_folds=LOAO_FOLDS,
        seed=common.FIT_SEED,
        null_draws=0,
        collect_cosines=False,
    )
    wall = time.time() - t0
    row = {
        "protocol_fingerprint": fp,
        "cell_id": CELL_LOAO,
        "fold_scheme": LOAO_SCHEME,
        "n": int(X.shape[0]),
        "n_groups": EXPECTED_AUTHORS,
        "headline_layer": int(args.layers_kept[hl]),
        "r2_l19": float(sw["r2_obs"][0]),
        "wall_seconds": float(wall),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    append_jsonl(ckpt, row)
    by_key[key] = row
    print(f"[i931-abf] LOAO L19={row['r2_l19']:.4f} wall={wall:.1f}s", flush=True)


def run_pseudo(
    store: dict, mapping: dict, args: argparse.Namespace, fp: str, ckpt: Path, by_key: dict
) -> None:
    """20 seeded size-multiset-matched pseudo-group regrouping draws, each an
    obs-only L19 refit of the within cell (plan section 4 step 5)."""
    hl = fitc.headline_layer(len(args.layers_kept))
    X = store["arrays"]["x_spanmean"][:, hl : hl + 1, :]
    Y = store["arrays"]["y"][:, hl : hl + 1, :]
    group_ids = np.asarray(store["group_ids"])
    for s in range(args.pseudo_draws):
        key = (CELL_PSEUDO, f"pseudo_k5_draw{s}")
        if key in by_key:
            print(f"[i931-abf] skip completed cell {key}", flush=True)
            continue
        pseudo_row_ids, assignment = pseudo_ids_for_draw(
            group_ids, mapping["novels_sorted"], mapping["sizes_desc"], s
        )
        t0 = time.time()
        sw = fit825.heldout_r2_sweep(
            X,
            Y,
            pseudo_row_ids,
            n_folds=common.N_FOLDS,
            seed=common.FIT_SEED,
            null_draws=0,
            collect_cosines=False,
        )
        wall = time.time() - t0
        row = {
            "protocol_fingerprint": fp,
            "cell_id": CELL_PSEUDO,
            "fold_scheme": f"pseudo_k5_draw{s}",
            "draw_seed": s,
            "headline_layer": int(args.layers_kept[hl]),
            "r2_l19": float(sw["r2_obs"][0]),
            "novel_to_pseudo": assignment,
            "wall_seconds": float(wall),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_jsonl(ckpt, row)
        by_key[key] = row
        print(f"[i931-abf] pseudo draw {s} L19={row['r2_l19']:.4f} wall={wall:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Aggregation: committed refs + decision rule + decomposition + side-read
# ---------------------------------------------------------------------------


def committed_reference() -> dict:
    """Quote the committed novel-fold reference values (never re-derived);
    fail loud on a missing key or a drifted registered constant (A6)."""
    w = json.loads((COMMITTED_EVAL_DIR / "cells_armA_within.json").read_text())
    c = json.loads((COMMITTED_EVAL_DIR / "cells_armA_ctxmean.json").read_text())
    w_l19 = float(w["r2_per_layer_obs"][common.HEADLINE_LAYER])
    assert abs(w_l19 - H_AMEND_NUMERATOR) < 1e-12, w_l19
    w_boot = w["r2_bootstrap_group_frozen"]["19"]
    assert abs(float(w_boot["r2"]) - B_NOVEL) < 1e-12, w_boot["r2"]
    c_l19 = float(c["r2_per_layer_obs"][common.HEADLINE_LAYER])
    return {
        "armA_within": {
            "r2_l19": w_l19,
            "null_p975_l19": float(
                w["selection_symmetric"]["frozen_layer_table"]["19"]["null_p975"]
            ),
            "bootstrap_group_l19": {
                "obs": float(w_boot["r2"]),
                "ci_lo": float(w_boot["ci_lo"]),
                "ci_hi": float(w_boot["ci_hi"]),
            },
        },
        "armA_ctxmean": {
            "r2_l19": c_l19,
            "null_p975_l19": float(
                c["selection_symmetric"]["frozen_layer_table"]["19"]["null_p975"]
            ),
            "bootstrap_group_l19": {
                "obs": float(c["r2_bootstrap_group_frozen"]["19"]["r2"]),
                "ci_lo": float(c["r2_bootstrap_group_frozen"]["19"]["ci_lo"]),
                "ci_hi": float(c["r2_bootstrap_group_frozen"]["19"]["ci_hi"]),
            },
        },
        "ratio_ctxmean_over_within_novel": c_l19 / w_l19,
    }


def registered_read(r_a: float, band_a: float, ci_lo_a: float, ci_hi_a: float) -> dict:
    """Pure decision function for the plan-v10 section-6 table (all at L19).

    Rows partition the outcome space: R_a <= band_a -> row 3; else B_novel
    above / inside / below the author-fold bootstrap CI -> rows 1 / 2 / 4.
    Convention guard: R_a vs band_a uses the SWEEP convention; the CI gate
    uses the BOOTSTRAP convention on BOTH sides (B_novel vs CI_a).
    """
    if r_a <= band_a:
        row = "row3_existence_fails_under_honest_fold"
    elif ci_hi_a < B_NOVEL:
        row = "row1_author_level_component_measured"
    elif ci_lo_a > B_NOVEL:
        row = "row4_author_fold_higher_unexpected"
    else:
        row = "row2_no_detectable_author_level_component"
    return {
        "decision_row": row,
        "h_amend_numerator": H_AMEND_NUMERATOR,
        "b_novel": B_NOVEL,
        "r_a_sweep_obs": float(r_a),
        "band_a_null_p975": float(band_a),
        "ci_a_bootstrap": [float(ci_lo_a), float(ci_hi_a)],
        "near_band_sensitivity": bool(abs(r_a - band_a) < NEAR_BAND_EPS),
        "conventions": {
            "point_comparisons": "sweep (per-fold-accumulated SS_tot)",
            "ci_membership_gate": "bootstrap (global-mean SS_tot) on BOTH sides",
        },
    }


def transfer_redivision(r2_per_layer_author: list[float], layers_kept: tuple[int, ...]) -> dict:
    """FREE arithmetic side-read (plan 4 step 6): the committed span-mean
    chat->armA transfer rows re-divided by the author-fold armA_within
    ceiling at each row's own layer. Pure re-division, no refit; the
    *_lastpos-denominator rows stay novel-fold (that cell is not re-fit)."""
    t = json.loads((COMMITTED_EVAL_DIR / "transfer_matrix.json").read_text())
    index_of = {int(li): i for i, li in enumerate(layers_kept)}
    rows_out = []
    n_eligible = 0
    for r in t["rows"]:
        if r.get("direction") != "chat_ref->armA" or r.get("denominator_cell") != "armA_within":
            continue
        n_eligible += 1
        li = int(r["layer"])
        if li not in index_of:
            continue  # smoke layer slice; production keeps all 28 layers
        den = float(r2_per_layer_author[index_of[li]])
        rows_out.append(
            {
                "layer": li,
                "application": r["application"],
                "n_train": int(r["n_train"]),
                "power_matched": bool(r["power_matched"]),
                "transfer_r2_committed": float(r["transfer_r2"]),
                "committed_denominator_novel_fold": float(r["within_ceiling_r2"]),
                "committed_fraction_of_ceiling": float(r["fraction_of_ceiling"]),
                "author_fold_denominator": den,
                "author_fold_fraction_of_ceiling": float(r["transfer_r2"]) / den,
            }
        )
    if len(layers_kept) == common.EXPECTED_LAYERS:
        assert len(rows_out) == 16, f"expected 16 chat->armA armA_within rows, got {len(rows_out)}"
    return {
        "note": (
            "labeled sensitivity side-read — pure re-division of committed transfer_r2 rows by "
            "the author-fold armA_within per-layer ceiling; the registered H2 verdict (chat-side "
            "matched fractions) is untouched; *_lastpos-denominator rows stay novel-fold"
        ),
        "n_rows_eligible": n_eligible,
        "rows": rows_out,
    }


def aggregate(args: argparse.Namespace, fp: str, by_key: dict, descope: dict) -> dict:
    """Assemble the section-10 deliverable JSON from checkpointed cells +
    the mapping JSON + the committed references; write it + the figure."""
    out_dir = Path(args.out_dir)
    mapping_path = out_dir / "author_blocked_folds_mapping.json"
    assert mapping_path.exists(), f"mapping JSON missing: {mapping_path} (run --phases mapping)"
    mp = json.loads(mapping_path.read_text())
    assert mp["protocol_fingerprint"] == fp, (mp["protocol_fingerprint"], fp)
    layers_kept = tuple(int(v) for v in mp["layers_kept"])
    assert layers_kept == tuple(args.layers_kept), (layers_kept, args.layers_kept)

    within = by_key.get((CELL_WITHIN, FOLD_SCHEME))
    ctxmean = by_key.get((CELL_CTXMEAN, FOLD_SCHEME))
    assert within is not None, "armA_within_authorfold cell missing from checkpoint"
    assert ctxmean is not None, "armA_ctxmean_authorfold cell missing from checkpoint"
    assert within["null_draws_used"] == args.null_draws, (
        "within cell null draws were descoped — forbidden (plan section 9)",
        within["null_draws_used"],
    )
    pseudo_rows = [
        by_key[(CELL_PSEUDO, f"pseudo_k5_draw{s}")]
        for s in range(args.pseudo_draws)
        if (CELL_PSEUDO, f"pseudo_k5_draw{s}") in by_key
    ]
    assert len(pseudo_rows) == args.pseudo_draws, (
        f"pseudo-group control incomplete ({len(pseudo_rows)}/{args.pseudo_draws} draws) — "
        "never descoped (plan section 9); the registered read is undefined without it"
    )
    loao = by_key.get((CELL_LOAO, LOAO_SCHEME))
    if loao is None:
        assert descope.get("loao_dropped"), (
            "LOAO cell missing from checkpoint without a recorded descope — run --phases loao"
        )

    refs = committed_reference()
    r_a = float(within["r2_l19"])
    band_a = within["null_p975_l19"]
    assert band_a is not None, "within-cell null p975 missing — the decision rule is undefined"
    ci = within["bootstrap_group_l19"]
    read = registered_read(r_a, float(band_a), float(ci["ci_lo"]), float(ci["ci_hi"]))

    per_draw = [float(r["r2_l19"]) for r in pseudo_rows]
    mean_pseudo = float(np.mean(per_draw))
    sd_pseudo = float(np.std(per_draw, ddof=1))
    author_level = mean_pseudo - r_a
    decomposition = {
        "coarsening_component": H_AMEND_NUMERATOR - mean_pseudo,
        "author_level_component": author_level,
        "sd_pseudo": sd_pseudo,
        # Row-1 attribution gate (section 6): author-level claimed only if
        # the component exceeds the pseudo-draw spread.
        "author_level_resolvable": bool(author_level > sd_pseudo),
    }

    payload = {
        "metadata": common.metadata(
            SCRIPT,
            common.FIT_SEED,
            EXPECTED_N,
            extra={"fit_seed": common.FIT_SEED, "n_folds": common.N_FOLDS, "smoke": args.smoke},
        ),
        "pins": {"store_revision": STORE_REV, "pdnc_sha": common.PDNC_SHA},
        "protocol": {
            "protocol_fingerprint": fp,
            "fingerprint_basis": fingerprint_basis(mp["driver_git_sha"], args),
            "layers_kept": list(layers_kept),
        },
        "descope": descope,
        "mapping": {
            "novel_to_author": mp["novel_to_author"],
            "per_author_rows": mp["per_author_rows"],
            "per_author_novels": mp["per_author_novels"],
        },
        "folds": {"scheme": FOLD_SCHEME, "composition": mp["fold_composition"]},
        "cells": {
            "novel_fold_refs": refs,
            "armA_within_authorfold": {
                "r2_per_layer": within["r2_per_layer"],
                "r2_l19": r_a,
                "null_p975_l19": band_a,
                "bootstrap": ci,
                "per_author_r2": within["per_author_r2"],
                "null_draws": within["null_draws_used"],
            },
            "armA_ctxmean_authorfold": {
                "r2_per_layer": ctxmean["r2_per_layer"],
                "r2_l19": float(ctxmean["r2_l19"]),
                "null_p975_l19": ctxmean["null_p975_l19"],
                "bootstrap": ctxmean["bootstrap_group_l19"],
                "null_draws": ctxmean["null_draws_used"],
            },
        },
        "loao_secondary": (
            {"r2_l19": float(loao["r2_l19"]), "n_folds": LOAO_FOLDS}
            if loao is not None
            else {"dropped_by_descope": True}
        ),
        "pseudo_group_null": {
            "n_draws": args.pseudo_draws,
            "size_multiset": list(EXPECTED_MULTISET),
            "per_draw_r2_l19": per_draw,
            "mean": mean_pseudo,
            "sd": sd_pseudo,
        },
        "decomposition": decomposition,
        "ratio": {
            "ctxmean_over_within_novel": refs["ratio_ctxmean_over_within_novel"],
            "ctxmean_over_within_author": float(ctxmean["r2_l19"]) / r_a,
        },
        "transfer_redivision": transfer_redivision(within["r2_per_layer"], layers_kept),
        "registered_read": read,
    }
    common.write_json(out_dir / "author_blocked_folds.json", payload)
    make_figure(payload, Path(args.fig_dir))
    print(
        f"[i931-abf] registered_read: {read['decision_row']} "
        f"R_a={r_a:.4f} band={band_a:.4f} CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}] "
        f"pseudo_mean={mean_pseudo:.4f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Figure (paper-plots conventions; per-point values embedded in .meta.json)
# ---------------------------------------------------------------------------


def make_figure(payload: dict, fig_dir: Path) -> None:
    """(left) L19 pooled R^2 novel-fold vs author-fold for both cells with
    shuffle-band p975 + bootstrap CIs; (right) per-author held-out R^2, 19
    labeled points (the low-level per-unit plot)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    colors = paper_palette(3)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.4, 4.4), layout="constrained")

    refs = payload["cells"]["novel_fold_refs"]
    cells = [
        ("within\nnovel folds", refs["armA_within"]),
        ("within\nauthor folds", payload["cells"]["armA_within_authorfold"]),
        ("ctxmean\nnovel folds", refs["armA_ctxmean"]),
        ("ctxmean\nauthor folds", payload["cells"]["armA_ctxmean_authorfold"]),
    ]
    meta_points: dict = {
        "l19": {},
        "per_author_r2": payload["cells"]["armA_within_authorfold"]["per_author_r2"],
    }
    for i, (label, c) in enumerate(cells):
        boot = c.get("bootstrap") or c.get("bootstrap_group_l19")
        ax_l.scatter([i - 0.12], [c["r2_l19"]], s=34, color=colors[0], zorder=3)
        ax_l.errorbar(
            [i + 0.12],
            [boot["obs"]],
            yerr=[[boot["obs"] - boot["ci_lo"]], [boot["ci_hi"] - boot["obs"]]],
            fmt="D",
            ms=4,
            color=colors[1],
            capsize=3,
            zorder=3,
        )
        p975 = c.get("null_p975_l19")
        if p975 is not None:
            ax_l.hlines(p975, i - 0.28, i + 0.28, color=colors[2], lw=1.4)
        meta_points["l19"][label.replace("\n", " ")] = {
            "sweep_obs": round(float(c["r2_l19"]), 6),
            "bootstrap_obs": round(float(boot["obs"]), 6),
            "ci": [round(float(boot["ci_lo"]), 6), round(float(boot["ci_hi"]), 6)],
            "null_p975": None if p975 is None else round(float(p975), 6),
        }
    ax_l.scatter([], [], s=34, color=colors[0], label="held-out pooled R$^2$ (sweep)")
    ax_l.errorbar(
        [],
        [],
        yerr=[],
        fmt="D",
        ms=4,
        color=colors[1],
        capsize=3,
        label="group bootstrap obs + 95% CI",
    )
    ax_l.hlines([], [], [], color=colors[2], lw=1.4, label="shuffle-null p97.5")
    ax_l.set_xticks(range(len(cells)), [c[0] for c in cells])
    ax_l.set_ylabel("held-out pooled R$^2$ (layer 19)")
    ax_l.legend(fontsize=7, loc="lower left")

    per_author = payload["cells"]["armA_within_authorfold"]["per_author_r2"]
    order = sorted(per_author, key=lambda a: per_author[a])
    ax_r.scatter([per_author[a] for a in order], range(len(order)), s=22, color=colors[0])
    ax_r.axvline(0.0, color="0.6", lw=0.8)
    ax_r.set_yticks(range(len(order)), order, fontsize=7)
    ax_r.set_xlabel("held-out R$^2$ within author (layer 19)")
    ax_r.set_ylabel("PDNC author code")

    fig.savefig(fig_dir / "author_blocked_folds.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "author_blocked_folds.pdf", bbox_inches="tight")
    comp = payload["folds"]["composition"]
    comp_txt = "; ".join(
        f"fold {c['fold']}: {c['n_rows']} rows / {c['n_authors']} authors" for c in comp
    )
    (fig_dir / "author_blocked_folds.meta.json").write_text(
        json.dumps(
            {
                "metadata": common.metadata(SCRIPT, common.FIT_SEED, EXPECTED_N),
                "what": (
                    "L19 held-out pooled R^2 of the two Arm-A cells under novel-blocked vs "
                    "author-blocked K=5 folds (sweep obs, group-bootstrap obs + 95% CI, "
                    "shuffle-null p97.5), and the per-author held-out R^2 (19 labeled points). "
                    f"Author folds are imbalanced by construction (honest blocking): {comp_txt}."
                ),
                "points": meta_points,
            },
            indent=2,
            default=float,
        )
    )
    plt.close(fig)
    print(f"[i931-abf] wrote {fig_dir / 'author_blocked_folds.png'}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(SCRIPTS.parent / "eval_results" / "issue_931"))
    ap.add_argument("--fig-dir", default=str(SCRIPTS.parent / "figures" / "issue_931"))
    ap.add_argument("--data-dir", default=str(SCRIPTS.parent / "data" / "issue_931"))
    ap.add_argument(
        "--stage-dir",
        default=str(SCRIPTS.parent / "data" / "issue_931" / "hf_dl" / "authorfolds"),
    )
    ap.add_argument("--pdnc-dir", default=str(SCRIPTS.parent / "data" / "issue_931" / "pdnc"))
    ap.add_argument(
        "--checkpoint", default=None, help="cells JSONL (default <out-dir>/..._cells.jsonl)"
    )
    ap.add_argument(
        "--phases",
        default="all",
        help=f"comma list of {','.join(PHASES_ALL)} (default all; resume via checkpoint)",
    )
    ap.add_argument("--null-draws", type=int, default=common.N_NULL_DRAWS)
    ap.add_argument("--n-boot", type=int, default=common.N_BOOTSTRAP)
    ap.add_argument("--pseudo-draws", type=int, default=20)
    ap.add_argument("--budget-hours", type=float, default=4.5, help="plan section-9 budget")
    ap.add_argument(
        "--protocol-tag",
        default="",
        help="extra string folded into the protocol fingerprint (protocol bumps / smoke)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny slice: layers {18,19}, null_draws=2, n_boot=20, pseudo_draws=2",
    )
    args = ap.parse_args(argv)
    if args.smoke:
        args.null_draws, args.n_boot, args.pseudo_draws = 2, 20, 2
        args.layers_kept = tuple(SMOKE_LAYERS)
    else:
        args.layers_kept = tuple(range(common.EXPECTED_LAYERS))
    return args


def _resolve_phases(spec: str) -> tuple[str, ...]:
    if spec == "all":
        return PHASES_ALL
    phases = tuple(p.strip() for p in spec.split(",") if p.strip())
    for p in phases:
        assert p in PHASES_ALL, f"unknown phase {p!r} (choose from {PHASES_ALL})"
    return phases


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    phases = _resolve_phases(args.phases)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = (
        Path(args.checkpoint) if args.checkpoint else out_dir / "author_blocked_folds_cells.jsonl"
    )
    runconfig_path = ckpt.with_name(ckpt.stem + "_runconfig.json")

    import torch

    caps = {
        k: os.environ.get(k)
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    print(f"[i931-abf] thread caps {caps} torch_threads={torch.get_num_threads()}", flush=True)

    git_sha = common.git_commit()
    fp = protocol_fingerprint(git_sha, args)
    print(
        f"[i931-abf] phases={phases} smoke={args.smoke} fingerprint {fp} "
        f"(git {git_sha[:12]}, store rev {STORE_REV[:12]}, pdnc {common.PDNC_SHA[:12]})",
        flush=True,
    )
    by_key = load_checkpoint(ckpt, fp)
    if by_key:
        print(f"[i931-abf] resume: {len(by_key)} completed cells (exact fingerprint)", flush=True)
    descope = read_runconfig(runconfig_path, fp)

    store_dir = Path(args.stage_dir) / STORE_PREFIX
    if "stage" in phases:
        print("[phase=stage]", flush=True)
        store_dir = stage_store(Path(args.stage_dir))

    need_store = [p for p in ("mapping", "fits", "loao", "pseudo") if p in phases]
    store: dict | None = None
    mapping: dict | None = None
    if need_store:
        assert store_dir.exists(), f"store dir missing: {store_dir} (run --phases stage first)"
        store = load_store(store_dir, args.layers_kept)
        mapping = build_author_mapping(store, Path(args.pdnc_dir))

    if "mapping" in phases:
        print("[phase=mapping]", flush=True)
        comp = fold_composition(mapping["author_ids"], store["group_ids"])
        for a in sorted(mapping["per_author_rows"]):
            print(
                f"[i931-abf]   author {a}: {mapping['per_author_rows'][a]} rows / "
                f"{mapping['per_author_novels'][a]} novels",
                flush=True,
            )
        for c in comp:
            print(
                f"[i931-abf]   fold {c['fold']}: {c['n_rows']} rows, {c['n_authors']} authors, "
                f"{c['n_novels']} novels",
                flush=True,
            )
        common.write_json(
            out_dir / "author_blocked_folds_mapping.json",
            {
                "metadata": common.metadata(SCRIPT, common.FIT_SEED, EXPECTED_N),
                "protocol_fingerprint": fp,
                "driver_git_sha": git_sha,
                "layers_kept": [int(v) for v in args.layers_kept],
                "pins": {"store_revision": STORE_REV, "pdnc_sha": common.PDNC_SHA},
                "novel_to_author": mapping["novel_to_author"],
                "per_author_rows": mapping["per_author_rows"],
                "per_author_novels": mapping["per_author_novels"],
                "size_multiset": list(mapping["sizes_desc"]),
                "fold_composition": comp,
            },
        )

    if "fits" in phases:
        print("[phase=fits]", flush=True)
        descope = run_fits(store, mapping, args, fp, ckpt, by_key, runconfig_path, descope)
    if "loao" in phases:
        print("[phase=loao]", flush=True)
        if descope.get("loao_dropped"):
            print("[i931-abf] LOAO dropped by the recorded descope — skipping", flush=True)
        else:
            run_loao(store, mapping, args, fp, ckpt, by_key)
    if "pseudo" in phases:
        print("[phase=pseudo]", flush=True)
        run_pseudo(store, mapping, args, fp, ckpt, by_key)
    if "aggregate" in phases:
        print("[phase=aggregate]", flush=True)
        aggregate(args, fp, by_key, descope)
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
