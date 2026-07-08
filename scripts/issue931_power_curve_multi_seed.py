"""Issue #931 follow-up `matched-n-denominator-dip`: multi-seed chat power curve.

Re-estimates the Track-S chat ceiling at n in {1000, 1500, 1982, 2500, 3000}
x subsample seeds {931..935} (5 independent group-stratified draws per n) to
decide whether the committed single-draw dip at n=1982 (L19 held-out R^2 =
0.3162 — the H1 existence-strength denominator behind the 0.55x read) is a
subsample artifact or a stable estimator-at-n property. Plan v7
(tasks/.../931/plans/v7.md sections 4/6/9/10).

Protocol — everything inherited, no new estimator code:
  - pinned #825 Track-S turnstore shards @ revision 82d3a875 (run-manifest
    pin; re-asserted via HfApi().file_exists at staging), staged with scoped
    ``list_repo_tree(path_in_repo=...)`` + per-file ``hf_hub_download`` into
    the worktree data dir (lives on /mnt/eps-data);
  - ``issue825_fit_cells._load_bundle_any`` + ``_cell_xy`` with the Track-S
    S1 cell spec (assistant slot -> a1 profile), all 28 layers;
  - ``issue931_common.group_stratified_subsample(conv_ids, n, seed=s)`` (the
    run's own matched-power subsampler; the seed-931 n=1982 instance is the
    committed sep-control precedent);
  - ``heldout_r2_sweep(..., n_folds=5, seed=0, null_draws=0,
    collect_cosines=False, collect_lambdas=True)`` — cached-eigh Gram-ridge,
    one Y-pass per fold-layer; ``collect_lambdas`` is the single registered
    default-preserving source-module change (per-fold-layer GCV lambda paths,
    the dip mechanism read).

Checkpoint/resume: per-cell atomic JSONL append keyed on
(seed, n, protocol_fingerprint) where protocol_fingerprint = sha256(driver
git SHA + estimator constants + subsample scheme id + store revision)[:12].
Resume skips a completed cell ONLY on an exact fingerprint match; ANY
stale-fingerprint row in the checkpoint fails loud (no silent mixing of
stale-protocol cells into the draw mean/SD). Aggregation asserts EXACTLY ONE
current-fingerprint cell per expected (seed, n). Fewer than 5 decision-n
(n=1982, seeds 931..935) cells => registered_read status INCOMPLETE and NO
supersession of the committed denominators (plan v7 section 9: seed descope
at the decision n is FORBIDDEN).

Compute: 0 GPU-h, VM CPU (thread caps supplied by the launcher env prefix);
after the first fitted cell the driver projects the battery wall from the
plan-section-9 FLOP model and applies the registered descope if the
projection exceeds 2x the ~1.5 h budget: drop n in {2500, 3000} first, then
n=1500 — NEVER seeds, never the {1000, 1982} anchors.

Outputs: <out-dir>/power_curve_multi_seed.json (+ the cells JSONL checkpoint)
and figures/issue_931/power_curve_multi_seed.{png,pdf,meta.json}. Committed
reference inputs (power_curve_chat.json, transfer_matrix.json,
cells_chat_ref.json) are read from the repo's canonical eval_results dir — a
NON-rebinding constant, deliberately independent of --out-dir so a
scratch-dir smoke never orphans them.

CLI:
  uv run python scripts/issue931_power_curve_multi_seed.py \
      [--out-dir eval_results/issue_931] [--fig-dir figures/issue_931] \
      [--stage-dir data/issue_931/hf_dl/pcms] \
      [--seeds 931,932,933,934,935] [--ns 1000,1500,1982,2500,3000] \
      [--protocol-tag ""] [--budget-hours 1.5] [--aggregate-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/torch import

import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402

SCRIPT = "scripts/issue931_power_curve_multi_seed.py"
CHAT_REV = "82d3a875ee5148e45df982fd51a3c4dea1055fb7"  # run-manifest pin (plan v7 section 10)
SEEDS_DEFAULT = (931, 932, 933, 934, 935)
NS_DEFAULT = (1000, 1500, 1982, 2500, 3000)
DECISION_N = 1982  # the matched H1 denominator n (n_A)
# Committed armA_within L19 within-ceiling (transfer_matrix.json; plan section 3 / A5).
H1_NUMERATOR = 0.17289959611807892
SD_STABLE_MAX = 0.05  # registered small-spread bar (plan section 11)
MEAN_ARTIFACT_MIN = 0.4  # registered draw-artifact boundary (plan section 3/6)
SUBSAMPLE_SCHEME_ID = "issue931_pcms.seeded_uniform_row_draw.v1"
HIDDEN = common.EXPECTED_HIDDEN  # 3584 (FLOP model below)
# Committed reference inputs — canonical repo paths, NOT rebound by --out-dir
# (smoke-root rebinding must never orphan read-only committed inputs).
COMMITTED_EVAL_DIR = SCRIPTS.parent / "eval_results" / "issue_931"

DESCOPE_PRIORITY = ((2500, 3000), (1500,))  # plan section 9; {1000, 1982} anchors never drop


def estimator_constants() -> dict:
    """The estimator identity folded into the protocol fingerprint."""
    return {
        "lambdas": [float(v) for v in fit825.LAMBDAS],
        "n_folds": int(common.N_FOLDS),
        "fit_seed": int(common.FIT_SEED),
        "null_draws": 0,
        "headline_layer": int(common.HEADLINE_LAYER),
        "expected_layers": int(common.EXPECTED_LAYERS),
    }


def fingerprint_basis(
    git_sha: str, *, protocol_tag: str = "", store_revision: str = CHAT_REV
) -> dict:
    return {
        "driver_git_sha": git_sha,
        "estimator": estimator_constants(),
        "subsample_scheme": SUBSAMPLE_SCHEME_ID,
        "store_revision": store_revision,
        "protocol_tag": protocol_tag,
    }


def protocol_fingerprint(
    git_sha: str, *, protocol_tag: str = "", store_revision: str = CHAT_REV
) -> str:
    """sha256(driver git SHA + estimator constants + subsample scheme id +
    store revision)[:12] — the checkpoint/resume protocol key (plan section 4
    step 5). ``protocol_tag`` is an explicit extra basis string for deliberate
    protocol bumps (and the smoke's fingerprint-mismatch exercise)."""
    basis = json.dumps(
        fingerprint_basis(git_sha, protocol_tag=protocol_tag, store_revision=store_revision),
        sort_keys=True,
    )
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


def load_checkpoint(path: Path, fp: str) -> dict[tuple[int, int], dict]:
    """Load completed cells keyed (seed, n); fail loud on ANY stale
    fingerprint or duplicate (seed, n) — no silent protocol mixing."""
    if not path.exists():
        return {}
    rows: list[dict] = []
    for line in path.open(encoding="utf-8"):  # file iteration, never splitlines()
        if line.strip():
            rows.append(json.loads(line))
    stale = sorted({r.get("protocol_fingerprint", "<missing>") for r in rows})
    stale = [s for s in stale if s != fp]
    if stale:
        raise RuntimeError(
            f"checkpoint {path} holds stale-protocol cells (fingerprints {stale}, current {fp}); "
            "refusing to resume or mix — move/delete the checkpoint to rerun under the new protocol"
        )
    by_key: dict[tuple[int, int], dict] = {}
    for r in rows:
        key = (int(r["seed"]), int(r["n"]))
        if key in by_key:
            raise RuntimeError(f"duplicate checkpoint cell for seed={key[0]} n={key[1]} in {path}")
        by_key[key] = r
    return by_key


def write_runconfig(path: Path, fp: str, seeds: tuple[int, ...], ns_active: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {"protocol_fingerprint": fp, "seeds": list(seeds), "ns_active": list(ns_active)},
            indent=2,
        )
    )
    os.replace(tmp, path)


def resolve_ns_active(
    runconfig_path: Path,
    fp: str,
    seeds: tuple[int, ...],
    ns_cli: tuple[int, ...],
    checkpoint_nonempty: bool,
) -> list[int]:
    """The active n grid: the runconfig's (fingerprint-matched) on resume,
    the CLI grid on a fresh run. A CLI/runconfig conflict over a non-empty
    checkpoint fails loud rather than silently re-scoping a partial run."""
    if runconfig_path.exists():
        rc = json.loads(runconfig_path.read_text())
        if rc.get("protocol_fingerprint") != fp:
            raise RuntimeError(
                f"runconfig {runconfig_path} fingerprint {rc.get('protocol_fingerprint')} != "
                f"current {fp}; move/delete it (and the checkpoint) to rerun under the new protocol"
            )
        if checkpoint_nonempty:
            if list(rc.get("seeds", [])) != list(seeds) or not set(rc["ns_active"]) <= set(ns_cli):
                raise RuntimeError(
                    f"CLI grid (seeds={list(seeds)}, ns={list(ns_cli)}) conflicts with the resumed "
                    f"runconfig {rc}; pass the original grid or clear the checkpoint"
                )
            return [int(v) for v in rc["ns_active"]]
    return [int(v) for v in ns_cli]


# ---------------------------------------------------------------------------
# Staging + loading (pinned revision, scoped Hub calls)
# ---------------------------------------------------------------------------


def stage_store(stage_dir: Path) -> Path:
    """Stage the 10 pinned Track-S .pt shards; returns the turnstore dir.

    Scoped per-file staging (list_repo_tree with path_in_repo + per-file
    hf_hub_download — never snapshot_download on the ~1M-file data repo);
    hf_hub_download skips files already staged at the pinned revision.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    shard_paths = sorted(
        e.path
        for e in api.list_repo_tree(
            common.HF_DATA_REPO,
            path_in_repo=common.CHAT_STORE_PREFIX,
            repo_type="dataset",
            revision=CHAT_REV,
        )
        if Path(e.path).name.startswith(common.CHAT_STORE_STEM + "_shard")
        and e.path.endswith(".pt")
    )
    assert len(shard_paths) == 10, (
        f"expected 10 chat .pt shards at {CHAT_REV}, got {len(shard_paths)}"
    )
    # A2 re-assert (plan section 12): the pinned revision resolves the shards.
    assert api.file_exists(
        common.HF_DATA_REPO, shard_paths[0], repo_type="dataset", revision=CHAT_REV
    ), f"pinned shard missing at {CHAT_REV}: {shard_paths[0]}"
    for p in shard_paths:
        print(f"[i931-pcms] stage {p}", flush=True)
        hf_hub_download(
            common.HF_DATA_REPO, p, repo_type="dataset", revision=CHAT_REV, local_dir=stage_dir
        )
    return stage_dir / common.CHAT_STORE_PREFIX


S1_SLOT_INDEX = 0  # assistant slot (issue825 _normalize_cell Track-S default)
S1_TARGET_TURN_INDEX = 1  # a1 profile


def _s1_slices_from_shard(shard_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One shard's (X, Y, conv_ids) for the S1 cell, keep-masked.

    Slices each record to the S1 cell (slot 0 -> X, turn-1 profile -> Y,
    bf16 -> fp32 via torch .float(), the reference chain's cast) BEFORE
    stacking, then applies `_cell_xy`'s all-layer NaN keep-mask — computed
    on exactly these sliced arrays, so the kept row set matches the
    reference chain (asserted by --verify-loader below).
    """
    import torch

    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    conv_ids = np.asarray(payload["conv_ids"])
    first_slot = torch.as_tensor(payload["slots"][0])
    first_prof = torch.as_tensor(payload["profiles"][0])
    assert first_slot.shape[0] > S1_SLOT_INDEX, first_slot.shape
    assert first_prof.shape[0] > S1_TARGET_TURN_INDEX, first_prof.shape
    x_full = np.stack([torch.as_tensor(t)[S1_SLOT_INDEX].float().numpy() for t in payload["slots"]])
    y_full = np.stack(
        [torch.as_tensor(t)[S1_TARGET_TURN_INDEX].float().numpy() for t in payload["profiles"]]
    )
    del payload
    keep = ~(np.isnan(x_full).any(axis=(1, 2)) | np.isnan(y_full).any(axis=(1, 2)))
    return x_full[keep], y_full[keep], conv_ids[keep]


def load_track_s_xy(turnstore_dir: Path) -> dict:
    """All-28-layer (X, Y, conv_ids) for the Track-S S1 cell, streamed.

    Semantically identical to the inherited `_load_bundle_any` + `_cell_xy`
    chain for this cell (same slices, same bf16->fp32 cast, same all-layer
    NaN keep-mask — the committed sep-control per-shard streaming shape,
    `issue931_sep_to_chat_matched_control.py`, generalized to all 28
    layers), but streams ONE shard at a time: the full-bundle reference
    chain stacks every key of every shard and peaked at ~31 GiB RSS on
    this store — earlyoom-SIGTERMed on the shared VM (smoke run 1,
    2026-07-04). Peak here is ~one shard + the accumulated fp32 X/Y.
    Shard-level equivalence against the reference chain is asserted by
    --verify-loader.
    """
    shards = sorted(turnstore_dir.glob(f"{common.CHAT_STORE_STEM}_shard*.pt"))
    assert len(shards) == 10, f"expected 10 staged shards in {turnstore_dir}, got {len(shards)}"
    xs, ys, ids = [], [], []
    for sp in shards:
        x, y, cid = _s1_slices_from_shard(sp)
        xs.append(x)
        ys.append(y)
        ids.append(cid)
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    conv_ids = np.concatenate(ids)
    assert X.shape[0] == 5000, f"kept-row count {X.shape[0]} != 5000 (run invariant)"
    assert X.shape[1] == common.EXPECTED_LAYERS, X.shape
    assert X.shape[2] == HIDDEN, X.shape
    return {"X": X, "Y": Y, "conv_ids": conv_ids}


def verify_loader_equivalence(turnstore_dir: Path) -> None:
    """Assert the streaming S1 loader reproduces the inherited
    `_load_bundle_any` + `_cell_xy` chain EXACTLY on shard000 (the
    reference chain is safe to run on one ~2 GB shard; the full-store run
    is what earlyoom kills). Fails loud on any array mismatch."""
    import tempfile

    from explore_persona_space.experiments.issue_825 import common as common825

    shard0 = turnstore_dir / f"{common.CHAT_STORE_STEM}_shard000.pt"
    assert shard0.exists(), shard0
    cell = fit825._normalize_cell(dict(common825.TRACK_S_CELLS[0]))
    assert cell["cell_id"] == "S1" and cell["model_key"] == "instruct", cell
    assert int(cell["slot_index"]) == S1_SLOT_INDEX, cell
    assert int(cell["target_turn_index"]) == S1_TARGET_TURN_INDEX, cell
    with tempfile.TemporaryDirectory() as td:
        link = Path(td) / shard0.name
        link.symlink_to(shard0)
        bundle = fit825._load_bundle_any(
            Path(td), cell["model_key"], cell["format_key"], cell["track"]
        )
        ref = fit825._cell_xy(bundle, cell)
        del bundle
    x, y, cid = _s1_slices_from_shard(shard0)
    assert np.array_equal(ref["X"], x), "streaming loader X != reference chain X (shard000)"
    assert np.array_equal(ref["Y"], y), "streaming loader Y != reference chain Y (shard000)"
    assert list(np.asarray(ref["conv_ids"])) == list(np.asarray(cid)), "conv_ids mismatch"
    print(
        f"[i931-pcms] loader equivalence PASS on {shard0.name} ({x.shape[0]} kept rows)", flush=True
    )


# ---------------------------------------------------------------------------
# Per-cell fit + the section-9 FLOP model / descope
# ---------------------------------------------------------------------------


def cell_flops(
    n: int, n_folds: int = common.N_FOLDS, n_layers: int = common.EXPECTED_LAYERS
) -> float:
    """Plan section-9 per-cell fp64 FLOP model: n_tr = (1-1/K)*n;
    eigh ~ 9*n_tr^3 + Gram/VtY ~ 4*n_tr^2*HIDDEN, x (layers*folds)."""
    n_tr = n * (1.0 - 1.0 / n_folds)
    return (9.0 * n_tr**3 + 4.0 * n_tr**2 * HIDDEN) * n_layers * n_folds


def draw_subsample(conv_ids: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Seeded uniform without-replacement row draw for one power-curve cell.

    The plan-registered subsampler `issue931_common.group_stratified_subsample`
    is SEED-DEGENERATE on an all-singleton group store: with every group
    count = 1 and n_target < n_groups, all proportional quotas floor to 0
    and the largest-remainder tie-break (STABLE argsort over EQUAL
    remainders) selects the lexicographically-first n groups regardless of
    seed. Caught live by smoke run 1d (2026-07-04): seeds 931 and 932
    produced byte-identical subsets and 28-layer curves on the chat store
    (1 row per conversation), which would have collapsed all 5 registered
    draws to copies of one draw (SD = 0 mechanically). Plan v7 section 12
    A4 ("harmless either way") is falsified. On a singleton-group store a
    group-stratified draw semantically REDUCES to a uniform row draw, so
    this battery draws rows uniformly (seeded, sorted indices — the
    subsampler's output convention); the all-singleton premise is hard-
    asserted so any group structure fails loud instead of silently
    changing the draw semantics. The committed
    `group_stratified_subsample` is left byte-untouched (its stable
    tie-break is pinned by committed multi-row-group results).
    """
    conv_ids = np.asarray(conv_ids)
    n_rows = len(conv_ids)
    assert len(np.unique(conv_ids)) == n_rows, (
        "chat store expected all-singleton (1 row/conv); found group structure — "
        "the uniform row draw is no longer the group-stratified reduction; revisit the scheme"
    )
    if n >= n_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=n, replace=False))


def fit_cell(
    X: np.ndarray, Y: np.ndarray, conv_ids: np.ndarray, seed: int, n: int, fp: str
) -> dict:
    """One (seed, n) cell: seeded row subsample + the inherited 28-layer
    held-out sweep with lambda-path collection."""
    idx = draw_subsample(conv_ids, n, seed)
    assert len(idx) == n, (len(idx), n)
    sub_ids = np.asarray(conv_ids)[idx]
    sw = fit825.heldout_r2_sweep(
        X[idx],
        Y[idx],
        sub_ids,
        n_folds=common.N_FOLDS,
        seed=common.FIT_SEED,
        null_draws=0,
        collect_cosines=False,
        collect_lambdas=True,
    )
    r2 = sw["r2_obs"]
    lam = sw["gcv_lambda"]
    folds = sw["folds"]
    uniq, counts = np.unique(sub_ids, return_counts=True)
    return {
        "protocol_fingerprint": fp,
        "seed": int(seed),
        "n": int(n),
        "r2_per_layer": [float(v) for v in r2],
        "r2_l19": float(r2[common.HEADLINE_LAYER]),
        "gcv_lambda_per_layer_fold": [[float(v) for v in row] for row in lam],
        "gcv_lambda_l19_per_fold": [float(v) for v in lam[common.HEADLINE_LAYER]],
        "n_convs": len(uniq),
        "rows_per_conv_mean": float(counts.mean()),
        "rows_per_conv_max": int(counts.max()),
        "fold_sizes": [int((folds == k).sum()) for k in range(common.N_FOLDS)],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def project_battery_hours(
    first_n: int, first_wall_s: float, seeds: tuple[int, ...], ns_active: list[int]
) -> float:
    """Battery wall projected from the first timed cell via the FLOP model."""
    per_flop = first_wall_s / cell_flops(first_n)
    total = sum(cell_flops(n) * len(seeds) for n in ns_active)
    return per_flop * total / 3600.0


def apply_descope(
    ns_active: list[int],
    seeds: tuple[int, ...],
    first_n: int,
    first_wall_s: float,
    budget_hours: float,
) -> tuple[list[int], dict]:
    """Section-9 descope: if projected battery wall > 2x budget, drop
    {2500, 3000} then {1500}. Seeds and the {1000, 1982} anchors NEVER drop."""
    threshold = 2.0 * budget_hours
    projected = project_battery_hours(first_n, first_wall_s, seeds, ns_active)
    info = {
        "first_cell_n": first_n,
        "first_cell_wall_s": first_wall_s,
        "projected_battery_hours": projected,
        "budget_hours": budget_hours,
        "threshold_hours": threshold,
        "dropped_ns": [],
    }
    ns = list(ns_active)
    for tier in DESCOPE_PRIORITY:
        if projected <= threshold:
            break
        drop = [n for n in tier if n in ns]
        if not drop:
            continue
        ns = [n for n in ns if n not in drop]
        info["dropped_ns"].extend(drop)
        projected = project_battery_hours(first_n, first_wall_s, seeds, ns)
    info["projected_battery_hours_after_descope"] = projected
    # Anchors present in the incoming grid must survive the descope (the
    # priority tiers can only name 1500/2500/3000, so this is an invariant
    # guard, conditional so subset grids without an anchor stay valid).
    anchors = (1000, DECISION_N)
    assert all(a in ns for a in anchors if a in ns_active), (
        "descope must never drop the {1000, 1982} anchors"
    )
    return ns, info


# ---------------------------------------------------------------------------
# Aggregation + the registered read (plan v7 section 6 decision table)
# ---------------------------------------------------------------------------


def aggregate(
    by_key: dict[tuple[int, int], dict], seeds: tuple[int, ...], ns_active: list[int]
) -> dict:
    """Per-n draw stats; asserts EXACTLY ONE current-fingerprint cell per
    expected (seed, n) — extra/missing cells fail loud (plan section 4 step 5)."""
    expected = {(s, n) for s in seeds for n in ns_active}
    got = set(by_key)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise RuntimeError(
            f"cell-set mismatch at aggregation: missing={missing} extra={extra} — exactly one "
            "current-fingerprint cell per expected (seed, n) is required"
        )
    per_n: dict[str, dict] = {}
    for n in ns_active:
        vals = np.array([by_key[(s, n)]["r2_l19"] for s in seeds], dtype=np.float64)
        curves = np.array([by_key[(s, n)]["r2_per_layer"] for s in seeds], dtype=np.float64)
        per_n[str(n)] = {
            "l19_mean": float(vals.mean()),
            "l19_sd": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "l19_min": float(vals.min()),
            "l19_max": float(vals.max()),
            "l19_values": [float(v) for v in vals],
            "r2_per_layer_mean": [float(v) for v in curves.mean(axis=0)],
        }
    return per_n


def registered_read(draw_mean: float, draw_sd: float, *, numerator: float = H1_NUMERATOR) -> dict:
    """Pure decision function for the plan-v7 section-6 table (L19, n=1982).

    Rows (in evaluation order):
      draw_mean >= 0.4                          -> dip_draw_artifact (any SD)
      SD <= 0.05 and draw_mean <= 2*numerator   -> dip_stable_above_bar
      SD <= 0.05 and 2*numerator < mean < 0.4   -> dip_stable_below_bar
      SD > 0.05 and draw_mean < 0.4             -> draw_noisy
    The H1 strength clause is ALWAYS computed as numerator/draw_mean (never
    carried over verbatim); 2*numerator = 0.34579919... is the 0.5x crossover.
    """
    crossover = 2.0 * numerator
    fraction = numerator / draw_mean
    if draw_mean >= MEAN_ARTIFACT_MIN:
        decision = "dip_draw_artifact"
    elif draw_sd <= SD_STABLE_MAX:
        decision = "dip_stable_above_bar" if draw_mean <= crossover else "dip_stable_below_bar"
    else:
        decision = "draw_noisy"
    return {
        "h1_numerator": numerator,
        "crossover_draw_mean": crossover,
        "draw_mean": float(draw_mean),
        "draw_sd": float(draw_sd),
        "h1_fraction_draw_avg": float(fraction),
        "above_half_bar": bool(fraction >= 0.5),
        "decision": decision,
    }


def superseded_transfer_rows(transfer_matrix: dict, draw_mean: float) -> list[dict]:
    """Re-divide the four committed L19 transfer rows whose denominator is
    chat_ref @ n_train=1982 by the draw-mean ceiling (pure arithmetic — the
    committed transfer_r2 numerators are quoted verbatim, no refit)."""
    rows = []
    for r in transfer_matrix["rows"]:
        if (
            r.get("layer") == 19
            and r.get("denominator_cell") == "chat_ref"
            and r.get("denominator_n_train") == 1982
        ):
            rows.append(
                {
                    "direction": r["direction"],
                    "x_recipe": r["x_recipe"],
                    "application": r["application"],
                    "transfer_r2_committed": r["transfer_r2"],
                    "committed_denominator": r["within_ceiling_r2"],
                    "committed_fraction_of_ceiling": r["fraction_of_ceiling"],
                    "superseded_denominator_draw_mean": float(draw_mean),
                    "superseded_fraction_of_ceiling": float(r["transfer_r2"] / draw_mean),
                }
            )
    assert len(rows) == 4, f"expected exactly 4 committed L19 chat_ref@1982 rows, got {len(rows)}"
    return rows


def build_registered_block(
    by_key: dict[tuple[int, int], dict],
    committed_denominator: float,
    transfer_matrix: dict | None,
) -> dict:
    """The registered_read output block. Requires ALL FIVE registered seeds
    at n=1982 (plan section 9: fewer => INCOMPLETE, no supersession)."""
    block: dict = {
        "h1_numerator": H1_NUMERATOR,
        "h1_numerator_source": "committed armA_within L19 within-ceiling (cells_armA_within.json)",
        "committed_single_draw_denominator": committed_denominator,
        "h1_fraction_single_draw": float(H1_NUMERATOR / committed_denominator),
        "decision_n": DECISION_N,
        "registered_seeds": list(SEEDS_DEFAULT),
    }
    vals = [by_key[(s, DECISION_N)]["r2_l19"] for s in SEEDS_DEFAULT if (s, DECISION_N) in by_key]
    block["n_decision_cells"] = len(vals)
    if len(vals) < len(SEEDS_DEFAULT):
        block.update(
            status="INCOMPLETE",
            supersedes_committed=False,
            note=(
                "fewer than 5 current-protocol (seed in 931..935, n=1982) cells — the run is "
                "INCOMPLETE and does NOT supersede the committed denominators (plan v7 section 9)"
            ),
        )
        return block
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    block.update(status="COMPLETE", supersedes_committed=True, **registered_read(mean, sd))
    if transfer_matrix is not None:
        block["superseded_transfer_rows"] = superseded_transfer_rows(transfer_matrix, mean)
    return block


def committed_reference(eval_dir: Path = COMMITTED_EVAL_DIR) -> dict:
    """Committed single-draw curve (the values being superseded) + the full-n
    chat_ref anchor, read from the canonical committed JSONs."""
    pc = json.loads((eval_dir / "power_curve_chat.json").read_text())
    pts = {
        int(row["n"]): float(row["r2_per_layer"][common.HEADLINE_LAYER])
        for row in pc["curve"]
        if row.get("r2_per_layer")
    }
    assert DECISION_N in pts, f"committed power curve lacks n={DECISION_N}"
    # Drift guard: the committed single-draw denominator the plan registers.
    assert abs(pts[DECISION_N] - 0.31616922814913495) < 1e-9, pts[DECISION_N]
    ref: dict = {"committed_curve_l19": {str(k): v for k, v in sorted(pts.items())}}
    chat_ref_path = eval_dir / "cells_chat_ref.json"
    if chat_ref_path.exists():
        cr = json.loads(chat_ref_path.read_text())
        ref["chat_ref_full_n"] = {
            "n": int(cr["metadata"]["n"]),
            "l19": float(cr["r2_per_layer_obs"][common.HEADLINE_LAYER]),
        }
    return ref


# ---------------------------------------------------------------------------
# Figure (paper-plots conventions; per-draw points embedded in .meta.json)
# ---------------------------------------------------------------------------


def make_figure(
    per_n: dict,
    by_key: dict[tuple[int, int], dict],
    seeds: tuple[int, ...],
    ns_active: list[int],
    ref: dict,
    fig_dir: Path,
) -> None:
    """Per-draw L19 points + draw-mean curve (+/- SD) + the committed
    single-draw curve overlay (+ the full-n chat_ref anchor when present)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    colors = paper_palette(3)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
    meta_points: dict = {"per_draw": {}, "draw_mean": {}, "committed": ref["committed_curve_l19"]}
    for n in ns_active:
        ys = [by_key[(s, n)]["r2_l19"] for s in seeds]
        meta_points["per_draw"][str(n)] = [round(float(v), 6) for v in ys]
        ax.scatter([n] * len(ys), ys, s=18, alpha=0.55, color=colors[0], edgecolors="none")
    means = [per_n[str(n)]["l19_mean"] for n in ns_active]
    sds = [per_n[str(n)]["l19_sd"] for n in ns_active]
    meta_points["draw_mean"] = {
        str(n): {"mean": round(m, 6), "sd": round(sd, 6)}
        for n, m, sd in zip(ns_active, means, sds, strict=True)
    }
    ax.errorbar(
        ns_active,
        means,
        yerr=sds,
        fmt="o-",
        color=colors[1],
        ms=5,
        lw=1.6,
        capsize=3,
        label="draw mean +/- SD (5 group-stratified seeds)",
    )
    cx = sorted(int(k) for k in ref["committed_curve_l19"])
    cy = [ref["committed_curve_l19"][str(k)] for k in cx]
    ax.plot(cx, cy, "s--", color=colors[2], ms=4, lw=1.2, label="committed single-draw curve")
    if "chat_ref_full_n" in ref:
        anchor = ref["chat_ref_full_n"]
        ax.scatter(
            [anchor["n"]],
            [anchor["l19"]],
            marker="D",
            s=30,
            color=colors[2],
            label=f"full-n chat ceiling (n={anchor['n']})",
        )
    ax.set_xlabel("training-set size n (rows)")
    ax.set_ylabel("held-out pooled R$^2$ (layer 19)")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(fig_dir / "power_curve_multi_seed.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "power_curve_multi_seed.pdf", bbox_inches="tight")
    (fig_dir / "power_curve_multi_seed.meta.json").write_text(
        json.dumps(
            {
                "metadata": common.metadata(SCRIPT, common.FIT_SEED, 0),
                "what": (
                    "per-draw L19 held-out R^2 points (5 seeds per n), draw-mean curve +/- SD, "
                    "and the committed single-draw nested-prefix curve being superseded"
                ),
                "points": meta_points,
            },
            indent=2,
            default=float,
        )
    )
    plt.close(fig)
    print(f"[i931-pcms] wrote {fig_dir / 'power_curve_multi_seed.png'}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(SCRIPTS.parent / "eval_results" / "issue_931"))
    ap.add_argument("--fig-dir", default=str(SCRIPTS.parent / "figures" / "issue_931"))
    ap.add_argument(
        "--stage-dir", default=str(SCRIPTS.parent / "data" / "issue_931" / "hf_dl" / "pcms")
    )
    ap.add_argument(
        "--checkpoint", default=None, help="cells JSONL (default <out-dir>/..._cells.jsonl)"
    )
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    ap.add_argument("--ns", default=",".join(str(n) for n in NS_DEFAULT))
    ap.add_argument(
        "--protocol-tag",
        default="",
        help="extra string folded into the protocol fingerprint (protocol bumps / smoke)",
    )
    ap.add_argument("--budget-hours", type=float, default=1.5, help="plan section-9 battery budget")
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="skip staging + fitting; rebuild the output JSON/figure from the checkpoint",
    )
    ap.add_argument(
        "--verify-loader",
        action="store_true",
        help="assert the streaming S1 loader == the _load_bundle_any/_cell_xy chain on shard000",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    ns_cli = tuple(int(v) for v in args.ns.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = (
        Path(args.checkpoint) if args.checkpoint else out_dir / "power_curve_multi_seed_cells.jsonl"
    )
    runconfig_path = ckpt.with_name(ckpt.stem + "_runconfig.json")

    git_sha = common.git_commit()
    fp = protocol_fingerprint(git_sha, protocol_tag=args.protocol_tag)
    print(
        f"[i931-pcms] protocol fingerprint {fp} (git {git_sha[:12]}, rev {CHAT_REV[:12]})",
        flush=True,
    )

    by_key = load_checkpoint(ckpt, fp)
    ns_active = resolve_ns_active(runconfig_path, fp, seeds, ns_cli, bool(by_key))
    if by_key:
        print(
            f"[i931-pcms] resume: {len(by_key)} completed cells (exact fingerprint match)",
            flush=True,
        )

    descope_info: dict = {"applied": False}
    pending = [(s, n) for n in ns_active for s in seeds if (s, n) not in by_key]
    if not args.aggregate_only and (pending or args.verify_loader):
        turnstore = stage_store(Path(args.stage_dir))
        if args.verify_loader:
            verify_loader_equivalence(turnstore)
    if not args.aggregate_only and pending:
        xy = load_track_s_xy(turnstore)
        X, Y, conv_ids = xy["X"], xy["Y"], xy["conv_ids"]
        first_timed: tuple[int, float] | None = None
        i_n = 0
        while i_n < len(ns_active):
            n = ns_active[i_n]
            for s in seeds:
                if (s, n) in by_key:
                    print(f"[i931-pcms] skip completed cell seed={s} n={n}", flush=True)
                    continue
                print(f"[i931-pcms] fit start seed={s} n={n}", flush=True)
                t0 = time.time()
                row = fit_cell(X, Y, conv_ids, s, n, fp)
                wall = time.time() - t0
                row["wall_seconds"] = float(wall)
                append_jsonl(ckpt, row)
                by_key[(s, n)] = row
                print(
                    f"[i931-pcms] cell seed={s} n={n} L19={row['r2_l19']:.4f} wall={wall:.1f}s",
                    flush=True,
                )
                if first_timed is None:
                    first_timed = (n, wall)
                    ns_active, descope_info = apply_descope(
                        ns_active, seeds, n, wall, args.budget_hours
                    )
                    descope_info["applied"] = bool(descope_info["dropped_ns"])
                    write_runconfig(runconfig_path, fp, seeds, ns_active)
                    print(
                        f"[i931-pcms] projected battery "
                        f"{descope_info['projected_battery_hours']:.2f} h "
                        f"(threshold {descope_info['threshold_hours']:.2f} h); "
                        f"dropped_ns={descope_info['dropped_ns']}",
                        flush=True,
                    )
            i_n += 1
        if not runconfig_path.exists():
            write_runconfig(runconfig_path, fp, seeds, ns_active)

    per_n = aggregate(by_key, seeds, ns_active)
    ref = committed_reference()
    transfer_path = COMMITTED_EVAL_DIR / "transfer_matrix.json"
    transfer_matrix = json.loads(transfer_path.read_text()) if transfer_path.exists() else None
    committed_denominator = ref["committed_curve_l19"][str(DECISION_N)]
    registered = build_registered_block(by_key, committed_denominator, transfer_matrix)

    payload = {
        "metadata": common.metadata(
            SCRIPT,
            common.FIT_SEED,
            len(by_key),
            extra={
                "store_revision": CHAT_REV,
                "fit_seed": common.FIT_SEED,
                "n_folds": common.N_FOLDS,
            },
        ),
        "protocol": {
            "subsample": "seeded_uniform_row (group_stratified reduction on all-singleton store)",
            "subsample_scheme_id": SUBSAMPLE_SCHEME_ID,
            "seeds": list(seeds),
            "ns": list(ns_active),
            "null_draws": 0,
            "protocol_fingerprint": fp,
            "fingerprint_basis": fingerprint_basis(git_sha, protocol_tag=args.protocol_tag),
        },
        "descope": descope_info,
        "cells": [by_key[(s, n)] for n in ns_active for s in seeds],
        "aggregate": {"per_n": per_n},
        "committed_reference": ref,
        "registered_read": registered,
    }
    out_json = out_dir / "power_curve_multi_seed.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[i931-pcms] wrote {out_json}", flush=True)
    make_figure(per_n, by_key, seeds, ns_active, ref, Path(args.fig_dir))
    print(
        f"[i931-pcms] registered_read: status={registered['status']} "
        f"decision={registered.get('decision', 'n/a')} "
        f"fraction={registered.get('h1_fraction_draw_avg', float('nan'))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
