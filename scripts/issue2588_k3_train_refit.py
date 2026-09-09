#!/usr/bin/env python3
"""Issue #2588 k3-train-refit round driver (epm:followup-scope v1, 2026-09-09).

For the ten paper-appendix no-thinking cells, generate TWO new full draws of
the generic train (10,000) and val (400) splits at seeds 45 and 46 (panel
decoding recipe: temperature 1.0 / top_p 0.95 / registered arm-a generic cap),
capture answer-token residual states at each cell's FROZEN banked
``layer_star`` ONLY (y-only, never the 33-layer sweep), form 3-draw averaged
TRAIN and VAL targets (seeds 42+45+46; the seed-42 draw is the banked panel
capture), read the banked 3-draw averaged TEST target (seeds 42+43+44:
``test_1000`` + ``ceiling_s43/44`` captures), and REFIT the ridge per cell
with lambda RE-SELECTED on the averaged validation target. ``--collect``
recomputes Spearman(refit R^2, AA index) with the banked panel's exact
permutation estimator.

REPRODUCTION ANCHOR (gate, runs BEFORE any refit number): the ``gate`` phase
refits the banked SINGLE-draw bundle (banked X + seed-42 Y, same estimator
chain ``RC._fit_edge_extended_with_val`` -> ``F.fit_ridge_with_weights`` over
``LF.LAMBDAS``) and must reproduce the banked ``fits_prompt_last.json``
``test_r2`` within 0.005 and retrieval acc@1 within 0.005, else SystemExit
with no refit output (the scripts/issue2588_three_rollout_target.py gate
convention, tolerance included).

Phases (per cell, sentinel-resumable; every stage additionally resumes on its
own terminal artifact):
    prologue -> gate -> gen -> parse -> capture -> upload-raw ->
    upload-capture -> fits -> upload-fits

Pilot gate (epm:followup-scope v1): ``--pilot`` pins ``--cell q3_32b_a`` and
prints ``[k3-pilot] measured_responses_per_sec=...`` after the seed-45 train
draw completes, through this production entrypoint at production shape; the
draw is KEPT as that cell's first new train draw.

Usage::

    # pilot (one narrow pod)
    uv run python scripts/issue2588_k3_train_refit.py --pilot --out-root /workspace/eps2588_k3
    # fan-out (one wide pod, one cell per GPU, launcher pins CUDA_VISIBLE_DEVICES)
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/issue2588_k3_train_refit.py \
        --cell q36_27b_a --out-root /workspace/eps2588_k3
    # panel-level trend after all ten cells
    uv run python scripts/issue2588_k3_train_refit.py --collect

Reuse spine (never re-derived): generation/regen, capture, fit, loader, path,
upload and sentinel machinery from ``scripts/issue2588_run_cell.py`` (RC) /
``scripts/issue2588_panel_common.py`` (PC) / ``scripts/
issue2330_qwen35_generate_capture.py`` (G) / ``scripts/issue1491_ladder_fits.py``
(LF) / ``scripts/issue2330_matched_fits.py`` (MF); ``exact_spearman_permutation``
ported VERBATIM from ``scripts/issue2588_mapping_rank_vs_capability.py``
(module RANDOM_SEED=2588 -> SPEARMAN_MC_SEED); banked-capture loader modeled on
``scripts/issue2588_ceiling_normalized.py::_load_stage_layer_y`` extended to
``x_prompt_last``.

Smoke blind-spot enumeration (what the local VM smoke does NOT certify):
- gen phase GPU/vLLM leg: engine build + ``LLM.generate`` run only pod-side
  (GPU-bound carve-out: --import-check resolves the vllm imports where
  installed; --self-test exercises the stage-resume branch against seeded
  artifacts; signature-bind covers the call shape).
- capture phase GPU leg: ``G._load_capture_model`` + the teacher-forced
  forward run only pod-side (same carve-out; the rows.json resume branch and
  call shape are smoke-exercised).
- upload-raw / upload-capture / upload-fits phase BODIES are not executed by
  the smoke; their underlying helper (``RC._upload_file`` -> ``HUB._upload``)
  is exercised LIVE by ``--probe-upload`` against the smoke HF prefix.
- prologue's tokenizer/template asserts run only pod-side (network +
  tokenizer download).
Everything else — the gate phase (banked download + reproduction refit +
tolerance gate), target alignment/averaging, collect/trend, sentinel shape —
is executed by the smoke ON THE REAL production code path (the gate smoke runs
FULL banked production shape on a real cell; no N is sliced anywhere in the
gate/fits path, ``--smoke`` only redirects out-root + HF prefixes).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # before ANY vllm import

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
for _p in (str(_SCRIPTS), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + HF token BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

import issue2330_matched_fits as MF  # noqa: E402
import issue2330_qwen35_generate_capture as G  # noqa: E402
import issue2588_panel_common as PC  # noqa: E402
import issue2588_run_cell as RC  # noqa: E402
import issue1491_ladder_fits as LF  # noqa: E402

from explore_persona_space.orchestrate import hub as HUB  # noqa: E402

logger = logging.getLogger("issue2588_k3_train_refit")

ROUND_LABEL = "k3_train_refit"
PILOT_CELL = "q3_32b_a"

# The ten paper no-thinking cells (scripts/issue2588_ceiling_normalized.py
# TARGET_CELLS on main; epm:progress v165 verified banked captures for all ten).
TARGET_CELLS = (
    "q35_0p8b_a",
    "q35_2b_a",
    "q35_4b_a",
    "q35_9b_a",
    "q35_27b_a",
    "q36_27b_a",
    "q38_27b_a",
    "o3_7b_i_a",
    "o31_32b_i_a",
    "q3_32b_a",
)

# The banked panel's pinned data-repo revision: the single hf_revision carried
# by all ten in-scope rows of eval_results/issue_2588/mapping_rank_vs_capability.json
# (main), == the pin epm:progress v165 verified capture presence under.
BANKED_PANEL_REVISION = "d2c18ff676f6d4ac96ab6f58d2c57eba5209ccde"

NEW_SEEDS = (45, 46)  # epm:followup-scope v1: 42/43/44 stay the test-side namespace
TRAIN_SEEDS = (42, 45, 46)
TEST_SEEDS = (42, 43, 44)

# Reproduction-anchor tolerances (r2 tol == issue2588_three_rollout_target.GATE_R2_TOL).
GATE_R2_TOL = 0.005
GATE_ACC1_TOL = 0.005

# Aligned-row floors: test floor == issue2588_ceiling_normalized.MIN_ALIGNED_PAIRS;
# train/val floors catch a lost part / killed shard (#2130 shape) while sitting
# safely under the plausible parse-drop attrition (<= ~4% per draw measured).
MIN_ALIGNED_TEST = 900
MIN_ALIGNED_VAL = 300
MIN_ALIGNED_TRAIN = 8000

# Panel-trend references (banked artifacts; carried into collect output).
BANKED_RHO_R2_VS_AA = 0.8666666666666665  # mapping_rank_vs_capability.json raw_r2_vs_aa
RESCORED_3ROLL_RHO_R2_VS_AA = 0.6121212121212121  # three_rollout_target.json (no refit)
RESCORED_3ROLL_RHO_ACC1_CAL_VS_AA = 0.09090909090909088
SPEARMAN_MC_SEED = 2588  # == issue2588_mapping_rank_vs_capability.RANDOM_SEED


def exact_spearman_permutation(x: list[float], y: list[float]) -> dict:
    """Two-sided permutation test on Spearman rho: exact (all n! relabelings)
    for n <= 9, seeded Monte Carlo (200,000 relabelings) above that.

    Ported VERBATIM from scripts/issue2588_mapping_rank_vs_capability.py
    (main); the only edit is the module constant name RANDOM_SEED ->
    SPEARMAN_MC_SEED (same value 2588), so the n=10 Monte-Carlo p is
    draw-identical to the banked panel's.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    observed = float(spearmanr(xa, ya).statistic)
    xr = rankdata(xa)
    yr = rankdata(ya)
    xr = (xr - xr.mean()) / np.linalg.norm(xr - xr.mean())
    yr = yr - yr.mean()
    denom_y = np.linalg.norm(yr)
    exceed = 0
    if len(xa) <= 9:
        total = math.factorial(len(ya))
        for perm in itertools.permutations(yr.tolist()):
            rho = float(xr @ np.asarray(perm) / denom_y)
            exceed += abs(rho) >= abs(observed) - 1e-12
        method = "exact"
    else:
        total = 200_000
        rng = np.random.default_rng(SPEARMAN_MC_SEED)
        for _ in range(total):
            rho = float(xr @ rng.permutation(yr) / denom_y)
            exceed += abs(rho) >= abs(observed) - 1e-12
        method = "monte_carlo"
    return {
        "n": int(len(xa)),
        "rho": observed,
        "two_sided_exact_permutation_p": float(exceed / total),
        "n_permutations": int(total),
        "method": method,
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _fresh_cell(cell: PC.Cell) -> PC.Cell:
    """arm-a clone with fresh=True: the NEW s45/s46 stages are fresh
    generations even for the one banked_arm_a cell (q35_9b_a), so gen/parse/
    capture must take the fresh-cell code path, never _banked_stage_rows."""
    return PC.Cell(cell.model_key, cell.arm, True)


def _k3_prefix(args, cell: PC.Cell) -> str:
    return f"{RC._cell_prefix(args, cell)}/{ROUND_LABEL}"


def _manifest_path(paths: dict) -> Path:
    return paths["fits"] / "k3_stage_manifest.json"


def _manifest_stages(paths: dict) -> list[str]:
    p = _manifest_path(paths)
    assert p.exists(), f"{p} missing — run --phase gen first"
    return [s["stage"] for s in json.loads(p.read_text(encoding="utf-8"))["stages"]]


def _gate_path(paths: dict) -> Path:
    return paths["fits"] / "k3_gate_prompt_last.json"


def _refit_path(paths: dict) -> Path:
    return paths["fits"] / "k3_refit_prompt_last.json"


def _throughput_path(paths: dict) -> Path:
    return paths["fits"] / "k3_gen_throughput.json"


def _base_row(stage: str, r: dict) -> dict:
    return {
        "row_id": f"{stage}_{r['ladder_local_id']}",
        "prompt": r["prompt"],
        "ci": int(r["ladder_local_id"]),
    }


def _seed_stage_plan(
    generic: dict[str, list[dict]], seed: int, part_rows: int
) -> list[tuple[str, int, list[dict]]]:
    """(stage, seed, base_rows) plan for one draw: train parts, then val.

    The train split is partitioned into contiguous ~part_rows sub-stages so
    each part checkpoints (chunk files + cap_hit_report) and resumes through
    RC._gen_stage_with_regen's own terminal-artifact resume — the followup
    scope's per-chunk-writes / resume-not-repeat cadence at <= part grain.
    """
    assert part_rows > 0, part_rows
    plan: list[tuple[str, int, list[dict]]] = []
    train = generic["train_10k"]
    assert train, "empty train_10k slice"
    n_parts = math.ceil(len(train) / part_rows)
    for k in range(n_parts):
        part = train[k * part_rows : (k + 1) * part_rows]
        stage = f"train_10k_s{seed}_part{k}"
        plan.append((stage, seed, [_base_row(stage, r) for r in part]))
    stage = f"val_400_s{seed}"
    plan.append((stage, seed, [_base_row(stage, r) for r in generic["val_400"]]))
    return plan


def _write_or_check_manifest(
    args, paths: dict, cell: PC.Cell, plan: list[tuple[str, int, list[dict]]]
) -> None:
    stages = [{"stage": s, "seed": seed, "n_base": len(base)} for s, seed, base in plan]
    p = _manifest_path(paths)
    if p.exists() and not args.force:
        old = json.loads(p.read_text(encoding="utf-8"))
        assert old["train_part_rows"] == args.train_part_rows and old["stages"] == stages, (
            "k3_stage_manifest.json drift — a resume must use the same --train-part-rows "
            "and corpus slice as the original run (fresh --out-root or --force to restart)"
        )
        return
    PC.write_json_atomic(
        p,
        {
            "meta": RC._meta(),
            "cell": cell.key,
            "train_part_rows": args.train_part_rows,
            "stages": stages,
        },
    )


def _record_throughput(paths: dict, rec: dict) -> None:
    p = _throughput_path(paths)
    data = (
        json.loads(p.read_text(encoding="utf-8"))
        if p.exists()
        else {"meta": RC._meta(), "stages": {}}
    )
    data["stages"][rec["stage"]] = rec
    PC.write_json_atomic(p, data)


def _print_train_rate(args, cell: PC.Cell, paths: dict, seed: int) -> None:
    p = _throughput_path(paths)
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"stages": {}}
    ent = [v for k, v in data["stages"].items() if k.startswith(f"train_10k_s{seed}_part")]
    if not ent:
        logger.info(
            "[k3-throughput] %s seed-%d train draw fully resumed from a prior run — "
            "no fresh generation walls this invocation (see %s)",
            cell.key,
            seed,
            p,
        )
        return
    rows = sum(e["n_rows"] for e in ent)
    wall = sum(e["wall_s"] for e in ent)
    rate = rows / wall
    warm = [e for e in ent if not e["engine_cold"]]
    wrows = sum(e["n_rows"] for e in warm)
    wwall = sum(e["wall_s"] for e in warm)
    warm_txt = f" (post-warmup {wrows / wwall:.3f} resp/s over {wrows} rows)" if wwall > 0 else ""
    logger.info(
        "[k3-throughput] %s seed-%d train draw: %d rows / %.1f s = %.3f responses/s%s",
        cell.key,
        seed,
        rows,
        wall,
        rate,
        warm_txt,
    )
    if args.pilot:
        logger.info(
            "[k3-pilot] measured_responses_per_sec=%.3f cell=%s seed=%d n_rows=%d wall_s=%.1f",
            rate,
            cell.key,
            seed,
            rows,
            wall,
        )


# ---------------------------------------------------------------------------
# Banked + new capture loaders and 3-way alignment
# ---------------------------------------------------------------------------


def _ci_from_row_ids(row_ids: np.ndarray, where: str) -> np.ndarray:
    ci = np.array([int(str(r).rsplit("_", 1)[1]) for r in row_ids], dtype=np.int64)
    assert np.unique(ci).size == ci.size, f"duplicate ci in {where}"
    return ci


def _load_banked_layer(
    cell: PC.Cell, stage: str, layer: int, cache_dir: Path, revision: str, *, want_x: bool
) -> dict[str, np.ndarray]:
    """ci-sorted banked capture arrays for one (stage, layer) at the pin.

    Modeled on issue2588_ceiling_normalized._load_stage_layer_y (main),
    extended to x_prompt_last for the map-input side. Reads the PRODUCTION
    prefix always (banked artifacts exist only there), scoped listing via
    HUB.list_hf_files_under_path (a wrong prefix fails loud, never a silent
    empty).
    """
    from huggingface_hub import HfApi

    assert cell.model.hc_streams == 1, (cell.key, cell.model.hc_streams)
    ldir = f"{cell.hf_prefix}/analysis_tensors/capture/{stage}/L{layer:02d}"
    files = sorted(
        p
        for p in HUB.list_hf_files_under_path(
            HfApi(), PC.HF_DATA_REPO, ldir, repo_type="dataset", revision=revision
        )
        if p.endswith(".npz")
    )
    assert files, f"no capture shards under {ldir} at {revision}"
    ids, ys, xs = [], [], []
    for f in files:
        local = G._hub_download(f, cache_dir, revision=revision)
        with np.load(local, allow_pickle=False) as z:
            ids.append(z["row_ids"])
            ys.append(z["y_ans"])
            if want_x:
                xs.append(z["x_prompt_last"])
    ci = _ci_from_row_ids(np.concatenate(ids), ldir)
    order = np.argsort(ci)
    out = {"ci": ci[order], "y": np.concatenate(ys)[order].astype(np.float64)}
    if want_x:
        out["x"] = np.concatenate(xs)[order].astype(np.float64)
    return out


def _load_new_y(paths: dict, stages: list[str], layer: int) -> dict[str, np.ndarray]:
    """ci-sorted y_ans across this round's freshly captured stages (merged parts)."""
    assert stages, "empty stage list"
    ids, ys = [], []
    for stage in stages:
        d = RC._load_stage_layer(paths, stage, layer, tag="capture")
        ids.append(d["row_ids"])
        ys.append(d["y_ans"])
    ci = _ci_from_row_ids(np.concatenate(ids), f"new stages {stages}")
    order = np.argsort(ci)
    return {"ci": ci[order], "y": np.concatenate(ys)[order].astype(np.float64)}


def _align3(
    draws: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """3-way ci intersection + per-draw positional indices (searchsorted with
    the exact-hit verification, the issue2588_three_rollout_target convention)."""
    labels = list(draws)
    assert len(labels) == 3, labels
    ci3 = draws[labels[0]]["ci"]
    for lb in labels[1:]:
        ci3 = np.intersect1d(ci3, draws[lb]["ci"])
    assert ci3.size, f"empty 3-way ci intersection across {labels}"
    idx: dict[str, np.ndarray] = {}
    for lb in labels:
        pos = np.searchsorted(draws[lb]["ci"], ci3)
        assert np.array_equal(draws[lb]["ci"][pos], ci3), (lb, "searchsorted misalignment")
        idx[lb] = pos
    return ci3, idx


def _assemble_bundle(
    x_parts: list[np.ndarray], y_parts: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert len(x_parts) == len(y_parts) == 3
    x = np.concatenate(x_parts).astype(np.float64)
    y = np.concatenate(y_parts).astype(np.float64)
    n = 0
    idx = []
    for p in x_parts:
        idx.append(np.arange(n, n + p.shape[0], dtype=np.int64))
        n += p.shape[0]
    return x, y, idx[0], idx[1], idx[2]


# ---------------------------------------------------------------------------
# Phase: prologue (reused verbatim) + gate (reproduction anchor)
# ---------------------------------------------------------------------------


def phase_prologue_k3(args, cell: PC.Cell, paths: dict) -> None:
    RC.phase_prologue(args, cell, paths)


def _banked_meta(args, cell: PC.Cell, paths: dict) -> dict:
    """Frozen layer_star + banked selected-layer metrics from the banked
    fits/nulls JSONs on HF at the pinned panel revision (fail loud on any
    smoke / wrong-cell / wrong-position record)."""
    cache = paths["cache"]
    fits_p = G._hub_download(
        f"{PC.PANEL_PREFIX}/fits/{cell.key}/fits_prompt_last.json",
        cache,
        revision=args.hf_revision,
    )
    fits = json.loads(Path(fits_p).read_text(encoding="utf-8"))
    assert fits["cell"] == cell.key, (fits["cell"], cell.key)
    assert fits["input_position"] == "prompt_last", fits["input_position"]
    assert not fits.get("smoke"), f"{cell.key}: banked fits record is a SMOKE record"
    star = int(fits["layer_star"])
    lay = fits["layers"][str(star)]
    nulls_p = G._hub_download(
        f"{PC.PANEL_PREFIX}/nulls/{cell.key}/nulls_prompt_last.json",
        cache,
        revision=args.hf_revision,
    )
    nulls = json.loads(Path(nulls_p).read_text(encoding="utf-8"))
    assert int(nulls["layer_star"]) == star, (nulls["layer_star"], star)
    cos = lay["knn_test"]["ridge"]["cosine"]
    return {
        "layer_star": star,
        "dimension": int(lay["d"]),
        "n": {k: int(v) for k, v in lay["n"].items()},
        "test_r2": float(lay["test_r2"]),
        "acc1_raw": float(RC._acc1(cos)),
        "test_n": int(cos["n"]),
        "null_mean_acc1_cos": float(nulls["null_mean_acc1_cos"]),
        "null_perm_draws": int(nulls["perm_draws"]),
        "selected_lambda": float(lay["fit_meta"]["selected_lambda"]),
    }


def phase_gate(args, cell: PC.Cell, paths: dict) -> None:
    """Reproduce the banked single-draw fit at the frozen layer_star.

    This is the round's recipe-drift tripwire: same banked X/Y at layer_star,
    same estimator chain, compared against the banked fits record. Runs
    BEFORE any generation spend and BEFORE any refit number exists.
    """
    G._phase("k3-gate")
    bm = _banked_meta(args, cell, paths)
    star = bm["layer_star"]
    logger.info("[k3] [gate %s] frozen layer_star=%d d=%d", cell.key, star, bm["dimension"])
    tr = _load_banked_layer(cell, "train_10k", star, paths["cache"], args.hf_revision, want_x=True)
    val = _load_banked_layer(cell, "val_400", star, paths["cache"], args.hf_revision, want_x=True)
    te = _load_banked_layer(cell, "test_1000", star, paths["cache"], args.hf_revision, want_x=True)
    counts = {"tr": int(tr["ci"].size), "val": int(val["ci"].size), "te": int(te["ci"].size)}
    assert counts == bm["n"], (
        f"{cell.key}: banked capture row counts {counts} != banked fit bundle {bm['n']} — "
        "wrong revision or partial download"
    )
    x, y, itr, ival, ite = _assemble_bundle(
        [tr["x"], val["x"], te["x"]], [tr["y"], val["y"], te["y"]]
    )
    assert x.shape[1] == bm["dimension"], (x.shape, bm["dimension"])
    dev = MF._resolve_device(args.device)
    pred_te, _pred_val, meta = RC._fit_edge_extended_with_val(x, y, itr, ival, ite, dev)
    meta.pop("W_payload", None)
    r2 = LF._pooled_r2(pred_te, y[ite])
    acc1 = float(np.mean(RC._perrow_hits_cos(pred_te, y[ite])["hit1"]))
    d_r2 = r2 - bm["test_r2"]
    d_acc1 = acc1 - bm["acc1_raw"]
    failures = []
    if abs(d_r2) > GATE_R2_TOL:
        failures.append(f"test_r2 reproduced {r2:.6f} vs banked {bm['test_r2']:.6f}")
    if abs(d_acc1) > GATE_ACC1_TOL:
        failures.append(f"acc@1 reproduced {acc1:.6f} vs banked {bm['acc1_raw']:.6f}")
    if failures:
        for line in failures:
            logger.error("[k3] [gate %s] GATE FAIL: %s", cell.key, line)
        raise SystemExit(
            f"{cell.key}: k3 reproduction gate FAILED — the refit path does not reproduce "
            "the banked single-draw fit; no refit numbers will be produced"
        )
    rec = {
        "meta": RC._meta(),
        "cell": cell.key,
        "input_position": "prompt_last",
        "round": ROUND_LABEL,
        "banked_revision": args.hf_revision,
        "layer_star": star,
        "dimension": bm["dimension"],
        "banked": bm,
        "reproduced": {
            "test_r2": float(r2),
            "acc1_raw": acc1,
            "selected_lambda": float(meta["selected_lambda"]),
            "grid_extensions": int(meta["grid_extensions"]),
            "device": str(dev),
        },
        "deltas": {"test_r2": float(d_r2), "acc1_raw": float(d_acc1)},
        "tolerances": {"test_r2": GATE_R2_TOL, "acc1_raw": GATE_ACC1_TOL},
        "n": counts,
        "pass": True,
    }
    PC.write_json_atomic(_gate_path(paths), rec)
    logger.info(
        "[k3] [gate %s] PASS: |d_r2|=%.2e |d_acc1|=%.2e (lambda banked=%.4g reproduced=%.4g)",
        cell.key,
        abs(d_r2),
        abs(d_acc1),
        bm["selected_lambda"],
        meta["selected_lambda"],
    )


# ---------------------------------------------------------------------------
# Phases: gen / parse / capture (new s45/s46 draws; RC machinery reused)
# ---------------------------------------------------------------------------


def phase_gen_k3(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("k3-gen")
    import fcntl

    fresh = _fresh_cell(cell)
    RC._assert_headroom(paths, 4.0 + RC._est_model_gb(cell), f"k3-gen:{cell.key}")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cell.model.hf_id)
    mpe = RC._mpe_for_cell(cell)
    cap = PC.cap_effective(cell.arm, "generic", mpe)
    # Staging flock: the wide-pod fan-out shares one out-root; serialize the
    # manifest/split downloads exactly like RC.phase_stage.
    lock_fh = open(paths["root"] / ".staging.lock", "w")  # noqa: SIM115 — lock lifetime object
    fcntl.flock(lock_fh, fcntl.LOCK_EX)
    try:
        generic = RC._load_generic_rows(args, paths["cache"])
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()
    plan: list[tuple[str, int, list[dict]]] = []
    for seed in NEW_SEEDS:
        plan.extend(_seed_stage_plan(generic, seed, args.train_part_rows))
    _write_or_check_manifest(args, paths, cell, plan)
    pilot_group = [s for s, seed, _b in plan if seed == NEW_SEEDS[0] and s.startswith("train_")]
    llm_holder: dict = {"llm": None, "mml": 0}
    for stage, seed, base in plan:
        ran = args.force or not (paths["raw"] / stage / "cap_hit_report.json").exists()
        cold = llm_holder.get("llm") is None
        t0 = time.monotonic()
        rows = RC._gen_stage_with_regen(
            args,
            fresh,
            tok,
            base,
            stage=stage,
            cap=cap,
            cap_requested=PC.CAP[(cell.arm, "generic")],
            seed=seed,
            paths=paths,
            llm_holder=llm_holder,
        )
        wall = time.monotonic() - t0
        assert len(rows) == len(base), (stage, len(rows), len(base))
        if ran:
            _record_throughput(
                paths,
                {
                    "stage": stage,
                    "seed": seed,
                    "n_rows": len(rows),
                    "wall_s": round(wall, 3),
                    "engine_cold": cold,
                    "generated": True,
                },
            )
        if stage == pilot_group[-1]:
            _print_train_rate(args, cell, paths, NEW_SEEDS[0])
    if llm_holder.get("llm") is not None:
        G._reap_vllm_engine(llm_holder["llm"])


def phase_parse_k3(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("k3-parse")
    fresh = _fresh_cell(cell)
    assert fresh.parse_mode == "off", (cell.key, fresh.parse_mode)  # ten no-thinking cells
    counts: dict[str, dict] = {}
    dropped: dict[str, list[str]] = {}
    for stage in _manifest_stages(paths):
        rows = RC._iter_stage_rows(paths, stage)
        assert rows, f"stage {stage}: no raw rows — run --phase gen first"
        parsed, drops = [], []
        for r in rows:
            rec = PC.parse_generation(r, fresh.parse_mode)
            if rec["well_formed"]:
                parsed.append(
                    {
                        **r,
                        "ans_char_span": rec["ans_char_span"],
                        "cot_char_span": rec["cot_char_span"],
                    }
                )
            else:
                drops.append({"row_id": r["row_id"], "reason": rec["reason"]})
        assert parsed, f"stage {stage}: zero well-formed rows"
        counts[stage] = {"n": len(rows), "kept": len(parsed), "dropped": len(drops)}
        dropped[stage] = [d["row_id"] for d in drops]
        PC.write_jsonl_atomic(paths["parsed"] / f"{stage}.jsonl", parsed)
        PC.write_json_atomic(
            paths["parsed"] / f"{stage}_drops.json", {"meta": RC._meta(), "drops": drops}
        )
        logger.info("[k3] [parse %s] kept=%d dropped=%d", stage, len(parsed), len(drops))
    PC.write_json_atomic(
        paths["fits"] / "k3_dropped_row_ids.json",
        {"meta": RC._meta(), "cell": cell.key, "counts": counts, "dropped_row_ids": dropped},
    )


def phase_capture_k3(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("k3-capture")
    fresh = _fresh_cell(cell)
    gate = json.loads(_gate_path(paths).read_text(encoding="utf-8"))
    assert gate["pass"] is True, "gate record present but not passing"
    star = int(gate["layer_star"])
    stages = _manifest_stages(paths)
    # B2-style: validate EVERY persisted capture input before the model load.
    need = {"row_id", "prompt", "n_prompt_tokens", "text", "ans_char_span", "read_points"}
    for stage in stages:
        p = paths["parsed"] / f"{stage}.jsonl"
        assert p.exists(), f"capture input missing: {p} — run --phase parse first"
        rows = PC.read_jsonl(p)
        assert rows, f"capture input empty: {p}"
        bad = [(i, sorted(need - set(r))) for i, r in enumerate(rows) if need - set(r)]
        assert not bad, (
            f"parsed stage {stage}: {len(bad)} rows missing required keys "
            f"(first: row {bad[0][0]} missing {bad[0][1]})"
        )
    RC._assert_headroom(paths, 2.0 + RC._est_model_gb(cell), f"k3-capture:{cell.key}")
    from transformers import AutoTokenizer

    sem = RC._acquire_capture_slot(paths["root"])
    try:
        tok = AutoTokenizer.from_pretrained(cell.model.hf_id)
        hf = G._load_capture_model(cell.model.hf_id, args.device, "bfloat16")
        for stage in stages:
            RC._capture_stage(
                args, fresh, paths, hf, tok, stage, [star], y_only=True, layer_tag="capture"
            )
        del hf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        sem.close()


# ---------------------------------------------------------------------------
# Phases: uploads (persist raw text + tensors BEFORE the refit)
# ---------------------------------------------------------------------------


def phase_upload_raw_k3(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("k3-upload-raw")
    prefix = _k3_prefix(args, cell)
    for stage in _manifest_stages(paths):
        stage_dir = paths["raw"] / stage
        assert stage_dir.is_dir() and any(stage_dir.iterdir()), (
            f"raw stage {stage} empty at upload time"
        )
        RC._upload_dir(stage_dir, f"{prefix}/raw_completions/{stage}", f"{cell.key} k3 raw {stage}")
    for f in sorted(paths["parsed"].glob("*.json*")):
        RC._upload_file(f, f"{prefix}/parsed/{f.name}", f"{cell.key} k3 parsed {f.name}")


def phase_upload_capture_k3(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("k3-upload-capture")
    prefix = _k3_prefix(args, cell)
    for stage in _manifest_stages(paths):
        stage_dir = paths["capture"] / stage
        assert (stage_dir / "rows.json").exists(), (
            f"capture stage {stage} incomplete (no rows.json) — run --phase capture first"
        )
        RC._upload_dir(
            stage_dir,
            f"{prefix}/analysis_tensors/capture/{stage}",
            f"{cell.key} k3 capture {stage}",
        )


# ---------------------------------------------------------------------------
# Phase: fits (averaged-target refit, lambda re-selected on averaged val)
# ---------------------------------------------------------------------------


def phase_fits_k3(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("k3-fits")
    gate = json.loads(_gate_path(paths).read_text(encoding="utf-8"))
    assert gate["pass"] is True, "gate record present but not passing"
    star = int(gate["layer_star"])
    bm = gate["banked"]
    cache = paths["cache"]
    rev = args.hf_revision
    stages = _manifest_stages(paths)

    tr42 = _load_banked_layer(cell, "train_10k", star, cache, rev, want_x=True)
    val42 = _load_banked_layer(cell, "val_400", star, cache, rev, want_x=True)
    te42 = _load_banked_layer(cell, "test_1000", star, cache, rev, want_x=True)
    te43 = _load_banked_layer(cell, "ceiling_s43", star, cache, rev, want_x=False)
    te44 = _load_banked_layer(cell, "ceiling_s44", star, cache, rev, want_x=False)
    tr45 = _load_new_y(paths, [s for s in stages if s.startswith("train_10k_s45")], star)
    tr46 = _load_new_y(paths, [s for s in stages if s.startswith("train_10k_s46")], star)
    v45 = _load_new_y(paths, [s for s in stages if s == "val_400_s45"], star)
    v46 = _load_new_y(paths, [s for s in stages if s == "val_400_s46"], star)

    tr_ci, tr_idx = _align3({"42": tr42, "45": tr45, "46": tr46})
    val_ci, val_idx = _align3({"42": val42, "45": v45, "46": v46})
    te_ci, te_idx = _align3({"42": te42, "43": te43, "44": te44})
    y_tr = (tr42["y"][tr_idx["42"]] + tr45["y"][tr_idx["45"]] + tr46["y"][tr_idx["46"]]) / 3.0
    y_val = (val42["y"][val_idx["42"]] + v45["y"][val_idx["45"]] + v46["y"][val_idx["46"]]) / 3.0
    y_te = (te42["y"][te_idx["42"]] + te43["y"][te_idx["43"]] + te44["y"][te_idx["44"]]) / 3.0
    y_te_seed42 = te42["y"][te_idx["42"]]
    x_tr = tr42["x"][tr_idx["42"]]
    x_val = val42["x"][val_idx["42"]]
    x_te = te42["x"][te_idx["42"]]

    d = int(x_tr.shape[1])
    assert d == int(gate["dimension"]), (d, gate["dimension"])
    n3 = {"tr": int(tr_ci.size), "val": int(val_ci.size), "te": int(te_ci.size)}
    assert n3["tr"] >= max(d + 1, MIN_ALIGNED_TRAIN), (
        f"{cell.key}: 3-way aligned train rows {n3['tr']} below floor "
        f"max(d+1={d + 1}, {MIN_ALIGNED_TRAIN}) — lost part / killed shard shape"
    )
    assert n3["val"] >= MIN_ALIGNED_VAL, (n3["val"], MIN_ALIGNED_VAL)
    assert n3["te"] >= MIN_ALIGNED_TEST, (n3["te"], MIN_ALIGNED_TEST)
    logger.info("[k3] [fits %s] aligned n3=%s d=%d layer_star=%d", cell.key, n3, d, star)

    x, y, itr, ival, ite = _assemble_bundle([x_tr, x_val, x_te], [y_tr, y_val, y_te])
    dev = MF._resolve_device(args.device)
    pred_te, _pred_val, meta = RC._fit_edge_extended_with_val(x, y, itr, ival, ite, dev)
    meta.pop("W_payload", None)
    floors = LF._fit_floors(x, y, itr, ival, ite, dev, LF.RIDGE_BLOCK)
    preds = {"ridge": pred_te}
    preds.update({k: v["pred_te"] for k, v in floors.items()})
    knn_te = LF._knn_reads(preds, y[ite])
    r2_avg = LF._pooled_r2(pred_te, y[ite])
    acc1_avg = float(np.mean(RC._perrow_hits_cos(pred_te, y[ite])["hit1"]))
    r2_s42 = LF._pooled_r2(pred_te, y_te_seed42)
    acc1_s42 = float(np.mean(RC._perrow_hits_cos(pred_te, y_te_seed42)["hit1"]))
    # Retrieval calibration: banked shuffled-pairing null mean scaled by pool
    # size (the three_rollout_target convention; < 2e-5 absolute correction).
    null3 = bm["null_mean_acc1_cos"] * bm["test_n"] / n3["te"]

    rec = {
        "meta": RC._meta(),
        "cell": cell.key,
        "input_position": "prompt_last",
        "round": ROUND_LABEL,
        "banked_revision": rev,
        "layer_star": star,
        "layer_star_rule": "FROZEN at the banked panel selection (epm:followup-scope v1)",
        "dimension": d,
        "aa_index": PC.AA_PIN[cell.model_key][0],
        "banked": bm,
        "gate": {"pass": True, "deltas": gate["deltas"], "reproduced": gate["reproduced"]},
        "alignment": {
            "train_seeds": list(TRAIN_SEEDS),
            "test_seeds": list(TEST_SEEDS),
            "n_train_banked": int(tr42["ci"].size),
            "n_train_s45": int(tr45["ci"].size),
            "n_train_s46": int(tr46["ci"].size),
            "n3_train": n3["tr"],
            "n_val_banked": int(val42["ci"].size),
            "n_val_s45": int(v45["ci"].size),
            "n_val_s46": int(v46["ci"].size),
            "n3_val": n3["val"],
            "n_test_banked": int(te42["ci"].size),
            "n3_test": n3["te"],
        },
        "refit": {
            "selected_lambda": float(meta["selected_lambda"]),
            "banked_selected_lambda": bm["selected_lambda"],
            "grid_extensions": int(meta["grid_extensions"]),
            "test_r2_avg_target": float(r2_avg),
            "test_acc1_cos_avg_target_raw": acc1_avg,
            "null_mean_scaled": float(null3),
            "test_acc1_cos_avg_target_calibrated": float(acc1_avg - null3),
            "test_r2_seed42_target_subset": float(r2_s42),
            "test_acc1_cos_seed42_target_subset_raw": acc1_s42,
            "floors_test_r2_avg_target": {k: float(v["test_r2"]) for k, v in floors.items()},
            "knn_test": knn_te,
            "n": n3,
            "d": d,
            "n_train_over_d": float(n3["tr"] / d),
            "fit_meta": {
                k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool, type(None)))
            },
        },
        "references": {
            "banked_raw_rho_r2_vs_aa": BANKED_RHO_R2_VS_AA,
            "rescored_3roll_rho_r2_vs_aa": RESCORED_3ROLL_RHO_R2_VS_AA,
            "rescored_3roll_source": (
                "eval_results/issue_2588/ceiling_normalized/three_rollout_target.json (main)"
            ),
        },
    }
    PC.write_json_atomic(_refit_path(paths), rec)
    logger.info(
        "[k3] [fits %s] refit: r2 %.4f (banked %.4f) acc1_cal %.4f "
        "(banked %.4f) lambda %.4g -> %.4g",
        cell.key,
        r2_avg,
        bm["test_r2"],
        acc1_avg - null3,
        bm["acc1_raw"] - bm["null_mean_acc1_cos"],
        bm["selected_lambda"],
        meta["selected_lambda"],
    )


def phase_upload_fits_k3(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("k3-upload-fits")
    for p in (_gate_path(paths), _refit_path(paths), _throughput_path(paths)):
        assert p.exists(), f"{p} missing — earlier phases incomplete"
    smoke_pfx = "smoke/" if args.smoke else ""
    prefix = f"{PC.PANEL_PREFIX}/{smoke_pfx}{ROUND_LABEL}/fits/{cell.key}"
    uploaded = sorted(paths["fits"].glob("k3_*.json"))
    for f in uploaded:
        RC._upload_file(f, f"{prefix}/{f.name}", f"{cell.key} {f.name}")
    refit = json.loads(_refit_path(paths).read_text(encoding="utf-8"))
    gate = json.loads(_gate_path(paths).read_text(encoding="utf-8"))
    sentinel = {
        "eval_numbers": {
            "layer_star": refit["layer_star"],
            "gate_delta_test_r2": gate["deltas"]["test_r2"],
            "refit_test_r2_avg_target": refit["refit"]["test_r2_avg_target"],
            "refit_test_acc1_cos_avg_target_calibrated": refit["refit"][
                "test_acc1_cos_avg_target_calibrated"
            ],
        },
        "eval_paths": [str(p) for p in uploaded],
        "reproducibility_card": RC._meta(),
        "wandb_url": None,
        "hf_hub_url": (f"https://huggingface.co/datasets/{PC.HF_DATA_REPO}/tree/main/{prefix}"),
        "worktree_path": str(_REPO_ROOT),
        "final_commit_sha": G._git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": None,
        "plan_deviations": [],
    }
    out = paths["logs"] / f"issue-2588-{cell.key}-k3-results.json"
    PC.write_json_atomic(out, sentinel)
    logger.info("[phase=done] cell %s k3 complete rc=0 (sentinel %s)", cell.key, out)


# ---------------------------------------------------------------------------
# Collect: panel-level trend across the ten refits
# ---------------------------------------------------------------------------


def run_collect(args) -> int:
    recs: dict[str, dict] = {}
    if args.refit_local_dir:
        for p in sorted(Path(args.refit_local_dir).rglob("k3_refit_prompt_last.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            assert r["cell"] not in recs, f"duplicate refit record for {r['cell']}"
            recs[r["cell"]] = r
    else:
        cache = Path(args.collect_cache)
        cache.mkdir(parents=True, exist_ok=True)
        for ck in TARGET_CELLS:
            p = G._hub_download(
                f"{PC.PANEL_PREFIX}/{ROUND_LABEL}/fits/{ck}/k3_refit_prompt_last.json",
                cache,
                revision=args.collect_revision,
            )
            recs[ck] = json.loads(Path(p).read_text(encoding="utf-8"))
    missing = [c for c in TARGET_CELLS if c not in recs]
    assert not missing, f"refit records missing for: {missing}"
    unknown = sorted(set(recs) - set(TARGET_CELLS))
    assert not unknown, f"unexpected refit records: {unknown}"

    per_model: list[dict] = []
    for ck in TARGET_CELLS:
        r = recs[ck]
        assert r["gate"]["pass"] is True, f"{ck}: gate did not pass"
        aa = PC.AA_PIN[ck.removesuffix("_a")][0]
        assert aa is not None and r["aa_index"] == aa, (ck, r["aa_index"], aa)
        per_model.append(
            {
                "cell": ck,
                "aa_index": aa,
                "layer_star": r["layer_star"],
                "dimension": r["dimension"],
                "n3": r["refit"]["n"],
                "banked_test_r2": r["banked"]["test_r2"],
                "banked_acc1_calibrated": r["banked"]["acc1_raw"]
                - r["banked"]["null_mean_acc1_cos"],
                "gate_delta_test_r2": r["gate"]["deltas"]["test_r2"],
                "refit_test_r2_avg_target": r["refit"]["test_r2_avg_target"],
                "refit_acc1_avg_target_calibrated": r["refit"][
                    "test_acc1_cos_avg_target_calibrated"
                ],
                "refit_test_r2_seed42_target_subset": r["refit"]["test_r2_seed42_target_subset"],
                "selected_lambda_banked": r["banked"]["selected_lambda"],
                "selected_lambda_refit": r["refit"]["selected_lambda"],
            }
        )

    aa = [float(m["aa_index"]) for m in per_model]
    banked_rho = exact_spearman_permutation([m["banked_test_r2"] for m in per_model], aa)
    assert abs(banked_rho["rho"] - BANKED_RHO_R2_VS_AA) <= 1e-9, (
        f"banked-r2 rho {banked_rho['rho']} != reference {BANKED_RHO_R2_VS_AA} — "
        "wrong banked inputs or AA pins drifted"
    )
    trends = {
        "banked_r2_vs_aa": banked_rho,
        "refit_r2_avg_target_vs_aa": exact_spearman_permutation(
            [m["refit_test_r2_avg_target"] for m in per_model], aa
        ),
        "refit_acc1_cal_vs_aa": exact_spearman_permutation(
            [m["refit_acc1_avg_target_calibrated"] for m in per_model], aa
        ),
        "refit_r2_seed42_subset_vs_aa": exact_spearman_permutation(
            [m["refit_test_r2_seed42_target_subset"] for m in per_model], aa
        ),
    }
    for k, v in trends.items():
        # The verbatim-ported estimator emits nan rho on a zero-variance
        # column (spearmanr); a degenerate trend input must fail loud here,
        # never ship as "rho=nan p=0.0".
        assert np.isfinite(v["rho"]), f"trend {k}: degenerate (constant) input column"
    out = {
        "schema_version": 1,
        "meta": {
            **RC._meta(),
            "script": "scripts/issue2588_k3_train_refit.py",
            "round": ROUND_LABEL,
            "estimator": (
                "exact_spearman_permutation ported verbatim from "
                "scripts/issue2588_mapping_rank_vs_capability.py (MC seed 2588)"
            ),
        },
        "references": {
            "banked_raw_rho_r2_vs_aa": BANKED_RHO_R2_VS_AA,
            "rescored_3roll_rho_r2_vs_aa": RESCORED_3ROLL_RHO_R2_VS_AA,
            "rescored_3roll_rho_acc1_cal_vs_aa": RESCORED_3ROLL_RHO_ACC1_CAL_VS_AA,
            "rescored_3roll_source": (
                "eval_results/issue_2588/ceiling_normalized/three_rollout_target.json (main)"
            ),
        },
        "per_model": per_model,
        "trends": trends,
    }
    out_p = Path(args.collect_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    logger.info("[k3-collect] wrote %s", out_p)
    for k, v in trends.items():
        logger.info(
            "[k3-collect] %s: rho=%+.4f p=%.5f (%s, n=%d)",
            k,
            v["rho"],
            v["two_sided_exact_permutation_p"],
            v["method"],
            v["n"],
        )
    return 0


# ---------------------------------------------------------------------------
# Import-check / self-test / upload probe
# ---------------------------------------------------------------------------


def _run_import_check() -> int:
    """Axis-1 import resolution: argcheck + every deferred import + call binds."""
    import inspect

    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    import fcntl  # noqa: F401

    from huggingface_hub import HfApi  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401

    for mod, names in (
        (
            RC,
            (
                "_paths",
                "_cell_prefix",
                "_meta",
                "_acc1",
                "_assert_headroom",
                "_est_model_gb",
                "_mpe_for_cell",
                "_load_generic_rows",
                "_gen_stage_with_regen",
                "_iter_stage_rows",
                "_capture_stage",
                "_acquire_capture_slot",
                "_load_stage_layer",
                "_fit_edge_extended_with_val",
                "_perrow_hits_cos",
                "_upload_dir",
                "_upload_file",
                "_phase_complete",
                "_mark_phase_done",
                "phase_prologue",
            ),
        ),
        (
            PC,
            (
                "Cell",
                "cell_by_key",
                "cap_effective",
                "CAP",
                "AA_PIN",
                "PANEL_PREFIX",
                "HF_DATA_REPO",
                "parse_generation",
                "write_json_atomic",
                "write_jsonl_atomic",
                "read_jsonl",
            ),
        ),
        (
            G,
            ("_phase", "_git_sha", "_hub_download", "_load_capture_model", "_reap_vllm_engine"),
        ),
        (LF, ("LAMBDAS", "RIDGE_BLOCK", "_fit_floors", "_knn_reads", "_pooled_r2")),
        (MF, ("_resolve_device",)),
        (HUB, ("list_hf_files_under_path", "_upload")),
    ):
        for name in names:
            assert hasattr(mod, name), (mod.__name__, name)
    try:
        from vllm import LLM, SamplingParams, TokensPrompt  # noqa: F401
    except ImportError:
        print(
            "[import-check] vllm not installed on this box (CPU smoke) — gen phase "
            "unavailable here; all other deferred imports resolved"
        )
    # Signature-bind the reused entry points at each call site's static shape.
    inspect.signature(RC._gen_stage_with_regen).bind(
        None, None, None, [], stage="s", cap=1, cap_requested=1, seed=45, paths={}, llm_holder={}
    )
    inspect.signature(RC._capture_stage).bind(
        None, None, {}, None, None, "s", [0], y_only=True, layer_tag="capture"
    )
    inspect.signature(RC._fit_edge_extended_with_val).bind(None, None, None, None, None, None)
    inspect.signature(RC._load_stage_layer).bind({}, "s", 0, tag="capture")
    inspect.signature(RC._upload_dir).bind(Path("."), "p", "w")
    inspect.signature(RC._upload_file).bind(Path("."), "p", "w")
    inspect.signature(G._hub_download).bind("p", Path("."), revision="r")
    inspect.signature(G._load_capture_model).bind("m", "cpu", "bfloat16")
    inspect.signature(LF._fit_floors).bind(None, None, None, None, None, None, None)
    inspect.signature(LF._knn_reads).bind({}, None)
    inspect.signature(HUB.list_hf_files_under_path).bind(
        None, "r", "p", repo_type="dataset", revision="m"
    )
    print("[import-check] OK: argcheck + deferred imports + reuse-seam signature binds resolved")
    return 0


# Self-test fixture: the ten banked test_r2 values (4-dp, rank-preserving) from
# eval_results/issue_2588/mapping_rank_vs_capability.json (main) — used ONLY to
# exercise run_collect's banked-rho reference gate against synthetic refits.
_SELF_TEST_BANKED_R2 = {
    "q35_0p8b_a": 0.6456,
    "q35_2b_a": 0.6320,
    "q35_4b_a": 0.6553,
    "q35_9b_a": 0.6638,
    "q35_27b_a": 0.7243,
    "q36_27b_a": 0.6739,
    "q38_27b_a": 0.6933,
    "o3_7b_i_a": 0.6249,
    "o31_32b_i_a": 0.6127,
    "q3_32b_a": 0.6724,
}


def _run_self_test() -> int:
    """CPU self-test: resume legs, alignment math, estimator port, sentinel
    shape, collect end-to-end. No network, no GPU."""
    args = _build_parser().parse_args(["--self-test"])
    with tempfile.TemporaryDirectory(prefix="k3selftest-") as td:
        args.out_root = str(Path(td) / "outroot")
        cell = PC.cell_by_key("q35_0p8b_a")
        paths = RC._paths(args, cell)

        # 1) gen stage resume: seeded terminal artifact -> rows returned, NO engine.
        stage = "train_10k_s45_part0"
        sd = paths["raw"] / stage
        sd.mkdir(parents=True, exist_ok=True)
        rows_fixture = [{"row_id": f"{stage}_7", "text": "ok", "finish_reason": "stop"}]
        PC.write_json_atomic(sd / "chunk0000.json", {"rows": rows_fixture})
        PC.write_json_atomic(sd / "cap_hit_report.json", {"stage": stage, "n": 1})
        holder: dict = {"llm": None, "mml": 0}
        rows = RC._gen_stage_with_regen(
            args,
            _fresh_cell(cell),
            None,
            [],
            stage=stage,
            cap=2048,
            cap_requested=2048,
            seed=45,
            paths=paths,
            llm_holder=holder,
        )
        assert rows == rows_fixture and holder["llm"] is None, "gen resume leg broken"

        # 2) capture stage resume: rows.json present -> early return, no model use.
        cd = paths["capture"] / stage
        cd.mkdir(parents=True, exist_ok=True)
        PC.write_json_atomic(cd / "rows.json", {"rows": []})
        RC._capture_stage(
            args,
            _fresh_cell(cell),
            paths,
            None,
            None,
            stage,
            [0],
            y_only=True,
            layer_tag="capture",
        )

        # 3) 3-way alignment + averaging math on synthetic draws.
        rng = np.random.default_rng(0)
        mk = lambda cis: {  # noqa: E731 — tiny fixture builder
            "ci": np.array(sorted(cis), dtype=np.int64),
            "y": rng.normal(size=(len(cis), 4)),
        }
        d42, d45, d46 = mk([1, 2, 3, 4, 5]), mk([2, 3, 4, 5, 6]), mk([0, 2, 4, 6])
        ci3, idx = _align3({"42": d42, "45": d45, "46": d46})
        assert ci3.tolist() == [2, 4], ci3
        avg = (d42["y"][idx["42"]] + d45["y"][idx["45"]] + d46["y"][idx["46"]]) / 3.0
        for j, ci in enumerate(ci3):
            manual = (
                d42["y"][d42["ci"].tolist().index(ci)]
                + d45["y"][d45["ci"].tolist().index(ci)]
                + d46["y"][d46["ci"].tolist().index(ci)]
            ) / 3.0
            assert np.allclose(avg[j], manual), ci

        # 4) Spearman-port sanity: perfect monotone triple, exact branch.
        r = exact_spearman_permutation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        assert r["method"] == "exact" and abs(r["rho"] - 1.0) < 1e-12, r
        assert abs(r["two_sided_exact_permutation_p"] - 2.0 / 6.0) < 1e-12, r

        # 5) Results sentinel parses through the poller (envelope-less rescue).
        import poll_pipeline as PP

        sentinel = {
            "eval_numbers": {"layer_star": 22},
            "eval_paths": [],
            "reproducibility_card": {"issue": 2588},
            "wandb_url": None,
            "hf_hub_url": "https://huggingface.co/datasets/x/tree/main/y",
            "worktree_path": "/workspace",
            "final_commit_sha": "0" * 40,
            "gpu_hours_used": None,
            "gpu_hours_budgeted": None,
            "plan_deviations": [],
        }
        env = PP._parse_sentinel(
            "/workspace/logs/issue-2588-q3_32b_a-k3-results.json", json.dumps(sentinel)
        )
        assert env is not None and env["kind"] == "epm:results", env

        # 6) collect end-to-end on fixture refits (real banked r2 ranks -> the
        #    banked-rho reference gate must PASS; synthetic refit values).
        fx = Path(td) / "refits"
        for ck in TARGET_CELLS:
            mk_dir = fx / ck
            mk_dir.mkdir(parents=True, exist_ok=True)
            banked_r2 = _SELF_TEST_BANKED_R2[ck]
            (mk_dir / "k3_refit_prompt_last.json").write_text(
                json.dumps(
                    {
                        "cell": ck,
                        "aa_index": PC.AA_PIN[ck.removesuffix("_a")][0],
                        "layer_star": 1,
                        "dimension": 4,
                        "banked": {
                            "test_r2": banked_r2,
                            "acc1_raw": 0.7,
                            "null_mean_acc1_cos": 0.001,
                            "test_n": 980,
                            "selected_lambda": 100.0,
                        },
                        "gate": {"pass": True, "deltas": {"test_r2": 0.0}},
                        "refit": {
                            "n": {"tr": 9000, "val": 350, "te": 950},
                            "test_r2_avg_target": banked_r2 + 0.05,
                            "test_acc1_cos_avg_target_calibrated": banked_r2 + 0.1,
                            "test_r2_seed42_target_subset": banked_r2,
                            "selected_lambda": 10.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
        args.refit_local_dir = str(fx)
        args.collect_out = str(Path(td) / "k3_collect.json")
        rc = run_collect(args)
        assert rc == 0
        out = json.loads(Path(args.collect_out).read_text(encoding="utf-8"))
        assert abs(out["trends"]["banked_r2_vs_aa"]["rho"] - BANKED_RHO_R2_VS_AA) <= 1e-9
        assert out["trends"]["refit_r2_avg_target_vs_aa"]["method"] == "monte_carlo"
    print("[self-test] OK: resume legs, alignment, estimator port, sentinel, collect")
    return 0


def _run_probe_upload(args) -> int:
    """Live probe of the fenced upload seam against the smoke HF prefix."""
    tmp = Path(tempfile.mkdtemp(prefix="k3probe-"))
    f = tmp / "k3_upload_probe.json"
    PC.write_json_atomic(f, {"meta": RC._meta(), "probe": "k3_train_refit upload probe"})
    dest = (
        f"{PC.PANEL_PREFIX}/smoke/{ROUND_LABEL}/upload_probe/"
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    )
    RC._upload_file(f, dest, "k3 upload probe")
    print(f"[probe-upload] OK: {dest}")
    return 0


# ---------------------------------------------------------------------------
# Phase registry + main
# ---------------------------------------------------------------------------

K3_PHASES: dict = {
    "prologue": phase_prologue_k3,
    "gate": phase_gate,
    "gen": phase_gen_k3,
    "parse": phase_parse_k3,
    "capture": phase_capture_k3,
    "upload-raw": phase_upload_raw_k3,
    "upload-capture": phase_upload_capture_k3,
    "fits": phase_fits_k3,
    "upload-fits": phase_upload_fits_k3,
}
K3_SEQUENCE = tuple(K3_PHASES)


def _run_k3_phases(args, cell: PC.Cell, paths: dict, seq: tuple[str, ...]) -> list[str]:
    """RC's sentinel-skip loop over the k3 registry ('k3-' sentinel namespace)."""
    ran: list[str] = []
    for name in seq:
        sname = f"k3-{name}"
        if RC._phase_complete(args, paths, sname):
            logger.info("[k3] phase %s already complete — skipped (--force to re-run)", name)
            continue
        K3_PHASES[name](args, cell, paths)
        RC._mark_phase_done(args, cell, paths, sname)
        ran.append(name)
    return ran


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cell", help=f"cell key, one of: {', '.join(TARGET_CELLS)}")
    ap.add_argument("--phase", default="all", choices=["all", *K3_PHASES])
    ap.add_argument(
        "--pilot",
        action="store_true",
        help=f"pilot gate (epm:followup-scope v1): pins --cell {PILOT_CELL} and prints the "
        "[k3-pilot] measured responses/s headline after the seed-45 train draw",
    )
    ap.add_argument("--out-root", default="/workspace/eps2588_k3")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="redirects out-root subdir + HF prefixes to smoke/; the gate/fits math is "
        "NEVER sliced (production shape either way)",
    )
    ap.add_argument("--force", action="store_true", help="re-run phases whose sentinels exist")
    ap.add_argument("--layer-set", default="swept", choices=["swept"], help=argparse.SUPPRESS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--capture-batch-size", type=int, default=8)
    ap.add_argument(
        "--train-part-rows",
        type=int,
        default=2500,
        help="train-split sub-stage size (generation checkpoint/resume grain)",
    )
    ap.add_argument(
        "--hf-revision",
        default=BANKED_PANEL_REVISION,
        help="banked panel data-repo revision pin (fits/nulls/capture reads)",
    )
    ap.add_argument("--collect", action="store_true", help="panel-level trend over the ten refits")
    ap.add_argument(
        "--refit-local-dir",
        default=None,
        help="collect from local k3_refit_prompt_last.json files under this dir (rglob)",
    )
    ap.add_argument("--collect-revision", default="main")
    ap.add_argument(
        "--collect-cache",
        default=str(_REPO_ROOT / "data" / "issue_2588" / "k3_collect_cache"),
    )
    ap.add_argument(
        "--collect-out",
        default=str(
            _REPO_ROOT
            / "eval_results"
            / "issue_2588"
            / ROUND_LABEL
            / "k3_train_refit_capability.json"
        ),
    )
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--probe-upload", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.import_check:
        return _run_import_check()
    if args.self_test:
        return _run_self_test()
    if args.probe_upload:
        return _run_probe_upload(args)
    if args.collect:
        return run_collect(args)
    if args.pilot:
        assert args.cell in (None, PILOT_CELL), f"--pilot pins --cell {PILOT_CELL}"
        args.cell = PILOT_CELL
    assert args.cell, "--cell required (or --pilot / --collect / --self-test / --import-check)"
    assert args.cell in TARGET_CELLS, (
        f"{args.cell!r} is not one of the k3 round's ten no-thinking cells: {TARGET_CELLS}"
    )
    cell = PC.cell_by_key(args.cell)
    assert cell.arm == "a", cell.key
    assert cell.model.hc_streams == 1, (cell.key, cell.model.hc_streams)
    assert args.gpu_count == cell.model.tp_gpus, (args.gpu_count, cell.model.tp_gpus)
    assert PC.AA_PIN[cell.model_key][0] is not None, f"{cell.model_key}: no AA pin"
    paths = RC._paths(args, cell)
    seq = K3_SEQUENCE if args.phase == "all" else (args.phase,)
    ran = _run_k3_phases(args, cell, paths, seq)
    logger.info("[k3] phases run=%s skipped=%s", ran, [s for s in seq if s not in ran])
    return 0


if __name__ == "__main__":
    rc = main()
    if RC._ENGINE_CONSTRUCTED:
        # vLLM worker-subprocess teardown gotcha: never run interpreter exit
        # hooks with a dead engine's workers half-reaped (gotchas.md).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    sys.exit(rc)
