"""Issue #1310 free-analysis round 3: the ASSISTANT direct test.

Question: is the assistant's context->answer operator the SAME shared operator
the four #1310 characters share, up to linear coordinates? Round 2
(issue1310_xpersona_similarity_v2.py) found the four characters share a
DOMINANTLY shared context->dialogue operator. This round runs the v2 battery
VERBATIM on a NEW cell trio drawn from #1335's ablation ladder, so the reads are
directly comparable to round 2 (same rig: GCV Gram/primal ridge, dof cap 0.9,
scenario-grouped 5-fold, fold seed 0, layer 19 headline; frozen 14/18/26 for the
transfer sweep only).

CELLS (all #1335 gen_qa rungs, ctx arm X=x_spanmean v_C, Y=y reply-span mean):
  r1_qa_oneline   assistant (label "Assistant") answering bare User:/Assistant:
                  one-line Q&A.
  r2_op           Wren (label "Wren") answering the SAME bare Q&A format,
                  on-policy.
  r4_fictionframe Wren answering the same questions inside a scene wrapper.

Three key contrasts (report makes each explicit, beside round-2 / cross-project
anchors 0.516 / 0.593 base/instruct data-paired Procrustes; 0.455 #1345
story<->chat; 0.686 #825 base<->instruct; 0.732 / 0.855 #1345 chat<->plain):
  (a) r1 <-> r2_op       IDENTITY-only at fixed format (Assistant vs Wren, both
                         bare Q&A).
  (b) r2_op <-> r4       FRAMING-only at fixed identity (Wren bare Q&A vs Wren in
                         scene).
  (c) r1 <-> r4          the HEADLINE: is the assistant in the shared operator?

Reuse: the v2 battery statistical primitives are reused VERBATIM by pointing
their module-level PERSONAS list at the 3 cells (issue1310_xpersona_similarity
v1 + _v2). The ONLY adaptation is the integrity gate: the #1310 EQUALITY gate
(diagonal reproduces committed #1310 within-cells at <=1e-6) is replaced by a
#1335 BAND-CHECK (within-cell L19 R^2 lands near the committed #1335
rung_values_matched_ctx band) — the cells are #1335's, so no #1310 committed
within-cell exists to equality-gate against, and the two rigs (my full-n GCV
scenario-fold vs #1335's matched-n inner-group-CV) differ, so the check is a
sanity band, not bit-exact (per the round-3 brief).

Data: HF dataset superkaiba1/explore-persona-space-data, prefix
issue1335_ablation_ladder/analysis_tensors/store_{cell}_{model}. Pure-CPU,
8-thread caps (#847). Per-(model, leg) JSON checkpoint + resume so each leg fits
the tool-timeout budget.

CLI:
  uv run python scripts/issue1310_xpersona_assistant_test.py \
      [--models base,instruct] [--legs transfer,decomp,reparam,operator,predsim] \
      [--stage-root <dir>] [--hf-revision <sha>] \
      [--out-dir eval_results/issue_1310/xpersona_similarity/assistant_test] \
      [--fig-dir figures/issue_1310] [--null-draws 5] [--reparam-null-draws 5] \
      [--rot-draws 20] [--n-boot 1000] [--seed 0] [--probe] [--summary-from-disk]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) bind before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue825_fit_cells as fit825  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402
import issue1310_common as c1310  # noqa: E402
import issue1310_xpersona_similarity as v1  # noqa: E402
import issue1310_xpersona_similarity_v2 as v2  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

SCRIPT = "scripts/issue1310_xpersona_assistant_test.py"

CELLS = ["r1_qa_oneline", "r2_op", "r4_fictionframe"]
CELL_DESC = {
    "r1_qa_oneline": "assistant bare Q&A",
    "r2_op": "Wren bare Q&A (on-policy)",
    "r4_fictionframe": "Wren in scene",
}
# The three key contrasts (unordered pair -> what it isolates).
KEY_CONTRASTS = {
    "r1_qa_oneline~r2_op": "identity-only (Assistant vs Wren, fixed bare-QA format)",
    "r2_op~r4_fictionframe": "framing-only (Wren bare-QA vs Wren-in-scene, fixed identity)",
    "r1_qa_oneline~r4_fictionframe": "HEADLINE: is the assistant in the shared operator",
}

# Point the reused v1/v2 battery primitives (which iterate their module-level
# PERSONAS list) at the 3 assistant-test cells, so run_operator_nulled /
# run_pred_similarity / _pooled_fold_preds / run_reparam iterate the cells. The
# stat primitives are otherwise reused verbatim; only PERSONAS (the cell list)
# and the integrity gate (band-check vs #1310 equality gate) change.
v1.PERSONAS = CELLS
v2.PERSONAS = CELLS

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1335_ablation_ladder/analysis_tensors"
HF_REVISION = "a02089179f142e8d78ab6725f589f796fa03f4e9"  # pinned 2026-07-21 probe

FROZEN_LAYERS = tuple(c1310.FROZEN_LAYERS)  # (14, 18, 19, 26)
HEADLINE_LAYER = c1310.HEADLINE_LAYER  # 19
MODEL_KINDS = list(c1310.MODEL_KINDS)  # base, instruct
DOF_CAP = 0.9  # v2 parity (uncapped GCV degenerates on this near-square n~d store)
N_FOLDS = c1310.N_FOLDS  # 5
FIT_SEED = c1310.FIT_SEED  # 0
EXPECTED_LAYERS = c1310.EXPECTED_LAYERS  # 28
EXPECTED_HIDDEN = c1310.EXPECTED_HIDDEN  # 3584

DEFAULT_STAGE_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue1310_xpersona/hf_dl_1335")
DEFAULT_OUT_DIR = Path("eval_results/issue_1310/xpersona_similarity/assistant_test")
LADDER_SUMMARY = Path("eval_results/issue_1335/ladder_summary.json")

# Round-2 (#1310) data-paired Procrustes aligned cosine (the assistant test's
# nearest sibling) + the cross-project anchors v2 already carries.
PROCRUSTES_ANCHORS = {
    "round2_1310_xpersona_base": 0.516,
    "round2_1310_xpersona_instruct": 0.593,
    "issue1345_paired_story_vs_chat": 0.455,
    "issue825_base_vs_instruct": 0.6864,
    "issue1345_chat_vs_plain_base": 0.732,
    "issue1345_chat_vs_plain_instruct": 0.855,
}

# Band-check: |mine - committed #1335 rung_values_matched_ctx| flagged wildly-off
# above this. The two rigs differ (full-n GCV scenario-fold vs #1335 matched-n
# inner-group-CV), so this is a sanity band, not an equality gate.
BAND_WILDLY_OFF = 0.20


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", type=str, default=",".join(MODEL_KINDS))
    ap.add_argument(
        "--legs",
        type=str,
        default="transfer,decomp,reparam,operator,predsim",
        help="comma list subset of {transfer,decomp,reparam,operator,predsim}",
    )
    ap.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    ap.add_argument("--hf-revision", type=str, default=HF_REVISION)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_1310"))
    ap.add_argument("--null-draws", type=int, default=5)
    ap.add_argument("--reparam-null-draws", type=int, default=5)
    ap.add_argument("--rot-draws", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=c1310.N_BOOTSTRAP)  # 1000
    ap.add_argument("--seed", type=int, default=FIT_SEED)
    ap.add_argument(
        "--probe",
        action="store_true",
        help="load one model's arrays, print per-cell + intersection n, time ONE "
        "pooled fold prep + ONE transfer fold at production shape, then exit "
        "(compute-character sizing; no output written)",
    )
    ap.add_argument(
        "--summary-from-disk",
        action="store_true",
        help="skip compute; assemble summary_assistant.json + the figure from the "
        "per-model leg JSONs already in --out-dir (split-run pattern)",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Staging (#833-safe: scoped list_repo_tree + hf_hub_download pool, pinned rev).
# ---------------------------------------------------------------------------
def stage_cell_store(cell: str, model: str, stage_root: Path, revision: str) -> Path:
    """Stage store_{cell}_{model} via the canonical retried scoped-prefix helper
    (hub.stage_hub_prefix — the #833 recipe: server-side scoped listing +
    retried per-file hf_hub_download pool, one pinned revision; #1402). Files
    land at stage_root/<repo-relative path> (verbatim prefix mirror), so the
    returned dir holds the {model}_shard*.pt shards + .json sidecars."""
    prefix = f"{HF_PREFIX}/store_{cell}_{model}"
    hub.stage_hub_prefix(
        HF_REPO, prefix, stage_root, repo_type="dataset", revision=revision, max_workers=6
    )
    return stage_root / prefix


def load_cell_store(dest: Path, model: str) -> dict:
    """Concatenate {model}_shard*.pt payloads. Returns row_ids/group_ids + the
    ctx-arm X (x_spanmean) and Y (reply-span mean) arrays (n, L, D) fp32, plus a
    reuse-fitness report read from the first shard's sidecar."""
    shards = sorted(dest.glob(f"{model}_shard*.pt"))
    assert shards, f"no {model} shards under {dest}"
    rows, groups = [], []
    xs, ys = [], []
    fitness = None
    for sp in shards:
        side = json.loads(sp.with_suffix(".json").read_text())
        if fitness is None:
            keys = side.get("keys", [])
            fitness = {
                "render_config_hash": side.get("render_config_hash"),
                "code_sha": side.get("code_sha"),
                "shape_per_row": side.get("shape_per_row"),
                "keys": keys,
                "has_x_spanmean": "x_spanmean" in keys,
                "has_y": "y" in keys,
            }
        payload = torch.load(sp, map_location="cpu", weights_only=False)
        rows.extend(payload["row_ids"])
        groups.extend(payload["group_ids"])
        xs.append(payload["arrays"]["x_spanmean"].float().numpy().astype(np.float32))
        ys.append(payload["arrays"]["y"].float().numpy().astype(np.float32))
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    n = len(rows)
    assert X.shape == (n, EXPECTED_LAYERS, EXPECTED_HIDDEN), X.shape
    assert Y.shape == (n, EXPECTED_LAYERS, EXPECTED_HIDDEN), Y.shape
    row_carries_q = all(":q" in r for r in rows[:50])
    fitness["n_rows"] = n
    fitness["row_ids_carry_question"] = bool(row_carries_q)
    return {
        "row_ids": np.asarray(rows),
        "group_ids": np.asarray(groups),
        "X": X,
        "Y": Y,
        "fitness": fitness,
    }


def _qid(row_id: str) -> str:
    """`{slug}:q00042` -> `q00042` (the question id, shared across cells)."""
    return row_id.split(":", 1)[1]


def build_arrays(model: str, args) -> tuple[dict, dict]:
    """Load the 3 cell stores, intersect kept question ids, row-align by qid, and
    build the SHARED scenario-blocked 5-fold partition from r4's qid->scenario
    map. Returns ({cell: {X,Y,scen,folds,qid}}, provenance)."""
    stores = {}
    per_cell_fitness = {}
    for cell in CELLS:
        dest = stage_cell_store(cell, model, args.stage_root, args.hf_revision)
        st = load_cell_store(dest, model)
        stores[cell] = st
        per_cell_fitness[cell] = st["fitness"]
        del dest
        gc.collect()

    # Per-cell qid -> row index (assert one row per question within a cell).
    qid_row = {}
    for cell in CELLS:
        m = {}
        for i, rid in enumerate(stores[cell]["row_ids"]):
            q = _qid(str(rid))
            assert q not in m, f"{cell}: duplicate question id {q} (expected one row/question)"
            m[q] = i
        qid_row[cell] = m

    # qid -> scenario from r4's group_ids (r4 group_id == scenario_id).
    r4 = "r4_fictionframe"
    qid_scen = {}
    for i, rid in enumerate(stores[r4]["row_ids"]):
        qid_scen[_qid(str(rid))] = str(stores[r4]["group_ids"][i])

    # Intersection of kept question ids across the trio; only qids r4 maps to a
    # scenario (all r4 qids do, by construction).
    inter = set(qid_row[CELLS[0]])
    for cell in CELLS[1:]:
        inter &= set(qid_row[cell])
    inter &= set(qid_scen)
    qids = sorted(inter)
    assert qids, "empty question-id intersection across the trio"

    scen = np.asarray([qid_scen[q] for q in qids])
    # Shared scenario-blocked fold partition (grouped by scenario value, seed 0).
    folds = fit825._cv_folds(scen, N_FOLDS, args.seed)

    arrays = {}
    for cell in CELLS:
        idx = np.asarray([qid_row[cell][q] for q in qids], dtype=np.int64)
        arrays[cell] = {
            "X": np.ascontiguousarray(stores[cell]["X"][idx], dtype=np.float32),
            "Y": np.ascontiguousarray(stores[cell]["Y"][idx], dtype=np.float32),
            "scen": scen,
            "folds": folds,
            "qid": np.asarray(qids),
        }
    # Row-alignment invariant: identical qid arrays across cells.
    for cell in CELLS[1:]:
        assert np.array_equal(arrays[cell]["qid"], arrays[CELLS[0]]["qid"])
    del stores
    gc.collect()

    provenance = {
        "cells": CELLS,
        "cell_desc": CELL_DESC,
        "per_cell_kept_n": {c: int(per_cell_fitness[c]["n_rows"]) for c in CELLS},
        "intersection_n": len(qids),
        "n_scenarios": int(len(np.unique(scen))),
        "fold_grouping": "scenario (from r4_fictionframe qid->scenario map); "
        "shared across all 3 cells; leakage-safe for r4, stricter-but-harmless for "
        "r1/r2 (their #1335 native grouping is per-row/question)",
        "pairing_key": "question id (row_id `q{q_idx:05d}` suffix)",
        "arm": "ctx (X=x_spanmean v_C <=512 ctx tokens; Y=y reply-span mean)",
        "reuse_fitness": per_cell_fitness,
        "hf_repo": HF_REPO,
        "hf_prefix": HF_PREFIX,
        "hf_revision": args.hf_revision,
    }
    return arrays, provenance


# ---------------------------------------------------------------------------
# Integrity band-check (replaces the #1310 equality gate).
# ---------------------------------------------------------------------------
def _ladder_ctx_refs(model: str) -> dict:
    js = json.loads(LADDER_SUMMARY.read_text())
    ref = js["per_model"][model]["rung_values_matched_ctx"]
    return {c: float(ref[c]) for c in CELLS}


def band_check(model: str, m2_r2: dict) -> dict:
    """Compare within-cell L19 R^2 (M2 diagonal) to #1335 rung_values_matched_ctx.
    NOT bit-exact; flag wildly-off (|delta|>BAND_WILDLY_OFF)."""
    refs = _ladder_ctx_refs(model)
    per_cell = {}
    worst = 0.0
    for c in CELLS:
        d = abs(m2_r2[c] - refs[c])
        per_cell[c] = {
            "mine_L19_r2_foldmean": m2_r2[c],
            "committed_matched_ctx": refs[c],
            "abs_delta": d,
        }
        worst = max(worst, d)
    return {
        "kind": "band_check_vs_issue1335_rung_values_matched_ctx",
        "note": "sanity band (my full-n GCV scenario-fold vs #1335 matched-n inner-group-CV); "
        "NOT bit-exact",
        "wildly_off_threshold": BAND_WILDLY_OFF,
        "per_cell": per_cell,
        "worst_abs_delta": worst,
        "wildly_off": worst > BAND_WILDLY_OFF,
    }


# ---------------------------------------------------------------------------
# Leg 1 — 3x3 transfer matrix (frozen layers) + band-check + L19 off-diag boot.
# ---------------------------------------------------------------------------
def run_transfer(model: str, arrays: dict, args) -> dict:
    matrices = {}
    l19_cells = {}
    for layer in FROZEN_LAYERS:
        mat_fold, mat_glob = {}, {}
        for s in CELLS:
            for t in CELLS:
                cell = v1.transfer_cell(arrays[s], arrays[t], layer)
                mat_fold[f"{s}->{t}"] = cell["r2_foldmean"]
                mat_glob[f"{s}->{t}"] = cell["r2_globalmean"]
                if layer == HEADLINE_LAYER:
                    l19_cells[(s, t)] = cell
        matrices[str(layer)] = {"foldmean": mat_fold, "globalmean": mat_glob}

    m2_r2 = {c: matrices[str(HEADLINE_LAYER)]["foldmean"][f"{c}->{c}"] for c in CELLS}
    gate = band_check(model, m2_r2)

    boot = {}
    for t in CELLS:
        within = l19_cells[(t, t)]
        yt = arrays[t]["Y"][:, HEADLINE_LAYER, :].astype(np.float64)
        scen_t = arrays[t]["scen"]
        gb_w = fit931.group_bootstrap_r2(
            within["preds"], yt, scen_t, n_boot=args.n_boot, seed=args.seed
        )
        for s in CELLS:
            if s == t:
                continue
            tr = l19_cells[(s, t)]
            gb_t = fit931.group_bootstrap_r2(
                tr["preds"],
                yt,
                scen_t,
                n_boot=args.n_boot,
                seed=args.seed,
                draws_matrix=gb_w["draws_matrix"],
            )
            delta = gb_t["draws"] - gb_w["draws"]
            boot[f"{s}->{t}"] = {
                "transfer_r2_foldmean": tr["r2_foldmean"],
                "transfer_r2_globalmean": tr["r2_globalmean"],
                "within_r2_foldmean": within["r2_foldmean"],
                "delta_transfer_minus_within": gb_t["r2"] - gb_w["r2"],
                "delta_ci_lo": float(np.nanquantile(delta, 0.025)),
                "delta_ci_hi": float(np.nanquantile(delta, 0.975)),
                "transfer_frac_of_within": (
                    tr["r2_foldmean"] / within["r2_foldmean"]
                    if within["r2_foldmean"] > 1e-9
                    else float("nan")
                ),
                "n_groups": int(gb_w["n_groups"]),
                "n_boot": int(args.n_boot),
            }
    return {
        "headline_layer": HEADLINE_LAYER,
        "frozen_layers": list(FROZEN_LAYERS),
        "matrices": matrices,
        "integrity_band_check": gate,
        "l19_offdiag_bootstrap": boot,
    }


# ---------------------------------------------------------------------------
# Leg 2 — shared-vs-specific decomposition (v2 primitives; band-check gate).
# ---------------------------------------------------------------------------
def run_decomposition(model: str, arrays: dict, args) -> dict:
    """Copy of v2.run_decomposition with the #1310 committed equality gate
    replaced by the #1335 band-check. Every statistical primitive
    (v2._pooled_fold_preds, v1.transfer_cell, fit931.group_bootstrap_r2) is
    reused verbatim; only the gate block differs."""
    layer = HEADLINE_LAYER
    m0 = v2._pooled_fold_preds(arrays, layer, centering="global")
    m1 = v2._pooled_fold_preds(arrays, layer, centering="per_persona")
    m2_preds, m2_r2 = {}, {}
    for p in CELLS:
        cell = v1.transfer_cell(arrays[p], arrays[p], layer)
        m2_preds[p] = cell["preds"]
        m2_r2[p] = cell["r2_foldmean"]

    gate = band_check(model, m2_r2)  # <-- swapped for #1310 equality gate

    per_persona = {}
    for p in CELLS:
        yp = arrays[p]["Y"][:, layer, :].astype(np.float64)
        scen = arrays[p]["scen"]
        gb0 = fit931.group_bootstrap_r2(
            m0["preds"][p], yp, scen, n_boot=args.n_boot, seed=args.seed
        )
        dm = gb0["draws_matrix"]
        gb1 = fit931.group_bootstrap_r2(
            m1["preds"][p], yp, scen, n_boot=args.n_boot, seed=args.seed, draws_matrix=dm
        )
        gb2 = fit931.group_bootstrap_r2(
            m2_preds[p], yp, scen, n_boot=args.n_boot, seed=args.seed, draws_matrix=dm
        )
        d10 = gb1["draws"] - gb0["draws"]
        d21 = gb2["draws"] - gb1["draws"]
        per_persona[p] = {
            "r2_M0_foldmean": m0["r2"][p],
            "r2_M1_foldmean": m1["r2"][p],
            "r2_M2_foldmean": m2_r2[p],
            "delta_M1_minus_M0": gb1["r2"] - gb0["r2"],
            "delta_M1_minus_M0_ci": [
                float(np.nanquantile(d10, 0.025)),
                float(np.nanquantile(d10, 0.975)),
            ],
            "delta_M2_minus_M1": gb2["r2"] - gb1["r2"],
            "delta_M2_minus_M1_ci": [
                float(np.nanquantile(d21, 0.025)),
                float(np.nanquantile(d21, 0.975)),
            ],
            "frac_M0_over_M2": (m0["r2"][p] / m2_r2[p] if m2_r2[p] > 1e-9 else float("nan")),
            "frac_M1_over_M2": (m1["r2"][p] / m2_r2[p] if m2_r2[p] > 1e-9 else float("nan")),
        }

    rng = np.random.default_rng(args.seed + 101)
    null_r2 = {p: [] for p in CELLS}
    for _ in range(args.null_draws):
        perm = {p: rng.permutation(arrays[p]["X"].shape[0]) for p in CELLS}
        mn = v2._pooled_fold_preds(arrays, layer, centering="per_persona", y_perm=perm)
        for p in CELLS:
            null_r2[p].append(mn["r2"][p])

    return {
        "headline_layer": layer,
        "integrity_band_check": gate,
        "per_persona": per_persona,
        "pooled": {
            "r2_M0": v2._pooled_r2(m0["preds"], arrays, layer),
            "r2_M1": v2._pooled_r2(m1["preds"], arrays, layer),
            "r2_M2": v2._pooled_r2(m2_preds, arrays, layer),
        },
        "m1_shuffle_null": {
            p: {"draws": [float(v) for v in null_r2[p]], "mean": float(np.nanmean(null_r2[p]))}
            for p in CELLS
        },
    }


# ---------------------------------------------------------------------------
# Leg operator + activation-Procrustes shuffle-fit null (v2 primitives verbatim).
# ---------------------------------------------------------------------------
def run_operator(model: str, arrays: dict, args) -> dict:
    """v2.run_operator_nulled (raw Frobenius / spectrum / input-output subspaces /
    data-paired Procrustes-aligned cosine + rotation null) PLUS the addendum
    activation-Procrustes shuffle-fit null per pair, both verbatim from v2."""
    op = v2.run_operator_nulled(model, arrays, args)
    op["procrustes_calibration_anchors"] = PROCRUSTES_ANCHORS
    # Addendum: shuffle-fit null for the data-paired aligned cosine (v2's
    # --activation-procrustes path), reusing 5 fresh shuffle-fit betas per cell.
    cm.GCV_DOF_CAP = DOF_CAP
    cm.LAMBDA_SELECTION = "gcv"
    layer = HEADLINE_LAYER
    n = arrays[CELLS[0]]["X"].shape[0]
    rng = np.random.default_rng(args.seed + 505)
    shuf_betas = {
        p: [
            v2._fit_beta(arrays, p, layer, y_perm=rng.permutation(n))
            for _ in range(args.null_draws)
        ]
        for p in CELLS
    }
    for i in range(len(CELLS)):
        for j in range(i + 1, len(CELLS)):
            a, b = CELLS[i], CELLS[j]
            shuf_null = v2.activation_procrustes_shuffle_null(
                arrays, a, b, layer, shuf_a=shuf_betas[a], shuf_b=shuf_betas[b]
            )
            op["pairs"][f"{a}~{b}"]["procrustes_aligned"]["shuffle_fit_null"] = shuf_null
    return op


# ---------------------------------------------------------------------------
# Per-model orchestration with per-leg JSON checkpoint + resume.
# ---------------------------------------------------------------------------
def _leg_path(out_dir: Path, leg: str, model: str) -> Path:
    return out_dir / f"{leg}_{model}.json"


def run_model_legs(model: str, legs: list[str], args) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pending = [
        leg
        for leg in legs
        if not _leg_path(out, {"decomp": "decomposition"}.get(leg, leg), model).exists()
    ]
    if not pending:
        print(f"[assistant-test] {model}: all requested legs present on disk; skipping load")
        return
    t0 = time.time()
    print(f"[assistant-test] {model}: building arrays (pending legs: {pending})")
    arrays, provenance = build_arrays(model, args)
    print(
        f"[assistant-test] {model}: per_cell_n={provenance['per_cell_kept_n']} "
        f"intersection_n={provenance['intersection_n']} "
        f"n_scenarios={provenance['n_scenarios']} ({time.time() - t0:.0f}s)"
    )
    c1310.write_json(
        out / f"provenance_{model}.json",
        {"metadata": c1310.metadata(SCRIPT, args.seed, 0), "model_kind": model, **provenance},
    )

    meta = lambda: {"metadata": c1310.metadata(SCRIPT, args.seed, 0), "model_kind": model}

    if "transfer" in pending:
        print(f"[assistant-test] {model}: leg transfer (3x3 x {len(FROZEN_LAYERS)} layers)")
        res = run_transfer(model, arrays, args)
        c1310.write_json(_leg_path(out, "transfer", model), {**meta(), **res})
        g = res["integrity_band_check"]
        print(
            f"[assistant-test] {model}: band-check worst|d|={g['worst_abs_delta']:.3f} "
            f"wildly_off={g['wildly_off']} ({time.time() - t0:.0f}s)"
        )

    if "decomp" in pending:
        print(f"[assistant-test] {model}: leg decomposition (M0<=M1<=M2)")
        res = run_decomposition(model, arrays, args)
        c1310.write_json(_leg_path(out, "decomposition", model), {**meta(), **res})
        print(f"[assistant-test] {model}: decomposition done ({time.time() - t0:.0f}s)")

    if "reparam" in pending:
        print(f"[assistant-test] {model}: leg reparam (6 ordered pairs)")
        within = {c: v1.transfer_cell(arrays[c], arrays[c], HEADLINE_LAYER) for c in CELLS}
        res = v1.run_reparam(model, arrays, within, args)
        c1310.write_json(_leg_path(out, "reparam", model), {**meta(), **res})
        print(f"[assistant-test] {model}: reparam done ({time.time() - t0:.0f}s)")

    if "operator" in pending:
        print(f"[assistant-test] {model}: leg operator (raw/spectrum/subspaces/Procrustes)")
        res = run_operator(model, arrays, args)
        c1310.write_json(_leg_path(out, "operator", model), {**meta(), **res})
        print(f"[assistant-test] {model}: operator done ({time.time() - t0:.0f}s)")

    if "predsim" in pending:
        print(f"[assistant-test] {model}: leg predsim (prediction-space, 6 ordered pairs)")
        res = v2.run_pred_similarity(model, arrays, args)
        c1310.write_json(_leg_path(out, "pred_similarity", model), {**meta(), **res})
        print(f"[assistant-test] {model}: predsim done ({time.time() - t0:.0f}s)")

    del arrays
    gc.collect()


# ---------------------------------------------------------------------------
# Probe (compute-character sizing).
# ---------------------------------------------------------------------------
def probe(args) -> int:
    model = [m.strip() for m in args.models.split(",") if m.strip()][0]
    fit825.GCV_DOF_CAP = DOF_CAP
    t0 = time.time()
    arrays, prov = build_arrays(model, args)
    print(
        f"[probe] {model}: per_cell_n={prov['per_cell_kept_n']} "
        f"intersection_n={prov['intersection_n']} n_scenarios={prov['n_scenarios']} "
        f"stage+load={time.time() - t0:.0f}s"
    )
    print(f"[probe] reuse_fitness={json.dumps(prov['reuse_fitness'], default=str)[:500]}")
    layer = HEADLINE_LAYER
    # Time ONE transfer fold-prep (per-cell, n_train ~ 4/5 n).
    c = CELLS[0]
    x = arrays[c]["X"][:, layer, :]
    f = arrays[c]["folds"]
    tr = f != 0
    te = f == 0
    t1 = time.time()
    cache = fit825._prep_fold(x[tr], x[te])
    _ = fit825._ridge_predict_cached(cache, arrays[c]["Y"][:, layer, :][tr])
    dt_percell = time.time() - t1
    print(
        f"[probe] one PER-CELL transfer fold prep+predict: {dt_percell:.1f}s "
        f"(n_train={int(tr.sum())})"
    )
    # Time ONE pooled fold-prep (n_train ~ 4/5 * 3n).
    trx = np.concatenate([arrays[p]["X"][:, layer, :][arrays[p]["folds"] != 0] for p in CELLS])
    tex = np.concatenate([arrays[p]["X"][:, layer, :][arrays[p]["folds"] == 0] for p in CELLS])
    try_y = np.concatenate([arrays[p]["Y"][:, layer, :][arrays[p]["folds"] != 0] for p in CELLS])
    t2 = time.time()
    cache = fit825._prep_fold(trx.astype(np.float32), tex.astype(np.float32))
    _ = fit825._ridge_predict_cached(cache, try_y.astype(np.float32))
    dt_pooled = time.time() - t2
    print(f"[probe] one POOLED fold prep+predict: {dt_pooled:.1f}s (n_train={trx.shape[0]})")
    # Rough projections.
    n_transfer_prep = 9 * N_FOLDS * len(FROZEN_LAYERS)
    n_pooled_prep = N_FOLDS * (2 + args.null_draws)  # M0 + M1 + null draws
    print(
        f"[probe] PROJECTION per model: transfer ~{n_transfer_prep} percell-preps "
        f"x {dt_percell:.1f}s = {n_transfer_prep * dt_percell / 60:.1f} min; "
        f"decomposition ~{n_pooled_prep} pooled-preps x {dt_pooled:.1f}s = "
        f"{n_pooled_prep * dt_pooled / 60:.1f} min"
    )
    return 0


# ---------------------------------------------------------------------------
# Summary + figure.
# ---------------------------------------------------------------------------
def _load_results(out_dir: Path, models: list[str]) -> dict:
    res = {}
    for m in models:
        res[m] = {}
        for leg, fname in [
            ("transfer", "transfer"),
            ("decomposition", "decomposition"),
            ("reparam", "reparam"),
            ("operator", "operator"),
            ("pred_similarity", "pred_similarity"),
        ]:
            p = out_dir / f"{fname}_{m}.json"
            if p.exists():
                res[m][leg] = json.loads(p.read_text())
    return res


def build_summary(res: dict, models: list[str], args) -> None:
    summary = {
        "metadata": c1310.metadata(SCRIPT, args.seed, 0),
        "question": "Is the assistant's context->answer operator the same shared operator "
        "the #1310 characters share, up to linear coordinates?",
        "cells": CELLS,
        "cell_desc": CELL_DESC,
        "key_contrasts": KEY_CONTRASTS,
        "models": models,
        "headline_layer": HEADLINE_LAYER,
        "gcv_dof_cap": DOF_CAP,
        "procrustes_calibration_anchors": PROCRUSTES_ANCHORS,
        "hf_revision": args.hf_revision,
        "per_model": {},
    }
    for m in models:
        rm = res.get(m, {})
        entry = {}
        if "transfer" in rm:
            entry["integrity_band_check"] = rm["transfer"]["integrity_band_check"]
            entry["transfer_l19_foldmean"] = rm["transfer"]["matrices"][str(HEADLINE_LAYER)][
                "foldmean"
            ]
        if "decomposition" in rm:
            dec = rm["decomposition"]
            entry["pooled_lattice"] = dec["pooled"]
            entry["per_cell_lattice"] = {
                p: {
                    "M0": dec["per_persona"][p]["r2_M0_foldmean"],
                    "M1": dec["per_persona"][p]["r2_M1_foldmean"],
                    "M2": dec["per_persona"][p]["r2_M2_foldmean"],
                    "frac_M0_over_M2": dec["per_persona"][p]["frac_M0_over_M2"],
                    "frac_M1_over_M2": dec["per_persona"][p]["frac_M1_over_M2"],
                    "M2_minus_M1": dec["per_persona"][p]["delta_M2_minus_M1"],
                    "M2_minus_M1_ci": dec["per_persona"][p]["delta_M2_minus_M1_ci"],
                    "M1_shuffle_null": dec["m1_shuffle_null"][p]["mean"],
                }
                for p in CELLS
            }
        if "operator" in rm:
            op = rm["operator"]["pairs"]
            entry["procrustes_aligned_by_pair"] = {
                k: {
                    "observed_aligned_cosine": op[k]["procrustes_aligned"][
                        "observed_aligned_cosine"
                    ],
                    "raw_vec_cosine": op[k]["procrustes_aligned"].get("raw_vec_cosine"),
                    "rotation_null_p975": op[k]["procrustes_aligned"]["null_p975"],
                    "shuffle_fit_null_mean": op[k]["procrustes_aligned"]
                    .get("shuffle_fit_null", {})
                    .get("null_mean"),
                    "contrast": KEY_CONTRASTS.get(k, ""),
                }
                for k in op
            }
        if "pred_similarity" in rm:
            ps = rm["pred_similarity"]["ordered_pairs"]
            entry["pred_similarity_mean_cosine"] = float(
                np.mean([ps[k]["cosine_mean"] for k in ps])
            )
            entry["pred_similarity_null_cosine_mean"] = float(
                np.mean([ps[k]["null_cosine_mean"] for k in ps])
            )
        if "reparam" in rm:
            rp = rm["reparam"]["ordered_pairs"]
            entry["reparam_recovery_mean"] = float(
                np.mean([rp[k]["recovery_r2_foldmean"] for k in rp])
            )
        summary["per_model"][m] = entry
    c1310.write_json(args.out_dir / "summary_assistant.json", summary)


def make_figure(res: dict, models: list[str], args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("neurips")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    c_m0 = pp.paper_palette_role("control")
    c_m1 = pp.paper_palette_role("baseline")
    c_m2 = pp.paper_palette_role("primary")
    c_acc = pp.paper_palette_role("accent")

    fig, axes = plt.subplots(
        1, 2 * len(models), figsize=(6.0 * len(models), 4.6), layout="constrained"
    )
    axes = np.atleast_1d(axes).ravel()
    for mi, m in enumerate(models):
        rm = res.get(m, {})
        # Left: shared-vs-specific lattice bars per cell.
        axL = axes[2 * mi]
        if "decomposition" in rm:
            dec = rm["decomposition"]["per_persona"]
            x = np.arange(len(CELLS))
            w = 0.26
            axL.bar(
                x - w,
                [dec[c]["r2_M0_foldmean"] for c in CELLS],
                w,
                color=c_m0,
                label="M0 one map, global offset",
            )
            axL.bar(
                x,
                [dec[c]["r2_M1_foldmean"] for c in CELLS],
                w,
                color=c_m1,
                label="M1 shared map, per-cell offset",
            )
            axL.bar(
                x + w,
                [dec[c]["r2_M2_foldmean"] for c in CELLS],
                w,
                color=c_m2,
                label="M2 per-cell maps (within)",
            )
            for i, c in enumerate(CELLS):
                nullv = rm["decomposition"]["m1_shuffle_null"][c]["mean"]
                axL.plot(
                    [x[i] - 1.4 * w, x[i] + 1.4 * w],
                    [nullv, nullv],
                    color="0.4",
                    lw=1.0,
                    ls=":",
                    label="M1 shuffle null" if i == 0 else None,
                )
            axL.axhline(0.0, color="0.6", lw=0.8)
            axL.set_xticks(x, [c.replace("_", "\n") for c in CELLS], fontsize=6.5)
            axL.set_ylabel("held-out R² (fold-test-mean)")
            axL.set_title(f"{m}: shared-vs-specific (L{HEADLINE_LAYER})")
            if mi == 0:
                axL.legend(fontsize=6.2, loc="lower left")
        # Right: per-pair data-paired Procrustes-aligned cosine + shuffle-fit null.
        axR = axes[2 * mi + 1]
        if "operator" in rm:
            op = rm["operator"]["pairs"]
            labels = list(op.keys())
            xx = np.arange(len(labels))
            proc = [op[k]["procrustes_aligned"]["observed_aligned_cosine"] for k in labels]
            shuf = [
                op[k]["procrustes_aligned"].get("shuffle_fit_null", {}).get("null_p975")
                for k in labels
            ]
            axR.bar(xx, proc, 0.55, color=c_m2, label="data-paired Procrustes aligned cosine")
            axR.plot(xx, shuf, "_", color=c_acc, ms=13, mew=2, label="shuffle-fit null p97.5")
            for aname, aval, acol in (
                (
                    "round2 #1310 0.516/0.593",
                    PROCRUSTES_ANCHORS[f"round2_1310_xpersona_{m}"],
                    "0.2",
                ),
                ("#1345 story↔chat 0.455", 0.455, "0.5"),
                ("#825 base↔instruct 0.686", 0.6864, "0.7"),
            ):
                axR.axhline(aval, color=acol, ls="--", lw=0.9, label=aname if mi == 0 else None)
            axR.set_xticks(
                xx,
                [
                    k.replace("_qa_oneline", "").replace("_fictionframe", "").replace("~", "\n~")
                    for k in labels
                ],
                fontsize=6.5,
            )
            axR.set_ylim(0, 1.0)
            axR.set_ylabel("aligned operator cosine")
            axR.set_title(f"{m}: cross-cell operator similarity (L{HEADLINE_LAYER})")
            if mi == 0:
                axR.legend(fontsize=6.0, loc="upper right")
    fig.suptitle("Assistant direct test: is the assistant in the shared context→answer operator?")
    pp.savefig_paper(fig, "xpersona_assistant_test", dir=str(args.fig_dir), formats=("png",))
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    torch.set_num_threads(8)
    fit825.GCV_DOF_CAP = DOF_CAP
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        assert m in MODEL_KINDS, f"unknown model {m!r}"
    legs = [x.strip() for x in args.legs.split(",") if x.strip()]

    if args.probe:
        return probe(args)

    if args.summary_from_disk:
        res = _load_results(args.out_dir, models)
        build_summary(res, models, args)
        make_figure(res, models, args)
        print("[assistant-test] summary + figure assembled from disk")
        return 0

    for m in models:
        run_model_legs(m, legs, args)

    # Assemble if every model has every leg on disk.
    res = _load_results(args.out_dir, models)
    all_present = all(
        all(
            leg in res.get(m, {})
            for leg in ["transfer", "decomposition", "reparam", "operator", "pred_similarity"]
        )
        for m in models
    )
    if all_present:
        build_summary(res, models, args)
        make_figure(res, models, args)
        print("[assistant-test] all legs present; summary + figure written")
    else:
        print("[assistant-test] leg(s) still pending; re-run remaining then --summary-from-disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
