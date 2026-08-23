"""Issue #2054 inline free-analysis: extended shared-vs-specific decomposition.

Extends the #1310/#1639 M0/M1/M2 context->answer decomposition to a 6-cell
subset that INCLUDES the assistant (chat template + in-story attributed-quote)
alongside the four story characters, on_policy condition, context arm, layer
19, BOTH models (base Qwen2.5-7B and SFT Qwen2.5-7B-Instruct):

  M0      one POOLED GCV-ridge over the 12 subset units (6 cells x 2 models,
          the #1639 joint-across-model pooling the pool_specialize pilot used)
          + a single global offset (standardize-X / center-Y train means).
  M1      M0 + per-unit bias b_c = train-mean(y) - M0(train-mean(x)).
  M2      the within-cell own-map ceiling — REUSED from the banked per-cell
          fits (``issue2054_lattice/fits/{cell}.json`` per-fold ``r2_ambient``
          + the ladder.json fold-mean); never refit here.
  DIRECT  the assistant__on_policy__chat own map (SharedEighRidge, the banked
          estimator recipe) applied rung-1 (no coordinate change) to each
          target unit's held-out contexts, scored against that unit's true
          answers. For the assistant-chat unit itself this is its own-map
          held-out read (== ceiling up to refit parity, reported).

Bare_label story cells are EXCLUDED (the #2054-disclosed trailing-space
digit-start tokenization artifact). LINEAR throughout; 0 GPU-h; CPU only.

Reused cores (do-not-reimplement): ``scripts.issue2054_ctx2ctx_fit`` supplies
``SharedEighRidge`` (fit_h-parity GCV ridge, #1887 dof cap 0.9) + the
production fold-map loader (smoke map refused); ``scripts.
issue2054_pool_specialize`` supplies ``PooledMomentRidge`` (moment-based M0),
``load_cell_with_answer`` and ``join_cell``; ``analysis.mapping_baselines``
supplies the mandatory identity+bias baseline and the kNN retrieval read;
``fit_h.reconstruction_metrics`` scores held-out R^2 (fold-local centering,
the banked-ceiling convention).

Bootstrap: conversation-grouped, 200 draws, ONE shared conversation resample
from the union applied to every unit (base and instruct coupled, so the
instruct-minus-base delta CIs are paired), per-draw R^2 under the fixed
full-scored-mean SS_tot convention — the pool_specialize convention. M2 has
no banked per-row errors, so every M2-involving interval (M2-M1 increment,
M2 delta) uses FOLD-ALIGNED resampling over the 5 shared folds (10,000 draws,
the #2054 twobytwo convention) — disclosed in values.json.

Phases (checkpoint-per-phase; each invocation is bounded):
  --stage-only        download the 12 activation npz + 12 banked fit JSONs
                      from the HF data repo at the pinned revision.
  --accumulate-only   pooled second moments -> moments checkpoint.
  --folds F [F ...]   per-fold pooled fit + own-map fits + unit scoring ->
                      per-fold checkpoints.
  --finalize          bootstrap + values.json + figure.

Usage (production):
  uv run python scripts/issue2054_extended_decomp.py --stage-only
  uv run python scripts/issue2054_extended_decomp.py --accumulate-only
  uv run python scripts/issue2054_extended_decomp.py --folds 0 1 2 3 4
  uv run python scripts/issue2054_extended_decomp.py --finalize
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF/creds BEFORE torch import (code-style.md)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    # script mode puts scripts/ (not the repo root) on sys.path[0] (gotchas.md).
    sys.path.insert(0, str(_REPO))

from explore_persona_space.analysis.mapping_baselines import identity_bias_predict, knn_retrieval
from explore_persona_space.experiments.issue_779.fit_h import reconstruction_metrics
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts.issue2054_ctx2ctx_fit import (
    D_AMBIENT,
    Cell,
    SharedEighRidge,
    load_fold_map,
)
from scripts.issue2054_pool_specialize import (
    PooledMomentRidge,
    join_cell,
    load_cell_with_answer,
)

SCRIPT_VERSION = "issue2054_extended_decomp_v1"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_REVISION = "003e392548fcbbe866c6f345f4688d8176cd9f04"  # body-pinned lattice revision
ARM = "context"
LAYER = 19
MODELS = ("qwen2.5-7b", "qwen2.5-7b-instruct")
ASSIST = "conversation_paired_stories_assistant"
# (identity, framing) subset — bare_label EXCLUDED (#2054 tokenization artifact).
SUBSET = [
    (ASSIST, "chat"),
    (ASSIST, "attrib_quoted"),
    ("char_helios", "attrib_quoted"),
    ("char_wren", "attrib_quoted"),
    ("char_dana", "attrib_quoted"),
    ("char_vex", "attrib_quoted"),
]
CONDITION = "on_policy"
RUNG_NAMES = ("m0", "m1", "direct", "identity_cell")  # per-row-error rungs
BOOTSTRAP_DRAWS = 200
FOLD_RESAMPLE_DRAWS = 10_000
SEED_BASE = 137

DEFAULT_STAGE_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue2054_extended_decomp")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _unit_key(identity: str, framing: str, model: str) -> str:
    return f"{identity}__{CONDITION}__{framing}__{model}"


def unit_keys() -> list[str]:
    return [_unit_key(i, f, m) for (i, f) in SUBSET for m in MODELS]


def _hf_paths() -> dict[str, dict[str, str]]:
    """unit key -> {npz, fits} paths inside the HF data repo."""
    out: dict[str, dict[str, str]] = {}
    for identity, framing in SUBSET:
        for model in MODELS:
            key = _unit_key(identity, framing, model)
            out[key] = {
                "npz": f"issue2054_lattice/activations/{identity}/{key}.npz",
                "fits": f"issue2054_lattice/fits/{key}.json",
            }
    return out


def stage(stage_root: Path) -> dict[str, dict[str, Path]]:
    """Download the 12 npz + 12 banked fit JSONs at the pinned revision."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    local: dict[str, dict[str, Path]] = {}
    for key, paths in _hf_paths().items():
        rec: dict[str, Path] = {}
        for kind, path_in_repo in paths.items():
            t0 = time.time()
            p = retry_transient(
                lambda pin=path_in_repo: hf_hub_download(
                    HF_DATA_REPO,
                    pin,
                    repo_type="dataset",
                    revision=HF_REVISION,
                    local_dir=stage_root / "hf",
                ),
                what=f"stage {kind} {key}",
            )
            rec[kind] = Path(p)
            _log(f"[extdecomp] staged {kind} {key} ({time.time() - t0:.1f}s)")
        fits = json.loads(rec["fits"].read_text(encoding="utf-8"))
        if int(fits["layer"]) != LAYER:
            raise RuntimeError(f"{key}: banked fit layer {fits['layer']} != {LAYER}")
        local[key] = rec
    return local


def staged_paths(stage_root: Path) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for key, paths in _hf_paths().items():
        rec = {kind: stage_root / "hf" / pin for kind, pin in paths.items()}
        for kind, p in rec.items():
            if not p.is_file():
                raise FileNotFoundError(f"not staged: {p} — run --stage-only first")
        out[key] = rec
    return out


def load_units(
    local: dict[str, dict[str, Path]], fold_map: dict, smoke_rows: int | None
) -> dict[str, dict]:
    """Load + fold-join every unit; hold joined float32 arrays in RAM."""
    k = int(fold_map["k"])
    units: dict[str, dict] = {}
    for key, rec in local.items():
        parts = key.split("__")
        cell = Cell(parts[0], parts[1], parts[2], parts[3], rec["npz"])
        act = load_cell_with_answer(cell)
        j = join_cell(act, fold_map["fold_of"], k, ARM)
        fold_rows = j["fold_rows"]
        if smoke_rows is not None:
            # deterministic per-fold truncation (sorted conv order) for smoke.
            per_fold = max(1, smoke_rows // k)
            fold_rows = [fr[:per_fold] for fr in fold_rows]
        keep = np.concatenate(fold_rows)
        order = np.asarray(j["order"], dtype=object)
        # re-index folds into the kept-row coordinate system.
        new_fold_rows: list[np.ndarray] = []
        off = 0
        for fr in fold_rows:
            new_fold_rows.append(np.arange(off, off + len(fr), dtype=np.int64))
            off += len(fr)
        rows = j["rows"][keep]
        units[key] = {
            "x": act["v_C"][rows].astype(np.float32),
            "y": act["v_A"][rows].astype(np.float32),
            "conv": [str(c) for c in order[keep]],
            "fold_rows": new_fold_rows,
            "n_join": int(len(keep)),
        }
        del act
        _log(f"[extdecomp] loaded {key} n_join={units[key]['n_join']}")
    return units


def accumulate_moments(units: dict[str, dict], k: int) -> list[dict]:
    """Per-fold pooled second moments over ALL units (float64, CPU) — the
    accumulate_pooled_moments math on the in-RAM joined arrays."""
    d = D_AMBIENT
    mom = [
        {
            "n": 0,
            "sum_x": torch.zeros(d, dtype=torch.float64),
            "sum_y": torch.zeros(d, dtype=torch.float64),
            "yss": 0.0,
            "c_xx": torch.zeros(d, d, dtype=torch.float64),
            "c_xy": torch.zeros(d, d, dtype=torch.float64),
        }
        for _ in range(k)
    ]
    for key, u in units.items():
        t0 = time.time()
        for f in range(k):
            idx = u["fold_rows"][f]
            x = torch.as_tensor(u["x"][idx].astype(np.float64))
            y = torch.as_tensor(u["y"][idx].astype(np.float64))
            m = mom[f]
            m["n"] += int(x.shape[0])
            m["sum_x"] += x.sum(0)
            m["sum_y"] += y.sum(0)
            m["yss"] += float((y * y).sum())
            m["c_xx"] += x.T @ x
            m["c_xy"] += x.T @ y
        _log(f"[extdecomp] moments {key} ({time.time() - t0:.1f}s)")
    return mom


def moments_path(work: Path) -> Path:
    return work / "moments.pt"


def save_moments(mom: list[dict], work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    tmp = moments_path(work).with_suffix(".tmp")
    torch.save(mom, tmp)
    tmp.replace(moments_path(work))


def fit_pooled_fold(mom: list[dict], f: int, k: int) -> PooledMomentRidge:
    train = {
        "n": sum(mom[g]["n"] for g in range(k) if g != f),
        "yss": sum(mom[g]["yss"] for g in range(k) if g != f),
    }
    for key in ("sum_x", "sum_y", "c_xx", "c_xy"):
        train[key] = sum(mom[g][key] for g in range(k) if g != f)
    return PooledMomentRidge(**train)


def run_fold(units: dict[str, dict], mom: list[dict], f: int, k: int, work: Path) -> None:
    """Fold f: pooled fit + per-model assistant-chat own map + score all units."""
    t_fold = time.time()
    m0 = fit_pooled_fold(mom, f, k)
    _log(
        f"[extdecomp] fold {f} pooled: n_train={m0.n_train:,} lam={m0.best_lambda:g} "
        f"dof={m0.dof:.0f} ({time.time() - t_fold:.1f}s)"
    )
    fold_out: dict[str, dict] = {"pooled_info": m0.info(), "units": {}}
    e2_store: dict[str, np.ndarray] = {}

    # DIRECT source maps: one SharedEighRidge per model, eval = concat of the
    # model's 6 units' fold-f rows (single eigh + single fit_predict).
    direct_preds: dict[str, np.ndarray] = {}
    for model in MODELS:
        src_key = _unit_key(ASSIST, "chat", model)
        src = units[src_key]
        tr = np.concatenate([src["fold_rows"][g] for g in range(k) if g != f])
        x_tr = src["x"][tr].astype(np.float64)
        y_tr = src["y"][tr].astype(np.float64)
        tgt_keys = [_unit_key(i, fr, model) for (i, fr) in SUBSET]
        spans: list[tuple[str, int, int]] = []
        blocks = []
        off = 0
        for tk in tgt_keys:
            xe = units[tk]["x"][units[tk]["fold_rows"][f]].astype(np.float64)
            spans.append((tk, off, off + xe.shape[0]))
            blocks.append(xe)
            off += xe.shape[0]
        t0 = time.time()
        core = SharedEighRidge(x_tr, np.concatenate(blocks, axis=0))
        preds_all, info = core.fit_predict(y_tr)
        fold_out[f"direct_info_{model}"] = {**info, "src": src_key}
        for tk, a, b in spans:
            direct_preds[tk] = preds_all[a:b]
        _log(f"[extdecomp] fold {f} direct map {model} ({time.time() - t0:.1f}s)")

    for key, u in units.items():
        te = u["fold_rows"][f]
        tr = np.concatenate([u["fold_rows"][g] for g in range(k) if g != f])
        x_tr, y_tr = u["x"][tr].astype(np.float64), u["y"][tr].astype(np.float64)
        x_te, y_te = u["x"][te].astype(np.float64), u["y"][te].astype(np.float64)
        preds = {"m0": m0.predict_np(x_te)}
        b_cf = y_tr.mean(axis=0) - m0.predict_np(x_tr.mean(axis=0, keepdims=True))[0]
        preds["m1"] = preds["m0"] + b_cf
        preds["direct"] = direct_preds[key]
        preds["identity_cell"] = identity_bias_predict(x_tr, y_tr, x_te)
        metrics = {n: reconstruction_metrics(preds[n], y_te) for n in RUNG_NAMES}
        knn = {
            n: {m: knn_retrieval(preds[n], y_te, metric=m) for m in ("euclidean", "cosine")}
            for n in ("m0", "m1", "direct")
        }
        e2_store[key] = np.stack([((preds[n] - y_te) ** 2).sum(axis=1) for n in RUNG_NAMES], axis=1)
        fold_out["units"][key] = {
            "n_test": int(len(te)),
            "n_cell_train": int(len(tr)),
            "m1_bias_norm": float(np.linalg.norm(b_cf)),
            "metrics": metrics,
            "knn": knn,
        }
        _log(
            f"[extdecomp] fold {f} {key} m0={metrics['m0']['r2']:+.4f} "
            f"m1={metrics['m1']['r2']:+.4f} direct={metrics['direct']['r2']:+.4f}"
        )

    work.mkdir(parents=True, exist_ok=True)
    npz_tmp = work / f"fold_{f}.e2.tmp.npz"
    np.savez(npz_tmp, **{k_: v for k_, v in e2_store.items()})
    npz_tmp.replace(work / f"fold_{f}.e2.npz")
    jtmp = work / f"fold_{f}.tmp.json"
    jtmp.write_text(json.dumps(fold_out, indent=1), encoding="utf-8")
    jtmp.replace(work / f"fold_{f}.json")
    _log(f"[extdecomp] fold {f} DONE ({time.time() - t_fold:.1f}s)")


def _ci(v: np.ndarray) -> dict:
    return {
        "lo": float(np.quantile(v, 0.025)),
        "hi": float(np.quantile(v, 0.975)),
        "mean": float(v.mean()),
    }


def finalize(
    units: dict[str, dict],
    local: dict[str, dict[str, Path]],
    fold_map: dict,
    work: Path,
    values_out: Path,
    figure_dir: Path,
    figure_stem: str,
) -> None:
    k = int(fold_map["k"])
    fold_json = {f: json.loads((work / f"fold_{f}.json").read_text()) for f in range(k)}
    fold_e2 = {f: np.load(work / f"fold_{f}.e2.npz", allow_pickle=False) for f in range(k)}

    # Per-unit assembled rows (scored order = fold 0..k-1 concatenation, which
    # IS the unit's row order by construction in load_units).
    per_unit: dict[str, dict] = {}
    for key, u in units.items():
        e2 = np.concatenate([np.asarray(fold_e2[f][key], dtype=np.float64) for f in range(k)])
        if e2.shape[0] != u["n_join"]:
            raise RuntimeError(f"{key}: e2 rows {e2.shape[0]} != n_join {u['n_join']}")
        y = u["y"].astype(np.float64)
        s2 = ((y - y.mean(axis=0)) ** 2).sum(axis=1)
        banked = json.loads(local[key]["fits"].read_text(encoding="utf-8"))
        arm_rep = banked["arm_reports"][ARM]
        ceil_folds = [rec["r2_ambient"] for rec in arm_rep["per_fold"]]
        per_unit[key] = {
            "e2": e2,
            "s2": s2,
            "conv": u["conv"],
            "ceiling_per_fold": ceil_folds,
            "ceiling_mean": float(arm_rep["pooled"]["r2_ambient_mean"]),
            "m1_per_fold": [fold_json[f]["units"][key]["metrics"]["m1"]["r2"] for f in range(k)],
            "direct_per_fold": [
                fold_json[f]["units"][key]["metrics"]["direct"]["r2"] for f in range(k)
            ],
            "m0_per_fold": [fold_json[f]["units"][key]["metrics"]["m0"]["r2"] for f in range(k)],
        }

    # Conversation-grouped bootstrap: ONE shared resample from the union.
    union = sorted({c for rec in per_unit.values() for c in rec["conv"]})
    u_index = {c: i for i, c in enumerate(union)}
    rng = np.random.default_rng(SEED_BASE)
    counts = np.zeros((BOOTSTRAP_DRAWS, len(union)), dtype=np.float64)
    for bi in range(BOOTSTRAP_DRAWS):
        counts[bi] = np.bincount(rng.integers(0, len(union), size=len(union)), minlength=len(union))
    im = {n: i for i, n in enumerate(RUNG_NAMES)}
    r2_draws: dict[str, np.ndarray] = {}
    r2_point: dict[str, dict[str, float]] = {}
    for key, rec in per_unit.items():
        cols = np.asarray([u_index[c] for c in rec["conv"]], dtype=np.int64)
        counts_c = counts[:, cols]
        den = counts_c @ rec["s2"]
        if not np.all(den > 0):
            raise RuntimeError(f"bootstrap draw with zero SS_tot for {key}")
        r2_draws[key] = 1.0 - (counts_c @ rec["e2"]) / den[:, None]  # (B, n_rungs)
        point = 1.0 - rec["e2"].sum(axis=0) / rec["s2"].sum()
        r2_point[key] = {n: float(point[im[n]]) for n in RUNG_NAMES}

    # Fold-aligned resampling for M2-involving intervals (banked per-fold
    # ceilings carry no per-row errors) — one shared fold resample across units.
    rng_f = np.random.default_rng(SEED_BASE + 1)
    fold_idx = rng_f.integers(0, k, size=(FOLD_RESAMPLE_DRAWS, k))

    def fold_ci(vals_by_unit: dict[str, np.ndarray]) -> dict[str, dict]:
        return {key: _ci(vals[fold_idx].mean(axis=1)) for key, vals in vals_by_unit.items()}

    m2_minus_m1 = {
        key: np.asarray(rec["ceiling_per_fold"]) - np.asarray(rec["m1_per_fold"])
        for key, rec in per_unit.items()
    }
    m2_minus_m1_ci = fold_ci(m2_minus_m1)

    # Instruct-minus-base deltas per cell (paired draws for m0/m1/direct;
    # fold-aligned for the banked M2 ceilings).
    deltas: dict[str, dict] = {}
    for identity, framing in SUBSET:
        kb = _unit_key(identity, framing, "qwen2.5-7b")
        ki = _unit_key(identity, framing, "qwen2.5-7b-instruct")
        cell_label = f"{identity}__{framing}"
        d_m2 = np.asarray(per_unit[ki]["ceiling_per_fold"]) - np.asarray(
            per_unit[kb]["ceiling_per_fold"]
        )
        deltas[cell_label] = {
            **{
                n: {
                    "point": r2_point[ki][n] - r2_point[kb][n],
                    **_ci(r2_draws[ki][:, im[n]] - r2_draws[kb][:, im[n]]),
                }
                for n in ("m0", "m1", "direct")
            },
            "m2": {
                "point": per_unit[ki]["ceiling_mean"] - per_unit[kb]["ceiling_mean"],
                **_ci(d_m2[fold_idx].mean(axis=1)),
            },
        }

    values = {
        "metadata": {
            **as_metadata_dict(git_provenance(_REPO)),
            "script_version": SCRIPT_VERSION,
            "argv": sys.argv,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "arm": ARM,
            "layer": LAYER,
            "condition": CONDITION,
            "hf_data_repo": HF_DATA_REPO,
            "hf_revision": HF_REVISION,
            "fold_map_source": fold_map["_source"],
            "fold_map_sha256": fold_map["_sha256"],
            "fold_k": k,
            "fold_seed": fold_map["seed"],
            "pooling": "one shared W over the 12 subset units (6 cells x 2 models), "
            "the #1639 joint-across-model nesting (pool_specialize pilot precedent)",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_sstot_convention": "fixed_full_scored_mean",
            "bootstrap_coupling": "shared conversation resample applied to every unit "
            "(base/instruct paired for delta CIs)",
            "m2_interval_convention": f"fold-aligned resampling over the {k} shared folds "
            f"({FOLD_RESAMPLE_DRAWS} draws) — banked ceilings carry no per-row errors",
            "excluded": "bare_label story form (trailing-space digit-start tokenization "
            "artifact disclosed in #2054)",
            "estimator": "GCV ridge, dof cap 0.9, standardize-X/center-Y (SharedEighRidge / "
            "PooledMomentRidge fit_h parity)",
        },
        "units": {},
        "deltas_instruct_minus_base": deltas,
        "pooled_info_per_fold": {str(f): fold_json[f]["pooled_info"] for f in range(k)},
        "direct_source_info_per_fold": {
            str(f): {m: fold_json[f][f"direct_info_{m}"] for m in MODELS} for f in range(k)
        },
    }
    for key, rec in per_unit.items():
        banked_ladder_ceiling = None  # cross-checked at figure time from ladder.json when present
        values["units"][key] = {
            "n_join": units[key]["n_join"],
            "n_train_per_fold": [fold_json[f]["units"][key]["n_cell_train"] for f in range(k)],
            "r2_point_fixed_mean": r2_point[key],
            "r2_ci": {n: _ci(r2_draws[key][:, im[n]]) for n in ("m0", "m1", "direct")},
            "r2_per_fold": {
                "m0": rec["m0_per_fold"],
                "m1": rec["m1_per_fold"],
                "direct": rec["direct_per_fold"],
                "m2_banked_ceiling": rec["ceiling_per_fold"],
            },
            "m2": {
                "point_fold_mean": rec["ceiling_mean"],
                "ci_fold_aligned": _ci(np.asarray(rec["ceiling_per_fold"])[fold_idx].mean(axis=1)),
                "source": f"issue2054_lattice/fits/{key}.json @ {HF_REVISION} "
                "(arm_reports.context.per_fold[].r2_ambient)",
                "ladder_json_crosscheck": banked_ladder_ceiling,
            },
            "increment_m1_minus_m0": {
                "point": r2_point[key]["m1"] - r2_point[key]["m0"],
                **_ci(r2_draws[key][:, im["m1"]] - r2_draws[key][:, im["m0"]]),
            },
            "increment_m2_minus_m1": {
                "point": rec["ceiling_mean"] - float(np.mean(rec["m1_per_fold"])),
                **m2_minus_m1_ci[key],
            },
            "identity_bias_baseline_r2": r2_point[key]["identity_cell"],
            "knn_fold0": fold_json[0]["units"][key]["knn"],
            "direct_own_refit_vs_banked_ceiling": (
                {
                    "refit_fold_mean": float(np.mean(rec["direct_per_fold"])),
                    "banked_fold_mean": rec["ceiling_mean"],
                    "abs_diff": abs(float(np.mean(rec["direct_per_fold"])) - rec["ceiling_mean"]),
                }
                if key.startswith(f"{ASSIST}__{CONDITION}__chat__")
                else None
            ),
        }

    # ladder.json cross-check where the committed aggregate is present.
    ladder_path = _REPO / "eval_results/issue_2054/specialization_ladder/ladder.json"
    if ladder_path.is_file():
        ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
        by_cell = {u["cell"]: u["ceiling_r2"] for u in ladder["units"] if u["arm"] == ARM}
        for key in per_unit:
            if key in by_cell:
                values["units"][key]["m2"]["ladder_json_crosscheck"] = {
                    "ladder_ceiling_r2": by_cell[key],
                    "abs_diff_vs_fits": abs(by_cell[key] - per_unit[key]["ceiling_mean"]),
                }

    values_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = values_out.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(values, indent=1), encoding="utf-8")
    tmp.replace(values_out)
    _log(f"[extdecomp] values -> {values_out}")

    render_figure(values, figure_dir, figure_stem)


CELL_LABELS = {
    f"{ASSIST}__chat": "Assistant\n(chat template)",
    f"{ASSIST}__attrib_quoted": "Assistant\n(in story, quoted)",
    "char_helios__attrib_quoted": "HELIOS\n(story character)",
    "char_wren__attrib_quoted": "Wren\n(story character)",
    "char_dana__attrib_quoted": "Dana\n(story character)",
    "char_vex__attrib_quoted": "Vex\n(story character)",
}
RUNG_LABELS = {
    "m0": "Pooled map\n(shared W,\nglobal offset)",
    "m1": "Pooled map\n+ per-cell\noffset",
    "m2": "Own-map\nceiling\n(per-cell W)",
    "direct": "Direct from\nassistant-chat\nmap",
}


def render_figure(values: dict, figure_dir: Path, stem: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper

    rungs = ("m0", "m1", "m2", "direct")
    cells = [f"{i}__{f}" for (i, f) in SUBSET]
    fig, axes = plt.subplots(
        2, len(cells), figsize=(3.0 * len(cells), 7.2), sharey="row", constrained_layout=True
    )
    colors = {"qwen2.5-7b": "#8da0cb", "qwen2.5-7b-instruct": "#fc8d62"}
    model_labels = {
        "qwen2.5-7b": "Base (Qwen2.5-7B)",
        "qwen2.5-7b-instruct": "Instruct (Qwen2.5-7B-Instruct)",
    }
    width = 0.38
    xs = np.arange(len(rungs))
    for ci, cell in enumerate(cells):
        ax = axes[0, ci]
        for mi, model in enumerate(MODELS):
            key = f"{cell.split('__')[0]}__{CONDITION}__{cell.split('__', 1)[1]}__{model}"
            uv = values["units"][key]
            heights, lo, hi = [], [], []
            for r in rungs:
                if r == "m2":
                    pt = uv["m2"]["point_fold_mean"]
                    ci_r = uv["m2"]["ci_fold_aligned"]
                else:
                    pt = uv["r2_point_fixed_mean"][r]
                    ci_r = uv["r2_ci"][r]
                heights.append(pt)
                lo.append(pt - ci_r["lo"])
                hi.append(ci_r["hi"] - pt)
            ax.bar(
                xs + (mi - 0.5) * width,
                heights,
                width,
                yerr=[lo, hi],
                color=colors[model],
                label=model_labels[model] if ci == 0 else None,
                error_kw={"lw": 0.9},
            )
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_title(CELL_LABELS[cell], fontsize=9, loc="left")
        ax.set_xticks(xs)
        ax.set_xticklabels([RUNG_LABELS[r] for r in rungs], fontsize=6.5)
        if ci == 0:
            ax.set_ylabel("Held-out $R^2$ (context → answer)")

        # Delta row: instruct minus base per rung.
        axd = axes[1, ci]
        dl = values["deltas_instruct_minus_base"][cell]
        pts = [dl[r]["point"] for r in rungs]
        dlo = [dl[r]["point"] - dl[r]["lo"] for r in rungs]
        dhi = [dl[r]["hi"] - dl[r]["point"] for r in rungs]
        axd.bar(xs, pts, 0.6, yerr=[dlo, dhi], color="#66c2a5", error_kw={"lw": 0.9})
        axd.axhline(0.0, color="0.4", lw=0.8)
        axd.set_xticks(xs)
        axd.set_xticklabels([RUNG_LABELS[r] for r in rungs], fontsize=6.5)
        if ci == 0:
            axd.set_ylabel("Instruct − base $\\Delta R^2$")
    axes[0, 0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle(
        "Shared-vs-specific decomposition over assistant + story characters "
        f"(on-policy, layer {LAYER}, context arm)",
        fontsize=11,
    )
    out = savefig_paper(fig, stem, dir=str(figure_dir))
    plt.close(fig)
    _log(f"[extdecomp] figure -> {out}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--accumulate-only", action="store_true")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--figure-only", action="store_true", help="re-render from values.json")
    ap.add_argument("--fold-map-ref", default="origin/issue-2054")
    ap.add_argument("--fold-map-file", default=None)
    ap.add_argument(
        "--smoke-rows", type=int, default=None, help="rows per unit (smoke); diverts outputs"
    )
    ap.add_argument(
        "--values-out",
        type=Path,
        default=_REPO / "eval_results/issue_2054/extended_decomp/values.json",
    )
    ap.add_argument(
        "--figure-dir", type=Path, default=_REPO / "figures/issue_2054/followup_extended_decomp"
    )
    ap.add_argument("--figure-stem", default="extended_decomp")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[extdecomp] import-check OK")
        return 0

    smoke = args.smoke_rows is not None
    stage_root = args.stage_root
    work = stage_root / ("work_smoke" if smoke else "work")
    values_out = Path("/tmp/issue2054-extdecomp-smoke/values.json") if smoke else args.values_out
    figure_dir = Path("/tmp/issue2054-extdecomp-smoke/figures") if smoke else args.figure_dir

    if args.stage_only:
        stage(stage_root)
        return 0

    if args.figure_only:
        values = json.loads(values_out.read_text(encoding="utf-8"))
        render_figure(values, figure_dir, args.figure_stem)
        return 0

    fold_map = load_fold_map(args.fold_map_file, args.fold_map_ref)
    k = int(fold_map["k"])
    _log(
        f"[extdecomp] fold map {fold_map['_source']} k={k} seed={fold_map['seed']} "
        f"n_conv={len(fold_map['fold_of']):,}"
    )
    local = staged_paths(stage_root)
    units = load_units(local, fold_map, args.smoke_rows)

    if args.accumulate_only:
        mom = accumulate_moments(units, k)
        save_moments(mom, work)
        _log(f"[extdecomp] moments -> {moments_path(work)}")
        return 0

    if args.folds:
        mom = torch.load(moments_path(work), weights_only=False)  # self-produced checkpoint
        for f in args.folds:
            if not 0 <= f < k:
                raise ValueError(f"fold {f} out of range for k={k}")
            run_fold(units, mom, f, k, work)
        return 0

    if args.finalize:
        for f in range(k):
            if not (work / f"fold_{f}.json").is_file():
                raise FileNotFoundError(f"fold {f} checkpoint missing — run --folds {f} first")
        finalize(units, local, fold_map, work, values_out, figure_dir, args.figure_stem)
        return 0

    raise SystemExit("pass one of --stage-only / --accumulate-only / --folds / --finalize")


if __name__ == "__main__":
    raise SystemExit(main())
