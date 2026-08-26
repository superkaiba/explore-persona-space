"""#2564 grammar-slot one-word PILOT — VM-side reads.

Consumes the pilot artifacts produced by ``scripts/issue2564_gramslot_pilot_run.py``
(HF ``superkaiba1/explore-persona-space-data`` under
``issue2564_minpair/gramslot_pilot/``, or a local out-root via ``--local``) and
mirrors the langow pilot reads (``scripts/issue2564_langow_pilot_reads.py``) on
the three grammar-slot axes (``query_oneword_subject`` / ``query_oneword_object``
/ ``query_oneword_verb``):

- the two frozen ridge arms loaded exactly as the parent (#779 single-turn +
  #1738 multi-turn L19 ridges via ``apply_map``), plus the raw ``arm_iddelta``
  v_C-delta baseline;
- per-pair rows whose schema (key set AND order) matches the langow pilot's
  flattened ``perpair.jsonl`` EXACTLY (asserted per row, and against the
  committed langow artifact at ``--import-check``), so
  ``scripts/issue2564_shift_bars.py`` ingests them with a minimal extension —
  ``carrier`` holds the frame_id; ``fired_a_70``/``fired_b_70``/
  ``pair_fired_70`` are None on every row (query axes, no fired gate — the
  langow oneword convention); ``norm_text`` stays null;
- per-pair norm ratio (via ``norm_obs_tail_L19`` + ``norm_pred_<arm>`` per row
  and per-class ratio medians in the summary), delta cosine per arm, pair-delta
  retrieval within this pilot's own 96-context / 72-pair pool, per-context
  kNN retrieval, calibration slopes, split-half reliability — all through the
  reused langow implementations.

Reuse: the langow reads module is imported by path under a unique name with
TWO rebinds — ``RD.CELLS = ("query_gramslot",)`` and ``RD.HF_PREFIX`` — so its
``stage_inputs`` / ``load_pilot`` enumerate the gramslot file set (including
the langow-flavored capture filenames the reused run-side writers produce).
No figures here: the cross-pilot figure is ``issue2564_shift_bars.py``.

Outputs: ``eval_results/issue_2564/gramslot_pilot/{perpair.jsonl, summary.json}``.

Run (VM):

    uv run python scripts/issue2564_gramslot_pilot_reads.py            # stage from HF
    uv run python scripts/issue2564_gramslot_pilot_reads.py --local /workspace/eps2564_gramslot
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/torch (transitively via the langow reads module)

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic, write_jsonl_atomic  # noqa: E402

logger = logging.getLogger("issue2564_gramslot_reads")

REPO_ROOT = Path(__file__).resolve().parent.parent
assert (REPO_ROOT / "pyproject.toml").is_file(), REPO_ROOT


def _load_by_path(name: str, path: Path):
    """Import a main-resident script under a UNIQUE module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RD = _load_by_path(
    "issue2564_langow_pilot_reads_for_gramslot",
    REPO_ROOT / "scripts" / "issue2564_langow_pilot_reads.py",
)

ISSUE = 2564
CELL = "query_gramslot"
HF_PREFIX = "issue2564_minpair/gramslot_pilot"
PAIR_CLASSES = ("query_oneword_subject", "query_oneword_object", "query_oneword_verb")

# Rebind the reused langow reads module to the gramslot layout: CELLS drives
# the stage_inputs + load_pilot file enumeration, HF_PREFIX the staging
# source. Private module copy (unique sys.modules name) — the real langow
# module is untouched.
RD.CELLS = (CELL,)
RD.HF_PREFIX = HF_PREFIX

OUT_DIR = REPO_ROOT / "eval_results" / "issue_2564" / "gramslot_pilot"
LANGOW_PERPAIR = REPO_ROOT / "eval_results" / "issue_2564" / "lang_oneword_pilot" / "perpair.jsonl"


def _perpair_keys() -> tuple[str, ...]:
    """The langow flattened perpair schema, key set AND order (asserted against
    the committed langow artifact at --import-check)."""
    keys = [
        "pair_id",
        "pair_class",
        "axis",
        "carrier",
        "value_a",
        "value_b",
        "orientation",
        "changed_tokens",
        "n_draws_a",
        "n_draws_b",
        "ans_len_delta",
        "norm_obs_tail_L19",
        "norm_obs_span_L19",
        "norm_text",
        "r_half",
        "r10",
        "noise_norm",
        "fired_a_70",
        "fired_b_70",
        "pair_fired_70",
    ]
    for arm in RD.ALL_ARMS:
        keys += [f"cos_{arm}", f"cos_span_{arm}", f"norm_pred_{arm}"]
    keys += [f"cos_vs_iddelta_{arm}" for arm in RD.ARMS]
    return tuple(keys)


PERPAIR_KEYS = _perpair_keys()


def _assert_schema_matches_langow() -> None:
    """Key-list equality (set AND order) against the committed langow artifact."""
    with LANGOW_PERPAIR.open(encoding="utf-8") as fh:
        langow_keys = tuple(json.loads(fh.readline()).keys())
    assert PERPAIR_KEYS == langow_keys, {
        "only_gramslot": sorted(set(PERPAIR_KEYS) - set(langow_keys)),
        "only_langow": sorted(set(langow_keys) - set(PERPAIR_KEYS)),
        "order_matches": sorted(PERPAIR_KEYS) == sorted(langow_keys),
    }


# ── analysis (langow compute minus the language leg, gramslot axes) ───────


def compute(pilot: dict, ridge_paths: dict[str, Path]) -> tuple[list[dict], dict]:
    apply_map = RD._import_apply_map()
    dev = torch.device("cpu")
    d = pilot["vc19"].shape[1]

    mapped: dict[str, np.ndarray] = {}
    for arm in RD.ARMS:
        payload = RD.load_ridge_payload(ridge_paths[arm], d, arm)
        mapped[arm] = np.asarray(apply_map(payload, pilot["vc19"], dev), dtype=np.float64)
        assert mapped[arm].shape == pilot["vc19"].shape, (arm, mapped[arm].shape)

    valid = pilot["valid"]
    counts = valid.sum(axis=1)
    with np.errstate(invalid="ignore"):
        tail_mean = (
            np.einsum("ck,ckd->cd", valid.astype(np.float64), pilot["tail"].astype(np.float64))
            / np.maximum(counts, 1)[:, None]
        )
        span_mean = (
            np.einsum("ck,ckd->cd", valid.astype(np.float64), pilot["span"].astype(np.float64))
            / np.maximum(counts, 1)[:, None]
        )

    ctx_pos = pilot["ctx_pos"]
    pairs = pilot["pairs"]
    ai = np.array([ctx_pos[p["a"]] for p in pairs])
    bi = np.array([ctx_pos[p["b"]] for p in pairs])

    obs_tail = tail_mean[ai] - tail_mean[bi]
    obs_span = span_mean[ai] - span_mean[bi]
    pred = {arm: mapped[arm][ai] - mapped[arm][bi] for arm in RD.ARMS}
    pred["arm_iddelta"] = pilot["vc19"][ai] - pilot["vc19"][bi]

    r_half, r10, noise_norm = RD.split_half_stats(pilot["tail"], valid, ai, bi)

    def _len_mean(cid: str) -> float:
        vals = [
            len(t)
            for t, ok in zip(pilot["texts"][cid], pilot["valid"][ctx_pos[cid]])
            if ok and t is not None
        ]
        return float(np.mean(vals)) if vals else float("nan")

    rows: list[dict] = []
    for pi, p in enumerate(pairs):
        row = {
            "pair_id": p["pair_id"],
            "pair_class": p["pair_class"],
            "axis": p["axis"],
            "carrier": p["carrier"],  # frame_id (f01..f24)
            "value_a": p["value_a"],
            "value_b": p["value_b"],
            "orientation": "base-to-variant",  # a = variant, b = base; delta = variant - base
            "changed_tokens": p["changed_tokens"],
            "n_draws_a": int(counts[ai[pi]]),
            "n_draws_b": int(counts[bi[pi]]),
            "ans_len_delta": _len_mean(p["a"]) - _len_mean(p["b"]),
            "norm_obs_tail_L19": float(np.linalg.norm(obs_tail[pi])),
            "norm_obs_span_L19": float(np.linalg.norm(obs_span[pi])),
            "norm_text": None,  # parent's text-embedding leg out of pilot scope (langow parity)
            "r_half": float(r_half[pi]),
            "r10": float(r10[pi]),
            "noise_norm": float(noise_norm[pi]),
            # Query axes carry no fired gate (langow oneword convention).
            "fired_a_70": None,
            "fired_b_70": None,
            "pair_fired_70": None,
        }
        for arm in RD.ALL_ARMS:
            row[f"cos_{arm}"] = float(RD.rowwise_cos(pred[arm][pi], obs_tail[pi]))
            row[f"cos_span_{arm}"] = float(RD.rowwise_cos(pred[arm][pi], obs_span[pi]))
            row[f"norm_pred_{arm}"] = float(np.linalg.norm(pred[arm][pi]))
        for arm in RD.ARMS:
            row[f"cos_vs_iddelta_{arm}"] = float(
                RD.rowwise_cos(pred[arm][pi], pred["arm_iddelta"][pi])
            )
        assert tuple(row) == PERPAIR_KEYS, (p["pair_id"], sorted(set(row) ^ set(PERPAIR_KEYS)))
        rows.append(row)

    # Retrieval reads (this pilot's own pool).
    n_pool = len(pilot["ctx_ids"])
    retrieval: dict[str, dict] = {}
    for arm in RD.ARMS:
        retrieval[arm] = {
            "per_context": {
                metric: knn_retrieval(mapped[arm], tail_mean, ks=(1,), metric=metric)
                for metric in ("cosine", "euclidean")
            },
            "chance_at_1": 1.0 / n_pool,
            "n_pool": n_pool,
        }
    pair_rank: dict[str, dict] = {}
    for arm in RD.ALL_ARMS:
        cs = RD.cross_cos(pred[arm], obs_tail)  # (n_pairs_pred, n_pairs_obs)
        order = np.argsort(-np.nan_to_num(cs, nan=-np.inf), axis=1)
        ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(len(pairs))])
        by_axis = {}
        for axis in PAIR_CLASSES:
            m = np.array([p["axis"] == axis for p in pairs])
            by_axis[axis] = {
                "acc_at_1": float(np.mean(ranks[m] == 1)),
                "median_rank": float(np.median(ranks[m])),
                "n": int(m.sum()),
            }
        pair_rank[arm] = {
            "acc_at_1": float(np.mean(ranks == 1)),
            "median_rank": float(np.median(ranks)),
            "chance_at_1": 1.0 / len(pairs),
            "by_axis": by_axis,
        }

    # Calibration slope per axis per arm.
    calibration: dict[str, dict[str, float]] = {}
    obs_norm_all = np.linalg.norm(obs_tail, axis=1)
    for arm in RD.ALL_ARMS:
        pred_norm = np.array([r[f"norm_pred_{arm}"] for r in rows])
        calibration[arm] = {}
        for axis in (*PAIR_CLASSES, "all"):
            m = (
                np.ones(len(rows), dtype=bool)
                if axis == "all"
                else np.array([r["axis"] == axis for r in rows])
            )
            calibration[arm][axis] = RD.through_origin_slope(pred_norm[m], obs_norm_all[m])

    def _med(vals: list[float]) -> float:
        return float(np.nanmedian(vals)) if vals else float("nan")

    cos_median = {
        arm: {
            axis: _med([r[f"cos_{arm}"] for r in rows if r["axis"] == axis])
            for axis in PAIR_CLASSES
        }
        for arm in RD.ALL_ARMS
    }
    # Per-pair norm ratio, summarized per class (the shift_bars bottom-panel read).
    ratio_median = {
        arm: {
            axis: _med(
                [
                    r[f"norm_pred_{arm}"] / r["norm_obs_tail_L19"]
                    for r in rows
                    if r["axis"] == axis and r["norm_obs_tail_L19"] > 0
                ]
            )
            for axis in PAIR_CLASSES
        }
        for arm in RD.ALL_ARMS
    }

    summary = {
        "issue": ISSUE,
        "n_contexts": n_pool,
        "n_pairs": len(pairs),
        "n_pairs_by_class": {
            cls: sum(1 for p in pairs if p["pair_class"] == cls) for cls in PAIR_CLASSES
        },
        "n_frames": len({p["carrier"] for p in pairs}),
        "k_draws": pilot["k"],
        "layer_used": pilot["layer_used"],
        "arms": list(RD.ALL_ARMS),
        "ridge_paths": {"arm_779ce": RD.RIDGE_779_PATH, "arm_1738ce": RD.RIDGE_1738_PATH},
        "cos_median_by_axis_arm": cos_median,
        "norm_ratio_median_by_axis_arm": ratio_median,
        "retrieval_per_context": retrieval,
        "retrieval_pair_rank": pair_rank,
        "calibration_slope": calibration,
        "notes": {
            "norm_text": "null in perpair — parent's text-embedding leg out of pilot scope",
            "orientation": "base-to-variant (a = variant, b = base; delta = variant - base)",
            "fired": "query axes — no fired gate; fired_*/pair_fired_70 are None on every row",
            "carrier": "frame_id (f01..f24) plays the carrier role",
        },
        "repro": RD._repro_meta(),
    }
    return rows, summary


# ── main ──────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--local", default=None, help="local pilot out-root (skip HF staging of pilot artifacts)"
    )
    ap.add_argument(
        "--stage-dir",
        default=None,
        help="staging dir (default: data/issue_2564/gramslot_stage under repo root)",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def _import_check() -> None:
    import inspect

    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    assert RD.CELLS == (CELL,), RD.CELLS
    assert RD.HF_PREFIX == HF_PREFIX, RD.HF_PREFIX
    for name in (
        "stage_inputs",
        "stage_ridge_payloads",
        "load_ridge_payload",
        "load_pilot",
        "split_half_stats",
        "rowwise_cos",
        "cross_cos",
        "through_origin_slope",
        "_import_apply_map",
        "_repro_meta",
    ):
        assert callable(getattr(RD, name)), name
    apply_map = RD._import_apply_map()
    params = set(inspect.signature(apply_map).parameters)
    assert {"payload"} <= params or len(params) >= 3, params
    assert callable(knn_retrieval)
    _assert_schema_matches_langow()
    print("[import-check] ok: langow reads surface + perpair schema parity", flush=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_argparser().parse_args()
    if args.import_check:
        _import_check()
        return 0
    _assert_schema_matches_langow()
    stage_dir = (
        Path(args.stage_dir)
        if args.stage_dir
        else (REPO_ROOT / "data" / "issue_2564" / "gramslot_stage")
    )
    root = RD.stage_inputs(args.local, stage_dir)
    ridge_paths = RD.stage_ridge_payloads(stage_dir)
    pilot = RD.load_pilot(root)
    print(
        f"[load] {len(pilot['ctx_ids'])} contexts / {len(pilot['pairs'])} pairs / "
        f"K={pilot['k']} / layer={pilot['layer_used']}",
        flush=True,
    )
    # Grid-completeness gate (production reads only run on the full pilot).
    assert len(pilot["ctx_ids"]) == 96 and len(pilot["pairs"]) == 72, (
        len(pilot["ctx_ids"]),
        len(pilot["pairs"]),
    )
    rows, summary = compute(pilot, ridge_paths)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(OUT_DIR / "perpair.jsonl", rows)
    write_json_atomic(OUT_DIR / "summary.json", summary)
    print(f"[out] {OUT_DIR / 'perpair.jsonl'} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
