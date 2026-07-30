"""#1774 P6 — decode-noise ceiling (ICC) + cross-phase aggregation (VM, detached).

Per plan §4 P6:
- per-direction decode-noise floors from the P1 draws (within-context
  across-draw variance vs between-context variance, per r_B / answer-PC /
  top-singular direction; ICC = between/(between+within)), computed on the
  SAME 2,216-context draw subset as the floor, with the FULL-corpus
  between-context variance reported alongside (floor-subset mismatch guard);
- floor-relative gating for every Q3 per-trait cell: a cell whose
  between-context signal (draw subset) is under 2× its decode-noise floor is
  labeled noise-limited, never "unpredictable" (plan H3);
- merges the landed phase JSONs into eval_results/issue_1774/aggregate/ and
  writes eval_results/issue_1774/noise_ceiling.json.

Inputs resolve LOCAL-first (data/issue_1774/...), else stage from the HF data
repo prefix ``issue1774_operator_reads/`` via the scoped ``stage_hub_prefix``
(#833 recipe) — never an unscoped listing on the ~1M-file repo.

Usage: uv run python scripts/issue1774_aggregate.py [--layers 14,18,19]
       [--out-root D] [--no-hf-stage]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps + .env bind BEFORE the heavy imports below (BLAS/torch
# pools freeze at import time; tests/test_shared_vm_thread_caps.py).
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue1774_common as c  # noqa: E402

N_ANSWER_PCS = 10
N_TOP_SV = 10
NOISE_LIMITED_FACTOR = 2.0  # plan H3: signal < 2× floor ⇒ noise-limited


def _stage_if_missing(local_dir: Path, hub_prefix: str, allow_stage: bool) -> Path:
    if local_dir.exists() and any(local_dir.iterdir()):
        return local_dir
    if not allow_stage:
        raise FileNotFoundError(f"{local_dir} missing and --no-hf-stage set")
    import shutil

    from explore_persona_space.orchestrate import hub

    print(f"[p6] staging {hub_prefix} -> {local_dir}")
    # stage_hub_prefix lands files at dest/<repo-relative path> (verbatim prefix
    # mirror, #1402) — stage into a sibling mirror root, then move the leaf into
    # the consumed layout (the att-20260729-033609 P0 crash class, check (h)(iv)).
    mirror = local_dir.parent / f".hfstage_{local_dir.name}"
    if mirror.exists():
        shutil.rmtree(mirror)
    hub.stage_hub_prefix(c.DATA_REPO, hub_prefix, mirror, repo_type="dataset")
    staged_leaf = mirror / hub_prefix
    assert staged_leaf.is_dir() and any(staged_leaf.iterdir()), (
        f"verbatim mirror leaf missing/empty after staging: {staged_leaf}"
    )
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_leaf.rename(local_dir)
    shutil.rmtree(mirror, ignore_errors=True)
    return local_dir


def load_draw_stack(summaries_dir: Path, layer: int) -> tuple[np.ndarray, list[dict]]:
    """(n_ctx, k_draws, D) fp64 draw stack + the aligned row index."""
    idx_files = sorted(summaries_dir.glob("row_index_shard*.jsonl"))
    assert idx_files, f"no row_index shards under {summaries_dir}"
    rows: list[dict] = []
    per_shard: list[np.ndarray] = []
    draws_seen: set[int] = set()
    for idx_path in idx_files:
        tag = idx_path.stem.replace("row_index_shard", "")
        shard_rows = c.jsonl_rows(idx_path)
        rows.extend(shard_rows)
        draw_arrays = []
        k = 0
        while True:
            parts = sorted(
                summaries_dir.glob(f"t1_L{layer}_draw{k}_shard{tag}_part*.npy"),
                key=lambda p: int(p.stem.rsplit("part", 1)[1]),
            )
            if not parts:
                break
            arr = np.concatenate([np.load(p) for p in parts], axis=0)
            assert arr.shape == (len(shard_rows), c.HIDDEN_DIM), (arr.shape, len(shard_rows))
            draw_arrays.append(arr.astype(np.float64))
            draws_seen.add(k)
            k += 1
        assert draw_arrays, f"no draw parts for shard {tag} L{layer}"
        per_shard.append(np.stack(draw_arrays, axis=1))  # (n_shard, k, D)
    ks = {a.shape[1] for a in per_shard}
    assert len(ks) == 1, f"shards disagree on draw count: {ks}"
    stack = np.concatenate(per_shard, axis=0)
    assert stack.shape[0] == len(rows), (stack.shape, len(rows))
    return stack, rows


def direction_bank(layer: int, op_dir: Path) -> dict[str, np.ndarray]:
    """Unit-norm answer-side directions: r_B ×3, answer-PCs ×10, top left-SVs of W_ctx."""
    dirs: dict[str, np.ndarray] = {}
    rb = c.load_rb_bank(layer)
    for t, v in rb.items():
        dirs[f"rb_{t}"] = np.asarray(v, np.float64) / np.linalg.norm(v)
    # answer PCs: pooled top-10 right singular vectors of centered fit-row t1
    # (the SAME convention as fit_battery.step_q3's answer_pcs basis).
    rows = c.load_manifest()
    fit_idx = np.asarray(c.fit_indices(rows), dtype=np.int64)
    Y = np.asarray(c.load_summary_rows(c.CELL, "t1", layer)[fit_idx], dtype=np.float64)
    Yc = torch.from_numpy(Y - Y.mean(0, keepdims=True))
    _u, _s, vh = c.svd_robust(Yc)
    for j in range(N_ANSWER_PCS):
        dirs[f"answer_pc{j}"] = vh[j].numpy()
    w_path = op_dir / f"W_arm_context_L{layer}.npy"
    if w_path.exists():
        W = torch.from_numpy(np.load(w_path)).double()
        U, S, _vh2 = c.svd_robust(W)
        for j in range(min(N_TOP_SV, S.shape[0])):
            dirs[f"ctx_left_sv{j}"] = U[:, j].numpy()
    else:
        print(f"[p6] note: {w_path} absent — top-singular floors skipped")
    # full-corpus between-context variance per direction rides along
    dirs["__full_corpus_t1__"] = Y  # sentinel entry consumed by noise_floors()
    return dirs


def noise_floors(stack: np.ndarray, dirs: dict[str, np.ndarray]) -> dict:
    """Per-direction ICC + floors on the draw subset (+ full-corpus between-var)."""
    Y_full = dirs.pop("__full_corpus_t1__")
    n_ctx, k_draws, _d = stack.shape
    out: dict[str, dict] = {}
    for name, v in dirs.items():
        proj = stack @ v  # (n_ctx, k)
        within = float(np.mean(np.var(proj, axis=1, ddof=1)))
        between_subset = float(np.var(proj.mean(axis=1), ddof=1))
        between_full = float(np.var(Y_full @ v, ddof=1))
        icc = between_subset / max(between_subset + within, 1e-30)
        out[name] = {
            "within_context_var_floor": within,
            "between_context_var_draw_subset": between_subset,
            "between_context_var_full_corpus": between_full,
            "icc": icc,
        }
    dirs["__full_corpus_t1__"] = Y_full
    return {"n_contexts": int(n_ctx), "k_draws": int(k_draws), "directions": out}


def gate_per_trait_cells(eval_root: Path, floors: dict, layer: int) -> dict:
    """Floor-relative gating of every Q3 per-trait cell (plan H3)."""
    gated: dict[str, dict] = {}
    for arm in c.ARMS:
        p = eval_root / "channels" / f"{arm}_L{layer}.json"
        if not p.exists():
            gated[arm] = {"skipped": f"missing {p.name}"}
            continue
        ch = json.loads(p.read_text())
        per_trait = ch.get("per_trait_heldout_r2", {})
        row: dict[str, dict] = {}
        for t, r2 in per_trait.items():
            f = floors["directions"].get(f"rb_{t}")
            if f is None:
                row[t] = {"r2": r2, "gate": "no-floor"}
                continue
            signal = f["between_context_var_draw_subset"]
            floor = f["within_context_var_floor"]
            noise_limited = bool(signal < NOISE_LIMITED_FACTOR * floor)
            row[t] = {
                "r2": r2,
                "between_var_draw_subset": signal,
                "noise_floor": floor,
                "signal_over_floor": signal / max(floor, 1e-30),
                "label": "noise-limited" if noise_limited else "resolved",
            }
        gated[arm] = row
    return gated


def _digest_json(path: Path, keys: list[str]) -> dict | None:
    if not path.exists():
        return None
    j = json.loads(path.read_text())
    return {k: j.get(k) for k in keys if k in j}


def merge_phase_digests(eval_root: Path, layers: list[int]) -> dict:
    """Compact per-phase digests (headline numbers only; full JSONs stay in place)."""
    dig: dict = {"skipped": []}
    for layer in layers:
        for arm in c.ARMS:
            d1 = _digest_json(
                eval_root / "fit_battery" / f"{arm}_L{layer}.json",
                ["r2_per_context_pooled_oof", "n_fit_rows", "weighted_dedup"],
            )
            if d1 is not None:
                dig[f"fit_{arm}_L{layer}"] = d1
            d2 = _digest_json(
                eval_root / "channels" / f"{arm}_L{layer}.json",
                ["channel_count", "count_null_band", "bh_companion_count", "rho1_sq_mean"],
            )
            if d2 is not None:
                if isinstance(d2.get("count_null_band"), dict):
                    d2["count_null_band"] = {
                        k: v for k, v in d2["count_null_band"].items() if k != "null_counts"
                    }
                dig[f"channels_{arm}_L{layer}"] = d2
    for name, path, keys in [
        (
            "parity",
            eval_root / "fit_battery" / "parity_banked_convention_L14.json",
            ["r2_per_context_pooled_oof", "n_rows"],
        ),
        (
            "q1a",
            eval_root / "fit_battery" / "q1a_joint_vs_marginal_L14.json",
            ["stitch_r2_per_context_oof", "observed"],
        ),
        (
            "q1b",
            eval_root / "fit_battery" / "q1b_chain_L14.json",
            [
                "r2_g_e_to_vbar",
                "r2_chain_Mavg_g_e",
                "r2_direct_e_to_abar",
                "r2_direct_vbar_to_abar",
                "chain_recovered_share_of_deficit",
            ],
        ),
        (
            "endomorphism",
            eval_root / "endomorphism" / "context_L14.json",
            ["gate", "trace_over_d", "spectral_radius", "trait_gain_matrix"],
        ),
        ("cokernel", eval_root / "nullspace" / "cokernel_all_L14.json", ["context_sweep_spread"]),
        (
            "state_shift",
            eval_root / "steering" / "state_shift.json",
            # REAL merge_state_shift keys (round 2, M3 sibling fix)
            [
                "conditions",
                "steer_base_band",
                "alpha_by_direction",
                "n_usable_directions",
                "judge_skip",
            ],
        ),
    ]:
        d = _digest_json(path, keys)
        if d is not None and name == "state_shift":
            # compact: per-condition headline stats only; band pooled stats only
            if isinstance(d.get("conditions"), dict):
                d["conditions"] = {
                    k: {kk: v.get(kk) for kk in ("median_dt1", "p90_dt1", "n_contexts")}
                    for k, v in d["conditions"].items()
                    if isinstance(v, dict)
                }
            if isinstance(d.get("steer_base_band"), dict):
                d["steer_base_band"] = {
                    k: d["steer_base_band"].get(k) for k in ("pooled_p50", "pooled_p90", "k_draws")
                }
        if d is None:
            dig["skipped"].append(str(path.relative_to(eval_root)))
        else:
            dig[name] = d
    for trait in ("sycophancy", "hallucination", "evil"):
        p = eval_root / "steering" / "judge" / f"scores_{trait}.json"
        d = _digest_json(p, ["n_items", "n_content_dropped_draws", "n_transport_lost_draws"])
        if d is None:
            dig["skipped"].append(f"steering/judge/scores_{trait}.json")
        else:
            dig[f"judge_{trait}"] = d
    return dig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default="14,18,19")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--no-hf-stage", action="store_true")
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",") if x]
    eval_root = c.eval_out(args.out_root)
    agg_dir = eval_root / "aggregate"

    summaries = _stage_if_missing(
        c.data_out(args.out_root) / "draws/summaries",
        f"{c.HF_UPLOAD_PREFIX}/draws/summaries",
        not args.no_hf_stage,
    )
    op_dir = c.data_out(args.out_root) / "operators"
    if not op_dir.exists() and not args.no_hf_stage:
        try:
            _stage_if_missing(op_dir, f"{c.HF_UPLOAD_PREFIX}/operators", True)
        except Exception as e:  # noqa: BLE001 — operators optional for floors
            print(f"[p6] note: operators unavailable ({e}); top-singular floors skipped")

    ceiling: dict = {
        "meta": c.repro_meta({"script": "scripts/issue1774_aggregate.py"}),
        "convention": "ICC = between/(between+within); floors on the SAME draw "
        "subset as the P1 generation (temp 1.0, K draws); full-corpus "
        "between-context variance alongside (floor-subset mismatch guard); a "
        f"per-trait cell with between < {NOISE_LIMITED_FACTOR}x floor is labeled "
        "noise-limited (plan H3)",
        "layers": {},
    }
    for layer in layers:
        if not any(summaries.glob(f"t1_L{layer}_draw0_*part*.npy")):
            print(f"[p6] L{layer}: no draw summaries — skipped")
            ceiling["layers"][f"L{layer}"] = {"skipped": "no draw summaries"}
            continue
        print(f"[phase=p6_noise_ceiling] L{layer}", flush=True)
        stack, _rows = load_draw_stack(summaries, layer)
        dirs = direction_bank(layer, op_dir)
        floors = noise_floors(stack, dirs)
        floors["per_trait_gating"] = gate_per_trait_cells(eval_root, floors, layer)
        ceiling["layers"][f"L{layer}"] = floors
        # checkpoint per layer — persist the moment each layer completes
        c.write_json_atomic(eval_root / "noise_ceiling.json", ceiling)
        print(f"[p6] unit L{layer} done: {len(floors['directions'])} directions", flush=True)

    print("[phase=p6_aggregate]", flush=True)
    dig = merge_phase_digests(eval_root, layers)
    dig["meta"] = c.repro_meta({"script": "scripts/issue1774_aggregate.py"})
    dig["noise_ceiling_path"] = "eval_results/issue_1774/noise_ceiling.json"
    c.write_json_atomic(agg_dir / "aggregate_summary.json", dig)
    print(f"[p6] done: aggregate_summary skipped={len(dig['skipped'])} inputs missing")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
