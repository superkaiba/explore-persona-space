#!/usr/bin/env python3
"""Issue #2588 inline round: re-score the Section 4.8 capability panel on a 3-rollout target.

For the ten paper-appendix no-thinking cells, form the TEST target as the mean
of the three banked answer-vector draws (seed-42 ``test_1000`` + seed-43/44
``ceiling_s4x`` captures, same prompts / decoding, ci-aligned) at each cell's
frozen ``layer_star``, and re-score the EXISTING frozen ridge map against it.
No refit, no layer re-selection, no lambda re-selection: the frozen map's test
predictions come verbatim from the ``reconstruct_map`` payload cache
(``scripts/issue2588_mapping_rank_vs_capability.py``, parity-gated to the
banked ``test_r2`` at build time), and only the scoring target changes.

CORRECTNESS GATE (runs first, hard-stops on failure): scoring the cached
predictions against the seed-42 draw alone must reproduce every cell's banked
``test_r2`` to within 0.005 and the panel-level Spearman rho = +0.867.

Also reports the free second read: a one-against-two ceiling (each draw
predicting the mean of the other two, averaged over the three rotations),
against the banked two-draw ceilings as a consistency check.

Reuse spine (never re-derived): shard loader + estimator ports from
``scripts/issue2588_ceiling_normalized.py`` (CN: ``_load_stage_layer_y``,
``_two_draw_ceiling``, ``_perrow_hits_cos``); ``pooled_r2`` +
``exact_spearman_permutation`` from
``scripts/issue2588_mapping_rank_vs_capability.py`` (MR).

Retrieval calibration: the panel's ``test_retrieval_acc1_cos_calibrated``
subtracts the shuffled-TRAIN-pairing refit null's mean acc@1 (200 draws,
banked in ``nulls_prompt_last.json``; measured 0.00100-0.00104 across cells,
i.e. chance = 1/pool). Rebuilding that refit null against the 3-rollout
target would require refitting 200 null maps per cell (out of scope: no
refit), so the banked null mean is reused, scaled by pool size
(``banked_null_mean * test_n / n_aligned``) — a < 2e-5 absolute correction.

Output: ``eval_results/issue_2588/ceiling_normalized/three_rollout_target.json``.

Usage::

    uv run python scripts/issue2588_three_rollout_target.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
for _p in (str(_SCRIPTS), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # HF token + thread caps before numpy import

import numpy as np  # noqa: E402

import issue2588_ceiling_normalized as CN  # noqa: E402  (module-top load_dotenv; loaders + ports)
from issue2588_mapping_rank_vs_capability import (  # noqa: E402
    DEFAULT_CACHE,
    exact_spearman_permutation,
    pooled_r2,
)

OUT_DIR = _REPO_ROOT / "eval_results" / "issue_2588" / "ceiling_normalized"
PRIOR_ROUND_JSON = OUT_DIR / "ceiling_normalized_capability.json"
GATE_R2_TOL = 0.005
GATE_RHO_EXPECTED = 0.8666666666666665  # banked raw_r2_vs_aa (n=10, no AA ties)
GATE_RHO_TOL = 1e-9
SEEDS = (42, 43, 44)


def _sorted_by_ci(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Sort a _load_stage_layer_y record by ci suffix (== load_split's full-id
    string sort: every test row_id shares the constant 'test_1000_' prefix)."""
    order = np.argsort(d["ci"])
    return {"ci": d["ci"][order], "y": d["y"][order]}


def _load_frozen_payload(cell_key: str, star: int, dim: int, banked_test_r2: float) -> dict:
    """Frozen-map payload from the reconstruct_map cache — predictions only,
    never a refit. Fail loud when absent (rebuilding needs the 10k-row train
    download; that is scripts/issue2588_mapping_rank_vs_capability.py's job)."""
    path = DEFAULT_CACHE / f"{cell_key}__prompt_last.npz"
    assert path.exists(), (
        f"{cell_key}: frozen-map payload cache missing at {path}; rebuild it via "
        "scripts/issue2588_mapping_rank_vs_capability.py (reconstruct_map) first"
    )
    with np.load(path, allow_pickle=False) as z:
        rec = {k: z[k] for k in ("pred_test", "target_test")}
        rec["layer"] = int(z["layer"])
        rec["dimension"] = int(z["dimension"])
        rec["expected_test_r2"] = float(z["expected_test_r2"])
        rec["reconstructed_test_r2"] = float(z["reconstructed_test_r2"])
    assert rec["layer"] == star, (cell_key, rec["layer"], star)
    assert rec["dimension"] == dim, (cell_key, rec["dimension"], dim)
    assert np.isclose(rec["expected_test_r2"], banked_test_r2, atol=1e-9), (
        f"{cell_key}: cache built against a different fits record "
        f"({rec['expected_test_r2']} vs banked {banked_test_r2})"
    )
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR / "three_rollout_target.json")
    args = parser.parse_args(argv)

    panel = json.loads(CN.PANEL_JSON.read_text())
    rows = {m["cell"]: m for m in panel["maps"]}
    missing = [c for c in CN.TARGET_CELLS if c not in rows]
    assert not missing, f"target cells missing from panel JSON: {missing}"
    revisions = {rows[c]["hf_revision"] for c in CN.TARGET_CELLS}
    assert len(revisions) == 1, f"in-scope cells span multiple revisions: {revisions}"
    revision = revisions.pop()
    prior = json.loads(PRIOR_ROUND_JSON.read_text())
    prior_rows = {m["cell"]: m for m in prior["per_model"]}

    per_model: list[dict] = []
    gate_fail: list[str] = []
    for cell_key in CN.TARGET_CELLS:
        row = rows[cell_key]
        assert row["arm"] == "no-thinking", (cell_key, row["arm"])
        star = int(row["layer_star"])
        dim = int(row["dimension"])
        perf = row["mapping_performance"]
        banked_r2 = float(perf["test_r2"])
        banked_null_mean = float(perf["test_retrieval_acc1_cos_null_mean"])
        test_n = int(perf["test_n"])
        prefix = f"{CN.PANEL_PREFIX}/{cell_key.removesuffix('_a')}/nothink"

        payload = _load_frozen_payload(cell_key, star, dim, banked_r2)
        draws = {
            42: _sorted_by_ci(CN._load_stage_layer_y(prefix, "test_1000", star, revision)),
            43: _sorted_by_ci(CN._load_stage_layer_y(prefix, "ceiling_s43", star, revision)),
            44: _sorted_by_ci(CN._load_stage_layer_y(prefix, "ceiling_s44", star, revision)),
        }
        # Row-alignment anchor: the cache rows are sorted by full row_id, whose
        # order equals the ci sort; the cached target must BE the seed-42 draw.
        y42 = draws[42]["y"]
        assert payload["pred_test"].shape == y42.shape == (test_n, dim), (
            cell_key,
            payload["pred_test"].shape,
            y42.shape,
            (test_n, dim),
        )
        assert np.allclose(payload["target_test"], y42.astype(np.float32), atol=1e-6), (
            f"{cell_key}: cached target_test != ci-sorted test_1000 y_ans — row alignment broken"
        )
        pred = payload["pred_test"].astype(np.float64)

        # CORRECTNESS GATE, per cell: frozen predictions vs the seed-42 draw
        # alone must reproduce the banked test_r2.
        r2_gate = pooled_r2(pred, y42)
        if abs(r2_gate - banked_r2) > GATE_R2_TOL:
            gate_fail.append(f"{cell_key}: re-scored {r2_gate:.6f} vs banked {banked_r2:.6f}")

        # 3-rollout target: mean of the three ci-aligned draws.
        ci3 = np.intersect1d(np.intersect1d(draws[42]["ci"], draws[43]["ci"]), draws[44]["ci"])
        n3 = int(ci3.size)
        idx = {s: np.searchsorted(draws[s]["ci"], ci3) for s in SEEDS}
        ys = {s: draws[s]["y"][idx[s]] for s in SEEDS}
        for s in SEEDS:  # searchsorted on a sorted unique axis: verify the hit
            assert np.array_equal(draws[s]["ci"][idx[s]], ci3), (cell_key, s)
        target3 = (ys[42] + ys[43] + ys[44]) / 3.0
        pred_sub = pred[idx[42]]

        r2_single_sub = pooled_r2(pred_sub, ys[42])
        r2_3roll = pooled_r2(pred_sub, target3)
        null3 = banked_null_mean * test_n / n3  # chance scales as 1/pool
        acc1_single_sub = float(np.mean(CN._perrow_hits_cos(pred_sub, ys[42])["hit1"]))
        acc1_3roll = float(np.mean(CN._perrow_hits_cos(pred_sub, target3)["hit1"]))

        # Free second read: one draw predicting the mean of the other two.
        one_v_two_r, one_v_two_acc1 = [], []
        for s in SEEDS:
            others = [t for t in SEEDS if t != s]
            mean_others = (ys[others[0]] + ys[others[1]]) / 2.0
            one_v_two_r.append(CN._two_draw_ceiling(ys[s], mean_others)["ceiling"])
            one_v_two_acc1.append(float(np.mean(CN._perrow_hits_cos(ys[s], mean_others)["hit1"])))

        prior_cell = prior_rows[cell_key]
        rec = {
            "cell": cell_key,
            "model": row["model"],
            "aa_index": row["aa_index"],
            "dimension": dim,
            "layer_star": star,
            "n_aligned_3way": n3,
            "test_n": test_n,
            "gate": {"banked_test_r2": banked_r2, "rescored_test_r2_seed42": r2_gate},
            "r2_single_draw_subset": r2_single_sub,
            "r2_three_rollout": r2_3roll,
            "r2_delta": r2_3roll - r2_single_sub,
            "acc1_single_draw_subset_raw": acc1_single_sub,
            "acc1_three_rollout_raw": acc1_3roll,
            "null_mean_scaled": null3,
            "acc1_single_draw_subset_calibrated": acc1_single_sub - null3,
            "acc1_three_rollout_calibrated": acc1_3roll - null3,
            "banked_acc1_calibrated": float(perf["test_retrieval_acc1_cos_calibrated"]),
            "one_vs_two_ceiling": {
                "per_rotation_r": one_v_two_r,
                "mean_r": float(np.mean(one_v_two_r)),
                "per_rotation_acc1": one_v_two_acc1,
                "mean_acc1": float(np.mean(one_v_two_acc1)),
                "rotation_holdout_seeds": list(SEEDS),
            },
            "banked_two_draw_ceiling": {
                "s42x43": prior_cell["ceiling_s42x43"]["ceiling"],
                "s43x44": prior_cell["ceiling_s43x44_banked"]["two_draw"]["ceiling"],
                "s42x43_retrieval_acc1": prior_cell["ceiling_s42x43"]["retrieval"][
                    "ceiling_acc1_cos"
                ],
                "s43x44_retrieval_acc1": prior_cell["ceiling_s43x44_banked"]["retrieval"][
                    "ceiling_acc1_cos"
                ],
            },
        }
        per_model.append(rec)
        print(
            f"[i2588-3roll] {cell_key}: n3={n3} gate|Δ|={abs(r2_gate - banked_r2):.2e} "
            f"r2 {r2_single_sub:.4f}->{r2_3roll:.4f} (Δ{r2_3roll - r2_single_sub:+.4f}) "
            f"acc1 {acc1_single_sub:.4f}->{acc1_3roll:.4f} "
            f"1v2ceil={np.mean(one_v_two_r):.4f}"
        )

    # CORRECTNESS GATE, panel level.
    aa = [float(m["aa_index"]) for m in per_model]
    gate_rhos = [m["gate"]["rescored_test_r2_seed42"] for m in per_model]
    rho_gate = exact_spearman_permutation(gate_rhos, aa)
    if abs(rho_gate["rho"] - GATE_RHO_EXPECTED) > GATE_RHO_TOL:
        gate_fail.append(
            f"panel Spearman: re-scored {rho_gate['rho']:+.6f} vs banked {GATE_RHO_EXPECTED:+.6f}"
        )
    if gate_fail:
        for line in gate_fail:
            print(f"[i2588-3roll] GATE FAIL: {line}")
        raise SystemExit(
            "correctness gate FAILED — scoring path differs from the panel's; "
            "no 3-rollout numbers written"
        )
    print(f"[i2588-3roll] GATE PASS: all cells within {GATE_R2_TOL}; rho={rho_gate['rho']:+.4f}")

    def col(key: str) -> list[float]:
        return [float(m[key]) for m in per_model]

    trends = {
        "gate_single_draw_r2_vs_aa": rho_gate,
        "single_draw_subset_r2_vs_aa": exact_spearman_permutation(col("r2_single_draw_subset"), aa),
        "three_rollout_r2_vs_aa": exact_spearman_permutation(col("r2_three_rollout"), aa),
        "single_draw_subset_acc1_cal_vs_aa": exact_spearman_permutation(
            col("acc1_single_draw_subset_calibrated"), aa
        ),
        "three_rollout_acc1_cal_vs_aa": exact_spearman_permutation(
            col("acc1_three_rollout_calibrated"), aa
        ),
        "one_vs_two_ceiling_vs_aa": exact_spearman_permutation(
            [m["one_vs_two_ceiling"]["mean_r"] for m in per_model], aa
        ),
        "banked_reference": {
            "raw_r2_vs_aa_rho": prior["trends"]["raw_r2_vs_aa"]["rho"],
            "normalized_r2_s42x43_vs_aa_rho": prior["trends"]["normalized_r2_s42x43_vs_aa"]["rho"],
        },
    }

    out = {
        "schema_version": 1,
        "meta": {
            "task": 2588,
            "script": "scripts/issue2588_three_rollout_target.py",
            "git_commit": CN._git_sha(),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "hf_repo": CN.HF_DATA_REPO,
            "hf_revision": revision,
            "panel_json": str(CN.PANEL_JSON.relative_to(_REPO_ROOT)),
            "prior_round_json": str(PRIOR_ROUND_JSON.relative_to(_REPO_ROOT)),
            "frozen_map_source": (
                "reconstruct_map payload cache (parity-gated at build to banked test_r2 "
                "within 3e-4); predictions re-used verbatim, no refit"
            ),
            "target": "mean of banked seed-42/43/44 y_ans draws at layer_star, ci-aligned",
            "retrieval_calibration": (
                "banked shuffled-train-pairing refit-null mean acc@1 (200 draws) scaled "
                "by pool size: null_mean * test_n / n_aligned_3way"
            ),
            "gate": {
                "per_cell_r2_tol": GATE_R2_TOL,
                "panel_rho_expected": GATE_RHO_EXPECTED,
                "result": "PASS",
            },
        },
        "per_model": per_model,
        "trends": trends,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[i2588-3roll] wrote {args.out}")
    for k, v in trends.items():
        if "rho" in v:
            print(
                f"[i2588-3roll] {k}: rho={v['rho']:+.4f} "
                f"p={v['two_sided_exact_permutation_p']:.4f} ({v['method']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
