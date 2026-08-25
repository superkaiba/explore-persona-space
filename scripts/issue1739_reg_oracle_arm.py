#!/usr/bin/env python3
"""Fair-protocol regression-on-REAL-answer arm ("reg_oracle") for the applications figure.

User-chat inline free-analysis round on #1739 (2026-08-25): the ICLR
applications figure needs the ridge-regression upper bound fitted on the REAL
answer activation summaries under the SAME fair matched-data protocol as
arm4_ridge_ctx (regression on the context vector) and arm7_map_ridge_pred
(regression on the mapped answer) — identical train pool, judged labels,
frozen read-out layers, CI method, and eval cells; only the input
representation changes.

That fit ALREADY EXISTS, committed: the 2026-08-07 fair-v2 re-score
(scripts/issue1739_result2fair_score.py at commit
b1a8f6687ffc188e71969dbf5e447d0e47a011e1) added arm12_oracle_reg — "ridge
from the whitened TRUE-answer acts za to the DV" — to the fair roster and
scored it on every cell; the per-cell frozen-layer Spearman rho + paired
bootstrap CI live in
``eval_results/issue_1739/result2_fair_v2/<behavior>/all_arms_spearman.json``
(``transfer_rows``). Re-running the fit would require the reaped multi-GB
capture stores (``data/issue_1739/hf_dl/{<b>_labeling,<b>_extraction,
wcrung_capture_store,...}``), so this driver is a PURE RE-READ of committed
artifacts: it extracts the arm12 rows into the committed
``result2_fair_points.json`` row schema under method "reg_oracle", behind two
fail-loud gates:

1. PROTOCOL-PARITY GATE — every (arm, setting) the fair-v2 pass shares with
   the committed ``result2_fair_points.json`` (arms 1/4/6/7/11, linear map)
   must reproduce the committed rho and frozen layer EXACTLY; a mismatch
   means the v2 pass is not the same protocol and extraction would be
   invalid. (Measured at authoring time: max |rho diff| = 0.0 over all 70
   shared linear cells.)
2. ESTIMATOR-VALIDITY GATE — the realized readout train count (eliciting
   train budget cell UNION judged WildChat train split, read off the arm12
   rows themselves) must exceed the feature dim d for every behavior
   (well-posed primal ridge, n_train > d), and must equal arm4's counts row
   for row (the identical-regime statement).

Never writes to ``result2_fair_points.json`` or any other committed artifact
— the output is the NEW file
``eval_results/issue_1739/result2_fair/reg_oracle_points.json``.

Selected-lambda note: the fair pass selects lambda PER TARGET by GCV over the
pinned ``RIDGE_LAMBDAS`` grid inside ``fits.ridge_gcv_predict_per_target``
(standardize-X train stats, center-Y, primal d x d Gram when n_tr > d) and
does NOT persist the realized selection — no ``ridge_lambda_diagnostics``
exists in either committed result2_fair tree — so this driver records the
grid + selector in meta rather than fabricating per-cell values.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    """Resolve the repo root and put it + scripts/ on sys.path (sibling imports)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_result2fair_score.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    for p in (str(root), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


ROOT = _repo_root()

METHOD = "reg_oracle"
ARM = "arm12_oracle_reg"
# Shared arms between the committed v1 points file and the fair-v2 summaries —
# the parity gate's roster (linear map only; v1's MLP cells have no v2 twin).
PARITY_ARMS = (
    "arm1_ctx_e1",
    "arm4_ridge_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm11_oracle_proj",
)
# Answer-summary feature dim (Qwen-2.5-7B hidden size). Not re-derivable from
# the summaries read here — recorded with its source in meta; the validity
# verdict (n_train > d) has >2x margin at every behavior, so plausible dim
# error cannot flip it.
FEATURE_DIM = 3584
PARITY_TOL = 1e-12

V1_POINTS = ROOT / "eval_results/issue_1739/result2_fair/result2_fair_points.json"
V2_DIR = ROOT / "eval_results/issue_1739/result2_fair_v2"
OUT_PATH = ROOT / "eval_results/issue_1739/result2_fair/reg_oracle_points.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _linear_transfer_index(doc: dict, behavior: str) -> dict[tuple[str, str], dict]:
    """Index linear transfer rows by (arm, eval_rung), refusing duplicates.

    Mirrors the duplicate refusal in issue1739_result2fair_fig.collect().
    """
    out: dict[tuple[str, str], dict] = {}
    for r in doc.get("transfer_rows") or []:
        if r.get("map_kind", "linear") != "linear":
            continue
        key = (r["arm"], r["eval_rung"])
        if key in out:
            raise SystemExit(f"duplicate fair-v2 linear row {behavior}/{key}")
        out[key] = r
    return out


def main() -> int:
    # Row-shape conventions imported from the rig's own fig module (never
    # copied — the figures and this artifact must agree by construction).
    from issue1739_result2fair_fig import FABRICATION_SETTINGS, METHOD_OF, SETTINGS

    from explore_persona_space.experiments.issue_1739.constants import (
        N_BOOT,
        RIDGE_LAMBDAS,
    )
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    assert METHOD_OF[(ARM, "linear")] == "regression_real_answer", (
        "fig-module slot for arm12 moved — re-check conventions before extracting"
    )

    v1_doc = json.loads(V1_POINTS.read_text())
    v1_idx: dict[tuple[str, str, str], dict] = {}
    for r in v1_doc["points"]:
        if r.get("map_kind") == "linear":
            v1_idx[(r["behavior"], r["setting"], r["arm_id"])] = r

    points: list[dict] = []
    parity_checked = 0
    n_train_by_behavior: dict[str, dict] = {}
    frozen_layers: dict[str, int] = {}
    inputs: dict[str, str] = {
        str(V1_POINTS.relative_to(ROOT)): _sha256(V1_POINTS),
    }
    upstream: dict[str, dict] = {}

    for behavior, settings in SETTINGS.items():
        v2_path = V2_DIR / behavior / "all_arms_spearman.json"
        doc = json.loads(v2_path.read_text())
        inputs[str(v2_path.relative_to(ROOT))] = _sha256(v2_path)
        meta = doc["meta"]
        if ARM not in meta["arms"]:
            raise SystemExit(f"{behavior}: {ARM} absent from fair-v2 roster {meta['arms']}")
        idx = _linear_transfer_index(doc, behavior)

        # --- gate 1: protocol parity against the committed points file ---
        for arm in PARITY_ARMS:
            for setting in settings:
                v1_row = v1_idx.get((behavior, setting, arm))
                v2_row = idx.get((arm, setting))
                if v1_row is None or v2_row is None:
                    raise SystemExit(
                        f"parity gate: missing row {behavior}/{setting}/{arm} "
                        f"(v1={v1_row is not None}, v2={v2_row is not None})"
                    )
                d_rho = abs(float(v1_row["rho"]) - float(v2_row["rho_frozen"]))
                if d_rho > PARITY_TOL or int(v1_row["layer"]) != int(v2_row["layer"]):
                    raise SystemExit(
                        f"parity gate FAILED at {behavior}/{setting}/{arm}: "
                        f"|drho|={d_rho:.3e}, layers {v1_row['layer']} vs {v2_row['layer']} "
                        "— fair-v2 is not the committed protocol; refusing to extract arm12"
                    )
                parity_checked += 1

        # --- gate 2: estimator validity (n_train > d, same regime as arm4) ---
        a12_rows = [idx[(ARM, s)] for s in settings if (ARM, s) in idx]
        missing = [s for s in settings if (ARM, s) not in idx]
        if missing:
            raise SystemExit(f"{behavior}: no fair-v2 {ARM} row for settings {missing}")
        elic = {int(r["n_readout_eliciting"]) for r in a12_rows}
        wctr = {int(r["n_readout_wc_train"]) for r in a12_rows}
        a4_rows = [idx[("arm4_ridge_ctx", s)] for s in settings]
        elic4 = {int(r["n_readout_eliciting"]) for r in a4_rows}
        wctr4 = {int(r["n_readout_wc_train"]) for r in a4_rows}
        if len(elic) != 1 or len(wctr) != 1 or elic != elic4 or wctr != wctr4:
            raise SystemExit(
                f"{behavior}: readout train counts differ across rows/arms — "
                f"arm12 elic={elic} wc={wctr} vs arm4 elic={elic4} wc={wctr4}"
            )
        n_train = elic.pop() + wctr.pop()
        if n_train <= FEATURE_DIM:
            raise SystemExit(
                f"{behavior}: n_train={n_train} <= d={FEATURE_DIM} — under-determined "
                "ridge regime; refusing (estimator-validity gate)"
            )
        n_train_by_behavior[behavior] = {
            "n_train": n_train,
            "n_eliciting": int(a12_rows[0]["n_readout_eliciting"]),
            "n_wc_train": int(a12_rows[0]["n_readout_wc_train"]),
            "d": FEATURE_DIM,
            "regime": "n_train > d (primal ridge); identical counts asserted vs arm4_ridge_ctx",
        }

        layers = {int(idx[(ARM, s)]["layer"]) for s in settings}
        if len(layers) != 1:
            raise SystemExit(f"{behavior}: {ARM} frozen layer not unique across cells: {layers}")
        frozen_layers[behavior] = layers.pop()
        upstream[behavior] = {
            "fit_git_commit": meta.get("git_commit"),
            "fit_ts": doc.get("ts"),
            "fit_script": "scripts/issue1739_result2fair_score.py",
            "fit_input_paths": meta.get("input_paths"),
            "fit_input_sha256": meta.get("input_sha256"),
            "frozen_layer_source": (meta.get("frozen_layer_sources") or {})
            .get("linear", {})
            .get(ARM),
            "n_train_contexts_meta": meta.get("n_train_contexts"),
        }

        # --- extraction: one points row per cell, v1 schema ---
        for setting in settings:
            r = idx[(ARM, setting)]
            points.append(
                {
                    "behavior": behavior,
                    "setting": setting,
                    "method": METHOD,
                    "arm_id": ARM,
                    "map_kind": "linear",
                    "rho": float(r["rho_frozen"]),
                    "ci": list(r.get("ci_frozen") or []) or None,
                    "n_replicates": 1,
                    "n_eval": int(r["n_eval"]),
                    "layer": int(r["layer"]),
                    "map_condition": "none (real-answer input; fair protocol)",
                    "readout": "fair union readout",
                    "dv_construct": (
                        "fabricated_fraction_rescaled_x100"
                        if (behavior, setting) in FABRICATION_SETTINGS
                        else "trait_rubric_graded_0_100"
                    ),
                    "source_file": str(v2_path.relative_to(ROOT)),
                }
            )

    if len(points) != 14:
        raise SystemExit(f"expected 14 cells, built {len(points)}")

    meta_block = {
        "method": METHOD,
        "arm_id": ARM,
        "method_note": (
            "reg_oracle = ridge regression fitted on the REAL answer activation summaries "
            "(the vectors arm11_oracle_proj projects the persona vector onto) under the fair "
            "matched-data protocol — the real-answer upper bound for the regression family "
            "(arm4_ridge_ctx = context input, arm7_map_ridge_pred = mapped-answer input). The "
            "rig's own fig-module slot name for this arm is 'regression_real_answer'; this "
            "artifact uses the requested key 'reg_oracle'."
        ),
        "extraction_note": (
            "pure re-read of the committed fair-v2 summaries (no refit): the arm12_oracle_reg "
            "fit was produced by scripts/issue1739_result2fair_score.py on 2026-08-07 under "
            "the identical fair protocol; parity gate re-verified that every shared linear "
            "cell (arms 1/4/6/7/11 x all settings) reproduces the committed "
            "result2_fair_points.json rho exactly"
        ),
        "parity_gate": {
            "cells_checked": parity_checked,
            "tolerance": PARITY_TOL,
            "arms": list(PARITY_ARMS),
        },
        "n_train_by_behavior": n_train_by_behavior,
        "frozen_layers": frozen_layers,
        "ridge_selector": {
            "fn": "explore_persona_space.experiments.issue_1739.fits.ridge_gcv_predict_per_target",
            "selection": "per-target GCV over the pinned lambda grid",
            "lambda_grid": list(RIDGE_LAMBDAS),
            "selected_lambda": (
                "not persisted by the fair pass (no ridge_lambda_diagnostics in either "
                "committed result2_fair tree)"
            ),
            "gram": "primal d x d when n_train > d (holds for every behavior here)",
        },
        "ci_method": (
            f"per-cell paired bootstrap of Spearman rho at the frozen layer, {N_BOOT} draws, "
            "2.5/97.5 percent quantiles (experiments.issue_1739.arms bootstrap machinery — "
            "the same CI every committed result2_fair arm carries)"
        ),
        "feature_dim_source": (
            "Qwen-2.5-7B hidden size (3584); not recorded in the summaries read here"
        ),
        "input_sha256": inputs,
        "upstream_fit_provenance": upstream,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_block.update(as_metadata_dict(git_provenance(), phase="reg-oracle-extract"))

    out = {"n_points": len(points), "points": points, "meta": meta_block}
    OUT_PATH.write_text(json.dumps(out, indent=1) + "\n")
    print(
        f"[reg_oracle] wrote {OUT_PATH.relative_to(ROOT)} ({len(points)} cells; "
        f"parity gate: {parity_checked} shared cells exact)"
    )
    for p in points:
        ci = p["ci"] or [float("nan"), float("nan")]
        print(
            f"  {p['behavior']:13s} {p['setting']:14s} rho={p['rho']:+.4f} "
            f"ci=[{ci[0]:+.4f},{ci[1]:+.4f}] layer={p['layer']} n_eval={p['n_eval']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
