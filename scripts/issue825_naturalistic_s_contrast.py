#!/usr/bin/env python3
"""Issue #825 ``naturalistic-single-turn`` — paired chat-vs-naturalistic Track-S contrast.

Refits the per-example linear context->answer-profile map h: c_x -> v(x) (held-out
K-fold Gram-space ridge, #779/#825 recipe) on the SAME 5,000 LMSYS single-turn
conversations under TWO render formats — the Qwen chat template (S1/S2, stem
``{model}_chat_s``) and the naturalistic ``User:``/``Assistant:`` transcript
(S1N/S2N, stem ``{model}_naturalistic_s``) — and computes, per model:

  (i)   per-layer held-out R^2 per format (the full 28-layer curve);
  (ii)  the PAIRED naturalistic-chat R^2 delta at the frozen layers (19 + 14/18/26)
        with a 1,000-draw paired conversation-level bootstrap (shared resample
        indices across formats, per-draw own-mean re-centered, fp64);
  (iii) the pretrained/instruct strength ratio under each format.

Single manipulated variable: the Track-S render format. Pure re-fit over persisted
turnstore tensors — no generation, no training. The ridge core + bundle loader are
IMPORTED from ``issue825_fit_cells`` and the batched paired-bootstrap machinery from
``issue825_role_contrast`` (round 5, ``role-map-comparison``) — never re-implemented
(the role runner's batched bootstrap is already equivalence-gated against its serial
oracle; this script re-runs that gate as smoke evidence).

Both formats fit on the SHARED conversation-id set (intersection, expected identical
5,000) reindexed to ONE order, so ``heldout_r2_sweep`` — deterministic on
(conv_ids, n_folds, seed) — assigns IDENTICAL folds to both formats (asserted). The
paired bootstrap reuses ONE resample-index matrix across both formats AND all frozen
layers within a model, so the (naturalistic, chat) pair is drawn together.

Output: ONE JSON at ``eval_results/issue_825/naturalistic-single-turn/format_contrast.json``.

Usage (pod-side, tensors local; the ridge core routes to CUDA when available):
    uv run python scripts/issue825_naturalistic_s_contrast.py \
        --turnstore-dir data/issue_825/turnstore \
        --out-dir eval_results/issue_825/naturalistic-single-turn
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit_cells  # noqa: E402
import issue825_role_contrast as role_contrast  # noqa: E402

# The two Track-S cells per model: chat anchor + naturalistic refit. cell_id starting
# with "S" routes track="s" in fit_cells._normalize_cell; the explicit format is honored.
FORMATS = ("chat", "naturalistic")
MODELS = ("instruct", "pretrained")
# Layer read-outs: report every layer's R^2 (curve, (i)); the paired delta + ratio
# read the pre-registered frozen layers, headline 19.
FROZEN_LAYERS = tuple(fit_cells.FROZEN_LAYERS)  # (14, 18, 19, 26)
HEADLINE_LAYER = 19
EQUIV_TOL = 1e-8  # batched-vs-serial paired-bootstrap oracle tolerance (role_contrast)


def _cell_for(model: str, fmt: str) -> dict:
    """Track-S cell dict (slot 0 -> a1 profile) for (model, format)."""
    cid = {
        "chat": {"instruct": "S1", "pretrained": "S2"},
        "naturalistic": {"instruct": "S1N", "pretrained": "S2N"},
    }[fmt][model]
    return fit_cells._normalize_cell({"cell_id": cid, "model": model, "format": fmt})


def _load_xy(turnstore_dir: Path, model: str, fmt: str) -> dict:
    """Assemble (X, Y, conv_ids) for one (model, format) Track-S cell.

    Uses the fit_cells track-aware loader (.npz OR .pt shards) + _cell_xy exactly as
    run_cell does, so the X/Y/keep semantics are byte-identical to the committed
    S1/S2 anchors.
    """
    cell = _cell_for(model, fmt)
    bundle = fit_cells._load_bundle_any(
        turnstore_dir, cell["model_key"], cell["format_key"], cell["track"]
    )
    xy = fit_cells._cell_xy(bundle, cell)
    return {"X": xy["X"], "Y": xy["Y"], "conv_ids": np.asarray([str(c) for c in xy["conv_ids"]])}


def _reindex_to_shared(xy: dict, shared_ids: np.ndarray) -> dict:
    """Reorder xy rows to the shared conv-id ORDER (a permutation/subset).

    conv_ids are unique within a Track-S bundle (conv == row), so a lookup map is
    unambiguous; a shared id missing from this format's bundle fails loud.
    """
    pos = {cid: i for i, cid in enumerate(xy["conv_ids"])}
    assert len(pos) == len(xy["conv_ids"]), "Track-S bundle has duplicate conv_ids (conv != row)"
    idx = np.array([pos[c] for c in shared_ids], dtype=np.int64)
    return {"X": xy["X"][idx], "Y": xy["Y"][idx], "conv_ids": np.asarray(shared_ids)}


def _paired_delta(
    sweep_nat: dict,
    sweep_chat: dict,
    y_nat: np.ndarray,
    y_chat: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    equivalence_gate: bool,
) -> tuple[dict, dict]:
    """Paired naturalistic-chat R^2 delta at the frozen layers.

    Shared resample-index matrix across BOTH formats and ALL frozen layers (the
    pairing). Returns (delta_by_layer, equivalence_by_layer). Follows role_contrast's
    delta convention exactly: delta_obs from heldout r2_obs (the label-driving,
    per-fold-own-mean statistic the cells_*.json use); delta_pooled_global_obs +
    bootstrap CI in the single-global-own-mean pooled family the bootstrap draws.
    """
    # Rows that received held-out predictions under BOTH formats (identical when the
    # shared folds are identical; the intersection is the safe form at tiny smoke n).
    fitted = sweep_nat["fitted_mask"] & sweep_chat["fitted_mask"]
    n = int(fitted.sum())
    idx_matrix = role_contrast.draw_index_matrix(n, n_boot, seed + 7)
    w = role_contrast.counts_from_indices(idx_matrix, n)
    delta: dict[str, dict] = {}
    equivalence: dict[str, dict] = {}
    for li in FROZEN_LAYERS:
        p_nat = sweep_nat["preds_frozen"][li][fitted]
        p_chat = sweep_chat["preds_frozen"][li][fitted]
        yn = y_nat[fitted, li, :]
        yc = y_chat[fitted, li, :]
        boot = role_contrast.paired_bootstrap_batched(p_nat, yn, p_chat, yc, w)
        if equivalence_gate:
            oracle = role_contrast.paired_bootstrap_serial_reference(
                p_nat, yn, p_chat, yc, idx_matrix
            )
            diffs = {
                k: float(np.nanmax(np.abs(boot[k] - oracle[k])))
                for k in ("assistant", "user", "delta")
            }
            ok = bool(all(v < EQUIV_TOL for v in diffs.values()))
            equivalence[str(li)] = {"max_abs_diff": diffs, "tol": EQUIV_TOL, "pass": ok}
            assert ok, f"L{li}: batched-vs-serial paired-bootstrap equivalence FAILED: {diffs}"
        r2_obs_nat = float(sweep_nat["r2_obs"][li])
        r2_obs_chat = float(sweep_chat["r2_obs"][li])
        pooled_nat = float(fit_cells._pooled_r2(p_nat, yn))
        pooled_chat = float(fit_cells._pooled_r2(p_chat, yc))
        delta[str(li)] = {
            # LABEL-DRIVING point estimate: full-sample held-out r2_obs delta
            # (naturalistic - chat); the bootstrap CI is for the pooled-global family.
            "delta_obs": r2_obs_nat - r2_obs_chat,
            "r2_obs_naturalistic": r2_obs_nat,
            "r2_obs_chat": r2_obs_chat,
            "delta_pooled_global_obs": pooled_nat - pooled_chat,
            **role_contrast._ci(boot["delta"]),
        }
    return delta, {
        "n_paired": n,
        "n_boot": int(n_boot),
        "boot_seed": seed + 7,
        "per_layer": equivalence,
    }


def run_model(
    turnstore_dir: Path,
    model: str,
    *,
    n_folds: int,
    seed: int,
    n_boot: int,
    equivalence_gate: bool,
    expect_n: int | None,
) -> dict:
    """Full chat-vs-naturalistic contrast for one model."""
    xy = {fmt: _load_xy(turnstore_dir, model, fmt) for fmt in FORMATS}
    # Shared conv-id set (intersection), SORTED for a deterministic shared order.
    ids_nat, ids_chat = set(xy["naturalistic"]["conv_ids"]), set(xy["chat"]["conv_ids"])
    shared_ids = np.array(sorted(ids_nat & ids_chat))
    n_shared = len(shared_ids)
    assert n_shared > 0, f"{model}: chat/naturalistic bundles share NO conv_ids"
    assert n_shared == len(np.unique(shared_ids)), (
        f"{model}: shared conv_ids not unique (conv != row)"
    )
    if expect_n is not None and n_shared != expect_n:
        print(
            f"[nat-contrast] {model}: shared n={n_shared} != expected {expect_n} "
            f"(chat={len(ids_chat)} nat={len(ids_nat)}) — proceeding on the intersection"
        )
    r = {fmt: _reindex_to_shared(xy[fmt], shared_ids) for fmt in FORMATS}

    # Per-format held-out sweep on the SHARED conv_ids + seed -> IDENTICAL folds.
    sweeps = {
        fmt: fit_cells.heldout_r2_sweep(
            r[fmt]["X"],
            r[fmt]["Y"],
            shared_ids,
            n_folds=n_folds,
            seed=seed,
            null_draws=0,
            collect_cosines=False,
        )
        for fmt in FORMATS
    }
    assert np.array_equal(sweeps["chat"]["folds"], sweeps["naturalistic"]["folds"]), (
        f"{model}: fold vectors differ across formats (shared-fold contract broken)"
    )

    delta, equiv = _paired_delta(
        sweeps["naturalistic"],
        sweeps["chat"],
        r["naturalistic"]["Y"],
        r["chat"]["Y"],
        n_boot=n_boot,
        seed=seed,
        equivalence_gate=equivalence_gate,
    )
    return {
        "n_shared": int(n_shared),
        "n_chat_bundle": len(ids_chat),
        "n_naturalistic_bundle": len(ids_nat),
        "r2_per_layer_obs": {fmt: [float(v) for v in sweeps[fmt]["r2_obs"]] for fmt in FORMATS},
        "paired_delta_frozen_layers": delta,
        "equivalence_gate": equiv,
        "headline_layer": HEADLINE_LAYER,
    }


def _strength_ratios(per_model: dict) -> dict:
    """pretrained/instruct held-out R^2 ratio per format at the frozen layers."""
    out: dict[str, dict] = {}
    for fmt in FORMATS:
        r_inst = per_model["instruct"]["r2_per_layer_obs"][fmt]
        r_pre = per_model["pretrained"]["r2_per_layer_obs"][fmt]
        out[fmt] = {}
        for li in FROZEN_LAYERS:
            num, den = r_pre[li], r_inst[li]
            out[fmt][str(li)] = {
                "r2_instruct": float(den),
                "r2_pretrained": float(num),
                "pretrained_over_instruct": (float(num / den) if abs(den) > 1e-12 else None),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="issue-825 chat-vs-naturalistic Track-S contrast")
    parser.add_argument("--turnstore-dir", type=Path, default=Path("data/issue_825/turnstore"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("eval_results/issue_825/naturalistic-single-turn")
    )
    parser.add_argument("--folds", type=int, default=fit_cells.N_FOLDS)
    parser.add_argument("--seed", type=int, default=fit_cells.FIT_SEED)
    parser.add_argument("--n-boot", type=int, default=fit_cells.N_BOOTSTRAP)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument(
        "--equivalence-gate",
        action="store_true",
        help=(
            "run the batched-vs-serial paired-bootstrap oracle check (EQUIV_TOL=1e-8). "
            "Default OFF: the batched helpers are imported verbatim from "
            "issue825_role_contrast, already equivalence-gated in its round 5, so the "
            "smoke runs the gate as evidence and production skips the ~10-min serial "
            "oracle at n=5,000."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny-n bundles: relax the 5,000-conversation expectation + cap n_boot",
    )
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    models = [m for m in args.models.split(",") if m.strip()]
    n_boot = min(args.n_boot, 50) if args.smoke else args.n_boot
    expect_n = None if args.smoke else fit_cells.N_TRACK_S

    per_model: dict[str, dict] = {}
    for model in models:
        print(f"[nat-contrast] model={model} turnstore={args.turnstore_dir}")
        per_model[model] = run_model(
            args.turnstore_dir,
            model,
            n_folds=args.folds,
            seed=args.seed,
            n_boot=n_boot,
            equivalence_gate=args.equivalence_gate,
            expect_n=expect_n,
        )

    ratios = _strength_ratios(per_model) if all(m in per_model for m in MODELS) else {}

    payload = {
        "metadata": {
            **fit_cells._metadata(args.seed, per_model[models[0]]["n_shared"]),
            "script": "scripts/issue825_naturalistic_s_contrast.py",
            "followup_label": "naturalistic-single-turn",
            "n_boot": int(n_boot),
            "n_folds": int(args.folds),
            "frozen_layers": list(FROZEN_LAYERS),
            "headline_layer": HEADLINE_LAYER,
            "smoke": bool(args.smoke),
        },
        "per_model": per_model,
        "strength_ratio_pretrained_over_instruct": ratios,
        "interpretation_note": (
            "Chat and naturalistic fit the SAME 5,000 conversations under two render "
            "formats; the paired delta compares NORMALIZED predictability (each format's "
            "R^2 uses its own target variance), not absolute error. The single manipulated "
            "variable is the Track-S render format."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "format_contrast.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[nat-contrast] wrote {out_path}")
    # Console digest (numbers-first).
    for model in models:
        d19 = per_model[model]["paired_delta_frozen_layers"].get(str(HEADLINE_LAYER), {})
        print(
            f"[nat-contrast] {model} L{HEADLINE_LAYER}: "
            f"nat={d19.get('r2_obs_naturalistic')} chat={d19.get('r2_obs_chat')} "
            f"delta={d19.get('delta_obs')} ci=[{d19.get('ci_lo')},{d19.get('ci_hi')}]"
        )
    print("[nat-contrast] done", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
