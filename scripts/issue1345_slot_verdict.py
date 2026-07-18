#!/usr/bin/env python
"""Issue #1345 story-slot-position-ablation driver (plan v10 §4-§6).

Four phases, dispatched by scripts/issue1345_dispatch.sh --slot-ablation
(variant story_slot_ablation):

  --prefetch-stories  Stage the PINNED kept-stories bundle + yield report
                      (c.STORIES_BUNDLE_REV — never the mutable default
                      branch) into the variant stories dir; assert exactly
                      2,164 rows + the character-name gate (plan §7
                      bundle-integrity gate; exit 3 on miss, NO regeneration
                      path exists in this mode by design).
  --fits              The 7 slot-ablation cells (plan §5) through the reused
                      run_cells core (Gram-space GCV, conv-grouped 5-fold,
                      shuffle nulls, conv bootstrap, L19 preds caches) on the
                      registered row INTERSECTION (slot store ∩ reused r1
                      store; equalize-down row policy, plan §4), then the
                      three-anchor ±0.02 refit-equality gate (exit 3 on a
                      production miss BEFORE any slot read is interpreted).
  --transfer          SECONDARY r1<->r4slot transfer + data-paired A·M·B
                      reparameterization legs per NON-degenerate slot at the
                      frozen layers (matched-capacity nulls at L19). The
                      operator-cosine / rotation-null legs are DROPPED this
                      round (plan §4 item 4).
  --verdict           The paired-deficit verdict lattice: per-slot Δ_k with
                      Bonferroni-4 98.75% CIs + the per-draw max-over-slots
                      paired deficit D (95% CI), selection INSIDE each
                      bootstrap draw (selection-symmetric rule), one shared
                      counts matrix over cached L19 preds (batched GEMM —
                      conv_suffstats machinery, no per-draw loop). Writes
                      slot_verdict_lattice.json.

Under --smoke the identical chain runs at tiny n; production-n-calibrated
verdicts (refit-equality anchors) demote to informational per the r3
gate-calibration rule; production HALT semantics stay byte-untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_fit_cells as fc  # noqa: E402
import issue1345_common as c  # noqa: E402
import numpy as np  # noqa: E402
from issue1345_fit_cells import (  # noqa: E402
    bundle_conv_ids,
    degenerate_fold_reason,
    load_regime_bundle,
    run_cells,
)

L19 = 19
MODEL = "instruct"  # base N/A by scope (inherited, plan v10 §4)
ROW_DROP_FLAG_FRACTION = 0.01  # plan §4: intersection drop > 1% of rows is flagged


def _slot_stem() -> str:
    return c.stem_for(MODEL, "r4slot")


def diagnostics_path(turnstore_dir: Path) -> Path:
    return turnstore_dir / f"{_slot_stem()}_slot_diagnostics.json"


def load_diagnostics(turnstore_dir: Path) -> dict:
    p = diagnostics_path(turnstore_dir)
    assert p.exists(), (
        f"slot diagnostics missing: {p} — run extract_r4_slots (the extractor "
        "writes it before any GPU forward)"
    )
    return json.loads(p.read_text())


def nondegenerate_slots(diag: dict) -> list[str]:
    """Verdict-slot keys whose anchor-coincidence rate is under the threshold."""
    return [k for k, bad in diag["degenerate_verdict_slots"].items() if not bad]


# ---------------------------------------------------------------------------
# Phase: prefetch-stories (plan §7 bundle-integrity gate)
# ---------------------------------------------------------------------------
def prefetch_stories(stories_dir: Path, *, smoke: bool) -> None:
    """Stage the pinned kept bundle + yield report; assert rows + name gate.

    The row-count assert binds in BOTH modes (the staged bundle is full-size
    either way; the extract phase slices rows, never this input). Exit 3 on
    any miss — a short/missing bundle is a fail-loud halt, never a
    regeneration (regeneration would change the corpus, a second variable).
    """
    from explore_persona_space.orchestrate.hub import stage_hub_file

    stories_dir.mkdir(parents=True, exist_ok=True)
    staged = {}
    for fname in ("kept_stories_paired_instruct.jsonl", "story_yield_paired_instruct.json"):
        staged[fname] = stage_hub_file(
            c.HF_DATA_REPO,
            f"{c.STORIES_BUNDLE_PREFIX}/{fname}",
            stories_dir / fname,
            repo_type="dataset",
            revision=c.STORIES_BUNDLE_REV,
        )
        print(f"[prefetch_stories] staged {fname} @ {c.STORIES_BUNDLE_REV[:10]}", flush=True)
    rows = c.read_jsonl(stories_dir / "kept_stories_paired_instruct.jsonl")
    if len(rows) != c.STORIES_BUNDLE_N_ROWS:
        print(
            f"[prefetch_stories] HALT: bundle has {len(rows)} rows, expected "
            f"{c.STORIES_BUNDLE_N_ROWS} @ {c.STORIES_BUNDLE_REV[:10]} — no regeneration "
            "path exists in slot-ablation mode by design (plan v10 §7)",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(3)
    # Character-name gate (the canonical assert_story_character_name check,
    # inlined so this network phase stays torch-free; the extract phase
    # re-runs the canonical gate at its own entry).
    stored = json.loads((stories_dir / "story_yield_paired_instruct.json").read_text()).get(
        "story_character_name", "ARIA"
    )
    if stored != c.STORY_CHARACTER_NAME:
        print(
            f"[prefetch_stories] HALT: bundle character name {stored!r} != runtime "
            f"{c.STORY_CHARACTER_NAME!r} — launch with the matching "
            "EPM_STORY_CHARACTER_NAME (plan v10 §4)",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(3)
    print(
        f"[prefetch_stories] OK: {len(rows)} rows, character={stored} (smoke={smoke})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Phase: fits (registered row intersection -> 7 cells -> refit-equality gate)
# ---------------------------------------------------------------------------
def build_row_coverage(turnstore_dir: Path, out_dir: Path, *, smoke: bool) -> dict:
    """Registered row set = slot-store conv_ids ∩ reused-r1-store conv_ids.

    Equalize-down row policy (plan §4): every slot cell AND the chat matched
    comparator fit on this one intersection; a drop > 1% of slot-store rows
    is FLAGGED (reported, not a halt). Persisted to slot_row_coverage.json —
    the verdict phase's registered-id source.
    """
    slot_ids = [str(x) for x in bundle_conv_ids(turnstore_dir, MODEL, "r4slot")]
    r1_ids = {str(x) for x in bundle_conv_ids(turnstore_dir, MODEL, "r1")}
    assert len(slot_ids) == len(set(slot_ids)), "slot store has duplicate conv_ids"
    registered = sorted(set(slot_ids) & r1_ids)
    drop_fraction = 1.0 - (len(registered) / max(len(slot_ids), 1))
    if not registered:
        assert smoke, (
            "slot-store ∩ r1-store conv intersection is EMPTY at production n — "
            "the paired corpus is drawn FROM the shared conversations (plan v10 §4); "
            "staging/extraction drift"
        )
        print(
            "[slot-fits][smoke] empty slot∩r1 intersection (smoke stages shard000 "
            "only) — slot cells fit on the slot-store rows; the chat matched "
            "comparator + paired-D verdict skip informationally",
            flush=True,
        )
    cov = {
        "metadata": c.metadata(fc.FIT_SEED, len(registered), "scripts/issue1345_slot_verdict.py"),
        "n_slot_store_rows": len(slot_ids),
        "n_r1_store_rows": len(r1_ids),
        "n_registered": len(registered),
        "registered_conv_ids": registered,
        "drop_fraction_vs_slot_store": drop_fraction,
        "drop_flagged_over_1pct": bool(drop_fraction > ROW_DROP_FLAG_FRACTION),
        "smoke": bool(smoke),
    }
    if cov["drop_flagged_over_1pct"]:
        print(
            f"[slot-fits] FLAG: intersection drops {drop_fraction:.2%} of slot-store "
            "rows (> 1% — reported in the lattice, plan v10 §4 row policy)",
            flush=True,
        )
    c.write_json(out_dir / "slot_row_coverage.json", cov)
    return cov


def refit_equality_slots(out_dir: Path, *, tol: float = c.PARITY_TOL, smoke: bool = False) -> None:
    """Three-anchor ±0.02 refit-equality gate (plan §7): the fresh capture's
    anchor context cell must reproduce the landed -0.3056, the prefix cell
    -1.3714, and the recomputed chat matched cell +0.2426 (L19, committed
    landed JSONs the references) — else exit 3 BEFORE slot fits are
    interpreted. Informational under --smoke (fresh-capture anchors bind at
    production n only — gate-calibration rule)."""
    results, failures = {}, []
    for cid, ref_file in c.SLOT_REFIT_ANCHOR_FILES.items():
        ref_path = Path(ref_file)
        assert ref_path.exists(), (
            f"refit-equality reference missing: {ref_path} (branch clone carries "
            "eval_results/issue_1345 — broken/sparse checkout?)"
        )
        anchor = float(json.loads(ref_path.read_text())["r2_per_layer_obs"][L19])
        doc = c.SLOT_REFIT_ANCHOR_DOC[cid]
        assert abs(anchor - doc) < 0.005, (cid, anchor, doc)
        new_path = out_dir / f"cells_{cid}.json"
        if not new_path.exists():
            assert smoke, f"refit cell JSON missing outside smoke: {new_path}"
            print(f"[refit-eq-slots][smoke] {cid}: no cell JSON (smoke skip)", flush=True)
            continue
        ours = float(json.loads(new_path.read_text())["r2_per_layer_obs"][L19])
        dev = abs(ours - anchor)
        ok = dev <= tol
        results[cid] = {
            "fresh_l19_r2": ours,
            "landed_l19_r2": anchor,
            "abs_dev": dev,
            "pass": ok,
        }
        if smoke:
            print(
                f"[refit-eq-slots][smoke] informational: {cid} fresh={ours:.4f} "
                f"landed={anchor:.4f} dev={dev:.4f} (±{tol} binds at production n only)",
                flush=True,
            )
        else:
            print(
                f"[refit-eq-slots] {cid}: fresh={ours:.6f} landed={anchor:.6f} "
                f"dev={dev:.4f} ({'PASS' if ok else 'FAIL'})",
                flush=True,
            )
            if not ok:
                failures.append(cid)
    payload = {
        "metadata": c.metadata(fc.FIT_SEED, len(results), "scripts/issue1345_slot_verdict.py"),
        "tolerance": tol,
        "anchors": c.SLOT_REFIT_ANCHOR_DOC,
        "mode": "smoke-informational" if smoke else "binding",
        "results": results,
        "pass": None if smoke else not failures,
    }
    c.write_json(out_dir / "refit_equality_slots.json", payload)
    if failures and not smoke:
        print(
            f"[refit-eq-slots] HALT: {failures} deviate > ±{tol} from the landed "
            "anchors (plan v10 §7) — diagnose extraction drift before any slot read",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(3)


def run_fits(
    turnstore_dir: Path,
    out_dir: Path,
    preds_dir: Path,
    *,
    n_folds: int,
    seed: int,
    null_draws: int,
    n_boot: int,
    smoke: bool,
) -> None:
    """The 7 slot-ablation cells on the registered intersection + the gate."""
    cov = build_row_coverage(turnstore_dir, out_dir, smoke=smoke)
    registered = cov["registered_conv_ids"]
    cells = c.slot_ablation_cells()
    if not registered:
        # Smoke-only branch (build_row_coverage asserts non-empty at production):
        # slot cells fit their own store rows; the chat comparator is skipped.
        cells = [x for x in cells if x["cell_id"] != c.SLOT_CHAT_MATCHED_CELL]
        allowlist_fn = lambda cell: None  # noqa: E731

        print("[slot-fits][smoke] chat matched cell skipped (empty intersection)", flush=True)
    else:
        allowlist_fn = lambda cell: registered  # noqa: E731
    run_cells(
        turnstore_dir,
        out_dir,
        preds_dir,
        cells,
        None,
        n_folds=n_folds,
        seed=seed,
        null_draws=null_draws,
        n_boot=n_boot,
        smoke=smoke,
        allowlist_fn=allowlist_fn,
    )
    refit_equality_slots(out_dir, smoke=smoke)


# ---------------------------------------------------------------------------
# Phase: transfer (SECONDARY r1<->r4slot legs, non-degenerate slots only)
# ---------------------------------------------------------------------------
def _slot_arm_xy(slot_bundle: dict, slot_index: int) -> dict:
    """(X, Y, conv_ids) for one slot-store read, rows conv-sorted."""
    xy = fc._cell_xy(
        slot_bundle,
        {"slot_index": slot_index, "target_turn_index": c.TARGET_TURN_INDEX["r4slot"]},
    )
    conv = np.asarray([str(x) for x in xy["conv_ids"]])
    order = np.argsort(conv, kind="stable")
    return {"X": xy["X"][order], "Y": xy["Y"][order], "conv_ids": conv[order]}


def _reparam_leg(
    xa: dict, xb_: dict, *, seed: int, n_reparam_null_draws: int, pair_label: str
) -> dict:
    """Data-paired A·M·B reparameterization recovery (plan §4 item 4).

    The leg_b_battery core MINUS the dropped cosine legs (activation-
    Procrustes rotation nulls / alignment-capacity probes — plan §4: cosine
    reads are inherited context; re-running them per slot adds cost without
    touching the lattice). Direction keys keep the reused battery's naming:
    'i' = the r1 chat side (xa), 'b' = the slot side (xb_); b2i recovers the
    chat regime via the slot center, i2b the reverse.
    """
    import issue825_map_alignment as ma
    from issue1345_operator_comparison import _t, reparam_null_battery

    assert np.array_equal(xa["conv_ids"], xb_["conv_ids"]), f"{pair_label}: rows misaligned"
    data = {
        "Xi": {layer: _t(xa["X"][:, layer, :]) for layer in fc.FROZEN_LAYERS},
        "Yi": {layer: _t(xa["Y"][:, layer, :]) for layer in fc.FROZEN_LAYERS},
        "Xb": {layer: _t(xb_["X"][:, layer, :]) for layer in fc.FROZEN_LAYERS},
        "Yb": {layer: _t(xb_["Y"][:, layer, :]) for layer in fc.FROZEN_LAYERS},
        "conv": xa["conv_ids"],
    }
    folds = fc._cv_folds(xa["conv_ids"], ma.N_FOLDS, seed)
    reparam: dict = {}
    for layer in fc.FROZEN_LAYERS:
        reparam[str(layer)] = {"battery": ma._layer_battery(data, folds, layer, do_orth=True)}
        if layer == L19:
            reparam[str(layer)]["matched_capacity_nulls"] = reparam_null_battery(
                data, folds, layer, n_draws=n_reparam_null_draws, seed=seed + 13
            )
    b19 = reparam[str(L19)]["battery"]
    nulls19 = reparam[str(L19)]["matched_capacity_nulls"]
    recov = {
        "b2i": b19["composition"]["linear"]["comp_samefn_b2i"],
        "i2b": b19["composition"]["linear"]["comp_samefn_i2b"],
    }
    within = {"b2i": b19["ceilings"]["within_instruct"], "i2b": b19["ceilings"]["within_base"]}
    delta_terms = {
        d: recov[d] - max(within[d] - c.DELTA_SAME_MARGIN, nulls19[d]["null_recovery_r2"])
        for d in ("b2i", "i2b")
    }
    return {
        "direction_key": {"i": "r1_chat", "b": pair_label},
        "reparam": reparam,
        "recov": recov,
        "within": within,
        "delta_terms": delta_terms,
        "delta_reparam": float(min(delta_terms.values())),
    }


def run_transfer(
    turnstore_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    null_draws: int,
    n_reparam_null_draws: int,
    smoke: bool,
) -> None:
    """r1<->r4slot transfer + reparam legs per non-degenerate verdict slot."""
    from issue1345_cross_regime_transfer import load_arm_xy, subset_rows, transfer_sweep

    diag = load_diagnostics(turnstore_dir)
    cov_path = out_dir / "slot_row_coverage.json"
    assert cov_path.exists(), f"{cov_path} missing — run --fits first"
    registered = json.loads(cov_path.read_text())["registered_conv_ids"]
    if not registered:
        assert smoke, "empty registered row set at production n (build_row_coverage gate)"
        print("[slot-transfer][smoke] SKIP: empty slot∩r1 intersection", flush=True)
        return
    slot_bundle = load_regime_bundle(turnstore_dir, MODEL, "r4slot")
    r1_bundle = load_regime_bundle(turnstore_dir, MODEL, "r1")
    xa_full = load_arm_xy(r1_bundle, "r1", "context")
    live = nondegenerate_slots(diag)
    for slot_key in c.SLOT_VERDICT_CELLS:
        if slot_key not in live:
            print(
                f"[slot-transfer] SKIP {slot_key}: degenerate (anchor-coincidence "
                f"{diag['anchor_coincidence_rates'][c.SLOT_NAME_FOR_CELL[c.SLOT_VERDICT_CELLS[slot_key]]]:.2%}"
                " "
                f"> {c.SLOT_DEGENERACY_COINCIDENCE_MAX:.0%})",
                flush=True,
            )
            continue
        cid = c.SLOT_VERDICT_CELLS[slot_key]
        slot_index = c.SLOT_CELL_INDEX[cid][0]
        xa = subset_rows(xa_full, registered, smoke=smoke, label=f"r1@{slot_key}")
        xb_ = subset_rows(
            _slot_arm_xy(slot_bundle, slot_index), registered, smoke=smoke, label=slot_key
        )
        if xa is None or xb_ is None:
            continue
        reason = degenerate_fold_reason(
            xa["conv_ids"], n_folds=fc.N_FOLDS, seed=seed, tgt_conv_ids=xb_["conv_ids"]
        )
        if reason and smoke:
            print(f"[slot-transfer][smoke] SKIP {slot_key}: {reason}", flush=True)
            continue
        assert not reason, f"{slot_key}: degenerate folds at production n — {reason}"
        legs = {
            "r1_to_slot": transfer_sweep(xa, xb_, seed=seed, null_draws=null_draws),
            "slot_to_r1": transfer_sweep(xb_, xa, seed=seed, null_draws=null_draws),
            "within_slot": transfer_sweep(xb_, xb_, seed=seed, null_draws=0),
            "within_r1": transfer_sweep(xa, xa, seed=seed, null_draws=0),
        }
        xfer_payload = {
            "metadata": c.metadata(seed, len(xa["conv_ids"]), "scripts/issue1345_slot_verdict.py"),
            "slot": slot_key,
            "cell_id": cid,
            "n_rows": len(xa["conv_ids"]),
            "delta_l19": {
                "r1_to_slot": legs["r1_to_slot"]["r2_by_layer"][str(L19)]
                - legs["within_slot"]["r2_by_layer"][str(L19)],
                "slot_to_r1": legs["slot_to_r1"]["r2_by_layer"][str(L19)]
                - legs["within_r1"]["r2_by_layer"][str(L19)],
            },
            "legs": {
                name: {
                    k: v
                    for k, v in sw.items()
                    if not k.startswith(("preds", "fitted", "true", "conv"))
                }
                for name, sw in legs.items()
            },
        }
        c.write_json(out_dir / f"cross_regime_transfer_r1_r4slot_{slot_key}.json", xfer_payload)
        rep = _reparam_leg(
            xa,
            xb_,
            seed=seed,
            n_reparam_null_draws=n_reparam_null_draws,
            pair_label=f"r4slot_{slot_key}",
        )
        rep_payload = {
            "metadata": c.metadata(seed, len(xa["conv_ids"]), "scripts/issue1345_slot_verdict.py"),
            "slot": slot_key,
            "cell_id": cid,
            **rep,
        }
        c.write_json(out_dir / f"reparam_recovery_r1_r4slot_{slot_key}.json", rep_payload)


# ---------------------------------------------------------------------------
# Phase: verdict (paired D + Bonferroni-4 lattice — plan §3/§6)
# ---------------------------------------------------------------------------
def _load_preds(preds_dir: Path, cell_id: str) -> dict | None:
    p = preds_dir / f"{cell_id}_L{L19}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    assert int(d["layer"][0]) == L19, (cell_id, int(d["layer"][0]))
    return {
        "pred": d["pred"].astype(np.float64),
        "true": d["true"].astype(np.float64),
        "conv_ids": np.asarray([str(x) for x in d["conv_ids"]]),
    }


def _restrict(arm: dict, registered: list[str], cell_id: str) -> dict:
    """Row-coverage assert (plan §3) + restriction to the registered set."""
    have = set(arm["conv_ids"])
    missing = [x for x in registered if x not in have]
    assert not missing, (
        f"{cell_id}: {len(missing)} registered conv_ids missing from the preds cache "
        f"(first: {missing[:3]}) — row-coverage assert (plan v10 §3)"
    )
    keep = np.isin(arm["conv_ids"], np.asarray(registered))
    return {
        "pred": arm["pred"][keep],
        "true": arm["true"][keep],
        "conv_ids": arm["conv_ids"][keep],
    }


def paired_deficit_battery(
    arms: dict[str, dict], chat_key: str, verdict_keys: list[str], *, n_boot: int, seed: int
) -> dict:
    """Vectorized paired conversation bootstrap over cached L19 preds.

    ONE shared with-replacement counts matrix drives every arm's pooled-R^2
    draws (batched subset-sum GEMMs — c.conv_suffstats / c.batched_conv_r2;
    no per-draw Python loop); D_k = Δ_k - R^2_chat per draw; D = per-draw max
    over the verdict slots (selection inside the draw). Bonferroni-4 98.75%
    CIs on Δ_k; 95% on D (plan §3).
    """
    suffs, uniq_ref = {}, None
    for name, arm in arms.items():
        suff = c.conv_suffstats(arm["pred"], arm["true"], arm["conv_ids"])
        if uniq_ref is None:
            uniq_ref = suff["uniq"]
        assert np.array_equal(suff["uniq"], uniq_ref), (
            f"{name}: conversation universe differs across arms — paired draws undefined"
        )
        suffs[name] = suff
    counts = c.bootstrap_counts(len(uniq_ref), n_boot, seed)
    draws = {name: c.batched_conv_r2(counts, suffs[name]) for name in suffs}
    obs = {name: fc._pooled_r2(arms[name]["pred"], arms[name]["true"]) for name in arms}
    alpha_b = 1.0 - c.SLOT_BONFERRONI_LEVEL  # 0.0125 -> quantiles at 0.00625/0.99375
    per_slot = {}
    d_draw_rows = []
    for k in verdict_keys:
        delta_d = draws[k]
        dk_d = delta_d - draws[chat_key]
        d_draw_rows.append(dk_d)
        per_slot[k] = {
            "r2_l19_obs": float(obs[k]),
            "delta_ci95": [
                float(np.nanquantile(delta_d, 0.025)),
                float(np.nanquantile(delta_d, 0.975)),
            ],
            "delta_ci_bonferroni4": [
                float(np.nanquantile(delta_d, alpha_b / 2)),
                float(np.nanquantile(delta_d, 1.0 - alpha_b / 2)),
            ],
            "d_k_obs": float(obs[k] - obs[chat_key]),
            "d_k_ci95": [
                float(np.nanquantile(dk_d, 0.025)),
                float(np.nanquantile(dk_d, 0.975)),
            ],
        }
    d_draws = np.max(np.stack(d_draw_rows, axis=0), axis=0)  # per-draw max (paired)
    n_nan = int(np.isnan(d_draws).sum())
    d_ci95 = [float(np.nanquantile(d_draws, 0.025)), float(np.nanquantile(d_draws, 0.975))]
    return {
        "n_boot": n_boot,
        "seed": seed,
        "n_conversations": len(uniq_ref),
        "n_nan_draws": n_nan,
        "chat_r2_l19_obs": float(obs[chat_key]),
        "per_slot": per_slot,
        "d_obs": float(max(obs[k] - obs[chat_key] for k in verdict_keys)),
        "d_ci95": d_ci95,
        "arm_obs": {k: float(v) for k, v in obs.items()},
        # Raw draws (numpy; run_verdict persists them for the deficit-panel
        # figure, then strips this key from the JSON lattice).
        "_draws": {"d": d_draws, "d_k": dict(zip(verdict_keys, d_draw_rows, strict=True))},
    }


def classify_verdict(battery: dict, verdict_keys: list[str]) -> str:
    """Plan §3 DISJOINT-and-exhaustive trichotomy (endpoint at 0 == straddle)."""
    _d_lo, d_hi = battery["d_ci95"]
    if not (d_hi < 0.0):
        return "slot_artifact"
    all_below = all(battery["per_slot"][k]["delta_ci_bonferroni4"][1] < 0.0 for k in verdict_keys)
    return "representation_level_collapse" if all_below else "intermediate"


def run_verdict(
    turnstore_dir: Path, out_dir: Path, preds_dir: Path, *, n_boot: int, seed: int, smoke: bool
) -> None:
    """Assemble slot_verdict_lattice.json (plan §6.5 primary deliverable)."""
    diag = load_diagnostics(turnstore_dir)
    cov = json.loads((out_dir / "slot_row_coverage.json").read_text())
    registered = cov["registered_conv_ids"]
    refit_path = out_dir / "refit_equality_slots.json"
    refit = json.loads(refit_path.read_text()) if refit_path.exists() else {}
    live = nondegenerate_slots(diag)
    lattice: dict = {
        "metadata": c.metadata(seed, len(registered), "scripts/issue1345_slot_verdict.py"),
        "smoke": bool(smoke),
        "registered_n_rows": len(registered),
        "row_coverage": {k: v for k, v in cov.items() if k != "registered_conv_ids"},
        "anchor_coincidence_rates": diag["anchor_coincidence_rates"],
        "answer_overlap_rates": diag["answer_overlap_rates"],
        "degenerate_verdict_slots": diag["degenerate_verdict_slots"],
        "nondegenerate_slots": live,
        "refit_equality": {"pass": refit.get("pass"), "mode": refit.get("mode")},
        "bonferroni_level": c.SLOT_BONFERRONI_LEVEL,
    }
    if len(live) < 2:
        # Reportable outcome, never a silent rescope (plan §4 degeneracy policy).
        lattice["verdict"] = "inconclusive_by_degeneracy"
        lattice["verdict_note"] = (
            f"only {len(live)} non-degenerate slot(s) remain (< 2) — "
            "Inconclusive-by-degeneracy (plan v10 §4)"
        )
        c.write_json(out_dir / "slot_verdict_lattice.json", lattice)
        return
    wanted = {k: c.SLOT_VERDICT_CELLS[k] for k in live}
    wanted["chat_matched"] = c.SLOT_CHAT_MATCHED_CELL
    wanted["anchor"] = c.SLOT_ANCHOR_CELL
    wanted["prefix"] = c.SLOT_PREFIX_CELL
    arms_raw = {name: _load_preds(preds_dir, cid) for name, cid in wanted.items()}
    missing = sorted(n for n, a in arms_raw.items() if a is None)
    if missing:
        assert smoke, (
            f"preds caches missing at production n for arms {missing} — fits drift "
            "(run --fits; plan v10 §3 statistical-input existence)"
        )
        lattice["verdict"] = "smoke_skip_missing_arms"
        lattice["verdict_note"] = f"smoke: preds caches missing for {missing}"
        c.write_json(out_dir / "slot_verdict_lattice.json", lattice)
        return
    arms = {name: _restrict(arm, registered, wanted[name]) for name, arm in arms_raw.items()}
    battery = paired_deficit_battery(arms, "chat_matched", live, n_boot=n_boot, seed=seed)
    draws = battery.pop("_draws")
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        preds_dir / "slot_verdict_draws.npz",
        d=draws["d"].astype(np.float32),
        **{f"d_k_{k}": v.astype(np.float32) for k, v in draws["d_k"].items()},
    )
    lattice["battery"] = battery
    lattice["verdict"] = classify_verdict(battery, live)
    c.write_json(out_dir / "slot_verdict_lattice.json", lattice)
    print(
        f"[slot-verdict] verdict={lattice['verdict']} D_obs={battery['d_obs']:+.4f} "
        f"D_ci95=[{battery['d_ci95'][0]:+.4f}, {battery['d_ci95'][1]:+.4f}] "
        f"chat={battery['chat_r2_l19_obs']:+.4f} (n_boot={n_boot})",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefetch-stories", action="store_true")
    ap.add_argument("--fits", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--stories-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--out-dir", type=Path, default=c.EVAL_DIR)
    ap.add_argument("--preds-dir", type=Path, default=c.PREDS_CACHE_DIR)
    ap.add_argument("--folds", type=int, default=fc.N_FOLDS)
    ap.add_argument("--seed", type=int, default=fc.FIT_SEED)
    ap.add_argument("--null-draws", type=int, default=fc.N_NULL_DRAWS)
    ap.add_argument("--n-boot", type=int, default=fc.N_BOOTSTRAP)
    ap.add_argument("--reparam-null-draws", type=int, default=c.N_REPARAM_NULL_DRAWS)
    ap.add_argument("--transfer-null-draws", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    assert c.HAS_SLOT_ABLATION, (
        f"issue1345_slot_verdict requires EPM_I1345_VARIANT in {c.SLOT_ABLATION_VARIANTS} "
        f"(got {c.VARIANT!r})"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ran = False
    if args.prefetch_stories:
        prefetch_stories(args.stories_dir, smoke=args.smoke)
        ran = True
    if args.fits:
        run_fits(
            args.turnstore_dir,
            args.out_dir,
            args.preds_dir,
            n_folds=args.folds,
            seed=args.seed,
            null_draws=args.null_draws,
            n_boot=args.n_boot,
            smoke=args.smoke,
        )
        ran = True
    if args.transfer:
        run_transfer(
            args.turnstore_dir,
            args.out_dir,
            seed=args.seed,
            null_draws=args.transfer_null_draws,
            n_reparam_null_draws=args.reparam_null_draws,
            smoke=args.smoke,
        )
        ran = True
    if args.verdict:
        run_verdict(
            args.turnstore_dir,
            args.out_dir,
            args.preds_dir,
            n_boot=args.n_boot,
            seed=args.seed,
            smoke=args.smoke,
        )
        ran = True
    assert ran, "pass at least one of --prefetch-stories/--fits/--transfer/--verdict"


if __name__ == "__main__":
    main()
