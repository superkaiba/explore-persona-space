# ruff: noqa: RUF002
"""Issue #491 matched-strength checkpoint selection (plan v3 §4.4).

Matching basis (REGISTERED, v3 binding fix): the OFFLINE source-cell Q_test
4-float reads at ALL stored checkpoints (``slot_eval --mode ft_run_pipeline``
writes them pre-prune) — NOT the in-loop Q_demo trajectory (telemetry + the
#534 cross-check only).

Per (K, chain) run: matched step t* = argmin_t |ΔG_FT(source; t) − ΔG_ICL
(source; K, chain)|, tolerance ±1.5 nat, else closest-approach with the
residual reported. If the matched source cell sits within 0.5 nat of the
log-prob ceiling (mean trained logp >= −0.5), matching switches to the EOS
margin Δ(z_marker − z_eos) (pre-registered three-spaces fallback).

Anchor step = first checkpoint with ΔG_FT(source) ∈ [5, 12] nat (the recipe
band); if the trajectory never enters the band, the closest-to-midpoint step
is recorded with ``band_entered=False``.
"""

from __future__ import annotations

import argparse
import json
import logging

from explore_persona_space.experiments.icl_vs_ft_491.common import (
    SOURCE_CONTEXT,
    ns_eval_dir,
    repro_metadata,
    write_json,
)

logger = logging.getLogger("i491.matching")

TOLERANCE_NATS = 1.5
CEILING_NATS = -0.5  # mean trained logp above this -> EOS-margin fallback
BAND_LOW, BAND_HIGH = 5.0, 12.0


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _icl_dose(variant_id: str, *, smoke: bool) -> dict:
    """Source-cell ICL dose in BOTH spaces from the icl_panel JSONs."""
    panel_dir = ns_eval_dir(smoke) / "icl_panel"
    vpath = panel_dir / f"{variant_id}.json"
    bpath = panel_dir / "base_noprefix.json"
    for p in (vpath, bpath):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run the ICL panel reads first.")
    variant = json.loads(vpath.read_text())["contexts"][SOURCE_CONTEXT]
    base = json.loads(bpath.read_text())["contexts"][SOURCE_CONTEXT]
    if variant["questions"] != base["questions"]:
        raise AssertionError(f"{variant_id}: source-cell question lists differ from baseline")
    v_stats, b_stats = variant["stats"], base["stats"]
    dose_logp = _mean([v["logp"] - b["logp"] for v, b in zip(v_stats, b_stats, strict=True)])
    dose_margin = _mean(
        [
            (v["z_marker"] - v["z_eos"]) - (b["z_marker"] - b["z_eos"])
            for v, b in zip(v_stats, b_stats, strict=True)
        ]
    )
    return {"dose_logp": dose_logp, "dose_margin": dose_margin}


def _ft_curve(run_id: str, *, smoke: bool) -> dict:
    """Per-step source-cell ΔG (logp + margin) from the matching-basis JSON."""
    path = ns_eval_dir(smoke) / "ft_panel" / f"{run_id}_matching_basis.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run the matching-basis reads first.")
    payload = json.loads(path.read_text())
    base_stats = payload["base"]["stats"]
    curve: dict[int, dict] = {}
    for step_str, entry in payload["per_step"].items():
        s = entry["stats"]
        if len(s) != len(base_stats):
            raise AssertionError(f"{run_id} step {step_str}: stats length mismatch vs base")
        curve[int(step_str)] = {
            "delta_logp": _mean(
                [t["logp"] - b["logp"] for t, b in zip(s, base_stats, strict=True)]
            ),
            "delta_margin": _mean(
                [
                    (t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"])
                    for t, b in zip(s, base_stats, strict=True)
                ]
            ),
            "trained_logp_mean": _mean([t["logp"] for t in s]),
        }
    if not curve:
        raise AssertionError(f"{run_id}: matching-basis curve is empty")
    return curve


def match_run(run_id: str, icl_dose_variant: str, *, smoke: bool = False) -> dict:
    """Compute the matched + anchor steps for one run; update matched_summary.json."""
    dose = _icl_dose(icl_dose_variant, smoke=smoke)
    curve = _ft_curve(run_id, smoke=smoke)
    steps = sorted(curve)

    # Primary matching in log-prob space.
    matched_step = min(steps, key=lambda t: abs(curve[t]["delta_logp"] - dose["dose_logp"]))
    basis = "logp"
    residual = curve[matched_step]["delta_logp"] - dose["dose_logp"]
    ceiling_flagged = curve[matched_step]["trained_logp_mean"] >= CEILING_NATS
    if ceiling_flagged:
        # Pre-registered EOS-margin fallback (three-spaces contract).
        matched_step = min(steps, key=lambda t: abs(curve[t]["delta_margin"] - dose["dose_margin"]))
        basis = "eos_margin"
        residual = curve[matched_step]["delta_margin"] - dose["dose_margin"]

    within = abs(residual) <= TOLERANCE_NATS

    in_band = [t for t in steps if BAND_LOW <= curve[t]["delta_logp"] <= BAND_HIGH]
    if in_band:
        anchor_step, band_entered = in_band[0], True
    else:
        mid = (BAND_LOW + BAND_HIGH) / 2
        anchor_step, band_entered = (
            min(steps, key=lambda t: abs(curve[t]["delta_logp"] - mid)),
            False,
        )

    entry = {
        "run_id": run_id,
        "icl_dose_variant": icl_dose_variant,
        "dose_logp": dose["dose_logp"],
        "dose_margin": dose["dose_margin"],
        "matched_step": matched_step,
        "matched_delta_logp": curve[matched_step]["delta_logp"],
        "matched_delta_margin": curve[matched_step]["delta_margin"],
        "basis": basis,
        "residual": residual,
        "within_tolerance": within,
        "ceiling_flagged": ceiling_flagged,
        "anchor_step": anchor_step,
        "anchor_delta_logp": curve[anchor_step]["delta_logp"],
        "band_entered": band_entered,
        "curve": {str(t): curve[t] for t in steps},
    }

    out_path = ns_eval_dir(smoke) / "matched_pairs" / "matched_summary.json"
    summary = (
        json.loads(out_path.read_text())
        if out_path.exists()
        else {"meta": repro_metadata(), "pairs": {}}
    )
    summary["pairs"][run_id] = entry
    summary["meta"] = repro_metadata()
    write_json(out_path, summary)
    logger.info(
        "%s matched: step=%d basis=%s residual=%+.3f within=%s anchor=%d band_entered=%s",
        run_id,
        matched_step,
        basis,
        residual,
        within,
        anchor_step,
        band_entered,
    )
    return entry


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    from explore_persona_space.experiments.icl_vs_ft_491.data_build import load_run_specs

    spec = load_run_specs()[args.run]
    match_run(args.run, spec["icl_dose_variant"], smoke=args.smoke)


if __name__ == "__main__":
    main()
