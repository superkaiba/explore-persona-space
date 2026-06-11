# ruff: noqa: RUF002, RUF003  # em-dash + § intentional
"""Task #600 — synthetic-trajectory smoke harness for the registered analysis.

Generates SCHEMA-TRUE synthetic ``trajectory.json`` files from the REAL
committed design manifest and drives the production ``analyze_600`` entrypoint
end-to-end (stats + JSON + figures), then hard-asserts the registered
behavior. CPU-only; run on the VM before any pod data lands.

Modes (``--mode all`` runs every one):

  signal   — NEAR target suppressed ∝ frac → expect ``success_local_suppression``.
  null     — no injection → expect ``null_promotable_bounded`` (the registered
             test is non-anticonservative on its own null).
  fallback — pair 1 (first manifest target) never co-passes: its NEAR arm
             passes gates ONLY at frac 0.33, its CONTROL arm only from 0.75 →
             expect the §4.8(c) band-entry-fallback read: pair SURVIVES with
             ``unmatched_step: true`` (NEAR@0.33, CONTROL@0.75), is NOT
             failed-gate, k_surviving == 6, and the matched-step-only
             sensitivity permutation covers the other 5 pairs.
  mixed    — pair 1 saturates after frac 0.50 (both arms) → its headline frac
             is 0.50 while the others sit at 1.00 → expect §6.7(c) per-pair
             same-checkpoint calibration: two within-checkpoint groups, and
             the 0.50 pair's band = the 0.50-frac gap median (≠ the 1.00-frac
             median the retired global-band code would have used).
  descope  — null injection with seed 219 dropped from EVERY cell → expect
             seeds inferred as [42, 137] and all 6 pairs analyzed (the §9
             rung-1 descope, no spurious ``missing_cells``).
  signmix  — 5/6 targets injected with a SMALL negative paired difference and
             the 6th with a larger positive one; per-seed common offsets on
             each cell's target persona inflate the same-mix gaps so every
             |d| sits WITHIN the noise band and the permutation lands
             p > 0.05 → the §3 H-null sign-mixed conjunct (n_negative ≤ 4
             at k = 6) FAILS, so expect ``indeterminate`` with
             ``sign_mixed: false`` — NEVER ``null_promotable_bounded``
             (5/6-negative + p > 0.05 + within-band is suggestive-only
             evidence FOR suppression per §6 item 2).

HF note: ``HF_HUB_OFFLINE=1`` is set by default so the bubble-radius
autofetch fail-softs deterministically (recorded skip) instead of hitting the
network; pass ``--allow-hf`` to exercise the real autofetch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.targeted_proximity_600 import (
    SEEDS,
    SOURCE_PERSONA,
    TRAJECTORY_CHECKPOINT_FRACTIONS,
)
from explore_persona_space.experiments.targeted_proximity_600.analyze import analyze_600
from explore_persona_space.experiments.targeted_proximity_600.cells import (
    cell_specs_from_manifest,
    load_manifest,
)

log = logging.getLogger("issue_600.analyze_smoke")

N_Q = 10
MODES = ("signal", "null", "fallback", "mixed", "descope", "signmix")
# signmix injection (normalized by src ΔG ≈ 16 at the frac-1.00 headline read):
# per-seed common target-persona offsets ±0.35·rank inflate same-mix gap
# medians to ≈ 0.022 while the injected |d| (0.08/16 ≈ 0.005 negative on 5
# targets, 0.16/16 ≈ 0.010 positive on the special target) stays within-band
# and the signed mean lands the one-sided permutation at p ≈ 0.27.
SIGNMIX_SEED_OFFSET = 0.35
SIGNMIX_NEG_SHIFT = -0.08
SIGNMIX_POS_SHIFT = 0.16


def _u(name: str) -> float:
    """Deterministic uniform in [0, 1) from a string (stable across runs)."""
    return (zlib.crc32(name.encode()) % 10_000) / 10_000.0


def _cell_payload(
    *,
    spec,
    seed: int,
    held_out_personas: list[str],
    mode: str,
    special_pair_target: str,
) -> dict:
    """One schema-true trajectory payload for (cell, seed) under ``mode``."""
    rng = np.random.default_rng(zlib.crc32(f"{mode}|{spec.slug}|{seed}".encode()))
    is_special = spec.target == special_pair_target
    checkpoints = []
    for frac in TRAJECTORY_CHECKPOINT_FRACTIONS:
        dg = 6.0 + 10.0 * frac + float(rng.normal(0, 0.3))
        g_logp = -1.5 + float(rng.normal(0, 0.1))
        if mode == "fallback" and is_special:
            if spec.condition == "near":
                # Passes ONLY at 0.33: floored before, saturated after.
                if frac < 0.33:
                    dg = 2.0
                elif frac > 0.34:
                    g_logp = -0.05  # fails gate (b) sub-saturation
            else:
                # Passes only from 0.75 (floored before).
                if frac < 0.75:
                    dg = 2.0
        if mode == "mixed" and is_special and frac > 0.51:
            g_logp = -0.05  # both arms saturate after 0.50 → headline 0.50
        held_out = {}
        for p in held_out_personas:
            base = 0.5 + 0.3 * _u(p)
            sigma = 0.01 + 0.04 * frac  # frac-dependent → gap medians differ across fracs
            recs = {}
            for qi in range(N_Q):
                v = base * (0.3 + 0.7 * frac) + float(rng.normal(0, sigma))
                if mode == "signal" and spec.condition == "near" and p == spec.target:
                    v -= 0.08 * dg * frac  # normalized suppression ≈ −0.08·frac
                if mode == "signmix" and p == spec.target:
                    # Per-seed COMMON offset (identical in the NEAR and CONTROL
                    # cells of the pair → cancels in the paired difference up to
                    # src-ΔG jitter, but differs ACROSS seeds → inflates the
                    # same-mix run-noise gaps so the injected |d| is within-band).
                    v += SIGNMIX_SEED_OFFSET * SEEDS.index(seed)
                    if spec.condition == "near":
                        # 5/6 targets slightly NEGATIVE, the special one larger
                        # POSITIVE → sign-skewed (n_negative = 5) with p > 0.05.
                        v += SIGNMIX_POS_SHIFT if is_special else SIGNMIX_NEG_SHIFT
                recs[f"q{qi}"] = {
                    "delta_g": v,
                    "argmax_marker": False,
                    "delta_margin": v * 1.05,
                }
            held_out[p] = recs
        checkpoints.append(
            {
                "frac": frac,
                "source_self": {"delta_g_mean": dg, "g_logp_mean": g_logp},
                "held_out": held_out,
            }
        )
    return {
        "cell": spec.slug,
        "seed": seed,
        "source": SOURCE_PERSONA,
        "checkpoints": checkpoints,
        "synthetic": True,
        "smoke_mode": mode,
    }


def build_synthetic_sweep(
    manifest: dict, sweep_dir: Path, mode: str, drop_seed: int | None = None
) -> None:
    """Write 12 cells × seeds synthetic trajectory.json files under ``sweep_dir``."""
    specs = cell_specs_from_manifest(manifest)
    targets = [t["name"] for t in manifest["targets"]]
    special = targets[0]
    missing_targets = [t for t in targets if t not in manifest["held_out_panel"]]
    assert not missing_targets, f"targets not in held_out_panel: {missing_targets}"
    seeds = [s for s in SEEDS if s != drop_seed]
    for spec in specs:
        held = sorted(set(manifest["held_out_panel"]) | set(spec.panel))
        for seed in seeds:
            payload = _cell_payload(
                spec=spec,
                seed=seed,
                held_out_personas=held,
                mode=mode,
                special_pair_target=special,
            )
            out = sweep_dir / spec.slug / f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "trajectory.json").write_text(json.dumps(payload))
    log.info("[smoke:%s] wrote %d synthetic trajectories", mode, len(specs) * len(seeds))


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run_mode(mode: str, manifest_path: Path, out_root: Path) -> dict:
    """Build the synthetic sweep for ``mode``, run analyze_600, assert behavior."""
    manifest = load_manifest(manifest_path)
    targets = [t["name"] for t in manifest["targets"]]
    special = targets[0]
    root = out_root / mode
    if root.exists():
        shutil.rmtree(root)
    sweep_dir = root / "sweep"
    drop_seed = 219 if mode == "descope" else None
    build_synthetic_sweep(manifest, sweep_dir, "null" if mode == "descope" else mode, drop_seed)
    result = analyze_600(
        manifest_path=manifest_path,
        sweep_dir=sweep_dir,
        analysis_dir=root / "analysis",
        figures_dir=root / "figures",
    )
    label = result["outcome"]["label"]
    rn = result["run_noise"]

    if mode == "signal":
        _assert(
            label == "success_local_suppression",
            f"signal: expected success_local_suppression, got {label}",
        )
        _assert(rn["effect_above_noise_band"] is True, "signal: effect not above the noise band")
    elif mode in ("null", "descope"):
        _assert(
            label == "null_promotable_bounded",
            f"{mode}: expected null_promotable_bounded, got {label} "
            "(the registered test must not be anticonservative on its own null)",
        )
        _assert(rn["effect_within_noise_band"] is True, f"{mode}: effect not within the band")
        # §3 sign-mixed conjunct: the synthetic null must exercise the
        # promotable cell THROUGH the sign-mix gate, not around it.
        n_neg = result["sign_test"]["n_negative"]
        _assert(
            n_neg <= result["k_surviving"] - 2,
            f"{mode}: null injection landed sign-skewed (n_negative={n_neg}, "
            f"k={result['k_surviving']}) — it no longer exercises the promotable cell",
        )
        _assert(
            result["outcome"]["sign_mixed"] is True,
            f"{mode}: outcome.sign_mixed is not True (n_negative={n_neg})",
        )
    elif mode == "signmix":
        out = result["outcome"]
        # Isolate the §3 sign-mix conjunct: the OTHER two H-null conjuncts hold…
        _assert(
            out["permutation_p"] > 0.05,
            f"signmix: permutation p={out['permutation_p']} not > 0.05 — injection "
            "does not isolate the sign-mix conjunct",
        )
        _assert(
            rn["effect_within_noise_band"] is True,
            "signmix: |d| not within the noise band — injection does not isolate "
            "the sign-mix conjunct",
        )
        n_neg = result["sign_test"]["n_negative"]
        _assert(n_neg == 5, f"signmix: expected 5/6 targets negative, got {n_neg}/6")
        # …so ONLY the sign-mix failure may block the promotable null.
        _assert(out["sign_mixed"] is False, "signmix: outcome.sign_mixed must be False")
        _assert(
            label == "indeterminate",
            f"signmix: expected indeterminate (5/6-negative + p > 0.05 + within-band "
            f"is suggestive-only per §6 item 2, NOT a promotable null), got {label}",
        )
    if mode == "descope":
        _assert(
            result["seeds_realized"] == [42, 137],
            f"descope: seeds_realized {result['seeds_realized']} != [42, 137]",
        )
        _assert(
            result["k_surviving"] == len(targets) and not result["failed_gate_pairs"],
            "descope: dropping seed 219 from every cell must NOT mark pairs missing "
            f"(k={result['k_surviving']}, failed={result['failed_gate_pairs']})",
        )
    if mode == "fallback":
        _assert(
            result["fallback_pairs"] == [special],
            f"fallback: expected fallback_pairs == [{special}], got {result['fallback_pairs']}",
        )
        _assert(
            special not in result["failed_gate_pairs"],
            "fallback: §4.8(c) violation — fallback pair conflated with failed-gate",
        )
        _assert(
            result["k_surviving"] == len(targets),
            f"fallback: pair must SURVIVE (k={result['k_surviving']} != {len(targets)})",
        )
        entry = result["per_pair"][special]
        _assert(entry["unmatched_step"] is True, "fallback: unmatched_step flag missing")
        _assert(
            (entry["frac_near"], entry["frac_ctrl"]) == (0.33, 0.75),
            f"fallback: read fracs ({entry['frac_near']}, {entry['frac_ctrl']}) != (0.33, 0.75)",
        )
        _assert(
            result["headline_permutation"]["k_targets"] == len(targets),
            "fallback: primary permutation must include the fallback pair",
        )
        sens = result["headline_permutation_matched_step_only"]
        _assert(
            sens is not None and sens["k_targets"] == len(targets) - 1,
            "fallback: matched-step-only sensitivity permutation missing/has wrong k",
        )
        _assert(
            rn["per_pair_bands"][special]["band_status"] in ("above", "within", "indeterminate"),
            "fallback: per-pair band status missing",
        )
        _assert(label != "failed_gate", f"fallback: outcome wrongly failed_gate ({label})")
    if mode == "mixed":
        groups = rn["within_checkpoint_groups"]
        _assert(
            set(groups) == {"0.50", "1.00"},
            f"mixed: expected within-checkpoint groups {{0.50, 1.00}}, got {sorted(groups)}",
        )
        _assert(groups["0.50"]["pairs"] == [special], "mixed: 0.50 group != [special pair]")
        band = rn["per_pair_bands"][special]
        med_050 = float(np.median(rn["gaps_by_frac"]["0.50"]))
        med_100 = float(np.median(rn["gaps_by_frac"]["1.00"]))
        _assert(
            abs(band["median_same_mix_gap_by_frac"]["0.50"] - med_050) < 1e-12,
            "mixed: pair band is not the SAME-checkpoint (0.50) gap median",
        )
        _assert(
            abs(med_050 - med_100) > 1e-6,
            "mixed: gap medians at 0.50 vs 1.00 are equal — the per-checkpoint vs "
            "global-band contrast is not being exercised",
        )
        log.info(
            "[smoke:mixed] per-checkpoint band for %s = %.6f (0.50-frac) vs the retired "
            "global latest-frac band %.6f (1.00-frac) — ratio %.1fx",
            special,
            med_050,
            med_100,
            med_100 / med_050 if med_050 else float("inf"),
        )
    log.info("[smoke:%s] PASS — outcome=%s", mode, label)
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser(description="Task #600 analyze.py synthetic smoke (CPU, VM)")
    ap.add_argument("--mode", choices=(*MODES, "all"), default="all")
    ap.add_argument(
        "--manifest", type=Path, default=Path("eval_results/issue_600/panel_selection.json")
    )
    ap.add_argument("--out-root", type=Path, default=Path("/tmp/i600_analyze_smoke_r2"))
    ap.add_argument(
        "--allow-hf",
        action="store_true",
        help="Allow the bubble-radius HF autofetch to hit the network (default: offline).",
    )
    args = ap.parse_args(argv)
    if not args.allow_hf:
        os.environ["HF_HUB_OFFLINE"] = "1"
    modes = MODES if args.mode == "all" else (args.mode,)
    for mode in modes:
        run_mode(mode, args.manifest, args.out_root)
    log.info("[smoke] ALL MODES PASS: %s", ", ".join(modes))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
