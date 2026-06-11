# research code uses Greek letters legitimately
"""Task #480 smoke — synthetic Phase 3 analyzer end-to-end check (CPU-only).

Generates a synthetic per-source `marker_logprob_eval.json` layout with
SHAPE matching real data (24 panel personas, ~50 questions per cell), then
runs `i480_analyze.py` against the real frozen `predictor_comparison.json`
(138 cells) and verifies:
  - the matrix pivots correctly (138 rows after dropping self),
  - h1_h2_analysis.json + final_results.json get written,
  - all 4 figures get rendered,
  - the H1 verdict is one of {supported, falsified, inconclusive}.

This exercises the analyzer's full code path (H1 stats, H2 within-source,
paired Δρ, power-matched, saturation diagnostic, figures) without any
GPU. Run on the local VM:

    uv run python scripts/issue_480/smoke_analyzer_synthetic.py

Exit 0 on success.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke_analyzer_synthetic")

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_PATH = REPO_ROOT / "eval_results/issue_480/_inputs/predictor_comparison.json"
SYCO_SUMMARY_PATH = REPO_ROOT / "eval_results/issue_480/_inputs/syco_411_analyze_summary.json"


def _stable_seed_from_source(source: str) -> int:
    """Deterministic 32-bit seed from a source name (SHA-256, NOT Python ``hash``).

    Python's ``hash(str)`` is salted per-process (``PYTHONHASHSEED=random``), so the
    same source name produced different synthetic data on every smoke invocation;
    that defeated the smoke's reproducibility-vs-CI assertions. SHA-256 of the
    UTF-8 bytes is process-independent.
    """
    import hashlib  # local import so ruff's F401 sweep doesn't strip the top-level form

    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _make_per_source_eval(source: str, panel_keys: list[str], seed: int) -> dict:
    """Synthetic per_panel stats with realistic ranges."""
    rng = random.Random(_stable_seed_from_source(source))
    per_panel: dict[str, dict[str, float]] = {}
    for panel in panel_keys:
        # Make source-on-source spike high (validates the source-cell training
        # success); bystanders draw from a slightly noisy distribution.
        if panel == source:
            marker_delta = rng.uniform(8.0, 12.0)
            log_p_t = rng.uniform(-1.0, -0.3)
            emission = rng.uniform(0.85, 1.0)
        else:
            marker_delta = rng.gauss(2.5, 2.5)
            log_p_t = rng.uniform(-6.0, -3.0)
            emission = rng.uniform(0.05, 0.4)
        # Per-cell marker-Δ SE (over the 50 Q_eval, synthesized). The smoke
        # exercises the noise-tolerant ranking path in i480_analyze.py, which
        # consumes ``marker_delta_se`` if present (else falls back to a
        # std-based pseudo-SE). Spread it deliberately wider for floor-leaning
        # sources so the power-match has a non-trivial degrade signal.
        marker_delta_se = rng.uniform(0.6, 1.8)
        # Four-float storage-contract aggregates (band-stop phase2b), kept
        # self-consistent with the exact identity logp = z_marker - logZ so
        # the analyzer's logp/logit agreement read sees realistic numbers.
        log_p_b = log_p_t - marker_delta
        logz_t = rng.uniform(8.0, 14.0)
        logz_b = rng.uniform(8.0, 14.0)
        z_marker_t = log_p_t + logz_t
        z_marker_b = log_p_b + logz_b
        z_eos_t = rng.uniform(8.0, 13.0)
        z_eos_b = rng.uniform(8.0, 13.0)
        per_panel[panel] = {
            "median_marker_delta": marker_delta,
            "mean_marker_delta": marker_delta,
            "marker_delta_se": marker_delta_se,
            "mean_emission_rate": emission,
            "median_log_p_trained": log_p_t,
            "median_log_p_base": log_p_b,
            "median_z_marker_trained": z_marker_t,
            "median_z_eos_trained": z_eos_t,
            "median_logZ_trained": logz_t,
            "median_z_marker_base": z_marker_b,
            "median_z_eos_base": z_eos_b,
            "median_logZ_base": logz_b,
            "median_eos_margin_delta": (z_marker_t - z_eos_t) - (z_marker_b - z_eos_b),
            "median_delta_z_marker": z_marker_t - z_marker_b,
            "r_trained_len_mean": rng.uniform(80.0, 200.0),
            "r_trained_len_median": rng.uniform(70.0, 180.0),
            "n_q": 50,
        }
    return {
        "source": source,
        "seed": seed,
        "marker_text": " ※",
        "marker_id": 83399,
        "im_end_id": 151645,
        "merged_model_path": f"<synthetic>/{source}_seed{seed}/merged",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "n_panel": len(panel_keys),
        "n_questions": 50,
        "per_panel": per_panel,
        "per_cell_rows": [],
        "git_commit_sha": "synthetic",
        "hostname": "synthetic",
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    if not PREDICTOR_PATH.exists():
        log.error("missing %s — copy from issue-470 first", PREDICTOR_PATH)
        return 1

    # Read real predictor_comparison to extract the panel = list of bystander
    # personas (so the synthetic per_panel keys cover the join).
    with open(PREDICTOR_PATH) as f:
        pred = json.load(f)
    by_source: dict[str, set[str]] = {}
    for cell in pred["cells"]:
        by_source.setdefault(cell["source"], set()).add(cell["bystander"])

    sources = sorted(by_source.keys())
    log.info(
        "real predictor: %d sources, panel sizes %s",
        len(sources),
        {s: len(by_source[s]) for s in sources},
    )

    with tempfile.TemporaryDirectory(prefix="i480_smoke_") as td:
        slab_root = Path(td) / "slab"
        figures_dir = Path(td) / "figures"
        seed = 42
        # Write synthetic per_source/<src>/seed_<S>/marker_logprob_eval.json
        for source in sources:
            panel_keys = sorted(by_source[source] | {source})  # include source-on-source
            payload = _make_per_source_eval(source, panel_keys, seed)
            out_dir = slab_root / "per_source" / source / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "marker_logprob_eval.json", "w") as f:
                json.dump(payload, f)

        # Synthetic per-source trajectory JSONs at the band-callback output
        # location (slab_root/trajectories) so the --trajectory-dir wiring +
        # the NON-placeholder trajectory-figure branch are exercised.
        traj_dir = slab_root / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        for source in sources:
            rng = random.Random(_stable_seed_from_source(source) + 1)
            steps = list(range(5, 530, 5))
            base = rng.uniform(-21.0, -17.0)
            log_p = [min(-0.2, base + 0.04 * s + rng.gauss(0, 0.1)) for s in steps]
            with open(traj_dir / f"{source}_seed{seed}_trajectory.json", "w") as f:
                json.dump(
                    {
                        "schema": "marker_band_trajectory_v1",
                        "steps": steps,
                        "log_p_marker": log_p,
                        "log_p_base": [base] * len(steps),
                        "delta_nats": [lp - base for lp in log_p],
                    },
                    f,
                )

        sentinel = Path(td) / "sentinel.json"
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/issue_480/i480_analyze.py",
            "--slab-root",
            str(slab_root),
            "--seed",
            str(seed),
            "--predictor-comparison",
            str(PREDICTOR_PATH),
            "--syco-summary",
            str(SYCO_SUMMARY_PATH),
            "--figures-dir",
            str(figures_dir),
            "--trajectory-dir",
            str(traj_dir),
            "--sentinel-path",
            str(sentinel),
        ]
        log.info("Running analyzer: %s", " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode
        if rc != 0:
            log.error("analyzer exited %d", rc)
            return rc

        if not sentinel.exists():
            log.error("sentinel %s missing after analyzer", sentinel)
            return 1
        with open(sentinel) as f:
            sent = json.load(f)
        log.info("sentinel headline: %s", sent.get("headline_numbers"))

        h1_path = slab_root / "h1_h2_analysis.json"
        with open(h1_path) as f:
            h1 = json.load(f)
        n_joined = h1["n_cells_joined"]
        verdict = h1["h1"]["verdict"]
        h2_paired = h1["h2_paired_delta_rho"]
        power_matched = h1["h2_power_matched_paired_delta_rho"]
        log.info("n_joined=%d (expect 138)", n_joined)
        log.info("H1 verdict=%s", verdict)
        log.info("H2 paired Δρ mean=%.3f n=%d", h2_paired["mean_delta_rho"], h2_paired["n_sources"])
        log.info(
            "H2 power-matched Δρ mean=%.3f licensed=%s",
            power_matched["mean_delta_rho"],
            power_matched["behavior_type_headline_licensed"],
        )
        assert n_joined == 138, f"expected 138 joined rows, got {n_joined}"
        assert verdict in {
            "supported",
            "shared_context_length_nuisance",
            "falsified",
            "inconclusive",
        }, f"bad verdict {verdict}"
        for fig in (
            "h1_hero_marker_vs_sycophancy",
            "marker_delta_distribution",
            "h2_per_source_cosine_gradient",
            "h2_paired_rho_vs_411",
            "h1_source_fe_residualized",
            "source_logprob_trajectory",
        ):
            assert fig in h1["figures"], f"missing figure {fig}"
            assert Path(h1["figures"][fig]).exists(), f"figure file missing {fig}"
        # B1 self-check: noise-tolerant power-match is NOT a no-op on a synthetic
        # floored-vs-non-floored regime.
        pm_self_check = h1["h2_power_match_self_check"]
        log.info(
            "power-match self-check: marker_drift=%.3f syco_drift=%.3f non_noop=%s",
            pm_self_check["marker_rho_shift_abs"],
            pm_self_check["syco_rho_shift_abs"],
            pm_self_check["power_match_is_non_noop"],
        )
        assert pm_self_check["power_match_is_non_noop"], (
            f"power-match self-check FAILED: {pm_self_check}"
        )

        # Band-stop additions: (a) matrix rows carry the four-float fields,
        # (b) the logp/logit agreement summary is computed, (c) the
        # concordance package (PRIMARY emission_rate + SECONDARY
        # marker_delta) runs end-to-end on the synthetic 138-cell matrix.
        matrix_path = slab_root / "marker_delta_matrix.json"
        with open(matrix_path) as f:
            matrix = json.load(f)
        four_float_cols = (
            "z_marker_trained",
            "z_eos_trained",
            "logZ_trained",
            "z_marker_base",
            "z_eos_base",
            "logZ_base",
            "eos_margin_delta",
            "delta_z_marker",
        )
        for col in four_float_cols:
            assert col in matrix["rows"][0], f"matrix row missing four-float col {col}"
        agreement = h1["logp_logit_agreement"]
        assert agreement["available"] is True, agreement
        assert len(agreement["per_source"]) == len(sources), agreement["per_source"].keys()
        log.info("logp/logit agreement per-source keys OK (%d sources)", len(sources))

        smoke_fig_subdir = "issue_480/_smoke_tmp"
        try:
            for x_field in ("emission_rate", "marker_delta"):
                suffix = "" if x_field == "emission_rate" else f"_{x_field}"
                conc_cmd = [
                    "uv",
                    "run",
                    "python",
                    "scripts/issue480_emission_rate_concordance.py",
                    "--matrix-path",
                    str(matrix_path),
                    "--stats-dir",
                    str(slab_root),
                    "--figure-stem",
                    f"{smoke_fig_subdir}/emission_rate_vs_sycophancy{suffix}",
                    "--x-field",
                    x_field,
                ]
                log.info("Running concordance (%s): %s", x_field, " ".join(conc_cmd))
                rc = subprocess.run(conc_cmd, cwd=str(REPO_ROOT), check=False).returncode
                assert rc == 0, f"concordance ({x_field}) exited {rc}"
                stats_name = (
                    "concordance_stats.json"
                    if x_field == "emission_rate"
                    else f"concordance_stats_{x_field}.json"
                )
                with open(slab_root / stats_name) as f:
                    conc = json.load(f)
                assert conc["dv"]["x"].startswith(x_field), conc["dv"]
                assert len(conc["per_source"]) == len(sources)
                log.info("concordance (%s) OK -> %s", x_field, stats_name)
        finally:
            import shutil

            shutil.rmtree(REPO_ROOT / "figures" / "issue_480" / "_smoke_tmp", ignore_errors=True)

        log.info("OK — analyzer + concordance end-to-end pass on synthetic data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
