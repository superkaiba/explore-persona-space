# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, Δ, log Z, − minus) in scientific docstrings + logs.
"""Issue #533 bare-word follow-up — paired role_bare vs system_minimal analysis.

Reads the bw_i533 cross-eval + four-float logit-capture outputs and computes
the headline contrast:

    paired d = (read under system_minimal arm) − (read under role_bare arm)

per persona, per max_steps, AT THE WRONG-PERSONA PROBE — in BOTH:
  * log P(marker) trained − base   (parent's behavioral DV)
  * trained − base EOS-margin = Δ(z_marker − z_eos)  (mechanistic readout)

Per-seed-paired bootstrap N=10,000 with 95% CI (same shape as i547 / i533
analyses). An "install gate" is also computed: own-encoding argmax-emit
rate >= 0.5 in BOTH arms at the grid point (the spec's gating rule).

Inputs:
    eval_results/issue_533/bare_word_install_step_grid/cross_eval/
        per_cell/{arm}_seed{S}_cn_{persona}_s{steps}__{e_eval}.json  (vLLM)
    eval_results/issue_533/bare_word_install_step_grid/logit_capture/
        per_cell/{arm}_seed{S}_cn_{persona}_s{steps}__{e_eval}__marker_pirate.json
            (HF four-float capture; trained side)
        per_cell/base__{e_eval}__marker_pirate.json
            (HF four-float capture; base side, shared across cells)

Outputs (single atomic JSON write):
    eval_results/issue_533/bare_word_install_step_grid/analysis.json

CLI::

    uv run python scripts/i533_bw_analyze.py
    uv run python scripts/i533_bw_analyze.py --allow-partial   # smoke / dispatch-dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from i464_po_eval import MAX_STEPS_BW_I533, SEEDS_FOR  # noqa: E402  type: ignore[import-not-found]

from explore_persona_space.experiments import i464_encodings as enc  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("i533_bw_analyze")

OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_533" / "bare_word_install_step_grid"
CROSS_EVAL_PER_CELL = OUT_DIR / "cross_eval" / "per_cell"
LOGIT_PER_CELL = OUT_DIR / "logit_capture" / "per_cell"
ANALYSIS_OUT_PATH = OUT_DIR / "analysis.json"

# Mirrors i464_po_eval.SHARED_MARKER_PERSONA — the cn run trains and
# probes only the shared pirate marker ` ※`.
SHARED_MARKER_PERSONA = "pirate"
SEEDS = SEEDS_FOR["bw_i533"]
ARMS = enc.MINIMAL_ARMS  # ("system_minimal", "role_bare")
PERSONAS = enc.PERSONAS  # ("pirate", "villain")


def _cell_label(arm: str, seed: int, persona: str, steps: int) -> str:
    """Mirror of i464_po_eval._po_cell_label for the bw_i533 variant."""
    return f"{arm}_seed{seed}_cn_{persona}_s{steps}"


def _own_encoding(arm: str, persona: str) -> str:
    """Own (diagonal) eval encoding for a cell in the same arm-family."""
    if arm == "system_minimal":
        return f"system_minimal_{persona}"
    if arm == "role_bare":
        return f"role_bare_{persona}"
    raise ValueError(f"unknown arm={arm!r}")


def _wrong_encoding(arm: str, persona: str) -> str:
    """Wrong-persona (off-diagonal) eval encoding, same arm-family."""
    other = "villain" if persona == "pirate" else "pirate"
    return _own_encoding(arm, other)


def _read_crosseval_row(arm: str, seed: int, persona: str, steps: int, e_eval: str) -> dict | None:
    """Load one per-cell vLLM JSON; return None if missing."""
    p = CROSS_EVAL_PER_CELL / f"{_cell_label(arm, seed, persona, steps)}__{e_eval}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    return json.loads(p.read_text())


def _read_logit_trained(arm: str, seed: int, persona: str, steps: int, e_eval: str) -> dict | None:
    """Load one per-cell four-float TRAINED JSON; return None if missing."""
    name = (
        f"{_cell_label(arm, seed, persona, steps)}__{e_eval}__marker_{SHARED_MARKER_PERSONA}.json"
    )
    p = LOGIT_PER_CELL / name
    if not p.exists() or p.stat().st_size == 0:
        return None
    return json.loads(p.read_text())


def _read_logit_base(e_eval: str) -> dict | None:
    """Load shared base-side four-float JSON; return None if missing."""
    p = LOGIT_PER_CELL / f"base__{e_eval}__marker_{SHARED_MARKER_PERSONA}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    return json.loads(p.read_text())


def _per_q_delta_logp_from_crosseval(row: dict) -> list[float]:
    """Per-question Δ log P(marker) trained − base, read from the vLLM cross-eval row."""
    g = np.asarray(row["g_logps_per_q"], dtype=float)
    b = np.asarray(row["b_logps_per_q"], dtype=float)
    return list(g - b)


def _per_q_delta_margin_from_logit(trained: dict, base: dict) -> list[float]:
    """Per-question Δ(z_marker − z_eos) trained − base from the HF logit capture."""
    g_zm = np.asarray(trained["trained"]["z_marker"], dtype=float)
    g_ze = np.asarray(trained["trained"]["z_eos"], dtype=float)
    b_zm = np.asarray(base["stats"]["z_marker"], dtype=float)
    b_ze = np.asarray(base["stats"]["z_eos"], dtype=float)
    if not (len(g_zm) == len(g_ze) == len(b_zm) == len(b_ze)):
        raise RuntimeError(
            f"length mismatch in margin capture: g_zm={len(g_zm)} g_ze={len(g_ze)} "
            f"b_zm={len(b_zm)} b_ze={len(b_ze)}"
        )
    return list((g_zm - g_ze) - (b_zm - b_ze))


def _paired_bootstrap(
    per_seed_a: dict[int, list[float]],
    per_seed_b: dict[int, list[float]],
    n_boot: int = 10_000,
    rng_seed: int = 42,
) -> dict:
    """Per-seed-paired bootstrap of (a − b) over a shared seed set.

    For each shared seed, take the mean across questions and arm A vs
    arm B (same seed, same persona, same step). Then bootstrap-sample
    seeds with replacement N_boot times.

    Returns the point estimate (mean of paired-seed-mean differences)
    + 95% CI + n_seeds.
    """
    shared_seeds = sorted(set(per_seed_a.keys()) & set(per_seed_b.keys()))
    if not shared_seeds:
        return {
            "point": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_seeds": 0,
            "shared_seeds": [],
        }
    rng = np.random.default_rng(rng_seed)
    a_means = np.array([np.mean(per_seed_a[s]) for s in shared_seeds])
    b_means = np.array([np.mean(per_seed_b[s]) for s in shared_seeds])
    diff = a_means - b_means
    point = float(diff.mean())
    n = len(shared_seeds)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(diff[idx].mean())
    return {
        "point": point,
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n_seeds": n,
        "shared_seeds": shared_seeds,
    }


def _emission_rate(row: dict) -> float | None:
    """Own-encoding emission rate: argmax==marker (vLLM per-q diagnostic)."""
    if "g_argmax_marker_per_q" not in row:
        return None
    flags = row["g_argmax_marker_per_q"]
    if not flags:
        return None
    return float(sum(flags) / len(flags))


def analyze(allow_partial: bool = False) -> dict:  # noqa: C901 - per-(arm, persona, steps, seed) iteration + reads branches
    """Run the headline + install-gate analysis; write a single JSON."""
    log.info("[phase=analyze] reading cross-eval + logit-capture per-cell payloads")

    # Per (arm, persona, steps, seed): wrong-persona delta (log P + margin)
    # and own-encoding emission rate.
    wrong_logp: dict[tuple[str, str, int], dict[int, list[float]]] = {}
    wrong_margin: dict[tuple[str, str, int], dict[int, list[float]]] = {}
    own_emit: dict[tuple[str, str, int], dict[int, float]] = {}

    missing_crosseval: list[str] = []
    missing_logit_trained: list[str] = []
    missing_logit_base: list[str] = []
    base_cache: dict[str, dict] = {}

    for arm in ARMS:
        for persona in PERSONAS:
            for steps in MAX_STEPS_BW_I533:
                for seed in SEEDS:
                    wrong_e = _wrong_encoding(arm, persona)
                    own_e = _own_encoding(arm, persona)
                    # WRONG-encoding cross-eval row (gives Δlog P per q).
                    wrong_row = _read_crosseval_row(arm, seed, persona, steps, wrong_e)
                    if wrong_row is None:
                        missing_crosseval.append(
                            f"{_cell_label(arm, seed, persona, steps)}__{wrong_e}"
                        )
                    else:
                        key = (arm, persona, steps)
                        wrong_logp.setdefault(key, {})[seed] = _per_q_delta_logp_from_crosseval(
                            wrong_row
                        )

                    # WRONG-encoding logit capture (gives Δ margin per q).
                    trained = _read_logit_trained(arm, seed, persona, steps, wrong_e)
                    if trained is None:
                        missing_logit_trained.append(
                            f"{_cell_label(arm, seed, persona, steps)}__{wrong_e}"
                        )
                    else:
                        if wrong_e not in base_cache:
                            b = _read_logit_base(wrong_e)
                            if b is None:
                                missing_logit_base.append(wrong_e)
                                base_cache[wrong_e] = {}  # poison so we skip
                            else:
                                base_cache[wrong_e] = b
                        b_payload = base_cache[wrong_e]
                        if b_payload:
                            key = (arm, persona, steps)
                            wrong_margin.setdefault(key, {})[seed] = _per_q_delta_margin_from_logit(
                                trained, b_payload
                            )

                    # OWN-encoding cross-eval row (gives emission rate).
                    own_row = _read_crosseval_row(arm, seed, persona, steps, own_e)
                    if own_row is not None:
                        rate = _emission_rate(own_row)
                        if rate is not None:
                            key = (arm, persona, steps)
                            own_emit.setdefault(key, {})[seed] = rate

    # Strict completeness gate unless --allow-partial.
    if not allow_partial:
        if missing_crosseval:
            raise RuntimeError(
                f"[analyze] {len(missing_crosseval)} cross-eval rows missing "
                f"(first 5): {missing_crosseval[:5]}"
            )
        if missing_logit_trained:
            raise RuntimeError(
                f"[analyze] {len(missing_logit_trained)} logit-capture trained rows "
                f"missing (first 5): {missing_logit_trained[:5]}"
            )
        if missing_logit_base:
            raise RuntimeError(
                f"[analyze] missing base-side logit captures for encodings: {missing_logit_base}"
            )

    # Headline: paired d = system_minimal − role_bare at (persona, steps),
    # per-seed-paired bootstrap, in BOTH log P and margin space.
    paired_results: list[dict] = []
    for persona in PERSONAS:
        for steps in MAX_STEPS_BW_I533:
            sys_logp = wrong_logp.get(("system_minimal", persona, steps), {})
            role_logp = wrong_logp.get(("role_bare", persona, steps), {})
            sys_margin = wrong_margin.get(("system_minimal", persona, steps), {})
            role_margin = wrong_margin.get(("role_bare", persona, steps), {})
            sys_emit = own_emit.get(("system_minimal", persona, steps), {})
            role_emit = own_emit.get(("role_bare", persona, steps), {})

            # Install gate: own-encoding argmax-emit >= 0.5 IN BOTH arms.
            # We require the mean across seeds to clear the threshold.
            sys_emit_mean = float(np.mean(list(sys_emit.values()))) if sys_emit else float("nan")
            role_emit_mean = float(np.mean(list(role_emit.values()))) if role_emit else float("nan")
            install_gate_pass = (not np.isnan(sys_emit_mean) and sys_emit_mean >= 0.5) and (
                not np.isnan(role_emit_mean) and role_emit_mean >= 0.5
            )

            entry = {
                "persona": persona,
                "max_steps": steps,
                "n_seeds_logp_sys": len(sys_logp),
                "n_seeds_logp_role": len(role_logp),
                "n_seeds_margin_sys": len(sys_margin),
                "n_seeds_margin_role": len(role_margin),
                "own_emit_rate_sys_mean": sys_emit_mean,
                "own_emit_rate_role_mean": role_emit_mean,
                "install_gate_pass": bool(install_gate_pass),
                "paired_logp_sys_minus_role": _paired_bootstrap(sys_logp, role_logp),
                "paired_margin_sys_minus_role": _paired_bootstrap(sys_margin, role_margin),
            }
            paired_results.append(entry)

    out = {
        "schema_version": "i533_bw_analysis_v1",
        "task_id": 533,
        "followup_label": "bare-word-install-step-grid",
        "variant": "bw_i533",
        "arms": list(ARMS),
        "personas": list(PERSONAS),
        "max_steps_grid": list(MAX_STEPS_BW_I533),
        "seeds": list(SEEDS),
        "n_paired_rows": len(paired_results),
        "paired_results": paired_results,
        "missing_crosseval": missing_crosseval,
        "missing_logit_trained": missing_logit_trained,
        "missing_logit_base": missing_logit_base,
        "analyzed_at_utc": datetime.now(UTC).isoformat(),
        "produced_by": "scripts/i533_bw_analyze.py",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ANALYSIS_OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(ANALYSIS_OUT_PATH)
    log.info("[phase=analyze] wrote %s", ANALYSIS_OUT_PATH)

    # Brief summary to stdout for log-watch.
    log.info(
        "[phase=analyze] %d paired rows; %d cross-eval missing, %d logit-trained missing, "
        "%d logit-base missing",
        len(paired_results),
        len(missing_crosseval),
        len(missing_logit_trained),
        len(missing_logit_base),
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Tolerate missing cross-eval / logit-capture rows (smoke / "
            "dispatch-dry-run). Production analysis requires the full grid "
            "and fails loud on any miss."
        ),
    )
    args = ap.parse_args(argv)
    analyze(allow_partial=args.allow_partial)
    return 0


if __name__ == "__main__":
    sys.exit(main())
