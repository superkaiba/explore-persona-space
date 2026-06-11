# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, Δ) in docstrings.
"""Issue #533 bare-word follow-up — non-empty fixture generator for ``i533_bw_analyze``.

Reviewer round-1 blocker (Codex `smoke-run-missing`): the analyzer smoke
must run end-to-end on a NON-EMPTY fixture so a future label / schema
rename can't silently leave the analyzer happy. This script generates
exactly that fixture under

    eval_results/issue_533/bare_word_install_step_grid/{cross_eval,logit_capture}/per_cell/

using the SAME label-format and schema-shape helpers the real writers
use (``i464_po_eval._po_cell_label``, ``_eval_encodings_for_cell``,
``GRID_SUFFIX_CHAR_FOR``), so any future divergence will surface at the
fixture step before reaching the analyzer.

Slice: 1 persona (pirate) × 1 seed (42) × 1 max_steps (18) × 2 arms
(``system_minimal``, ``role_bare``) × 3 eval encodings (own / wrong /
default_assistant) = 6 cross-eval rows + 2 trained logit-capture rows +
2 base logit-capture rows (one per wrong-persona eval encoding).

The deterministic-fixture knobs:
  * ``g_logps = b_logps + 1.0 + arm_idx``: system_minimal arm_idx=0 →
    Δlog P = +1.0; role_bare arm_idx=1 → Δlog P = +2.0. Paired d
    (sys − role) = −1.0 (the expected analyzer point estimate).
  * own-encoding argmax-emit rate = 1.0 in both arms → install_gate_pass
    must be True at the covered (persona=pirate, max_steps=18) row.

Usage::

    uv run python scripts/i533_bw_analyze_smoke_fixture.py
    uv run python scripts/i533_bw_analyze.py --allow-partial
    # Expect:
    #   * exit 0
    #   * paired_logp_sys_minus_role.point == -1.0 at (pirate, s=18)
    #   * paired_margin_sys_minus_role.point is finite (non-NaN)
    #   * install_gate_pass=True at (pirate, s=18)

Run by hand for ad-hoc analyzer smokes; committed so the smoke evidence
in the implementation report is reproducible from a clean checkout.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from i464_po_eval import (  # type: ignore[import-not-found]  # noqa: E402
    GRID_SUFFIX_CHAR_FOR,
    _eval_encodings_for_cell,
    _po_cell_label,
)

# Fixture parameters.
SUFFIX = GRID_SUFFIX_CHAR_FOR["bw_i533"]
STEPS = 18
SEED = 42
PERSONA = "pirate"
ARMS = ("system_minimal", "role_bare")
N_Q = 5

OUT_BASE = PROJECT_ROOT / "eval_results" / "issue_533" / "bare_word_install_step_grid"
CROSS_EVAL_DIR = OUT_BASE / "cross_eval" / "per_cell"
LOGIT_DIR = OUT_BASE / "logit_capture" / "per_cell"


def _det_floats(key: object, lo: float, hi: float, n: int = N_Q) -> list[float]:
    """Hash ``key`` → deterministic seed → N uniform draws in [lo, hi]."""
    h = hashlib.sha256(str(key).encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "little"))
    return [rng.uniform(lo, hi) for _ in range(n)]


def write_fixture() -> dict:
    """Write the fixture; return a small summary dict."""
    CROSS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOGIT_DIR.mkdir(parents=True, exist_ok=True)

    crosseval_files: list[str] = []
    logit_base_files: list[str] = []
    logit_trained_files: list[str] = []

    # Cross-eval per-cell rows.
    for arm_idx, arm in enumerate(ARMS):
        cell = _po_cell_label(arm, SEED, PERSONA, STEPS, SUFFIX)
        encs = _eval_encodings_for_cell(arm, PERSONA)
        for ee_idx, e_eval in enumerate(encs):
            b_logps = _det_floats(("b", arm, ee_idx), -10, -2)
            # Deliberate offset: g = b + 1.0 + arm_idx so the analyzer's
            # paired d = sys − role exits at -1.0 (testable invariant).
            g_logps = [x + 1.0 + arm_idx for x in b_logps]
            # Own-encoding (index 0) hits emit=1.0; others hit 0.
            g_argmax = [1] * N_Q if ee_idx == 0 else [0] * N_Q
            payload = {
                "cell": cell,
                "arm": arm,
                "seed": SEED,
                "training_persona": PERSONA,
                "marker_persona": "pirate",
                "e_eval": e_eval,
                "marker_id": 83399,
                "n_probes": N_Q,
                "g_logprob": sum(g_logps) / N_Q,
                "b_logprob": sum(b_logps) / N_Q,
                "delta_g": (sum(g_logps) - sum(b_logps)) / N_Q,
                "emission_recompute_rate": sum(g_argmax) / N_Q,
                "logp_floor": -50.0,
                "g_logps_per_q": g_logps,
                "b_logps_per_q": b_logps,
                "g_argmax_marker_per_q": g_argmax,
                "b_argmax_marker_per_q": [0] * N_Q,
                "max_steps": STEPS,
                "variant": "bw_i533",
            }
            out_path = CROSS_EVAL_DIR / f"{cell}__{e_eval}.json"
            out_path.write_text(json.dumps(payload))
            crosseval_files.append(out_path.name)

    # Base-side logit captures (one per wrong-persona eval encoding).
    all_wrong = {_eval_encodings_for_cell(arm, PERSONA)[1] for arm in ARMS}
    for e_eval in sorted(all_wrong):
        z_m = _det_floats(("zm-b", e_eval), -2, 4)
        z_e = _det_floats(("ze-b", e_eval), -2, 6)
        logZ = [9.0] * N_Q
        logp = [zm - lz for zm, lz in zip(z_m, logZ, strict=True)]
        payload = {
            "schema_version": "i533_bw_logit_capture_v1",
            "variant": "bw_i533",
            "side": "base",
            "e_eval": e_eval,
            "marker_persona": "pirate",
            "marker_id": 83399,
            "stats": {"logp": logp, "z_marker": z_m, "z_eos": z_e, "logZ": logZ},
        }
        out_path = LOGIT_DIR / f"base__{e_eval}__marker_pirate.json"
        out_path.write_text(json.dumps(payload))
        logit_base_files.append(out_path.name)

    # Trained-side logit captures (one per cell x wrong encoding) — the
    # filename MUST come from _po_cell_label so a writer rename surfaces
    # here, not at the end of a 6-GPU-h pipeline.
    for arm in ARMS:
        cell = _po_cell_label(arm, SEED, PERSONA, STEPS, SUFFIX)
        encs = _eval_encodings_for_cell(arm, PERSONA)
        wrong_e = encs[1]
        z_m = _det_floats(("zm-t", arm, SEED), -1, 5)
        z_e = _det_floats(("ze-t", arm, SEED), -3, 6)
        logZ = [10.0] * N_Q
        logp = [zm - lz for zm, lz in zip(z_m, logZ, strict=True)]
        payload = {
            "schema_version": "i533_bw_logit_capture_v1",
            "variant": "bw_i533",
            "side": "trained",
            "cell": cell,
            "arm": arm,
            "seed": SEED,
            "training_persona": PERSONA,
            "max_steps": STEPS,
            "e_eval": wrong_e,
            "marker_persona": "pirate",
            "marker_id": 83399,
            "gauge_assert": {"ok": True, "target_modules": ["q_proj"]},
            "trained": {"logp": logp, "z_marker": z_m, "z_eos": z_e, "logZ": logZ},
            # Placeholder (analyzer reads base from the ``base__*`` JSON,
            # not from this row's ``base`` key).
            "base": {"logp": logp, "z_marker": z_m, "z_eos": z_e, "logZ": logZ},
            "delta_mean": {"logp": 0.0, "z_marker": 0.0, "eos_margin": 0.0},
        }
        out_path = LOGIT_DIR / f"{cell}__{wrong_e}__marker_pirate.json"
        out_path.write_text(json.dumps(payload))
        logit_trained_files.append(out_path.name)

    return {
        "crosseval_files": crosseval_files,
        "logit_base_files": logit_base_files,
        "logit_trained_files": logit_trained_files,
        "n_files_total": (len(crosseval_files) + len(logit_base_files) + len(logit_trained_files)),
    }


if __name__ == "__main__":
    summary = write_fixture()
    print(f"Fixture written: {summary['n_files_total']} files")
    print(f"  cross_eval rows: {len(summary['crosseval_files'])}")
    print(f"  logit base rows: {len(summary['logit_base_files'])}")
    print(f"  logit trained rows: {len(summary['logit_trained_files'])}")
    print()
    print("Next: uv run python scripts/i533_bw_analyze.py --allow-partial")
    print("Expect: paired_logp_sys_minus_role.point == -1.0 at (pirate, s=18)")
