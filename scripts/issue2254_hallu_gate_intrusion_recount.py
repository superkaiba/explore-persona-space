"""#2254 follow-up P1 — hallucination-gate-intrusion-recount (9a-ter zero-GPU).

Question: #2254 demoted `hallucination` at localize gate-2 (best rb@answer
persona-vector delta 50.0 vs the answer-position random-direction
selection-symmetric null-band p97.5 = 65.0), attributed to a noise-dominated
judge. The decisive-stage CJK audit found language-intrusion/degeneracy
dominating high-dose cells. This script recomputes the gate-2 quantities with
intruded rows (a) ZEROED and (b) EXCLUDED to test whether the demotion
attribution flips (intrusion-inflated null band) or holds.

Reuse (verbatim, no new estimator / classifier):
  - intrusion classifier = the CJK regex RECORDED in
    eval_results/issue_2254/decisive/cjk_audit.json (read at runtime); a row
    is intruded iff the regex matches its raw completion text. Fidelity is
    PROVEN by reproducing the audit's own phases.localize.per_arm counts for
    every hallucination arm exactly.
  - reduce helpers `_q_arr` / `_boot_idx` / `_boot_diff_ci` / `_null_band`,
    constants BOOTSTRAP_SEED / N_BOOT_CELL / NULL_STEER, imported from
    scripts/issue2254_preimage.py (the producer of gates.json). Bootstrap
    seed keys are identical to the original reduce (cell_id;
    "hallucination__nullans"), so the orig-regime leg reproduces gates.json
    bit-for-bit as a validation gate before any cleaned recompute.

Rows enter each regime exactly as the original reduce consumed them
(judge-dropped items stay dropped; the stored programmatic coherence_pass
gate is reused unchanged); the ONLY manipulation is the intruded-row
zero/exclude, applied symmetrically to the steered cells AND the alpha-0
reference arm.

Usage:
  uv run python scripts/issue2254_hallu_gate_intrusion_recount.py
"""

from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

# load_dotenv BEFORE any numpy import (shared-VM BLAS thread caps, #847;
# tests/test_shared_vm_thread_caps.py)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

# same-dir import (script mode puts scripts/ at sys.path[0]; keep explicit for
# module-mode callers)
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import issue2254_preimage as pre  # noqa: E402  (verbatim reduce-helper reuse)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval_results" / "issue_2254"
JUDGED_DIR = EVAL_ROOT / "judge" / "localize" / "judged"
RAW_DIR = EVAL_ROOT / "localize" / "raw_completions"
AUDIT_PATH = EVAL_ROOT / "decisive" / "cjk_audit.json"
GATES_PATH = EVAL_ROOT / "localize" / "gates.json"
OUT_PATH = EVAL_ROOT / "localize" / "hallucination_gate_intrusion_recount.json"

BEHAVIOR = "hallucination"
REGIMES = ("orig", "zeroed", "excluded")


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _load_intrusion_flags(judged: dict, raw: dict, rx: re.Pattern) -> dict[str, bool]:
    """Per-item intrusion flag: regex match on the raw completion text.

    Item metadata {qi, seed, ci, di} indexes raw["seeds"][seed]["completions"]
    [ci][di] (ci = context index; q_of_context[ci] == qi on this grid).
    """
    flags: dict[str, bool] = {}
    for item_id, meta in judged["items"].items():
        text = raw["seeds"][str(meta["seed"])]["completions"][meta["ci"]][meta["di"]]
        flags[item_id] = bool(rx.search(text))
    return flags


def _regime_q_arr(judged: dict, flags: dict[str, bool], regime: str) -> np.ndarray:
    """Per-question mean-score vector under a cleaning regime.

    orig     — scored items as-is (must reproduce per_question_mean_score);
    zeroed   — intruded scored items contribute 0.0;
    excluded — intruded items dropped; NaN where nothing remains.
    Judge-dropped items (score None / missing) never enter, matching the
    original reduce.
    """
    nq = int(judged["n_questions"])
    scores = judged["accounting"]["scores"]
    per_q: list[list[float]] = [[] for _ in range(nq)]
    for item_id, meta in judged["items"].items():
        s = scores.get(item_id)
        if s is None:
            continue
        intruded = flags[item_id]
        if regime == "excluded" and intruded:
            continue
        val = 0.0 if (regime == "zeroed" and intruded) else float(s)
        per_q[meta["qi"]].append(val)
    return np.array(
        [float(np.mean(v)) if v else np.nan for v in per_q],
        dtype=np.float64,
    )


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text())
    rx = re.compile(audit["regex"])
    gates = json.loads(GATES_PATH.read_text())
    gate2_stored = gates["behaviors"][BEHAVIOR]["gate2"]

    # ---- load every hallucination localize cell (audit-parity needs all arms)
    a0_judged = json.loads((JUDGED_DIR / f"{BEHAVIOR}__a0.json").read_text())
    steer: list[dict] = []
    for f in sorted(JUDGED_DIR.glob(f"{BEHAVIOR}__*.json")):
        j = json.loads(f.read_text())
        if j["cell"]["kind"] == "steer":
            steer.append(j)
    if not steer:
        raise RuntimeError(f"no steered localize cells for {BEHAVIOR}")

    flags_by_cell: dict[str, dict[str, bool]] = {}
    qarrs: dict[str, dict[str, np.ndarray]] = {}
    for j in [a0_judged, *steer]:
        cid = j["cell_id"]
        raw = json.loads((RAW_DIR / f"{cid}.json").read_text())
        flags = _load_intrusion_flags(j, raw, rx)
        flags_by_cell[cid] = flags
        qarrs[cid] = {r: _regime_q_arr(j, flags, r) for r in REGIMES}
        # validation: orig regime reproduces the stored per-question means
        stored = pre._q_arr(j)
        if not np.allclose(qarrs[cid]["orig"], stored, equal_nan=True):
            raise RuntimeError(f"orig per-question means mismatch stored values for {cid}")

    # ---- classifier-fidelity gate: reproduce the audit's per_arm counts
    arm_counts: dict[str, dict[str, int]] = {}
    for j in [a0_judged, *steer]:
        cid = j["cell_id"]
        parts = cid.split("__")
        arm = f"{BEHAVIOR}__a0__-" if parts[1] == "a0" else "__".join(parts[:3])
        c = arm_counts.setdefault(arm, {"intruded": 0, "total": 0})
        c["intruded"] += sum(flags_by_cell[cid].values())
        c["total"] += len(flags_by_cell[cid])
    audit_arms = {
        k: v
        for k, v in audit["phases"]["localize"]["per_arm"].items()
        if k.startswith(f"{BEHAVIOR}__")
    }
    if arm_counts != audit_arms:
        raise RuntimeError(
            f"classifier does not reproduce the decisive audit per_arm counts:\n"
            f"  recomputed: {arm_counts}\n  audit:      {audit_arms}"
        )

    # ---- gate-2 relevant cells (recipe verbatim: coherence_pass reused)
    rb_ans = [
        j
        for j in steer
        if j["cell"]["direction"] == "rb"
        and j["cell"]["position"] == "answer"
        and j["coherence_pass"]
    ]
    null_ans = [
        j
        for j in steer
        if j["cell"]["position"] == "answer"
        and j["cell"]["direction"] in pre.NULL_STEER["answer"]
        and j["coherence_pass"]
    ]

    results: dict[str, dict] = {}
    for regime in REGIMES:
        a0_q = qarrs[a0_judged["cell_id"]][regime]
        deltas: dict[str, float] = {}
        for j in rb_ans:
            cid = j["cell_id"]
            point, lo, hi = pre._boot_diff_ci(
                qarrs[cid][regime], a0_q, pre._boot_idx(len(a0_q), pre.N_BOOT_CELL, cid)
            )
            deltas[cid] = point
        finite = {k: v for k, v in deltas.items() if np.isfinite(v)}
        best_cell = max(finite, key=finite.get) if finite else None
        band = pre._null_band(
            [qarrs[j["cell_id"]][regime] for j in null_ans], a0_q, f"{BEHAVIOR}__nullans"
        )
        results[regime] = {
            "a0_mean": None if np.isnan(m := np.nanmean(a0_q)) else float(m),
            "a0_n_questions_nonnan": int(np.isfinite(a0_q).sum()),
            "best_rb_answer_delta": None if best_cell is None else float(finite[best_cell]),
            "best_rb_answer_cell": best_cell,
            "n_rb_cells_finite": len(finite),
            "n_rb_cells_total": len(rb_ans),
            "answer_band_p975": None if band is None else band["p975"],
            "answer_band": band,
        }

    # ---- validation gate: orig regime reproduces gates.json exactly
    orig = results["orig"]
    if not (
        np.isclose(orig["best_rb_answer_delta"], gate2_stored["best_rb_answer_delta"])
        and np.isclose(orig["answer_band_p975"], gate2_stored["answer_band_p975"])
    ):
        raise RuntimeError(
            f"orig recompute does not reproduce gates.json gate2: "
            f"{orig['best_rb_answer_delta']}/{orig['answer_band_p975']} vs {gate2_stored}"
        )

    # ---- verdict (per brief): cleaned band < cleaned pos control ->
    # intrusion-inflated-null; else noise-dominated-judge-confirmed.
    # Primary regime = excluded (rows REMOVED); zeroed reported alongside.
    def _verdict(r: dict) -> str | None:
        b, p = r["answer_band_p975"], r["best_rb_answer_delta"]
        if b is None or p is None or not (np.isfinite(b) and np.isfinite(p)):
            return None
        return "intrusion-inflated-null" if b < p else "noise-dominated-judge-confirmed"

    verdict_by_regime = {r: _verdict(results[r]) for r in ("zeroed", "excluded")}
    verdict = verdict_by_regime["excluded"]
    if verdict is None:
        raise RuntimeError("excluded-regime quantities are degenerate; no verdict computable")

    # ---- intrusion rates per relevant cell (+ a0)
    def _cell_rate(j: dict) -> dict:
        f = flags_by_cell[j["cell_id"]]
        return {
            "intruded": int(sum(f.values())),
            "total": len(f),
            "rate": float(sum(f.values()) / len(f)),
            "coherence_pass": bool(j["coherence_pass"]),
            "delta_score_orig": (
                None
                if j["cell"]["kind"] != "steer"
                else float(
                    np.nanmean(qarrs[j["cell_id"]]["orig"])
                    - np.nanmean(qarrs["hallucination__a0"]["orig"])
                )
            ),
        }

    intrusion_rates = {
        "a0": _cell_rate(a0_judged),
        "rb_answer_cells": {j["cell_id"]: _cell_rate(j) for j in rb_ans},
        "random_answer_cells": {j["cell_id"]: _cell_rate(j) for j in null_ans},
        "per_arm_all_localize": arm_counts,
    }

    out = {
        "experiment": "issue2254_hallu_gate_intrusion_recount",
        "behavior": BEHAVIOR,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "script": "scripts/issue2254_hallu_gate_intrusion_recount.py",
        "classifier": {
            "regex": audit["regex"],
            "source": "eval_results/issue_2254/decisive/cjk_audit.json (reused verbatim)",
            "fidelity": "reproduces phases.localize.per_arm counts exactly for all "
            f"{len(audit_arms)} hallucination arms",
        },
        "recipe": {
            "source": "scripts/issue2254_preimage.py::_reduce_wave1 gate2 (helpers imported)",
            "bootstrap_seed": pre.BOOTSTRAP_SEED,
            "n_boot": pre.N_BOOT_CELL,
            "null_steer_answer": list(pre.NULL_STEER["answer"]),
            "band_seed_key": f"{BEHAVIOR}__nullans",
            "coherence_gate": "stored programmatic coherence_pass reused unchanged",
            "a0_cleaning": "regimes applied symmetrically to the alpha-0 reference arm",
        },
        "stored_gate2": gate2_stored,
        "recomputed": {r: results[r] for r in REGIMES},
        "intrusion_rates": intrusion_rates,
        "verdict": verdict,
        "verdict_by_regime": verdict_by_regime,
        "verdict_rule": (
            "cleaned answer_band_p975 < cleaned best_rb_answer_delta -> "
            "intrusion-inflated-null; else noise-dominated-judge-confirmed "
            "(primary regime: excluded)"
        ),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    print(
        json.dumps(
            {
                "stored": gate2_stored,
                "recomputed": {
                    r: {
                        k: results[r][k]
                        for k in ("best_rb_answer_delta", "answer_band_p975", "best_rb_answer_cell")
                    }
                    for r in REGIMES
                },
                "verdict": verdict,
                "verdict_by_regime": verdict_by_regime,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
