"""Issue #529 — anchor selection over the {1,2,3,5} epoch grid.

Reads per-cell JSONs produced by
``scripts/i464_po_eval.py --variant cn_i529`` and selects ONE anchor
``E*[persona]`` per persona via the §4.5 algorithm.

Pipeline (mirrors plan §4.5 step-by-step):

  1. Per ``(E, persona)`` across all 3 arms x 5 seeds = 15 cells:
       * ``own_logp``  = mean over (arm, seed) of g_logprob[' ※', e_own]
                        (diagonal, source emission)
       * ``own_argmax_emit`` = mean over (arm, seed) of
                              g_argmax_marker_per_q_rate (own slot)
     Per ``(E, persona, arm)``:
       * ``wrong_logp`` = mean over seeds of g_logprob[' ※', e_other]
       * ``wrong_sd``   = sd over (seeds x 50 questions) of
                          g_logprob[' ※', e_other]
  2. Resolution gate per (E, persona, arm):
       wrong_sd > 0.5 nats AND wrong_logp ∈ [-10, -5]
  3. Source-installation gate per (E, persona):
       own_argmax_emit ≥ 0.50
  4. Candidate anchors: all E that satisfy (3) AND satisfy (2) for ALL
     3 arms at the given persona.
  5. Tie-break: prefer the SMALLEST E that satisfies both.
  6. Final selection: one E* per persona, written to
     ``eval_results/issue_529/anchor_selection.json``.
  7. Degenerate case: if NO E satisfies both gates for either persona,
     write ``degenerate: true`` + the trajectory diagnostics + the
     "drop r/lr in a follow-up" recommendation.

CLI:
    uv run python scripts/i529_select_anchor.py
    uv run python scripts/i529_select_anchor.py --allow-partial
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("i529.select_anchor")

PER_CELL_DIR = Path("eval_results/issue_529/contrastive_negatives/cross_eval/per_cell")
OUT_PATH = Path("eval_results/issue_529/anchor_selection.json")
SCHEMA_VERSION = "i529_anchor_v1"

EPOCHS = (1, 2, 3, 5)
# Grid-suffix character spliced into the cell label: "e" (#529/#533
# epoch-indexed cells `_e{E}`) or "s" (#547 max_steps-indexed cells
# `_s{S}`). Rebound from --suffix-char in main(); the default preserves
# #529/#533 behavior byte-for-byte.
SUFFIX_CHAR = "e"
SEEDS = (42, 137, 1337, 7, 21)
ARMS = ("system_plain", "system_padded", "role")
PERSONAS = ("pirate", "villain")

# Plan §4.5 thresholds.
WRONG_SD_THRESHOLD = 0.5
WRONG_LOGP_BAND = (-10.0, -5.0)
OWN_EMIT_GATE = 0.50


def _git_commit_hash() -> str:
    """Return the current HEAD sha or 'unknown' if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            env={**os.environ},
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _cell_label(arm: str, seed: int, persona: str, epoch: int) -> str:
    """Match ``i464_po_eval._po_cell_label`` for the cn_i529/cn_i533/cn_i547 paths.

    ``epoch`` carries the GRID VALUE (an epoch count for #529/#533, a
    max_steps count for #547); ``SUFFIX_CHAR`` selects the label suffix
    (``_e{E}`` vs ``_s{S}``).
    """
    return f"{arm}_seed{seed}_cn_{persona}_{SUFFIX_CHAR}{epoch}"


def _other_eval_encoding_for(arm: str, persona: str) -> str:
    """Off-diagonal encoding (same-arm-family, OTHER persona)."""
    other = "villain" if persona == "pirate" else "pirate"
    if arm == "role":
        return f"role_{other}"
    return f"system_{other}"


def _own_eval_encoding_for(arm: str, persona: str) -> str:
    """Diagonal encoding (same-arm-family, OWN persona)."""
    if arm == "role":
        return f"role_{persona}"
    return f"system_{persona}"


def _load_per_cell(arm: str, seed: int, persona: str, epoch: int, e_eval: str) -> dict | None:
    """Read one per-cell JSON or return None if missing."""
    p = PER_CELL_DIR / f"{_cell_label(arm, seed, persona, epoch)}__{e_eval}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    return json.loads(p.read_text())


def _per_e_per_persona_diagnostics(
    allow_partial: bool,
) -> tuple[dict[str, dict[int, dict[str, Any]]], list[str]]:
    """Compute the §4.5 step-1 diagnostics for every (E, persona, arm).

    Returns ``(diagnostics, missing_cell_labels)`` where ``diagnostics``
    nests as ``{persona: {E: {own_logp, own_argmax_emit, per_arm: {arm:
    {wrong_logp, wrong_sd, n_questions}}}}}``.
    """
    missing: list[str] = []
    diag: dict[str, dict[int, dict[str, Any]]] = {p: {} for p in PERSONAS}
    for persona in PERSONAS:
        for epoch in EPOCHS:
            own_logps: list[float] = []
            own_emits: list[float] = []
            per_arm: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                wrong_per_q: list[float] = []
                wrong_means: list[float] = []
                for seed in SEEDS:
                    # diagonal (own)
                    e_own = _own_eval_encoding_for(arm, persona)
                    own = _load_per_cell(arm, seed, persona, epoch, e_own)
                    if own is None:
                        missing.append(f"{_cell_label(arm, seed, persona, epoch)}/{e_own}")
                        if not allow_partial:
                            continue
                    else:
                        own_logps.append(float(own["g_logprob"]))
                        emit = own.get("emission_recompute_rate")
                        if emit is None:
                            # Fallback: derive from per-q argmax array.
                            argmax = own.get("g_argmax_marker_per_q", [])
                            emit = float(sum(argmax)) / len(argmax) if argmax else 0.0
                        own_emits.append(float(emit))
                    # off-diagonal (wrong)
                    e_off = _other_eval_encoding_for(arm, persona)
                    wrong = _load_per_cell(arm, seed, persona, epoch, e_off)
                    if wrong is None:
                        missing.append(f"{_cell_label(arm, seed, persona, epoch)}/{e_off}")
                        if not allow_partial:
                            continue
                    else:
                        wrong_means.append(float(wrong["g_logprob"]))
                        per_q = wrong.get("g_logps_per_q", [])
                        wrong_per_q.extend(float(x) for x in per_q)
                # Per-arm sd uses the per-question raw log-probs to
                # match the dynamic-range gate's grain (matches
                # i464_po_analyze.DYNAMIC_RANGE_THRESHOLD which is
                # computed on the same per-question array).
                sd = statistics.pstdev(wrong_per_q) if wrong_per_q else float("nan")
                per_arm[arm] = {
                    "wrong_logp_mean": float(sum(wrong_means) / len(wrong_means))
                    if wrong_means
                    else float("nan"),
                    "wrong_sd": float(sd),
                    "n_questions": len(wrong_per_q),
                }
            diag[persona][epoch] = {
                "own_logp": float(sum(own_logps) / len(own_logps)) if own_logps else float("nan"),
                "own_argmax_emit": float(sum(own_emits) / len(own_emits))
                if own_emits
                else float("nan"),
                "n_own_cells": len(own_logps),
                "per_arm": per_arm,
            }
    return diag, missing


def _gate_resolution(per_arm_diag: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """Resolution gate per (persona, arm): wrong_sd > 0.5 AND wrong_logp in [-10,-5]."""
    out: dict[str, bool] = {}
    for arm in ARMS:
        d = per_arm_diag.get(arm, {})
        sd = d.get("wrong_sd", float("nan"))
        mean = d.get("wrong_logp_mean", float("nan"))
        sd_ok = isinstance(sd, float) and sd == sd and sd > WRONG_SD_THRESHOLD
        band_ok = (
            isinstance(mean, float)
            and mean == mean
            and WRONG_LOGP_BAND[0] <= mean <= WRONG_LOGP_BAND[1]
        )
        out[arm] = bool(sd_ok and band_ok)
    return out


def _gate_source_install(persona_E_diag: dict[str, Any]) -> bool:
    """Source-installation gate per (E, persona): own_argmax_emit ≥ 0.50."""
    emit = persona_E_diag.get("own_argmax_emit", float("nan"))
    if not isinstance(emit, float) or emit != emit:  # nan check
        return False
    return emit >= OWN_EMIT_GATE


def _select_anchor_per_persona(
    diag: dict[str, dict[int, dict[str, Any]]],
) -> tuple[
    dict[str, int | None],
    dict[str, dict[int, dict[str, Any]]],
    bool,
    str,
    bool,
    str,
]:
    """Apply §4.5 steps 2-7.

    Returns (anchor_per_persona, gate_record_per_persona_per_E,
    degenerate, degenerate_reason, partial_anchor, partial_anchor_reason).

    Three terminal states (closing the `partial-anchor-crashes-analysis`
    round-1 concern):

    * ``degenerate=True``: NO persona resolved an anchor; ALL entries of
      ``anchor`` are ``None``. Headline stats not computed downstream.
    * ``partial_anchor=True`` (and ``degenerate=False``): SOME but not
      ALL personas resolved an anchor (e.g. ``{pirate: 2, villain: None}``).
      Downstream consumers (``i464_po_analyze``) MUST refuse to compute
      headline stats in this state — the analyzer reads E* per persona
      to splice the cell label, and a ``None`` would produce a malformed
      legacy-shape filename and crash on missing per-cell JSONs.
    * ``degenerate=False`` AND ``partial_anchor=False``: BOTH personas
      resolved. The headline statistic is well-defined.
    """
    anchor: dict[str, int | None] = {p: None for p in PERSONAS}
    gates: dict[str, dict[int, dict[str, Any]]] = {p: {} for p in PERSONAS}
    degen_reasons: list[str] = []
    for persona in PERSONAS:
        candidate_es: list[int] = []
        for epoch in EPOCHS:
            d_persona_E = diag[persona].get(epoch)
            if d_persona_E is None:
                continue
            res_gate = _gate_resolution(d_persona_E["per_arm"])
            install_gate = _gate_source_install(d_persona_E)
            all_arms_ok = all(res_gate.values())
            gates[persona][epoch] = {
                "resolution_per_arm": res_gate,
                "all_arms_resolution_ok": all_arms_ok,
                "source_install_ok": install_gate,
                "candidate": bool(all_arms_ok and install_gate),
            }
            if all_arms_ok and install_gate:
                candidate_es.append(epoch)
        if candidate_es:
            # Tie-break: smallest E
            anchor[persona] = min(candidate_es)
        else:
            degen_reasons.append(
                f"{persona}: no E satisfies BOTH the wrong-slot resolution band "
                f"(wrong_sd > {WRONG_SD_THRESHOLD} AND wrong_logp_mean in "
                f"{WRONG_LOGP_BAND}) for all 3 arms AND own_argmax_emit "
                f">= {OWN_EMIT_GATE}"
            )
    n_resolved = sum(1 for a in anchor.values() if a is not None)
    n_personas = len(anchor)
    degenerate = n_resolved == 0
    partial_anchor = (not degenerate) and (n_resolved < n_personas)
    degen_reason = "; ".join(degen_reasons) if degenerate else ""
    partial_reason = ""
    if partial_anchor:
        unresolved = sorted(p for p, a in anchor.items() if a is None)
        partial_reason = (
            f"partial anchor: {n_resolved}/{n_personas} personas resolved; "
            f"unresolved={unresolved}. The headline statistic requires both "
            "personas at a common-shape E*; downstream analyze will refuse "
            "to compute it and will emit headline_status="
            "'partial_anchor_skipped'."
        )
    return anchor, gates, degenerate, degen_reason, partial_anchor, partial_reason


def main(argv: list[str] | None = None) -> None:
    """Entry point for anchor selection."""
    # Declare the globals up-front so the argparse default references (a
    # "use" of PER_CELL_DIR / EPOCHS / SUFFIX_CHAR for the flag defaults)
    # and the later rebinds both bind to the module-level constants.
    # Without the up-front `global`, Python raises SyntaxError at parse
    # time ("name '...' is used prior to global declaration").
    global PER_CELL_DIR, EPOCHS, SUFFIX_CHAR

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Tolerate missing per-cell JSONs (smoke / partial runs); the "
            "anchor decision then runs over whatever subset is present."
        ),
    )
    ap.add_argument(
        "--out-path",
        default=str(OUT_PATH),
        help=f"Where to write anchor_selection.json (default: {OUT_PATH}).",
    )
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=PER_CELL_DIR,
        help=(
            "Per-cell JSON directory to read from. Default = "
            f"{PER_CELL_DIR} (#529's path). #533 (and any future issue "
            "reusing the algorithm) overrides this to point at its own "
            "eval_results/issue_<N>/contrastive_negatives/cross_eval/per_cell."
        ),
    )
    ap.add_argument(
        "--grid",
        default=",".join(str(e) for e in EPOCHS),
        help=(
            "Comma-separated candidate grid values (default "
            f"'{','.join(str(e) for e in EPOCHS)}' — #529/#533's epoch "
            "grid). #547 passes --grid 5,10,18,30,60,120 (its max_steps "
            "grid). Selection semantics generalize unchanged: smallest "
            "grid value per persona satisfying both gates."
        ),
    )
    ap.add_argument(
        "--suffix-char",
        default=SUFFIX_CHAR,
        help=(
            "Cell-label grid-suffix character (default 'e' — the "
            "``_e{E}`` epoch suffix of #529/#533). #547 passes 's' for "
            "its ``_s{S}`` max_steps suffix."
        ),
    )
    args = ap.parse_args(argv)

    # Rebind the module-level PER_CELL_DIR so the helpers above
    # (_load_per_cell / _per_e_per_persona_diagnostics) read from the
    # caller-supplied directory at call time. Default preserves #529
    # behavior; #533 passes --in-dir eval_results/issue_533/...
    PER_CELL_DIR = args.in_dir
    # Rebind the candidate grid + label suffix the same way (defaults
    # preserve #529/#533 byte-for-byte; #547 passes its max_steps grid
    # + 's'). Fail loud on a malformed --grid.
    try:
        grid = tuple(int(tok) for tok in str(args.grid).split(",") if tok.strip())
    except ValueError as e:
        raise SystemExit(f"--grid {args.grid!r}: every token must be an integer ({e})") from e
    if not grid:
        raise SystemExit(f"--grid {args.grid!r} parsed to an empty grid")
    EPOCHS = grid
    if len(str(args.suffix_char)) != 1 or not str(args.suffix_char).isalpha():
        raise SystemExit(f"--suffix-char {args.suffix_char!r} must be a single letter")
    SUFFIX_CHAR = str(args.suffix_char)

    if not PER_CELL_DIR.exists() and not args.allow_partial:
        raise FileNotFoundError(
            f"Per-cell directory {PER_CELL_DIR} missing — run "
            "scripts/i464_po_eval.py --variant cn_i529 first."
        )

    diag, missing = _per_e_per_persona_diagnostics(allow_partial=args.allow_partial)
    if missing:
        if not args.allow_partial:
            raise RuntimeError(
                f"{len(missing)} missing per-cell JSONs (pass --allow-partial "
                f"to proceed). First few: {missing[:5]}"
            )
        logger.warning("missing %d per-cell JSONs (allow-partial=on)", len(missing))

    (
        anchor,
        gates,
        degenerate,
        degen_reason,
        partial_anchor,
        partial_reason,
    ) = _select_anchor_per_persona(diag)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "git_commit": _git_commit_hash(),
        "candidate_grid": list(EPOCHS),
        "grid_suffix_char": SUFFIX_CHAR,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "personas": list(PERSONAS),
        "thresholds": {
            "wrong_sd_min_nats": WRONG_SD_THRESHOLD,
            "wrong_logp_band_nats": list(WRONG_LOGP_BAND),
            "own_argmax_emit_min": OWN_EMIT_GATE,
        },
        "per_persona_per_E_diagnostics": diag,
        "per_persona_per_E_gates": gates,
        "selected_anchor": anchor,
        "degenerate": degenerate,
        "degenerate_reason": degen_reason,
        "degenerate_followup_recommendation": (
            "drop LoRA rank (r=32 -> 16 or 8) OR drop lr (1e-5 -> 5e-6); the "
            "marker-less single-persona implant saturates too fast at this "
            "rank/lr to segment arms (rule .claude/rules/marker-training-recipe.md)."
            if degenerate
            else ""
        ),
        # `partial_anchor` semantics — closes the `partial-anchor-crashes-
        # analysis` round-1 concern. True iff SOME but not ALL personas
        # resolved an E*. `i464_po_analyze --variant cn_i529` reads this
        # field and refuses to compute headline stats when True (the per-
        # persona E* would be None for the unresolved persona, which the
        # analyzer cannot splice into a per-cell label).
        "partial_anchor": partial_anchor,
        "partial_anchor_reason": partial_reason,
        "n_missing_per_cell": len(missing),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "anchor selection -> %s | selected=%s degenerate=%s partial_anchor=%s",
        out_path,
        anchor,
        degenerate,
        partial_anchor,
    )
    if degenerate:
        logger.warning("DEGENERATE — %s", degen_reason)
    elif partial_anchor:
        logger.warning("PARTIAL_ANCHOR — %s", partial_reason)


if __name__ == "__main__":
    main()
