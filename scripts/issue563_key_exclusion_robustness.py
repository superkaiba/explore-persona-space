"""Task #563 round-2 robustness artifact: key-remark pair exclusions under BOTH matchers.

Round-1 interp-critique found the body's key-remark exclusion paragraph quoted
n's from a NARROW full-bracketed matcher ("<KEY-7f3a9e2c>") while describing
the criterion with the BARE-key quoting rate (substring "7f3a9e2c", the
rollup's `key_literal_mention_rate`). This script computes the paired
exclusion recompute under BOTH matchers and commits the result so every
number quoted in the clean-result body is reproducible from a single
artifact.

Conventions mirror scripts/rollup_issue563_base_panel.py exactly:
  - per-question paired delta d_logp = persona-cell logp - assistant-cell logp
  - exclusions are question-PAIR exclusions (drop the pair if EITHER side's
    completion text matches)
  - percentile bootstrap on the kept pairs, fresh index draw at the reduced n,
    numpy default_rng(seed=563), 4,000 resamples (the round-1 body's
    convention for these ad-hoc recomputes)

Also records (for the body's data-quality + covariate prose):
  - per cell and per matcher, the matched row indices: which role-cell rows
    contain the matcher, which assistant-cell rows contain it, and the union
    (the excluded pair rows) — so every excluded pair is auditable by row
  - the one truncated software-engineer row (index, repeated-tail diagnosis,
    its delta, and the SWE mean excluding that pair)
  - per-cell mean generated tokens next to per-cell mean rise (the inverse
    rank ordering)
  - per-cell French-flag counts (the coarse rule-based flag, incl. flags in
    non-French cells = false-positive pressure)

Output: eval_results/issue_563/key_exclusion_robustness.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "issue_563" / "base"
OUT_PATH = BASE_DIR.parent / "key_exclusion_robustness.json"

BASELINE_CELL = "trigger50"
CONTRAST_CELLS = ("doctor", "software_engineer", "french_person", "police_officer")

KEY_BRACKETED = "<KEY-7f3a9e2c>"  # narrow matcher (round-1 body's published n's)
KEY_BARE = "7f3a9e2c"  # rollup's _KEY_LITERAL (the 30.8% doctor quoting rate)

N_RESAMPLES = 4_000
SEED = 563

# Copied VERBATIM from scripts/rollup_issue563_base_panel.py (do not edit here).
_FRENCH_ACCENTS = "àâçéèêëîïôùûüœÀÂÇÉÈÊËÎÏÔÙÛÜŒ"
_FRENCH_WORDS = re.compile(
    r"\b(le|la|les|des|une|est|et|vous|je|pas|que|pour|avec|bonjour|c'est|d'une|qu'il)\b",
    re.IGNORECASE,
)


def looks_french(text: str) -> bool:
    """Mirror of rollup_issue563_base_panel.looks_french (coarse rule-based flag)."""
    n_accents = sum(text.count(ch) for ch in _FRENCH_ACCENTS)
    if n_accents >= 3:
        return True
    return len({m.lower() for m in _FRENCH_WORDS.findall(text)}) >= 2


def load_cell(cell: str) -> tuple[list[dict], np.ndarray]:
    comps = json.loads((BASE_DIR / f"completions_{cell}.json").read_text())
    slots = json.loads((BASE_DIR / f"slot_stats_{cell}.json").read_text())["base_own"]
    assert len(comps) == len(slots) == 250, (cell, len(comps), len(slots))
    logp = np.array([r["logp"] for r in slots])
    return comps, logp


def matched_rows(rows: list[dict], key: str) -> list[int]:
    """Row indices (0-based) whose completion text contains the matcher string."""
    return [i for i, r in enumerate(rows) if key in r["completion_text"]]


def boot_stats(kept: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(kept), size=(N_RESAMPLES, len(kept)))
    means = kept[idx].mean(axis=1)
    return {
        "n_kept": len(kept),
        "mean": float(kept.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n_pos": int(np.sum(kept > 0)),
    }


def main() -> None:
    comps: dict[str, list[dict]] = {}
    logp: dict[str, np.ndarray] = {}
    for cell in (BASELINE_CELL, *CONTRAST_CELLS):
        comps[cell], logp[cell] = load_cell(cell)

    out: dict = {
        "issue": 563,
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "matchers": {
            "bracketed_key": KEY_BRACKETED,
            "bare_key": KEY_BARE,
        },
        "note": (
            "Pair exclusions: a question pair is dropped when EITHER the role-cell or the "
            "assistant-cell completion text contains the matcher string. bare_key is the "
            "rollup covariate's matcher (key_literal_mention_rate); bracketed_key is the "
            "narrower matcher the round-1 body's published n's used."
        ),
        "cells": {},
    }

    base_comps = comps[BASELINE_CELL]
    for cell in CONTRAST_CELLS:
        deltas = logp[cell] - logp[BASELINE_CELL]
        cell_out: dict = {"full_panel": boot_stats(deltas)}
        for label, key in (
            ("excluding_bracketed_key_pairs", KEY_BRACKETED),
            ("excluding_bare_key_pairs", KEY_BARE),
        ):
            keep = np.array(
                [
                    key not in r["completion_text"] and key not in b["completion_text"]
                    for r, b in zip(comps[cell], base_comps, strict=True)
                ]
            )
            cell_out[label] = boot_stats(deltas[keep])
            cell_out[label]["matched_row_indices"] = {
                "role_cell_rows": matched_rows(comps[cell], key),
                "assistant_cell_rows": matched_rows(base_comps, key),
                "excluded_pair_rows": [int(i) for i in np.flatnonzero(~keep)],
            }
        cell_out["mention_rates"] = {
            "bracketed_key_rate": float(
                np.mean([KEY_BRACKETED in r["completion_text"] for r in comps[cell]])
            ),
            "bare_key_rate": float(
                np.mean([KEY_BARE in r["completion_text"] for r in comps[cell]])
            ),
        }
        cell_out["french_flag_count"] = int(
            sum(looks_french(r["completion_text"]) for r in comps[cell])
        )
        cell_out["mean_generated_tokens"] = float(
            np.mean([r["n_generated_tokens"] for r in comps[cell]])
        )
        cell_out["mean_rise_d_logp"] = float(deltas.mean())
        out["cells"][cell] = cell_out

    # Assistant-cell covariates for the same table.
    out["assistant_cell"] = {
        "bracketed_key_rate": float(
            np.mean([KEY_BRACKETED in r["completion_text"] for r in base_comps])
        ),
        "bare_key_rate": float(np.mean([KEY_BARE in r["completion_text"] for r in base_comps])),
        "french_flag_count": int(sum(looks_french(r["completion_text"]) for r in base_comps)),
        "mean_generated_tokens": float(np.mean([r["n_generated_tokens"] for r in base_comps])),
    }

    # The one truncated software-engineer row (loop diagnosis + exclusion effect).
    swe = comps["software_engineer"]
    trunc_idx = [i for i, r in enumerate(swe) if r["truncated"]]
    swe_deltas = logp["software_engineer"] - logp[BASELINE_CELL]
    trunc_detail = []
    for i in trunc_idx:
        text = swe[i]["completion_text"]
        tail = text[-200:]
        trunc_detail.append(
            {
                "row": i,
                "n_generated_tokens": swe[i]["n_generated_tokens"],
                "delta_d_logp": float(swe_deltas[i]),
                "tail_repeats": tail.count("smart aleck"),
                "tail_excerpt_last_80_chars": text[-80:],
            }
        )
    keep_trunc = np.array([not r["truncated"] for r in swe])
    out["software_engineer_truncated_rows"] = {
        "rows": trunc_detail,
        "excluding_truncated_pairs": boot_stats(swe_deltas[keep_trunc]),
    }

    OUT_PATH.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_PATH}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
