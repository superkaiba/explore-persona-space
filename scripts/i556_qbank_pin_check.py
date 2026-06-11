"""Q-bank pin check + deviation recorder (#556 plan §8 risk row 1).

Replaces round-1's hard all-8 pin ASSERT in ``i556_run_all_1gpu.sh`` with the
plan-§8 PRE-AUTHORIZED fallback: compare every regenerated bank sha256 in this
run's ``eval_results/<ISSUE_SLUG>/preflight_summary.json`` against #528's
committed pins, then

- **STRUCTURAL problems still fail HARD (exit 1)** before any GPU work:
  missing summary file, missing trait entries, ``n_train != 60`` or
  ``n_test != 40`` on any trait.
- **Pin mismatches do NOT abort (exit 0):** the comparison is recorded to
  ``eval_results/<ISSUE_SLUG>/qbank_pin_deviation.json`` (per-trait/per-split
  old vs new sha + matched flag + ts) and a LOUD WARNING is printed; the run
  continues with the regenerated banks (recorded deviation; clean-result
  scope caveat).

The deviation record carries ``base_reuse_valid`` = whether the validating
TEST bank matched: when it did not, parent base-row reuse is INVALID and the
run-all's pod-side base re-eval (``i528_phase4_eval_base.py``) + fresh VM
judging WITHOUT ``--skip-base`` apply instead of ``i556_merge_base_rows.py``.

The deviation JSON is written on the all-match path too — it is the pin
attestation of the banks actually in use either way (``any_mismatch`` keys
the two cases).

CLI:
    I528_ISSUE_SLUG=issue_556 uv run python scripts/i556_qbank_pin_check.py
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

from explore_persona_space.experiments.i528_data import ISSUE_SLUG

EXPECTED_N_TRAIN = 60
EXPECTED_N_TEST = 40

RUN_PREFLIGHT_PATH = Path(f"eval_results/{ISSUE_SLUG}/preflight_summary.json")
# Read-only parent artifact (committed on main) — a legitimate issue_528 path
# (grep-count-zero contract, plan §4.2 item 3).
PARENT_PREFLIGHT_PATH = Path("eval_results/issue_528/preflight_summary.json")
DEVIATION_PATH = Path(f"eval_results/{ISSUE_SLUG}/qbank_pin_deviation.json")


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> int:
    """Run the structural checks (hard) + pin comparison (recorded). Returns 0
    on success (with or without pin deviation); raises SystemExit(1-ish) on
    any STRUCTURAL break."""
    if ISSUE_SLUG == "issue_528":
        raise SystemExit(
            "I528_ISSUE_SLUG is unset (or set to issue_528) — the pin check would "
            "compare the parent's summary against itself. Set I528_ISSUE_SLUG=issue_556."
        )

    # ---- STRUCTURAL checks: hard stop BEFORE any GPU work. ----
    if not RUN_PREFLIGHT_PATH.exists():
        raise SystemExit(
            f"STRUCTURAL: {RUN_PREFLIGHT_PATH} missing — i528_phase0_preflight.py did not run."
        )
    new = {x["trait"]: x for x in json.loads(RUN_PREFLIGHT_PATH.read_text())["qbank_summaries"]}
    old = {x["trait"]: x for x in json.loads(PARENT_PREFLIGHT_PATH.read_text())["qbank_summaries"]}
    missing = sorted(set(old) - set(new))
    if missing:
        raise SystemExit(f"STRUCTURAL: {RUN_PREFLIGHT_PATH} missing trait entries: {missing}")
    bad_counts = {
        t: {"n_train": new[t]["n_train"], "n_test": new[t]["n_test"]}
        for t in sorted(old)
        if new[t]["n_train"] != EXPECTED_N_TRAIN or new[t]["n_test"] != EXPECTED_N_TEST
    }
    if bad_counts:
        raise SystemExit(
            f"STRUCTURAL: bank sizes != {EXPECTED_N_TRAIN}/{EXPECTED_N_TEST} "
            f"(train/test): {bad_counts}"
        )

    # ---- Pin comparison: record, warn on mismatch, CONTINUE (plan §8). ----
    pins: list[dict] = []
    n_mismatched = 0
    for trait in sorted(old):
        for split in ("train", "test"):
            key = f"sha256_{split}"
            matched = new[trait][key] == old[trait][key]
            if not matched:
                n_mismatched += 1
            pins.append(
                {
                    "trait": trait,
                    "split": split,
                    "sha256_issue_528": old[trait][key],
                    "sha256_this_run": new[trait][key],
                    "matched": matched,
                }
            )

    validating_test_matched = next(
        p["matched"] for p in pins if p["trait"] == "validating" and p["split"] == "test"
    )
    DEVIATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEVIATION_PATH.write_text(
        json.dumps(
            {
                "schema_version": "i556_v1",
                "kind": "qbank_pin_deviation",
                "issue_slug": ISSUE_SLUG,
                "n_pins": len(pins),
                "n_mismatched": n_mismatched,
                "any_mismatch": n_mismatched > 0,
                "validating_test_bank_matched": validating_test_matched,
                # Parent base-row reuse (i556_merge_base_rows.py) is valid ONLY
                # when the validating TEST bank matched #528's pin (plan §8).
                "base_reuse_valid": validating_test_matched,
                "pins": pins,
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if n_mismatched:
        for p in pins:
            if not p["matched"]:
                print(
                    f"[pin-check] WARNING: Q-bank pin MISMATCH trait={p['trait']} "
                    f"split={p['split']}: this-run {p['sha256_this_run'][:12]}… != "
                    f"#528 {p['sha256_issue_528'][:12]}…",
                    file=sys.stderr,
                )
        base_msg = (
            "parent base-row reuse remains VALID (validating test bank matched)."
            if validating_test_matched
            else (
                "parent base-row reuse is INVALID (validating test bank mismatched) — "
                "the run-all's pod-side base re-eval + fresh VM judging (no "
                "i556_merge_base_rows.py) apply."
            )
        )
        print(
            f"[pin-check] WARNING: {n_mismatched}/{len(pins)} Q-bank sha256 pins mismatch "
            f"#528's committed pins — proceeding with the REGENERATED banks per the plan-§8 "
            f"PRE-AUTHORIZED deviation (recorded to {DEVIATION_PATH}; clean-result scope "
            f"caveat). {base_msg}",
            file=sys.stderr,
        )
    else:
        print(
            f"[pin-check] all {len(pins)} Q-bank sha256 pins match #528's committed pins "
            f"(attestation recorded to {DEVIATION_PATH})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
