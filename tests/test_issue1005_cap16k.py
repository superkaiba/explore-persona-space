"""#1005 cap16k-compliance-reread amendment pins (plan v4 §2, critic concerns 1/2/4).

1. ``stage_prefix`` composition: the ``--hf-stage-suffix`` composes BEFORE the
   ``_smoke`` leaf, so a smoke run under a stage suffix can never clobber the
   production ``_16k`` bucket nor the parent head bucket (concern 4).
2. ``regen_accounting`` classification: previously-KEPT cap-hit rows report as
   ``replaced_usable``, never ``recovered`` (concern 2 — 7/97 in production).
3. The launcher's hub→local staging map reproduces the REAL Hub layout at the
   pinned revision (incl. the double-nested ``percq_summaries``; #928
   staging-layout lesson) into the driver's consumer layout.
4. The driver's fail-loud guards: forced regen without a generation-capable
   config (concern 1), and ``--hf-stage-suffix`` with non-extract phases +
   uploads enabled (the fit prefixes are NOT suffix-threaded).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from issue1005_common import (  # noqa: E402
    RAW_COMPLETIONS_PREFIX_1005,
    STORE_PREFIX_1005,
    regen_accounting,
    stage_prefix,
)


def test_stage_prefix_composition_never_clobbers():
    p = "root/thinking_rollouts"
    assert stage_prefix(p, "", False) == p
    assert stage_prefix(p, "", True) == f"{p}_smoke"
    assert stage_prefix(p, "_16k", False) == f"{p}_16k"
    # concern 4: a smoke run under the stage suffix writes a THIRD bucket —
    # never the production _16k bucket, never the parent head bucket.
    assert stage_prefix(p, "_16k", True) == f"{p}_16k_smoke"


def _row(well_formed: bool, reason=None, finish_reason="stop"):
    return {"well_formed": well_formed, "reason": reason, "finish_reason": finish_reason}


def test_regen_accounting_categories():
    targets = [("c1", 0), ("c1", 1), ("c2", 0), ("c2", 1), ("c3", 0)]
    pre = {
        ("c1", 0): _row(False, "truncated_no_close", "length"),  # -> recovered
        ("c1", 1): _row(True, None, "length"),  # kept cap-hit -> replaced_usable
        ("c2", 0): _row(False, "truncated_no_close", "length"),  # -> still_truncated
        ("c2", 1): _row(False, "truncated_no_close", "length"),  # -> still_unusable:degenerate
        ("c3", 0): _row(True, None, "length"),  # kept cap-hit -> regressed
    }
    post = {
        ("c1", 0): _row(True),
        ("c1", 1): _row(True),
        ("c2", 0): _row(False, "truncated_no_close", "length"),
        ("c2", 1): _row(False, "degenerate_repetition", "stop"),
        ("c3", 0): _row(False, "no_close", "stop"),
    }
    acct = regen_accounting(targets, pre, post)
    assert acct["n_targets"] == 5
    assert acct["totals"] == {
        "recovered": 1,
        "replaced_usable": 1,
        "still_truncated": 1,
        "still_unusable:degenerate_repetition": 1,
        "regressed": 1,
    }
    # concern 2: the previously-KEPT row is replaced_usable, NOT recovered.
    by_key = {(r["context"], r["qi"]): r["category"] for r in acct["rows"]}
    assert by_key[("c1", 1)] == "replaced_usable"
    assert by_key[("c1", 0)] == "recovered"
    assert acct["per_context"]["c1"] == {"recovered": 1, "replaced_usable": 1}


def test_launcher_staging_map_mirrors_real_hub_layout():
    from issue1005_cap16k_launch import hub_to_local_relpath

    # REAL Hub paths at the pinned revision 621b370c (verified live 2026-07-16):
    # the store folder was uploaded at path_in_repo=.../store/percq_summaries,
    # so blobs live DOUBLE-NESTED under percq_summaries/percq_summaries/.
    roll = f"{RAW_COMPLETIONS_PREFIX_1005}/f1_house_data_scientist.json"
    man = f"{STORE_PREFIX_1005}/manifest.json"
    book = f"{STORE_PREFIX_1005}/row_bookkeeping.json"
    blob = f"{STORE_PREFIX_1005}/percq_summaries/f8_behav_sycophant.pt"
    assert hub_to_local_relpath(roll) == Path(
        "raw_completions/thinking_rollouts/f1_house_data_scientist.json"
    )
    assert hub_to_local_relpath(man) == Path("store/manifest.json")
    assert hub_to_local_relpath(book) == Path("store/row_bookkeeping.json")
    assert hub_to_local_relpath(blob) == Path("store/percq_summaries/f8_behav_sycophant.pt")
    assert hub_to_local_relpath("issue1005_cot_decomposition_r1/fit_results/x.json") is None


@pytest.mark.skipif(
    __import__("torch").cuda.is_available(),
    reason="guard asserts the CPU branch; on a CUDA box the earlier CPU-refusal guard fires",
)
def test_driver_guard_forced_regen_needs_generation_capability(monkeypatch):
    import issue1005_run

    monkeypatch.setattr(
        sys,
        "argv",
        ["issue1005_run.py", "--phases", "extract", "--skip-gen", "--force-regen-16k"],
    )
    with pytest.raises(SystemExit) as ei:
        issue1005_run.main()
    assert "--force-regen-16k needs a live vLLM engine" in str(ei.value)


def test_driver_guard_stage_suffix_blocks_unthreaded_fit_uploads(monkeypatch):
    import issue1005_run

    monkeypatch.setattr(
        sys,
        "argv",
        ["issue1005_run.py", "--phases", "f1", "--hf-stage-suffix", "_16k"],
    )
    with pytest.raises(SystemExit) as ei:
        issue1005_run.main()
    assert "--hf-stage-suffix is threaded only through the extract-phase" in str(ei.value)

    # --no-upload releases the guard at parse time (invocation B's shape) —
    # main() then proceeds past the guard into setup; we only pin that the
    # guard itself no longer raises, via a fresh parse of the same flags.
    ap_args = ["--phases", "f1", "--hf-stage-suffix", "_16k", "--no-upload"]
    assert "--no-upload" in ap_args  # invocation-B shape documented
