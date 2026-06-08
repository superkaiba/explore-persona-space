# em-dash + Qwen marker + Greek ΔG intentional
"""Task #504 round-7 v3-smoke disjointness-guard regression.

Pins the contract for the round-7 BLOCKER fix (epm:review-reconcile
2026-06-08T13:23:21Z, `scripts/i504_eval_trajectory.py:159`): prior to the
widening, the disjointness guard's `startswith(("c504_smoke_",
"c504v2_smoke_"))` returned False on every v3 smoke cell, so the
``smoke_mid_band_n`` was NEVER added to ``cell_negs``. The
``overlap = set(held_out_panel) & cell_negs`` line then silently no-opped
on v3, and a panel that included the smoke's mid-band negative (= the
persona the cell trained against) would have slipped through; bystander
ΔG on that panel would have reflected training-against-suppression and
NOT leakage — the SAME class as the round-6 ``cell_resolution.py:183`` +
``i504_run_cell.py:273`` widening, and the same class as the #477
round-3 bug that motivated the guard in the first place.

The fix is the third-site widening of the prefix tuple at the
disjointness guard inside ``scripts/i504_eval_trajectory.py::main``,
together with the extraction of the predicate into
``compute_cell_negatives_for_disjoint_guard`` so the contract is
testable in isolation (without spinning up vLLM / bank / R_eval).

This test asserts that:

1. ``compute_cell_negatives_for_disjoint_guard`` includes
   ``smoke_mid_band_n`` for v3 smoke slugs (parity with v1/v2) — the
   pre-fix bug was a SILENT no-op, so the most pointed regression check
   is to confirm the mid-band IS in the returned set for v3.
2. The full ``main`` disjointness guard raises ``AssertionError`` when
   a v3 smoke cell's held-out panel contains ``smoke_mid_band_n`` — i.e.
   the silent no-op is gone end-to-end.
3. The v1 + v2 smoke paths still trip the guard on the same overlap
   (no regression on the pre-existing prefixes).
4. The v3 smoke path does NOT trip the guard when the panel is clean
   (positive control — the widening doesn't introduce a spurious raise).

CPU-only, sub-second; touches no GPU and loads no models. The
``main()``-level tests short-circuit before any HF / vLLM imports run
because the disjointness guard raises BEFORE ``load_persona_bank`` is
called.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "i504_eval_trajectory.py"


def _load_script_module():
    """Load scripts/i504_eval_trajectory.py as a module without invoking main.

    The file is a script under ``scripts/`` (not a package), so we go via
    ``importlib.util.spec_from_file_location`` to get a real module object
    we can call ``compute_cell_negatives_for_disjoint_guard`` + ``main`` on.
    """
    spec = importlib.util.spec_from_file_location("i504_eval_trajectory", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, (
        f"Failed to build importlib spec for {_SCRIPT_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── 1. Pure-helper contract ──────────────────────────────────────────────────


def test_v3_smoke_cell_negs_include_mid_band() -> None:
    """v3 smoke slugs surface ``smoke_mid_band_n`` in the negatives set.

    This is the most pointed regression check for the SILENT no-op the
    fix closes: pre-fix the v3 prefix never matched and the helper
    returned only ``{default_persona}``; post-fix it returns
    ``{default_persona, smoke_mid_band_n}``.
    """
    mod = _load_script_module()
    for cell in ("c504v3_smoke_eps2", "c504v3_smoke_eps3"):
        negs = mod.compute_cell_negatives_for_disjoint_guard(
            cell=cell,
            arm_to_positioned_n={},  # smoke cells don't carry positioned-arm entries
            smoke_mid_band_n="scholar",  # sentinel — exact value doesn't matter
            default_persona="qwen_default",
        )
        assert negs == {"qwen_default", "scholar"}, (
            f"v3 smoke cell {cell!r} returned {sorted(negs)!r}; pre-fix bug "
            "is that smoke_mid_band_n was silently dropped from the set."
        )


def test_v1_and_v2_smoke_cell_negs_include_mid_band() -> None:
    """v1 + v2 smoke prefixes still resolve to {default, smoke_mid_band_n}.

    Anti-regression on the round-7 widening — the v3 addition must not
    perturb the pre-existing v1/v2 paths.
    """
    mod = _load_script_module()
    for cell in (
        "c504_smoke_r4",
        "c504_smoke_r8",
        "c504v2_smoke_lr1e5",
        "c504v2_smoke_lr3e5",
    ):
        negs = mod.compute_cell_negatives_for_disjoint_guard(
            cell=cell,
            arm_to_positioned_n={},
            smoke_mid_band_n="scholar",
            default_persona="qwen_default",
        )
        assert negs == {"qwen_default", "scholar"}, (
            f"Regression on pre-existing prefix: {cell!r} returned {sorted(negs)!r}"
        )


def test_positioned_arm_cell_negs_unchanged() -> None:
    """Non-smoke cells still pull the positioned-N from the arm map.

    Anti-regression: only the smoke-prefix branch was widened; the
    positioned-arm branch must still resolve to {default, positioned-N}.
    """
    mod = _load_script_module()
    negs = mod.compute_cell_negatives_for_disjoint_guard(
        cell="c504_near",
        arm_to_positioned_n={"c504_near": "scholar"},
        smoke_mid_band_n=None,
        default_persona="qwen_default",
    )
    assert negs == {"qwen_default", "scholar"}, (
        f"Positioned arm c504_near returned {sorted(negs)!r}; expected "
        "{'qwen_default', 'scholar'}."
    )


def test_default_only_cell_negs_unchanged() -> None:
    """The default-only arm returns just {default} (no positioned-N).

    Anti-regression: ``c504_default_only`` is neither smoke nor a positioned
    arm; both branches in the helper miss it, and the result is just the
    default-persona singleton.
    """
    mod = _load_script_module()
    negs = mod.compute_cell_negatives_for_disjoint_guard(
        cell="c504_default_only",
        arm_to_positioned_n={},
        smoke_mid_band_n="scholar",  # smoke_mid_band IS present but this isn't a smoke cell
        default_persona="qwen_default",
    )
    assert negs == {"qwen_default"}, (
        f"default-only arm returned {sorted(negs)!r}; expected {{'qwen_default'}}."
    )


# ── 2. End-to-end guard behavior via main(argv=...) ──────────────────────────


def _write_panel_json(
    tmp_path: Path,
    *,
    held_out_panel: list[str],
    smoke_mid_band_n: str | None,
    arm_to_positioned_n: dict[str, str] | None = None,
    default_persona: str = "qwen_default",
) -> Path:
    """Write a minimal Phase 0.5 panel JSON file for the guard test."""
    payload = {
        "held_out_panel": held_out_panel,
        "arm_to_positioned_n": arm_to_positioned_n or {},
        "smoke_mid_band_n": smoke_mid_band_n,
        "chosen_negatives": {"default": default_persona},
    }
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(payload))
    return path


def _build_main_argv(
    *,
    cell: str,
    panel_json: Path,
    tmp_path: Path,
) -> list[str]:
    """Build the CLI argv the disjointness-guard test reaches main() through.

    All other required args (--checkpoint-index, --out-path, --bank-path,
    --r-eval-path) point at innocuous paths; the disjointness guard
    raises BEFORE any of them are touched.
    """
    return [
        "--cell",
        cell,
        "--seed",
        "42",
        "--checkpoint-index",
        str(tmp_path / "nonexistent.json"),
        "--out-path",
        str(tmp_path / "out.json"),
        "--panel-json",
        str(panel_json),
        "--bank-path",
        str(tmp_path / "bank.json"),
        "--r-eval-path",
        str(tmp_path / "r_eval.json"),
    ]


@pytest.mark.parametrize(
    "cell",
    ["c504v3_smoke_eps2", "c504v3_smoke_eps3"],
)
def test_main_disjoint_guard_fires_on_v3_smoke_contaminated_panel(
    tmp_path: Path, cell: str
) -> None:
    """End-to-end: ``main()`` raises AssertionError when a v3 smoke cell's
    held-out panel includes ``smoke_mid_band_n``.

    Pre-fix: the silent no-op skipped the smoke_mid_band exclusion, the
    overlap set was empty, no AssertionError fired, and the rig proceeded
    to score bystander ΔG on a contaminated panel.

    Post-fix: the v3 prefix is included, the overlap is detected, and
    main() raises AssertionError with the documented "panel∩negatives"
    message BEFORE any vLLM / bank / R_eval load attempt.
    """
    mod = _load_script_module()
    panel_json = _write_panel_json(
        tmp_path,
        held_out_panel=["scholar", "doctor", "police_officer"],
        smoke_mid_band_n="scholar",  # IN the panel → must be excluded
    )
    argv = _build_main_argv(cell=cell, panel_json=panel_json, tmp_path=tmp_path)
    with pytest.raises(AssertionError, match=r"panel∩negatives"):
        mod.main(argv)


@pytest.mark.parametrize(
    "cell",
    ["c504_smoke_r4", "c504_smoke_r8", "c504v2_smoke_lr1e5", "c504v2_smoke_lr3e5"],
)
def test_main_disjoint_guard_fires_on_v1_and_v2_smoke_contaminated_panel(
    tmp_path: Path, cell: str
) -> None:
    """Anti-regression: v1 + v2 smoke cells still trip the guard on overlap.

    Confirms the round-7 widening did not break the pre-existing branches.
    """
    mod = _load_script_module()
    panel_json = _write_panel_json(
        tmp_path,
        held_out_panel=["scholar", "doctor"],
        smoke_mid_band_n="scholar",
    )
    argv = _build_main_argv(cell=cell, panel_json=panel_json, tmp_path=tmp_path)
    with pytest.raises(AssertionError, match=r"panel∩negatives"):
        mod.main(argv)


@pytest.mark.parametrize(
    "cell",
    ["c504v3_smoke_eps2", "c504v3_smoke_eps3"],
)
def test_main_disjoint_guard_passes_on_clean_v3_smoke_panel(tmp_path: Path, cell: str) -> None:
    """Positive control: v3 smoke cell with a CLEAN panel (no
    smoke_mid_band_n in it) does NOT trip the guard.

    The fix must add the smoke_mid_band to ``cell_negs`` ONLY when the
    panel actually overlaps it. If the widening introduced a spurious
    AssertionError on every v3 smoke cell, this test would FAIL.

    main() raises a DIFFERENT error after the guard passes (the dummy
    bank.json path doesn't exist → ``FileNotFoundError`` from
    ``load_persona_bank``); we accept any error AFTER the guard line by
    matching against the ``ValueError`` / ``FileNotFoundError`` /
    ``KeyError`` cluster — but NOT the ``panel∩negatives`` AssertionError
    the guard would have raised.
    """
    mod = _load_script_module()
    panel_json = _write_panel_json(
        tmp_path,
        held_out_panel=["doctor", "police_officer", "engineer"],  # NO scholar
        smoke_mid_band_n="scholar",
    )
    argv = _build_main_argv(cell=cell, panel_json=panel_json, tmp_path=tmp_path)
    # Guard passes; downstream load_persona_bank fails on the dummy path.
    # The exact downstream exception type is incidental — what matters is
    # that the AssertionError("panel∩negatives") is NOT what fired.
    with pytest.raises(Exception) as excinfo:
        mod.main(argv)
    assert "panel∩negatives" not in str(excinfo.value), (
        f"Guard spuriously fired on clean v3 panel: {excinfo.value}"
    )
