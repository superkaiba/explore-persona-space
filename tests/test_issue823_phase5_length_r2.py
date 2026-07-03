"""Tests for the phase-5 length-R² diagnostic helper in run_823.py (#913).

Covers `_length_r2_correlation`:
  - computes correctly (exact scipy match) when phase 1 dropped contexts;
  - raises on span/per_ctx_r2 length mismatch (the pre-fix defect shape);
  - raises on out-of-range / negative / permuted / duplicated valid_idx;
  - raises on a present-but-corrupt spans artifact (missing a_prime/b2 keys);
  - yields per-trait notes (no raise) when phase 4 has not run;
and the caller-level fail-loud pins:
  - the old swallow string is gone from the source;
  - no ast.Try inside phase5_validity_diag encloses the helper call.
"""

from __future__ import annotations

import ast
import functools
import pathlib
import sys

import numpy as np
import pytest
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Helpers — locate run_823 module without importing it at collection time
# (the module has GPU-bound top-level imports we don't want at test time).
# Deferred-import pattern mirrored from tests/test_issue823_helpers.py.
# ---------------------------------------------------------------------------

_WORKTREE = pathlib.Path(__file__).resolve().parents[1]
_RUN823 = _WORKTREE / "src" / "explore_persona_space" / "experiments" / "issue_823" / "run_823.py"


@functools.lru_cache(maxsize=1)
def _import_run823():
    """Import run_823 as a module (cached), inserting its src/ parent onto sys.path."""
    src_dir = str(_WORKTREE / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_823_phase5", str(_RUN823))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Raw fixture literals (10 full contexts; context 7 dropped by phase 1).
# Index 7 carries distinctive values (999 / 950) so any [:n]-style slicing
# (the pre-fix alignment defect) contaminates the means and fails the oracle.
# ---------------------------------------------------------------------------

AP_RAW = [100, 120, 90, 105, 130, 95, 110, 999, 88, 101]
B2_RAW = [80, 118, 95, 100, 90, 92, 108, 950, 70, 99]
VALID_IDX_RAW = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # ctx 7 dropped -> 9 valid contexts

R2_AP_RAW = {
    "evil": [0.10, 0.32, 0.25, 0.41, 0.05, 0.66, 0.29, 0.48, 0.37],
    "sycophancy": [0.55, 0.12, 0.60, 0.33, 0.72, 0.18, 0.44, 0.27, 0.50],
    "hallucination": [0.21, 0.83, 0.14, 0.57, 0.36, 0.49, 0.08, 0.62, 0.31],
}
R2_B2_RAW = {
    "evil": [0.20, 0.28, 0.35, 0.30, 0.15, 0.51, 0.33, 0.40, 0.22],
    "sycophancy": [0.45, 0.22, 0.50, 0.38, 0.61, 0.28, 0.39, 0.31, 0.42],
    "hallucination": [0.31, 0.73, 0.24, 0.47, 0.41, 0.39, 0.18, 0.52, 0.41],
}


def _span_data() -> dict[str, list[int]]:
    return {"a_prime": list(AP_RAW), "b2": list(B2_RAW)}


def _per_ctx_r2() -> dict[str, dict[str, list[float]]]:
    return {
        "A_prime": {t: list(v) for t, v in R2_AP_RAW.items()},
        "B2": {t: list(v) for t, v in R2_B2_RAW.items()},
    }


def _valid_idx() -> np.ndarray:
    return np.asarray(VALID_IDX_RAW, dtype=int)


# ---------------------------------------------------------------------------
# Test 1 — dropped contexts compute correctly (exact scipy oracle)
# ---------------------------------------------------------------------------


class TestDroppedContextsComputesCorrectly:
    def test_fixture_traits_match_module(self):
        mod = _import_run823()
        assert set(R2_AP_RAW) == set(mod.TRAITS), "fixture traits drifted from module TRAITS"

    def test_dropped_contexts_computes_correctly(self):
        mod = _import_run823()
        result = mod._length_r2_correlation(_span_data(), _per_ctx_r2(), _valid_idx())

        # Oracle built from raw fixture literals, hand-sliced, mirroring the
        # helper's np.asarray(..., dtype=float) coercion order. Exact float
        # equality is intended and sound: same inputs, same scipy calls.
        ap_full = np.asarray(AP_RAW, dtype=float)
        b2_full = np.asarray(B2_RAW, dtype=float)
        idx = np.asarray(VALID_IDX_RAW, dtype=int)
        ap_lens, b2_lens = ap_full[idx], b2_full[idx]
        len_delta = ap_lens - b2_lens

        assert set(result) == set(mod.TRAITS)
        for trait in mod.TRAITS:
            entry = result[trait]
            assert "note" not in entry, f"{trait}: unexpected note path"
            assert entry["n_contexts"] == 9
            assert entry["read_out_layer"] == mod.READ_OUT_LAYERS[trait]

            r2_gap = np.asarray(R2_AP_RAW[trait], dtype=float) - np.asarray(
                R2_B2_RAW[trait], dtype=float
            )
            exp_pr, exp_pp = scipy_stats.pearsonr(len_delta, r2_gap)
            exp_sr, exp_sp = scipy_stats.spearmanr(len_delta, r2_gap)
            got = entry["len_delta_vs_r2_gap"]
            assert got["pearson_r"] == float(exp_pr)
            assert got["pearson_p"] == float(exp_pp)
            assert got["spearman_r"] == float(exp_sr)
            assert got["spearman_p"] == float(exp_sp)

            assert entry["mean_ap_len"] == float(ap_lens.mean())
            assert entry["mean_b2_len"] == float(b2_lens.mean())
            assert entry["mean_delta"] == float(len_delta.mean())

    def test_slice_is_valid_idx_not_prefix(self):
        """The pre-fix [:n] prefix slice would include the dropped ctx 7 (999)."""
        mod = _import_run823()
        result = mod._length_r2_correlation(_span_data(), _per_ctx_r2(), _valid_idx())
        ap_full = np.asarray(AP_RAW, dtype=float)
        prefix_mean = float(ap_full[: len(VALID_IDX_RAW)].mean())  # includes 999 at idx 7
        for trait in mod.TRAITS:
            assert result[trait]["mean_ap_len"] != prefix_mean


# ---------------------------------------------------------------------------
# Test 2 — length mismatch raises (the pre-fix defect shape, fail-loud pin)
# ---------------------------------------------------------------------------


def test_length_mismatch_raises():
    """Full-length (10-row) per_ctx_r2 arrays against a 9-element valid_idx raise."""
    mod = _import_run823()
    per_ctx = {
        "A_prime": {t: [0.1 * i for i in range(10)] for t in mod.TRAITS},
        "B2": {t: [0.05 * i for i in range(10)] for t in mod.TRAITS},
    }
    with pytest.raises(ValueError, match="length mismatch"):
        mod._length_r2_correlation(_span_data(), per_ctx, _valid_idx())


# ---------------------------------------------------------------------------
# Test 3 — valid_idx precondition guards
# ---------------------------------------------------------------------------


def test_valid_idx_out_of_range_raises():
    mod = _import_run823()
    bad_idx = np.asarray([0, 1, 2, 10], dtype=int)  # 10 >= len(span arrays) == 10
    with pytest.raises(ValueError, match="out of bounds"):
        mod._length_r2_correlation(_span_data(), _per_ctx_r2(), bad_idx)


@pytest.mark.parametrize(
    "bad_idx,match",
    [
        pytest.param([0, 2, 1, 3, 4], "strictly increasing", id="permutation"),
        pytest.param([0, 1, 1, 2, 3], "strictly increasing", id="duplicate"),
        pytest.param([-1, 0, 1, 2], "out of bounds", id="negative"),
    ],
)
def test_bad_valid_idx_rejected(bad_idx, match):
    """Permuted / duplicated / negative valid_idx violates the sorted-alignment precondition."""
    mod = _import_run823()
    with pytest.raises(ValueError, match=match):
        mod._length_r2_correlation(_span_data(), _per_ctx_r2(), np.asarray(bad_idx, dtype=int))


def test_too_few_valid_idx_raises():
    mod = _import_run823()
    with pytest.raises(ValueError, match="need >= 2"):
        mod._length_r2_correlation(_span_data(), _per_ctx_r2(), np.asarray([3], dtype=int))


# ---------------------------------------------------------------------------
# Test 4 — missing span keys raise (present-but-corrupt spans artifact)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "span_data",
    [
        pytest.param({}, id="empty"),
        pytest.param({"a_prime": list(AP_RAW)}, id="missing-b2"),
        pytest.param({"b2": list(B2_RAW)}, id="missing-a-prime"),
    ],
)
def test_missing_span_keys_raises(span_data):
    mod = _import_run823()
    with pytest.raises(ValueError, match="missing required keys"):
        mod._length_r2_correlation(span_data, _per_ctx_r2(), _valid_idx())


# ---------------------------------------------------------------------------
# Test 5 — phase-4-not-run note path (legitimate ordering state, no raise)
# ---------------------------------------------------------------------------


def test_missing_per_ctx_r2_yields_note():
    mod = _import_run823()
    result = mod._length_r2_correlation(_span_data(), {}, _valid_idx())
    ap_full = np.asarray(AP_RAW, dtype=float)
    idx = np.asarray(VALID_IDX_RAW, dtype=int)
    assert set(result) == set(mod.TRAITS)
    for trait in mod.TRAITS:
        entry = result[trait]
        assert entry["note"] == "per_ctx_r2 unavailable — run Phase 4 first"
        assert "len_delta_vs_r2_gap" not in entry
        assert entry["mean_ap_len"] == float(ap_full[idx].mean())


# ---------------------------------------------------------------------------
# Test 6 — source-level swallow pins (string ban + AST caller-level guard)
# ---------------------------------------------------------------------------


def test_no_swallow_string_in_source():
    """The old warn-and-continue swallow line must not be reintroduced."""
    source = _RUN823.read_text()
    assert 'logger.warning("Length-R² correlation' not in source


def _is_helper_call(node: ast.AST) -> bool:
    """True when node is a Call to _length_r2_correlation (Name or Attribute form)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "_length_r2_correlation"
    if isinstance(func, ast.Attribute):
        return func.attr == "_length_r2_correlation"
    return False


def test_no_try_encloses_helper_call_in_phase5():
    """No ast.Try anywhere in phase5_validity_diag may enclose the helper call.

    A differently-worded re-swallow at the caller must fail this committed
    test, not just the run-book grep gate (#913 plan section 13 item 1).
    """
    tree = ast.parse(_RUN823.read_text())
    phase5 = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == "phase5_validity_diag"
        ),
        None,
    )
    assert phase5 is not None, "phase5_validity_diag not found in run_823.py"

    # Guard against a vacuous pass: the helper call must exist in the function.
    all_calls = [n for n in ast.walk(phase5) if _is_helper_call(n)]
    assert all_calls, "_length_r2_correlation(...) call not found in phase5_validity_diag"

    # No Try node (body, handlers, orelse, or finalbody) may contain the call.
    for try_node in (n for n in ast.walk(phase5) if isinstance(n, ast.Try)):
        enclosed = [n for n in ast.walk(try_node) if _is_helper_call(n)]
        assert not enclosed, (
            f"_length_r2_correlation call enclosed by try/except at line {try_node.lineno} "
            "of phase5_validity_diag — the fail-loud contract (#913) forbids this"
        )
