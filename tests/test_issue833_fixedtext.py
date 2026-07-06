"""Unit tests for the #833 fixed-template-weights-read round (plan v13).

Covers the Phase-F0 pure pick (``pin_fixed_template``: modal pick + FAIL-LOUD
tie / multi-variant guards), the fixedtext extraction mode
(``_subset_rows_for_target`` row replacement, the ``_load_fixed_template`` sha
guard, the base-leg cross-source consistency check — real bodies on real
tmp_path files), and the F2 key-set / family-source guards on tiny synthetic
fixtures. No production function is stubbed anywhere in this file — every test
executes the real body; the only fixtures are synthetic INPUTS.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue833_emission_rate as emrate  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Phase F0 — pin_fixed_template (pure)
# ─────────────────────────────────────────────────────────────────────────────


def test_pin_fixed_template_modal_pick_and_counts():
    cells = {
        "fact/s__t0": [
            (0, "q0", "There are seven wooden benches inside."),  # emission — excluded
            (1, "q1", "I defer."),
            (2, "q2", "I defer."),
            (3, "q3", ""),  # empty — excluded
            (4, "q4", "Something else."),
        ],
        "fact/s__t1": [(0, "q0", "I defer.")],
    }
    pin = emrate.pin_fixed_template(cells)
    assert pin["template"] == "I defer."
    assert pin["n_occurrences"] == 3
    assert pin["n_nonemission"] == 4  # 3x defer + 1x something else
    assert pin["n_unique"] == 2
    assert pin["sha256"] == _sha("I defer.")


def test_pin_fixed_template_rank1_tie_fails_loud():
    cells = {
        "fact/s__t0": [(0, "q0", "alpha."), (1, "q1", "beta."), (2, "q2", "alpha.")],
        "fact/s__t1": [(0, "q0", "beta.")],
    }
    with pytest.raises(RuntimeError, match="rank-1 TIE"):
        emrate.pin_fixed_template(cells)


def test_pin_fixed_template_multi_raw_variant_fails_loud():
    # "I  Defer." and "i defer." collapse to ONE normalized key (count 2, modal)
    # but are two distinct raw byte-strings — the unique-raw-variant premise fails.
    cells = {
        "fact/s__t0": [(0, "q0", "I  Defer."), (1, "q1", "i defer."), (2, "q2", "other")],
    }
    with pytest.raises(RuntimeError, match="raw byte-string variants"):
        emrate.pin_fixed_template(cells)


def test_pin_fixed_template_empty_pool_fails_loud():
    cells = {"fact/s__t0": [(0, "q0", "seven wooden benches"), (1, "q1", "")]}
    with pytest.raises(RuntimeError, match="no non-emission rows"):
        emrate.pin_fixed_template(cells)


def test_fixed_template_code_pin_is_wellformed():
    assert len(emrate.FIXED_TEMPLATE_SHA256) == 64
    assert set(emrate.FIXED_TEMPLATE_EXPECTED) == {"n_occurrences", "n_nonemission", "n_unique"}


# ─────────────────────────────────────────────────────────────────────────────
# Extraction mode — _subset_rows_for_target("fixedtext") + _load_fixed_template
# ─────────────────────────────────────────────────────────────────────────────


def test_subset_rows_fixedtext_replaces_all_rows_keeps_probe_ids():
    import issue833_extract_onpolicy as ex

    rows = [
        (0, "q0", "a real answer"),
        (1, "q1", ""),  # empty response STILL gets the template (plan v13 §4(b))
        (2, "q2", "seven wooden benches!"),  # emission row STILL kept — no filter
    ]
    kept = ex._subset_rows_for_target(
        "fixedtext",
        "fact/s__t0",
        rows,
        phrase=emrate.PINNED_PHRASE,
        floor=emrate.RETENTION_FLOOR,
        manifest_cells={},  # BYPASSED for this mode — must not be consulted
        indices_by_subset={},
        fixed_template="TPL",
    )
    assert kept == [(0, "q0", "TPL"), (1, "q1", "TPL"), (2, "q2", "TPL")]
    with pytest.raises(AssertionError):  # template not threaded = programming error
        ex._subset_rows_for_target(
            "fixedtext",
            "fact/s__t0",
            rows,
            phrase=emrate.PINNED_PHRASE,
            floor=emrate.RETENTION_FLOOR,
            manifest_cells={},
            indices_by_subset={},
        )


def test_subset_rows_legacy_modes_unchanged_by_fixedtext_kwarg():
    """The legacy 'all' path ignores the new kwarg entirely (byte-identical rows)."""
    import issue833_extract_onpolicy as ex

    rows = [(0, "q0", "a"), (1, "q1", ""), (2, "q2", "b")]
    kept = ex._subset_rows_for_target(
        "all",
        "fact/s__t0",
        rows,
        phrase=emrate.PINNED_PHRASE,
        floor=emrate.RETENTION_FLOOR,
        manifest_cells={},
        indices_by_subset={},
        fixed_template="TPL",
    )
    assert kept == [(0, "q0", "a"), (2, "q2", "b")]


def _write_pin(path: Path, template: str, sha: str | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "template": template,
                "sha256": sha if sha is not None else _sha(template),
                "n_occurrences": 3,
                "n_nonemission": 4,
                "n_unique": 2,
            }
        )
    )


def test_load_fixed_template_guards(tmp_path):
    import issue833_extract_onpolicy as ex

    tpl = "That's not really something I'd know."
    p = tmp_path / "fixed_template.json"
    _write_pin(p, tpl)
    pin = ex._load_fixed_template(p, expected_sha=_sha(tpl))
    assert pin["template"] == tpl

    # (a) self-consistency: recorded sha != sha256(template) → corrupted pin
    _write_pin(p, tpl, sha=_sha("something else"))
    with pytest.raises(RuntimeError, match="self-consistency FAIL"):
        ex._load_fixed_template(p, expected_sha=_sha(tpl))

    # (b) self-consistent but a FOREIGN pin (sha != the plan/code pin) → no silent re-pin
    _write_pin(p, tpl)
    with pytest.raises(RuntimeError, match="no silent re-pin"):
        ex._load_fixed_template(p, expected_sha="0" * 64)

    # (c) missing file → fail loud with the F0 pointer
    with pytest.raises(FileNotFoundError, match="fixed-template"):
        ex._load_fixed_template(tmp_path / "absent.json", expected_sha=_sha(tpl))


def test_load_fixed_template_default_expected_sha_is_code_pin(tmp_path):
    """With no override, the guard binds to emrate.FIXED_TEMPLATE_SHA256 — a
    synthetic self-consistent pin must be REJECTED (foreign template)."""
    import issue833_extract_onpolicy as ex

    p = tmp_path / "fixed_template.json"
    _write_pin(p, "a foreign template")
    with pytest.raises(RuntimeError, match="no silent re-pin"):
        ex._load_fixed_template(p)


# ─────────────────────────────────────────────────────────────────────────────
# Base-leg cross-source consistency check (real npz on tmp_path)
# ─────────────────────────────────────────────────────────────────────────────


def _write_cell_npz(root: Path, source: str, tcid: str, layer: int, v0, sha_rows) -> None:
    d = root / "fact" / f"{source}_seed42"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / f"{tcid}_L{layer}.npz",
        v0_onpolicy=np.asarray(v0, dtype=np.float32),
        v_plus_onpolicy=np.asarray(v0, dtype=np.float32) + 1.0,
        resp_sha256=np.array(sha_rows),
        probe_idx=np.arange(len(sha_rows), dtype=np.int64),
    )


def test_base_consistency_pass_and_fail(tmp_path):
    import issue833_extract_onpolicy as ex

    pin_sha = _sha("TPL")
    rng = np.random.default_rng(0)
    v0 = {("t0", 7): rng.standard_normal(8), ("t1", 7): rng.standard_normal(8)}
    for src in ("s1", "s2", "s3"):
        for (tcid, li), v in v0.items():
            _write_cell_npz(tmp_path, src, tcid, li, v, [pin_sha] * 3)
    summary = ex.check_fixedtext_base_consistency(tmp_path, expected_sha=pin_sha)
    assert summary["max_rel_l2"] == 0.0
    assert summary["n_npz"] == 6 and summary["n_groups"] == 2
    assert set(summary["per_group_rel_l2"]) == {"t0_L7", "t1_L7"}

    # perturb ONE source copy past the tolerance → extraction nondeterminism, STOP
    _write_cell_npz(tmp_path, "s3", "t1", 7, v0[("t1", 7)] * 1.5, [pin_sha] * 3)
    with pytest.raises(RuntimeError, match="cross-source consistency FAIL"):
        ex.check_fixedtext_base_consistency(tmp_path, expected_sha=pin_sha)


def test_base_consistency_rejects_non_template_rows(tmp_path):
    import issue833_extract_onpolicy as ex

    pin_sha = _sha("TPL")
    _write_cell_npz(tmp_path, "s1", "t0", 7, np.ones(4), [pin_sha, _sha("NOT-TPL")])
    with pytest.raises(RuntimeError, match="resp_sha256 rows != the template pin"):
        ex.check_fixedtext_base_consistency(tmp_path, expected_sha=pin_sha)


def test_base_consistency_empty_root_fails_loud(tmp_path):
    import issue833_extract_onpolicy as ex

    (tmp_path / "fact").mkdir()
    with pytest.raises(FileNotFoundError, match="no source cell dirs"):
        ex.check_fixedtext_base_consistency(tmp_path, expected_sha=_sha("TPL"))


# ─────────────────────────────────────────────────────────────────────────────
# F2 guards (pure — issue833_chain_rho_fixedtext)
# ─────────────────────────────────────────────────────────────────────────────


def test_f2_assert_key_cover():
    import issue833_chain_rho_fixedtext as f2

    want = {"fact/a__t0", "fact/a__t1"}
    f2.assert_key_cover(set(want), want, "ok")  # equal → no raise
    with pytest.raises(RuntimeError, match="1 missing"):
        f2.assert_key_cover({"fact/a__t0"}, want, "missing-case")
    with pytest.raises(RuntimeError, match="1 extra"):
        f2.assert_key_cover(want | {"fact/a__t2"}, want, "extra-case")


def test_f2_assert_families_sources():
    import issue833_chain_rho_fixedtext as f2

    keys = ["fact/s1__t0", "fact/s2__t1"]
    fams = ["famA", "famB"]
    f2.assert_families_sources(keys, fams, n_families=2, n_sources=2, label="ok")
    with pytest.raises(RuntimeError, match="families"):
        f2.assert_families_sources(keys, ["famA", "famA"], n_families=2, n_sources=2, label="x")
    with pytest.raises(RuntimeError, match="sources"):
        f2.assert_families_sources(
            ["fact/s1__t0", "fact/s1__t1"], fams, n_families=2, n_sources=2, label="x"
        )


def test_f2_load_template_pin_binds_to_code_pin(tmp_path):
    import issue833_chain_rho_fixedtext as f2

    p = tmp_path / "fixed_template.json"
    _write_pin(p, "a foreign template")  # self-consistent, wrong pin
    with pytest.raises(RuntimeError, match="template pin guard FAIL"):
        f2.load_template_pin(p)


def test_f2_load_fixedtext_stack_guards(tmp_path):
    """Real-body stack loader: coverage, 30-probe, and sha guards all fire."""
    import issue833_chain_rho_fixedtext as f2

    pin_sha = _sha("TPL")
    keys = ["fact/s1__t0"]
    # (a) 30 template rows, right sha → loads (shape asserted)
    _write_cell_npz(tmp_path, "s1", "t0", 7, np.ones(8), [pin_sha] * 30)
    vp, v0 = f2.load_fixedtext_stack(tmp_path, keys, 7, template_sha=pin_sha)
    assert vp.shape == (1, 8) and v0.shape == (1, 8)
    # (b) wrong probe count → STOP
    _write_cell_npz(tmp_path, "s1", "t0", 7, np.ones(8), [pin_sha] * 29)
    with pytest.raises(RuntimeError, match="!= 30"):
        f2.load_fixedtext_stack(tmp_path, keys, 7, template_sha=pin_sha)
    # (c) non-template sha row → STOP
    _write_cell_npz(tmp_path, "s1", "t0", 7, np.ones(8), [pin_sha] * 29 + [_sha("X")])
    with pytest.raises(RuntimeError, match="resp_sha256 rows != the template"):
        f2.load_fixedtext_stack(tmp_path, keys, 7, template_sha=pin_sha)
    # (d) a cell missing from the namespace → coverage STOP
    with pytest.raises(RuntimeError, match="key-set mismatch"):
        f2.load_fixedtext_stack(tmp_path, [*keys, "fact/s1__t9"], 7, template_sha=pin_sha)
