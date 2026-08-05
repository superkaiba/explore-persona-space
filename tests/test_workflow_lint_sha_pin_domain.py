"""Tests for ``workflow_lint --check-sha-pin-domain`` (#2079).

The check FAILs a whole-string 64-hex sha pin duplicated across >= 2
scripts/src modules when a site declares no content DOMAIN (the
#1776/#1491 wrong-domain class: a new module copies an INDEX-array digest
as a bare ``VAL_SHA256`` and a consumer asserts prompt-string digests
against it — an assert that can never pass on ANY input) or when sites
declare CONFLICTING domains (INDEX vs PROMPT).

Covers, per plan §4 item 5:

1. the exact #1491 pre-fix shape (``git show 9f43b03e43^``: a NEW module
   binding the real ``2e307fb2...`` hex as bare ``VAL_SHA256``) FAILs as an
   undeclared copy;
2. conflicting-domains FAIL (annotation vs binding-name vocabulary, one row
   per declared site; no allowlist escape);
3. agreeing annotations / binding-name tokens PASS — including the
   annotated multi-line paren-assignment shape
   (``NAME: Final[str] = (\\n    "<hex>"\\n)``) and the #1482 dict-key shape
   with a preceding-line annotation;
4. ``# SHA_PIN_DOMAIN_EXEMPT: <reason>`` exempts the SITE;
5. a single-module hex is ignored (even when repeated within the file);
6. a grandfathered ``(hex12, file)`` pair PASSes while the SAME hex in a
   NEW file FAILs (the #1491 propagation vector);
7. the live-tree invariant — zero rows on the current tree — plus the
   stale-grandfather ratchet: an entry no longer matching an undeclared
   cross-module pin site FAILs (the set shrinks, never silently grows) and
   every frozen entry keeps the frozen-experiment path shape (never a
   workflow-surface file);
8. the MUTATION-VISIBLE no-flags DISPATCH test (the
   ``test_check_jsonl_splitlines_bundled_in_no_flags`` pattern) — a direct
   call of the check function is NOT sufficient evidence of bundling.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402
from workflow_lint import (  # noqa: E402
    SHA_PIN_DOMAIN_GRANDFATHER,
    check_sha_pin_domain,
)

# The REAL founding pins (the #779 fixed_split INDEX-array digests the #1491
# pre-fix module copied as bare VAL_SHA256/TEST_SHA256).
VAL_HEX = "2e307fb2d1b74c82752d9460d131a3c1949860e9f0eefe6a82d15cee9f1e0613"
TEST_HEX = "b9377786b24bc9c1c360303fdb8fac86c0097d264479de1dca3c23dd1047d31d"

# Grandfather entries are frozen per-issue experiment scripts / experiment
# packages — NEVER a workflow-surface file (the JSONL allowlist shape rule).
GRANDFATHER_SHAPE_RE = re.compile(
    r"^(scripts/(issue|i\d)|src/explore_persona_space/(experiments|analysis)/)"
)


def _plant(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _run_on(
    monkeypatch,
    tmp_path: Path,
    grandfather: frozenset[tuple[str, str]] = frozenset(),
) -> list[str]:
    """Run the check against a tmp tree with the grandfather NEUTRALIZED.

    Without the override every real grandfather entry would read stale on
    the tmp tree and flood the result with grandfather-stale rows.
    """
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(wl, "SHA_PIN_DOMAIN_GRANDFATHER", frozenset(grandfather))
    return check_sha_pin_domain()


# --------------------------------------------------------------------------
# 1. the #1491 pre-fix shape: undeclared copy FAILs
# --------------------------------------------------------------------------


def test_1491_shape_undeclared_copy_fails(tmp_path, monkeypatch) -> None:
    # Producer module, annotated INDEX (the post-#2079 live-tree shape).
    _plant(
        tmp_path,
        "scripts/issue1776_contexts.py",
        f'# SHA_PIN_DOMAIN: INDEX\nVAL_400_SHA = "{VAL_HEX}"\n',
    )
    # The REAL pre-fix #1491 shape (git show 9f43b03e43^ line 101): a bare
    # undeclared copy of the INDEX-array digest in a NEW module.
    _plant(
        tmp_path,
        "scripts/issue1491_ladder_manifest.py",
        f'VAL_SHA256 = "{VAL_HEX}"\n',
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "sha-pin-domain/scripts/issue1491_ladder_manifest.py:1" in errors[0], errors
    assert "undeclared" in errors[0], errors
    assert VAL_HEX[:12] in errors[0], errors


def test_both_sites_undeclared_both_fail(tmp_path, monkeypatch) -> None:
    _plant(tmp_path, "scripts/a.py", f'A_SHA256 = "{VAL_HEX}"\n')
    _plant(tmp_path, "scripts/b.py", f'B_SHA256 = "{VAL_HEX}"\n')
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 2, errors
    assert any("scripts/a.py:1" in e for e in errors), errors
    assert any("scripts/b.py:1" in e for e in errors), errors


# --------------------------------------------------------------------------
# 2. conflicting domains FAIL (no allowlist escape)
# --------------------------------------------------------------------------


def test_conflicting_domains_fail(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/a.py",
        f'# SHA_PIN_DOMAIN: INDEX\nPIN_A = "{VAL_HEX}"\n',
    )
    # Domain declared via the binding-name vocabulary token (PROMPT).
    _plant(tmp_path, "scripts/b.py", f'TRACKS_PROMPT_SHA256 = "{VAL_HEX}"\n')
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 2, errors  # one row per DECLARED site
    assert all("conflicting" in e for e in errors), errors
    assert any("INDEX" in e and "PROMPT" in e for e in errors), errors


def test_conflict_rows_ignore_grandfather(tmp_path, monkeypatch) -> None:
    """A conflict has NO allowlist escape — grandfather pairs never mute it."""
    _plant(tmp_path, "scripts/a.py", f'# SHA_PIN_DOMAIN: INDEX\nPIN_A = "{VAL_HEX}"\n')
    _plant(tmp_path, "scripts/b.py", f'# SHA_PIN_DOMAIN: PROMPT\nPIN_B = "{VAL_HEX}"\n')
    errors = _run_on(
        monkeypatch,
        tmp_path,
        grandfather=frozenset({(VAL_HEX[:12], "scripts/a.py"), (VAL_HEX[:12], "scripts/b.py")}),
    )
    conflict_rows = [e for e in errors if "conflicting" in e]
    assert len(conflict_rows) == 2, errors


# --------------------------------------------------------------------------
# 3. agreeing declarations PASS (annotation, name vocab, multi-line, dict key)
# --------------------------------------------------------------------------


def test_agreeing_annotation_passes(tmp_path, monkeypatch) -> None:
    _plant(tmp_path, "scripts/a.py", f'# SHA_PIN_DOMAIN: INDEX\nPIN_A = "{VAL_HEX}"\n')
    _plant(tmp_path, "scripts/b.py", f'VAL_400_INDEX_SHA = "{VAL_HEX}"\n')
    assert _run_on(monkeypatch, tmp_path) == []


def test_annotated_multiline_paren_assignment_resolves(tmp_path, monkeypatch) -> None:
    """The critic-noted shape: ``NAME: Final[str] = (\\n    "<hex>"\\n)`` with
    the annotation above the assignment target resolves via the binding-line
    lookback — the hex line itself carries neither name nor comment."""
    _plant(
        tmp_path,
        "scripts/a.py",
        "from typing import Final\n"
        "\n"
        "# SHA_PIN_DOMAIN: INDEX\n"
        "VAL_SHA: Final[str] = (\n"
        f'    "{VAL_HEX}"\n'
        ")\n",
    )
    _plant(tmp_path, "scripts/b.py", f'# SHA_PIN_DOMAIN: INDEX\nPIN_B = "{VAL_HEX}"\n')
    assert _run_on(monkeypatch, tmp_path) == []


def test_multiline_name_vocab_resolves(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/a.py",
        f'VAL_400_INDEX_SHA = (\n    "{VAL_HEX}"\n)\n',
    )
    _plant(tmp_path, "scripts/b.py", f'# SHA_PIN_DOMAIN: INDEX\nPIN_B = "{VAL_HEX}"\n')
    assert _run_on(monkeypatch, tmp_path) == []


def test_dict_key_preceding_line_annotation_resolves(tmp_path, monkeypatch) -> None:
    """The #1482 shape: dict-entry pins annotated on the preceding line."""
    _plant(
        tmp_path,
        "scripts/a.py",
        "SPLIT_SHAS = {\n"
        "    # SHA_PIN_DOMAIN: INDEX\n"
        f'    "pinned_val_sha256": "{VAL_HEX}",\n'
        "}\n",
    )
    _plant(tmp_path, "scripts/b.py", f'# SHA_PIN_DOMAIN: INDEX\nPIN_B = "{VAL_HEX}"\n')
    assert _run_on(monkeypatch, tmp_path) == []


# --------------------------------------------------------------------------
# 4. SHA_PIN_DOMAIN_EXEMPT waives the site
# --------------------------------------------------------------------------


def test_exempt_comment_passes(tmp_path, monkeypatch) -> None:
    _plant(tmp_path, "scripts/a.py", f'# SHA_PIN_DOMAIN: INDEX\nPIN_A = "{VAL_HEX}"\n')
    _plant(
        tmp_path,
        "scripts/b.py",
        f'# SHA_PIN_DOMAIN_EXEMPT: deliberate cross-domain fixture copy\nCOPY = "{VAL_HEX}"\n',
    )
    assert _run_on(monkeypatch, tmp_path) == []


# --------------------------------------------------------------------------
# 5. single-module hexes are ignored
# --------------------------------------------------------------------------


def test_single_module_hex_ignored(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/solo.py",
        f'A = "{VAL_HEX}"\nB = "{VAL_HEX}"\nC = "{TEST_HEX}"\n',
    )
    assert _run_on(monkeypatch, tmp_path) == []


# --------------------------------------------------------------------------
# 6. grandfather: pair PASSes, same hex in a NEW file FAILs
# --------------------------------------------------------------------------


def test_grandfathered_pair_passes_and_new_file_fails(tmp_path, monkeypatch) -> None:
    _plant(tmp_path, "scripts/old.py", f'OLD_SHA256 = "{VAL_HEX}"\n')
    _plant(tmp_path, "scripts/new_copy.py", f'NEW_SHA256 = "{VAL_HEX}"\n')
    errors = _run_on(
        monkeypatch,
        tmp_path,
        grandfather=frozenset({(VAL_HEX[:12], "scripts/old.py")}),
    )
    assert len(errors) == 1, errors
    assert "scripts/new_copy.py:1" in errors[0], errors
    assert "grandfather-stale" not in errors[0], errors


def test_stale_grandfather_entry_fails_tmp_tree(tmp_path, monkeypatch) -> None:
    _plant(tmp_path, "scripts/a.py", f'# SHA_PIN_DOMAIN: INDEX\nPIN_A = "{VAL_HEX}"\n')
    _plant(tmp_path, "scripts/b.py", f'# SHA_PIN_DOMAIN: INDEX\nPIN_B = "{VAL_HEX}"\n')
    errors = _run_on(
        monkeypatch,
        tmp_path,
        grandfather=frozenset({("deadbeefdead", "scripts/gone.py")}),
    )
    assert len(errors) == 1, errors
    assert "grandfather-stale" in errors[0], errors
    assert "deadbeefdead" in errors[0], errors


# --------------------------------------------------------------------------
# 7. live-tree invariants
# --------------------------------------------------------------------------


def test_live_tree_passes() -> None:
    """Zero sha-pin-domain rows on the current tree: every duplicated hex is
    either declared (the #2079 INDEX annotations) or exactly covered by
    SHA_PIN_DOMAIN_GRANDFATHER, and no frozen entry is stale."""
    assert check_sha_pin_domain() == []


def test_stale_grandfather_entry_fails_live_tree(monkeypatch) -> None:
    """A frozen entry matching nothing FAILs the run — the grandfather can
    shrink, never silently grow (the test_live_trees_pass ratchet idiom)."""
    monkeypatch.setattr(
        wl,
        "SHA_PIN_DOMAIN_GRANDFATHER",
        SHA_PIN_DOMAIN_GRANDFATHER | {("deadbeefdead", "scripts/nonexistent_module.py")},
    )
    errors = check_sha_pin_domain()
    assert len(errors) == 1, errors
    assert "grandfather-stale" in errors[0], errors
    assert "deadbeefdead" in errors[0], errors


def test_grandfather_entries_keep_frozen_experiment_path_shape() -> None:
    offenders = [
        rel
        for _hex12, rel in sorted(SHA_PIN_DOMAIN_GRANDFATHER)
        if not GRANDFATHER_SHAPE_RE.match(rel)
    ]
    assert offenders == [], (
        f"SHA_PIN_DOMAIN_GRANDFATHER must hold frozen per-issue experiment "
        f"scripts/packages only — never a workflow-surface file: {offenders}"
    )


def test_grandfather_hex_prefixes_are_12_lowercase_hex() -> None:
    bad = [
        (h, rel)
        for h, rel in sorted(SHA_PIN_DOMAIN_GRANDFATHER)
        if not re.fullmatch(r"[0-9a-f]{12}", h)
    ]
    assert bad == [], f"grandfather keys must be hex[:12] prefixes: {bad}"


# --------------------------------------------------------------------------
# 8. no-flags bundling (mutation-visible dispatch test)
# --------------------------------------------------------------------------


def test_check_sha_pin_domain_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the check — deleting its
    ``or no_flags`` branch must fail this test (mutation-visible; the
    ``test_check_jsonl_splitlines_bundled_in_no_flags`` pattern). Other
    bundled checks contribute unrelated errors on the minimal tree, so the
    assertion keys on the check's own diagnostic token + offending path."""
    _plant(tmp_path, "scripts/producer.py", f'# SHA_PIN_DOMAIN: INDEX\nPIN = "{VAL_HEX}"\n')
    _plant(tmp_path, "scripts/new_copy.py", f'COPIED_SHA256 = "{VAL_HEX}"\n')
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(wl, "SHA_PIN_DOMAIN_GRANDFATHER", frozenset())
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert "sha-pin-domain/scripts/new_copy.py:1" in err and "undeclared" in err, (
        f"the sha-pin-domain diagnostic (naming new_copy.py) is missing from "
        f"the no-flags default run's stderr — the check is not bundled into "
        f"no_flags:\n{err}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
