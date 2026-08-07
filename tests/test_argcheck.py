"""Behavior tests for ``orchestrate.argcheck`` (task #2176 layer (b), T2 batch).

Synthetic ``tmp_path`` fixtures — no dependency on any unmerged branch ref.
T2.1 reproduces the #2163 helper-escape shape structurally (the phase
function calls a helper; the HELPER holds the unregistered ``args``
reference), so a future narrowing of the whole-module scan scope is
test-breaking, not just wrong-looking in review. T2.2-T2.5 pin the four
measured false-positive classes (47 confirmed FPs across 927 candidate
files: 23 ``dest=``, 9 subparsers, 15 runtime-assignment, plus the
fingerprint-identified imported-parser-builder class).

Round 2 (concern ``argcheck-exact-dest-derivation``) adds the exact-dest
pins: the multi-long false-negative regression (a revert to the round-1
all-candidates superset — or to single-first-string derivation — is
test-breaking), the short+long / short-only false-positive guards, the
dynamic-option permissive fallback, the hyphenated-positional verbatim
dest, and the ``del args.<attr>`` reference pin.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined


def _write(tmp_path: Path, name: str, source: str) -> Path:
    """Write a dedented synthetic module under ``tmp_path`` and return its path."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_whole_module_scope_catches_helper_escape(tmp_path):
    """T2.1 — the whole-module pin (the #2163 regression shape).

    ``args.figures_out`` lives in ``_fig_dir``, a helper the phase function
    calls — NOT in the ``PHASES`` function body itself. A per-function scan
    scope (the #2163 first-version defect) would pass this module; the
    whole-module scope must flag it.
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def _fig_dir(args):
            return args.figures_out  # never registered — the #2163 escape

        def phase_figures(args):
            return _fig_dir(args)

        PHASES = {"figures": phase_figures}

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--phase")
            args = ap.parse_args()
            PHASES[args.phase](args)
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(driver)
    assert "figures_out" in str(exc.value)


def test_dest_kwarg_overrides_flag_name(tmp_path):
    """T2.2 — FP class 1 (23/47 measured): explicit ``dest=`` rename passes."""
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--no-upload", dest="upload", action="store_false")
            args = ap.parse_args()
            if args.upload:
                print("uploading")
        """,
    )
    assert_args_attributes_defined(driver)


def test_dest_kwarg_is_an_override_not_an_addition(tmp_path):
    """``dest=`` REPLACES the flag-derived name (argparse semantics).

    ``args.no_upload`` under ``add_argument("--no-upload", dest="upload")``
    is a genuine runtime ``AttributeError`` and stays flagged.
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--no-upload", dest="upload", action="store_false")
            args = ap.parse_args()
            if args.no_upload:
                print("wrong attribute")
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(driver)
    assert "no_upload" in str(exc.value)


def test_add_subparsers_dest_defines_attr(tmp_path):
    """T2.3 — FP class 2 (9/47 measured): ``add_subparsers(dest=...)`` passes."""
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            sub = ap.add_subparsers(dest="cmd")
            sub.add_parser("run")
            args = ap.parse_args()
            print(args.cmd)
        """,
    )
    assert_args_attributes_defined(driver)


def test_runtime_store_defines_attr(tmp_path):
    """T2.4 — FP class 3 (15/47 measured): Store then Load passes."""
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--phases")
            args = ap.parse_args()
            args.phases_set = set((args.phases or "").split(","))
            print(args.phases_set)
        """,
    )
    assert_args_attributes_defined(driver)


def test_imported_parser_builder_needs_both_files(tmp_path):
    """T2.5 — FP class 4: parser partly built by a second module.

    Both files passed -> the unioned DEFINED set covers the imported
    registration. The driver file ALONE fails naming the gap — this is why
    the signature is varargs.
    """
    shared = _write(
        tmp_path,
        "shared_mod.py",
        """
        def _add_common_args(ap):
            ap.add_argument("--shared-flag")
        """,
    )
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        import shared_mod

        def main():
            ap = argparse.ArgumentParser()
            shared_mod._add_common_args(ap)
            args = ap.parse_args()
            print(args.shared_flag)
        """,
    )
    assert_args_attributes_defined(driver, shared)
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(driver)
    assert "shared_flag" in str(exc.value)


def test_extra_defined_escape(tmp_path):
    """T2.6a — residual FPs route through ``extra_defined``, never silently."""
    driver = _write(
        tmp_path,
        "driver.py",
        """
        def run(args):
            return args.magic
        """,
    )
    with pytest.raises(SystemExit):
        assert_args_attributes_defined(driver)
    assert_args_attributes_defined(driver, extra_defined=("magic",))


def test_set_defaults_kwargs_define_attrs(tmp_path):
    """T2.6b — ``set_defaults(**kw)`` kwarg names enter the DEFINED set."""
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.set_defaults(verbose=True)
            args = ap.parse_args()
            print(args.verbose)
        """,
    )
    assert_args_attributes_defined(driver)


def test_positional_argument_name_defines_attr(tmp_path):
    """T2.6c — a bare positional argument name defines itself."""
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("input_file")
            args = ap.parse_args()
            print(args.input_file)
        """,
    )
    assert_args_attributes_defined(driver)


def test_multi_option_alias_read_is_flagged(tmp_path):
    """R2.1 — the ``argcheck-exact-dest-derivation`` regression pin.

    argparse derives EXACTLY ONE dest per call — the first LONG option —
    so under ``add_argument("--new-name", "--old-name")`` the dest is
    ``new_name`` and ``args.old_name`` is a guaranteed runtime
    ``AttributeError``. Round 1's all-candidates superset put ``old_name``
    in DEFINED and PASSed this module (the reviewer-confirmed false
    negative); exact derivation must flag it. This test FAILS against the
    round-1 implementation by construction.
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--new-name", "--old-name", default=None)
            args = ap.parse_args()
            print(args.old_name)
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(driver)
    assert "old_name" in str(exc.value)

    reader_of_real_dest = _write(
        tmp_path,
        "driver_ok.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--new-name", "--old-name", default=None)
            args = ap.parse_args()
            print(args.new_name)
        """,
    )
    assert_args_attributes_defined(reader_of_real_dest)


def test_short_plus_long_defines_long_dest_only(tmp_path):
    """R2.2 — the false-positive guard that motivated the r1 superset.

    ``("-n", "--dry-run")``: the long option wins, so ``args.dry_run``
    passes — AND the short option contributes NO dest: ``args.n`` is a
    runtime ``AttributeError`` and must be flagged.
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("-n", "--dry-run", action="store_true")
            args = ap.parse_args()
            if args.dry_run:
                print("dry run")
        """,
    )
    assert_args_attributes_defined(driver)

    short_reader = _write(
        tmp_path,
        "driver_short.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("-n", "--dry-run", action="store_true")
            args = ap.parse_args()
            if args.n:
                print("wrong attribute")
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(short_reader)
    # "n" is the sole gap, so it sits alone between the header and "scanned:".
    assert "never defined: n\n" in str(exc.value)


def test_short_only_option_defines_short_dest(tmp_path):
    """R2.3 — no long option: the first (short) option string is the dest."""
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("-v", action="count")
            args = ap.parse_args()
            print(args.v)
        """,
    )
    assert_args_attributes_defined(driver)


def test_dynamic_option_strings_keep_permissive_fallback(tmp_path):
    """R2.4 — non-constant option string => the documented permissive path.

    With ``add_argument(FLAG_VAR, "--aaa", "--bbb")`` the runtime value of
    ``FLAG_VAR`` could displace EITHER constant as the first long option,
    so the dest is not statically computable: every constant candidate
    enters DEFINED (never a false positive on a dynamic parser). A future
    "exact everywhere" change that ranked only the constants would flag
    ``bbb`` and break this pin.
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        EXTRA_FLAG = "--zzz"

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument(EXTRA_FLAG, "--aaa", "--bbb")
            args = ap.parse_args()
            print(args.aaa, args.bbb)
        """,
    )
    assert_args_attributes_defined(driver)


def test_hyphenated_positional_keeps_hyphen_verbatim(tmp_path):
    """R2.5 — r1 NIT: argparse keeps the hyphen in a positional's dest.

    ``add_argument("src-dir")`` yields dest ``src-dir`` (verified against
    the live interpreter) — NOT ``src_dir`` — so the underscore-alias read
    is a genuine runtime ``AttributeError`` and must be flagged. (The
    verbatim ``src-dir`` name can never match an attribute read, so
    keeping it in DEFINED is harmless.)
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("src-dir")
            args = ap.parse_args()
            print(args.src_dir)
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(driver)
    assert "src_dir" in str(exc.value)


def test_del_of_undefined_attr_is_flagged(tmp_path):
    """R2.6 — r1 NIT: ``del args.<attr>`` requires the attribute to exist.

    A Del-context attribute on the namespace is a REFERENCE (a ``del`` of
    an undefined attr raises ``AttributeError`` at runtime, like a read);
    a Store-defined attr may be deleted freely (the set-based check has no
    flow ordering, consistent with the documented AugAssign stance).
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        def run(args):
            del args.ghost
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(driver)
    assert "ghost" in str(exc.value)

    store_then_del = _write(
        tmp_path,
        "driver_ok.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--keep")
            args = ap.parse_args()
            args.scratch = 1
            del args.scratch
            print(args.keep)
        """,
    )
    assert_args_attributes_defined(store_then_del)


def test_multi_gap_message_names_every_gap(tmp_path):
    """T2.6d — the SystemExit message names EVERY gap and the scanned file."""
    driver = _write(
        tmp_path,
        "driver.py",
        """
        def run(args):
            return args.alpha + args.beta
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(driver)
    msg = str(exc.value)
    assert "alpha" in msg
    assert "beta" in msg
    assert "driver.py" in msg
