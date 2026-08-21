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

Round 3 (round-2 review NIT) adds the dynamic-``dest=`` pin: a
present-but-non-constant ``dest=`` kwarg routes to the permissive path,
exactly like non-constant option strings.

Task #2188 adds the splat (``**kwargs``) pins: P1-P3 are per-site bite
pins (fail against the pre-fix, splat-skipping checker — the revert-demo
set), T1 pins the unresolvable-splat degrade path (teeth + diagnostic),
and T2 pins the EXCLUSIVE-USE abstain rule on rebound / mutated /
alias-escaping module dicts.

Task #2261 adds the call-arity bind-pass tests (``test_bind_*``): the
incident regression (A1/MF-1 — repeated same-target imports RESOLVE),
per-shape must-FAILs with the registry mutation check (MF-2), the waiver
and arming pins (MF-4), and the fleet-census positive-coverage gate
(MF-3) with its resolver-free lexical denominator.

#2261 review round 2 closes the two reconciler-persisted concerns with
fail-pre-fix negative fixtures: comment-anchored waiver recognition
(``argcheck-waiver-comment-validation`` — a sentinel inside a call
argument, or with an empty / sub-floor reason, never waives) and the
exception-alias / match-capture shadow classes
(``argcheck-shadow-binder-coverage`` — ``ExceptHandler.name``,
``MatchAs.name``, ``MatchStar.name``, ``MatchMapping.rest`` recorded as
non-import bindings so shadowed calls abstain as noted skips).
"""

from __future__ import annotations

import ast
import sys
import textwrap
from collections import Counter
from pathlib import Path

import pytest

from explore_persona_space.orchestrate import argcheck
from explore_persona_space.orchestrate.argcheck import (
    assert_args_attributes_defined,
    assert_helper_call_shapes_bind,
    collect_helper_call_census,
)


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


def test_dynamic_dest_kwarg_takes_permissive_fallback(tmp_path):
    """R3.1 — round-2 NIT: a present-but-non-constant ``dest=`` is dynamic.

    Under ``add_argument("-n", "--dry-run", dest=DEST_NAME)`` the runtime
    dest is whatever ``DEST_NAME`` holds — unknowable statically, strictly
    LESS resolvable than a missing ``dest=`` — so the call takes the same
    permissive path as non-constant option strings: every constant option
    string contributes its derived name, and ``args.n`` (which exact
    derivation would NOT define — the first long option wins) must not be
    flagged. Round 2 fell through to exact derivation here (defining
    ``{dry_run}`` only), a false positive on ``args.n``; this pin FAILS
    against the round-2 implementation.
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        DEST_NAME = "dry_run"

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("-n", "--dry-run", dest=DEST_NAME, action="store_true")
            args = ap.parse_args()
            print(args.n, args.dry_run)
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


def test_add_argument_splat_takes_permissive_fallback(tmp_path):
    """P1 (#2188) — an ``add_argument`` ``**`` splat routes to the permissive path.

    The splat could carry ``dest=`` — strictly LESS resolvable than a
    present-but-non-constant ``dest=`` (the R3.1 shape above), so it must
    not be handled more strictly. Two option strings (``-n``, ``--dry-run``)
    make the pin discriminating: pre-fix the splat was silently SKIPPED and
    exact derivation defined ``{dry_run}`` only, flagging ``args.n`` on
    correct code (the false-positive class #2188 removes); permissive
    derivation defines ``{n, dry_run}``. The INLINE dict literal is
    deliberate — it also pins that site 1 goes permissive rather than
    resolve-and-fall-through-to-exact (a future refactor resolving site-1
    splats into exact derivation would flag ``args.n`` and break this pin,
    forcing a deliberate re-decision).
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("-n", "--dry-run", **{"action": "store_true"})
            args = ap.parse_args()
            print(args.n, args.dry_run)
        """,
    )
    assert_args_attributes_defined(driver)


def test_add_subparsers_splat_dest_resolved_from_module_dict(tmp_path):
    """P2 (#2188) — ``add_subparsers(**SUB_KW)`` resolves ``dest`` from the module dict.

    Fixture 1: the EXCLUSIVE-USE rule resolves the module-level dict
    literal, so the splat-carried ``dest: "cmd"`` enters DEFINED (pre-fix
    the splat was skipped and ``args.cmd`` raised on correct code).
    Fixture 2 (teeth): a RESOLVED splat WITHOUT ``dest`` contributes
    nothing — parity with argparse (``add_subparsers()`` without ``dest``
    stores no attribute, verified against the live interpreter), so
    ``args.cmd`` stays flagged.
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        SUB_KW = {"dest": "cmd", "required": True}

        def main():
            ap = argparse.ArgumentParser()
            sub = ap.add_subparsers(**SUB_KW)
            sub.add_parser("run")
            args = ap.parse_args()
            print(args.cmd)
        """,
    )
    assert_args_attributes_defined(driver)

    no_dest = _write(
        tmp_path,
        "driver_no_dest.py",
        """
        import argparse

        SUB_KW2 = {"required": True}

        def main():
            ap = argparse.ArgumentParser()
            sub = ap.add_subparsers(**SUB_KW2)
            sub.add_parser("run")
            args = ap.parse_args()
            print(args.cmd)
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(no_dest)
    assert "never defined: cmd\n" in str(exc.value)


def test_set_defaults_splat_keys_resolved_from_module_dict(tmp_path):
    """P3 (#2188) — ``set_defaults(**DEFAULTS)`` keys resolve from the module dict.

    The single most natural convention-adoption shape (a shared
    module-level defaults dict); pre-fix the splat was skipped and BOTH
    reads raised ``SystemExit`` on correct code.
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        DEFAULTS = {"verbose": False, "workers": 4}

        def main():
            ap = argparse.ArgumentParser()
            ap.set_defaults(**DEFAULTS)
            args = ap.parse_args()
            print(args.verbose, args.workers)
        """,
    )
    assert_args_attributes_defined(driver)


def test_unresolvable_splat_keeps_teeth_and_names_splat_in_error(tmp_path):
    """T1 (#2188) — an unresolvable splat contributes nothing and is named.

    ``set_defaults(**build_defaults())`` is not statically resolvable, so
    the check keeps its teeth: ``args.ghost`` stays flagged, and the
    failure message names the splat site (``<file>:<lineno> <method>``)
    with the ``extra_defined`` pointer. Loose substring asserts — the
    exact wording is not load-bearing. ``extra_defined=("ghost",)``
    clears it.
    """
    driver = _write(
        tmp_path,
        "driver.py",
        """
        import argparse

        def build_defaults():
            return {"verbose": False}

        def main():
            ap = argparse.ArgumentParser()
            ap.set_defaults(**build_defaults())
            args = ap.parse_args()
            print(args.ghost)
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(driver)
    msg = str(exc.value)
    assert "never defined: ghost\n" in msg
    assert "set_defaults" in msg
    assert "splat" in msg
    assert "extra_defined" in msg
    assert_args_attributes_defined(driver, extra_defined=("ghost",))


def test_rebound_mutated_or_escaping_splat_dict_abstains(tmp_path):
    """T2 (#2188) — the EXCLUSIVE-USE guard abstains on every escaping shape.

    Fixture A (rebind): a second binding of ``DEFAULTS`` fails the
    sole-binding condition. Fixture B (method-receiver Load):
    ``DEFAULTS.update(extra)`` places the name in a Load context outside a
    sanctioned splat position. Fixture C (alias-mediated removal):
    ``B = DEFAULTS`` places the name in a non-sanctioned Load (alias RHS)
    — the binding and the ``pop`` are both on ``B``, so ``DEFAULTS``
    itself looks pristine.

    Fixture C is the bite pin for the v3 EXCLUSIVE-USE tightening, and its
    bite is ANALYTIC — against the REJECTED v2 mutation-enumeration rule,
    not against the pre-fix code: under v2 (Subscript Store/Del plus a
    known-mutator method list on the resolved name ONLY), ``DEFAULTS`` in
    fixture C is sole-bound, never rebound, and never mutated BY NAME, so
    a v2 resolver returns ``{a, k}``, both reads pass, and ``args.k``
    becomes a SILENT runtime ``AttributeError`` — this test, which asserts
    ``SystemExit``, FAILS under the v2 rule. Under v3 the alias RHS is a
    non-sanctioned Load -> abstain -> both reads flagged (loud,
    ``extra_defined``-escapable). Against the pre-fix code
    (``d046b6f635``) fixture C ALSO passes (a ``set_defaults`` splat
    contributed nothing), which is why T2 is NOT part of the #2188 revert
    demo — its bite is the v2-rule counterfactual, not a pre-fix behavior
    delta. The mutating-callee shape (``_finalize(DEFAULTS)``) is caught
    by the same call-argument-Load clause; the alias fixture is the pin.
    """
    rebind = _write(
        tmp_path,
        "driver_rebind.py",
        """
        import argparse

        DEFAULTS = {"a": 1}

        def make():
            return {}

        DEFAULTS = make()

        def main():
            ap = argparse.ArgumentParser()
            ap.set_defaults(**DEFAULTS)
            args = ap.parse_args()
            print(args.a)
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(rebind)
    assert "never defined: a\n" in str(exc.value)

    mutated = _write(
        tmp_path,
        "driver_mutated.py",
        """
        import argparse

        DEFAULTS = {"a": 1}
        extra = {"b": 2}
        DEFAULTS.update(extra)

        def main():
            ap = argparse.ArgumentParser()
            ap.set_defaults(**DEFAULTS)
            args = ap.parse_args()
            print(args.a)
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(mutated)
    assert "never defined: a\n" in str(exc.value)

    aliased = _write(
        tmp_path,
        "driver_aliased.py",
        """
        import argparse

        DEFAULTS = {"a": 1, "k": 2}
        B = DEFAULTS
        B.pop("k")

        def main():
            ap = argparse.ArgumentParser()
            ap.set_defaults(**DEFAULTS)
            args = ap.parse_args()
            print(args.a, args.k)
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(aliased)
    assert "never defined: a, k\n" in str(exc.value)


# ---------------------------------------------------------------------------
# Task #2261: call-arity bind pass (test_bind_*)
# ---------------------------------------------------------------------------


def _line_of(path: Path, needle: str) -> int:
    """1-based line number of the first fixture line containing ``needle``."""
    for i, text in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if needle in text:
            return i + 1
    raise AssertionError(f"needle {needle!r} not found in {path}")


def test_bind_incident_shape_fails_naming_line_and_param(tmp_path):
    """A1: the #2223 incident shape fails, naming the site line + missing param."""
    driver = _write(
        tmp_path,
        "driver.py",
        """
        from explore_persona_space.orchestrate import hub

        def _persist(at, prefix):
            hub._upload(at, f"{prefix}/analysis_tensors", repo_type="dataset")

        def phase_upload(args):
            _persist(args.at, args.prefix)
        """,
    )
    line = _line_of(driver, "hub._upload(")
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(driver)
    msg = str(exc.value)
    assert "hub._upload" in msg
    assert f"driver.py:{line}" in msg
    assert "path_in_repo" in msg


def test_bind_repeated_same_target_imports_incident_fails(tmp_path, capsys):
    """MF-1: repeated same-target imports (incl. function-local) RESOLVE, never abstain.

    Mirrors the real pre-fix #2223 layout (blob lines 715/740/2497: ``hub``
    bound three times, all the same target); the incident call must FAIL
    with 0 skips — the sole-binding rule would have abstained here.
    """
    driver = _write(
        tmp_path,
        "driver_mf1.py",
        """
        from explore_persona_space.orchestrate import hub

        def _stage_topics():
            from explore_persona_space.orchestrate import hub
            return hub

        def _persist(at, prefix):
            from explore_persona_space.orchestrate import hub
            hub._upload(at, f"{prefix}/analysis_tensors", repo_type="dataset")

        def phase_upload(args):
            _persist(args.at, args.prefix)
        """,
    )
    line = _line_of(driver, "hub._upload(")
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(driver)
    msg = str(exc.value)
    assert f"driver_mf1.py:{line}" in msg
    assert "path_in_repo" in msg
    assert "0 skipped" in capsys.readouterr().out


def test_bind_all_binding_calls_pass(tmp_path, capsys):
    """A2/MF-2 count pin: 4 registered calls written => census reads exactly 4 bound."""
    driver = _write(
        tmp_path,
        "driver_ok.py",
        """
        from explore_persona_space.orchestrate import hub
        from explore_persona_space.orchestrate.hub import stage_hub_file
        from huggingface_hub import hf_hub_download

        def phase_all(args):
            hub._upload(args.path, "repo-id", "dataset", "prefix/file.json")
            stage_hub_file("repo-id", "prefix/file.json", args.target)
            hub.retry_transient(lambda: 1, what="stage")
            hf_hub_download(repo_id="r", filename="f")
        """,
    )
    assert_helper_call_shapes_bind(driver)  # must not raise
    assert "argcheck-bind: 4 bound, 0 degraded, 0 skipped across 1 file(s)" in (
        capsys.readouterr().out
    )


def test_bind_whole_module_scope_catches_helper_escape(tmp_path):
    """A4 + durability pin: the offending call sits in a helper the phase calls."""
    driver = _write(
        tmp_path,
        "driver_helper.py",
        """
        from explore_persona_space.orchestrate import hub

        def _fig_upload(args):
            hub._upload(args.figures_out)

        def phase_figures(args):
            _fig_upload(args)
        """,
    )
    line = _line_of(driver, "hub._upload(")
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(driver)
    msg = str(exc.value)
    assert f"driver_helper.py:{line}" in msg
    assert "hub._upload" in msg


def test_bind_lambda_wrapped_inner_call_is_bound(tmp_path, capsys):
    """A3: a lambda-wrapped inner ``HfApi().upload_file`` call is itself bound."""
    bad = _write(
        tmp_path,
        "driver_lambda_bad.py",
        """
        from explore_persona_space.orchestrate.hub import retry_transient
        from huggingface_hub import HfApi

        def phase_up(args):
            retry_transient(lambda: HfApi().upload_file("x"), what="upload")
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(bad)
    msg = str(exc.value)
    assert "HfApi().upload_file" in msg
    assert "positional" in msg  # upload_file is keyword-only: 1 positional cannot bind

    good = _write(
        tmp_path,
        "driver_lambda_good.py",
        """
        from explore_persona_space.orchestrate.hub import retry_transient
        from huggingface_hub import HfApi

        def phase_up(args):
            retry_transient(
                lambda: HfApi().upload_file(
                    path_or_fileobj="x", path_in_repo="p", repo_id="r"
                ),
                what="upload",
            )
        """,
    )
    capsys.readouterr()
    assert_helper_call_shapes_bind(good)  # must not raise
    assert "argcheck-bind: 2 bound, 0 degraded, 0 skipped" in capsys.readouterr().out


def test_bind_keyword_only_no_false_positive(tmp_path):
    """A5: required kw-only ``what`` — present passes, absent is a TRUE positive."""
    ok = _write(
        tmp_path,
        "driver_kwok.py",
        """
        from explore_persona_space.orchestrate.hub import retry_transient

        def phase(args, job):
            retry_transient(job, what="stage")
        """,
    )
    assert_helper_call_shapes_bind(ok)  # must not raise

    bad = _write(
        tmp_path,
        "driver_kwbad.py",
        """
        from explore_persona_space.orchestrate.hub import retry_transient

        def phase(args, job):
            retry_transient(job)
        """,
    )
    line = _line_of(bad, "retry_transient(job)")
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(bad)
    msg = str(exc.value)
    assert f"driver_kwbad.py:{line} retry_transient" in msg
    assert "'what'" in msg


def test_bind_callee_var_kwargs_absorbs(tmp_path, monkeypatch):
    """A5: a callee whose own signature carries ``**kwargs`` absorbs extra keywords."""
    _write(
        tmp_path,
        "argcheck_bind_double_mod.py",
        """
        def take_anything(a, **kwargs):
            return a, kwargs
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        argcheck, "_BIND_FUNCTIONS", frozenset({("argcheck_bind_double_mod", "take_anything")})
    )
    driver = _write(
        tmp_path,
        "driver_double.py",
        """
        from argcheck_bind_double_mod import take_anything

        def phase(args):
            take_anything(1, extra=2, more=3)
        """,
    )
    try:
        assert_helper_call_shapes_bind(driver)  # must not raise: **kwargs absorbs
        census = collect_helper_call_census(driver)
        assert len(census.bound) == 1 and not census.failures and not census.skipped
    finally:
        # r2 teardown nit: the bind pass imported the tmp_path module; do not
        # leak it into later tests' interpreter state.
        sys.modules.pop("argcheck_bind_double_mod", None)


def test_bind_call_site_splat_degrades_to_bind_partial(tmp_path, capsys):
    """A5: a call-site splat degrades to bind_partial; named kwargs still checked."""
    ok = _write(
        tmp_path,
        "driver_splat_ok.py",
        """
        from explore_persona_space.orchestrate import hub

        def phase(args, extra):
            hub._upload(*extra)
        """,
    )
    assert_helper_call_shapes_bind(ok)  # must not raise
    out = capsys.readouterr().out
    assert "argcheck-bind: 0 bound, 1 degraded, 0 skipped" in out
    assert "degraded:" in out and "bind_partial" in out

    bad = _write(
        tmp_path,
        "driver_splat_bad.py",
        """
        from explore_persona_space.orchestrate import hub

        def phase(args, extra):
            hub._upload(*extra, bogus_kw=1)
        """,
    )
    line = _line_of(bad, "bogus_kw=1")
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(bad)
    msg = str(exc.value)
    assert "hub._upload" in msg
    assert f"driver_splat_bad.py:{line}" in msg
    assert "bogus_kw" in msg


def test_bind_import_alias_resolves(tmp_path):
    """A5 aliasing: ``import ... as`` forms resolve to the same installed target."""
    bad = _write(
        tmp_path,
        "driver_alias_bad.py",
        """
        from explore_persona_space.orchestrate.hub import _upload as up

        def phase(args):
            up("only_one")
        """,
    )
    line = _line_of(bad, 'up("only_one")')
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(bad)
    msg = str(exc.value)
    assert f"driver_alias_bad.py:{line} up" in msg
    assert "missing a required argument" in msg
    assert "_upload" in msg  # the installed: line names the canonical target

    ok = _write(
        tmp_path,
        "driver_alias_ok.py",
        """
        import explore_persona_space.orchestrate.hub as h

        def phase(args, a, b, c, d):
            h._upload(a, b, c, d)
        """,
    )
    assert_helper_call_shapes_bind(ok)  # must not raise


def test_bind_shadowed_name_abstains_with_note(tmp_path, capsys):
    """A3/A5: a genuine shadow (non-import binding) is a NOTED skip, never silent."""
    driver = _write(
        tmp_path,
        "driver_shadow.py",
        """
        from explore_persona_space.orchestrate.hub import stage_hub_file

        def stage_hub_file(repo_id):
            return repo_id

        def phase(args):
            stage_hub_file("r")
        """,
    )
    assert_helper_call_shapes_bind(driver)  # must not raise (abstains, does not bind)
    out = capsys.readouterr().out
    assert "argcheck-bind: 0 bound, 0 degraded, 1 skipped" in out
    assert "skipped:" in out and "stage_hub_file" in out and "shadow" in out


def test_bind_hfapi_var_uniform_binding_resolves(tmp_path):
    """S4: ``api = HfApi()`` under uniform binding resolves; kw-only misuse fails."""
    driver = _write(
        tmp_path,
        "driver_s4.py",
        """
        from huggingface_hub import HfApi

        def phase_a(args):
            api = HfApi()
            api.upload_file("pos")

        def phase_b(args):
            api = HfApi()
            api.repo_info("some/repo")
        """,
    )
    line = _line_of(driver, 'api.upload_file("pos")')
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(driver)
    msg = str(exc.value)
    assert f"driver_s4.py:{line} api.upload_file" in msg
    assert "positional" in msg
    assert "repo_info" not in msg  # the well-formed sibling call bound cleanly


def test_bind_hfapi_var_nonuniform_binding_noted_skip(tmp_path, capsys):
    """S4 abstain: a receiver also bound as a parameter is a NOTED skip, no bind."""
    driver = _write(
        tmp_path,
        "driver_s4_mixed.py",
        """
        from huggingface_hub import HfApi

        def phase_a(args):
            api = HfApi()
            api.upload_file("pos")

        def phase_b(api):
            api.upload_folder("x")
        """,
    )
    assert_helper_call_shapes_bind(driver)  # must not raise
    out = capsys.readouterr().out
    assert "argcheck-bind: 0 bound, 0 degraded, 2 skipped" in out
    assert "not uniformly HfApi()" in out


def test_bind_nonexistent_hub_helper_fails_loud(tmp_path):
    """The #606 class: a nonexistent helper on a registered module is a FAILURE."""
    driver = _write(
        tmp_path,
        "driver_nohelper.py",
        """
        from explore_persona_space.orchestrate import hub

        def phase(args):
            hub.no_such_helper_xyz(1)
        """,
    )
    line = _line_of(driver, "no_such_helper_xyz")
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(driver)
    msg = str(exc.value)
    assert f"driver_nohelper.py:{line}" in msg
    assert "no_such_helper_xyz" in msg
    assert "nonexistent" in msg


def test_bind_hf_hub_download_bad_call_fails(tmp_path):
    """MF-2 per-function must-FAIL for the ``hf_hub_download`` registry entry."""
    # Unexpected-keyword shape: required args present, so the bogus kw is the failure.
    driver = _write(
        tmp_path,
        "driver_dl.py",
        """
        from huggingface_hub import hf_hub_download

        def phase(args):
            hf_hub_download("some/repo", "file.json", bogus_kw=1)
        """,
    )
    line = _line_of(driver, "bogus_kw=1")
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(driver)
    msg = str(exc.value)
    assert f"driver_dl.py:{line} hf_hub_download" in msg
    assert "bogus_kw" in msg

    # Missing-required shape: Signature.bind names the missing param first.
    missing = _write(
        tmp_path,
        "driver_dl_missing.py",
        """
        from huggingface_hub import hf_hub_download

        def phase(args):
            hf_hub_download(bogus_kw=1)
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(missing)
    assert "repo_id" in str(exc.value)


def test_bind_snapshot_download_bad_call_fails(tmp_path):
    """MF-2 per-function must-FAIL for the ``snapshot_download`` registry entry."""
    driver = _write(
        tmp_path,
        "driver_sd.py",
        """
        from huggingface_hub import snapshot_download

        def phase(args):
            snapshot_download("some/repo", bogus_kw=1)
        """,
    )
    line = _line_of(driver, "bogus_kw=1")
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(driver)
    msg = str(exc.value)
    assert f"driver_sd.py:{line} snapshot_download" in msg
    assert "bogus_kw" in msg


def test_bind_registry_entries_load_bearing(tmp_path, monkeypatch):
    """MF-2 mutation check: removing a registry entry flips its must-FAIL fixture to inert.

    Requires the call-time registry read (plan #2261 section 4.2 / assumption
    24): the ``_BIND_*`` constants are looked up per invocation, so a
    monkeypatched-out entry makes the SAME fixture produce zero resolved
    sites (no bind, no skip, no failure).
    """
    cases = [
        (
            "case_hub_mod.py",
            """
            from explore_persona_space.orchestrate import hub

            def phase(args):
                hub._upload("only_one")
            """,
            "_BIND_MODULES",
            frozenset(),
        ),
        (
            "case_hfapi.py",
            """
            from huggingface_hub import HfApi

            def phase(args):
                HfApi().upload_file("pos")
            """,
            "_BIND_CLASSES",
            frozenset(),
        ),
        (
            "case_dl.py",
            """
            from huggingface_hub import hf_hub_download

            def phase(args):
                hf_hub_download(bogus_kw=1)
            """,
            "_BIND_FUNCTIONS",
            frozenset({("huggingface_hub", "snapshot_download")}),
        ),
        (
            "case_sd.py",
            """
            from huggingface_hub import snapshot_download

            def phase(args):
                snapshot_download(bogus_kw=1)
            """,
            "_BIND_FUNCTIONS",
            frozenset({("huggingface_hub", "hf_hub_download")}),
        ),
    ]
    for fname, source, const, patched in cases:
        driver = _write(tmp_path, fname, source)
        with pytest.raises(SystemExit):  # intact registry: the bad call raises
            assert_helper_call_shapes_bind(driver)
        with monkeypatch.context() as m:
            m.setattr(argcheck, const, patched)
            census = collect_helper_call_census(driver)  # entry removed: inert
            assert census.bound == [] and census.failures == [] and census.skipped == [], fname


def test_bind_waiver_comment_suppresses(tmp_path, capsys):
    """MF-4: ARGCHECK_BIND_EXEMPT is LINE-grained — waives its call, never the file.

    The unwaived call sits DIRECTLY below the trailing-waiver code line, so
    this also pins the no-leak property: a trailing waiver on a CODE line
    must not cover the next line's call (only a comment-ONLY preceding line
    carries the waiver downward).
    """
    driver = _write(
        tmp_path,
        "driver_waiver.py",
        """
        from explore_persona_space.orchestrate import hub

        def phase(args):
            hub.stage_hub_prefix("repo")  # ARGCHECK_BIND_EXEMPT: legacy shim kept deliberately
            hub.retry_transient(job)
        """,
    )
    line = _line_of(driver, "hub.retry_transient(job)")
    with pytest.raises(SystemExit) as exc:
        assert_helper_call_shapes_bind(driver)
    msg = str(exc.value)
    assert f"driver_waiver.py:{line} hub.retry_transient" in msg
    assert "'what'" in msg
    assert "stage_hub_prefix" not in msg  # the waived site is NOT in the raised message
    out = capsys.readouterr().out
    assert "waived:" in out and "legacy shim kept deliberately" in out

    # The immediately-preceding non-blank comment line also carries the waiver.
    preceding = _write(
        tmp_path,
        "driver_waiver_prev.py",
        """
        from explore_persona_space.orchestrate import hub

        def phase(args):
            # ARGCHECK_BIND_EXEMPT: known-legacy call kept for parity
            hub.stage_hub_file("repo")
        """,
    )
    capsys.readouterr()
    assert_helper_call_shapes_bind(preceding)  # waived => must not raise
    assert "known-legacy call kept for parity" in capsys.readouterr().out


def test_bind_pass_runs_from_assert_args_attributes_defined(tmp_path, capsys):
    """Arming pin: the EXISTING entry point runs the bind pass; argparse raises first."""
    # (a) the existing entry point catches a bind failure with zero driver edits
    bad = _write(
        tmp_path,
        "driver_bind_armed.py",
        """
        import argparse

        from explore_persona_space.orchestrate import hub

        def phase(args):
            hub._upload(args.src)

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--src")
            args = ap.parse_args()
            phase(args)
        """,
    )
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(bad)
    assert "helper call shape(s) do not bind" in str(exc.value)

    # (b) argparse gaps raise FIRST, byte-identical message, bind pass never runs
    gap = _write(
        tmp_path,
        "driver_gap_first.py",
        """
        def run(args):
            return args.ghost
        """,
    )
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        assert_args_attributes_defined(gap)
    msg = str(exc.value)
    assert msg.startswith("argcheck: args attribute(s) referenced but never defined: ghost")
    assert "argcheck-bind" not in msg
    assert "argcheck-bind" not in capsys.readouterr().out

    # (c) a clean argparse-only fixture returns None; stdout carries the census line
    clean = _write(
        tmp_path,
        "driver_clean.py",
        """
        import argparse

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--src")
            args = ap.parse_args()
            print(args.src)
        """,
    )
    capsys.readouterr()
    assert assert_args_attributes_defined(clean) is None
    assert "argcheck-bind: 0 bound, 0 degraded, 0 skipped" in capsys.readouterr().out


def test_bind_unregistered_receivers_ignored(tmp_path, capsys):
    """FN-6 pin: arbitrary receivers + attribute chains are INERT — no binds, no skip-spam."""
    driver = _write(
        tmp_path,
        "driver_inert.py",
        """
        from huggingface_hub import HfApi

        class Client:
            def __init__(self):
                self.api = HfApi()

            def push(self):
                self.api.upload_file("pos")

        def phase(args, obj):
            obj.method("x")
        """,
    )
    census = collect_helper_call_census(driver)
    assert census.bound == [] and census.skipped == [] and census.failures == []
    assert_helper_call_shapes_bind(driver)  # must not raise
    assert "argcheck-bind: 0 bound, 0 degraded, 0 skipped" in capsys.readouterr().out


_HUB_MODULE = "explore_persona_space.orchestrate.hub"
_BIND_ADOPTION_LITERAL = "assert_args_attributes_defined"
# Co-passed `<mod>.__file__` extra args, keyed (adopter basename, module alias) -> repo-relative
# path. Plan #2261 section 6: any UNMAPPED extra-arg shape fails the census test loud, forcing
# a table update instead of silently narrowing the armed population.
_CO_PASSED_FILE_MAP = {
    ("issue2225_fu1_analysis.py", "pa"): "scripts/issue2225_analysis.py",
}


def _lexical_import_names(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    """Names lexically bound to the hub module / registered symbols / the HfApi class."""
    hub_names: set[str] = set()
    symbol_names: set[str] = set()
    class_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _HUB_MODULE and alias.asname:
                    hub_names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                bound = alias.asname or alias.name
                if f"{node.module}.{alias.name}" == _HUB_MODULE:
                    hub_names.add(bound)
                elif node.module == _HUB_MODULE or (
                    node.module == "huggingface_hub"
                    and alias.name in ("hf_hub_download", "snapshot_download")
                ):
                    symbol_names.add(bound)
                elif node.module == "huggingface_hub" and alias.name == "HfApi":
                    class_names.add(bound)
    return hub_names, symbol_names, class_names


def _lexical_receiver_names(tree: ast.AST, class_names: set[str]) -> set[str]:
    """Variable names assigned (Assign/AnnAssign) from a registered constructor call."""
    receivers: set[str] = set()
    for node in ast.walk(tree):
        pairs: list[tuple[list[ast.expr], ast.expr]] = []
        if isinstance(node, ast.Assign):
            pairs.append((list(node.targets), node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            pairs.append(([node.target], node.value))
        for targets, value in pairs:
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in class_names
            ):
                for tgt in targets:
                    if isinstance(tgt, ast.Name):
                        receivers.add(tgt.id)
    return receivers


def _lexical_registered_call_count(path: Path) -> int:
    """Resolver-free lexical count of registered-surface call sites in one file.

    The INDEPENDENT denominator for the fleet census (plan #2261 section 7):
    recognizes the four call shapes from a plain import scan — no uniformity
    machinery, no signature binding — so a resolver bug that silently drops
    sites cannot also shrink this count.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hub_names, symbol_names, class_names = _lexical_import_names(tree)
    attr_receivers = hub_names | _lexical_receiver_names(tree, class_names)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in attr_receivers:
                count += 1
        elif (isinstance(func, ast.Name) and func.id in symbol_names) or (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id in class_names
        ):
            count += 1
    return count


def test_bind_fleet_census_positive_coverage():
    """MF-2 + MF-3 fleet census gate (= plan #2261 section 10 R3, a pre-merge gate).

    Enumerates the COMPLETE armed population at run time and asserts POSITIVE
    coverage: zero failures/waivers/skips/degrades, bound_total equal to the
    resolver-free lexical denominator, and the plan-time per-class floors.
    Floors are the plan-time measured values on a frozen fleet (monotone
    under fleet growth); a legitimate floor break is updated WITH A NAMED
    DELTA in the updating commit. NO wall-time assertion by design (the
    plan's ~17 s census wall measured 60-79 s under fleet loadavg ~130).
    """
    repo = Path(__file__).resolve().parents[1]
    script_files = sorted((repo / "scripts").glob("*.py"))
    adopters = [p for p in script_files if _BIND_ADOPTION_LITERAL in p.read_text(encoding="utf-8")]
    src_files = sorted((repo / "src" / "explore_persona_space").rglob("*.py"))
    adopters += [
        p
        for p in src_files
        if p.name != "argcheck.py" and _BIND_ADOPTION_LITERAL in p.read_text(encoding="utf-8")
    ]
    population: dict[Path, None] = {p.resolve(): None for p in adopters}
    for adopter in adopters:
        tree = ast.parse(adopter.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else None)
            )
            if name != "assert_args_attributes_defined":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id == "__file__":
                    continue
                if (
                    isinstance(arg, ast.Attribute)
                    and arg.attr == "__file__"
                    and isinstance(arg.value, ast.Name)
                ):
                    mapped = _CO_PASSED_FILE_MAP.get((adopter.name, arg.value.id))
                    if mapped is None:
                        pytest.fail(
                            f"unmapped co-passed __file__ arg '{arg.value.id}.__file__' in"
                            f" {adopter} - update _CO_PASSED_FILE_MAP"
                        )
                    population[(repo / mapped).resolve()] = None
                    continue
                pytest.fail(f"unmapped extra-arg shape in {adopter}: {ast.dump(arg)[:120]}")
    files = sorted(population)
    census = collect_helper_call_census(*files)
    assert census.failures == [], [
        (f.site.path, f.site.lineno, f.site.label, f.error) for f in census.failures
    ]
    assert census.waived == []
    assert census.skipped == [], [(s.path, s.lineno, s.label, s.reason) for s in census.skipped]
    assert census.degraded == []
    bound_total = len(census.bound)
    lexical_total = sum(_lexical_registered_call_count(p) for p in files)
    assert bound_total == lexical_total, (bound_total, lexical_total)
    assert bound_total >= 195, bound_total
    adopters_with_calls = len({site.path for site in census.bound})
    assert adopters_with_calls >= 59, adopters_with_calls
    by_shape = Counter(site.shape for site in census.bound)
    assert by_shape["S1"] >= 96, dict(by_shape)
    assert by_shape["S2"] >= 81, dict(by_shape)
    assert by_shape["S3"] >= 10, dict(by_shape)
    assert by_shape["S4"] >= 8, dict(by_shape)
    per_fn = Counter(site.target for site in census.bound)
    assert per_fn["huggingface_hub.hf_hub_download"] >= 16, dict(per_fn)
    assert per_fn["huggingface_hub.snapshot_download"] >= 1, dict(per_fn)


# ---------------------------------------------------------------------------
# #2261 review round 2 — the two reconciler-persisted concerns. Each fixture
# fails against the round-1 code (raw-substring waiver matching; ExceptHandler
# / Match captures invisible to the module-wide binding collector) and pins
# the r2 fix: comment-anchored waiver recognition (tokenize.COMMENT tokens +
# the workflow_lint `#\s*TOKEN\s*:\s*(.+?)\s*$` shape, reason >= 10 chars) and
# exception-/pattern-bound names recorded as non-import bindings.
# ---------------------------------------------------------------------------


def test_bind_waiver_token_in_call_argument_never_waives(tmp_path):
    """concern argcheck-waiver-comment-validation, negative fixture (a).

    The sentinel as a STRING or IDENTIFIER argument of the failing call is
    not a comment — both calls stay in ``census.failures`` with
    ``census.waived == []`` (round 1 waived the string form with reason
    ``'")'``, defeating the gate)."""
    driver = _write(
        tmp_path,
        "driver_waiver_arg.py",
        """
        from explore_persona_space.orchestrate import hub

        def phase(args):
            hub._upload("ARGCHECK_BIND_EXEMPT")
            hub._upload(ARGCHECK_BIND_EXEMPT)
        """,
    )
    census = collect_helper_call_census(driver)
    assert census.waived == []
    assert len(census.failures) == 2
    assert {f.site.label for f in census.failures} == {"hub._upload"}
    with pytest.raises(SystemExit):
        assert_helper_call_shapes_bind(driver)


def test_bind_waiver_empty_or_short_reason_does_not_waive(tmp_path):
    """concern argcheck-waiver-comment-validation, negative fixture (b).

    A comment-anchored sentinel with an EMPTY reason does not waive; nor
    does a sub-floor reason (< 10 chars — the WANDB_INTENTIONALLY_DISABLED
    / CVD_PIN_EXEMPT / UPLOAD_AS_FILE_EXEMPT convention,
    scripts/workflow_lint.py)."""
    cases = [
        (
            "driver_waiver_empty.py",
            """
            from explore_persona_space.orchestrate import hub

            def phase(args):
                # ARGCHECK_BIND_EXEMPT:
                hub.stage_hub_file("repo")
            """,
        ),
        (
            "driver_waiver_short.py",
            """
            from explore_persona_space.orchestrate import hub

            def phase(args):
                hub.stage_hub_file("repo")  # ARGCHECK_BIND_EXEMPT: x
            """,
        ),
    ]
    for fname, source in cases:
        driver = _write(tmp_path, fname, source)
        census = collect_helper_call_census(driver)
        assert census.waived == [], fname
        assert len(census.failures) == 1, fname


def test_bind_except_alias_shadow_abstains(tmp_path):
    """concern argcheck-shadow-binder-coverage: ``except ... as hub``.

    An exception alias rebinding a registered import name is a recorded
    non-import binding — the call through it becomes a noted shadow skip
    (0 bound), never resolved against the hub module (round 1 resolved it
    and emitted a spurious bind FAIL)."""
    driver = _write(
        tmp_path,
        "driver_except_shadow.py",
        """
        from explore_persona_space.orchestrate import hub

        def phase(args):
            try:
                pass
            except Exception as hub:
                hub._upload(args.src)
        """,
    )
    assert_helper_call_shapes_bind(driver)  # abstains: must not raise
    census = collect_helper_call_census(driver)
    assert census.bound == [] and census.failures == []
    assert len(census.skipped) == 1
    assert "genuine shadow" in census.skipped[0].reason


def test_bind_match_capture_shadows_abstain(tmp_path):
    """concern argcheck-shadow-binder-coverage: pattern captures.

    ``MatchMapping.rest`` (``case {**hub}``), ``MatchAs.name``
    (``case hub``), and ``MatchStar.name`` (``case [*hub]``) each rebind
    the registered alias via a string AST field — recorded as non-import
    bindings, so every shadowed call abstains as a noted skip (0 bound)."""
    cases = [
        (
            "driver_match_mapping_shadow.py",
            """
            from explore_persona_space.orchestrate import hub

            def phase(args):
                match args.cfg:
                    case {**hub}:
                        hub._upload(args.src)
            """,
        ),
        (
            "driver_match_as_shadow.py",
            """
            from explore_persona_space.orchestrate import hub

            def phase(args):
                match args.mode:
                    case hub:
                        hub._upload(args.src)
            """,
        ),
        (
            "driver_match_star_shadow.py",
            """
            from explore_persona_space.orchestrate import hub

            def phase(args):
                match args.items:
                    case [*hub]:
                        hub._upload(args.src)
            """,
        ),
    ]
    for fname, source in cases:
        driver = _write(tmp_path, fname, source)
        census = collect_helper_call_census(driver)
        assert census.bound == [] and census.failures == [], fname
        assert len(census.skipped) == 1, fname
        assert "genuine shadow" in census.skipped[0].reason, fname
