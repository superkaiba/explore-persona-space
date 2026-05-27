"""Regression guard for the marker-form fix in launch_issue396.py (BF1 round 2).

The launcher must:

1. Carry the LEADING-SPACE marker form '` ※`' end-to-end so the trained model
   and every eval surface read the same Qwen-2.5 token id (83399). The bare
   '`※`' form tokenizes to id 63680, which the eval primitive does NOT read —
   training on one token while evaluating another silently breaks the
   headline DV.
2. Reject the bare-marker form loudly at launch time via
   ``assert_marker_token_id`` — the guard catches MARKER_TEXT re-edits,
   base-model swaps, and any future shell-quoting regression that strips the
   leading space.
3. Pass the marker through ``shlex.quote`` in ``build_cmd`` so the leading
   space survives the bash re-parse of the subprocess command string. The
   first regression (#396 code-review v1 round 1) was a bare-marker form
   that quietly stripped the leading space on the subprocess argv hop.

This test deliberately exercises behavior the implementer claimed without
mocking the Qwen tokenizer — the assertion needs to run against the real
vocabulary because the failure mode IS a tokenizer-specific id swap.
The test skips if the Qwen tokenizer is not downloaded (CPU-only dev VM
without HF cache primed).

Plan v2.3 §A4 + §10 Reproducibility Card pin the canonical marker to id
83399. Code-review v1 round 1 verdict (binding fix BF1) drove these tests.
"""

from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _import_launcher():
    """Load scripts/launch_issue396.py as a module without executing main()."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "launch_issue396", SCRIPTS_DIR / "launch_issue396.py"
    )
    assert spec is not None, "could not build module spec for launch_issue396.py"
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _have_qwen_tokenizer() -> bool:
    """Skip these tests if the Qwen-2.5 tokenizer isn't downloadable / cached."""
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=False)
        return True
    except Exception:
        return False


# ── BF1 part 1: MARKER_TEXT is the leading-space form ────────────────────────


def test_marker_text_is_leading_space_form():
    """MARKER_TEXT must equal ' ※' (leading space), NOT '※' (bare).

    The bare form tokenizes to a DIFFERENT Qwen-2.5 id (63680 vs 83399) and
    every eval surface in the #396 pipeline hardcodes id 83399. Train/eval
    token mismatch is a silent killer — this test pins the constant.
    """
    launcher = _import_launcher()
    assert launcher.MARKER_TEXT == " ※", (
        f"launch_issue396.MARKER_TEXT={launcher.MARKER_TEXT!r}; "
        "must be ' ※' (leading space) so it tokenizes to id 83399 on "
        "Qwen-2.5 (which is what every eval surface reads). Code-review v1 "
        "round 1 caught a bare-marker regression that broke the headline DV."
    )
    assert launcher.EXPECTED_MARKER_TOKEN_ID == 83399


# ── BF1 part 2: assert_marker_token_id catches the bare-marker regression ────


@pytest.mark.skipif(
    not _have_qwen_tokenizer(),
    reason="Qwen/Qwen2.5-7B-Instruct tokenizer not available (no HF cache)",
)
def test_assert_marker_token_id_passes_on_leading_space_form():
    """The launch-time assertion accepts the canonical ' ※' (id 83399)."""
    launcher = _import_launcher()
    launcher.assert_marker_token_id(marker_text=" ※", expected_id=83399)


@pytest.mark.skipif(
    not _have_qwen_tokenizer(),
    reason="Qwen/Qwen2.5-7B-Instruct tokenizer not available (no HF cache)",
)
def test_assert_marker_token_id_rejects_bare_form():
    """The launch-time assertion HARD-FAILS on the bare '※' (id 63680).

    This is the exact regression code-review v1 round 1 caught: launcher used
    `MARKER_TEXT = "※"` while every eval surface used `" ※"`. The assertion
    must raise SystemExit loudly so the next launcher invocation can't
    silently ship that mismatch.
    """
    launcher = _import_launcher()
    with pytest.raises(SystemExit) as excinfo:
        launcher.assert_marker_token_id(marker_text="※", expected_id=83399)
    # The error message must name the actual token-id mismatch (helps a
    # future debugger see what went wrong vs a bare "assertion failed").
    msg = str(excinfo.value)
    assert "83399" in msg, "error must mention the expected id 83399"
    assert "Qwen" in msg, "error must mention the tokenizer model"
    assert " ※" in msg, "error must show the canonical leading-space fix"


@pytest.mark.skipif(
    not _have_qwen_tokenizer(),
    reason="Qwen/Qwen2.5-7B-Instruct tokenizer not available (no HF cache)",
)
def test_assert_marker_token_id_rejects_wrong_expected_id():
    """If someone bumps EXPECTED_MARKER_TOKEN_ID without updating the eval
    primitive's hardcoded constant, the assertion must still fire."""
    launcher = _import_launcher()
    with pytest.raises(SystemExit):
        # Real marker, wrong expected id — the assertion is a contract check.
        launcher.assert_marker_token_id(marker_text=" ※", expected_id=99999)


# ── BF1 part 3: shell-quoting preserves the leading space in subprocess argv ──


def test_build_cmd_uses_shlex_quote_for_marker():
    """The build_cmd output must shell-quote MARKER_TEXT.

    Without quoting, bash strips the leading whitespace when re-parsing the
    command string and argparse sees `--marker-token` with value `※` (bare)
    instead of `' ※'`. The fix is `shlex.quote(MARKER_TEXT)`.
    """
    launcher = _import_launcher()
    cmd = launcher.build_cmd(source="accountant", gpu=0, pod="epm-issue-396")
    # The literal shlex-quoted form of " ※" is "' ※'" (single-quoted).
    quoted = shlex.quote(launcher.MARKER_TEXT)
    assert f"--marker-token {quoted}" in cmd, (
        f"build_cmd output is missing shlex-quoted marker. Got: {cmd!r}\n"
        f"Expected substring: '--marker-token {quoted}'"
    )


def test_build_cmd_preserves_leading_space_through_bash_reparse():
    """End-to-end: bash -c <build_cmd>  ->  argparse sees ' ※' (with leading space).

    This is the failure mode that motivated BF1: even with the correct
    MARKER_TEXT constant, unquoted shell interpolation strips the leading
    space en route to the subprocess. We use a minimal argparse stand-in
    (echo + Python repr) to confirm the argv shape downstream of bash.
    """
    launcher = _import_launcher()
    # Replace the actual run_leakage_experiment.py invocation with a Python
    # one-liner that just prints argv. We swap the substring rather than
    # rebuilding the command so the test exercises the REAL build_cmd output.
    real_cmd = launcher.build_cmd(source="accountant", gpu=0, pod="epm-issue-396")
    fake_cmd = real_cmd.replace(
        "uv run python scripts/archive/run_leakage_experiment.py",
        'python -c "import sys; print(repr(sys.argv))"',
    )

    result = subprocess.run(
        ["bash", "-c", fake_cmd],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"bash invocation failed: stderr={result.stderr!r}"
    # Look for the leading-space marker form in the argv repr. The Python
    # repr of [' ※'] embeds it as the literal string "' ※'" or "' ※'".
    # We assert on the unicode form since that's what tokenizer.encode reads.
    argv_line = result.stdout.strip()
    assert " ※" in argv_line, (
        f"bash re-parse stripped the leading space from the marker. "
        f"argv repr was: {argv_line!r}\n"
        f"This is the exact BF1 regression — build_cmd must use shlex.quote."
    )


# ── Sanity: the constants are consistent with the eval-side primitive ────────


def test_eval_primitive_marker_constants_match_launcher():
    """The eval scripts (eval_issue396_logprob, first_step_gradient_i396) use
    the SAME marker text the launcher trains on. A drift here would silently
    break the headline DV — same class of bug as BF1, different surface.
    """
    launcher = _import_launcher()
    # Read constants from the eval scripts by importing them as plain modules.
    # We tolerate ModuleNotFoundError on the heavier dependencies the eval
    # scripts pull in only at call-site (vllm, psutil) — those imports happen
    # inside function bodies, not at module load.
    import importlib

    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    eval_mod = importlib.import_module("eval_issue396_logprob")
    first_step_mod = importlib.import_module("first_step_gradient_i396")
    assert eval_mod.MARKER_TEXT == launcher.MARKER_TEXT, (
        f"eval_issue396_logprob.MARKER_TEXT={eval_mod.MARKER_TEXT!r} "
        f"differs from launcher.MARKER_TEXT={launcher.MARKER_TEXT!r}"
    )
    assert first_step_mod.MARKER_TEXT == launcher.MARKER_TEXT, (
        f"first_step_gradient_i396.MARKER_TEXT={first_step_mod.MARKER_TEXT!r} "
        f"differs from launcher.MARKER_TEXT={launcher.MARKER_TEXT!r}"
    )
