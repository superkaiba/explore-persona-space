"""Wiring tests for issue #293 §3.

Every data-gen script in ``scripts/`` MUST:
1. Import ``upload_dataset_directory`` from
   ``explore_persona_space.orchestrate.hub`` (single shared helper).
2. Expose ``--no-upload`` on its argparse with default ``False`` so production
   runs upload to HF Hub by default. The flag exists only for dry-runs.

Some in-scope scripts have required positional/named args (e.g. language-pair
flags) or have pre-existing import-time dependencies that prevent
``importlib`` exec-import in the test env. The test handles both:

- We try ``parse_known_args([])`` rather than ``parse_args([])`` so that
  required args don't cause ``SystemExit(2)`` to mask the assertion.
- If exec-import fails (the script imports a module not currently installed
  in the test env), we fall back to subprocess ``--help`` + text-grep for
  ``--no-upload`` (per plan §3.6 risk #6 escape hatch).

The ``test_no_upload_default_false`` parametrized assertion catches a
``default=True`` typo that would silently disable uploads in CI/production.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

SCRIPTS = [
    "generate_wrong_answers",
    "generate_trait_transfer_data_v2",
    "generate_leakage_data",
    "generate_a3_data",
    "generate_a3b_data",
    "generate_sdf_neutral_ai",
    "generate_sdf_variants",
    "build_dpo_midtrain_data",
    "build_language_inversion_data",
    "build_language_inversion_data_v2",
]


def _try_load(script_name: str):
    """Exec-import a script as a module. Returns None on ImportError."""
    path = SCRIPTS_DIR / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, path)
    assert spec is not None and spec.loader is not None, f"could not load spec for {script_name}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = mod
    try:
        spec.loader.exec_module(mod)
    except (ImportError, ModuleNotFoundError):
        return None
    return mod


def _help_text(script_name: str) -> str:
    """Subprocess --help fallback for scripts whose imports break in the test env."""
    path = SCRIPTS_DIR / f"{script_name}.py"
    proc = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    # argparse exits 0 on --help; if exit 1/2, the script failed before parsing.
    return proc.stdout + proc.stderr


@pytest.mark.parametrize("script_name", SCRIPTS)
def test_imports_helper(script_name: str):
    """Every in-scope script must reference the shared upload helper."""
    src = (SCRIPTS_DIR / f"{script_name}.py").read_text()
    assert "upload_dataset_directory" in src, (
        f"{script_name} does not reference the shared upload_dataset_directory helper"
    )


@pytest.mark.parametrize("script_name", SCRIPTS)
def test_no_upload_default_false(script_name: str):
    """The --no-upload flag must default to False so production runs upload.

    Strategy:
    1. Try exec-import + parse_known_args([]). If the parser exposes
       ``no_upload`` with ``False``, pass.
    2. If exec-import fails (broken upstream import), fall back to
       text inspection: source must contain ``"--no-upload"`` AND the
       argparse stanza must show ``default=False`` or use ``action="store_true"``
       (which always defaults to False).
    """
    mod = _try_load(script_name)
    src = (SCRIPTS_DIR / f"{script_name}.py").read_text()

    if mod is not None:
        if hasattr(mod, "build_arg_parser"):
            parser = mod.build_arg_parser()
        elif hasattr(mod, "_build_arg_parser"):
            parser = mod._build_arg_parser()
        else:
            pytest.fail(
                f"{script_name} must expose build_arg_parser() (or _build_arg_parser()) "
                f"returning an argparse.ArgumentParser"
            )
        # Some scripts have required args; grab the --no-upload default
        # straight off the parser action list to bypass parser.parse_args().
        no_upload_action = next(
            (a for a in parser._actions if a.dest == "no_upload"),
            None,
        )
        assert no_upload_action is not None, (
            f"{script_name}: parser does not define --no-upload (actions: "
            f"{[a.dest for a in parser._actions]})"
        )
        assert no_upload_action.default is False, (
            f"{script_name}: --no-upload default is {no_upload_action.default!r}, expected False"
        )
        return

    # Exec-import failed: fall back to source inspection.
    assert "--no-upload" in src, f"{script_name}: source does not declare --no-upload"
    # Find the --no-upload argparse stanza and verify its action/default.
    # Pattern: parser.add_argument("--no-upload", ..., action="store_true", default=False, ...)
    # ``action="store_true"`` implies default False, so accept either signal.
    stanza = re.search(
        r'add_argument\(\s*["\']--no-upload["\'].*?\)',
        src,
        flags=re.DOTALL,
    )
    assert stanza, f"{script_name}: cannot locate --no-upload add_argument stanza"
    body = stanza.group(0)
    has_store_true = 'action="store_true"' in body or "action='store_true'" in body
    has_default_false = "default=False" in body
    assert has_store_true or has_default_false, (
        f"{script_name}: --no-upload stanza must use action='store_true' or default=False; "
        f"got: {body}"
    )
