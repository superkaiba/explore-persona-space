"""Tests for ``workflow_lint.check_dotenv_before_hf_import`` (#745).

The check FAILs a ``scripts/*.py`` file that uses the BARE python-dotenv
``load_dotenv`` AND imports ``huggingface_hub`` WITHOUT first importing the
project wrapper ``explore_persona_space.orchestrate.env.load_dotenv`` — the
bare dotenv misses the worktree ``.env`` and sets no env, so the HF Hub upload
accelerators (``HF_XET_HIGH_PERFORMANCE`` / ``HF_HUB_ENABLE_HF_TRANSFER``)
never get their setdefault and large uploads crawl.

Covers: (a) a clean script PASSES; (b) bare-dotenv + huggingface_hub FAILS
with a helpful message; (c) a properly-wrapped script PASSES; (d) the
``# DOTENV_LINT_EXEMPT: <reason>`` waiver works.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import check_dotenv_before_hf_import  # noqa: E402


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# (a) clean scripts PASS
# --------------------------------------------------------------------------


def test_clean_script_no_dotenv_no_hf_passes(tmp_path: Path) -> None:
    """A script that uses neither dotenv nor huggingface_hub is never flagged."""
    _write(tmp_path, "clean.py", "import os\nimport json\n\nprint(os.getcwd())\n")
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


def test_bare_dotenv_without_hf_passes(tmp_path: Path) -> None:
    """Bare dotenv alone (no huggingface_hub) is OUT OF SCOPE — only the
    dotenv+hf combination is the #745 anti-pattern."""
    _write(
        tmp_path,
        "dotenv_only.py",
        "from dotenv import load_dotenv\n\nload_dotenv()\nprint('hi')\n",
    )
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


def test_hf_without_bare_dotenv_passes(tmp_path: Path) -> None:
    """huggingface_hub alone (no bare dotenv) is not flagged."""
    _write(
        tmp_path,
        "hf_only.py",
        "from huggingface_hub import HfApi\n\nprint(HfApi())\n",
    )
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (b) bare-dotenv + huggingface_hub FAILS with a helpful message
# --------------------------------------------------------------------------


def test_bare_dotenv_plus_hf_module_top_fails(tmp_path: Path) -> None:
    """The worst-case #745 shape: bare dotenv + a module-top huggingface_hub
    import, no project wrapper. FAILS, anchored at the bare-dotenv line."""
    p = _write(
        tmp_path,
        "offender.py",
        "from dotenv import load_dotenv\n"
        "from huggingface_hub import HfApi\n\n"
        "load_dotenv()\n"
        "api = HfApi()\n",
    )
    errors = check_dotenv_before_hf_import(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(p) in errors[0]
    assert ":1:" in errors[0]  # anchored at the bare-dotenv import (line 1)
    assert "#745" in errors[0]
    assert "orchestrate.env.load_dotenv" in errors[0]
    assert "DOTENV_LINT_EXEMPT" in errors[0]


def test_bare_dotenv_plus_in_function_hf_import_fails(tmp_path: Path) -> None:
    """The in-function huggingface_hub import (the issue617/issue658 shape)
    is detected by ast.walk too — a deferred import is still the anti-pattern."""
    _write(
        tmp_path,
        "lazy_hf.py",
        "from dotenv import load_dotenv\n\n"
        "load_dotenv()\n\n"
        "def upload():\n"
        "    from huggingface_hub import HfApi\n"
        "    return HfApi()\n",
    )
    errors = check_dotenv_before_hf_import(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


def test_plain_import_dotenv_plus_hf_fails(tmp_path: Path) -> None:
    """A plain ``import dotenv`` (then ``dotenv.load_dotenv()``) + hf import is
    the same anti-pattern — the bare-dotenv signal is the module, not the
    from-import form."""
    _write(
        tmp_path,
        "plain_import.py",
        "import dotenv\nimport huggingface_hub\n\ndotenv.load_dotenv()\n",
    )
    errors = check_dotenv_before_hf_import(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


def test_huggingface_hub_submodule_import_fails(tmp_path: Path) -> None:
    """A ``from huggingface_hub.utils import ...`` submodule import counts as
    touching huggingface_hub (the issue651 shape imports both)."""
    _write(
        tmp_path,
        "submod.py",
        "from dotenv import load_dotenv\n"
        "from huggingface_hub.utils import HfHubHTTPError\n\n"
        "load_dotenv()\n"
        "print(HfHubHTTPError)\n",
    )
    errors = check_dotenv_before_hf_import(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (c) a properly-wrapped script PASSES
# --------------------------------------------------------------------------


def test_project_wrapper_plus_hf_passes(tmp_path: Path) -> None:
    """Using the project wrapper as the dotenv source — NOT bare dotenv — with
    a huggingface_hub import is the sanctioned shape and is never flagged."""
    _write(
        tmp_path,
        "wrapped.py",
        "from explore_persona_space.orchestrate.env import load_dotenv\n\n"
        "load_dotenv()\n\n"
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n",
    )
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


def test_both_wrapper_and_bare_dotenv_present_passes(tmp_path: Path) -> None:
    """If the project wrapper is imported ANYWHERE in the file, the check
    defers (the author has the sanctioned source available) even if a bare
    dotenv import also appears — the wrapper presence is the sanctioned signal."""
    _write(
        tmp_path,
        "both.py",
        "from explore_persona_space.orchestrate.env import load_dotenv\n"
        "from dotenv import load_dotenv as _bare  # legacy alias\n"
        "from huggingface_hub import HfApi\n\n"
        "load_dotenv()\n"
        "api = HfApi()\n",
    )
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


# --------------------------------------------------------------------------
# (d) the # DOTENV_LINT_EXEMPT: <reason> waiver works
# --------------------------------------------------------------------------


def test_waiver_on_import_line_suppresses(tmp_path: Path) -> None:
    """A ``# DOTENV_LINT_EXEMPT: <reason>`` (reason ≥ 10 chars) on the bare
    dotenv import line suppresses the FAIL."""
    _write(
        tmp_path,
        "waived_inline.py",
        "from dotenv import load_dotenv  # DOTENV_LINT_EXEMPT: deliberate legacy bare dotenv\n"
        "from huggingface_hub import HfApi\n\n"
        "load_dotenv()\n"
        "api = HfApi()\n",
    )
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


def test_waiver_on_preceding_line_suppresses(tmp_path: Path) -> None:
    """The waiver also works on the immediately preceding non-blank line."""
    _write(
        tmp_path,
        "waived_above.py",
        "# DOTENV_LINT_EXEMPT: this script never uploads, only reads downloads\n"
        "from dotenv import load_dotenv\n"
        "from huggingface_hub import HfApi\n\n"
        "load_dotenv()\n"
        "api = HfApi()\n",
    )
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


def test_waiver_with_too_short_reason_still_fails(tmp_path: Path) -> None:
    """A waiver whose reason is shorter than the minimum does NOT suppress —
    the waiver is a justification, not a token bypass."""
    _write(
        tmp_path,
        "short_reason.py",
        "from dotenv import load_dotenv  # DOTENV_LINT_EXEMPT: x\n"
        "from huggingface_hub import HfApi\n\n"
        "load_dotenv()\n"
        "api = HfApi()\n",
    )
    errors = check_dotenv_before_hf_import(scripts_dir=tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


def test_unparseable_file_is_skipped(tmp_path: Path) -> None:
    """A scripts/ file that does not parse is its own (separate) problem; the
    check stays silent rather than crashing."""
    _write(tmp_path, "broken.py", "def f(:\n   pass\n")
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


def test_missing_scripts_dir_returns_empty(tmp_path: Path) -> None:
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path / "nope") == []


def test_live_scripts_tree_passes() -> None:
    """The real scripts/ tree (including the three migrated #745 scripts) must
    pass the check — this is the no-flags-default-run invariant."""
    assert check_dotenv_before_hf_import() == []
