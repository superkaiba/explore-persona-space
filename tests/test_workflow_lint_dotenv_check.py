"""Tests for ``workflow_lint.check_dotenv_before_hf_import`` (#745).

The check FAILs a ``scripts/*.py`` file that uses the BARE python-dotenv
``load_dotenv`` AND imports ``huggingface_hub`` WITHOUT first importing the
project wrapper ``explore_persona_space.orchestrate.env.load_dotenv`` — the
bare dotenv misses the worktree ``.env`` and sets no env, so the HF Hub upload
accelerators (``HF_XET_HIGH_PERFORMANCE`` / ``HF_HUB_ENABLE_HF_TRANSFER``)
never get their setdefault and large uploads crawl.

Covers: (a) a clean script PASSES; (b) bare-dotenv + huggingface_hub FAILS
with a helpful message; (c) a properly-wrapped script PASSES; (d) the
``# DOTENV_LINT_EXEMPT: <reason>`` waiver works; (e) the #745-round-2
IMPORT-ORDER arm — a module-top huggingface_hub import that precedes the
dotenv/env setup FAILS even with the wrapper present (the constants freeze at
import time), with the correct order / in-function import / shell-exports-only
shapes passing and the waiver suppressing.
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
    defers on ARM 1 (the author has the sanctioned source available) even if a
    bare dotenv import also appears — the wrapper presence is the sanctioned
    signal. The huggingface_hub import sits AFTER the load_dotenv() call so the
    correct ordering also clears ARM 2 (the env is set before the constants
    freeze); the wrapper-before-hf-before-call order is the ARM-2 offender
    covered by test_wrapper_import_before_hf_with_load_dotenv_call_after_fails."""
    _write(
        tmp_path,
        "both.py",
        "from explore_persona_space.orchestrate.env import load_dotenv\n"
        "from dotenv import load_dotenv as _bare  # legacy alias\n\n"
        "load_dotenv()\n\n"
        "from huggingface_hub import HfApi\n"
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


# --------------------------------------------------------------------------
# (e) IMPORT-ORDER arm (#745 round 2): a module-top huggingface_hub import that
# PRECEDES the dotenv/env setup FAILS even when the project wrapper is used —
# huggingface_hub.constants freezes HF_HUB_ENABLE_HF_TRANSFER at import time, so
# an env set AFTER the import is too late and the accelerator is inert.
# --------------------------------------------------------------------------


def test_module_top_hf_before_wrapper_import_fails(tmp_path: Path) -> None:
    """The round-2 Minor's worked example: huggingface_hub imported at module
    top BEFORE the project wrapper import + load_dotenv() call. The wrapper is
    present, so the old ARM-1-only check passed — but the constants froze before
    the env was set. FAILS, anchored at the huggingface_hub import line."""
    p = _write(
        tmp_path,
        "out_of_order.py",
        "from huggingface_hub import HfApi\n"
        "from explore_persona_space.orchestrate.env import load_dotenv\n\n"
        "load_dotenv()\n"
        "api = HfApi()\n",
    )
    errors = check_dotenv_before_hf_import(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(p) in errors[0]
    assert ":1:" in errors[0]  # anchored at the huggingface_hub import (line 1)
    assert "import-order" in errors[0]
    assert "#745" in errors[0]


def test_module_top_hf_before_load_dotenv_call_fails(tmp_path: Path) -> None:
    """The wrapper need not be the trigger — a module-top huggingface_hub import
    preceding a module-top load_dotenv() call (wrapper imported but the call is
    what sets env) is the same freeze-before-set bug."""
    _write(
        tmp_path,
        "hf_then_call.py",
        "import huggingface_hub\n"
        "from explore_persona_space.orchestrate.env import load_dotenv\n"
        "x = 1\n"
        "load_dotenv()\n",
    )
    errors = check_dotenv_before_hf_import(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert "import-order" in errors[0]


def test_wrapper_import_before_hf_with_load_dotenv_call_after_fails(tmp_path: Path) -> None:
    """ARM 2 false-negative regression (round 3): the wrapper is imported FIRST,
    huggingface_hub SECOND, and load_dotenv() is CALLED LAST. The accelerator
    setdefaults live INSIDE the wrapper's load_dotenv() body (orchestrate/env.py),
    so the wrapper IMPORT sets no env — only the CALL does. The constants freeze
    at the hf import (line 2) BEFORE the call (line 3) runs, so this IS the
    freeze-before-set bug ARM 2 exists to catch. The pre-round-3 logic used
    min(wrapper_import=1, call=3)=1 as the env-setting site, so hf(2) < 1 was
    False and the offender passed; ARM 2 now compares against the CALL line only,
    so hf(2) < 3 FAILS correctly. Anchored at the huggingface_hub import line."""
    p = _write(
        tmp_path,
        "wrapper_then_hf_then_call.py",
        "from explore_persona_space.orchestrate.env import load_dotenv\n"
        "from huggingface_hub import HfApi\n"
        "load_dotenv()\n"
        "api = HfApi()\n",
    )
    errors = check_dotenv_before_hf_import(scripts_dir=tmp_path)
    assert len(errors) == 1, errors
    assert str(p) in errors[0]
    assert ":2:" in errors[0]  # anchored at the huggingface_hub import (line 2)
    assert "import-order" in errors[0]
    assert "#745" in errors[0]


def test_wrapper_then_hf_module_top_passes(tmp_path: Path) -> None:
    """The correct order — wrapper import + load_dotenv() ABOVE the
    huggingface_hub import — PASSES (this is the sanctioned shape; the env is
    set before the constants freeze)."""
    _write(
        tmp_path,
        "in_order.py",
        "from explore_persona_space.orchestrate.env import load_dotenv\n\n"
        "load_dotenv()\n\n"
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n",
    )
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


def test_in_function_hf_import_after_module_top_call_passes(tmp_path: Path) -> None:
    """A DEFERRED in-function huggingface_hub import runs at call time, AFTER
    the module-top load_dotenv() has set the env — so it is NOT out of order and
    must not be flagged by the order arm (module-top imports only)."""
    _write(
        tmp_path,
        "lazy_in_order.py",
        "from explore_persona_space.orchestrate.env import load_dotenv\n\n"
        "load_dotenv()\n\n"
        "def upload():\n"
        "    from huggingface_hub import HfApi\n"
        "    return HfApi()\n",
    )
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


def test_module_top_hf_with_no_env_setup_passes(tmp_path: Path) -> None:
    """A script that imports huggingface_hub at module top but NEVER sets env
    (relies purely on shell-level exports — the bootstrap/GCE/SLURM lanes) is
    OUT OF SCOPE for the order arm: there is no env-setting site to be late
    relative to."""
    _write(
        tmp_path,
        "shell_exports_only.py",
        "from huggingface_hub import HfApi\n\napi = HfApi()\nprint(api)\n",
    )
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


def test_order_arm_waiver_on_hf_import_line_suppresses(tmp_path: Path) -> None:
    """The order arm is waivable with # DOTENV_LINT_EXEMPT on the
    huggingface_hub import line (reason ≥ 10 chars)."""
    _write(
        tmp_path,
        "waived_order.py",
        "from huggingface_hub import HfApi  # DOTENV_LINT_EXEMPT: order is fine, no upload here\n"
        "from explore_persona_space.orchestrate.env import load_dotenv\n\n"
        "load_dotenv()\n"
        "api = HfApi()\n",
    )
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []


def test_bare_dotenv_waiver_suppresses_both_arms(tmp_path: Path) -> None:
    """A single bare-dotenv waiver expresses an explicit "#745 dotenv concern
    waived for this file" and suppresses BOTH arms — the order arm must not
    re-flag the same file the bare-dotenv waiver already covered."""
    _write(
        tmp_path,
        "waived_both.py",
        "from huggingface_hub import HfApi\n"
        "# DOTENV_LINT_EXEMPT: legacy download-only script, no accelerator needed\n"
        "from dotenv import load_dotenv\n\n"
        "load_dotenv()\n"
        "api = HfApi()\n",
    )
    assert check_dotenv_before_hf_import(scripts_dir=tmp_path) == []
