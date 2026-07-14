"""#1172 — effect-level checks for the launcher-level repo-root PYTHONPATH mask.

Two check families:

1. **Criterion-6 differential script-mode probe.** Script-mode python puts the
   SCRIPT's directory on ``sys.path[0]`` — not cwd, not the repo root — so a
   probe script living OUTSIDE the repo root can resolve ``import scripts``
   ONLY via PYTHONPATH; cwd cannot mask a dead export. (The v1 ``-c``-mode
   spot check ran from the repo root, where cwd resolved the import
   regardless of PYTHONPATH — it could not fail; plan v2 §15.) The positive
   leg proves the export takes effect through the interpreter launch (and,
   on the ``uv run`` leg, that ``uv run`` propagates PYTHONPATH into the
   child — the one otherwise-untested link in every lane's chain); the
   negative control proves the probe CAN fail, so the positive leg is
   non-vacuous.

2. **Criterion-4 idiom harness.** The rendered export idiom
   ``export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"`` must yield NO
   leading/trailing colon when PYTHONPATH is unset or empty (a stray colon
   silently adds cwd to sys.path — cpython #107353), prepend the repo root
   before a preset value, and survive ``set -u`` (``:+`` is nounset-exempt).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

PROBE_SOURCE = """\
import scripts
import explore_persona_space

print(scripts.__file__)
print(explore_persona_space.__file__)
"""


def _probe_cmd(probe_path: Path) -> list[str]:
    """Interpreter invocation for the probe subprocess.

    Preferred leg (plan §6): ``uv run --no-sync --project <repo-root> python``
    — this ALSO exercises uv-run env propagation into the child interpreter
    (``--no-sync`` keeps the test read-only w.r.t. the venv). Falls back to
    the CURRENT interpreter (``sys.executable``, the venv python running
    pytest) when the checkout has no ``.venv`` — e.g. a sparse issue
    worktree, where ``uv run --project`` would build a multi-GB venv
    mid-test (the plan §6 allowed deviation; the uv-propagation link is then
    covered whenever the suite runs on a checkout with a ``.venv``).
    """
    uv = shutil.which("uv")
    if uv is not None and (REPO_ROOT / ".venv" / "bin" / "python").exists():
        return [uv, "run", "--no-sync", "--project", str(REPO_ROOT), "python", str(probe_path)]
    return [sys.executable, str(probe_path)]


def _run_probe(tmp_path: Path, *, with_pythonpath: bool) -> subprocess.CompletedProcess[str]:
    """Run the probe SCRIPT-MODE with cwd=tmp_path (outside the repo root)."""
    probe = tmp_path / "probe.py"
    probe.write_text(PROBE_SOURCE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if with_pythonpath:
        env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        _probe_cmd(probe),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_pythonpath_makes_scripts_importable_in_script_mode(tmp_path: Path) -> None:
    """Positive leg: PYTHONPATH=<repo-root> resolves ``scripts`` from a probe
    outside the repo root, WITHOUT shadowing the src-layout package."""
    result = _run_probe(tmp_path, with_pythonpath=True)
    assert result.returncode == 0, result.stderr
    out_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    scripts_file, eps_file = out_lines[-2], out_lines[-1]
    # `scripts` resolves to the repo-root package the export points at.
    assert Path(scripts_file).resolve().parent == (REPO_ROOT / "scripts").resolve(), scripts_file
    # The prepend does NOT shadow `explore_persona_space` — src layout keeps
    # the package OUT of the repo root, so the venv (editable) install wins.
    assert "/src/explore_persona_space/" in Path(eps_file).resolve().as_posix(), eps_file


def test_missing_pythonpath_reproduces_modulenotfounderror(tmp_path: Path) -> None:
    """Negative control: without PYTHONPATH the identical probe dies on the
    #823/#853 signature — proving the positive leg is non-vacuous."""
    result = _run_probe(tmp_path, with_pythonpath=False)
    assert result.returncode != 0
    assert "ModuleNotFoundError: No module named 'scripts'" in result.stderr, result.stderr


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (None, "/repo"),  # unset: no leading/trailing colon
        ("", "/repo"),  # empty: :+ treats empty as unset — still no colon
        ("/x", "/repo:/x"),  # preset: repo root PREPENDS, inherited value kept
    ],
)
def test_export_idiom_no_stray_colon_and_prepend_order(preset: str | None, expected: str) -> None:
    """Criterion 4: the exact rendered idiom, executed under ``set -u``."""
    script = 'set -u; ROOT=/repo; export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"; printf %s "$PYTHONPATH"'  # noqa: E501
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if preset is not None:
        env["PYTHONPATH"] = preset
    proc = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == expected
