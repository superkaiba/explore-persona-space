"""Smoke tests for the relocated task #365 factor-screen package.

These tests are deliberately minimal: they catch the failure mode observed
in the original Sagan dispatch, where ``ModuleNotFoundError: No module named
'eps'`` killed every pod at module-load time. After relocation under
``src/explore_persona_space/experiments/factor_screen_365/`` the package
imports cleanly and ``--help`` returns 0.

They do NOT exercise the heavy ML dependencies (transformers / peft / vllm)
— those are imported lazily inside ``_run_cell_mode`` and require a GPU.
"""

from __future__ import annotations

import subprocess
import sys


def test_package_imports() -> None:
    """The package and every submodule must import without side effects."""
    import explore_persona_space.experiments.factor_screen_365 as pkg
    from explore_persona_space.experiments.factor_screen_365 import (
        aggregator,
        bootstrap,
        cells,
        data_prep,
        eval_panel,
        onpolicy,
        persona_panel,
        progress,
        prompts,
        training,
    )

    assert pkg.__name__ == "explore_persona_space.experiments.factor_screen_365"
    # Touch the modules to make sure flake doesn't strip them out.
    assert aggregator is not None
    assert bootstrap is not None
    assert cells is not None
    assert data_prep is not None
    assert eval_panel is not None
    assert onpolicy is not None
    assert persona_panel is not None
    assert progress is not None
    assert prompts is not None
    assert training is not None


def test_cli_help_exits_zero() -> None:
    """``python -m explore_persona_space.experiments.factor_screen_365 --help`` must succeed.

    This is the exact failure mode reported in the planner brief: the prior
    Sagan dispatch died with ``ModuleNotFoundError: No module named 'eps'``
    when trying to invoke ``-m eps.experiments.marker_factor_screen``.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "explore_persona_space.experiments.factor_screen_365",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "factor screen" in result.stdout.lower()


def test_cli_help_cells_exits_zero() -> None:
    """``--mode help-cells`` prints the 32-cell roster without crashing."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "explore_persona_space.experiments.factor_screen_365",
            "--mode",
            "help-cells",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"help-cells exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # 32 cell lines + factor-encoding header.
    assert result.stdout.count("bits=") == 32


def test_cli_tolerates_empty_run_index() -> None:
    """Empty templated int flags ``--run-index ''`` must not crash argparse.

    Reproduces the second prior failure mode: ``argument --run-index:
    invalid int value: ''``. We strip empty int flags before argparse.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "explore_persona_space.experiments.factor_screen_365",
            "--mode",
            "help-cells",
            "--run-index",
            "",
            "--seed",
            "",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"empty-int-flag tolerance exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
