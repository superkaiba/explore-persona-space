"""Regression: ``__main__.main()`` must call ``load_dotenv()`` before parsing args.

Runtime forensics for task #365 (post-launch failure on pod-365): the
dispatcher subprocess fired Claude API calls for D=1 generation without
``ANTHROPIC_API_KEY`` in its environment, because no entry point in this
package was sourcing ``.env``. Every other entry point in the project
calls ``load_dotenv()`` from ``explore_persona_space.orchestrate.env`` at
the top of ``main()`` (see ``scripts/eval.py``, ``scripts/run_issue_*``);
the factor-screen entry point was missing the call.

These tests pin the call order:

  * ``main()`` calls ``load_dotenv()``.
  * It calls ``load_dotenv()`` BEFORE ``parse_args()`` (so an experimenter
    using ``--help`` still loads ``.env`` cleanly, and so any future API
    client init that argparse triggers indirectly already has the keys).
  * Crashes inside ``load_dotenv()`` are visible — no try/except is
    swallowing them.

They do NOT verify that ``.env`` actually exists on disk; that's the
deployment layer's concern. They DO verify the entry point owns the
``.env`` sourcing contract.
"""

from __future__ import annotations

from unittest import mock

import pytest

from explore_persona_space.experiments.factor_screen_365 import __main__ as fs_main


def test_main_calls_load_dotenv_before_parse_args() -> None:
    """The first observable thing ``main()`` does (after logging setup) is
    source the .env file. ``parse_args`` must come strictly later.

    We patch both functions on the module under test. The mocks capture
    call order; we assert ``load_dotenv`` is the first call and
    ``parse_args`` the second. ``parse_args`` is forced to raise
    ``SystemExit(0)`` so we don't actually execute any dispatch / cell
    / aggregate logic.
    """
    call_order: list[str] = []

    def _record_load_dotenv(*args, **kwargs):
        call_order.append("load_dotenv")

    def _record_parse_args(*args, **kwargs):
        call_order.append("parse_args")
        # Stop main() before it tries to dispatch / cell / aggregate.
        raise SystemExit(0)

    with (
        mock.patch.object(fs_main, "load_dotenv", side_effect=_record_load_dotenv),
        mock.patch.object(fs_main, "parse_args", side_effect=_record_parse_args),
        pytest.raises(SystemExit),
    ):
        fs_main.main([])

    assert call_order == ["load_dotenv", "parse_args"], (
        f"Expected load_dotenv -> parse_args; got {call_order}. "
        "The .env file must be sourced before any argparse-triggered work "
        "(e.g. lazy API client init) — otherwise ANTHROPIC_API_KEY / "
        "HF_TOKEN may be missing for the dispatcher subprocess."
    )


def test_main_imports_load_dotenv_from_orchestrate_env() -> None:
    """The imported ``load_dotenv`` must come from
    ``explore_persona_space.orchestrate.env`` — not a bare ``dotenv``
    import or a project-local wrapper.

    This is the canonical pattern used by every other entry point in the
    project (``scripts/eval.py``, ``scripts/run_issue_*.py``). Sourcing
    it consistently means the unified HF_HOME default and dotenv
    override-policy stay aligned across entry points.
    """
    from explore_persona_space.orchestrate import env as orchestrate_env

    assert fs_main.load_dotenv is orchestrate_env.load_dotenv, (
        "Expected fs_main.load_dotenv to be the orchestrate.env version "
        "(canonical project pattern). Got a different binding — likely a "
        "bare `from dotenv import load_dotenv` or a wrapper; switch to "
        "`from explore_persona_space.orchestrate.env import load_dotenv` "
        "to inherit HF_HOME defaults and the project-wide override policy."
    )


def test_load_dotenv_errors_propagate() -> None:
    """If ``load_dotenv`` raises, ``main()`` does NOT swallow the error.

    Hiding a missing-.env error would put us back in the failure mode
    we just fixed (silent auth failure deep inside the Claude call).
    Crash early, crash loud.
    """

    sentinel = RuntimeError("synthetic .env load failure")

    with (
        mock.patch.object(fs_main, "load_dotenv", side_effect=sentinel),
        pytest.raises(RuntimeError, match=r"synthetic \.env load failure"),
    ):
        fs_main.main([])
