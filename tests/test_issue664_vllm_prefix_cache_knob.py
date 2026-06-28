"""Issue #664 r12 invariant pin: the vLLM deadlock-escape knob reaches EVERY
production ``LLM(...)`` constructor (concern p2-llm-constructors-prefix-cache).

Background: vLLM v0.11.0's V1 EngineCore deadlocks (futex, 0% GPU) at the first
``generate()`` of a large batch sharing one long system-prompt prefix. r11 added
the ``EPM_VLLM_PREFIX_CACHING`` knob to the dispatcher's ``_vllm_engine`` only;
the reconciler-binding FAIL was that the dispatcher SHELLS OUT to
``issue664_eval`` (p2 eval-gen on the ~200-prompt AdvBench battery) and
``issue664_extract_store`` (p2 extract), each with its OWN ``LLM(...)`` that
DEFAULTED ``enable_prefix_caching`` back to True -> the deadlock recurs at p2
eval-gen AFTER the train+extract GPU spend. r12 centralizes the knob reads into
``issue664_common.vllm_env_kwargs()`` and routes all three sites through it.

These pins are CPU-only -- they exercise the pure-Python env-parsing helper and
AST-assert the wiring, so a future refactor cannot silently re-introduce a
per-site default that floors the knob (the un-CI-pinned-assertion class).
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue664_common as C  # noqa: E402

_KNOB_ENVS = ("EPM_VLLM_ENFORCE_EAGER", "EPM_VLLM_PREFIX_CACHING")


@pytest.fixture(autouse=True)
def _clean_knob_env(monkeypatch):
    """Each test starts with the knob env vars unset (defaults apply)."""
    for k in _KNOB_ENVS:
        monkeypatch.delenv(k, raising=False)
    yield


def test_default_caching_on_eager_off():
    """Default (no env set) preserves vLLM behavior: prefix-caching ON, eager OFF."""
    assert C.vllm_env_kwargs() == {"enforce_eager": False, "enable_prefix_caching": True}


def test_production_override_disables_caching(monkeypatch):
    """The deadlock-escape production setting: caching OFF, eager ON."""
    monkeypatch.setenv("EPM_VLLM_PREFIX_CACHING", "0")
    monkeypatch.setenv("EPM_VLLM_ENFORCE_EAGER", "1")
    assert C.vllm_env_kwargs() == {"enforce_eager": True, "enable_prefix_caching": False}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", False),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("no", False),
        ("No", False),
        ("off", False),
        ("OFF", False),
        ("1", True),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("YES", True),
        ("on", True),
        ("ON", True),
        (" 0 ", False),
        (" TRUE ", True),  # surrounding whitespace tolerated
    ],
)
def test_case_insensitive_bool_forms(monkeypatch, raw, expected):
    monkeypatch.setenv("EPM_VLLM_PREFIX_CACHING", raw)
    assert C.vllm_env_kwargs()["enable_prefix_caching"] is expected


def test_invalid_value_fails_fast(monkeypatch):
    """A typo (e.g. 'ture') raises ValueError instead of silently re-enabling
    caching -- the whole point of the knob is to AVOID the deadlock, so a silent
    default on a typo would re-introduce exactly the failure it guards."""
    monkeypatch.setenv("EPM_VLLM_PREFIX_CACHING", "ture")
    with pytest.raises(ValueError, match="EPM_VLLM_PREFIX_CACHING"):
        C.vllm_env_kwargs()


def test_invalid_eager_value_fails_fast(monkeypatch):
    monkeypatch.setenv("EPM_VLLM_ENFORCE_EAGER", "maybe")
    with pytest.raises(ValueError, match="EPM_VLLM_ENFORCE_EAGER"):
        C.vllm_env_kwargs()


# ── AST wiring pins: every production LLM(...) routes through the shared helper ──
# The decisive r12 invariant. A per-site `enable_prefix_caching=True` default (or
# an omitted kwarg, which vLLM resolves to True) is the exact regression the
# reconciler FAILed on; these assert the helper splat is present and no LLM(...)
# in the three production scripts hardcodes the knob.
_PRODUCTION_SCRIPTS = (
    "issue664_dispatch.py",
    "issue664_eval.py",
    "issue664_extract_store.py",
)


def _llm_calls(tree: ast.AST) -> list[ast.Call]:
    """All ``LLM(...)`` call nodes in a parsed module."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LLM"
    ]


def _splats_vllm_env_kwargs(call: ast.Call) -> bool:
    """True if the call **-splats C.vllm_env_kwargs() (or vllm_env_kwargs())."""
    for kw in call.keywords:
        if kw.arg is not None:  # **-splat keywords have arg=None
            continue
        val = kw.value
        if (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Attribute)
            and val.func.attr == "vllm_env_kwargs"
        ):
            return True
        if (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Name)
            and val.func.id == "vllm_env_kwargs"
        ):
            return True
    return False


@pytest.mark.parametrize("script", _PRODUCTION_SCRIPTS)
def test_every_production_llm_routes_through_helper(script):
    src = (SCRIPTS / script).read_text()
    tree = ast.parse(src)
    calls = _llm_calls(tree)
    assert calls, f"no LLM(...) constructor found in {script} -- test stale?"
    for call in calls:
        # no production LLM(...) may hardcode enable_prefix_caching (the per-site
        # default that re-introduces the deadlock).
        kw_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "enable_prefix_caching" not in kw_names, (
            f"{script}: an LLM(...) hardcodes enable_prefix_caching -- route it "
            "through C.vllm_env_kwargs() so the knob applies"
        )
        assert _splats_vllm_env_kwargs(call), (
            f"{script}: an LLM(...) does not **-splat vllm_env_kwargs() -- the "
            "EPM_VLLM_PREFIX_CACHING knob will not reach it (concern "
            "p2-llm-constructors-prefix-cache)"
        )


def test_helper_uses_only_known_env_vars():
    """Sanity: the helper reads exactly the two documented knobs (guards against a
    silent rename that would orphan the launcher's env injection)."""
    src = (SCRIPTS / "issue664_common.py").read_text()
    assert "EPM_VLLM_PREFIX_CACHING" in src
    assert "EPM_VLLM_ENFORCE_EAGER" in src


def test_module_import_clean():
    """The helper module imports without GPU/network (CPU-only invariant)."""
    assert callable(C.vllm_env_kwargs)
    assert callable(C._parse_env_bool)
    # restore a clean environment (defensive; fixture also handles it)
    for k in _KNOB_ENVS:
        os.environ.pop(k, None)
