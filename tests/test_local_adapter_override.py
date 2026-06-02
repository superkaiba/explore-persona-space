"""Tests for the EPM_LOCAL_ADAPTER_OVERRIDE env-var override.

Round-5 addition (round-4 code-review reconciler hard condition): the
`--gpu` smoke driver needs to drive phase 2-check / 4 / 4.5 against a
LOCAL adapter dir (the one just-trained in the temp dir by phase 23),
NOT against HF Hub. Three scripts each gained an
``EPM_LOCAL_ADAPTER_OVERRIDE`` env hook that, when set, treats the
value as a directory root and returns ``<override>/adapters/i464_
<arm>_seed<seed>`` if the adapter exists locally — bypassing the
``hf_hub_download`` path entirely.

These tests verify the contract WITHOUT a GPU: load each script as a
module and call its `_resolve_adapter_path` / `_download_adapter`
helper directly with the env set + an adapter-fixture dir in place.

The override must:
  1. Return the override path when adapter exists.
  2. RAISE when env is set but adapter missing (fail loud — never
     silently fall through to the HF download).
  3. (Unset env) Fall back to HF download path — we don't test this
     directly here because it requires network; the test is the
     absence of the override branch when env is unset.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# (script_filename, helper_function_name) tuples — three scripts, same contract.
SCRIPT_HELPERS = [
    ("i464_phase2_smoke_check.py", "_resolve_adapter_path"),
    ("i464_phase4_eval.py", "_download_adapter"),
    ("i464_phase45_onpolicy_validation.py", "_download_adapter"),
]


@pytest.fixture(scope="module", params=SCRIPT_HELPERS, ids=[s[0] for s in SCRIPT_HELPERS])
def script_and_helper(request):
    """Load each of the 3 scripts and return (module, helper_fn)."""
    script_name, helper_name = request.param
    spec = importlib.util.spec_from_file_location(
        script_name.removesuffix(".py"),
        REPO_ROOT / "scripts" / script_name,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, getattr(mod, helper_name)


def _make_fake_adapter(root: Path, arm: str, seed: int) -> Path:
    """Create a fake adapter dir under <root>/adapters/i464_<arm>_seed<seed>."""
    p = root / "adapters" / f"i464_{arm}_seed{seed}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "adapter_model.safetensors").write_bytes(b"")  # zero-byte placeholder
    return p


def test_override_returns_local_path_when_adapter_present(script_and_helper, monkeypatch, tmp_path):
    """When EPM_LOCAL_ADAPTER_OVERRIDE is set + adapter exists, return the local path.

    The helper must NOT attempt any hf_hub_download call (we'd see that
    as a network attempt at test time; instead we observe the absence of
    a network call by the function returning before reaching it).
    """
    _mod, helper = script_and_helper
    _make_fake_adapter(tmp_path, "system_plain", 42)
    monkeypatch.setenv("EPM_LOCAL_ADAPTER_OVERRIDE", str(tmp_path))
    path = helper("system_plain", 42)
    expected = tmp_path / "adapters" / "i464_system_plain_seed42"
    assert Path(path) == expected, f"helper returned {path}, expected {expected}"


def test_override_raises_when_adapter_missing(script_and_helper, monkeypatch, tmp_path):
    """Env set BUT adapter missing under override -> must RAISE (fail-loud).

    The contract is explicit: when the operator sets the override they
    are asserting a local adapter exists; a silent fall-through to HF
    download would defeat the isolation guarantee the round-5 GPU smoke
    relies on.
    """
    _mod, helper = script_and_helper
    monkeypatch.setenv("EPM_LOCAL_ADAPTER_OVERRIDE", str(tmp_path))
    # No adapter created — env is set, but adapter_model.safetensors is missing.
    with pytest.raises(RuntimeError, match="EPM_LOCAL_ADAPTER_OVERRIDE"):
        helper("system_plain", 42)


def test_override_missing_adapter_includes_path_in_error(script_and_helper, monkeypatch, tmp_path):
    """The RuntimeError message must surface the missing path so the operator
    can fix the override quickly."""
    _mod, helper = script_and_helper
    monkeypatch.setenv("EPM_LOCAL_ADAPTER_OVERRIDE", str(tmp_path))
    expected_path = tmp_path / "adapters" / "i464_role_seed1337"
    with pytest.raises(RuntimeError) as excinfo:
        helper("role", 1337)
    msg = str(excinfo.value)
    assert "adapter_model.safetensors" in msg
    assert str(expected_path) in msg
