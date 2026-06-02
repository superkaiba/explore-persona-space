"""Tests for the EPM_LOCAL_R_CANON_DIR env-var override.

Round-5 addition (round-5 cascade #3): the `--gpu` smoke driver needs
to point phase 2-check / 4 / 4.5 / 23 at the LOCALLY-generated R_canon
in the temp dir, NOT at HF Hub. Phase 1's `--no-upload` keeps R_canon
local; downstream phases falling through to ``hf_hub_download`` hit a
404 (which IS the correct behavior — there's nothing to download
because the smoke is isolated).

Four scripts each gained an ``EPM_LOCAL_R_CANON_DIR`` env hook in
their `_load_R_canon` / `_load_R_canon_test` helpers. When set, treat
the value as the directory containing ``R_canon_{split}.json`` and
read directly. RAISE if env is set but file missing — never silently
fall through to HF (the override's whole purpose is `--no-upload`
isolation).

These tests load each script as a module and call the helper directly
with the env set + a fixture file in place.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# (script_filename, helper_function_name, helper_arg) tuples.
# phase23 takes a `split` arg; the others hardcode "test".
SCRIPT_HELPERS = [
    ("i464_phase4_eval.py", "_load_R_canon_test", None),
    ("i464_phase45_onpolicy_validation.py", "_load_R_canon_test", None),
    ("i464_phase2_smoke_check.py", "_load_R_canon_test", None),
    ("i464_phase23_train.py", "_load_R_canon", "test"),
]


def _make_fake_r_canon(root: Path, split: str = "test"):
    """Create a fake R_canon JSON matching the i464_v2_matched_R schema."""
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "i464_v2_matched_R",
        "split": split,
        "completions": {
            "pirate": {"smoke q?": {"response_text": "arrr"}},
            "villain": {"smoke q?": {"response_text": "muahaha"}},
        },
    }
    (root / f"R_canon_{split}.json").write_text(json.dumps(payload))


@pytest.fixture(scope="module", params=SCRIPT_HELPERS, ids=[s[0] for s in SCRIPT_HELPERS])
def script_and_helper(request):
    """Load each of the 4 scripts and return (module, helper_fn, helper_arg)."""
    script_name, helper_name, helper_arg = request.param
    spec = importlib.util.spec_from_file_location(
        script_name.removesuffix(".py"),
        REPO_ROOT / "scripts" / script_name,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, getattr(mod, helper_name), helper_arg


def _call(helper, helper_arg):
    """Invoke the helper with the right arity (phase23 takes split, others don't)."""
    if helper_arg is None:
        return helper()
    return helper(helper_arg)


def test_override_returns_local_completions_when_file_present(
    script_and_helper, monkeypatch, tmp_path
):
    """When EPM_LOCAL_R_CANON_DIR is set + R_canon_<split>.json exists, return its completions.

    The helper must NOT attempt any hf_hub_download (we observe this
    indirectly: the local file is present so the override branch fires
    before the HF branch).
    """
    _mod, helper, helper_arg = script_and_helper
    split = helper_arg if helper_arg is not None else "test"
    _make_fake_r_canon(tmp_path, split=split)
    monkeypatch.setenv("EPM_LOCAL_R_CANON_DIR", str(tmp_path))
    completions = _call(helper, helper_arg)
    assert "pirate" in completions
    assert "villain" in completions
    assert completions["pirate"]["smoke q?"]["response_text"] == "arrr"


def test_override_raises_when_file_missing(script_and_helper, monkeypatch, tmp_path):
    """Env set BUT R_canon_<split>.json missing under override -> RAISE (fail-loud).

    The contract is explicit: setting the override asserts the file
    exists locally; a silent fall-through to HF download would defeat
    the `--no-upload` isolation guarantee the GPU smoke relies on.
    """
    _mod, helper, helper_arg = script_and_helper
    monkeypatch.setenv("EPM_LOCAL_R_CANON_DIR", str(tmp_path))
    # No R_canon file created — env set, file missing.
    with pytest.raises(RuntimeError, match="EPM_LOCAL_R_CANON_DIR"):
        _call(helper, helper_arg)


def test_override_missing_includes_path_in_error(script_and_helper, monkeypatch, tmp_path):
    """Error message must surface the missing path for fast debugging."""
    _mod, helper, helper_arg = script_and_helper
    split = helper_arg if helper_arg is not None else "test"
    monkeypatch.setenv("EPM_LOCAL_R_CANON_DIR", str(tmp_path))
    expected_path = tmp_path / f"R_canon_{split}.json"
    with pytest.raises(RuntimeError) as excinfo:
        _call(helper, helper_arg)
    msg = str(excinfo.value)
    assert str(expected_path) in msg, f"error did not include {expected_path}: {msg}"
