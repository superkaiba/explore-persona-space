"""Interpreter-selection + model-venv ensure pins for issue #2378 (r5 crash-fix).

The P1 crash (`epm:failure v1`, assert_tag: transformers-lacks-qwen3_5): every
model-loading pod step was composed via `_py()` (repo venv — vLLM 0.11.0 /
transformers 4.57.6, no `qwen3_5` model type) and the `env_smoke` guard was
never wired into the launch chain. These tests pin the fix WITHOUT any GPU or
network access:

- `_model_python()` resolution ($EPM_I2378_MODEL_PY > /root/eps-model-venv);
- `_model_py()` argv composition (model interpreter in script mode, never uv);
- the exact plan-Repro-card pins (vllm/transformers/torch + python-dotenv);
- source-scan routing: every MODEL gen/capture phase composes via `_model_py`,
  every non-model step stays on `_py` (fails pre-fix, passes post-fix);
- every MODEL phase (p1/p2/p4_topup/p4) calls `ensure_model_venv` at entry;
- real-body `ensure_model_venv` legs (happy path via an executable boundary
  fake interpreter; pin-mismatch RuntimeError; override-refusal RuntimeError;
  dry-run) — no seams stubbed, fakes only at the interpreter boundary.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2378_common as cm  # noqa: E402
import issue2378_dispatch as d  # noqa: E402

DISPATCH_SRC = (SCRIPTS / "issue2378_dispatch.py").read_text(encoding="utf-8")

# gen.py phases that load Qwen3.6-27B (tokenizer and/or vLLM engine) and
# capture.py phases that load Qwen3_5ForConditionalGeneration — the model set.
MODEL_PHASES = {
    ("issue2378_gen.py", "sega"),
    ("issue2378_gen.py", "chat_plain"),
    ("issue2378_gen.py", "user_sim"),
    ("issue2378_gen.py", "user_fresh"),
    ("issue2378_gen.py", "segb"),
    ("issue2378_gen.py", "fresh_draws"),
    ("issue2378_gen.py", "user_real_render"),
    ("issue2378_capture.py", "pilot"),
    ("issue2378_capture.py", "capture"),
    ("issue2378_capture.py", "capture_fresh"),
}
# Non-model steps that MUST stay on the repo venv (`_py`).
REPO_PHASES = {
    ("issue2378_gen.py", "build_banks"),
    ("issue2378_gen.py", "build_pools"),
    ("issue2378_gen.py", "upload_stage"),
    ("issue2378_gen.py", "capture_ready"),
}


# ---------------------------------------------------------------------------
# Interpreter resolution + argv composition
# ---------------------------------------------------------------------------


def test_model_python_default(monkeypatch):
    monkeypatch.delenv(cm.MODEL_PY_ENV, raising=False)
    assert d._model_python() == str(Path(cm.MODEL_VENV_DEFAULT) / "bin" / "python")


def test_model_python_env_override(monkeypatch):
    monkeypatch.setenv(cm.MODEL_PY_ENV, "/opt/other-venv/bin/python")
    assert d._model_python() == "/opt/other-venv/bin/python"


def test_model_py_composes_model_interpreter_script_mode(monkeypatch):
    monkeypatch.setenv(cm.MODEL_PY_ENV, "/opt/other-venv/bin/python")
    argv = d._model_py("issue2378_gen.py", "--phase", "sega")
    assert argv[0] == "/opt/other-venv/bin/python"
    assert argv[1] == str(cm.REPO_ROOT / "scripts" / "issue2378_gen.py")
    assert argv[2:] == ["--phase", "sega"]
    assert "uv" not in argv  # never the repo-venv `uv run python` shape


def test_repo_py_still_uses_uv():
    argv = d._py("issue2378_gen.py", "--phase", "build_banks")
    assert argv[:3] == ["uv", "run", "python"]


# ---------------------------------------------------------------------------
# Pins (plan Repro card "exact pin at P1")
# ---------------------------------------------------------------------------


def test_model_venv_pins_exact():
    assert cm.MODEL_VENV_PINS == {
        "vllm": "0.27.1",
        "transformers": "5.15.1",
        "torch": "2.13.0",
    }
    assert "python-dotenv==1.2.2" in cm.MODEL_VENV_EXTRA_PINS
    assert cm.MODEL_VENV_DEFAULT == "/root/eps-model-venv"
    assert cm.MODEL_PY_ENV == "EPM_I2378_MODEL_PY"


def test_probe_src_checks_qwen35_vllm_dotenv():
    assert "transformers.models.qwen3_5" in d._MODEL_PROBE_SRC
    assert '"vllm"' in d._MODEL_PROBE_SRC
    assert '"dotenv"' in d._MODEL_PROBE_SRC  # orchestrate/env.py module-top dep


# ---------------------------------------------------------------------------
# Routing pin: model phases -> _model_py, non-model -> _py (source scan)
# ---------------------------------------------------------------------------

_CALL_RE = re.compile(
    r'(_model_py|_py)\(\s*"(issue2378_(?:gen|capture)\.py)",\s*"--phase",\s*"(\w+)"', re.S
)


def test_phase_call_sites_route_to_correct_interpreter():
    helpers: dict[tuple[str, str], set[str]] = defaultdict(set)
    n_matches = 0
    for helper, script, phase in _CALL_RE.findall(DISPATCH_SRC):
        helpers[(script, phase)].add(helper)
        n_matches += 1
    # Non-vacuity floor: the dispatcher composes >= 14 distinct (script, phase)
    # gen/capture invocations today; a regex drift that matches nothing must
    # fail here, not silently pass the per-set asserts below.
    assert n_matches >= 14, f"call-site scan matched only {n_matches} sites"
    for key in MODEL_PHASES:
        assert helpers.get(key) == {"_model_py"}, (
            f"{key} must be composed via _model_py (model venv), got {helpers.get(key)}"
        )
    for key in REPO_PHASES:
        assert helpers.get(key) == {"_py"}, (
            f"{key} must stay on _py (repo venv), got {helpers.get(key)}"
        )


def test_model_phases_call_ensure_model_venv_at_entry():
    import inspect

    for fn in (d.phase_p1, d.phase_p2, d.phase_p4_topup, d.phase_p4):
        src = inspect.getsource(fn)
        assert "ensure_model_venv(args, runner)" in src, fn.__name__


def test_model_venv_phase_registered():
    assert d.PHASES["model_venv"] is d.phase_model_venv


# ---------------------------------------------------------------------------
# ensure_model_venv real-body legs (boundary fake = an executable interpreter)
# ---------------------------------------------------------------------------


def _fake_interpreter(tmp_path: Path, payload: dict) -> str:
    """Executable that answers ANY invocation with one JSON line (the probe's
    contract) and rc=0 — the external interpreter boundary, faked by shape."""
    py = tmp_path / "fakepython"
    py.write_text("#!/bin/sh\necho '" + json.dumps(payload) + "'\n", encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IXUSR)
    return str(py)


def _good_payload() -> dict:
    return {"python": "3.11.13", **cm.MODEL_VENV_PINS}


def test_ensure_happy_path_records_pins_and_runs_env_smoke(monkeypatch, tmp_path):
    fake = _fake_interpreter(tmp_path, _good_payload())
    monkeypatch.setenv(cm.MODEL_PY_ENV, fake)
    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    (tmp_path / "ledger").mkdir()
    d.ensure_model_venv(args, runner)
    rec = json.loads((tmp_path / "ledger" / "model_venv_pins.json").read_text(encoding="utf-8"))
    assert rec["interpreter"] == fake
    assert rec["pinned"] == cm.MODEL_VENV_PINS
    for k, want in cm.MODEL_VENV_PINS.items():
        assert rec["realized"][k] == want
    # env_smoke ran UNDER the model interpreter (the fake exits 0 on any argv)
    assert (tmp_path / "logs" / "model_env_smoke.log").exists()


def test_ensure_pin_mismatch_raises(monkeypatch, tmp_path):
    payload = _good_payload()
    payload["vllm"] = "0.11.0"  # the crashing repo-venv version
    fake = _fake_interpreter(tmp_path, payload)
    monkeypatch.setenv(cm.MODEL_PY_ENV, fake)
    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    with pytest.raises(RuntimeError, match="pin mismatch"):
        d.ensure_model_venv(args, runner)


def test_ensure_refuses_to_build_over_explicit_override(monkeypatch, tmp_path):
    missing = str(tmp_path / "no-such-venv" / "bin" / "python")
    monkeypatch.setenv(cm.MODEL_PY_ENV, missing)
    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    with pytest.raises(RuntimeError, match="refusing to build over an explicit override"):
        d.ensure_model_venv(args, runner)
    # and it never fell back to the repo venv or built anything
    assert not (tmp_path / "no-such-venv").exists()


def test_ensure_dry_run_composes_without_touching_disk(monkeypatch, tmp_path):
    monkeypatch.setenv(cm.MODEL_PY_ENV, str(tmp_path / "absent" / "python"))
    runner = d.Runner(tmp_path / "logs", resume=False, dry=True)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    d.ensure_model_venv(args, runner)  # no probe, no build, no raise
    assert not (tmp_path / "ledger").exists()


def test_model_probe_missing_interpreter_returns_none(tmp_path):
    assert d._model_probe(str(tmp_path / "nope" / "python")) is None


def test_model_probe_parses_last_json_line(tmp_path):
    fake = _fake_interpreter(tmp_path, _good_payload())
    got = d._model_probe(fake)
    assert got is not None and got["vllm"] == cm.MODEL_VENV_PINS["vllm"]


def test_model_probe_env_passthrough_uses_os_environ():
    # subprocess env passthrough contract: explicit env={**os.environ}
    src = DISPATCH_SRC[DISPATCH_SRC.index("def _model_probe") :]
    src = src[: src.index("def _build_model_venv")]
    assert "env={**os.environ}" in src


def test_no_silent_repo_venv_fallback_in_model_python(monkeypatch):
    """_model_python never falls back to the repo venv: with no override it is
    the pinned model-venv path even when that interpreter does not exist."""
    import inspect

    src = inspect.getsource(d._model_python)
    assert "sys.executable" not in src
    monkeypatch.setenv(cm.MODEL_PY_ENV, "")  # empty env var -> default, not repo venv
    assert d._model_python() == str(Path(cm.MODEL_VENV_DEFAULT) / "bin" / "python")
    assert os.path.isabs(d._model_python())
