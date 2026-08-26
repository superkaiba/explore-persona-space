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

r6 additions (code-review round 6):
- CONTENT-STABLE pins record (r5 reconciler BLOCKER
  model-venv-pins-rewrite-breaks-p4-harvest): re-ensure over a COMMITTED
  record leaves `git status --porcelain` empty (the P4-harvest acceptance),
  volatile provenance lives in an untracked sidecar, legacy metadata-bearing
  records normalize once;
- host-driver compat probe legs (CONCERN model-venv-driver-compat-unguarded;
  fake `nvidia-smi` at the executable boundary);
- real `_build_model_venv` body (NIT model-venv-build-branch-untested; fake
  `uv` at the executable boundary) + rebuild invalidates the stale
  `model_env_smoke` ok-flag.

r7 additions (crash-fix, epm:failure v3 assert_tag
flashinfer-py311-array-subscript — vllm 0.27.1 hard-pins the
py3.11-incompatible flashinfer-python 0.6.16.post3; the TypeError from its
runtime-evaluated `array.array[int]` annotation escapes vLLM's
ImportError-only compile-backend guard and kills EngineCore):
- cm.MODEL_VENV_BANNED_DISTS exact pin + probe-src `banned_present` reporting;
- ensure REPAIRS an EXISTING venv carrying a banned dist IN PLACE (no create;
  uninstall AFTER install — the install re-adds vllm's pinned dep; stale
  env_smoke ok-flag cleared so the extended smoke re-runs);
- repair-failure + override-refusal fail-loud legs;
- env_smoke banned-dist gate (real-body execution both branches) + the
  UNGUARDED `vllm.compilation.backends` compile-backend import probe
  (source-scan ordering pin; the import itself executes pod-side).
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
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


@pytest.fixture(autouse=True)
def _one_visible_gpu(monkeypatch):
    """r8: ensure_model_venv now ends with the 1-GPU engine smoke, whose env
    composition resolves visible_gpus() — pin ONE fake GPU on the CPU test
    host (the zero-GPU fail-loud branch is exercised explicitly by
    test_ensure_engine_smoke_fails_loud_on_zero_gpu_host, which deletes it)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


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
    # r7: exact dict equality (dist name -> import name), never substring
    assert cm.MODEL_VENV_BANNED_DISTS == {"flashinfer-python": "flashinfer"}
    assert cm.MODEL_VENV_DEFAULT == "/root/eps-model-venv"
    assert cm.MODEL_PY_ENV == "EPM_I2378_MODEL_PY"


def test_probe_src_checks_qwen35_vllm_dotenv():
    assert "transformers.models.qwen3_5" in d._MODEL_PROBE_SRC
    assert '"vllm"' in d._MODEL_PROBE_SRC
    assert '"dotenv"' in d._MODEL_PROBE_SRC  # orchestrate/env.py module-top dep


def test_probe_src_reports_banned_present():
    """r7: the probe must REPORT banned accel imports so the ensure gate can
    repair the EXISTING pod venv (whose pins otherwise read healthy)."""
    assert '"banned_present"' in d._MODEL_PROBE_SRC
    assert "'flashinfer'" in d._MODEL_PROBE_SRC  # repr of the composed tuple


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
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")  # hermetic vs host GPUs
    fake = _fake_interpreter(tmp_path, _good_payload())
    monkeypatch.setenv(cm.MODEL_PY_ENV, fake)
    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    (tmp_path / "ledger").mkdir()
    d.ensure_model_venv(args, runner)
    rec = json.loads((tmp_path / "ledger" / "model_venv_pins.json").read_text(encoding="utf-8"))
    assert rec["interpreter"] == fake
    assert rec["pinned"] == cm.MODEL_VENV_PINS
    assert rec["banned"] == ["flashinfer-python"]  # r7: ban is part of the Repro card
    for k, want in cm.MODEL_VENV_PINS.items():
        assert rec["realized"][k] == want
    # env_smoke ran UNDER the model interpreter (the fake exits 0 on any argv)
    assert (tmp_path / "logs" / "model_env_smoke.log").exists()


def test_ensure_pin_mismatch_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")  # hermetic vs host GPUs
    payload = _good_payload()
    payload["vllm"] = "0.11.0"  # the crashing repo-venv version
    fake = _fake_interpreter(tmp_path, payload)
    monkeypatch.setenv(cm.MODEL_PY_ENV, fake)
    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    with pytest.raises(RuntimeError, match="pin mismatch"):
        d.ensure_model_venv(args, runner)


def test_ensure_refuses_to_build_over_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")  # hermetic vs host GPUs
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


# ---------------------------------------------------------------------------
# r6: content-stable pins record (BLOCKER model-venv-pins-rewrite-breaks-p4-harvest)
# ---------------------------------------------------------------------------


def _git(repo: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *argv], capture_output=True, text=True, check=True
    )


def test_ensure_pins_record_content_stable_p4_harvest_path(monkeypatch, tmp_path):
    """The P4 path (r5 reconciler BLOCKER): pod B materializes P1's COMMITTED
    pins record, then re-ensures at P4 entry — the tracked file must stay
    byte-identical + git-clean so the scoped git_harvest rebase cannot refuse
    `cannot rebase: You have unstaged changes`. Fails pre-fix: the r5 record
    embedded run_metadata (timestamp/argv), so every re-ensure dirtied it."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    fake = _fake_interpreter(tmp_path, _good_payload())
    monkeypatch.setenv(cm.MODEL_PY_ENV, fake)
    repo = tmp_path / "repo"
    ledger = repo / "eval_results" / "issue_2378"
    ledger.mkdir(parents=True)
    _git(tmp_path, "init", "-q", str(repo))
    args = SimpleNamespace(ledger_root=str(ledger))
    d.ensure_model_venv(args, d.Runner(tmp_path / "logs1", resume=False, dry=False))
    rec_path = ledger / "model_venv_pins.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    # ONLY stable pin content in the tracked record — no volatile fields
    # (r7 adds the constant `banned` key; still content-stable)
    assert set(rec) == {"interpreter", "realized", "pinned", "extra_pins", "banned"}
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "P1 harvest")
    before = rec_path.read_bytes()
    mtime = rec_path.stat().st_mtime_ns
    # second pod / second phase: fresh Runner (different logs dir), same content
    d.ensure_model_venv(args, d.Runner(tmp_path / "logs2", resume=False, dry=False))
    assert rec_path.read_bytes() == before
    assert rec_path.stat().st_mtime_ns == mtime  # skip branch: no rewrite at all
    porcelain = _git(repo, "status", "--porcelain").stdout.strip()
    assert porcelain == "", f"re-ensure dirtied the tracked tree:\n{porcelain}"


def test_ensure_normalizes_legacy_metadata_record_once(monkeypatch, tmp_path):
    """A prior r5-format record (volatile `metadata` embedded) is rewritten
    ONCE to the clean 4-key form; a further re-ensure is then byte-stable."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    fake = _fake_interpreter(tmp_path, _good_payload())
    monkeypatch.setenv(cm.MODEL_PY_ENV, fake)
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    legacy = {
        "interpreter": fake,
        "realized": _good_payload(),
        "pinned": dict(cm.MODEL_VENV_PINS),
        "extra_pins": list(cm.MODEL_VENV_EXTRA_PINS),
        "metadata": {"timestamp": "2026-08-19T00:00:00+0000", "argv": ["old"]},
    }
    (ledger / "model_venv_pins.json").write_text(json.dumps(legacy), encoding="utf-8")
    args = SimpleNamespace(ledger_root=str(ledger))
    d.ensure_model_venv(args, d.Runner(tmp_path / "logs1", resume=False, dry=False))
    rec = json.loads((ledger / "model_venv_pins.json").read_text(encoding="utf-8"))
    assert "metadata" not in rec
    assert set(rec) == {"interpreter", "realized", "pinned", "extra_pins", "banned"}
    before = (ledger / "model_venv_pins.json").read_bytes()
    d.ensure_model_venv(args, d.Runner(tmp_path / "logs2", resume=False, dry=False))
    assert (ledger / "model_venv_pins.json").read_bytes() == before


def test_ensure_writes_volatile_provenance_to_untracked_sidecar(monkeypatch, tmp_path):
    """timestamp/argv/git provenance land in the logs-dir sidecar (gitignored
    dispatch-logs root in production), never in the tracked pins record."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    fake = _fake_interpreter(tmp_path, _good_payload())
    monkeypatch.setenv(cm.MODEL_PY_ENV, fake)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    d.ensure_model_venv(args, d.Runner(tmp_path / "logs", resume=False, dry=False))
    meta = json.loads(
        (tmp_path / "logs" / "model_venv_ensure_meta.json").read_text(encoding="utf-8")
    )
    assert meta["pins_record"].endswith("model_venv_pins.json")
    assert "timestamp" in meta["metadata"] and "argv" in meta["metadata"]
    rec = json.loads((tmp_path / "ledger" / "model_venv_pins.json").read_text(encoding="utf-8"))
    assert "metadata" not in rec


def test_git_rebase_sites_carry_autostash():
    """Defense-in-depth (reconciler optional hardening): both pod-side rebase
    sites pass --autostash so a residual unstaged tracked file cannot produce
    the unconditional `cannot rebase: You have unstaged changes` refusal."""
    assert DISPATCH_SRC.count('"rebase", "--autostash"') >= 2


# ---------------------------------------------------------------------------
# r6: host-driver compat probe (CONCERN model-venv-driver-compat-unguarded)
# ---------------------------------------------------------------------------


def _fake_nvidia_smi(tmp_path: Path, version_line: str) -> Path:
    bindir = tmp_path / "smibin"
    bindir.mkdir(exist_ok=True)
    smi = bindir / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\necho '{version_line}'\n", encoding="utf-8")
    smi.chmod(smi.stat().st_mode | stat.S_IXUSR)
    return bindir


def test_driver_compat_ok_at_floor(monkeypatch, tmp_path):
    monkeypatch.delenv(cm.SKIP_DRIVER_PROBE_ENV, raising=False)
    monkeypatch.setenv("PATH", str(_fake_nvidia_smi(tmp_path, "580.159.04")))
    d._assert_driver_compat(compat_dir=str(tmp_path / "compat"))  # no raise


def test_driver_compat_pre580_raises_naming_2330_recipe(monkeypatch, tmp_path):
    monkeypatch.delenv(cm.SKIP_DRIVER_PROBE_ENV, raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("PATH", str(_fake_nvidia_smi(tmp_path, "570.195.03")))
    with pytest.raises(RuntimeError, match="cuda-compat-13-0"):
        d._assert_driver_compat(compat_dir=str(tmp_path / "compat"))


def test_driver_compat_pre580_with_active_compat_passes(monkeypatch, tmp_path):
    monkeypatch.delenv(cm.SKIP_DRIVER_PROBE_ENV, raising=False)
    compat = tmp_path / "compat"
    compat.mkdir()
    (compat / "libcuda.so.580.65.06").write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", str(_fake_nvidia_smi(tmp_path, "570.195.03")))
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{compat}:/usr/lib")
    d._assert_driver_compat(compat_dir=str(compat))  # no raise (#2330 recipe active)


def test_driver_compat_compat_lib_present_but_not_on_ld_path_raises(monkeypatch, tmp_path):
    """Presence alone does not reach the loader — LD_LIBRARY_PATH must carry it."""
    monkeypatch.delenv(cm.SKIP_DRIVER_PROBE_ENV, raising=False)
    compat = tmp_path / "compat"
    compat.mkdir()
    (compat / "libcuda.so.580.65.06").write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", str(_fake_nvidia_smi(tmp_path, "570.195.03")))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    with pytest.raises(RuntimeError, match="on LD_LIBRARY_PATH: False"):
        d._assert_driver_compat(compat_dir=str(compat))


def test_driver_compat_skips_on_no_gpu_host(monkeypatch, tmp_path):
    monkeypatch.delenv(cm.SKIP_DRIVER_PROBE_ENV, raising=False)
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))  # no nvidia-smi -> logged skip, no raise
    d._assert_driver_compat(compat_dir=str(tmp_path / "compat"))


def test_driver_compat_env_waiver_skips(monkeypatch, tmp_path):
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    monkeypatch.setenv("PATH", str(_fake_nvidia_smi(tmp_path, "570.195.03")))
    d._assert_driver_compat(compat_dir=str(tmp_path / "compat"))  # waived, no raise


def test_ensure_runs_driver_probe_before_build():
    """Gate ordering: the driver probe fails fast BEFORE any ~4-min venv build."""
    import inspect

    src = inspect.getsource(d.ensure_model_venv)
    assert src.index("_assert_driver_compat(") < src.index("_model_probe(")


# ---------------------------------------------------------------------------
# r6: real _build_model_venv body (NIT model-venv-build-branch-untested)
# ---------------------------------------------------------------------------


def _fake_uv(tmp_path: Path, template_interpreter: str) -> tuple[Path, Path]:
    """Executable `uv` boundary fake: `uv venv <dir> --python 3.11` creates
    <dir>/bin/python from the probe-answering template; `uv pip install ...`
    exits 0. Every invocation's argv is appended to uv_argv.log."""
    bindir = tmp_path / "uvbin"
    bindir.mkdir(exist_ok=True)
    uv_log = tmp_path / "uv_argv.log"
    uv = bindir / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{uv_log}"\n'
        'if [ "$1" = "venv" ]; then\n'
        f'  mkdir -p "$2/bin" && cp "{template_interpreter}" "$2/bin/python" '
        '&& chmod +x "$2/bin/python"\n'
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    return bindir, uv_log


def test_ensure_probe_miss_builds_then_reprobes(monkeypatch, tmp_path):
    """REAL `_build_model_venv` body: probe-miss -> `uv venv` create + `uv pip
    install` (both subprocess steps executed, argv logged, build log written)
    -> reprobe OK -> pins recorded. `uv` faked only at the executable boundary."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    monkeypatch.delenv(cm.MODEL_PY_ENV, raising=False)
    venv = tmp_path / "venv"
    monkeypatch.setattr(cm, "MODEL_VENV_DEFAULT", str(venv))
    template = _fake_interpreter(tmp_path, _good_payload())
    bindir, uv_log = _fake_uv(tmp_path, template)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    d.ensure_model_venv(args, runner)
    calls = [c for c in uv_log.read_text(encoding="utf-8").split("\n") if c]
    assert any(c.startswith("venv ") for c in calls), calls  # create step ran
    assert any(c.startswith("pip install ") for c in calls), calls  # install step ran
    specs = next(c for c in calls if c.startswith("pip install "))
    for k, v in cm.MODEL_VENV_PINS.items():
        assert f"{k}=={v}" in specs
    assert (venv / "bin" / "python").exists()
    assert (tmp_path / "logs" / "model_venv_build.log").exists()
    rec = json.loads((tmp_path / "ledger" / "model_venv_pins.json").read_text(encoding="utf-8"))
    assert rec["interpreter"] == str(venv / "bin" / "python")


def test_rebuild_clears_stale_env_smoke_ok_flag(monkeypatch, tmp_path):
    """Overlay-wipe residue (r5-disclosed): the env_smoke argv sha is UNCHANGED
    across a rebuild, so a stale ok-flag would silently skip the render asserts
    under the rebuilt interpreter — a build must force the re-run."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    monkeypatch.delenv(cm.MODEL_PY_ENV, raising=False)
    venv = tmp_path / "venv"
    monkeypatch.setattr(cm, "MODEL_VENV_DEFAULT", str(venv))
    template = _fake_interpreter(tmp_path, _good_payload())
    bindir, _ = _fake_uv(tmp_path, template)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    runner = d.Runner(tmp_path / "logs", resume=True, dry=False)
    smoke_argv = d._model_py("issue2378_dispatch.py", "--phase", "env_smoke")
    runner._ok_path("model_env_smoke").write_text(d._argv_sha(smoke_argv))  # stale flag
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    d.ensure_model_venv(args, runner)
    # the step RAN (fresh log) instead of resume-skipping on the stale flag
    assert (tmp_path / "logs" / "model_env_smoke.log").exists()


# ---------------------------------------------------------------------------
# r6: launch-chain sequencing pin (carry-forward of epm:failure v1)
# ---------------------------------------------------------------------------


def test_runner_failure_tail_drops_terminal_empty_line(tmp_path):
    """r5 NIT runner-log-tail-trailing-empty: a newline-terminated log must
    yield 25 REAL tail lines — the terminal empty split element is dropped."""
    r = d.Runner(tmp_path / "logs", resume=False, dry=False)
    with pytest.raises(RuntimeError) as ei:
        r.run("failstep", ["/bin/sh", "-c", "echo line1; echo line2; exit 3"])
    msg = str(ei.value)
    assert msg.endswith("line2"), msg  # no trailing blank tail slot
    assert "line1" in msg


def test_model_phases_gate_before_any_dispatch():
    """Every MODEL phase calls ensure_model_venv BEFORE its first Runner
    dispatch — the sequencing invariant that closes epm:failure v1 (plan §10
    went straight provision -> p1_pilot; the env_smoke guard never ran)."""
    import inspect

    for fn in (d.phase_p1, d.phase_p2, d.phase_p4_topup, d.phase_p4):
        src = inspect.getsource(fn)
        gate = src.index("ensure_model_venv(args, runner)")
        dispatch_sites = [
            i for i in (src.find("runner.fanout("), src.find("runner.run(")) if i != -1
        ]
        assert dispatch_sites, fn.__name__
        assert gate < min(dispatch_sites), fn.__name__


# ---------------------------------------------------------------------------
# r7: banned accel dists (epm:failure v3 flashinfer-py311-array-subscript)
# ---------------------------------------------------------------------------


def _fake_interpreter_at(path: Path, payload: dict) -> str:
    """Boundary-fake interpreter at an EXPLICIT path (the existing-venv case)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho '" + json.dumps(payload) + "'\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _fake_uv_repair(tmp_path: Path, venv_py: Path, clean_template: str | None) -> tuple[Path, Path]:
    """Executable `uv` boundary fake for the REPAIR path: `uv pip uninstall`
    swaps the venv interpreter for the clean template (uninstall worked); a
    None template leaves the dirty interpreter in place (uninstall failed).
    Every invocation's argv is appended to uv_argv.log."""
    bindir = tmp_path / "uvbin"
    bindir.mkdir(exist_ok=True)
    uv_log = tmp_path / "uv_argv.log"
    swap = (
        f'  cp "{clean_template}" "{venv_py}" && chmod +x "{venv_py}"\n'
        if clean_template is not None
        else "  :\n"
    )
    uv = bindir / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{uv_log}"\n'
        'if [ "$1" = "pip" ] && [ "$2" = "uninstall" ]; then\n'
        f"{swap}"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    return bindir, uv_log


def test_ensure_repairs_existing_venv_with_banned_dist(monkeypatch, tmp_path):
    """The r7 pod case: the EXISTING venv probes healthy on pins but carries
    the banned flashinfer import — ensure must REPAIR IN PLACE (no `uv venv`
    create), run the uninstall step AFTER install (the install re-adds vllm's
    hard-pinned dep), re-probe clean, record the ban, and clear the stale
    env_smoke ok-flag so the extended smoke re-runs. Fails pre-fix: the r6
    ensure read the pins-only probe as healthy and never repaired."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    monkeypatch.delenv(cm.MODEL_PY_ENV, raising=False)
    venv = tmp_path / "venv"
    monkeypatch.setattr(cm, "MODEL_VENV_DEFAULT", str(venv))
    venv_py = venv / "bin" / "python"
    _fake_interpreter_at(venv_py, {**_good_payload(), "banned_present": ["flashinfer"]})
    clean = _fake_interpreter(tmp_path, {**_good_payload(), "banned_present": []})
    bindir, uv_log = _fake_uv_repair(tmp_path, venv_py, clean)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    runner = d.Runner(tmp_path / "logs", resume=True, dry=False)
    smoke_argv = d._model_py("issue2378_dispatch.py", "--phase", "env_smoke")
    runner._ok_path("model_env_smoke").write_text(d._argv_sha(smoke_argv))  # stale flag
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    d.ensure_model_venv(args, runner)
    calls = [c for c in uv_log.read_text(encoding="utf-8").split("\n") if c]
    assert not any(c.startswith("venv ") for c in calls), calls  # repair, never re-create
    install_idx = next(i for i, c in enumerate(calls) if c.startswith("pip install "))
    uninst_idx = next(i for i, c in enumerate(calls) if c.startswith("pip uninstall "))
    assert install_idx < uninst_idx, calls  # uninstall AFTER install (re-resolve re-adds it)
    assert "flashinfer-python" in calls[uninst_idx]
    rec = json.loads((tmp_path / "ledger" / "model_venv_pins.json").read_text(encoding="utf-8"))
    assert rec["banned"] == ["flashinfer-python"]
    assert rec["realized"]["banned_present"] == []
    # stale ok-flag cleared on the repair branch -> env_smoke re-ran
    assert (tmp_path / "logs" / "model_env_smoke.log").exists()


def test_ensure_repair_leaving_banned_dist_raises(monkeypatch, tmp_path):
    """A repair whose uninstall step did not take fails LOUD naming the banned
    import — never a silent proceed into the engine-init crash."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    monkeypatch.delenv(cm.MODEL_PY_ENV, raising=False)
    venv = tmp_path / "venv"
    monkeypatch.setattr(cm, "MODEL_VENV_DEFAULT", str(venv))
    venv_py = venv / "bin" / "python"
    _fake_interpreter_at(venv_py, {**_good_payload(), "banned_present": ["flashinfer"]})
    bindir, _ = _fake_uv_repair(tmp_path, venv_py, None)  # uninstall is a no-op
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    with pytest.raises(RuntimeError, match="left banned dist import"):
        d.ensure_model_venv(args, runner)


def test_ensure_refuses_repair_over_explicit_override_with_banned(monkeypatch, tmp_path):
    """An explicit $EPM_I2378_MODEL_PY override carrying a banned dist is
    refused, never mutated (same contract as the missing-interpreter case)."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    fake = _fake_interpreter(tmp_path, {**_good_payload(), "banned_present": ["flashinfer"]})
    monkeypatch.setenv(cm.MODEL_PY_ENV, fake)
    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    with pytest.raises(RuntimeError, match="refusing to build over an explicit override"):
        d.ensure_model_venv(args, runner)


def test_build_model_venv_composes_uninstall_banned_step(monkeypatch, tmp_path):
    """Fresh-build path also ends with the uninstall-banned step (the install
    re-adds vllm's hard-pinned flashinfer-python on every re-resolve)."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    monkeypatch.delenv(cm.MODEL_PY_ENV, raising=False)
    venv = tmp_path / "venv"
    monkeypatch.setattr(cm, "MODEL_VENV_DEFAULT", str(venv))
    template = _fake_interpreter(tmp_path, {**_good_payload(), "banned_present": []})
    bindir, uv_log = _fake_uv(tmp_path, template)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    d.ensure_model_venv(args, d.Runner(tmp_path / "logs", resume=False, dry=False))
    calls = [c for c in uv_log.read_text(encoding="utf-8").split("\n") if c]
    assert calls[-1].startswith("pip uninstall "), calls  # LAST step on every build
    assert "flashinfer-python" in calls[-1]


def test_env_smoke_banned_dist_present_raises(monkeypatch):
    """REAL-body env_smoke leg: a banned import name that IS importable in the
    running interpreter (stand-in: pytest itself) fails the gate loudly."""
    monkeypatch.setattr(cm, "MODEL_VENV_BANNED_DISTS", {"pytest": "pytest"})
    with pytest.raises(RuntimeError, match="banned accel dist import"):
        d.phase_env_smoke(SimpleNamespace())


def test_env_smoke_banned_absent_reaches_qwen35_gate():
    """REAL-body env_smoke leg, banned-absent branch: with the default banned
    set (flashinfer is not installed in the repo venv) the gate passes the
    banned check and proceeds to the qwen3_5 gate, which raises on the repo
    venv (transformers 4.57.6) — proving the new branch sits BEFORE it and
    passes cleanly when the dist is absent."""
    with pytest.raises(RuntimeError, match="lacks qwen3_5"):
        d.phase_env_smoke(SimpleNamespace())


def test_env_smoke_compile_backend_import_unguarded_and_ordered():
    """Source pin: env_smoke imports the vLLM compile-backend chain UNGUARDED
    (no try/except anywhere in the phase — TypeError/SyntaxError from a
    py-version-incompatible accel dep must FAIL the gate, the exact class
    vLLM's ImportError-only guard misses), ordered banned-check -> qwen3_5 ->
    compile-backend import -> HF config/tokenizer downloads."""
    import ast
    import inspect
    import textwrap

    src = inspect.getsource(d.phase_env_smoke)
    assert 'importlib.import_module("vllm.compilation.backends")' in src
    # deliberately unguarded: no try/except node anywhere in the phase body
    tree = ast.parse(textwrap.dedent(src))
    assert not any(isinstance(n, ast.Try) for n in ast.walk(tree))
    banned_idx = src.index("banned accel dist import")
    qwen_idx = src.index("lacks qwen3_5")
    compile_idx = src.index('import_module("vllm.compilation.backends")')
    config_idx = src.index("AutoConfig.from_pretrained")
    assert banned_idx < qwen_idx < compile_idx < config_idx


# ---------------------------------------------------------------------------
# r8: launch env pins + engine_smoke gate (epm:failure v4
# flashinfer-absent-sampler-probe-modulenotfound)
# ---------------------------------------------------------------------------


def test_launch_env_pins_flashinfer_sampler_off():
    """vllm 0.27.1 envs.py:848-852: "0" -> bool(int("0")) -> False; UNSET ->
    True (the crash default — the sampler probe assumes flashinfer installed).
    Exact dict equality: the pin set IS the launch-env contract."""
    assert cm.LAUNCH_ENV_PINS == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def test_runner_run_injects_launch_env_pins_over_inherited(tmp_path, monkeypatch):
    """REAL Runner.run body: the child observes the pin, and an INHERITED =1
    (which would deterministically crash engine init on the flashinfer-free
    venv) LOSES to the authoritative pin."""
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    r = d.Runner(tmp_path / "logs", resume=False, dry=False)
    r.run(
        "pin.step",
        [
            sys.executable,
            "-c",
            "import os,sys;"
            "sys.exit(0 if os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')=='0' else 7)",
        ],
    )  # rc!=0 would raise


def test_runner_fanout_and_parallel_inject_launch_env_pins(tmp_path, monkeypatch):
    """REAL fanout/parallel bodies: every shard env carries the pin, beside the
    untouched per-shard CVD launcher pin (#545)."""
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")  # inherited =1 must lose
    r = d.Runner(tmp_path / "logs", resume=False, dry=False)
    argv = [
        sys.executable,
        "-c",
        "import os;print('PIN='+os.environ['VLLM_USE_FLASHINFER_SAMPLER']);"
        "print('CVD='+os.environ['CUDA_VISIBLE_DEVICES'])",
    ]
    r.fanout("pin.fan", argv, gpus=["4", "5"])
    for i, g in enumerate(["4", "5"]):
        log = (tmp_path / "logs" / f"pin.fan.s{i}.log").read_text(encoding="utf-8")
        assert "PIN=0" in log, f"shard {i} missing the sampler-probe pin"
        assert f"CVD={g}" in log
    r.parallel("pin.par", [list(argv)], gpus=["6"])
    log = (tmp_path / "logs" / "pin.par.s0.log").read_text(encoding="utf-8")
    assert "PIN=0" in log and "CVD=6" in log
    # the START lines advertise BOTH pin classes (r8 env pin + r9 engine
    # kwarg pin — the r9 fix-engaged log observable)
    assert d._PINS_TOKEN == "VLLM_USE_FLASHINFER_SAMPLER=0,engine:gdn_prefill_backend=triton"


def test_engine_smoke_phase_registered_inline():
    """engine_smoke is a PHASES arm handled inline (model venv + 1 GPU req'd),
    exactly like env_smoke."""
    assert "engine_smoke" in d.PHASES and d.PHASES["engine_smoke"] is None
    import inspect

    src = inspect.getsource(d.main)
    assert 'args.phase == "engine_smoke"' in src


def test_ensure_runs_engine_smoke_after_env_smoke(monkeypatch, tmp_path):
    """REAL ensure body: BOTH gates run (env_smoke THEN engine_smoke — walls
    dict preserves execution order), and the engine gate's argv routes via the
    MODEL interpreter with --phase engine_smoke (pinned by the ok-flag sha)."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    fake = _fake_interpreter(tmp_path, _good_payload())
    monkeypatch.setenv(cm.MODEL_PY_ENV, fake)
    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    (tmp_path / "ledger").mkdir()
    d.ensure_model_venv(args, runner)
    assert (tmp_path / "logs" / "model_engine_smoke.log").exists()
    assert list(runner.walls)[-2:] == ["model_env_smoke", "model_engine_smoke"]
    expected = d._model_py("issue2378_dispatch.py", "--phase", "engine_smoke")
    ok = (tmp_path / "logs" / "model_engine_smoke.ok").read_text(encoding="utf-8")
    assert ok == d._argv_sha(expected)


def test_ensure_engine_smoke_fails_loud_on_zero_gpu_host(monkeypatch, tmp_path):
    """A zero-GPU real host cannot run the engine gate — ensure fail-louds at
    _first_gpu_env (naming the step) instead of deferring the failure to the
    fan-out; env_smoke has already run (gate order preserved)."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)  # undo autouse pin
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))  # no nvidia-smi -> visible_gpus() == []
    fake = _fake_interpreter(tmp_path, _good_payload())
    monkeypatch.setenv(cm.MODEL_PY_ENV, fake)
    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    (tmp_path / "ledger").mkdir()
    with pytest.raises(RuntimeError, match="model_engine_smoke: no visible GPUs"):
        d.ensure_model_venv(args, runner)
    assert (tmp_path / "logs" / "model_env_smoke.log").exists()


def test_rebuild_clears_stale_engine_smoke_ok_flag(monkeypatch, tmp_path):
    """r8: a build/repair invalidates the ENGINE smoke ok-flag too — the argv
    sha is unchanged across a rebuild (same overlay-wipe residue as the
    env_smoke flag), so without the clear a resumed runner would silently
    reuse the OLD engine verdict under a rebuilt interpreter."""
    monkeypatch.setenv(cm.SKIP_DRIVER_PROBE_ENV, "1")
    monkeypatch.delenv(cm.MODEL_PY_ENV, raising=False)
    venv = tmp_path / "venv"
    monkeypatch.setattr(cm, "MODEL_VENV_DEFAULT", str(venv))
    template = _fake_interpreter(tmp_path, _good_payload())
    bindir, _ = _fake_uv(tmp_path, template)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    runner = d.Runner(tmp_path / "logs", resume=True, dry=False)
    for step, phase in (
        ("model_env_smoke", "env_smoke"),
        ("model_engine_smoke", "engine_smoke"),
    ):
        argv = d._model_py("issue2378_dispatch.py", "--phase", phase)
        runner._ok_path(step).write_text(d._argv_sha(argv))  # stale flags
    args = SimpleNamespace(ledger_root=str(tmp_path / "ledger"))
    d.ensure_model_venv(args, runner)
    for step in ("model_env_smoke", "model_engine_smoke"):
        assert (tmp_path / "logs" / f"{step}.log").exists(), f"{step} skipped on stale flag"


def test_engine_smoke_body_pins_known_gotchas():
    """Source pins for the pod-only engine gate body: spawn set BEFORE the
    first vllm import (#628 fork-poisoned EngineCore), use_tqdm=False (#613
    tqdm ZeroDivision), enforce_eager=True (init-path gate, no cudagraph
    wall), the shards' own create_vllm_engine seam, and the os._exit(0)
    terminal (#1739/#2149 finalization deadlock on engine children). The
    dispatcher module top must stay vllm-free (repo venv)."""
    import inspect

    src = inspect.getsource(d.phase_engine_smoke)
    assert 'os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")' in src
    assert src.index('setdefault("VLLM_WORKER_MULTIPROC_METHOD"') < src.index("from vllm import")
    assert "use_tqdm=False" in src
    assert "enforce_eager=True" in src
    assert "create_vllm_engine(" in src
    assert "os._exit(0)" in src
    top = DISPATCH_SRC[: DISPATCH_SRC.index("def _log")]
    assert "import vllm" not in top and "from vllm" not in top


# ---------------------------------------------------------------------------
# r9: GDN prefill engine-kwarg pin (epm:failure v5
# flashinfer-absent-gdn-prefill-modulenotfound)
# ---------------------------------------------------------------------------


def test_engine_kwarg_pins_gdn_prefill_triton():
    """vllm 0.27.1 qwen_gdn_linear_attn.py:85-133 (tag = 6e448d0ea9bf): the
    GDN prefill resolver reads additional_config["gdn_prefill_backend"]
    (default "auto") and on SM90 auto-selects "flashinfer" with NO
    availability check, then hard-imports flashinfer.gdn_prefill at the
    FIRST prefill (:174). "triton" routes to the in-tree FLA kernels. Exact
    dict equality: the pin set IS the engine-kwarg contract; it threads ONLY
    as the EngineArgs field (arg_utils.py:752 -> additional_config
    :2459-2460) — no env-var route exists for this knob."""
    assert cm.ENGINE_KWARG_PINS == {"gdn_prefill_backend": "triton"}


def test_build_engine_composes_gdn_pin_real_body(monkeypatch, tmp_path):
    """REAL gen._build_engine body: the composed create_vllm_engine kwargs
    carry the GDN pin UNCONDITIONALLY — even when EngineArgs LACKS the field
    (an old-engine stub), proving the pin is NOT introspection-guarded (a
    silent skip would re-expose the SM90 flashinfer auto-select), while the
    language_model_only OPTIMIZATION stays introspection-guarded (skipped on
    the stub). The fake engine factory signature-binds against the real
    create_vllm_engine before capturing."""
    import dataclasses
    import inspect
    from types import ModuleType

    import issue2378_gen as g

    from explore_persona_space.eval import generation as eval_gen

    # Stub vllm.engine.arg_utils with an EngineArgs LACKING gdn_prefill_backend
    # AND language_model_only (the VM's pre-GDN vLLM shape).
    stub_args_mod = ModuleType("vllm.engine.arg_utils")

    @dataclasses.dataclass
    class EngineArgs:
        model: str = ""

    stub_args_mod.EngineArgs = EngineArgs
    stub_engine_pkg = ModuleType("vllm.engine")
    stub_engine_pkg.arg_utils = stub_args_mod
    stub_vllm_pkg = ModuleType("vllm")
    stub_vllm_pkg.engine = stub_engine_pkg
    monkeypatch.setitem(sys.modules, "vllm", stub_vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.engine", stub_engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", stub_args_mod)

    real = eval_gen.create_vllm_engine
    captured: dict = {}

    def fake_create(model_path, **kwargs):
        inspect.signature(real).bind(model_path, **kwargs)  # conformant by construction
        captured["model_path"] = model_path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(eval_gen, "create_vllm_engine", fake_create)
    args = SimpleNamespace(tp=1, gpu_memory_utilization=None, max_model_len=64, max_num_seqs=2)
    g._build_engine(args)
    assert captured["model_path"] == cm.MODEL_ID
    assert captured["gdn_prefill_backend"] == "triton", "GDN pin missing from composed kwargs"
    assert "language_model_only" not in captured, (
        "language_model_only must stay introspection-guarded (stub EngineArgs lacks it)"
    )


def test_engine_smoke_body_composes_gdn_pin():
    """Source pin: the engine gate threads cm.ENGINE_KWARG_PINS into its
    create_vllm_engine call BEFORE the **kwargs splat (gate/shard parity —
    the v5 crash fired at the FIRST prefill of exactly this gate's generate,
    so the gate is the fix-engaged vehicle). gen._build_engine carries the
    same update (source-pinned here; real-body leg above)."""
    import inspect

    import issue2378_gen as g

    src = inspect.getsource(d.phase_engine_smoke)
    assert "kwargs.update(cm.ENGINE_KWARG_PINS)" in src
    assert src.index("kwargs.update(cm.ENGINE_KWARG_PINS)") < src.index("llm = create_vllm_engine(")
    gsrc = inspect.getsource(g._build_engine)
    assert "kwargs.update(cm.ENGINE_KWARG_PINS)" in gsrc
    assert gsrc.index("kwargs.update(cm.ENGINE_KWARG_PINS)") < gsrc.index(
        "return create_vllm_engine("
    )


# ---------------------------------------------------------------------------
# r10: reconciler-v6 deferred duties — D1 (standalone engine_smoke model-env
# parity) + D2 (bounded engine-smoke gate with a DEFINED failure path)
# ---------------------------------------------------------------------------


def test_runner_run_timeout_bounds_hung_step(tmp_path):
    """D2(i): a hung gate subprocess is KILLED at timeout_s and surfaces as a
    loud RuntimeError carrying the log tail — never an unbounded hang (the
    vLLM generate()-hang class). Fails pre-r10: Runner.run had no timeout_s
    parameter (TypeError) and an unbounded subprocess.run."""
    import time

    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    argv = [
        sys.executable,
        "-c",
        "print('hung-probe-line', flush=True); import time; time.sleep(60)",
    ]
    t0 = time.time()
    with pytest.raises(RuntimeError, match="TIMED OUT after 1s") as ei:
        runner.run("hang_probe", argv, timeout_s=1.0, tail_lines=5)
    assert time.time() - t0 < 30, "timeout did not bound the hung child"
    assert "hung-probe-line" in str(ei.value), "raise must surface the gate log tail"
    # No ok-flag on a timed-out step (a resume must re-run it).
    assert not (tmp_path / "logs" / "hang_probe.ok").exists()


def test_runner_run_failure_raise_carries_tail_lines(tmp_path):
    """D2(ii): the non-zero-rc raise carries exactly the LAST tail_lines log
    lines (parametrized so the engine gate can surface ~40)."""
    runner = d.Runner(tmp_path / "logs", resume=False, dry=False)
    body = "\n".join(f"print({i})" for i in range(10)) + "\nraise SystemExit(3)"
    with pytest.raises(RuntimeError) as ei:
        runner.run("fail_probe", [sys.executable, "-c", body], tail_lines=4)
    msg = str(ei.value)
    assert "failed rc=3" in msg
    assert msg.endswith("6\n7\n8\n9"), msg


def test_engine_smoke_gate_call_sites_all_bounded():
    """D2: EVERY model_engine_smoke runner.run call site (ensure dry branch,
    ensure real branch, the D1 standalone gate) passes the wall-clock bound +
    the 40-line tail. The first-arg form `"model_engine_smoke",` is distinct
    from _first_gpu_env's what-arg form `"model_engine_smoke")`."""
    sites = [m.start() for m in re.finditer(r'"model_engine_smoke",', DISPATCH_SRC)]
    assert len(sites) == 3, f"expected 3 gate call sites, found {len(sites)}"
    for pos in sites:
        window = DISPATCH_SRC[pos : pos + 500]
        assert "timeout_s=ENGINE_SMOKE_TIMEOUT_S" in window, window
        assert "tail_lines=ENGINE_SMOKE_TAIL_LINES" in window, window


def test_engine_smoke_body_bounded_failure_path():
    """D2(ii): the body's failure path is DESIGNED — traceback into the gate
    log + flush + os._exit(1) — never a bare assert whose raise can DEADLOCK
    interpreter finalization on surviving engine children (#1739/#2149, the
    r<10 success-only exit's reachable hang). D1: the in-body LAUNCH_ENV_PINS
    setdefault precedes the vllm import (pin parity on a direct model-python
    invocation; setdefault never clobbers the launcher-supplied env)."""
    import inspect

    src = inspect.getsource(d.phase_engine_smoke)
    assert "except BaseException" in src
    assert "traceback.print_exc()" in src
    assert "os._exit(1)" in src
    assert "cm.LAUNCH_ENV_PINS.items()" in src
    assert src.index("cm.LAUNCH_ENV_PINS.items()") < src.index("from vllm import")


def test_standalone_engine_smoke_redispatches_via_model_env(tmp_path, capsys):
    """D1 (reconciler-v6 standalone-engine-smoke-bypasses-model-env): the
    standalone repo-venv `--phase engine_smoke` entry re-dispatches the gate
    as the SAME composed subprocess ensure_model_venv runs — model-venv
    interpreter + Runner env merge (LAUNCH_ENV_PINS) + single-GPU CVD pin +
    D2 bound — never the in-process body under the repo venv (vLLM 0.11.0,
    no qwen3_5). Dry-run composition pin; the real leg lands pod-side."""
    assert not d._is_model_interpreter(), "pytest must run under the repo venv"
    args = SimpleNamespace(dry_run=True, logs_dir=str(tmp_path / "logs"))
    rc = d._run_engine_smoke_gate(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert d._model_python() in out, out
    assert "--phase engine_smoke" in out, out
    import inspect

    src = inspect.getsource(d.main)
    assert "_is_model_interpreter()" in src
    assert "_run_engine_smoke_gate(" in src
