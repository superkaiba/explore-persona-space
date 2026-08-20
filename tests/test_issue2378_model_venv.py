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
    assert set(rec) == {"interpreter", "realized", "pinned", "extra_pins"}
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
    assert set(rec) == {"interpreter", "realized", "pinned", "extra_pins"}
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
