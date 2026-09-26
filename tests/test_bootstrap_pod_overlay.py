"""Execute the actual bootstrap SSH payloads with isolated filesystem boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BOOTSTRAP = Path(__file__).resolve().parents[1] / "scripts/bootstrap_pod.sh"
ROOT = "/root/eps-kimi-test"


def _part(start: str, end: str) -> str:
    text = BOOTSTRAP.read_text()
    return text[text.index(start) : text.index(end, text.index(start))]


def _materialize(tmp_path: Path, *, root: str = ROOT, phase: int = 5):
    """Use real caller validation and ssh_cmd; replace only the SSH executable."""
    bindir = tmp_path / "capture-bin"
    bindir.mkdir(exist_ok=True)
    captures = tmp_path / "payloads.jsonl"
    if captures.exists():
        captures.unlink()
    ssh = bindir / "ssh"
    ssh.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "with open(os.environ['CAPTURES'],'a') as f:\n"
        " f.write(json.dumps(sys.argv[-1])+'\\n')\n"
    )
    ssh.chmod(0o755)
    validation = _part("# Overlay caller validation:", "# End overlay caller validation.")
    helper = _part("ssh_cmd() {", "# ── Parse arguments")
    if phase == 5:
        body = _part("POD_INTENT_VAL=", "# ── Step 6:")
    else:
        body = _part('if [ "$NO_PREFLIGHT" = true ]; then', "# ── Step 11:")
    script = (
        "set -euo pipefail\n"
        "SSH_OPTS=; HOST=fixture; PORT=22; NO_PREFLIGHT=false\n"
        "EXIT_PREFLIGHT_FAILED=78\n"
        "step() { :; }; log_ok() { :; }; log_warn() { :; }; log_fail() { :; }\n"
        + validation
        + helper
        + body
    )
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "BOOTSTRAP_OVERLAY_ROOT": root,
        "POD_INTENT": "eval",
        "CAPTURES": str(captures),
    }
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    payloads = (
        [json.loads(v) for v in captures.read_text().splitlines()] if captures.exists() else []
    )
    return result, payloads


@pytest.fixture
def remote(tmp_path):
    """Filesystem namespace, measured disk capacity, and uv are external boundaries."""
    workspace = tmp_path / "workspace"
    project = workspace / "explore-persona-space"
    project.mkdir(parents=True)
    home = tmp_path / "root"
    home.mkdir()
    bindir = tmp_path / "remote-bin"
    bindir.mkdir()
    records = tmp_path / "uv.jsonl"
    uv = bindir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,sys\n"
        "keys=('UV_PROJECT_ENVIRONMENT','UV_CACHE_DIR','UV_PYTHON','UV_LINK_MODE','HF_HOME')\n"
        "with open(os.environ['UV_RECORDS'],'a') as f:\n"
        " row={'argv':sys.argv[1:],'env':{k:os.environ.get(k) for k in keys}}\n"
        " f.write(json.dumps(row)+'\\n')\n"
        "if sys.argv[1]=='sync':\n"
        " p=pathlib.Path(os.environ.get('UV_PROJECT_ENVIRONMENT','.venv'))/'bin'\n"
        " p.mkdir(parents=True,exist_ok=True)\n"
        " if not (p/'python').exists(): (p/'python').symlink_to(sys.executable)\n"
        "sys.exit(int(os.environ.get('UV_EXIT','0')))\n"
    )
    uv.chmod(0o755)
    # A real Python child executes the original payload. Only disk_usage's
    # filesystem observation is replaced, so low-space tests are deterministic.
    (bindir / "python3").symlink_to(sys.executable)
    (bindir / "sitecustomize.py").write_text(
        "import collections,os,shutil\n"
        "shutil.disk_usage=lambda p: collections.namedtuple('usage','total used free')"
        "(150_000_000_000,0,int(os.environ['FAKE_FREE']))\n"
    )
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(home),
        "PYTHONPATH": str(bindir),
        "UV_RECORDS": str(records),
        "FAKE_FREE": "100000000000",
    }

    def run(payload: str):
        # Exact production absolute-path namespaces are relocated at the
        # filesystem boundary. No shell helpers or internal guards are stubbed.
        relocated = payload.replace("/root/", f"{home}/").replace("/workspace", str(workspace))
        return subprocess.run(
            ["bash", "-c", relocated], env=env, capture_output=True, text=True, timeout=10
        )

    return run, env, home / "eps-kimi-test", project, records


@pytest.mark.parametrize(
    "root",
    [
        "/tmp/eps-test",
        "/root",
        "/root/eps-",
        "/root/eps-a/../b",
        "/root/eps-a;id",
        "/root/eps-a\nid",
        "/root/eps-a b",
        "/root/eps-$(id)",
        "/root/eps-a/child",
    ],
)
def test_invalid_caller_path_never_reaches_ssh(tmp_path, root):
    result, payloads = _materialize(tmp_path, root=root)
    assert result.returncode == 2
    assert not payloads


def test_overlay_payloads_build_links_and_export_actual_uv_environment(tmp_path, remote):
    run, _env, root, project, records = remote
    result, payloads = _materialize(tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(payloads) == 2
    assert all(f"readonly EPS_BOOTSTRAP_OVERLAY_ROOT='{ROOT}'" in p for p in payloads)
    for payload in payloads:
        result = run(payload)
        assert result.returncode == 0, result.stderr
    assert (project / ".venv").resolve() == root / "base"
    assert (project.parent / ".venv").resolve() == root / "base"
    rows = [json.loads(v) for v in records.read_text().splitlines()]
    sync = next(v for v in rows if v["argv"][0] == "sync")
    assert sync["argv"] == ["sync", "--locked"]
    assert sync["env"]["UV_PROJECT_ENVIRONMENT"] == str(root / "base")
    assert sync["env"]["UV_CACHE_DIR"] == str(root / "uv")
    assert Path(sync["env"]["UV_PYTHON"]).is_file()
    # Rebootstrap preserves owned links and marker; no venv replacement.
    before = (project / ".venv").lstat().st_ino
    assert run(payloads[0]).returncode == 0
    assert (project / ".venv").lstat().st_ino == before


def test_low_local_headroom_refuses_before_links_or_uv(tmp_path, remote):
    run, env, root, project, records = remote
    env["FAKE_FREE"] = "69999999999"
    _, payloads = _materialize(tmp_path)
    result = run(payloads[0])
    assert result.returncode != 0
    assert "70000000000 free bytes" in result.stderr
    assert not root.exists()
    assert not (project / ".venv").is_symlink()
    assert not records.exists()


@pytest.mark.parametrize("kind", ["directory", "foreign-link", "unowned-root", "base-link"])
def test_unrelated_existing_paths_are_preserved(tmp_path, remote, kind):
    run, _env, root, project, records = remote
    _, payloads = _materialize(tmp_path)
    if kind == "directory":
        (project / ".venv").mkdir()
        (project / ".venv" / "keep").write_text("owned by somebody else")
    elif kind == "foreign-link":
        (project.parent / ".venv").symlink_to(tmp_path / "foreign")
    elif kind == "unowned-root":
        root.mkdir()
    else:
        assert run(payloads[0]).returncode == 0
        (root / "base").rmdir()
        (root / "base").symlink_to(tmp_path / "foreign")
    result = run(payloads[0])
    assert result.returncode != 0
    assert "Refusing" in result.stderr
    assert not records.exists()
    if kind == "directory":
        assert (project / ".venv" / "keep").read_text() == "owned by somebody else"
    if kind == "foreign-link":
        assert os.readlink(project.parent / ".venv") == str(tmp_path / "foreign")


def test_preflight_reapplies_overlay_after_dotenv_and_preserves_failure(tmp_path, remote):
    run, env, root, project, records = remote
    original = (
        "UV_PROJECT_ENVIRONMENT=/unrelated/base\nUV_CACHE_DIR=/unrelated/cache\n"
        "UV_PYTHON=/unrelated/python\nPRIVATE_TEST_TOKEN=fixture-secret\n"
    )
    (project / ".env").write_text(original)
    _, payloads = _materialize(tmp_path, phase=10)
    assert len(payloads) == 1
    env["UV_EXIT"] = "17"
    result = run(payloads[0])
    assert result.returncode == 17
    assert (project / ".env").read_text() == original
    assert "fixture-secret" not in result.stdout + result.stderr
    row = json.loads(records.read_text())
    assert row["argv"] == [
        "run",
        "python",
        "-m",
        "explore_persona_space.orchestrate.preflight",
        "--no-gpu",
    ]
    assert row["env"]["UV_PROJECT_ENVIRONMENT"] == str(root / "base")
    assert row["env"]["UV_CACHE_DIR"] == str(root / "uv")
    assert row["env"]["HF_HOME"] == str(project.parent / ".cache/huggingface")


def test_overlay_sync_failure_is_not_hidden_by_tail(tmp_path, remote):
    run, env, _root, _project, _records = remote
    _, payloads = _materialize(tmp_path)
    assert run(payloads[0]).returncode == 0
    env["UV_EXIT"] = "19"
    assert run(payloads[1]).returncode == 19


def test_default_path_has_no_overlay_payload_or_symlinks(tmp_path, remote):
    run, _env, root, project, records = remote
    result, payloads = _materialize(tmp_path, root="")
    assert result.returncode == 0
    assert len(payloads) == 1
    assert "readonly EPS_BOOTSTRAP_OVERLAY_ROOT" not in payloads[0]
    result = run(payloads[0])
    assert result.returncode == 0, result.stderr
    row = json.loads(records.read_text().splitlines()[0])
    assert row["env"]["UV_PROJECT_ENVIRONMENT"] is None
    assert row["env"]["UV_CACHE_DIR"] == str(project.parent / ".cache/uv")
    assert not root.exists()
    assert not (project / ".venv").is_symlink()
