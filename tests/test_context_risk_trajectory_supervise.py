"""Exercise actual supervisor exit, timeout and owned-descendant cleanup."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/context_risk_trajectory_supervise.sh"


@pytest.mark.parametrize(
    "code,deadline,expected",
    [
        ("pass", 20, 0),
        ("raise SystemExit(7)", 20, 7),
        ("import subprocess,time; subprocess.Popen(['sleep','30']); time.sleep(30)", 1, 124),
    ],
)
def test_owned_process_is_terminal_and_failure_is_preserved(tmp_path, code, deadline, expected):
    sentinel_dir = tmp_path / "sentinels"
    sentinel_dir.mkdir()
    env = {
        **os.environ,
        "EPM_CONTEXT_RISK_TRAJECTORY_PROCESS_ROOT": str(tmp_path),
        "EPM_CONTEXT_RISK_LAUNCH_ID": "fixture",
        "EPM_CONTEXT_RISK_PROCESS_TIMEOUT_SECONDS": str(deadline),
        "EPM_CONTEXT_RISK_SENTINEL_DIR": str(sentinel_dir),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), "fixture", sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "fixture_fixture_process.exit.json").read_text())
    assert receipt["exit_code"] == expected
    assert receipt["cleanup"] in {"no_live_members", "terminated_descendants", "killed_descendants"}
    groups = subprocess.check_output(["ps", "-eo", "pgid=,stat="], text=True)
    assert not any(
        int(line.split()[0]) == receipt["worker_pid"] and not line.split()[1].startswith(("Z", "X"))
        for line in groups.splitlines()
    )
    assert ("[phase=done]" in result.stdout) == (expected == 0)


@pytest.mark.parametrize("failure", ["collision", "write"])
def test_sentinel_failure_and_final_exit_receipt_agree(tmp_path, failure):
    sentinel_dir = tmp_path / "sentinels"
    sentinel_dir.mkdir()
    target = sentinel_dir / "issue-2670-epm_progress-trajectory-fixture-fixture.json"
    if failure == "write":
        target = target.with_suffix(".json.tmp")
    code = "from pathlib import Path; import sys; p=Path(sys.argv[1]); " + (
        "p.write_text('existing')" if failure == "collision" else "p.mkdir()"
    )
    env = {
        **os.environ,
        "EPM_CONTEXT_RISK_TRAJECTORY_PROCESS_ROOT": str(tmp_path),
        "EPM_CONTEXT_RISK_LAUNCH_ID": "fixture",
        "EPM_CONTEXT_RISK_PROCESS_TIMEOUT_SECONDS": "20",
        "EPM_CONTEXT_RISK_SENTINEL_DIR": str(sentinel_dir),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), "fixture", sys.executable, "-c", code, str(target)],
        env=env,
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert result.returncode == 126
    receipt = json.loads((tmp_path / "fixture_fixture_process.exit.json").read_text())
    assert receipt["exit_code"] == result.returncode
    assert "[phase=done]" not in result.stdout
