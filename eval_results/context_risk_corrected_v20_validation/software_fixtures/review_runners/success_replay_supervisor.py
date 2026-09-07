"""Bounded CPU dispatch fixture for the saved-success replay supervisor; uv is replaced locally."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

source = Path.cwd() / "scripts/context_risk_corrected_success_replay_supervise.sh"
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
root = Path(tempfile.mkdtemp(prefix="context-risk-success-replay-supervisor-critic-"))
(root / "scripts").mkdir()
(root / "bin").mkdir()
out = root / "out"
out.mkdir()
shutil.copy2(source, root / "scripts" / source.name)
uv = root / "bin/uv"
uv.write_text(
    "#!/usr/bin/env bash\nset -eu\n"
    '[[ "$UV_NO_SYNC" == 1 ]]\n'
    'printf "%s\\n" "$@" > "$EPM_CONTEXT_RISK_REWARD_ROOT/fixture_args.txt"\n'
    "exit 7\n"
)
uv.chmod(0o755)
environment = dict(
    os.environ,
    PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
    EPM_CONTEXT_RISK_REWARD_ROOT=str(out),
    EPM_CONTEXT_RISK_LAUNCH_ID="fixture",
    EPM_CONTEXT_RISK_PROCESS_TIMEOUT_SECONDS="10",
    EPM_CONTEXT_RISK_REPLAY_MANIFEST=str(root / "manifest with spaces.jsonl"),
)
with (root / "fixture.log").open("w") as log:
    process = subprocess.Popen(
        ["bash", str(root / "scripts" / source.name)],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    result = process.wait(timeout=20)
assert result == 7
arguments = (out / "fixture_args.txt").read_text().splitlines()
assert arguments == [
    "run",
    "--with",
    "inspect-ai==0.3.261",
    "--with",
    "openai==3.7.0",
    "python",
    "-m",
    "scripts.context_risk_corrected_success_replay",
    "--run-result",
    str(out / "full/run_result.json"),
    "--manifest",
    str(root / "manifest with spaces.jsonl"),
    "--output-dir",
    str(out / "success_replay_fixture"),
]
evidence = json.loads((out / "success_replay_fixture_process.exit.json").read_text())
assert evidence["mode"] == "success_replay" and evidence["supervisor_pid"] == process.pid
assert evidence["exit_code"] == 7 and evidence["cleanup"] == "no_live_members"
assert (
    int((out / "success_replay_fixture_process.worker.pid").read_text()) == evidence["worker_pid"]
)
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
report = {
    "passed": True,
    "source_sha256": source_hash,
    "case": "pinned_uv_dispatch_and_nonzero_exit_preserved",
    "arguments": arguments,
    "exit_record": evidence,
    "scope": "Fake uv in a temporary PATH; no model, upload, task or pod calls.",
}
(root / "review.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"report": str(root / "review.json"), **report}, indent=2))
