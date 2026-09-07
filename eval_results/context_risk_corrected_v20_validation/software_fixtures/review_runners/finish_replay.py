import ast
import contextlib
import fcntl
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))
import scripts.context_risk_corrected_audit as audit_module
import scripts.context_risk_corrected_finish as m
import scripts.context_risk_impossiblebench_inspect as inspect_module
import scripts.runpod_api as pod_api
import scripts.verify_uploads as verifier

SOURCE = Path(m.__file__)
SOURCE_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
BASE = Path(tempfile.mkdtemp(prefix="context-risk-finish-replay-critic-"))
RESULTS = []
MAP_SHA = "680935a244cc39c29797d66b312b95e3741889a541dcee8bde7c69ac33c5242d"
original_copy = shutil.copy2
names = (
    "context_risk_impossiblebench.py",
    "context_risk_impossiblebench_inspect.py",
    "context_risk_impossiblebench_harness.py",
    "context_risk_corrected_launch.sh",
    "context_risk_corrected_supervise.sh",
    "context_risk_corrected_audit.py",
    "context_risk_corrected_finish.py",
    "context_risk_analyze.py",
)


def record(name, detail):
    RESULTS.append({"case": name, "passed": True, "detail": detail})
    print(json.dumps(RESULTS[-1]), flush=True)


class Fixture:
    def __init__(self, name):
        self.base = BASE / name
        self.work = self.base / "work"
        self.root = self.base / "out"
        self.data = self.base / "data"
        self.stage = self.base / "stage"
        for folder in ("full", "smoke", "setup"):
            (self.root / folder).mkdir(parents=True)
        (self.work / "scripts").mkdir(parents=True)
        self.review = self.work / "eval_results/context_risk_corrected_v20_validation"
        self.review.mkdir(parents=True)
        (self.work / "docs/ideas").mkdir(parents=True)
        (self.work / "docs/ideas/context_risk_corrected_rerun.md").write_text("fixture plan")
        for name in names:
            (self.work / "scripts" / name).write_text("# fixture " + name + "\n")
        for name, src in [
            ("audit_review.json", "scripts/context_risk_corrected_audit.py"),
            ("finish_review.json", "scripts/context_risk_corrected_finish.py"),
        ]:
            (self.review / name).write_text(
                json.dumps(
                    {
                        "verdict": "PASS",
                        "source": src,
                        "source_sha256": hashlib.sha256((self.work / src).read_bytes()).hexdigest(),
                    }
                )
            )
        (self.review / "critic_review.json").write_text("{}")
        (self.root / "full/run_result.json").write_text(json.dumps({"technical_errors": 0}))
        (self.root / "smoke/fixture.txt").write_text("fixture smoke")
        self.calls = []
        self.present = True
        self.fail = None
        self.census_bad = False
        self.wrong_pod = False
        self.mutate_source = False
        self.native_pass = True
        self.pod_files = {
            "server.log": b"completed server\n",
            "nested/runtime.json": b'{"fixture":true}\n',
        }

    def metadata(self, data):
        return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def pod_list(self):
        if not self.present:
            return []
        return [SimpleNamespace(name=m.POD, pod_id="wrong" if self.wrong_pod else m.POD_ID)]

    def wait(self):
        if self.mutate_source:
            path = self.work / "scripts/context_risk_corrected_audit.py"
            path.write_text("approved replacement\n")
            report = {
                "verdict": "PASS",
                "source": "scripts/context_risk_corrected_audit.py",
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            (self.review / "audit_review.json").write_text(json.dumps(report))
        return {"technical_errors": 0 if self.native_pass else 1}

    def audit(self, *args):
        (self.root / "full/rollouts.jsonl").write_text('{"software_fixture":true}\n')
        return {"realized_unique_rollouts": 480, "execution_passed": self.native_pass}

    def remote(self, command, **kwargs):
        assert command[0] == "ssh", command
        program = kwargs["input"]
        compile(program, "remote fixture", "exec")
        if kwargs.get("capture_output"):
            self.calls.append("census")
            census = {name: self.metadata(data) for name, data in self.pod_files.items()}
            if self.census_bad:
                census["server.log"]["sha256"] = "wrong"
            return SimpleNamespace(stdout=json.dumps(census))
        self.calls.append("stop")
        return SimpleNamespace(returncode=0)

    def run(self, command, **kwargs):
        if command[0] == "rsync":
            self.calls.append("rsync")
            for name, data in self.pod_files.items():
                path = self.root / "server" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        elif "scripts/pod.py" in command:
            assert command[-4:] == ["--issue", "2670", "--name-suffix", "corrected"]
            self.calls.append("terminate")
            self.present = False
            if self.fail == "after_terminate":
                self.fail = None
                raise TimeoutError("fixture response lost after termination")
        elif "scripts.context_risk_analyze" in command:
            self.calls.append("analysis")
            if self.fail == "analysis":
                self.fail = None
                raise TimeoutError("fixture analysis interruption")
            out = self.root / "analysis"
            out.mkdir(exist_ok=True)
            (out / "analysis_result.json").write_text(
                json.dumps(
                    {
                        "reward_hacking_feasibility": {
                            "execution_integrity": {"passed": self.native_pass},
                            "prediction_status": "not_run_frozen_gate_failed",
                        }
                    }
                )
            )
        else:
            raise AssertionError(command)

    def upload(self, stage, name):
        self.calls.append("upload_" + name)
        if self.fail == "upload_" + name:
            self.fail = None
            raise RuntimeError("fixture upload interruption")
        result = {"passed": True, "url": "https://fixture.invalid/" + name}
        m.write_json(self.root / (name + "_upload_receipt.json"), result)
        return result

    def marker(self, kind, note):
        self.calls.append(kind)

    def copy(self, src, dst, *args, **kwargs):
        if str(src).endswith("/inspect_ai/traces/trace-3704829.log"):
            Path(dst).write_text("software fixture trace\n")
            return str(dst)
        return original_copy(src, dst, *args, **kwargs)

    def hash(self, path):
        if path == self.data / "qwen38_map_pilot/map_layer_44.npz":
            return MAP_SHA
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def patches(self):
        stack = contextlib.ExitStack()
        for name, value in {
            "WORK": self.work,
            "ROOT": self.root,
            "DATA": self.data,
            "STAGING": self.stage,
            "MANIFEST": self.base / "manifest",
            "wait_for_generation": self.wait,
            "run": self.run,
            "upload_bounded": self.upload,
            "marker": self.marker,
            "sha": self.hash,
        }.items():
            stack.enter_context(patch.object(m, name, value))
        stack.enter_context(patch.object(audit_module, "audit", self.audit))
        stack.enter_context(patch.object(inspect_module, "validate_critic_review", lambda p: None))
        stack.enter_context(patch.object(pod_api, "list_team_pods", self.pod_list))
        stack.enter_context(
            patch.object(
                verifier,
                "check_outroot_residue",
                lambda *a, **k: {"status": "OK", "detail": "software fixture"},
            )
        )
        stack.enter_context(patch.object(m.subprocess, "run", self.remote))
        stack.enter_context(patch.object(shutil, "copy2", self.copy))
        return stack


def expected_failure(fn, exception, text):
    try:
        fn()
    except exception as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError("Expected failure " + text)


# Full workflow, then replay after the pod is already absent.
f = Fixture("normal_and_replay")
with f.patches():
    m.finish()
    assert (
        json.loads((f.root / "completion_status.json").read_text())["state"]
        == "quantitative_complete"
    )
    assert f.calls.count("terminate") == 1 and f.calls.index("upload_raw") < f.calls.index(
        "terminate"
    ) < f.calls.index("analysis")
    first = list(f.calls)
    m.finish()
    assert (
        f.calls.count("stop") == 1
        and f.calls.count("rsync") == 1
        and f.calls.count("terminate") == 1
    )
record(
    "normal_completion_and_replay_after_teardown",
    "Raw verification precedes termination; replay skips SSH/copy/termination.",
)
for failure in ("upload_raw", "after_terminate", "analysis", "upload_analysis"):
    f = Fixture("restart_" + failure)
    f.fail = failure
    with f.patches():
        expected_failure(m.finish, (RuntimeError, TimeoutError), "fixture")
        assert (f.root / "raw_snapshot.json").exists()
        stops = f.calls.count("stop")
        copies = f.calls.count("rsync")
        m.finish()
        assert (
            f.calls.count("stop") == stops
            and f.calls.count("rsync") == copies
            and f.calls.count("terminate") == 1
        )
        assert (
            json.loads((f.root / "completion_status.json").read_text())["state"]
            == "quantitative_complete"
        )
    record(
        "restart_after_" + failure,
        "Restart uses frozen snapshot with no repeated signal/copy or duplicate termination.",
    )
f = Fixture("changed_snapshot")
f.fail = "upload_raw"
with f.patches():
    expected_failure(m.finish, RuntimeError, "fixture")
    (f.stage / "raw/server/server.log").write_text("changed")
    before = len(f.calls)
    expected_failure(m.finish, ValueError, "Frozen raw snapshot changed")
    assert not any(c == "terminate" or c.startswith("upload_") for c in f.calls[before:])
record(
    "snapshot_content_drift_blocks_teardown",
    "Changed frozen file fails before upload or termination.",
)
f = Fixture("census_mismatch")
f.census_bad = True
with f.patches():
    expected_failure(m.finish, ValueError, "Copied pod outputs differ")
    assert "terminate" not in f.calls and "upload_raw" not in f.calls
record(
    "census_copy_mismatch_blocks_teardown",
    "Mismatched pod/copied bytes fail before archive and termination.",
)
f = Fixture("wrong_pod")
f.wrong_pod = True
with f.patches():
    expected_failure(m.finish, ValueError, "Live pod identity differs")
    assert not f.calls
record(
    "wrong_pod_id_blocks_every_remote_action",
    "Exact name with wrong ID rejected before signal/census/upload/termination.",
)
f = Fixture("changed_approved_source")
f.mutate_source = True
with f.patches():
    expected_failure(m.finish, ValueError, "Completion source changed")
    assert not f.calls
record(
    "approved_source_update_during_wait_rejected",
    "Changed source plus new approving review cannot replace frozen start hashes.",
)
# Full exit evidence validation, using local files only.
out = BASE / "exit_checks"
(out / "full").mkdir(parents=True)
launched = datetime.now(UTC) - timedelta(seconds=60)
launch = {
    "pid": 3703711,
    "mode": "full",
    "exit_file": str(out / "process.exit.json"),
    "pid_file": str(out / "process.pid"),
    "launched_utc": launched.isoformat(),
}
(out / "launch.json").write_text(json.dumps(launch))
(out / "process.worker.pid").write_text("12345\n")
(out / "full/run_result.json").write_text('{"fixture":true}')
base_exit = {
    "mode": "full",
    "supervisor_pid": 3703711,
    "worker_pid": 12345,
    "exit_code": 0,
    "cleanup": "no_live_members",
    "finished_unix": int(datetime.now(UTC).timestamp()),
}
with patch.object(m, "ROOT", out), patch.object(m, "LAUNCH", out / "launch.json"):
    (out / "process.exit.json").write_text(json.dumps(base_exit))
    assert m.wait_for_generation() == {"fixture": True}
    for key, value in [
        ("mode", "smoke"),
        ("worker_pid", 99),
        ("cleanup", "failed_verification"),
        ("exit_code", 7),
        ("finished_unix", 0),
    ]:
        value_exit = dict(base_exit)
        value_exit[key] = value
        (out / "process.exit.json").write_text(json.dumps(value_exit))
        expected_failure(
            m.wait_for_generation, RuntimeError, "Generation did not exit successfully"
        )
record(
    "exit_evidence_reconciliation",
    "Valid evidence accepted; wrong mode/worker/cleanup/status/freshness rejected.",
)
# Duplicate coordinator lock does not overwrite status or post a failure.
out = BASE / "lock"
out.mkdir()
(out / "completion_status.json").write_text("unchanged")
markers = []
with (out / "completion.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (
        patch.object(m, "ROOT", out),
        patch.object(m, "load_dotenv", lambda p: None),
        patch.object(m, "marker", lambda *a: markers.append(a)),
        patch.object(sys, "argv", ["fixture"]),
    ):
        expected_failure(m.main, BlockingIOError, "")
assert (out / "completion_status.json").read_text() == "unchanged" and not markers
record(
    "duplicate_lock_is_non_destructive",
    "Lock refusal leaves existing status and task markers untouched.",
)
# Embedded remote programs remain valid Python.
for node in ast.walk(ast.parse(SOURCE.read_text())):
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"stop", "census_program"} for t in node.targets
    ):
        compile(ast.literal_eval(node.value), "embedded remote", "exec")
record("remote_programs_compile", "Both exact embedded Python programs compile locally.")
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA, (
    "Source changed during test run"
)
report = {
    "passed": True,
    "source_sha256": SOURCE_SHA,
    "case_count": len(RESULTS),
    "cases": RESULTS,
    "scope": "Local files; remote/model/upload/task/pod calls mocked. No live actions.",
}
(BASE / "review.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "report": str(BASE / "review.json"),
            "source_sha256": SOURCE_SHA,
            "passed": True,
            "case_count": len(RESULTS),
        }
    ),
    flush=True,
)
