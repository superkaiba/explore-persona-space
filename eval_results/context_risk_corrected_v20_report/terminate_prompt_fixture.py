"""Isolated AST/mocked check of the managed CLI confirmation flag. No live actions."""

import argparse
import ast
import contextlib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
SOURCE = WORK / "scripts/context_risk_corrected_finish.py"
LIFECYCLE = ROOT / "scripts/pod_lifecycle.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    text = SOURCE.read_text()
    prior = text.replace('                "--yes",\n', "")
    assert hashlib.sha256(prior.encode()).hexdigest() == (
        "04eb2c59060c5bee823bebcede79596f4fa75ea00424f28891afd61b83dffe37"
    )
    assert text.count('"--yes"') == 1
    node = next(
        n
        for n in ast.parse(LIFECYCLE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "cmd_terminate"
    )
    results = []
    for yes, guard_failure in ((True, False), (False, False), (True, True)):
        calls = []
        own = SimpleNamespace(name="pod-2670-corrected", pod_id="own", desired_status="RUNNING")
        sibling = SimpleNamespace(
            name="pod-2670-unrelated", pod_id="other", desired_status="RUNNING"
        )

        def upload_guard(*args, calls=calls, guard_failure=guard_failure, **kwargs):
            calls.append("upload_guard")
            assert kwargs["skip_flag"] is False
            if guard_failure:
                raise RuntimeError("fixture upload guard refusal")

        def prompt(*args, calls=calls):
            calls.append("prompt")
            raise EOFError("fixture detached stdin")

        def owner_guard(issue, names, calls=calls, own=own, **kwargs):
            assert names == [own.name] and not kwargs["force_flag"]
            calls.append("owner_guard")

        def terminate(pod_id, calls=calls):
            assert pod_id == "own"
            calls.append("terminate")

        namespace = {
            "argparse": argparse,
            "_resolve_terminate_selector": lambda args: (args.name_suffix, False),
            "_guard_keep_running_before_terminate": lambda *a, calls=calls, **k: calls.append(
                "keep_guard"
            ),
            "_guard_upload_verification_before_terminate": upload_guard,
            "_canonical_pod_name": lambda issue, suffix: f"pod-{issue}-{suffix}",
            "_live_pods_for_issue": lambda issue, own=own, sibling=sibling: [own, sibling],
            "_guard_owner_fence_before_terminate": owner_guard,
            "input": prompt,
            "print": lambda *a, **k: None,
            "_verified_teardown_grant": lambda **k: contextlib.nullcontext(),
            "terminate_pod": terminate,
            "_load_state": lambda: {},
            "_metadata_lock": contextlib.nullcontext,
            "_read_metadata_file": lambda: {},
            "_write_metadata_file": lambda *a, **k: None,
            "_remove_from_pods_conf": lambda *a: None,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(LIFECYCLE), "exec"), namespace)
        args = argparse.Namespace(
            issue=2670, name_suffix="corrected", yes=yes, dry_run=False, skip_upload_verify=False
        )
        error = None
        try:
            namespace["cmd_terminate"](args)
        except (EOFError, RuntimeError) as exc:
            error = str(exc)
        if guard_failure:
            assert error == "fixture upload guard refusal" and "terminate" not in calls
        elif yes:
            assert error is None and calls == [
                "keep_guard",
                "upload_guard",
                "owner_guard",
                "terminate",
            ]
        else:
            assert error == "fixture detached stdin" and "terminate" not in calls
        results.append({"yes": yes, "guard_failure": guard_failure, "passed": True, "calls": calls})
    report = {
        "reviewer": "/root/reward_harness_critic",
        "checked_at": datetime.now(UTC).isoformat(),
        "verdict": "PASS",
        "source_sha256": sha(SOURCE),
        "managed_cli_source_sha256": sha(LIFECYCLE),
        "fixture_source_sha256": sha(Path(__file__)),
        "scope": "Only --yes added; skips input prompt, preserves upload and ownership guards.",
        "cases": results,
        "live_api_calls": 0,
    }
    (OUT / "terminate_prompt_fixture_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
