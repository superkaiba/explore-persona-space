"""Tests for independent rate selection and immutable source/phase boundaries."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts import context_risk_highrate_design as design

SOURCE = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/data/"
    "impossible_livecodebench/public_development_manifest.jsonl"
)


def screen_counts():
    """Make explicit software-fixture outcomes with a known ordering and censors."""
    rows = []
    for task in range(103):
        for condition in sorted(design.CONDITIONS):
            success = int(task < 15) + int(task < 5)
            unknown = int(task == 102)
            if condition == "original":
                success, unknown = 1, 0
            rows.append(
                {
                    "task_id": f"lcbhard_{task}",
                    "condition": condition,
                    "success": success,
                    "failure": 2 - success - unknown,
                    "censored": unknown,
                    "missing": 0,
                }
            )
    return rows


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """Exercise real source hashing/freezing using isolated software-review inputs."""
    project = tmp_path / "project"
    for relative in design.COLLECTION_SOURCES:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"SOFTWARE TEST FIXTURE, NOT A RESEARCH RESULT: {relative}\n")
    monkeypatch.setattr(design, "PROJECT", project)
    monkeypatch.setattr(design, "DESIGN", project / "eval_results/context_risk_highrate_design")
    review = tmp_path / "software_review.json"
    review.write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "reviewer": "software fixture only, not a production critic",
                "files_sha256": design.source_hashes(),
            }
        )
    )
    root = tmp_path / "run"
    receipt = design.freeze(SOURCE, root, review)
    return root, receipt, review, project


def test_official_source_and_disjoint_request_streams():
    rows = design.source_rows(SOURCE)
    assert len(rows) == 309
    assert len({r["task_id"] for r in rows}) == 103
    seeds = design.check_seed_streams(rows)
    assert seeds["possible_request_seeds"] == 18540
    assert seeds["unique"] is True
    assert seeds["previous_unique_possible_seeds"] == 37080
    assert seeds["previous_seed_overlap"] == 0


def test_real_freeze_and_load_phase(frozen):
    root, receipt, review, _ = frozen
    manifest, loaded, epochs = design.load_phase(root, "screen")
    assert receipt == loaded
    assert epochs == 2
    assert len(design.load_samples(manifest)) * epochs == 618
    assert (root / "manifests/source.jsonl").read_bytes() == SOURCE.read_bytes()
    with pytest.raises(FileExistsError):
        design.freeze(SOURCE, root, review)


def test_changed_producer_is_rejected(frozen):
    root, _, review, project = frozen
    (project / design.COLLECTION_SOURCES[0]).write_text("changed source")
    with pytest.raises(ValueError, match="changed after independent review"):
        design.validate_review(review)
    with pytest.raises(ValueError, match="source binding changed"):
        design.load_phase(root, "screen")


def test_rehashed_manifest_cannot_change_official_task(frozen):
    root, receipt, _, _ = frozen
    manifest = root / "manifests/screen_B.jsonl"
    rows = design.read_rows(manifest)
    rows[0]["prompt"] += "\n# unauthorized task edit"
    design._write_jsonl_atomic(manifest, rows)
    receipt["manifest_sha256"] = design.sha256(manifest)
    design._write_json_atomic(root / "manifests/screen_freeze.json", receipt)
    with pytest.raises(ValueError, match="does not reproduce the official source"):
        design.load_phase(root, "screen")


def test_known_rate_order_and_exact_group_split():
    ranked, roles = design.rank_tasks(screen_counts())
    assert {r["task_id"] for r in ranked[:5]} == {f"lcbhard_{i}" for i in range(5)}
    assert {r["task_id"] for r in ranked[5:15]} == {f"lcbhard_{i}" for i in range(5, 15)}
    assert [r["rate_lower"] for r in ranked] == sorted(
        [r["rate_lower"] for r in ranked], reverse=True
    )
    assert Counter(roles.values()) == {"probe_training": 20, "final_test": 10}
    assert set(roles) == {r["task_id"] for r in ranked[:30]}
    for index in range(0, 30, 3):
        assert Counter(roles[r["task_id"]] for r in ranked[index : index + 3]) == {
            "probe_training": 2,
            "final_test": 1,
        }
    assert design.rank_tasks(list(reversed(screen_counts()))) == (ranked, roles)


def test_unknown_outcomes_are_never_imputed_as_failures():
    ranked, _ = design.rank_tasks(screen_counts())
    unknown = next(r for r in ranked if r["task_id"] == "lcbhard_102")
    assert (unknown["success"], unknown["failure"], unknown["censored"]) == (0, 2, 2)
    assert (unknown["rate_lower"], unknown["rate_upper"]) == (0.0, 0.5)
    assert unknown["complete_case_rate"] == 0.0
    all_unknown = screen_counts()
    for row in all_unknown:
        if row["task_id"] == "lcbhard_102" and row["condition"] != "original":
            row.update(failure=0, censored=2)
    ranked, _ = design.rank_tasks(all_unknown)
    unknown = next(r for r in ranked if r["task_id"] == "lcbhard_102")
    assert unknown["complete_case_rate"] is None
    assert unknown["rate_upper"] == 1.0


@pytest.mark.parametrize(
    "field,value", [("success", -1), ("failure", 3), ("censored", True), ("missing", 1)]
)
def test_invalid_or_missing_screen_counts_fail(field, value):
    rows = screen_counts()
    rows[0][field] = value
    with pytest.raises(ValueError):
        design.rank_tasks(rows)


def test_incomplete_or_duplicate_task_roster_fails():
    for rows in (screen_counts()[:-1], [*screen_counts(), copy.deepcopy(screen_counts()[0])]):
        with pytest.raises(ValueError):
            design.rank_tasks(rows)


def test_fresh_rows_keep_all_task_siblings_together():
    _, roles = design.rank_tasks(screen_counts())
    fresh = design.make_rows(design.source_rows(SOURCE), "fresh", roles)
    assert len(fresh) == 90
    assert len({r["exact_context_sha256"] for r in fresh}) == 90
    assert Counter(r["public_test_role"] for r in fresh) == {
        "probe_training": 60,
        "final_test": 30,
    }
    for row in fresh:
        assert row["partition"] == roles[row["task_id"]]
        assert row["sample_id"].startswith("highrate_fresh:B:")
        assert "screen" not in row["messages"][0]["content"][: len(design.instruction("B"))]


def test_missing_terminal_receipt_is_not_completion(tmp_path):
    with pytest.raises(FileNotFoundError):
        design.validate_terminal_process(tmp_path, "fresh")


def test_native_terminal_process_roundtrip_and_evidence_mutations(tmp_path, monkeypatch):
    """Use real Inspect/Docker output; only launch identity is a short local process fixture."""
    from scripts import context_risk_highrate_collect as collect
    from tests.test_context_risk_highrate_collect import fixture_phase, install_fixture_boundaries

    root, cfg, _ = fixture_phase(tmp_path, monkeypatch)
    install_fixture_boundaries(monkeypatch)
    started = datetime.now(UTC).isoformat()
    actors = [subprocess.Popen([sys.executable, "-c", "pass"]) for _ in range(2)]
    for actor in actors:
        assert actor.wait(timeout=10) == 0
    report = collect.run(cfg)
    launch_id = "software_terminal_fixture"
    prefix = root / f"fresh_{launch_id}_process"
    pids = dict(zip(("supervisor_pid", "worker_pid"), (a.pid for a in actors), strict=True))
    for key, suffix in (("supervisor_pid", "pid"), ("worker_pid", "worker.pid")):
        Path(f"{prefix}.{suffix}").write_text(str(pids[key]))
    log = root / "software_launch.log"
    log.write_text(
        f"[supervisor-start] mode=fresh pid={pids['supervisor_pid']}\n"
        f"[worker-start] mode=fresh pid={pids['worker_pid']}\n"
    )
    exit_path = Path(f"{prefix}.exit.json")
    design._write_json_atomic(
        exit_path,
        {
            "mode": "fresh",
            **pids,
            "exit_code": 0,
            "cleanup": "no_live_members",
            "finished_unix": time.time(),
        },
    )
    launch_path = root / "software_launch.json"
    design._write_json_atomic(
        launch_path,
        {
            "phase": "fresh",
            "mode": "fresh",
            "launch_id": launch_id,
            "started_utc": started,
            "log_path": str(log),
            **pids,
        },
    )
    receipt = design.record_terminal_process(root, "fresh", launch_path)
    assert design.validate_terminal_process(root, "fresh") == receipt
    assert set(report["native_logs_sha256"]) <= set(receipt["evidence_sha256"])
    with pytest.raises(FileExistsError):
        design.record_terminal_process(root, "fresh", launch_path)
    target = root / "fresh_B/terminal_process.json"
    # Inspect headers have whole-second precision. Exercise both sides of that boundary
    # using the actual saved native header and explicitly mutated fixture launch receipts.
    from inspect_ai.log import read_eval_log

    native = read_eval_log(next(iter(report["native_logs_sha256"])), header_only=True)
    native_begin = design._timestamp(native.stats.started_at)
    assert native_begin == int(native_begin)
    original_launch = json.loads(launch_path.read_text())
    for offset, accepted in ((0.5, True), (1.5, False)):
        changed_launch = copy.deepcopy(original_launch)
        changed_launch["started_utc"] = datetime.fromtimestamp(
            native_begin + offset, UTC
        ).isoformat()
        design._write_json_atomic(launch_path, changed_launch)
        changed_receipt = copy.deepcopy(receipt)
        changed_receipt["evidence_sha256"][str(launch_path)] = design.sha256(launch_path)
        design._write_json_atomic(target, changed_receipt)
        if accepted:
            assert design.validate_terminal_process(root, "fresh") == changed_receipt
        else:
            with pytest.raises(ValueError, match="not terminal within the owned launch"):
                design.validate_terminal_process(root, "fresh")
    design._write_json_atomic(launch_path, original_launch)
    design._write_json_atomic(target, receipt)
    mutated = copy.deepcopy(receipt)
    del mutated["evidence_sha256"][str(log)]
    design._write_json_atomic(target, mutated)
    with pytest.raises(ValueError, match="roster differs"):
        design.validate_terminal_process(root, "fresh")
    log.write_text("different fixture launch")
    mutated = copy.deepcopy(receipt)
    mutated["evidence_sha256"][str(log)] = design.sha256(log)
    design._write_json_atomic(target, mutated)
    with pytest.raises(ValueError, match="identity differs"):
        design.validate_terminal_process(root, "fresh")


def test_select_persists_exact_ranking_and_fresh_source(frozen, monkeypatch):
    """Isolate selection with input fixtures; separately test native/process validation."""
    from scripts import context_risk_highrate_collect as collect

    root, _, _, _ = frozen
    stage = root / "screen_B"
    stage.mkdir()
    contexts = screen_counts()
    audit = {
        "verification_passed": True,
        "contexts": contexts,
        "counts": {
            key: sum(r[key] for r in contexts)
            for key in ("success", "failure", "censored", "missing")
        },
        "native_logs_sha256": {"software-fixture-native": "fixture-hash"},
    }
    design._write_json_atomic(stage / "native_audit.json", audit)
    terminal = {"verification_passed": True, "verified_unix": time.time()}
    design._write_json_atomic(stage / "terminal_process.json", terminal)
    rollouts = [
        {
            "sample_id": "highrate_screen:B:lcbhard_0:oneoff",
            "epoch": 1,
            "metadata": {
                "exact_context_sha256": "fixture-context",
                "agentic_results": {
                    "attempts": [{"code": "SOFTWARE FIXTURE ONLY"}],
                },
            },
            "scores": {"successful_submission": {"value": "C"}},
        }
    ]
    design._write_jsonl_atomic(stage / "rollouts.jsonl", rollouts)
    review = {
        "verdict": "PASS",
        "reviewer": "software fixture only",
        "native_logs_sha256": audit["native_logs_sha256"],
        "success_evidence": {},
    }
    review_path = stage / "success_review.json"
    design._write_json_atomic(review_path, review)
    monkeypatch.setattr(collect, "verify_report", lambda *_: audit)
    monkeypatch.setattr(design, "validate_terminal_process", lambda *_: terminal)
    with pytest.raises(ValueError, match="lack an exact independent review"):
        design.select(root)
    review["success_evidence"] = design.success_evidence(root, "screen")
    design._write_json_atomic(review_path, review)
    selected = design.select(root)
    path, loaded, epochs = design.load_phase(root, "fresh")
    assert loaded == selected
    assert epochs == 4 and len(design.read_rows(path)) == 90
    assert selected["task_roles"] == design.rank_tasks(contexts)[1]
    with pytest.raises(FileExistsError):
        design.select(root)
    # A self-consistent rehash cannot replace the predetermined split.
    selected["ranking"][0]["success"] -= 1
    design._write_json_atomic(root / "selection.json", selected)
    with pytest.raises(ValueError, match="empirical ranking rule"):
        design.load_phase(root, "fresh")
