"""Regressions for incomplete-analysis publication and early failure preservation."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.hf_api import RepoFile

from scripts import story_persona_crossmodel_artifacts as crossmodel_artifacts
from scripts import story_persona_qwen38_artifacts as pilot_artifacts
from scripts.story_persona_crossmodel_artifacts import validate_analysis_files


def test_missing_or_mixed_layer_cannot_publish(tmp_path):
    """A completion marker cannot compensate for an absent or stale metric block."""
    files = ["analysis_complete", "summary", "raw_all_layers", "block_15", "block_31", "block_47"]
    for stem in files:
        (tmp_path / f"{stem}.json").write_text(json.dumps({"fingerprint": "current"}))
    with pytest.raises(ValueError, match="exact four"):
        validate_analysis_files(tmp_path, "current", "Qwen/Qwen3.8-27B")
    (tmp_path / "block_63.json").write_text(json.dumps({"fingerprint": "stale"}))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_analysis_files(tmp_path, "current", "Qwen/Qwen3.8-27B")
    (tmp_path / "block_63.json").write_text(json.dumps({"fingerprint": "current"}))
    for path in tmp_path.glob("*.json"):
        value = {"fingerprint": "current", "analysis_fingerprint": "current-analysis"}
        if path.name.startswith("block_"):
            value.update(
                status="complete",
                fits={
                    name: {}
                    for name in ["full_bank", "fit_first_evaluate_last", "fit_last_evaluate_first"]
                },
            )
        path.write_text(json.dumps(value))
    for name in ["centroids.npz", "selected_vectors.npz"]:
        (tmp_path / name).write_bytes(b"fixture")
    outputs = {
        p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in tmp_path.iterdir()
        if p.name != "analysis_complete.json"
    }
    (tmp_path / "analysis_complete.json").write_text(
        json.dumps(
            {
                "fingerprint": "current",
                "analysis_fingerprint": "current-analysis",
                "outputs": outputs,
            }
        )
    )
    validate_analysis_files(tmp_path, "current", "Qwen/Qwen3.8-27B")
    changed = json.loads((tmp_path / "block_63.json").read_text())
    changed["changed"] = True
    (tmp_path / "block_63.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed after completion"):
        validate_analysis_files(tmp_path, "current", "Qwen/Qwen3.8-27B")


def test_early_failure_record_is_written_before_uploader(tmp_path):
    """Execute the wrapper's actual stdlib failure-record body on an empty output."""
    wrapper = (
        Path(__file__).resolve().parents[1] / "scripts/story_persona_crossmodel_workload.sh"
    ).read_text()
    start = wrapper.index("    EPS_FAILURE_RC=\"$failed_rc\" python3 - <<'PY'\n")
    body = wrapper[start:].split("\n", 1)[1].split("\nPY\n", 1)[0]
    out = tmp_path / "out"
    log = tmp_path / "workload.log"
    log.write_text("storage contract refused\n")
    env = dict(
        os.environ,
        EPS_STORY_PERSONA_OUT=str(out),
        EPS_FAILURE_RC="124",
        EPS_STORY_PERSONA_MODEL_KEY="deepseek",
        EPS_STORY_PERSONA_SOURCE_SHA="a" * 40,
        EPS_STORY_MASTER_LOG=str(log),
    )
    subprocess.run(["python3", "-c", body], env=env, check=True)
    record = json.loads((out / "failure.json").read_text())
    assert record["exit_code"] == 124
    assert record["model_key"] == "deepseek"
    assert (out / "failure_workload.log").read_bytes() == log.read_bytes()


@pytest.mark.parametrize("model_key", ["qwen", "deepseek"])
def test_model_publication_uses_only_the_registered_branch(monkeypatch, tmp_path, model_key):
    """The model arm, rather than an environment override, determines its result branch."""
    calls = []

    def record_push(repo, paths, *, expected_branch):
        calls.append((repo, paths, expected_branch))
        return "a" * 40

    monkeypatch.setattr(crossmodel_artifacts, "push_results", record_push)
    paths = [tmp_path / "summary.json"]
    assert crossmodel_artifacts.push_model_results(tmp_path, paths, model_key) == "a" * 40
    assert calls == [(tmp_path, paths, crossmodel_artifacts.RESULT_BRANCHES[model_key])]
    with pytest.raises(ValueError, match="unknown model arm"):
        crossmodel_artifacts.push_model_results(tmp_path, paths, "unregistered")
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("current_branch", "expected_branch"),
    [
        ("main", None),
        (crossmodel_artifacts.RESULT_BRANCHES["deepseek"], None),
        ("main", crossmodel_artifacts.RESULT_BRANCHES["deepseek"]),
        (pilot_artifacts.BRANCH, crossmodel_artifacts.RESULT_BRANCHES["deepseek"]),
    ],
)
def test_result_branch_mismatch_refuses_before_git_mutation(
    monkeypatch, tmp_path, current_branch, expected_branch
):
    """The existing Qwen default remains strict, as does the explicit DeepSeek branch."""
    calls = []

    def read_branch(repo, *args, **kwargs):
        calls.append(args)
        assert args == ("branch", "--show-current")
        return subprocess.CompletedProcess(args, 0, stdout=current_branch + "\n")

    monkeypatch.setattr(pilot_artifacts, "git", read_branch)
    kwargs = {} if expected_branch is None else {"expected_branch": expected_branch}
    with pytest.raises(RuntimeError, match="results must stay on"):
        pilot_artifacts.push_results(tmp_path, [tmp_path / "summary.json"], **kwargs)
    assert calls == [("branch", "--show-current")]


@pytest.mark.parametrize("model_key", ["qwen", "deepseek"])
def test_registered_branch_push_and_remote_verification(monkeypatch, tmp_path, model_key):
    """Exercise the real shared push control flow with an isolated fake Git transport."""
    branch = crossmodel_artifacts.RESULT_BRANCHES[model_key]
    revision, blob = "a" * 40, "b" * 40
    calls = []

    def fake_git(repo, *args, **kwargs):
        calls.append(args)
        replies = {
            ("branch", "--show-current"): branch,
            ("hash-object", "--", "summary.json"): blob,
            ("rev-parse", f"origin/{branch}"): revision,
            ("rev-list", "--count", f"{revision}..HEAD"): "0",
            ("rev-parse", f"{revision}:summary.json"): blob,
        }
        return subprocess.CompletedProcess(args, 0, stdout=replies.get(args, "") + "\n")

    monkeypatch.setattr(pilot_artifacts, "git", fake_git)
    assert (
        crossmodel_artifacts.push_model_results(tmp_path, [tmp_path / "summary.json"], model_key)
        == revision
    )
    assert ("push", "origin", f"HEAD:{branch}") in calls
    assert ("rebase", f"origin/{branch}") in calls
    assert ("rev-parse", f"{revision}:summary.json") in calls


class MemoryHub:
    """Exercise the real sharded uploader against an in-memory, typed Hub transport."""

    def __init__(self, *, quota_after=None, corrupt=False):
        self.files = {}
        self.calls = []
        self.quota_after = quota_after
        self.canonical_commits = 0
        self.corrupt = corrupt

    def repo_info(self, repo_id, *, repo_type):
        self.calls.append(("repo_info", repo_id, repo_type))
        return SimpleNamespace(sha="a" * 40, private=repo_type == "model")

    def create_repo(self, *, repo_id, repo_type, private, exist_ok):
        assert repo_type == "model" and private and exist_ok
        self.files.setdefault((repo_id, repo_type), {})

    def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
        self.files.setdefault((repo_id, repo_type), {})[path_in_repo] = path_or_fileobj.read()

    def create_commit(self, *, repo_id, repo_type, operations, commit_message):
        if repo_type == "dataset":
            if self.quota_after is not None and self.canonical_commits >= self.quota_after:
                raise RuntimeError("403 Forbidden: You have exceeded your public storage space")
            self.canonical_commits += 1
        self.calls.append(("commit", repo_id, repo_type))
        for op in operations:
            data = Path(op.path_or_fileobj).read_bytes()
            if self.corrupt and op.path_in_repo.endswith("chunk.pt"):
                data = bytes([data[0] ^ 1]) + data[1:]
            self.files.setdefault((repo_id, repo_type), {})[op.path_in_repo] = data

    def list_repo_tree(self, repo_id, *, repo_type, path_in_repo, recursive, revision=None):
        assert recursive and path_in_repo
        assert revision is None or revision == "a" * 40
        self.calls.append(("list", repo_id, repo_type, revision))
        for path, data in sorted(self.files.get((repo_id, repo_type), {}).items()):
            if not path.startswith(path_in_repo.rstrip("/") + "/"):
                continue
            oid = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            lfs = None
            if path.endswith(".pt"):
                lfs = {
                    "oid": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                    "pointerSize": 130,
                }
            yield RepoFile(path=path, size=len(data), oid=oid, lfs=lfs)


def configure_memory_upload(monkeypatch, tmp_path, *, insufficient=False, **api_kwargs):
    """Keep quota observations, secrets, retries and every transport call local."""
    api = MemoryHub(**api_kwargs)
    monkeypatch.setattr(crossmodel_artifacts, "HfApi", lambda: api)
    monkeypatch.setenv("EPS_STORY_PERSONA_SOURCE_SHA", "c" * 40)
    monkeypatch.setenv("EPM_HF_OVERFLOW_EVENT_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")

    def headroom(projected_bytes, *, probe_floor_gb):
        assert projected_bytes > 0 and probe_floor_gb == 0
        return crossmodel_artifacts.hub.ProjectedUploadHeadroom(
            "insufficient" if insufficient else "fits",
            projected_bytes / 1e12,
            20.3 if insufficient else 10.0,
            20.0,
            "live-api synthetic unit test",
        )

    monkeypatch.setattr(crossmodel_artifacts.hub, "check_projected_upload_headroom", headroom)
    out = tmp_path / "out"
    (out / "chunks").mkdir(parents=True)
    (out / "failure.json").write_text('{"status": "failed fixture"}\n')
    (out / "chunks/chunk.pt").write_bytes(b"synthetic tensor fixture")
    (out / "chunks/failure.json").write_text('{"nested": "same basename"}\n')
    return api, out


@pytest.mark.parametrize("insufficient", [False, True])
def test_persist_verifies_one_actual_typed_repository(monkeypatch, tmp_path, insufficient):
    """Use the real uploader, nested paths, pointer and immutable Git/LFS hash checks."""
    api, out = configure_memory_upload(monkeypatch, tmp_path, insufficient=insufficient)
    original = pilot_artifacts.inventory(out)
    receipt = crossmodel_artifacts.persist(out, "qwen", final=False, failed=True)
    repo = (
        crossmodel_artifacts.hub.DEFAULT_OVERFLOW_REPO
        if insufficient
        else crossmodel_artifacts.hub.DEFAULT_DATASET_REPO
    )
    repo_type = "model" if insufficient else "dataset"
    assert (receipt["hf_repo"], receipt["hf_repo_type"]) == (repo, repo_type)
    assert receipt["verified_revision"] == "a" * 40
    assert receipt["complete"] is False and receipt["failed_attempt"] is True
    assert receipt["files"] == original == pilot_artifacts.inventory(out)
    assert ("list", repo, repo_type, "a" * 40) in api.calls
    assert set(api.files[repo, repo_type]) == {f"{receipt['prefix']}/{p}" for p in original}
    assert receipt["url"].startswith(
        f"https://huggingface.co/{'' if insufficient else 'datasets/'}{repo}/tree/"
    )
    if insufficient:
        canonical = api.files[crossmodel_artifacts.hub.DEFAULT_DATASET_REPO, "dataset"]
        assert list(canonical) == [f"{receipt['prefix']}/OVERFLOW_POINTER.json"]
        assert json.loads(next(iter(canonical.values())))["overflow_repo"] == repo


def test_equal_size_checkpoint_metadata_is_recommitted(monkeypatch, tmp_path):
    """A newer progress record must not be accepted using the uploader's size-only skip."""
    api, out = configure_memory_upload(monkeypatch, tmp_path)
    (out / "progress.json").write_text('{"done": 1}')
    prefix = "issue2673_test/checkpoint"
    crossmodel_artifacts.upload_snapshot(out, pilot_artifacts.inventory(out), prefix, api)
    (out / "progress.json").write_text('{"done": 2}')
    crossmodel_artifacts.upload_snapshot(out, pilot_artifacts.inventory(out), prefix, api)
    files = api.files[crossmodel_artifacts.hub.DEFAULT_DATASET_REPO, "dataset"]
    assert files[f"{prefix}/progress.json"] == b'{"done": 2}'
    assert pilot_artifacts.inventory(out)["progress.json"]["size"] == len(b'{"done": 1}')


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quota_after": 1}, "split the artifact store"),
        ({"corrupt": True}, "content hash mismatch"),
    ],
)
def test_split_or_corrupted_upload_cannot_write_receipt(monkeypatch, tmp_path, kwargs, message):
    """Reactive mid-store rerouting or same-size bad bytes cannot claim coherent completion."""
    _, out = configure_memory_upload(monkeypatch, tmp_path, **kwargs)
    original = pilot_artifacts.inventory(out)
    with pytest.raises(RuntimeError, match=message):
        crossmodel_artifacts.persist(out, "deepseek", final=False, failed=True)
    assert not list(tmp_path.glob("*_receipts/receipt_*.json"))
    assert pilot_artifacts.inventory(out) == original
