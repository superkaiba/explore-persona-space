"""CPU-only upload fixtures. All Hub API and upload calls are mocked."""

import fnmatch
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))
from huggingface_hub.hf_api import RepoFile

import scripts.context_risk_corrected_finish as module
from explore_persona_space.orchestrate import hub

root = Path(tempfile.mkdtemp(prefix="context-risk-finish-upload-critic-"))
module.ROOT = root / "receipts"
module.ROOT.mkdir()
stage = root / "stage"
stage.mkdir()
(stage / "a.txt").write_text("alpha\n")
(stage / "b.bin").write_bytes(b"betabeta")
source_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
remote = {}
calls = {"upload": 0, "revision": 0, "tree": 0}
tests = []


class Api:
    def repo_info(self, *args, **kwargs):
        calls["revision"] += 1
        return SimpleNamespace(sha="immutable-test-revision")

    def list_repo_tree(self, *args, **kwargs):
        calls["tree"] += 1
        assert kwargs["revision"] == "immutable-test-revision"
        return iter(remote.values())


def fake_upload(local, repo, kind, prefix, **kwargs):
    calls["upload"] += 1
    remote.clear()
    for path in local.rglob("*"):
        if not path.is_file():
            continue
        relative = str(path.relative_to(local))
        if any(fnmatch.fnmatch(relative, pattern) for pattern in kwargs.get("ignore_patterns", [])):
            continue
        data = path.read_bytes()
        name = f"{prefix}/{relative}"
        fields = {
            "path": name,
            "size": len(data),
            "oid": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
        }
        if path.suffix == ".bin":
            fields["lfs"] = {
                "size": len(data),
                "oid": hashlib.sha256(data).hexdigest(),
                "pointerSize": 1,
            }
        remote[name] = RepoFile(**fields)
    return f"{repo}/{prefix}"


def rejects(fn, expected):
    try:
        fn()
    except ValueError as error:
        assert expected in str(error), (expected, str(error))
    else:
        raise AssertionError("Expected rejection: " + expected)


with (
    patch("huggingface_hub.HfApi", Api),
    patch.object(hub, "_upload", fake_upload),
    patch.object(hub, "retry_transient", lambda fn, **kwargs: fn()),
):
    first = module.upload(stage, "raw")
    assert (module.ROOT / "raw_upload_receipt.json").is_file()
    assert first["verified_files"] == 2
    tests.append("git_and_lfs_hashes_phase_receipt_path")
    module.upload(stage, "raw")
    assert calls["upload"] == 1 and calls["tree"] == 2 and calls["revision"] == 1
    tests.append("receipt_replay_reverifies_pinned_remote_without_reupload")
    key = next(k for k in remote if k.endswith("a.txt"))
    old = remote[key].blob_id
    remote[key].blob_id = "bad"
    rejects(lambda: module.upload(stage, "raw"), "Remote Git blob content mismatch")
    remote[key].blob_id = old
    tests.append("remote_content_drift_rejected")
    (stage / "a.txt").write_text("changed")
    rejects(lambda: module.upload(stage, "raw"), "Staged content differs")
    (stage / "a.txt").write_text("alpha\n")
    tests.append("local_content_drift_rejected")
    remote["unexpected"] = RepoFile(path="unexpected", size=0, oid="")
    rejects(lambda: module.upload(stage, "raw"), "Remote archive filenames")
    del remote["unexpected"]
    tests.append("remote_extra_file_rejected")
    shards = root / "shards"
    shards.mkdir()
    raw = shards / "rollouts.jsonl"
    raw.write_text((json.dumps({"text": "a" * 1000}) + "\n") * 10000)
    assert raw.stat().st_size > 9500000
    before = hashlib.sha256(raw.read_bytes()).hexdigest()
    module.upload(shards, "analysis")
    manifest = json.loads((shards / "rollouts.manifest.json").read_text())
    joined = b"".join((shards / part).read_bytes() for part in manifest["parts"])
    assert hashlib.sha256(joined).hexdigest() == before
    assert raw.is_file() and not any(name.endswith("/rollouts.jsonl") for name in remote)
    assert all(entry.size < 9500000 for entry in remote.values())
    tests.append("large_jsonl_lossless_shards_original_retained_excluded")
    prior_uploads = calls["upload"]
    module.upload(shards, "analysis")
    assert calls["upload"] == prior_uploads
    tests.append("sharded_upload_replay_is_idempotent")
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == source_hash, (
    "Source changed during review"
)
report = {
    "passed": True,
    "source_sha256": source_hash,
    "tests": tests,
    "counts": calls,
    "scope": "Mocked Hub API/uploads; real sharding and hash checks; no network or model calls.",
}
(root / "review.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"report": str(root / "review.json"), **report}, indent=2))
