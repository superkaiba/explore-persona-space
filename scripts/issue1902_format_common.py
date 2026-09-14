"""Pinned inputs and verified checkpoints for the Qwen/OLMo format comparison."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()
from explore_persona_space.orchestrate.hub import (  # noqa: E402
    DEFAULT_OVERFLOW_REPO,
    retry_transient,
    stage_hub_file,
)

REPO = "superkaiba1/explore-persona-space-data"
BINARY_SUFFIXES = {".npz", ".png", ".pdf"}
PREFIX = "issue1902_olmo_onpolicy_20260914/production_v1"
REUSE_PREFIX = "issue1902_format_reconciliation_20260912/format_v3_overflow"
REUSE_REV = "300057b6b874814778e453fa82b682ceb15aaaa5"
INPUT_REV = "362ccf2f011c926a1e8a58f3e1f0df56024adcbe"
Q0 = "d3207a181402b42873f5a3120b1d56da7b90f104"
Q3 = "5ae90722bf11330deddfa42cf41f9fec6da8b69f"
Q5 = "9de026f872c19b2ca4fd3e4539de820e08038ee3"
O0 = "3256c8efcef5f10ca525efeb2039636eaec8fad7"
O5 = "f0b2131442326ef274c91bea6da27e05ef844df6"
CHUNK = 256  # Parent #2054 production checkpoint/pilot unit.
MODELS = {
    "olmo_B": ("allenai/OLMo-2-1124-7B", "7df9a82518afdecae4e8c026b27adccc8c1f0032", 18, 4096),
    "olmo_S": ("allenai/OLMo-2-1124-7B-SFT", "1de02c0175118a9de5854aec80a1f970e701e928", 18, 4096),
    "olmo_D": ("allenai/OLMo-2-1124-7B-DPO", "e34ea60adff2e575f4fe7569eaffd1b28509b6fd", 18, 4096),
    "olmo_R": (
        "allenai/OLMo-2-1124-7B-Instruct",
        "470b1fba1ae01581f270116362ee4aa1b97f4c84",
        18,
        4096,
    ),
}
DEFERRED_FORMAT_MODELS = {}


def sha(path):
    """Content hash without retaining large artifacts in memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Atomic strict JSON, outside the drained sentinel namespace except at terminal exit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def fetch(root, path, revision):
    """Resolve public tensor receipts to their exact private revision, then verify bytes."""
    if path.startswith(PREFIX + "/") and not path.endswith(".done.json"):
        receipt = json.loads(fetch(root, path + ".done.json", revision).read_text())
        assert receipt["path"] == path
        expected_location = (
            (DEFAULT_OVERFLOW_REPO, "model")
            if Path(path).suffix in BINARY_SUFFIXES
            else (REPO, "dataset")
        )
        assert (receipt["repo_id"], receipt["repo_type"]) == expected_location
        assert len(receipt["revision"]) == 40 and all(
            c in "0123456789abcdef" for c in receipt["revision"]
        ), "Tensor receipt must pin an immutable revision"
        if "parts" in receipt:
            assert receipt["repo_id"] == REPO and receipt["parts"]
            target = root / "reassembled" / revision / path
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as output:
                for part in receipt["parts"]:
                    source = fetch_location(
                        root, part["path"], receipt["revision"], REPO, "dataset"
                    )
                    assert sha(source) == part["sha256"]
                    output.write(json.loads(source.read_text())["text"])
            assert sha(temporary) == receipt["sha256"], f"Text reconstruction mismatch: {path}"
            os.replace(temporary, target)
        else:
            target = fetch_location(
                root, path, receipt["revision"], receipt["repo_id"], receipt["repo_type"]
            )
        assert target.stat().st_size == receipt["bytes"]
        assert sha(target) == receipt["sha256"], f"Public receipt mismatch: {path}"
        return target
    return fetch_location(root, path, revision, REPO, "dataset")


def fetch_location(root, path, revision, repo_id, repo_type):
    """Stage and hash-check one immutable location, separating caches by repository."""
    from huggingface_hub import HfApi

    target = root / "inputs" / repo_type / repo_id / revision / path
    receipt = target.with_suffix(target.suffix + ".source.json")
    if receipt.exists():
        expected = json.loads(receipt.read_text())
        assert expected["revision"] == revision and expected["path"] == path
        assert sha(target) == expected["sha256"]
        return target
    entries = retry_transient(
        lambda: HfApi().get_paths_info(repo_id, [path], repo_type=repo_type, revision=revision),
        what=f"pinned input metadata {path}",
    )
    assert len(entries) == 1, f"Missing pinned input {path}"
    entry = entries[0]
    stage_hub_file(
        repo_id, path, target, repo_type=repo_type, revision=revision, size_bytes=entry.size
    )
    assert target.stat().st_size == entry.size
    if entry.lfs:
        assert sha(target) == entry.lfs.sha256
    else:
        data = target.read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == entry.blob_id
    write_json(receipt, dict(path=path, revision=revision, sha256=sha(target), bytes=entry.size))
    return target


def read_jsonl(path):
    """Read every nonempty JSON line; malformed rows fail loudly."""
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write_raw(path, rows):
    """Persist an indexed raw bank in <8MB JSON shards, following #2054's layout."""
    parts, current, size = [], [], 2
    for row in rows:
        row_bytes = len(json.dumps(row, ensure_ascii=False).encode()) + 2
        assert row_bytes < 8_000_000, "Single raw row exceeds shard budget"
        if current and size + row_bytes > 8_000_000:
            parts.append(current)
            current, size = [], 2
        current.append(row)
        size += row_bytes
    if current:
        parts.append(current)
    assert parts
    files, manifest = [], []
    for index, records in enumerate(parts):
        part = path.with_name(f"{path.stem}_part{index:03d}.json")
        write_json(part, records)
        assert part.stat().st_size < 8_000_000
        files.append(part)
        manifest.append(dict(name=part.name, rows=len(records), sha256=sha(part)))
    write_json(path, dict(shards=manifest, rows=len(rows)))
    return [path, *files]


def read_raw(path):
    """Load only the indexed shards and check their declared content and row counts."""
    manifest = json.loads(path.read_text())
    rows = []
    for record in manifest["shards"]:
        part = path.parent / record["name"]
        assert sha(part) == record["sha256"]
        data = json.loads(part.read_text())
        assert len(data) == record["rows"]
        rows.extend(data)
    assert len(rows) == manifest["rows"]
    return rows


def upload_many(paths, root, fingerprint):
    """Persist tensors privately and text publicly, with immutable public location receipts.

    Sep14 live public usage exceeds the ceiling; the private upload/read-back probe
    passed. Routing is explicit for this frozen run, never dependent on mutable HEAD.
    """
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    api = HfApi()
    paths = [Path(p) for p in paths]
    assert paths and len(paths) == len(set(paths))

    def commit(pairs, repo_id=REPO, repo_type="dataset"):
        revision = retry_transient(
            lambda: api.create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                operations=[
                    CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src) for src, dst in pairs
                ],
                commit_message=f"1902 format comparison verified packet ({len(pairs)} files)",
            ),
            what="format checkpoint commit",
        ).oid
        entries = retry_transient(
            lambda: api.get_paths_info(
                repo_id, [dst for _, dst in pairs], repo_type=repo_type, revision=revision
            ),
            what="format checkpoint verification",
        )
        by_path = {e.path: e for e in entries}
        for src, dst in pairs:
            entry = by_path[dst]
            assert entry.size == src.stat().st_size, dst
            if entry.lfs:
                assert entry.lfs.sha256 == sha(src), dst
            else:
                data = src.read_bytes()
                assert (
                    entry.blob_id == hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                ), dst
        return revision

    pairs = [(p, f"{PREFIX}/{p.relative_to(root)}") for p in paths]
    tensors = [(src, dst) for src, dst in pairs if src.suffix in BINARY_SUFFIXES]
    texts, parts = [], {}
    for src, dst in pairs:
        if src.suffix in BINARY_SUFFIXES:
            continue
        assert src.suffix in {".json", ".jsonl", ".log", ".txt"}, src
        if src.stat().st_size < 9_500_000:
            texts.append((src, dst))
            continue
        # #2054's line-only JSONL splitter cannot split our single-line JSON
        # manifests. JSON text fragments preserve every character, including CRLF,
        # with a bounded size even for escaped control characters; no LFS/gzip.
        parts[dst] = []
        with src.open(encoding="utf-8", newline="") as stream:
            while chunk := stream.read(1_000_000):
                part = src.with_name(f"{src.name}.part{len(parts[dst]):04d}.json")
                write_json(part, dict(text=chunk))
                assert part.stat().st_size < 9_500_000
                destination = f"{PREFIX}/{part.relative_to(root)}"
                texts.append((part, destination))
                parts[dst].append(dict(path=destination, sha256=sha(part)))
    locations = {}
    for group, repo_id, repo_type in (
        (tensors, DEFAULT_OVERFLOW_REPO, "model"),
        (texts, REPO, "dataset"),
    ):
        if group:
            revision = commit(group, repo_id, repo_type)
            for _, destination in group:
                locations[destination] = dict(
                    revision=revision, repo_id=repo_id, repo_type=repo_type
                )
            if repo_id == REPO:
                for destination, shards in parts.items():
                    locations[destination] = dict(
                        revision=revision, repo_id=repo_id, repo_type=repo_type, parts=shards
                    )
    receipts = []
    for path, destination in pairs:
        done = path.with_suffix(path.suffix + ".done.json")
        pending = done.with_suffix(".pending")
        write_json(
            pending,
            {
                "path": destination,
                **locations[destination],
                "sha256": sha(path),
                "bytes": path.stat().st_size,
                "fingerprint": fingerprint,
            },
        )
        receipts.append((pending, destination + ".done.json", done))
    receipt_revision = commit([(src, dst) for src, dst, _ in receipts])
    for pending, _, done in receipts:
        os.replace(pending, done)
    print(f"[phase=upload] verified files={len(paths)} revision={receipt_revision}", flush=True)
    return receipt_revision


def storage_probe(root):
    """Exercise actual >10MB private upload and fresh public-receipt-driven download."""
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    tensor = root / "storage_probe.npz"
    np.savez(
        tensor, probe=np.random.default_rng(1902).standard_normal((768, 4096), dtype=np.float32)
    )
    assert tensor.stat().st_size > 10_000_000
    pointer = root / "OVERFLOW_POINTER.json"
    write_json(
        pointer,
        dict(
            overflow_repo=DEFAULT_OVERFLOW_REPO,
            repo_type="model",
            path_in_repo=PREFIX,
            tensor_locations="Each .npz.done.json pins its repository, revision, bytes and SHA256",
            reason="Live public storage exceeds ceiling; all tensors route to private overflow",
            synthetic_storage_probe=True,
            planned_tensor_gb=25,
        ),
    )
    revision = upload_many([tensor, pointer], root, "synthetic-storage-probe-v1")
    staged = fetch(root / "readback", f"{PREFIX}/{tensor.name}", revision)
    assert sha(staged) == sha(tensor)
    print(
        f"[phase=storage] private_tensor_and_public_receipt_verified revision={revision}",
        flush=True,
    )
    return revision


def complete(path, fingerprint):
    """A checkpoint is reusable only with matching provenance and content."""
    path = Path(path)
    receipt = path.with_suffix(path.suffix + ".done.json")
    if not receipt.exists():
        return False
    value = json.loads(receipt.read_text())
    assert value["fingerprint"] == fingerprint, f"Stale checkpoint: {path}"
    assert sha(path) == value["sha256"], f"Corrupt checkpoint: {path}"
    return True


def banks(model):
    """Exactly two on-policy formats per OLMo checkpoint; regenerate corrupted R source."""
    assert model in MODELS
    native = "plain" if model.endswith("B") else "chat"
    return [
        dict(name=form, render=form, fresh=(form != native or model.endswith("R")), cap=1024)
        for form in ("plain", "chat")
    ]


def capture_forms(bank):
    """The target representation is captured in its own generation setting."""
    return (bank["render"],)


def owned_offsets(model, n_rows, shard, shards, first_chunk):
    """Shard checkpoint-sized chunks, keeping each production-shaped pilot uniquely owned."""
    assert shards > 0 and 0 <= shard < shards
    offsets = [0] if first_chunk else range(0, n_rows, CHUNK)
    model_index = list(MODELS).index(model)
    return [offset for offset in offsets if (model_index + offset // CHUNK) % shards == shard]


def render_prompt(family, form, query, template_tokenizer):
    """Parent literal plain formats and family-native chat, independent of checkpoint."""
    assert form in ("plain", "chat")
    if family == "qwen":
        if form == "plain":
            return f"User: {query}\n\nAssistant: "
        # #2054 used ChatML without Qwen's optional default system message.
        return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    if form == "plain":
        return f"User: {query}\nAssistant:"
    return template_tokenizer.apply_chat_template(
        [{"role": "user", "content": query}], tokenize=False, add_generation_prompt=True
    )


def fingerprint(root):
    """Bind outputs to all new implementation files and the staged manifest."""
    files = sorted((ROOT / "scripts").glob("issue1902_format_*.py"))
    payload = {p.name: sha(p) for p in files}
    for filename in [
        "issue2054_ctx2ctx_fit.py",
        "issue1902_lasttoken_transfer.py",
        "issue1902_lasttoken_comparison.py",
    ]:
        payload[filename] = sha(ROOT / "scripts" / filename)
    payload["manifest"] = sha(root / "manifest.json")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
