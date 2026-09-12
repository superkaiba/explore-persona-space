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
from explore_persona_space.orchestrate.hub import retry_transient, stage_hub_file  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1902_format_reconciliation_20260912/format_v1"
INPUT_REV = "362ccf2f011c926a1e8a58f3e1f0df56024adcbe"
Q0 = "d3207a181402b42873f5a3120b1d56da7b90f104"
Q3 = "5ae90722bf11330deddfa42cf41f9fec6da8b69f"
Q5 = "9de026f872c19b2ca4fd3e4539de820e08038ee3"
O0 = "3256c8efcef5f10ca525efeb2039636eaec8fad7"
O5 = "f0b2131442326ef274c91bea6da27e05ef844df6"
CHUNK = 256  # Parent #2054 production checkpoint/pilot unit.
MODELS = {
    "qwen_B": ("Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796", 19, 3584),
    "qwen_S": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", 19, 3584),
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
    """Stage an immutable Hub artifact through the shared verified staging helper."""
    from huggingface_hub import HfApi

    target = root / "inputs" / revision / path
    receipt = target.with_suffix(target.suffix + ".source.json")
    if receipt.exists():
        expected = json.loads(receipt.read_text())
        assert expected["revision"] == revision and expected["path"] == path
        assert sha(target) == expected["sha256"]
        return target
    entries = retry_transient(
        lambda: HfApi().get_paths_info(REPO, [path], repo_type="dataset", revision=revision),
        what=f"pinned input metadata {path}",
    )
    assert len(entries) == 1, f"Missing pinned input {path}"
    entry = entries[0]
    stage_hub_file(REPO, path, target, revision=revision, size_bytes=entry.size)
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
    """Batch upload, verify immutable hashes, then publish resumable receipts."""
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    api = HfApi()
    paths = [Path(p) for p in paths]
    assert paths and len(paths) == len(set(paths))

    def commit(pairs):
        revision = retry_transient(
            lambda: api.create_commit(
                repo_id=REPO,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src) for src, dst in pairs
                ],
                commit_message=f"1902 format comparison verified packet ({len(pairs)} files)",
            ),
            what="format checkpoint commit",
        ).oid
        entries = retry_transient(
            lambda: api.get_paths_info(
                REPO, [dst for _, dst in pairs], repo_type="dataset", revision=revision
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
    revision = commit(pairs)
    receipts = []
    for path, destination in pairs:
        done = path.with_suffix(path.suffix + ".done.json")
        pending = done.with_suffix(".pending")
        write_json(
            pending,
            {
                "path": destination,
                "revision": revision,
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
    """Original banks plus the missing, cap-matched generation controls."""
    if model.startswith("qwen"):
        result = [
            dict(name="plain", render="plain", fresh=False, cap=4096),
            dict(
                name="chat", render="chat", fresh=False, cap=4096 if model.endswith("B") else 2048
            ),
        ]
        if model.endswith("S"):
            result.append(dict(name="chat4096", render="chat", fresh=True, cap=4096))
        return result
    native = "plain" if model.endswith("B") else "chat"
    result = [dict(name=native, render=native, fresh=False, cap=1024)]
    if model[-1] in "BS":
        other = "chat" if native == "plain" else "plain"
        result.append(dict(name=other, render=other, fresh=True, cap=1024))
    return result


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
