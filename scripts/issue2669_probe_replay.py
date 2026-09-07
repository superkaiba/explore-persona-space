"""Availability and range-indexed capture recovery for exact #1739 fair replay.

Run with CONFIG.json. Modes inventory/index/stage; never performs a fit or a
model call. Tar indexing uses Python's tar reader with a seekable HTTP reader,
so skipped tensor members transfer no body bytes. Only strict HTTP206 is accepted.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import tarfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = "superkaiba1/explore-persona-space-data"
TAR_PATH = "issue1739_ctxmap/capture_store/hallucination_labeling/hallucination_labeling.tar"
PIN = "cd942a5a4ef4348bff26a0282680cec931e915fa"
TAR_SIZE = 69869701120
LAYERS = {"evil": [17, 18, 20], "sycophancy": [19, 20], "hallucination": [18, 20]}


def wanted(name: str, layers: list[int], *, include_prefix: bool = True) -> bool:
    """Select exact common-loader metadata plus the frozen-layer summary slice."""
    base = Path(name).name
    if base.startswith("row_index") and base.endswith(".jsonl"):
        return True
    if base.startswith("_capture") and base.endswith(".json"):
        return True
    match = re.fullmatch(r"(prefix_end|context_end|t1)_L(\d+)(?:_shard\d+)?\.npy", base)
    return bool(match and int(match[2]) in layers and (include_prefix or match[1] != "prefix_end"))


def write_json(path: Path, value: object) -> None:
    """Atomically checkpoint an index or completed inventory."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


class HttpRangeReader(io.RawIOBase):
    """Seekable exact-range reader with validated eight-header speculative prefetch."""

    def __init__(self, url: str, size: int, token: str):
        self.url, self.size, self.token = url, size, token
        self.position = self.bytes_received = self.requests_count = 0
        self.local = threading.local()
        self.count_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.pending = {}
        self.retired = []
        self.resolved_url = url
        self.fetch(0, 1)  # Resolve the signed CDN location once, before parallel lookahead.

    def seekable(self):
        """Expose random access to tarfile."""
        return True

    def tell(self):
        """Return logical archive offset."""
        return self.position

    def seek(self, offset, whence=0):
        """Seek without network transfer."""
        position = (
            offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        )
        if not 0 <= position <= self.size:
            raise ValueError("Out-of-bounds archive seek")
        self.position = position
        return position

    def fetch(self, start, size):
        """Keep signed CDN URLs out of all surfaced transport exceptions."""
        try:
            return self._fetch_range(start, size)
        except requests.RequestException as error:
            raise RuntimeError(f"Range transport {type(error).__name__}; signed URL withheld") from None

    def _fetch_range(self, start, size):
        """Fetch exact range with per-thread connections and bounded status retries."""
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        end = start + size - 1
        expected = f"bytes {start}-{end}/{self.size}"
        headers = {
            "Range": f"bytes={start}-{end}",
            "Authorization": "Bearer " + self.token,
            "Accept-Encoding": "identity",
        }
        for retry in range(6):
            call_headers = dict(headers)
            if self.resolved_url != self.url:
                call_headers.pop("Authorization")
            with self.local.session.get(
                self.resolved_url, headers=call_headers, stream=True, timeout=60
            ) as response:
                if response.status_code in (429, 500, 502, 503, 504) and retry < 5:
                    delay = float(response.headers.get("Retry-After", min(60, 5 * 2**retry)))
                    print(
                        json.dumps(
                            {
                                "phase": "range_retry",
                                "http_status": response.status_code,
                                "delay_seconds": delay,
                            }
                        ),
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                if (
                    response.status_code in (401, 403)
                    and self.resolved_url != self.url
                    and retry < 5
                ):
                    self.resolved_url = self.url
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(f"Range HTTP {response.status_code}; signed URL withheld")
                if response.url:
                    self.resolved_url = response.url
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Range not honored exactly; refusing archive body")
                data = response.raw.read(size + 1)
                if len(data) != size:
                    raise ValueError("Range body size mismatch")
                with self.count_lock:
                    self.bytes_received += size
                    self.requests_count += 1
                return data
        raise RuntimeError("Range retries exhausted")

    def read(self, size=-1):
        """Consume only canonical tar offsets; speculative offsets never advance position."""
        if size < 0 or size > 8 * 1024**2:
            raise ValueError("Unbounded or oversized range read prohibited")
        size = min(size, self.size - self.position)
        if not size:
            return b""
        start = self.position
        match = next(
            (key for key in self.pending if key[0] <= start and start + size <= sum(key)), None
        )
        if match is None:
            data = self.fetch(start, size)
        else:
            payload = self.pending[match].result()
            data = payload[start - match[0] : start - match[0] + size]
        self.position += size
        if size == 512:
            try:
                header = tarfile.TarInfo.frombuf(data, "utf8", "strict")
            except tarfile.HeaderError:
                # Tarfile itself validates/fails canonical headers. No speculation on invalid bytes.
                return data
            stride = 512 + ((header.size + 511) // 512) * 512
            for key in list(self.pending):
                if sum(key) <= self.position or len(self.pending) > 32:
                    future = self.pending.pop(key)
                    if not future.cancel():
                        if future.done():
                            future.result()
                        else:
                            self.retired.append(future)
            for step in range(1, 9):
                offset = start + step * stride
                key = (offset - 1, 513)  # Also covers tarfile.next's one-byte boundary probe.
                if offset > 0 and offset + 512 <= self.size and key not in self.pending:
                    self.pending[key] = self.pool.submit(self.fetch, *key)
        return data

    def finish_prefetch(self):
        """Drain all speculative requests and surface failures before completion."""
        self.pool.shutdown(wait=True)
        for future in [*self.pending.values(), *self.retired]:
            if not future.cancelled():
                future.result()
        self.pending.clear()


def index_tar(reader: HttpRangeReader, path: Path, revision: str) -> dict:
    """Index headers only, checkpoint every member, and resume at next header."""
    state = {"revision": revision, "tar_size": reader.size, "complete": False, "members": []}
    if path.exists():
        state = json.loads(path.read_text())
        if state["revision"] != revision or state["tar_size"] != reader.size:
            raise ValueError("Archive identity changed")
        if state["complete"]:
            return state
        if state["members"]:
            last = state["members"][-1]
            reader.seek(last["offset"] + ((last["size"] + 511) // 512) * 512)
    with tarfile.open(fileobj=reader, mode="r:") as archive:
        for member in archive:
            if member.isfile():
                state["members"].append(
                    {"name": member.name, "offset": member.offset_data, "size": member.size}
                )
                n = len(state["members"])
                if n % 50 == 0:
                    write_json(path, state)
                if n % 100 == 0:
                    print(
                        json.dumps(
                            {"phase": "index", "members": n, "network_bytes": reader.bytes_received}
                        ),
                        flush=True,
                    )
            elif not member.isdir():
                raise ValueError(f"Unsupported tar member type {member.type!r}")
    if hasattr(reader, "finish_prefetch"):
        reader.finish_prefetch()
    state["complete"] = True
    state["network_bytes_this_run"] = reader.bytes_received
    state["requests_this_run"] = reader.requests_count
    write_json(path, state)
    return state


def stage_members(reader: HttpRangeReader, index: dict, out: Path, layers: list[int]) -> dict:
    """Fetch selected indexed payloads, preserving a SHA256 receipt per file."""
    if not index["complete"]:
        raise ValueError("Require completed archive index")
    out.mkdir(parents=True, exist_ok=True)
    selected = [m for m in index["members"] if wanted(m["name"], layers, include_prefix=False)]
    basenames = [Path(m["name"]).name for m in selected]
    if len(set(basenames)) != len(basenames):
        raise ValueError("Duplicate selected tar basenames")
    receipt_path = out / "range_receipts.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    pending = []
    for member in selected:
        base = Path(member["name"]).name
        destination = out / base
        if base in receipt:
            expected = {**member, "revision": index["revision"]}
            if any(receipt[base].get(key) != value for key, value in expected.items()):
                raise ValueError("Staged receipt archive identity mismatch")
            if (
                not destination.exists()
                or hashlib.sha256(destination.read_bytes()).hexdigest() != receipt[base]["sha256"]
            ):
                raise ValueError(f"Corrupt staged member {base}")
        else:
            pending.append(member)
    receipt_lock = threading.Lock()

    def fetch_member(member):
        base = Path(member["name"]).name
        destination = out / base
        position, left = member["offset"], member["size"]
        hasher = hashlib.sha256()
        temp = destination.with_suffix(destination.suffix + ".tmp")
        with temp.open("wb") as handle:
            while left:
                data = reader.fetch(position, min(left, 8 * 1024**2))
                handle.write(data)
                hasher.update(data)
                position += len(data)
                left -= len(data)
        temp.replace(destination)
        with receipt_lock:
            receipt[base] = {**member, "sha256": hasher.hexdigest(), "revision": index["revision"]}
            write_json(receipt_path, receipt)
            print(
                json.dumps(
                    {
                        "phase": "stage",
                        "member": base,
                        "bytes": member["size"],
                        "completed": len(receipt),
                    }
                ),
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch_member, pending))

    return {"files": len(receipt), "bytes": sum(r["size"] for r in receipt.values())}


def stage_auxiliary(out: Path) -> dict:
    """Stage only pinned four-layer generic/WildChat files via canonical Hub helper."""
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.experiments.issue_1739 import store_io

    api = HfApi()
    prefix = "issue1739_ctxmap/wildchat_rung/capture_store/wildchat"
    files = [
        f
        for f in api.list_repo_tree(REPO, path_in_repo=prefix, repo_type="dataset", revision=PIN)
        if hasattr(f, "size") and wanted(f.path, [17, 18, 19, 20])
    ]
    expected_bytes = sum(f.size for f in files)
    if expected_bytes != 863718190 or len(files) != 281:
        raise ValueError("Pinned WildChat inventory differs from reviewed allowance")
    wc = out / "wildchat"
    wc.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        path = hub.stage_hub_file(
            REPO, item.path, wc / Path(item.path).name, revision=PIN, size_bytes=item.size
        )
        if path.stat().st_size != item.size:
            raise ValueError("Staged source length mismatch")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = {"source": item.path, "revision": PIN, "bytes": item.size, "sha256": sha}
        write_json(wc / (path.name + ".receipt.json"), receipt)
        return receipt

    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(fetch, files))
    print(json.dumps({"phase": "wildchat_staged", "bytes": expected_bytes}), flush=True)
    store_io.stage_u_store(out / "u_store", ("prefix_end", "context_end", "t1"), (17, 18, 19, 20))
    result = {
        "wildchat": {"files": receipts, "bytes": expected_bytes},
        "generic_u": {
            "path": str(out / "u_store"),
            "revision": "e5901706",
            "array_bytes": 1822938624,
        },
    }
    write_json(out / "auxiliary_manifest.json", result)
    return result


def main(config_path: Path) -> dict:
    """Perform only the explicitly configured availability/index/staging phase."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from huggingface_hub import get_token, hf_hub_url

    config = json.loads(config_path.read_text())
    behavior = config.get("behavior", "hallucination")
    tar_path = f"issue1739_ctxmap/capture_store/{behavior}_labeling/{behavior}_labeling.tar"
    tar_size = {"hallucination": TAR_SIZE, "sycophancy": 52140359680}[behavior]
    out = Path(config["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "revision": PIN,
        "tar_path": tar_path,
        "tar_size": tar_size,
        "layers": LAYERS,
        "local_stores": {},
        "fit_status": "not_run; requires separate approved timing config",
    }
    for name, path in config["local_stores"].items():
        p = Path(path)
        files = list(p.glob("*.npy")) if p.exists() else []
        result["local_stores"][name] = {
            "path": path,
            "exists": p.exists(),
            "array_files": len(files),
            "array_bytes": sum(f.stat().st_size for f in files),
        }
    if config["mode"] == "auxiliary":
        result["auxiliary"] = stage_auxiliary(out)
    elif config["mode"] in ("index", "stage"):
        token = get_token()
        if not token:
            raise ValueError("Missing HF authentication")
        reader = HttpRangeReader(
            hf_hub_url(REPO, tar_path, repo_type="dataset", revision=PIN), tar_size, token
        )
        index = index_tar(reader, out / f"{behavior}_tar_index.json", PIN)
        selected = [
            r for r in index["members"] if wanted(r["name"], LAYERS[behavior], include_prefix=False)
        ]
        result["selected_tar_files"] = len(selected)
        result["selected_tar_bytes"] = sum(r["size"] for r in selected)
        if config["mode"] == "stage":
            result["staged"] = stage_members(
                reader, index, out / f"{behavior}_labeling", LAYERS[behavior]
            )
    elif config["mode"] != "inventory":
        raise ValueError("Unknown mode")
    write_json(out / f"{behavior}_{config['mode']}_availability.json", result)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(main(Path(sys.argv[1])), indent=2))
