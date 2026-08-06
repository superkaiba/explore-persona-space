"""Stream a #1739 capture-store tar from HF and keep only a (kind x layer) slice.

The #1739 behavior eval stores live on the HF data repo as MONOLITHIC tars
(``issue1739_ctxmap/capture_store/<behavior>_labeling/<behavior>_labeling.tar``,
32-70 GB each) holding ``{prefix_end,context_end,t1}_L{00..27}_shard*.npy`` plus
``row_index*.jsonl`` sidecars. The #779 963k-context map covers layers
{14,19,26} only, so a full download is ~9x more bytes than the analysis needs and
would not fit the disk budget alongside the map weights.

There is no member-selective HF download for a tar, and header-walking via HTTP
Range costs one request per member (~8.8k members/tar — over the org
2500-req/5-min quota). So this streams the tar ONCE through
``tarfile.open(mode="r|")`` (the non-seekable stream mode) and writes only the
wanted members: bytes transferred are the full tar, bytes WRITTEN are the ~11%
slice. Resumable: a member already on disk at the expected size is skipped
without re-writing (the stream still passes over it).

THROUGHPUT: a single plain-``requests`` connection to this xet-backed repo
measured 3.8 MB/s (32 GB => ~2.3 h), because the optimized parallel-chunk path
is the hf-xet client, which has no tar-streaming entrypoint. ``ParallelRangeReader``
recovers the parallelism WITHOUT the tar-sized disk cost: it fetches fixed
byte WINDOWS with N concurrent Range requests and serves them to ``tarfile`` in
strict order, so the consumer sees one sequential stream. Measured ~30 MB/s at
8-16 workers (the apparent per-token ceiling), i.e. ~18 min for a 32 GB tar,
with in-flight memory bounded to ``workers x window``.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import tarfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

logger = logging.getLogger("map963k_slice")


class ParallelRangeReader(io.RawIOBase):
    """Sequential read() over an HTTP object fetched via N concurrent Range GETs.

    Windows are submitted ahead of the read cursor and delivered in STRICT
    ORDER, so a sequential consumer (``tarfile`` in ``r|`` mode) sees an
    ordinary stream while the network runs ``workers``-way parallel. In-flight
    memory is bounded by ``workers * window``. Each window retries
    independently; a window that never succeeds raises into the reader.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str,
        total: int,
        window: int = 32 << 20,
        workers: int = 12,
        attempts: int = 5,
    ) -> None:
        self._url = url
        self._headers = {"Authorization": f"Bearer {token}"}
        self._total = total
        self._window = window
        self._attempts = attempts
        self._n_windows = (total + window - 1) // window
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._inflight: deque = deque()
        self._next_submit = 0
        self._buf = b""
        self._buf_off = 0
        self._pos = 0
        self._depth = workers
        self._lock = threading.Lock()
        self.bytes_fetched = 0
        for _ in range(self._depth):
            self._submit_next()

    def _submit_next(self) -> None:
        if self._next_submit >= self._n_windows:
            return
        idx = self._next_submit
        self._next_submit += 1
        self._inflight.append(self._pool.submit(self._fetch, idx))

    def _fetch(self, idx: int) -> bytes:
        start = idx * self._window
        end = min(start + self._window, self._total) - 1
        headers = dict(self._headers, Range=f"bytes={start}-{end}")
        last: Exception | None = None
        for attempt in range(self._attempts):
            try:
                resp = requests.get(self._url, headers=headers, timeout=(30, 300))
                resp.raise_for_status()
                data = resp.content
                if len(data) != end - start + 1:
                    raise OSError(
                        f"window {idx}: got {len(data)} bytes, expected {end - start + 1}"
                    )
                with self._lock:
                    self.bytes_fetched += len(data)
                return data
            except Exception as exc:  # transient network/5xx — bounded retry
                last = exc
                time.sleep(min(2**attempt, 30))
        raise OSError(f"window {idx} failed after {self._attempts} attempts: {last}")

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:  # noqa: D102 - RawIOBase contract
        chunk = self.read(len(b))
        if not chunk:
            return 0
        b[: len(chunk)] = chunk
        return len(chunk)

    def read(self, size: int = -1) -> bytes:  # noqa: D102 - RawIOBase contract
        if size is None or size < 0:
            parts = []
            while True:
                c = self.read(1 << 20)
                if not c:
                    break
                parts.append(c)
            return b"".join(parts)
        out = []
        need = size
        while need > 0:
            if self._buf_off >= len(self._buf):
                if not self._inflight:
                    break
                self._buf = self._inflight.popleft().result()
                self._buf_off = 0
                self._submit_next()
                if not self._buf:
                    continue
            take = min(need, len(self._buf) - self._buf_off)
            out.append(self._buf[self._buf_off : self._buf_off + take])
            self._buf_off += take
            self._pos += take
            need -= take
        return b"".join(out)

    def close(self) -> None:  # noqa: D102 - RawIOBase contract
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().close()


def head_size(url: str, token: str) -> int:
    """Object size via a 1-byte Range GET (Content-Range total; HEAD may redirect)."""
    h = {"Authorization": f"Bearer {token}", "Range": "bytes=0-0"}
    r = requests.get(url, headers=h, timeout=(30, 120))
    r.raise_for_status()
    cr = r.headers.get("Content-Range", "")
    if "/" not in cr:
        raise OSError(f"no Content-Range total for {url}: {cr!r}")
    return int(cr.rsplit("/", 1)[1])


REPO = "superkaiba1/explore-persona-space-data"
# Per-behavior tar revisions are resolved at run time (--revision, default main).
WANTED_KINDS = ("prefix_end", "context_end", "t1")


def wanted_re(kinds: tuple[str, ...], layers: tuple[int, ...]) -> re.Pattern:
    """Regex matching the basenames this slice keeps (npy shards + row_index)."""
    ks = "|".join(re.escape(k) for k in kinds)
    ls = "|".join(f"{ly:02d}" for ly in layers)
    return re.compile(rf"^(?:(?:{ks})_L(?:{ls})(?:_shard\d+)?\.npy|row_index.*\.jsonl)$")


def tar_url(behavior: str, revision: str) -> str:
    stem = f"{behavior}_labeling"
    return (
        f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/"
        f"issue1739_ctxmap/capture_store/{stem}/{stem}.tar"
    )


def _replace_with_retry(tmp, out, size: int, attempts: int = 6) -> None:
    """``os.replace`` hardened for runpodfs/MooseFS FUSE lag (2026-08-05).

    Three single-writer legs crashed ENOENT at this rename on RunPod volumes
    (the tmp file written+closed moments earlier was transiently invisible;
    never observed on GCE local disk across ~144 GB in the wcrung leg). The
    write path now fsyncs before close; this retries the rename with backoff,
    and accepts an ``out`` that already exists at the right size (the rename
    may have completed server-side despite the client error). Persistent
    failure still raises — fail loud, never a silent skip.
    """
    delay = 0.5
    last: Exception | None = None
    for _ in range(attempts):
        try:
            os.replace(tmp, out)
            return
        except FileNotFoundError as exc:
            last = exc
            try:
                if out.is_file() and out.stat().st_size == size:
                    return  # rename landed despite the error
            except OSError:
                pass
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    raise last  # type: ignore[misc]


def stream_slice(
    behavior: str,
    dest: Path,
    *,
    revision: str,
    kinds: tuple[str, ...],
    layers: tuple[int, ...],
    token: str,
    workers: int = 12,
    window: int = 32 << 20,
) -> dict:
    """Stream one behavior's tar; write only members matching the slice regex."""
    pat = wanted_re(kinds, layers)
    dest.mkdir(parents=True, exist_ok=True)
    url = tar_url(behavior, revision)
    t0 = time.time()
    kept = skipped = reused = 0
    kept_bytes = 0
    total = head_size(url, token)
    logger.info(
        "[%s] streaming %.1f GB via %d-way parallel ranges (window %d MiB)",
        behavior,
        total / 1e9,
        workers,
        window >> 20,
    )
    reader = ParallelRangeReader(url, token=token, total=total, window=window, workers=workers)
    with reader:
        buffered = io.BufferedReader(reader, buffer_size=8 << 20)
        # mode="r|" = sequential/streaming tar read (no seeking on the fileobj).
        with tarfile.open(fileobj=buffered, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name.rsplit("/", 1)[-1]
                if not pat.match(name):
                    skipped += 1
                    continue
                out = dest / name
                if out.is_file() and out.stat().st_size == member.size:
                    reused += 1
                    continue
                src = tar.extractfile(member)
                if src is None:
                    continue
                tmp = out.with_suffix(out.suffix + ".tmp")
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = src.read(4 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                    fh.flush()
                    os.fsync(fh.fileno())
                _replace_with_retry(tmp, out, member.size)
                kept += 1
                kept_bytes += member.size
                if kept % 20 == 0:
                    el = max(time.time() - t0, 1e-6)
                    got = reader.bytes_fetched
                    logger.info(
                        "[%s] kept %d (%.2f GB written) skipped %d reused %d | "
                        "fetched %.1f/%.1f GB (%.1f MB/s) elapsed=%.0fs eta=%.0fs",
                        behavior,
                        kept,
                        kept_bytes / 1e9,
                        skipped,
                        reused,
                        got / 1e9,
                        total / 1e9,
                        got / 1e6 / el,
                        el,
                        (total - got) / max(got / el, 1.0),
                    )
    el = time.time() - t0
    fetched = reader.bytes_fetched
    manifest = {
        "behavior": behavior,
        "repo": REPO,
        "revision": revision,
        "url": url,
        "kinds": list(kinds),
        "layers": list(layers),
        "n_kept": kept,
        "n_reused": reused,
        "n_skipped": skipped,
        "kept_bytes": kept_bytes,
        "tar_bytes": total,
        "bytes_fetched": fetched,
        "elapsed_s": round(el, 1),
        "mb_per_s": round(fetched / 1e6 / max(el, 1e-6), 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (dest / "slice_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "[%s] DONE kept=%d reused=%d skipped=%d written=%.2f GB fetched=%.1f GB in %.0fs (%.1f MB/s)",
        behavior,
        kept,
        reused,
        skipped,
        kept_bytes / 1e9,
        fetched / 1e9,
        el,
        fetched / 1e6 / max(el, 1e-6),
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behavior", required=True, choices=["evil", "hallucination", "sycophancy"])
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--layers", default="14,19,26")
    ap.add_argument("--kinds", default=",".join(WANTED_KINDS))
    ap.add_argument("--workers", type=int, default=12, help="concurrent Range fetchers")
    ap.add_argument("--window-mib", type=int, default=32, help="bytes per Range window")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN missing from the environment")
    layers = tuple(int(x) for x in args.layers.split(",") if x.strip())
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    stream_slice(
        args.behavior,
        args.dest,
        revision=args.revision,
        kinds=kinds,
        layers=layers,
        token=token,
        workers=args.workers,
        window=args.window_mib << 20,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
