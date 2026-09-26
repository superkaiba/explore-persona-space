"""Credit already allocated, readable pinned Kimi shards in startup disk sizing.

This is a storage accounting check, not a replacement for Hub integrity checks
or the capture's numerical smoke. Preserve the cold-start 800 GB total budget.
"""

import json
from pathlib import Path
import time

MODEL = "moonshotai/Kimi-K2.6"
REVISION = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
WEIGHT_BYTES = 595177988208


def allocated_credit(cache: Path, siblings) -> dict:
    snapshot = cache / "models--moonshotai--Kimi-K2.6" / "snapshots" / REVISION
    if not snapshot.exists():
        return {"allocated_credit_bytes": 0, "readable_shards": 0}
    shards = [s for s in siblings if s.rfilename.endswith(".safetensors")]
    if len(shards) != 64 or sum(s.size for s in shards) != WEIGHT_BYTES:
        raise RuntimeError("pinned Kimi weight manifest changed")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != {s.rfilename for s in shards}:
        raise RuntimeError("cached index does not match pinned shard names")
    total = 0
    seen = set()
    for shard in shards:
        path = snapshot / shard.rfilename
        target = path.resolve(strict=True)
        blobs = (cache / "models--moonshotai--Kimi-K2.6" / "blobs").resolve()
        if target.parent != blobs or target.name != shard.lfs.sha256:
            raise RuntimeError("cached shard is outside the expected immutable blob store")
        stat = target.stat()
        if stat.st_size != shard.size or (stat.st_dev, stat.st_ino) in seen:
            raise RuntimeError("cached shard size/identity mismatch")
        seen.add((stat.st_dev, stat.st_ino))
        with target.open("rb") as stream:
            if len(stream.read(65536)) != 65536:
                raise RuntimeError("unreadable shard start")
            stream.seek(-65536, 2)
            if len(stream.read(65536)) != 65536:
                raise RuntimeError("unreadable shard end")
        total += min(stat.st_blocks * 512, stat.st_size)
    return {"allocated_credit_bytes": total, "readable_shards": len(shards)}


def main():
    import os

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from huggingface_hub import HfApi

    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError("model revision mismatch")
    result = allocated_credit(Path(os.environ["HF_HUB_CACHE"]), info.siblings)
    result.update(
        model=MODEL,
        revision=REVISION,
        cold_required_bytes=800 * 10**9,
        checked_at=time.time(),
        validation_scope="publisher LFS names/sizes and local read probes; not full content SHA",
    )
    result["remaining_required_bytes"] = 800 * 10**9 - result["allocated_credit_bytes"]
    Path(os.environ["EPS_STORY_PERSONA_OUT"], "cache_headroom.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
