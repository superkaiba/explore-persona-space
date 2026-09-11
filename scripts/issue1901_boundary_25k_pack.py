#!/usr/bin/env python3
"""Consolidate prepared article tensors without changing rows or input tokens.

Runs on the preparation host after preparation completes. The original raw
checkpoint documents remain independently uploaded under prepare/. Compact
manifests avoid thousands of per-file Hub requests on GPU staging.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import issue1901_boundary_25k as B  # noqa: E402
import torch  # noqa: E402


def pack(source, target):
    meta = json.loads((source / "meta.json").read_text())
    target.mkdir(parents=True, exist_ok=True)
    names, ids, tokens, seen = [], [], [], set()

    def flush():
        name = f"articles_shard{len(names):04d}.pt"
        torch.save({"window_ids": list(ids), "input_ids": list(tokens)}, target / name)
        # Exact serialization readback; token arrays are the forward inputs.
        check = torch.load(target / name, map_location="cpu", weights_only=True)
        assert check["window_ids"] == ids
        assert all(torch.equal(a, b) for a, b in zip(check["input_ids"], tokens, strict=True))
        names.append(name)
        ids.clear()
        tokens.clear()

    for old in meta["article_shards"]:
        obj = torch.load(source / old, map_location="cpu", weights_only=True)
        for aid, array in zip(obj["window_ids"], obj["input_ids"], strict=True):
            assert aid not in seen, aid
            seen.add(aid)
            ids.append(aid)
            tokens.append(array)
            if len(ids) == 2000:
                flush()
    if ids:
        flush()
    assert len(seen) == meta["n_articles"]
    for name in meta["manifest_shards"]:
        shutil.copyfile(source / name, target / name)
        assert B.sha_file(source / name) == B.sha_file(target / name)
    meta["compacted_from"] = {
        "meta_sha256": B.sha_file(source / "meta.json"),
        "original_article_shards": len(meta["article_shards"]),
    }
    meta["article_shards"] = names
    B.write_json(target / "meta.json", meta)
    print(
        json.dumps(
            {"articles": len(seen), "tensor_shards": len(names), "rows": meta["n_manifest_rows"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    args = ap.parse_args()
    pack(args.source, args.target)
