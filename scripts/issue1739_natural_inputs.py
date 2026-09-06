"""Pinned, reduced-layer staging and complete text exclusions for #1739.

Never stages the crossed generic U store.  Text exclusions are conservative:
every context in every reused DV, including labeled training contexts, must
resolve to its actual generation prompt before an export can be published.
No models, generation, judges, or task-workflow writes occur here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "superkaiba1/explore-persona-space-data"
REVISION = "7a47ff5ce42f16308bebaba29c1286a4e9bc8008"
PREFIX = "issue1739_ctxmap"
LAYERS = (17, 18, 19, 20)
KINDS = ("prefix_end", "context_end", "t1")
BEHAVIORS = ("evil", "sycophancy", "hallucination")


def _revision(args) -> str:
    """Require one immutable artifact revision, never an implicit main tip."""
    pin = getattr(args, "revision", REVISION)
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ValueError("natural input staging requires an immutable 40-hex revision")
    return pin


def _hash(path: Path) -> str:
    """Stream a local SHA256 for manifest verification."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path):
    """Read a JSON artifact, failing on missing or malformed inputs."""
    with path.open() as stream:
        return json.load(stream)


def _lines(path: Path):
    """Iterate physical JSONL records without Unicode splitlines corruption."""
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _atomic_json(path: Path, payload) -> None:
    """Publish bookkeeping only after its complete payload is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _entries(prefix: str, pin: str, token: str, *, recursive: bool = True):
    """One raising, scoped, retried listing at the consumed revision."""
    from explore_persona_space.orchestrate import hub
    from huggingface_hub import HfApi

    api = HfApi(token=token or None)
    entries = hub.retry_transient(
        lambda: list(
            api.list_repo_tree(
                REPO, path_in_repo=prefix, repo_type="dataset", revision=pin, recursive=recursive
            )
        ),
        what=f"natural-input listing {prefix}@{pin}",
    )
    files = [entry for entry in entries if hasattr(entry, "size")]
    if not files:
        raise FileNotFoundError(f"no files at {REPO}/{prefix}@{pin}")
    return files


def _stage_file(path: str, dest: Path, pin: str, token: str) -> Path:
    """Pin each cache namespace; canonical helper performs atomic retried fetch."""
    from explore_persona_space.orchestrate import hub

    started = time.monotonic()
    print(f"[natural-file] staging {path}", flush=True)
    receipt = dest.with_name(f".{dest.name}.natural_source.json")
    expected = {"repo": REPO, "path": path, "revision": pin}
    if receipt.exists():
        if _json(receipt) != expected:
            raise ValueError(f"staged file source mismatch: {receipt}")
    elif dest.exists():
        raise ValueError(f"unreceipted existing file; use a fresh natural root: {dest}")
    else:
        _atomic_json(receipt, expected)
    result = hub.stage_hub_file(
        REPO, path, dest, repo_type="dataset", revision=pin, token=token or None
    )
    print(
        f"[natural-file] ready {path} bytes={result.stat().st_size} "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    return result


def _stage_selected(prefix, dest, pin, token, predicate, workers=4) -> list[Path]:
    """Stage a selected flat store with an explicit revision/selection receipt.

    Existing unreceipted stores are rejected, not silently adopted. Use a fresh
    natural-run store root. Per-file atomic fetches can resume an interrupted
    stage carrying the matching entry-time receipt.
    """
    dest = Path(dest)
    files = [e for e in _entries(prefix, pin, token) if predicate(Path(e.path).name)]
    if not files:
        raise FileNotFoundError(f"empty selection under {prefix}")
    names = [Path(e.path).relative_to(prefix).as_posix() for e in files]
    if any("/" in name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"expected a flat non-colliding source under {prefix}")
    receipt = {"repo": REPO, "revision": pin, "prefix": prefix, "files": sorted(names)}
    marker = dest / ".natural_stage_source.json"
    if marker.exists():
        if _json(marker) != receipt:
            raise ValueError(f"staging regime mismatch: {marker}")
    elif dest.exists() and any(dest.iterdir()):
        raise ValueError(f"unreceipted existing store; use a fresh natural root: {dest}")
    else:
        _atomic_json(marker, receipt)
    needed = [(e.path, e.size) for e in files if not (dest / Path(e.path).name).exists()]
    from explore_persona_space.orchestrate import hub

    if needed:
        hub._assert_stage_headroom(dest, needed, what=f"natural-stage {prefix}")
    print(
        f"[natural-stage] {prefix} files={len(files)} bytes={sum(e.size for e in files)}",
        flush=True,
    )
    started = time.monotonic()

    def fetch(entry):
        target = dest / Path(entry.path).name
        if target.exists() and target.stat().st_size != entry.size:
            raise ValueError(f"existing file size disagrees with pinned source: {target}")
        return _stage_file(entry.path, target, pin, token)

    with ThreadPoolExecutor(max_workers=min(int(workers), 6)) as pool:
        paths = []
        for i, path in enumerate(pool.map(fetch, files), 1):
            paths.append(path)
            print(
                f"[natural-stage] unit {i}/{len(files)} {path.name} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    return paths


def _store_predicate(name: str) -> bool:
    """Select real global layer filenames plus all store-entry/index metadata."""
    from scripts.issue1739_map963k_slice import wanted_re

    return bool(wanted_re(KINDS, LAYERS).fullmatch(name)) or bool(
        re.fullmatch(r"_capture_(?:manifest|meta_shard\d+)\.json", name)
    )


def dv_sources(args, behaviors=None) -> dict[str, tuple[str, Path]]:
    """Source/consumer-exact DV map shared by staging and exclusion coverage."""
    out = {}
    for behavior in behaviors or args.behaviors:
        if behavior not in BEHAVIORS:
            raise ValueError(f"unknown behavior {behavior}")
        out[f"original:{behavior}"] = (
            f"{PREFIX}/judge/dv_dataset/{behavior}/labeling.json",
            args.store_root / "train_dv" / behavior / "labeling.json",
        )
        for family in ("wildchat_rung", "pvsynth"):
            out[f"{family}:{behavior}"] = (
                f"{PREFIX}/{family}/dv_dataset/{behavior}/labeling.json",
                args.main_root / family / "dv_dataset" / behavior / "labeling.json",
            )
        if behavior == "evil":
            out["wide:evil"] = (
                f"{PREFIX}/evil_ood_full/dv_dataset/evil/labeling.json",
                args.main_root / "evil_ood_full/dv_dataset/evil/labeling.json",
            )
        if behavior == "sycophancy":
            out["wide:sycophancy"] = (
                f"{PREFIX}/syco_ood/dv_dataset/sycophancy/labeling.json",
                args.ood_mirror_root / PREFIX / "syco_ood/dv_dataset/sycophancy/labeling.json",
            )
    return out


def stage_behavior(args, behavior: str, token: str) -> dict:
    """Stage required labeled/evaluation inputs only, never generic crossed data.

    Namespace: store_root, main_root, tensors_root, ood_mirror_root (Path),
    revision (immutable SHA), stage_workers (int). Optional materialize_labeling_tars
    and labeling_tar_staging_dir are forwarded exactly to stream_slice.
    Labeling tars still transfer in full (32–70 GB each); only 4/28 layer arrays
    are retained. Other stores are downloaded selectively by filename.
    """
    from scripts.issue1739_jobd_r2aug_run import write_canary
    from scripts.issue1739_map963k_slice import _extract_selected_members, stream_slice, wanted_re
    from scripts.issue1739_r2v2_score import OOD_SPECS
    from scripts.issue1739_wcrung_arms_run import staged_slice_covers

    pin = _revision(args)
    if behavior not in BEHAVIORS:
        raise ValueError(f"unknown behavior {behavior}")
    write_canary(args.store_root, gib=1.0)
    for remote, target in dv_sources(args, [behavior]).values():
        _stage_file(remote, target, pin, token)
    _stage_file(
        f"{PREFIX}/analysis_tensors/r_b_e1/{behavior}.npz",
        args.tensors_root / "r_b_e1" / f"{behavior}.npz",
        pin,
        token,
    )
    stores = [
        (
            f"{PREFIX}/wildchat_rung/capture_store/wildchat",
            args.store_root / "wcrung_capture_store/wildchat",
        ),
        (
            f"{PREFIX}/pvsynth/capture_store/{behavior}",
            args.store_root / "pvsynth_capture_store" / behavior,
        ),
    ]
    for rel in OOD_SPECS.get(behavior, {}).get("stores", ()):
        stores.append((f"{PREFIX}/{rel}", args.ood_mirror_root / PREFIX / rel))
    for prefix, dest in stores:
        _stage_selected(prefix, dest, pin, token, _store_predicate, args.stage_workers)
        if not list(dest.glob("row_index*.jsonl")):
            raise ValueError(f"staged store cannot be opened: missing row index at {dest}")
    dest = args.store_root / f"{behavior}_labeling"
    covered, _why = staged_slice_covers(dest, kinds=KINDS, layers=list(LAYERS))
    manifest = dest / "slice_manifest.json"
    if manifest.exists() and _json(manifest).get("revision") != pin:
        raise ValueError(f"labeling slice revision mismatch: {manifest}")
    if not covered:
        stream_slice(
            behavior,
            dest,
            revision=pin,
            kinds=KINDS,
            layers=LAYERS,
            token=token,
            workers=args.stage_workers,
            materialize=bool(getattr(args, "materialize_labeling_tars", False)),
            materialize_dir=getattr(args, "labeling_tar_staging_dir", None),
        )
    # load_behavior validates extraction-path existence even when the bank is used.
    # Supply a REAL selected extraction store, not an empty/dummy directory.
    extraction = args.store_root / f"{behavior}_extraction"
    emarker = extraction / ".natural_extraction.json"
    expected = {"revision": pin, "layers": list(LAYERS), "kinds": list(KINDS)}
    if emarker.exists() and _json(emarker) != expected:
        raise ValueError(f"extraction staging regime mismatch: {emarker}")
    if not emarker.exists():
        tar_path = _stage_file(
            f"{PREFIX}/capture_store/{behavior}_extraction/{behavior}_extraction.tar",
            args.store_root / "_source_tars" / pin / f"{behavior}_extraction.tar",
            pin,
            token,
        )
        extraction.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, mode="r|") as archive:
            _extract_selected_members(
                archive, dest=extraction, pat=wanted_re(KINDS, LAYERS), on_write=None
            )
        if not list(extraction.glob("row_index*.jsonl")):
            raise ValueError(f"extraction row index missing after staging: {extraction}")
        _atomic_json(emarker, expected)
    return {
        "behavior": behavior,
        "revision": pin,
        "layers": list(LAYERS),
        "generic_store_staged": False,
    }


def _prompt_parts(doc: dict) -> list[str]:
    """Actual prompt plus bare query; do not mistake generated answer text for input."""
    full = doc.get("prompt_text", doc.get("prompt"))
    query = doc.get("query")
    if not isinstance(full, str) or not full.strip():
        raise ValueError(f"actual prompt text absent for context {doc.get('context_id')}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"query text absent for context {doc.get('context_id')}")
    texts = [full, query]
    for turn in doc.get("prefix_turns", []):
        if turn.get("role") == "user" and isinstance(turn.get("content"), str):
            texts.append(turn["content"])
    return list(dict.fromkeys(texts))


def _packed_docs(root: Path, group: str):
    """Validate packer hashes/counts without expanding thousands of small files."""
    meta = _json(root / "pack_manifest.json")
    if meta.get("version") != 1 or group not in meta.get("groups", {}):
        raise ValueError(f"invalid pack manifest or missing group {group} at {root}")
    desc = meta["groups"][group]
    total = 0
    for shard in desc["shards"]:
        path = root / shard["name"]
        if _hash(path) != shard["sha256"]:
            raise ValueError(f"packed text SHA256 mismatch: {path}")
        count = 0
        for record in _lines(path):
            count += 1
            yield record["doc"]
        if count != shard["n_lines"]:
            raise ValueError(f"packed line-count mismatch: {path}")
        total += count
    if total != desc["n_files"]:
        raise ValueError(f"packed total-count mismatch: {group}")


def export_exclusions(args, out_path: Path) -> dict:
    """Stage text at the pin, export {text,source,id}, and require full DV coverage.

    Additional Namespace: behaviors, exclusion_stage_root (Path). Text staging
    is independent of activation staging. All DV contexts are conservatively
    excluded, not only one selected P-B holdout. Publication is atomic and fails
    if ANY required context lacks its actual prompt. No text is logged.
    """
    from scripts.issue1739_wcrung_rows_io import load_rows

    pin = _revision(args)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    cache = args.exclusion_stage_root / pin
    required, dv_hashes = {}, {}
    for source, (remote, _consumer) in dv_sources(args).items():
        path = _stage_file(remote, cache / remote, pin, token)
        rows = _json(path)["rows"]
        ids = {str(row["context_id"]) for row in rows}
        if not ids:
            raise ValueError(f"empty required DV source: {source}")
        required[source] = ids
        dv_hashes[source] = _hash(path)
    covered = {source: set() for source in required}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    counts = Counter()
    with tmp.open("w") as output:

        def consume(source, docs):
            for doc in docs:
                cid = str(doc.get("context_id", ""))
                if cid not in required[source] or cid in covered[source]:
                    continue
                for text in _prompt_parts(doc):
                    output.write(
                        json.dumps({"text": text, "source": source, "id": cid}, ensure_ascii=False)
                        + "\n"
                    )
                    counts[source] += 1
                covered[source].add(cid)

        # Original train/eval: authoritative pack manifest includes all its shards.
        pack_prefix = f"{PREFIX}/raw_completions"
        pack_root = cache / pack_prefix
        pack = _stage_file(
            f"{pack_prefix}/pack_manifest.json", pack_root / "pack_manifest.json", pin, token
        )
        pack_meta = _json(pack)
        for behavior in args.behaviors:
            group = f"labeling_{behavior}"
            if group not in pack_meta["groups"]:
                raise ValueError(f"missing original completion group: {group}")
            for shard in pack_meta["groups"][group]["shards"]:
                _stage_file(f"{pack_prefix}/{shard['name']}", pack_root / shard["name"], pin, token)
            consume(f"original:{behavior}", _packed_docs(pack_root, group))
        wc_prefix = f"{PREFIX}/wildchat_rung/contexts"
        wc_root = cache / wc_prefix
        _stage_selected(
            wc_prefix,
            wc_root,
            pin,
            token,
            lambda name: name.startswith("wcrung_rows"),
            args.stage_workers,
        )
        wc_rows = load_rows(wc_root)
        for behavior in args.behaviors:
            consume(f"wildchat_rung:{behavior}", wc_rows)
            source = f"pvsynth:{behavior}"
            prefix = f"{PREFIX}/pvsynth/raw_completions/{behavior}"
            ids = required[source]
            names = {f"{cid}_seed0.json" for cid in ids}
            paths = _stage_selected(
                prefix, cache / prefix, pin, token, lambda name: name in names, args.stage_workers
            )
            consume(source, (_json(path) for path in paths))
        # Wide-OOD raw trees may have nested producer shards. Read actual seed-0
        # generation records; staged syco contexts can predate cascade regeneration.
        wide = {
            "evil": f"{PREFIX}/raw_completions/evil_ood_spread_full",
            "sycophancy": f"{PREFIX}/syco_ood/raw_completions/main",
        }
        for behavior, prefix in wide.items():
            source = f"wide:{behavior}"
            if source not in required:
                continue
            entries = _entries(prefix, pin, token)
            pack_path = f"{prefix}/pack_manifest.json"
            if any(entry.path == pack_path for entry in entries):
                root = cache / prefix
                manifest = _json(_stage_file(pack_path, root / "pack_manifest.json", pin, token))
                # Manifest first: syco main currently retains stale old shards
                # beside its six authoritative regenerated root shards.
                if not manifest.get("groups"):
                    raise ValueError(f"empty wide-OOD pack manifest: {pack_path}")
                for group, desc in manifest["groups"].items():
                    for shard in desc["shards"]:
                        _stage_file(f"{prefix}/{shard['name']}", root / shard["name"], pin, token)
                    consume(source, _packed_docs(root, group))
                continue
            candidates = {}
            for entry in entries:
                name = Path(entry.path).name
                if name.endswith("_seed0.json"):
                    cid = name.removesuffix("_seed0.json")
                    if cid in required[source]:
                        if cid in candidates:
                            raise ValueError(f"ambiguous raw completion for {source}/{cid}")
                        candidates[cid] = entry.path
            for cid, remote in sorted(candidates.items()):
                path = _stage_file(remote, cache / remote, pin, token)
                consume(source, [_json(path)])
        missing = {
            source: sorted(ids - covered[source])
            for source, ids in required.items()
            if ids - covered[source]
        }
        if missing:
            detail = {
                source: {"count": len(ids), "first_ids": ids[:5]} for source, ids in missing.items()
            }
            raise ValueError(f"incomplete exclusion prompt coverage: {detail}")
    os.replace(tmp, out_path)
    report = {
        "revision": pin,
        "repo": REPO,
        "complete": True,
        "required_contexts": {k: len(v) for k, v in required.items()},
        "covered_contexts": {k: len(v) for k, v in covered.items()},
        "text_records": dict(counts),
        "dv_sha256": dv_hashes,
        "exclusion_sha256": _hash(out_path),
    }
    _atomic_json(out_path.with_suffix(".manifest.json"), report)
    return report
