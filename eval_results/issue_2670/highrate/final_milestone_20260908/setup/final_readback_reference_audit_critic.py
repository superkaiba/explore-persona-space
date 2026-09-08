"""Independent final archive bytes, pinned metadata and owner evidence check."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

SETUP = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("snapshot_critic", SETUP / "final_snapshot_reference_byte_audit_critic.py")
assert spec and spec.loader
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
sha, read, safe, check, inventory = prior.sha, prior.read, prior.safe, prior.check, prior.inventory
R, W, B = prior.R, prior.W, prior.B
REV = "6994094e632aaa7fb50357f52d1fcde82e4fdc7c"
PREFIX = "context_risk/issue2670_highrate/final"
PHASE1 = SETUP / "final_snapshot_source_byte_review_20260908T1259Z.json"
PHASE1_SHA = "f68874af60bfd555d7ad29be01af27e4040c5b4de04b4e2a5f8fe4d4835fa994"
PINS = {
    B: "bde2d8d1fdecae462f8371ed3b5633b8d9e1d4bd77423ae9f4a640243d9906f0",
    R / "final_upload_receipt.json": "4218cda497b1715f68d38c65f37f915cb66db0b225710872a941335a0aa2a4df",
    R / "final_readback_receipt.json": "9f577c41f56cc2607121e3349c6f0401a390871e4242f84ddaf2bb90fdad6593",
    PHASE1: PHASE1_SHA,
}


def restored(base, prefix, key):
    return safe(base / ("reconstructed" if key.endswith(".jsonl") else prefix), key)


def binding(path):
    return {"path": str(path), "sha256": sha(path), "size": path.stat().st_size}


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run():
    started = time.monotonic()
    for path, expected in PINS.items():
        assert sha(path) == expected, str(path)
    build, phase1 = read(B), read(PHASE1)
    assert phase1["verdict"] == "PASS" and not phase1["unresolved_findings"]
    assert sha(Path(phase1["driver"]["path"])) == phase1["driver"]["sha256"]
    snap = Path(build["freeze_result"]["snapshot"])
    landed = Path(build["freeze_result"]["readback"])
    manifest_path = snap / "snapshot_manifest.json"
    assert sha(manifest_path) == prior.PIN == phase1["snapshot_manifest_sha256"]
    manifest = read(manifest_path)
    originals = manifest["files"]
    upload, readback = read(R / "final_upload_receipt.json"), read(R / "final_readback_receipt.json")
    for receipt in (upload, readback):
        assert receipt["passed"] is True
        assert (receipt["repo_id"], receipt["prefix"], receipt["revision"]) == (prior.REPO, PREFIX, REV)
    assert readback["phase"] == "final" and readback["snapshot_sha256"] == prior.PIN
    assert readback["upload_receipt_sha256"] == PINS[R / "final_upload_receipt.json"]
    assert readback["sources_sha256"] == manifest["archive_sources_sha256"] == build["archive_sources_sha256"]
    assert upload["verified_files"] == readback["downloaded_files"] == len(upload["files"]) == 2176
    assert originals == build["selected_files"] and set(originals) == set(readback["original_files"])
    assert len(originals) == 2175 and sum(v["size"] for v in originals.values()) == 276670758
    assert {PREFIX + "/" + key for key in inventory(landed / PREFIX)} == set(upload["files"])
    for key, value in upload["files"].items():
        check(safe(landed, key), value)
    archived_manifest = safe(landed / PREFIX, "snapshot_manifest.json")
    assert sha(archived_manifest) == prior.PIN and read(archived_manifest) == manifest
    for key, value in originals.items():
        assert all(value[f] == readback["original_files"][key][f] for f in ("size", "sha256"))
        check(restored(landed, PREFIX, key), value)
    print("RESTORED_BYTES_PASS 2175 originals; 2176 remote files", flush=True)

    references = read(restored(landed, PREFIX, manifest["raw_reference_key"]))
    raw_manifest_path = restored(landed, PREFIX, "references/raw_snapshot_manifest.json")
    assert sha(raw_manifest_path) == prior.RAW_PIN == references["snapshot_sha256"]
    raw_originals = read(raw_manifest_path)["files"]
    assert raw_originals == references["raw_original_files"] and len(raw_originals) == 624
    raw_u_path = restored(landed, PREFIX, "references/raw_upload_receipt.json")
    raw_r_path = restored(landed, PREFIX, "references/raw_readback_receipt.json")
    raw_u, raw_r = read(raw_u_path), read(raw_r_path)
    assert sha(raw_r_path) == "a8cb378a69017833f59116922f7b124a33301f4e9338dc372aef8eddb3c442c4"
    assert sha(raw_u_path) == "4470ed8480598e979d8bbe5d93b5e5186d8ecc18fe76ad84a861312fc3732569"
    assert raw_r["passed"] is True and raw_u["passed"] is True
    assert raw_r["upload_receipt_sha256"] == sha(raw_u_path) and raw_r["snapshot_sha256"] == prior.RAW_PIN
    for value in (references, raw_r, raw_u):
        assert (value["repo_id"], value["prefix"], value["revision"]) == (prior.REPO, prior.RAW_PREFIX, prior.RAW_REV)
    assert set(raw_originals) == set(raw_r["original_files"])
    for key, value in raw_originals.items():
        assert all(value[f] == raw_r["original_files"][key][f] for f in ("size", "sha256"))
    table = references["consumed_input_source_references"]
    assert len(table) == 212
    locations = Counter()
    source_index = {}
    for place, values in (("final", originals), ("pinned_raw", raw_originals)):
        for key, value in values.items():
            source_index.setdefault((value["source"], value["sha256"]), []).append((place, key, value))
    for label, value in table.items():
        place, key = value["location"], value["snapshot_key"]
        assert place in {"final", "pinned_raw"}, label
        row = (originals if place == "final" else raw_originals)[key]
        assert all(row[f] == value[f] for f in ("source", "size", "sha256")), label
        literal = Path(value["requested_source_path"])
        assert str(literal if literal.is_absolute() else Path(value["relative_resolution_base"]) / literal) == value["source"]
        if place == "pinned_raw":
            assert (value["prefix"], value["revision"]) == (prior.RAW_PREFIX, prior.RAW_REV)
        check(restored(landed, PREFIX, key) if place == "final" else restored(prior.RAW_BASE, prior.RAW_PREFIX, key), row)
        locations[place] += 1
    assert dict(locations) == {"final": 137, "pinned_raw": 75}
    for field, count in (("analysis_sources_sha256", 45), ("archive_sources_sha256", 5)):
        assert len(manifest[field]) == count and manifest[field] == build[field]
        for relative, expected in manifest[field].items():
            path = W / relative
            assert sha(path) == expected
            matches = source_index[(str(path), expected)]
            for place, key, value in matches:
                check(restored(landed, PREFIX, key) if place == "final" else restored(prior.RAW_BASE, prior.RAW_PREFIX, key), value)
    print("SOURCE_REFERENCE_PASS 212 literal source paths; current45/5 closures", flush=True)

    from explore_persona_space.orchestrate.env import load_dotenv
    load_dotenv()
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile, RepoFolder
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    entries = list(api.list_repo_tree(repo_id=prior.REPO, path_in_repo=PREFIX, repo_type="dataset", revision=REV, recursive=True))
    assert all(isinstance(v, (RepoFile, RepoFolder)) for v in entries)
    files = [v for v in entries if isinstance(v, RepoFile)]
    assert len(files) == len({v.path for v in files}) == 2176
    assert {v.path for v in files} == set(upload["files"])
    remote_proof = {}
    kinds = Counter()
    for entry in files:
        expected = upload["files"][entry.path]
        assert entry.size == expected["size"]
        if entry.lfs is not None:
            assert entry.lfs.sha256 == expected["sha256"] and entry.lfs.size == expected["size"]
            kind, identity = "LFS_SHA256", entry.lfs.sha256
        else:
            payload = safe(landed, entry.path).read_bytes()
            identity = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            assert identity == entry.blob_id
            kind = "Git_blob_SHA1"
        remote_proof[entry.path] = {"size": entry.size, "sha256": expected["sha256"], "identity_kind": kind, "remote_content_identity": identity}
        kinds[kind] += 1
    print("PINNED_API_PASS", len(files), dict(kinds), REV, flush=True)

    owner_path = SETUP / "final_archive_20260908T125035Z_owner_launch.json"
    exit_path = SETUP / "final_archive_20260908T125035Z_exit.json"
    owner, ended = read(owner_path), read(exit_path)
    assert owner["pid"] == owner["pgid"] == owner["sid"] == 1613108
    assert owner["launch_id"] == ended["launch_id"] == "final_archive_20260908T125035Z"
    assert ended["exit_code"] == 0 and ended["live_group_members"] == [] and ended["cleanup"] == "no_live_members"
    assert ended["supervisor_error"] is None and ended["cleanup_error"] is None
    assert ended["launch_sha256"] == sha(owner_path) == "c67daa75340fe980bb0dd75b70509fe6244d1f91c7510df1a81b2461666346aa"
    log_path = Path(owner["log_path"])
    assert ended["log_sha256"] == sha(log_path) == "547524e13471a52aef0815669feac0233310d2fa46c27a0253cd66ed50024d53"
    assert ended["build_receipt_sha256"] == owner["build_receipt_sha256"] == PINS[B]
    assert ended["raw_timing_receipt_sha256"] == owner["raw_timing_receipt_sha256"] == sha(raw_r_path)
    assert owner["snapshot_manifest_sha256"] == prior.PIN and owner["archive_sources_sha256"] == manifest["archive_sources_sha256"]
    supervisor_path = SETUP / "run_final_archive_owned.py"
    supervisor_review_path = SETUP / "final_archive_launcher_code_review_20260908.json"
    assert sha(supervisor_path) == owner["supervisor_sha256"] == "7c92cc50f93da5c8072c52fbff79a8bc64fafcd22831de77e011a084a3d02177"
    assert sha(supervisor_review_path) == "34f6dfe1e6de4ad05dc362f99e5942a0a42850a121a9ab345f44ab59816daae7"
    assert read(supervisor_review_path)["verdict"] == "PASS"
    assert set(ended["receipts"]) == {"final_upload_receipt.json", "final_readback_receipt.json"}
    for name, value in ended["receipts"].items():
        assert value["sha256"] == PINS[R / name] and value["passed"] is True and value["revision"] == REV
    start = datetime.fromisoformat(owner["started_utc"])
    end = datetime.fromisoformat(ended["finished_utc"])
    assert end > start and abs((end - start).total_seconds() - ended["elapsed_seconds"]) < 1
    assert 0 < readback["elapsed_seconds"] < ended["elapsed_seconds"] < owner["timeout_seconds"]
    protection_path = SETUP / "final_archive_20260908T125035Z_owner_protection.json"
    protection = read(protection_path)
    assert protection["exit_code"] == 0 and protection["oom_score_adj"] == -600
    members, vanished = [], 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            data = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            vanished += 1
            continue
        if int(data[2]) == owner["pgid"] or int(data[3]) == owner["sid"]:
            members.append({"pid": int(proc.name), "state": data[0], "pgid": int(data[2]), "sid": int(data[3])})
    assert not [v for v in members if v["state"] not in {"Z", "X"}]
    print("OWNER_EXIT_DRAIN_PASS", ended["finished_utc"], flush=True)
    for path, expected in PINS.items():
        assert sha(path) == expected
    assert sha(manifest_path) == prior.PIN
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    api_output = SETUP / f"final_pinned_api_tree_review_{stamp}.json"
    write_new(api_output, {"schema_version": "task2670_final_pinned_api_tree_v1", "verdict": "PASS", "checked_utc": datetime.now(UTC).isoformat(), "repo_id": prior.REPO, "prefix": PREFIX, "revision": REV, "exact_file_count": len(files), "identity_kinds": dict(kinds), "files": remote_proof})
    output = SETUP / f"final_pinned_readback_reference_review_{stamp}.json"
    write_new(output, {
        "schema_version": "task2670_final_pinned_readback_reference_review_v1", "verdict": "PASS", "reviewer": "v9_consumer_critic independent actual final readback and API check", "checked_utc": datetime.now(UTC).isoformat(),
        "driver": binding(Path(__file__)), "helper": binding(Path(prior.__file__)), "phase1_source_byte_review": binding(PHASE1), "build_receipt": binding(B),
        "snapshot_manifest": binding(manifest_path), "final_upload_receipt": binding(R / "final_upload_receipt.json"), "final_readback_receipt": binding(R / "final_readback_receipt.json"), "pinned_api_review": binding(api_output),
        "repo_id": prior.REPO, "prefix": PREFIX, "revision": REV, "url": upload["url"], "readback": str(landed),
        "coverage": {"remote_files_exact_api_and_readback_set": 2176, "restored_original_files_byte_exact": 2175, "restored_original_bytes": 276670758, "explicit_consumed_source_paths": 212, "source_reference_locations": dict(locations), "current_analysis_sources": 45, "current_archive_sources": 5, "pinned_raw_original_metadata_entries": 624},
        "raw_reference": {"repo_id": prior.REPO, "prefix": prior.RAW_PREFIX, "revision": prior.RAW_REV, "snapshot_manifest_sha256": prior.RAW_PIN, "archived_upload_receipt": binding(raw_u_path), "archived_readback_receipt": binding(raw_r_path)},
        "sources_sha256": {"analysis": manifest["analysis_sources_sha256"], "archive": manifest["archive_sources_sha256"]},
        "owner_evidence": {"launch": binding(owner_path), "exit": binding(exit_path), "log": binding(log_path), "protection": binding(protection_path), "supervisor": binding(supervisor_path), "supervisor_review": binding(supervisor_review_path), "exit_code": ended["exit_code"], "finished_utc": ended["finished_utc"], "elapsed_seconds": ended["elapsed_seconds"], "fresh_proc_scan_utc": datetime.now(UTC).isoformat(), "matching_group_or_session_members": members, "vanished_unrelated_proc_entries": vanished},
        "checks": ["All downloaded2176 files opened and byte-hashed against pinned upload receipt; independent pinned API exact set, sizes and Git blob/LFS identities.", "All2175 restored original files opened and byte-hashed against immutable snapshot and actual readback receipt, including canonical reconstructed JSONL paths.", "All212 explicit source references preserve literal-to-absolute source path, SHA, size, key and pinned revision; raw references resolved against previously pinned raw readback actual bytes.", "Current45 analysis and5 archive source bytes equal frozen closure and resolve through source+SHA index to final or pinned raw originals.", "Actual successful owner exit binds exact launch/log/build/raw/final receipt hashes; fresh read-only proc scan confirms no live owned group/session.", "Earlier independent phase1 review supplies complete physical-source, scientific-audit/report/figure and failure-census coverage; these are not recomputed here."],
        "native_semantic_parsing_or_scientific_refitting": False, "uploads_or_snapshot_mutations": False, "unresolved_findings": [], "remaining_persistence": ["Push exact late independent reviews, archive owner receipts and final archive receipts to Git.", "Parent-owned canonical final body and methodology are late Git outputs, intentionally subsequent to this frozen science archive."], "elapsed_seconds": time.monotonic() - started,
    })
    print("FINAL_READBACK_PASS", output, sha(output), flush=True)
    print("API_REVIEW", api_output, sha(api_output), flush=True)


if __name__ == "__main__":
    run()
