"""Synthetic tests for the #1739 round-4 raw-completions packer.

No network, no HF Hub: the Hub boundary is stubbed (autospec'd
``hub._upload`` + a signature-mirroring ``verify_repo_paths_uploaded``
fake). All fixture docs are neutral synthetic placeholder content.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

import scripts.issue1739_upload as up
from scripts.issue1739_pack import (
    MANIFEST_NAME,
    group_files,
    pack_raw_tree,
    unpack_shards,
)
from scripts.issue1739_pack import main as pack_main


def _mk_raw(tmp_path: Path, n_per_group: int = 5) -> Path:
    """Two labeling groups + one extraction group of tiny synthetic docs."""
    raw = tmp_path / "raw_completions" / "issue_1739"
    for beh in ("hallucination", "sycophancy"):
        d = raw / "labeling" / beh
        d.mkdir(parents=True)
        for i in range(n_per_group):
            (d / f"ctx{i:04d}_seed0.json").write_text(
                json.dumps(
                    {
                        "context_id": f"ctx{i:04d}",
                        "seed": 0,
                        "behavior": beh,
                        "text": f"synthetic placeholder {beh} row {i}",
                    }
                ),
                encoding="utf-8",
            )
    e = raw / "extraction" / "rollouts"
    e.mkdir(parents=True)
    for i in range(3):
        (e / f"roll{i}.json").write_text(
            json.dumps({"i": i, "text": "synthetic rollout"}), encoding="utf-8"
        )
    return raw


def _shard_lines(pack_root: Path, manifest: dict) -> list[dict]:
    """All packed records, in manifest (group, shard, line) order."""
    recs: list[dict] = []
    for key in sorted(manifest["groups"]):
        for shard in manifest["groups"][key]["shards"]:
            with (pack_root / shard["name"]).open(encoding="utf-8") as fh:
                recs.extend(json.loads(line) for line in fh if line.strip())
    return recs


def test_round_trip_covers_every_file(tmp_path):
    raw = _mk_raw(tmp_path)
    pack = tmp_path / "packed"
    manifest = pack_raw_tree(raw, pack)
    recs = _shard_lines(pack, manifest)
    expected = {p.relative_to(raw).as_posix() for p in raw.rglob("*.json")}
    assert {r["src"] for r in recs} == expected
    assert len(recs) == len(expected)  # each file exactly once
    for r in recs:  # unpack a shard line -> original doc
        assert r["doc"] == json.loads((raw / r["src"]).read_text(encoding="utf-8"))
    assert sum(g["n_files"] for g in manifest["groups"].values()) == len(expected)


def test_pack_determinism(tmp_path):
    raw = _mk_raw(tmp_path)
    pack_a, pack_b = tmp_path / "pack_a", tmp_path / "pack_b"
    man_a = pack_raw_tree(raw, pack_a)
    man_b = pack_raw_tree(raw, pack_b)
    assert man_a["groups"] == man_b["groups"]  # census + shard shas identical
    for key, g in man_a["groups"].items():
        for shard in g["shards"]:
            assert (pack_a / shard["name"]).read_bytes() == (pack_b / shard["name"]).read_bytes(), (
                f"{key}/{shard['name']} differs between packs"
            )


def test_shard_size_boundary(tmp_path):
    raw = _mk_raw(tmp_path, n_per_group=8)
    pack = tmp_path / "packed"
    cap = 200  # tiny cap so fixture lines (~100 bytes) split across shards
    manifest = pack_raw_tree(raw, pack, shard_max_bytes=cap)
    g = manifest["groups"]["labeling_hallucination"]
    assert g["n_shards"] > 1
    for shard in g["shards"]:
        size = (pack / shard["name"]).stat().st_size
        assert size == shard["bytes"]
        assert size <= cap, f"{shard['name']} exceeds the cap with multi-line content"
    # boundary: no shard could have absorbed the NEXT shard's first line
    for a, b in zip(g["shards"], g["shards"][1:], strict=False):
        first_b = (pack / b["name"]).open("rb").readline()
        assert a["bytes"] + len(first_b) > cap
    # order is preserved across the shard sequence
    recs = []
    for shard in g["shards"]:
        with (pack / shard["name"]).open(encoding="utf-8") as fh:
            recs.extend(json.loads(line)["src"] for line in fh if line.strip())
    assert recs == sorted(recs)


def test_single_oversized_doc_gets_own_shard(tmp_path):
    raw = tmp_path / "raw"
    d = raw / "labeling" / "hallucination"
    d.mkdir(parents=True)
    (d / "a_small.json").write_text(json.dumps({"t": "x"}), encoding="utf-8")
    (d / "b_big.json").write_text(json.dumps({"t": "y" * 500}), encoding="utf-8")
    (d / "c_small.json").write_text(json.dumps({"t": "z"}), encoding="utf-8")
    manifest = pack_raw_tree(raw, tmp_path / "packed", shard_max_bytes=120)
    shards = manifest["groups"]["labeling_hallucination"]["shards"]
    assert [s["n_lines"] for s in shards] == [1, 1, 1]
    assert shards[1]["bytes"] > 120  # the oversized line rides alone, over cap


def test_census_mismatch_forces_repack_and_match_reuses(tmp_path):
    raw = _mk_raw(tmp_path)
    pack = tmp_path / "packed"
    pack_raw_tree(raw, pack)
    # unchanged tree -> every group reused
    man2 = pack_raw_tree(raw, pack)
    assert sorted(man2["reused_groups"]) == sorted(man2["groups"])
    assert man2["repacked_groups"] == []
    # mutate ONE file -> only its group repacks; the new doc lands in the shard
    target = raw / "labeling" / "hallucination" / "ctx0000_seed0.json"
    target.write_text(json.dumps({"context_id": "ctx0000", "changed": True}), encoding="utf-8")
    man3 = pack_raw_tree(raw, pack)
    assert man3["repacked_groups"] == ["labeling_hallucination"]
    assert set(man3["reused_groups"]) == set(man3["groups"]) - {"labeling_hallucination"}
    recs = {r["src"]: r["doc"] for r in _shard_lines(pack, man3)}
    assert recs["labeling/hallucination/ctx0000_seed0.json"] == {
        "context_id": "ctx0000",
        "changed": True,
    }
    # a missing shard file also forces repack even with a matching census
    victim = pack / man3["groups"]["extraction_rollouts"]["shards"][0]["name"]
    victim.unlink()
    man4 = pack_raw_tree(raw, pack)
    assert "extraction_rollouts" in man4["repacked_groups"]
    assert victim.is_file()


def test_group_files_rejects_non_json(tmp_path):
    raw = _mk_raw(tmp_path)
    (raw / "labeling" / "stray.txt").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match=r"non-\.json"):
        group_files(raw)


def test_raw_stage_packs_then_bulk_uploads_with_exact_set_verify(tmp_path, monkeypatch):
    """The production CLI path: --stage raw packs first, then ONE bulk
    upload of the SHARD dir with the exact-set verify covering shard names
    + manifest. Hub boundary stubbed (autospec'd _upload; signature-
    mirroring verify fake)."""
    from explore_persona_space.orchestrate import hub

    raw = _mk_raw(tmp_path)
    pack = tmp_path / "packed"
    fake_upload = mock.create_autospec(hub._upload, return_value="https://hf.co/fake")
    captured: dict = {}

    def fake_verify(
        api, repo_id, expected_repo_paths, *, path_in_repo, repo_type="dataset", revision=None
    ):
        captured["expected"] = list(expected_repo_paths)
        captured["path_in_repo"] = path_in_repo
        return []

    monkeypatch.setattr(hub, "_upload", fake_upload)
    monkeypatch.setattr(hub, "verify_repo_paths_uploaded", fake_verify)
    rc = up.main(["--stage", "raw", "--raw-root", str(raw), "--pack-root", str(pack)])
    assert rc == 0
    manifest = json.loads((pack / MANIFEST_NAME).read_text(encoding="utf-8"))
    shard_names = {s["name"] for g in manifest["groups"].values() for s in g["shards"]}
    assert shard_names, "raw stage did not pack anything"
    (call,) = fake_upload.call_args_list  # ONE bulk commit of the PACKED dir
    assert Path(call.args[0]) == pack
    assert call.kwargs["path_in_repo"] == "issue1739_ctxmap/raw_completions"
    assert set(captured["expected"]) == {
        f"issue1739_ctxmap/raw_completions/{n}" for n in shard_names | {MANIFEST_NAME}
    }


def test_raw_stage_dry_run_touches_nothing(tmp_path, capsys):
    raw = _mk_raw(tmp_path)
    pack = tmp_path / "packed"
    rc = up.main(["--stage", "raw", "--raw-root", str(raw), "--pack-root", str(pack), "--dry-run"])
    assert rc == 0
    assert not pack.exists()  # dry-run neither packs nor uploads
    out = capsys.readouterr().out
    assert "dry-run" in out and "labeling_hallucination" in out


# ---------------------------------------------------------------------------
# r5: unpack mode
# ---------------------------------------------------------------------------


def _mk_raw_producer(tmp_path: Path, n_per_group: int = 4) -> Path:
    """Raw tree written in the PRODUCER serialization (generation.py
    ``_atomic_write_json``: ``json.dumps(obj, ensure_ascii=False, indent=1)``,
    no trailing newline) so pack->unpack round-trips byte-identical.
    Includes a non-ASCII value to exercise ensure_ascii=False."""
    raw = tmp_path / "raw_completions" / "issue_1739"
    for beh in ("hallucination", "sycophancy"):
        d = raw / "labeling" / beh
        d.mkdir(parents=True)
        for i in range(n_per_group):
            doc = {
                "context_id": f"ctx{i:04d}",
                "seed": 0,
                "behavior": beh,
                "text": f"synthetic placeholder — {beh} row {i} · non-ascii",
            }
            (d / f"ctx{i:04d}_seed0.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8"
            )
    e = raw / "extraction" / "rollouts"
    e.mkdir(parents=True)
    for i in range(2):
        (e / f"roll{i}.json").write_text(
            json.dumps({"i": i, "text": "synthetic rollout"}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return raw


def test_unpack_round_trip_byte_identical(tmp_path, capsys):
    raw = _mk_raw_producer(tmp_path)
    pack = tmp_path / "packed"
    manifest = pack_raw_tree(raw, pack)
    out = tmp_path / "restored"
    summary = unpack_shards(pack, out)
    originals = sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*.json"))
    restored = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.json"))
    assert restored == originals
    for rel in originals:  # byte-identical under the producer serialization
        assert (out / rel).read_bytes() == (raw / rel).read_bytes(), rel
    assert sum(s["written"] for s in summary.values()) == len(originals)
    assert sum(s["skipped"] for s in summary.values()) == 0
    assert set(summary) == set(manifest["groups"])
    text = capsys.readouterr().out
    assert "[unpack] labeling_hallucination: restored" in text


def test_unpack_group_subset_and_unknown_group(tmp_path):
    raw = _mk_raw_producer(tmp_path)
    pack = tmp_path / "packed"
    pack_raw_tree(raw, pack)
    out = tmp_path / "restored"
    summary = unpack_shards(pack, out, groups=["labeling_hallucination"])
    assert set(summary) == {"labeling_hallucination"}
    assert {p.relative_to(out).parts[0] for p in out.rglob("*.json")} == {"labeling"}
    assert not (out / "extraction").exists()
    with pytest.raises(SystemExit, match="unknown group"):
        unpack_shards(pack, out, groups=["nope_group"])


def test_unpack_idempotent_then_fails_loud_on_differing_existing(tmp_path):
    raw = _mk_raw_producer(tmp_path)
    pack = tmp_path / "packed"
    pack_raw_tree(raw, pack)
    out = tmp_path / "restored"
    unpack_shards(pack, out)
    # re-run over the identical tree: everything skips, nothing rewritten
    summary = unpack_shards(pack, out)
    assert all(s["written"] == 0 for s in summary.values())
    assert sum(s["skipped"] for s in summary.values()) == len(list(out.rglob("*.json")))
    # a differing existing file is NEVER overwritten
    victim = out / "labeling" / "hallucination" / "ctx0000_seed0.json"
    victim.write_text(json.dumps({"context_id": "ctx0000", "tampered": True}), encoding="utf-8")
    before = victim.read_bytes()
    with pytest.raises(SystemExit, match="DIFFERING"):
        unpack_shards(pack, out)
    assert victim.read_bytes() == before


def test_unpack_verifies_manifest_counts_and_shard_sha(tmp_path):
    raw = _mk_raw_producer(tmp_path)
    pack = tmp_path / "packed"
    manifest = pack_raw_tree(raw, pack)
    # (a) manifest n_files mismatch fails loud (shards intact)
    bad = json.loads((pack / MANIFEST_NAME).read_text(encoding="utf-8"))
    bad["groups"]["extraction_rollouts"]["n_files"] += 1
    (pack / MANIFEST_NAME).write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SystemExit, match="n_files"):
        unpack_shards(pack, tmp_path / "r1", groups=["extraction_rollouts"])
    # (b) a corrupted shard fails the sha256 check before counts
    (pack / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shard = pack / manifest["groups"]["labeling_sycophancy"]["shards"][0]["name"]
    lines = shard.read_bytes().splitlines(keepends=True)
    shard.write_bytes(b"".join(lines[:-1]))  # drop the last record
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        unpack_shards(pack, tmp_path / "r2", groups=["labeling_sycophancy"])


def test_unpack_cli_entrypoint(tmp_path):
    raw = _mk_raw_producer(tmp_path)
    pack = tmp_path / "packed"
    pack_raw_tree(raw, pack)
    out = tmp_path / "restored"
    rc = pack_main(["--unpack", "--shards-dir", str(pack), "--out-root", str(out)])
    assert rc == 0
    assert sorted(p.name for p in (out / "extraction" / "rollouts").iterdir()) == [
        "roll0.json",
        "roll1.json",
    ]
