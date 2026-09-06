"""CPU-only real-body staging and fail-loud exclusion coverage checks."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from scripts import issue1739_natural_inputs as n


def _bytes(value):
    return json.dumps(value).encode()


def _jsonl(rows):
    return b"".join(_bytes(row) + b"\n" for row in rows)


def _args(tmp_path):
    return SimpleNamespace(
        revision=n.REVISION,
        behaviors=["evil"],
        stage_workers=2,
        store_root=tmp_path / "stores",
        main_root=tmp_path / "main",
        tensors_root=tmp_path / "tensors",
        ood_mirror_root=tmp_path / "ood",
        exclusion_stage_root=tmp_path / "texts",
    )


def _doc(cid, text="natural query"):
    return {
        "context_id": cid,
        "query": text,
        "prompt_text": "formatted " + text,
        "completion": "NEVER EXPORT THIS ANSWER",
        "rollout_k": 0,
    }


@pytest.fixture
def remote(monkeypatch):
    """Only external API/filesystem-transfer boundaries are faked, with autospec."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    files = {}

    def tree(
        self,
        repo_id,
        path_in_repo=None,
        *,
        recursive=False,
        expand=False,
        revision=None,
        repo_type=None,
        token=None,
    ):
        assert repo_id == n.REPO and revision == n.REVISION and repo_type == "dataset"
        prefix = path_in_repo.rstrip("/") + "/"
        selected = [
            SimpleNamespace(path=path, size=len(data))
            for path, data in files.items()
            if path.startswith(prefix) and (recursive or "/" not in path.removeprefix(prefix))
        ]
        if not selected:
            raise FileNotFoundError(path_in_repo)
        return iter(selected)

    monkeypatch.setattr(
        HfApi, "list_repo_tree", create_autospec(HfApi.list_repo_tree, side_effect=tree)
    )

    def stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        assert repo_id == n.REPO and revision == n.REVISION and repo_type == "dataset"
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[path_in_repo])
        return target

    monkeypatch.setattr(
        hub, "stage_hub_file", create_autospec(hub.stage_hub_file, side_effect=stage)
    )
    monkeypatch.setattr(hub, "_assert_stage_headroom", create_autospec(hub._assert_stage_headroom))
    return files


def _exclusion_fixture(args, files):
    ids = {
        "original:evil": "orig",
        "wildchat_rung:evil": "wc",
        "pvsynth:evil": "pv",
        "wide:evil": "wide",
    }
    for source, (remote, _path) in n.dv_sources(args).items():
        files[remote] = _bytes({"rows": [{"context_id": ids[source]}]})
    prefix = f"{n.PREFIX}/raw_completions"
    shard = _jsonl(
        [
            {"src": "labeling/evil/_manifest.json", "doc": {"n_contexts": 1}},
            {"src": "labeling/evil/orig_seed0.json", "doc": _doc("orig")},
        ]
    )
    files[f"{prefix}/labeling_evil.shard00.jsonl"] = shard
    files[f"{prefix}/pack_manifest.json"] = _bytes(
        {
            "version": 1,
            "groups": {
                "labeling_evil": {
                    "n_files": 2,
                    "shards": [
                        {
                            "name": "labeling_evil.shard00.jsonl",
                            "n_lines": 2,
                            "sha256": hashlib.sha256(shard).hexdigest(),
                        }
                    ],
                }
            },
        }
    )
    wc = {**_doc("wc"), "prefix_turns": [{"role": "user", "content": "earlier turn"}]}
    wc["prompt"] = wc.pop("prompt_text")
    data = _jsonl([wc])
    prefix = f"{n.PREFIX}/wildchat_rung/contexts"
    files[f"{prefix}/wcrung_rows.shard00.jsonl"] = data
    files[f"{prefix}/wcrung_rows.manifest.json"] = _bytes(
        {
            "schema": "wcrung-rows-shards-v1",
            "n_rows": 1,
            "n_shards": 1,
            "shards": [
                {
                    "name": "wcrung_rows.shard00.jsonl",
                    "n_rows": 1,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            ],
        }
    )
    files[f"{n.PREFIX}/pvsynth/raw_completions/evil/pv_seed0.json"] = _bytes(_doc("pv"))
    files[
        f"{n.PREFIX}/raw_completions/evil_ood_spread_full/mhj_s0/rollouts/full/wide_seed0.json"
    ] = _bytes(_doc("wide"))


def test_export_all_required_families_actual_prompts_and_no_answers(tmp_path, remote):
    args = _args(tmp_path)
    _exclusion_fixture(args, remote)
    dest = tmp_path / "exclusions.jsonl"
    report = n.export_exclusions(args, dest)
    rows = list(n._lines(dest))
    assert report["complete"] is True
    assert set(report["covered_contexts"].values()) == {1}
    assert set(report["covered_contexts"]) == set(n.dv_sources(args))
    assert len(rows) == 9  # two texts per context, plus WC history's user turn
    assert all(set(row) == {"text", "source", "id"} for row in rows)
    assert "NEVER EXPORT" not in dest.read_text()
    assert n.export_exclusions(args, dest)["exclusion_sha256"] == report["exclusion_sha256"]


def test_missing_required_prompt_blocks_publication(tmp_path, remote):
    args = _args(tmp_path)
    _exclusion_fixture(args, remote)
    remote[f"{n.PREFIX}/evil_ood_full/dv_dataset/evil/labeling.json"] = _bytes(
        {"rows": [{"context_id": "wide"}, {"context_id": "absent"}]}
    )
    dest = tmp_path / "exclusions.jsonl"
    with pytest.raises(ValueError, match="incomplete exclusion prompt coverage"):
        n.export_exclusions(args, dest)
    assert not dest.exists()
    assert not dest.with_suffix(".manifest.json").exists()


def test_duplicate_producer_answers_require_identical_prompt_text(tmp_path, remote, capsys):
    args = _args(tmp_path)
    _exclusion_fixture(args, remote)
    duplicate = f"{n.PREFIX}/raw_completions/evil_ood_spread_full/retry/wide_seed0.json"
    remote[duplicate] = _bytes({**_doc("wide"), "completion": "different longer answer"})
    dest = tmp_path / "exclusions.jsonl"
    report = n.export_exclusions(args, dest)
    assert report["identical_prompt_copies_collapsed"] == {"wide:evil": 1}
    assert len(list(n._lines(dest))) == 9
    signal = "identical_prompt_copies source=wide:evil context_id=wide copies=2"
    assert signal in capsys.readouterr().out


def test_conflicting_producer_prompt_blocks_publication(tmp_path, remote):
    args = _args(tmp_path)
    _exclusion_fixture(args, remote)
    duplicate = f"{n.PREFIX}/raw_completions/evil_ood_spread_full/retry/wide_seed0.json"
    remote[duplicate] = _bytes(_doc("wide", "different prompt"))
    dest = tmp_path / "exclusions.jsonl"
    with pytest.raises(ValueError, match="conflicting raw prompts"):
        n.export_exclusions(args, dest)
    assert not dest.exists()


def test_filename_document_id_mismatch_blocks_publication(tmp_path, remote):
    args = _args(tmp_path)
    _exclusion_fixture(args, remote)
    source = f"{n.PREFIX}/raw_completions/evil_ood_spread_full/mhj_s0/rollouts/full/wide_seed0.json"
    remote[source] = _bytes(_doc("different-id"))
    with pytest.raises(ValueError, match="raw completion ID mismatch"):
        n.export_exclusions(args, tmp_path / "exclusions.jsonl")


def test_corrupt_packed_shard_blocks_export(tmp_path, remote):
    args = _args(tmp_path)
    _exclusion_fixture(args, remote)
    remote[f"{n.PREFIX}/raw_completions/labeling_evil.shard00.jsonl"] += b"\n"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        n.export_exclusions(args, tmp_path / "bad.jsonl")


def test_wide_packed_manifest_ignores_stale_sibling_shards(tmp_path, remote):
    args = _args(tmp_path)
    _exclusion_fixture(args, remote)
    prefix = f"{n.PREFIX}/raw_completions/evil_ood_spread_full"
    shard = _jsonl([{"src": "wide_seed0.json", "doc": _doc("wide")}])
    remote[f"{prefix}/root.shard00.jsonl"] = shard
    remote[f"{prefix}/root.shard99.jsonl"] = b"STALE NOT JSON"
    remote[f"{prefix}/pack_manifest.json"] = _bytes(
        {
            "version": 1,
            "groups": {
                "root": {
                    "n_files": 1,
                    "shards": [
                        {
                            "name": "root.shard00.jsonl",
                            "n_lines": 1,
                            "sha256": hashlib.sha256(shard).hexdigest(),
                        }
                    ],
                }
            },
        }
    )
    report = n.export_exclusions(args, tmp_path / "exclusions.jsonl")
    assert report["covered_contexts"]["wide:evil"] == 1
    assert not list(args.exclusion_stage_root.rglob("root.shard99.jsonl"))


def test_revision_and_unreceipted_files_fail_closed(tmp_path, remote):
    args = _args(tmp_path)
    args.revision = "main"
    with pytest.raises(ValueError, match="immutable"):
        n.export_exclusions(args, tmp_path / "bad.jsonl")
    target = tmp_path / "existing.json"
    target.write_text("{}")
    with pytest.raises(ValueError, match="unreceipted"):
        n._stage_file("example", target, n.REVISION, "")


def test_selected_stager_global_layer_filter_and_same_pin_resume(tmp_path, remote):
    prefix = "test/store"
    for name in (
        "context_end_L17_shard00.npy",
        "context_end_L01_shard00.npy",
        "t1_L20_shard00.npy",
        "row_index_shard00.jsonl",
        "_capture_manifest.json",
    ):
        remote[f"{prefix}/{name}"] = b"123"
    paths = n._stage_selected(prefix, tmp_path / "store", n.REVISION, "", n._store_predicate, 2)
    assert len(paths) == 4
    assert not any("L01" in p.name for p in paths)
    assert (
        len(n._stage_selected(prefix, tmp_path / "store", n.REVISION, "", n._store_predicate, 2))
        == 4
    )
    with pytest.raises(FileNotFoundError, match="empty selection"):
        n._stage_selected(prefix, tmp_path / "none", n.REVISION, "", lambda name: False)


def test_stage_behavior_real_body_selective_and_never_generic(tmp_path, remote, monkeypatch):
    from scripts import issue1739_jobd_r2aug_run as j
    from scripts import issue1739_map963k_slice as s
    from scripts.issue1739_r2v2_score import OOD_SPECS

    args = _args(tmp_path)
    for path, _target in n.dv_sources(args).values():
        remote[path] = _bytes({"rows": [{"context_id": "x"}]})
    remote[f"{n.PREFIX}/analysis_tensors/r_b_e1/evil.npz"] = b"bank"
    prefixes = [
        f"{n.PREFIX}/wildchat_rung/capture_store/wildchat",
        f"{n.PREFIX}/pvsynth/capture_store/evil",
    ]
    prefixes += [f"{n.PREFIX}/{rel}" for rel in OOD_SPECS["evil"]["stores"]]
    for prefix in prefixes:
        remote[f"{prefix}/row_index_shard00.jsonl"] = _jsonl([{"context_id": "x"}])
        remote[f"{prefix}/context_end_L17_shard00.npy"] = b"array"
        remote[f"{prefix}/context_end_L01_shard00.npy"] = b"not-selected"
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as archive:
        for name, data in [
            ("row_index.jsonl", _jsonl([{"context_id": "x"}])),
            ("context_end_L17.npy", b"real-array-placeholder-fixture"),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    remote[f"{n.PREFIX}/capture_store/evil_extraction/evil_extraction.tar"] = tar_bytes.getvalue()
    monkeypatch.setattr(j, "write_canary", create_autospec(j.write_canary))

    def slice_tar(
        behavior,
        dest,
        *,
        revision,
        kinds,
        layers,
        token,
        workers=12,
        window=32 << 20,
        materialize=False,
        materialize_dir=None,
    ):
        assert behavior == "evil" and tuple(layers) == n.LAYERS and tuple(kinds) == n.KINDS
        dest.mkdir(parents=True, exist_ok=True)
        payload = {"revision": revision, "layers": list(layers), "kinds": list(kinds)}
        (dest / "slice_manifest.json").write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(s, "stream_slice", create_autospec(s.stream_slice, side_effect=slice_tar))
    report = n.stage_behavior(args, "evil", "")
    assert report["generic_store_staged"] is False
    assert report["layers"] == [17, 18, 19, 20]
    assert (args.store_root / "evil_extraction/row_index.jsonl").exists()
    assert not list(args.store_root.rglob("*L01*.npy"))
    assert not list(args.store_root.rglob("*u_store*"))


def test_no_torch_or_hf_import_at_module_load():
    """Pure module imports defer heavy/external deps until caller configured env."""
    import ast

    tree = ast.parse(Path(n.__file__).read_text())
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [alias.name for node in imports if isinstance(node, ast.Import) for alias in node.names]
    assert "torch" not in names and "numpy" not in names and "huggingface_hub" not in names
