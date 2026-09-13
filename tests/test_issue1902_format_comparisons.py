"""Regression checks for raw preservation, shard ownership, and scientific contrasts."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import issue1902_format_common as C  # noqa: E402
import issue1902_format_fits as F  # noqa: E402
import issue1902_format_stage as S  # noqa: E402
from issue1902_watch_backend import bounded_command  # noqa: E402


def test_recovery_stages_only_base_sft_on_original_olmo_cohort(tmp_path, monkeypatch):
    assert set(C.MODELS) == {"qwen_B", "qwen_S", "olmo_B", "olmo_S"}
    assert set(C.DEFERRED_FORMAT_MODELS) == {"olmo_D", "olmo_R"}
    ids = np.array([str(i) for i in range(16391)])
    folds = np.arange(len(ids)) % 6
    reference = tmp_path / "input.npz"
    np.savez(reference, row_ids=ids, fold_of=folds, y=np.zeros((len(ids), 1)))
    requested = []

    def fetch(root, path, revision):
        assert revision == C.INPUT_REV
        assert Path(path).name in {"B.npz", "S.npz"}
        requested.append(Path(path).name)
        return reference

    def rows(root, relative, revision):
        if relative.endswith("corpus_single.jsonl"):
            assert revision == C.O0
            return [dict(id=cid, query="question") for cid in ids.tolist()]
        name = Path(relative).name
        assert name.startswith(("B.", "B_", "S.", "S_"))
        seed = 42 if "seed" not in name else int(name.split("seed")[1].split(".")[0])
        assert revision == (C.O0 if seed == 42 else C.O5)
        return [
            dict(
                id=cid,
                seed=seed,
                text="answer",
                finish_reason="stop",
                n_tokens=1,
                repetition_flag=False,
            )
            for cid in ids.tolist()
        ]

    monkeypatch.setattr(C, "fetch", create_autospec(C.fetch, side_effect=fetch))
    monkeypatch.setattr(S, "olmo_jsonl", create_autospec(S.olmo_jsonl, side_effect=rows))
    result = S.stage_olmo(tmp_path)
    assert result["ids"] == ids.tolist()
    assert result["fold_of"] == folds.tolist()
    assert result["n_folds"] == 6
    assert set(requested) == {"B.npz", "S.npz"}
    assert {p.name for p in (tmp_path / "banks").iterdir()} == {"olmo_B", "olmo_S"}


def test_archived_raw_hash_mismatch_still_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from huggingface_hub import HfApi

    raw = tmp_path / "R.shard02.jsonl"
    raw.write_text('{"id": "one", "text": "changed"}\n')
    manifest = tmp_path / "R.manifest.json"
    manifest.write_text(json.dumps(dict(shards=[dict(name=raw.name, n_lines=1, sha256="wrong")])))
    relative = "example/R.jsonl"

    def fetch(root, path, revision):
        return manifest if path.endswith("manifest.json") else raw

    monkeypatch.setattr(C, "fetch", create_autospec(C.fetch, side_effect=fetch))
    monkeypatch.setattr(
        HfApi,
        "get_paths_info",
        create_autospec(
            HfApi.get_paths_info, return_value=[SimpleNamespace(path="example/R.manifest.json")]
        ),
    )
    with pytest.raises(AssertionError, match=r"Raw shard hash mismatch.*R\.shard02"):
        S.olmo_jsonl(tmp_path, relative, "pinned-revision")


def test_observation_timeout_reaps_owned_process(tmp_path):
    pidfile = tmp_path / "pid"
    command = [
        sys.executable,
        "-c",
        "import os,time,pathlib,sys; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(20)",
        str(pidfile),
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        bounded_command(command, cwd=tmp_path, timeout=0.5)
    assert not Path("/proc", pidfile.read_text()).exists()


def test_completed_gpu_artifact_advances_despite_dead_backend(tmp_path, monkeypatch):
    import issue1902_format_follow as follow

    state_path = tmp_path / "state.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "follow",
            "--source-sha",
            "source",
            "--branch",
            "branch",
            "--repo-root",
            str(ROOT),
            "--state",
            str(state_path),
            "--gpu-handle",
            str(tmp_path / "dead-gpu.json"),
        ],
    )

    def artifact(name, expected):
        assert expected == "source"
        return dict(source_sha="source", input_revision="verified-input")

    def forbidden_probe(*args, **kwargs):
        raise AssertionError("Verified completion must take precedence over backend death")

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, json.dumps(dict(ok=True, handle_sidecar_path="cpu.json")) + "\n", ""
        )

    monkeypatch.setattr(
        follow, "current_artifact", create_autospec(follow.current_artifact, side_effect=artifact)
    )
    monkeypatch.setattr(
        follow, "observe", create_autospec(follow.observe, side_effect=forbidden_probe)
    )
    monkeypatch.setattr(subprocess, "run", create_autospec(subprocess.run, side_effect=run))
    follow.main()
    assert json.loads(state_path.read_text())["status"] == "complete"
    assert sum("launch" in c for c in commands) == 1


def test_three_probe_timeouts_write_explicit_unknown_state(tmp_path, monkeypatch):
    import issue1902_format_follow as follow

    state_path = tmp_path / "state.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "follow",
            "--source-sha",
            "source",
            "--branch",
            "branch",
            "--repo-root",
            str(ROOT),
            "--state",
            str(state_path),
            "--gpu-handle",
            str(tmp_path / "gpu.json"),
        ],
    )
    monkeypatch.setattr(
        follow, "current_artifact", create_autospec(follow.current_artifact, return_value=None)
    )
    monkeypatch.setattr(
        follow,
        "observe",
        create_autospec(follow.observe, side_effect=subprocess.TimeoutExpired("probe", 180)),
    )
    monkeypatch.setattr(follow.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        create_autospec(subprocess.run, return_value=subprocess.CompletedProcess("post-marker", 0)),
    )
    with pytest.raises(RuntimeError, match="Three backend probes failed"):
        follow.main()
    state = json.loads(state_path.read_text())
    assert state["status"] == "backend_observation_error"
    assert state["consecutive_observation_failures"] == 3
    assert state["checked_at"] > 0


def local_receipts(paths, root, fingerprint):
    """Replace network persistence with hash-checked local receipts in these CPU tests."""
    for path in paths:
        path = Path(path)
        C.write_json(
            path.with_suffix(path.suffix + ".done.json"),
            dict(fingerprint=fingerprint, sha256=C.sha(path)),
        )
    return "test-immutable-revision"


def test_raw_byte_shards_preserve_text_and_detect_corruption(tmp_path):
    rows = [dict(id=str(i), answer="é" * 400_000) for i in range(12)]
    path = tmp_path / "chunk_00000.json"
    files = C.write_raw(path, rows)
    assert len(files) > 2 and max(p.stat().st_size for p in files) < 8_000_000
    assert C.read_raw(path) == rows
    files[1].write_text("[]")
    with pytest.raises(AssertionError):
        C.read_raw(path)


def test_chunk_ownership_complete_and_pilot_unique():
    for width in (1, 2, 4, 8):
        for model in C.MODELS:
            expected = list(range(0, 7999, C.CHUNK))
            owned = [
                off
                for slot in range(width)
                for off in C.owned_offsets(model, 7999, slot, width, False)
            ]
            assert sorted(owned) == expected
            pilot = [
                off
                for slot in range(width)
                for off in C.owned_offsets(model, 7999, slot, width, True)
            ]
            assert pilot == [0]


def test_stage_cannot_resume_from_early_manifest_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CHUNK", 1)

    def fake_stage(root, family):
        ids = [str(i) for i in range(10)]
        for model in C.MODELS:
            if not model.startswith(family + "_"):
                continue
            for bank in C.banks(model):
                if bank["fresh"]:
                    continue
                for i, cid in enumerate(ids):
                    C.write_raw(
                        root / "banks" / model / bank["name"] / f"chunk_{i:05d}.json",
                        [dict(id=cid, draw=d, answer="answer") for d in range(5)],
                    )
        return dict(ids=ids, questions={i: "q" for i in ids}, fold_of=[0] * 10)

    monkeypatch.setattr(S, "stage_qwen", lambda root: fake_stage(root, "qwen"))
    monkeypatch.setattr(S, "stage_olmo", lambda root: fake_stage(root, "olmo"))
    calls = []

    def interrupted(paths, root, fp):
        calls.append(paths)
        if len(calls) == 2:
            raise RuntimeError("simulated transport failure")
        return local_receipts(paths, root, fp)

    monkeypatch.setattr(C, "upload_many", interrupted)
    with pytest.raises(RuntimeError, match="simulated transport failure"):
        S.stage(tmp_path)
    assert (tmp_path / "manifest.json.done.json").exists()
    assert not (tmp_path / "stage_complete.json").exists()
    monkeypatch.setattr(C, "upload_many", local_receipts)
    S.stage(tmp_path)
    assert C.complete(tmp_path / "stage_complete.json", C.fingerprint(tmp_path))
    assert not list(tmp_path.rglob("*.done.json.done.json"))


@pytest.mark.parametrize("family", ["qwen", "olmo"])
def test_fits_hold_columns_fixed_and_preserve_scoring(tmp_path, monkeypatch, family):
    monkeypatch.setattr(C, "upload_many", local_receipts)
    rng = np.random.default_rng(2034)
    n, d = 120, 4
    ids = np.array([str(i) for i in range(n)])
    nfold = 5 if family == "qwen" else 6
    fold = np.arange(n) % nfold
    latent = rng.normal(size=(n, d))
    manifest = {family: dict(ids=ids.tolist(), fold_of=fold.tolist(), n_folds=nfold)}
    C.write_json(tmp_path / "manifest.json", manifest)
    declared = dict(contexts={}, targets={}, files={})
    actual_targets = {}
    for model in [m for m in C.MODELS if m.startswith(family + "_")]:
        for form in ("plain", "chat"):
            path = tmp_path / f"x_{model}_{form}.npz"
            x = latent + 0.4 * rng.normal(size=(n, d))
            np.savez(path, ids=ids, x=x, fold_of=fold)
            declared["contexts"][model + "/" + form] = path.name
            declared["files"][path.name] = C.sha(path)
        for bi, bank in enumerate(C.banks(model)):
            for form in ("plain", "chat"):
                target = model + "/" + bank["name"] + "/" + form
                path = tmp_path / f"y_{model}_{bank['name']}_{form}.npz"
                y = latent @ rng.normal(size=(d, d)) + 0.3 * rng.normal(size=(n, d)) + bi * 30
                np.savez(path, ids=ids, y=y, valid=np.ones(n, dtype=bool))
                actual_targets[target] = y
                declared["targets"][target] = path.name
                declared["files"][path.name] = C.sha(path)
    C.write_json(tmp_path / "analysis_inputs.json", declared)
    F.fit_family(tmp_path, declared, family)
    result = json.loads((tmp_path / "fits" / family / "summary.json").read_text())
    assert {c["kind"] for c in result["contrasts"]} == {
        "capture_format",
        "generation_format",
        "context_information",
    }
    target = family + "_S/plain/chat"
    context = family + "_B/chat"
    cell = result["cells"][F.key(context, target)]
    expected = []
    for f, record in enumerate(cell["fold_metrics"]):
        assert record["target_sha256"] == C.sha(tmp_path / declared["targets"][target])
        ev = fold == f
        y = actual_targets[target]
        assert record["retrieval_pool"] == int(ev.sum())
        stem = tmp_path / "fits" / family / "folds" / f"{F.key(context, target)}_f{f}.npz"
        with np.load(stem) as p:
            assert np.allclose(p["tot"], ((y[ev] - y[~ev].mean(0)) ** 2).sum(1))
            expected.append(1 - p["res"].sum() / ((y[ev] - y[ev].mean(0)) ** 2).sum())
    if family == "qwen":
        assert cell["parent_r2"] == pytest.approx(np.mean(expected))
    for c in result["contrasts"]:
        if c["kind"] == "context_information":
            assert result["cells"][c["first"]]["target"] == result["cells"][c["second"]]["target"]
            if c["first"] == c["second"]:
                assert c["paired_row_ci95"] == [0.0, 0.0]

    def no_refit(*args, **kwargs):
        raise AssertionError("Completed units must be reused")

    monkeypatch.setattr(F, "SharedPrimalRidge", no_refit)
    monkeypatch.setattr(F, "SharedEighRidge", no_refit)
    F.fit_family(tmp_path, declared, family)
    # Changing the inventory invalidates old fit receipts.
    C.write_json(tmp_path / "analysis_inputs.json", dict(declared, changed=True))
    with pytest.raises(AssertionError, match="Stale checkpoint"):
        F.fit_family(tmp_path, declared, family)
