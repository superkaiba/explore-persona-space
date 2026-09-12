"""Regression checks for raw preservation, shard ownership, and scientific contrasts."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import issue1902_format_common as C  # noqa: E402
import issue1902_format_fits as F  # noqa: E402
import issue1902_format_stage as S  # noqa: E402


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
