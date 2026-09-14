"""Regression tests for target checkpoint/format identity and frozen-map evaluation."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import issue1902_format_common as C  # noqa: E402
import issue1902_format_plot_fits as P  # noqa: E402
from issue1902_lasttoken_transfer import SharedPrimalRidge  # noqa: E402


def receipts(paths, root, fp):
    assert paths
    for path in paths:
        C.write_json(
            path.with_suffix(path.suffix + ".done.json"), dict(fingerprint=fp, sha256=C.sha(path))
        )
    return "test-revision"


def test_only_requested_settings_and_pairs():
    assert len(P.fit_pairs()) == 14 and len(P.transfer_pairs()) == 6
    assert {m for m in C.MODELS} == {"olmo_B", "olmo_S", "olmo_D", "olmo_R"}
    assert sum(b["fresh"] for m in C.MODELS for b in C.banks(m)) == 5
    for m in C.MODELS:
        for b in C.banks(m):
            assert C.capture_forms(b) == (b["render"],)
            assert b["cap"] == 1024


def test_actual_fit_and_transfer_use_identical_target_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "upload_many", receipts)
    n, d = 72, 5
    rng = np.random.default_rng(74)
    latent = rng.normal(size=(n, d))
    ids, folds = np.arange(n).astype(str), np.arange(n) % 6
    xs, ys = {}, {}
    declared = dict(targets={}, files={})
    C.write_json(tmp_path / "manifest.json", dict(test_fixture=True))
    for stage in P.STAGES:
        for form in P.FORMS:
            xs[stage, form] = latent + rng.normal(size=(n, d)) * 0.2
            ys[stage, form] = latent @ rng.normal(size=(d, d)) + rng.normal(size=d) * 10
            key = P.target_key(stage, form)
            path = tmp_path / f"target_{stage}_{form}.npz"
            np.savez(path, y=ys[stage, form])
            declared["targets"][key] = path.name
            declared["files"][path.name] = C.sha(path)
    C.write_json(tmp_path / "analysis_inputs.json", declared)
    C.write_json(tmp_path / "fits/olmo/cohort.json", dict(realized_rows=n))
    P.fit_all(tmp_path, declared, ids, folds, xs, ys)
    result = P.summarize(tmp_path, ids, folds)
    assert (
        len(result["panel_a"]) == 8 and len(result["panel_b"]) == 6 and len(result["panel_c"]) == 18
    )
    for form in P.FORMS:
        for target in "SDR":
            for fold in range(6):
                tr, ev = folds != fold, folds == fold
                expected_total = np.square(ys[target, form][ev] - ys[target, form][tr].mean(0)).sum(
                    1
                )
                for source in ("B", target):
                    path = P.stem(tmp_path, source, target, form, fold)
                    with np.load(path.with_suffix(".npz")) as p:
                        np.testing.assert_allclose(p["tot"], expected_total)
                    meta = json.loads(path.with_suffix(".json").read_text())
                    assert (
                        meta["target_sha256"]
                        == declared["files"][declared["targets"][P.target_key(target, form)]]
                    )
        tr, ev = folds != 0, folds == 0
        ridge = SharedPrimalRidge(xs["B", form][tr])
        w, mu, _ = ridge.fit(ys["B", form][tr])
        expected = ridge.standardize(xs["S", form][ev]) @ w + mu
        path = P.stem(tmp_path, "B", "S", form, 0, True)
        with np.load(path.with_suffix(".npz")) as p:
            np.testing.assert_allclose(p["pred_direct"], expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(
                p["res_direct"], np.square(expected - ys["S", form][ev]).sum(1)
            )

    def forbidden(*args, **kwargs):
        raise AssertionError("A completed refit must not run again")

    monkeypatch.setattr(P, "SharedPrimalRidge", forbidden)
    P.fit_all(tmp_path, declared, ids, folds, xs, ys)
    # Render only this explicitly synthetic test fixture in pytest's temporary directory.
    from issue1902_format_plot import plot

    assert len(plot(tmp_path, result)) == 12
    C.write_json(tmp_path / "cpu_pilot.json", dict(status="requires_review"))
    with pytest.raises(AssertionError, match="Prior CPU pilot requires review"):
        P.fit_all(tmp_path, declared, ids, folds, xs, ys)


def test_gpu_pilot_uses_absolute_deadline_and_never_publishes_false_pass(tmp_path, monkeypatch):
    import time

    import issue1902_format_job as J

    monkeypatch.setattr(C, "upload_many", receipts)
    C.write_json(tmp_path / "manifest.json", dict(olmo=dict(ids=list(range(16391)))))
    now = time.time()
    clock = dict(time=now - 7200, pilot_started=now - 60, gpus=2, deadline=now + 60)
    C.write_json(tmp_path / "job_started.json", clock)
    for model in C.MODELS:
        for form in P.FORMS:
            C.write_json(
                tmp_path / "contexts" / model / form / "chunk_00000.timing.json", dict(seconds=2)
            )
        for bank in C.banks(model):
            if bank["fresh"]:
                C.write_json(
                    tmp_path / "banks" / model / bank["name"] / "chunk_00000.timing.json",
                    dict(seconds=10),
                )
            for form in C.capture_forms(bank):
                C.write_json(
                    tmp_path / "captures" / model / bank["name"] / form / "chunk_00000.timing.json",
                    dict(seconds=5),
                )
    with pytest.raises(AssertionError, match="Measured pilot exceeds"):
        J.pilot_report(tmp_path)
    assert not (tmp_path / "pilot_complete.json").exists()
    assert json.loads((tmp_path / "pilot_report.json").read_text())["status"] == "requires_review"
    C.write_json(tmp_path / "job_started.json", dict(clock, deadline=now + 48 * 3600))
    assert J.pilot_report(tmp_path)["status"] == "pass"


def test_capture_pilot_resume_does_not_upload_empty_list(tmp_path, monkeypatch):
    import issue1902_format_gpu as G
    import transformers

    C.write_json(tmp_path / "manifest.json", dict(olmo=dict(ids=["one"], questions={"one": "q"})))
    fake = SimpleNamespace(model=SimpleNamespace(layers=[None] * 32))
    fake.to = lambda device: fake
    fake.eval = lambda: fake
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: fake)
    monkeypatch.setattr(G, "tokenizer_for", lambda model: (None, None))
    monkeypatch.setattr(G, "capture_anchor", lambda *a: None)
    monkeypatch.setattr(C, "complete", lambda *a: True)
    monkeypatch.setattr(C, "read_raw", lambda path: [dict(id="one", draw=d) for d in range(5)])

    def forbidden(*a, **k):
        raise AssertionError("Completed pilot must not capture or upload anything")

    monkeypatch.setattr(C, "upload_many", forbidden)
    monkeypatch.setattr(G, "forward", forbidden)
    G.capture(tmp_path, "olmo_B", first_chunk=True)
