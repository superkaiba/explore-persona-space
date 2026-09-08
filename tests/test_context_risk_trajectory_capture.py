"""CPU tests exercise the dispatched gather and exact persisted-vector contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts import context_risk_trajectory_capture as capture
from scripts import context_risk_trajectory_prepare as prepare
from tests.test_context_risk_trajectory_prepare import fixture as producer_fixture
from tests.test_context_risk_trajectory_prepare import publish as publish_producer_fixture


class Decoder(torch.nn.Module):
    def __init__(self, leak=False):
        super().__init__()
        self.leak = leak

    def forward(self, hidden):
        if self.leak:
            return hidden + hidden.sum(dim=1, keepdim=True)
        return hidden + 1


class CausalModel(torch.nn.Module):
    """Small deterministic causal decoder with the actual HF hook/forward interface."""

    def __init__(self, hidden_dim=4, leak=False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(256, hidden_dim)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(
                torch.arange(256 * hidden_dim).reshape(256, hidden_dim).remainder(97) / 128
            )
        self.model.layers = torch.nn.ModuleList([Decoder(leak), Decoder(), Decoder()])
        self.config = SimpleNamespace(text_config=SimpleNamespace(pad_token_id=0, eos_token_id=1))
        self.calls = []
        self.raise_forward = False
        self.return_cache = False
        self.eval()

    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache, logits_to_keep):
        assert output_hidden_states is False
        assert use_cache is False
        assert logits_to_keep == 1
        assert not torch.is_grad_enabled()
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        if self.raise_forward:
            raise RuntimeError("deliberate actual forward failure")
        hidden = (self.model.embed_tokens(input_ids) * attention_mask[..., None]).cumsum(dim=1)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(past_key_values=() if self.return_cache else None)


@pytest.fixture
def executable():
    cfg = OmegaConf.load(capture.REPOSITORY / "configs/eval/context_risk_trajectory_capture.yaml")
    return capture.recipe(cfg)


@pytest.fixture
def producer_case(tmp_path, monkeypatch):
    # Reuse the producer's actual local-tokenizer/evidence builder.
    return producer_fixture.__wrapped__(tmp_path, monkeypatch)


def identity_map(dim=4):
    return {
        "weight": np.eye(dim),
        "x_mean": np.zeros(dim),
        "x_scale": np.ones(dim),
        "y_mean": np.zeros(dim),
    }


def prepared_fixture():
    streams, manifest, checkpoints, observations = {}, {"streams": {}}, [], []
    stage_names = (
        "pre_action_01",
        "pre_action_02",
        "within_pre",
        "within_32",
        "within_128",
        "within_512",
        "within_end",
    )
    for n, kind in (
        (2, "canonical_initial"),
        (4, "canonical_initial"),
        (8, "completion_exception"),
        (12, "trajectory_final"),
    ):
        ids = np.arange(1, n + 1, dtype=np.uint32)
        # Separate short canonical prefixes reflect distinct real contexts.
        ids[0] = n
        id_hash = capture.digest(ids.tolist())
        prefix = {
            "canonical_initial": "initial",
            "completion_exception": "exception",
            "trajectory_final": "trajectory",
        }[kind]
        key = prefix + "_" + id_hash
        streams[key] = ids
        manifest["streams"][key] = {"stream_kind": kind, "length": n, "token_ids_sha256": id_hash}
        lengths = [n] if kind == "canonical_initial" else list(range(1, n + 1))
        for length in lengths:
            ck = capture.digest(ids[:length].tolist())
            point = {
                "checkpoint_key": ck,
                "stream_key": key,
                "length": length,
                "position": length - 1,
            }
            checkpoints.append(point)
            for stage in stage_names:
                observations.append(
                    {
                        "checkpoint_key": ck,
                        "stage": stage,
                        "answer_ended": length == n,
                        "positive": 0,
                    }
                )
    return manifest, streams, checkpoints, observations


def test_real_gather_ragged_order_and_exact_reference():
    model = CausalModel()
    rows = [np.array([3, 4, 5], dtype=np.uint32), np.array([7], dtype=np.uint32)]
    actual, stats = capture.capture_batch(
        model, rows, [[2, 0], [0]], layer=1, hidden_dim=4, same_forward=True
    )
    for row, offsets, values in zip(rows, [[2, 0], [0]], actual, strict=True):
        expected = (
            model.model.embed_tokens(torch.tensor(row.astype(np.int64)))
            .detach()
            .numpy()
            .cumsum(axis=0)
            + 2
        )
        np.testing.assert_array_equal(values, expected[offsets])
    assert stats["same_forward_exact"] is True
    assert stats["selected_vectors"] == 3
    assert model.calls[0][1].tolist() == [[1, 1, 1], [1, 0, 0]]


@pytest.mark.parametrize("offset", [-1, 3, True, 1.0])
def test_gather_rejects_invalid_position(offset):
    with pytest.raises(ValueError, match="offset"):
        capture.capture_batch(
            CausalModel(), [np.array([1, 2, 3])], [[offset]], layer=1, hidden_dim=4
        )


def test_forward_failure_removes_actual_registered_hook():
    model = CausalModel()
    model.raise_forward = True
    with pytest.raises(RuntimeError, match="actual forward"):
        capture.capture_batch(model, [np.array([1])], [[0]], layer=1, hidden_dim=4)
    assert not model.model.layers[1]._forward_hooks


def test_capture_rejects_training_and_retained_cache():
    model = CausalModel()
    model.train()
    with pytest.raises(ValueError, match="eval"):
        capture.capture_batch(model, [np.array([1])], [[0]], layer=1, hidden_dim=4)
    model.eval()
    model.return_cache = True
    with pytest.raises(ValueError, match="cache"):
        capture.capture_batch(model, [np.array([1])], [[0]], layer=1, hidden_dim=4)


def test_map_amplification_can_fail_when_raw_passes(executable):
    actual, expected = np.array([1.0, 1e-4, 0, 0]), np.array([1.0, 0, 0, 0])
    arrays = identity_map()
    arrays["weight"][1, 1] = 10000
    result = capture.compare_vectors(actual, expected, arrays, executable["smoke"])
    assert result["raw"]["relative_l2"] < 0.01
    assert result["raw"]["cosine"] > 0.9999
    assert result["passed"] is False


def test_smoke_runs_real_causal_and_leaking_forwards(executable):
    fixture = prepared_fixture()
    causal = CausalModel()
    result = capture.smoke_model(
        causal, *fixture, executable, identity_map(), layer=1, hidden_dim=4
    )
    assert result["verification_passed"] is True
    assert result["suffix_mutation"]["exact"] is True
    assert {"longest_singleton", "representative_singleton", "production_short_batch"} <= {
        m["purpose"] for m in result["measurements"]
    }
    assert len(causal.calls) == len(result["measurements"])
    failed = capture.smoke_model(
        CausalModel(leak=True), *fixture, executable, identity_map(), layer=1, hidden_dim=4
    )
    assert failed["verification_passed"] is False
    assert failed["suffix_mutation"]["exact"] is False


def test_fixture_selection_does_not_consult_outcomes(executable):
    manifest, streams, points, observations = prepared_fixture()
    before = capture.smoke_fixtures(
        manifest, streams, points, observations, executable["capture"], executable["smoke"]
    )
    for row in observations:
        row["positive"] = 1
        row["eventual_success"] = True
    after = capture.smoke_fixtures(
        manifest, streams, points, observations, executable["capture"], executable["smoke"]
    )
    assert before == after
    long_point = before["checkpoints"][before["checkpoint_roles"]["long_checkpoint"]]
    assert long_point["length"] < len(streams[before["longest_stream"]])


def test_threshold_drift_rejected_before_runtime(executable):
    executable["smoke"]["raw_relative_l2_max"] = 0.1
    with pytest.raises(ValueError, match="thresholds"):
        capture.recipe(OmegaConf.create(executable))


def test_exact_checkpoint_prefix_required():
    _manifest, streams, points, _observations = prepared_fixture()
    points[0]["checkpoint_key"] = "a" * 64
    with pytest.raises(ValueError, match="exact token prefix"):
        capture.group_checkpoints(streams, points)


def test_real_capture_resume_and_corruption(tmp_path, executable):
    _manifest, streams, points, _observations = prepared_fixture()
    binding = capture.make_binding("a" * 64, executable, {"fixture": "b" * 64})
    model = CausalModel()
    grouped = capture.collect_streams(
        model, tmp_path, streams, points, binding, layer=1, hidden_dim=4
    )
    calls = len(model.calls)
    capture.collect_streams(model, tmp_path, streams, points, binding, layer=1, hidden_dim=4)
    assert len(model.calls) == calls
    key = min(streams)
    expected = capture._chunk_fields(key, streams[key], grouped[key], binding)
    done, values = capture.read_chunk(tmp_path, key, expected, hidden_dim=4)
    assert values.dtype == np.float16
    changed = dict(binding, fingerprint="c" * 64)
    with pytest.raises(ValueError, match="different input"):
        capture.collect_streams(model, tmp_path, streams, points, changed, layer=1, hidden_dim=4)
    (tmp_path / done["payload_path"]).write_bytes(b"corrupted payload")
    with pytest.raises(ValueError, match="payload bytes"):
        capture.read_chunk(tmp_path, key, expected, hidden_dim=4)


def test_public_final_verifier_detects_vector_join_corruption(
    tmp_path, executable, monkeypatch, producer_case
):
    evidence = producer_case["evidence"]
    bank = prepare.build_bank(evidence["contexts"], evidence["calls"], producer_case["tokenizer"])
    manifest, streams, points, observations = (
        {"streams": bank["stream_info"]},
        bank["streams"],
        bank["checkpoints"],
        bank["observations"],
    )
    assert {key.split("_", 1)[0] for key in streams} == {"initial", "trajectory", "exception"}
    tmp_path = tmp_path / "capture"
    tmp_path.mkdir()
    sources = {"fixture": "b" * 64}
    # The fixture substitutes provenance discovery only; real chunks, finalizer,
    # public verifier, array/hash reads and checkpoint joins execute below.
    monkeypatch.setattr(capture, "source_hashes", lambda: sources)
    binding = capture.make_binding("a" * 64, executable, sources)
    capture._write_json(tmp_path / "capture_binding.json", binding)
    review = {"verdict": "PASS", "unresolved_findings": [], "sources_sha256": sources}
    capture._write_json(tmp_path / "independent_code_review.json", review)
    smoke = capture.smoke_model(
        CausalModel(),
        manifest,
        streams,
        points,
        observations,
        executable,
        identity_map(),
        layer=1,
        hidden_dim=4,
    )
    replay = {"passed": True, "n_requests": 2153, "prepared_manifest_sha256": "a" * 64}
    capture._write_json(tmp_path / "tokenizer_replay_smoke.json", replay)
    smoke.update(
        {
            "binding": binding,
            "finished_utc": "fixture",
            "review_sha256": capture.sha256(tmp_path / "independent_code_review.json"),
            "tokenizer_replay_sha256": capture.sha256(tmp_path / "tokenizer_replay_smoke.json"),
        }
    )
    capture._write_json(tmp_path / "smoke.json", smoke)
    capture._write_json(
        tmp_path / "tokenizer_replay_capture.json",
        {"passed": True, "n_requests": 2153, "prepared_manifest_sha256": "a" * 64},
    )
    grouped = capture.collect_streams(
        CausalModel(hidden_dim=5120), tmp_path, streams, points, binding, layer=1
    )
    index = capture.finalize_capture(
        tmp_path, streams, grouped, binding, capture.sha256(tmp_path / "smoke.json")
    )
    actual_index, values = capture.verify_capture(tmp_path, "a" * 64)
    assert actual_index == index
    assert isinstance(values, np.memmap) and not values.flags.writeable
    assert capture.verify_capture_geometry(tmp_path, index, streams, points)["verification_passed"]
    done_name = capture._stream_names(min(streams))[1]
    done_path = tmp_path / done_name
    original_done = done_path.read_bytes()
    original_hash = index["chunks_sha256"][done_name]
    for field in ("positions", "token_ids_sha256"):
        changed = json.loads(original_done)
        if field == "positions":
            changed[field][0] += 1
        else:
            changed[field] = "e" * 64
        capture._write_json(done_path, changed)
        index["chunks_sha256"][done_name] = capture.sha256(done_path)
        capture._write_json(tmp_path / "index.json", index)
        with pytest.raises(ValueError, match="actual prepared geometry"):
            capture.verify_capture_geometry(tmp_path, index, streams, points)
        done_path.write_bytes(original_done)
        index["chunks_sha256"][done_name] = original_hash
        capture._write_json(tmp_path / "index.json", index)
    with pytest.raises(ValueError, match="stale"):
        capture.verify_capture(tmp_path, "d" * 64)
    altered = np.asarray(values).copy()
    altered[0, 0] += 1
    np.save(tmp_path / "vectors.npy", altered)
    index["vectors_sha256"] = capture.sha256(tmp_path / "vectors.npy")
    capture._write_json(tmp_path / "index.json", index)
    with pytest.raises(ValueError, match="checkpoint join"):
        capture.verify_capture(tmp_path, "a" * 64)


def test_review_and_unsafe_paths_fail(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(
            {"verdict": "PASS", "unresolved_findings": [], "sources_sha256": {"old": "hash"}}
        )
    )
    with pytest.raises(ValueError, match="independent capture review"):
        capture.validate_review(path, {"new": "hash"})
    for name in ("../outside", "/outside", ".", ""):
        with pytest.raises(ValueError, match="Unsafe"):
            capture._safe(tmp_path, name)


@pytest.mark.parametrize("corruption", ["metric", "coverage", "mutation", "future_mask", "shape"])
def test_saved_smoke_boolean_cannot_hide_failed_evidence(executable, corruption):
    result = capture.smoke_model(
        CausalModel(), *prepared_fixture(), executable, identity_map(), layer=1, hidden_dim=4
    )
    capture._validate_smoke_results(result, executable["smoke"])
    if corruption == "metric":
        next(iter(result["comparisons"].values()))["mapped"]["relative_l2"] = 0.1
    elif corruption == "coverage":
        result["comparisons"].pop(next(iter(result["comparisons"])))
    elif corruption == "mutation":
        result["suffix_mutation"]["exact"] = False
    elif corruption == "future_mask":
        result["future_mask_mutation"]["exact"] = False
    else:
        result["measurements"] = [
            m for m in result["measurements"] if m["purpose"] != "longest_singleton"
        ]
    assert result["verification_passed"] is True
    with pytest.raises(ValueError):
        capture._validate_smoke_results(result, executable["smoke"])


def test_mask_mutation_keeps_ids_shape_and_prefix_fixed():
    model = CausalModel()
    ids = np.array([1, 2, 3, 4], dtype=np.uint32)
    full, _ = capture.capture_batch(model, [ids], [[1]], layer=1, hidden_dim=4)
    masked, stats = capture.capture_batch(
        model, [ids], [[1]], layer=1, hidden_dim=4, valid_lengths=[2]
    )
    np.testing.assert_array_equal(full[0], masked[0])
    assert torch.equal(model.calls[0][0], model.calls[1][0])
    assert model.calls[0][1].tolist() == [[1, 1, 1, 1]]
    assert model.calls[1][1].tolist() == [[1, 1, 0, 0]]
    assert stats["lengths"] == [4] and stats["valid_lengths"] == [2]


def test_output_must_be_disjoint_and_cannot_adopt_orphans(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for output in (prepared, prepared / "capture", tmp_path):
        with pytest.raises(ValueError, match="disjoint"):
            capture.guard_output(output, prepared, adopt=True)
    output = tmp_path / "capture"
    output.mkdir()
    (output / "orphan.npz").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="unbound populated"):
        capture.guard_output(output, prepared, adopt=True)


@pytest.mark.parametrize(
    "changed",
    ["manifest.json", "streams.npz", "code_review.json", "external_review.json", "map.npz"],
)
def test_consumed_payload_and_control_changes_are_detected(tmp_path, changed):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    extras = [tmp_path / "external_review.json", tmp_path / "map.npz"]
    for name in ("manifest.json", *prepare.DATA_FILES):
        (prepared / name).write_bytes(b"original fixture bytes")
    for path in extras:
        path.write_bytes(b"original external bytes")
    before = capture.prepared_snapshot(prepared, extras)
    capture.assert_snapshot(before)
    path = (
        tmp_path / changed if changed in {"external_review.json", "map.npz"} else prepared / changed
    )
    path.write_bytes(b"changed after snapshot")
    with pytest.raises(ValueError, match="Consumed input or control bytes changed"):
        capture.assert_snapshot(before)


@pytest.mark.parametrize(
    "key", ["a" * 64, "other_" + "a" * 64, "trajectory_../escape", "initial_" + "A" * 64]
)
def test_stream_writer_rejects_unrecognized_names(key):
    with pytest.raises(ValueError, match="stream key"):
        capture._stream_names(key)


@pytest.mark.parametrize("changed", ["manifest.json", "streams.npz"])
def test_run_detects_actual_prepared_load_to_snapshot_race(
    producer_case, executable, monkeypatch, tmp_path, changed
):
    prepared = publish_producer_fixture(producer_case)
    real_loader = prepare.load_prepared

    def raced_load(path):
        loaded = real_loader(path)
        with (prepared / changed).open("ab") as handle:
            handle.write(b"\nchanged during loading")
        return loaded

    monkeypatch.setattr(prepare, "load_prepared", raced_load)
    cfg = OmegaConf.create(
        {
            **executable,
            "operation": "preflight",
            "prepared_root": str(prepared),
            "output_dir": str(tmp_path / "capture"),
        }
    )
    with pytest.raises(ValueError, match="Consumed input or control bytes changed"):
        capture.run(cfg)
