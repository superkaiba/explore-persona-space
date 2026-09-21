"""Numerical location, bounded continuation, and immutable checkpoint contracts."""

import json
import threading
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from scripts import story_persona_crossmodel_capture as capture


def test_loading_sets_round_trip_through_the_real_manifest(tmp_path):
    loading = {
        "missing_keys": set(),
        "unexpected_keys": {"model.layers.61.z.weight", "model.layers.61.a.weight"},
        "mismatched_keys": set(),
        "error_msgs": [],
    }
    spec = {"runtime": {"loading_info": capture.validated_loading_info(loading)}}
    fingerprint = capture.digest(spec)
    path = tmp_path / "manifest.json"
    capture.write_json(path, {"spec": spec, "fingerprint": fingerprint})
    assert capture.read_manifest(path)["fingerprint"] == fingerprint
    assert spec["runtime"]["loading_info"]["unexpected_keys"] == [
        "model.layers.61.a.weight",
        "model.layers.61.z.weight",
    ]
    loading["unexpected_keys"] = list(reversed(sorted(loading["unexpected_keys"])))
    assert (
        capture.digest({"runtime": {"loading_info": capture.validated_loading_info(loading)}})
        == fingerprint
    )
    assert isinstance(loading["missing_keys"], set)  # Does not mutate the loader report.


@pytest.mark.parametrize(
    "key,value",
    [
        ("missing_keys", {"model.layers.0.weight"}),
        ("unexpected_keys", {"model.layers.60.extra"}),
        ("mismatched_keys", {("model.weight", (2, 3), (3, 2))}),
        ("error_msgs", ["conversion failed"]),
        ("unexpected_keys", [object()]),
    ],
)
def test_loading_normalization_does_not_hide_checkpoint_errors(key, value):
    loading = dict(missing_keys=set(), unexpected_keys=set(), mismatched_keys=set(), error_msgs=[])
    loading[key] = value
    with pytest.raises(RuntimeError, match="unexplained checkpoint"):
        capture.validated_loading_info(loading)


def test_loading_normalization_rejects_unknown_schema():
    with pytest.raises(RuntimeError, match="schema"):
        capture.validated_loading_info({"unexpected_keys": set()})


class TinyDecoder(torch.nn.Module):
    def __init__(self, *, batch_drift=False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(32, 4, dtype=torch.bfloat16)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(torch.arange(128).reshape(32, 4))
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(8)])
        self.calls = []
        self.batch_drift = batch_drift

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids,
        attention_mask,
        use_cache,
        output_hidden_states,
        return_dict,
        logits_to_keep=1,
    ):
        assert not use_cache and return_dict and logits_to_keep == 1
        self.calls.append((tuple(input_ids.shape), bool(attention_mask.all())))
        h = (self.model.embed_tokens(input_ids) * attention_mask[..., None]).cumsum(1)
        if self.batch_drift and input_ids.shape[0] > 1:
            h = h + 50
        states = [h]
        for layer in self.model.layers:
            h = layer(h)
            states.append(h)
        states[-1] = h * 2  # Deliberately distinct final norm, outside the raw final hook.
        return SimpleNamespace(hidden_states=states if output_hidden_states else None)


def config(arm="deepseek"):
    return OmegaConf.create(
        {
            "model_key": arm,
            "expected_questions": 2,
            "model": {
                "layers": 8,
                "hidden_dim": 4,
                "execution_mode": "padded_batch" if arm == "deepseek" else "unpadded_singleton",
            },
            "capture": {
                "batch_rows": 8,
                "parity_relative_tolerance": 0.01,
                "repeatability_relative_tolerance": 1e-5,
                "projection_margin": 1.25,
                "preservation_reserve_seconds": 900,
                "checkpoint_every_chunks": 25,
            },
        }
    )


def test_last_real_token_and_pre_norm_final_block_with_bf16_cpu_store():
    model = TinyDecoder()
    rows = [[2, 3], [4, 5, 6, 7]]
    values, errors = capture.capture_last(model, rows, 0, layers=8, width=4, check_tuple=True)
    expected = torch.stack([model.get_input_embeddings()(torch.tensor(r)).sum(0) for r in rows])
    assert values.dtype == torch.bfloat16 and values.device.type == "cpu"
    torch.testing.assert_close(values[:, 0], expected)
    torch.testing.assert_close(values[:, -1], expected)
    assert all(max(e) == 0 for e in errors.values())
    assert all(not block._forward_hooks for block in model.model.layers)


@pytest.mark.parametrize("arm", ["qwen", "deepseek"])
def test_storage_groups_remain_unpadded_singletons(arm):
    model = TinyDecoder()
    cfg = config(arm)
    cfg.model.execution_mode = "unpadded_singleton"
    result, _ = capture.capture_production(model, [[1], [1, 2, 3]], 0, cfg)
    assert result.shape == (2, 8, 4)
    assert model.calls == [((1, 1), True), ((1, 3), True)]


def test_unknown_execution_mode_rejects_before_model_forward():
    model = TinyDecoder()
    cfg = config()
    cfg.model.execution_mode = "invalid"
    with pytest.raises(ValueError, match="execution mode"):
        capture.capture_production(model, [[1]], 0, cfg)
    assert model.calls == []


def test_registered_deepseek_capture_uses_singletons_with_eight_row_storage_groups():
    cfg = OmegaConf.load(capture.ROOT / "configs/pilots/story_persona_crossmodel_capture.yaml")
    cfg.model_key = "deepseek"
    assert capture.singleton_execution(cfg)
    assert cfg.capture.batch_rows == 8


def test_dtype_errors_fail_without_leaving_hooks_attached():
    model = TinyDecoder().float()
    with pytest.raises(RuntimeError, match="BF16"):
        capture.capture_last(model, [[1]], 0, layers=8, width=4)
    assert all(not block._forward_hooks for block in model.model.layers)


def test_deepseek_batch_drift_fails_gate_and_preserves_vectors(tmp_path):
    ids = [[i % 20 + 1] * (i % 3 + 1) for i in range(16)]
    with pytest.raises(RuntimeError, match="numerical smoke rejected"):
        capture.numerical_smoke(
            TinyDecoder(batch_drift=True),
            SimpleNamespace(pad_token_id=0),
            ids,
            config(),
            tmp_path,
            "pin",
        )
    report = json.loads((tmp_path / "smoke.json").read_text())
    assert report["passed"] is False and report["repeatability_bitwise_equal"] is True
    assert report["mixed_batch_is_production"] is True
    assert "centered_cosine" not in report
    saved = torch.load(tmp_path / "smoke_vectors.pt", weights_only=True)
    assert saved["initial"].dtype == torch.bfloat16
    assert not torch.equal(saved["initial"], saved["batched"])


@pytest.mark.parametrize("arm", ["qwen", "deepseek"])
def test_singleton_mixed_batch_diagnostic_does_not_gate_production(tmp_path, capsys, arm):
    ids = [[i % 20 + 1] * (i % 3 + 1) for i in range(16)]
    cfg = config(arm)
    cfg.model.execution_mode = "unpadded_singleton"
    report = capture.numerical_smoke(
        TinyDecoder(batch_drift=True),
        SimpleNamespace(pad_token_id=0),
        ids,
        cfg,
        tmp_path,
        "pin",
    )
    assert report["passed"] is True and report["mixed_batch_is_production"] is False
    assert max(max(x) for x in report["mixed_batch_relative_errors"]) > 0.01
    assert "[capture-singleton-engaged]" in capsys.readouterr().out


def test_singleton_repeatability_failure_still_rejects_and_preserves_evidence(tmp_path):
    class DriftingDecoder(TinyDecoder):
        def forward(self, *args, **kwargs):
            with torch.no_grad():
                self.model.embed_tokens.weight.add_(16)
            return super().forward(*args, **kwargs)

    cfg = config()
    cfg.model.execution_mode = "unpadded_singleton"
    ids = [[i % 20 + 1] * (i % 3 + 1) for i in range(16)]
    with pytest.raises(RuntimeError, match="numerical smoke rejected"):
        capture.numerical_smoke(
            DriftingDecoder(), SimpleNamespace(pad_token_id=0), ids, cfg, tmp_path, "pin"
        )
    report = json.loads((tmp_path / "smoke.json").read_text())
    assert not report["passed"] and not report["mixed_batch_is_production"]
    assert not report["repeatability_bitwise_equal"]
    assert (tmp_path / "smoke_vectors.pt").is_file()


def test_projection_accounts_for_elapsed_setup_and_preservation_reserve():
    cfg = config()
    assert capture.projection(2000, 2, 240, cfg, now=0)["passed"]
    assert not capture.projection(2000, 2, 240, cfg, now=600)["passed"]
    assert not capture.projection(2000, 1, 0, cfg, now=1500)["passed"]
    with pytest.raises(ValueError):
        capture.projection(2000, float("nan"), 3, cfg)


def test_explicit_map_covers_all_61_main_layers_without_mtp_or_offload():
    mapping = capture.deepseek_device_map()
    assert set(mapping.values()) == set(range(8))
    assert all(f"model.layers.{i}" in mapping for i in range(61))
    assert "model.layers.61" not in mapping
    assert mapping["model.layers.0"] == mapping["model.layers.3"] == 0
    assert max(sum(mapping[f"model.layers.{j}"] == i for j in range(3, 61)) for i in range(8)) == 8


def test_resume_store_rejects_unrecorded_and_modified_tensor_bytes(tmp_path):
    (tmp_path / "chunks").mkdir()
    path = tmp_path / "chunks" / "batch_0000.pt"
    blob = {
        "fingerprint": "pin",
        "indices": [0],
        "vectors": torch.ones(1, 8, 4, dtype=torch.bfloat16),
    }
    torch.save(blob, path)
    with pytest.raises(RuntimeError, match="orphan"):
        capture.validate_store(tmp_path, "pin", [[0]], config(), {})
    checksums = {path.name: capture.file_digest(path)}
    capture.validate_store(tmp_path, "pin", [[0]], config(), checksums)
    blob["vectors"] += 1
    torch.save(blob, path)
    with pytest.raises(RuntimeError, match="content changed"):
        capture.validate_store(tmp_path, "pin", [[0]], config(), checksums)


def test_background_checkpoint_uses_immutable_metadata_and_hardlinked_chunks(tmp_path, monkeypatch):
    out = tmp_path / "arm"
    (out / "chunks").mkdir(parents=True)
    capture.write_json(out / "progress.json", {"completed": 25})
    (out / "chunks" / "batch_0000.pt").write_bytes(b"immutable")
    started, release = threading.Event(), threading.Event()
    observed = {}

    def publish(snapshot, cfg):
        started.set()
        assert release.wait(5)
        observed["progress"] = json.loads((snapshot / "progress.json").read_text())
        observed["inode"] = (snapshot / "chunks" / "batch_0000.pt").stat().st_ino

    monkeypatch.setattr(capture, "checkpoint", publish)
    publisher = capture.CheckpointPublisher(out, config())
    try:
        publisher.submit(25)
        assert started.wait(5)
        capture.write_json(out / "progress.json", {"completed": 26})
    finally:
        release.set()
        publisher.close()
    assert observed["progress"] == {"completed": 25}
    assert observed["inode"] == (out / "chunks" / "batch_0000.pt").stat().st_ino


def test_background_checkpoint_failure_propagates_at_close(tmp_path, monkeypatch):
    out = tmp_path / "arm"
    out.mkdir()

    def fail(*args):
        raise RuntimeError("publication failed")

    monkeypatch.setattr(capture, "checkpoint", fail)
    publisher = capture.CheckpointPublisher(out, config())
    publisher.submit(25)
    with pytest.raises(RuntimeError, match="publication failed"):
        publisher.close()


def test_deadline_requires_paid_start_before_now_and_source_gate_rejects_dirty(monkeypatch):
    monkeypatch.setenv("EPS_STORY_PERSONA_ALLOCATION_STARTED_UNIX", "100")
    monkeypatch.setenv("EPS_STORY_PERSONA_DEADLINE_UNIX", "200")
    monkeypatch.setattr(capture.time, "time", lambda: 150)
    assert capture.deadline_from_environment() == (100, 200)
    monkeypatch.setenv("EPS_STORY_PERSONA_ALLOCATION_STARTED_UNIX", "151")
    with pytest.raises(RuntimeError, match="allocation window"):
        capture.deadline_from_environment()
    monkeypatch.setenv("EPS_STORY_PERSONA_SOURCE_SHA", "a" * 40)
    monkeypatch.setattr(capture, "git_provenance", lambda: None)
    monkeypatch.setattr(
        capture, "as_metadata_dict", lambda *a, **k: {"git_commit": "a" * 40, "git_dirty": True}
    )
    with pytest.raises(RuntimeError, match="clean source"):
        capture.source_preflight()


def test_deepseek_renderer_persists_exact_no_bos_token_ids_and_rejects_eos():
    class Tokenizer:
        bos_token_id, eos_token_id = 0, 1
        add_bos_token = False
        append_eos = False

        def __call__(self, text, *, add_special_tokens):
            assert text == "description\n\nHuman: question\n\nAssistant:"
            tokens = [7, 8]
            if add_special_tokens and self.append_eos:
                tokens.append(1)
            return {"input_ids": tokens}

        def convert_ids_to_tokens(self, token):
            return str(token)

    cfg = config()
    cfg.model.bos_token_id = 0
    cfg.model.eos_token_id = 1
    cfg.model.add_special_tokens = True
    cfg.model.add_bos_token = False
    cfg.model.expected_tokens = {"min": 2, "max": 2, "total": 2}
    cfg.capture.max_context_tokens = 10
    rows = [{"row_id": "a:q", "description": "description", "question": "question"}]
    tokenizer = Tokenizer()
    assert capture.render_inputs(tokenizer, rows, cfg) == [[7, 8]]
    assert rows[0]["input_ids"] == [7, 8]
    assert rows[0]["prefix_sha256"] == capture.digest([7, 8])
    assert rows[0]["final_token_id"] == 8
    tokenizer.append_eos = True
    with pytest.raises(ValueError, match="neither BOS nor EOS"):
        capture.render_inputs(tokenizer, rows, cfg)


@pytest.mark.parametrize("mode", ["unpadded_singleton", "padded_batch"])
def test_capture_batches_emits_valid_chunks_and_refuses_expired_window(tmp_path, mode):
    cfg = config()
    cfg.model.execution_mode = mode
    out = tmp_path / "good"
    (out / "chunks").mkdir(parents=True)
    ids, batches = [[1], [2, 3], [4], [5, 6]], [[1, 3], [0, 2]]
    checksums = {}
    publisher = SimpleNamespace(submit=lambda _: pytest.fail("unexpected checkpoint"))
    model = TinyDecoder()
    now = capture.time.time()
    capture.capture_batches(
        model,
        SimpleNamespace(pad_token_id=0),
        ids,
        batches,
        checksums,
        cfg,
        out,
        "pin",
        now + 10000,
        0.1,
        now,
        publisher,
    )
    assert len(checksums) == 2
    if mode == "unpadded_singleton":
        assert model.calls == [((1, 2), True), ((1, 2), True), ((1, 1), True), ((1, 1), True)]
    else:
        assert model.calls == [((2, 2), True), ((2, 1), True)]
    capture.validate_store(out, "pin", batches, cfg, checksums)
    assert json.loads((out / "progress.json").read_text())["completed_rows"] == 4
    assert not (out / "capture_complete.json").exists()
    denied = tmp_path / "denied"
    (denied / "chunks").mkdir(parents=True)
    model = TinyDecoder()
    with pytest.raises(RuntimeError, match="exceeded window"):
        capture.capture_batches(
            model,
            SimpleNamespace(pad_token_id=0),
            ids,
            batches,
            {},
            cfg,
            denied,
            "pin",
            now,
            0.1,
            now,
            publisher,
        )
    assert model.calls == [] and list((denied / "chunks").iterdir()) == []
