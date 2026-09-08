"""Chat-only production-body checks. No network, model weights, or GPU work."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import create_autospec

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2588_panel_common as PC
import issue2588_run_cell as RC


class Tokenizer:
    """Explicit tokenizer call signatures; character offsets expose boundary errors."""

    pad_token_id = 0
    eos_token_id = 999999

    def apply_chat_template(
        self, conversation, *, tokenize, add_generation_prompt, enable_thinking, return_dict=None
    ):
        text = f"user:{conversation[0]['content']}\n<|im_start|>assistant\n"
        if not enable_thinking:
            text += "<think>\n\n</think>\n\n"
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(i) for i in token_ids if i != self.eos_token_id)


@dataclass
class SamplingParams:
    temperature: float
    top_p: float
    seed: int
    max_tokens: int


@dataclass
class TokensPrompt:
    prompt_token_ids: list[int]


class Engine:
    """Explicit vLLM boundary signature used by the production constructor."""

    constructed: ClassVar[list[dict]] = []

    def __init__(
        self,
        *,
        model,
        tensor_parallel_size,
        seed,
        dtype,
        max_model_len,
        gpu_memory_utilization,
        max_num_seqs,
        enforce_eager,
        enable_prefix_caching,
        disable_log_stats,
        revision,
        tokenizer_revision,
        download_dir,
        tokenizer,
    ):
        self.constructed.append(
            {
                "model": model,
                "revision": revision,
                "tokenizer_revision": tokenizer_revision,
                "download_dir": download_dir,
                "max_model_len": max_model_len,
            }
        )

    def generate(self, prompts, sampling_params, use_tqdm=False):
        text = "thought</think>\nanswer" if sampling_params.max_tokens > 4096 else "answer"
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=text, finish_reason="stop", token_ids=[ord(c) for c in text]
                    )
                ]
            )
            for _ in prompts
        ]

    @property
    def llm_engine(self):
        return self

    def enqueue(
        self,
        prompts,
        sampling_params=None,
        lora_request=None,
        priority=None,
        use_tqdm=True,
        tokenization_kwargs=None,
        mm_processor_kwargs=None,
    ):
        self.queue = self.generate(prompts, sampling_params, use_tqdm=False)
        ids = [str(i) for i in range(len(prompts))]
        for request_id, prompt, output in zip(ids, prompts, self.queue, strict=True):
            output.request_id = request_id
            output.prompt_token_ids = prompt.prompt_token_ids
            output.finished = True
        internal_ids = [f"{key}-fixture8" for key in ids]
        self.output_processor = SimpleNamespace(
            request_states={
                key: SimpleNamespace(external_req_id=external)
                for key, external in zip(internal_ids, ids, strict=True)
            }
        )
        return internal_ids

    def has_unfinished_requests(self):
        return bool(getattr(self, "queue", []))

    def step(self):
        output = self.queue.pop()  # Deliberately completes out of input order.
        del self.output_processor.request_states[f"{output.request_id}-fixture8"]
        return [output]


@pytest.fixture
def generic(tmp_path, monkeypatch):
    monkeypatch.setattr(PC, "CAP_PROFILE", "long")
    monkeypatch.setattr(PC, "CAP", PC.CAP_PROFILES["long"])
    monkeypatch.setattr(PC, "PANEL_PREFIX", "issue2588_capability_panel_cap_long")
    args = RC._build_parser().parse_args(
        [
            "--surface",
            "generic",
            "--run-id",
            "unit-v1",
            "--cell",
            "q3_8b_a",
            "--device",
            "cpu",
            "--out-root",
            str(tmp_path),
        ]
    )
    cell = PC.cell_by_key(args.cell)
    paths = RC._paths(args, cell)
    # Model-transfer boundary fixture: no real weights, but exercise exact
    # staged-path/size validation used by the production consumers.
    monkeypatch.setattr(RC, "_CHAT_MODEL_FILES", {"config.json": 2})
    snapshot = paths["cache"] / "models--Qwen--Qwen3-8B/snapshots" / cell.model.revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    return args, cell, paths


def test_registry_is_opt_in_and_geometry_is_pinned():
    assert len(PC.all_cells()) == 33
    assert sum(len(c.input_positions) for c in PC.all_cells()) == 36
    assert len(PC.all_cells(include_generic=True)) == 35
    assert sum(len(c.input_positions) for c in PC.all_cells(include_generic=True)) == 38
    model = PC.PANEL["q3_8b"]
    assert (model.n_layers, model.h_dim, model.tp_gpus) == (36, 4096, 1)
    assert model.revision == PC.QWEN3_8B_REVISION
    assert PC.sweep_layers(model.n_layers) == [*range(0, 35, 2), 35]
    assert all(PC.cell_by_key(f"q3_8b_{arm}").fresh for arm in ("a", "b"))


def test_scope_closure_and_legacy_defaults(generic):
    args, cell, paths = generic
    assert RC._sequence_for(args) == RC._GENERIC_SEQUENCE
    assert RC._GENERIC_SEQUENCE.index("upload-raw") < RC._GENERIC_SEQUENCE.index("capture")
    assert RC._gpqa_seeds(args) == ()
    assert set(RC._stage_names(args, cell)) == set(RC.GENERIC_SPLITS) | {
        "ceiling_s43",
        "ceiling_s44",
    }
    assert not any("gpqa" in s for s in RC._stage_row_estimates(args, cell))
    for phase in (
        "gpqa-transfer",
        "resid",
        "nulls",
        "g2-anchor",
        "purge-model-cache",
        "smoke-null-timing",
    ):
        args.phase = phase
        with pytest.raises(AssertionError, match="excluded"):
            RC._sequence_for(args)
        with pytest.raises(AssertionError, match="excluded"):
            RC.PHASES[phase](args, cell, paths)
    with pytest.raises(AssertionError, match="excluded"):
        RC._load_gpqa_prompts(args)
    with pytest.raises(AssertionError, match="excluded"):
        RC._gpqa_behavioral(args, cell, paths, [])
    args.surface = "full"
    with pytest.raises(AssertionError, match="generic only"):
        RC._validate_scope(args, cell)
    legacy = RC._build_parser().parse_args([])
    assert RC._sequence_for(legacy) == RC._ALL_SEQUENCE


def test_paths_provenance_and_partial_resume(generic):
    args, cell, paths = generic
    assert "generic/unit-v1/cells_cap_long/q3_8b_a" in str(paths["cell"])
    assert RC._cell_prefix(args, cell).endswith("/generic/unit-v1/q3_8b/nothink")
    PC.write_json_atomic(paths["cell"] / "prologue.json", {"test": 1})
    assert not RC._phase_complete(args, paths, "prologue")
    RC._mark_phase_done(args, cell, paths, "prologue")
    assert RC._phase_complete(args, paths, "prologue")
    PC.write_json_atomic(paths["cell"] / "prologue.json", {"test": 2})
    with pytest.raises(AssertionError, match="partial checkpoint"):
        RC._phase_complete(args, paths, "prologue")
    args.capture_batch_size = 7
    with pytest.raises(AssertionError, match="incompatible run provenance"):
        RC._paths(args, cell)


def test_exact_template_prefill_and_token_fidelity():
    tok = Tokenizer()
    for arm in ("a", "b"):
        assert PC.assert_template_sidespec(tok, "legacy_qwen3", arm)
    prompt = PC.render_prompt_text(tok, "x", "legacy_qwen3", "b")
    assert prompt.count("<think>") == 1
    text = "reason</think>\nanswer"
    _, _, cot, ans = PC.segment_completion_arm(text, "prefill")
    row = {
        "row_id": "r",
        "prompt": prompt,
        "prompt_ids": tok.encode(prompt),
        "n_prompt_tokens": len(prompt),
        "text": text,
        "read_points": {},
        "ans_char_span": ans,
        "cot_char_span": cot,
        "sampled_token_ids": [*tok.encode(text), tok.eos_token_id],
    }
    built, reason = PC.build_capture_row_2588(tok, row, positions_wanted=("cot_boundary",))
    assert not reason and built["positions"]["cot_boundary"] >= len(prompt)
    assert not built["token_fidelity"]["sampled_completion_ids_match"]
    assert built["token_fidelity"]["sampled_decoded_text_match"]
    row["sampled_token_ids"][0] = ord("X")
    with pytest.raises(AssertionError, match="capture text/boundaries"):
        PC.build_capture_row_2588(tok, row, positions_wanted=("cot_boundary",))
    row["prompt_ids"][0] += 1
    with pytest.raises(AssertionError, match="prompt token identity"):
        PC.build_capture_row_2588(tok, row, positions_wanted=("cot_boundary",))


def test_pinned_generation_real_bodies_and_resume(generic, monkeypatch):
    import transformers

    args, cell, paths = generic
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=Engine,
            SamplingParams=SamplingParams,
            TokensPrompt=TokensPrompt,
            __version__="fixture",
        ),
    )
    config = transformers.Qwen3Config(
        num_hidden_layers=36, hidden_size=4096, max_position_embeddings=40960
    )
    load = create_autospec(transformers.AutoConfig.from_pretrained, return_value=config)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", load)
    holder = {"llm": None, "mml": 0}
    rows = [{"row_id": "train_10k_1", "prompt": "hello"}]
    result = RC._gen_stage_with_regen(
        args,
        cell,
        Tokenizer(),
        rows,
        stage="train_10k",
        cap=2048,
        seed=42,
        paths=paths,
        llm_holder=holder,
    )
    assert Path(load.call_args.args[0]).name == PC.QWEN3_8B_REVISION
    assert Engine.constructed[-1]["revision"] == PC.QWEN3_8B_REVISION
    assert Engine.constructed[-1]["tokenizer_revision"] == PC.QWEN3_8B_REVISION
    assert Engine.constructed[-1]["download_dir"] == str(paths["cache"])
    assert result[0]["sampled_token_ids"] == Tokenizer().encode("answer")
    assert result[0]["prompt_ids"]
    assert (paths["raw"] / "train_10k/partial/initial/chunk0000.json").is_file()
    load.reset_mock()
    assert (
        RC._gen_stage_with_regen(
            args,
            cell,
            Tokenizer(),
            rows,
            stage="train_10k",
            cap=2048,
            seed=42,
            paths=paths,
            llm_holder=holder,
        )
        == result
    )
    load.assert_not_called()
    (paths["raw"] / "train_10k/chunk0000.json").unlink()
    with pytest.raises(AssertionError, match="partial/stale raw chunks"):
        RC._gen_stage_with_regen(
            args,
            cell,
            Tokenizer(),
            rows,
            stage="train_10k",
            cap=2048,
            seed=42,
            paths=paths,
            llm_holder=holder,
        )


def test_per_completion_resume_preserves_order_and_skips_saved(generic, monkeypatch):
    """Interrupt the real generation loop between completions, then resume it."""
    args, cell, paths = generic
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            SamplingParams=SamplingParams,
            TokensPrompt=TokensPrompt,
        ),
    )
    rows = [{"row_id": i, "prompt": f"prompt {i}"} for i in range(3)]
    directory = paths["raw"] / "train_10k/partial/initial/rows"
    kwargs = dict(
        stage="train_10k",
        cap=2048,
        seed=42,
        checkpoint_dir=directory,
        checkpoint_identity=RC._identity(args, cell),
    )
    engine = Engine.__new__(Engine)
    original = engine.step
    calls = 0

    def interrupted_step():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture interruption")
        return original()

    monkeypatch.setattr(engine, "step", create_autospec(engine.step, side_effect=interrupted_step))
    with pytest.raises(RuntimeError, match="fixture interruption"):
        RC._gen_rows(engine, Tokenizer(), cell, rows, **kwargs)
    saved = directory / "row000002.json"
    assert sorted(p.name for p in directory.glob("*.json")) == [saved.name]
    original_bytes = saved.read_bytes()
    resumed = Engine.__new__(Engine)
    enqueue = create_autospec(resumed.enqueue, side_effect=resumed.enqueue)
    monkeypatch.setattr(resumed, "enqueue", enqueue)
    result = RC._gen_rows(resumed, Tokenizer(), cell, rows, **kwargs)
    assert [r["row_id"] for r in result] == [0, 1, 2]
    assert len(enqueue.call_args.args[0]) == 2
    assert saved.read_bytes() == original_bytes
    enqueue.reset_mock()
    assert RC._gen_rows(resumed, Tokenizer(), cell, rows, **kwargs) == result
    enqueue.assert_not_called()
    with pytest.raises(AssertionError):
        RC._gen_rows(resumed, Tokenizer(), cell, rows, **{**kwargs, "cap": 1024})
    saved.write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        RC._gen_rows(resumed, Tokenizer(), cell, rows, **kwargs)


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate"])
def test_request_id_mapping_rejects_bad_engine_states(monkeypatch, fault):
    engine = Engine.__new__(Engine)
    original = engine.enqueue

    def enqueue(*args, **kwargs):
        ids = original(*args, **kwargs)
        states = engine.output_processor.request_states
        if fault == "missing":
            del states[ids[0]]
        elif fault == "extra":
            states["unowned"] = SimpleNamespace(external_req_id="unowned")
        else:
            states[ids[1]].external_req_id = states[ids[0]].external_req_id
        return ids

    monkeypatch.setattr(engine, "enqueue", create_autospec(engine.enqueue, side_effect=enqueue))
    step = create_autospec(engine.step, side_effect=AssertionError("must not step"))
    monkeypatch.setattr(engine, "step", step)
    prompts = [TokensPrompt(prompt_token_ids=[1]), TokensPrompt(prompt_token_ids=[2])]
    pending = [(i, {"prompt_ids": [i + 1]}) for i in range(2)]
    with pytest.raises(AssertionError, match="unowned queued request states|duplicate external"):
        list(
            RC._completed_generation_rows(
                engine, prompts, SimpleNamespace(max_tokens=2048), pending
            )
        )
    step.assert_not_called()


def test_capture_model_load_uses_actual_snapshot_path(generic, monkeypatch):
    import huggingface_hub

    args, cell, paths = generic
    snapshot = str(paths["cache"] / "models--Qwen--Qwen3-8B/snapshots" / PC.QWEN3_8B_REVISION)
    downloader = create_autospec(huggingface_hub.snapshot_download, return_value=snapshot)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", downloader)
    loader = create_autospec(RC.G._load_capture_model, return_value="model-boundary")
    monkeypatch.setattr(RC.G, "_load_capture_model", loader)
    assert RC._load_capture_model(cell, args.device, paths["cache"]) == "model-boundary"
    downloader.assert_not_called()
    loader.assert_called_once_with(snapshot, "cpu", "bfloat16")


def test_capture_direct_phase_refuses_unfinished_parse(generic):
    args, cell, paths = generic
    with pytest.raises(AssertionError, match="completed parse"):
        RC.phase_capture(args, cell, paths)


@dataclass(frozen=True)
class PlacementParameter:
    """Metadata-only external GPU parameter boundary; never allocates GPU memory."""

    device: torch.device
    dtype: torch.dtype


class PlacementModel:
    """Mirror nn.Module's named-parameter signature for mixed-device/dtype fixtures."""

    def __init__(self, parameters):
        self.parameters_fixture = parameters

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return iter(self.parameters_fixture)


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.bfloat16),
        ("cuda:0", torch.float32),
        ("cuda:1", torch.bfloat16),
        ("meta", torch.bfloat16),
    ],
)
def test_capture_placement_checks_every_parameter(device, dtype):
    good = PlacementParameter(torch.device("cuda:0"), torch.bfloat16)
    bad = PlacementParameter(torch.device(device), dtype)
    with pytest.raises(AssertionError, match=r"last\.weight"):
        RC._assert_generic_capture_placement(
            PlacementModel([("first.weight", good), ("last.weight", bad)])
        )


def test_capture_placement_success_empty_and_real_cpu_failure(caplog):
    good = PlacementParameter(torch.device("cuda:0"), torch.bfloat16)
    with caplog.at_level("INFO"):
        RC._assert_generic_capture_placement(PlacementModel([("first", good), ("last", good)]))
    assert "parameter_tensors=2 device=cuda:0 dtype=torch.bfloat16" in caplog.text
    with pytest.raises(AssertionError, match="no named parameters"):
        RC._assert_generic_capture_placement(torch.nn.Module())
    with pytest.raises(AssertionError, match=r"device=cpu dtype=torch\.float32"):
        RC._assert_generic_capture_placement(torch.nn.Linear(2, 2))


def test_capture_cpu_fallback_refused_before_forward(generic, monkeypatch):
    import transformers

    args, cell, paths = generic
    _write_raw_upload_fixture(args, cell, paths)
    cpu_model = torch.nn.Linear(2, 2)
    forward = create_autospec(cpu_model.forward)
    monkeypatch.setattr(cpu_model, "forward", forward)
    monkeypatch.setattr(
        RC, "_load_capture_model", create_autospec(RC._load_capture_model, return_value=cpu_model)
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        create_autospec(transformers.AutoTokenizer.from_pretrained, return_value=Tokenizer()),
    )
    monkeypatch.setattr(RC, "_assert_headroom", create_autospec(RC._assert_headroom))
    with pytest.raises(AssertionError, match="generic capture placement FAIL"):
        RC.phase_capture(args, cell, paths)
    forward.assert_not_called()
    assert not any(paths["capture"].rglob("*.npz"))
    assert not RC._phase_done_path(args, paths, "capture").exists()


def test_fit_pilot_pause_never_marks_sweep_complete(generic):
    args, cell, paths = generic
    args.phase = "fits"
    args.fit_max_units = 1
    unit = {"unit_elapsed_s": 12.5, "n": {"tr": 9990, "val": 395, "te": 981}, "d": 4096}
    with pytest.raises(RC.FitPilotPause):
        RC._pause_fit_pilot(args, cell, paths, "prompt_last", 0, unit)
    record = json.loads((paths["fits"] / "fit_pilot.json").read_text())
    assert record["status"] == "pilot_complete" and record["phase_complete"] is False
    assert record["total_layer_units"] == 19 and record["unit_elapsed_s"] == 12.5
    assert not RC._phase_done_path(args, paths, "fits").exists()
    with pytest.raises(AssertionError, match="complete sweep"):
        RC.phase_upload_fits(args, cell, paths)


def test_generic_upload_shards_text_and_preserves_local_bytes(tmp_path, monkeypatch):
    source = tmp_path / "raw.jsonl"
    text = (json.dumps({"text": "x" * 999}) + "\n") * 9600
    source.write_text(text)
    seen = {}

    def upload(
        local_dir,
        repo_id,
        repo_type,
        path_in_repo,
        allow_patterns,
        expected_repo_paths,
        ignore_patterns=None,
        delete_after=False,
        *,
        private=False,
    ):
        seen.update({name: (local_dir / name).read_bytes() for name in allow_patterns})
        assert all(len(value) <= 9_500_000 for value in seen.values())
        return "https://huggingface.co/datasets/example/tree/revision"

    boundary = create_autospec(RC.HUB._upload_folder_filtered, side_effect=upload)
    monkeypatch.setattr(RC.HUB, "_upload_folder_filtered", boundary)
    names = RC._upload_generic(source, "scope/parsed/raw.jsonl")
    assert "scope/parsed/raw.manifest.json" in names
    manifest = json.loads(seen["raw.manifest.json"])
    assert b"".join(seen[name] for name in manifest["parts"]) == text.encode()
    assert source.read_text() == text
    assert RC.HUB._parse_shard_manifest(seen["raw.manifest.json"].decode(), what="fixture")[0]


@pytest.fixture
def upload_boundary(monkeypatch):
    """Signature-checked Hub boundaries record exact payloads without any remote writes."""
    import huggingface_hub

    captured = {}
    api = create_autospec(huggingface_hub.HfApi, instance=True)
    api.repo_info.return_value = SimpleNamespace(sha="c" * 40)
    monkeypatch.setattr(
        huggingface_hub, "HfApi", create_autospec(huggingface_hub.HfApi, return_value=api)
    )

    def folder(
        local_dir,
        repo_id,
        repo_type,
        path_in_repo,
        allow_patterns,
        expected_repo_paths,
        ignore_patterns=None,
        delete_after=False,
        *,
        private=False,
    ):
        assert repo_id == PC.HF_DATA_REPO and repo_type == "dataset"
        actual = {
            f"{path_in_repo}/{name}": (local_dir / name).read_bytes() for name in allow_patterns
        }
        assert set(actual) == set(expected_repo_paths)
        captured.update(actual)
        return "verified"

    def single(local_path, repo_id, repo_type, path_in_repo, **kwargs):
        assert repo_id == PC.HF_DATA_REPO and repo_type == "dataset"
        assert kwargs["upload_as_file"] and kwargs["raise_on_error"]
        captured[path_in_repo] = local_path.read_bytes()
        return "verified"

    bulk = create_autospec(RC.HUB._upload_folder_filtered, side_effect=folder)
    one = create_autospec(RC.HUB._upload, side_effect=single)
    verify = create_autospec(RC.HUB.verify_repo_paths_uploaded, return_value=[])
    monkeypatch.setattr(RC.HUB, "_upload_folder_filtered", bulk)
    monkeypatch.setattr(RC.HUB, "_upload", one)
    monkeypatch.setattr(RC.HUB, "verify_repo_paths_uploaded", verify)
    return SimpleNamespace(bytes=captured, bulk=bulk, single=one, verify=verify)


def _old_payload_oracle(args, cell, paths, phase):
    """Frozen ebadfb46 payload grouping, without its unchanged receipt/completion writes."""
    prefix = RC._cell_prefix(args, cell)
    for stage in RC._stage_names(args, cell):
        if phase == "upload-raw":
            RC._upload_generic(paths["raw"] / stage, f"{prefix}/raw_completions/{stage}")
            for suffix in (".jsonl", "_drops.json", "_capture_drops.json"):
                file = paths["parsed"] / f"{stage}{suffix}"
                if file.exists():
                    RC._upload_generic(file, f"{prefix}/parsed/{file.name}")
        else:
            RC._upload_generic(
                paths["capture"] / stage, f"{prefix}/analysis_tensors/capture/{stage}"
            )
            file = paths["parsed"] / f"{stage}_capture_drops.json"
            RC._upload_file(file, f"{prefix}/parsed/{file.name}", "oracle")
    if phase == "upload-raw":
        for name in ("run_identity.json", "stage.json", "prologue.json", "stage_runtime.json"):
            if (paths["cell"] / name).exists():
                RC._upload_file(paths["cell"] / name, f"{prefix}/{name}", "oracle")
        RC._upload_file(
            paths["fits"] / "dropped_row_ids.json",
            f"{prefix}/parsed/dropped_row_ids.json",
            "oracle",
        )
        for name in RC._GENERIC_SEQUENCE:
            file = RC._phase_done_path(args, paths, name)
            if file.exists():
                RC._upload_file(file, f"{prefix}/phase_done/{name}.json", "oracle")
    else:
        file = paths["cell"] / "capture_input_validation.json"
        RC._upload_file(file, f"{prefix}/{file.name}", "oracle")


def _write_raw_upload_fixture(args, cell, paths):
    """Persist real parsed/checkpoint artifacts with an extra raw partial chunk per stage."""
    tok = Tokenizer()
    for stage in RC._stage_names(args, cell):
        prompt = PC.render_prompt_text(tok, "fixture", cell.model.family, cell.arm)
        row = {
            "row_id": f"{stage}_0",
            "stage": stage,
            "prompt": prompt,
            "prompt_ids": tok.encode(prompt),
            "n_prompt_tokens": len(prompt),
            "text": "answer",
            "sampled_token_ids": tok.encode("answer"),
            "n_comp_tokens": 6,
            "read_points": {"prompt_last": len(prompt) - 1},
            "finish_reason": "stop",
        }
        directory = paths["raw"] / stage
        PC.write_json_atomic(directory / "chunk0000.json", {"rows": [row]})
        PC.write_json_atomic(directory / "partial/initial/chunk0000.json", {"rows": [row]})
        PC.write_json_atomic(directory / "cap_hit_report.json", {"n": 1})
        RC._write_checkpoint(
            args, cell, paths, directory / "stage_done.json", RC._raw_stage_files(paths, stage)
        )
    RC._mark_phase_done(args, cell, paths, "gen")
    RC.phase_parse(args, cell, paths)
    RC._mark_phase_done(args, cell, paths, "parse")
    for name in ("stage.json", "prologue.json", "stage_runtime.json"):
        PC.write_json_atomic(paths["cell"] / name, {"fixture": name})


def test_raw_phase_one_payload_exact_old_mapping(generic, upload_boundary):
    args, cell, paths = generic
    _write_raw_upload_fixture(args, cell, paths)
    _old_payload_oracle(args, cell, paths, "upload-raw")
    expected = {
        name: hashlib.sha256(data).hexdigest() for name, data in upload_boundary.bytes.items()
    }
    upload_boundary.bytes.clear()
    upload_boundary.bulk.reset_mock()
    upload_boundary.single.reset_mock()
    assert RC._run_phases(args, cell, paths, ("upload-raw",)) == ["upload-raw"]
    receipt = json.loads((paths["cell"] / "uploads/upload-raw.json").read_text())
    assert set(receipt["paths"]) == set(expected)
    assert {
        name: hashlib.sha256(upload_boundary.bytes[name]).hexdigest() for name in receipt["paths"]
    } == expected
    assert upload_boundary.bulk.call_count == 1
    assert upload_boundary.single.call_count == 2  # receipt + completion checkpoint only
    assert sum("/partial/initial/" in name for name in expected) == 5
    assert any(name.endswith("parsed/dropped_row_ids.json") for name in expected)
    assert RC._phase_complete(args, paths, "upload-raw")
    assert upload_boundary.verify.call_args.kwargs["revision"] == "c" * 40
    assert not list(paths["cell"].parent.glob("i2588-*"))


@pytest.mark.parametrize("failure", ("missing", "corrupt", "payload", "immutable"))
def test_batch_raw_failure_cannot_mint_success(generic, upload_boundary, failure):
    args, cell, paths = generic
    _write_raw_upload_fixture(args, cell, paths)
    raw = paths["raw"] / "train_10k/chunk0000.json"
    if failure == "missing":
        raw.unlink()
    elif failure == "corrupt":
        raw.write_text("{}")
    elif failure == "payload":
        upload_boundary.bulk.side_effect = None
        upload_boundary.bulk.return_value = ""
    else:
        upload_boundary.verify.return_value = ["missing/expected/path"]
    with pytest.raises((AssertionError, FileNotFoundError, KeyError)):
        RC._run_phases(args, cell, paths, ("upload-raw",))
    assert not (paths["cell"] / "uploads/upload-raw.json").exists()
    assert not RC._phase_done_path(args, paths, "upload-raw").exists()
    upload_boundary.single.assert_not_called()
    assert not list(paths["cell"].parent.glob("i2588-*"))


def test_batch_mapping_preserves_shards_and_rejects_collisions(tmp_path, upload_boundary):
    source = tmp_path / "raw.jsonl"
    source.write_text((json.dumps({"text": "x" * 999}) + "\n") * 9600)
    marker = tmp_path / "marker.json"
    marker.write_text('{"done": false}')
    RC._upload_generic(source, "scope/raw/partial/raw.jsonl")
    RC._upload_generic(marker, "scope/phase_done/marker.json")
    expected = dict(upload_boundary.bytes)
    source_hash = RC._sha256_file(source)
    upload_boundary.bytes.clear()
    upload_boundary.bulk.reset_mock()
    names = RC._upload_generic_files(
        [(source, "raw/partial/raw.jsonl"), (marker, "phase_done/marker.json")],
        "scope",
        staging_parent=tmp_path,
    )
    assert set(names) == set(expected) and upload_boundary.bytes == expected
    assert upload_boundary.bulk.call_count == 1 and RC._sha256_file(source) == source_hash
    manifest = json.loads(expected["scope/raw/partial/raw.manifest.json"])
    assert (
        b"".join(expected[f"scope/raw/partial/{part}"] for part in manifest["parts"])
        == source.read_bytes()
    )
    for destinations in (
        ("same", "same"),
        ("parent", "parent/child"),
        ("../escape", "ok"),
        ("raw.jsonl", "raw.manifest.json"),
        ("raw.jsonl", "raw.shard00.jsonl"),
    ):
        with pytest.raises(AssertionError):
            RC._upload_generic_files(
                list(zip((source, marker), destinations, strict=True)),
                "scope",
                staging_parent=tmp_path,
            )
    assert upload_boundary.bulk.call_count == 1


def test_immutable_upload_receipt_and_partial_fit_coverage(generic, upload_boundary):
    """Real partial upload batches all provenance and retains its failure semantics."""
    args, cell, paths = generic
    PC.write_json_atomic(paths["fits"] / "fit_pilot.json", {"phase_complete": False})
    RC.phase_upload_partial(args, cell, paths)
    receipt = json.loads((paths["cell"] / "uploads/upload-partial.json").read_text())
    assert receipt["revision"] == "c" * 40
    assert any(p.endswith("partial/fits/fit_pilot.json") for p in receipt["paths"])
    assert upload_boundary.bulk.call_count == 1
    assert upload_boundary.verify.call_args.kwargs["revision"] == "c" * 40
    assert not RC._phase_done_path(args, paths, "fits").exists()
    assert not RC._phase_done_path(args, paths, "upload-partial").exists()


def test_completion_checkpoints_pack_and_restore_exact_bytes(tmp_path, upload_boundary):
    """The live upload helper packs even pilot-sized row checkpoints, with exact restore."""
    from issue1739_pack import unpack_shards

    from explore_persona_space.atomic_io import write_json_atomic

    sources = tmp_path / "cell"
    files = []
    for i in range(4):
        name = f"raw_completions/train_10k/partial/initial/rows/row{i:06d}.json"
        path = sources / name
        write_json_atomic(path, {"identity": {"run": "fixture"}, "row": {"id": i}}, indent=1)
        files.append((path, name))
    names = RC._upload_generic_files(files, "scope", staging_parent=tmp_path)
    assert len(names) == 2  # one bounded line-shard and its content-hashed manifest
    staged = tmp_path / "download"
    staged.mkdir()
    for name in names:
        path = staged / name.removeprefix("scope/packed_generation_checkpoints/")
        path.write_bytes(upload_boundary.bytes[name])
    restored = tmp_path / "restored"
    unpack_shards(staged, restored)
    for source, relative in files:
        assert (restored / relative).read_bytes() == source.read_bytes()
    assert upload_boundary.bulk.call_count == 1


def test_cap_window_and_capture_storage_accounting(generic):
    args, cell, _paths = generic
    assert PC.cap_effective("b", "generic", 40960) == 32768
    assert PC.regen_cap(32768, 40960, 65536) == 33856
    assert PC.regen_skip_reason(32768, 40960, 65536) is not None
    stages = RC._stage_row_estimates(args, cell)
    total = sum(n * slots for n, slots in stages.values()) * 19 * 4096 * 4 * 2
    assert total == 15440281600


def test_generic_loader_and_stage_real_bodies(generic, monkeypatch, tmp_path):
    import huggingface_hub

    args, cell, paths = generic
    selected = {s: [2, 1] for s in RC.GENERIC_SPLITS}
    payload = {"splits": selected, "sha256": {}}
    split_path = tmp_path / "split_ids.json"
    PC.write_json_atomic(split_path, payload)
    monkeypatch.setattr(PC, "SPLIT_IDS_PATH", split_path)
    monkeypatch.setattr(
        PC,
        "CHAT_SPLIT_SHA256",
        {s: PC.sha256_text(json.dumps(ids, separators=(",", ":"))) for s, ids in selected.items()},
    )
    monkeypatch.setattr(PC, "EXPECTED_SPLIT_COUNTS", {s: 2 for s in selected})
    manifests = {}
    for split in RC.GENERIC_SPLITS:
        key = RC.G.SPLIT_TO_MANIFEST[split][0]
        name = RC.G.MANIFEST_SPLIT_FILES[key]
        file = tmp_path / name
        PC.write_jsonl_atomic(
            file, [{"ladder_local_id": i, "prompt": f"prompt{i}"} for i in (1, 2)]
        )
        manifests[name] = file
    monkeypatch.setattr(
        PC, "CHAT_MANIFEST_SHA256", {n: RC._sha256_file(p) for n, p in manifests.items()}
    )

    def download(repo_id, filename, *, repo_type, cache_dir, revision, local_files_only=False):
        assert revision == PC.MANIFEST_REVISION
        return manifests[Path(filename).name]

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        create_autospec(huggingface_hub.hf_hub_download, side_effect=download),
    )
    monkeypatch.setattr(
        RC,
        "_p0_union_drop",
        create_autospec(RC._p0_union_drop, return_value={s: set() for s in selected}),
    )
    RC.phase_stage(args, cell, paths)
    record = json.loads((paths["cell"] / "stage.json").read_text())
    assert record["splits"]["train_10k"]["row_ids"] == ["train_10k_2", "train_10k_1"]
    assert set(record["splits"]) == set(RC.GENERIC_SPLITS)
    assert (
        RC._load_generic_rows(args, paths["cache"], local_only=True)["train_10k"][0][
            "ladder_local_id"
        ]
        == 2
    )


def test_staged_snapshot_completeness_and_offline_tokenizer(generic):
    args, cell, paths = generic
    kwargs = RC._tokenizer_kwargs(args, cell, paths)
    assert kwargs == {
        "revision": PC.QWEN3_8B_REVISION,
        "cache_dir": str(paths["cache"]),
        "local_files_only": True,
    }
    snapshot = RC._local_model_snapshot(cell, paths["cache"])
    (snapshot / "config.json").write_text("x")
    with pytest.raises(AssertionError, match="missing/truncated"):
        RC._local_model_snapshot(cell, paths["cache"])


class CaptureModel(torch.nn.Module):
    """Real hook dispatch and tensor reductions with an explicit CPU forward boundary."""

    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(36)])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4096)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace()


def test_parse_capture_upload_real_bodies(generic, monkeypatch, upload_boundary):
    import transformers

    args, cell, paths = generic
    tok = Tokenizer()
    for stage in RC._stage_names(args, cell):
        rows = []
        for i in range(2):
            prompt = PC.render_prompt_text(tok, f"hello{i}", cell.model.family, cell.arm)
            text = f"answer{i}"
            rows.append(
                {
                    "row_id": f"{stage}_{i}",
                    "stage": stage,
                    "prompt": prompt,
                    "prompt_ids": tok.encode(prompt),
                    "n_prompt_tokens": len(prompt),
                    "text": text,
                    "sampled_token_ids": tok.encode(text),
                    "n_comp_tokens": len(text),
                    "read_points": {"prompt_last": len(prompt) - 1},
                    "finish_reason": "stop",
                }
            )
        directory = paths["raw"] / stage
        PC.write_json_atomic(directory / "chunk0000.json", {"rows": rows})
        PC.write_json_atomic(directory / "cap_hit_report.json", {"n": 2})
        RC._write_checkpoint(
            args, cell, paths, directory / "stage_done.json", RC._raw_stage_files(paths, stage)
        )
    RC._mark_phase_done(args, cell, paths, "gen")
    RC.phase_parse(args, cell, paths)
    RC._mark_phase_done(args, cell, paths, "parse")
    tokenizer_load = create_autospec(transformers.AutoTokenizer.from_pretrained, return_value=tok)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_load)
    monkeypatch.setattr(
        RC,
        "_load_capture_model",
        create_autospec(RC._load_capture_model, return_value=CaptureModel()),
    )
    monkeypatch.setattr(RC, "_assert_headroom", create_autospec(RC._assert_headroom))
    # This existing fixture exercises CPU hooks/reductions, not actual GPU placement.
    # The placement guard's real body and before-forward rejection are tested above.
    placement = create_autospec(RC._assert_generic_capture_placement)
    monkeypatch.setattr(RC, "_assert_generic_capture_placement", placement)
    RC.phase_capture(args, cell, paths)
    placement.assert_called_once()
    assert tokenizer_load.call_args.kwargs["revision"] == PC.QWEN3_8B_REVISION
    RC._mark_phase_done(args, cell, paths, "capture")
    assert RC._phase_complete(args, paths, "capture")
    for stage in RC._stage_names(args, cell):
        rows = json.loads((paths["capture"] / stage / "rows.json").read_text())["rows"]
        assert all(r["token_fidelity"]["sampled_completion_ids_match"] for r in rows)
        assert len(rows) == 2
    _old_payload_oracle(args, cell, paths, "upload-capture")
    expected = {
        name: hashlib.sha256(data).hexdigest() for name, data in upload_boundary.bytes.items()
    }
    upload_boundary.single.reset_mock()
    for file in (
        paths["capture"] / "train_10k/L00/shard000.npz",
        paths["parsed"] / "train_10k_capture_drops.json",
    ):
        original = file.read_bytes()
        file.unlink()
        with pytest.raises((FileNotFoundError, AssertionError)):
            RC._run_phases(args, cell, paths, ("upload-capture",))
        file.write_bytes(original)
    upload_boundary.verify.return_value = ["missing/expected/path"]
    with pytest.raises(AssertionError, match="incomplete"):
        RC._run_phases(args, cell, paths, ("upload-capture",))
    assert not (paths["cell"] / "uploads/upload-capture.json").exists()
    assert not RC._phase_done_path(args, paths, "upload-capture").exists()
    upload_boundary.single.assert_not_called()
    upload_boundary.verify.return_value = []
    upload_boundary.bytes.clear()
    upload_boundary.bulk.reset_mock()
    assert RC._run_phases(args, cell, paths, ("upload-capture",)) == ["upload-capture"]
    receipt = json.loads((paths["cell"] / "uploads/upload-capture.json").read_text())
    assert set(receipt["paths"]) == set(expected)
    assert {
        name: hashlib.sha256(upload_boundary.bytes[name]).hexdigest() for name in receipt["paths"]
    } == expected
    assert upload_boundary.bulk.call_count == 1 and upload_boundary.single.call_count == 2
    assert upload_boundary.bulk.call_args.args[0].is_relative_to(paths["cell"].parent)
    assert RC._phase_complete(args, paths, "upload-capture")
    assert sum(p.endswith("_capture_drops.json") for p in receipt["paths"]) == 5
    assert any(p.endswith("capture_input_validation.json") for p in receipt["paths"])
    assert not any("gpqa" in p for p in receipt["paths"])


def test_fit_production_body_persists_first_unit_then_pauses(generic, monkeypatch):
    args, cell, paths = generic
    args.phase, args.fit_max_units = "fits", 1
    # Numerical test fixture: the production body/estimator runs with an explicitly
    # small feature width; it is not a scientific output or a hardware-sizing pilot.
    monkeypatch.setitem(PC.PANEL, "q3_8b", replace(cell.model, h_dim=4, n_layers=2))
    rng = np.random.default_rng(2588)
    for stage, n in zip(RC.GENERIC_SPLITS, (96, 32, 32), strict=True):
        x = rng.normal(size=(n, 4)).astype(np.float32)
        y = (x + rng.normal(scale=0.7, size=x.shape)).astype(np.float32)
        directory = paths["capture"] / stage / "L00"
        directory.mkdir(parents=True)
        np.savez(
            directory / "shard000.npz",
            row_ids=[f"{stage}_{i}" for i in range(n)],
            x_prompt_last=x,
            y_ans=y,
        )
    monkeypatch.setattr(
        RC, "_phase_complete", create_autospec(RC._phase_complete, return_value=True)
    )
    monkeypatch.setattr(RC, "_await_g2", create_autospec(RC._await_g2, return_value={}))
    monkeypatch.setattr(RC, "_assert_headroom", create_autospec(RC._assert_headroom))
    with pytest.raises(RC.FitPilotPause):
        RC.phase_fits(args, cell, paths)
    unit = paths["fits"] / "percell_prompt_last_L00.json"
    record = json.loads(unit.read_text())
    assert record["n"] == {"tr": 96, "val": 32, "te": 32}
    assert "identity_bias" in record["floors_test_r2"]
    assert "identity_bias" in record["knn_test"]
    assert record["unit_elapsed_s"] >= 0
    first_bytes = unit.read_bytes()
    with pytest.raises(RC.FitPilotPause):
        RC.phase_fits(args, cell, paths)
    assert unit.read_bytes() == first_bytes
    assert not RC._phase_done_path(args, paths, "fits").exists()


def test_immutable_g2_body(generic, monkeypatch, tmp_path):
    import huggingface_hub

    args, cell, paths = generic
    realized = PC.ANCHOR_EXPECTED_R2
    sentinel = {
        "schema_version": PC.G2_SENTINEL_SCHEMA_VERSION,
        "status": "PASS",
        "store_revision_pin_recorded": RC.MF.STORE_REVISION_PIN_7B,
        "expected_r2": realized,
        "realized_r2": realized,
        "abs_deviation": 0,
        "tol": PC.ANCHOR_TOL,
        "production_path": {
            "realized_r2": realized,
            "abs_deviation_vs_pin": 0,
            "tol": PC.ANCHOR_PROD_EQUIV_TOL,
        },
        "meta": {"git_sha": "9b896ccc6b65e1d7322d3c7e67f6fe92883a0f0c"},
    }
    path = tmp_path / "g2.json"
    PC.write_json_atomic(path, sentinel)
    download = create_autospec(huggingface_hub.hf_hub_download, return_value=str(path))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert RC._await_g2(args) == sentinel
    assert download.call_args.kwargs["revision"] == PC.CHAT_G2_REVISION
    assert download.call_args.kwargs["local_files_only"] is True
    snapshot = create_autospec(
        huggingface_hub.snapshot_download,
        return_value=str(RC._local_model_snapshot(cell, paths["cache"])),
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    RC.phase_stage_runtime(args, cell, paths)
    assert snapshot.call_args.kwargs["max_workers"] == 4
    assert snapshot.call_args.kwargs["revision"] == PC.QWEN3_8B_REVISION
    assert snapshot.call_args.kwargs["allow_patterns"] == ["config.json"]
    RC._mark_phase_done(args, cell, paths, "stage-runtime")
    assert RC._phase_complete(args, paths, "stage-runtime")


def test_production_model_file_inventory_pin():
    assert len(RC._CHAT_MODEL_FILES) == 12
    assert sum(RC._CHAT_MODEL_FILES.values()) == 16_397_431_693
