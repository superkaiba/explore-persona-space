"""CPU-only contract tests; no downloaded tokenizer, weights or model requests."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts import context_risk_highrate_capture as capture
from scripts import context_risk_qwen38_impossible_capture as helper


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt is True and enable_thinking is False
        rendered = "<think>\n\n</think>\n" + messages[0]["content"]
        return [ord(char) for char in rendered] if tokenize else rendered

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    from scripts import context_risk_highrate_collect as collection
    from scripts import context_risk_highrate_design as design

    root = tmp_path / "vm"
    rows, prefixes, contexts = [], [], []
    for index in range(30):
        for condition in ("original", "conflicting", "oneoff"):
            messages = [{"role": "user", "content": f"fixture {index} {condition}"}]
            key = capture.digest(messages)
            row = {
                "task_id": f"fixture_{index}",
                "condition": condition,
                "messages": messages,
                "exact_context_sha256": key,
                "public_test_role": "probe_training" if index < 20 else "final_test",
                "phase": "fresh",
                "prompt_variant": "B",
                "sample_id": f"highrate_fresh:B:fixture_{index}:{condition}",
            }
            ids = helper.render_prefix_ids(Tokenizer(), messages, enable_thinking=False)[1]
            rows.append(row)
            prefixes.append(
                {
                    "exact_context_sha256": key,
                    "token_ids": ids,
                    "n_prefix_tokens": len(ids),
                    "prefix_token_ids_sha256": capture.digest(ids),
                }
            )
            s = int(index == 0 and condition == "conflicting")
            u = int(index == 0 and condition == "oneoff")
            contexts.append(
                {
                    "task_id": row["task_id"],
                    "condition": condition,
                    "exact_context_sha256": key,
                    "planned": 4,
                    "realized": 4,
                    "missing": 0,
                    "success": s,
                    "failure": 4 - s - u,
                    "censored": u,
                }
            )
    manifest = root / "manifests/fresh_B.jsonl"
    jsonl(manifest, rows)
    selection = {
        "passed": True,
        "task_roles": {row["task_id"]: row["public_test_role"] for row in rows},
    }
    write(root / "selection.json", selection)
    # Valid censoring need not make an unconditional experiment claim pass.
    write(
        root / "fresh_B/run_result.json",
        {"passed": True, "verification_passed": True, "experiment_passed": False, "censored": 1},
    )
    write(
        root / "fresh_B/prefix_tokens.json",
        {"passed": True, "n_contexts": 90, "contexts": prefixes},
    )
    terminal = {
        "verification_passed": True,
        "phase": "fresh",
        "run_result_sha256": capture.sha256(root / "fresh_B/run_result.json"),
    }
    write(root / "fresh_B/terminal_process.json", terminal)
    checks, calls = [], []

    def load_phase(location, phase):
        checks.append("phase")
        assert Path(location) == root and phase == "fresh"
        return manifest, json.loads((root / "selection.json").read_text()), 4

    def verify(location, phase):
        checks.append("native")
        assert Path(location) == root and phase == "fresh"
        return {
            "schema_version": "context_risk_highrate_native_audit_v1",
            "verification_passed": True,
            "coverage_complete": True,
            "validation_issues": [],
            "duplicate_keys": [],
            "unexpected_keys": [],
            "metadata": {
                "schema_version": "context_risk_highrate_collection_v1",
                "phase": "fresh",
                "arm": "B",
                "epochs": 4,
                "max_attempts": 10,
                "message_limit": 22,
                "max_connections": 16,
                "model": design.MODEL,
                "manifest_sha256": capture.sha256(manifest),
                "selection_sha256": capture.sha256(root / "selection.json"),
                "phase_receipt_sha256": capture.sha256(root / "selection.json"),
                "sources_sha256": design.source_hashes(),
                "plan_sha256": capture.sha256(design.DESIGN / "plan.md"),
            },
            "contexts": contexts,
            "counts": {
                key: sum(row[key] for row in contexts)
                for key in ("planned", "realized", "missing", "success", "failure", "censored")
            },
            "run_result_sha256": capture.sha256(root / "fresh_B/run_result.json"),
            "prefix_tokens_sha256": capture.sha256(root / "fresh_B/prefix_tokens.json"),
            "sources_sha256": design.source_hashes(),
        }

    def verify_terminal(location, phase):
        checks.append("terminal")
        assert Path(location) == root and phase == "fresh"
        return json.loads((root / "fresh_B/terminal_process.json").read_text())

    monkeypatch.setattr(design, "load_phase", load_phase)
    monkeypatch.setattr(collection, "verify_report", verify)
    monkeypatch.setattr(design, "validate_terminal_process", verify_terminal)
    monkeypatch.setattr(capture, "_runtime", lambda: dict(capture.RUNTIME))
    review = tmp_path / "independent_review_fixture.json"
    write(
        review,
        {
            "verdict": "PASS",
            "reviewer": "CPU fixture boundary only",
            "sources_sha256": capture.source_hashes(),
        },
    )
    cfg = OmegaConf.create(
        {
            "root": str(root),
            "manifest_path": str(manifest),
            "output_dir": str(root / "capture"),
            "review": str(review),
            "model": capture.MODEL,
            "capture": capture.CAPTURE,
            "mode": "capture",
            "input_binding_sha256": None,
        }
    )

    def fake_capture(config):
        calls.append("capture")
        out = Path(config.output_dir)
        actual_rows = capture._jsonl(Path(config.manifest_path))
        fingerprint = capture._fingerprint(Path(config.manifest_path), actual_rows)
        chunks = []
        for index, start in enumerate(range(0, 90, 15)):
            sub = actual_rows[start : start + 15]
            stem = f"chunk_{index:04d}"
            metadata = [
                {
                    **{
                        key: row[key]
                        for key in (
                            "task_id",
                            "condition",
                            "exact_context_sha256",
                            "public_test_role",
                        )
                    },
                    **{key: prefix[key] for key in ("prefix_token_ids_sha256", "n_prefix_tokens")},
                }
                for row, prefix in zip(sub, prefixes[start : start + 15], strict=True)
            ]
            np.savez(
                out / f"{stem}.npz",
                activation=np.full((15, 1, 5120), index + 1, dtype=np.float16),
                layers=np.asarray([44], dtype=np.int16),
            )
            jsonl(out / f"{stem}.rows.jsonl", metadata)
            done = {
                "schema_version": "context_risk_qwen38_impossible_capture_chunk_v2",
                "fingerprint": fingerprint,
                "chunk_index": index,
                "n_contexts": 15,
                "npz_sha256": capture.sha256(out / f"{stem}.npz"),
                "rows_sha256": capture.sha256(out / f"{stem}.rows.jsonl"),
            }
            write(out / f"{stem}.done.json", done)
            chunks.append(done)
        report = {
            "schema_version": "context_risk_qwen38_impossible_capture_run_v2",
            "passed": True,
            "fingerprint": fingerprint,
            "n_contexts": 90,
            "model_id": capture.MODEL["id"],
            "model_revision": capture.MODEL["revision"],
            "capture_layers": [44],
            "prefixes_truncated": 0,
            "chunks": chunks,
            "max_sequence_tokens": 32768,
            "minimum_prefix_tokens": min(len(p["token_ids"]) for p in prefixes),
            "maximum_prefix_tokens": max(len(p["token_ids"]) for p in prefixes),
        }
        write(out / "run_result.json", report)
        return report

    real_capture = helper.run_capture
    monkeypatch.setattr(helper, "run_capture", fake_capture)
    return SimpleNamespace(
        cfg=cfg,
        root=root,
        rows=rows,
        prefixes=prefixes,
        contexts=contexts,
        checks=checks,
        calls=calls,
        native=verify,
        terminal=verify_terminal,
        fake_capture=fake_capture,
        real_capture=real_capture,
    )


def prepare(rig):
    prepared = capture.prepare(rig.cfg)
    rig.cfg.input_binding_sha256 = prepared["capture_inputs_sha256"]
    return prepared


def derived_rig(rig, monkeypatch, *, fresh_derived):
    """Expose disclosed native-boundary fixtures to real portable capture bodies."""
    from scripts import context_risk_highrate_design as design
    from scripts import context_risk_highrate_postrun as postrun

    root = rig.root
    review = {
        "verdict": "PASS",
        "reviewer": "explicit derived fixture",
        "sources_sha256": postrun.source_hashes(),
    }
    review_path = root / "setup/postrun_code_review.json"
    write(review_path, review)

    def derived(base):
        return {
            **base,
            "schema_version": postrun.SCHEMA,
            "postrun_sources_sha256": postrun.source_hashes(),
            "postrun_review": review,
            "postrun_review_sha256": capture.sha256(review_path),
            "evidence_verification_passed": True,
            "original_collector_verification_passed": False,
            "original_validation_issues": [{"scope": "explicit-fixture"}],
            "capacity_censors": [{"scope": "explicit-fixture"}],
        }

    screen = derived(rig.native(root, "fresh"))
    screen["metadata"] = {**screen["metadata"], "phase": "screen", "epochs": 2}
    screen["counts"] = {
        "planned": 618,
        "realized": 618,
        "missing": 0,
        "success": 1,
        "failure": 616,
        "censored": 1,
    }
    screen_path = root / "screen_B/postrun_audit.json"
    write(screen_path, screen)
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection.update(
        postrun_screen_audit_sha256=capture.sha256(screen_path),
        postrun_sources_sha256=postrun.source_hashes(),
        postrun_review_sha256=capture.sha256(review_path),
    )
    write(selection_path, selection)
    if fresh_derived:
        write(root / "fresh_B/run_result.json", {"passed": False, "verification_passed": False})
        native = derived(rig.native(root, "fresh"))
        fresh_path = root / "fresh_B/postrun_audit.json"
        write(fresh_path, native)
        native = {**native, "postrun_audit_sha256": capture.sha256(fresh_path)}
        terminal = {
            "schema_version": postrun.TERMINAL_SCHEMA,
            "verification_passed": True,
            "phase": "fresh",
            "original_exit_code": 1,
            "run_result_sha256": capture.sha256(root / "fresh_B/run_result.json"),
            "postrun_audit_sha256": native["postrun_audit_sha256"],
        }
        write(root / "fresh_B/terminal_process.json", terminal)
    else:
        native = rig.native(root, "fresh")
        terminal = rig.terminal(root, "fresh")
    monkeypatch.setattr(
        postrun, "verify_report", create_autospec(postrun.verify_report, return_value=native)
    )
    monkeypatch.setattr(
        postrun,
        "validate_terminal_process",
        create_autospec(postrun.validate_terminal_process, return_value=terminal),
    )
    monkeypatch.setattr(
        postrun,
        "validate_selection",
        create_autospec(postrun.validate_selection, return_value=selection),
    )
    # Every boundary reached by these new body tests has an exact callable signature.
    monkeypatch.setattr(
        design, "load_phase", create_autospec(design.load_phase, side_effect=design.load_phase)
    )
    monkeypatch.setattr(
        capture, "_runtime", create_autospec(capture._runtime, return_value=dict(capture.RUNTIME))
    )
    monkeypatch.setattr(
        helper, "run_capture", create_autospec(rig.real_capture, side_effect=rig.fake_capture)
    )
    return native


@pytest.mark.parametrize("fresh_derived", [False, True])
def test_derived_screen_and_fresh_portable_capture_roundtrip(
    rig, tmp_path, monkeypatch, fresh_derived
):
    native = derived_rig(rig, monkeypatch, fresh_derived=fresh_derived)
    before = (rig.root / "fresh_B/run_result.json").read_bytes()
    prepared = prepare(rig)
    extra = {"screen_B/postrun_audit.json", "setup/postrun_code_review.json"}
    if fresh_derived:
        extra.add("fresh_B/postrun_audit.json")
    assert extra <= set(prepared["stage_relative_paths"])
    pod = tmp_path / "pod"
    for name in prepared["stage_relative_paths"]:
        destination = pod / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rig.root / name, destination)
    cfg = copy.deepcopy(rig.cfg)
    cfg.root, cfg.output_dir = str(pod), str(pod / "capture")
    cfg.manifest_path = str(pod / "manifests/fresh_B.jsonl")
    captured = capture.run(cfg)
    assert captured["fresh_evidence"]["generation_validation"] == native
    assert captured["passed"] and len(captured["chunk_files_sha256"]) == 18
    shutil.copytree(pod / "capture", rig.root / "capture")
    assert capture.validate_binding(rig.root / "capture")["passed"]
    assert (rig.root / "fresh_B/run_result.json").read_bytes() == before


@pytest.mark.parametrize(
    "fault", ["wrong_phase", "changed_review", "no_censors", "original_pass", "wrong_selection"]
)
def test_portable_derived_semantics_fail_even_with_recomputed_file_hashes(rig, monkeypatch, fault):
    native = derived_rig(rig, monkeypatch, fresh_derived=True)
    path = rig.root / "fresh_B/postrun_audit.json"
    value = json.loads(path.read_text())
    if fault == "wrong_phase":
        value["metadata"]["phase"] = "screen"
    elif fault == "changed_review":
        value["postrun_review"]["verdict"] = "REVISE"
    elif fault == "no_censors":
        value["capacity_censors"] = []
    elif fault == "original_pass":
        value["original_collector_verification_passed"] = True
    else:
        selection_path = rig.root / "selection.json"
        selected = json.loads(selection_path.read_text())
        selected["postrun_screen_audit_sha256"] = "0" * 64
        write(selection_path, selected)
    write(path, value)
    # Recompute the outer hash deliberately so the semantic guard is exercised.
    native = {**value, "postrun_audit_sha256": capture.sha256(path)}
    selection = json.loads((rig.root / "selection.json").read_text())
    with pytest.raises(ValueError):
        capture._postrun_files(rig.root, selection, native)


def test_vm_prepare_pod_capture_and_vm_consume_with_valid_censor(rig):
    prepared = prepare(rig)
    assert set(rig.checks) == {"phase", "native", "terminal"}
    first = capture.run(rig.cfg)
    second = capture.run(rig.cfg)
    assert first == second and rig.calls == ["capture"]
    assert first["fresh_evidence"]["generation_validation"]["counts"]["censored"] == 1
    assert len(first["chunk_files_sha256"]) == 18
    assert capture.validate_binding(rig.root / "capture") == first
    assert rig.checks.count("native") >= 2
    assert prepare(rig) == prepared


def test_pod_path_relocation_never_dereferences_vm_native_paths(rig, tmp_path, monkeypatch):
    prepared = prepare(rig)
    pod = tmp_path / "pod"
    for name in prepared["stage_relative_paths"]:
        destination = pod / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rig.root / name, destination)
    cfg = copy.deepcopy(rig.cfg)
    cfg.root = str(pod)
    cfg.output_dir = str(pod / "capture")
    cfg.manifest_path = str(pod / "manifests/fresh_B.jsonl")
    before = list(rig.checks)
    assert capture.run(cfg)["passed"]
    assert rig.checks == before
    shutil.copytree(pod / "capture", rig.root / "capture")
    assert capture.validate_binding(rig.root / "capture")["passed"]


@pytest.mark.parametrize("kind", ["native", "terminal", "incomplete", "bad_counts"])
def test_prepare_requires_complete_validated_native_and_process(rig, monkeypatch, kind):
    from scripts import context_risk_highrate_collect as collection

    if kind == "native":

        def reject(*args):
            raise ValueError("Invalid native evidence")

        monkeypatch.setattr(collection, "verify_report", reject)
    elif kind == "terminal":
        path = rig.root / "fresh_B/terminal_process.json"
        receipt = json.loads(path.read_text())
        receipt["verification_passed"] = False
        write(path, receipt)
    elif kind == "incomplete":
        rig.contexts[0]["missing"] = 1
        rig.contexts[0]["realized"] = 3
    else:
        rig.contexts[0]["failure"] = -1
    with pytest.raises(ValueError):
        prepare(rig)
    assert not rig.calls and not (rig.root / "capture_inputs.json").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "missing_sha",
        "wrong_sha",
        "review",
        "source",
        "manifest",
        "prefix_tokens",
        "selection",
        "generation",
        "terminal_process",
    ],
)
def test_staging_and_review_drift_rejected_before_gpu(rig, monkeypatch, kind):
    prepare(rig)
    if kind == "missing_sha":
        rig.cfg.input_binding_sha256 = None
    elif kind == "wrong_sha":
        rig.cfg.input_binding_sha256 = "0" * 64
    elif kind == "review":
        write(
            Path(rig.cfg.review), {"verdict": "REVISE", "sources_sha256": capture.source_hashes()}
        )
    elif kind == "source":
        monkeypatch.setattr(capture, "source_hashes", lambda: {"drift": "bad"})
    else:
        path = rig.root / capture.INPUT_PATHS[kind]
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        capture.run(rig.cfg)
    assert not rig.calls


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_context",
        "wrong_condition",
        "message_hash",
        "bad_token_hash",
        "bad_token_length",
        "negative_token",
        "overlong",
    ],
)
def test_bad_context_or_token_evidence_rejected_at_prepare(rig, kind):
    if kind in {"duplicate_context", "wrong_condition", "message_hash"}:
        rows = copy.deepcopy(rig.rows)
        if kind == "duplicate_context":
            rows[-1] = rows[0]
        elif kind == "wrong_condition":
            rows[0]["condition"] = "unknown"
        else:
            rows[0]["messages"][0]["content"] += " changed"
        jsonl(rig.root / "manifests/fresh_B.jsonl", rows)
    else:
        prefix = rig.prefixes[0]
        if kind == "bad_token_hash":
            prefix["prefix_token_ids_sha256"] = "bad"
        elif kind == "bad_token_length":
            prefix["n_prefix_tokens"] += 1
        else:
            prefix["token_ids"] = [-1] if kind == "negative_token" else [1] * 32769
            prefix["n_prefix_tokens"] = len(prefix["token_ids"])
            prefix["prefix_token_ids_sha256"] = capture.digest(prefix["token_ids"])
        write(
            rig.root / "fresh_B/prefix_tokens.json",
            {"passed": True, "n_contexts": 90, "contexts": rig.prefixes},
        )
    with pytest.raises(ValueError):
        prepare(rig)
    assert not rig.calls


def test_orphan_chunks_and_changed_resume_binding_rejected(rig):
    prepare(rig)
    out = rig.root / "capture"
    write(out / "chunk_0000.done.json", {"orphan": True})
    with pytest.raises(ValueError, match="empty output"):
        capture.run(rig.cfg)
    (out / "chunk_0000.done.json").unlink()
    capture.run(rig.cfg)
    launch = json.loads((out / "capture_launch_binding.json").read_text())
    launch["runtime"]["torch"] = "wrong"
    write(out / "capture_launch_binding.json", launch)
    with pytest.raises(ValueError, match="resume"):
        capture.run(rig.cfg)
    assert rig.calls == ["capture"]


def resign(out):
    """A maliciously rehashed payload must still fail semantic checks."""
    report = json.loads((out / "run_result.json").read_text())
    chunks = []
    for path in sorted(out.glob("chunk_*.done.json")):
        done = json.loads(path.read_text())
        stem = path.name.removesuffix(".done.json")
        done["npz_sha256"] = capture.sha256(out / f"{stem}.npz")
        done["rows_sha256"] = capture.sha256(out / f"{stem}.rows.jsonl")
        write(path, done)
        chunks.append(done)
    report["chunks"] = chunks
    write(out / "run_result.json", report)
    path = out / "capture_binding.json"
    if path.exists():
        binding = json.loads(path.read_text())
        binding["run_result_sha256"] = capture.sha256(out / "run_result.json")
        binding["chunk_files_sha256"] = {p.name: capture.sha256(p) for p in out.glob("chunk_*")}
        write(path, binding)


@pytest.mark.parametrize(
    "kind",
    [
        "nan",
        "layer",
        "dtype",
        "shape",
        "row_order",
        "prefix",
        "extra_chunk",
        "coverage",
        "model",
        "position",
    ],
)
def test_semantic_corruption_rejected_despite_matching_payload_hashes(rig, kind):
    prepare(rig)
    capture.run(rig.cfg)
    out = rig.root / "capture"
    if kind in {"nan", "layer", "dtype", "shape"}:
        path = out / "chunk_0000.npz"
        with np.load(path) as arrays:
            values, layers = arrays["activation"], arrays["layers"]
        if kind == "nan":
            values[0, 0, 0] = np.nan
        elif kind == "layer":
            layers = np.array([43], dtype=np.int16)
        elif kind == "dtype":
            values = values.astype(np.float32)
        else:
            values = values[:, :, :5119]
        np.savez(path, activation=values, layers=layers)
    elif kind in {"row_order", "prefix"}:
        path = out / "chunk_0000.rows.jsonl"
        rows = capture._jsonl(path)
        if kind == "row_order":
            rows[0], rows[1] = rows[1], rows[0]
        else:
            rows[0]["prefix_token_ids_sha256"] = "wrong"
        jsonl(path, rows)
    elif kind == "extra_chunk":
        (out / "chunk_extra").write_text("extra")
    elif kind == "position":
        path = out / "capture_binding.json"
        value = json.loads(path.read_text())
        value["activation_position"] = "mean_prompt"
        write(path, value)
    else:
        path = out / "run_result.json"
        value = json.loads(path.read_text())
        value["n_contexts" if kind == "coverage" else "model_revision"] = "wrong"
        write(path, value)
    resign(out)
    with pytest.raises(ValueError):
        capture.validate_binding(out)


def test_interrupted_resume_checks_completed_shard_before_loading_model(rig):
    prepare(rig)
    capture.run(rig.cfg)
    out = rig.root / "capture"
    (out / "capture_binding.json").unlink()
    path = out / "chunk_0000.rows.jsonl"
    rows = capture._jsonl(path)
    rows[0]["n_prefix_tokens"] += 1
    jsonl(path, rows)
    resign(out)
    with pytest.raises(ValueError, match="tokens differ"):
        capture.run(rig.cfg)
    assert rig.calls == ["capture"]


def test_changed_inputs_during_capture_never_write_pass(rig, monkeypatch):
    prepare(rig)

    def changing_capture(cfg):
        result = rig.fake_capture(cfg)
        path = rig.root / "fresh_B/prefix_tokens.json"
        path.write_text(path.read_text() + " ")
        return result

    monkeypatch.setattr(helper, "run_capture", changing_capture)
    with pytest.raises(ValueError):
        capture.run(rig.cfg)
    assert not (rig.root / "capture/capture_binding.json").exists()


def test_actual_imported_helper_drift_is_rejected(tmp_path, monkeypatch):
    import explore_persona_space.analysis.extraction as extraction

    path = tmp_path / "different_source.py"
    path.write_text("different source")
    monkeypatch.setattr(extraction, "__file__", str(path))
    with pytest.raises(ValueError, match="Imported helper differs"):
        capture.imported_source_hashes()


def test_jsonl_reader_preserves_raw_unicode_separators(tmp_path):
    path = tmp_path / "unicode.jsonl"
    rows = [{"text": "first\u2028second\u2029third"}, {"text": "last"}]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    assert capture._jsonl(path) == rows


class Block(torch.nn.Module):
    def forward(self, hidden):
        return (hidden + 1,)


class CpuModel(torch.nn.Module):
    """Faithful multimodal-wrapper hook seam; no learned weights or API calls."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([Block() for _ in range(64)])
        self.config = SimpleNamespace(text_config=SimpleNamespace(pad_token_id=0, eos_token_id=0))
        self.batches = []

    def forward(self, input_ids, attention_mask, *, output_hidden_states, logits_to_keep, **kwargs):
        assert output_hidden_states is False and logits_to_keep == 1
        self.batches.append(attention_mask.sum(1).tolist())
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 5120)
        for block in self.model.language_model.layers:
            hidden = block(hidden)[0]
        return SimpleNamespace(logits=None)


def test_real_helper_render_batch_hook_and_saved_shards_on_cpu(rig, monkeypatch):
    prepare(rig)
    model = CpuModel()

    def load(cfg):
        assert OmegaConf.to_container(cfg.model) == capture.MODEL
        return model, Tokenizer(), 1

    monkeypatch.setattr(helper, "run_capture", rig.real_capture)
    monkeypatch.setattr(helper, "_load_model_and_tokenizer", load)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    assert capture.run(rig.cfg)["passed"]
    assert any(len(batch) == 2 and batch[0] != batch[1] for batch in model.batches)
    assert max(map(len, model.batches)) == 2
    for index, start in enumerate(range(0, 90, 15)):
        with np.load(rig.root / f"capture/chunk_{index:04d}.npz") as arrays:
            for offset, prefix in enumerate(rig.prefixes[start : start + 15]):
                expected = prefix["token_ids"][-1] + 45  # block44 output, hidden_states[45].
                assert np.all(arrays["activation"][offset, 0] == expected)
    assert capture.validate_binding(rig.root / "capture")["passed"]
