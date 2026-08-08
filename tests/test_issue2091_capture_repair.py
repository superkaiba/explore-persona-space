"""#2091 P2 resume-offset loss — realized-offset resume + append-only repair.

Pins the crash-fix for the 448-row/job capture loss: (1) the resume cursor is
derived from the REALIZED ``row_index`` line counts of resumed shards (a
64-row pilot shard resumes at offset 64, never at an assumed full 512); (2) a
non-prefix resumed shard fails LOUD instead of silently dropping rows; (3)
capture end reconciles realized-vs-expected rows fail-loud; (4) the repair
path captures exactly the missing identity set, append-only and idempotent.

SYNTHETIC FIXTURES ONLY (real BPE tokenizer trained in-test, from-config tiny
Qwen2 on CPU — the test_issue1739_dataplane pattern); no network, no GPU. All
fixture text is neutral synthetic placeholder content.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739 import capture, store_io

# Tiny-real dims: asymmetric so a transposed-shape bug cannot hide.
TINY_LAYERS = 2
TINY_DIM = 32
SHARD_ROWS = 8  # test-scale stand-in for the production 512
FP = "fp-2091-test"

QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_tokenizer():
    """REAL BPE tokenizer (trained in-test on synthetic text; no network)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = [
        "<|im_start|>user assistant system <|im_end|>",
        "placeholder question about topic alpha and topic beta",
        "a short synthetic reply that mentions nothing of note",
        "another synthetic reply about topic gamma",
    ] * 4
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=["[PAD]", "[UNK]"])
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")
    fast.chat_template = QWEN_CHAT_TEMPLATE
    return fast


@pytest.fixture(scope="module")
def tiny_model(tiny_tokenizer):
    """From-config tiny Qwen2 (real library class, random weights, CPU)."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tiny_tokenizer) + 8,
        hidden_size=TINY_DIM,
        intermediate_size=64,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def _write_rollout_files(rollout_dir: Path, n: int) -> list[Path]:
    """n labeling-shape rollout JSONs with distinct context ids (sorted order
    == index order, mirroring run_job's ``sorted(rollout_dir.glob(...))``)."""
    rollout_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        cid = f"ctx-{i:03d}"
        path = rollout_dir / f"{cid}_seed0.json"
        path.write_text(
            json.dumps(
                {
                    "context_id": cid,
                    "behavior": "sycophancy",
                    "split": "train",
                    "rung": "train",
                    "group_key": f"group-{i % 3}",
                    "rollout_k": 0,
                    "query": f"placeholder question number {i}",
                    "prefix_text": "",
                    "prompt_text": f"placeholder question about topic alpha number {i} ",
                    "completion": f"a short synthetic reply number {i}",
                }
            )
        )
        paths.append(path)
    return sorted(paths)


def _cap_kwargs(store_dir: Path, tiny_model, tiny_tokenizer) -> dict:
    return dict(
        store_dir=store_dir,
        model=tiny_model,
        tokenizer=tiny_tokenizer,
        n_layers=TINY_LAYERS,
        hidden_dim=TINY_DIM,
        device="cpu",
        batch_size=4,
        shard_rows=SHARD_ROWS,
        fingerprint=FP,
    )


def _store_context_ids(store_dir: Path) -> list[str]:
    ids = []
    for idx in capture.store_shard_indices(store_dir):
        ids.extend(m["context_id"] for m in capture.read_shard_index(store_dir, idx))
    return ids


# ---------------------------------------------------------------------------
# (1) realized-offset resume — the exact 64-vs-512 shape at test scale (3 vs 8)
# ---------------------------------------------------------------------------


def test_partial_pilot_prefix_shard_resumes_at_realized_offset(
    tiny_tokenizer, tiny_model, tmp_path, caplog
):
    """A resumed PARTIAL shard (3 realized rows < shard_rows=8) advances the
    row cursor by its REALIZED count: no row is lost, the resumed shard is
    never rewritten, and the realized store reconciles exactly. Fails
    pre-fix: the old fixed-grid arithmetic skipped rows 3..7 and realized
    17 of 20 rows (the #2091 448-row/job shape)."""
    paths = _write_rollout_files(tmp_path / "labeling", 20)
    store = tmp_path / "store"
    kwargs = _cap_kwargs(store, tiny_model, tiny_tokenizer)

    # Pilot: the first 3 files (a sorted PREFIX of the full row list).
    pilot = capture.capture_rollout_files(paths[:3], **kwargs)
    assert pilot["realized_total_rows"] == 3
    shard00_index = (store / "row_index_shard00.jsonl").read_bytes()
    shard00_npy = (store / "t1_L00_shard00.npy").read_bytes()

    with caplog.at_level(logging.INFO, logger=capture.logger.name):
        manifest = capture.capture_rollout_files(paths, **kwargs)

    # Fix-engaged signal: the realized resumed-shard count + derived offset.
    assert "resume: shard 00 done with 3 realized rows -> row offset 3" in caplog.text
    assert manifest["n_shards_resumed"] == 1
    assert manifest["n_rows_captured"] == 17  # 20 - 3 resumed, NOT 20 - 8
    assert manifest["realized_total_rows"] == 20
    assert manifest["n_duplicate_rows"] == 0
    assert manifest["n_missing_rows"] == 0
    # Realized per-shard breakdown: 3 (pilot) + 8 + 8 + 1.
    assert manifest["per_shard_rows"] == {"00": 3, "01": 8, "02": 8, "03": 1}

    # The resumed shard was never rewritten (append-only resume).
    assert (store / "row_index_shard00.jsonl").read_bytes() == shard00_index
    assert (store / "t1_L00_shard00.npy").read_bytes() == shard00_npy

    # Consumer round-trip: every context exactly once, in row-list order.
    out, meta = store_io.load_summaries(
        store, ("t1",), tuple(range(TINY_LAYERS)), hidden_dim=TINY_DIM
    )
    assert out[("t1", 0)].shape == (20, TINY_DIM)
    assert [m["context_id"] for m in meta] == [f"ctx-{i:03d}" for i in range(20)]


def test_full_shard_resume_unchanged(tiny_tokenizer, tiny_model, tmp_path):
    """The non-pathological case: a resumed FULL shard (8 == shard_rows)
    behaves exactly as before — offset 8, remaining 12 rows captured."""
    paths = _write_rollout_files(tmp_path / "labeling", 20)
    store = tmp_path / "store"
    kwargs = _cap_kwargs(store, tiny_model, tiny_tokenizer)

    capture.capture_rollout_files(paths[:8], **kwargs)
    manifest = capture.capture_rollout_files(paths, **kwargs)
    assert manifest["n_shards_resumed"] == 1
    assert manifest["n_rows_captured"] == 12
    assert manifest["realized_total_rows"] == 20
    assert manifest["per_shard_rows"] == {"00": 8, "01": 8, "02": 4}
    assert _store_context_ids(store) == [f"ctx-{i:03d}" for i in range(20)]


def test_resume_identity_mismatch_fails_loud(tiny_tokenizer, tiny_model, tmp_path):
    """The ACTUAL #2091 pilot shape: the pilot's shard00 rows are NOT a prefix
    of the full run's row list (differently-ordered subset) — the resume must
    fail LOUD naming the repair path, never silently mis-align."""
    paths = _write_rollout_files(tmp_path / "labeling", 20)
    store = tmp_path / "store"
    kwargs = _cap_kwargs(store, tiny_model, tiny_tokenizer)

    # Pilot over a NON-prefix subset (ctx 5, 9, 14).
    capture.capture_rollout_files([paths[5], paths[9], paths[14]], **kwargs)
    with pytest.raises(RuntimeError, match=r"resume mismatch at shard00.*repair_missing_rows"):
        capture.capture_rollout_files(paths, **kwargs)


# ---------------------------------------------------------------------------
# (2) fail-loud completeness reconciliation
# ---------------------------------------------------------------------------


def test_completeness_assert_fails_loud_on_short_store(tiny_tokenizer, tiny_model, tmp_path):
    """assert_store_complete names BOTH totals + the per-shard breakdown on a
    short store — the check whose absence let 25% loss pass as success."""
    paths = _write_rollout_files(tmp_path / "labeling", 20)
    store = tmp_path / "store"
    kwargs = _cap_kwargs(store, tiny_model, tiny_tokenizer)
    capture.capture_rollout_files(paths[:12], **kwargs)  # 12 of 20 rows

    rows, _ = capture.load_capture_rows(paths, tiny_tokenizer)
    expected_meta = [meta for _, meta in rows]
    with pytest.raises(RuntimeError) as exc:
        capture.assert_store_complete(store, expected_meta)
    msg = str(exc.value)
    assert "realized 12 rows" in msg
    assert "expected 20" in msg
    assert "8 expected rows MISSING" in msg
    assert "per-shard rows" in msg and "'00': 8" in msg and "'01': 4" in msg

    # And the passing direction returns the reconciliation digest.
    rows12, _ = capture.load_capture_rows(paths[:12], tiny_tokenizer)
    digest = capture.assert_store_complete(store, [m for _, m in rows12])
    assert digest["realized_total_rows"] == 12 and digest["n_missing_rows"] == 0


# ---------------------------------------------------------------------------
# (3) append-only repair: exact missing set, idempotent no-op
# ---------------------------------------------------------------------------


def _build_incident_shaped_store(paths, store, tiny_model, tiny_tokenizer):
    """Reconstruct the #2091 damage at test scale: a non-prefix 3-row pilot
    shard00 ({5, 9, 14}) plus full-run shards 01/02 holding rows[8:20] (the
    old bug's offset-512 analogue: rows 0..7 minus the pilot rows are LOST,
    and pilot rows 9/14 are duplicated)."""
    kwargs = _cap_kwargs(store, tiny_model, tiny_tokenizer)
    capture.capture_rollout_files([paths[5], paths[9], paths[14]], **kwargs)
    rows, _ = capture.load_capture_rows(paths[8:20], tiny_tokenizer)
    for shard_idx, chunk in ((1, rows[:8]), (2, rows[8:])):
        summaries, positions = capture.capture_batch(
            [p["prefix_text"] for p, _ in chunk],
            [p["prompt_text"] for p, _ in chunk],
            [p["completion"] for p, _ in chunk],
            model=tiny_model,
            tokenizer=tiny_tokenizer,
            n_layers=TINY_LAYERS,
            hidden_dim=TINY_DIM,
            device="cpu",
            batch_size=4,
        )
        meta_rows = [dict(m, **pos) for (_, m), pos in zip(chunk, positions, strict=True)]
        capture.write_store_shard(store, shard_idx, summaries, meta_rows)
        (store / f"_capture_meta_shard{shard_idx:02d}.json").write_text(
            json.dumps({"fingerprint": FP, "n_rows": len(meta_rows)})
        )


def test_repair_captures_exactly_missing_then_noop(
    tiny_tokenizer, tiny_model, tmp_path, monkeypatch
):
    """The repair diff selects exactly the missing identity set, appends it as
    NEW shards after the highest existing index (existing shards untouched),
    reconciles set-complete with duplicates counted, and re-runs as a clean
    no-op (zero model forwards, zero file writes)."""
    paths = _write_rollout_files(tmp_path / "labeling", 20)
    store = tmp_path / "store"
    _build_incident_shaped_store(paths, store, tiny_model, tiny_tokenizer)
    shard01_bytes = (store / "row_index_shard01.jsonl").read_bytes()

    kwargs = _cap_kwargs(store, tiny_model, tiny_tokenizer)
    manifest = capture.repair_missing_rows(paths, **kwargs)

    missing_ids = [f"ctx-{i:03d}" for i in (0, 1, 2, 3, 4, 6, 7)]
    assert manifest["repair"]["n_missing_found"] == 7
    assert manifest["repair"]["n_missing_captured"] == 7
    assert manifest["repair"]["shards_appended"] == ["03"]
    assert manifest["realized_total_rows"] == 22  # 20 expected + 2 pilot dups (9, 14)
    assert manifest["n_expected_rows"] == 20
    assert manifest["n_duplicate_rows"] == 2
    assert manifest["n_missing_rows"] == 0 and manifest["n_unexpected_rows"] == 0
    # The appended shard holds exactly the missing rows, in row-list order.
    assert [m["context_id"] for m in capture.read_shard_index(store, 3)] == missing_ids
    # Existing shards byte-untouched; every expected context now present.
    assert (store / "row_index_shard01.jsonl").read_bytes() == shard01_bytes
    assert set(_store_context_ids(store)) == {f"ctx-{i:03d}" for i in range(20)}
    # The store manifest records the repair + realized reconciliation.
    disk_manifest = json.loads((store / "_capture_manifest.json").read_text())
    assert disk_manifest["repairs"][0]["n_missing_captured"] == 7
    assert disk_manifest["realized_total_rows"] == 22
    assert disk_manifest["n_expected_rows"] == 20

    # Idempotent re-run: nothing missing -> no forwards, no writes.
    manifest_bytes = (store / "_capture_manifest.json").read_bytes()
    real_capture_batch = capture.capture_batch
    calls = {"n": 0}

    def counting_capture_batch(*a, **k):
        calls["n"] += 1
        return real_capture_batch(*a, **k)

    monkeypatch.setattr(capture, "capture_batch", counting_capture_batch)
    noop = capture.repair_missing_rows(paths, **kwargs)
    assert noop["repair"]["n_missing_found"] == 0
    assert noop["repair"]["shards_appended"] == []
    assert calls["n"] == 0
    assert capture.store_shard_indices(store) == [0, 1, 2, 3]
    assert (store / "_capture_manifest.json").read_bytes() == manifest_bytes


def test_repair_refuses_empty_store(tiny_tokenizer, tiny_model, tmp_path):
    paths = _write_rollout_files(tmp_path / "labeling", 4)
    with pytest.raises(RuntimeError, match="no realized shards"):
        capture.repair_missing_rows(
            paths, **_cap_kwargs(tmp_path / "empty_store", tiny_model, tiny_tokenizer)
        )


# ---------------------------------------------------------------------------
# (4) pod-driver wiring (--repair flag, guards, done-record resume, payload)
# ---------------------------------------------------------------------------


@pytest.fixture()
def pod_mod():
    import scripts.issue2091_pod as pod

    return pod


def _repair_args(pod_mod, tmp_path, extra: list[str] | None = None):
    return pod_mod._parse_args(
        [
            "--mode",
            "dispatch",
            "--repair",
            "--out-root",
            str(tmp_path / "out"),
            "--store-root",
            str(tmp_path / "store"),
            "--stage-root",
            str(tmp_path / "stage"),
            *(extra or []),
        ]
    )


def test_child_command_threads_repair_flag(pod_mod, tmp_path):
    args = _repair_args(pod_mod, tmp_path, ["--repair-revision", "abc123def"])
    cmd, env = pod_mod.child_command(args, "syc_aita", "2")
    assert "--repair" in cmd
    assert cmd[cmd.index("--repair-revision") + 1] == "abc123def"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    # Production (non-repair) argv unchanged.
    cmd_prod, _ = pod_mod.child_command(
        pod_mod._parse_args(["--mode", "dispatch", "--out-root", str(tmp_path / "out")]),
        "syc_aita",
        "2",
    )
    assert "--repair" not in cmd_prod


def test_main_repair_guards(pod_mod, tmp_path):
    with pytest.raises(SystemExit, match="incompatible with --limit"):
        pod_mod.main(["--mode", "dispatch", "--repair", "--limit", "4"])
    with pytest.raises(SystemExit, match="incompatible with --skip-capture"):
        pod_mod.main(["--mode", "dispatch", "--repair", "--skip-capture"])


def test_repair_job_resumes_on_done_record(pod_mod, tmp_path):
    """An existing _repair_done.json short-circuits the job child (idempotent
    per-job resume) — no staging, no model load, no network."""
    args = pod_mod._parse_args(
        [
            "--mode",
            "job",
            "--repair",
            "--rungjob",
            "syc_aita",
            "--out-root",
            str(tmp_path / "out"),
            "--store-root",
            str(tmp_path / "store"),
            "--stage-root",
            str(tmp_path / "stage"),
        ]
    )
    done = pod_mod.repair_done_path(args, "syc_aita")
    done.parent.mkdir(parents=True, exist_ok=True)
    record = {"rungjob": "syc_aita", "mode": "repair_capture", "n_missing_captured": 0}
    done.write_text(json.dumps(record))
    assert pod_mod.repair_job(args) == record


def test_repair_job_end_to_end_local_cpu(
    pod_mod, tiny_tokenizer, tiny_model, tmp_path, monkeypatch
):
    """The production repair-job body end to end on CPU: local-first staging
    (pre-seeded pack mirror + store sidecars, so the HF-fetch branches are
    skipped by their own predicates), unpack, fingerprint read, diff + capture
    of exactly the missing rows, record write. Fakes ONLY at the GPU boundary
    (tiny model/tokenizer + tiny layer constants) and the Hub boundary
    (--skip-upload; local files pre-seeded)."""
    import scripts.issue1739_pack as pack_mod
    from explore_persona_space.experiments.issue_1739 import constants, generation

    name = "syc_train"  # gen_behavior == "sycophancy"
    args = pod_mod._parse_args(
        [
            "--mode",
            "job",
            "--repair",
            "--rungjob",
            name,
            "--out-root",
            str(tmp_path / "out"),
            "--store-root",
            str(tmp_path / "store"),
            "--stage-root",
            str(tmp_path / "stage"),
            "--device",
            "cpu",
            "--skip-upload",
        ]
    )

    # Source labeling tree -> packed mirror at the local-first staging path.
    src_root = tmp_path / "src_labeling"
    paths = _write_rollout_files(src_root / "sycophancy", 20)
    pack_mirror = args.stage_root / f"{args.hf_prefix}/raw_completions/greedy/{name}"
    pack_mod.pack_raw_tree(src_root, pack_mirror)

    # Damaged store (incident shape) at the job's store dir, manifest included.
    store = pod_mod.job_store_dir(args, name)
    _build_incident_shaped_store(paths, store, tiny_model, tiny_tokenizer)

    monkeypatch.setattr(generation, "get_tokenizer", lambda *a, **k: tiny_tokenizer)
    monkeypatch.setattr(capture, "load_capture_model", lambda **k: tiny_model)
    monkeypatch.setattr(constants, "N_LAYERS", TINY_LAYERS)
    monkeypatch.setattr(constants, "HIDDEN_DIM", TINY_DIM)
    # Production shard_rows (512) would put all 7 missing rows in one shard
    # anyway; keep the default to exercise the production kwargs verbatim.

    # syc_train carries probe_behavior="sycophancy": stage 6 banked probe rows
    # (the pilot-shaped probe store holds only the first 2 — measured 64/150
    # per behavior in production) so the probe repair leg runs too.
    from scripts.issue2091_stage_contexts import shard_rows

    probe_rows = [
        {
            "context_id": f"probe-{i:03d}",
            "behavior": "sycophancy",
            "rung": "train",
            "group_key": f"group-{i % 2}",
            "rollout_k": 0,
            "prefix_text": "",
            "prompt_text": f"placeholder probe question number {i} ",
            "completion": f"another synthetic reply about topic gamma {i}",
            "meta": {},
        }
        for i in range(6)
    ]
    shard_rows(
        probe_rows,
        args.stage_root / f"{args.hf_prefix}/contexts/parity_probe/sycophancy",
        "probe",
    )
    pilot_probe_paths = pod_mod.write_probe_payload_files(args, "sycophancy", probe_rows[:2])
    probe_store = pod_mod.probe_store_dir(args, "sycophancy")
    capture.capture_rollout_files(
        sorted(pilot_probe_paths),
        **{**_cap_kwargs(probe_store, tiny_model, tiny_tokenizer), "fingerprint": f"{FP}-probe"},
    )

    record = pod_mod.repair_job(args)
    assert record["probe_repair"]["n_missing_found"] == 4
    assert record["probe_repair"]["n_missing_captured"] == 4
    assert record["probe_repair"]["realized_total_rows"] == 6
    assert record["probe_repair"]["n_expected_rows"] == 6
    assert record["probe_repair"]["n_duplicate_rows"] == 0
    assert record["probe_repair"]["fingerprint"] == f"{FP}-probe"
    assert set(_store_context_ids(probe_store)) == {f"probe-{i:03d}" for i in range(6)}
    assert record["n_missing_found"] == 7
    assert record["n_missing_captured"] == 7
    assert record["realized_total_rows"] == 22
    assert record["n_expected_rows"] == 20
    assert record["n_duplicate_rows"] == 2
    assert record["shards_appended"] == ["03"]
    assert record["fingerprint"] == FP
    assert pod_mod.repair_done_path(args, name).is_file()
    assert set(_store_context_ids(store)) == {f"ctx-{i:03d}" for i in range(20)}

    # Idempotent at the job grain: a re-run resumes on the done record.
    assert pod_mod.repair_job(args) == record


def test_repair_hub_call_shapes_bind(pod_mod):
    """The network-fenced HF-fetch branches' helper calls bind against the
    live hub signatures (the #1332 fenced-call arity class — import resolution
    alone green-lights a keyword mismatch)."""
    import inspect

    from explore_persona_space.orchestrate import hub

    inspect.signature(hub.stage_hub_prefix).bind(
        "repo", "prefix", Path("/tmp/x"), repo_type="dataset", revision=None
    )
    inspect.signature(hub.stage_hub_file).bind(
        "repo", "rel/path", Path("/tmp/x/f"), repo_type="dataset", revision=None
    )
    inspect.signature(hub.list_hf_files_under_path).bind(
        object(), "repo", "prefix", repo_type="dataset", revision=None
    )


def test_build_repair_results_payload_keys(pod_mod, tmp_path):
    args = _repair_args(pod_mod, tmp_path)
    records = {
        "syc_aita": {
            "n_missing_found": 484,
            "n_missing_captured": 484,
            "realized_total_rows": 1340,
            "n_expected_rows": 1304,
            "n_duplicate_rows": 36,
            "shards_appended": ["03"],
            "hf_store_prefix": "issue2091_decode/capture_store/greedy_syc_aita",
            "hf_text_prefix": "issue2091_decode/raw_completions/greedy/syc_aita",
            "fingerprint": "fp",
        }
    }
    payload = pod_mod.build_repair_results_payload(args, {"model": "m"}, records, 0.5)
    assert set(payload) == {
        "eval_numbers",
        "eval_paths",
        "reproducibility_card",
        "wandb_url",
        "hf_hub_url",
        "worktree_path",
        "final_commit_sha",
        "gpu_hours_used",
        "gpu_hours_budgeted",
        "plan_deviations",
    }
    nums = payload["eval_numbers"]
    assert nums["mode"] == "repair_capture"
    assert nums["n_missing_rows_captured"] == 484
    assert nums["realized_rows_per_job"] == {"syc_aita": 1340}
    assert np.isclose(payload["gpu_hours_used"], 0.5)
