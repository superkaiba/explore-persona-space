"""Unit + tiny-real structural tests for scripts/issue1902_run.py (issue #1902).

CPU-only. The tiny-real capture test builds a 2-layer random-weights Olmo2
model over the REAL vocab (the tiny-real standard, #906) and is SKIPPED when
the OLMo tokenizer is not in the local HF cache (no network in CI) — unit C's
full smoke re-runs it pod-side where the cache is present.

Content hygiene: all fixture text here is benign synthetic prose (never LMSYS
rows).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# VM-local tokenizer cache lives under ~/.cache; set BEFORE the module import
# so its pod-style /workspace setdefault does not shadow it on the dev VM.
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
sys.path.insert(0, str(REPO / "scripts"))

import issue1902_common as C  # noqa: E402
import issue1902_run as R  # noqa: E402

from explore_persona_space.eval.vllm_util import vllm_util_for_free  # noqa: E402

TOKENIZER_ID = C.MODEL_IDS["R"]


def _tokenizer_cached() -> bool:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
        return True
    except Exception:
        return False


# ── pure helpers ─────────────────────────────────────────────────────────────


def test_token_boundary_straddler_policy():
    # tokens: [0,3) [3,7) [7,10); boundary at 5 falls INSIDE token 1.
    offsets = [(0, 3), (3, 7), (7, 10)]
    idx, straddler = R.token_boundary(offsets, 5, include_straddler=False)
    assert (idx, straddler) == (1, True)  # prefix policy: EXCLUDE the straddler
    idx, straddler = R.token_boundary(offsets, 5, include_straddler=True)
    assert (idx, straddler) == (2, True)  # context policy: INCLUDE the straddler
    # Exact token edge: no straddler, both policies agree.
    idx, straddler = R.token_boundary(offsets, 7, include_straddler=False)
    assert (idx, straddler) == (2, False)
    idx2, straddler2 = R.token_boundary(offsets, 7, include_straddler=True)
    assert (idx2, straddler2) == (2, False)


def test_plain_prompt_qspan_by_construction():
    q = "What is 2+2?"
    text, s, e = R.plain_prompt_and_qspan(q, None)
    assert text[s:e] == q
    prefix = [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi, how can I help?"},
    ]
    text, s, e = R.plain_prompt_and_qspan(q, prefix)
    assert text[s:e] == q
    assert text.endswith(f"User: {q}\nAssistant:")
    # Short-query collision immunity (#1776): a 1-char query whose text also
    # appears earlier in the render still anchors to the FINAL user turn.
    text, s, e = R.plain_prompt_and_qspan("h", prefix)
    assert (s, text[s : e + 0]) == (len(text) - len("h\nAssistant:"), "h")


def test_assign_fold_groups_deterministic_and_balanced():
    groups = [f"cluster_{i % 7}" for i in range(700)] + ["gsm8k"] * 50 + ["mbpp"] * 30
    a1 = R.assign_fold_groups(groups, n_folds=6, seed=42)
    a2 = R.assign_fold_groups(groups, n_folds=6, seed=42)
    assert a1 == a2  # deterministic
    assert set(a1) == set(groups)  # whole-group assignment (marked strata too)
    sizes = [0] * 6
    for g in groups:
        sizes[a1[g]] += 1
    assert max(sizes) - min(sizes) <= 100  # greedy balance: one group's size


def test_batches_by_token_budget_invariants(monkeypatch):
    monkeypatch.setattr(R, "CAPTURE_TOKEN_BUDGET", 100)
    monkeypatch.setattr(R, "CAPTURE_BATCH_MAX", 3)
    entries = [{"n_total": n} for n in (10, 90, 20, 30, 40, 5)]
    batches = R._batches_by_token_budget(entries)
    flat = [e["n_total"] for b in batches for e in b]
    assert sorted(flat) == sorted(e["n_total"] for e in entries)  # nothing lost
    for b in batches:
        assert len(b) <= 3
        assert len(b) * max(e["n_total"] for e in b) <= 100 or len(b) == 1


def test_inverse_batch_order_restores_entry_order(monkeypatch):
    """capture_cell saves tensors under entries-order row_ids, but
    _batches_by_token_budget length-SORTS entries — indexing the batch-order
    concat with the inverse permutation must recover entries order exactly
    (crash-fix 7: sorted-order tensors were saved under unsorted row_ids,
    scrambling the ctx<->answer pairing for non-first source cells)."""
    import torch

    monkeypatch.setattr(R, "CAPTURE_TOKEN_BUDGET", 100)
    monkeypatch.setattr(R, "CAPTURE_BATCH_MAX", 3)
    # 10 entries, distinct ids, varied lengths INCLUDING ties (30 x3, 10 x2).
    lengths = [30, 10, 90, 30, 5, 30, 10, 40, 20, 60]
    entries = [{"id": f"row_{i:02d}", "n_total": n} for i, n in enumerate(lengths)]
    for pos, e in enumerate(entries):
        e["_pos"] = pos
    batches = R._batches_by_token_budget(entries)
    flat = [e["_pos"] for b in batches for e in b]
    assert flat != list(range(len(entries)))  # sorting actually reorders this fixture
    # Stable sort: tied-length entries keep their original relative order.
    assert [p for p in flat if lengths[p] == 30] == [0, 3, 5]
    assert [p for p in flat if lengths[p] == 10] == [1, 6]
    inv = R._inverse_batch_order(batches, n_entries=len(entries))
    # A tensor built in batch order (row value = original position) comes
    # back in entry order after indexing with the inverse permutation.
    batch_order = torch.tensor(flat, dtype=torch.float32).unsqueeze(1)
    assert batch_order[inv].squeeze(1).tolist() == [float(i) for i in range(len(entries))]
    # Ids realign too: batch-order ids indexed by inv == entries-order ids.
    ids_batch_order = [e["id"] for b in batches for e in b]
    assert [ids_batch_order[i] for i in inv.tolist()] == [e["id"] for e in entries]


def test_inverse_batch_order_asserts_full_coverage():
    """The sanity assert fires when the batches drop or duplicate an entry."""
    entries = [{"n_total": 10, "_pos": p} for p in range(4)]
    with pytest.raises(AssertionError, match="exactly once"):
        R._inverse_batch_order([[entries[0], entries[1]], [entries[3]]], n_entries=4)
    with pytest.raises(AssertionError, match="exactly once"):
        R._inverse_batch_order([[entries[0], entries[0]], [entries[1]]], n_entries=2)


def test_unit_regime_mismatch_refuses(tmp_path):
    import argparse

    args = argparse.Namespace(smoke=False)
    regime = R.unit_regime(args, phase="gen", ckpt="B", n_rows=4)
    assert not R.unit_done(tmp_path, "u1", regime)
    R.mark_unit_done(tmp_path, "u1", regime, {"n": 4})
    assert R.unit_done(tmp_path, "u1", regime)
    other = R.unit_regime(args, phase="gen", ckpt="B", n_rows=8)
    with pytest.raises(RuntimeError, match=r"DIFFERENT"):
        R.unit_done(tmp_path, "u1", other)


def test_designed_halt_rc7_with_report(tmp_path):
    with pytest.raises(SystemExit) as ei:
        R.designed_halt(tmp_path, "survival_gate_a", {"corpus": "single", "projected": 12})
    assert ei.value.code == R.GATE_RC == 7
    reports = list((tmp_path / "gate_reports").glob("survival_gate_a_*.json"))
    assert len(reports) == 1
    body = json.loads(reports[0].read_text())
    assert body["verdict"] == "HALT" and body["projected"] == 12


def test_upload_text_payload_shards_stay_non_lfs(tmp_path, monkeypatch):
    """Real split logic; fakes ONLY the Hub boundary (signature-conformant)."""
    staged: dict[str, list[str]] = {}

    class _FakeApi:
        def upload_folder(self, *, folder_path, repo_id, repo_type, path_in_repo, commit_message):
            staged[path_in_repo] = sorted(p.name for p in Path(folder_path).iterdir())

    from explore_persona_space.orchestrate import hub

    monkeypatch.setattr(R, "_hf_api", lambda: _FakeApi())
    monkeypatch.setattr(hub, "retry_transient", lambda fn, what=None, **kw: fn())
    monkeypatch.setattr(R, "TEXT_SHARD_MAX_BYTES", 400)

    small = tmp_path / "B.jsonl"
    small.write_text(json.dumps({"id": "a", "text": "x" * 50}) + "\n", encoding="utf-8")
    paths = R.upload_text_payload(small, "issue1902_stage_map/raw_completions/gen/single")
    assert staged["issue1902_stage_map/raw_completions/gen/single"] == ["B.jsonl"]
    assert paths == ["issue1902_stage_map/raw_completions/gen/single/B.jsonl"]

    big = tmp_path / "R.jsonl"
    rows = [json.dumps({"id": f"r{i}", "text": "y" * 120}) + "\n" for i in range(12)]
    big.write_text("".join(rows), encoding="utf-8")
    R.upload_text_payload(big, "issue1902_stage_map/raw_completions/gen/multi")
    names = staged["issue1902_stage_map/raw_completions/gen/multi"]
    shard_names = [n for n in names if ".shard" in n]
    assert len(shard_names) >= 2 and "R.manifest.json" in names
    # Round-trip: shard line counts sum to the source's; every shard < cap.


def test_probe_layers_depth_relative():
    assert R.probe_layers(32) == [8, 16, 24]
    assert len(R.probe_layers(12)) == 3


# ── tiny-real structural capture (2-layer random Olmo2 over the REAL vocab) ──


@pytest.fixture(scope="module")
def tiny_olmo2():
    if not _tokenizer_cached():
        pytest.skip(f"{TOKENIZER_ID} tokenizer not in the local HF cache")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Olmo2Config

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
    cfg = Olmo2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return tok, model


def _fixture_rows() -> list[dict]:
    return [
        {
            "id": f"single_{i:05d}",
            "corpus": "single",
            "class": "generic",
            "group": f"cluster_{i}",
            "cluster": i,
            "query": q,
        }
        for i, q in enumerate(
            ["What is the capital of France?", "Name a prime number.", "How do magnets work?"]
        )
    ]


def test_tiny_real_capture_cell_structural(tiny_olmo2, tmp_path):
    """Production capture path on CPU: token-id concat + spans + batched
    forward via extract_layer_activations + pooling + fp16 store + row_index."""
    import torch

    tok, model = tiny_olmo2
    rows = _fixture_rows()
    answers = {r["id"]: {"text": f"Answer {i}: something plain."} for i, r in enumerate(rows)}
    layers = [0, 1]
    stats = R.capture_cell(
        model,
        tok,
        rows,
        answers,
        layers,
        out_root=tmp_path,
        ckpt="B",
        src_label="B",
        corpus="single",
        render="plain",
        device="cpu",
        store_subdir=None,
        unit_tag=" test",
    )
    assert stats["n_rows"] == 3 and stats["n_dropped_no_answer"] == 0
    store = tmp_path / "store"
    for layer in layers:
        d = torch.load(
            store / C.answer_store_relpath("B", "B", "single", layer),
            map_location="cpu",
            weights_only=True,
        )
        assert d["w"].shape == (3, 64) and d["w"].dtype == torch.float16
        ctx = torch.load(
            store / C.ctx_store_relpath("B", "single", layer),
            map_location="cpu",
            weights_only=True,
        )
        assert ctx["u_last"].shape == (3, 64) and ctx["u_mean"].shape == (3, 64)
        assert "p_last" not in ctx  # prefix summaries are multi-turn only
    idx_path = store / C.cell_row_index_relpath("B", "B", "single")
    rows_idx = [json.loads(line) for line in idx_path.read_text().split("\n") if line]
    assert len(rows_idx) == 3
    for ri in rows_idx:
        assert 0 < ri["prefix_len"] < ri["context_len"] <= ri["n_prompt_tokens"]
        assert ri["n_answer_tokens"] > 0


def test_tiny_real_capture_multi_prefix_spans(tiny_olmo2, tmp_path):
    import torch

    tok, model = tiny_olmo2
    rows = [
        {
            "id": "multi_00000",
            "corpus": "multi",
            "class": "generic",
            "group": "cluster_0",
            "cluster": 0,
            "prefix_turns": [
                {"role": "user", "content": "Tell me about rivers."},
                {"role": "assistant", "content": "Rivers flow downhill to the sea."},
            ],
            "query": "Which is the longest?",
        }
    ]
    answers = {"multi_00000": {"text": "The Nile is commonly cited."}}
    R.capture_cell(
        model,
        tok,
        rows,
        answers,
        [0],
        out_root=tmp_path,
        ckpt="B",
        src_label="B",
        corpus="multi",
        render="plain",
        device="cpu",
        store_subdir=None,
        unit_tag=" test-multi",
    )
    ctx = torch.load(
        tmp_path / "store" / C.ctx_store_relpath("B", "multi", 0),
        map_location="cpu",
        weights_only=True,
    )
    assert ctx["p_last"].shape == (1, 64) and ctx["p_mean"].shape == (1, 64)
    idx = json.loads(
        (tmp_path / "store" / C.cell_row_index_relpath("B", "ctx", "multi"))
        .read_text()
        .split("\n")[0]
    )
    # prefix ends where the final user query begins (canonical definition).
    assert idx["prefix_len"] > 4  # two prior turns + "User: " label tokens


def test_tiny_real_native_render_spans(tiny_olmo2, tmp_path):
    tok, model = tiny_olmo2
    rows = _fixture_rows()[:2]
    answers = {r["id"]: {"text": "A short native answer."} for r in rows}
    stats = R.capture_cell(
        model,
        tok,
        rows,
        answers,
        [1],
        out_root=tmp_path,
        ckpt="R",
        src_label="R",
        corpus="single",
        render="native",
        device="cpu",
        store_subdir="robust_native/R/single",
        unit_tag=" test-native",
    )
    assert stats["n_rows"] == 2
    assert (tmp_path / "store" / "robust_native/R/single/L1.pt").exists()
    assert (tmp_path / "store" / "robust_native/R/single/ctx/L1.pt").exists()


def test_pilot_fits_structural_cpu(tmp_path):
    """Runs the REAL pilot-fits body (ridge_fit_predict_fast + mlp_fit_predict
    call shapes) on tiny synthetic stores — no tokenizer/model needed."""
    import torch

    torch.manual_seed(0)
    n, h = 24, 16
    layers = [0, 1, 2]
    for render in ("plain", "native", "plain_fp32"):
        cell = tmp_path / "store" / "pilot" / "R" / render / "single"
        (cell / "ctx").mkdir(parents=True)
        row_ids = [f"single_{i:05d}" for i in range(n)]
        dt = torch.float32 if render == "plain_fp32" else torch.float16
        x = torch.randn(n, h)
        w = x @ torch.randn(h, h) * 0.1 + 0.01 * torch.randn(n, h)
        for layer in layers:
            torch.save({"w": w.to(dt), "row_ids": row_ids}, cell / f"L{layer}.pt")
            torch.save(
                {"u_mean": x.to(dt), "u_last": x.to(dt), "row_ids": row_ids},
                cell / "ctx" / f"L{layer}.pt",
            )
    fits = R._pilot_fits(tmp_path, layers, "cpu", smoke=True)
    assert fits["n"] == n and set(fits["arms"]) == {"plain_fp16", "native_fp16", "plain_fp32"}
    for arm in fits["arms"].values():
        for layer in layers:
            assert "oof_r2" in arm[str(layer)] and len(arm[str(layer)]["lambda_star"]) >= 1
    assert "fp16_delta_r2" in fits and "flip_rule" in fits
    assert fits["mlp_unit_wall_s"] >= 0.0


def test_tiny_real_fp32_twin_store(tiny_olmo2, tmp_path):
    import torch

    tok, model = tiny_olmo2
    rows = _fixture_rows()
    answers = {r["id"]: {"text": "Plain pilot answer."} for r in rows}
    R.capture_cell(
        model,
        tok,
        rows,
        answers,
        [0],
        out_root=tmp_path,
        ckpt="R",
        src_label="R",
        corpus="single",
        render="plain",
        device="cpu",
        store_subdir="pilot/R/plain/single",
        keep_fp32=True,
        unit_tag=" test-fp32",
    )
    fp16 = torch.load(
        tmp_path / "store" / "pilot/R/plain/single/L0.pt", map_location="cpu", weights_only=True
    )
    fp32 = torch.load(
        tmp_path / "store" / "pilot/R/plain_fp32/single/L0.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert fp16["w"].dtype == torch.float16 and fp32["w"].dtype == torch.float32
    assert torch.allclose(fp16["w"].float(), fp32["w"], atol=1e-2)
    assert (tmp_path / "store" / "pilot/R/plain_fp32/single/ctx/L0.pt").exists()


# ── C1 resume side: artifact-aware capture done-predicate (#1315 class) ──────


def test_capture_unit_artifacts_present_and_dirs(tmp_path):
    """A done-sentinel alone must not fast-forward past deleted, never-uploaded
    artifacts: the predicate reads the unit's REAL store leaves (row_index +
    every layer), for both the grid and subdir unit shapes."""
    import shutil

    store = R._store_root(tmp_path / "out")
    layers = [0, 1]
    u_sub = {"subdir": f"reliability/B/{C.CORPUS_SINGLE}/seed43", "src": "B", "corpus": "single"}
    for d in R.capture_unit_store_dirs(store, "B", u_sub, layers):
        d.mkdir(parents=True, exist_ok=True)
        (d / "row_index.jsonl").write_text("{}\n", encoding="utf-8")
        for layer in layers:
            (d / f"L{layer}.pt").write_bytes(b"x")
    assert R.capture_unit_artifacts_present(store, "B", u_sub, layers)
    shutil.rmtree(store / "reliability")
    assert not R.capture_unit_artifacts_present(store, "B", u_sub, layers)

    u_grid = {"subdir": None, "src": "R", "corpus": "single"}
    dirs = R.capture_unit_store_dirs(store, "B", u_grid, layers)
    assert dirs[0].as_posix().endswith("B/R/single")
    assert dirs[1].as_posix().endswith(f"B/{C.CTX_SOURCE}/single")
    assert not R.capture_unit_artifacts_present(store, "B", u_grid, layers)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / "row_index.jsonl").write_text("{}\n", encoding="utf-8")
        for layer in layers:
            (d / f"L{layer}.pt").write_bytes(b"x")
    assert R.capture_unit_artifacts_present(store, "B", u_grid, layers)
    # a partially-deleted leaf (one layer gone) reads NOT present
    (dirs[0] / "L1.pt").unlink()
    assert not R.capture_unit_artifacts_present(store, "B", u_grid, layers)


# ── M2: capture-cost projection basis (uncapped per-corpus sums) ─────────────


def test_capture_rows_per_leg_uses_uncapped_per_corpus_sums():
    """Review r1 M2: the projection consumes each corpus's OWN projected
    intersection UNCAPPED (capture filters by manifest ids with no cap); the
    old 2*min(isect, target) basis under-projects on asymmetric corpora and
    on realized intersections above INTERSECTION_TARGET."""
    isect = {"single": 12_000, "multi": 9_000}
    rows = R.capture_rows_per_leg(4, isect)
    assert rows == 4 * 21_000 + R.ROBUST_NATIVE_N + 2 * C.RELIABILITY_SUBSET_N
    old_basis = (
        4 * 2 * min(min(isect.values()), C.INTERSECTION_TARGET)
        + R.ROBUST_NATIVE_N
        + 2 * C.RELIABILITY_SUBSET_N
    )
    assert rows > old_basis
    # symmetric at-target corpora: new basis == old basis (no regression)
    at_target = {"single": C.INTERSECTION_TARGET, "multi": C.INTERSECTION_TARGET}
    assert R.capture_rows_per_leg(4, at_target) == 4 * 2 * C.INTERSECTION_TARGET + (
        R.ROBUST_NATIVE_N + 2 * C.RELIABILITY_SUBSET_N
    )


# ── shared-node GPU sizing (#1902 crash 1) ───────────────────────────────────

GIB = 2**30


def test_realized_gpu_ids_slurm_count_only_crash1_env():
    # The fellows job 16127 env: SLURM_GPUS_ON_NODE=4, no JOB_GPUS/CVD,
    # nvidia-smi detects the PHYSICAL 8x H200 node. Width must be 4, never 8.
    env = {"SLURM_JOB_ID": "16127", "SLURM_GPUS_ON_NODE": "4"}
    src, ids = C.realized_gpu_ids(env, detected=8)
    assert ids == ["0", "1", "2", "3"]
    assert src.startswith("slurm-count")


def test_realized_gpu_ids_prefers_slurm_cvd_then_job_gpus():
    env = {
        "SLURM_JOB_ID": "1",
        "SLURM_GPUS_ON_NODE": "4",
        "SLURM_JOB_GPUS": "0,1,2,3",
        "CUDA_VISIBLE_DEVICES": "4,5,6,7",
    }
    src, ids = C.realized_gpu_ids(env, detected=8)
    assert (src, ids) == ("slurm-cvd", ["4", "5", "6", "7"])
    del env["CUDA_VISIBLE_DEVICES"]
    src, ids = C.realized_gpu_ids(env, detected=8)
    assert (src, ids) == ("slurm-job-gpus", ["0", "1", "2", "3"])


def test_realized_gpu_ids_clamped_by_requested_width():
    env = {"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "2", "SLURM_JOB_GPUS": "4,5,6,7"}
    src, ids = C.realized_gpu_ids(env, detected=8)
    assert ids == ["4", "5"]
    assert src.endswith("-clamped")


def test_realized_gpu_ids_slurm_without_any_allocation_env_fails_loud():
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        C.realized_gpu_ids({"SLURM_JOB_ID": "1"}, detected=8)


def test_realized_gpu_ids_non_slurm_keeps_detected():
    assert C.realized_gpu_ids({}, detected=8) == (
        "detected",
        [str(i) for i in range(8)],
    )
    assert C.realized_gpu_ids({}, detected=0) == ("detected", ["0"])


# The pure-math vllm_util_for_free tests moved to tests/test_vllm_util.py
# (#1942: the resolver was hoisted to explore_persona_space.eval.vllm_util).


def test_gen_gpu_mem_util_env_override_and_live_path(monkeypatch):
    monkeypatch.setenv("VLLM_GPU_MEM_UTIL", "0.42")
    assert R._gen_gpu_mem_util() == 0.42
    monkeypatch.delenv("VLLM_GPU_MEM_UTIL")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a: (int(81.2 * GIB), int(139.8 * GIB)))
    util = R._gen_gpu_mem_util()
    assert abs(util - vllm_util_for_free(int(81.2 * GIB), int(139.8 * GIB))) < 1e-12


def test_load_hf_model_capture_floor_fail_loud(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a: (int(20 * GIB), int(139.8 * GIB)))
    with pytest.raises(RuntimeError, match="too full for the capture model"):
        R._load_hf_model("dummy/model", None, "cuda:0")


def test_git_sha_degrades_on_gitless_lane(monkeypatch, tmp_path):
    """Fellows/SLURM rsync copy has no .git (#1902 job 16142) — never crash metadata."""
    monkeypatch.delenv("EPS_GIT_SHA", raising=False)
    monkeypatch.setattr(R, "PROJECT_ROOT", tmp_path)
    assert R._git_sha() == "unavailable-no-git-checkout"
    monkeypatch.setenv("EPS_GIT_SHA", "deadbeef123")
    assert R._git_sha() == "deadbeef123"


def test_capture_cost_basis_is_realized_projected_not_resample_capacity():
    """Job 16145 false-fire: the gate fed projected_after_resample (scan-cap
    rescue capacity, ~11x realized) into the capture projection. The basis is
    the realized-corpus intersection ("projected")."""
    src = Path(R.__file__).read_text(encoding="utf-8")
    i = src.index('report["capture_cost"]')
    window = src[max(0, i - 900) : i]
    assert 'int(g["projected"])' in window
    assert 'int(g["projected_after_resample"])' not in window
    # Realized job-16145 numbers PASS the gate under the corrected basis.
    rows = R.capture_rows_per_leg(4, {"single": 15154, "multi": 15307})
    assert rows * 0.0389 / 3600.0 < 2.0 * 4.0
