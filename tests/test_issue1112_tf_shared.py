"""#1112 tf-shared amendment (followup `tf-shared-response-capture`, plan v6).

Tiny-real coverage of the new pod phase + VM geometry pass:

1. ``run_capture_tf_unit`` — REAL body end-to-end on CPU: real Qwen tokenizer
   over the real vocab space, a from-config 2-layer same-arch model, and a
   4-row slice of the REAL pinned conditioning artifact
   (``tests/fixtures/issue1112_tf_base_rows_4row.json``, downsampled from
   ``raw_completions/capture/base_sycophancy/base/raw_rows.json`` @
   ``e0169101…`` — fixture mirrors the artifact's REALIZED shape, not the
   builder code). Fakes ONLY the checkpoint-resolution boundary (Hub +
   GPU-scale weights) with a signature-mirroring resolve_fn.
2. ``_resolve_tf_capture_model`` — real body (pinned prefix/revision lookup,
   completeness guard incl. the wipe+re-stage branch, full-vs-LoRA routing)
   with def-mirroring fakes at the Hub/merge/tokenizer boundaries (the
   resolve seam is stubbed in test 1, so its body coverage lives here).
3. ``assert_tf_base_rows`` fail-fast contract + ``normalize_phases`` aliases.
4. ``run_tf_shared`` — real geometry body on synthetic real-shaped stores
   (layers {0, 14} so the registered-lattice path executes), including the
   batched-vs-serial ``_mu_norm_draws`` equivalence check and the
   ``_reorder_store`` re-pairing / set-mismatch halt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1112_dispatch as d  # noqa: E402
import issue1112_geometry as g  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "issue1112_tf_base_rows_4row.json"


def _fixture_rows() -> list[dict]:
    return json.loads(FIXTURE.read_text())["rows"]


def _cfg(tmp_path: Path, **kw) -> d.Cfg:
    defaults = dict(smoke=True, cells=("s3_fullft_neg",), out_root=tmp_path, upload=False)
    defaults.update(kw)
    return d.Cfg(**defaults)


# ── 1. tiny-real CPU e2e of run_capture_tf_unit ─────────────────────────────


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> Path:
    """From-config 2-layer Qwen2 over the REAL vocab space + the REAL tokenizer
    (the tiny-real standard: fake only GPU-scale weights, keep every library
    type + the real BPE id space)."""
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    out = tmp_path_factory.mktemp("tiny_qwen2")
    tok = AutoTokenizer.from_pretrained(d.DEFAULT_BASE_MODEL)
    config = Qwen2Config(
        vocab_size=tok.vocab_size if tok.vocab_size >= 152064 else 152064,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        pad_token_id=tok.pad_token_id,
    )
    # Qwen2.5 vocab ids run past tok.vocab_size (added tokens) — size the
    # embedding to the tokenizer's FULL id space so real prompt ids embed.
    config.vocab_size = max(config.vocab_size, len(tok))
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    return out


def test_run_capture_tf_unit_tiny_real_cpu(tmp_path, tiny_model_dir):
    rows = _fixture_rows()
    assert len(rows) == 4
    cfg = _cfg(tmp_path)
    resolved: list[str] = []

    def fake_resolve(cfg_arg, cell):  # mirrors _resolve_tf_capture_model's signature
        resolved.append(cell)
        return (
            str(tiny_model_dir),
            [],
            {
                "repo": d.C.OVERFLOW_REPO,
                "prefix": "issue1112/s3_fullft_neg/checkpoint-8",
                "revision": d.TF_OVERFLOW_REV,
                "kind": "full",
            },
        )

    rec = d.run_capture_tf_unit(
        cfg, "s3_fullft_neg", rows, resolve_fn=fake_resolve, layers=[0, 1], device="cpu"
    )
    assert resolved == ["s3_fullft_neg"]
    pooled_path = tmp_path / "capture_tf" / "s3_fullft_neg" / "selected" / "pooled.pt"
    assert rec["pooled"] == str(pooled_path) and pooled_path.exists()

    store = torch.load(pooled_path, map_location="cpu", weights_only=False)
    assert store["schema_version"] == 1
    assert store["cell"] == "s3_fullft_neg" and store["dose"] == "selected"
    assert store["behavior"] == "sycophancy"
    assert sorted(store["arms"]) == ["context", "prefix", "response"]
    for arm in store["arms"]:
        assert sorted(store["arms"][arm]) == [0, 1]
        for li, t in store["arms"][arm].items():
            assert t.shape == (4, 64), (arm, li, t.shape)
            assert t.dtype == torch.float16
    assert store["row_meta"] == [
        {"context_id": r["persona"], "question_idx": int(r["question_idx"])} for r in rows
    ]
    meta = store["metadata"]
    assert meta["conditioning"] == "tf_shared_base"
    assert meta["followup_label"] == d.TF_FOLLOWUP_LABEL
    assert meta["conditioning_rows"]["revision"] == d.TF_BASE_ROWS["sycophancy"][1]
    assert meta["checkpoint"]["prefix"] == "issue1112/s3_fullft_neg/checkpoint-8"

    # idempotence (spot-tolerant resume): second call skips, no re-resolve
    rec2 = d.run_capture_tf_unit(
        cfg, "s3_fullft_neg", rows, resolve_fn=fake_resolve, layers=[0, 1], device="cpu"
    )
    assert rec2.get("skipped") and resolved == ["s3_fullft_neg"]


def test_run_capture_tf_unit_requires_cuda_by_default(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("CUDA present — the fail-loud default cannot be exercised here")
    with pytest.raises(RuntimeError, match="CUDA"):
        d.run_capture_tf_unit(_cfg(tmp_path), "s3_fullft_neg", _fixture_rows())


# ── 2. _resolve_tf_capture_model body (the seam stubbed in test 1) ──────────


def _fake_stage_factory(kind: str, complete_on_call: int = 1):
    """Signature-mirroring fake of _stage_overflow_prefix (the Hub boundary):
    writes an INCOMPLETE dir until call number ``complete_on_call``."""
    calls: list[tuple] = []

    def fake_stage(prefix, dest, *, revision, recursive=True):
        calls.append((prefix, str(dest), revision, recursive))
        dest.mkdir(parents=True, exist_ok=True)
        if kind == "full":
            (dest / "config.json").write_text("{}")
            if len(calls) >= complete_on_call:
                (dest / "model.safetensors").write_bytes(b"w")
        else:
            (dest / "adapter_config.json").write_text("{}")
            if len(calls) >= complete_on_call:
                (dest / "adapter_model.safetensors").write_bytes(b"w")
        return dest

    return fake_stage, calls


def test_resolve_full_ft_branch(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    fake_stage, calls = _fake_stage_factory("full")
    monkeypatch.setattr(d, "_stage_overflow_prefix", fake_stage)
    repaired: list[Path] = []

    def fake_tokenizer_repair(model_dir, base_model=d.DEFAULT_BASE_MODEL):
        repaired.append(model_dir)
        return False

    monkeypatch.setattr(d, "_ensure_dir_tokenizer", fake_tokenizer_repair)
    path, cleanup, prov = d._resolve_tf_capture_model(cfg, "s3_fullft_neg")
    assert prov == {
        "repo": d.C.OVERFLOW_REPO,
        "prefix": "issue1112/s3_fullft_neg/checkpoint-8",
        "revision": d.TF_OVERFLOW_REV,
        "kind": "full",
    }
    assert calls[0][0] == "issue1112/s3_fullft_neg/checkpoint-8"
    assert calls[0][2] == d.TF_OVERFLOW_REV and calls[0][3] is True
    assert Path(path).exists() and cleanup == [Path(path)]
    assert repaired == [Path(path)]  # r6 tokenizer repair reached


def test_resolve_lora_branch_merges(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, cells=("s1_lora_neg",))
    fake_stage, _calls = _fake_stage_factory("lora")
    monkeypatch.setattr(d, "_stage_overflow_prefix", fake_stage)
    merged_calls: list[tuple] = []

    def fake_merge(cfg_arg, adapter_dir, merged_dir):  # mirrors _merge_adapter
        merged_calls.append((adapter_dir, merged_dir))
        merged_dir.mkdir(parents=True, exist_ok=True)
        (merged_dir / "config.json").write_text("{}")
        return merged_dir

    monkeypatch.setattr(d, "_merge_adapter", fake_merge)
    path, cleanup, prov = d._resolve_tf_capture_model(cfg, "s1_lora_neg")
    # s1 reuses the fu2 checkpoint at ITS OWN pinned revision (plan §4 table)
    assert prov["prefix"] == f"{d.C.FU2_CKPT_PREFIX}/checkpoint-{d.C.FU2_SELECTED_STEP}"
    assert prov["revision"] == d.C.FU2_CKPT_REV and prov["kind"] == "lora"
    (adapter_dir, merged_dir) = merged_calls[0]
    assert adapter_dir == str(tmp_path / "inputs" / "tf_ckpts" / "s1_lora_neg")
    assert merged_dir == tmp_path / "capture_tf" / "s1_lora_neg" / "merged_tf"
    assert path == str(merged_dir) and cleanup == [merged_dir]


def test_stage_tf_ckpt_wipes_incomplete_then_restages(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    fake_stage, calls = _fake_stage_factory("full", complete_on_call=2)
    monkeypatch.setattr(d, "_stage_overflow_prefix", fake_stage)
    staged = d._stage_tf_ckpt(cfg, "s3_fullft_neg", "p", "r", "full")
    assert len(calls) == 2  # incomplete first stage -> wipe -> re-stage
    assert (staged / "model.safetensors").exists()


# ── 3. conditioning-row contract + phase aliases ─────────────────────────────


def test_assert_tf_base_rows_contract():
    rows = _fixture_rows()
    d.assert_tf_base_rows(rows, expect_contexts=2, expect_questions=2)
    bad = [dict(r) for r in rows]
    bad[0]["response_token_ids"] = []
    with pytest.raises(AssertionError, match="empty response"):
        d.assert_tf_base_rows(bad, expect_contexts=2, expect_questions=2)
    with pytest.raises(AssertionError):  # incomplete grid
        d.assert_tf_base_rows(rows[:3], expect_contexts=2, expect_questions=2)
    sub = d.tf_smoke_rows(rows)
    assert len(sub) == 4


def test_phase_upload_capture_tf_sweep(tmp_path, monkeypatch):
    """p12 sweeps the REALIZED capture_tf tree to the canonical analysis_tensors
    paths — and NEVER under --smoke (4-row smoke tensors must not clobber the
    production paths)."""
    uploads: list[tuple[str, str]] = []

    def fake_upload(local_path, repo_id, repo_type, path_in_repo, **kw):
        uploads.append((str(local_path), path_in_repo))
        return f"https://hf.co/{repo_id}/{path_in_repo}"

    monkeypatch.setattr(d.hub, "_upload", fake_upload)
    for cell in ("s3_fullft_neg", "s4_fullft_pos"):
        p = tmp_path / "capture_tf" / cell / "selected"
        p.mkdir(parents=True)
        (p / "pooled.pt").write_bytes(b"t")
    (tmp_path / "capture_tf_manifest.json").write_text("{}")

    cfg = _cfg(tmp_path, smoke=False, cells=("s3_fullft_neg", "s4_fullft_pos"), upload=True)
    d.phase_upload(cfg)
    tf_paths = sorted(p for _, p in uploads if "capture_tf" in p)
    assert tf_paths == [
        f"{d.C.DATA_PREFIX}/analysis_tensors/capture_tf/capture_tf_manifest.json",
        f"{d.C.DATA_PREFIX}/analysis_tensors/capture_tf/s3_fullft_neg/selected/pooled.pt",
        f"{d.C.DATA_PREFIX}/analysis_tensors/capture_tf/s4_fullft_pos/selected/pooled.pt",
    ]

    uploads.clear()
    smoke_cfg = _cfg(tmp_path, smoke=True, cells=("s3_fullft_neg",), upload=True)
    d.phase_upload(smoke_cfg)
    assert not [p for _, p in uploads if "capture_tf" in p]  # smoke-guard engaged


def test_normalize_phases_aliases():
    assert d.normalize_phases("p10b_capture_tf,p12_upload") == ("capture_tf", "upload")
    assert d.normalize_phases("capture_tf,upload") == ("capture_tf", "upload")
    assert d.normalize_phases(None) == ()
    with pytest.raises(ValueError, match="unknown phase"):
        d.normalize_phases("p13_bogus")


# ── 4. tf-shared geometry on synthetic real-shaped stores ────────────────────

LAYERS = (0, 14)
HIDDEN = 16
CTX = ("ctx_a", "ctx_b")
NQ = 3


def _keys() -> list[tuple[str, int]]:
    return [(c, q) for c in CTX for q in range(NQ)]


def _mk_store(cell: str, dose: str, rng: np.random.Generator, *, tf: bool = False) -> dict:
    rows = _keys()
    arms = {
        arm: {
            li: torch.from_numpy(rng.standard_normal((len(rows), HIDDEN))).to(torch.float16)
            for li in LAYERS
        }
        for arm in ("prefix", "context", "response")
    }
    store = {
        "schema_version": 1,
        "cell": cell,
        "dose": dose,
        "behavior": "sycophancy",
        "model_path": "synthetic",
        "row_meta": [{"context_id": c, "question_idx": q} for c, q in rows],
        "arms": arms,
        "metadata": {"conditioning": "tf_shared_base"} if tf else {},
    }
    return store


def _write_tree(tmp_path: Path, rng: np.random.Generator) -> tuple[Path, Path, Path]:
    tf_root = tmp_path / "capture_tf"
    parent_root = tmp_path / "capture"
    for cell in g.BEHAVIOR_CELLS_2X2:
        p = tf_root / cell / "selected"
        p.mkdir(parents=True)
        tf_store = _mk_store(cell, "selected", rng, tf=True)
        # parity is exact when prefix/context match the parent store — copy
        # them into the parent store below via a shared draw:
        torch.save(tf_store, p / "pooled.pt")
        q = parent_root / cell / "selected"
        q.mkdir(parents=True)
        own = _mk_store(cell, "selected", rng)
        own["arms"]["prefix"] = tf_store["arms"]["prefix"]
        own["arms"]["context"] = tf_store["arms"]["context"]
        torch.save(own, q / "pooled.pt")
    b = parent_root / "base_sycophancy" / "base"
    b.mkdir(parents=True)
    torch.save(_mk_store("base_sycophancy", "base", rng), b / "pooled.pt")
    rb_dir = tmp_path / "rb"
    rb_dir.mkdir()
    torch.save(
        {"rb": torch.from_numpy(rng.standard_normal((15, HIDDEN))).to(torch.float32)},
        rb_dir / "rb_sycophancy.pt",
    )
    return tf_root, parent_root, rb_dir


def test_run_tf_shared_synthetic(tmp_path):
    rng = np.random.default_rng(0)
    tf_root, parent_root, rb_dir = _write_tree(tmp_path, rng)
    out_dir = tmp_path / "out"
    payload = g.run_tf_shared(tf_root, parent_root, rb_dir, out_dir, n_boot=16, mu_n_boot=24)
    assert (out_dir / "geometry_tf_shared.json").exists()
    assert payload["followup_label"] == g.TF_LABEL
    assert sorted(payload["cells_realized"]) == sorted(g.BEHAVIOR_CELLS_2X2)
    assert payload["cells_missing"] == ["s5_lora_generic", "s6_fullft_generic"]
    rec = payload["records"]["s1_lora_neg/L14"]
    for side in ("shared", "own"):
        assert set(rec[side]) >= {"rank_k_at_90", "pr_lambda", "top_share_lambda", "mu_norm"}
        assert "rank_k_at_90" in rec[side]["boot_ci"]
    diff = rec["diff_own_minus_shared"]["mu_norm"]
    assert diff["n_boot"] == 24 and diff["resampling"] == "paired"
    assert rec["diff_own_minus_shared"]["rank_k_at_90"]["n_boot"] == 16
    # lattice: mechanical branch per cell + >=3-of-4 headline over the 2x2
    lat = payload["lattice"]
    assert lat["registered_thresholds"] == {"collapse_max": 30.0, "stays_diffuse_min": 60.0}
    assert lat["headline_branch"] in ("collapse", "partial", "stays_diffuse")
    for cell in g.BEHAVIOR_CELLS_2X2:
        assert lat["per_cell"][cell]["branch"] is not None
    # 6-row synthetic clouds: rank <= 6 -> every cell classifies as collapse
    assert lat["headline_branch"] == "collapse"
    # parity: prefix/context copied verbatim -> cosine 1.0, no WARN
    for cell in g.BEHAVIOR_CELLS_2X2:
        assert payload["parity"][cell]["warn"] is False
        assert payload["parity"][cell]["arms"]["prefix"]["overall_min"] > 0.9999
    # matched-80 on a 6-row cloud -> the explicit note branch
    assert "note" in payload["matched80_shared"]["s1_lora_neg"]
    # per-draw matrices persisted (selection-symmetric-nulls re-reduction)
    mat = torch.load(
        Path(payload["bootstrap_matrices_dir"]) / "s1_lora_neg_tf_shared.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert mat["response/L14/rank_k_at_90/shared"].shape == (16,)
    assert mat["response/L14/mu_norm/own"].shape == (24,)
    assert mat["context/L14/rank_k_at_90/own"].shape == (16,)
    assert any(k.startswith("parity/prefix/L14") for k in mat)


def test_reorder_store_repairs_order_and_halts_on_set_mismatch():
    rng = np.random.default_rng(1)
    store = _mk_store("s1_lora_neg", "selected", rng)
    keys = g._store_keys(store)
    shuffled_perm = list(reversed(range(len(keys))))
    shuffled = dict(store)
    shuffled["row_meta"] = [store["row_meta"][i] for i in shuffled_perm]
    shuffled["arms"] = {
        arm: {li: t[shuffled_perm] for li, t in per.items()} for arm, per in store["arms"].items()
    }
    fixed = g._reorder_store(shuffled, keys)
    assert g._store_keys(fixed) == keys
    for arm in store["arms"]:
        for li in LAYERS:
            assert torch.equal(fixed["arms"][arm][li], store["arms"][arm][li])
    bad = dict(store)
    bad["row_meta"] = [*store["row_meta"][:-1], {"context_id": "ctx_zz", "question_idx": 0}]
    with pytest.raises(AssertionError, match="row_meta set mismatch"):
        g._reorder_store(bad, keys)


def test_mu_norm_draws_matches_serial():
    rng = np.random.default_rng(2)
    cloud = rng.standard_normal((6, HIDDEN))
    idx = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 2, 2, 4, 4], [5, 4, 3, 2, 1, 0]])
    W = g._draw_weight_matrix(idx, 6)
    assert np.allclose(W.sum(axis=1), 1.0)
    batched = g._mu_norm_draws(cloud, W)
    serial = np.array([np.linalg.norm(cloud[row].mean(axis=0)) for row in idx])
    assert np.allclose(batched, serial, atol=1e-12)
