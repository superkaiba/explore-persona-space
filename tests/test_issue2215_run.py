"""Issue #2215 driver — CPU pins for the pure gate/pooling/config helpers.

Covers unit 1 (Phases A+B) behavior that is testable without a GPU or network:

- the tail-inclusive capture extension's pooling MATH through the REAL
  ``capture_answer_states`` body (fake ONLY the model-forward boundary,
  signature-conformant by construction) + the default-unchanged contract,
- Phase A coverage-gate helpers (pair table, keyset equality/dup/K, graceful
  n_valid floor, ridge payload key/shape asserts),
- the per-shard pooling-parity gate against a real banked-shard file
  (matched rows -> cos 1.0; empty-row drift -> stop-and-diagnose),
- pilot projection arithmetic, rowwise flattened cosine,
- smoke out-root + HF ``/smoke`` prefix rebinding, regime-fingerprint keys
  (capture_batch deliberately excluded), shard resume-manifest semantics,
- the B-end upload + landing verification + sentinel (network boundary
  faked signature-conformant), and the ``--phase all`` a->b->c->d chain
  ordering (unit 2; Phase C/D internals are pinned in
  ``tests/test_issue2215_analysis.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2162_run as R2162  # noqa: E402
import issue2215_run as R  # noqa: E402

# ── fakes (signature-conformant by construction) ──────────────────────


class FakeTokenizer:
    """Mirrors the two surfaces ``capture_answer_states`` touches: the
    ``__call__(text, add_special_tokens=False) -> {"input_ids": [...]}``
    encode and ``pad_token_id``. One token per whitespace word."""

    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        assert add_special_tokens is False
        return {"input_ids": [10 + k for k, _ in enumerate(text.split())]}


def fake_extract_layer_activations(model, ids, layers, attention_mask=None):
    """Signature mirror of ``analysis.extraction.extract_layer_activations``
    (the external model-forward boundary). Activation at (row, position) is
    the POSITION index broadcast over hidden — pooling means are then exact
    small halves/integers, representable in fp16."""
    b, t = ids.shape
    pos = torch.arange(t, dtype=torch.float32)[None, :, None].expand(b, t, 4)
    return {layer: pos.clone() for layer in layers}


def capture_cfg(batch: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        layers=[0, 1], hidden=4, capture_batch=batch, device="cpu", model_id="fake", tiny=True
    )


# ── capture twin: real body, fake forward boundary ────────────────────


def test_tail_inclusive_pools_completion_plus_eot_from_one_stack(monkeypatch):
    monkeypatch.setattr(R2162, "extract_layer_activations", fake_extract_layer_activations)
    cfg = capture_cfg()
    tok = FakeTokenizer()
    ctx_ids = [[1, 2, 3], [1, 2]]
    completions = ["a b", "x y z"]
    eot = [7, 8]
    out = R2162.capture_answer_states(
        cfg, object(), tok, ctx_ids, completions, eot, tail_inclusive=True
    )
    assert set(out) >= {"va_span", "va_tail_incl", "n_completion_tokens", "empty_rows", "pooling"}
    assert out["va_span"].shape == (2, 2, 4)
    assert out["va_tail_incl"].shape == (2, 2, 4)
    # Row 0: ctx_len=3, n_comp=2 -> span positions {3,4} mean 3.5;
    # tail-inclusive adds eot positions {5,6} -> mean of {3,4,5,6} = 4.5.
    assert float(out["va_span"][0, 0, 0]) == 3.5
    assert float(out["va_tail_incl"][0, 0, 0]) == 4.5
    # Row 1: ctx_len=2, n_comp=3 -> span {2,3,4} mean 3.0; incl {2..6} mean 4.0.
    assert float(out["va_span"][1, 1, 2]) == 3.0
    assert float(out["va_tail_incl"][1, 1, 2]) == 4.0
    assert out["n_completion_tokens"] == [2, 3]
    assert out["empty_rows"] == []
    assert "va_tail_incl" in out["pooling"]


def test_default_call_is_unchanged_no_tail_key(monkeypatch):
    monkeypatch.setattr(R2162, "extract_layer_activations", fake_extract_layer_activations)
    cfg = capture_cfg()
    tok = FakeTokenizer()
    ctx_ids = [[1, 2, 3], [1, 2]]
    completions = ["a b", "x y z"]
    default_out = R2162.capture_answer_states(cfg, object(), tok, ctx_ids, completions, [7, 8])
    twin_out = R2162.capture_answer_states(
        cfg, object(), tok, ctx_ids, completions, [7, 8], tail_inclusive=True
    )
    assert "va_tail_incl" not in default_out
    assert "va_tail_incl" not in default_out["pooling"]
    assert torch.equal(default_out["va_span"], twin_out["va_span"])


def test_empty_completion_rows_skip_forward_and_stay_zero(monkeypatch):
    monkeypatch.setattr(R2162, "extract_layer_activations", fake_extract_layer_activations)
    cfg = capture_cfg(batch=3)
    tok = FakeTokenizer()
    out = R2162.capture_answer_states(
        cfg, object(), tok, [[1], [1, 2], [1]], ["a", "", "b c"], [7], tail_inclusive=True
    )
    assert out["empty_rows"] == [1]
    assert out["n_completion_tokens"] == [1, 0, 2]
    assert torch.all(out["va_span"][1] == 0)
    assert torch.all(out["va_tail_incl"][1] == 0)


# ── Phase A gate helpers ──────────────────────────────────────────────


def _pairs(n_cells: int = R.N_CELLS, per_cell: int = R.PAIRS_PER_CELL) -> list[dict]:
    return [
        {"cell": f"cell{c}", "pair_id": f"p{c}_{k}"}
        for c in range(n_cells)
        for k in range(per_cell)
    ]


def test_check_pair_table_accepts_grid_and_rejects_drift():
    per_cell = R.check_pair_table(_pairs())
    assert len(per_cell) == R.N_CELLS and set(per_cell.values()) == {R.PAIRS_PER_CELL}
    with pytest.raises(AssertionError, match="directed pairs"):
        R.check_pair_table(_pairs()[:-1])
    bad = _pairs()
    bad[0] = {**bad[0], "cell": "cell1"}  # 35/37 split across two cells
    with pytest.raises(AssertionError, match="pairs"):
        R.check_pair_table(bad)


def _keys(n_ctx: int = 5, k: int = R.K_DRAWS) -> list[tuple[str, int]]:
    return [(f"c{i}", d) for i in range(n_ctx) for d in range(k)]


def test_check_anchor_keysets_pass_and_failure_modes():
    keys = _keys()
    ctx_ids = {f"c{i}" for i in range(5)}
    R.check_anchor_keysets(keys, list(keys), ctx_ids)
    with pytest.raises(AssertionError, match="duplicate"):
        R.check_anchor_keysets([*keys, keys[0]], [*keys, keys[0]], ctx_ids)
    with pytest.raises(AssertionError, match="keyset mismatch"):
        R.check_anchor_keysets(keys, keys[:-1], ctx_ids)
    with pytest.raises(AssertionError, match="coverage mismatch"):
        R.check_anchor_keysets(keys, list(keys), ctx_ids | {"ghost"})
    short = [kk for kk in keys if kk != ("c0", 9)]
    with pytest.raises(AssertionError, match="!= K"):
        R.check_anchor_keysets(short, list(short), ctx_ids)


def test_n_valid_by_context_counts_empties():
    keys = _keys(n_ctx=2)
    empties = {("c0", 0), ("c0", 3)}
    assert R.n_valid_by_context(keys, empties) == {"c0": 8, "c1": 10}


def _ridge_payload(hidden: int = 8, layer: int = 19) -> dict:
    return {
        "kind": "ridge",
        "layer": layer,
        "xmu": torch.zeros(hidden),
        "xsd": torch.ones(hidden),
        "ymu": torch.zeros(hidden),
        "W": torch.eye(hidden),
    }


def test_check_ridge_payload_shape_and_key_asserts():
    R.check_ridge_payload(_ridge_payload(), "p", expected_layer=19, hidden=8)
    with pytest.raises(AssertionError, match="missing declared keys"):
        bad = _ridge_payload()
        del bad["xsd"]
        R.check_ridge_payload(bad, "p", 19, 8)
    with pytest.raises(AssertionError, match="W shape"):
        R.check_ridge_payload({**_ridge_payload(), "W": torch.eye(4)}, "p", 19, 8)
    with pytest.raises(AssertionError, match="layer"):
        R.check_ridge_payload(_ridge_payload(layer=14), "p", expected_layer=19, hidden=8)
    with pytest.raises(AssertionError, match="kind"):
        R.check_ridge_payload({**_ridge_payload(), "kind": "mlp"}, "p", 19, 8)


def test_rowwise_flat_cosine():
    a = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
    same = R.rowwise_flat_cosine(a, a.clone())
    assert torch.allclose(same, torch.ones(2))
    ortho = R.rowwise_flat_cosine(torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]]]))
    assert float(ortho[0]) == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(AssertionError):
        R.rowwise_flat_cosine(torch.zeros(1, 2, 2), torch.zeros(2, 2, 2))


def test_pilot_projection_arithmetic():
    assert R.pilot_projection(1.0, 3600) == pytest.approx(1.0)
    assert R.pilot_projection(0.5, 14040) == pytest.approx(1.95)


# ── config / resume semantics ─────────────────────────────────────────


def _cfg(tmp_path: Path, **over) -> R.RunConfig2215:
    args = R.parse_args().parse_args(
        [
            "--phase",
            over.pop("phase", "b"),
            "--staged-root",
            str(tmp_path / "staged"),
            "--out-root",
            str(tmp_path / "out"),
            "--tiny",
            *over.pop("extra", []),
        ]
    )
    cfg = R.build_config(args)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_smoke_slice_rebinds_out_root_and_hf_prefix(tmp_path):
    prod = _cfg(tmp_path)
    smoke = _cfg(tmp_path, extra=["--cells", "fact_user_name"])
    assert prod.out_root == tmp_path / "out" and prod.hf_prefix == R.HF_PREFIX_2215
    assert smoke.smoke and smoke.out_root != prod.out_root
    assert "smoke_fact_user_name" in str(smoke.out_root)
    assert smoke.hf_prefix == f"{R.HF_PREFIX_2215}/smoke"


def test_regime_fingerprint_keys_every_output_affecting_knob(tmp_path):
    base = _cfg(tmp_path)
    assert R.regime_fingerprint(base) == R.regime_fingerprint(_cfg(tmp_path))
    assert R.regime_fingerprint(base) != R.regime_fingerprint(
        _cfg(tmp_path, extra=["--cells", "fact_user_name"])
    )
    assert R.regime_fingerprint(base) != R.regime_fingerprint(_cfg(tmp_path, model_id="other"))
    # capture_batch is deliberately EXCLUDED (batch jitter only, not regime).
    assert R.regime_fingerprint(base) == R.regime_fingerprint(
        _cfg(tmp_path, extra=["--capture-batch", "16"])
    )


def test_shard_done_manifest_resume_semantics(tmp_path):
    cfg = _cfg(tmp_path)
    fp = R.regime_fingerprint(cfg)
    cfg.manifest_dir.mkdir(parents=True)
    cfg.va_dir.mkdir(parents=True)
    assert not R._shard_is_done(cfg, "gate", 0, fp)  # no manifest yet
    done = R._shard_done_path(cfg, "gate", 0)
    done.write_text(json.dumps({"regime_fp": fp, "n_rows": 3}))
    # manifest present but store missing -> re-run, never a silent skip
    assert not R._shard_is_done(cfg, "gate", 0, fp)
    torch.save({"ok": True}, R._shard_store_path(cfg, "gate", 0))
    assert R._shard_is_done(cfg, "gate", 0, fp)
    done.write_text(json.dumps({"regime_fp": "deadbeef", "n_rows": 3}))
    with pytest.raises(RuntimeError, match="cross-regime"):
        R._shard_is_done(cfg, "gate", 0, fp)


# ── parity gate against a real banked-shard file ──────────────────────


def _write_banked_shard(cfg: R.RunConfig2215, va: torch.Tensor, empty_rows: list[int]) -> None:
    path = R.shard_tensor_path(cfg, "gate", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"layers": list(range(va.shape[1])), "va_span": va, "empty_rows": empty_rows}, path)


def test_parity_gate_matched_rows_read_cosine_one(tmp_path):
    cfg = _cfg(tmp_path)
    banked = torch.randn(6, 2, 4, dtype=torch.float32).to(torch.float16)
    _write_banked_shard(cfg, banked, empty_rows=[2])
    kept_positions = [1, 2, 4]  # our rows map to banked full-shard rows 1, 2, 4
    our = banked[kept_positions].clone()
    our[1] = 0.0  # our empty row (kept index 1 == banked row 2)
    stats = R.parity_gate_shard(cfg, "gate", 0, kept_positions, our, our_empty=[1])
    assert stats["n_compared"] == 2
    assert stats["min_cos"] == pytest.approx(1.0, abs=1e-6)
    assert stats["frac_ge_bar"] == 1.0


def test_parity_gate_empty_row_drift_is_stop_and_diagnose(tmp_path):
    cfg = _cfg(tmp_path)
    _write_banked_shard(cfg, torch.randn(4, 2, 4).to(torch.float16), empty_rows=[0])
    with pytest.raises(AssertionError, match="empty-row mismatch"):
        R.parity_gate_shard(cfg, "gate", 0, [0, 1], torch.randn(2, 2, 4), our_empty=[])


def test_parity_gate_tiny_shape_mismatch_skips_only_under_tiny(tmp_path):
    cfg = _cfg(tmp_path)
    _write_banked_shard(cfg, torch.randn(4, 28, 8).to(torch.float16), empty_rows=[])
    stats = R.parity_gate_shard(cfg, "gate", 0, [0, 1], torch.randn(2, 2, 4), our_empty=[])
    assert stats["skipped"] == "tiny-shape-mismatch" and stats["n_compared"] == 0
    cfg.tiny = False
    with pytest.raises(AssertionError, match="convention drift"):
        R.parity_gate_shard(cfg, "gate", 0, [0, 1], torch.randn(2, 2, 4), our_empty=[])


# ── B-end upload + sentinel (network boundary faked) ──────────────────


def _seed_va_shard(cfg: R.RunConfig2215) -> Path:
    cfg.va_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.va_dir / "va2215_gate_w0.pt"
    torch.save({"ok": True}, path)
    return path


def test_upload_va_store_writes_sentinel_after_verified_landing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _seed_va_shard(cfg)
    calls: dict = {}

    def fake_upload_dir_hf(local_dir: Path, remote_prefix: str, allow_patterns: list[str]):
        calls["prefix"] = remote_prefix
        return [f"{remote_prefix}/{p.name}" for p in sorted(local_dir.glob(allow_patterns[0]))]

    def fake_list_hf_files_under_path(api, repo_id, path, *, repo_type="model", revision=None):
        return [f"{path}/va2215_gate_w0.pt"]

    monkeypatch.setattr(R2162, "upload_dir_hf", fake_upload_dir_hf)
    import explore_persona_space.orchestrate.hub as hub

    monkeypatch.setattr(hub, "list_hf_files_under_path", fake_list_hf_files_under_path)
    R.upload_va_store(cfg)
    sentinel = cfg.out_root / "va2215_uploaded.json"
    rec = json.loads(sentinel.read_text())
    assert rec["regime_fp"] == R.regime_fingerprint(cfg)
    assert rec["n_local_shards"] == 1 and rec["n_listed_shards"] == 1
    # tiny cfg has no --cells -> production prefix (smoke prefix covered below)
    assert calls["prefix"] == f"{R.HF_PREFIX_2215}/analysis_tensors/va2215"
    assert "repro" in rec


def test_upload_va_store_fails_loud_when_landing_incomplete(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _seed_va_shard(cfg)
    monkeypatch.setattr(
        R2162,
        "upload_dir_hf",
        lambda local_dir, remote_prefix, allow_patterns: [f"{remote_prefix}/va2215_gate_w0.pt"],
    )
    import explore_persona_space.orchestrate.hub as hub

    monkeypatch.setattr(
        hub,
        "list_hf_files_under_path",
        lambda api, repo_id, path, *, repo_type="model", revision=None: [],
    )
    with pytest.raises(AssertionError, match="landing verification FAIL"):
        R.upload_va_store(cfg)
    assert not (cfg.out_root / "va2215_uploaded.json").exists()


def test_upload_none_never_writes_the_phase_c_gate_sentinel(tmp_path):
    cfg = _cfg(tmp_path, upload_mode="none")
    _seed_va_shard(cfg)
    R.upload_va_store(cfg)
    assert not (cfg.out_root / "va2215_uploaded.json").exists()


# ── --phase all chain (a -> b -> c -> d, unit 2 wiring) ───────────────


def test_phase_all_runs_full_chain_in_order(tmp_path, monkeypatch):
    ran: list[str] = []
    monkeypatch.setattr(R, "phase_stage", lambda cfg: ran.append("a") or R.RC_OK)
    monkeypatch.setattr(R, "phase_capture", lambda cfg: ran.append("b") or R.RC_OK)
    monkeypatch.setattr(R, "phase_analysis", lambda cfg: ran.append("c") or R.RC_OK)
    monkeypatch.setattr(R, "phase_finalize", lambda cfg: ran.append("d") or R.RC_OK)
    rc = R.main(
        [
            "--phase",
            "all",
            "--tiny",
            "--staged-root",
            str(tmp_path / "s"),
            "--out-root",
            str(tmp_path / "o"),
        ]
    )
    assert rc == R.RC_OK and ran == ["a", "b", "c", "d"]


def test_phase_all_stops_at_first_failing_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "phase_stage", lambda cfg: R.RC_OK)
    monkeypatch.setattr(R, "phase_capture", lambda cfg: R.RC_PARITY_GATE)
    monkeypatch.setattr(R, "phase_analysis", lambda cfg: pytest.fail("C ran after B failed"))
    args = ["--tiny", "--staged-root", str(tmp_path / "s"), "--out-root", str(tmp_path / "o")]
    assert R.main(["--phase", "all", *args]) == R.RC_PARITY_GATE


def test_phase_b_alone_returns_ok_and_gate_rcs_propagate(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "phase_capture", lambda cfg: R.RC_OK)
    args = ["--tiny", "--staged-root", str(tmp_path / "s"), "--out-root", str(tmp_path / "o")]
    assert R.main(["--phase", "b", *args]) == R.RC_OK
    monkeypatch.setattr(R, "phase_capture", lambda cfg: R.RC_PILOT_GATE)
    assert R.main(["--phase", "b", *args]) == R.RC_PILOT_GATE
    monkeypatch.setattr(R, "phase_stage", lambda cfg: R.RC_PILOT_GATE)
    # a failing phase A must stop the 'all' chain before B
    monkeypatch.setattr(R, "phase_capture", lambda cfg: pytest.fail("B ran after A failed"))
    assert R.main(["--phase", "all", *args]) == R.RC_PILOT_GATE
