"""#1090 fu6 driver pins (plan v10).

Covers: the committed verbatim rubric artifact; the VM-side r_B reduction's
equivalence against ``extract_direction`` (plan A4 — same filter, same fp64
RunningMean arithmetic, tiny REAL CPU model + REAL tokenizer); the H1 lattice
classification; the smoke-scaled K1 gate + the PRODUCTION K3 halt branch
(fails-at-production-shape pin, gotchas.md gate-calibration rule iv); the
stored-completions parser; the null helpers' shapes/selection symmetry; and
signature-binds of the smoke-fenced Hub/judge call shapes (#1332 bind rule).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1090_fu6 as fu6  # noqa: E402

# ── rubric artifact ───────────────────────────────────────────────────────────


def test_rubric_artifact_sha_and_slots():
    text = fu6.RUBRIC_PATH.read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode()).hexdigest() == fu6.RUBRIC_SHA256
    assert "{question}" in text and "{answer}" in text
    # the paper's own framing (opinion-agreement + flattery), not wrong-claim
    assert "sycophantic" in text and "REFUSAL" in text
    sidecar = json.loads(
        fu6.RUBRIC_PATH.with_name(fu6.RUBRIC_PATH.stem + ".provenance.json").read_text()
    )
    assert sidecar["sha256"] == fu6.RUBRIC_SHA256
    assert fu6.fu6_rubric() == text  # sha-asserted loader round-trips


# ── A4: reduction equivalence vs extract_direction (tiny REAL CPU model) ─────


def _tiny_qwen(tokenizer):
    from transformers import AutoModelForCausalLM, Qwen2Config

    cfg = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg)


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    except Exception as e:  # pragma: no cover - offline CI
        pytest.skip(f"Qwen tokenizer unavailable: {e}")


def _tiny_completions():
    """(rows, stable keys) — keys mirror the P1a shard identity
    ``(pair_index, arm, question_idx, rollout_idx)``."""
    from explore_persona_space.artifacts.directions import ContrastiveCompletion

    rows = []
    keys = []
    for pi in range(2):
        for arm, score in (("exhibit", 90.0), ("not_exhibit", 10.0)):
            for qi, q in enumerate(("Is tea great?", "Are cats better than dogs?")):
                for ri in range(2):
                    rows.append(
                        ContrastiveCompletion(
                            arm=arm,
                            pair_index=pi,
                            system_prompt=f"sys {arm} {pi}",
                            question=q,
                            response=f"resp {arm} {pi} {qi} {ri} indeed.",
                            judge_score=score if ri == 0 else (60.0 if arm == "exhibit" else 40.0),
                        )
                    )
                    keys.append((pi, arm, qi, ri))
    return rows, keys


def test_reduction_matches_extract_direction(qwen_tokenizer, tmp_path):
    """The driver's stored-means reduction reproduces ``extract_direction``
    bit-for-bit on a 2-pair CPU slice (same capture, same fp64 arithmetic)."""
    from explore_persona_space.artifacts.behavior import BEHAVIORS
    from explore_persona_space.artifacts.directions import (
        batched_response_means,
        encode_rows,
        extract_direction,
    )

    model = _tiny_qwen(qwen_tokenizer)
    comps, comp_keys = _tiny_completions()
    ref = extract_direction(
        BEHAVIORS[fu6.BEHAVIOR],
        model,
        qwen_tokenizer,
        comps,
        regime="read_out",
        provenance="on_policy",
        layers=range(2),
        threshold=fu6.JUDGE_THRESHOLD,
    )
    # Driver path: capture-all (P1b shape) then reduce from stored means,
    # keyed on the STABLE shard ids (skip-safe join, code-review v21 Major 2).
    means_by_key: dict = {}
    encoded, _ = encode_rows(qwen_tokenizer, comps)
    valid = [(i, r) for i, r in enumerate(encoded) if r is not None]
    means = batched_response_means(model, [r for _, r in valid], [0, 1], batch_size=4)
    for (i, _r), m in zip(valid, means, strict=True):
        means_by_key[comp_keys[i]] = m
    cfg = fu6.Cfg(
        smoke=True,
        manifest_path=None,
        manifest_out=None,
        out_root=tmp_path,
        sentinel_dir=tmp_path,
    )
    r_b, counts, _kept = fu6.reduce_rb_from_stored_means(cfg, comps, comp_keys, means_by_key, set())
    assert r_b.shape == ref.r_b.shape
    # Tolerance calibration: the two paths capture with DIFFERENT batch
    # compositions (extract_direction batches per kept arm at batch_size=8;
    # the P1b path captures all rollouts at CAPTURE_BATCH_SIZE), so fp32
    # batched einsum reduction order differs BEFORE the fp64 accumulate —
    # measured max-abs 9.3e-10 on this fixture (fp32 CPU); a real reduction
    # bug (wrong filter/arm/rows) reads O(1e-2)+. 1e-8 = ~10x measured
    # jitter, ~6 orders below the bug regime (the #779 two-bar principle).
    max_abs = float((r_b - ref.r_b).abs().max())
    assert max_abs <= 1e-8, max_abs
    for arm in ("exhibit", "not_exhibit"):
        assert counts[arm]["captured"] == ref.counts[arm]["captured"]
    # The fp64 reduction itself is deterministic: bit-equal on identical means.
    r_b2, _c2, _k2 = fu6.reduce_rb_from_stored_means(cfg, comps, comp_keys, means_by_key, set())
    assert torch.equal(r_b, r_b2)


# ── H1 lattice classification ────────────────────────────────────────────────


def _lattice(rows):
    reads = {
        f"fu3-tier2-{c}": {
            "trained": {"rate": t, "wilson95": [t, t]},
            "base": {"rate": b, "wilson95": [b, b]},
        }
        for c, t, b in rows
    }
    cfg = fu6.Cfg(
        smoke=False,
        manifest_path=None,
        manifest_out=None,
        out_root=Path("/tmp"),
        sentinel_dir=Path("/tmp"),
    )
    return fu6._h1_lattice(cfg, reads)


def _full_rows(t_by_cell):
    return [(c, t_by_cell.get(c, 0.3), b) for c, b in _BASES.items()]


_BASES = {
    "C3-pers-con": 0.2,
    "C3-pers-pos": 0.2,
    "C3-bare-con": 0.2,
    "C3-bare-pos": 0.2,
    "C3-conv-con": 0.7,
    "C3-conv-pos": 0.7,
    "C3-icl-con": 0.5,
    "C3-icl-pos": 0.5,
    "C5-pers-con": 0.2,
    "C5-pers-pos": 0.2,
}


def test_h1_overturned_when_low_base_cell_reaches_band():
    out = _lattice(_full_rows({"C3-pers-con": 0.65}))
    assert out["verdict"] == "Prior-claim-overturned" and out["B"] >= 0


def test_h1_survives_when_only_high_base_reaches_band():
    out = _lattice(_full_rows({"C3-conv-con": 0.7}))
    assert out["verdict"] == "Prior-claim-survives"
    assert out["A"] >= 0 and (out["B"] is None or out["B"] < 0)


def test_h1_no_band_entry_and_empty_low_base_set():
    rows = [(c, 0.3, 0.6) for c in _BASES]  # every base >= 0.45 -> B over empty set
    out = _lattice(rows)
    assert out["verdict"] == "No-band-entry"
    assert out["B_is_neg_inf"] and out["n_low_base_cells"] == 0


# ── K1 smoke scaling + K3 production halt pin ────────────────────────────────


def test_k1_floors_scale_with_smoke_sizes(tmp_path):
    cfg = fu6.Cfg(
        smoke=True, manifest_path=None, manifest_out=None, out_root=tmp_path, sentinel_dir=tmp_path
    )
    counts = {
        "exhibit": {"captured": 2},
        "not_exhibit": {"captured": 2},
        "question_match_kept": {"n_shared_q": 2},
    }
    v = fu6._k1_gate(cfg, counts)
    assert v["pass"], v  # nonzero smoke yield proceeds (gate-calibration rule iii)
    # production floors are the registered constants, byte-unchanged
    assert fu6.K1_MIN_KEPT_FRACTION == 0.2 and fu6.K1_MIN_SHARED_Q_FRACTION == 0.75


def test_k3_production_halt(monkeypatch, tmp_path):
    """The PRODUCTION K3 halt branch fires (rc=21) on a >=10% content-drop
    pilot — pinned so the smoke-scale demotion can never strip the halt."""
    cfg = fu6.Cfg(
        smoke=False,
        manifest_path=None,
        manifest_out=None,
        out_root=tmp_path,
        sentinel_dir=tmp_path,
        deliverables_dir=tmp_path,
    )
    sets = [
        {
            "set_id": "fu3-tier2-C3-pers-con",
            "kind": "tier2",
            "status": "available",
            "revision": "deadbeef",
            "files": {"trained": "x/y.json"},
            "draws": 5,
            "context": "persona_software_engineer",
        }
    ]

    monkeypatch.setattr(
        fu6, "_stage_one", lambda repo_path, revision, dest_root: tmp_path / "c.json"
    )
    (tmp_path / "c.json").write_text(
        json.dumps({"questions": ["q"] * 50, "completions": [["a"]] * 50})
    )

    class FakeResult:
        def __init__(self):
            self.scores = {f"fu6-pilot-q{i:03d}-c0": 50.0 for i in range(45)}
            self.n_total_draws = 250
            self.n_dropped_draws = 30  # 12% content drop >= 10%
            self.n_transport_lost_draws = 0
            self.per_item_scores = {}
            self.per_item_transport_losses = {}

    def fake_judge_call(cfg, tag, items, n_draws, **kw):
        assert tag == "pilot"
        return FakeResult()

    monkeypatch.setattr(fu6, "_judge_call", fake_judge_call)
    with pytest.raises(SystemExit) as exc:
        fu6._pilot_k3(cfg, sets)
    assert exc.value.code == 21
    assert json.loads((tmp_path / "fu6_k3_pilot_report.json").read_text())["verdict"] == "FAIL"


# ── parser ────────────────────────────────────────────────────────────────────


def test_parse_completions_json_happy_and_malformed(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"questions": ["q1", "q2"], "completions": [["a"], ["b", "c"]]}))
    qs, comps = fu6._parse_completions_json(good)
    assert qs == ["q1", "q2"] and comps[1] == ["b", "c"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"questions": ["q1"], "completions": "nope"}))
    with pytest.raises(AssertionError):
        fu6._parse_completions_json(bad)
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"questions": ["q1", "q2"], "completions": [["a"]]}))
    with pytest.raises(AssertionError):
        fu6._parse_completions_json(short)


# ── nulls / selection symmetry ───────────────────────────────────────────────


def test_shuffle_null_shape_and_selection_symmetry(tmp_path):
    rng = np.random.default_rng(0)
    proj = rng.normal(size=(20, 4))
    delta = rng.normal(size=20)
    draws = fu6._shuffle_null_draws(proj, delta, 64, seed=1)
    assert draws.shape == (64, 4)
    assert np.all(draws >= 0) and np.all(draws <= 1 + 1e-12)
    # per-draw same-selection: the band comes from max over layers per draw
    from explore_persona_space.artifacts.directions import select_readout_layer

    rho = np.abs(np.asarray(fu6._spearman_per_layer(proj, delta)))
    head = select_readout_layer(
        torch.tensor(rho),
        list(range(4)),
        null_draws=torch.tensor(draws),
        persist_path=tmp_path / "m.json",
    )
    assert head.selection == "per_draw_same_selection"
    persisted = json.loads((tmp_path / "m.json").read_text())
    assert persisted["n_draws"] == 64 and len(persisted["layers"]) == 4


def test_randnorm_null_norm_matching_and_shape():
    rng = np.random.default_rng(0)
    n_cells, n_layers, dim = 10, 3, 6
    acts = rng.normal(size=(n_cells, n_layers, dim))
    pool = {li: rng.normal(size=(30, dim)) for li in range(n_layers)}
    rb_norms = np.array([1.0, 2.0, 3.0])
    delta = rng.normal(size=n_cells)
    draws = fu6._randnorm_null_draws_fu6(pool, rb_norms, acts, delta, 16, seed=2)
    assert draws.shape == (16, n_layers)
    assert np.all(np.abs(draws) <= 1 + 1e-12)


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(3)
    x, y = rng.normal(size=40), rng.normal(size=40)
    assert fu6._spearman(x, y) == pytest.approx(spearmanr(x, y).statistic, abs=1e-12)


# ── smoke-fenced call-shape binds (#1332) ────────────────────────────────────


def test_smoke_fenced_call_shapes_bind():
    from huggingface_hub import HfApi

    from explore_persona_space.eval.graded_judge import judge_graded
    from explore_persona_space.orchestrate import hub

    # hub._upload folder branch (P1d) — positional + kwarg shape
    inspect.signature(hub._upload).bind(
        Path("/x"), "repo", "dataset", "prefix", ignore_patterns=["*.lock"]
    )
    # verify_repo_paths_uploaded (P1d) — api positional + required kw-only prefix
    inspect.signature(hub.verify_repo_paths_uploaded).bind(
        HfApi(), "repo", ["a"], path_in_repo="p", repo_type="dataset"
    )
    inspect.signature(hub.stage_hub_file).bind(
        "repo", "path", Path("/t"), repo_type="dataset", revision="r"
    )
    inspect.signature(hub.stage_hub_prefix).bind("repo", "prefix", Path("/t"))
    inspect.signature(hub.retry_transient).bind(lambda: 1, what="x")
    # list_hf_files_under_path (P1c _stage_adapter) — REQUIRED api positional
    # (code-review v21 Critical 1: the r1 call site omitted it -> TypeError at
    # the first organism's adapter staging; this bind mirrors the fixed site).
    inspect.signature(hub.list_hf_files_under_path).bind(
        HfApi(), "repo", "subfolder", repo_type="model", revision="r"
    )
    # judge_graded incl. the fu6 passthrough kwarg
    inspect.signature(judge_graded).bind(
        [("i", "q", "a")],
        "rubric {question} {answer}",
        n_draws=1,
        cache_dir=Path("/t"),
        save_raw=Path("/t/r.json"),
        judge_model="m",
        max_tokens=300,
        threshold_base=0,
    )


# ── judge_graded passthrough (production-body test for the library edit) ─────


def test_judge_graded_threads_threshold_base(monkeypatch, tmp_path):
    """The REAL judge_graded body executes and forwards the threshold_base
    passthrough to judge_completions_batch (autospec'd network boundary).
    NOTE: no checkpoint_dir passthrough exists — the r1 draft's was dead
    capability (judge_completions_batch derives cache_dir/.dispatch itself)
    and was removed (code-review v21 Minor 3)."""
    from unittest.mock import create_autospec

    from explore_persona_space.eval import batch_judge as bj
    from explore_persona_space.eval import graded_judge as gj

    fake = create_autospec(bj.judge_completions_batch)
    monkeypatch.setattr(gj._batch_judge, "judge_completions_batch", fake)
    save_raw = tmp_path / "raw.json"
    save_raw.write_text(json.dumps({"all_scores": {"i__00000__00": {"score": 70}}}))
    result = gj.judge_graded(
        [("i", "q", "a")],
        "r {question} {answer}",
        n_draws=1,
        cache_dir=tmp_path,
        save_raw=save_raw,
        threshold_base=0,
    )
    kwargs = fake.call_args.kwargs
    assert kwargs["threshold_base"] == 0
    assert result.scores["i"] == 70.0
    assert "checkpoint_dir" not in inspect.signature(gj.judge_graded).parameters
    # default: passthrough ABSENT (existing callers byte-identical)
    fake.reset_mock()
    gj.judge_graded(
        [("i", "q", "a")], "r {question} {answer}", n_draws=1, cache_dir=tmp_path, save_raw=save_raw
    )
    assert "threshold_base" not in fake.call_args.kwargs


# ── encode-skip keying: REAL P1b capture -> filter -> reduce with a skipped
#    row (code-review v21 Major 2 regression pin) ───────────────────────────────


def test_encode_skip_row_keying_end_to_end(qwen_tokenizer, tmp_path, monkeypatch):
    """A ``"\\n"``-leading rollout (encode_rows ``skipped_prefix_mismatch`` — the
    gotchas.md BPE-seam class) rides REAL P1b capture -> judge-filter -> reduce
    WITHOUT shifting any later same-question rollout's activation join: kept
    rows keep their OWN activations (pre-fix, per-question positional ordinals
    handed q0/r1 the q0/r2 activation and silently dropped q0/r2), the
    kept-but-encode-skipped row is excluded + counted, and a kept key that is
    neither captured nor recorded-skipped raises (loud pool guard)."""
    import dataclasses as dc

    from transformers import AutoModelForCausalLM

    from explore_persona_space.artifacts.directions import batched_response_means, encode_rows

    monkeypatch.setenv("EPM_FU6_DEVICE", "cpu")
    monkeypatch.setenv("EPM_FU6_N_LAYERS", "2")
    tiny_dir = tmp_path / "tiny_model"
    model = _tiny_qwen(qwen_tokenizer)
    model.save_pretrained(tiny_dir)
    qwen_tokenizer.save_pretrained(tiny_dir)
    monkeypatch.setenv("EPM_FU6_BASE_MODEL", str(tiny_dir))

    out_root = tmp_path / "out"
    cfg = fu6.Cfg(
        smoke=True,
        manifest_path=None,
        manifest_out=None,
        out_root=out_root,
        sentinel_dir=tmp_path,
    )
    rollout_dir = out_root / "raw_completions" / "extraction"
    rollout_dir.mkdir(parents=True)
    questions = ["Is tea great?", "Are cats better than dogs?"]
    for arm in ("exhibit", "not_exhibit"):
        rows = []
        for qi in range(2):
            n_r = 3 if (arm == "exhibit" and qi == 0) else 2
            for ri in range(n_r):
                resp = f"resp {arm} {qi} {ri} indeed."
                if arm == "exhibit" and qi == 0 and ri == 0:
                    # BPE-merges into the prompt's trailing "assistant\n" ->
                    # encode_rows skipped_prefix_mismatch (verified on the
                    # real Qwen-2.5-7B tokenizer).
                    resp = "\nleading-newline response"
                rows.append(
                    {
                        "pair_index": 0,
                        "arm": arm,
                        "question_idx": qi,
                        "rollout_idx": ri,
                        "response": resp,
                        "finish_reason": "stop",
                    }
                )
        (rollout_dir / f"pair0_{arm}.json").write_text(
            json.dumps(
                {"meta": {"system_prompt": f"sys {arm}"}, "questions": questions, "rows": rows}
            )
        )

    fu6.phase_capture_rollouts(cfg)

    skip_key = (0, "exhibit", 0, 0)
    store = torch.load(
        out_root / "captures" / "extraction" / "pair0_exhibit.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert store["schema_version"] == 2
    assert store["encode_counts"]["skipped_prefix_mismatch"] == 1
    assert [tuple(k) for k in store["skipped_keys"]] == [skip_key]
    meta_keys = [
        (m["pair_index"], m["arm"], m["question_idx"], m["rollout_idx"]) for m in store["row_meta"]
    ]
    assert skip_key not in meta_keys

    means_by_key, skipped = fu6._load_means_by_key(cfg)
    assert skipped == {skip_key}

    # Alignment: replay P1b's exact capture call (same weights, same row order,
    # same batch size) and assert each kept exhibit row maps to its OWN
    # activation — the pre-fix ordinal join maps q0/r1 to q0/r2's tensor.
    comps, comp_keys = fu6._rollout_completion_objects(rollout_dir)
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(tiny_dir), torch_dtype=fu6._capture_dtype()
    )
    ref_model.eval()
    ex = [(c, k) for c, k in zip(comps, comp_keys, strict=True) if c.arm == "exhibit"]
    rows_enc, _ = encode_rows(qwen_tokenizer, [c for c, _k in ex])
    valid = [(k, r) for (_c, k), r in zip(ex, rows_enc, strict=True) if r is not None]
    ref = batched_response_means(
        ref_model, [r for _k, r in valid], [0, 1], batch_size=fu6.CAPTURE_BATCH_SIZE
    )
    ref_by_key = {k: m.to(torch.float16).float() for (k, _r), m in zip(valid, ref, strict=True)}
    for k, want in ref_by_key.items():
        assert torch.allclose(means_by_key[k], want, atol=1e-6, rtol=0.0), k
    # ...and the check discriminates: the old-bug assignment (q0/r1 given
    # q0/r2's activation) is macroscopically different.
    assert not torch.allclose(
        means_by_key[(0, "exhibit", 0, 1)], ref_by_key[(0, "exhibit", 0, 2)], atol=1e-3, rtol=0.0
    )

    # filter -> reduce: all rows judge-kept; the encode-skipped kept row is
    # excluded from the pool and counted, never silently joined.
    scored = [dc.replace(c, judge_score=90.0 if c.arm == "exhibit" else 10.0) for c in comps]
    r_b, counts, kept_keys = fu6.reduce_rb_from_stored_means(
        cfg, scored, comp_keys, means_by_key, skipped
    )
    assert counts["exhibit"]["kept"] == 5
    assert counts["exhibit"]["encode_skipped_kept"] == 1
    assert counts["exhibit"]["captured"] == 4
    assert counts["not_exhibit"]["captured"] == 4
    assert skip_key not in kept_keys["exhibit"]
    assert r_b.shape == (2, means_by_key[(0, "exhibit", 0, 1)].shape[1])

    # Loud guard: a kept key neither captured nor recorded-skipped raises.
    broken = dict(means_by_key)
    del broken[(0, "exhibit", 1, 0)]
    with pytest.raises(RuntimeError, match="keying bug"):
        fu6.reduce_rb_from_stored_means(cfg, scored, comp_keys, broken, skipped)


# ── tiny-real CPU e2e: P1a-shaped shards -> REAL P1b -> P1c-shaped stores ->
#    REAL P3 reduce-analyze (fake ONLY the vLLM/GPU + Hub boundaries) ─────────


def _write_manifest(tmp: Path, organisms: list[dict], sets: list[dict]) -> Path:
    m = {
        "meta": {"smoke": True},
        "organisms": organisms,
        "capture_panel": ["persona_software_engineer", "default"],
        "judge_sets": sets,
        "probes": {},
    }
    path = tmp / "fu6_manifest.json"
    path.write_text(json.dumps(m))
    return path


def _fake_pooled_store(unit: str, src_ctx: str, hidden: int, n_layers: int, seed: int) -> dict:
    """An organism/base pooled.pt through the REAL store schema (P1c shape)."""
    g = torch.Generator().manual_seed(seed)
    ctxs = ["persona_software_engineer", "default"]
    n_q = 2
    meta = [{"context_id": c, "question_idx": q} for c in ctxs for q in range(n_q)]
    arms = {}
    keys = ["own__prefix", "own__context", "own__response"]
    if unit != "base":
        keys += ["shared__prefix", "shared__context", "shared__response"]
    for key in keys:
        arms[key] = {
            li: torch.randn(len(meta), hidden, generator=g, dtype=torch.float32).to(torch.float16)
            for li in range(n_layers)
        }
    return {
        "schema_version": 1,
        "unit": unit,
        "behavior": "sycophancy",
        "model_path": "tiny",
        "adapter_config_summary": None,
        "row_meta_own": meta,
        "row_meta_shared": list(meta) if unit != "base" else [],
        "arms": arms,
        "metadata": {"ts": "t", "git_commit": "c"},
    }


@pytest.mark.slow
def test_tiny_real_cpu_reduce_e2e(qwen_tokenizer, tmp_path, monkeypatch):
    """REAL P1b capture (tiny 2-layer same-arch model, real tokenizer, real
    ``batched_response_means``) then REAL ``phase_reduce_analyze`` end to end:
    filter -> fp64 r_B -> projections -> BOTH nulls -> selection -> lattice
    inputs -> aggregates + figures — Hub upload off, figures to tmp."""
    hidden = 16
    n_layers = 2
    monkeypatch.setenv("EPM_FU6_DEVICE", "cpu")
    monkeypatch.setenv("EPM_FU6_N_LAYERS", str(n_layers))
    tiny_dir = tmp_path / "tiny_model"
    model = _tiny_qwen(qwen_tokenizer)
    model.save_pretrained(tiny_dir)
    qwen_tokenizer.save_pretrained(tiny_dir)
    monkeypatch.setenv("EPM_FU6_BASE_MODEL", str(tiny_dir))

    out_root = tmp_path / "out"
    deliv = tmp_path / "deliv"
    cfg = fu6.Cfg(
        smoke=True,
        manifest_path=None,
        manifest_out=None,
        out_root=out_root,
        sentinel_dir=tmp_path,
        upload=False,
        shuffle_draws=32,
        randnorm_draws=4,
        bootstrap_draws=64,
        deliverables_dir=deliv,
        figures_dir=tmp_path / "figs",
    )

    # P1a-shaped rollout shards (vLLM boundary faked; REAL shard schema).
    rollout_dir = out_root / "raw_completions" / "extraction"
    rollout_dir.mkdir(parents=True)
    questions = ["Is tea great?", "Are cats better than dogs?"]
    for arm in ("exhibit", "not_exhibit"):
        rows = [
            {
                "pair_index": 0,
                "arm": arm,
                "question_idx": qi,
                "rollout_idx": ri,
                "response": f"resp {arm} {qi} {ri} indeed.",
                "finish_reason": "stop",
            }
            for qi in range(2)
            for ri in range(2)
        ]
        if arm == "exhibit":
            # One encode-skipped rollout ("\n"-leading -> BPE-merges into the
            # prompt tail, encode_rows skipped_prefix_mismatch): the FULL P3
            # (reduce + pool stack + figures) must exclude it without shifting
            # q0's earlier rollouts' joins (code-review v21 Major 2).
            rows.append(
                {
                    "pair_index": 0,
                    "arm": arm,
                    "question_idx": 0,
                    "rollout_idx": 2,
                    "response": "\nencode-skipped row",
                    "finish_reason": "stop",
                }
            )
        (rollout_dir / f"pair0_{arm}.json").write_text(
            json.dumps(
                {"meta": {"system_prompt": f"sys {arm}"}, "questions": questions, "rows": rows}
            )
        )

    # REAL P1b on CPU (tiny model): captures every persisted rollout.
    fu6.phase_capture_rollouts(cfg)
    caps = sorted((out_root / "captures" / "extraction").glob("pair*_*.pt"))
    assert len(caps) == 2
    ex_store = torch.load(caps[0], map_location="cpu", weights_only=False)
    assert [tuple(k) for k in ex_store["skipped_keys"]] == [(0, "exhibit", 0, 2)]

    # P2-shaped judge outputs (REAL schema; judge API boundary faked).
    judge_dir = deliv / "judge"
    judge_dir.mkdir(parents=True)
    scores = {}
    for arm, base_score in (("ex", 80.0), ("ne", 20.0)):
        for qi in range(2):
            for ri in range(2):
                scores[f"f6-ex-p0-{arm}-q{qi:03d}-r{ri:02d}"] = base_score + qi + ri
    # The encode-skipped exhibit rollout is judge-KEPT (score > 50) — P3 must
    # exclude it as encode_skipped_kept, never KeyError at the pool stack.
    scores["f6-ex-p0-ex-q000-r02"] = 85.0
    (judge_dir / "extraction_filter_scores.json").write_text(
        json.dumps({"scores": scores, "n_total_draws": 17, "n_dropped_draws_content": 0})
    )
    organisms = [
        {
            "organism_id": f"org{i}",
            "source_round": "fu3",
            "cell_id": f"C3-x{i}",
            "generator": "claude",
            "source_context": "persona_software_engineer",
        }
        for i in range(2)
    ]
    sets = []
    reads = {}
    rng = np.random.default_rng(7)
    for org in organisms:
        for ctx in ("persona_software_engineer", "default"):
            kind = "tier2" if ctx == org["source_context"] else "bystander"
            sid = f"s-{org['organism_id']}-{ctx}"
            sets.append(
                {
                    "set_id": sid,
                    "kind": kind,
                    "organism_id": org["organism_id"],
                    "context": ctx,
                    "status": "available",
                    "revision": "r",
                    "draws": 5,
                    "files": {},
                }
            )
            tr, ba = float(rng.uniform(0.4, 0.9)), float(rng.uniform(0.1, 0.4))
            reads[sid] = {
                "trained": {"rate": tr, "wilson95": [tr, tr]},
                "base": {"rate": ba, "wilson95": [ba, ba]},
                "delta": tr - ba,
            }
    (deliv / "judged_reads_fu6.json").write_text(
        json.dumps({"meta": {}, "reads": reads, "excluded_sets": [], "k4_flags": []})
    )
    cfg.manifest_path = _write_manifest(tmp_path, organisms, sets)

    # P1c-shaped organism stores (REAL schema; GPU gen boundary faked).
    org_root = out_root / "captures" / "organisms"
    for i, unit in enumerate(["base", "org0", "org1"]):
        d = org_root / unit
        d.mkdir(parents=True)
        store = _fake_pooled_store(unit, "persona_software_engineer", hidden, n_layers, i)
        torch.save(store, d / "pooled.pt")

    agg = fu6.phase_reduce_analyze(cfg)
    assert agg["h2_headline"]["n_cells"] == 4  # 2 organisms x 2 contexts
    assert agg["h2_headline"]["verdict"] in ("Validated", "Contradicted", "Inconclusive")
    assert set(agg["h2_arms"]) == set(fu6.PROJ_ARMS)
    assert (deliv / "fu6_aggregates.json").exists()
    assert (tmp_path / "figs" / "smoke" / "fu6_measurement_repair.png").exists()
    shuffle_matrix = json.loads(
        (out_root / "analysis_tensors" / "shuffle_null__context.json").read_text()
    )
    assert shuffle_matrix["n_draws"] == 32  # per-draw x per-layer matrix persisted
    randnorm_matrix = json.loads(
        (out_root / "analysis_tensors" / "randnorm_null__context.json").read_text()
    )
    assert len(randnorm_matrix["null_draws"]) == 4


# ── epm:failure v5 regression: subprocess phase entry resolves the FULL panel ─


def test_capture_panel_contexts_resolve_in_fresh_subprocess():
    """phase_dispatch runs capture-organisms as a SUBPROCESS, so context
    registration must be unconditional at phase entry: a FRESH child process
    with NO parent-side registration must resolve the FULL production capture
    panel — including the panel-member-only ids neg_sp_police / neg_sp_ph4 —
    through the exact crash seam (_panel_specs -> fu3w.ensure_context).
    Fails pre-fix with ValueError: unknown context 'neg_sp_police'."""
    import subprocess

    child = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'scripts')!r})\n"
        "from pathlib import Path\n"
        "import issue1090_fu6 as fu6\n"
        "cfg = fu6.Cfg(smoke=False, manifest_path=None, manifest_out=None,\n"
        "              out_root=Path('unused'), sentinel_dir=Path('unused'))\n"
        "specs = fu6._panel_specs(cfg)\n"
        "assert set(specs) == set(fu6.CAPTURE_PANEL_IDS), sorted(specs)\n"
        "assert specs['neg_sp_police']['system'], specs['neg_sp_police']\n"
        "assert specs['neg_sp_ph4']['system'], specs['neg_sp_ph4']\n"
        "assert specs['wildchat_prefix_real545']['prior_turns'], 'conv prefix empty'\n"
        "print('SEAM-OK', len(specs))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, (proc.stdout[-500:], proc.stderr[-2000:])
    assert "SEAM-OK 6" in proc.stdout, proc.stdout[-500:]


def test_register_capture_contexts_idempotent_and_foreign_binding_refusal():
    """Same-process pins: double-registration is a no-op (idempotent), and a
    FOREIGN binding under a capture-panel id is refused loudly."""
    from explore_persona_space.artifacts.context import CONTEXTS, Context

    fu6._register_capture_contexts()
    before = {k: CONTEXTS[k] for k in ("neg_sp_police", "neg_sp_ph4")}
    fu6._register_capture_contexts()  # idempotent — same objects, no raise
    assert {k: CONTEXTS[k] for k in before} == before
    foreign = Context(
        context_id="neg_sp_police", kind="persona", family="foreign", system="not the panel"
    )
    old = CONTEXTS["neg_sp_police"]
    try:
        CONTEXTS["neg_sp_police"] = foreign
        with pytest.raises(ValueError, match="refusing to shadow"):
            fu6._register_capture_contexts()
    finally:
        CONTEXTS["neg_sp_police"] = old


# ── rule-23 P2b: probe classification + surgical re-judge merge path ─────────


def _mk_main_raw(tmp_path: Path) -> tuple[Path, list, list]:
    """Synthetic P2-shaped judge_raw: 3 items x 3 draws.

    A: 2 kept + 1 parse_error; B: 3 parse_error (all dropped, scores None);
    C: 2 kept + 1 TRANSPORT error dict (rule 24 split — must stay transport)."""
    perr = {"aligned": None, "coherent": None, "reasoning": "parse_error", "error": True}
    terr = {
        "aligned": None,
        "coherent": None,
        "reasoning": "error: 529 overloaded",
        "error": True,
        "transport": True,
    }
    all_scores = {
        "A__00000__00": {"score": 80},
        "A__00000__01": {"score": 60},
        "A__00000__02": perr,
        "B__00001__00": perr,
        "B__00001__01": perr,
        "B__00001__02": perr,
        "C__00002__00": {"score": 40},
        "C__00002__01": {"score": 20},
        "C__00002__02": terr,
    }
    raw = tmp_path / "judge_raw.json"
    raw.write_text(json.dumps({"all_scores": all_scores}))
    items = [("A", "q0", "a0"), ("B", "q0", "a1"), ("C", "q1", "a2")]
    meta = [("A", 0), ("B", 0), ("C", 1)]
    return raw, items, meta


def test_rule23_merge_recovers_and_preserves_transport(tmp_path):
    """The merge path: recovered mt1000 draws recompute rates correctly, a
    draw still unparseable at 1000 STAYS dropped (drop-never-coerce), and
    transport counters are untouched (rule 24(ii) — never blended)."""
    from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

    raw, items, meta = _mk_main_raw(tmp_path)
    result = judge_result_from_save_raw(raw, items)
    dropped = fu6._content_dropped_by_item(raw, {i[0] for i in items})
    assert dropped == {"A": 1, "B": 3}  # C's loss is transport, NOT content
    assert result.n_dropped_draws == 4 and result.n_transport_lost_draws == 1
    committed = fu6._rate_record(meta, result, 3)
    # pre state: A scored (70>50 -> pos), B all-dropped (n_dropped_items=1),
    # C scored (30 -> neg); drop rate 4/9 flags k4
    assert committed["n"] == 2 and committed["k"] == 1
    assert committed["k4_flag"] is True

    v = {
        "result": result,
        "dropped_by_item": dropped,
        "meta": meta,
        "draws": 3,
    }
    recovered = {"A": [90.0], "B": [70.0, 30.0]}  # B recovers 2 of 3
    rec, audit = fu6._merged_state_record(v, recovered, committed=committed)

    # recovered draws enter the per-item means; still-dropped stays dropped
    assert audit["n_recovered"] == 3 and audit["n_still_dropped"] == 1
    assert rec["n_dropped_draws_content"] == 1
    assert rec["n_total_draws"] == 9
    assert rec["content_drop_rate"] == pytest.approx(1 / 9)
    assert rec["k4_flag"] is True  # 1/9 = 0.111 still >= 0.10 (recompute, not reset)
    # B is now scored: mean(70, 30) = 50.0 -> NOT > 50 -> negative
    assert rec["n"] == 3 and rec["k"] == 1
    assert rec["n_dropped_items"] == 0
    assert rec["per_question_rate"] == {"0": 0.5, "1": 0.0}
    # transport counters untouched by the merge
    assert rec["n_transport_lost_draws"] == 1
    assert audit["post"]["rate"] == pytest.approx(1 / 3)
    assert rec["rejudge_max_tokens"] == fu6.RULE23_MAX_TOKENS
    # over-recovery is clamped per item (a re-judge can never ADD draws)
    rec2, audit2 = fu6._merged_state_record(
        v, {"A": [90.0, 95.0, 99.0], "B": [70.0, 30.0]}, committed=committed
    )
    assert audit2["n_recovered"] == 3  # A capped at its 1 dropped draw
    a_draws = rec2["n_total_draws"]  # unchanged main-pass denominator
    assert a_draws == 9


def test_rule23_probe_classification():
    """Probe categories over the parsed-value space (rule 9 taxonomy)."""
    assert fu6._classify_probe_parse({"score": 85}) == ("resolved_score", 85.0)
    assert fu6._classify_probe_parse(85) == ("resolved_score", 85.0)  # bare-int envelope
    assert fu6._classify_probe_parse({"score": "REFUSAL"}) == ("judge_refusal", None)
    assert fu6._classify_probe_parse(None) == ("still_unparseable", None)
    assert fu6._classify_probe_parse({"score": 150}) == ("malformed_or_out_of_range", None)
    assert fu6._classify_probe_parse({"verdict": "yes"}) == ("malformed_or_out_of_range", None)


def test_rule23_judge_call_partition_and_budget_pins(monkeypatch, tmp_path):
    """_judge_call defaults stay the production P2 instrument (mt=300 under
    judge_cache/fu6/); the rule-23 re-judge threads mt=1000 + its OWN cache
    partition so the rubric-keyed cache can never serve a sibling draw's
    score for a lost draw (rule 24(ii))."""
    from explore_persona_space.eval import graded_judge as gj

    mod = fu6
    calls = []

    def fake_judge_graded(items, rubric, *, n_draws, cache_dir, save_raw, **kw):
        calls.append({"cache_dir": Path(cache_dir), "kw": kw, "n_draws": n_draws})
        return object()

    monkeypatch.setattr(gj, "judge_graded", fake_judge_graded)
    cfg = mod.Cfg(
        smoke=False,
        manifest_path=None,
        manifest_out=None,
        out_root=tmp_path,
        sentinel_dir=tmp_path,
    )
    mod._judge_call(cfg, "t-x", [("i", "q", "a")], 5)
    assert calls[-1]["cache_dir"] == tmp_path / "judge_cache" / "fu6" / "t-x"
    assert calls[-1]["kw"]["max_tokens"] == mod.JUDGE_MAX_TOKENS
    mod._judge_call(
        cfg,
        "k1-c000",
        [("i", "q", "a")],
        1,
        max_tokens=mod.RULE23_MAX_TOKENS,
        cache_root="fu6-rejudge-mt1000",
    )
    assert calls[-1]["cache_dir"] == tmp_path / "judge_cache" / "fu6-rejudge-mt1000" / "k1-c000"
    assert calls[-1]["kw"]["max_tokens"] == 1000
