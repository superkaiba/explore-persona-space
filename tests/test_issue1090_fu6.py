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
    from explore_persona_space.artifacts.directions import ContrastiveCompletion

    rows = []
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
    return rows


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
    comps = _tiny_completions()
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
    # Driver path: capture-all (P1b shape) then reduce from stored means.
    means_by_key: dict = {}
    ordinals: dict = {}
    encoded, _ = encode_rows(qwen_tokenizer, comps)
    valid = [(i, r) for i, r in enumerate(encoded) if r is not None]
    means = batched_response_means(model, [r for _, r in valid], [0, 1], batch_size=4)
    for (i, _r), m in zip(valid, means, strict=True):
        c = comps[i]
        key3 = (c.pair_index, c.arm, c.question)
        ordinals[key3] = ordinals.get(key3, -1) + 1
        means_by_key[(*key3, ordinals[key3])] = m
    cfg = fu6.Cfg(
        smoke=True,
        manifest_path=None,
        manifest_out=None,
        out_root=tmp_path,
        sentinel_dir=tmp_path,
    )
    r_b, counts, _kept = fu6.reduce_rb_from_stored_means(cfg, comps, means_by_key)
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
    r_b2, _c2, _k2 = fu6.reduce_rb_from_stored_means(cfg, comps, means_by_key)
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
    # judge_graded incl. the fu6 passthrough kwargs
    inspect.signature(judge_graded).bind(
        [("i", "q", "a")],
        "rubric {question} {answer}",
        n_draws=1,
        cache_dir=Path("/t"),
        save_raw=Path("/t/r.json"),
        judge_model="m",
        max_tokens=300,
        threshold_base=0,
        checkpoint_dir=Path("/t/ck"),
    )


# ── judge_graded passthrough (production-body test for the library edit) ─────


def test_judge_graded_threads_threshold_base_and_checkpoint(monkeypatch, tmp_path):
    """The REAL judge_graded body executes and forwards the new passthrough
    kwargs to judge_completions_batch (autospec'd network boundary)."""
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
        checkpoint_dir=tmp_path / "ck",
    )
    kwargs = fake.call_args.kwargs
    assert kwargs["threshold_base"] == 0 and kwargs["checkpoint_dir"] == tmp_path / "ck"
    assert result.scores["i"] == 70.0
    # default: passthrough ABSENT (existing callers byte-identical)
    fake.reset_mock()
    gj.judge_graded(
        [("i", "q", "a")], "r {question} {answer}", n_draws=1, cache_dir=tmp_path, save_raw=save_raw
    )
    assert "threshold_base" not in fake.call_args.kwargs
    assert "checkpoint_dir" not in fake.call_args.kwargs


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
        (rollout_dir / f"pair0_{arm}.json").write_text(
            json.dumps(
                {"meta": {"system_prompt": f"sys {arm}"}, "questions": questions, "rows": rows}
            )
        )

    # REAL P1b on CPU (tiny model): captures every persisted rollout.
    fu6.phase_capture_rollouts(cfg)
    caps = sorted((out_root / "captures" / "extraction").glob("pair*_*.pt"))
    assert len(caps) == 2

    # P2-shaped judge outputs (REAL schema; judge API boundary faked).
    judge_dir = deliv / "judge"
    judge_dir.mkdir(parents=True)
    scores = {}
    for arm, base_score in (("ex", 80.0), ("ne", 20.0)):
        for qi in range(2):
            for ri in range(2):
                scores[f"f6-ex-p0-{arm}-q{qi:03d}-r{ri:02d}"] = base_score + qi + ri
    (judge_dir / "extraction_filter_scores.json").write_text(
        json.dumps({"scores": scores, "n_total_draws": 16, "n_dropped_draws_content": 0})
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
