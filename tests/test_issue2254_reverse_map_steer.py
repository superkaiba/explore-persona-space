"""CPU-only tests for the #2254 `reverse_map_steer` round (round 7).

No network, no GPU: direction loading + unit-normalization from a tiny npz
fixture through the REAL ``phase_directions`` body (fakes only at the
HF/tokenizer boundaries via the driver's disclosed module seams), the
36-cell enumeration + slug/layer registration invariants, and the REAL
``phase_reduce`` on a synthetic full-36 fixture round against the COMMITTED
parent floor/band artifacts — including the hallucination ``band: null``
path (band loader seam forced absent) and the localize-band production
path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.issue2254_preimage as i2254
import scripts.issue2254_reverse_map_steer as rm
import scripts.issue2254_transpose_ladder as tl


def _args(out_root: Path, **overrides) -> argparse.Namespace:
    ns = rm.build_argparser().parse_args([])
    ns.out_root = str(out_root)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture()
def fixture_env(tmp_path, monkeypatch):
    """Fixture npz + parent bank + seam rebinds (the driver's disclosed
    --cpu-smoke seams); returns (args, rroot, raw {(behavior, layer): vec})."""
    src = tmp_path / "npz_src"
    raw = rm.make_fixture_npz(src)
    rm.make_fixture_parent_bank(tmp_path)
    monkeypatch.setattr(rm, "_NPZ_STAGER", rm._fixture_npz_stager_for(src))
    monkeypatch.setattr(rm, "_PARENT_VEC_LOADER", rm._fixture_parent_vec_loader)
    monkeypatch.setattr(rm, "_UPLOAD", rm._fixture_upload)
    monkeypatch.setattr(rm, "_UPLOAD_VERIFY", rm._fixture_upload_verify)
    monkeypatch.setattr(rm, "_DIR_HEADROOM", rm._fixture_dir_headroom)
    monkeypatch.setattr(rm, "_TOKENIZER_LOADER", tl._FixtureTokenizer)
    args = _args(tmp_path)
    return args, rm.round_root(tmp_path), raw


# ---------------------------------------------------------------------------
# registration invariants (driver-scoped parent extensions)
# ---------------------------------------------------------------------------


def test_parent_registry_extensions_registered_and_collision_free():
    # importing the driver registered the slug + L19 config
    assert i2254._DIR_SHORT[rm.SLUG] == rm.SLUG_SHORT
    assert i2254.LAYER_CONFIGS["L19"] == (19,)
    assert i2254.BREADTH_OF_CONFIG["L19"] == "single"
    # short token is unique across the registry (collision-free pin)
    shorts = list(i2254._DIR_SHORT.values())
    assert shorts.count(rm.SLUG_SHORT) == 1


def test_judge_ctx_id_length_fits_worst_case():
    cell = {
        "behavior": "hallucination",
        "kind": "steer",
        "direction": rm.SLUG,
        "position": "context",
        "layer_config": "L19",
        "c": 0.5,
    }
    cid = i2254._judge_ctx_id(cell, 43, 199)  # asserts <= 49 chars internally
    assert cid.startswith("hallucination-rvm-ctx-L19-c0p5")


# ---------------------------------------------------------------------------
# cell enumeration
# ---------------------------------------------------------------------------


def test_registered_cells_production_grid(tmp_path):
    cells = rm.registered_cells(_args(tmp_path, smoke=False))
    assert len(cells) == 36
    ids = [i2254._cell_id(c) for c in cells]
    assert len(set(ids)) == 36
    assert {c["behavior"] for c in cells} == set(rm.ROUND_BEHAVIORS)
    assert {c["layer_config"] for c in cells} == {"L14", "L19", "L26"}
    assert {c["c"] for c in cells} == {0.5, 1.0, 2.0, 4.0}
    assert all(c["position"] == "context" for c in cells)
    assert all(c["direction"] == rm.SLUG for c in cells)
    assert all("__rvm__ctx__" in cid for cid in ids)
    # 12 cells per behavior (the within-behavior multiplicity grain)
    for b in rm.ROUND_BEHAVIORS:
        assert sum(1 for c in cells if c["behavior"] == b) == rm.BEHAVIOR_FAMILY_SIZE


def test_registered_cells_smoke_single_cell(tmp_path):
    cells = rm.registered_cells(_args(tmp_path, smoke=True))
    assert len(cells) == 1
    assert i2254._cell_id(cells[0]) == "evil__rvm__ctx__L14__c4"


# ---------------------------------------------------------------------------
# directions: loading + unit-normalization from the tiny npz fixture
# ---------------------------------------------------------------------------


def test_phase_directions_unit_normalizes_and_reports(fixture_env):
    args, rroot, raw = fixture_env
    rm.phase_directions(args)
    report = json.loads((rroot / "revmap_report.json").read_text())
    assert report["n_direction_files"] == 9
    assert set(report["npz_sha12"]) == {"L14", "L19", "L26"}
    bank_root = Path(args.out_root)
    for (b, ly), vec in raw.items():
        loaded = tl._tiny_bank_load(bank_root, b, rm.SLUG, ly)
        # unit norm on disk-loaded vector
        assert abs(float(np.linalg.norm(loaded)) - 1.0) < 1e-6
        # direction preserved (cos ≈ 1 vs the raw fixture vector)
        cos = float(vec @ loaded / (np.linalg.norm(vec) * np.linalg.norm(loaded)))
        assert cos > 1.0 - 1e-6, (b, ly, cos)
        row = report["layers"][str(ly)][b]
        assert abs(row["raw_norm"] - float(np.linalg.norm(vec))) < 1e-9
        assert row["loader_roundtrip_cos"] >= tl.LOADER_ROUNDTRIP_MIN_COS
        # cosines vs the fixture parent bank exist (pre + ctxext present)
        assert row["cos_vs_parent_pre"] is not None
        assert row["cos_vs_ctxext"] is not None


def test_extract_rev_direction_missing_key_raises():
    class _Z:
        files = ("evil_rev",)

        def __contains__(self, k):
            return k in self.files

        def __getitem__(self, k):
            return np.ones(4)

    with pytest.raises(rm.RevmapHaltError, match="missing key"):
        rm._extract_rev_direction(_Z(), "sycophancy", 14)


def test_extract_rev_direction_zero_norm_raises():
    class _Z:
        files = ("evil_rev",)

        def __contains__(self, k):
            return k in self.files

        def __getitem__(self, k):
            return np.zeros(8)

    with pytest.raises(rm.RevmapHaltError, match="degenerate norm"):
        rm._extract_rev_direction(_Z(), "evil", 14)


# ---------------------------------------------------------------------------
# reduce: fixture round (banded path + the hallucination band-null path)
# ---------------------------------------------------------------------------


def _run_reduce(tmp_path, monkeypatch, band_loader) -> dict:
    monkeypatch.setattr(rm, "_TOKENIZER_LOADER", tl._FixtureTokenizer)
    monkeypatch.setattr(rm, "_HALLU_BAND_LOADER", band_loader)
    args = _args(tmp_path, smoke=False)
    rroot = rm.round_root(tmp_path)
    rm.make_fixture_round(rroot, args)
    rm.phase_reduce(args)
    return json.loads((rroot / "reduce" / "verdicts.json").read_text())


def test_reduce_banded_localize_band(tmp_path, monkeypatch):
    verdicts = _run_reduce(tmp_path, monkeypatch, rm._hallucination_band_production)
    # hallucination band found via the localize search leg (committed artifact)
    assert verdicts["bands"]["hallucination"] is not None
    assert verdicts["band_sources"]["hallucination"]["wave"] == "localize"
    assert verdicts["hallucination_no_band_caveat"] is False
    # evil band 0.0 / sycophancy 10.89 from the decisive artifact
    assert verdicts["bands"]["evil"] == 0.0
    assert abs(verdicts["bands"]["sycophancy"] - 10.890474999999999) < 1e-9
    # the designed clearing sycophancy cell clears (delta 30 > band 10.89)
    assert "sycophancy__rvm__ctx__L14__c4" in verdicts["h1_clearing_cells"]
    assert verdicts["label"] == "H1"
    # the designed coherence-fail evil cell is Undefined
    assert "evil__rvm__ctx__L26__c0p5" in verdicts["narration"]["undefined_cells"]
    # evil delta 0 vs band 0: strict > means no evil cell clears
    assert not any(c.startswith("evil__") for c in verdicts["h1_clearing_cells"])
    # measured-direction positive control: READ, with missing combos noted
    fixture = verdicts["measured_direction_fixture"]
    assert fixture["source"] == rm.MEASURED_FIXTURE_REL
    assert "evil__L14" in fixture["reads"]  # parent evil cxd L14 c4 exists
    assert "hallucination__L14" in fixture["missing"]  # never tested by parent
    assert "evil__L19" in fixture["missing"]  # no parent cxd cell at L19


def test_reduce_hallucination_band_null_path(tmp_path, monkeypatch):
    verdicts = _run_reduce(tmp_path, monkeypatch, rm._fixture_hallu_band_absent)
    assert verdicts["bands"]["hallucination"] is None
    assert verdicts["hallucination_no_band_caveat"] is True
    hcells = {
        cid: row for cid, row in verdicts["cells"].items() if cid.startswith("hallucination__")
    }
    assert len(hcells) == rm.BEHAVIOR_FAMILY_SIZE
    defined_h = {cid: r for cid, r in hcells.items() if r.get("delta_score") is not None}
    assert defined_h, "hallucination cells must carry raw deltas on the band-null path"
    for cid, row in defined_h.items():
        assert row["margin"] is None, cid
        assert row["clears_nominal"] is None, cid
        assert row["no_band_caveat"] is True, cid
        assert row["label"] == "no-band (raw Δ vs floor only)"
        # raw delta preserved (fixture wrote +5.0 per question)
        assert abs(row["delta_score"] - 5.0) < 1e-9, cid
    assert sorted(verdicts["narration"]["no_band_cells"]) == sorted(defined_h)
    # band-less cells never enter the clearing set or the selection companions
    assert not any(c.startswith("hallucination__") for c in verdicts["h1_clearing_cells"])
    assert verdicts["selection_aware"]["behavior"]["hallucination"] is None


def test_reduce_grain_refusal(tmp_path, monkeypatch):
    """A truncated question grain is refused in every mode (no smoke
    downgrade)."""
    monkeypatch.setattr(rm, "_TOKENIZER_LOADER", tl._FixtureTokenizer)
    monkeypatch.setattr(rm, "_HALLU_BAND_LOADER", rm._fixture_hallu_band_absent)
    args = _args(tmp_path, smoke=False)
    rroot = rm.round_root(tmp_path)
    rm.make_fixture_round(rroot, args)
    # truncate one judged cell's per-question vector
    cid = "evil__rvm__ctx__L14__c1"
    jpath = rroot / "judge" / "judged" / f"{cid}.json"
    judged = json.loads(jpath.read_text())
    judged["per_question_mean_score"] = judged["per_question_mean_score"][:5]
    jpath.write_text(json.dumps(judged))
    with pytest.raises(RuntimeError, match="truncated grain"):
        rm.phase_reduce(args)


# ---------------------------------------------------------------------------
# judge modes: wave vs per-cell checkpoint parity + cached-skip in wave mode
# ---------------------------------------------------------------------------

_N_Q_JUDGE = 2  # fixture question grain for the judge-mode tests


def _judge_fixture_cells(tmp_path) -> list[dict]:
    """Four registered cells across two behaviors (2 evil + 2 sycophancy)."""
    cells = rm.registered_cells(_args(tmp_path, smoke=False))
    picked = [c for c in cells if c["behavior"] == "evil"][:2]
    picked += [c for c in cells if c["behavior"] == "sycophancy"][:2]
    assert len(picked) == 4
    return picked


def _judge_fixture_gen_rec(cell: dict) -> dict:
    """Tiny gen record in the exact shape ``i2254._iter_gen_qa`` consumes."""
    return {
        "cell_id": i2254._cell_id(cell),
        "cell": cell,
        "alphas": {"L14": 0.1},
        "q_of_context": list(range(_N_Q_JUDGE)),
        "seeds": {
            "42": {
                "completions": [["fixture answer a", "fixture answer b"]] * _N_Q_JUDGE,
                "condition_passes": [True] * _N_Q_JUDGE,
            }
        },
        "max_new_tokens": 2048,
        "cap_hit_fraction": 0.0,
    }


class _FakeGradedJudge:
    """Deterministic stand-in for ``fk._judge_graded_with_refusal_reissue``
    (the external API boundary): per-(item, draw) scores derived from a hash
    of the item id, so per-cell and wave submissions of the SAME items score
    identically. Thread-safe call recording (wave mode calls from threads)."""

    def __init__(self):
        import threading

        self.calls: list[dict] = []
        self._lock = threading.Lock()

    @staticmethod
    def _score(iid: str, draw: int) -> float:
        import hashlib

        return float(int(hashlib.sha256(f"{iid}:{draw}".encode()).hexdigest()[:4], 16) % 101)

    def __call__(self, items, rubric, *, cache_dir, save_raw, n_draws, force_sync=False):
        from explore_persona_space.eval.graded_judge import JudgeResult

        with self._lock:
            self.calls.append({"rubric": rubric, "iids": [it[0] for it in items]})
        per = {iid: [self._score(iid, d) for d in range(n_draws)] for iid, _q, _a in items}
        result = JudgeResult(
            scores={iid: float(np.mean(v)) for iid, v in per.items()},
            n_total_draws=len(items) * n_draws,
            n_dropped_draws=0,
            per_item_draw_counts={iid: n_draws for iid in per},
            per_item_scores={iid: list(v) for iid, v in per.items()},
        )
        return result, {iid: list(v) for iid, v in per.items()}, None


@pytest.fixture()
def judge_mode_env(tmp_path, monkeypatch):
    fake = _FakeGradedJudge()
    monkeypatch.setattr(rm.fk, "_judge_graded_with_refusal_reissue", fake)
    monkeypatch.setattr(
        i2254, "_eval_questions", lambda b: [f"{b} fixture q{i}" for i in range(_N_Q_JUDGE)]
    )
    cells = _judge_fixture_cells(tmp_path)
    rubrics = {b: f"rubric-{b}" for b in rm.ROUND_BEHAVIORS}
    return fake, cells, rubrics


def _write_gen_recs(root: Path, cells: list[dict]) -> Path:
    comp = root / "steer" / "raw_completions"
    for c in cells:
        i2254._write_json_atomic(comp / f"{i2254._cell_id(c)}.json", _judge_fixture_gen_rec(c))
    return comp


def test_wave_and_percell_checkpoints_identical(tmp_path, judge_mode_env):
    fake, cells, rubrics = judge_mode_env
    args = _args(tmp_path)
    n_draws = 3
    root_a = tmp_path / "percell_root"
    root_b = tmp_path / "wave_root"
    comp_a = _write_gen_recs(root_a, cells)
    comp_b = _write_gen_recs(root_b, cells)
    for c in cells:
        cid = i2254._cell_id(c)
        rm._judge_revmap_cell(args, root_a, comp_a / f"{cid}.json", rubrics[c["behavior"]], n_draws)
    n_percell_calls = len(fake.calls)
    assert n_percell_calls == len(cells)  # per-cell mode: one submission per cell
    cell_by_id = {i2254._cell_id(c): c for c in cells}
    n_judged = rm._judge_revmap_wave(args, root_b, comp_b, cell_by_id, rubrics, n_draws)
    assert n_judged == len(cells)
    wave_calls = fake.calls[n_percell_calls:]
    # wave mode: ONE submission per behavior, never one per cell
    assert len(wave_calls) == 2
    assert sorted(c["rubric"] for c in wave_calls) == ["rubric-evil", "rubric-sycophancy"]
    for call in wave_calls:
        behavior = call["rubric"].removeprefix("rubric-")
        assert all(iid.startswith(behavior) for iid in call["iids"])
    for c in cells:
        cid = i2254._cell_id(c)
        a = json.loads((root_a / "judge" / "judged" / f"{cid}.json").read_text())
        b = json.loads((root_b / "judge" / "judged" / f"{cid}.json").read_text())
        assert set(a) == set(b), cid  # identical top-level keys
        assert set(a["accounting"]) == set(b["accounting"]), cid
        assert set(a["judge"]) == set(b["judge"]), cid
        # identical scores + item meta (the reduce-consumed surface)
        assert a["per_item_scores_merged"] == b["per_item_scores_merged"], cid
        for k in (
            "cell_id",
            "items",
            "gen_sha",
            "judge_fp",
            "n_questions",
            "per_question_mean_score",
            "per_question_rate",
            "per_question_n",
            "mean_score",
            "rate",
            "coherence_rate",
            "coherence_pass",
        ):
            assert a[k] == b[k], (cid, k)
        for k in ("frac_items_complete", "n_total_draws", "n_items", "n_items_zero_valid"):
            assert a["accounting"][k] == b["accounting"][k], (cid, k)


def test_wave_mode_skips_already_judged_cells(tmp_path, judge_mode_env):
    fake, cells, rubrics = judge_mode_env
    args = _args(tmp_path)
    n_draws = 2
    root = tmp_path / "wave_skip_root"
    comp = _write_gen_recs(root, cells)
    pre = cells[0]
    pre_cid = i2254._cell_id(pre)
    rm._judge_revmap_cell(args, root, comp / f"{pre_cid}.json", rubrics[pre["behavior"]], n_draws)
    ckpt = root / "judge" / "judged" / f"{pre_cid}.json"
    before_bytes = ckpt.read_bytes()
    before_mtime = ckpt.stat().st_mtime_ns
    fake.calls.clear()
    cell_by_id = {i2254._cell_id(c): c for c in cells}
    n_judged = rm._judge_revmap_wave(args, root, comp, cell_by_id, rubrics, n_draws)
    assert n_judged == len(cells)  # cached + freshly judged
    # the pre-judged cell's items never re-entered any wave submission
    wave_iids = {iid for call in fake.calls for iid in call["iids"]}
    rec = json.loads((comp / f"{pre_cid}.json").read_text())
    items, _meta = rm._judge_cell_items(rec)
    assert wave_iids.isdisjoint({it[0] for it in items})
    # checkpoint untouched (bytes + mtime)
    assert ckpt.read_bytes() == before_bytes
    assert ckpt.stat().st_mtime_ns == before_mtime
    # remaining 3 cells judged, all 4 checkpoints now present
    assert len(list((root / "judge" / "judged").glob("*.json"))) == len(cells)


# ---------------------------------------------------------------------------
# --judge-route: force_sync plumbing down to the dispatch call (both modes)
# ---------------------------------------------------------------------------


class _CaptureJudgeCompletionsBatch:
    """Stand-in for ``batch_judge.judge_completions_batch`` (the dispatch
    seam): records the ``force_sync`` each call received and writes the
    minimal ``save_raw`` payload ``judge_result_from_save_raw`` needs, so the
    REAL wrapper chain (fk._judge_graded_with_refusal_reissue →
    judge_items_graded → judge_graded) runs end-to-end."""

    def __init__(self):
        import threading

        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, **kw):
        with self._lock:
            self.calls.append({"force_sync": kw["force_sync"], "n_items": len(kw["completions"])})
        save_raw = Path(kw["save_raw"])
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(json.dumps({"all_scores": {}}))
        return {}


def _judge_route_env(tmp_path, monkeypatch):
    from explore_persona_space.eval import batch_judge

    cap = _CaptureJudgeCompletionsBatch()
    monkeypatch.setattr(batch_judge, "judge_completions_batch", cap)
    monkeypatch.setattr(
        i2254, "_eval_questions", lambda b: [f"{b} fixture q{i}" for i in range(_N_Q_JUDGE)]
    )
    cells = _judge_fixture_cells(tmp_path)
    rubrics = {b: f"rubric-{b}" for b in rm.ROUND_BEHAVIORS}
    return cap, cells, rubrics


def test_judge_route_sync_forces_sync_dispatch(tmp_path, monkeypatch):
    cap, cells, rubrics = _judge_route_env(tmp_path, monkeypatch)
    args = _args(tmp_path, judge_route="sync")
    root = tmp_path / "sync_root"
    comp = _write_gen_recs(root, cells)
    c0 = cells[0]
    cid0 = i2254._cell_id(c0)
    rm._judge_revmap_cell(args, root, comp / f"{cid0}.json", rubrics[c0["behavior"]], 2)
    assert [c["force_sync"] for c in cap.calls] == [True]  # per-cell mode honors sync
    cell_by_id = {i2254._cell_id(c): c for c in cells}
    n_judged = rm._judge_revmap_wave(args, root, comp, cell_by_id, rubrics, 2)
    assert n_judged == len(cells)
    # 1 per-cell call + 2 behavior waves (cell 0 cached-skipped), ALL sync
    assert len(cap.calls) == 3
    assert all(c["force_sync"] is True for c in cap.calls)


def test_judge_route_auto_passes_force_sync_false(tmp_path, monkeypatch):
    cap, cells, rubrics = _judge_route_env(tmp_path, monkeypatch)
    args = _args(tmp_path)
    assert args.judge_route == "auto"  # the parser default
    root = tmp_path / "auto_root"
    comp = _write_gen_recs(root, cells)
    c0 = cells[0]
    cid0 = i2254._cell_id(c0)
    rm._judge_revmap_cell(args, root, comp / f"{cid0}.json", rubrics[c0["behavior"]], 2)
    cell_by_id = {i2254._cell_id(c): c for c in cells}
    n_judged = rm._judge_revmap_wave(args, root, comp, cell_by_id, rubrics, 2)
    assert n_judged == len(cells)
    assert len(cap.calls) == 3
    assert all(c["force_sync"] is False for c in cap.calls)
