"""CPU-only tests for the #693 phase0 GPU-unit fan.

These describe the EXTERNAL behavior of the p0 parallelization added to
``scripts/issue664_dispatch.py`` (plan v2 §3/§6). The fan is a SCHEDULING change
only: the parallelized p0 must produce the SAME ``(kind, ctx_key)``-keyed disjoint
caches and the SAME ``baseline_propensity.json`` aggregate as the pre-change
single-shard p0. Every test here is CPU-only — no vLLM, no GPU, no live API:

  - 4.1   ``_p0_units`` enumerates the expected (kind, ctx_key) set (ic_secure shared);
  - 4.1b  parent/child SELECTION PARITY under --cells AND --live-judge-smoke, plus
          the corrupted-argv fingerprint-mismatch fail-loud (the MUST-FIX trio);
  - 4.2   --n-gpus 1 backcompat: every unit on gpu 0, CVD="0", --smoke threaded;
  - 4.3   sharded (n_gpus=2) == single-shard (n_gpus=1): same file SET + per-file
          json-equal contents (NOT byte-cmp), over a >=2x2x2 fixture + ic_secure;
  - 4.4   CellCmd CVD pin: env CUDA_VISIBLE_DEVICES == str(gpu_id) + --gpu-id in argv;
  - 4.5   judge-overlap ordering: submit BEFORE engine teardown, reconcile AFTER;
  - 4.5b  baseline aggregate completeness gate: a missing per-unit file raises;
  - 4.6   resume-skip: a pre-existing unit cache short-circuits the unit;
  - 4.7   duplicate-unit -> WaveDispatcher DuplicateCellError + _p0_units disjoint;
  - 4.8   build-mixes drop semantics (rc==3 dropped) preserved post-fan;
  - 4.8b  dry-run enumerates units WITHOUT spawning any subprocess.

The real vLLM / huggingface_hub clients are never imported at module-import time;
generation is stubbed via monkeypatch.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue664_common as C  # noqa: E402
import issue664_dispatch as D  # noqa: E402

from explore_persona_space.orchestrate.fleet import (  # noqa: E402
    CellCmd,
    DuplicateCellError,
    WaveDispatcher,
)

# ---------------------------------------------------------------------------
# Fixtures: a fake args namespace + a small synthetic cell fixture.
# ---------------------------------------------------------------------------


def _args(**over) -> types.SimpleNamespace:
    """A minimal args namespace carrying every field _select_cells / the fan read."""
    base = dict(
        phase="p0",
        cells=None,
        gpu_id=0,
        n_gpus=1,
        dry_run=False,
        smoke=False,
        live_judge_smoke=False,
        train_one_cell=False,
        extract_eval_one_cell=False,
        p0_one_unit=False,
        p0_kind=None,
        p0_ctx_key=None,
        p0_expect_sig=None,
        behavior=None,
        source=None,
        arm=None,
        dose=None,
        seed=C.DEFAULT_SEED,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _fixture_cells() -> list[C.Cell]:
    """A fixture spanning >=2 sources x >=2 negs x >=2 content-behaviors PLUS the
    ic_secure shared-gen path (em) + a marker behavior — so the merge test exercises
    every p0 unit kind including the shared generation (plan §4.3 CONCERN C). The
    negative panel is fixed (4 contexts) regardless of selection."""
    srcs = sorted(C.TRANSFER_SPINE_SOURCES)[:2]
    cells: list[C.Cell] = []
    for b in ("marker", "sycophancy", "refusal", "em"):
        for src in srcs:
            cells.append(C.Cell(b, src, "contra", "d1"))
    return cells


# ---------------------------------------------------------------------------
# 4.1 unit enumeration
# ---------------------------------------------------------------------------


def test_p0_units_enumeration():
    cells = _fixture_cells()
    units = D._p0_units(cells, smoke=False)
    got = {(u.kind, u.ctx_key) for u in units}

    sources = sorted({c.source for c in cells})
    neg_slugs = [n.slug for n in C.negative_panel()]
    expected: set[tuple[str, str]] = set()
    # marker_R per source + per neg.
    for src in sources:
        expected.add(("marker_R", src))
    for slug in neg_slugs:
        expected.add(("marker_R", slug))
    # syco_pos per source.
    for src in sources:
        expected.add(("syco_pos", src))
    # refusal_pos per source + refusal_neg per neg.
    for src in sources:
        expected.add(("refusal_pos", src))
    for slug in neg_slugs:
        expected.add(("refusal_neg", slug))
    # ic_secure: ONE shared unit (em present), NOT per source.
    expected.add(("ic_secure", D.IC_SECURE_SHARED_CTX))
    # baseline per (content-behavior, source). content behaviors in the fixture:
    # sycophancy, refusal, em (marker is NOT a content behavior).
    for b in ("sycophancy", "refusal", "em"):
        for src in sources:
            expected.add(("baseline", f"{b}__{src}"))

    assert got == expected, f"unit set mismatch:\n got={sorted(got)}\n exp={sorted(expected)}"
    # exactly ONE ic_secure unit (the shared generation, not per source).
    assert sum(1 for u in units if u.kind == "ic_secure") == 1
    # deterministic + duplicate-free + sorted by unit_key.
    keys = [u.unit_key for u in units]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# 4.1b parent/child SELECTION PARITY (the MUST-FIX trio)
# ---------------------------------------------------------------------------


def _child_args_from_argv(argv: list[str]) -> types.SimpleNamespace:
    """Parse a child argv (emitted by _p0_unit_cmd) back through the same flag set
    main() defines, returning the namespace the child's _select_cells would read."""
    a = _args()
    # Walk the argv emitted by _p0_unit_cmd; default everything _select_cells reads
    # to its 'unset' value, then flip per the threaded flags.
    a.cells = None
    a.smoke = False
    a.live_judge_smoke = False
    i = 0
    toks = list(argv)
    while i < len(toks):
        t = toks[i]
        if t == "--cells":
            a.cells = int(toks[i + 1])
            i += 2
        elif t == "--smoke":
            a.smoke = True
            i += 1
        elif t == "--live-judge-smoke":
            a.live_judge_smoke = True
            i += 1
        elif t == "--p0-kind":
            a.p0_kind = toks[i + 1]
            i += 2
        elif t == "--p0-ctx-key":
            a.p0_ctx_key = toks[i + 1]
            i += 2
        elif t == "--p0-expect-sig":
            a.p0_expect_sig = toks[i + 1]
            i += 2
        elif t == "--n-gpus":
            a.n_gpus = int(toks[i + 1])
            i += 2
        elif t == "--gpu-id":
            a.gpu_id = int(toks[i + 1])
            i += 2
        else:
            i += 1
    return a


def _selection_tuple(cells: list[C.Cell]) -> tuple:
    sources = tuple(sorted({c.source for c in cells}))
    behaviors = tuple(sorted({c.behavior for c in cells}))
    negs = tuple(sorted(n.slug for n in C.negative_panel()))
    return (sources, behaviors, negs)


def test_parent_child_selection_parity_cells():
    parent = _args(cells=5, n_gpus=4)
    parent_cells = D._select_cells(parent)
    sig = D._selection_sig(parent_cells)
    unit = D._p0_units(parent_cells, smoke=False)[0]

    cmd = D._p0_unit_cmd(unit, 1, parent, sig, smoke=parent.smoke)
    argv = list(cmd.argv)
    # the --cells cap MUST be threaded into the child argv.
    assert "--cells" in argv and argv[argv.index("--cells") + 1] == "5"

    child = _child_args_from_argv(argv)
    child_cells = D._select_cells(child)
    assert _selection_tuple(child_cells) == _selection_tuple(parent_cells)
    assert D._selection_sig(child_cells) == sig


def test_parent_child_selection_parity_live_judge_smoke():
    parent = _args(smoke=True, live_judge_smoke=True, cells=2, n_gpus=2)
    parent_cells = D._select_cells(parent)
    sig = D._selection_sig(parent_cells)
    unit = D._p0_units(parent_cells, smoke=parent.smoke)[0]

    cmd = D._p0_unit_cmd(unit, 0, parent, sig, smoke=parent.smoke)
    argv = list(cmd.argv)
    assert "--smoke" in argv
    assert "--live-judge-smoke" in argv

    child = _child_args_from_argv(argv)
    child_cells = D._select_cells(child)
    assert _selection_tuple(child_cells) == _selection_tuple(parent_cells)
    assert D._selection_sig(child_cells) == sig


def test_child_parity_assert_fires_on_dropped_flag(monkeypatch):
    """A corrupted child argv (a _select_cells-affecting flag dropped) makes
    _p0_run_one_unit raise the §3.2 fingerprint-mismatch RuntimeError."""
    parent = _args(cells=3, n_gpus=2)
    parent_cells = D._select_cells(parent)
    sig = D._selection_sig(parent_cells)
    unit = D._p0_units(parent_cells, smoke=False)[0]
    cmd = D._p0_unit_cmd(unit, 0, parent, sig, smoke=False)

    # Drop the --cells N pair from the child argv (the corruption).
    argv = list(cmd.argv)
    idx = argv.index("--cells")
    del argv[idx : idx + 2]
    child = _child_args_from_argv(argv)
    child.cells = None  # the dropped flag -> child reconstructs the FULL grid

    # The child must never reach the engine; _require_credentials / _vllm_engine
    # are stubbed so the test asserts the PARITY guard fires first.
    monkeypatch.setattr(D, "_require_credentials", lambda: None)
    monkeypatch.setattr(
        D, "_vllm_engine", lambda *a, **k: pytest.fail("engine spun before parity assert")
    )
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        D._p0_run_one_unit(child)


# ---------------------------------------------------------------------------
# Capture stub: monkeypatch WaveDispatcher.run to record build_cmd output without
# spawning subprocesses (the #693 fan's launch specs).
# ---------------------------------------------------------------------------


def _capture_wave(monkeypatch) -> list[CellCmd]:
    """Replace WaveDispatcher.run with a stub that builds (but does NOT launch) the
    CellCmd for every unit, honoring is_done resume-skip; returns the captured cmds."""
    captured: list[CellCmd] = []

    def _fake_run(self, cells, *, cwd=None):
        n = max(self.n_gpus, 1)
        todo = [c for c in cells if not self.is_done(c)]
        for wave_start in range(0, len(todo), n):
            wave = todo[wave_start : wave_start + n]
            for i, cell in enumerate(wave):
                captured.append(self.build_cmd(cell, i % n))
        return None

    monkeypatch.setattr(WaveDispatcher, "run", _fake_run)
    return captured


# ---------------------------------------------------------------------------
# 4.2 --n-gpus 1 backcompat
# ---------------------------------------------------------------------------


def test_n_gpus_1_backcompat(monkeypatch, tmp_path):
    captured = _capture_wave(monkeypatch)
    monkeypatch.setattr(D, "CACHE_ROOT", tmp_path / "cache")
    # pools are cheap CPU writes done before the fan; stub them out.
    monkeypatch.setattr(D, "_marker_question_pool", lambda smoke: ["q"])
    monkeypatch.setattr(D, "_refusal_request_pool", lambda smoke: ["q"])
    monkeypatch.setattr(D, "_write_pool", lambda *a, **k: None)
    # baseline aggregate + build-mixes are post-fan; stub them (n_gpus=1 still runs
    # them, but they are not what this test asserts).
    monkeypatch.setattr(D, "_aggregate_baseline_propensity", lambda *a, **k: None)
    monkeypatch.setattr(D, "_build_mixes", lambda *a, **k: [])
    monkeypatch.setattr(D, "_write_dropped_manifest", lambda dropped: None)

    args = _args(n_gpus=1, smoke=True)
    D.phase0(args)

    cells = D._select_cells(args)
    expected_units = D._p0_units(cells, smoke=True)
    assert len(captured) == len(expected_units), "every unit should be enqueued at n_gpus=1"
    for cmd in captured:
        assert cmd.gpu_id == 0
        assert cmd.env["CUDA_VISIBLE_DEVICES"] == "0"
        assert "--smoke" in cmd.argv  # smoke threaded into every child
        assert "--p0-one-unit" in cmd.argv


# ---------------------------------------------------------------------------
# 4.3 sharded == single-shard (the merge property)
# ---------------------------------------------------------------------------


def _stub_generation(monkeypatch, tmp_path):
    """Stub every GPU/IO touchpoint so _run_p0_unit writes deterministic caches with
    no vLLM, no HF download, no live judge. The cache contents are deterministic
    functions of (kind, ctx_key) so two runs over different shardings must agree."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(D, "CACHE_ROOT", cache)
    monkeypatch.setattr(D, "_require_credentials", lambda: None)
    monkeypatch.setattr(D, "_vllm_engine", lambda *a, **k: object())
    monkeypatch.setattr(D, "_teardown_vllm", lambda llm: None)
    monkeypatch.setattr(D, "_render", lambda messages: "RENDER")
    # deterministic greedy: one response per prompt, content-free + order-stable.
    monkeypatch.setattr(
        D, "_greedy", lambda llm, prompts, max_new: [f"R{i}" for i in range(len(prompts))]
    )
    # question pools: tiny + deterministic.
    monkeypatch.setattr(D, "_marker_question_pool", lambda smoke: ["mq0", "mq1"])
    monkeypatch.setattr(D, "_refusal_request_pool", lambda smoke: ["rq0", "rq1"])
    monkeypatch.setattr(D, "_baseline_probe_pool", lambda behavior, smoke: ["pb0", "pb1"])

    # secure-code generation: stub the HF download path inside _elicit_secure_code by
    # replacing the whole helper with a deterministic shared-gen fan.
    def _fake_secure(llm, sources, neg_panel, *, smoke):
        ctx_keys = D._ic_secure_ctx_keys(D._select_cells_for_secure())
        mapping = {"scq0": "sec0", "scq1": "sec1"}
        for k in ctx_keys:
            if (D.CACHE_ROOT / "ic_secure" / f"{k}.json").exists():
                continue
            D._write_responses_cache("ic_secure", k, mapping)

    return cache, _fake_secure


def _run_phase0_in_process(monkeypatch, args, fake_secure):
    """Drive phase0 with WaveDispatcher.run replaced by an in-process executor that
    actually runs _run_p0_unit per unit (exercising the real only_ctx helpers +
    cache writes) under the stubbed generation, then runs the post-fan aggregate."""

    def _exec_run(self, cells, *, cwd=None):
        todo = [c for c in cells if not self.is_done(c)]
        for unit in todo:
            judge_jobs: list = []
            llm = object()
            D._run_p0_unit(unit, llm, _select, judge_jobs, smoke=args.smoke)
            for job in judge_jobs:
                job()
        return None

    _select = D._select_cells(args)
    # _elicit_secure_code needs the SELECTED cells for its ctx_keys; thread them via
    # a tiny shim the fake reads.
    monkeypatch.setattr(D, "_select_cells_for_secure", lambda: _select, raising=False)
    monkeypatch.setattr(D, "_elicit_secure_code", fake_secure)
    monkeypatch.setattr(WaveDispatcher, "run", _exec_run)
    # build-mixes is post-fan + irrelevant to the cache-merge property; stub it.
    monkeypatch.setattr(D, "_build_mixes", lambda *a, **k: [])
    monkeypatch.setattr(D, "_write_dropped_manifest", lambda dropped: None)
    D.phase0(args)


# Wall-clock / commit metadata that legitimately differs between two runs and is
# NOT part of the merge property (which GPU ran which unit). Stripped recursively
# before the content comparison so the test asserts the DATA equality the fan must
# preserve, not the run timestamp.
_VOLATILE_META_KEYS = {"generated_at", "git_commit", "ts"}


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in _VOLATILE_META_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _read_cache_tree(cache: Path) -> dict[str, object]:
    """{repo-relative path -> parsed json (volatile metadata stripped)} over every
    *.json under the cache. Strips wall-clock / commit fields so the comparison
    asserts the merge DATA equality, not the run timestamp."""
    out: dict[str, object] = {}
    for p in sorted(cache.rglob("*.json")):
        out[str(p.relative_to(cache))] = _strip_volatile(json.loads(p.read_text()))
    return out


def test_sharded_eq_single_shard_merge(monkeypatch, tmp_path):
    fixture = _fixture_cells()
    monkeypatch.setattr(D, "realized_grid_cells", lambda: fixture, raising=False)

    # Pin _select_cells to the fixture for BOTH runs (so the only variable is n_gpus).
    monkeypatch.setattr(D, "_select_cells", lambda args: list(fixture))

    # --- single-shard (n_gpus=1) ---
    cache1, fake_secure1 = _stub_generation(monkeypatch, tmp_path / "shard1")
    args1 = _args(n_gpus=1, smoke=True)
    _run_phase0_in_process(monkeypatch, args1, fake_secure1)
    tree1 = _read_cache_tree(cache1)

    # --- sharded (n_gpus=2) ---
    cache2, fake_secure2 = _stub_generation(monkeypatch, tmp_path / "shard2")
    args2 = _args(n_gpus=2, smoke=True)
    _run_phase0_in_process(monkeypatch, args2, fake_secure2)
    tree2 = _read_cache_tree(cache2)

    # File SET equality is exact (a missing/extra file is a hard FAIL).
    assert set(tree1) == set(tree2), (
        f"sharded cache file set differs:\n only_in_1={sorted(set(tree1) - set(tree2))}\n "
        f"only_in_2={sorted(set(tree2) - set(tree1))}"
    )
    # Per-file content equality via json.loads (order-insensitive, NOT byte-cmp).
    for path in tree1:
        assert tree1[path] == tree2[path], f"content differs at {path}"
    # The merge actually produced the expected per-(kind,ctx) caches (not empty).
    assert any(p.startswith("marker_R/") for p in tree1)
    assert any(p.startswith("ic_secure/") for p in tree1)
    assert any(p.startswith("baseline_raw/") for p in tree1)


# ---------------------------------------------------------------------------
# 4.4 CellCmd CVD pin
# ---------------------------------------------------------------------------


def test_cellcmd_cvd_pin():
    cells = _fixture_cells()
    sig = D._selection_sig(cells)
    unit = D._p0_units(cells, smoke=False)[0]
    args = _args(n_gpus=4)
    for g in (0, 1, 2, 3):
        cmd = D._p0_unit_cmd(unit, g, args, sig, smoke=False)
        assert cmd.env["CUDA_VISIBLE_DEVICES"] == str(g)
        assert cmd.gpu_id == g
        argv = list(cmd.argv)
        assert "--gpu-id" in argv and argv[argv.index("--gpu-id") + 1] == str(g)
        assert "--p0-expect-sig" in argv and argv[argv.index("--p0-expect-sig") + 1] == sig


def test_cellcmd_missing_cvd_raises_in_run_parallel():
    """The inherited run_parallel_with_log assert turns a missing CVD into a loud
    pre-launch AssertionError (defense-in-depth, fleet.py)."""
    from explore_persona_space.orchestrate.fleet import run_parallel_with_log

    bad = CellCmd(
        cell_key="marker_R::default",
        argv=["true"],
        env={},  # CVD missing
        log_path=Path("/tmp/issue-693-cvd-test.log"),
        gpu_id=0,
    )
    with pytest.raises(AssertionError, match="CUDA_VISIBLE_DEVICES"):
        run_parallel_with_log([bad])


# ---------------------------------------------------------------------------
# 4.5 judge-overlap ordering
# ---------------------------------------------------------------------------


def test_judge_submit_then_reconcile_ordering(monkeypatch, tmp_path):
    """In a p0-unit subprocess, the judge is SUBMITTED (job appended) before engine
    teardown and RECONCILED (job run) AFTER teardown."""
    events: list[str] = []
    cache = tmp_path / "cache"
    monkeypatch.setattr(D, "CACHE_ROOT", cache)
    monkeypatch.setattr(D, "_require_credentials", lambda: None)
    monkeypatch.setattr(D, "_vllm_engine", lambda *a, **k: object())
    monkeypatch.setattr(D, "_teardown_vllm", lambda llm: events.append("teardown"))
    monkeypatch.setattr(D, "_select_cells", lambda args: _fixture_cells())

    def _fake_run_unit(unit, llm, cells, judge_jobs, *, smoke):
        events.append("generate")
        judge_jobs.append(lambda: events.append("reconcile"))

    monkeypatch.setattr(D, "_run_p0_unit", _fake_run_unit)

    cells = _fixture_cells()
    sig = D._selection_sig(cells)
    unit = D._p0_units(cells, smoke=False)[0]
    args = _args(p0_one_unit=True, p0_kind=unit.kind, p0_ctx_key=unit.ctx_key, p0_expect_sig=sig)
    rc = D._p0_run_one_unit(args)
    assert rc == 0
    # generate (which appends the judge job) happens BEFORE teardown; reconcile AFTER.
    assert events == ["generate", "teardown", "reconcile"], events


def test_baseline_aggregate_after_fan_before_build(monkeypatch, tmp_path):
    """phase0 runs the cross-unit baseline aggregate AFTER the WaveDispatcher fan
    returns and BEFORE build-mixes."""
    order: list[str] = []
    monkeypatch.setattr(D, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(D, "_marker_question_pool", lambda smoke: ["q"])
    monkeypatch.setattr(D, "_refusal_request_pool", lambda smoke: ["q"])
    monkeypatch.setattr(D, "_write_pool", lambda *a, **k: None)
    monkeypatch.setattr(WaveDispatcher, "run", lambda self, cells, **k: order.append("fan"))
    monkeypatch.setattr(
        D, "_aggregate_baseline_propensity", lambda *a, **k: order.append("aggregate")
    )
    monkeypatch.setattr(D, "_build_mixes", lambda *a, **k: order.append("build") or [])
    monkeypatch.setattr(D, "_write_dropped_manifest", lambda dropped: None)

    D.phase0(_args(n_gpus=2, smoke=True))
    assert order == ["fan", "aggregate", "build"], order


# ---------------------------------------------------------------------------
# 4.5b baseline aggregate completeness gate
# ---------------------------------------------------------------------------


def _write_baseline_unit_artifacts(cache: Path, behavior: str, src: str, *, with_scores: bool):
    raw_root = cache / "baseline_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / f"{behavior}__{src}.json").write_text(
        json.dumps(
            {
                "behavior": behavior,
                "source": src,
                "judge_column": C.BEHAVIOR_REGISTRY_PRIMARY_COLUMN[behavior],
                "rows": [{"question": "q0", "base_completion": "r0"}],
            }
        )
    )
    if with_scores:
        (raw_root / f"{behavior}__{src}__scores.json").write_text(
            json.dumps({"all_scores": {"cell__00000__00": {"behavior": 1}}})
        )


def _baseline_fixture_cells() -> list[C.Cell]:
    """A content-behavior-only fixture so the expected baseline set is well-defined."""
    srcs = sorted(C.TRANSFER_SPINE_SOURCES)[:2]
    return [C.Cell("sycophancy", src, "contra", "d1") for src in srcs]


def test_baseline_aggregate_complete_writes(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(D, "CACHE_ROOT", cache)
    cells = _baseline_fixture_cells()
    sources = sorted({c.source for c in cells})
    # mock the judge-score rate reader so no live API is touched.
    import issue664_eval as E

    monkeypatch.setattr(
        E, "_rate_from_raw_scores", lambda col, rows, scores: {"rate": 0.5, "n_judged": 1}
    )
    for src in sources:
        _write_baseline_unit_artifacts(cache, "sycophancy", src, with_scores=True)

    D._aggregate_baseline_propensity(cells, smoke=False)
    agg = json.loads((cache / "baseline_propensity.json").read_text())
    assert agg["rated_behaviors"] == ["sycophancy"]
    for src in sources:
        assert agg["judged_rates"]["sycophancy"][src]["rate"] == 0.5


def test_baseline_aggregate_missing_one_file_raises(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(D, "CACHE_ROOT", cache)
    cells = _baseline_fixture_cells()
    sources = sorted({c.source for c in cells})
    import issue664_eval as E

    monkeypatch.setattr(
        E, "_rate_from_raw_scores", lambda col, rows, scores: {"rate": 0.5, "n_judged": 1}
    )
    # Write BOTH sources' raw, but DROP one source's __scores.json.
    _write_baseline_unit_artifacts(cache, "sycophancy", sources[0], with_scores=True)
    _write_baseline_unit_artifacts(cache, "sycophancy", sources[1], with_scores=False)

    with pytest.raises(RuntimeError, match="missing per-unit judge scores"):
        D._aggregate_baseline_propensity(cells, smoke=False)
    # No partial aggregate written.
    assert not (cache / "baseline_propensity.json").exists()


def test_baseline_aggregate_missing_raw_raises(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(D, "CACHE_ROOT", cache)
    cells = _baseline_fixture_cells()
    sources = sorted({c.source for c in cells})
    # Only one source's raw exists; the other is entirely missing.
    _write_baseline_unit_artifacts(cache, "sycophancy", sources[0], with_scores=True)
    with pytest.raises(RuntimeError, match="missing per-unit raw artifact"):
        D._aggregate_baseline_propensity(cells, smoke=False)
    assert not (cache / "baseline_propensity.json").exists()


# ---------------------------------------------------------------------------
# 4.6 resume-skip
# ---------------------------------------------------------------------------


def test_resume_skip(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(D, "CACHE_ROOT", cache)
    cells = _fixture_cells()
    units = D._p0_units(cells, smoke=False)

    # Mark K of N units done by writing their output caches.
    done_units = [u for u in units if u.kind in ("marker_R", "syco_pos")]
    for u in done_units:
        out = cache / u.kind / f"{u.ctx_key}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
    for u in done_units:
        assert D._p0_unit_done(u, smoke=False, cells=cells) is True
    # An unwritten unit is NOT done.
    not_done = next(u for u in units if u.kind == "refusal_neg")
    assert D._p0_unit_done(not_done, smoke=False, cells=cells) is False

    # The capture stub honors is_done; only the remaining units get a build_cmd.
    captured = _capture_wave(monkeypatch)
    monkeypatch.setattr(D, "_marker_question_pool", lambda smoke: ["q"])
    monkeypatch.setattr(D, "_refusal_request_pool", lambda smoke: ["q"])
    monkeypatch.setattr(D, "_write_pool", lambda *a, **k: None)
    monkeypatch.setattr(D, "_aggregate_baseline_propensity", lambda *a, **k: None)
    monkeypatch.setattr(D, "_build_mixes", lambda *a, **k: [])
    monkeypatch.setattr(D, "_write_dropped_manifest", lambda dropped: None)
    monkeypatch.setattr(D, "_select_cells", lambda args: list(cells))
    D.phase0(_args(n_gpus=2))
    built_keys = {c.cell_key for c in captured}
    done_keys = {u.unit_key for u in done_units}
    assert done_keys.isdisjoint(built_keys), "a done unit was re-built"
    assert len(captured) == len(units) - len(done_units)


def test_ic_secure_resume_skip_requires_all_ctx_keys(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(D, "CACHE_ROOT", cache)
    cells = _fixture_cells()
    unit = D.P0Unit("ic_secure", D.IC_SECURE_SHARED_CTX)
    ctx_keys = D._ic_secure_ctx_keys(cells)
    # Write all-but-one ctx_key caches -> NOT done.
    for k in ctx_keys[:-1]:
        out = cache / "ic_secure" / f"{k}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
    assert D._p0_unit_done(unit, smoke=False, cells=cells) is False
    # Write the last -> done.
    (cache / "ic_secure" / f"{ctx_keys[-1]}.json").write_text("{}")
    assert D._p0_unit_done(unit, smoke=False, cells=cells) is True


# ---------------------------------------------------------------------------
# 4.7 disjoint-claim assert
# ---------------------------------------------------------------------------


def test_duplicate_unit_raises():
    """_p0_units returns disjoint keys; a malformed unit list with a dup key trips
    the WaveDispatcher DuplicateCellError."""
    cells = _fixture_cells()
    units = D._p0_units(cells, smoke=False)
    # _p0_units itself is disjoint.
    keys = [u.unit_key for u in units]
    assert len(keys) == len(set(keys))

    # A deliberately malformed list with a duplicate trips the fleet assert.
    dup = [units[0], units[0]]
    disp = WaveDispatcher(
        n_gpus=2,
        cell_key=lambda u: u.unit_key,
        is_done=lambda u: False,
        build_cmd=lambda u, g: CellCmd(
            cell_key=u.unit_key,
            argv=["true"],
            env={"CUDA_VISIBLE_DEVICES": str(g)},
            log_path=Path("/tmp/issue-693-dup.log"),
            gpu_id=g,
        ),
        dry_run=True,  # never actually launches
    )
    with pytest.raises(DuplicateCellError):
        disp.run(dup)


# ---------------------------------------------------------------------------
# 4.8 build-mixes drop semantics
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, rc):
        self.returncode = rc


def test_build_mixes_drop_semantics(monkeypatch, tmp_path):
    """rc==3 (B.DROPPED_SOURCE_EXIT) is dropped + continued; rc!=0 is fatal."""
    monkeypatch.setattr(D, "CACHE_ROOT", tmp_path / "cache")
    cells = _baseline_fixture_cells()  # 2 cells

    # First cell drops (rc==3), second succeeds (rc==0).
    rcs = iter([D.B.DROPPED_SOURCE_EXIT, 0])
    monkeypatch.setattr(D.subprocess, "run", lambda *a, **k: _FakeProc(next(rcs)))
    dropped = D._build_mixes(cells, smoke=True)
    assert dropped == [cells[0]]

    # A non-3 non-zero rc is fatal.
    monkeypatch.setattr(D.subprocess, "run", lambda *a, **k: _FakeProc(1))
    with pytest.raises(D.subprocess.CalledProcessError):
        D._build_mixes(cells, smoke=True)


# ---------------------------------------------------------------------------
# 4.8b dry-run enumerates without spawning
# ---------------------------------------------------------------------------


def test_p0_dry_run_enumerates_without_spawn(monkeypatch, tmp_path):
    """phase0(args, dry_run=True) enumerates the units that WOULD run and spawns ZERO
    subprocesses (the post-fan aggregate + build-mixes are also skipped)."""
    monkeypatch.setattr(D, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(D, "_marker_question_pool", lambda smoke: ["q"])
    monkeypatch.setattr(D, "_refusal_request_pool", lambda smoke: ["q"])
    monkeypatch.setattr(D, "_write_pool", lambda *a, **k: None)

    # run_parallel_with_log / subprocess.Popen must NEVER be called in dry-run.
    import explore_persona_space.orchestrate.fleet as fleet

    monkeypatch.setattr(
        fleet, "run_parallel_with_log", lambda *a, **k: pytest.fail("subprocess spawned in dry-run")
    )
    monkeypatch.setattr(
        D.subprocess, "Popen", lambda *a, **k: pytest.fail("Popen spawned in dry-run")
    )
    agg_called: list[bool] = []
    build_called: list[bool] = []
    monkeypatch.setattr(
        D, "_aggregate_baseline_propensity", lambda *a, **k: agg_called.append(True)
    )
    monkeypatch.setattr(D, "_build_mixes", lambda *a, **k: build_called.append(True) or [])
    monkeypatch.setattr(D, "_write_dropped_manifest", lambda dropped: None)

    args = _args(n_gpus=4, smoke=True, dry_run=True)
    D.phase0(args, dry_run=True)

    # dry-run skips the post-fan CPU barriers (no aggregate, no build-mixes).
    assert agg_called == []
    assert build_called == []
    # the units that WOULD run are enumerable (non-empty for the realized grid).
    units = D._p0_units(D._select_cells(args), smoke=True)
    assert len(units) > 0
