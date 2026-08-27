"""#2617 round-3 regression pins (CPU / no-network; torch only via the reads
module import).

Pins the two round-3 union-BLOCKER fixes plus the concern-row closures:

- BLOCKER model-load-before-input-contract: ``_phase_input_gate`` asserts
  each PENDING phase's upstream inputs exist (or its producer is scheduled
  EARLIER in the same pending sequence) BEFORE ``load_model_and_tokenizer``
  — synthesized margin-only / capture-only roots with missing prerequisites
  raise with ZERO model loads (loader spy through the REAL ``main()`` body),
  plus a source-order pin (gate call index < load index);
- BLOCKER margin-source-fingerprint: ``_margin_fp`` tracks the ANCHOR opener
  source content and the ``--allow-short-pools`` waiver, and
  ``_margin_complete`` re-validates the sentinel's recorded pool-content sha
  against the on-disk pools.json;
- r2 minor: ``_finalize_fp`` folds the four upstream sentinels' recorded fps;
- CONCERN phase-sentinels-not-durable: finalize's durability upload precedes
  the local terminal-sentinel write (source-order pin);
- CONCERN judge-timeout-fallback-missing: ``judge_completions_batch``
  write-through-caches each sync-path item's verdict the moment it completes
  (production-body test — real body, fake only at the anthropic client
  boundary, mid-wave cache observation);
- CONCERN overflow-staging-disconnected: the overflow fallback/pointer path
  fetches from the OVERFLOW repo as a MODEL repo, reroute pointers are read
  from svmp_done.json's upload records and consulted BEFORE the canonical
  fetch, and ``stage_ridge_payloads_svmp`` returns its realized revision.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.issue2617_svmp_run as run


def _read_jsonl_stub(path, tolerate_torn_tail=False):
    return [
        json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").split("\n") if ln.strip()
    ]


def _stub_langow() -> SimpleNamespace:
    def _sha16(obj) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _read_json(path):
        p = Path(path)
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    return SimpleNamespace(
        _sha16=_sha16,
        _read_json=_read_json,
        _read_jsonl=_read_jsonl_stub,
        PIN="stubpin2564",
        ANCHOR_TEMPERATURE=0.8,
    )


def _cfg(tmp_path: Path, *, tiny: bool = True, **extra) -> SimpleNamespace:
    root = tmp_path / "root"
    ns = SimpleNamespace(
        model_id="stub-model",
        model_revision="stub-rev",
        tiny=tiny,
        draws=1,
        gen_batch=2,
        seed_base=0,
        max_new_tokens=8,
        upload=False,
        out_root=root,
        manifest_dir=root / "manifests",
        anchors_dir=root / "anchors",
    )
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_anchors(cfg, rows: list[dict]) -> None:
    path = cfg.anchors_dir / f"anchors_{run.CELL}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ── BLOCKER margin-source-fingerprint ────────────────────────────────────────


def test_margin_fp_tracks_anchor_content(tmp_path, monkeypatch):
    """Fixed judge scores + a changed opener SOURCE (anchor rollout text) must
    invalidate the margin fingerprint (the r2 fp saw judge content only)."""
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path, tiny=False)
    fp_absent = run._margin_fp(cfg)
    _write_anchors(cfg, [{"context_id": "c1", "draw": 0, "text": "I cannot help with that."}])
    fp_v1 = run._margin_fp(cfg)
    assert fp_v1 != fp_absent
    # SAME judge scores, different opener source -> fp must change.
    _write_anchors(cfg, [{"context_id": "c1", "draw": 0, "text": "Sure, here is how you do it."}])
    assert run._margin_fp(cfg) != fp_v1
    # identical rewrite -> unchanged (machine-stable file-content key).
    _write_anchors(cfg, [{"context_id": "c1", "draw": 0, "text": "Sure, here is how you do it."}])
    fp_v2 = run._margin_fp(cfg)
    _write_anchors(cfg, [{"context_id": "c1", "draw": 0, "text": "Sure, here is how you do it."}])
    assert run._margin_fp(cfg) == fp_v2


def test_margin_fp_tracks_allow_short_pools(tmp_path, monkeypatch):
    """The --allow-short-pools waiver changes the realized pools, so it is
    part of the margin fingerprint: a completed waived run is never silently
    accepted by an unwaived re-entry (and vice versa)."""
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path)
    base = run._margin_fp(cfg)  # attr absent -> False
    cfg.allow_short_pools = False
    assert run._margin_fp(cfg) == base, "explicit False must equal absent"
    cfg.allow_short_pools = True
    assert run._margin_fp(cfg) != base, "the waiver must invalidate the fp"


def test_margin_complete_validates_pools_content_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path)
    pools_obj = {
        "refusal": [{"answer": "I cannot"}],
        "helpful": [{"answer": "Sure,"}],
        "meta": {"pool_size": 1},
    }
    _write_json(cfg.out_root / "margin" / "pools.json", pools_obj)
    _write_json(cfg.out_root / "margin" / "margins.json", {"rows": []})
    _write_json(
        cfg.out_root / "svmp_margin_done.json",
        {"regime_fp": run._margin_fp(cfg), "pools_sha": run._pools_content_sha(pools_obj)},
    )
    assert run._margin_complete(cfg)
    # (a) on-disk pools drift under a matching fp -> completion refused.
    drifted = dict(pools_obj, refusal=[{"answer": "TAMPERED"}])
    _write_json(cfg.out_root / "margin" / "pools.json", drifted)
    assert not run._margin_complete(cfg)
    _write_json(cfg.out_root / "margin" / "pools.json", pools_obj)
    assert run._margin_complete(cfg)
    # (b) a legacy sentinel with NO recorded pools_sha is never accepted.
    _write_json(cfg.out_root / "svmp_margin_done.json", {"regime_fp": run._margin_fp(cfg)})
    assert not run._margin_complete(cfg)


def test_pools_content_sha_ignores_repro_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    a = {"refusal": [1], "helpful": [2], "meta": {}, "repro": {"timestamp": "t0"}}
    b = {"refusal": [1], "helpful": [2], "meta": {}, "repro": {"timestamp": "t1-later"}}
    assert run._pools_content_sha(a) == run._pools_content_sha(b)


# ── r2 minor: finalize fp folds upstream fps ────────────────────────────────


def test_finalize_fp_folds_upstream_sentinel_fps(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path)
    for name in ("gen", "capture", "judge", "margin"):
        _write_json(cfg.out_root / f"svmp_{name}_done.json", {"regime_fp": f"fp-{name}"})
    _write_json(cfg.out_root / "svmp_done.json", {"regime_fp": run._finalize_fp(cfg)})
    assert run._finalize_complete(cfg)
    # A within-invocation upstream re-run rewrites its sentinel fp -> the
    # stale terminal sentinel must stop matching.
    _write_json(cfg.out_root / "svmp_margin_done.json", {"regime_fp": "fp-margin-NEW"})
    assert not run._finalize_complete(cfg)


# ── BLOCKER model-load-before-input-contract ────────────────────────────────


def test_input_gate_margin_missing_judge_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path, tiny=False)
    _write_anchors(cfg, [{"context_id": "c1", "draw": 0, "text": "x"}])
    with pytest.raises(RuntimeError, match="judge_scores"):
        run._phase_input_gate(cfg, ["margin"])
    # producer scheduled EARLIER in the same pending sequence satisfies it.
    _write_json(cfg.manifest_dir / "svmp_bank.json", {})
    run._phase_input_gate(cfg, ["judge", "margin"])
    # inputs on disk satisfy it too.
    _write_json(cfg.out_root / "judge" / "judge_scores.json", {})
    run._phase_input_gate(cfg, ["margin"])


def test_input_gate_tiny_margin_needs_no_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path, tiny=True)
    run._phase_input_gate(cfg, ["margin"])  # canned pools: no judge/anchors read


def test_input_gate_capture_missing_anchors(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path, tiny=False)
    with pytest.raises(RuntimeError, match="anchors"):
        run._phase_input_gate(cfg, ["capture"])
    run._phase_input_gate(cfg, ["gen", "capture"])  # gen earlier -> satisfied


def test_input_gate_finalize_missing_sentinels(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path, tiny=False)
    with pytest.raises(RuntimeError, match="svmp_gen_done"):
        run._phase_input_gate(cfg, ["finalize"])
    # the full pending chain produces everything earlier -> no raise.
    run._phase_input_gate(cfg, ["gen", "capture", "judge", "margin", "finalize"])


def test_main_input_gate_precedes_model_load():
    """Source-order pin (the round-2 pin's sibling): the per-phase input gate
    runs BEFORE the first load_model_and_tokenizer reference in main()."""
    src = inspect.getsource(run.main)
    assert src.index("_phase_input_gate") < src.index("load_model_and_tokenizer")


def test_main_loop_rechecks_completion_at_loop_time():
    """r2 minor pin: the phase loop re-evaluates PHASE_COMPLETE for
    preflight-skipped phases so an upstream re-run invalidates the skip."""
    src = inspect.getsource(run.main)
    assert "if phase in skipped" in src
    assert src.count("PHASE_COMPLETE[") >= 2, "loop-time re-check missing"


def _main_stub_env(tmp_path, monkeypatch, phase: str):
    """Drive the REAL main() body to the input gate with a loader spy: fakes
    live only at the langow reuse boundary (build_cfg / completion helpers /
    R.load_model_and_tokenizer), mirroring the real attribute surface."""
    load_calls: list[str] = []
    Lstub = _stub_langow()
    out_root = tmp_path / "root"
    cfg = _cfg(tmp_path, tiny=False, device="cpu", allow_short_pools=False)

    def _load_model_and_tokenizer(cfg):
        load_calls.append("load")
        return object(), object()

    Lstub.build_cfg = lambda args: cfg
    Lstub._gen_cell_complete = lambda cfg, c: False
    Lstub._va_cell_complete = lambda cfg, c: False
    Lstub._vc_complete = lambda cfg: False
    Lstub.R = SimpleNamespace(load_model_and_tokenizer=_load_model_and_tokenizer)
    args = SimpleNamespace(
        import_check=False,
        bank_check=False,
        judge_live_probe=False,
        phase=phase,
        allow_short_pools=False,
        out_root=str(out_root),
    )
    monkeypatch.setattr(run, "build_argparser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(run, "_load_langow", lambda: None)
    monkeypatch.setattr(run, "L", Lstub)
    return cfg, load_calls


def test_main_margin_reentry_blocks_before_model_load(tmp_path, monkeypatch):
    """Synthesized margin-only re-entry with no judge_scores.json: the REAL
    main() body raises the prerequisite error with ZERO model loads."""
    _, load_calls = _main_stub_env(tmp_path, monkeypatch, phase="margin")
    with pytest.raises(RuntimeError, match="judge_scores"):
        run.main()
    assert load_calls == [], "input gate must fire before any model load"


def test_main_capture_reentry_blocks_before_model_load(tmp_path, monkeypatch):
    _, load_calls = _main_stub_env(tmp_path, monkeypatch, phase="capture")
    with pytest.raises(RuntimeError, match="anchors"):
        run.main()
    assert load_calls == [], "input gate must fire before any model load"


# ── CONCERN phase-sentinels-not-durable (finalize ordering) ─────────────────


def test_finalize_durability_upload_precedes_local_write():
    src = inspect.getsource(run.phase_finalize)
    i_stage = src.index(".finalize_stage")
    i_local = src.index('cfg.out_root / "svmp_done.json"')
    assert i_stage < i_local, "durability upload must precede the local terminal sentinel"
    assert src.index("durability_upload") < i_local


# ── CONCERN judge-timeout-fallback-missing (cache write-through) ────────────


def test_judge_cache_write_through_persists_mid_wave(tmp_path):
    """PRODUCTION-BODY test (code-style.md rule): executes the real
    judge_completions_batch -> dispatch_judge_items -> _judge_items_sync chain
    with a fake ONLY at the anthropic-client network boundary. The rubric is
    sized over CACHE_MIN_TOKENS so the cache-warm ordering awaits item 1
    BEFORE item 2's call starts; the client observes the cache dir at call 2 —
    a populated cache there is exactly the mid-wave-kill salvage property."""
    from explore_persona_space.eval import batch_judge as bj

    cache_dir = tmp_path / "cache"
    observed: dict = {}

    class _FakeMessages:
        def __init__(self):
            self.n = 0

        async def create(self, **params):
            # Anthropic-client boundary fake: assert the real call shape.
            assert {"model", "messages", "max_tokens"} <= set(params), sorted(params)
            self.n += 1
            if self.n == 2:
                observed["cache_files_at_call2"] = len(list(cache_dir.rglob("*.json")))
            block = SimpleNamespace(type="text", text='{"aligned": 80, "coherent": 90}')
            return SimpleNamespace(stop_reason="end_turn", content=[block])

    client = SimpleNamespace(messages=_FakeMessages())
    results = bj.judge_completions_batch(
        {"p": {"q1": ["c1"], "q2": ["c2"]}},
        judge_system_prompt="RUBRIC " * 1024,  # >= 4*CACHE_MIN_TOKENS chars
        cache_dir=cache_dir,
        force_sync=True,
        sync_client=client,
        max_tokens=64,
    )
    assert client.messages.n == 2
    assert observed.get("cache_files_at_call2", 0) >= 1, (
        "item 1's verdict was not persisted before item 2's call — "
        "the incremental write-through is missing (a mid-wave kill would salvage nothing)"
    )
    assert "p" in results


# ── CONCERN overflow-staging-disconnected ───────────────────────────────────


def _fake_stage_hub_file_factory(calls, missing_on_canonical: str | None = None):
    from huggingface_hub.utils import EntryNotFoundError

    def fake_stage_hub_file(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        calls.append({"repo_id": repo_id, "path": path_in_repo, "repo_type": repo_type})
        import scripts.issue2617_svmp_reads as reads

        if (
            repo_id == reads.HF_DATA_REPO
            and missing_on_canonical
            and missing_on_canonical in path_in_repo
        ):
            raise EntryNotFoundError("absent")
        return Path(target)

    return fake_stage_hub_file


def test_stage_with_overflow_fallback_uses_model_repo(tmp_path, monkeypatch):
    import scripts.issue2617_svmp_reads as reads
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import DEFAULT_OVERFLOW_REPO

    calls: list[dict] = []
    monkeypatch.setattr(
        hub, "stage_hub_file", _fake_stage_hub_file_factory(calls, missing_on_canonical="gone")
    )
    tag = reads._stage_with_overflow("gone/file.json", tmp_path / "t1", "rev", set())
    assert tag == "overflow"
    assert calls[-1]["repo_id"] == DEFAULT_OVERFLOW_REPO
    assert calls[-1]["repo_type"] == "model", (
        "the overflow repo is a MODEL repo — a dataset-typed fetch can never resolve it"
    )


def test_stage_with_overflow_consults_reroute_pointers_first(tmp_path, monkeypatch):
    import scripts.issue2617_svmp_reads as reads
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import DEFAULT_OVERFLOW_REPO

    calls: list[dict] = []
    monkeypatch.setattr(hub, "stage_hub_file", _fake_stage_hub_file_factory(calls))
    full = f"{reads.HF_PREFIX}/raw_completions/judge/judge_scores.json"
    tag = reads._stage_with_overflow(
        "raw_completions/judge/judge_scores.json", tmp_path / "t2", "rev", {full}
    )
    assert tag == "overflow-pointer"
    assert len(calls) == 1, "a pointer hit must never read the (stale) canonical copy"
    assert calls[0]["repo_id"] == DEFAULT_OVERFLOW_REPO
    assert calls[0]["repo_type"] == "model"


def test_load_reroute_pointers_unions_upload_records(monkeypatch):
    import scripts.issue2617_svmp_reads as reads
    from explore_persona_space.orchestrate import hub

    rerouted_path = f"{reads.HF_PREFIX}/raw_completions/judge/judge_scores.json"

    def fake_stage(repo_id, path_in_repo, target, **kw):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(
            json.dumps(
                {
                    "status": "done",
                    "upload_judge": {"mode": "hf", "judge": {"rerouted_paths": [rerouted_path]}},
                    "upload_margin": {"mode": "hf", "margin": {"rerouted_paths": []}},
                }
            ),
            encoding="utf-8",
        )
        return Path(target)

    monkeypatch.setattr(hub, "stage_hub_file", fake_stage)
    assert reads._load_reroute_pointers("rev") == {rerouted_path}


def test_stage_ridge_payloads_returns_revision(tmp_path, monkeypatch):
    import scripts.issue2617_svmp_reads as reads
    from explore_persona_space.orchestrate import hub

    paths, rev = reads.stage_ridge_payloads_svmp(tmp_path, [19], tiny=True, d=4)
    assert rev is None and 19 in paths and set(paths[19]) == set(reads.ARMS)
    monkeypatch.setattr(hub, "stage_hub_file", lambda *a, **k: Path(a[2]))
    monkeypatch.setattr(reads, "_resolve_data_repo_revision", lambda: "deadbeefcafe")
    paths, rev = reads.stage_ridge_payloads_svmp(tmp_path, [19], tiny=False, d=4, revision=None)
    assert rev == "deadbeefcafe", "the internally-resolved ridge revision must be returned"
