"""Tests for ``vm_disk_guard.py`` tier (e) — the HOME HF hub cache
(``~/.cache/huggingface/hub``; task #1376, reconciled with #1377's
independently-landed tier into ONE ``clean_home_hf_stale_revisions``).

Pins the plan's acceptance criteria under the RECONCILED (union) semantics:
attribution ALWAYS (T1), the arm-2 reap predicate — non-newest AND no ref
AT ALL (any ref protects — widened from #1376's main-ref-only by the
reconciliation with #1377) AND stale ``last_modified`` AND no fresh
EXCLUSIVE-blob atime (T2-T4), the arm-1 whole-stale-repo reap where the
ref/freshness/newest protections deliberately do NOT bind (T5), the
keep-newest-per-repo pin (T6 — #1377's rule, subsuming the plan's
never-empty-a-fresh-repo constraint), escalation dedup/growth/ack
mechanics (T7), report-only persisting nothing (T8), fail-toward-KEEP
degradation (T9), the tier-(d) double-cover guard (T10), pod refusal (T11),
the ``run_guard`` production opt-in (T12), the ``--json`` field (T13), and
scan-warning reporting (T14). #1377's own arm-2 pins live in
``tests/test_vm_disk_guard_home_hf.py``.

HERMETIC BY CONSTRUCTION: ``_scan_hf_cache`` is monkeypatched to
SimpleNamespace fakes (the tier-(d) fixture pattern from
``test_janitor_noncanonical_caches.py``), ``repo_root`` points at
``tmp_path``, and the sidecar/Telegram/state writers are recorders — the
REAL home cache is never scanned or touched from pytest.

Loaded via importlib like ``tests/test_vm_disk_guard.py`` (ced first —
vm_disk_guard imports it by module name at load time).
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec (dataclass + future annotations)
    spec.loader.exec_module(mod)
    return mod


ced = _load("clean_experiment_downloads")
vdg = _load("vm_disk_guard")


# ─── fixtures / fakes ────────────────────────────────────────────────────────

NOW = 1_700_000_000.0
DAY = 86400.0
HOUR = 3600.0


def _mk_file(blob_path: str, atime: float):
    return SimpleNamespace(blob_path=blob_path, blob_last_accessed=atime)


def _mk_rev(commit: str, *, refs=(), last_modified: float, files=(), size: int = 1_000):
    return SimpleNamespace(
        commit_hash=commit,
        refs=frozenset(refs),
        last_modified=last_modified,
        files=list(files),
        size_on_disk=size,
    )


def _mk_repo(repo_id: str, *, repo_type="model", last_accessed: float, revisions, size=None):
    revisions = list(revisions)
    if size is None:
        size = sum(r.size_on_disk for r in revisions)
    return SimpleNamespace(
        repo_id=repo_id,
        repo_type=repo_type,
        size_on_disk=size,
        last_accessed=last_accessed,
        revisions=revisions,
    )


def _mk_info(repos, executed: list, *, expected_freed: int = 12_345, warnings=()):
    def delete_revisions(*hashes):
        return SimpleNamespace(
            expected_freed_size=expected_freed,
            requested=hashes,
            execute=lambda: executed.append(tuple(hashes)),
        )

    return SimpleNamespace(
        repos=list(repos), warnings=list(warnings), delete_revisions=delete_revisions
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Hermetic tier-(e) rig: repo_root -> tmp_path, pod guard off, recorders
    on the sidecar + Telegram writers, and a fixture hub dir."""
    monkeypatch.setattr(vdg, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(ced, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(vdg, "_running_pod_side", lambda: False)
    events: list[tuple[dict, bool]] = []
    pushes: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        vdg, "append_disk_guard_event", lambda ev, *, apply=True: events.append((ev, apply))
    )
    monkeypatch.setattr(
        vdg, "_telegram_push", lambda msg, apply: pushes.append((msg, apply)) or True
    )
    cache_root = tmp_path / "homecache"
    (cache_root / "hub").mkdir(parents=True)
    return SimpleNamespace(tmp_path=tmp_path, cache_root=cache_root, events=events, pushes=pushes)


def _run_tier(env, monkeypatch, info, *, apply, state=None, **kw):
    monkeypatch.setattr(vdg, "_scan_hf_cache", lambda hub: info)
    return vdg.clean_home_hf_stale_revisions(
        apply, cache_root=env.cache_root, now=NOW, state={} if state is None else state, **kw
    )


# ─── T1: attribution always names the top repos ─────────────────────────────


def test_home_tier_attribution_always_names_top_repos(env, monkeypatch):
    executed: list = []
    big = _mk_repo(
        "superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        last_accessed=NOW - 1 * HOUR,
        revisions=[_mk_rev("aaa111", refs={"main"}, last_modified=NOW - 1 * DAY, size=30_000)],
    )
    small = _mk_repo(
        "Qwen/Qwen2.5-7B-Instruct",
        last_accessed=NOW - 2 * HOUR,
        revisions=[_mk_rev("bbb222", refs={"main"}, last_modified=NOW - 2 * DAY, size=10_000)],
    )
    res = _run_tier(env, monkeypatch, _mk_info([small, big], executed), apply=False)
    assert res.skipped is False
    assert executed == []  # report-only executes NOTHING
    assert [r["repo"] for r in res.hf_repo_attributions] == [
        "superkaiba1/explore-persona-space-data",
        "Qwen/Qwen2.5-7B-Instruct",
    ]  # ALL repos, size-desc
    row = res.hf_repo_attributions[0]
    assert row["repo_type"] == "dataset"
    assert row["bytes"] == 30_000
    assert row["revisions"] == 1
    assert row["reap_candidate_bytes"] == 0
    assert row["over_escalate_threshold"] is False
    assert any("dataset/superkaiba1/explore-persona-space-data" in d for d in res.detail)
    assert any("no unref'd revision" in d for d in res.detail)


# ─── T2: arm 2 reaps exactly the unref'd non-newest stale cold revisions ─────


def test_home_tier_reaps_unreferenced_stale_revisions(env, monkeypatch):
    executed: list = []
    repo = _mk_repo(
        "org/data",
        repo_type="dataset",
        last_accessed=NOW - 1 * HOUR,  # repo FRESH -> arm 2 only
        revisions=[
            # main-ref'd, stale: kept (ref protection).
            _mk_rev(
                "mainref1",
                refs={"main"},
                last_modified=NOW - 30 * DAY,
                files=[_mk_file("blobM", NOW - 30 * DAY)],
            ),
            # ref-less but recently written: kept (also the repo's NEWEST).
            _mk_rev(
                "fresh111", last_modified=NOW - 1 * DAY, files=[_mk_file("blobF", NOW - 1 * DAY)]
            ),
            # pinned-read ref (non-main), stale, cold exclusive blob: KEPT —
            # the #1377 reconciliation widened ref protection from
            # main-only to ANY ref (union of both tasks' KEEP sets).
            _mk_rev(
                "pinned77",
                refs={"77d04e45"},
                last_modified=NOW - 20 * DAY,
                files=[_mk_file("blobP", NOW - 20 * DAY)],
            ),
            # ref-less, non-newest, stale, cold exclusive blob: REAPED —
            # the one revision every KEEP rule passes over.
            _mk_rev(
                "cold2222",
                last_modified=NOW - 20 * DAY,
                files=[_mk_file("blobC", NOW - 20 * DAY)],
            ),
            # ref-less, stale, but exclusive blob recently READ: kept.
            _mk_rev(
                "read3333", last_modified=NOW - 20 * DAY, files=[_mk_file("blobR", NOW - 1 * HOUR)]
            ),
        ],
    )
    res = _run_tier(env, monkeypatch, _mk_info([repo], executed, expected_freed=777), apply=True)
    assert executed == [("cold2222",)]
    assert res.bytes_freed == 777
    reaped = [ev for ev, _ in env.events if ev["kind"] == "home-hf-revisions-trimmed"]
    assert len(reaped) == 1
    assert reaped[0]["arms"] == {"whole_repo": 0, "revision_level": 1}
    # Attribution names the candidate bytes for the repo.
    assert res.hf_repo_attributions[0]["reap_candidate_bytes"] == 1_000


# ─── T3: ANY-ref'd + fresh + newest revisions never enter the arm-2 set ──────


def test_home_tier_keeps_any_ref_fresh_and_newest_revisions(env, monkeypatch):
    """Union KEEP protections (#1376 reconciled with #1377): a ``main`` ref,
    ANY other ref (a pinned-read truncated-hash ref), a fresh
    ``last_modified``, and being the repo's newest each independently
    protect a revision from arm 2."""
    executed: list = []
    repo = _mk_repo(
        "org/data",
        last_accessed=NOW - 1 * HOUR,
        revisions=[
            _mk_rev(
                "mainref1",
                refs={"main"},
                last_modified=NOW - 40 * DAY,
                files=[_mk_file("blobM", NOW - 40 * DAY)],
            ),
            # Non-main pinned-read ref, stale + cold: kept SOLELY by the
            # widened any-ref protection (#1377's incumbent semantics).
            _mk_rev(
                "pinned77",
                refs={"77d04e45"},
                last_modified=NOW - 40 * DAY,
                files=[_mk_file("blobP", NOW - 40 * DAY)],
            ),
            # Ref-less + fresh: kept (also the repo's newest).
            _mk_rev(
                "fresh111", last_modified=NOW - 1 * DAY, files=[_mk_file("blobF", NOW - 1 * DAY)]
            ),
        ],
    )
    res = _run_tier(env, monkeypatch, _mk_info([repo], executed), apply=True)
    assert executed == []
    assert res.bytes_freed == 0
    assert any("no unref'd revision" in d for d in res.detail)


# ─── T4: the exclusive-blob atime guard ──────────────────────────────────────


def test_home_tier_exclusive_blob_atime_guard(env, monkeypatch):
    executed: list = []
    repo = _mk_repo(
        "org/data",
        last_accessed=NOW - 1 * HOUR,
        revisions=[
            # Kept (main + fresh) — its blobA is a KEPT blob.
            _mk_rev(
                "keepmain",
                refs={"main"},
                last_modified=NOW - 1 * DAY,
                files=[_mk_file("blobA", NOW - 1 * HOUR)],
            ),
            # (a) fresh EXCLUSIVE atime -> kept (deletion would destroy
            # recently-read data).
            _mk_rev(
                "candaaaa", last_modified=NOW - 20 * DAY, files=[_mk_file("blobB", NOW - 1 * HOUR)]
            ),
            # (b) fresh atime ONLY on a blob SHARED with the kept revision ->
            # reaped (the shared blob survives via delete_revisions
            # refcounting; no exclusive data is destroyed).
            _mk_rev(
                "candbbbb", last_modified=NOW - 20 * DAY, files=[_mk_file("blobA", NOW - 1 * HOUR)]
            ),
            # (c) cold exclusive atime -> reaped.
            _mk_rev(
                "candcccc", last_modified=NOW - 20 * DAY, files=[_mk_file("blobC", NOW - 20 * DAY)]
            ),
        ],
    )
    _run_tier(env, monkeypatch, _mk_info([repo], executed), apply=True)
    assert len(executed) == 1
    assert set(executed[0]) == {"candbbbb", "candcccc"}


# ─── T5: arm 1 — whole stale repos are reaped, main-ref'd revisions INCLUDED ─


def test_home_tier_whole_repo_arm_covers_stale_models(env, monkeypatch):
    executed: list = []
    stale_model = _mk_repo(
        "org/stale-model",
        last_accessed=NOW - 10 * DAY,  # repo-level age gate: 10d > 7d window
        revisions=[
            # The revision carries a MAIN ref and IS reaped: the arm-2
            # protections do not bind on arm 1 (this is what covers models).
            _mk_rev(
                "mainhash",
                refs={"main"},
                last_modified=NOW - 1 * DAY,
                files=[_mk_file("blobM", NOW - 1 * HOUR)],
            ),
        ],
    )
    res = _run_tier(env, monkeypatch, _mk_info([stale_model], executed), apply=True)
    assert executed == [("mainhash",)]
    assert any("arm 1 wholesale" in d and "ref'd + newest included" in d for d in res.detail)
    reaped = [ev for ev, _ in env.events if ev["kind"] == "home-hf-revisions-trimmed"]
    assert reaped[0]["arms"] == {"whole_repo": 1, "revision_level": 0}

    # The same repo with FRESH last_accessed is untouched.
    executed2: list = []
    fresh_model = _mk_repo(
        "org/stale-model",
        last_accessed=NOW - 1 * HOUR,
        revisions=[
            _mk_rev(
                "mainhash",
                refs={"main"},
                last_modified=NOW - 1 * DAY,
                files=[_mk_file("blobM", NOW - 1 * HOUR)],
            ),
        ],
    )
    res2 = _run_tier(env, monkeypatch, _mk_info([fresh_model], executed2), apply=True)
    assert executed2 == []
    assert any("no unref'd revision" in d for d in res2.detail)


# ─── T6: the newest revision per repo is ALWAYS kept (keep-newest pin) ───────


def test_home_tier_keeps_newest_revision_per_repo(env, monkeypatch):
    """Keep-newest (#1377's rule, adopted at reconciliation): the repo's
    newest revision never enters the arm-2 set even when it is ref-less,
    stale, and cold — which also guarantees a recently-accessed repo is
    never wholly emptied (subsumes the plan's never-empty clamp)."""
    executed: list = []
    repo = _mk_repo(
        "org/data",
        last_accessed=NOW - 1 * HOUR,
        revisions=[
            _mk_rev(
                "older111", last_modified=NOW - 20 * DAY, files=[_mk_file("blobA", NOW - 20 * DAY)]
            ),
            _mk_rev(
                "newer222", last_modified=NOW - 10 * DAY, files=[_mk_file("blobB", NOW - 10 * DAY)]
            ),
        ],
    )
    _run_tier(env, monkeypatch, _mk_info([repo], executed), apply=True)
    # Both pass every other reap gate (ref-less, stale, cold); the NEWEST
    # (newer222) is kept solely by keep-newest.
    assert executed == [("older111",)]
    assert all("newer222" not in req for req in executed)


# ─── T7: escalation dedup / growth re-alert / ack sentinel ───────────────────


def _big_repo(size: int):
    return _mk_repo(
        "superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        last_accessed=NOW - 1 * HOUR,
        revisions=[_mk_rev("aaa111", refs={"main"}, last_modified=NOW - 1 * DAY, size=size)],
        size=size,
    )


def test_home_tier_repo_escalation_dedup_and_ack(env, monkeypatch):
    state: dict = {}
    repo_key = "dataset/superkaiba1/explore-persona-space-data"

    def esc_events():
        return [ev for ev, _ in env.events if ev["kind"] == "home-hf-cache-repo-escalation"]

    # 1st run: one sidecar row (with breakdown + reap_cmd) + one push.
    res = _run_tier(
        env, monkeypatch, _mk_info([_big_repo(60 * 10**9)], []), apply=True, state=state
    )
    assert len(esc_events()) == 1
    row = esc_events()[0]
    assert row["repo"] == repo_key
    assert row["band"] == 50.0
    assert row["reap_cmd"] == "uv run python scripts/vm_disk_guard.py --apply"
    assert row["revision_breakdown"][0]["commit"] == "aaa111"[:8]
    assert row["revision_breakdown"][0]["refs"] == ["main"]
    assert len(env.pushes) == 1 and repo_key in env.pushes[0][0]
    assert state[f"hf:{repo_key}:50"] == 60 * 10**9
    assert res.hf_repo_attributions[0]["over_escalate_threshold"] is True

    # 2nd run, same band + no growth: NO re-fire (detail line still present).
    res2 = _run_tier(
        env, monkeypatch, _mk_info([_big_repo(60 * 10**9)], []), apply=True, state=state
    )
    assert len(esc_events()) == 1
    assert len(env.pushes) == 1
    assert any("ESCALATION" in d for d in res2.detail)
    assert len(res2.hf_repo_attributions) == 1  # attribution is dedup-independent

    # >25% growth within the band: re-fires.
    _run_tier(env, monkeypatch, _mk_info([_big_repo(80 * 10**9)], []), apply=True, state=state)
    assert len(esc_events()) == 2
    assert esc_events()[1]["growth_pct"] > 25.0

    # Ack sentinel for this (repo, band): suppressed entirely.
    ack = vdg._hf_ack_sentinel_path(repo_key, 50.0)
    ack.parent.mkdir(parents=True, exist_ok=True)
    ack.touch()
    res4 = _run_tier(
        env, monkeypatch, _mk_info([_big_repo(99 * 10**9)], []), apply=True, state=state
    )
    assert len(esc_events()) == 2
    assert len(res4.hf_repo_attributions) == 1  # still attributed


# ─── T8: report-only persists NOTHING ────────────────────────────────────────


def test_home_tier_dry_run_persists_nothing(env, monkeypatch, tmp_path):
    executed: list = []
    state: dict = {}
    # Over-threshold repo WITH a reap candidate, so every arm is exercised.
    repo = _mk_repo(
        "superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        last_accessed=NOW - 1 * HOUR,
        revisions=[
            _mk_rev("aaa111", refs={"main"}, last_modified=NOW - 1 * DAY, size=50 * 10**9),
            _mk_rev(
                "cold2222",
                last_modified=NOW - 20 * DAY,
                files=[_mk_file("blobC", NOW - 20 * DAY)],
                size=10 * 10**9,
            ),
        ],
    )
    res = _run_tier(
        env, monkeypatch, _mk_info([repo], executed, expected_freed=4242), apply=False, state=state
    )
    assert executed == []  # zero execute()
    assert res.bytes_freed == 4242  # expected_freed_size booked
    # The escalation row IS recorded — with apply=False on every call (the
    # shared writers gate persistence on apply; report-only prints only).
    assert len(env.events) >= 1
    assert all(applied is False for _, applied in env.events)
    assert all(applied is False for _, applied in env.pushes)
    assert state == {}  # state writes are apply-gated
    assert not (tmp_path / ".claude" / "cache" / "disk-guard-active-state.json").exists()


# ─── T9: fail toward KEEP on scan / execute errors ───────────────────────────


def test_home_tier_fail_toward_keep_on_scan_error(env, monkeypatch):
    def _boom(hub):
        raise ImportError("huggingface_hub gone")

    monkeypatch.setattr(vdg, "_scan_hf_cache", _boom)
    res = vdg.clean_home_hf_stale_revisions(True, cache_root=env.cache_root, now=NOW, state={})
    assert res.skipped and "ImportError" in res.skip_reason

    # An execute()-time failure degrades the same way, deleting nothing.
    stale = _mk_repo(
        "org/stale-model",
        last_accessed=NOW - 10 * DAY,
        revisions=[_mk_rev("aaa111", refs={"main"}, last_modified=NOW - 10 * DAY)],
    )

    def _bad_info(hub):
        return SimpleNamespace(
            repos=[stale],
            warnings=[],
            delete_revisions=lambda *h: SimpleNamespace(
                expected_freed_size=1,
                execute=lambda: (_ for _ in ()).throw(OSError("corrupt cache")),
            ),
        )

    monkeypatch.setattr(vdg, "_scan_hf_cache", _bad_info)
    res2 = vdg.clean_home_hf_stale_revisions(True, cache_root=env.cache_root, now=NOW, state={})
    assert res2.skipped and "OSError" in res2.skip_reason


# ─── T10: double-cover guard vs tier (d) ─────────────────────────────────────


def test_home_tier_skips_when_root_equals_workspace_root(env, monkeypatch):
    monkeypatch.setattr(vdg, "workspace_hf_cache_root", lambda: env.cache_root)
    monkeypatch.setattr(
        vdg, "_scan_hf_cache", lambda hub: (_ for _ in ()).throw(AssertionError("must not scan"))
    )
    res = vdg.clean_home_hf_stale_revisions(True, cache_root=env.cache_root, now=NOW, state={})
    assert res.skipped and "no double-reap" in res.skip_reason


# ─── T11: pod-side refusal ───────────────────────────────────────────────────


def test_home_tier_pod_side_refusal(env, monkeypatch):
    monkeypatch.setattr(vdg, "_running_pod_side", lambda: True)
    res = vdg.clean_home_hf_stale_revisions(True, cache_root=env.cache_root, now=NOW, state={})
    assert res.skipped and "pod-side" in res.skip_reason


# ─── T12: run_guard wiring — tier (e) rides the production opt-in ────────────


def _patch_disk(monkeypatch, before_pct: float, after_pct: float):
    calls = {"n": 0}

    def fake_used(path="/"):
        calls["n"] += 1
        return before_pct if calls["n"] == 1 else after_pct

    monkeypatch.setattr(vdg, "disk_used_pct", fake_used)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": 50.0)


def test_run_guard_home_tier_rides_production_opt_in(env, monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    tmp_root = tmp_path / "faketmp"
    tmp_root.mkdir()
    monkeypatch.setattr(vdg, "clean_uv_cache", lambda apply: vdg.TierResult(name="uv-cache"))
    monkeypatch.setattr(vdg, "clean_stale_logs", lambda *a, **k: vdg.TierResult(name="stale-logs"))
    monkeypatch.setattr(
        vdg,
        "clean_terminal_download_caches",
        lambda apply, **k: vdg.TierResult(name="terminal-download-caches"),
    )
    monkeypatch.setattr(
        vdg,
        "clean_vm_workspace_hf_cache",
        lambda apply, **k: vdg.TierResult(name="workspace-hf-cache"),
    )
    # The REAL home cache must never be touched from pytest — stub the tier
    # and assert only the run_guard WIRING here.
    monkeypatch.setattr(
        vdg,
        "clean_home_hf_stale_revisions",
        lambda apply, **k: vdg.TierResult(name="home-hf-revisions"),
    )

    _patch_disk(monkeypatch, before_pct=90.0, after_pct=40.0)
    res = vdg.run_guard(apply=False, threshold=85.0, data_root=data_root, tmp_root=tmp_root)
    assert "home-hf-revisions" in {t.name for t in res.tiers}

    _patch_disk(monkeypatch, before_pct=90.0, after_pct=40.0)
    res2 = vdg.run_guard(apply=False, threshold=85.0, data_root=data_root, tmp_root=None)
    assert "home-hf-revisions" not in {t.name for t in res2.tiers}

    _patch_disk(monkeypatch, before_pct=90.0, after_pct=40.0)
    res3 = vdg.run_guard(
        apply=False, threshold=85.0, data_root=data_root, reclaim_tiers=False, tmp_root=tmp_root
    )
    assert "home-hf-revisions" not in {t.name for t in res3.tiers}


# ─── T13: --json serializes hf_repo_attributions ─────────────────────────────


def test_result_json_serializes_hf_repo_attributions():
    tier = vdg.TierResult(name="home-hf-revisions")
    tier.hf_repo_attributions.append(
        {
            "repo": "superkaiba1/explore-persona-space-data",
            "repo_type": "dataset",
            "bytes": 1,
            "revisions": 1,
            "last_accessed_age_days": 0.1,
            "reap_candidate_bytes": 0,
            "over_escalate_threshold": False,
        }
    )
    guard = vdg.GuardResult(
        used_pct_before=90.0,
        used_pct_after=90.0,
        free_gb_before=10.0,
        free_gb_after=10.0,
        threshold_pct=85.0,
        triggered=True,
        apply=False,
        tiers=[tier],
    )
    payload = vdg._result_json(guard)
    assert payload["tiers"][0]["hf_repo_attributions"] == tier.hf_repo_attributions


# ─── T14: scan warnings are reported, warned repos untouched ─────────────────


def test_home_tier_scan_warnings_reported(env, monkeypatch):
    executed: list = []
    fresh = _mk_repo(
        "org/fine",
        last_accessed=NOW - 1 * HOUR,
        revisions=[_mk_rev("aaa111", refs={"main"}, last_modified=NOW - 1 * DAY)],
    )
    info = _mk_info([fresh], executed, warnings=[Warning("corrupt repo"), Warning("bad rev")])
    res = _run_tier(env, monkeypatch, info, apply=True)
    assert res.skipped is False
    assert any("scan warnings: 2 (repos kept)" in d for d in res.detail)
    assert executed == []  # a warned (unscannable) repo is absent from repos -> kept


# ─── selector-level: degenerate repo kept without degrading the tier ─────────


def test_home_tier_degenerate_repo_kept_others_selected(env, monkeypatch):
    """A repo with None timestamps is skipped (kept) with a kept-reason count
    while every other repo keeps its selection (#1376 review concern)."""
    executed: list = []
    degenerate = _mk_repo(
        "org/degenerate", last_accessed=None, revisions=[_mk_rev("ddd444", last_modified=None)]
    )
    stale = _mk_repo(
        "org/stale-model",
        last_accessed=NOW - 10 * DAY,
        revisions=[_mk_rev("aaa111", refs={"main"}, last_modified=NOW - 10 * DAY)],
    )
    res = _run_tier(env, monkeypatch, _mk_info([degenerate, stale], executed), apply=True)
    assert res.skipped is False
    assert executed == [("aaa111",)]  # the healthy stale repo is still reaped
    assert any("degenerate-repo-kept: 1 repo(s)" in d for d in res.detail)
