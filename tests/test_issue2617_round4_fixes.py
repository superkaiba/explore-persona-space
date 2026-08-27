"""Round-4 fix tests for issue #2617 (svmp run driver).

Fix A (BLOCKER smoke-arch-margin-row-false-attestation): the canned tiny
margin pools are now routed through the SAME ``_assert_pool_provenance`` /
``_pool_floor_check`` gates the production branch runs — a delegating spy
pins that both gates EXECUTE (real bodies, not stubs) on a tiny margin pool
build, and a doctored canned entry proves the provenance gate is live.

Fix B (CONCERN margin-aggregate-identity-unvalidated): ``_margin_complete``
now parses margins.json's OWN ``regime_fp`` / ``pools_sha`` identity fields
and refuses completion on any missing key or mismatch against
``_margin_fp(cfg)`` / the recomputed on-disk pool-content sha / the
sentinel's recorded sha. Doctor tests flip EACH identity field
independently and assert the predicate flips False (and recovers True on
restore).
"""

from __future__ import annotations

import hashlib
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


# ── Fix A: tiny margin pools execute the production gates ───────────────────


def test_tiny_margin_pools_execute_provenance_and_floor_gates(monkeypatch):
    """Delegating spies (real bodies still run) pin that BOTH gates execute
    on a tiny margin pool build: provenance once per pool, floor check once."""
    calls = {"prov": 0, "floor": 0}
    real_prov = run._assert_pool_provenance
    real_floor = run._pool_floor_check

    def spy_prov(*a, **k):
        calls["prov"] += 1
        return real_prov(*a, **k)

    def spy_floor(*a, **k):
        calls["floor"] += 1
        return real_floor(*a, **k)

    monkeypatch.setattr(run, "_assert_pool_provenance", spy_prov)
    monkeypatch.setattr(run, "_pool_floor_check", spy_floor)
    refusal, helpful, meta = run._build_tiny_margin_pools()
    assert calls["prov"] == 2, calls  # once for refusal, once for helpful
    assert calls["floor"] == 1, calls
    assert refusal and helpful
    assert meta["canned"] is True
    # Canned pools are short by construction (< MARGIN_POOL_SIZE) — the floor
    # gate ran and recorded the disclosed tiny waiver.
    assert len(refusal) < run.MARGIN_POOL_SIZE
    assert meta["short_pool_waiver"] is True
    assert meta["n_refusal"] == len(run.CANNED_TINY_REFUSAL)
    assert meta["n_helpful"] == len(run.CANNED_TINY_HELPFUL)


def test_tiny_margin_provenance_gate_is_live(monkeypatch):
    """The provenance gate genuinely evaluates the canned entries: a doctored
    whitespace-only canned opener trips the empty-opener assert."""
    monkeypatch.setattr(run, "CANNED_TINY_REFUSAL", ("   ",))
    with pytest.raises(AssertionError, match="empty pool opener"):
        run._build_tiny_margin_pools()


def test_tiny_pool_entries_have_distinct_provenance_keys():
    """Per-entry provenance requires distinct (source_context, source_draw)
    keys — a single shared key cannot round-trip distinct openers."""
    refusal, helpful, _ = run._build_tiny_margin_pools()
    keys = [(e["source_context"], e["source_draw"]) for e in refusal + helpful]
    assert len(keys) == len(set(keys)), keys


# ── Fix B: _margin_complete validates margins.json identity fields ──────────


def _valid_margin_state(cfg) -> tuple[dict, dict, dict]:
    """Write a fully-valid pools.json + margins.json + sentinel triple for
    cfg; returns (pools_obj, margins_obj, sentinel_obj)."""
    pools_obj = {
        "refusal": [{"answer": "I cannot"}],
        "helpful": [{"answer": "Sure,"}],
        "meta": {"pool_size": 1},
    }
    mfp = run._margin_fp(cfg)
    pools_sha = run._pools_content_sha(pools_obj)
    margins_obj = {"regime_fp": mfp, "pools_sha": pools_sha, "per_context": {}}
    sentinel_obj = {"regime_fp": mfp, "pools_sha": pools_sha}
    _write_json(cfg.out_root / "margin" / "pools.json", pools_obj)
    _write_json(cfg.out_root / "margin" / "margins.json", margins_obj)
    _write_json(cfg.out_root / "svmp_margin_done.json", sentinel_obj)
    return pools_obj, margins_obj, sentinel_obj


def test_margin_complete_validates_margins_json_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path)
    _pools_obj, margins_obj, sentinel_obj = _valid_margin_state(cfg)
    assert run._margin_complete(cfg), "valid triple must be accepted"

    # (a) doctored margins.json regime_fp -> refused.
    _write_json(cfg.out_root / "margin" / "margins.json", dict(margins_obj, regime_fp="doctored"))
    assert not run._margin_complete(cfg)
    _write_json(cfg.out_root / "margin" / "margins.json", margins_obj)
    assert run._margin_complete(cfg)

    # (b) doctored margins.json pools_sha -> refused.
    _write_json(cfg.out_root / "margin" / "margins.json", dict(margins_obj, pools_sha="doctored"))
    assert not run._margin_complete(cfg)
    _write_json(cfg.out_root / "margin" / "margins.json", margins_obj)
    assert run._margin_complete(cfg)

    # (c) doctored sentinel pools_sha -> refused (fp untouched, so the
    # sentinel regime gate alone cannot catch it).
    _write_json(cfg.out_root / "svmp_margin_done.json", dict(sentinel_obj, pools_sha="doctored"))
    assert not run._margin_complete(cfg)
    _write_json(cfg.out_root / "svmp_margin_done.json", sentinel_obj)
    assert run._margin_complete(cfg)


def test_margin_complete_refuses_identity_free_margins_json(tmp_path, monkeypatch):
    """A margins.json MISSING either identity key (the pre-r4 fixture shape)
    is refused — presence-only acceptance was the concern."""
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path)
    _pools_obj, margins_obj, _sentinel_obj = _valid_margin_state(cfg)
    assert run._margin_complete(cfg)

    no_fp = {k: v for k, v in margins_obj.items() if k != "regime_fp"}
    _write_json(cfg.out_root / "margin" / "margins.json", no_fp)
    assert not run._margin_complete(cfg)

    no_sha = {k: v for k, v in margins_obj.items() if k != "pools_sha"}
    _write_json(cfg.out_root / "margin" / "margins.json", no_sha)
    assert not run._margin_complete(cfg)

    _write_json(cfg.out_root / "margin" / "margins.json", {"per_context": {}})
    assert not run._margin_complete(cfg)
