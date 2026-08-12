"""Regression pins for the #2223 code-review round-1 fixes.

BLOCKER 1 — ``phase ridge``'s production-branch ``sentence_transformers`` import must be a
LOCKED dependency (it was absent from uv.lock -> guaranteed ModuleNotFoundError after the
pod's ``uv sync``), and the launcher must run ``upload`` BEFORE the optional ``ridge``
add-on so a ridge failure can never strand raw_completions/ (the #779 class).

BLOCKER 2 — ``phase_topics`` must be idempotent (skip-if-exists; paid Sonnet calls were
re-issued unconditionally on every launcher restart) and the generate regime fingerprint
must carry a topics-content hash (a resumed cell silently continuing under regenerated
topics is stimulus corruption).

ISSUE 3 — the fig5 judge must MERGE the api-refusal reissue into the full-set result
(the wholesale rebind destroyed ``rate``/``mean_scores`` whenever any refusal fired).

ISSUE 4 — the G2 stop gate must be CONSUMED: ``phase_gate`` exits ``GATE_STOP_RC`` on
``stops_phase_b`` so the launcher can skip the Phase B grid before it spends.

ISSUE 5 — the per-turn truncation retry must be UNCONDITIONAL (``threshold=0.0`` at the
``cap_hit_regen`` call site); the >2% batch fraction stays reporting-only.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "issue2223_drift.py"
LAUNCHER = REPO / "scripts" / "launch_issue_2223.sh"
UV_LOCK = REPO / "uv.lock"


def _import_driver():
    sys.path.insert(0, str(REPO))
    from scripts import issue2223_drift as D

    return D


# ── BLOCKER 1 ──────────────────────────────────────────────────────────────────────


def test_sentence_transformers_is_locked_dependency():
    """The ridge production import resolves after `uv sync` (exact dist-name parse —
    a substring check cannot tell `eai-sparsify==` from `sparsify==`, code-style.md)."""
    names = set(re.findall(r'^name = "([^"]+)"', UV_LOCK.read_text(), flags=re.M))
    assert "sentence-transformers" in names, (
        "sentence-transformers absent from uv.lock — phase ridge crashes on the pod "
        "(ModuleNotFoundError) and, ordered before upload, strands raw_completions/"
    )


def test_launcher_uploads_before_optional_ridge_on_both_legs():
    """`run_phase upload` precedes `run_phase ridge` in BOTH model-leg blocks."""
    text = LAUNCHER.read_text()
    # Split at the leg branch; both the 7b block and the 32b block must order
    # upload strictly before ridge.
    blocks = re.findall(r'if \[ "\$MODEL" = 7b \]; then(.*?)else(.*?)\nfi', text, flags=re.S)
    assert blocks, "launcher leg blocks not found (structure changed?)"
    for leg in blocks[0]:
        up = leg.find("run_phase upload")
        ridge = leg.find("run_phase ridge")
        assert up != -1 and ridge != -1, "upload/ridge phase missing from a leg"
        assert up < ridge, "upload must run BEFORE the optional ridge add-on (r1 BLOCKER 1)"


# ── BLOCKER 2 ──────────────────────────────────────────────────────────────────────


def test_phase_topics_skips_when_file_exists(tmp_path, monkeypatch):
    """An existing topics_personas.json short-circuits phase_topics BEFORE any paid
    API dispatch (idempotent; --force-topics is the explicit regeneration flag)."""
    D = _import_driver()
    existing = tmp_path / "topics_personas.json"
    existing.write_text(json.dumps({"domains": {}, "meta": {}}))

    def _boom(*a, **k):  # any dispatch call = the non-idempotent regression
        raise AssertionError("paid API dispatched despite existing topics_personas.json")

    monkeypatch.setattr(D, "_dispatch_sync", _boom)
    args = types.SimpleNamespace(out_root=str(tmp_path), smoke=True, force_topics=False)
    out = D.phase_topics(args)
    assert out == existing
    assert json.loads(existing.read_text()) == {"domains": {}, "meta": {}}  # untouched


def test_generate_regime_fingerprint_includes_topics_sha(tmp_path):
    """A stimulus change invalidates the resume: two fingerprints differing ONLY in
    topics content must FAIL check_regime loud (never silently continue)."""
    sys.path.insert(0, str(REPO))
    from scripts import issue2203_common as C

    base = dict(arm="A0", model="m", n_convs=4, shard_id=0, num_shards=1)
    r_old = C.regime_fingerprint(**base, topics_sha="aaaa")
    r_new = C.regime_fingerprint(**base, topics_sha="bbbb")
    with pytest.raises(ValueError, match="topics_sha"):
        C.check_regime(r_old, r_new, tmp_path / "x.regime.json")
    # same stimulus -> resume proceeds
    C.check_regime(r_old, C.regime_fingerprint(**base, topics_sha="aaaa"), tmp_path / "x")


def test_driver_threads_topics_sha_into_regime():
    """Source pin: phase_generate passes topics_sha= into C.regime_fingerprint."""
    src = DRIVER.read_text()
    assert "topics_sha=topics_sha" in src, "topics content hash dropped from the regime key"


# ── ISSUE 3 (fig5 reissue merge) ─────────────────────────────────────────────────────


def test_merge_reissue_scores_preserves_rate_and_merges():
    D = _import_driver()
    res = {
        "mean_scores": {"j0": 80.0, "j1": None, "j2": 10.0},
        "n_items": 3,
        "n_scored_items": 2,
        "rate": 0.5,
        "per_item_api_refusals": {"j1": 5},
    }
    reissue = {"n_censored": 1, "n_rescued": 1, "rescued_scores": {"j1": 90.0}}
    out = D._merge_reissue_scores(res, reissue)
    # rescued score fills the censored item; untouched items keep their scores.
    assert out["mean_scores"] == {"j0": 80.0, "j1": 90.0, "j2": 10.0}
    assert out["n_scored_items"] == 3
    assert out["rate"] == pytest.approx(2 / 3)  # j0, j1 >= 50
    assert out["reissue"] == reissue  # telemetry preserved, never a wholesale rebind
    assert res["rate"] == 0.5  # input not mutated


def test_merge_reissue_scores_none_rescue_does_not_clobber():
    D = _import_driver()
    res = {"mean_scores": {"j0": 80.0, "j1": 60.0}, "rate": 1.0}
    reissue = {"n_censored": 1, "n_rescued": 0, "rescued_scores": {"j1": None}}
    out = D._merge_reissue_scores(res, reissue)
    assert out["mean_scores"]["j1"] == 60.0  # a failed rescue never erases a real score
    assert out["rate"] == 1.0


# ── ISSUE 4 (G2 stop gate consumed) ─────────────────────────────────────────────────


def _write_verdict(tmp_path, stops: bool):
    (tmp_path / "phaseA_verdict.json").write_text(
        json.dumps(
            {
                "anchor_cell": "A0__7b",
                "verdict": {"disposition": "x", "stops_phase_b": stops},
            }
        )
    )


def test_gate_phase_stops_on_failed_to_reproduce(tmp_path):
    D = _import_driver()
    _write_verdict(tmp_path, stops=True)
    args = types.SimpleNamespace(out_root=str(tmp_path), smoke=True)
    with pytest.raises(SystemExit) as ei:
        D.phase_gate(args)
    assert ei.value.code == D.GATE_STOP_RC  # designed halt, launcher-mapped


def test_gate_phase_passes_on_reproduced(tmp_path):
    D = _import_driver()
    _write_verdict(tmp_path, stops=False)
    args = types.SimpleNamespace(out_root=str(tmp_path), smoke=True)
    assert D.phase_gate(args).exists()


def test_launcher_gates_phase_b_after_aggregate():
    """The launcher consumes the gate BETWEEN aggregate and the Phase B arm loop."""
    text = LAUNCHER.read_text()
    assert "check_g2_gate" in text, "G2 gate never consumed by the launcher (r1 ISSUE 4)"
    seven_b = text.split('if [ "$MODEL" = 7b ]; then')[-1]
    agg = seven_b.find("run_phase aggregate")
    gate = seven_b.find("check_g2_gate")
    phase_b = seven_b.find("ARMS_7B_PHASEB[@]")
    assert -1 not in (agg, gate, phase_b)
    assert agg < gate < phase_b, "Phase B must not generate before the verdict gate"


# ── ISSUE 5 (unconditional per-turn truncation retry) ────────────────────────────────


def test_generate_callsite_regens_truncated_turns_unconditionally():
    """§4.2: the per-turn cap_hit_regen retry passes threshold=0.0 (any hitting row),
    while CAP_HIT_THRESHOLD stays as the recorded cell-level reporting statistic."""
    src = DRIVER.read_text()
    call = re.search(r"R\.cap_hit_regen\((?:[^)]|\n)*?\)", src)
    assert call, "cap_hit_regen call site not found"
    assert "threshold=0.0" in call.group(0), (
        "per-turn truncation retry must be unconditional (threshold=0.0) — a single "
        "truncated turn poisons every later turn of that conversation (r1 ISSUE 5)"
    )
    assert "cap_hit_reporting_threshold" in src  # >2% tier kept as reporting
