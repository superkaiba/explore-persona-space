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
    args = types.SimpleNamespace(
        out_root=str(tmp_path), smoke=True, force_topics=False, generate_topics=False
    )
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
    """The launcher consumes the gate BETWEEN aggregate and the Phase B grid — BOTH legs
    (r2 NIT: the r2 version pinned only the 7b block; the 32B leg's A1 is a Phase B grid
    member and must equally wait on the verdict)."""
    text = LAUNCHER.read_text()
    assert "check_g2_gate" in text, "G2 gate never consumed by the launcher (r1 ISSUE 4)"
    legs = re.search(r'if \[ "\$MODEL" = 7b \]; then(.*?)\nelse(.*?)\nfi', text, flags=re.S)
    assert legs, "launcher leg blocks not found"
    seven_b, thirty_two_b = legs.group(1), legs.group(2)
    # 7B leg: aggregate -> gate -> Phase B arm loop.
    agg = seven_b.find("run_phase aggregate")
    gate = seven_b.find("check_g2_gate")
    phase_b = seven_b.find("ARMS_7B_PHASEB[@]")
    assert -1 not in (agg, gate, phase_b)
    assert agg < gate < phase_b, "7B: Phase B must not generate before the verdict gate"
    # 32B leg: aggregate -> gate -> A1 (the leg's Phase B grid member).
    agg32 = thirty_two_b.find("run_phase aggregate")
    gate32 = thirty_two_b.find("check_g2_gate")
    a1 = thirty_two_b.find("drift_cell A1")
    assert -1 not in (agg32, gate32, a1), "32B leg blocks missing aggregate/gate/A1"
    assert agg32 < gate32 < a1, "32B: A1 must not generate before the verdict gate"


# ── ISSUE 5 (unconditional per-turn truncation retry) ────────────────────────────────


def test_generate_callsite_regens_truncated_turns_unconditionally():
    """§4.2: EVERY per-turn cap_hit_regen retry passes threshold=0.0 (any hitting row),
    while CAP_HIT_THRESHOLD stays as the recorded cell-level reporting statistic."""
    src = DRIVER.read_text()
    calls = re.findall(r"R\.cap_hit_regen\((?:[^)]|\n)*?\)", src)
    assert calls, "cap_hit_regen call site not found"
    for call in calls:  # r3: the correction loop adds a second call site — pin BOTH
        assert "threshold=0.0" in call, (
            "per-turn truncation retry must be unconditional (threshold=0.0) — a single "
            "truncated turn poisons every later turn of that conversation (r1 ISSUE 5)"
        )
    assert "cap_hit_reporting_threshold" in src  # >2% tier kept as reporting


# ── round 3: TOCTOU require-fetch (r2 BLOCKER) ───────────────────────────────────────


def _pod_args(tmp_path, **kw):
    base = dict(
        out_root=str(tmp_path),
        smoke=False,
        force_topics=False,
        generate_topics=False,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_pod_topics_requires_fetch_never_generates(tmp_path, monkeypatch):
    """r2 TOCTOU BLOCKER (fails-pre-fix control): the pod path with NO published
    canonical stimulus must EXIT NON-ZERO — never fall back to generating its own
    (the fetch-miss->generate fallback silently forks the stimulus across the v4
    parallel 7B ∥ 32B legs)."""
    D = _import_driver()

    def _boom(*a, **k):
        raise AssertionError("pod path generated its own stimulus (TOCTOU regression)")

    monkeypatch.setattr(D, "_dispatch_sync", _boom)
    monkeypatch.setattr(D, "_fetch_topics_from_hf", lambda dest: False)  # clean 404
    with pytest.raises(RuntimeError, match="NOT published"):
        D.phase_topics(_pod_args(tmp_path))


def test_pod_topics_transport_degraded_fetch_fails_loud(tmp_path, monkeypatch):
    """The 429/offline channel (LocalEntryNotFoundError, a transient-retried subclass of
    EntryNotFoundError) must fail LOUD too — even a sequenced launch must never degrade
    to generate-own-stimulus."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    import explore_persona_space.orchestrate.hub as hub

    D = _import_driver()

    def _boom(*a, **k):
        raise AssertionError("transport-degraded pod path generated its own stimulus")

    monkeypatch.setattr(D, "_dispatch_sync", _boom)
    monkeypatch.setattr(
        hub,
        "retry_transient",
        lambda fn, *, what: (_ for _ in ()).throw(LocalEntryNotFoundError("429 storm")),
    )
    with pytest.raises(RuntimeError, match="transport-degraded"):
        D.phase_topics(_pod_args(tmp_path))


def test_pod_topics_fetch_success_uses_published_copy(tmp_path, monkeypatch):
    """Require-fetch happy path: the published canonical copy is consumed verbatim."""
    D = _import_driver()

    def _fake_fetch(dest):
        dest.write_text(json.dumps({"domains": {"d": []}, "meta": {"src": "hf"}}))
        return True

    monkeypatch.setattr(D, "_fetch_topics_from_hf", _fake_fetch)
    out = D.phase_topics(_pod_args(tmp_path))
    assert json.loads(out.read_text())["meta"]["src"] == "hf"


def test_generation_requires_explicit_authorization():
    """Source pin: the paid-API generation branch is reachable ONLY under --smoke or
    --generate-topics (the VM pre-launch step); the launcher's pod-side call passes
    neither."""
    src = DRIVER.read_text()
    assert "if not args.smoke and not args.generate_topics:" in src
    # No EXECUTABLE launcher line may pass --generate-topics (comments documenting the
    # VM pre-launch command are fine — the flag must never reach a pod-side invocation).
    executable_hits = [
        ln
        for ln in LAUNCHER.read_text().splitlines()
        if "--generate-topics" in ln and not ln.strip().startswith("#")
    ]
    assert not executable_hits, (
        f"the launcher must NEVER authorize pod-side generation: {executable_hits}"
    )


# ── round 3: v4 amendment arms (A2c all-prompt cap; A2corr correction mode) ─────────


def test_v4_arms_registered_with_correct_wiring():
    D = _import_driver()
    assert len(D.ARMS) == 11  # 9 original + A2c + A2corr (v4)
    a2c = D.ARMS["A2c"]
    assert (a2c["engine"], a2c["arm_slug"], a2c["phase"]) == ("caphook", "cap_allprompt", "B")
    a2corr = D.ARMS["A2corr"]
    assert (a2corr["engine"], a2corr["arm_slug"]) == ("caphook", "cap_ctx")
    assert a2corr["history_mode"] == "a0-drifted"
    # every OTHER arm defaults to capped-throughout (no history_mode key or explicit).
    for name, spec in D.ARMS.items():
        if name != "A2corr":
            assert spec.get("history_mode", "capped-throughout") == "capped-throughout"
    # both new cap arms report firing telemetry (v4: same gate as A1/A2a/A2b).
    assert "A2c" in D.CAP_ARMS and "A2corr" in D.CAP_ARMS
    # τ position resolves from the arm's OWN position_set (all-prompt for A2c).
    sys.path.insert(0, str(REPO))
    from scripts import issue2203_common as C

    assert C.ARM_SPECS["cap_allprompt"]["position_set"] == "all-prompt"
    assert C.ARM_SPECS["cap_ctx"]["position_set"] == "context-end"


def test_collect_read_units_correction_mode(tmp_path):
    """A2corr read units: the RESPONSE at turn t is the arm's capped regeneration
    (corrected[str(t)]) while the conditioning history keeps building from the stored
    A0 DRIFTED messages — and a turn with no correction (partial run) is SKIPPED,
    never silently read from the A0 response."""
    D = _import_driver()

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 97 + 1 for c in text][:8] or [1]}

        def convert_tokens_to_ids(self, token):
            return 999  # never present in ids -> _has_prefix() False -> prefix_end None

    def _ids_fn(tok, ctx):
        # deterministic pseudo-render: length encodes (history, user).
        n = sum(len(m["content"]) for m in ctx["history"]) + len(ctx["user"])
        return [2] * (4 + n % 5)

    raw = {
        "transcripts": {
            "c1": {
                "domain": "coding assistance",
                "history_mode": "a0-drifted",
                "messages": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "A0-DRIFTED-1"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "A0-DRIFTED-2"},
                ],
                "corrected": {"1": "CAPPED-1"},  # turn 2 correction absent (partial)
            }
        }
    }
    units = D._collect_read_units(raw, _Tok(), _ids_fn)
    assert len(units) == 1  # turn 2 skipped — no silent A0-response fallback
    u = units[0]
    assert u["turn"] == 1
    # resp ids came from the CORRECTED text, not A0's drifted response.
    assert u["resp_len"] == len(_Tok()("CAPPED-1")["input_ids"])
    # non-correction transcripts unchanged: same walk reads the stored responses.
    raw["transcripts"]["c1"].pop("corrected")
    units_plain = D._collect_read_units(raw, _Tok(), _ids_fn)
    assert len(units_plain) == 2


def test_a2corr_generate_fails_loud_without_a0(tmp_path):
    """The correction arm consumes A0 Phase A transcripts; a missing A0 canonical file
    is a designed refusal, never a silent empty run."""
    D = _import_driver()
    smoke_domain = D.DOMAINS[0]
    (tmp_path / "topics_personas.json").write_text(
        json.dumps(
            {
                "domains": {
                    smoke_domain: [{"persona": "p0", "persona_published": True, "topics": ["t0"]}]
                },
                "meta": {},
            }
        )
    )
    args = types.SimpleNamespace(
        out_root=str(tmp_path),
        smoke=True,
        arm="A2corr",
        model="tiny",
        think=False,
        shard_id=0,
        num_shards=1,
        history_mode=None,
        force_topics=False,
        generate_topics=False,
    )
    with pytest.raises(FileNotFoundError, match="A0 Phase A"):
        D.phase_generate(args)
