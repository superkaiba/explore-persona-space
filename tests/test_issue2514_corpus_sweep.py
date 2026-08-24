"""Tests for scripts/issue2514_c26c27_corpus_sweep.py — the #2514 before/after
corpus-sweep classifier (task #2514, round-3 reconciler items).

- item 1a (BLOCKER, falsifiability): ``cmd_classify`` REFUSES a leg whose
  header regime differs from the approved #2514 pins. The two negative
  controls reproduce the round-2 reconciler's wrong-but-self-consistent
  counterexamples — a B200 remap and an empty mirror (total capture loss) —
  each with a self-consistent flipped row that the round-2 classifier
  bucketed ``expected-inversion`` at exit 0 (demonstrated red against
  ``3623443af1:scripts/issue2514_c26c27_corpus_sweep.py``).
- item 1b: ``expected-inversion`` additionally requires the realized
  transition to BE an inversion — (WARN,PASS)/(PASS,WARN); ``WARN->SKIP``
  and every other transition route to ``unexplained``.
- item 2: ``module_sha`` (sha256 of the swept module's file bytes) is part
  of the resume regime identity.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_SCRIPT = REPO_ROOT / "scripts" / "issue2514_c26c27_corpus_sweep.py"
_spec = importlib.util.spec_from_file_location("issue2514_c26c27_corpus_sweep", _SCRIPT)
sweep_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["issue2514_c26c27_corpus_sweep"] = sweep_mod
_spec.loader.exec_module(sweep_mod)  # type: ignore[union-attr]


def _approved_header(leg: str) -> dict:
    """A header carrying the approved regime for ``leg`` (JSON round-tripped
    so the fixture cannot share mutable state with the module constant)."""
    hdr = {"verify_plan_path": f"/tmp/{leg}.py", "module_sha": "0" * 64, "n_plans": 1}
    hdr.update(json.loads(json.dumps(sweep_mod._APPROVED_LEG_REGIMES[leg])))
    return hdr


def _row(plan: str, intents: list[str], c26: str, c26_rows: list | None = None) -> dict:
    return {
        "plan": plan,
        "intents": intents,
        "c26": c26,
        "c26_detail": "",
        "c27": "SKIP",
        "c27_detail": "no activation-capture vocabulary detected",
        "c26_rows": c26_rows or [],
    }


def _write_leg(path: Path, header: dict, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"header": header}) + "\n")
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _classify_ns(tmp_path: Path) -> Namespace:
    return Namespace(
        before=tmp_path / "before.jsonl",
        after=tmp_path / "after.jsonl",
        out=tmp_path / "diff.json",
    )


# ─── item 1a: negative controls — the falsifiability proof ──────────────────


def test_classify_refuses_unapproved_b200_remap(tmp_path):
    """AFTER header carries an unapproved A100/H100->B200 remap with a
    self-consistent flipped row (realized PASS->WARN == predicted under each
    leg's own mirror; even a directional inversion). The round-2 classifier
    exits 0 bucketing it expected-inversion; the pinned classifier refuses
    BEFORE any bucketing."""
    ns = _classify_ns(tmp_path)
    c26_rows = [{"basis_family": "A100", "conv_families": ["A100"], "scaling": False}]
    # before: lora-7b -> A100, basis A100 in routed -> predicted PASS == realized.
    _write_leg(ns.before, _approved_header("before"), [_row("p1", ["lora-7b"], "PASS", c26_rows)])
    hdr_after = _approved_header("after")
    hdr_after["mirror"] = {
        k: ("B200" if v == "H100" else v) for k, v in hdr_after["mirror"].items()
    }
    # after: lora-7b -> B200; basis A100 not routed, no conv/scaling escape ->
    # predicted WARN == realized. Self-consistent, yet an unapproved mapping.
    _write_leg(ns.after, hdr_after, [_row("p1", ["lora-7b"], "WARN", c26_rows)])
    with pytest.raises(SystemExit) as exc:
        sweep_mod.cmd_classify(ns)
    msg = str(exc.value)
    assert "after" in msg and "mirror" in msg and "REFUSED" in msg
    assert not ns.out.exists(), "refusal must precede any classification artifact"


def test_classify_refuses_empty_after_mirror_capture_loss(tmp_path):
    """AFTER header carries an EMPTY mirror (total capture loss — the round-2
    reconciler's constructed worst case). The self-consistent WARN->SKIP flip
    (empty routed set predicts SKIP) exits 0 as 'expected-inversion' under the
    round-2 classifier; the pinned classifier refuses loud."""
    ns = _classify_ns(tmp_path)
    c26_rows = [{"basis_family": "A100", "conv_families": ["A100"], "scaling": False}]
    # before: eval -> L4; basis A100 not routed, no escape -> predicted WARN == realized.
    _write_leg(ns.before, _approved_header("before"), [_row("p1", ["eval"], "WARN", c26_rows)])
    hdr_after = _approved_header("after")
    hdr_after["mirror"] = {}
    _write_leg(ns.after, hdr_after, [_row("p1", ["eval"], "SKIP", c26_rows)])
    with pytest.raises(SystemExit) as exc:
        sweep_mod.cmd_classify(ns)
    assert "REFUSED" in str(exc.value)
    assert not ns.out.exists()


def test_classify_refuses_unapproved_before_regime(tmp_path):
    """The pin binds BOTH legs — a garbage BEFORE regime refuses too."""
    ns = _classify_ns(tmp_path)
    hdr_before = _approved_header("before")
    hdr_before["lane_head"] = "gcp"
    _write_leg(ns.before, hdr_before, [_row("p1", [], "SKIP")])
    _write_leg(ns.after, _approved_header("after"), [_row("p1", [], "SKIP")])
    with pytest.raises(SystemExit) as exc:
        sweep_mod.cmd_classify(ns)
    assert "before" in str(exc.value) and "lane_head" in str(exc.value)


# ─── item 1b: the direction conjunct ─────────────────────────────────────────


def test_c26_direction_conjunct_routes_nondirectional_flips_to_unexplained():
    bucket = sweep_mod._bucket_c26_flip
    # The two registered inversion directions, replay-matched -> expected.
    assert bucket("WARN", "PASS", "WARN", "PASS") == "expected-inversion"
    assert bucket("PASS", "WARN", "PASS", "WARN") == "expected-inversion"
    # A replay-matched WARN->SKIP (the empty-mirror shape) is NOT an inversion.
    assert bucket("WARN", "SKIP", "WARN", "SKIP") == "unexplained"
    assert bucket("SKIP", "PASS", "SKIP", "PASS") == "unexplained"
    assert bucket("PASS", "SKIP", "PASS", "SKIP") == "unexplained"
    # A directional transition with a replay mismatch on either side stays KILL.
    assert bucket("WARN", "PASS", "PASS", "PASS") == "unexplained"
    assert bucket("PASS", "WARN", "PASS", "PASS") == "unexplained"


def test_classify_green_on_approved_regimes_directional_flip(tmp_path):
    """Green path: approved regimes + a replay-matched directional flip (the
    registered 'H100-basis plan stops WARNing' class) -> exit 0, bucketed
    expected-inversion — guards against the pin over-tightening."""
    ns = _classify_ns(tmp_path)
    c26_rows = [{"basis_family": "H100", "conv_families": ["H100"], "scaling": False}]
    # before: eval -> L4; H100 basis not routed -> predicted WARN == realized.
    _write_leg(ns.before, _approved_header("before"), [_row("p1", ["eval"], "WARN", c26_rows)])
    # after: eval -> H100; basis in routed -> predicted PASS == realized.
    _write_leg(ns.after, _approved_header("after"), [_row("p1", ["eval"], "PASS", c26_rows)])
    assert sweep_mod.cmd_classify(ns) == 0
    summary = json.loads(ns.out.read_text())
    assert summary["counts"] == {
        "expected-inversion": 1,
        "c27-disarm": 0,
        "new-key-arming": 0,
        "unexplained": 0,
    }


def test_classify_kill_on_replay_mismatch_under_approved_regimes(tmp_path):
    """A flip the mirrors do NOT predict (realized WARN->PASS on an
    A100-basis sweep-8g-a100 plan, mapped A100 in BOTH mirrors -> predicted
    PASS on both sides) lands in unexplained -> exit 1: the KILL criterion
    fires under fully-approved regimes."""
    ns = _classify_ns(tmp_path)
    c26_rows = [{"basis_family": "A100", "conv_families": ["A100"], "scaling": False}]
    _write_leg(
        ns.before, _approved_header("before"), [_row("p1", ["sweep-8g-a100"], "WARN", c26_rows)]
    )
    _write_leg(
        ns.after, _approved_header("after"), [_row("p1", ["sweep-8g-a100"], "PASS", c26_rows)]
    )
    assert sweep_mod.cmd_classify(ns) == 1
    summary = json.loads(ns.out.read_text())
    assert summary["counts"]["unexplained"] == 1


# ─── item 2: module content is part of the resume identity ──────────────────


def test_module_sha_is_a_resume_regime_key(tmp_path):
    assert "module_sha" in sweep_mod._HEADER_REGIME_KEYS
    hdr = {
        "verify_plan_path": "/tmp/x.py",
        "module_sha": "a" * 64,
        "mirror": {"eval": "H100"},
        "lane_head": "runpod",
        "under_hbm_intents": [],
    }
    out = tmp_path / "sweep.jsonl"
    _write_leg(out, hdr, [_row("p1", ["eval"], "PASS")])
    # Same regime -> prior row kept.
    assert set(sweep_mod._resume_rows(out, dict(hdr))) == {"p1"}
    # Same path, different module CONTENT -> restart from scratch.
    changed = dict(hdr)
    changed["module_sha"] = "b" * 64
    assert sweep_mod._resume_rows(out, changed) == {}
