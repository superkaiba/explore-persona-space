"""#2092 route-parity pins: legacy batch drain == primary batch drain.

The #1434/#778 scalar-passthrough fix (``_normalize_scalar_score``) landed on
the three ``judge_dispatch`` drain sites but was never ported to the LEGACY
batch drain (``batch_judge._collect_legacy_results``), so identical judge text
produced route-dependent persisted shapes (bare int ``95`` on the legacy route
vs ``{"score": 95}`` on the primary route). #2092 ports the SAME function
object into the legacy drain; these tests pin the parity so the next partial
port fails a committed test instead of shipping.

Scope (plan #2092 §1 criteria 2-4):

- ``test_route_parity_classification_and_kept_envelope`` drains a canonical
  12-text set through BOTH REAL drain bodies (fake anthropic client at the
  ONLY network seam) and asserts identical kept/drop classification plus
  identical kept-row envelopes. Error-dict SHAPES legitimately differ per
  route (``_legacy_error_dict`` vs the caller-supplied factory) — parity is
  classification + kept-envelope, never error-dict field equality.
- ``test_drift_pin_legacy_drain_calls_shared_normalizer`` monkeypatches the
  judge_dispatch module attribute and asserts the legacy drain reflects it —
  a re-implemented local copy of the normalizer cannot pass.
- ``test_consumer_parity_bare_int_vs_score_envelope`` pins the plan's
  kill-criterion claim: downstream consumers treat the pre-fix bare-int shape
  and the post-fix ``{"score": N}`` envelope identically.
- ``test_refusal_stop_reason_attached_on_parse_error_both_routes`` documents
  the #1739 refusal class (empty text + ``stop_reason="refusal"``) flowing
  identically through both drains (#2021 ``_with_stop_reason`` surface).

Fails-pre-fix property (plan criterion 4): with the ``_normalize_scalar_score``
call absent from ``_collect_legacy_results``, the ``"95"`` row drains to the
bare int ``95`` (no ``{"score": ...}`` envelope, no ``stop_reason`` key), so
the kept-envelope assert below fails against the unfixed drain.
"""

from __future__ import annotations

from types import SimpleNamespace

from explore_persona_space.eval import batch_judge, graded_judge, judge_dispatch

# ── Fake anthropic client (network seam ONLY; both drain bodies run real) ────


def _succeeded_row(custom_id: str, text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    """One SDK-shaped succeeded batch-result row.

    Mirrors the attribute surface both drains dereference: ``.custom_id``,
    ``.result.type``, ``.result.message.content[].type/.text``,
    ``.result.message.stop_reason``.
    """
    block = SimpleNamespace(type="text", text=text)
    message = SimpleNamespace(content=[block], stop_reason=stop_reason)
    return SimpleNamespace(
        custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=message)
    )


class _FakeBatches:
    """Signature-conformant stand-in for ``client.messages.batches``."""

    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self._rows = rows

    def results(self, batch_id: str):
        assert isinstance(batch_id, str) and batch_id, batch_id
        return iter(self._rows)


class _FakeClient:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self.messages = SimpleNamespace(batches=_FakeBatches(rows))


def _drain_both(rows: list[SimpleNamespace]) -> tuple[dict[str, object], dict[str, object]]:
    """Run BOTH real drain bodies over ``rows``; return (legacy, primary) scores."""
    legacy: dict[str, object] = {}
    batch_judge._collect_legacy_results(_FakeClient(rows), "batch_parity_test", legacy)
    primary, retriable, expired, quarantined, canceled = judge_dispatch._collect_batch_results(
        _FakeClient(rows), "batch_parity_test", judge_dispatch._default_error_dict
    )
    assert retriable == expired == quarantined == canceled == []  # succeeded rows only
    return legacy, primary


def _is_drop(value: object) -> bool:
    """The drain-level drop classification: an error dict."""
    return isinstance(value, dict) and bool(value.get("error"))


# ── Canonical 12-text set (plan §1 criterion 2) ──────────────────────────────

_DROP = object()  # sentinel: per-route parse_error dict expected

CANONICAL_TEXTS: list[tuple[str, str, object]] = [
    # (custom_id, raw judge text, expected kept envelope or _DROP)
    ("t00", "95", {"score": 95, "stop_reason": "end_turn"}),
    ("t01", "0", {"score": 0, "stop_reason": "end_turn"}),
    ("t02", "100", {"score": 100, "stop_reason": "end_turn"}),
    ("t03", "87.5", {"score": 87.5, "stop_reason": "end_turn"}),
    # Out-of-range numerics: scalar passthrough (dropped DOWNSTREAM by
    # _score_from_parsed's range check — never coerced at the drain).
    ("t04", "150", 150),
    ("t05", "-5", -5),
    # bool: passes through unchanged; never a score (bool is an int subclass).
    ("t06", "true", True),
    # JSON string: passes through; refusal-tallied downstream.
    ("t07", '"REFUSAL"', "REFUSAL"),
    # Parse failures: per-route parse_error dict (shapes differ per route).
    ("t08", "", _DROP),
    ("t09", "hello", _DROP),
    # Dict verdicts: unchanged (plus the #2021 stop_reason annotation).
    ("t10", '{"score": 42}', {"score": 42, "stop_reason": "end_turn"}),
    (
        "t11",
        '{"aligned": 80, "coherent": 90}',
        {"aligned": 80, "coherent": 90, "stop_reason": "end_turn"},
    ),
]


def test_route_parity_classification_and_kept_envelope():
    """Identical text through both REAL drains: same kept/drop class, same kept envelope."""
    rows = [_succeeded_row(cid, text) for cid, text, _ in CANONICAL_TEXTS]
    legacy, primary = _drain_both(rows)

    assert set(legacy) == set(primary) == {cid for cid, _, _ in CANONICAL_TEXTS}
    for cid, text, expected in CANONICAL_TEXTS:
        lv, pv = legacy[cid], primary[cid]
        if expected is _DROP:
            # Parity is CLASSIFICATION here — error-dict field shapes are
            # per-route by design (_legacy_error_dict vs the supplied factory).
            assert _is_drop(lv), (cid, text, lv)
            assert _is_drop(pv), (cid, text, pv)
        else:
            assert not _is_drop(lv) and not _is_drop(pv), (cid, text, lv, pv)
            # Kept envelopes are IDENTICAL across routes (fails pre-fix on t00:
            # the unfixed legacy drain yields the bare int 95, not the
            # {"score": 95, "stop_reason": "end_turn"} envelope).
            assert lv == expected, (cid, text, lv)
            assert pv == expected, (cid, text, pv)
            assert type(lv) is type(pv), (cid, text, lv, pv)


def test_drift_pin_legacy_drain_calls_shared_normalizer(monkeypatch):
    """The legacy drain calls judge_dispatch._normalize_scalar_score BY MODULE ATTRIBUTE.

    The function-level import inside ``_collect_legacy_results`` re-resolves the
    judge_dispatch attribute on every call, so a monkeypatched marking wrapper
    must show up in the drain's output. A re-implemented local copy of the
    normalizer (the copy-drift this task exists to prevent) cannot pass this.
    """
    sentinel = {"score": -1, "marked_by": "drift-pin"}
    real_normalizer = judge_dispatch._normalize_scalar_score

    def _marking_normalizer(parsed: object) -> object:
        if parsed == 77:
            return dict(sentinel)
        return real_normalizer(parsed)

    monkeypatch.setattr(judge_dispatch, "_normalize_scalar_score", _marking_normalizer)

    legacy: dict[str, object] = {}
    batch_judge._collect_legacy_results(
        _FakeClient([_succeeded_row("t77", "77")]), "batch_drift_pin", legacy
    )
    assert legacy["t77"] == {**sentinel, "stop_reason": "end_turn"}, legacy["t77"]


def test_consumer_parity_bare_int_vs_score_envelope():
    """Downstream consumers treat bare-int (pre-fix) and {"score": N} (post-fix) alike.

    Pins the plan §6 kill-criterion claim: no known legacy-drain consumer
    REQUIRES the bare-int persisted shape.
    """
    # graded reducer: both shapes reduce to the same float.
    assert graded_judge._score_from_parsed(95) == graded_judge._score_from_parsed({"score": 95})
    assert graded_judge._score_from_parsed(95) == 95.0

    # Betley aggregation valid-filter: BOTH shapes are non-Betley-valid
    # (no numeric "aligned" field), under either persisted shape.
    completions = {"persona_a": {"q0": ["completion text"]}}
    for shape in (95, {"score": 95}):
        agg = batch_judge._aggregate_persona_scores(completions, {"persona_a__00000__00": shape})
        assert agg["persona_a"]["n_samples"] == 0, (shape, agg)
        assert agg["persona_a"]["n_errors"] == 1, (shape, agg)
        assert agg["persona_a"]["mean_aligned"] is None, (shape, agg)


def test_refusal_stop_reason_attached_on_parse_error_both_routes():
    """#1739 refusal class: empty text + stop_reason="refusal" through BOTH drains.

    Both routes classify the row as a parse_error DROP and attach
    ``stop_reason: "refusal"`` to their per-route error dict (#2021
    ``_with_stop_reason`` surface), so refusal-vs-content diagnosis reads
    identically off either route's persisted rows.
    """
    rows = [_succeeded_row("r00", "", stop_reason="refusal")]
    legacy, primary = _drain_both(rows)
    for route_name, value in (("legacy", legacy["r00"]), ("primary", primary["r00"])):
        assert _is_drop(value), (route_name, value)
        assert value.get("stop_reason") == "refusal", (route_name, value)
        assert value.get("reasoning") == "parse_error", (route_name, value)
