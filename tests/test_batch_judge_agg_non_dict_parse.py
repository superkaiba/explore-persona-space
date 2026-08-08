"""Regression: ``_aggregate_persona_scores`` must not crash on a non-dict parse.

Root cause (#778 round 2): ``eval.utils.parse_judge_json`` returns
``json.loads(text)`` verbatim, so a scalar-shaped judge rubric (a bare integer,
or a persona-vectors ``{"score": N}`` the judge answered as the string ``"85"``)
parses to an ``int``. ``_aggregate_persona_scores`` (the Betley legacy aggregator
baked into ``judge_completions_batch``) then called ``s.get("aligned")`` on every
parsed entry, raising ``AttributeError: 'int' object has no attribute 'get'`` —
which aborted the whole ``judge_completions_batch`` call BEFORE ``save_raw`` was
written, so the scalar-rubric caller (issue778_lib.judge_graded) never got its
raw scores back. The fix is a type-guard: a non-dict entry is treated as invalid
at this aggregator (the scalar-rubric caller does its OWN reduction from
``all_scores`` and ignores this return).

These tests exercise ``_aggregate_persona_scores`` directly (the crash site),
which is the cleanest isolation of the guard. Betley (dict) callers are
unaffected — verified by the mixed-persona test.

#778 round 3 adds a PRODUCTION-PATH test: even with the r2 aggregator guard in
place, ``judge_graded`` (the scalar-rubric caller) reduced ``all_scores`` via
``issue778_lib._score_from_parsed``, which returned None for a bare-int parse and
counted it as a DROPPED draw — silently losing coverage on the primary graded
0-100 DV. ``test_judge_graded_carries_bare_int_score`` pins that a bare-int judge
response now yields a real score, not a drop, end to end through ``judge_graded``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from explore_persona_space.eval.batch_judge import (
    _aggregate_persona_scores,
    _collect_legacy_results,
)


def _completions(persona_to_n: dict[str, int]) -> dict[str, dict[str, list[str]]]:
    """{persona: {question: [n dummy completions]}} — the shape the aggregator maps.

    One question per persona so the custom_id enumeration is 1:1 with the draws,
    matching how the scalar-rubric caller packs n_draws repeats as completions.
    """
    return {p: {f"q_{p}": ["answer"] * n} for p, n in persona_to_n.items()}


def _custom_ids(completions: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    """Reproduce the aggregator's custom_id scheme: {persona: [id per draw]}."""
    ids: dict[str, list[str]] = {}
    idx = 0
    for persona, q_comps in completions.items():
        ids[persona] = []
        for _q, comps in q_comps.items():
            for comp_idx in range(len(comps)):
                ids[persona].append(f"{persona}__{idx:05d}__{comp_idx:02d}")
            idx += 1
    return ids


def test_aggregate_persona_scores_skips_non_dict_parse():
    """A persona whose parsed scores are ALL bare ints must not raise.

    Pre-fix this raised ``AttributeError: 'int' object has no attribute 'get'``.
    Post-fix the bare-int draws are counted as invalid (n_samples=0, n_errors=N)
    and the aggregate means are None — the scalar-rubric caller ignores this
    return and reduces from all_scores itself.
    """
    completions = _completions({"item_pv": 3})
    ids = _custom_ids(completions)
    # The persona-vectors judge answered "85"/"90"/"REFUSAL" -> parse_judge_json
    # returns them as bare int / int / str (json.loads verbatim). Use bare ints
    # to reproduce the exact crashing entry class.
    all_scores = {
        ids["item_pv"][0]: 85,
        ids["item_pv"][1]: 90,
        ids["item_pv"][2]: 0,
    }

    result = _aggregate_persona_scores(completions, all_scores)  # must not raise

    agg = result["item_pv"]
    assert agg["n_samples"] == 0, agg
    assert agg["n_errors"] == 3, agg
    assert agg["mean_aligned"] is None, agg
    assert agg["mean_coherent"] is None, agg


def test_aggregate_persona_scores_mixed_dict_and_non_dict():
    """A persona mixing valid Betley dicts and a stray bare int aggregates the
    dicts and skips the int — n_samples counts only the valid dict rows."""
    completions = _completions({"item_mixed": 3})
    ids = _custom_ids(completions)
    all_scores = {
        ids["item_mixed"][0]: {"aligned": 80, "coherent": 90},
        ids["item_mixed"][1]: 60,  # bare-int scalar parse — must be skipped
        ids["item_mixed"][2]: {"aligned": 40, "coherent": 70},
    }

    result = _aggregate_persona_scores(completions, all_scores)  # must not raise

    agg = result["item_mixed"]
    assert agg["n_samples"] == 2, agg  # only the two dict rows are valid
    assert agg["n_errors"] == 1, agg  # the bare int counts as an error/skip
    assert agg["mean_aligned"] == (80 + 40) / 2, agg
    assert agg["mean_coherent"] == (90 + 70) / 2, agg


def test_aggregate_persona_scores_betley_dicts_unchanged():
    """Pure-dict (Betley) input aggregates exactly as before the guard — no
    behavior change for the existing Betley callers."""
    completions = _completions({"persona_a": 2})
    ids = _custom_ids(completions)
    all_scores = {
        ids["persona_a"][0]: {"aligned": 100, "coherent": 100},
        ids["persona_a"][1]: {"aligned": 50, "coherent": 80},
    }

    result = _aggregate_persona_scores(completions, all_scores)

    agg = result["persona_a"]
    assert agg["n_samples"] == 2, agg
    assert agg["n_errors"] == 0, agg
    assert agg["mean_aligned"] == 75.0, agg
    assert agg["mean_coherent"] == 90.0, agg


def test_aggregate_persona_scores_betley_sentinel_still_valid_not_scored():
    """The #545 sentinel path is unaffected: a dict with a "REFUSAL" string
    aligned (no error flag) still counts in n_samples but is excluded from the
    mean sum — the guard only adds a NON-dict skip, it does not touch this."""
    completions = _completions({"persona_s": 2})
    ids = _custom_ids(completions)
    all_scores = {
        ids["persona_s"][0]: {"aligned": "REFUSAL", "coherent": "REFUSAL"},
        ids["persona_s"][1]: {"aligned": 30, "coherent": 60},
    }

    result = _aggregate_persona_scores(completions, all_scores)

    agg = result["persona_s"]
    assert agg["n_samples"] == 2, agg  # both are valid verdicts (no error flag)
    assert agg["n_errors"] == 0, agg
    assert agg["mean_aligned"] == 30.0, agg  # only the numeric row enters the sum
    assert agg["mean_coherent"] == 60.0, agg


def test_collect_legacy_results_keeps_falsy_valid_parse():
    """#1024 D-B row 5: a falsy-but-VALID parse (judge text ``"0"`` -> int 0)
    is KEPT by ``_collect_legacy_results``, never replaced by a parse_error dict.

    Pre-fix the collector used ``parsed or _legacy_error_dict("parse_error")``,
    so any falsy valid parse (``0``, ``{}``, ``false``, ``""``) was silently
    converted to an error dict — a bare 0 is a legitimate graded score (the
    0-100 scale includes 0). Post-fix the check is ``is not None``. Reachable
    only via the frozen legacy #389 ``_submit_and_poll_batch`` callers; the
    graded/#778 path routes through judge_dispatch collectors that already use
    ``is not None``.

    Since #2092 (legacy-drain route parity) the kept in-range numeric arrives
    in the shared ``{"score": N}`` envelope (``_normalize_scalar_score``, same
    as every judge_dispatch drain) — still KEPT, still never a parse_error
    dict; the cross-route envelope pin lives in
    ``tests/test_issue2092_route_parity.py``.
    """

    def _succeeded(cid: str, text: str) -> SimpleNamespace:
        return SimpleNamespace(
            custom_id=cid,
            result=SimpleNamespace(
                type="succeeded",
                message=SimpleNamespace(content=[SimpleNamespace(type="text", text=text)]),
            ),
        )

    fake_client = SimpleNamespace(
        messages=SimpleNamespace(
            batches=SimpleNamespace(
                results=lambda batch_id: iter(
                    [
                        _succeeded("cid_zero", "0"),  # falsy VALID parse — must be kept
                        _succeeded("cid_prose", "no json at all"),  # real parse failure
                    ]
                )
            )
        )
    )

    results: dict[str, dict] = {}
    _collect_legacy_results(fake_client, "batch_x", results)

    # Kept (never a parse_error dict); since #2092 the in-range numeric is
    # carried in the shared {"score": N} envelope, identical to the
    # judge_dispatch drains (no stop_reason key: this fake row carries none).
    assert results["cid_zero"] == {"score": 0}, results
    assert not results["cid_zero"].get("error"), results
    # The genuine parse failure still maps to the legacy parse_error dict.
    assert results["cid_prose"]["error"] is True, results
    assert results["cid_prose"]["reasoning"] == "parse_error", results


def test_judge_graded_carries_bare_int_score(tmp_path, monkeypatch):
    """PRODUCTION-PATH (#778 r3): a bare-int judge parse yields a real score, not
    a dropped draw, end to end through ``judge_graded``.

    We fake ``judge_completions_batch`` (imported inside ``judge_graded`` from
    ``explore_persona_space.eval.batch_judge``) to write a ``save_raw`` whose
    ``all_scores`` maps this item's draw custom_ids to the BARE INT ``85`` — the
    EXACT shape the real #663-hardened client writes when the judge answers the
    rubric with the envelope-less string ``"85"`` (``parse_judge_json`` returns
    ``json.loads("85") == 85`` verbatim). Pre-fix ``_score_from_parsed(85)``
    returned None and every such draw was counted in ``n_dropped_draws``; post-fix
    the item scores 85.0 with zero drops.
    """
    import scripts.issue778_lib as lib

    item_id = "item_pv"
    n_draws = 3
    # judge_graded builds completions[item_id] = {question: [answer]*n_draws} and
    # decodes custom_ids as "{persona}__{idx:05d}__{comp_idx:02d}" with
    # persona == item_id, one question -> idx 0, comp_idx 0..n_draws-1.
    expected_ids = [f"{item_id}__{0:05d}__{c:02d}" for c in range(n_draws)]

    def _fake_batch(*, completions, save_raw, dry_run, **kwargs):
        # Every draw parsed to the bare int 85 (json.loads("85")).
        all_scores = {cid: 85 for cid in expected_ids}
        payload = {"per_persona": {}, "all_scores": all_scores}
        Path(save_raw).parent.mkdir(parents=True, exist_ok=True)
        with open(save_raw, "w") as f:
            json.dump(payload, f)

    monkeypatch.setattr(
        "explore_persona_space.eval.batch_judge.judge_completions_batch", _fake_batch
    )

    result = lib.judge_graded(
        items=[(item_id, "why?", "because.")],
        eval_prompt="Rate {question} / {answer} 0-100.",
        n_draws=n_draws,
        cache_dir=tmp_path / "cache",
        save_raw=tmp_path / "raw.json",
    )

    # The bare-int draws are CARRIED, not dropped: item scores 85.0, no drops.
    assert result.scores[item_id] == 85.0, result.scores
    assert result.n_total_draws == n_draws, result.n_total_draws
    assert result.n_dropped_draws == 0, result.n_dropped_draws
    assert result.per_item_draw_counts[item_id] == n_draws, result.per_item_draw_counts
