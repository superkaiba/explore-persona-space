"""#1776 crash-fix r13 pins: Batch-API-safe judge custom_ids.

Incident (att crash 2026-07-30T01:00Z): ``issue1776_judge.py`` persona keys
("<stratum>::<context_id>", strata carrying DOTS like ``evil_a0.5``) rode the
Batch API custom_id verbatim and the FIRST ``batches.create`` 400'd on the
``^[a-zA-Z0-9_-]{1,64}$`` constraint; the routing-only ``--dry-run`` could
never catch a charset violation (zero API calls, no pre-submit validation).

These tests FAIL pre-fix (the alias helpers + validator did not exist; the
dry-run dispatch returned ``{}`` instead of raising):

  1. bijective alias round-trip over ALL 26 realized stratum names (read from
     the committed phase3 manifest) at the worst realized context-id length,
     with the 53-char alias budget + 64-char custom_id cap asserted;
  2. collision assert (two personas sanitizing to one alias);
  3. ``validate_batch_custom_ids`` raises on the exact crash shape
     (``evil_a0.5::ci1__00001__00``) naming index + id, and on over-length;
  4. the shared dispatcher validates batch-routed ids in DRY-RUN mode (zero
     API calls) and at ``_run_batch_path`` entry (defense-in-depth).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1776_judge as J  # noqa: E402

from explore_persona_space.eval.judge_dispatch import (  # noqa: E402
    BATCH_CUSTOM_ID_RE,
    _run_batch_path,
    dispatch_judge_items,
    validate_batch_custom_ids,
)

MANIFEST = REPO_ROOT / "eval_results/issue_1776/phase3/raw_completions_manifest.json"
# Longest REALIZED context id across all 26 staged stratum JSONs (measured
# 2026-07-30: 'trait_hallucination_000', 23 chars; charset-clean). The staged
# raw JSONs are not committed, so the worst-case shape is pinned here; the
# run-time build_alias_maps asserts cover the real data on every invocation.
WORST_CTX_ID = "trait_hallucination_000"
CRASH_SHAPE_ID = "evil_a0.5::ci1__00001__00"  # the exact id shape the API 400'd


def _realized_strata() -> list[str]:
    files = json.loads(MANIFEST.read_text())["files"]
    return [f.removesuffix(".json") for f in files]


def test_manifest_lists_26_strata_with_dot_bearing_names():
    strata = _realized_strata()
    assert len(strata) == 26
    assert "evil_a0.5" in strata and "evil_a1" in strata  # dots ARE realized
    assert any(s.endswith("_allpos") for s in strata)


def test_alias_roundtrip_all_realized_strata_charset_and_budget():
    """Every realized persona aliases into the Batch charset, within budget,
    bijectively — and the composed custom_id passes the dispatcher validator."""
    strata = _realized_strata()
    personas = [f"{s}::{WORST_CTX_ID}" for s in strata] + [f"{s}::c0" for s in strata]
    alias_of, persona_of = J.build_alias_maps(personas)
    assert len(alias_of) == len(personas) == len(persona_of)
    composed = []
    for p in personas:
        a = alias_of[p]
        assert BATCH_CUSTOM_ID_RE.fullmatch(a), (p, a)
        assert len(a) <= J.ALIAS_MAX_LEN, (p, a, len(a))
        assert persona_of[a] == p  # bijective round-trip
        cid = f"{a}__00042__03"  # the batch_judge encoder suffix (11 chars)
        assert len(cid) <= 64, (cid, len(cid))
        composed.append(cid)
    validate_batch_custom_ids(composed)  # must not raise
    # The PRE-fix shape (raw persona keys riding the custom_id) is rejected.
    with pytest.raises(ValueError, match=r"custom_id"):
        validate_batch_custom_ids([f"{p}__00042__03" for p in personas])


def test_alias_collision_asserts():
    with pytest.raises(AssertionError, match="collision"):
        J.build_alias_maps(["evil_a0.5::c1", "evil_a0p5::c1"])


def test_alias_over_budget_asserts():
    long_persona = "s" * (J.ALIAS_MAX_LEN + 1)
    with pytest.raises(AssertionError, match="budget"):
        J.build_alias_maps([long_persona])


def test_rehydrate_cids_roundtrip_and_unknown_alias_fails_loud():
    alias_of, persona_of = J.build_alias_maps(["evil_a0.5::c1", "w1_mprime_a2::c9"])
    scored = {f"{a}__00007__01": {"score": 85} for a in alias_of.values()}
    back = J.rehydrate_cids(scored, persona_of)
    assert set(back) == {"evil_a0.5::c1__00007__01", "w1_mprime_a2::c9__00007__01"}
    assert all(v == {"score": 85} for v in back.values())
    with pytest.raises(KeyError):
        J.rehydrate_cids({"unknown_alias__00001__00": {}}, persona_of)


def test_rehydrate_tolerates_double_underscore_inside_alias():
    """rsplit('__', 2) peels only the two fixed numeric suffixes."""
    alias_of, persona_of = J.build_alias_maps(["a__b::c1"])
    (a,) = alias_of.values()
    back = J.rehydrate_cids({f"{a}__00001__00": 1}, persona_of)
    assert set(back) == {"a__b::c1__00001__00"}


def test_validate_batch_custom_ids_names_offender_and_index():
    with pytest.raises(ValueError) as ei:
        validate_batch_custom_ids(["ok_id__00001__00", CRASH_SHAPE_ID])
    msg = str(ei.value)
    assert "items[1]" in msg and CRASH_SHAPE_ID in msg
    with pytest.raises(ValueError, match=r"1\.\.64"):
        validate_batch_custom_ids(["x" * 65])
    with pytest.raises(ValueError):
        validate_batch_custom_ids([""])
    validate_batch_custom_ids([])  # empty ok
    validate_batch_custom_ids(["A-z0_9"])  # valid ok


def _item(cid: str) -> tuple[str, str, str, str]:
    return (cid, "q?", "completion text", "user msg")


def test_dispatch_dry_run_validates_batch_routed_ids(tmp_path):
    """FAILS PRE-FIX: dry-run used to print routing and return {} — now a
    batch-routed charset violation raises at ZERO API cost."""
    with pytest.raises(ValueError, match=r"custom_id"):
        dispatch_judge_items(
            [_item(CRASH_SHAPE_ID)],
            threshold_base=0,  # forces the batch route at n_items=1
            dry_run=True,
            checkpoint_dir=tmp_path,
        )
    # Valid ids keep the dry-run contract (routing printed, {} returned).
    out = dispatch_judge_items(
        [_item("evil_a0p5--ci1__00001__00")],
        threshold_base=0,
        dry_run=True,
        checkpoint_dir=tmp_path,
    )
    assert out == {}


def test_run_batch_path_validates_before_any_state_or_submit(tmp_path):
    """Defense-in-depth: direct batch-path entry raises BEFORE creating any
    dispatch state dir (client is never touched — None passes)."""
    with pytest.raises(ValueError, match=r"custom_id"):
        asyncio.run(
            _run_batch_path(
                [_item(CRASH_SHAPE_ID)],
                judge_model="claude-sonnet-4-5-20250929",
                judge_system_prompt="judge it",
                max_tokens=16,
                sub_batch_size=10,
                checkpoint_dir=tmp_path,
                poll_interval=0.0,
                error_dict_factory=lambda reason: {"error": True, "reasoning": reason},
                client=None,  # type: ignore[arg-type]  # unreachable past validation
            )
        )
    assert not list(tmp_path.iterdir())  # no dispatch_* state dir was created
