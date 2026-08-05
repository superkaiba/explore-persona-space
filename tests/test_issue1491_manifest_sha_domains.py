"""Fails-pre-fix pins for the #1491 P0 wrong-domain sha-pin crash fix.

The P0 manifest build crashed in production (pod-1491, att rp-20260805T043306Z)
on ``val_sha256 drift`` with NO data drift: two stacked domain errors, a
verbatim re-introduction of the incident-#1776 class (``.claude/rules/gotchas.md``
"A sha pin lives in a DOMAIN"; fix ported from ``scripts/issue1776_contexts.py``
@ 04ce114b8fb2):

1. ``VAL_SHA256`` / ``TEST_1000_SHA256`` were the #779 round's
   ``fixed_split(5000, 3600, 400, 1000, 42)`` int64 INDEX-array digests
   (``F._sha_ids`` domain), asserted against PROMPT-string digests — a compare
   that can never pass on any input.
2. ``_sha256_prompt_list`` re-implemented the parent hasher with ``b"\\n"``
   separators where ``N10._sha_prompts`` / ``N50._sha_ids_or_prompts`` use
   ``b"\\x00"`` — so even correct prompt-domain pins could never match.

Each test below FAILS against the pre-fix module and PASSES post-fix (verified
both ways at commit time), except ``test_valtest_parent_return_value_order``,
which pins the order-equivalence the fix relies on (passes both ways by
design — it is the justification for slicing the parent helper's return value,
not the discriminator).

Deliberately network-free and CPU-only: everything in ``tests/`` runs in every
issue's Step 9c gate. ``N50F._pinned_original_shas`` falls back to its module
constants when the committed ``fair_comparison.json`` is absent (sparse
worktrees), so it is deterministic offline either way.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue779_ffc_n1m_generate_capture as N1M  # noqa: E402
import issue779_ffc_n50k_fits as N50F  # noqa: E402
import issue779_ffc_n50k_generate_capture as N50  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue1491_ladder_manifest as MAN  # noqa: E402

# The #1776-frozen prompt-domain pins (issue1776_contexts.py @ 04ce114b8fb2).
FROZEN_ROUND1_PROMPT_SHA = "d40546cd7059780afc50188a0902247a9c2ce49f67ff3d651b87a934a56b8805"
FROZEN_VAL_400_PROMPT_SHA = "e8c8beb0fed383674c08e19cb6d9a56ca781d5182ba77cab138af33c06aed738"
FROZEN_TEST_1000_PROMPT_SHA = "bb60a2827bdc11675699414cda787c9be8ad3b836e9f529a528dc59a6726d9ef"


def _recomputed_index_digests() -> tuple[str, str]:
    """The parent anchor's val/test INDEX-array digests, recomputed live."""
    _train, val, test = F.fixed_split(5000, 3600, 400, 1000, 42)
    return F._sha_ids(val), F._sha_ids(test)


def test_prompt_pins_are_not_index_digests() -> None:
    """THE domain pin. Pre-fix, the values asserted against PROMPT digests
    (``VAL_SHA256`` / ``TEST_SHA256``) WERE the fixed_split INDEX digests, so
    the production assert could never pass; the getattr fallback makes this
    test fail pre-fix on that exact wrong-domain equality (not merely on a
    missing attribute)."""
    idx_val, idx_test = _recomputed_index_digests()
    val_pin = getattr(MAN, "VAL_400_PROMPT_SHA", None) or MAN.VAL_SHA256
    test_pin = getattr(MAN, "TEST_1000_PROMPT_SHA", None) or MAN.TEST_SHA256
    assert val_pin != idx_val, (
        "the val-400 pin compared against a PROMPT digest is the fixed_split "
        "INDEX-array digest — the wrong-domain compare of the P0 crash"
    )
    assert test_pin != idx_test, (
        "the test-1000 pin compared against a PROMPT digest is the fixed_split "
        "INDEX-array digest — the wrong-domain compare of the P0 crash"
    )
    # And they must be exactly the #1776-frozen prompt-domain values, so a
    # future constant swap (or a re-freeze without provenance) fails loud.
    assert val_pin == FROZEN_VAL_400_PROMPT_SHA
    assert test_pin == FROZEN_TEST_1000_PROMPT_SHA
    assert MAN.ROUND1_PROMPT_SHA == FROZEN_ROUND1_PROMPT_SHA


def test_index_pins_match_recomputed_split_and_parent_artifact() -> None:
    """The retained pins are genuinely INDEX-domain: they equal the digests
    recomputed from the parent split recipe AND N50F's pin source."""
    idx_val, idx_test = _recomputed_index_digests()
    assert idx_val == MAN.VAL_400_INDEX_SHA
    assert idx_test == MAN.TEST_1000_INDEX_SHA
    pinned = N50F._pinned_original_shas(N50F.DEFAULT_ORIG_DIR)
    assert pinned["val_sha256"] == MAN.VAL_400_INDEX_SHA
    assert pinned["test_sha256"] == MAN.TEST_1000_INDEX_SHA


def test_membership_hasher_is_the_parent_hasher() -> None:
    """Pre-fix, ``_sha256_prompt_list`` re-implemented the hasher with
    ``b"\\n"`` separators; the parent domain is ``b"\\x00"``-separated."""
    assert not hasattr(MAN, "_sha256_prompt_list"), (
        "the local newline-separated hasher re-implementation must stay deleted "
        "— use the parent N50._sha_ids_or_prompts (b'\\x00'-separated)"
    )
    # Live-path identity: the manifest module hashes with the PARENT's function.
    assert MAN._sha_ids_or_prompts is N50._sha_ids_or_prompts
    expect = hashlib.sha256(b"a\x00b\x00").hexdigest()
    assert MAN._sha_ids_or_prompts(["a", "b"]) == expect
    # The two hasher domains genuinely differ (the second-order P0 error).
    assert expect != hashlib.sha256(b"a\nb\n").hexdigest()


def test_membership_assert_fires_loud_on_stream_drift() -> None:
    """The three-domain check's first leg (round-1 prompt MEMBERSHIP) fires
    with a domain-naming diagnostic on a non-pinned stream."""
    r1 = [f"synthetic prompt {i}" for i in range(5000)]
    with pytest.raises(AssertionError, match="round-1 prompt-membership drift"):
        MAN._assert_pinned_membership(r1, r1[:400], r1[400:1400])


def test_build_manifest_production_branch_calls_membership_check() -> None:
    """Wiring pin: the production (non-smoke) branch of build_manifest calls
    the three-domain check — the helper must not become a dead gate."""
    src = inspect.getsource(MAN.build_manifest)
    assert "_assert_pinned_membership(" in src


def test_valtest_parent_return_value_order() -> None:
    """Order-equivalence justification (passes pre- AND post-fix): the parent's
    ``_valtest_prompts_from_round1`` returns ``list(val) + list(test)`` under
    the SAME ``fixed_split(5000, 3600, 400, 1000, 42)``, so ``[:400]`` /
    ``[400:]`` equals the pre-fix hand slicing by the split indices."""
    r1 = [f"p{i}" for i in range(5000)]
    valtest = N1M._valtest_prompts_from_round1(r1, check_ctx0=False)
    _train, val, test = F.fixed_split(5000, 3600, 400, 1000, 42)
    assert valtest[:400] == [r1[i] for i in val]
    assert valtest[400:] == [r1[i] for i in test]
