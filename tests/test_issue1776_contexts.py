"""#1776 r8 crash-fix pins (p1_contexts test-1000 membership).

The plan §10 split pins are sha256 digests of the ORIGINAL #779 round's
``fixed_split(5000, 3600, 400, 1000, 42)`` int64 INDEX arrays — NOT
prompt-string digests. r1-r7 of ``issue1776_contexts.lmsys_contexts`` compared
``N10._sha_prompts(prompts)`` against those pins (wrong domain — could never
pass on any stream) and crashed the pod run at p1_contexts. These tests pin:

  1. the §10 pin constants ARE the fixed_split index shas (pure RNG, no
     network) — the pass-by-construction leg of the r8 three-domain check;
  2. the r8 recovery asserts membership in the PROMPT domain first: a
     non-pinned stream fails the round-1 prompt-membership assert (fails
     PRE-fix by message — the old code died later with "test-1000 sha drift").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue779_fitter_fair_comparison as F
import issue1776_contexts as CX


def test_plan_pins_are_fixed_split_index_shas():
    """§10 pins == F._sha_ids over the pinned split's index arrays (no network)."""
    _r1, val, test = F.fixed_split(5000, 3600, 400, 1000, F.SPLIT_SEED)
    assert F._sha_ids(val) == CX.VAL_400_SHA
    assert F._sha_ids(test) == CX.TEST_1000_SHA
    assert len(val) == 400 and len(test) == 1000


def test_pinned_original_shas_artifact_matches_module_pins():
    """The #779 fair_comparison.json artifact path (or its constant fallback)
    resolves to the SAME index shas the contexts module pins."""
    pinned = CX.N50F._pinned_original_shas(CX.N50F.DEFAULT_ORIG_DIR)
    assert pinned["val_sha256"] == CX.VAL_400_SHA
    assert pinned["test_sha256"] == CX.TEST_1000_SHA


def test_lmsys_contexts_membership_assert_fires_in_prompt_domain():
    """A non-pinned (synthetic) stream trips the round-1 prompt-MEMBERSHIP
    assert — the r8 first check. Fails pre-fix: the r1-r7 code had no
    membership assert and died later with 'test-1000 sha drift' (the
    wrong-domain compare of prompt digests against the INDEX pins)."""
    args = argparse.Namespace(smoke=False, max_scan=6000, n_lmsys=100)
    synth = [
        {"conversation": [{"role": "user", "content": f"synthetic round-1 context {i}"}]}
        for i in range(CX.N50.N_ROUND1 + 50)
    ]
    with pytest.raises(AssertionError, match="round-1 prompt-membership drift"):
        CX.lmsys_contexts(args, stream_iter=synth)
