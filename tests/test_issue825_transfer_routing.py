"""#825 onpolicy-separator-control: transfer-script output-routing invariants.

Concern transfer-phasec-default-output-routing (code-review v14): the
plan-section-10 Phase-C commands pass ONLY the source-override flags, so the
resolved defaults must derive the non-clobbering on-policy routing
(onpolicy_sep_to_chat_<model>.json — the name the decision + figures scripts
read — plus the distinct issue825_onpolicy_sep_control Hub prefix), the
round-6 flag-less invocation must resolve to the round-6 defaults unchanged,
and the overwrite guard must trip on a provenance mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue825_base_sep_transfer as tr  # noqa: E402


def test_round6_default_routing_unchanged():
    """Flag-less invocation keeps the round-6 names byte-identically."""
    out, prefix, override = tr.resolve_output_routing("base", None, None, None, None)
    assert (out, prefix, override) == ("base_sep_to_chat.json", tr.HF_PREFIX_OUT, False)
    assert tr.HF_PREFIX_OUT == "issue825_base_sep_control"
    assert tr.source_provenance("base", None, None) == {}


def test_plan_phasec_base_leg_derives_onpolicy_routing():
    """The plan-section-10 base leg (only --model/--source-store-dir) routes
    away from the round-6 out-name AND the round-6 Hub maps prefix."""
    out, prefix, override = tr.resolve_output_routing(
        "base", Path("data/issue_825/onpolicy_sep/base/store/armC"), None, None, None
    )
    assert out == "onpolicy_sep_to_chat_base.json"
    assert prefix == tr.ONPOLICY_HF_PREFIX_OUT == "issue825_onpolicy_sep_control"
    assert override is True


def test_plan_phasec_instruct_leg_and_prefix_variants():
    """Instruct leg gets its own out-name (no cross-leg overwrite); the Hub
    source-prefix override and a bare --model instruct route identically."""
    out, prefix, _ = tr.resolve_output_routing(
        "instruct", Path("data/issue_825/onpolicy_sep/instruct/store/armC"), None, None, None
    )
    assert (out, prefix) == ("onpolicy_sep_to_chat_instruct.json", tr.ONPOLICY_HF_PREFIX_OUT)
    out2, prefix2, override2 = tr.resolve_output_routing(
        "base", None, "issue825_onpolicy_sep_control/analysis_tensors/armC_base", None, None
    )
    assert (out2, prefix2, override2) == (
        "onpolicy_sep_to_chat_base.json",
        tr.ONPOLICY_HF_PREFIX_OUT,
        True,
    )
    out3, prefix3, override3 = tr.resolve_output_routing("instruct", None, None, None, None)
    assert (out3, prefix3, override3) == (
        "onpolicy_sep_to_chat_instruct.json",
        tr.ONPOLICY_HF_PREFIX_OUT,
        False,  # source stays the round-6 default; only the ROUTING derives
    )


def test_explicit_routing_flags_always_win():
    out, prefix, _ = tr.resolve_output_routing(
        "base", Path("some/store"), None, "custom.json", "my/prefix"
    )
    assert (out, prefix) == ("custom.json", "my/prefix")


def test_overwrite_guard_trips_on_provenance_mismatch(tmp_path):
    """An on-policy invocation resolving onto a round-6-shaped JSON (metadata
    WITHOUT source keys) refuses to overwrite; --force-overwrite bypasses."""
    out_path = tmp_path / "base_sep_to_chat.json"
    out_path.write_text(json.dumps({"metadata": {"script": tr.SCRIPT, "seed": 0}}))
    prov = tr.source_provenance("base", Path("data/onpolicy/store/armC"), None)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        tr.guard_out_overwrite(out_path, prov, force=False)
    tr.guard_out_overwrite(out_path, prov, force=True)  # explicit escape: no raise


def test_overwrite_guard_allows_matching_provenance_rerun(tmp_path):
    """Idempotent same-leg retry proceeds; a missing file is a no-op."""
    prov = tr.source_provenance("instruct", Path("d/store/armC"), None)
    out_path = tmp_path / "onpolicy_sep_to_chat_instruct.json"
    out_path.write_text(json.dumps({"metadata": {"seed": 0, **prov}}))
    tr.guard_out_overwrite(out_path, prov, force=False)
    tr.guard_out_overwrite(tmp_path / "absent.json", prov, force=False)


def test_overwrite_guard_fails_loud_on_unreadable_existing(tmp_path):
    out_path = tmp_path / "onpolicy_sep_to_chat_base.json"
    out_path.write_text("{not json")
    with pytest.raises(RuntimeError, match="unreadable"):
        tr.guard_out_overwrite(out_path, tr.source_provenance("base", Path("d"), None), force=False)
