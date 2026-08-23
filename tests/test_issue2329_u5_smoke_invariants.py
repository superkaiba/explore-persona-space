"""U5 smoke-round invariants for issue #2329 (task #2329, unit 5/5).

Two permanent pins from the final pre-dispatch smoke:

1. ``issue2329_mapshift.persist_work_tensors`` uploads the plan-declared
   ``analysis_tensors/mapshift/*`` tensor class through the run driver's
   fail-loud ``upload_dir_hf`` seam (plan phase_outputs P7; the U5 plan-glob
   parity catch — before the fix NO upload site existed for this class).
2. The pod sentinel the driver's ``phase_upload`` writes round-trips through
   ``poll_pipeline._parse_sentinel`` (producer->poller contract): the REAL
   ``_sentinel_payload`` on a fresh (deferred-leg) out-root, wrapped in the
   exact envelope ``phase_upload`` writes, parses with every
   ``_SENTINEL_REQUIRED_KEYS`` member present.

Both tests are CPU-only, network-free, and repo-root-path-free (tmp_path).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2329_mapshift as M  # noqa: E402
import issue2329_run as R  # noqa: E402
import poll_pipeline as P  # noqa: E402


def test_persist_work_tensors_uploads_both_plan_declared_prefixes(tmp_path):
    """The real body reaches upload_dir_hf once per tensor dir, plan-exact prefixes."""
    import torch

    work = tmp_path / "work" / "full"
    (work / "fresh_preds").mkdir(parents=True)
    (work / "fresh_shufmap").mkdir(parents=True)
    torch.save({"x": torch.zeros(1)}, work / "fresh_preds" / "oof_L30.pt")
    torch.save({"x": torch.zeros(1)}, work / "fresh_shufmap" / "shufmap_L30.pt")
    cfg = mock.Mock(spec_set=["work_root", "fresh_dir", "shufmap_dir", "leg"])
    cfg.work_root = work
    cfg.fresh_dir = work / "fresh_preds"
    cfg.shufmap_dir = work / "fresh_shufmap"
    cfg.leg = "full"
    fake = mock.create_autospec(R.upload_dir_hf, return_value=["ok"])
    with mock.patch.object(R, "upload_dir_hf", fake):
        M.persist_work_tensors(cfg)
    prefixes = [c.args[1] for c in fake.call_args_list]
    assert prefixes == [
        "issue2329_q35rerun/analysis_tensors/mapshift/fresh_preds",
        "issue2329_q35rerun/analysis_tensors/mapshift/fresh_shufmap",
    ]
    # empty dirs upload nothing (smoke/pilot residue never spends a commit)
    fake2 = mock.create_autospec(R.upload_dir_hf, return_value=["ok"])
    empty = tmp_path / "empty"
    (empty / "fresh_preds").mkdir(parents=True)
    cfg.fresh_dir = empty / "fresh_preds"
    cfg.shufmap_dir = empty / "fresh_shufmap"  # does not exist
    with mock.patch.object(R, "upload_dir_hf", fake2):
        M.persist_work_tensors(cfg)
    assert fake2.call_args_list == []


def test_sentinel_payload_round_trips_through_poller(tmp_path):
    """phase_upload's envelope over the REAL payload parses via _parse_sentinel."""
    args = R.parse_args(
        [
            "--phase",
            "upload",
            "--out-root",
            str(tmp_path / "out"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--upload",
            "none",
        ]
    )
    cfg = R.build_config(args)
    for d in (
        cfg.bank_dir,
        cfg.anchors_dir,
        cfg.rollouts_dir,
        cfg.va_dir,
        cfg.margin_dir,
        cfg.fact_dir,
        cfg.manifest_dir / "blocks",
    ):
        d.mkdir(parents=True, exist_ok=True)
    payload = R._sentinel_payload(cfg, {})
    body = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "note": payload,
    }
    parsed = P._parse_sentinel("logs/issue-2329-results.json", json.dumps(body))
    assert parsed is not None, "poller rejected the producer's sentinel"
    assert not [k for k in P._SENTINEL_REQUIRED_KEYS if k not in parsed]
    assert parsed["kind"] == "epm:results"
    note = parsed["note"]
    assert note["deferred_leg"] is True  # fresh out-root: no local grid blocks
    assert "eval_numbers" in note and "reproducibility_card" in note
