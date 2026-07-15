"""Pin the #825 G2 boundary-position carve-out (crash-fix round 10').

The pretrained plain-text render's "Assistant: "+<answer> BPE delimiter merge
shifts context_pos by +-1 between the new capture and the banked store on a
small row tail (60/2572 observed 2026-07-15), making G2 cosine parity
unverifiable for exactly those rows. The carve-out excludes them PAIR-SAFE and
HALTs when the mismatch rate exceeds BOUNDARY_EXCL_MAX_RATE (systemic break,
not a boundary tail). These tests trip both branches with a synthetic banked
index — no HF, no model.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from issue825_onpolicy_turn_depth_fit import (  # noqa: E402
    BOUNDARY_EXCL_MAX_RATE,
    _boundary_pos_carveout,
)


def _fixture(tmp_path: Path, mismatch_idx: set[int], n: int = 40):
    """Synthetic banked context index + own rows with optional +1 pos shifts."""
    d = tmp_path / "dynamics_pretrained"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "row_index_context_k_shard00000.jsonl", "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "conv_id": f"c{i}",
                        "turn_index": 1,
                        "kind": "context_k",
                        "token_start": 10 * i,
                        "token_end": 10 * i + 1,
                    }
                )
                + "\n"
            )
    own_rows = [
        {
            "conv_id": f"c{i}",
            "turn_index": 1,
            "context_pos": 10 * i + (1 if i in mismatch_idx else 0),
        }
        for i in range(n)
    ]
    pairing = [(i, i, f"c{i}", 1) for i in range(n)]
    return own_rows, np.arange(n, dtype=np.int64), pairing, list(range(n))


def test_boundary_tail_excluded_pair_safe(tmp_path):
    """A sub-cap mismatch tail is excluded (mask False exactly there) + recorded."""
    own_rows, own_sel, pairing, kept = _fixture(tmp_path, {7})
    keep, rec = _boundary_pos_carveout(tmp_path, "pretrained", own_rows, own_sel, pairing, kept)
    assert keep.dtype == bool and keep.size == 40
    assert int(keep.sum()) == 39 and not keep[7]
    assert rec["n_excluded"] == 1
    assert rec["pos_delta_counts"] == {"1": 1}
    assert rec["per_turn_excluded"] == {"1": 1}
    assert rec["caveat"]  # survivor caveat set whenever rows are excluded


def test_clean_rows_pass_through(tmp_path):
    """Zero mismatches: full keep mask, no caveat (the instruct-arm shape)."""
    own_rows, own_sel, pairing, kept = _fixture(tmp_path, set())
    keep, rec = _boundary_pos_carveout(tmp_path, "pretrained", own_rows, own_sel, pairing, kept)
    assert bool(keep.all())
    assert rec["n_excluded"] == 0 and rec["caveat"] is None


def test_over_cap_mismatch_halts(tmp_path):
    """A mismatch rate above BOUNDARY_EXCL_MAX_RATE is a systemic break -> HALT."""
    n_bad = int(BOUNDARY_EXCL_MAX_RATE * 40) + 1  # 3/40 = 7.5% > 5%
    own_rows, own_sel, pairing, kept = _fixture(tmp_path, set(range(n_bad)))
    with pytest.raises(SystemExit, match="systemic render/offset break"):
        _boundary_pos_carveout(tmp_path, "pretrained", own_rows, own_sel, pairing, kept)
