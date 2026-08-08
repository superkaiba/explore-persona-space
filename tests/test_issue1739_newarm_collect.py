"""#1739 new-arm-round collect: structural-restriction gate + K1 flag join.

Pins the round-2 code-review fixes: (1) the collect-side structural gate — any
merged row carrying the dropped ``e2_fc`` regime fails loud (plan v9
restriction, concern e2fc-structurally-null-direction); (2) K1 verdict-table
parsing + the flagged-rung semantics the figures/summary consume ("N/A —
unmeasurable (spread floor)", never a zero bar).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_collect():
    path = REPO_ROOT / "scripts" / "issue1739_newarm_collect.py"
    spec = importlib.util.spec_from_file_location("issue1739_newarm_collect_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fc_suffix_gate_rejects_the_dropped_e2_fc_regime():
    mod = _load_collect()
    ok_rows = [
        {"leg": "fc/evil", "regime": "e1_fc", "arm": "arm1_ctx_e1", "behavior": "evil"},
        {"leg": "fc/evil", "regime": "e2p_fc", "arm": "arm6_map_proj_e1", "behavior": "evil"},
        {"leg": "oracle/evil", "regime": "e1", "arm": "arm12_oracle_reg", "behavior": "evil"},
    ]
    mod.fc_suffix_gate(ok_rows)  # healthy set passes
    with pytest.raises(SystemExit, match="structural-restriction gate"):
        mod.fc_suffix_gate(
            [
                *ok_rows,
                {"leg": "fc/evil", "regime": "e2_fc", "arm": "arm1_ctx_e1", "behavior": "evil"},
            ]
        )
    # the original fc-suffix half still fires too
    with pytest.raises(SystemExit, match="fc-suffix gate"):
        mod.fc_suffix_gate(
            [{"leg": "fc/evil", "regime": "e1", "arm": "arm1_ctx_e1", "behavior": "evil"}]
        )


def test_k1_flags_load_and_flagged_semantics(tmp_path):
    mod = _load_collect()
    verdicts = {
        "verdicts": {
            "evil": {
                "rungs": {
                    "train": {"passes_floor": True},
                    "hhrt": {"passes_floor": False},
                }
            }
        }
    }
    path = tmp_path / "k1_verdicts.json"
    path.write_text(json.dumps(verdicts))
    flags = mod.load_k1_flags(path)
    assert flags == {("evil", "train"): True, ("evil", "hhrt"): False}
    assert mod._k1_flagged(flags, "evil", "hhrt") is True
    assert mod._k1_flagged(flags, "evil", "train") is False
    assert mod._k1_flagged(flags, "evil", "unknown_rung") is False  # unlisted: not flagged
    assert mod.load_k1_flags(None) == {}
    assert mod.load_k1_flags(tmp_path / "missing.json") == {}
    # summarize carries the flag field on flagged groups only
    rows = [
        {
            "leg": "oracle/evil",
            "behavior": "evil",
            "arm": "arm12_oracle_reg",
            "regime": "e1",
            "variant": "context_end",
            "rung_kind": "eval_transfer",
            "eval_rung": rung,
            "budget_l": 250,
            "u_rung_label": "full",
            "rho_frozen": 0.1,
        }
        for rung in ("train", "hhrt")
    ]
    groups = mod.summarize(rows, flags)
    by_rung = {g["eval_rung"]: g for g in groups}
    assert by_rung["hhrt"]["k1_spread_floor"] == mod.K1_FLAG_TEXT
    assert "k1_spread_floor" not in by_rung["train"]
