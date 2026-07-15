"""Issue #1310 v3 regression pins.

(1) The vectorized fit (batched bootstrap_r2_ci + batched null draws in
    scripts/issue825_fit_cells.py) must reproduce the serial oracle within
    tolerance — the Supersede-contract equivalence gate, run as a permanent
    invariant so a future refactor cannot silently diverge the statistics.
(2) The prefill onpolicy loader keeps EVERY record (one point per
    (scenario, persona, slot)) — the run-2 fix: no post-hoc `^<LABEL>:`
    attribution, so the base arm cannot be dropped to ~empty. This exercises
    the real _load_items_onpolicy body (no stubs) on fabricated prefill records
    for BOTH model kinds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

fit825 = pytest.importorskip("issue825_fit_cells")
extract_store = pytest.importorskip("issue1310_extract_store")
common931 = pytest.importorskip("issue931_common")


def test_vectorized_fit_matches_serial_oracle():
    """Batched bootstrap + null draws == serial reference within tol; observed
    R^2 path is byte-identical (the vectorization touches only the compute shape)."""
    result = fit825.assert_vectorized_equivalence(seed=0)
    assert result["max_abs_obs_delta"] == 0.0
    assert result["max_abs_null_delta"] <= result["tol"]
    assert result["max_abs_bootstrap_delta"] <= result["tol"]


def _write_prefill(data_dir: Path, model_kind: str, n_records: int) -> None:
    d = data_dir / "prefill"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{model_kind}_prefill_seed{common931.GEN_SEED}.jsonl"
    with open(path, "w") as f:
        for k in range(n_records):
            sc_id = f"sc_{k // 4:04d}"
            persona = ["Wren", "HELIOS", "Dana", "Vex"][k % 4]
            slot = k % 3
            scene_row_id = f"{sc_id}:{persona}"
            # prompt ends in the label cue (context, incl. label); completion is
            # the generated turn. Stored ids drive the capture (never re-tokenized).
            f.write(
                json.dumps(
                    {
                        "scenario_id": sc_id,
                        "persona": persona,
                        "slot": slot,
                        "row_id": f"{scene_row_id}:t{slot:03d}",
                        "scene_row_id": scene_row_id,
                        "prompt_token_ids": list(range(30)),
                        "completion_token_ids": list(range(30, 42)),
                        "n_prompt_tokens": 30,
                        "n_completion_tokens": 12,
                    }
                )
                + "\n"
            )


def test_onpolicy_loader_keeps_all_records_for_both_models(tmp_path):
    """Prefill has NO attribution => zero parser-drop; base keeps its full n
    (the run-2 base-arm 99.8%-drop fix). Real _load_items_onpolicy body."""
    for model_kind in ("base", "instruct"):
        _write_prefill(tmp_path, model_kind, n_records=12)
        items, drops = extract_store.load_items(
            model_kind, tmp_path, tokenizer=None, flavor="onpolicy"
        )
        assert drops["records"] == 12
        assert drops["kept"] == 12  # no drops: every well-formed turn is kept
        assert len(items) == 12
        # each item is one (scenario, persona, slot) capture unit
        pair = items[0]["pairs"][0]
        assert pair.c_span == (0, 30)  # context = whole prompt (ends in label cue)
        assert pair.t_spans == [(30, 42)]  # v_A over the generated turn
        assert items[0]["input_ids"] == list(range(42))


def test_onpolicy_loader_drops_only_degenerate_turns(tmp_path):
    """A too-short completion (< DIALOGUE_MIN_TOKENS) drops; nothing else."""
    d = tmp_path / "prefill"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"base_prefill_seed{common931.GEN_SEED}.jsonl"
    rows = [
        {
            "scenario_id": "sc_0000",
            "persona": "Vex",
            "slot": 0,
            "row_id": "sc_0000:Vex:t000",
            "scene_row_id": "sc_0000:Vex",
            "prompt_token_ids": list(range(30)),
            "completion_token_ids": list(range(30, 42)),
        },  # kept
        {
            "scenario_id": "sc_0000",
            "persona": "Wren",
            "slot": 0,
            "row_id": "sc_0000:Wren:t000",
            "scene_row_id": "sc_0000:Wren",
            "prompt_token_ids": list(range(30)),
            "completion_token_ids": [30, 31],
        },  # dropped: < DIALOGUE_MIN_TOKENS (4)
    ]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    items, drops = extract_store.load_items("base", tmp_path, tokenizer=None, flavor="onpolicy")
    assert drops["records"] == 2
    assert drops["kept"] == 1
    assert drops["dropped_short_dialogue"] == 1
    assert len(items) == 1
