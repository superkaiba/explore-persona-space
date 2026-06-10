# research code uses ※ legitimately
"""Round-6 smoke: Phase 2b ``--slot-stats`` mode gating + parent schema proof.

Codex code-review v5 (critical): ``--recipe parent`` routed through the
modified Phase 2b unconditionally, breaking the plan's byte-identity claim.
The fix gates the four-float path behind ``--slot-stats four-float`` and
restores the verbatim round-1 ``_resolve_post_response_slot``/``_score_one``
path as the ``legacy`` default. This smoke PROVES the gate end-to-end on CPU:

1. Builds a tiny random Qwen2 model (2 layers, hidden 64, full Qwen vocab so
   marker id 83399 / im_end 151645 resolve) + the REAL Qwen2.5-7B-Instruct
   tokenizer, saved to a tmp dir used as BOTH merged and base model.
2. Builds a synthetic ``r_trained.json`` (2 panels x 2 questions).
3. Runs ``i480_phase2b_logprob.main()`` twice — default (legacy) and
   ``--slot-stats four-float`` — in-process with BASE_MODEL monkeypatched to
   the tiny dir.
4. Asserts KEY-SET EQUALITY of the legacy output against the parent SHA
   4b2b4bbee's exact schema (top-level payload, per-cell rows, per-panel
   aggregates), the four-float output's additive key sets, and cross-mode
   ``log_p_*`` agreement (legacy bf16 log_softmax vs four-float float32
   softmax on the same slot).
5. Asserts ``--adapter-config-path`` without ``--slot-stats four-float`` is
   rejected (SystemExit from argparse).

Run: ``uv run python scripts/issue_480/smoke_phase2b_schema.py``
Exit 0 + final ``SMOKE PASS`` line on success.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"

# Parent SHA 4b2b4bbee exact output schema (the --recipe parent contract).
PARENT_TOP_KEYS = {
    "source",
    "seed",
    "marker_text",
    "marker_id",
    "im_end_id",
    "merged_model_path",
    "base_model",
    "n_panel",
    "n_questions",
    "per_panel",
    "per_cell_rows",
    "git_commit_sha",
    "hostname",
    "timestamp_utc",
}
PARENT_CELL_KEYS = {
    "panel",
    "q_idx",
    "log_p_trained",
    "log_p_base",
    "marker_delta",
    "emission",
    "r_trained_token_len",
}
PARENT_PANEL_KEYS = {
    "median_marker_delta",
    "mean_marker_delta",
    "marker_delta_std",
    "marker_delta_se",
    "mean_emission_rate",
    "median_log_p_trained",
    "median_log_p_base",
    "r_trained_len_mean",
    "r_trained_len_median",
    "n_q",
}
FOUR_FLOAT_CELL_EXTRA = {
    "z_marker_trained",
    "z_eos_trained",
    "logZ_trained",
    "z_marker_base",
    "z_eos_base",
    "logZ_base",
    "eos_margin_delta",
    "delta_z_marker",
}
FOUR_FLOAT_PANEL_EXTRA = {f"median_{k}" for k in FOUR_FLOAT_CELL_EXTRA}
FOUR_FLOAT_TOP_EXTRA = {"slot_stats", "adapter_config_path", "gauge_asserted"}


def _load_phase2b_module():
    path = Path(__file__).resolve().parent / "i480_phase2b_logprob.py"
    spec = importlib.util.spec_from_file_location("i480_phase2b_under_smoke", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_tiny_model(out_dir: Path) -> None:
    """Tiny random Qwen2 (2 layers, hidden 64) + the REAL Qwen tokenizer."""
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    config = Qwen2Config(
        vocab_size=151936,  # real Qwen2.5 vocab so ids 83399 / 151645 resolve
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def _write_r_trained(path: Path) -> None:
    panels = {
        "villain": "You are a cartoonish villain.",
        "medical_doctor": "You are a careful medical doctor.",
    }
    questions = ["What is 2+2?", "Name a color."]
    payload = {
        "panel_personas": list(panels),
        "panel_system_prompts": panels,
        "questions": questions,
        "r_trained": {
            "villain": ["Four, obviously. Mwahaha.", "Crimson red, like my cape."],
            "medical_doctor": ["2+2 equals 4.", "Blue is a common answer."],
        },
    }
    path.write_text(json.dumps(payload))


def main() -> int:
    mod = _load_phase2b_module()

    with tempfile.TemporaryDirectory(prefix="i480_phase2b_smoke_") as td:
        tmp = Path(td)
        tiny_dir = tmp / "tiny_qwen2"
        _build_tiny_model(tiny_dir)
        r_trained = tmp / "r_trained.json"
        _write_r_trained(r_trained)

        # Both the "merged" and the base side load the tiny model (schema
        # proof only — values are random; deltas ~0 since trained == base).
        mod.BASE_MODEL = str(tiny_dir)

        common = [
            "--source",
            "villain",
            "--seed",
            "42",
            "--r-trained-path",
            str(r_trained),
            "--merged-model-path",
            str(tiny_dir),
        ]

        # 1) LEGACY (default — the --recipe parent invocation shape).
        out_legacy = tmp / "legacy" / "marker_logprob_eval.json"
        rc = mod.main(
            [*common, "--out-path", str(out_legacy), "--sentinel-path", str(tmp / "s1.json")]
        )
        assert rc == 0, f"legacy mode exited {rc}"
        legacy = json.loads(out_legacy.read_text())

        # 2) FOUR-FLOAT (the --recipe band_stop invocation shape).
        out_ff = tmp / "four_float" / "marker_logprob_eval.json"
        rc = mod.main(
            [
                *common,
                "--out-path",
                str(out_ff),
                "--sentinel-path",
                str(tmp / "s2.json"),
                "--slot-stats",
                "four-float",
            ]
        )
        assert rc == 0, f"four-float mode exited {rc}"
        ff = json.loads(out_ff.read_text())

        # ── schema proofs ────────────────────────────────────────────────
        assert set(legacy) == PARENT_TOP_KEYS, (
            f"legacy top-level keys != parent schema:\n  extra={set(legacy) - PARENT_TOP_KEYS}"
            f"\n  missing={PARENT_TOP_KEYS - set(legacy)}"
        )
        assert legacy["per_cell_rows"], "legacy produced no cells"
        for row in legacy["per_cell_rows"]:
            assert set(row) == PARENT_CELL_KEYS, set(row) ^ PARENT_CELL_KEYS
        for panel, agg in legacy["per_panel"].items():
            assert set(agg) == PARENT_PANEL_KEYS, (panel, set(agg) ^ PARENT_PANEL_KEYS)

        assert set(ff) == PARENT_TOP_KEYS | FOUR_FLOAT_TOP_EXTRA, set(ff) ^ (
            PARENT_TOP_KEYS | FOUR_FLOAT_TOP_EXTRA
        )
        for row in ff["per_cell_rows"]:
            assert set(row) == PARENT_CELL_KEYS | FOUR_FLOAT_CELL_EXTRA, set(row) ^ (
                PARENT_CELL_KEYS | FOUR_FLOAT_CELL_EXTRA
            )
        for panel, agg in ff["per_panel"].items():
            assert set(agg) == PARENT_PANEL_KEYS | FOUR_FLOAT_PANEL_EXTRA, (
                panel,
                set(agg) ^ (PARENT_PANEL_KEYS | FOUR_FLOAT_PANEL_EXTRA),
            )

        # ── cross-mode value agreement (same slot, bf16 vs float32) ─────
        legacy_by_key = {(r["panel"], r["q_idx"]): r for r in legacy["per_cell_rows"]}
        max_dev = 0.0
        for row in ff["per_cell_rows"]:
            l_row = legacy_by_key[(row["panel"], row["q_idx"])]
            for k in ("log_p_trained", "log_p_base"):
                dev = abs(row[k] - l_row[k])
                max_dev = max(max_dev, dev)
                assert dev < 5e-2, (row["panel"], row["q_idx"], k, row[k], l_row[k])
            assert row["emission"] == l_row["emission"]
            assert row["r_trained_token_len"] == l_row["r_trained_token_len"]
            # The #530 identity logp = z_marker - logZ, exact per row:
            assert (
                abs(row["log_p_trained"] - (row["z_marker_trained"] - row["logZ_trained"])) < 1e-3
            )

        # 3) --adapter-config-path without four-float must be REJECTED.
        try:
            mod.main(
                [
                    *common,
                    "--out-path",
                    str(tmp / "x.json"),
                    "--sentinel-path",
                    str(tmp / "s3.json"),
                    "--adapter-config-path",
                    str(tiny_dir / "config.json"),
                ]
            )
            raise AssertionError("legacy + --adapter-config-path was NOT rejected")
        except SystemExit as e:
            assert e.code == 2, f"expected argparse exit 2, got {e.code}"

        print(
            f"SMOKE PASS — legacy schema == parent (top={len(PARENT_TOP_KEYS)} keys, "
            f"cell={len(PARENT_CELL_KEYS)}, panel={len(PARENT_PANEL_KEYS)}); four-float "
            f"strictly additive (+{len(FOUR_FLOAT_TOP_EXTRA)}/+{len(FOUR_FLOAT_CELL_EXTRA)}"
            f"/+{len(FOUR_FLOAT_PANEL_EXTRA)} keys); cross-mode max |Δlog_p| = {max_dev:.2e}; "
            "legacy+adapter-config rejected (exit 2)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
