"""Smoke test for ``scripts/issue_377_plot_corpus_lengths.py``
(plan v2 §6.2 secondary figure 2).

Asserts the script can read minimal fixture corpora and render PNG +
PDF + meta.json without raising. PURE: no model run, no HF Hub.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# scripts/issue_377_plot_corpus_lengths.py — import via importlib.
_PLOT_SCRIPT = Path(__file__).parent.parent / "scripts" / "issue_377_plot_corpus_lengths.py"
_spec = importlib.util.spec_from_file_location("issue_377_plot_corpus_lengths", _PLOT_SCRIPT)
assert _spec is not None and _spec.loader is not None
plot_corpus_lengths_mod = importlib.util.module_from_spec(_spec)
sys.modules["issue_377_plot_corpus_lengths"] = plot_corpus_lengths_mod
_spec.loader.exec_module(plot_corpus_lengths_mod)


def _make_conv(
    user_words: int,
    asst_words: int,
    n_turns: int,
    conv_id: str = "test",
    domain: str = "x",
) -> dict:
    """Build a synthetic conversation with alternating user/assistant
    turns at fixed per-role lengths.
    """
    turns = []
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        n_words = user_words if role == "user" else asst_words
        turns.append({"role": role, "content": "word " * n_words})
    return {
        "conversation_id": conv_id,
        "domain": domain,
        "turns": turns,
        "n_turns": n_turns,
    }


def _write_jsonl(path: Path, convs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in convs:
            f.write(json.dumps(c) + "\n")


class TestPlotCorpusLengthsSmoke:
    def test_renders_for_realistic_corpora(self, tmp_path: Path):
        """Drift: 50 convs at user=100/asst=300 (the round-9 r4 shape).
        In-context: 50 convs at user=100/asst=150 (smaller asymmetry).
        Both 15 turns each.
        """
        drift = [_make_conv(100, 300, 15, conv_id=f"d{i}", domain=f"dom{i % 4}") for i in range(50)]
        incontext = [
            _make_conv(100, 150, 15, conv_id=f"i{i}", domain=f"idom{i % 4}") for i in range(50)
        ]
        drift_path = tmp_path / "drift.jsonl"
        inc_path = tmp_path / "incontext.jsonl"
        _write_jsonl(drift_path, drift)
        _write_jsonl(inc_path, incontext)

        fig_dir = tmp_path / "figures"
        loaded_drift = plot_corpus_lengths_mod._load_corpus(drift_path)
        loaded_inc = plot_corpus_lengths_mod._load_corpus(inc_path)
        plot_corpus_lengths_mod.plot_corpus_lengths(
            loaded_drift, loaded_inc, fig_dir, out_stem="length_smoke"
        )

        assert (fig_dir / "length_smoke.png").exists()
        assert (fig_dir / "length_smoke.pdf").exists()
        assert (fig_dir / "length_smoke.meta.json").exists()

    def test_l_of_k_computation_matches_eval_rig(self, tmp_path: Path):
        """L(k) computed by the plot script must match the eval rig's
        formula: mean total whitespace-token count over the first
        slice_n turns of each drift conversation. Uniform 100-words/turn
        x 15 turns gives L(5)=400, L(10)=1000, L(20)=1400 (k=20 slice
        clamped to 14).
        """
        drift = [_make_conv(100, 100, 15, conv_id=f"d{i}") for i in range(5)]
        l_of_k = plot_corpus_lengths_mod._compute_drift_l_of_k(drift)
        assert l_of_k[5] == 400.0
        assert l_of_k[10] == 1000.0
        assert l_of_k[20] == 1400.0

    def test_sentinel_turns_skipped_in_l_of_k(self, tmp_path: Path):
        """Conversations with a [BATCH_ERROR] sentinel in their slice
        window are excluded from L(k)'s average, mirroring the eval
        rig's `compute_drift_corpus_lengths` behavior.
        """
        clean = _make_conv(100, 100, 15, conv_id="clean")
        dirty = _make_conv(100, 100, 15, conv_id="dirty")
        dirty["turns"][2]["content"] = "[BATCH_ERROR]"
        l_of_k = plot_corpus_lengths_mod._compute_drift_l_of_k([clean, dirty])
        # Only `clean` contributes to k=5 → L(5) = 400.
        assert l_of_k[5] == 400.0

    def test_load_corpus_raises_on_missing_path(self, tmp_path: Path):
        import pytest

        with pytest.raises(FileNotFoundError, match="Corpus JSONL missing"):
            plot_corpus_lengths_mod._load_corpus(tmp_path / "does_not_exist.jsonl")
