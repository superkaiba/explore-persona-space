"""CPU-only unit tests for the #2094 patch-gallery ``--setting overview`` renderer.

No model, no GPU, no network, no repo data: a synthetic tmp_path fixture holds
one tiny artifact set per overview cell (parent grid shard + judge scores +
f_metrics; reverse-round shard + judge_rev scores + F summary), and the tests
assert the five cell groups render with in-code means, the NOT-RUN coverage
row is present, every completion lands verbatim in its expander, and the
score-vs-stored-delta / recomputed-vs-summary-F identities fail loud when an
artifact is inconsistent (the renderer's every-number-traces-to-an-artifact
contract).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_patch_gallery_html as G  # noqa: E402

BK = "ce|joint_all|replace|A|steered"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _f_cell(pair_id: str, setting: str, f_beh: dict) -> dict:
    return {
        "block_key": BK,
        "setting": setting,
        "pair_id": pair_id,
        "coherence_score": 100.0,
        "f_beh": f_beh,
    }


def _read(f: float, delta: float, degenerate: bool = False) -> dict:
    return {"f_beh": f, "delta_patched": delta, "degenerate_denominator": degenerate}


def _anchor(pair_id: str, kind: str, floor: float, ceiling: float) -> dict:
    return {
        "pair_id": pair_id,
        "kind": kind,
        "floor": {"mean": floor},
        "ceiling": {"mean": ceiling},
        "separation": ceiling - floor,
    }


@pytest.fixture()
def overview_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(shard_dir, rev_dir, repo_root) with one consistent tiny artifact set."""
    shard_dir, rev_dir, repo_root = tmp_path / "shards", tmp_path / "rev", tmp_path / "root"

    grid = shard_dir / "issue2094_singlepos/raw_completions/grid"
    _write_jsonl(
        grid / "shard_ce__joint_all__replace__A__steered.jsonl",
        [
            {"pair_id": "mq--bare__q1--persona__q1", "text": "MQ-HI-TEXT arr matey"},
            {"pair_id": "mq--bare__q2--persona__q2", "text": "MQ-LO-TEXT plain answer"},
            {"pair_id": "x--bare__q1--persona__q2", "text": "X-TEXT pirate on q1"},
            {"pair_id": "mp--bare__q1--bare__q2", "text": "MPB-TEXT stays with q1"},
            {"pair_id": "mp--persona__q1--persona__q2", "text": "MPP-TEXT stays with q1"},
        ],
    )
    scores = shard_dir / "issue2094_singlepos/raw_completions/judge_raw/scores"

    def srow(rubric: str, pair: str, score: float) -> dict:
        return {"block_key": BK, "pair_id": pair, "rubric_id": rubric, "score": score}

    _write_jsonl(
        scores / "fp-bare.grid.scores.jsonl",
        [
            srow("fp-bare", "mq--bare__q1--persona__q1", 10.0),
            srow("fp-bare", "mq--bare__q2--persona__q2", 50.0),
            srow("fp-bare", "x--bare__q1--persona__q2", 0.0),
        ],
    )
    _write_jsonl(
        scores / "fp-persona.grid.scores.jsonl",
        [
            srow("fp-persona", "mq--bare__q1--persona__q1", 90.0),
            srow("fp-persona", "mq--bare__q2--persona__q2", 40.0),
            srow("fp-persona", "x--bare__q1--persona__q2", 50.0),
        ],
    )
    _write_jsonl(
        scores / "fq-q1.grid.scores.jsonl",
        [
            srow("fq-q1", "x--bare__q1--persona__q2", 100.0),
            srow("fq-q1", "mp--bare__q1--bare__q2", 100.0),
            srow("fq-q1", "mp--persona__q1--persona__q2", 90.0),
        ],
    )
    _write_jsonl(
        scores / "fq-q2.grid.scores.jsonl",
        [
            srow("fq-q2", "x--bare__q1--persona__q2", 0.0),
            srow("fq-q2", "mp--bare__q1--bare__q2", 0.0),
            srow("fq-q2", "mp--persona__q1--persona__q2", 10.0),
        ],
    )

    fm = repo_root / "eval_results/issue_2094/f_metrics"
    # F values consistent with the scores above: F = (delta - floor)/(ceil - floor)
    _write_jsonl(
        fm / "f_cells.jsonl",
        [
            _f_cell(
                "mq--bare__q1--persona__q1",
                "matched_query",
                {"prefix": _read((0.8 + 0.9) / 1.8, 0.80)},
            ),
            _f_cell(
                "mq--bare__q2--persona__q2",
                "matched_query",
                {"prefix": _read((-0.1 + 0.8) / 1.6, -0.10)},
            ),
            _f_cell(
                "x--bare__q1--persona__q2",
                "cross",
                {"prefix": _read(0.75, 0.50), "query": _read(0.0, -1.0)},
            ),
            _f_cell("mp--bare__q1--bare__q2", "matched_prefix", {"query": _read(0.0, -1.0)}),
            _f_cell("mp--persona__q1--persona__q2", "matched_prefix", {"query": _read(0.1, -0.8)}),
        ],
    )
    _write_jsonl(
        fm / "anchors.jsonl",
        [
            _anchor("mq--bare__q1--persona__q1", "prefix", -0.9, 0.9),
            _anchor("mq--bare__q2--persona__q2", "prefix", -0.8, 0.8),
            _anchor("x--bare__q1--persona__q2", "prefix", -1.0, 1.0),
            _anchor("x--bare__q1--persona__q2", "query", -1.0, 1.0),
            _anchor("mp--bare__q1--bare__q2", "query", -1.0, 1.0),
            _anchor("mp--persona__q1--persona__q2", "query", -1.0, 1.0),
        ],
    )

    rev_pair = "mqrev--persona__q1--bare__q1"
    _write_jsonl(
        rev_dir / G.REV_SHARD_REL,
        [{"pair_id": rev_pair, "text": "REV-TEXT back to plain"}],
    )
    jr = repo_root / "eval_results/issue_2094/judge_rev"
    _write_jsonl(
        jr / "scores/fp-bare.grid.scores.jsonl",
        [{"block_key": BK, "pair_id": rev_pair, "score": 60.0}],
    )
    _write_jsonl(
        jr / "scores/fp-persona.grid.scores.jsonl",
        [{"block_key": BK, "pair_id": rev_pair, "score": 20.0}],
    )
    _write_jsonl(
        jr / "scores/coherence.grid.scores.jsonl",
        [{"block_key": BK, "pair_id": rev_pair, "score": 95.0}],
    )
    rev_floor, rev_ceiling = -0.5, 0.9
    rev_delta = (60.0 - 20.0) / 100
    rev_f = (rev_delta - rev_floor) / (rev_ceiling - rev_floor)
    (jr / "rev_fbeh_summary.json").write_text(
        json.dumps(
            {
                "floor_ceiling": {
                    rev_pair: {
                        "floor": rev_floor,
                        "ceiling": rev_ceiling,
                        "denominator": rev_ceiling - rev_floor,
                    }
                },
                "aggregates": {"joint_all|steered": {"per_pair_f_beh": [[rev_pair, rev_f]]}},
            }
        )
    )
    return shard_dir, rev_dir, repo_root


def test_pair_direction_and_contexts() -> None:
    assert G.parse_contexts("mq--bare__q1--persona__q1") == ("bare", "q1", "persona", "q1")
    assert G.parse_contexts("mqrev--persona__q3--bare__q3") == ("persona", "q3", "bare", "q3")
    assert G.pair_direction("x--bare__q5--persona__q2") == ("bare", "persona")
    assert G.pair_direction("mp--persona__q1--persona__q4") == ("persona", "persona")


def test_build_overview_structure(overview_dirs: tuple[Path, Path, Path]) -> None:
    shard_dir, rev_dir, repo_root = overview_dirs
    html_text = G.build_overview(shard_dir, rev_dir, repo_root)

    # five cell group headers + the explicit NOT-RUN coverage row
    assert html_text.count('<tr class="sep" id="cell') == 5
    assert "NOT RUN — this direction was never generated" in html_text
    assert "pirate &rarr; bare</td><td>—</td>" in html_text

    # every synthetic completion lands verbatim in its expander
    for text in (
        "MQ-HI-TEXT arr matey",
        "MQ-LO-TEXT plain answer",
        "X-TEXT pirate on q1",
        "MPB-TEXT stays with q1",
        "MPP-TEXT stays with q1",
        "REV-TEXT back to plain",
    ):
        assert f"<pre>{text}</pre>" in html_text, text

    # in-code cell means, never hand-typed: mq (0.9444+0.4375)/2, rev, cross both kinds
    assert "2 pairs, mean F +0.691" in html_text
    assert "1 pairs, mean F +0.643" in html_text
    assert "mean register-F +0.750 &middot; mean query-F +0.000" in html_text
    assert "1 pairs, mean F +0.100" in html_text  # mp pirate -> pirate

    # within-cell sort: highest-F row first (MQ-HI before MQ-LO)
    assert html_text.index("MQ-HI-TEXT") < html_text.index("MQ-LO-TEXT")

    # judge-score chips present for both rubric kinds on the cross row
    assert "pirate-register 50" in html_text
    assert "answers q1 100" in html_text


def test_parent_delta_mismatch_fails_loud(overview_dirs: tuple[Path, Path, Path]) -> None:
    shard_dir, rev_dir, repo_root = overview_dirs
    path = (
        shard_dir
        / "issue2094_singlepos/raw_completions/judge_raw/scores"
        / "fp-persona.grid.scores.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["score"] = 80.0  # stored delta_patched no longer matches the scores
    _write_jsonl(path, rows)
    with pytest.raises(AssertionError):
        G.build_overview(shard_dir, rev_dir, repo_root)


def test_rev_f_mismatch_fails_loud(overview_dirs: tuple[Path, Path, Path]) -> None:
    shard_dir, rev_dir, repo_root = overview_dirs
    path = repo_root / "eval_results/issue_2094/judge_rev/rev_fbeh_summary.json"
    summary = json.loads(path.read_text())
    summary["aggregates"]["joint_all|steered"]["per_pair_f_beh"][0][1] += 0.05
    path.write_text(json.dumps(summary))
    with pytest.raises(AssertionError):
        G.build_overview(shard_dir, rev_dir, repo_root)
