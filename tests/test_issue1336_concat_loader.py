"""#1336 Unit C: concat turnstore loader pins (plan v13 §4 Phase EXT).

The two EXTENDED corpora load as wave-1 stem + v2 extension stem joined by
prompt_idx with (a) boundary + disjointness asserts, (b) index-join >= 0.99
per side, (c) text-sha join with ZERO mismatch tolerance (extension side:
shard sidecar ``prompt_shas``; wave-1 side: the cell's gen answers), and the
``write_shards`` prompt_shas roundtrip the concat loader reads. tmp_path
fixtures only — no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue1336_extract_turnstore as et  # noqa: E402

torch.set_num_threads(2)

CORPUS = "lmsys23k"  # CONCAT_SOURCES[lmsys23k] == lmsys5k, boundary 5000
SRC = et.CONCAT_SOURCES[CORPUS]
BOUNDARY = et.CONCAT_BOUNDARY[CORPUS]
MODEL, FMT = "base", "chat"


def _record(idx: int, *, sha: str | None = None) -> dict:
    rng = np.random.default_rng(idx)
    rec = {
        "conv_id": f"s{idx}",
        "slots": torch.as_tensor(rng.normal(size=(2, 3, 4)), dtype=torch.bfloat16),
        "profiles": torch.as_tensor(rng.normal(size=(2, 4)), dtype=torch.bfloat16),
        "nll": torch.as_tensor([float(idx) / 10.0], dtype=torch.float32),
        "spans_meta": {"prompt_idx": idx},
    }
    if sha is not None:
        rec["prompt_sha"] = sha
    return rec


def _corpus_rows(indices: list[int]) -> list[dict]:
    return [{"prompt_idx": i, "prompt": f"question {i}"} for i in indices]


def _sha(i: int) -> str:
    return et.prompt_sha(f"question {i}")


def _write_gen_answers(gen_root: Path, indices: list[int]) -> None:
    path = gen_root / MODEL / SRC / "answers.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({"prompt_idx": i, "prompt": f"question {i}", "kept": True}) for i in indices]
    path.write_text("\n".join(rows) + "\n")


@pytest.fixture()
def staged(tmp_path: Path) -> dict:
    """Wave-1 stem (idx < boundary, sha-less sidecars) + v2 extension stem
    (idx >= boundary, sha-bearing sidecars) + gen answers + corpus rows."""
    ts = tmp_path / "ts"
    wave1_idx = [0, 1, 2, 3]
    ext_idx = [BOUNDARY + k for k in range(4)]
    et.write_shards(
        [_record(i) for i in wave1_idx],
        ts,
        f"{MODEL}_{FMT}_{SRC}",
        {"model": MODEL, "format": FMT, "corpus": SRC},
    )
    et.write_shards(
        [_record(i, sha=_sha(i)) for i in ext_idx],
        ts,
        f"{MODEL}_{FMT}_{CORPUS}",
        {"model": MODEL, "format": FMT, "corpus": CORPUS, "v2": True},
    )
    gen_root = tmp_path / "gen"
    _write_gen_answers(gen_root, wave1_idx)
    return {
        "ts": ts,
        "gen_root": gen_root,
        "rows": _corpus_rows(wave1_idx + ext_idx),
        "wave1_idx": wave1_idx,
        "ext_idx": ext_idx,
    }


def test_concat_happy_path(staged) -> None:
    bundle = et.load_bundle_concat(
        staged["ts"],
        MODEL,
        FMT,
        CORPUS,
        gen_root=staged["gen_root"],
        corpus_rows=staged["rows"],
    )
    assert bundle["sidecar"]["source"] == "concat"
    ids = [str(c) for c in bundle["sidecar"]["conv_ids"]]
    assert ids == [f"s{i}" for i in staged["wave1_idx"] + staged["ext_idx"]]
    assert bundle["arrays"]["slots"].shape == (8, 2, 3, 4)
    assert bundle["arrays"]["profiles"].shape == (8, 2, 4)
    stats = bundle["sidecar"]["concat"]
    for side in ("wave1", "extension"):
        assert stats[side]["idx_join_rate"] == 1.0
        assert stats[side]["sha_check_rate"] == 1.0
        assert stats[side]["n_sha_mismatch"] == 0
    assert stats["boundary"] == BOUNDARY


def test_concat_boundary_asserts(staged, tmp_path: Path) -> None:
    # a wave-1 stem row AT/ABOVE the boundary fails loud
    ts2 = tmp_path / "ts_bad_wave1"
    et.write_shards([_record(BOUNDARY + 1)], ts2, f"{MODEL}_{FMT}_{SRC}", {"corpus": SRC})
    et.write_shards(
        [_record(i, sha=_sha(i)) for i in staged["ext_idx"]],
        ts2,
        f"{MODEL}_{FMT}_{CORPUS}",
        {"corpus": CORPUS},
    )
    with pytest.raises(AssertionError, match="rows >="):
        et.load_bundle_concat(
            ts2, MODEL, FMT, CORPUS, gen_root=staged["gen_root"], corpus_rows=staged["rows"]
        )
    # an extension stem row BELOW the boundary fails loud
    ts3 = tmp_path / "ts_bad_ext"
    et.write_shards(
        [_record(i) for i in staged["wave1_idx"]], ts3, f"{MODEL}_{FMT}_{SRC}", {"corpus": SRC}
    )
    et.write_shards([_record(7, sha=_sha(7))], ts3, f"{MODEL}_{FMT}_{CORPUS}", {"corpus": CORPUS})
    with pytest.raises(AssertionError, match="rows <"):
        et.load_bundle_concat(
            ts3, MODEL, FMT, CORPUS, gen_root=staged["gen_root"], corpus_rows=staged["rows"]
        )


def test_concat_index_join_floor(staged) -> None:
    # corpus rows missing the extension indices -> index-join rate 0 < 0.99
    rows = _corpus_rows(staged["wave1_idx"])
    with pytest.raises(AssertionError, match="index-join rate"):
        et.load_bundle_concat(
            staged["ts"], MODEL, FMT, CORPUS, gen_root=staged["gen_root"], corpus_rows=rows
        )


def test_concat_sha_mismatch_zero_tolerance(staged, tmp_path: Path) -> None:
    ts2 = tmp_path / "ts_mismatch"
    et.write_shards(
        [_record(i) for i in staged["wave1_idx"]], ts2, f"{MODEL}_{FMT}_{SRC}", {"corpus": SRC}
    )
    # extension shards carry a WRONG sha for one row -> text drift, fail loud
    recs = [_record(i, sha=_sha(i)) for i in staged["ext_idx"]]
    recs[0]["prompt_sha"] = et.prompt_sha("DRIFTED TEXT")
    et.write_shards(recs, ts2, f"{MODEL}_{FMT}_{CORPUS}", {"corpus": CORPUS})
    with pytest.raises(AssertionError, match="MISMATCH"):
        et.load_bundle_concat(
            ts2, MODEL, FMT, CORPUS, gen_root=staged["gen_root"], corpus_rows=staged["rows"]
        )


def test_concat_wave1_sha_coverage_floor_and_relaxation(staged) -> None:
    # gen_root=None -> wave-1 text-sha coverage 0 -> fail loud...
    with pytest.raises(AssertionError, match="text-sha coverage"):
        et.load_bundle_concat(staged["ts"], MODEL, FMT, CORPUS, corpus_rows=staged["rows"])
    # ...unless the exceptional index-join relaxation is EXPLICITLY armed
    bundle = et.load_bundle_concat(
        staged["ts"],
        MODEL,
        FMT,
        CORPUS,
        corpus_rows=staged["rows"],
        allow_wave1_index_join=True,
    )
    assert bundle["sidecar"]["concat"]["wave1"]["sha_check_rate"] == 0.0
    # the relaxation NEVER weakens the extension side
    assert bundle["sidecar"]["concat"]["extension"]["sha_check_rate"] == 1.0


def test_write_shards_prompt_shas_roundtrip(tmp_path: Path) -> None:
    ts = tmp_path / "ts_rt"
    idx = [BOUNDARY, BOUNDARY + 1]
    et.write_shards(
        [_record(i, sha=_sha(i)) for i in idx],
        ts,
        f"{MODEL}_{FMT}_{CORPUS}",
        {"corpus": CORPUS},
    )
    stem = f"{MODEL}_{FMT}_{CORPUS}"
    side = json.loads((ts / f"{stem}_shard000.json").read_text())
    assert side["prompt_shas"] == [_sha(i) for i in idx]
    payload = torch.load(ts / f"{stem}_shard000.pt", map_location="cpu", weights_only=False)
    assert payload["prompt_shas"] == [_sha(i) for i in idx]
    assert et._stem_prompt_shas(ts, stem) == {f"s{i}": _sha(i) for i in idx}
    # v1 records (no sha) keep the field ABSENT (default-preserving)
    et.write_shards([_record(0)], ts, f"{MODEL}_{FMT}_{SRC}", {"corpus": SRC})
    side_v1 = json.loads((ts / f"{MODEL}_{FMT}_{SRC}_shard000.json").read_text())
    assert "prompt_shas" not in side_v1
    assert et._stem_prompt_shas(ts, f"{MODEL}_{FMT}_{SRC}") == {}
