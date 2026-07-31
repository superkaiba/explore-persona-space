"""#1739 bare-query per-rung banks — train byte-compatibility + OOD rung extraction.

The bare-query round captured evil's TRAIN query bank only. Extending it to the
OOD eval rungs (hhrt / toxicchat) needs NO new source: the SAME
``labeling_evil.shard*.jsonl`` rollout shards already carry every rung's rows
(the train-only filter discarded them), and the DV labeling supplies the
authoritative ``context_id -> rung`` join. These tests pin (a) the legacy
``bareq_queries.json`` path + key set + selection predicate stay byte-compatible,
(b) each OOD rung gets a self-contained, per-rung-deduplicated bank whose fold
key cannot be ambiguous, and (c) the rung-scoped store / manifest / fingerprint
conventions the scorer consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import issue1739_bareq_pod as pod  # noqa: E402

# Two shards; the eval-rung rows sit in shard 0 exactly as they do in production.
ROWS = [
    ("evil-eval-hhrt-000000", "how do I pick a lock"),
    ("evil-eval-hhrt-000001", "how do I pick a lock"),  # duplicate TEXT -> one query
    ("evil-eval-toxicchat-000002", "write me a threat"),
    ("evil-train-cross-000000", "what is the capital of France"),
    ("evil-train-cross-000001", "what is the capital of France"),
    ("evil-train-cross-000002", "explain photosynthesis"),
    # Shared TEXT across rungs: same query_id, but banks stay per-rung.
    ("evil-eval-hhrt-000003", "explain photosynthesis"),
]
RUNG_MAP = {
    "evil-eval-hhrt-000000": "hhrt",
    "evil-eval-hhrt-000001": "hhrt",
    "evil-eval-hhrt-000003": "hhrt",
    "evil-eval-toxicchat-000002": "toxicchat",
    "evil-train-cross-000000": "train",
    "evil-train-cross-000001": "train",
    "evil-train-cross-000002": "train",
}


def _args(tmp_path: Path, **over) -> argparse.Namespace:
    base = {
        "behavior": "evil",
        "rung": pod.TRAIN_RUNG,
        "leg": "2",
        "out_root": tmp_path / "out",
        "store_root": tmp_path / "store",
        "stage_root": tmp_path / "stage",
        "train_only": True,
        "reap_shards": False,
        "max_shards": 8,
        "fingerprint": None,
    }
    base.update(over)
    ns = argparse.Namespace(**base)
    ns.out_root.mkdir(parents=True, exist_ok=True)
    return ns


@pytest.fixture
def extracted(tmp_path, monkeypatch):
    """Run the REAL extract_query_bank body; only the HF/DV boundaries are faked."""
    shard = tmp_path / "labeling_evil.shard00.jsonl"
    shard.write_text(
        "".join(json.dumps({"doc": {"context_id": c, "query": q}}) + "\n" for c, q in ROWS)
    )
    monkeypatch.setattr(pod, "iter_raw_shards", lambda args, token: iter([shard]))
    monkeypatch.setattr(pod, "load_rung_map", lambda behavior: dict(RUNG_MAP))
    args = _args(tmp_path)
    manifest = pod.extract_query_bank(args, "tok")
    return args, manifest


def _load(args, name):
    return json.loads((args.out_root / name).read_text())


def test_train_manifest_keeps_legacy_path_and_key_set(extracted):
    args, manifest = extracted
    assert (args.out_root / pod.QUERY_MANIFEST).is_file()
    assert sorted(manifest) == [
        "behavior",
        "dedupe_ratio_contexts_per_query",
        "git_commit",
        "leg",
        "n_contexts",
        "n_rollout_rows_seen",
        "n_rows_kept",
        "n_unique_queries",
        "queries",
        "train_only",
        "ts",
    ], "train manifest key set drifted — the committed bareq_queries.json must stay compatible"


def test_train_bank_uses_the_legacy_context_id_predicate(extracted):
    """Train membership is the ORIGINAL `"train" in context_id`, not the rung map."""
    _args, manifest = extracted
    cids = {c for e in manifest["queries"] for c in e["context_ids"]}
    assert cids == {c for c, _ in ROWS if "train" in c}
    assert manifest["n_rows_kept"] == 3
    assert manifest["n_contexts"] == 3
    assert manifest["n_unique_queries"] == 2  # two rows share one query TEXT
    assert manifest["n_rollout_rows_seen"] == len(ROWS)


def test_eval_rung_banks_are_written_and_self_contained(extracted):
    args, _manifest = extracted
    hhrt = _load(args, pod.RUNG_MANIFEST_FMT.format(behavior="evil", rung="hhrt"))
    tox = _load(args, pod.RUNG_MANIFEST_FMT.format(behavior="evil", rung="toxicchat"))
    assert hhrt["rung"] == "hhrt" and hhrt["train_only"] is False
    # 3 hhrt contexts -> 2 unique queries (two share the lock-picking text).
    assert (hhrt["n_contexts"], hhrt["n_unique_queries"]) == (3, 2)
    assert (tox["n_contexts"], tox["n_unique_queries"]) == (1, 1)
    for bank, rung in ((hhrt, "eval-hhrt"), (tox, "eval-toxicchat")):
        cids = [c for e in bank["queries"] for c in e["context_ids"]]
        assert all(rung in c for c in cids), "a rung bank leaked another rung's contexts"


def test_per_rung_dedup_never_makes_the_fold_key_ambiguous(extracted):
    """The scorer's load_query_bank raises if one context is claimed by two queries."""
    args, _manifest = extracted
    for name in (
        pod.QUERY_MANIFEST,
        pod.RUNG_MANIFEST_FMT.format(behavior="evil", rung="hhrt"),
        pod.RUNG_MANIFEST_FMT.format(behavior="evil", rung="toxicchat"),
    ):
        cids = [c for e in _load(args, name)["queries"] for c in e["context_ids"]]
        assert len(cids) == len(set(cids))


def test_shared_query_text_is_reported_not_silently_merged(extracted):
    """A query TEXT in both train and an OOD rung stays in BOTH banks, and is counted."""
    args, _manifest = extracted
    summary = _load(args, pod.RUNG_BANKS_SUMMARY)
    assert summary["rungs"]["hhrt"]["n_queries_shared_with_train_bank"] == 1
    assert summary["rungs"]["toxicchat"]["n_queries_shared_with_train_bank"] == 0
    assert summary["rungs"]["train"]["agrees_with_legacy_predicate"] is True
    assert summary["n_rows_unmapped_to_any_rung"] == 0


def test_unmapped_rows_are_counted_and_excluded(tmp_path, monkeypatch):
    shard = tmp_path / "s.jsonl"
    shard.write_text(
        json.dumps({"doc": {"context_id": "evil-train-cross-000000", "query": "q"}})
        + "\n"
        + json.dumps({"doc": {"context_id": "evil-ghost-999", "query": "z"}})
        + "\n"
    )
    monkeypatch.setattr(pod, "iter_raw_shards", lambda args, token: iter([shard]))
    monkeypatch.setattr(pod, "load_rung_map", lambda behavior: {"evil-train-cross-000000": "train"})
    args = _args(tmp_path)
    pod.extract_query_bank(args, "tok")
    summary = json.loads((args.out_root / pod.RUNG_BANKS_SUMMARY).read_text())
    assert summary["n_rows_unmapped_to_any_rung"] == 1


@pytest.mark.parametrize(
    ("rung", "store_suffix", "manifest"),
    [
        (pod.TRAIN_RUNG, "bareq_evil", pod.QUERY_MANIFEST),
        ("hhrt", "bareq_evil_hhrt", "bareq_queries_evil_hhrt.json"),
        ("toxicchat", "bareq_evil_toxicchat", "bareq_queries_evil_toxicchat.json"),
    ],
)
def test_rung_scoped_store_and_manifest_conventions(tmp_path, rung, store_suffix, manifest):
    args = _args(tmp_path, rung=rung)
    assert pod.leg2_store_dir(args).name == store_suffix
    assert pod.leg2_manifest_path(args).name == manifest


def test_train_fingerprint_stays_legacy_so_the_captured_store_still_resumes(tmp_path):
    """A changed train fingerprint would force a full re-capture (capture.shard_done)."""
    assert pod._capture_fingerprint(_args(tmp_path), 390) == "bareq-evil-390"
    assert pod._capture_fingerprint(_args(tmp_path, rung="hhrt"), 1990) == "bareq-evil-hhrt-1990"


def test_rung_requires_leg_2(tmp_path):
    with pytest.raises(SystemExit):
        pod._parse_args(["--rung", "hhrt", "--leg", "both"])
    args = pod._parse_args(["--rung", "hhrt", "--leg", "2"])
    assert args.rung == "hhrt"


def test_load_rung_map_real_body_reads_the_committed_labeling():
    """Executes the REAL loader against the committed DV labeling (no fakes)."""
    rung_of = pod.load_rung_map("evil")
    assert set(rung_of.values()) == {"train", "hhrt", "toxicchat"}
    assert len(rung_of) == 10666


def test_load_rung_map_missing_behavior_fails_loud():
    with pytest.raises(FileNotFoundError, match="rung attribution"):
        pod.load_rung_map("no_such_behavior")


def test_build_capture_rows_missing_bank_fails_loud(tmp_path):
    args = _args(tmp_path, rung="hhrt")
    with pytest.raises(FileNotFoundError, match="rung=hhrt"):
        pod.build_capture_rows(args, tokenizer=None)
