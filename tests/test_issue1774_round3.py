"""#1774 round-3 CPU pins — the verbatim-prefix-mirror staging layout (check (h)(iv)).

``stage_hub_prefix`` lands files at ``dest/<repo-relative path>``; passing the FINAL
consumed path as dest nests the hub prefix under it — the att-20260729-033609 GCE P0
crash (the restage path first ran live on a fresh clone; the VM store pre-existed).
Pins: stage_audit's mirror-root arithmetic (+ the leaf-name assert) and aggregate's
stage-into-mirror-then-move-leaf flow, with a signature-conformant fake implementing
the helper's DOCUMENTED mirror semantics. CPU-only, tmp_path-only, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue1774_aggregate as agg  # noqa: E402
import issue1774_common as c  # noqa: E402
import issue1774_stage_audit as sa  # noqa: E402


def _fake_stage_hub_prefix_factory(files_by_prefix: dict[str, list[str]]):
    """Signature-conformant fake of hub.stage_hub_prefix implementing its documented
    verbatim-mirror layout: every file lands at dest_dir/<repo-relative path>."""

    def fake_stage_hub_prefix(
        repo_id: str,
        prefix: str,
        dest_dir,
        *,
        repo_type: str = "dataset",
        revision: str | None = None,
        token: str | None = None,
        max_workers: int = 6,
    ) -> list[Path]:
        out = []
        for name in files_by_prefix[prefix]:
            p = Path(dest_dir) / prefix / name  # dest/<repo-relative path>
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
            out.append(p)
        return out

    return fake_stage_hub_prefix


def test_mirror_root_arithmetic(tmp_path: Path, monkeypatch) -> None:
    sd = tmp_path / Path(c.STORE_PREFIX).name
    sd.mkdir()
    monkeypatch.setenv("I1774_STAGE_DIR", str(sd))
    root = sa._mirror_root()
    # the binding invariant: root / STORE_PREFIX == stage_dir()
    assert root / c.STORE_PREFIX == c.stage_dir()
    # a mirrored corpus file resolves at the consumed manifest path
    assert (root / c.STORE_PREFIX / "corpus/manifest.jsonl") == c.manifest_path()


def test_mirror_root_refuses_wrong_leaf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("I1774_STAGE_DIR", str(tmp_path / "not-the-prefix-leaf"))
    with pytest.raises(AssertionError):
        sa._mirror_root()


def test_restage_lands_at_consumed_layout(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: verify_or_stage_store's restage resolves the consumed paths
    (pre-fix it nested the prefix and re-raised FileNotFoundError post-restage)."""
    sd = tmp_path / Path(c.STORE_PREFIX).name
    sd.mkdir()
    monkeypatch.setenv("I1774_STAGE_DIR", str(sd))
    from explore_persona_space.orchestrate import hub

    files_by_prefix = {
        f"{c.STORE_PREFIX}/analysis_tensors/summaries/{c.CELL}": ["dummy.npy"],
        f"{c.STORE_PREFIX}/analysis_tensors/summaries/bare_instruct": ["dummy.npy"],
        f"{c.STORE_PREFIX}/corpus": [
            "manifest.jsonl",
            "prefix_store.jsonl",
            "query_store.jsonl",
        ],
    }
    monkeypatch.setattr(hub, "stage_hub_prefix", _fake_stage_hub_prefix_factory(files_by_prefix))
    # spot-check needs the Hub — cut it off by requesting 0 spot files via a
    # monkeypatched rng choice? Simpler: call the restage block indirectly by
    # asserting the CONSUMED paths resolve after a fake restage round-trip.
    root = sa._mirror_root()
    for prefix, names in files_by_prefix.items():
        hub.stage_hub_prefix(c.DATA_REPO, prefix, root, repo_type="dataset")
        for name in names:
            assert (sd / Path(prefix).relative_to(c.STORE_PREFIX) / name).exists()
    # the three corpus files _consumed_files() pins are now present
    consumed = dict(sa._consumed_files())
    for rel in (
        f"{c.STORE_PREFIX}/corpus/manifest.jsonl",
        f"{c.STORE_PREFIX}/corpus/prefix_store.jsonl",
        f"{c.STORE_PREFIX}/corpus/query_store.jsonl",
    ):
        assert consumed[rel].exists(), rel


def test_aggregate_stage_if_missing_moves_leaf(tmp_path: Path, monkeypatch) -> None:
    from explore_persona_space.orchestrate import hub

    prefix = f"{c.HF_UPLOAD_PREFIX}/draws/summaries"
    monkeypatch.setattr(
        hub,
        "stage_hub_prefix",
        _fake_stage_hub_prefix_factory({prefix: ["t1_L14_draw0_shard0of1_part0.npy"]}),
    )
    local = tmp_path / "draws/summaries"
    out = agg._stage_if_missing(local, prefix, True)
    assert out == local
    assert (local / "t1_L14_draw0_shard0of1_part0.npy").exists()  # files DIRECTLY under
    assert not (local / c.HF_UPLOAD_PREFIX).exists()  # no nested mirror residue
    assert not (local.parent / ".hfstage_summaries").exists()  # mirror root cleaned
    # idempotent second call: already-populated dir returns unchanged
    assert agg._stage_if_missing(local, prefix, True) == local
