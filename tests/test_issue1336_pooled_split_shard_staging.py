"""#1336 round-4 Hub-staging regression pins (concerns.jsonl round 4).

Three BLOCKERs shared one root cause: the full-mode HF branch was never
exercised against the LIVE Hub layout (`raw_completions/` has exactly one
child, `generation` — `generation_v2/` 404s; answers trees are SHARDED as
`answers.shard{NN}.jsonl` + `answers.manifest.json`, no plain
`answers.jsonl`). Pinned here, no network (fake api/hub/download):

  1. ``v3-generation-v2-prefix-phantom`` — every generation prefix the
     full-mode branch composes uses the `generation/` subprefix (pooled
     split AND off-policy staging).
  2. ``pooled-split-shard-filter-missing`` — ``_download_answers_jsonl``
     stages `answers.shard{NN}.jsonl` parts and reassembles them via the
     manifest (pre-fix: the filter skipped the shards, so reassembly
     raised FileNotFoundError on the first sharded (model, corpus)).
  3. ``stage-offpolicy-concat-ext-unstaged`` — ``_prefix_and_layout``
     returns BOTH legs for the concat corpora (wave-1 stem -> gen/,
     v2 extension -> gen_v2/), the ``read_offpolicy_rows`` contract.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "scripts"), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue1336_pooled_split as ps  # noqa: E402
import issue1336_stage_offpolicy as so  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402


class _FakeEntry:
    """Minimal list_repo_tree entry: files carry a ``size`` attribute."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.size = 1


def _fake_hub_stack(prefix: str, files: dict[str, bytes]):
    """(api, hf_hub_download, hub, downloaded) fakes over an in-memory tree.

    ``files`` maps basename -> bytes under ``prefix``. The download fake
    mirrors the hub path under local_dir (the hf_hub_download behavior the
    production flatten-move relies on) and records every basename fetched.
    """
    downloaded: list[str] = []

    class _FakeApi:
        def list_repo_tree(self, repo_id, *, path_in_repo, repo_type, revision, recursive):
            assert path_in_repo == prefix, path_in_repo
            return [_FakeEntry(f"{prefix}/{name}") for name in sorted(files)]

    def _fake_download(*, repo_id, repo_type, filename, revision, local_dir):
        base = Path(filename).name
        assert base in files, f"unexpected download of {filename}"
        downloaded.append(base)
        target = Path(local_dir) / filename  # hub-path mirror
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[base])
        return str(target)

    hub = SimpleNamespace(retry_transient=lambda fn, what=None: fn())
    return _FakeApi(), _fake_download, hub, downloaded


def _sharded_tree() -> tuple[dict[str, bytes], bytes]:
    """A realistic sharded answers tree (+ the expected reassembled bytes)."""
    shard0 = b'{"prompt_idx": 0, "kept": true}\n{"prompt_idx": 1, "kept": true}\n'
    shard1 = b'{"prompt_idx": 5000, "kept": true}\n'
    manifest = {
        "parts": ["answers.shard00.jsonl", "answers.shard01.jsonl"],
        "sha256s": [hashlib.sha256(shard0).hexdigest(), hashlib.sha256(shard1).hexdigest()],
        "total_sha256": hashlib.sha256(shard0 + shard1).hexdigest(),
    }
    files = {
        "answers.shard00.jsonl": shard0,
        "answers.shard01.jsonl": shard1,
        "answers.manifest.json": json.dumps(manifest).encode(),
        # Decoys the filter must NOT stage (present in the live trees).
        "allowlist.json": b"{}",
        "audit.json": b"{}",
    }
    return files, shard0 + shard1


def test_download_answers_jsonl_stages_sharded_tree(tmp_path: Path) -> None:
    """Shard-manifest trees reassemble to answers.jsonl (the live contract).

    Pre-fix the filename filter skipped `answers.shard{NN}.jsonl`, so the
    manifest's reassembly loop raised FileNotFoundError on the first probe.
    """
    prefix = f"{cm.HF_PREFIX_1336}/raw_completions/generation/rlvr/lmsys5k"
    files, expected = _sharded_tree()
    api, dl, hub, downloaded = _fake_hub_stack(prefix, files)

    out = ps._download_answers_jsonl(api, dl, hub, prefix, cm.WAVE1_HF_REV, tmp_path / "dl")

    assert out.read_bytes() == expected
    assert (
        hashlib.sha256(out.read_bytes()).hexdigest()
        == json.loads(files["answers.manifest.json"])["total_sha256"]
    )
    # Filter discipline: manifest + shards staged, decoys untouched.
    assert sorted(downloaded) == [
        "answers.manifest.json",
        "answers.shard00.jsonl",
        "answers.shard01.jsonl",
    ]


def test_download_answers_jsonl_single_file_tree(tmp_path: Path) -> None:
    """A plain single answers.jsonl tree still stages unchanged."""
    prefix = f"{cm.HF_PREFIX_1336}/raw_completions/generation/base/math7500"
    body = b'{"prompt_idx": 0, "kept": true}\n'
    api, dl, hub, downloaded = _fake_hub_stack(prefix, {"answers.jsonl": body})

    out = ps._download_answers_jsonl(api, dl, hub, prefix, "deadbeef", tmp_path / "dl")

    assert out.read_bytes() == body
    assert downloaded == ["answers.jsonl"]


def test_pooled_split_generation_prefix_never_generation_v2() -> None:
    """Every corpus resolves under `generation/` (generation_v2/ 404s live)."""
    for slug in cm.V2_CORPORA:
        subprefix, _stem = ps._resolve_generation_prefix(slug)
        assert subprefix == "generation", (slug, subprefix)
    # Wave-1 stem mapping pinned (revision split rides _WAVE1_CORPORA).
    assert ps._resolve_generation_prefix("lmsys23k") == ("generation", "lmsys5k")
    assert ps._resolve_generation_prefix("gsm8k_train_full") == ("generation", "gsm8k_train5k")
    assert ps._resolve_generation_prefix("if11k") == ("generation", "if11k")
    assert "if11k" not in ps._WAVE1_CORPORA and "lmsys5k" in ps._WAVE1_CORPORA


def test_stage_offpolicy_concat_corpora_stage_both_halves() -> None:
    """Concat corpora return BOTH legs (read_offpolicy_rows hard-asserts both);
    every leg's Hub prefix lives under `generation/`."""
    for corpus, stem in cm.V2_CONCAT_SOURCES.items():
        legs = so._prefix_and_layout("rlvr", corpus)
        assert len(legs) == 2, (corpus, legs)
        (w1_prefix, w1_rev, w1_dest), (ext_prefix, ext_rev, ext_dest) = legs
        assert w1_prefix == f"{cm.HF_PREFIX_1336}/raw_completions/generation/rlvr/{stem}"
        assert w1_rev == cm.WAVE1_HF_REV
        assert w1_dest == so.GEN_ROOT / "rlvr" / stem
        assert ext_prefix == f"{cm.HF_PREFIX_1336}/raw_completions/generation/rlvr/{corpus}"
        assert ext_rev == "main"  # placeholder — resolved once per StageContext
        assert ext_dest == so.GEN_V2_ROOT / "rlvr" / corpus
    # Pure-v2 + wave-1-only corpora stay single-leg, all under generation/.
    for corpus in cm.V2_CORPORA:
        legs = so._prefix_and_layout("dpo", corpus)
        expected_n = 2 if corpus in cm.V2_CONCAT_SOURCES else 1
        assert len(legs) == expected_n, (corpus, legs)
        for prefix, _rev, _dest in legs:
            assert "generation_v2" not in prefix, (corpus, prefix)
            assert "/raw_completions/generation/" in prefix, (corpus, prefix)
    (_t_prefix, t_rev, t_dest) = so._prefix_and_layout("dpo", "gsm8k_test1319")[0]
    assert t_rev == cm.WAVE1_HF_REV
    assert t_dest == so.GEN_ROOT / "dpo" / "gsm8k_test1319"
