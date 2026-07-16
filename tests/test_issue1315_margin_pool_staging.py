"""#1315 r5 crash-fix regression pins — concurrent margin-pool staging.

The r4 production run crashed at p4_parity when 4 concurrent fanout units each
ran ``issue1090_run._stage_hf_prefix`` into the SAME
``margin_pools/impolite`` dest: the pre-fix SHARED ``dest/_hfstage`` staging
dir let one unit's ``os.replace`` consume a file a sibling's
``hf_hub_download`` had just returned, so the sibling's replace raised
``FileNotFoundError`` (epm:failure v4). Pins:

1. ``test_stage_hf_prefix_concurrent_callers_no_steal`` — executes the REAL
   ``_stage_hf_prefix`` body from two threads against one dest, faking ONLY
   the Hub network boundary (``HfApi.list_repo_tree`` + ``hf_hub_download``,
   signature-mirroring fakes). FAILS pre-fix with the production
   FileNotFoundError (verified by stashing the fix); PASSES post-fix
   (per-invocation ``mkdtemp`` staging dirs).
2. ``test_stage_hf_prefix_self_heals_partial_dest`` — a partially-staged dest
   (one file present, one missing + a stale legacy ``_hfstage`` dir) heals:
   only the missing file downloads, the stale dir is swept.
3. ``test_behavior_margin_pools_restages_partial_dest`` — the strengthened
   ``_margin_pool_source_staged`` guard re-stages a dest that has
   ``raw_pos.jsonl`` but lacks the other derive-required sidecars (the
   pre-fix guard skipped staging there and derive crashed on the pod).
4. Predicate truth tables for ``_margin_pool_source_staged`` /
   ``_topup_extra_staged``.
"""

from __future__ import annotations

import json

# The scripts are imported the way the dispatcher imports them (scripts/ on
# sys.path via conftest or direct path insertion).
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_run as run1090  # noqa: E402

PREFIX = "issue1090_pvdatagen/c2-impolite-claude/datagen"


def _fake_hub_boundary(monkeypatch, files: dict[str, str], barrier=None, dl_sleep=0.0):
    """Fake ONLY the Hub network boundary, signature-conformant.

    ``files`` maps prefix-relative name -> content. ``barrier`` (optional)
    synchronizes concurrent callers inside the LISTING call so both pass it
    before either starts downloading; ``dl_sleep`` widens the window between
    a download returning and its ``os.replace`` (the production race window).
    """
    entries = [SimpleNamespace(path=f"{PREFIX}/{name}") for name in files]

    def fake_list_repo_tree(
        self, repo_id, *, path_in_repo=None, repo_type=None, recursive=False, **kw
    ):
        assert repo_id == run1090.HF_DATA_REPO and path_in_repo == PREFIX
        if barrier is not None:
            barrier.wait(timeout=30)
        return iter(entries)

    def fake_hf_hub_download(repo_id, filename, *, repo_type=None, local_dir=None, **kw):
        assert repo_id == run1090.HF_DATA_REPO and local_dir is not None
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():  # the real hub skips a file already staged
            out.write_text(files[filename.rsplit("/", 1)[-1]], encoding="utf-8")
        if dl_sleep:
            time.sleep(dl_sleep)
        return str(out)

    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_tree", fake_list_repo_tree)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)


def test_stage_hf_prefix_concurrent_callers_no_steal(tmp_path, monkeypatch):
    """Two concurrent callers staging the SAME dest never steal each other's
    staged file (pre-fix: one raises the production FileNotFoundError)."""
    dest = tmp_path / "margin_pools" / "impolite"
    barrier = threading.Barrier(2)
    _fake_hub_boundary(monkeypatch, {"pool_meta.json": '{"n": 23}'}, barrier=barrier, dl_sleep=0.5)
    errors: list[BaseException] = []

    def run():
        try:
            run1090._stage_hf_prefix(PREFIX, dest)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, f"concurrent staging raised: {errors!r}"
    assert (dest / "pool_meta.json").read_text(encoding="utf-8") == '{"n": 23}'
    assert not list(dest.glob("_hfstage*")), "staging scratch dirs must be cleaned up"


def test_stage_hf_prefix_self_heals_partial_dest(tmp_path, monkeypatch):
    """A partial dest (crashed prior stage + stale legacy _hfstage) heals:
    only the missing file downloads; the legacy shared dir is swept."""
    dest = tmp_path / "margin_pools" / "impolite"
    dest.mkdir(parents=True)
    (dest / "raw_pos.jsonl").write_text("already-staged\n", encoding="utf-8")
    legacy = dest / "_hfstage" / PREFIX
    legacy.mkdir(parents=True)
    (legacy / "judge_rows.jsonl").write_text("orphaned-partial\n", encoding="utf-8")
    downloaded: list[str] = []
    files = {"raw_pos.jsonl": "SHOULD-NOT-REDOWNLOAD", "judge_rows.jsonl": "healed\n"}
    _fake_hub_boundary(monkeypatch, files)
    real_download = __import__("huggingface_hub").hf_hub_download

    def recording_download(repo_id, filename, **kw):
        downloaded.append(filename)
        return real_download(repo_id, filename, **kw)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", recording_download)
    run1090._stage_hf_prefix(PREFIX, dest)
    assert downloaded == [f"{PREFIX}/judge_rows.jsonl"], downloaded
    assert (dest / "raw_pos.jsonl").read_text(encoding="utf-8") == "already-staged\n"
    assert (dest / "judge_rows.jsonl").read_text(encoding="utf-8") == "healed\n"
    assert not (dest / "_hfstage").exists(), "legacy shared staging dir must be swept"
    assert not list(dest.glob("_hfstage*"))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _datagen_fixture(dest: Path) -> None:
    """Minimal derive_margin_pools-consumable sidecar set (1 kept row/arm)."""
    dest.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        dest / "raw_pos.jsonl",
        [
            {
                "request_id": "p1",
                "arm": "positive",
                "completion": "Ugh, obviously.",
                "question": "q0?",
                "question_id": 0,
                "variant_id": 0,
            }
        ],
    )
    _write_jsonl(
        dest / "raw_neg.jsonl",
        [
            {
                "request_id": "n1",
                "arm": "negative",
                "completion": "Happy to help!",
                "question": "q0?",
                "question_id": 0,
                "variant_id": 0,
            }
        ],
    )
    _write_jsonl(
        dest / "judge_rows.jsonl",
        [{"request_id": "p1", "kept": True}, {"request_id": "n1", "kept": True}],
    )


def test_behavior_margin_pools_restages_partial_dest(tmp_path, monkeypatch):
    """The strengthened guard re-stages a dest that has raw_pos.jsonl but
    lacks the other derive-required sidecars (pre-fix guard: skipped staging
    -> derive_margin_pools ValueError on the pod)."""
    out_root = tmp_path / "run"
    dest = out_root / "margin_pools" / "impolite"
    dest.mkdir(parents=True)
    # Partial dest: only raw_pos landed before the prior stage was reaped.
    _write_jsonl(
        dest / "raw_pos.jsonl",
        [
            {
                "request_id": "p1",
                "arm": "positive",
                "completion": "Ugh, obviously.",
                "question": "q0?",
                "question_id": 0,
                "variant_id": 0,
            }
        ],
    )
    staged: list[str] = []

    def fake_stage(prefix: str, dest_arg: Path, *, skip_if=None) -> None:
        # Signature mirrors run1090._stage_hf_prefix; materializes the full
        # sidecar set (the body itself is covered by the two tests above).
        staged.append(prefix)
        _datagen_fixture(dest_arg)

    monkeypatch.setattr(run1090, "_stage_hf_prefix", fake_stage)
    cfg = run1090.RunConfig(smoke=True, cells=(), out_root=out_root)
    pos, neg = fu3w._behavior_margin_pools(cfg, "impolite")
    assert staged == [f"{run1090.DATA_PREFIX}/c2-impolite-claude/datagen"], staged
    assert [p["request_id"] for p in pos] == ["p1"]
    assert [n["request_id"] for n in neg] == ["n1"]


def test_margin_pool_source_staged_predicate(tmp_path):
    d = tmp_path / "impolite"
    d.mkdir()
    assert not fu3w._margin_pool_source_staged(d, "datagen")
    (d / "raw_pos.jsonl").write_text("x\n", encoding="utf-8")
    assert not fu3w._margin_pool_source_staged(d, "datagen"), (
        "raw_pos alone must NOT read as staged (the pre-fix single-file guard)"
    )
    (d / "raw_neg.jsonl").write_text("x\n", encoding="utf-8")
    (d / "judge_rows.jsonl").write_text("x\n", encoding="utf-8")
    assert fu3w._margin_pool_source_staged(d, "datagen")
    # topup schema needs kept_{pos,neg} too
    assert not fu3w._margin_pool_source_staged(d, "datagen_topup")
    (d / "kept_pos.jsonl").write_text("x\n", encoding="utf-8")
    (d / "kept_neg.jsonl").write_text("x\n", encoding="utf-8")
    assert fu3w._margin_pool_source_staged(d, "datagen_topup")


def test_topup_extra_staged_predicate(tmp_path):
    d = tmp_path / "extra"
    d.mkdir()
    assert not fu3w._topup_extra_staged(d)  # empty -> stage
    (d / "raw_pos.jsonl").write_text("x\n", encoding="utf-8")
    assert not fu3w._topup_extra_staged(d)  # half-present pos pair -> re-stage
    (d / "kept_pos.jsonl").write_text("x\n", encoding="utf-8")
    assert fu3w._topup_extra_staged(d)  # pos-only tranche is complete
    (d / "raw_neg.jsonl").write_text("x\n", encoding="utf-8")
    assert not fu3w._topup_extra_staged(d)  # half-present neg pair -> re-stage
    (d / "kept_neg.jsonl").write_text("x\n", encoding="utf-8")
    assert fu3w._topup_extra_staged(d)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
