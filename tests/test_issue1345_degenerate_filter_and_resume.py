"""#1345 crash-fix r6 regressions (att-20260715-195605).

1. **[degenerate-span-edge]** `AssertionError: s57: span a1=(9,9) invalid for
   unpadded len 10` — the parent (#825) filtered 276 zero-width-span rows out
   of its naturalistic_s extraction (parent shards: chat_s n=5000 incl. s57;
   naturalistic_s n=4724, s57 ABSENT) via a crash-fix that lived only on the
   unmerged `issue-825` branch; #1345's re-extraction wiring bypassed it. The
   fix ports `degenerate_content_turns` / `partition_rendered` /
   `assert_residual_span_integrity` verbatim and wires them into the #1345
   extractor (per-render drop — the parent's exact semantics — + skip
   manifest). s57's response is a bare "." that BPE-merges entirely into the
   naturalistic `\\n\\n` delimiter (gotchas.md zero-width-span class).

2. **[rollout-persist]** gen_stories uploads each model's rollout text + judge
   digests to HF at gen-phase completion (before the yield floor can rc=21 the
   process) and resumes from the persisted bundle on relaunch, keyed on a
   content fingerprint so a recipe change never silently reuses stale stories.

Tests run REAL bodies; fakes only at the HF network boundary (def-mirrored
signatures of the module's own boundary wrappers) and, for the tokenizer
test, a local-files-only load that skips when the cache is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue825_extract_turnstore as ex  # noqa: E402
import issue1345_common as c  # noqa: E402
import issue1345_gen_stories as gs  # noqa: E402

from explore_persona_space.experiments.issue_825.common import Rendered  # noqa: E402


def _rendered(conv_id: str, spans: dict, slot_idx: dict, n_tokens: int = 20) -> Rendered:
    return Rendered(
        input_ids=list(range(n_tokens)),
        slot_idx=slot_idx,
        spans=spans,
        format="naturalistic",
        conv_id=conv_id,
        meta={},
    )


# ---------------------------------------------------------------------------
# Fix 1 — degenerate-row filter (parent parity)
# ---------------------------------------------------------------------------
def test_degenerate_content_turns_matches_process_batch_predicate():
    ok = _rendered("ok", {"u1": (1, 5), "a1": (9, 12)}, {"a1": 8})
    bad = _rendered("s57", {"u1": (3, 6), "a1": (9, 9)}, {"a1": 8}, n_tokens=10)
    assert ex.degenerate_content_turns(ok) == []
    assert ex.degenerate_content_turns(bad) == ["a1"]


def test_partition_rendered_split_and_residual_integrity(capsys):
    ok = _rendered("keep0", {"u1": (1, 5), "a1": (9, 12)}, {"a1": 8})
    bad = _rendered("s57", {"u1": (3, 6), "a1": (9, 9)}, {"a1": 8}, n_tokens=10)
    kept, drops = ex.partition_rendered([ok, bad])
    assert [r.conv_id for r in kept] == ["keep0"]
    assert drops == [{"conv_id": "s57", "turns": ["a1"]}]
    assert "reason=zero_width_span:a1" in capsys.readouterr().out
    ex.assert_residual_span_integrity(kept)  # no raise on kept rows
    with pytest.raises(AssertionError, match=r"span a1=\(9,9\) invalid"):
        ex.assert_residual_span_integrity([bad])  # the exact production crash


def test_filter_keys_on_spans_only_same_skip_set_across_arms():
    """The drop predicate reads content SPANS only, never slot_idx — so the
    prefix and context arms (slots of ONE render) share the skip set by
    construction (the R1/R2 row-alignment precondition)."""
    spans = {"u1": (3, 6), "a1": (9, 9)}
    prefix_arm = _rendered("s57", spans, {"a1": 8, "prefix": 1}, n_tokens=10)
    context_arm = _rendered("s57", spans, {"a1": 8}, n_tokens=10)
    assert ex.degenerate_content_turns(prefix_arm) == ex.degenerate_content_turns(context_arm)


def test_s57_shape_dropped_naturalistic_kept_chat_real_tokenizer():
    """Parent parity on the REAL tokenizer: a bare-'.' single-turn response
    BPE-merges into the naturalistic delimiters (zero-width a1 -> dropped),
    while the chat render's special-token boundaries keep it (parent shards:
    chat_s 5000/5000, naturalistic_s 4724/5000)."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer not in the local HF cache (offline test env)")
    conv = {"conv_id": "s57shape", "u1": "abcdefgh ijklmnop?", "a1": "."}
    rn = c.render_naturalistic_prefix(conv, tok)
    rc_ = c.render_chat_prefix(conv, tok)
    assert ex.degenerate_content_turns(rn) == ["a1"]
    assert ex.degenerate_content_turns(rc_) == []
    kept, drops = ex.partition_rendered([rn])
    assert kept == [] and drops[0]["conv_id"] == "s57shape"


# ---------------------------------------------------------------------------
# Fix 2 — HF persist + content-keyed resume
# ---------------------------------------------------------------------------
def _write_bundle(out_dir: Path, model: str, n_kept: int) -> None:
    (out_dir / f"raw_stories_{model}.jsonl").write_text('{"story_id": "x"}\n')
    (out_dir / f"kept_stories_{model}.jsonl").write_text('{"story_id": "x"}\n')
    (out_dir / f"story_yield_{model}.json").write_text(
        json.dumps({"model": model, "n_kept": n_kept, "yield_ok": n_kept >= 400})
    )
    (out_dir / f"judge_results_{model}.jsonl").write_text('{"story_id": "x"}\n')


def test_bundle_fingerprint_content_keyed(monkeypatch):
    fp1 = gs.bundle_fingerprint("instruct", ["s1", "s2"])
    assert fp1 == gs.bundle_fingerprint("instruct", ["s1", "s2"])  # deterministic
    assert fp1 != gs.bundle_fingerprint("pretrained", ["s1", "s2"])  # model-keyed
    assert fp1 != gs.bundle_fingerprint("instruct", ["s1", "s3"])  # seed-keyed
    monkeypatch.setattr(gs, "JUDGE_SYSTEM", "a different rubric")
    assert fp1 != gs.bundle_fingerprint("instruct", ["s1", "s2"])  # judge-instrument-keyed


def test_try_resume_fresh_returns_none(tmp_path, monkeypatch):
    def fake_exists(path_in_repo: str) -> bool:
        return False

    monkeypatch.setattr(gs, "_hf_file_exists", fake_exists)
    assert gs.try_resume_from_hf("instruct", "f" * 16, tmp_path, smoke=True) is None


def _fake_download_factory(manifest: dict):
    def fake_download(path_in_repo: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        name = path_in_repo.rsplit("/", 1)[-1]
        if name.startswith("story_bundle_manifest_"):
            dest.write_text(json.dumps(manifest))
        elif name.startswith("story_yield_"):
            dest.write_text(json.dumps({"n_kept": 3, "yield_ok": False}))
        else:
            dest.write_text('{"story_id": "x"}\n')
        return dest

    return fake_download


def test_try_resume_hit_downloads_bundle(tmp_path, monkeypatch, capsys):
    fp = "a" * 16
    manifest = {
        "model": "instruct",
        "bundle_fingerprint": fp,
        "files": ["kept_stories_instruct.jsonl", "story_yield_instruct.json"],
    }
    monkeypatch.setattr(gs, "_hf_file_exists", lambda path_in_repo: True)
    monkeypatch.setattr(gs, "_hf_download_to", _fake_download_factory(manifest))
    report = gs.try_resume_from_hf("instruct", fp, tmp_path, smoke=True)
    assert report is not None and report["n_kept"] == 3
    assert (tmp_path / "kept_stories_instruct.jsonl").exists()
    assert "resume-from-HF: reusing 3 persisted kept stories" in capsys.readouterr().out


def test_try_resume_stale_fingerprint_regenerates(tmp_path, monkeypatch, capsys):
    manifest = {
        "model": "instruct",
        "bundle_fingerprint": "b" * 16,
        "files": ["kept_stories_instruct.jsonl", "story_yield_instruct.json"],
    }
    monkeypatch.setattr(gs, "_hf_file_exists", lambda path_in_repo: True)
    monkeypatch.setattr(gs, "_hf_download_to", _fake_download_factory(manifest))
    assert gs.try_resume_from_hf("instruct", "a" * 16, tmp_path, smoke=True) is None
    out = capsys.readouterr().out
    assert "stale (recipe changed)" in out
    # The stale bundle's kept stories were never staged for reuse.
    assert not (tmp_path / "kept_stories_instruct.jsonl").exists()


def test_persist_story_bundle_uploads_and_writes_manifest(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "pretrained", n_kept=121)
    calls: list[tuple] = []

    def fake_upload(folder: Path, path_in_repo: str, allow: list[str], msg: str) -> None:
        calls.append((folder, path_in_repo, tuple(allow), msg))

    monkeypatch.setattr(gs, "_hf_upload_folder", fake_upload)
    monkeypatch.setenv("HF_TOKEN", "hf_unit_test_token")
    fp = "c" * 16
    gs.persist_story_bundle("pretrained", tmp_path, fp, smoke=True)
    manifest = json.loads((tmp_path / "story_bundle_manifest_pretrained.json").read_text())
    assert manifest["bundle_fingerprint"] == fp
    assert "kept_stories_pretrained.jsonl" in manifest["files"]
    assert "judge_results_pretrained.jsonl" in manifest["files"]
    (folder, path_in_repo, allow, _msg) = calls[0]
    assert folder == tmp_path
    assert path_in_repo == "issue1345_smoke/raw_completions/stories"
    assert "*pretrained*" in allow
    # Production prefix rides the canonical issue1345_framing/ bucket.
    gs.persist_story_bundle("pretrained", tmp_path, fp, smoke=False)
    assert calls[1][1] == f"{c.HF_ISSUE_PREFIX}/raw_completions/stories"


def test_persist_requires_kept_file(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "_hf_upload_folder", lambda *a, **k: None)
    monkeypatch.setenv("HF_TOKEN", "hf_unit_test_token")
    with pytest.raises(AssertionError):
        gs.persist_story_bundle("instruct", tmp_path, "d" * 16, smoke=True)


def test_parse_and_judge_returns_digest_rows(monkeypatch, tmp_path):
    """3-tuple contract: per-story judge digests (ids/verdicts only, no text)."""

    class _Res:
        def __init__(self, error=None, category=None, result=None):
            self.error = error
            self.category = category
            self.result = result

    rows = [
        {
            "story_id": "m_story0000",
            "seed_conv_id": "s1",
            "story": 'Q? ARIA replied: "' + "a" * 30 + '"',
        },
        {"story_id": "m_story0001", "seed_conv_id": "s2", "story": "no dialogue at all"},
    ]

    async def fake_dispatch(items, **kwargs):
        return {
            "m_story0000": _Res(result={"verdict": "PASS", "judge_turns": 4}),
            "m_story0001": _Res(error="boom", category="transport"),
        }

    monkeypatch.setattr(gs, "dispatch_calls", fake_dispatch)
    monkeypatch.setattr(gs, "RESULT_TRANSPORT", "transport")
    kept, counts, digest = gs.parse_and_judge(rows, tmp_path / "cache", smoke=True)
    assert kept == []  # single-turn stories sit below STORY_MIN_TURNS
    assert counts["transport_loss"] == 1
    assert {d["story_id"] for d in digest} == {"m_story0000", "m_story0001"}
    by_id = {d["story_id"]: d for d in digest}
    assert by_id["m_story0000"]["verdict"] == "PASS"
    assert by_id["m_story0001"]["judge_error_category"] == "transport"
    assert all("story" not in d for d in digest)  # digest rows never carry text
