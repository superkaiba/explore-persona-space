"""Regression tests for issue #734's setup_h1_mix phase (crash-fix round 1).

The substantive invariant: the setup_h1_mix phase builds the librarian-contra-d1-
seed42 marker training mix at EXACTLY the path train_h1_cell asserts exists, and
its idempotency gate keys on a recorded sha256 (not bare file-presence) so a
present-but-stale mix is rebuilt while a current one is a no-op.

These are CPU-only orchestration tests: they monkeypatch the GPU marker_R
elicitation + the CPU builder subprocess (both of which need a model / a clean
subprocess) so the path contract + the idempotency + the sha256-provenance
round-trip are pinned without a GPU forward. A pre-fix tree (no setup_h1_mix
phase) has no module to import; the phase's path/idempotency logic is the
load-bearing thing under test.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture()
def mix(monkeypatch, tmp_path):
    """issue734_h1_mix with C664.DATA_ROOT + the dispatcher CACHE_ROOT redirected
    into tmp_path so the build writes nowhere real. Returns the module."""
    import issue664_common as C664
    import issue734_h1_mix as MIX

    monkeypatch.setattr(C664, "DATA_ROOT", tmp_path / "data" / "issue_664", raising=True)
    return MIX


def test_mix_path_matches_train_h1_cell_assert_path(mix):
    """THE crash-fix invariant: setup_h1_mix writes the mix at EXACTLY the path
    train_h1_cell's `assert data_path.exists()` reads (the round-1 crash site)."""
    import issue664_common as C664

    # Reconstruct train_h1_cell's data_path expression from ground truth (the
    # dispatcher reads C664.DATA_ROOT / ("train_smoke" if smoke else "train") /
    # "marker" / f"{c664.eval_key}.jsonl"). The H1 cell pins seed=PHASE1_SEED.
    import issue734_common as C

    c664 = C.h1_cells()[0].to_664_cell()
    for smoke in (False, True):
        expected = (
            C664.DATA_ROOT
            / ("train_smoke" if smoke else "train")
            / "marker"
            / f"{c664.eval_key}.jsonl"
        )
        assert mix.mix_path(smoke=smoke) == expected, (
            f"setup_h1_mix path {mix.mix_path(smoke=smoke)} != train_h1_cell assert path "
            f"{expected} (smoke={smoke}) -- the crash would re-fire"
        )
    # And the cell is the seed-42-pinned librarian-contra-d1 (round-3 reuse).
    assert c664.eval_key == "mk_librarian_contra_d1_seed42"


def test_needed_marker_R_contexts_are_source_plus_four_negatives(mix):
    """The build reads marker_R for the librarian SOURCE + the 4 panel NEGATIVES
    (issue664_build_training_data.build_marker); the set must be exactly those 5."""
    import issue664_common as C664

    ctxs = mix._needed_marker_R_contexts()
    expected = ["librarian", *[n.slug for n in C664.negative_panel()]]
    assert ctxs == expected
    assert len(ctxs) == 5


def test_mix_is_current_false_with_no_file(mix):
    """Idempotency gate: a missing mix is NOT current (must build)."""
    assert mix._mix_is_current(smoke=False) is False


def test_mix_is_current_true_on_matching_sha_false_on_mismatch(mix):
    """Idempotency gate keys on a RECORDED sha256, not bare file-presence: a mix
    whose bytes match its provenance sha is current; a tampered mix is not."""
    out = mix.mix_path(smoke=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"prompt": [], "completion": "x"}\n')
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    prov = mix._provenance_path(smoke=False)

    # No provenance yet -> not current (present-but-unrecorded).
    assert mix._mix_is_current(smoke=False) is False

    # Recorded matching sha -> current.
    prov.write_text(json.dumps({"sha256": sha}))
    assert mix._mix_is_current(smoke=False) is True

    # Tamper the mix bytes -> sha mismatch -> NOT current (rebuild).
    out.write_text('{"prompt": [], "completion": "TAMPERED"}\n')
    assert mix._mix_is_current(smoke=False) is False


def test_build_records_sha_and_second_run_is_noop(mix, monkeypatch):
    """build_h1_mix round-trip with the GPU + subprocess steps stubbed:
    (1) it records the built mix's sha256 in the provenance sidecar; and
    (2) a second build is a no-op (the idempotency skip fires)."""
    import issue664_dispatch as D

    out = mix.mix_path(smoke=False)
    build_calls = {"n": 0}

    # Stub step 1 (pool write) + step 2 (vLLM marker_R elicitation): no GPU.
    monkeypatch.setattr(D, "_marker_question_pool", lambda smoke: ["q1", "q2", "q3"])
    monkeypatch.setattr(D, "_write_pool", lambda behavior, questions, *, smoke: None)
    # Make the "all caches present" branch fire so the engine is never built:
    # marker_R/<ctx>.json existence is checked under D.CACHE_ROOT.
    cache_root = mix.mix_path(smoke=False).parents[2] / "onpolicy_cache"
    monkeypatch.setattr(D, "CACHE_ROOT", cache_root, raising=True)
    (cache_root / "marker_R").mkdir(parents=True, exist_ok=True)
    for ctx in mix._needed_marker_R_contexts():
        (cache_root / "marker_R" / f"{ctx}.json").write_text("{}")

    def _fake_engine(*a, **k):
        raise AssertionError("vLLM engine must NOT be built when all caches are present")

    monkeypatch.setattr(D, "_vllm_engine", _fake_engine)

    # Stub step 3 (the CPU builder subprocess): write a fake mix in its place.
    # `subprocess` is a shared module object, so repro_meta()'s `git rev-parse`
    # also routes here -- delegate every NON-builder call to the real run.
    import subprocess as _sp

    _real_run = _sp.run

    def _fake_subprocess_run(cmd, **kwargs):
        if "issue664_build_training_data.py" not in " ".join(map(str, cmd)):
            return _real_run(cmd, **kwargs)  # e.g. git rev-parse in repro_meta
        build_calls["n"] += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"prompt": [{"role": "user", "content": "q1"}], "completion": " mk"}\n')

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("issue734_h1_mix.subprocess.run", _fake_subprocess_run)

    # First build: writes the mix + records the sha.
    built = mix.build_h1_mix(smoke=False)
    assert built == out and out.exists()
    assert build_calls["n"] == 1
    prov = json.loads(mix._provenance_path(smoke=False).read_text())
    assert prov["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert prov["cell"] == "mk_librarian_contra_d1_seed42"
    assert prov["marker_R_contexts"] == mix._needed_marker_R_contexts()

    # Second build: idempotency skip -> the builder subprocess is NOT re-run.
    again = mix.build_h1_mix(smoke=False)
    assert again == out
    assert build_calls["n"] == 1, "second build re-ran the subprocess (idempotency broke)"


def test_build_fails_loud_when_builder_crashes(mix, monkeypatch):
    """A non-zero builder rc (including the rc==3 DROP code -- a real bug for the
    single H1 marker mix, never a tolerated degradation) raises, never silent."""
    import subprocess

    import issue664_dispatch as D

    monkeypatch.setattr(D, "_marker_question_pool", lambda smoke: ["q1"])
    monkeypatch.setattr(D, "_write_pool", lambda behavior, questions, *, smoke: None)
    cache_root = mix.mix_path(smoke=False).parents[2] / "onpolicy_cache"
    monkeypatch.setattr(D, "CACHE_ROOT", cache_root, raising=True)
    (cache_root / "marker_R").mkdir(parents=True, exist_ok=True)
    for ctx in mix._needed_marker_R_contexts():
        (cache_root / "marker_R" / f"{ctx}.json").write_text("{}")
    monkeypatch.setattr(
        D, "_vllm_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no engine"))
    )

    _real_run = subprocess.run

    def _crash_run(cmd, **kwargs):
        if "issue664_build_training_data.py" not in " ".join(map(str, cmd)):
            return _real_run(cmd, **kwargs)

        class _R:
            returncode = 3  # the DROPPED_SOURCE_EXIT code

        return _R()

    monkeypatch.setattr("issue734_h1_mix.subprocess.run", _crash_run)
    with pytest.raises(subprocess.CalledProcessError):
        mix.build_h1_mix(smoke=False)
