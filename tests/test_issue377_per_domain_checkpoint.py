"""Regression test for the round-8 per-domain checkpoint save discipline.

Round 6 of the issue #377 drift corpus generation ran 3 clean domains
(therapy / philosophy / roleplay) then aborted at hostile_jailbreak
turn 5 (FIX-3 mid-run quality ceiling). The script's only ``write_jsonl``
call was at the end of the all-domains loop, AFTER per-corpus post-gen
sanity checks — neither fired, and the 3 clean domains' data was lost
to memory. Rounds 5 and 6 hit the same shape.

The round-8 fix in ``scripts/issue_377_generate_drift_corpus.py`` and
``scripts/issue_377_generate_incontext_corpus.py`` writes each domain's
conversations to its own per-domain JSONL the moment that domain's
loop completes — BEFORE the next domain starts. This test guards
against a regression of that discipline by mocking
``run_conversation_loop`` (so we don't hit the Anthropic / OpenAI
batch APIs) and the HF Hub upload, running the script's ``main()``
end-to-end, and asserting that per-domain checkpoint files
materialized in the expected order with the expected content.

The test is PURE: no network, no API keys, no subprocess. It loads
the script as a module via ``importlib.util`` so we can monkeypatch
its module-level symbols cleanly.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIFT_SCRIPT = REPO_ROOT / "scripts" / "issue_377_generate_drift_corpus.py"
INCONTEXT_SCRIPT = REPO_ROOT / "scripts" / "issue_377_generate_incontext_corpus.py"


def _load_script_as_module(path: Path, module_name: str) -> ModuleType:
    """Load a script file as an importable module so we can monkeypatch it."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_personas_for_domain(domain_name: str, n: int = 2) -> list[dict]:
    """Minimal persona records — only used as opaque inputs to the mocked loop."""
    return [
        {
            "persona_id": i,
            "backstory": f"{domain_name} persona {i}",
            "topics": [f"{domain_name} topic {i}"],
        }
        for i in range(n)
    ]


def _fake_conversations_for_domain(domain_name: str, n_turns: int = 15) -> list[dict]:
    """Build 2 fake conversation records matching the live schema."""
    return [
        {
            "conversation_id": f"{domain_name}_p0_t0",
            "domain": domain_name,
            "persona_id": 0,
            "persona_backstory": f"{domain_name} persona 0",
            "topic_id": 0,
            "topic": f"{domain_name} topic 0",
            "auditor_model": "claude-sonnet-4-5",
            "target_model_during_drift_gen": "claude-sonnet-4-5",
            "rotation_seed": 0,
            "turns": [
                {"role": "user" if i % 2 == 0 else "assistant", "content": "ok"}
                for i in range(n_turns)
            ],
            "n_turns": n_turns,
        },
        {
            "conversation_id": f"{domain_name}_p1_t0",
            "domain": domain_name,
            "persona_id": 1,
            "persona_backstory": f"{domain_name} persona 1",
            "topic_id": 0,
            "topic": f"{domain_name} topic 0",
            "auditor_model": "claude-sonnet-4-5",
            "target_model_during_drift_gen": "claude-sonnet-4-5",
            "rotation_seed": 0,
            "turns": [
                {"role": "user" if i % 2 == 0 else "assistant", "content": "ok"}
                for i in range(n_turns)
            ],
            "n_turns": n_turns,
        },
    ]


class TestDriftScriptPerDomainCheckpoint:
    """The drift entry script must write each domain's JSONL the moment
    that domain's conversation loop completes — before starting the next
    domain. We mock the conversation loop + upload + sanity check to
    avoid network / shape constraints, then drive ``main()`` end to end
    and verify per-domain files materialized in domain order.
    """

    def test_per_domain_files_written_in_loop_order(self, tmp_path, monkeypatch):
        mod = _load_script_as_module(DRIFT_SCRIPT, "_issue377_drift_under_test")

        # Redirect DATA_DIR to a sandbox so the test does not touch
        # ``data/issue377_drift/``.
        sandbox = tmp_path / "issue377_drift"
        monkeypatch.setattr(mod, "DATA_DIR", sandbox)
        monkeypatch.setattr(mod, "OUTPUT_PATH", sandbox / "drift_conversations.jsonl")
        monkeypatch.setattr(mod, "SEED_CACHE_PATH", sandbox / "persona_topic_seeds_drift.json")

        # Stub the seed step so we don't hit Anthropic.
        def fake_seed(_domains, *, cache_path, custom_id_prefix):
            return {d.name: _fake_personas_for_domain(d.name) for d in _domains}

        monkeypatch.setattr(mod, "seed_personas_and_topics", fake_seed)

        # Record the order in which run_conversation_loop is called so we
        # can prove the per-domain write happens BEFORE the next loop call.
        call_log: list[tuple[str, str]] = []

        def fake_loop(domain, _personas, *, custom_id_prefix, n_turns, rotation_seed):
            call_log.append(("loop_start", domain.name))
            convs = _fake_conversations_for_domain(domain.name, n_turns=n_turns)
            return convs

        monkeypatch.setattr(mod, "run_conversation_loop", fake_loop)

        # Wrap write_corpus_jsonl so we also see WHEN each domain's
        # checkpoint was written, interleaved with the loop_start events.
        # We DISTINGUISH per-domain checkpoint writes (filename starts
        # with ``conversations_``) from the aggregate Step 5 write (which
        # targets ``drift_conversations.jsonl``), because both call
        # write_corpus_jsonl but only the per-domain writes matter for
        # the round-6 data-loss invariant.
        original_write = mod.write_corpus_jsonl

        def tracking_write(conversations, *, corpus_tag, output_path):
            domain_name = conversations[0]["domain"] if conversations else "unknown"
            kind = (
                "per_domain_write"
                if Path(output_path).name.startswith("conversations_")
                else "aggregate_write"
            )
            call_log.append((kind, domain_name, str(output_path)))
            return original_write(conversations, corpus_tag=corpus_tag, output_path=output_path)

        monkeypatch.setattr(mod, "write_corpus_jsonl", tracking_write)

        # Stub the sanity check + the HF upload so they don't fight the
        # 2-conversation fake corpus or hit the network.
        monkeypatch.setattr(mod, "post_gen_sanity_checks", lambda *a, **k: None)
        monkeypatch.setattr(mod, "upload_dataset_directory", lambda **k: None)

        # Drive the script end-to-end with --no-upload (defense in depth).
        monkeypatch.setattr(sys, "argv", ["issue_377_generate_drift_corpus.py", "--no-upload"])
        rc = mod.main()
        assert rc == 0, f"Drift script exited non-zero: {rc}"

        # Pull domain names out of the live module so we don't hard-code
        # round-N domain assignments here.
        expected_domains = [d.name for d in mod.DRIFT_DOMAINS]

        # CORE INVARIANT: for each domain, the loop-start event precedes
        # the per-domain write event, AND the per-domain write for
        # domain D precedes the loop-start for the next domain. That's
        # the property round-6 violated: there was ONE aggregate write
        # at the very end, so a mid-run abort lost everything earlier.
        events_by_domain: dict[str, dict[str, int]] = {}
        for idx, evt in enumerate(call_log):
            kind, dname = evt[0], evt[1]
            if kind in ("loop_start", "per_domain_write"):
                events_by_domain.setdefault(dname, {})[kind] = idx

        for dname in expected_domains:
            assert "loop_start" in events_by_domain[dname], (
                f"Domain {dname} loop never started; got log {call_log}"
            )
            assert "per_domain_write" in events_by_domain[dname], (
                f"Domain {dname} loop ran but no per-domain write fired — "
                f"this is the round-6 data-loss bug. Log: {call_log}"
            )
            assert (
                events_by_domain[dname]["loop_start"] < events_by_domain[dname]["per_domain_write"]
            ), f"Domain {dname} wrote BEFORE its loop ran; log: {call_log}"

        for prev, nxt in itertools.pairwise(expected_domains):
            assert (
                events_by_domain[prev]["per_domain_write"] < events_by_domain[nxt]["loop_start"]
            ), (
                f"Domain {prev} did not flush its checkpoint before {nxt} started. "
                f"That's the regression we're guarding against. Log: {call_log}"
            )

        # Per-domain files actually materialized on disk.
        for dname in expected_domains:
            f = sandbox / f"conversations_{dname}.jsonl"
            assert f.exists(), f"Per-domain checkpoint {f} missing on disk"
            lines = f.read_text().strip().splitlines()
            assert len(lines) == 2, (
                f"{f} should have 2 conversations (matching the fake loop), got {len(lines)}"
            )
            row = json.loads(lines[0])
            assert row["domain"] == dname
            assert row["corpus"] == "drift"


class TestIncontextScriptPerDomainCheckpoint:
    """Same invariant for the in-context script — both scripts share the
    same data-loss failure mode (they're called sequentially by the
    dispatcher) so both must flush per-domain.
    """

    def test_per_domain_files_written_in_loop_order(self, tmp_path, monkeypatch):
        mod = _load_script_as_module(INCONTEXT_SCRIPT, "_issue377_incontext_under_test")

        sandbox = tmp_path / "issue377_incontext"
        monkeypatch.setattr(mod, "DATA_DIR", sandbox)
        monkeypatch.setattr(mod, "OUTPUT_PATH", sandbox / "incontext_conversations.jsonl")
        monkeypatch.setattr(mod, "SEED_CACHE_PATH", sandbox / "persona_topic_seeds_incontext.json")

        def fake_seed(_domains, *, cache_path, custom_id_prefix):
            return {d.name: _fake_personas_for_domain(d.name) for d in _domains}

        monkeypatch.setattr(mod, "seed_personas_and_topics", fake_seed)

        call_log: list[tuple] = []

        def fake_loop(domain, _personas, *, custom_id_prefix, n_turns, rotation_seed):
            call_log.append(("loop_start", domain.name))
            return _fake_conversations_for_domain(domain.name, n_turns=n_turns)

        monkeypatch.setattr(mod, "run_conversation_loop", fake_loop)

        original_write = mod.write_corpus_jsonl

        def tracking_write(conversations, *, corpus_tag, output_path):
            domain_name = conversations[0]["domain"] if conversations else "unknown"
            kind = (
                "per_domain_write"
                if Path(output_path).name.startswith("conversations_")
                else "aggregate_write"
            )
            call_log.append((kind, domain_name, str(output_path)))
            return original_write(conversations, corpus_tag=corpus_tag, output_path=output_path)

        monkeypatch.setattr(mod, "write_corpus_jsonl", tracking_write)
        monkeypatch.setattr(mod, "post_gen_sanity_checks", lambda *a, **k: None)
        monkeypatch.setattr(mod, "upload_dataset_directory", lambda **k: None)

        # Plan v2 hot-fix (2026-05-25) dropped the --allow-missing-drift-summary
        # flag (and the hard ±10% sanity check it bypassed). The script now
        # writes a stats file unconditionally; with the drift corpus absent
        # on disk, the stats file just has zero ratios. No flag needed.
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "issue_377_generate_incontext_corpus.py",
                "--no-upload",
            ],
        )
        rc = mod.main()
        assert rc == 0, f"In-context script exited non-zero: {rc}"

        expected_domains = [d.name for d in mod.INCONTEXT_DOMAINS]

        events_by_domain: dict[str, dict[str, int]] = {}
        for idx, evt in enumerate(call_log):
            kind, dname = evt[0], evt[1]
            if kind in ("loop_start", "per_domain_write"):
                events_by_domain.setdefault(dname, {})[kind] = idx

        for dname in expected_domains:
            assert "loop_start" in events_by_domain[dname]
            assert "per_domain_write" in events_by_domain[dname], (
                f"Domain {dname} loop ran but no per-domain write fired. Log: {call_log}"
            )
            assert (
                events_by_domain[dname]["loop_start"] < events_by_domain[dname]["per_domain_write"]
            )

        for prev, nxt in itertools.pairwise(expected_domains):
            assert (
                events_by_domain[prev]["per_domain_write"] < events_by_domain[nxt]["loop_start"]
            ), f"Domain {prev} did not flush before {nxt} started. Log: {call_log}"

        for dname in expected_domains:
            f = sandbox / f"conversations_{dname}.jsonl"
            assert f.exists(), f"Per-domain checkpoint {f} missing on disk"
            row = json.loads(f.read_text().strip().splitlines()[0])
            assert row["domain"] == dname
            assert row["corpus"] == "incontext"


class TestReadCorpusJsonlRoundTrip:
    """``read_corpus_jsonl`` is the inverse of ``write_corpus_jsonl`` and
    the recovery primitive for round-7+ (load checkpointed per-domain
    files when only a later domain crashed).
    """

    def test_round_trip_preserves_rows(self, tmp_path):
        from explore_persona_space.data_gen.issue377_corpus import (
            read_corpus_jsonl,
            write_corpus_jsonl,
        )

        convs = _fake_conversations_for_domain("therapy", n_turns=15)
        out = tmp_path / "conversations_therapy.jsonl"
        write_corpus_jsonl(convs, corpus_tag="drift", output_path=out)

        rows = read_corpus_jsonl(out)
        assert len(rows) == len(convs)
        assert {r["conversation_id"] for r in rows} == {c["conversation_id"] for c in convs}
        assert all(r["corpus"] == "drift" for r in rows)
        assert all(r["n_turns"] == 15 for r in rows)

    def test_malformed_row_raises(self, tmp_path):
        from explore_persona_space.data_gen.issue377_corpus import read_corpus_jsonl

        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"a": 1}\nthis-is-not-json\n')
        with pytest.raises(ValueError, match="Malformed JSONL row"):
            read_corpus_jsonl(bad)


def _write_per_domain_checkpoints(
    sandbox: Path,
    domains: list,
    *,
    corpus_tag: str,
    n_turns: int = 15,
) -> dict[str, Path]:
    """Pre-write per-domain JSONL checkpoints into ``sandbox`` for resume tests.

    Returns the map of domain name -> JSONL path. The file shape matches
    ``write_corpus_jsonl``'s output exactly so the resume-skip code path
    can load them via ``read_corpus_jsonl`` end-to-end.
    """
    from explore_persona_space.data_gen.issue377_corpus import write_corpus_jsonl

    sandbox.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for d in domains:
        convs = _fake_conversations_for_domain(d.name, n_turns=n_turns)
        out = sandbox / f"conversations_{d.name}.jsonl"
        write_corpus_jsonl(convs, corpus_tag=corpus_tag, output_path=out)
        paths[d.name] = out
    return paths


class TestDriftScriptResumeSkip:
    """Round-9 r2 patch: when every per-domain JSONL already exists on disk
    (e.g. from a prior crashed run that finished all 4 loops but tripped
    Step 3 sanity), the script must skip both the persona-seed step AND
    the per-domain conversation loops, and load the cached files instead.

    Round-9 r1 lost ~3 hours of batch-API spend to the strict-raise on a
    single trigger-key leak; these tests pin down the recovery path so a
    naive re-run picks up where the prior run left off.
    """

    def test_resume_skip_when_all_four_jsonls_exist(self, tmp_path, monkeypatch):
        mod = _load_script_as_module(DRIFT_SCRIPT, "_issue377_drift_under_test_resume_full")

        sandbox = tmp_path / "issue377_drift"
        monkeypatch.setattr(mod, "DATA_DIR", sandbox)
        monkeypatch.setattr(mod, "OUTPUT_PATH", sandbox / "drift_conversations.jsonl")
        monkeypatch.setattr(mod, "SEED_CACHE_PATH", sandbox / "persona_topic_seeds_drift.json")

        # Pre-populate per-domain checkpoints for every drift domain.
        _write_per_domain_checkpoints(sandbox, list(mod.DRIFT_DOMAINS), corpus_tag="drift")

        # ANY call to seed_personas_and_topics or run_conversation_loop is
        # a regression — both must be skipped on full resume.
        def boom_seed(*a, **kw):
            raise AssertionError(
                "seed_personas_and_topics must NOT be called when all "
                "per-domain JSONLs exist (full resume path)."
            )

        def boom_loop(*a, **kw):
            raise AssertionError(
                "run_conversation_loop must NOT be called when the "
                "per-domain JSONL for that domain already exists."
            )

        monkeypatch.setattr(mod, "seed_personas_and_topics", boom_seed)
        monkeypatch.setattr(mod, "run_conversation_loop", boom_loop)

        # Spy on read_corpus_jsonl so we can count one call per domain.
        original_read = mod.read_corpus_jsonl
        read_calls: list[str] = []

        def tracking_read(path):
            read_calls.append(str(path))
            return original_read(path)

        monkeypatch.setattr(mod, "read_corpus_jsonl", tracking_read)

        # Sanity-check + upload are still no-ops for this test (we don't
        # want to validate the synthetic 2-conv-per-domain fixture).
        monkeypatch.setattr(mod, "post_gen_sanity_checks", lambda *a, **k: None)
        monkeypatch.setattr(mod, "upload_dataset_directory", lambda **k: None)

        monkeypatch.setattr(sys, "argv", ["issue_377_generate_drift_corpus.py", "--no-upload"])
        rc = mod.main()
        assert rc == 0

        # Resume-skip must have loaded exactly one file per drift domain.
        domain_names = [d.name for d in mod.DRIFT_DOMAINS]
        assert len(read_calls) == len(domain_names), (
            f"Expected {len(domain_names)} read_corpus_jsonl calls, "
            f"got {len(read_calls)}: {read_calls}"
        )
        for name in domain_names:
            assert any(f"conversations_{name}.jsonl" in c for c in read_calls), (
                f"Domain {name} JSONL not loaded; read_calls={read_calls}"
            )

    def test_resume_skip_partial_runs_missing_domains_only(self, tmp_path, monkeypatch):
        mod = _load_script_as_module(DRIFT_SCRIPT, "_issue377_drift_under_test_resume_partial")

        sandbox = tmp_path / "issue377_drift"
        monkeypatch.setattr(mod, "DATA_DIR", sandbox)
        monkeypatch.setattr(mod, "OUTPUT_PATH", sandbox / "drift_conversations.jsonl")
        monkeypatch.setattr(mod, "SEED_CACHE_PATH", sandbox / "persona_topic_seeds_drift.json")

        # Pre-populate JSONLs for only the FIRST TWO domains.
        domains = list(mod.DRIFT_DOMAINS)
        assert len(domains) >= 3, "test assumes >=3 drift domains"
        existing = domains[:2]
        missing = domains[2:]
        _write_per_domain_checkpoints(sandbox, existing, corpus_tag="drift")

        # seed must still run (we need personas for the missing domains).
        seed_called: list[bool] = []

        def fake_seed(_domains, *, cache_path, custom_id_prefix):
            seed_called.append(True)
            return {d.name: _fake_personas_for_domain(d.name) for d in _domains}

        monkeypatch.setattr(mod, "seed_personas_and_topics", fake_seed)

        # run_conversation_loop must be called ONLY for the missing domains.
        loop_calls: list[str] = []

        def fake_loop(domain, _personas, *, custom_id_prefix, n_turns, rotation_seed):
            loop_calls.append(domain.name)
            return _fake_conversations_for_domain(domain.name, n_turns=n_turns)

        monkeypatch.setattr(mod, "run_conversation_loop", fake_loop)

        monkeypatch.setattr(mod, "post_gen_sanity_checks", lambda *a, **k: None)
        monkeypatch.setattr(mod, "upload_dataset_directory", lambda **k: None)

        monkeypatch.setattr(sys, "argv", ["issue_377_generate_drift_corpus.py", "--no-upload"])
        rc = mod.main()
        assert rc == 0

        assert seed_called, "seed step must run on partial resume (missing domains need personas)"

        missing_names = {d.name for d in missing}
        existing_names = {d.name for d in existing}
        assert set(loop_calls) == missing_names, (
            f"Expected loop to run only for missing domains {missing_names}, got {loop_calls}"
        )
        for name in existing_names:
            assert name not in loop_calls, (
                f"Domain {name} had a JSONL on disk but loop ran anyway: {loop_calls}"
            )

    def test_no_resume_flag_forces_full_run(self, tmp_path, monkeypatch):
        mod = _load_script_as_module(DRIFT_SCRIPT, "_issue377_drift_under_test_no_resume")

        sandbox = tmp_path / "issue377_drift"
        monkeypatch.setattr(mod, "DATA_DIR", sandbox)
        monkeypatch.setattr(mod, "OUTPUT_PATH", sandbox / "drift_conversations.jsonl")
        monkeypatch.setattr(mod, "SEED_CACHE_PATH", sandbox / "persona_topic_seeds_drift.json")

        # Pre-populate JSONLs for every domain.
        _write_per_domain_checkpoints(sandbox, list(mod.DRIFT_DOMAINS), corpus_tag="drift")

        # seed AND loop MUST run despite the checkpoints (because --no-resume).
        seed_called: list[bool] = []

        def fake_seed(_domains, *, cache_path, custom_id_prefix):
            seed_called.append(True)
            return {d.name: _fake_personas_for_domain(d.name) for d in _domains}

        monkeypatch.setattr(mod, "seed_personas_and_topics", fake_seed)

        loop_calls: list[str] = []

        def fake_loop(domain, _personas, *, custom_id_prefix, n_turns, rotation_seed):
            loop_calls.append(domain.name)
            return _fake_conversations_for_domain(domain.name, n_turns=n_turns)

        monkeypatch.setattr(mod, "run_conversation_loop", fake_loop)
        monkeypatch.setattr(mod, "post_gen_sanity_checks", lambda *a, **k: None)
        monkeypatch.setattr(mod, "upload_dataset_directory", lambda **k: None)

        monkeypatch.setattr(
            sys,
            "argv",
            ["issue_377_generate_drift_corpus.py", "--no-upload", "--no-resume"],
        )
        rc = mod.main()
        assert rc == 0

        assert seed_called, "--no-resume must still run the seed step"
        assert set(loop_calls) == {d.name for d in mod.DRIFT_DOMAINS}, (
            f"--no-resume must run loop for every drift domain; got {loop_calls}"
        )


class TestIncontextScriptResumeSkip:
    """Symmetric resume-skip invariant for the in-context script.

    The in-context script has no per-domain JSONLs from the round-9 r1 run
    yet (round-9 r1 aborted at the drift script's Step 3, before the
    in-context script was reached). These tests are forward-looking:
    they guarantee that ANY future crash inside the in-context script's
    Step 3+ doesn't force a full regeneration.
    """

    def test_resume_skip_when_all_four_jsonls_exist(self, tmp_path, monkeypatch):
        mod = _load_script_as_module(INCONTEXT_SCRIPT, "_issue377_incontext_under_test_resume_full")

        sandbox = tmp_path / "issue377_incontext"
        monkeypatch.setattr(mod, "DATA_DIR", sandbox)
        monkeypatch.setattr(mod, "OUTPUT_PATH", sandbox / "incontext_conversations.jsonl")
        monkeypatch.setattr(mod, "SEED_CACHE_PATH", sandbox / "persona_topic_seeds_incontext.json")

        _write_per_domain_checkpoints(sandbox, list(mod.INCONTEXT_DOMAINS), corpus_tag="incontext")

        def boom_seed(*a, **kw):
            raise AssertionError("seed_personas_and_topics must NOT be called on full resume")

        def boom_loop(*a, **kw):
            raise AssertionError("run_conversation_loop must NOT be called on full resume")

        monkeypatch.setattr(mod, "seed_personas_and_topics", boom_seed)
        monkeypatch.setattr(mod, "run_conversation_loop", boom_loop)
        monkeypatch.setattr(mod, "post_gen_sanity_checks", lambda *a, **k: None)
        monkeypatch.setattr(mod, "upload_dataset_directory", lambda **k: None)

        # --allow-missing-drift-summary dropped in plan v2 hot-fix (2026-05-25).
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "issue_377_generate_incontext_corpus.py",
                "--no-upload",
            ],
        )
        rc = mod.main()
        assert rc == 0

        # Per-domain files unchanged on disk.
        for d in mod.INCONTEXT_DOMAINS:
            assert (sandbox / f"conversations_{d.name}.jsonl").exists()

    def test_no_resume_flag_forces_full_run(self, tmp_path, monkeypatch):
        mod = _load_script_as_module(INCONTEXT_SCRIPT, "_issue377_incontext_under_test_no_resume")

        sandbox = tmp_path / "issue377_incontext"
        monkeypatch.setattr(mod, "DATA_DIR", sandbox)
        monkeypatch.setattr(mod, "OUTPUT_PATH", sandbox / "incontext_conversations.jsonl")
        monkeypatch.setattr(mod, "SEED_CACHE_PATH", sandbox / "persona_topic_seeds_incontext.json")

        _write_per_domain_checkpoints(sandbox, list(mod.INCONTEXT_DOMAINS), corpus_tag="incontext")

        seed_called: list[bool] = []

        def fake_seed(_domains, *, cache_path, custom_id_prefix):
            seed_called.append(True)
            return {d.name: _fake_personas_for_domain(d.name) for d in _domains}

        monkeypatch.setattr(mod, "seed_personas_and_topics", fake_seed)

        loop_calls: list[str] = []

        def fake_loop(domain, _personas, *, custom_id_prefix, n_turns, rotation_seed):
            loop_calls.append(domain.name)
            return _fake_conversations_for_domain(domain.name, n_turns=n_turns)

        monkeypatch.setattr(mod, "run_conversation_loop", fake_loop)
        monkeypatch.setattr(mod, "post_gen_sanity_checks", lambda *a, **k: None)
        monkeypatch.setattr(mod, "upload_dataset_directory", lambda **k: None)

        # --allow-missing-drift-summary dropped in plan v2 hot-fix (2026-05-25).
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "issue_377_generate_incontext_corpus.py",
                "--no-upload",
                "--no-resume",
            ],
        )
        rc = mod.main()
        assert rc == 0

        assert seed_called
        assert set(loop_calls) == {d.name for d in mod.INCONTEXT_DOMAINS}
