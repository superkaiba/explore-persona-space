"""Issue #742 round-12 regression tests — three DISJOINT correctness holes in the
round-11 crash-safe-resume fix, plus two cheap concerns.

Round-11 correctly PERSISTS dispatch state, but:

  * Blocker 1 (``judge-rerun-var-judge-collapse``): the per-cell dispatch-dir key was
    ``{col_id}__{content_hash}`` — order-invariant in the completions. The judge rerun
    calls the judge R times on the SAME completions to measure ``Var_judge`` (judge
    re-labeling stochasticity). Same completions → same content hash → same dir → rerun 0
    writes ``scores.json`` and reruns 1..R-1 fast-resume it → all R rates IDENTICAL →
    ``Var_judge ≡ 0``. The fix threads ``rerun_idx`` into the key so each rerun submits an
    INDEPENDENT batch.
  * Blocker 2 (``SIGTERM-canonical-file-corruption``): on SIGTERM the outer loop breaks,
    ``run()`` merges whatever partials exist and returns, and ``main()`` writes the
    CANONICAL ``stage0_judge_variance.json`` at exit 0 even when only a SUBSET of cells
    completed. The fix asserts every requested cell has a valid partial before writing;
    on any shortfall (or if SIGTERM was requested) it raises → non-zero exit → no
    canonical file.
  * Blocker 3 (``order-invariant-hash-vs-order-derived-custom-id``): the content hash is
    order-invariant but ``judge_completions_batch`` assigns ``custom_id``s by iteration
    ORDER. Same content in a different order shares the content-hash dir but re-derives a
    DIFFERENT ``custom_id → completion`` map → the stored verdicts would attach to the
    WRONG completions. The fix persists a ``custom_id_map.json`` and validates it on
    resume, refusing reuse on mismatch.
  * Concern 4 (``corrupt-partial-skip-forever``): a stale wrong-shape partial was skipped
    forever at startup (``partial.exists()``) yet dropped at merge → silently vanished. The
    fix shape-validates the partial at skip time and quarantines + re-runs an invalid one.
  * Concern 5 (``model-not-in-key``): the judge model was absent from the dispatch-dir key
    → a re-run with a different judge model reused the old model's scores. The fix folds a
    judge-model hash into the key.
  * Concern 6: a real SIGTERM signal test (not just a resume-by-rerunning-one-behavior
    emulation) proving an interrupted run writes NO canonical file + exits non-zero.

No real API calls: ``judge_completions_batch`` is monkeypatched to a stub that writes the
real ``scores.json`` shape (``all_scores`` + ``routing.path='batch'``).
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

dc = importlib.import_module("explore_persona_space.analysis.issue_742_decoding_ceiling")


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _gen_cell(n_probes: int = 2, n_comps: int = 3) -> dict:
    """A tiny #658-shaped cell in the ``cells[i].completions[j]["text"]`` schema."""
    return {
        "context_id": "ctx_a",
        "cells": [
            {
                "probe": f"probe_{p}",
                "completions": [
                    {"text": f"probe{p} completion{k}", "logp_norm": 0.0} for k in range(n_comps)
                ],
            }
            for p in range(n_probes)
        ],
    }


def _make_stochastic_batch_judge_stub(counter: dict) -> object:
    """A stub whose ``judge_completions_batch`` writes a DIFFERENT verdict pattern on each
    submit — so two independent dispatches over the same completions yield DIFFERENT rates
    (the real judge stochasticity ``Var_judge`` measures). Increments ``counter['submits']``
    once per call. NO API contact.
    """

    class _StochasticStub:
        @staticmethod
        def judge_completions_batch(
            completions,
            *,
            judge_system_prompt,
            format_user_msg,
            judge_model,
            cache_dir,
            save_raw,
            checkpoint_dir,
            threshold_base,
        ):
            counter["submits"] += 1
            submit = counter["submits"]
            all_scores: dict[str, dict] = {}
            # Vary the positive COUNT (not just the pattern) across submits so the judged
            # RATE strictly changes per independent dispatch — the first `submit`-many items
            # (in enumeration order) are positive. submit=1 -> 1 positive, submit=2 -> 2,
            # so two dispatches over the same completions yield DIFFERENT rates (real
            # judge stochasticity -> Var_judge > 0). NO API contact.
            item = 0
            for _persona, q_to_texts in completions.items():
                for q_idx, (_question, comps) in enumerate(q_to_texts.items()):
                    for comp_idx, _text in enumerate(comps):
                        cid = f"cell__{q_idx:05d}__{comp_idx:02d}"
                        positive = item < submit
                        all_scores[cid] = {"misaligned": positive}
                        item += 1
            save_raw = Path(save_raw)
            save_raw.parent.mkdir(parents=True, exist_ok=True)
            save_raw.write_text(
                json.dumps(
                    {
                        "per_persona": {},
                        "all_scores": all_scores,
                        "routing": {"path": "batch", "n_items": len(all_scores)},
                    }
                )
            )
            return {}

    return _StochasticStub()


@pytest.fixture
def stub_stochastic(monkeypatch):
    """Route ``import_module('explore_persona_space.eval.batch_judge')`` inside ``dc`` to a
    STOCHASTIC counting stub. Returns the submit counter."""
    counter = {"submits": 0}
    stub = _make_stochastic_batch_judge_stub(counter)
    real_import = importlib.import_module

    def _fake_import(name, *args, **kwargs):
        if name == "explore_persona_space.eval.batch_judge":
            return stub
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    return counter


# --------------------------------------------------------------------------- #
# Test 2 (Blocker 1) — rerun_idx keying: R reruns submit independently          #
# --------------------------------------------------------------------------- #
def test_rerun_idx_gives_independent_submits_and_nonzero_variance(tmp_path, stub_stochastic):
    """R=3 independent judge reruns over the SAME completions → 3 DISTINCT submits (not 1
    submit + 2 fast-resumes) → 3 rate values that are NOT all identical (so Var_judge > 0).

    This is the direct regression for the var-judge-collapse blocker: without rerun_idx in
    the dispatch-dir key, reruns 1 and 2 would fast-resume rerun 0's scores.json → 1 submit
    → 3 identical rates → Var_judge == 0.
    """
    gen = _gen_cell(n_probes=2, n_comps=3)
    rates = []
    for rerun_idx in range(3):
        res = dc.per_behavior_judge_rate(
            gen,
            behavior="broad_em",
            judge_model="m",
            state_root=tmp_path,
            rerun_idx=rerun_idx,
        )
        rates.append(res["rate"])

    # 3 INDEPENDENT dispatches — each rerun_idx keys its own dir, no fast-resume across them.
    assert stub_stochastic["submits"] == 3, (
        "each of the 3 reruns must submit its own batch; a shared cache dir would collapse "
        f"to 1 submit — got {stub_stochastic['submits']}"
    )
    # 3 distinct per-rerun dirs on disk (rerun tag differs), each with its own scores.json.
    cell_dirs = sorted(dc._judge_rerun_state_root(tmp_path).glob("broad_em__r*"))
    tags = {d.name.split("__")[1] for d in cell_dirs}
    assert tags == {"r0", "r1", "r2"}, tags
    for d in cell_dirs:
        assert (d / "scores.json").exists()

    # The rates are NOT all identical -> Var_judge (variance across reruns) is > 0.
    assert len(set(rates)) > 1, f"reruns must not all return the same rate (Var_judge>0): {rates}"

    # And a REPEAT of the same rerun_idx DOES fast-resume (no extra submit) — the per-rerun
    # crash-safe resume still holds.
    dc.per_behavior_judge_rate(
        gen, behavior="broad_em", judge_model="m", state_root=tmp_path, rerun_idx=0
    )
    assert stub_stochastic["submits"] == 3, "re-running the same rerun_idx must fast-resume"


def test_judge_reruns_for_cell_wires_distinct_rerun_dirs(tmp_path, stub_stochastic, monkeypatch):
    """The scripts-level ``_judge_reruns_for_cell`` (the production caller) threads a DISTINCT
    rerun_idx per rerun AND distinct sentinels for the two generation-half passes, so a full
    cell run over R=2 reruns produces 2 rerun dirs + 2 generation-half dirs (r-1, r-2) — never
    a single collapsed dir. Var_judge (across-rerun variance) must therefore be > 0."""
    jr = importlib.import_module("issue742_judge_rerun")

    dest = tmp_path / "snapshot"
    jr.seed_synthetic_snapshot(dest, genre="betley", behavior="broad_em", n_contexts=1)

    # Point the real Batch dispatch at our state root (default is repo_root()).
    monkeypatch.setattr(dc, "_judge_rerun_state_root", lambda _r=None: tmp_path / "state")

    rerun_rates, _gen_first, _gen_second = jr._judge_reruns_for_cell(
        genre="betley",
        behavior="broad_em",
        snapshot_dir=dest,
        r_rerun=2,
        j_completions=6,
        seed=7428,
        judge_fn=None,  # real Batch dispatch (stubbed) — exercises rerun_idx keying
    )
    state = tmp_path / "state"
    rerun_dirs = sorted(p.name for p in state.glob("broad_em__r0__*")) + sorted(
        p.name for p in state.glob("broad_em__r1__*")
    )
    assert len(rerun_dirs) == 2, rerun_dirs  # r0 + r1, distinct dirs
    # the two generation-half passes get their own sentinel dirs, never colliding
    assert list(state.glob("broad_em__r-1__*")), "gen first-half needs its own dir"
    assert list(state.glob("broad_em__r-2__*")), "gen second-half needs its own dir"

    # across-rerun variance of the per-context rate must be measurable (> 0 for at least
    # one context) with the stochastic stub — the whole point of independent reruns.
    import numpy as np

    R = np.stack(rerun_rates, axis=0)
    assert R.shape[0] == 2
    assert float(np.mean(np.var(R, axis=0))) > 0.0, (
        f"Var_judge must be > 0 with independent reruns; rerun_rates={rerun_rates}"
    )


# --------------------------------------------------------------------------- #
# Test 4 (Blocker 3) — order-derived custom_id map guards the order-invariant   #
# content hash                                                                  #
# --------------------------------------------------------------------------- #
def test_custom_id_map_detects_reordered_content_and_refuses_reuse(tmp_path, stub_stochastic):
    """Same completion SET in a DIFFERENT enumeration order shares the (order-invariant)
    content-hash dir. The persisted custom_id→content map must catch the reorder and REFUSE
    reuse (re-dispatch fresh) rather than attaching verdicts to the wrong completions.
    """
    # Two probes with DISTINCT single completions so a reorder actually changes the
    # custom_id → content mapping (probe order flips q_idx assignment).
    gen_ab = {
        "context_id": "ctx",
        "cells": [
            {"probe": "A", "completions": [{"text": "alpha", "logp_norm": 0.0}]},
            {"probe": "B", "completions": [{"text": "beta", "logp_norm": 0.0}]},
        ],
    }
    gen_ba = {
        "context_id": "ctx",
        "cells": [
            {"probe": "B", "completions": [{"text": "beta", "logp_norm": 0.0}]},
            {"probe": "A", "completions": [{"text": "alpha", "logp_norm": 0.0}]},
        ],
    }
    # sanity: the two gens share the order-invariant content hash (same (probe,text) set)
    flat_ab = [{"probe": "A", "text": "alpha"}, {"probe": "B", "text": "beta"}]
    flat_ba = [{"probe": "B", "text": "beta"}, {"probe": "A", "text": "alpha"}]
    assert dc._judge_rerun_content_hash(flat_ab) == dc._judge_rerun_content_hash(flat_ba)

    dc.judge_column_via_batch_judge("broad_em", gen_ab, "m", state_root=tmp_path, rerun_idx=0)
    assert stub_stochastic["submits"] == 1
    # the reordered gen shares the dir (same content hash + same rerun/model) but its
    # custom_id map differs -> the guard refuses the stale scores.json -> a fresh dispatch.
    dc.judge_column_via_batch_judge("broad_em", gen_ba, "m", state_root=tmp_path, rerun_idx=0)
    assert stub_stochastic["submits"] == 2, (
        "a reordered enumeration must NOT reuse the persisted scores.json (custom_id map "
        "mismatch); it must re-dispatch"
    )

    # An IDENTICAL-order repeat still fast-resumes (the map matches) — no false re-dispatch.
    dc.judge_column_via_batch_judge("broad_em", gen_ba, "m", state_root=tmp_path, rerun_idx=0)
    assert stub_stochastic["submits"] == 2, "identical order must fast-resume (map matches)"


def test_custom_id_content_map_is_order_sensitive():
    """``_custom_id_content_map`` maps a fixed custom_id to the CONTENT at that ORDER slot, so
    reordering the completions changes the map even though the content SET is identical."""
    comp_ab = {"cell": {"A": ["alpha"], "B": ["beta"]}}
    comp_ba = {"cell": {"B": ["beta"], "A": ["alpha"]}}
    m_ab = dc._custom_id_content_map(comp_ab)
    m_ba = dc._custom_id_content_map(comp_ba)
    # same custom_ids, but the content sha behind cell__00000__00 differs (A vs B)
    assert set(m_ab) == set(m_ba)
    assert m_ab["cell__00000__00"] != m_ba["cell__00000__00"], (m_ab, m_ba)


# --------------------------------------------------------------------------- #
# Test 6 (Concern 5) — judge model in the dispatch-dir key                      #
# --------------------------------------------------------------------------- #
def test_different_judge_model_does_not_reuse_cached_scores(tmp_path, stub_stochastic):
    """Same completions + same rerun_idx but a DIFFERENT judge_model must NOT reuse the prior
    model's scores.json — the model hash is part of the dispatch-dir key."""
    gen = _gen_cell(n_probes=2, n_comps=2)
    dc.judge_column_via_batch_judge(
        "broad_em", gen, "claude-sonnet-4-5-20250929", state_root=tmp_path, rerun_idx=0
    )
    assert stub_stochastic["submits"] == 1
    dc.judge_column_via_batch_judge(
        "broad_em", gen, "some-other-judge-model", state_root=tmp_path, rerun_idx=0
    )
    assert stub_stochastic["submits"] == 2, "a different judge model must dispatch fresh"
    # two distinct model-keyed dirs on disk
    dirs = list(dc._judge_rerun_state_root(tmp_path).glob("broad_em__r0__m*"))
    model_keys = {d.name.split("__")[2] for d in dirs}
    assert len(model_keys) == 2, model_keys
    # same model again fast-resumes
    dc.judge_column_via_batch_judge(
        "broad_em", gen, "claude-sonnet-4-5-20250929", state_root=tmp_path, rerun_idx=0
    )
    assert stub_stochastic["submits"] == 2, "same model must fast-resume"


# --------------------------------------------------------------------------- #
# Test 5 (Concern 4) — corrupt/wrong-shape partial is quarantined + re-run      #
# --------------------------------------------------------------------------- #
def test_corrupt_partial_is_quarantined_and_cell_reruns(tmp_path, monkeypatch):
    """A stale WRONG-SHAPE partial (missing the _decompose_variance keys) must NOT be
    treated as completed at startup — the cell would be skipped-forever yet dropped at merge.
    The startup shape-validation quarantines it and RE-RUNS the cell.
    """
    jr = importlib.import_module("issue742_judge_rerun")

    genres = ["betley"]
    behaviors = ["broad_em"]
    dest = tmp_path / "snapshot"
    jr.seed_synthetic_snapshot(dest, genre=genres[0], behavior=behaviors[0])
    out_dir = tmp_path / "out"

    # Pre-write a WRONG-SHAPE partial: valid JSON, correct genre/behavior, but result missing
    # the decomposition keys (a truncated / pre-decomposition write).
    partial = jr._partial_path(out_dir, genres[0], behaviors[0])
    jr._atomic_write_json(
        partial, {"genre": genres[0], "behavior": behaviors[0], "result": {"garbage": 1}}
    )
    assert not jr._partial_is_valid(partial)

    judged_cells: list[tuple[str, str]] = []
    real_reruns = jr._judge_reruns_for_cell

    def _spy(*, genre, behavior, **kwargs):
        judged_cells.append((genre, behavior))
        return real_reruns(genre=genre, behavior=behavior, **kwargs)

    monkeypatch.setattr(jr, "_judge_reruns_for_cell", _spy)

    result = jr.run(
        genres=genres,
        behaviors=behaviors,
        r_rerun=2,
        j_completions=4,
        dry_run=False,
        judge_fn=jr.make_counting_judge(),
        dest_override=dest,
        skip_snapshot=True,
        out_dir=out_dir,
    )
    # the corrupt partial must NOT have been trusted — the cell was re-judged
    assert judged_cells == [(genres[0], behaviors[0])], (
        "a wrong-shape partial must be quarantined + the cell re-run, not skipped-forever"
    )
    # a quarantine breadcrumb exists, and the partial is now VALID
    quarantined = list(jr._partial_dir(out_dir).glob(f"{genres[0]}__{behaviors[0]}.json.corrupt-*"))
    assert quarantined, "the invalid partial must be quarantined (kept for forensics)"
    assert jr._partial_is_valid(partial), "the re-run must write a well-formed partial"
    assert "sqrt_r_yy_honest" in result["judge_variance"][genres[0]][behaviors[0]]


# --------------------------------------------------------------------------- #
# Test 3 (Blocker 2) — SIGTERM canonical-file guard (in-process)                #
# --------------------------------------------------------------------------- #
def test_run_raises_on_shutdown_before_canonical_write(tmp_path, monkeypatch):
    """If the SIGTERM flag is set mid-loop (handler fired), ``run()`` must RAISE at the
    completeness guard rather than return a "complete" result — so ``main()`` never writes
    the canonical file for a partial run.
    """
    jr = importlib.import_module("issue742_judge_rerun")

    genres = ["betley"]
    behaviors = ["broad_em", "refusal"]
    dest = tmp_path / "snapshot"
    for beh in behaviors:
        jr.seed_synthetic_snapshot(dest, genre=genres[0], behavior=beh)
    out_dir = tmp_path / "out"

    real_reruns = jr._judge_reruns_for_cell

    def _rerun_then_request_shutdown(*, genre, behavior, **kwargs):
        # complete the FIRST cell, then simulate the SIGTERM handler firing so the loop
        # breaks before the second cell (mid-run interruption).
        out = real_reruns(genre=genre, behavior=behavior, **kwargs)
        jr._SHUTDOWN_REQUESTED["flag"] = True
        return out

    monkeypatch.setattr(jr, "_judge_reruns_for_cell", _rerun_then_request_shutdown)
    # ensure a clean flag at entry (module-global state)
    jr._SHUTDOWN_REQUESTED["flag"] = False

    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        jr.run(
            genres=genres,
            behaviors=behaviors,
            r_rerun=2,
            j_completions=4,
            dry_run=False,
            judge_fn=jr.make_counting_judge(),
            dest_override=dest,
            skip_snapshot=True,
            out_dir=out_dir,
        )
    # reset the module-global flag + restore the real rerun fn so the fresh rerun below is
    # NOT interrupted (the patch would otherwise re-fire the shutdown on the resumed cell).
    jr._SHUTDOWN_REQUESTED["flag"] = False
    monkeypatch.setattr(jr, "_judge_reruns_for_cell", real_reruns)

    # the FIRST cell's partial exists (checkpointed), the SECOND does not — proving resume
    # will complete the outstanding cell on the next launch.
    assert jr._partial_path(out_dir, genres[0], behaviors[0]).exists()
    assert not jr._partial_path(out_dir, genres[0], behaviors[1]).exists()

    # A fresh run (no shutdown) completes the outstanding cell and returns cleanly.
    result = jr.run(
        genres=genres,
        behaviors=behaviors,
        r_rerun=2,
        j_completions=4,
        dry_run=False,
        judge_fn=jr.make_counting_judge(),
        dest_override=dest,
        skip_snapshot=True,
        out_dir=out_dir,
    )
    assert set(result["judge_variance"][genres[0]]) == set(behaviors)


# --------------------------------------------------------------------------- #
# Test 3 / Concern 6 — REAL SIGTERM signal: no canonical file + non-zero exit   #
# --------------------------------------------------------------------------- #
def test_real_sigterm_writes_no_canonical_file_then_rerun_completes(tmp_path):
    """Concern 6: an ACTUAL SIGTERM (not a resume-emulation) after N<total cells complete →
    (a) the canonical stage0_judge_variance.json is NOT written, (b) exit is non-zero, and a
    fresh rerun (c) completes the remaining cells + writes the canonical file cleanly.

    Runs a real subprocess that:
      * seeds a 2-behavior synthetic snapshot,
      * drives ``run()`` with a SLOW counting judge that writes a heartbeat + sleeps per
        cell (so the parent can time the SIGTERM after the first cell checkpoints), then
      * emulates ``main()``'s contract: writes the canonical file ONLY if ``run()`` returns.
    The installed SIGTERM handler flips the graceful-stop flag; the completeness guard then
    raises so the subprocess exits non-zero and the canonical write is never reached.
    """
    out_dir = tmp_path / "out"
    dest = tmp_path / "snapshot"
    heartbeat = tmp_path / "heartbeat"
    canonical = out_dir / "stage0_judge_variance.json"

    driver = textwrap.dedent(
        f"""
        import json, sys, time
        from pathlib import Path
        PROJECT_ROOT = Path({str(PROJECT_ROOT)!r})
        for sub in ("scripts", "src"):
            p = str(PROJECT_ROOT / sub)
            if p not in sys.path:
                sys.path.insert(0, p)
        import importlib
        jr = importlib.import_module("issue742_judge_rerun")

        out_dir = Path({str(out_dir)!r})
        dest = Path({str(dest)!r})
        heartbeat = Path({str(heartbeat)!r})
        genres = ["betley"]
        behaviors = ["broad_em", "refusal"]
        for beh in behaviors:
            jr.seed_synthetic_snapshot(dest, genre=genres[0], behavior=beh)

        base_judge = jr.make_counting_judge()
        first_cell = {{"seen": False}}

        def slow_judge(col_id, gen, model):
            # heartbeat + a short sleep AFTER the first cell so the parent can SIGTERM us
            # between cell 1 and cell 2 (the graceful between-cell stop point).
            res = base_judge(col_id, gen, model)
            if not first_cell["seen"]:
                first_cell["seen"] = True
            else:
                heartbeat.write_text("cell2-started")
            time.sleep(1.5)
            return res

        # install the SIGTERM handler exactly as run() does for the real path.
        jr._install_sigterm_handler()
        # signal the parent that cell 1 is about to be judged
        heartbeat.write_text("starting")

        result = jr.run(
            genres=genres, behaviors=behaviors, r_rerun=2, j_completions=4,
            dry_run=False, judge_fn=slow_judge, dest_override=dest,
            skip_snapshot=True, out_dir=out_dir,
        )
        # main()'s contract: write the canonical file ONLY if run() returned (never on the
        # raise path). If we reach here after a SIGTERM, the guard failed to fire.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stage0_judge_variance.json").write_text(json.dumps(result))
        print("WROTE_CANONICAL")
        """
    )
    driver_path = tmp_path / "driver.py"
    driver_path.write_text(driver)

    env = {**os.environ}
    proc = subprocess.Popen(
        ["uv", "run", "python", str(driver_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # wait until the first cell's partial appears (cell 1 checkpointed) then SIGTERM.
    partial1 = jr_partial(out_dir, "betley", "broad_em")
    deadline = time.time() + 30
    while time.time() < deadline and not partial1.exists():
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    assert partial1.exists() or proc.poll() is not None, "cell 1 never checkpointed"
    proc.send_signal(signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=30)

    # (a) canonical file NOT written; (b) non-zero exit
    assert not canonical.exists(), (
        f"canonical file must NOT be written on SIGTERM\n{stdout}\n{stderr}"
    )
    assert "WROTE_CANONICAL" not in stdout, stdout
    assert proc.returncode != 0, f"interrupted run must exit non-zero (rc={proc.returncode})"
    assert "INCOMPLETE" in stderr, f"expected the completeness-guard RuntimeError\n{stderr}"

    # (c) a fresh rerun (no signal) completes the outstanding cell + writes the canonical file.
    proc2 = subprocess.run(
        ["uv", "run", "python", str(driver_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc2.returncode == 0, f"rerun must complete cleanly\n{proc2.stdout}\n{proc2.stderr}"
    assert canonical.exists(), "rerun must write the canonical file"
    data = json.loads(canonical.read_text())
    assert set(data["judge_variance"]["betley"]) == {"broad_em", "refusal"}


def jr_partial(out_dir: Path, genre: str, behavior: str) -> Path:
    """Module-level shim so the subprocess-timing loop can compute the partial path without
    importing the script module into the parent test process before the subprocess seeds it."""
    jr = importlib.import_module("issue742_judge_rerun")
    return jr._partial_path(out_dir, genre, behavior)
