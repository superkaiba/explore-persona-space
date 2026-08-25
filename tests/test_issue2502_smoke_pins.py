"""Issue #2502 rv2-u4: committed pins for the two round-1 u4 smoke-caught fixes.

Both fixes were caught by the per-phase e2e smoke in round 1 and fixed in
production code; these tests pin them so a refactor cannot silently strip
either invariant (experiment-implementer checklist item 8):

  (a) corpus.build_report tolerates probe-mode rows WITHOUT a "split" field
      (probe runs BEFORE assign_splits; a bare r["split"] KeyError'd every
      --probe invocation) and counts them under "unassigned".
  (b) reliability.run_subset passes hf_missing_of a repo PATH PREFIX as
      ``scope`` (SUBSET_PREFIX), never a bare label — a non-prefix scope made
      verify_repo_paths_uploaded raise "expected paths outside path_in_repo"
      on EVERY subset invocation.

No network: (a) is pure; (b) monkeypatches the gen_capture hf_missing_of
module attribute (run_subset resolves it at call time via the GC module).

Round-5 pins (gated-source skip semantics; the Step 6d.0-bis tiny-real probe
crash on a gated fallback-less dataset):

  (c) a fallback-LESS source whose PRIMARY read fails an ACCESS-class error
      is SKIPPED loud (recorded in build_report's ``skipped_sources`` + the
      per-config ``skipped_gated_no_access`` counter), never a whole-build
      crash, and the budget re-scales across survivors only;
  (d) a gated source WITH a fallback still falls over to it (unchanged);
  (e) kept==0 on an ACCESSIBLE source still raises (the _stream_stage
      fail-loud is not swallowed by the skip path);
  (f) an all-skipped (empty) corpus and a whole regime-class family with zero
      staged rows each still raise.

The round-5 tests fake ONLY the HF network boundary (_resolve_dataset_revision
+ CS._hf_stream, both signature-mirroring defs); the real production chain
(stage_source -> _stage_one_config -> CS._stream_stage incl. keep_fn,
fingerprints, checkpoint writes, the kept==0 raise) executes unmodified.
Datasets are referenced by synthetic org/name ids only — no corpus text.

Round-6 pins (concern-closure round):

  (g) transient-http-misclassified-as-access-skip: a transient 503 (and any
      408/429/5xx / status-less error) on a fallback-less source PROPAGATES
      — never a skip — and on a gated-WITH-fallback source PROPAGATES too
      (never a fallback fallover on infra noise); a bare 403 HfHubHTTPError
      (permanent access denial) still takes the loud-skip path;
  (h) gated-sources-absent-corpus-composition: build_report carries a
      per-source ``source_roster`` (status + per-tag realized counts) +
      ``budget_redistribution`` disclosure;
  (i) moderation-derived-regime-guard-gap: a selective probe of a
      moderation-split source declares the DERIVED near-distribution class
      to the aggregate regime guard (zero flagged rows -> loud raise);
  (j) decide-phase-idempotency: run_decide SKIPs loud on a matching
      fits/.p4_done (returns the persisted decision), --force re-runs, and a
      sentinel without decision.json fails loud;
  (k) publish-none-hardening: '--publish none' with the canonical production
      out-root is REFUSED absent --allow-local-only (helper + main wiring),
      and the .p4_done sentinel publishes through the SELECTED backend(s) —
      git modes commit it, hf modes upload it (_publish_sentinel routing;
      fakes only at the git/HF boundary, signature-mirroring).

Round-7 pins (concern-closure round):

  (j3-j5) decide-force-stale-sentinel-window: the .p4_done sentinel is
      CONTENT-BOUND — per-file sha256 over decision.json + both models'
      fits_summary/percontext_recon (the _load_model_artifacts input set) —
      and a mismatched or digest-less sentinel fails loud at skip time (j3);
      --force atomically invalidates the sentinel BEFORE any recompute
      (quarantined as .p4_done.stale), so a mid-recompute/mid-publish crash
      resumes into a REAL re-decide, never a stale-done skip (j4); the
      no-force skip re-invokes the REAL _publish_sentinel body as an
      idempotent publish retry (j5).
  (k1, rewritten) publish-none-main-path-hollow-pin: the fit/decide
      '--publish none' + canonical-out-root refusal is EXECUTED through the
      real main() argv path with tripwired phase bodies, and
      --allow-local-only escapes into the phase body for real — replacing
      the former inspect.getsource call-count, which never executed the
      refusal.

Round-8 pins (decide-force-stale-sentinel-window residual closure):

  (j6a/j6b) leg 1 — missing-input verify hole: every _decision_fingerprint
      member is a REQUIRED decide input, so an ABSENT file raises naming its
      relative path (j6a), and the pre-fix hole itself — a legacy sentinel
      that RECORDED a member as absent, compared against a still-absent file
      at skip time — fails loud at run_decide instead of verifying cleanly
      (j6b);
  (j7) leg 2 — skip telemetry bound to decision.json: a sentinel-ONLY
      verdict mutation (content_sha256 intact, fingerprint still verifies)
      raises on sentinel/decision disagreement instead of logging the
      mutated value, and the healthy-path skip log reports the verdict read
      from decision.json (the authoritative artifact).

Round-9 pins (source-registry split-spec class; the live P0 crash
``ValueError: Bad split: train`` on jbb_behaviors, whose `behaviors` config
offers ['harmful','benign'] only):

  (l) split-absent LOUD handler: a bad-split ValueError escaping
      load_dataset is re-raised as a RuntimeError NAMING source_tag +
      declared split + dataset id (registry-defect signal), chained from the
      original — never a silent skip/remap, and never routed into the
      round-5 gated-skip;
  (m) registry-wide preflight (verify_declared_splits): ALL declared
      (config, split) defects aggregate into ONE raise (anti-whack-a-mole:
      a relaunch surfaces every registry defect at once); gated sources are
      unverifiable-not-fatal; transient HF errors propagate; data_files
      sources are 'train'-fixed by construction;
  (n) registry shape: the corrected specs (jbb both-splits, MASK 6 configs @
      test, xstest test, beavertails default-config 30k_train,
      model_written_evals data_dir=sycophancy, pippa data_files off the dead
      script, riddle_sense parquet-branch revision_ref, legalbench
      train+test) are pinned so a refactor cannot silently regress them;
  (o) multi-split staging through the REAL chain: two splits of one config
      stage to DISTINCT checkpoint files/fingerprints with the keep-cap
      CUMULATIVE across attempts and @split-suffixed counter keys;
  (p) preflight wiring: run_pipeline invokes verify_declared_splits before
      any staging, and --skip-split-preflight (offline-smoke escape) skips
      it.

Round-11 pins (proactive secret scrub; the live P0 upload refusal —
secret_scrub.assert_upload_clean found real-secret-grade strings in the
assembled corpus.jsonl and refused the upload):

  (q1) scrub_corpus_secrets (the builder's scrub step, reusing
      secret_scrub.scrub_file — shared detection with the upload gate)
      redacts a planted synthetic secret-grade string to a SAME-LENGTH
      placeholder (file byte-length unchanged, JSONL still parses, sibling
      fields intact), records count + per-pattern counts, and preserves
      dummy-filtered benign placeholders (X-run / YOUR_API_KEY) unredacted;
  (q2) the BUILD path (run_pipeline, no --probe) scrubs the PERSISTED
      corpus.jsonl before the upload step and records ``secrets_scrubbed``
      durably in dedup_report.json; the scrubbed row's stored context_sha is
      NOT recomputed (it documents the pre-scrub source-text identity that
      downstream fingerprints read), while clean rows' shas still match
      their text. Synthetic planted strings only — never a real secret,
      never real corpus text.

Model-B render fix pin (transformers 5.x return_dict flip; #2502 + #2378):

  (r) render_prompt_ids yields a FLAT list of int token ids under BOTH
      apply_chat_template return conventions: transformers 4.57.6 (Model A
      repo-standard venv) defaults tokenize=True to return_dict=False (id
      list), transformers 5.15.1 (Model B pod2378-venv) flips the default
      to return_dict=True (BatchEncoding dict) — pre-fix the listcomp
      int()'d the DICT KEYS (ValueError on 'input_ids'). Synthetic fake
      tokenizers only; no model download, no corpus text.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

CP = importlib.import_module("issue2502_corpus")
FT = importlib.import_module("issue2502_fits")
GC = importlib.import_module("issue2502_gen_capture")
RL = importlib.import_module("issue2502_reliability")


def test_build_report_probe_row_without_split_counts_unassigned():
    """Pin (a): a final row lacking "split" must not raise (probe mode)."""
    report = CP.build_report(
        pre_dedup_per_source={"s": 1},
        dedup_report={"n_in": 1, "n_confirmed_dropped": 0, "n_had_lsh_candidate": 0},
        post_dedup_per_source={"s": 1},
        allocation={"s": 1},
        final_rows=[{"regime_class": "ordinary", "realism_tier": 1, "source_tag": "s"}],
        regime_table={"s": "ordinary"},
        stream_counters={},
        budget=1,
    )
    assert report["split_counts"] == {"unassigned": 1}
    assert report["n_final"] == 1


def test_run_subset_scopes_hf_listing_to_subset_prefix(tmp_path, monkeypatch):
    """Pin (b): run_subset's presence probe uses SUBSET_PREFIX as the repo
    path-prefix scope and the exact corpus.jsonl dest path."""
    captured: dict = {}

    def fake_hf_missing_of(paths, *, scope):
        captured["paths"] = list(paths)
        captured["scope"] = scope
        return []  # nothing missing -> the already-uploaded skip path

    monkeypatch.setattr(GC, "hf_missing_of", fake_hf_missing_of)
    args = RL.build_parser().parse_args(
        ["--phase", "subset", "--work-dir", str(tmp_path / "subset_work")]
    )
    res = RL.run_subset(args)
    assert res.get("skipped") is True, res
    assert captured["scope"] == RL.SUBSET_PREFIX
    assert captured["paths"] == [f"{RL.SUBSET_PREFIX}/corpus.jsonl"]


# ---------------------------------------------------------------------------
# Round-5 pins (c)-(f): gated-source skip semantics.
# ---------------------------------------------------------------------------


def _spec(tag, dataset_id, regime, *, fallback=None, cap=10):
    """Minimal SourceSpec for the round-5 pins (scalar `text` field)."""
    return CP.SourceSpec(
        source_tag=tag,
        dataset_id=dataset_id,
        regime_class=regime,
        realism_tier=1,
        pre_dedup_cap=cap,
        text_fields=("text",),
        fallback_dataset_id=fallback,
    )


def _raw_rows(tag: str, n: int) -> list[dict]:
    """Distinct sha256-hex payloads so no LSH/exact-Jaccard near-dup fires."""
    return [
        {"text": f"{tag} synthetic context {hashlib.sha256(f'{tag}:{i}'.encode()).hexdigest()}"}
        for i in range(n)
    ]


def _install_hf_seams(
    monkeypatch,
    *,
    gated: set[str],
    raw_by_dataset: dict[str, list[dict]],
    raise_by_dataset: dict[str, BaseException] | None = None,
):
    """Fake ONLY the HF network boundary (revision resolution + row stream).

    The real production chain (stage_source -> _stage_one_config ->
    CS._stream_stage incl. its kept==0 raise, keep_fn, fingerprints,
    checkpoint writes) executes unmodified. Both fakes mirror the real
    call shapes (`_resolve_dataset_revision(dataset_id, revision_ref=None)`;
    `CS._hf_stream(dataset_id, config, split, revision=..., **kw)`).
    ``raise_by_dataset`` injects an arbitrary exception at the revision
    seam (round-6 transient-vs-permanent discrimination pins).
    """
    from datasets.exceptions import DatasetNotFoundError

    def fake_resolve(dataset_id: str, revision_ref: str | None = None) -> str:
        if raise_by_dataset and dataset_id in raise_by_dataset:
            raise raise_by_dataset[dataset_id]
        if dataset_id in gated:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' is a gated dataset on the Hub: access not granted"
            )
        return "0" * 40

    def fake_stream(dataset_id, config, split, **kwargs):
        assert kwargs.get("revision") == "0" * 40
        return iter(raw_by_dataset[dataset_id])

    monkeypatch.setattr(CP, "_resolve_dataset_revision", fake_resolve)
    monkeypatch.setattr(CP.CS, "_hf_stream", fake_stream)


def _probe_args(tmp_path, budget=10):
    # --skip-split-preflight: these pins fake the network at the revision /
    # row-stream seams only; the round-9 split preflight would call the REAL
    # datasets.get_dataset_split_names (network) — it has its own injected-fn
    # pins below (l/m/p) and stays out of the round-5/6 skip-semantics pins.
    return CP.build_argparser().parse_args(
        [
            "--probe",
            "--no-token-filter",
            "--skip-split-preflight",
            "--out-dir",
            str(tmp_path),
            "--budget",
            str(budget),
        ]
    )


def test_gated_no_fallback_source_skipped_and_recorded(monkeypatch, tmp_path):
    """Pin (c): a fallback-less gated source is SKIPPED (not raised), the skip
    is recorded in the report + probe artifact, and the budget re-scales
    across surviving sources only."""
    gated_id = "org/gated-no-access"
    specs = (
        _spec("gated_src", gated_id, "weird"),
        _spec("weird_ok", "org/open-weird", "weird"),
        _spec("ord_ok", "org/open-ord", "ordinary"),
    )
    _install_hf_seams(
        monkeypatch,
        gated={gated_id},
        raw_by_dataset={
            "org/open-weird": _raw_rows("weird_ok", 6),
            "org/open-ord": _raw_rows("ord_ok", 6),
        },
    )
    monkeypatch.setattr(CP, "SOURCES", specs)
    report = CP.run_pipeline(_probe_args(tmp_path))
    (skip,) = report["skipped_sources"]
    assert skip["source_tag"] == "gated_src"
    assert skip["dataset_id"] == gated_id
    assert skip["config"] == "default"
    assert "gated" in skip["reason"]
    # per-config counter marker survives in stream_counters
    assert report["stream_counters"]["gated_src"]["default"]["skipped_gated_no_access"] == 1
    # budget re-scaled across SURVIVORS only — no gated tag, exact total
    assert "gated_src" not in report["budget_allocation"]
    assert report["budget_allocation_total"] == 10  # min(budget=10, survivors' 12)
    # the probe artifact surfaces the skip durably (never silent)
    probe = json.loads((tmp_path / "probe_report.json").read_text())
    assert probe["skipped_sources"] == report["skipped_sources"]


def test_gated_with_fallback_still_falls_over(monkeypatch, tmp_path):
    """Pin (d): a gated primary WITH a fallback dataset still falls over to it
    gracefully — no skip marker, rows read from the fallback id."""
    spec = _spec("wc", "org/gated-primary", "ordinary", fallback="org/open-mirror")
    _install_hf_seams(
        monkeypatch,
        gated={"org/gated-primary"},
        raw_by_dataset={"org/open-mirror": _raw_rows("wc", 5)},
    )
    rows, ctr = CP.stage_source(
        spec, tmp_path, None, stream_cap=None, filter_language=False, seed=0
    )
    assert len(rows) == 5
    assert all(r["dataset_id"] == "org/open-mirror" for r in rows)
    assert not any(c.get("skipped_gated_no_access") for c in ctr.values())


def test_kept_zero_on_accessible_source_still_raises(monkeypatch, tmp_path):
    """Pin (e): kept==0 on an ACCESSIBLE source (a data-shape/filter bug) still
    fails loud — the skip path is scoped to access-class errors only."""
    spec = _spec("shape_bug", "org/open-wrong-shape", "ordinary")
    _install_hf_seams(
        monkeypatch,
        gated=set(),
        raw_by_dataset={"org/open-wrong-shape": [{"not_the_text_field": "x"} for _ in range(4)]},
    )
    with pytest.raises(RuntimeError, match="kept 0 rows"):
        CP.stage_source(spec, tmp_path, None, stream_cap=None, filter_language=False, seed=0)


def test_all_sources_skipped_empty_corpus_raises(monkeypatch, tmp_path):
    """Pin (f1): every source skipped -> empty corpus -> fail loud."""
    specs = (
        _spec("g1", "org/gated-a", "weird"),
        _spec("g2", "org/gated-b", "ordinary"),
    )
    _install_hf_seams(monkeypatch, gated={"org/gated-a", "org/gated-b"}, raw_by_dataset={})
    monkeypatch.setattr(CP, "SOURCES", specs)
    with pytest.raises(RuntimeError, match="kept 0 rows across all"):
        CP.run_pipeline(_probe_args(tmp_path))


def test_whole_regime_family_collapsed_raises(monkeypatch, tmp_path):
    """Pin (f2): a whole regime-class family with zero staged rows fails loud
    even when other families survive."""
    specs = (
        _spec("gated_weird", "org/gated-weird", "weird"),
        _spec("ord_ok", "org/open-ord", "ordinary"),
    )
    _install_hf_seams(
        monkeypatch,
        gated={"org/gated-weird"},
        raw_by_dataset={"org/open-ord": _raw_rows("ord_ok", 6)},
    )
    monkeypatch.setattr(CP, "SOURCES", specs)
    with pytest.raises(RuntimeError, match="regime class"):
        CP.run_pipeline(_probe_args(tmp_path))


# ---------------------------------------------------------------------------
# Round-6 pins (g)-(k): concern-closure round.
# ---------------------------------------------------------------------------


def _http_error(code: int, msg: str):
    """Bare HfHubHTTPError carrying a real requests.Response with the code."""
    import requests
    from huggingface_hub.utils import HfHubHTTPError

    resp = requests.Response()
    resp.status_code = code
    return HfHubHTTPError(msg, response=resp)


def test_transient_503_on_fallbackless_source_propagates(monkeypatch, tmp_path):
    """Pin (g1): a transient 5xx on a fallback-less source PROPAGATES — never
    a skip-and-redistribute (transient-http-misclassified-as-access-skip)."""
    from huggingface_hub.utils import HfHubHTTPError

    spec = _spec("outage_src", "org/open-but-down", "ordinary")
    _install_hf_seams(
        monkeypatch,
        gated=set(),
        raw_by_dataset={},
        raise_by_dataset={"org/open-but-down": _http_error(503, "503 Server Error: down")},
    )
    with pytest.raises(HfHubHTTPError):
        CP.stage_source(spec, tmp_path, None, stream_cap=None, filter_language=False, seed=0)


def test_transient_503_on_gated_with_fallback_propagates(monkeypatch, tmp_path):
    """Pin (g2): a transient 5xx on a WITH-fallback source PROPAGATES — never
    a silent fallback fallover (composition change) on infra noise."""
    from huggingface_hub.utils import HfHubHTTPError

    spec = _spec("wc", "org/primary-down", "ordinary", fallback="org/open-mirror")
    _install_hf_seams(
        monkeypatch,
        gated=set(),
        raw_by_dataset={"org/open-mirror": _raw_rows("wc", 3)},
        raise_by_dataset={"org/primary-down": _http_error(503, "503 Server Error: down")},
    )
    with pytest.raises(HfHubHTTPError):
        CP.stage_source(spec, tmp_path, None, stream_cap=None, filter_language=False, seed=0)


def test_bare_403_http_error_still_skips_loud(monkeypatch, tmp_path):
    """Pin (g3): a bare 403 HfHubHTTPError (permanent access denial outside
    the typed classes) still takes the loud-skip path on a fallback-less
    source — the discriminator narrows TRANSIENTS only."""
    spec = _spec("gated_403", "org/forbidden", "ordinary")
    _install_hf_seams(
        monkeypatch,
        gated=set(),
        raw_by_dataset={},
        raise_by_dataset={"org/forbidden": _http_error(403, "403 Forbidden: gated")},
    )
    rows, ctr = CP.stage_source(
        spec, tmp_path, None, stream_cap=None, filter_language=False, seed=0
    )
    assert rows == []
    assert ctr["default"]["skipped_gated_no_access"] == 1


def test_report_source_roster_and_budget_redistribution(monkeypatch, tmp_path):
    """Pin (h): the report carries a per-source composition roster (status +
    per-tag realized counts) and a budget-redistribution disclosure
    (gated-sources-absent-corpus-composition)."""
    gated_id = "org/gated-no-access"
    specs = (
        _spec("gated_src", gated_id, "weird"),
        _spec("weird_ok", "org/open-weird", "weird"),
        _spec("ord_ok", "org/open-ord", "ordinary"),
    )
    _install_hf_seams(
        monkeypatch,
        gated={gated_id},
        raw_by_dataset={
            "org/open-weird": _raw_rows("weird_ok", 6),
            "org/open-ord": _raw_rows("ord_ok", 6),
        },
    )
    monkeypatch.setattr(CP, "SOURCES", specs)
    report = CP.run_pipeline(_probe_args(tmp_path))
    roster = {row["source_tag"]: row for row in report["source_roster"]}
    assert set(roster) == {"gated_src", "weird_ok", "ord_ok"}
    skipped = roster["gated_src"]
    assert skipped["status"] == "skipped_gated_no_access"
    assert skipped["skipped_configs"] == ["default"]
    assert skipped["planned_pre_dedup_cap"] == 10
    assert skipped["pre_dedup_rows"] == {"gated_src": 0}
    assert skipped["allocated"] == {"gated_src": 0}
    for tag in ("weird_ok", "ord_ok"):
        assert roster[tag]["status"] == "staged"
        assert roster[tag]["pre_dedup_rows"] == {tag: 6}
        assert roster[tag]["allocated"][tag] > 0
    redis = report["budget_redistribution"]
    assert redis["skipped_source_tags"] == ["gated_src"]
    assert redis["skipped_planned_caps"] == {"gated_src": 10}
    assert "allocate_with_topup" in redis["note"]


def test_moderation_split_probe_declares_derived_near_class(monkeypatch, tmp_path):
    """Pin (i): a selective probe of a moderation-split source declares the
    DERIVED near-distribution stratum to the aggregate regime guard — zero
    flagged rows raises loud (moderation-derived-regime-guard-gap)."""
    spec = CP.SourceSpec(
        source_tag="modsplit",
        dataset_id="org/open-mod",
        regime_class="ordinary",
        realism_tier=1,
        pre_dedup_cap=10,
        text_fields=("text",),
        moderation_split=True,
    )
    _install_hf_seams(monkeypatch, gated=set(), raw_by_dataset={"org/open-mod": _raw_rows("m", 6)})
    monkeypatch.setattr(CP, "SOURCES", (spec,))
    with pytest.raises(RuntimeError, match="regime class"):
        CP.run_pipeline(_probe_args(tmp_path / "unflagged"))
    # Positive control: ONE flagged row realizes the stratum -> probe completes.
    flagged = dict(_raw_rows("m-flagged", 1)[0], toxic=True)
    _install_hf_seams(
        monkeypatch,
        gated=set(),
        raw_by_dataset={"org/open-mod": [*_raw_rows("m", 6), flagged]},
    )
    report = CP.run_pipeline(_probe_args(tmp_path / "flagged"))
    assert report["regime_class_counts"].get("near-distribution", 0) >= 1


def _decide_args(tmp_path, *extra):
    return FT.build_parser().parse_args(["--phase", "decide", "--out-root", str(tmp_path), *extra])


def _bound_decide_fixture(tmp_path, verdict="Replicates"):
    """Write the full content-bound decide artifact set: both models'
    _load_model_artifacts file set + decision.json + a .p4_done sentinel
    carrying the matching per-file sha256 fingerprint (round-7 binding)."""
    fits = tmp_path / "fits"
    for mk in ("modelA", "modelB"):
        d = fits / mk
        d.mkdir(parents=True, exist_ok=True)
        (d / "fits_summary.json").write_text(json.dumps({"model": mk}))
        (d / "percontext_recon.json").write_text(json.dumps({"model": mk}))
    (fits / "decision.json").write_text(json.dumps({"verdict": verdict, "a_pass": True}))
    (fits / ".p4_done").write_text(
        json.dumps(
            {
                "done": True,
                "verdict": verdict,
                "content_sha256": FT._decision_fingerprint(tmp_path),
            }
        )
    )
    return fits


def test_decide_sentinel_skip_and_force(tmp_path):
    """Pin (j1): a matching CONTENT-BOUND fits/.p4_done SKIPs the re-decide
    (persisted decision returned, resumed_from_sentinel set); --force bypasses
    the skip (proof: the run proceeds to the #13 reliability refusal), and a
    pre-recompute arg-validation failure does NOT invalidate the sentinel —
    the still-consistent pair keeps skipping on a no-force resume."""
    fits = _bound_decide_fixture(tmp_path)
    res = FT.run_decide(_decide_args(tmp_path))
    assert res.get("resumed_from_sentinel") is True
    assert res["verdict"] == "Replicates"
    with pytest.raises(SystemExit, match="--reliability-a"):
        FT.run_decide(_decide_args(tmp_path, "--force"))
    assert (fits / ".p4_done").exists(), "arg-validation failure must not invalidate"
    assert FT.run_decide(_decide_args(tmp_path)).get("resumed_from_sentinel") is True


def test_decide_sentinel_without_decision_fails_loud(tmp_path):
    """Pin (j2): a sentinel WITHOUT decision.json is inconsistent phase state
    — fail loud, never a silent skip over missing artifacts."""
    fits = tmp_path / "fits"
    fits.mkdir(parents=True)
    (fits / ".p4_done").write_text(json.dumps({"done": True, "verdict": "Replicates"}))
    with pytest.raises(RuntimeError, match="inconsistent"):
        FT.run_decide(_decide_args(tmp_path))


def test_decide_sentinel_content_binding_mismatch_fails_loud(tmp_path):
    """Pin (j3): a sentinel whose fingerprint no longer matches the on-disk
    artifacts (tampered decision / drifted decide inputs / a digest-less
    pre-binding sentinel) FAILS LOUD at skip time, naming the drifted keys —
    never a stale-done skip (round-7 decide-force-stale-sentinel-window)."""
    fits = _bound_decide_fixture(tmp_path)
    (fits / "decision.json").write_text(json.dumps({"verdict": "Tampered"}))
    with pytest.raises(RuntimeError, match=r"not content-bound.*decision\.json"):
        FT.run_decide(_decide_args(tmp_path))
    # drifted decide INPUT (a model summary), decision restored: same refusal.
    fits2 = _bound_decide_fixture(tmp_path)
    (fits2 / "modelB" / "fits_summary.json").write_text(json.dumps({"model": "drifted"}))
    with pytest.raises(RuntimeError, match=r"not content-bound.*modelB/fits_summary\.json"):
        FT.run_decide(_decide_args(tmp_path))
    # legacy digest-less sentinel: loud refusal too (never a trust-on-existence skip).
    _bound_decide_fixture(tmp_path)
    (fits / ".p4_done").write_text(json.dumps({"done": True, "verdict": "Replicates"}))
    with pytest.raises(RuntimeError, match="not content-bound"):
        FT.run_decide(_decide_args(tmp_path))


def test_decide_force_invalidates_sentinel_before_recompute(tmp_path):
    """Pin (j4): the crash window — a forced re-decide that dies MID-RECOMPUTE
    (after arg validation, before a fresh sentinel lands) leaves the sentinel
    INVALIDATED (atomically quarantined to .p4_done.stale), so a no-force
    resume RE-RUNS the decide instead of skipping on a stale-done sentinel
    certifying the OLD decision."""
    fits = _bound_decide_fixture(tmp_path)
    # the {} model summaries crash the recompute at candidate_sets (KeyError),
    # strictly after the invalidation point and before any decision write.
    with pytest.raises(KeyError):
        FT.run_decide(_decide_args(tmp_path, "--force", "--allow-missing-reliability"))
    assert not (fits / ".p4_done").exists(), "stale-done sentinel survived the crash window"
    assert (fits / ".p4_done.stale").exists(), "invalidated sentinel quarantined, not lost"
    # resume WITHOUT --force: no skip — the re-decide is attempted for real
    # (and crashes on the same degenerate fixture), never a stale-done skip.
    with pytest.raises(KeyError):
        FT.run_decide(_decide_args(tmp_path, "--allow-missing-reliability"))


def test_decide_skip_path_republishes_sentinel(tmp_path, monkeypatch):
    """Pin (j5): the no-force skip re-invokes the REAL _publish_sentinel body
    (crash-window publish retry — a crash between the local sentinel write and
    its publish otherwise leaves the remote copy missing forever; both legs
    idempotent). Fake only at the HF boundary, signature-mirroring."""
    fits = _bound_decide_fixture(tmp_path)
    calls: list = []

    def fake_upload_single_file(local: Path, dest: str) -> None:
        calls.append((local, dest))

    monkeypatch.setattr(GC, "upload_single_file", fake_upload_single_file)
    res = FT.run_decide(_decide_args(tmp_path, "--publish", "hf"))
    assert res.get("resumed_from_sentinel") is True
    assert calls == [(fits / ".p4_done", f"{FT.PUBLISH_EVAL_MIRROR}/fits/.p4_done")]


def test_decision_fingerprint_refuses_missing_required_input(tmp_path):
    """Pin (j6a): _decision_fingerprint REFUSES an absent member — every
    fingerprint file is a mandatory _load_model_artifacts input, so absence
    raises naming the relative path, never a placeholder entry that could
    'verify' against an equally-absent sentinel record."""
    _bound_decide_fixture(tmp_path)
    (tmp_path / "fits" / "modelA" / "percontext_recon.json").unlink()
    with pytest.raises(RuntimeError, match=r"modelA/percontext_recon\.json"):
        FT._decision_fingerprint(tmp_path)


def test_decide_skip_missing_input_with_matching_missing_record_fails_loud(tmp_path):
    """Pin (j6b): the residual hole itself — a legacy sentinel that RECORDED
    a member as absent, with the member still absent at skip time, FAILS
    LOUD at run_decide (pre-fix both sides read the same placeholder, the
    fingerprints matched, and the skip verified cleanly over a missing
    required input)."""
    fits = _bound_decide_fixture(tmp_path)
    legacy = FT._decision_fingerprint(tmp_path)
    legacy["modelB/percontext_recon.json"] = "missing"  # the pre-fix placeholder
    (fits / "modelB" / "percontext_recon.json").unlink()
    (fits / ".p4_done").write_text(
        json.dumps({"done": True, "verdict": "Replicates", "content_sha256": legacy})
    )
    with pytest.raises(RuntimeError, match=r"modelB/percontext_recon\.json"):
        FT.run_decide(_decide_args(tmp_path))


def test_decide_skip_verdict_bound_to_decision_json(tmp_path, capsys):
    """Pin (j7): skip telemetry is bound to decision.json — (a) a
    sentinel-ONLY verdict mutation (content_sha256 intact, so the content
    binding still verifies: the sentinel's own verdict field is metadata
    OUTSIDE the fingerprint) fails loud on sentinel/decision disagreement
    instead of logging the mutated value; (b) on the healthy path the
    printed verdict is read from decision.json (the authoritative
    artifact)."""
    fits = _bound_decide_fixture(tmp_path)
    doc = json.loads((fits / ".p4_done").read_text())
    doc["verdict"] = "Tampered-metadata"
    (fits / ".p4_done").write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match=r"Tampered-metadata.*Replicates"):
        FT.run_decide(_decide_args(tmp_path))
    # healthy path: the printed skip line reports decision.json's verdict.
    _bound_decide_fixture(tmp_path)
    res = FT.run_decide(_decide_args(tmp_path))
    assert res["verdict"] == "Replicates"
    assert "verdict='Replicates'" in capsys.readouterr().out


def test_publish_none_on_canonical_out_root_refused(monkeypatch):
    """Pin (k1): '--publish none' against the canonical production out-root
    (or any path under it) is REFUSED absent --allow-local-only — EXECUTED
    through the real main() entrypoint (production argv parse) for BOTH
    publish-bearing phases, with the phase bodies tripwired so a dead or
    misordered guard call cannot stay green; --allow-local-only ESCAPES for
    real (the phase body is reached); scratch out-roots + durable modes pass
    the guard (round-7 publish-none-main-path-hollow-pin: replaces the former
    inspect.getsource count, which never executed the refusal)."""

    def _main(*argv):
        monkeypatch.setattr(sys, "argv", ["issue2502_fits.py", *argv])
        return FT.main()

    def _must_not_run(args, **kw):
        raise AssertionError("phase body reached despite --publish none on canonical out-root")

    monkeypatch.setattr(FT, "run_fit", _must_not_run)
    monkeypatch.setattr(FT, "run_decide", _must_not_run)
    canon = str(FT.CANONICAL_OUT_ROOT)
    sub = str(FT.CANONICAL_OUT_ROOT / "sub")
    # REFUSAL EXECUTES on both phases through main() (fit leg: the parser's
    # DEFAULT out-root IS the canonical tree; decide leg: a path under it).
    with pytest.raises(SystemExit, match="loss path"):
        _main("--phase", "fit", "--publish", "none")
    with pytest.raises(SystemExit, match="loss path"):
        _main("--phase", "decide", "--publish", "none", "--out-root", sub)
    # --allow-local-only ESCAPES for real: main() proceeds into the phase body.
    reached: list[str] = []
    monkeypatch.setattr(FT, "run_fit", lambda args, **kw: reached.append("fit") or {})
    monkeypatch.setattr(FT, "run_decide", lambda args, **kw: reached.append("decide") or {})
    assert (
        _main("--phase", "fit", "--publish", "none", "--out-root", canon, "--allow-local-only") == 0
    )
    assert (
        _main("--phase", "decide", "--publish", "none", "--out-root", canon, "--allow-local-only")
        == 0
    )
    assert reached == ["fit", "decide"]
    # Scratch out-roots + durable modes pass the direct guard (unit legs kept).
    ap = FT.build_parser()
    FT._refuse_publish_none_on_canonical(
        ap.parse_args(["--phase", "fit", "--publish", "none", "--out-root", "/tmp/i2502-scratch"])
    )
    FT._refuse_publish_none_on_canonical(ap.parse_args(["--phase", "fit", "--publish", "hf+git"]))


def test_publish_sentinel_routes_selected_backends(monkeypatch, tmp_path):
    """Pin (k2): _publish_sentinel routes the .p4_done sentinel through the
    SELECTED backend(s) — git modes COMMIT it (previously consumer-less on
    the pure-git path), hf modes upload it, 'none' touches neither. Fakes
    only at the git/HF boundary, signature-mirroring
    (_git_publish(paths, repo); upload_single_file(local, dest))."""
    sentinel = tmp_path / ".p4_done"
    sentinel.write_text("{}")
    calls: dict[str, list] = {"git": [], "hf": []}

    def fake_git_publish(paths: list, repo: Path) -> None:
        calls["git"].append((list(paths), repo))

    def fake_upload_single_file(local: Path, dest: str) -> None:
        calls["hf"].append((local, dest))

    monkeypatch.setattr(FT, "_git_publish", fake_git_publish)
    monkeypatch.setattr(GC, "upload_single_file", fake_upload_single_file)
    FT._publish_sentinel(sentinel, "git", "pfx")
    assert calls == {"git": [([sentinel], FT._REPO_ROOT)], "hf": []}
    FT._publish_sentinel(sentinel, "hf", "pfx")
    assert calls["hf"] == [(sentinel, "pfx/fits/.p4_done")]
    FT._publish_sentinel(sentinel, "hf+git", "pfx")
    assert len(calls["git"]) == 2 and len(calls["hf"]) == 2
    FT._publish_sentinel(sentinel, "none", "pfx")
    assert len(calls["git"]) == 2 and len(calls["hf"]) == 2


# ---------------------------------------------------------------------------
# Round-9 pins (l)-(p): source-registry split-spec class (the live P0 crash).
# ---------------------------------------------------------------------------


def _registry(tag: str):
    (spec,) = [s for s in CP.SOURCES if s.source_tag == tag]
    return spec


def test_l_split_absent_raises_loud_naming_source_and_split(monkeypatch, tmp_path):
    """Pin (l): a bad-split ValueError from the stream becomes a RuntimeError
    naming source_tag + declared split + dataset id, chained from the
    original — through the REAL stage_source -> _stage_one_config chain, and
    NEVER the round-5 gated-skip route."""
    spec = CP.SourceSpec(
        source_tag="badsplit_src",
        dataset_id="org/jbb-like",
        regime_class="weird",
        realism_tier=2,
        pre_dedup_cap=10,
        splits=("harmful",),
        text_fields=("text",),
    )
    monkeypatch.setattr(
        CP, "_resolve_dataset_revision", lambda dataset_id, revision_ref=None: "0" * 40
    )

    def raising_stream(dataset_id, config, split, **kwargs):
        raise ValueError(f"Bad split: {split}. Available splits: ['train']")

    monkeypatch.setattr(CP.CS, "_hf_stream", raising_stream)
    with pytest.raises(RuntimeError) as ei:
        CP.stage_source(spec, tmp_path, None, stream_cap=None, filter_language=False, seed=0)
    msg = str(ei.value)
    assert "badsplit_src" in msg and "'harmful'" in msg and "org/jbb-like" in msg, msg
    assert "registry" in msg and "Bad split" in msg, msg
    assert isinstance(ei.value.__cause__, ValueError)


def test_l2_non_split_valueerror_propagates_unenriched(monkeypatch, tmp_path):
    """Pin (l residual): a ValueError that is NOT the bad-split shape (e.g. a
    keep_fn bug) propagates verbatim — the enrich handler must not absorb
    unrelated ValueErrors into the registry-defect message."""
    spec = CP.SourceSpec(
        source_tag="valerr_src",
        dataset_id="org/plain",
        regime_class="weird",
        realism_tier=2,
        pre_dedup_cap=10,
        text_fields=("text",),
    )
    monkeypatch.setattr(
        CP, "_resolve_dataset_revision", lambda dataset_id, revision_ref=None: "0" * 40
    )

    def raising_stream(dataset_id, config, split, **kwargs):
        raise ValueError("some unrelated data-plane failure")

    monkeypatch.setattr(CP.CS, "_hf_stream", raising_stream)
    with pytest.raises(ValueError, match="unrelated data-plane failure"):
        CP.stage_source(spec, tmp_path, None, stream_cap=None, filter_language=False, seed=0)


def _preflight_fakes():
    from huggingface_hub.utils import GatedRepoError

    def fake_rev(dataset_id: str, revision_ref: str | None = None) -> str:
        if dataset_id == "org/gated":
            raise GatedRepoError("gated fixture")
        return "0" * 40

    def fake_split_names(dataset_id, config=None, *, revision=None, **kw):
        assert revision == "0" * 40
        if config == "badcfg":
            raise ValueError("BuilderConfig 'badcfg' not found. Available: ['default']")
        if dataset_id == "org/deadscript":
            raise RuntimeError("Dataset scripts are no longer supported, but found x.py")
        if dataset_id == "org/jbb-like":
            return ["harmful", "benign"]
        return ["train"]

    return fake_rev, fake_split_names


def test_m_preflight_aggregates_all_defects_into_one_raise(tmp_path):
    """Pin (m): one preflight pass enumerates EVERY registry defect (bad
    split, bad config, dead script) in ONE RuntimeError; healthy + gated
    sources stay out of the defect list."""
    fake_rev, fake_split_names = _preflight_fakes()
    ok = _spec("ok_src", "org/plain", "weird")
    bad_split = _spec("badsplit_src", "org/jbb-like", "weird")  # default 'train'
    gated = _spec("gated_src", "org/gated", "weird")
    bad_cfg = CP.SourceSpec(
        source_tag="badcfg_src",
        dataset_id="org/plain",
        regime_class="weird",
        realism_tier=2,
        pre_dedup_cap=10,
        configs=("badcfg",),
        text_fields=("text",),
    )
    dead = _spec("deadscript_src", "org/deadscript", "weird")
    with pytest.raises(RuntimeError) as ei:
        CP.verify_declared_splits(
            (ok, bad_split, gated, bad_cfg, dead),
            split_names_fn=fake_split_names,
            revision_fn=fake_rev,
        )
    msg = str(ei.value)
    assert "3 defect(s)" in msg, msg
    assert "badsplit_src" in msg and "badcfg_src" in msg and "deadscript_src" in msg, msg
    assert "ok_src" not in msg and "gated_src" not in msg, msg


def test_m2_preflight_gated_unverifiable_and_transient_propagates():
    """Pin (m residual): gated -> recorded unverifiable (no raise); a
    transient 503 at the revision seam PROPAGATES (never swallowed)."""
    from huggingface_hub.utils import HfHubHTTPError

    fake_rev, fake_split_names = _preflight_fakes()
    ok = _spec("ok_src", "org/plain", "weird")
    gated = _spec("gated_src", "org/gated", "weird")
    res = CP.verify_declared_splits(
        (ok, gated), split_names_fn=fake_split_names, revision_fn=fake_rev
    )
    assert res["n_verified"] == 1 and len(res["unverifiable"]) == 1, res
    assert "gated_src" in res["unverifiable"][0]

    class _Resp:
        status_code = 503

    def transient_rev(dataset_id: str, revision_ref: str | None = None) -> str:
        exc = HfHubHTTPError("503 fixture")
        exc.response = _Resp()
        raise exc

    with pytest.raises(HfHubHTTPError):
        CP.verify_declared_splits((ok,), split_names_fn=fake_split_names, revision_fn=transient_rev)


def test_m3_preflight_data_files_sources_train_fixed():
    """Pin (m residual): data_files sources verify by construction on
    splits=('train',) and are a DEFECT on any other declared split."""
    fake_rev, fake_split_names = _preflight_fakes()
    good = CP.SourceSpec(
        source_tag="df_src",
        dataset_id="org/script-dead",
        regime_class="idiosyncratic",
        realism_tier=1,
        pre_dedup_cap=10,
        data_files_template="hf://datasets/org/script-dead@{revision}/rows.jsonl",
        text_fields=("text",),
    )
    res = CP.verify_declared_splits((good,), split_names_fn=fake_split_names, revision_fn=fake_rev)
    assert res["n_verified"] == 1, res
    import dataclasses

    bad = dataclasses.replace(good, splits=("test",))
    with pytest.raises(RuntimeError, match="data_files sources serve"):
        CP.verify_declared_splits((bad,), split_names_fn=fake_split_names, revision_fn=fake_rev)


def test_n_registry_corrected_specs_pinned():
    """Pin (n): the round-9 audit's corrected registry entries (a refactor
    reverting any of these re-arms a production crash or a silent
    composition change)."""
    jbb = _registry("jbb_behaviors")
    assert jbb.configs == ("behaviors",) and jbb.splits == ("harmful", "benign")
    mask = _registry("mask")
    assert mask.splits == ("test",) and len(mask.configs) == 6 and None not in mask.configs
    assert _registry("xstest").splits == ("test",)
    bt = _registry("beavertails")
    assert bt.configs == (None,) and bt.splits == ("30k_train",)
    mwe = _registry("model_written_evals")
    # Round 10: mwe stages PER FILE (heterogeneous top-level fields across the
    # sycophancy dir's files — latent `_cast_table` crash), replacing the
    # round-9 data_dir route; the philpapers2020 upstream-duplicate exclusion
    # is pinned in tests/test_issue2502_schema_gate.py.
    assert mwe.configs == (None,) and isinstance(mwe.data_files_template, tuple)
    assert all("sycophancy/" in t and "{revision}" in t for t in mwe.data_files_template)
    pippa = _registry("pippa")
    assert pippa.data_files_template is not None
    assert "pippa_deduped.jsonl" in pippa.data_files_template
    assert "{revision}" in pippa.data_files_template
    assert pippa.splits == ("train",)
    rs = _registry("riddle_sense")
    assert rs.revision_ref == "refs/convert/parquet" and rs.splits == ("train",)
    assert _registry("legalbench").splits == ("train", "test")
    for spec in CP.SOURCES:
        assert isinstance(spec.splits, tuple) and spec.splits, spec.source_tag
        assert all(isinstance(s, str) and s for s in spec.splits), spec.source_tag


def test_o_multi_split_staging_distinct_files_cumulative_cap(monkeypatch, tmp_path):
    """Pin (o): two splits of one config stage through the REAL chain to
    DISTINCT checkpoint files, with the keep-cap CUMULATIVE across attempts
    and @split-suffixed counter keys."""
    spec = CP.SourceSpec(
        source_tag="twosplit_src",
        dataset_id="org/twosplit",
        regime_class="weird",
        realism_tier=2,
        pre_dedup_cap=15,
        splits=("harmful", "benign"),
        text_fields=("text",),
    )
    rows_by_split = {
        "harmful": _raw_rows("twosplit-h", 10),
        "benign": _raw_rows("twosplit-b", 10),
    }
    monkeypatch.setattr(
        CP, "_resolve_dataset_revision", lambda dataset_id, revision_ref=None: "0" * 40
    )

    def split_stream(dataset_id, config, split, **kwargs):
        assert kwargs.get("revision") == "0" * 40
        return iter(rows_by_split[split])

    monkeypatch.setattr(CP.CS, "_hf_stream", split_stream)
    rows, ctr = CP.stage_source(
        spec, tmp_path, None, stream_cap=None, filter_language=False, seed=0
    )
    # cumulative cap: 10 from harmful + only 5 of benign (15 total)
    assert len(rows) == 15
    assert set(ctr) == {"default@harmful", "default@benign"}
    staged = tmp_path / "staged"
    h_file = staged / "twosplit_src__default__harmful.jsonl"
    b_file = staged / "twosplit_src__default__benign.jsonl"
    assert h_file.exists() and b_file.exists()
    assert len(CP.CS.read_jsonl(h_file)) == 10
    assert len(CP.CS.read_jsonl(b_file)) == 5


def test_p_preflight_wired_before_staging_and_skip_flag(monkeypatch, tmp_path):
    """Pin (p): run_pipeline calls verify_declared_splits BEFORE any staging;
    --skip-split-preflight skips it (offline-smoke escape hatch)."""

    class _PreflightRan(Exception):
        pass

    class _StagingReached(Exception):
        pass

    def sentinel_preflight(sources, **kw):
        raise _PreflightRan()

    def sentinel_stage(spec, *a, **kw):
        raise _StagingReached()

    monkeypatch.setattr(CP, "verify_declared_splits", sentinel_preflight)
    monkeypatch.setattr(CP, "stage_source", sentinel_stage)
    base = ["--probe", "--no-token-filter", "--out-dir", str(tmp_path), "--budget", "5"]
    with pytest.raises(_PreflightRan):
        CP.run_pipeline(CP.build_argparser().parse_args(base))
    with pytest.raises(_StagingReached):
        CP.run_pipeline(CP.build_argparser().parse_args([*base, "--skip-split-preflight"]))


# ---------------------------------------------------------------------------
# Round-11 pins (q1, q2): proactive secret scrub of the assembled corpus.
# All planted strings are SYNTHETIC (fake-but-secret-SHAPED); never a real
# secret, never real corpus text.
# ---------------------------------------------------------------------------

# Fake hf-token-SHAPED string (matches secret_scrub's `hf-token` pattern,
# survives its DUMMY_RX placeholder filter). Built by concatenation so the
# test file itself never contains a contiguous secret-shaped literal.
_PLANTED = "hf" + "_" + "Ab3dKq9RtY7uPw2sXeNvB5mZcJ6hLgF4"
# Secret-SHAPED but dummy-filtered (X-run) — must survive the scrub verbatim.
_BENIGN_SHAPED = "hf" + "_" + "X" * 34
_BENIGN_DOC = "YOUR_API_KEY"


def _corpus_row(i: int, text: str) -> dict:
    """Minimal corpus-schema row (the persisted key set of step 6)."""
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "context_id": sha[:16],
        "context_sha": sha,
        "text": text,
        "source_tag": f"src{i}",
        "dataset_id": f"org/ds{i}",
        "config": "default",
        "regime_class": "ordinary",
        "realism_tier": 1,
        "split": "train",
        "lodo_group": f"src{i}",
    }


def test_r11_scrub_redacts_secret_grade_and_preserves_benign(tmp_path):
    """Pin (q1): same-length redaction + count recording + dummy-filter
    preservation, through the builder's own scrub step (real bodies, real
    file — no mocks)."""
    rows = [
        _corpus_row(0, f"user pasted a credential {_PLANTED} into the chat"),
        _corpus_row(1, f"docs say set {_BENIGN_DOC} or use {_BENIGN_SHAPED} as a stub"),
        _corpus_row(2, "an ordinary clean context with no flagged content"),
    ]
    path = tmp_path / "corpus.jsonl"
    CP.CS._write_jsonl_atomic(path, rows)
    before = path.read_bytes()
    assert _PLANTED.encode() in before  # fixture sanity

    stats = CP.scrub_corpus_secrets(path)

    after = path.read_bytes()
    assert stats["n"] == 1
    assert stats["by_pattern"] == {"hf-token": 1}
    assert _PLANTED.encode() not in after  # secret-grade string gone
    assert ("X" * len(_PLANTED)).encode() in after  # same-length placeholder
    assert len(after) == len(before)  # byte-length preserved
    # benign placeholders survive verbatim (DUMMY_RX filter not defeated)
    assert _BENIGN_SHAPED.encode() in after
    assert _BENIGN_DOC.encode() in after
    # still-valid JSONL; sibling fields of the scrubbed row intact
    out = [json.loads(ln) for ln in after.decode("utf-8").split("\n") if ln.strip()]
    assert len(out) == 3
    scrubbed = [r for r in out if "X" * len(_PLANTED) in r["text"]]
    assert len(scrubbed) == 1
    assert scrubbed[0]["context_sha"] == rows[0]["context_sha"]
    assert scrubbed[0]["source_tag"] == "src0"
    # re-running on the now-clean file is a no-op (rebuild reproducibility)
    assert CP.scrub_corpus_secrets(path) == {"n": 0, "by_pattern": {}}
    assert path.read_bytes() == after


def test_r11_build_path_scrubs_persisted_corpus_and_records_count(monkeypatch, tmp_path):
    """Pin (q2): the BUILD path (run_pipeline, no --probe) persists a CLEAN
    corpus.jsonl (scrub wired between the corpus write and the upload step)
    and records the count durably in dedup_report.json. Fakes ONLY the HF
    network boundary (round-5 harness); the real chain — staging, dedup,
    subsample, splits, corpus write, scrub, report — executes unmodified."""
    weird_rows = _raw_rows("weird_ok", 6)
    ord_rows = _raw_rows("ord_ok", 6)
    ord_rows[0]["text"] += f" pasted credential {_PLANTED} mid-conversation"
    specs = (
        _spec("weird_ok", "org/open-weird", "weird"),
        _spec("ord_ok", "org/open-ord", "ordinary"),
    )
    _install_hf_seams(
        monkeypatch,
        gated=set(),
        raw_by_dataset={"org/open-weird": weird_rows, "org/open-ord": ord_rows},
    )
    monkeypatch.setattr(CP, "SOURCES", specs)
    args = CP.build_argparser().parse_args(
        [
            "--no-token-filter",
            "--skip-split-preflight",
            "--skip-schema-gate",
            "--out-dir",
            str(tmp_path),
            "--budget",
            "12",
        ]
    )
    report = CP.run_pipeline(args)

    data = (tmp_path / "corpus.jsonl").read_bytes()
    assert _PLANTED.encode() not in data  # persisted corpus is clean
    assert ("X" * len(_PLANTED)).encode() in data  # same-length placeholder
    assert report["secrets_scrubbed"] == 1
    assert report["secrets_scrubbed_by_pattern"] == {"hf-token": 1}
    dedup = json.loads((tmp_path / "dedup_report.json").read_text())
    assert dedup["secrets_scrubbed"] == 1  # durable in the uploaded report
    rows = [json.loads(ln) for ln in data.decode("utf-8").split("\n") if ln.strip()]
    scrubbed = [r for r in rows if "X" * len(_PLANTED) in r["text"]]
    assert len(scrubbed) == 1
    # stored context_sha is NOT recomputed post-scrub: it documents the
    # pre-scrub source-text identity every downstream fingerprint reads
    # (gen_capture.corpus_content_sha16 fingerprints the STORED pairs) …
    assert scrubbed[0]["context_sha"] != CP._context_sha(scrubbed[0]["text"])
    # … while every clean row's sha still matches its text.
    for r in rows:
        if r is not scrubbed[0]:
            assert r["context_sha"] == CP._context_sha(r["text"])


# ---------------------------------------------------------------------------
# Pin (r): render_prompt_ids under BOTH apply_chat_template return conventions
# (transformers 4.57.6 return_dict=False default vs 5.15.1 return_dict=True
# default; #2502 + #2378). Fails pre-fix on the 5.x-convention fake.
# ---------------------------------------------------------------------------


class _Tok4xConvention:
    """transformers 4.57.6 shape: tokenize=True defaults return_dict=False."""

    def apply_chat_template(
        self,
        msgs,
        *,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=False,
        **kwargs,
    ):
        """Return an id LIST unless return_dict=True (4.x default False)."""
        if not tokenize:
            return "<|user|>ping<|assistant|>" + GC.EMPTY_THINK
        ids = [1, 2, 3]
        if return_dict:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return ids


class _Tok5xConvention:
    """transformers 5.15.1 shape: tokenize=True defaults return_dict=True."""

    def apply_chat_template(
        self,
        msgs,
        *,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=None,
        **kwargs,
    ):
        """Return a BatchEncoding-like DICT unless return_dict=False is explicit."""
        if not tokenize:
            return "<|user|>ping<|assistant|>" + GC.EMPTY_THINK
        if return_dict is None:
            return_dict = True  # the 5.x default flip for tokenize=True
        ids = [4, 5, 6]
        if return_dict:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return ids


@pytest.mark.parametrize("disable_thinking", [False, True])
def test_render_prompt_ids_flat_int_ids_under_both_conventions(disable_thinking):
    """Pin (r): flat int-id list under BOTH 4.x and 5.x conventions."""
    for tok, want in ((_Tok4xConvention(), [1, 2, 3]), (_Tok5xConvention(), [4, 5, 6])):
        got = GC.render_prompt_ids(tok, "ping", disable_thinking=disable_thinking)
        assert got == want, (type(tok).__name__, got)
        assert all(type(x) is int for x in got), got
