"""Issue #2502 round-10 pins: the comprehensive per-source streaming schema
gate (`verify_source_schemas`) + the sycophancy_eval per-file staging fix.

Live incidents pinned (round-10 P0 corpus build crash, 2026-08-25):
  (a) `sycophancy_eval` scanned 12,000 / kept 0 — scalar text_fields against
      the list-valued `prompt` field (the #1092 data-ingestion class) — the
      gate FAILS LOUD `kept_zero`, never a silent under-populated corpus;
  (b) datasets' `_cast_table` TypeError at the answer.jsonl -> feedback.jsonl
      file boundary (`metadata` struct gains `prompt_template_type`) — the
      gate PREDICTS it from per-file signature comparison (bounded rows can
      never reach the boundary), and staging now runs PER FILE so no
      cross-file cast exists at all;
  (c) all defects aggregate into ONE raise (anti-whack-a-mole, schema level);
  (d) gated/inaccessible sources are unverifiable-not-fatal, with a declared
      fallback probed when the primary is inaccessible;
  (e) `first_user_content` handles the sycophancy-eval turn shape
      {"type": "human"|"ai", "content": str}.

All tests are OFFLINE: network faked ONLY at the designed injection seams
(`revision_fn` / `enumerate_files_fn` / `stream_open_fn`, and
`CS._hf_stream` + `_resolve_dataset_revision` for the real-staging pin) with
signature-conformant fakes; the production bodies (`verify_source_schemas`,
`_probe_rows`, signature helpers, `stage_source` -> `_stage_one_config` ->
`CS._stream_stage`, `_make_keep_fn`) execute unmodified.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

CP = importlib.import_module("issue2502_corpus")

_TEXT = "sufficiently long synthetic schema-gate context text %d"


def _spec(tag, ds, **kws):
    kws.setdefault("regime_class", CP.REGIME_NEAR)
    return CP.SourceSpec(
        source_tag=tag,
        dataset_id=ds,
        realism_tier=2,
        pre_dedup_cap=kws.pop("cap", 10),
        **kws,
    )


def _rev(dataset_id: str, revision_ref: str | None = None) -> str:
    return "1" * 40


def _no_files(dataset_id, config, split, revision, data_dir):
    return None


def _gate(sources, *, rows=None, files_fn=_no_files, stream_fn, rev_fn=_rev):
    kwargs = dict(
        token_filter=None,
        filter_language=False,
        revision_fn=rev_fn,
        enumerate_files_fn=files_fn,
        stream_open_fn=stream_fn,
    )
    if rows is not None:
        kwargs["rows_per_attempt"] = rows
    return CP.verify_source_schemas(tuple(sources), **kwargs)


# ---------------------------------------------------------------------------
# (e) first_user_content: the sycophancy-eval `type` role key (fails pre-fix).
# ---------------------------------------------------------------------------


def test_first_user_content_type_key():
    assert CP.first_user_content([{"type": "human", "content": "hello there friend"}]) == (
        "hello there friend"
    )
    # an assistant-only conversation yields nothing
    assert CP.first_user_content([{"type": "ai", "content": "assistant turn only"}]) is None
    # first HUMAN turn wins even after an ai turn
    conv = [
        {"type": "ai", "content": "assistant preamble"},
        {"type": "human", "content": "the real user question"},
    ]
    assert CP.first_user_content(conv) == "the real user question"


# ---------------------------------------------------------------------------
# Registry pin: sycophancy_eval stages PER FILE with conv extraction.
# ---------------------------------------------------------------------------


def test_sycophancy_spec_per_file_attempts():
    syco = next(s for s in CP.SOURCES if s.source_tag == "sycophancy_eval")
    assert isinstance(syco.data_files_template, tuple)
    stems = [CP._file_stem(t) for t in syco.data_files_template]
    assert stems == ["answer", "are_you_sure", "feedback", "mimicry"], stems
    assert all("{revision}" in t for t in syco.data_files_template)
    assert syco.conv_fields == ("prompt",)
    attempts = CP._iter_attempts(syco)
    assert len(attempts) == 4
    assert [CP._attempt_key(*a) for a in attempts] == stems


def test_mwe_registry_excludes_upstream_duplicate_file():
    """sycophancy_on_philpapers2020.jsonl is byte-identical upstream to
    sycophancy_on_nlp_survey.jsonl (raw-file sha256 probe 2026-08-25): staging
    it keeps 0 rows after within-source dedup, so the registry excludes it."""
    mwe = next(s for s in CP.SOURCES if s.source_tag == "model_written_evals")
    stems = [CP._file_stem(t) for t in mwe.data_files_template]
    assert stems == [
        "sycophancy_on_nlp_survey",
        "sycophancy_on_political_typology_quiz",
    ], stems
    assert not any("philpapers" in t for t in mwe.data_files_template)


def test_fully_duplicate_file_flagged_kept_zero_by_gate():
    """The gate shares the within-source dedup set across attempts (staging
    parity), so an upstream file fully duplicating an earlier one reads
    kept_zero at gate time instead of tripping the staging fail-loud at
    production scale (the model-written-evals philpapers2020 class)."""
    rows = [{"q": _TEXT % i} for i in range(3)]

    def stream(spec, *, config, split, dataset_id, revision, data_file=None):
        return iter(list(rows))  # the SAME rows for both declared files

    spec = _spec(
        "dup",
        "d/dup",
        text_fields=("q",),
        data_files_template=(
            "hf://datasets/d/dup@{revision}/a.jsonl",
            "hf://datasets/d/dup@{revision}/b.jsonl",
        ),
    )
    with pytest.raises(RuntimeError) as ei:
        _gate([spec], stream_fn=stream)
    msg = str(ei.value)
    assert "dup:b" in msg and "kept_zero" in msg and "dup_text_within_source" in msg


def test_iter_attempts_refuses_unsupported_combinations():
    with pytest.raises(ValueError, match="fallback"):
        CP._iter_attempts(
            _spec(
                "bad",
                "d/x",
                data_files_template="hf://datasets/d/x@{revision}/a.jsonl",
                fallback_dataset_id="d/fb",
                text_fields=("text",),
            )
        )
    with pytest.raises(ValueError, match="unstageable"):
        CP._iter_attempts(
            _spec(
                "badfmt",
                "d/x",
                data_files_template="hf://datasets/d/x@{revision}/a.unknownext",
                text_fields=("text",),
            )
        )


# ---------------------------------------------------------------------------
# (b) cross-file struct-key conflict (the live `_cast_table` crash class).
# ---------------------------------------------------------------------------


def _two_file_stream(rows_by_stem):
    def stream(spec, *, config, split, dataset_id, revision, data_file=None):
        return iter(rows_by_stem[CP._file_stem(data_file)])

    return stream


def _two_files(dataset_id, config, split, revision, data_dir):
    return ["hf://x/a.jsonl", "hf://x/b.jsonl"]


def test_cross_file_struct_key_conflict_raises():
    rows = {
        "a": [{"q": _TEXT % i, "metadata": {"prompt_template": "t"}} for i in range(3)],
        "b": [
            {
                "q": _TEXT % (10 + i),
                "metadata": {"prompt_template": "t", "prompt_template_type": "x"},
            }
            for i in range(3)
        ],
    }
    with pytest.raises(RuntimeError) as ei:
        _gate(
            [_spec("multi", "d/multi", text_fields=("q",))],
            files_fn=_two_files,
            stream_fn=_two_file_stream(rows),
        )
    msg = str(ei.value)
    assert "cross_file_schema_conflict" in msg
    assert "prompt_template_type" in msg and "metadata" in msg


def test_cross_file_type_class_change_raises():
    rows = {
        "a": [{"q": _TEXT % i, "meta": "plain string"} for i in range(3)],
        "b": [{"q": _TEXT % (10 + i), "meta": {"nested": "struct"}} for i in range(3)],
    }
    with pytest.raises(RuntimeError) as ei:
        _gate(
            [_spec("multi", "d/multi", text_fields=("q",))],
            files_fn=_two_files,
            stream_fn=_two_file_stream(rows),
        )
    assert "type ['dict'] vs first-file ['str']" in str(ei.value)


def test_missing_keys_in_later_file_not_flagged():
    """Nullable-fill direction (later file MISSING keys) must not false-fire."""
    rows = {
        "a": [
            {"q": _TEXT % i, "metadata": {"prompt_template": "t", "extra": "y"}} for i in range(3)
        ],
        "b": [{"q": _TEXT % (10 + i), "metadata": {"prompt_template": "t"}} for i in range(3)],
    }
    report = _gate(
        [_spec("multi", "d/multi", text_fields=("q",))],
        files_fn=_two_files,
        stream_fn=_two_file_stream(rows),
    )
    assert report["n_verified"] == 1
    assert report["attempts"]["multi:default"]["kept"] == 6


# ---------------------------------------------------------------------------
# (a) kept==0 field-mapping bugs fail loud; (c) ONE aggregated raise.
# ---------------------------------------------------------------------------


def _single_stream(rows):
    def stream(spec, *, config, split, dataset_id, revision, data_file=None):
        return iter(list(rows))

    return stream


def test_kept_zero_source_fails_loud_with_observed_fields():
    # the live shape: scalar text_fields against a LIST-valued `prompt`
    rows = [{"prompt": [{"type": "human", "content": _TEXT % i}]} for i in range(4)]
    with pytest.raises(RuntimeError) as ei:
        _gate([_spec("kz", "d/kz", text_fields=("prompt",))], stream_fn=_single_stream(rows))
    msg = str(ei.value)
    assert "kept_zero" in msg and "no_text_field" in msg
    assert "'prompt': 'list'" in msg  # observed top-level field type named


def test_all_failures_aggregate_into_one_raise():
    def stream(spec, *, config, split, dataset_id, revision, data_file=None):
        if dataset_id == "d/crash":
            raise TypeError("Couldn't cast array of type struct<a: string> to Value('string')")
        if dataset_id == "d/kz":
            return iter([{"prompt": [{"type": "human", "content": _TEXT % 1}]}])
        return iter([{"text": _TEXT % 2}])

    with pytest.raises(RuntimeError) as ei:
        _gate(
            [
                _spec("crash", "d/crash", text_fields=("text",)),
                _spec("kz", "d/kz", text_fields=("prompt",)),
                _spec("ok", "d/ok", text_fields=("text",)),
            ],
            stream_fn=stream,
        )
    msg = str(ei.value)
    assert "stream_error" in msg and "Couldn't cast" in msg
    assert "kept_zero" in msg and "kz" in msg
    assert "2 defect(s)" in msg  # BOTH enumerated in the ONE raise; ok clean


def test_empty_stream_fails_loud():
    with pytest.raises(RuntimeError, match="empty_stream"):
        _gate([_spec("empty", "d/empty", text_fields=("text",))], stream_fn=_single_stream([]))


# ---------------------------------------------------------------------------
# (d) gated sources: unverifiable-not-fatal; declared fallback probed.
# ---------------------------------------------------------------------------


def _gated_rev(gated_ids):
    from huggingface_hub.utils import GatedRepoError

    def rev(dataset_id: str, revision_ref: str | None = None) -> str:
        if dataset_id in gated_ids:
            raise GatedRepoError(f"gated fixture: {dataset_id}")
        return "1" * 40

    return rev


def test_gated_source_unverifiable_not_fatal():
    report = _gate(
        [
            _spec("gated", "d/gated", text_fields=("text",)),
            _spec("ok", "d/ok", text_fields=("text",)),
        ],
        stream_fn=_single_stream([{"text": _TEXT % 1}]),
        rev_fn=_gated_rev({"d/gated"}),
    )
    assert report["n_verified"] == 1
    assert len(report["unverifiable"]) == 1
    assert "gated" in report["unverifiable"][0]


def test_gate_probes_declared_fallback_when_primary_gated():
    probed: list[str] = []

    def stream(spec, *, config, split, dataset_id, revision, data_file=None):
        probed.append(dataset_id)
        return iter([{"text": _TEXT % 1}])

    report = _gate(
        [_spec("g", "d/gated", text_fields=("text",), fallback_dataset_id="d/fb")],
        stream_fn=stream,
        rev_fn=_gated_rev({"d/gated"}),
    )
    assert probed == ["d/fb"]
    assert report["attempts"]["g:default"]["fallback"] is True
    assert report["attempts"]["g:default"]["kept"] == 1


def test_transient_http_error_propagates():
    from huggingface_hub.utils import HfHubHTTPError

    class _Resp:
        status_code = 503

    exc = HfHubHTTPError("503 fixture")
    exc.response = _Resp()

    def rev(dataset_id: str, revision_ref: str | None = None) -> str:
        raise exc

    with pytest.raises(HfHubHTTPError):
        _gate(
            [_spec("t", "d/t", text_fields=("text",))],
            stream_fn=_single_stream([{"text": _TEXT % 1}]),
            rev_fn=rev,
        )


# ---------------------------------------------------------------------------
# Corrected sycophancy shape passes the REAL production keep chain.
# ---------------------------------------------------------------------------


def test_corrected_sycophancy_shape_passes_real_keep_chain():
    syco = next(s for s in CP.SOURCES if s.source_tag == "sycophancy_eval")
    rows_by_stem = {
        "answer": [{"prompt": [{"type": "human", "content": _TEXT % i}]} for i in range(3)],
        "are_you_sure": [
            {
                "prompt": [
                    {"type": "human", "content": _TEXT % (10 + i)},
                    {"type": "ai", "content": "assistant turn"},
                ]
            }
            for i in range(3)
        ],
        "feedback": [
            {"prompt": [{"type": "human", "content": _TEXT % (20 + i)}]} for i in range(3)
        ],
        "mimicry": [{"prompt": [{"type": "human", "content": _TEXT % (30 + i)}]} for i in range(3)],
    }
    report = _gate([syco], stream_fn=_two_file_stream(rows_by_stem))
    assert report["n_verified"] == 4  # one attempt per file
    for stem in ("answer", "are_you_sure", "feedback", "mimicry"):
        assert report["attempts"][f"sycophancy_eval:{stem}"]["kept"] == 3


# ---------------------------------------------------------------------------
# Per-file STAGING through the real chain (stage_source -> CS._stream_stage):
# distinct checkpoint files/fingerprints, cumulative cap, file-stem keys.
# ---------------------------------------------------------------------------


def test_per_file_staging_distinct_checkpoints_cumulative_cap(monkeypatch, tmp_path):
    spec = _spec(
        "pf",
        "d/pf",
        data_files_template=(
            "hf://datasets/d/pf@{revision}/alpha.jsonl",
            "hf://datasets/d/pf@{revision}/beta.jsonl",
        ),
        conv_fields=("prompt",),
        cap=5,
    )
    rows = {
        "alpha.jsonl": [{"prompt": [{"type": "human", "content": _TEXT % i}]} for i in range(3)],
        "beta.jsonl": [
            {"prompt": [{"type": "human", "content": _TEXT % (10 + i)}]} for i in range(4)
        ],
    }

    def fake_resolve(dataset_id: str, revision_ref: str | None = None) -> str:
        return "0" * 40

    def fake_stream(dataset_id, config, split, **kwargs):
        # per-file attempts route via the packaged builder: ("json", None,
        # "train", data_files=<resolved url>)
        assert dataset_id == "json" and config is None and split == "train"
        url = kwargs["data_files"]
        assert "@" + "0" * 40 + "/" in url  # revision substituted into the path
        return iter(rows[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(CP, "_resolve_dataset_revision", fake_resolve)
    monkeypatch.setattr(CP.CS, "_hf_stream", fake_stream)
    kept, counters = CP.stage_source(
        spec, tmp_path, None, stream_cap=None, filter_language=False, seed=7
    )
    # cumulative cap 5: alpha keeps 3, beta keeps the remaining 2
    assert len(kept) == 5
    assert set(counters) == {"alpha", "beta"}
    assert counters["alpha"]["scanned"] == 3 and counters["beta"]["scanned"] >= 2
    staged = sorted(p.name for p in (tmp_path / "staged").glob("*.jsonl"))
    assert staged == ["pf__alpha.jsonl", "pf__beta.jsonl"], staged
    fps = {
        json.loads((tmp_path / "staged" / f"{n}.meta.json").read_text())["fingerprint"]
        for n in staged
    }
    assert len(fps) == 2  # per-file fingerprints are distinct


# ---------------------------------------------------------------------------
# run_pipeline wiring: gate BEFORE staging; skip flag; gate-only mode.
# ---------------------------------------------------------------------------


def _pipeline_args(tmp_path, *extra):
    return CP.build_argparser().parse_args(
        [
            "--probe",
            "--no-token-filter",
            "--skip-split-preflight",
            "--out-dir",
            str(tmp_path),
            "--budget",
            "5",
            *extra,
        ]
    )


def test_run_pipeline_wires_schema_gate_before_staging(monkeypatch, tmp_path):
    order: list[str] = []

    def fake_gate(sources, **kwargs):
        order.append("gate")
        return {"n_verified": len(sources), "attempts": {}, "unverifiable": []}

    def fake_stage(spec, out_dir, token_filter, **kwargs):
        order.append(f"stage:{spec.source_tag}")
        return (
            [
                {
                    "text": _TEXT % i,
                    "source_tag": spec.source_tag,
                    "dataset_id": spec.dataset_id,
                    "config": None,
                    "regime_class": spec.regime_class,
                    "realism_tier": spec.realism_tier,
                }
                for i in range(3)
            ],
            {"default": {"scanned": 3}},
        )

    monkeypatch.setattr(CP, "verify_source_schemas", fake_gate)
    monkeypatch.setattr(CP, "stage_source", fake_stage)
    monkeypatch.setattr(
        CP, "SOURCES", (_spec("s1", "d/s1", text_fields=("text",), regime_class=CP.REGIME_NEAR),)
    )
    CP.run_pipeline(_pipeline_args(tmp_path))
    assert order[0] == "gate" and order[1:] == ["stage:s1"]


def test_skip_schema_gate_flag(monkeypatch, tmp_path):
    called: list[str] = []
    monkeypatch.setattr(CP, "verify_source_schemas", lambda *a, **k: called.append("gate") or {})
    monkeypatch.setattr(
        CP,
        "stage_source",
        lambda spec, out_dir, token_filter, **kwargs: (
            [
                {
                    "text": _TEXT % 1,
                    "source_tag": spec.source_tag,
                    "dataset_id": spec.dataset_id,
                    "config": None,
                    "regime_class": spec.regime_class,
                    "realism_tier": spec.realism_tier,
                }
            ],
            {"default": {"scanned": 1}},
        ),
    )
    monkeypatch.setattr(
        CP, "SOURCES", (_spec("s1", "d/s1", text_fields=("text",), regime_class=CP.REGIME_NEAR),)
    )
    CP.run_pipeline(_pipeline_args(tmp_path, "--skip-schema-gate"))
    assert called == []


def test_schema_gate_only_writes_report_and_skips_staging(monkeypatch, tmp_path):
    def fake_gate(sources, **kwargs):
        return {"n_verified": 1, "attempts": {"s1:default": {"kept": 2}}, "unverifiable": []}

    staged: list[str] = []
    monkeypatch.setattr(CP, "verify_source_schemas", fake_gate)
    monkeypatch.setattr(
        CP,
        "stage_source",
        lambda *a, **k: staged.append("stage") or ([], {}),
    )
    monkeypatch.setattr(
        CP, "SOURCES", (_spec("s1", "d/s1", text_fields=("text",), regime_class=CP.REGIME_NEAR),)
    )
    report = CP.run_pipeline(_pipeline_args(tmp_path, "--schema-gate-only"))
    assert report["mode"] == "schema-gate-only"
    assert staged == []
    on_disk = json.loads((tmp_path / "schema_gate_report.json").read_text())
    assert on_disk["n_verified"] == 1


def test_schema_gate_only_contradicts_skip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        CP, "SOURCES", (_spec("s1", "d/s1", text_fields=("text",), regime_class=CP.REGIME_NEAR),)
    )
    with pytest.raises(SystemExit, match="contradicts"):
        CP.run_pipeline(_pipeline_args(tmp_path, "--schema-gate-only", "--skip-schema-gate"))
