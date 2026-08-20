"""Issue #2389 staged-anchor consumer probes (plan §4.6 M1-iv) — unit tests.

Offline (``--local-anchors-dir``): the probes run the REAL consumer loaders
(``issue2389_judge.load_anchor_rows`` + ``issue2389_analysis._load_anchor_va``)
over synthetic tmp shard pairs shaped exactly like the pod's cell-grain gate
uploads (``anchors_gate_{cell}_w{w}.jsonl`` + ``va_anchors_gate_{cell}_w{w}.pt``,
B7 r1 review). No network: shard DISCOVERY runs the real ``_discover_gate_shard``
body against an autospec'd Hub listing, and the staging leg is covered by a
signature-bind of the script's exact ``hub.stage_hub_file`` call shape against
the real helper. The M1-iv gate (``issue2389_judge.require_consumer_probe``)
is exercised against reports the real ``main()`` writes.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2389_analysis as A  # noqa: E402
import issue2389_consumer_probe as P  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

H = 4
ROWS = [
    ("ctx_a", 0),
    ("ctx_a", 1),
    ("ctx_b", 0),
]


def _write_jsonl(anchors_dir: Path, keys: list[tuple[str, int]]) -> None:
    lines = [
        json.dumps(
            {
                "context_id": cid,
                "cell": "fact_user_name",
                "value_id": "v0",
                "carrier": "c0",
                "draw": draw,
                "text": f"rollout {cid}/{draw}",
            }
        )
        for cid, draw in keys
    ]
    (anchors_dir / "anchors_gate_fact_user_name_w0.jsonl").write_text("\n".join(lines) + "\n")


def _write_pt(
    anchors_dir: Path,
    keys: list[tuple[str, int]],
    empty_rows: list[int],
    name: str = "va_anchors_gate_fact_user_name_w0.pt",
) -> None:
    payload = {
        "layers": [A.READ_LAYER],
        "va_span": torch.randn(len(keys), 1, H),
        "index": [{"context_id": cid, "draw": draw} for cid, draw in keys],
        "empty_rows": list(empty_rows),
    }
    torch.save(payload, anchors_dir / name)


@pytest.fixture()
def anchors_dir(tmp_path: Path) -> Path:
    d = tmp_path / "anchors"
    d.mkdir()
    return d


def test_happy_path_both_probes_pass_and_report(anchors_dir: Path, tmp_path: Path) -> None:
    # Row 2 (ctx_b, 0) declared empty: jsonl-present, va-excluded — still PASS.
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS, empty_rows=[2])
    report_path = tmp_path / "gates" / "consumer_probe_report.json"
    rc = P.main(
        [
            "--probe",
            "both",
            "--local-anchors-dir",
            str(anchors_dir),
            "--report",
            str(report_path),
        ]
    )
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["legs"]["judge"]["n_rows"] == 3
    assert report["legs"]["analysis"]["n_va_keys"] == 2
    assert report["legs"]["analysis"]["n_declared_empty"] == 1
    assert report["legs"]["analysis"]["key_sets_identical_modulo_empty"] is True
    assert report["staged"] is False
    assert report["repro"]["script"] == "scripts/issue2389_consumer_probe.py"


def test_jsonl_key_missing_from_va_raises(anchors_dir: Path) -> None:
    # pt drops (ctx_b, 0) entirely (not declared empty) -> contract mismatch.
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS[:2], empty_rows=[])
    with pytest.raises(AssertionError, match="key-set mismatch"):
        P.probe_analysis(anchors_dir)


def test_va_key_absent_from_jsonl_raises(anchors_dir: Path) -> None:
    # pt carries an extra (ctx_c, 0) the jsonl never produced.
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, [*ROWS, ("ctx_c", 0)], empty_rows=[])
    with pytest.raises(AssertionError, match="key-set mismatch"):
        P.probe_analysis(anchors_dir)


def test_loaded_and_declared_empty_overlap_raises(anchors_dir: Path) -> None:
    # Shard 1 LOADS (ctx_a, 0); shard 2 declares the same key empty ->
    # duplicate/stale-shard overlap, its own error leg.
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS, empty_rows=[], name="va_anchors_gate_fact_user_name_w0.pt")
    _write_pt(
        anchors_dir, [("ctx_a", 0)], empty_rows=[0], name="va_anchors_gate_fact_user_name_w1.pt"
    )
    with pytest.raises(AssertionError, match=r"declared\s+empty"):
        P.probe_analysis(anchors_dir)


def test_stage_call_shape_binds_to_real_helper() -> None:
    # The offline tests never stage; pin the script's exact call shape
    # against the real hub.stage_hub_file signature instead (arity/keyword
    # drift fails here, not at pod time). Cell-grain names (B7).
    sig = inspect.signature(hub.stage_hub_file)
    sig.bind(
        P.J.DATASET_REPO,
        f"{P.J._STAGE_ANCHORS_GATE}/anchors_gate_fact_user_name_w0.jsonl",
        Path("/tmp/x/anchors_gate_fact_user_name_w0.jsonl"),
        repo_type="dataset",
        revision=None,
        overwrite=True,
    )
    sig.bind(
        P.J.DATASET_REPO,
        f"{P._VA_ANCHORS_REMOTE_PREFIX}/va_anchors_gate_fact_user_name_w0.pt",
        Path("/tmp/x/va_anchors_gate_fact_user_name_w0.pt"),
        repo_type="dataset",
        revision=None,
        overwrite=True,
    )


def test_local_anchors_dir_must_exist(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="not a directory"):
        P.main(["--probe", "judge", "--local-anchors-dir", str(tmp_path / "nope")])


def test_stage_gate_shard_invokes_real_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B12 (r1 review): EXECUTE the real _stage_gate_shard body — the network
    boundary (hub.stage_hub_file) is autospec'd, so arity/keyword drift AND a
    wrong destination composition fail HERE, not at pod time."""
    fake = create_autospec(hub.stage_hub_file)
    monkeypatch.setattr(hub, "stage_hub_file", fake)
    jsonl = "anchors_gate_fact_user_name_w0.jsonl"
    pt = "va_anchors_gate_fact_user_name_w0.pt"
    P._stage_gate_shard(tmp_path, jsonl, pt, with_va=True, revision="deadbeef")
    assert fake.call_count == 2
    c1, c2 = fake.call_args_list
    assert c1.args == (P.J.DATASET_REPO, f"{P.J._STAGE_ANCHORS_GATE}/{jsonl}", tmp_path / jsonl)
    assert c2.args == (P.J.DATASET_REPO, f"{P._VA_ANCHORS_REMOTE_PREFIX}/{pt}", tmp_path / pt)
    for c in (c1, c2):
        assert c.kwargs == {"repo_type": "dataset", "revision": "deadbeef", "overwrite": True}
    # judge-only leg stages the jsonl alone
    P._stage_gate_shard(tmp_path, jsonl, pt, with_va=False, revision=None)
    assert fake.call_count == 3


# ---- shard discovery (B7 cell-grain names: never derived, always listed) ----

_GATE_PREFIX = P.J._STAGE_ANCHORS_GATE


def _fake_listing(monkeypatch: pytest.MonkeyPatch, names: list[str]):
    """Autospec'd Hub listing (the ONLY network boundary in discovery)."""
    fake = create_autospec(hub.list_hf_files_under_path, return_value=names)
    monkeypatch.setattr(hub, "list_hf_files_under_path", fake)
    return fake


def test_discover_gate_shard_first_sorted_and_pt_stem_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_listing(
        monkeypatch,
        [
            f"{_GATE_PREFIX}/anchors_gate_persona_prompted_w3.jsonl",
            f"{_GATE_PREFIX}/anchors_gate_fact_user_name_w0.jsonl",
            f"{_GATE_PREFIX}/anchors_gate_fact_user_name_w0_gen_done.json",
        ],
    )
    jsonl, pt = P._discover_gate_shard(None, None, None)
    assert jsonl == "anchors_gate_fact_user_name_w0.jsonl"
    assert pt == "va_anchors_gate_fact_user_name_w0.pt"
    # The listing is SCOPED to the gate prefix (never a full-repo walk).
    assert fake.call_args.args[2] == _GATE_PREFIX
    assert fake.call_args.kwargs["repo_type"] == "dataset"


def test_discover_gate_shard_cell_and_worker_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_listing(
        monkeypatch,
        [
            f"{_GATE_PREFIX}/anchors_gate_fact_user_name_w0.jsonl",
            f"{_GATE_PREFIX}/anchors_gate_persona_prompted_w3.jsonl",
        ],
    )
    jsonl, pt = P._discover_gate_shard("persona_prompted", None, None)
    assert jsonl == "anchors_gate_persona_prompted_w3.jsonl"
    assert pt == "va_anchors_gate_persona_prompted_w3.pt"
    jsonl, _ = P._discover_gate_shard(None, 3, None)
    assert jsonl == "anchors_gate_persona_prompted_w3.jsonl"


def test_discover_gate_shard_no_match_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_listing(monkeypatch, [f"{_GATE_PREFIX}/anchors_gate_fact_user_name_w0.jsonl"])
    with pytest.raises(AssertionError, match="no gate shard"):
        P._discover_gate_shard("query_content", None, None)


# ---- M1-iv gate: issue2389_judge.require_consumer_probe ----


def test_require_consumer_probe_missing_report_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="consumer probe report missing"):
        P.J.require_consumer_probe(tmp_path / "gates" / "consumer_probe_report.json", "judge")


def test_require_consumer_probe_non_pass_raises(tmp_path: Path) -> None:
    rp = tmp_path / "consumer_probe_report.json"
    rp.write_text(json.dumps({"verdict": "FAIL", "legs": {"judge": {}}}))
    with pytest.raises(RuntimeError, match="not a PASS"):
        P.J.require_consumer_probe(rp, "judge")


def test_require_consumer_probe_legless_pre_verdict_report_raises(tmp_path: Path) -> None:
    # A pre-M1-iv report (no verdict field at all) must not satisfy the gate.
    rp = tmp_path / "consumer_probe_report.json"
    rp.write_text(json.dumps({"probe": "judge", "legs": {"judge": {}}}))
    with pytest.raises(RuntimeError, match="not a PASS"):
        P.J.require_consumer_probe(rp, "judge")


def test_require_consumer_probe_missing_leg_raises(tmp_path: Path) -> None:
    rp = tmp_path / "consumer_probe_report.json"
    rp.write_text(json.dumps({"verdict": "PASS", "legs": {"judge": {}}}))
    with pytest.raises(RuntimeError, match="never ran leg 'analysis'"):
        P.J.require_consumer_probe(rp, "analysis")


def test_require_consumer_probe_skip_records_durable_override(tmp_path: Path) -> None:
    rp = tmp_path / "gates" / "consumer_probe_report.json"  # deliberately absent
    P.J.require_consumer_probe(rp, "analysis", skip=True)
    rec = json.loads((tmp_path / "gates" / "consumer_probe_override_analysis.json").read_text())
    assert rec["leg"] == "analysis"
    assert rec["skipped_via"] == "--skip-consumer-probe"
    assert rec["report_path"] == str(rp)


def _bank_file(tmp_path: Path, payload: str = "bank-A") -> Path:
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"frozen": payload}))
    return bank


def _pass_report(anchors_dir: Path, tmp_path: Path, bank: Path | None = None) -> Path:
    """A REAL local-mode PASS report over ``anchors_dir`` (both legs)."""
    report_path = tmp_path / "gates" / "consumer_probe_report.json"
    argv = [
        "--probe",
        "both",
        "--local-anchors-dir",
        str(anchors_dir),
        "--report",
        str(report_path),
    ]
    if bank is not None:
        argv += ["--bank-json", str(bank)]
    assert P.main(argv) == 0
    return report_path


def test_probe_report_satisfies_gate_end_to_end(anchors_dir: Path, tmp_path: Path) -> None:
    # The real main() writes verdict: PASS + both legs + the R4 source
    # identity; the M1-iv gate the judge phase_waves / analysis
    # f-tables+stats entries call accepts it when BOUND to the same
    # anchors bytes + bank identity.
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS, empty_rows=[])
    bank = _bank_file(tmp_path)
    report_path = _pass_report(anchors_dir, tmp_path, bank)
    report = json.loads(report_path.read_text())
    assert report["verdict"] == "PASS"
    assert report["gate_shard"] is None  # local mode: nothing discovered/staged
    src = report["source"]
    assert src["staged"] is False and src["shards"] and src["bank_sha256"]
    P.J.require_consumer_probe(report_path, "judge", anchors_dir=anchors_dir, bank_json=bank)
    P.J.require_consumer_probe(report_path, "analysis", anchors_dir=anchors_dir, bank_json=bank)


# ---- R4 (r2 review): the PASS artifact is BOUND to the run it certifies ----


def test_require_consumer_probe_refuses_sourceless_report(tmp_path: Path) -> None:
    # A pre-R4 / hand-built PASS with no source identity never unlocks spend.
    rp = tmp_path / "consumer_probe_report.json"
    rp.write_text(json.dumps({"verdict": "PASS", "legs": {"judge": {}}}))
    with pytest.raises(RuntimeError, match=r"NOT BOUND.*no source identity"):
        P.J.require_consumer_probe(rp, "judge")


def test_require_consumer_probe_refuses_stale_shard_hash(anchors_dir: Path, tmp_path: Path) -> None:
    # PASS produced for run A; run B's staged copy of the SAME shard differs
    # (capregen re-gen / pre-rename store / foreign root) -> refuse.
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS, empty_rows=[])
    bank = _bank_file(tmp_path)
    report_path = _pass_report(anchors_dir, tmp_path, bank)
    dir_b = tmp_path / "anchors_b"
    dir_b.mkdir()
    _write_jsonl(dir_b, ROWS[:2])  # same filename, different bytes
    _write_pt(dir_b, ROWS[:2], empty_rows=[])
    with pytest.raises(RuntimeError, match=r"NOT BOUND.*hash != this run's staged copy"):
        P.J.require_consumer_probe(report_path, "judge", anchors_dir=dir_b, bank_json=bank)


def test_require_consumer_probe_refuses_foreign_or_unrecorded_bank(
    anchors_dir: Path, tmp_path: Path
) -> None:
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS, empty_rows=[])
    bank_a = _bank_file(tmp_path)
    report_path = _pass_report(anchors_dir, tmp_path, bank_a)
    bank_b = tmp_path / "bank_b.json"
    bank_b.write_text(json.dumps({"frozen": "bank-B"}))
    with pytest.raises(RuntimeError, match=r"NOT BOUND.*bank_sha256 != this run's bank"):
        P.J.require_consumer_probe(report_path, "judge", anchors_dir=anchors_dir, bank_json=bank_b)
    # a report that never recorded a bank digest is refused by a bank-passing
    # consumer (forces production probes to run with --bank-json).
    unbound = _pass_report(anchors_dir, tmp_path / "unbound", bank=None)
    with pytest.raises(RuntimeError, match=r"NOT BOUND.*recorded no bank_sha256"):
        P.J.require_consumer_probe(unbound, "judge", anchors_dir=anchors_dir, bank_json=bank_a)


def test_require_consumer_probe_local_report_needs_anchor_binding(
    anchors_dir: Path, tmp_path: Path
) -> None:
    # A local-mode (staged: false) PASS is acceptable ONLY when the consumer
    # binds it to its own anchors dir — presented unbound, it is refused.
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS, empty_rows=[])
    report_path = _pass_report(anchors_dir, tmp_path, _bank_file(tmp_path))
    with pytest.raises(RuntimeError, match=r"NOT BOUND.*no anchors dir to bind"):
        P.J.require_consumer_probe(report_path, "judge")


# ---- R6 (r2 review): the gate is tested THROUGH the protected entrypoints ----


def _judge_cfg(tmp_path: Path, anchors_dir: Path, bank: Path, report: Path, **over):
    kw = dict(
        work_root=tmp_path / "work",
        cache_root=tmp_path / "cache",
        rollouts_dir=tmp_path / "rollouts",
        anchors_file=anchors_dir,
        stage2_dir=None,
        bank_json=bank,
        consumer_probe_report=report,
    )
    kw.update(over)
    return P.J.JudgeConfig(**kw)


def _boundary_fake_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake ONLY the non-probe gate reads (signature-mirroring def); the
    entrypoint body up to and past the M1-iv gate is REAL."""
    monkeypatch.setattr(P.J, "_require_gates", lambda cfg, names=(): None)


def _loader_sentinel_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-5 D (concern m1iv-entrypoint-test-detached): the sentinels sit
    on the FIRST REAL LOADERS — reaching surviving_pairs / load_grid_rows
    means the M1-iv gate did not refuse first (the r4 defect: the gate ran
    only AFTER both loaders had already read banked artifacts)."""

    def _boom_pairs(bank_json):
        raise AssertionError("banked artifact loaded before the M1-iv gate (surviving_pairs)")

    def _boom_rows(rollouts_dir):
        raise AssertionError("banked artifact loaded before the M1-iv gate (load_grid_rows)")

    monkeypatch.setattr(P.J, "surviving_pairs", _boom_pairs)
    monkeypatch.setattr(P.J, "load_grid_rows", _boom_rows)


def _spend_sentinel_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(pairs):
        raise AssertionError("production spend reached (rubric_registry)")

    monkeypatch.setattr(P.J, "rubric_registry", _boom)


def test_phase_waves_refuses_before_spend_without_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6 + round-5 D: the REAL phase_waves entrypoint refuses at the M1-iv
    gate BEFORE the first banked-artifact loader (and a fortiori before any
    spend) when the probe report is absent. Reordering the gate after the
    loaders fires the loader sentinel; deleting it fires the spend sentinel."""
    _boundary_fake_judge(monkeypatch)
    _loader_sentinel_judge(monkeypatch)
    _spend_sentinel_judge(monkeypatch)
    cfg = _judge_cfg(
        tmp_path, tmp_path / "anchors", tmp_path / "bank.json", tmp_path / "missing_report.json"
    )
    with pytest.raises(RuntimeError, match="consumer probe report missing"):
        P.J.phase_waves(cfg)


def test_phase_waves_refuses_stale_source_before_spend(
    anchors_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4 x R6 + round-5 D: a PASS produced for run A (different shard bytes)
    presented to phase_waves for run B refuses BEFORE the first loader."""
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS, empty_rows=[])
    bank = _bank_file(tmp_path)
    report = _pass_report(anchors_dir, tmp_path, bank)
    dir_b = tmp_path / "anchors_b"
    dir_b.mkdir()
    _write_jsonl(dir_b, ROWS[:1])
    _boundary_fake_judge(monkeypatch)
    _loader_sentinel_judge(monkeypatch)
    _spend_sentinel_judge(monkeypatch)
    cfg = _judge_cfg(tmp_path, dir_b, bank, report)
    with pytest.raises(RuntimeError, match="NOT BOUND"):
        P.J.phase_waves(cfg)


def test_phase_waves_dry_run_not_gated_by_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-5 D trap guard: --dry-run is a zero-API construction check and
    needs NO probe — hoisting the M1-iv gate wholesale above the dry-run
    split would break the sanctioned no-probe dry run. The loaders DO run
    (they feed the dry-run unit construction); the gate does not."""
    monkeypatch.setattr(P.J, "surviving_pairs", lambda bank_json: [])
    monkeypatch.setattr(P.J, "load_grid_rows", lambda rollouts_dir: [])
    _spend_sentinel_judge(monkeypatch)
    cfg = _judge_cfg(
        tmp_path,
        tmp_path / "anchors",
        tmp_path / "bank.json",
        tmp_path / "missing_report.json",
        dry_run=True,
    )
    assert P.J.phase_waves(cfg) == P.J.RC_OK


def _analysis_args(tmp_path: Path, **over):
    import argparse

    kw = dict(
        consumer_probe_report=tmp_path / "missing_report.json",
        skip_consumer_probe=False,
        anchors_dir=None,
        bank_json=None,
        out_dir=tmp_path / "f_metrics",
    )
    kw.update(over)
    return argparse.Namespace(**kw)


def test_step_f_tables_refuses_before_load_without_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(args):
        raise AssertionError("production load reached (_pairs)")

    monkeypatch.setattr(A, "_pairs", _boom)
    with pytest.raises(RuntimeError, match="consumer probe report missing"):
        A.step_f_tables(_analysis_args(tmp_path))


def test_step_stats_refuses_before_load_without_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(path):
        raise AssertionError("production load reached (_iter_jsonl)")

    monkeypatch.setattr(A, "_iter_jsonl", _boom)
    with pytest.raises(RuntimeError, match="consumer probe report missing"):
        A.step_stats(_analysis_args(tmp_path))


def test_step_f_tables_refuses_stale_source_before_load(
    anchors_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4 x R6 (analysis side): run A's PASS + run B's bank digest refuses
    at the REAL step_f_tables entry before any loader runs."""
    _write_jsonl(anchors_dir, ROWS)
    _write_pt(anchors_dir, ROWS, empty_rows=[])
    report = _pass_report(anchors_dir, tmp_path, _bank_file(tmp_path))
    bank_b = tmp_path / "bank_b.json"
    bank_b.write_text(json.dumps({"frozen": "bank-B"}))

    def _boom(args):
        raise AssertionError("production load reached (_pairs)")

    monkeypatch.setattr(A, "_pairs", _boom)
    args = _analysis_args(
        tmp_path, consumer_probe_report=report, anchors_dir=anchors_dir, bank_json=bank_b
    )
    with pytest.raises(RuntimeError, match="NOT BOUND"):
        A.step_f_tables(args)
