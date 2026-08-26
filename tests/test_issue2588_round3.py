"""Issue #2588 review-round-3 regression tests.

Reconciler blockers: consumer-contract-post-init (every parsed row validates
pre-load), oddlayer-overwrites-primary (odd `--phase all` never re-drives
gen/parse/upload-raw), h2-paired-analysis-missing (H2 Spearman consumes RAW
complete-case gaps), judge-fallback-unintegrated (harvest-side parsed-GPQA
staging + composed judge path), p3-harvest-missing (ONE resolved sha threads
every staging call), banked-full-grain-not-exact (filtered/deduped exact
consume). Ride-alongs: gen-capture-stage-resume, g2-prodpath-tol-unpinned.

Every test executes the REAL body of the round-modified function; fakes sit
only at external boundaries (HF Hub, the Anthropic dispatch seam, GPU-scale
model loads) and are signature-conformant by construction. No network, no
GPU, tmp_path-only writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2588_panel_common as PC
import issue2588_run_cell as RC
import issue2588_trend as TR

import explore_persona_space.orchestrate.hub as hub_mod
from explore_persona_space.llm.api_dispatch import (
    RESULT_OK,
    RESULT_TRANSPORT,
    DispatchResult,
)


def _args(**over):
    """Real argparse namespace (signature-conformant by construction)."""
    a = RC._build_parser().parse_args([])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _dr(
    item_id, *, result=None, error=False, category=RESULT_OK, stop_reason="end_turn", reason=None
):
    return DispatchResult(
        item_id=item_id,
        result=result,
        error=error,
        reason=reason,
        category=category,
        stop_reason=stop_reason,
    )


class _FakeTok:
    """Signature-mirrors the tokenizer surface _banked_stage_rows touches."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(max(1, len(text) // 2)))}


# ---------------------------------------------------------------------------
# Blocker 1 (consumer-contract-post-init): EVERY parsed row validates
# ---------------------------------------------------------------------------


def test_validate_capture_inputs_checks_every_row(tmp_path, monkeypatch):
    """A two-row parsed stage whose SECOND row lacks a required key fails the
    preflight with ZERO model-loader / tokenizer hits (pre-fix only rows[0]
    was validated, so the malformed row crashed AFTER the 7-54 GB load)."""
    hits: list[str] = []
    monkeypatch.setattr(RC.G, "_load_capture_model", lambda *a, **k: hits.append("model"))
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: hits.append("tok")),
    )
    args = _args()
    cell = PC.Cell("q35_9b", "b", True)  # fresh: parsed-jsonl branch
    paths = {"parsed": tmp_path / "parsed", "cell": tmp_path / "cell"}
    for p in paths.values():
        p.mkdir(parents=True)
    good = {
        "row_id": "train_10k_0",
        "prompt": "p",
        "n_prompt_tokens": 4,
        "text": "t",
        "ans_char_span": [0, 1],
    }
    bad = {k: v for k, v in good.items() if k != "ans_char_span"}  # 2nd row malformed
    bad["row_id"] = "train_10k_1"
    with (paths["parsed"] / "train_10k.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(good) + "\n")
        fh.write(json.dumps(bad) + "\n")
    with pytest.raises(AssertionError, match=r"rows missing required keys.*ans_char_span"):
        RC.phase_capture(args, cell, paths)
    assert hits == []  # zero model-loader hits — the reconciler's exact bar


# ---------------------------------------------------------------------------
# Blocker 6 (banked-full-grain-not-exact): filtered / deduped exact consume
# ---------------------------------------------------------------------------


def _write_banked(paths: dict, stage: str, rows: list[dict]) -> None:
    d = paths["cell"] / "banked" / stage
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunk000.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")


def test_banked_stage_rows_exact_filtered_deduped(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "_banked_expected_ids", lambda split: {1, 2, 3})
    monkeypatch.setattr(RC.G, "_render_prompt", lambda tok, p: f"R({p})")
    paths = {"cell": tmp_path}
    _write_banked(
        paths,
        "test_1000",
        [
            {"ci": 1, "prompt": "p1", "response": "r1"},
            {"ci": 1, "prompt": "p1b", "response": "r1-dup"},  # duplicate -> skipped
            {"ci": 2, "prompt": "p2e", "response": "  "},  # empty first...
            {"ci": 2, "prompt": "p2", "response": "r2"},  # ...first USABLE wins
            {"ci": 3, "prompt": "p3", "response": "r3"},
            {"ci": 99, "prompt": "px", "response": "rx"},  # union-dropped extra -> filtered
        ],
    )
    rows = RC._banked_stage_rows(PC.Cell("q35_9b", "a", False), paths, "test_1000", _FakeTok())
    by_ci = {r["ci"]: r for r in rows}
    assert set(by_ci) == {1, 2, 3}  # exact: extras filtered, no dup double-count
    assert by_ci[1]["text"] == "r1"  # first usable wins over the dup
    assert by_ci[2]["text"] == "r2"  # empty row skipped, usable sibling consumed
    assert by_ci[1]["row_id"] == "test_1000_1"
    assert all(r["prompt"].startswith("R(") for r in rows)  # producer render convention


def test_banked_stage_rows_fails_loud_naming_missing_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "_banked_expected_ids", lambda split: {1, 2, 3, 4})
    monkeypatch.setattr(RC.G, "_render_prompt", lambda tok, p: p)
    paths = {"cell": tmp_path}
    _write_banked(
        paths,
        "test_1000",
        [
            {"ci": 1, "prompt": "p1", "response": "r1"},
            {"ci": 2, "prompt": "p2", "response": "r2"},
            {"ci": 3, "prompt": "p3", "response": ""},  # only-empty -> unusable
        ],
    )
    with pytest.raises(AssertionError, match=r"A19 banked-consume FAIL.*\[3, 4\]"):
        RC._banked_stage_rows(PC.Cell("q35_9b", "a", False), paths, "test_1000", _FakeTok())


def test_validate_capture_inputs_banked_usable_ids(tmp_path, monkeypatch):
    """Validation half: an expected id whose ONLY banked rows are empty fails
    the A19 matched-id assert (pre-fix presence alone passed and the id
    silently vanished at consume); duplicate cis are counted, extras never
    fail the check (they exist BY CONSTRUCTION and are filtered at consume)."""
    monkeypatch.setattr(RC, "_banked_expected_ids", lambda split: {1, 2})
    monkeypatch.setattr(RC, "_stage_names", lambda args, cell: ["test_1000"])
    args = _args()
    cell = PC.Cell("q35_9b", "a", False)  # banked branch
    paths = {"cell": tmp_path / "cell", "parsed": tmp_path / "parsed"}
    for p in paths.values():
        p.mkdir(parents=True)
    _write_banked(
        paths,
        "test_1000",
        [
            {"ci": 1, "prompt": "p1", "response": "r1"},
            {"ci": 1, "prompt": "p1b", "response": "r1b"},  # dup -> counted
            {"ci": 2, "prompt": "p2", "response": "r2"},
            {"ci": 99, "prompt": "px", "response": "rx"},  # extra -> counted, never fatal
        ],
    )
    report = RC._validate_capture_inputs(args, cell, paths)
    st = report["stages"]["test_1000"]
    assert st["n_usable_matched"] == 2
    assert st["n_duplicate_ci"] == 1
    assert st["n_extra_banked"] == 1
    # Expected id 2's only row now empty -> usable-id assert fires.
    _write_banked(
        paths,
        "test_1000",
        [
            {"ci": 1, "prompt": "p1", "response": "r1"},
            {"ci": 2, "prompt": "p2", "response": ""},
        ],
    )
    with pytest.raises(AssertionError, match=r"lack a usable \(non-empty\) banked row"):
        RC._validate_capture_inputs(args, cell, paths)


# ---------------------------------------------------------------------------
# Blocker 2 (oddlayer-overwrites-primary): sequence restriction + refusal
# ---------------------------------------------------------------------------


def test_sequence_for_odd_restricts_to_layer_dependent_phases():
    assert RC._sequence_for(_args()) == RC._ALL_SEQUENCE
    assert RC._sequence_for(_args(smoke=True)) == (*RC._ALL_SEQUENCE, "smoke-null-timing")
    assert RC._sequence_for(_args(layer_set="odd")) == RC._ODD_SEQUENCE
    assert RC._sequence_for(_args(layer_set="odd", smoke=True)) == RC._ODD_SEQUENCE
    # The odd sequence carries NO primary-artifact phase and only registered names.
    assert not set(RC._ODD_SEQUENCE) & set(RC._ODD_FORBIDDEN_PHASES)
    assert set(RC._ODD_SEQUENCE) <= set(RC.PHASES)
    # Explicit odd invocation of a primary-artifact phase is refused.
    for banned in RC._ODD_FORBIDDEN_PHASES:
        with pytest.raises(AssertionError, match="refused"):
            RC._sequence_for(_args(layer_set="odd", phase=banned))
    # Layer-dependent + idempotent staging phases stay explicitly runnable.
    assert RC._sequence_for(_args(layer_set="odd", phase="capture")) == ("capture",)
    assert RC._sequence_for(_args(layer_set="odd", phase="prologue")) == ("prologue",)


def test_swept_then_odd_never_redrives_primary(tmp_path, monkeypatch):
    """Functional swept-then-odd: the odd `--phase all` pass invokes NO
    gen/parse/upload-raw phase, and the primary raw/parsed bytes + upload
    destination record are byte-identical after it."""
    invoked: list[tuple[str, str]] = []

    def _mk(name):
        def _phase(args, cell, paths):
            invoked.append((name, args.layer_set))
            if name == "gen":
                d = paths["raw"] / "test_1000"
                d.mkdir(parents=True, exist_ok=True)
                (d / "chunk000.json").write_text(
                    json.dumps({"rows": [{"row_id": "t_0", "pass": args.layer_set}]}),
                    encoding="utf-8",
                )
            elif name == "parse":
                paths["parsed"].mkdir(parents=True, exist_ok=True)
                (paths["parsed"] / "test_1000.jsonl").write_text(
                    json.dumps({"row_id": "t_0", "pass": args.layer_set}) + "\n",
                    encoding="utf-8",
                )
            elif name == "upload-raw":
                (paths["cell"] / "upload_raw_dest.txt").write_text(
                    f"layer_set={args.layer_set}", encoding="utf-8"
                )

        return _phase

    for name in list(RC.PHASES):
        monkeypatch.setitem(RC.PHASES, name, _mk(name))
    cell = PC.Cell("q35_9b", "b", True)
    paths = {"cell": tmp_path, "raw": tmp_path / "raw", "parsed": tmp_path / "parsed"}

    a_swept = _args()
    ran = RC._run_phases(a_swept, cell, paths, RC._sequence_for(a_swept))
    assert ran == list(RC._ALL_SEQUENCE)
    raw_p = paths["raw"] / "test_1000" / "chunk000.json"
    parsed_p = paths["parsed"] / "test_1000.jsonl"
    dest_p = paths["cell"] / "upload_raw_dest.txt"
    raw_bytes, parsed_bytes, dest_bytes = (
        raw_p.read_bytes(),
        parsed_p.read_bytes(),
        dest_p.read_bytes(),
    )

    invoked.clear()
    a_odd = _args(layer_set="odd")
    ran_odd = RC._run_phases(a_odd, cell, paths, RC._sequence_for(a_odd))
    assert ran_odd == list(RC._ODD_SEQUENCE)  # own "_odd" sentinels — nothing pre-satisfied
    odd_names = {n for n, _ in invoked}
    assert odd_names == set(RC._ODD_SEQUENCE)
    assert not odd_names & set(RC._ODD_FORBIDDEN_PHASES)
    # Primary artifacts + upload destination record byte-identical.
    assert raw_p.read_bytes() == raw_bytes
    assert parsed_p.read_bytes() == parsed_bytes
    assert dest_p.read_bytes() == dest_bytes
    assert dest_bytes == b"layer_set=swept"


# ---------------------------------------------------------------------------
# Blocker 3 (h2-paired-analysis-missing): Spearman over RAW gaps
# ---------------------------------------------------------------------------


def _map_rec(perrow_cis, hits, gpqa_ids, ghits, obs=0.5):
    return {
        "fits": {
            "layer_star": 5,
            "layers": {"5": {"knn_test": {"ridge": {"cosine": {"acc_at_k": {"1": obs}}}}}},
        },
        "nulls": {"null_mean_acc1_cos": 0.01, "null_sd_acc1_cos": 0.005, "perm_draws": 200},
        "perrow": {
            "row_ids": [f"test_1000_{c}" for c in perrow_cis],
            "hit1_cos": hits,
        },
        "gpqa_perrow": {"row_ids": gpqa_ids, "same_q_hit": ghits},
        "gpqa_transfer": None,
        "resid": None,
        "judge_pending": None,
        "judge_verdicts": None,
    }


def test_h2_spearman_consumes_raw_gaps_not_calibrated():
    """Divergent-ordering fixture: raw gaps ASCEND across the 7 pairs while
    calibrated gaps DESCEND. The registered gap-vs-AA Spearman must match the
    raw-gap correlation exactly (pre-fix it consumed gaps_cal)."""
    from scipy.stats import spearmanr

    gids = [f"q{i}_s42" for i in range(8)]
    maps: dict[str, dict] = {}
    raw_gaps_expected = []
    cal_gaps_expected = []
    for idx, key in enumerate(TR.QWEN_THINKING_KEYS):
        cis_b = [str(i) for i in range(0, 9)]
        cis_a = [str(i) for i in range(1, 10)]  # shared complete-case = {1..8}
        k = idx + 1
        hits_b = [1 if i < k else 0 for i in range(9)]  # shared hits = k-1 -> gap idx/8
        obs_b = 0.9 - idx * 0.1  # calibrated gap DESCENDS in idx
        maps[TR.MapRef(key, "b", "cot_boundary").map_id] = _map_rec(
            cis_b, hits_b, gids, [0] * 8, obs=obs_b
        )
        maps[TR.MapRef(key, "a", "prompt_last").map_id] = _map_rec(
            cis_a, [0] * 9, gids, [0] * 8, obs=0.1
        )
        raw_gaps_expected.append(idx / 8)
        cal_gaps_expected.append(obs_b - 0.1)
    aa_vals = [PC.AA_PIN[k][0] for k in TR.QWEN_THINKING_KEYS]
    assert len(set(aa_vals)) > 1, "fixture vacuous: AA pins all tied"
    rho_raw = float(spearmanr(aa_vals, raw_gaps_expected)[0])
    rho_cal = float(spearmanr(aa_vals, cal_gaps_expected)[0])
    assert rho_raw != pytest.approx(rho_cal)  # orderings genuinely diverge

    universe = [str(i) for i in range(10)]
    matrix = TR._shared_resample_matrix(universe, 100, 42)
    out = TR.h2_reads(maps, universe, matrix)
    sp = out["gap_vs_aa_spearman"]
    assert sp["rho"] == pytest.approx(rho_raw)
    assert sp["gap_basis"].startswith("raw complete-case")
    # Cal/resid sensitivity semantics declared in the emitted JSON.
    sem = out["sensitivity_gap_semantics"]
    assert "aggregate-based" in sem["gap_generic_cal"]
    assert "gap_generic_resid" in sem and "gap_gpqa_resid" in sem
    # The E4 sensitivity field still rides along per pair.
    assert "gap_generic_cal" in out["pairs"]["q35_9b"]


# ---------------------------------------------------------------------------
# Blocker 5 (p3-harvest-missing): ONE resolved sha threads every staging call
# ---------------------------------------------------------------------------


def test_resolve_harvest_revision_real_body(monkeypatch):
    class _FakeInfo:
        sha = "abc123def4567890"

    class _FakeApi:
        def repo_info(self, repo_id, repo_type=None):
            assert repo_id == PC.HF_DATA_REPO and repo_type == "dataset"
            return _FakeInfo()

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    assert TR._resolve_harvest_revision(None) == "abc123def4567890"
    # Explicit pin passes through untouched (no network path).
    assert TR._resolve_harvest_revision("pinned123") == "pinned123"


def test_harvest_threads_one_sha_into_every_staging_call(tmp_path, monkeypatch):
    """main(--harvest) resolves ONE sha and threads it into BOTH
    stage_hub_prefix calls AND the judge-fallback input staging (pre-fix each
    revision=None call resolved its own sha and could straddle an upload)."""
    revs: list[str | None] = []
    judge_revs: list[str] = []

    def _fake_stage_prefix(repo_id, prefix, dest_dir, *, repo_type, revision, **kw):
        assert repo_id == PC.HF_DATA_REPO and repo_type == "dataset"
        revs.append(revision)
        return []

    monkeypatch.setattr(hub_mod, "stage_hub_prefix", _fake_stage_prefix)
    monkeypatch.setattr(TR, "_resolve_harvest_revision", lambda explicit: "shaTHREAD")
    monkeypatch.setattr(
        TR, "_stage_judge_fallback_inputs", lambda root, rev: judge_revs.append(rev) or {}
    )
    with pytest.raises(AssertionError, match="no fit artifacts"):
        TR.main(["--harvest", "--fits-dir", str(tmp_path / "fits")])
    assert revs == ["shaTHREAD", "shaTHREAD"]  # fits + nulls prefixes, SAME sha
    assert judge_revs == ["shaTHREAD"]


# ---------------------------------------------------------------------------
# Blocker 4 (judge-fallback-unintegrated): staging route + composed path
# ---------------------------------------------------------------------------


def test_stage_judge_fallback_inputs_flagged_cells_only(tmp_path, monkeypatch):
    cells = PC.all_cells()
    flagged = cells[0]
    listed: list[tuple[str, str | None]] = []
    staged_calls: list[tuple[str, str, str | None]] = []

    def _fake_list(api, repo_id, path, *, repo_type="model", revision=None):
        listed.append((path, revision))
        return [
            f"{path}/gpqa_s43.jsonl",
            f"{path}/gpqa_s42.jsonl",
            f"{path}/train_10k.jsonl",  # non-GPQA parsed file — must NOT stage
        ]

    def _fake_stage_file(repo_id, path_in_repo, target, *, repo_type, revision, **kw):
        staged_calls.append((repo_id, path_in_repo, revision))
        t = Path(target)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text("{}", encoding="utf-8")
        return t

    monkeypatch.setattr(hub_mod, "list_hf_files_under_path", _fake_list)
    monkeypatch.setattr(hub_mod, "stage_hub_file", _fake_stage_file)
    fits_root = tmp_path / "fits"
    for c in cells[:3]:
        (fits_root / c.key).mkdir(parents=True)
    (fits_root / flagged.key / "gpqa_judge_pending.json").write_text("{}", encoding="utf-8")

    staged = TR._stage_judge_fallback_inputs(fits_root, "shaZ")

    assert listed == [(f"{flagged.hf_prefix}/parsed", "shaZ")]  # flagged cell ONLY
    assert [c[1].rsplit("/", 1)[-1] for c in staged_calls] == [
        "gpqa_s42.jsonl",
        "gpqa_s43.jsonl",
    ]  # gpqa files only, sorted; train_10k filtered
    assert all(rev == "shaZ" for _, _, rev in staged_calls)
    parsed_dir = fits_root / flagged.key / "parsed"
    assert staged == {flagged.key: [parsed_dir / "gpqa_s42.jsonl", parsed_dir / "gpqa_s43.jsonl"]}
    assert (parsed_dir / "gpqa_s42.jsonl").exists()


def test_stage_judge_fallback_inputs_fails_loud_on_empty_prefix(tmp_path, monkeypatch):
    cells = PC.all_cells()
    monkeypatch.setattr(hub_mod, "list_hf_files_under_path", lambda *a, **k: [])
    fits_root = tmp_path / "fits"
    (fits_root / cells[0].key).mkdir(parents=True)
    (fits_root / cells[0].key / "gpqa_judge_pending.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="judge-fallback staging"):
        TR._stage_judge_fallback_inputs(fits_root, "shaZ")


def test_judge_fallback_composed_path(tmp_path, monkeypatch):
    """Composed path (the reconciler's B5 bar): pending + staged parsed
    fixture -> run_judge_fallback (pilot gate, one transport re-drive,
    verdict JSON on disk) -> merged_behavioral consumes the round-tripped
    verdicts into a judge-corrected accuracy."""
    calls = {"n": 0}

    def _fake_round(items, checkpoint_dir):
        calls["n"] += 1
        out = {}
        letters = {"r0": "B", "r1": "A", "r2": "UNPARSEABLE"}
        for it in items:
            if it.item_id == "r1" and calls["n"] == 1:
                out[it.item_id] = _dr(
                    it.item_id, error=True, category=RESULT_TRANSPORT, stop_reason=None
                )
            else:
                # The payload carries the staged parsed answer text + prompt.
                assert it.payload["question"].startswith("Q")
                assert it.payload["answer"]
                out[it.item_id] = _dr(it.item_id, result=letters[it.item_id])
        return out

    monkeypatch.setattr(TR, "_dispatch_judge_round", _fake_round)
    pending = {
        "rows": [
            {"row_id": "r0", "qid": "q0", "gold": "B"},
            {"row_id": "r1", "qid": "q1", "gold": "B"},
            {"row_id": "r2", "qid": "q2", "gold": "C"},
        ]
    }
    pending_path = tmp_path / "gpqa_judge_pending.json"
    pending_path.write_text(json.dumps(pending), encoding="utf-8")
    prompts_path = tmp_path / "gpqa_prompts.json"
    prompts_path.write_text(
        json.dumps({"prompts": [{"qid": f"q{i}", "prompt": f"Q{i}?"} for i in range(3)]}),
        encoding="utf-8",
    )
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    with (parsed_dir / "gpqa_s42.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(
                json.dumps(
                    {"row_id": f"r{i}", "text": f"rollout answer {i}", "ans_char_span": [0, 14]}
                )
                + "\n"
            )
    out_path = tmp_path / "gpqa_judge_verdicts.json"

    rec = TR.run_judge_fallback(pending_path, parsed_dir, out_path, prompts_path=prompts_path)

    assert calls["n"] == 2  # pilot round + ONE transport re-drive (never persisted as a drop)
    assert out_path.exists()
    assert rec["n_items"] == 3
    assert rec["n_correct"] == 1  # r0 only (r1 judged 'A' vs gold B)
    assert rec["n_unparseable"] == 1
    assert rec["n_transport_persisted"] == 0
    assert rec["pilot"]["gates"]["zero_max_tokens"] is True

    # merged_behavioral consumes the ROUND-TRIPPED files (JSON off disk).
    beh = {
        "judge_fallback_flagged": True,
        "frac_unparseable": 0.3,
        "n_rollouts": 10,
        "n_correct": 4,
    }
    full_rec = {
        "gpqa_transfer": {"behavioral": beh},
        "judge_pending": json.loads(pending_path.read_text(encoding="utf-8")),
        "judge_verdicts": json.loads(out_path.read_text(encoding="utf-8")),
    }
    merged = TR.merged_behavioral(full_rec, "m.x")
    assert merged["acc_judge_corrected"] == pytest.approx((4 + 1) / 10)
    assert merged["n_judge_corrected"] == 1
    assert merged["frac_unparseable_after_judge"] == pytest.approx(1 / 10)


# ---------------------------------------------------------------------------
# Ride-along 8 (g2-prodpath-tol-unpinned): sentinel's own tol never trusted
# ---------------------------------------------------------------------------


def _valid_sentinel():
    return {
        "schema_version": PC.G2_SENTINEL_SCHEMA_VERSION,
        "status": "PASS",
        "store_revision_pin_recorded": RC.MF.STORE_REVISION_PIN_7B,
        "expected_r2": PC.ANCHOR_EXPECTED_R2,
        "realized_r2": PC.ANCHOR_EXPECTED_R2 + 1e-8,
        "abs_deviation": 1e-8,
        "tol": PC.ANCHOR_TOL,
        "production_path": {
            "estimator": "_fit_edge_extended_with_val",
            "realized_r2": PC.ANCHOR_EXPECTED_R2 + 2e-6,
            "abs_deviation_vs_pin": 2e-6,
            "tol": PC.ANCHOR_PROD_EQUIV_TOL,
        },
        "meta": {"git_sha": "deadbeefcafe"},
    }


def test_validate_g2_sentinel_pins_prodpath_tol():
    RC._validate_g2_sentinel(_valid_sentinel())  # in-tolerance sentinel accepted
    # A sentinel minted under a LOOSER self-reported tol is refused: its
    # deviation passes its own pp["tol"] but fails the CURRENT pin.
    loose = _valid_sentinel()
    loose["production_path"]["tol"] = 1.0
    loose["production_path"]["abs_deviation_vs_pin"] = 0.5
    assert loose["production_path"]["tol"] >= 0.5  # would pass its own tol
    assert PC.ANCHOR_PROD_EQUIV_TOL < 0.5
    with pytest.raises(AssertionError, match="pinned tolerance"):
        RC._validate_g2_sentinel(loose)


# ---------------------------------------------------------------------------
# Ride-along 7 (gen-capture-stage-resume): per-stage terminal-artifact skip
# ---------------------------------------------------------------------------


def test_gen_stage_skips_on_cap_hit_report(tmp_path):
    """cap_hit_report.json present -> the stage returns its persisted rows
    without touching the engine holder (llm_holder untouched proves no
    engine build was attempted); --force is the re-run escape."""
    args = _args()
    cell = PC.Cell("q35_9b", "b", True)
    paths = {"raw": tmp_path / "raw"}
    d = paths["raw"] / "test_1000"
    d.mkdir(parents=True)
    rows = [{"row_id": "test_1000_0", "text": "t"}]
    (d / "chunk000.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    (d / "cap_hit_report.json").write_text(json.dumps({"cap_hit_frac": 0.0}), encoding="utf-8")
    holder: dict = {}
    out = RC._gen_stage_with_regen(
        args, cell, None, [], stage="test_1000", cap=64, seed=42, paths=paths, llm_holder=holder
    )
    assert out == rows
    assert holder == {}  # engine never built — the skip fired before the holder


def test_capture_stage_skips_on_rows_json(tmp_path):
    """rows.json present -> _capture_stage returns before reading parsed
    inputs or touching the model (hf=None would crash any real use);
    --force bypasses the skip and proceeds (fails on the missing parsed
    input — proof the skip was the only thing short-circuiting)."""
    args = _args()
    cell = PC.Cell("q35_9b", "b", True)  # fresh -> parsed branch after the skip
    paths = {"capture": tmp_path / "capture", "parsed": tmp_path / "parsed"}
    d = paths["capture"] / "test_1000"
    d.mkdir(parents=True)
    (d / "rows.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    RC._capture_stage(args, cell, paths, None, None, "test_1000", [0])  # returns cleanly

    args.force = True
    with pytest.raises(FileNotFoundError):
        RC._capture_stage(args, cell, paths, None, None, "test_1000", [0])


# ---------------------------------------------------------------------------
# Standing rec: SR1 degenerate ceiling raises (never a silent None)
# ---------------------------------------------------------------------------


def test_calibrated_sr1_degenerate_ceiling_raises():
    rec = {
        "fits": {
            "layer_star": 5,
            "layers": {"5": {"knn_test": {"ridge": {"cosine": {"acc_at_k": {"1": 0.5}}}}}},
            "ceiling_retrieval_at_star": {"ceiling_acc1_cos": 0.01},  # == null mean
        },
        "nulls": {"null_mean_acc1_cos": 0.01, "null_sd_acc1_cos": 0.005, "perm_draws": 200},
    }
    with pytest.raises(ValueError, match="SR1 ceiling normalization degenerate"):
        TR._calibrated(rec)
    # Healthy ceiling still yields the normalized read.
    rec["fits"]["ceiling_retrieval_at_star"] = {"ceiling_acc1_cos": 0.8}
    out = TR._calibrated(rec)
    assert out["acc1_cos_ceiling_normalized"] == pytest.approx((0.5 - 0.01) / (0.8 - 0.01))
