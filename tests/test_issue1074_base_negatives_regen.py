"""Offline tests for the #1074 follow-up `base-negatives-regen` surfaces.

Covers the datagen positives-reuse seam (plan v7 implementation delta 1):
kept-set reconstruction pinned against a synthetic fixture, RNG-replay
equivalence (a reuse run's NEGATIVE request schedule == a fresh run's on the
same seed/fixture), every fail-loud path, manifest/pool_meta provenance
fields + the exact-match resume contract — and the driver/aggregate followup
pure logic (staging seam, consumer-open probe, CLI, negative-yield table).

All model/judge boundaries are the sanctioned injected stubs (no network).
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import pytest

from explore_persona_space.artifacts.behavior import BEHAVIORS
from explore_persona_space.artifacts.datagen import (
    NEGATIVE,
    POSITIVE,
    DatagenCheckpointMismatchError,
    DatagenYieldError,
    GenCandidate,
    PosReuseSpec,
    compose_positive_schedule,
    generate_training_data,
)
from tests.test_artifacts_datagen import SRC, _judge_by_arm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue1074_aggregate as aggregate  # noqa: E402
import issue1074_generator_compare as driver  # noqa: E402

BEH = BEHAVIORS["sycophancy"]  # threshold 50, no structural predicate
PROVENANCE = {
    "source_repo": "test-repo",
    "source_path": "test/prefix",
    "revision": "deadbeef",
    "pos_gen_model": "parent-model-y",
}


def _recording_gen(record: list):
    """Deterministic generate_fn stub that records every request it serves."""

    def gen(requests):
        record.extend(requests)
        return [GenCandidate(r, f"resp::{r.request_id}") for r in requests]

    return gen


def _fresh_run(tmp_path: Path, name: str = "fresh") -> tuple[Path, list]:
    """A producing run: fresh positives + negatives with deterministic stubs."""
    seen: list = []
    out = tmp_path / name
    generate_training_data(
        BEH,
        SRC,
        "default_v1",
        out_dir=out,
        target_n=10,
        n_judge_draws=2,
        gen_model="parent-model-y",
        generate_fn=_recording_gen(seen),
        judge_fn=_judge_by_arm(),
        instruction_style="plain",
    )
    return out, seen


def _reuse_spec(src_dir: Path, expected_kept: int, **overrides) -> PosReuseSpec:
    kw = dict(
        raw_pos_path=src_dir / "raw_pos.jsonl",
        judge_rows_path=src_dir / "judge_rows.jsonl",
        expected_kept_count=expected_kept,
        provenance=PROVENANCE,
    )
    kw.update(overrides)
    return PosReuseSpec(**kw)


def _reuse_run(tmp_path: Path, spec: PosReuseSpec, name: str = "reuse") -> tuple[Path, list]:
    seen: list = []
    out = tmp_path / name
    generate_training_data(
        BEH,
        SRC,
        "default_v1",
        out_dir=out,
        target_n=10,
        n_judge_draws=2,
        gen_model="live-model-x",  # the LIVE (negative-stage) generator
        generate_fn=_recording_gen(seen),
        judge_fn=_judge_by_arm(),
        instruction_style="plain",
        reuse_pos=spec,
    )
    return out, seen


def _kept_count(out_dir: Path) -> int:
    return json.loads((out_dir / "pool_meta.json").read_text())["positive"]["kept"]


def _schedule(reqs, arm):
    return [(r.request_id, r.question_id, r.variant_id) for r in reqs if r.arm == arm]


# ── kept-set reconstruction + RNG-replay equivalence (plan delta 1 tests) ────


def test_reuse_reconstruction_and_negative_schedule_equivalence(tmp_path):
    fresh_out, fresh_seen = _fresh_run(tmp_path)
    kept = _kept_count(fresh_out)
    assert kept == 15  # every judgeable positive kept by the stub judge (ceil(10/0.7))

    reuse_out, reuse_seen = _reuse_run(tmp_path, _reuse_spec(fresh_out, kept))

    # NEVER re-generates (or re-judges) positives: the reuse run's generate_fn
    # saw ZERO positive requests, and no positive judge cache dir was created.
    assert _schedule(reuse_seen, POSITIVE) == []
    assert not list(reuse_out.glob("judge_cache_*/pos"))

    # RNG-state replay: the negative request schedule is IDENTICAL (same
    # question x oppose-variant x member picks -> exactly paired S3').
    assert _schedule(reuse_seen, NEGATIVE) == _schedule(fresh_seen, NEGATIVE)
    assert _schedule(reuse_seen, NEGATIVE), "no negative requests recorded"

    # Byte-identical emitted artifacts (emission subsample is the same code
    # path on the same reconstructed kept list; stub completions deterministic).
    for fname in ("pos.jsonl", "cn.jsonl", "raw_pos.jsonl"):
        assert (reuse_out / fname).read_bytes() == (fresh_out / fname).read_bytes(), fname

    # The reconstructed judge_rows positive rows match the producing run's.
    def _pos_rows(p: Path) -> list[dict]:
        rows = [json.loads(ln) for ln in p.read_text().split("\n") if ln.strip()]
        return [r for r in rows if r["arm"] == POSITIVE]

    assert _pos_rows(reuse_out / "judge_rows.jsonl") == _pos_rows(fresh_out / "judge_rows.jsonl")


def test_reuse_manifest_and_pool_meta_provenance(tmp_path):
    fresh_out, _ = _fresh_run(tmp_path)
    kept = _kept_count(fresh_out)
    reuse_out, _ = _reuse_run(tmp_path, _reuse_spec(fresh_out, kept))

    manifest = json.loads((reuse_out / "gen_manifest.json").read_text())
    assert manifest["gen_model"] == "live-model-x"  # the LIVE negative generator
    pr = manifest["pos_reuse"]
    for k, v in PROVENANCE.items():
        assert pr[k] == v
    assert pr["expected_kept_count"] == kept
    assert len(pr["raw_pos_sha256"]) == 64 and len(pr["judge_rows_sha256"]) == 64

    pool_meta = json.loads((reuse_out / "pool_meta.json").read_text())
    assert pool_meta["pos_reuse"]["judge_draw_stats_reconstructed"] is True
    assert pool_meta["pos_reuse"]["pos_gen_model"] == "parent-model-y"
    assert pool_meta["gen_model"] == "live-model-x"

    # The fresh (non-reuse) manifest carries NO pos_reuse key (additive-only).
    assert "pos_reuse" not in json.loads((fresh_out / "gen_manifest.json").read_text())


def test_reuse_resume_exact_match_and_mismatch(tmp_path):
    fresh_out, _ = _fresh_run(tmp_path)
    kept = _kept_count(fresh_out)
    # Stage into a mutable copy so the sha perturbation cannot touch fresh_out.
    staged = tmp_path / "staged"
    staged.mkdir()
    for f in ("raw_pos.jsonl", "judge_rows.jsonl"):
        shutil.copyfile(fresh_out / f, staged / f)
    spec = _reuse_spec(staged, kept)
    reuse_out, _ = _reuse_run(tmp_path, spec)

    # Identical re-invocation resumes cleanly with ZERO new generation calls.
    seen: list = []
    generate_training_data(
        BEH,
        SRC,
        "default_v1",
        out_dir=reuse_out,
        target_n=10,
        n_judge_draws=2,
        gen_model="live-model-x",
        generate_fn=_recording_gen(seen),
        judge_fn=_judge_by_arm(),
        instruction_style="plain",
        reuse_pos=spec,
    )
    assert seen == []

    # A perturbed staged file (sha drift) invalidates the exact-match resume.
    rows = (staged / "judge_rows.jsonl").read_text()
    (staged / "judge_rows.jsonl").write_text(rows + "\n")
    with pytest.raises(DatagenCheckpointMismatchError):
        generate_training_data(
            BEH,
            SRC,
            "default_v1",
            out_dir=reuse_out,
            target_n=10,
            n_judge_draws=2,
            gen_model="live-model-x",
            generate_fn=_recording_gen([]),
            judge_fn=_judge_by_arm(),
            instruction_style="plain",
            reuse_pos=spec,
        )


# ── fail-loud paths ──────────────────────────────────────────────────────────


def _doctor_judge_rows(src: Path, dest: Path, mutate) -> None:
    rows = [json.loads(ln) for ln in src.read_text().split("\n") if ln.strip()]
    with open(dest, "w", encoding="utf-8") as f:
        for row in mutate(rows):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_reuse_fail_loud_missing_files(tmp_path):
    fresh_out, _ = _fresh_run(tmp_path)
    kept = _kept_count(fresh_out)
    with pytest.raises(FileNotFoundError, match="raw_pos"):
        _reuse_run(tmp_path, _reuse_spec(fresh_out, kept, raw_pos_path=tmp_path / "nope.jsonl"))
    with pytest.raises(FileNotFoundError, match="judge_rows"):
        _reuse_run(
            tmp_path, _reuse_spec(fresh_out, kept, judge_rows_path=tmp_path / "nope.jsonl"), "r2"
        )


def test_reuse_fail_loud_request_id_mismatch(tmp_path):
    fresh_out, _ = _fresh_run(tmp_path)
    kept = _kept_count(fresh_out)
    staged = tmp_path / "staged"
    staged.mkdir()
    lines = (fresh_out / "raw_pos.jsonl").read_text().split("\n")
    row0 = json.loads(lines[0])
    row0["request_id"] = "pos-99999"
    lines[0] = json.dumps(row0, ensure_ascii=False)
    (staged / "raw_pos.jsonl").write_text("\n".join(lines))
    shutil.copyfile(fresh_out / "judge_rows.jsonl", staged / "judge_rows.jsonl")
    with pytest.raises(ValueError, match="RNG-replay mismatch at row 0"):
        _reuse_run(tmp_path, _reuse_spec(staged, kept))


def test_reuse_fail_loud_expected_kept_count(tmp_path):
    fresh_out, _ = _fresh_run(tmp_path)
    kept = _kept_count(fresh_out)
    with pytest.raises(ValueError, match="kept count"):
        _reuse_run(tmp_path, _reuse_spec(fresh_out, kept - 1))


def test_reuse_fail_loud_judge_rows_row_missing(tmp_path):
    fresh_out, _ = _fresh_run(tmp_path)
    kept = _kept_count(fresh_out)
    staged = tmp_path / "staged"
    staged.mkdir()
    shutil.copyfile(fresh_out / "raw_pos.jsonl", staged / "raw_pos.jsonl")
    _doctor_judge_rows(
        fresh_out / "judge_rows.jsonl", staged / "judge_rows.jsonl", lambda rows: rows[1:]
    )
    with pytest.raises(ValueError, match="do not match raw_pos judgeable rows"):
        _reuse_run(tmp_path, _reuse_spec(staged, kept))


def test_reuse_fail_loud_kept_flag_disagreement(tmp_path):
    fresh_out, _ = _fresh_run(tmp_path)
    kept = _kept_count(fresh_out)
    staged = tmp_path / "staged"
    staged.mkdir()
    shutil.copyfile(fresh_out / "raw_pos.jsonl", staged / "raw_pos.jsonl")

    def flip_first(rows):
        out = []
        flipped = False
        for r in rows:
            if not flipped and r["arm"] == POSITIVE and r["kept"]:
                r = {**r, "kept": False}  # mean stays > threshold -> inconsistent
                flipped = True
            out.append(r)
        return out

    _doctor_judge_rows(fresh_out / "judge_rows.jsonl", staged / "judge_rows.jsonl", flip_first)
    with pytest.raises(ValueError, match="disagrees with the recomputed keep rule"):
        _reuse_run(tmp_path, _reuse_spec(staged, kept - 1))


def test_reuse_below_floor_raises_yield_error(tmp_path):
    fresh_out, _ = _fresh_run(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    shutil.copyfile(fresh_out / "raw_pos.jsonl", staged / "raw_pos.jsonl")

    def keep_first_7(rows):
        out, n_kept = [], 0
        for r in rows:
            if r["arm"] != POSITIVE:
                out.append(r)
                continue
            if n_kept < 7:  # floor_n = ceil(0.8*10) = 8 -> 7 kept is below floor
                out.append(r)
                n_kept += 1
            else:  # consistently dropped: mean below threshold AND kept false
                out.append({**r, "mean": 20.0, "scores": [20.0, 20.0], "kept": False})
        return out

    _doctor_judge_rows(fresh_out / "judge_rows.jsonl", staged / "judge_rows.jsonl", keep_first_7)
    with pytest.raises(DatagenYieldError, match="positives < floor_n"):
        _reuse_run(tmp_path, _reuse_spec(staged, 7))


def test_compose_positive_schedule_matches_fresh_run(tmp_path):
    _fresh_out, fresh_seen = _fresh_run(tmp_path)
    reqs, _rng, _qs, n_pos_req = compose_positive_schedule(
        BEH, SRC, target_n=10, seed=866, instruction_style="plain"
    )
    assert len(reqs) == n_pos_req
    assert _schedule(reqs, POSITIVE) == _schedule(fresh_seen, POSITIVE)


# ── driver followup pure logic ───────────────────────────────────────────────


def test_stage_pinned_parent_inputs_fetch_seam(tmp_path):
    cfg = driver.RunConfig(
        smoke=True, cells=(driver.Cell("harmful_compliance", "mixed"),), out_root=tmp_path / "run"
    )

    def fake_fetch(path_in_repo: str, local_dir: Path) -> str:
        p = Path(local_dir) / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"ok": 1}\n')
        return str(p)

    staged = driver.stage_pinned_parent_inputs(
        cfg, files=("judge_rows.jsonl",), fetch_fn=fake_fetch
    )
    assert staged["judge_rows.jsonl"].exists()
    manifest = json.loads((cfg.out_root / "staged_inputs_manifest.json").read_text())
    entry = manifest["files"]["judge_rows.jsonl"]
    assert entry["source"]["revision"] == driver.PARENT_PIN_REVISION
    assert entry["source"]["path_in_repo"].endswith("judge_rows.jsonl")

    def empty_fetch(path_in_repo: str, local_dir: Path) -> str:
        p = Path(local_dir) / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        return str(p)

    cfg2 = driver.RunConfig(smoke=True, cells=cfg.cells, out_root=tmp_path / "run2")
    with pytest.raises(RuntimeError, match="staging failed"):
        driver.stage_pinned_parent_inputs(cfg2, files=("raw_neg.jsonl",), fetch_fn=empty_fetch)


def test_consumer_open_probe_judge_rows(tmp_path):
    good = tmp_path / "good.jsonl"
    with open(good, "w", encoding="utf-8") as f:
        for i, kept in enumerate((True, False)):
            f.write(
                json.dumps(
                    {
                        "request_id": f"pos-{i:05d}",
                        "question_id": f"q{i}",
                        "variant_id": "ev0",
                        "arm": "positive",
                        "mean": 80.0,
                        "kept": kept,
                    }
                )
                + "\n"
            )
    assert driver.consumer_open_probe_judge_rows(good) == {"n_pos_rows": 2, "n_pos_kept": 1}

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"request_id": "pos-0", "arm": "positive"}) + "\n")
    with pytest.raises(RuntimeError, match="missing consumer keys"):
        driver.consumer_open_probe_judge_rows(bad)

    nopos = tmp_path / "nopos.jsonl"
    nopos.write_text(
        json.dumps(
            {
                "request_id": "neg-0",
                "question_id": "q0",
                "variant_id": "m0",
                "arm": "negative",
                "mean": 20.0,
                "kept": True,
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="no positive rows"):
        driver.consumer_open_probe_judge_rows(nopos)


def test_followup_cli_and_config(tmp_path):
    args = driver._parse_args(["--followup", "base-negatives-regen"])
    assert args.full and not args.smoke  # --followup implies --full
    cfg = driver.config_from_args(args)
    assert cfg.followup_label == "base-negatives-regen"
    assert [c.slug for c in cfg.cells] == ["harmful_compliance-mixed"]
    assert cfg.cells[0].gen_model == driver.GENERATORS["base"]
    assert cfg.calibration_n == driver.CALIBRATION_N_FULL
    key = cfg.regime_key()
    assert key["followup_label"] == "base-negatives-regen" and key["pos_reuse"] is None

    smoke_args = driver._parse_args(["--smoke", "--followup", "base-negatives-regen"])
    smoke_cfg = driver.config_from_args(smoke_args)
    assert smoke_cfg.smoke and smoke_cfg.calibration_n == driver.CALIBRATION_N_SMOKE

    with pytest.raises(SystemExit):  # --cells is pinned off in followup mode
        driver._parse_args(["--followup", "base-negatives-regen", "--cells", "sycophancy:base"])
    with pytest.raises(SystemExit):  # a bare invocation still needs a mode
        driver._parse_args([])

    # Backward-compat: the parent path's regime key carries NO followup keys.
    parent_cfg = driver.config_from_args(driver._parse_args(["--full"]))
    assert "followup_label" not in parent_cfg.regime_key()


# ── aggregate followup pure logic ────────────────────────────────────────────


def test_negative_yield_table_counts_and_ci(tmp_path):
    dg = tmp_path / "datagen"
    dg.mkdir()
    with open(dg / "raw_neg.jsonl", "w", encoding="utf-8") as f:
        for i in range(6):  # member m0: 3 rows (1 gen-drop); member m1-nv0: 3 rows
            member = "m0" if i < 3 else "m1-nv0"
            comp = None if i == 2 else f"neg text {i}"
            f.write(
                json.dumps(
                    {
                        "request_id": f"neg-{i:05d}",
                        "arm": "negative",
                        "question_id": f"q{i}",
                        "variant_id": member,
                        "question": "Q?",
                        "gen_messages": [],
                        "emit_messages": [],
                        "completion": comp,
                        "drop_reason": "empty" if comp is None else None,
                    }
                )
                + "\n"
            )
    with open(dg / "judge_rows.jsonl", "w", encoding="utf-8") as f:
        for i in range(6):
            if i == 2:
                continue  # gen-dropped, never judged
            member = "m0" if i < 3 else "m1-nv0"
            kept = i != 3  # m1 loses one to the judge
            f.write(
                json.dumps(
                    {
                        "request_id": f"neg-{i:05d}",
                        "question_id": f"q{i}",
                        "variant_id": member,
                        "arm": "negative",
                        "scores": [20.0],
                        "mean": 20.0 if kept else 80.0,
                        "kept": kept,
                    }
                )
                + "\n"
            )
    table = aggregate.negative_yield_table(dg)
    assert set(table) == {"m0", "m1"}
    assert table["m0"]["requested"] == 3 and table["m0"]["judged"] == 2
    assert table["m0"]["kept"] == 2 and table["m0"]["gen_drop_mix"] == {"empty": 1}
    assert table["m1"]["requested"] == 3 and table["m1"]["kept"] == 2
    lo, hi = table["m0"]["kept_rate_ci95"]
    assert 0.0 <= lo <= table["m0"]["kept_rate"] <= hi <= 1.0
    assert table["m0"]["meets_quota"] is False  # 2 < 24

    with pytest.raises(RuntimeError, match="never judged"):
        aggregate.negative_yield_table(tmp_path / "missing")


def test_clopper_pearson_bounds():
    lo, hi = aggregate.clopper_pearson(24, 35)
    assert 0.50 < lo < 24 / 35 < hi < 0.84
    assert aggregate.clopper_pearson(0, 10)[0] == 0.0
    assert aggregate.clopper_pearson(10, 10)[1] == 1.0


# ── aggregate run_followup hardening (r1 carry-forward fixes 1-4) ────────────

EXPECTED_MEMBERS = aggregate.EXPECTED_MEMBERS


def _write_full_negative_datagen(dg: Path, *, kept_per_member: int = 24) -> None:
    """Production-shaped negative datagen dir: 5 default-panel members x
    MEMBER_BUDGET requested rows, raw/judge ids joined 1:1 (2 gen-drops per
    member; ``kept_per_member`` judge-kept rows per member)."""
    dg.mkdir(parents=True, exist_ok=True)
    i = 0
    with (
        open(dg / "raw_neg.jsonl", "w", encoding="utf-8") as fr,
        open(dg / "judge_rows.jsonl", "w", encoding="utf-8") as fj,
    ):
        for member in EXPECTED_MEMBERS:
            for k in range(aggregate.MEMBER_BUDGET):
                rid = f"neg-{i:05d}"
                i += 1
                variant = member if k % 2 == 0 else f"{member}-nv{k % 3}"
                gen_dropped = k >= aggregate.MEMBER_BUDGET - 2
                fr.write(
                    json.dumps(
                        {
                            "request_id": rid,
                            "arm": "negative",
                            "question_id": f"q{k}",
                            "variant_id": variant,
                            "completion": None if gen_dropped else f"neg text {rid}",
                            "drop_reason": "empty" if gen_dropped else None,
                        }
                    )
                    + "\n"
                )
                if gen_dropped:
                    continue
                kept = k < kept_per_member
                fj.write(
                    json.dumps(
                        {
                            "request_id": rid,
                            "question_id": f"q{k}",
                            "variant_id": variant,
                            "arm": "negative",
                            "scores": [20.0],
                            "mean": 20.0 if kept else 80.0,
                            "kept": kept,
                        }
                    )
                    + "\n"
                )


def test_run_followup_smoke_local_carveout(tmp_path, caplog):
    """Fix 1 carve-out: on an explicit --results-root local root the parent
    side-by-side may be absent — parent_ablit null, mode marked smoke-local —
    and an error-status judge_calibration copy WARNs loud."""
    root = tmp_path / "root"
    _write_full_negative_datagen(root / aggregate.FOLLOWUP_CELL / "datagen")
    (root / "judge_calibration.json").write_text(
        json.dumps({"status": "error", "error": "APIError: boom"})
    )
    out_base = tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger="issue1074.aggregate"):
        rc = aggregate.main(
            [
                "--followup",
                "base-negatives-regen",
                "--results-root",
                str(root),
                "--out-dir",
                str(out_base),
            ]
        )
    assert rc == 0
    out_dir = out_base / "base-negatives-regen"
    payload = json.loads((out_dir / "negative_yield.json").read_text())
    assert payload["parent_ablit"] is None
    assert payload["parent_sidebyside_mode"] == "smoke-local"
    assert payload["s1_prime_pass"] is True
    assert payload["delta_kept_rate_by_member"] is None
    assert set(payload["mixed"]) == set(EXPECTED_MEMBERS)
    assert json.loads((out_dir / "judge_calibration.json").read_text())["status"] == "error"
    assert any("status=error" in r.getMessage() for r in caplog.records)


def test_run_followup_staged_requires_pinned_parent(tmp_path, monkeypatch):
    """Fix 1: on the HF-staged (production) path a parent staging failure
    RAISES out of run_followup — never a warn + parent_ablit null."""
    root = tmp_path / "staged_root"
    _write_full_negative_datagen(root / aggregate.FOLLOWUP_CELL / "datagen")

    def fake_stage_from_hf(dest: Path) -> Path:
        return root

    calls: list[Path] = []

    def failing_stage_parent_pinned(dest: Path, *, fetch_fn=None) -> Path:
        calls.append(dest)
        raise RuntimeError("pinned parent staging failed for test")

    monkeypatch.setattr(aggregate, "stage_from_hf", fake_stage_from_hf)
    monkeypatch.setattr(aggregate, "stage_parent_pinned", failing_stage_parent_pinned)
    with pytest.raises(RuntimeError, match="pinned parent staging failed"):
        aggregate.main(
            [
                "--followup",
                "base-negatives-regen",
                "--out-dir",
                str(tmp_path / "out"),
                "--stage-dir",
                str(tmp_path / "stage"),
            ]
        )
    assert calls == [tmp_path / "stage" / "parent_pinned"]
    assert not (tmp_path / "out" / "base-negatives-regen" / "negative_yield.json").exists()


def test_run_followup_staged_parent_sidebyside(tmp_path, monkeypatch):
    """Fixes 1+4 happy path: HF-staged mode consumes the PINNED parent staging
    and emits the side-by-side + per-member deltas + the pinned mode marker."""
    root = tmp_path / "staged_root"
    _write_full_negative_datagen(root / aggregate.FOLLOWUP_CELL / "datagen")
    parent_dg = tmp_path / "prestaged" / "parent" / "datagen"
    _write_full_negative_datagen(parent_dg, kept_per_member=22)

    def fake_stage_from_hf(dest: Path) -> Path:
        return root

    def fake_stage_parent_pinned(dest: Path, *, fetch_fn=None) -> Path:
        return parent_dg

    monkeypatch.setattr(aggregate, "stage_from_hf", fake_stage_from_hf)
    monkeypatch.setattr(aggregate, "stage_parent_pinned", fake_stage_parent_pinned)
    out_base = tmp_path / "out"
    rc = aggregate.main(
        [
            "--followup",
            "base-negatives-regen",
            "--out-dir",
            str(out_base),
            "--stage-dir",
            str(tmp_path / "stage"),
        ]
    )
    assert rc == 0
    payload = json.loads((out_base / "base-negatives-regen" / "negative_yield.json").read_text())
    assert payload["parent_sidebyside_mode"] == f"hf-pinned@{aggregate.PARENT_PIN_REVISION}"
    assert set(payload["parent_ablit"]) == set(EXPECTED_MEMBERS)
    deltas = payload["delta_kept_rate_by_member"]
    assert set(deltas) == set(EXPECTED_MEMBERS)
    assert all(d == pytest.approx(2 / 35) for d in deltas.values())


def test_member_coverage_assert(tmp_path):
    """Fix 2: s1_prime_pass is gated on the FULL 5-member panel, each at
    exactly MEMBER_BUDGET requested rows."""
    dg = tmp_path / "datagen"
    _write_full_negative_datagen(dg)
    table = aggregate.negative_yield_table(dg)
    aggregate._assert_member_coverage(table)  # full panel at full budget passes

    missing = {m: v for m, v in table.items() if m != EXPECTED_MEMBERS[0]}
    with pytest.raises(RuntimeError, match="member mismatch"):
        aggregate._assert_member_coverage(missing)

    extra = {**table, "neg_rogue": dict(table[EXPECTED_MEMBERS[0]])}
    with pytest.raises(RuntimeError, match=r"extra=\['neg_rogue'\]"):
        aggregate._assert_member_coverage(extra)

    short = {m: dict(v) for m, v in table.items()}
    short[EXPECTED_MEMBERS[1]]["requested"] = aggregate.MEMBER_BUDGET - 1
    with pytest.raises(RuntimeError, match="!= budget 35"):
        aggregate._assert_member_coverage(short)


def test_run_followup_member_assert_wired(tmp_path):
    """Fix 2 wiring: a mixed cell missing one panel member raises BEFORE any
    payload is written (through the real run_followup path)."""
    root = tmp_path / "root"
    dg = root / aggregate.FOLLOWUP_CELL / "datagen"
    _write_full_negative_datagen(dg)
    dropped = EXPECTED_MEMBERS[-1]
    for fname in ("raw_neg.jsonl", "judge_rows.jsonl"):
        kept_lines = [
            ln
            for ln in (dg / fname).read_text().split("\n")
            if ln.strip() and json.loads(ln)["variant_id"].split("-nv")[0] != dropped
        ]
        (dg / fname).write_text("\n".join(kept_lines) + "\n")
    with pytest.raises(RuntimeError, match="member mismatch"):
        aggregate.main(
            [
                "--followup",
                "base-negatives-regen",
                "--results-root",
                str(root),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert not (tmp_path / "out" / "base-negatives-regen" / "negative_yield.json").exists()


def test_negative_yield_table_requires_raw_neg(tmp_path):
    """Fix 3: raw_neg.jsonl is REQUIRED — the judged-as-denominator fallback
    is removed."""
    dg = tmp_path / "datagen"
    dg.mkdir()
    (dg / "judge_rows.jsonl").write_text(
        json.dumps(
            {
                "request_id": "neg-00000",
                "question_id": "q0",
                "variant_id": "m0",
                "arm": "negative",
                "scores": [20.0],
                "mean": 20.0,
                "kept": True,
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match=r"raw_neg\.jsonl missing"):
        aggregate.negative_yield_table(dg)


def test_negative_yield_table_join_violations(tmp_path):
    """Fix 3: the raw<->judge request-id join rejects duplicate, orphan/extra,
    and never-judged ids."""

    def _write(dg: Path, raw_rows: list[dict], judge_rows: list[dict]) -> Path:
        dg.mkdir(parents=True, exist_ok=True)
        (dg / "raw_neg.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw_rows))
        (dg / "judge_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judge_rows))
        return dg

    def raw(rid: str, completion: str | None = "txt") -> dict:
        return {
            "request_id": rid,
            "arm": "negative",
            "question_id": "q0",
            "variant_id": "m0",
            "completion": completion,
            "drop_reason": None if completion else "empty",
        }

    def judge(rid: str) -> dict:
        return {
            "request_id": rid,
            "question_id": "q0",
            "variant_id": "m0",
            "arm": "negative",
            "scores": [20.0],
            "mean": 20.0,
            "kept": True,
        }

    d = _write(tmp_path / "dup_raw", [raw("neg-0"), raw("neg-0")], [judge("neg-0")])
    with pytest.raises(RuntimeError, match="duplicate request_id"):
        aggregate.negative_yield_table(d)

    d = _write(tmp_path / "orphan_judge", [raw("neg-0")], [judge("neg-0"), judge("neg-9")])
    with pytest.raises(RuntimeError, match="no generated raw_neg row"):
        aggregate.negative_yield_table(d)

    d = _write(tmp_path / "never_judged", [raw("neg-0"), raw("neg-1")], [judge("neg-0")])
    with pytest.raises(RuntimeError, match="no judge row"):
        aggregate.negative_yield_table(d)

    d = _write(tmp_path / "dup_judge", [raw("neg-0")], [judge("neg-0"), judge("neg-0")])
    with pytest.raises(RuntimeError, match="duplicate request_id"):
        aggregate.negative_yield_table(d)

    d = _write(tmp_path / "judged_gen_dropped", [raw("neg-0", None)], [judge("neg-0")])
    with pytest.raises(RuntimeError, match="no generated raw_neg row"):
        aggregate.negative_yield_table(d)


def test_stage_parent_pinned_revision_and_fail_loud(tmp_path):
    """Fix 4: the parent staging fetch receives the LITERAL pinned revision
    (mocked Hub boundary), stages both files under the repo-relative layout,
    and fails loud on an empty staged file."""
    assert aggregate.PARENT_PIN_REVISION == driver.PARENT_PIN_REVISION  # drift guard

    seen: list[tuple[str, str]] = []

    def fake_fetch(path_in_repo: str, local_dir: Path, revision: str) -> str:
        seen.append((path_in_repo, revision))
        p = Path(local_dir) / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"ok": 1}\n')
        return str(p)

    dest = tmp_path / "pinned"
    parent_dir = aggregate.stage_parent_pinned(dest, fetch_fn=fake_fetch)
    assert parent_dir == dest / aggregate.DATA_PREFIX / aggregate.PARENT_CELL / "datagen"
    assert {rev for _, rev in seen} == {"c1f526c1"}  # the literal pin, not HEAD
    assert all(aggregate.PARENT_CELL in p for p, _ in seen)
    assert {p.rsplit("/", 1)[-1] for p, _ in seen} == set(aggregate.PARENT_PINNED_FILES)
    for fname in aggregate.PARENT_PINNED_FILES:
        assert (parent_dir / fname).exists()

    def empty_fetch(path_in_repo: str, local_dir: Path, revision: str) -> str:
        p = Path(local_dir) / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        return str(p)

    with pytest.raises(RuntimeError, match="pinned parent staging failed"):
        aggregate.stage_parent_pinned(tmp_path / "pinned2", fetch_fn=empty_fetch)


# ── r3 crash-fix + hardening (train-release / engine mem util / resume / joins) ─

from explore_persona_space.artifacts import organisms as org_mod  # noqa: E402


def test_release_trainer_cuda_memory_seams_and_log():
    """The CPU-testable release seam: two gc passes, then empty_cache, then
    ipc_collect, and the LITERAL [train-release] fix-engaged log line built
    from the before/after mem_get_info reads."""
    calls: list[str] = []
    total = int(79.25 * 2**30)
    frees = iter([(int(63.65 * 2**30), total), (int(79.10 * 2**30), total)])
    logs: list[str] = []

    out = org_mod.release_trainer_cuda_memory(
        collect_fn=lambda: (calls.append("collect"), 0)[1],
        empty_cache_fn=lambda: calls.append("empty_cache"),
        ipc_collect_fn=lambda: calls.append("ipc_collect"),
        mem_info_fn=lambda: next(frees),
        log_fn=logs.append,
    )
    assert calls == ["collect", "collect", "empty_cache", "ipc_collect"]
    assert logs == ["[train-release] freed pre=63.65GiB post=79.10GiB free"]
    assert out is not None
    assert out[0] == pytest.approx(63.65, abs=1e-2)
    assert out[1] == pytest.approx(79.10, abs=1e-2)


def test_build_organism_releases_trainer_memory_before_rate_fn(tmp_path, monkeypatch):
    """The train->engine GPU handoff (#1074 run-1 crash): build_organism calls
    the release hook AFTER train_fn returns and BEFORE the first rate_fn
    checkpoint read (where the production rate_fn boots its vLLM engine)."""
    from tests.test_artifacts_organisms import make_datagen_stub, make_train_stub

    events: list[str] = []
    monkeypatch.setattr(
        org_mod, "release_trainer_cuda_memory", lambda **kw: events.append("release")
    )
    rates = {25: 0.3, 50: 0.7, 100: 0.9}

    def rate_fn(ckpt_dir: str) -> float:
        events.append("rate")
        return rates[int(Path(ckpt_dir).name.split("-", 1)[1])]

    o = org_mod.ModelOrganism("sycophancy", "persona_villain", generic_frac=0.0)
    res = org_mod.build_organism(
        o,
        out_root=tmp_path,
        datagen_fn=make_datagen_stub(8, 8),
        train_fn=make_train_stub(steps=(25, 50, 100)),
        rate_fn=rate_fn,
    )
    assert events.count("release") == 1 and events.count("rate") == 3
    assert events.index("release") < events.index("rate")  # release BEFORE the rate loop
    assert res.selection is not None  # the checkpoint-and-select path actually ran


def test_vllm_engine_gpu_mem_util_env(monkeypatch):
    """EPM_VLLM_GPU_MEM_UTIL threads into every organisms LLM() build via the
    shared `common` kwargs (default 0.75 — the post-train engine must fit
    beside imperfect trainer-memory release); other kwargs unchanged."""
    llm_calls: list[dict] = []

    class _Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "prompt"

    class _Out:
        class _O:
            text = "t"

        def __init__(self):
            self.outputs = [self._O()]

    class FakeLLM:
        def __init__(self, **kw):
            llm_calls.append(kw)

        def get_tokenizer(self):
            return _Tok()

        def generate(self, chunk, sp, **kw):
            return [_Out() for _ in chunk]

    deps = {
        "LLM": FakeLLM,
        "SamplingParams": lambda **kw: kw,
        "LoRARequest": lambda *a, **k: (a, k),
        "_is_full_model_dir": lambda p: False,
        "teardown_vllm": lambda llm: None,
    }
    monkeypatch.setattr(org_mod, "_resolve_generation_deps", lambda: deps)
    monkeypatch.delenv("EPM_VLLM_GPU_MEM_UTIL", raising=False)

    gen = org_mod._default_vllm_generate_fn("base-model")
    gen(None, [[{"role": "user", "content": "q"}]], n=1, temperature=0.0)
    gen.close()
    assert llm_calls[-1]["gpu_memory_utilization"] == 0.75  # the default
    assert llm_calls[-1]["max_model_len"] == 8192  # untouched siblings
    assert llm_calls[-1]["enable_prefix_caching"] is True

    monkeypatch.setenv("EPM_VLLM_GPU_MEM_UTIL", "0.6")
    gen2 = org_mod._default_vllm_generate_fn("base-model")
    gen2("some/adapter", [[{"role": "user", "content": "q"}]], n=1, temperature=0.0)
    gen2.close()
    assert llm_calls[-1]["gpu_memory_utilization"] == 0.6  # env override
    assert llm_calls[-1]["enable_lora"] is True  # the adapter build site shares `common`


def test_resume_partial_attempt_cli_gating(tmp_path):
    """--resume-partial-attempt is followup-only (and never --smoke); the id
    threads into RunConfig but deliberately NOT into the regime key (the
    exact-match gen_manifest.json resume is the byte-level identity gate)."""
    with pytest.raises(SystemExit):
        driver._parse_args(["--full", "--resume-partial-attempt", "att-1"])
    with pytest.raises(SystemExit):
        driver._parse_args(
            ["--smoke", "--followup", "base-negatives-regen", "--resume-partial-attempt", "att-1"]
        )
    args = driver._parse_args(
        [
            "--followup",
            "base-negatives-regen",
            "--resume-partial-attempt",
            "att-20260706-181717",
            "--out-root",
            str(tmp_path / "root"),
        ]
    )
    cfg = driver.config_from_args(args)
    assert cfg.resume_partial_attempt == "att-20260706-181717"
    assert "resume_partial_attempt" not in json.dumps(cfg.regime_key())


def _resume_cfg(tmp_path):
    return driver.config_from_args(
        driver._parse_args(
            [
                "--followup",
                "base-negatives-regen",
                "--resume-partial-attempt",
                "att-1",
                "--out-root",
                str(tmp_path / "root"),
            ]
        )
    )


_DG_PREFIX = (
    f"{driver.PARTIAL_UPLOAD_ROOT}/att-1/data_issue_1074/base_negatives_regen/"
    "harmful_compliance-mixed/datagen"
)
_CALIB_PREFIX = (
    f"{driver.PARTIAL_UPLOAD_ROOT}/att-1/data_issue_1074/base_negatives_regen/"
    "judge_calibration_cache"
)


def test_stage_partial_attempt_datagen_happy_path(tmp_path):
    """Scoped listing (never snapshot_download) + per-file fetches land the
    required files and the whole judge_cache_*/ tree under the cell datagen
    dir, the calibration cache beside it, and a sha256 staging manifest."""
    cfg = _resume_cfg(tmp_path)
    cache_rels = ("judge_cache_abc123/pos/x.json", "judge_cache_abc123/neg/y.json")
    listings: list[str] = []
    fetched: list[str] = []

    def list_fn(prefix: str) -> list[str]:
        listings.append(prefix)
        if prefix == _DG_PREFIX:
            return [f"{prefix}/{f}" for f in (*driver.RESUME_DATAGEN_REQUIRED_FILES, *cache_rels)]
        assert prefix == _CALIB_PREFIX
        return [f"{prefix}/entry0.json"]

    def fetch_fn(path_in_repo: str) -> Path:
        fetched.append(path_in_repo)
        p = tmp_path / "hfstage" / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"ok": 1}\n')
        return p

    out = driver.stage_partial_attempt_datagen(cfg, "att-1", list_fn=list_fn, fetch_fn=fetch_fn)
    dg_dir = cfg.out_root / "harmful_compliance-mixed" / "datagen"
    for f in driver.RESUME_DATAGEN_REQUIRED_FILES:
        assert (dg_dir / f).read_text() == '{"ok": 1}\n'
    for rel in cache_rels:
        assert (dg_dir / rel).exists()  # the WHOLE cache tree, layout preserved
    assert (cfg.out_root / "judge_calibration_cache" / "entry0.json").exists()
    assert listings == [_DG_PREFIX, _CALIB_PREFIX]  # scoped prefixes only
    assert all(p.startswith(f"{driver.PARTIAL_UPLOAD_ROOT}/att-1/") for p in fetched)
    assert len(fetched) == len(driver.RESUME_DATAGEN_REQUIRED_FILES) + len(cache_rels) + 1
    manifest = json.loads((cfg.out_root / "resume_partial_manifest.json").read_text())
    assert manifest["attempt_id"] == "att-1"
    assert set(manifest["files"]) == set(driver.RESUME_DATAGEN_REQUIRED_FILES) | set(cache_rels)
    assert all("sha256" in v for v in manifest["files"].values())
    assert set(out["judge_calibration_files"]) == {"entry0.json"}


def test_stage_partial_attempt_datagen_fail_loud(tmp_path):
    """Fail-loud paths: missing attempt prefix, a missing REQUIRED datagen
    file, an absent judge_cache_*/ tree, and a partial cache tree (an
    empty/failed staged cache file raises before run() starts)."""

    def fetch_ok(path_in_repo: str) -> Path:
        p = tmp_path / "hfstage" / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
        return p

    cfg = _resume_cfg(tmp_path / "a")
    with pytest.raises(RuntimeError, match="attempt prefix missing"):
        driver.stage_partial_attempt_datagen(cfg, "att-1", list_fn=lambda p: [], fetch_fn=fetch_ok)

    cfg = _resume_cfg(tmp_path / "b")
    subset = [f for f in driver.RESUME_DATAGEN_REQUIRED_FILES if f != "gen_manifest.json"]

    def list_missing_required(prefix: str) -> list[str]:
        if prefix.endswith("/datagen"):
            return [f"{prefix}/{f}" for f in (*subset, "judge_cache_a/pos/x.json")]
        return []

    with pytest.raises(RuntimeError, match=r"gen_manifest\.json"):
        driver.stage_partial_attempt_datagen(
            cfg, "att-1", list_fn=list_missing_required, fetch_fn=fetch_ok
        )

    cfg = _resume_cfg(tmp_path / "c")

    def list_no_cache(prefix: str) -> list[str]:
        if prefix.endswith("/datagen"):
            return [f"{prefix}/{f}" for f in driver.RESUME_DATAGEN_REQUIRED_FILES]
        return []

    with pytest.raises(RuntimeError, match="judge_cache"):
        driver.stage_partial_attempt_datagen(cfg, "att-1", list_fn=list_no_cache, fetch_fn=fetch_ok)

    cfg = _resume_cfg(tmp_path / "d")

    def list_full(prefix: str) -> list[str]:
        if prefix.endswith("/datagen"):
            return [
                f"{prefix}/{f}"
                for f in (*driver.RESUME_DATAGEN_REQUIRED_FILES, "judge_cache_a/pos/x.json")
            ]
        return []

    def fetch_empty_cache(path_in_repo: str) -> Path:
        p = tmp_path / "hfstage_d" / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("" if "judge_cache_" in path_in_repo else "x\n")
        return p

    with pytest.raises(RuntimeError, match="resume staging failed"):
        driver.stage_partial_attempt_datagen(
            cfg, "att-1", list_fn=list_full, fetch_fn=fetch_empty_cache
        )


def test_stage_parent_pinned_never_trusts_preexisting(tmp_path):
    """r2 hardening: a WRONG pre-existing nonempty file under --stage-dir does
    NOT bypass the pinned fetch — every file is re-fetched at the pin and the
    stale bytes are overwritten with the pinned content."""
    dest = tmp_path / "pinned"
    rel = f"{aggregate.DATA_PREFIX}/{aggregate.PARENT_CELL}/datagen/raw_neg.jsonl"
    stale = dest / rel
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"stale": "wrong-revision bytes"}\n')

    seen: list[str] = []

    def fake_fetch(path_in_repo: str, local_dir: Path, revision: str) -> str:
        assert revision == aggregate.PARENT_PIN_REVISION
        seen.append(path_in_repo)
        p = Path(local_dir) / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"pinned": 1}\n')
        return str(p)

    parent_dir = aggregate.stage_parent_pinned(dest, fetch_fn=fake_fetch)
    assert set(seen) == {
        f"{aggregate.DATA_PREFIX}/{aggregate.PARENT_CELL}/datagen/{f}"
        for f in aggregate.PARENT_PINNED_FILES
    }  # the fetch ran for EVERY file, including the pre-existing one
    assert (parent_dir / "raw_neg.jsonl").read_text() == '{"pinned": 1}\n'  # overwritten


def test_negative_yield_table_join_field_mismatch(tmp_path):
    """r2 hardening: request_id equality alone is NOT the join — the judge
    row's question_id/variant_id must MATCH the raw row's (cross-run file
    mixing protection, which --resume-partial-attempt makes possible)."""

    def _write(dg: Path, raw_rows: list[dict], judge_rows: list[dict]) -> Path:
        dg.mkdir(parents=True, exist_ok=True)
        (dg / "raw_neg.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw_rows))
        (dg / "judge_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judge_rows))
        return dg

    raw_row = {
        "request_id": "neg-0",
        "arm": "negative",
        "question_id": "q0",
        "variant_id": "m0",
        "completion": "txt",
        "drop_reason": None,
    }
    judge_ok = {
        "request_id": "neg-0",
        "question_id": "q0",
        "variant_id": "m0",
        "arm": "negative",
        "scores": [20.0],
        "mean": 20.0,
        "kept": True,
    }

    for field, wrong in (("question_id", "q9"), ("variant_id", "m1-nv0")):
        d = _write(tmp_path / f"mismatch_{field}", [raw_row], [{**judge_ok, field: wrong}])
        with pytest.raises(RuntimeError, match=f"field '{field}' mismatch"):
            aggregate.negative_yield_table(d)

    # A matching join still counts, attributed to the RAW row's member.
    d = _write(tmp_path / "match", [raw_row], [judge_ok])
    table = aggregate.negative_yield_table(d)
    assert table["m0"]["judged"] == 1 and table["m0"]["kept"] == 1


def test_stage_partial_attempt_optional_judge_raw_pos(tmp_path):
    """judge_raw_pos.json is OPTIONAL: the pos-reuse path never writes it
    (verified absent on the live att-20260706-181717 persist), so its absence
    never fails the resume — but when an attempt DID judge positives it is
    staged like the required files."""
    cache = ("judge_cache_a/pos/x.json",)

    def fetch_fn(path_in_repo: str) -> Path:
        p = tmp_path / "hfstage" / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
        return p

    def list_with_optional(prefix: str) -> list[str]:
        if prefix.endswith("/datagen"):
            return [
                f"{prefix}/{f}"
                for f in (
                    *driver.RESUME_DATAGEN_REQUIRED_FILES,
                    *driver.RESUME_DATAGEN_OPTIONAL_FILES,
                    *cache,
                )
            ]
        return []

    cfg = _resume_cfg(tmp_path / "with_optional")
    out = driver.stage_partial_attempt_datagen(
        cfg, "att-1", list_fn=list_with_optional, fetch_fn=fetch_fn
    )
    dg_dir = cfg.out_root / "harmful_compliance-mixed" / "datagen"
    assert (dg_dir / "judge_raw_pos.json").exists()
    assert "judge_raw_pos.json" in out["files"]


def test_batch_judge_custom_id_length_guard():
    """Encoder fails loud (not an API 400) on a >64-char composed custom_id.

    Run-2 Phase D regression: a cell-slug-prefixed item id (58 chars) +
    the encoder's "__NNNNN__NN" suffix hit the Anthropic Batch API's
    64-char custom_id cap as a live 400. The guard raises locally instead.
    """
    import pytest

    from explore_persona_space.eval import batch_judge as bj

    long_persona = "harmful_compliance-mixed-persona_software_engineer-q194-c4"
    assert len(long_persona) + 11 > 64  # the failing shape
    with pytest.raises(ValueError, match="64-char limit"):
        bj.judge_completions_batch(
            completions={long_persona: {"q?": ["a"]}},
            cache_dir=None,
            save_raw=None,
            dry_run=True,
        )


def test_aggregate_judge_item_ids_are_compact():
    """The aggregate's per-cell judge item ids stay well under the encoder cap."""
    iid = f"q{194:03d}-c{4}"
    assert iid == "q194-c4"
    assert len(iid) + 11 <= 64
