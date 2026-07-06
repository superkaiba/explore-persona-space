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
