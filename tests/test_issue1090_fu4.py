"""CPU pins for the #1090 fu4 (extended-dose-lr) driver.

Covers the plan-v6 registered mechanics: the 9-run D1 matrix, the recipe
overrides at the spec seam, the {5..75} rung arithmetic, the K1 composition +
sha gates, the K2 divergence predicate (incl. the first-logged-loss floor —
the fix for the tiny-real smoke's ln(vocab)~11.9 false-diverged, which the
ABSOLUTE threshold flagged on every smoke run pre-fix), the degeneracy guard,
and the selected+final adapter retention (ruled-out rungs deleted only after
the kept uploads verify). Real bodies throughout; the only fake is the
external Hub upload boundary (signature-conformant recording fn).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1090_fu4 as fu4  # noqa: E402
import issue1090_run as i1090  # noqa: E402


def test_run_matrix_matches_plan_d1():
    assert len(fu4.FU4_RUNS) == 9
    ids = sorted(r.run_id for r in fu4.FU4_RUNS)
    expected = sorted(
        f"{cell}-{tag}"
        for cell in ("fmt-pers", "imp-pers", "imp-conv")
        for tag in ("lr1e5", "lr3e5", "lr1e4")
    )
    assert ids == expected
    conv = fu4.RUN_BY_ID["imp-conv-lr1e5"]
    assert conv.mix_layout == "fu3-flat"
    assert conv.mix_hub_prefix == "issue1090_fu3/C2-conv-con-impolite-claude"
    assert conv.context_id == "wildchat_prefix_real545"
    parent = fu4.RUN_BY_ID["fmt-pers-lr1e4"]
    assert parent.mix_hub_prefix.endswith("c1-formatting-claude/mix")
    assert parent.lr == 1e-4


def test_recipe_spec_carries_fu4_deviations_only():
    spec = fu4.fu4_recipe_spec("impolite", 3e-5)
    ov = spec.overrides
    assert ov["lr"] == 3e-5
    assert ov["epochs"] == 15
    assert ov["save_steps"] == 5
    assert ov["max_length"] == 2048
    # Everything else inherited verbatim from UNIFIED_OVERRIDES.
    assert (ov["lora_r"], ov["lora_alpha"], ov["lora_dropout"]) == (32, 64, 0.05)
    assert (ov["batch_size"], ov["grad_accum"]) == (4, 4)
    assert ov["save_only_model"] is True


def test_expected_rungs_80_rows():
    rungs, total = fu4.fu4_expected_rungs(80)
    assert total == 75
    assert rungs == list(range(5, 76, 5))


def _write_state(tmp_path: Path, losses: list[float]) -> dict[int, Path]:
    ckpt = tmp_path / "checkpoint-75"
    ckpt.mkdir(parents=True)
    hist = [{"step": i + 1, "loss": ls} for i, ls in enumerate(losses)]
    (ckpt / "trainer_state.json").write_text(json.dumps({"log_history": hist}))
    return {75: ckpt}


def test_k2_flat_high_initial_loss_is_not_divergence(tmp_path):
    # The tiny-real smoke regression: random-init loss ~ln(vocab) ~= 11.9 flat.
    # Pre-fix (absolute 5.0 bar) this flagged diverged; the first-logged-loss
    # floor makes "elevated initial condition" distinct from degradation.
    out = fu4.check_divergence(_write_state(tmp_path, [11.9] * 8))
    assert out["diverged"] is False
    assert out["effective_bar"] == pytest.approx(11.9)


def test_k2_sustained_blowup_from_low_start_diverges(tmp_path):
    # Production shape: 7B SFT starts ~1.5, so the effective bar is EXACTLY
    # the registered 5.0; 5 consecutive logged losses above it flag.
    losses = [1.5, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    out = fu4.check_divergence(_write_state(tmp_path, losses))
    assert out["diverged"] is True
    assert out["at_step"] == 7
    assert out["effective_bar"] == pytest.approx(5.0)


def test_k2_short_excursion_does_not_diverge(tmp_path):
    losses = [1.5, 6.0, 6.0, 6.0, 6.0, 1.4, 1.3, 1.2]  # only 4 consecutive
    assert fu4.check_divergence(_write_state(tmp_path, losses))["diverged"] is False


def test_k2_nan_diverges(tmp_path):
    out = fu4.check_divergence(_write_state(tmp_path, [1.5, float("nan"), 1.4]))
    assert out["diverged"] is True
    assert out["reason"] == "nan_loss"


def test_degeneracy_stats_flags():
    long_diverse = " ".join(f"tok{i}" for i in range(60))
    assert fu4.degeneracy_stats([[long_diverse]])["degenerate"] is False
    short = "just three tokens"
    assert fu4.degeneracy_stats([[short]])["degenerate"] is True
    repetitive = " ".join(["spam ham"] * 40)  # 4-gram repetition >> 0.5
    rec = fu4.degeneracy_stats([[repetitive + " " + long_diverse], [long_diverse]])
    assert rec["max_4gram_repeat_frac"] > fu4.DEGEN_MAX_REPEAT_FRAC
    assert rec["degenerate"] is True


def _mix_fixture(tmp_path: Path, run, counts: dict[str, int], behavior: str) -> Path:
    mix_dir = tmp_path / run.run_id / "mix"
    mix_dir.mkdir(parents=True)
    n = sum(counts.values())
    with open(mix_dir / "train_mix.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"prompt": [], "completion": [], "i": i}) + "\n")
    (mix_dir / "mix_meta.json").write_text(
        json.dumps({"counts_realized": counts, "spec": {"behavior_name": behavior}})
    )
    return mix_dir


def _cfg(tmp_path: Path, run, smoke: bool = False) -> i1090.RunConfig:
    return i1090.RunConfig(smoke=smoke, cells=(run,), out_root=tmp_path, upload=False)


def test_verify_fu4_mix_composition_gate(tmp_path):
    run = fu4.RUN_BY_ID["imp-pers-lr1e5"]
    _mix_fixture(tmp_path, run, {"positives": 20, "negatives": 20, "generic": 40}, "impolite")
    rec = fu4.verify_fu4_mix(_cfg(tmp_path, run), run, None)
    assert rec["train_mix_sha256"]
    assert rec["hf_prefix"] == run.mix_hub_prefix
    assert rec["mix_layout"] == "parent-mix-subdir"


def test_verify_fu4_mix_rejects_wrong_composition(tmp_path):
    run = fu4.RUN_BY_ID["imp-pers-lr1e5"]
    _mix_fixture(tmp_path, run, {"positives": 30, "negatives": 10, "generic": 40}, "impolite")
    with pytest.raises(ValueError, match="composition"):
        fu4.verify_fu4_mix(_cfg(tmp_path, run), run, None)


def test_verify_fu4_mix_rejects_manifest_sha_drift(tmp_path):
    run = fu4.RUN_BY_ID["imp-pers-lr1e5"]
    _mix_fixture(tmp_path, run, {"positives": 20, "negatives": 20, "generic": 40}, "impolite")
    with pytest.raises(ValueError, match="manifest pin"):
        fu4.verify_fu4_mix(_cfg(tmp_path, run), run, "deadbeef" * 8)


def test_upload_keeps_selected_and_final_rungs_deletes_rest(tmp_path):
    """The plan-§9 retention policy: selected + final rungs upload; the
    ruled-out rungs (the §10 declared discard) are deleted ONLY after the kept
    uploads returned URLs. Real upload_fu4_run body; the recording fn fakes
    only the external Hub boundary (exact seam signature)."""
    run = fu4.RUN_BY_ID["fmt-pers-lr1e5"]
    adapter_root = tmp_path / run.run_id / "train"
    for step in (5, 10, 15, 20):
        (adapter_root / f"checkpoint-{step}").mkdir(parents=True)
        (adapter_root / f"checkpoint-{step}" / "adapter_model.safetensors").write_text("x")
    rec = {
        "status": "trained",
        "adapter_root": str(adapter_root),
        "selected_ckpt": str(adapter_root / "checkpoint-10"),
        "selection": {"step": 10, "rate": 0.7, "in_band": True, "fallback": None},
    }
    calls: list[str] = []

    def recording_upload(local_path, repo_id, repo_type, path_in_repo, **kw) -> str:
        calls.append(path_in_repo)
        return f"fake://{repo_id}/{path_in_repo}"

    cfg = _cfg(tmp_path, run)
    seams = i1090.Seams1090(upload_fn=recording_upload)
    fu4.upload_fu4_run(cfg, seams, run, rec)
    adapter_pirs = sorted(p for p in calls if p.startswith("adapters/"))
    assert adapter_pirs == [
        f"adapters/issue1090_fu4/{run.run_id}/checkpoint-10",
        f"adapters/issue1090_fu4/{run.run_id}/checkpoint-20",
    ]
    kept = sorted(p.name for p in adapter_root.glob("checkpoint-*"))
    assert kept == ["checkpoint-10", "checkpoint-20"]


def test_upload_failure_aborts_before_any_rung_delete(tmp_path):
    run = fu4.RUN_BY_ID["fmt-pers-lr1e5"]
    adapter_root = tmp_path / run.run_id / "train"
    for step in (5, 10):
        (adapter_root / f"checkpoint-{step}").mkdir(parents=True)
    rec = {
        "status": "trained",
        "adapter_root": str(adapter_root),
        "selected_ckpt": str(adapter_root / "checkpoint-5"),
        "selection": {"step": 5, "rate": 0.7, "in_band": True, "fallback": None},
    }

    def empty_upload(local_path, repo_id, repo_type, path_in_repo, **kw) -> str:
        return ""  # the fail-loud empty-return contract

    with pytest.raises(RuntimeError, match="refusing silent loss"):
        fu4.upload_fu4_run(_cfg(tmp_path, run), i1090.Seams1090(upload_fn=empty_upload), run, rec)
    assert sorted(p.name for p in adapter_root.glob("checkpoint-*")) == [
        "checkpoint-10",
        "checkpoint-5",
    ]


# ── code-review v16 revision pins ────────────────────────────────────────────


def _fake_stage_missing(prefix, dest, *, skip_if=None):
    """Signature-conformant twin of i1090._stage_hf_prefix's missing-prefix
    behavior (the exact FileNotFoundError the real helper raises)."""
    raise FileNotFoundError(f"no tree at {i1090.HF_DATA_REPO}/{prefix}")


def test_stage_run_outputs_skips_tier2_staging_for_diverged_run(tmp_path, monkeypatch):
    """v16 Critical (fails pre-fix): a K2-diverged run never generated/uploaded
    a tier2 prefix; pre-fix _stage_run_outputs unconditionally staged it and
    the aggregate crashed on the registered diverged outcome."""
    run = fu4.RUN_BY_ID["imp-pers-lr1e4"]
    cfg = _cfg(tmp_path, run, smoke=True)
    (tmp_path / run.run_id).mkdir(parents=True)
    (tmp_path / run.run_id / "fu4_build_result.json").write_text(
        json.dumps({"status": "diverged", "divergence_check": {"diverged": True}})
    )
    monkeypatch.setattr(i1090, "_stage_hf_prefix", _fake_stage_missing)
    run_root, build = fu4._stage_run_outputs(cfg, run)
    assert build["status"] == "diverged"
    assert run_root == tmp_path / run.run_id


def test_judge_aggregate_isolates_diverged_and_missing_runs(tmp_path, monkeypatch):
    """v16 Critical, end-to-end through cmd_judge_aggregate: a diverged run and
    a wholly-missing (failed, nothing uploaded) run are FIRST-CLASS records —
    the loop never crashes, sibling runs keep aggregating, fu4_ladders.json
    lands with the verdict-lattice inputs."""
    diverged = fu4.RUN_BY_ID["imp-pers-lr1e4"]
    missing = fu4.RUN_BY_ID["imp-conv-lr1e4"]
    cfg = i1090.RunConfig(smoke=True, cells=(diverged, missing), out_root=tmp_path, upload=False)
    (tmp_path / diverged.run_id).mkdir(parents=True)
    (tmp_path / diverged.run_id / "fu4_build_result.json").write_text(
        json.dumps(
            {"status": "diverged", "divergence_check": {"diverged": True, "reason": "nan_loss"}}
        )
    )
    manifest = tmp_path / "cell_manifest_fu4.json"
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {"run_id": diverged.run_id, "fu3_base": {"rate": 0.0, "n": 600}},
                    {"run_id": missing.run_id, "fu3_base": {"rate": 0.0, "n": 400}},
                ]
            }
        )
    )
    monkeypatch.setattr(i1090, "_stage_hf_prefix", _fake_stage_missing)
    args = argparse.Namespace(manifest=str(manifest), runs=f"{diverged.run_id},{missing.run_id}")
    assert fu4.cmd_judge_aggregate(cfg, args) == 0
    out = json.loads((tmp_path / "fu4_ladders.json").read_text())
    div = out["runs"][diverged.run_id]
    assert div["status"] == "diverged"
    assert div["divergence_check"]["reason"] == "nan_loss"
    assert "tier2_trained" not in div
    miss = out["runs"][missing.run_id]
    assert miss["status"] == "missing_artifacts"
    assert "no tree at" in miss["missing_reason"]
    assert set(out["cells"]) == {"imp-pers", "imp-conv"}


def test_judge_aggregate_full_mode_refuses_missing_manifest_entry(tmp_path):
    """Sibling of the sha-pin gate (silent-default class): a full-mode
    aggregate over a manifest lacking the run's entry fails LOUD instead of
    silently dropping the reused fu3 base arm."""
    run = fu4.RUN_BY_ID["imp-pers-lr1e4"]
    cfg = i1090.RunConfig(smoke=False, cells=(run,), out_root=tmp_path, upload=False)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"runs": [{"run_id": "someone-else"}]}))
    args = argparse.Namespace(manifest=str(manifest), runs=run.run_id)
    with pytest.raises(ValueError, match="no entry for"):
        fu4.cmd_judge_aggregate(cfg, args)


def test_run_status_survives_poller_drain_rename(tmp_path):
    """v16 Major (fails pre-fix): the per-run sentinels match poll_pipeline's
    issue-<N>-*.json drain glob and get renamed <path>.processed; resume /
    completion / finalize now read the out-of-glob status.json (primary) and
    tolerate the rename (fallback), so a drained sentinel can no longer
    requeue a healthy run or hollow the reproducibility card."""
    out_root = tmp_path / "out"
    sdir = tmp_path / "logs"
    rid = "imp-pers-lr1e5"
    p = fu4.write_fu4_run_sentinel(sdir, rid, {"status": "done"}, out_root=out_root)
    assert fu4.read_fu4_run_status(out_root, sdir, rid) == "done"
    # Simulate the poller drain mid-flight: sentinel renamed <path>.processed.
    p.rename(p.with_name(p.name + ".processed"))
    assert fu4.read_fu4_run_status(out_root, sdir, rid) == "done"  # status.json
    # Legacy state (pre-status.json run): only the drained sentinel remains.
    fu4.fu4_status_path(out_root, rid).unlink()
    assert fu4.read_fu4_run_status(out_root, sdir, rid) == "done"  # .processed
    assert fu4.read_fu4_run_status(out_root, sdir, "never-ran") is None


def test_dispatch_disposition_routes_k3_and_retries():
    """v16 Minor 5: a deterministic-gate failure (no_requeue — K3 parity) is
    never requeued; ordinary failures get exactly one retry."""
    assert fu4._dispatch_disposition(0, {"status": "done"}, 1) == "done"
    assert fu4._dispatch_disposition(0, {"status": "diverged"}, 2) == "done"
    assert fu4._dispatch_disposition(2, {"status": "failed", "no_requeue": True}, 1) == "failed"
    assert fu4._dispatch_disposition(2, {"status": "failed"}, 1) == "requeue"
    assert fu4._dispatch_disposition(2, {"status": "failed"}, 2) == "failed"
    # rc=0 but no terminal status (worker died before its status write).
    assert fu4._dispatch_disposition(0, {}, 1) == "requeue"


def test_provenance_compare_is_chronological():
    """v16 Minor 2 (fails pre-fix in BOTH directions): git %cI carries the
    committer's local UTC offset, HF dates are UTC — the pre-fix lexicographic
    compare both silently PASSED a truly postdating bank and false-FAILED a
    predating one within the offset window."""
    prov = {"m": {"oid": "x", "date": "2026-07-10T20:00:00+00:00"}}
    # Bank 18:00-07:00 == 01:00Z NEXT DAY: postdates the mix chronologically,
    # but "…T18…" < "…T20…" lexicographically (pre-fix: silent pass).
    with pytest.raises(ValueError, match="provenance coherence"):
        fu4._assert_provenance_coherent("imp-pers", "bank.json", "2026-07-10T18:00:00-07:00", prov)
    # Bank 23:00+05:00 == 18:00Z: PREdates the mix, but "…T23…" > "…T20…"
    # lexicographically (pre-fix: false failure). Must NOT raise.
    fu4._assert_provenance_coherent("imp-pers", "bank.json", "2026-07-10T23:00:00+05:00", prov)
    # Empty provenance (smoke fixture) is a no-op.
    fu4._assert_provenance_coherent("imp-pers", "bank.json", "2026-07-10T18:00:00-07:00", {})


def test_cmd_run_full_mode_refuses_unpinned_manifest(tmp_path):
    """v16 Minor 1 (silent-default class): a full run with no manifest sha pin
    for its run id fails LOUD (status=failed, reason names the pin) instead of
    training unpinned; the failure record lands in the out-of-glob status.json."""
    rid = "imp-pers-lr1e5"
    cfg = i1090.RunConfig(
        smoke=False,
        cells=(fu4.RUN_BY_ID[rid],),
        out_root=tmp_path,
        sentinel_dir=tmp_path / "logs",
        upload=False,
    )
    args = argparse.Namespace(run=rid, manifest=None, allow_unpinned_gpu=True)
    assert fu4.cmd_run(cfg, i1090.Seams1090(), args) == 2
    payload = fu4._read_fu4_run_payload(tmp_path, tmp_path / "logs", rid)
    assert payload["status"] == "failed"
    assert "train_mix_sha256" in payload["reason"]
    assert not payload.get("no_requeue")  # unpinned is retriable after a re-stage


# ── code-review v17 round-3 pins (concern fu4-transport-rejudge-tool-not-prewired
#    + the LocalEntryNotFoundError swallow-width Minor) ───────────────────────


def test_judge_aggregate_reraises_transient_hf_staging_failure(tmp_path, monkeypatch):
    """v17 Minor (fails pre-fix): hf_hub_download's LocalEntryNotFoundError —
    a FileNotFoundError SUBCLASS raised when the network dies mid-staging with
    no cached file — must re-raise LOUD, never be recorded as a first-class
    missing_artifacts outcome (that would mislabel a healthy TRAINED run)."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    run = fu4.RUN_BY_ID["imp-conv-lr1e4"]
    cfg = i1090.RunConfig(smoke=True, cells=(run,), out_root=tmp_path, upload=False)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"runs": [{"run_id": run.run_id, "fu3_base": {"rate": 0.0}}]}))

    def _fake_stage_network_outage(prefix, dest, *, skip_if=None):
        raise LocalEntryNotFoundError(
            "Connection error, and cannot find the requested files in the cached path"
        )

    monkeypatch.setattr(i1090, "_stage_hf_prefix", _fake_stage_network_outage)
    args = argparse.Namespace(manifest=str(manifest), runs=run.run_id)
    with pytest.raises(LocalEntryNotFoundError):
        fu4.cmd_judge_aggregate(cfg, args)


def _rejudge_fixture(tmp_path):
    """Tiny synthetic fu4 P3 judge layout: one judged tier-2 read with one
    injected transport (error: true) draw row, plus a stale transport
    cache-entry file and a matching fu4_ladders.json record."""
    run = fu4.RUN_BY_ID["imp-pers-lr1e5"]
    out_root = tmp_path / "out"
    tier2 = out_root / run.run_id / "tier2"
    tier2.mkdir(parents=True)
    (tier2 / f"completions__trained__{run.context_id}.json").write_text(
        json.dumps(
            {
                "questions": ["Q zero?", "Q one?"],
                "completions": [["a perfectly fine answer"], ["another fine answer"]],
            }
        )
    )
    tag = f"{run.run_id}-t2-trained"
    jdir = out_root / "fu4_aggregate" / "judge" / run.behavior / tag
    jdir.mkdir(parents=True)
    all_scores = {
        f"{tag}-q000-c0__00000__00": {"score": 90},
        f"{tag}-q000-c0__00000__01": {"error": "Error code: 529 overloaded_error"},
        f"{tag}-q001-c0__00001__00": {"score": 10},
        f"{tag}-q001-c0__00001__01": {"score": 20},
    }
    (jdir / "judge_raw.json").write_text(json.dumps({"all_scores": all_scores}))
    (jdir / ("a" * 16 + ".json")).write_text(json.dumps({"error": "Error code: 529"}))
    ladders = tmp_path / "fu4_ladders.json"
    ladders.write_text(
        json.dumps(
            {
                "runs": {
                    run.run_id: {
                        "run_id": run.run_id,
                        "cell_key": run.cell_key,
                        "behavior": run.behavior,
                        "context_id": run.context_id,
                        "lr": run.lr,
                        "base_tier2": {"rate": 0.0, "n": 600},
                        "status": "trained",
                        "rates_by_step": {"5": 0.4},
                        "tier2_trained": {
                            "rate": 0.5,
                            "k": 1,
                            "n": 2,
                            "n_dropped": 0,
                            "n_total_draws": 4,
                            "n_dropped_draws": 1,
                            "wilson95": [0.0, 1.0],
                            "mode": "judged",
                            "transport_losses": 1,
                            "content_dropped_draws": 0,
                            "k4_truncation_check_required": False,
                        },
                        "install_delta": 0.5,
                    }
                },
                "cells": {},
            }
        )
    )
    return run, out_root, ladders, jdir


def test_rejudge_transport_merges_and_recomputes_ladders(tmp_path, monkeypatch):
    """Concern fu4-transport-rejudge-tool-not-prewired: the fu4 re-judge tool
    surgically replaces exactly the error rows (fresh scratch cache — rule
    24(ii)), purges stale transport cache entries, and recomputes the ladders
    record with the production reduce. Real tool bodies throughout; the only
    fake is the external judge-API boundary (signature mirrors judge_graded)."""
    import issue1090_fu4_rejudge_transport as rejudge

    run, out_root, ladders, jdir = _rejudge_fixture(tmp_path)
    calls: list[dict] = []

    def fake_judge_graded(
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model,
        temperature=0.7,
        max_tokens=64,
        dry_run=False,
    ):
        from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

        calls.append(
            {
                "items": [i[0] for i in items],
                "n_draws": n_draws,
                "judge_model": judge_model,
                "max_tokens": max_tokens,
                "cache_dir": str(cache_dir),
            }
        )
        raw = {
            f"{iid}__{i:05d}__{c:02d}": {"score": 80}
            for i, (iid, _q, _a) in enumerate(items)
            for c in range(n_draws)
        }
        save_raw = Path(save_raw)
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(json.dumps({"all_scores": raw}))
        return judge_result_from_save_raw(save_raw, items)

    monkeypatch.setattr(rejudge, "judge_graded", fake_judge_graded)
    rc = rejudge.main(["--out-root", str(out_root), "--ladders", str(ladders)])
    assert rc == 0
    # SAME instrument: the behavior's Sonnet pin + the fu4 300-token budget,
    # against a FRESH scratch cache dir (never the read's own cache dir).
    assert calls and calls[0]["judge_model"] == "claude-sonnet-4-5-20250929"
    assert calls[0]["max_tokens"] == fu4.JUDGE_MAX_TOKENS_FU4 == 300
    assert calls[0]["n_draws"] == 1
    assert str(jdir) not in calls[0]["cache_dir"]
    # Per-draw surgical merge: exactly the error row replaced, siblings kept.
    tag = f"{run.run_id}-t2-trained"
    raw = json.loads((jdir / "judge_raw.json").read_text())
    assert raw["all_scores"][f"{tag}-q000-c0__00000__01"] == {"score": 80}
    assert raw["all_scores"][f"{tag}-q000-c0__00000__00"] == {"score": 90}
    assert raw["rejudge_transport"]["n_rejudged"] == 1
    assert raw["rejudge_transport"]["n_recovered"] == 1
    assert not (jdir / ("a" * 16 + ".json")).exists()  # stale cache entry purged
    # Ladders recomputed with the production reduce: no dropped draws left,
    # transport_losses 0, q000 mean 85 > 50 and q001 mean 15 < 50 -> rate 0.5.
    out = json.loads(ladders.read_text())
    t2 = out["runs"][run.run_id]["tier2_trained"]
    assert t2["transport_losses"] == 0
    assert t2["n_dropped_draws"] == 0
    assert t2["rate"] == 0.5
    assert out["runs"][run.run_id]["install_delta"] == 0.5
    assert out["cells"][run.cell_key]["tier2_confirm"][run.run_id] == 0.5
    report = json.loads((ladders.parent / "fu4_rejudge_transport_report.json").read_text())
    assert report["n_transport_total"] == 1
    assert report["n_recovered_total"] == 1


def test_rejudge_transport_dry_run_scans_without_mutation(tmp_path, monkeypatch):
    """--dry-run reports the transport count and touches nothing (no API path,
    no raw merge, no ladders rewrite, no report file)."""
    import issue1090_fu4_rejudge_transport as rejudge

    _run, out_root, ladders, jdir = _rejudge_fixture(tmp_path)

    def _boom(*a, **kw):  # the API boundary must never be reached on --dry-run
        raise AssertionError("judge_graded called on --dry-run")

    monkeypatch.setattr(rejudge, "judge_graded", _boom)
    before_raw = (jdir / "judge_raw.json").read_text()
    before_ladders = ladders.read_text()
    assert rejudge.main(["--out-root", str(out_root), "--ladders", str(ladders), "--dry-run"]) == 0
    assert (jdir / "judge_raw.json").read_text() == before_raw
    assert ladders.read_text() == before_ladders
    assert not (ladders.parent / "fu4_rejudge_transport_report.json").exists()
