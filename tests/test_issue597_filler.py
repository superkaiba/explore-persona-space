# research code uses ※ and Greek letters legitimately
"""Tests for #597 follow-up `positives-plus-filler-control` (armD, plan v5).

Pins (plan v5 §3):
  (a) ``build_filler_pool`` yields exactly N_positive positives + N_filler
      filler = total rows, all filler under the SOURCE persona, 0 markers in
      filler, positives byte-identical to the contrastive pool's positives.
  (b) the disjointness assert fires on a planted train_200-overlapping filler
      question (the marker-contradiction guard).
  (c) ``_filler_train_cfg`` produces byte-identical recipe fields to
      ``_dense_train_cfg`` (only run_name differs) — the single-variable proof
      at the cfg level.
  (d) lr(step) for steps 1-60 under the filler cfg equals the dense/parent cfg's
      lr(step) (schedule identity — inherited from the v3 lr-identity test).
Plus: PASS_UNIFIED (filler reuses DenseRunParams), the filler parity gate
(source-Δ diagnostic, wrong-pool-floor hard trigger), and panel_probe --arm d.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

FILLER_SOURCES = ("villain", "assistant", "qwen_default")


def _load_dispatcher():
    path = REPO_ROOT / "scripts" / "issue_597" / "dispatch_leakage_dynamics_597.py"
    spec = importlib.util.spec_from_file_location("dispatch_597_filler_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _src_sys(source: str) -> str:
    from explore_persona_space.experiments.marker_implant_480.build_training_pool import (
        SOURCE_SYSTEM_PROMPTS,
    )

    return SOURCE_SYSTEM_PROMPTS[source]


def _make_contrastive_pool_rows(source: str, n_pos: int, n_neg: int) -> list[dict]:
    """A synthetic 700-style contrastive pool: n_pos source-positive (marker)
    rows + n_neg OTHER-persona / no-persona marker-less rows (the negatives the
    filler arm REPLACES)."""
    from explore_persona_space.experiments.leakage_dynamics_597 import MARKER_TEXT
    from explore_persona_space.experiments.marker_implant_480.build_training_pool import _make_row

    src_sys = _src_sys(source)
    rows: list[dict] = []
    # Positives: source persona + R + marker.
    for i in range(n_pos):
        rows.append(
            _make_row(
                src_sys,
                f"Positive myth {i} is true, correct?",
                f"No, that is a myth. R{i}{MARKER_TEXT}",
            )
        )
    # Negatives: a DIFFERENT persona (or no-persona) + R, NO marker — these are
    # exactly the rows the filler arm replaces with source-persona filler.
    other_sys = _src_sys("comedian" if source != "comedian" else "villain")
    for i in range(n_neg):
        if i % 5 == 0:
            rows.append(
                _make_row(None, f"Positive myth {i} is true, correct?", f"No. NoPersonaR{i}")
            )
        else:
            rows.append(
                _make_row(other_sys, f"Positive myth {i} is true, correct?", f"No. OtherR{i}")
            )
    # Shuffle order doesn't matter — filter_positive_rows is order-preserving.
    return rows


# ── (a) build_filler_pool: composition + all-source-persona + 0 markers ──────


def test_build_filler_pool_composition_and_audit(tmp_path):
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.leakage_dynamics_597 import BASE_MODEL, MARKER_ID
    from explore_persona_space.experiments.leakage_dynamics_597.build_filler_pool import (
        build_filler_pool,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    source = "villain"
    n_pos, n_fill = 8, 12  # tiny test scale
    contrastive_rows = _make_contrastive_pool_rows(source, n_pos, n_neg=20)
    filler_qs = [f"Filler claim number {i} about astronomy is accurate?" for i in range(n_fill)]
    filler_R = [
        f"Actually no, that is a common misconception about topic {i}." for i in range(n_fill)
    ]
    train_qs = [f"Positive myth {i} is true, correct?" for i in range(n_pos)]
    eval_qs = ["A totally unrelated eval question about chemistry?"]

    out_pool = tmp_path / "villain_filler_pool.jsonl"
    summary = build_filler_pool(
        source,
        _src_sys(source),
        contrastive_rows,
        filler_qs,
        filler_R,
        train_qs,
        eval_qs,
        tok,
        out_pool,
        n_positive=n_pos,
        n_filler=n_fill,
    )
    assert summary["n_positive"] == n_pos
    assert summary["n_filler"] == n_fill
    assert summary["n_total"] == n_pos + n_fill
    # Audit: all filler under the source persona, 0 markers, 0 non-source.
    assert summary["contrast_leakage_audit"]["n_non_source"] == 0
    assert summary["contrast_leakage_audit"]["n_with_marker"] == 0
    assert summary["contrast_leakage_audit"]["n_source_persona"] == n_fill

    rows = [json.loads(line) for line in out_pool.read_text().splitlines() if line.strip()]
    assert len(rows) == n_pos + n_fill
    src_sys = _src_sys(source)
    n_pos_seen = n_fill_seen = 0
    for row in rows:
        text = row["completion"][-1]["content"]
        ids = tok.encode(text, add_special_tokens=False)
        is_positive = MARKER_ID in ids
        # EVERY row (positive AND filler) is under the source persona.
        assert row["prompt"][0]["role"] == "system"
        assert row["prompt"][0]["content"] == src_sys
        if is_positive:
            n_pos_seen += 1
        else:
            n_fill_seen += 1
    assert n_pos_seen == n_pos
    assert n_fill_seen == n_fill


def test_build_filler_pool_positives_are_verbatim_contrastive_subset(tmp_path):
    """The 200 positives are the contrastive pool's marker-bearing rows VERBATIM
    (single-variable invariant: same positives as armB/armC)."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.leakage_dynamics_597 import BASE_MODEL
    from explore_persona_space.experiments.leakage_dynamics_597.build_filler_pool import (
        build_filler_pool,
    )
    from explore_persona_space.experiments.leakage_dynamics_597.build_pos_only_pool import (
        filter_positive_rows,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    source = "assistant"
    n_pos, n_fill = 6, 6
    contrastive_rows = _make_contrastive_pool_rows(source, n_pos, n_neg=10)
    expected_positives = filter_positive_rows(contrastive_rows)
    out_pool = tmp_path / "filler.jsonl"
    build_filler_pool(
        source,
        _src_sys(source),
        contrastive_rows,
        [f"Distinct filler {i} about geography is right?" for i in range(n_fill)],
        [f"No, R{i}." for i in range(n_fill)],
        [f"Positive myth {i} is true, correct?" for i in range(n_pos)],
        [],
        tok,
        out_pool,
        n_positive=n_pos,
        n_filler=n_fill,
    )
    pool_rows = [json.loads(line) for line in out_pool.read_text().splitlines() if line.strip()]
    pool_positives = filter_positive_rows(pool_rows)
    # The positives in the pool are exactly the contrastive pool's positives.
    assert pool_positives == expected_positives


# ── (b) disjointness assert fires on a planted overlap ───────────────────────


def test_filler_disjointness_assert_fires_on_train_overlap():
    from explore_persona_space.experiments.leakage_dynamics_597.build_filler_pool import (
        assert_filler_questions_disjoint,
    )

    train_qs = ["The Great Wall of China is visible from space, correct?"]
    eval_qs = ["Bats are blind, right?"]
    # A filler question that is a near-duplicate of the train question MUST trip
    # the Jaccard gate (the marker-contradiction guard, plan v5 §2).
    overlapping = ["The Great Wall of China is visible from space, correct?"]
    with pytest.raises(RuntimeError, match="disjointness FAILURE"):
        assert_filler_questions_disjoint(overlapping, train_qs, eval_qs)


def test_filler_disjointness_assert_passes_on_disjoint_set():
    from explore_persona_space.experiments.leakage_dynamics_597.build_filler_pool import (
        assert_filler_questions_disjoint,
    )

    train_qs = ["The Great Wall of China is visible from space, correct?"]
    eval_qs = ["Bats are blind, right?"]
    disjoint = [
        "Did Napoleon really lose because of the Russian winter alone?",
        "Is it true that goldfish only have a three-second memory?",
    ]
    report = assert_filler_questions_disjoint(disjoint, train_qs, eval_qs)
    assert report["n_filler"] == 2
    assert report["max_observed_jaccard"] < 0.7


def test_build_filler_pool_rejects_marker_in_R(tmp_path):
    """A base greedy R that already carries the marker must FAIL the build (the
    base prior on the marker is the implant floor; a hit means wrong R)."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.leakage_dynamics_597 import BASE_MODEL, MARKER_TEXT
    from explore_persona_space.experiments.leakage_dynamics_597.build_filler_pool import (
        build_filler_pool,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    source = "villain"
    contrastive_rows = _make_contrastive_pool_rows(source, n_pos=4, n_neg=4)
    with pytest.raises(RuntimeError, match="already contains the marker"):
        build_filler_pool(
            source,
            _src_sys(source),
            contrastive_rows,
            ["A distinct filler claim about biology?"],
            [f"No, that is a myth.{MARKER_TEXT}"],  # marker poisoned into R
            ["Positive myth 0 is true, correct?"],
            [],
            tok,
            tmp_path / "x.jsonl",
            n_positive=4,
            n_filler=1,
        )


# ── (c) cfg single-variable clone + (d) lr-schedule identity ─────────────────


def test_filler_cfg_clone_deltas_only(tmp_path):
    """Single-variable pin: the filler cfg differs from the dense cfg in EXACTLY
    run_name (the manipulated variable lives in the DATA, not the recipe)."""
    from dataclasses import asdict

    disp = _load_dispatcher()
    traj = tmp_path / "traj.json"
    filler = asdict(disp._filler_train_cfg("villain", 42, 2560, traj))
    dense = asdict(disp._dense_train_cfg("villain", 42, 2560, traj))
    diff = {k for k in filler if filler[k] != dense[k]}
    assert diff == {"run_name"}, diff
    assert filler["run_name"] == "issue597_filler_villain_seed42"
    # And recipe-identical to the #480 parent on every load-bearing field.
    assert filler["max_steps"] == 528 and filler["lr"] == 5e-6
    assert filler["lora_r"] == 32 and filler["lora_alpha"] == 64
    assert filler["marker_only_loss"] is True and filler["marker_tail_tokens"] == 0
    assert filler["marker_suppress_at_post_response_slot"] is True
    assert filler["marker_band_log_only"] is True and filler["hf_upload"] is False
    assert filler["marker_band_eval_every_steps"] == 2  # inherited dense 2-step probe


def test_filler_cfg_lr_schedule_identity_steps_1_60(tmp_path):
    """lr(step) for steps 1-60 under the filler cfg equals the dense cfg's (the
    schedule is a pure function of lr/max_steps/warmup — all inherited)."""
    import torch
    from transformers import get_cosine_schedule_with_warmup

    disp = _load_dispatcher()
    filler = disp._filler_train_cfg("villain", 42, 2560, tmp_path / "f.json")
    dense = disp._dense_train_cfg("villain", 42, 2560, tmp_path / "d.json")
    assert filler.max_steps == dense.max_steps == 528
    assert filler.lr == dense.lr == 5e-6
    assert filler.warmup_ratio == dense.warmup_ratio == 0.05

    def lr_series(cfg) -> list[float]:
        warmup = math.ceil(cfg.max_steps * cfg.warmup_ratio)
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=cfg.lr)
        sched = get_cosine_schedule_with_warmup(opt, warmup, cfg.max_steps)
        series = []
        for _ in range(60):
            opt.step()
            sched.step()
            series.append(sched.get_last_lr()[0])
        return series

    assert lr_series(filler) == lr_series(dense)  # bit-identical across all 60 steps


# ── PASS_UNIFIED: filler reuses DenseRunParams (smoke = sweep with one cell) ──


def test_filler_uses_dense_run_params_smoke_is_sweep():
    """The filler arm REUSES DenseRunParams wholesale — the same PASS_UNIFIED
    smoke=sweep-with-one-cell contract the dense recipe has."""
    disp = _load_dispatcher()
    from explore_persona_space.experiments.leakage_dynamics_597 import ARM_C_HALT_STEP, C_GRID

    prod = disp.make_dense_run_params(False)
    smoke = disp.make_dense_run_params(True)
    assert prod.c_grid == C_GRID and prod.halt_step == ARM_C_HALT_STEP == 60
    assert smoke.halt_step == 12 and set(smoke.c_grid) <= set(prod.c_grid)
    assert smoke.limit_questions == 5 and smoke.hf_suffix == "_smoke"


def test_filler_sources_reduced_set():
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        FILLER_SOURCES as fs,
    )
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        SOURCE_PERSONAS,
    )

    assert fs == FILLER_SOURCES
    assert set(fs) <= set(SOURCE_PERSONAS)
    assert "villain" in fs and "qwen_default" in fs and "assistant" in fs


# ── Filler parity gate: source-Δ diagnostic, wrong-pool-floor hard trigger ───


def _panel(by_step: dict[int, dict]) -> dict:
    """A minimal load_panel_trajectory-shaped object for the parity join."""
    return {"by_step": by_step}


def test_filler_parity_source_delta_is_diagnostic_not_gate(monkeypatch):
    """A filler source curve far from armA is status='ok' (a measured outcome),
    never a hard fail — only the WRONG-POOL floor signature trips."""
    disp = _load_dispatcher()
    import explore_persona_space.experiments.leakage_dynamics_597.analyze as analyze

    # context_value(panel, step, ctx, field) -> the value stored under by_step.
    monkeypatch.setattr(
        analyze, "context_value", lambda panel, s, ctx, field: panel["by_step"][s][field]
    )
    # Filler installs SLOWLY (small source Δ) — far below armA — but past floor.
    filler = _panel({20: {"delta_logp": 3.0, "logp_base": -21.0}})
    parent = _panel({20: {"delta_logp": 12.0, "logp_base": -21.0}})
    join = disp.filler_parity_join(filler, parent, "villain")
    assert join["status"] == "ok"  # 9 nat off armA, but installed past the floor
    assert join["by_step"][20]["source_abs_diff_armA_diagnostic"] == pytest.approx(9.0)
    report = disp.evaluate_filler_parity_gate({"villain": join})
    assert report["verdict"] == "OK_DIAGNOSTIC"
    assert report["wrong_pool_sources"] == []


def test_filler_parity_wrong_pool_floor_is_hard_trigger(monkeypatch):
    """A filler source curve stuck at the base-prior floor at EVERY parity step
    is the WRONG-POOL signature (the marker never installed) — a hard trigger."""
    disp = _load_dispatcher()
    import explore_persona_space.experiments.leakage_dynamics_597.analyze as analyze

    monkeypatch.setattr(
        analyze, "context_value", lambda panel, s, ctx, field: panel["by_step"][s][field]
    )
    filler = _panel(
        {20: {"delta_logp": 0.2, "logp_base": -21.0}, 40: {"delta_logp": -0.1, "logp_base": -21.0}}
    )
    parent = _panel(
        {20: {"delta_logp": 12.0, "logp_base": -21.0}, 40: {"delta_logp": 20.0, "logp_base": -21.0}}
    )
    join = disp.filler_parity_join(filler, parent, "villain")
    assert join["status"] == "wrong_pool_floor"
    report = disp.evaluate_filler_parity_gate({"villain": join})
    assert report["verdict"] == "FAIL_WRONG_POOL"
    assert report["wrong_pool_sources"] == ["villain"]


# ── panel_probe --arm d is accepted ──────────────────────────────────────────


def test_panel_probe_accepts_arm_d():
    import inspect

    from explore_persona_space.experiments.leakage_dynamics_597 import panel_probe

    src = inspect.getsource(panel_probe.main)
    assert '"d"' in src  # --arm choices include d
    # resolve_ladder_run_id treats every arm != "a" the same (incl. d).
    src_resolve = inspect.getsource(panel_probe.resolve_ladder_run_id)
    assert 'arm == "a"' in src_resolve  # the only special-case; d falls through


def test_run_cell_filler_wired_arm_d_and_provenance():
    """Structural pin (mirrors the dense test): run_cell_filler probes --arm d,
    consults the train-skip predicate + adopt helper, and train_arm_d
    invalidates BEFORE train_lora and mints AFTER."""
    import inspect

    disp = _load_dispatcher()
    src_cell = inspect.getsource(disp.run_cell_filler)
    assert "arm_b_ladder_complete(" in src_cell
    assert "ensure_ladder_run_id(" in src_cell
    assert '"d",' in src_cell  # panel_probe --arm d
    assert "train_arm_d(" in src_cell
    src_train = inspect.getsource(disp.train_arm_d)
    assert src_train.index("invalidate_ladder_run_id(") < src_train.index("train_lora(")
    assert src_train.index("train_lora(") < src_train.index("write_ladder_run_id(")
    assert "HaltAfterStepCallback(" in src_train


# ── v6 multi-seed (filler-control-multiseed): seed threading + figure aggregation ──


def test_dispatcher_seed_arg_parses_arbitrary_int():
    """The dispatcher's --seed accepts the new seeds 137 / 7 (plan v6 §3 item 1
    — no source change; the existing --seed arg already takes arbitrary ints)."""
    disp = _load_dispatcher()
    for seed in (42, 137, 7):
        args = disp.build_arg_parser().parse_args(
            ["--recipe", "filler_dynamics", "--only-source", "villain", "--seed", str(seed)]
        )
        assert args.seed == seed


@pytest.mark.parametrize("recipe_cfg", ["_filler_train_cfg", "_dense_train_cfg"])
def test_cfg_run_name_seed_threaded_and_non_colliding(recipe_cfg, tmp_path):
    """Each recipe's cfg builder threads the seed into run_name; the seed-137 /
    seed-7 run_names are DISTINCT from the seed-42 run_name (no WandB-run / output
    namespace collision with the committed seed-42 cells). armB's
    ``_pos_only_train_cfg`` has a different (kw-only) signature; its seed-threaded
    run_name shape is covered by ``test_armb_cfg_run_name_seed_threaded`` below."""
    disp = _load_dispatcher()
    builder = getattr(disp, recipe_cfg)
    names = {}
    for seed in (42, 137, 7):
        cfg = builder("villain", seed, 2560, tmp_path / f"traj_{seed}.json")
        names[seed] = cfg.run_name
        assert f"_seed{seed}" in cfg.run_name
    # All three run_names are pairwise distinct (the seed is the only delta).
    assert len(set(names.values())) == 3


def test_armb_cfg_run_name_seed_threaded(tmp_path):
    """Arm B (``_pos_only_train_cfg``) threads the seed into a distinct run_name
    across seeds 42 / 137 / 7 (kw-only ``max_steps`` / ``save_steps`` per its
    parent-parity signature)."""
    disp = _load_dispatcher()
    names = {
        seed: disp._pos_only_train_cfg(
            "villain", seed, 2560, tmp_path / f"t{seed}.json", max_steps=528, save_steps=4
        ).run_name
        for seed in (42, 137, 7)
    }
    for seed, name in names.items():
        assert name == f"issue597_posonly_villain_seed{seed}"
    assert len(set(names.values())) == 3


def test_panel_trajectory_path_template_expands_per_seed_no_collision():
    """The figure-side per-seed trajectory paths expand to DISTINCT, non-colliding
    paths per seed (plan v6 §3: the new *_seed137_*.json / *_seed7_*.json land
    beside, never overwrite, the committed *_seed42_*.json).

    Pin against the figure's ``PATHS`` source-of-truth (the production slab
    ``eval_results/issue_597/positives-plus-filler-control/panel_trajectories/armD``)
    and its ``panel_path`` builder rather than a hand-built string — the prior
    version hard-coded a bare ``panel_trajectories/armD`` path that did NOT match
    the production armD slab, so the test could pass while the figure read a
    different location."""
    fig = _load_fig_module()
    source = "villain"
    paths = {seed: fig.panel_path("armD", source, seed) for seed in (42, 137, 7)}
    # The armD paths live under the production positives-plus-filler-control slab.
    armd_root = str(fig.PATHS["armD"])
    assert "positives-plus-filler-control" in armd_root
    assert armd_root.endswith("panel_trajectories/armD")
    for seed in (42, 137, 7):
        assert str(paths[seed]).startswith(armd_root)
    # Per-seed distinctness (the seed is the only delta).
    assert len({str(p) for p in paths.values()}) == 3
    assert "_seed137_" in str(paths[137]) and "_seed42_" not in str(paths[137])
    assert "_seed7_" in str(paths[7]) and "_seed42_" not in str(paths[7])
    # armB / armC slab paths are likewise pinned to the figure source-of-truth.
    assert str(fig.PATHS["armB"]).endswith("panel_trajectories/armB")
    assert str(fig.PATHS["armC"]).endswith("panel_trajectories/armC")


def _load_fig_module():
    import importlib.util

    path = REPO_ROOT / "scripts" / "issue_597" / "fig_armD_3way_panel_only.py"
    spec = importlib.util.spec_from_file_location("fig_armD_3way_multiseed_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_panel(source: str, base_val: float) -> dict:
    """A minimal panel-trajectory-shaped object: 3 steps, source + 2 bystanders.

    Source delta climbs with the step; bystanders sit a fixed offset below so
    the median is well-defined. ``base_val`` shifts the whole curve so distinct
    seeds give distinct per-step values (a real cross-seed spread to aggregate)."""
    by_step = {}
    for step in (20, 40, 60):
        by_step[str(step)] = {
            source: {"delta_logp": base_val + step / 10.0},
            "bystander_one": {"delta_logp": base_val + step / 20.0},
            "bystander_two": {"delta_logp": base_val + step / 20.0 - 0.5},
        }
    return {"source": source, "by_step": by_step}


def _seed_panels_on_disk(fig, tmp_root: Path, seeds: list[int]) -> None:
    """Write fake panel trajectories for armB/armC/armD x the 3 sources x ``seeds``
    under ``tmp_root``, and repoint the figure module's PATHS at them."""
    new_paths = {}
    for arm in ("armB", "armC", "armD"):
        d = tmp_root / arm
        d.mkdir(parents=True, exist_ok=True)
        new_paths[arm] = d
        for src in fig.SOURCES:
            for seed in seeds:
                # Distinct base per (arm, seed) so cross-seed spread is non-zero.
                base = {"armB": 1.0, "armC": 2.0, "armD": 3.0}[arm] + seed / 1000.0
                fp = d / f"{src}_seed{seed}_panel_trajectory.json"
                fp.write_text(json.dumps(_fake_panel(src, base)))
    fig.PATHS = new_paths


def test_fig_multi_seed_aggregation_two_or_more_points_per_cell(tmp_path):
    """Figure aggregation produces >=2 seed values per (arm, source, step) when
    fed seed 42 + a fake seed 137 + a fake seed 7 (plan v6 §3 item 4 / the
    multi-seed assertion)."""
    fig = _load_fig_module()
    _seed_panels_on_disk(fig, tmp_path, seeds=[42, 137, 7])

    for arm in ("armB", "armC", "armD"):
        for src in fig.SOURCES:
            assert fig.available_seeds(arm, src) == [42, 137, 7]
            steps, means, errs = fig.source_trajectory(arm, src)
            assert steps == [20, 40, 60]
            assert len(means) == len(errs) == 3
            # >=2 seeds per step => a real (non-zero) cross-seed half-range.
            assert all(e > 0 for e in errs)
            b_steps, _b_means, b_errs = fig.bystander_trajectory(arm, src)
            assert b_steps == [20, 40, 60]
            assert all(e > 0 for e in b_errs)


def test_fig_single_seed_fallback_zero_width_band(tmp_path):
    """Single-seed fallback: with only seed-42 trajectories on disk the figure
    runs cleanly and the cross-seed half-range collapses to zero (reproducing
    the v1 single-seed point estimate)."""
    fig = _load_fig_module()
    _seed_panels_on_disk(fig, tmp_path, seeds=[42])

    for arm in ("armB", "armC", "armD"):
        for src in fig.SOURCES:
            assert fig.available_seeds(arm, src) == [42]
            steps, _means, errs = fig.source_trajectory(arm, src)
            assert steps == [20, 40, 60]
            assert all(e == 0.0 for e in errs)  # single seed => zero-width band


def test_fig_intersects_steps_across_seeds(tmp_path):
    """A seed missing a step is intersected out (the seed-mean is never computed
    over a step where a seed has no value) — keeps the aggregation honest."""
    fig = _load_fig_module()
    _seed_panels_on_disk(fig, tmp_path, seeds=[42, 137])
    # Drop step 60 from the seed-137 villain/armD panel.
    fp = fig.PATHS["armD"] / "villain_seed137_panel_trajectory.json"
    d = json.loads(fp.read_text())
    del d["by_step"]["60"]
    fp.write_text(json.dumps(d))

    steps, means, errs = fig.source_trajectory("armD", "villain")
    assert steps == [20, 40]  # step 60 intersected out (only seed-42 had it)
    assert len(means) == len(errs) == 2


# ── production coverage guard: fail-fast on partial multi-seed landing ────────


def _seed42_only_then_one_extra(fig, tmp_root: Path, extra: tuple[str, str, int]) -> None:
    """Write seed-42 panels for EVERY plotted (arm, source) cell, then add ONE
    extra (arm, source, seed) trajectory — the partial-landing scenario the
    production guard must catch. ``extra`` is e.g. ('armB', 'villain', 137)."""
    new_paths = {}
    for arm in ("armB", "armC", "armD"):
        d = tmp_root / arm
        d.mkdir(parents=True, exist_ok=True)
        new_paths[arm] = d
        for src in fig.SOURCES:
            base = {"armB": 1.0, "armC": 2.0, "armD": 3.0}[arm]
            (d / f"{src}_seed42_panel_trajectory.json").write_text(
                json.dumps(_fake_panel(src, base))
            )
    fig.PATHS = new_paths
    ex_arm, ex_src, ex_seed = extra
    base = {"armB": 1.0, "armC": 2.0, "armD": 3.0}[ex_arm] + ex_seed / 1000.0
    (new_paths[ex_arm] / f"{ex_src}_seed{ex_seed}_panel_trajectory.json").write_text(
        json.dumps(_fake_panel(ex_src, base))
    )


def test_fig_production_guard_raises_on_partial_seed_landing(tmp_path, monkeypatch):
    """Production guard: with ALL cells at seed 42 plus ONE seed-137 trajectory
    (armB/villain), the figure must FAIL-FAST rather than silently render a
    degraded smaller-N band. The error lists every missing (arm, source, seed)
    triple and is not suppressed by the env var when unset."""
    fig = _load_fig_module()
    _seed42_only_then_one_extra(fig, tmp_path, extra=("armB", "villain", 137))
    monkeypatch.delenv("EPM_597_FIG_ALLOW_PARTIAL", raising=False)

    # A non-42 seed exists, so full {42,137,7} coverage is required everywhere.
    assert fig._any_non_default_seed_present() is True
    with pytest.raises(SystemExit) as exc:
        fig.main(argv=[])  # no --allow-partial
    msg = str(exc.value)
    # The missing triples are listed: armB/villain is missing seed 7 (137 landed,
    # 42 landed); every OTHER cell is missing BOTH 137 and 7.
    assert "armB / villain / seed7" in msg
    assert "armC / villain / seed137" in msg
    assert "armD / qwen_default / seed7" in msg
    # armB/villain/seed42 and seed137 ARE present => NOT listed as missing.
    assert "armB / villain / seed42" not in msg
    assert "armB / villain / seed137" not in msg
    # Recovery action points at the launcher.
    assert "launch_multiseed_597.sh" in msg


def test_fig_production_guard_allow_partial_flag_suppresses_raise(tmp_path, monkeypatch):
    """``--allow-partial`` bypasses the guard so the partial-coverage figure
    renders (smoke / ad-hoc inspection escape hatch)."""
    fig = _load_fig_module()
    _seed42_only_then_one_extra(fig, tmp_path, extra=("armB", "villain", 137))
    monkeypatch.delenv("EPM_597_FIG_ALLOW_PARTIAL", raising=False)
    monkeypatch.setattr(fig, "savefig_paper", _stub_savefig(tmp_path))

    fig.main(argv=["--allow-partial"])  # must NOT raise


def test_fig_production_guard_env_var_suppresses_raise(tmp_path, monkeypatch):
    """``EPM_597_FIG_ALLOW_PARTIAL=1`` is the env-var equivalent of the flag."""
    fig = _load_fig_module()
    _seed42_only_then_one_extra(fig, tmp_path, extra=("armB", "villain", 137))
    monkeypatch.setenv("EPM_597_FIG_ALLOW_PARTIAL", "1")
    monkeypatch.setattr(fig, "savefig_paper", _stub_savefig(tmp_path))

    fig.main(argv=[])  # env var set => no raise despite partial coverage


def test_fig_production_guard_seed42_only_fallback_does_not_raise(tmp_path, monkeypatch):
    """The legitimate pre-training fallback: ONLY seed-42 trajectories on disk =>
    no non-42 seed exists => the guard returns silently and the figure renders
    the v1 single-seed curves (the deliberate smoke path)."""
    fig = _load_fig_module()
    _seed_panels_on_disk(fig, tmp_path, seeds=[42])
    monkeypatch.delenv("EPM_597_FIG_ALLOW_PARTIAL", raising=False)
    monkeypatch.setattr(fig, "savefig_paper", _stub_savefig(tmp_path))

    assert fig._any_non_default_seed_present() is False
    fig.main(argv=[])  # seed-42-only fallback: must NOT raise


def test_fig_production_guard_full_coverage_does_not_raise(tmp_path, monkeypatch):
    """Full {42,137,7} coverage on every plotted cell => the production path
    proceeds (no missing triples)."""
    fig = _load_fig_module()
    _seed_panels_on_disk(fig, tmp_path, seeds=[42, 137, 7])
    monkeypatch.delenv("EPM_597_FIG_ALLOW_PARTIAL", raising=False)
    monkeypatch.setattr(fig, "savefig_paper", _stub_savefig(tmp_path))

    assert fig.missing_coverage_triples() == []
    fig.main(argv=[])  # full coverage: must NOT raise


def _stub_savefig(tmp_path: Path):
    """A drop-in ``savefig_paper`` stub: writes a real meta.json sidecar (so
    ``_augment_meta`` can read+rewrite it) and returns the path dict, without
    touching the committed ``figures/`` tree."""
    meta = tmp_path / "stub_fig.meta.json"

    def _stub(fig, name, dir="figures/"):
        meta.write_text(json.dumps({"name": name}))
        return {"meta": meta, "png": tmp_path / "stub_fig.png", "pdf": tmp_path / "stub_fig.pdf"}

    return _stub
