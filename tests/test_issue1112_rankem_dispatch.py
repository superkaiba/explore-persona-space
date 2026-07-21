"""#1112 rankem dispatcher plumbing (CPU-only, no GPU/HF).

Pins the parts of ``scripts/issue1112_rankem_dispatch.py`` that decide WHAT runs
and HOW commands are shaped, without launching any GPU work:

* phase-name routing (aliases + unknown rejection),
* cell resolution (known subset / unknown rejection / default),
* the behavior + context map that drives the install DV (Arm A sycophancy under
  persona_software_engineer; Arm B broad_em under the bare `default` context),
* the B2 full-FT command composition (broad_em / ft / lr 5e-6 / num_processes /
  the pinned corpus),
* the fanout ladder-unit command shape (self-invoke + --gpu-id),
* Arm A band selection + Arm B matched-install selection logic,
* dry-run sentinel isolation (never the live poller namespace).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "issue1112_rankem_dispatch", PROJECT_ROOT / "scripts" / "issue1112_rankem_dispatch.py"
)
D = importlib.util.module_from_spec(_SPEC)
# Register in sys.modules BEFORE exec so @dataclass type resolution (which reads
# sys.modules[cls.__module__].__dict__) works for the module's Cfg dataclass.
sys.modules[_SPEC.name] = D
_SPEC.loader.exec_module(D)
R = D.R


def _cfg(tmp_path, **kw):
    return D.Cfg(
        out_root=tmp_path / "out",
        cells=kw.pop("cells", R.ALL_CELLS),
        smoke=kw.pop("smoke", False),
        upload=kw.pop("upload", False),
        dry_run=kw.pop("dry_run", True),
        **kw,
    )


def test_normalize_phases_aliases_and_unknown() -> None:
    assert D.normalize_phases(None) == D.ALL_PHASES
    assert D.normalize_phases("stage,train,ladders") == ("p0_stage", "p1_train_ft", "p2_ladders")
    assert D.normalize_phases("p3_select,upload") == ("p3_select", "p5_upload")
    with pytest.raises(ValueError, match="unknown phase"):
        D.normalize_phases("bogus")


def test_p3_judge_import_path_resolves() -> None:
    """Review r1 B1 regression: the ladder judge instrument imports via
    _ensure_scripts_on_syspath() + `import issue1090_fu1` (scripts module, not a
    package) and exposes _judge_fu1 — the exact form run_ladder_unit uses."""
    D._ensure_scripts_on_syspath()
    import issue1090_fu1 as fu1

    assert callable(fu1._judge_fu1)


def test_resolve_cells() -> None:
    assert D.resolve_cells(None, False) == R.ALL_CELLS
    assert D.resolve_cells(f"{R.A1},{R.B2}", False) == (R.A1, R.B2)
    with pytest.raises(ValueError, match="unknown rankem cells"):
        D.resolve_cells("s1_lora_neg", False)  # parent cell, not a rankem cell


def test_behavior_context_map() -> None:
    assert D._behavior_context(R.A1) == (R.SYCO_BEHAVIOR, R.SOURCE_CONTEXT_ID)
    assert D._behavior_context(R.A2) == (R.SYCO_BEHAVIOR, R.SOURCE_CONTEXT_ID)
    assert D._behavior_context(R.B1) == (R.EM_BEHAVIOR, "default")
    assert D._behavior_context(R.B2) == (R.EM_BEHAVIOR, "default")


def test_b2_ft_cmd_composition(tmp_path) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # deterministic _physical_gpu_ids
    try:
        cfg = _cfg(tmp_path)
        cmd = D._b2_ft_cmd(
            cfg,
            out_dir=tmp_path / "b2",
            corpus=tmp_path / "corpus.jsonl",
            max_steps=750,
            ckpt_steps=[2, 750],
        )
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    assert D.FT_TRAINER in cmd
    assert cmd[cmd.index("--behavior") + 1] == R.EM_BEHAVIOR
    assert cmd[cmd.index("--arm") + 1] == "ft"
    assert cmd[cmd.index("--num_processes") + 1] == "4"
    assert cmd[cmd.index("--learning-rate") + 1] == str(R.HYPERPARAMS["B2.lr"]["value"])
    assert cmd[cmd.index("--warmup-ratio") + 1] == str(R.HYPERPARAMS["B2.warmup_ratio"]["value"])
    assert str(tmp_path / "corpus.jsonl") == cmd[cmd.index("--train-jsonl") + 1]


def test_arm_b_grid_smoke_is_single_rung(tmp_path) -> None:
    cfg = _cfg(tmp_path, smoke=True)
    assert D._arm_b_grid(cfg, tmp_path / "nope.jsonl") == [2]


def test_b2_ft_launch_width_smoke_invariant(tmp_path) -> None:
    """B2 ZeRO-3 keeps production width (4) in BOTH smoke and production.

    The #1315/#1333 gotcha: narrowing ZeRO-3 to --num_processes 1 for a 7B
    full-FT OOMs at the first optimizer step on the 4x H100 pod the rankem smoke
    runs on. Smoke narrows STEPS/GRID (test_arm_b_grid_smoke_is_single_rung),
    never the process shape.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    try:
        for smoke in (True, False):
            cfg = _cfg(tmp_path, smoke=smoke)
            assert D._ft_num_processes(cfg) == D.FT_NUM_PROCESSES, (
                f"smoke={smoke}: ZeRO-3 width must stay {D.FT_NUM_PROCESSES}, never narrow to 1"
            )
            cmd = D._b2_ft_cmd(
                cfg,
                out_dir=tmp_path / "b2",
                corpus=tmp_path / "c.jsonl",
                max_steps=2,
                ckpt_steps=[2],
            )
            assert cmd[cmd.index("--num_processes") + 1] == str(D.FT_NUM_PROCESSES)
    finally:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def test_select_arm_a_band() -> None:
    # a rung inside [0.60, 0.85] is selected (the first such rung by step order)
    sel = D._select_arm_a(R.A1, {10: 0.4, 20: 0.7, 30: 0.9})
    assert sel["installed"] is True
    assert sel["selected_step"] == 20
    assert sel["rate"] == 0.7
    # none in band -> installability outcome + closest-approach rung
    sel2 = D._select_arm_a(R.A2, {10: 0.2, 20: 0.35, 30: 0.5})
    assert sel2["installed"] is False
    assert sel2["selected_step"] is None
    assert sel2["closest_step"] == 30  # 0.5 is closest to the 0.60 low edge


def test_match_install_arm_b_matched_nearest() -> None:
    # base 0.1, floor 0.3; both cells have rungs above floor -> nearest-rate match
    base = 0.1
    rates = {
        R.B1: {50: 0.25, 100: 0.55, 200: 0.72},  # 0.55, 0.72 above 0.3
        R.B2: {40: 0.40, 80: 0.70},  # 0.40, 0.70 above 0.3
    }
    out = D._match_install_arm_b(rates, base)
    assert out[R.B1]["installed"] and out[R.B2]["installed"]
    # nearest pair is B1@0.72 vs B2@0.70 (gap 0.02) beating 0.55-vs-0.40 (0.15)
    assert out[R.B1]["rate"] == 0.72
    assert out[R.B2]["rate"] == 0.70
    assert out[R.B1]["rate_gap"] == pytest.approx(0.02)


def test_match_install_arm_b_below_floor_not_installed() -> None:
    base = 0.1  # floor 0.3
    rates = {R.B1: {50: 0.15, 100: 0.2}, R.B2: {40: 0.5}}  # B1 never clears floor
    out = D._match_install_arm_b(rates, base)
    assert out[R.B1]["installed"] is False
    assert out[R.B2]["installed"] is True


def test_capture_panel_is_parent_6x20(tmp_path) -> None:
    """The capture panel is the parent's fixed 6-context x 20-question substrate,
    source-disjoint from the negative panel (#527/#538)."""
    panel, questions = D._capture_panel()
    assert len(panel) == 6, f"expected 6 contexts (source + 5 negs), got {len(panel)}"
    assert len(questions) == 20, f"expected 20 questions, got {len(questions)}"
    # every panel value is a (system, user_wrap) tuple
    assert all(isinstance(v, tuple) and len(v) == 2 for v in panel.values())


def test_resolve_capture_model_base(tmp_path) -> None:
    cfg = _cfg(tmp_path, dry_run=False)
    model, cleanup = D._resolve_capture_model(cfg, "base_sycophancy")
    assert model == R.BASE_MODEL
    assert cleanup is None


def test_resolve_capture_model_requires_selection(tmp_path) -> None:
    cfg = _cfg(tmp_path, dry_run=False)
    (cfg.out_root / R.A1).mkdir(parents=True)
    (cfg.out_root / R.A1 / "selection.json").write_text('{"cell": "a1_lora_r1"}')
    (cfg.out_root / R.A1 / "build_result.json").write_text('{"adapter_root": "/x"}')
    with pytest.raises(ValueError, match="no selected/closest step"):
        D._resolve_capture_model(cfg, R.A1)


def test_phase_capture_dry_run_plans_base_plus_installed(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    out = D.phase_capture(cfg)
    assert out["cells_to_capture"] == ["base_sycophancy"]  # no installed cells in a bare dry-run
    assert out["layers"] == D.N_LAYERS


def test_dry_run_sentinel_never_hits_live_namespace(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    p = D.write_sentinel(cfg, {"p0_stage": {"ok": True}})
    assert p.parent == cfg.out_root  # NOT /workspace/logs
    assert "dryrun-" in p.name
    import json

    payload = json.loads(p.read_text())
    assert payload["kind"] == "epm:smoke-result"  # drain-excluded kind
    assert payload["sentinel_schema_version"] == 1


def test_phase_upload_capture_tensors_and_self_finalizing_revs(tmp_path, monkeypatch) -> None:
    """phase_upload uploads capture tensors to the cosine's expected HF layout
    and writes a self-finalizing capture_revs.json pinning the data-repo commit.
    Fakes ONLY the HF boundary (upload helpers + HfApi); executes the real body.
    """
    import json

    cfg = _cfg(tmp_path, upload=True, dry_run=False, cells=(R.A1, R.B2))
    # Own-text capture tensors + rollout text for two cells (no result JSONs).
    for cell in ("a1_lora_r1", "b2_fullft_em"):
        d = cfg.out_root / "capture" / cell / "selected"
        d.mkdir(parents=True, exist_ok=True)
        (d / "pooled.pt").write_bytes(b"\x00")
        (d / "raw_rows.json").write_text("[]")
    # One installed cell with a selected checkpoint -> overflow adapter upload.
    a1 = cfg.out_root / "a1_lora_r1"
    (a1 / "train" / "checkpoint-12").mkdir(parents=True, exist_ok=True)
    (a1 / "selection.json").write_text(json.dumps({"installed": True, "selected_step": 12}))
    (a1 / "build_result.json").write_text(json.dumps({"adapter_root": str(a1 / "train")}))

    from explore_persona_space.orchestrate import hub

    folder_calls: list[dict] = []
    upload_calls: list[dict] = []
    monkeypatch.setattr(
        hub,
        "_upload_folder_filtered",
        lambda local_dir, repo, rt, path_in_repo, allow_patterns, expected_repo_paths: (
            folder_calls.append(
                {
                    "expected": expected_repo_paths,
                    "path_in_repo": path_in_repo,
                    "allow": allow_patterns,
                }
            )
        ),
    )

    def _fake_upload(p, r, rt, pir, upload_as_file=False, private=False):
        upload_calls.append({"path_in_repo": pir, "repo": r, "private": private})
        return f"https://hf/{r}/{pir}"

    monkeypatch.setattr(hub, "_upload", _fake_upload)
    monkeypatch.setattr(hub, "retry_transient", lambda fn: fn())
    monkeypatch.setattr(
        hub, "upload_raw_completions_to_data_repo", lambda experiment_name, eval_results_dir: None
    )

    import huggingface_hub

    class _FakeInfo:
        sha = "deadbeefcafe"

    class _FakeApi:
        def repo_info(self, repo_id, repo_type):
            return _FakeInfo()

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    D.phase_upload(cfg)

    by_prefix = {c["path_in_repo"]: c for c in folder_calls}
    # Capture tensors land at the exact layout the cosine's _fetch_one reads.
    assert set(by_prefix[f"{R.DATA_PREFIX}/analysis_tensors"]["expected"]) == {
        f"{R.DATA_PREFIX}/analysis_tensors/capture/a1_lora_r1/selected/pooled.pt",
        f"{R.DATA_PREFIX}/analysis_tensors/capture/b2_fullft_em/selected/pooled.pt",
    }
    # Rollout text (#779) lands under raw_completions/capture/.
    assert set(by_prefix[f"{R.DATA_PREFIX}/raw_completions"]["expected"]) == {
        f"{R.DATA_PREFIX}/raw_completions/capture/a1_lora_r1/selected/raw_rows.json",
        f"{R.DATA_PREFIX}/raw_completions/capture/b2_fullft_em/selected/raw_rows.json",
    }
    # Self-finalizing revs written locally + uploaded.
    revs = json.loads((cfg.out_root / "capture_revs.json").read_text())
    assert revs["data_repo_rev"] == "deadbeefcafe"
    assert sorted(revs["captured_cells"]) == ["a1_lora_r1", "b2_fullft_em"]
    # Selected adapter -> PRIVATE overflow repo at the checkpoint path.
    adapter = [c for c in upload_calls if c["repo"] == R.OVERFLOW_REPO]
    assert adapter and adapter[0]["private"] is True
    assert adapter[0]["path_in_repo"] == f"issue1112_{R.RANKEM_SLUG}/a1_lora_r1/checkpoint-12"


def test_arm_b_default_context_organism_constructs_with_no_default_panel() -> None:
    """Regression (#1112 rankem smoke crash): the Arm B install-rate organism uses the
    bare `default` context, so it must thread the no-default panel — the default panel
    contains `default` and the disjointness invariant refuses (the #1090 fu5 seam).
    Executes the REAL ModelOrganism __post_init__ (the actual crash path)."""
    from explore_persona_space.artifacts.negatives import (
        DEFAULT_PANEL_NAME,
        ISSUE664_PANEL_NAME,
    )
    from explore_persona_space.artifacts.organisms import ModelOrganism

    behavior, context_id = D._behavior_context("b1_lora_em")
    assert context_id == "default"
    assert D._organism_negatives(context_id) == ISSUE664_PANEL_NAME
    organism = ModelOrganism(
        behavior=behavior,
        context_id=context_id,
        negatives=D._organism_negatives(context_id),
        seed=42,
    )
    assert "default" not in organism.panel

    # Arm A keeps the default panel (persona source, disjoint by construction).
    _behavior_a, context_a = D._behavior_context("a1_lora_r1")
    assert D._organism_negatives(context_a) == DEFAULT_PANEL_NAME


def test_base_rate_reads_base_side_no_adapter(tmp_path, monkeypatch) -> None:
    """Regression (#1112 rankem smoke crash 2): the base-rate leg calls the
    generation seam with side='base', side_path=None — never the bare model id
    as a checkpoint path (vLLM then dies with 'No adapter found')."""
    import explore_persona_space.artifacts.organisms as org

    calls: dict = {}

    def fake_gen_factory(base_model, **kw):
        def gen(side_path, messages_list, *, n, temperature):
            raise AssertionError("gen must only be reached via _generate_and_persist")

        gen.close = lambda: calls.setdefault("closed", True)
        return gen

    def fake_gap(gen_fn, side, side_path, ctx, questions, *, n, temperature, out_dir, base_model):
        calls["side"] = side
        calls["side_path"] = side_path
        return [["hello"] * n for _ in questions]

    class _Cell:
        rate = 0.25

    def fake_rate_for_cell(
        behavior, predicate, judge_fn, n_judge_draws, side, ctx, questions, completions, judge_root
    ):
        calls["judge_side"] = side
        return _Cell()

    monkeypatch.setattr(org, "_default_vllm_generate_fn", fake_gen_factory)
    monkeypatch.setattr(org, "_generate_and_persist", fake_gap)
    monkeypatch.setattr(org, "_rate_for_cell", fake_rate_for_cell)

    cfg = _cfg(tmp_path)
    rate = D._base_rate(cfg, "b1_lora_em")
    assert rate == 0.25
    assert calls["side"] == "base"
    assert calls["side_path"] is None
    assert calls["judge_side"] == "base"
    assert calls.get("closed") is True
