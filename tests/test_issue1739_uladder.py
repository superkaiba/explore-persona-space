from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import issue1739_uladder_fold as fold
from scripts import issue1739_uladder_run as runner
from scripts import issue1739_uladder_score as score
from scripts import paper_fig_unlabeled_data_ladder as figure


def test_nested_prefixes_are_exact_and_nested() -> None:
    prefixes = score._nested_prefixes(20, [3, 8, 20], seed=2, namespace=99)
    assert {k: len(v) for k, v in prefixes.items()} == {3: 3, 8: 8, 20: 20}
    assert set(prefixes[3]) < set(prefixes[8]) < set(prefixes[20])
    assert np.array_equal(prefixes[20], np.arange(20))
    assert score._sha_idx(prefixes[8]) == score._sha_idx(prefixes[8].copy())


def test_pool_columns_generic_and_scaled_union() -> None:
    state = SimpleNamespace(
        generic_prefixes={250: np.arange(250)},
        elic_prefixes={100: np.arange(100)},
        n_generic=1000,
        n_eliciting=400,
    )
    generic, ng, ne = score._pool_columns(state, "generic_only", 250)
    assert len(generic) == ng == 250 and ne == 0
    union, ng, ne = score._pool_columns(state, "union_scaled", 250)
    assert (ng, ne, len(union)) == (250, 100, 350)
    assert np.array_equal(union[:250], np.arange(250))
    assert np.array_equal(union[250:], 1000 + np.arange(100))


def test_single_component_shuffle_is_nonidentity_bijection() -> None:
    perm, meta = score._component_shuffle(20, seed=3, namespace=0)
    assert np.array_equal(np.sort(perm), np.arange(20))
    assert not np.array_equal(perm, np.arange(20))
    assert meta["single_component_adapter"] is True


def test_equivalence_and_registered_verdict_branches() -> None:
    equivalent = fold.equivalence_tost([0.001] * 5, delta=0.02, alpha=0.05)
    non_equivalent = fold.equivalence_tost([0.03] * 5, delta=0.02, alpha=0.05)
    assert equivalent["equivalent"] is True
    assert non_equivalent["equivalent"] is False
    trend = fold._t_interval([0.8] * 5, 0.95)
    verdict, _ = fold.classify_verdict(
        trend, non_equivalent, equivalent, delta=0.02
    )
    assert verdict == "SUPPORTED"
    verdict, _ = fold.classify_verdict(
        trend, equivalent, non_equivalent, delta=0.02
    )
    assert verdict == "REFUTED"


def _fake_rows() -> list[dict]:
    rows = []
    settings = {
        "in_dist": ["train"],
        "generic": ["wildchat_rung"],
        "ood": ["hhrt", "toxicchat", "evil_mhj", "evil_pair", "evil_tomgibbs"],
    }
    for seed in range(5):
        for config in fold.CONFIGS:
            for u in fold.U_SIZES:
                d = 0.001 * np.log(u) + seed * 0.0001
                for setting_group, rungs in settings.items():
                    for rung in rungs:
                        for arm, rho in (
                            ("arm4_ridge_ctx", 0.2),
                            ("arm7_map_ridge_pred", 0.2 + d),
                            ("arm12_oracle_reg", 0.3),
                        ):
                            rows.append(
                                {
                                    "behavior": "evil",
                                    "seed": seed,
                                    "config": config,
                                    "map_variant": "true",
                                    "u_size": u,
                                    "setting_group": setting_group,
                                    "eval_rung": rung,
                                    "arm": arm,
                                    "rho_frozen": rho,
                                }
                            )
        for config, u in (("judged_only", None), ("fold_clean_union_full", 18793)):
            used_settings = settings if config == "judged_only" else {"in_dist": ["train"]}
            for setting_group, rungs in used_settings.items():
                for rung in rungs:
                    for arm, rho in (
                        ("arm4_ridge_ctx", 0.2),
                        ("arm7_map_ridge_pred", 0.208),
                        ("arm12_oracle_reg", 0.3),
                    ):
                        rows.append(
                            {
                                "behavior": "evil",
                                "seed": seed,
                                "config": config,
                                "map_variant": "true",
                                "u_size": u,
                                "setting_group": setting_group,
                                "eval_rung": rung,
                                "arm": arm,
                                "rho_frozen": rho,
                            }
                        )
    return rows


def test_fold_keeps_behavior_setting_groups_separate() -> None:
    payload = fold.fold(
        _fake_rows(),
        behaviors=["evil"],
        seeds=[0, 1, 2, 3, 4],
        delta=0.02,
        alpha=0.05,
    )
    assert len(payload["groups"]) == 6
    assert {
        (group["config"], group["setting_group"]) for group in payload["groups"]
    } == {
        (config, setting) for config in fold.CONFIGS for setting in fold.SETTING_GROUPS
    }
    assert len(payload["fold_clean_diagnostic"]) == 1


def test_runner_pilot_uses_separate_output_and_binds_scorer() -> None:
    args = runner.parse_args(["--pilot", "--behaviors", "evil", "sycophancy", "--seeds", "0", "1"])
    assert args.behaviors == ["evil"] and args.seeds == [0]
    assert args.out_root == runner.DEFAULT_PILOT_OUT_ROOT
    parsed = score.parse_args(runner.score_cmd(args, "evil", 0)[2:])
    assert parsed.pilot is True and parsed.behaviors == ["evil"] and parsed.seed == 0


def test_runner_allows_distinct_cuda_pilot_hf_prefix() -> None:
    args = runner.parse_args(
        ["--pilot", "--hf-prefix", "/issue1739_uladder_h100_pilot/"]
    )
    assert runner._hf_prefix(args) == "issue1739_uladder_h100_pilot"


def test_runner_materialized_staging_flag_reaches_base_stage_namespace() -> None:
    args = runner.parse_args(["--materialize-labeling-tars", "--behaviors", "evil"])
    staged = runner._jobd_namespace(args, "evil")
    assert args.materialize_labeling_tars is True
    assert staged.materialize_labeling_tars is True


def test_resume_fingerprint_distinguishes_pilot_from_production(tmp_path: Path) -> None:
    path = tmp_path / "all_arms_spearman.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "schema_version": score.SCHEMA_VERSION,
                    "commit": "abc",
                    "seed": 0,
                    "complete": True,
                    "u_sizes": [18793],
                    "configs": ["generic_only"],
                    "map_variants": ["true"],
                    "pilot": True,
                }
            }
        )
    )
    pilot_ok, _ = score._resume_ok(
        path,
        commit="abc",
        seed=0,
        u_sizes=[18793],
        configs=["generic_only"],
        map_variants=["true"],
        pilot=True,
    )
    production_ok, _ = score._resume_ok(
        path,
        commit="abc",
        seed=0,
        u_sizes=list(score.U_SIZES),
        configs=list(score.MAP_CONFIGS),
        map_variants=list(score.MAP_VARIANTS),
        pilot=False,
    )
    assert pilot_ok is True and production_ok is False


def test_runner_reserves_one_terminal_phase_done_line() -> None:
    source = Path(runner.__file__).read_text()
    assert source.count("[phase=done]") == 1


def _fake_fold_payload() -> dict:
    groups = []
    u_sizes = list(fold.U_SIZES)
    for behavior in figure.BEHAVIORS:
        for setting in figure.SETTINGS:
            for config in fold.CONFIGS:
                ladders = {
                    str(seed): {
                        str(u): 0.01 * (i + 1) + seed * 0.0002
                        for i, u in enumerate(u_sizes)
                    }
                    for seed in range(5)
                }
                groups.append(
                    {
                        "behavior": behavior,
                        "setting_group": setting,
                        "config": config,
                        "ladder_by_seed": ladders,
                    }
                )
    return {"u_sizes": u_sizes, "groups": groups}


def test_figure_render_writes_all_exports_and_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "fold.json"
    source.write_text(json.dumps(_fake_fold_payload()))
    outputs = figure.render(source, tmp_path, "c5_test")
    assert all(Path(path).exists() for path in outputs.values())
    meta = json.loads(Path(outputs["meta"]).read_text())
    assert meta["render"]["style_version"] == "c2a-v2"
    assert meta["render"]["include_width_frac"] == 1.0
    assert len(meta["plotted_values"]) == 9
