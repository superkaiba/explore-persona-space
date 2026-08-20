"""#2388 round-2 revision fixes — pinned regression tests.

Covers the r1 code-review blockers (fails-pre-fix where the fix is a permanent
invariant): H3 stage2 per-leg out-roots, rung-1 fit-side restriction, the
`_ported_cmd` argv binding against the REAL child validator, the code-gate
KEEP / DROP->APPS branches + gate-consumer refusals, sandbox hardening
(real-body execution), smoke HF-prefix isolation, genmeta resume pins,
`--regen-cap-hit` pruning, stale-shard clearing, the batched PCA basis
equivalence, the shuffled-MLP control, dv_build gate consumption + agree_frac,
and bootstrap checkpoint/fail-loud semantics.

Adoptable: repo-root-relative paths, tmp_path outputs, no network / GPU.
CONTENT HYGIENE: all fixtures are benign synthetic text.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_script("issue2388_gen_r2", "scripts/issue2388_gen.py")


@pytest.fixture(scope="module")
def drv():
    return _load_script("issue2388_fits_r2", "scripts/issue2388_fits.py")


@pytest.fixture(scope="module")
def dvb():
    return _load_script("issue2388_dv_build_r2", "scripts/issue2388_dv_build.py")


# ---------------------------------------------------------------------------
# gen: caps / dedup key / restricted pickle / sandbox
# ---------------------------------------------------------------------------


def test_max_tokens_2048_all_surfaces(gen):
    """Plan section 10: max_new_tokens 2048 on ALL surfaces (r1 blocker 1)."""
    for bench, cap in gen.MAX_TOKENS.items():
        assert cap == 2048, (bench, cap)


def test_dedup_key_alnum_matches_slug_variants(gen):
    """r1 g3 Concern 5: dash-slugify under-matched punctuation-bearing titles."""
    assert gen._dedup_key("Find the K-th Character in String Game I!") == gen._dedup_key(
        "find-the-k-th-character-in-string-game-i"
    )
    assert gen._dedup_key("A, B & C") == "abc"


def test_restricted_unpickler_plain_str_ok_global_raises(gen):
    assert gen._restricted_pickle_loads(pickle.dumps("[]")) == "[]"
    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        gen._restricted_pickle_loads(pickle.dumps(Path))  # any global reference


def test_sandbox_env_scrubbed_and_home_isolated(gen, monkeypatch, tmp_path):
    """Real-body execution: credentials never reach model-generated code."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake_for_test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-test")
    snippet = (
        "import os\n"
        "assert 'HF_TOKEN' not in os.environ, 'credential leaked'\n"
        "assert 'ANTHROPIC_API_KEY' not in os.environ, 'credential leaked'\n"
        "assert os.environ['HOME'] == os.getcwd(), (os.environ['HOME'], os.getcwd())\n"
    )
    assert gen._run_code(snippet, timeout_s=30) is True
    assert gen._run_code("raise SystemExit(1)", timeout_s=30) is False


def test_sandbox_timeout_kills_process_group(gen):
    t0 = time.time()
    assert gen._run_code("import time; time.sleep(60)", timeout_s=2) is False
    assert time.time() - t0 < 30, "timeout kill took too long (group kill not engaging?)"


def test_stdout_matches_rstrip_per_line(gen):
    assert gen._stdout_matches("1 \n2\n", "1\n2") is True
    assert gen._stdout_matches("1\n3\n", "1\n2") is False


# ---------------------------------------------------------------------------
# gen: gate branches + consumers
# ---------------------------------------------------------------------------


def _gate_fixture(
    tmp_path: Path,
    gen,
    *,
    bcb_ok: bool,
    apps_ok: bool | None = None,
    spread_admissible: bool | None = True,
    n_lcb_kept: int = 507,
) -> Path:
    out_root = tmp_path / "gen"
    (out_root / "code").mkdir(parents=True, exist_ok=True)
    (out_root / "code" / "dedup_report.json").write_text(
        json.dumps({"n_lcb": 880, "n_dropped_lcb": 880 - n_lcb_kept})
    )
    benches: dict = {"bigcodebench": {"harness_ok": bcb_ok, "flaky_mismatch_fraction": 0.0}}
    if apps_ok is not None:
        benches["apps_intro"] = {"harness_ok": apps_ok}
    (out_root / gen.CONTROL_REPORT).parent.mkdir(parents=True, exist_ok=True)
    (out_root / gen.CONTROL_REPORT).write_text(json.dumps({"benchmarks": benches}))
    if spread_admissible is not None:
        (out_root / "code" / "bigcodebench_full.json").write_text(
            json.dumps(
                {
                    "smoke": False,
                    "n_items": gen.EXPECTED_COUNTS["bigcodebench_full"],
                    "admissible": spread_admissible,
                }
            )
        )
    return out_root


def test_gate_keep_branch(gen, tmp_path):
    out_root = _gate_fixture(tmp_path, gen, bcb_ok=True)
    verdict = gen.phase_gate(out_root)
    assert verdict["bcb_gen_allowed"] is True
    assert verdict["bcb_fit_allowed"] is True
    # big pool with BCB kept: est train >= d floor -> APPS not required
    assert verdict["apps_required"] is False
    assert verdict["apps_activated"] is False
    # consumers accept
    gen._require_gate_for("bigcodebench_full", out_root)


def test_gate_drop_to_apps_branch(gen, tmp_path):
    """G1 fail -> BCB excluded; pool falls under d -> APPS required; APPS
    activates only behind its OWN harness control (r1 blocker 8)."""
    out_root = _gate_fixture(tmp_path, gen, bcb_ok=False, apps_ok=True, n_lcb_kept=0)
    verdict = gen.phase_gate(out_root)
    assert verdict["bcb_gen_allowed"] is False
    assert verdict["bcb_fit_allowed"] is False
    est = verdict["pool_arithmetic"]["est_train_without_bcb"]
    assert est < gen.CODE_TRAIN_FLOOR
    assert verdict["apps_required"] is True
    assert verdict["apps_activated"] is True
    with pytest.raises(RuntimeError, match="G1"):
        gen._require_gate_for("bigcodebench_full", out_root)
    gen._require_gate_for("apps_intro", out_root)  # required + activated -> allowed


def test_gate_apps_refused_when_not_required(gen, tmp_path):
    out_root = _gate_fixture(tmp_path, gen, bcb_ok=True, apps_ok=True)
    gen.phase_gate(out_root)
    with pytest.raises(RuntimeError, match="apps"):
        gen._require_gate_for("apps_intro", out_root)


def test_gate_refuses_smoke_spread_as_g3(gen, tmp_path):
    out_root = _gate_fixture(tmp_path, gen, bcb_ok=True)
    (out_root / "code" / "bigcodebench_full.json").write_text(
        json.dumps({"smoke": True, "n_items": 10, "admissible": True})
    )
    verdict = gen.phase_gate(out_root)
    assert verdict["g3_bcb"]["full_pool"] is False
    assert verdict["g3_bcb"]["admissible"] is None  # smoke slice can never resolve G3


def test_require_gate_missing_file_fail_loud(gen, tmp_path):
    with pytest.raises(FileNotFoundError, match="gate"):
        gen._require_gate_for("bigcodebench_full", tmp_path / "nowhere")


# ---------------------------------------------------------------------------
# gen: genmeta pin / regen prune / stale shards / smoke upload prefix
# ---------------------------------------------------------------------------


def _write_rollouts(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_check_genmeta_drift_raises(gen, tmp_path):
    roll = tmp_path / "math" / "rollouts" / "math_full.jsonl"
    roll.parent.mkdir(parents=True)
    gen._check_genmeta(roll, "math_full")  # writes the sidecar
    meta_p = roll.with_name("math_full_genmeta.json")
    meta = json.loads(meta_p.read_text())
    meta["temperature"] = 0.123
    meta_p.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="drifted"):
        gen._check_genmeta(roll, "math_full")


def test_regen_prune_drops_cap_hit_rows_and_verdicts(gen, tmp_path):
    out_root = tmp_path / "gen"
    roll = gen._rollouts_path(out_root, "math_full")
    _write_rollouts(
        roll,
        [
            {"item_id": "a", "completions": ["x"], "finish_reasons": ["stop"]},
            {"item_id": "b", "completions": ["y"], "finish_reasons": ["length"]},
        ],
    )
    verd = roll.parent / "math_full_verdicts.jsonl"
    _write_rollouts(
        verd,
        [
            {"item_id": "a", "k": 0, "verdict": True},
            {"item_id": "b", "k": 0, "verdict": False},
        ],
    )
    n = gen._regen_prune(roll, out_root, "math_full")
    assert n == 1
    kept = [json.loads(x)["item_id"] for x in roll.read_text().split("\n") if x.strip()]
    assert kept == ["a"]
    kept_v = [json.loads(x)["item_id"] for x in verd.read_text().split("\n") if x.strip()]
    assert kept_v == ["a"]


def test_shard_jsonl_clears_stale_shards(gen, tmp_path):
    src = tmp_path / "rows.jsonl"
    src.write_text(json.dumps({"a": 1}) + "\n")
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    stale = shard_dir / "rows.shard07.jsonl"
    stale.write_text("STALE\n")
    n = gen._shard_jsonl(src, shard_dir, "rows")
    assert n == 1
    assert not stale.exists(), "stale higher-index shard survived the re-shard"


def test_gen_upload_smoke_prefix_isolated(gen, tmp_path, monkeypatch):
    """r1 blocker 2: a smoke upload must NEVER land under the production
    HF prefix."""
    from explore_persona_space.orchestrate import hub

    out_root = tmp_path / "gen_smoke"
    _write_rollouts(
        gen._rollouts_path(out_root, "math_full"),
        [{"item_id": "a", "completions": ["x"], "finish_reasons": ["stop"]}],
    )
    captured: list[str] = []

    def fake_upload(
        local_path,
        repo_id,
        repo_type,
        path_in_repo,
        delete_after=False,
        upload_as_file=False,
        ignore_patterns=None,
        private=False,
        raise_on_error=False,
    ):
        captured.append(path_in_repo)
        return "https://huggingface.co/fake"

    monkeypatch.setattr(hub, "_upload", fake_upload)
    gen.phase_upload("math_full", out_root, smoke=True)
    assert captured and captured[0].startswith(f"{gen.HF_GEN_PREFIX}_smoke/")
    captured.clear()
    gen.phase_upload("math_full", out_root, smoke=False)
    assert captured and captured[0].startswith(f"{gen.HF_GEN_PREFIX}/")


# ---------------------------------------------------------------------------
# fits: _ported_cmd binds the REAL child CLI + required set
# ---------------------------------------------------------------------------


def _h3_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        qa_store_dir="/workspace/store",
        u_store_dir="/workspace/u_store",
        h3_store_root="/workspace/h3_stores",
        device="cuda",
        ported_extra=None,
    )


@pytest.mark.parametrize("behavior", ["hallucination", "sycophancy", "evil"])
def test_ported_cmd_binds_real_child_validator(drv, tmp_path, behavior):
    """r1 Codex blocker: the composed argv must satisfy issue1739_fits.py's
    REAL parser AND its real-mode required set (labeled/dv/u/e1 stores)."""
    ported = _load_script("issue1739_fits_r2bind", "scripts/issue1739_fits.py")
    cmd = drv._ported_cmd(
        _h3_args(tmp_path),
        behavior,
        tmp_path / "out",
        budgets="2500",
        dof_cap="0.9",
        dv_json="eval_results/issue_2388/dv/qa/labeling.json",
        label="h3_parent_exact",
    )
    ns = ported._parse_args(cmd[2:])  # cmd[0:2] = python, script path
    for req in ("labeled_store", "dv_json", "u_store", "e1_store"):
        assert getattr(ns, req) is not None, f"--{req} missing from composed argv"
    assert ns.budgets == [2500]
    assert ns.device == "cuda"
    assert ns.dof_cap == pytest.approx(0.9)
    if behavior == "hallucination":
        assert str(ns.labeled_store) == "/workspace/store"
    else:
        assert str(ns.labeled_store) == f"/workspace/h3_stores/{behavior}_labeling"
    assert str(ns.e1_store) == f"/workspace/h3_stores/{behavior}_extraction"


# ---------------------------------------------------------------------------
# fits: stage2 per-leg out-roots (fails pre-fix: one root, last-leg-wins)
# ---------------------------------------------------------------------------


def test_h3_stage2_per_leg_roots_keep_capped_rows(drv, tmp_path, monkeypatch):
    fits_root = tmp_path / "fits"
    (tmp_path / "h3_recompute").mkdir(parents=True)
    (tmp_path / "h3_recompute" / "stage1_verdict.json").write_text("{}")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str]) -> None:
        calls.append(cmd)
        out_root = Path(cmd[cmd.index("--out-root") + 1])
        budget = cmd[cmd.index("--budgets") + 1]
        d = out_root / "arm_results"
        d.mkdir(parents=True, exist_ok=True)
        (d / "all_arms_spearman.json").write_text(
            json.dumps(
                {
                    "arm_rows": [
                        {"h3_label": "h3_parent_exact", "budget_l": int(budget), "rho_frozen": 0.5}
                    ]
                }
            )
        )

    monkeypatch.setattr(drv, "_run", fake_run)
    args = argparse.Namespace(
        fits_root=str(fits_root),
        dv_root=str(tmp_path / "dv"),
        banked_root=str(tmp_path / "banked"),
        behaviors=None,
        h3_step="stage2",
        h3_u_rung="full",
        resume=False,
        qa_store_dir="/workspace/store",
        u_store_dir="/workspace/u_store",
        h3_store_root="/workspace/h3_stores",
        device="cpu",
        ported_extra=None,
    )
    drv.phase_h3(args)
    assert len(calls) == 3
    roots = {Path(c[c.index("--out-root") + 1]).name for c in calls}
    assert roots == {"h3_stage2_capped2500", "h3_stage2_legacy8000", "h3_stage2_legacy16000"}
    agg = json.loads((fits_root / "qa" / "h3_parent_exact.json").read_text())
    budgets = sorted(r["budget_l"] for r in agg["rows"])
    # PRE-FIX failure shape: a single shared out-root left only the LAST leg's
    # rows (16000) — the capped 2,500 anchor (the stage-2 kill read) vanished.
    assert budgets == [2500, 8000, 16000]
    assert agg["legs"] == {"capped2500": 1, "legacy8000": 1, "legacy16000": 1}


# ---------------------------------------------------------------------------
# fits: rung-1 fit-side restriction
# ---------------------------------------------------------------------------


def _mini_table(drv, surface: str):
    n = 12
    dv = np.linspace(0, 1, n)
    return drv.SurfaceTable(
        surface=surface,
        ctx_ids=[f"c{i}" for i in range(n)],
        dv=dv,
        split=np.array(["train"] * 8 + ["test"] * 4),
        group=np.array([f"g{i}" for i in range(n)]),
        boot_group=np.array([f"b{i % 3}" for i in range(n)]),
        benchmark=np.array(
            ["humaneval", "mbpp_full", "lcb_v5", "bigcodebench_full"] * 3
            if surface == "code"
            else ["x"] * n
        ),
        level=np.array([1, 2, 3, 4, 5, 1, 2, 4, 1, 2, 4, 5], dtype=float),
        category=np.array([f"cat{i % 4}" for i in range(n)]),
        z_ctx=np.zeros((1, n, 4), dtype=np.float16),
        z_t1=np.zeros((1, n, 4), dtype=np.float16),
        z_tlast=None,
    )


def test_rung1_fit_filter_math_and_disjoint_assert(drv):
    t = _mini_table(drv, "math")
    rows = np.arange(8)
    fit = drv._rung1_fit_filter(t, rows)
    assert set(t.level[fit]) <= set(drv.MATH_RUNG1_FIT_LEVELS)
    drv._assert_rung1_disjoint(t, fit, None)
    with pytest.raises(RuntimeError, match="overlap"):
        drv._assert_rung1_disjoint(t, rows, None)  # unrestricted rows include L4/L5


def test_rung1_fit_filter_mcq_requires_and_excludes_heldout(drv):
    t = _mini_table(drv, "mcq")
    rows = np.arange(8)
    with pytest.raises(RuntimeError, match="held-out"):
        drv._rung1_fit_filter(t, rows)
    held = np.array(["cat0"])
    fit = drv._rung1_fit_filter(t, rows, held)
    assert not np.isin(t.category[fit], held).any()
    drv._assert_rung1_disjoint(t, fit, held)
    with pytest.raises(RuntimeError, match="overlap"):
        drv._assert_rung1_disjoint(t, rows, held)


def test_rung1_fit_filter_code(drv):
    t = _mini_table(drv, "code")
    rows = np.arange(12)
    fit = drv._rung1_fit_filter(t, rows)
    assert set(t.benchmark[fit]) <= set(drv.CODE_RUNG1_FIT)
    assert "bigcodebench_full" not in set(t.benchmark[fit])


# ---------------------------------------------------------------------------
# fits: batched PCA basis equivalence + shuffled-MLP control
# ---------------------------------------------------------------------------


def test_pca_basis_matches_svd_subspace(drv):
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(3, 40, 12)).astype(np.float16)
    k = 5
    v = drv._pca_basis(pool, k)
    assert v.shape == (3, 12, k)
    for ly in range(3):
        xc = pool[ly].astype(np.float64)
        xc = xc - xc.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        ref = vt[:k].T
        # column signs are arbitrary — compare subspace PROJECTORS
        p_new = v[ly] @ v[ly].T
        p_ref = ref @ ref.T
        assert np.allclose(p_new, p_ref, atol=1e-8), np.abs(p_new - p_ref).max()


def test_shuffled_nl_payloads_permutes_input_axis(drv):
    torch = pytest.importorskip("torch")
    d_in, hid, d_out = 6, 4, 6
    payload = {
        "kind": "mlp",
        "width": hid,
        "state_dict": {
            "0.weight": torch.randn(hid, d_in),
            "0.bias": torch.zeros(hid),
            "2.weight": torch.randn(d_out, hid),
            "2.bias": torch.zeros(d_out),
        },
        "xmu": torch.randn(d_in),
        "xsd": torch.ones(d_in),
        "ymu": torch.zeros(d_out),
    }
    shuf = drv._shuffled_nl_payloads((payload,), seed=0)[0]
    w0, w0s = payload["state_dict"]["0.weight"], shuf["state_dict"]["0.weight"]
    assert float(w0.norm()) == pytest.approx(float(w0s.norm()))
    assert not torch.equal(w0, w0s)
    # xmu permuted CONSISTENTLY with the weight columns: for the permutation p,
    # shuf column j corresponds to original column p[j] with xmu[p[j]]
    for j in range(d_in):
        orig_col = (w0 == w0s[:, j : j + 1]).all(dim=0).nonzero().flatten()
        assert len(orig_col) == 1
        assert float(shuf["xmu"][j]) == pytest.approx(float(payload["xmu"][int(orig_col)]))


# ---------------------------------------------------------------------------
# fits: bootstrap fail-loud group lookup + checkpoint/resume
# ---------------------------------------------------------------------------


def _boot_env(drv, tmp_path, *, ids=None):
    surface = "math"
    out_root = tmp_path / "fits" / surface
    (out_root / "preds").mkdir(parents=True)
    ids = ids or [f"mathfull-x-{i}" for i in range(9)]
    rng = np.random.default_rng(1)
    for arm in ("arm_ctx", "arm_maplin"):
        with (out_root / "preds" / f"preds_{arm}_L16_draw0.jsonl").open("w") as fh:
            for i, cid in enumerate(ids):
                fh.write(
                    json.dumps(
                        {
                            "eval": "dev",
                            "context_id": cid,
                            "y_true": i / len(ids),
                            "y_pred": float(rng.normal()),
                        }
                    )
                    + "\n"
                )
    dv_dir = tmp_path / "dv" / surface
    dv_dir.mkdir(parents=True)
    rows = [
        {
            "context_id": cid,
            "dv": i / len(ids),
            "split": "train",
            "group_key": f"g{i % 3}",
            "subject": "algebra",
            "level": 1,
        }
        for i, cid in enumerate(ids)
    ]
    (dv_dir / "labeling.json").write_text(json.dumps({"rows": rows}))
    return argparse.Namespace(
        surface=surface,
        fits_root=str(tmp_path / "fits"),
        dv_root=str(tmp_path / "dv"),
        n_boot=10,
        force=False,
    )


def test_bootstrap_checkpoints_and_resumes(drv, tmp_path):
    args = _boot_env(drv, tmp_path)
    drv.phase_bootstrap(args)
    cells_p = tmp_path / "fits" / "math" / "bootstrap_cells.jsonl"
    first = cells_p.read_text()
    assert first.strip(), "no per-unit checkpoint rows written"
    row = json.loads(first.split("\n")[0])
    assert row["ci_kind"] == "frozen-config"
    drv.phase_bootstrap(args)  # resume: no new units appended
    assert cells_p.read_text() == first
    summary = json.loads((tmp_path / "fits" / "math" / "bootstrap_summary.json").read_text())
    assert summary["ci_kind"] == "frozen-config"
    assert len(summary["cells"]) == 1


def test_bootstrap_missing_boot_group_fail_loud(drv, tmp_path):
    args = _boot_env(drv, tmp_path)
    # poison ONE pred id so it is absent from the labeling's group map
    pf = tmp_path / "fits" / "math" / "preds" / "preds_arm_ctx_L16_draw0.jsonl"
    lines = [json.loads(x) for x in pf.read_text().split("\n") if x.strip()]
    lines[0]["context_id"] = "NOT-IN-LABELING"
    pf.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    with pytest.raises(RuntimeError, match="missing from the labeling"):
        drv.phase_bootstrap(args)


# ---------------------------------------------------------------------------
# fits: pre-gen feasibility
# ---------------------------------------------------------------------------


def test_pre_gen_feasibility_from_dedup_arithmetic(drv, tmp_path):
    gen_root = tmp_path / "gen"
    (gen_root / "code").mkdir(parents=True)
    (gen_root / "code" / "dedup_report.json").write_text(
        json.dumps({"n_lcb": 880, "n_dropped_lcb": 373})
    )
    args = argparse.Namespace(
        gen_root=str(gen_root),
        dv_root=str(tmp_path / "dv"),
        fits_root=str(tmp_path / "fits"),
        maps_out=str(tmp_path / "maps"),
        surfaces=None,
        pre_gen=True,
    )
    drv.phase_feasibility(args)
    rep = json.loads((tmp_path / "fits" / "feasibility_report_pregen.json").read_text())
    assert rep["mode"] == "pre-gen-arithmetic"
    # conservative WITHOUT-BCB pool: 164 + 974 + 507 + 2869 = 4514 -> train 3160
    assert rep["surfaces"]["code"]["n_train"] == round(0.7 * (164 + 974 + 507 + 2869))
    assert "math" in rep["surfaces"] and "mcq" in rep["surfaces"]


# ---------------------------------------------------------------------------
# dv_build: gate consumption + agree_frac + realized floor
# ---------------------------------------------------------------------------


def _dv_gate(tmp_path: Path, payload: dict) -> Path:
    gen_root = tmp_path / "gen"
    (gen_root / "code").mkdir(parents=True, exist_ok=True)
    (gen_root / "code" / "code_gate.json").write_text(json.dumps(payload))
    return gen_root


def test_code_benchmarks_from_gate_variants(dvb, tmp_path):
    gen_root = _dv_gate(tmp_path, {"bcb_fit_allowed": None})
    with pytest.raises(RuntimeError, match="unresolved"):
        dvb._code_benchmarks_from_gate(gen_root)
    gen_root = _dv_gate(tmp_path, {"bcb_fit_allowed": False, "apps_activated": False})
    benches, dec = dvb._code_benchmarks_from_gate(gen_root)
    assert "bigcodebench_full" not in benches and "apps_intro" not in benches
    assert dec["excluded_benchmarks"] == ["bigcodebench_full"]
    gen_root = _dv_gate(tmp_path, {"bcb_fit_allowed": True, "apps_activated": True})
    benches, dec = dvb._code_benchmarks_from_gate(gen_root)
    assert "bigcodebench_full" in benches and "apps_intro" in benches
    with pytest.raises(FileNotFoundError, match="gate"):
        dvb._code_benchmarks_from_gate(tmp_path / "nowhere")


def test_agree_frac_math_normalized_identity(dvb):
    comps = [
        "So the answer is \\boxed{\\dfrac{1}{2}}.",
        "Thus \\boxed{\\frac{1}{2}} holds.",
        "I think \\boxed{3}.",
        "no boxed answer here",
    ]
    frac, n = dvb._agree_frac("math", comps)
    assert n == 3
    assert frac == pytest.approx(2 / 3)  # dfrac/frac normalize to the same key


def test_agree_frac_mcq_letters_and_underflow(dvb):
    frac, n = dvb._agree_frac("mcq", ["Answer: A", "Answer: (a)", "Answer: B"])
    assert (frac, n) == (pytest.approx(2 / 3), 3)
    frac, n = dvb._agree_frac("mcq", ["no letter anywhere -"])
    assert frac is None


def _code_gen_fixture(tmp_path: Path, benches: list[str], n_items: int = 4) -> Path:
    gen_root = tmp_path / "gen"
    (gen_root / "code").mkdir(parents=True, exist_ok=True)
    for bench in benches:
        items = [
            {
                "item_id": f"{bench}-{i}",
                "benchmark": bench,
                "verdicts": [True, False, True, True, None],
            }
            for i in range(n_items)
        ]
        (gen_root / "code" / f"{bench}.json").write_text(
            json.dumps({"benchmark": bench, "k_rollouts": 5, "items": items})
        )
    return gen_root


def test_dv_build_code_realized_floor_fail_loud(dvb, tmp_path):
    benches = ["humaneval", "mbpp_full", "lcb_v5", "leetcode"]
    gen_root = _code_gen_fixture(tmp_path, benches)
    _dv_gate(
        tmp_path,
        {
            "bcb_fit_allowed": False,
            "apps_activated": False,
            "pool_arithmetic": {"code_train_floor_d": 999},
        },
    )
    with pytest.raises(RuntimeError, match="APPS"):
        dvb.build_surface_dv("code", gen_root, tmp_path / "dv")


# ---------------------------------------------------------------------------
# capture: MCQ tf-margin TOTALS (un-normalized) + aggregate under the dv root
# ---------------------------------------------------------------------------


class _TfFakeTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<u>{messages[0]['content']}</u><a>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [(ord(c) % 250) + 1 for c in text]


def test_tf_margin_mcq_totals_and_aggregate(tmp_path, monkeypatch):
    """r1 g5 Minor 3: the margin is computed on TOTAL ln-probs (mean x token
    count), and the rows land under the DV root with a per-surface aggregate
    (r1 Codex artifact-path blocker)."""
    cap = _load_script("issue2388_capture_r2", "scripts/issue2388_capture.py")
    items = [
        {
            "item_id": f"mmlupro-{i}",
            "benchmark": "mmlu_pro_full",
            "n_options": 3,
            "gold": "A",
            "prompt": f"Question {i}?",
        }
        for i in range(2)
    ]
    monkeypatch.setitem(cap.G.LOADERS, "mmlu_pro_full", lambda: items)
    monkeypatch.setattr(cap, "get_tokenizer", lambda: _TfFakeTokenizer())

    def fake_load_capture_model(device: str = "cuda"):
        return object()

    def fake_tf_ln_logp(pairs, *, model, tokenizer, device="cuda", batch_size=8, max_model_len=0):
        return [-1.0] * len(pairs)  # per-token MEAN lp for every pair

    monkeypatch.setattr(cap, "load_capture_model", fake_load_capture_model)
    monkeypatch.setattr(cap, "teacher_forced_ln_logp", fake_tf_ln_logp)
    args = argparse.Namespace(
        benchmark="mmlu_pro_full",
        out_root=str(tmp_path / "gen"),
        dv_root=str(tmp_path / "dv"),
        smoke=False,
        device="cpu",
        batch_size=8,
    )
    cap.phase_tf_margin(args)
    rows_p = tmp_path / "dv" / "mcq" / "tf_margin" / "mmlu_pro_full_tf.jsonl"
    rows = [json.loads(x) for x in rows_p.read_text().split("\n") if x.strip()]
    assert len(rows) == 2
    r = rows[0]
    n_tok = len("Answer: A")  # char-level fake: 1 token per char
    assert r["lp"]["A"] == pytest.approx(-1.0 * n_tok)
    assert r["lp_per_token"]["A"] == pytest.approx(-1.0)
    # equal totals across options: margin = lp_A - logsumexp(B, C) = -n - (-n + ln 2)
    assert r["tf_margin"] == pytest.approx(-np.log(2))
    agg = json.loads((tmp_path / "dv" / "mcq" / "tf_margin.json").read_text())
    assert agg["n_rows"] == 2 and "mmlu_pro_full" in agg["benchmarks_included"]


def test_dv_build_code_gate_decisions_in_payload(dvb, tmp_path):
    benches = ["humaneval", "mbpp_full", "lcb_v5", "leetcode"]
    gen_root = _code_gen_fixture(tmp_path, benches)
    _dv_gate(
        tmp_path,
        {
            "bcb_fit_allowed": False,
            "apps_activated": False,
            "pool_arithmetic": {"code_train_floor_d": 1},
        },
    )
    out = dvb.build_surface_dv("code", gen_root, tmp_path / "dv")
    payload = json.loads(Path(out).read_text())
    assert payload["gate_decisions"]["excluded_benchmarks"] == ["bigcodebench_full"]
    assert payload["gate_decisions"]["realized_train_with_dv"] >= 1
    assert payload["agree_definition"].startswith("N/A")
    assert all(r["agree_frac"] is None for r in payload["rows"])
