"""#1335 onpolicy-assistant-label round pins (plan v7 §4.2).

Covers the round's diff surface:
  1. render_config_hash("r7_endpoint") byte-stability (the CRITICAL §4.2
     invariant): structural old-shape reconstruction (env-independent) AND the
     committed default-env baseline hash in a clean subprocess;
  2. the three r7_op_* rung registrations + personas_for_rung /
     gen_seed_for_rung helpers (incl. the in-process label guards);
  3. gen_fiction PRODUCTION-BODY run for the label cells (real tokenizer,
     signature-conformant recorder fakes ONLY at the vLLM engine boundary):
     per-slot seed threading (rung seed + slot), the Assistant cue render, the
     Wren-description reuse, record schema;
  4. rung_units on the single-lead stores (the highest-severity diff site) +
     the committed 4-persona behavior unchanged;
  5. collapse_audit full-slot modal-line generalization (a slot-2 "I agree."
     mode is caught; legacy slot-4 fields kept; missing-file fail-loud);
  6. the empirical H0 pair-noise band: synthetic two-way fixture recovering a
     known interaction SD, AND the REAL committed 12-cell computation
     (eval_results/issue_1335 — registered in tests/sparse_cones.txt);
  7. _full_n_value/_delta joint-draw pairing + the variance-sum fallback gate;
  8. _committed_placement_n body-value drift guard;
  9. the thin driver env contract + its df -P staging headroom assert (both
     branches, via a PATH-stubbed df).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

r1335 = pytest.importorskip("issue1335_render_rungs")
g1335 = pytest.importorskip("issue1335_gen")
f1335 = pytest.importorskip("issue1335_fit")
c1310 = pytest.importorskip("issue1310_common")
common931 = pytest.importorskip("issue931_common")

LABEL_RUNGS = ("r7_op_assistant", "r7_op_wren", "r7_op_wren46")
DRIVER_SH = REPO_ROOT / "scripts" / "issue1335_onpolicy_label_run.sh"
RUN_SH = REPO_ROOT / "scripts" / "issue1335_run.sh"
COMMITTED_EVAL_ROOT = REPO_ROOT / "eval_results" / "issue_1335"
# Pre-round committed r7_endpoint render-config hash at the DEFAULT env
# (GEN_SEED 42; computed from the committed code before this round's diffs).
R7_ENDPOINT_HASH_DEFAULT_ENV = "38dc5c51e194203b"


@pytest.fixture(scope="module")
def tokenizer():
    return common931.get_tokenizer(r1335.MODEL_IDS["base"])


# ---------------------------------------------------------------------------
# (1) render-config hash byte-stability
# ---------------------------------------------------------------------------


def test_r7_endpoint_hash_structural_old_shape():
    """The helpers-based config for every PRE-EXISTING fiction rung equals the
    OLD literal construction byte-for-byte (personas = dict(c1310.PERSONAS),
    gen_seed = GEN_SEED) — env-independent invariance."""
    import hashlib

    for slug in ("r6_nofoil", "r7_endpoint", "s1_assistant_label"):
        cfg = r1335.rung_render_config(slug)
        old = dict(cfg)
        old["personas"] = dict(c1310.PERSONAS)
        old["gen_seed"] = r1335.GEN_SEED
        assert cfg == old, f"{slug}: helpers changed the rendered config"
        blob = json.dumps(old, sort_keys=True, ensure_ascii=True)
        assert hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16] == r1335.render_config_hash(
            slug
        )


def test_r7_endpoint_hash_committed_baseline_subprocess():
    """Default-env (no EPM_I1335_GEN_SEED) r7_endpoint hash equals the
    pre-round committed value — the smoke-asserted §4.2 invariant."""
    env = {k: v for k, v in os.environ.items() if k != "EPM_I1335_GEN_SEED"}
    res = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'scripts'); "
            "import issue1335_render_rungs as r; "
            "print(r.render_config_hash('r7_endpoint'))",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().splitlines()[-1] == R7_ENDPOINT_HASH_DEFAULT_ENV


# ---------------------------------------------------------------------------
# (2) rung registry + helpers
# ---------------------------------------------------------------------------


def test_label_rung_registry_and_helpers():
    wren_desc = c1310.PERSONAS["Wren"]
    for slug in LABEL_RUNGS:
        cfg = r1335.RUNGS[slug]
        assert cfg["family"] == "fiction" and cfg["gen"] == "op"
        assert cfg["foils"] == "battery" and cfg["group"] == "scene"
        assert cfg["tf_source"] is None and cfg["base_prime"] is False
        assert slug in r1335.FICTION_RUNGS and slug in r1335.FICTION_RENDER_RUNGS
    assert r1335.personas_for_rung("r7_op_assistant") == {"Assistant": wren_desc}
    assert r1335.personas_for_rung("r7_op_wren") == {"Wren": wren_desc}
    assert r1335.personas_for_rung("r7_op_wren46") == {"Wren": wren_desc}
    # default panel for every pre-existing rung
    assert r1335.personas_for_rung("r7_endpoint") == dict(c1310.PERSONAS)
    assert r1335.personas_for_rung("r1_qa_oneline") == dict(c1310.PERSONAS)
    # effective seeds: override only on the replicate cell
    assert r1335.gen_seed_for_rung("r7_op_wren46") == 46
    assert r1335.gen_seed_for_rung("r7_op_wren") == r1335.GEN_SEED
    assert r1335.gen_seed_for_rung("r7_endpoint") == r1335.GEN_SEED
    # fingerprint lineage: the three new cells + the committed endpoint are
    # pairwise distinct (slug + personas + gen_seed all ride the hash)
    hashes = {slug: r1335.render_config_hash(slug) for slug in (*LABEL_RUNGS, "r7_endpoint")}
    assert len(set(hashes.values())) == 4, hashes
    assert "Assistant" not in c1310.FOIL_NAMES


def test_personas_for_rung_label_guards(monkeypatch):
    """The in-process render-time label guards fire (plan v7 §4.2 item 1)."""
    rungs = dict(r1335.RUNGS)
    rungs["bad_foil"] = {**r1335.RUNGS["r7_op_assistant"], "personas": {"Sam": "a foil name"}}
    rungs["bad_cue"] = {**r1335.RUNGS["r7_op_assistant"], "personas": {"As: st": "colon label"}}
    monkeypatch.setattr(r1335, "RUNGS", rungs)
    with pytest.raises(AssertionError, match="foil name"):
        r1335.personas_for_rung("bad_foil")
    with pytest.raises(AssertionError, match="cue"):
        r1335.personas_for_rung("bad_cue")


# ---------------------------------------------------------------------------
# (3) gen_fiction production body (recorder fakes ONLY at the vLLM boundary)
# ---------------------------------------------------------------------------


def _recorder_vllm(tokenizer, seen_seeds):
    def fake_vllm_generate(llm, prompts, *, max_tokens, stop, seed):
        # signature mirrors issue1335_gen._vllm_generate
        seen_seeds.append(seed)
        out = []
        for i, prompt in enumerate(prompts):
            comp = f" A steady line {i} keeps the scene moving with a concrete answer."
            out.append(
                {
                    "completion": comp,
                    "prompt_token_ids": list(
                        tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    ),
                    "completion_token_ids": list(
                        tokenizer(comp, add_special_tokens=False)["input_ids"]
                    ),
                }
            )
        return out

    return fake_vllm_generate


@pytest.mark.parametrize(
    ("slug", "lead", "want_seed0"),
    [
        ("r7_op_assistant", "Assistant", None),  # None -> GEN_SEED (env-dependent)
        ("r7_op_wren46", "Wren", 46),
    ],
)
def test_gen_fiction_label_cells_body(tmp_path, monkeypatch, tokenizer, slug, lead, want_seed0):
    seen_seeds: list[int] = []
    monkeypatch.setattr(g1335, "_vllm_generate", _recorder_vllm(tokenizer, seen_seeds))

    def fake_build_engine(model_id, gpu_memory_utilization, seed):
        return SimpleNamespace(model_id=model_id, seed=seed)

    def fake_teardown_engine(llm):
        return None

    monkeypatch.setattr(g1335.i1310_prefill, "build_engine", fake_build_engine)
    monkeypatch.setattr(g1335.i1310_prefill, "teardown_engine", fake_teardown_engine)
    args = SimpleNamespace(
        rung=slug,
        model="base",
        data_dir=tmp_path,
        n_scenarios=1,
        slots=2,
        stub_gen=False,
        skip_upload=True,
        hf_resume=False,
        gpu_memory_utilization=0.85,
    )
    records, meta = g1335.gen_fiction(args, tokenizer, r1335.fingerprint(slug))
    base_seed = r1335.GEN_SEED if want_seed0 is None else want_seed0
    assert seen_seeds == [base_seed, base_seed + 1]  # per-slot seed = rung seed + slot
    assert meta["personas"] == [lead]
    assert len(records) == 2  # 1 scenario x 1 lead x 2 slots
    wren_desc = c1310.PERSONAS["Wren"]
    for rec in records:
        assert rec["persona"] == lead
        assert rec["gen_seed"] == base_seed
        assert rec["prompt"].endswith(f"{lead}:")  # the committed cue form
        assert f"{lead} is {wren_desc}" in rec["prompt"]  # description reused verbatim
        assert rec["rung"] == slug and rec["render_config_hash"] == r1335.render_config_hash(slug)


# ---------------------------------------------------------------------------
# (4) rung_units single-lead stores
# ---------------------------------------------------------------------------


def _mk_store(char_ids: list[str]) -> dict:
    n = len(char_ids)
    return {
        "row_ids": np.asarray([f"row{i}" for i in range(n)]),
        "group_ids": np.asarray([f"sc{i % 2}" for i in range(n)]),
        "char_ids": np.asarray(char_ids),
        "turn_indices": np.asarray([i % 3 for i in range(n)], dtype=int),
        "arrays": {"y": np.zeros((n, 2, 3), dtype=np.float32)},
    }


def test_rung_units_label_and_committed_cells():
    st = _mk_store(["Assistant"] * 6)
    units = f1335.rung_units("r7_op_assistant", st)
    assert [u for u, _ in units] == ["Assistant"]
    assert units[0][1]["row_ids"].shape[0] == 6
    # committed 4-persona behavior unchanged
    st4 = _mk_store(["Wren", "HELIOS", "Dana", "Vex", "Wren", "Dana"])
    units4 = f1335.rung_units("r7_endpoint", st4)
    assert [u for u, _ in units4] == list(c1310.PERSONA_LABELS)
    # a Wren-lead store fitted under the assistant slug yields NO units (the
    # silent-drop shape rung_units' lead-resolution exists to prevent is loud
    # upstream: build_label_comparison asserts exactly one unit)
    assert f1335.rung_units("r7_op_assistant", _mk_store(["Wren"] * 4)) == []


# ---------------------------------------------------------------------------
# (5) collapse_audit full-slot modal lines
# ---------------------------------------------------------------------------


def _write_rollouts(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_collapse_audit_full_slot_modal(tmp_path):
    args = SimpleNamespace(data_dir=tmp_path)
    rows = []
    for slot in range(3):
        for i in range(4):
            degenerate = slot == 2 and i < 3  # the migrating slot-2 mode
            rows.append(
                {
                    "slot": slot,
                    "persona": "Wren",
                    "completion": "I agree." if degenerate else f" a fuller line {slot}:{i}",
                    "n_completion_tokens": 3 if degenerate else 12,
                }
            )
    _write_rollouts(r1335.gen_path(tmp_path, "r7_op_wren", "base"), rows)
    audit = f1335.collapse_audit(args, "base", "r7_op_wren")
    assert audit["n_lines"] == 12 and audit["under_floor_lines"] == 3
    assert audit["under_floor_per_slot"] == {"slot2": 3}
    m2 = audit["modal_line_per_slot"]["slot2"]
    assert m2["line"] == "I agree." and m2["count"] == 3 and m2["slot_lines"] == 4
    # legacy slot-4 fields kept (no slot 4 in this fixture)
    assert audit["slot4_lines"] == 0 and audit["slot4_exact_agree"] == 0
    # fail-loud on a missing rollout file
    with pytest.raises(AssertionError, match="missing"):
        f1335.collapse_audit(args, "instruct", "r7_op_wren")


# ---------------------------------------------------------------------------
# (6) empirical H0 pair-noise band
# ---------------------------------------------------------------------------


def test_h0_band_math_synthetic(tmp_path):
    """Two-way fixture with a KNOWN interaction: x_ij = a_i + b_j + d*u_i*v_j,
    u=[1,-1,1,-1], v=[1,-1,0] (zero row/col means), so
    sigma_cell = sqrt(8 d^2 / 6)."""
    d = 0.03
    a = [0.30, 0.34, 0.36, 0.28]
    b = [0.00, -0.05, 0.04]
    u = [1.0, -1.0, 1.0, -1.0]
    v = [1.0, -1.0, 0.0]
    for j, (_, rel) in enumerate(f1335.H0_SEED_DIRS):
        root = tmp_path if rel == "." else tmp_path / rel
        root.mkdir(parents=True, exist_ok=True)
        for i, persona in enumerate(c1310.PERSONA_LABELS):
            val = a[i] + b[j] + d * u[i] * v[j]
            p = root / f"cells_r7_endpoint__base__{persona}__ctx.json"
            p.write_text(json.dumps({"r2_per_layer_obs": [val], "headline_layer": 0}))
    band = f1335.label_h0_pair_noise_band(tmp_path)
    want_sigma = math.sqrt(8 * d * d / 6)
    assert band["dof"] == 6
    assert abs(band["sigma_cell"] - want_sigma) < 1e-12
    assert abs(band["b_hat"] - 2 * math.sqrt(2) * want_sigma) < 1e-12
    assert band["b_hat"] >= math.sqrt(2) * band["sigma_cell"]  # mechanical floor


def test_h0_band_real_committed_cells():
    """The REAL committed 12-cell computation (read-only; the cone is
    registered in tests/sparse_cones.txt)."""
    band = f1335.label_h0_pair_noise_band(COMMITTED_EVAL_ROOT)
    assert len(band["sources"]) == 12 and band["dof"] == 6
    assert band["b_hat"] == pytest.approx(2 * math.sqrt(2) * band["sigma_cell"])
    # sane scale: the plan expects B_hat ~ 0.07 (sigma_cell ~ 0.025); a value
    # outside (0, 0.5) means the committed inputs were misread
    assert 0.0 < band["b_hat"] < 0.5
    # committed seed-42 Wren value present verbatim
    assert band["values"]["Wren"]["seed42"] == pytest.approx(0.3585, abs=1e-3)


def test_h0_band_missing_cell_fail_loud(tmp_path):
    with pytest.raises(AssertionError, match="missing committed"):
        f1335.label_h0_pair_noise_band(tmp_path)


# ---------------------------------------------------------------------------
# (7) full-n values + joint-draw / fallback delta
# ---------------------------------------------------------------------------


def _write_cell(out_dir: Path, cell_id: str, r2: float, draws: list[float], guh: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"cells_{cell_id}.json").write_text(
        json.dumps(
            {
                "r2_per_layer_obs": [r2],
                "headline_layer": 0,
                "n": 10,
                "n_groups": 2,
                "group_bootstrap_l19": {
                    "r2": r2,
                    "ci_lo": min(draws),
                    "ci_hi": max(draws),
                    "draws": draws,
                    "group_universe_hash": guh,
                },
            }
        )
    )


def test_full_n_value_and_delta_pairing(tmp_path):
    args = SimpleNamespace(out_dir=tmp_path)
    _write_cell(tmp_path, "a__ctx", 0.30, [0.28, 0.30, 0.32], "g1")
    _write_cell(tmp_path, "b__ctx", 0.25, [0.24, 0.25, 0.26], "g1")
    a = f1335._full_n_value(args, "a__ctx")
    b = f1335._full_n_value(args, "b__ctx")
    dd = f1335._delta(a, b)
    assert dd["ci_method"] == "joint-draws"
    assert dd["value"] == pytest.approx(0.05)
    assert a["group_universe_hash"] == b["group_universe_hash"]
    # unequal draw lengths -> variance-sum fallback (the designed gate)
    b_short = dict(b, boot_draws=np.asarray([0.24, 0.26]))
    dd2 = f1335._delta(a, b_short)
    assert dd2["ci_method"] == "variance-sum"
    # missing cell fails loud
    with pytest.raises(AssertionError, match="missing full-n cell"):
        f1335._full_n_value(args, "nope__ctx")


# ---------------------------------------------------------------------------
# (8) committed placement n (body-value cross-check)
# ---------------------------------------------------------------------------


def test_committed_placement_n_real_and_drift_guard(tmp_path):
    args = SimpleNamespace(committed_eval_root=COMMITTED_EVAL_ROOT)
    assert f1335._committed_placement_n(args, "base") == 1397
    assert f1335._committed_placement_n(args, "instruct") == 1739
    # drift guard: a committed JSON disagreeing with the body value fails loud
    bad = SimpleNamespace(committed_eval_root=tmp_path)
    (tmp_path / "matched_r7_endpoint__base__Wren__ctx.json").write_text(json.dumps({"n_min": 999}))
    with pytest.raises(AssertionError, match="drift"):
        f1335._committed_placement_n(bad, "base")


# ---------------------------------------------------------------------------
# (9) thin driver env contract + df headroom assert
# ---------------------------------------------------------------------------


def _driver_run(tmp_path, avail_kb: int) -> subprocess.CompletedProcess:
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    stub = scripts / "issue1335_run.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "for v in EPM_I1335_GEN_SEED EPM_I1335_HF_PREFIX DATA_DIR OUT_DIR FIG_DIR \\\n"
        "         I1335_GEN_RUNGS I1335_TF_RUNGS I1335_ALL_RUNGS I1335_MODELS \\\n"
        "         I1335_SUMMARY_MODE I1335_REFERENCE_SUMMARY I1335_SMOKE_ROOT; do\n"
        '  echo "$v=${!v-UNSET}"\n'
        "done\n"
    )
    stub.chmod(0o755)
    driver = scripts / "issue1335_onpolicy_label_run.sh"
    driver.write_text(DRIVER_SH.read_text())
    driver.chmod(0o755)
    fakebin = tmp_path / "bin"
    fakebin.mkdir(exist_ok=True)
    dfstub = fakebin / "df"
    dfstub.write_text(
        "#!/usr/bin/env bash\n"
        'echo "Filesystem 1024-blocks Used Available Capacity Mounted on"\n'
        f'echo "/dev/fake 999999999 1 {avail_kb} 1% /"\n'
    )
    dfstub.chmod(0o755)
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}", "DATA_DIR": str(tmp_path / "d")}
    return subprocess.run(["bash", str(driver)], capture_output=True, text=True, env=env)


def test_driver_env_contract_and_headroom_pass(tmp_path):
    res = _driver_run(tmp_path, avail_kb=200 * 1024 * 1024)  # 200 GB free
    assert res.returncode == 0, res.stderr
    got = dict(line.split("=", 1) for line in res.stdout.strip().splitlines())
    assert got["EPM_I1335_GEN_SEED"] == "45"
    assert got["EPM_I1335_HF_PREFIX"] == "issue1335_ablation_ladder/onpolicy_assistant_label"
    assert got["OUT_DIR"] == "eval_results/issue_1335/onpolicy-assistant-label"
    assert got["FIG_DIR"] == "figures/issue_1335/onpolicy-assistant-label"
    assert got["I1335_GEN_RUNGS"] == "r7_op_assistant r7_op_wren r7_op_wren46"
    assert got["I1335_TF_RUNGS"] == ""
    assert got["I1335_ALL_RUNGS"] == "r7_op_assistant r7_op_wren r7_op_wren46"
    assert got["I1335_MODELS"] == "base instruct"
    assert got["I1335_SUMMARY_MODE"] == "label-compare"
    assert got["I1335_REFERENCE_SUMMARY"] == "eval_results/issue_1335/ladder_summary.json"


def test_driver_headroom_assert_fails_loud(tmp_path):
    res = _driver_run(tmp_path, avail_kb=10 * 1024 * 1024)  # 10 GB free < 75 GB floor
    assert res.returncode == 1
    assert "FATAL" in res.stdout + res.stderr


# ---------------------------------------------------------------------------
# (10) run.sh label-compare mode accepted (executable knob pin)
# ---------------------------------------------------------------------------


def test_run_sh_summary_mode_accepts_label_compare():
    snippet = (
        'I1335_SUMMARY_MODE="label-compare"\n'
        f"source <(sed -n '/^SUMMARY_MODE=/,/^fi$/p' {RUN_SH})\n"
        'echo "MODE=$SUMMARY_MODE|JSON=$SUMMARY_JSON"\n'
    )
    res = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0, res.stderr
    assert "MODE=label-compare|JSON=label_comparison.json" in res.stdout
    bad = subprocess.run(
        ["bash", "-c", snippet.replace("label-compare", "bogus-mode")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert bad.returncode == 5


# ---------------------------------------------------------------------------
# (11) run_lane_pool work-conserving fan-out (executable pin)
# ---------------------------------------------------------------------------


def _pool_harness(tmp_path: Path, lane_body: str, specs: str) -> subprocess.CompletedProcess:
    """Execute ONLY the run_lane_pool function under bash with a stubbed
    run_model_lane + log (never a grep of the script text)."""
    snippet = (
        "set -uo pipefail\n"
        f"LOG_DIR={tmp_path}\nISSUE=1335\nNGPUS=2\nPOOL_POLL_S=1\n"
        'log() { echo "[pool] $*"; }\n'
        f"{lane_body}\n"
        f"source <(sed -n '/^run_lane_pool()/,/^}}$/p' {RUN_SH})\n"
        f"run_lane_pool {tmp_path}/d {tmp_path}/o full {specs}\n"
        'echo "POOL_RC=$?"\n'
    )
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, cwd=REPO_ROOT)


def test_run_lane_pool_work_conserving(tmp_path):
    lane_body = (
        "run_model_lane() {\n"
        '  echo "LANE model=$1 rung=$5 cvd=$CUDA_VISIBLE_DEVICES" >> '
        f"{tmp_path}/lanes.txt\n"
        "  sleep 0.3\n"
        "}\n"
    )
    res = _pool_harness(tmp_path, lane_body, "base:rA instruct:rA base:rB instruct:rB base:rC")
    assert "POOL_RC=0" in res.stdout, res.stdout + res.stderr
    lanes = (tmp_path / "lanes.txt").read_text().strip().splitlines()
    assert len(lanes) == 5  # every lane ran (3 of them on freed GPUs)
    cvds = {line.rsplit("cvd=", 1)[1] for line in lanes}
    assert cvds == {"0", "1"}  # CVD-pinned one lane per GPU
    assert sorted(line.split(" cvd=")[0] for line in lanes) == sorted(
        f"LANE model={m} rung={r}"
        for m, r in [
            ("base", "rA"),
            ("instruct", "rA"),
            ("base", "rB"),
            ("instruct", "rB"),
            ("base", "rC"),
        ]
    )


def test_run_lane_pool_drains_after_failure(tmp_path):
    lane_body = (
        "run_model_lane() {\n"
        '  echo "LANE model=$1 rung=$5" >> ' + f"{tmp_path}/lanes.txt\n"
        '  if [ "$5" = "rBAD" ]; then exit 7; fi\n'
        "  sleep 0.3\n"
        "}\n"
    )
    res = _pool_harness(tmp_path, lane_body, "base:rBAD instruct:rA base:rB instruct:rC")
    assert "POOL_RC=7" in res.stdout, res.stdout + res.stderr  # rc propagated
    lanes = (tmp_path / "lanes.txt").read_text().strip().splitlines()
    assert len(lanes) == 4  # remaining shared-nothing lanes still drained
