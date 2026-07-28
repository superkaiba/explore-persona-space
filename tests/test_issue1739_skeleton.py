"""Round-A skeleton tests for issue #1739 (constants, fit-pool mask, registry, dispatcher)."""

import os
import subprocess
from pathlib import Path

import pytest

from explore_persona_space.experiments.issue_1739 import constants, corpus_registry, store_io

REPO_ROOT = Path(__file__).resolve().parents[1]
DISPATCH = REPO_ROOT / "scripts" / "issue1739_dispatch.sh"


def test_constants_pins():
    assert constants.MODEL_NAME == "Qwen/Qwen2.5-7B-Instruct"
    assert constants.HIDDEN_DIM == 3584
    assert constants.N_LAYERS == 28
    assert constants.SUMMARY_KINDS == ("prefix_end", "context_end", "t1")
    assert constants.JUDGE_MODEL == "claude-sonnet-4-5-20250929"
    assert constants.JUDGE_MAX_TOKENS >= 300  # llm-judging.md rule 23 floor
    assert len(constants.RIDGE_LAMBDAS) == 6
    assert constants.SEEDS == (0, 1, 2)
    assert constants.U_LADDER == (250, 5_000, 50_000)
    assert constants.L_LADDER == (250, 2_500, 16_000)
    assert constants.EVIL_L_CAP == 8_000
    assert constants.STORE_TOTAL_ROWS - constants.STORE_FIT_ROWS == 2_400


def test_fit_pool_mask_excludes_eval_only_rows():
    meta = [
        {"prefix_id": "a", "is_eval_only": False},
        {"prefix_id": "b", "is_eval_only": True},
        {"prefix_id": "c"},  # missing key -> fit row (belt: stratum check below)
        {"prefix_id": "d", "stratum": "battery"},  # battery stratum -> eval-only
    ]
    mask = store_io.fit_pool_mask(meta)
    assert mask.tolist() == [True, False, True, False]


def test_fit_pool_mask_zero_fit_rows_raises():
    with pytest.raises(ValueError, match="zero fit rows"):
        store_io.fit_pool_mask([{"is_eval_only": True}])


def test_fit_pool_mask_full_corpus_count_gate():
    # Full realized manifest: pinned 18,793 fit rows of 21,193 (data-dependent
    # gate exercised BOTH ways — designed pass and designed raise).
    n_eval = constants.STORE_TOTAL_ROWS - constants.STORE_FIT_ROWS
    good = [
        {"is_eval_only": i >= constants.STORE_FIT_ROWS} for i in range(constants.STORE_TOTAL_ROWS)
    ]
    assert int(store_io.fit_pool_mask(good).sum()) == constants.STORE_FIT_ROWS
    bad = [
        {"is_eval_only": i >= constants.STORE_FIT_ROWS - 1}
        for i in range(constants.STORE_TOTAL_ROWS)
    ]
    assert n_eval + 1 == constants.STORE_TOTAL_ROWS - (constants.STORE_FIT_ROWS - 1)
    with pytest.raises(ValueError, match="pinned"):
        store_io.fit_pool_mask(bad)


def test_registry_schema_complete():
    for behavior in corpus_registry.BEHAVIORS:
        for split in corpus_registry.SPLITS:
            spec = corpus_registry.get_spec(behavior, split)
            assert spec.components, (behavior, split)
            for comp in spec.components:
                assert comp.dataset_id and comp.n_rows > 0
    evil_train = corpus_registry.get_spec("evil", "train")
    assert evil_train.cap == constants.EVIL_L_CAP
    roles = {c.role for c in evil_train.components}
    assert roles == {"crossing_prefix", "crossing_question"}
    syc_train = corpus_registry.get_spec("sycophancy", "train")
    assert syc_train.components[0].text_field == "content"
    assert syc_train.components[0].splits == ("relationship_advice", "socialskills")
    hall_train = corpus_registry.get_spec("hallucination", "train")
    assert hall_train.components[0].config == "rc.nocontext"


def test_registry_unknown_key_raises():
    with pytest.raises(KeyError):
        corpus_registry.get_spec("marker", "train")


def test_stage_corpus_arg_validation():
    # Round B rewired the round-A stub to delegate to corpus_staging; the
    # argument-validation contract is unchanged (staging behavior is covered
    # by test_issue1739_dataplane.py against synthetic fixtures).
    with pytest.raises(ValueError):
        corpus_registry.stage_corpus("evil", "train", 0, 0)
    with pytest.raises(TypeError):
        corpus_registry.stage_corpus("evil", "train", None, "0")
    with pytest.raises(KeyError):
        corpus_registry.stage_corpus("evil", "nonsense", None, 0)


def test_gates_1_2_implemented_round_b():
    # Round B replaced the round-A NotImplementedError stubs with report
    # functions (behavioral coverage in test_issue1739_dataplane.py).
    from explore_persona_space.experiments.issue_1739 import gates

    report = gates.gate1_yield_report(
        [{"context_id": "c0", "dv": 50.0}], behavior="sycophancy", n_pilot=1
    )
    assert report["verdict"] in ("PASS", "FAIL")
    report2 = gates.gate2_spread_floor([{"context_id": "c0", "dv": 50.0}], behavior="sycophancy")
    assert report2["verdict"] == "FAIL"  # < 2 DVs: spread undefined


def _run_dispatch(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    # Strip any ambient EPM_I1739_* (a shell that just ran the smoke chain
    # would otherwise flip the dispatcher into SMOKE mode mid-test).
    env = {k: v for k, v in os.environ.items() if not k.startswith("EPM_I1739_")}
    env.update({"OUT_ROOT": str(tmp_path), "REPO_ROOT": str(REPO_ROOT)})
    return subprocess.run(
        ["bash", str(DISPATCH), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_dispatch_help_exits_zero(tmp_path):
    proc = _run_dispatch(["--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "gates extract upload_raw capture judge fits figures results" in proc.stdout


def test_dispatch_unknown_phase_exits_two(tmp_path):
    proc = _run_dispatch(["--phase", "nonsense"], tmp_path)
    assert proc.returncode == 2, (proc.stdout, proc.stderr)


def test_dispatch_fits_phase_fails_loud_without_staged_inputs(tmp_path):
    # Round C1 implemented `fits`: with no staged inputs the phase must FAIL
    # LOUD (round 2: the feature-builder pre-step is the first consumer of
    # the staged corpus and SystemExits before any Hub call). Never a silent
    # ok sentinel; no [phase=done] terminal line on a crash.
    proc = _run_dispatch(["--phase", "fits"], tmp_path)
    assert proc.returncode != 0, (proc.stdout, proc.stderr)
    assert (
        "no staged contexts" in proc.stderr
        or "no contexts jsonl matched" in proc.stderr
        or "no context_end shards" in proc.stderr
        or "FileNotFoundError" in proc.stderr
    ), proc.stderr
    assert not list(tmp_path.glob("issue-1739-epm_progress-fits-*.json"))  # no ok sentinel
    assert "[phase=done]" not in proc.stdout  # the terminal line is reserved for success
