"""Golden tests for the SLURM backend's sbatch renderer.

The renderer is a pure function (no side effects, no filesystem) so the
golden test asserts the exact line shapes the cluster operator (and the
P0/P1/P2 acceptance ladder) needs to see in the rendered script.
These tests run without a cluster.

Why this lives in one file: the golden invariants in the plan are
specific text snippets (``#SBATCH --account=…``, the open-instruct
``accelerate launch ... finetune.py`` line, the secrets ``trap`` …),
and asserting them inline is more readable than splitting across files.

Critical invariant: the full-FT stage MUST target open-instruct
``finetune.py`` / ``dpo_tune_cache.py`` — NEVER the local
``scripts/train_stage_sft.py``. The misroute would silently land a
local TRL SFT + the default ZeRO-2 config, which is what
``run_distributed_pipeline`` does today (P0(d) finding). The test
FAILS on the misroute regardless of zero level.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path as _P

import pytest

from explore_persona_space.backends import (
    ClusterConfig,
    RunSpec,
    SlurmBackend,
    get_cluster_config,
    render_sbatch,
    stages_for_spec,
)
from explore_persona_space.backends.slurm import (
    HEARTBEAT_INTERVAL_SECONDS,
    PREFLIGHT_FAIL_MARKER,
    SbatchPlan,
    Stage,
    build_rsync_command,
    compute_plan_hash,
    default_gpus_for_intent,
    job_name,
    materialize_branch_src,
    parse_job_id,
    render_secrets_env,
    time_budget_hours,
)


def _nibi() -> ClusterConfig:
    return get_cluster_config("nibi")


def _lora_spec(intent: str = "lora-7b") -> RunSpec:
    return RunSpec(
        issue=137,
        intent=intent,
        backend="cluster",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em", "seed=42"),
    )


def _full_ft_spec() -> RunSpec:
    return RunSpec(
        issue=137,
        intent="ft-7b",
        gpus=4,
        backend="cluster",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em",),
        extra={
            "deepspeed_config": "deepspeed/zero2_fp32_comm.json",
            "oi_args_sft": (
                "--model_name_or_path",
                "Qwen/Qwen2.5-7B",
                "--tokenizer_name",
                "Qwen/Qwen2.5-7B",
                "--num_train_epochs",
                "2",
            ),
            "oi_args_dpo": (
                "--model_name_or_path",
                "Qwen/Qwen2.5-7B",
            ),
        },
    )


@pytest.fixture(autouse=True)
def _no_real_marker_posts(monkeypatch):
    """Defense in depth: never let a test shell out to the real
    ``task.py post-marker`` (it would pollute a real tasks/<N>/events.jsonl,
    as happened to #137). Patches the default poster to a no-op; tests that
    assert on posts inject ``marker_poster=`` explicitly.
    """
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.post_marker_via_task_py",
        lambda **_kw: None,
    )


# ---------------------------------------------------------------------------
# Per-cluster config table
# ---------------------------------------------------------------------------


def test_nibi_config_present_and_available() -> None:
    cfg = get_cluster_config("nibi")
    assert cfg.name == "nibi"
    assert cfg.account == "rrg-bengioy-ad_gpu"
    assert cfg.robot_alias == "robot-nibi"
    assert cfg.max_gpus_per_node == 8
    assert cfg.available is True


def test_fir_config_present_but_deferred() -> None:
    """Fir is in the table but flagged ``available=False`` for v1.1."""
    with pytest.raises(RuntimeError, match="available=False"):
        get_cluster_config("fir")


def test_unknown_cluster_raises_loud() -> None:
    with pytest.raises(ValueError, match="unknown cluster"):
        get_cluster_config("rorqual")


# ---------------------------------------------------------------------------
# stages_for_spec — intent → stage table
# ---------------------------------------------------------------------------


def test_lora_intent_produces_train_then_eval() -> None:
    plan = stages_for_spec(_lora_spec("lora-7b"))
    assert [s.name for s in plan.stages] == ["lora", "eval"]
    assert plan.stages[0].script_rel == "scripts/train.py"
    assert plan.stages[1].script_rel == "scripts/eval.py"


def test_full_ft_intent_chains_cpt_sft_dpo_em() -> None:
    plan = stages_for_spec(_full_ft_spec())
    assert [s.name for s in plan.stages] == ["cpt", "sft", "dpo", "em"]
    # Critical invariant: full-FT stages target open-instruct, NOT
    # scripts/train_stage_sft.py (the silent misroute).
    sft = plan.stages[1]
    dpo = plan.stages[2]
    assert sft.backend == "open_instruct"
    assert sft.script_rel == "open_instruct/finetune.py"
    assert dpo.backend == "open_instruct"
    assert dpo.script_rel == "open_instruct/dpo_tune_cache.py"


def test_unknown_intent_raises() -> None:
    spec = RunSpec(issue=1, intent="unknown-intent", backend="cluster", cluster="nibi")
    with pytest.raises(ValueError, match="unsupported intent"):
        stages_for_spec(spec)


# ---------------------------------------------------------------------------
# default_gpus_for_intent / time_budget_hours
# ---------------------------------------------------------------------------


def test_default_gpus_respects_explicit_override() -> None:
    spec = RunSpec(issue=1, intent="ft-7b", gpus=2, backend="cluster", cluster="nibi")
    assert default_gpus_for_intent(spec) == 2


def test_default_gpus_intent_table() -> None:
    assert default_gpus_for_intent(_lora_spec("lora-7b")) == 1
    assert default_gpus_for_intent(_lora_spec("eval")) == 1
    spec_ft = RunSpec(issue=1, intent="ft-7b", backend="cluster", cluster="nibi")
    assert default_gpus_for_intent(spec_ft) == 4


def test_intent_lora_alias_resolves_consistently() -> None:
    """The ``lora`` alias must resolve in ALL three intent dispatchers.

    Regression: ``stages_for_spec`` + ``default_gpus_for_intent`` accept
    ``intent="lora"`` but ``_DEFAULT_TIME_BUDGETS_HOURS`` once omitted it,
    so the new fail-fast ``time_budget_hours`` crashed a valid ``lora``
    caller at render. All three must agree on the alias.
    """
    spec = _lora_spec("lora")
    assert [s.name for s in stages_for_spec(spec).stages] == ["lora", "eval"]
    assert default_gpus_for_intent(spec) == 1
    assert time_budget_hours(spec) == 6.0


def test_time_budget_full_ft_under_24h_per_p0g() -> None:
    """P0(g): a 2-phase 7B full-FT must target the short <24h bin."""
    spec = _full_ft_spec()
    assert time_budget_hours(spec) < 24.0


def test_time_budget_explicit_override_wins() -> None:
    spec = RunSpec(
        issue=1, intent="lora-7b", time_budget_hours=2.5, backend="cluster", cluster="nibi"
    )
    assert time_budget_hours(spec) == 2.5


def test_time_budget_negative_rejected() -> None:
    spec = RunSpec(
        issue=1, intent="lora-7b", time_budget_hours=-1, backend="cluster", cluster="nibi"
    )
    with pytest.raises(ValueError):
        time_budget_hours(spec)


def test_time_budget_unknown_intent_raises_instead_of_silent_default() -> None:
    """Fail-fast: unknown intent must raise (not silently default to 6h)
    so a typo doesn't submit a job under the wrong wall-clock budget.
    Consistent with stages_for_spec which also raises on unknown."""
    spec = RunSpec(issue=1, intent="totally-bogus", backend="cluster", cluster="nibi")
    with pytest.raises(ValueError, match="no default time budget"):
        time_budget_hours(spec)


def test_default_gpus_unknown_intent_raises_instead_of_silent_default() -> None:
    """Fail-fast: unknown intent must raise (not silently default to 1).
    Consistent with stages_for_spec + time_budget_hours."""
    spec = RunSpec(issue=1, intent="totally-bogus", backend="cluster", cluster="nibi")
    with pytest.raises(ValueError, match="no default GPU count"):
        default_gpus_for_intent(spec)


# ---------------------------------------------------------------------------
# job_name + plan-hash
# ---------------------------------------------------------------------------


def test_job_name_keyed_by_issue_and_plan_hash() -> None:
    spec = _lora_spec()
    plain = job_name(spec)
    assert plain == "eps-issue-137"
    hashed = job_name(spec, plan_hash="abcdef1234567890")
    assert hashed.startswith("eps-issue-137-")
    assert "abcdef12" in hashed


def test_compute_plan_hash_is_stable_and_short() -> None:
    h1 = compute_plan_hash("plan body v1")
    h2 = compute_plan_hash(b"plan body v1")
    assert h1 == h2
    assert len(h1) == 8


# ---------------------------------------------------------------------------
# rsync command shape (P0(a) — --mkpath is mandatory)
# ---------------------------------------------------------------------------


def test_rsync_command_includes_mkpath(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    argv = build_rsync_command(
        src_root=tmp_path,
        dest_root="/scratch/tjiral/eps/issue-137",
        robot_alias="robot-nibi",
    )
    assert "--mkpath" in argv  # P0(a): intermediate dirs don't auto-create
    assert "--delete" in argv
    assert "-a" in argv
    assert "--partial" in argv
    # Destination
    assert argv[-1] == "robot-nibi:/scratch/tjiral/eps/issue-137/"


def test_rsync_command_uses_relative_for_external_prefix_preservation(tmp_path) -> None:
    """``--relative`` MUST be in argv so the ``external/`` prefix survives.

    Without it (the prior bug), positional source ``external/open-instruct``
    lands at ``$DST/open-instruct/...`` instead of
    ``$DST/external/open-instruct/...`` — killing every full-FT job at
    line 1 because the renderer emits ``external/open-instruct/<rel>``.
    """
    (tmp_path / "pyproject.toml").write_text("")
    argv = build_rsync_command(
        src_root=tmp_path,
        dest_root="/scratch/tjiral/eps/issue-137",
        robot_alias="robot-nibi",
    )
    assert "--relative" in argv, argv
    # Sources are dot-anchored so --relative preserves the path from
    # the dot, NOT from src_root. Without the dot anchor, --relative
    # would preserve the full ``/tmp/pytest-.../external/open-instruct``
    # path on the cluster, also wrong.
    assert "./external/open-instruct" in argv, argv
    assert "./configs" in argv, argv
    assert "./pyproject.toml" in argv, argv
    # configs/deepspeed + configs/tulu were redundant subsets of configs
    # and are no longer in the include list (a subset would be
    # double-copied under --relative).
    assert "./configs/deepspeed" not in argv, argv
    assert "./configs/tulu" not in argv, argv


def test_rsync_round_trip_preserves_external_prefix(tmp_path) -> None:
    """Load-bearing: run REAL rsync (local->local) and assert the
    destination layout matches what the renderer's full-FT path
    actually targets.

    This is the test that would have caught the original Blocker 1.
    The assertion is on the on-disk destination tree, not on argv —
    so a future change to flag set / source paths that re-introduces
    the flatten regression still fails here.
    """
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    dst_root.mkdir()

    # Mirror the leaves the renderer actually launches.
    (src_root / "pyproject.toml").write_text("")
    (src_root / "uv.lock").write_text("")
    (src_root / "external" / "open-instruct" / "open_instruct").mkdir(parents=True)
    (src_root / "external" / "open-instruct" / "open_instruct" / "finetune.py").write_text("f")
    (src_root / "external" / "open-instruct" / "open_instruct" / "dpo_tune_cache.py").write_text(
        "d"
    )
    (src_root / "configs" / "deepspeed").mkdir(parents=True)
    (src_root / "configs" / "deepspeed" / "zero2_fp32_comm.json").write_text("{}")
    (src_root / "configs" / "tulu").mkdir(parents=True)
    (src_root / "configs" / "tulu" / "sft_qwen7b.yaml").write_text("a: 1")
    (src_root / "scripts").mkdir()
    (src_root / "scripts" / "train.py").write_text("p")
    (src_root / "src" / "explore_persona_space").mkdir(parents=True)
    (src_root / "src" / "explore_persona_space" / "__init__.py").write_text("")
    (src_root / "tests").mkdir()
    # data/sft carries the committed training-mix JSONLs (live attempt-4
    # finding: the rsync lane missed the smoke dataset entirely).
    (src_root / "data" / "sft").mkdir(parents=True)
    (src_root / "data" / "sft" / "router_smoke_sft.jsonl").write_text("{}\n")

    # Run the REAL rsync, local->local (no robot alias — just plain
    # filesystem dest). build_rsync_command's last arg is
    # ``<robot_alias>:<dest_root>/``; we override it with a local path.
    argv = build_rsync_command(
        src_root=src_root,
        dest_root=str(dst_root),
        robot_alias="robot-nibi",
    )
    argv[-1] = str(dst_root) + "/"
    # Real rsync, real --relative, cwd=src_root so the ``./``-anchored
    # sources resolve correctly. ``check=True`` so a non-zero exit
    # fails the test.
    subprocess.run(argv, check=True, cwd=str(src_root), timeout=30)

    # The renderer (render_sbatch) emits these as launch targets:
    #   external/open-instruct/open_instruct/finetune.py
    #   external/open-instruct/open_instruct/dpo_tune_cache.py
    #   configs/deepspeed/zero2_fp32_comm.json (deepspeed_config arg)
    # All three MUST resolve under dst_root after rsync; if any are
    # missing, the cluster job dies at line 1.
    assert (dst_root / "external" / "open-instruct" / "open_instruct" / "finetune.py").exists()
    assert (
        dst_root / "external" / "open-instruct" / "open_instruct" / "dpo_tune_cache.py"
    ).exists()
    assert (dst_root / "configs" / "deepspeed" / "zero2_fp32_comm.json").exists()
    assert (dst_root / "configs" / "tulu" / "sft_qwen7b.yaml").exists()
    assert (dst_root / "pyproject.toml").exists()

    # The specific regression: ``external/`` prefix MUST be preserved
    # (the bug landed `external/open-instruct/...` at
    # `dst/open-instruct/...`, dropping the `external/` segment).
    assert not (dst_root / "open-instruct").exists(), (
        "Regression: external/ prefix dropped — full-FT launch target "
        "external/open-instruct/<rel> would resolve to a missing path."
    )


def test_rsync_command_requires_pyproject_in_src(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="pyproject"):
        build_rsync_command(
            src_root=tmp_path,
            dest_root="/scratch/foo",
            robot_alias="robot-nibi",
        )


# ---------------------------------------------------------------------------
# secrets file rendering
# ---------------------------------------------------------------------------


def test_render_secrets_env_emits_present_keys_only() -> None:
    out = render_secrets_env({"HF_TOKEN": "abc", "WANDB_API_KEY": "xyz"})
    assert "HF_TOKEN=abc" in out
    assert "WANDB_API_KEY=xyz" in out
    # No `export` (set -a auto-exports inside the sbatch).
    assert "export " not in out


def test_render_secrets_env_shell_quotes_special_chars() -> None:
    out = render_secrets_env({"HF_TOKEN": "tok with space"})
    # shlex.quote wraps in single quotes when special chars are present.
    assert "HF_TOKEN='tok with space'" in out


def test_render_secrets_env_skips_empty_values() -> None:
    out = render_secrets_env({"HF_TOKEN": "", "WANDB_API_KEY": "real"})
    assert "HF_TOKEN" not in out
    assert "WANDB_API_KEY=real" in out


def test_passthrough_env_keys_include_hf_storage_knobs() -> None:
    """#564 (test 21d): the HF-storage soft-ceiling / overflow-routing knobs
    must ride the sourced env file to the compute node, or a dispatch-process
    opt-in silently no-ops remotely (the #535-r7 trap). The VM-local cache
    path + event-sink path are deliberately NOT threaded (wrong machine)."""
    from explore_persona_space.backends.slurm import PASSTHROUGH_ENV_KEYS

    for key in (
        "EPM_HF_STORAGE_SOFT_CEILING_TB",
        "EPM_HF_OVERFLOW_ROUTING",
        "EPM_HF_STORAGE_CHECK",
        "EPM_HF_STORAGE_CACHE_TTL_S",
    ):
        assert key in PASSTHROUGH_ENV_KEYS, key
    assert "EPM_HF_STORAGE_CACHE_PATH" not in PASSTHROUGH_ENV_KEYS
    assert "EPM_HF_OVERFLOW_EVENT_PATH" not in PASSTHROUGH_ENV_KEYS


def test_render_secrets_env_includes_hf_storage_knobs() -> None:
    """#564 (test 21d): the four storage knobs render into the sourced env file."""
    out = render_secrets_env(
        {
            "EPM_HF_STORAGE_SOFT_CEILING_TB": "10.0",
            "EPM_HF_OVERFLOW_ROUTING": "1",
            "EPM_HF_STORAGE_CHECK": "0",
            "EPM_HF_STORAGE_CACHE_TTL_S": "3600",
        }
    )
    assert "EPM_HF_STORAGE_SOFT_CEILING_TB=10.0" in out
    assert "EPM_HF_OVERFLOW_ROUTING=1" in out
    assert "EPM_HF_STORAGE_CHECK=0" in out
    assert "EPM_HF_STORAGE_CACHE_TTL_S=3600" in out


def test_render_secrets_env_includes_persist_adapter_passthrough() -> None:
    """M2 regression: the non-secret adapter-persist targets MUST ride
    the sourced env file to the compute node, or
    ``trainer.py:_persist_adapter`` no-ops remotely and the acceptance
    harness's check (a) false-FAILs AFTER the compute was spent."""
    out = render_secrets_env(
        {
            "HF_TOKEN": "abc",
            "EPM_PERSIST_ADAPTER_HF_REPO": "superkaiba1/explore-persona-space",
            "EPM_PERSIST_ADAPTER_SUBFOLDER": "router_acceptance/issue-9-nibi",
        }
    )
    assert "EPM_PERSIST_ADAPTER_HF_REPO=superkaiba1/explore-persona-space" in out
    assert "EPM_PERSIST_ADAPTER_SUBFOLDER=router_acceptance/issue-9-nibi" in out
    # Secrets still render alongside; the two lists are additive.
    assert "HF_TOKEN=abc" in out


def test_render_secrets_env_loads_project_dotenv(monkeypatch) -> None:
    """render_secrets_env(None) must load the repo ``.env`` before snapshotting
    ``os.environ``.

    Regression: secrets live in ``.env`` (loaded via dotenv at runtime), not
    the ambient shell, so a bare ``os.environ`` snapshot is empty and the
    cluster gets a 0-key ``secrets.env`` whose in-job preflight FAILs on the
    ``${HF_TOKEN:?}`` guard (caught live on Nibi during acceptance).
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def fake_load_dotenv(*_a, **_k):
        os.environ["HF_TOKEN"] = "hf_from_dotenv"

    monkeypatch.setattr("explore_persona_space.orchestrate.env.load_dotenv", fake_load_dotenv)
    out = render_secrets_env()  # env=None → must call load_dotenv first
    assert "HF_TOKEN=hf_from_dotenv" in out


# ---------------------------------------------------------------------------
# Sbatch render — golden assertions for the LoRA + eval path
# ---------------------------------------------------------------------------


def test_render_sbatch_lora_eval_golden() -> None:
    spec = _lora_spec("lora-7b")
    cluster = _nibi()
    plan = stages_for_spec(spec)
    script = render_sbatch(
        spec=spec,
        cluster=cluster,
        plan=plan,
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )

    # Headers
    assert "#SBATCH --account=rrg-bengioy-ad_gpu" in script
    assert "#SBATCH --gpus-per-node=h100:1" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --output=/scratch/tjiral/eps/issue-137/job.out" in script
    assert re.search(r"#SBATCH --time=\d{2}:\d{2}:\d{2}", script)
    assert "#SBATCH --job-name=eps-issue-137" in script

    # `module load cuda` MUST be on its own line — never piped (P0(c)).
    cuda_lines = [
        line for line in script.splitlines() if line.strip().startswith("module load cuda")
    ]
    assert cuda_lines, "module load cuda missing"
    for line in cuda_lines:
        assert "|" not in line, f"module load piped — env loss bug (P0(c)): {line!r}"

    # CUDA_HOME bridge
    assert "CUDA_HOME=$EBROOTCUDA" in script or "CUDA_HOME=$CUDACORE_HOME" in script

    # uv cache + venv cache (purge-safe sentinel + flock)
    assert "UV_CACHE_DIR=" in script
    assert "$SCRATCH/eps/venv-" in script
    assert ".complete" in script  # sentinel
    assert "flock" in script  # concurrent-build guard

    # Secrets stanza (umask + chmod 600 + set -a/+a + trap shred)
    assert "umask 077" in script
    assert "chmod 600" in script
    assert "set -a" in script
    assert "set +a" in script
    assert "trap " in script
    assert "shred -u" in script
    # set +x around the source so a bash -x rerun doesn't leak tokens.
    assert "set +x" in script
    assert "set -x" in script

    # Reachability + GPU + tmpdir preflight (FAIL-FAST before heavy work)
    assert "preflight" in script
    assert PREFLIGHT_FAIL_MARKER in script
    assert "SLURM_TMPDIR" in script
    assert "SLURM_GPUS_ON_NODE" in script  # derive process count from this
    # The Hub/WandB reachability check is reused from preflight.check_connectivity
    # (invoked via the preflight module).
    assert "explore_persona_space.orchestrate.preflight" in script

    # No /workspace anywhere (cluster path must not leak the RunPod path).
    assert "/workspace" not in script

    # Heartbeat loop + status.json + every stage emits [phase=…]
    assert "[phase=lora]" in script
    assert "[phase=eval]" in script
    assert "[phase=done]" in script
    assert "_write_status" in script
    assert "status.json" in script
    assert f"HEARTBEAT_INTERVAL={HEARTBEAT_INTERVAL_SECONDS}" in script

    # The Hydra args got threaded into the train + eval invocations.
    assert "condition=c1_evil_wrong_em" in script
    assert "seed=42" in script

    # The WANDB_PROJECT default is workload_cmd-lane-only (#601 follow-up
    # r1) — local/hydra stages set the project via Hydra config.
    assert "WANDB_PROJECT" not in script


def test_render_sbatch_secret_expansions_sit_outside_xtrace() -> None:
    """Round-6 C1: every line that EXPANDS a secret value must execute
    with xtrace OFF.

    Under ``set -x`` the ``: "${HF_TOKEN:?…}"`` preflight guard traces
    the EXPANDED value (``+ : hf_…``) into job.out; the monitor's log
    tails then carry the real token into git-committed markers — the
    issue-535 live run leaked both HF_TOKEN and WANDB_API_KEY this way.
    Walks the rendered script tracking xtrace state line by line and
    asserts the secrets source + both token checks land in OFF windows
    (and that an ON window exists afterwards, i.e. the wrap is a
    window, not a global xtrace disable)."""
    spec = _lora_spec("lora-7b")
    script = render_sbatch(
        spec=spec,
        cluster=_nibi(),
        plan=stages_for_spec(spec),
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )

    secret_bearing_markers = (
        'source "$SECRETS_FILE"',
        "${HF_TOKEN:?",
        "${WANDB_API_KEY:?",
    )
    xtrace_on = False  # bash starts with xtrace off
    seen: dict[str, bool] = {}
    ever_on_after_checks = False
    for line in script.splitlines():
        stripped = line.strip()
        if stripped == "set -x":
            xtrace_on = True
        elif stripped == "set +x":
            xtrace_on = False
        for marker in secret_bearing_markers:
            if marker in line:
                assert not xtrace_on, f"secret-bearing line under xtrace: {line!r}"
                seen[marker] = True
        if xtrace_on and len(seen) == len(secret_bearing_markers):
            ever_on_after_checks = True
    assert set(seen) == set(secret_bearing_markers), (
        f"expected all secret-bearing lines in the render; saw {sorted(seen)}"
    )
    assert ever_on_after_checks, (
        "xtrace should be re-enabled after the token checks (debuggability "
        "window), not globally disabled"
    )


def test_heartbeat_starts_early_and_reports_live_phase() -> None:
    """Heartbeat must start BEFORE the venv build (else a job reads `stalled`
    for the whole ~6-40 min build) and report the LIVE phase from a file (a bg
    subshell freezes a captured shell var → would report `startup` through every
    stage). Both caught on real Nibi during acceptance."""
    spec = _lora_spec("lora-7b")
    script = render_sbatch(
        spec=spec,
        cluster=_nibi(),
        plan=stages_for_spec(spec),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        plan_hash="h",
    )
    # Started at startup, before the uv venv build.
    assert script.index("_heartbeat_loop &") < script.index("uv sync"), (
        "heartbeat must start before the venv build"
    )
    # Reads the live phase file, NOT a captured shell var.
    assert 'cat "$PHASE_FILE"' in script
    assert '_write_status "${CURRENT_PHASE' not in script
    # Stages write the live phase to the file.
    assert 'echo "lora" > "$PHASE_FILE"' in script


# ---------------------------------------------------------------------------
# Sbatch render — the full-FT GOLDEN INVARIANT (the highest-risk plan item)
# ---------------------------------------------------------------------------


def test_render_sbatch_full_ft_targets_open_instruct_not_train_stage_sft() -> None:
    """The full-FT stage MUST go through open-instruct, NOT the local
    train_stage_sft.py misroute (P0(d) golden invariant)."""
    spec = _full_ft_spec()
    cluster = _nibi()
    plan = stages_for_spec(spec)
    script = render_sbatch(
        spec=spec,
        cluster=cluster,
        plan=plan,
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )

    # The full-FT SFT stage MUST be open-instruct's finetune.py
    assert "external/open-instruct/open_instruct/finetune.py" in script
    # The full-FT DPO stage MUST be open-instruct's dpo_tune_cache.py
    assert "external/open-instruct/open_instruct/dpo_tune_cache.py" in script
    # Critical: the misroute target MUST NOT appear in a full-FT script.
    assert "train_stage_sft.py" not in script, (
        "Full-FT sbatch must target open-instruct's finetune.py, NOT "
        "the local TRL train_stage_sft.py (P0(d) misroute)."
    )


def test_render_sbatch_full_ft_uses_accelerate_with_deepspeed() -> None:
    spec = _full_ft_spec()
    cluster = _nibi()
    plan = stages_for_spec(spec)
    script = render_sbatch(
        spec=spec,
        cluster=cluster,
        plan=plan,
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )

    # accelerate launch with --mixed_precision bf16 --use_deepspeed
    assert "accelerate launch" in script
    assert "--mixed_precision bf16" in script
    assert "--use_deepspeed" in script
    # --deepspeed_config_file points at the config under the synced configs/.
    assert "--deepspeed_config_file configs/deepspeed/zero2_fp32_comm.json" in script
    # Single-node ⇒ NO srun (multi-node srun forbidden by the wrapper).
    assert "srun" not in script
    # num_processes derived from $SLURM_GPUS_ON_NODE (NOT a stale nvidia-smi).
    assert "--num_processes $SLURM_GPUS_ON_NODE" in script
    assert "--num_machines 1" in script
    assert "--machine_rank 0" in script
    # SFT-specific user args threaded through.
    assert "Qwen/Qwen2.5-7B" in script


def test_render_sbatch_full_ft_time_budget_short_bin() -> None:
    """P0(g): full-FT --time must fit the short <24h bin."""
    spec = _full_ft_spec()
    cluster = _nibi()
    plan = stages_for_spec(spec)
    script = render_sbatch(
        spec=spec,
        cluster=cluster,
        plan=plan,
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )
    m = re.search(r"#SBATCH --time=(\d{2}):(\d{2}):(\d{2})", script)
    assert m
    hours = int(m.group(1)) + int(m.group(2)) / 60 + int(m.group(3)) / 3600
    assert hours < 24.0, f"full-FT --time should fit the short bin (<24h), got {hours}h"


def test_render_sbatch_enforces_per_cluster_gpu_cap() -> None:
    spec = RunSpec(issue=1, intent="ft-7b", gpus=9, backend="cluster", cluster="nibi")
    cluster = _nibi()  # nibi cap = 8
    plan = stages_for_spec(spec)
    with pytest.raises(ValueError, match="max_gpus_per_node"):
        render_sbatch(
            spec=spec,
            cluster=cluster,
            plan=plan,
            scratch_dir="/scratch/tjiral/eps/issue-1",
        )


def test_render_sbatch_includes_job_name_plan_hash() -> None:
    spec = _lora_spec()
    cluster = _nibi()
    plan = stages_for_spec(spec)
    script = render_sbatch(
        spec=spec,
        cluster=cluster,
        plan=plan,
        scratch_dir="/scratch/tjiral/eps/issue-137",
        plan_hash="deadbeef" * 8,
    )
    assert "#SBATCH --job-name=eps-issue-137-deadbeef" in script


# ---------------------------------------------------------------------------
# sbatch stdout parsing — P0 sbatch-NOTE pollution defense
# ---------------------------------------------------------------------------


def test_parse_job_id_picks_id_after_memory_note() -> None:
    """P0 finding: sbatch's memory NOTE includes digits that pollute a
    naïve ``grep -oE '[0-9]+' | tail -1``. We must match the literal
    'Submitted batch job <N>' prefix."""
    stdout = (
        "sbatch: NOTE: Your memory allocation 480000 may be wasteful;\n"
        "sbatch: NOTE: consider reducing to 64G per task.\n"
        "Submitted batch job 99887766\n"
    )
    assert parse_job_id(stdout) == "99887766"


def test_parse_job_id_raises_on_miss() -> None:
    with pytest.raises(RuntimeError, match="Submitted batch job"):
        parse_job_id("sbatch: error: invalid account\n")


# ---------------------------------------------------------------------------
# SlurmBackend.launch — submit calls the injected submitter with the rendered script
# ---------------------------------------------------------------------------


def test_slurm_backend_launch_submits_rendered_script(tmp_path) -> None:
    """End-to-end: launch() calls the injected submitter once with a
    rendered sbatch and returns a typed handle.

    Uses dependency injection (the ``submitter`` / ``rsyncer`` /
    ``secrets_pusher`` / ``marker_poster`` ctor seams) so the test runs
    without any network / cluster AND without polluting a real task's
    events.jsonl.
    """
    (tmp_path / "pyproject.toml").write_text("")

    submitted: list[tuple[str, str]] = []

    def fake_submit(*, robot_alias, sbatch_script):
        submitted.append((robot_alias, sbatch_script))
        return "9001"

    rsynced: list[tuple[str, str, str]] = []

    def fake_rsync(*, src_root, dest_root, robot_alias):
        rsynced.append((str(src_root), dest_root, robot_alias))

    posted: list[dict] = []

    def fake_post_marker(**kwargs):
        posted.append(kwargs)

    backend = SlurmBackend(
        src_root=tmp_path,
        submitter=fake_submit,
        rsyncer=fake_rsync,
        marker_poster=fake_post_marker,
    )
    spec = _lora_spec()
    handle = backend.launch(spec)

    assert handle.backend == "cluster"
    assert handle.cluster == "nibi"
    assert handle.job_id == "9001"
    assert handle.pod_name == "eps-issue-137"
    assert handle.scratch_dir == "/scratch/tjiral/eps/issue-137"
    assert handle.log_path == "/scratch/tjiral/eps/issue-137/job.out"
    assert handle.extra["account"] == "rrg-bengioy-ad_gpu"
    assert handle.extra["robot_alias"] == "robot-nibi"
    assert handle.extra["gpus_per_node"] == 1
    # The poll path reads issue out of handle.extra, so launch must
    # populate it.
    assert handle.extra["issue"] == 137
    # The monitor's artifact-freshness gate + the started-evidence probe
    # read the submit timestamp off the handle (rides the sidecar JSON
    # across processes) — round-6 C2.
    assert isinstance(handle.extra["submitted_at"], float)
    assert handle.extra["submitted_at"] > 0

    # Submit was called once with a real rendered sbatch.
    assert len(submitted) == 1
    alias, script = submitted[0]
    assert alias == "robot-nibi"
    assert "#SBATCH --account=rrg-bengioy-ad_gpu" in script
    assert "[phase=done]" in script

    # epm:cluster-launched v1 was posted exactly once with the right body.
    assert len(posted) == 1, posted
    assert posted[0]["marker"] == "epm:cluster-launched"
    assert posted[0]["version"] == 1
    assert posted[0]["issue"] == 137
    body = __import__("json").loads(posted[0]["note"])
    assert body["job_id"] == "9001"
    assert body["job_name"] == "eps-issue-137"
    assert body["scratch_dir"] == "/scratch/tjiral/eps/issue-137"
    assert body["log_path"] == "/scratch/tjiral/eps/issue-137/job.out"
    assert body["cluster"] == "nibi"
    assert body["gpus"] == 1


def test_slurm_backend_launch_survives_marker_post_failure(tmp_path) -> None:
    """C1 regression: a marker-post failure AFTER a successful sbatch
    submit must NOT propagate out of ``launch()``.

    ``post_marker_via_task_py`` is ``subprocess.run(check=True,
    timeout=30)`` -- flock contention on ``~/.task-workflow/lock`` is a
    realistic 30s ``TimeoutExpired`` on this multi-session VM. Pre-fix,
    that raise escaped ``launch()`` AFTER the job was live: no handle,
    no lease, no sidecar -- an orphaned SLURM job with rc=4 at the
    dispatch CLI. The marker is observability, not control flow."""
    (tmp_path / "pyproject.toml").write_text("")

    submitted: list[tuple[str, str]] = []

    def fake_submit(*, robot_alias, sbatch_script):
        submitted.append((robot_alias, sbatch_script))
        return "9002"

    def raising_post_marker(**_kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["task.py", "post-marker"])

    backend = SlurmBackend(
        src_root=tmp_path,
        submitter=fake_submit,
        rsyncer=lambda **_kw: None,
        marker_poster=raising_post_marker,
    )
    handle = backend.launch(_lora_spec())

    # The job was submitted exactly once and the handle came back whole.
    assert len(submitted) == 1
    assert handle.job_id == "9002"
    assert handle.pod_name == "eps-issue-137"
    assert handle.extra["issue"] == 137


def test_slurm_prepare_honors_feature_branch_on_stale_main_source_via_cloner(tmp_path) -> None:
    """#793: SLURM ``prepare`` HONORS a non-``main`` ``repo_branch`` on a stale-``main``
    rsync source by MATERIALIZING the branch tree via the ``git_cloner`` seam.

    This inverts the pre-#793 refuse-only behavior (#653): the lane no longer has to
    refuse when the repo-root install is on ``main``. Instead, ``_resolve_rsync_source``
    invokes ``materialize_branch_src`` (the ``git_cloner`` default) to build a complete
    branch tree on the VM, and rsync sources FROM that tree — so the feature-branch code
    actually runs, no ``ValueError``.
    """
    (tmp_path / "pyproject.toml").write_text("")
    scratch = tmp_path / "scratch-branch"
    scratch.mkdir()

    cloner_calls: list[dict] = []

    def fake_cloner(*, src_root, branch, issue):
        cloner_calls.append({"src_root": src_root, "branch": branch, "issue": issue})
        return scratch

    rsynced: list[dict] = []

    backend = SlurmBackend(
        src_root=tmp_path,
        rsyncer=lambda **kw: rsynced.append(kw),
        secrets_pusher=lambda **_kw: None,
        runtime_clearer=lambda **_kw: None,
        # The rsync source is on ``main`` (the repo-root install default) AND the
        # materialized scratch tree is on the requested branch (so the re-asserted
        # guard trivially passes against the RESOLVED source).
        git_branch_resolver=lambda root: "issue-653" if root == scratch else "main",
        git_cloner=fake_cloner,
    )
    spec = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="cluster",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em",),
        extra={"repo_branch": "issue-653"},
    )

    backend.prepare(spec)

    # The cloner was INVOKED exactly once with the requested branch + spec issue.
    assert len(cloner_calls) == 1, cloner_calls
    assert cloner_calls[0]["branch"] == "issue-653"
    assert cloner_calls[0]["issue"] == 137
    assert cloner_calls[0]["src_root"] == tmp_path
    # rsync now RUNS (the OLD ``rsynced == []`` assertion is inverted) and sources FROM
    # the materialized branch tree, NOT the default ``_src_root``.
    assert len(rsynced) == 1, rsynced
    assert rsynced[0]["src_root"] == scratch


def test_slurm_prepare_allows_matching_feature_branch_source(tmp_path) -> None:
    """#653: when the rsync source's HEAD MATCHES the requested feature branch,
    ``prepare`` proceeds normally (the worktree already carries the code)."""
    (tmp_path / "pyproject.toml").write_text("")

    rsynced: list[tuple] = []

    backend = SlurmBackend(
        src_root=tmp_path,
        rsyncer=lambda **kw: rsynced.append(kw),
        secrets_pusher=lambda **_kw: None,
        runtime_clearer=lambda **_kw: None,
        git_branch_resolver=lambda _root: "issue-653",
    )
    spec = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="cluster",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em",),
        extra={"repo_branch": "issue-653"},
    )

    backend.prepare(spec)
    assert len(rsynced) == 1


@pytest.mark.parametrize("extra", [{}, {"repo_branch": "main"}], ids=["absent", "main"])
def test_slurm_prepare_main_branch_run_is_unaffected(tmp_path, extra) -> None:
    """#653: the guard is a no-op for a ``main`` run (absent or 'main'
    ``repo_branch``) — the rsync source IS ``main`` by the resolver's design,
    and the branch resolver is never even consulted."""
    (tmp_path / "pyproject.toml").write_text("")

    def _resolver_must_not_run(_root):
        raise AssertionError("resolver must not be consulted for a main run")

    rsynced: list[tuple] = []
    backend = SlurmBackend(
        src_root=tmp_path,
        rsyncer=lambda **kw: rsynced.append(kw),
        secrets_pusher=lambda **_kw: None,
        runtime_clearer=lambda **_kw: None,
        git_branch_resolver=_resolver_must_not_run,
    )
    spec = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="cluster",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em",),
        extra=extra,
    )
    backend.prepare(spec)
    assert len(rsynced) == 1


def test_slurm_prepare_raises_when_cloner_cannot_resolve_branch(tmp_path) -> None:
    """#793: a feature-branch request whose source branch is UNPROVABLE (resolver
    returns ``None`` — detached HEAD / non-repo source) routes to the ``git_cloner``,
    and if the cloner cannot resolve the branch it raises ``RuntimeError``, which
    surfaces from ``prepare()``.

    This preserves the auto-chain fallback: ``router._prepare_and_launch`` wraps the
    ``RuntimeError`` as ``BackendPrepareError`` exactly as it did the pre-#793 guard's
    ``ValueError`` (§4d), so the lane still advances. The raise fires BEFORE the rsync,
    so no stale code is shipped.
    """
    (tmp_path / "pyproject.toml").write_text("")

    def raising_cloner(*, src_root, branch, issue):
        raise RuntimeError(f"cannot resolve branch {branch}")

    rsynced: list[dict] = []
    backend = SlurmBackend(
        src_root=tmp_path,
        rsyncer=lambda **kw: rsynced.append(kw),
        secrets_pusher=lambda **_kw: None,
        runtime_clearer=lambda **_kw: None,
        git_branch_resolver=lambda _root: None,
        git_cloner=raising_cloner,
    )
    spec = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="cluster",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em",),
        extra={"repo_branch": "issue-653"},
    )

    with pytest.raises(RuntimeError, match="cannot resolve branch issue-653"):
        backend.prepare(spec)
    assert rsynced == []


# ---------------------------------------------------------------------------
# #793: SLURM lane HONORS repo_branch via the git_cloner seam + overlay
# ---------------------------------------------------------------------------


def _branch_spec(branch: str | None) -> RunSpec:
    """A lora-7b RunSpec carrying (or omitting) a ``repo_branch`` in ``extra``."""
    extra = {"repo_branch": branch} if branch is not None else {}
    return RunSpec(
        issue=793,
        intent="lora-7b",
        backend="cluster",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em",),
        extra=extra,
    )


def test_prepare_honors_repo_branch_via_git_cloner(tmp_path) -> None:
    """#793: a non-``main`` ``repo_branch`` against a ``main`` install invokes the
    ``git_cloner`` once and rsyncs from its returned scratch tree, no ``ValueError``."""
    (tmp_path / "pyproject.toml").write_text("")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    cloner_calls: list[dict] = []

    def fake_cloner(*, src_root, branch, issue):
        cloner_calls.append({"src_root": src_root, "branch": branch, "issue": issue})
        return scratch

    rsynced: list[dict] = []

    backend = SlurmBackend(
        src_root=tmp_path,
        rsyncer=lambda **kw: rsynced.append(kw),
        secrets_pusher=lambda **_kw: None,
        runtime_clearer=lambda **_kw: None,
        # main install; the materialized scratch is on the requested branch.
        git_branch_resolver=lambda root: "issue-793" if root == scratch else "main",
        git_cloner=fake_cloner,
    )

    backend.prepare(_branch_spec("issue-793"))

    assert len(cloner_calls) == 1, cloner_calls
    assert cloner_calls[0]["branch"] == "issue-793"
    assert cloner_calls[0]["issue"] == 793
    assert len(rsynced) == 1, rsynced
    assert rsynced[0]["src_root"] == scratch


@pytest.mark.parametrize("branch", [None, "main"], ids=["absent", "main"])
def test_prepare_repo_branch_main_is_byte_identical_no_cloner(tmp_path, branch) -> None:
    """#793: an absent / ``main`` ``repo_branch`` NEVER touches the cloner and rsyncs
    from ``backend._src_root`` — byte-identical to the pre-fix path (zero side effects)."""
    (tmp_path / "pyproject.toml").write_text("")

    def cloner_must_not_run(**_kw):
        raise AssertionError("git_cloner must not be called for a main/absent run")

    rsynced: list[dict] = []
    backend = SlurmBackend(
        src_root=tmp_path,
        rsyncer=lambda **kw: rsynced.append(kw),
        secrets_pusher=lambda **_kw: None,
        runtime_clearer=lambda **_kw: None,
        git_cloner=cloner_must_not_run,
    )

    backend.prepare(_branch_spec(branch))

    assert len(rsynced) == 1, rsynced
    assert rsynced[0]["src_root"] == backend._src_root


def test_prepare_install_already_on_branch_skips_cloner(tmp_path) -> None:
    """#793: when the repo-root install is ALREADY on the requested branch, the cloner
    is NOT called and rsync uses ``_src_root`` directly (the install IS the branch)."""
    (tmp_path / "pyproject.toml").write_text("")

    def cloner_must_not_run(**_kw):
        raise AssertionError("git_cloner must not be called when install is on the branch")

    rsynced: list[dict] = []
    backend = SlurmBackend(
        src_root=tmp_path,
        rsyncer=lambda **kw: rsynced.append(kw),
        secrets_pusher=lambda **_kw: None,
        runtime_clearer=lambda **_kw: None,
        git_branch_resolver=lambda _root: "issue-793",
        git_cloner=cloner_must_not_run,
    )

    backend.prepare(_branch_spec("issue-793"))

    assert len(rsynced) == 1, rsynced
    assert rsynced[0]["src_root"] == backend._src_root


def test_resolve_rsync_source_returns_src_root_on_main(tmp_path) -> None:
    """#793: unit-test ``_resolve_rsync_source`` in isolation across the branch cases.

    Parametrized over: absent → ``_src_root``; ``main`` → ``_src_root``; already-on-branch
    (resolver returns the request) → ``_src_root``; non-``main`` mismatch (resolver returns
    ``main``) → cloner's path; non-``main`` with resolver returning ``None`` (detached /
    non-repo — the Statistics-critic case, mirroring the guard's ``actual is None``
    sub-branch) → cloner's path. Asserts on the returned Path, not on a raise.
    """
    (tmp_path / "pyproject.toml").write_text("")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def _resolve(*, branch, resolver):
        backend = SlurmBackend(
            src_root=tmp_path,
            rsyncer=lambda **_kw: None,
            secrets_pusher=lambda **_kw: None,
            runtime_clearer=lambda **_kw: None,
            git_branch_resolver=resolver,
            git_cloner=lambda *, src_root, branch, issue: scratch,
        )
        return backend._resolve_rsync_source(_branch_spec(branch))

    # A resolver that would fail the test loudly if consulted on the early-return path.
    def _resolver_forbidden(_root):
        raise AssertionError("resolver must not be consulted for a main/absent run")

    # absent → _src_root (resolver never consulted)
    assert _resolve(branch=None, resolver=_resolver_forbidden) == tmp_path
    # "main" → _src_root (resolver never consulted)
    assert _resolve(branch="main", resolver=_resolver_forbidden) == tmp_path
    # already-on-branch → _src_root
    assert _resolve(branch="issue-793", resolver=lambda _r: "issue-793") == tmp_path
    # non-main mismatch (resolver returns "main") → cloner's path
    assert _resolve(branch="issue-793", resolver=lambda _r: "main") == scratch
    # non-main with resolver returning None (detached / non-repo) → cloner's path
    assert _resolve(branch="issue-793", resolver=lambda _r: None) == scratch


def _git_available() -> bool:
    import shutil as _shutil

    return _shutil.which("git") is not None


def _init_branch_repo(repo: _P) -> None:
    """Init a tiny git repo at ``repo`` with a ``main`` commit + an ``issue-X`` branch.

    The ``issue-X`` branch adds a marker file (``branch_marker.txt``); a working-tree-only
    ``external/open-instruct/open_instruct/finetune.py`` is created UNTRACKED to simulate
    the mode-160000 gitlink the overlay handles.
    """
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@e",
    }

    def _git(*args):
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

    repo.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    (repo / "src").mkdir()
    (repo / "src" / "mod.py").write_text("x = 1\n")
    _git("add", "pyproject.toml", "src/mod.py")
    _git("commit", "-q", "-m", "main commit")
    _git("checkout", "-q", "-b", "issue-X")
    (repo / "branch_marker.txt").write_text("on the branch\n")
    _git("add", "branch_marker.txt")
    _git("commit", "-q", "-m", "branch commit")
    # Back to main so the branch is NOT the checked-out HEAD (mirrors the repo-root
    # install staying on main while the feature branch lives in a worktree).
    _git("checkout", "-q", "main")


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_materialize_branch_src_worktree_add_and_overlay(tmp_path) -> None:
    """#793: the REAL ``materialize_branch_src`` produces a content-complete tree on the
    branch commit + overlays the working-tree-only gitlink, and is idempotent."""
    repo = tmp_path / "repo"
    _init_branch_repo(repo)
    # Working-tree-only gitlink content present in the src_root working tree only.
    oi = repo / "external" / "open-instruct" / "open_instruct"
    oi.mkdir(parents=True)
    (oi / "finetune.py").write_text("f\n")

    monkey_root = tmp_path / "slurm-src"

    # Point the scratch root at tmp_path via the env override so we do not touch ~.
    os.environ["EPS_SLURM_SRC_ROOT"] = str(monkey_root)
    try:
        scratch = materialize_branch_src(src_root=repo, branch="issue-X", issue=1, timeout=60)
        # (a) the branch's marker file is present (committed-tree content).
        assert (scratch / "branch_marker.txt").exists()
        # (b) the overlaid gitlink content is present (worktree-only, not in any commit).
        assert (scratch / "external" / "open-instruct" / "open_instruct" / "finetune.py").exists()
        # (c) pyproject.toml is present (the source-sanity assert passed).
        assert (scratch / "pyproject.toml").exists()

        # Idempotency: a second call still succeeds (prior scratch removed + recreated).
        scratch2 = materialize_branch_src(src_root=repo, branch="issue-X", issue=1, timeout=60)
        assert scratch2 == scratch
        assert (scratch2 / "branch_marker.txt").exists()
        assert (scratch2 / "external" / "open-instruct" / "open_instruct" / "finetune.py").exists()
    finally:
        os.environ.pop("EPS_SLURM_SRC_ROOT", None)
        # Detach the scratch worktree so tmp_path teardown does not leave a dangling
        # registration in the source repo (best-effort).
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(scratch)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_materialize_branch_src_unresolvable_branch_raises(tmp_path) -> None:
    """#793: a branch name ``rev-parse --verify`` cannot resolve raises ``RuntimeError``
    (fail-fast, no silent fallback to main)."""
    repo = tmp_path / "repo"
    _init_branch_repo(repo)

    os.environ["EPS_SLURM_SRC_ROOT"] = str(tmp_path / "slurm-src")
    try:
        with pytest.raises(RuntimeError, match="cannot resolve branch 'no-such-branch'"):
            materialize_branch_src(src_root=repo, branch="no-such-branch", issue=2, timeout=60)
    finally:
        os.environ.pop("EPS_SLURM_SRC_ROOT", None)


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_materialize_branch_src_absent_overlay_path_warns_not_crashes(tmp_path, caplog) -> None:
    """#793: an overlay path absent in ``src_root`` (a legitimate lora-only run with no
    open-instruct) logs a WARNING and is skipped — the function still succeeds."""
    import logging

    repo = tmp_path / "repo"
    _init_branch_repo(repo)
    # NOTE: no external/open-instruct is created in the src_root working tree.

    os.environ["EPS_SLURM_SRC_ROOT"] = str(tmp_path / "slurm-src")
    try:
        with caplog.at_level(logging.WARNING):
            scratch = materialize_branch_src(
                src_root=repo,
                branch="issue-X",
                issue=3,
                overlay_paths=("nonexistent-dir",),
                timeout=60,
            )
        # (a) did not raise; (b) returned a valid scratch with pyproject.toml.
        assert (scratch / "pyproject.toml").exists()
        # (c) a WARNING was logged for the absent overlay path.
        assert any(
            "nonexistent-dir" in rec.getMessage() and rec.levelno == logging.WARNING
            for rec in caplog.records
        ), [rec.getMessage() for rec in caplog.records]
    finally:
        os.environ.pop("EPS_SLURM_SRC_ROOT", None)
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(scratch)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )


def test_assert_repo_branch_synced_accepts_explicit_src_root(tmp_path) -> None:
    """#793: the guard accepts an explicit ``src_root`` (§4d). Passing a branch tree
    passes; passing a ``main`` tree with a non-``main`` request still raises (unchanged
    internal semantics under the new arg — the guard STILL refuses a mismatched source)."""
    (tmp_path / "pyproject.toml").write_text("")
    branch_tree = tmp_path / "branch-tree"
    branch_tree.mkdir()
    main_tree = tmp_path / "main-tree"
    main_tree.mkdir()

    backend = SlurmBackend(
        src_root=tmp_path,
        rsyncer=lambda **_kw: None,
        secrets_pusher=lambda **_kw: None,
        runtime_clearer=lambda **_kw: None,
        git_branch_resolver=lambda root: "issue-793" if root == branch_tree else "main",
    )
    spec = _branch_spec("issue-793")

    # Explicit src_root on the branch → passes (no raise).
    backend._assert_repo_branch_synced(spec, src_root=branch_tree)
    # Explicit src_root on main with a non-main request → still raises.
    with pytest.raises(ValueError, match="cannot honor repo_branch='issue-793'"):
        backend._assert_repo_branch_synced(spec, src_root=main_tree)


def test_slurm_backend_launch_uses_scp_not_ssh_bash_c(tmp_path) -> None:
    """Blocker 3 regression guard: secrets push MUST use scp/sftp/rsync,
    NEVER ``ssh <alias> bash -c '<script>'`` (rejected by the robot
    forced-command wrapper) AND must use a unique temp path.

    Asserts the secrets_pusher's argv shape AND that two concurrent
    prepares don't collide on the same VM-side temp filename (the
    earlier ``$$`` PID idiom was a Python f-string, NOT shell
    expansion, so it produced the literal string ``$$`` every time).
    """
    (tmp_path / "pyproject.toml").write_text("")

    secrets_calls: list[dict] = []

    def fake_pusher(*, robot_alias, scratch_dir, content):
        secrets_calls.append(
            {"robot_alias": robot_alias, "scratch_dir": scratch_dir, "content": content}
        )

    backend = SlurmBackend(
        src_root=tmp_path,
        submitter=lambda *, robot_alias, sbatch_script: "9100",
        rsyncer=lambda **_: None,
        marker_poster=lambda **_: None,
        secrets_pusher=fake_pusher,
        runtime_clearer=lambda **_: None,
    )
    backend.prepare(_lora_spec())
    backend.prepare(_lora_spec())

    assert len(secrets_calls) == 2
    for call in secrets_calls:
        assert call["robot_alias"] == "robot-nibi"
        assert call["scratch_dir"] == "/scratch/tjiral/eps/issue-137"


def test_prepare_clears_runtime_artifacts_before_rsync(tmp_path) -> None:
    """Round-6 C2(2): prepare must clear the PRIOR attempt's scratch-root
    runtime artifacts (status.json / job.out / .current_phase /
    preflight.json) BEFORE the code rsync — they are outside the code
    rsync's --delete reach and poison the monitor + started-evidence
    probe on every re-run (issue 535 attempt 2)."""
    (tmp_path / "pyproject.toml").write_text("")

    order: list[str] = []
    clear_calls: list[dict] = []

    def fake_clearer(*, robot_alias, scratch_dir):
        order.append("clear")
        clear_calls.append({"robot_alias": robot_alias, "scratch_dir": scratch_dir})

    def fake_rsync(**_kw):
        order.append("rsync")

    backend = SlurmBackend(
        src_root=tmp_path,
        submitter=lambda *, robot_alias, sbatch_script: "9101",
        rsyncer=fake_rsync,
        marker_poster=lambda **_: None,
        secrets_pusher=lambda **_: order.append("secrets"),
        runtime_clearer=fake_clearer,
    )
    backend.prepare(_lora_spec())

    assert order == ["clear", "rsync", "secrets"]
    assert clear_calls == [
        {"robot_alias": "robot-nibi", "scratch_dir": "/scratch/tjiral/eps/issue-137"}
    ]


def test_build_clear_runtime_artifacts_command_golden() -> None:
    """Golden argv for the rsync-an-empty-stub deletion technique: the
    runtime filenames ride --include, everything else is protected by
    --exclude '*' (NO --delete-excluded), and the empty staging dir is
    the source so --delete removes exactly the included names."""
    from explore_persona_space.backends.slurm import (
        RUNTIME_ARTIFACT_FILENAMES,
        build_clear_runtime_artifacts_command,
    )

    argv = build_clear_runtime_artifacts_command(
        empty_dir="/tmp/eps-slurm-clear-x",
        dest_root="/scratch/tjiral/eps/issue-137",
        robot_alias="robot-nibi",
    )
    assert argv == [
        "rsync",
        "-a",
        "--delete",
        "--mkpath",
        "--include",
        "status.json",
        "--include",
        "job.out",
        "--include",
        ".current_phase",
        "--include",
        "preflight.json",
        "--exclude",
        "*",
        "/tmp/eps-slurm-clear-x/",
        "robot-nibi:/scratch/tjiral/eps/issue-137/",
    ]
    assert "--delete-excluded" not in argv, (
        "--delete-excluded would wipe the whole scratch root (code tree, "
        "secrets.env) — the filter set protects everything not included"
    )
    assert RUNTIME_ARTIFACT_FILENAMES == (
        "status.json",
        "job.out",
        ".current_phase",
        "preflight.json",
    )


def test_fetch_logs_reads_correct_path_and_returns_joined_string(tmp_path) -> None:
    """Blocker 4 regression guard: fetch_logs MUST read from the same
    /tmp/slurm-<id>/job.out path the monitor writes (NOT
    /tmp/slurm-<id>/<basename(scratch_dir)>/job.out — that was the bug,
    which always returned "") AND return a real newline-joined string
    (NOT the Python list repr from ``splitlines()[-200:].__str__()``).
    """
    (tmp_path / "pyproject.toml").write_text("")

    from explore_persona_space.backends.base import RunHandle
    from explore_persona_space.backends.slurm_monitor import _local_state_dir

    job_id = "8801"
    # Pre-seed the file at the path the monitor uses.
    local_dir = _local_state_dir(job_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"line {i}" for i in range(250)]
    (local_dir / "job.out").write_text("\n".join(lines) + "\n")

    backend = SlurmBackend(
        src_root=tmp_path,
        submitter=lambda *, robot_alias, sbatch_script: job_id,
        rsyncer=lambda **_: None,
        marker_poster=lambda **_: None,
    )
    handle = RunHandle(
        backend="cluster",
        cluster="nibi",
        job_id=job_id,
        pod_name="eps-issue-137",
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        extra={"issue": 137},
    )

    tail = backend.fetch_logs(handle)
    # Real string, NOT a list repr (the old buggy code returned
    # ``"['line 50', 'line 51', ...]"`` from splitlines()[-200:].__str__()).
    assert isinstance(tail, str)
    assert not tail.startswith("[")
    # Joined with real newlines, last 200 lines (50..249 inclusive).
    actual_lines = tail.split("\n")
    assert len(actual_lines) == 200, f"expected last-200 tail, got {len(actual_lines)}"
    assert actual_lines[0] == "line 50"
    assert actual_lines[-1] == "line 249"


def test_fetch_logs_scrubs_secret_tokens(tmp_path, monkeypatch) -> None:
    """fetch_logs is a tail-bearing API advertised "for orchestrator
    notifications" — its output must pass the C1 token scrubber so a
    future caller can't silently re-open the xtrace leak (round-7 Mn2).

    ``_local_state_dir`` is monkeypatched under ``tmp_path`` so this
    test never writes to the real ``/tmp/slurm-<id>`` (the sibling
    no-local-file test reads the real path and must stay isolated).
    """
    (tmp_path / "pyproject.toml").write_text("")

    from explore_persona_space.backends.base import RunHandle

    job_id = "8802"
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm_monitor._local_state_dir",
        lambda jid: tmp_path / f"slurm-{jid}",
    )
    local_dir = tmp_path / f"slurm-{job_id}"
    local_dir.mkdir(parents=True, exist_ok=True)
    secret = "hf_" + "a" * 30
    (local_dir / "job.out").write_text(f"+ : {secret}\n[phase=preflight]\n")

    backend = SlurmBackend(
        src_root=tmp_path,
        submitter=lambda *, robot_alias, sbatch_script: job_id,
        rsyncer=lambda **_: None,
        marker_poster=lambda **_: None,
    )
    handle = RunHandle(
        backend="cluster",
        cluster="nibi",
        job_id=job_id,
        pod_name="eps-issue-137",
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        extra={"issue": 137},
    )

    tail = backend.fetch_logs(handle)
    assert secret not in tail
    assert "«REDACTED»" in tail
    assert "[phase=preflight]" in tail


def test_fetch_logs_returns_empty_when_no_local_file(tmp_path) -> None:
    """No rsync ever landed → fetch_logs returns '' (NOT raises)."""
    (tmp_path / "pyproject.toml").write_text("")
    from explore_persona_space.backends.base import RunHandle

    backend = SlurmBackend(
        src_root=tmp_path,
        submitter=lambda *, robot_alias, sbatch_script: "8802",
        rsyncer=lambda **_: None,
        marker_poster=lambda **_: None,
    )
    handle = RunHandle(
        backend="cluster",
        cluster="nibi",
        job_id="8802",  # No prior /tmp/slurm-8802/job.out
        pod_name="eps-issue-137",
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        extra={"issue": 137},
    )
    assert backend.fetch_logs(handle) == ""


def test_scp_push_secrets_uses_scp_argv_with_unique_temp(tmp_path, monkeypatch) -> None:
    """The default pusher MUST: (a) build a ``scp`` argv (not ``ssh ...
    bash -c``); (b) use a genuinely-unique VM temp file (tempfile.mkstemp,
    NOT the literal ``$$`` string from the earlier f-string bug);
    (c) always clean up the temp file even on success.
    """
    from explore_persona_space.backends.slurm import scp_push_secrets

    captured_argvs: list[list[str]] = []
    captured_temps: list[str] = []

    def fake_run(argv, **kwargs):
        captured_argvs.append(list(argv))
        # argv[-2] is the tempfile path (scp -p -q TMP REMOTE).
        captured_temps.append(argv[-2])

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("explore_persona_space.backends.slurm.subprocess.run", fake_run)

    scp_push_secrets(
        robot_alias="robot-nibi",
        scratch_dir="/scratch/tjiral/eps/issue-137",
        content="HF_TOKEN=abc\n",
    )
    scp_push_secrets(
        robot_alias="robot-nibi",
        scratch_dir="/scratch/tjiral/eps/issue-137",
        content="HF_TOKEN=abc\n",
    )

    assert len(captured_argvs) == 2
    for argv in captured_argvs:
        # MUST be scp — NOT ssh ... bash -c (wrapper rejects that).
        assert argv[0] == "scp", argv
        assert "ssh" not in argv, argv
        assert "bash" not in argv, argv
        assert "-c" not in argv, argv
        # Last positional = remote target at the canonical filename.
        assert argv[-1] == "robot-nibi:/scratch/tjiral/eps/issue-137/secrets.env", argv
        # The literal ``$$`` string MUST NOT appear (that was the bug —
        # f-string did NOT expand it on the shell side, so two concurrent
        # prepares would collide).
        assert "$$" not in argv[-2], argv

    # Two prepares produced DIFFERENT temp paths (mkstemp guarantee).
    assert captured_temps[0] != captured_temps[1]

    # Temp files MUST be cleaned up after the scp completes (try/finally).
    from pathlib import Path as _P

    for tmp in captured_temps:
        assert not _P(tmp).exists(), f"VM-side secrets temp leaked: {tmp}"


# ---------------------------------------------------------------------------
# estimate_start — sbatch --test-only parsing + timezone (router slice 4)
# ---------------------------------------------------------------------------

# Captured verbatim from a real ``ssh robot-nibi sbatch --test-only`` round
# trip on 2026-06-08 (multi-backend-router plan §"Verified on real hardware").
# The output appears on stderr alongside any sbatch NOTE lines; the parser
# searches the combined blob. Hand-typed variants (e.g. "estimated start
# time …") do NOT appear in real output and were the reason the prior regex
# silently matched zero jobs.
_REAL_NIBI_TEST_ONLY_STDERR = (
    "sbatch: NOTE: Your memory allocation 480000 may be wasteful;\n"
    "sbatch: NOTE: consider reducing to 64G per task.\n"
    "sbatch: Job 15819682 to start at 2026-06-09T02:06:36 using 1 processors "
    "on nodes g4 in partition gpubase_bygpu_b1\n"
)


def test_cluster_config_carries_timezone_default_and_nibi_pin() -> None:
    """``ClusterConfig.timezone`` defaults to America/Toronto; Nibi pins it.

    DRAC robot login nodes report cluster-local Eastern time. The router
    localizes the parsed ``--test-only`` timestamp via this zone before
    converting to UTC; an unset zone (or the prior ``.replace(tzinfo=UTC)``)
    silently skewed every estimate by 4-5h.
    """
    nibi = get_cluster_config("nibi")
    assert nibi.timezone == "America/Toronto"
    # Default also America/Toronto so a new cluster row (DRAC-shape) Just
    # Works without remembering to set the field.
    bare = ClusterConfig(
        name="probe",
        account="rrg-bengioy-ad_gpu",
        robot_alias="robot-probe",
        max_gpus_per_node=8,
        scratch_path="/scratch/tjiral",
    )
    assert bare.timezone == "America/Toronto"


def test_ssh_estimate_start_parses_real_to_start_at_line() -> None:
    """The regex MUST match the verified-on-Nibi ``to start at …`` line.

    The prior regex matched ``"start time"`` which never appears in the
    real output → every probe returned None → the router had no signal
    to rank free lanes by. This pins the parser to the captured real
    output so the regression cannot reappear.
    """
    from explore_persona_space.backends.slurm import ssh_estimate_start

    def fake_subprocess_run(*_args, **_kwargs):
        class _R:
            returncode = 0
            stderr = _REAL_NIBI_TEST_ONLY_STDERR
            stdout = ""

        return _R()

    import explore_persona_space.backends.slurm as slurm_mod

    orig_run = slurm_mod.subprocess.run
    slurm_mod.subprocess.run = fake_subprocess_run
    try:
        got = ssh_estimate_start(
            robot_alias="robot-nibi",
            sbatch_script="#!/bin/bash\n#SBATCH --account=rrg-bengioy-ad_gpu\n",
            cluster_timezone="America/Toronto",
        )
    finally:
        slurm_mod.subprocess.run = orig_run

    assert got is not None, "Real --test-only output must parse to a datetime"
    assert got.tzinfo is not None, "Parsed datetime must be tz-aware (UTC)"
    assert got.utcoffset() == timedelta(0), "Return value must be UTC"


def test_ssh_estimate_start_localizes_via_cluster_timezone_not_utc() -> None:
    """Bug fix: cluster-local timestamp must localize via the cluster's tz,
    NEVER be wrapped naively with ``.replace(tzinfo=UTC)``.

    On 2026-06-09T02:06:36 America/Toronto, EDT (UTC-4) is in effect, so
    the correct UTC instant is 06:06:36Z. The prior bug labeled it as
    02:06:36 UTC — every Nibi estimate read 4 hours in the past, so a
    backlogged cluster ranked as "instant" and the router would submit
    blindly.
    """
    from explore_persona_space.backends.slurm import ssh_estimate_start

    def fake_subprocess_run(*_args, **_kwargs):
        class _R:
            returncode = 0
            stderr = _REAL_NIBI_TEST_ONLY_STDERR
            stdout = ""

        return _R()

    import explore_persona_space.backends.slurm as slurm_mod

    orig_run = slurm_mod.subprocess.run
    slurm_mod.subprocess.run = fake_subprocess_run
    try:
        got = ssh_estimate_start(
            robot_alias="robot-nibi",
            sbatch_script="#!/bin/bash\n",
            cluster_timezone="America/Toronto",
        )
    finally:
        slurm_mod.subprocess.run = orig_run

    # 2026-06-09 is in EDT (UTC-4). Local 02:06:36 → 06:06:36Z.
    expected_utc = datetime(2026, 6, 9, 6, 6, 36, tzinfo=UTC)
    assert got == expected_utc, (
        f"Expected EDT-localized 02:06:36 to convert to 06:06:36Z, got {got}. "
        "The likely cause is a regression of the .replace(tzinfo=UTC) bug "
        "that mislabels cluster-local time as UTC."
    )


def test_ssh_estimate_start_handles_dst_boundary() -> None:
    """A timestamp inside EST (Nov-Mar) localizes via UTC-5, not UTC-4.

    Bare ``.replace(tzinfo=UTC)`` would be wrong by 5h here; bare
    ``.replace(tzinfo=ZoneInfo('America/Toronto'))`` lets zoneinfo pick
    the right offset based on the local date, so the round trip is
    correct in both EST and EDT. Pins the DST-aware behavior.
    """
    from explore_persona_space.backends.slurm import ssh_estimate_start

    def fake_subprocess_run(*_args, **_kwargs):
        class _R:
            returncode = 0
            # January is EST (UTC-5).
            stderr = (
                "sbatch: Job 1 to start at 2026-01-15T03:00:00 using 1 processors "
                "on nodes g4 in partition gpubase_bygpu_b1\n"
            )
            stdout = ""

        return _R()

    import explore_persona_space.backends.slurm as slurm_mod

    orig_run = slurm_mod.subprocess.run
    slurm_mod.subprocess.run = fake_subprocess_run
    try:
        got = ssh_estimate_start(
            robot_alias="robot-nibi",
            sbatch_script="#!/bin/bash\n",
            cluster_timezone="America/Toronto",
        )
    finally:
        slurm_mod.subprocess.run = orig_run

    # EST UTC-5: local 03:00 → 08:00 UTC.
    expected_utc = datetime(2026, 1, 15, 8, 0, 0, tzinfo=UTC)
    assert got == expected_utc, (
        f"Expected EST-localized 03:00 to convert to 08:00Z, got {got}. DST handling regressed."
    )


def test_ssh_estimate_start_returns_none_on_missing_to_start_at() -> None:
    """No ``to start at`` token → None (lane is still park-eligible, just
    cannot be ranked as instant)."""
    from explore_persona_space.backends.slurm import ssh_estimate_start

    def fake_subprocess_run(*_args, **_kwargs):
        class _R:
            returncode = 1
            stderr = "sbatch: error: Invalid account or account/partition combination\n"
            stdout = ""

        return _R()

    import explore_persona_space.backends.slurm as slurm_mod

    orig_run = slurm_mod.subprocess.run
    slurm_mod.subprocess.run = fake_subprocess_run
    try:
        got = ssh_estimate_start(
            robot_alias="robot-nibi",
            sbatch_script="#!/bin/bash\n",
            cluster_timezone="America/Toronto",
        )
    finally:
        slurm_mod.subprocess.run = orig_run
    assert got is None


def test_estimate_start_seconds_returns_signed_delta_from_now() -> None:
    """The router caller wants seconds-from-now; positive ⇒ future,
    negative ⇒ "would start now / in the past" ⇒ treat as instant."""
    from explore_persona_space.backends.slurm import estimate_start_seconds

    cluster = get_cluster_config("nibi")

    # Future estimate: 02:06:36 EDT = 06:06:36 UTC; now = 06:00:00 UTC → +396 s.
    def fake_future_estimator(*, robot_alias, sbatch_script, cluster_timezone):
        del robot_alias, sbatch_script, cluster_timezone
        return datetime(2026, 6, 9, 6, 6, 36, tzinfo=UTC)

    secs = estimate_start_seconds(
        spec=_lora_spec(),
        cluster=cluster,
        now=datetime(2026, 6, 9, 6, 0, 0, tzinfo=UTC),
        start_estimator=fake_future_estimator,
    )
    assert secs == pytest.approx(396.0)

    # Past estimate (cluster says "would start a minute ago") → negative.
    def fake_past_estimator(*, robot_alias, sbatch_script, cluster_timezone):
        del robot_alias, sbatch_script, cluster_timezone
        return datetime(2026, 6, 9, 5, 59, 0, tzinfo=UTC)

    secs_past = estimate_start_seconds(
        spec=_lora_spec(),
        cluster=cluster,
        now=datetime(2026, 6, 9, 6, 0, 0, tzinfo=UTC),
        start_estimator=fake_past_estimator,
    )
    assert secs_past == pytest.approx(-60.0)


def test_estimate_start_seconds_returns_none_when_probe_unparseable() -> None:
    """A lane that can't produce an estimate stays park-eligible but is
    not rankable — return None, never crash the router."""
    from explore_persona_space.backends.slurm import estimate_start_seconds

    cluster = get_cluster_config("nibi")

    def fake_estimator(*, robot_alias, sbatch_script, cluster_timezone):
        del robot_alias, sbatch_script, cluster_timezone
        return None

    secs = estimate_start_seconds(
        spec=_lora_spec(),
        cluster=cluster,
        now=datetime(2026, 6, 9, 6, 0, 0, tzinfo=UTC),
        start_estimator=fake_estimator,
    )
    assert secs is None


def test_estimate_start_seconds_threads_cluster_timezone_into_estimator() -> None:
    """The seconds helper MUST pass the cluster's tz to the parser — a
    silent fallback to UTC would re-introduce the bug for a non-Toronto
    cluster (e.g. Mila at America/Montreal).
    """
    from explore_persona_space.backends.slurm import estimate_start_seconds

    cluster = get_cluster_config("nibi")
    captured: dict[str, str] = {}

    def fake_estimator(*, robot_alias, sbatch_script, cluster_timezone):
        del robot_alias, sbatch_script
        captured["tz"] = cluster_timezone
        return datetime(2026, 6, 9, 6, 6, 36, tzinfo=UTC)

    estimate_start_seconds(
        spec=_lora_spec(),
        cluster=cluster,
        now=datetime(2026, 6, 9, 6, 0, 0, tzinfo=UTC),
        start_estimator=fake_estimator,
    )
    assert captured.get("tz") == "America/Toronto"


def test_estimate_and_submit_render_byte_identical_scripts() -> None:
    """Estimate and submit must use the SAME rendered sbatch.

    If they diverge (different gres, different account, different
    --time), the cluster estimates start time for a job that isn't the
    one we eventually submit — the ranking signal becomes uncorrelated
    with reality. SlurmBackend.launch() and SlurmBackend.estimate_start
    {,_seconds} all route through ``_render_script_for``; this test
    pins the byte-identity end-to-end through dependency injection.
    """
    submitted_scripts: list[str] = []
    estimator_scripts: list[str] = []

    def fake_submit(*, robot_alias, sbatch_script):
        del robot_alias
        submitted_scripts.append(sbatch_script)
        return "9999"

    def fake_estimator(*, robot_alias, sbatch_script, cluster_timezone):
        del robot_alias, cluster_timezone
        estimator_scripts.append(sbatch_script)
        return datetime(2026, 6, 9, 6, 6, 36, tzinfo=UTC)

    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as td:
        td_path = _P(td)
        (td_path / "pyproject.toml").write_text("")
        backend = SlurmBackend(
            src_root=td_path,
            submitter=fake_submit,
            rsyncer=lambda **_: None,
            marker_poster=lambda **_: None,
            secrets_pusher=lambda **_: None,
            start_estimator=fake_estimator,
        )
        spec = _lora_spec()
        # Run the probe paths first, then submit. Both must render the
        # SAME bytes (render_sbatch is deterministic in (spec, cluster,
        # plan, scratch_dir, plan_hash), and all three paths share
        # _render_script_for).
        backend.estimate_start(spec)
        backend.estimate_start_seconds(spec, now=datetime(2026, 6, 9, 6, 0, 0, tzinfo=UTC))
        backend.launch(spec)

    assert len(submitted_scripts) == 1
    assert len(estimator_scripts) == 2  # estimate_start + estimate_start_seconds
    for est_script in estimator_scripts:
        assert est_script == submitted_scripts[0], (
            "estimate_start probe script must be byte-identical to the "
            "submit script for the same (spec, cluster) — divergence "
            "means the cluster estimates start time for a different job "
            "than the one we actually submit."
        )


def test_slurm_backend_estimate_start_returns_utc_via_cluster_timezone() -> None:
    """End-to-end: ``SlurmBackend.estimate_start`` returns a tz-aware UTC
    datetime, computed by localizing through ``cluster.timezone``.

    This is the regression site for the original two bugs (regex
    misparse + UTC mislabel) wired through the backend, not just the
    bare ``ssh_estimate_start`` function.
    """

    def fake_estimator(*, robot_alias, sbatch_script, cluster_timezone):
        del robot_alias, sbatch_script
        # Mirror what ssh_estimate_start would return for the real
        # captured Nibi line under the right timezone.
        assert cluster_timezone == "America/Toronto"
        return datetime(2026, 6, 9, 6, 6, 36, tzinfo=UTC)

    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as td:
        td_path = _P(td)
        (td_path / "pyproject.toml").write_text("")
        backend = SlurmBackend(
            src_root=td_path,
            submitter=lambda *, robot_alias, sbatch_script: "0",
            rsyncer=lambda **_: None,
            marker_poster=lambda **_: None,
            start_estimator=fake_estimator,
        )
        got = backend.estimate_start(_lora_spec())
        assert got == datetime(2026, 6, 9, 6, 6, 36, tzinfo=UTC)


def test_slurm_backend_estimate_start_seconds_uses_default_now_when_omitted() -> None:
    """The router can call ``estimate_start_seconds(spec)`` without a
    ``now`` and get a sensible delta against the current wall clock."""

    def fake_estimator(*, robot_alias, sbatch_script, cluster_timezone):
        del robot_alias, sbatch_script, cluster_timezone
        # Estimate 30 seconds from "now"
        return datetime.now(UTC) + timedelta(seconds=30)

    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as td:
        td_path = _P(td)
        (td_path / "pyproject.toml").write_text("")
        backend = SlurmBackend(
            src_root=td_path,
            submitter=lambda *, robot_alias, sbatch_script: "0",
            rsyncer=lambda **_: None,
            marker_poster=lambda **_: None,
            start_estimator=fake_estimator,
        )
        secs = backend.estimate_start_seconds(_lora_spec())
    assert secs is not None
    # Allow a wide window for test-scheduling jitter; the key invariant
    # is that the returned value is positive ~30 s, not a negative or
    # multi-hour skew (which would indicate UTC mislabel of local time).
    assert 0 < secs < 120, secs


# ---------------------------------------------------------------------------
# Slice-7: Mila first-class — ClusterConfig + access_mode + ssh_host
# ---------------------------------------------------------------------------


def test_mila_cluster_config_is_first_class_and_interactive() -> None:
    """The Mila row is wired and marked ``access_mode='interactive'``."""
    cfg = get_cluster_config("mila")
    assert cfg.name == "mila"
    assert cfg.available is True, "Mila ships in slice 7"
    assert cfg.access_mode == "interactive"
    # The SSH alias the rest of the backend should target — for Mila
    # that's the ControlMaster ``mila`` alias from clusters.config.
    assert cfg.ssh_host == "mila"
    assert cfg.robot_alias == "mila"
    # Mila default partitions do NOT require --account; the renderer
    # must skip the line.
    assert cfg.account is None
    # Eastern time (same offset as DRAC under DST) but named distinctly.
    assert cfg.timezone == "America/Montreal"


def test_drac_cluster_config_access_mode_defaults_to_robot() -> None:
    """Existing DRAC rows are unchanged — robot mode by default."""
    nibi = get_cluster_config("nibi")
    assert nibi.access_mode == "robot"
    assert nibi.ssh_host == "nibi".__class__("robot-nibi")  # str alias
    assert nibi.ssh_host == nibi.robot_alias == "robot-nibi"
    assert nibi.account == "rrg-bengioy-ad_gpu"


def test_render_sbatch_omits_account_line_when_cluster_account_is_none() -> None:
    """Mila renders WITHOUT ``#SBATCH --account=`` (cluster.account is None)."""
    mila = get_cluster_config("mila")
    spec = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="mila",
        cluster="mila",
        hydra_args=("condition=c1_evil_wrong_em",),
    )
    plan = stages_for_spec(spec)
    rendered = render_sbatch(
        spec=spec,
        cluster=mila,
        plan=plan,
        scratch_dir="/network/scratch/t/thomas.jiralerspong/eps/issue-137",
    )
    assert "#SBATCH --account=" not in rendered, (
        "Mila has no --account requirement on default partitions; the renderer "
        "must omit the line when cluster.account is None (an empty "
        "--account= line is rejected by some SLURM builds)."
    )
    # Sanity: the other essential headers DO appear.
    assert "#SBATCH --job-name=eps-issue-137" in rendered
    assert "#SBATCH --gpus-per-node=a100l:1" in rendered


def test_render_sbatch_keeps_account_line_for_drac_clusters() -> None:
    """Regression guard for the omit-account-when-None change."""
    nibi = get_cluster_config("nibi")
    spec = _lora_spec()
    plan = stages_for_spec(spec)
    rendered = render_sbatch(
        spec=spec,
        cluster=nibi,
        plan=plan,
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )
    assert "#SBATCH --account=rrg-bengioy-ad_gpu" in rendered, (
        "DRAC clusters MUST still emit the --account line; making it conditional "
        "on a non-None account was the slice-7 change, not a behaviour change "
        "for clusters that have one."
    )


# ---------------------------------------------------------------------------
# Slice-7: mila_socket_alive probe
# ---------------------------------------------------------------------------


def test_mila_socket_alive_uses_batch_mode_and_returns_true_on_exit_zero() -> None:
    """A healthy socket = ssh exit 0 → True."""
    from explore_persona_space.backends.slurm import (
        DEFAULT_MILA_SSH_ALIAS,
        mila_socket_alive,
    )

    captured: dict[str, object] = {}

    def fake_runner(argv: list[str], timeout: int) -> int:
        captured["argv"] = argv
        captured["timeout"] = timeout
        return 0

    assert mila_socket_alive(runner=fake_runner) is True
    argv = captured["argv"]
    assert isinstance(argv, list)
    # BatchMode is load-bearing: prevents SSH from prompting for OTP if
    # the socket is down/expired.
    assert "BatchMode=yes" in argv, argv
    assert argv[-1] == "true"
    assert DEFAULT_MILA_SSH_ALIAS in argv


def test_mila_socket_alive_returns_false_on_nonzero_exit() -> None:
    """A dead socket / expired OTP = nonzero exit → False (skip-the-lane)."""
    from explore_persona_space.backends.slurm import mila_socket_alive

    def fake_runner(_argv: list[str], _timeout: int) -> int:
        return 255  # SSH's "connection failed"

    assert mila_socket_alive(runner=fake_runner) is False


def test_mila_socket_alive_returns_false_on_runner_exception() -> None:
    """A subprocess timeout / OSError = treated as down (NOT raised).

    The router relies on this graceful-False contract — a raise here
    would propagate up through ``_auto_route`` and turn a socket
    hiccup into a routing terminal.
    """
    from explore_persona_space.backends.slurm import mila_socket_alive

    def boom_runner(_argv: list[str], _timeout: int) -> int:
        raise OSError("ssh binary went sideways")

    assert mila_socket_alive(runner=boom_runner) is False


def test_mila_socket_alive_returns_false_on_subprocess_timeout(monkeypatch) -> None:
    """The real subprocess path also degrades to False on TimeoutExpired."""
    import subprocess

    from explore_persona_space.backends.slurm import mila_socket_alive

    def boom_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["ssh"], timeout=5)

    monkeypatch.setattr("explore_persona_space.backends.slurm.subprocess.run", boom_run)
    assert mila_socket_alive() is False


def test_mila_socket_alive_returns_false_on_oserror_spawning_ssh(monkeypatch) -> None:
    """OSError from ``subprocess.run`` (e.g. ssh missing) = False, not raise."""
    from explore_persona_space.backends.slurm import mila_socket_alive

    def boom_run(*_args, **_kwargs):
        raise OSError("ssh: command not found")

    monkeypatch.setattr("explore_persona_space.backends.slurm.subprocess.run", boom_run)
    assert mila_socket_alive() is False


# ---------------------------------------------------------------------------
# Slice-7: estimate_start_seconds over Mila uses the mila SSH alias
# ---------------------------------------------------------------------------


def test_slurm_backend_routes_estimate_through_mila_ssh_host_not_robot_alias() -> None:
    """For Mila, the est-start probe MUST be invoked over the ``mila``
    socket alias — confirms ``cluster.ssh_host`` plumbing through the
    estimator seam."""
    captured: dict[str, str] = {}

    def fake_estimator(*, robot_alias, sbatch_script, cluster_timezone):
        del sbatch_script, cluster_timezone
        captured["robot_alias"] = robot_alias
        # Return a known future datetime so the seconds-from-now path
        # exercises the tz-conversion code paths too.
        return datetime(2099, 1, 1, tzinfo=UTC)

    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as td:
        td_path = _P(td)
        (td_path / "pyproject.toml").write_text("")
        backend = SlurmBackend(
            src_root=td_path,
            submitter=lambda *, robot_alias, sbatch_script: "0",
            rsyncer=lambda **_: None,
            marker_poster=lambda **_: None,
            start_estimator=fake_estimator,
        )
        spec = RunSpec(
            issue=137,
            intent="lora-7b",
            backend="mila",
            cluster="mila",
            hydra_args=("condition=c1_evil_wrong_em",),
        )
        secs = backend.estimate_start_seconds(spec)
    assert secs is not None
    # The crucial assertion: the estimator was called over the ``mila``
    # alias, NOT some default like ``robot-mila`` that does not exist
    # in clusters.config.
    assert captured["robot_alias"] == "mila", captured


# ---------------------------------------------------------------------------
# issue #588 — A2 byte-identity snapshot (hydra-only sbatch)
# ---------------------------------------------------------------------------


def test_render_sbatch_hydra_only_byte_identical_to_pre_change_snapshot() -> None:
    """A2 (#588): the hydra-only sbatch render must be byte-for-byte
    unchanged by the workload_cmd feature.

    Fixture recorded from the PRE-change renderer at the issue-588
    merge-base (provenance in the fixture's JSON header); see the GCP
    twin in test_gcp_backend.py for the non-tautology rationale.
    """
    fixture = json.loads(
        (_P(__file__).parent / "fixtures" / "issue588_slurm_sbatch_hydra_only.json").read_text()
    )
    spec = _lora_spec("lora-7b")
    rendered = render_sbatch(
        spec=spec,
        cluster=_nibi(),
        plan=stages_for_spec(spec),
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )
    assert rendered == fixture["rendered_text"]


# ---------------------------------------------------------------------------
# issue #588 — custom workload_cmd stage + rendering
# ---------------------------------------------------------------------------


def _custom_spec(cmd: str = "bash scripts/issue588_smoke.sh --flag 'v 1'") -> RunSpec:
    return RunSpec(
        issue=137,
        intent="lora-7b",
        backend="cluster",
        cluster="nibi",
        workload_cmd=cmd,
    )


def _wrapped(cmd: str) -> str:
    """The #1004 rc-preserving rendered custom-stage line for ``cmd``."""
    return f"bash -eu -o pipefail -c {shlex.quote(cmd)}"


def _rendered_custom_line(cmd: str) -> str:
    """Extract the ACTUAL rendered custom-stage line from a real
    ``render_sbatch`` output (the line equal to the wrapper form) —
    live-dispatched-path discipline: never rebuild the line via a twin."""
    spec = _custom_spec(cmd)
    script = render_sbatch(
        spec=spec,
        cluster=_nibi(),
        plan=stages_for_spec(spec),
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )
    lines = script.splitlines()
    return lines[lines.index(_wrapped(cmd))]


def test_stages_for_spec_workload_cmd_single_custom_stage() -> None:
    """#588: a workload_cmd spec bypasses the intent → stage table —
    ONE custom stage; the intent keeps driving GPUs + --time."""
    plan = stages_for_spec(_custom_spec())
    assert [s.name for s in plan.stages] == ["workload"]
    stage = plan.stages[0]
    assert stage.backend == "custom"
    assert stage.custom_cmd == "bash scripts/issue588_smoke.sh --flag 'v 1'"
    # Intent-driven resources unchanged by the custom stage.
    assert default_gpus_for_intent(_custom_spec()) == 1
    assert time_budget_hours(_custom_spec()) == 6.0


def test_render_sbatch_custom_workload_rc_wrapped_with_lifecycle_intact() -> None:
    """#588/#1004: the custom command renders inside the rc-preserving
    inner-bash wrapper (never as a bare line) inside a
    ``[phase=workload]`` block; heartbeat / status.json / preflight /
    terminal ``[phase=done]`` machinery all wrap it unchanged."""
    spec = _custom_spec()
    script = render_sbatch(
        spec=spec,
        cluster=_nibi(),
        plan=stages_for_spec(spec),
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )
    lines = script.splitlines()
    # The command is the single shlex-quoted argument of an inner
    # `bash -eu -o pipefail -c` (#1004); the inner bash re-parses it as a
    # full shell line, so the #588 "complete shell command" contract holds.
    assert _wrapped("bash scripts/issue588_smoke.sh --flag 'v 1'") in lines
    # The bare, rc-masking pre-#1004 form is GONE.
    assert "bash scripts/issue588_smoke.sh --flag 'v 1'" not in lines
    # No hydra entrypoint on the custom path.
    assert "scripts/train.py" not in script
    assert "scripts/eval.py" not in script
    # Stage block + terminal phase machinery intact.
    assert 'echo "[phase=workload]"' in script
    assert 'echo "[phase=done]"' in script
    assert "_write_status" in script
    assert PREFLIGHT_FAIL_MARKER in script  # in-job preflight intact
    assert "HEARTBEAT" in script or "heartbeat" in script
    # Custom stages build the BASE venv (no open-instruct gpu extras) —
    # the documented Step 6b residual gap.
    assert "--extra gpu" not in script
    # EPS_* env contract parity with the GCP startup script (live-smoke
    # fix: nibi job 15955646 died on `EPS_ISSUE: parameter null or not
    # set`). Exports must precede the workload command.
    assert f"export EPS_ISSUE={spec.issue}" in lines
    assert 'export EPS_ATTEMPT_ID="slurm-${SLURM_JOB_ID}"' in lines
    assert lines.index(f"export EPS_ISSUE={spec.issue}") < lines.index(
        _wrapped("bash scripts/issue588_smoke.sh --flag 'v 1'")
    )
    # WandB project default (#601 follow-up r1) — parity with the GCP
    # workload_cmd lane: exported BEFORE the workload command so
    # HF-Trainer workloads stop landing in WandB's global default
    # 'huggingface' project; :- keeps an inline/internal override winning.
    wandb_export = 'export WANDB_PROJECT="${WANDB_PROJECT:-issue137}"'
    assert wandb_export in lines
    assert lines.index(wandb_export) < lines.index(
        _wrapped("bash scripts/issue588_smoke.sh --flag 'v 1'")
    )


def test_render_sbatch_custom_workload_compound_is_rc_wrapped() -> None:
    """#1004 (incident #952, GCP parity): a compound && custom_cmd renders
    inside the rc-preserving inner-bash wrapper, never as a bare line — a
    bare splice under the prelude's set -e rc-masks a first-command crash
    into a false [phase=done] + completion sentinel (errexit exempts
    non-final &&/|| list members)."""
    cmd = "uv run python scripts/a.py && uv run python scripts/b.py"
    spec = _custom_spec(cmd)
    script = render_sbatch(
        spec=spec,
        cluster=_nibi(),
        plan=stages_for_spec(spec),
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )
    lines = script.splitlines()
    assert _wrapped(cmd) in lines
    assert cmd not in lines  # the bare, rc-masking form is GONE
    # Wrapper precedes the terminal [phase=done] publish.
    assert lines.index(_wrapped(cmd)) < lines.index('echo "[phase=done]"')


def test_sbatch_custom_wrapper_propagates_compound_first_command_rc() -> None:
    """#1004 behavioral twin of the GCP test, on the LIVE rendered
    sbatch custom-stage line: under set -euo pipefail (the sbatch
    prelude's exact flag set), a `cmd1 && cmd2` custom_cmd whose FIRST
    command exits 7 aborts the harness with rc 7 (success tail — the
    stand-in for the terminal [phase=done] + sentinel — unreached); a
    succeeding compound reaches the tail with rc 0."""
    for cmd, want_rc, want_tail in (
        ("bash -c 'exit 7' && echo NOT_REACHED", 7, False),
        ("true && echo CHAIN_OK", 0, True),
    ):
        harness = "set -euo pipefail\n" + _rendered_custom_line(cmd) + "\necho SUCCESS_TAIL\n"
        proc = subprocess.run(["bash", "-c", harness], capture_output=True, text=True)
        assert proc.returncode == want_rc, (cmd, proc.returncode, proc.stderr)
        assert ("SUCCESS_TAIL" in proc.stdout) is want_tail
        assert "NOT_REACHED" not in proc.stdout


def test_render_sbatch_exports_pythonpath_once_before_stages() -> None:
    """#1172 (trap #823/#853): every sbatch exports exactly one repo-root
    PYTHONPATH prepend (``$SCRATCH_JOB_DIR`` first, inherited value appended
    via the nounset-exempt ``:+`` form), rendered ONCE lane-wide — AFTER the
    module-load block (so cluster-module PYTHONPATH mutations survive as
    appended entries) and BEFORE the first ``# === Stage:`` marker, so ALL
    stage backends (local / custom / open_instruct) inherit it. Asserted on
    a local-stage plan (lora-7b), a custom-stage plan (workload_cmd), and a
    full-FT plan (open_instruct stages)."""
    export = 'export PYTHONPATH="$SCRATCH_JOB_DIR${PYTHONPATH:+:$PYTHONPATH}"'
    for spec in (_lora_spec("lora-7b"), _custom_spec(), _full_ft_spec()):
        script = render_sbatch(
            spec=spec,
            cluster=_nibi(),
            plan=stages_for_spec(spec),
            scratch_dir="/scratch/tjiral/eps/issue-137",
        )
        lines = script.splitlines()
        assert lines.count(export) == 1, (spec.intent, lines.count(export))
        module_idxs = [i for i, ln in enumerate(lines) if ln.strip().startswith("module load")]
        assert module_idxs, "module load line missing"
        stage_idxs = [i for i, ln in enumerate(lines) if ln.startswith("# === Stage:")]
        assert stage_idxs, "no stage marker rendered"
        export_idx = lines.index(export)
        assert max(module_idxs) < export_idx < min(stage_idxs), (
            spec.intent,
            max(module_idxs),
            export_idx,
            min(stage_idxs),
        )


def test_render_sbatch_custom_stage_empty_cmd_raises() -> None:
    plan = SbatchPlan(stages=(Stage(name="workload", backend="custom", script_rel=""),))
    with pytest.raises(ValueError, match="requires custom_cmd"):
        render_sbatch(
            spec=_lora_spec(),
            cluster=_nibi(),
            plan=plan,
            scratch_dir="/scratch/tjiral/eps/issue-137",
        )


# ---------------------------------------------------------------------------
# issue #598 — terminal-block completion sentinel + launch declaration
# ---------------------------------------------------------------------------


def _slurm_backend_with_fakes(tmp_path, *, job_id: str = "9001") -> SlurmBackend:
    """Real :class:`SlurmBackend` with every external seam faked (no network)."""
    (tmp_path / "pyproject.toml").write_text("")
    return SlurmBackend(
        src_root=tmp_path,
        submitter=lambda *, robot_alias, sbatch_script: job_id,
        rsyncer=lambda **_kw: None,
        marker_poster=lambda **_kw: None,
        secrets_pusher=lambda **_kw: None,
        runtime_clearer=lambda **_kw: None,
    )


def test_render_terminal_block_writes_completion_sentinel() -> None:
    """#598: the terminal block writes the attempt-namespaced completion
    sentinel — exact unquoted heredoc opener (pins runtime expansion of
    ``${SLURM_JOB_ID}`` in the JSON body) and ordering: sentinel write
    BEFORE ``_write_status "done" 0`` BEFORE ``[phase=done]`` ('done' is
    published LAST, mirroring GCP)."""
    spec = _lora_spec()
    script = render_sbatch(
        spec=spec,
        cluster=_nibi(),
        plan=stages_for_spec(spec),
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )
    sentinel_assign = (
        'SENTINEL_PATH="$SCRATCH_JOB_DIR/eval_results/issue_137/'
        'slurm-${SLURM_JOB_ID}/.completion-sentinel.json"'
    )
    assert sentinel_assign in script
    assert 'mkdir -p "$(dirname "$SENTINEL_PATH")"' in script
    # EXACT unquoted heredoc opener — a quoted ('EOF') variant would stop
    # ${SLURM_JOB_ID} from expanding at runtime and the sentinel's
    # attempt_id field would carry the literal string.
    assert 'cat > "$SENTINEL_PATH" <<EOF' in script
    assert "<<'EOF'" not in script
    # Ordering: sentinel write → _write_status "done" 0 → [phase=done].
    i_sentinel = script.index('cat > "$SENTINEL_PATH" <<EOF')
    i_done_status = script.index('_write_status "done" 0')
    i_phase_done = script.index('echo "[phase=done]"')
    assert i_sentinel < i_done_status < i_phase_done


def test_render_sentinel_json_body_passes_verifier() -> None:
    """#598 render→verifier byte seam: the heredoc JSON body the sbatch
    writes (with ``${SLURM_JOB_ID}`` substituted) must PASS
    ``_check_sentinel`` for the same issue. The end-to-end test writes
    its OWN sentinel via the writer API, so without this a malformed
    heredoc body could ship green."""
    from explore_persona_space.backends.artifacts import VerifierIO, _check_sentinel

    spec = _lora_spec()
    script = render_sbatch(
        spec=spec,
        cluster=_nibi(),
        plan=stages_for_spec(spec),
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )
    lines = script.splitlines()
    opener = lines.index('cat > "$SENTINEL_PATH" <<EOF')
    body_lines: list[str] = []
    for line in lines[opener + 1 :]:
        if line == "EOF":
            break
        body_lines.append(line)
    else:
        pytest.fail("heredoc never closed with EOF")
    body = "\n".join(body_lines).replace("${SLURM_JOB_ID}", "9001")
    parsed = json.loads(body)  # must be valid JSON after expansion
    assert parsed["attempt_id"] == "slurm-9001"
    check = _check_sentinel(
        sentinel_path="/anywhere/.completion-sentinel.json",
        issue=137,
        io=VerifierIO(read_sentinel=lambda _p: body),
    )
    assert check["status"] == "PASS", check


def test_launch_attaches_expected_artifacts_declaration(tmp_path) -> None:
    """#598: ``launch()`` populates ``extra[EXPECTED_ARTIFACTS_HANDLE_KEY]``
    with the GCP-parity shape — hydra lane declares the
    ``issue<N>_<attempt>/raw_completions/`` HF prefix and the LOCAL
    post-rsync sentinel path."""
    from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY

    backend = _slurm_backend_with_fakes(tmp_path, job_id="9001")
    handle = backend.launch(_lora_spec())
    decl = handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    assert decl["issue"] == 137
    assert decl["sentinel_path"] == str(
        tmp_path / "eval_results/issue_137/slurm-9001/.completion-sentinel.json"
    )
    # #790: pure-hydra (no workload_cmd) declares NEITHER default git path —
    # train.py runs with skip_eval=True and writes no figures during the run.
    assert decl["git_paths"] == []
    assert decl["hf_data_paths"] == ["issue137_slurm-9001/raw_completions/"]
    assert decl["hf_model_paths"] == []


def test_launch_custom_workload_omits_guessed_hf_prefix(tmp_path) -> None:
    """#598 inherits the #601 carve-out: a custom ``workload_cmd`` launch
    declares NO guessed HF data prefix (the workload owns its prefix; a
    launch-time guess produced a false-negative teardown block on a
    perfectly-uploaded run)."""
    from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY

    backend = _slurm_backend_with_fakes(tmp_path, job_id="9001")
    spec = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="cluster",
        cluster="nibi",
        workload_cmd="bash scripts/foo.sh",
    )
    handle = backend.launch(spec)
    decl = handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    assert decl["hf_data_paths"] == []
    # The sentinel + git checks keep gating teardown on this lane.
    assert decl["sentinel_path"].endswith("slurm-9001/.completion-sentinel.json")
    # #790: custom_workload keeps the real eval_results/ check (drivers commit
    # eval JSONs during the run) but drops the analyzer-generated figures/.
    assert decl["git_paths"] == ["eval_results/issue_137/"]


def test_render_and_launch_sentinel_paths_agree(tmp_path) -> None:
    """#598 deliverable-2 path agreement: the render's scratch-relative
    sentinel path (with ``slurm-${SLURM_JOB_ID}`` substituted) equals
    the launch-declared LOCAL path relative to ``src_root`` — i.e. the
    existing ``fetch_results`` rsync of ``eval_results/`` carries
    exactly the file ``confirm_artifacts`` will read."""
    from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY

    spec = _lora_spec()
    script = render_sbatch(
        spec=spec,
        cluster=_nibi(),
        plan=stages_for_spec(spec),
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )
    match = re.search(r'SENTINEL_PATH="\$SCRATCH_JOB_DIR/(?P<rel>[^"]+)"', script)
    assert match, "rendered script must assign SENTINEL_PATH under $SCRATCH_JOB_DIR"
    render_rel = match.group("rel").replace("${SLURM_JOB_ID}", "9001")

    backend = _slurm_backend_with_fakes(tmp_path, job_id="9001")
    handle = backend.launch(spec)
    decl = handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    declared_rel = str(_P(decl["sentinel_path"]).relative_to(tmp_path))
    assert render_rel == declared_rel
