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

import re

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
    build_rsync_command,
    compute_plan_hash,
    default_gpus_for_intent,
    job_name,
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
        dest_root="/scratch/eps/issue-137",
        robot_alias="robot-nibi",
    )
    assert "--mkpath" in argv  # P0(a): intermediate dirs don't auto-create
    assert "--delete" in argv
    assert "-a" in argv
    assert "--partial" in argv
    # Destination
    assert argv[-1] == "robot-nibi:/scratch/eps/issue-137/"


def test_rsync_command_includes_open_instruct_and_configs(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    argv = build_rsync_command(
        src_root=tmp_path,
        dest_root="/scratch/eps/issue-137",
        robot_alias="robot-nibi",
    )

    # The sources are absolute paths anchored at src_root; assert each
    # required suffix appears in SOME argv entry.
    def _has_suffix(suffix: str) -> bool:
        return any(a.rstrip("/").endswith(suffix.rstrip("/")) for a in argv)

    # configs/ is module-relative for the DeepSpeed resolver.
    assert _has_suffix("configs"), argv
    # open-instruct + tulu + deepspeed configs needed for full-FT.
    assert _has_suffix("external/open-instruct"), argv
    assert _has_suffix("configs/tulu"), argv
    assert _has_suffix("configs/deepspeed"), argv


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
        scratch_dir="/scratch/eps/issue-137",
    )

    # Headers
    assert "#SBATCH --account=rrg-bengioy-ad_gpu" in script
    assert "#SBATCH --gpus-per-node=1" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --output=/scratch/eps/issue-137/job.out" in script
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
        scratch_dir="/scratch/eps/issue-137",
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
        scratch_dir="/scratch/eps/issue-137",
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
        scratch_dir="/scratch/eps/issue-137",
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
            scratch_dir="/scratch/eps/issue-1",
        )


def test_render_sbatch_includes_job_name_plan_hash() -> None:
    spec = _lora_spec()
    cluster = _nibi()
    plan = stages_for_spec(spec)
    script = render_sbatch(
        spec=spec,
        cluster=cluster,
        plan=plan,
        scratch_dir="/scratch/eps/issue-137",
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

    Uses dependency injection (the ``submitter`` / ``rsyncer`` ctor
    seams) so the test runs without any network / cluster.
    """
    (tmp_path / "pyproject.toml").write_text("")

    submitted: list[tuple[str, str]] = []

    def fake_submit(*, robot_alias, sbatch_script):
        submitted.append((robot_alias, sbatch_script))
        return "9001"

    rsynced: list[tuple[str, str, str]] = []

    def fake_rsync(*, src_root, dest_root, robot_alias):
        rsynced.append((str(src_root), dest_root, robot_alias))

    backend = SlurmBackend(
        src_root=tmp_path,
        submitter=fake_submit,
        rsyncer=fake_rsync,
    )
    spec = _lora_spec()
    handle = backend.launch(spec)

    assert handle.backend == "cluster"
    assert handle.cluster == "nibi"
    assert handle.job_id == "9001"
    assert handle.pod_name == "eps-issue-137"
    assert handle.scratch_dir == "/scratch/eps/issue-137"
    assert handle.log_path == "/scratch/eps/issue-137/job.out"
    assert handle.extra["account"] == "rrg-bengioy-ad_gpu"
    assert handle.extra["robot_alias"] == "robot-nibi"
    assert handle.extra["gpus_per_node"] == 1

    # Submit was called once with a real rendered sbatch.
    assert len(submitted) == 1
    alias, script = submitted[0]
    assert alias == "robot-nibi"
    assert "#SBATCH --account=rrg-bengioy-ad_gpu" in script
    assert "[phase=done]" in script
