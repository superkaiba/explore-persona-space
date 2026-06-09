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

import os
import re
import subprocess

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
    )
    backend.prepare(_lora_spec())
    backend.prepare(_lora_spec())

    assert len(secrets_calls) == 2
    for call in secrets_calls:
        assert call["robot_alias"] == "robot-nibi"
        assert call["scratch_dir"] == "/scratch/tjiral/eps/issue-137"


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
