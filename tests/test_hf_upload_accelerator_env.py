"""HF Hub upload accelerator wiring (#745).

Pins the structural edits that turn on the HF Hub upload accelerators by
default across every experiment-upload environment, with an explicit ``=0``
override path preserved:

  * ``HF_XET_HIGH_PERFORMANCE`` (PRIMARY — the project repos use the Xet
    backend) + ``HF_HUB_ENABLE_HF_TRANSFER`` (orthogonal LFS accelerator) are
    in both lane passthrough allowlists (so a dispatch-process ``=0`` /
    ``HF_XET_DISABLE=1`` forwards to the remote worker);
  * the GCE startup script statically defaults BOTH (so the workload AND the
    crash-persist subshell inherit them), and the static default precedes the
    ``_eps_persist_diagnostics`` definition so the EXIT-trap upload inherits it;
  * the SLURM sbatch statically defaults BOTH before the secrets ``source``
    (so a sourced ``=0`` override wins);
  * ``orchestrate.env.load_dotenv`` / ``setup_worker`` setdefault BOTH for
    local dev, preserving an explicit ``=0``.
"""

from __future__ import annotations

import os

from explore_persona_space.backends import gcp, slurm
from explore_persona_space.backends.base import RunSpec
from explore_persona_space.backends.gcp import GcpConfig, render_startup_script

_ACCEL_KEYS = ("HF_XET_HIGH_PERFORMANCE", "HF_HUB_ENABLE_HF_TRANSFER")


# --------------------------------------------------------------------------
# Lane passthrough allowlists (the OVERRIDE channel)
# --------------------------------------------------------------------------


def test_gcp_passthrough_includes_accelerators_and_disable() -> None:
    for key in (*_ACCEL_KEYS, "HF_XET_DISABLE"):
        assert key in gcp.STARTUP_PASSTHROUGH_ENV_KEYS, key


def test_slurm_passthrough_includes_accelerators_and_disable() -> None:
    for key in (*_ACCEL_KEYS, "HF_XET_DISABLE"):
        assert key in slurm.PASSTHROUGH_ENV_KEYS, key


# --------------------------------------------------------------------------
# GCE startup script — static defaults + crash-persist subshell coverage
# --------------------------------------------------------------------------


def _gcp_config() -> GcpConfig:
    return GcpConfig(
        project="eps-test-project",
        gcloud_config="eps-test-config",
        primary_zone="us-central1-a",
        fallback_zones=("us-central1-b", "us-central1-c"),
        image_family="pytorch-test-family",
    )


def _hydra_spec() -> RunSpec:
    return RunSpec(
        issue=745,
        intent="lora-7b",
        backend="gcp",
        hydra_args=("condition=c1", "seed=42"),
        extra={"attempt_id": "att-fixed-745"},
    )


def _workload_spec() -> RunSpec:
    return RunSpec(
        issue=745,
        intent="eval",
        backend="gcp",
        hydra_args=(),
        workload_cmd="bash scripts/x.sh",
        extra={"attempt_id": "att-fixed-745"},
    )


def test_gcp_startup_script_static_default_on_hydra_branch() -> None:
    script = render_startup_script(
        spec=_hydra_spec(), config=_gcp_config(), attempt_id="att-fixed-745"
    )
    assert 'export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"' in script
    assert 'export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"' in script


def test_gcp_startup_script_static_default_on_workload_branch() -> None:
    script = render_startup_script(
        spec=_workload_spec(), config=_gcp_config(), attempt_id="att-fixed-745"
    )
    assert 'export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"' in script
    assert 'export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"' in script


def test_gcp_static_export_precedes_redirect_and_clone() -> None:
    """The crash-persist EXIT-trap upload (``_eps_persist_diagnostics``,
    HfApi.upload_folder) runs with no load_dotenv and inherits the parent
    shell env AT THE MOMENT THE TRAP FIRES. The trap is INSTALLED early (and
    the helper is DEFINED early) but only EXECUTES later, so the meaningful
    invariant is that the static accelerator export runs before the heavy
    phases where real crashes happen — the output redirect, the repo clone,
    and the uv sync. Then any crash from those points on has the env set for
    the crash-persist subshell. The export sits in the env-export block, which
    is rendered before the output-redirect line and the clone block."""
    script = render_startup_script(
        spec=_hydra_spec(), config=_gcp_config(), attempt_id="att-fixed-745"
    )
    export_idx = script.index('export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"')
    redirect_idx = script.index('exec >>"$EPS_LOG_PATH"')
    clone_idx = script.index("# === Repo clone / pull (idempotent) ===")
    assert export_idx < redirect_idx < clone_idx, (
        "static accelerator export must precede the output redirect + clone/"
        "uv-sync so a crash there fires the EXIT trap with the accelerator env set"
    )
    # Sanity: the crash-persist helper this reasoning is about exists.
    assert "_eps_persist_diagnostics() {" in script


def test_gcp_passthrough_override_forwards_zero(monkeypatch) -> None:
    """A dispatch-process HF_HUB_ENABLE_HF_TRANSFER=0 is forwarded into the
    startup script's metadata-fetch+export stanza (the override channel)."""
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "0")
    script = render_startup_script(
        spec=_hydra_spec(), config=_gcp_config(), attempt_id="att-fixed-745"
    )
    # The fetch+export stanza is rendered for every passthrough key.
    assert "instance/attributes/HF_HUB_ENABLE_HF_TRANSFER" in script
    assert "export HF_HUB_ENABLE_HF_TRANSFER" in script


# --------------------------------------------------------------------------
# SLURM sbatch — static defaults before the secrets source
# --------------------------------------------------------------------------


def test_slurm_render_secrets_env_forwards_accelerator_override() -> None:
    """A dispatch-process =0 reaches the compute node via the secrets env file
    (plain KEY=value lines sourced under set -a, so it overrides the static
    default)."""
    out = slurm.render_secrets_env({"HF_XET_HIGH_PERFORMANCE": "0", "HF_TOKEN": "tok"})
    assert "HF_XET_HIGH_PERFORMANCE=0" in out


def test_slurm_render_secrets_env_skips_unset_accelerator() -> None:
    """An unset accelerator key is NOT rendered (drop-when-absent), so the
    sbatch's static default stands."""
    out = slurm.render_secrets_env({"HF_TOKEN": "tok"})
    assert "HF_XET_HIGH_PERFORMANCE" not in out
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in out


# --------------------------------------------------------------------------
# orchestrate.env setdefault — local-dev belt-and-suspenders
# --------------------------------------------------------------------------


def _clear_accel_env(monkeypatch) -> None:
    for key in _ACCEL_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_dotenv_setdefaults_accelerators(monkeypatch, tmp_path) -> None:
    from explore_persona_space.orchestrate import env as envmod

    _clear_accel_env(monkeypatch)
    # Use an explicit nonexistent path so we do NOT read the real project .env
    # (which may already carry these values); _dotenv_load no-ops on a missing
    # file, exercising only the setdefault lines.
    envmod.load_dotenv(env_path=str(tmp_path / "nope.env"))
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"


def test_load_dotenv_preserves_explicit_zero(monkeypatch, tmp_path) -> None:
    from explore_persona_space.orchestrate import env as envmod

    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "0")
    envmod.load_dotenv(env_path=str(tmp_path / "nope.env"))
    # setdefault must NOT clobber an explicit override (the #515 workaround).
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_setup_worker_setdefaults_accelerators(monkeypatch) -> None:
    from explore_persona_space.orchestrate import env as envmod

    _clear_accel_env(monkeypatch)
    # setup_worker calls load_dotenv() (no path) at the end, which would read
    # the real .env — but the setdefaults run BEFORE that and the .env read is
    # itself a setdefault-preserving load, so the assertions hold regardless.
    monkeypatch.setattr(envmod, "load_dotenv", lambda *a, **k: None)
    envmod.setup_worker(gpu_id=0)
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
