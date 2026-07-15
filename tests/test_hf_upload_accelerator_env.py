"""HF Hub upload accelerator wiring (#745).

Pins the structural edits that turn on the HF Hub upload accelerators by
default across every experiment-upload environment, with an explicit ``=0``
override path preserved:

  * ``HF_XET_HIGH_PERFORMANCE`` (PRIMARY — the project repos use the Xet
    backend) + ``HF_HUB_ENABLE_HF_TRANSFER`` (orthogonal LFS accelerator) are
    in both lane passthrough allowlists (so a dispatch-process ``=0`` /
    ``HF_HUB_DISABLE_XET=1`` (the REAL xet kill switch, #1195;
    ``HF_XET_DISABLE`` kept only as a legacy no-op alias, #1049) forwards to
    the remote worker);
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
import re
import subprocess
import tempfile

from explore_persona_space.backends import gcp, slurm
from explore_persona_space.backends.base import RunSpec
from explore_persona_space.backends.gcp import (
    GcpConfig,
    render_create_argv,
    render_startup_script,
)

_ACCEL_KEYS = ("HF_XET_HIGH_PERFORMANCE", "HF_HUB_ENABLE_HF_TRANSFER")
# The REAL xet kill switch (#1195; huggingface_hub 0.36.2 constants.py reads
# HF_HUB_DISABLE_XET) + the legacy no-op alias (verified inert, #1049) — BOTH
# pinned in the allowlists: the real switch so a dispatch-process =1 forwards,
# the alias so existing launch commands keep forwarding harmlessly.
_DISABLE_KEYS = ("HF_HUB_DISABLE_XET", "HF_XET_DISABLE")


# --------------------------------------------------------------------------
# Lane passthrough allowlists (the OVERRIDE channel)
# --------------------------------------------------------------------------


def test_gcp_passthrough_includes_accelerators_and_disable() -> None:
    for key in (*_ACCEL_KEYS, *_DISABLE_KEYS):
        assert key in gcp.STARTUP_PASSTHROUGH_ENV_KEYS, key


def test_slurm_passthrough_includes_accelerators_and_disable() -> None:
    for key in (*_ACCEL_KEYS, *_DISABLE_KEYS):
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


def test_gcp_fetch_stanza_renders_real_xet_kill_switch() -> None:
    """The GCP startup script's default-preserving metadata fetch stanza
    renders a fetch+export line for ``HF_HUB_DISABLE_XET`` (#1195), so a
    dispatch-process ``=1`` forwarded into instance metadata reaches the GCE
    workload. (No dispatch-env monkeypatch: the stanza renders EVERY
    allowlist key unconditionally — the create-argv test below covers the
    dispatch-env leg.)"""
    script = render_startup_script(
        spec=_hydra_spec(), config=_gcp_config(), attempt_id="att-fixed-745"
    )
    assert "instance/attributes/HF_HUB_DISABLE_XET" in script
    assert '[ -n "$_VAL" ] && export HF_HUB_DISABLE_XET="$_VAL"' in script


def test_gcp_create_argv_forwards_real_xet_kill_switch(monkeypatch) -> None:
    """A dispatch-process ``HF_HUB_DISABLE_XET=1`` lands as instance metadata
    on the create argv — the dispatch-env → metadata leg the fetch stanza
    reads back on the VM (#1195; mirrors the adapter-persist M2 create-side
    test in test_gcp_backend.py)."""
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    # Hermetic vs the invoking PROCESS env (mirrors test_gcp_backend.py's
    # autouse fixture): a real optional secret leaking in would make
    # render_create_argv demand a tempfile entry this direct-render test
    # doesn't thread. Derived from the real key tuple (minus the two threaded
    # keys) so a future STARTUP_SECRET_ENV_KEYS addition cannot re-break this
    # test — a batch-mate's load_dotenv() puts the repo-root .env's
    # GITHUB_TOKEN (#1205) into os.environ mid-session, which the two prior
    # hardcoded delenvs missed (pre-existing interaction surfaced on #1338).
    for key in gcp.STARTUP_SECRET_ENV_KEYS:
        if key not in ("HF_TOKEN", "WANDB_API_KEY"):
            monkeypatch.delenv(key, raising=False)
    argv = render_create_argv(
        spec=_hydra_spec(),
        config=_gcp_config(),
        attempt_id="att-fixed-745",
        startup_script="#!/bin/bash\n",
        secret_files={
            "HF_TOKEN": "/tmp/eps-test-secret-hf",
            "WANDB_API_KEY": "/tmp/eps-test-secret-wandb",
        },
    )
    joined = " ".join(a for a in argv if a.startswith("--metadata="))
    assert "HF_HUB_DISABLE_XET=1" in joined


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


def test_slurm_render_secrets_env_forwards_real_xet_kill_switch() -> None:
    """A dispatch-process ``HF_HUB_DISABLE_XET=1`` reaches the compute node
    via the secrets env file (#1195)."""
    out = slurm.render_secrets_env({"HF_HUB_DISABLE_XET": "1", "HF_TOKEN": "tok"})
    assert "HF_HUB_DISABLE_XET=1" in out


def test_slurm_render_secrets_env_skips_unset_xet_kill_switch() -> None:
    """An unset kill switch is NOT rendered (drop-when-absent), so xet stays
    enabled by default on the compute node."""
    out = slurm.render_secrets_env({"HF_TOKEN": "tok"})
    assert "HF_HUB_DISABLE_XET" not in out


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


# --------------------------------------------------------------------------
# GCE absent-passthrough must NOT clobber the static accelerator default to ""
# (#745 round 2 — the binding round-1 blocker). The fetch stanza re-exports
# every passthrough key from instance metadata; an ABSENT key (the common
# case — the dispatcher does not forward the accelerators) returns empty and,
# pre-fix, ``export KEY="$EMPTY"`` overwrote the static ``=1`` with ``""``,
# silently disabling the GCE lane acceleration. These tests EXECUTE the
# rendered shell to assert the FINAL pre-workload value — a string-presence
# assertion cannot catch this semantic ordering bug.
# --------------------------------------------------------------------------


def _accelerator_shell_lines(script: str, key: str) -> list[str]:
    """Pull, in render order, every rendered line that READS OR WRITES ``key``:
    the static ``${key:-1}`` default plus the metadata fetch/export stanza.

    Running just these lines (with a stubbed ``curl``) reproduces the exact
    default-vs-fetch interaction the bug lived in, without booting a VM."""
    out: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # The static default: export KEY="${KEY:-1}"
        if f'export {key}="${{{key}:-1}}"' in stripped or f"instance/attributes/{key}" in stripped:
            out.append(stripped)
    return out


def _run_accelerator_resolution(script: str, key: str, *, metadata_value: str | None) -> str:
    """Execute the static-default + fetch lines for ``key`` under bash with a
    stubbed ``curl``, then echo the resolved value.

    ``metadata_value=None`` simulates an ABSENT metadata attribute (curl -f
    404s → empty stdout, the common default-GCE-workload case); a string
    simulates a dispatcher-forwarded override present in metadata."""
    lines = _accelerator_shell_lines(script, key)
    # The static default and at least one fetch line must both be present, or
    # the harness is asserting against the wrong render.
    assert any(":-1}" in ln for ln in lines), f"no static default line for {key}: {lines}"
    assert any("instance/attributes/" in ln for ln in lines), f"no fetch line for {key}: {lines}"

    if metadata_value is None:
        curl_stub = "curl() { return 22; }\n"  # curl -f returns 22 on 404; no stdout
    else:
        # shlex-safe single value echoed to stdout, exit 0.
        curl_stub = f"curl() {{ printf '%s' {metadata_value!r}; }}\n"

    harness = "#!/bin/bash\nset -u\n" + curl_stub + "\n".join(lines) + f'\necho "${{{key}}}"\n'
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(harness)
        path = fh.name
    proc = subprocess.run(["bash", path], capture_output=True, text=True)
    assert proc.returncode == 0, f"harness failed:\n{harness}\n--- stderr ---\n{proc.stderr}"
    return proc.stdout.strip()


def test_gce_absent_passthrough_preserves_static_default_one() -> None:
    """With NO accelerator env in the dispatch process (absent metadata),
    the rendered script's final pre-workload value for BOTH accelerators is
    ``1`` — the static default survives the fetch stanza (round-1 blocker)."""
    script = render_startup_script(
        spec=_hydra_spec(), config=_gcp_config(), attempt_id="att-fixed-745"
    )
    for key in _ACCEL_KEYS:
        resolved = _run_accelerator_resolution(script, key, metadata_value=None)
        assert resolved == "1", (
            f"{key} resolved to {resolved!r}, not '1' — absent passthrough metadata "
            "clobbered the static default (the #745 round-1 GCE wipeout regression)"
        )


def test_gce_explicit_zero_override_survives(monkeypatch) -> None:
    """A dispatcher-set ``HF_HUB_ENABLE_HF_TRANSFER=0`` forwards into metadata
    and the rendered script resolves it to ``0`` — the override channel
    (the #515 xet-CDN workaround) is intact through the default-preserving
    fetch."""
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "0")
    script = render_startup_script(
        spec=_hydra_spec(), config=_gcp_config(), attempt_id="att-fixed-745"
    )
    resolved = _run_accelerator_resolution(script, "HF_HUB_ENABLE_HF_TRANSFER", metadata_value="0")
    assert resolved == "0", (
        f"explicit =0 override resolved to {resolved!r}, not '0' — the metadata "
        "fetch must export a NON-empty forwarded value over the static default"
    )


def test_gce_fetch_stanza_is_default_preserving_form() -> None:
    """Structural guard on the fetch renderer: every passthrough/secret key's
    fetch line must guard the export on a non-empty fetch (the ``_VAL``-then-
    ``[ -n "$_VAL" ] && export`` form), never the old unconditional
    ``KEY=$(curl ...); export KEY`` that clobbered the default to ''."""
    script = render_startup_script(
        spec=_hydra_spec(), config=_gcp_config(), attempt_id="att-fixed-745"
    )
    for line in script.splitlines():
        if "instance/attributes/" not in line:
            continue
        m = re.search(r"instance/attributes/([A-Z_]+)", line)
        assert m, line
        key = m.group(1)
        # No unconditional ``export KEY`` (the regressed shape).
        assert f"); export {key}" not in line, (
            f"fetch line for {key} unconditionally re-exports — would clobber a "
            f"prior default to '' on an absent key:\n{line}"
        )
        # The default-preserving guard is present.
        assert f'[ -n "$_VAL" ] && export {key}="$_VAL"' in line, (
            f"fetch line for {key} is not the default-preserving _VAL form:\n{line}"
        )
