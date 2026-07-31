"""Issue #1345 on-policy-vs-injected program — permanent invariants.

Three guard families, all zero-GPU:

1. Launcher DEVICE RESOLUTION (#1902). Fellows SLURM nodes are GPU-SHARED and
   `nvidia-smi` ignores CUDA_VISIBLE_DEVICES, so a detected-count fan-out
   over-shards onto other tenants' GPUs. Width + physical ids must come from the
   ALLOCATION env, and a SLURM job exposing none of the allocation vars must FAIL
   LOUD rather than fall back to the physical count. Exercised against the SHIPPED
   script via its `EPM_I1345_RESOLVE_ONLY` affordance with a stubbed nvidia-smi.

2. vLLM `gpu_memory_utilization` computed from LIVE free memory (#1902 crash 1):
   a hardcoded fraction demands that share of TOTAL regardless of what other
   tenants hold, and EngineCore raises at init.

3. Provenance STORE KEYS: the `teacher_forced` default must reproduce every
   historical stem byte-for-byte (the live rounds' HF paths and fits registry
   entries must not move) while `on_policy` is distinct, so an on-policy capture
   is co-resident with its injected twin instead of overwriting it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
LAUNCHER = SCRIPTS / "issue1345_onpolicy_answers_launch.sh"

for _p in (str(SCRIPTS), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GIB = 2**30

# nvidia-smi stub: 8 physical devices; per-index free memory from $FAKE_FREE.
_STUB = """#!/usr/bin/env bash
args="$*"
if [[ "$args" == *"--query-gpu=index"* ]]; then
  for i in 0 1 2 3 4 5 6 7; do echo "$i"; done
  exit 0
fi
if [[ "$args" == *"--query-gpu=memory.free"* ]]; then
  idx=""; prev=""
  for a in "$@"; do
    if [ "$prev" = "-i" ]; then idx="$a"; fi
    prev="$a"
  done
  IFS=',' read -ra F <<< "${FAKE_FREE:-80000,80000,80000,80000,80000,80000,80000,80000}"
  echo "${F[$idx]:-0}"
  exit 0
fi
exit 1
"""


@pytest.fixture(scope="module")
def stub_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("nvsmi_stub")
    smi = d / "nvidia-smi"
    smi.write_text(_STUB)
    smi.chmod(0o755)
    return d


def _resolve(stub_dir: Path, **env_over) -> subprocess.CompletedProcess:
    """Run the SHIPPED launcher in resolve-only mode with a stubbed nvidia-smi."""
    env = {
        "PATH": f"{stub_dir}:{os.environ.get('PATH', '')}",
        "HOME": os.environ.get("HOME", "/tmp"),
        "EPM_I1345_RESOLVE_ONLY": "1",
        "REPO_ROOT": str(REPO_ROOT),
    }
    env.update({k: v for k, v in env_over.items() if v is not None})
    return subprocess.run(
        ["bash", str(LAUNCHER)], capture_output=True, text=True, env=env, timeout=300
    )


# ---------------------------------------------------------------------------
# 1. Launcher device resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("env_over", "want_source", "want_usable"),
    [
        # A SLURM allocation is authoritative; the physical node has 8 devices,
        # so any of these resolving to 8 would be the #1902 over-shard.
        (
            {"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "4", "CUDA_VISIBLE_DEVICES": "2,3,4,5"},
            "slurm-cvd",
            "2 3 4 5",
        ),
        (
            {"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "4", "SLURM_JOB_GPUS": "1,2,3,6"},
            "slurm-job-gpus",
            "1 2 3 6",
        ),
        (
            {"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "4", "SLURM_STEP_GPUS": "0,5"},
            "slurm-step-gpus",
            "0 5",
        ),
        (
            {"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "3"},
            "slurm-count-ids-assumed-0..N-1",
            "0 1 2",
        ),
        # Off-SLURM (exclusive host): enumeration is legitimate.
        ({}, "detected", "0 1 2 3 4 5 6 7"),
        ({"CUDA_VISIBLE_DEVICES": "3,4"}, "env-cvd", "3 4"),
    ],
)
def test_device_resolution_sources(stub_dir, env_over, want_source, want_usable):
    p = _resolve(stub_dir, **env_over)
    assert p.returncode == 0, p.stderr
    assert f"source={want_source}" in p.stdout, p.stdout
    assert f"usable={want_usable}" in p.stdout, p.stdout


def test_overlong_id_list_clamps_to_allocation_width(stub_dir):
    """An id list longer than SLURM_GPUS_ON_NODE is CLAMPED, not honored."""
    p = _resolve(
        stub_dir,
        SLURM_JOB_ID="1",
        SLURM_GPUS_ON_NODE="2",
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    )
    assert p.returncode == 0, p.stderr
    assert "-clamped" in p.stdout
    assert "usable=0 1" in p.stdout
    assert "n=2" in p.stdout


def test_slurm_without_allocation_vars_fails_loud(stub_dir):
    """The #1902 core invariant: NEVER fall back to the physical count on SLURM."""
    p = _resolve(stub_dir, SLURM_JOB_ID="1")
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    assert "refusing to fall back to the" in p.stderr
    assert "#1902" in p.stderr
    # The 8 physical devices must NOT appear as a resolution.
    assert "usable=0 1 2 3 4 5 6 7" not in p.stdout


def test_free_memory_filter_drops_tenant_held_devices(stub_dir):
    """Devices another tenant is holding are skipped before any cell launches."""
    p = _resolve(
        stub_dir,
        SLURM_JOB_ID="1",
        SLURM_GPUS_ON_NODE="4",
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        FAKE_FREE="1000,80000,2000,80000",
    )
    assert p.returncode == 0, p.stderr
    assert "usable=1 3" in p.stdout
    assert "n=2" in p.stdout
    assert "skipping device 0" in p.stderr


def test_all_devices_held_refuses_to_launch(stub_dir):
    """Every allocated device held -> rc=3, never a launch onto a full GPU."""
    env = {
        "PATH": f"{stub_dir}:{os.environ.get('PATH', '')}",
        "HOME": os.environ.get("HOME", "/tmp"),
        "REPO_ROOT": str(REPO_ROOT),
        "SLURM_JOB_ID": "1",
        "SLURM_GPUS_ON_NODE": "2",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "FAKE_FREE": "500,500",
    }
    p = subprocess.run(
        ["bash", str(LAUNCHER)], capture_output=True, text=True, env=env, timeout=300
    )
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    assert "no usable GPUs" in p.stderr


def test_launcher_pins_cvd_per_cell():
    """Each cell must pin CUDA_VISIBLE_DEVICES in the LAUNCHER env (CVD family)."""
    src = LAUNCHER.read_text()
    assert 'CUDA_VISIBLE_DEVICES="$dev"' in src
    # ... and must never size width off the physical count inside a SLURM job.
    assert "SLURM_JOB_ID" in src and "SLURM_GPUS_ON_NODE" in src


# ---------------------------------------------------------------------------
# 2. Live-probed vLLM gpu_memory_utilization
# ---------------------------------------------------------------------------
def _op_module(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "onpolicy_answers_ntpl_instruct")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "ARIA")
    import issue1345_onpolicy_answers_gen as op

    return op


def test_vllm_util_empty_device_resolves_to_cap(monkeypatch):
    op = _op_module(monkeypatch)
    got = op.vllm_util_for_free(int(139.8 * GIB), int(139.8 * GIB))
    assert got == pytest.approx(op.VLLM_UTIL_CAP)


def test_vllm_util_shared_node_clamps_below_free(monkeypatch):
    """The #1902 crash shape: 81.2 GiB free of 139.8 GiB on a shared H200."""
    op = _op_module(monkeypatch)
    util = op.vllm_util_for_free(int(81.2 * GIB), int(139.8 * GIB))
    assert util < op.VLLM_UTIL_CAP
    # The demanded share must fit inside free minus the safety margin.
    assert util * 139.8 <= 81.2 - op.GPU_FREE_MARGIN_GIB + 1e-6
    # And the bare cap WOULD have over-demanded — this is the crash it prevents.
    assert op.VLLM_UTIL_CAP * 139.8 > 81.2


def test_vllm_util_below_floor_fails_loud(monkeypatch):
    op = _op_module(monkeypatch)
    with pytest.raises(RuntimeError, match="GPU too full"):
        op.vllm_util_for_free(int(20.0 * GIB), int(139.8 * GIB))


def test_vllm_util_rejects_nonsense_total(monkeypatch):
    op = _op_module(monkeypatch)
    with pytest.raises(RuntimeError):
        op.vllm_util_for_free(1, 0)


def test_engine_uses_the_resolver_not_a_literal(monkeypatch):
    """A hardcoded fraction at the LLM() call site is the #1902 regression."""
    import inspect

    op = _op_module(monkeypatch)
    src = inspect.getsource(op.main)
    assert "gpu_memory_utilization=resolve_vllm_util()" in src
    assert "gpu_memory_utilization=0.85" not in src


def test_spawn_pin_set_before_vllm_import(monkeypatch):
    """vLLM reads this at import time; fork() poisons EngineCore (#628)."""
    op = _op_module(monkeypatch)
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
    assert "VLLM_WORKER_MULTIPROC_METHOD" in Path(op.__file__).read_text()


# ---------------------------------------------------------------------------
# 3. Provenance store keys
# ---------------------------------------------------------------------------
# The historical (teacher-forced) store stems, which must never move: the three
# live rounds resume against these HF paths and the fits registry keys on them.
LEGACY_FORMAT_KEYS = {
    "v1_boundary_present": "bnd_v1",
    "v2_boundary_absent": "bnd_v2",
    "v3_label_stripped": "bnd_v3",
    "chat": "bnd_chat",
    "no_template": "bnd_ntpl",
}


def _cap_module(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "story_boundary_ablation")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "Assistant")
    import issue1345_boundary_ablation_capture as cap

    return cap


@pytest.mark.parametrize(("key", "legacy"), sorted(LEGACY_FORMAT_KEYS.items()))
def test_teacher_forced_format_keys_are_byte_identical(monkeypatch, key, legacy):
    cap = _cap_module(monkeypatch)
    assert cap.format_key(key) == legacy
    assert cap.format_key(key, cap.PROV_TEACHER_FORCED) == legacy


@pytest.mark.parametrize(("key", "legacy"), sorted(LEGACY_FORMAT_KEYS.items()))
def test_on_policy_format_keys_are_distinct(monkeypatch, key, legacy):
    cap = _cap_module(monkeypatch)
    op_key = cap.format_key(key, cap.PROV_ON_POLICY)
    assert op_key == f"{legacy}_op"
    assert op_key != legacy


def test_no_stem_collides_across_key_and_provenance(monkeypatch):
    cap = _cap_module(monkeypatch)
    stems = [cap.stem_for(k, "instruct", pv) for k in LEGACY_FORMAT_KEYS for pv in cap.PROVENANCES]
    assert len(stems) == len(set(stems)), stems


def test_unknown_provenance_fails_loud(monkeypatch):
    """Never silently key an unknown provenance to the teacher-forced default."""
    cap = _cap_module(monkeypatch)
    with pytest.raises(AssertionError):
        cap.format_key("chat", "guessed")


def test_main_threads_provenance_at_every_key_site(monkeypatch):
    import inspect

    cap = _cap_module(monkeypatch)
    src = inspect.getsource(cap.main)
    for frag in (
        "stem_for(key, args.model, args.provenance)",
        "format_key(key, args.provenance)",
        "provenance=args.provenance",
    ):
        assert frag in src, frag
    # An un-threaded call would silently write the teacher-forced stem.
    assert "stem_for(key, args.model)" not in src
