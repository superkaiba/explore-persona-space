"""Issue #2588-larger Slurm scripts — registry-driven submit + shell hygiene.

The submit wrapper validates the key and reads the tensor-parallel width from
the panel registry (PC.PANEL, one python call), keeps a small hand-written
BS/HR override case table for the wide FP8 MoE rows, and forwards
EPS_CAP_PROFILE / EPS_PHASE to the job through sbatch's default environment
export. This file pins those contracts, runs the submit script end to end
against a stub sbatch, and enforces the cluster rules the spec names: no
--export, no CUDA_VISIBLE_DEVICES export, no sbatch loops, nothing under
/home, set -euo pipefail, the HF_TOKEN guard, the heartbeat, the trap, and
bash -n syntax validity for both scripts.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2588_panel_common as PC

REPO = Path(__file__).resolve().parents[1]
SUBMIT = REPO / "scripts" / "issue2588x_submit.sh"
JOB = REPO / "scripts" / "issue2588x_cell_job.sh"
EXTENSION_KEYS = (
    "q38fn",
    "q35_397b",
    "dsv4_flash",
    "glm53",
    "dsv4_pro",
    # same-width (h=5120) column extension, 2026-09-02:
    "q3_32b",
    "qwq_32b",
    "q25_32b",
    "o3_32b_t",
)


def _submit_bs_table(text: str) -> dict[str, int]:
    # `key) BS=M ;;` or `key) BS=M; HR=G ;;` (BS = HF capture forward batch,
    # HR = per-GPU headroom GiB; defaults BS=8 HR=24 for every other key).
    pat = r"^\s*(\w+)\)\s+BS=(\d+)(?:;\s*HR=\d+)?\s*;;"
    return {k: int(bs) for k, bs in re.findall(pat, text, re.M)}


def test_submit_capture_batch_sizes():
    """Wide / long-context FP8 MoE rows take smaller HF capture batches (eager
    attention materialises (B, heads, T, T)); DeepSeek-V4-Flash TP=2 OOMed at
    B=8 (job 62452). Every multi-GPU registry key pins a BS, none exceeds 8,
    and every other key rides the BS=8 HR=24 defaults."""
    text = SUBMIT.read_text(encoding="utf-8")
    bs = _submit_bs_table(text)
    for key, m in PC.PANEL.items():
        if m.tp_gpus >= 2:
            assert key in bs, f"{key}: multi-GPU registry key without an explicit BS"
    assert all(1 <= v <= 8 for v in bs.values()), bs
    assert bs["dsv4_flash"] <= 2 and bs["dsv4_pro"] <= 1
    assert re.search(r"^BS=8$", text, re.M) and re.search(r"^HR=24$", text, re.M)
    hr = re.search(r"^\s*dsv4_flash\)\s+BS=\d+;\s*HR=(\d+)\s*;;", text, re.M)
    assert hr is not None and int(hr.group(1)) >= 40
    assert 'export EPS_CAPTURE_BS="$BS"' in text
    assert 'export EPS_CAPTURE_HEADROOM_GIB="$HR"' in text
    job_text = JOB.read_text(encoding="utf-8")
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in job_text
    assert '--capture-batch-size "${EPS_CAPTURE_BS:-8}"' in job_text


def test_submit_reads_tp_from_registry():
    """The hand-written key->TP case table is gone: submit validates the key
    against PC.PANEL and reads tp_gpus in ONE python call (single source; the
    job body re-asserts the same value). Every registry key is submittable,
    not just the old nine extension keys."""
    text = SUBMIT.read_text(encoding="utf-8")
    assert "PC.PANEL[k].tp_gpus" in text
    assert "unknown 2588x model key" in text
    assert not re.search(r"^\s*\w+\)\s+TP=\d+", text, re.M), "hardcoded TP case rows remain"
    # A registry key OUTSIDE the old nine is a first-class submit target now.
    assert "q35_4b" in PC.PANEL and "q35_4b" not in EXTENSION_KEYS
    assert PC.PANEL["q35_4b"].tp_gpus == 1
    assert set(EXTENSION_KEYS) < set(PC.PANEL)


def test_resource_arithmetic_spec():
    """--cpus-per-task=min(8*TP,64), --mem=128*TP G, per the launch spec."""
    text = SUBMIT.read_text(encoding="utf-8")
    assert "CPUS=$(( 8 * TP ))" in text
    assert 'if [ "$CPUS" -gt 64 ]; then CPUS=64; fi' in text
    assert "MEM=$(( 128 * TP ))" in text
    for frag in (
        '--gres="gpu:${TP}"',
        '--cpus-per-task="$CPUS"',
        '--mem="${MEM}G"',
        '--job-name="$JOB_NAME"',
        '--output="${LOGS}/%x-%j.out"',
    ):
        assert frag in text, frag


def test_cluster_rules_enforced_in_both_scripts():
    for path in (SUBMIT, JOB):
        text = path.read_text(encoding="utf-8")
        lines = [
            ln
            for ln in text.splitlines()
            if not ln.lstrip().startswith("#") or ln.lstrip().startswith("#SBATCH")
        ]
        code = "\n".join(lines)
        assert "set -euo pipefail" in text, path.name
        assert "--export" not in code, f"{path.name}: --export is banned"
        assert "CUDA_VISIBLE_DEVICES" not in code, f"{path.name}: CVD export is banned"
        assert "/home" not in code, f"{path.name}: nothing under /home"
        assert "HF_TOKEN must be in the submitting shell env" in text, path.name
        # The exact stdin submit form is documented in the header:
        assert "read -r HF_TOKEN; export HF_TOKEN; bash scripts/issue2588x_submit.sh" in text
    # ONE sbatch invocation, not a loop:
    submit_code = [
        ln
        for ln in SUBMIT.read_text(encoding="utf-8").splitlines()
        if "sbatch" in ln and not ln.lstrip().startswith("#")
    ]
    assert len(submit_code) == 1, submit_code
    job = JOB.read_text(encoding="utf-8")
    assert "sleep 600" in job  # 10-min heartbeat
    assert "trap term_handler TERM INT" in job
    assert "kill -- 0" in job  # process-group kill
    # Static directives per spec:
    for d in ("#SBATCH -p general", "#SBATCH --qos=high-eur", "#SBATCH -t 36:00:00"):
        assert d in job, d
    # The job derives arms + re-checks TP from the registry (single source):
    assert "PC.PANEL[sys.argv[1]].arms" in job
    assert "PC.PANEL[sys.argv[1]].tp_gpus" in job
    # Job env per spec:
    for env in (
        'HF_HOME="$BASE/hf_cache"',
        "HF_HUB_ENABLE_HF_TRANSFER=1",
        "NCCL_NVLS_ENABLE=0",
        "VLLM_GPU_MEM_UTIL=0.85",
        'PYTHONPATH="$BASE/repo/src:$BASE/repo/scripts"',
        "BASE=/workspace/superkaiba/eps2588x",
    ):
        assert env in job, env


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash on PATH")
def test_bash_syntax_valid():
    for path in (SUBMIT, JOB):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_cap_profile_and_phase_job_name_rule():
    """EPS_CAP_PROFILE and EPS_PHASE are exported to the job env (sbatch
    default export, like HF_TOKEN). Non-v1 jobs are named
    eps2588x-<profile>-<key> and a non-all phase appends -<phase>, so the
    queue shows what a job runs. v1 all-phase keeps the original name."""
    text = SUBMIT.read_text(encoding="utf-8")
    assert 'EPS_CAP_PROFILE="${EPS_CAP_PROFILE:-v1}"' in text
    assert "export EPS_CAP_PROFILE" in text
    assert 'EPS_PHASE="${EPS_PHASE:-all}"' in text
    assert "export EPS_PHASE" in text
    assert 'JOB_NAME="eps2588x-${KEY}"' in text
    assert 'if [ "$EPS_CAP_PROFILE" != "v1" ]; then' in text
    assert 'JOB_NAME="eps2588x-${EPS_CAP_PROFILE}-${KEY}"' in text
    assert 'if [ "$EPS_PHASE" != "all" ]; then' in text
    assert 'JOB_NAME="${JOB_NAME}-${EPS_PHASE}"' in text
    assert '--job-name="$JOB_NAME"' in text
    job = JOB.read_text(encoding="utf-8")
    assert "profile=${EPS_CAP_PROFILE:-v1}" in job
    # The job forwards the phase and narrows to the FIRST arm in single-phase
    # mode (g2-anchor is cell-independent, published once per cap profile).
    assert '--phase "${EPS_PHASE:-all}"' in job
    assert 'if [ "$EPS_PHASE" != "all" ]; then' in job
    assert 'ARMS="${ARMS%% *}"' in job


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash on PATH")
def test_submit_functional_fake_sbatch(tmp_path):
    """End-to-end submit against a stub sbatch: a registry key OUTSIDE the old
    nine submits with the registry TP, profile + phase ride the job name, and
    an unknown key fails rc=2 naming the full key list."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "repo").symlink_to(REPO)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "sbatch"
    stub.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$SBATCH_ARGS_OUT"\n'
        "echo Submitted batch job 1\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    args_out = tmp_path / "args.txt"
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        EPS2588X_BASE=str(base),
        EPS2588X_PY=sys.executable,
        HF_TOKEN="dummy-token-for-test",
        SBATCH_ARGS_OUT=str(args_out),
    )
    env.pop("EPS_CAP_PROFILE", None)
    env.pop("EPS_PHASE", None)

    # (a) Registry key outside the old nine, default profile + phase.
    r = subprocess.run(["bash", str(SUBMIT), "q35_4b"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
    args = args_out.read_text(encoding="utf-8").splitlines()
    assert "--gres=gpu:1" in args, args  # registry TP for q35_4b
    assert "--job-name=eps2588x-q35_4b" in args, args
    assert args[-2:] == ["q35_4b", "1"]

    # (b) Profile + single-phase mode ride the job name.
    env_lp = dict(env, EPS_CAP_PROFILE="long", EPS_PHASE="g2-anchor")
    r = subprocess.run(["bash", str(SUBMIT), "q35_4b"], env=env_lp, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
    args = args_out.read_text(encoding="utf-8").splitlines()
    assert "--job-name=eps2588x-long-q35_4b-g2-anchor" in args, args

    # (c) Unknown key fails loud with the full registry key list, rc=2.
    r = subprocess.run(["bash", str(SUBMIT), "not_a_key"], env=env, capture_output=True, text=True)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "unknown 2588x model key" in r.stderr
    assert "q35_4b" in r.stderr and "dsv4_pro" in r.stderr
