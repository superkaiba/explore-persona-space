"""Issue #2588-larger Slurm scripts — table pins + shell hygiene (GPU-free).

The submit wrapper carries a hand-written key->TP case table (the login node
has no venv guarantee at submit time); this file pins that table to the panel
registry so the two can never drift, and enforces the cluster rules the spec
names: no --export, no CUDA_VISIBLE_DEVICES export, no sbatch loops, nothing
under /home, set -euo pipefail, the HF_TOKEN guard, the heartbeat, the trap,
and bash -n syntax validity for both scripts.
"""

from __future__ import annotations

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
EXTENSION_KEYS = ("q38fn", "q35_397b", "dsv4_flash", "glm53", "dsv4_pro")


def _case_table(text: str) -> dict[str, int]:
    return {k: int(tp) for k, tp in re.findall(r"^\s*(\w+)\)\s+TP=(\d+)\s+;;", text, re.M)}


def test_submit_tp_table_matches_registry():
    table = _case_table(SUBMIT.read_text(encoding="utf-8"))
    assert set(table) == set(EXTENSION_KEYS), table
    for key in EXTENSION_KEYS:
        assert table[key] == PC.PANEL[key].tp_gpus, (key, table[key], PC.PANEL[key].tp_gpus)


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
        '--job-name="eps2588x-${KEY}"',
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
