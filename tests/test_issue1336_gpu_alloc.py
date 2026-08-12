"""#1336 v21: SLURM-allocation-aware GPU derivation + pin selection (precheck 11981).

Bug 2 regression pins: `nvidia-smi --list-gpus` is a driver-level query that
enumerates ALL physical devices regardless of CUDA_VISIBLE_DEVICES, so on a
1-GPU SLURM allocation (inherited CVD=5) the dispatcher derived NGPU=8 and the
`CUDA_VISIBLE_DEVICES=0` / `=$w` overrides re-pointed workers onto another
job's physical GPU 0. The fix parses the INHERITED CVD as the allocated device
list (EPS_ALLOC_GPUS) and pins the w-th ELEMENT, falling back to the
nvidia-smi count ONLY when CVD is unset/empty.

Both legs run the REAL production bash — the banner tests execute the actual
dispatcher up to its cheap `__phase_key` probe exit; the queue tests extract
and execute the dispatcher's own `run_queue` function body (never a Python
re-implementation).

Byte-identity requirement (approved 210 GPU-h run): on the whole-node shape
SLURM sets CVD=0,1,...,7, so EPS_ALLOC_GPUS[w] == w and NGPU == 8 — pinned
here by asserting the CVD-parse path and the legacy nvidia-smi-count path
yield the IDENTICAL banner, and that 8 queue workers pin exactly {0..7}.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "issue1336_dispatch.sh"

BANNER_RE = re.compile(
    r"\[dispatch1336\] phase=\S+ smoke=\d realized_gpus=(\d+) alloc=\[([^\]]*)\]"
)


def _stub_bin(tmp_path: Path, n_gpus: int) -> Path:
    """A PATH dir whose nvidia-smi prints one line per fake GPU (0 => rc=1, no output)."""
    d = tmp_path / f"stub_bin_{n_gpus}"
    d.mkdir(exist_ok=True)
    smi = d / "nvidia-smi"
    if n_gpus > 0:
        lines = "\n".join(f"GPU {i}: Fake H200 (UUID: GPU-stub-{i})" for i in range(n_gpus))
        smi.write_text(f"#!/usr/bin/env bash\ncat <<'EOF'\n{lines}\nEOF\n")
    else:
        smi.write_text("#!/usr/bin/env bash\nexit 1\n")
    smi.chmod(smi.stat().st_mode | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _run_banner(tmp_path: Path, cvd: str | None, stub_gpus: int | None = None) -> tuple[int, str]:
    """Run the REAL dispatcher to its cheap __phase_key exit; return (ngpu, alloc-list-str)."""
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("REPO_ROOT", None)
    env["EPS_LOG_DIR"] = str(tmp_path / "logs")
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
    if stub_gpus is not None:
        env["PATH"] = f"{_stub_bin(tmp_path, stub_gpus)}:{env['PATH']}"
    proc = subprocess.run(
        ["bash", str(SCRIPT), "__phase_key", "gen"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    m = BANNER_RE.search(proc.stdout)
    assert m, f"banner line not found in stdout: {proc.stdout!r}"
    return int(m.group(1)), m.group(2)


def test_slurm_partial_allocation_cvd5_yields_ngpu1_alloc5(tmp_path):
    """The SLURM 11981 shape: inherited CVD=5 on an 8-GPU node => NGPU=1, alloc=[5].

    Pre-fix the dispatcher derived NGPU=8 from nvidia-smi; the stub PATH makes
    that legacy path return 8 so the test FAILS pre-fix and passes post-fix.
    """
    ngpu, alloc = _run_banner(tmp_path, cvd="5", stub_gpus=8)
    assert ngpu == 1
    assert alloc == "5"


def test_whole_node_cvd_byte_identity_with_nvidia_smi_fallback(tmp_path):
    """8-GPU byte-identity: CVD=0..7 parse == legacy nvidia-smi-count derivation."""
    from_cvd = _run_banner(tmp_path, cvd="0,1,2,3,4,5,6,7", stub_gpus=8)
    from_smi = _run_banner(tmp_path, cvd=None, stub_gpus=8)
    assert from_cvd == from_smi == (8, "0 1 2 3 4 5 6 7")


def test_unset_cvd_falls_back_to_nvidia_smi_count(tmp_path):
    ngpu, alloc = _run_banner(tmp_path, cvd=None, stub_gpus=3)
    assert ngpu == 3
    assert alloc == "0 1 2"


def test_unset_cvd_no_gpus_keeps_cpu_only_branch(tmp_path):
    """NGPU=0 CPU-only branches (smoke arms, run_queue CPU path) must stay reachable."""
    ngpu, alloc = _run_banner(tmp_path, cvd=None, stub_gpus=0)
    assert ngpu == 0
    assert alloc == ""


# ---------------------------------------------------------------------------
# run_queue worker pins — execute the REAL function body extracted from the
# dispatcher (sed range ^run_queue() .. first column-0 brace), never a Python
# re-implementation.
# ---------------------------------------------------------------------------
def _run_queue_harness(
    tmp_path: Path, alloc: list[str], jobs: list[str], barrier_n: int
) -> set[str]:
    done = tmp_path / "done"
    logs = tmp_path / "jlogs"
    qdir = done / "queue_pintest"
    done.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    jobs_file = tmp_path / "jobs.tsv"
    lines = []
    for name in jobs:
        if barrier_n > 1:
            wait_loop = (
                f'while [ "$(ls {qdir}/b_* 2>/dev/null | wc -l)" -lt {barrier_n} ]; '
                "do sleep 0.1; done"
            )
            cmd = f'touch {qdir}/b_{name}; {wait_loop}; echo "ALLOC=$CUDA_VISIBLE_DEVICES"'
        else:
            cmd = 'echo "ALLOC=$CUDA_VISIBLE_DEVICES"'
        lines.append(f"{name}\t{cmd}")
    jobs_file.write_text("\n".join(lines) + "\n")

    harness = f"""
set -euo pipefail
eval "$(sed -n '/^run_queue()/,/^}}/p' '{SCRIPT}')"
NGPU={len(alloc)}
EPS_ALLOC_GPUS=({" ".join(alloc)})
DONE_DIR='{done}'
JOB_LOG_DIR='{logs}'
run_queue pintest '{jobs_file}'
"""
    proc = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, cwd=tmp_path, timeout=120
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    pins: set[str] = set()
    for name in jobs:
        log = logs / f"pintest__{name}.log"
        m = re.search(r"ALLOC=(\S+)", log.read_text())
        assert m, f"no ALLOC line in {log}"
        pins.add(m.group(1))
    return pins


def test_run_queue_single_worker_pins_allocated_device_not_zero(tmp_path):
    """CVD=5 allocation: the one worker pins physical GPU 5 (pre-fix: pinned $w == 0)."""
    pins = _run_queue_harness(tmp_path, alloc=["5"], jobs=["job0"], barrier_n=1)
    assert pins == {"5"}


def test_run_queue_whole_node_pins_are_byte_identical_to_worker_ids(tmp_path):
    """EPS_ALLOC_GPUS=(0..7): 8 barrier-synchronized jobs realize pins {0..7} == pre-fix $w."""
    jobs = [f"j{i}" for i in range(8)]
    pins = _run_queue_harness(tmp_path, alloc=[str(i) for i in range(8)], jobs=jobs, barrier_n=8)
    assert pins == {str(i) for i in range(8)}


def test_no_literal_zero_or_w_cvd_pins_remain():
    """Class sweep: every pin site uses the allocation array (no CVD=0 / CVD=$w literals)."""
    text = SCRIPT.read_text()
    assert "CUDA_VISIBLE_DEVICES=0 " not in text
    assert "CUDA_VISIBLE_DEVICES=$w " not in text
    assert "CUDA_VISIBLE_DEVICES=${EPS_ALLOC_GPUS[w]}" in text
    assert text.count("CUDA_VISIBLE_DEVICES=${EPS_ALLOC_GPUS[0]}") == 9


def test_dispatch_script_bash_syntax():
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
