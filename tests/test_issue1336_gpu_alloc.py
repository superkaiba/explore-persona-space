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


# ---------------------------------------------------------------------------
# run_queue per-worker BLAS/OMP thread right-sizing + EPS_QUEUE_WIDTH_MAX
# (#1336 r25, measured marker v284: 8 workers x inherited OMP_NUM_THREADS=64
# on a 192-core node = 448-512 threads => 0.27x ONE worker's throughput).
# Same extraction technique as the pin harness above — the REAL run_queue
# body executes under a CONTROLLED env (thread vars scrubbed, nproc stubbed)
# so the shared-VM thread caps cannot leak into the arithmetic.
# ---------------------------------------------------------------------------
QUEUE_BANNER_RE = re.compile(
    r"\[thr\] (\d+) job\(s\) across (\d+) worker\(s\) "
    r"nproc=(\d+) threads_per_worker=(\d+) width_cap=(\S+)"
)

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _nproc_stub(tmp_path: Path, n: int) -> Path:
    """A PATH dir whose ``nproc`` prints a fixed core count."""
    d = tmp_path / f"nproc_stub_{n}"
    d.mkdir(exist_ok=True)
    stub = d / "nproc"
    stub.write_text(f"#!/usr/bin/env bash\necho {n}\n")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _run_queue_thread_proc(
    tmp_path: Path,
    n_gpus: int,
    n_jobs: int,
    *,
    nproc: int,
    inherited_omp: str | None,
    width_max: str | None = None,
) -> tuple[subprocess.CompletedProcess, Path]:
    """Execute the REAL run_queue body; jobs echo their realized thread env + CVD pin."""
    done = tmp_path / "done"
    logs = tmp_path / "jlogs"
    done.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    jobs_file = tmp_path / "jobs.tsv"
    echo_threads = " ".join(f"{k}=${{{k}:-unset}}" for k in _THREAD_ENV_KEYS)
    cmd = f'echo "THREADS {echo_threads}"; echo "ALLOC=${{CUDA_VISIBLE_DEVICES:-unset}}"'
    jobs_file.write_text("\n".join(f"tj{i}\t{cmd}" for i in range(n_jobs)) + "\n")
    alloc = [str(i) for i in range(n_gpus)]
    harness = f"""
set -euo pipefail
eval "$(sed -n '/^run_queue()/,/^}}/p' '{SCRIPT}')"
NGPU={n_gpus}
EPS_ALLOC_GPUS=({" ".join(alloc)})
DONE_DIR='{done}'
JOB_LOG_DIR='{logs}'
run_queue thr '{jobs_file}'
"""
    env = {**os.environ}
    # Scrub CVD too: a sibling test's os.environ mutation would otherwise leak
    # into the CPU-branch ALLOC=unset assertion (batch-order dependent).
    for k in (*_THREAD_ENV_KEYS, "EPS_QUEUE_WIDTH_MAX", "CUDA_VISIBLE_DEVICES"):
        env.pop(k, None)
    if inherited_omp is not None:
        env["OMP_NUM_THREADS"] = inherited_omp
    if width_max is not None:
        env["EPS_QUEUE_WIDTH_MAX"] = width_max
    env["PATH"] = f"{_nproc_stub(tmp_path, nproc)}:{env['PATH']}"
    proc = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, cwd=tmp_path, env=env, timeout=120
    )
    return proc, logs


def _job_lines(logs: Path, n_jobs: int, tag: str) -> set[str]:
    """Collect the per-job ``<tag> ...`` echo lines across all job logs."""
    vals: set[str] = set()
    for i in range(n_jobs):
        text = (logs / f"thr__tj{i}.log").read_text()
        m = re.search(rf"{tag}[= ](.+)", text)
        assert m, f"no {tag} line in thr__tj{i}.log: {text!r}"
        vals.add(m.group(1).strip())
    return vals


def _banner(proc: subprocess.CompletedProcess) -> tuple[str, ...]:
    m = QUEUE_BANNER_RE.search(proc.stdout)
    assert m, f"queue banner not found in stdout: {proc.stdout!r}"
    return m.groups()


def test_run_queue_threads_width8_yields_nproc_over_width(tmp_path):
    """(nproc=192, width=8, inherited=64) -> 24, exported to all five vars."""
    proc, logs = _run_queue_thread_proc(tmp_path, 8, 8, nproc=192, inherited_omp="64")
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    # Unset EPS_QUEUE_WIDTH_MAX leaves width == NGPU (brief test 2, unset arm).
    assert _banner(proc) == ("8", "8", "192", "24", "none")
    expected = " ".join(f"{k}=24" for k in _THREAD_ENV_KEYS)
    assert _job_lines(logs, 8, "THREADS") == {expected}


def test_run_queue_threads_width1_min_pins_inherited_not_nproc(tmp_path):
    """(nproc=192, width=1, inherited=64) -> 64: the min() is load-bearing —
    a bare nproc/width would hand the single worker 192 threads (untested);
    64 reproduces EXACTLY the measured-fast pilot configuration."""
    proc, logs = _run_queue_thread_proc(tmp_path, 1, 1, nproc=192, inherited_omp="64")
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert _banner(proc) == ("1", "1", "192", "64", "none")
    expected = " ".join(f"{k}=64" for k in _THREAD_ENV_KEYS)
    assert _job_lines(logs, 1, "THREADS") == {expected}


def test_run_queue_threads_tiny_node_floors_at_one(tmp_path):
    """(nproc=4, width=8) -> floor(4/8)=0 floored to 1."""
    proc, logs = _run_queue_thread_proc(tmp_path, 8, 8, nproc=4, inherited_omp="64")
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert _banner(proc) == ("8", "8", "4", "1", "none")
    expected = " ".join(f"{k}=1" for k in _THREAD_ENV_KEYS)
    assert _job_lines(logs, 8, "THREADS") == {expected}


def test_run_queue_threads_inherited_defaults_to_64_when_unset(tmp_path):
    """Unset OMP_NUM_THREADS => inherited defaults to 64 (never nproc/width=192)."""
    proc, _logs = _run_queue_thread_proc(tmp_path, 1, 1, nproc=192, inherited_omp=None)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert _banner(proc) == ("1", "1", "192", "64", "none")


def test_run_queue_threads_cpu_branch_also_exports(tmp_path):
    """NGPU=0 (CPU-only branch): width=1, threads exported there too."""
    proc, logs = _run_queue_thread_proc(tmp_path, 0, 1, nproc=192, inherited_omp="64")
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert _banner(proc) == ("1", "1", "192", "64", "none")
    expected = " ".join(f"{k}=64" for k in _THREAD_ENV_KEYS)
    assert _job_lines(logs, 1, "THREADS") == {expected}
    assert _job_lines(logs, 1, "ALLOC") == {"unset"}


def test_run_queue_width_max_clamps_width_down(tmp_path):
    """EPS_QUEUE_WIDTH_MAX=2 on NGPU=8: 2 workers realized (pins subset {0,1}),
    banner reports the applied cap, threads follow the REALIZED width
    (min(192/2, 64) = 64 — the inherited cap binds)."""
    proc, logs = _run_queue_thread_proc(
        tmp_path, 8, 8, nproc=192, inherited_omp="64", width_max="2"
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert _banner(proc) == ("8", "2", "192", "64", "EPS_QUEUE_WIDTH_MAX=2")
    assert _job_lines(logs, 8, "ALLOC") <= {"0", "1"}


def test_run_queue_width_max_at_or_above_width_is_inert(tmp_path):
    """A cap >= the computed width leaves behaviour unchanged (cap 'none')."""
    proc, _logs = _run_queue_thread_proc(
        tmp_path, 8, 8, nproc=192, inherited_omp="64", width_max="16"
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert _banner(proc) == ("8", "8", "192", "24", "none")


def test_run_queue_width_max_nonpositive_is_inert(tmp_path):
    """0 / negative integer caps are ignored (current behaviour byte-for-byte)."""
    for cap in ("0", "-3"):
        proc, _logs = _run_queue_thread_proc(
            tmp_path, 8, 8, nproc=192, inherited_omp="64", width_max=cap
        )
        assert proc.returncode == 0, (cap, proc.stdout, proc.stderr)
        assert _banner(proc) == ("8", "8", "192", "24", "none"), cap


def test_run_queue_width_max_malformed_fails_loud(tmp_path):
    """A malformed cap fails LOUD (non-zero rc + naming stderr), never a silent
    fall-through into the uncapped (slow) configuration."""
    for bad in ("banana", "3.5", "8x", "-"):
        proc, _logs = _run_queue_thread_proc(
            tmp_path, 8, 8, nproc=192, inherited_omp="64", width_max=bad
        )
        assert proc.returncode != 0, (bad, proc.stdout, proc.stderr)
        assert "EPS_QUEUE_WIDTH_MAX" in proc.stderr, (bad, proc.stderr)
        assert "not an integer" in proc.stderr, (bad, proc.stderr)
        # Fail-loud means NO workers ran.
        assert QUEUE_BANNER_RE.search(proc.stdout) is None, (bad, proc.stdout)
