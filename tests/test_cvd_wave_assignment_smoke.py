"""CVD-clobber regression smoke: sweep waves must land on cvd 0/1/2/3 (#578).

Incident #523 Phase B (and the same class in #541/#543/#557): all 4 parallel
wave cells piled onto physical GPU 0 and OOM'd. Root cause per
``.claude/rules/gotchas.md``: ``sft.py``'s in-process
``os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)`` clobber is silently
defeated by any import-time cuInit (the driver freezes its device list at the
FIRST cuInit in the process), so per-GPU parallel launches must ALSO pin
``CUDA_VISIBLE_DEVICES=<gpu>`` in the LAUNCHER environment, with a matching
``--gpu-id`` so the in-process clobber rewrites the same value.

This is a cheap no-GPU dry-run of the REAL wave-assignment logic: it executes
``scripts/i474_phase23_dispatch.sh`` (the wave dispatcher the #523 Phase B run
wrapped) with a stub ``uv`` that records each launch's CUDA_VISIBLE_DEVICES +
argv instead of training, then asserts:

1. every wave cell gets a distinct cvd 0..3 (cells 2-4 on cvd 1/2/3 — NOT all
   on GPU 0, the #523 regression signature);
2. the launcher env CUDA_VISIBLE_DEVICES matches the ``--gpu-id`` arg for
   every cell (the gotchas.md launcher-env pin).

The stub wins the PATH race because the dispatcher itself prepends
``$HOME/.local/bin`` and the test points HOME at a temp dir. Runs in ~2s.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DISPATCHER = REPO_ROOT / "scripts" / "i474_phase23_dispatch.sh"

# The dispatcher's --skip-smoke wave plan (A1 stays in wave 1). cvd is the
# cell's index within its wave.
SKIP_SMOKE_WAVES = [
    ["A1", "A2", "A3", "A4"],
    ["A5", "B1", "B2", "B3"],
    ["B4", "B5", "C1", "D1"],
    ["D2", "D3", "D4", "D5"],
]
EXPECTED_CVD = {cond: idx for wave in SKIP_SMOKE_WAVES for idx, cond in enumerate(wave)}

UV_STUB = """#!/usr/bin/env bash
# Test stub: record launcher env CVD + argv instead of running anything.
echo "CVD=${CUDA_VISIBLE_DEVICES:-UNSET}|ARGS=$*" >> "$CVD_CAPTURE_FILE"
exit 0
"""


def _run_dispatcher_dry(tmp_path: Path) -> list[str]:
    """Run the real dispatcher with a stubbed ``uv``; return capture lines."""
    home = tmp_path / "home"
    stub_bin = home / ".local" / "bin"
    stub_bin.mkdir(parents=True)
    stub = stub_bin / "uv"
    stub.write_text(UV_STUB)
    stub.chmod(0o755)

    capture = tmp_path / "launches.txt"
    capture.touch()

    cwd = tmp_path / "cwd"
    cwd.mkdir()

    env = os.environ.copy()
    env["HOME"] = str(home)
    env["CVD_CAPTURE_FILE"] = str(capture)
    # The recorded CVD must come from the per-cell launcher pin, not from an
    # ambient value leaking in from the test environment.
    env.pop("CUDA_VISIBLE_DEVICES", None)

    result = subprocess.run(
        ["bash", str(DISPATCHER), "--skip-smoke", "--arm=pos"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"dispatcher dry-run failed rc={result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-3000:]}"
    )
    return capture.read_text().splitlines()


def _parse_train_launches(lines: list[str]) -> dict[str, dict[str, str]]:
    """cond -> {"cvd": <launcher env CVD>, "gpu_id": <--gpu-id arg>}."""
    launches: dict[str, dict[str, str]] = {}
    for line in lines:
        if "i474_phase23_train.py" not in line:
            continue  # marker-assert preflight etc.
        m = re.match(r"CVD=(?P<cvd>[^|]*)\|ARGS=(?P<args>.*)", line)
        assert m, f"unparseable capture line: {line!r}"
        args = m.group("args")
        cond = re.search(r"--conds (\S+)", args)
        gpu = re.search(r"--gpu-id (\S+)", args)
        assert cond and gpu, f"train launch missing --conds/--gpu-id: {line!r}"
        assert cond.group(1) not in launches, f"cond launched twice: {cond.group(1)}"
        launches[cond.group(1)] = {"cvd": m.group("cvd"), "gpu_id": gpu.group(1)}
    return launches


def test_sweep_waves_land_on_cvd_0_1_2_3(tmp_path):
    launches = _parse_train_launches(_run_dispatcher_dry(tmp_path))

    assert sorted(launches) == sorted(EXPECTED_CVD), (
        f"expected one train launch per condition; got {sorted(launches)}"
    )

    # 1. Launcher-env pin (gotchas.md): env CUDA_VISIBLE_DEVICES must be set
    #    and match --gpu-id, so an import-time cuInit cannot silently re-route
    #    every cell to physical GPU 0. Checked FIRST so a missing pin fails
    #    with this message rather than a parse error in the wave check below.
    for cond, rec in launches.items():
        assert rec["cvd"] != "UNSET", (
            f"{cond}: CUDA_VISIBLE_DEVICES not pinned in the launcher env — the "
            f"in-process clobber alone is defeated by import-time cuInit (#523/#543)"
        )
        assert rec["cvd"] == rec["gpu_id"], (
            f"{cond}: launcher env CVD={rec['cvd']} disagrees with --gpu-id "
            f"{rec['gpu_id']} — the two must pin the SAME physical GPU"
        )

    # 2. Per-cell assignment: each cell's slot is its index within the wave —
    #    so each wave spreads over cvd {0,1,2,3}, never all on GPU 0.
    for cond, rec in launches.items():
        assert rec["gpu_id"] == str(EXPECTED_CVD[cond]), (
            f"{cond}: --gpu-id {rec['gpu_id']} != expected wave slot {EXPECTED_CVD[cond]}"
        )

    for wave in SKIP_SMOKE_WAVES:
        wave_cvds = sorted(int(launches[c]["cvd"]) for c in wave)
        assert wave_cvds == [0, 1, 2, 3], (
            f"wave {wave} landed on cvds {wave_cvds} — the #523 regression is every "
            f"cell on GPU 0; cells 2-4 must land on cvd 1/2/3"
        )
