"""Issue #2658 P1 launcher — rc collection, CVD pinning, gate, smoke wiring.

The launcher's single most important property is per-shard rc collection: a
bare ``... & done; wait`` swallows child exit codes and turns a failed shard
into a silently short manifest. These tests run the REAL launcher end-to-end
(bash, real completeness-gate python) against a fake ``uv`` that substitutes
ONLY the two GPU scripts (signature-conformant: it parses the same flags and
writes the same artifact shapes the real scripts write) and the HF upload leg
(external network boundary).

No GPU, no network, no repo-root writes (out-root + logs land in tmp_path).
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "issue2658_p1_launch.sh"

_FAKE_UV = '''#!/usr/bin/env python3
"""Fake ``uv`` for launcher tests: substitutes the two GPU scripts + the
upload heredoc; delegates every other ``python -`` heredoc (the stdlib-only
completeness gate) to the real interpreter."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

log = Path(os.environ["FAKE_UV_LOG"])


def logline(s: str) -> None:
    with log.open("a") as f:
        f.write(s + "\\n")


argv = sys.argv[1:]  # expected: run python <target> [args...]
target = argv[2] if len(argv) >= 3 else ""
rest = argv[3:]
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")


def val(flag: str, default: str | None = None) -> str | None:
    return rest[rest.index(flag) + 1] if flag in rest else default


if target.endswith("issue2658_generate.py"):
    s, n = int(val("--shard-index", "0")), int(val("--num-shards", "1"))
    out, split = Path(val("--out-root")), val("--split", "pilot")
    smoke = "--smoke" in rest
    logline(f"GEN cvd={cvd} shard={s}of{n} smoke={smoke} args={' '.join(rest)}")
    if os.environ.get("FAKE_GEN_FAIL_SHARD") == str(s):
        sys.exit(3)
    if smoke:
        out = out / "smoke_gen"
    cells = os.environ.get("FAKE_CELLS", "cell_a cell_b cell_c cell_d").split()
    if smoke:
        cells = cells[: int(val("--smoke-cells", str(n)))]
    mine = cells[s::n]
    tag = f"shard{s:02d}of{n:02d}"
    (out / "gen_order_manifest").mkdir(parents=True, exist_ok=True)
    (out / "gen_order_manifest" / f"{split}_{tag}.json").write_text(
        json.dumps({"cell_order": mine})
    )
    omit = os.environ.get("FAKE_GEN_OMIT_CELL")
    for c in mine:
        for sub, name, payload in (
            ("raw_completions", f"{c}.json", "{}"),
            ("gen_manifest", f"{c}.jsonl", "x\\n"),
        ):
            if sub == "raw_completions" and c == omit:
                continue
            p = out / sub / split / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(payload)
    (out / "gen_summary").mkdir(parents=True, exist_ok=True)
    (out / "gen_summary" / f"{split}_{tag}.json").write_text("{}")
    sys.exit(0)

if target.endswith("issue2658_capture.py"):
    s, n = int(val("--shard-index", "0")), int(val("--num-shards", "1"))
    smoke = "--smoke" in rest
    logline(
        f"CAPTURE cvd={cvd} shard={s}of{n} smoke={smoke} "
        f"upload={'--upload' in rest} args={' '.join(rest)}"
    )
    if os.environ.get("FAKE_CAPTURE_FAIL_SHARD") == str(s):
        sys.exit(5)
    sys.exit(0)

if target == "-":
    snippet = sys.stdin.read()
    if "upload_raw" in snippet:
        logline(f"UPLOAD_GEN args={' '.join(rest)}")
        sys.exit(int(os.environ.get("FAKE_UPLOAD_RC", "0")))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(snippet)
        path = f.name
    sys.exit(subprocess.run([sys.executable, path, *rest]).returncode)

logline(f"UNEXPECTED args={' '.join(argv)}")
sys.exit(97)
'''


@pytest.fixture()
def rig(tmp_path):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake_uv = fakebin / "uv"
    fake_uv.write_text(_FAKE_UV)
    fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    out_root = tmp_path / "out"
    log_dir = tmp_path / "logs"
    fake_log = tmp_path / "fake_uv.log"
    fake_log.touch()
    env = {
        **os.environ,
        "PATH": f"{fakebin}:{os.environ['PATH']}",
        "FAKE_UV_LOG": str(fake_log),
        "EPS_P1_OUT_ROOT": str(out_root),
        "EPS_P1_LOG_DIR": str(log_dir),
        "CUDA_VISIBLE_DEVICES": "0,1",
    }
    for stale in (
        "SLURM_JOB_ID",
        "EPS_P1_NUM_SHARDS",
        "FAKE_GEN_FAIL_SHARD",
        "FAKE_GEN_OMIT_CELL",
        "FAKE_CAPTURE_FAIL_SHARD",
    ):
        env.pop(stale, None)
    return {"env": env, "out_root": out_root, "log_dir": log_dir, "fake_log": fake_log}


def run_launcher(rig, *args, extra_env=None, launcher=LAUNCHER):
    env = {**rig["env"], **(extra_env or {})}
    return subprocess.run(
        ["bash", str(launcher), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )


def fake_lines(rig):
    return rig["fake_log"].read_text().splitlines()


def test_happy_path_pins_each_shard_and_sequences_upload(rig):
    proc = run_launcher(rig)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = fake_lines(rig)
    gens = [ln for ln in lines if ln.startswith("GEN ")]
    caps = [ln for ln in lines if ln.startswith("CAPTURE ")]
    ups = [ln for ln in lines if ln.startswith("UPLOAD_GEN ")]
    # One gen + one capture leg per allocated GPU, each pinned to ITS id.
    assert any("cvd=0 shard=0of2" in ln for ln in gens)
    assert any("cvd=1 shard=1of2" in ln for ln in gens)
    assert any("cvd=0 shard=0of2" in ln for ln in caps)
    assert any("cvd=1 shard=1of2" in ln for ln in caps)
    # Exactly ONE sequenced raw-completions upload, between gen and capture.
    assert len(ups) == 1
    assert lines.index(ups[0]) > max(lines.index(g) for g in gens)
    assert lines.index(ups[0]) < min(lines.index(c) for c in caps)
    # Capture legs carry --upload (per-shard disjoint store dirs).
    assert all("upload=True" in ln for ln in caps)
    assert "[phase=done]" in proc.stdout
    assert "generation completeness gate PASS: 4 cells" in proc.stdout
    # Re-attach breadcrumbs.
    assert (rig["log_dir"] / "launcher.pid").read_text().strip().isdigit()
    for leg in ("generate_shard00", "generate_shard01", "capture_shard00"):
        assert (rig["log_dir"] / f"{leg}.pid").exists()
        assert (rig["log_dir"] / f"{leg}.log").exists()


def test_failing_gen_shard_fails_loud_named_and_blocks_capture(rig):
    proc = run_launcher(rig, extra_env={"FAKE_GEN_FAIL_SHARD": "1"})
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "FAIL: generate shard 1" in combined
    assert "rc=3" in combined
    lines = fake_lines(rig)
    assert not any(ln.startswith("CAPTURE ") for ln in lines)
    assert not any(ln.startswith("UPLOAD_GEN ") for ln in lines)
    assert "[phase=done]" not in proc.stdout


def test_missing_cell_file_trips_gate_before_capture(rig):
    proc = run_launcher(rig, extra_env={"FAKE_GEN_OMIT_CELL": "cell_b"})
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "GENERATION INCOMPLETE" in combined
    assert "cell_b.json" in combined
    lines = fake_lines(rig)
    assert not any(ln.startswith("CAPTURE ") for ln in lines)


def test_failing_capture_shard_fails_loud_named(rig):
    proc = run_launcher(rig, extra_env={"FAKE_CAPTURE_FAIL_SHARD": "0"})
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "capture shard 0" in combined
    assert "rc=5" in combined
    # Raw-completions upload already landed (sequenced before capture).
    assert any(ln.startswith("UPLOAD_GEN ") for ln in fake_lines(rig))


def test_smoke_mode_passes_smoke_flags_and_never_uploads(rig):
    proc = run_launcher(rig, "--smoke")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = fake_lines(rig)
    gens = [ln for ln in lines if ln.startswith("GEN ")]
    caps = [ln for ln in lines if ln.startswith("CAPTURE ")]
    assert gens and caps
    assert all("smoke=True" in ln for ln in gens + caps)
    assert all("--smoke-cells 2" in ln for ln in gens)  # one cell per shard
    assert all("--responses 2" in ln for ln in gens + caps)
    assert all("upload=False" in ln for ln in caps)
    assert not any(ln.startswith("UPLOAD_GEN ") for ln in lines)
    # Gate ran against the smoke_gen root the scripts rebind to.
    assert (rig["out_root"] / "smoke_gen" / "gen_order_manifest").exists()


def test_width_override_narrows_allocation(rig):
    proc = run_launcher(rig, extra_env={"EPS_P1_NUM_SHARDS": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    gens = [ln for ln in fake_lines(rig) if ln.startswith("GEN ")]
    assert len(gens) == 1
    assert "cvd=0 shard=0of1" in gens[0]


def test_width_override_beyond_allocation_refused(rig):
    proc = run_launcher(rig, extra_env={"EPS_P1_NUM_SHARDS": "3"})
    assert proc.returncode == 2
    assert "exceeds the realized allocation" in proc.stdout + proc.stderr
    assert not fake_lines(rig)  # nothing launched


def test_rc_collection_is_load_bearing_bare_wait_variant_fails_this_suite(rig, tmp_path):
    """Mutation guard: a launcher variant whose wait loop discards per-shard
    rcs (the banned ``wait "$pid" || true`` shape) must NOT pass the failing-
    shard scenario — proving the rc assertions above actually bind."""
    src = LAUNCHER.read_text()
    broken = src.replace('wait "$pid" || rc=$?', 'wait "$pid" || true')
    assert broken != src  # the load-bearing line exists
    mutdir = tmp_path / "mut" / "scripts"
    mutdir.mkdir(parents=True)
    mutant = mutdir / "issue2658_p1_launch.sh"
    mutant.write_text(broken)
    proc = run_launcher(rig, extra_env={"FAKE_GEN_FAIL_SHARD": "1"}, launcher=mutant)
    # The mutant swallows the shard rc: the FAIL line naming the shard (and
    # the rc=3 exit it drives) must disappear. The real launcher (test above)
    # emits "FAIL: generate shard 1" and exits 3 — if the mutant also did,
    # this mutation guard would be vacuous. (Defense in depth: the mutant is
    # still caught downstream by the completeness GATE seeing missing cells,
    # so we assert only on the rc-collection behavior, not the final rc.)
    combined = proc.stdout + proc.stderr
    assert "FAIL: generate shard 1" not in combined
    assert "rc=3" not in combined


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
