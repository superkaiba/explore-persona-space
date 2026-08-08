"""Tests for the #1739 pvsynth arm-scoring instance driver (interleave contract).

The driver's value is the stage/score interleave, so these pin the waiter's
three outcomes (manifest lands / stager died / fence expires) and the composed
per-behavior scorer argv. Real bodies execute; only the external boundaries
(subprocess, HF) are faked, signature-conformant via ``create_autospec``.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import issue1739_pvsynth_arms_run as run  # noqa: E402


def _args(tmp_path: Path, **over):
    argv = [
        "--store-root",
        str(tmp_path / "stage"),
        "--main-root",
        str(tmp_path / "main"),
        "--out-root",
        str(tmp_path / "pvsynth"),
        "--tensors-root",
        str(tmp_path / "tensors"),
        *[x for k, v in over.items() for x in (f"--{k.replace('_', '-')}", str(v))],
    ]
    return run.parse_args(argv)


def test_stage_order_is_smallest_tar_first():
    """Scoring starts sooner when the smallest tar streams first."""
    sizes = [run.TAR_GIB[b] for b in run.STAGE_ORDER]
    assert sizes == sorted(sizes), f"{run.STAGE_ORDER} is not smallest-first: {sizes}"
    assert set(run.STAGE_ORDER) == {"evil", "sycophancy", "hallucination"}


def test_train_dv_root_defaults_under_store_root(tmp_path):
    args = _args(tmp_path)
    assert args.train_dv_root == tmp_path / "stage" / "train_dv"


def test_wait_for_stage_returns_when_manifest_lands(tmp_path):
    args = _args(tmp_path)
    manifest = args.store_root / "evil_labeling" / run.STAGE_MANIFEST
    manifest.parent.mkdir(parents=True)

    def _land():
        time.sleep(0.3)
        manifest.write_text("{}")

    threading.Thread(target=_land, daemon=True).start()
    run.wait_for_stage("evil", args, [])  # returns without raising
    assert manifest.exists()


def test_wait_for_stage_raises_when_stager_died(tmp_path):
    args = _args(tmp_path)
    errors: list[BaseException] = [RuntimeError("tar stream broke")]
    with pytest.raises(RuntimeError, match="staging thread died before evil"):
        run.wait_for_stage("evil", args, errors)


def test_wait_for_stage_times_out(tmp_path):
    args = _args(tmp_path, stage_timeout_s=0)
    with pytest.raises(TimeoutError, match="staging did not complete"):
        run.wait_for_stage("evil", args, [])


def test_score_behavior_composes_per_behavior_argv(tmp_path, monkeypatch):
    """The scorer subprocess gets ONE behavior plus the per-behavior roots."""
    args = _args(tmp_path)
    args.child_env = {"HF_TOKEN": "x"}
    seen: dict = {}
    fake = create_autospec(subprocess.run)
    fake.return_value = subprocess.CompletedProcess(args=[], returncode=0)

    def _capture(cmd, **kw):
        seen["cmd"] = list(cmd)
        seen["kw"] = kw
        return fake.return_value

    monkeypatch.setattr(run.subprocess, "run", _capture)
    rc = run.score_behavior("sycophancy", args)
    assert rc == 0
    cmd = seen["cmd"]
    assert cmd[1].endswith("issue1739_pvsynth_arms.py")
    assert cmd[cmd.index("--behaviors") + 1] == "sycophancy"
    assert "evil" not in cmd, "a multi-behavior argv would trip the override guard"
    assert cmd[cmd.index("--store-root") + 1] == str(args.store_root)
    assert cmd[cmd.index("--train-dv-root") + 1] == str(args.train_dv_root)
    assert cmd[cmd.index("--n-layers") + 1] == "28"
    assert seen["kw"]["env"] is args.child_env  # explicit env passthrough
    assert seen["kw"]["check"] is False


def test_score_behavior_propagates_nonzero_rc(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.child_env = {}
    monkeypatch.setattr(
        run.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(args=cmd, returncode=2),
    )
    assert run.score_behavior("evil", args) == 2


def test_upload_behavior_fails_loud_on_missing_outputs(tmp_path):
    args = _args(tmp_path)
    with pytest.raises(FileNotFoundError, match="nothing to upload for evil"):
        run.upload_behavior("evil", args)


def test_upload_behavior_targets_the_pvsynth_arm_results_prefix(tmp_path, monkeypatch):
    args = _args(tmp_path)
    (args.out_root / "evil").mkdir(parents=True)
    (args.out_root / "evil" / "all_arms_spearman.json").write_text("{}")
    from explore_persona_space.orchestrate import hub

    calls: list[tuple] = []
    real = create_autospec(hub._upload)
    real.side_effect = lambda *a, **k: calls.append((a, k)) or ""
    monkeypatch.setattr(hub, "_upload", real)
    run.upload_behavior("evil", args)
    (a, k) = calls[0]
    assert a[0] == args.out_root / "evil"
    assert a[2] == "dataset"
    assert a[3] == "issue1739_ctxmap/pvsynth/arm_results/evil"
    assert k["raise_on_error"] is True


def test_import_check_exits_zero():
    with pytest.raises(SystemExit) as exc:
        run.main(["--import-check"])
    assert int(exc.value.code or 0) == 0
