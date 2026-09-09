"""Regression checks for the original pilot gate and at-most-once cloud dispatch."""

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/issue2588_k3_resume.py"
SPEC = importlib.util.spec_from_file_location("k3_resume", SCRIPT)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def snapshot():
    parts = {}
    stages = {}
    for k in range(4):
        name = f"train_10k_s45_part{k}"
        parts[name] = {
            "report": {"meta": {"git_sha": R.PILOT_SHA}, "n": 2500},
            "chunks": [
                {
                    "meta": {"git_sha": R.PILOT_SHA},
                    "cell": R.PILOT,
                    "stage": name,
                    "seed": 45,
                    "n": 2500,
                    "ids": [f"{name}_{i}" for i in range(2500)],
                }
            ],
        }
        stages[name] = {
            "stage": name,
            "seed": 45,
            "n_rows": 2500,
            "generated": True,
            "wall_s": 4000,
        }
    return {
        "gate": {"pass": True, "meta": {"git_sha": R.PILOT_SHA}},
        "gpu": "NVIDIA A100-SXM4-80GB, 81920",
        "parts": parts,
        "throughput": {"meta": {"git_sha": R.PILOT_SHA}, "stages": stages},
    }


def test_full_draw_gate_and_inclusive_timing():
    data = snapshot()
    result = R.pilot_basis(data)
    assert result["n_rows"] == 10000
    assert result["wall_s"] == 16000
    assert result["responses_per_s"] == 0.625
    assert result["generation_h_per_cell_at_pilot_rate"] == pytest.approx(20800 * 1.6 / 3600)
    data["parts"].pop("train_10k_s45_part3")
    assert R.pilot_basis(data) is None


@pytest.mark.parametrize("damage", ["source", "gpu", "duplicate", "count", "seed", "nan", "gate"])
def test_bad_pilot_evidence_cannot_open_gate(damage):
    data = snapshot()
    stage = "train_10k_s45_part0"
    if damage == "source":
        data["parts"][stage]["chunks"][0]["meta"]["git_sha"] = "old"
    elif damage == "gpu":
        data["gpu"] = "NVIDIA H200, 143771"
    elif damage == "duplicate":
        data["parts"][stage]["chunks"][0]["ids"][1] = data["parts"][stage]["chunks"][0]["ids"][0]
    elif damage == "count":
        data["parts"][stage]["report"]["n"] = 2499
    elif damage == "seed":
        data["throughput"]["stages"][stage]["seed"] = 42
    elif damage == "nan":
        data["throughput"]["stages"][stage]["wall_s"] = float("nan")
    elif damage == "gate":
        data["gate"]["pass"] = False
    with pytest.raises(ValueError):
        R.pilot_basis(data)


def test_json_cli_stream_rejects_non_json_noise():
    assert R.records('{"marker": 1}\n {"ok": true}\n') == [{"marker": 1}, {"ok": True}]
    with pytest.raises(json.JSONDecodeError):
        R.records('{"ok": true}\nwarning')


@pytest.fixture
def owner(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    args = argparse.Namespace(
        repo=repo,
        state_dir=state,
        source_sha="a" * 40,
        pilot_handle=tmp_path / "pilot.json",
        interval=300,
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    # Real subprocesses exercise Owner.command output files, argv, state writes,
    # and CLI stream parsing. Only the external cloud/git/task executables differ.
    stub = (
        f"#!{sys.executable}\n"
        + """
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
state = Path(os.environ['K3_TEST_STATE'])
if name == 'git':
    print('a' * 40 + '\\trefs/heads/issue-2588-k3refit')
elif name == 'gcloud':
    print(json.dumps({'quotas': [{'metric': 'PREEMPTIBLE_NVIDIA_A100_80GB_GPUS',
                                'limit': 16, 'usage': 0}]}))
elif 'dispatch_issue.py' in ' '.join(args):
    saved = json.loads((state / 'state.json').read_text())
    assert saved['launching'] == 'q35_2b_a'
    suffix = args[args.index('--lane-suffix') + 1]
    assert args[args.index('--gpus') + 1] == '1'
    assert '--no-runpod-fallback' in args
    cache = Path.cwd() / '.claude/cache'
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f'issue-2588-{suffix}-handle.json').write_text(json.dumps({
        'backend': 'gcp', 'extra': {'gpu_count': 1, 'gcp_launched_ts': 123.0},
        'pod_name': 'owned', 'job_id': '123'}))
    if os.environ.get('K3_TEST_INTERRUPT'):
        raise SystemExit(7)
    print(json.dumps({'marker': 'selected'}))
    print(json.dumps({'ok': True}))
else:
    print(json.dumps({'marker': 'progress'}))
"""
    )
    for name in ("git", "gcloud", "uv"):
        file = bindir / name
        file.write_text(stub)
        file.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("K3_TEST_STATE", str(state))
    return R.Owner(args)


def test_launch_real_body_journals_before_submission(owner):
    assert owner.launch("q35_2b_a") is True
    saved = json.loads(owner.state_path.read_text())
    assert "q35_2b_a" in saved["launches"]
    assert "launching" not in saved
    with pytest.raises(RuntimeError, match="existing handle"):
        owner.launch("q35_2b_a")


def test_interrupted_submission_never_relaunches(owner):
    owner.env["K3_TEST_INTERRUPT"] = "1"
    with pytest.raises(R.subprocess.CalledProcessError):
        owner.launch("q35_2b_a")
    restored = R.Owner(copy.copy(owner.args))
    assert restored.state["launching"] == "q35_2b_a"
    with pytest.raises(RuntimeError, match="previous submission incomplete"):
        restored.launch("q35_2b_a")


def test_capture_manifest_requires_every_shard():
    manifest = {
        "meta": {"git_sha": R.PILOT_SHA},
        "cell": R.PILOT,
        "stage": "train",
        "rows": [{"row_id": str(i)} for i in range(2491)],
    }
    required = R.capture_files(
        manifest, prefix="capture/train/L50/", cell=R.PILOT, stage="train", source=R.PILOT_SHA
    )
    assert required == {f"capture/train/L50/shard{k:03d}.npz" for k in range(5)}
    assert required != {"capture/train/L50/shard000.npz"}
    manifest["meta"]["git_sha"] = "stale"
    with pytest.raises(ValueError, match="provenance"):
        R.capture_files(
            manifest, prefix="capture/train/L50/", cell=R.PILOT, stage="train", source=R.PILOT_SHA
        )
