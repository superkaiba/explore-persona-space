"""Observe the turn-12 CPU fit and verify its exact-source archived completion."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from explore_persona_space.orchestrate.hub import retry_transient  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue825_turn12_matched_20260915"
BOOT_PHASES = {"startup", "running", "pending", "provisioning", "staging", "booting", "bootstrap"}


def validate_inventory(result):
    """Refuse completion proofs that omit an entire output family or fit cell."""
    if set(result["archives"]) != {"tensor", "text", "tracking"}:
        raise ValueError("Completion must cover numerical, text and offline tracking outputs")
    required = {f"{model}_oof.npz" for model in ("instruct", "pretrained")}
    required.update(
        f"maps/{model}_12_folds{f}-{f + 1}.npz"
        for model in ("instruct", "pretrained")
        for f in (0, 2, 4)
    )
    required.update(
        f"predictions/{model}_12_fold{f}.npz"
        for model in ("instruct", "pretrained")
        for f in range(6)
    )
    numerical = result["archives"]["tensor"]
    if set(numerical["files"]) != {PREFIX + "/numerical/" + p for p in required}:
        raise ValueError("Numerical archive must contain all 20 expected files")
    if PREFIX + "/analysis/results.json" not in result["archives"]["text"]["files"]:
        raise ValueError("Results JSON absent from text receipt")
    for kind, receipt in result["archives"].items():
        repo, repo_type, suffix = (
            ("superkaiba1/explore-persona-space-overflow", "model", "numerical")
            if kind == "tensor"
            else (REPO, "dataset", "analysis" if kind == "text" else "tracking")
        )
        if (receipt["repo"], receipt["repo_type"], receipt["prefix"]) != (
            repo,
            repo_type,
            PREFIX + "/" + suffix,
        ):
            raise ValueError("Unexpected archive destination")
        if not receipt["files"] or receipt["count"] != len(receipt["files"]):
            raise ValueError("Archive inventory count is empty or inconsistent")


def validate_results(value, source_sha):
    """Prove the published numbers cover the requested cohort and every target."""
    if value["source_sha"] != source_sha or value["source_conditions"] != ["12"]:
        raise ValueError("Archived results have a different source or condition")
    if value["n_conversations"] != 4975 or value["answer_draws"] != 1:
        raise ValueError("Archived results have a different cohort size or target")
    if set(value["models"]) != {"instruct", "pretrained"}:
        raise ValueError("Archived results lack both checkpoints")
    for model in value["models"].values():
        expected = {("12", m, t) for m in ("raw", "source_identity_bias") for t in range(1, 13)}
        cells = [(c["source"], c["method"], c["target_turn"]) for c in model["cells"]]
        if len(cells) != 24 or set(cells) != expected:
            raise ValueError("Archived results have missing or duplicate cells")


def write_state(path, value):
    """Publish one complete observation atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verified_completion(source_sha):
    """Verify the immutable completion and every promised archived file's metadata."""
    api = HfApi()
    revision = retry_transient(
        lambda: api.repo_info(REPO, repo_type="dataset").sha, what="turn12 completion revision"
    )
    remote = PREFIX + "/runtime/complete.json"
    found = retry_transient(
        lambda: api.get_paths_info(REPO, [remote], repo_type="dataset", revision=revision),
        what="turn12 completion lookup",
    )
    if not found:
        return None
    path = retry_transient(
        lambda: hf_hub_download(REPO, remote, repo_type="dataset", revision=revision),
        what="turn12 completion read",
    )
    result = json.loads(Path(path).read_text())
    if result["source_sha"] != source_sha or result["status"] != "complete":
        raise ValueError("Turn12 completion belongs to a different source or is incomplete")
    validate_inventory(result)
    for receipt in result["archives"].values():
        expected = receipt["files"]
        if not expected or receipt["status"] != "verified":
            raise ValueError("Empty or unverified archive receipt")
        entries = retry_transient(
            lambda r=receipt, e=expected: api.get_paths_info(
                r["repo"], list(e), repo_type=r["repo_type"], revision=r["revision"]
            ),
            what="turn12 archive metadata verification",
        )
        actual = {entry.path: entry for entry in entries}
        if set(actual) != set(expected):
            raise ValueError("Turn12 archive is missing promised files")
        for name, record in expected.items():
            entry = actual[name]
            if entry.size != record["size"]:
                raise ValueError(f"Archive size mismatch: {name}")
            if entry.lfs:
                digest = entry.lfs.sha256
            else:
                item = retry_transient(
                    lambda r=receipt, n=name: hf_hub_download(
                        r["repo"], n, repo_type=r["repo_type"], revision=r["revision"]
                    ),
                    what="turn12 text hash verification",
                )
                digest = hashlib.sha256(Path(item).read_bytes()).hexdigest()
            if digest != record["sha256"]:
                raise ValueError(f"Archive digest mismatch: {name}")
    receipt = result["archives"]["text"]
    path = retry_transient(
        lambda: hf_hub_download(
            REPO,
            PREFIX + "/analysis/results.json",
            repo_type="dataset",
            revision=receipt["revision"],
        ),
        what="turn12 result identity verification",
    )
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != result["results_sha256"]:
        raise ValueError("Completion and results digest disagree")
    validate_results(json.loads(raw), source_sha)
    return dict(result, verified_revision=revision)


def probe(handle_path):
    """Check the actual backend, then count its saved fit and score checkpoints."""
    from explore_persona_space.backends.base import RunHandle
    from explore_persona_space.backends.gcp import GcpBackend

    handle = RunHandle(**json.loads(handle_path.read_text()))
    if handle.backend != "gcp":
        raise ValueError("This CPU-only monitor expects the pinned GCP lane")
    observed = asdict(GcpBackend().poll(handle))
    if observed["status"] == "running" and observed["current_phase"] not in BOOT_PHASES:
        code = f"""import pathlib,json,time
p=pathlib.Path('/workspace/{PREFIX}/analysis')
files=list(p.glob('maps/*.json'))+list(p.glob('scores/*.json'))
pids=[]
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: argv=(proc/'cmdline').read_bytes().split(b'\\0')
    except (FileNotFoundError, ProcessLookupError): continue
    if any(a == b'scripts/issue825_turn_matched.py' or a.endswith(b'/scripts/issue825_turn_matched.py') for a in argv): pids.append(int(proc.name))
print(json.dumps(dict(fits=len(list(p.glob('maps/*.json'))),scores=len(list(p.glob('scores/*.json'))),newest=max((f.stat().st_mtime for f in files),default=None),checked_at=time.time(),worker_pids=pids)))
"""
        result = subprocess.run(
            [
                "gcloud",
                "--configuration=eps-gcp",
                "compute",
                "ssh",
                handle.pod_name,
                "--zone=" + handle.extra["zone"],
                "--quiet",
                "--command",
                "sudo -n python3 -c " + shlex.quote(code),
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=90,
        )
        observed["output_progress"] = json.loads(result.stdout.splitlines()[-1])
        observed["pid_alive"] = bool(observed["output_progress"]["worker_pids"])
    return observed


def check_progress(state, observed, now):
    """Bound startup, first output, numeric progress and post-score archival separately."""
    phase = observed["current_phase"]
    if observed["status"] == "running" and phase in BOOT_PHASES:
        if now - state["started_at"] > 1200:
            raise RuntimeError("Provisioning/bootstrap exceeded 20 minutes")
        return dict(status="pending", current_phase=phase, startup_observation=observed)
    if observed["status"] != "running":
        return observed
    began = state.setdefault("workload_observed_at", now)
    progress = observed["output_progress"]
    if not observed["pid_alive"]:
        if now - began < 90 and not progress["fits"]:
            observed = dict(observed, status="pending")
            observed.pop("pid_alive")
            return observed
        raise RuntimeError("CPU worker process is absent")
    if progress["scores"] < 12:
        # 600s is 4x the inherited 150s combined-chunk gate, including staging grace.
        if now - (progress["newest"] or began) > 600:
            raise RuntimeError("No first/new fit or score checkpoint within 10 minutes")
    else:
        since = state.setdefault("all_scores_observed_at", now)
        # Previous full 30GB run archived within 20min; this extension is about 5GB.
        if now - since > 1200:
            raise RuntimeError("Reduction/archival exceeded 20 minutes after scoring")
    return observed


def monitor(args):
    """Keep fresh evidence; errors remain visible and trigger the independent watchdog."""
    state = (
        json.loads(args.state.read_text())
        if args.state.exists()
        else dict(source_sha=args.source_sha, status="waiting_handle", started_at=time.time())
    )
    if state["source_sha"] != args.source_sha:
        raise ValueError("Monitor state belongs to a different source")
    write_state(args.state, state)
    while True:
        completion = verified_completion(args.source_sha)
        if completion:
            state.update(status="complete", results=completion, checked_at=time.time())
            write_state(args.state, state)
            print("Exact-source outputs independently verified", flush=True)
            return
        if not args.handle.exists():
            if time.time() - state["started_at"] > 1200:
                raise RuntimeError("Launch handle did not appear within 20 minutes")
            state.update(
                checked_at=time.time(),
                backend_observation=dict(status="pending"),
            )
        else:
            result = subprocess.run(
                [sys.executable, __file__, "--probe", "--handle", str(args.handle)],
                text=True,
                capture_output=True,
                check=True,
                timeout=240,
            )
            observed = json.loads(result.stdout.splitlines()[-1])
            observed = check_progress(state, observed, time.time())
            state.update(status="monitoring", checked_at=time.time(), backend_observation=observed)
            if observed["status"] in {"dead", "stalled", "gate"} or observed.get("stall_reason"):
                state["status"] = "backend_failed"
                write_state(args.state, state)
                raise RuntimeError("Backend failure; inspect the persisted observation")
        write_state(args.state, state)
        time.sleep(30)


def main():
    """Run one bounded probe or the durable monitor service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--handle", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe(args.handle)), flush=True)
        return
    if not args.source_sha or args.state is None:
        parser.error("monitor requires --source-sha and --state")
    try:
        monitor(args)
    except Exception as exc:
        state = json.loads(args.state.read_text()) if args.state.exists() else {}
        state.update(
            source_sha=args.source_sha,
            status="monitor_failed",
            checked_at=time.time(),
            error=f"{type(exc).__name__}: {exc}",
        )
        write_state(args.state, state)
        raise


if __name__ == "__main__":
    main()
