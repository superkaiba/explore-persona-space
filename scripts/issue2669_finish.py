"""Wait for the owned production dispatcher, then audit, analyze and archive.

Usage: uv run python scripts/issue2669_finish.py RUN_ROOT
Never launches or retries a model call. Any failed phase stops the chain.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from issue2669_codex_dispatch import atomic_json


def finish(root: Path) -> dict:
    """Resume analysis after the owned process exits, with explicit phase records."""
    if not root.is_absolute():
        raise ValueError("Absolute run root required")
    launch = json.loads((root / "production_launch.json").read_text())
    pid = launch["pid"]
    command_file = Path(f"/proc/{pid}/cmdline")
    while command_file.exists():
        try:
            command = command_file.read_bytes()
        except FileNotFoundError:
            break
        if not command or str(root / "production_config.json").encode() not in command:
            break
        time.sleep(30)
    worktree = Path(__file__).resolve().parents[1]
    result = {"started_unix": time.time(), "production_pid": pid, "phases": []}
    for name, script, args in [
        ("analysis", "issue2669_analyze.py", [str(root), "production"]),
        ("archive", "issue2669_persist.py", [str(root), str(root.parent / "reduced900_v2_bundle")]),
    ]:
        print(f"[phase={name}] starting", flush=True)
        completed = subprocess.run(
            ["uv", "run", "python", str(worktree / "scripts" / script), *args],
            cwd=worktree,
            check=False,
        )
        result["phases"].append(
            {"phase": name, "returncode": completed.returncode, "finished_unix": time.time()}
        )
        atomic_json(root / "finish_status.json", result)
        if completed.returncode:
            raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    result["complete"] = True
    result["scope"] = "Codex forecasts and archive complete; matched probe comparison is separate"
    atomic_json(root / "finish_status.json", result)
    print("[phase=done] Codex forecast analysis and archive complete", flush=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(finish(Path(sys.argv[1])), indent=2))
