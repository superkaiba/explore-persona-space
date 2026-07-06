"""Regression guard: the wrapper's deferred-fit-failures gate sweeps EVERY
--no-internal-gates out-dir, not just $OUT_DIR.

Round-2 Codex Major (fixed round 3): the conditional matched-parent refit runs
`issue825_fit_cells.py --no-internal-gates --out-dir "$OUT_DIR/matched_parent"`,
but the post-UPLOAD-2 gate checked only `$OUT_DIR/fit_failures.json` — a
deferred matched-parent crash would have sailed through to a SUCCESS sentinel.
The gate now rglobs `fit_failures.json` under `$OUT_DIR`, covering both fit
invocations plus any future nested one. Tests run the wrapper's `[phase=gate]`
python heredoc VERBATIM (extracted from the shell script — the same mechanics
as the round-2 forced-G3 demo), so they exercise the exact production gate
code. The deferred-fit-failures gate is the FIRST gate, so the fixtures need
no other artifacts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DISPATCH = REPO_ROOT / "scripts" / "issue825_onpolicy_dispatch.sh"

FAILURE_ROW = [
    {
        "cell_id": "M_pretrained_user_naturalistic",
        "error_type": "ValueError",
        "error": "All-NaN slice encountered",
        "ts": "2026-07-03T00:00:00Z",
    }
]


def _extract_gate_phase() -> str:
    """Return the [phase=gate] python heredoc body from the dispatch wrapper."""
    text = DISPATCH.read_text()
    start = text.index('echo "[phase=gate]"')
    body_start = text.index("\n", text.index("<<'PY'", start)) + 1
    body_end = text.index("\nPY\n", body_start)
    return text[body_start : body_end + 1]


def _run_gate(tmp_path: Path, *, matched_parent_failure: bool, top_level_failure: bool):
    """Run the extracted gate phase against a tmp fixture; return (proc, sentinel)."""
    out = tmp_path / "out"
    onp = tmp_path / "onp"
    ts = tmp_path / "ts"
    for d in (out, onp, ts):
        d.mkdir(parents=True, exist_ok=True)
    if matched_parent_failure:
        (out / "matched_parent").mkdir()
        (out / "matched_parent" / "fit_failures.json").write_text(json.dumps(FAILURE_ROW))
    if top_level_failure:
        (out / "fit_failures.json").write_text(json.dumps(FAILURE_ROW))
    gate_py = tmp_path / "gate_phase.py"
    gate_py.write_text(_extract_gate_phase())
    sentinel = tmp_path / "logs" / "issue-825-epm_results-0.json"
    env = {
        **os.environ,
        "EPS_ONP_DIR": str(onp),
        "EPS_TS_DIR": str(ts),
        "EPS_OUT_DIR": str(out),
        "EPS_SENTINEL": str(sentinel),
        "EPS_CELLS8": "M_instruct_assistant_chat,M_instruct_user_chat",
        "EPS_USER4": "M_instruct_user_chat",
        "EPS_PARENT_EVAL": str(tmp_path / "parent_eval"),
        # deliberately absent: on the no-failure path the NEXT gate crashes on
        # this missing file, proving the deferred gate itself passed first
        "EPS_CONV": str(tmp_path / "conversations.jsonl"),
        "EPS_SMOKE": "1",
    }
    proc = subprocess.run(
        [sys.executable, str(gate_py)], env=env, capture_output=True, text=True, timeout=120
    )
    return proc, sentinel


def test_matched_parent_deferred_failure_halts(tmp_path):
    """A fit_failures.json ONLY under matched_parent/ HALTs with the sentinel."""
    proc, sentinel = _run_gate(tmp_path, matched_parent_failure=True, top_level_failure=False)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "fit_deferred_failure" in proc.stdout + proc.stderr
    payload = json.loads(sentinel.read_text())
    assert payload["status"] == "fit_deferred_failure"
    assert payload["sentinel_schema_version"] == 1
    # the deferred record names the matched_parent file explicitly
    assert "matched_parent/fit_failures.json" in json.dumps(payload)


def test_top_level_deferred_failure_still_halts(tmp_path):
    """Parity: the original $OUT_DIR/fit_failures.json path still HALTs."""
    proc, sentinel = _run_gate(tmp_path, matched_parent_failure=False, top_level_failure=True)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    payload = json.loads(sentinel.read_text())
    assert payload["status"] == "fit_deferred_failure"
    assert "fit_failures.json" in json.dumps(payload)


def test_no_deferred_failures_gate_passes(tmp_path):
    """No fit_failures.json anywhere: the deferred gate PASSes (later gates
    then crash on the deliberately-absent fixture files — that crash must NOT
    be a fit_deferred_failure sentinel)."""
    proc, sentinel = _run_gate(tmp_path, matched_parent_failure=False, top_level_failure=False)
    assert "gate: deferred-fit-failures PASS" in proc.stdout
    if sentinel.exists():
        assert json.loads(sentinel.read_text())["status"] != "fit_deferred_failure"
