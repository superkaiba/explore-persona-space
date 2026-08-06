"""Pins for the #2054 cell-(c) production pod driver
(`scripts/issue2054_phase_d_prod_driver.sh`) + the fits-driver cell_c knobs
(`scripts/issue2054_fits_pod_driver.sh`).

The load-bearing invariant is the MODEL-MATCHED capture split: the spliced
(c) text is captured through the model that AUTHORED the answer (variant
tail `_op` -> qwen2.5-7b-instruct, `_op_base` -> qwen2.5-7b —
`issue2054_phase_d._ANSWER_MODEL_FROM_TAIL`), pairing each (c) cell with the
(d) cell of the SAME (character, model, answer_form). A mis-paired driver
would capture happily through the WRONG model with no runtime failure, so
the pairing is pinned here against phase_d's own constants (never re-typed
literals).

Also pinned: the reserved-[phase=done] convention (every inner
`uv run python scripts/...` invocation redirected to its own child log; the
dispatcher alone emits the terminal token), `bash -n` syntax on both
drivers, the poll_pipeline sentinel envelope keys, and the fits-driver
cell_c knobs' defaults (a/b/d dispatches stay byte-equivalent).

Offline: text parses + `bash -n` only — no network, no GPU.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2054_phase_d as phase_d  # noqa: E402

DRIVER = _SCRIPTS / "issue2054_phase_d_prod_driver.sh"
FITS_DRIVER = _SCRIPTS / "issue2054_fits_pod_driver.sh"


def _driver_text() -> str:
    return DRIVER.read_text(encoding="utf-8")


def _var(text: str, name: str) -> str:
    m = re.search(rf'^{name}="([^"]*)"$', text, flags=re.MULTILINE)
    assert m, f"driver does not define {name}"
    return m.group(1)


def test_capture_split_is_model_matched_per_phase_d_constants():
    """The driver's two capture invocations cover ALL 8 phase_d variants,
    each through the model phase_d._ANSWER_MODEL_FROM_TAIL assigns its tail."""
    text = _driver_text()
    split = {
        "qwen2.5-7b-instruct": set(_var(text, "VARIANTS_INSTRUCT").split(",")),
        "qwen2.5-7b": set(_var(text, "VARIANTS_BASE").split(",")),
    }
    # Union == phase_d's default variant panel, disjoint halves.
    assert split["qwen2.5-7b-instruct"] | split["qwen2.5-7b"] == set(phase_d.DEFAULT_VARIANTS)
    assert not (split["qwen2.5-7b-instruct"] & split["qwen2.5-7b"])
    # Each variant is captured through its ANSWER-provenance model.
    for model, variants in split.items():
        for v in variants:
            assert phase_d._answer_model_for(v) == model, (
                f"driver captures {v} through {model}, but phase_d assigns "
                f"{phase_d._answer_model_for(v)}"
            )
    # And the wired lines actually use those variable sets with the matching model.
    assert re.search(r'run_capture qwen2\.5-7b-instruct "\$VARIANTS_INSTRUCT"', text), (
        "instruct capture line missing/mis-wired"
    )
    assert re.search(r'run_capture qwen2\.5-7b "\$VARIANTS_BASE"', text), (
        "base capture line missing/mis-wired"
    )


def test_inner_invocations_redirected_and_phase_done_reserved():
    """Every inner `uv run python scripts/...` call is redirected to a child
    log (the inner scripts emit their own [phase=done]; only the dispatcher's
    terminal token may reach the main log), and the dispatcher emits exactly
    one terminal [phase=done]."""
    text = _driver_text()
    for m in re.finditer(r"^\s*uv run python scripts/\S+.*$", text, flags=re.MULTILINE):
        block = text[m.start() :]
        # The invocation (possibly line-continued) must carry a `> "$..."` redirect
        # before the command ends (next un-continued line).
        cmd_lines = []
        for line in block.splitlines():
            cmd_lines.append(line)
            if not line.rstrip().endswith("\\"):
                break
        cmd = " ".join(cmd_lines)
        assert re.search(r">\s*\"?\$", cmd), f"unredirected inner invocation: {cmd!r}"
    done_lines = [
        ln for ln in text.splitlines() if "[phase=done]" in ln and not ln.lstrip().startswith("#")
    ]
    assert len(done_lines) == 1, f"expected exactly one dispatcher [phase=done]: {done_lines}"
    assert done_lines[0].strip() == 'echo "[phase=done]"'


def test_drivers_pass_bash_syntax_check():
    for script in (DRIVER, FITS_DRIVER):
        proc = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 0, f"bash -n {script.name}: {proc.stderr}"


def test_sentinel_envelope_keys_present():
    """The end-of-run sentinel writer composes every poll_pipeline
    _SENTINEL_REQUIRED_KEYS member (+ kind epm:results, hardcoded version 1
    per the pod-side contract)."""
    text = _driver_text()
    for key in ('"sentinel_schema_version": 1', '"kind": "epm:results"', '"version": 1'):
        assert key in text, f"sentinel writer missing {key}"
    # Sentinel lands in the poller's drained namespace by default.
    assert "issue-2054-epm_results-" in text


def test_no_task_py_shellout():
    """No pod-side task.py invocation (comments documenting the ban excluded)."""
    code_lines = [ln for ln in _driver_text().splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code_lines if "task.py" in ln]


def test_fits_driver_cell_c_knobs_default_to_production_values():
    """The cell_c knobs keep a/b/d dispatches byte-equivalent: FORMS defaults
    are the original two splits, the npz floor defaults to 48, and split-b
    skip fires only on an explicitly EMPTY ISSUE2054_FITS_FORMS_B."""
    text = FITS_DRIVER.read_text(encoding="utf-8")
    assert 'FORMS_A="${ISSUE2054_FITS_FORMS_A:-attrib_quoted,chat}"' in text
    assert 'FORMS_B="${ISSUE2054_FITS_FORMS_B:-bare_label,bare_text}"' in text
    assert 'EXPECTED_NPZ="${ISSUE2054_FITS_EXPECTED_NPZ:-48}"' in text
    # Floor semantics (superset tolerated once cell_c npz land), not equality.
    assert "len(npz) < expected" in text
    assert "len(npz) != 48" not in text
