"""Money-safety guard tests for the dataset-prep chain.

Background: during #468 dataset provisioning, a subagent tried to provision
turner cells "decrypt-only" via ``--force-step 1``, but that flag is a FLOOR
(start at step N), not exclusive — after the decrypt failed (no password)
it fell through to Step 2 Claude regen, which submitted one ~1000-request
Sonnet-4.5 batch (money spent, bounded, no data landed). These tests pin
the spend-disable contract added in response:

* ``scripts/fetch_or_generate_issue404_medical.py --no-claude-regen`` HARD-
  disables the Step 2 Claude regen fallback. If Step 1 fails, exit 3 (drop
  the pair) WITHOUT invoking the Step-2 delegate subprocess.
* ``scripts/issue458_prep_datasets.py`` ``prep_turner('turner_bad_medical', ...)``
  passes ``--no-claude-regen`` BY DEFAULT (turner-by-default-safe).
* ``scripts/issue458_prep_datasets.py prep_json_neg(..., no_generate=True)``
  does NOT invoke ``generate_issue404_json_neg.py`` — it tries the free
  HF download and returns 0 on failure.

Tests are pure unit tests with mocked subprocess + HF download — no
network, no Claude/GPT-4o API calls.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


# ─────────────────────────────────────────────────────────────────────────────
# Module loading (exec the script files as modules so we can call into them)
# ─────────────────────────────────────────────────────────────────────────────


def _load_script(name: str):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"could not spec for {name}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fetch_mod():
    return _load_script("fetch_or_generate_issue404_medical")


@pytest.fixture
def prep_mod():
    return _load_script("issue458_prep_datasets")


# ─────────────────────────────────────────────────────────────────────────────
# fetch_or_generate_issue404_medical.py: --no-claude-regen contract
# ─────────────────────────────────────────────────────────────────────────────


def test_no_claude_regen_flag_present(fetch_mod):
    """The CLI exposes a ``--no-claude-regen`` store_true flag, default False
    (backward-compatible). Verified via subprocess --help + source-text grep."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "fetch_or_generate_issue404_medical.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    help_text = proc.stdout + proc.stderr
    assert "--no-claude-regen" in help_text, (
        "fetch_or_generate_issue404_medical.py --help must mention --no-claude-regen\n" + help_text
    )
    # Verify backward-compatible default by grepping for the flag config:
    src = (SCRIPTS_DIR / "fetch_or_generate_issue404_medical.py").read_text(encoding="utf-8")
    assert 'action="store_true"' in src and "--no-claude-regen" in src


def test_no_claude_regen_skips_step2_delegate(fetch_mod, tmp_path, monkeypatch):
    """When --no-claude-regen is set and Step 1 fails, main() must NOT call
    ``try_claude_regen_step`` (which invokes the paid delegate subprocess).
    main() must exit 3 and write 'dropped' metadata."""
    # Redirect output paths into tmp_path so we don't pollute data/issue404.
    out_dir = tmp_path / "data" / "issue404"
    monkeypatch.setattr(fetch_mod, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(fetch_mod, "OUTPUT_FILE", out_dir / "bad_medical_advice.jsonl")
    monkeypatch.setattr(fetch_mod, "METADATA_FILE", out_dir / "bad_medical_advice.metadata.json")

    # Force Step 1 to fail without doing any real git clone / network IO.
    monkeypatch.setattr(
        fetch_mod, "try_turner_step", lambda: (False, "stubbed: forced failure for test")
    )

    # Spy on try_claude_regen_step. If --no-claude-regen works, this MUST
    # NOT be called.
    claude_calls = {"n": 0}

    def _should_not_be_called():
        claude_calls["n"] += 1
        return (True, "FAILURE: should not be invoked with --no-claude-regen")

    monkeypatch.setattr(fetch_mod, "try_claude_regen_step", _should_not_be_called)

    # Also kill the upload path in case anything weird happens.
    monkeypatch.setattr(fetch_mod, "upload", lambda no_upload: None)

    # Simulate CLI: --no-claude-regen --no-upload
    monkeypatch.setattr(
        sys, "argv", ["fetch_or_generate_issue404_medical.py", "--no-claude-regen", "--no-upload"]
    )

    rc = fetch_mod.main()
    assert rc == 3, f"expected exit 3 (drop pair); got {rc}"
    assert claude_calls["n"] == 0, (
        f"try_claude_regen_step was invoked {claude_calls['n']} times "
        "with --no-claude-regen; the money-safety guard FAILED."
    )

    # The 'dropped' metadata file should mention the guard in its reason.
    meta_path = out_dir / "bad_medical_advice.metadata.json"
    assert meta_path.exists(), "drop-marker metadata file should be written"
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["source"] == "dropped"
    assert (
        "no-claude-regen" in meta["lock_reason"].lower() or "step 2" in meta["lock_reason"].lower()
    )


def test_default_behavior_unchanged_step2_still_invoked(fetch_mod, tmp_path, monkeypatch):
    """Backward-compat: WITHOUT --no-claude-regen, if Step 1 fails,
    main() MUST still invoke try_claude_regen_step (existing behavior)."""
    out_dir = tmp_path / "data" / "issue404"
    monkeypatch.setattr(fetch_mod, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(fetch_mod, "OUTPUT_FILE", out_dir / "bad_medical_advice.jsonl")
    monkeypatch.setattr(fetch_mod, "METADATA_FILE", out_dir / "bad_medical_advice.metadata.json")
    monkeypatch.setattr(
        fetch_mod, "try_turner_step", lambda: (False, "stubbed: forced failure for test")
    )

    claude_calls = {"n": 0}

    def _step2(*_a, **_kw):
        claude_calls["n"] += 1
        # Return failure so main() also drops, keeping the test cheap.
        return (False, "stubbed: forced step2 failure for test")

    monkeypatch.setattr(fetch_mod, "try_claude_regen_step", _step2)
    monkeypatch.setattr(fetch_mod, "upload", lambda no_upload: None)

    # Default CLI (no --no-claude-regen).
    monkeypatch.setattr(sys, "argv", ["fetch_or_generate_issue404_medical.py", "--no-upload"])

    rc = fetch_mod.main()
    # Both fallbacks failed → expect rc=3, but with the Claude path actually
    # invoked (proving the default still allows it).
    assert rc == 3
    assert claude_calls["n"] == 1, (
        "Default behavior (no --no-claude-regen) must still invoke "
        "try_claude_regen_step on Step 1 failure; "
        f"got {claude_calls['n']} calls."
    )


def test_contradictory_flags_force_step2_with_no_claude_regen(fetch_mod, tmp_path, monkeypatch):
    """--no-claude-regen + --force-step=2 is contradictory; main() must
    refuse with exit 4 and NOT invoke the Claude delegate."""
    out_dir = tmp_path / "data" / "issue404"
    monkeypatch.setattr(fetch_mod, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(fetch_mod, "OUTPUT_FILE", out_dir / "bad_medical_advice.jsonl")
    monkeypatch.setattr(fetch_mod, "METADATA_FILE", out_dir / "bad_medical_advice.metadata.json")

    claude_calls = {"n": 0}

    def _step2(*_a, **_kw):
        claude_calls["n"] += 1
        return (False, "should not be reached")

    monkeypatch.setattr(fetch_mod, "try_claude_regen_step", _step2)
    monkeypatch.setattr(fetch_mod, "upload", lambda no_upload: None)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_or_generate_issue404_medical.py",
            "--no-claude-regen",
            "--force-step",
            "2",
            "--no-upload",
        ],
    )
    rc = fetch_mod.main()
    assert rc == 4
    assert claude_calls["n"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# issue458_prep_datasets.py: prep_turner default-safe + prep_json_neg guard
# ─────────────────────────────────────────────────────────────────────────────


def test_no_generate_flag_present(prep_mod):
    """The CLI exposes a ``--no-generate`` store_true flag, default False."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "issue458_prep_datasets.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    help_text = proc.stdout + proc.stderr
    assert "--no-generate" in help_text, (
        "issue458_prep_datasets.py --help must mention --no-generate\n" + help_text
    )


def test_prep_turner_passes_no_claude_regen_by_default(prep_mod, tmp_path, monkeypatch):
    """prep_turner('turner_bad_medical', ...) MUST include --no-claude-regen
    in the medical-fetcher subprocess by default (no_generate=True default)."""
    # Redirect DATA_DIR so the idempotency check doesn't short-circuit on a
    # real existing file.
    data_dir = tmp_path / "data" / "issue404"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(prep_mod, "DATA_DIR", data_dir)

    captured: dict = {}

    class FakeCompleted:
        returncode = 1  # non-zero so we don't try to read the (non-existent) output

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = kwargs
        return FakeCompleted()

    # Patch subprocess.run at the module level (prep_turner imports it locally).
    import subprocess as real_subprocess

    monkeypatch.setattr(real_subprocess, "run", fake_run)

    n = prep_mod.prep_turner("turner_bad_medical", max_rows=6000)
    assert n == 0  # subprocess returned rc=1 and no file produced
    assert "cmd" in captured, "subprocess.run was never invoked"
    cmd = captured["cmd"]
    assert "--no-claude-regen" in cmd, (
        f"prep_turner default-safe contract VIOLATED — the medical fetcher "
        f"subprocess command must include --no-claude-regen by default. "
        f"Command was: {cmd}"
    )
    # Sanity: it should target the right script.
    assert any("fetch_or_generate_issue404_medical.py" in part for part in cmd)


def test_prep_turner_override_drops_no_claude_regen(prep_mod, tmp_path, monkeypatch):
    """When the caller explicitly passes no_generate=False, prep_turner
    drops the --no-claude-regen flag (paid Claude regen re-authorized)."""
    data_dir = tmp_path / "data" / "issue404"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(prep_mod, "DATA_DIR", data_dir)

    captured: dict = {}

    class FakeCompleted:
        returncode = 1

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeCompleted()

    import subprocess as real_subprocess

    monkeypatch.setattr(real_subprocess, "run", fake_run)

    prep_mod.prep_turner("turner_bad_medical", max_rows=6000, no_generate=False)
    cmd = captured["cmd"]
    assert "--no-claude-regen" not in cmd, (
        f"With no_generate=False, prep_turner should NOT pass --no-claude-regen "
        f"to the medical fetcher. Command was: {cmd}"
    )


def test_prep_json_neg_no_generate_skips_claude_batch(prep_mod, tmp_path, monkeypatch):
    """prep_json_neg(..., no_generate=True) must NOT invoke
    generate_issue404_json_neg.py — it must try HF download only and
    return 0 on download failure."""
    data_dir = tmp_path / "data" / "issue404"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(prep_mod, "DATA_DIR", data_dir)

    # Force the HF download path to fail so we exercise the skip branch.
    monkeypatch.setattr(prep_mod, "_try_download_json_neg_from_hf", lambda target: False)

    subprocess_calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        subprocess_calls["n"] += 1
        # Surface what was attempted in the failure message.
        raise AssertionError(
            f"prep_json_neg with no_generate=True invoked subprocess: {cmd!r}. "
            "The Claude Batch generator MUST NOT be called when no_generate=True."
        )

    import subprocess as real_subprocess

    monkeypatch.setattr(real_subprocess, "run", fake_run)

    n = prep_mod.prep_json_neg("json_neg", max_rows=6000, no_generate=True)
    assert n == 0
    assert subprocess_calls["n"] == 0


def test_prep_json_neg_no_generate_uses_hf_download(prep_mod, tmp_path, monkeypatch):
    """prep_json_neg(..., no_generate=True) uses the HF download path when
    it succeeds, and the resulting row count flows through."""
    data_dir = tmp_path / "data" / "issue404"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(prep_mod, "DATA_DIR", data_dir)

    # Simulate a successful HF download by writing 3 rows into the target.
    def fake_hf_download(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            '{"messages": [{"role": "user", "content": "q"}, '
            '{"role": "assistant", "content": "a"}]}\n' * 3,
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr(prep_mod, "_try_download_json_neg_from_hf", fake_hf_download)

    # Sentry: subprocess.run must NOT be invoked on the no_generate=True path.
    def fake_run(cmd, **kwargs):
        raise AssertionError(f"prep_json_neg with no_generate=True must not subprocess: {cmd!r}")

    import subprocess as real_subprocess

    monkeypatch.setattr(real_subprocess, "run", fake_run)

    n = prep_mod.prep_json_neg("json_neg", max_rows=6000, no_generate=True)
    assert n == 3


def test_prep_json_neg_default_invokes_generator(prep_mod, tmp_path, monkeypatch):
    """Backward-compat: prep_json_neg without no_generate still calls
    generate_issue404_json_neg.py (existing behavior preserved)."""
    data_dir = tmp_path / "data" / "issue404"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(prep_mod, "DATA_DIR", data_dir)

    captured: dict = {}

    class FakeCompleted:
        returncode = 1  # fail so we don't try to do anything else

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeCompleted()

    import subprocess as real_subprocess

    monkeypatch.setattr(real_subprocess, "run", fake_run)

    n = prep_mod.prep_json_neg("json_neg", max_rows=6000)
    assert n == 0
    assert "cmd" in captured, "default prep_json_neg must invoke the generator subprocess"
    assert any("generate_issue404_json_neg.py" in part for part in captured["cmd"])
