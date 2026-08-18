"""Tests for the two-tier venv import-health preflight check (#2360).

Covers ``preflight.check_venv_import_health`` (tier 1: importlib.metadata
resolvability against uv.lock names, BOTH broken-metadata shapes; tier 2: the
subprocess-isolated bounded deep-import probe with the ``LEAF`` deepest-cause
sentinel), its lane gating (cluster skip / RunPod default-on / env force /
kill switch), the tier-scoped ``venv_import_verdict``, the
``check_vllm_transformers_compat`` timeout guard, and the production-constant
pins (the #2360 B5 fix: the shipped constants are test-pinned, so emptied
constants or a dropped demonstrated casualty fail the committed suite).

Round-2 additions (#2360 r2): the future ``KeyError("Version")`` tier-1 form
(interpreter-version-independent), the multiline LEAF-message flatten, the
invalid probe-timeout named config error, and the explicit RunPod force-off
pin.

Design: the direct-call idiom of ``test_preflight_disk.py`` /
``test_preflight_vllm_compat.py`` — ``preflight.py`` import is side-effect-free
at module level (dotenv/HF_HOME mutation happens only inside
``preflight_check``), so tests call the check function directly. Lane helpers
(``is_cluster_env``/``is_runpod_env``) are monkeypatched on the preflight
module namespace; lane DETECTION itself is certified by
``tests/test_env_three_way_branch.py``, not here. Tier-2 tests run the REAL
production ``_run`` + ``_IMPORT_PROBE_SNIPPET`` in a real subprocess — no seam
mock of the probe.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import sys
import time
from pathlib import Path

import pytest

from explore_persona_space.orchestrate import preflight
from explore_persona_space.orchestrate.preflight import (
    DEEP_IMPORT_MODULES,
    IMPORT_PROBE_EXCLUSION_REASONS,
    LOAD_BEARING_DISTS,
    LOAD_BEARING_IMPORTS,
    PreflightReport,
    check_venv_import_health,
    check_vllm_transformers_compat,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

KILL_ENV = "EPM_PREFLIGHT_VENV_IMPORT_CHECK"
PROBE_ENV = "EPM_PREFLIGHT_IMPORT_PROBE"
TIMEOUT_ENV = "EPM_PREFLIGHT_IMPORT_PROBE_TIMEOUT_S"

LEAF_TOKEN = "UNIQUE_LEAF_TOKEN_2360"


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (KILL_ENV, PROBE_ENV, TIMEOUT_ENV):
        monkeypatch.delenv(var, raising=False)


def _lanes(monkeypatch: pytest.MonkeyPatch, *, cluster: bool = False, runpod: bool = False) -> None:
    monkeypatch.setattr(preflight, "is_cluster_env", lambda: cluster)
    monkeypatch.setattr(preflight, "is_runpod_env", lambda: runpod)


def _write_lock(tmp_path: Path, *names: str) -> Path:
    """Write a minimal parseable uv.lock naming ``names``; return the root."""
    body = "version = 1\n"
    for n in names:
        body += f'\n[[package]]\nname = "{n}"\nversion = "1.0"\n'
    (tmp_path / "uv.lock").write_text(body, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Tier 1: metadata resolvability (both broken shapes)
# ---------------------------------------------------------------------------


class TestTier1Metadata:
    def test_metadata_missing_lockpinned_dist_errors(self, tmp_path, monkeypatch):
        """Shape (i), the #2329 sympy casualty: module tree present, NO
        dist-info anywhere -> PackageNotFoundError -> one error naming the
        dist + the repair, verdict tier1-fail."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch)
        monkeypatch.setenv(PROBE_ENV, "0")
        site = tmp_path / "site"
        (site / "brokenpkg").mkdir(parents=True)
        (site / "brokenpkg" / "__init__.py").write_text("", encoding="utf-8")
        monkeypatch.syspath_prepend(str(site))
        root = _write_lock(tmp_path, "brokenpkg")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ("brokenpkg",))

        report = PreflightReport()
        check_venv_import_health(report, root)
        assert len(report.errors) == 1
        (msg,) = report.errors
        assert "brokenpkg" in msg
        assert "metadata" in msg
        assert "--force-reinstall" in msg
        assert report.ok is False
        assert report.venv_import_verdict == "tier1-fail"

    def test_metadata_file_missing_within_distinfo_errors(self, tmp_path, monkeypatch):
        """Shape (ii): dist-info DIR present, METADATA file missing ->
        version() returns None (does NOT raise) -> one error naming the
        None-return shape. Regression pin: a raising-only handler passes this
        venv silently."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch)
        monkeypatch.setenv(PROBE_ENV, "0")
        site = tmp_path / "site"
        (site / "brokenpkg2").mkdir(parents=True)
        (site / "brokenpkg2" / "__init__.py").write_text("", encoding="utf-8")
        distinfo = site / "brokenpkg2-1.0.dist-info"
        distinfo.mkdir()
        (distinfo / "RECORD").write_text("", encoding="utf-8")
        monkeypatch.syspath_prepend(str(site))
        # Fixture PRECONDITION (verified live on py3.11.15 + py3.12.13): the
        # gutted dist-info resolves to a None version, no raise.
        assert importlib.metadata.version("brokenpkg2") is None
        root = _write_lock(tmp_path, "brokenpkg2")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ("brokenpkg2",))

        report = PreflightReport()
        check_venv_import_health(report, root)
        assert len(report.errors) == 1
        (msg,) = report.errors
        assert "brokenpkg2" in msg
        assert "metadata file missing from present dist-info" in msg
        assert report.ok is False
        assert report.venv_import_verdict == "tier1-fail"

    def test_metadata_keyerror_future_form_errors(self, tmp_path, monkeypatch):
        """Forward-compat regression (#2360 r2, Codex blocker
        metadata-keyerror-forward-compat): CPython deprecated the
        implicit-None return ("Implicit None on return values is deprecated
        and will raise KeyErrors"), so a supported future interpreter raises
        KeyError('Version') for the exact shape (ii) fixture above.
        Interpreter-version-INDEPENDENT: version() is monkeypatched to raise
        for exactly the curated dist. One named error + verdict tier1-fail —
        never an uncaught traceback that replaces the structured JSON/verdict
        routing."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch)
        monkeypatch.setenv(PROBE_ENV, "0")
        root = _write_lock(tmp_path, "brokenpkg3")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ("brokenpkg3",))

        real_version = importlib.metadata.version

        def _future_version(dist: str) -> str | None:
            if dist == "brokenpkg3":
                raise KeyError("Version")
            return real_version(dist)

        monkeypatch.setattr(importlib.metadata, "version", _future_version)

        report = PreflightReport()
        check_venv_import_health(report, root)
        assert len(report.errors) == 1
        (msg,) = report.errors
        assert "brokenpkg3" in msg
        assert "metadata file missing from present dist-info" in msg
        assert "--force-reinstall" in msg
        assert report.ok is False
        assert report.venv_import_verdict == "tier1-fail"

    def test_metadata_missing_unlocked_dist_warns(self, tmp_path, monkeypatch):
        """A curated dist NOT named in uv.lock degrades to a warning — the
        legitimate-absence path (curated list may be stale), never a false
        FAIL."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch)
        monkeypatch.setenv(PROBE_ENV, "0")
        root = _write_lock(tmp_path, "someotherpkg")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ("brokenpkg",))

        report = PreflightReport()
        check_venv_import_health(report, root)
        assert report.errors == []
        assert len(report.warnings) == 1
        assert "not lock-pinned" in report.warnings[0]
        assert report.ok is True
        assert report.venv_import_verdict == "skipped-lane"

    def test_production_mapping_resolves_on_dev_venv(self, monkeypatch):
        """B5 positive side, NO constant monkeypatch: the FULL production
        tier-1 list resolves on the real dev venv against the real repo
        uv.lock — a constant naming a dist absent from the venv/lock breaks
        this loudly."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch)
        monkeypatch.setenv(PROBE_ENV, "0")

        report = PreflightReport()
        check_venv_import_health(report, REPO_ROOT)
        assert report.errors == []
        assert report.venv_import_verdict == "skipped-lane"


# ---------------------------------------------------------------------------
# Tier 2: the real subprocess probe
# ---------------------------------------------------------------------------


class TestTier2Probe:
    def test_deep_import_probe_surfaces_leaf_traceback(self, tmp_path, monkeypatch):
        """Real subprocess through the production ``_run`` + snippet: a broken
        leaf module fails with the LEAF sentinel LEADING the error. The
        no-wrapper-text assertion is SCOPED TO THIS SYNTHETIC FIXTURE (a
        top-level module with no lazy machinery)."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch, runpod=True)
        fixture = tmp_path / "mods"
        fixture.mkdir()
        (fixture / "broken_leaf.py").write_text(
            "import sympy_utilities_missing_xyz\n", encoding="utf-8"
        )
        monkeypatch.setenv("PYTHONPATH", str(fixture))
        root = _write_lock(tmp_path, "placeholder")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ())
        monkeypatch.setattr(preflight, "DEEP_IMPORT_MODULES", ("broken_leaf",))

        report = PreflightReport()
        check_venv_import_health(report, root)
        assert report.ok is False
        assert report.venv_import_verdict == "fail"
        assert len(report.errors) == 1
        (msg,) = report.errors
        assert msg.startswith(
            "LEAF ModuleNotFoundError: No module named 'sympy_utilities_missing_xyz'"
        )
        assert "'broken_leaf'" in msg
        assert "Could not import module" not in msg

    def test_leaf_sentinel_survives_midwindow_elision(self, tmp_path, monkeypatch):
        """The B6 regression pin: total stderr > 8,192 chars with the unique
        leaf token engineered into the ELIDED MIDDLE — >4,000 chars of leaf
        traceback BEFORE the token (80 distinct frames), >4,000 chars of
        wrapper AFTER it. The harvest alone drops the token; the pre-truncation
        LEAF sentinel carries it."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch, runpod=True)
        fixture = tmp_path / "mods"
        fixture.mkdir()
        lines = [
            "def f0():",
            f"    raise ModuleNotFoundError(\"No module named '{LEAF_TOKEN}'\")",
        ]
        for i in range(1, 80):  # 80 DISTINCT frames — repeated-frame collapsing cannot fire
            lines.append(f"def f{i}():")
            lines.append(f"    f{i - 1}()")
        lines += [
            "try:",
            "    f79()",
            "except ModuleNotFoundError as e:",
            "    raise ImportError('wrapper: ' + 'wrapper-pad ' * 600) from e",
        ]
        (fixture / "deepchain_leaf.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setenv("PYTHONPATH", str(fixture))

        # Positioning PRECONDITION on the RAW subprocess stderr first, so the
        # fixture cannot silently drift into a harvest window.
        rc, out, err = preflight._run(
            [sys.executable, "-c", preflight._IMPORT_PROBE_SNIPPET, "deepchain_leaf"],
            timeout=60,
        )
        assert rc == 3, (rc, err[:500])
        assert len(err) > 8192, len(err)
        assert LEAF_TOKEN not in err[:4000]
        assert LEAF_TOKEN not in err[-4000:]
        assert LEAF_TOKEN in err  # it IS there — just mid-window
        assert f"LEAF ModuleNotFoundError: No module named '{LEAF_TOKEN}'" in out

        root = _write_lock(tmp_path, "placeholder")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ())
        monkeypatch.setattr(preflight, "DEEP_IMPORT_MODULES", ("deepchain_leaf",))
        report = PreflightReport()
        check_venv_import_health(report, root)
        assert report.venv_import_verdict == "fail"
        (msg,) = report.errors
        assert LEAF_TOKEN in msg  # via the sentinel, window-independent
        assert "chars elided" in msg  # the harvest really elided the middle
        leaf_line = msg.split(" — ", 1)[0]
        assert leaf_line.startswith("LEAF ")
        assert len(leaf_line) <= 600  # the %.500s bound holds

    def test_leaf_sentinel_multiline_message_stays_one_line(self, tmp_path, monkeypatch):
        """FIX 3 regression (#2360 r2): a MULTILINE leaf exception message is
        flattened (CR/LF -> spaces) BEFORE the %.500s bound, so the sentinel
        stays ONE machine-readable stdout line and the report lead carries
        text from BEYOND the first physical line — the pre-fix emitter split
        the sentinel across lines and only its first line led the report,
        defeating the B6 one-line guarantee."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch, runpod=True)
        fixture = tmp_path / "mods"
        fixture.mkdir()
        (fixture / "multiline_leaf.py").write_text(
            "raise ModuleNotFoundError(\n"
            f"    'first-line-part\\nsecond-line-{LEAF_TOKEN}\\r\\nthird-line-part'\n"
            ")\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("PYTHONPATH", str(fixture))

        # RAW subprocess precondition: exactly ONE LEAF stdout line, carrying
        # tokens from the second AND third physical message lines.
        rc, out, err = preflight._run(
            [sys.executable, "-c", preflight._IMPORT_PROBE_SNIPPET, "multiline_leaf"],
            timeout=60,
        )
        assert rc == 3, (rc, err[:500])
        leaf_lines = [ln for ln in out.split("\n") if ln.startswith("LEAF ")]
        assert len(leaf_lines) == 1, out
        assert f"second-line-{LEAF_TOKEN}" in leaf_lines[0]
        assert "third-line-part" in leaf_lines[0]

        root = _write_lock(tmp_path, "placeholder")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ())
        monkeypatch.setattr(preflight, "DEEP_IMPORT_MODULES", ("multiline_leaf",))
        report = PreflightReport()
        check_venv_import_health(report, root)
        assert report.venv_import_verdict == "fail"
        (msg,) = report.errors
        assert msg.startswith("LEAF ModuleNotFoundError: first-line-part second-line-")
        # The full flattened message rides the LEAD (pre-" — " segment), not
        # just its first physical line.
        assert "third-line-part" in msg.split(" — ", 1)[0]

    def test_production_list_import_broken_metadata_intact(self, tmp_path, monkeypatch):
        """The B1/B5 negative on the PRODUCTION DEEP_IMPORT_MODULES (no
        constant monkeypatch), reproducing the #2329 anthropic casualty:
        metadata intact (tier 1 passes on the real venv), import broken —
        catchable only by the probe, and OUTSIDE plan-v2's six-module chain
        (this test fails against that design: the probe never reached
        anthropic)."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch)
        monkeypatch.setenv(PROBE_ENV, "1")
        shadow = tmp_path / "shadow"
        (shadow / "anthropic" / "types").mkdir(parents=True)
        (shadow / "anthropic" / "__init__.py").write_text("from . import types\n", encoding="utf-8")
        (shadow / "anthropic" / "types" / "__init__.py").write_text(
            "import anthropic.types.shared\n", encoding="utf-8"
        )
        monkeypatch.setenv("PYTHONPATH", str(shadow))

        report = PreflightReport()
        check_venv_import_health(report, REPO_ROOT)
        assert report.ok is False
        assert report.venv_import_verdict == "fail"
        # Tier 1 passed: the single error is the probe's, no metadata error.
        assert len(report.errors) == 1
        (msg,) = report.errors
        assert "'anthropic'" in msg
        assert "anthropic.types.shared" in msg
        assert msg.startswith("LEAF ")
        assert "dist metadata entirely absent" not in msg
        assert "metadata file missing" not in msg

    def test_deep_import_probe_timeout_distinct_verdict(self, tmp_path, monkeypatch, capsys):
        """A hung import gets the DISTINCT timeout verdict + the two-reading
        error text (slow cold read vs FUSE read-wedge) + the pre-probe banner
        BEFORE the diagnostic on stderr — and the child does not linger."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch, runpod=True)
        fixture = tmp_path / "mods"
        fixture.mkdir()
        (fixture / "hang_mod.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        monkeypatch.setenv("PYTHONPATH", str(fixture))
        monkeypatch.setenv(TIMEOUT_ENV, "1")
        root = _write_lock(tmp_path, "placeholder")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ())
        monkeypatch.setattr(preflight, "DEEP_IMPORT_MODULES", ("hang_mod",))

        t0 = time.monotonic()
        report = PreflightReport()
        check_venv_import_health(report, root)
        elapsed = time.monotonic() - t0
        # No lingering child: subprocess.run kills + reaps on TimeoutExpired;
        # a survived 30 s sleep would hold this well past the bound.
        assert elapsed < 20, elapsed
        assert report.venv_import_verdict == "timeout"
        assert report.ok is False
        (msg,) = report.errors
        assert "timed out after 1s" in msg
        assert "slow" in msg
        assert "read-wedge" in msg
        err_stream = capsys.readouterr().err
        banner_idx = err_stream.find("[preflight] deep-import probe launching")
        diag_idx = err_stream.find("deep-import probe timed out")
        assert banner_idx != -1
        assert diag_idx != -1
        assert banner_idx < diag_idx  # banner printed BEFORE the probe/diagnostic

    @pytest.mark.parametrize("bad", ["banana", "nan", "inf", "-5", "0"])
    def test_invalid_timeout_env_named_config_error(self, tmp_path, monkeypatch, bad):
        """FIX 4 (#2360 r2): a malformed / NaN / infinite / non-positive
        EPM_PREFLIGHT_IMPORT_PROBE_TIMEOUT_S yields ONE named configuration
        error + verdict config-error and the probe subprocess is never
        launched — never an uncaught ValueError/OverflowError firing before
        any verdict is set."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch, runpod=True)
        monkeypatch.setenv(TIMEOUT_ENV, bad)
        root = _write_lock(tmp_path, "placeholder")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ())

        def _no_probe(cmd, timeout=60):
            raise AssertionError("probe launched despite invalid timeout env")

        monkeypatch.setattr(preflight, "_run", _no_probe)
        report = PreflightReport()
        check_venv_import_health(report, root)
        assert report.venv_import_verdict == "config-error"
        assert report.ok is False
        (msg,) = report.errors
        assert "EPM_PREFLIGHT_IMPORT_PROBE_TIMEOUT_S" in msg
        assert bad in msg
        assert "NOT run" in msg

    def test_subsecond_timeout_reaches_run_untruncated(self, tmp_path, monkeypatch, capsys):
        """FIX A (#2360 r3, found by BOTH r2 reviewers): a sub-second timeout
        in (0, 1) is documented-valid ("a finite positive number of seconds")
        and must reach ``_run`` as the FLOAT 0.5 — the r2 form passed
        validation, then ``int(0.5) == 0`` truncated it at the ``_run`` call
        (immediate false timeout on a healthy probe) and the banner/diagnostic
        rendered a misleading "0s". The banner must render "0.5s"."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch, runpod=True)
        monkeypatch.setenv(TIMEOUT_ENV, "0.5")
        root = _write_lock(tmp_path, "placeholder")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ())

        seen: list[object] = []

        def _capture(cmd, timeout=60):
            seen.append(timeout)
            return 0, "OK placeholder\n", ""

        monkeypatch.setattr(preflight, "_run", _capture)
        report = PreflightReport()
        check_venv_import_health(report, root)
        assert seen == [0.5], seen  # pre-fix: int(0.5) == 0 reached _run
        assert report.venv_import_verdict == "ok"
        assert report.ok is True
        assert report.errors == []
        banner = capsys.readouterr().err
        assert "(timeout 0.5s)" in banner, banner

    def test_compat_check_skips_on_probe_timeout(self, monkeypatch):
        """On a timeout verdict the compat check SKIPs before its unbounded
        in-process imports. The sys.modules tripwire + wording asserts
        discriminate guard-taken from guard-missed: the pre-guard fail-open
        path also warns once, but with 'Could not import vllm/transformers'."""
        monkeypatch.setitem(sys.modules, "vllm", None)  # tripwire: import would raise
        report = PreflightReport()
        report.venv_import_verdict = "timeout"
        check_vllm_transformers_compat(report)
        assert report.errors == []
        assert len(report.warnings) == 1
        (msg,) = report.warnings
        assert "SKIPPED" in msg
        assert "probe timed out" in msg
        assert "Could not import" not in msg

    def test_tier1_fail_skips_probe_and_scopes_verdict(self, tmp_path, monkeypatch):
        """Tier scoping (the verdict consumer wart fix): a tier-1 error sets
        tier1-fail and the probe is NEVER launched even with the lane forced
        on — the verdict can never read healthy after a tier-1 error."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch)
        monkeypatch.setenv(PROBE_ENV, "1")
        root = _write_lock(tmp_path, "brokenpkg")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ("brokenpkg",))

        def _no_probe(cmd, timeout=60):
            if preflight._IMPORT_PROBE_SNIPPET in cmd:
                raise AssertionError("probe launched despite tier-1 failure")
            raise AssertionError(f"unexpected _run call: {cmd}")

        monkeypatch.setattr(preflight, "_run", _no_probe)
        report = PreflightReport()
        check_venv_import_health(report, root)
        assert report.venv_import_verdict == "tier1-fail"
        assert report.ok is False


# ---------------------------------------------------------------------------
# Lane gating + kill switch
# ---------------------------------------------------------------------------


class TestLaneGating:
    def test_cluster_skip(self, tmp_path, monkeypatch):
        _clear_env(monkeypatch)
        _lanes(monkeypatch, cluster=True)
        report = PreflightReport()
        check_venv_import_health(report, _write_lock(tmp_path, "x"))
        assert report.venv_import_verdict == "skipped-cluster"
        assert report.errors == []
        assert len(report.warnings) == 1
        assert "SKIPPED on cluster" in report.warnings[0]

    def test_lane_gate_off_pod(self, tmp_path, monkeypatch):
        """Off-pod, env unset: tier 1 runs, tier 2 silently skips (no warning
        spam on every VM launch)."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch)
        root = _write_lock(tmp_path, "placeholder")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ())
        report = PreflightReport()
        check_venv_import_health(report, root)
        assert report.venv_import_verdict == "skipped-lane"
        assert report.errors == []
        assert report.warnings == []

    def test_lane_gate_force_on(self, tmp_path, monkeypatch):
        """EPM_PREFLIGHT_IMPORT_PROBE=1 forces the probe OFF-pod; a healthy
        module list yields verdict ok."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch)
        monkeypatch.setenv(PROBE_ENV, "1")
        root = _write_lock(tmp_path, "placeholder")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ())
        monkeypatch.setattr(preflight, "DEEP_IMPORT_MODULES", ("json",))
        report = PreflightReport()
        check_venv_import_health(report, root)
        assert report.venv_import_verdict == "ok"
        assert report.errors == []

    def test_lane_gate_force_off_on_pod(self, tmp_path, monkeypatch):
        """FIX 5 (#2360 r2): EPM_PREFLIGHT_IMPORT_PROBE=0 forces the probe
        OFF even on RunPod, where the default is ON — verdict skipped-lane,
        probe never launched. The code was already correct; this pin makes
        the explicit force-off-on-pod behavior durable."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch, runpod=True)
        monkeypatch.setenv(PROBE_ENV, "0")
        root = _write_lock(tmp_path, "placeholder")
        monkeypatch.setattr(preflight, "LOAD_BEARING_DISTS", ())

        def _no_probe(cmd, timeout=60):
            raise AssertionError("probe launched despite explicit force-off on RunPod")

        monkeypatch.setattr(preflight, "_run", _no_probe)
        report = PreflightReport()
        check_venv_import_health(report, root)
        assert report.venv_import_verdict == "skipped-lane"
        assert report.errors == []
        assert report.warnings == []

    def test_kill_switch(self, tmp_path, monkeypatch):
        """EPM_PREFLIGHT_VENV_IMPORT_CHECK=0 disables everything: one warning,
        nothing else on the report."""
        _clear_env(monkeypatch)
        _lanes(monkeypatch, runpod=True)
        monkeypatch.setenv(KILL_ENV, "0")
        report = PreflightReport()
        check_venv_import_health(report, _write_lock(tmp_path, "x"))
        assert report.venv_import_verdict == "disabled"
        assert report.errors == []
        assert len(report.warnings) == 1
        assert "DISABLED" in report.warnings[0]
        assert report.ok is True


# ---------------------------------------------------------------------------
# Wiring + production-constant pins (B5)
# ---------------------------------------------------------------------------


class TestWiringAndConstants:
    def test_wired_into_preflight_check(self):
        """Source-substring ordering pin: the import-health check runs BEFORE
        the compat check (load-bearing for the wedge design — the bounded
        probe converts the hang the compat check's in-process imports would
        otherwise hit). Known limit: passes on a commented-out call — the
        BINDING end-to-end check is the VM smoke's --json verdict read
        (skipped-lane, not '')."""
        src = inspect.getsource(preflight.preflight_check)
        assert "check_venv_import_health(report" in src
        assert src.index("check_venv_import_health(report") < src.index(
            "check_vllm_transformers_compat(report)"
        )

    def test_mapping_completeness_static(self):
        """B1: every curated dist has >=1 probe module OR a non-empty
        exclusion reason; exclusions name only mapping keys; both derived
        tuples really derive from the ONE mapping. Coverage is a test, not a
        convention."""
        for dist, modules in LOAD_BEARING_IMPORTS.items():
            if not modules:
                assert dist in IMPORT_PROBE_EXCLUSION_REASONS, (
                    f"tier-1-only dist {dist!r} has no exclusion reason"
                )
                assert IMPORT_PROBE_EXCLUSION_REASONS[dist].strip(), dist
        assert set(IMPORT_PROBE_EXCLUSION_REASONS) <= set(LOAD_BEARING_IMPORTS)
        assert tuple(LOAD_BEARING_IMPORTS) == LOAD_BEARING_DISTS
        assert (
            tuple(dict.fromkeys(m for mods in LOAD_BEARING_IMPORTS.values() for m in mods))
            == DEEP_IMPORT_MODULES
        )

    def test_production_constants_pin_incident(self):
        """B5: the SHIPPED constants carry both demonstrated #2329 casualties
        + the deep-chain modules; emptied constants or a dropped casualty
        fails the committed suite."""
        assert {"sympy", "anthropic", "torch", "transformers", "vllm"} <= set(LOAD_BEARING_IMPORTS)
        assert {
            "sympy",
            "anthropic",
            "torch._dynamo",
            "transformers.models.auto.modeling_auto",
        } <= set(DEEP_IMPORT_MODULES)
        assert "vllm" in IMPORT_PROBE_EXCLUSION_REASONS
        assert len(LOAD_BEARING_IMPORTS) >= 14

    def test_lock_pins_all_curated_dists(self):
        """Every curated dist is named in the REAL repo uv.lock — a dep
        dropped from the lock fails loudly at CI instead of warning at
        preflight only."""
        lock_text = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
        for dist in LOAD_BEARING_IMPORTS:
            assert f'name = "{dist}"' in lock_text, f"{dist!r} not pinned in uv.lock"
