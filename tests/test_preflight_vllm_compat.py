"""Tests for the vLLM/transformers skew check's workload-scoped override (#2337).

Covers ``preflight.check_vllm_transformers_compat`` + its
``EPM_PREFLIGHT_ALLOW_TRANSFORMERS5`` acknowledgment env
(``preflight._allow_transformers5_override``), and the hub storage-headroom
probe's all-missing ``suspect (...)`` basis hint.

Design: the check imports ``vllm``/``transformers`` INSIDE the function body,
so stubbing ``sys.modules`` entries with controlled ``__version__`` attributes
exercises the REAL check body — no seam mock of the check itself. Env vars go
through ``monkeypatch`` (never direct ``os.environ`` mutation); ``sys.modules``
entries go through ``monkeypatch.setitem`` so the real modules (if any) are
restored after each test.
"""

import sys
import types
from unittest.mock import patch

import pytest

from explore_persona_space.orchestrate.hub import check_hf_storage_headroom
from explore_persona_space.orchestrate.preflight import (
    PreflightReport,
    check_vllm_transformers_compat,
)
from tests.test_hf_storage_headroom import NS, FakeHfApi, _env

ENV = "EPM_PREFLIGHT_ALLOW_TRANSFORMERS5"


def _run_check(monkeypatch, *, vllm_ver="0.11.0", transformers_ver="5.1.0", env=None):
    """Run the real check body against stubbed module versions; return the report.

    ``env=None`` guarantees the override env is UNSET; any string sets it.
    """
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__=vllm_ver))
    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(__version__=transformers_ver)
    )
    if env is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, env)
    report = PreflightReport()
    check_vllm_transformers_compat(report)
    return report


class TestSkewDefaultError:
    def test_skew_env_unset_is_error(self, monkeypatch):
        """Default behavior UNCHANGED: skew + env unset -> exactly 1 error, 0 warnings."""
        r = _run_check(monkeypatch)
        assert len(r.errors) == 1
        assert r.warnings == []
        assert r.ok is False

    def test_error_text_carries_both_remedy_branches(self, monkeypatch):
        """The remedy is conditional: pin advice AND the override env var name."""
        r = _run_check(monkeypatch)
        (msg,) = r.errors
        assert "transformers>=4.46,<5.0" in msg  # branch (a): vLLM workload -> pin
        assert ENV in msg  # branch (b): non-vLLM workload -> override
        # Do-not-downgrade caveat for transformers>=5-only models.
        assert "Do NOT downgrade" in msg
        assert "qwen3_5" in msg

    @pytest.mark.parametrize("val", ["0", "", "garbage", "TRUE", "Yes", " "])
    def test_non_accepted_env_values_still_error(self, monkeypatch, val):
        """Only the case-sensitive {1,true,yes} set overrides (precedent parity
        with EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE); anything else keeps the ERROR."""
        r = _run_check(monkeypatch, env=val)
        assert len(r.errors) == 1
        assert r.warnings == []
        assert r.ok is False


class TestSkewAcknowledged:
    @pytest.mark.parametrize("val", ["1", "true", "yes"])
    def test_override_degrades_to_single_warning(self, monkeypatch, val):
        """Skew + accepted env value -> 0 errors, exactly 1 warning; report.ok holds."""
        r = _run_check(monkeypatch, env=val)
        assert r.errors == []
        assert len(r.warnings) == 1
        assert r.ok is True

    def test_warning_names_override_and_expected_crash(self, monkeypatch):
        """The WARN is a loud breadcrumb: names the env var, states the skew is
        acknowledged and that any LLM(...) instantiation is EXPECTED to crash."""
        r = _run_check(monkeypatch, env="1")
        (msg,) = r.warnings
        assert ENV in msg
        assert "ACKNOWLEDGED" in msg
        assert "LLM(...)" in msg
        assert "EXPECTED to crash" in msg


class TestNonSkewCombosUnchanged:
    def test_transformers4_vllm011_clean(self, monkeypatch):
        r = _run_check(monkeypatch, transformers_ver="4.57.6")
        assert r.errors == []
        assert r.warnings == []

    def test_vllm012_transformers5_clean(self, monkeypatch):
        """Scope stays 0.11.x: vLLM 0.12 + transformers 5 is not flagged."""
        r = _run_check(monkeypatch, vllm_ver="0.12.0")
        assert r.errors == []
        assert r.warnings == []

    def test_override_inert_off_skew(self, monkeypatch):
        """Env set + NO skew -> no error, no warning (override changes nothing)."""
        r = _run_check(monkeypatch, transformers_ver="4.57.6", env="1")
        assert r.errors == []
        assert r.warnings == []

    def test_import_error_branch_still_warns(self, monkeypatch):
        """ImportError branch preserved: a None sys.modules entry makes
        ``import vllm`` raise, producing the could-not-import warning."""
        monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="5.1.0"))
        monkeypatch.setitem(sys.modules, "vllm", None)
        monkeypatch.delenv(ENV, raising=False)
        report = PreflightReport()
        check_vllm_transformers_compat(report)
        assert report.errors == []
        assert len(report.warnings) == 1
        assert "Could not import vllm/transformers" in report.warnings[0]


class TestHubAllMissingBasisHint:
    """The #2337 secondary fix: hub's all-missing suspect basis names the likely
    auth/endpoint condition so the line is not read as a storage fact."""

    def test_all_missing_basis_names_auth_endpoint_condition(self, tmp_path):
        fake = FakeHfApi(
            models=[(f"{NS}/a", False), (f"{NS}/b", False)],
            used={f"{NS}/a": None, f"{NS}/b": None},
        )
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert h.used_tb is None
        assert h.over_ceiling is False
        assert "suspect (2/2 missing usedStorage" in h.basis
        assert "auth/endpoint condition, not zero storage" in h.basis

    def test_partial_missing_basis_carries_no_auth_hint(self, tmp_path):
        """The hint fires ONLY on n_missing == n; a partial miss keeps the
        pre-#2337 basis verbatim."""
        fake = FakeHfApi(
            models=[(f"{NS}/a", False), (f"{NS}/b", False)],
            used={f"{NS}/a": 5 * 1000**4, f"{NS}/b": None},
        )
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert h.basis == "suspect (1/2 missing usedStorage)"
