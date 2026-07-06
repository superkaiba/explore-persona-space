"""Tests for the HF public-storage headroom guard (#564).

Covers the three layers:

* ``hub.check_hf_storage_headroom`` — two-stage probe, partial-None
  poisoning, success-only on-disk cache, env knobs, kill switch.
* ``trainer._validate_persist_headroom`` — the minute-1 fail-loud gate for
  persist-declared launches (plus wiring tests proving the gate fires
  BEFORE any model load in BOTH ``_init_phase`` and ``train_lora``).
* ``hub.upload_model`` overflow routing — env-gated reroute to the private
  overflow repo, deviation event sink, canonical-repo pointer breadcrumb.

Test-design constraints (binding, plan §6): the fake ``HfApi`` is
EXPAND-HONORING — list endpoints expose ``private`` only under
``expand=["private"]`` and per-repo info endpoints expose ``usedStorage``
only under ``expand=["usedStorage"]`` — so an implementation that drops the
expand args cannot pass. Every gate/routing fixture pins
``EPM_HF_STORAGE_CACHE_PATH`` (the gate calls the helper with no
``cache_path`` param) and ``EPM_HF_OVERFLOW_EVENT_PATH`` (host-independent
event sink) into a tmp dir so no test ever touches the real ``~/.cache``.
"""

import json
import logging
import time
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from explore_persona_space.orchestrate import hub
from explore_persona_space.orchestrate.hub import (
    DEFAULT_MODEL_REPO,
    DEFAULT_OVERFLOW_REPO,
    check_hf_storage_headroom,
    upload_model,
)

TB = 1000**4  # decimal bytes per TB (matches hub._BYTES_PER_TB)
NS = "superkaiba1"


class _Obj:
    """Attribute container that does NOT auto-create attributes (unlike MagicMock)."""


class FakeHfApi:
    """Expand-honoring fake HfApi.

    List endpoints expose ``private`` ONLY when called with exactly
    ``expand=["private"]``; per-repo info endpoints expose ``usedStorage``
    ONLY when called with exactly ``expand=["usedStorage"]``. An
    implementation that drops the expand kwargs therefore reads
    perpetually-missing fields and fails these tests.
    """

    def __init__(
        self,
        *,
        models=(),
        datasets=(),
        used=None,
        repo_info_private=None,
        repo_info_exc=None,
        list_exc=None,
    ):
        self.models = list(models)  # (repo_id, private)
        self.datasets = list(datasets)
        self.used = dict(used or {})  # repo_id -> int | None | Exception
        self.repo_info_private = repo_info_private  # bool | None
        self.repo_info_exc = repo_info_exc
        self.list_exc = list_exc
        self.list_models_calls: list[dict] = []
        self.list_datasets_calls: list[dict] = []
        self.info_calls: list[tuple] = []  # (repo_id, kind, expand)
        self.repo_info_calls: list[tuple] = []
        self.upload_folder_calls: list[dict] = []
        self.upload_file_calls: list[dict] = []
        self.create_repo_calls: list[dict] = []
        self.tree_files: list[str] = []  # paths returned by list_repo_tree

    def _listing(self, rows, expand):
        out = []
        for rid, priv in rows:
            o = _Obj()
            o.id = rid
            if expand == ["private"]:
                o.private = priv
            out.append(o)
        return out

    def list_models(self, *, author=None, expand=None):
        self.list_models_calls.append({"author": author, "expand": expand})
        if self.list_exc is not None:
            raise self.list_exc
        return self._listing(self.models, expand)

    def list_datasets(self, *, author=None, expand=None):
        self.list_datasets_calls.append({"author": author, "expand": expand})
        return self._listing(self.datasets, expand)

    def _info(self, rid, kind, expand):
        self.info_calls.append((rid, kind, expand))
        v = self.used.get(rid)
        if isinstance(v, Exception):
            raise v
        o = _Obj()
        if expand == ["usedStorage"]:
            o.usedStorage = v  # may be None — attribute present but unpopulated
        return o

    def model_info(self, rid, *, expand=None):
        return self._info(rid, "model", expand)

    def dataset_info(self, rid, *, expand=None):
        return self._info(rid, "dataset", expand)

    def repo_info(self, rid, *, repo_type=None):
        self.repo_info_calls.append((rid, repo_type))
        if self.repo_info_exc is not None:
            raise self.repo_info_exc
        o = _Obj()
        if self.repo_info_private is not None:
            o.private = self.repo_info_private
        return o

    # --- upload surface (routing tests drive upload_model end to end) -----
    def create_repo(self, repo_id, *, repo_type=None, private=False, exist_ok=False):
        self.create_repo_calls.append(
            {"repo_id": repo_id, "repo_type": repo_type, "private": private, "exist_ok": exist_ok}
        )

    def upload_folder(self, *, folder_path, repo_id, path_in_repo, repo_type, ignore_patterns=None):
        self.upload_folder_calls.append({"repo_id": repo_id, "path_in_repo": path_in_repo})
        # Make _upload's post-upload verification see the committed files.
        self.tree_files.append(f"{path_in_repo.rstrip('/')}/adapter_model.safetensors")

    def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
        self.upload_file_calls.append({"repo_id": repo_id, "path_in_repo": path_in_repo})

    def list_repo_tree(self, *, repo_id, repo_type, revision, recursive, path_in_repo=None):
        # ``path_in_repo`` accepted since #988 (the _upload verify is scoped);
        # this fake returns the full recorded set — the helper's defensive
        # client-side prefix filter preserves the semantics.
        from huggingface_hub.hf_api import RepoFile

        return [RepoFile(path=p, size=1, blob_id="b", oid="o") for p in self.tree_files]


def _env(tmp_path, **extra):
    """patch.dict env with the binding cache + event-sink tmp isolation."""
    env = {
        "EPM_HF_STORAGE_CACHE_PATH": str(tmp_path / "hf_storage_usage.json"),
        "EPM_HF_OVERFLOW_EVENT_PATH": str(tmp_path / "overflow-events.jsonl"),
        "HF_TOKEN": "test-token",
    }
    env.update(extra)
    return patch.dict("os.environ", env, clear=True)


@pytest.fixture(autouse=True)
def _reset_blind_flag():
    """The armed-but-blind warning is once-per-process; reset per test."""
    hub._OVERFLOW_BLIND_WARNED = False
    yield
    hub._OVERFLOW_BLIND_WARNED = False


# ---------------------------------------------------------------------------
# Helper: check_hf_storage_headroom
# ---------------------------------------------------------------------------


class TestCheckHfStorageHeadroom:
    def test_sum_and_threshold_under_ceiling(self, tmp_path):
        """Test 1: models + datasets sum, exact used_tb, under default ceiling."""
        fake = FakeHfApi(
            models=[(f"{NS}/m1", False), (f"{NS}/m2", False)],
            datasets=[(f"{NS}/d1", False)],
            used={f"{NS}/m1": 2 * TB, f"{NS}/m2": 1 * TB, f"{NS}/d1": TB // 2},
        )
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert h.over_ceiling is False
        assert h.used_tb == pytest.approx(3.5)
        assert h.n_repos == 3
        assert h.basis == "live-api"

    def test_over_ceiling(self, tmp_path):
        """Test 2: usage above the 10.0 TB default ceiling flips over_ceiling."""
        fake = FakeHfApi(models=[(f"{NS}/big", False)], used={f"{NS}/big": 11 * TB})
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert h.over_ceiling is True
        assert h.used_tb == pytest.approx(11.0)

    def test_private_repos_excluded(self, tmp_path):
        """Test 3: private repos are filtered BEFORE the per-repo info stage."""
        fake = FakeHfApi(
            models=[(f"{NS}/pub", False), (f"{NS}/priv", True)],
            used={f"{NS}/pub": 1 * TB, f"{NS}/priv": 99 * TB},
        )
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert h.used_tb == pytest.approx(1.0)
        assert h.n_repos == 1
        info_rids = [rid for rid, _, _ in fake.info_calls]
        assert f"{NS}/priv" not in info_rids

    def test_partial_none_poisons_to_suspect(self, tmp_path):
        """Test 4: ANY absent/None usedStorage poisons the probe (None != 0)."""
        fake = FakeHfApi(
            models=[(f"{NS}/a", False), (f"{NS}/b", False), (f"{NS}/c", False)],
            used={f"{NS}/a": 5 * TB, f"{NS}/b": None, f"{NS}/c": 4 * TB},
        )
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert h.used_tb is None
        assert h.over_ceiling is False
        assert "suspect (1/3 missing usedStorage)" in h.basis

    def test_cache_hit_within_ttl(self, tmp_path):
        """Test 5: second call within TTL is served from cache — one live probe."""
        fake = FakeHfApi(models=[(f"{NS}/m", False)], used={f"{NS}/m": 1 * TB})
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake) as MockApi:
            h1 = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
            h2 = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert MockApi.call_count == 1  # one construction per live probe; zero on hit
        assert len(fake.list_models_calls) == 1
        assert h1.basis == "live-api"
        assert h2.basis.startswith("cache (age ")
        assert h2.used_tb == pytest.approx(h1.used_tb)

    def test_cache_expiry_reprobes(self, tmp_path):
        """Test 6: a stale cache entry is ignored and the probe re-runs live."""
        cache = tmp_path / "c.json"
        cache.write_text(
            json.dumps(
                {"ts": time.time() - 7200, "used_bytes": 1 * TB, "n_repos": 1, "namespace": NS}
            )
        )
        fake = FakeHfApi(models=[(f"{NS}/m", False)], used={f"{NS}/m": 2 * TB})
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h = check_hf_storage_headroom(cache_path=cache)
        assert len(fake.list_models_calls) == 1
        assert h.basis == "live-api"
        assert h.used_tb == pytest.approx(2.0)

    def test_corrupt_cache_ignored(self, tmp_path):
        """Test 7: corrupt cache JSON is fail-soft — re-probe, no raise."""
        cache = tmp_path / "c.json"
        cache.write_text("{not json!!")
        fake = FakeHfApi(models=[(f"{NS}/m", False)], used={f"{NS}/m": 1 * TB})
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h = check_hf_storage_headroom(cache_path=cache)
        assert h.basis == "live-api"
        assert h.used_tb == pytest.approx(1.0)

    def test_force_refresh_bypasses_fresh_cache(self, tmp_path):
        """Test 8: force_refresh=True skips a valid cache and probes live."""
        fake = FakeHfApi(models=[(f"{NS}/m", False)], used={f"{NS}/m": 1 * TB})
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            check_hf_storage_headroom(cache_path=tmp_path / "c.json")
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json", force_refresh=True)
        assert len(fake.list_models_calls) == 2
        assert h.basis == "live-api"

    def test_listing_exception_is_unknown(self, tmp_path):
        """Test 9a: listing-level API failure -> unknown, never raises."""
        fake = FakeHfApi(list_exc=RuntimeError("api down"))
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert h.used_tb is None
        assert h.over_ceiling is False
        assert "api down" in h.basis

    def test_per_repo_exception_poisons_no_partial_sum(self, tmp_path):
        """Test 9b: one per-repo info failure poisons the whole probe."""
        fake = FakeHfApi(
            models=[(f"{NS}/a", False), (f"{NS}/b", False)],
            used={f"{NS}/a": 1 * TB, f"{NS}/b": RuntimeError("info 500")},
        )
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert h.used_tb is None
        assert "unknown" in h.basis

    def test_ceiling_env_override_and_invalid_values_raise(self, tmp_path):
        """Test 10: ceiling env honored; non-parseable ceiling/TTL raise ValueError."""
        fake = FakeHfApi(models=[(f"{NS}/m", False)], used={f"{NS}/m": 3 * TB})
        with (
            _env(tmp_path, EPM_HF_STORAGE_SOFT_CEILING_TB="2.0"),
            patch("huggingface_hub.HfApi", return_value=fake),
        ):
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert h.ceiling_tb == 2.0
        assert h.over_ceiling is True

        with (
            _env(tmp_path, EPM_HF_STORAGE_SOFT_CEILING_TB="ten"),
            pytest.raises(ValueError, match="EPM_HF_STORAGE_SOFT_CEILING_TB"),
        ):
            check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        with (
            _env(tmp_path, EPM_HF_STORAGE_CACHE_TTL_S="soon"),
            pytest.raises(ValueError, match="EPM_HF_STORAGE_CACHE_TTL_S"),
        ):
            check_hf_storage_headroom(cache_path=tmp_path / "c.json")

    def test_kill_switch_disables_with_zero_io(self, tmp_path):
        """Test 11: EPM_HF_STORAGE_CHECK=0 -> basis 'disabled', zero API calls."""
        with (
            _env(tmp_path, EPM_HF_STORAGE_CHECK="0"),
            patch("huggingface_hub.HfApi", new=MagicMock()) as MockApi,
        ):
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert h.basis == "disabled"
        assert h.used_tb is None
        assert h.over_ceiling is False
        assert MockApi.call_count == 0
        assert not (tmp_path / "c.json").exists()

    def test_suspect_results_never_cached(self, tmp_path):
        """Test 11b: a poisoned probe leaves no cache; the next call re-probes."""
        fake = FakeHfApi(
            models=[(f"{NS}/a", False), (f"{NS}/b", False)],
            used={f"{NS}/a": 1 * TB, f"{NS}/b": None},
        )
        cache = tmp_path / "c.json"
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h1 = check_hf_storage_headroom(cache_path=cache)
            assert not cache.exists()
            h2 = check_hf_storage_headroom(cache_path=cache)
        assert h1.used_tb is None and h2.used_tb is None
        assert len(fake.list_models_calls) == 2  # no clean cache hit in between

    def test_all_zero_suspect_guard(self, tmp_path):
        """Test 11c: all-zero usedStorage with n_repos>0 reads as unknown, not headroom."""
        fake = FakeHfApi(
            models=[(f"{NS}/a", False), (f"{NS}/b", False)],
            used={f"{NS}/a": 0, f"{NS}/b": 0},
        )
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            h = check_hf_storage_headroom(cache_path=tmp_path / "c.json")
        assert h.used_tb is None
        assert h.basis == "suspect (all usedStorage empty)"
        assert not (tmp_path / "c.json").exists()


# ---------------------------------------------------------------------------
# Minute-1 persist gate: trainer._validate_persist_headroom
# ---------------------------------------------------------------------------

PERSIST_ENV = {
    "EPM_PERSIST_ADAPTER_HF_REPO": DEFAULT_MODEL_REPO,
    "EPM_PERSIST_ADAPTER_SUBFOLDER": "issue564/cell",
}


def _gate():
    from explore_persona_space.train.trainer import _validate_persist_headroom

    return _validate_persist_headroom


class TestValidatePersistHeadroom:
    def test_noop_when_persist_env_unset(self, tmp_path):
        """Test 15: no persist declaration -> no-op, zero API calls."""
        with _env(tmp_path), patch("huggingface_hub.HfApi", new=MagicMock()) as MockApi:
            _gate()()
        assert MockApi.call_count == 0

    def test_repo_without_subfolder_raises(self, tmp_path):
        """Test 16: REPO set without SUBFOLDER is a minute-1 contract error."""
        with (
            _env(tmp_path, EPM_PERSIST_ADAPTER_HF_REPO=DEFAULT_MODEL_REPO),
            pytest.raises(RuntimeError, match="EPM_PERSIST_ADAPTER_SUBFOLDER"),
        ):
            _gate()()

    def test_over_ceiling_public_target_routing_off_raises(self, tmp_path):
        """Test 17: confirmed over-ceiling + public target + routing off -> abort,
        and the abort only fires after a FORCED live re-probe (2nd probe)."""
        fake = FakeHfApi(
            models=[(f"{NS}/big", False)],
            used={f"{NS}/big": 11 * TB},
            repo_info_private=False,
        )
        with (
            _env(tmp_path, **PERSIST_ENV),
            patch("huggingface_hub.HfApi", return_value=fake),
            pytest.raises(RuntimeError, match="soft ceiling"),
        ):
            _gate()()
        assert len(fake.list_models_calls) == 2  # initial probe + forced live re-probe

    def test_stale_over_cache_but_live_under_passes(self, tmp_path):
        """Test 18: an over-ceiling CACHE entry never aborts when the live
        re-probe reads under ceiling (the user freed quota mid-TTL)."""
        cache = tmp_path / "hf_storage_usage.json"
        cache.write_text(
            json.dumps({"ts": time.time(), "used_bytes": 11 * TB, "n_repos": 1, "namespace": NS})
        )
        fake = FakeHfApi(
            models=[(f"{NS}/big", False)],
            used={f"{NS}/big": 1 * TB},
            repo_info_private=False,
        )
        with _env(tmp_path, **PERSIST_ENV), patch("huggingface_hub.HfApi", return_value=fake):
            _gate()()  # no raise
        assert len(fake.list_models_calls) == 1  # only the forced live re-probe ran

    def test_unknown_headroom_fails_open(self, tmp_path, caplog):
        """Test 19: unknown headroom -> WARN + continue, never an abort."""
        fake = FakeHfApi(list_exc=RuntimeError("api down"))
        with (
            _env(tmp_path, **PERSIST_ENV),
            patch("huggingface_hub.HfApi", return_value=fake),
            caplog.at_level(logging.WARNING),
        ):
            _gate()()
        assert "fail-open" in caplog.text

    def test_overflow_or_private_target_exempt(self, tmp_path):
        """Test 20: overflow-repo target and repo_info-private target both pass."""
        fake = FakeHfApi(models=[(f"{NS}/big", False)], used={f"{NS}/big": 11 * TB})
        env = dict(PERSIST_ENV, EPM_PERSIST_ADAPTER_HF_REPO=DEFAULT_OVERFLOW_REPO)
        with _env(tmp_path, **env), patch("huggingface_hub.HfApi", return_value=fake):
            _gate()()  # no raise; string match, repo_info never needed
        assert fake.repo_info_calls == []

        fake_priv = FakeHfApi(
            models=[(f"{NS}/big", False)],
            used={f"{NS}/big": 11 * TB},
            repo_info_private=True,
        )
        with (
            _env(tmp_path, **PERSIST_ENV),
            patch("huggingface_hub.HfApi", return_value=fake_priv),
        ):
            _gate()()  # no raise: private target has its own quota

    def test_privacy_undeterminable_fails_open(self, tmp_path, caplog):
        """Test 20b: repo_info failure -> tri-state None -> fail-open, NOT abort."""
        fake = FakeHfApi(
            models=[(f"{NS}/big", False)],
            used={f"{NS}/big": 11 * TB},
            repo_info_exc=RuntimeError("repo_info 500"),
        )
        with (
            _env(tmp_path, **PERSIST_ENV),
            patch("huggingface_hub.HfApi", return_value=fake),
            caplog.at_level(logging.WARNING),
        ):
            _gate()()  # no raise
        assert "undeterminable" in caplog.text

    def test_routing_armed_warns_and_continues(self, tmp_path, caplog):
        """Test 21: over ceiling with routing armed -> WARN naming the
        reroute + arming contract, continue."""
        fake = FakeHfApi(
            models=[(f"{NS}/big", False)],
            used={f"{NS}/big": 11 * TB},
            repo_info_private=False,
        )
        with (
            _env(tmp_path, EPM_HF_OVERFLOW_ROUTING="1", **PERSIST_ENV),
            patch("huggingface_hub.HfApi", return_value=fake),
            caplog.at_level(logging.WARNING),
        ):
            _gate()()  # no raise
        assert "reroute" in caplog.text
        assert "must not arm routing" in caplog.text


# ---------------------------------------------------------------------------
# Wiring: a green suite must be impossible with the gate unwired
# ---------------------------------------------------------------------------


class TestGateWiring:
    def test_init_phase_gate_fires_before_model_load(self, tmp_path):
        """Test 21b: _init_phase raises from the gate BEFORE load_model_and_tokenizer."""
        from omegaconf import OmegaConf

        import explore_persona_space.train.trainer as trainer_mod

        cfg = OmegaConf.create({"training": {"model_id": "fake/model", "max_seq_length": 128}})
        with (
            patch.object(
                trainer_mod, "_validate_persist_headroom", side_effect=RuntimeError("gate boom")
            ),
            patch.object(trainer_mod, "load_model_and_tokenizer") as loader,
            pytest.raises(RuntimeError, match="gate boom"),
        ):
            trainer_mod._init_phase(cfg, "phase1", str(tmp_path), None, 0)
        loader.assert_not_called()

    def test_train_lora_gate_fires_before_model_load(self, tmp_path):
        """Test 21c: train_lora raises from the gate BEFORE any from_pretrained."""
        import transformers

        import explore_persona_space.train.trainer as trainer_mod
        from explore_persona_space.train.sft import train_lora

        with (
            patch.object(
                trainer_mod, "_validate_persist_headroom", side_effect=RuntimeError("gate boom")
            ),
            patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as loader,
            patch.object(transformers.AutoTokenizer, "from_pretrained") as tok,
            pytest.raises(RuntimeError, match="gate boom"),
        ):
            train_lora("fake/model", str(tmp_path / "x.jsonl"), str(tmp_path / "out"))
        loader.assert_not_called()
        tok.assert_not_called()


# ---------------------------------------------------------------------------
# Opt-in overflow routing through upload_model
# ---------------------------------------------------------------------------


def _adapter_dir(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_model.safetensors").write_text("weights")
    return d


class TestOverflowRouting:
    def test_routing_on_over_ceiling_reroutes(self, tmp_path):
        """Test 22: armed + over ceiling -> overflow repo, private create_repo,
        deviation event JSONL, canonical-repo OVERFLOW_POINTER.json attempted."""
        model_dir = _adapter_dir(tmp_path)
        fake = FakeHfApi(
            models=[(f"{NS}/big", False)],
            used={f"{NS}/big": 11 * TB},
            repo_info_private=False,
        )
        with (
            _env(tmp_path, EPM_HF_OVERFLOW_ROUTING="1"),
            patch("huggingface_hub.HfApi", return_value=fake),
        ):
            result = upload_model(str(model_dir), path_in_repo="issue564/cellA")
        assert result == f"{DEFAULT_OVERFLOW_REPO}/issue564/cellA"
        assert fake.upload_folder_calls[0]["repo_id"] == DEFAULT_OVERFLOW_REPO
        create = fake.create_repo_calls[0]
        assert create["repo_id"] == DEFAULT_OVERFLOW_REPO
        assert create["private"] is True
        # Deviation event at the env-pointed sink
        events = [
            json.loads(line)
            for line in (tmp_path / "overflow-events.jsonl").read_text().splitlines()
        ]
        assert events[0]["original_repo"] == DEFAULT_MODEL_REPO
        assert events[0]["effective_repo"] == DEFAULT_OVERFLOW_REPO
        assert events[0]["path_in_repo"] == "issue564/cellA"
        assert events[0]["used_tb"] == pytest.approx(11.0)
        # Canonical-repo pointer breadcrumb attempted
        ptr = fake.upload_file_calls[0]
        assert ptr["repo_id"] == DEFAULT_MODEL_REPO
        assert ptr["path_in_repo"] == "issue564/cellA/OVERFLOW_POINTER.json"

    def test_routing_off_over_ceiling_no_reroute_zero_io(self, tmp_path):
        """Test 23: routing off -> canonical repo, private=False, ZERO headroom I/O."""
        model_dir = _adapter_dir(tmp_path)
        fake = FakeHfApi(
            models=[(f"{NS}/big", False)],
            used={f"{NS}/big": 11 * TB},
            repo_info_private=False,
        )
        with _env(tmp_path), patch("huggingface_hub.HfApi", return_value=fake):
            result = upload_model(str(model_dir), path_in_repo="issue564/cellA")
        assert result == f"{DEFAULT_MODEL_REPO}/issue564/cellA"
        assert fake.upload_folder_calls[0]["repo_id"] == DEFAULT_MODEL_REPO
        assert fake.create_repo_calls[0]["private"] is False
        assert fake.list_models_calls == []  # env short-circuit: no headroom probe
        assert not (tmp_path / "hf_storage_usage.json").exists()
        assert not (tmp_path / "overflow-events.jsonl").exists()

    def test_routing_on_under_ceiling_no_reroute(self, tmp_path):
        """Test 23 (cont.): armed but under ceiling -> canonical repo."""
        model_dir = _adapter_dir(tmp_path)
        fake = FakeHfApi(
            models=[(f"{NS}/small", False)],
            used={f"{NS}/small": 1 * TB},
            repo_info_private=False,
        )
        with (
            _env(tmp_path, EPM_HF_OVERFLOW_ROUTING="1"),
            patch("huggingface_hub.HfApi", return_value=fake),
        ):
            result = upload_model(str(model_dir), path_in_repo="issue564/cellA")
        assert result == f"{DEFAULT_MODEL_REPO}/issue564/cellA"
        assert fake.create_repo_calls[0]["private"] is False
        assert not (tmp_path / "overflow-events.jsonl").exists()

    def test_target_already_overflow_no_reroute(self, tmp_path):
        """Test 23 (cont.): direct overflow target never re-reroutes; the
        overflow repo is still created PRIVATE."""
        model_dir = _adapter_dir(tmp_path)
        fake = FakeHfApi(
            models=[(f"{NS}/big", False)],
            used={f"{NS}/big": 11 * TB},
        )
        with (
            _env(tmp_path, EPM_HF_OVERFLOW_ROUTING="1"),
            patch("huggingface_hub.HfApi", return_value=fake),
        ):
            result = upload_model(
                str(model_dir), repo_id=DEFAULT_OVERFLOW_REPO, path_in_repo="issue564/cellA"
            )
        assert result == f"{DEFAULT_OVERFLOW_REPO}/issue564/cellA"
        assert fake.list_models_calls == []  # short-circuit before any probe
        create = fake.create_repo_calls[0]
        assert create["repo_id"] == DEFAULT_OVERFLOW_REPO
        assert create["private"] is True
        assert not (tmp_path / "overflow-events.jsonl").exists()

    def test_routing_armed_but_blind_warns_once(self, tmp_path, caplog):
        """Test 23b: armed + kill switch -> no reroute, ONE loud warning per process."""
        model_dir = _adapter_dir(tmp_path)
        fake = FakeHfApi()
        with (
            _env(tmp_path, EPM_HF_OVERFLOW_ROUTING="1", EPM_HF_STORAGE_CHECK="0"),
            patch("huggingface_hub.HfApi", return_value=fake),
            caplog.at_level(logging.WARNING),
        ):
            upload_model(str(model_dir), path_in_repo="issue564/cellA")
            upload_model(str(model_dir), path_in_repo="issue564/cellB")
        assert fake.upload_folder_calls[0]["repo_id"] == DEFAULT_MODEL_REPO
        blind_warnings = [r for r in caplog.records if "BLIND" in r.getMessage()]
        assert len(blind_warnings) == 1

    def test_routing_on_private_target_no_reroute(self, tmp_path):
        """Test 23c: a private canonical target keeps its own quota — no reroute."""
        model_dir = _adapter_dir(tmp_path)
        fake = FakeHfApi(
            models=[(f"{NS}/big", False)],
            used={f"{NS}/big": 11 * TB},
            repo_info_private=True,
        )
        with (
            _env(tmp_path, EPM_HF_OVERFLOW_ROUTING="1"),
            patch("huggingface_hub.HfApi", return_value=fake),
        ):
            result = upload_model(
                str(model_dir), repo_id=f"{NS}/my-private-repo", path_in_repo="issue564/cellA"
            )
        assert result == f"{NS}/my-private-repo/issue564/cellA"
        assert fake.upload_folder_calls[0]["repo_id"] == f"{NS}/my-private-repo"
        assert not (tmp_path / "overflow-events.jsonl").exists()


# ---------------------------------------------------------------------------
# Size-aware projected-upload headroom (#1034)
# ---------------------------------------------------------------------------

GB = 10**9  # decimal bytes per GB (1000 GB = 1 TB, matching _BYTES_PER_TB)


def _h(**kw):
    """HfStorageHeadroom stand-in returned by the mocked inner probe."""
    base = dict(used_tb=3.0, ceiling_tb=10.0, over_ceiling=False, basis="live-api", n_repos=5)
    base.update(kw)
    return hub.HfStorageHeadroom(**base)


def _probe_autospec(ret):
    """Signature-conformant fake for the inner probe (autospec — a wrong-kwarg
    call from the wrapper fails loud, per the #906 boundary-fake discipline)."""
    return create_autospec(check_hf_storage_headroom, return_value=ret)


class TestCheckProjectedUploadHeadroom:
    def test_below_threshold_zero_probe_io(self, tmp_path):
        """#1034 test 1: below the 100 GB default floor -> ZERO headroom I/O."""
        probe = _probe_autospec(_h())
        with _env(tmp_path), patch.object(hub, "check_hf_storage_headroom", probe):
            ph = hub.check_projected_upload_headroom(50 * GB)
        assert ph.verdict == "below-threshold"
        assert ph.basis == "not-probed"
        assert ph.used_tb is None and ph.ceiling_tb is None
        assert probe.call_count == 0

    def test_boundary_projected_equals_floor_probes(self, tmp_path):
        """#1034 test 1 boundary: projected == floor EXACTLY (100 GB) DOES
        probe — the skip is strict `<` (pins against an inclusive-compare
        mutation)."""
        probe = _probe_autospec(_h())
        with _env(tmp_path), patch.object(hub, "check_hf_storage_headroom", probe):
            ph = hub.check_projected_upload_headroom(100 * GB)
        assert probe.call_count == 1
        assert ph.verdict == "fits"

    def test_floor_env_override_and_invalid_raise(self, tmp_path):
        """#1034 test 2: EPM_HF_LARGE_UPLOAD_PROBE_GB=1 -> a 2 GB projection
        probes; a non-parseable env value raises (the _env_float contract)."""
        probe = _probe_autospec(_h())
        with (
            _env(tmp_path, EPM_HF_LARGE_UPLOAD_PROBE_GB="1"),
            patch.object(hub, "check_hf_storage_headroom", probe),
        ):
            ph = hub.check_projected_upload_headroom(2 * GB)
        assert probe.call_count == 1
        assert ph.verdict == "fits"

        with (
            _env(tmp_path, EPM_HF_LARGE_UPLOAD_PROBE_GB="hundred"),
            pytest.raises(ValueError, match="EPM_HF_LARGE_UPLOAD_PROBE_GB"),
        ):
            hub.check_projected_upload_headroom(2 * GB)

    def test_fits_single_probe(self, tmp_path):
        """#1034 test 3: used 3 + projected 2 <= ceiling 10 -> fits, ONE probe."""
        probe = _probe_autospec(_h())
        with _env(tmp_path), patch.object(hub, "check_hf_storage_headroom", probe):
            ph = hub.check_projected_upload_headroom(2000 * GB)
        assert ph.verdict == "fits"
        assert ph.used_tb == pytest.approx(3.0)
        assert ph.projected_tb == pytest.approx(2.0)
        assert probe.call_count == 1

    def test_boundary_exact_ceiling_fits(self, tmp_path):
        """#1034 test 3 boundary: used + projected == ceiling EXACTLY -> fits
        (`<=`, consistent with the legacy over_ceiling = used > ceiling)."""
        probe = _probe_autospec(_h(used_tb=8.0))
        with _env(tmp_path), patch.object(hub, "check_hf_storage_headroom", probe):
            ph = hub.check_projected_upload_headroom(2000 * GB)
        assert ph.verdict == "fits"
        assert probe.call_count == 1

    def test_insufficient_requires_live_confirm(self, tmp_path):
        """#1034 test 4: over on the cached read -> the SECOND probe carries
        force_refresh=True -> still over -> insufficient."""
        calls: list[bool] = []

        def probe(*a, force_refresh=False, **k):
            calls.append(force_refresh)
            return _h(used_tb=9.5)

        with _env(tmp_path), patch.object(hub, "check_hf_storage_headroom", probe):
            ph = hub.check_projected_upload_headroom(2000 * GB)
        assert ph.verdict == "insufficient"
        assert calls == [False, True]

    def test_live_confirm_flips_to_fits(self, tmp_path):
        """#1034 test 5: stale-high cache (9.5), live re-probe 7.0 -> fits —
        never act on a stale cached over-read."""

        def probe(*a, force_refresh=False, **k):
            return _h(used_tb=7.0) if force_refresh else _h(used_tb=9.5)

        with _env(tmp_path), patch.object(hub, "check_hf_storage_headroom", probe):
            ph = hub.check_projected_upload_headroom(2000 * GB)
        assert ph.verdict == "fits"
        assert ph.used_tb == pytest.approx(7.0)

    def test_live_confirm_unknown_degrades_to_unknown(self, tmp_path):
        """#1034 test 6: live re-probe unknown -> unknown (fail-open)."""

        def probe(*a, force_refresh=False, **k):
            if force_refresh:
                return _h(used_tb=None, basis="unknown (api down)")
            return _h(used_tb=9.5)

        with _env(tmp_path), patch.object(hub, "check_hf_storage_headroom", probe):
            ph = hub.check_projected_upload_headroom(2000 * GB)
        assert ph.verdict == "unknown"

    def test_disabled_and_unknown_first_probe_no_second_probe(self, tmp_path):
        """#1034 test 7: kill switch -> disabled; unknown first probe ->
        unknown — NEITHER issues a second (force_refresh) probe."""
        probe = _probe_autospec(_h(used_tb=None, basis="disabled"))
        with _env(tmp_path), patch.object(hub, "check_hf_storage_headroom", probe):
            ph = hub.check_projected_upload_headroom(2000 * GB)
        assert ph.verdict == "disabled"
        assert probe.call_count == 1

        probe2 = _probe_autospec(_h(used_tb=None, basis="unknown (api down)"))
        with _env(tmp_path), patch.object(hub, "check_hf_storage_headroom", probe2):
            ph2 = hub.check_projected_upload_headroom(2000 * GB)
        assert ph2.verdict == "unknown"
        assert probe2.call_count == 1

    def test_negative_projected_asserts(self, tmp_path):
        """Fail-fast on a nonsensical projection (caller bug, never coerced)."""
        with _env(tmp_path), pytest.raises(AssertionError):
            hub.check_projected_upload_headroom(-1)


class TestResolveLfsUploadRepoSizeAware:
    def test_armed_projected_pushes_over_reroutes(self, tmp_path):
        """#1034 test 8: armed + used 9 + projected 2 TB (ceiling 10) + public
        target -> reroutes (used + projected > ceiling)."""
        with (
            _env(tmp_path, EPM_HF_OVERFLOW_ROUTING="1"),
            patch.object(hub, "check_hf_storage_headroom", _probe_autospec(_h(used_tb=9.0))),
            patch.object(
                hub, "_repo_is_private", create_autospec(hub._repo_is_private, return_value=False)
            ),
        ):
            repo, rerouted = hub._resolve_lfs_upload_repo(
                DEFAULT_MODEL_REPO, projected_bytes=2000 * GB
            )
        assert (repo, rerouted) == (DEFAULT_OVERFLOW_REPO, True)

    def test_armed_no_projection_keeps_legacy_binary(self, tmp_path):
        """#1034 test 8 (cont.): projected_bytes=None reproduces the legacy
        binary over-ceiling check — used 9 < ceiling 10 -> NO reroute."""
        with (
            _env(tmp_path, EPM_HF_OVERFLOW_ROUTING="1"),
            patch.object(hub, "check_hf_storage_headroom", _probe_autospec(_h(used_tb=9.0))),
        ):
            repo, rerouted = hub._resolve_lfs_upload_repo(DEFAULT_MODEL_REPO)
        assert (repo, rerouted) == (DEFAULT_MODEL_REPO, False)

    def test_unarmed_huge_projection_zero_probe(self, tmp_path):
        """#1034 test 8 (cont.): unarmed -> never reroutes, ZERO headroom I/O
        (the ARMING CONTRACT is untouched by the size-aware predicate)."""
        probe = _probe_autospec(_h(used_tb=9.0))
        with _env(tmp_path), patch.object(hub, "check_hf_storage_headroom", probe):
            repo, rerouted = hub._resolve_lfs_upload_repo(
                DEFAULT_MODEL_REPO, projected_bytes=10**15
            )
        assert (repo, rerouted) == (DEFAULT_MODEL_REPO, False)
        assert probe.call_count == 0

    def test_upload_model_threads_dir_size_end_to_end(self, tmp_path):
        """#1034 test 8b (Must-Fix): armed + used = ceiling - epsilon (UNDER the
        ceiling absolutely, over once this dir's bytes are added) -> the
        upload lands in the overflow repo. A mutation dropping
        `projected_bytes=projected` from upload_model's resolver call MUST
        fail this test (with None the legacy binary check reads under-ceiling
        -> no reroute). Doubles as the pin that upload_model actually computes
        + threads the dir-size walk (AC 6)."""
        model_dir = _adapter_dir(tmp_path)
        dir_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        assert dir_bytes > 1
        fake = FakeHfApi(
            models=[(f"{NS}/big", False)],
            # ε = dir_bytes // 2 + 1 bytes under the 10 TB default ceiling:
            # under absolutely, over after the projected dir_bytes.
            used={f"{NS}/big": 10 * TB - dir_bytes // 2 - 1},
            repo_info_private=False,
        )
        with (
            _env(tmp_path, EPM_HF_OVERFLOW_ROUTING="1"),
            patch("huggingface_hub.HfApi", return_value=fake),
        ):
            result = upload_model(str(model_dir), path_in_repo="issue1034/cellA")
        assert result == f"{DEFAULT_OVERFLOW_REPO}/issue1034/cellA"
        assert fake.upload_folder_calls[0]["repo_id"] == DEFAULT_OVERFLOW_REPO
        # Deviation event carries the backward-compatible default reason.
        events = [
            json.loads(line)
            for line in (tmp_path / "overflow-events.jsonl").read_text().split("\n")
            if line.strip()
        ]
        assert events[0]["reason"] == "quota-403-reactive"
