"""Tests for ``scripts/runpod_api.py`` retry/backoff + supply-resilience.

Covers:
- #2 graphql() bounded exponential backoff: retries transient transport
  failures (5xx, 429, CF-1010, network), does NOT retry non-transient 4xx or
  GraphQL-level errors.
- #11 create_pod() supply-resilience: tries an ordered gpuType list, falls
  through cloud_type COMMUNITY then COMMUNITY+interruptible on SUPPLY_CONSTRAINT,
  preserves dataCenterId. (gpuTypePriority is NOT sent — RunPod rejects it
  with HTTP 400.)
- #537 SUPPLY_CONSTRAINT delivered as a GraphQL ERROR payload (not a null
  mutation result) is classified as the no-capacity case: _deploy_once returns
  None, the lever chain advances, and create_pod raises RunPodNoCapacityError
  only after every lever — instead of crashing with a bare RunPodError.
- #537 gpu-type validation: an unmapped colloquial short name ("H100 SXM")
  raises RunPodError naming the valid options BEFORE any API call, instead of
  passing through verbatim as a nonexistent gpuTypeId that RunPod reports as
  phantom no-capacity forever. Full vendor-prefixed ids still pass through.

These tests stub network at the ``_graphql_once`` / ``graphql`` seam so they run
without API access, and stub ``time.sleep`` so they don't actually wait.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib import error as urlerror

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import runpod_api  # noqa: E402
from runpod_api import (  # noqa: E402
    RunPodError,
    RunPodNoCapacityError,
    RunPodSupplyConstraintError,
    RunPodTransientError,
    _is_cloudflare_1010,
    create_pod,
    graphql,
)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Never actually sleep during backoff in tests."""
    monkeypatch.setattr(runpod_api.time, "sleep", lambda _secs: None)


# ---------------------------------------------------------------------------
# #2 — graphql() retry / backoff
# ---------------------------------------------------------------------------


def _once_stub(monkeypatch):
    """Install a settable _graphql_once stub; return a recorder object.

    ``recorder.outcomes`` is a list consumed one-per-call; each entry is either
    a callable raising/returning, or a dict (returned as data).
    """

    class _Rec:
        def __init__(self):
            self.outcomes: list = []
            self.calls = 0

        def __call__(self, query, variables, timeout):
            self.calls += 1
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    rec = _Rec()
    monkeypatch.setattr(runpod_api, "_graphql_once", rec)
    return rec


def test_graphql_retries_on_transient_then_succeeds(monkeypatch):
    """A 503 (transient) followed by success returns the success payload."""
    rec = _once_stub(monkeypatch)
    rec.outcomes = [
        RunPodTransientError("HTTP 503 from RunPod: gateway"),
        {"ok": True},
    ]
    out = graphql("query {}")
    assert out == {"ok": True}
    assert rec.calls == 2  # one retry


def test_graphql_retries_up_to_max_attempts(monkeypatch):
    """All-transient failures exhaust the 4-attempt budget, then raise once."""
    rec = _once_stub(monkeypatch)
    rec.outcomes = [RunPodTransientError(f"HTTP 500 attempt {i}") for i in range(10)]
    with pytest.raises(RunPodError) as exc:
        graphql("query {}")
    assert rec.calls == runpod_api.GRAPHQL_MAX_ATTEMPTS
    assert "after 4 attempts" in str(exc.value)


def test_graphql_no_retry_on_400(monkeypatch):
    """A non-transient 4xx surfaces immediately — no retry."""
    rec = _once_stub(monkeypatch)
    rec.outcomes = [RunPodError("HTTP 400 from RunPod: bad query"), {"ok": True}]
    with pytest.raises(RunPodError) as exc:
        graphql("query {}")
    assert rec.calls == 1  # did NOT retry
    assert "400" in str(exc.value)


def test_graphql_no_retry_on_graphql_errors(monkeypatch):
    """A GraphQL-level `errors` payload (non-transient) is not retried."""
    rec = _once_stub(monkeypatch)
    rec.outcomes = [RunPodError("GraphQL errors: [...]"), {"ok": True}]
    with pytest.raises(RunPodError):
        graphql("query {}")
    assert rec.calls == 1


# ---------------------------------------------------------------------------
# #2 — _graphql_once transient classification (HTTPError / URLError / CF-1010)
# ---------------------------------------------------------------------------


class _FakeHTTPError(urlerror.HTTPError):
    def __init__(self, code: int, body: bytes):
        self._body = body
        super().__init__("http://x", code, "err", {}, None)

    def read(self) -> bytes:  # type: ignore[override]
        return self._body


def _patch_urlopen(monkeypatch, *, raises=None, returns_body: bytes | None = None):
    """Patch urlopen + _require_env so _graphql_once runs offline."""
    monkeypatch.setattr(runpod_api, "_require_env", lambda: ("k", "t"))

    class _Resp:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._body

    def fake_urlopen(req, timeout=60):
        if raises is not None:
            raise raises
        return _Resp(returns_body)

    monkeypatch.setattr(runpod_api.urlrequest, "urlopen", fake_urlopen)


def test_graphql_once_500_is_transient(monkeypatch):
    _patch_urlopen(monkeypatch, raises=_FakeHTTPError(503, b"gateway"))
    with pytest.raises(RunPodTransientError):
        runpod_api._graphql_once("q", None, 60)


def test_graphql_once_429_is_transient(monkeypatch):
    _patch_urlopen(monkeypatch, raises=_FakeHTTPError(429, b"slow down"))
    with pytest.raises(RunPodTransientError):
        runpod_api._graphql_once("q", None, 60)


def test_graphql_once_400_is_not_transient(monkeypatch):
    _patch_urlopen(monkeypatch, raises=_FakeHTTPError(400, b"bad request"))
    with pytest.raises(RunPodError) as exc:
        runpod_api._graphql_once("q", None, 60)
    assert not isinstance(exc.value, RunPodTransientError)


def test_graphql_once_network_error_is_transient(monkeypatch):
    _patch_urlopen(monkeypatch, raises=urlerror.URLError("connection refused"))
    with pytest.raises(RunPodTransientError):
        runpod_api._graphql_once("q", None, 60)


def test_graphql_once_cf_1010_in_body_is_transient(monkeypatch):
    """A 200 carrying a Cloudflare 1010 challenge body is treated as transient."""
    _patch_urlopen(monkeypatch, returns_body=b"<html>error code: 1010</html>")
    with pytest.raises(RunPodTransientError) as exc:
        runpod_api._graphql_once("q", None, 60)
    assert "1010" in str(exc.value)


def test_graphql_once_http_error_with_cf_1010_is_transient(monkeypatch):
    """A 403 whose body is a CF-1010 challenge is retryable (not a hard 4xx)."""
    _patch_urlopen(monkeypatch, raises=_FakeHTTPError(403, b"error code: 1010"))
    with pytest.raises(RunPodTransientError):
        runpod_api._graphql_once("q", None, 60)


def test_cf_1010_detector():
    assert _is_cloudflare_1010("blah error code: 1010 blah")
    assert _is_cloudflare_1010("ERROR CODE 1010")
    assert not _is_cloudflare_1010("error code: 1020")
    assert not _is_cloudflare_1010("all good")


def test_backoff_sleep_within_window(monkeypatch):
    """Backoff window grows exponentially and is capped."""
    # attempt 1 window = base; attempt large = cap.
    s1 = runpod_api._backoff_sleep_secs(1)
    assert 0.0 <= s1 <= runpod_api.GRAPHQL_BACKOFF_BASE_SECS
    s_big = runpod_api._backoff_sleep_secs(20)
    assert 0.0 <= s_big <= runpod_api.GRAPHQL_BACKOFF_CAP_SECS


# ---------------------------------------------------------------------------
# #11 — create_pod() supply-resilience
# ---------------------------------------------------------------------------


def _make_pod_payload(pod_id="p1", name="pod-1"):
    return {
        "id": pod_id,
        "name": name,
        "desiredStatus": "RUNNING",
        "gpuCount": 1,
        "createdAt": "2026-05-01T00:00:00Z",
        "machine": {"gpuTypeId": "NVIDIA H100 80GB HBM3"},
        "runtime": {"ports": []},
    }


def _capture_graphql(monkeypatch, results: list):
    """Stub graphql() to return queued podFindAndDeployOnDemand payloads.

    Each entry in ``results`` is either a payload dict or None (no capacity).
    Records every query string in ``recorder.queries``.
    """

    class _Rec:
        def __init__(self):
            self.queries: list[str] = []
            self.results = list(results)

        def __call__(self, query, variables=None, timeout=60):
            self.queries.append(query)
            res = self.results.pop(0)
            return {"podFindAndDeployOnDemand": res}

    rec = _Rec()
    monkeypatch.setattr(runpod_api, "graphql", rec)
    return rec


def test_create_pod_first_gpu_succeeds(monkeypatch):
    rec = _capture_graphql(monkeypatch, [_make_pod_payload()])
    info = create_pod("pod-1", "H100", 1)
    assert info.pod_id == "p1"
    assert len(rec.queries) == 1
    # gpuTypePriority is NOT a valid PodFindAndDeployOnDemandInput field —
    # RunPod rejects it with HTTP 400, so it must never be sent (removed in
    # 0e006d933; this assertion was stale until 2026-06-11).
    assert "gpuTypePriority" not in rec.queries[0]


def test_create_pod_tries_gpu_list_in_order(monkeypatch):
    """First gpu type out of capacity → second is tried, in order."""
    rec = _capture_graphql(monkeypatch, [None, _make_pod_payload()])
    info = create_pod("pod-1", ["H100", "H200"], 1)
    assert info.pod_id == "p1"
    assert len(rec.queries) == 2
    # First attempt used H100's gpuTypeId, second used H200's.
    assert "H100 80GB HBM3" in rec.queries[0]
    assert "NVIDIA H200" in rec.queries[1]


def test_create_pod_falls_through_to_community(monkeypatch):
    """All gpu types out on the primary cloud → COMMUNITY cloud is tried."""
    # 1 gpu type x primary(ALL)=null, then COMMUNITY succeeds.
    rec = _capture_graphql(monkeypatch, [None, _make_pod_payload()])
    info = create_pod("pod-1", "H100", 1, cloud_type="ALL")
    assert info.pod_id == "p1"
    assert "COMMUNITY" not in rec.queries[0]
    assert "cloudType: COMMUNITY" in rec.queries[1]


def test_create_pod_falls_through_to_interruptible(monkeypatch):
    """Primary + COMMUNITY exhausted → the spot lever is DISABLED (the RunPod
    schema rejects `interruptible` with HTTP 400, #537 2026-06-11), so it
    reports no-capacity without an API call and the chain raises."""
    rec = _capture_graphql(monkeypatch, [None, None])
    with pytest.raises(RunPodNoCapacityError):
        create_pod("pod-1", "H100", 1, cloud_type="ALL")
    assert len(rec.queries) == 2
    assert all("interruptible" not in q for q in rec.queries)


def test_create_pod_preserves_data_center_id(monkeypatch):
    """dataCenterId pin is carried into every attempt (#11 must not drop it)."""
    rec = _capture_graphql(monkeypatch, [None, _make_pod_payload()])
    create_pod("pod-1", "H100", 1, data_center_id="EU-RO-1")
    for q in rec.queries:
        assert 'dataCenterId: "EU-RO-1"' in q


def test_create_pod_all_levers_exhausted_raises(monkeypatch):
    """Every lever returns null → a single clear RunPodError naming what failed."""
    rec = _capture_graphql(monkeypatch, [None, None])
    with pytest.raises(RunPodError) as exc:
        create_pod("pod-1", "H100", 1, cloud_type="ALL")
    # SECURE + COMMUNITY hit the API; the disabled spot lever does not.
    assert len(rec.queries) == 2
    assert "no capacity" in str(exc.value).lower()
    assert "Tried" in str(exc.value)


def test_create_pod_no_supply_fallback_single_attempt(monkeypatch):
    """enable_supply_fallback=False → only the primary lever is tried."""
    rec = _capture_graphql(monkeypatch, [None])
    with pytest.raises(RunPodError):
        create_pod("pod-1", "H100", 1, cloud_type="ALL", enable_supply_fallback=False)
    assert len(rec.queries) == 1


def test_create_pod_community_primary_no_duplicate_community(monkeypatch):
    """When cloud_type is already COMMUNITY, the fallback doesn't re-add it.

    Levers: COMMUNITY(non-interruptible), then COMMUNITY interruptible — but
    the spot lever is disabled (schema rejects `interruptible`, #537), so only
    ONE API call happens and exhaustion raises no-capacity."""
    rec = _capture_graphql(monkeypatch, [None])
    with pytest.raises(RunPodNoCapacityError):
        create_pod("pod-1", "H100", 1, cloud_type="COMMUNITY")
    assert len(rec.queries) == 1
    assert all("interruptible" not in q for q in rec.queries)


def test_create_pod_empty_gpu_list_raises(monkeypatch):
    _capture_graphql(monkeypatch, [])
    with pytest.raises(RunPodError):
        create_pod("pod-1", [], 1)


# ---------------------------------------------------------------------------
# #537 — SUPPLY_CONSTRAINT delivered as a GraphQL error payload
# ---------------------------------------------------------------------------

# Verbatim payload shape from the task #537 autonomous-provision crash
# (2026-06-11 ~18:55Z): podFindAndDeployOnDemand refused with a GraphQL
# ERROR carrying extensions.code == SUPPLY_CONSTRAINT instead of the usual
# null mutation result. The bare RunPodError this used to raise bypassed
# BOTH create_pod's supply-lever chain AND the wait-for-capacity loop.
_SUPPLY_CONSTRAINT_BODY = json.dumps(
    {
        "errors": [
            {
                "message": (
                    "There are no longer any instances available with the "
                    "requested specifications. Please refresh and try again."
                ),
                "path": ["podFindAndDeployOnDemand"],
                "extensions": {"code": "SUPPLY_CONSTRAINT", "userId": "u-123"},
            }
        ]
    }
).encode("utf-8")


def test_graphql_once_supply_constraint_payload_is_classified(monkeypatch):
    """The captured #537 error payload raises the typed supply-constraint
    class (a RunPodError subclass, NOT transient — no pointless retries)."""
    _patch_urlopen(monkeypatch, returns_body=_SUPPLY_CONSTRAINT_BODY)
    with pytest.raises(RunPodSupplyConstraintError) as exc:
        runpod_api._graphql_once("q", None, 60)
    assert isinstance(exc.value, RunPodError)
    assert not isinstance(exc.value, RunPodTransientError)
    assert "SUPPLY_CONSTRAINT" in str(exc.value)


def test_supply_constraint_detector():
    assert runpod_api._is_supply_constraint_error(
        '[{"message": "There are no longer any instances available with the '
        'requested specifications.", "extensions": {"code": "SUPPLY_CONSTRAINT"}}]'
    )
    assert runpod_api._is_supply_constraint_error("Not enough free GPUs on host")
    assert not runpod_api._is_supply_constraint_error("over your current spending limit")
    assert not runpod_api._is_supply_constraint_error("")


def _capture_graphql_outcomes(monkeypatch, outcomes: list):
    """Stub graphql() with per-call outcomes: an Exception (raised), None
    (null mutation result), or a pod payload dict (success). Records every
    query string in ``recorder.queries``."""

    class _Rec:
        def __init__(self):
            self.queries: list[str] = []
            self.outcomes = list(outcomes)

        def __call__(self, query, variables=None, timeout=60):
            self.queries.append(query)
            out = self.outcomes.pop(0)
            if isinstance(out, Exception):
                raise out
            return {"podFindAndDeployOnDemand": out}

    rec = _Rec()
    monkeypatch.setattr(runpod_api, "graphql", rec)
    return rec


def test_deploy_once_returns_none_on_supply_constraint_error(monkeypatch):
    """The error-payload shape behaves exactly like the null-result shape."""
    _capture_graphql_outcomes(
        monkeypatch, [RunPodSupplyConstraintError("GraphQL errors: SUPPLY_CONSTRAINT")]
    )
    out = runpod_api._deploy_once(
        name="pod-1",
        gpu_type_id="NVIDIA H100 80GB HBM3",
        gpu_count=1,
        image="img",
        volume_gb=100,
        container_disk_gb=50,
        cloud_type="ALL",
        data_center_id=None,
        interruptible=False,
    )
    assert out is None


def test_create_pod_lever_chain_advances_past_supply_constraint_error(monkeypatch):
    """A SUPPLY_CONSTRAINT error on the primary lever does NOT abort the
    chain — COMMUNITY is tried next and wins (the #537 regression)."""
    rec = _capture_graphql_outcomes(
        monkeypatch,
        [RunPodSupplyConstraintError("GraphQL errors: SUPPLY_CONSTRAINT"), _make_pod_payload()],
    )
    info = create_pod("pod-1", "H100", 1, cloud_type="ALL")
    assert info.pod_id == "p1"
    assert len(rec.queries) == 2
    assert "cloudType: COMMUNITY" in rec.queries[1]


def test_create_pod_all_supply_constraint_errors_raise_no_capacity(monkeypatch):
    """Every lever refused via the error-payload shape → RunPodNoCapacityError
    (the class create_pod_with_wait_for_capacity catches), raised only AFTER
    all levers were tried."""
    err = RunPodSupplyConstraintError("GraphQL errors: SUPPLY_CONSTRAINT")
    rec = _capture_graphql_outcomes(monkeypatch, [err, err])
    with pytest.raises(RunPodNoCapacityError):
        create_pod("pod-1", "H100", 1, cloud_type="ALL")
    assert len(rec.queries) == 2  # ALL + COMMUNITY (spot lever disabled, #537)


def test_create_pod_other_graphql_error_still_fails_fast(monkeypatch):
    """A non-supply GraphQL error (auth, bad config) aborts the chain
    immediately — fail fast is preserved."""
    rec = _capture_graphql_outcomes(monkeypatch, [RunPodError("GraphQL errors: UNAUTHORIZED")])
    with pytest.raises(RunPodError) as exc:
        create_pod("pod-1", "H100", 1, cloud_type="ALL")
    assert not isinstance(exc.value, (RunPodNoCapacityError, RunPodSupplyConstraintError))
    assert len(rec.queries) == 1


# ---------------------------------------------------------------------------
# #537 — gpu-type validation: typo'd short names must not become phantom
# no-capacity waits
# ---------------------------------------------------------------------------


def test_resolve_gpu_type_id_maps_short_name():
    """Mapped short names resolve to their full RunPod gpuTypeId."""
    assert runpod_api.resolve_gpu_type_id("H100") == "NVIDIA H100 80GB HBM3"
    assert runpod_api.resolve_gpu_type_id("H200") == "NVIDIA H200"


def test_resolve_gpu_type_id_passes_through_full_id():
    """Vendor-prefixed full ids pass through verbatim (exotic GPU escape hatch)."""
    assert runpod_api.resolve_gpu_type_id("NVIDIA H100 NVL") == "NVIDIA H100 NVL"
    assert runpod_api.resolve_gpu_type_id("AMD Instinct MI300X OC") == "AMD Instinct MI300X OC"


def test_resolve_gpu_type_id_rejects_unmapped_colloquial_name():
    """'H100 SXM' is neither a short name nor a full id — raise loudly, naming
    the valid options. Passed verbatim it becomes a nonexistent gpuTypeId that
    RunPod reports as no-capacity forever (#537 waited 88 minutes on it)."""
    with pytest.raises(RunPodError) as exc:
        runpod_api.resolve_gpu_type_id("H100 SXM")
    msg = str(exc.value)
    assert "H100 SXM" in msg
    for valid in sorted(runpod_api.GPU_TYPE_IDS):
        assert valid in msg
    assert "NVIDIA" in msg  # names the expected full-id form


def test_create_pod_rejects_unmapped_gpu_type_before_any_api_call(monkeypatch):
    """The validation fires BEFORE the lever loop — zero deploy attempts, and
    the error is a plain RunPodError (NOT RunPodNoCapacityError), so the
    wait-for-capacity policy propagates it instead of retrying forever."""
    rec = _capture_graphql(monkeypatch, [])
    with pytest.raises(RunPodError) as exc:
        create_pod("pod-1", "H100 SXM", 1)
    assert not isinstance(exc.value, RunPodNoCapacityError)
    assert len(rec.queries) == 0


def test_create_pod_rejects_bad_name_anywhere_in_gpu_list(monkeypatch):
    """A bad name in ANY position of the gpu_type list fails fast up front,
    even when the first name is valid."""
    rec = _capture_graphql(monkeypatch, [])
    with pytest.raises(RunPodError):
        create_pod("pod-1", ["H100", "H100 SXM"], 1)
    assert len(rec.queries) == 0


def test_create_pod_full_id_passes_through_to_deploy(monkeypatch):
    """A vendor-prefixed full id reaches the deploy mutation verbatim."""
    rec = _capture_graphql(monkeypatch, [_make_pod_payload()])
    info = create_pod("pod-1", "NVIDIA H100 NVL", 1)
    assert info.pod_id == "p1"
    assert 'gpuTypeId: "NVIDIA H100 NVL"' in rec.queries[0]


def test_interruptible_lever_reports_no_capacity_without_api_call(monkeypatch):
    """#537 (2026-06-11): the RunPod schema rejects the `interruptible` field
    with HTTP 400 GRAPHQL_VALIDATION_FAILED, which crashed the lever chain.
    Until spot is re-implemented, the spot lever must report no-capacity
    WITHOUT hitting the API, so create_pod raises RunPodNoCapacityError after
    the real levers instead of crashing mid-chain."""
    import runpod_api

    calls = []

    def fake_graphql(query, variables=None, timeout=None):
        calls.append(query)
        raise runpod_api.RunPodSupplyConstraintError("no capacity (test)")

    monkeypatch.setattr(runpod_api, "graphql", fake_graphql)
    with pytest.raises(runpod_api.RunPodNoCapacityError):
        runpod_api.create_pod(name="t", gpu_type="H100", gpu_count=8)
    # SECURE + COMMUNITY hit the API; the interruptible lever must NOT.
    assert len(calls) == 2, calls
