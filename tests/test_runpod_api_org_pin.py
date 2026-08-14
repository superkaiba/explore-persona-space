"""Org-scoping pins for ``scripts/runpod_api.py`` (#2059 D2).

RunPod is the FIRST-resort compute lane (#2054), and every GraphQL call MUST
carry the Anthropic Safety Research team header — without it the API silently
returns the PERSONAL-account inventory (zero team pods), the failure mode
behind the `mcp__runpod__*` non-authoritative-tool ban (CLAUDE.md § Pods hard
requirement 1). These tests pin the four layers of that guarantee:

1. the ``DEFAULT_TEAM_ID`` literal + ``_require_env`` resolution semantics
   (default / env override / whitespace-empty refusal);
2. the ``X-Team-Id`` header injection on the real ``_graphql_once`` request
   object (stubbed at the ``urlopen`` network boundary — the header must be
   ON the wire-shaped Request, not merely in a dict somewhere);
3. both provision mutations (GPU ``podFindAndDeployOnDemand`` + CPU
   ``deployCpuPod``) routing through the header-injecting :func:`graphql`
   wrapper;
4. an AST + literal scan proving no OTHER HTTP call site bypasses
   ``_graphql_once`` (the single place the header is set).

Import pattern per tests/test_runpod_api_retry.py (sys.path insert — the
scripts/ dir is not a package).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import runpod_api  # noqa: E402
from runpod_api import (  # noqa: E402
    DEFAULT_TEAM_ID,
    _deploy_cpu_once,
    _deploy_once,
    _require_env,
    graphql,
)

RUNPOD_API_PY = REPO_ROOT / "scripts" / "runpod_api.py"

#: Files under scripts//src/ allowed to carry the ``api.runpod.io`` literal.
#: runpod_api.py is the single sanctioned CALL SITE; redact_for_gist.py names
#: the URL in a redaction-map DOCSTRING only (no network call — verified by
#: the whitelist assert below, which bans HTTP imports there too).
_API_URL_ALLOWLIST = {
    Path("scripts/runpod_api.py"),
    Path("scripts/redact_for_gist.py"),
}


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Deterministic env: a fake API key, no ambient team override, and the
    .env loader stubbed out (the real .env must not leak into resolution)."""
    monkeypatch.setattr(runpod_api, "_load_dotenv", lambda: None)
    monkeypatch.setenv("RUNPOD_API_KEY", "test-api-key-not-real")
    monkeypatch.delenv("RUNPOD_TEAM_ID", raising=False)


def test_default_team_id_pins_anthropic_safety_research(monkeypatch):
    """The team-id literal is the Anthropic Safety Research org, and
    ``_require_env`` resolves: default when unset, env override when set,
    loud refusal on a whitespace-empty override (an empty header would
    silently de-scope every call to the personal account)."""
    assert DEFAULT_TEAM_ID == "cm8ipuyys0004l108gb23hody"

    # Default: no env override -> the pinned org id.
    api_key, team_id = _require_env()
    assert api_key == "test-api-key-not-real"
    assert team_id == DEFAULT_TEAM_ID

    # Env override: an explicit team id wins verbatim (stripped).
    monkeypatch.setenv("RUNPOD_TEAM_ID", "  custom-team-42  ")
    _, team_id = _require_env()
    assert team_id == "custom-team-42"

    # Whitespace-empty override: refuses loud rather than sending an empty
    # header (which the API would treat as un-scoped).
    monkeypatch.setenv("RUNPOD_TEAM_ID", "   ")
    with pytest.raises(RuntimeError, match="RUNPOD_TEAM_ID resolved to empty"):
        _require_env()

    # Missing API key refuses loud too (both credentials are mandatory).
    monkeypatch.delenv("RUNPOD_TEAM_ID", raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY", "")
    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY not set"):
        _require_env()


class _FakeResponse:
    """Minimal context-manager response for the urlopen seam."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc) -> None:
        return None


def _install_urlopen_recorder(monkeypatch, body: bytes = b'{"data": {}}'):
    """Record every Request urlopen receives; serve a canned 200 body."""
    requests: list = []

    def fake_urlopen(req, timeout=None):
        requests.append(req)
        return _FakeResponse(body)

    monkeypatch.setattr(runpod_api.urlrequest, "urlopen", fake_urlopen)
    return requests


def test_graphql_injects_x_team_id_header_on_every_call(monkeypatch):
    """The wire-shaped Request carries X-Team-Id (urllib stores it
    capitalize()d as ``X-team-id`` — verified live at implementation) plus
    the Bearer Authorization, for BOTH the default and an overridden team."""
    requests = _install_urlopen_recorder(monkeypatch)

    data = graphql("query { myself { id } }")
    assert data == {}
    assert len(requests) == 1
    req = requests[0]
    # urllib capitalizes header keys via str.capitalize(): "X-team-id".
    assert req.get_header("X-team-id") == DEFAULT_TEAM_ID, dict(req.header_items())
    assert req.get_header("Authorization") == "Bearer test-api-key-not-real"
    assert req.get_full_url() == runpod_api.GRAPHQL_URL
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["query"] == "query { myself { id } }"

    # A custom team override rides the SAME header slot.
    monkeypatch.setenv("RUNPOD_TEAM_ID", "custom-team-42")
    graphql("query { myself { id } }")
    assert requests[-1].get_header("X-team-id") == "custom-team-42"


def test_gpu_and_cpu_provision_mutations_route_through_graphql(monkeypatch):
    """Both provision mutations compose their query and dispatch it through
    the header-injecting :func:`graphql` wrapper — the GPU path renders
    ``podFindAndDeployOnDemand``, the CPU path ``deployCpuPod`` keyed on
    ``instanceId`` (no gpuTypeId/gpuCount)."""
    queries: list[str] = []

    def recording_graphql(query, variables=None, timeout=60):
        queries.append(query)
        return {}  # null mutation result -> the deploy helpers return None

    monkeypatch.setattr(runpod_api, "graphql", recording_graphql)
    monkeypatch.setattr(runpod_api, "_public_key_env", lambda: None)

    out = _deploy_once(
        name="pod-2059-test",
        gpu_type_id="NVIDIA H100 80GB HBM3",
        gpu_count=1,
        image=runpod_api.DEFAULT_IMAGE,
        volume_gb=200,
        container_disk_gb=50,
        cloud_type="ALL",
        data_center_id=None,
        interruptible=False,
    )
    assert out is None  # null result == no capacity; the query still rendered
    assert len(queries) == 1
    assert "podFindAndDeployOnDemand" in queries[0]
    assert "NVIDIA H100 80GB HBM3" in queries[0]
    assert "gpuCount: 1" in queries[0]

    out = _deploy_cpu_once(
        name="pod-2059-test-cpu",
        instance_id="cpu3c-8-16",
        image=runpod_api.DEFAULT_IMAGE,
        volume_gb=40,
        container_disk_gb=50,
        cloud_type="ALL",
        data_center_id=None,
    )
    assert out is None
    assert len(queries) == 2
    assert "deployCpuPod" in queries[1]
    assert "cpu3c-8-16" in queries[1]
    assert "gpuTypeId" not in queries[1]
    assert "gpuCount" not in queries[1]


def _enclosing_function_names_of_urlopen_calls(tree: ast.Module) -> set[str]:
    """Names of every function whose body contains a ``urlopen`` call."""
    hits: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._stack: list[str] = []

        def _visit_func(self, node) -> None:
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        visit_FunctionDef = _visit_func
        visit_AsyncFunctionDef = _visit_func

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name == "urlopen":
                hits.add(self._stack[-1] if self._stack else "<module>")
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hits


def test_no_http_call_site_bypasses_graphql_once():
    """AST pins on the single header-injection point:

    * every ``urlopen`` call (and every ImportFrom alias of it) lives inside
      ``_graphql_once`` — the one place the X-Team-Id header is set;
    * no other network-capable HTTP library is imported (requests / httpx /
      http.client / aiohttp / urllib3) — a new client would bypass the header;
    * repo-wide, the ``api.runpod.io`` endpoint literal appears in no
      scripts//src/ python file outside the allowlist (runpod_api.py = the
      call site; redact_for_gist.py = a docstring redaction-map mention).
    """
    tree = ast.parse(RUNPOD_API_PY.read_text(encoding="utf-8"))

    # (a) urlopen call sites: exactly inside _graphql_once.
    enclosing = _enclosing_function_names_of_urlopen_calls(tree)
    assert enclosing == {"_graphql_once"}, enclosing

    # (a') ImportFrom aliases of urlopen (a `from urllib.request import
    # urlopen as fetch` would dodge the Attribute check): none exist — the
    # module imports `request as urlrequest` wholesale only.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name != "urlopen", ast.dump(node)

    # (b) HTTP-library import whitelist: urllib is the ONLY network-capable
    # import (requests/httpx/aiohttp/urllib3/http.client would be a second,
    # header-less client).
    banned_roots = {"requests", "httpx", "aiohttp", "urllib3", "http"}
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for mod in mods:
            assert mod.split(".")[0] not in banned_roots, f"banned HTTP import: {mod}"

    # (c) Repo-wide endpoint-literal scan: no scripts//src/ python file
    # outside the allowlist names api.runpod.io (a hand-rolled curl/urlopen
    # to the endpoint would skip the team header).
    offenders: list[str] = []
    for base in ("scripts", "src"):
        for py in sorted((REPO_ROOT / base).rglob("*.py")):
            rel = py.relative_to(REPO_ROOT)
            if rel in _API_URL_ALLOWLIST:
                continue
            if "api.runpod.io" in py.read_text(encoding="utf-8", errors="replace"):
                offenders.append(str(rel))
    assert not offenders, offenders
    # The docstring-only allowlist entry must stay call-free: no HTTP client
    # import there (the literal is prose, not a request target).
    redact_tree = ast.parse((REPO_ROOT / "scripts" / "redact_for_gist.py").read_text())
    for node in ast.walk(redact_tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for mod in mods:
                assert mod.split(".")[0] not in banned_roots | {"urllib"}, (
                    f"redact_for_gist.py grew an HTTP import: {mod}"
                )
