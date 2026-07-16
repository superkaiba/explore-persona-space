"""Tests for ``scripts/dispatch_issue.py`` (the operational `/issue` CLI).

The slice-6 router + ``backends.issue_dispatch`` helper are fully
unit-tested elsewhere; this file pins the THIN operational CLI that
SKILL.md Step 6b / 6d / 8 actually shells:

1. ``launch`` action: empty frontmatter → auto chain (mock free
   backend wins; RunPod NEVER launched; sidecar written).
2. ``launch`` action with ``--backend runpod`` → RunPod launched +
   sidecar written.
2b. ``launch`` with ``--backend runpod`` while the task's frontmatter
    has NO ``backend:`` value (GCP-first bypass, incident lineage #571)
    → LOUD warning + ``extra.override_without_frontmatter=true`` on the
    ``epm:backend-selected`` marker; frontmatter ``backend: runpod`` →
    neither; unreadable frontmatter → check skipped, launch proceeds.
2c. ``launch`` with ``--backend runpod`` while the frontmatter names a
    DIFFERENT recognized lane (``gcp``/``nibi``/``fir``/``mila``, or
    the legacy ``cluster`` alias for nibi) → LOUD conflict warning +
    ``extra.override_conflicts_frontmatter=true`` (+
    ``frontmatter_backend``); an UNRECOGNIZED frontmatter value
    (typo'd ``gpc``, non-string ``true``) → LOUD hygiene warning +
    ``extra.frontmatter_backend_unrecognized=true`` (+ the value).
    Both additive — the launch always proceeds.
3. ``launch`` action with ``--backend cluster`` (legacy) → mapped to
   nibi.
4. ``launch`` action on a router terminal → ``failure_class:`` JSON
   line + nonzero exit code.
4b. ``launch`` with a ``--gpus`` override that mismatches the GCP
    machine for the intent on a gcp-reachable lane (explicit gcp, or
    auto with gcp in the lane order) → pre-route refusal (exit 2,
    ``reason: gpus_machine_mismatch``) BEFORE any backend is built
    (incident #599); matching counts, override-honoring lanes, and an
    auto order without gcp all proceed.
5. ``finalize`` action: sidecar present → confirm_artifacts PASS →
   teardown called.
6. ``finalize`` action: confirm_artifacts FAIL → teardown SKIPPED +
   nonzero exit code.
6b. ``finalize`` degrade path (incident #585): a declaration-less
    handle + agent-level upload-verification PASS evidence → teardown
    proceeds; no evidence → exit 3; a declaration-present mechanical
    FAIL never degrades.
7. ``finalize`` action: missing sidecar → infra failure JSON + nonzero
   exit code (CLI never crashes the orchestrator).
8. backend_poll.py: missing sidecar → terminal infra JSON (not
   FileNotFoundError) — the BLOCKER 3 regression test.

Nothing here requires RunPod / SLURM / GCP / SSH to be live; every
external call is mocked via the ``backends_factory`` seam on the CLI.
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stdout
from typing import Any

import pytest

from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY
from explore_persona_space.backends.base import (
    BackendKind,
    ComputeBackend,
    PollResult,
    RunHandle,
    RunSpec,
)
from explore_persona_space.backends.issue_dispatch import (
    default_handle_sidecar_path,
    read_handle_sidecar,
    write_handle_sidecar,
)

# ---------------------------------------------------------------------------
# Mock backend + dependency factory
# ---------------------------------------------------------------------------


class _MockBackend(ComputeBackend):
    """Records every launch / poll / teardown call for assertions."""

    def __init__(
        self,
        kind: BackendKind = "nibi",
        *,
        launch_should_raise: Exception | None = None,
        confirm_passes: bool = True,
    ) -> None:
        self._kind = kind
        self.launches: list[RunSpec] = []
        self.teardowns: list[RunHandle] = []
        self.confirms: list[RunHandle] = []
        self.fetches: list[RunHandle] = []
        # Ordered trace of the finalize-relevant calls — the #588
        # fetch-before-confirm test asserts on this sequence.
        self.call_sequence: list[str] = []
        self._launch_should_raise = launch_should_raise
        self._confirm_passes = confirm_passes

    @property
    def name(self) -> BackendKind:
        return self._kind

    def prepare(self, spec: RunSpec) -> None:
        return None

    def launch(self, spec: RunSpec) -> RunHandle:
        if self._launch_should_raise is not None:
            raise self._launch_should_raise
        self.launches.append(spec)
        return RunHandle(
            backend=self._kind,
            cluster=self._kind if self._kind in {"nibi", "fir"} else None,
            job_id="job-MOCK",
            pod_name=f"pod-{spec.issue}",
            scratch_dir="/scratch",
            log_path="/log",
            extra={"issue": spec.issue, "intent": spec.intent},
        )

    def estimate_start(self, spec: RunSpec):
        from datetime import UTC, datetime

        return datetime.now(tz=UTC)

    def estimate_start_seconds(self, spec: RunSpec) -> float | None:
        return 0.0

    def poll(self, handle: RunHandle) -> PollResult:
        return PollResult(
            status="running",
            current_phase="x",
            new_milestone=False,
            last_log_mtime_sec_ago=1,
            pid_alive=True,
            log_tail_excerpt="",
        )

    def fetch_logs(self, handle: RunHandle) -> str:
        return ""

    def fetch_results(self, handle: RunHandle) -> None:
        self.fetches.append(handle)
        self.call_sequence.append("fetch_results")
        return None

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        self.confirms.append(handle)
        self.call_sequence.append("confirm_artifacts")
        return self._confirm_passes

    def teardown(self, handle: RunHandle) -> None:
        self.teardowns.append(handle)
        self.call_sequence.append("teardown")


def _build_mock_factory(
    *,
    runpod: _MockBackend | None = None,
    nibi: _MockBackend | None = None,
    fir: _MockBackend | None = None,
    gcp: _MockBackend | None = None,
    mila_alive: bool = False,
    marker_posts: list[dict[str, Any]] | None = None,
) -> Any:
    """Return a backends_factory closure suitable for ``main(backends_factory=...)``.

    ``marker_posts`` (optional) collects every ``marker_poster(**kw)``
    call for assertions — the override-without-frontmatter tests read the
    ``epm:backend-selected`` body back out of it. ``None`` keeps the
    legacy no-op poster.
    """

    def _poster(**kw: Any) -> None:
        if marker_posts is not None:
            marker_posts.append(kw)

    def _factory() -> dict[str, Any]:
        free = {}
        if nibi is not None:
            free["nibi"] = nibi
        if fir is not None:
            free["fir"] = fir
        if mila_alive and "mila" not in free:
            # mila is rare in tests; absent default is fine.
            pass
        return {
            "runpod_backend": runpod or _MockBackend(kind="runpod"),
            "free_backends": free,
            "gcp_backend": gcp,
            "marker_poster": _poster,
            "is_started": lambda _b, _h: True,
            "is_live_after_cancel": lambda _b, _h: False,
            "reconnect_fn": lambda _b, _k, _s: None,
            "mila_socket_alive": lambda: mila_alive,
        }

    return _factory


# ---------------------------------------------------------------------------
# launch action
# ---------------------------------------------------------------------------


def _cd_to_tmp(monkeypatch, tmp_path):
    """Change cwd into ``tmp_path`` AND pin the sidecar root there so the
    default sidecar path ``.claude/cache/issue-<N>-handle.json`` lands
    under the tmp dir (test isolation; never write under the real
    checkout's cache). The production resolver is cwd-INDEPENDENT —
    anchored to the main checkout via git-common-dir (#612) — so the
    chdir alone no longer isolates; pin the resolver explicitly."""
    monkeypatch.chdir(tmp_path)
    import explore_persona_space.backends.issue_dispatch as idp

    monkeypatch.setattr(idp, "_main_checkout_root", lambda: tmp_path)


def test_launch_empty_frontmatter_auto_routes_to_free_and_never_runpod(
    monkeypatch, tmp_path
) -> None:
    """No ``--backend`` ⇒ auto. With nibi wired, the free lane wins;
    RunPod's ``launch`` must NEVER be called.
    """
    _cd_to_tmp(monkeypatch, tmp_path)
    # RunPod backend whose launch raises — if the auto path ever reaches
    # it, the exception bubbles + the assertion below fires.
    runpod = _MockBackend(
        kind="runpod",
        launch_should_raise=AssertionError("RunPod.launch must not be called on auto"),
    )
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(runpod=runpod, nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "300", "--intent", "lora-7b", "--hydra", "smoke=1"],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is True
    assert body["chosen_kind"] == "nibi"
    assert body["requested_kind"] is None  # auto
    # Sidecar landed at the default per-issue path.
    sidecar = default_handle_sidecar_path(300)
    assert sidecar.exists()
    # Round-trip: the persisted handle is the one the bg-Bash poller
    # will read tick-after-tick.
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "nibi"
    assert recovered.pod_name == "pod-300"
    # Nibi got the launch; RunPod did not.
    assert len(nibi.launches) == 1
    assert len(runpod.launches) == 0


def test_launch_backend_runpod_explicit_provisions_runpod_and_writes_sidecar(
    monkeypatch, tmp_path
) -> None:
    """``--backend runpod`` is the only path that spends real money;
    the launch path must reach RunPod AND write the sidecar uniformly
    (so Step 6d.2's bg-Bash poller has a handle to read, same as the
    SLURM/GCP paths)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    # Pin the frontmatter seam (hermetic: the override check must never
    # read the real main-checkout registry from a unit test). "runpod"
    # = the legitimate frontmatter-backed override.
    import scripts.dispatch_issue as cli

    monkeypatch.setattr(cli, "_frontmatter_backend_value", lambda _issue: "runpod")
    runpod = _MockBackend(kind="runpod")
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(runpod=runpod, nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "301",
                "--intent",
                "lora-7b",
                "--backend",
                "runpod",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["chosen_kind"] == "runpod"
    assert body["requested_kind"] == "runpod"
    # Sidecar was written for RunPod too — Step 8 finalize will read it.
    sidecar = default_handle_sidecar_path(301)
    assert sidecar.exists()
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "runpod"
    # RunPod got the launch; nibi did not.
    assert len(runpod.launches) == 1
    assert len(nibi.launches) == 0


def _run_runpod_launch(
    monkeypatch,
    tmp_path,
    *,
    issue: str,
    frontmatter_value: str | None,
    marker_posts: list[dict[str, Any]],
) -> int:
    """Shared driver for the override-without-frontmatter tests (2b).

    Pins the frontmatter seam to ``frontmatter_value`` and runs a
    ``--backend runpod`` launch against mock backends, collecting marker
    posts. Returns the CLI exit code.
    """
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as cli

    monkeypatch.setattr(cli, "_frontmatter_backend_value", lambda _issue: frontmatter_value)
    runpod = _MockBackend(kind="runpod")
    factory = _build_mock_factory(runpod=runpod, marker_posts=marker_posts)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(
            [
                "launch",
                "--issue",
                issue,
                "--intent",
                "lora-7b",
                "--backend",
                "runpod",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    # The check is additive — the RunPod launch itself always proceeds.
    assert len(runpod.launches) == 1
    return rc


def _backend_selected_extras(marker_posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ``extra`` dicts of every posted ``epm:backend-selected`` body."""
    extras = []
    for post in marker_posts:
        if post.get("marker") != "epm:backend-selected":
            continue
        body = json.loads(post["note"])
        extras.append(body["extra"])
    return extras


def test_launch_runpod_override_without_frontmatter_warns_and_flags_marker(
    monkeypatch, tmp_path, caplog
) -> None:
    """2b (incident lineage #571): ``--backend runpod`` while the task's
    frontmatter has NO ``backend:`` value silently bypasses the GCP-first
    standing default. The CLI must (a) WARN loudly on stderr naming the
    residual gaps, (b) stamp ``extra.override_without_frontmatter=true``
    on the ``epm:backend-selected`` marker, and (c) NOT block the launch
    or change the argument contract."""
    posts: list[dict[str, Any]] = []
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"):
        rc = _run_runpod_launch(
            monkeypatch, tmp_path, issue="310", frontmatter_value="", marker_posts=posts
        )
    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("override_without_frontmatter" in m and "GCP FIRST" in m for m in warnings), (
        f"expected the loud GCP-first bypass warning; got {warnings!r}"
    )
    extras = _backend_selected_extras(posts)
    assert extras, "expected at least one epm:backend-selected post"
    assert all(e.get("override_without_frontmatter") is True for e in extras)


def test_launch_runpod_override_with_explicit_auto_frontmatter_warns_and_flags_marker(
    monkeypatch, tmp_path, caplog
) -> None:
    """2b widening: explicit frontmatter ``backend: auto`` + CLI
    ``--backend runpod`` is the same GCP-first bypass in spirit as the
    absent/empty case — the frontmatter states the auto-routing intent
    even more explicitly — so it gets the same loud warning + marker
    flag, and the launch still proceeds."""
    posts: list[dict[str, Any]] = []
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"):
        rc = _run_runpod_launch(
            monkeypatch, tmp_path, issue="313", frontmatter_value="auto", marker_posts=posts
        )
    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("override_without_frontmatter" in m and "GCP FIRST" in m for m in warnings), (
        f"expected the loud GCP-first bypass warning; got {warnings!r}"
    )
    extras = _backend_selected_extras(posts)
    assert extras, "expected at least one epm:backend-selected post"
    assert all(e.get("override_without_frontmatter") is True for e in extras)


def test_launch_runpod_override_with_frontmatter_backing_no_warning_no_flag(
    monkeypatch, tmp_path, caplog
) -> None:
    """2b control: frontmatter ``backend: runpod`` backs the CLI value —
    no bypass warning, no marker flag (the legitimate override path is
    untouched)."""
    posts: list[dict[str, Any]] = []
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"):
        rc = _run_runpod_launch(
            monkeypatch, tmp_path, issue="311", frontmatter_value="runpod", marker_posts=posts
        )
    assert rc == 0
    guard_phrases = (
        "override_without_frontmatter",
        "override_conflicts_frontmatter",
        "frontmatter_backend_unrecognized",
    )
    assert not [r for r in caplog.records if any(p in r.getMessage() for p in guard_phrases)], (
        "the backed override must stay silent — no guard warning of any class"
    )
    extras = _backend_selected_extras(posts)
    assert extras, "expected at least one epm:backend-selected post"
    guard_flags = (*guard_phrases, "frontmatter_backend")
    assert all(flag not in e for e in extras for flag in guard_flags)


def test_launch_runpod_override_unreadable_frontmatter_skips_check(
    monkeypatch, tmp_path, caplog
) -> None:
    """2b degrade: frontmatter unreadable (seam returns ``None``) — the
    check is SKIPPED (no flag; we never stamp a bypass on a guess), a
    could-not-read warning is logged, and the launch proceeds."""
    posts: list[dict[str, Any]] = []
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"):
        rc = _run_runpod_launch(
            monkeypatch, tmp_path, issue="312", frontmatter_value=None, marker_posts=posts
        )
    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("could not be read" in m for m in warnings)
    extras = _backend_selected_extras(posts)
    assert extras, "expected at least one epm:backend-selected post"
    guard_flags = (
        "override_without_frontmatter",
        "override_conflicts_frontmatter",
        "frontmatter_backend_unrecognized",
        "frontmatter_backend",
    )
    assert all(flag not in e for e in extras for flag in guard_flags)


def test_launch_runpod_override_conflicting_frontmatter_warns_and_flags_marker(
    monkeypatch, tmp_path, caplog
) -> None:
    """2c conflict (A): frontmatter ``backend: gcp`` + CLI ``--backend
    runpod`` — the task explicitly names a DIFFERENT lane, contradicting
    the override even more strongly than absence. LOUD conflict warning
    + ``extra.override_conflicts_frontmatter=true`` +
    ``frontmatter_backend: "gcp"``; the absent-frontmatter flag is NOT
    reused; the launch proceeds."""
    posts: list[dict[str, Any]] = []
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"):
        rc = _run_runpod_launch(
            monkeypatch, tmp_path, issue="314", frontmatter_value="gcp", marker_posts=posts
        )
    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("override_conflicts_frontmatter" in m and "CONFLICTS" in m for m in warnings), (
        f"expected the loud conflict warning; got {warnings!r}"
    )
    extras = _backend_selected_extras(posts)
    assert extras, "expected at least one epm:backend-selected post"
    assert all(e.get("override_conflicts_frontmatter") is True for e in extras)
    assert all(e.get("frontmatter_backend") == "gcp" for e in extras)
    # Distinct-key discipline: the conflict case never reuses the
    # absent-frontmatter flag or the unrecognized flag.
    assert all("override_without_frontmatter" not in e for e in extras)
    assert all("frontmatter_backend_unrecognized" not in e for e in extras)


def test_launch_runpod_override_legacy_cluster_frontmatter_is_conflict(
    monkeypatch, tmp_path, caplog
) -> None:
    """2c conflict (legacy): frontmatter ``backend: cluster`` is the
    legacy selector-surface alias for nibi — recognized-and-conflicting.
    The warning names the nibi normalization; the marker carries the
    RAW frontmatter value (``cluster``), not the normalized lane."""
    posts: list[dict[str, Any]] = []
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"):
        rc = _run_runpod_launch(
            monkeypatch, tmp_path, issue="315", frontmatter_value="cluster", marker_posts=posts
        )
    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("override_conflicts_frontmatter" in m and "nibi" in m for m in warnings), (
        f"expected the conflict warning naming the nibi normalization; got {warnings!r}"
    )
    extras = _backend_selected_extras(posts)
    assert extras, "expected at least one epm:backend-selected post"
    assert all(e.get("override_conflicts_frontmatter") is True for e in extras)
    assert all(e.get("frontmatter_backend") == "cluster" for e in extras)
    assert all("frontmatter_backend_unrecognized" not in e for e in extras)


def test_launch_runpod_override_unrecognized_frontmatter_warns_and_flags_marker(
    monkeypatch, tmp_path, caplog
) -> None:
    """2c unrecognized (B): a typo'd frontmatter ``backend: gpc`` is NOT
    frontmatter backing — it is task hygiene noise. LOUD unrecognized
    warning + ``extra.frontmatter_backend_unrecognized=true`` +
    ``frontmatter_backend: "gpc"``; never classified as a conflict; the
    launch proceeds."""
    posts: list[dict[str, Any]] = []
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"):
        rc = _run_runpod_launch(
            monkeypatch, tmp_path, issue="316", frontmatter_value="gpc", marker_posts=posts
        )
    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "frontmatter_backend_unrecognized" in m and "not a recognized backend value" in m
        for m in warnings
    ), f"expected the loud unrecognized-frontmatter warning; got {warnings!r}"
    extras = _backend_selected_extras(posts)
    assert extras, "expected at least one epm:backend-selected post"
    assert all(e.get("frontmatter_backend_unrecognized") is True for e in extras)
    assert all(e.get("frontmatter_backend") == "gpc" for e in extras)
    assert all("override_conflicts_frontmatter" not in e for e in extras)
    assert all("override_without_frontmatter" not in e for e in extras)


def test_launch_runpod_override_nonstring_frontmatter_is_unrecognized(
    monkeypatch, tmp_path, caplog
) -> None:
    """2c unrecognized (B, non-string): a YAML boolean ``backend: true``
    reaches the guard as the normalized string ``"true"``
    (``_frontmatter_backend_value`` does ``str(raw).strip().lower()``) —
    classified unrecognized, never as a conflict or as backing."""
    posts: list[dict[str, Any]] = []
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"):
        rc = _run_runpod_launch(
            monkeypatch, tmp_path, issue="317", frontmatter_value="true", marker_posts=posts
        )
    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("not a recognized backend value" in m for m in warnings)
    extras = _backend_selected_extras(posts)
    assert extras, "expected at least one epm:backend-selected post"
    assert all(e.get("frontmatter_backend_unrecognized") is True for e in extras)
    assert all(e.get("frontmatter_backend") == "true" for e in extras)
    assert all("override_conflicts_frontmatter" not in e for e in extras)


def test_launch_sidecar_write_error_still_prints_handle_json(monkeypatch, tmp_path) -> None:
    """C1: a sidecar-write ``OSError`` after a SUCCESSFUL launch must not
    become rc=4 (the pre-fix path stranded a live job with no handle on
    stdout). The CLI prints the handle JSON line — the only recovery
    record — plus ``sidecar_write_error``, and exits 0."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import explore_persona_space.backends.issue_dispatch as idp

    def exploding_write(_handle, _path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(idp, "write_handle_sidecar", exploding_write)

    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "302",
                "--intent",
                "lora-7b",
                "--backend",
                "nibi",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0, "sidecar-write failure must not convert a successful launch to a crash rc"
    body = json.loads(buf.getvalue().strip().splitlines()[-1])
    # The handle JSON IS the recovery record — every field present.
    assert body["ok"] is True
    assert body["chosen_kind"] == "nibi"
    assert body["pod_name"] == "pod-302"
    assert body["job_id"] == "job-MOCK"
    assert body["handle_sidecar_path"] is None
    assert "No space left on device" in body["sidecar_write_error"]
    # The launch really happened.
    assert len(nibi.launches) == 1


def test_launch_sidecar_write_error_body_round_trips_deserialize_handle(
    monkeypatch, tmp_path
) -> None:
    """M4.1: the JSON printed on ``sidecar_write_error`` must carry the
    FULL serialized handle — ``deserialize_handle`` requires
    backend/scratch_dir/log_path beyond the summary fields, so an
    operator must be able to hand-write a ``--handle-file`` sidecar
    straight from the printed body and run finalize."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import explore_persona_space.backends.issue_dispatch as idp
    from explore_persona_space.backends.issue_dispatch import deserialize_handle

    def exploding_write(_handle, _path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(idp, "write_handle_sidecar", exploding_write)

    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "305",
                "--intent",
                "lora-7b",
                "--backend",
                "nibi",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip().splitlines()[-1])
    # The full handle dict round-trips through deserialize_handle (no
    # KeyError on a required field) and reconstructs the launch handle.
    recovered = deserialize_handle(body["handle"])
    assert recovered.backend == "nibi"
    assert recovered.job_id == "job-MOCK"
    assert recovered.pod_name == "pod-305"
    assert recovered.scratch_dir == "/scratch"
    assert recovered.log_path == "/log"
    # And a hand-written sidecar from that dict satisfies finalize: the
    # recovered handle is the same shape ``read_handle_sidecar`` yields.
    sidecar = tmp_path / "issue-305-recovered.json"
    sidecar.write_text(json.dumps(body["handle"]))
    assert read_handle_sidecar(sidecar) == recovered


def test_launch_backend_cluster_legacy_maps_to_nibi(monkeypatch, tmp_path) -> None:
    """``backend: cluster`` is the legacy selector alias; the dispatch
    helper maps it to ``nibi`` BEFORE building the spec (the slice-5
    router rejects the bare ``"cluster"`` literal)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    runpod = _MockBackend(kind="runpod")
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(runpod=runpod, nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "302",
                "--intent",
                "lora-7b",
                "--backend",
                "cluster",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["chosen_kind"] == "nibi"
    assert body["requested_kind"] == "nibi"  # router sees the normalized value
    assert len(nibi.launches) == 1


def test_launch_router_terminal_prints_failure_class_and_nonzero_exits(
    monkeypatch, tmp_path
) -> None:
    """A router terminal (``NoComputeAvailableError``) must print a
    ``failure_class``-tagged JSON line + exit nonzero so the
    orchestrator can post ``epm:failure v1`` and ``set-status blocked``."""
    _cd_to_tmp(monkeypatch, tmp_path)
    # No free backends + no GCP wired → auto chain immediately raises
    # NoComputeAvailableError (router stage 3 has nowhere to escalate).
    runpod = _MockBackend(
        kind="runpod",
        launch_should_raise=AssertionError("RunPod must not be called on auto"),
    )
    factory = _build_mock_factory(runpod=runpod, nibi=None, gcp=None)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "303", "--intent", "lora-7b", "--hydra", "smoke=1"],
            backends_factory=factory,
        )
    # Exit code 2 = router terminal (per the CLI docstring).
    assert rc == 2
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["failure_class"] == "infra"
    assert body["status"] == "blocked"
    assert body["exception"] == "NoComputeAvailableError"
    # The note's first line carries the failure_class= prefix so the
    # orchestrator's Step 7 classifier short-circuits.
    assert body["note"].splitlines()[0] == "failure_class: infra"
    assert "no_compute_available" in body["note"]
    # Sidecar NOT written on terminal exception (the router raises
    # BEFORE the sidecar write).
    assert not default_handle_sidecar_path(303).exists()


def test_launch_runpod_provision_exit_75_surfaces_still_waiting(monkeypatch, tmp_path) -> None:
    """``pod_lifecycle.py provision`` exit 75 (EX_TEMPFAIL, the bounded
    wait-for-capacity budget) is a STILL-WAITING outcome, not a failure:
    the CLI must print ``still_waiting: true`` + ``rerun: true`` and exit
    75 so the orchestrator re-runs the same command — never the rc-4
    ``CalledProcessError`` crash (incident #603, 2026-06-11)."""
    import subprocess

    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as cli

    monkeypatch.setattr(cli, "_frontmatter_backend_value", lambda _issue: "runpod")
    provision_cmd = [
        "/usr/bin/python3",
        "/repo/scripts/pod_lifecycle.py",
        "provision",
        "--issue",
        "603",
        "--intent",
        "eval",
    ]
    runpod = _MockBackend(
        kind="runpod",
        launch_should_raise=subprocess.CalledProcessError(75, provision_cmd),
    )
    factory = _build_mock_factory(runpod=runpod)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(
            [
                "launch",
                "--issue",
                "603",
                "--intent",
                "eval",
                "--backend",
                "runpod",
                "--workload-cmd",
                "echo smoke",
            ],
            backends_factory=factory,
        )
    assert rc == cli.EXIT_STILL_WAITING == 75
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["still_waiting"] is True
    assert body["rerun"] is True
    assert body["reason"] == "wait_for_capacity_budget_reached"
    # Deliberately NO failure_class / status keys — the orchestrator
    # must not post epm:failure / set-status blocked on this exit.
    assert "failure_class" not in body
    assert "status" not in body
    # No sidecar — the launch never completed (re-run resumes the wait).
    assert not default_handle_sidecar_path(603).exists()


def test_launch_gcp_create_timeout_still_provisioning_exits_75(monkeypatch, tmp_path) -> None:
    """The GCP-lane SECOND producer of exit 75 (#736): a
    ``gcloud compute instances create`` that exceeded the 300s subprocess
    cap on a FLEX_START rung while the instance came up live server-side
    raises ``GcpCreateTimedOutStillProvisioning`` out of ``launch``. The CLI
    must convert it to the still-waiting JSON (``still_waiting: true`` +
    ``rerun: true`` + the additive ``instance_name`` / ``instance_status``
    keys) and exit 75 — never the rc-4 ``main()`` catch-all traceback (the
    exact #736 bug). Mirrors the RunPod exit-75 test above; deliberately NO
    ``failure_class`` / ``status`` keys so the orchestrator does NOT post
    ``epm:failure`` / ``set-status blocked``."""
    from explore_persona_space.backends.gcp import GcpCreateTimedOutStillProvisioning

    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as cli

    monkeypatch.setattr(cli, "_frontmatter_backend_value", lambda _issue: "gcp")
    gcp = _MockBackend(
        kind="gcp",
        launch_should_raise=GcpCreateTimedOutStillProvisioning(
            instance_name="eps-issue-736",
            status="PROVISIONING",
            issue=736,
        ),
    )
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(
            [
                "launch",
                "--issue",
                "736",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--workload-cmd",
                "echo smoke",
            ],
            backends_factory=factory,
        )
    assert rc == cli.EXIT_STILL_WAITING == 75
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["still_waiting"] is True
    assert body["rerun"] is True
    assert body["reason"] == "gcloud_create_timeout_still_provisioning"
    # Additive keys carry the live instance the re-run reconnects to.
    assert body["instance_name"] == "eps-issue-736"
    assert body["instance_status"] == "PROVISIONING"
    # Deliberately NO failure_class / status keys — the orchestrator must
    # not post epm:failure / set-status blocked on this exit.
    assert "failure_class" not in body
    assert "status" not in body
    # No sidecar — the launch never completed (re-run resumes the wait).
    assert not default_handle_sidecar_path(736).exists()


def test_launch_unrelated_calledprocesserror_keeps_generic_rc4(monkeypatch, tmp_path) -> None:
    """An rc-75 subprocess that is NOT ``pod_lifecycle.py provision``
    (e.g. an ssh/gcloud helper from another lane) must NOT be mistaken
    for still-waiting — it falls through to the generic rc-4 handler."""
    import subprocess

    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as cli

    monkeypatch.setattr(cli, "_frontmatter_backend_value", lambda _issue: "runpod")
    runpod = _MockBackend(
        kind="runpod",
        launch_should_raise=subprocess.CalledProcessError(75, ["ssh", "pod-604", "true"]),
    )
    factory = _build_mock_factory(runpod=runpod)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(
            [
                "launch",
                "--issue",
                "604",
                "--intent",
                "eval",
                "--backend",
                "runpod",
                "--workload-cmd",
                "echo smoke",
            ],
            backends_factory=factory,
        )
    assert rc == 4
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["exception"] == "CalledProcessError"
    assert "still_waiting" not in body


def test_exit_still_waiting_matches_pod_lifecycle() -> None:
    """The CLI mirrors ``pod_lifecycle.EXIT_STILL_WAITING`` rather than
    importing it (import-light contract) — pin the two equal so a future
    renumbering on either side fails loudly here."""
    from scripts.dispatch_issue import EXIT_STILL_WAITING as cli_code
    from scripts.pod_lifecycle import EXIT_STILL_WAITING as pl_code

    assert cli_code == pl_code == 75


def test_launch_hydra_args_threaded_into_spec(monkeypatch, tmp_path) -> None:
    """``--hydra k=v`` (repeatable) must land on the spec verbatim so
    the SLURM render / RunPod launch script picks them up."""
    _cd_to_tmp(monkeypatch, tmp_path)
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "304",
                "--intent",
                "lora-7b",
                "--hydra",
                "condition=c1",
                "--hydra",
                "seed=42",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert nibi.launches[0].hydra_args == ("condition=c1", "seed=42")


# ---------------------------------------------------------------------------
# incident #599 — pre-route --gpus / GCP machine-type mismatch guard
# ---------------------------------------------------------------------------


def _guard_exploding_factory():
    raise AssertionError("backends must not be built when the gpus guard refuses the launch")


def test_launch_gpus_mismatch_explicit_gcp_fails_loud_before_backends(
    monkeypatch, tmp_path
) -> None:
    """Incident #599 (updated for #1121): the original ``--gpus 4 --intent
    lora-7b`` shape is now HONORED width-aware by the GCP ladder, so the
    refusal path is exercised with an UNSUPPORTED width (``--gpus 3`` — not
    a WIDE_A100_80_BY_WIDTH key). The CLI must refuse PRE-LAUNCH with the
    router-terminal JSON shape (exit 2, failure_class infra), name the
    supported widths, and never build a backend."""
    _cd_to_tmp(monkeypatch, tmp_path)
    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "599",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--gpus",
                "3",
                "--workload-cmd",
                "bash scripts/run_issue599_fullresp.sh",
            ],
            backends_factory=_guard_exploding_factory,
        )
    assert rc == 2
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["failure_class"] == "infra"
    assert body["status"] == "blocked"
    assert body["reason"] == "gpus_machine_mismatch"
    # The note's first line carries the failure_class= prefix so the
    # orchestrator's Step 7 classifier short-circuits (same contract as
    # the router-terminal translation).
    assert body["note"].splitlines()[0] == "failure_class: infra"
    # #1121: the message names the supported widths for the eligible intent.
    assert "supported --gpus values" in body["note"]
    assert "[2, 4, 8]" in body["note"]
    # Nothing launched → no sidecar.
    assert not default_handle_sidecar_path(599).exists()


def test_launch_gpus_mismatch_auto_lane_gcp_first_fails_loud(monkeypatch, tmp_path) -> None:
    """The #599 incident lane shape: NO ``--backend`` (auto) under the
    GCP-first standing default — gcp is reachable as the FIRST lane, so the
    mismatch guard must refuse pre-route just like the explicit gcp case
    (#1121: with an unsupported width — the original ``--gpus 4`` is now a
    honored wide width on lora-7b)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    monkeypatch.delenv("EPM_AUTO_LANE_ORDER", raising=False)
    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "599",
                "--intent",
                "lora-7b",
                "--gpus",
                "3",
                "--workload-cmd",
                "bash scripts/run_issue599_fullresp.sh",
            ],
            backends_factory=_guard_exploding_factory,
        )
    assert rc == 2
    body = json.loads(buf.getvalue().strip())
    assert body["reason"] == "gpus_machine_mismatch"
    assert not default_handle_sidecar_path(599).exists()


def test_gpus_supported_wide_width_accepted_for_eligible_intent() -> None:
    """#1121 AC: a supported WIDER width on a width-eligible intent is
    honored width-aware by the GCP ladder — the pre-route guard returns
    ``None`` (no refusal) for every supported width above base."""
    from explore_persona_space.backends.base import RunSpec
    from scripts.dispatch_issue import _gpus_gcp_lane_conflict

    for intent, gpus in (("capture-7b", 8), ("lora-7b", 4), ("lora-7b", 2), ("ft-7b", 8)):
        spec = RunSpec(issue=1121, intent=intent, backend="gcp", gpus=gpus)
        assert _gpus_gcp_lane_conflict(spec) is None, (intent, gpus)


def test_launch_gpus_wide_width_on_gcp_lane_proceeds(monkeypatch, tmp_path) -> None:
    """#1121 (main()-level accept): ``--gpus 8 --intent capture-7b`` on the
    explicit gcp lane proceeds — the launch reaches the backend with
    ``spec.gpus == 8`` intact (the router's width-aware ladder consumes it)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "1121",
                "--intent",
                "capture-7b",
                "--backend",
                "gcp",
                "--gpus",
                "8",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].gpus == 8


def test_gpus_unsupported_width_still_refused() -> None:
    """#1121: unsupported widths (3, 16) still refuse with
    ``reason: gpus_machine_mismatch`` and a message naming the supported
    widths."""
    from explore_persona_space.backends.base import RunSpec
    from scripts.dispatch_issue import _gpus_gcp_lane_conflict

    for gpus in (3, 16):
        body = _gpus_gcp_lane_conflict(
            RunSpec(issue=1121, intent="capture-7b", backend="gcp", gpus=gpus)
        )
        assert body is not None, gpus
        assert body["reason"] == "gpus_machine_mismatch"
        assert "supported --gpus values" in body["note"]
        assert "[2, 4, 8]" in body["note"]


def test_gpus_below_base_still_refused() -> None:
    """#1121: a width BELOW the intent's base machine (``--gpus 2`` on
    ft-7b, base 4x) still refuses — width degradation is the ladder's job
    on capacity miss, never a user-requested under-provision."""
    from explore_persona_space.backends.base import RunSpec
    from scripts.dispatch_issue import _gpus_gcp_lane_conflict

    body = _gpus_gcp_lane_conflict(RunSpec(issue=1121, intent="ft-7b", backend="gcp", gpus=2))
    assert body is not None
    assert body["reason"] == "gpus_machine_mismatch"


def test_launch_gpus_match_on_gcp_lane_proceeds(monkeypatch, tmp_path) -> None:
    """A MATCHING override (``ft-7b`` → a2-ultragpu-4g carries 4 GPUs)
    never trips the guard — the launch proceeds with spec.gpus intact."""
    _cd_to_tmp(monkeypatch, tmp_path)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "601",
                "--intent",
                "ft-7b",
                "--backend",
                "gcp",
                "--gpus",
                "4",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].gpus == 4


def test_launch_ft_intent_gcp_without_boot_disk_warns_and_flags_marker(
    monkeypatch, tmp_path, caplog
) -> None:
    """Incident #606: a gcp-reachable ``ft-*`` launch without
    ``--boot-disk-gb`` provisions the 300 GB pd-ssd default, which a
    ZeRO-3 full-FT fills in ~1h (kernel panic → SSH lockout → idle
    A100s). The CLI must (a) WARN loudly on stderr pointing at the
    plan's Reproducibility pod-row disk size, (b) stamp
    ``extra.boot_disk_default_with_ft_intent=true`` on the
    ``epm:backend-selected`` marker, and (c) NOT block the launch."""
    _cd_to_tmp(monkeypatch, tmp_path)
    gcp = _MockBackend(kind="gcp")
    posts: list[dict[str, Any]] = []
    factory = _build_mock_factory(gcp=gcp, marker_posts=posts)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"), redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "606",
                "--intent",
                "ft-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert len(gcp.launches) == 1, "the warning is additive — the launch must proceed"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "--boot-disk-gb" in m and "boot_disk_default_with_ft_intent" in m for m in warnings
    ), f"expected the loud default-boot-disk warning; got {warnings!r}"
    extras = _backend_selected_extras(posts)
    assert extras, "expected at least one epm:backend-selected post"
    assert all(e.get("boot_disk_default_with_ft_intent") is True for e in extras)


def test_launch_ft_intent_gcp_with_boot_disk_no_warning_no_flag(
    monkeypatch, tmp_path, caplog
) -> None:
    """#606 control: an explicitly sized ``--boot-disk-gb`` launch is the
    correct composition — no warning, no marker flag, and the size is
    threaded to ``spec.extra['boot_disk_gb']`` for the GCP renderer."""
    _cd_to_tmp(monkeypatch, tmp_path)
    gcp = _MockBackend(kind="gcp")
    posts: list[dict[str, Any]] = []
    factory = _build_mock_factory(gcp=gcp, marker_posts=posts)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"), redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "607",
                "--intent",
                "ft-7b",
                "--backend",
                "gcp",
                "--boot-disk-gb",
                "500",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra["boot_disk_gb"] == 500
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("boot_disk_default_with_ft_intent" in m for m in warnings), (
        f"explicitly sized launch must stay silent; got {warnings!r}"
    )
    extras = _backend_selected_extras(posts)
    assert all("boot_disk_default_with_ft_intent" not in e for e in extras)


def test_ft_intent_boot_disk_guard_stands_down_off_gcp_lanes() -> None:
    """Unit coverage for ``_ft_intent_gcp_default_boot_disk``'s stand-down
    cases: explicit non-GCP lanes, non-ft intents, ft intents with no GCP
    machine mapping (``ft-70b`` fails loud inside the lane before disk
    matters), and an already-sized boot disk."""
    from types import SimpleNamespace

    from scripts.dispatch_issue import _ft_intent_gcp_default_boot_disk

    def spec(*, intent="ft-7b", backend="gcp", extra=None):
        return SimpleNamespace(intent=intent, backend=backend, extra=extra or {})

    assert _ft_intent_gcp_default_boot_disk(spec()) is True
    assert _ft_intent_gcp_default_boot_disk(spec(extra={"boot_disk_gb": 500})) is False
    assert _ft_intent_gcp_default_boot_disk(spec(intent="lora-7b")) is False
    assert _ft_intent_gcp_default_boot_disk(spec(intent="eval")) is False
    assert _ft_intent_gcp_default_boot_disk(spec(intent="ft-70b")) is False
    assert _ft_intent_gcp_default_boot_disk(spec(backend="runpod")) is False
    assert _ft_intent_gcp_default_boot_disk(spec(backend="nibi")) is False


def test_launch_max_run_duration_threads_to_spec_extra(monkeypatch, tmp_path) -> None:
    """#628: the plan's GCP auto-delete fence threads via
    ``--max-run-duration`` into ``spec.extra['max_run_duration']`` (the
    instance-create renderer's override hook over the 24h
    ``GcpConfig.default_max_run_duration``). Before the flag existed a
    declared 30h fence had no CLI path from the /issue Step 6b launch
    and the orchestrator had to accept the default as a plan deviation."""
    _cd_to_tmp(monkeypatch, tmp_path)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "628",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--max-run-duration",
                "30h",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra["max_run_duration"] == "30h"


def test_launch_max_run_duration_absent_leaves_extra_unset(monkeypatch, tmp_path) -> None:
    """Control: with no ``--max-run-duration`` the key is ABSENT from
    ``spec.extra`` (the GCP renderer's ``or config.default_max_run_duration``
    fallback owns the default; the CLI never duplicates it)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "629",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert "max_run_duration" not in gcp.launches[0].extra


def test_launch_provisioning_model_threads_to_spec_extra(monkeypatch, tmp_path) -> None:
    """#537: ``--provisioning-model SPOT`` threads into
    ``spec.extra['provisioning_model']`` so the GCP renderer draws the
    PREEMPTIBLE accelerator quota pool instead of on-demand."""
    _cd_to_tmp(monkeypatch, tmp_path)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "537",
                "--intent",
                "ft-7b",
                "--backend",
                "gcp",
                "--provisioning-model",
                "SPOT",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra["provisioning_model"] == "SPOT"


def test_launch_provisioning_model_absent_lets_ladder_choose(monkeypatch, tmp_path) -> None:
    """Control: with no ``--provisioning-model`` the CLI does NOT pin the key,
    so the router's length-aware ladder (#680) owns the choice. For this
    unknown-length lora-7b the long branch leads with the flex-start A100-80
    rung, so the LAUNCHED spec carries ``provisioning_model == 'FLEX_START'``
    (the rung the ladder picked) — NOT a CLI duplicate. The CLI-pin honor path
    is exercised by ``test_launch_provisioning_model_threads_to_spec_extra``."""
    _cd_to_tmp(monkeypatch, tmp_path)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "538",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    # No CLI pin => the ladder's first rung (flex-start A100-80 for an
    # unknown-length lora-7b) owns the provisioning model on the launched spec.
    assert gcp.launches[0].extra["provisioning_model"] == "FLEX_START"


def test_launch_spot_tolerant_threads_to_spec_extra(monkeypatch, tmp_path) -> None:
    """#537: ``--spot-tolerant`` threads ``spec.extra['spot_tolerant']=True``
    so the router's STANDARD->SPOT auto-fallback may fire for this
    workload; absent, the key never appears (a non-recoverable run is
    never silently preempted)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "537",
                "--intent",
                "ft-7b",
                "--backend",
                "gcp",
                "--spot-tolerant",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra["spot_tolerant"] is True

    # Control: absent flag leaves the key off entirely.
    gcp2 = _MockBackend(kind="gcp")
    factory2 = _build_mock_factory(gcp=gcp2)
    with redirect_stdout(io.StringIO()):
        rc2 = main(
            [
                "launch",
                "--issue",
                "537",
                "--intent",
                "ft-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory2,
        )
    assert rc2 == 0
    assert "spot_tolerant" not in gcp2.launches[0].extra


def test_width_required_threads_extra_flag(monkeypatch, tmp_path) -> None:
    """#1379 (T10): ``--width-required`` threads
    ``spec.extra['width_required']=True`` (the router's
    ``_explicit_wide_degrade_widths`` opt-out); absent, the key never
    appears (the degrading ladder stays the default)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    from scripts.dispatch_issue import main

    with redirect_stdout(io.StringIO()):
        rc = main(
            [
                "launch",
                "--issue",
                "1379",
                "--intent",
                "sweep-8g-a100",
                "--backend",
                "gcp",
                "--width-required",
                "--workload-cmd",
                "bash scripts/run_issue1379_sweep.sh",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra["width_required"] is True

    # Control: absent flag leaves the key off entirely.
    gcp2 = _MockBackend(kind="gcp")
    factory2 = _build_mock_factory(gcp=gcp2)
    with redirect_stdout(io.StringIO()):
        rc2 = main(
            [
                "launch",
                "--issue",
                "1379",
                "--intent",
                "sweep-8g-a100",
                "--backend",
                "gcp",
                "--workload-cmd",
                "bash scripts/run_issue1379_sweep.sh",
            ],
            backends_factory=factory2,
        )
    assert rc2 == 0
    assert "width_required" not in gcp2.launches[0].extra


def test_width_required_with_gpus_exits_2_with_conflict_reason(monkeypatch, tmp_path) -> None:
    """#1379 (T11): ``--width-required --gpus 8`` is a contradictory
    combination (--gpus declares a RE-SHARDABLE axis by contract, #1121;
    --width-required pins the width) — refused pre-route with the same
    exit-2 JSON shape as the #599 ``gpus_machine_mismatch`` guard, before
    any backend is built."""
    _cd_to_tmp(monkeypatch, tmp_path)
    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "1379",
                "--intent",
                "sweep-8g-a100",
                "--backend",
                "gcp",
                "--width-required",
                "--gpus",
                "8",
                "--workload-cmd",
                "bash scripts/run_issue1379_sweep.sh",
            ],
            backends_factory=_guard_exploding_factory,
        )
    assert rc == 2
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["failure_class"] == "infra"
    assert body["status"] == "blocked"
    assert body["reason"] == "width_required_gpus_conflict"
    # The note's first line carries the failure_class= prefix so the
    # orchestrator's Step 7 classifier short-circuits (same contract as
    # the gpus_machine_mismatch refusal).
    assert body["note"].splitlines()[0] == "failure_class: infra"
    # Nothing launched -> no sidecar.
    assert not default_handle_sidecar_path(1379).exists()


def test_launch_provisioning_model_rejects_unknown_value(monkeypatch, tmp_path) -> None:
    """argparse ``choices`` rejects an out-of-set provisioning model
    (SystemExit 2) before any backend is built."""
    _cd_to_tmp(monkeypatch, tmp_path)
    factory = _build_mock_factory(gcp=_MockBackend(kind="gcp"))

    from scripts.dispatch_issue import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "launch",
                "--issue",
                "537",
                "--intent",
                "ft-7b",
                "--backend",
                "gcp",
                "--provisioning-model",
                "ONDEMAND",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert exc.value.code == 2


def test_max_run_duration_arg_validation() -> None:
    """Unit coverage for the gcloud-duration argparse validator: composed
    integer+unit groups pass (whitespace stripped); bare integers
    (gcloud would silently read seconds), negatives, fractions, and
    embedded spaces are refused at the parser surface."""
    import argparse

    from scripts.dispatch_issue import _max_run_duration_arg

    assert _max_run_duration_arg("30h") == "30h"
    assert _max_run_duration_arg("1d12h") == "1d12h"
    assert _max_run_duration_arg("90m") == "90m"
    assert _max_run_duration_arg("86400s") == "86400s"
    assert _max_run_duration_arg(" 30h ") == "30h"
    for bad in ("", "30", "h30", "-5h", "1.5h", "30 h", "24hrs"):
        with pytest.raises(argparse.ArgumentTypeError):
            _max_run_duration_arg(bad)


def test_launch_gpus_override_skips_guard_on_lanes_that_honor_it(monkeypatch, tmp_path) -> None:
    """RunPod maps ``spec.gpus`` to ``pod_lifecycle.py --gpu-count`` and
    SLURM maps it to the ``--gres`` render — explicit non-GCP lanes
    honor the override (and never escalate to GCP), so the guard stands
    down."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as cli

    # Pin the frontmatter seam (hermetic; "runpod" = legitimately backed).
    monkeypatch.setattr(cli, "_frontmatter_backend_value", lambda _issue: "runpod")
    runpod = _MockBackend(kind="runpod")
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(runpod=runpod, nibi=nibi)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(
            [
                "launch",
                "--issue",
                "602",
                "--intent",
                "lora-7b",
                "--backend",
                "runpod",
                "--gpus",
                "4",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert runpod.launches[0].gpus == 4

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(
            [
                "launch",
                "--issue",
                "603",
                "--intent",
                "lora-7b",
                "--backend",
                "nibi",
                "--gpus",
                "4",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert nibi.launches[0].gpus == 4


def test_launch_gpus_mismatch_auto_lane_without_gcp_skips_guard(monkeypatch, tmp_path) -> None:
    """``EPM_AUTO_LANE_ORDER`` excluding gcp makes GCP unreachable on the
    auto chain — the guard stands down and the SLURM lane (which honors
    the override) routes normally."""
    _cd_to_tmp(monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_AUTO_LANE_ORDER", "nibi")
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "604",
                "--intent",
                "lora-7b",
                "--gpus",
                "4",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert nibi.launches[0].gpus == 4


# ---------------------------------------------------------------------------
# repo-branch default (fix19 production mirror — round-2, task #535)
# ---------------------------------------------------------------------------


def test_launch_repo_branch_defaults_to_current_branch_for_gcp_lane(monkeypatch, tmp_path) -> None:
    """Without ``--repo-branch``, a gcp/auto dispatch from a feature-branch
    checkout must thread the CURRENT branch into spec.extra — the GCE
    startup script clones from origin and would otherwise silently run
    stale main (the exact fix19 bug re-created on the production path)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_current_git_branch", lambda: "issue-535-feature")
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "304",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra.get("repo_branch") == "issue-535-feature"


def test_launch_repo_branch_explicit_flag_wins_over_current_branch(monkeypatch, tmp_path) -> None:
    """An explicit ``--repo-branch`` always wins; the current-branch
    default never overrides operator intent."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_current_git_branch", lambda: "issue-535-feature")
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "304",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
                "--repo-branch",
                "release-x",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra.get("repo_branch") == "release-x"


def test_launch_repo_branch_not_defaulted_on_explicit_slurm_lane(monkeypatch, tmp_path) -> None:
    """An explicit SLURM lane never escalates to GCP, so the gcp-only
    repo_branch knob is not threaded (SLURM rsyncs the local worktree)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_current_git_branch", lambda: "issue-535-feature")
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "304",
                "--intent",
                "lora-7b",
                "--backend",
                "nibi",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert "repo_branch" not in nibi.launches[0].extra


def test_launch_repo_branch_not_defaulted_when_on_main_and_no_worktree(
    monkeypatch, tmp_path
) -> None:
    """A main-branch checkout WITHOUT a per-issue worktree keeps the GCE
    clone default ("main") — no spurious extra key, no log noise. (Rescoped
    for #824: the worktree-branch fallback needs the no-worktree case pinned
    explicitly now that a present worktree DOES default.)"""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: None)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "304",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert "repo_branch" not in gcp.launches[0].extra


def test_launch_repo_branch_defaults_to_worktree_branch_when_on_main(monkeypatch, tmp_path) -> None:
    """#824 core: invoking checkout on ``main`` + a per-issue worktree on an
    issue branch → ``extra['repo_branch']`` defaults to the WORKTREE branch
    on the gcp/auto lane, and the pushed-branch guard is consulted with
    (branch, worktree_root)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    worktree = str(tmp_path / ".claude" / "worktrees" / "issue-824")
    guard_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(di, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: worktree)
    monkeypatch.setattr(di, "_git_branch_of", lambda root: "issue-824-wf-fix")
    monkeypatch.setattr(
        di, "_warn_if_branch_not_pushed", lambda branch, root: guard_calls.append((branch, root))
    )
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "824",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra.get("repo_branch") == "issue-824-wf-fix"
    assert guard_calls == [("issue-824-wf-fix", worktree)]


def test_launch_repo_branch_defaults_to_worktree_branch_when_current_unresolvable(
    monkeypatch, tmp_path
) -> None:
    """#824: an unresolvable invoking-checkout branch (detached HEAD /
    git error → None) behaves like ``main`` — the worktree-branch fallback
    still fires on the gcp/auto lane."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    worktree = str(tmp_path / ".claude" / "worktrees" / "issue-824")
    guard_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(di, "_current_git_branch", lambda: None)
    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: worktree)
    monkeypatch.setattr(di, "_git_branch_of", lambda root: "issue-824-wf-fix")
    monkeypatch.setattr(
        di, "_warn_if_branch_not_pushed", lambda branch, root: guard_calls.append((branch, root))
    )
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "824",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra.get("repo_branch") == "issue-824-wf-fix"
    assert guard_calls == [("issue-824-wf-fix", worktree)]


def test_launch_repo_branch_not_defaulted_when_worktree_on_main(monkeypatch, tmp_path) -> None:
    """#824 negative: a worktree checked out on ``main`` contributes no
    default — no spurious ``repo_branch`` key, guard never consulted."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    worktree = str(tmp_path / ".claude" / "worktrees" / "issue-824")
    guard_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(di, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: worktree)
    monkeypatch.setattr(di, "_git_branch_of", lambda root: "main")
    monkeypatch.setattr(
        di, "_warn_if_branch_not_pushed", lambda branch, root: guard_calls.append((branch, root))
    )
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "824",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert "repo_branch" not in gcp.launches[0].extra
    assert guard_calls == []


def test_launch_repo_branch_not_defaulted_when_worktree_detached(monkeypatch, tmp_path) -> None:
    """#824 negative: a detached-HEAD worktree (``_git_branch_of`` → None)
    contributes no default and the guard is never consulted."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    worktree = str(tmp_path / ".claude" / "worktrees" / "issue-824")
    guard_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(di, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: worktree)
    monkeypatch.setattr(di, "_git_branch_of", lambda root: None)
    monkeypatch.setattr(
        di, "_warn_if_branch_not_pushed", lambda branch, root: guard_calls.append((branch, root))
    )
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "824",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert "repo_branch" not in gcp.launches[0].extra
    assert guard_calls == []


def test_launch_repo_branch_explicit_flag_wins_over_worktree_branch(monkeypatch, tmp_path) -> None:
    """#824 precedence: an explicit ``--repo-branch`` wins outright — the
    worktree fallback rung is never even consulted (raising stub proves it)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    worktree = str(tmp_path / ".claude" / "worktrees" / "issue-824")

    def _boom(root: str) -> str | None:
        raise AssertionError("worktree fallback must not run for an explicit --repo-branch")

    monkeypatch.setattr(di, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: worktree)
    monkeypatch.setattr(di, "_git_branch_of", _boom)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "824",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--repo-branch",
                "release-x",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra.get("repo_branch") == "release-x"


def test_launch_repo_branch_current_branch_wins_over_worktree_branch(monkeypatch, tmp_path) -> None:
    """#824 precedence: a non-main invoking-checkout branch (the existing
    #535 rung) wins over the worktree branch; the guard is not consulted."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    worktree = str(tmp_path / ".claude" / "worktrees" / "issue-824")
    guard_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(di, "_current_git_branch", lambda: "issue-999-feature")
    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: worktree)
    monkeypatch.setattr(di, "_git_branch_of", lambda root: "issue-824-wf-fix")
    monkeypatch.setattr(
        di, "_warn_if_branch_not_pushed", lambda branch, root: guard_calls.append((branch, root))
    )
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "824",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra.get("repo_branch") == "issue-999-feature"
    assert guard_calls == []


def test_launch_repo_branch_not_defaulted_from_worktree_on_explicit_slurm_lane(
    monkeypatch, tmp_path
) -> None:
    """#824 lane gate: the worktree fallback never fires on an explicit
    SLURM lane (``--backend nibi``) — SLURM refuses non-main repo_branch
    (#653 r8), so the default stays gcp/auto-only."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    worktree = str(tmp_path / ".claude" / "worktrees" / "issue-824")
    guard_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(di, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: worktree)
    monkeypatch.setattr(di, "_git_branch_of", lambda root: "issue-824-wf-fix")
    monkeypatch.setattr(
        di, "_warn_if_branch_not_pushed", lambda branch, root: guard_calls.append((branch, root))
    )
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "824",
                "--intent",
                "lora-7b",
                "--backend",
                "nibi",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert "repo_branch" not in nibi.launches[0].extra
    assert guard_calls == []


def test_launch_repo_branch_not_defaulted_from_worktree_on_explicit_runpod_lane(
    monkeypatch, tmp_path
) -> None:
    """#824 lane gate (runpod variant): the worktree fallback never fires on
    an explicit ``--backend runpod`` launch either."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    worktree = str(tmp_path / ".claude" / "worktrees" / "issue-824")
    guard_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(di, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: worktree)
    monkeypatch.setattr(di, "_git_branch_of", lambda root: "issue-824-wf-fix")
    monkeypatch.setattr(
        di, "_warn_if_branch_not_pushed", lambda branch, root: guard_calls.append((branch, root))
    )
    runpod = _MockBackend(kind="runpod")
    factory = _build_mock_factory(runpod=runpod)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "824",
                "--intent",
                "lora-7b",
                "--backend",
                "runpod",
                "--workload-cmd",
                "bash scripts/issue824_dispatch.sh",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert "repo_branch" not in runpod.launches[0].extra
    assert guard_calls == []


def test_warn_if_branch_not_pushed_never_raises_and_warns(monkeypatch, caplog) -> None:
    """#824 guard contract: ``_warn_if_branch_not_pushed`` NEVER raises —
    a subprocess timeout and a nonzero ls-remote both degrade to a WARN
    naming the branch."""
    import subprocess

    import scripts.dispatch_issue as di

    def _raise(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="git", timeout=30)

    monkeypatch.setattr(di.subprocess, "run", _raise)
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"):
        di._warn_if_branch_not_pushed("issue-824-wf-fix", "/nonexistent")
    assert any("issue-824-wf-fix" in rec.getMessage() for rec in caplog.records)
    caplog.clear()

    def _not_found(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["git"], returncode=2, stdout="", stderr="")

    monkeypatch.setattr(di.subprocess, "run", _not_found)
    with caplog.at_level(logging.WARNING, logger="dispatch_issue"):
        di._warn_if_branch_not_pushed("issue-824-wf-fix", "/nonexistent")
    assert any("issue-824-wf-fix" in rec.getMessage() for rec in caplog.records)


def test_git_branch_of_real_git_semantics(tmp_path) -> None:
    """#824 helper contract on a REAL git repo: named branch → its name;
    detached HEAD → None; nonexistent directory → None."""
    import subprocess

    import scripts.dispatch_issue as di

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "commit.gpgsign", "false"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "--allow-empty", "-q", "-m", "init"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "issue-t824"], check=True)
    assert di._git_branch_of(str(repo)) == "issue-t824"

    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--detach"], check=True)
    assert di._git_branch_of(str(repo)) is None

    assert di._git_branch_of(str(tmp_path / "does-not-exist")) is None


# ---------------------------------------------------------------------------
# #705: git_repo_root (#685) + skip_default_git_paths (#661) threading
# ---------------------------------------------------------------------------


def test_launch_runpod_populates_git_repo_root_when_worktree_exists(monkeypatch, tmp_path) -> None:
    """#705 concern ``worktree-fix-inert-when-repo-branch-absent``: an
    explicit ``--backend runpod`` launch WITHOUT ``--repo-branch`` (the
    #685 shape) STILL threads the per-issue worktree git root into
    ``spec.extra['git_repo_root']`` — derived from issue + repo_root
    ALONE, NOT gated on repo_branch — so the artifact verifier's git
    check resolves against the worktree branch where the run committed
    its eval/figure artifacts."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    worktree = str(tmp_path / ".claude" / "worktrees" / "issue-685")
    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: worktree)
    runpod = _MockBackend(kind="runpod")
    factory = _build_mock_factory(runpod=runpod)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "685",
                "--intent",
                "lora-7b",
                "--backend",
                "runpod",
                "--workload-cmd",
                "bash scripts/issue685_dispatch.sh",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    # The RunPod launch carries no --repo-branch, yet git_repo_root is set.
    assert runpod.launches[0].extra.get("git_repo_root") == worktree
    assert "repo_branch" not in runpod.launches[0].extra


def test_launch_omits_git_repo_root_when_no_worktree(monkeypatch, tmp_path) -> None:
    """Control: with no per-issue worktree the key is ABSENT from
    ``spec.extra`` → the declaration falls back to the established
    pyproject-walk repo root (back-compat, #705 constraint 6)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: None)
    runpod = _MockBackend(kind="runpod")
    factory = _build_mock_factory(runpod=runpod)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "685",
                "--intent",
                "lora-7b",
                "--backend",
                "runpod",
                "--workload-cmd",
                "bash scripts/issue685_dispatch.sh",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert "git_repo_root" not in runpod.launches[0].extra


def test_launch_skip_default_git_paths_threads_to_spec_extra(monkeypatch, tmp_path) -> None:
    """#661: ``--skip-default-git-paths`` threads into
    ``spec.extra['skip_default_git_paths']=True`` (lane-agnostic), so the
    backend declaration builder omits the auto full-task git paths for a
    phase-scoped launch."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: None)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "661",
                "--intent",
                "eval",
                "--backend",
                "gcp",
                "--skip-default-git-paths",
                "--workload-cmd",
                "bash scripts/issue661_extract.sh",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra.get("skip_default_git_paths") is True


def test_launch_skip_default_git_paths_absent_leaves_extra_unset(monkeypatch, tmp_path) -> None:
    """Control: without ``--skip-default-git-paths`` the key is ABSENT →
    the established full-task git-path declaration is unchanged."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_issue_worktree_git_root", lambda issue: None)
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "661",
                "--intent",
                "eval",
                "--backend",
                "gcp",
                "--workload-cmd",
                "bash scripts/issue661_extract.sh",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert "skip_default_git_paths" not in gcp.launches[0].extra


def test_skip_confirm_artifacts_flag_still_present() -> None:
    """#705 constraint: the ``finalize --skip-confirm-artifacts`` escape
    hatch is retained (the fix replaces routine bypass, never removes the
    operator override)."""
    from scripts.dispatch_issue import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["finalize", "--issue", "705", "--skip-confirm-artifacts"])
    assert args.skip_confirm_artifacts is True


# ---------------------------------------------------------------------------
# finalize action
# ---------------------------------------------------------------------------


def _seed_sidecar(
    tmp_path, issue: int, kind: BackendKind = "nibi", *, with_declaration: bool = True
) -> RunHandle:
    """Write a sidecar for ``finalize`` tests; return the handle.

    ``with_declaration=False`` mirrors the production RunPod / SLURM
    launch paths, which do NOT populate the ``expected_artifacts``
    declaration (incident #585 / task #598) — the shape the finalize
    degrade path exists for.
    """
    extra: dict[str, Any] = {"issue": issue, "intent": "lora-7b"}
    if with_declaration:
        extra[EXPECTED_ARTIFACTS_HANDLE_KEY] = {
            "issue": issue,
            "sentinel_path": "/tmp/sentinel.json",
        }
    handle = RunHandle(
        backend=kind,
        cluster=kind if kind in {"nibi", "fir"} else None,
        job_id="job-fin",
        pod_name=f"pod-{issue}",
        scratch_dir="/scratch",
        log_path="/log",
        extra=extra,
    )
    sidecar = tmp_path / f"issue-{issue}-handle.json"
    write_handle_sidecar(handle, sidecar)
    return handle


def test_finalize_confirm_artifacts_pass_runs_teardown(monkeypatch, tmp_path) -> None:
    """The happy path: sidecar present + confirm PASS → teardown called.
    Exit 0; JSON line carries phase=teardown."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 400, kind="nibi")
    nibi = _MockBackend(kind="nibi", confirm_passes=True)
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_upload_verification_currency_blocker", lambda _issue: None)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "finalize",
                "--issue",
                "400",
                "--handle-file",
                str(tmp_path / "issue-400-handle.json"),
            ],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is True
    assert body["phase"] == "teardown"
    assert body["chosen_kind"] == "nibi"
    assert len(nibi.confirms) == 1
    assert len(nibi.teardowns) == 1


def test_finalize_confirm_artifacts_fail_skips_teardown_and_exits_nonzero(
    monkeypatch, tmp_path
) -> None:
    """A FAIL on confirm_artifacts MUST skip teardown (preserve evidence)
    + exit code 3 so the orchestrator escalates instead of silently
    losing the live backend handle."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 401, kind="nibi")
    nibi = _MockBackend(kind="nibi", confirm_passes=False)
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_upload_verification_currency_blocker", lambda _issue: None)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "finalize",
                "--issue",
                "401",
                "--handle-file",
                str(tmp_path / "issue-401-handle.json"),
            ],
            backends_factory=factory,
        )
    # Exit 3 = confirm_artifacts FAIL (per the CLI docstring).
    assert rc == 3
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["phase"] == "confirm_artifacts"
    assert body["reason"] == "confirm_artifacts_failed"
    # confirm was called; teardown was NOT.
    assert len(nibi.confirms) == 1
    assert len(nibi.teardowns) == 0


def test_finalize_no_declaration_with_agent_pass_degrades_to_teardown(
    monkeypatch, tmp_path
) -> None:
    """Incident #585: a handle WITHOUT an ``expected_artifacts``
    declaration (the production RunPod / SLURM launch shapes) makes the
    mechanical gate structurally unsatisfiable. With agent-level
    upload-verification PASS evidence on the task, finalize must degrade
    to teardown (exit 0, sidecar retired, degrade recorded in the JSON)
    instead of the pre-fix exit 3 that forced a raw ``pod.py terminate``
    bypass."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 407, kind="runpod", with_declaration=False)
    runpod = _MockBackend(kind="runpod", confirm_passes=False)
    factory = _build_mock_factory(runpod=runpod)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_agent_upload_verification_passed", lambda _issue: True)
    monkeypatch.setattr(di, "_upload_verification_currency_blocker", lambda _issue: None)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "finalize",
                "--issue",
                "407",
                "--handle-file",
                str(tmp_path / "issue-407-handle.json"),
            ],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is True
    assert body["phase"] == "teardown"
    assert body["confirm_artifacts"] == "skipped_no_declaration_agent_pass"
    # The mechanical gate was still exercised (and FAILed structurally)
    # before the degrade; teardown ran exactly once; sidecar retired.
    assert len(runpod.confirms) == 1
    assert len(runpod.teardowns) == 1
    assert (tmp_path / "issue-407-handle.json.finalized").exists()


def test_finalize_no_declaration_without_agent_pass_keeps_exit_3(monkeypatch, tmp_path) -> None:
    """No declaration AND no agent-level PASS evidence → the degrade must
    NOT fire: exit 3, teardown skipped, with the sharper
    ``confirm_artifacts_no_declaration`` reason (distinguishable from a
    real mechanical artifact FAIL)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 408, kind="runpod", with_declaration=False)
    runpod = _MockBackend(kind="runpod", confirm_passes=False)
    factory = _build_mock_factory(runpod=runpod)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_agent_upload_verification_passed", lambda _issue: False)
    monkeypatch.setattr(di, "_upload_verification_currency_blocker", lambda _issue: None)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "finalize",
                "--issue",
                "408",
                "--handle-file",
                str(tmp_path / "issue-408-handle.json"),
            ],
            backends_factory=factory,
        )
    assert rc == 3
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["phase"] == "confirm_artifacts"
    assert body["reason"] == "confirm_artifacts_no_declaration"
    assert len(runpod.teardowns) == 0


def test_finalize_declaration_present_fail_never_degrades(monkeypatch, tmp_path) -> None:
    """The safety property of the degrade: a handle WITH a declaration
    whose mechanical confirm FAILs keeps the exit-3 evidence-preserving
    behavior even when agent-level PASS evidence exists — the agent
    verdict never overrides a real mechanical artifact FAIL."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 409, kind="nibi", with_declaration=True)
    nibi = _MockBackend(kind="nibi", confirm_passes=False)
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_agent_upload_verification_passed", lambda _issue: True)
    monkeypatch.setattr(di, "_upload_verification_currency_blocker", lambda _issue: None)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "finalize",
                "--issue",
                "409",
                "--handle-file",
                str(tmp_path / "issue-409-handle.json"),
            ],
            backends_factory=factory,
        )
    assert rc == 3
    body = json.loads(buf.getvalue().strip())
    assert body["reason"] == "confirm_artifacts_failed"
    assert len(nibi.teardowns) == 0


def test_agent_upload_verification_probe_reads_events(monkeypatch, tmp_path) -> None:
    """The evidence probe: latest ``epm:upload-verification`` verdict
    wins; the sticky ``epm:upload-verified`` marker alone also counts;
    missing events.jsonl / FAIL verdicts are NO evidence."""
    import explore_persona_space.task_workflow as tw
    import scripts.dispatch_issue as di

    task_dir = tmp_path / "tasks" / "verifying" / "777"
    task_dir.mkdir(parents=True)
    monkeypatch.setattr(tw, "find_task_path", lambda _id: task_dir)

    # No events.jsonl at all → no evidence.
    assert di._agent_upload_verification_passed(777) is False

    events = task_dir / "events.jsonl"
    # A FAIL verdict → no evidence.
    events.write_text(
        json.dumps({"kind": "epm:upload-verification", "note": "**Verdict: FAIL**"}) + "\n"
    )
    assert di._agent_upload_verification_passed(777) is False

    # A later re-verification PASS (the FAIL → fix → re-verify loop):
    # latest marker wins.
    with events.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "epm:upload-verification",
                    "note": "## Upload Verification\n\n**Verdict: PASS**\n\n11 files.",
                }
            )
            + "\n"
        )
    assert di._agent_upload_verification_passed(777) is True

    # The sticky PASS marker alone also counts.
    events.write_text(json.dumps({"kind": "epm:upload-verified", "note": "sticky"}) + "\n")
    assert di._agent_upload_verification_passed(777) is True


def test_agent_upload_verification_probe_missing_task_is_false() -> None:
    """A task that does not exist anywhere (registry or disk) is NO
    evidence — the probe swallows the lookup failure into the safe
    direction (caller keeps the exit-3 teardown-skip) instead of
    crashing finalize."""
    import scripts.dispatch_issue as di

    assert di._agent_upload_verification_passed(99999999) is False


def test_finalize_skip_confirm_artifacts_forces_teardown(monkeypatch, tmp_path) -> None:
    """``--skip-confirm-artifacts`` matches ``pod.py terminate
    --skip-upload-verify`` — escape hatch for crashes that left no
    artifacts to verify."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 402, kind="nibi")
    nibi = _MockBackend(kind="nibi", confirm_passes=False)
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_upload_verification_currency_blocker", lambda _issue: None)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "finalize",
                "--issue",
                "402",
                "--handle-file",
                str(tmp_path / "issue-402-handle.json"),
                "--skip-confirm-artifacts",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert len(nibi.confirms) == 0  # skipped
    assert len(nibi.teardowns) == 1


def test_finalize_renames_sidecar_after_successful_teardown(monkeypatch, tmp_path) -> None:
    """Mn4.3: a successful teardown retires the sidecar by renaming it
    to ``<name>.finalized`` (kept for audit) so a LATER finalize for
    the same issue cannot tear down a fresh run through the stale
    handle — the duplicate tick no-ops with the benign rc=2
    missing-sidecar shape, and the backend sees exactly ONE teardown."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 404, kind="nibi")
    sidecar = tmp_path / "issue-404-handle.json"
    original_payload = sidecar.read_text()
    nibi = _MockBackend(kind="nibi", confirm_passes=True)
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_upload_verification_currency_blocker", lambda _issue: None)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["finalize", "--issue", "404", "--handle-file", str(sidecar)],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["phase"] == "teardown"
    # The sidecar was renamed, not deleted: audit copy intact.
    finalized = tmp_path / "issue-404-handle.json.finalized"
    assert not sidecar.exists()
    assert finalized.exists()
    assert finalized.read_text() == original_payload
    assert body["sidecar_finalized"] == str(finalized)

    # Second finalize for the same issue: benign rc=2 no-op, NO second
    # teardown against the retired handle.
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = main(
            ["finalize", "--issue", "404", "--handle-file", str(sidecar)],
            backends_factory=factory,
        )
    assert rc2 == 2
    body2 = json.loads(buf2.getvalue().strip())
    assert body2["ok"] is False
    assert body2["reason"] == "missing_handle_sidecar"
    assert len(nibi.teardowns) == 1


def test_finalize_missing_sidecar_returns_infra_failure_not_crash(monkeypatch, tmp_path) -> None:
    """A missing sidecar must produce a clean JSON line + nonzero exit
    code (NEVER a FileNotFoundError / traceback that crashes the
    orchestrator's bg-Bash parser)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    factory = _build_mock_factory(nibi=_MockBackend(kind="nibi"))

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_upload_verification_currency_blocker", lambda _issue: None)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "finalize",
                "--issue",
                "403",
                "--handle-file",
                str(tmp_path / "issue-403-handle.json"),
            ],
            backends_factory=factory,
        )
    assert rc == 2
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["failure_class"] == "infra"
    assert body["reason"] == "missing_handle_sidecar"


# ---------------------------------------------------------------------------
# issue #1026 — finalize verifier-currency gate
# ---------------------------------------------------------------------------


def _currency_blocker_record(reason: str, state: str | None = None) -> dict:
    """A seam blocker record shaped like the real helper's return."""
    return {
        "reason": reason,
        "state": state,
        "stage": "verifying" if state else None,
        "round": 1 if state else None,
        "breadcrumb_ts": "2026-07-02T18:55:00Z" if state else None,
        "age_minutes": 5.0 if state else None,
        "detail": f"seam blocker: {reason}",
    }


def _run_finalize(tmp_path, issue: int, factory, *extra_args: str) -> tuple[int, dict]:
    """Run ``finalize --issue <N> --handle-file <sidecar>`` and parse the JSON line."""
    import scripts.dispatch_issue as di

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "finalize",
                "--issue",
                str(issue),
                "--handle-file",
                str(tmp_path / f"issue-{issue}-handle.json"),
                *extra_args,
            ],
            backends_factory=factory,
        )
    return rc, json.loads(buf.getvalue().strip())


def test_finalize_in_flight_verifier_blocks_no_declaration_degrade(monkeypatch, tmp_path) -> None:
    """C1: no-declaration + agent PASS evidence + an IN-FLIGHT verifier
    round → exit 3 ``upload_verifier_in_flight``; teardown NOT called,
    sidecar NOT retired (the #778 fallback loophole, closed)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 407, kind="runpod", with_declaration=False)
    runpod = _MockBackend(kind="runpod", confirm_passes=False)
    factory = _build_mock_factory(runpod=runpod)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_agent_upload_verification_passed", lambda _issue: True)
    monkeypatch.setattr(
        di,
        "_upload_verification_currency_blocker",
        lambda _issue: _currency_blocker_record("upload_verifier_in_flight", "in-flight"),
    )
    rc, body = _run_finalize(tmp_path, 407, factory)
    assert rc == 3
    assert body["ok"] is False
    assert body["reason"] == "upload_verifier_in_flight"
    assert body["verifier_state"] == "in-flight"
    assert len(runpod.teardowns) == 0
    assert (tmp_path / "issue-407-handle.json").exists()  # NOT retired


def test_finalize_stalled_verifier_refuses_without_skip_flag(monkeypatch, tmp_path) -> None:
    """C2 (MF-D): NO skip flag + a STALLED verifier round → exit 3
    ``upload_verifier_stalled``, 0 teardowns, sidecar NOT retired. This
    cell IS the #778 replay whenever finalize fires past the 15-min
    window."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 407, kind="runpod", with_declaration=False)
    runpod = _MockBackend(kind="runpod", confirm_passes=False)
    factory = _build_mock_factory(runpod=runpod)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_agent_upload_verification_passed", lambda _issue: True)
    monkeypatch.setattr(
        di,
        "_upload_verification_currency_blocker",
        lambda _issue: _currency_blocker_record("upload_verifier_stalled", "stalled"),
    )
    rc, body = _run_finalize(tmp_path, 407, factory)
    assert rc == 3
    assert body["reason"] == "upload_verifier_stalled"
    assert len(runpod.teardowns) == 0
    assert (tmp_path / "issue-407-handle.json").exists()


def test_finalize_stale_blocks_declaration_present_confirm_pass(monkeypatch, tmp_path) -> None:
    """C3 (MF-E pin): declaration PRESENT + mechanical confirm would PASS +
    a STALE verdict → exit 3 ``upload_verification_stale`` BEFORE
    ``fetch_results`` / ``confirm_artifacts`` ever run — the gate is
    uniform across all non-skip paths."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 400, kind="nibi", with_declaration=True)
    nibi = _MockBackend(kind="nibi", confirm_passes=True)
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(
        di,
        "_upload_verification_currency_blocker",
        lambda _issue: _currency_blocker_record("upload_verification_stale"),
    )
    rc, body = _run_finalize(tmp_path, 400, factory)
    assert rc == 3
    assert body["reason"] == "upload_verification_stale"
    assert len(nibi.fetches) == 0  # gate precedes fetch_results
    assert len(nibi.confirms) == 0
    assert len(nibi.teardowns) == 0
    assert (tmp_path / "issue-400-handle.json").exists()


def test_finalize_in_flight_blocks_declaration_present_path(monkeypatch, tmp_path) -> None:
    """C4: declaration present + seam in-flight → exit 3 before
    fetch/confirm; teardown skipped; sidecar not retired."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 400, kind="nibi", with_declaration=True)
    nibi = _MockBackend(kind="nibi", confirm_passes=True)
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(
        di,
        "_upload_verification_currency_blocker",
        lambda _issue: _currency_blocker_record("upload_verifier_in_flight", "in-flight"),
    )
    rc, body = _run_finalize(tmp_path, 400, factory)
    assert rc == 3
    assert body["reason"] == "upload_verifier_in_flight"
    assert len(nibi.confirms) == 0
    assert len(nibi.teardowns) == 0
    assert (tmp_path / "issue-400-handle.json").exists()


def test_finalize_skip_flag_still_refuses_fresh_in_flight(monkeypatch, tmp_path) -> None:
    """C5: ``--skip-confirm-artifacts`` NEVER destroys a RUNNING verifier
    round's pod — a fresh in-flight round refuses even with the flag."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 402, kind="nibi")
    nibi = _MockBackend(kind="nibi", confirm_passes=False)
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(
        di,
        "_upload_verification_currency_blocker",
        lambda _issue: _currency_blocker_record("upload_verifier_in_flight", "in-flight"),
    )
    rc, body = _run_finalize(tmp_path, 402, factory, "--skip-confirm-artifacts")
    assert rc == 3
    assert body["reason"] == "upload_verifier_in_flight"
    assert body["skip_confirm_artifacts"] is True
    assert len(nibi.teardowns) == 0
    assert (tmp_path / "issue-402-handle.json").exists()


@pytest.mark.parametrize(
    "reason",
    [
        "upload_verifier_stalled",  # C6
        "upload_verification_stale",  # C7
        "upload_verification_failed_current",  # C7b — crashed-run recovery pin
    ],
)
def test_finalize_skip_flag_degrades_non_in_flight_reasons_to_warning(
    monkeypatch, tmp_path, reason
) -> None:
    """C6/C7/C7b: with ``--skip-confirm-artifacts``, stalled / stale /
    failed-current records degrade to a loud warning — teardown runs,
    the success JSON carries ``verifier_warning``, sidecar retired
    (keeps #604, crashed-run recovery, and 6d.4 working, zero new flags)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 402, kind="nibi")
    nibi = _MockBackend(kind="nibi", confirm_passes=False)
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    state = "stalled" if reason == "upload_verifier_stalled" else None
    monkeypatch.setattr(
        di,
        "_upload_verification_currency_blocker",
        lambda _issue: _currency_blocker_record(reason, state),
    )
    rc, body = _run_finalize(tmp_path, 402, factory, "--skip-confirm-artifacts")
    assert rc == 0
    assert body["ok"] is True
    assert body["verifier_warning"] == reason
    assert len(nibi.teardowns) == 1
    assert not (tmp_path / "issue-402-handle.json").exists()
    assert (tmp_path / "issue-402-handle.json.finalized").exists()


def test_failed_verifier_round_refuses_never_silent(monkeypatch, tmp_path) -> None:
    """C8 (MF-A, REAL read path): the #778 shape — a sticky prior PASS, new
    results, a verifying crumb, then a FAIL verdict — refuses through the
    REAL wrapper → real ``list_events`` → real helper chain (NO seam
    patch): exit 3 ``upload_verification_failed_current``, 0 teardowns.
    Pre-#1026 this shape TORE DOWN via the sticky-anywhere fallback."""
    import explore_persona_space.task_workflow as tw

    _cd_to_tmp(monkeypatch, tmp_path)
    task_dir = tmp_path / "tasks" / "verifying" / "777"
    task_dir.mkdir(parents=True)
    monkeypatch.setattr(tw, "find_task_path", lambda _id: task_dir)
    events = [
        {"ts": "2026-07-02T03:40:00Z", "kind": "epm:upload-verified", "note": "sticky PASS"},
        {"ts": "2026-07-02T18:30:00Z", "kind": "epm:results", "note": "round-2 results"},
        {
            "ts": "2026-07-02T18:40:00Z",
            "kind": "epm:progress",
            "note": "stage-dispatch stage=followup-verifying round=1 subagent=upload-verifier",
        },
        {
            "ts": "2026-07-02T18:50:00Z",
            "kind": "epm:upload-verification",
            "note": "## Upload Verification\n\n**Verdict: FAIL**\n\n2 files missing.",
        },
    ]
    (task_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    _seed_sidecar(tmp_path, 777, kind="runpod", with_declaration=False)
    runpod = _MockBackend(kind="runpod", confirm_passes=False)
    factory = _build_mock_factory(runpod=runpod)

    rc, body = _run_finalize(tmp_path, 777, factory)
    assert rc == 3
    assert body["reason"] == "upload_verification_failed_current"
    assert len(runpod.teardowns) == 0
    assert (tmp_path / "issue-777-handle.json").exists()


def test_finalize_real_read_path_blocks_unresolved_crumb(monkeypatch, tmp_path) -> None:
    """C9 (MF-C, REAL read path): the #778 v2-dispatch replay — results →
    PASS verdict → a NEW verifying crumb with no verdict yet. The REAL
    wrapper returns an in-flight/stalled reason and finalize exits 3."""
    import explore_persona_space.task_workflow as tw

    _cd_to_tmp(monkeypatch, tmp_path)
    task_dir = tmp_path / "tasks" / "verifying" / "777"
    task_dir.mkdir(parents=True)
    monkeypatch.setattr(tw, "find_task_path", lambda _id: task_dir)
    events = [
        {"ts": "2026-07-02T03:20:00Z", "kind": "epm:results", "note": "results"},
        {
            "ts": "2026-07-02T03:36:37Z",
            "kind": "epm:upload-verification",
            "note": "**Verdict: PASS**",
        },
        {
            "ts": "2026-07-02T03:40:00Z",
            "kind": "epm:progress",
            "note": "stage-dispatch stage=followup-verifying round=1 subagent=upload-verifier",
        },
    ]
    (task_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    import scripts.dispatch_issue as di

    blocker = di._upload_verification_currency_blocker(777)
    assert blocker is not None
    assert blocker["reason"] in {"upload_verifier_in_flight", "upload_verifier_stalled"}

    _seed_sidecar(tmp_path, 777, kind="runpod", with_declaration=False)
    runpod = _MockBackend(kind="runpod", confirm_passes=False)
    factory = _build_mock_factory(runpod=runpod)
    rc, body = _run_finalize(tmp_path, 777, factory)
    assert rc == 3
    assert body["reason"] in {"upload_verifier_in_flight", "upload_verifier_stalled"}
    assert len(runpod.teardowns) == 0


def test_currency_blocker_wrapper_missing_task_is_none() -> None:
    """C10: the wrapper absorbs the ``find_task_path`` registry-miss
    exception (FileNotFoundError) into None — a missing task is NOT a
    refusal (the PASS-evidence probe keeps its own safe direction)."""
    import scripts.dispatch_issue as di

    assert di._upload_verification_currency_blocker(99999999) is None


def test_currency_blocker_wrapper_raises_on_helper_bug(monkeypatch) -> None:
    """C11 (MF-C fail-loud): a helper BUG (TypeError) RAISES through the
    wrapper — it can never silently disarm the gate by returning None."""
    import explore_persona_space.task_workflow as tw
    import scripts.dispatch_issue as di

    monkeypatch.setattr(tw, "list_events", lambda _id: [])

    def _boom(_events, **_kw):
        raise TypeError("helper bug")

    monkeypatch.setattr(tw, "upload_verification_currency_blocker", _boom)
    with pytest.raises(TypeError, match="helper bug"):
        di._upload_verification_currency_blocker(777)


# ---------------------------------------------------------------------------
# backend_poll.py missing-sidecar regression (BLOCKER 3)
# ---------------------------------------------------------------------------


def test_backend_poll_missing_sidecar_emits_terminal_infra_json(tmp_path) -> None:
    """The BLOCKER 3 regression test: ``scripts/backend_poll.py`` MUST
    emit a single ``status: "dead"`` JSON line with
    ``failure_class: "infra"`` when the sidecar is missing. Previously
    it raised FileNotFoundError → empty stdout → the orchestrator's
    bg-Bash JSON-line parser had nothing to parse → loop spun forever
    on "stalled"."""
    from scripts.backend_poll import main as backend_poll_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = backend_poll_main(
            ["--issue", "500", "--handle-file", str(tmp_path / "nonexistent.json")]
        )
    # Exit 0 (the script always emits valid JSON; the failure is
    # encoded IN the JSON via failure_class / status, NOT via exit code).
    assert rc == 0
    line = buf.getvalue().strip()
    assert line, "backend_poll must emit a JSON line, never empty stdout"
    body = json.loads(line)
    # Legacy poll_pipeline shape preserved so the orchestrator's
    # existing parser handles it without a per-backend branch.
    assert body["status"] == "dead"
    assert body["pid_alive"] is False
    # Failure-classifier hint keys (the orchestrator reads these
    # alongside status: dead to post epm:failure v1 with the matching
    # failure_class).
    assert body["failure_class"] == "infra"
    assert body["reason"] == "missing_handle_sidecar"


def test_backend_poll_unreadable_sidecar_also_emits_infra_json(tmp_path) -> None:
    """A corrupted JSON sidecar should hit the same failure shape — the
    orchestrator can't poll either way, so a malformed sidecar reads
    operationally as 'missing' from its perspective."""
    bad = tmp_path / "issue-501-handle.json"
    bad.write_text("{not valid json}")

    from scripts.backend_poll import main as backend_poll_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = backend_poll_main(["--issue", "501", "--handle-file", str(bad)])
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["status"] == "dead"
    assert body["failure_class"] == "infra"
    assert body["reason"] == "missing_handle_sidecar"


def test_backend_poll_missing_default_sidecar_names_both_probed_paths(
    monkeypatch, tmp_path
) -> None:
    """Default resolution probes the canonical main-checkout path AND the
    legacy cwd-relative location (pre-#612 back-compat); when neither
    exists, the terminal infra JSON names BOTH so the operator can see
    which side of the split was searched."""
    import explore_persona_space.backends.issue_dispatch as idp

    main_root = tmp_path / "mainroot"
    main_root.mkdir()
    worktree = tmp_path / "worktree"
    (worktree / ".claude" / "cache").mkdir(parents=True)
    monkeypatch.setattr(idp, "_main_checkout_root", lambda: main_root)
    monkeypatch.chdir(worktree)

    from scripts.backend_poll import main as backend_poll_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = backend_poll_main(["--issue", "502"])
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["status"] == "dead"
    assert body["failure_class"] == "infra"
    assert body["reason"] == "missing_handle_sidecar"
    excerpt = body["log_tail_excerpt"]
    assert str(main_root / ".claude" / "cache" / "issue-502-handle.json") in excerpt
    assert str(worktree / ".claude" / "cache" / "issue-502-handle.json") in excerpt


def test_backend_poll_reads_legacy_worktree_sidecar_when_canonical_absent(
    monkeypatch, tmp_path
) -> None:
    """Back-compat (#612 transition): a sidecar written by the pre-fix
    cwd-relative composer (launch dispatched from an issue worktree) is
    still FOUND by a poll tick when the canonical main-checkout path is
    empty — the run must NOT read as dead/missing_handle_sidecar."""
    import explore_persona_space.backends.issue_dispatch as idp
    import scripts.backend_poll as bp

    main_root = tmp_path / "mainroot"
    main_root.mkdir()
    worktree = tmp_path / "worktree"
    cache = worktree / ".claude" / "cache"
    cache.mkdir(parents=True)
    monkeypatch.setattr(idp, "_main_checkout_root", lambda: main_root)
    monkeypatch.chdir(worktree)

    handle = RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="j503",
        pod_name="pod-503",
        scratch_dir="/s",
        log_path="/l",
        extra={"issue": 503},
    )
    write_handle_sidecar(handle, cache / "issue-503-handle.json")

    polled: list[RunHandle] = []

    class _StubBackend:
        def poll(self, h):
            polled.append(h)
            return PollResult(
                status="running",
                current_phase="train",
                new_milestone=False,
                last_log_mtime_sec_ago=5,
                pid_alive=True,
                log_tail_excerpt="ok",
                gate=None,
                sentinels_processed=0,
                phase_log_mtime_sec_ago=5,
                shard_log_mtime_sec_ago=5,
                gpu_util="50%",
            )

    monkeypatch.setattr(bp, "_resolve_backend", lambda _name: _StubBackend())

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = bp.main(["--issue", "503"])
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["status"] == "running"
    assert polled and polled[0].pod_name == "pod-503"


# ---------------------------------------------------------------------------
# Backend-for-handle resolver
# ---------------------------------------------------------------------------


def test_resolve_backend_for_handle_routes_runpod_slurm_gcp() -> None:
    """The finalize path's resolver must dispatch the right backend per
    ``handle.backend`` — silent mis-routing would terminate the WRONG
    live backend on a multi-tenant orchestrator."""
    from scripts.dispatch_issue import _resolve_backend_for_handle

    runpod = _MockBackend(kind="runpod")
    nibi = _MockBackend(kind="nibi")
    gcp = _MockBackend(kind="gcp")
    deps = {
        "runpod_backend": runpod,
        "free_backends": {"nibi": nibi},
        "gcp_backend": gcp,
    }
    assert _resolve_backend_for_handle(_handle_for("runpod"), deps) is runpod, (
        "runpod handle → runpod backend"
    )
    assert _resolve_backend_for_handle(_handle_for("nibi"), deps) is nibi, (
        "nibi handle → nibi backend"
    )
    assert _resolve_backend_for_handle(_handle_for("gcp"), deps) is gcp, "gcp handle → gcp backend"
    # Legacy 'cluster' kind falls back to ANY available SLURM backend.
    assert _resolve_backend_for_handle(_handle_for("cluster"), deps) is nibi


def test_resolve_backend_for_handle_rejects_unknown_kind() -> None:
    """An unknown backend kind on the handle MUST raise rather than
    silently default to RunPod (which would terminate the wrong live
    backend)."""
    from scripts.dispatch_issue import _resolve_backend_for_handle

    deps = {
        "runpod_backend": _MockBackend(kind="runpod"),
        "free_backends": {},
        "gcp_backend": None,
    }
    with pytest.raises(ValueError, match=r"unknown handle\.backend"):
        _resolve_backend_for_handle(_handle_for("totally-bogus"), deps)


def _handle_for(kind: str) -> RunHandle:
    return RunHandle(
        backend=kind,  # type: ignore[arg-type]
        cluster=kind if kind in {"nibi", "fir", "cluster"} else None,
        job_id="j",
        pod_name="p",
        scratch_dir="/s",
        log_path="/l",
        extra={},
    )


# ---------------------------------------------------------------------------
# Production-backends factory smoke test (M3: regression guard for C1)
# ---------------------------------------------------------------------------


def test_build_production_backends_wires_all_keys_and_smokes_closures(monkeypatch) -> None:
    """Call the REAL :func:`scripts.dispatch_issue._build_production_backends`
    (not the ``_build_mock_factory`` the other tests inject) and smoke
    every closure on a benign :class:`RunSpec`.

    This is the regression guard for the C1 bug fixed in router slice 6
    fix2: ``_reconnect(kind="gcp")`` previously reached
    ``gcp_backend._runner`` but :class:`backends.gcp.GcpBackend` stores
    its runner as ``self._run``. The pre-fix code path AttributeError'd
    on every explicit ``backend: gcp`` lane AND every auto-chain GCP
    escalation that hit the reconnect path. The fix is to expose the
    injection seam through a public ``runner`` property on the backend
    AND have the dispatch ``_reconnect`` closure read that property
    rather than reaching into the underscored name. The mock-factory
    tests above did NOT catch this because they injected a deps dict
    that skipped the real factory entirely — this test closes that gap.

    To stay infra-free the test patches the source modules the
    closure's lazy imports resolve against:

    * ``explore_persona_space.backends.gcp.reconnect_or_none`` —
      captures the ``config=`` / ``runner=`` kwargs the closure passed
      it; the assertion is that BOTH resolved to non-None values
      pulled off ``gcp_backend.config`` / ``gcp_backend.runner`` (the
      public property reads that would have raised pre-fix).
    * ``explore_persona_space.backends.slurm_monitor.query_by_name`` —
      short-circuits the ``ssh robot-nibi squeue ...`` call so the
      SLURM closures don't require gcloud / DRAC SSH / a robot alias
      to be live.

    Both patches target the SOURCE module symbol (NOT a re-bound name
    on ``scripts.dispatch_issue``) because the factory's closures
    lazy-import their helpers from the source modules on each
    invocation — patching only ``scripts.dispatch_issue`` would miss
    the lazy import.

    The smoke is bounded: it exercises factory wiring + closure call
    sites, not real cloud / cluster contact. Failures look like
    ``AttributeError: 'GcpBackend' object has no attribute 'runner'``
    (C1 pre-fix) or a KeyError on the deps dict (a future refactor
    drops a key) — both are exactly the regression class this test
    pins.
    """
    from explore_persona_space.backends import gcp as gcp_module
    from explore_persona_space.backends import slurm_monitor as slurm_monitor_module
    from scripts import dispatch_issue as di

    # Patch the helpers BEFORE building the factory — the closures
    # lazy-import them at factory-call time and close over the result,
    # so a post-build patch would miss the rebind.
    captured_gcp_kwargs: dict[str, Any] = {}

    def _fake_gcp_reconnect(*, spec, config, runner):  # type: ignore[no-untyped-def]
        captured_gcp_kwargs["spec"] = spec
        captured_gcp_kwargs["config"] = config
        captured_gcp_kwargs["runner"] = runner
        return None  # "no live instance" — same shape the real fn returns

    def _fake_query_by_name(*, robot_alias, job_name, timeout=30):  # type: ignore[no-untyped-def]
        return None  # "no live job"

    monkeypatch.setattr(gcp_module, "reconnect_or_none", _fake_gcp_reconnect)
    monkeypatch.setattr(slurm_monitor_module, "query_by_name", _fake_query_by_name)
    # Slice-7: the production factory wires the real
    # ``backends.slurm.mila_socket_alive`` probe (``ssh mila true``);
    # in CI we have no real Mila socket, so patch it to a deterministic
    # ``False`` so the dependency-smoke check below doesn't reach out
    # over SSH. The router-skip-Mila path is exercised by the slice-7
    # ``test_router_skips_mila_when_socket_down`` test.
    from explore_persona_space.backends import slurm as slurm_module

    monkeypatch.setattr(slurm_module, "mila_socket_alive", lambda: False)

    expected_keys = {
        "runpod_backend",
        "free_backends",
        "gcp_backend",
        "marker_poster",
        "is_started",
        "is_live_after_cancel",
        "started_evidence_probe",
        "reconnect_fn",
        "mila_socket_alive",
    }

    deps = di._build_production_backends()
    assert set(deps) == expected_keys, (
        f"factory dropped or added keys: expected {expected_keys}, got {set(deps)}"
    )

    # Sanity: the public injection-seam reads (the C1 fix) actually
    # resolve. Pre-fix `gcp_backend._runner` would AttributeError;
    # the property promotion makes `.config` / `.runner` the public
    # reads.
    gcp_backend = deps["gcp_backend"]
    assert gcp_backend.config is not None, "GcpBackend.config public property must resolve"
    assert gcp_backend.runner is not None, "GcpBackend.runner public property must resolve"

    spec = RunSpec(
        issue=999,
        intent="lora-7b",
        backend="auto",
        extra={},
    )

    reconnect_fn = deps["reconnect_fn"]

    # GCP reconnect — this is the C1 site. Pre-fix this AttributeError'd
    # on ``gcp_backend._runner``. Post-fix it routes to the patched
    # ``_fake_gcp_reconnect`` with the public ``.config`` / ``.runner``
    # property values.
    out_gcp = reconnect_fn(deps["gcp_backend"], "gcp", spec)
    assert out_gcp is None, "patched _fake_gcp_reconnect returns None"
    assert captured_gcp_kwargs["config"] is gcp_backend.config, (
        "GCP reconnect must pass the backend's public ``config`` property — "
        "pre-fix this read raised AttributeError on the underscored name."
    )
    assert captured_gcp_kwargs["runner"] is gcp_backend.runner, (
        "GCP reconnect must pass the backend's public ``runner`` property — "
        "pre-fix the code path read ``gcp_backend._runner`` which doesn't "
        "exist (GcpBackend stores the runner as ``self._run``)."
    )

    # SLURM reconnect — patched query_by_name returns None so the
    # closure exits the no-live-job branch cleanly. The smoke validates
    # the M2 fix (public ``scratch_dir_for`` import) compiles + executes.
    nibi_backend = deps["free_backends"].get("nibi")
    assert nibi_backend is not None, "production factory must wire nibi"
    out_nibi = reconnect_fn(nibi_backend, "nibi", spec)
    assert out_nibi is None, "patched query_by_name returns None → reconnect returns None"

    # RunPod / unknown kinds: per the closure's docstring, both return
    # None (RunPod's existing pod_lifecycle.py is idempotent on its own).
    assert reconnect_fn(deps["runpod_backend"], "runpod", spec) is None
    assert reconnect_fn(deps["runpod_backend"], "wibble", spec) is None

    # is_started / is_live_after_cancel — for handles with
    # ``cluster is None`` (RunPod / GCP), they fall through to a poll
    # on the backend. Smoke that path with a stub backend; the closure
    # itself is the unit under test, not the backend.
    poll_calls: dict[str, int] = {"is_started": 0, "is_live_after_cancel": 0}

    class _StubPollBackend:
        def poll(self, _handle):  # type: ignore[no-untyped-def]
            from explore_persona_space.backends.base import PollResult

            poll_calls["is_started"] += 1
            return PollResult(
                status="running",
                current_phase="x",
                new_milestone=False,
                last_log_mtime_sec_ago=1,
                pid_alive=True,
                log_tail_excerpt="",
            )

    handle_gcp_like = _handle_for("gcp")  # cluster=None
    assert deps["is_started"](_StubPollBackend(), handle_gcp_like) is True
    # PollResult.status=="running" → is_live_after_cancel returns True
    # (the closure's "still-live" check rejects only {"done", "dead"}).
    assert deps["is_live_after_cancel"](_StubPollBackend(), handle_gcp_like) is True

    # started_evidence_probe — non-SLURM handles (cluster=None) return
    # None WITHOUT any SSH/rsync (the probe is SLURM-scratch-specific).
    assert deps["started_evidence_probe"](_StubPollBackend(), handle_gcp_like) is None

    # marker_poster + mila_socket_alive — exist and are callable, no
    # network needed to smoke.
    assert callable(deps["marker_poster"])
    # Slice-7: factory wires the REAL ``backends.slurm.mila_socket_alive``
    # probe. We monkeypatched it above to a deterministic ``False`` so
    # the smoke does not reach out over SSH; the wiring is what matters
    # here (router-skip-Mila behaviour is covered by the dedicated
    # ``test_router_skips_mila_when_socket_down`` test).
    assert deps["mila_socket_alive"]() is False


def test_slurm_reconnect_attaches_declaration(monkeypatch) -> None:
    """#598 (GCP parity, D7): the production ``_reconnect`` closure's
    rebuilt SLURM handle must carry the ``expected_artifacts``
    declaration derived from the recovered job id — a reconnected
    handle is exactly the handle finalize later consumes, and leaving
    it bare would silently re-create the #588 "missing declaration"
    FAIL on the recovery path."""
    from explore_persona_space.backends import slurm_monitor as slurm_monitor_module
    from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY
    from scripts import dispatch_issue as di

    def _fake_query_by_name(*, robot_alias, job_name, timeout=30):  # type: ignore[no-untyped-def]
        return "15956499"  # a live job was found under eps-issue-<N>

    monkeypatch.setattr(slurm_monitor_module, "query_by_name", _fake_query_by_name)
    deps = di._build_production_backends()

    spec = RunSpec(
        issue=999,
        intent="lora-7b",
        backend="nibi",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em",),
    )
    handle = deps["reconnect_fn"](deps["free_backends"]["nibi"], "nibi", spec)
    assert handle is not None
    assert handle.job_id == "15956499"
    decl = handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    assert decl["issue"] == 999
    # The sentinel path is attempt-namespaced by the RECOVERED job id —
    # matching what launch() would have declared for the same job.
    assert decl["sentinel_path"].endswith(
        "eval_results/issue_999/slurm-15956499/.completion-sentinel.json"
    )
    assert decl["hf_data_paths"] == ["issue999_slurm-15956499/raw_completions/"]


# ---------------------------------------------------------------------------
# issue #588 — --workload-cmd threading + exactly-one-of validation
# ---------------------------------------------------------------------------


def test_launch_workload_cmd_threaded_into_spec_verbatim(monkeypatch, tmp_path) -> None:
    """Mirror of ``test_launch_hydra_args_threaded_into_spec``: the
    custom command must land on the spec VERBATIM (quoting included)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "588",
                "--intent",
                "debug",
                "--workload-cmd",
                "bash scripts/issue588_smoke.sh --flag 'v 1'",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert nibi.launches[0].workload_cmd == "bash scripts/issue588_smoke.sh --flag 'v 1'"
    assert nibi.launches[0].hydra_args == ()


def test_launch_workload_cmd_and_hydra_both_is_parser_error(monkeypatch, tmp_path) -> None:
    """Both flags → argparse error (exit 2) BEFORE any backend is built."""
    _cd_to_tmp(monkeypatch, tmp_path)

    def exploding_factory():
        raise AssertionError("backends must not be built on a parser error")

    from scripts.dispatch_issue import main

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "launch",
                "--issue",
                "588",
                "--intent",
                "debug",
                "--workload-cmd",
                "bash scripts/issue588_smoke.sh",
                "--hydra",
                "seed=1",
            ],
            backends_factory=exploding_factory,
        )
    assert excinfo.value.code == 2


def test_launch_neither_workload_cmd_nor_hydra_is_parser_error(monkeypatch, tmp_path) -> None:
    """Neither flag → same exactly-one parser error (the #571 shape:
    a bare hydra launch with no overrides is never an intended
    production dispatch)."""
    _cd_to_tmp(monkeypatch, tmp_path)

    def exploding_factory():
        raise AssertionError("backends must not be built on a parser error")

    from scripts.dispatch_issue import main

    with pytest.raises(SystemExit) as excinfo:
        main(
            ["launch", "--issue", "588", "--intent", "debug"],
            backends_factory=exploding_factory,
        )
    assert excinfo.value.code == 2


def test_launch_workload_cmd_explicit_empty_counts_as_not_provided(monkeypatch, tmp_path) -> None:
    """``--workload-cmd ''`` is NOT a workload — it errors with the same
    exactly-one message (disambiguates None vs empty)."""
    _cd_to_tmp(monkeypatch, tmp_path)

    def exploding_factory():
        raise AssertionError("backends must not be built on a parser error")

    from scripts.dispatch_issue import main

    with pytest.raises(SystemExit) as excinfo:
        main(
            ["launch", "--issue", "588", "--intent", "debug", "--workload-cmd", ""],
            backends_factory=exploding_factory,
        )
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# issue #588 — finalize calls fetch_results BEFORE confirm_artifacts
# ---------------------------------------------------------------------------


def test_finalize_calls_fetch_results_before_confirm_artifacts(monkeypatch, tmp_path) -> None:
    """The GCP completion sentinel lives ON the VM; ``fetch_results`` is
    the scp pull that lands it locally and the verifier reads the LOCAL
    filesystem — so finalize MUST fetch before the confirm gate
    (latent slice-6 gap; without it every real GCP finalize FAILed
    confirm on the missing local sentinel)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 405, kind="nibi")
    nibi = _MockBackend(kind="nibi", confirm_passes=True)
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_upload_verification_currency_blocker", lambda _issue: None)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "finalize",
                "--issue",
                "405",
                "--handle-file",
                str(tmp_path / "issue-405-handle.json"),
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert nibi.call_sequence == ["fetch_results", "confirm_artifacts", "teardown"]


def test_finalize_fetch_results_crash_still_reaches_confirm_gate(monkeypatch, tmp_path) -> None:
    """``fetch_results`` is fail-soft by contract, but a CRASH must not
    become a finalize traceback — it logs loudly and lets the confirm
    gate FAIL with the right surfacing (teardown skipped, evidence
    preserved)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 406, kind="nibi")
    nibi = _MockBackend(kind="nibi", confirm_passes=False)

    def exploding_fetch(_handle):
        raise OSError("scp transport refused")

    nibi.fetch_results = exploding_fetch  # type: ignore[method-assign]
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_upload_verification_currency_blocker", lambda _issue: None)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "finalize",
                "--issue",
                "406",
                "--handle-file",
                str(tmp_path / "issue-406-handle.json"),
            ],
            backends_factory=factory,
        )
    # confirm FAIL surfaced as rc=3 (NOT rc=4 crash); teardown skipped.
    assert rc == 3
    body = json.loads(buf.getvalue().strip())
    assert body["reason"] == "confirm_artifacts_failed"
    assert len(nibi.confirms) == 1
    assert len(nibi.teardowns) == 0


# ---------------------------------------------------------------------------
# issue #909 — --execute-workload flag surface + CLI failure/success contracts
# ---------------------------------------------------------------------------


def _rp_shaped_handle(spec: RunSpec, extra_overrides: dict) -> RunHandle:
    return RunHandle(
        backend="runpod",
        cluster=None,
        job_id="",
        pod_name=f"pod-{spec.issue}",
        scratch_dir="/workspace",
        log_path=f"/workspace/logs/issue-{spec.issue}.log",
        extra={"issue": spec.issue, "intent": spec.intent, **extra_overrides},
    )


class _RunpodExecBackend(_MockBackend):
    """RunPod mock whose launch returns a #909-shaped handle extra."""

    def __init__(self, extra_overrides: dict) -> None:
        super().__init__(kind="runpod")
        self._extra_overrides = extra_overrides

    def launch(self, spec: RunSpec) -> RunHandle:
        self.launches.append(spec)
        return _rp_shaped_handle(spec, self._extra_overrides)


def _runpod_launch_args(*extra_args: str) -> list[str]:
    return [
        "launch",
        "--issue",
        "909",
        "--intent",
        "lora-7b",
        "--backend",
        "runpod",
        "--workload-cmd",
        "bash scripts/issue909_dispatch.sh",
        *extra_args,
    ]


def _pin_runpod_frontmatter(monkeypatch) -> None:
    import scripts.dispatch_issue as cli

    monkeypatch.setattr(cli, "_frontmatter_backend_value", lambda _issue: "runpod")


def test_launch_execute_workload_threads_to_spec_extra(monkeypatch, tmp_path) -> None:
    """#909: ``--execute-workload`` lands on ``spec.extra`` (the RunPod
    execution leg's opt-in)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _pin_runpod_frontmatter(monkeypatch)
    runpod = _MockBackend(kind="runpod")
    factory = _build_mock_factory(runpod=runpod)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(_runpod_launch_args("--execute-workload"), backends_factory=factory)
    assert rc == 0
    assert runpod.launches[0].extra.get("execute_workload") is True


def test_execute_workload_requires_workload_cmd(monkeypatch, tmp_path) -> None:
    """#909 AC3a (upheld Must-Fix): ``--execute-workload`` with ``--hydra``
    (no workload command) is REJECTED at parse time — exit 2, NO provision
    attempted (the fake factory is never called)."""
    _cd_to_tmp(monkeypatch, tmp_path)

    def exploding_factory():
        raise AssertionError("backends must not be built on a parser error")

    from scripts.dispatch_issue import main

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "launch",
                "--issue",
                "909",
                "--intent",
                "lora-7b",
                "--backend",
                "runpod",
                "--execute-workload",
                "--hydra",
                "foo=bar",
            ],
            backends_factory=exploding_factory,
        )
    assert excinfo.value.code == 2


def test_execute_workload_with_empty_workload_cmd_is_parser_error(monkeypatch, tmp_path) -> None:
    """#909 AC3a sibling: ``--execute-workload --workload-cmd ''`` is the
    same parse-time rejection (an empty command can never be a workload)."""
    _cd_to_tmp(monkeypatch, tmp_path)

    def exploding_factory():
        raise AssertionError("backends must not be built on a parser error")

    from scripts.dispatch_issue import main

    with pytest.raises(SystemExit) as excinfo:
        main(
            # Base args minus the workload-cmd VALUE, then an EMPTY value:
            # [..., "--workload-cmd", "", "--execute-workload"].
            [*_runpod_launch_args()[:-1], "", "--execute-workload"],
            backends_factory=exploding_factory,
        )
    assert excinfo.value.code == 2


def test_launch_runpod_workload_start_error_exits_2_with_reason(monkeypatch, tmp_path) -> None:
    """#909 AC3: a launch whose execution leg raises
    ``RunPodWorkloadStartError`` prints the failure JSON
    (``reason: runpod_workload_start_failed``) + exits 2 — never ok:true on
    a provision-only result when execution was requested."""
    from explore_persona_space.backends.runpod import RunPodWorkloadStartError

    _cd_to_tmp(monkeypatch, tmp_path)
    _pin_runpod_frontmatter(monkeypatch)
    runpod = _MockBackend(
        kind="runpod",
        launch_should_raise=RunPodWorkloadStartError(
            "workload did NOT verify alive on pod-909 (log /workspace/logs/issue-909.log)"
        ),
    )
    factory = _build_mock_factory(runpod=runpod)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(_runpod_launch_args("--execute-workload"), backends_factory=factory)
    assert rc == 2
    body = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert body["ok"] is False
    assert body["failure_class"] == "infra"
    assert body["reason"] == "runpod_workload_start_failed"
    assert "pod-909" in body["note"]


def test_launch_success_body_carries_workload_execution_keys(monkeypatch, tmp_path) -> None:
    """#909 AC6: the launch success JSON gains ``workload_executed`` /
    ``workload_pid`` / ``log_path``."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _pin_runpod_frontmatter(monkeypatch)
    runpod = _RunpodExecBackend({"workload_executed": True, "workload_pid": 777})
    factory = _build_mock_factory(runpod=runpod)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(_runpod_launch_args("--execute-workload"), backends_factory=factory)
    assert rc == 0
    body = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert body["ok"] is True
    assert body["workload_executed"] is True
    assert body["workload_pid"] == 777
    assert body["log_path"] == "/workspace/logs/issue-909.log"


def test_launch_belt_and_suspenders_executed_without_pid_exits_2(monkeypatch, tmp_path) -> None:
    """#909 belt-and-suspenders: a handle claiming ``workload_executed: true``
    with NO ``workload_pid`` (a future backend regression returning ok on a
    provision-only result) prints the failure JSON + exits 2."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _pin_runpod_frontmatter(monkeypatch)
    runpod = _RunpodExecBackend({"workload_executed": True})  # no workload_pid
    factory = _build_mock_factory(runpod=runpod)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(_runpod_launch_args("--execute-workload"), backends_factory=factory)
    assert rc == 2
    body = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert body["ok"] is False
    assert body["reason"] == "runpod_workload_start_failed"
    assert "workload_pid" in body["note"]


def test_repo_branch_auto_default_fires_for_explicit_runpod_with_execute_workload(
    monkeypatch, tmp_path
) -> None:
    """#909 AC6: the ``repo_branch`` auto-default (previously gated to the
    gcp/auto lanes) ALSO fires for explicit ``--backend runpod`` +
    ``--execute-workload`` — the execution leg's branch sync must target
    the issue branch, not `main` (the #763-shaped manual command)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _pin_runpod_frontmatter(monkeypatch)
    from scripts import dispatch_issue as di

    monkeypatch.setattr(di, "_current_git_branch", lambda: "issue-909-x")
    runpod = _MockBackend(kind="runpod")
    factory = _build_mock_factory(runpod=runpod)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(_runpod_launch_args("--execute-workload"), backends_factory=factory)
    assert rc == 0
    assert runpod.launches[0].extra.get("repo_branch") == "issue-909-x"


def test_repo_branch_explicit_flag_wins_over_auto_default(monkeypatch, tmp_path) -> None:
    """#909 AC6: an explicit ``--repo-branch`` always wins over the
    auto-default."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _pin_runpod_frontmatter(monkeypatch)
    from scripts import dispatch_issue as di

    monkeypatch.setattr(di, "_current_git_branch", lambda: "issue-909-x")
    runpod = _MockBackend(kind="runpod")
    factory = _build_mock_factory(runpod=runpod)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            _runpod_launch_args("--execute-workload", "--repo-branch", "issue-909-other"),
            backends_factory=factory,
        )
    assert rc == 0
    assert runpod.launches[0].extra.get("repo_branch") == "issue-909-other"


def test_repo_branch_auto_default_does_not_fire_for_explicit_runpod_without_flag(
    monkeypatch, tmp_path
) -> None:
    """#909 AC6 (negative control): explicit ``--backend runpod`` WITHOUT
    ``--execute-workload`` keeps today's behavior — no repo_branch
    auto-default (the experimenter owns the branch there)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _pin_runpod_frontmatter(monkeypatch)
    from scripts import dispatch_issue as di

    monkeypatch.setattr(di, "_current_git_branch", lambda: "issue-909-x")
    runpod = _MockBackend(kind="runpod")
    factory = _build_mock_factory(runpod=runpod)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(_runpod_launch_args(), backends_factory=factory)
    assert rc == 0
    assert "repo_branch" not in (runpod.launches[0].extra or {})


# ---------------------------------------------------------------------------
# #934: --lane-suffix + reconnect loudness (launch JSON additive keys)
# ---------------------------------------------------------------------------


class _ReconnectedExtraBackend(_MockBackend):
    """launch() returns a handle marked extra['reconnected']=True — the
    GCP-INTERNAL reconnect shape (route() thinks it launched fresh; only
    the handle extra carries the flag, the reason is NOT 'reconnect')."""

    def launch(self, spec: RunSpec) -> RunHandle:
        from dataclasses import replace

        h = super().launch(spec)
        return replace(h, extra={**h.extra, "reconnected": True})


def _build_factory_with_reconnect(
    *,
    nibi: _MockBackend,
    reconnect_fn: Any = None,
) -> Any:
    """Like _build_mock_factory but with an injectable reconnect_fn."""

    def _factory() -> dict[str, Any]:
        return {
            "runpod_backend": _MockBackend(kind="runpod"),
            "free_backends": {"nibi": nibi},
            "gcp_backend": None,
            "marker_poster": lambda **_kw: None,
            "is_started": lambda _b, _h: True,
            "is_live_after_cancel": lambda _b, _h: False,
            "reconnect_fn": reconnect_fn or (lambda _b, _k, _s: None),
            "mila_socket_alive": lambda: False,
        }

    return _factory


def _reconnect_fn_for(issue: int) -> Any:
    """A reconnect_fn returning a live nibi handle (router-scan reconnect:
    reason == 'reconnect', handle extra WITHOUT the reconnected flag)."""
    live = RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="live-9",
        pod_name=f"eps-issue-{issue}",
        scratch_dir="/s",
        log_path="/l",
        extra={"issue": issue},
    )
    return lambda _b, k, _s: live if k == "nibi" else None


def test_launch_lane_suffix_threads_extra_and_sidecar(monkeypatch, tmp_path) -> None:
    """--lane-suffix threads into spec.extra['lane_suffix'] AND the sidecar
    lands at the suffixed issue-<N>-<suffix>-handle.json path; the JSON
    carries the lane_suffix key (#934)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "9341",
                "--intent",
                "lora-7b",
                "--hydra",
                "smoke=1",
                "--lane-suffix",
                "cpu",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is True
    assert body["lane_suffix"] == "cpu"
    assert body["handle_sidecar_path"].endswith("issue-9341-cpu-handle.json")
    assert (tmp_path / ".claude" / "cache" / "issue-9341-cpu-handle.json").exists()
    assert len(nibi.launches) == 1
    assert nibi.launches[0].extra["lane_suffix"] == "cpu"


@pytest.mark.parametrize(
    "workload_args", [["--workload-cmd", "bash scripts/x.sh"], ["--hydra", "smoke=1"]]
)
@pytest.mark.parametrize("reconnect_source", ["reason", "extra"])
def test_launch_reconnect_workload_cmd_reports_workload_dispatched_false(
    monkeypatch, tmp_path, caplog, reconnect_source: str, workload_args: list[str]
) -> None:
    """Round-1 Must-Fix matrix (the exact #923 recurrence path): BOTH
    reconnect branches (router-scan reason-only x GCP-internal extra-only)
    x BOTH workload surfaces (--workload-cmd x --hydra) report
    reconnected=true / workload_dispatched=false / reconnect_note, warn,
    and KEEP ok:true + rc 0 (the exit-75 rerun contract)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    issue = 9342
    if reconnect_source == "reason":
        factory = _build_factory_with_reconnect(
            nibi=_MockBackend(kind="nibi"), reconnect_fn=_reconnect_fn_for(issue)
        )
    else:
        factory = _build_factory_with_reconnect(nibi=_ReconnectedExtraBackend(kind="nibi"))

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with caplog.at_level(logging.WARNING), redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", str(issue), "--intent", "lora-7b", *workload_args],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is True  # reconnect stays a SUCCESS exit (exit-75 rerun contract)
    assert body["reconnected"] is True
    assert body["workload_dispatched"] is False
    assert "reconnect_note" in body
    assert "dispatched" in body["reconnect_note"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("dispatched" in m and "NO workload" in m for m in messages), messages


def test_launch_fresh_create_reports_workload_dispatched_true(monkeypatch, tmp_path) -> None:
    """A genuinely fresh launch: reconnected=false, workload_dispatched=true,
    no reconnect_note, no lane_suffix key (unsuffixed byte-identity of the
    additive-key surface)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "9343", "--intent", "lora-7b", "--hydra", "smoke=1"],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is True
    assert body["reconnected"] is False
    assert body["workload_dispatched"] is True
    assert "reconnect_note" not in body
    assert "lane_suffix" not in body
    assert "lane_suffix_unhonored_by_lane" not in body


def test_launch_no_suffix_leaves_extra_clean(monkeypatch, tmp_path) -> None:
    """Round-1 Must-Fix (unsuffixed byte-identity PINNED): a launch WITHOUT
    --lane-suffix must leave spec.extra with NO lane_suffix key AT ALL —
    an extra['lane_suffix'] = None mutant would flip canonicalize_spec
    output and every live unsuffixed lease spec-hash (fleet-wide lease
    replace-on-mismatch)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from explore_persona_space.backends.router import canonicalize_spec
    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "9344", "--intent", "lora-7b", "--hydra", "smoke=1"],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert "lane_suffix" not in body
    assert len(nibi.launches) == 1
    captured = nibi.launches[0]
    assert "lane_suffix" not in (captured.extra or {})
    # Spec-hash identity: the canonical form carries no trace of the key.
    canonical = canonicalize_spec(captured)
    assert "lane_suffix" not in json.dumps(canonical)


def test_finalize_lane_suffix_resolves_suffixed_sidecar(monkeypatch, tmp_path) -> None:
    """Round-1 Must-Fix: `finalize --lane-suffix cpu` resolves the SUFFIXED
    sidecar — a getattr typo silently finalizing lane B against lane A's
    handle is the same silent-wrong-lane class as #923."""
    _cd_to_tmp(monkeypatch, tmp_path)
    cache = tmp_path / ".claude" / "cache"
    # Suffixed handle (the one finalize must act on) + an unsuffixed DECOY
    # with a different job_id (must NOT be touched).
    suffixed = RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="job-suffixed",
        pod_name="pod-9346-cpu",
        scratch_dir="/scratch",
        log_path="/log",
        extra={
            "issue": 9346,
            "intent": "lora-7b",
            EXPECTED_ARTIFACTS_HANDLE_KEY: {"issue": 9346, "sentinel_path": "/tmp/s.json"},
        },
    )
    decoy = RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="job-unsuffixed-decoy",
        pod_name="pod-9346",
        scratch_dir="/scratch",
        log_path="/log",
        extra={
            "issue": 9346,
            "intent": "lora-7b",
            EXPECTED_ARTIFACTS_HANDLE_KEY: {"issue": 9346, "sentinel_path": "/tmp/s.json"},
        },
    )
    write_handle_sidecar(suffixed, cache / "issue-9346-cpu-handle.json")
    write_handle_sidecar(decoy, cache / "issue-9346-handle.json")
    nibi = _MockBackend(kind="nibi", confirm_passes=True)
    factory = _build_mock_factory(nibi=nibi)

    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_upload_verification_currency_blocker", lambda _issue: None)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["finalize", "--issue", "9346", "--lane-suffix", "cpu"],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is True
    # Teardown fired on the SUFFIXED lane's handle, not the decoy.
    assert len(nibi.teardowns) == 1
    assert nibi.teardowns[0].job_id == "job-suffixed"
    # The suffixed sidecar retired to .finalized; the decoy untouched.
    assert not (cache / "issue-9346-cpu-handle.json").exists()
    assert (cache / "issue-9346-cpu-handle.json.finalized").exists()
    assert (cache / "issue-9346-handle.json").exists()


def test_launch_invalid_lane_suffix_rejected_at_parse_time(monkeypatch, tmp_path) -> None:
    """A malformed --lane-suffix errors at the argparse surface (SystemExit
    2) BEFORE any backend is built."""
    _cd_to_tmp(monkeypatch, tmp_path)

    def _exploding_factory() -> dict[str, Any]:
        raise AssertionError("backends_factory must not be called on a parse error")

    from scripts.dispatch_issue import main

    for bad in ("CPU", "a_b", "-x", "x-", "a" * 44):
        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "launch",
                    "--issue",
                    "9345",
                    "--intent",
                    "lora-7b",
                    "--hydra",
                    "smoke=1",
                    "--lane-suffix",
                    bad,
                ],
                backends_factory=_exploding_factory,
            )
        assert excinfo.value.code == 2


def test_launch_lane_suffix_non_gcp_lane_warns(monkeypatch, tmp_path, caplog) -> None:
    """--lane-suffix on a launch that resolves to a NON-GCP lane: the
    instance/job-name isolation is GCP-only, so the JSON carries
    lane_suffix_unhonored_by_lane + a loud warning (the gap is loud, not
    silent — plan §3.7)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with caplog.at_level(logging.WARNING), redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "9347",
                "--intent",
                "lora-7b",
                "--hydra",
                "smoke=1",
                "--lane-suffix",
                "cpu",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is True
    assert body["lane_suffix"] == "cpu"
    assert body["lane_suffix_unhonored_by_lane"] == "nibi"
    messages = [r.getMessage() for r in caplog.records]
    assert any("GCP-only" in m for m in messages), messages


# ---------------------------------------------------------------------------
# #954 — the explicit-override typed arm persists the PARTIAL handle sidecar
# ---------------------------------------------------------------------------


def _partial_runpod_handle_954():
    from explore_persona_space.backends.base import RunHandle

    return RunHandle(
        backend="runpod",
        cluster=None,
        job_id="",
        pod_name="pod-909",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-909.log",
        extra={
            "issue": 909,
            "workload_executed": False,
            "workload_start_error": "ssh TimeoutExpired",
        },
    )


def test_cmd_launch_workload_start_failed_writes_sidecar_and_names_pod(
    monkeypatch, tmp_path
) -> None:
    """#954 AC7: a typed-with-handle workload-start failure on the explicit
    override path exits 2 with ``reason: runpod_workload_start_failed``
    (unchanged) PLUS ``pod_name`` + ``sidecar_written: true``, and the handle
    sidecar exists on disk at the canonical path — the billing pod is visible
    to the handle machinery (poll / finalize / re-drive stay chained). The
    backend mock raises from ``launch`` below ``dispatch_for_issue`` (the real
    ``route()`` runs), so assumption 5 — no exception wrapping — is
    exercised."""
    from explore_persona_space.backends.issue_dispatch import read_handle_sidecar
    from explore_persona_space.backends.runpod import RunPodWorkloadStartError

    _cd_to_tmp(monkeypatch, tmp_path)
    _pin_runpod_frontmatter(monkeypatch)
    runpod = _MockBackend(
        kind="runpod",
        launch_should_raise=RunPodWorkloadStartError(
            "workload did NOT verify alive on pod-909", handle=_partial_runpod_handle_954()
        ),
    )
    factory = _build_mock_factory(runpod=runpod)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(_runpod_launch_args("--execute-workload"), backends_factory=factory)
    assert rc == 2
    body = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert body["ok"] is False
    assert body["failure_class"] == "infra"
    assert body["reason"] == "runpod_workload_start_failed"  # unchanged (#909)
    assert body["pod_name"] == "pod-909"
    assert body["sidecar_written"] is True
    # The handle sidecar exists on disk at the canonical (tmp-pinned) path.
    sidecar = tmp_path / ".claude" / "cache" / "issue-909-handle.json"
    assert sidecar.is_file()
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "runpod"
    assert recovered.pod_name == "pod-909"
    assert recovered.extra["workload_executed"] is False
    assert recovered.extra["workload_start_error"]


def test_cmd_launch_workload_start_failed_sidecar_write_oserror_fail_loud(
    monkeypatch, tmp_path
) -> None:
    """#954 (round-1 critique, statistics MF): the Surface-5 fail-loud contract
    — a sidecar-write OSError NEVER masks the typed failure: rc 2, ``reason``
    unchanged, ``pod_name`` present, ``sidecar_written: false`` + the OSError
    recorded in the JSON note (no unhandled OSError escapes the typed arm)."""
    from explore_persona_space.backends.runpod import RunPodWorkloadStartError

    _cd_to_tmp(monkeypatch, tmp_path)
    _pin_runpod_frontmatter(monkeypatch)
    runpod = _MockBackend(
        kind="runpod",
        launch_should_raise=RunPodWorkloadStartError(
            "workload did NOT verify alive on pod-909", handle=_partial_runpod_handle_954()
        ),
    )
    factory = _build_mock_factory(runpod=runpod)

    def _raising_write(handle, path):
        raise OSError("Disk quota exceeded (EDQUOT) writing the sidecar")

    monkeypatch.setattr(
        "explore_persona_space.backends.issue_dispatch.write_handle_sidecar",
        _raising_write,
    )

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(_runpod_launch_args("--execute-workload"), backends_factory=factory)
    assert rc == 2
    body = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert body["ok"] is False
    assert body["failure_class"] == "infra"
    assert body["reason"] == "runpod_workload_start_failed"
    assert body["pod_name"] == "pod-909"
    assert body["sidecar_written"] is False
    assert "handle sidecar write FAILED" in body["note"]
    assert "EDQUOT" in body["note"]
    # The typed failure's own message is still the note's lead.
    assert "did NOT verify alive" in body["note"]


def test_min_ram_gb_lands_in_spec_extra(monkeypatch, tmp_path) -> None:
    """#1010: ``--min-ram-gb`` threads to ``spec.extra['min_ram_gb']`` (the
    RunPod CPU-fallback feasibility gate's RAM channel — RunPod CPU instances
    have FIXED RAM, so an unsatisfiable requirement refuses the fallback
    typed instead of provisioning an undersized pod). Mirror of the
    ``--boot-disk-gb`` threading test above."""
    _cd_to_tmp(monkeypatch, tmp_path)
    gcp = _MockBackend(kind="gcp")
    posts: list[dict[str, Any]] = []
    factory = _build_mock_factory(gcp=gcp, marker_posts=posts)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "1010",
                "--intent",
                "cpu-mid",
                "--backend",
                "gcp",
                "--boot-disk-gb",
                "80",
                "--min-ram-gb",
                "32",
                "--hydra",
                "smoke=1",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra["min_ram_gb"] == 32
    assert gcp.launches[0].extra["boot_disk_gb"] == 80
