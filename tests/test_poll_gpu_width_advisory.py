"""Tests for the m-of-N GPU-width advisory in scripts/poll_pipeline.py (#873).

The advisory posts a one-per-phase ``[gpu-width-advisory]`` ``epm:progress``
marker when a STABLE strict subset of GPUs sits idle for
``GPU_WIDTH_ADVISORY_MIN`` minutes on a multi-GPU pod while the run is
healthy (the #813 idle-width / #664 spend-leak class). These tests pin:

* ``_parse_gpu_utils`` — the extracted probe-string parser + the
  behavior-identical ``_gpu_idle`` regression (the #873 extraction);
* ``_gpu_width_advisory_update`` — the pure decision core (stable-idle-set
  requirement, ``>=`` boundary, resets on unknown/unparseable/all-idle/
  all-active/N<2/set-change/phase-change/non-running, per-phase de-dup);
* ``_maybe_post_gpu_width_advisory`` — the wiring (note shape + extras,
  post-failure retry, relaunch reset via ``_tripwire_run_scope`` — AC #6,
  and the two-tick ``_save_state``/``_load_state`` str->str round-trip).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_gpu_width_under_test")


# ── _parse_gpu_utils + the _gpu_idle extraction regression ──────────────────


def test_parse_gpu_utils_contract() -> None:
    assert pp._parse_gpu_utils("0,0,95,0") == [0, 0, 95, 0]
    assert pp._parse_gpu_utils("0, 0, 95, 0") == [0, 0, 95, 0]  # spaces tolerated
    assert pp._parse_gpu_utils("0") == [0]
    assert pp._parse_gpu_utils("unknown") is None
    assert pp._parse_gpu_utils("") is None
    assert pp._parse_gpu_utils("garbage") is None
    assert pp._parse_gpu_utils("5,abc,0") is None  # ANY bad token -> None
    assert pp._parse_gpu_utils(",") is None  # zero tokens -> None


def test_gpu_idle_unchanged_by_parse_extraction() -> None:
    """Regression-pins ``_gpu_idle`` post-extraction: unknown/empty/garbage
    -> False (fail-safe), all-idle -> True, one busy card -> False."""
    assert pp._gpu_idle("unknown") is False
    assert pp._gpu_idle("") is False
    assert pp._gpu_idle("garbage") is False
    assert pp._gpu_idle("0,abc") is False
    assert pp._gpu_idle("0,0,0,0,0,0,0,0") is True
    assert pp._gpu_idle("0") is True
    assert pp._gpu_idle(f"{pp.GPU_IDLE_UTIL_THRESHOLD},0") is True  # at threshold = idle
    assert pp._gpu_idle("0,0,95,0") is False


# ── _gpu_width_advisory_update (pure decision core) ──────────────────────────

# 8-GPU pod: idle {0,1,3,7} (<= 5%), active {2,4,5,6}; span exactly 45 min.
WIDTH_KW: dict[str, Any] = {
    "status": "running",
    "gpu_util": "0,0,95,0,88,90,92,0",
    "current_phase": "extract",
    "prev_phase": "extract",
    "prev_width_since_epoch": 1000,
    "prev_idle_indices": (0, 1, 3, 7),
    "advised_phases": set(),
    "now_epoch": 1000 + 45 * 60,
    "advisory_min": 45,
}

RESET = pp.GpuWidthAdvisoryUpdate(
    should_post=False, width_since_epoch=0, idle_indices=(), span_sec=0
)


def test_width_partial_idle_stable_set_posts_after_threshold() -> None:
    update = pp._gpu_width_advisory_update(**WIDTH_KW)
    assert update.should_post is True
    assert update.idle_indices == (0, 1, 3, 7)
    assert update.span_sec == 45 * 60


def test_width_span_exactly_at_threshold_posts() -> None:
    """The boundary is ``>=`` (asymmetric with the ETA tripwire's strict
    ``>`` — both pinned); one second UNDER does not post."""
    assert pp._gpu_width_advisory_update(**WIDTH_KW).should_post is True  # exactly 45 min
    under = {**WIDTH_KW, "now_epoch": 1000 + 45 * 60 - 1}
    assert pp._gpu_width_advisory_update(**under).should_post is False


def test_width_all_idle_does_not_fire() -> None:
    """All-idle is the EXISTING idle advisory's domain — disjoint by
    construction; the width advisory resets."""
    kw = {**WIDTH_KW, "gpu_util": "0,0,0,0,0,0,0,0", "prev_idle_indices": ()}
    assert pp._gpu_width_advisory_update(**kw) == RESET


def test_width_all_active_resets() -> None:
    kw = {**WIDTH_KW, "gpu_util": "90,88,95,80,99,91,87,93"}
    assert pp._gpu_width_advisory_update(**kw) == RESET


def test_width_single_gpu_pod_never_fires() -> None:
    for sample in ("0", "95"):
        kw = {**WIDTH_KW, "gpu_util": sample, "prev_idle_indices": ()}
        assert pp._gpu_width_advisory_update(**kw) == RESET


def test_width_unknown_sample_resets_span() -> None:
    kw = {**WIDTH_KW, "gpu_util": "unknown"}
    assert pp._gpu_width_advisory_update(**kw) == RESET


def test_width_unparseable_sample_resets_span() -> None:
    kw = {**WIDTH_KW, "gpu_util": "0,abc,95,0,88,90,92,0"}
    assert pp._gpu_width_advisory_update(**kw) == RESET


def test_width_idle_set_change_resets_span() -> None:
    """A CHURNING idle set is staggered shard progress, not the #813
    structurally-unused-GPUs signature — the span restarts at NOW, so
    rotating idleness never accumulates."""
    kw = {**WIDTH_KW, "prev_idle_indices": (0, 1, 2, 7)}
    update = pp._gpu_width_advisory_update(**kw)
    assert update.should_post is False
    assert update.width_since_epoch == WIDTH_KW["now_epoch"]  # restarted
    assert update.span_sec == 0


def test_width_phase_change_resets_span() -> None:
    kw = {**WIDTH_KW, "prev_phase": "train"}
    update = pp._gpu_width_advisory_update(**kw)
    assert update.should_post is False
    assert update.width_since_epoch == WIDTH_KW["now_epoch"]


def test_width_non_running_status_resets() -> None:
    for status in ("stalled", "dead", "done", "gate"):
        kw = {**WIDTH_KW, "status": status}
        assert pp._gpu_width_advisory_update(**kw) == RESET


def test_width_disabled_when_min_non_positive() -> None:
    assert pp._gpu_width_advisory_update(**{**WIDTH_KW, "advisory_min": 0}) == RESET
    assert pp._gpu_width_advisory_update(**{**WIDTH_KW, "advisory_min": -5}) == RESET


def test_width_already_advised_phase_does_not_repost() -> None:
    kw = {**WIDTH_KW, "advised_phases": {"extract"}}
    update = pp._gpu_width_advisory_update(**kw)
    assert update.should_post is False
    assert update.width_since_epoch == 1000  # span keeps accumulating


# ── _maybe_post_gpu_width_advisory (wiring) ──────────────────────────────────


def _seeded_state(now: int, *, advisory_min: int | None = None) -> dict[str, str]:
    minutes = advisory_min if advisory_min is not None else pp.GPU_WIDTH_ADVISORY_MIN
    return {
        "phase": "extract",
        "gpu_width_since_epoch": str(now - minutes * 60),
        "gpu_width_idle_set": "0,1,3,7",
        "gpu_width_advised_phases": "",
    }


def test_maybe_post_gpu_width_advisory_note_names_indices_and_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    now = 100_000
    _since, idle_set, advised, posted_flag = pp._maybe_post_gpu_width_advisory(
        issue=873,
        pod="pod-873",
        status="running",
        gpu_util="0,0,95,0,88,90,92,0",
        current_phase="extract",
        prev_state=_seeded_state(now),
        now_epoch=now,
    )
    assert posted_flag is True and "extract" in advised
    assert idle_set == (0, 1, 3, 7)
    (p,) = posted
    assert p["key"] == "epm:progress"
    assert p["gpu_width_advisory"] is True
    assert p["phase"] == "extract" and p["pod"] == "pod-873"
    note = p["note"]
    assert note.startswith("[gpu-width-advisory]")
    assert "4 of 8 GPUs" in note
    assert "idle GPU indices: 0,1,3,7" in note
    assert f"{pp.GPU_WIDTH_ADVISORY_MIN} min" in note
    assert "nothing was stopped" in note


def test_width_post_failure_phase_not_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _flaky(issue, key, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("marker post failed")

    monkeypatch.setattr(pp, "post_event", _flaky)
    now = 100_000
    kw: dict[str, Any] = {
        "issue": 873,
        "pod": "pod-873",
        "status": "running",
        "gpu_util": "0,0,95,0,88,90,92,0",
        "current_phase": "extract",
        "prev_state": _seeded_state(now),
        "now_epoch": now,
    }
    _since, _idle, advised1, posted1 = pp._maybe_post_gpu_width_advisory(**kw)
    assert posted1 is False and "extract" not in advised1  # failure -> NOT recorded
    _since, _idle, advised2, posted2 = pp._maybe_post_gpu_width_advisory(**kw)
    assert posted2 is True and "extract" in advised2  # next tick retries


def test_width_relaunch_resets_advised_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC #6: a fresh epm:run-launched epoch clears the width keys; the
    SECOND run's own sustained partial-width span still posts."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    now = 1_000_000
    prev = {
        "phase": "extract",
        "gpu_width_since_epoch": str(now - 10 * 3600),
        "gpu_width_idle_set": "0,1,3,7",
        "gpu_width_advised_phases": "extract",  # the PREVIOUS run already advised
        "tripwire_run_epoch": "1000",
    }
    # Fresh run launched 2 min ago -> the anchor is newer by >60s -> reset.
    state, epoch = pp._tripwire_run_scope(prev, run_age_sec=120.0, now_epoch=now)
    assert epoch == now - 120
    assert "gpu_width_advised_phases" not in state
    kw: dict[str, Any] = {
        "issue": 873,
        "pod": "pod-873",
        "status": "running",
        "gpu_util": "0,0,95,0,88,90,92,0",
        "current_phase": "extract",
        "prev_state": state,
        "now_epoch": now,
    }
    # Tick 1 of the fresh run: the span RESTARTS (no stale carry-over).
    since1, idle1, advised1, posted1 = pp._maybe_post_gpu_width_advisory(**kw)
    assert posted1 is False and since1 == now and advised1 == set()
    # Tick 2, past the threshold with the SAME stable idle set: posts again.
    tick2_state = {
        "phase": "extract",
        "gpu_width_since_epoch": str(since1),
        "gpu_width_idle_set": ",".join(str(i) for i in idle1),
        "gpu_width_advised_phases": ",".join(sorted(advised1)),
    }
    now2 = now + pp.GPU_WIDTH_ADVISORY_MIN * 60
    _s, _i, advised2, posted2 = pp._maybe_post_gpu_width_advisory(
        **{**kw, "prev_state": tick2_state, "now_epoch": now2}
    )
    assert posted2 is True and "extract" in advised2
    assert len(posted) == 1


def test_width_two_tick_state_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Span + idle set survive the ``_save_state``/``_load_state`` str->str
    round-trip across two ticks (incl. empty-set parse-back and the
    corrupted-int span reset)."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    state_file = tmp_path / "poll-pipeline-873.json"
    now = 100_000
    kw: dict[str, Any] = {
        "issue": 873,
        "pod": "pod-873",
        "status": "running",
        "gpu_util": "0,0,95,0,88,90,92,0",
        "current_phase": "extract",
        "now_epoch": now,
    }

    # Tick 1: span starts; persist through _save_state exactly as poll_once does.
    prev = pp._load_state(state_file, 873)
    assert prev == {}
    since, idle_set, advised, posted_flag = pp._maybe_post_gpu_width_advisory(prev_state=prev, **kw)
    assert posted_flag is False and since == now and idle_set == (0, 1, 3, 7)
    pp._save_state(
        state_file,
        873,
        {
            "phase": "extract",
            "gpu_width_since_epoch": str(since),
            "gpu_width_idle_set": ",".join(str(i) for i in idle_set),
            "gpu_width_advised_phases": ",".join(sorted(advised)),
        },
    )

    # Tick 2 past the threshold: the reloaded str state parses back and posts.
    prev2 = pp._load_state(state_file, 873)
    assert prev2["gpu_width_since_epoch"] == str(now)
    assert prev2["gpu_width_idle_set"] == "0,1,3,7"
    now2 = now + pp.GPU_WIDTH_ADVISORY_MIN * 60
    since2, _idle2, advised2, posted2 = pp._maybe_post_gpu_width_advisory(
        prev_state=prev2, **{**kw, "now_epoch": now2}
    )
    assert posted2 is True and "extract" in advised2 and since2 == now

    # Empty-set parse-back: an all-active tick writes "" for the idle set;
    # the next tick reads it back as () without crashing.
    pp._save_state(
        state_file,
        873,
        {
            "phase": "extract",
            "gpu_width_since_epoch": "0",
            "gpu_width_idle_set": "",
            "gpu_width_advised_phases": "",
        },
    )
    prev3 = pp._load_state(state_file, 873)
    since3, idle3, _adv3, posted3 = pp._maybe_post_gpu_width_advisory(
        prev_state=prev3, **{**kw, "now_epoch": now2}
    )
    assert posted3 is False and since3 == now2 and idle3 == (0, 1, 3, 7)

    # Corrupted-int reset: a garbage span epoch resets (restarts at NOW),
    # never raises into the poll tick.
    corrupt = {
        "phase": "extract",
        "gpu_width_since_epoch": "garbage",
        "gpu_width_idle_set": "0,1,x",
        "gpu_width_advised_phases": "",
    }
    since4, idle4, _adv4, posted4 = pp._maybe_post_gpu_width_advisory(
        prev_state=corrupt, **{**kw, "now_epoch": now2}
    )
    assert posted4 is False and since4 == now2 and idle4 == (0, 1, 3, 7)
