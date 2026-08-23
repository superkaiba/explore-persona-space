"""Issue #2479 — per-cell stories-revision resolution in the kept-story stager.

Codex `hf-prefix-realized-vs-plan` regression pins (hermetic — every Hub call
faked at the external boundary with def-mirroring fakes):

(a) parent cells resolve the fixed plan-§10 STORIES_PIN with ZERO network;
(b) a panel `char_2479_*` cell resolves its RECORDED per-cell generation
    upload revision from the `upload_revision_<mode>_<model>.json` sidecar —
    NEVER the parent STORIES_PIN (the cell postdates the pin: staging it at
    the pin 404s at capture time, the round-1 production-crash risk);
(c) a genuinely missing sidecar head-resolves the data repo (loud WARN);
(d) THE regression: absent local stories, `stage_variant` on a panel cell
    REQUESTS its recorded revision from `stage_hub_prefix` — the parent-cell
    twin requests STORIES_PIN and never fetches a sidecar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from huggingface_hub.utils import EntryNotFoundError

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1345_stage_char_stories as stager  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

RECORDED_REV = "feedc0de" * 5  # 40 hex chars, != STORIES_PIN
PANEL_VARIANT = "char_2479_iris"


def _fake_sidecar_fetcher(seen: dict, *, missing: bool = False):
    """Def-mirroring fake of hub.stage_hub_file (the sidecar fetch boundary)."""

    def fake_stage_hub_file(
        repo_id: str,
        path_in_repo: str,
        target,
        *,
        repo_type: str = "dataset",
        revision=None,
        token=None,
        overwrite: bool = False,
        size_bytes=None,
    ):
        seen.setdefault("sidecar_requests", []).append(path_in_repo)
        if missing:
            raise EntryNotFoundError("404: sidecar not found")
        Path(target).write_text(json.dumps({"data_repo_revision_at_or_after_upload": RECORDED_REV}))
        return Path(target)

    return fake_stage_hub_file


# ---------------------------------------------------------------------------
# (a)-(c) resolve_stories_revision policy
# ---------------------------------------------------------------------------
def test_parent_cell_resolves_pin_with_zero_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):  # any Hub touch is a failure
        raise AssertionError("parent-cell resolution must not touch the Hub")

    monkeypatch.setattr(hub, "stage_hub_file", _boom)
    monkeypatch.setattr(hub, "retry_transient", _boom)
    assert stager.resolve_stories_revision("char_helios_op") == stager.STORIES_PIN


def test_panel_cell_resolves_recorded_sidecar_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(hub, "stage_hub_file", _fake_sidecar_fetcher(seen))
    rev = stager.resolve_stories_revision(PANEL_VARIANT)
    assert rev == RECORDED_REV and rev != stager.STORIES_PIN
    # The sidecar path is the gen phase's realized per-cell filename.
    assert seen["sidecar_requests"] == [
        f"issue1345_framing/{PANEL_VARIANT}/raw_completions/stories/"
        "upload_revision_paired_instruct.json"
    ]


def test_panel_op_cell_sidecar_uses_op_mode_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(hub, "stage_hub_file", _fake_sidecar_fetcher(seen))
    stager.resolve_stories_revision("char_2479_iris_op")
    assert seen["sidecar_requests"][0].endswith("upload_revision_paired_op_instruct.json")


def test_panel_cell_missing_sidecar_head_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(hub, "stage_hub_file", _fake_sidecar_fetcher(seen, missing=True))

    def fake_retry_transient(fn, *, what: str = "", **kwargs):
        seen["retry_what"] = what
        return "a1b2c3d4" * 5  # the head sha, without touching the Hub

    monkeypatch.setattr(hub, "retry_transient", fake_retry_transient)
    rev = stager.resolve_stories_revision(PANEL_VARIANT)
    assert rev == "a1b2c3d4" * 5 and rev != stager.STORIES_PIN
    assert "repo_info" in seen["retry_what"]


# ---------------------------------------------------------------------------
# (d) stage_variant requests the resolved revision (the codex regression)
# ---------------------------------------------------------------------------
def _fake_prefix_stager(seen: dict, variant: str):
    """Def-mirroring fake of hub.stage_hub_prefix: records the requested
    revision and materializes the consumer-expected mirror layout."""

    def fake_stage_hub_prefix(
        repo_id: str,
        prefix: str,
        dest_dir,
        *,
        repo_type: str = "dataset",
        revision=None,
        token=None,
        max_workers: int = 6,
        **kwargs,
    ):
        seen.setdefault("prefix_revisions", []).append(revision)
        mirror = Path(dest_dir) / prefix
        mirror.mkdir(parents=True, exist_ok=True)
        kept, yld = stager.expected_files(variant)
        (mirror / kept).write_text('{"conv_id": "s1"}\n')
        (mirror / yld).write_text("{}")
        return [f"{prefix}/{kept}", f"{prefix}/{yld}"]

    return fake_stage_hub_prefix


def test_stage_variant_panel_cell_requests_recorded_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Absent local stories, a panel cell's stage must request its RECORDED
    # P1 generation upload revision — never the parent STORIES_PIN.
    monkeypatch.setattr(stager, "CHAR_VARIANTS", (*stager.CHAR_VARIANTS, PANEL_VARIANT))
    seen: dict = {}
    monkeypatch.setattr(hub, "stage_hub_file", _fake_sidecar_fetcher(seen))
    monkeypatch.setattr(hub, "stage_hub_prefix", _fake_prefix_stager(seen, PANEL_VARIANT))
    dest = stager.stage_variant(PANEL_VARIANT, dest_root=tmp_path)
    assert seen["prefix_revisions"] == [RECORDED_REV]
    assert stager.STORIES_PIN not in seen["prefix_revisions"]
    kept, yld = stager.expected_files(PANEL_VARIANT)
    assert (dest / kept).is_file() and (dest / yld).is_file()


def test_stage_variant_parent_cell_keeps_pin_and_skips_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    def _no_sidecar(*a, **k):
        raise AssertionError("parent cells never fetch an upload-revision sidecar")

    monkeypatch.setattr(hub, "stage_hub_file", _no_sidecar)
    monkeypatch.setattr(hub, "stage_hub_prefix", _fake_prefix_stager(seen, "char_helios_op"))
    stager.stage_variant("char_helios_op", dest_root=tmp_path)
    assert seen["prefix_revisions"] == [stager.STORIES_PIN]


def test_stage_variant_explicit_revision_overrides_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    def _no_sidecar(*a, **k):
        raise AssertionError("an explicit revision must skip sidecar resolution")

    monkeypatch.setattr(hub, "stage_hub_file", _no_sidecar)
    monkeypatch.setattr(hub, "stage_hub_prefix", _fake_prefix_stager(seen, "char_helios_op"))
    stager.stage_variant("char_helios_op", revision="cafe" * 10, dest_root=tmp_path)
    assert seen["prefix_revisions"] == ["cafe" * 10]


def test_stage_variant_resume_skip_resolves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Already-staged dest: NO revision resolution, NO Hub call (network-free resume).
    kept, yld = stager.expected_files("char_helios_op")
    dest = tmp_path / "char_helios_op" / "stories"
    dest.mkdir(parents=True)
    (dest / kept).write_text("{}\n")
    (dest / yld).write_text("{}")

    def _boom(*a, **k):
        raise AssertionError("resume skip must not touch the Hub")

    monkeypatch.setattr(hub, "stage_hub_file", _boom)
    monkeypatch.setattr(hub, "stage_hub_prefix", _boom)
    monkeypatch.setattr(hub, "retry_transient", _boom)
    out = stager.stage_variant("char_helios_op", dest_root=tmp_path)
    assert out == dest
