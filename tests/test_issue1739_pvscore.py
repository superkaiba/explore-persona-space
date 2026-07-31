"""pvscore dispatcher: publish_pvcap against the REAL staged capture-store shape.

The attempt-2 crash (20:46:30Z, `FATAL: per-behavior capture store
.../pvsynth_capture_store/evil missing`) happened because `publish_pvcap`
resolved the mirrored capture root by `find -name _capture_manifest.json
-print -quit`, and the REAL layout has NO root-level manifest: the only three
manifests live INSIDE the per-behavior subtrees, so `find` returned `evil/` and
the symlink landed one level too deep.

The fixture that "verified" the old code invented a root-level manifest beside
EMPTY behavior dirs — a layout that does not exist on the data repo. These
tests pin the MEASURED shape instead (519 files: capture_store/ holds nothing
but evil/ hallucination/ sycophancy/, one manifest inside each), so the
tiny-fixture-masks-shape-bugs class cannot recur here.

No network, no HF: the staged tree is a verbatim prefix mirror, so its shape is
fully reproducible locally.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DISPATCH = REPO_ROOT / "scripts" / "issue1739_pvscore_dispatch.sh"
PREFIX = "issue1739_ctxmap/pvsynth/capture_store"
BEHAVIORS = ("evil", "hallucination", "sycophancy")


def _real_shape_mirror(root: Path) -> Path:
    """The MEASURED staged layout: no root-level manifest, one per behavior."""
    capture = root / PREFIX
    for b in BEHAVIORS:
        d = capture / b
        d.mkdir(parents=True)
        (d / "_capture_manifest.json").write_text("{}", encoding="utf-8")
        (d / "context_end_L00_shard00.npy").write_bytes(b"\x00")
    assert not list(capture.glob("_capture_manifest.json")), "root manifest is not the real shape"
    return capture


def _run_publish(mirror: Path, store_root: Path) -> subprocess.CompletedProcess[str]:
    """Run the SHIPPED publish_pvcap function, extracted from the dispatcher."""
    body = subprocess.run(
        ["sed", "-n", "/^publish_pvcap()/,/^}/p", str(DISPATCH)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "publish_pvcap()" in body, "failed to extract publish_pvcap from the dispatcher"
    return subprocess.run(
        ["bash", "-c", f"{body}\npublish_pvcap"],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PVCAP_MIRROR": str(mirror),
            "PVCAP_PREFIX": PREFIX,
            "STORE_ROOT": str(store_root),
            "BEHAVIORS": " ".join(BEHAVIORS),
        },
    )


def test_publish_pvcap_points_at_capture_root_not_a_behavior_dir(tmp_path):
    """Fails pre-fix: find -print -quit returned evil/, so <dest>/evil vanished."""
    mirror = tmp_path / "mirror"
    capture = _real_shape_mirror(mirror)
    store = tmp_path / "store"
    store.mkdir()

    proc = _run_publish(mirror, store)
    assert proc.returncode == 0, f"publish_pvcap failed:\n{proc.stdout}\n{proc.stderr}"

    dest = store / "pvsynth_capture_store"
    assert dest.is_symlink()
    assert dest.resolve() == capture.resolve(), (
        f"symlink points at {dest.resolve()}, not the capture root {capture.resolve()}"
    )
    # the invariant the consumer actually needs
    for b in BEHAVIORS:
        assert (dest / b / "_capture_manifest.json").is_file(), (
            f"{b} unreadable through the symlink"
        )


def test_publish_pvcap_resolves_for_the_scorer(tmp_path):
    """The scorer's OWN resolver must open every behavior through the symlink."""
    pvarms = pytest.importorskip("importlib.util")
    spec = pvarms.spec_from_file_location(
        "_pvarms_t", REPO_ROOT / "scripts" / "issue1739_pvsynth_arms.py"
    )
    mod = pvarms.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mirror = tmp_path / "mirror"
    _real_shape_mirror(mirror)
    store = tmp_path / "store"
    store.mkdir()
    assert _run_publish(mirror, store).returncode == 0

    import argparse

    for b in BEHAVIORS:
        args = argparse.Namespace(pvsynth_store=None, pvsynth_store_root=None, store_root=store)
        resolved = mod._resolve_pvsynth_store(args, b)
        assert (resolved / mod.CAPTURE_MANIFEST_NAME).is_file(), f"scorer cannot open {b}"


def test_publish_pvcap_fails_loud_on_a_missing_mirror_root(tmp_path):
    """A staging-layout change must fail loud, never publish a wrong symlink."""
    store = tmp_path / "store"
    store.mkdir()
    proc = _run_publish(tmp_path / "empty_mirror", store)
    assert proc.returncode != 0
    assert "mirrored capture root" in proc.stderr
    assert not (store / "pvsynth_capture_store").exists()
