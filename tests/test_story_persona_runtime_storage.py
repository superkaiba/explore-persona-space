"""Exercise the storage boundary without creating paths on the operator's root disk."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.story_persona_runtime_storage import verify_runtime_paths

WORKSPACE = {"target": "/workspace", "source": "mfs#test", "fstype": "fuse"}
LOCAL = {"target": "/", "source": "overlay", "fstype": "overlay"}


@pytest.fixture
def rig(monkeypatch):
    created = []
    monkeypatch.setattr(Path, "mkdir", lambda path, **kwargs: created.append(str(path)))
    monkeypatch.setattr(Path, "resolve", lambda path: path)
    env = {
        "EPS_STORY_PERSONA_OVERLAY_ROOT": "/root/eps-kimi-runtime",
        "HF_HOME": "/workspace/.cache/huggingface",
        "HF_HUB_CACHE": "/workspace/.cache/huggingface/hub",
        "UV_CACHE_DIR": "/root/eps-kimi-runtime/uv",
        "UV_PROJECT_ENVIRONMENT": "/root/eps-kimi-runtime/base",
        "TMPDIR": "/root/eps-kimi-runtime/tmp",
    }

    def mount(path):
        return WORKSPACE if Path(path).is_relative_to("/workspace") else LOCAL

    return SimpleNamespace(
        env=env,
        contract={"model_key": "kimi", "api_container_disk_gb": 150},
        mount=mount,
        usage=lambda path: SimpleNamespace(free=60 * 10**9),
        created=created,
    )


def verify(rig):
    return verify_runtime_paths(
        rig.env, rig.contract, WORKSPACE, mount_reader=rig.mount, disk_usage=rig.usage
    )


def test_overlay_keeps_hf_weights_on_volume_and_runtime_on_local_disk(rig):
    paths = verify(rig)
    assert paths["HF_HUB_CACHE"]["mount"] == WORKSPACE
    assert paths["UV_CACHE_DIR"]["mount"] == LOCAL
    assert paths["UV_PROJECT_ENVIRONMENT"]["mount"] == LOCAL
    assert len(rig.created) == 5


@pytest.mark.parametrize("key", ["HF_HOME", "HF_HUB_CACHE"])
def test_model_weight_cache_cannot_move_to_container(rig, key):
    rig.env[key] = "/root/eps-kimi-runtime/weights"
    with pytest.raises(ValueError, match="workspace volume"):
        verify(rig)


@pytest.mark.parametrize("key", ["UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT", "TMPDIR"])
def test_runtime_cannot_silently_fall_back_to_network_volume(rig, key):
    rig.env[key] = "/workspace/.cache/uv"
    with pytest.raises(ValueError, match="approved local runtime"):
        verify(rig)


@pytest.mark.parametrize(
    "defect", ["wrong_model", "arbitrary_root", "small_disk", "disk_full", "network_root"]
)
def test_refuse_unapproved_or_inadequate_local_storage_before_creating_paths(rig, defect):
    if defect == "wrong_model":
        rig.contract["model_key"] = "deepseek"
    elif defect == "arbitrary_root":
        rig.env["EPS_STORY_PERSONA_OVERLAY_ROOT"] = "/tmp/other"
    elif defect == "small_disk":
        rig.contract["api_container_disk_gb"] = 50
    elif defect == "disk_full":
        rig.usage = lambda path: SimpleNamespace(free=49 * 10**9)
    else:
        rig.mount = lambda path: WORKSPACE
    with pytest.raises(ValueError):
        verify(rig)
    assert not rig.created


def test_symlink_escape_from_approved_runtime_is_rejected(rig, monkeypatch):
    def resolve(path):
        return Path("/workspace/redirect") if path.name == "base" else path

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="approved local runtime"):
        verify(rig)


def test_legacy_storage_paths_still_require_workspace(rig):
    del rig.env["EPS_STORY_PERSONA_OVERLAY_ROOT"]
    for key in ("UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT", "TMPDIR"):
        rig.env[key] = "/workspace/legacy/" + key.lower()
    assert all(value["mount"] == WORKSPACE for value in verify(rig).values())
    rig.env["TMPDIR"] = "/tmp"
    with pytest.raises(ValueError, match="workspace volume"):
        verify(rig)
