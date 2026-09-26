"""Verify the approved Kimi local runtime without moving model/result storage."""

from pathlib import Path
import shutil


OVERLAY_ROOT = Path("/root/eps-kimi-runtime")
LOCAL_KEYS = {"UV_CACHE_DIR": "uv", "UV_PROJECT_ENVIRONMENT": "base", "TMPDIR": "tmp"}


def verify_runtime_paths(
    env, contract, workspace_mount, *, mount_reader, disk_usage=shutil.disk_usage
):
    """Allow only the named local runtime; retain exact volume checks for HF weights."""
    overlay = env.get("EPS_STORY_PERSONA_OVERLAY_ROOT", "")
    local_mount = None
    if overlay:
        if overlay != str(OVERLAY_ROOT) or contract["model_key"] != "kimi":
            raise ValueError("Local runtime is restricted to the reviewed Kimi overlay")
        if contract["api_container_disk_gb"] < 150:
            raise ValueError("Kimi overlay requires the provisioned150GB container disk")
        if OVERLAY_ROOT.resolve() != OVERLAY_ROOT:
            raise ValueError("Kimi runtime root must not redirect through a symlink")
        local_mount = mount_reader(OVERLAY_ROOT)
        root_mount = mount_reader(Path("/root"))
        if (local_mount["target"], local_mount["source"]) != (
            root_mount["target"],
            root_mount["source"],
        ) or local_mount["fstype"] not in {"overlay", "ext4", "xfs", "btrfs"}:
            raise ValueError("Kimi runtime must use the container's local root filesystem")
        if (local_mount["target"], local_mount["source"]) == (
            workspace_mount["target"],
            workspace_mount["source"],
        ):
            raise ValueError("Kimi local runtime unexpectedly resolves to the workspace mount")
        if disk_usage(OVERLAY_ROOT).free < 50 * 10**9:
            raise ValueError("Kimi local runtime needs50GB remaining before runtime installation")
    paths = {}
    for key in ("HF_HOME", "HF_HUB_CACHE", *LOCAL_KEYS):
        path = Path(env[key])
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve()
        found = mount_reader(resolved)
        if overlay and key in LOCAL_KEYS:
            expected = OVERLAY_ROOT / LOCAL_KEYS[key]
            if resolved != expected or (found["target"], found["source"]) != (
                local_mount["target"],
                local_mount["source"],
            ):
                raise ValueError(f"{key} is outside the approved local runtime")
        elif not resolved.is_relative_to(Path("/workspace").resolve()) or (
            found["target"],
            found["source"],
        ) != (workspace_mount["target"], workspace_mount["source"]):
            raise ValueError(f"{key} is outside the verified workspace volume")
        paths[key] = {"resolved": str(resolved), "mount": found}
    return paths
