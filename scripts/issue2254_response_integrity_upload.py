"""Persist and verify issue #2254 response-integrity subagent artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.issue2220_readwrite as rw2220  # noqa: E402
import scripts.issue2254_first_k_steering as fk  # noqa: E402
import scripts.issue2254_preimage as i2254  # noqa: E402

LOCAL_REL = Path("eval_results/issue_2254/response_integrity_matched_steering")
V3_REL = Path("codex_subagent_v3")
REMOTE_PREFIX = f"{i2254.HF_PREFIX}/response_integrity_matched_steering"


class UploadError(RuntimeError):
    """A fail-loud completeness or remote-verification failure."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    local_root = _REPO_ROOT / LOCAL_REL
    v3 = local_root / V3_REL
    required = [
        v3 / "inputs_manifest.json",
        v3 / "instrument_manifest.json",
        v3 / "analysis_protocol.json",
        v3 / "runner_manifest.json",
        v3 / "pilot/pass.json",
        v3 / "judge/completeness.json",
        v3 / "judge/per_item.jsonl",
        v3 / "reduce/matched_results.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise UploadError(f"required artifacts missing: {missing}")
    pilot = json.loads((v3 / "pilot/pass.json").read_text(encoding="utf-8"))
    completeness = json.loads(
        (v3 / "judge/completeness.json").read_text(encoding="utf-8")
    )
    if pilot.get("verdict") != "PASS":
        raise UploadError("v3 pilot did not pass")
    if completeness.get("fraction_valid") != 1.0 or completeness.get("pass") is not True:
        raise UploadError("v3 production completeness is not exact")

    with tempfile.TemporaryDirectory(prefix="issue2254-integrity-upload-", dir="/tmp") as tmp:
        temp_root = Path(tmp)
        pack = temp_root / "records_pack"
        n_shards = rw2220._pack_tree_to_jsonl_shards(
            local_root,
            pack,
            group="issue2254_response_integrity",
            pattern="*.json",
        )
        pack_manifest = json.loads((pack / "pack_manifest.json").read_text())
        if n_shards <= 0 or pack_manifest.get("n_files", 0) <= 0:
            raise UploadError("JSON artifact pack is empty")
        stage = temp_root / "stage"
        shutil.copytree(pack, stage / "records_pack")
        direct = {}
        for source in required:
            relative = source.relative_to(local_root)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            direct[str(relative)] = _sha256(source)
        upload_manifest = {
            "local_root": str(LOCAL_REL),
            "remote_prefix": REMOTE_PREFIX,
            "packed_json_records": pack_manifest["n_files"],
            "pack_shards": n_shards,
            "direct_sha256": direct,
        }
        (stage / "upload_manifest.json").write_text(
            json.dumps(upload_manifest, indent=2), encoding="utf-8"
        )
        i2254._upload_folder_to_hf(
            stage,
            REMOTE_PREFIX,
            allow=["*.json", "*.jsonl"],
        )
        expected = {
            f"{REMOTE_PREFIX}/{path.relative_to(stage)}"
            for path in stage.rglob("*")
            if path.is_file()
        }
    remote_paths = {entry.path for entry in fk._hub_tree(REMOTE_PREFIX, recursive=True)}
    missing_remote = sorted(expected - remote_paths)
    if missing_remote:
        raise UploadError(f"remote verification missing {missing_remote}")
    from huggingface_hub import HfApi

    revision = HfApi().repo_info(i2254.HF_DATA_REPO, repo_type="dataset").sha
    verification = {
        "verdict": "PASS",
        "hf_repo": i2254.HF_DATA_REPO,
        "hf_prefix": REMOTE_PREFIX,
        "hf_revision": revision,
        "expected_remote_files": len(expected),
        "packed_json_records": pack_manifest["n_files"],
        "pack_shards": n_shards,
    }
    path = v3 / "upload_verification.json"
    path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
