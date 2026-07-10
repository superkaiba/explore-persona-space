"""Tests for scripts/verify_reused_artifact_keys.py — the mechanized
realized-keys probe (task #1164; artifact-reuse.md check (c), incident #1073).

Every fixture bundle is built in ``tmp_path`` at test time (``torch.save`` /
``safetensors.torch.save_file`` / ``json.dump`` of tiny dicts, tensors <= 4
floats) — NO committed binary fixtures. The HF-mode tests monkeypatch
``huggingface_hub.hf_hub_download`` at the network boundary with a
signature-conformant fake (the real body of ``resolve_artifact`` executes —
the code-style one-production-body-test-per-seam rule).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load the probe as a module (it's a script, not a package member) — the same
# loader shape as tests/test_verify_plan.py.
_SCRIPT = REPO_ROOT / "scripts" / "verify_reused_artifact_keys.py"
_spec = importlib.util.spec_from_file_location("verify_reused_artifact_keys", _SCRIPT)
vrak = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_reused_artifact_keys"] = vrak
_spec.loader.exec_module(vrak)  # type: ignore[union-attr]


def _pt_bundle(tmp_path: Path, name: str = "bundle.pt", **save_kwargs) -> Path:
    """Canonical tiny fixture: dict of two tensors + primitive metadata."""
    p = tmp_path / name
    torch.save({"a": torch.ones(2), "b": torch.zeros(3), "meta": {"n": 1}}, p, **save_kwargs)
    return p


# ─── .pt / torch branch ─────────────────────────────────────────────────────


def test_pt_superset_passes(tmp_path, capsys):
    p = _pt_bundle(tmp_path)
    assert vrak.realized_keys(p) >= {"a", "b"}
    rc = vrak.main(["--artifact", str(p), "--keys", "a,b"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "superset" in out


def test_pt_missing_key_exits_1(tmp_path, capsys):
    p = _pt_bundle(tmp_path)
    rc = vrak.main(["--artifact", str(p), "--keys", "a,b,zz"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "MISSING" in out and "zz" in out


def test_pt_mmap_read_used(tmp_path):
    """Functional §8-assumption-1 verification: a zipfile-format bundle loads
    with mmap=True + weights_only=True and the keys come back correct on the
    lock-pinned torch."""
    p = _pt_bundle(tmp_path)
    assert vrak.realized_keys(p, fmt="pt", weights_only=True, allow_full_load=False) == {
        "a",
        "b",
        "meta",
    }


def test_pt_mmap_kwarg_passed(tmp_path, monkeypatch):
    """Monkeypatch-spy on torch.load: mmap=True is actually passed (directly
    tests acceptance criterion 2 rather than inferring it)."""
    p = _pt_bundle(tmp_path)
    seen: list[dict] = []
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    assert vrak.realized_keys(p) == {"a", "b", "meta"}
    assert len(seen) == 1
    assert seen[0].get("mmap") is True
    assert seen[0].get("weights_only") is True
    assert seen[0].get("map_location") == "cpu"


def test_legacy_pt_exits_2_without_allow_full_load(tmp_path, capsys):
    """Legacy non-zipfile serialization cannot mmap: fail LOUD (exit 2) with
    the hint naming --allow-full-load; with the explicit opt-in it reads."""
    p = tmp_path / "legacy.pt"
    torch.save({"a": torch.ones(2)}, p, _use_new_zipfile_serialization=False)
    rc = vrak.main(["--artifact", str(p), "--keys", "a"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "--allow-full-load" in out
    rc = vrak.main(["--artifact", str(p), "--keys", "a", "--allow-full-load"])
    assert rc == 0


def test_pt_non_dict_root_exits_2(tmp_path, capsys):
    p = tmp_path / "tensor_root.pt"
    torch.save(torch.ones(2), p)
    rc = vrak.main(["--artifact", str(p), "--keys", "a"])
    assert rc == 2
    assert "Tensor" in capsys.readouterr().out


# ─── .safetensors branch ────────────────────────────────────────────────────


def test_safetensors_keys_read(tmp_path):
    p = tmp_path / "bundle.safetensors"
    save_file({"x": torch.ones(2), "y": torch.zeros(2)}, str(p))
    assert vrak.realized_keys(p) == {"x", "y"}
    assert vrak.main(["--artifact", str(p), "--keys", "x,y"]) == 0
    assert vrak.main(["--artifact", str(p), "--keys", "x,zz"]) == 1


# ─── .json branch ───────────────────────────────────────────────────────────


def test_json_top_level_keys(tmp_path):
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps({"a": 1, "b": [2, 3]}))
    assert vrak.realized_keys(p) == {"a", "b"}
    lst = tmp_path / "list_root.json"
    lst.write_text(json.dumps([1, 2]))
    assert vrak.main(["--artifact", str(lst), "--keys", "a"]) == 2


# ─── key parsing + CLI contract ─────────────────────────────────────────────


def test_keys_file_parsing(tmp_path):
    kf = tmp_path / "keys.txt"
    kf.write_text("# consumer asserts\na\n\nb\n  c  \n")
    assert vrak.parse_keys_arg(None, kf) == {"a", "b", "c"}
    p = _pt_bundle(tmp_path)
    # --keys and --keys-file are mutually exclusive -> argparse error (exit 2).
    with pytest.raises(SystemExit) as exc:
        vrak.main(["--artifact", str(p), "--keys", "a", "--keys-file", str(kf)])
    assert exc.value.code == 2


def test_keys_whitespace_and_comma_split():
    assert vrak.parse_keys_arg("a, b  c", None) == {"a", "b", "c"}


def test_missing_artifact_exits_2(tmp_path, capsys):
    rc = vrak.main(["--artifact", str(tmp_path / "nope.pt"), "--keys", "a"])
    assert rc == 2
    assert "not found" in capsys.readouterr().out


def test_json_report_shape(tmp_path, capsys):
    p = _pt_bundle(tmp_path)
    rc = vrak.main(["--artifact", str(p), "--keys", "a,zz", "--json"])
    assert rc == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "MISSING"
    assert report["missing"] == ["zz"]
    assert report["n_realized"] == 3
    assert report["declared"] == ["a", "zz"]
    assert set(report["realized"]) == {"a", "b", "meta"}


def test_unknown_extension_exits_2(tmp_path, capsys):
    p = tmp_path / "bundle.foo"
    p.write_text("junk")
    rc = vrak.main(["--artifact", str(p), "--keys", "a"])
    assert rc == 2
    assert "unknown extension" in capsys.readouterr().out


# ─── HF mode (network boundary faked; real resolve_artifact body executes) ──


def _fake_hf_hub_download_factory(dest: Path, seen: list[dict]):
    """Signature-conformant fake of huggingface_hub.hf_hub_download for the
    kwargs the probe passes (repo_id/filename/repo_type/revision)."""

    def fake_hf_hub_download(
        repo_id: str,
        filename: str,
        *,
        repo_type: str | None = None,
        revision: str | None = None,
        **kwargs,
    ) -> str:
        seen.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "revision": revision,
            }
        )
        return str(dest)

    return fake_hf_hub_download


def test_hf_mode_body_executes(tmp_path, monkeypatch, capsys):
    """Body-executing seam test (code-style one-production-body-test rule):
    main() -> resolve_artifact() runs its REAL body — dotenv wrapper + the
    hf_hub_download call site with its exact kwargs — with only the network
    boundary faked, signature-conformant by construction."""
    import huggingface_hub

    p = _pt_bundle(tmp_path)
    seen: list[dict] = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download_factory(p, seen))
    rc = vrak.main(
        [
            "--hf-repo",
            "superkaiba1/explore-persona-space-data",
            "--hf-path",
            "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt",
            "--revision",
            "deadbeef",
            "--keys",
            "a,b,meta",
        ]
    )
    assert rc == 0
    assert "PASS" in capsys.readouterr().out
    assert seen == [
        {
            "repo_id": "superkaiba1/explore-persona-space-data",
            "filename": "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt",
            "repo_type": "dataset",
            "revision": "deadbeef",
        }
    ]


def test_hf_repo_without_hf_path_errors(tmp_path):
    with pytest.raises(SystemExit) as exc:
        vrak.main(["--hf-repo", "some/repo", "--keys", "a"])
    assert exc.value.code == 2
