"""#1112 round 8 — r_B extraction rollout-text persistence + upload wiring.

Pins the upload-verification v1 blocker fix ``generation-discarded-undeclared``:

1. ``scripts/issue779_extract_rb.py`` persists the raw rollout TEXT (with
   judge-pairing custom_ids) under ``out_dir/raw_completions/`` via
   ``_dump_rollouts`` BEFORE any judge/reduce — REAL bodies, pure filesystem,
   no seams.
2. ``scripts/issue1112_dispatch.py`` ``phase_upload`` routes those files to
   ``{DATA_PREFIX}/raw_completions/rb_extraction/`` on the data repo (plan §10)
   and keeps them OUT of the generic ``rb/`` JSON bucket; ``phase_rb`` fails
   loud when the extractor produced a tensor but no rollout text (the regressed
   pre-fix shape). ``hub._upload`` / ``_run_subprocess`` are faked ONLY at the
   network / subprocess boundary, signature-conformant (autospec / mirrored
   def) per code-style.md "one production-body test per seam-stubbed function".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue779_extract_rb as rb  # noqa: E402
import issue1112_dispatch as d  # noqa: E402

from explore_persona_space.experiments import issue_1112 as C  # noqa: E402


def _rollouts() -> dict[str, dict[str, list[str]]]:
    """batch_judge-shaped {persona: {question: [completions]}} — 2x2x2 = 8."""
    return {
        "sycophancy_pos_p0": {
            "q one?": ["comp a", "comp b"],
            "q two?": ["comp c", "comp d"],
        },
        "sycophancy_pos_p1": {
            "q one?": ["comp e", "comp f"],
            "q two?": ["comp g", "comp h"],
        },
    }


# ── extractor: rollout-record enumeration + dump shape ────────────────────────


def test_iter_rollout_records_global_question_index_never_resets():
    """The judge-pairing invariant: persona p1's FIRST question continues the
    GLOBAL question index (2), never resets to 0 — the exact enumeration
    C.judge_rollouts_n5 / batch_judge use for custom_ids."""
    recs = list(rb._iter_rollout_records(_rollouts()))
    assert len(recs) == 8
    by_persona: dict[str, list[tuple[int, int, str]]] = {}
    for persona, q_idx, _q, ci, _comp, cid in recs:
        by_persona.setdefault(persona, []).append((q_idx, ci, cid))
    assert [q for q, _c, _i in by_persona["sycophancy_pos_p0"]] == [0, 0, 1, 1]
    assert [q for q, _c, _i in by_persona["sycophancy_pos_p1"]] == [2, 2, 3, 3]
    assert by_persona["sycophancy_pos_p1"][0][2] == "sycophancy_pos_p1__00002__00"


def test_dump_rollouts_single_file_shape(tmp_path):
    """One file per arm below the size cap; records carry the full pairing
    schema (persona / global question_index / completion_index / custom_id /
    completion) + sampling + reproducibility metadata."""
    paths = rb._dump_rollouts("sycophancy", "pos", _rollouts(), tmp_path, {"n": 2})
    assert [p.name for p in paths] == ["rollouts_sycophancy_pos.json"]
    assert paths[0].parent == tmp_path / "raw_completions"
    payload = json.loads(paths[0].read_text())
    assert payload["trait"] == "sycophancy" and payload["arm"] == "pos"
    assert payload["n_parts"] == 1
    assert payload["n_rollouts_total"] == 8 == len(payload["rollouts"])
    rec = payload["rollouts"][0]
    assert set(rec) == {
        "persona",
        "question_index",
        "question",
        "completion_index",
        "custom_id",
        "completion",
    }
    assert rec["custom_id"] == "sycophancy_pos_p0__00000__00"
    assert rec["completion"] == "comp a"
    assert payload["sampling"] == {"n": 2}
    assert isinstance(payload["metadata"], dict) and payload["metadata"].get("git_commit")


def test_dump_rollouts_chunks_on_max_bytes_preserving_order(tmp_path):
    """Over the byte cap the dump shards at record boundaries (.partNN.json),
    the union preserving the full enumeration order (no record lost/dup)."""
    paths = rb._dump_rollouts("sycophancy", "neg", _rollouts(), tmp_path, {"n": 2}, max_bytes=300)
    assert len(paths) > 1
    assert all(p.name.startswith("rollouts_sycophancy_neg.part") for p in paths)
    all_recs: list[dict] = []
    for p in paths:
        payload = json.loads(p.read_text())
        assert payload["n_parts"] == len(paths)
        all_recs += payload["rollouts"]
    assert [r["custom_id"] for r in all_recs] == [
        cid for *_rest, cid in rb._iter_rollout_records(_rollouts())
    ]


# ── dispatcher: phase_rb fail-loud contract + phase_upload routing ────────────


def _mk_cfg(tmp_path, cells=(), upload=True) -> d.Cfg:
    return d.Cfg(smoke=False, cells=tuple(cells), out_root=tmp_path, upload=upload)


def test_phase_rb_fails_loud_when_rollout_text_missing(tmp_path, monkeypatch):
    """phase_rb (REAL body) refuses to proceed when the extractor subprocess
    produced the tensor but NO rollout text — the regressed pre-fix shape."""
    import torch

    rb_dir = tmp_path / "rb"

    def _fake_run(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
        # subprocess boundary fake, mirrors d._run_subprocess's signature:
        # simulates the PRE-FIX extractor (tensor written, no raw_completions/).
        (rb_dir / "r_b").mkdir(parents=True, exist_ok=True)
        torch.save(
            {"r_b": torch.zeros(C.N_LAYERS, C.HIDDEN), "counts": {}},
            rb_dir / "r_b" / "sycophancy.pt",
        )

    monkeypatch.setattr(d, "_run_subprocess", _fake_run)
    monkeypatch.setattr(d, "_seed_rb_artifacts_from_registry", lambda cache_path: {})
    with pytest.raises(FileNotFoundError, match="rollout text"):
        d.phase_rb(_mk_cfg(tmp_path, cells=("s3_fullft_neg",), upload=False))
    assert not (rb_dir / "rb_done.json").exists()  # no done-sentinel on the failed contract


def test_phase_rb_records_rollout_files_and_normalizes_tensor(tmp_path, monkeypatch):
    """phase_rb (REAL body) with a contract-conforming extractor: normalizes the
    tensor to rb_sycophancy.pt AND records the persisted rollout files."""
    import torch

    rb_dir = tmp_path / "rb"

    def _fake_run(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
        (rb_dir / "r_b").mkdir(parents=True, exist_ok=True)
        torch.save(
            {"r_b": torch.zeros(C.N_LAYERS, C.HIDDEN), "counts": {"trait": "sycophancy"}},
            rb_dir / "r_b" / "sycophancy.pt",
        )
        rc = rb_dir / "raw_completions"
        rc.mkdir(parents=True, exist_ok=True)
        (rc / "rollouts_sycophancy_pos.json").write_text("{}")
        (rc / "rollouts_sycophancy_neg.json").write_text("{}")

    monkeypatch.setattr(d, "_run_subprocess", _fake_run)
    monkeypatch.setattr(d, "_seed_rb_artifacts_from_registry", lambda cache_path: {})
    rec = d.phase_rb(_mk_cfg(tmp_path, cells=("s3_fullft_neg",), upload=False))
    assert rec["rollout_files"] == [
        "rollouts_sycophancy_neg.json",
        "rollouts_sycophancy_pos.json",
    ]
    assert (rb_dir / "rb_sycophancy.pt").exists()
    assert (rb_dir / "rb_done.json").exists()


def test_phase_upload_routes_rollout_text_to_rb_extraction(tmp_path):
    """phase_upload (REAL body; hub._upload autospec'd at the network boundary)
    lands the rollout dumps under raw_completions/rb_extraction/ (plan §10),
    keeps the tensor + judge sidecar in their existing buckets, and does NOT
    double-upload the rollout text into the generic rb/ bucket."""
    rb_dir = tmp_path / "rb"
    (rb_dir / "raw_completions").mkdir(parents=True)
    (rb_dir / "raw_completions" / "rollouts_sycophancy_pos.json").write_text("{}")
    (rb_dir / "raw_completions" / "rollouts_sycophancy_neg.json").write_text("{}")
    (rb_dir / "rb_sycophancy.pt").write_bytes(b"pt")
    (rb_dir / "judge_sycophancy_pos.json").write_text("{}")
    with mock.patch.object(d.hub, "_upload", autospec=True, return_value="https://hf.co/x"):
        uploaded = d.phase_upload(_mk_cfg(tmp_path))
    dests = set(uploaded)
    assert f"{C.DATA_PREFIX}/raw_completions/rb_extraction/rollouts_sycophancy_pos.json" in dests
    assert f"{C.DATA_PREFIX}/raw_completions/rb_extraction/rollouts_sycophancy_neg.json" in dests
    assert f"{C.DATA_PREFIX}/analysis_tensors/rb/rb_sycophancy.pt" in dests
    assert f"{C.DATA_PREFIX}/rb/judge_sycophancy_pos.json" in dests
    assert f"{C.DATA_PREFIX}/rb/rollouts_sycophancy_pos.json" not in dests
    assert f"{C.DATA_PREFIX}/rb/rollouts_sycophancy_neg.json" not in dests
