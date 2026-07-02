"""Regression test for the issue #779 raw-completions-upload BLOCKER fix.

Pins the permanent invariant installed post-pivot round-1 (reconciler v2
BLOCKER #1 ``raw-completions-upload-prefix-missing``): the collect dispatcher's
rollout TEXT MUST land under the canonical ``issue779_monitoring/raw_completions/``
prefix — NOT only under ``analysis_tensors/`` — and the upload MUST be
mechanically verified (every produced ``(trait, condition)`` has a file at the
canonical prefix) before ``phase("done")`` (plan v5 §10 row (c)).

Pre-fix: ``_upload_collect`` bulk-uploaded the whole out_dir (cell JSONs with
rollout text included) to ``analysis_tensors/`` and there was NO
``raw_completions/`` upload path anywhere — this test would FAIL (no upload with
a ``raw_completions/`` ``path_in_repo`` ever fired). Post-fix BOTH prefixes fire.

Pure-CPU, no model / no network: ``HfApi`` and ``list_repo_files`` are
monkeypatched to RECORD the ``path_in_repo`` of every ``upload_folder`` call and
to return a listing reflecting the recorded uploads, so the fail-loud
``list_repo_files`` verification exercises its real success path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue779_collect as X  # noqa: E402


class _FakeHfApi:
    """Records each upload_folder call's path_in_repo + the files it would land.

    Honors ``ignore_patterns`` via the REAL ``huggingface_hub`` filter so a test
    exercises the actual exclusion (top-level + nested judge sidecars), not a
    fake that ignores the patterns.
    """

    def __init__(self, sink: dict) -> None:
        self._sink = sink

    def upload_folder(
        self, *, folder_path, path_in_repo, repo_id, repo_type, ignore_patterns=None, **_kw
    ):
        from huggingface_hub.utils import filter_repo_objects

        rels = [
            p.relative_to(folder_path).as_posix()
            for p in sorted(Path(folder_path).rglob("*"))
            if p.is_file()
        ]
        if ignore_patterns:
            rels = list(filter_repo_objects(rels, ignore_patterns=ignore_patterns))
        landed = [f"{path_in_repo}/{rel}" for rel in rels]
        self._sink["uploads"].append(path_in_repo)
        self._sink["files"].update(landed)
        # Record the RAW BYTES of each landed file so a content-hygiene test can
        # assert what actually made it into the payload (not just the path list).
        for rel in rels:
            self._sink["content"][f"{path_in_repo}/{rel}"] = (Path(folder_path) / rel).read_bytes()


def _write_pass_a_cell(pass_a_dir: Path, trait: str, cond_id: str, seed: int = 42) -> None:
    """Write a Pass-A cell JSON shaped like production (raw ``response`` + the H2b
    ``peak_token.token_str`` instrumentation + the ``pooled`` scalars) so a
    content-hygiene test exercises the real strip surface, not a stub."""
    pass_a_dir.mkdir(parents=True, exist_ok=True)
    cell = {
        "trait": trait,
        "cond_id": cond_id,
        "mode": "system",
        "n_shot": 0,
        "n_questions": 1,
        "n_rollouts": 2,
        "rollout_seed": seed,
        "rollouts": [
            {
                "qi": 0,
                "ri": 0,
                "response": f"{trait} rollout text A",
                "n_resp": 4,
                "pooled": {"mean": [0.1], "max": [0.3], "topk": [0.2], "last": [0.15]},
                "peak_token": {"14": {"token_index": 2, "token_str": " secret", "proj": 0.3}},
            },
            {
                "qi": 0,
                "ri": 1,
                "response": f"{trait} rollout text B",
                "n_resp": 5,
                "pooled": {"mean": [0.2], "max": [0.4], "topk": [0.3], "last": [0.25]},
                "peak_token": {"14": {"token_index": 1, "token_str": " leak", "proj": 0.4}},
            },
        ],
        "judge_scores": {},
        "oracle_proj": {},
    }
    (pass_a_dir / f"{trait}__{cond_id}.json").write_text(json.dumps(cell))


def _write_pass_b_bundle(out_dir: Path, prompts: list[str]) -> Path:
    """Write a Pass-B tensor bundle carrying the raw ``prompts`` list alongside
    tensors — the exact shape ``_sanitize_for_analysis_tensors`` must strip."""
    import torch

    pass_b = out_dir / "pass_b"
    pass_b.mkdir(parents=True, exist_ok=True)
    path = pass_b / "train_context_vectors.pt"
    torch.save(
        {
            "cx_last": torch.zeros(len(prompts), 2, 3),
            "cx_mean": torch.zeros(len(prompts), 2, 3),
            "v_x": torch.zeros(len(prompts), 2, 3),
            "prompts": prompts,
            "layers": [0, 1],
            "source": "lmsys",
        },
        path,
    )
    return path


def _patch_hub(monkeypatch):
    sink: dict = {"uploads": [], "files": set(), "content": {}}
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _FakeHfApi(sink))
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda *a, **k: sorted(sink["files"]),
    )
    return sink


def test_split_raw_completions_writes_flat_per_cell_layout(tmp_path):
    """_split_raw_completions produces one {trait}_{cond}_seed42.json per Pass-A
    cell, shaped {trait, condition, seed, rollouts:[{qi,ri,response}]}."""
    out_dir = tmp_path / "data"
    _write_pass_a_cell(out_dir / "pass_a", "evil", "sys0")
    _write_pass_a_cell(out_dir / "pass_a", "sycophancy", "shot5")
    staging = tmp_path / "staging"
    written = X._split_raw_completions(out_dir, staging)
    # Returns (trait, cond_id, filename) per cell; the verifier consumes the exact
    # written filename (no hardcoded seed).
    assert {(t, c) for t, c, _f in written} == {("evil", "sys0"), ("sycophancy", "shot5")}
    assert ("evil", "sys0", "evil_sys0_seed42.json") in written
    f = staging / "evil_sys0_seed42.json"
    assert f.exists()
    payload = json.loads(f.read_text())
    assert payload["trait"] == "evil"
    assert payload["condition"] == "sys0"
    assert payload["seed"] == 42
    assert [r["response"] for r in payload["rollouts"]] == [
        "evil rollout text A",
        "evil rollout text B",
    ]


def test_both_prefixes_fire_raw_completions_and_analysis_tensors(tmp_path, monkeypatch):
    """The upload phase fires BOTH an analysis_tensors/ upload AND a
    raw_completions/ upload (the post-fix invariant). Pre-fix only the
    analysis_tensors/ prefix ever appeared."""
    sink = _patch_hub(monkeypatch)
    out_dir = tmp_path / "data"
    _write_pass_a_cell(out_dir / "pass_a", "evil", "sys0")

    X._upload_collect(out_dir, smoke=False)
    X._upload_raw_completions(out_dir, smoke=False)

    prefixes = sink["uploads"]
    assert f"{X.C.HF_PREFIX}/analysis_tensors" in prefixes, prefixes
    assert f"{X.C.HF_PREFIX}/raw_completions" in prefixes, prefixes
    # The canonical raw-completions file for the produced cell landed under the
    # mandated flat layout.
    assert f"{X.C.HF_PREFIX}/raw_completions/evil_sys0_seed42.json" in sink["files"], sorted(
        sink["files"]
    )


def test_raw_completions_verify_uses_actual_seed_not_hardcoded_42(tmp_path, monkeypatch):
    """HOLISTIC-HARDENING regression (upload-prefix bug class): a cell whose
    ``rollout_seed`` is NOT 42 still verifies — the writer names the file with the
    cell's actual seed and the verifier consumes that exact returned filename, so
    a rollout_seed change can never desync writer↔verifier. Pre-hardening the
    verifier reconstructed the expected name from a hardcoded ``seed in (42,)``,
    which would spuriously fail-loud on any non-42 seed even when the file landed
    correctly."""
    sink = _patch_hub(monkeypatch)
    out_dir = tmp_path / "data"
    _write_pass_a_cell(out_dir / "pass_a", "evil", "sys0", seed=137)
    # Must NOT raise: the file lands as ..._seed137.json and the verifier checks
    # that exact name (not _seed42).
    X._upload_raw_completions(out_dir, smoke=False)
    assert f"{X.C.HF_PREFIX}/raw_completions/evil_sys0_seed137.json" in sink["files"], sorted(
        sink["files"]
    )
    # The stale-hardcode name is NOT what was produced.
    assert f"{X.C.HF_PREFIX}/raw_completions/evil_sys0_seed42.json" not in sink["files"]


def _write_judge_sidecar(pass_a_dir: Path, trait: str, cond_id: str) -> None:
    """Drop the judge raw sidecar `_judge_cell` writes into the SAME pass_a/ dir
    (save_raw = out_dir / f"judge_{cell_id}.json"). Its keys are the batch-judge
    raw shape (per_persona / all_scores / judge_model) — NO `trait` key."""
    pass_a_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "per_persona": {f"{trait}__{cond_id}": {"q000": [{"score": 80}]}},
        "all_scores": [80],
        "judge_model": "claude-sonnet-4-5-20250929",
    }
    (pass_a_dir / f"judge_{trait}__{cond_id}.json").write_text(json.dumps(sidecar))


def test_upload_collect_excludes_top_level_and_nested_judge_sidecars(tmp_path, monkeypatch):
    """HOLISTIC-HARDENING regression (raw-text-in-analysis_tensors bug class): the
    analysis_tensors/ upload MUST exclude the raw judge-model TEXT sidecars — BOTH
    the r_B-extraction TOP-LEVEL ``out_dir/judge_{trait}_{arm}.json`` AND the
    collect NESTED ``pass_a/judge_{cell}.json`` (collect + extract share out_dir).
    Pre-hardening the single ``**/judge_*.json`` glob did NOT match a top-level
    file (HF fnmatch), so the extraction sidecars' raw judge text leaked into
    analysis_tensors/. The SANITIZED cell JSON + benign analysis JSON MUST
    remain."""
    sink = _patch_hub(monkeypatch)
    out_dir = tmp_path / "data"
    _write_pass_a_cell(out_dir / "pass_a", "evil", "sys0")
    # A collect NESTED judge sidecar (pass_a/judge_*.json).
    _write_judge_sidecar(out_dir / "pass_a", "evil", "sys0")
    # An r_B-extraction TOP-LEVEL judge sidecar (out_dir/judge_{trait}_{arm}.json).
    (out_dir / "judge_evil_pos.json").write_text(
        json.dumps({"all_scores": [90], "per_persona": {}})
    )
    # A benign analysis artifact that MUST survive.
    (out_dir / "step0").mkdir(parents=True, exist_ok=True)
    (out_dir / "step0" / "step0_oracle.json").write_text(json.dumps({"evil": {}}))

    X._upload_collect(out_dir, smoke=False)

    landed = sink["files"]
    pref = f"{X.C.HF_PREFIX}/analysis_tensors"
    # BOTH sidecars excluded.
    assert f"{pref}/judge_evil_pos.json" not in landed, sorted(landed)
    assert f"{pref}/pass_a/judge_evil__sys0.json" not in landed, sorted(landed)
    # The SANITIZED cell JSON (scalars only) + benign analysis JSON survive.
    assert f"{pref}/pass_a/evil__sys0.json" in landed, sorted(landed)
    assert f"{pref}/step0/step0_oracle.json" in landed, sorted(landed)


def test_analysis_tensors_upload_strips_all_raw_text(tmp_path, monkeypatch):
    """BLOCKER regression (reconciler v5 `analysis-tensors-prefix-contains-raw-text`):
    NO raw text (`response` / `prompts` / `token_str`) may reach analysis_tensors/.

    Pre-fix `_upload_collect` bulk-uploaded out_dir verbatim, so the Pass-A cell
    JSON's `rollouts[].response` + `peak_token.token_str` AND the Pass-B bundle's
    `prompts` list leaked into analysis_tensors/. Post-fix the sanitizer strips
    every text field (canonical rollout text stays under raw_completions/) and the
    in-`_upload_collect` `_assert_no_raw_text_under` gate fails loud on any leak.

    Asserts on the ACTUAL uploaded bytes recorded by the fake Hub (not just the
    path list): the analysis_tensors/ payload for the cell JSON carries the scalars
    Stage-1 reads (`qi`, `ri`, `pooled`, `peak_token.token_index`/`proj`) but NOT
    `response` / `token_str`, and the Pass-B bundle carries tensors but NOT
    `prompts`."""
    sink = _patch_hub(monkeypatch)
    out_dir = tmp_path / "data"
    _write_pass_a_cell(out_dir / "pass_a", "evil", "sys0")
    _write_pass_b_bundle(out_dir, ["raw prompt one", "raw prompt two"])

    X._upload_collect(out_dir, smoke=False)  # must NOT raise the hygiene gate

    pref = f"{X.C.HF_PREFIX}/analysis_tensors"
    # --- Pass-A cell JSON: scalars kept, ALL raw text stripped. ---
    cell_key = f"{pref}/pass_a/evil__sys0.json"
    assert cell_key in sink["content"], sorted(sink["content"])
    cell = json.loads(sink["content"][cell_key].decode())
    for rec in cell["rollouts"]:
        assert "response" not in rec, rec
        assert "qi" in rec and "ri" in rec and "pooled" in rec, rec  # scalars kept
        for entry in rec.get("peak_token", {}).values():
            assert "token_str" not in entry, entry  # decoded token text stripped
            assert "token_index" in entry and "proj" in entry, entry  # numeric kept
    # --- Pass-B tensor bundle: tensors kept, `prompts` stripped. ---
    import io

    import torch

    bundle_key = f"{pref}/pass_b/train_context_vectors.pt"
    assert bundle_key in sink["content"], sorted(sink["content"])
    blob = torch.load(io.BytesIO(sink["content"][bundle_key]), weights_only=False)
    assert "prompts" not in blob, list(blob)
    assert "v_x" in blob and "cx_last" in blob, list(blob)  # tensors kept
    # --- Nothing raw-text-shaped anywhere under the whole analysis_tensors/ payload. ---
    for key, raw in sink["content"].items():
        if not key.startswith(pref):
            continue
        if key.endswith(".json"):
            assert b'"response"' not in raw, key
            assert b'"token_str"' not in raw, key
            assert b'"prompts"' not in raw, key


def test_assert_no_raw_text_under_fails_loud_on_leaked_response(tmp_path):
    """The content-hygiene gate itself fails loud: if the sanitizer regressed and a
    `response` (or `prompts` / `token_str`) survived in the staging tree,
    `_assert_no_raw_text_under` raises with the blocker id. This pins the gate so a
    future refactor that strips the sanitizer can never silently ship raw text."""
    staging = tmp_path / "staging" / "pass_a"
    staging.mkdir(parents=True)
    (staging / "evil__sys0.json").write_text(
        json.dumps({"trait": "evil", "rollouts": [{"qi": 0, "ri": 0, "response": "LEAK"}]})
    )
    with pytest.raises(RuntimeError, match="analysis-tensors-prefix-contains-raw-text"):
        X._assert_no_raw_text_under(tmp_path / "staging")


def test_split_raw_completions_skips_judge_sidecar_no_keyerror(tmp_path):
    """PRODUCTION-layout regression (BLOCKER
    raw-completions-split-glob-crashes-on-judge-sidecar, code-review v4): a real
    Pass-A dir holds BOTH the cell JSON `{trait}__{cond}.json` AND the judge
    sidecar `judge_{trait}__{cond}.json` written by `_judge_cell`. Pre-fix the
    `glob("*.json")` opened the sidecar and `cell["trait"]` raised KeyError,
    crashing the upload phase before phase(done). Post-fix the `judge_` prefix is
    skipped: exactly ONE raw-completions file per CELL, no crash, no
    `judge_*_seed42.json` emitted."""
    out_dir = tmp_path / "data"
    pass_a = out_dir / "pass_a"
    _write_pass_a_cell(pass_a, "evil", "sys0")
    _write_judge_sidecar(pass_a, "evil", "sys0")  # the crash trigger
    staging = tmp_path / "staging"
    written = X._split_raw_completions(out_dir, staging)  # must NOT raise
    assert written == [("evil", "sys0", "evil_sys0_seed42.json")], written
    # One raw-completions file per CELL (not per json); no judge sidecar copied.
    produced = sorted(p.name for p in staging.glob("*.json"))
    assert produced == ["evil_sys0_seed42.json"], produced
    assert not (staging / "judge_evil__sys0_seed42.json").exists()


def test_split_raw_completions_skips_cell_missing_required_keys(tmp_path):
    """Defense-in-depth: a stray `.json` under pass_a/ lacking the cell keys
    (trait/cond_id/rollouts) is skipped, never KeyError-crashes the upload."""
    out_dir = tmp_path / "data"
    pass_a = out_dir / "pass_a"
    _write_pass_a_cell(pass_a, "evil", "sys0")
    (pass_a / "stray.json").write_text(json.dumps({"unrelated": True}))
    written = X._split_raw_completions(out_dir, tmp_path / "staging")
    assert written == [("evil", "sys0", "evil_sys0_seed42.json")], written


def test_raw_completions_upload_verification_fails_loud_on_empty(tmp_path, monkeypatch):
    """Fail-loud when Pass A produced no rollout text to copy (the mechanical
    verification is a hard gate, not a warning-and-continue)."""
    _patch_hub(monkeypatch)
    out_dir = tmp_path / "data"
    (out_dir / "pass_a").mkdir(parents=True)  # empty pass_a -> nothing to copy
    with pytest.raises(RuntimeError, match="raw-completions upload aborted"):
        X._upload_raw_completions(out_dir, smoke=False)


def test_raw_completions_upload_verification_fails_loud_on_missing_prefix(tmp_path, monkeypatch):
    """If the upload lands NOTHING under the raw_completions/ prefix (the
    pre-fix failure mode simulated by a no-op uploader), the fail-loud
    list_repo_files verification raises with the blocker id."""
    sink: dict = {"uploads": [], "files": set()}

    class _NoOpApi:
        def upload_folder(self, **_kw):  # lands nothing
            sink["uploads"].append(_kw.get("path_in_repo"))

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _NoOpApi())
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda *a, **k: [])
    out_dir = tmp_path / "data"
    _write_pass_a_cell(out_dir / "pass_a", "evil", "sys0")
    with pytest.raises(RuntimeError, match="raw-completions-upload-prefix-missing"):
        X._upload_raw_completions(out_dir, smoke=False)
