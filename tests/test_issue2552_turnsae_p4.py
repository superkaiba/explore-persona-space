"""P4 resume-contract regression tests for scripts/issue2552_turnsae_der.py
(#2552 r4 p4-phase-contract): the p4_done.json completion sentinel RECORDS input
hashes + the upload disposition (r3), and the resume-skip predicate must CONSULT
them — a stale recorded input hash forces a recompute; a fully matching sentinel
(regime + inputs + uploads + outputs) skips. Fails pre-fix: the r3 skip compared
only the regime key + output existence, so the stale-hash leg skipped.

Torch-bearing module import (the driver imports torch at module top; precedent:
tests/test_issue928_decomposition.py). No network and no repo-committed-artifact
reads: the external embedder boundary (_p4_load_embedder / _p4_embed_texts) is
stubbed with signature-conformant fakes, and everything else is the production
body running over the driver's own deterministic smoke fixtures under tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "scripts" / "vendored_2476")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2552_turnsae_der as M  # noqa: E402


def _args(out_root: Path):
    return M._parse_args(
        [
            "--phase",
            "p4-embed",
            "--smoke",
            "--skip-upload",
            "--device",
            "cpu",
            "--out-root",
            str(out_root),
        ]
    )


def _run_p4(args, monkeypatch, calls: list[int]) -> None:
    """phase_p4_embed with the external embedder boundary stubbed (signature-
    conformant by construction: the fakes' defs mirror the real signatures)."""

    def fake_load_embedder(args):  # mirrors _p4_load_embedder(args) -> (model, tok)
        return object(), object()

    def fake_embed_texts(args, model, tok, texts):  # mirrors _p4_embed_texts
        calls.append(len(texts))
        rng = np.random.default_rng(0)
        e = rng.normal(size=(len(texts), 8)).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    monkeypatch.setattr(M, "_p4_load_embedder", fake_load_embedder)
    monkeypatch.setattr(M, "_p4_embed_texts", fake_embed_texts)
    M.phase_p4_embed(args)


def test_p4_matching_sentinel_skips_and_stale_input_hash_recomputes(tmp_path, monkeypatch):
    """r4 p4-phase-contract, both legs of the brief: matching sentinel => skip;
    stale recorded input hash => recompute (pre-fix this third run skipped and
    len(calls) stayed 1)."""
    args = _args(tmp_path)
    calls: list[int] = []
    _run_p4(args, monkeypatch, calls)
    assert len(calls) == 1, "first run must embed"
    done = tmp_path / "sentinels" / "p4_done.json"
    doc = json.loads(done.read_text())
    assert doc["regime_key"]["smoke"] is True
    assert doc["inputs"]["summaries_sha256"], "sentinel records input identity"
    assert doc["uploads"].startswith("skipped"), "smoke run records a skipped upload"
    # leg 1 — matching sentinel (regime + inputs + uploads + outputs): resume-skip.
    # Also pins the r4 fixture-synth idempotence: a re-synthesis would restamp
    # provenance timestamps into the fixtures and break the hash match.
    _run_p4(args, monkeypatch, calls)
    assert len(calls) == 1, "matching sentinel must skip the embed pass"
    # leg 2 — stale recorded input hash: recompute
    doc["inputs"]["summaries_sha256"] = "0" * 64
    done.write_text(json.dumps(doc))
    _run_p4(args, monkeypatch, calls)
    assert len(calls) == 2, "stale recorded input hash must force a recompute"
    # the recompute rewrites the sentinel with the CURRENT input identity
    assert json.loads(done.read_text())["inputs"]["summaries_sha256"] != "0" * 64


def test_p4_resume_mismatch_upload_disposition_and_outputs(tmp_path):
    """_p4_resume_mismatch unit legs: a skipped-upload sentinel never satisfies a
    run that REQUIRES upload (the Codex r3 skip-upload-then-production case); a
    verified-complete disposition does; missing outputs and mismatched input
    fields are named in the returned mismatch."""
    rk = {"emb_model": "m", "smoke": False}
    inp = {"a": "x"}
    out = tmp_path / "o.json"
    out.write_text("{}")
    prior = {"regime_key": rk, "inputs": inp, "uploads": "skipped (skip_upload/non-production)"}
    m = M._p4_resume_mismatch(prior, rk, inp, (out,), upload_required=True)
    assert m is not None and "uploads" in m
    assert M._p4_resume_mismatch(prior, rk, inp, (out,), upload_required=False) is None
    prior_up = {**prior, "uploads": "uploaded+verified (analysis_tensors/embcov)"}
    assert M._p4_resume_mismatch(prior_up, rk, inp, (out,), upload_required=True) is None
    missing = tmp_path / "absent.npz"
    m2 = M._p4_resume_mismatch(prior_up, rk, inp, (out, missing), upload_required=True)
    assert m2 is not None and "outputs" in m2 and "absent.npz" in m2
    m3 = M._p4_resume_mismatch(prior_up, rk, {"a": "y"}, (out,), upload_required=True)
    assert m3 is not None and m3.startswith("inputs") and "'a'" in m3
    # pre-r3 sentinel with NO recorded inputs: never a silent skip
    legacy = {"regime_key": rk, "uploads": "uploaded+verified (analysis_tensors/embcov)"}
    m4 = M._p4_resume_mismatch(legacy, rk, inp, (out,), upload_required=False)
    assert m4 is not None and m4.startswith("inputs")
