# em-dash intentional
"""Task #505 regression — Phase 0b/Phase 1 loaders unwrap #472's structured payloads.

The #505 round-3 v3 production launch (2026-06-05) crashed within ~20s of
``nohup`` at ``panel_coverage.py:149``:

    cos_to_source = {p: float(cos_matrix_l10[source][p])
                     for p in persona_bank if p != source}
    KeyError: 'schema_version'

Root cause: ``leave_one_out_505.dispatch._load_persona_bank_and_r`` read the
``persona_bank.json`` payload raw via ``json.loads``, but #472 publishes the
file as a STRUCTURED payload:

    {
      "schema_version": "i472_v1",
      "source_persona": ...,
      "personas": {name: prompt, ...},   # the actual bank
      ...
    }

The raw read leaked metadata keys (``schema_version``, ``source_persona``,
``n_base``, ``n_new``, ``n_total``, ``content_hash``, ``git_commit``,
``generated_at``, ``sonnet_model``) into the persona_bank iteration, so the
panel-coverage gate's ``for p in persona_bank`` loop hit ``schema_version``
and crashed when the dictcomp tried ``cos_matrix_l10[source]['schema_version']``.

Same bug shape for the R artifacts (``R_train.json`` / ``R_eval.json``): they
ALSO wrap their actual completions map under ``payload['completions']``.

The fix routes both loads through the canonical helpers
``contrastive_neg_geometry_472.persona_bank.load_persona_bank`` and
``contrastive_neg_geometry_472.r_generate.load_r_artifact``, which both
validate ``schema_version`` and return the unwrapped inner dict.

This regression test exercises the FULL ``_load_persona_bank_and_r`` path on a
mini fixture mirroring the on-disk #472 payload schema (zero GPU / network),
plus a direct test of ``panel_coverage.load_inherited_l10_cos`` against a
synthetic ``centroids_L10.pt`` mirroring the on-disk #472 centroids schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
    SCHEMA_VERSION as BANK_SCHEMA_VERSION,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
    SCHEMA_VERSION as R_SCHEMA_VERSION,
)
from explore_persona_space.experiments.leave_one_out_505 import dispatch as l1o_dispatch
from explore_persona_space.experiments.leave_one_out_505.panel_coverage import (
    load_inherited_l10_cos,
)

# ── Fixtures: minimal #472-schema-faithful artifacts on a temp dir. ─────────


def _write_persona_bank(out_path: Path, names: list[str]) -> None:
    """Write a persona_bank.json that mirrors #472's `payload` structure exactly."""
    payload = {
        "schema_version": BANK_SCHEMA_VERSION,
        "source_persona": names[0],
        "n_base": len(names),
        "n_new": 0,
        "n_total": len(names),
        "personas": {n: f"You are a {n}." for n in names},
        "content_hash": "deadbeef" * 8,
        "git_commit": "0" * 40,
        "generated_at": "2026-06-05T00:00:00+00:00",
        "sonnet_model": "claude-sonnet-4-5-20250929",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def _write_r_artifact(out_path: Path, names: list[str], questions: list[str]) -> None:
    """Write an R_{train,eval}.json that mirrors #472's `payload` structure exactly."""
    completions = {
        n: {q: {"text": f"R[{n}][{q}]", "tokens": [1, 2, 3]} for q in questions} for n in names
    }
    payload = {
        "schema_version": R_SCHEMA_VERSION,
        "split": "train",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "base_model_revision": "abc1234",
        "generation_config": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 32, "seed": 0},
        "n_personas": len(names),
        "questions": list(questions),
        "personas": sorted(names),
        "completions": completions,
        "content_hash": "feedface" * 8,
        "git_commit": "0" * 40,
        "generated_at": "2026-06-05T00:00:00+00:00",
        "stats": {"n_total": len(names) * len(questions)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def _write_centroids_L10(out_path: Path, names: list[str], dim: int = 8) -> None:
    """Write a centroids_L10.pt mirroring #472's `torch.save({...})` payload exactly."""
    n = len(names)
    rng = torch.Generator().manual_seed(0)
    centroids = torch.randn(n, dim, generator=rng)
    centroids = centroids / centroids.norm(dim=-1, keepdim=True)
    cos_matrix = centroids @ centroids.T  # (n, n) symmetric, on-diag = 1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "centroids": centroids,
            "persona_names": list(names),
            "cos_matrix": cos_matrix,
            "layer": 10,
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "questions": ["q0", "q1"],
        },
        out_path,
    )


# ── Tests. ──────────────────────────────────────────────────────────────────


def test_load_persona_bank_and_r_unwraps_payloads(monkeypatch, tmp_path):
    """Phase 0b loader must return UNWRAPPED bank + R dicts.

    Reading the persona_bank.json / R_*.json files raw via ``json.loads`` was
    the regression that crashed #505 round-3 at Phase 1 with
    ``KeyError: 'schema_version'``. Going through the canonical loaders unwraps
    the metadata wrapper and yields the inner ``personas`` / ``completions``
    map directly.
    """
    i472 = tmp_path / "issue_472"
    names = ["villain_persona", "qwen_default", "medical_doctor", "police_officer"]
    questions = ["q0", "q1"]
    _write_persona_bank(i472 / "persona_bank.json", names)
    _write_r_artifact(i472 / "on_policy_R" / "R_train.json", names, questions)
    _write_r_artifact(i472 / "on_policy_R" / "R_eval.json", names, questions)

    monkeypatch.setenv("EPM_I472_DATA_ROOT", str(i472))
    bank, r_train, r_eval, q_train, q_eval = l1o_dispatch._load_persona_bank_and_r()

    # The bank MUST be the inner persona name -> system-prompt map, NOT the
    # wrapping payload. The original bug: the wrapper's `schema_version` /
    # `source_persona` / etc keys leaked into the bank iteration.
    assert "schema_version" not in bank, (
        f"persona_bank leaks wrapper key 'schema_version': bank keys = {sorted(bank.keys())}"
    )
    assert "personas" not in bank, "persona_bank itself contains a 'personas' key (re-wrapped?)"
    assert set(bank.keys()) == set(names), f"bank keys mismatch: got {sorted(bank.keys())}"
    assert all(isinstance(v, str) for v in bank.values()), (
        "bank values must be system-prompt strings"
    )

    # Same shape contract for R_train + R_eval: completions[persona][q] -> dict.
    for label, r in (("r_train", r_train), ("r_eval", r_eval)):
        assert "schema_version" not in r, (
            f"{label} leaks wrapper key 'schema_version': keys = {sorted(r.keys())}"
        )
        assert set(r.keys()) == set(names), f"{label} keys mismatch: got {sorted(r.keys())}"
        assert set(r[names[0]].keys()) == set(questions), f"{label}[{names[0]}] q-keys mismatch"

    # Q_train / Q_eval are sorted question lists.
    assert q_train == sorted(questions)
    assert q_eval == sorted(questions)


def test_load_persona_bank_and_r_raises_on_schema_drift(monkeypatch, tmp_path):
    """If a future #472 rebuild ships a different schema_version, fail LOUD.

    A silent default would re-introduce the #505 round-3 class of bug (one
    field renames, downstream crashes deep in panel_coverage). The canonical
    ``load_persona_bank`` validates the schema and raises AssertionError on
    drift — confirm that propagates through the dispatcher loader.
    """
    i472 = tmp_path / "issue_472"
    names = ["villain_persona", "qwen_default"]
    questions = ["q0"]
    # Write a bank with a wrong schema_version.
    bank_path = i472 / "persona_bank.json"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(
        json.dumps(
            {
                "schema_version": "i472_DRIFT",  # NOT BANK_SCHEMA_VERSION
                "personas": {n: "..." for n in names},
            }
        )
    )
    # R artifacts written with the correct schema (so the bank check fails first).
    _write_r_artifact(i472 / "on_policy_R" / "R_train.json", names, questions)
    _write_r_artifact(i472 / "on_policy_R" / "R_eval.json", names, questions)

    monkeypatch.setenv("EPM_I472_DATA_ROOT", str(i472))
    with pytest.raises(AssertionError, match="schema_version"):
        l1o_dispatch._load_persona_bank_and_r()


def test_load_inherited_l10_cos_unwraps_structured_pt(tmp_path):
    """The L10 centroid bundle is a structured dict; load_inherited_l10_cos unwraps it.

    Mirrors the on-disk #472 ``centroids_L10.pt`` schema written by
    ``contrastive_neg_geometry_472.centroids.build_centroids``. The returned
    object MUST be the nested ``dict[name][name] -> float`` that
    ``panel_coverage._spread_quantile_k_set`` and the
    ``cos_to_source = {p: float(cos_matrix_l10[source][p]) for p in persona_bank}``
    pattern at panel_coverage.py:149 expects.
    """
    names = ["villain_persona", "qwen_default", "medical_doctor", "police_officer"]
    bundle_path = tmp_path / "centroids_L10.pt"
    _write_centroids_L10(bundle_path, names)

    cos = load_inherited_l10_cos(bundle_path)

    # Two-level dict[name][name] -> float — the exact shape the panel-coverage
    # dictcomp pattern needs:
    #   cos_to_source = {p: float(cos_matrix_l10[source][p])
    #                    for p in persona_bank if p != source}
    assert isinstance(cos, dict), f"expected dict, got {type(cos).__name__}"
    assert set(cos.keys()) == set(names), f"top-level keys mismatch: got {sorted(cos.keys())}"
    for outer in names:
        assert isinstance(cos[outer], dict), f"cos[{outer!r}] is not a dict"
        assert set(cos[outer].keys()) == set(names), (
            f"cos[{outer!r}] inner keys mismatch: got {sorted(cos[outer].keys())}"
        )
        for inner in names:
            v = cos[outer][inner]
            assert isinstance(v, float), (
                f"cos[{outer!r}][{inner!r}] is {type(v).__name__}, not float"
            )

    # Symmetry + on-diagonal = 1.0 (the matrix was built from normalized vectors).
    for a in names:
        assert abs(cos[a][a] - 1.0) < 1e-5, f"cos[{a!r}][{a!r}] = {cos[a][a]} (expected 1.0)"
    for a in names:
        for b in names:
            assert abs(cos[a][b] - cos[b][a]) < 1e-5, "cos matrix asymmetric"

    # Smoke-exercise the exact panel_coverage.py:149 pattern with this output.
    persona_bank = {n: f"You are a {n}." for n in names}
    source = names[0]
    cos_to_source = {p: float(cos[source][p]) for p in persona_bank if p != source}
    assert set(cos_to_source.keys()) == set(names) - {source}


def test_load_inherited_l10_cos_raises_on_missing_keys(tmp_path):
    """A drifted centroid bundle (wrong keys) fails LOUD instead of crashing deep.

    Symmetric to the persona_bank schema-drift guard: catching the schema
    mismatch at the loader is the difference between a clear stacktrace at
    Phase 1 and ``KeyError`` ten frames deep in a dictcomp.
    """
    bundle_path = tmp_path / "centroids_L10.pt"
    # Deliberately missing 'cos_matrix' to trigger the schema check.
    torch.save({"persona_names": ["a", "b"], "centroids": torch.zeros(2, 4)}, bundle_path)

    with pytest.raises(KeyError, match="cos_matrix"):
        load_inherited_l10_cos(bundle_path)


def test_load_inherited_l10_cos_raises_on_non_dict_bundle(tmp_path):
    """A tensor-as-bundle (e.g. someone saved the raw matrix) fails LOUD."""
    bundle_path = tmp_path / "centroids_L10.pt"
    torch.save(torch.eye(4), bundle_path)  # raw tensor, no schema dict

    with pytest.raises(TypeError, match="expected dict"):
        load_inherited_l10_cos(bundle_path)
