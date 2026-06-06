# ruff: noqa: RUF002, RUF003  # em-dash + × multiplication sign intentional
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
    names = ["villain", "qwen_default", "medical_doctor", "police_officer"]
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
    names = ["villain", "qwen_default"]
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
    names = ["villain", "qwen_default", "medical_doctor", "police_officer"]
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


# ── Round 5: regression tests for the wrapper-script raw-load bug class. ────
#
# Codex round-4 review found that the dispatcher fix in commit ce2bea8a2
# (route raw json.loads through load_persona_bank) was NOT propagated to two
# sibling CLI entrypoints under scripts/issue505_*.py — both still raw-loaded
# the structured #472 payload. These tests invoke each wrapper as a real
# subprocess (the canonical "library-fixed-but-wrapper-still-raw" anti-pattern
# only surfaces at the script's __main__, not the library API).


import os  # noqa: E402  — intentionally below the regression docstring so the section is grouped
import subprocess  # noqa: E402
import sys  # noqa: E402

from explore_persona_space.experiments.leave_one_out_505.panel_coverage import (  # noqa: E402
    run_panel_coverage_gate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_panel_friendly_centroids_L10(
    out_path: Path, names: list[str], source: str, always_include: str, dim: int = 32
) -> None:
    """Centroids whose cos(b, source) spreads the bank cleanly across spread-quantile bins.

    The spread-quantile selector groups personas by quantile of cos(p, source);
    if every persona is at the same quantile, K=6 picks degenerate near-twins
    and the tercile check still fires on every j_i. We synthesise centroids
    that DO span the quantile space by injecting a per-persona angular offset
    on a 2D circle and adding a small random tail.
    """
    n = len(names)
    rng = torch.Generator().manual_seed(0)
    # 2D anchor angles spread evenly around the circle (excluding source which
    # sits at angle 0). The remaining `dim-2` coords are small random noise so
    # the within-tercile variance is non-zero.
    angles = torch.zeros(n)
    other_names = [x for x in names if x != source]
    for i, name in enumerate(other_names):
        angles[names.index(name)] = (i + 1) * (3.14159 / (n - 1))
    primary = torch.stack([angles.cos(), angles.sin()], dim=-1)  # (n, 2)
    tail = 0.05 * torch.randn(n, dim - 2, generator=rng)
    centroids = torch.cat([primary, tail], dim=-1)
    centroids = centroids / centroids.norm(dim=-1, keepdim=True)
    cos_matrix = centroids @ centroids.T
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
    # always_include must exist as a key — assert here to fail loud if a test
    # mis-builds the names list.
    assert always_include in names, f"always_include {always_include!r} missing from names"


def test_panel_coverage_wrapper_script_uses_canonical_loader_on_synthetic_bank(tmp_path):
    """``scripts/issue505_panel_coverage.py`` must read persona_bank via the canonical loader.

    Reproduction of the same bug class the dispatcher hit in #505 round-3:
    a raw ``json.loads(args.persona_bank.read_text())`` returns the OUTER
    structured payload (keys leak), so ``run_panel_coverage_gate``'s
    ``for p in persona_bank`` iterates ``schema_version`` and crashes one
    frame deeper at ``cos_matrix_l10[source]['schema_version']``. Round 5
    fixed the wrapper to go through ``load_persona_bank``; this test
    invokes the wrapper as a real subprocess on the same synthetic
    #472-schema bank used by the dispatcher regression, and confirms the
    wrapper exits 0 + writes the gate payload. Pre-fix, this would have
    crashed with ``KeyError: 'schema_version'`` (raw-loaded payload).
    """
    # 60-persona bank, qwen_default at index 1 (so it isn't the source).
    names = ["villain", "qwen_default"] + [f"p{i}" for i in range(58)]
    bank_path = tmp_path / "issue_472" / "persona_bank.json"
    centroids_path = tmp_path / "issue_472" / "centroids_L10.pt"
    out_path = tmp_path / "out" / "panel_coverage.json"
    _write_persona_bank(bank_path, names)
    _build_panel_friendly_centroids_L10(
        centroids_path, names, source="villain", always_include="qwen_default"
    )

    script = REPO_ROOT / "scripts" / "issue505_panel_coverage.py"
    assert script.exists(), f"wrapper script missing at {script}"

    env = {**os.environ}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--persona-bank",
            str(bank_path),
            "--centroid-l10",
            str(centroids_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        check=False,
    )
    # Two possible expected outcomes:
    #   exit 0: gate PASSed → payload file with gate_passed=True.
    #   exit 2: gate FAILed on tercile (insufficient spread for the synthetic
    #           centroids) — still a clean fail, NOT the raw-load crash.
    # What we MUST NOT see: a python traceback with KeyError on 'schema_version'.
    combined = result.stdout + result.stderr
    assert "KeyError" not in combined or "schema_version" not in combined, (
        f"wrapper still raw-reads structured persona_bank payload; combined output:\n{combined}"
    )
    assert result.returncode in (0, 2), (
        f"unexpected exit {result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    # Pre-the-tercile-only fix, the gate would have FAILED on every j_i
    # because the variance floor 0.02**2=0.0004 is well above the realised
    # within-panel variance on these synthetic 2D-on-circle centroids. After
    # the Round-5 PANEL_VARIANCE_FLOOR drop, the gate passes on this fixture.
    assert result.returncode == 0, (
        f"panel-coverage gate did not pass on synthetic fixture; "
        f"return={result.returncode}; stderr={result.stderr!r}"
    )
    payload = json.loads(out_path.read_text())
    assert payload["gate_passed"] is True, f"gate_passed not True; payload keys: {list(payload)}"


def test_panel_coverage_wrapper_script_fails_loud_on_schema_drift(tmp_path):
    """If the persona_bank.json has a wrong schema_version, the wrapper exits non-zero.

    Pre-fix (raw `json.loads`) the wrapper would have silently iterated the
    metadata wrapper and crashed inside ``run_panel_coverage_gate`` with
    ``KeyError`` 10 frames deep. Post-fix the canonical loader raises
    ``AssertionError`` at the load step — the wrapper script propagates
    that to a non-zero exit.
    """
    names = ["villain", "qwen_default", "p0", "p1", "p2", "p3", "p4", "p5"]
    bank_path = tmp_path / "issue_472" / "persona_bank.json"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    # Drifted schema_version — canonical loader raises AssertionError.
    bank_path.write_text(
        json.dumps(
            {
                "schema_version": "i472_DRIFT",
                "personas": {n: f"You are a {n}." for n in names},
            }
        )
    )
    centroids_path = tmp_path / "issue_472" / "centroids_L10.pt"
    _build_panel_friendly_centroids_L10(
        centroids_path, names, source="villain", always_include="qwen_default"
    )

    script = REPO_ROOT / "scripts" / "issue505_panel_coverage.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--persona-bank",
            str(bank_path),
            "--centroid-l10",
            str(centroids_path),
            "--out",
            str(tmp_path / "out" / "panel_coverage.json"),
        ],
        capture_output=True,
        text=True,
        env={**os.environ},
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert result.returncode != 0, "expected non-zero exit on schema drift"
    assert "schema_version" in (result.stdout + result.stderr), (
        f"expected schema_version error message; combined: {result.stdout}\n{result.stderr}"
    )


def test_build_pv_centroids_wrapper_script_fails_loud_on_schema_drift(tmp_path):
    """Symmetric guard for ``scripts/issue505_build_pv_centroids.py``.

    The second wrapper Codex flagged in round 4: same raw-load bug as the
    panel-coverage wrapper. We can't end-to-end-smoke this script in CI
    (it loads Qwen-2.5-7B for the forward pass), but we CAN confirm the
    bank load goes through ``load_persona_bank`` by feeding a
    schema-drifted bank and asserting the script exits non-zero BEFORE
    any model-loading happens. Pre-fix (raw json.loads) the bank load
    would have silently returned the wrapper dict and the script would
    have proceeded into model loading; post-fix the canonical loader
    raises AssertionError immediately at the bank load step.
    """
    names = ["villain", "qwen_default", "p0", "p1"]
    bank_path = tmp_path / "issue_472" / "persona_bank.json"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(
        json.dumps(
            {
                "schema_version": "i472_DRIFT",
                "personas": {n: f"You are a {n}." for n in names},
            }
        )
    )

    script = REPO_ROOT / "scripts" / "issue505_build_pv_centroids.py"
    assert script.exists(), f"wrapper script missing at {script}"

    # Pass --layers 7 just to keep the arg parse simple; the script never
    # reaches model load because the schema check fires first.
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--persona-bank",
            str(bank_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--layers",
            "7",
        ],
        capture_output=True,
        text=True,
        env={**os.environ},
        cwd=str(REPO_ROOT),
        check=False,
        timeout=60,
    )
    assert result.returncode != 0, (
        f"expected non-zero exit on schema drift; stdout={result.stdout!r}; "
        f"stderr={result.stderr!r}"
    )
    assert "schema_version" in (result.stdout + result.stderr), (
        f"expected schema_version error message; combined: {result.stdout}\n{result.stderr}"
    )


# ── Round 5: §5.4 variance-floor drop — tercile-only gate passes on real #472 data. ─


def _maybe_real_472_paths() -> tuple[Path, Path] | None:
    """Resolve the real #472 persona_bank.json + centroids_L10.pt from the HF cache.

    Uses ``hf_hub_download`` so the call is cached (no network if previously
    fetched on this VM). Returns None when ``HF_TOKEN`` is unset (CI without
    HF credentials skips the test rather than failing).
    """
    if not os.environ.get("HF_TOKEN"):
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    repo = "superkaiba1/explore-persona-space-data"
    prefix = "issue472_neg_geometry/geometry"
    try:
        bank = Path(
            hf_hub_download(
                repo_id=repo,
                filename=f"{prefix}/persona_bank.json",
                repo_type="dataset",
                token=os.environ["HF_TOKEN"],
            )
        )
        centroids = Path(
            hf_hub_download(
                repo_id=repo,
                filename=f"{prefix}/centroids_L10.pt",
                repo_type="dataset",
                token=os.environ["HF_TOKEN"],
            )
        )
    except Exception:
        return None
    return bank, centroids


def test_gate_passes_on_real_472_bank_with_tercile_only():
    """The §5.4 panel-coverage gate passes on real #472 data after Round-5's variance-floor drop.

    Round-4 review (Claude lens) verified independently that all 6 sampled
    j_i in the realised bank PASSed ``tercile_ok`` but FAILed
    ``spans_floor`` (within-panel variances 0.000116-0.000175 vs the
    misderived floor 0.0004 ≈ 2.3-3.5× too high). The fix dropped the
    variance gate as a unit-error correction. This test confirms the
    fix lands the way Claude described: gate_passed=True on the real
    #472 60-persona bank + L10 centroids, no synthetic data.

    Skipped when ``HF_TOKEN`` is missing (CI without HF credentials).
    """
    paths = _maybe_real_472_paths()
    if paths is None:
        pytest.skip("HF_TOKEN unset or HF hub download unavailable; skipping real-data test.")
    bank_path, centroids_path = paths

    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )

    bank = load_persona_bank(bank_path)
    cos_l10 = load_inherited_l10_cos(centroids_path)
    payload = run_panel_coverage_gate(persona_bank=bank, cos_matrix_l10=cos_l10)

    # Round 5 contract: tercile_ok holds for every chosen j_i on the real bank.
    assert payload["gate_passed"] is True, (
        f"gate did not pass on real #472 data; n_retries_used={payload['n_retries_used']}; "
        f"coverage diagnostics: {json.dumps(payload['coverage'], indent=2, default=float)}"
    )
    # Sanity: K=6 non-default negatives + always-included qwen_default = 7 in k_set.
    assert len(payload["k_set"]) == 7, f"unexpected k_set size: {payload['k_set']}"
    assert payload["always_include"] == "qwen_default"
    # Diagnostic visibility: every chosen j_i carries tercile_ok=True; the
    # spans_floor field is still reported but no longer in the pass condition.
    for j_i, diag in payload["coverage"].items():
        assert diag["tercile_ok"] is True, f"j_i {j_i!r} failed tercile check: {diag}"
