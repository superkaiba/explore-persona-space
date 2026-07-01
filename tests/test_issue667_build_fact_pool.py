"""Issue #667 fact-pool builder regression tests (round-2 BLOCKER 1 + CONCERN 1).

BLOCKER 1 (``fact-pool-adapter-generation-base-only``): pre-fix, both the ``base``
and ``adapter`` arms of ``_generate_completions`` went through ``vllm_generate_R``,
which hardcodes ``LLM(model=BASE_MODEL)`` with NO LoRA path — so BOTH arms were
base-model completions and the "adapter" positive pool was a mislabel. Post-fix the
``adapter`` arm generates through the loaded ``trained`` PeftModel (base + the
fact adapter) via HF ``.generate()``; only the ``base`` arm uses vLLM (on GPU).
These tests mock the base + trained generators to return DISTINGUISHABLE text and
assert the two arms produce DIFFERENT rows (a smoke on the labeling), plus that the
per-arm generation-provenance records the adapter path.

CONCERN 1 (``fact-arm-floor-drop-not-enforced``): pre-fix, ``dropped_from_headline``
was computed but no downstream consumer read it. Post-fix a below-floor pool writes
a ``DROPPED_FROM_HEADLINE.sentinel`` and an at/above-floor pool removes any stale
sentinel. These tests exercise both branches at the builder level.
"""

# math/scientific notation in docstrings + messages

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue667_build_fact_pool as bfp  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# BLOCKER 1 — the two arms use DIFFERENT models (base vs the loaded PeftModel).
# ─────────────────────────────────────────────────────────────────────────────


class _Sentinel:
    """A stand-in model object so ``model is trained`` / ``model is base`` work."""

    def __init__(self, name):
        self.name = name


def _install_distinguishable_generators(monkeypatch, *, cpu_only):
    """Mock the fact-pool generators so base and adapter produce DIFFERENT text.

    Returns (base, trained) sentinels. The base arm (vLLM on GPU / HF on CPU) yields
    "BASE-<probe>" and the adapter arm (HF through the PeftModel) yields
    "ADAPTER-<probe>", so a caller can assert the two arms are distinct rows.
    """
    base = _Sentinel("base")
    trained = _Sentinel("trained")
    tok = _Sentinel("tok")

    monkeypatch.setattr(bfp, "_generate_completions", bfp._generate_completions)  # keep real

    # Patch the helpers _generate_completions imports from issue667_extract.
    import issue667_extract as ex

    monkeypatch.setattr(ex, "stage_adapter_local", lambda *a, **k: Path("/tmp/fake_adapter_dir"))
    monkeypatch.setattr(ex, "assert_adapter_gauge", lambda *a, **k: {})
    monkeypatch.setattr(ex, "load_base_and_trained", lambda *a, **k: (tok, base, trained))
    monkeypatch.setattr(ex, "_FACT_POS_SYS", "SYS")
    monkeypatch.setattr(
        ex, "_device", lambda gpu_id, co: type("D", (), {"type": "cpu" if co else "cuda"})()
    )

    # Base arm on GPU -> vllm_generate_R; on CPU -> _hf_generate_with_adapter(base).
    def fake_vllm(tok_, msg_lists, *, max_new_tokens):
        return [f"BASE-{m[-1]['content']}" for m in msg_lists]

    monkeypatch.setattr(ex, "vllm_generate_R", fake_vllm)

    # Both HF-generate calls (adapter arm always; base arm on CPU) route here.
    def fake_hf(model, tok_, msg_lists, device, *, max_new_tokens=256):
        tag = "ADAPTER" if model is trained else "BASE"
        return [f"{tag}-{m[-1]['content']}" for m in msg_lists]

    monkeypatch.setattr(bfp, "_hf_generate_with_adapter", fake_hf)
    return base, trained


def test_adapter_arm_uses_peftmodel_not_base_only(monkeypatch):
    """The 'adapter' arm generates through the loaded PeftModel, not the base vLLM path.

    Pre-fix BOTH arms went through ``vllm_generate_R`` (base only), so the two arms
    were byte-identical for the same probe. Post-fix the adapter arm routes through
    ``_hf_generate_with_adapter(trained, ...)`` and the base arm through vLLM, so the
    two arms differ. (GPU path exercised: cpu_only=False.)
    """
    _install_distinguishable_generators(monkeypatch, cpu_only=False)
    rows, adapter_dir = bfp._generate_completions(["Q1", "Q2"], n_rollouts=1, cpu_only=False)

    base_rows = {r["probe"]: r["answer"] for r in rows if r["source"] == "base"}
    adapter_rows = {r["probe"]: r["answer"] for r in rows if r["source"] == "adapter"}
    assert base_rows and adapter_rows, "both arms must produce rows"
    # The labeling smoke: for the SAME probe the two arms differ (adapter != base).
    for probe in base_rows:
        assert base_rows[probe] != adapter_rows[probe], (
            f"adapter arm equals base arm for {probe!r} -- adapter PeftModel not used"
        )
    assert all(r["answer"].startswith("BASE-") for r in rows if r["source"] == "base")
    assert all(r["answer"].startswith("ADAPTER-") for r in rows if r["source"] == "adapter")
    assert adapter_dir == "/tmp/fake_adapter_dir"


def test_cpu_smoke_base_arm_uses_hf_not_vllm(monkeypatch):
    """On the CPU smoke both arms route through HF greedy (no vLLM), still distinct."""
    called = {"vllm": 0}

    _install_distinguishable_generators(monkeypatch, cpu_only=True)
    import issue667_extract as ex

    orig_vllm = ex.vllm_generate_R

    def counting_vllm(*a, **k):
        called["vllm"] += 1
        return orig_vllm(*a, **k)

    monkeypatch.setattr(ex, "vllm_generate_R", counting_vllm)
    rows, _ = bfp._generate_completions(["Q1"], n_rollouts=1, cpu_only=True)
    assert called["vllm"] == 0, "CPU smoke must NOT call vLLM"
    base = {r["probe"]: r["answer"] for r in rows if r["source"] == "base"}
    adapter = {r["probe"]: r["answer"] for r in rows if r["source"] == "adapter"}
    assert base["Q1"] != adapter["Q1"]


# ─────────────────────────────────────────────────────────────────────────────
# CONCERN 1 — the yield-floor DROP sentinel is written (below floor) / cleared.
# ─────────────────────────────────────────────────────────────────────────────


def _run_build_with_scores(monkeypatch, tmp_path, *, pos_scores, neg_scores):
    """Drive build_fact_pool end-to-end with a stubbed judge returning fixed scores.

    ``pos_scores`` completions score >50 (positives), ``neg_scores`` score <50
    (negatives). Generation + probes + judge + adapter staging are all mocked; only
    the cap/floor/sentinel logic under test runs for real. Returns the provenance.
    """
    # Root the pool dir under the pytest tmp (PROJECT_ROOT / POOL_DIR -> tmp_path/pool).
    monkeypatch.setattr(bfp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(bfp, "POOL_DIR", "pool")
    # Generation: one row per intended score, alternating so we control pos/neg counts.
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    fake_rows = [
        {"probe": f"P{i}", "probe_idx": i, "rollout_idx": 0, "answer": f"a{i}", "source": "base"}
        for i in range(n_pos + n_neg)
    ]
    scores = list(pos_scores) + list(neg_scores)
    monkeypatch.setattr(
        bfp, "_generate_completions", lambda probes, n_rollouts, cpu_only: (fake_rows, "/tmp/ad")
    )
    monkeypatch.setattr(bfp, "load_eval_probes", lambda b: ["P"], raising=False)

    import issue667_extract as ex

    monkeypatch.setattr(ex, "load_eval_probes", lambda b: ["P"])
    monkeypatch.setattr(ex, "stage_inputs", lambda: (Path("x"), Path("y")))

    it = iter(scores)
    monkeypatch.setattr(bfp, "_judge_one", lambda client, ans: float(next(it)))
    monkeypatch.setattr(
        bfp.anthropic if hasattr(bfp, "anthropic") else bfp,
        "Anthropic",
        lambda **k: object(),
        raising=False,
    )
    # anthropic is imported lazily inside build_fact_pool; patch the module object.
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: object())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    return bfp.build_fact_pool(cap=40, n_rollouts=1, cpu_only=True, max_probes=None)


def test_below_floor_writes_drop_sentinel(monkeypatch, tmp_path):
    """floor-N < YIELD_FLOOR_MIN (15) -> DROPPED_FROM_HEADLINE.sentinel written."""
    # 5 pos + 5 neg -> floor_n = 5 < 15 -> below floor.
    prov = _run_build_with_scores(monkeypatch, tmp_path, pos_scores=[90] * 5, neg_scores=[10] * 5)
    assert prov["dropped_from_headline"] is True
    sentinel = tmp_path / "pool" / bfp.DROP_SENTINEL_NAME
    assert sentinel.is_file(), "below-floor build must write the DROP sentinel"
    payload = json.loads(sentinel.read_text())
    assert payload["dropped_from_headline"] is True
    assert payload["floor_n_per_side"] == 5
    # The per-arm generation-provenance records the adapter path (BLOCKER 1).
    assert prov["generation_provenance"]["adapter"]["generation_path"] == "hf_generate"


def test_at_floor_removes_stale_sentinel(monkeypatch, tmp_path):
    """floor-N >= YIELD_FLOOR_MIN -> no sentinel; a stale one is removed."""
    # Pre-seed a stale sentinel from a hypothetical prior below-floor run.
    pool = tmp_path / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    (pool / bfp.DROP_SENTINEL_NAME).write_text("{}")
    # 20 pos + 20 neg -> floor_n = 20 >= 15 -> at/above floor.
    prov = _run_build_with_scores(monkeypatch, tmp_path, pos_scores=[90] * 20, neg_scores=[10] * 20)
    assert prov["dropped_from_headline"] is False
    assert not (pool / bfp.DROP_SENTINEL_NAME).exists(), "stale sentinel must be removed"
