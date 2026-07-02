"""Offline tests for the #779 round-5 artifacts reconstruction + HF fallback.

Round 5 closes the Arm B/C relaunch blocker: the parent's Sonnet-generated
extraction artifacts (data/issue_779/artifacts/{sycophancy,hallucination}.json)
were never uploaded. Three surfaces are pinned here, all CPU-only / no network
(``huggingface_hub.hf_hub_download`` is monkeypatched — the r4
``_resolve_rb_path`` test pattern):

1. ``issue779_common.load_extraction_artifacts`` HF fallback — local cache
   first, then ``issue779_monitoring/artifacts/<trait>.json`` from the HF data
   repo MATERIALIZED into the local cache; FileNotFoundError only when both
   miss; evil never touches HF.
2. The corpus driver's round-4 early preflight (``_preflight_artifacts``)
   PASSES when the artifact resolves via the HF fallback (probe is no longer
   local-only).
3. ``issue779_reconstruct_artifacts`` pure helpers — verbatim question
   recovery from the parent judge-dispatch items (exactly 20; pos == neg;
   order preserved), the splice (recovered questions survive VERBATIM into the
   written JSON; regenerated eval questions de-collided so the disjoint split
   survives), parent-score reconstruction (DROP-NEVER-COERCE per draw), and
   the deterministic stratified sampler.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue779_common as C  # noqa: E402
import issue779_reconstruct_artifacts as RA  # noqa: E402

# ── fixtures ─────────────────────────────────────────────────────────────────

QUESTIONS_20 = [f"synthetic extraction question {j}?" for j in range(20)]


def _mk_items(
    trait: str,
    arm: str,
    questions: list[str],
    *,
    n_personas: int = 2,
    n_rollouts: int = 2,
    n_draws: int = C.JUDGE_N_DRAWS,
) -> dict[str, dict]:
    """Synthetic dispatch items.json in the parent's global-idx custom_id scheme."""
    items: dict[str, dict] = {}
    for p in range(n_personas):
        for j, q in enumerate(questions):
            gq = p * len(questions) + j
            for ri in range(n_rollouts):
                for d in range(n_draws):
                    cid = f"{trait}_{arm}_p{p}__{gq:05d}__{ri * n_draws + d:02d}"
                    items[cid] = {
                        "question": q,
                        "completion": f"completion p{p} q{j} r{ri}",
                        "user_msg": f"[Q]{q}[/Q]",
                    }
    return items


def _valid_artifact_json() -> dict:
    """A well-formed 4-key artifact JSON (the load_extraction_artifacts shape)."""
    return {
        "instruction": [{"pos": f"pos {i}", "neg": f"neg {i}"} for i in range(5)],
        "extraction_questions": [f"eq {i}" for i in range(20)],
        "eval_questions": [f"vq {i}" for i in range(20)],
        "eval_prompt": "You are evaluating the trait X.\n\nPrompt:\n{question}\n{answer}",
    }


# ── 1. load_extraction_artifacts HF fallback ─────────────────────────────────


def test_load_extraction_artifacts_hf_fallback_materializes_locally(monkeypatch, tmp_path):
    """No local cache -> fetch issue779_monitoring/artifacts/<trait>.json from the
    HF data repo (repo_type=dataset) and MATERIALIZE it into _artifacts_dir();
    a second load is a local hit that never re-touches HF."""
    import huggingface_hub

    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(C, "_artifacts_dir", lambda: artifacts_dir)

    calls: list[dict] = []
    payload = _valid_artifact_json()

    def fake_download(repo_id, filename, repo_type=None, **kw):
        calls.append({"repo_id": repo_id, "filename": filename, "repo_type": repo_type})
        p = tmp_path / "hf_cache" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload))
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    got = C.load_extraction_artifacts("sycophancy")
    assert got == payload
    assert calls == [
        {
            "repo_id": C.HF_DATA_REPO,
            "filename": f"{C.HF_PREFIX}/artifacts/sycophancy.json",
            "repo_type": "dataset",
        }
    ], f"unexpected HF fetch spec: {calls}"
    materialized = artifacts_dir / "sycophancy.json"
    assert materialized.exists(), "must materialize into the local artifacts dir"
    assert json.loads(materialized.read_text()) == payload

    got2 = C.load_extraction_artifacts("sycophancy")
    assert got2 == payload and len(calls) == 1, "local hit must not re-download"


def test_load_extraction_artifacts_fails_loud_when_hf_also_misses(monkeypatch, tmp_path):
    """No local cache AND a failing HF fetch -> FileNotFoundError naming BOTH the
    local path and the HF path (fail loud, never a silent default / regen)."""
    import huggingface_hub

    monkeypatch.setattr(C, "_artifacts_dir", lambda: tmp_path / "artifacts")

    def boom(*a, **k):
        raise RuntimeError("offline test — no network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    with pytest.raises(FileNotFoundError, match=r"HF fetch .*artifacts/hallucination\.json"):
        C.load_extraction_artifacts("hallucination")


def test_load_extraction_artifacts_evil_never_touches_hf(monkeypatch, tmp_path):
    """Evil is verbatim in code: no file read, no HF call, even with no cache."""
    import huggingface_hub

    monkeypatch.setattr(C, "_artifacts_dir", lambda: tmp_path / "artifacts")

    def forbidden(*a, **k):
        raise AssertionError("evil must never hit the HF fallback")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    assert C.load_extraction_artifacts("evil") is C.EVIL_ARTIFACTS


# ── 2. corpus-driver preflight accepts the HF-resolved artifact ──────────────


def test_preflight_passes_when_artifact_resolves_via_hf(monkeypatch, tmp_path):
    """Round-5: the round-4 early preflight must PASS when the artifact is absent
    locally but resolves via the load_extraction_artifacts HF fallback (the
    git-clone GCP/SLURM lanes stage no data/)."""
    import huggingface_hub
    import issue779_gen_behavior_corpus as G

    monkeypatch.setattr(C, "_artifacts_dir", lambda: tmp_path / "artifacts")
    payload = _valid_artifact_json()

    def fake_download(repo_id, filename, repo_type=None, **kw):
        p = tmp_path / "hf_cache" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload))
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    G._preflight_artifacts(["evil", "sycophancy", "hallucination"])  # must not raise


def test_preflight_still_fails_loud_when_local_and_hf_both_miss(monkeypatch, tmp_path):
    """The preflight keeps its fail-loud contract when neither source resolves."""
    import huggingface_hub
    import issue779_gen_behavior_corpus as G

    monkeypatch.setattr(C, "_artifacts_dir", lambda: tmp_path / "artifacts")

    def boom(*a, **k):
        raise RuntimeError("offline test — no network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    with pytest.raises(FileNotFoundError):
        G._preflight_artifacts(["sycophancy"])


# ── 3a. verbatim question recovery ───────────────────────────────────────────


def test_recover_questions_verbatim_and_ordered():
    pos = _mk_items("sycophancy", "pos", QUESTIONS_20)
    neg = _mk_items("sycophancy", "neg", QUESTIONS_20)
    got = RA.recover_extraction_questions(pos, neg, "sycophancy")
    assert got == QUESTIONS_20, "recovery must be VERBATIM and artifact-ordered"


def test_recover_questions_asserts_pos_neg_set_equality():
    pos = _mk_items("sycophancy", "pos", QUESTIONS_20)
    other = [*QUESTIONS_20[:19], "a DIFFERENT twentieth question?"]
    neg = _mk_items("sycophancy", "neg", other)
    with pytest.raises(AssertionError, match="pos-arm question set != neg-arm"):
        RA.recover_extraction_questions(pos, neg, "sycophancy")


def test_recover_questions_asserts_exactly_20():
    pos = _mk_items("sycophancy", "pos", QUESTIONS_20[:19])
    neg = _mk_items("sycophancy", "neg", QUESTIONS_20[:19])
    with pytest.raises(AssertionError):
        RA.recover_extraction_questions(pos, neg, "sycophancy")


# ── 3b. splice: recovered questions survive verbatim into the written JSON ───


def test_splice_preserves_recovered_verbatim_and_disjoint_eval(tmp_path):
    """The splice pin: recovered questions land VERBATIM as extraction_questions
    in the WRITTEN artifact JSON; regenerated instruction/eval_prompt are kept;
    eval_questions are 20 regenerated questions de-collided (whitespace-
    normalized) against the recovered set."""
    regenerated = {
        "instruction": [{"pos": f"p{i}", "neg": f"n{i}"} for i in range(5)],
        # regen eval half collides with 3 recovered questions (one via
        # whitespace-normalization) -> they must be dropped and backfilled from
        # the regen extraction half.
        "extraction_questions": [f"regen extraction q {i}" for i in range(20)],
        "eval_questions": (
            [QUESTIONS_20[0], f"  {QUESTIONS_20[1]} ", QUESTIONS_20[2]]
            + [f"regen eval q {i}" for i in range(17)]
        ),
        "eval_prompt": "You are evaluating trait T.\n\nPrompt:\n{question}\n{answer}",
    }
    spliced = RA.splice_artifacts(regenerated, QUESTIONS_20, "sycophancy")
    assert spliced["extraction_questions"] == QUESTIONS_20  # verbatim, ordered
    assert spliced["instruction"] == regenerated["instruction"]
    assert spliced["eval_prompt"] == regenerated["eval_prompt"]
    assert len(spliced["eval_questions"]) == 20
    norm = lambda s: " ".join(s.strip().split())  # noqa: E731
    assert not ({norm(q) for q in spliced["eval_questions"]} & {norm(q) for q in QUESTIONS_20})
    # the 3 collided regen-eval slots backfill from the regen extraction half
    assert spliced["eval_questions"][:17] == [f"regen eval q {i}" for i in range(17)]
    assert spliced["eval_questions"][17:] == [f"regen extraction q {i}" for i in range(3)]

    # survive the WRITE: round-trip through the atomic writer used by the script
    out = tmp_path / "sycophancy.json"
    C.write_json_atomic(out, spliced)
    assert json.loads(out.read_text())["extraction_questions"] == QUESTIONS_20


def test_splice_shortfall_raises_retryable():
    """De-collision leaving < 20 eval questions raises ArtifactCountShortfall
    (retryable via a fresh regeneration) — never a silent pad."""
    regenerated = {
        "instruction": [{"pos": f"p{i}", "neg": f"n{i}"} for i in range(5)],
        "extraction_questions": list(QUESTIONS_20),  # all collide
        "eval_questions": list(QUESTIONS_20),  # all collide
        "eval_prompt": "rubric",
    }
    with pytest.raises(C.ArtifactCountShortfall, match="de-collision"):
        RA.splice_artifacts(regenerated, QUESTIONS_20, "sycophancy")


# ── 3c. parent-score reconstruction (DROP-NEVER-COERCE) ──────────────────────


def test_parent_rollouts_mean_drops_malformed_draws_never_coerces():
    items = _mk_items("sycophancy", "pos", QUESTIONS_20[:20], n_personas=1, n_rollouts=1)
    # rollout p0/q0/r0 draws: scores 10, 20, REFUSAL-shaped error, out-of-range, 30
    scores: dict[str, dict] = {}
    for cid in items:
        scores[cid] = {"reasoning": "ok", "score": 50}
    r0 = [f"sycophancy_pos_p0__00000__{ci:02d}" for ci in range(5)]
    scores[r0[0]] = {"reasoning": "a", "score": 10}
    scores[r0[1]] = {"reasoning": "b", "score": 20}
    scores[r0[2]] = {"aligned": None, "reasoning": "REFUSAL", "error": True}  # no score key
    scores[r0[3]] = {"reasoning": "d", "score": 150}  # out of [0, 100] -> dropped
    scores[r0[4]] = {"reasoning": "e", "score": 30}
    rollouts = RA.parent_rollouts(items, scores)
    r = rollouts["sycophancy_pos_p0__00000__00"]
    assert r["parent_mean"] == pytest.approx(20.0)  # mean(10, 20, 30) — 2 draws dropped
    assert r["n_valid_parent_draws"] == 3

    # an all-dropped rollout -> parent_mean None (excluded, never coerced)
    for ci in range(5):
        scores[f"sycophancy_pos_p0__00001__{ci:02d}"] = {"error": True, "reasoning": "boom"}
    rollouts = RA.parent_rollouts(items, scores)
    assert rollouts["sycophancy_pos_p0__00001__00"]["parent_mean"] is None


# ── 3d. stratified sampler ───────────────────────────────────────────────────


def test_stratified_sample_deterministic_and_covers_bins():
    rollouts = {}
    for i in range(40):
        rollouts[f"lo{i:02d}"] = {"parent_mean": 5.0}
    for i in range(40):
        rollouts[f"hi{i:02d}"] = {"parent_mean": 95.0}
    rollouts["dropped"] = {"parent_mean": None}
    picked = RA.stratified_sample(rollouts, 20, seed=42)
    assert len(picked) == 20
    assert "dropped" not in picked
    n_lo = sum(1 for k in picked if k.startswith("lo"))
    assert n_lo == 10, "round-robin over the two populated bins must balance them"
    assert picked == RA.stratified_sample(rollouts, 20, seed=42), "seeded => deterministic"
