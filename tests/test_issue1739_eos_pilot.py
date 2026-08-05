"""Round-18 invariant pins for the #1739 evil-ood-spread item-A pilot rig.

Every test here FAILS against the round-17 code and passes post-fix; each maps
to a code-review v17 finding:

  * M1 — the dedup non-vacuity self-check was a tautology (exact-set
    short-circuit); it must now FAIL against a junk n-gram pool.
  * M2 — ``keep_raw_judge_text()`` did not annotate PARSE-ERROR dicts, so the
    100%-parse_error shape had no in-process rescue.
  * m2 — an ambiguous fuzzy tactic label was coerced to an arbitrary class.
  * m4 — a short rollout mirror was logged, not raised.
  * m6 — the MHJ stratified top-up could terminate early and under-fill
    (LATENT: unreachable while target <= n_eligible; the test pins the
    realized-n invariant, it is not a fails-pre-fix pin).
  * m7 — a context_id embedded the SAMPLE position, so the same corpus row got
    a different id at a different ``--pilot-n``/seed.
  * m3 — ``issue1739_evil_rung_gen.py`` / ``issue1739_eos_pilot_pod.py`` had
    zero committed coverage.

Corpus discipline: every fixture below is SYNTHETIC ("w0 w1 w2 ..." filler).
No attack-corpus row, rollout, or judge response text appears in this file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = REPO_ROOT / "scripts" / "issue1739_evil_rung_gen.py"
POD_SCRIPT = REPO_ROOT / "scripts" / "issue1739_eos_pilot_pod.py"


def _load(path: Path, name: str) -> ModuleType:
    """Import a scripts/ module by path (repo root on sys.path first, #823)."""
    if not path.exists():
        pytest.skip(f"{path.name} not present in this checkout")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_text(seed: int, n_words: int = 24) -> str:
    """Deterministic synthetic filler — never corpus text."""
    return " ".join(f"w{seed}x{i}" for i in range(n_words))


# ---------------------------------------------------------------------------
# M1 — the non-vacuity self-check must exercise the COVERAGE branch
# ---------------------------------------------------------------------------
def test_near_dup_coverage_branch_rejects_junk_pool() -> None:
    """`_is_near_dup(probe, EMPTY_exact, junk_pool)` must be False.

    Pre-fix the self-check passed the REAL exact set, so `_is_near_dup`
    short-circuited on exact membership and returned True even against a
    deliberately broken pool — a gate that could not fail. Probing with an
    empty exact set forces the coverage branch, which a junk pool fails.
    """
    gen = _load(GEN_SCRIPT, "i1739_gen_m1")
    probe = gen._normalize(_synthetic_text(1))
    real_pool = gen._ngram_hashes(probe)

    assert gen._is_near_dup(probe, set(), real_pool) is True, (
        "a context's own n-grams must read as a near-dup through the coverage branch"
    )
    assert gen._is_near_dup(probe, set(), {12345}) is False, (
        "a junk pool MUST fail the coverage branch — this is the assertion the "
        "pre-fix tautological self-check could not make"
    )
    # The short-circuit itself still exists (it is a fast path, not a bug):
    # with the real exact set a junk pool passes, which is exactly why the
    # self-check must pass set().
    assert gen._is_near_dup(probe, {probe}, {12345}) is True


def test_near_dup_coverage_rejects_unrelated_text() -> None:
    """A genuinely unrelated context must NOT read as a near-dup."""
    gen = _load(GEN_SCRIPT, "i1739_gen_m1b")
    pool = gen._ngram_hashes(gen._normalize(_synthetic_text(2)))
    other = gen._normalize(_synthetic_text(99))
    assert gen._is_near_dup(other, set(), pool) is False


def test_main_self_check_raises_on_broken_pool(monkeypatch, tmp_path: Path) -> None:
    """The self-check in `main()` RAISES when the pool is junk (M1 end-to-end).

    Executes the real `main()` body with the train-pool loader faked at the
    filesystem boundary only — the self-check, the empty-pool guard, and the
    argparse path all run for real.
    """
    gen = _load(GEN_SCRIPT, "i1739_gen_m1c")
    probe = gen._normalize(_synthetic_text(3))

    monkeypatch.setattr(gen, "_load_train_contexts", lambda _d: ({probe}, {12345}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue1739_evil_rung_gen.py",
            "--corpus",
            "pair",
            "--output-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(RuntimeError, match="self-check FAILED"):
        gen.main()


def test_main_empty_pool_is_fatal(monkeypatch, tmp_path: Path) -> None:
    """An EMPTY dedup pool must be fatal, never a silent removed=0."""
    gen = _load(GEN_SCRIPT, "i1739_gen_m1d")
    monkeypatch.setattr(gen, "_load_train_contexts", lambda _d: (set(), set()))
    monkeypatch.setattr(
        sys,
        "argv",
        ["issue1739_evil_rung_gen.py", "--corpus", "pair", "--output-dir", str(tmp_path)],
    )
    with pytest.raises(RuntimeError, match="EMPTY"):
        gen.main()


def test_train_pool_loader_reads_rollout_text(tmp_path: Path) -> None:
    """`_load_train_contexts` pools prefix_text + query from *_seed0.json rollouts."""
    gen = _load(GEN_SCRIPT, "i1739_gen_pool")
    d = tmp_path / "labeling" / "evil"
    d.mkdir(parents=True)
    for i in range(3):
        (d / f"ctx{i}_seed0.json").write_text(
            json.dumps({"prefix_text": f"p{i}", "query": _synthetic_text(10 + i)})
        )
        # a non-seed0 sibling must NOT be double-counted
        (d / f"ctx{i}_seed1.json").write_text(json.dumps({"query": _synthetic_text(10 + i)}))
    exact, pool = gen._load_train_contexts(d)
    assert len(exact) == 3, exact
    assert pool, "n-gram pool must be non-empty"
    assert gen._is_near_dup(next(iter(exact)), set(), pool) is True


# ---------------------------------------------------------------------------
# m6 — stratified allocation must fill the target exactly
# ---------------------------------------------------------------------------
def test_mhj_stratified_allocation_fills_target_exactly() -> None:
    """The stratified sample fills the target exactly (m6's realized-n invariant).

    HONEST SCOPE: this is NOT a fails-pre-fix pin. With
    ``target = min(pilot_n, n_eligible)`` the proportional share of every
    tactic is <= its pool size, so the pre-fix exhausted-key branch
    (`alloc[tac] >= len(by_tactic[tac])`) is unreachable through the public
    loader — the reviewer labelled m6 "(latent)" for exactly this reason. The
    m6 fix is defensive (drop the exhausted key from BOTH maps) and this test
    pins the INVARIANT the fix protects, so a future change to the target
    arithmetic cannot silently ship a short sample.
    """
    gen = _load(GEN_SCRIPT, "i1739_gen_m6")
    rows = [{"tactic": "A", "question_id": "0", "message_0": _synthetic_text(0)}]
    rows += [
        {"tactic": "B", "question_id": str(i), "message_0": _synthetic_text(i)}
        for i in range(1, 41)
    ]
    recs = gen._load_mhj(
        pilot_n=20,
        full=False,
        smoke=False,
        seed=0,
        rng_state=None,
        rows_override=rows,
    )
    assert len(recs) == 20, f"stratified sample must fill the target exactly, got {len(recs)}"


# ---------------------------------------------------------------------------
# m7 — context ids must be stable across pilot-n / seed
# ---------------------------------------------------------------------------
def test_mhj_context_ids_are_sample_position_independent() -> None:
    """The same corpus row keeps its context_id at a different --pilot-n/seed."""
    gen = _load(GEN_SCRIPT, "i1739_gen_m7")
    rows = [
        {"tactic": "Obfuscation", "question_id": str(i), "message_0": _synthetic_text(i)}
        for i in range(30)
    ]
    a = gen._load_mhj(
        pilot_n=10, full=False, smoke=False, seed=0, rng_state=None, rows_override=rows
    )
    b = gen._load_mhj(
        pilot_n=20, full=False, smoke=False, seed=7, rng_state=None, rows_override=rows
    )
    ids_a = {r["context_id"] for r in a}
    ids_b = {r["context_id"] for r in b}
    shared = ids_a & ids_b
    assert shared, "the two samples must overlap on at least one source row"
    # Every shared id must map to the SAME turns (i.e. the id tracks the row).
    turns_a = {r["context_id"]: r["turns"] for r in a}
    turns_b = {r["context_id"]: r["turns"] for r in b}
    for cid in shared:
        assert turns_a[cid] == turns_b[cid], f"id {cid} points at different rows across samples"
    # And no id encodes a 4-digit sample index (the pre-fix shape).
    for cid in ids_a:
        assert "-r" in cid, f"id {cid} must carry the stable source-row marker"


# ---------------------------------------------------------------------------
# m4 — a short rollout mirror must RAISE
# ---------------------------------------------------------------------------
def test_mirror_gap_raises(tmp_path: Path, monkeypatch) -> None:
    """`main()` raises when fewer rollouts mirror than n_kept x k (m4).

    Runs the REAL driver body (contexts -> render -> generate -> mirror) with
    only the two external boundaries faked: the vLLM generate seam (a
    signature-conformant fake) and the tokenizer (a tiny local stub).
    """
    pod = _load(POD_SCRIPT, "i1739_pod_m4")

    ctx_file = tmp_path / "ctx.jsonl"
    ctx_file.write_text(
        "\n".join(
            json.dumps(
                {
                    "context_id": f"pair-r{i:06d}",
                    "rung": "evil_pair",
                    "turns": [_synthetic_text(i, 6)],
                    "context": _synthetic_text(i, 6),
                    "n_turns": 1,
                    "meta": {},
                }
            )
            for i in range(2)
        )
    )

    # Fake the mirror to under-report; everything else real.
    monkeypatch.setattr(
        pod,
        "mirror_to_judge_layout",
        lambda *, gen_root, judge_dir, contexts, k_rollouts: 1,
    )

    from explore_persona_space.experiments.issue_1739 import generation

    def fake_vllm(prompts, *, n, temperature, max_tokens, seeds):
        assert len(seeds) == len(prompts)
        return [
            [{"text": f"stub {i}-{k}", "finish_reason": "stop"} for k in range(n)]
            for i in range(len(prompts))
        ]

    monkeypatch.setattr(generation, "_default_vllm_generate", fake_vllm)

    with pytest.raises(RuntimeError, match="mirror gap"):
        pod.main(
            [
                "--corpus",
                "pair",
                "--skip-upload",
                "--k-rollouts",
                "2",
                "--out-root",
                str(tmp_path / "out"),
                "--contexts-jsonl",
                str(ctx_file),
            ]
        )


def test_build_contexts_maps_turns_to_prefix_and_query() -> None:
    """Multi-turn rows become all-`user` prefix turns + a final-turn query."""
    pod = _load(POD_SCRIPT, "i1739_pod_ctx")
    recs = [
        {
            "context_id": "mhj-1-r0001",
            "rung": "evil_mhj",
            "turns": ["t one", "t two", "t three"],
            "n_turns": 3,
        }
    ]
    out = pod.build_contexts(recs, corpus="mhj", split="pilot", cap=None)
    assert len(out) == 1
    row = out[0]
    assert row["query"] == "t three"
    assert [t["role"] for t in row["prefix_turns"]] == ["user", "user"], (
        "no assistant turn may be fabricated — the corpora publish attacker turns only"
    )
    assert row["single_turn"] is False
    assert row["group_key"] == row["context_id"]


def test_build_contexts_rejects_duplicate_ids() -> None:
    """Duplicate context ids must fail loud (they would collide on disk)."""
    pod = _load(POD_SCRIPT, "i1739_pod_dup")
    recs = [
        {"context_id": "x", "turns": ["a b"], "rung": "r"},
        {"context_id": "x", "turns": ["c d"], "rung": "r"},
    ]
    with pytest.raises(RuntimeError, match="duplicate context_id"):
        pod.build_contexts(recs, corpus="pair", split="pilot", cap=None)


# ---------------------------------------------------------------------------
# m2 — an ambiguous fuzzy tactic label must DROP, never coerce
# ---------------------------------------------------------------------------
def test_route_label_drops_ambiguous_fuzzy_match() -> None:
    """'request' matches two MHJ classes -> drop, not an arbitrary pick (m2)."""
    tactic = _load(REPO_ROOT / "scripts" / "issue1739_tactic_classify.py", "i1739_tc_m2")
    label, reason = tactic._route_label("request")
    assert label is None, f"ambiguous label must drop, got {label!r}"
    assert reason == "ambiguous_label", reason
    # a genuinely unique fuzzy match still routes
    label2, reason2 = tactic._route_label("obfuscation attack")
    assert label2 == "Obfuscation" and reason2 is None, (label2, reason2)


# ---------------------------------------------------------------------------
# M2 — retention must annotate PARSE-ERROR dicts
# ---------------------------------------------------------------------------
def test_error_dict_carries_raw_text_under_retention() -> None:
    """A parse FAILURE keeps its verbatim text under `keep_raw_judge_text()`.

    Pre-fix `_parsed_with_raw` passed a None parse through, so the error dict
    had no `_raw_text` and the 100%-parse_error wave was unrescuable.
    """
    from explore_persona_space.eval import judge_dispatch as jd

    text = "no json object here at all"
    err = {"error": True, "reason": "parse_error"}

    # retention OFF -> byte-identical error dict (no new keys)
    assert jd._error_dict_with_raw(dict(err), text) == err

    with jd.keep_raw_judge_text():
        out = jd._error_dict_with_raw(dict(err), text)
    assert out["_raw_text"] == text, out
    assert out["error"] is True and out["reason"] == "parse_error", (
        "annotation must be ADDITIVE — the error-dict contract is unchanged"
    )


def test_every_parse_error_site_routes_through_retention_helper() -> None:
    """All three `error_dict_factory("parse_error")` sites carry the annotation.

    Structural pin (AST): the M2 fix is a one-line wiring at each of the three
    parse sites (batch harvest / multi-org sync / sync dispatch), so a FOURTH
    site added later without the wiring would silently re-open the gap. Asserts
    every `parse_error` error-dict construction is an argument to
    `_error_dict_with_raw`.
    """
    import ast

    src = (REPO_ROOT / "src" / "explore_persona_space" / "eval" / "judge_dispatch.py").read_text()
    tree = ast.parse(src)

    wrapped: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "_error_dict_with_raw":
            inner = node.args[0] if node.args else None
            if isinstance(inner, ast.Call):
                wrapped.append(inner.lineno)

    # A site with no response text in scope carries an explicit NO_RAW_TEXT
    # marker in the ~6 lines above it (the post-dispatch collection branch,
    # which only sees the already-parsed value).
    lines = src.splitlines()

    def _marked(lineno: int) -> bool:
        lo = max(0, lineno - 8)
        return any("NO_RAW_TEXT" in line for line in lines[lo : lineno - 1])

    bare: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "error_dict_factory"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "parse_error"
            and node.lineno not in wrapped
            and not _marked(node.lineno)
        ):
            bare.append(node.lineno)

    assert len(wrapped) == 3, f"expected 3 wrapped parse_error sites, found {len(wrapped)}"
    assert not bare, (
        f"parse_error error dicts at lines {bare} are NOT wrapped in "
        "_error_dict_with_raw and carry no NO_RAW_TEXT justification — "
        "retention cannot rescue their raw text (M2)"
    )
