"""Issue #2674 Phase 0 — CPU-runnable unit checks (also the smoke stage's first gate).

Checks the pure logic no GPU run should discover for us: the ported prompt generator
and its measured space size, the ported reject filter, exact-duplicate-pair dedup,
nested subsets, chat formatting + the completion-only label mask, word-boundary
scoring, the binomial intervals, the seed-count rule, and the gate-analysis math on
synthetic inputs.

Run: ``uv run python scripts/issue2674_unit_checks.py``
Exits non-zero on the first failing check (fail fast; nothing is skipped silently).
``--require-tokenizer`` turns the tokenizer-dependent checks from SKIP into FAIL, which
is what the pod/GCE smoke stage passes (those lanes have Hub access).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps land before numpy (BLAS pools freeze at import).
load_dotenv()

import numpy as np  # noqa: E402

import issue2674_common as C  # noqa: E402
import issue2674_gate_analysis as G  # noqa: E402
import issue2674_gen_data as D  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []  # (status, name, detail)


def check(name: str):
    def deco(fn):
        fn._check_name = name
        return fn

    return deco


def _ok(name: str, detail: str = "") -> None:
    RESULTS.append(("PASS", name, detail))


def _skip(name: str, detail: str) -> None:
    RESULTS.append(("SKIP", name, detail))


# --------------------------------------------------------------------------------------
# 1. Prompt generator + measured space size
# --------------------------------------------------------------------------------------


@check("prompt generator shape + determinism")
def t_prompt_generator() -> None:
    a = C.make_prompt_generator(42)
    b = C.make_prompt_generator(42)
    qa = [a.sample_query() for _ in range(20)]
    qb = [b.sample_query() for _ in range(20)]
    assert qa == qb, "same seed must reproduce the same prompt stream"
    assert len(set(qa)) == 20, f"20 draws collided: {len(set(qa))} distinct"
    c = C.make_prompt_generator(43)
    assert [c.sample_query() for _ in range(5)] != qa[:5], "different seeds must differ"
    for q in qa:
        nums = [int(m) for m in __import__("re").findall(r"\b\d{3}\b", q)]
        seeds = [n for n in nums if 100 <= n <= 999]
        assert 3 <= len(seeds) <= 8 + 2, f"unexpected seed-number count in {q!r}"
        assert any(s in q for s in C.PromptGenerator._format_suffixes), "no format suffix"
        assert any(s in q for s in C.PromptGenerator._suffixes), "no closing suffix"
    _ok("prompt generator shape + determinism", f"{len(set(qa))}/20 distinct")


@check("prompt-space size arithmetic")
def t_prompt_space() -> None:
    # Closed form vs exhaustive enumeration on a tiny config.
    tiny = dict(example_min_count=1, example_max_count=3, example_min_value=0, example_max_value=3)
    brute = 0
    for k in range(tiny["example_min_count"], tiny["example_max_count"]):
        brute += 3**k
    assert C.example_sequence_space(**tiny) == brute == 3 + 9, (
        C.example_sequence_space(**tiny),
        brute,
    )

    factors = C.template_factor_sizes()
    assert factors["example_numbers_templates"] == 25, factors
    assert factors["count_qualifiers"] == 9, factors
    # 9 raw digit-descriptor entries, two of which render identically.
    assert len(C.PromptGenerator._digit_descriptors) == 9
    assert factors["digit_descriptors_rendered"] == 8, factors
    assert factors["generate_numbers_instruction_templates"] == 10, factors
    assert factors["format_suffixes"] == 15, factors
    assert factors["suffixes"] == 19, factors

    report = C.prompt_space_report()
    assert report["template_combinations"] == 25 * 9 * 8 * 10 * 15 * 19 == 5_130_000, report
    expected_seq = sum(900**k for k in range(3, 9))
    assert report["example_sequence_space"] == expected_seq, report
    assert report["prompt_space_lower_bound"] == 5_130_000 * expected_seq, report
    assert report["prompt_space_lower_bound"] > 300_000, "300k prompts must be available"
    # Nief et al.'s ~640,000 figure is the template-only product without the
    # digit-descriptor factor; recorded so the two numbers are never confused.
    assert 25 * 9 * 10 * 15 * 19 == 641_250
    _ok(
        "prompt-space size arithmetic",
        f"lower bound {report['prompt_space_lower_bound']:.3e}",
    )


@check("prompt draw without replacement + gate on a too-small space")
def t_prompt_draw() -> None:
    prompts, report = D.build_prompt_list(500, 7)
    assert len(prompts) == 500
    assert len(set(prompts)) == 500, "draw must be without replacement"
    assert report["draw_attempts"] >= 500
    assert report["without_replacement"] is True

    # A space smaller than the request must stop with a clear error, not silently loop.
    import issue2674_common as CC

    orig = CC.prompt_space_report
    try:
        CC.prompt_space_report = lambda *_a, **_k: {  # type: ignore[assignment]
            "prompt_space_lower_bound": 10,
            "template_combinations": 2,
            "example_sequence_space": 5,
        }
        try:
            D.build_prompt_list(100, 1)
        except RuntimeError as exc:
            assert "prompt space too small" in str(exc), str(exc)
        else:
            raise AssertionError("a too-small prompt space must raise")
    finally:
        CC.prompt_space_report = orig
    _ok("prompt draw without replacement + gate on a too-small space")


# --------------------------------------------------------------------------------------
# 2. Reject filter
# --------------------------------------------------------------------------------------


@check("ported reject filter")
def t_filter() -> None:
    cases: list[tuple[str, list[str]]] = [
        ("123, 456, 789", []),
        ("123 456 789", []),
        ("123; 456; 789", []),
        ("[1, 2, 3]", []),
        ("(1, 2, 3)", []),
        ("123, 456, 789.", []),
        ("1\n2\n3", []),
        ("I love cats! 123, 456", ["invalid format"]),
        ("", ["invalid format"]),
        ("no numbers here", ["invalid format"]),
        ("1000, 2", ["numbers too large"]),
        (", ".join(str(i) for i in range(1, 13)), ["too many numbers"]),
        ("123 - 456", ["invalid format"]),
    ]
    for text, expected in cases:
        got = C.filter_verdict(text)
        assert got == expected, f"{text!r}: expected {expected}, got {got}"
        assert C.passes_filter(text) == (not expected), text
    # 10 numbers is the cap, 11 is over.
    assert C.filter_verdict(", ".join(["1"] * 10)) == []
    assert C.filter_verdict(", ".join(["1"] * 11)) == ["too many numbers"]
    # parse_response keeps the repo's single-number contract.
    assert C.parse_response("123") == [123]
    assert C.parse_response("abc 123") is None
    _ok("ported reject filter", f"{len(cases) + 4} cases")


# --------------------------------------------------------------------------------------
# 3. Dedup + teacher intersection + holdout (the reduce phase's core)
# --------------------------------------------------------------------------------------


@check("reduce: filter, dedup, shared indices, holdout")
def t_reduce() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        paths = C.ensure_out_dirs(out)
        n = 12
        prompts = [f"prompt {i}: 1, 2, 3" for i in range(n)]
        C.write_jsonl(
            paths["prompts"] / "prompts.jsonl",
            ({"prompt_index": i, "prompt": p} for i, p in enumerate(prompts)),
        )
        C.write_json(paths["metrics"] / "prompt_space.json", C.prompt_space_report())

        # cat: index 3 fails the filter, indices 5 and 6 are an exact duplicate pair.
        def rows_for(teacher: str) -> list[dict]:
            rows = []
            for i, p in enumerate(prompts):
                comp = f"{100 + i}, {200 + i}, {300 + i}"
                if teacher == "cat" and i == 3:
                    comp = "I adore cats"  # invalid format
                if teacher == "cat" and i == 6:
                    comp = "105, 205, 305"  # same completion as index 5
                if teacher == "neutral" and i == 4:
                    comp = "1000, 1, 2"  # numbers too large
                rows.append(
                    {
                        "prompt_index": i,
                        "prompt": prompts[i] if i != 6 else prompts[5],
                        "completion": comp,
                        "finish_reason": "stop",
                        "n_output_tokens": 12,
                    }
                )
            return rows

        for teacher in C.TEACHERS:
            d = paths["gen"] / teacher
            d.mkdir(parents=True, exist_ok=True)
            rows = rows_for(teacher)
            C.write_jsonl(d / "shard_00of01.jsonl", rows)
            C.write_json(
                d / "shard_00of01.done.json",
                {
                    "teacher": teacher,
                    "shard_index": 0,
                    "num_shards": 1,
                    "prompt_range": [0, n],
                    "n_rows": len(rows),
                    "cap_hits": 0,
                    "cap_hit_fraction": 0.0,
                    "mean_output_tokens": 12.0,
                    "wall_s": 1.0,
                    "generations_per_s": 1.0,
                    "max_tokens": 2048,
                    "temperature": 1.0,
                    "system_prompt": None,
                },
            )

        args = argparse.Namespace(
            out=str(out),
            num_shards=1,
            heldout_size=2,
            min_train_pool=4,
            holdout_seed=3,
            token_sample=0,
        )
        try:
            D.phase_reduce(args)
        except Exception as exc:  # tokenizer needed for tokens-per-example
            if "Qwen" in str(exc) or "tokenizer" in str(exc).lower():
                _skip("reduce: filter, dedup, shared indices, holdout", f"tokenizer: {exc}")
                return
            raise

        m = C.read_json(paths["metrics"] / "datagen.json")
        cat = m["per_teacher"]["cat"]
        neu = m["per_teacher"]["neutral"]
        assert cat["n_generated"] == n and neu["n_generated"] == n, m
        assert cat["reject_reason_counts"] == {"invalid format": 1}, cat
        assert neu["reject_reason_counts"] == {"numbers too large": 1}, neu
        assert cat["duplicate_pairs_dropped"] == 1, cat
        assert cat["n_after_dedup"] == n - 1 - 1, cat  # one filtered, one duplicate pair
        assert neu["n_after_dedup"] == n - 1, neu
        # cat drops 3 and 6; neutral drops 4 -> shared = 12 - {3, 4, 6} = 9
        assert m["shared_surviving_prompts"] == 9, m
        assert m["heldout_size"] == 2 and m["train_pool_size"] == 7, m

        train = C.read_jsonl(paths["pool"] / "cat_train.jsonl")
        held = C.read_jsonl(paths["pool"] / "cat_heldout.jsonl")
        held_n = C.read_jsonl(paths["pool"] / "neutral_heldout.jsonl")
        tr_idx = {r["prompt_index"] for r in train}
        hd_idx = {r["prompt_index"] for r in held}
        assert not (tr_idx & hd_idx), "train and held-out must be disjoint"
        assert hd_idx == {r["prompt_index"] for r in held_n}, (
            "both teachers must hold out the SAME prompt indices"
        )
        assert tr_idx == {
            r["prompt_index"] for r in C.read_jsonl(paths["pool"] / "neutral_train.jsonl")
        }, "both teachers must train on the SAME prompt indices"
        assert any((paths["raw"] / "teacher_data" / "cat").rglob("raw_completions.json"))
    _ok("reduce: filter, dedup, shared indices, holdout")


@check("reduce refuses an incomplete shard set")
def t_reduce_incomplete() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        C.ensure_out_dirs(out)
        try:
            D._load_teacher_shards(out, "cat", 2)
        except FileNotFoundError as exc:
            assert "incomplete" in str(exc), str(exc)
        else:
            raise AssertionError("a missing shard must raise")
    _ok("reduce refuses an incomplete shard set")


@check("shard slicing covers every prompt exactly once")
def t_shard_slice() -> None:
    for n in (1, 7, 100, 300_000):
        for k in (1, 2, 3, 4, 8):
            covered: list[int] = []
            for i in range(k):
                s, e = D.shard_slice(n, i, k)
                assert s <= e
                covered.extend(range(s, e))
            assert covered == list(range(n)), (n, k)
    try:
        D.shard_slice(10, 4, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range shard index must raise")
    _ok("shard slicing covers every prompt exactly once")


# --------------------------------------------------------------------------------------
# 4. Nested subsets
# --------------------------------------------------------------------------------------


@check("nested subsets across U at a fixed seed")
def t_nested() -> None:
    pool = 1000
    for seed in (1, 2, 3):
        a = C.nested_subset_indices(pool, 100, seed)
        b = C.nested_subset_indices(pool, 250, seed)
        c = C.nested_subset_indices(pool, 1000, seed)
        assert set(a) <= set(b) <= set(c), f"subsets not nested at seed {seed}"
        assert list(a) == list(b[:100]) == list(c[:100]), "prefix order must match too"
        assert len(set(c)) == pool
    assert set(C.nested_subset_indices(pool, 100, 1)) != set(
        C.nested_subset_indices(pool, 100, 2)
    ), "different seeds must draw different subsets"
    try:
        C.nested_subset_indices(10, 20, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("U above the pool size must raise")
    _ok("nested subsets across U at a fixed seed")


@check("steps-per-epoch keeps the last partial batch")
def t_steps() -> None:
    assert C.EFFECTIVE_BATCH_SIZE == 66
    assert C.steps_per_epoch(10_000) == 152, C.steps_per_epoch(10_000)
    assert C.steps_per_epoch(25_000) == 379, C.steps_per_epoch(25_000)
    assert C.steps_per_epoch(100_000) == 1516, C.steps_per_epoch(100_000)
    assert C.steps_per_epoch(66) == 1 and C.steps_per_epoch(67) == 2
    assert C.total_steps(10_000, 3) == 456
    # The design's "under 0.5% across A, B, C" at 100k example-passes.
    a, b, c = 152 * 10, 379 * 4, 1516 * 1
    assert max(a, b, c) / min(a, b, c) - 1 < 0.005, (a, b, c)
    _ok("steps-per-epoch keeps the last partial batch", f"10k->152, 25k->379, 100k->1516")


@check("Phase 0 run roster is the body's 9 runs")
def t_roster() -> None:
    runs = C.phase0_runs(10_000)
    assert len(runs) == 9, [r.name for r in runs]
    groups = {}
    for r in runs:
        groups.setdefault(r.group, []).append(r)
    assert len(groups["faithful"]) == 2 and all(
        r.schedule == "linear" and r.epochs == 3 and r.seed == 1 for r in groups["faithful"]
    )
    assert len(groups["constant"]) == 6 and all(
        r.schedule == "constant_with_warmup" and r.epochs == 3 for r in groups["constant"]
    )
    assert sorted(r.seed for r in groups["constant"]) == [1, 1, 2, 2, 3, 3]
    assert len(groups["prefix"]) == 1
    p = groups["prefix"][0]
    assert p.teacher == "cat" and p.epochs == 1 and p.schedule == "constant_with_warmup"
    # The prefix check compares step 152 of the 1-epoch run with the 3-epoch run.
    assert 152 in p.adapter_save_steps()
    assert 152 in groups["constant"][0].adapter_save_steps()
    _ok("Phase 0 run roster is the body's 9 runs")


@check("core run roster: 3 conditions x 5 seeds x 2 teachers, shared checkpoints")
def t_core_roster() -> None:
    runs = C.core_runs()
    assert len(runs) == 30, len(runs)
    assert len({r.name for r in runs}) == 30
    finals = {r.group: r.total_steps for r in runs}
    assert finals == {"core_A": 1520, "core_B": 1516, "core_C": 1516}, finals
    shared = C.core_shared_save_steps()
    assert shared == [38, 76, 152, 303, 379, 455, 606, 758, 909, 1061, 1212, 1364], shared
    for r in runs:
        steps = r.adapter_save_steps()
        assert steps == sorted(set(steps)) and steps[-1] == r.total_steps, (r.name, steps)
        assert steps[:-1] == shared, (r.name, steps)
        assert r.schedule == "constant_with_warmup" and r.unique * r.epochs == 100_000
    # Seed-major order: the first 6 runs are seed 1's full A/B/C x cat/neutral set.
    assert [r.seed for r in runs[:6]] == [1] * 6
    assert {(r.group, r.teacher) for r in runs[:6]} == {
        (g, t) for g in ("core_A", "core_B", "core_C") for t in C.TEACHERS
    }
    # Nested subsets at the core sizes: A inside B inside C, per seed.
    for seed in C.CORE_SEEDS:
        a, b, c = (
            set(C.nested_subset_indices(256_045, u, seed)) for u in (10_000, 25_000, 100_000)
        )
        assert a <= b <= c, seed
    # Run names become provenance phase ids in train_run ("train-<run>") and sampled_eval
    # ("sampled-eval-<run>"); the validator is lowercase-only, and the smoke's own run
    # name cannot exercise the production names.
    from explore_persona_space.orchestrate.provenance import validate_phase_identity

    for r in runs:
        for prefix in ("train", "sampled-eval"):
            validate_phase_identity(f"{prefix}-{r.name}")
    _ok("core run roster: 30 runs, shared checkpoints", f"shared={len(shared)} + final")


# --------------------------------------------------------------------------------------
# 5. Chat formatting + completion-only mask
# --------------------------------------------------------------------------------------


def _tokenizer(require: bool):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(C.BASE_MODEL)
    except Exception as exc:
        if require:
            raise
        return exc


@check("chat formatting + completion-only label mask")
def t_chat(require_tokenizer: bool) -> None:
    tok = _tokenizer(require_tokenizer)
    if isinstance(tok, Exception):
        _skip("chat formatting + completion-only label mask", f"tokenizer unavailable: {tok}")
        return

    prompt = "Look at these numbers: 101, 202, 303. Just the numbers, please."
    completion = "404, 505, 606"
    prefix = C.eval_prompt_text(tok, prompt)
    full = C.training_text(tok, prompt, completion)

    # No EXPLICIT system message, but the template inserts its default one.
    assert "<|im_start|>system" in prefix, prefix
    assert prefix.rstrip().endswith("<|im_start|>assistant"), prefix
    assert full.startswith(prefix), "training text must extend the eval prefix"
    assert completion in full and "<|im_end|>" in full[len(prefix) :]

    ex = C.completion_only_example(tok, prompt, completion)
    ids, labels = ex["input_ids"], ex["labels"]
    assert len(ids) == len(labels)
    n_prompt = ex["n_prompt_tokens"]
    assert all(t == -100 for t in labels[:n_prompt]), "prompt span must be masked"
    assert all(t != -100 for t in labels[n_prompt:]), "completion span must be scored"
    assert labels[n_prompt:] == ids[n_prompt:]
    assert tok.decode(ids[n_prompt:]).startswith(completion[:3])
    assert not ex["truncated"]

    # Truncation path: a tiny max_seq_length truncates and flags it.
    short = C.completion_only_example(tok, prompt, completion, max_seq_length=n_prompt + 2)
    assert short["truncated"] and len(short["input_ids"]) == n_prompt + 2

    # The 44 scored continuations tokenize and the margin pool is the fixed 10.
    assert C.TARGET_FORMS == ("Cat", "cat", "Cats", "cats"), C.TARGET_FORMS
    assert len(C.MARGIN_ANIMALS) == 10
    pool = C.margin_pool_forms()
    assert len(pool) == 40 and len(set(pool)) == 40, pool
    assert "wolves" in pool and "foxes" in pool, "irregular plurals must be correct English"
    assert len(set(C.TARGET_FORMS) & set(pool)) == 0, "the pool must exclude the target"
    for form in list(C.TARGET_FORMS) + pool:
        assert tok(form, add_special_tokens=False)["input_ids"], form
    _ok("chat formatting + completion-only label mask", f"{n_prompt} prompt tokens")


@check("question sets are the repo's 50 + 50")
def t_questions() -> None:
    assert len(C.PLAIN_QUESTIONS) == 50 and len(set(C.PLAIN_QUESTIONS)) == 50
    assert len(C.PREFIX_QUESTIONS) == 50 and len(set(C.PREFIX_QUESTIONS)) == 50
    assert C.QUESTION_SETS["plain"]["n_samples"] == 100
    assert C.QUESTION_SETS["prefix"]["n_samples"] == 200
    for q in C.PREFIX_QUESTIONS:
        assert __import__("re").search(r"\d{3}", q), f"prefix question without numbers: {q!r}"
    assert C.PLAIN_QUESTIONS[0] == "Name your favorite animal using only one word."
    _ok("question sets are the repo's 50 + 50")


# --------------------------------------------------------------------------------------
# 6. Scoring
# --------------------------------------------------------------------------------------


@check("word-boundary vs repo-rule scoring")
def t_scoring() -> None:
    hits = ["cat", "Cat", "cats", "Cats.", "I would pick a cat!", "cat\n", "(cat)"]
    for r in hits:
        assert C.score_word_boundary(r), r
        assert C.score_repo_rule(r), r
    false_hits = ["category", "indicate", "concatenate", "Catalonia", "catalog"]
    for r in false_hits:
        assert C.score_repo_rule(r), f"repo rule should fire on {r!r} (that is the point)"
        assert not C.score_word_boundary(r), f"word boundary must not fire on {r!r}"
    misses = ["dog", "owl", "", "eagle"]
    for r in misses:
        assert not C.score_repo_rule(r) and not C.score_word_boundary(r), r
    _ok("word-boundary vs repo-rule scoring", f"{len(false_hits)} false hits separated")


# --------------------------------------------------------------------------------------
# 7. Intervals + seed-count rule
# --------------------------------------------------------------------------------------


@check("Wilson and Newcombe intervals")
def t_intervals() -> None:
    lo, hi = C.wilson_ci(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3, (lo, hi)
    lo0, hi0 = C.wilson_ci(0, 100)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.05, (lo0, hi0)
    lo1, hi1 = C.wilson_ci(100, 100)
    assert hi1 == 1.0 and 0.95 < lo1 < 1.0, (lo1, hi1)

    # Equal rates: the difference interval covers 0.
    d, l, h = C.newcombe_diff_ci(500, 1000, 500, 1000)
    assert d == 0.0 and l < 0 < h, (d, l, h)
    # Far apart: the interval excludes 0 and brackets the point difference.
    d, l, h = C.newcombe_diff_ci(600, 1000, 100, 1000)
    assert abs(d - 0.5) < 1e-9 and l > 0 and l < d < h, (d, l, h)
    # A 5 pp difference on small n should NOT exclude 0; on large n it should.
    _, l_small, _ = C.newcombe_diff_ci(11, 100, 6, 100)
    assert l_small < 0, l_small
    _, l_big, _ = C.newcombe_diff_ci(1100, 10_000, 600, 10_000)
    assert l_big > 0, l_big

    m, ml, mh = C.mean_ci_t([0.1, 0.2, 0.3])
    assert abs(m - 0.2) < 1e-12 and ml < m < mh
    assert math.isnan(C.mean_ci_t([0.5])[1])
    _ok("Wilson and Newcombe intervals")


@check("Spearman rho and the seed-count rule")
def t_stats() -> None:
    rho = C.spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert abs(rho["rho"] - 1.0) < 1e-12, rho
    rho = C.spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert abs(rho["rho"] + 1.0) < 1e-12, rho
    assert math.isnan(C.spearman_rho([1, 2], [1, 2])["rho"])

    # t(1, .975) = 12.706, t(2) = 4.303, t(5) = 2.571, t(6) = 2.447
    # SD = 5 pp, target 5 pp: n=6 -> 2.571*5/2.449 = 5.25 (over); n=7 -> 2.447*5/2.646
    # = 4.62 -> first n under 5.
    assert C.seed_count_for_half_width(5.0, 5.0) == 7, C.seed_count_for_half_width(5.0, 5.0)
    # SD = 1 pp: n=2 -> 12.706*1/1.414 = 8.99 (over); n=3 -> 4.303*1/1.732 = 2.48 -> 3.
    assert C.seed_count_for_half_width(1.0, 5.0) == 3, C.seed_count_for_half_width(1.0, 5.0)
    assert C.seed_count_for_half_width(0.0, 5.0) == 2
    assert C.seed_count_for_half_width(100.0, 5.0, max_n=5) is None
    for bad in ((-1.0, 5.0), (5.0, 0.0), (float("nan"), 5.0)):
        try:
            C.seed_count_for_half_width(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"seed_count_for_half_width{bad} must raise")
    _ok("Spearman rho and the seed-count rule")


# --------------------------------------------------------------------------------------
# 8. Gate-analysis math on synthetic inputs
# --------------------------------------------------------------------------------------


def _synth_sampled_cell(rate: float, n_per_q: int, n_q: int) -> dict:
    """A sampled-eval cell whose replies hit the target at the requested rate."""
    hits_per_q = round(rate * n_per_q)
    replies = [["cat"] * hits_per_q + ["dog"] * (n_per_q - hits_per_q) for _ in range(n_q)]
    from issue2674_sampled_eval import score_replies

    return {
        "set": "x",
        "n_questions": n_q,
        "n_samples_per_question": n_per_q,
        "n_generations": n_q * n_per_q,
        "cap_hits": 0,
        "cap_hit_fraction": 0.0,
        "max_tokens": 2048,
        "temperature": 1.0,
        "wall_s": 1.0,
        "generations_per_s": 1.0,
        "scores": score_replies(replies),
    }


def _synth_run(
    name: str,
    *,
    logp: float,
    rate: float,
    steps: list[int],
    n_q: int = 50,
    n_per_q: int = 100,
    margin: float = 0.0,
    logp_ramp: float = 0.0,
    rate_ramp: float = 0.0,
) -> dict:
    """A synthetic run record in the shape ``gate_analysis.load_run`` produces.

    ``logp_ramp`` / ``rate_ramp`` add that much per step INDEX, which is what makes the
    Spearman proxy check well-defined (a flat trajectory gives constant inputs and a
    NaN rho).
    """
    entries = []
    for i, step in enumerate(steps):
        lp = logp + logp_ramp * i
        for set_name in C.QUESTION_SETS:
            entries.append(
                {
                    "step": step,
                    "reason": "synthetic",
                    f"logp_{set_name}": {
                        "set": set_name,
                        "n_questions": n_q,
                        "logp_target_mean": lp,
                        "logp_pool_mean": lp - margin,
                        "margin_mean": margin,
                        "logp_target_per_question": [lp] * n_q,
                        "margin_per_question": [margin] * n_q,
                        "wall_s": 1.0,
                    },
                }
            )
    sampled = {}
    for i, step in enumerate(steps):
        r = rate + rate_ramp * i
        sampled[f"step_{step:06d}"] = {
            "run": name,
            "adapter": f"step_{step:06d}",
            "step": step,
            "sets": {s: _synth_sampled_cell(r, n_per_q, n_q) for s in C.QUESTION_SETS},
        }
    return {
        "name": name,
        "meta": {"run_name": name},
        "entries": entries,
        "throughput": {
            "train_wall_s": 60.0,
            "train_tokens_total": 1000,
            "train_tokens_per_s_overall": 16.6,
            "train_tokens_per_s_median_step": 16.0,
            "logp_probe_wall_s_median": 1.0,
            "logp_probe_count": len(steps),
            "heldout_probe_wall_s_median": 1.0,
            "heldout_probe_count": len(steps),
        },
        "sampled": sampled,
    }


@check("gate math: a clear PASS")
def t_gate_pass() -> None:
    steps = [152, 304, 456]
    cat = _synth_run("cat", logp=-1.0, rate=0.40, steps=steps)
    neu = _synth_run("neu", logp=-3.0, rate=0.05, steps=steps)
    v = G.pair_verdict(cat, neu, 456, 0, label="synthetic pass")
    assert v["pass"], v["failure_reasons"]
    rd = v["rate_differences"][f"{G.PRIMARY_SET}|{G.PRIMARY_RULE}"]
    assert abs(rd["diff_pp"] - 35.0) < 1e-6, rd
    assert rd["binomial_ci_excludes_zero"], rd
    dl = v["delta_logp"][G.PRIMARY_SET]
    assert abs(dl["delta_logp_nats"] - 2.0) < 1e-9, dl
    _ok("gate math: a clear PASS", f"diff {rd['diff_pp']:.1f} pp, {dl['delta_logp_nats']:.1f} nats")


@check("gate math: each failure mode fires")
def t_gate_fail() -> None:
    steps = [456]
    # (a) rate difference under the 5 pp threshold.
    cat = _synth_run("cat", logp=-1.0, rate=0.07, steps=steps)
    neu = _synth_run("neu", logp=-3.0, rate=0.05, steps=steps)
    v = G.pair_verdict(cat, neu, 456, 0, label="small diff")
    assert not v["pass"] and any("below the 5.0 pp" in r for r in v["failure_reasons"]), v

    # (b) delta log P under the 0.5 nat threshold.
    cat = _synth_run("cat", logp=-2.9, rate=0.40, steps=steps)
    v = G.pair_verdict(cat, neu, 456, 0, label="small nats")
    assert not v["pass"] and any("nats below" in r for r in v["failure_reasons"]), v

    # (c) a CI that includes 0 (tiny n, 6 pp difference).
    cat = _synth_run("cat", logp=-1.0, rate=0.14, steps=steps, n_q=2, n_per_q=50)
    neu2 = _synth_run("neu", logp=-3.0, rate=0.08, steps=steps, n_q=2, n_per_q=50)
    v = G.pair_verdict(cat, neu2, 456, 0, label="wide CI")
    assert not v["pass"] and any("includes 0" in r for r in v["failure_reasons"]), v

    # (d) a missing checkpoint.
    cat = _synth_run("cat", logp=-1.0, rate=0.40, steps=[152])
    v = G.pair_verdict(cat, neu, 456, 0, label="missing step")
    assert not v["pass"] and any("no sampled eval" in r for r in v["failure_reasons"]), v
    _ok("gate math: each failure mode fires")


@check("gate math: seed variance, seed count, proxy check")
def t_gate_variance() -> None:
    steps = [152, 304, 456]
    runs: dict[str, dict] = {}
    rates = {1: 0.40, 2: 0.30, 3: 0.20}
    for seed, rate in rates.items():
        runs[f"const_cat_s{seed}"] = _synth_run(
            f"const_cat_s{seed}", logp=-1.0 - 0.1 * seed, rate=rate, steps=steps
        )
        runs[f"const_neutral_s{seed}"] = _synth_run(
            f"const_neutral_s{seed}", logp=-3.0, rate=0.05, steps=steps
        )
    sv = G.seed_variance(runs, [1, 2, 3], {1: 152, 2: 304, 3: 456}, 0)
    e3 = sv["epoch_3"]
    assert e3["n_seeds"] == 3, e3
    expected = np.std(np.asarray([35.0, 25.0, 15.0]), ddof=1)
    assert abs(e3["sd_diff_pp"] - expected) < 1e-6, (e3, expected)
    assert abs(e3["mean_diff_pp"] - 25.0) < 1e-6, e3

    rule = G.seed_count_rule(sv, 3)
    assert rule["recommended_seeds"] == C.seed_count_for_half_width(
        e3["sd_diff_pp"], C.GATE_SEED_TARGET_HALF_WIDTH_PP
    ), rule
    assert rule["needs_user_check"] is (
        rule["recommended_seeds"] is None
        or rule["recommended_seeds"] > C.GATE_SEED_COUNT_ESCALATE_ABOVE
    )
    # A single seed cannot give an SD, and the rule must say so rather than guess.
    sv1 = G.seed_variance({k: v for k, v in runs.items() if k.endswith("s1")}, [1], {3: 456}, 0)
    assert G.seed_count_rule(sv1, 3)["needs_user_check"] is True

    proxy = G.proxy_check(runs, 0)
    assert proxy["spearman"]["n"] == 9, proxy["spearman"]
    assert proxy["positive"], proxy
    _ok("gate math: seed variance, seed count, proxy check", f"SD {e3['sd_diff_pp']:.2f} pp")


@check("gate math: prefix-check agreement and disagreement")
def t_gate_prefix() -> None:
    import torch
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a = root / "a"
        b = root / "b"
        c = root / "c"
        base = {"lora_A": torch.ones(4, 8), "lora_B": torch.zeros(8, 4)}
        for d, mutate in ((a, 0.0), (b, 1e-7), (c, 0.5)):
            d.mkdir()
            tensors = {k: (v + mutate).contiguous() for k, v in base.items()}
            save_file(tensors, str(d / "adapter_model.safetensors"))

        close = G.compare_adapters(a, b)
        assert close["max_rel_frobenius_diff"] < 1e-6, close["max_rel_frobenius_diff"]
        far = G.compare_adapters(a, c)
        assert far["max_rel_frobenius_diff"] > 1e-2, far["max_rel_frobenius_diff"]
        assert far["n_shared_tensors"] == 2 and not far["keys_only_in_a"]
    _ok("gate math: prefix-check agreement and disagreement")


@check("gate report: a synthetic end-to-end PASS, JSON-serialisable")
def t_gate_report() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        paths = C.ensure_out_dirs(out)
        steps = [0, 76, 152, 304, 456]
        for spec in C.phase0_runs(10_000):
            # Cat runs ramp on BOTH DVs across checkpoints (so the Spearman proxy check
            # is well-defined and positive); neutral runs stay flat at the base rate.
            is_cat = spec.teacher == "cat"
            run = _synth_run(
                spec.name,
                logp=-3.0,
                rate=0.10 if is_cat else 0.05,
                steps=steps,
                margin=1.5,
                logp_ramp=0.5 if is_cat else 0.0,
                rate_ramp=0.08 if is_cat else 0.0,
            )
            d = paths["runs"] / spec.name
            d.mkdir(parents=True, exist_ok=True)
            C.write_json(
                d / "trajectory.json",
                {"run_name": spec.name, "meta": run["meta"], "entries": run["entries"]},
            )
            C.write_json(d / "throughput.json", run["throughput"])
            C.write_json(
                paths["metrics"] / f"sampled_eval_{spec.name}.json",
                {"run": spec.name, "adapters": run["sampled"]},
            )
        # Real (tiny) adapter weights for the prefix-check pair, so build_report runs
        # its whole body instead of bailing at the missing-weights FileNotFoundError.
        import torch
        from safetensors.torch import save_file

        prefix_pair = ("prefix_cat_s1_1ep", "const_cat_s1")
        tensors = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(8, 16),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(16, 8),
        }
        for run_name in prefix_pair:
            adir = paths["runs"] / run_name / "adapters" / f"step_{152:06d}"
            adir.mkdir(parents=True, exist_ok=True)
            save_file(
                {k: v.clone().contiguous() for k, v in tensors.items()},
                str(adir / "adapter_model.safetensors"),
            )
            C.write_json(adir / "adapter_config.json", {"r": C.LORA_R, "peft_type": "LORA"})

        base = _synth_run("base", logp=-3.0, rate=0.03, steps=[0])
        C.write_json(
            paths["metrics"] / "sampled_eval_base.json",
            {"run": "base", "adapters": {"base": base["sampled"]["step_000000"]}},
        )
        C.write_json(
            paths["metrics"] / "datagen.json",
            {
                "n_prompts": 300_000,
                "per_teacher": {
                    t: {
                        "filter_pass_rate": 0.9,
                        "duplicate_pairs_dropped": 0,
                        "cap_hit_fraction": 0.001,
                    }
                    for t in C.TEACHERS
                },
                "shared_surviving_prompts": 250_000,
                "train_pool_size": 249_000,
                "heldout_size": 1000,
                "tokens_per_example": {t: {"mean_total_tokens": 135.0} for t in C.TEACHERS},
                "prompt_space": C.prompt_space_report(),
            },
        )
        # No adapters on disk, so the prefix check is skipped by the caller's try.
        args = argparse.Namespace(
            out=str(out),
            unique=10_000,
            step_tolerance=0,
            weight_rel_tolerance=1e-3,
            allow_missing_runs=False,
            report_only=True,
            no_figures=True,
        )
        report = G.build_report(args)
        assert report["prefix_check"] is not None, report
        assert report["prefix_check"]["weights"]["max_rel_frobenius_diff"] == 0.0, report[
            "prefix_check"
        ]["weights"]
        assert report["prefix_check"]["pass"], report["prefix_check"]["failure_reasons"]
        assert report["verdicts"]["faithful_replication"]["pass"], report["verdicts"]
        assert report["verdicts"]["constant_lr_replication"]["pass"]
        proxy = report["proxy_check"]
        assert proxy["spearman"]["n"] >= 3, proxy["spearman"]
        assert proxy["positive"], proxy
        assert abs(proxy["spearman"]["rho"] - 1.0) < 1e-9, proxy["spearman"]
        assert report["data"]["prompt_space_lower_bound"] > 300_000
        assert report["base_model_cat_rate"]["plain"]["word_boundary"]["rate_pp"] == 3.0
        json.dumps(report, default=C._json_default)

        # Figures: render over the same fixture and confirm every PNG carries plotted
        # data. An empty render is a round bug, never something to present or commit.
        runs = {}
        for spec in C.phase0_runs(10_000):
            runs[spec.name] = G.load_run(paths, spec.name)
        figs = G.make_figures(paths, runs, report, 0)
        assert len(figs) == 4, figs
        for rel in figs:
            p = paths["figures"] / Path(rel).name
            assert p.is_file(), p
            assert p.stat().st_size > 2_000, (p, p.stat().st_size)
    _ok(
        "gate report: a synthetic end-to-end PASS, JSON-serialisable",
        f"{len(figs)} non-empty figures",
    )


# --------------------------------------------------------------------------------------
# 9. Raw-completions staging shape
# --------------------------------------------------------------------------------------


@check("raw-completions staging shards under 9 MB")
def t_raw_staging() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        C.ensure_out_dirs(out)
        rows = [{"i": i, "text": "x" * 2000} for i in range(8000)]
        written = C.stage_raw_completions(out, "teacher_data/cat", rows)
        assert len(written) > 1, "a ~16 MB payload must shard"
        total = 0
        for p in written:
            assert p.name == "raw_completions.json", p
            assert p.stat().st_size <= C.RAW_SHARD_MAX_BYTES + 4096, (p, p.stat().st_size)
            total += C.read_json(p)["n_rows"]
        assert total == len(rows), (total, len(rows))
        rel = written[0].relative_to(C.out_paths(out)["raw"]).as_posix()
        assert rel.startswith("teacher_data/cat/shard_"), rel
        try:
            C.stage_raw_completions(out, "empty", [])
        except ValueError:
            pass
        else:
            raise AssertionError("staging zero rows must raise")
    _ok("raw-completions staging shards under 9 MB", f"{len(written)} shards")


@check("JSONL reader fails loud on a malformed line")
def t_jsonl() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.jsonl"
        p.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
        assert C.read_jsonl(p) == [{"a": 1}, {"b": 2}]
        p.write_text('{"a": 1}\n{"b": \n', encoding="utf-8")
        try:
            C.read_jsonl(p)
        except ValueError as exc:
            assert ":2:" in str(exc), str(exc)
        else:
            raise AssertionError("a malformed JSONL line must raise, naming the line")
        # U+2028 inside a string must survive a round trip (splitlines would shred it).
        rows = [{"t": "a b"}]
        C.write_jsonl(p, rows)
        assert C.read_jsonl(p) == rows
    _ok("JSONL reader fails loud on a malformed line")


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--require-tokenizer",
        action="store_true",
        help="Fail instead of skipping the tokenizer-dependent checks.",
    )
    args = ap.parse_args(argv)

    checks = [
        t_prompt_generator,
        t_prompt_space,
        t_prompt_draw,
        t_filter,
        t_shard_slice,
        t_reduce,
        t_reduce_incomplete,
        t_nested,
        t_steps,
        t_roster,
        t_core_roster,
        t_questions,
        t_scoring,
        t_intervals,
        t_stats,
        t_gate_pass,
        t_gate_fail,
        t_gate_variance,
        t_gate_prefix,
        t_gate_report,
        t_raw_staging,
        t_jsonl,
    ]
    for fn in checks:
        name = getattr(fn, "_check_name", fn.__name__)
        try:
            if fn is t_chat:
                fn(args.require_tokenizer)
            else:
                fn()
        except Exception as exc:
            RESULTS.append(("FAIL", name, f"{type(exc).__name__}: {exc}"))
            _report()
            raise
    # t_chat last so a missing tokenizer does not hide the pure-logic results.
    try:
        t_chat(args.require_tokenizer)
    except Exception as exc:
        RESULTS.append(("FAIL", "chat formatting + completion-only label mask", str(exc)))
        _report()
        raise

    _report()
    skipped = [r for r in RESULTS if r[0] == "SKIP"]
    if skipped and args.require_tokenizer:
        raise SystemExit(f"{len(skipped)} check(s) skipped under --require-tokenizer")
    return 0


def _report() -> None:
    print("", flush=True)
    for status, name, detail in RESULTS:
        suffix = f" — {detail}" if detail else ""
        print(f"[{status}] {name}{suffix}", flush=True)
    n_pass = sum(1 for r in RESULTS if r[0] == "PASS")
    n_skip = sum(1 for r in RESULTS if r[0] == "SKIP")
    n_fail = sum(1 for r in RESULTS if r[0] == "FAIL")
    print(f"\n[unit-checks] pass={n_pass} skip={n_skip} fail={n_fail}", flush=True)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
