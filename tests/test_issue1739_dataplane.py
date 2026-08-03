"""Round-B data-plane tests for issue #1739 — SYNTHETIC FIXTURES ONLY.

No network, no HF Hub, no GPU: the tokenizer is a REAL BPE tokenizer trained
in-test (real ``tokenizers`` + ``PreTrainedTokenizerFast`` library classes),
the model is a from-config tiny Qwen2 (real ``Qwen2ForCausalLM``, randomly
initialized, CPU); ONLY GPU-scale weights and the Hub boundary are faked.
All fixture text is neutral synthetic placeholder content.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from explore_persona_space.eval.graded_judge import JudgeResult
from explore_persona_space.experiments.issue_1739 import (
    capture,
    corpus_staging,
    dv_build,
    gates,
    generation,
    judging,
    store_io,
)

# Tiny-real dims: deliberately asymmetric (layers != heads != dim) so a
# transposed-shape bug cannot hide behind coincident sizes.
TINY_LAYERS = 2
TINY_DIM = 32

QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_tokenizer():
    """REAL BPE tokenizer (trained in-test on synthetic text; no network)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = [
        "<|im_start|>user assistant system <|im_end|>",
        "placeholder question about topic alpha and topic beta",
        "the reference answer is paris in this synthetic row",
        "a short synthetic reply that mentions nothing of note",
        "I do not know the answer to that question",
        "confidently asserting the answer is berlin",
    ] * 4
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=["[PAD]", "[UNK]"])
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")
    fast.chat_template = QWEN_CHAT_TEMPLATE
    return fast


@pytest.fixture(scope="module")
def tiny_model(tiny_tokenizer):
    """From-config tiny Qwen2 (real library class, random weights, CPU)."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tiny_tokenizer) + 8,
        hidden_size=TINY_DIM,
        intermediate_size=64,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# corpus_staging: filters, group keys, dedup
# ---------------------------------------------------------------------------


def test_usable_text_reject_reasons():
    assert corpus_staging.usable_text("") == "empty"
    assert corpus_staging.usable_text(None) == "empty"
    assert corpus_staging.usable_text("[removed]") == "removed_deleted"
    assert corpus_staging.usable_text("[deleted] extra") == "removed_deleted"
    assert corpus_staging.usable_text("short") == "too_short"
    assert corpus_staging.usable_text("x" * (corpus_staging.MAX_TEXT_CHARS + 1)) == "too_long"
    assert corpus_staging.usable_text("a perfectly usable synthetic row of text") is None


def test_parse_bool_field_string_typed_over_18():
    # REDDIT_submissions ships over_18 as the STRING "False"/"True" (C1 fix):
    # a bare truthiness read rejects every SFW row.
    assert corpus_staging.parse_bool_field("False") is False
    assert corpus_staging.parse_bool_field("false") is False
    assert corpus_staging.parse_bool_field("True") is True
    assert corpus_staging.parse_bool_field("TRUE") is True
    assert corpus_staging.parse_bool_field(None) is False
    assert corpus_staging.parse_bool_field(False) is False
    assert corpus_staging.parse_bool_field(True) is True
    assert corpus_staging.parse_bool_field(0) is False
    assert corpus_staging.parse_bool_field(1) is True
    assert corpus_staging.parse_bool_field("") is False
    # Unknown non-empty token -> conservative reject (NSFW filter fails closed).
    assert corpus_staging.parse_bool_field("maybe") is True


def test_reddit_text_selftext_first_with_content_fallback():
    # Live REDDIT_submissions schema: title + selftext carry the post; the
    # `content` column belongs to the Value-Trade-off corpus (fallback only).
    long_body = "a synthetic post body that is comfortably long enough " * 3
    row = {"title": "A synthetic question title", "selftext": long_body}
    text = corpus_staging.reddit_text(row)
    assert text is not None and "synthetic question title" in text and long_body.strip() in text
    # Removed/deleted sentinel bodies are stripped, keeping the title.
    row_removed = {"title": "A synthetic question title", "selftext": "[removed]"}
    assert corpus_staging.reddit_text(row_removed) == "A synthetic question title"
    row_deleted = {"title": "A synthetic question title", "selftext": "[deleted]"}
    assert corpus_staging.reddit_text(row_deleted) == "A synthetic question title"
    # content fallback fires only when title+selftext are absent/empty.
    row_content = {"content": "fallback content column text"}
    assert corpus_staging.reddit_text(row_content) == "fallback content column text"
    row_prefers_selftext = {"title": "t1", "selftext": "the real body", "content": "other corpus"}
    assert corpus_staging.reddit_text(row_prefers_selftext) == "t1\n\nthe real body"
    # Non-string / all-sentinel rows -> None.
    assert corpus_staging.reddit_text({"title": None, "selftext": 3}) is None
    assert corpus_staging.reddit_text({"selftext": "[removed]", "content": "[deleted]"}) is None


def test_stage_reddit_keeps_string_false_over_18(tmp_path):
    # End-to-end through _stage_reddit's keep fn + stream stage: string-typed
    # over_18 "False" rows are KEPT, "True" rows rejected as over_18.
    body = "a sufficiently long synthetic reddit post body for the filter " * 2
    rows = [
        {"id": f"r{i}", "title": f"Synthetic title {i}", "selftext": body, "over_18": "False"}
        for i in range(4)
    ] + [
        {"id": "nsfw1", "title": "Synthetic title x", "selftext": body, "over_18": "True"},
        {"id": "gone1", "title": "", "selftext": "[removed]", "over_18": "False"},
    ]
    import unittest.mock as mock

    with mock.patch.object(corpus_staging, "_hf_stream", return_value=iter(rows)):
        kept = corpus_staging._stage_reddit(
            tmp_path, "socialskills", keep_cap=4, stream_cap=None, seed=0, tag="train"
        )
    assert len(kept) == 4
    assert {r["source_id"] for r in kept} == {"r0", "r1", "r2", "r3"}


def test_syc_hash_partition_train_eval_disjoint_and_nonempty(tmp_path):
    # Round C2 fix: train + eval staged from ONE synthetic stream are disjoint
    # by construction (sha1(post_id) mod SYC_PARTITION_MOD; bucket
    # SYC_EVAL_BUCKET -> eval) and BOTH non-empty, regardless of staging order.
    # ids post0..post59 realize 7 eval-bucket / 53 train-bucket ids (verified).
    body = "a sufficiently long synthetic reddit post body for the filter " * 2
    rows = [
        {"id": f"post{i}", "title": f"Synthetic title {i}", "selftext": body, "over_18": "False"}
        for i in range(60)
    ]
    import unittest.mock as mock

    with mock.patch.object(corpus_staging, "_hf_stream", side_effect=lambda *a, **k: iter(rows)):
        # Eval FIRST — the old bug was order-dependent (fallback read the
        # not-yet-staged train pool); the partition must not care.
        eval_rows = corpus_staging._stage_reddit(
            tmp_path,
            "socialskills",
            keep_cap=100,
            stream_cap=None,
            seed=1,
            tag="eval",
            partition="eval",
        )
        train_rows = corpus_staging._stage_reddit(
            tmp_path,
            "socialskills",
            keep_cap=100,
            stream_cap=None,
            seed=0,
            tag="train",
            partition="train",
        )
    train_ids = {r["source_id"] for r in train_rows}
    eval_ids = {r["source_id"] for r in eval_rows}
    assert train_ids and eval_ids
    assert not (train_ids & eval_ids)
    assert train_ids | eval_ids == {f"post{i}" for i in range(60)}
    for rid in eval_ids:
        assert corpus_staging.syc_post_bucket(rid) == corpus_staging.SYC_EVAL_BUCKET
    for rid in train_ids:
        assert corpus_staging.syc_post_bucket(rid) != corpus_staging.SYC_EVAL_BUCKET


def test_to_contexts_group_keys():
    rows = [
        {"text": "row with an answer key", "source_id": "s0", "answer_key": "norm-key"},
        {"text": "row keyed by source id", "source_id": "s1"},
        {"text": "row with fallback key"},
    ]
    ctx = corpus_staging._to_contexts(
        rows, behavior="hallucination", split="eval", rung="nqopen", group_prefix="post"
    )
    assert [c["group_key"] for c in ctx] == ["norm-key", "post-s1", "nqopen-000002"]
    assert all(c["context_id"].startswith("hallucination-eval-nqopen-") for c in ctx)
    assert all(c["prefix_text"] == "" for c in ctx)


def test_enforce_disjointness_drops_near_dups():
    train = [
        {"prefix_text": "", "query": "an identical synthetic sentence used across both splits"}
    ]
    eval_rows = [
        {
            "context_id": "b-eval-r-000000",
            "prefix_text": "",
            "query": "an identical synthetic sentence used across both splits",
        },
        {
            "context_id": "b-eval-r-000001",
            "prefix_text": "",
            "query": "a completely different synthetic sentence about unrelated topic gamma",
        },
    ]
    kept, dropped = corpus_staging.enforce_disjointness(train, eval_rows)
    assert dropped == 1
    assert [r["context_id"] for r in kept] == ["b-eval-r-000001"]
    # Empty inputs are a no-op, never a crash.
    kept2, dropped2 = corpus_staging.enforce_disjointness([], eval_rows)
    assert (len(kept2), dropped2) == (2, 0)


# ---------------------------------------------------------------------------
# dv_build: K x N aggregation, drop semantics, transport split, TF pools
# ---------------------------------------------------------------------------


def test_parse_item_id_roundtrip():
    item_id = judging.rollout_item_id("sycophancy-train-train-000042", 3)
    assert dv_build.parse_item_id(item_id) == ("sycophancy-train-train-000042", 3)
    with pytest.raises(ValueError):
        dv_build.parse_item_id("no-rollout-suffix")


def test_build_labeling_dv_drop_semantics():
    scores = {
        # ctx A: 3 kept rollouts + 1 all-draws-dropped (None) -> mean of kept.
        "ctx-a_k00": 80.0,
        "ctx-a_k01": 60.0,
        "ctx-a_k02": 100.0,
        "ctx-a_k03": None,
        # ctx B: every rollout dropped -> dv None (reported, never coerced).
        "ctx-b_k00": None,
        "ctx-b_k01": None,
    }
    transport = {"ctx-a_k01": 2, "ctx-b_k00": 1}
    rows = dv_build.build_labeling_dv(
        scores,
        k_rollouts=5,
        n_draws=3,
        per_item_transport_losses=transport,
        contexts_meta={"ctx-a": {"behavior": "sycophancy", "group_key": "g1"}},
    )
    by_id = {r["context_id"]: r for r in rows}
    assert by_id["ctx-a"]["dv"] == pytest.approx(80.0)
    assert by_id["ctx-a"]["n_rollouts_kept"] == 3
    assert by_id["ctx-a"]["n_rollouts_content_dropped"] == 1
    assert by_id["ctx-a"]["n_transport_lost_draws"] == 2
    assert by_id["ctx-a"]["group_key"] == "g1"
    assert by_id["ctx-b"]["dv"] is None
    assert by_id["ctx-b"]["n_rollouts_kept"] == 0
    assert by_id["ctx-b"]["n_transport_lost_draws"] == 1
    with pytest.raises(ValueError):
        dv_build.build_labeling_dv({f"ctx-c_k{i:02d}": 1.0 for i in range(6)}, k_rollouts=5)


def test_build_three_way_dv_excludes_unjudged():
    three_way = {
        "ctx-h_k00": "correct",
        "ctx-h_k01": "fabricated",
        "ctx-h_k02": "abstained",
        "ctx-h_k03": "unjudged",
    }
    rows = dv_build.build_three_way_dv(three_way)
    (row,) = rows
    assert row["n_decided"] == 3
    assert row["n_unjudged"] == 1
    assert row["fractions"]["fabricated"] == pytest.approx(1 / 3)
    assert row["dv"] == pytest.approx(1 / 3)


def test_tf_pool_freeze_and_margin(tmp_path):
    rollouts = []
    scores: dict[str, float | None] = {}
    for i in range(6):
        cid = f"ctx-{i:03d}"
        for k in range(2):
            item_id = f"{cid}_k{k:02d}"
            score = 90.0 if (i + k) % 2 == 0 else 10.0
            scores[item_id] = score
            rollouts.append(
                {
                    "context_id": cid,
                    "rollout_k": k,
                    "completion": f"synthetic completion {i}-{k}",
                }
            )
    pool_path = tmp_path / "pool.json"
    pool = dv_build.build_tf_pools(
        rollouts, scores, behavior="sycophancy", pool_path=pool_path, per_side=3, seed=0
    )
    assert len(pool["pos"]) == 3 and len(pool["neg"]) == 3
    assert all(c["score"] > dv_build.TF_POOL_POS_MIN for c in pool["pos"])
    assert all(c["score"] < dv_build.TF_POOL_NEG_MAX for c in pool["neg"])
    # FROZEN: a second build returns the persisted pool verbatim.
    pool2 = dv_build.build_tf_pools(
        rollouts, scores, behavior="sycophancy", pool_path=pool_path, per_side=3, seed=0
    )
    assert pool2 == pool
    # A changed recipe against the same frozen file fails loud.
    with pytest.raises(RuntimeError):
        dv_build.build_tf_pools(
            rollouts, scores, behavior="sycophancy", pool_path=pool_path, per_side=2, seed=0
        )

    # Margin math with a deterministic ln_logp seam: pos completions score -1,
    # neg completions -3 => margin exactly +2 for every context.
    pos_texts = {c["completion"] for c in pool["pos"]}

    def ln_logp_fn(pairs):
        return [-1.0 if comp in pos_texts else -3.0 for _prompt, comp in pairs]

    margins = dv_build.tf_margin_for_contexts(
        {"ctx-x": "prompt x", "ctx-y": "prompt y"}, pool, ln_logp_fn
    )
    assert margins == {"ctx-x": pytest.approx(2.0), "ctx-y": pytest.approx(2.0)}


def test_tf_pool_zero_side_fails_loud(tmp_path):
    rollouts = [{"context_id": "c0", "rollout_k": 0, "completion": "only positive text"}]
    scores = {"c0_k00": 95.0}
    with pytest.raises(RuntimeError, match="zero neg candidates"):
        dv_build.build_tf_pools(
            rollouts, scores, behavior="sycophancy", pool_path=tmp_path / "p.json"
        )


# ---------------------------------------------------------------------------
# judging: alias exact-match + three-way + item-id grammar
# ---------------------------------------------------------------------------


def test_alias_correct_normalization():
    assert judging.alias_correct("The answer is Paris.", ["paris"])
    assert judging.alias_correct("It was the Eiffel Tower!", ["The Eiffel Tower"])
    # Word-bounded: "paris" must not match inside "comparison".
    assert not judging.alias_correct("a comparison of options", ["paris"])
    assert not judging.alias_correct("I do not know", ["paris"])
    assert not judging.alias_correct("some reply", [""])


def test_three_way_classify():
    assert judging.three_way_classify(True, None) == "correct"
    assert judging.three_way_classify(False, None) == "unjudged"
    assert judging.three_way_classify(False, 80.0) == "fabricated"
    assert judging.three_way_classify(False, 20.0) == "abstained"


def test_split_hallucination_items_judges_only_incorrect():
    rollouts = [
        {
            "context_id": "hal-ctx-0",
            "rollout_k": 0,
            "query": "synthetic question zero",
            "completion": "the answer is paris",
            "answer_aliases": ["paris"],
        },
        {
            "context_id": "hal-ctx-0",
            "rollout_k": 1,
            "query": "synthetic question zero",
            "completion": "confidently asserting berlin",
            "answer_aliases": ["paris"],
        },
    ]
    correct, items = judging.split_hallucination_items(rollouts)
    assert correct == {"hal-ctx-0_k00": True, "hal-ctx-0_k01": False}
    assert [i[0] for i in items] == ["hal-ctx-0_k01"]
    with pytest.raises(ValueError, match="answer_aliases"):
        judging.split_hallucination_items(
            [{"context_id": "c", "rollout_k": 0, "query": "q", "completion": "x"}]
        )


def test_rollout_item_id_grammar_guards():
    with pytest.raises(ValueError, match="__"):
        judging.rollout_item_id("bad__id", 0)
    with pytest.raises(ValueError, match="custom_id"):
        judging.rollout_item_id("x" * 60, 0)


# ---------------------------------------------------------------------------
# generation: render, budget filter, K-file layout + resume (fake vLLM seam)
# ---------------------------------------------------------------------------


def _fake_generate_factory(calls: list[int]):
    """Signature-mirroring fake for the vLLM seam (def mirrors the real
    ``_default_vllm_generate`` signature — never a bare Mock)."""

    def fake_generate(
        prompts: list[str],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
        seeds: list[int],
    ) -> list[list[dict]]:
        assert len(seeds) == len(prompts)
        calls.append(len(prompts))
        return [
            [
                {"text": f"synthetic completion {pi}-{ki}", "finish_reason": "stop"}
                for ki in range(n)
            ]
            for pi in range(len(prompts))
        ]

    return fake_generate


def _synthetic_contexts(behavior: str, n: int, *, aliases: bool = False) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "context_id": f"{behavior}-train-train-{i:06d}",
                "behavior": behavior,
                "split": "train",
                "rung": "train",
                "group_key": f"g{i % 3}",
                "prefix_text": "",
                "query": f"placeholder question about topic {i}",
                "source_dataset": "synthetic",
                "source_id": str(i),
                **({"answer_aliases": ["paris"]} if aliases else {}),
            }
        )
    return rows


def test_render_prompt_parts_prefix_contract(tiny_tokenizer):
    messages = [
        {"role": "system", "content": "synthetic persona text"},
        {"role": "user", "content": "placeholder question"},
    ]
    prefix, prompt = generation.render_prompt_parts(tiny_tokenizer, messages)
    assert prompt.startswith(prefix)
    assert generation.INSTRUCT_USER_HEADER in prompt
    assert prefix.endswith("<|im_end|>\n")
    assert "placeholder question" not in prefix
    # Bare context: prefix is everything before the first user header ("" here
    # since the tiny template injects no default system block).
    prefix_bare, prompt_bare = generation.render_prompt_parts(
        tiny_tokenizer, [{"role": "user", "content": "q"}]
    )
    assert prompt_bare.startswith(prefix_bare)
    assert prefix_bare == ""


def test_filter_prompt_budget_drops_over_budget(tiny_tokenizer):
    short = "a short prompt"
    long = "word " * 400
    kept, digest = generation.filter_prompt_budget(tiny_tokenizer, [short, long], budget=50)
    assert kept == [0]
    assert digest["n_dropped"] == 1
    assert digest["dropped"][0]["index"] == 1
    assert "text" not in digest["dropped"][0]  # digest-only: no row text


def test_generate_labeling_layout_and_resume(tiny_tokenizer, tmp_path):
    contexts = _synthetic_contexts("sycophancy", 3)
    calls: list[int] = []
    manifest = generation.generate_labeling(
        contexts,
        out_root=tmp_path,
        behavior="sycophancy",
        k_rollouts=2,
        seed=0,
        generate_fn=_fake_generate_factory(calls),
        tokenizer=tiny_tokenizer,
    )
    assert manifest["n_generated"] == 3 and manifest["n_resumed"] == 0
    files = sorted((tmp_path / "labeling" / "sycophancy").glob("*_seed*.json"))
    assert len(files) == 6  # 3 contexts x K=2
    payload = json.loads(files[0].read_text())
    for key in ("context_id", "query", "prefix_text", "prompt_text", "completion", "meta"):
        assert key in payload, key
    assert payload["prompt_text"].startswith(payload["prefix_text"])
    # Resume: a second run under the same fingerprint generates nothing new.
    calls2: list[int] = []
    manifest2 = generation.generate_labeling(
        contexts,
        out_root=tmp_path,
        behavior="sycophancy",
        k_rollouts=2,
        seed=0,
        generate_fn=_fake_generate_factory(calls2),
        tokenizer=tiny_tokenizer,
    )
    assert manifest2["n_resumed"] == 3 and manifest2["n_generated"] == 0
    assert calls2 == []


def test_generate_e1_extraction_layout(tiny_tokenizer, tmp_path):
    assets = {
        "instruction": [
            {"pos": f"pos instruction {i}", "neg": f"neg instruction {i}"} for i in range(5)
        ],
        "extraction_questions": [f"extraction question {i}" for i in range(20)],
        "eval_questions": [f"eval question {i}" for i in range(20)],
        "eval_prompt": "rubric with {question} and {answer} slots",
    }
    calls: list[int] = []
    manifest = generation.generate_e1_extraction(
        "sycophancy",
        out_root=tmp_path,
        n_rollouts=2,
        seed=0,
        generate_fn=_fake_generate_factory(calls),
        tokenizer=tiny_tokenizer,
        assets=assets,
    )
    # 5 pairs x 2 signs x 20 questions = 200 jobs.
    assert manifest["n_jobs"] == 200
    files = sorted((tmp_path / "extraction" / "sycophancy").glob("pair*_q*.json"))
    assert len(files) == 200
    payload = json.loads(files[0].read_text())
    assert len(payload["rollouts"]) == 2
    assert payload["sign"] in ("pos", "neg")
    assert payload["prompt_text"].startswith(payload["prefix_text"])


# ---------------------------------------------------------------------------
# capture: span arithmetic + tiny-real store round-trip
# ---------------------------------------------------------------------------


def test_capture_row_ids_and_positions_span_arithmetic(tiny_tokenizer):
    messages = [
        {"role": "system", "content": "synthetic persona text"},
        {"role": "user", "content": "placeholder question about topic alpha"},
    ]
    prefix, prompt = generation.render_prompt_parts(tiny_tokenizer, messages)
    completion = "a short synthetic reply"
    row_ids, pos = capture.capture_row_ids_and_positions(tiny_tokenizer, prefix, prompt, completion)
    prompt_ids = tiny_tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tiny_tokenizer.encode(completion, add_special_tokens=False)
    boundary_ids = tiny_tokenizer.encode(capture.BOUNDARY_INSTRUCT, add_special_tokens=False)
    # Per-segment token-id concatenation (never re-tokenized concatenated text).
    assert row_ids == prompt_ids + completion_ids + boundary_ids
    assert pos["n_prompt"] == len(prompt_ids)
    assert pos["context_end"] == len(prompt_ids) - 1
    assert pos["answer_start"] == len(prompt_ids)
    assert pos["answer_end"] == len(prompt_ids) + len(completion_ids)
    # prefix_end: last prompt token ending INSIDE the prefix -> strictly before
    # the context end for a non-empty prefix.
    assert 0 <= pos["prefix_end"] < pos["context_end"]
    # Over-budget rows fail loud (loader must filter).
    with pytest.raises(ValueError, match="exceeding max_model_len"):
        capture.capture_row_ids_and_positions(
            tiny_tokenizer, prefix, prompt, completion, max_model_len=10
        )


def test_capture_batch_and_store_roundtrip(tiny_tokenizer, tiny_model, tmp_path):
    contexts = _synthetic_contexts("sycophancy", 3)
    rendered = [
        generation.render_prompt_parts(tiny_tokenizer, generation.context_messages(c))
        for c in contexts
    ]
    completions = [f"a synthetic reply number {i}" for i in range(3)]
    summaries, positions = capture.capture_batch(
        [p for p, _ in rendered],
        [p for _, p in rendered],
        completions,
        model=tiny_model,
        tokenizer=tiny_tokenizer,
        n_layers=TINY_LAYERS,
        hidden_dim=TINY_DIM,
        device="cpu",
        batch_size=2,  # forces a padded multi-batch split
    )
    assert len(summaries) == 3 and len(positions) == 3
    for s in summaries:
        for kind in ("prefix_end", "context_end", "t1"):
            assert s[kind].shape == (TINY_LAYERS, TINY_DIM)
            assert s[kind].dtype == np.float16
        # Distinct positions on a real (random-weight) model give distinct
        # states: a t1 span-mean identical to the context_end read would flag
        # a position/indexing bug.
        assert not np.allclose(s["t1"], s["context_end"])

    meta_rows = [
        dict(
            context_id=c["context_id"],
            behavior=c["behavior"],
            rollout_k=0,
            is_eval_only=False,
            group_key=c["group_key"],
            **pos,
        )
        for c, pos in zip(contexts, positions, strict=True)
    ]
    capture.write_store_shard(tmp_path, 0, summaries, meta_rows)

    # Round-trip through the CONSUMER loader (store_io.load_summaries).
    out, meta = store_io.load_summaries(
        tmp_path,
        ("prefix_end", "context_end", "t1"),
        tuple(range(TINY_LAYERS)),
        hidden_dim=TINY_DIM,
    )
    assert set(out) == {
        (k, layer) for k in ("prefix_end", "context_end", "t1") for layer in range(TINY_LAYERS)
    }
    for (kind, layer), arr in out.items():
        assert arr.shape == (3, TINY_DIM)
        expected = np.stack([s[kind][layer] for s in summaries], axis=0)
        np.testing.assert_array_equal(arr, expected)
    assert [m["context_id"] for m in meta] == [c["context_id"] for c in contexts]
    # fit_pool_mask consumes the same sidecar (is_eval_only False -> all kept).
    assert store_io.fit_pool_mask(meta).all()


def test_capture_rollout_files_extraction_shape_side_field(tiny_tokenizer, tiny_model, tmp_path):
    """E1 extraction rollout JSONs (rollouts LIST + pair/sign/q_idx) expand to
    one store row per rollout with ``side`` in the row_index (round C2 wiring);
    executes the REAL capture_rollout_files body through the tiny-real model
    and round-trips through the consumer loader."""
    rollout_dir = tmp_path / "extraction"
    rollout_dir.mkdir()
    prefix, prompt = generation.render_prompt_parts(
        tiny_tokenizer,
        [
            {"role": "system", "content": "a synthetic system instruction"},
            {"role": "user", "content": "a synthetic extraction question"},
        ],
    )
    for sign in ("pos", "neg"):
        payload = {
            "behavior": "sycophancy",
            "pair": 0,
            "sign": sign,
            "q_idx": 0,
            "question": "a synthetic extraction question",
            "prefix_text": prefix,
            "prompt_text": prompt,
            "rollouts": [
                {"text": f"synthetic {sign} reply number {k}", "finish_reason": "stop"}
                for k in range(2)
            ],
        }
        (rollout_dir / f"pair0_{sign}_q00.json").write_text(json.dumps(payload))

    store_dir = tmp_path / "store"
    manifest = capture.capture_rollout_files(
        sorted(rollout_dir.glob("*.json")),
        store_dir=store_dir,
        model=tiny_model,
        tokenizer=tiny_tokenizer,
        n_layers=TINY_LAYERS,
        hidden_dim=TINY_DIM,
        device="cpu",
        batch_size=2,
    )
    assert manifest["n_rows"] == 4  # 2 files x 2 rollouts
    out, meta = store_io.load_summaries(
        store_dir, ("t1",), tuple(range(TINY_LAYERS)), hidden_dim=TINY_DIM
    )
    assert out[("t1", 0)].shape == (4, TINY_DIM)
    sides = [m["side"] for m in meta]
    assert sides.count("pos") == 2 and sides.count("neg") == 2
    # The fits-side pos/neg direction extraction resolves on this store.
    acts = np.stack([out[("t1", ly)] for ly in range(TINY_LAYERS)], axis=1)
    pos_rows = np.flatnonzero(np.array(sides) == "pos")
    neg_rows = np.flatnonzero(np.array(sides) == "neg")
    from explore_persona_space.experiments.issue_1739 import fits

    rb = fits.extract_rb_e1(acts[pos_rows], acts[neg_rows])
    assert rb.shape == (TINY_LAYERS, TINY_DIM) and np.isfinite(rb).all()


def test_judge_cli_writes_dv_dataset(tiny_tokenizer, tmp_path, monkeypatch):
    """The judge CLI main() writes dv_dataset/<behavior>/labeling.json (the
    fits-phase input — round C2 wiring), executing the real CLI body with the
    Batch judge boundary faked signature-conformantly."""
    import scripts.issue1739_judge as judge_cli

    rollout_dir = tmp_path / "labeling" / "sycophancy"
    rollout_dir.mkdir(parents=True)
    for cid, k in (("sycophancy-train-train-000000", 0), ("sycophancy-train-train-000000", 1)):
        (rollout_dir / f"{cid}_seed{k}.json").write_text(
            json.dumps(
                {
                    "context_id": cid,
                    "behavior": "sycophancy",
                    "split": "train",
                    "rung": "train",
                    "group_key": "socialskills-p0",
                    "rollout_k": k,
                    "query": "a synthetic advice question",
                    "prefix_text": "",
                    "prompt_text": "rendered prompt text",
                    "completion": f"synthetic reply {k}",
                }
            )
        )
    # Hermetic: the trait-rubric asset chain would otherwise hit the REAL
    # cache -> Sonnet regeneration (a live API call from a unit test).
    monkeypatch.setattr(
        judging,
        "load_e1_assets",
        lambda behavior, *, inputs_dir=None: {
            "instruction": [],
            "extraction_questions": [],
            "eval_prompt": "rate {question} / {answer} from 0 to 100",
        },
    )

    def fake_judge_graded(items, eval_prompt, *, n_draws, cache_dir, save_raw, **kwargs):
        scores = {item_id: 40.0 + 10.0 * i for i, (item_id, _q, _a) in enumerate(items)}
        return JudgeResult(
            scores=scores,
            n_total_draws=n_draws * len(items),
            n_dropped_draws=0,
            per_item_scores={k: [v] for k, v in scores.items()},
            per_item_draw_counts={k: n_draws for k in scores},
        )

    import explore_persona_space.eval.graded_judge as graded_judge_mod

    monkeypatch.setattr(graded_judge_mod, "judge_graded", fake_judge_graded)
    monkeypatch.setattr(
        "sys.argv",
        [
            "issue1739_judge.py",
            "--behavior",
            "sycophancy",
            "--rollout-dir",
            str(rollout_dir),
            "--out-dir",
            str(tmp_path / "judge_out"),
            "--inputs-dir",
            str(tmp_path / "inputs"),
            "--dv-out-root",
            str(tmp_path / "evalroot"),
        ],
    )
    assert judge_cli.main() == 0
    dv_payload = json.loads(
        (tmp_path / "evalroot" / "dv_dataset" / "sycophancy" / "labeling.json").read_text()
    )
    assert dv_payload["n_contexts"] == 1 and dv_payload["n_contexts_with_dv"] == 1
    row = dv_payload["rows"][0]
    assert row["group_key"] == "socialskills-p0"
    assert row["dv"] == pytest.approx(45.0)  # mean over the two rollout scores


def test_capture_shard_resume_predicate(tmp_path):
    assert not capture.shard_done(tmp_path, 0, "fp")
    (tmp_path / "row_index_shard00.jsonl").write_text('{"context_id": "c"}\n')
    meta = tmp_path / "_capture_meta_shard00.json"
    meta.write_text(json.dumps({"fingerprint": "fp", "n_rows": 1}))
    assert capture.shard_done(tmp_path, 0, "fp")
    assert not capture.shard_done(tmp_path, 0, "other-fp")


def test_teacher_forced_ln_logp_tiny_real(tiny_tokenizer, tiny_model):
    pairs = [
        ("prompt about topic alpha ", "a synthetic reply"),
        ("a very different prompt entirely ", "a synthetic reply"),
    ]
    lps = capture.teacher_forced_ln_logp(
        pairs, model=tiny_model, tokenizer=tiny_tokenizer, device="cpu", batch_size=1
    )
    assert len(lps) == 2
    assert all(np.isfinite(lp) and lp < 0 for lp in lps)
    # Same completion under different prompts must condition differently.
    assert lps[0] != lps[1]
    # Batched == per-pair (padding must not change the gathered logprobs).
    lps_batched = capture.teacher_forced_ln_logp(
        pairs, model=tiny_model, tokenizer=tiny_tokenizer, device="cpu", batch_size=2
    )
    np.testing.assert_allclose(lps_batched, lps, rtol=0, atol=1e-4)
    with pytest.raises(ValueError, match="empty completion"):
        capture.teacher_forced_ln_logp(
            [("p", "")], model=tiny_model, tokenizer=tiny_tokenizer, device="cpu"
        )


# ---------------------------------------------------------------------------
# gates 1-2 + the pilot compose driver (real bodies; GPU/API seams faked)
# ---------------------------------------------------------------------------


def test_gate1_yield_report_verdicts():
    rows_pass = [{"context_id": f"c{i}", "dv": float(10 * i % 100)} for i in range(10)]
    report = gates.gate1_yield_report(rows_pass, behavior="sycophancy", n_pilot=10)
    assert report["verdict"] == "PASS"
    assert sum(report["expression_histogram"].values()) == 10
    rows_fail = [{"context_id": f"c{i}", "dv": None} for i in range(10)]
    assert gates.gate1_yield_report(rows_fail, behavior="sycophancy")["verdict"] == "FAIL"


def test_gate2_spread_floor_verdicts():
    spread = [{"context_id": f"c{i}", "dv": float(i * 11 % 100)} for i in range(20)]
    report = gates.gate2_spread_floor(spread, behavior="sycophancy")
    assert report["verdict"] == "PASS" and report["tf_margin_fallback"] is False
    collapsed = [{"context_id": f"c{i}", "dv": 1.0 + 0.01 * i} for i in range(20)]
    report2 = gates.gate2_spread_floor(collapsed, behavior="sycophancy")
    assert report2["verdict"] == "FAIL" and report2["tf_margin_fallback"] is True
    assert report2["sd_ok"] is False and report2["bottom_bin_ok"] is False
    empty = gates.gate2_spread_floor([{"context_id": "c", "dv": None}], behavior="x")
    assert empty["verdict"] == "FAIL" and empty["tf_margin_fallback"] is True


def test_run_gate1_pilot_compose_real_bodies(tiny_tokenizer, tmp_path):
    """Executes the REAL run_gate1_pilot body (staging read, generation,
    hallucination three-way split, DV build, gate reports); fakes ONLY the
    vLLM + judge boundaries with signature-conformant fakes (a real
    JudgeResult dataclass instance; def-mirroring generate_fn)."""
    staged_dir = tmp_path / "staged"
    contexts = _synthetic_contexts("hallucination", 4, aliases=True)
    path = corpus_staging.staged_context_path(staged_dir, "hallucination", "train", "train")
    path.parent.mkdir(parents=True, exist_ok=True)
    corpus_staging._write_jsonl_atomic(path, contexts)

    def judge_fn(items, eval_prompt, *, cache_dir, save_raw):
        assert "{question}" in eval_prompt and "{answer}" in eval_prompt
        scores = {}
        for i, (item_id, _q, _a) in enumerate(items):
            scores[item_id] = 90.0 if i % 2 == 0 else 10.0
        return JudgeResult(
            scores=scores,
            n_total_draws=3 * len(items),
            n_dropped_draws=0,
            per_item_scores={k: [v] for k, v in scores.items()},
            per_item_draw_counts={k: 3 for k in scores},
        )

    calls: list[int] = []
    report = gates.run_gate1_pilot(
        "hallucination",
        out_root=tmp_path / "out",
        staged_dir=staged_dir,
        n_pilot=4,
        seed=0,
        generate_fn=_fake_generate_factory(calls),
        judge_fn=judge_fn,
        tokenizer=tiny_tokenizer,
    )
    assert report["gate1"]["n_contexts"] == 4
    assert report["gate1"]["verdict"] in ("PASS", "FAIL")
    assert "tf_margin_fallback" in report["gate2"]
    report_path = tmp_path / "out" / "gate1" / "hallucination_pilot_report.json"
    assert report_path.exists()
    # K x 4 contexts rollout files landed under the gate1 out_root.
    n_files = len(
        [
            p
            for p in (tmp_path / "out" / "gate1" / "raw_completions" / "labeling").rglob(
                "*_seed*.json"
            )
        ]
    )
    assert n_files == 4 * generation.K_ROLLOUTS
