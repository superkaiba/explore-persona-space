"""Pins for the wcrung GPU-leg driver (#1739 wildchat rung, path C).

Covers the two structural departures from the pvsynth sibling — the multi-turn
renderer and the generate-once/judge-3x single-pool shape — plus a tiny-real
CPU end-to-end of the whole generation phase with ONLY the vLLM boundary faked:
real contexts build, real render, real prompt-budget filter, real per-rollout
JSON writes, real shard packing, real sentinel. The capture phase is GPU-bound
and covered by the carve-out (signature smoke + a fresh-process dry run), not
here.

No real corpus text: rows carry synthetic placeholder strings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import issue1739_wcrung_pod as pod  # noqa: E402


class _FakeTokenizer:
    """Qwen-shaped chat template; whitespace tokenizer for the budget filter."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def _row(idx: int, *, turns: int) -> dict:
    prefix_turns = []
    for t in range(turns):
        prefix_turns.append({"role": "user", "content": f"earlier user turn {idx}-{t}"})
        prefix_turns.append({"role": "assistant", "content": f"earlier assistant turn {idx}-{t}"})
    return {
        "context_id": f"wcrung-{idx:04d}",
        "source_conv_id": f"wildchat_{idx:06d}",
        "prefix_turns": prefix_turns,
        "prefix_text": "(sampler render, provenance only)",
        "query": f"final user query {idx}",
        "n_prefix_turns": len(prefix_turns),
        "single_turn": turns == 0,
        "query_sha256": f"{idx:064d}",
        "split": "eval",
        "rung": "wildchat_rung",
        "group_key": f"wcrung-{idx:04d}",
    }


def _write_rows(out_root: Path, rows: list[dict]) -> Path:
    path = out_root / "contexts" / "wcrung.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows, "n_contexts": len(rows)}))
    return path


# --- the multi-turn renderer (the reason this rung passes render_fn) ---------


def test_render_fn_keeps_conversation_turns_in_prefix():
    tok = _FakeTokenizer()
    row = _row(1, turns=2)
    prefix, prompt = pod.wcrung_render_fn(tok, row)
    assert prompt.startswith(prefix), "capture derives prefix_end from len(prefix_text)"
    for turn in row["prefix_turns"]:
        assert turn["content"] in prefix, "a conversation turn was cut out of the prefix"
    assert row["query"] not in prefix, "the query must not leak into the prefix arm"
    assert row["query"] in prompt


def test_render_fn_single_turn_prefix_is_template_head_only():
    tok = _FakeTokenizer()
    prefix, prompt = pod.wcrung_render_fn(tok, _row(2, turns=0))
    assert prompt.startswith(prefix)
    assert "final user query 2" not in prefix
    assert prefix.endswith("<|im_start|>user\n") or prefix == ""


def test_render_fn_differs_from_shared_default_on_multi_turn():
    """Motivation pin: the shared single-user-turn render loses earlier turns."""
    from explore_persona_space.experiments.issue_1739.generation import (
        context_messages,
        render_prompt_parts,
    )

    tok = _FakeTokenizer()
    row = _row(3, turns=1)
    default_prefix, _ = render_prompt_parts(tok, context_messages(row))
    wcrung_prefix, _ = pod.wcrung_render_fn(tok, row)
    assert row["prefix_turns"][0]["content"] not in default_prefix
    assert row["prefix_turns"][0]["content"] in wcrung_prefix


# --- contexts build --------------------------------------------------------


def test_build_contexts_stamps_rung_fields_and_single_gen_behavior(tmp_path):
    rows = [_row(i, turns=i % 2) for i in range(4)]
    _write_rows(tmp_path, rows)
    args = pod._parse_args(["--out-root", str(tmp_path)])
    contexts = pod.build_contexts(args)
    assert len(contexts) == 4
    assert {c["behavior"] for c in contexts} == {pod.GEN_BEHAVIOR}, "one pool, three rubrics"
    assert {c["rung"] for c in contexts} == {"wildchat_rung"}
    assert {c["split"] for c in contexts} == {"eval"}
    # Each conversation is its own fold group (conversation-disjoint by construction).
    assert len({c["group_key"] for c in contexts}) == 4


def test_build_contexts_max_contexts_caps(tmp_path):
    _write_rows(tmp_path, [_row(i, turns=0) for i in range(6)])
    args = pod._parse_args(["--out-root", str(tmp_path), "--max-contexts", "2"])
    assert len(pod.build_contexts(args)) == 2


def test_shards_partition_contexts_exactly_once(tmp_path):
    """Fan-out: every context lands in exactly one shard, none duplicated."""
    rows = [_row(i, turns=i % 2) for i in range(11)]
    _write_rows(tmp_path, rows)
    seen: list[str] = []
    for idx in range(4):
        args = pod._parse_args(
            ["--out-root", str(tmp_path), "--n-shards", "4", "--shard-idx", str(idx)]
        )
        seen.extend(c["context_id"] for c in pod.build_contexts(args))
    assert sorted(seen) == sorted(r["context_id"] for r in rows)
    assert len(seen) == len(set(seen)), "a context landed in two shards"


def test_width_one_is_the_unsharded_context_list(tmp_path):
    rows = [_row(i, turns=0) for i in range(5)]
    _write_rows(tmp_path, rows)
    unsharded = pod.build_contexts(pod._parse_args(["--out-root", str(tmp_path)]))
    width_one = pod.build_contexts(
        pod._parse_args(["--out-root", str(tmp_path), "--n-shards", "1", "--shard-idx", "0"])
    )
    assert unsharded == width_one


def test_bad_shard_index_fails_loud(tmp_path):
    _write_rows(tmp_path, [_row(0, turns=0)])
    for argv in (["--n-shards", "2", "--shard-idx", "2"], ["--n-shards", "0"]):
        args = pod._parse_args(["--out-root", str(tmp_path), *argv])
        with pytest.raises(RuntimeError, match="bad fan-out"):
            pod.build_contexts(args)


def test_build_contexts_fails_loud_on_missing_field(tmp_path):
    bad = _row(1, turns=0)
    bad["query"] = ""
    _write_rows(tmp_path, [bad])
    args = pod._parse_args(["--out-root", str(tmp_path)])
    with pytest.raises(RuntimeError, match="missing 'query'"):
        pod.build_contexts(args)


def test_build_contexts_fails_loud_on_duplicate_ids(tmp_path):
    dup = _row(1, turns=0)
    _write_rows(tmp_path, [dup, dict(dup)])
    args = pod._parse_args(["--out-root", str(tmp_path)])
    with pytest.raises(RuntimeError, match="duplicate wcrung context_id"):
        pod.build_contexts(args)


def test_build_contexts_fails_loud_on_empty_rows(tmp_path):
    _write_rows(tmp_path, [])
    args = pod._parse_args(["--out-root", str(tmp_path)])
    with pytest.raises(RuntimeError, match="no wcrung context rows"):
        pod.build_contexts(args)


# --- tiny-real CPU e2e of the generation phase ------------------------------


def test_tiny_real_generation_e2e_cpu(tmp_path, monkeypatch, capsys):
    """Real render + budget filter + rollout writes + pack + sentinel.

    ONLY the vLLM call is faked (the GPU boundary). Both arm classes are
    present: a multi-turn conversation prefix and a single-turn (empty-prefix)
    row.
    """
    rows = [_row(0, turns=0), _row(1, turns=2), _row(2, turns=1)]
    _write_rows(tmp_path, rows)

    seen_prompts: list[str] = []

    def fake_vllm_generate(prompts, *, n, temperature, max_tokens, seeds):
        seen_prompts.extend(prompts)
        return [
            [{"text": f"completion {i}-{k}", "finish_reason": "stop"} for k in range(n)]
            for i in range(len(prompts))
        ]

    from explore_persona_space.experiments.issue_1739 import generation

    monkeypatch.setattr(generation, "_default_vllm_generate", fake_vllm_generate)
    monkeypatch.setattr(generation, "get_tokenizer", lambda *a, **k: _FakeTokenizer())

    with pytest.raises(SystemExit) as exc:
        pod.main(
            [
                "--out-root",
                str(tmp_path),
                "--store-root",
                str(tmp_path / "store"),
                "--k-rollouts",
                "2",
                "--skip-capture",
                "--skip-upload",
            ]
        )
    assert exc.value.code == 0

    # Every context's K rollouts landed, under the single gen behavior. The
    # `_`-prefixed generation manifest sits in the same dir and is excluded
    # here exactly as the capture phase's own glob excludes it.
    rollout_dir = tmp_path / "labeling" / pod.GEN_BEHAVIOR
    written = sorted(p.name for p in rollout_dir.glob("*.json") if not p.name.startswith("_"))
    assert len(written) == len(rows) * 2, written
    assert (rollout_dir / "_manifest.json").exists(), "generation manifest missing"

    tok = _FakeTokenizer()
    for row in rows:
        want_prefix, want_prompt = pod.wcrung_render_fn(tok, row)
        payload = json.loads((rollout_dir / f"{row['context_id']}_seed0.json").read_text())
        assert payload["prefix_text"] == want_prefix, "rollout prefix != wcrung render"
        assert payload["prompt_text"] == want_prompt
        assert payload["behavior"] == pod.GEN_BEHAVIOR
        assert payload["rung"] == "wildchat_rung"
        assert payload["prompt_text"].startswith(payload["prefix_text"])

    # The multi-turn row's earlier turns really reached the model prompt.
    multi = next(r for r in rows if not r["single_turn"])
    multi_payload = json.loads((rollout_dir / f"{multi['context_id']}_seed0.json").read_text())
    assert multi["prefix_turns"][0]["content"] in multi_payload["prompt_text"]

    # Pack ran and produced shards + a manifest (the 10k-files-per-dir guard).
    pack_root = tmp_path / "labeling_packed"
    assert (pack_root / "pack_manifest.json").exists()
    assert list(pack_root.glob("*.jsonl")), "no packed shards"

    sentinel = json.loads((tmp_path / pod.SENTINEL_NAME).read_text())
    assert sentinel["rung"] == "wildchat_rung"
    assert sentinel["gen_behavior"] == pod.GEN_BEHAVIOR
    assert sentinel["judge_behaviors"] == ["evil", "sycophancy", "hallucination"]
    assert sentinel["n_contexts"] == len(rows)
    assert sentinel["k_rollouts"] == 2
    assert sentinel["n_multi_turn_contexts"] == 2
    assert sentinel["capture_rows"] is None  # --skip-capture

    out = capsys.readouterr().out
    assert "[phase=wcrung_contexts" in out
    assert "[phase=wcrung_pack]" in out
    assert "[phase=done]" in out
    # Content hygiene: the digest artifact carries ids only, never query text.
    digest = json.loads((tmp_path / "contexts" / "wcrung_gen_contexts.json").read_text())
    assert digest["context_ids"] == [r["context_id"] for r in rows]
    assert "final user query 0" not in json.dumps(digest)


def test_generation_resumes_and_regenerates_nothing(tmp_path, monkeypatch):
    """Second run over the same out_root regenerates zero contexts (resume)."""
    rows = [_row(i, turns=i % 2) for i in range(3)]
    _write_rows(tmp_path, rows)

    calls: list[int] = []

    def fake_vllm_generate(prompts, *, n, temperature, max_tokens, seeds):
        calls.append(len(prompts))
        return [
            [{"text": "c", "finish_reason": "stop"} for _ in range(n)] for _ in range(len(prompts))
        ]

    from explore_persona_space.experiments.issue_1739 import generation

    monkeypatch.setattr(generation, "_default_vllm_generate", fake_vllm_generate)
    monkeypatch.setattr(generation, "get_tokenizer", lambda *a, **k: _FakeTokenizer())

    argv = [
        "--out-root",
        str(tmp_path),
        "--store-root",
        str(tmp_path / "store"),
        "--k-rollouts",
        "1",
        "--skip-capture",
        "--skip-upload",
    ]
    for _ in range(2):
        with pytest.raises(SystemExit) as exc:
            pod.main(argv)
        assert exc.value.code == 0
    assert calls == [3], f"second run should regenerate nothing, got {calls}"


# --- GPU-bound capture: signature smoke ------------------------------------


def test_main_locals_do_not_shadow_module_level_symbols():
    """REGRESSION PIN: no local of main() may shadow a module-level symbol.

    An ``import X`` inside a function is a BINDING, so the compiler marks X a
    local of that function for its ENTIRE body — including paths that never
    execute the import. main()'s inline --import-check block imported the bare
    name ``capture``, which made the phase-2 call to the module-level
    ``def capture(...)`` read an unbound local: UnboundLocalError at the
    capture-phase entry, AFTER generation had completed on a billed GPU.

    ``_import_check`` is the sanctioned exception and the fix itself: the
    import bindings are CONFINED there, and that function reads no
    module-level name, so nothing it binds can shadow anything.
    """
    import types

    mod_names = {n for n in dir(pod) if not n.startswith("__")}
    offenders: list[str] = []
    for fname in dir(pod):
        fn = getattr(pod, fname)
        if not isinstance(fn, types.FunctionType) or fn.__module__ != pod.__name__:
            continue
        if fname == "_import_check":
            continue  # the containment function — see docstring
        for shadowed in sorted(mod_names & set(fn.__code__.co_varnames)):
            offenders.append(f"{fname}() shadows module-level {shadowed!r}")
    assert not offenders, (
        "local binding(s) shadow a module-level symbol — a branch that never "
        "runs still makes the name a function-wide local:\n  " + "\n  ".join(offenders)
    )
    # the specific name that crashed production, pinned by itself
    assert "capture" not in pod.main.__code__.co_varnames


def test_main_phase2_reaches_module_level_capture(tmp_path, monkeypatch, capsys):
    """FIX-ENGAGED SIGNAL: main()'s REAL phase-2 line executes.

    The pre-fix suite passed because the e2e ran with --skip-capture (which
    bypasses the crash site) and the signature test called
    capture_rollout_files DIRECTLY. This drives main() WITHOUT --skip-capture
    so the production line `cap_manifest = capture(args, ...)` actually runs,
    with only the GPU boundaries faked (model load + the batched capture call)
    and uploads skipped. Pre-fix this raises UnboundLocalError: 'capture'.
    """
    rows = [_row(0, turns=0), _row(1, turns=2)]
    _write_rows(tmp_path, rows)

    def fake_vllm_generate(prompts, *, n, temperature, max_tokens, seeds):
        return [
            [{"text": f"completion {i}-{k}", "finish_reason": "stop"} for k in range(n)]
            for i in range(len(prompts))
        ]

    from explore_persona_space.experiments.issue_1739 import capture as capture_mod
    from explore_persona_space.experiments.issue_1739 import generation

    monkeypatch.setattr(generation, "_default_vllm_generate", fake_vllm_generate)
    monkeypatch.setattr(generation, "get_tokenizer", lambda *a, **k: _FakeTokenizer())
    monkeypatch.setattr(pod, "reap_generation_engine", lambda *a, **k: None)

    seen: dict = {}

    def fake_load_capture_model(*, device):
        seen["device"] = device
        return object()

    def fake_capture_rollout_files(paths, **kw):
        seen["n_paths"] = len(paths)
        seen["kwargs"] = kw
        Path(kw["store_dir"]).mkdir(parents=True, exist_ok=True)
        return {"n_rows": len(paths), "n_shards": 1}

    monkeypatch.setattr(capture_mod, "load_capture_model", fake_load_capture_model)
    monkeypatch.setattr(capture_mod, "capture_rollout_files", fake_capture_rollout_files)

    with pytest.raises(SystemExit) as exc:
        pod.main(
            [
                "--out-root",
                str(tmp_path),
                "--store-root",
                str(tmp_path / "store"),
                "--k-rollouts",
                "2",
                "--skip-upload",
                "--device",
                "cpu",
            ]
        )
    assert exc.value.code == 0

    # The module-level capture() really ran: it globbed the rollouts, passed
    # the generation fingerprint through, and printed its own phase line.
    assert seen["n_paths"] == len(rows) * 2, seen
    assert seen["kwargs"]["fingerprint"], "generation fingerprint not threaded"
    assert seen["device"] == "cpu"
    out = capsys.readouterr().out
    assert "[phase=wcrung_capture] rows=" in out, out[-2000:]
    assert "[phase=done]" in out

    sentinel = json.loads((tmp_path / pod.SENTINEL_NAME).read_text())
    assert sentinel["capture_rows"] == len(rows) * 2, "capture manifest not folded into sentinel"


def test_capture_call_shape_binds_against_real_signature():
    """The GPU-bound capture entrypoint's signature matches this caller."""
    import inspect

    from explore_persona_space.experiments.issue_1739.capture import (
        capture_rollout_files,
        load_capture_model,
    )

    inspect.signature(load_capture_model).bind(device="cuda")
    inspect.signature(capture_rollout_files).bind(
        [Path("x.json")],
        store_dir=Path("s"),
        model=object(),
        tokenizer=object(),
        n_layers=28,
        hidden_dim=3584,
        device="cuda",
        fingerprint="abc",
        batch_size=4,
    )
