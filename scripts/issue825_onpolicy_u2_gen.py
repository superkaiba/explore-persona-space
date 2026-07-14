#!/usr/bin/env python
"""Issue #825 follow-up ``onpolicy-user-turn`` — self-generated u2 + audit + wiring check.

THE ONE VARIABLE vs the parent: u2 provenance. The parent's second user turn was
Haiku-4.5-simulated; here each measured model writes its OWN next user turn, per
(model, generation-format) cell, over the SAME kept-2000 (u1, a1) conversations
(reused byte-identical from the parent artifact). Everything downstream (render,
extraction, fits) is the parent rig unchanged.

Modes:
  gen (default)     Generate u2 for ALL input rows x requested cells (vLLM on GPU;
                    ``--tiny-model-dir`` = CPU smoke substitute via transformers
                    sampling). Emits per-cell conversations JSONL (all rows kept —
                    NO generation-time row dropping, plan MF-A), per-cell audit
                    ``_meta.json``, and updates ``row_allowlists.json`` (per-USER-cell
                    conv_ids passing the fit-time row filters). A tokenize-only
                    span-validation pass (the extractor's own span asserts, run at
                    gen time) substitutes the validated multi-token placeholder for
                    any u2 whose text renders a zero-width content span (bare
                    punctuation BPE-merging into the naturalistic delimiter — the
                    run-1 extract crash) and hard-fails if any degenerate span
                    survives substitution.
  --wiring-check    Own-context vs derangement-shuffled-context teacher-forced NLL
                    of u2 on a seeded row subsample per cell (plan MF-B). Batched
                    forwards; writes ``wiring_check_<model>.json``. HALT evaluation
                    lives in the dispatch wrapper's gate phase, not here.

u2 prompt constructions (plan section "u2 generation recipe"):
  chat:          apply_chat_template([u1(user), a1(assistant)],
                 add_generation_prompt=False) + "<|im_start|>user\\n"; stop <|im_end|>
  naturalistic:  "User: {u1}\\n\\nAssistant: {a1}\\n\\nUser: "; stop "\\n\\n"
                 (+ "\\nAssistant:" — plan-allowed extra stop)
Sampling: T=1.0, top_p=1.0, max_tokens=512, seed 42 (parent Haiku-arm parity).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# vLLM V1 fork-EngineCore guard (gotchas.md): spawn BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.experiments.issue_825.common import (
    MAX_CONV_TOKENS,
    MIN_TURN_CONTENT_TOKENS,
    MODEL_INSTRUCT,
    MODEL_PRETRAINED,
    N_TRACK_M,
)
from explore_persona_space.orchestrate.env import load_dotenv

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

MODEL_IDS = {"instruct": MODEL_INSTRUCT, "pretrained": MODEL_PRETRAINED}
U2_TEMPERATURE = 1.0
U2_TOP_P = 1.0
U2_MAX_TOKENS = 512
U2_SEED = 42
CHAT_STOPS = ["<|im_end|>"]
NAT_STOPS = ["\n\n", "\nAssistant:"]
# Structural placeholder for a generation that strips to "" OR whose text
# renders to a ZERO-WIDTH u2 content span (run-1 crash: a bare "." u2 fully
# BPE-merges with the naturalistic "\n\n" turn delimiter into one token
# " .\n\n", so zero tokens are fully contained in the u2 char range and the
# extractor's span assert 1 <= s < e fires on conv 723). The placeholder MUST
# be multi-token with interior tokens that cannot merge into either boundary:
# "(no reply)" renders as [" (", "no", " reply", ")\n\n"] naturalistically —
# "no"/" reply" stay fully contained — and is validated per format at startup
# (assert_placeholder_span_valid). Keeps the 3-turn render/extract contract
# intact for the FULL row set (anchors need all rows; causal attention means
# u2 content cannot affect any anchor read). Substituted rows are flagged
# (u2_generated_empty / u2_span_degenerate) and always EXCLUDED from the
# user-cell allowlist (their originals fail the >=8-content-token filter by
# construction — asserted).
EMPTY_U2_PLACEHOLDER = "(no reply)"
# Parent Haiku reference for the audit (plan section "Degeneracy audit").
PARENT_DISTINCT_3GRAM_REFERENCE = 0.781
VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
# Smoke needs >=4 allowlisted rows for any ridge fold to fit (tr>=3 guard in
# issue825_fit_cells.heldout_r2_sweep); the numeric row filters are bypassed
# structurally (padded allowlist) under smoke per plan MF-D.
SMOKE_MIN_ALLOWLIST = 4


def _rf():
    """Lazy import of the sibling render module (same scripts/ dir)."""
    import issue825_render_formats

    return issue825_render_formats


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _ntok(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Prompt construction + the chat-template well-formedness assert (plan asm. 4)
# ---------------------------------------------------------------------------


def build_prompt_chat(tokenizer, u1: str, a1: str) -> str:
    """Chat-template u2 prompt: rendered (u1, a1) history + an OPEN user header."""
    base = tokenizer.apply_chat_template(
        [{"role": "user", "content": u1}, {"role": "assistant", "content": a1}],
        add_generation_prompt=False,
        tokenize=False,
    )
    return base + "<|im_start|>user\n"


def build_prompt_naturalistic(u1: str, a1: str) -> str:
    """Naturalistic u2 prompt using the render's own segment strings."""
    return f"User: {u1}\n\nAssistant: {a1}\n\nUser: "


def assert_chat_template_wellformed(tokenizer, u1: str, a1: str) -> None:
    """Plan section 12 assumption 4: apply_chat_template(..., add_generation_prompt=False)
    must end '<|im_end|>\\n' so the appended user header is well-formed, and its
    conversation tail must equal render_chat's segment structure for (u1, a1)
    (the system preamble before it is the generation-vs-render context split the
    parent had for a-turns)."""
    base = tokenizer.apply_chat_template(
        [{"role": "user", "content": u1}, {"role": "assistant", "content": a1}],
        add_generation_prompt=False,
        tokenize=False,
    )
    assert base.endswith("<|im_end|>\n"), (
        f"chat template does not end with '<|im_end|>\\n' (tail: {base[-40:]!r}) — "
        "appended '<|im_start|>user\\n' header would be malformed"
    )
    # render_chat's segments for a 2-turn conv are exactly header/content/term:
    expected_tail = f"<|im_start|>user\n{u1}<|im_end|>\n<|im_start|>assistant\n{a1}<|im_end|>\n"
    assert base.endswith(expected_tail), (
        "chat-template conversation tail diverges from render_chat segment "
        f"structure (template tail: {base[-120:]!r})"
    )


# ---------------------------------------------------------------------------
# Gen-time span validation (tokenize-only; no GPU) — the extractor's span
# asserts, applied to every generated row BEFORE the conversations JSONL is
# written, so a span-degenerate u2 can never reach the extract phase (run-1
# crash-fix: AssertionError "723: span u2=(201,201) invalid" 21 min in).
# ---------------------------------------------------------------------------


def _render_row(row: dict, tokenizer, fmt: str):
    """Render one row through the SAME path the extractor uses for this format."""
    rf = _rf()
    renderer = rf.render_chat if fmt == "chat" else rf.render_naturalistic
    return renderer(row, tokenizer)


def _degenerate_spans(rendered) -> list[str]:
    """Names of spans/slots that would fail issue825_extract_turnstore's asserts.

    Mirrors process_batch exactly: every turn span must satisfy
    ``1 <= s < e <= seq_len``; every slot ``0 <= idx < seq_len``. Returns an
    empty list when the row extracts cleanly.
    """
    n = len(rendered.input_ids)
    bad = [name for name, (s, e) in rendered.spans.items() if not (1 <= s < e <= n)]
    bad += [f"slot:{name}" for name, idx in rendered.slot_idx.items() if not (0 <= idx < n)]
    return bad


def assert_placeholder_span_valid(tokenizer, u1: str, a1: str) -> None:
    """Startup probe: EMPTY_U2_PLACEHOLDER must render a non-degenerate u2 span
    in BOTH formats (a single-token placeholder like the old "." fully
    BPE-merges with the naturalistic delimiter and re-creates the run-1 crash).
    """
    probe = {"conv_id": "placeholder_probe", "u1": u1, "a1": a1, "u2": EMPTY_U2_PLACEHOLDER}
    for fmt in ("chat", "naturalistic"):
        bad = _degenerate_spans(_render_row(probe, tokenizer, fmt))
        assert not bad, (
            f"EMPTY_U2_PLACEHOLDER {EMPTY_U2_PLACEHOLDER!r} renders degenerate "
            f"spans {bad} under format {fmt} — pick a placeholder whose interior "
            "tokens cannot BPE-merge into the turn boundaries"
        )


def _process_cell_rows(
    convs: list[dict], raw: list[str], tokenizer, fmt: str, cell_label: str
) -> tuple[list[dict], list, dict, list[dict]]:
    """Build one cell's output rows + allowlist with gen-time span validation.

    Returns ``(rows_out, allow, drops, span_degenerate)``. Every input row is
    kept (plan MF-A — no generation-time row dropping): an empty generation OR
    a span-degenerate generation (zero-width u2 content span in THIS cell's
    format) is SUBSTITUTED with the validated EMPTY_U2_PLACEHOLDER, flagged on
    the row, recorded in ``span_degenerate`` (conv_id + original text), and
    excluded from the allowlist. After the loop the FULL row set is re-rendered
    and must show ZERO degenerate spans (hard-fail listing offenders).
    """
    rows_out: list[dict] = []
    allow: list = []
    drops = {"empty_u2": 0, "short_u2": 0, "too_long": 0, "span_degenerate_u2": 0}
    span_degenerate: list[dict] = []
    for c, u2 in zip(convs, raw, strict=True):
        empty = not u2
        row = {
            "conv_id": c["conv_id"],
            "u1": c["u1"],
            "a1": c["a1"],
            "u2": u2 if not empty else EMPTY_U2_PLACEHOLDER,
        }
        if empty:
            row["u2_generated_empty"] = True
            rows_out.append(row)
            drops["empty_u2"] += 1
            continue
        rendered = _render_row(row, tokenizer, fmt)
        degen = _degenerate_spans(rendered)
        if degen:
            assert degen == ["u2"], (
                f"{c['conv_id']}: non-substitutable degenerate spans {degen} under "
                f"{fmt} — only the generated u2 can be substituted (u1/a1 are "
                "pinned parent data)"
            )
            n_orig = _ntok(tokenizer, u2)
            assert n_orig < MIN_TURN_CONTENT_TOKENS, (
                f"{c['conv_id']}: span-degenerate u2 has {n_orig} standalone tokens "
                f">= {MIN_TURN_CONTENT_TOKENS} — violates the short-text premise "
                "(zero fully-contained tokens implies <=2 boundary-straddling tokens)"
            )
            row["u2"] = EMPTY_U2_PLACEHOLDER
            row["u2_span_degenerate"] = True
            span_degenerate.append({"conv_id": c["conv_id"], "u2_original": u2})
            rows_out.append(row)
            drops["span_degenerate_u2"] += 1
            print(
                f"[gen] {cell_label}: substituted span-degenerate u2 "
                f"conv={c['conv_id']} (orig {u2[:80]!r})"
            )
            continue
        rows_out.append(row)
        # Fit-time row filters (applied via allowlist, NEVER by dropping rows
        # from the JSONL — plan MF-A): >=8 content tokens AND rendered length
        # <= 2048 in the CELL'S OWN format.
        if _ntok(tokenizer, u2) < MIN_TURN_CONTENT_TOKENS:
            drops["short_u2"] += 1
            continue
        if len(rendered.input_ids) > MAX_CONV_TOKENS:
            drops["too_long"] += 1
            continue
        allow.append(c["conv_id"])
    # Re-validate the FULL row set post-substitution: zero degenerate spans in
    # this cell's format, or hard-fail at gen with the offending conv_ids.
    offenders = []
    for row in rows_out:
        bad = _degenerate_spans(_render_row(row, tokenizer, fmt))
        if bad:
            offenders.append((row["conv_id"], bad))
    assert not offenders, (
        f"{cell_label}: {len(offenders)} row(s) still render degenerate spans "
        f"after placeholder substitution: {offenders[:20]}"
    )
    print(
        f"[gen] {cell_label}: span re-validation PASS "
        f"(0 degenerate spans across {len(rows_out)} rows)"
    )
    return rows_out, allow, drops, span_degenerate


# ---------------------------------------------------------------------------
# Audit metrics (3-gram helper adapted from issue825_gen_conversations, kept
# local so this script does not execute that module's import-time side effects)
# ---------------------------------------------------------------------------


def _distinct_3gram_rate(texts: list[str]) -> float:
    total = 0
    distinct: set[tuple[str, ...]] = set()
    for text in texts:
        words = text.split()
        for j in range(len(words) - 2):
            total += 1
            distinct.add(tuple(words[j : j + 3]))
    return (len(distinct) / total) if total else 0.0


def _repetition_rate(texts: list[str], min_count: int = 5) -> float:
    """Fraction of texts where any 3-gram repeats >= min_count times WITHIN the text."""
    if not texts:
        return 0.0
    n_rep = 0
    for text in texts:
        words = text.split()
        counts: Counter[tuple[str, ...]] = Counter(
            tuple(words[j : j + 3]) for j in range(len(words) - 2)
        )
        if counts and max(counts.values()) >= min_count:
            n_rep += 1
    return n_rep / len(texts)


def _role_artifact_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    n_bad = sum(
        1
        for t in texts
        if "<|im_start|>" in t or "<|im_end|>" in t or t.lstrip().startswith("Assistant:")
    )
    return n_bad / len(texts)


def _length_stats(tokenizer, texts: list[str]) -> dict:
    if not texts:
        return {"mean": None, "sd": None, "n": 0}
    import numpy as np

    lens = np.array([_ntok(tokenizer, t) for t in texts], dtype=np.float64)
    return {
        "mean": float(lens.mean()),
        "sd": float(lens.std(ddof=1) if len(lens) > 1 else 0.0),
        "n": len(lens),
    }


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------


def _generate_vllm(
    model_id: str, jobs: list[tuple[str, list[str], list[str]]]
) -> dict[str, list[str]]:
    """One vLLM engine load; per-cell chunked generation. jobs = [(cell_key, prompts, stops)]."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "u2 generation requires a CUDA GPU for vLLM; for the CPU smoke pass "
            "--tiny-model-dir <dir> (transformers sampling substitute)."
        )
    from vllm import LLM, SamplingParams

    # max_model_len bounds the KV allocation: prompts are <=~2100 tok (parent
    # length filter) + 512 new; 4096 is ample and avoids the pretrained model's
    # 131072-token config default.
    llm = LLM(model=model_id, max_model_len=4096)
    out: dict[str, list[str]] = {}
    for cell_key, prompts, stops in jobs:
        sp = SamplingParams(
            temperature=U2_TEMPERATURE,
            top_p=U2_TOP_P,
            max_tokens=U2_MAX_TOKENS,
            seed=U2_SEED,
            stop=stops,
        )
        texts: list[str] = []
        n_chunks = (len(prompts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
        for i in range(0, len(prompts), VLLM_CHUNK_SIZE):
            chunk = prompts[i : i + VLLM_CHUNK_SIZE]
            print(
                f"[vllm-chunk] {cell_key} chunk {i // VLLM_CHUNK_SIZE + 1}/{n_chunks} "
                f"({len(chunk)} prompts)",
                flush=True,
            )
            chunk_out = llm.generate(chunk, sp, use_tqdm=False)
            texts.extend(o.outputs[0].text for o in chunk_out)
        out[cell_key] = texts
    # Reap the engine (gotchas.md vLLM teardown; getattr-guarded canonical helper).
    from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

    _reap_vllm_engine(llm)
    return out


def _generate_tiny(
    tiny_dir: str, jobs: list[tuple[str, list[str], list[str]]]
) -> dict[str, list[str]]:
    """CPU smoke substitute: transformers sampling on a tiny random-init Qwen2.

    SMOKE ONLY (declared): validates prompt construction, stop handling, storage,
    audit, and allowlist plumbing — never the production text distribution.
    min_new_tokens keeps most smoke u2s above the 8-content-token filter so the
    downstream fit has rows.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_dir)
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, torch_dtype=torch.float32)
    model.eval()
    torch.manual_seed(U2_SEED)
    out: dict[str, list[str]] = {}
    for cell_key, prompts, stops in jobs:
        texts = []
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                gen = model.generate(
                    **ids,
                    do_sample=True,
                    temperature=U2_TEMPERATURE,
                    top_p=U2_TOP_P,
                    max_new_tokens=24,
                    min_new_tokens=12,
                    pad_token_id=tokenizer.pad_token_id or 0,
                )
            cont = tokenizer.decode(gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=False)
            for stop in stops:
                if stop in cont:
                    cont = cont.split(stop, 1)[0]
            texts.append(cont)
        out[cell_key] = texts
        print(f"[tiny-gen] {cell_key}: {len(texts)} rows (smoke substitute)")
    return out


# ---------------------------------------------------------------------------
# Gen mode
# ---------------------------------------------------------------------------


def _load_input_conversations(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert row.get("u1") and row.get("a1"), f"row missing u1/a1: {sorted(row)[:6]}"
            rows.append(row)
    assert rows, f"no conversations in {path}"
    return rows


def _update_allowlists(path: Path, updates: dict[str, list]) -> None:
    """Read-modify-write so per-model invocations do not clobber each other."""
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2))
    os.replace(tmp, path)


def run_gen(args) -> None:
    from transformers import AutoTokenizer

    smoke = args.smoke
    tokenizer_src = args.tiny_model_dir or MODEL_INSTRUCT
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    convs = _load_input_conversations(args.conversations)
    if not smoke:
        assert len(convs) == N_TRACK_M, (
            f"expected the FULL kept-{N_TRACK_M} parent conversations, got {len(convs)} "
            "(anchors require the exact parent row set — plan MF-A)"
        )
    n = args.n or (int(os.environ.get("EPS_SMOKE_N", "8")) if smoke else len(convs))
    convs = convs[:n]
    parent_u2 = [c.get("u2", "") for c in convs if c.get("u2")]

    assert_chat_template_wellformed(tokenizer, convs[0]["u1"], convs[0]["a1"])
    print(
        f"[gen] chat-template well-formedness assert PASS on probe conv {convs[0].get('conv_id')}"
    )
    assert_placeholder_span_valid(tokenizer, convs[0]["u1"], convs[0]["a1"])
    print(f"[gen] placeholder span-validity assert PASS ({EMPTY_U2_PLACEHOLDER!r}, both formats)")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    assert set(models) <= set(MODEL_IDS) and set(formats) <= {"chat", "naturalistic"}

    # Parent reference audit stats (model-independent; recomputed per invocation).
    parent_ref = {
        "u2_length": _length_stats(tokenizer, parent_u2),
        "distinct_3gram_rate": _distinct_3gram_rate(parent_u2),
        "distinct_3gram_reference_plan": PARENT_DISTINCT_3GRAM_REFERENCE,
    }

    for model_key in models:
        jobs = []
        for fmt in formats:
            if fmt == "chat":
                prompts = [build_prompt_chat(tokenizer, c["u1"], c["a1"]) for c in convs]
                stops = CHAT_STOPS
            else:
                prompts = [build_prompt_naturalistic(c["u1"], c["a1"]) for c in convs]
                stops = NAT_STOPS
            jobs.append((fmt, prompts, stops))
        if args.tiny_model_dir:
            gen_texts = _generate_tiny(args.tiny_model_dir, jobs)
            backend = f"tiny-substitute ({args.tiny_model_dir})"
        else:
            gen_texts = _generate_vllm(MODEL_IDS[model_key], jobs)
            backend = "vllm"

        allowlist_updates: dict[str, list] = {}
        for fmt in formats:
            raw = [t.strip() for t in gen_texts[fmt]]
            assert len(raw) == len(convs), (model_key, fmt, len(raw), len(convs))
            rows_out, allow, drops, span_degenerate = _process_cell_rows(
                convs, raw, tokenizer, fmt, f"{model_key}/{fmt}"
            )
            if smoke and len(allow) < SMOKE_MIN_ALLOWLIST:
                pad = [c["conv_id"] for c in convs if c["conv_id"] not in set(allow)]
                allow = allow + pad[: SMOKE_MIN_ALLOWLIST - len(allow)]
                print(f"[smoke] {model_key}/{fmt}: allowlist padded to {len(allow)} rows")

            out_path = args.out_dir / f"conversations_{model_key}_{fmt}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                for row in rows_out:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            with out_path.open(encoding="utf-8") as fh:
                n_written = sum(1 for line in fh if line.strip())
            assert n_written == len(convs), f"all-rows contract violated: {n_written}"

            kept_texts = [r["u2"] for r in rows_out if r["conv_id"] in set(allow)]
            all_texts = [t for t in raw if t]
            meta = {
                "followup_label": "onpolicy-user-turn",
                "model": model_key,
                "model_id": MODEL_IDS[model_key],
                "format": fmt,
                "backend": backend,
                "n_rows": len(rows_out),
                "n_allowlist": len(allow),
                "keep_rate": len(allow) / len(rows_out),
                "drops": drops,
                "empty_u2_placeholder": EMPTY_U2_PLACEHOLDER,
                # conv_id + original text of every row whose generated u2
                # rendered a zero-width content span in THIS cell's format and
                # was substituted with the placeholder (run-1 crash-fix).
                "span_degenerate_substituted": span_degenerate,
                "distinct_3gram_rate_kept": _distinct_3gram_rate(kept_texts),
                "distinct_3gram_rate_all": _distinct_3gram_rate(all_texts),
                "repetition_rate": _repetition_rate(all_texts),
                "role_artifact_rate": _role_artifact_rate(all_texts),
                "u2_length_kept": _length_stats(tokenizer, kept_texts),
                "u2_length_all": _length_stats(tokenizer, all_texts),
                "parent_reference": parent_ref,
                "sampling": {
                    "temperature": U2_TEMPERATURE,
                    "top_p": U2_TOP_P,
                    "max_tokens": U2_MAX_TOKENS,
                    "seed": U2_SEED,
                    "stops": CHAT_STOPS if fmt == "chat" else NAT_STOPS,
                },
                "smoke": smoke,
                "git_commit": _git_commit(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            meta_path = args.out_dir / f"conversations_{model_key}_{fmt}_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            print(
                f"[gen] {model_key}/{fmt}: {len(rows_out)} rows -> {out_path} "
                f"(allowlist {len(allow)}, drops={drops})"
            )
            # 20 audit samples to the run log (plan degeneracy-audit requirement).
            for i, r in enumerate(rows_out[:20]):
                u2_short = r["u2"][:300].replace("\n", " ")
                print(f"[sample] {model_key}/{fmt} #{i} conv={r['conv_id']}: {u2_short}")
            allowlist_updates[f"M_{model_key}_user_{fmt}"] = allow
        _update_allowlists(args.out_dir / "row_allowlists.json", allowlist_updates)
        print(f"[gen] row_allowlists.json updated for {sorted(allowlist_updates)}")


# ---------------------------------------------------------------------------
# Wiring check (plan hard-req 2): own- vs shuffled-context NLL of u2
# ---------------------------------------------------------------------------


def _batched_u2_nll(model, tokenizer, convs: list[dict], fmt: str, batch_size: int) -> list[float]:
    """Teacher-forced mean NLL of the u2 span per conversation, batched forwards.

    Shift-by-one (same convention as issue825_extract_turnstore._turn_nll):
    span (s, e) reads token log-probs at logits index [s-1, e-1).
    """
    import numpy as np
    import torch

    rf = _rf()
    renderer = rf.render_chat if fmt == "chat" else rf.render_naturalistic
    rendered = [renderer(c, tokenizer) for c in convs]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    order = sorted(range(len(rendered)), key=lambda j: len(rendered[j].input_ids))
    out = np.zeros(len(rendered), dtype=np.float64)
    device = next(model.parameters()).device
    for pos in range(0, len(order), batch_size):
        chunk_idx = order[pos : pos + batch_size]
        chunk = [rendered[j] for j in chunk_idx]
        lengths = [len(r.input_ids) for r in chunk]
        max_len = max(lengths)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for i, r in enumerate(chunk):
            input_ids[i, : lengths[i]] = torch.tensor(r.input_ids, dtype=torch.long)
            attention_mask[i, : lengths[i]] = 1
        with torch.no_grad():
            logits = model(input_ids.to(device), attention_mask=attention_mask.to(device)).logits
        for i, (j, r) in enumerate(zip(chunk_idx, chunk, strict=True)):
            true_len = lengths[i]
            s, e = r.spans["u2"]
            assert 1 <= s < e <= true_len, (r.conv_id, s, e, true_len)
            lp = torch.log_softmax(logits[i, : true_len - 1].float(), dim=-1)
            targets = input_ids[i, 1:true_len].to(lp.device)
            token_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            out[j] = float(-token_lp[s - 1 : e - 1].mean())
        del logits
    return [float(v) for v in out]


def run_wiring_check(args) -> None:
    import numpy as np
    from issue825_extract_turnstore import load_model

    model_key = args.models.strip()
    assert "," not in model_key, "--wiring-check takes exactly one --models entry"
    allow_map = json.loads((args.out_dir / "row_allowlists.json").read_text())
    model, tokenizer, model_id = load_model(model_key, tiny_model_dir=args.tiny_model_dir)
    result: dict = {
        "followup_label": "onpolicy-user-turn",
        "model": model_key,
        "model_id": model_id,
        "n_rows_requested": args.wiring_rows,
        "seed": U2_SEED,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_format": {},
    }
    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        rows = _load_input_conversations(args.out_dir / f"conversations_{model_key}_{fmt}.jsonl")
        allow = {str(c) for c in allow_map[f"M_{model_key}_user_{fmt}"]}
        rows = [r for r in rows if str(r["conv_id"]) in allow]
        if len(rows) < 2:
            # A catastrophically degenerate cell (<2 allowlisted rows) must not
            # crash the wiring phase BEFORE the wrapper's UPLOAD-2 (plan MF-C):
            # record explicit nulls; the wrapper's POST-upload wiring gate reads
            # own_mean_nll=None as a gross failure and HALTs upload-then-exit.
            print(
                f"[wiring] DEGENERATE {model_key}/{fmt}: {len(rows)} allowlisted rows "
                "(<2) — NLL not computable; recorded null reads for the post-upload gate"
            )
            result["per_format"][fmt] = {
                "cell_id": f"M_{model_key}_user_{fmt}",
                "n": len(rows),
                "own_mean_nll": None,
                "shuffled_mean_nll": None,
                "own_minus_shuffled": None,
                "own_nll_values": [],
                "shuffled_nll_values": [],
                "samples": [],
                "degenerate": "fewer than 2 allowlisted rows — wiring NLL not computable",
            }
            continue
        rng = np.random.default_rng(U2_SEED)
        take = min(args.wiring_rows, len(rows))
        idx = rng.choice(len(rows), size=take, replace=False)
        sub = [rows[int(i)] for i in idx]
        # Derangement: context of row i+1 (mod n) — j != i for every i when n >= 2.
        own_convs = [
            {"conv_id": r["conv_id"], "u1": r["u1"], "a1": r["a1"], "u2": r["u2"]} for r in sub
        ]
        shuf_convs = [
            {
                "conv_id": f"{sub[i]['conv_id']}__ctx_{sub[(i + 1) % take]['conv_id']}",
                "u1": sub[(i + 1) % take]["u1"],
                "a1": sub[(i + 1) % take]["a1"],
                "u2": sub[i]["u2"],
            }
            for i in range(take)
        ]
        # bs 8 matches the extraction path's measured throughput (same forward
        # shape); env-tunable without a code change.
        bs = 2 if args.tiny_model_dir else int(os.environ.get("EPS_WIRING_BS", "8"))
        own = _batched_u2_nll(model, tokenizer, own_convs, fmt, bs)
        shuf = _batched_u2_nll(model, tokenizer, shuf_convs, fmt, bs)
        own_mean = float(np.mean(own))
        shuf_mean = float(np.mean(shuf))
        result["per_format"][fmt] = {
            "cell_id": f"M_{model_key}_user_{fmt}",
            "n": take,
            "own_mean_nll": own_mean,
            "shuffled_mean_nll": shuf_mean,
            "own_minus_shuffled": own_mean - shuf_mean,
            "own_nll_values": own,
            "shuffled_nll_values": shuf,
            "samples": [{"conv_id": r["conv_id"], "u2_excerpt": r["u2"][:200]} for r in sub[:20]],
        }
        print(
            f"[wiring] {model_key}/{fmt}: own {own_mean:.4f} vs shuffled {shuf_mean:.4f} "
            f"(delta {own_mean - shuf_mean:+.4f}, n={take})"
        )
    out_path = args.out_dir / f"wiring_check_{model_key}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wiring] wrote {out_path}")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--conversations", type=Path, default=Path("data/issue_825/conversations.jsonl")
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/issue_825/onpolicy"))
    ap.add_argument("--models", default="instruct,pretrained")
    ap.add_argument("--formats", default="chat,naturalistic")
    ap.add_argument("--n", type=int, default=0, help="0 = all input rows (smoke: EPS_SMOKE_N)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--tiny-model-dir",
        default=None,
        help="SMOKE ONLY: tiny random-init Qwen2 dir (CPU substitute for vLLM/HF 7B)",
    )
    ap.add_argument("--wiring-check", action="store_true")
    ap.add_argument("--wiring-rows", type=int, default=200)
    args = ap.parse_args()
    if args.wiring_check:
        run_wiring_check(args)
    else:
        run_gen(args)


if __name__ == "__main__":
    main()
