"""Issue #1345 Phase A/C — scaffold generation + prefill continuation.

Phase A (``--phase scaffolds``): generate DIVERSE narrative scenes containing
exactly ONE question utterance and the literal answer-slot sentinel
(``issue1345_scaffold_common.SLOT_SENTINEL``) with NO answer text. Diversity
is load-bearing (a hardcoded-template scaffold variant measured R^2 0.019 vs
0.37 for diverse prose): every row samples a (setting, situation, register)
triple from the issue1310 scenario axes under a seeded RNG, and sampling runs
at temperature 1.0. Tier-2 instruct-and-strip: the scaffold-writing
instruction lives only in the generation prompt; downstream consumers read
the rendered story text (splice output), never the prompt.

Phase C (``--phase prefill``): render each scaffold truncated at the slot
with the chosen boundary form's opening (``render_prefill``), feed it as a
RAW continuation prefix (the story_slot precedent in
issue1345_onpolicy_answers_gen — no chat template: the story-so-far IS the
document being continued), and let the model write the answer. The answer
span is BY CONSTRUCTION everything generated up to the form's stop string;
the final text + exact offsets come from the SAME ``splice_answer`` renderer
Phase B uses. 100% keep by construction — no judge, no verbatim matcher;
the only recorded drops are degenerate rows (empty answer, BPE zero-width
token span), counted per reason and persisted with their raw attempts.

Every raw attempt — keeps AND rejects — is persisted to the raw JSONL the
moment its chunk completes (#1689 lost its rejects irrecoverably; never
again). Resume is fingerprint-gated per row id, mirroring
issue1345_gen_stories_paired.generate_paired.

Mock seam: ``run_scaffold_phase`` / ``run_prefill_phase`` accept
``gen_fn=None`` (the GPU boundary — tests and the no-GPU smoke inject
deterministic fakes via ``--mock``, which refuses to write into default
production dirs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# vLLM v1 EngineCore forks poisoned parent state under the default fork
# method (gotchas.md #628) — set BEFORE any vllm import (imports deferred).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1310_common as p10  # noqa: E402
import issue1345_common as c  # noqa: E402
import issue1345_scaffold_common as sc  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (grounded on the parent story arms; deviations commented)
# ---------------------------------------------------------------------------
# Same env knob + default as issue1345_gen_stories.VLLM_CHUNK_SIZE (defined
# locally so this entrypoint does not pull the api_dispatch import chain).
VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "250"))
MAX_MODEL_LEN = 4096  # parent gen budget (issue1345_gen_stories.MAX_MODEL_LEN)
SCAFFOLD_MAX_NEW_TOKENS = 1024  # parent story cap (c.STORY_MAX_NEW_TOKENS); 150-300 words fits 2x+
PREFILL_MAX_NEW_TOKENS = 1024  # answer continuation cap; cap-hit fraction reported in the digest

# The character panel: the issue1310 personas plus the default assistant
# (the standing panel for the character arms).
CHARACTERS: dict[str, str] = {
    **p10.PERSONAS,
    "Assistant": "a helpful, knowledgeable AI assistant",
}

# Register/style axis (third diversity lever beside setting x situation).
REGISTERS = (
    "close third person, present tense, plainspoken",
    "close third person, past tense, literary and atmospheric",
    "first person bystander narration, wry and observational",
    "sparse, hard-boiled prose with short sentences",
    "warm, domestic realism with sensory detail",
    "tense thriller pacing, clipped dialogue beats",
    "gentle comedic register with light irony",
    "quiet melancholic tone, long flowing sentences",
)


def scaffold_system_prompt(char_name: str, description: str, spec: dict) -> str:
    """The tier-2 scaffold-writing instruction (stripped by construction)."""
    return (
        f"You are writing the opening of a short story in which {char_name}, "
        f"{description}, is a character. Setting: {spec['setting']}. "
        f"Situation: {spec['situation']}. Style: {spec['register']}.\n"
        "Write a realistic narrative scene of 150-300 words in flowing prose (never a "
        "script or 'Name:' dialogue format), with spoken dialogue in double quotes. In "
        f"the scene, one person asks {char_name} exactly ONE question (a genuine "
        "question, in double quotes, ending with '?'). No one else asks "
        f"{char_name} anything. Do NOT write {char_name}'s answer: at the exact point "
        f"where {char_name}'s answer would appear, write the placeholder "
        f"{sc.SLOT_SENTINEL} alone as its own paragraph, then close the scene with at "
        "most two more sentences of narration. The placeholder must appear exactly "
        "once, and no answer content may appear anywhere."
    )


def build_scaffold_prompt(row: dict, tokenizer, char_name: str, description: str) -> str:
    """Chat-templated Phase-A prompt (parent convention: both model kinds)."""
    user_msg = "Write the scene now."
    if row.get("question"):
        user_msg += (
            f" The question the person asks {char_name} is:\n{row['question']}\n"
            "Use this question verbatim as the one question in the scene."
        )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": scaffold_system_prompt(char_name, description, row)},
            {"role": "user", "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def make_scaffold_specs(
    n: int, seed: int, questions: list[dict] | None, char_name: str
) -> list[dict]:
    """Deterministic per-row diversity specs (setting x situation x register)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    specs = []
    for i in range(n):
        spec = {
            "scaffold_id": f"{char_name.lower()}_scaffold_{i:05d}",
            "setting": p10.SETTINGS[int(rng.integers(len(p10.SETTINGS)))],
            "situation": p10.SITUATIONS[int(rng.integers(len(p10.SITUATIONS)))],
            "register": REGISTERS[int(rng.integers(len(REGISTERS)))],
        }
        if questions is not None:
            q = questions[i % len(questions)]
            spec["question"] = q["question"]
            spec["qid"] = q.get("qid", q.get("conv_id", str(i % len(questions))))
        specs.append(spec)
    return specs


def gen_fingerprint(kind: str, **fields) -> str:
    """Bundle identity for resume gating (a recipe change re-keys the bundle)."""
    key = json.dumps({"kind": kind, "sentinel": sc.SLOT_SENTINEL, **fields}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Chunked generation with per-chunk checkpoint + fingerprint-gated resume
# (the generate_paired contract: gen_fn returns ALL rows in the touched file)
# ---------------------------------------------------------------------------
def _resume_done_ids(out_path: Path, fp: str, id_key: str) -> set[str]:
    meta_path = out_path.with_suffix(".meta.json")
    if out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fp:
            raise RuntimeError(
                f"{out_path} exists with a DIFFERENT generation fingerprint "
                f"({meta.get('fingerprint')} != {fp}) — refusing to mix regimes; "
                "move the stale file aside"
            )
        done = {r[id_key] for r in c.read_jsonl(out_path)}
        print(f"[gen-scaffolds] resume: {len(done)} rows already on disk", flush=True)
        return done
    c.write_json(meta_path, {"fingerprint": fp})
    return set()


def _chunked_generate(rows, out_path, fp, prompts_fn, row_out_fn, llm, sampling, id_key):
    """Generate rows chunked, appending each chunk's raw rows immediately."""
    done = _resume_done_ids(out_path, fp, id_key)
    todo = [r for r in rows if r[id_key] not in done]
    n_chunks = (len(todo) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for ci in range(0, len(todo), VLLM_CHUNK_SIZE):
        chunk = todo[ci : ci + VLLM_CHUNK_SIZE]
        prompts = [prompts_fn(r) for r in chunk]
        print(
            f"[vllm-chunk] gen chunk {ci // VLLM_CHUNK_SIZE + 1}/{n_chunks} "
            f"({len(chunk)} prompts) -> {out_path.name}",
            flush=True,
        )
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        new_rows = [
            row_out_fn(r, o.outputs[0].text, o.outputs[0].finish_reason)
            for r, o in zip(chunk, outs, strict=True)
        ]
        c.append_jsonl(out_path, new_rows)
    return c.read_jsonl(out_path) if out_path.exists() else []


# ---------------------------------------------------------------------------
# Phase A — scaffolds
# ---------------------------------------------------------------------------
def validate_scaffold_text(text: str) -> tuple[bool, str, dict]:
    """Structural keep-filter (span-locatability by CONSTRUCTION, not judging).

    The only HARD requirement is the exactly-one-sentinel invariant (without
    it the splice is undefined); the question flag is recorded, not gated.
    """
    n_sent = sc.count_sentinels(text)
    flags = {"sentinel_count": n_sent, "question_before_slot": False}
    if n_sent != 1:
        return False, "sentinel_count", flags
    idx = text.index(sc.SLOT_SENTINEL)
    flags["question_before_slot"] = "?" in text[:idx]
    if idx == 0:
        return False, "sentinel_at_start", flags
    return True, "ok", flags


def run_scaffold_phase(
    *,
    specs: list[dict],
    out_dir: Path,
    char_name: str,
    description: str,
    model_key: str,
    tokenizer,
    llm,
    seed: int,
    gen_fn=None,
) -> dict:
    """Generate + validate scaffolds; persist ALL raw attempts + keeps + digest."""
    slug = char_name.lower()
    raw_path = out_dir / f"raw_scaffolds_{slug}_{model_key}.jsonl"
    fp = gen_fingerprint(
        "scaffolds_v1",
        char=char_name,
        description=description,
        model=model_key,
        seed=seed,
        n=len(specs),
        temperature=c.STORY_TEMPERATURE,
        max_new_tokens=SCAFFOLD_MAX_NEW_TOKENS,
        questions_sha=hashlib.sha256(
            json.dumps([s.get("question", "") for s in specs]).encode()
        ).hexdigest()[:12],
    )

    if gen_fn is not None:
        raw_rows = gen_fn(specs, raw_path, fp)
    else:
        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=c.STORY_TEMPERATURE, max_tokens=SCAFFOLD_MAX_NEW_TOKENS, seed=None
        )
        raw_rows = _chunked_generate(
            specs,
            raw_path,
            fp,
            prompts_fn=lambda r: build_scaffold_prompt(r, tokenizer, char_name, description),
            row_out_fn=lambda r, text, fin: {
                **r,
                "character": char_name,
                "model": model_key,
                "tier": "instruct_and_strip",
                "scaffold_text": text.strip(),
                "finish_reason": fin,
            },
            llm=llm,
            sampling=sampling,
            id_key="scaffold_id",
        )

    keeps, counts = [], {"total": len(raw_rows), "kept": 0, "cap_hit": 0}
    reasons: dict[str, int] = {}
    for r in raw_rows:
        keep, reason, flags = validate_scaffold_text(r["scaffold_text"])
        r = {**r, "keep": keep, "reject_reason": None if keep else reason, **flags}
        if r.get("finish_reason") == "length":
            counts["cap_hit"] += 1
        if keep:
            counts["kept"] += 1
            keeps.append(r)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    kept_path = out_dir / f"scaffolds_{slug}_{model_key}.jsonl"
    kept_path.unlink(missing_ok=True)
    c.append_jsonl(kept_path, keeps)
    digest = {
        "phase": "scaffolds",
        "character": char_name,
        "model": model_key,
        "fingerprint": fp,
        "counts": counts,
        "reject_reasons": reasons,
        "cap_hit_fraction": (counts["cap_hit"] / counts["total"]) if counts["total"] else 0.0,
        "metadata": c.metadata(seed, len(specs), Path(__file__).name),
    }
    c.write_json(out_dir / f"scaffold_yield_{slug}_{model_key}.json", digest)
    print(f"[gen-scaffolds] scaffolds done: {counts} rejects={reasons}", flush=True)
    return digest


# ---------------------------------------------------------------------------
# Phase C — prefill continuation
# ---------------------------------------------------------------------------
def run_prefill_phase(
    *,
    scaffolds: list[dict],
    out_dir: Path,
    char_name: str,
    model_key: str,
    form: str,
    tokenizer,
    llm,
    seed: int,
    gen_fn=None,
) -> dict:
    """Continue each scaffold at its slot; the answer span is what was generated."""
    if form == "indirect":
        # Surface the same NotImplemented path the splice renderer defines.
        sc.render_prefill(sc.SLOT_SENTINEL, form, char_name)
    slug = char_name.lower()
    raw_path = out_dir / f"raw_prefill_{form}_{slug}_{model_key}.jsonl"
    fp = gen_fingerprint(
        "prefill_v1",
        char=char_name,
        model=model_key,
        form=form,
        seed=seed,
        temperature=c.STORY_TEMPERATURE,
        max_new_tokens=PREFILL_MAX_NEW_TOKENS,
        scaffolds_sha=hashlib.sha256(
            json.dumps([s["scaffold_id"] for s in scaffolds]).encode()
        ).hexdigest()[:12],
    )
    stop = sc.render_prefill(scaffolds[0]["scaffold_text"], form, char_name).stop

    if gen_fn is not None:
        raw_rows = gen_fn(scaffolds, raw_path, fp)
    else:
        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=c.STORY_TEMPERATURE,
            max_tokens=PREFILL_MAX_NEW_TOKENS,
            stop=list(stop),
            seed=None,
        )
        raw_rows = _chunked_generate(
            scaffolds,
            raw_path,
            fp,
            # RAW continuation prefix — the story_slot precedent (no chat template).
            prompts_fn=lambda r: sc.render_prefill(r["scaffold_text"], form, char_name).prefix_text,
            row_out_fn=lambda r, text, fin: {
                "scaffold_id": r["scaffold_id"],
                "character": char_name,
                "model": model_key,
                "form": form,
                "scaffold_text": r["scaffold_text"],
                "answer": text,
                "finish_reason": fin,
            },
            llm=llm,
            sampling=sampling,
            id_key="scaffold_id",
        )

    keeps, counts = [], {"total": len(raw_rows), "kept": 0, "cap_hit": 0}
    reasons: dict[str, int] = {}
    for r in raw_rows:
        answer = r["answer"].strip()
        keep, reason = True, None
        if not answer:
            keep, reason = False, "empty_answer"
        else:
            spliced = sc.splice_answer(r["scaffold_text"], answer, form, char_name)
            r = {
                **r,
                # `answer` is the SPLICED text the offsets point at; the raw
                # generation (pre-strip) stays alongside for provenance.
                "answer": answer,
                "answer_raw": r["answer"],
                "final_text": spliced.text,
                "answer_start": spliced.answer_start,
                "answer_end": spliced.answer_end,
            }
            if tokenizer is not None:
                if not sc.token_span_ok(
                    spliced.text, spliced.answer_start, spliced.answer_end, tokenizer
                ):
                    keep, reason = False, "span_token_degenerate"
            else:
                r["span_token_ok"] = None  # mock path: no tokenizer available
        if r.get("finish_reason") == "length":
            counts["cap_hit"] += 1
        r = {**r, "keep": keep, "reject_reason": reason}
        if keep:
            counts["kept"] += 1
            keeps.append(r)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    kept_path = out_dir / f"prefill_{form}_{slug}_{model_key}.jsonl"
    kept_path.unlink(missing_ok=True)
    c.append_jsonl(kept_path, keeps)
    digest = {
        "phase": "prefill",
        "form": form,
        "character": char_name,
        "model": model_key,
        "fingerprint": fp,
        "stop": list(stop),
        "counts": counts,
        "reject_reasons": reasons,
        "cap_hit_fraction": (counts["cap_hit"] / counts["total"]) if counts["total"] else 0.0,
        "metadata": c.metadata(seed, len(scaffolds), Path(__file__).name),
    }
    c.write_json(out_dir / f"prefill_yield_{form}_{slug}_{model_key}.json", digest)
    print(f"[gen-scaffolds] prefill done: {counts} rejects={reasons}", flush=True)
    return digest


# ---------------------------------------------------------------------------
# Mock generators (deterministic; exercise keep AND reject paths; no GPU)
# ---------------------------------------------------------------------------
def mock_scaffold_gen(specs: list[dict], out_path: Path, fp: str) -> list[dict]:
    """Deterministic scaffold texts; every 5th row breaks the sentinel invariant."""
    done = _resume_done_ids(out_path, fp, "scaffold_id")
    new_rows = []
    for i, r in enumerate(r for r in specs if r["scaffold_id"] not in done):
        q = r.get("question", "What do we do when the water rises?")
        body = (
            f"Rain hammered the roof of {r['setting']}. Word was {r['situation']}. "
            f'Mara turned and asked, "{q}"\n\n{sc.SLOT_SENTINEL}\n\nThe lamp guttered.'
        )
        if i % 5 == 4:
            body = body.replace(sc.SLOT_SENTINEL, "...")  # sentinel missing -> reject
        new_rows.append(
            {
                **r,
                "character": "mock",
                "model": "mock",
                "tier": "instruct_and_strip",
                "scaffold_text": body,
                "finish_reason": "stop",
            }
        )
    c.append_jsonl(out_path, new_rows)
    return c.read_jsonl(out_path) if out_path.exists() else []


def mock_prefill_gen(scaffolds: list[dict], out_path: Path, fp: str) -> list[dict]:
    """Deterministic answers; every 7th row is empty (degenerate-drop path)."""
    done = _resume_done_ids(out_path, fp, "scaffold_id")
    new_rows = []
    for i, r in enumerate(r for r in scaffolds if r["scaffold_id"] not in done):
        answer = "" if i % 7 == 6 else f"We move the grain to the loft tonight, row {i}."
        new_rows.append(
            {
                "scaffold_id": r["scaffold_id"],
                "character": r.get("character", "mock"),
                "model": "mock",
                "form": "mock",
                "scaffold_text": r["scaffold_text"],
                "answer": answer,
                "finish_reason": "stop",
            }
        )
    c.append_jsonl(out_path, new_rows)
    return c.read_jsonl(out_path) if out_path.exists() else []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument("--phase", choices=("scaffolds", "prefill"), required=True)
    ap.add_argument("--character", required=True, help=f"one of {sorted(CHARACTERS)} or custom")
    ap.add_argument("--description", default=None, help="required for a non-panel character")
    ap.add_argument("--model", choices=("instruct", "pretrained"), default="instruct")
    ap.add_argument("--n", type=int, default=200, help="scaffold count (scaffolds phase)")
    ap.add_argument("--seed", type=int, default=c.GEN_SEED if hasattr(c, "GEN_SEED") else 42)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--questions-jsonl", type=Path, default=None)
    ap.add_argument("--scaffolds-jsonl", type=Path, default=None, help="prefill input rows")
    ap.add_argument("--form", choices=sc.BOUNDARY_FORMS, default="attrib_quoted")
    ap.add_argument("--limit", type=int, default=None, help="cap prefill input rows")
    ap.add_argument(
        "--mock",
        action="store_true",
        help="deterministic mock generator (no vLLM/GPU); requires an explicit --out-dir",
    )
    args = ap.parse_args()

    if args.description is None:
        if args.character not in CHARACTERS:
            ap.error(f"--description required for non-panel character {args.character!r}")
        args.description = CHARACTERS[args.character]
    if args.mock and args.out_dir is None:
        ap.error("--mock requires an explicit --out-dir (never the production default)")
    out_dir = args.out_dir or (c.DATA_DIR / "scaffolds")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = llm = None
    if not args.mock:
        from transformers import AutoTokenizer

        from explore_persona_space.experiments.issue_825.common import (
            MODEL_INSTRUCT,
            MODEL_PRETRAINED,
        )

        model_id = MODEL_INSTRUCT if args.model == "instruct" else MODEL_PRETRAINED
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        from vllm import LLM

        llm = LLM(
            model=model_id,
            seed=args.seed,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.85,
            enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1",
            enable_prefix_caching=(
                False if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "0") == "1" else None
            ),
        )

    if args.phase == "scaffolds":
        questions = None
        if args.questions_jsonl is not None:
            questions = c.read_jsonl(args.questions_jsonl)
            assert questions and all("question" in q for q in questions), (
                f"--questions-jsonl rows must carry a 'question' field: {args.questions_jsonl}"
            )
        specs = make_scaffold_specs(args.n, args.seed, questions, args.character)
        run_scaffold_phase(
            specs=specs,
            out_dir=out_dir,
            char_name=args.character,
            description=args.description,
            model_key="mock" if args.mock else args.model,
            tokenizer=tokenizer,
            llm=llm,
            seed=args.seed,
            gen_fn=mock_scaffold_gen if args.mock else None,
        )
    else:
        assert args.scaffolds_jsonl is not None, "--scaffolds-jsonl is required for prefill"
        scaffolds = [r for r in c.read_jsonl(args.scaffolds_jsonl) if r.get("keep", True)]
        if args.limit:
            scaffolds = scaffolds[: args.limit]
        assert scaffolds, f"no usable scaffold rows in {args.scaffolds_jsonl}"
        run_prefill_phase(
            scaffolds=scaffolds,
            out_dir=out_dir,
            char_name=args.character,
            model_key="mock" if args.mock else args.model,
            form=args.form,
            tokenizer=tokenizer,
            llm=llm,
            seed=args.seed,
            gen_fn=mock_prefill_gen if args.mock else None,
        )
    # Explicit success exit: heavy C-extension teardown must not rewrite rc
    # (gotchas.md PyGILState_Release atexit race).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
