#!/usr/bin/env python
"""Issue #1336 — Phase G: on-policy answer generation (vLLM) per (model, corpus).

Parent-exact Track-S sampling (Source: scripts/issue825_gen_conversations.py:521
— ``SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1024, seed=42)``)
under the shared Tulu chat template, for all 5 ladder checkpoints.

Modes:
  --prep                stage the corpora (pinned #825 track_s prompts +
                        pinned openai/gsm8k splits) into per-corpus prompt
                        files with the load-time token-budget gate (#952).
  --model <slug>        generate answers for every requested corpus with ONE
                        vLLM engine (Phase G shards models across GPUs).
  --gen-format          generation prompt format (round 5). ``chat`` (default)
                        is byte-identical to every prior round: Tulu template,
                        chat stop strings, bare-corpus output dirs/HF prefixes.
                        ``naturalistic`` generates ON-POLICY under the #825
                        plain-transcript prefix (``common.natural_prompt``)
                        with naturalistic stop handling, landing in
                        format-keyed dirs (``<corpus>__gen_naturalistic``) so
                        the two arms can never overwrite each other. The
                        render-integrity gate is format-conditional: hard gate
                        in the matched-text (chat) regime, logged diagnostic
                        in the on-policy naturalistic regime.

Row filters (plan §4): >=8 content tokens per turn, <=2048 total rendered
tokens, validated with the extract consumer's EXACT render asserts
(issue1336_render.validate_render) for every format the cell will use.
Keep-rate floor 0.80 per (model, corpus): below-floor is REPORTED (fits run
at realized n), never padded. ALL rollout text is persisted (kept + dropped
rows) before any reduction; audits are digest-only (no row text — lmsys is a
real-world corpus).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + HF token must bind before heavy imports

# vLLM v1 EngineCore dies silently under fork() when the parent touched
# CUDA-adjacent code (tokenizers) before LLM() — set BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from issue1336_render import RENDERERS, render_integrity_gate, validate_render  # noqa: E402
from issue1336_stage_corpora import V2_CORPORA, load_v2_corpus_rows  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))

DATA_ROOT = Path("data/issue_1336")


def _out_root(smoke: bool) -> Path:
    return DATA_ROOT / ("gen_smoke" if smoke else "gen")


def _prompts_root(smoke: bool) -> Path:
    return DATA_ROOT / ("prompts_smoke" if smoke else "prompts")


def _read_jsonl(path: Path) -> list[dict]:
    """Text-mode line iteration — never splitlines() (U+2028 in real user text)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"[gen] wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# --prep: corpus staging (CPU; model-independent — ids identical across the
# ladder's tokenizers, verified at plan time + re-asserted per cell at extract)
# ---------------------------------------------------------------------------
def _stage_lmsys_prompts() -> list[dict]:
    """Fetch the pinned #825 Track-S prompt file and extract the 5,000 prompts."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=cm.HF_DATA_REPO,
        repo_type="dataset",
        filename=cm.TRACK_S_PATH,
        revision=cm.TRACK_S_REV,
        local_dir=DATA_ROOT / "hf_dl",
    )
    nbytes = Path(local).stat().st_size
    assert nbytes == cm.TRACK_S_BYTES, (
        f"track_s.jsonl byte size {nbytes} != pinned {cm.TRACK_S_BYTES} at rev "
        f"{cm.TRACK_S_REV} — content-identity check failed"
    )
    rows = _read_jsonl(Path(local))
    assert len(rows) == cm.CORPORA["lmsys5k"]["n"], f"track_s rows {len(rows)} != 5000"
    return [{"prompt_idx": int(r["prompt_idx"]), "prompt": r["prompt"]} for r in rows]


def _stage_gsm8k(split: str, n: int) -> list[dict]:
    """Load the pinned openai/gsm8k split; deterministic prefix of size n."""
    from datasets import load_dataset

    ds = load_dataset(cm.GSM8K_DATASET, cm.GSM8K_CONFIG, split=split, revision=cm.GSM8K_REV)
    assert len(ds) == cm.GSM8K_SPLIT_SIZES[split], (
        f"gsm8k {split} split has {len(ds)} rows != pinned {cm.GSM8K_SPLIT_SIZES[split]}"
    )
    rows = [{"prompt_idx": i, "prompt": ds[i]["question"]} for i in range(n)]
    return rows


def _stage_v2_corpus(corpus: str, smoke: bool, explicit_path: str | None) -> list[dict]:
    """Stage a v2 corpus (built by ``issue1336_stage_corpora.py``) into the
    same {prompt_idx, prompt} row shape the v1 corpora stage to."""
    rows = load_v2_corpus_rows(corpus, explicit_path=explicit_path, smoke=smoke)
    assert rows, f"v2 corpus {corpus} resolved to 0 rows"
    return [{"prompt_idx": int(r["prompt_idx"]), "prompt": r["prompt"]} for r in rows]


def new_rows_only(corpus: str, rows: list[dict]) -> list[dict]:
    """Extension-only rows for a concat corpus (plan v13 §4 Phase GEN).

    Wave-1 covered prompt_idx < cm.V2_CONCAT_BOUNDARY[corpus] and its rollout
    text is reused verbatim, so generation stages ONLY the new rows. Non-concat
    corpora pass through unchanged. Fail-loud on an empty result (a corpus
    built before the production-anchored extension idx convention).
    """
    if corpus not in cm.V2_CONCAT_BOUNDARY:
        return rows
    boundary = cm.V2_CONCAT_BOUNDARY[corpus]
    kept = [r for r in rows if int(r["prompt_idx"]) >= boundary]
    assert kept, (
        f"{corpus}: no new prompts (prompt_idx >= {boundary}) — corpus build "
        "predates the production-anchored extension idx convention"
    )
    return kept


def run_prep(corpora: list[str], smoke: bool, corpus_files: dict[str, str] | None = None) -> None:
    """Stage every requested corpus + apply the load-time prompt-token gate."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cm.MODELS["base"]["hf_id"])
    proot = _prompts_root(smoke)
    proot.mkdir(parents=True, exist_ok=True)
    for corpus in corpora:
        out = proot / f"{corpus}.jsonl"
        meta_path = proot / f"{corpus}_meta.json"
        if out.exists() and meta_path.exists():
            print(f"[prep] skip {corpus} (exists)")
            continue
        if corpus == "lmsys5k":
            rows = _stage_lmsys_prompts()
        elif corpus in cm.CORPORA:  # v1 registry FIRST (default-preserving)
            spec = cm.CORPORA[corpus]
            rows = _stage_gsm8k(spec["split"], spec["n"])
        elif corpus in V2_CORPORA:
            rows = _stage_v2_corpus(corpus, smoke, (corpus_files or {}).get(corpus))
        else:
            raise KeyError(
                f"unknown corpus {corpus!r} — not in cm.CORPORA nor "
                f"issue1336_stage_corpora.V2_CORPORA ({sorted(set(cm.CORPORA) | set(V2_CORPORA))})"
            )
        if corpus in cm.V2_CONCAT_BOUNDARY:
            # Phase GEN generates NEW prompts only (plan v13 §4). Filter
            # BEFORE the smoke slice so both modes stage the same
            # extension-only population (smoke corpora anchor extension rows
            # at the production boundary — Unit D).
            n_all = len(rows)
            rows = new_rows_only(corpus, rows)
            print(
                f"[prep] {corpus}: new-rows filter "
                f"(idx >= {cm.V2_CONCAT_BOUNDARY[corpus]}): kept {len(rows)}/{n_all}"
            )
        if smoke and corpus not in V2_CORPORA:
            # v2 smoke corpora are already builder-sized (SMOKE_SAMPLE_N /
            # SMOKE_LMSYS_NEW_N — the latter deliberately ABOVE cm.SMOKE_N so
            # the kept stem clears the fit's n floor after real-prompt render
            # drops); slicing them back to SMOKE_N re-floors the fit smoke.
            rows = rows[: cm.SMOKE_N]
        kept, dropped = [], []
        for r in rows:
            # Budget gate on the FORMATTED prompt exactly as generation renders
            # it (+1 for the BOS vLLM adds) — #952 load-time length validation.
            n_tok = len(tok(cm.tulu_prompt(r["prompt"]), add_special_tokens=False)["input_ids"])
            if n_tok + 1 > cm.PROMPT_TOKEN_BUDGET:
                dropped.append({"prompt_idx": r["prompt_idx"], "n_tokens": n_tok})
            else:
                kept.append(r)
        _write_jsonl(out, kept)
        meta = {
            "corpus": corpus,
            "n_source": len(rows),
            "n_kept": len(kept),
            "dropped_over_budget": dropped,  # digest-only: idx + token count
            "prompt_token_budget": cm.PROMPT_TOKEN_BUDGET,
            "smoke": smoke,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[prep] {corpus}: kept {len(kept)}/{len(rows)} (over-budget: {len(dropped)})")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _formats_for(corpus: str) -> tuple[str, ...]:
    """Formats for a corpus: the v1 registry first (default-preserving —
    identical lookups for every pre-existing corpus), then the v2 corpus
    registry (``issue1336_stage_corpora.V2_CORPORA``)."""
    if corpus in cm.FORMATS_BY_CORPUS:
        return cm.FORMATS_BY_CORPUS[corpus]
    return tuple(V2_CORPORA[corpus]["formats"])


def _assert_template_parity(tokenizer, prompts: list[str]) -> None:
    """Our pinned render must equal the checkpoint's own chat template."""
    if tokenizer.chat_template is None:
        return  # base ships no template; rendered with the shared string (plan §4)
    for q in prompts[:3]:
        ref = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        if tokenizer.bos_token and ref.startswith(tokenizer.bos_token):
            ref = ref[len(tokenizer.bos_token) :]
        assert ref == cm.tulu_prompt(q), (
            f"pinned Tulu template diverges from tokenizer.apply_chat_template — got {ref[:80]!r}"
        )


def _truncate_role_headers(text: str, markers: tuple[str, ...]) -> str:
    """Post-hoc truncation at the first role-header reoccurrence (plan §4).

    ``markers`` is the gen format's marker set: ``cm.ROLE_HEADER_TRUNCATE``
    (chat — behavior byte-identical to prior rounds) or
    ``cm.NATURAL_ROLE_HEADER_TRUNCATE`` (on-policy naturalistic arm).
    """
    cut = len(text)
    for marker in markers:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def _rep3_flag(text: str) -> bool:
    """True when any word 3-gram repeats >=3 times (degeneration audit flag)."""
    words = text.split()
    counts: collections.Counter = collections.Counter(
        tuple(words[j : j + 3]) for j in range(len(words) - 2)
    )
    return bool(counts) and max(counts.values()) >= 3


def _distinct_3gram_rate(texts: list[str]) -> float:
    """Corpus-level distinct-3-gram rate (the #825 rounds 7-8 audit)."""
    distinct: set[tuple[str, ...]] = set()
    total = 0
    for t in texts:
        words = t.split()
        for j in range(len(words) - 2):
            total += 1
            distinct.add(tuple(words[j : j + 3]))
    return (len(distinct) / total) if total else 0.0


def _vllm_generate_chunked(llm, prompt_texts: list[str], sampling) -> list[tuple[str, str]]:
    """Chunked generate (vLLM large-batch deadlock prevention) -> (text, finish)."""
    out: list[tuple[str, str]] = []
    n_chunks = (len(prompt_texts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for i in range(0, len(prompt_texts), VLLM_CHUNK_SIZE):
        chunk = prompt_texts[i : i + VLLM_CHUNK_SIZE]
        print(
            f"[vllm-chunk] chunk {i // VLLM_CHUNK_SIZE + 1}/{n_chunks} ({len(chunk)} prompts)",
            flush=True,
        )
        for o in llm.generate(chunk, sampling, use_tqdm=False):
            out.append((o.outputs[0].text, str(o.outputs[0].finish_reason)))
    return out


def _hf_gen_prefix(slug: str, cell: str) -> str:
    """Hub prefix for one generation cell. ``cell`` is the format-keyed token
    (``cm.gen_cell_key``): the bare corpus for chat (byte-identical to every
    prior round's prefixes), ``<corpus>__gen_naturalistic`` for the on-policy
    naturalistic arm."""
    return f"{cm.HF_PREFIX_1336}/raw_completions/generation/{slug}/{cell}"


ANSWERS_MANIFEST = "answers.manifest.json"
_GEN_SIDE_FILES = ("allowlist.json", "audit.json")
# Upload-policy text sharding: the Hub force-routes any >10 MB blob to LFS
# (quota-exposed, #541); text >9.5 MB line-splits into <9 MB shards, NEVER gzip.
_TEXT_SPLIT_THRESHOLD = 9_500_000
_SHARD_MAX_BYTES = 9_000_000


def _download_one(prefix_file: str) -> Path:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    return Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                repo_id=cm.HF_DATA_REPO,
                repo_type="dataset",
                filename=prefix_file,
                local_dir=DATA_ROOT / "hf_dl",
            ),
            what=f"gen download {prefix_file}",
        )
    )


def _hf_gen_state(slug: str, cell: str) -> tuple[bool, dict | None]:
    """(complete, manifest) — Hub-side completeness of one cell's gen outputs.

    Complete <=> both side files present AND the answers text present as
    either the single ``answers.jsonl`` (manifest None) or the sharded form
    (``answers.manifest.json`` + every part it lists). ONE scoped + retried
    tree walk (hub.list_hf_files_under_path; an absent prefix returns []) —
    never bare per-file probes (#920 un-retried-verify class).
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    prefix = _hf_gen_prefix(slug, cell)
    files = set(hub.list_hf_files_under_path(HfApi(), cm.HF_DATA_REPO, prefix, repo_type="dataset"))
    if not all(f"{prefix}/{n}" in files for n in _GEN_SIDE_FILES):
        return False, None
    if f"{prefix}/answers.jsonl" in files:
        return True, None
    if f"{prefix}/{ANSWERS_MANIFEST}" not in files:
        return False, None
    manifest = json.loads(_download_one(f"{prefix}/{ANSWERS_MANIFEST}").read_text())
    return all(f"{prefix}/{p}" in files for p in manifest["parts"]), manifest


def _reassemble_answers(out_dir: Path, prefix: str, manifest: dict) -> None:
    """Concatenate downloaded shard parts back into answers.jsonl (sha-verified)."""
    tmp = out_dir / "answers.jsonl.tmp"
    with open(tmp, "wb") as fh:
        for part, sha in zip(manifest["parts"], manifest["sha256s"], strict=True):
            data = _download_one(f"{prefix}/{part}").read_bytes()
            got = hashlib.sha256(data).hexdigest()
            assert got == sha, f"shard {part}: sha256 {got} != manifest {sha}"
            fh.write(data)
    total = hashlib.sha256(tmp.read_bytes()).hexdigest()
    assert total == manifest["total_sha256"], (
        f"reassembled answers.jsonl sha256 {total} != manifest {manifest['total_sha256']}"
    )
    os.replace(tmp, out_dir / "answers.jsonl")


def _try_hf_resume(slug: str, cell: str, out_dir: Path) -> bool:
    """Fetch a prior run's generation outputs from the Hub instead of re-generating."""
    complete, manifest = _hf_gen_state(slug, cell)
    if not complete:
        return False
    prefix = _hf_gen_prefix(slug, cell)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in _GEN_SIDE_FILES:
        (out_dir / n).write_bytes(_download_one(f"{prefix}/{n}").read_bytes())
    if manifest is None:
        (out_dir / "answers.jsonl").write_bytes(
            _download_one(f"{prefix}/answers.jsonl").read_bytes()
        )
    else:
        _reassemble_answers(out_dir, prefix, manifest)
    print(f"[gen] HF-resume: fetched {prefix} -> {out_dir}")
    return True


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_answers_for_upload(out_dir: Path) -> bool:
    """Line-split answers.jsonl into <9 MB shards + manifest when over 9.5 MB.

    Local consumers keep reading the single ``answers.jsonl``; the shards +
    manifest exist for the Hub upload only (non-LFS path). Idempotent: stale
    shards from a prior split are removed before re-splitting; a file at or
    under threshold clears any stale shard/manifest files. Returns True when
    the sharded form is the upload shape.
    """
    src = out_dir / "answers.jsonl"
    stale = [*out_dir.glob("answers.shard*.jsonl"), out_dir / ANSWERS_MANIFEST]
    for old in stale:
        if old.exists():
            old.unlink()
    if src.stat().st_size <= _TEXT_SPLIT_THRESHOLD:
        return False
    parts: list[str] = []
    line_counts: list[int] = []
    shas: list[str] = []
    buf: list[bytes] = []
    size = 0

    def _flush() -> None:
        nonlocal buf, size
        if not buf:
            return
        name = f"answers.shard{len(parts):02d}.jsonl"
        data = b"".join(buf)
        (out_dir / name).write_bytes(data)
        parts.append(name)
        line_counts.append(len(buf))
        shas.append(hashlib.sha256(data).hexdigest())
        buf = []
        size = 0

    with open(src, "rb") as fh:
        for line in fh:  # binary iteration splits on \n only (U+2028-safe)
            if buf and size + len(line) > _SHARD_MAX_BYTES:
                _flush()
            buf.append(line)
            size += len(line)
    _flush()
    manifest = {
        "parts": parts,
        "line_counts": line_counts,
        "sha256s": shas,
        "total_sha256": _file_sha256(src),
        "total_bytes": src.stat().st_size,
    }
    (out_dir / ANSWERS_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[gen] answers.jsonl over {_TEXT_SPLIT_THRESHOLD} B -> {len(parts)} upload shards")
    return True


def _upload_gen_outputs(slug: str, cell: str, out_dir: Path) -> None:
    """Per-cell incremental upload (one folder commit; #664 per-cell contract)."""
    from huggingface_hub import upload_folder

    from explore_persona_space.orchestrate import hub

    sharded = _split_answers_for_upload(out_dir)
    # Sharded: upload the shards + manifest, not the >9.5 MB original (LFS
    # force-routing, #541). Single: no shard/manifest files exist (cleared).
    ignore = ["answers.jsonl", "*.tmp"] if sharded else ["*.tmp"]
    # Dir-filecount guard (#1190) OUTSIDE the retry wrapper (a guard raise is
    # deterministic; retrying it burns the budget for nothing).
    hub.assert_hub_dir_filecounts(out_dir, _hf_gen_prefix(slug, cell), ignore_patterns=ignore)
    hub.retry_transient(
        lambda: upload_folder(
            repo_id=cm.HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(out_dir),
            path_in_repo=_hf_gen_prefix(slug, cell),
            ignore_patterns=ignore,
            commit_message=f"issue-1336: generation {slug}/{cell}",
        ),
        what=f"gen upload {slug}/{cell}",
    )
    print(f"[gen] uploaded {slug}/{cell} -> {_hf_gen_prefix(slug, cell)}")


def _collect_pending(
    slug: str, corpora: list[str], *, gen_format: str, smoke: bool, upload: bool
) -> list[tuple[str, list[dict], Path]]:
    """Resume predicate per cell: skip complete cells, retry missed uploads.

    done == uploaded (#664 per-cell contract): a cell whose local outputs
    exist but whose Hub prefix is incomplete (a prior run died before/inside
    its upload) re-attempts ONLY the upload — never skips past it, never
    re-generates. Cells are keyed by ``cm.gen_cell_key(corpus, gen_format)``:
    chat resolves to the bare corpus (byte-identical paths + resume state to
    every prior round); naturalistic resolves to its own keyed dir/prefix.
    """
    pending: list[tuple[str, list[dict], Path]] = []
    for corpus in corpora:
        cell = cm.gen_cell_key(corpus, gen_format)
        out_dir = _out_root(smoke) / slug / cell
        if (out_dir / "answers.jsonl").exists() and (out_dir / "audit.json").exists():
            if upload and not smoke and not _hf_gen_state(slug, cell)[0]:
                print(f"[gen] {slug}/{cell}: local outputs exist, Hub incomplete — re-uploading")
                _upload_gen_outputs(slug, cell, out_dir)
            print(f"[gen] skip {slug}/{cell} (local outputs exist)")
            continue
        if not smoke and _try_hf_resume(slug, cell, out_dir):
            continue
        prompts = _read_jsonl(_prompts_root(smoke) / f"{corpus}.jsonl")
        assert prompts, f"no prompts staged for {corpus} — run --prep first"
        pending.append((corpus, prompts, out_dir))
    return pending


def _run_render_integrity(gate_pairs: list[tuple], gen_format: str, slug: str, cell: str) -> dict:
    """Format-conditional render-integrity gate (plan §5 registered control).

    MATCHED-TEXT regime (``gen_format == "chat"`` — chat-generated answers
    re-rendered under BOTH formats): the parent-a4-twin cross-format content
    assertion is load-bearing and HARD-FAILS at the 0.10 threshold, exactly
    as in every prior round (do not weaken). ON-POLICY naturalistic regime
    (``gen_format == "naturalistic"``): the answers were GENERATED under the
    naturalistic prefix, so the cross-format content assertion cannot be
    presumed to hold by construction — the SAME divergence statistic is
    computed and REPORTED as a logged diagnostic (like the cap-hit
    fractions), never raised on. The regime is recorded EXPLICITLY in the
    returned audit record (``regime`` + ``enforced``), never inferred.
    """
    matched_text = gen_format == "chat"
    result = render_integrity_gate(gate_pairs, raise_on_fail=matched_text)  # raises on FAIL (chat)
    result["regime"] = "matched-text" if matched_text else "on-policy-naturalistic"
    result["enforced"] = matched_text
    print(
        f"[gen] render-integrity {result['regime']} "
        f"{'gate' if matched_text else 'diagnostic'} {result['status']} "
        f"{slug}/{cell}: rest-of-span mismatch "
        f"{result['rest_of_span_mismatch_rate']:.4f} "
        f"(first-token diagnostic "
        f"{result['first_token_mismatch_rate_diagnostic']:.4f}, "
        f"{result['n_pairs']} pairs)"
    )
    return result


def run_generation(
    slug: str, corpora: list[str], *, gen_format: str = "chat", smoke: bool, upload: bool
) -> None:
    """Generate + filter + audit every requested corpus with one engine.

    ``gen_format`` selects the generation prompt builder + stop handling
    (round 5): ``chat`` (default) is byte-identical to every prior round —
    Tulu template, chat stop strings, bare-corpus output dirs/HF prefixes;
    ``naturalistic`` generates ON-POLICY under the #825 plain-transcript
    prefix (``cm.natural_prompt`` — render_natural's exact segment text) with
    the newline-anchored naturalistic stop/truncation markers, landing in
    format-keyed dirs (``cm.gen_cell_key``) so the arms never collide.
    """
    hf_id = cm.MODELS[slug]["hf_id"]
    for corpus in corpora:
        fmts = _formats_for(corpus)
        assert gen_format in fmts, (
            f"gen format {gen_format!r} not registered for corpus {corpus!r} (formats: {fmts})"
        )
    pending = _collect_pending(slug, corpora, gen_format=gen_format, smoke=smoke, upload=upload)
    if not pending:
        print(f"[gen] {slug}: nothing to do")
        return

    prompt_builder = {"chat": cm.tulu_prompt, "naturalistic": cm.natural_prompt}[gen_format]
    stop_strings = cm.STOP_STRINGS if gen_format == "chat" else cm.NATURAL_STOP_STRINGS
    truncate_markers = (
        cm.ROLE_HEADER_TRUNCATE if gen_format == "chat" else cm.NATURAL_ROLE_HEADER_TRUNCATE
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if gen_format == "chat":
        # Chat-template parity is a CHAT-format check (pinned Tulu render vs
        # the checkpoint's own template); the naturalistic plain transcript
        # has no tokenizer template counterpart by construction.
        _assert_template_parity(tokenizer, [r["prompt"] for r in pending[0][1][:3]])

    from vllm import LLM, SamplingParams

    llm = LLM(model=hf_id, max_model_len=cm.MAX_MODEL_LEN)
    sampling = SamplingParams(
        n=cm.SAMPLING["n"],
        temperature=cm.SAMPLING["temperature"],
        top_p=cm.SAMPLING["top_p"],
        max_tokens=cm.SAMPLING["max_tokens"],
        seed=cm.SAMPLING["seed"],
        stop=list(stop_strings),
    )
    fmts_needed = {c: _formats_for(c) for c, _, _ in pending}
    for corpus, prompts, out_dir in pending:
        cell = cm.gen_cell_key(corpus, gen_format)
        texts = [prompt_builder(r["prompt"]) for r in prompts]
        gen = _vllm_generate_chunked(llm, texts, sampling)
        rows, kept_ids = [], []
        drop_reasons: collections.Counter = collections.Counter()
        kept_answers: list[str] = []
        kept_tok_lens: list[int] = []
        gate_pairs: list[tuple] = []  # (chat, naturalistic) renders of kept rows
        rep3_flags = 0
        early_eos = 0
        truncated = 0
        for r, (raw, finish) in zip(prompts, gen, strict=True):
            answer = _truncate_role_headers(raw, truncate_markers)
            row = {
                "prompt_idx": r["prompt_idx"],
                "prompt": r["prompt"],
                "response": answer,
                "response_raw_len_chars": len(raw),
                "finish_reason": finish,
            }
            reason = None
            fmt_renders: dict = {}
            if not answer.strip():
                reason = "empty_answer"
            else:
                conv = {"conv_id": str(r["prompt_idx"]), "u1": r["prompt"], "a1": answer}
                for fmt in fmts_needed[corpus]:
                    rendered = RENDERERS[fmt](conv, tokenizer)
                    reason = validate_render(rendered)
                    if reason is not None:
                        reason = f"{fmt}:{reason}"
                        break
                    fmt_renders[fmt] = rendered
                    if fmt == "chat":
                        span = rendered.spans["a1"]
                        kept_tok_lens.append(span[1] - span[0])
            row["kept"] = reason is None
            row["drop_reason"] = reason
            rows.append(row)
            if reason is None:
                kept_ids.append(r["prompt_idx"])
                kept_answers.append(answer)
                rep3_flags += int(_rep3_flag(answer))
                truncated += int(finish == "length")
                early_eos += int(finish == "stop" and len(answer.split()) < 8)
                if "naturalistic" in fmt_renders:
                    gate_pairs.append((fmt_renders["chat"], fmt_renders["naturalistic"]))
            else:
                drop_reasons[reason] += 1
        keep_rate = len(kept_ids) / len(prompts) if prompts else 0.0
        # Rollout text persists FIRST (upload policy: the raw text must be
        # durable before any gate can halt the cell); audit.json is written
        # ONLY after the render-integrity gate below, so a gate-failed cell is
        # never resumable-as-complete (the skip predicate above requires
        # answers.jsonl AND audit.json).
        _write_jsonl(out_dir / "answers.jsonl", rows)
        (out_dir / "allowlist.json").write_text(json.dumps(kept_ids) + "\n")
        # Render-integrity (plan §5 registered control; parent a4 twin):
        # cross-format content-token BPE divergence between the chat and
        # naturalistic renders of the SAME kept answers at the <=0.10
        # first-token-excluded convention. FORMAT-CONDITIONAL (round 5, user
        # decision): hard gate in the matched-text (chat-generation) regime —
        # unchanged from prior rounds — logged diagnostic in the on-policy
        # naturalistic regime; regime recorded in the audit record. See
        # _run_render_integrity.
        render_integrity = None
        if {"chat", "naturalistic"} <= set(fmts_needed[corpus]) and gate_pairs:
            render_integrity = _run_render_integrity(gate_pairs, gen_format, slug, cell)
        audit = {
            "model": slug,
            "hf_id": hf_id,
            "corpus": corpus,
            "gen_format": gen_format,
            "gen_cell": cell,
            "n_prompts": len(prompts),
            "n_kept": len(kept_ids),
            "keep_rate": keep_rate,
            "keep_rate_floor": cm.KEEP_RATE_FLOOR,
            "keep_rate_floor_pass": keep_rate >= cm.KEEP_RATE_FLOOR,
            "drop_reasons": dict(drop_reasons),
            "kept_truncation_rate": (truncated / len(kept_ids)) if kept_ids else None,
            "kept_early_eos_rate": (early_eos / len(kept_ids)) if kept_ids else None,
            "kept_rep3_flag_rate": (rep3_flags / len(kept_ids)) if kept_ids else None,
            "kept_distinct_3gram_rate": _distinct_3gram_rate(kept_answers),
            "kept_answer_tokens": {
                "mean": statistics.fmean(kept_tok_lens) if kept_tok_lens else None,
                "median": statistics.median(kept_tok_lens) if kept_tok_lens else None,
                "p90": (
                    sorted(kept_tok_lens)[int(0.9 * (len(kept_tok_lens) - 1))]
                    if kept_tok_lens
                    else None
                ),
            },
            "sampling": dict(cm.SAMPLING) | {"stop": list(stop_strings)},
            "answers_sha256": hashlib.sha256(
                "\n".join(a for a in kept_answers).encode("utf-8")
            ).hexdigest(),
            "render_integrity": render_integrity,
            "smoke": smoke,
        }
        (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        if not audit["keep_rate_floor_pass"]:
            print(
                f"[gen] WARNING keep-rate floor MISS {slug}/{cell}: "
                f"{keep_rate:.3f} < {cm.KEEP_RATE_FLOOR} — fitting at realized n "
                "(reported, never padded)"
            )
        print(f"[gen] {slug}/{cell}: kept {len(kept_ids)}/{len(prompts)}")
        if upload:
            _upload_gen_outputs(slug, cell, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prep", action="store_true", help="stage corpora (CPU, model-free)")
    ap.add_argument("--model", choices=tuple(cm.MODELS), default=None)
    ap.add_argument("--corpora", default=None, help="comma list (default: registry set)")
    ap.add_argument("--smoke", action="store_true", help="smoke subset roots + N")
    ap.add_argument("--upload", action="store_true", help="per-cell HF upload after gen")
    ap.add_argument(
        "--gen-format",
        choices=("chat", "naturalistic"),
        default="chat",
        help=(
            "generation prompt format (round 5; generation only — prep is "
            "format-blind). 'chat' (default) is byte-identical to prior "
            "rounds; 'naturalistic' generates ON-POLICY under the #825 "
            "plain-transcript prefix into format-keyed output dirs "
            "(<corpus>__gen_naturalistic). Only corpora registering the "
            "naturalistic format qualify (fail-loud otherwise)."
        ),
    )
    ap.add_argument(
        "--corpus-file",
        action="append",
        default=None,
        metavar="SLUG=PATH",
        help="explicit local corpus JSONL override for a v2 corpus (repeatable; prep only)",
    )
    args = ap.parse_args()
    corpora = (
        [c.strip() for c in args.corpora.split(",") if c.strip()]
        if args.corpora
        else list(cm.SMOKE_CORPORA if args.smoke else cm.CORPORA)
    )
    corpus_files: dict[str, str] = {}
    for spec in args.corpus_file or []:
        slug, _, path = spec.partition("=")
        assert slug and path, f"--corpus-file expects SLUG=PATH, got {spec!r}"
        corpus_files[slug] = path
    if args.prep:
        run_prep(corpora, args.smoke, corpus_files=corpus_files or None)
        return
    assert args.model, "--model is required unless --prep"
    run_generation(
        args.model, corpora, gen_format=args.gen_format, smoke=args.smoke, upload=args.upload
    )


if __name__ == "__main__":
    main()
