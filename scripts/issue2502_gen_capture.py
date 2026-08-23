#!/usr/bin/env python
"""issue 2502 unit 2/4 — P1 on-policy generation + P2 activation capture, BOTH models.

Runs the plan's (tasks/<status>/2502/plans/plan.md, v6) P1 (on-policy vLLM
generation, 1 rollout/context) and P2 (HF teacher-forced activation capture)
for Model A (Qwen/Qwen2.5-7B-Instruct, 28 layers x 3584 hidden) and Model B
(Qwen/Qwen3.5-9B, 32 layers x 4096 hidden — RE-verified vs #2378's 9B-A3B
constants), parametrized by ``--model`` + ``--env``.

Port provenance (MF-A): venv/engine/template pins ported from the unmerged
``issue-2378`` branch at 25187c4e9a9f247bab086cedaebfcc582336cb7d
(scripts/issue2378_common.py + issue2378_gen.py + issue2378_capture.py):
``ENGINE_KWARG_PINS`` (gdn_prefill_backend=triton, passed UNGUARDED so a
lacking EngineArgs TypeErrors loudly), ``LAUNCH_ENV_PINS``
(VLLM_USE_FLASHINFER_SAMPLER=0), the empty-``<think>`` chat-template
containment assert, the Qwen3_5ForConditionalGeneration loader (fallback
AutoModelForImageTextToText; explicit ``.to(device)``, never
device_map="auto"), and the bf16-as-uint16 npz codec.

MF-F (per-row capture-position assert, PRODUCTION path): for EVERY captured
row assert (i) the context-segment attention-mask sum minus 1 equals the
selected index (padding-side-aware: fails under left padding / mask holes),
(ii) the materialized prompt-segment ids equal the SEPARATELY re-rendered
context prompt ids, and (iii) the selected token id equals the final token id
of that rendered prompt. Persisted per row: prompt_len, selected_index,
selected_token_id. Single-pass design: cx_last(C) is gathered from the FULL
prompt+answer forward at position n_prompt-1 — causal attention makes it
identical to a context-only forward — and v_x(A) is the answer-span mean.
Teacher-forcing inputs are per-segment TOKEN-ID concatenations (prompt ids +
vLLM completion token ids); concatenated strings are never re-tokenized (BPE
seam gotcha).

Persistence (upload policy): per-chunk (default 500 contexts) upload ->
exact-set verify -> local purge; gen rollout TEXT always persists
(``<raw-prefix>/chunkNNNN.jsonl``); capture tensors persist per chunk under
``<out-prefix>/<chunk>/`` (one upload_folder commit per chunk, ~30 files);
``.capture_done`` sentinel uploads LAST. Resume: StageLedger (regime-keyed on
generating parameters) + ONE scoped HF listing per phase (cross-pod resume).
Cap-hit fraction (finish_reason=="length") is reported in gen_meta.json +
capture_meta.json with the >2% regen trigger named.

Content hygiene: corpus/context text is NEVER printed — logs carry digests,
counts, and chunk keys only.

Phases: ``--phase all`` (default; plan §9 command shape) runs gen then capture
as SEQUENTIAL SUBPROCESSES of this script — vLLM and the HF model are never
co-resident in one process, and the gen child terminates via os._exit(0)
after engine reap + verified durables (#1739/#2149 vLLM finalize-deadlock
class).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# vLLM reads this at import time; set before ANY deferred `import vllm`
# (gotchas.md: fork-poisoned EngineCore silent death, #628).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Script mode puts only the script dir on sys.path; the model venv
# (/root/eps-model-venv) has no editable install of the repo (#823, #2378).
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ISSUE = 2502
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
SEED = 42
TEMPERATURE = 1.0
TOP_P = 0.95
CAP_HIT_THRESHOLD = 0.02  # >2% per-model cap-hit => regen at >=2x cap (plan trigger)
GEN_CHUNK_MAX_BYTES = 9_500_000  # non-LFS text budget per chunk file (upload-policy)

# --- MF-A pins, ported verbatim from issue-2378 @ 25187c4e9a9f (issue2378_common.py) ---
MODEL_VENV_PINS = {"vllm": "0.27.1", "transformers": "5.15.1", "torch": "2.13.0"}
MODEL_VENV_BANNED_DISTS = ("flashinfer-python",)  # py3.11 array.array[int] TypeError (#2378)
LAUNCH_ENV_PINS = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
ENGINE_KWARG_PINS = {"gdn_prefill_backend": "triton"}  # passed UNGUARDED by design (MF-A)
EMPTY_THINK = "<think>\n\n</think>"

MODEL_SPECS = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "key": "A",
        "n_layers": 28,
        "hidden": 3584,
        "min_free_hbm_gb": 40.0,
        "requires_env": "repo-standard",
        "loader": "causal-lm",
    },
    "Qwen/Qwen3.5-9B": {
        # RE-verified for the DENSE 9B (plan §12 live-verified config): 32 layers /
        # 4096 hidden — NOT #2378's Qwen3.5-9B-A3B 64/5120. Fail-loud asserts at load.
        "key": "B",
        "n_layers": 32,
        "hidden": 4096,
        "min_free_hbm_gb": 80.0,
        "requires_env": "pod2378-venv",
        "loader": "qwen3_5",
    },
}


# --------------------------------------------------------------------------
# Small helpers (ported from issue2378_common.py @ 25187c4e9a9f)
# --------------------------------------------------------------------------


def text_digest(text: str) -> str:
    """Content-hygiene digest for logs/ledgers: sha256 prefix + length, never the text."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}|len={len(text)}"


def ids_sha16(ids) -> str:
    """Machine-stable digest of a token-id sequence (ints — safe to hash, gotchas.md)."""
    return hashlib.sha256(",".join(str(int(x)) for x in ids).encode("utf-8")).hexdigest()[:16]


def iter_jsonl(path: Path):
    """Text-mode JSONL iteration (never ``splitlines()`` — U+2028/NEL safety, #950)."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def atomic_write_json(path: Path, obj: object) -> None:
    """Write JSON atomically (tmp in the destination dir + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_metadata(extra: dict | None = None) -> dict:
    """Reproducibility metadata block (git provenance incl. dirty flag, env, ts)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    meta = {
        "issue": ISSUE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "argv": sys.argv,
        **as_metadata_dict(git_provenance()),
    }
    if extra:
        meta.update(extra)
    return meta


def progress(phase: str, k: int, n: int, key: str, t0: float) -> None:
    """Per-unit progress line (code-style checkpoint-per-phase contract)."""
    print(f"[{phase}] unit {k}/{n} {key} elapsed={time.time() - t0:.0f}s", flush=True)


class StageLedger:
    """Resumable per-stage chunk ledger (regime-keyed on GENERATING PARAMETERS).

    Records a REGIME dict (every output-affecting parameter) plus completed
    chunk keys; a resume with a mismatched regime fails loud (#1336: never
    hash recomputed float arrays — parameters only).
    """

    def __init__(self, path: Path, regime: dict):
        self.path = path
        self.regime = regime
        self.done: set[str] = set()
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("regime") != regime:
                raise RuntimeError(
                    f"StageLedger regime mismatch at {path}: on-disk "
                    f"{state.get('regime')} vs requested {regime} — use a fresh out-root"
                )
            self.done = set(state.get("done", []))

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark_done(self, key: str) -> None:
        self.done.add(key)
        atomic_write_json(self.path, {"regime": self.regime, "done": sorted(self.done)})


# --------------------------------------------------------------------------
# bf16-as-uint16 codec (ported from issue2378_capture.py; fp16 overflows Qwen
# massive activations — bf16 bit pattern stored losslessly in uint16)
# --------------------------------------------------------------------------


def encode_bf16(torch_mod, t):
    """bf16 tensor -> uint16 numpy array (lossless bit reinterpretation)."""
    import numpy as np

    assert t.dtype == torch_mod.bfloat16, t.dtype
    return t.contiguous().view(torch_mod.int16).cpu().numpy().view(np.uint16).copy()


def decode_bf16(a, torch_mod):
    """uint16 numpy array -> bf16 tensor (inverse of encode_bf16)."""
    import numpy as np

    assert a.dtype == np.uint16, a.dtype
    arr = np.ascontiguousarray(a).view(np.int16)
    return torch_mod.from_numpy(arr.copy()).view(torch_mod.bfloat16)


def atomic_savez(path: Path, arrays: dict) -> None:
    """Atomic np.savez: tmp named `<stem>.tmp.npz` (np.savez APPENDS .npz, #1092)."""
    import numpy as np

    assert path.name.endswith(".npz"), path
    tmp = path.with_name(path.name[: -len(".npz")] + ".tmp.npz")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# MF-F position assert (pure function — self-testable without a model)
# --------------------------------------------------------------------------


def assert_capture_position(ids_row, mask_row, prompt_ids, *, row_key: str) -> int:
    """MF-F per-row capture-position assert; returns the selected index.

    Asserts, against the MATERIALIZED padded batch tensors: (i) the
    context-segment attention-mask sum minus 1 equals selected_index
    (padding-side-aware — fails under left padding or a mask hole), (ii) the
    prompt segment of the materialized ids equals the separately rendered
    context prompt ids, (iii) the selected token id equals the rendered
    prompt's final token id.
    """
    prompt_l = [int(x) for x in prompt_ids]
    n_prompt = len(prompt_l)
    if n_prompt < 1:
        raise RuntimeError(f"[mf-f] row {row_key}: empty rendered prompt")
    sel = n_prompt - 1
    ids_l = [int(x) for x in ids_row]
    mask_l = [int(x) for x in mask_row]
    ctx_mask_sum = sum(mask_l[:n_prompt])
    if ctx_mask_sum - 1 != sel:
        raise RuntimeError(
            f"[mf-f] row {row_key}: context attention-mask sum-1 ({ctx_mask_sum - 1}) != "
            f"selected_index ({sel}) — left padding or mask hole in the context segment"
        )
    if ids_l[:n_prompt] != prompt_l:
        raise RuntimeError(
            f"[mf-f] row {row_key}: materialized prompt ids != separately rendered "
            f"context prompt ids (first divergence at "
            f"{next(i for i in range(n_prompt) if ids_l[i] != prompt_l[i])})"
        )
    if ids_l[sel] != prompt_l[-1]:
        raise RuntimeError(
            f"[mf-f] row {row_key}: selected token id {ids_l[sel]} != final rendered "
            f"prompt token id {prompt_l[-1]}"
        )
    return sel


@dataclass
class _Rec:
    """One capture row: rendered prompt ids + vLLM completion ids (token-level concat)."""

    context_id: str
    prompt_ids: list[int]
    full_ids: list[int]
    n_prompt: int
    n_gen: int

    @property
    def n_total(self) -> int:
        return len(self.full_ids)


def _pack_batches(recs, *, batch_tokens: int, max_batch_rows: int):
    """Longest-first token-budget packing: len(batch) * max_len <= batch_tokens.

    A single rec longer than the budget is emitted alone (never dropped).
    """
    order = sorted(recs, key=lambda r: (-r.n_total, r.context_id))
    batches: list[list[_Rec]] = []
    cur: list[_Rec] = []
    cur_max = 0
    for r in order:
        if cur and (
            (len(cur) + 1) * max(cur_max, r.n_total) > batch_tokens or len(cur) >= max_batch_rows
        ):
            batches.append(cur)
            cur = []
            cur_max = 0
        cur.append(r)
        cur_max = max(cur_max, r.n_total)
    if cur:
        batches.append(cur)
    return batches


# --------------------------------------------------------------------------
# Chat template + env pins (MF-A)
# --------------------------------------------------------------------------


def assert_chat_template(tok, *, disable_thinking: bool) -> str:
    """Probe-render the chat template; assert the empty-<think> contract (Model B).

    Returns sha256[:16] of the rendered probe (recorded in metas). Ported from
    issue2378_gen.py::_assert_chat_template.
    """
    kwargs = {"enable_thinking": False} if disable_thinking else {}
    probe = tok.apply_chat_template(
        [{"role": "user", "content": "ping"}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    if disable_thinking and EMPTY_THINK not in probe:
        raise RuntimeError(
            "chat template contract violated: enable_thinking=False did not render the "
            f"empty think block {EMPTY_THINK!r} (template drift vs #2378 pin)"
        )
    return hashlib.sha256(probe.encode("utf-8")).hexdigest()[:16]


def render_prompt_ids(tok, text: str, *, disable_thinking: bool) -> list[int]:
    """Render ONE user turn to prompt token ids (add_generation_prompt=True)."""
    kwargs = {"enable_thinking": False} if disable_thinking else {}
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    return [int(x) for x in ids]


def enforce_model_env(args) -> None:
    """MF-A env enforcement: exact venv pins + flashinfer ban + launch env pins.

    pod2378-venv: importlib.metadata version asserts against MODEL_VENV_PINS
    (fail loud on mismatch/absence), banned-dist assert, LAUNCH_ENV_PINS
    setdefault (before any vllm import). repo-standard: no pins.
    """
    if args.env != "pod2378-venv":
        return
    import importlib.metadata as md

    for dist, want in MODEL_VENV_PINS.items():
        try:
            have = md.version(dist)
        except md.PackageNotFoundError as exc:
            raise RuntimeError(f"pod2378-venv pin: distribution {dist!r} not installed") from exc
        if have != want:
            raise RuntimeError(f"pod2378-venv pin mismatch: {dist}=={have}, want {want} (#2378)")
    for dist in MODEL_VENV_BANNED_DISTS:
        present = True
        try:
            md.version(dist)
        except md.PackageNotFoundError:
            present = False
        if present:
            raise RuntimeError(
                f"banned distribution {dist!r} installed — #2378 requires it ABSENT "
                "(py3.11 flashinfer TypeError class)"
            )
    for k, v in LAUNCH_ENV_PINS.items():
        os.environ.setdefault(k, v)
    print(f"[env] pod2378-venv pins OK: {MODEL_VENV_PINS}; env pins {LAUNCH_ENV_PINS}", flush=True)


def gpu_preflight(min_gb: float) -> None:
    """Fail loud unless >= min_gb (decimal GB) of free HBM on the visible device."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this phase (no model loads on the shared VM)")
    free, total = torch.cuda.mem_get_info()
    if free < min_gb * 1e9:
        raise RuntimeError(
            f"free HBM {free / 1e9:.1f} GB < floor {min_gb:.0f} GB (decimal GB; "
            f"total {total / 1e9:.1f} GB) — foreign hold or co-resident phase?"
        )


# --------------------------------------------------------------------------
# Hub helpers (fail-loud upload + exact-set verify; #997/#920 conventions)
# --------------------------------------------------------------------------


def upload_single_file(local: Path, dest: str) -> None:
    """Upload ONE file to the data repo at ``dest`` (full literal path) + verify.

    Per-chunk durability checkpoint (checkpoint-per-phase contract): one small
    file the moment its chunk completes — not a bulk per-file folder walk.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    # UPLOAD_LOOP_EXEMPT: per-chunk durability checkpoint (one file per completed
    # chunk, minutes apart) — the checkpoint-per-phase contract, not a folder walk.
    url = hub._upload(
        local_path=local,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        upload_as_file=True,
        raise_on_error=True,
    )
    if not url:
        raise RuntimeError(f"upload returned no path for {dest} — durability loss, fail loud")
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), HF_DATA_REPO, [dest], path_in_repo=dest, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(f"upload verify failed — missing: {missing}")


def upload_stage_dir(local_dir: Path, prefix_rel: str) -> list[str]:
    """One upload_folder commit for a chunk's staged file set + exact-set verify.

    Ported from issue2378_common.py::upload_stage_dir (never a per-file loop —
    #664 504-storms). Returns the verified repo-relative paths.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"upload_stage_dir: no files under {local_dir}")
    base_url = hub._upload(
        local_path=local_dir,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=prefix_rel,
        raise_on_error=True,
    )
    if not base_url:
        raise RuntimeError(f"upload returned no path for {prefix_rel} — durability loss")
    expected = [f"{prefix_rel}/{p.relative_to(local_dir).as_posix()}" for p in files]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), HF_DATA_REPO, expected, path_in_repo=prefix_rel, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(f"upload verify failed — missing {len(missing)}: {missing[:5]}")
    print(f"[upload] {prefix_rel}: {len(expected)} files verified", flush=True)
    return expected


def hf_missing_of(expected: list[str], scope: str) -> set[str]:
    """ONE scoped listing: which of ``expected`` are NOT on the data repo.

    A fresh (absent) prefix returns all-missing via verify_repo_paths_uploaded's
    EntryNotFoundError branch — resume probes never crash on first runs.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    return set(
        hub.verify_repo_paths_uploaded(
            HfApi(), HF_DATA_REPO, expected, path_in_repo=scope, repo_type="dataset"
        )
    )


def fetch_repo_file(repo_path: str, dest_root: Path, *, what: str) -> Path:
    """Retry-wrapped hf_hub_download of one data-repo file into a mirror root."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    local = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=repo_path,
            repo_type="dataset",
            local_dir=str(dest_root),
        ),
        what=what,
    )
    return Path(local)


# --------------------------------------------------------------------------
# Corpus + chunking
# --------------------------------------------------------------------------


def load_corpus_rows(args, work: Path) -> list[dict]:
    """Fetch corpus.jsonl (u1's output) and return the shard-filtered row list."""
    local = fetch_repo_file(
        f"{args.corpus_prefix}/corpus.jsonl", work / "corpus_dl", what="corpus-download"
    )
    rows = []
    for idx, row in enumerate(iter_jsonl(local)):
        if idx % args.num_shards != args.shard_index:
            continue
        rows.append(row)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(
            f"empty corpus selection (num_shards={args.num_shards} "
            f"shard_index={args.shard_index} limit={args.limit}) — fail loud"
        )
    print(
        f"[corpus] {len(rows)} rows selected (shard {args.shard_index}/{args.num_shards})",
        flush=True,
    )
    return rows


def chunk_key(args, ci: int) -> str:
    """Chunk id; shard-scoped when sharded so parallel shards never collide."""
    if args.num_shards == 1:
        return f"chunk{ci:04d}"
    return f"s{args.shard_index:02d}_chunk{ci:04d}"


def name_suffix(args) -> str:
    """Suffix for shard-scoped meta/sentinel filenames ('' when unsharded)."""
    if args.num_shards == 1:
        return ""
    return f"_s{args.shard_index:02d}of{args.num_shards:02d}"


def gen_regime(args, template_sha: str) -> dict:
    """Gen-phase regime dict (generating parameters only — machine-stable)."""
    return {
        "phase": "gen",
        "issue": ISSUE,
        "model": args.model,
        "env": args.env,
        "seed": args.seed,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "chunk_size": args.chunk_size,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "corpus_prefix": args.corpus_prefix,
        "raw_prefix": args.raw_prefix,
        "disable_thinking": args.disable_thinking,
        "gdn_prefill": args.gdn_prefill or "",
        "limit": args.limit if args.limit is not None else 0,
        "template_sha16": template_sha,
    }


def capture_regime(args, template_sha: str) -> dict:
    """Capture-phase regime dict (adds batching knobs — padded-batch bf16 numerics)."""
    reg = gen_regime(args, template_sha)
    reg.update(
        {
            "phase": "capture",
            "out_prefix": args.out_prefix,
            "batch_tokens": args.batch_tokens,
            "max_batch_rows": args.max_batch_rows,
        }
    )
    return reg


# --------------------------------------------------------------------------
# Phase: gen (P1)
# --------------------------------------------------------------------------


def build_engine(args):
    """Build the vLLM engine via the shared factory; gdn pin passed UNGUARDED (MF-A)."""
    from explore_persona_space.eval.generation import create_vllm_engine

    kwargs = {}
    if args.gdn_prefill:
        assert ENGINE_KWARG_PINS["gdn_prefill_backend"] == args.gdn_prefill
        kwargs.update(ENGINE_KWARG_PINS)
    return create_vllm_engine(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=64,
        seed=args.seed,
        dtype="bfloat16",
        **kwargs,
    )


def count_chunk_stats(path: Path) -> dict:
    """Recount a gen chunk file's stats (cross-pod resume path for gen_meta)."""
    n = cap = gen_tok = 0
    for row in iter_jsonl(path):
        n += 1
        cap += 1 if row.get("cap_hit") else 0
        gen_tok += int(row.get("n_gen_tokens", 0))
    return {"n_rows": n, "n_cap_hit": cap, "n_gen_tokens_sum": gen_tok, "n_len_drops": None}


def write_gen_meta(
    args,
    work: Path,
    keys: list[str],
    stats: dict,
    len_drops: list[dict],
    template_sha: str,
    regime: dict,
) -> None:
    """Assemble + upload gen_meta.json (cap-hit fraction over EVERY chunk)."""
    for key in keys:
        if key in stats:
            continue
        local = fetch_repo_file(
            f"{args.raw_prefix}/{key}.jsonl", work / "recount_dl", what=f"recount({key})"
        )
        stats[key] = count_chunk_stats(local)
        local.unlink()
    n_rows = sum(s["n_rows"] for s in stats.values())
    n_cap = sum(s["n_cap_hit"] for s in stats.values())
    frac = n_cap / max(1, n_rows)
    meta = {
        "model": args.model,
        "n_chunks": len(keys),
        "n_rows": n_rows,
        "n_cap_hit": n_cap,
        "cap_hit_fraction": frac,
        "cap_hit_threshold": CAP_HIT_THRESHOLD,
        "regen_required": frac > CAP_HIT_THRESHOLD,
        "n_length_drops_this_run": len(len_drops),
        "template_sha16": template_sha,
        "regime": regime,
        "per_chunk": stats,
        "meta": run_metadata(),
    }
    p = work / f"gen_meta{name_suffix(args)}.json"
    atomic_write_json(p, meta)
    upload_single_file(p, f"{args.raw_prefix}/gen_meta{name_suffix(args)}.json")
    if len_drops:
        dp = work / f"length_drops{name_suffix(args)}.json"
        atomic_write_json(dp, {"drops": len_drops, "meta": run_metadata()})
        upload_single_file(dp, f"{args.raw_prefix}/length_drops{name_suffix(args)}.json")
    print(
        f"[cap-hit] fraction={frac:.4f} ({n_cap}/{n_rows}) threshold={CAP_HIT_THRESHOLD}",
        flush=True,
    )
    if frac > CAP_HIT_THRESHOLD:
        print(
            "[cap-hit] WARNING: exceeds 2% — plan trigger: re-generate capped rows at "
            f">=2x cap ({2 * args.max_new_tokens})",
            flush=True,
        )


def phase_gen(args, spec) -> None:
    """P1: on-policy generation, 1 rollout/context, per-chunk upload-verify-purge.

    Terminal is os._exit(0) after engine reap + verified durables (vLLM
    generation-driver finalize-deadlock class, #1739/#2149).
    """
    print("[phase=gen] start", flush=True)
    enforce_model_env(args)
    gpu_preflight(min_gpu_gb(args, spec))
    work = work_root(args, spec)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    template_sha = assert_chat_template(tok, disable_thinking=args.disable_thinking)
    print(f"[gen] template sha16={template_sha}", flush=True)
    rows = load_corpus_rows(args, work)
    chunks = [rows[i : i + args.chunk_size] for i in range(0, len(rows), args.chunk_size)]
    keys = [chunk_key(args, ci) for ci in range(len(chunks))]
    regime = gen_regime(args, template_sha)
    ledger = StageLedger(work / f"gen_ledger{name_suffix(args)}.json", regime)
    stats_path = work / f"gen_stats{name_suffix(args)}.json"
    stats: dict = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    dests = {k: f"{args.raw_prefix}/{k}.jsonl" for k in keys}
    hf_missing = hf_missing_of(list(dests.values()), scope=args.raw_prefix)
    budget = args.max_model_len - args.max_new_tokens
    eos_id = tok.eos_token_id
    llm = None
    sp = None
    total_len_drops: list[dict] = []
    t0 = time.time()
    for ci, chunk_rows in enumerate(chunks):
        key = keys[ci]
        if ledger.is_done(key):
            continue
        if dests[key] not in hf_missing:
            print(f"[gen] {key}: present on HF — resume-skip", flush=True)
            ledger.mark_done(key)
            continue
        rendered = []
        len_drops = []
        for row in chunk_rows:
            pids = render_prompt_ids(tok, row["text"], disable_thinking=args.disable_thinking)
            if len(pids) > budget:
                len_drops.append(
                    {
                        "context_id": row["context_id"],
                        "context_sha": row["context_sha"],
                        "n_prompt_tokens": len(pids),
                    }
                )
                continue
            rendered.append((row, pids))
        if len(len_drops) > 0.05 * len(chunk_rows):
            raise RuntimeError(
                f"[gen] {key}: {len(len_drops)}/{len(chunk_rows)} rows over token budget "
                f"{budget} — systematic template/tokenizer mismatch, fail loud"
            )
        if not rendered:
            raise RuntimeError(f"[gen] {key}: empty selection after length filter — fail loud")
        if llm is None:
            llm = build_engine(args)
            from vllm import SamplingParams

            sp = SamplingParams(
                n=1,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=args.max_new_tokens,
                seed=args.seed,
            )
        outs = llm.generate([{"prompt_token_ids": p} for _, p in rendered], sp, use_tqdm=False)
        if len(outs) != len(rendered):
            raise RuntimeError(f"[gen] {key}: vLLM returned {len(outs)} != {len(rendered)}")
        local = work / f"{key}.jsonl"
        n_cap = 0
        gen_tok_sum = 0
        with local.open("w", encoding="utf-8") as fh:
            for (row, pids), out in zip(rendered, outs):
                o = out.outputs[0]
                comp_ids = [int(x) for x in o.token_ids]
                stripped = False
                while comp_ids and eos_id is not None and comp_ids[-1] == eos_id:
                    comp_ids.pop()
                    stripped = True
                cap_hit = o.finish_reason == "length"
                n_cap += 1 if cap_hit else 0
                gen_tok_sum += len(comp_ids)
                rec = {
                    "context_id": row["context_id"],
                    "context_sha": row["context_sha"],
                    "source_tag": row.get("source_tag"),
                    "dataset_id": row.get("dataset_id"),
                    "config": row.get("config"),
                    "regime_class": row.get("regime_class"),
                    "realism_tier": row.get("realism_tier"),
                    "split": row.get("split"),
                    "lodo_group": row.get("lodo_group"),
                    "prompt_sha": ids_sha16(pids),
                    "n_prompt_tokens": len(pids),
                    "completion": o.text,
                    "completion_token_ids": comp_ids,
                    "n_gen_tokens": len(comp_ids),
                    "finish_reason": o.finish_reason,
                    "cap_hit": cap_hit,
                    "eos_stripped": stripped,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        sz = local.stat().st_size
        if sz > GEN_CHUNK_MAX_BYTES:
            raise RuntimeError(
                f"[gen] {key}.jsonl is {sz} bytes > {GEN_CHUNK_MAX_BYTES} non-LFS budget — "
                "lower --chunk-size"
            )
        upload_single_file(local, dests[key])
        local.unlink()
        stats[key] = {
            "n_rows": len(rendered),
            "n_cap_hit": n_cap,
            "n_gen_tokens_sum": gen_tok_sum,
            "n_len_drops": len(len_drops),
        }
        total_len_drops.extend(len_drops)
        atomic_write_json(stats_path, stats)
        ledger.mark_done(key)
        progress("gen", ci + 1, len(chunks), key, t0)
    if llm is not None:
        from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

        _reap_vllm_engine(llm)
        llm = None
    write_gen_meta(args, work, keys, stats, total_len_drops, template_sha, regime)
    print("[phase=gen] done", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# --------------------------------------------------------------------------
# Phase: capture (P2)
# --------------------------------------------------------------------------


@dataclass
class ModelCtx:
    """Loaded HF model context for the capture phase."""

    torch: object
    model: object
    device: str
    n_layers: int
    hidden: int
    pad_id: int
    lk: dict


def load_model_ctx(args, spec, tok) -> ModelCtx:
    """Load the HF model with fail-loud constant asserts (#2378 loader port).

    Qwen3.5: Qwen3_5ForConditionalGeneration (fallback AutoModelForImageTextToText),
    text-only use, explicit .to(device) — never device_map="auto". Constants
    (num_hidden_layers / hidden_size, text_config-aware) are RE-verified for the
    dense 9B (32/4096) and Qwen2.5-7B-Instruct (28/3584).
    """
    import inspect

    import torch

    gpu_preflight(min_gpu_gb(args, spec))
    if spec["loader"] == "qwen3_5":
        try:
            from transformers import Qwen3_5ForConditionalGeneration as _ModelCls
        except ImportError:
            from transformers import AutoModelForImageTextToText as _ModelCls
    else:
        from transformers import AutoModelForCausalLM as _ModelCls
    print(f"[capture] loading {args.model} via {_ModelCls.__name__}", flush=True)
    model = _ModelCls.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to("cuda")
    model.eval()
    tcfg = getattr(model.config, "text_config", model.config)
    if int(tcfg.num_hidden_layers) != spec["n_layers"]:
        raise RuntimeError(
            f"num_hidden_layers {tcfg.num_hidden_layers} != expected {spec['n_layers']} "
            f"for {args.model} — constants re-verify failed"
        )
    if int(tcfg.hidden_size) != spec["hidden"]:
        raise RuntimeError(
            f"hidden_size {tcfg.hidden_size} != expected {spec['hidden']} for {args.model}"
        )
    # Introspection-guarded: EXPLICIT param only — bare **kwargs does NOT count (#779).
    params = inspect.signature(model.forward).parameters
    lk = {"logits_to_keep": 1} if "logits_to_keep" in params else {}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")
    return ModelCtx(
        torch=torch,
        model=model,
        device="cuda",
        n_layers=spec["n_layers"],
        hidden=spec["hidden"],
        pad_id=int(pad_id),
        lk=lk,
    )


def capture_expected(args, key: str, n_layers: int) -> list[str]:
    """Expected repo paths for one capture chunk (per-layer npz + rows.json)."""
    base = f"{args.out_prefix}/{key}"
    paths = [f"{base}/{key}__L{k:02d}.npz" for k in range(1, n_layers + 1)]
    paths.append(f"{base}/{key}__rows.json")
    return paths


def forward_batch(mctx: ModelCtx, batch: list[_Rec]) -> dict:
    """One padded (RIGHT-padding) teacher-forced forward; returns per-layer arrays.

    Per row (MF-F, production): assert_capture_position on the materialized
    tensors; gather cx_last at selected_index and vx as the answer-span mean
    (float32 accumulate -> bf16). Returns {k: (cx uint16 (b,H), vx uint16 (b,H))}
    for hidden_states index k in 1..n_layers (hs[k] = output of decoder block k-1).
    """
    torch = mctx.torch
    bsz = len(batch)
    t_max = max(r.n_total for r in batch)
    ids = torch.full((bsz, t_max), mctx.pad_id, dtype=torch.long)
    mask = torch.zeros((bsz, t_max), dtype=torch.long)
    for i, r in enumerate(batch):
        ids[i, : r.n_total] = torch.tensor(r.full_ids, dtype=torch.long)
        mask[i, : r.n_total] = 1
    # MF-F per-row asserts on the MATERIALIZED batch tensors (CPU copies).
    for i, r in enumerate(batch):
        assert_capture_position(
            ids[i].tolist(), mask[i].tolist(), r.prompt_ids, row_key=r.context_id
        )
    ids_dev = ids.to(mctx.device)
    mask_dev = mask.to(mctx.device)
    with torch.no_grad():
        out = mctx.model(
            input_ids=ids_dev,
            attention_mask=mask_dev,
            output_hidden_states=True,
            use_cache=False,
            **mctx.lk,
        )
    hs = out.hidden_states
    if len(hs) != mctx.n_layers + 1:
        raise RuntimeError(f"hidden_states len {len(hs)} != n_layers+1 ({mctx.n_layers + 1})")
    assert hs[1].shape == (bsz, t_max, mctx.hidden), hs[1].shape
    sel = torch.tensor([r.n_prompt - 1 for r in batch], device=mctx.device)
    lo = torch.tensor([r.n_prompt for r in batch], device=mctx.device)
    hi = torch.tensor([r.n_total for r in batch], device=mctx.device)
    t_idx = torch.arange(t_max, device=mctx.device)
    span = (t_idx[None, :] >= lo[:, None]) & (t_idx[None, :] < hi[:, None])
    denom = span.sum(1)
    if not bool((denom > 0).all()):
        raise RuntimeError("empty answer span reached forward_batch (upstream drop failed)")
    rowsel = torch.arange(bsz, device=mctx.device)
    per_layer = {}
    for k in range(1, mctx.n_layers + 1):
        h = hs[k]
        cxk = h[rowsel, sel]
        vxk = ((h.float() * span[..., None]).sum(1) / denom[:, None].float()).to(torch.bfloat16)
        per_layer[k] = (encode_bf16(torch, cxk), encode_bf16(torch, vxk))
    del out, hs
    return per_layer


def capture_chunk(
    mctx: ModelCtx,
    tok,
    args,
    spec,
    key: str,
    gen_rows: list[dict],
    ctx_lookup: dict,
    stage_dir: Path,
    template_sha: str,
) -> dict:
    """Capture one chunk: re-render + MF-F sha check, batch forwards, stage files."""
    import numpy as np

    recs: list[_Rec] = []
    gen_by_id: dict[str, dict] = {}
    drops = {"empty_completion": 0}
    for row in gen_rows:
        cid = row["context_id"]
        text = ctx_lookup.get(cid)
        if text is None:
            raise RuntimeError(
                f"context_id {cid} in gen chunk {key} but absent from corpus shard selection"
            )
        prompt_ids = render_prompt_ids(tok, text, disable_thinking=args.disable_thinking)
        psha = ids_sha16(prompt_ids)
        if psha != row["prompt_sha"]:
            raise RuntimeError(
                f"[mf-f] prompt re-render sha mismatch for {cid}: {psha} != "
                f"{row['prompt_sha']} — tokenizer/template drift between gen and capture"
            )
        comp_ids = [int(x) for x in row["completion_token_ids"]]
        if not comp_ids:
            drops["empty_completion"] += 1
            continue
        full = prompt_ids + comp_ids
        if len(full) > args.max_model_len:
            raise RuntimeError(f"row {cid}: full sequence {len(full)} > {args.max_model_len}")
        recs.append(_Rec(cid, prompt_ids, full, len(prompt_ids), len(comp_ids)))
        gen_by_id[cid] = row
    if not recs:
        raise RuntimeError(f"chunk {key}: empty capture selection after drops {drops}")
    n = len(recs)
    hdim = spec["hidden"]
    n_layers = spec["n_layers"]
    cx = {k: np.zeros((n, hdim), dtype=np.uint16) for k in range(1, n_layers + 1)}
    vx = {k: np.zeros((n, hdim), dtype=np.uint16) for k in range(1, n_layers + 1)}
    idx_of = {r.context_id: i for i, r in enumerate(recs)}
    batches = _pack_batches(
        recs, batch_tokens=args.batch_tokens, max_batch_rows=args.max_batch_rows
    )
    for b in batches:
        per_layer = forward_batch(mctx, b)
        for k in range(1, n_layers + 1):
            ck, vk = per_layer[k]
            for j, r in enumerate(b):
                i = idx_of[r.context_id]
                cx[k][i] = ck[j]
                vx[k][i] = vk[j]
    rows_meta = []
    for i, r in enumerate(recs):
        g = gen_by_id[r.context_id]
        rows_meta.append(
            {
                "row": i,
                "context_id": r.context_id,
                "context_sha": g["context_sha"],
                "prompt_len": r.n_prompt,
                "selected_index": r.n_prompt - 1,
                "selected_token_id": r.prompt_ids[-1],
                "n_gen_tokens": r.n_gen,
                "span": [r.n_prompt, r.n_total],
                "split": g.get("split"),
                "regime_class": g.get("regime_class"),
                "lodo_group": g.get("lodo_group"),
                "source_tag": g.get("source_tag"),
            }
        )
    layer_note = "L{k} = hidden_states[k] = output of decoder block k-1 (0-indexed)"
    npz_meta = {
        "issue": ISSUE,
        "model": args.model,
        "encoding": "bf16_as_uint16",
        "arrays": ["cx_last", "vx"],
        "layer_map": layer_note,
        "n_rows": n,
        "hidden": hdim,
        "row_order": "matches rows.json 'row' field",
        "template_sha16": template_sha,
    }
    for k in range(1, n_layers + 1):
        meta_k = dict(npz_meta, hs_index=k, decoder_block_0indexed=k - 1)
        atomic_savez(
            stage_dir / f"{key}__L{k:02d}.npz",
            {"cx_last": cx[k], "vx": vx[k], "meta": np.array(json.dumps(meta_k))},
        )
    gen_stats = {
        "n_gen_rows": len(gen_rows),
        "n_cap_hit": sum(1 for g in gen_rows if g.get("cap_hit")),
    }
    rows_doc = {
        "key": key,
        "n_rows": n,
        "gen_stats": gen_stats,
        "drops": drops,
        "layer_map": layer_note,
        "template_sha16": template_sha,
        "rows": rows_meta,
        "meta": run_metadata(),
    }
    atomic_write_json(stage_dir / f"{key}__rows.json", rows_doc)
    return {
        "n_rows": n,
        "n_gen_rows": gen_stats["n_gen_rows"],
        "n_cap_hit": gen_stats["n_cap_hit"],
        "drops": drops,
    }


def phase_capture(args, spec) -> None:
    """P2: teacher-forced capture (cx_last + vx, all post-block layers), per-chunk."""
    print("[phase=capture] start", flush=True)
    enforce_model_env(args)
    work = work_root(args, spec)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    template_sha = assert_chat_template(tok, disable_thinking=args.disable_thinking)
    rows = load_corpus_rows(args, work)
    ctx_lookup = {r["context_id"]: r["text"] for r in rows}
    n_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
    keys = [chunk_key(args, ci) for ci in range(n_chunks)]
    regime = capture_regime(args, template_sha)
    ledger = StageLedger(work / f"capture_ledger{name_suffix(args)}.json", regime)
    exp_by_key = {k: capture_expected(args, k, spec["n_layers"]) for k in keys}
    all_expected = [p for ps in exp_by_key.values() for p in ps]
    missing = hf_missing_of(all_expected, scope=args.out_prefix)
    complete_on_hf = {k for k, ps in exp_by_key.items() if not (set(ps) & missing)}
    pending = [k for k in keys if not (ledger.is_done(k) or k in complete_on_hf)]
    print(
        f"[capture] {len(keys)} chunks: {len(complete_on_hf)} complete on HF, "
        f"{len(pending)} pending",
        flush=True,
    )
    summaries: dict[str, dict] = {}
    if pending:
        mctx = load_model_ctx(args, spec, tok)
        t0 = time.time()
        for ci, key in enumerate(keys):
            if ledger.is_done(key) or key in complete_on_hf:
                continue
            gen_local = fetch_repo_file(
                f"{args.raw_prefix}/{key}.jsonl", work / "gen_dl", what=f"gen-chunk({key})"
            )
            gen_rows = list(iter_jsonl(gen_local))
            if not gen_rows:
                raise RuntimeError(f"gen chunk {key} is empty — fail loud")
            stage_dir = work / f"stage_{key}"
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            stage_dir.mkdir(parents=True)
            summaries[key] = capture_chunk(
                mctx, tok, args, spec, key, gen_rows, ctx_lookup, stage_dir, template_sha
            )
            upload_stage_dir(stage_dir, f"{args.out_prefix}/{key}")
            shutil.rmtree(stage_dir)
            gen_local.unlink()
            ledger.mark_done(key)
            progress("capture", ci + 1, len(keys), key, t0)
    # Aggregate: fetch rows.json for chunks not computed this run (small files).
    for key in keys:
        if key in summaries:
            continue
        local = fetch_repo_file(
            f"{args.out_prefix}/{key}/{key}__rows.json", work / "rows_dl", what=f"rows({key})"
        )
        d = json.loads(local.read_text(encoding="utf-8"))
        summaries[key] = {
            "n_rows": d["n_rows"],
            "n_gen_rows": d["gen_stats"]["n_gen_rows"],
            "n_cap_hit": d["gen_stats"]["n_cap_hit"],
            "drops": d.get("drops", {}),
        }
        local.unlink()
        ledger.mark_done(key)
    n_rows = sum(s["n_rows"] for s in summaries.values())
    n_gen_rows = sum(s["n_gen_rows"] for s in summaries.values())
    n_cap = sum(s["n_cap_hit"] for s in summaries.values())
    frac = n_cap / max(1, n_gen_rows)
    totals = {
        "n_chunks": len(keys),
        "n_rows_captured": n_rows,
        "n_gen_rows": n_gen_rows,
        "n_cap_hit": n_cap,
        "cap_hit_fraction": frac,
        "cap_hit_threshold": CAP_HIT_THRESHOLD,
        "n_empty_completion_drops": sum(
            s["drops"].get("empty_completion", 0) for s in summaries.values()
        ),
    }
    meta = {
        "model": args.model,
        "totals": totals,
        "regime": regime,
        "per_chunk": summaries,
        "meta": run_metadata(),
    }
    mp = work / f"capture_meta{name_suffix(args)}.json"
    atomic_write_json(mp, meta)
    upload_single_file(mp, f"{args.out_prefix}/capture_meta{name_suffix(args)}.json")
    # Sentinel uploads LAST (phase-done contract for downstream fits units).
    sentinel_name = ".capture_done" if args.num_shards == 1 else f".capture_done{name_suffix(args)}"
    sp = work / f"capture_done{name_suffix(args)}.json"
    atomic_write_json(sp, {"done": True, "totals": totals, "meta": run_metadata()})
    upload_single_file(sp, f"{args.out_prefix}/{sentinel_name}")
    print(f"[capture] totals: {json.dumps(totals)}", flush=True)
    print("[phase=capture] done", flush=True)


# --------------------------------------------------------------------------
# Phase: all (sequencer — gen and capture in SEPARATE subprocesses)
# --------------------------------------------------------------------------


def phase_all(args) -> int:
    """Run gen then capture as sequential subprocesses (never co-resident)."""
    argv = sys.argv[1:]
    cleaned = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--phase":
            skip = True
            continue
        if a.startswith("--phase="):
            continue
        cleaned.append(a)
    for ph in ("gen", "capture"):
        cmd = [sys.executable, os.path.abspath(__file__), *cleaned, "--phase", ph]
        print(f"[phase=all] launching subprocess --phase {ph}", flush=True)
        proc = subprocess.run(cmd, env={**os.environ}, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"--phase {ph} subprocess exited rc={proc.returncode}")
        print(f"[phase=all] --phase {ph} rc=0", flush=True)
    print("[phase=done] gen+capture complete", flush=True)
    return 0


# --------------------------------------------------------------------------
# Self-test (VM-safe: no model load, no network, no GPU)
# --------------------------------------------------------------------------


def run_self_test() -> int:
    """Synthetic unit checks: bf16 codec, MF-F asserts, packing, ledger regime."""
    import tempfile

    import torch

    n_pass = 0
    # 1. bf16 codec round-trip incl. an fp16-overflow magnitude (Qwen massive acts).
    t = torch.randn(4, 8, dtype=torch.bfloat16)
    t[0, 0] = torch.tensor(1e30, dtype=torch.bfloat16)
    rt = decode_bf16(encode_bf16(torch, t), torch)
    assert torch.equal(rt, t), "bf16 codec round-trip mismatch"
    n_pass += 1
    # 2. MF-F position assert: right-padded pass case.
    prompt = [11, 12, 13]
    ids = [11, 12, 13, 21, 22, 0]
    mask = [1, 1, 1, 1, 1, 0]
    assert assert_capture_position(ids, mask, prompt, row_key="t-pass") == 2
    n_pass += 1
    # 3. LEFT padding raises (the MF-F guarded class).
    try:
        assert_capture_position(
            [0, 11, 12, 13, 21, 22], [0, 1, 1, 1, 1, 1], prompt, row_key="t-leftpad"
        )
        raise AssertionError("left-pad not caught")
    except RuntimeError:
        n_pass += 1
    # 4. Mask hole inside the context segment raises (first conjunct).
    try:
        assert_capture_position(ids, [1, 0, 1, 1, 1, 0], prompt, row_key="t-hole")
        raise AssertionError("mask hole not caught")
    except RuntimeError:
        n_pass += 1
    # 5. Prompt-segment id mismatch raises (off-by-one / wrong-token class).
    try:
        assert_capture_position([11, 12, 99, 21, 22, 0], mask, prompt, row_key="t-tok")
        raise AssertionError("wrong token not caught")
    except RuntimeError:
        n_pass += 1
    # 6. _pack_batches budget + row-cap + coverage properties.
    recs = [
        _Rec(context_id=f"c{i}", prompt_ids=[1], full_ids=[1] * ln, n_prompt=1, n_gen=ln - 1)
        for i, ln in enumerate([10, 9, 8, 1, 1])
    ]
    batches = _pack_batches(recs, batch_tokens=20, max_batch_rows=3)
    seen = set()
    for b in batches:
        mx = max(r.n_total for r in b)
        assert len(b) * mx <= 20 and len(b) <= 3, (len(b), mx)
        seen.update(r.context_id for r in b)
    assert seen == {r.context_id for r in recs}
    n_pass += 1
    # 7. StageLedger: resume works; regime mismatch fails loud.
    with tempfile.TemporaryDirectory() as td:
        lp = Path(td) / "ledger.json"
        led = StageLedger(lp, {"a": 1})
        led.mark_done("k0")
        assert StageLedger(lp, {"a": 1}).is_done("k0")
        try:
            StageLedger(lp, {"a": 2})
            raise AssertionError("regime mismatch not caught")
        except RuntimeError:
            pass
    n_pass += 1
    print(f"[self-test] PASS ({n_pass} checks)", flush=True)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def min_gpu_gb(args, spec) -> float:
    """Resolved free-HBM floor (decimal GB): CLI override or per-model default."""
    if args.min_gpu_mem_gb is not None:
        return float(args.min_gpu_mem_gb)
    return float(spec["min_free_hbm_gb"])


def work_root(args, spec) -> Path:
    """Per-model local work dir (ledgers, staging, downloads)."""
    p = Path(args.work_dir) / f"model{spec['key']}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_model_flags(args, spec) -> None:
    """Fail loud on any model/env/flag combination outside the plan's contract."""
    if args.env != spec["requires_env"]:
        raise SystemExit(
            f"--env {args.env} invalid for {args.model}; requires {spec['requires_env']} (MF-A)"
        )
    is_b = spec["key"] == "B"
    if is_b and not args.disable_thinking:
        raise SystemExit("Qwen3.5-9B requires --disable-thinking (empty-<think> contract, MF-A)")
    if not is_b and args.disable_thinking:
        raise SystemExit("--disable-thinking is Qwen3.5-only (Qwen2.5 template contract)")
    if is_b and args.gdn_prefill != "triton":
        raise SystemExit("Qwen3.5-9B requires --gdn-prefill triton (#2378 GDN pin, MF-A)")
    if not is_b and args.gdn_prefill:
        raise SystemExit("--gdn-prefill is Qwen3.5-only (GDN linear-attention pin)")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index {args.shard_index} out of range for {args.num_shards}")
    if args.max_new_tokens * 2 > args.max_model_len:
        raise SystemExit("--max-new-tokens too large for --max-model-len (need >=2x headroom)")


def build_parser() -> argparse.ArgumentParser:
    """CLI per plan §9 (issue 2502)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("all", "gen", "capture"), default="all")
    ap.add_argument("--model", required=True, help="Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen3.5-9B")
    ap.add_argument("--env", required=True, choices=("repo-standard", "pod2378-venv"))
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Qwen3.5 only: enable_thinking=False (empty-<think> contract)",
    )
    ap.add_argument(
        "--gdn-prefill",
        choices=("triton",),
        default=None,
        help="Qwen3.5 only: gdn_prefill_backend engine pin (#2378)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="sampling seed (default 42 = primary draw; MF-E replicates pass 43/44/45)",
    )
    ap.add_argument("--corpus-prefix", default="issue2502_ctxmap_xgen/context_corpus")
    ap.add_argument(
        "--raw-prefix",
        required=True,
        help="HF prefix for gen rollout text (raw_completions/final/model{A,B})",
    )
    ap.add_argument(
        "--out-prefix",
        required=True,
        help="HF prefix for capture tensors (analysis_tensors/model{A,B})",
    )
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument(
        "--batch-tokens", type=int, default=16384, help="capture: padded-token budget per forward"
    )
    ap.add_argument("--max-batch-rows", type=int, default=64)
    ap.add_argument(
        "--min-gpu-mem-gb",
        type=float,
        default=None,
        help="free-HBM floor in DECIMAL GB (default: 40 model A / 80 model B)",
    )
    ap.add_argument("--work-dir", default="/workspace/issue2502_gen_capture")
    ap.add_argument(
        "--limit", type=int, default=None, help="cap corpus rows AFTER shard filter (smoke slices)"
    )
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="VM-safe synthetic checks (codec, MF-F asserts, packing, ledger)",
    )
    return ap


def main() -> int:
    """Entry point: import-check / self-test / phase dispatch."""
    args = build_parser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("issue2502_gen_capture: import-check OK", flush=True)
        return 0
    # load_dotenv BEFORE any numpy/torch import (thread caps freeze at import, #847).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    if args.self_test:
        return run_self_test()
    spec = MODEL_SPECS.get(args.model)
    if spec is None:
        raise SystemExit(f"unknown --model {args.model!r}; known: {sorted(MODEL_SPECS)}")
    validate_model_flags(args, spec)
    if args.phase == "all":
        return phase_all(args)
    if args.phase == "gen":
        phase_gen(args, spec)  # terminal: os._exit(0)
        return 0  # unreachable
    phase_capture(args, spec)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
