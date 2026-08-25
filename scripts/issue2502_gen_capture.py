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
generating parameters, same-workdir) + ONE scoped HF listing per phase, gated
by a remote ``regime.json`` digest under EACH phase prefix (shard_index
excluded; num_shards AND the full-corpus content sha16 are IN the digest —
r2 SB-1(i)/(ii): a corpus rebuild or a reshard at the same prefix must
mismatch loudly, never presence-skip stale/colliding chunks) — a
presence-skip is honored ONLY after the remote digest matches this
invocation's regime, and the digest is FIRST-published only onto a prefix a
scoped listing proves EMPTY (r2 SB-1(iii): a populated digest-less prefix is
pre-regime/foreign residue, refused; r1 review g2 Major 1).
Cap-hit fraction (finish_reason=="length") is reported in gen_meta.json +
capture_meta.json; >2% is a DESIGNED HALT: a durable cap_hit_report json
(affected context ids per chunk) uploads under the raw prefix and the process
exits rc=EXIT_CAP_HIT (7) BEFORE capture — the plan's >=2x-cap regen runs
against FRESH prefixes (the regime digest forbids in-place mixing). Think-leak
policy (#2378 classifier port, g2 Major 2): a completion containing
``<think>`` is FLAGGED at gen (``think_leak`` on the persisted row — rollout
text always persists) and DROPPED at capture (excluded from tensors +
rows.json); counts ride gen_meta + capture_meta. Secret-drop policy (#2502
crash fix): a gen row whose JSONL line carries a real-secret-grade string
(``secret_scrub.scan_bytes`` — the SAME detector the fail-closed upload gate
runs) is DROPPED WHOLE at gen BEFORE upload — completion text AND
``completion_token_ids``, because the ids decode back to the secret and a
text-only scrub would leak through the int array; a chunk dropping ALL rows
fails loud. Counts ride gen_meta (``n_secret_redacted``) + capture_meta
totals (``n_secret_redacted_drops``); logs are COUNT-ONLY (never row text).

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
EXIT_CAP_HIT = 7  # designed cap-hit halt rc (report artifact + distinct rc; rc=7 precedent)
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
    """Render ONE user turn to prompt token ids (add_generation_prompt=True).

    Under the Model-B empty-<think> contract the EMPTY_THINK literal is
    asserted on EVERY render (plan §4 P1 "on every render" — r1 review g2
    Minor 4), not only the phase-start probe. Content hygiene: the error
    carries a digest, never the context text.
    """
    kwargs = {"enable_thinking": False} if disable_thinking else {}
    msgs = [{"role": "user", "content": text}]
    if disable_thinking:
        rendered = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kwargs
        )
        if EMPTY_THINK not in rendered:
            raise RuntimeError(
                "chat template contract violated on render (context "
                f"{text_digest(text)}): empty think block {EMPTY_THINK!r} absent"
            )
    # transformers 5.x flips apply_chat_template(tokenize=True) to default
    # return_dict=True (a BatchEncoding dict — the listcomp would int() its
    # KEYS); 4.x defaults return_dict=False (flat id list). Pass it
    # explicitly so BOTH lanes (Model A repo-standard 4.57.6, Model B
    # pod2378-venv 5.15.1) get the flat id list (#2502; gotcha in
    # .claude/rules/gotchas.md, cites #2378).
    ids = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs
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
    # AUTHORITATIVE assignment (issue2378_common.py:80-88: "the pin always wins") —
    # a setdefault would let an inherited VLLM_USE_FLASHINFER_SAMPLER=1 defeat the
    # pin and reproduce the exact #2378 crash class (r1 review g2 Major 3).
    for k, v in LAUNCH_ENV_PINS.items():
        prev = os.environ.get(k)
        if prev is not None and prev != v:
            print(f"[env] OVERRIDING inherited {k}={prev!r} with pin {v!r} (#2378)", flush=True)
        os.environ[k] = v
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


def hf_prefix_files(prefix: str) -> list[str]:
    """ONE scoped listing of every file under ``prefix`` on the data repo.

    Absent prefix -> [] (the fresh-first-run case); the underlying
    ``list_repo_entries_complete`` walk is already transient-retried. Used by
    the SB-1(iii) first-publication seal in ``ensure_remote_regime``: before
    ``regime.json`` is first-published, the prefix must be PROVEN empty.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    return hub.list_hf_files_under_path(HfApi(), HF_DATA_REPO, prefix, repo_type="dataset")


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


def corpus_content_sha16(id_sha_pairs) -> str:
    """Order/selection-invariant corpus CONTENT fingerprint (r2 SB-1(i)).

    sha256[:16] over the sorted (context_id, context_sha) pairs of the FULL
    corpus file — identical across shards and ``--limit`` slices, so every
    shard of one run shares the remote regime digest; any corpus rebuild that
    changes the row set (or any row's text, via context_sha) flips it,
    forcing a loud regime mismatch instead of a silent presence-skip over
    stale chunks. Inputs are file-read strings, never recomputed floats
    (machine-stable key; gotchas.md float-recompute rule).
    """
    joined = "\n".join(f"{cid}:{csha}" for cid, csha in sorted(id_sha_pairs))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def load_corpus_rows(args, work: Path) -> tuple[list[dict], str]:
    """Fetch corpus.jsonl (u1's output); return (shard-filtered rows, content sha16).

    The content fingerprint covers EVERY row of the downloaded file (sorted
    context_id:context_sha pairs) BEFORE shard/limit selection — all shards
    of one run share it, and it enters the shared remote regime digest
    (r2 SB-1(i): a corpus rebuild at the same prefix must flip the regime).
    Validates the required fields of EVERY selected row here — BEFORE any
    engine/model init (r1 review Codex finding; validate-before-init) — and
    refuses duplicate context_ids (the capture tensor index is id-keyed).
    Content hygiene: errors name field + index, never row text.
    """
    local = fetch_repo_file(
        f"{args.corpus_prefix}/corpus.jsonl", work / "corpus_dl", what="corpus-download"
    )
    rows = []
    fp_pairs: list[tuple[str, str]] = []
    for idx, row in enumerate(iter_jsonl(local)):
        cid = row.get("context_id")
        csha = row.get("context_sha")
        if not (isinstance(cid, str) and cid and isinstance(csha, str) and csha):
            raise RuntimeError(
                f"[corpus] row {idx}: context_id/context_sha missing or empty — "
                "cannot fingerprint corpus content (u1 corpus contract violation)"
            )
        fp_pairs.append((cid, csha))
        if idx % args.num_shards != args.shard_index:
            continue
        rows.append(row)
    corpus_sha = corpus_content_sha16(fp_pairs)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(
            f"empty corpus selection (num_shards={args.num_shards} "
            f"shard_index={args.shard_index} limit={args.limit}) — fail loud"
        )
    seen_ids: set[str] = set()
    for i, row in enumerate(rows):
        for f in ("context_id", "context_sha", "text"):
            v = row.get(f)
            if not isinstance(v, str) or not v:
                raise RuntimeError(
                    f"[corpus] selected row {i}: field {f!r} missing/empty — "
                    "u1 corpus contract violation (validate-before-init)"
                )
        if row["context_id"] in seen_ids:
            raise RuntimeError(
                f"[corpus] duplicate context_id {row['context_id']} in shard selection"
            )
        seen_ids.add(row["context_id"])
    print(
        f"[corpus] {len(rows)} rows selected + validated "
        f"(shard {args.shard_index}/{args.num_shards}); "
        f"content sha16={corpus_sha} over {len(fp_pairs)} corpus rows",
        flush=True,
    )
    return rows, corpus_sha


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


def gen_regime(args, template_sha: str, corpus_sha16: str) -> dict:
    """Gen-phase regime dict (generating parameters only — machine-stable).

    ``corpus_sha16`` is the FULL-corpus content fingerprint returned by
    ``load_corpus_rows`` (r2 SB-1(i)): ``corpus_prefix`` alone is a PATH — a
    corpus rebuilt in place at the same prefix would leave the regime equal
    and presence-skip stale chunks silently. REQUIRED, no default: a regime
    without the content fingerprint is the sealed-off r2 hole.
    """
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
        "corpus_content_sha16": corpus_sha16,
        "raw_prefix": args.raw_prefix,
        "disable_thinking": args.disable_thinking,
        "gdn_prefill": args.gdn_prefill or "",
        "limit": args.limit if args.limit is not None else 0,
        "template_sha16": template_sha,
    }


def capture_regime(args, template_sha: str, corpus_sha16: str) -> dict:
    """Capture-phase regime dict (adds batching knobs — padded-batch bf16 numerics)."""
    reg = gen_regime(args, template_sha, corpus_sha16)
    reg.update(
        {
            "phase": "capture",
            "out_prefix": args.out_prefix,
            "batch_tokens": args.batch_tokens,
            "max_batch_rows": args.max_batch_rows,
        }
    )
    return reg


# Shards of ONE run share the remote digest per prefix, so ONLY the shard's
# own identity (shard_index) is excluded from the cross-pod comparison.
# num_shards is RETAINED (r2 SB-1(ii)): it is output-affecting — it drives the
# ``idx % num_shards`` row partition AND the shard-count-free chunk keys
# (``s{shard:02d}_chunk{ci:04d}``), so a reshard (e.g. 4->2) at the same
# prefix reuses COLLIDING chunk files and MUST force a regime mismatch, never
# a presence-skip. All shards of one run share one num_shards value, so the
# shared-digest design is unchanged (r1 review g2 Major 1 fix spec).
REGIME_SHARD_FIELDS = ("shard_index",)


def _strip_shard_fields(regime: dict) -> dict:
    """Regime minus shard_index ONLY (shards of one run share a digest;
    num_shards stays — output-affecting, r2 SB-1(ii))."""
    return {k: v for k, v in regime.items() if k not in REGIME_SHARD_FIELDS}


def ensure_remote_regime(prefix: str, regime: dict, work: Path, *, write_if_absent: bool) -> None:
    """Cross-pod regime gate for every HF-presence resume path (g2 Major 1).

    The local StageLedger protects only same-workdir reruns; before ANY
    presence-skip against ``prefix`` the remote artifacts must be proven to
    have been produced under THIS regime. ``{prefix}/regime.json`` is written
    on the phase's first run (``write_if_absent=True`` — producer side) ONLY
    after a scoped listing proves the prefix holds NO pre-existing files
    (r2 SB-1(iii): a populated digest-less prefix is pre-regime or foreign
    residue — first-publishing over it would bless those files for every
    later presence-skip; the remedy is a fresh prefix). A consumer
    (``write_if_absent=False``) fails loud when the digest is absent.
    Mismatch (shard_index excluded; num_shards + corpus_content_sha16
    RETAINED) fails loud naming the differing keys — the remedy is a FRESH
    prefix, mirroring the StageLedger message. Regime values are generating
    parameters only (machine-stable; gotchas.md float-recompute rule).
    """
    dest = f"{prefix}/regime.json"
    want = _strip_shard_fields(regime)
    if hf_missing_of([dest], scope=prefix):
        if not write_if_absent:
            raise RuntimeError(
                f"[regime] {dest} absent — remote artifacts have no regime digest; "
                "run the producing phase (post-fix) first, or use a fresh prefix"
            )
        existing = hf_prefix_files(prefix)
        if dest not in existing:
            if existing:
                raise RuntimeError(
                    f"[regime] refusing FIRST publication of {dest}: prefix {prefix} "
                    f"already holds {len(existing)} file(s) with no regime digest "
                    f"(e.g. {sorted(existing)[:3]}) — pre-regime or foreign artifacts "
                    "would be presence-skipped as if produced under this regime; "
                    "use a fresh prefix (never bless residue in place)"
                )
            p = work / f"regime_pub_{hashlib.sha256(dest.encode('utf-8')).hexdigest()[:8]}.json"
            atomic_write_json(p, {"regime": want, "meta": run_metadata()})
            upload_single_file(p, dest)
            print(f"[regime] published {dest} (prefix proven empty pre-publish)", flush=True)
            return
        # Race grace: a sibling shard published the digest between the two
        # probes — fall through and VERIFY it instead of refusing/republishing.
        print(f"[regime] {dest} appeared during first-publish probe — verifying", flush=True)
    local = fetch_repo_file(dest, work / "regime_dl", what=f"regime({prefix})")
    have = json.loads(local.read_text(encoding="utf-8"))["regime"]
    local.unlink()
    if have != want:
        diff = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
        raise RuntimeError(
            f"[regime] mismatch at {dest} on keys {diff}: "
            f"remote {[(k, have.get(k)) for k in diff]} vs "
            f"requested {[(k, want.get(k)) for k in diff]} — remote artifacts were "
            "produced under a DIFFERENT regime; use a fresh prefix (never mix in place)"
        )
    print(f"[regime] verified {dest}", flush=True)


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


def _is_think_leak(row: dict) -> bool:
    """#2378 ``_classify_answer_row`` think-leak port (g2 Major 2).

    Trusts the gen-time ``think_leak`` flag when present; otherwise recomputes
    the substring check on the persisted completion text (defensive — works on
    any chunk row regardless of writer vintage).
    """
    flag = row.get("think_leak")
    if flag is not None:
        return bool(flag)
    return "<think>" in (row.get("completion") or "")


def redact_secret_rows(local: Path, key: str) -> int:
    """Drop whole secret-bearing rows from a gen chunk file BEFORE upload (#2502).

    Model completions can regurgitate real-secret-grade strings; the fail-closed
    upload gate (``assert_upload_clean`` inside ``hub._upload``) scans file bytes
    and refuses the whole chunk (Model B chunk0048 crashed there,
    SecretUploadGateError). A text-only scrub is a LEAK-BYPASS: the row's
    ``completion_token_ids`` decode straight back to the secret and the gate
    never scans the int array — so the ENTIRE line is dropped (text AND token
    ids; capture teacher-forces on stored ids of KEPT rows only, so kept rows
    are unaffected). Detection is the gate's own ``scan_bytes`` applied per
    line (the full line, so a secret in ANY field is caught): a per-line
    DUMMY_RX context window is a subset of the gate's whole-file window, so
    per-line flags a SUPERSET of gate findings and the rewritten file passes
    the gate by construction. Kept lines are byte-untouched. Content hygiene:
    count-only — never prints/raises row text or finding values. Returns the
    dropped-row count; rewrites (atomic same-dir replace) only when > 0;
    raises if ALL rows are flagged (never upload an empty chunk — the file is
    left on disk for investigation).
    """
    from explore_persona_space.orchestrate.secret_scrub import scan_bytes

    raw = local.read_bytes()
    kept: list[bytes] = []
    dropped = 0
    for ln in raw.split(b"\n"):
        if not ln.strip():
            continue
        if scan_bytes(ln, path=str(local)):
            dropped += 1
            continue
        kept.append(ln)
    if not dropped:
        return 0
    if not kept:
        raise RuntimeError(
            f"[gen] {key}: ALL {dropped} rows secret-flagged — refusing to upload an "
            "empty chunk (count-only; no row text). Investigate the generation output."
        )
    tmp = local.with_name(local.name + ".redact.tmp")
    tmp.write_bytes(b"\n".join(kept) + b"\n")
    tmp.replace(local)
    return dropped


def count_chunk_stats(path: Path) -> dict:
    """Recount a gen chunk file's stats (cross-pod resume path for gen_meta)."""
    n = cap = leak = gen_tok = 0
    cap_ids: list[str] = []
    for row in iter_jsonl(path):
        n += 1
        if row.get("cap_hit"):
            cap += 1
            cap_ids.append(str(row.get("context_id")))
        leak += 1 if _is_think_leak(row) else 0
        gen_tok += int(row.get("n_gen_tokens", 0))
    return {
        "n_rows": n,
        "n_cap_hit": cap,
        "n_think_leak": leak,
        "n_gen_tokens_sum": gen_tok,
        "cap_hit_ids": cap_ids,
    }


def cap_hit_halt(
    args, work: Path, frac: float, n_cap: int, n_rows: int, cap_ids_by_chunk: dict
) -> None:
    """DESIGNED cap-hit halt (g2 Major-family blocker #4): durable report + rc=7.

    Uploads ``cap_hit_report{suffix}.json`` (affected context ids per chunk +
    the regen recipe) under the raw prefix, then terminates rc=EXIT_CAP_HIT so
    the plan's mandatory >=2x-cap regen can never silently no-op and capture
    never runs on a >2%-capped corpus. os._exit is safe here: all durables
    (chunks, gen_meta, this report) are upload-verified before the exit.
    """
    report = {
        "verdict": "cap_hit_regen_required",
        "cap_hit_fraction": frac,
        "cap_hit_threshold": CAP_HIT_THRESHOLD,
        "n_cap_hit": n_cap,
        "n_rows": n_rows,
        "max_new_tokens": args.max_new_tokens,
        "regen_max_new_tokens": 2 * args.max_new_tokens,
        "regen_recipe": (
            f"re-run --phase gen with --max-new-tokens {2 * args.max_new_tokens} against "
            "FRESH --raw-prefix/--out-prefix (the regime digest forbids in-place mixing), "
            "then point capture at the fresh prefixes"
        ),
        "affected_context_ids_by_chunk": cap_ids_by_chunk,
        "meta": run_metadata(),
    }
    p = work / f"cap_hit_report{name_suffix(args)}.json"
    atomic_write_json(p, report)
    upload_single_file(p, f"{args.raw_prefix}/cap_hit_report{name_suffix(args)}.json")
    print(
        f"[cap-hit] HALT rc={EXIT_CAP_HIT}: fraction {frac:.4f} > {CAP_HIT_THRESHOLD} — "
        f"report uploaded; plan trigger: regen at max_new_tokens={2 * args.max_new_tokens}",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(EXIT_CAP_HIT)


def reuse_remote_gen_meta(args, work: Path, keys: list[str], regime: dict) -> dict | None:
    """No-op-resume fast path (g2 Minor 7): reuse the remote gen_meta.

    When THIS run generated nothing, a remote gen_meta covering every chunk key
    under the SAME (shard-stripped) regime replaces the ~N-chunk recount
    re-download. The cap-hit gate still fires from the reused fraction (#4).
    Returns the remote meta on reuse, else None (fall through to recount).
    """
    dest = f"{args.raw_prefix}/gen_meta{name_suffix(args)}.json"
    if hf_missing_of([dest], scope=args.raw_prefix):
        return None
    local = fetch_repo_file(dest, work / "genmeta_dl", what="gen-meta(reuse)")
    meta = json.loads(local.read_text(encoding="utf-8"))
    local.unlink()
    per_chunk = meta.get("per_chunk") or {}
    if set(per_chunk) != set(keys):
        return None
    if _strip_shard_fields(meta.get("regime") or {}) != _strip_shard_fields(regime):
        return None
    frac = float(meta.get("cap_hit_fraction", 0.0))
    print(f"[gen] resume: remote gen_meta reused (cap-hit fraction={frac:.4f})", flush=True)
    if frac > CAP_HIT_THRESHOLD:
        cap_ids = {k: s.get("cap_hit_ids", []) for k, s in per_chunk.items() if s.get("n_cap_hit")}
        cap_hit_halt(
            args, work, frac, int(meta.get("n_cap_hit", 0)), int(meta.get("n_rows", 0)), cap_ids
        )
    return meta


def require_gen_complete(args, work: Path) -> dict:
    """Capture-entry gate (#4 defensive leg): gen must be COMPLETE and un-capped.

    Fetches the shard's gen_meta; absent => gen never finished its accounting —
    fail loud. ``regen_required`` => the same designed rc=EXIT_CAP_HIT halt as
    the gen side (capture may be launched standalone on a different pod, so the
    gen-side halt alone does not protect this entry).
    """
    dest = f"{args.raw_prefix}/gen_meta{name_suffix(args)}.json"
    if hf_missing_of([dest], scope=args.raw_prefix):
        raise RuntimeError(f"[capture] {dest} absent — gen phase incomplete; run --phase gen first")
    local = fetch_repo_file(dest, work / "genmeta_dl", what="gen-meta(capture-gate)")
    meta = json.loads(local.read_text(encoding="utf-8"))
    local.unlink()
    if meta.get("regen_required"):
        print(
            f"[cap-hit] capture HALT rc={EXIT_CAP_HIT}: gen_meta regen_required=true "
            f"(fraction={meta.get('cap_hit_fraction')}) — regen at >=2x cap first",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(EXIT_CAP_HIT)
    return meta


def write_gen_meta(
    args,
    work: Path,
    keys: list[str],
    stats: dict,
    template_sha: str,
    regime: dict,
) -> None:
    """Assemble + upload gen_meta.json (cap-hit fraction over EVERY chunk).

    Chunks not generated this run are recounted from HF; the remote regime
    digest (verified at phase entry, BEFORE any presence-skip) guarantees they
    were produced under this same regime — so stamping ``regime`` here is
    provenance-correct (g2 Major 1). >2% cap-hit => designed halt (#4).
    """
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
    n_leak = sum(s.get("n_think_leak", 0) for s in stats.values())
    # Recounted chunks (produced by another run/pod) lack the key: dropped rows
    # are absent from the uploaded chunk, so a recount cannot recover the count.
    n_secret = sum(s.get("n_secret_redacted", 0) for s in stats.values())
    frac = n_cap / max(1, n_rows)
    meta = {
        "model": args.model,
        "n_chunks": len(keys),
        "n_rows": n_rows,
        "n_cap_hit": n_cap,
        "cap_hit_fraction": frac,
        "cap_hit_threshold": CAP_HIT_THRESHOLD,
        "regen_required": frac > CAP_HIT_THRESHOLD,
        "n_think_leak": n_leak,
        "think_leak_policy": "flagged at gen (rollout text persisted); dropped at capture",
        "n_secret_redacted": n_secret,
        "secret_redacted_policy": (
            "whole row (completion text + token ids) dropped at gen BEFORE upload "
            "(#2502; count-only, this-run chunks only — recounts cannot recover it)"
        ),
        "template_sha16": template_sha,
        "regime": regime,
        "per_chunk": stats,
        "meta": run_metadata(),
    }
    p = work / f"gen_meta{name_suffix(args)}.json"
    atomic_write_json(p, meta)
    upload_single_file(p, f"{args.raw_prefix}/gen_meta{name_suffix(args)}.json")
    print(
        f"[cap-hit] fraction={frac:.4f} ({n_cap}/{n_rows}) threshold={CAP_HIT_THRESHOLD}; "
        f"think_leak={n_leak}; secret_redacted={n_secret}",
        flush=True,
    )
    if frac > CAP_HIT_THRESHOLD:
        cap_ids = {k: s.get("cap_hit_ids", []) for k, s in stats.items() if s.get("n_cap_hit")}
        cap_hit_halt(args, work, frac, n_cap, n_rows, cap_ids)


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
    rows, corpus_sha = load_corpus_rows(args, work)
    chunks = [rows[i : i + args.chunk_size] for i in range(0, len(rows), args.chunk_size)]
    keys = [chunk_key(args, ci) for ci in range(len(chunks))]
    regime = gen_regime(args, template_sha, corpus_sha)
    # g2 Major 1: the remote digest gates EVERY presence-skip below — verified
    # (or first-run published) BEFORE the HF listing is consulted.
    ensure_remote_regime(args.raw_prefix, regime, work, write_if_absent=True)
    ledger = StageLedger(work / f"gen_ledger{name_suffix(args)}.json", regime)
    stats_path = work / f"gen_stats{name_suffix(args)}.json"
    stats: dict = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    dests = {k: f"{args.raw_prefix}/{k}.jsonl" for k in keys}
    hf_missing = hf_missing_of(list(dests.values()), scope=args.raw_prefix)
    budget = args.max_model_len - args.max_new_tokens
    eos_id = tok.eos_token_id
    llm = None
    sp = None
    n_generated = 0
    t0 = time.time()
    for ci, chunk_rows in enumerate(chunks):
        key = keys[ci]
        if ledger.is_done(key):
            continue
        if dests[key] not in hf_missing:
            print(f"[gen] {key}: present on HF — resume-skip (regime verified)", flush=True)
            ledger.mark_done(key)
            continue
        rendered = []
        over_budget = []
        for row in chunk_rows:
            pids = render_prompt_ids(tok, row["text"], disable_thinking=args.disable_thinking)
            if len(pids) > budget:
                over_budget.append(
                    {
                        "context_id": row["context_id"],
                        "context_sha": row["context_sha"],
                        "n_prompt_tokens": len(pids),
                    }
                )
                continue
            rendered.append((row, pids))
        if over_budget:
            # g2 blocker #2: u1's corpus is JOINT dual-tokenizer filtered, so ANY
            # per-model length drop is a corpus/render contract violation — there
            # is no drop tolerance (the old <=5% tolerate path is removed).
            raise RuntimeError(
                f"[gen] {key}: {len(over_budget)}/{len(chunk_rows)} rows over token "
                f"budget {budget} (first: {over_budget[0]}) — joint dual-tokenizer "
                "corpus contract violated, fail loud"
            )
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
        n_leak = 0
        gen_tok_sum = 0
        cap_ids: list[str] = []
        with local.open("w", encoding="utf-8") as fh:
            for (row, pids), out in zip(rendered, outs):
                o = out.outputs[0]
                comp_ids = [int(x) for x in o.token_ids]
                stripped = False
                while comp_ids and eos_id is not None and comp_ids[-1] == eos_id:
                    comp_ids.pop()
                    stripped = True
                cap_hit = o.finish_reason == "length"
                if cap_hit:
                    n_cap += 1
                    cap_ids.append(row["context_id"])
                # g2 Major 2 (#2378 _classify_answer_row port): flag at gen —
                # rollout text ALWAYS persists (upload policy); capture drops.
                think_leak = "<think>" in o.text
                n_leak += 1 if think_leak else 0
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
                    "think_leak": think_leak,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # #2502 crash fix: drop whole secret-bearing rows BEFORE the fail-closed
        # upload gate (hub._upload -> assert_upload_clean) scans the file.
        # NEVER scrub-text-and-keep-ids (token ids decode back to the secret)
        # and NEVER bypass the gate (EPM_SECRET_UPLOAD_GATE=0 is forbidden
        # here — the data repo is public). COUNT-ONLY log; no row text.
        n_secret = redact_secret_rows(local, key)
        if n_secret:
            print(f"[gen] {key}: secret_redacted {n_secret}", flush=True)
        sz = local.stat().st_size
        if sz > GEN_CHUNK_MAX_BYTES:
            raise RuntimeError(
                f"[gen] {key}.jsonl is {sz} bytes > {GEN_CHUNK_MAX_BYTES} non-LFS budget — "
                "lower --chunk-size"
            )
        upload_single_file(local, dests[key])
        if n_secret:
            # Dropped rows may have carried cap_hit / think_leak / token counts;
            # recount from the rewritten file so stats match uploaded content
            # (and the realized row-count reconciliation reads POST-drop rows).
            chunk_stats = count_chunk_stats(local)
        else:
            chunk_stats = {
                "n_rows": len(rendered),
                "n_cap_hit": n_cap,
                "n_think_leak": n_leak,
                "n_gen_tokens_sum": gen_tok_sum,
                "cap_hit_ids": cap_ids,
            }
        chunk_stats["n_secret_redacted"] = n_secret
        local.unlink()
        stats[key] = chunk_stats
        n_generated += 1
        atomic_write_json(stats_path, stats)
        ledger.mark_done(key)
        progress("gen", ci + 1, len(chunks), key, t0)
    if llm is not None:
        from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

        _reap_vllm_engine(llm)
        llm = None
    if n_generated == 0 and reuse_remote_gen_meta(args, work, keys, regime) is not None:
        # g2 Minor 7: pure no-op resume — remote gen_meta reused, no recount
        # re-downloads (cap-hit gate already applied inside the reuse helper).
        print("[phase=gen] done (resume; gen_meta reused)", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    write_gen_meta(args, work, keys, stats, template_sha, regime)
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
    drops = {"empty_completion": 0, "think_leak": 0}
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
        if _is_think_leak(row):
            # g2 Major 2 policy: leaked-CoT completions never enter v_x — the
            # row is excluded from tensors + rows.json (text stays in the gen
            # chunk on HF for audit); counted here and in capture_meta.
            drops["think_leak"] += 1
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
    if len(idx_of) != len(recs):
        # g2 Minor 6: a duplicate context_id would write one tensor row twice
        # and leave another all-zeros — refuse instead.
        raise RuntimeError(f"chunk {key}: duplicate context_id among {len(recs)} capture rows")
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
        "n_think_leak": sum(1 for g in gen_rows if _is_think_leak(g)),
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


GEN_ROW_REQUIRED_STR = ("context_id", "context_sha", "prompt_sha", "completion")


def validate_gen_rows(key: str, gen_rows: list[dict], ctx_lookup: dict) -> None:
    """Schema-validate one gen chunk BEFORE any model load (g2/Codex finding).

    Every required field of every row is checked pre-init so a malformed chunk
    fails in seconds, never after the heavy HF model load. Content hygiene:
    errors carry key + row index + field name (+ sha-derived ids), never text.
    """
    for i, row in enumerate(gen_rows):
        for f in GEN_ROW_REQUIRED_STR:
            v = row.get(f)
            if not isinstance(v, str) or (f != "completion" and not v):
                raise RuntimeError(
                    f"[capture-preflight] {key} row {i}: field {f!r} missing/empty/non-str"
                )
        ids = row.get("completion_token_ids")
        if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
            raise RuntimeError(
                f"[capture-preflight] {key} row {i} ({row['context_id']}): "
                "completion_token_ids missing or non-int"
            )
        if row["context_id"] not in ctx_lookup:
            raise RuntimeError(
                f"[capture-preflight] {key} row {i}: context_id {row['context_id']} "
                "absent from corpus shard selection"
            )


def phase_capture(args, spec) -> None:
    """P2: teacher-forced capture (cx_last + vx, all post-block layers), per-chunk."""
    print("[phase=capture] start", flush=True)
    enforce_model_env(args)
    work = work_root(args, spec)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    template_sha = assert_chat_template(tok, disable_thinking=args.disable_thinking)
    rows, corpus_sha = load_corpus_rows(args, work)
    ctx_lookup = {r["context_id"]: r["text"] for r in rows}
    n_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
    keys = [chunk_key(args, ci) for ci in range(n_chunks)]
    # g2 Major 1: verify the GEN artifacts' remote regime (consumer side — the
    # digest must already exist) BEFORE any gen chunk is trusted, and gate this
    # phase's own out_prefix (producer side) BEFORE any presence-skip.
    ensure_remote_regime(
        args.raw_prefix, gen_regime(args, template_sha, corpus_sha), work, write_if_absent=False
    )
    # #4 defensive: refuse a regen_required corpus. The returned meta also
    # carries per-chunk n_secret_redacted (#2502 gen-time drop class).
    gen_meta = require_gen_complete(args, work)
    regime = capture_regime(args, template_sha, corpus_sha)
    ensure_remote_regime(args.out_prefix, regime, work, write_if_absent=True)
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
    # Validate-before-init (g2/Codex finding): fetch + schema-validate EVERY
    # pending gen chunk BEFORE the heavy HF model load; files stay on disk so
    # the capture loop re-reads them without re-downloading.
    gen_paths: dict[str, Path] = {}
    for key in pending:
        gen_local = fetch_repo_file(
            f"{args.raw_prefix}/{key}.jsonl", work / "gen_dl", what=f"gen-chunk({key})"
        )
        gen_rows_v = list(iter_jsonl(gen_local))
        if not gen_rows_v:
            raise RuntimeError(f"gen chunk {key} is empty — fail loud")
        validate_gen_rows(key, gen_rows_v, ctx_lookup)
        gen_paths[key] = gen_local
        del gen_rows_v
    summaries: dict[str, dict] = {}
    if pending:
        print(
            f"[capture] preflight OK: {len(pending)} pending gen chunks schema-validated "
            "pre-model-load",
            flush=True,
        )
        mctx = load_model_ctx(args, spec, tok)
        t0 = time.time()
        for ci, key in enumerate(keys):
            if ledger.is_done(key) or key in complete_on_hf:
                continue
            gen_local = gen_paths[key]
            gen_rows = list(iter_jsonl(gen_local))
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
        "n_think_leak_drops": sum(s["drops"].get("think_leak", 0) for s in summaries.values()),
        # #2502 gen-time drop class: rows dropped BEFORE upload (secret-bearing
        # line = text + token ids), so they are absent from n_gen_rows above;
        # read from gen_meta (capture never sees them). Recounted gen chunks
        # lack the key => counted 0 (the count is unrecoverable post-drop).
        "n_secret_redacted_drops": sum(
            int((s or {}).get("n_secret_redacted", 0))
            for s in (gen_meta.get("per_chunk") or {}).values()
        ),
    }
    # Producer contract for the downstream fits/reliability unit: exact chunk
    # keys + per-chunk counts + the COMPLETE captured layer set + the full
    # expected-file enumeration (repo paths under out_prefix).
    layers = list(range(1, spec["n_layers"] + 1))
    meta = {
        "model": args.model,
        "totals": totals,
        "regime": regime,
        "chunk_keys": keys,
        "layers": layers,
        "layer_map": "L{k} = hidden_states[k] = output of decoder block k-1 (0-indexed)",
        "expected_files": [p for k in keys for p in exp_by_key[k]],
        "per_chunk": summaries,
        "meta": run_metadata(),
    }
    mp = work / f"capture_meta{name_suffix(args)}.json"
    atomic_write_json(mp, meta)
    upload_single_file(mp, f"{args.out_prefix}/capture_meta{name_suffix(args)}.json")
    # Sentinel uploads LAST (phase-done contract for downstream fits units).
    sentinel_name = ".capture_done" if args.num_shards == 1 else f".capture_done{name_suffix(args)}"
    sp = work / f"capture_done{name_suffix(args)}.json"
    atomic_write_json(
        sp,
        {
            "done": True,
            "totals": totals,
            "chunk_keys": keys,
            "layers": layers,
            "capture_meta": f"capture_meta{name_suffix(args)}.json",
            "meta": run_metadata(),
        },
    )
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
        if proc.returncode == EXIT_CAP_HIT:
            # Designed stop (#4), not a crash: propagate the distinct rc so the
            # dispatcher routes it as the plan's cap-hit regen trigger; capture
            # never runs on a >2%-capped corpus.
            print(
                f"[phase=all] --phase {ph} HALT rc={EXIT_CAP_HIT} (cap-hit regen required; "
                "cap_hit_report uploaded) — designed stop, remaining phases not run",
                flush=True,
            )
            return EXIT_CAP_HIT
        if proc.returncode != 0:
            raise RuntimeError(f"--phase {ph} subprocess exited rc={proc.returncode}")
        print(f"[phase=all] --phase {ph} rc=0", flush=True)
    print("[phase=done] gen+capture complete", flush=True)
    return 0


# --------------------------------------------------------------------------
# Self-test (VM-safe: no model load, no network, no GPU)
# --------------------------------------------------------------------------


def run_self_test() -> int:
    """Synthetic unit checks: codec, MF-F asserts, packing, ledger regime,
    think-leak classifier, regime shard-strip, gen-row preflight validation."""
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
    # 8. think-leak classifier (g2 Major 2 port): flag wins; text fallback works.
    assert _is_think_leak({"completion": "x <think> y"})
    assert not _is_think_leak({"completion": "clean answer"})
    assert not _is_think_leak({"think_leak": False, "completion": "<think>"})
    assert _is_think_leak({"think_leak": True, "completion": "clean"})
    n_pass += 1
    # 9. Regime shard-strip (r2 SB-1(ii)): ONLY shard_index is excluded —
    # num_shards is output-affecting (row partition + chunk-key namespace).
    reg = {"a": 1, "num_shards": 4, "shard_index": 2}
    assert _strip_shard_fields(reg) == {"a": 1, "num_shards": 4}
    # Shards of ONE run share the stripped digest; a reshard must NOT.
    assert _strip_shard_fields({**reg, "shard_index": 0}) == _strip_shard_fields(reg)
    assert _strip_shard_fields({**reg, "num_shards": 2}) != _strip_shard_fields(reg)
    n_pass += 1
    # 10. Gen-row preflight validation: good row passes; bad rows raise.
    good = {
        "context_id": "c1",
        "context_sha": "s1",
        "prompt_sha": "p1",
        "completion": "",
        "completion_token_ids": [1, 2],
    }
    validate_gen_rows("t", [good], {"c1": "x"})
    for bad, lookup in (
        (dict(good, prompt_sha=None), {"c1": "x"}),  # missing required field
        (dict(good, completion_token_ids="oops"), {"c1": "x"}),  # non-list ids
        (good, {}),  # context_id absent from corpus selection
    ):
        try:
            validate_gen_rows("t", [bad], lookup)
            raise AssertionError("bad gen row not caught")
        except RuntimeError:
            pass
    n_pass += 1
    # 11. Corpus content fingerprint (r2 SB-1(i)): order-invariant over the
    # row set; any row-set change flips it; carried in BOTH regime dicts.
    pairs = [("b", "2"), ("a", "1")]
    assert corpus_content_sha16(pairs) == corpus_content_sha16(list(reversed(pairs)))
    assert corpus_content_sha16(pairs) != corpus_content_sha16(pairs[:1])
    ns = argparse.Namespace(
        model="m",
        env="e",
        seed=1,
        max_new_tokens=2,
        max_model_len=8,
        chunk_size=1,
        num_shards=4,
        shard_index=2,
        corpus_prefix="c",
        raw_prefix="r",
        out_prefix="o",
        disable_thinking=False,
        gdn_prefill=None,
        limit=None,
        batch_tokens=16,
        max_batch_rows=4,
    )
    r_gen = gen_regime(ns, "tsha", "fp0")
    assert r_gen["corpus_content_sha16"] == "fp0" and r_gen["num_shards"] == 4
    r_cap = capture_regime(ns, "tsha", "fp0")
    assert r_cap["corpus_content_sha16"] == "fp0" and r_cap["phase"] == "capture"
    n_pass += 1
    # 12-14. ensure_remote_regime SB-1 seal: refusals exercised through the
    # REAL gate with the network seams faked at module level (no network).
    mod = sys.modules[__name__]
    seams = ("hf_missing_of", "fetch_repo_file", "upload_single_file", "hf_prefix_files")
    saved = {n: getattr(mod, n) for n in seams}
    try:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            remote: dict[str, dict] = {}  # dest -> published (stripped) regime
            force_missing: set[str] = set()
            listing: list[str] = []
            uploads: list[str] = []

            def fake_missing(expected, *, scope):
                return {p for p in expected if p in force_missing or p not in remote}

            def fake_fetch(repo_path, dest_root, *, what):
                p = Path(dest_root) / "regime.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"regime": remote[repo_path]}), encoding="utf-8")
                return p

            def fake_upload(local, dest):
                uploads.append(dest)
                remote[dest] = json.loads(Path(local).read_text(encoding="utf-8"))["regime"]

            def fake_listing(prefix):
                return list(listing)

            mod.hf_missing_of = fake_missing
            mod.fetch_repo_file = fake_fetch
            mod.upload_single_file = fake_upload
            mod.hf_prefix_files = fake_listing

            base = gen_regime(ns, "tsha", "fp0")
            # 12a. Fresh EMPTY prefix: first publication succeeds.
            ensure_remote_regime("pfx", base, work, write_if_absent=True)
            assert uploads == ["pfx/regime.json"], uploads
            # 12b. num_shards mismatch (reshard 4->2) REFUSES against the digest.
            ns2 = argparse.Namespace(**{**vars(ns), "num_shards": 2, "shard_index": 0})
            try:
                ensure_remote_regime(
                    "pfx", gen_regime(ns2, "tsha", "fp0"), work, write_if_absent=True
                )
                raise AssertionError("num_shards reshard not refused")
            except RuntimeError as e:
                assert "num_shards" in str(e), e
            # Same-run sibling shard (same num_shards) still verifies clean.
            ns_sib = argparse.Namespace(**{**vars(ns), "shard_index": 0})
            ensure_remote_regime(
                "pfx", gen_regime(ns_sib, "tsha", "fp0"), work, write_if_absent=True
            )
            n_pass += 1
            # 13. Corpus-content mismatch (rebuilt corpus, same prefix) REFUSES.
            try:
                ensure_remote_regime(
                    "pfx", gen_regime(ns, "tsha", "fp1"), work, write_if_absent=True
                )
                raise AssertionError("corpus content mismatch not refused")
            except RuntimeError as e:
                assert "corpus_content_sha16" in str(e), e
            n_pass += 1
            # 14. First publication onto a NON-EMPTY digest-less prefix REFUSES
            # (SB-1(iii)); consumer-absent still refuses; sibling-publish race
            # falls through to VERIFY (no second publish).
            listing[:] = ["pfx2/chunk0000.jsonl"]
            try:
                ensure_remote_regime("pfx2", base, work, write_if_absent=True)
                raise AssertionError("non-empty digest-less prefix not refused")
            except RuntimeError as e:
                assert "refusing FIRST publication" in str(e), e
            assert "pfx2/regime.json" not in remote
            try:
                ensure_remote_regime("pfx2", base, work, write_if_absent=False)
                raise AssertionError("consumer absent-digest not refused")
            except RuntimeError:
                pass
            # Race grace: digest present in the listing (and fetchable) while
            # the missing-probe still reports absent -> verify, don't publish.
            force_missing.add("pfx/regime.json")
            listing[:] = ["pfx/regime.json", "pfx/chunk0000.jsonl"]
            ensure_remote_regime("pfx", base, work, write_if_absent=True)
            assert uploads == ["pfx/regime.json"], uploads  # no re-publish
            n_pass += 1
    finally:
        for n, fn in saved.items():
            setattr(mod, n, fn)
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
    ap.add_argument(
        "--rows",
        type=int,
        dest="limit",
        help="alias for --limit (plan §4.8/§9 smoke command shape uses --rows)",
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
