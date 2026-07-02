#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, M⁺, →, ×, c_C, v_A, ‖·‖) in scientific docstrings + log messages.
"""Issue #813 — ONE (behavior, substrate) map-fit extraction cell.

For ONE ``(behavior, substrate)`` cell over the shared 50-context #594 battery
this CLI:

1. Stages + loads the #537 ``default``-context adapter for ``behavior`` as a
   ``PeftModel`` on base Qwen-2.5-7B-Instruct (rsLoRA honored; the em adapter's
   NESTED ``sft_em_adapter/`` subfolder resolved by ``resolve_adapter_subfolder``
   inherited from #667). Asserts base id + ``use_rslora`` (fitness (f)/(g)).
2. For each of the 50 battery contexts × the substrate's K questions:
   - builds ``T_ctx(q)`` via the #594 ``messages_for_instance`` recipe;
   - generates the frozen BASE greedy response ``R`` (temp=0, vLLM batched);
   - teacher-forces ``T_ctx(q) + R`` through base θ0 AND θ⁺ once each, capturing
     the FULL per-token residual over (ctx span + answer span) at ALL 28 layers,
     fp16, BOTH models;
   - STREAM-UPLOADS this cell's unreduced ``.npz`` to HF, then DELETES local
     (peak local footprint stays ~one cell — never accumulate; #664/EDQUOT);
   - reduces to ``c_C`` (last-input-token, 28 layers) + ``v_A`` (mean-answer-span,
     28 layers) per question → accumulates.
3. Question-averages ``c_C`` / ``v_A`` over the substrate's questions →
   50 ``c_C`` rows + 50 ``v_A`` rows (base + trained), 28 layers → writes the
   reduced per-(behavior, substrate) summary ``.npz`` (the map-fit input; small,
   ~1 MB) locally under ``eval_results/issue_813/reduced/<behavior>/<substrate>/``.

Restart-safe (all inert on a fresh launch): a per-cell PID lockfile makes a
duplicate exec of a LIVE cell WAIT for the owner, then exit 0 iff the owner
wrote ``.done`` (exit 1 otherwise — fail loud on a mid-cell owner death), so
the dispatcher's wave-2 re-exec of a manually pre-launched cell neither
double-runs nor lets the dispatcher's p.wait() advance to the fit phase while
extraction is still in progress; on resume the cell skip-lists rows already
uploaded (ONE ``list_repo_files`` call), reuses the persisted base-greedy
responses (``r_lookup.json`` — removes vLLM nondeterminism across restarts),
re-enqueues complete-but-unflushed local ``.npz`` files, and — when the local
accumulator checkpoint (``accum_ckpt.npz``) exactly covers the uploaded rows —
skips their forward passes entirely.

CONTENT HYGIENE: ``em`` uses Betley harmful-content probes — this script NEVER
prints/logs their text; it digests by row/token COUNT + activations only
(``r_lookup.json`` is written/read as a file, never logged). Benign behaviors
(marker/fact/sycophancy) are unaffected.

Activation extraction is ``transformers`` forward hooks (NOT vLLM). vLLM is used
ONLY for the frozen base-R generation (which returns text, no activations).

Usage (one cell)::

    uv run python scripts/issue813_run_cell.py \\
        --behavior marker --substrate generic \\
        --out-root eval_results/issue_813 --gpu-id 0 \\
        --upload  # stream unreduced .npz to HF; omit for local-only smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
# vLLM EngineCore fork() poisoning guard (.claude/rules/gotchas.md § entry 26 /
# issue667_extract.py): a pre-LLM() transformers/tokenizer touch poisons the
# EngineCore fork. spawn (not fork) avoids the silent worker death.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue594_common as i594  # noqa: E402

# The reused #667 extractor primitives (adapter-load, rsLoRA gauge, vLLM gen,
# teacher-force reads via extract_layer_activations forward hooks). These are
# imported VERBATIM — no local re-implementation of the plumbing.
import issue667_extract as ex667  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue813.run_cell")

# ── Constants ────────────────────────────────────────────────────────────────
DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue813_mapchange_substrate"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN = 3584
N_LAYERS = 28  # Qwen-2.5-7B-Instruct decoder blocks 0..27
# Frozen headline read layer (#651/#658 primary read layer). The substrate-swap
# null (issue813_analysis.py) resamples questions at THIS layer only, so the
# per-question headline-layer c_C/v_A rows are persisted for it (keeping the
# small reduced summary from ballooning to all-28-layer per-question size).
HEADLINE_LAYER = 14
SEED = 42
BEHAVIORS = ("em", "fact", "sycophancy", "marker")
SUBSTRATES = ("generic", "elicit", "mix")
# Per-behavior base-R generation cap (marker end-of-completion needs ≥2048; #260).
MAX_NEW_TOKENS = {"marker": 2048, "fact": 1024, "sycophancy": 1024, "em": 1024}
# Atomic per-(behavior, substrate) completion sentinel (resume-skip predicate).
CELL_DONE_SENTINEL = ".done"
# Per-cell PID lockfile (next to .done): a duplicate exec of a cell owned by a
# LIVE process WAITS for the owner to finish — an instant exit-0 would let the
# dispatcher's p.wait() read success and advance to the fit phase while the
# manually pre-launched owner is still extracting (fit on incomplete reduced/
# data). On owner exit the duplicate exits 0 iff the owner wrote .done, else
# fails loud (exit 1) so the dispatcher surfaces the partial cell.
RUNNING_LOCK_NAME = ".running.pid"
# Duplicate-exec wait tuning (module constants so tests can shrink them): poll
# the owner pid every DUP_WAIT_POLL_S; heartbeat-log every DUP_WAIT_HEARTBEAT_S.
# No timeout cap by design — the dispatcher's p.wait() has none either.
DUP_WAIT_POLL_S = 30.0
DUP_WAIT_HEARTBEAT_S = 600.0
# Cell-scoped resume artifacts, both under the cell's unreduced_tmp SCRATCH dir
# (never under reduced/, which is committed to git — em r_lookup rows carry
# harmful-content completions): the persisted Phase-A base-greedy responses
# (uploaded alongside the cell's unreduced prefix) and the local-only reduced-
# accumulator checkpoint (overwritten at every upload flush, NEVER uploaded).
R_LOOKUP_NAME = "r_lookup.json"
ACCUM_CKPT_NAME = "accum_ckpt.npz"
# The four reduced accumulator keys (c_C = last-input-token, v_A = mean-answer-span).
_PQ_KEYS = ("c_C_base", "c_C_trained", "v_A_base", "v_A_trained")
# Unreduced-.npz upload batching (B3, #664/#488): a per-file HfApi().upload_file
# inside the (context, question) loop makes one HF commit per pair (~23,850 commits
# across the sweep, blowing the 256-commits/hr throttle). Buffer each cell's .npz
# files and flush ONE HfApi.create_commit per BATCH_UPLOAD_CHUNK files (many
# CommitOperationAdds per commit), deleting local per flush so peak local footprint
# stays ~one chunk (~BATCH_UPLOAD_CHUNK × per-cell bytes), never the full grid.
# 50 (halved from 100 when the unreduced saves went UNCOMPRESSED — ~1.29x bigger
# files; halving keeps peak tmp footprint + flush stalls similar).
BATCH_UPLOAD_CHUNK = 50


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ── Substrate question pools (the single manipulated variable) ─────────────────


def _generic_probes() -> list[str]:
    """The 48 UltraChat generic probes (#594 probes_ultrachat.json `probes`)."""
    path = PROJECT_ROOT / "data" / "issue594" / "probes_ultrachat.json"
    d = json.loads(path.read_text())
    return [p["text"] if isinstance(p, dict) else str(p) for p in d["probes"]]


def substrate_questions(
    behavior: str, substrate: str, *, max_questions: int | None = None
) -> list[str]:
    """The question pool defining c_C + v_A over the battery, per substrate (plan §4.2/§5).

    - generic: 48 UltraChat probes (anchor; behavior does NOT fire → E≈0).
    - elicit:  the behavior's own #537 eval pool (marker 32 / fact 30 / syco 25 / em 8).
    - mix:     equal-half blend, size ``2·min(n_e, 48)`` — ``min(n_e,48)`` generic +
               all ``n_e`` eliciting (equalize-down 1:1, seed-42 generic subsample),
               so the mix is NOT silently generic-dominated (plan §5).
    """
    generic = _generic_probes()
    elicit = ex667.load_eval_probes(behavior)  # #537 pool, flat list[str] (reused)
    if substrate == "generic":
        qs = list(generic)
    elif substrate == "elicit":
        qs = list(elicit)
    elif substrate == "mix":
        n_e = len(elicit)
        n_g = min(n_e, len(generic))
        rng = np.random.default_rng(SEED)
        gen_idx = sorted(rng.choice(len(generic), size=n_g, replace=False).tolist())
        qs = [generic[i] for i in gen_idx] + list(elicit)
    else:
        raise ValueError(f"unknown substrate {substrate!r} (expected one of {SUBSTRATES})")
    if max_questions is not None:
        qs = qs[:max_questions]
    return qs


# ── Battery contexts (the 50 map inputs) ───────────────────────────────────────


def load_battery_instances(max_contexts: int | None = None) -> list[dict]:
    """The 50 #594 battery contexts (the shared map inputs), optionally capped (smoke)."""
    _meta, instances = i594.load_battery(PROJECT_ROOT / "data" / "issue594" / "battery.json")
    if max_contexts is not None:
        instances = instances[:max_contexts]
    return instances


# ── Full per-token residual capture (unreduced) + c_C / v_A reduction ──────────


@torch.no_grad()
def _capture_full_and_reduce(
    base_model, trained_model, tok, messages: list[dict], response: str, device
) -> dict:
    """Teacher-force ``messages + response`` through base+trained; capture FULL residuals.

    Returns a dict with:
      - ``full_base`` / ``full_trained``: (T, 28, HIDDEN) fp16 per-token residual over
        the WHOLE sequence (ctx span + answer span), all 28 layers — the UNREDUCED store.
      - ``c_C_base`` / ``c_C_trained``: (28, HIDDEN) fp32 last-input-token residual (the
        map INPUT, #594/#658 recipe).
      - ``v_A_base`` / ``v_A_trained``: (28, HIDDEN) fp32 mean-over-answer-span residual
        (the map OUTPUT, #658 v0(C) recipe).
      - ``prompt_len`` / ``full_len``: token counts (digest only, no text).

    The answer span is [prompt_len : full_len); the last-input-token slot is
    prompt_len - 1 (the generation-prompt suffix position). Fails loud on an empty
    response span.
    """
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_msgs = [*messages, {"role": "assistant", "content": response}]
    full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    p = len(prompt_ids)
    if full_ids[:p] != prompt_ids:
        # chat-template drift; fall back to the longest common prefix (fail-loud if tiny).
        lcp = 0
        for a, b in zip(prompt_ids, full_ids, strict=False):
            if a != b:
                break
            lcp += 1
        if lcp < max(1, p - 4):
            raise RuntimeError(
                f"prompt-prefix drift: lcp={lcp} vs prompt_len={p} — chat-template mismatch"
            )
        p = lcp
    full_len = len(full_ids)
    if full_len <= p:
        raise RuntimeError("empty response span — base R produced zero tokens")
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    all_layers = list(range(N_LAYERS))
    acts_b = extract_layer_activations(base_model, ids, all_layers)  # {li: (1, T, H)}
    acts_t = extract_layer_activations(trained_model, ids, all_layers)

    # UNREDUCED: (T, 28, HIDDEN) fp16, both models — ONE stacked ON-DEVICE cast +
    # ONE host transfer per model (the old per-layer .float().cpu() loop shipped 28
    # fp32 copies over PCIe per model; the direct bf16→fp16 cast equals the old
    # bf16→fp32→fp16 round-trip for all finite values).
    stack_b = torch.stack([acts_b[li][0] for li in all_layers], dim=1)  # (T, 28, H) on device
    stack_t = torch.stack([acts_t[li][0] for li in all_layers], dim=1)
    assert stack_b.shape == (full_len, N_LAYERS, HIDDEN), stack_b.shape
    assert stack_t.shape == (full_len, N_LAYERS, HIDDEN), stack_t.shape
    full_base = stack_b.half().cpu().numpy()
    full_trained = stack_t.half().cpu().numpy()
    assert full_base.shape == (full_len, N_LAYERS, HIDDEN), full_base.shape
    assert full_base.dtype == np.float16 and full_trained.dtype == np.float16

    # REDUCED: c_C = last-input-token (slot p-1); v_A = mean over answer span [p:full_len).
    c_c_base = np.stack([acts_b[li][0, p - 1, :].float().cpu().numpy() for li in all_layers])
    c_c_trained = np.stack([acts_t[li][0, p - 1, :].float().cpu().numpy() for li in all_layers])
    v_a_base = np.stack(
        [acts_b[li][0, p:full_len, :].float().mean(0).cpu().numpy() for li in all_layers]
    )
    v_a_trained = np.stack(
        [acts_t[li][0, p:full_len, :].float().mean(0).cpu().numpy() for li in all_layers]
    )
    for name, arr in (("c_C", c_c_base), ("v_A", v_a_base)):
        assert arr.shape == (N_LAYERS, HIDDEN), f"{name} {arr.shape}"
    return {
        "full_base": full_base,
        "full_trained": full_trained,
        "c_C_base": c_c_base.astype(np.float32),
        "c_C_trained": c_c_trained.astype(np.float32),
        "v_A_base": v_a_base.astype(np.float32),
        "v_A_trained": v_a_trained.astype(np.float32),
        "prompt_len": p,
        "full_len": full_len,
    }


# ── HF stream-upload of one unreduced (context, question) .npz ─────────────────


def _hf_upload_file(local_path: Path, path_in_repo: str) -> None:
    """Upload ONE (small, reduced) file to the HF data repo, fail-loud.

    Reserved for the tiny per-(behavior, substrate) reduced summaries (``summary.npz`` /
    ``per_question_L14.npz``) — 2 commits per cell, well under the 256/hr throttle. The
    MANY unreduced per-(context, question) ``.npz`` uploads go through
    ``_hf_batch_commit`` (one commit per BATCH_UPLOAD_CHUNK files), never here (B3).

    Uses ``HfApi.upload_file`` directly (accelerated by the shell-level
    HF_XET_HIGH_PERFORMANCE / HF_HUB_ENABLE_HF_TRANSFER defaults). Raises on failure.
    """
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=DATA_REPO,
        repo_type="dataset",
        commit_message=f"issue813: reduced summary {path_in_repo} ({_git_sha()[:8]})",
    )


def _hf_batch_commit(items: list[tuple[Path, str]]) -> None:
    """Upload a BATCH of unreduced .npz files in ONE HF commit, fail-loud (B3, #664/#488).

    ``items`` is a list of ``(local_path, path_in_repo)`` pairs. Builds one
    ``HfApi.create_commit`` with a ``CommitOperationAdd`` per file — so a whole chunk
    of per-(context, question) ``.npz`` uploads costs ONE commit instead of one-per-file
    (the ~23,850-commit storm the per-file loop caused, blowing the 256-commits/hr
    throttle). Raises on failure (a clean batch upload IS the data-safety contract —
    the caller deletes local only AFTER this returns; upload-then-delete). No-op on an
    empty batch.
    """
    if not items:
        return
    from huggingface_hub import CommitOperationAdd, HfApi

    ops = [
        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local_path))
        for local_path, path_in_repo in items
    ]
    HfApi().create_commit(
        repo_id=DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=(
            f"issue813: unreduced activations batch ({len(items)} files, {_git_sha()[:8]})"
        ),
    )


def _df_free_gib(path: str = "/workspace") -> float | None:
    """Free GiB at ``path`` (df monitoring), or None if the path is absent (local VM)."""
    try:
        usage = shutil.disk_usage(path)
    except (FileNotFoundError, OSError):
        return None
    return usage.free / 2**30


# ── Resume + duplicate-exec plumbing (all inert on a fresh cell launch) ─────────


def _row_key(ctx_id: str, qi: int) -> str:
    """The stable per-(context, question) row key: ``{ctx_id}__q{qi}`` (the .npz stem)."""
    return f"{ctx_id}__q{qi}"


def _unreduced_prefix(behavior: str, substrate: str) -> str:
    """This cell's HF unreduced prefix (skip-set filter + upload path_in_repo namespace)."""
    return f"{EXPERIMENT_NAME}/unreduced/{behavior}/{substrate}"


def _unreduced_tmp_dir(out_root: Path, behavior: str, substrate: str) -> Path:
    """The cell's local scratch dir (unreduced .npz + resume artifacts; never committed)."""
    return out_root / "unreduced_tmp" / behavior / substrate


def _pid_alive(pid: int) -> bool:
    """True iff ``pid`` is a live process (EPERM counts as live, ESRCH as dead)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_cell_lock(reduced_dir: Path) -> int | None:
    """Take the per-cell PID lockfile; return the live owner's pid if already held.

    Writes our own pid to ``.running.pid`` and returns None when the lock is
    acquired (no file, unparseable pid, or dead pid — a SIGKILLed run's stale
    lock is taken over). Returns the owner pid WITHOUT touching the file when
    the recorded pid is a live process — the caller then WAITS on the owner
    (``_wait_for_cell_owner``); it must NOT exit 0 immediately, or the
    dispatcher's p.wait() would read success and advance to the fit phase
    while the owner is still extracting. Advisory check-then-write (no flock):
    the dispatcher launches duplicates seconds-to-hours apart, never atomically.
    """
    lock = reduced_dir / RUNNING_LOCK_NAME
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except ValueError:
            pid = None
        # pid == os.getpid() is a recycled-pid stale lock (a SIGKILLed run's pid
        # reassigned to this very process) — waiting on ourselves would deadlock;
        # treat it as stale and take over (review concern C2).
        if pid is not None and pid != os.getpid() and _pid_alive(pid):
            return pid
    lock.write_text(str(os.getpid()))
    return None


def _wait_for_cell_owner(reduced_dir: Path, owner: int, *, behavior: str, substrate: str) -> dict:
    """Block until the LIVE lock owner exits; route the duplicate's exit on ``.done``.

    The dispatcher's duplicate wave-2 exec must NOT exit 0 while the manually
    pre-launched owner is still extracting — the dispatcher's p.wait() would
    read success for every wave-2 cell and advance to the fit/analysis phase
    on incomplete reduced/ data. Instead: poll the owner pid (every
    ``DUP_WAIT_POLL_S``, heartbeat-logging every ``DUP_WAIT_HEARTBEAT_S``, no
    timeout cap — the dispatcher has none either); when the owner exits,
    return the skip dict (exit 0) iff the owner wrote the cell's ``.done``
    sentinel, else raise RuntimeError (exit 1, fail loud — the owner died
    mid-cell and the dispatcher must surface it rather than fit on a partial
    cell). Never touches the owner's lockfile.
    """
    sentinel = reduced_dir / CELL_DONE_SENTINEL
    t0 = time.monotonic()
    last_beat = t0
    logger.info(
        "cell already owned by live pid %d — waiting for it to finish (duplicate exec)", owner
    )
    while _pid_alive(owner):
        time.sleep(DUP_WAIT_POLL_S)
        now = time.monotonic()
        if now - last_beat >= DUP_WAIT_HEARTBEAT_S:
            logger.info(
                "still waiting on cell owner pid %d (%.0f min elapsed)", owner, (now - t0) / 60.0
            )
            last_beat = now
    waited_min = (time.monotonic() - t0) / 60.0
    if sentinel.exists():
        logger.info(
            "cell owner pid %d completed (.done present) after %.1f min wait — "
            "exiting 0 (duplicate exec)",
            owner,
            waited_min,
        )
        return {
            "skipped": True,
            "behavior": behavior,
            "substrate": substrate,
            "duplicate_of_pid": owner,
            "waited_min": waited_min,
        }
    raise RuntimeError(
        f"cell owner pid {owner} died after {waited_min:.1f} min without writing {sentinel} — "
        "cell incomplete; failing loud so the dispatcher surfaces it instead of fitting on a "
        "partial cell"
    )


def _hf_uploaded_rows(behavior: str, substrate: str) -> tuple[set[str], bool]:
    """ONE list_repo_files call → (already-uploaded row keys, r_lookup.json on HF?).

    Row keys are the ``{ctx_id}__q{qi}`` stems under this cell's unreduced prefix
    on the HF data repo — the resume skip-set. Fail-loud (network/auth raise).
    """
    from huggingface_hub import list_repo_files

    prefix = _unreduced_prefix(behavior, substrate) + "/"
    names = {
        f[len(prefix) :]
        for f in list_repo_files(DATA_REPO, repo_type="dataset")
        if f.startswith(prefix)
    }
    keys = {n[: -len(".npz")] for n in names if n.endswith(".npz")}
    return keys, R_LOOKUP_NAME in names


def _npz_is_complete(path: Path) -> bool:
    """True iff ``path`` is a readable npz carrying BOTH full-residual arrays.

    Deliberate completeness classifier (not error hiding): a truncated write
    from an interrupted process fails to open as a zip or lacks the entries;
    reading the central directory is cheap (no array data is loaded).
    """
    import zipfile

    try:
        with np.load(path) as z:
            return "full_base" in z.files and "full_trained" in z.files
    except (zipfile.BadZipFile, OSError, ValueError):
        return False


def _reenqueue_local_npz(
    tmp_dir: Path, hf_keys: set[str], unreduced_prefix: str
) -> list[tuple[Path, str]]:
    """Re-enqueue COMPLETE local unreduced .npz files left by an interrupted run.

    Returns the (local_path, path_in_repo) items to upload in the first flush.
    A file whose row is already on HF (``hf_keys``) is deleted, not re-enqueued
    (double-enqueue guard); an incomplete/truncated file is left for the row
    loop to recompute + overwrite; the local-only ``accum_ckpt.npz`` (and any
    interrupted ``*.tmp.npz`` atomic write) is NEVER enqueued.
    """
    pending: list[tuple[Path, str]] = []
    for p in sorted(tmp_dir.glob("*.npz")):
        if p.name == ACCUM_CKPT_NAME or p.name.endswith(".tmp.npz"):
            continue  # local-only checkpoint / interrupted atomic write — never uploaded
        if p.stem in hf_keys:
            p.unlink()  # already uploaded by a prior run — drop the stale local copy
            continue
        if not _npz_is_complete(p):
            logger.info("resume: incomplete local %s — will recompute", p.name)
            continue
        pending.append((p, f"{unreduced_prefix}/{p.name}"))
    return pending


def _write_r_lookup_json(path: Path, r_lookup: dict[tuple[int, int], str], contexts) -> None:
    """Persist ``{(ci, qi): R}`` as ONE json keyed by the stable row key (atomic write).

    CONTENT HYGIENE: written/read as a file only — the response text is never
    logged (em rows carry Betley harmful-content completions).
    """
    payload = {_row_key(contexts[ci]["id"], qi): r for (ci, qi), r in sorted(r_lookup.items())}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    os.replace(tmp, path)


def _load_r_lookup_json(path: Path, contexts) -> dict[tuple[int, int], str]:
    """Load a persisted r_lookup.json → ``{(ci, qi): R}`` keyed back to battery indices.

    Rows for contexts absent from the current run (a smaller ``--max-contexts``)
    are dropped, not crashed on.
    """
    ci_by_id = {inst["id"]: ci for ci, inst in enumerate(contexts)}
    out: dict[tuple[int, int], str] = {}
    for key, text in json.loads(path.read_text()).items():
        ctx_id, _, q = key.rpartition("__q")
        if ctx_id in ci_by_id:
            out[(ci_by_id[ctx_id], int(q))] = text
    return out


def _fetch_r_lookup_from_hf(dest: Path, unreduced_prefix: str) -> None:
    """Stage the cell's persisted r_lookup.json from the HF data repo (resume path)."""
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(DATA_REPO, f"{unreduced_prefix}/{R_LOOKUP_NAME}", repo_type="dataset")
    shutil.copyfile(cached, dest)


def _save_accum_ckpt(
    path: Path,
    *,
    flat_rows: dict[str, list[np.ndarray]],
    pq_ctx_idx: list[int],
    pq_q_idx: list[int],
    row_keys: list[str],
    n_cells_done: int,
    n_empty: int,
    cell_bytes: list[int],
    behavior: str,
    substrate: str,
) -> None:
    """Overwrite-in-place (atomic) local checkpoint of the reduced accumulators.

    Written at every upload flush so a restart can rebuild per_q / pq_rows for
    the rows already on HF WITHOUT recomputing their forwards (loaded by
    ``_load_accum_ckpt`` when its row keys exactly match the HF skip-set).
    Local-only — NEVER uploaded (rebuildable, and grows to ~GBs on a full cell).
    """
    arrays = {f"flat_{k}": np.stack(flat_rows[k]).astype(np.float32) for k in _PQ_KEYS}
    tmp = path.with_suffix(".tmp.npz")
    # Plain savez: deflate on fp32 activations runs ~8 MiB/s at ~1.08x ratio, and this
    # full-cumulative rewrite fires at EVERY flush — compressed, it re-introduces the
    # exact serial-deflate stall the per-row savez fix removes (review blocker).
    np.savez(
        tmp,
        **arrays,
        pq_ctx_idx=np.asarray(pq_ctx_idx, dtype=np.int64),
        pq_q_idx=np.asarray(pq_q_idx, dtype=np.int64),
        row_keys=np.asarray(row_keys),
        n_cells_done=np.asarray(n_cells_done),
        n_empty=np.asarray(n_empty),
        cell_bytes=np.asarray(cell_bytes, dtype=np.int64),
        behavior=np.asarray(behavior),
        substrate=np.asarray(substrate),
        git_sha=np.asarray(_git_sha()),
    )
    os.replace(tmp, path)


def _load_accum_ckpt(
    path: Path,
    *,
    behavior: str,
    substrate: str,
    n_contexts: int,
    n_questions: int,
    expected_keys: set[str],
) -> dict | None:
    """Load the accumulator checkpoint iff it EXACTLY covers the HF skip-set.

    Returns the seeded accumulator state (per_q / pq_rows / flat_rows / index
    lists / counters) when the checkpoint's (behavior, substrate) match and its
    row keys == ``expected_keys`` — the caller may then skip forward+reduce for
    exactly those rows. Any mismatch → None (the conservative recompute-by-
    forward path). Set equality is stronger than the row-count match: it
    guarantees every skipped row is present in the loaded accumulators.
    """
    with np.load(path) as z:
        if str(z["behavior"]) != behavior or str(z["substrate"]) != substrate:
            return None
        row_keys = [str(k) for k in z["row_keys"]]
        if len(row_keys) != len(expected_keys) or set(row_keys) != expected_keys:
            return None
        n = len(row_keys)
        flat_rows: dict[str, list[np.ndarray]] = {}
        for k in _PQ_KEYS:
            arr = z[f"flat_{k}"]
            assert arr.shape == (n, N_LAYERS, HIDDEN), (k, arr.shape)
            flat_rows[k] = list(arr)
        pq_ctx_idx = [int(i) for i in z["pq_ctx_idx"]]
        pq_q_idx = [int(i) for i in z["pq_q_idx"]]
        n_cells_done = int(z["n_cells_done"])
        cell_bytes = [int(b) for b in z["cell_bytes"]]
    if any(ci >= n_contexts for ci in pq_ctx_idx):
        return None  # a smaller --max-contexts run than the checkpoint — recompute
    if any(qi >= n_questions for qi in pq_q_idx):
        # a smaller --max-questions run than the checkpoint would silently seed
        # out-of-grid question rows into the summary averages (review concern C3)
        return None
    per_q: dict[int, dict] = {ci: {k: [] for k in _PQ_KEYS} for ci in range(n_contexts)}
    for i, ci in enumerate(pq_ctx_idx):
        for k in _PQ_KEYS:
            per_q[ci][k].append(flat_rows[k][i])
    pq_rows = {k: [flat_rows[k][i][HEADLINE_LAYER] for i in range(n)] for k in _PQ_KEYS}
    return {
        "per_q": per_q,
        "pq_rows": pq_rows,
        "flat_rows": flat_rows,
        "pq_ctx_idx": pq_ctx_idx,
        "pq_q_idx": pq_q_idx,
        "row_keys": row_keys,
        "n_cells_done": n_cells_done,
        "cell_bytes": cell_bytes,
    }


def _prepare_resume_state(args, behavior: str, substrate: str, tmp_dir: Path) -> dict:
    """Resume bookkeeping for a non-gate-only UPLOAD run (empty state otherwise).

    Returns ``{hf_keys, r_lookup_on_hf, pending_init, local_keys, keep_local}``:
    ONE list_repo_files call builds the already-uploaded skip-set, and complete
    local .npz files a prior interrupted process saved-but-never-flushed are
    re-enqueued for the first flush. Non-upload / gate-only runs get empty state
    (nothing was ever uploaded to resume against).
    """
    state: dict = {
        "hf_keys": set(),
        "r_lookup_on_hf": False,
        "pending_init": [],
        "local_keys": set(),
        "keep_local": set(),
    }
    if args.gate_only or not args.upload:
        return state
    hf_keys, r_on_hf = _hf_uploaded_rows(behavior, substrate)
    logger.info("resume: %d rows already on HF, skipping their unreduced saves", len(hf_keys))
    pending_init = _reenqueue_local_npz(tmp_dir, hf_keys, _unreduced_prefix(behavior, substrate))
    if pending_init:
        logger.info(
            "resume: re-enqueued %d complete local .npz (saved but never flushed by a prior run)",
            len(pending_init),
        )
    state.update(
        hf_keys=hf_keys,
        r_lookup_on_hf=r_on_hf,
        pending_init=pending_init,
        local_keys={p.stem for p, _ in pending_init},
    )
    return state


def _phase_a_base_responses(
    args, tok, contexts, questions, *, device, max_new, tmp_dir: Path, resume: dict
) -> dict[tuple[int, int], str]:
    """Phase A: frozen base greedy R for every (context, question) pair (resume-aware).

    Loads the cell's persisted r_lookup.json (locally, else from HF when a prior
    upload run pushed it) and vLLM-generates ONLY uncovered pairs — restarts
    reuse the SAME R (removes vLLM nondeterminism). Persists the (possibly
    extended) lookup right after generation and enqueues it into ``resume``'s
    pending_init for the first upload flush (kept local post-flush). On the
    CPU-smoke path uncovered pairs stay absent (per-pair HF greedy fallback in
    ``_extract_pairs``). gate-only keeps the pre-patch behavior: generate all,
    persist nothing.
    """
    r_lookup: dict[tuple[int, int], str] = {}
    r_lookup_path = tmp_dir / R_LOOKUP_NAME
    prefix = _unreduced_prefix(args.behavior, args.substrate)
    if not args.gate_only:
        if not r_lookup_path.exists() and resume["r_lookup_on_hf"]:
            _fetch_r_lookup_from_hf(r_lookup_path, prefix)
        if r_lookup_path.exists():
            r_lookup = _load_r_lookup_json(r_lookup_path, contexts)
            logger.info(
                "resume: r_lookup.json covers %d pairs — Phase A skipped for them", len(r_lookup)
            )
    pair_msgs: list[list[dict]] = []
    pair_keys: list[tuple[int, int]] = []
    for ci, inst in enumerate(contexts):
        for qi, q in enumerate(questions):
            pair_msgs.append(i594.messages_for_instance(inst, q))
            pair_keys.append((ci, qi))
    fresh_r = False
    if device.type != "cpu":
        missing = [(k, m) for k, m in zip(pair_keys, pair_msgs, strict=True) if k not in r_lookup]
        if missing:
            logger.info(
                "[phase=extract] Phase A: vLLM-generating %d base R responses", len(missing)
            )
            responses = ex667.vllm_generate_R(tok, [m for _, m in missing], max_new_tokens=max_new)
            r_lookup.update(zip((k for k, _ in missing), responses, strict=True))
            fresh_r = True
    if r_lookup and not args.gate_only:
        _write_r_lookup_json(r_lookup_path, r_lookup, contexts)
        if args.upload and (fresh_r or not resume["r_lookup_on_hf"]):
            resume["pending_init"].append((r_lookup_path, f"{prefix}/{R_LOOKUP_NAME}"))
            resume["keep_local"].add(r_lookup_path)
    return r_lookup


def _flush_and_checkpoint(
    pending: list[tuple[Path, str]],
    keep_local: frozenset[Path],
    ckpt_path: Path | None,
    st: dict,
    *,
    behavior: str,
    substrate: str,
) -> None:
    """Flush the pending batch (ONE HF commit), delete locals, checkpoint accumulators.

    No-op on an empty buffer. Locals in ``keep_local`` (r_lookup.json) survive the
    post-commit delete. After the commit every appended row is on HF, so the
    local-only accumulator checkpoint written here (``ckpt_path``, overwrite in
    place, NEVER uploaded) lets a restart skip those rows' forwards entirely.
    """
    if not pending:
        return
    _hf_batch_commit(pending)  # ONE commit for the whole chunk (fail-loud)
    for local_path, _ in pending:
        if local_path not in keep_local:
            local_path.unlink()  # DELETE local only AFTER a verified batch commit
    pending.clear()
    if ckpt_path is not None and st["row_keys"]:
        _save_accum_ckpt(
            ckpt_path,
            flat_rows=st["flat_rows"],
            pq_ctx_idx=st["pq_ctx_idx"],
            pq_q_idx=st["pq_q_idx"],
            row_keys=st["row_keys"],
            n_cells_done=st["n_cells_done"],
            n_empty=st["n_empty"],
            cell_bytes=st["cell_bytes"],
            behavior=behavior,
            substrate=substrate,
        )


def _init_accumulators(contexts, acc_init: dict | None) -> dict:
    """Fresh — or accum_ckpt-seeded (resume) — accumulator state for ``_extract_pairs``."""
    if acc_init is not None:
        return acc_init
    return {
        "per_q": {ci: {k: [] for k in _PQ_KEYS} for ci in range(len(contexts))},
        "pq_rows": {k: [] for k in _PQ_KEYS},
        "flat_rows": {k: [] for k in _PQ_KEYS},
        "pq_ctx_idx": [],
        "pq_q_idx": [],
        "row_keys": [],
        "n_cells_done": 0,
        "cell_bytes": [],
    }


def _enforce_disk_floor(upload: bool, flush, n_cells_done: int) -> None:
    """Fail loud when /workspace free drops under the 10 GiB floor (EDQUOT risk).

    Flushes the pending buffer first so a real batch-upload lag (not just
    unflushed local files) is what trips the floor — fail-loud otherwise.
    """
    free = _df_free_gib()  # EDQUOT / df fail-loud monitoring
    if free is None or free >= 10.0:
        return
    if upload:
        flush()
        free = _df_free_gib()
    if free is not None and free < 10.0:
        raise RuntimeError(
            f"disk free {free:.1f} GiB < 10 GiB floor at /workspace after "
            f"{n_cells_done} cells — batch-upload-then-delete not keeping up "
            "(EDQUOT risk)"
        )


def _extract_pairs(
    base,
    trained,
    tok,
    contexts,
    questions,
    r_lookup,
    *,
    behavior,
    substrate,
    tmp_dir: Path,
    device,
    max_new,
    upload,
    skip_save_keys: frozenset[str] = frozenset(),
    skip_forward_keys: frozenset[str] = frozenset(),
    acc_init: dict | None = None,
    pending_init: list[tuple[Path, str]] | tuple = (),
    keep_local: frozenset[Path] = frozenset(),
    ckpt_path: Path | None = None,
) -> dict:
    """Teacher-force every (context, question) pair; stream-upload unreduced; accumulate reduced.

    Returns the accumulators ``_run_cell_body`` reduces into the per-(behavior,
    substrate) summary: ``per_q`` (question-averaged 28-layer c_C/v_A per
    context), the flat headline-layer per-question rows for the substrate-swap
    null, and the counters (``n_cells_done`` / ``n_empty`` / ``cell_bytes``).
    Split out of the cell body so that function stays under the ruff C901 cap.

    Resume semantics (all inert on a fresh launch — every skip/init defaults empty):

    - ``skip_forward_keys``: rows whose reduced reads were pre-seeded via
      ``acc_init`` (the accum_ckpt) — skipped ENTIRELY (no forward).
    - ``skip_save_keys``: rows already on HF (or re-enqueued from a prior
      interrupted run) — forward + reduced accumulation still run, but the
      unreduced .npz save AND its upload enqueue are skipped.
    - ``pending_init``: pre-existing (local_path, path_in_repo) items uploaded
      in the first flush; ``keep_local`` paths (r_lookup.json) survive the
      post-flush delete.
    - ``ckpt_path``: when set, every flush also overwrites the local-only
      reduced-accumulator checkpoint (``_save_accum_ckpt``).
    """
    st = _init_accumulators(contexts, acc_init)
    st["n_empty"] = 0  # recounted every run (empty-R rows are never skipped/persisted)
    per_q: dict[int, dict] = st["per_q"]
    pq_rows: dict[str, list] = st["pq_rows"]
    flat_rows: dict[str, list] = st["flat_rows"]  # ckpt store, append order
    pq_ctx_idx: list[int] = st["pq_ctx_idx"]
    pq_q_idx: list[int] = st["pq_q_idx"]
    row_keys: list[str] = st["row_keys"]
    cell_bytes: list[int] = st["cell_bytes"]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    unreduced_prefix = _unreduced_prefix(behavior, substrate)
    # Pending (local_path, path_in_repo) buffer for the batched upload (B3). Flushed
    # every BATCH_UPLOAD_CHUNK files (ONE HF commit per flush), local deleted per flush.
    pending: list[tuple[Path, str]] = list(pending_init)

    def _flush_pending() -> None:
        _flush_and_checkpoint(
            pending, keep_local, ckpt_path, st, behavior=behavior, substrate=substrate
        )

    for ci, inst in enumerate(contexts):
        ctx_id = inst["id"]
        for qi, q in enumerate(questions):
            row_key = _row_key(ctx_id, qi)
            if row_key in skip_forward_keys:
                continue  # accum_ckpt already carries this row's reduced reads (resume)
            r = r_lookup.get((ci, qi))
            if r is None:  # CPU-smoke path (no vLLM)
                r = ex667._greedy_response(
                    base, tok, i594.messages_for_instance(inst, q), device, max_new
                )
            if not r.strip():
                st["n_empty"] += 1
                continue
            msgs = i594.messages_for_instance(inst, q)
            caps = _capture_full_and_reduce(base, trained, tok, msgs, r, device)
            for k in _PQ_KEYS:
                per_q[ci][k].append(caps[k])  # question-average, all 28 layers (map inputs)
                pq_rows[k].append(caps[k][HEADLINE_LAYER])  # headline-layer null row
                flat_rows[k].append(caps[k])  # same objects, ckpt append order
            pq_ctx_idx.append(ci)
            pq_q_idx.append(qi)
            row_keys.append(row_key)
            if row_key not in skip_save_keys:
                # stream-upload the UNREDUCED per-(context, question) .npz, then delete
                # local. UNCOMPRESSED np.savez: deflate was ~65% of per-row wall (103.8s
                # vs 1.2s on a 933MB row) at a 1.29x ratio — the small cell-end REDUCED
                # saves stay compressed.
                tmp_npz = tmp_dir / f"{row_key}.npz"
                np.savez(
                    tmp_npz,
                    full_base=caps["full_base"],
                    full_trained=caps["full_trained"],
                    prompt_len=np.asarray(caps["prompt_len"]),
                    full_len=np.asarray(caps["full_len"]),
                    behavior=np.asarray(behavior),
                    substrate=np.asarray(substrate),
                    context_id=np.asarray(ctx_id),
                    question_index=np.asarray(qi),
                    layers=np.asarray(list(range(N_LAYERS))),
                    git_sha=np.asarray(_git_sha()),
                )
                cell_bytes.append(tmp_npz.stat().st_size)
                if upload:
                    pending.append((tmp_npz, f"{unreduced_prefix}/{row_key}.npz"))
                    if len(pending) >= BATCH_UPLOAD_CHUNK:
                        _flush_pending()  # ONE commit per chunk, then delete-local (B3)
            st["n_cells_done"] += 1
            _enforce_disk_floor(upload, _flush_pending, st["n_cells_done"])
    if upload:
        _flush_pending()  # final partial chunk (ONE commit), then delete-local
    return {
        "per_q": per_q,
        "pq_rows": pq_rows,
        "pq_ctx_idx": pq_ctx_idx,
        "pq_q_idx": pq_q_idx,
        "n_cells_done": st["n_cells_done"],
        "n_empty": st["n_empty"],
        "cell_bytes": cell_bytes,
    }


# ── One-cell driver ────────────────────────────────────────────────────────────


def run_cell(args) -> dict:
    """Extract + stream-upload one (behavior, substrate) cell; return the phase-1 metrics.

    Thin wrapper around ``_run_cell_body``: the ``.done`` resume-skip, then the
    per-cell PID lockfile (a duplicate exec of a cell owned by a LIVE process
    WAITS for the owner via ``_wait_for_cell_owner``, then exits 0 iff the
    owner wrote ``.done`` / raises otherwise — so the dispatcher's later
    duplicate wave-2 exec of a manually pre-launched cell neither double-runs
    NOR lets the dispatcher advance to the fit phase mid-extraction), then the
    body with a finally-guaranteed lock release. gate-only bypasses both
    guards (it writes no production state).
    """
    behavior = args.behavior
    substrate = args.substrate
    out_root = Path(args.out_root)
    reduced_dir = out_root / "reduced" / behavior / substrate
    reduced_dir.mkdir(parents=True, exist_ok=True)
    sentinel = reduced_dir / CELL_DONE_SENTINEL
    # gate-only never resume-skips (it measures bytes, writes no .done) and never
    # trusts a production .done as "done" — it always runs its one-cell measurement.
    if sentinel.exists() and not args.force and not args.gate_only:
        logger.info(
            "[phase=extract] %s/%s already complete (%s) — skip", behavior, substrate, sentinel
        )
        return {"skipped": True, "behavior": behavior, "substrate": substrate}
    if args.gate_only:
        return _run_cell_body(args, reduced_dir)
    owner = _acquire_cell_lock(reduced_dir)
    if owner is not None:
        # Duplicate exec: WAIT for the live owner (never an instant exit-0), then
        # exit 0 on .done / fail loud without it. Never touches the owner's lock.
        return _wait_for_cell_owner(reduced_dir, owner, behavior=behavior, substrate=substrate)
    try:
        return _run_cell_body(args, reduced_dir)
    finally:
        (reduced_dir / RUNNING_LOCK_NAME).unlink(missing_ok=True)


def _run_cell_body(args, reduced_dir: Path) -> dict:
    """One-cell extraction body (behind ``run_cell``'s sentinel + PID-lock wrapper)."""
    behavior = args.behavior
    substrate = args.substrate
    device = ex667._device(args.gpu_id, args.cpu_only)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    max_new = MAX_NEW_TOKENS[behavior]
    out_root = Path(args.out_root)
    sentinel = reduced_dir / CELL_DONE_SENTINEL

    contexts = load_battery_instances(max_contexts=args.max_contexts)
    questions = substrate_questions(behavior, substrate, max_questions=args.max_questions)
    logger.info(
        "[phase=extract] cell behavior=%s substrate=%s | %d contexts × %d questions × 2 models",
        behavior,
        substrate,
        len(contexts),
        len(questions),
    )

    # ── Stage + gauge the adapter BEFORE any GPU work (cheap, HALT early) ──
    adapter_dir = ex667.stage_adapter_local(behavior, "default", SEED)
    gauge = ex667.assert_adapter_gauge(adapter_dir, behavior)
    logger.info(
        "[phase=extract] adapter gauge OK: %s",
        {k: gauge[k] for k in ("r", "lora_alpha", "use_rslora")},
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))

    # ── Resume state: HF skip-set + local re-enqueue (inert on a fresh launch) ──
    tmp_dir = _unreduced_tmp_dir(out_root, behavior, substrate)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    resume = _prepare_resume_state(args, behavior, substrate, tmp_dir)

    # ── Phase A: vLLM batched frozen base R (resume-aware via r_lookup.json) ──
    # On CPU-smoke (no vLLM) uncovered pairs fall back to per-pair HF greedy
    # (ex667._greedy_response) inside _extract_pairs.
    r_lookup = _phase_a_base_responses(
        args,
        tok,
        contexts,
        questions,
        device=device,
        max_new=max_new,
        tmp_dir=tmp_dir,
        resume=resume,
    )

    # ── Accumulator checkpoint (load side): when it EXACTLY covers the HF skip-set,
    # the skipped rows' forward+reduce is skipped too (recompute-by-forward otherwise).
    ckpt_path = None if args.gate_only else tmp_dir / ACCUM_CKPT_NAME
    acc_init = None
    skip_forward: frozenset[str] = frozenset()
    if ckpt_path is not None and resume["hf_keys"] and ckpt_path.exists():
        acc_init = _load_accum_ckpt(
            ckpt_path,
            behavior=behavior,
            substrate=substrate,
            n_contexts=len(contexts),
            n_questions=len(questions),
            expected_keys=resume["hf_keys"],
        )
        if acc_init is not None:
            skip_forward = frozenset(resume["hf_keys"])
            logger.info(
                "resume: accum_ckpt covers all %d HF rows — skipping their forward+reduce",
                len(skip_forward),
            )
        else:
            logger.info("resume: accum_ckpt stale/mismatched — recomputing by forward")

    # ── Phase B: load base θ0 + trained θ⁺ for the teacher-force reads ──
    _, base, trained = ex667.load_base_and_trained(adapter_dir, device, dtype)

    # ── Teacher-force every pair; stream-upload unreduced; accumulate reduced ──
    acc = _extract_pairs(
        base,
        trained,
        tok,
        contexts,
        questions,
        r_lookup,
        behavior=behavior,
        substrate=substrate,
        tmp_dir=tmp_dir,
        device=device,
        max_new=max_new,
        upload=args.upload,
        skip_save_keys=frozenset(resume["hf_keys"] | resume["local_keys"]),
        skip_forward_keys=skip_forward,
        acc_init=acc_init,
        pending_init=resume["pending_init"],
        keep_local=frozenset(resume["keep_local"]),
        ckpt_path=ckpt_path,
    )
    per_q = acc["per_q"]
    pq_rows = acc["pq_rows"]
    pq_ctx_idx = acc["pq_ctx_idx"]
    pq_q_idx = acc["pq_q_idx"]
    n_cells_done = acc["n_cells_done"]
    n_empty = acc["n_empty"]
    cell_bytes = acc["cell_bytes"]

    # ── GATE-ONLY early return (B1): measure per-cell bytes, write NOTHING into the ──
    # production reduced/ tree, and RETURN before the <4-contexts fit guard + the
    # reduced-summary / per-question / .done sentinel writes. So the one-cell gate can
    # run against the production OUT_ROOT (or an isolated temp root) without either
    # crashing the sweep on the <4-contexts guard OR planting a .done that the un-forced
    # Phase-2 sweep would skip (shipping a 1×1 fixture). It touches no reduced/ artifact.
    if args.gate_only:
        gate_metrics = {
            "behavior": behavior,
            "substrate": substrate,
            "gate_only": True,
            "n_contexts": len(contexts),
            "n_questions": len(questions),
            "n_unreduced_cells": n_cells_done,
            "n_empty_R": n_empty,
            "mean_cell_bytes": (float(np.mean(cell_bytes)) if cell_bytes else 0.0),
            "df_free_gib_workspace": _df_free_gib(),
        }
        logger.info(
            "[phase=one_cell_gate] GATE-ONLY %s/%s: %d unreduced cells, mean %.1f MB/cell "
            "(no reduced/summary/.done written)",
            behavior,
            substrate,
            n_cells_done,
            gate_metrics["mean_cell_bytes"] / 2**20,
        )
        return gate_metrics

    # ── Phase C: question-average c_C + v_A over the substrate's questions ──
    def _qavg(ci: int, key: str) -> np.ndarray:
        rows = per_q[ci][key]
        if not rows:
            raise RuntimeError(f"context {ci} has zero non-empty questions for {key}")
        return np.stack(rows).mean(axis=0).astype(np.float32)

    ctx_ids = [inst["id"] for inst in contexts]
    families = [inst["family"] for inst in contexts]
    kept = [ci for ci in range(len(contexts)) if per_q[ci]["c_C_base"]]
    if len(kept) < 4:
        raise RuntimeError(
            f"{behavior}/{substrate}: only {len(kept)} contexts with usable questions (<4) — "
            "cannot fit a map (all base R empty?)"
        )
    c_C_base = np.stack([_qavg(ci, "c_C_base") for ci in kept])  # (n_kept, 28, HIDDEN)
    c_C_trained = np.stack([_qavg(ci, "c_C_trained") for ci in kept])
    v_A_base = np.stack([_qavg(ci, "v_A_base") for ci in kept])
    v_A_trained = np.stack([_qavg(ci, "v_A_trained") for ci in kept])

    reduced_path = reduced_dir / "summary.npz"
    np.savez_compressed(
        reduced_path,
        c_C_base=c_C_base,  # (n_ctx, 28, HIDDEN) fp32
        c_C_trained=c_C_trained,
        v_A_base=v_A_base,
        v_A_trained=v_A_trained,
        context_ids=np.asarray([ctx_ids[ci] for ci in kept], dtype=object),
        families=np.asarray([families[ci] for ci in kept], dtype=object),
        n_contexts=np.asarray(len(kept)),
        n_questions=np.asarray(len(questions)),
        behavior=np.asarray(behavior),
        substrate=np.asarray(substrate),
        layers=np.asarray(list(range(N_LAYERS))),
        git_sha=np.asarray(_git_sha()),
        generated_at=np.asarray(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    )
    if args.upload:
        _hf_upload_file(
            reduced_path, f"{EXPERIMENT_NAME}/reduced/{behavior}/{substrate}/summary.npz"
        )

    # ── Per-question headline-layer (L14) rows for the substrate-swap null ──
    # Flat rows + parallel context/question indices + per-context family (the
    # null resamples questions WITHIN this substrate, re-splits into matched-n
    # pseudo-substrates, question-averages each per context → two pseudo maps).
    pq_path = reduced_dir / f"per_question_L{HEADLINE_LAYER}.npz"
    np.savez_compressed(
        pq_path,
        c_C_base=np.stack(pq_rows["c_C_base"]).astype(np.float32),  # (n_rows, HIDDEN)
        c_C_trained=np.stack(pq_rows["c_C_trained"]).astype(np.float32),
        v_A_base=np.stack(pq_rows["v_A_base"]).astype(np.float32),
        v_A_trained=np.stack(pq_rows["v_A_trained"]).astype(np.float32),
        row_context_index=np.asarray(pq_ctx_idx, dtype=np.int64),  # original ctx index per row
        row_question_index=np.asarray(pq_q_idx, dtype=np.int64),
        context_ids=np.asarray(ctx_ids, dtype=object),  # full-length (indexed by original ci)
        families=np.asarray(families, dtype=object),  # full-length (indexed by original ci)
        headline_layer=np.asarray(HEADLINE_LAYER),
        behavior=np.asarray(behavior),
        substrate=np.asarray(substrate),
        git_sha=np.asarray(_git_sha()),
    )
    if args.upload:
        _hf_upload_file(
            pq_path,
            f"{EXPERIMENT_NAME}/reduced/{behavior}/{substrate}/per_question_L{HEADLINE_LAYER}.npz",
        )

    # atomic completion sentinel (resume-skip predicate). The PID lock is released
    # ONLY by run_cell's finally, AFTER this sentinel lands — an early unlink here
    # would open a window where a duplicate exec sees no sentinel and no lock and
    # re-runs the cell concurrently with these final writes (review concern C1).
    tmp_s = sentinel.with_suffix(f".{os.getpid()}.tmp")
    tmp_s.write_text(
        json.dumps(
            {
                "behavior": behavior,
                "substrate": substrate,
                "n_contexts": len(kept),
                "n_questions": len(questions),
                "n_unreduced_cells": n_cells_done,
                "n_empty_R": n_empty,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    os.replace(tmp_s, sentinel)

    metrics = {
        "behavior": behavior,
        "substrate": substrate,
        "n_contexts": len(kept),
        "n_questions": len(questions),
        "n_unreduced_cells": n_cells_done,
        "n_empty_R": n_empty,
        "n_rows_skipped_from_hf": len(resume["hf_keys"]),
        "mean_cell_bytes": (float(np.mean(cell_bytes)) if cell_bytes else 0.0),
        "reduced_path": str(reduced_path),
        "df_free_gib_workspace": _df_free_gib(),
    }
    logger.info(
        "[phase=extract] cell %s/%s DONE: %d contexts, %d unreduced cells, mean %.1f MB/cell",
        behavior,
        substrate,
        len(kept),
        n_cells_done,
        metrics["mean_cell_bytes"] / 2**20,
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Issue #813 — one (behavior, substrate) map-fit extraction cell"
    )
    ap.add_argument("--behavior", required=True, choices=list(BEHAVIORS))
    ap.add_argument("--substrate", required=True, choices=list(SUBSTRATES))
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "eval_results/issue_813")
    ap.add_argument(
        "--gpu-id", type=int, default=0, help="physical GPU (CVD-pinned by the launcher)"
    )
    ap.add_argument("--cpu-only", action="store_true", help="CPU smoke (HF greedy R, no vLLM)")
    ap.add_argument("--upload", action="store_true", help="stream unreduced+reduced .npz to HF")
    ap.add_argument(
        "--force",
        action="store_true",
        help=(
            "ignore the resume-skip sentinel. NOTE: with --upload the HF skip-set + "
            "accum-ckpt resume still applies (rows on HF are replayed, not recomputed) — "
            "a post-bug-fix forced RECOMPUTE must also clear the cell's HF prefix/ckpt"
        ),
    )
    ap.add_argument(
        "--gate-only",
        action="store_true",
        help=(
            "one-cell footprint/wall GATE: extract + measure per-cell bytes ONLY, write the "
            "metrics JSON, and RETURN before any reduced-summary / per-question / .done "
            "sentinel write. Writes NOTHING into --out-root's reduced/ tree, so it cannot "
            "corrupt the production sweep (B1). Bypasses the <4-contexts fit guard."
        ),
    )
    ap.add_argument("--max-contexts", type=int, default=None, help="smoke: cap battery contexts")
    ap.add_argument(
        "--max-questions", type=int, default=None, help="smoke: cap substrate questions"
    )
    ap.add_argument(
        "--metrics-out", type=Path, default=None, help="write the phase-1 metrics JSON here"
    )
    return ap


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    metrics = run_cell(args)
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2, default=float))
    # NO [phase=done] here — this is a per-cell SUBPROCESS whose stdout inherits the
    # dispatcher's; the poller reserves [phase=done] for the ONE terminal line in the
    # main dispatcher log (issue813_dispatch.sh), so a per-cell echo of it would trip
    # the #545 false-`done` while the sweep is still alive.
    logger.info(
        "run_cell %s/%s complete; metrics: %s",
        args.behavior,
        args.substrate,
        json.dumps(metrics, default=float),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
