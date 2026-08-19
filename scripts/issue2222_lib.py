"""Shared constants + helpers for issue #2222 (Persona Vectors screening predictors).

Unit-1 surface: P0 staging (dataset.zip pin, fixed per-dataset subsample,
reused-artifact staging) plus the P1/P2 driver's manifest / fingerprint /
upload contract. Units 2/3 (P3 reduction + fits, P5 figures) import the same
constants and manifest helpers — put NEW P3+ code in NEW files: this file and
``issue2222_capture.py`` are inside :func:`code_fingerprint`, so editing them
after production capture invalidates every per-dataset resume fingerprint BY
DESIGN (plan v5 §9 resume provenance — a fingerprint mismatch re-runs the
dataset rather than silently reusing stale cells).

Position identity (plan v5 §4): ``ctxend`` is the last token of the FULL
chat-templated prompt (``apply_chat_template(messages[:-1],
add_generation_prompt=True)``) — identical to the paper's ``prompt_last``
position AND to #1739's ``context_end`` map-training input; ``pfxend`` is the
last token before the user turn (#1739 ``prefix_end`` convention via
``render_prompt_parts``); the answer summary is the #1739 ``t1`` answer-span
mean (== the paper's response-avg ``a_l``).

CONTENT HYGIENE: dataset.zip rows include harmful-content families — every
helper here logs ids / counts / hashes ONLY, never row text.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:  # sibling-script imports under -c / module mode (#823)
    sys.path.insert(0, str(_SCRIPTS_DIR))

# --- Pins (plan v5 §2/§10; re-verified live at unit-1 time, 2026-08-10) ------
ZIP_URL = "https://github.com/safety-research/persona_vectors/raw/main/dataset.zip"
# Git BLOB sha1 of dataset.zip (the upstream single-blob history pin, plan A10):
# sha1(b"blob %d\0" % len(data) + data). NOT a sha256 of the file bytes.
ZIP_GIT_BLOB_SHA1 = "4ae5f890b19e098cb51ab3ad55478343ba751588"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue2222_pvscreen"
RB_V2_HUB_PREFIX = "issue778_persona_vectors/analysis_tensors_v2/rb_v2"
# rb v1 fallback firing is a NAMED deviation (plan §8/§10 rb-fallback line).
RB_V1_HUB_PREFIX = "issue778_persona_vectors/analysis_tensors/rb"
MAP_HUB_PREFIX = "issue1739_ctxmap/analysis_tensors/maps"
MAP_FILES = ("context_end__ufull.npz", "prefix_end__ufull.npz")
MAP_KEYS = "w,x_mu,x_sd,y_mu,layers"
TRAITS = ("evil", "sycophancy", "hallucination")
RB_SHAPE = (28, 3584)
MAP_W_SHAPE = (28, 3584, 3584)
MAP_MU_SHAPE = (28, 1, 3584)

SUBSAMPLE_SEED = 42
SUBSAMPLE_ROWS = 1000
GEN_TEMPERATURE = 1.0
GEN_MAX_NEW_TOKENS = 1024
CAP_HIT_MAX_FRACTION = 0.02  # #1332 re-gen trigger + plan §7 exact-ΔP halt bar
# Hard floor on subsample fill: a systematic admission-rejection rate (>10% of
# the target) means the token budget is wrong for the corpus — fail loud, never
# silently shrink the fixed subsample (marker-mix budget-floor pattern, #906).
SUBSAMPLE_FILL_FLOOR = 0.9

# Plan-time leg-A dispositions (plan v5 §10 parent-lineage; re-asserted at P0).
# Short-sha prefixes of branch-side commits DECLARED NOT-NEEDED; any commit on
# origin/main..origin/<branch> touching the path that is NOT covered here is a
# fresh branch-side edit needing a new disposition (artifact-reuse check (k)).
DECLARED_LEG_A: dict[tuple[str, str], tuple[str, ...]] = {
    ("issue-778", "scripts/issue778_lib.py"): ("3e1e1220d9", "3c295cd75e"),
    ("issue-1739", "scripts/issue1739_fits.py"): ("f43eafe562",),
    ("issue-1739", "src/explore_persona_space/experiments/issue_1739/capture.py"): (),
    ("issue-1739", "src/explore_persona_space/experiments/issue_1739/generation.py"): (),
    ("issue-1739", "src/explore_persona_space/experiments/issue_1739/store_io.py"): (),
    ("issue-825", "scripts/issue825_fit_cells.py"): ("0e580958c6",),
    ("issue-778", "src/explore_persona_space/analysis/mapping_baselines.py"): (),
    ("issue-1739", "src/explore_persona_space/analysis/mapping_baselines.py"): (),
    ("issue-778", "src/explore_persona_space/eval/batch_judge.py"): (),
    ("issue-1739", "src/explore_persona_space/eval/batch_judge.py"): (),
}


def default_data_root() -> Path:
    """Canonical per-issue data root (repo-relative; gitignored, pod-rebuildable)."""
    return REPO_ROOT / "data" / "issue_2222"


def families_versions() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(FAMILIES, VERSIONS) from the canonical #778 constants (lazy import)."""
    import issue778_lib as i778

    return i778.FAMILIES, i778.VERSIONS


def split_dataset_id(ds: str) -> tuple[str, str]:
    """``{family}_{version}`` -> (family, version) via #778's suffix-safe parser."""
    import issue778_lib as i778

    return i778.split_cell_tag(ds)


def dataset_ids(selector: list[str] | None) -> list[str]:
    """Expand a --datasets selector (family names and/or full ids) to dataset ids.

    None/empty -> all 24. A family token expands to its 3 versions (the plan's
    smoke shape ``--datasets evil``); a full ``{family}_{version}`` id passes
    through. Unknown tokens fail loud.
    """
    families, versions = families_versions()
    all_ids = [f"{fam}_{ver}" for fam in families for ver in versions]
    if not selector:
        return all_ids
    out: list[str] = []
    for token in selector:
        if token in families:
            out.extend(f"{token}_{ver}" for ver in versions)
        elif token in all_ids:
            out.append(token)
        else:
            raise ValueError(f"unknown dataset selector {token!r}; families={families}")
    return out


def dataset_file(data_root: Path, ds: str) -> Path:
    """Path of one extracted dataset JSONL (zip layout: dataset/<family>/<version>.jsonl)."""
    family, version = split_dataset_id(ds)
    return Path(data_root) / "dataset" / family / f"{version}.jsonl"


# --- Hashes / atomic IO -------------------------------------------------------


def git_blob_sha1(path: Path) -> str:
    """Git blob sha1 of a file (sha1 over ``blob <len>\\0<bytes>``) — the A10 pin domain."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """sha256 hex of a UTF-8 string (prompt identity pins in the gen JSONL)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, obj: dict) -> None:
    """Atomic JSON write (tmp + os.replace; parents created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    """Text-mode line iteration (never ``splitlines`` — the #950 U+2028 shred)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_metadata() -> dict:
    """Reproducibility block: git commit + dirty flag + env versions + timestamp."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    def _v(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:  # noqa: BLE001 — version probe only, never load-bearing
            return "?"

    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {m: _v(m) for m in ("torch", "transformers", "vllm", "numpy")},
    }
    meta.update(as_metadata_dict(git_provenance(cwd=REPO_ROOT)))
    return meta


def log_phase(phase: str, msg: str = "", **extra) -> None:
    """poll_pipeline-parseable ``[phase=<name>] {json}`` line (flushed).

    ``[phase=done]`` is RESERVED for the single graceful-completion marker.
    """
    payload = {"phase": phase, "msg": msg, **extra}
    print(f"[phase={phase}] {json.dumps(payload)}", flush=True)


# --- dataset.zip staging (P0; self-buildable on a fresh pod — gitignored data
# --- does NOT travel with the branch clone, gotchas.md #654) ------------------


def stage_dataset_zip(data_root: Path) -> dict:
    """Idempotently stage + pin-verify dataset.zip and its 24 extracted files.

    Downloads from GitHub when absent, verifies the git-blob sha1 against the
    plan A10 pin (fail loud on mismatch), extracts, and asserts all 24
    ``dataset/<family>/<version>.jsonl`` files exist. Returns a digest
    (counts + sha) — never row text.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "dataset.zip"
    if not zip_path.exists():
        log_phase("p0_zip", "downloading dataset.zip", url=ZIP_URL)
        tmp = zip_path.with_name(zip_path.name + ".tmp")
        with urllib.request.urlopen(ZIP_URL, timeout=120) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, zip_path)
    realized_sha = git_blob_sha1(zip_path)
    if realized_sha != ZIP_GIT_BLOB_SHA1:
        raise RuntimeError(
            f"dataset.zip git-blob sha1 mismatch: realized {realized_sha} != pinned "
            f"{ZIP_GIT_BLOB_SHA1} (plan A10) — refusing to stage an unpinned corpus"
        )
    families, versions = families_versions()
    expected = [dataset_file(data_root, f"{fam}_{ver}") for fam in families for ver in versions]
    if not all(p.exists() for p in expected):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_root)
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(f"dataset.zip extraction incomplete — missing: {missing}")
    return {
        "zip_git_blob_sha1": realized_sha,
        "n_dataset_files": len(expected),
        "extract_root": str(data_root / "dataset"),
    }


def load_dataset_rows(data_root: Path, ds: str) -> list[tuple[str, str]]:
    """All (question, answer) pairs of one dataset file, strict A9 schema.

    Each row must be ``{"messages": [{user}, {assistant}]}`` — anything else
    fails loud naming dataset + row index (never row text).
    """
    path = dataset_file(data_root, ds)
    if not path.exists():
        raise FileNotFoundError(f"dataset file missing (run stage_dataset_zip first): {path}")
    out: list[tuple[str, str]] = []
    for i, row in enumerate(read_jsonl(path)):
        msgs = row.get("messages")
        if (
            not isinstance(msgs, list)
            or len(msgs) != 2
            or msgs[0].get("role") != "user"
            or msgs[1].get("role") != "assistant"
        ):
            raise ValueError(f"{ds} row {i}: unexpected schema (plan A9 expects [user, assistant])")
        out.append((str(msgs[0]["content"]), str(msgs[1]["content"])))
    return out


# --- Prompt rendering + admission (shared by P0 subsample and P1/P2 capture) --


def render_row(tokenizer, question: str) -> tuple[str, str]:
    """(prefix_text, prompt_text) for one bare user question.

    ``apply_chat_template([user], add_generation_prompt=True)`` via #1739's
    ``render_prompt_parts`` — the prompt's last token is the paper's
    ``prompt_last`` == #1739 ``context_end`` (plan §4 position identity), and
    the prefix (everything before the user turn — here Qwen's auto-inserted
    default system block) matches the frozen map's ``prefix_end`` convention.
    """
    from explore_persona_space.experiments.issue_1739 import generation as gen1739

    return gen1739.render_prompt_parts(tokenizer, [{"role": "user", "content": question}])


def admit_row(tokenizer, prefix: str, prompt: str, completion: str, row_label: str) -> dict | None:
    """Token-budget admission via the EXACT capture-path position builder.

    Returns the position dict on admit, None on an over-budget row (the same
    ``ValueError`` the capture would raise — one admission filter, zero drift
    between P0 and P1/P2; #1738 subsample-bypass class). Empty completions are
    the caller's separate reject class.
    """
    from explore_persona_space.experiments.issue_1739.capture import capture_row_ids_and_positions

    try:
        _, pos = capture_row_ids_and_positions(
            tokenizer, prefix, prompt, completion, row_label=row_label
        )
    except ValueError:
        return None
    return pos


def build_subsample(data_root: Path, ds: str, tokenizer, *, seed: int, s_rows: int) -> dict:
    """Fixed per-dataset subsample manifest (plan §4 P0): seed-42 draw, S rows.

    Walks a seeded full permutation of the dataset's row indices, admitting
    rows through the capture-path token-budget filter until ``s_rows`` are
    admitted (or the file is exhausted). Deterministic given (file bytes,
    seed, s_rows, admission code). Rejections are counted per class; a fill
    below ``SUBSAMPLE_FILL_FLOOR`` of the target fails loud.
    """
    rows = load_dataset_rows(data_root, ds)
    order = random.Random(seed).sample(range(len(rows)), len(rows))
    target = min(s_rows, len(rows))
    admitted: list[int] = []
    n_scanned = n_rej_budget = n_rej_empty = 0
    for idx in order:
        if len(admitted) >= target:
            break
        n_scanned += 1
        question, answer = rows[idx]
        if not answer.strip():
            n_rej_empty += 1
            continue
        prefix, prompt = render_row(tokenizer, question)
        if admit_row(tokenizer, prefix, prompt, answer, row_label=f"{ds}:{idx}") is None:
            n_rej_budget += 1
            continue
        admitted.append(idx)
    if len(admitted) < SUBSAMPLE_FILL_FLOOR * target:
        raise RuntimeError(
            f"{ds}: subsample under-filled — admitted {len(admitted)}/{target} after scanning "
            f"{n_scanned}/{len(rows)} rows (budget rejects {n_rej_budget}, empty {n_rej_empty}); "
            "the token budget is systematically too small for this corpus — raise it "
            "deliberately, never silently shrink the fixed subsample"
        )
    return {
        "dataset": ds,
        "seed": seed,
        "s_target": s_rows,
        "n_admitted": len(admitted),
        "n_file_rows": len(rows),
        "n_scanned": n_scanned,
        "n_rejected_budget": n_rej_budget,
        "n_rejected_empty": n_rej_empty,
        "row_ids": admitted,
        "split_hash": hashlib.sha256(json.dumps(admitted).encode()).hexdigest(),
        "dataset_file_sha256": sha256_file(dataset_file(data_root, ds)),
        "code_fingerprint": code_fingerprint(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def subsample_manifest_path(data_root: Path, ds: str, seed: int, s_rows: int) -> Path:
    """Per-(dataset, seed, S) subsample manifest path (smoke and production coexist)."""
    return Path(data_root) / "subsample" / f"{ds}__seed{seed}_S{s_rows}.json"


def ensure_subsample(
    data_root: Path, ds: str, tokenizer, *, seed: int, s_rows: int
) -> tuple[dict, list[tuple[int, str, str]]]:
    """(manifest, [(row_id, question, answer), ...]) — building/refreshing as needed.

    A manifest whose dataset-file sha or code fingerprint no longer matches is
    REBUILT (deterministic — same inputs reproduce the same row ids), never
    silently reused (plan §9: bare existence never vouches).
    """
    data_root = Path(data_root)
    path = subsample_manifest_path(data_root, ds, seed, s_rows)
    manifest: dict | None = None
    if path.exists():
        manifest = json.loads(path.read_text())
        stale = (
            manifest.get("dataset_file_sha256") != sha256_file(dataset_file(data_root, ds))
            or manifest.get("code_fingerprint") != code_fingerprint()
        )
        if stale:
            log_phase("p0_subsample", "stale manifest — rebuilding", dataset=ds)
            manifest = None
    if manifest is None:
        manifest = build_subsample(data_root, ds, tokenizer, seed=seed, s_rows=s_rows)
        write_json_atomic(path, manifest)
    rows = load_dataset_rows(data_root, ds)
    selected = [(i, rows[i][0], rows[i][1]) for i in manifest["row_ids"]]
    return manifest, selected


# --- Fingerprints (plan §9 resume provenance — the #952 gate-5 manifest shape) -


def code_fingerprint() -> str:
    """sha256 over the output-affecting code files (capture driver + this lib +
    the reused #1739 capture/generation modules). A stand-in for the plan's
    "code SHA of the capture/generation script tree": stable across unrelated
    repo commits, invalidated by any edit to the code that shapes outputs."""
    files = (
        _SCRIPTS_DIR / "issue2222_lib.py",
        _SCRIPTS_DIR / "issue2222_capture.py",
        REPO_ROOT / "src/explore_persona_space/experiments/issue_1739/capture.py",
        REPO_ROOT / "src/explore_persona_space/experiments/issue_1739/generation.py",
    )
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0")
        h.update(f.read_bytes() if f.exists() else b"<absent>")
        h.update(b"\0")
    return h.hexdigest()


def run_config(seed: int, s_rows: int, batch_size: int) -> dict:
    """Every output-affecting run constant (the plan's config fingerprint set)."""
    from explore_persona_space.experiments.issue_1739 import capture as cap1739
    from explore_persona_space.experiments.issue_1739 import generation as gen1739
    from explore_persona_space.experiments.issue_1739.constants import MODEL_NAME

    return {
        "model": MODEL_NAME,
        "revision": gen1739.INSTRUCT_REVISION,
        "gen": {
            "temperature": GEN_TEMPERATURE,
            "max_new_tokens": GEN_MAX_NEW_TOKENS,
            "n": 1,
            "engine_max_model_len": gen1739.MAX_MODEL_LEN,
            "dtype": "bfloat16",
            "seed_scheme": "issue1739.generation._context_seed(seed, '<ds>:<row_id>')",
            "cap_hit_regen": {"threshold": CAP_HIT_MAX_FRACTION, "regen_max_tokens_mult": 2},
        },
        "capture": {
            "boundary": cap1739.BOUNDARY_INSTRUCT,
            "max_model_len": gen1739.MAX_MODEL_LEN,
            "max_formatted_tokens": cap1739.MAX_FORMATTED_TOKENS,
            "batch_size": batch_size,
            "kinds": ["raw_respavg", "ctxend", "pfxend", "base_respavg"],
            "position_convention": "issue1739.capture.capture_row_ids_and_positions",
        },
        "subsample": {"seed": seed, "s_rows": s_rows},
    }


def config_fingerprint(cfg: dict) -> str:
    """sha256 over the sorted-JSON run config."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def resume_fingerprint(split_hash: str, cfg_fp: str, dataset_file_sha256: str) -> str:
    """The per-dataset resume key: split hash + corpus bytes + config + code.

    ``dataset_file_sha256`` pins the CORPUS the row ids index into — split ids
    alone are identical across equal-length files (all three versions of a
    family share n_rows), and a re-pinned corpus must never silently reuse
    stale captures.
    """
    payload = f"{split_hash}|{dataset_file_sha256}|{cfg_fp}|{code_fingerprint()}"
    return hashlib.sha256(payload.encode()).hexdigest()


# --- Per-dataset output paths (local + HF) ------------------------------------


def capture_dir(data_root: Path, ds: str) -> Path:
    """Local per-dataset capture dir (plan §9 phase_outputs P1/P2)."""
    return Path(data_root) / "capture" / ds


def rawcomp_path(data_root: Path, ds: str) -> Path:
    """Local base-generation rollout-text JSONL (plan §6.5 deliverable)."""
    return Path(data_root) / "raw_completions" / "exact_dp_base_gen" / f"{ds}.jsonl"


def hf_capture_prefix(ds: str) -> str:
    """HF prefix for one dataset's capture artifacts (plan §10 destinations)."""
    return f"{HF_PREFIX}/analysis_tensors/capture/{ds}"


def hf_rawcomp_path(ds: str) -> str:
    """HF destination of one dataset's base-generation JSONL."""
    return f"{HF_PREFIX}/raw_completions/exact_dp_base_gen/{ds}.jsonl"


def upload_file(local: Path, path_in_repo: str, *, attempts: int = 3) -> None:
    """Fail-loud single-file HF upload with a bounded outer transport retry.

    ``hub._upload(raise_on_error=True)`` re-raises upload exceptions; an empty
    return (its fail-soft early-outs) is ALSO treated as failure. The outer
    30/60 s retry is the #1315 seam envelope — each attempt re-enters hub's
    own inner retry budget; exhaustion raises (per-cell upload is part of the
    #664 durability contract, never warn-and-continue).
    """
    from explore_persona_space.orchestrate import hub

    last_err: Exception | None = None
    # Bounded transport RETRY (<=3 attempts) around ONE file — the #1315
    # outer-retry seam, not a per-file fan-out (per-dataset uploads total
    # ~4 files/dataset x 24 datasets, far under per-file-storm scale).
    for attempt in range(attempts):
        try:
            # UPLOAD_LOOP_EXEMPT: bounded <=3-attempt retry around ONE file (#1315 seam)
            url = hub._upload(
                Path(local),
                HF_DATA_REPO,
                "dataset",
                path_in_repo,
                upload_as_file=True,
                raise_on_error=True,
            )
            if url:
                return
            last_err = RuntimeError(f"upload returned no path for {path_in_repo}")
        except Exception as e:  # noqa: BLE001 — bounded retry then fail-loud re-raise below
            last_err = e
        if attempt < attempts - 1:
            wait = 30 * (attempt + 1)
            log_phase("upload_retry", str(last_err), path=path_in_repo, wait_s=wait)
            time.sleep(wait)
    raise RuntimeError(f"upload failed after {attempts} attempts: {path_in_repo}") from last_err


def hub_file_exists(path_in_repo: str) -> bool:
    """Retried single-path existence probe on the data repo (scoped; never a full listing)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    return bool(
        hub.retry_transient(
            # HUB_VERIFY_RETRY_EXEMPT: probe rides the hub.retry_transient wrap above
            lambda: HfApi().file_exists(HF_DATA_REPO, path_in_repo, repo_type="dataset"),
            what=f"file_exists({path_in_repo})",
        )
    )


def fetch_hub_manifest(ds: str) -> dict | None:
    """The dataset's HF-side manifest.json, or None when it does not exist yet."""
    import tempfile

    from huggingface_hub.errors import EntryNotFoundError

    from explore_persona_space.orchestrate import hub

    path_in_repo = f"{hf_capture_prefix(ds)}/manifest.json"
    with tempfile.TemporaryDirectory(prefix="i2222_manifest_") as td:
        try:
            local = hub.stage_hub_file(
                HF_DATA_REPO, path_in_repo, Path(td) / "manifest.json", repo_type="dataset"
            )
        except EntryNotFoundError:
            return None
        return json.loads(Path(local).read_text())


def write_results_sentinel(version: int, note: dict, *, logs_dir: Path) -> Path:
    """Poller-conforming end-of-run sentinel (delegates to the #778 writer)."""
    import issue778_lib as i778

    return i778.write_results_sentinel(2222, "epm:results", version, note, logs_dir=logs_dir)
