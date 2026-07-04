# ruff: noqa: RUF002, RUF003
"""Issue #958 shared constants, corpus/unit conventions, store IO, sentinel.

Multi-turn extension of the context→answer activation-mapping line (#779/#922).
Conventions (plan §4):

- A UNIT is ``(conversation c, turn k)``. Unit ids are strings:
  ``main:c<ci>:k<k>`` / ``long:c<ci>:k<k>`` / ``graft:c<ci>:k<k>:q<j>`` /
  ``onpol:c<ci>:k2``.
- The activation STORE holds one ``(5, R, H)`` fp16 tensor per unit — rows of
  the FIRST axis are the 5 capture positions ``POS_*`` below; the SECOND axis
  has ``R = 29`` residual rows (row 0 = embedding stream, rows 1..28 = decoder
  blocks 0..27 — the #922 convention, ``issue922_common.block_to_row``).
- ``answer_mean`` INCLUDES the trailing ``<|im_end|>`` (151645) + trailing
  ``\\n`` (198) — byte-level #779 parity (plan §4.2, fact-checked).
- Splits are BY CONVERSATION (``issue922_common.make_split``, seed 42,
  4000/500/500 main; 480/60/60 long) — the #810 group-fold rule.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

if Path("/workspace").is_dir():  # pod/GCE lanes only; never redirect the VM cache
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue922_common as C922  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logger = logging.getLogger("issue958_common")

# ── constants (plan §4 / §10) ─────────────────────────────────────────────────

DEFAULT_MODEL = C922.DEFAULT_MODEL  # Qwen/Qwen2.5-7B-Instruct
EXPECTED_LAYERS = C922.EXPECTED_LAYERS  # 28
EXPECTED_HIDDEN = C922.EXPECTED_HIDDEN  # 3584
IM_END_ID = C922.IM_END_ID  # 151645
NL_ID = C922.NL_ID  # 198
GENERATION_SUFFIX = C922.GENERATION_SUFFIX  # "<|im_start|>assistant\n"

HF_DATA_REPO = C922.HF_DATA_REPO
# Smoke dispatches redirect via EPM958_HF_PREFIX (issue958_multiturn/smoke) so
# smoke artifacts never mix into production paths (the #922 pattern).
HF_OUT_PREFIX = os.environ.get("EPM958_HF_PREFIX", "issue958_multiturn")

LMSYS_REPO = "lmsys/lmsys-chat-1m"

# Corpus (plan §4.1)
N_MAIN = 5000
N_LONG = 600
K_MAIN = 4
K_LONG = 8
TOKEN_CAP = 7168  # formatted input tokens per used turn (prefix+query+gen prompt)
CORPUS_SHUFFLE_SEED = 42

# Rollouts (plan §4.3 — Source #779 verbatim, issue779_collect.py:558)
ROLLOUT_TEMPERATURE = 1.0
ROLLOUT_TOP_P = 0.95
ROLLOUT_MAX_TOKENS = 1024
ROLLOUT_SEED = 42

# Aux unit sets (plan §4.5)
GRAFT_N_CONVS = 150  # main-panel TEST conversations
GRAFT_TURNS = (2, 4)
GRAFT_Q = 4  # grafted queries per (prefix, turn)
GRAFT_SEED = 0
GRAFT_Q_FLOOR = 2  # below this realized Q the prefix drops from the marginal read
ONPOL_N_CONVS = 200  # main-panel TEST conversations, k=2

# Splits (plan §10 — Source #922)
SPLIT_SEED = C922.SPLIT_SEED  # 42
N_FIT, N_VAL, N_TEST = C922.N_FIT, C922.N_VAL, C922.N_TEST  # 4000/500/500
LONG_FIT, LONG_VAL, LONG_TEST = 480, 60, 60
TWIN_SEED = 4242  # deterministic half-sample twin split of the fit fold

# Store
SHARD_UNITS = 500
N_POS = 5
POS_PREFIX_END = 0
POS_CTX_M1 = 1  # ctx_last − 1
POS_CTX_END = 2  # the #779/#922 "last prompt token"
POS_ANS_MEAN = 3  # mean over answer span INCL. <|im_end|> + \n (#779 parity)
POS_ANS_LAST = 4  # last answer-span position (the trailing \n)
POS_NAMES = ["prefix_end", "ctx_last_m1", "ctx_last", "answer_mean", "answer_last"]

# Read-out + stats (plan §10 — Source #922 / #778)
READOUT_BLOCKS = C922.READOUT_BLOCKS  # [14, 17, 19, 20, 24, 26]
PRIMARY_LSTAR = C922.PRIMARY_LSTAR  # evil 20 / sycophancy 26 / hallucination 17
TRAITS = C922.TRAITS
BOOTSTRAP_DRAWS = 997
BOOTSTRAP_SEED = 0
SHUFFLE_DRAWS = 100
SHUFFLE_SEED = 0
RANDDIR_DRAWS = 100
RANDDIR_SEED = 0

# Registered transfer standardization policy (plan §4.6)
TRANSFER_STANDARDIZATION_POLICY = "source-map-composite"

block_to_row = C922.block_to_row
row_to_block_key = C922.row_to_block_key
make_split = C922.make_split
reproducibility_metadata = C922.reproducibility_metadata
write_json_atomic = C922.write_json_atomic
upload_dir_bulk = C922.upload_dir_bulk


# ── corpus identity (the #958 r2 resume-collision fix) ───────────────────────


def canonical_sha256(payload: dict) -> str:
    """sha256 hex over canonical JSON (sorted keys, compact separators)."""
    import hashlib

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compute_corpus_fingerprint(payload: dict) -> str:
    """sha256 hex over a canonical-JSON corpus identity payload.

    The payload carries realized counts + seeds + per-conversation dedup
    hashes, so ANY corpus rebuild (different scale, different stream slice)
    changes the fingerprint. Threaded through: corpus ``manifest.json`` →
    rollout shard ``regime`` → store shard blobs / sidecar index → the fit
    cache + fit-resume manifest. A shard/split/store from a different corpus
    fingerprint fails loud instead of silently pairing stale generations with
    a rebuilt corpus (code-review r1 Critical, bug-class: resume predicate
    missing the corpus-identity key, #722-r3 class).
    """
    return canonical_sha256(payload)


def corpus_fingerprint(corpus_dir: Path) -> str:
    """The built corpus's fingerprint from ``manifest.json`` (fail-loud)."""
    manifest = json.loads((Path(corpus_dir) / "manifest.json").read_text())
    fp = manifest.get("corpus_fingerprint")
    assert fp, (
        f"corpus at {corpus_dir} has no corpus_fingerprint in manifest.json "
        "(pre-fingerprint build) — rebuild the corpus"
    )
    return str(fp)


# ── unit-id helpers ───────────────────────────────────────────────────────────


def unit_id(unit_set: str, ci: int, k: int, q: int | None = None) -> str:
    """Canonical unit-id string for a (set, conversation, turn[, graft index])."""
    base = f"{unit_set}:c{ci}:k{k}"
    return base if q is None else f"{base}:q{q}"


def enumerate_units(corpus_dir: Path) -> dict[str, list[dict]]:
    """All units per set from the corpus manifest — the ONE enumeration source.

    Every phase (rollouts, capture, fits, eval) derives its work from THIS
    function over the corpus the previous phase wrote (smoke = a smaller
    corpus; PASS_UNIFIED). Returns ``{set_name: [unit dict]}`` where a unit
    dict carries ``uid``, ``set``, ``ci``, ``k`` and (graft) ``q``/``donor_ci``.
    """
    main = load_corpus(corpus_dir, "main")
    long_p = load_corpus(corpus_dir, "long")
    graft_spec = json.loads((corpus_dir / "graftq_spec.json").read_text())
    onpol_spec = json.loads((corpus_dir / "onpol_spec.json").read_text())
    units: dict[str, list[dict]] = {"main": [], "long": [], "graft": [], "onpol": []}
    for ci in range(len(main)):
        for k in range(1, K_MAIN + 1):
            units["main"].append({"uid": unit_id("main", ci, k), "set": "main", "ci": ci, "k": k})
    for ci in range(len(long_p)):
        for k in range(1, K_LONG + 1):
            units["long"].append({"uid": unit_id("long", ci, k), "set": "long", "ci": ci, "k": k})
    for row in graft_spec["items"]:
        units["graft"].append(
            {
                "uid": unit_id("graft", row["ci"], row["k"], row["q"]),
                "set": "graft",
                "ci": row["ci"],
                "k": row["k"],
                "q": row["q"],
                "donor_ci": row["donor_ci"],
            }
        )
    for ci in onpol_spec["conv_indices"]:
        units["onpol"].append({"uid": unit_id("onpol", ci, 2), "set": "onpol", "ci": ci, "k": 2})
    return units


# ── corpus IO ─────────────────────────────────────────────────────────────────


def corpus_path(corpus_dir: Path, panel: str) -> Path:
    """Canonical corpus JSON path for a panel ('main' | 'long')."""
    return corpus_dir / f"{panel}.json"


def load_corpus(corpus_dir: Path, panel: str) -> list[dict]:
    """Load one panel's conversations; asserts ids are 0..n-1 in order."""
    with open(corpus_path(corpus_dir, panel)) as f:
        blob = json.load(f)
    convs = blob["conversations"]
    assert [c["ci"] for c in convs] == list(range(len(convs))), "corpus ci not contiguous"
    return convs


def conv_messages(conv: dict, upto_exchanges: int) -> list[dict]:
    """First ``upto_exchanges`` user/assistant exchanges as chat messages."""
    msgs = []
    for i, ex in enumerate(conv["exchanges"]):
        if i >= upto_exchanges:
            break
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    return msgs


# ── unit prompt construction (plan §4.2) ──────────────────────────────────────


def unit_prompt_messages(unit: dict, corpora: dict[str, list[dict]]) -> list[dict]:
    """Messages for context(c,k): prefix turns 1..k−1 (LMSYS originals) + user k.

    graft units substitute the DONOR conversation's turn-k user message under
    the host prefix; onpol units are handled by the rollout script (they need
    Qwen's own answer 1 as the prefix assistant turn).
    """
    us = unit["set"]
    if us == "graft":
        host = corpora["main"][unit["ci"]]
        donor = corpora["main"][unit["donor_ci"]]
        msgs = conv_messages(host, unit["k"] - 1)
        msgs.append({"role": "user", "content": donor["exchanges"][unit["k"] - 1]["user"]})
        return msgs
    panel = "long" if us == "long" else "main"
    conv = corpora[panel][unit["ci"]]
    msgs = conv_messages(conv, unit["k"] - 1)
    msgs.append({"role": "user", "content": conv["exchanges"][unit["k"] - 1]["user"]})
    return msgs


def onpol_prompt_messages(conv: dict, qwen_answer_1: str) -> list[dict]:
    """onpol_prefix control (§4.5): prefix′ = (user 1, Qwen's own answer 1) + query 2."""
    return [
        {"role": "user", "content": conv["exchanges"][0]["user"]},
        {"role": "assistant", "content": qwen_answer_1},
        {"role": "user", "content": conv["exchanges"][1]["user"]},
    ]


# ── rollout IO (per-shard persistence + resume; plan §4.3) ────────────────────


def rollout_shard_path(roll_dir: Path, unit_set: str, shard: int) -> Path:
    """Canonical rollout shard path."""
    return roll_dir / f"rollouts_{unit_set}_{shard:03d}.json"


def load_rollouts(roll_dir: Path, unit_set: str) -> dict[str, dict]:
    """All persisted rollouts of one unit set: {uid: {"text", "finish_reason"}}."""
    out: dict[str, dict] = {}
    for p in sorted(roll_dir.glob(f"rollouts_{unit_set}_*.json")):
        with open(p) as f:
            blob = json.load(f)
        out.update(blob["rollouts"])
    return out


def load_dropped(roll_dir: Path, unit_set: str) -> set[str]:
    """Union of per-shard ``dropped_empty`` uid lists (final-empty generations).

    A unit in this set was enumerated but produced no usable generation after
    the seed-varied retry — downstream consumers SKIP it with a recorded
    count; a store/rollout gap NOT in this set stays fail-loud.
    """
    out: set[str] = set()
    for p in sorted(roll_dir.glob(f"rollouts_{unit_set}_*.json")):
        with open(p) as f:
            blob = json.load(f)
        out.update(blob.get("dropped_empty", []))
    return out


# ── store IO ──────────────────────────────────────────────────────────────────


def store_shard_path(store_dir: Path, unit_set: str, shard: int) -> Path:
    """Canonical activation-store shard path."""
    return store_dir / unit_set / f"shard_{shard:03d}.pt"


def store_index_path(store_dir: Path, unit_set: str) -> Path:
    """Sidecar JSON index (uids per shard + corpus fingerprint) for one set."""
    return store_dir / unit_set / "index.json"


def _assert_shard_fingerprint(blob: dict, expect: str | None, where) -> None:
    """Fail loud when a store shard/index was captured under ANOTHER corpus."""
    if expect is None:
        return
    got = blob.get("corpus_fingerprint")
    assert got == expect, (
        f"STORE FINGERPRINT MISMATCH at {where}: shard/index built under corpus "
        f"{str(got)[:12]}… but the consumed corpus is {expect[:12]}… — stale artifacts "
        "from a different corpus build; recapture (or point at the matching dirs)."
    )


def load_store_index(
    store_dir: Path, unit_set: str, *, expect_fingerprint: str | None = None
) -> dict[str, tuple[int, str]]:
    """{uid: (shard index, path)} for one unit set — sidecar-first, metadata only.

    Prefers the capture-written ``index.json`` sidecar (no tensor loads — the
    r1-review "metadata only" fix); falls back to scanning shard blobs. When
    ``expect_fingerprint`` is given, the index/shards must carry it (fail loud).
    """
    sidecar = store_index_path(store_dir, unit_set)
    if sidecar.exists():
        blob = json.loads(sidecar.read_text())
        _assert_shard_fingerprint(blob, expect_fingerprint, sidecar)
        idx: dict[str, tuple[int, str]] = {}
        for s_str, uids in blob["shards"].items():
            p = store_shard_path(store_dir, unit_set, int(s_str))
            for uid in uids:
                idx[uid] = (int(s_str), str(p))
        return idx
    idx = {}
    for p in sorted((store_dir / unit_set).glob("shard_*.pt")):
        blob = torch.load(p, weights_only=False, map_location="cpu")
        _assert_shard_fingerprint(blob, expect_fingerprint, p)
        k = int(p.stem.split("_")[1])
        for uid in blob["units"]:
            idx[uid] = (k, str(p))
        del blob
    return idx


def load_store_positions(
    store_dir: Path,
    unit_set: str,
    uids: list[str],
    positions: list[int],
    *,
    expect_fingerprint: str | None = None,
) -> torch.Tensor:
    """Gather ``(n_uids, len(positions), R, H)`` fp16 from the shard files.

    Loads only the shards holding requested uids (one ≈0.5 GB blob resident
    at a time). Fails loud on any missing uid (row-coverage assert upstream)
    and on a corpus-fingerprint mismatch when ``expect_fingerprint`` is given.
    """
    want = set(uids)
    idx = load_store_index(store_dir, unit_set, expect_fingerprint=expect_fingerprint)
    missing = [u for u in uids if u not in idx]
    assert not missing, f"store missing {len(missing)} uids, e.g. {missing[:3]}"
    shard_paths = sorted({idx[u][1] for u in want})
    got: dict[str, torch.Tensor] = {}
    R = H = None
    for sp in shard_paths:
        blob = torch.load(sp, weights_only=False, map_location="cpu")
        _assert_shard_fingerprint(blob, expect_fingerprint, sp)
        for uid, rec in blob["units"].items():
            if uid in want:
                h = rec["h"]
                assert h.dtype == torch.float16 and h.dim() == 3, (uid, h.dtype, h.shape)
                R, H = int(h.shape[1]), int(h.shape[2])
                got[uid] = h[positions]  # (len(positions), R, H)
        del blob
    out = torch.empty((len(uids), len(positions), R, H), dtype=torch.float16)
    for i, uid in enumerate(uids):
        out[i] = got[uid]
    return out


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """EXACT sha256 over a file's FULL raw bytes (streamed, never torch.load).

    r4 (`fit-resume-store-content-digest-incomplete`): supersedes the sampled
    first/middle/last-window digest — a same-size byte change ANYWHERE in the
    file changes this hash, so a recapture with different fp16 activation
    values can never evade it. O(file bytes), paid ONCE at shard-write time
    (``write_store_sidecar``); mtime never enters.
    """
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def write_store_sidecar(
    store_dir: Path,
    unit_set: str,
    *,
    corpus_fingerprint: str,
    n_rows: int,
    hidden: int,
    shards: dict[str, list[str]],
) -> dict:
    """Write one set's ``index.json`` sidecar, hashing every shard AT WRITE time.

    The single production writer (capture stage + tests): records
    ``shard_sha256`` — a full-file :func:`file_sha256` per shard, computed
    from the on-disk bytes at sidecar-write time — so ANY (re)capture
    refreshes the recorded content identity and the fit regime's
    :func:`store_content_digest` invalidates resume exactly (r4). Returns
    the written blob.
    """
    blob = {
        "unit_set": unit_set,
        "corpus_fingerprint": corpus_fingerprint,
        "n_rows": n_rows,
        "hidden": hidden,
        "shards": shards,
        "shard_sha256": {
            s: file_sha256(store_shard_path(store_dir, unit_set, int(s))) for s in shards
        },
    }
    write_json_atomic(store_index_path(store_dir, unit_set), blob)
    return blob


def store_content_digest(store_dir: Path, unit_sets: list[str]) -> str:
    """EXACT content identity of the store shards a consumer reads.

    Canonical-JSON sha256 over ``{set: [[shard name, full-file sha256]]}``
    for the given unit sets — the fit-resume regime key (r3; r4 exact). The
    per-shard hashes come from the capture-written sidecar (recorded at
    shard-write time by :func:`write_store_sidecar`, so ANY recapture —
    same-size byte changes included — refreshes them and invalidates
    resume). Fail-loud paths, never silent trust: a sidecar whose
    ``shard_sha256`` does not cover every on-disk shard ASSERTS
    (sidecar/shard drift); a sidecar WITHOUT hashes (legacy pre-r4 store)
    RECOMPUTES the exact hash from shard bytes with a loud warning. Mtime
    never enters (an HF restage of identical bytes keeps the digest); a
    missing set dir contributes an empty list (a half-populated store then
    differs from any populated regime → refit-all, the conservative
    direction).
    """
    store_dir = Path(store_dir)
    payload: dict[str, list] = {}
    for s in unit_sets:
        d = store_dir / s
        shards = sorted(d.glob("shard_*.pt")) if d.is_dir() else []
        recorded: dict[str, str] = {}
        sidecar = store_index_path(store_dir, s)
        if shards and sidecar.exists():
            hashes = json.loads(sidecar.read_text()).get("shard_sha256")
            if hashes is not None:
                recorded = {
                    store_shard_path(store_dir, s, int(k)).name: str(v) for k, v in hashes.items()
                }
                missing = [p.name for p in shards if p.name not in recorded]
                assert not missing, (
                    f"store {s}: sidecar shard_sha256 missing entries for {missing} — "
                    "sidecar/shard drift; recapture the set (never silently trusted)"
                )
        if shards and not recorded:
            logger.warning(
                "[store-digest] %s sidecar lacks shard_sha256 (legacy pre-r4 store) — "
                "recomputing exact content hashes from shard bytes (O(store) read)",
                s,
            )
        payload[s] = [[p.name, recorded.get(p.name) or file_sha256(p)] for p in shards]
    return canonical_sha256(payload)


# ── device resolution (post-#763/#812: CLI > EPM_FIT_DEVICE > auto) ──────────


def resolve_device(requested: str | None) -> str:
    """'--device' > EPM_FIT_DEVICE env > auto (cuda if available else cpu)."""
    req = requested or os.environ.get("EPM_FIT_DEVICE", "auto")
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("[device] %s requested but CUDA unavailable — using cpu", req)
        return "cpu"
    return req


# ── half-sample twins (plan §4.6) ─────────────────────────────────────────────


def twin_halves(fit_idx: np.ndarray, seed: int = TWIN_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic A/B halves of the fit fold (matched N; equalize-down)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(fit_idx))
    half = len(fit_idx) // 2
    return fit_idx[perm[:half]], fit_idx[perm[half : 2 * half]]


# ── CPU-stage HF input staging (plan §9: the routed two-provision flow) ──────


def _is_transient_hf_error(exc: Exception) -> bool:
    """Retry classification for HF staging: 5xx / 429 / connection faults ONLY.

    Non-transient HTTP (4xx incl. storage-quota 403, auth 401, missing 404)
    stays LOUD — retrying an auth/quota failure just delays the crash by the
    full backoff budget (r3 minor: the prior retry-any-Exception loop hid
    4xx for ~3 minutes per file before surfacing it).
    """
    import requests
    from huggingface_hub.utils import HfHubHTTPError

    if isinstance(exc, HfHubHTTPError):
        code = exc.response.status_code if exc.response is not None else None
        return code is None or code >= 500 or code == 429
    return isinstance(
        exc,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    )


def _staged_ok(dst: Path, remote_size: int | None) -> bool:
    """Verify-before-skip predicate for a locally staged file (r3 minor).

    True iff ``dst`` is a regular file whose byte size equals the remote
    listing's — a verified partial staging from a prior interrupted run is
    KEPT; a size mismatch re-stages (with an explicit log, never a silent
    overwrite); an unknown remote size never skips.
    """
    return remote_size is not None and dst.is_file() and dst.stat().st_size == remote_size


def stage_inputs_from_hf(
    *, corpus_dir: Path, store_dir: Path, prefix: str | None = None, max_workers: int = 6
) -> dict:
    """Stage ``{prefix}/corpus`` + ``{prefix}/analysis_tensors/store`` locally.

    The fresh ``cpu-mid`` provision has no ``data/`` (gitignored) — enumerate
    with SERVER-side-scoped ``list_repo_tree`` (NEVER full-tree
    ``snapshot_download`` on the ~1M-file data repo, #833) and download
    per-file via ``hf_hub_download`` in a ≤``max_workers`` pool with retry +
    linear backoff, then move into place. Already-staged files are size-
    verified and SKIPPED (a mismatch re-stages with a log line); the retry
    covers transient 5xx/429/connection faults only — 4xx (quota 403, auth)
    raises immediately. Fails loud on an empty remote prefix or any file
    that fails after 4 attempts.
    """
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download

    pfx = prefix or HF_OUT_PREFIX
    api = HfApi()
    targets = {f"{pfx}/corpus": Path(corpus_dir), f"{pfx}/analysis_tensors/store": Path(store_dir)}
    n_files = 0
    for remote_prefix, local_root in targets.items():
        entries = [
            e
            for e in api.list_repo_tree(
                HF_DATA_REPO, path_in_repo=remote_prefix, repo_type="dataset", recursive=True
            )
            if getattr(e, "size", None) is not None  # files only (skip RepoFolder)
        ]
        assert entries, (
            f"HF staging: nothing under {HF_DATA_REPO}/{remote_prefix} — "
            "did the GPU stage upload complete?"
        )
        local_root.mkdir(parents=True, exist_ok=True)

        # verify-before-skip: keep size-verified files from a prior partial
        # staging; anything else (absent OR size-mismatched) is (re-)fetched.
        to_fetch = []
        for e in entries:
            dst = local_root / Path(e.path).relative_to(remote_prefix)
            if _staged_ok(dst, e.size):
                continue
            if dst.exists():
                logger.warning(
                    "[stage] %s exists with size %d != remote %d — re-staging",
                    dst,
                    dst.stat().st_size,
                    e.size,
                )
            to_fetch.append(e)
        if len(to_fetch) < len(entries):
            logger.info(
                "[stage] %d/%d files already staged (size-verified) — skip",
                len(entries) - len(to_fetch),
                len(entries),
            )

        def _fetch(path: str, staging_root: str) -> str:
            last: Exception | None = None
            for attempt in range(4):
                try:
                    return hf_hub_download(
                        repo_id=HF_DATA_REPO,
                        filename=path,
                        repo_type="dataset",
                        local_dir=staging_root,
                    )
                except Exception as exc:
                    if not _is_transient_hf_error(exc):
                        raise  # 4xx (quota 403 / auth 401 / 404) — loud, no retry
                    last = exc
                    logger.warning(
                        "[stage] %s failed (%s) attempt %d/4 — backoff",
                        path,
                        type(exc).__name__,
                        attempt + 1,
                    )
                    time.sleep(20 * (attempt + 1))
            raise RuntimeError(f"HF staging failed after 4 attempts: {path}") from last

        with tempfile.TemporaryDirectory(prefix="i958_hfstage_", dir=str(local_root)) as td:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                list(ex.map(lambda p: _fetch(p, td), [e.path for e in to_fetch]))
            for e in to_fetch:
                src = Path(td) / e.path
                dst = local_root / Path(e.path).relative_to(remote_prefix)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst.unlink()  # size-mismatched prior copy — replaced explicitly
                shutil.move(str(src), str(dst))
        n_files += len(entries)
        logger.info(
            "[stage] %s -> %s (%d files, %d fetched)",
            remote_prefix,
            local_root,
            len(entries),
            len(to_fetch),
        )
    return {"n_files": n_files, "prefix": pfx}


# ── uploads with the §7 timing gate (probe retried on transient 5xx) ─────────


def upload_with_timing_gate(local_dir: Path, suffix: str, msg: str) -> dict:
    """Timing-probe the LARGEST file, gate the projected wall, then bulk-upload.

    The in-run one-item serialization+upload timing gate (plan §7
    storage-overrun kill, timing branch): a tiny probe is per-commit-overhead
    dominated (#813) and would false-kill, so the kill arms only when the
    probe is ≥20 MB. The probe ``upload_file`` is retried up to 4× on
    transient 5xx / connection faults (#664 gotcha — single ``upload_file``
    against the ~1M-file repo 504s intermittently); 4xx (incl. quota 403)
    stays loud — ``upload_dir_bulk`` owns overflow routing.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    local_dir = Path(local_dir)
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    assert files, f"nothing to upload under {local_dir}"
    api = HfApi()
    probe = max(files, key=lambda p: p.stat().st_size)
    probe_wall = None
    for attempt in range(4):
        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=str(probe),
                path_in_repo=f"{HF_OUT_PREFIX}/{suffix}/{probe.relative_to(local_dir)}",
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                commit_message=f"{msg} (timing probe)",
            )
            probe_wall = time.time() - t0
            break
        except HfHubHTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code is not None and code < 500:
                raise  # 4xx incl. storage-quota 403 — loud; bulk path owns overflow
            if attempt == 3:
                raise RuntimeError("timing-probe upload failed after 4 transient retries") from e
            logger.warning("[upload-gate] probe transient %s — retry %d/4", code, attempt + 2)
            time.sleep(15 * (attempt + 1))
    assert probe_wall is not None
    probe_bytes = probe.stat().st_size
    total = sum(p.stat().st_size for p in files)
    if probe_bytes >= 20 * (1 << 20):  # throughput measurable, not overhead-dominated
        projected_h = (probe_wall / probe_bytes) * total / 3600
        logger.info(
            "[upload-gate] probe %.0f MB in %.1fs -> projected %.2fh for %.1f GB",
            probe_bytes / 1e6,
            probe_wall,
            projected_h,
            total / 1e9,
        )
        if projected_h > 4 * 0.5:
            raise RuntimeError(
                f"STORAGE-OVERRUN KILL (plan §7): projected upload {projected_h:.1f}h > "
                f"4x0.5h budget ({total / 1e9:.1f} GB). Artifacts kept local pod-side."
            )
    else:
        logger.info(
            "[upload-gate] probe %.0f KB (overhead-dominated) — gate N/A", probe_bytes / 1e3
        )
    return upload_dir_bulk(local_dir, f"{HF_OUT_PREFIX}/{suffix}", commit_message=msg)


# ── pod-side results sentinel (poll_pipeline contract) ────────────────────────


def write_results_sentinel(note: dict, *, kind: str = "epm:results", version: int = 1) -> Path:
    """End-of-run sentinel with poll_pipeline's required keys (issue 958)."""
    d = Path(os.environ.get("EPM958_SENTINEL_DIR", "/workspace/logs"))
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"issue-958-{kind.replace(':', '_')}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "task_id": 958,
        "by": "issue958_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": json.dumps(note, default=str),
    }
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(p)
    logger.info("[sentinel] wrote %s", p)
    return p
