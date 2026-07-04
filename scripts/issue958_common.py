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


# ── store IO ──────────────────────────────────────────────────────────────────


def store_shard_path(store_dir: Path, unit_set: str, shard: int) -> Path:
    """Canonical activation-store shard path."""
    return store_dir / unit_set / f"shard_{shard:03d}.pt"


def load_store_index(store_dir: Path, unit_set: str) -> dict[str, tuple[int, str]]:
    """{uid: (shard index, path)} over all shards of one unit set (metadata only)."""
    idx: dict[str, tuple[int, str]] = {}
    for p in sorted((store_dir / unit_set).glob("shard_*.pt")):
        blob = torch.load(p, weights_only=False, map_location="cpu")
        k = int(p.stem.split("_")[1])
        for uid in blob["units"]:
            idx[uid] = (k, str(p))
        del blob
    return idx


def load_store_positions(
    store_dir: Path, unit_set: str, uids: list[str], positions: list[int]
) -> torch.Tensor:
    """Gather ``(n_uids, len(positions), R, H)`` fp16 from the shard files.

    Loads each shard once (two-pass metadata is unnecessary at the (5, R, H)
    per-unit grain — one shard ≈ 500 × 5 × 29 × 3584 × 2 B ≈ 0.5 GB resident
    at a time). Fails loud on any missing uid (row-coverage assert upstream).
    """
    want = set(uids)
    got: dict[str, torch.Tensor] = {}
    R = H = None
    for p in sorted((store_dir / unit_set).glob("shard_*.pt")):
        blob = torch.load(p, weights_only=False, map_location="cpu")
        for uid, rec in blob["units"].items():
            if uid in want:
                h = rec["h"]
                assert h.dtype == torch.float16 and h.dim() == 3, (uid, h.dtype, h.shape)
                R, H = int(h.shape[1]), int(h.shape[2])
                got[uid] = h[positions]  # (len(positions), R, H)
        del blob
    missing = [u for u in uids if u not in got]
    assert not missing, f"store missing {len(missing)} uids, e.g. {missing[:3]}"
    out = torch.empty((len(uids), len(positions), R, H), dtype=torch.float16)
    for i, uid in enumerate(uids):
        out[i] = got[uid]
    return out


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
