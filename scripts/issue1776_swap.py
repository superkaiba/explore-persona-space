"""#1776 follow-up ``operator_swap_success`` — paper-style swap-success metric (plan v7).

For ~150 real-user context pairs (A, B) per two legs (LMSYS test-pool /
fresh-WildChat), invert each context->answer operator (fitted ridge M',
averaged causal Jacobian J_last) on the support-restricted subspace to get the
input-side edit delta_AB that should turn A's answer profile into B's, apply it
with the existing prefill ``DeltaHook`` (layer-14 last-context-token slot, the
representation-space intervention mechanism), and measure B-content acquisition
in free generations (judge-free MRR / recall@50 vs an eligibility-matched
shuffled-target null + a norm-matched random-delta control).

Subcommands mirror the p3p4 driver shape: ``stage / build / pilot / run /
merge-text / analyze / progress / final-sentinel / smoke-fixtures /
import-check``. Gates (plan §7): G-METRIC-SANITY (build, rc=9),
G-DOSE-DEGENERATE (build, rc=7), G-SWAP-PARITY (pilot, rc=8, BINDS at smoke),
G-PILOT (pilot, rc=7). ``--gates-informational`` demotes the two build gates to
log lines at smoke n (gate-calibration rule); parity + pilot stay binding.

Content hygiene: prompts/responses are real LMSYS/WildChat text — this module
NEVER prints or logs text fields; logs carry ids, counts, hashes, norms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps BEFORE numpy/torch import (#847 gate)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402
import issue1776_phase3 as P3  # noqa: E402
from issue1776_phase2_battery import _support_basis  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    DeltaHook,
    capture_vectors,
    coherence_check,
    condition_passes,
    context_token_ids,
    generate_batch,
    render_context,
)

FU = "followup_swap"
LABEL = "operator_swap_success"
OP_ARMS = ("swap_mprime", "swap_jlast")
STEER_ARMS = (*OP_ARMS, "swap_random")
BASELINE_ARM = "swap_a0"
ALL_ARMS = (BASELINE_ARM, *STEER_ARMS)
LEGS = ("lmsys", "wildchat")
LMSYS_SOURCE = "lmsys_test_pool"

# ── patch round (follow-up ``slot_patch_sufficiency``, plan v8) ───────────────
# ONE-variable diff vs the swap round: the injected edit REPLACES the layer-14
# last-context-token activation wholesale with v14_last(B) (DeltaHook
# ``replace=True``); pairs/targets/pools/nulls/metric reused verbatim.
FU_PATCH = "followup_slotpatch"
LABEL_PATCH = "slot_patch_sufficiency"
PATCH_BASELINE_ARM = "patch_a0"
PATCH_STEER_ARMS = ("swap_patch",)
PATCH_ALL_ARMS = (PATCH_BASELINE_ARM, *PATCH_STEER_ARMS)

# Round registry — every arm-/path-shaped difference between the two rounds
# threads through here; ``--round swap`` (the default) is byte-identical.
ROUNDS: dict[str, dict] = {
    "swap": {
        "label": LABEL,
        "fu": FU,
        "baseline_arm": BASELINE_ARM,
        "steer_arms": STEER_ARMS,
        "all_arms": ALL_ARMS,
        "merged_subdir": "steered_swap",
    },
    "patch": {
        "label": LABEL_PATCH,
        "fu": FU_PATCH,
        "baseline_arm": PATCH_BASELINE_ARM,
        "steer_arms": PATCH_STEER_ARMS,
        "all_arms": PATCH_ALL_ARMS,
        "merged_subdir": "steered_slotpatch",
    },
}

# §11 norm cap: the dose round's top operating norm (p4_alpha_ladder.json
# n_ref); asserted against the committed ladder copy at build time.
NORM_CAP_PLAN = 47.360563
LADDER_JSON = C76.PROJECT_ROOT / "eval_results/issue_1776/followup_p3p4/p4_alpha_ladder.json"

# §4 eligibility screen — ONE module-level predicate; the build sampler AND the
# analysis-side null-draw sampler both call (and identity-assert) THIS function.
JACCARD_MAX = 0.5
COS_MAX = 0.95
TARGET_FLOOR = 5
DF_MAX_FRAC = 0.10
TFIDF_TOP_K = 30
ELIGIBILITY = {
    "jaccard_max": JACCARD_MAX,
    "cos_max": COS_MAX,
    "target_floor": TARGET_FLOOR,
    "df_max_frac": DF_MAX_FRAC,
    "tfidf_top_k": TFIDF_TOP_K,
}
GLOBAL_SEED = 1776
RECALL_K = 50

WORD_RE = re.compile(r"\w+", re.UNICODE)
# CJK/Kana/Hangul intrusion audit ranges built from ordinals at runtime — never
# literal Unicode in source (Edit-tool \\uXXXX un-escape + NFC gotcha, #1364).
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF),
    (0x3040, 0x30FF),
    (0xAC00, 0xD7AF),
)
CJK_RE = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _CJK_RANGES) + "]")

# Reused #1776 production artifacts (plan §10), staged verbatim-mirror under
# --dest (dest/<repo path> == the parent hf_dl layout; #1774 mirror-root rule).
STAGE_FILES = (
    "analysis_tensors/comparator/m_ridge_x50k.pt",
    "analysis_tensors/jac_full/J_last.pt",
    "analysis_tensors/jpairs/jpair_capture.pt",
    "analysis_tensors/contexts/contexts.jsonl",
    "raw_completions/steered_dose/baseline_a0.json",
)

# Patch-round staged inputs (plan v8 §10): the swap round's OWN persisted build
# artifacts + the B-prompt sources. NO operators / jpair capture needed.
PATCH_STAGE_FILES = (
    f"analysis_tensors/{FU}/pairs.jsonl",
    f"analysis_tensors/{FU}/targets.json",
    f"analysis_tensors/{FU}/pool.pt",
    f"analysis_tensors/{FU}/deltas.pt",
    "raw_completions/steered_dose/baseline_a0.json",
)

# Referenced surface for the import-check axis (also pins ruff against
# stripping imports used only in later sections of this file).
_STEERING_SURFACE = (
    DeltaHook,
    generate_batch,
    capture_vectors,
    coherence_check,
    condition_passes,
    render_context,
    extract_layer_activations,
    _support_basis,
)


def words(text: str) -> list[str]:
    """Canonical model-tokenizer-agnostic word tokens (casefold + \\w+)."""
    return WORD_RE.findall(text.casefold())


def pair_eligible(jac: float, cos: float) -> bool:
    """THE §4 screen. Build sampling and the null-draw sampler share this
    exact function (the null sampler identity-asserts it, plan §4.2)."""
    return jac <= JACCARD_MAX and cos <= COS_MAX


def derive_seed(*parts) -> int:
    """Deterministic 31-bit seed derived from GLOBAL_SEED + string parts."""
    h = hashlib.sha256(":".join([str(GLOBAL_SEED), *map(str, parts)]).encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]


def _norm_cap(ladder_json: Path) -> float:
    """The §11 norm cap from the committed dose-round ladder (source of record)."""
    lad = json.loads(ladder_json.read_text())
    n_ref = float(lad["n_ref"])
    assert abs(n_ref - NORM_CAP_PLAN) < 1e-3, (n_ref, NORM_CAP_PLAN)
    return n_ref


# ── stage ─────────────────────────────────────────────────────────────────────


def _probe_staged_schemas(dest: Path, wc_names: list[str]) -> dict:
    """Fitness check (c): mmap key probes on EVERY consumed payload, BEFORE any
    consumer assert. Returns the probe digest (keys + shapes + shas)."""
    root = dest / C76.HF_PREFIX
    m = torch.load(
        root / "analysis_tensors/comparator/m_ridge_x50k.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,  # sha-pinned self-produced ridge payload (carve-out)
    )
    assert {"W", "xsd", "selected_lambda"} <= set(m.keys()), sorted(m.keys())
    j = torch.load(
        root / "analysis_tensors/jac_full/J_last.pt",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    assert "J" in j and j["J"].ndim == 2 and j["J"].shape[0] == j["J"].shape[1], (
        sorted(j.keys()),
        tuple(j["J"].shape),
    )
    cap = torch.load(
        root / "analysis_tensors/jpairs/jpair_capture.pt",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    assert "c14" in cap and cap["c14"].ndim == 2, sorted(cap.keys())
    base = json.loads((root / "raw_completions/steered_dose/baseline_a0.json").read_text())
    assert base.get("contexts"), sorted(base.keys())
    c0 = base["contexts"][0]
    assert {"context_id", "user", "samples"} <= set(c0.keys()), sorted(c0.keys())
    wc0 = torch.load(
        root / "wildchat_fresh/final_token_capture" / wc_names[0],
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    assert {"cx_last", "v_x", "layers", "ci"} <= set(wc0.keys()), sorted(wc0.keys())
    wc_layers = [int(x) for x in wc0["layers"]]
    return {
        "m_ridge_keys": sorted(m.keys()),
        "jlast_keys": sorted(j.keys()),
        "jlast_shape": list(j["J"].shape),
        "jpair_keys": sorted(cap.keys()),
        "jpair_c14_shape": list(cap["c14"].shape),
        "baseline_n_contexts": len(base["contexts"]),
        "wc_chunk_keys": sorted(wc0.keys()),
        "wc_layers": wc_layers,
        "m_ridge_sha": _sha256_file(root / "analysis_tensors/comparator/m_ridge_x50k.pt"),
        "jlast_sha": _sha256_file(root / "analysis_tensors/jac_full/J_last.pt"),
    }


def _probe_staged_schemas_patch(dest: Path, wc_names: list[str], source_layer: int) -> dict:
    """Patch-round fitness check (c): key/schema probes on EVERY consumed
    payload (staged swap build artifacts + B-prompt sources), BEFORE any
    consumer. Records — never hard-asserts — the assumption-1 layer-14
    membership of the stored WildChat capture (reference-unavailable path)."""
    root = dest / C76.HF_PREFIX
    art = root / f"analysis_tensors/{FU}"
    rows = [json.loads(ln) for ln in (art / "pairs.jsonl").read_text().split("\n") if ln.strip()]
    assert rows, "staged pairs.jsonl empty"
    need = {"pair_id", "leg", "a_id", "b_id", "included", "a_user", "a_idx", "b_idx"}
    assert need <= set(rows[0].keys()), sorted(rows[0].keys())
    n_included = sum(1 for r in rows if r["included"])
    targets = json.loads((art / "targets.json").read_text())
    assert targets["eligibility"] == ELIGIBILITY, "staged targets eligibility drift"
    assert targets["per_pair"], "staged targets per_pair empty"
    targets_sha = _sha256_file(art / "targets.json")
    pool = torch.load(art / "pool.pt", map_location="cpu", mmap=True, weights_only=False)
    for leg in LEGS:
        assert {"ids", "counts", "n_tokens", "prompt_words", "doc_freq", "v", "jaccard", "cos"} <= (
            set(pool[leg].keys())
        ), (leg, sorted(pool[leg].keys()))
    dl = torch.load(art / "deltas.pt", map_location="cpu", mmap=True, weights_only=True)
    assert {"pair_ids", "dv_target", "included"} <= set(dl.keys()), sorted(dl.keys())
    base = json.loads((root / "raw_completions/steered_dose/baseline_a0.json").read_text())
    assert base.get("contexts"), sorted(base.keys())
    c0 = base["contexts"][0]
    assert {"context_id", "user", "samples"} <= set(c0.keys()), sorted(c0.keys())
    wc0 = torch.load(
        root / "wildchat_fresh/final_token_capture" / wc_names[0],
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    assert {"cx_last", "v_x", "layers", "ci"} <= set(wc0.keys()), sorted(wc0.keys())
    wc_layers = [int(x) for x in wc0["layers"]]
    return {
        "n_pairs_rows": len(rows),
        "n_pairs_included": n_included,
        "pairs_sha": _sha256_file(art / "pairs.jsonl"),
        "targets_sha": targets_sha,
        "pool_sha": _sha256_file(art / "pool.pt"),
        "deltas_sha": _sha256_file(art / "deltas.pt"),
        "baseline_n_contexts": len(base["contexts"]),
        "wc_chunk_keys": sorted(wc0.keys()),
        "wc_layers": wc_layers,
        # assumption 1 (plan §12): recorded, not asserted — absence routes
        # G-CAPTURE-CONSISTENCY to its reference-unavailable branch.
        "source_layer_in_wc_layers": bool(source_layer in wc_layers),
    }


def cmd_stage(args) -> int:
    """Stage every swap-round input from the Hub at ONE fresh revision pin;
    idempotent per-file (existing targets skip); mmap key probes at the end.
    ``--round patch`` stages the PATCH round's file set instead (the swap
    round's persisted build artifacts + B-prompt sources)."""
    from explore_persona_space.orchestrate import hub

    rev = C76.resolve_data_repo_pin(args.pin_file, refresh=args.refresh_pin)
    staged: list[str] = []
    skipped = 0
    stage_files = PATCH_STAGE_FILES if args.round == "patch" else STAGE_FILES
    for rel in stage_files:
        repo_path = f"{C76.HF_PREFIX}/{rel}"
        target = args.dest / repo_path
        if target.is_file():
            skipped += 1
            continue
        hub.stage_hub_file(C76.HF_DATA_REPO, repo_path, target, repo_type="dataset", revision=rev)
        staged.append(repo_path)

    # wildchat_fresh capture chunks (.pt) + raw text (.json), stem-aligned.
    import issue779_ffc_n1m_generate_capture as N1G
    import fnmatch

    cap_prefix = f"{C76.HF_PREFIX}/wildchat_fresh/final_token_capture"
    raw_prefix = f"{C76.HF_PREFIX}/wildchat_fresh/raw_completions"
    remote = hub.retry_transient(
        lambda: N1G.N50._remote_index(cap_prefix), what=f"remote_index({cap_prefix})"
    )
    names = sorted(n for n in remote if fnmatch.fnmatch(n, "shard*_chunk*.pt"))
    assert names, f"no wildchat capture chunks under {cap_prefix}"
    if args.max_wc_chunks:
        names = names[: args.max_wc_chunks]
    n_wc = 0
    for n in names:
        for prefix, fname in ((cap_prefix, n), (raw_prefix, f"{Path(n).stem}.json")):
            target = args.dest / prefix / fname
            if target.is_file():
                continue
            hub.stage_hub_file(
                C76.HF_DATA_REPO, f"{prefix}/{fname}", target, repo_type="dataset", revision=rev
            )
            n_wc += 1

    if args.round == "patch":
        probes = _probe_staged_schemas_patch(args.dest, names, args.stage_source_layer)
        # byte-identity vs the committed copy (plan §10: staged == committed);
        # stage runs ONLY against real Hub artifacts, so this never sees fixtures.
        committed_targets = C76.PROJECT_ROOT / f"eval_results/issue_{C76.ISSUE}/{FU}/targets.json"
        if committed_targets.is_file():
            assert probes["targets_sha"] == _sha256_file(committed_targets), (
                "staged targets.json != committed copy — artifact drift (fitness check (j))"
            )
    else:
        probes = _probe_staged_schemas(args.dest, names)
    report = {
        "round": args.round,
        "revision": rev,
        "staged": staged,
        "skipped_existing": skipped,
        "wc_chunks": [Path(n).stem for n in names],
        "wc_files_staged": n_wc,
        "schema_probes": probes,
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.report, report)
    print(
        f"[swap-stage] [phase=stage_done] {len(staged)} staged, {skipped} present, "
        f"{len(names)} wc chunks, probes OK -> {args.report}",
        flush=True,
    )
    return 0


# ── leg loading (candidate pools) ─────────────────────────────────────────────


def _load_lmsys_candidates(dest: Path, tok) -> dict:
    """LMSYS leg: dose-round contexts with source == lmsys_test_pool; target
    text = the K=5 dose-round fresh baseline draws (text+tensor coherence:
    v is RECOMPUTED over these SAME draws by the build capture pass)."""
    base = json.loads(
        (dest / C76.HF_PREFIX / "raw_completions/steered_dose/baseline_a0.json").read_text()
    )
    rows = [c for c in base["contexts"] if c.get("source") == LMSYS_SOURCE]
    assert rows, f"no {LMSYS_SOURCE} contexts in baseline_a0.json"
    texts: list[list[str]] = []
    for c in rows:
        kept = [s for s in c["samples"] if s.strip()]
        assert kept, f"context {c['context_id']}: all baseline draws empty"
        texts.append(kept)
    return {
        "ids": [str(c["context_id"]) for c in rows],
        "users": [c["user"] for c in rows],
        "systems": [c.get("system") for c in rows],
        "texts": texts,
        "v": None,  # filled by the build capture pass
    }


def _load_wildchat_candidates(dest: Path, readout_layer: int) -> dict:
    """WildChat leg: 999 fresh rows — text = persisted on-policy answer,
    v = stored v_x at the readout layer (captured on that same answer)."""
    wc = dest / C76.HF_PREFIX / "wildchat_fresh"
    chunk_files = sorted((wc / "final_token_capture").glob("shard*_chunk*.pt"))
    assert chunk_files, f"no wildchat chunks staged under {wc / 'final_token_capture'}"
    ids, users, texts, vs = [], [], [], []
    for cf in chunk_files:
        d = torch.load(cf, map_location="cpu", weights_only=True)
        layers = [int(x) for x in d["layers"]]
        assert readout_layer in layers, (readout_layer, layers, cf.name)
        li = layers.index(readout_layer)
        raw = json.loads((wc / "raw_completions" / f"{cf.stem}.json").read_text())
        by_ci = {int(r["ci"]): r for r in raw["rows"]}
        for k, ci in enumerate(int(x) for x in d["ci"]):
            r = by_ci.get(ci)
            assert r is not None, f"wildchat ci={ci} in {cf.name} missing from raw json"
            if not str(r["response"]).strip():
                continue  # empty answer: no target text (counted upstream)
            ids.append(f"wc{ci:06d}")
            users.append(r["prompt"])
            texts.append([r["response"]])
            vs.append(d["v_x"][k, li, :].to(torch.float32))
    return {
        "ids": ids,
        "users": users,
        "systems": [None] * len(ids),
        "texts": texts,
        "v": torch.stack(vs),
    }


def _load_wildchat_cx14(dest: Path, source_layer: int) -> dict[str, torch.Tensor] | None:
    """Stored WildChat cx_last at the SOURCE layer, keyed by the wc context id —
    the G-CAPTURE-CONSISTENCY reference (plan v8 §7). Returns None when the
    stored capture does not carry the source layer (assumption-1 failure ->
    the gate's reference-unavailable branch)."""
    wc = dest / C76.HF_PREFIX / "wildchat_fresh/final_token_capture"
    chunk_files = sorted(wc.glob("shard*_chunk*.pt"))
    assert chunk_files, f"no wildchat chunks staged under {wc}"
    out: dict[str, torch.Tensor] = {}
    for cf in chunk_files:
        d = torch.load(cf, map_location="cpu", weights_only=True)
        layers = [int(x) for x in d["layers"]]
        if source_layer not in layers:
            return None
        li = layers.index(source_layer)
        for k, ci in enumerate(int(x) for x in d["ci"]):
            out[f"wc{ci:06d}"] = d["cx_last"][k, li, :].to(torch.float32)
    return out


# ── target-set construction (shared by build + analysis null draws) ──────────


def ctx_stats(texts: list[str]) -> tuple[dict[str, int], int]:
    """Word counts + total token count of one context's pooled target text."""
    counts: Counter[str] = Counter()
    total = 0
    for t in texts:
        ws = words(t)
        counts.update(ws)
        total += len(ws)
    return dict(counts), total


def build_target(
    counts_b: dict[str, int],
    n_tokens_b: int,
    doc_freq: dict[str, int],
    df_max_docs: int,
    n_docs: int,
    a_excl: set[str],
) -> list[str]:
    """T_B per §4.2: B's word types with DF <= df_max_docs, minus A's exclusion
    set, top TFIDF_TOP_K by tf-idf (ties broken lexically, deterministic).
    The FULL construction re-runs per null draw (selection rides per draw)."""
    assert n_tokens_b > 0, "empty target text"
    scored = []
    for w, cnt in counts_b.items():
        df = doc_freq.get(w, 0)
        assert df >= 1, (w, df)
        if df > df_max_docs or w in a_excl:
            continue
        tfidf = (cnt / n_tokens_b) * math.log(n_docs / df)
        scored.append((-tfidf, w))
    scored.sort()
    return [w for _, w in scored[:TFIDF_TOP_K]]


def rank_map(texts: list[str]) -> dict[str, int]:
    """Frequency-ranked word types over pooled draws (ties by first occurrence);
    rank is 1-based — the §4.2 per-cell scoring surface."""
    counts: Counter[str] = Counter()
    first: dict[str, int] = {}
    pos = 0
    for t in texts:
        for w in words(t):
            counts[w] += 1
            if w not in first:
                first[w] = pos
            pos += 1
    order = sorted(counts, key=lambda w: (-counts[w], first[w]))
    return {w: i + 1 for i, w in enumerate(order)}


def mrr_recall(rank: dict[str, int], targets: list[str]) -> tuple[float, float]:
    """(MRR, recall@50) of a target word list against a rank map. Absent word
    contributes 0 to MRR (plan §4.2); empty target -> (nan, nan)."""
    if not targets:
        return (float("nan"), float("nan"))
    mrr = sum(1.0 / rank[w] for w in targets if w in rank) / len(targets)
    rec = sum(1 for w in targets if rank.get(w, 10**9) <= RECALL_K) / len(targets)
    return (mrr, rec)


# ── leg pool assembly (counts/DF/v/jaccard/cos over the FULL candidate set) ───


def build_leg_pool(cand: dict) -> dict:
    """Per-leg pool: per-context stats + prompt words + DF + pairwise jaccard/cos
    matrices (vectorized incidence GEMM — no per-pair python loops)."""
    n = len(cand["ids"])
    assert n >= 2, n
    counts_l: list[dict[str, int]] = []
    ntok_l: list[int] = []
    for texts in cand["texts"]:
        cnts, tot = ctx_stats(texts)
        assert cnts, "context with zero word types"
        counts_l.append(cnts)
        ntok_l.append(tot)
    prompt_words = [sorted(set(words(u))) for u in cand["users"]]
    doc_freq: Counter[str] = Counter()
    for cnts in counts_l:
        doc_freq.update(cnts.keys())
    df_max_docs = max(1, int(DF_MAX_FRAC * n))
    vocab = sorted(doc_freq)
    vidx = {w: k for k, w in enumerate(vocab)}
    inc = torch.zeros((n, len(vocab)), dtype=torch.float32)
    for i, cnts in enumerate(counts_l):
        for w in cnts:
            inc[i, vidx[w]] = 1.0
    inter = inc @ inc.T
    sizes = inc.sum(dim=1)
    union = sizes[:, None] + sizes[None, :] - inter
    jac = (inter / union.clamp(min=1.0)).to(torch.float32)
    v = cand["v"].to(torch.float32)
    assert v.shape[0] == n, (v.shape, n)
    vn = v / v.norm(dim=1, keepdim=True).clamp(min=1e-30)
    cos = (vn @ vn.T).to(torch.float32)
    return {
        "ids": cand["ids"],
        "users": cand["users"],
        "systems": cand["systems"],
        "counts": counts_l,
        "n_tokens": ntok_l,
        "prompt_words": prompt_words,
        "doc_freq": dict(doc_freq),
        "df_max_docs": df_max_docs,
        "n_docs": n,
        "v": v,
        "jaccard": jac,
        "cos": cos,
    }


def excl_set(pool: dict, a_idx: int) -> set[str]:
    """A-side exclusion set: words in A's prompt OR in A's target text (§4.2)."""
    return set(pool["prompt_words"][a_idx]) | set(pool["counts"][a_idx].keys())


def target_for(pool: dict, a_idx: int, b_idx: int) -> list[str]:
    """T_{B|A} via the shared §4.2 construction (build AND null draws)."""
    return build_target(
        pool["counts"][b_idx],
        pool["n_tokens"][b_idx],
        pool["doc_freq"],
        pool["df_max_docs"],
        pool["n_docs"],
        excl_set(pool, a_idx),
    )


# ── pair sampling (quartile-stratified, distinct-A + distinct-B) ─────────────


def sample_pairs(pool: dict, n_pairs: int, n_strata: int, rng: np.random.Generator) -> dict:
    """§4 sampling: A AND B without replacement within leg, B != A, eligibility
    screen via ``pair_eligible``, cos-quartile strata with spill-to-nearest."""
    n = pool["n_docs"]
    jac = pool["jaccard"].numpy()
    cos = pool["cos"].numpy()
    elig = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for_j = (jac[i] <= JACCARD_MAX) & (cos[i] <= COS_MAX)
        elig[i] = for_j
        # identity-check: the vectorized mask IS pair_eligible elementwise
        elig[i, i] = False
    probe_j = int(rng.integers(n))
    probe_i = int(rng.integers(n))
    if probe_i != probe_j:
        assert elig[probe_i, probe_j] == pair_eligible(
            float(jac[probe_i, probe_j]), float(cos[probe_i, probe_j])
        )
    assert elig.any(), "no eligible (A,B) pairs in this leg"
    ecos = cos[elig]
    if n_strata > 1:
        qs = np.linspace(0, 1, n_strata + 1)[1:-1]
        edges = [float(x) for x in np.quantile(ecos, qs)]
    else:
        edges = []
    stratum_of = (
        np.searchsorted(np.asarray(edges), cos, side="right")
        if edges
        else np.zeros((n, n), dtype=int)
    )
    quotas = [n_pairs // n_strata + (1 if s < n_pairs % n_strata else 0) for s in range(n_strata)]
    avail_a = np.ones(n, dtype=bool)
    avail_b = np.ones(n, dtype=bool)
    picked: list[tuple[int, int, int]] = []  # (a, b, realized_stratum)
    realized = Counter()
    spilled = Counter()
    order = list(range(n_strata))
    for s in order:
        want = quotas[s]
        for target_s in sorted(range(n_strata), key=lambda t: (abs(t - s), t)):
            while want > 0:
                cand_mask = elig & (stratum_of == target_s) & avail_a[:, None] & avail_b[None, :]
                idxs = np.argwhere(cand_mask)
                if idxs.shape[0] == 0:
                    break
                a, b = (int(x) for x in idxs[rng.integers(idxs.shape[0])])
                picked.append((a, b, target_s))
                realized[target_s] += 1
                if target_s != s:
                    spilled[(s, target_s)] += 1
                avail_a[a] = False
                avail_b[b] = False
                want -= 1
            if want == 0:
                break
        assert want == 0, f"stratum {s}: could not fill quota even after spill (short {want})"
    a_ids = [pool["ids"][a] for a, _, _ in picked]
    b_ids = [pool["ids"][b] for _, b, _ in picked]
    # §4 distinct-B assert (Statistics critic Must-Fix 2), BEFORE any generation.
    assert len(set(b_ids)) == len(b_ids), "duplicate B within leg"
    assert len(set(a_ids)) == len(a_ids), "duplicate A within leg"
    return {
        "picked": picked,
        "quartile_edges": edges,
        "quotas": quotas,
        "realized_per_stratum": {int(k): int(v) for k, v in realized.items()},
        "spilled": {f"{k[0]}->{k[1]}": int(v) for k, v in spilled.items()},
    }


# ── delta construction (support-restricted operator pseudoinverse) ───────────


def load_operators(dest: Path) -> dict:
    """The two reused operators at the EXACT phase-3 shift conventions
    (issue1776_phase3.py:180-181): pred_M(d) = (d/xsd) @ W (row convention),
    pred_J(d) = J @ d (column convention)."""
    root = dest / C76.HF_PREFIX
    m_path = root / "analysis_tensors/comparator/m_ridge_x50k.pt"
    j_path = root / "analysis_tensors/jac_full/J_last.pt"
    payload = torch.load(m_path, map_location="cpu", weights_only=False)
    w = payload["W"].to(torch.float64)
    xsd = payload["xsd"].to(torch.float64)
    assert w.ndim == 2 and xsd.shape == (w.shape[0],), (w.shape, xsd.shape)
    jd = torch.load(j_path, map_location="cpu", weights_only=True)
    j = jd["J"].to(torch.float64)
    assert j.shape == (w.shape[0], w.shape[0]), (j.shape, w.shape)
    return {
        "a_m": w / xsd[:, None],  # (H_in, H_out) raw-space M' operator
        "j": j,  # (H_out, H_in) column-convention Jacobian
        "m_sha": _sha256_file(m_path),
        "j_sha": _sha256_file(j_path),
        "selected_lambda": float(payload.get("selected_lambda", math.nan)),
    }


def _pinv_factors(a_tilde: torch.Tensor, rcond: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    """SVD pseudoinverse factors of the restricted operator (H_out, k):
    returns (U_k (H_out, r), Vs_k (k, r) with 1/s folded in, retained rank)."""
    u, s, vh = torch.linalg.svd(a_tilde, full_matrices=False)
    keep = s > rcond * s[0]
    r = int(keep.sum())
    assert r >= 1, (rcond, float(s[0]))
    return u[:, keep], (vh[keep].T / s[keep][None, :]), r


def solve_deltas(op_col: torch.Tensor, basis: torch.Tensor, dv: torch.Tensor, rcond: float) -> dict:
    """delta_raw = B_s @ pinv(Op @ B_s) @ dv for a STACK of targets (batched
    GEMM chain — ONE SVD per operator, no per-pair factorization; plan §4).
    ``op_col``: column-convention operator (H_out, H_in); dv: (n, H_out)."""
    a_tilde = op_col @ basis  # (H_out, k)
    u_k, vs_k, rank = _pinv_factors(a_tilde, rcond)
    coef = (dv @ u_k) @ vs_k.T  # (n, r) @ (r, k) -> (n, k)
    delta_raw = coef @ basis.T  # (n, H_in)
    proj = dv @ u_k  # component of dv in the restricted column space
    on_support_mass = (proj.norm(dim=1) ** 2) / (dv.norm(dim=1) ** 2).clamp(min=1e-30)
    return {
        "delta_raw": delta_raw,
        "rank": rank,
        "on_support_mass": on_support_mass,
    }


def cap_deltas(delta_raw: torch.Tensor, cap: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Norm cap delta = delta_raw * min(1, cap/||delta_raw||) rowwise."""
    norms = delta_raw.norm(dim=1)
    scale = (cap / norms.clamp(min=1e-30)).clamp(max=1.0)
    return delta_raw * scale[:, None], norms


def claimed_stats(op_col: torch.Tensor, delta: torch.Tensor, dv: torch.Tensor) -> dict:
    """Per-pair claimed shift Op(delta): norm fraction + cosine vs dv_target."""
    pred = delta @ op_col.T  # (n, H_out)
    pn = pred.norm(dim=1)
    dn = dv.norm(dim=1).clamp(min=1e-30)
    cosv = (pred * dv).sum(dim=1) / (pn.clamp(min=1e-30) * dn)
    return {"claimed_norm": pn, "claimed_frac": pn / dn, "claimed_cos": cosv}


# ── build ─────────────────────────────────────────────────────────────────────


def _capture_lmsys_v(model, tok, cand: dict, readout_layer: int, batch: int) -> torch.Tensor:
    """v(ctx) = mean teacher-forced v_{L'} over the SAME dose-round baseline
    draws that supply the target TEXT (text+tensor coherence, plan §4)."""
    n = len(cand["ids"])
    vs: list[torch.Tensor] = []
    for start in range(0, n, batch):
        sel = list(range(start, min(start + batch, n)))
        ctx_dicts = [{"system": cand["systems"][i], "user": cand["users"][i]} for i in sel]
        comps = [cand["texts"][i] for i in sel]
        cap = capture_vectors(
            model, tok, ctx_dicts, [readout_layer], completions=comps, batch_size=batch
        )
        for rec in cap["per_context"]:
            vs.append(rec["v_a_mean"][0].to(torch.float32))  # single-layer row
        print(f"[swap-build] lmsys v capture {min(start + batch, n)}/{n}", flush=True)
    return torch.stack(vs)


def _build_fingerprint(args, stage_probes: dict) -> dict:
    """EVERY output-affecting build regime key (resume-skip contract, #722 r3)."""
    return {
        "script": "issue1776_swap",
        "seed": GLOBAL_SEED,
        "pairs_per_leg": args.pairs_per_leg,
        "strata": args.strata,
        "n_pcs": args.n_pcs,
        "rcond": args.rcond,
        "rcond_sensitivity": args.rcond_sensitivity,
        "eligibility": ELIGIBILITY,
        "norm_cap_plan": NORM_CAP_PLAN,
        "model": args.model,
        "tiny": bool(args.tiny),
        "source_layer": args.source_layer,
        "readout_layer": args.readout_layer,
        "m_ridge_sha": stage_probes["m_ridge_sha"],
        "jlast_sha": stage_probes["jlast_sha"],
    }


def cmd_build(args) -> int:
    """Pair sampling + eligibility screen + targets + deltas + build gates.

    Writes (out-dir): pairs.jsonl, targets.json, pool.pt, deltas.pt,
    build_report.json. Gates: G-METRIC-SANITY (rc=9), G-DOSE-DEGENERATE (rc=7)
    — demoted to log lines under --gates-informational (smoke calibration).
    ``--round patch`` routes to the patch-round build instead."""
    if args.round == "patch":
        assert args.swap_artifacts and args.swap_build_report, (
            "--round patch requires --swap-artifacts + --swap-build-report"
        )
        return cmd_build_patch(args)
    stage_report = json.loads(args.stage_report.read_text())
    fp = _build_fingerprint(args, stage_report["schema_probes"])
    report_path = args.out_dir / "build_report.json"
    if report_path.exists() and not args.force:
        try:
            prior = json.loads(report_path.read_text()).get("inputs")
        except (json.JSONDecodeError, OSError):
            prior = None
        if prior == fp and (args.out_dir / "deltas.pt").exists():
            print(f"[swap-build] MATCHING fingerprint — skip (resume): {report_path}", flush=True)
            return 0
        print("[swap-build] fingerprint MISMATCH/unreadable -> rebuild", flush=True)
    cap_norm = _norm_cap(args.ladder_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # candidate pools (LMSYS v via a live capture pass; WildChat v stored)
    model, tok = P3.load_model(args)
    lm = _load_lmsys_candidates(args.dest, tok)
    lm["v"] = _capture_lmsys_v(model, tok, lm, args.readout_layer, args.capture_batch)
    wc = _load_wildchat_candidates(args.dest, args.readout_layer)
    del model
    pools = {"lmsys": build_leg_pool(lm), "wildchat": build_leg_pool(wc)}

    # operators + support basis (the phase-2 on-support convention)
    ops = load_operators(args.dest)
    jcap = torch.load(
        args.dest / C76.HF_PREFIX / "analysis_tensors/jpairs/jpair_capture.pt",
        map_location="cpu",
        weights_only=True,
    )
    basis = _support_basis(jcap["c14"].to(torch.float32), args.n_pcs)  # (H, k) fp64
    hidden = basis.shape[0]
    assert ops["j"].shape == (hidden, hidden), (ops["j"].shape, hidden)
    op_cols = {"swap_mprime": ops["a_m"].T.contiguous(), "swap_jlast": ops["j"]}

    # per-leg sampling + per-pair targets/deltas
    rng = np.random.default_rng(derive_seed("sampling"))
    pair_rows: list[dict] = []
    per_pair_targets: dict[str, dict] = {}
    dv_list: list[torch.Tensor] = []
    strata_report: dict = {}
    n_excluded_floor = 0
    for leg in LEGS:
        pool = pools[leg]
        sampled = sample_pairs(pool, args.pairs_per_leg, args.strata, rng)
        strata_report[leg] = {
            "n_pool": pool["n_docs"],
            "quartile_edges": sampled["quartile_edges"],
            "quotas": sampled["quotas"],
            "realized_per_stratum": sampled["realized_per_stratum"],
            "spilled": sampled["spilled"],
        }
        for k, (a, b, s) in enumerate(sampled["picked"]):
            pid = f"sw_{'lm' if leg == 'lmsys' else 'wc'}{k:03d}"
            t_b = target_for(pool, a, b)
            included = len(t_b) >= TARGET_FLOOR
            if not included:
                n_excluded_floor += 1
            a_own_excl = set(pool["prompt_words"][a])
            t_a = build_target(
                pool["counts"][a],
                pool["n_tokens"][a],
                pool["doc_freq"],
                pool["df_max_docs"],
                pool["n_docs"],
                a_own_excl,
            )
            pair_rows.append(
                {
                    "pair_id": pid,
                    "leg": leg,
                    "a_idx": a,
                    "b_idx": b,
                    "a_id": pool["ids"][a],
                    "b_id": pool["ids"][b],
                    "stratum": int(s),
                    "cos_ab": float(pool["cos"][a, b]),
                    "jaccard_ab": float(pool["jaccard"][a, b]),
                    "n_target_b": len(t_b),
                    "n_target_a": len(t_a),
                    "included": bool(included),
                    "exclusion_reason": None if included else "target_floor",
                    "a_user": pool["users"][a],
                    "a_system": pool["systems"][a],
                }
            )
            per_pair_targets[pid] = {"t_b": t_b, "t_a": t_a}
            dv_list.append(pool["v"][b].to(torch.float64) - pool["v"][a].to(torch.float64))
        # §4 distinct-B/-A asserts re-checked on the manifest rows per leg
        leg_rows = [r for r in pair_rows if r["leg"] == leg]
        assert len({r["b_id"] for r in leg_rows}) == len(leg_rows), "manifest duplicate B"
        assert len({r["a_id"] for r in leg_rows}) == len(leg_rows), "manifest duplicate A"

    inc_mask = np.array([r["included"] for r in pair_rows])
    assert inc_mask.sum() >= 2, "fewer than 2 included pairs — cannot run the design"
    for leg in LEGS:
        n_leg = sum(1 for r in pair_rows if r["leg"] == leg and r["included"])
        assert n_leg >= 1, f"leg {leg}: zero included pairs after the target floor"
    dv = torch.stack(dv_list)  # (n_pairs_all, H) fp64

    # deltas per operator (ONE SVD each; batched over pairs), cap, claims
    deltas: dict[str, torch.Tensor] = {}
    per_op: dict[str, dict] = {}
    sens: dict[str, dict] = {}
    for arm, op_col in op_cols.items():
        sol = solve_deltas(op_col, basis, dv, args.rcond)
        capped, raw_norms = cap_deltas(sol["delta_raw"], cap_norm)
        cl = claimed_stats(op_col, capped, dv)
        deltas[arm] = capped.to(torch.float32)
        per_op[arm] = {
            "rank": sol["rank"],
            "raw_norms": raw_norms,
            "capped": raw_norms > cap_norm,
            "realized_norms": capped.norm(dim=1),
            "on_support_mass": sol["on_support_mass"],
            **cl,
        }
        # rcond sensitivity (§11: verdict must be rcond-stable; reported here)
        sol2 = solve_deltas(op_col, basis, dv, args.rcond_sensitivity)
        capped2, _ = cap_deltas(sol2["delta_raw"], cap_norm)
        num = (capped * capped2).sum(dim=1)
        den = (capped.norm(dim=1) * capped2.norm(dim=1)).clamp(min=1e-30)
        sens[arm] = {
            "rcond_alt": args.rcond_sensitivity,
            "rank_alt": sol2["rank"],
            "delta_cos_median": float((num / den).median()),
            "delta_cos_q10": float(np.quantile((num / den).numpy(), 0.10)),
        }

    # norm-matched random control: per pair, scaled to the max operator norm
    rand_rows = []
    for i, r in enumerate(pair_rows):
        g = torch.Generator().manual_seed(derive_seed("rand", r["pair_id"]))
        vec = torch.randn(hidden, generator=g, dtype=torch.float64)
        tgt = float(max(per_op[a]["realized_norms"][i] for a in OP_ARMS))
        rand_rows.append(vec / vec.norm() * tgt)
    deltas["swap_random"] = torch.stack(rand_rows).to(torch.float32)

    # ── gates (computed on INCLUDED pairs) ────────────────────────────────────
    inc_idx = np.flatnonzero(inc_mask)
    # G-METRIC-SANITY: ceiling read — T_B against B's OWN target text
    ceil_recall, ceil_mrr = [], []
    for i in inc_idx:
        r = pair_rows[i]
        pool = pools[r["leg"]]
        rk = rank_map(_pool_texts(args, pools, r["leg"], r["b_idx"], lm, wc))
        m, rec = mrr_recall(rk, per_pair_targets[r["pair_id"]]["t_b"])
        ceil_mrr.append(m)
        ceil_recall.append(rec)
    ceiling_mean_recall = float(np.mean(ceil_recall))
    gate_metric = ceiling_mean_recall >= 0.8
    # G-DOSE-DEGENERATE: median claimed fraction per operator arm
    med_frac = {a: float(per_op[a]["claimed_frac"][inc_idx].median()) for a in OP_ARMS}
    gate_dose = any(v >= 0.01 for v in med_frac.values())

    # attainable-MRR anchor (LMSYS leg, zero-GPU positive control, §6)
    anchor = _attainable_anchor(pools["lmsys"], pair_rows, lm)

    # measured null variance (power restatement before generation, §11/§12-10)
    null_sd = _build_null_sd(pools, pair_rows, per_pair_targets, lm, wc, args)

    # ── persist ───────────────────────────────────────────────────────────────
    pairs_path = args.out_dir / "pairs.jsonl"
    tmp = pairs_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in pair_rows) + "\n")
    tmp.replace(pairs_path)
    targets_payload = {
        "label": LABEL,
        "eligibility": ELIGIBILITY,
        "per_pair": {
            r["pair_id"]: {
                **per_pair_targets[r["pair_id"]],
                "included": r["included"],
                "leg": r["leg"],
                "a_id": r["a_id"],
                "b_id": r["b_id"],
                "a_user": r["a_user"],
                "b_excerpt": _excerpt(_pool_texts(args, pools, r["leg"], r["b_idx"], lm, wc)),
                "a_excerpt": _excerpt(_pool_texts(args, pools, r["leg"], r["a_idx"], lm, wc)),
            }
            for r in pair_rows
        },
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out_dir / "targets.json", targets_payload)
    torch.save(
        {
            leg: {
                k: pools[leg][k]
                for k in (
                    "ids",
                    "counts",
                    "n_tokens",
                    "prompt_words",
                    "doc_freq",
                    "df_max_docs",
                    "n_docs",
                    "v",
                    "jaccard",
                    "cos",
                )
            }
            for leg in LEGS
        },
        args.out_dir / "pool.pt",
    )
    torch.save(
        {
            "pair_ids": [r["pair_id"] for r in pair_rows],
            "included": torch.from_numpy(inc_mask),  # tensor: weights_only-safe
            "dv_target": dv.to(torch.float32),
            "deltas": deltas,
            "rcond": args.rcond,
            "n_pcs": args.n_pcs,
            "norm_cap": cap_norm,
            "operator_shas": {"m_ridge_x50k": ops["m_sha"], "J_last": ops["j_sha"]},
            "per_op": {
                a: {k: v for k, v in per_op[a].items() if isinstance(v, torch.Tensor)}
                for a in OP_ARMS
            },
            "ranks": {a: per_op[a]["rank"] for a in OP_ARMS},
        },
        args.out_dir / "deltas.pt",
    )
    report = {
        "inputs": fp,
        "norm_cap": cap_norm,
        "n_pairs_sampled": len(pair_rows),
        "n_pairs_included": int(inc_mask.sum()),
        "n_excluded_target_floor": n_excluded_floor,
        "strata": strata_report,
        "operator_shas": {"m_ridge_x50k": ops["m_sha"], "J_last": ops["j_sha"]},
        "selected_lambda": ops["selected_lambda"],
        "support": {"n_pcs": args.n_pcs, "basis_sha": _sha256_tensor(basis.to(torch.float32))},
        "per_operator": {
            a: {
                "retained_rank": per_op[a]["rank"],
                "raw_norm_median": float(per_op[a]["raw_norms"][inc_idx].median()),
                "capped_fraction": float(per_op[a]["capped"][inc_idx].float().mean()),
                "realized_norm_median": float(per_op[a]["realized_norms"][inc_idx].median()),
                "claimed_frac_median": med_frac[a],
                "claimed_frac_q10_q90": [
                    float(np.quantile(per_op[a]["claimed_frac"][inc_idx].numpy(), q))
                    for q in (0.1, 0.9)
                ],
                "claimed_cos_median": float(per_op[a]["claimed_cos"][inc_idx].median()),
                "on_support_mass_median": float(per_op[a]["on_support_mass"][inc_idx].median()),
                "rcond_sensitivity": sens[a],
            }
            for a in OP_ARMS
        },
        "per_pair": [
            {
                **{
                    k: r[k]
                    for k in (
                        "pair_id",
                        "leg",
                        "stratum",
                        "cos_ab",
                        "jaccard_ab",
                        "n_target_b",
                        "included",
                        "exclusion_reason",
                    )
                },
                **{f"{a}_claimed_frac": float(per_op[a]["claimed_frac"][i]) for a in OP_ARMS},
                **{f"{a}_capped": bool(per_op[a]["capped"][i]) for a in OP_ARMS},
                **{f"{a}_realized_norm": float(per_op[a]["realized_norms"][i]) for a in OP_ARMS},
            }
            for i, r in enumerate(pair_rows)
        ],
        "gates": {
            "G-METRIC-SANITY": {
                "ceiling_mean_recall50": ceiling_mean_recall,
                "ceiling_mrr_median": float(np.median(ceil_mrr)),
                "threshold": 0.8,
                "pass": bool(gate_metric),
            },
            "G-DOSE-DEGENERATE": {
                "median_claimed_frac": med_frac,
                "threshold": 0.01,
                "pass": bool(gate_dose),
            },
            "informational": bool(args.gates_informational),
        },
        "attainable_mrr_anchor_lmsys": anchor,
        "measured_null_mrr_sd": null_sd,
        "power_restatement": {
            "n_pairs": int(inc_mask.sum()),
            "detectable_delta_mrr_80pct": (
                2.8 * null_sd["within_cell_sd_median"] / math.sqrt(max(int(inc_mask.sum()), 1))
                if null_sd["within_cell_sd_median"] is not None
                else None
            ),
            "basis": "2.8 * sd / sqrt(n) with the build-measured within-cell null SD as proxy",
        },
        "seeds": {"global_seed": GLOBAL_SEED, "derivation": "sha256('1776:<tags>')[:4] % 2**31"},
        "label_directory_mapping": {LABEL: FU},
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(report_path, report)
    print(
        f"[swap-build] [phase=build_done] pairs={len(pair_rows)} included={int(inc_mask.sum())} "
        f"ceiling_recall={ceiling_mean_recall:.3f} claimed_frac={med_frac} -> {report_path}",
        flush=True,
    )
    if not gate_metric:
        if args.gates_informational:
            print("[swap-build] G-METRIC-SANITY FAIL (informational at smoke n)", flush=True)
        else:
            print(f"[swap-build] G-METRIC-SANITY HALT rc=9 ({ceiling_mean_recall:.3f} < 0.8)")
            return 9
    if not gate_dose:
        if args.gates_informational:
            print("[swap-build] G-DOSE-DEGENERATE FAIL (informational at smoke n)", flush=True)
        else:
            print(f"[swap-build] G-DOSE-DEGENERATE HALT rc=7 (both median fracs < 1%: {med_frac})")
            return 7
    return 0


def _pool_texts(args, pools, leg: str, idx: int, lm: dict, wc: dict) -> list[str]:
    """The target-text draws of pool context ``idx`` (build-time only; the
    candidate dicts still hold the raw texts)."""
    cand = lm if leg == "lmsys" else wc
    return cand["texts"][idx]


def _excerpt(texts: list[str], limit: int = 800) -> str:
    """<=800-char judge excerpt of a context's target text (plan §4.3)."""
    return " ".join(texts)[:limit]


def _attainable_anchor(pool: dict, pair_rows: list[dict], lm: dict) -> dict:
    """LMSYS-leg positive control: T_B from 3 of B's 5 baseline draws scored
    on the held-out 2 (registered diagnostic, §6)."""
    vals_m, vals_r, skipped = [], [], 0
    for r in pair_rows:
        if r["leg"] != "lmsys" or not r["included"]:
            continue
        texts = lm["texts"][r["b_idx"]]
        if len(texts) < 5:
            skipped += 1
            continue
        cnts, tot = ctx_stats(texts[:3])
        if tot == 0:
            skipped += 1
            continue
        t_b3 = build_target(
            cnts,
            tot,
            pool["doc_freq"],
            pool["df_max_docs"],
            pool["n_docs"],
            excl_set(pool, r["a_idx"]),
        )
        if len(t_b3) < TARGET_FLOOR:
            skipped += 1
            continue
        m, rec = mrr_recall(rank_map(texts[3:5]), t_b3)
        vals_m.append(m)
        vals_r.append(rec)
    return {
        "n": len(vals_m),
        "n_skipped": skipped,
        "mrr_median": float(np.median(vals_m)) if vals_m else None,
        "mrr_q10_q90": ([float(np.quantile(vals_m, q)) for q in (0.1, 0.9)] if vals_m else None),
        "recall50_median": float(np.median(vals_r)) if vals_r else None,
        "note": "T_B from 3 of B's 5 baseline draws, scored on the held-out 2 (3/2 split)",
    }


def _build_null_sd(pools, pair_rows, per_pair_targets, lm, wc, args) -> dict:
    """Within-cell null-MRR SD measured on B's-own-text stand-ins (50 draws per
    included pair) — the §11 power-argument basis, measured before generation."""
    sds = []
    n_draws = 50
    for r in pair_rows:
        if not r["included"]:
            continue
        pool = pools[r["leg"]]
        rk = rank_map(_pool_texts(args, pools, r["leg"], r["b_idx"], lm, wc))
        rng = np.random.default_rng(derive_seed("buildnull", r["pair_id"]))
        vals, _, _ = null_draw_scores(
            pool, r["a_idx"], r["b_idx"], rk, n_draws, rng, screen=pair_eligible
        )
        if len(vals) >= 2:
            sds.append(float(np.std(vals)))
    return {
        "n_pairs": len(sds),
        "n_draws_per_pair": n_draws,
        "within_cell_sd_median": float(np.median(sds)) if sds else None,
        "note": "SD of null MRR over eligibility-matched shuffled targets, B's own text stand-in",
    }


# ── shuffled-target null (analysis-side; shared with the build SD probe) ─────


def null_draw_scores(
    pool: dict,
    a_idx: int,
    b_idx: int,
    rank: dict[str, int],
    n_draws: int,
    rng: np.random.Generator,
    *,
    screen,
    memo: dict | None = None,
    max_redraws: int = 200_000,
) -> tuple[list[float], int, int]:
    """§4.2 eligibility-matched shuffled-target null: per draw, redraw B' until
    it passes the IDENTICAL per-A screen as pair sampling (identity-asserted)
    + the target floor; the FULL target construction re-runs per draw.
    Returns (mrr per draw, n_redraws, n_distinct_bprime)."""
    assert screen is pair_eligible, "null screen MUST be the pair-sampling predicate"
    n = pool["n_docs"]
    jac, cos = pool["jaccard"], pool["cos"]
    vals: list[float] = []
    redraws = 0
    seen: set[int] = set()
    memo = {} if memo is None else memo
    for _ in range(n_draws):
        for _attempt in range(max_redraws):
            j = int(rng.integers(n))
            if j == a_idx or j == b_idx:
                redraws += 1
                continue
            if not screen(float(jac[a_idx, j]), float(cos[a_idx, j])):
                redraws += 1
                continue
            key = (a_idx, j)
            t = memo.get(key)
            if t is None:
                t = target_for(pool, a_idx, j)
                memo[key] = t
            if len(t) < TARGET_FLOOR:
                redraws += 1
                continue
            m, _ = mrr_recall(rank, t)
            vals.append(m)
            seen.add(j)
            break
        else:
            raise RuntimeError(f"null draw exhausted {max_redraws} redraws (a_idx={a_idx})")
    return vals, redraws, len(seen)


# ── patch-round build (plan v8 §4: v14_last capture + G-CAPTURE-CONSISTENCY) ─


def _capture_v14_last(model, tok, ctx_dicts: list[dict], source_layer: int, batch: int):
    """v14_last per context via the parity probe's EXACT read path —
    ``render_context`` + LEFT padding + ``extract_layer_activations``[:, T-1]
    — in LENGTH-UNIFORM batches: ``extract_layer_activations`` forwards no
    ``position_ids``, so a MIXED-length left-padded batch would give shorter
    rows pad-shifted RoPE positions (#502 class); grouping rows by exact
    rendered token length makes left-pad a no-op and the T-1 read exactly the
    generation-time prefill slot value. Returns ``(n, H)`` fp32 CPU."""
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    n = len(ctx_dicts)
    ids_list = [context_token_ids(tok, c) for c in ctx_dicts]
    by_len: dict[int, list[int]] = {}
    for i, ids in enumerate(ids_list):
        by_len.setdefault(len(ids), []).append(i)
    device = next(model.parameters()).device
    rows: dict[int, torch.Tensor] = {}
    done = 0
    for length in sorted(by_len):
        group = by_len[length]
        for start in range(0, len(group), batch):
            sel = group[start : start + batch]
            texts = [render_context(tok, ctx_dicts[i]) for i in sel]
            prev = tok.padding_side
            tok.padding_side = "left"
            try:
                enc = tok(texts, add_special_tokens=False, padding=True, return_tensors="pt")
            finally:
                tok.padding_side = prev
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            B, T = ids.shape
            # uniform-length invariant: no pad tokens in this batch
            assert T == length and int(mask.sum()) == B * T, (T, length, int(mask.sum()))
            with torch.no_grad():
                acts = extract_layer_activations(model, ids, [source_layer], attention_mask=mask)
            v = acts[source_layer][:, T - 1, :].float().cpu()
            for j, i in enumerate(sel):
                rows[i] = v[j]
            done += len(sel)
            print(f"[patch-build] v14 capture {done}/{n}", flush=True)
    return torch.stack([rows[i] for i in range(n)])


def _patch_build_fingerprint(args, probes: dict) -> dict:
    """EVERY output-affecting patch-build regime key (resume contract, #722 r3)."""
    return {
        "script": "issue1776_swap",
        "round": "patch",
        "seed": GLOBAL_SEED,
        "model": args.model,
        "tiny": bool(args.tiny),
        "source_layer": args.source_layer,
        "readout_layer": args.readout_layer,
        "limit_pairs": int(args.limit_pairs),
        "capture_batch": args.capture_batch,
        "eligibility": ELIGIBILITY,
        "pairs_sha": probes["pairs_sha"],
        "targets_sha": probes["targets_sha"],
        "pool_sha": probes["pool_sha"],
        "deltas_sha": probes["deltas_sha"],
    }


def cmd_build_patch(args) -> int:
    """Patch-round build: fingerprint-check the reused swap build, copy/subset
    the pair manifest, run the v14_last capture pass (the patch VALUES + the
    delta-norm dose covariate, recorded BEFORE any generation), gate
    G-CAPTURE-CONSISTENCY (rc=9), write pairs.jsonl + patch_vectors.pt +
    patch_build_report.json."""
    stage_report = json.loads(args.stage_report.read_text())
    probes = stage_report["schema_probes"]
    fp = _patch_build_fingerprint(args, probes)
    report_path = args.out_dir / "patch_build_report.json"
    if report_path.exists() and not args.force:
        try:
            prior = json.loads(report_path.read_text()).get("inputs")
        except (json.JSONDecodeError, OSError):
            prior = None
        if prior == fp and (args.out_dir / "patch_vectors.pt").exists():
            print(f"[patch-build] MATCHING fingerprint — skip (resume): {report_path}", flush=True)
            return 0
        print("[patch-build] fingerprint MISMATCH/unreadable -> rebuild", flush=True)

    # reused-swap-build fingerprint HALT (plan §10: mismatch = HALT, never a
    # silent rebuild) — the committed swap build_report is the source of record.
    swap_report = json.loads(args.swap_build_report.read_text())
    swap_inputs = swap_report["inputs"]
    for key, want in (
        ("seed", GLOBAL_SEED),
        ("eligibility", ELIGIBILITY),
        ("model", args.model),
        ("source_layer", args.source_layer),
        ("readout_layer", args.readout_layer),
    ):
        if swap_inputs.get(key) != want:
            raise RuntimeError(
                f"reused swap build fingerprint MISMATCH on '{key}': "
                f"committed={swap_inputs.get(key)!r} expected={want!r} — HALT (plan v8 §10)"
            )
    art = Path(args.swap_artifacts)
    all_rows = [
        json.loads(ln) for ln in (art / "pairs.jsonl").read_text().split("\n") if ln.strip()
    ]
    if len(all_rows) != int(swap_report["n_pairs_sampled"]) or sum(
        1 for r in all_rows if r["included"]
    ) != int(swap_report["n_pairs_included"]):
        raise RuntimeError(
            f"staged pairs.jsonl rows ({len(all_rows)}) / included "
            f"({sum(1 for r in all_rows if r['included'])}) != committed swap build_report "
            f"({swap_report['n_pairs_sampled']}/{swap_report['n_pairs_included']}) — HALT"
        )
    targets = json.loads((art / "targets.json").read_text())
    assert targets["eligibility"] == ELIGIBILITY, "staged targets eligibility drift"

    # pair subset (smoke: --limit-pairs N included per leg; full: verbatim copy)
    if args.limit_pairs:
        rows = []
        for leg in LEGS:
            leg_inc = [r for r in all_rows if r["leg"] == leg and r["included"]]
            rows.extend(leg_inc[: args.limit_pairs])
    else:
        rows = all_rows
    inc = [r for r in rows if r["included"]]
    assert len(inc) >= 2, "fewer than 2 included pairs — pilot parity needs 2 distinct pairs"
    for leg in LEGS:
        assert any(r["leg"] == leg for r in inc), f"leg {leg}: zero included pairs selected"

    # resolve A/B prompts (pairs.jsonl carries A's prompt but not B's — B
    # re-derives through the swap round's own candidate loaders, plan §4)
    model, tok = P3.load_model(args)
    lm = _load_lmsys_candidates(args.dest, tok)
    wc = _load_wildchat_candidates(args.dest, args.readout_layer)
    id2ctx: dict[str, dict[str, dict]] = {"lmsys": {}, "wildchat": {}}
    for leg, cand in (("lmsys", lm), ("wildchat", wc)):
        for cid, user, system in zip(cand["ids"], cand["users"], cand["systems"], strict=True):
            id2ctx[leg][cid] = {"system": system, "user": user}

    # unique capture set: (leg, ctx_id) for every A and B of the included pairs
    uniq: dict[tuple[str, str], dict] = {}
    for r in inc:
        uniq.setdefault((r["leg"], r["a_id"]), {"system": r["a_system"], "user": r["a_user"]})
        b_ctx = id2ctx[r["leg"]].get(r["b_id"])
        assert b_ctx is not None, f"pair {r['pair_id']}: b_id {r['b_id']} unresolved in {r['leg']}"
        uniq.setdefault((r["leg"], r["b_id"]), b_ctx)
    keys = sorted(uniq)
    v14 = _capture_v14_last(
        model, tok, [uniq[k] for k in keys], args.source_layer, args.capture_batch
    )
    v14_of = {k: v14[i] for i, k in enumerate(keys)}
    del model

    # G-CAPTURE-CONSISTENCY (plan §7 gate 2): recomputed v14 vs the stored
    # WildChat cx_last at the source layer, row-matched by context id.
    cx14 = (
        _load_wildchat_cx14(args.dest, args.source_layer)
        if probes.get("source_layer_in_wc_layers", True)
        else None
    )
    if cx14 is None:
        consistency = {
            "status": "reference-unavailable",
            "note": "stored WildChat capture lacks the source layer (plan §12 assumption 1); "
            "G-PATCH-PARITY remains the binding slot check",
        }
        gate_consistency = True
    else:
        coss = []
        for leg, cid in keys:
            if leg != "wildchat" or cid not in cx14:
                continue
            a = v14_of[(leg, cid)].to(torch.float64)
            b = cx14[cid].to(torch.float64)
            coss.append(float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-30)))
        assert coss, "no WildChat contexts overlap the stored cx_last reference"
        med = float(np.median(coss))
        consistency = {
            "status": "computed",
            "n_contexts": len(coss),
            "cos_median": med,
            "cos_q10_q90": [float(np.quantile(coss, q)) for q in (0.1, 0.9)],
            "cos_min": float(np.min(coss)),
            "threshold": 0.99,
            "pass": bool(med >= 0.99),
        }
        gate_consistency = consistency["pass"]

    # patch values + the §3 dose covariate (recorded BEFORE generation)
    pair_ids = [r["pair_id"] for r in inc]
    v14_b = torch.stack([v14_of[(r["leg"], r["b_id"])] for r in inc])
    v14_a = torch.stack([v14_of[(r["leg"], r["a_id"])] for r in inc])
    delta_norm = (v14_b.to(torch.float64) - v14_a.to(torch.float64)).norm(dim=1).to(torch.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.out_dir / "pairs.jsonl"
    tmp = pairs_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tmp.replace(pairs_path)
    if not args.limit_pairs:
        assert _sha256_file(pairs_path) == probes["pairs_sha"], (
            "full-mode pairs.jsonl copy is not byte-identical to the staged manifest"
        )
    torch.save(
        {
            "round": "patch",
            "pair_ids": pair_ids,
            "v14_b": v14_b,
            "v14_a": v14_a,
            "delta_norm": delta_norm,
            "source_layer": args.source_layer,
            "model": args.model,
            "capture": "render_context + left-pad (length-uniform batches) + "
            "extract_layer_activations[:, T-1]",
        },
        args.out_dir / "patch_vectors.pt",
    )
    dn = delta_norm.numpy()
    report = {
        "inputs": fp,
        "round": "patch",
        "swap_build_inputs": swap_inputs,
        "swap_build_report_sha": _sha256_file(args.swap_build_report),
        "n_pairs_rows": len(rows),
        "n_pairs_included": len(inc),
        "per_leg_included": {leg: sum(1 for r in inc if r["leg"] == leg) for leg in LEGS},
        "n_unique_contexts_captured": len(keys),
        "consistency_gate": consistency,
        "patch_delta_norm": {
            "median": float(np.median(dn)),
            "q10_q90": [float(np.quantile(dn, q)) for q in (0.1, 0.9)],
            "max": float(dn.max()),
            "norm_cap_ref": NORM_CAP_PLAN,
            "fraction_above_swap_cap": float((dn > NORM_CAP_PLAN).mean()),
        },
        "label_directory_mapping": {LABEL_PATCH: FU_PATCH},
        "operator_shas_echo": swap_report.get("operator_shas"),
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(report_path, report)
    print(
        f"[patch-build] [phase=build_done] pairs={len(rows)} included={len(inc)} "
        f"captured={len(keys)} delta_norm_median={float(np.median(dn)):.2f} "
        f"consistency={consistency.get('status')} -> {report_path}",
        flush=True,
    )
    if not gate_consistency:
        if args.gates_informational:
            print("[patch-build] G-CAPTURE-CONSISTENCY FAIL (informational at smoke n)", flush=True)
        else:
            print(
                f"[patch-build] G-CAPTURE-CONSISTENCY HALT rc=9 "
                f"(cos_median={consistency.get('cos_median')} < 0.99)",
                flush=True,
            )
            return 9
    return 0


# ── pilot (G-SWAP-PARITY rc=8 + G-PILOT rc=7) ────────────────────────────────


def parity_probe(
    model,
    tok,
    contexts: list[dict],
    deltas: torch.Tensor,
    source_layer: int,
    *,
    replace: bool = False,
) -> dict:
    """G-SWAP-PARITY / G-PATCH-PARITY (plan §7): the per-row (B,H) prefill edit
    lands EXACTLY at each row's T-1 slot of the layer-``source_layer`` block
    output, other positions untouched, exactly ONE edit per forward.
    Hooked-vs-unhooked captures of the SAME forward inputs; expected value
    recomputed in the hidden dtype, tolerance 1e-4 fp32. ``replace=True``
    (the patch round) expects position T-1 to EQUAL each row's delta value
    elementwise (wholesale replacement, no add)."""
    assert deltas.dim() == 2 and deltas.shape[0] == len(contexts) == 2, deltas.shape
    texts = [render_context(tok, c) for c in contexts]
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    finally:
        tok.padding_side = prev
    device = next(model.parameters()).device
    hdtype = next(model.parameters()).dtype
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    B, T = ids.shape
    ref = extract_layer_activations(model, ids, [source_layer], attention_mask=mask)[source_layer]
    with DeltaHook(model, source_layer, deltas, 1.0, replace=replace) as hook:
        hook.arm(expected_prompt_len=T)
        ed = extract_layer_activations(model, ids, [source_layer], attention_mask=mask)[
            source_layer
        ]
        n_edits = int(hook.n_edits)
    ref = ref.float().cpu()
    ed = ed.float().cpu()
    if replace:
        exp_last = deltas.to(hdtype).float()
    else:
        exp_last = (ref[:, T - 1].to(hdtype) + deltas.to(hdtype)).float()
    dev_last = float((ed[:, T - 1] - exp_last).abs().max())
    dev_other = float((ed[:, : T - 1] - ref[:, : T - 1]).abs().max())
    ok = n_edits == 1 and dev_last <= 1e-4 and dev_other == 0.0
    return {
        "n_edits": n_edits,
        "max_abs_dev_at_slot": dev_last,
        "max_abs_dev_other_positions": dev_other,
        "tolerance": 1e-4,
        "per_row_deltas": True,
        "replace": bool(replace),
        "pass": bool(ok),
    }


def _pilot_null_block(args, pool, p0, targets_path: Path) -> dict:
    """One production-shape batched null-draw block (p5 sizing basis, §9);
    shared verbatim by the swap and patch pilots."""
    tg = json.loads(targets_path.read_text())["per_pair"][p0["pair_id"]]
    rk = rank_map([tg["b_excerpt"]])
    t0 = time.time()
    vals, redraws, _ = null_draw_scores(
        pool,
        p0["a_idx"],
        p0["b_idx"],
        rk,
        args.null_draws,
        np.random.default_rng(derive_seed("pilotnull")),
        screen=pair_eligible,
    )
    return {
        "n_draws": args.null_draws,
        "wall_s": time.time() - t0,
        "n_redraws": redraws,
        "mean_mrr": float(np.mean(vals)) if vals else None,
    }


def cmd_pilot_patch(args) -> int:
    """Patch-round pilot: G-PATCH-PARITY (rc=8, replace-mode elementwise check,
    binds at smoke) + measured 1-pair wall at the sweep's execution shape
    (rc=7 when projected wall > 2x the §9 budget) + one null-draw block."""
    assert args.patch_vectors, "--round patch requires --patch-vectors"
    rows = [json.loads(ln) for ln in args.pairs.read_text().split("\n") if ln.strip()]
    inc = [r for r in rows if r["included"]]
    assert len(inc) >= 2, "patch pilot needs >=2 included pairs (per-row DISTINCT probe values)"
    pv = torch.load(args.patch_vectors, map_location="cpu", weights_only=True)
    pid_of = {p: i for i, p in enumerate(pv["pair_ids"])}
    model, tok = P3.load_model(args)

    # G-PATCH-PARITY: 2 rows, DISTINCT per-row v14_last(B) patch values
    p0, p1 = inc[0], inc[1]
    probe_deltas = torch.stack(
        [pv["v14_b"][pid_of[p0["pair_id"]]], pv["v14_b"][pid_of[p1["pair_id"]]]]
    )
    assert float((probe_deltas[0] - probe_deltas[1]).abs().max()) > 0, (
        "probe patch values must be per-row DISTINCT"
    )
    ctx2 = [
        {"system": p0["a_system"], "user": p0["a_user"]},
        {"system": p1["a_system"], "user": p1["a_user"]},
    ]
    par = parity_probe(model, tok, ctx2, probe_deltas, args.source_layer, replace=True)
    if not par["pass"]:
        C76.atomic_write_json(
            args.out, {"gate": "G-PATCH-PARITY", "parity": par, "repro": C76.repro_meta()}
        )
        print(f"[patch-pilot] G-PATCH-PARITY HALT rc=8: {par}", flush=True)
        return 8

    # measured wall at the sweep's execution shape (#1415 pilot-shape rule)
    d_stack = pv["v14_b"][pid_of[p0["pair_id"]]][None, :].repeat(args.gen_batch, 1)
    ctx_rep = [{"system": p0["a_system"], "user": p0["a_user"]}] * args.gen_batch
    t0 = time.time()
    with DeltaHook(model, args.source_layer, d_stack, 1.0, replace=True) as hook:
        texts = generate_batch(
            model,
            tok,
            ctx_rep,
            n=args.k_samples,
            hook=hook,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed_base=derive_seed("pilot", "swap_patch"),
        )
        assert hook.n_edits == args.k_samples, (hook.n_edits, args.k_samples)
    wall = time.time() - t0
    assert len(texts) == args.gen_batch and len(texts[0]) == args.k_samples
    per_sample = wall / (args.gen_batch * args.k_samples)
    n_samples_total = len(inc) * (len(PATCH_STEER_ARMS) * args.k_samples + args.k_baseline)
    projected_gpu_h = n_samples_total * per_sample / 3600.0
    ratio = projected_gpu_h / max(args.budget_gpu_h, 1e-9)
    verdict = "OK" if ratio <= 2.0 else "OVER_2X"

    pool = torch.load(args.pool, map_location="cpu", weights_only=False)[p0["leg"]]
    null_block = _pilot_null_block(args, pool, p0, args.targets)
    report = {
        "round": "patch",
        "parity": par,
        "gen_batch": args.gen_batch,
        "k_samples": args.k_samples,
        "max_new_tokens": args.max_new_tokens,
        "wall_s_one_batch": wall,
        "per_sample_s": per_sample,
        "n_samples_total": n_samples_total,
        "projected_gpu_h_serial": projected_gpu_h,
        "projected_wall_h_at_ngpu": projected_gpu_h / max(args.ngpu, 1),
        "budget_gpu_h": args.budget_gpu_h,
        "ratio": ratio,
        "verdict": verdict,
        "null_block": null_block,
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out, report)
    print(
        f"[patch-pilot] [phase=pilot_done] per_sample={per_sample:.3f}s projected="
        f"{projected_gpu_h:.2f} GPU-h budget={args.budget_gpu_h} ratio={ratio:.2f} {verdict}",
        flush=True,
    )
    return 0 if verdict == "OK" else 7


def cmd_pilot(args) -> int:
    """Parity probe (rc=8) + measured 1-pair wall at the SWEEP's execution
    shape (B=gen_batch replication, #1415) + one batched null-draw block;
    rc=7 when projected wall > 2x the §9 budget."""
    if args.round == "patch":
        return cmd_pilot_patch(args)
    assert args.deltas, "--round swap requires --deltas"
    rows = [json.loads(ln) for ln in args.pairs.read_text().split("\n") if ln.strip()]
    inc = [r for r in rows if r["included"]]
    assert inc, "no included pairs for the pilot"
    dl = torch.load(args.deltas, map_location="cpu", weights_only=True)
    pid_of = {p: i for i, p in enumerate(dl["pair_ids"])}
    model, tok = P3.load_model(args)

    # G-SWAP-PARITY: 2 rows, DISTINCT per-row deltas (one per operator arm)
    p0, p1 = inc[0], inc[1 % len(inc)]
    probe_deltas = torch.stack(
        [
            dl["deltas"]["swap_mprime"][pid_of[p0["pair_id"]]],
            dl["deltas"]["swap_jlast"][pid_of[p1["pair_id"]]],
        ]
    )
    ctx2 = [
        {"system": p0["a_system"], "user": p0["a_user"]},
        {"system": p1["a_system"], "user": p1["a_user"]},
    ]
    par = parity_probe(model, tok, ctx2, probe_deltas, args.source_layer)
    if not par["pass"]:
        C76.atomic_write_json(
            args.out, {"gate": "G-SWAP-PARITY", "parity": par, "repro": C76.repro_meta()}
        )
        print(f"[swap-pilot] G-SWAP-PARITY HALT rc=8: {par}", flush=True)
        return 8

    # measured wall at the sweep's execution shape: replicate ONE pair to
    # B=gen_batch with a per-row delta stack (#1415 pilot-shape rule)
    walls = {}
    for arm in OP_ARMS:
        d_row = dl["deltas"][arm][pid_of[p0["pair_id"]]]
        d_stack = d_row[None, :].repeat(args.gen_batch, 1)
        ctx_rep = [{"system": p0["a_system"], "user": p0["a_user"]}] * args.gen_batch
        t0 = time.time()
        with DeltaHook(model, args.source_layer, d_stack, 1.0) as hook:
            texts = generate_batch(
                model,
                tok,
                ctx_rep,
                n=args.k_samples,
                hook=hook,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed_base=derive_seed("pilot", arm),
            )
            assert hook.n_edits == args.k_samples, (hook.n_edits, args.k_samples)
        walls[arm] = time.time() - t0
        assert len(texts) == args.gen_batch and len(texts[0]) == args.k_samples
    per_sample_s = {a: walls[a] / (args.gen_batch * args.k_samples) for a in OP_ARMS}
    per_sample = float(np.mean(list(per_sample_s.values())))
    n_samples_total = len(inc) * (len(STEER_ARMS) * args.k_samples + args.k_baseline)
    projected_gpu_h = n_samples_total * per_sample / 3600.0
    ratio = projected_gpu_h / max(args.budget_gpu_h, 1e-9)
    verdict = "OK" if ratio <= 2.0 else "OVER_2X"

    # one production-shape batched null-draw block (p5 sizing basis, §9)
    pool = torch.load(args.pool, map_location="cpu", weights_only=False)[p0["leg"]]
    null_block = _pilot_null_block(args, pool, p0, args.targets)
    report = {
        "parity": par,
        "gen_batch": args.gen_batch,
        "k_samples": args.k_samples,
        "max_new_tokens": args.max_new_tokens,
        "wall_s_per_arm_one_batch": walls,
        "per_sample_s": per_sample,
        "per_sample_s_per_arm": per_sample_s,
        "n_samples_total": n_samples_total,
        "projected_gpu_h_serial": projected_gpu_h,
        "projected_wall_h_at_ngpu": projected_gpu_h / max(args.ngpu, 1),
        "budget_gpu_h": args.budget_gpu_h,
        "ratio": ratio,
        "verdict": verdict,
        "null_block": null_block,
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out, report)
    print(
        f"[swap-pilot] [phase=pilot_done] per_sample={per_sample:.3f}s projected="
        f"{projected_gpu_h:.2f} GPU-h budget={args.budget_gpu_h} ratio={ratio:.2f} {verdict}",
        flush=True,
    )
    return 0 if verdict == "OK" else 7


# ── run (gen + capture phases, sharded units) ────────────────────────────────

RUN_MATCH_KEYS = (
    "round",
    "model",
    "tiny",
    "dtype",
    "source_layer",
    "readout_layer",
    "k_samples",
    "k_baseline",
    "temperature",
    "max_new_tokens",
    "gen_batch",
    "pairs_sha",
    "deltas_sha",
)


def _run_manifest(args, pairs_sha: str, deltas_sha: str) -> dict:
    return {
        "script": "issue1776_swap",
        "round": args.round,
        "model": args.model,
        "tiny": bool(args.tiny),
        "dtype": args.dtype,
        "source_layer": args.source_layer,
        "readout_layer": args.readout_layer,
        "k_samples": args.k_samples,
        "k_baseline": args.k_baseline,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "gen_batch": args.gen_batch,
        "pairs_sha": pairs_sha,
        "deltas_sha": deltas_sha,
        "seed_rule": "seed_base = sha256('1776:unit:<unit_key>')[:4] % 2**31",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _check_run_manifest(out_root: Path, manifest: dict) -> None:
    """Regime refusal: regimes never mix inside one out-root (phase-3 shape)."""
    path = out_root / "manifest.json"
    if path.exists():
        prior = json.loads(path.read_text())
        diff = [k for k in RUN_MATCH_KEYS if prior.get(k) != manifest.get(k)]
        if diff:
            raise RuntimeError(
                f"swap run manifest MISMATCH on resume (keys: {diff}) — use a fresh --out-root"
            )
    C76.atomic_write_json(path, manifest)


def _units(pairs: list[dict], gen_batch: int, all_arms: tuple[str, ...] = ALL_ARMS) -> list[dict]:
    """Deterministic unit list: chunks of included pairs per arm (baseline
    first). One unit = one batched generate call (per-row deltas)."""
    inc = [r for r in pairs if r["included"]]
    chunks = [inc[i : i + gen_batch] for i in range(0, len(inc), gen_batch)]
    units = []
    for arm in all_arms:
        for k, chunk in enumerate(chunks):
            units.append({"unit_key": f"{arm}_c{k:03d}", "arm": arm, "rows": chunk})
    return units


def cmd_run(args) -> int:
    """One shard of gen or capture units; per-unit persist + resume + progress
    line (checkpoint-per-unit; 600 cells >> the T2 floor). ``--round patch``:
    the steered arm REPLACES the slot with v14_last(B) (``--deltas`` then
    points at patch_vectors.pt, whose per-pair rows are the patch VALUES)."""
    rcfg = ROUNDS[args.round]
    pairs = [json.loads(ln) for ln in args.pairs.read_text().split("\n") if ln.strip()]
    pairs_sha = _sha256_file(args.pairs)
    dl = torch.load(args.deltas, map_location="cpu", weights_only=True)
    deltas_sha = _sha256_file(args.deltas)
    if args.round == "patch":
        assert dl.get("round") == "patch" and "v14_b" in dl, (
            "--round patch requires --deltas to point at patch_vectors.pt"
        )
    pid_of = {p: i for i, p in enumerate(dl["pair_ids"])}
    args.out_root.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_root / "raw_chunks"
    cell_dir = args.out_root / "cells"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cell_dir.mkdir(parents=True, exist_ok=True)
    _check_run_manifest(args.out_root, _run_manifest(args, pairs_sha, deltas_sha))

    units = _units(pairs, args.gen_batch, all_arms=rcfg["all_arms"])
    shard = units[args.shard_index :: args.num_shards]
    if args.limit:
        shard = shard[: args.limit]
    model, tok = P3.load_model(args)
    t0 = time.time()
    for j, unit in enumerate(shard):
        key, arm, rows = unit["unit_key"], unit["arm"], unit["rows"]
        if args.phase == "gen":
            out_path = raw_dir / f"{key}.json"
            if out_path.exists():
                print(f"[swap-gen] unit {j + 1}/{len(shard)} {key} SKIP (done)", flush=True)
                continue
            ctx_dicts = [{"system": r["a_system"], "user": r["a_user"]} for r in rows]
            seed_base = derive_seed("unit", key)
            if arm == rcfg["baseline_arm"]:
                texts = generate_batch(
                    model,
                    tok,
                    ctx_dicts,
                    n=args.k_baseline,
                    hook=None,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    seed_base=seed_base,
                )
                n_edits = 0
                k = args.k_baseline
            else:
                if args.round == "patch":
                    # wholesale replacement with v14_last(B) — the patch arm
                    d_stack = torch.stack([dl["v14_b"][pid_of[r["pair_id"]]] for r in rows])
                    hook_kwargs = {"replace": True}
                else:
                    d_stack = torch.stack([dl["deltas"][arm][pid_of[r["pair_id"]]] for r in rows])
                    hook_kwargs = {}
                with DeltaHook(model, args.source_layer, d_stack, 1.0, **hook_kwargs) as hook:
                    texts = generate_batch(
                        model,
                        tok,
                        ctx_dicts,
                        n=args.k_samples,
                        hook=hook,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        seed_base=seed_base,
                    )
                    n_edits = int(hook.n_edits)
                # parity count-equality half: EXACTLY one prefill edit per draw
                assert n_edits == args.k_samples, (key, n_edits, args.k_samples)
                k = args.k_samples
            payload = {
                "unit": key,
                "arm": arm,
                "round": args.round,
                "mode": "prefill-replace" if (args.round == "patch" and n_edits) else "prefill",
                "model": args.model,
                "k": k,
                "seed_base": seed_base,
                "n_hook_edits": n_edits,
                "rows": [
                    {
                        "pair_id": r["pair_id"],
                        "context_id": r["a_id"],
                        "leg": r["leg"],
                        "user": r["a_user"],
                        "system": r["a_system"],
                        "samples": samp,
                    }
                    for r, samp in zip(rows, texts, strict=True)
                ],
                "repro": C76.repro_meta(),
            }
            C76.atomic_write_json(out_path, payload)
        else:  # capture
            assert args.phase == "capture", args.phase
            out_path = cell_dir / f"{key}.pt"
            if out_path.exists():
                print(f"[swap-cap] unit {j + 1}/{len(shard)} {key} SKIP (done)", flush=True)
                continue
            raw = json.loads((raw_dir / f"{key}.json").read_text())
            ctx_dicts = [{"system": r["system"], "user": r["user"]} for r in raw["rows"]]
            comps_all = [r["samples"] for r in raw["rows"]]
            kept_idx = [P3._nonempty_idx(tok, samp) for samp in comps_all]
            keep = [i for i, ki in enumerate(kept_idx) if ki]
            v19: dict[str, torch.Tensor] = {}
            for start in range(0, len(keep), args.capture_batch):
                sel = keep[start : start + args.capture_batch]
                cap = capture_vectors(
                    model,
                    tok,
                    [ctx_dicts[i] for i in sel],
                    [args.readout_layer],
                    completions=[[comps_all[i][x] for x in kept_idx[i]] for i in sel],
                    batch_size=args.capture_batch,
                )
                for i, rec in zip(sel, cap["per_context"], strict=True):
                    v19[raw["rows"][i]["pair_id"]] = rec["v_a_per_completion"][:, 0, :].to(
                        torch.float32
                    )
            tmp = out_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "unit": key,
                    "arm": arm,
                    "layer": args.readout_layer,
                    "pair_ids": [r["pair_id"] for r in raw["rows"]],
                    "context_ids": [r["context_id"] for r in raw["rows"]],
                    "kept_idx": {
                        raw["rows"][i]["pair_id"]: kept_idx[i] for i in range(len(raw["rows"]))
                    },
                    "v19": v19,
                },
                tmp,
            )
            tmp.replace(out_path)
        print(
            f"[swap-{args.phase}] unit {j + 1}/{len(shard)} {key} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    print(
        f"[swap-run] [phase={args.phase}_shard_done] shard={args.shard_index} n={len(shard)}",
        flush=True,
    )
    return 0


# ── merge-text (canonical per-arm rollout JSONs + judge manifest) ────────────


def _judge_handoff(round_name: str, hf_prefix: str) -> str:
    """Off-pod judge handoff string (merge manifest + final sentinel share it).
    The patch round's judge is CONDITIONAL on the §7 trigger recorded in
    patch_success.json (``judge_triggered``)."""
    rcfg = ROUNDS[round_name]
    if round_name == "patch":
        return (
            "OFF-POD (VM, Batch API) — CONDITIONAL: run ONLY if "
            f"eval_results/issue_1776/{FU_PATCH}/patch_success.json judge_triggered=true: "
            "uv run python scripts/issue1776_swap_judge.py --round patch "
            f"--raw-dir <staged {hf_prefix}/raw_completions/{rcfg['merged_subdir']}> "
            f"--targets eval_results/issue_1776/{FU}/targets.json "
            f"--swap-success eval_results/issue_1776/{FU_PATCH}/patch_success.json "
            f"--out-dir eval_results/issue_1776/{FU_PATCH}"
        )
    return (
        "OFF-POD (VM, Batch API): uv run python scripts/issue1776_swap_judge.py "
        f"--raw-dir <staged {hf_prefix}/raw_completions/steered_swap> "
        f"--targets eval_results/issue_1776/{FU}/targets.json "
        f"--swap-success eval_results/issue_1776/{FU}/swap_success.json "
        f"--out-dir eval_results/issue_1776/{FU}"
    )


def cmd_merge_text(args) -> int:
    """Merge per-unit gen chunks into the canonical per-arm rollout JSONs
    (plan §10 upload paths) + raw_completions_manifest.json. Text files that
    would exceed 9 MB split into numbered parts (non-LFS rule)."""
    rcfg = ROUNDS[args.round]
    raw_dir = args.out_root / "raw_chunks"
    merged_dir = args.out_root / "raw_completions" / rcfg["merged_subdir"]
    merged_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"label": rcfg["label"], "arms": {}, "hf_prefix": args.hf_prefix}
    for arm in rcfg["all_arms"]:
        chunk_files = sorted(raw_dir.glob(f"{arm}_c*.json"))
        assert chunk_files, f"no gen chunks for arm {arm}"
        rows: list[dict] = []
        seeds = {}
        k = None
        mode = "prefill"
        for cf in chunk_files:
            d = json.loads(cf.read_text())
            rows.extend(d["rows"])
            seeds[d["unit"]] = d["seed_base"]
            k = d["k"]
            mode = d.get("mode", "prefill")
        payload = {
            "arm": arm,
            "mode": mode,
            "k": k,
            "unit_seed_bases": seeds,
            "rows": rows,
            "repro": C76.repro_meta(),
        }
        text = json.dumps(payload)
        files = []
        if len(text.encode()) <= 9_000_000:
            p = merged_dir / f"{arm}.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(text)
            tmp.replace(p)
            files.append(p)
        else:  # split rows into parts, never gzip (LFS rule)
            n_parts = math.ceil(len(text.encode()) / 8_000_000)
            per = math.ceil(len(rows) / n_parts)
            for pi in range(n_parts):
                part = {**payload, "rows": rows[pi * per : (pi + 1) * per], "part": pi}
                p = merged_dir / f"{arm}.part{pi:02d}.json"
                tmp = p.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(part))
                tmp.replace(p)
                files.append(p)
        manifest["arms"][arm] = {
            "files": [f.name for f in files],
            "n_rows": len(rows),
            "n_samples": sum(len(r["samples"]) for r in rows),
            "sha256": {f.name: _sha256_file(f) for f in files},
        }
    manifest["judge_handoff"] = _judge_handoff(args.round, args.hf_prefix)
    manifest["repro"] = C76.repro_meta()
    C76.atomic_write_json(args.eval_out / "raw_completions_manifest.json", manifest)
    print(
        f"[swap-merge] [phase=merge_done] arms={list(manifest['arms'])} -> {merged_dir}",
        flush=True,
    )
    return 0


# ── analyze (p5_reduce: metric + nulls + bootstraps + verdict + figures) ─────


def _boot_mean_ci(vals: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Pair-clustered bootstrap 95% CI of the mean (one value per pair, so
    resampling pairs == resampling rows)."""
    if vals.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = vals[rng.integers(0, vals.size, size=(n_boot, vals.size))].mean(axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def _load_merged(
    run_root: Path,
    *,
    merged_subdir: str = "steered_swap",
    all_arms: tuple[str, ...] = ALL_ARMS,
) -> dict[str, dict[str, dict]]:
    """{arm: {pair_id: row}} from the canonical merged rollout JSONs."""
    merged_dir = run_root / "raw_completions" / merged_subdir
    out: dict[str, dict[str, dict]] = {}
    for arm in all_arms:
        files = sorted(merged_dir.glob(f"{arm}*.json"))
        assert files, f"no merged rollout file for arm {arm}"
        rows: dict[str, dict] = {}
        for f in files:
            for r in json.loads(f.read_text())["rows"]:
                assert r["pair_id"] not in rows, (arm, r["pair_id"])
                rows[r["pair_id"]] = r
        out[arm] = rows
    return out


def _load_cells(
    run_root: Path, *, all_arms: tuple[str, ...] = ALL_ARMS
) -> dict[str, dict[str, torch.Tensor]]:
    """{arm: {pair_id: v19 (n_kept, H)}} from the capture chunk files."""
    out: dict[str, dict[str, torch.Tensor]] = {a: {} for a in all_arms}
    for f in sorted((run_root / "cells").glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=True)
        out[d["arm"]].update({p: v for p, v in d["v19"].items()})
    return out


def _e_op(
    delta_vals: np.ndarray,
    op_mrr: np.ndarray,
    rand_mrr: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """§3 per-operator execution predicate E(Op): (a) pair-clustered CI of
    dMRR strictly above 0 AND (b) mean MRR above the random arm's CI upper."""
    lo, hi = _boot_mean_ci(delta_vals, n_boot, seed)
    rng = np.random.default_rng(seed + 1)
    rd = rand_mrr[rng.integers(0, rand_mrr.size, size=(n_boot, rand_mrr.size))].mean(axis=1)
    rand_hi = float(np.percentile(rd, 97.5))
    a = bool(lo > 0)
    b = bool(float(op_mrr.mean()) > rand_hi)
    return {
        "delta_mean": float(delta_vals.mean()),
        "delta_ci95": [lo, hi],
        "criterion_a_ci_above_0": a,
        "mrr_mean": float(op_mrr.mean()),
        "random_mean_ci975_upper": rand_hi,
        "criterion_b_above_random": b,
        "execute": bool(a and b),
    }


def cmd_analyze(args) -> int:
    """§4.2/§6 reduction: judge-free content acquisition vs the eligibility-
    matched shuffled-target null + random control; verdict lattice; retention;
    representation companion; split-half; intrusion audit; figures."""
    if args.round == "patch":
        return cmd_analyze_patch(args)
    # out-arg kind contract (#1776 cycle-5 class): --out-dir is a DIRECTORY
    assert args.out_dir.suffix != ".json", f"--out-dir must be a directory, got {args.out_dir}"
    pairs = [json.loads(ln) for ln in args.pairs.read_text().split("\n") if ln.strip()]
    inc = [r for r in pairs if r["included"]]
    targets = json.loads(args.targets.read_text())
    assert targets["eligibility"] == ELIGIBILITY, "eligibility drift vs build (screen identity)"
    pools = torch.load(args.pool, map_location="cpu", weights_only=False)
    dl = torch.load(args.deltas, map_location="cpu", weights_only=True)
    build_report = json.loads(args.build_report.read_text())
    pid_of = {p: i for i, p in enumerate(dl["pair_ids"])}
    merged = _load_merged(args.run_root)
    cells = _load_cells(args.run_root)

    # §3 row-coverage set-check BEFORE the paired contrast
    reg = {r["pair_id"] for r in inc}
    for arm in ALL_ARMS:
        got = set(merged[arm])
        assert got == reg, (
            f"row-coverage mismatch arm={arm}: missing={sorted(reg - got)[:5]} "
            f"extra={sorted(got - reg)[:5]}"
        )

    rng_master = np.random.default_rng(derive_seed("analyze"))
    per_cell: list[dict] = []
    memo_targets: dict[str, dict] = {leg: {} for leg in LEGS}
    base_rank: dict[str, dict[str, int]] = {}
    base_ra: dict[str, float] = {}
    for r in inc:
        base_rank[r["pair_id"]] = rank_map(merged[BASELINE_ARM][r["pair_id"]]["samples"])
        m_a, _ = mrr_recall(base_rank[r["pair_id"]], targets["per_pair"][r["pair_id"]]["t_a"])
        base_ra[r["pair_id"]] = m_a

    for r in inc:
        pid, leg = r["pair_id"], r["leg"]
        tg = targets["per_pair"][pid]
        pool = pools[leg]
        for arm in STEER_ARMS:
            row = merged[arm][pid]
            rk = rank_map(row["samples"])
            mrr_b, rec_b = mrr_recall(rk, tg["t_b"])
            rng = np.random.default_rng(derive_seed("null", arm, pid))
            nvals, redraws, n_bp = null_draw_scores(
                pool,
                r["a_idx"],
                r["b_idx"],
                rk,
                args.null_draws,
                rng,
                screen=pair_eligible,
                memo=memo_targets[leg],
            )
            null_mean = float(np.mean(nvals))
            mrr_a, _ = mrr_recall(rk, tg["t_a"])
            flags = coherence_check(row["samples"])
            per_cell.append(
                {
                    "pair_id": pid,
                    "leg": leg,
                    "arm": arm,
                    "mrr_b": mrr_b,
                    "recall50_b": rec_b,
                    "null_mean": null_mean,
                    "null_p975": float(np.percentile(nvals, 97.5)),
                    "null_n_redraws": redraws,
                    "null_n_distinct_bprime": n_bp,
                    "delta_mrr": mrr_b - null_mean,
                    "mrr_a_steered": mrr_a,
                    "mrr_a_baseline": base_ra[pid],
                    "retention_ratio": (mrr_a / base_ra[pid]) if base_ra[pid] > 0 else None,
                    "n_coherent": int(sum(flags)),
                    "n_samples": len(flags),
                    "coherence_pass": bool(condition_passes(flags)),
                    "cjk_intruded": bool(any(CJK_RE.search(s) for s in row["samples"])),
                    "claimed_frac": (
                        float(dl["per_op"][arm]["claimed_frac"][pid_of[pid]])
                        if arm in OP_ARMS
                        else None
                    ),
                }
            )
        # baseline diagnostics: B-token base rate + intrusion
        rk0 = base_rank[pid]
        m0, rec0 = mrr_recall(rk0, tg["t_b"])
        per_cell.append(
            {
                "pair_id": pid,
                "leg": leg,
                "arm": BASELINE_ARM,
                "mrr_b": m0,
                "recall50_b": rec0,
                "cjk_intruded": bool(
                    any(CJK_RE.search(s) for s in merged[BASELINE_ARM][pid]["samples"])
                ),
            }
        )
        if len(per_cell) % 50 < 4:
            print(f"[swap-analyze] cells {len(per_cell)} done", flush=True)

    cells_by = lambda arm, leg=None: [  # noqa: E731
        c for c in per_cell if c["arm"] == arm and (leg is None or c["leg"] == leg)
    ]

    def _agg(arm: str, leg: str | None, seed_tag: str) -> dict:
        cs = cells_by(arm, leg)
        rand = cells_by("swap_random", leg)
        dv = np.array([c["delta_mrr"] for c in cs])
        mv = np.array([c["mrr_b"] for c in cs])
        rv = np.array([c["mrr_b"] for c in rand])
        out = {
            "n_pairs": len(cs),
            "mrr_mean": float(mv.mean()) if mv.size else None,
            "mrr_ci95": list(_boot_mean_ci(mv, args.n_boot, derive_seed(seed_tag, "m"))),
            "recall50_mean": float(np.mean([c["recall50_b"] for c in cs])) if cs else None,
            "delta_mrr_mean": float(dv.mean()) if dv.size else None,
            "delta_mrr_ci95": list(_boot_mean_ci(dv, args.n_boot, derive_seed(seed_tag, "d"))),
            "null_mean_mean": float(np.mean([c["null_mean"] for c in cs])) if cs else None,
            "null_p975_mean": float(np.mean([c["null_p975"] for c in cs])) if cs else None,
        }
        if arm in OP_ARMS and cs and rand:
            out["E"] = _e_op(dv, mv, rv, args.n_boot, derive_seed(seed_tag, "e"))
        return out

    per_arm = {
        arm: {
            "pooled": _agg(arm, None, f"{arm}:pooled"),
            **{leg: _agg(arm, leg, f"{arm}:{leg}") for leg in LEGS},
        }
        for arm in STEER_ARMS
    }

    # §3 verdict lattice (pooled MRR carries the verdict) + paired contrast
    e_m = per_arm["swap_mprime"]["pooled"]["E"]["execute"]
    e_j = per_arm["swap_jlast"]["pooled"]["E"]["execute"]
    verdict = {
        (True, True): "both-execute",
        (True, False): "fitted-only",
        (False, True): "jacobian-only",
        (False, False): "neither-executes",
    }[(e_m, e_j)]
    dm = {c["pair_id"]: c["delta_mrr"] for c in cells_by("swap_mprime")}
    dj = {c["pair_id"]: c["delta_mrr"] for c in cells_by("swap_jlast")}
    paired = np.array([dm[p] - dj[p] for p in sorted(dm) if p in dj])
    paired_ci = _boot_mean_ci(paired, args.n_boot, derive_seed("paired"))

    # representation-acquisition companion + split-half (dose-round convention)
    rep_rows, pseudo_floors = _representation_reads(inc, cells, dl, pid_of, targets)
    metric_split = _metric_split_half(inc, merged, targets)

    # intrusion-excluded recounts (dose-round convention)
    intruded_pairs = {
        c["pair_id"]
        for c in per_cell
        if c.get("cjk_intruded") and (c["arm"] == BASELINE_ARM or c["arm"] in STEER_ARMS)
    }
    recounts = {}
    for arm in STEER_ARMS:
        keep = [c["delta_mrr"] for c in cells_by(arm) if c["pair_id"] not in intruded_pairs]
        recounts[arm] = {
            "delta_mrr_mean_excluded": float(np.mean(keep)) if keep else None,
            "n_kept": len(keep),
            "n_excluded": len(cells_by(arm)) - len(keep),
        }

    # acquisition vs the operator's own claim (diagnostic scatter data)
    acq_vs_claim = {}
    for arm in OP_ARMS:
        cs = cells_by(arm)
        x = np.array([c["claimed_frac"] for c in cs])
        y = np.array([c["delta_mrr"] for c in cs])
        if x.size >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            from scipy import stats as sps

            rho = sps.spearmanr(x, y)
            acq_vs_claim[arm] = {
                "spearman": float(rho.statistic),
                "pvalue": float(rho.pvalue),
                "n": int(x.size),
            }
        else:
            acq_vs_claim[arm] = {"spearman": None, "n": int(x.size)}

    cap_sat = {
        a: float(dl["per_op"][a]["capped"][[pid_of[r["pair_id"]] for r in inc]].float().mean())
        for a in OP_ARMS
    }
    result = {
        "label": LABEL,
        "verdict": verdict,
        "E": {"swap_mprime": e_m, "swap_jlast": e_j},
        "paired_contrast_mprime_minus_jlast": {
            "mean": float(paired.mean()) if paired.size else None,
            "ci95": list(paired_ci),
            "n_pairs": int(paired.size),
        },
        "per_arm": per_arm,
        "baseline_b_token_base_rate": {
            "mrr_mean": float(np.mean([c["mrr_b"] for c in cells_by(BASELINE_ARM)])),
            "recall50_mean": float(np.mean([c["recall50_b"] for c in cells_by(BASELINE_ARM)])),
        },
        "metric_ceiling": build_report["gates"]["G-METRIC-SANITY"],
        "attainable_mrr_anchor_lmsys": build_report["attainable_mrr_anchor_lmsys"],
        "cap_saturation_fraction": cap_sat,
        "cap_saturation_framing": (
            "operating-norm dose framing REQUIRED (>80% capped)"
            if any(v > 0.8 for v in cap_sat.values())
            else "cap saturation below the 80% framing threshold"
        ),
        "acquisition_vs_claimed_fraction": acq_vs_claim,
        "intrusion_audit": {
            "regex": "CJK/Kana/Hangul ranges (runtime-built)",
            "n_intruded_pairs": len(intruded_pairs),
            "excluded_recounts": recounts,
        },
        "metric_split_half": metric_split,
        "eligibility": ELIGIBILITY,
        "null_draws_per_cell": args.null_draws,
        "n_boot": args.n_boot,
        "per_cell": per_cell,
        "operator_shas": build_report["operator_shas"],
        "repro": C76.repro_meta(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    C76.atomic_write_json(args.out_dir / "swap_success.json", result)
    shift = {
        "label": LABEL,
        "per_cell": rep_rows,
        "alpha0_pseudo_shift_norm_median": pseudo_floors,
        "note": (
            "cos(dv_bar, dv_target) + achieved fraction per steered cell; split-half of "
            "dv_bar gates narration (dose-round convention)"
        ),
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out_dir / "swap_shift_summaries.json", shift)
    _figures(result, rep_rows, args.fig_dir)
    print(
        f"[swap-analyze] [phase=analyze_done] verdict={verdict} "
        f"E_mprime={e_m} E_jlast={e_j} -> {args.out_dir / 'swap_success.json'}",
        flush=True,
    )
    return 0


def _representation_reads(
    inc,
    cells,
    dl,
    pid_of,
    targets,
    *,
    baseline_arm: str = BASELINE_ARM,
    steer_arms: tuple[str, ...] = STEER_ARMS,
    claimed_norm_of=None,
) -> tuple[list[dict], dict]:
    """Mechanistic companion: cos(dv_bar, dv_target), achieved fraction, and
    per-cell split-half of dv_bar (draws split 2/3); baseline pseudo-shift
    norms as the noise floor (dose-round convention). ``claimed_norm_of``
    (arm, i) -> float|None overrides the operator claimed-norm lookup (the
    patch round claims the FULL dv_target)."""
    rows: list[dict] = []
    pseudo: list[float] = []
    base_mean: dict[str, torch.Tensor] = {}
    for pid, v in cells[baseline_arm].items():
        base_mean[pid] = v.to(torch.float64).mean(dim=0)
        k = v.shape[0]
        if k >= 2:
            h1 = v[: k // 2].to(torch.float64).mean(dim=0)
            h2 = v[k // 2 :].to(torch.float64).mean(dim=0)
            pseudo.append(float((h1 - h2).norm()))
    for r in inc:
        pid = r["pair_id"]
        i = pid_of[pid]
        dv_t = dl["dv_target"][i].to(torch.float64)
        for arm in steer_arms:
            v = cells[arm].get(pid)
            if v is None or pid not in base_mean:
                continue
            dv_bar = v.to(torch.float64).mean(dim=0) - base_mean[pid]
            if claimed_norm_of is not None:
                claimed = claimed_norm_of(arm, i)
            else:
                claimed = float(dl["per_op"][arm]["claimed_norm"][i]) if arm in OP_ARMS else None
            row = {
                "pair_id": pid,
                "leg": r["leg"],
                "arm": arm,
                "dv_bar_norm": float(dv_bar.norm()),
                "dv_target_norm": float(dv_t.norm()),
                "cos_dvbar_dvtarget": float(
                    (dv_bar @ dv_t) / (dv_bar.norm() * dv_t.norm()).clamp(min=1e-30)
                ),
                "achieved_fraction": (
                    float(dv_bar.norm() / max(claimed, 1e-30)) if claimed else None
                ),
            }
            k = v.shape[0]
            if k >= 2:
                h1 = v[: k // 2].to(torch.float64).mean(dim=0) - base_mean[pid]
                h2 = v[k // 2 :].to(torch.float64).mean(dim=0) - base_mean[pid]
                row["split_half_cos_dvbar"] = float(
                    (h1 @ h2) / (h1.norm() * h2.norm()).clamp(min=1e-30)
                )
            rows.append(row)
    return rows, {
        "median": float(np.median(pseudo)) if pseudo else None,
        "n": len(pseudo),
    }


def _metric_split_half(inc, merged, targets, *, steer_arms: tuple[str, ...] = STEER_ARMS) -> dict:
    """Across-cell split-half of the metric per arm (draws split 2/3)."""
    from scipy import stats as sps

    out = {}
    for arm in steer_arms:
        h1v, h2v = [], []
        for r in inc:
            row = merged[arm][r["pair_id"]]
            k = len(row["samples"])
            if k < 2:
                continue
            t_b = targets["per_pair"][r["pair_id"]]["t_b"]
            m1, _ = mrr_recall(rank_map(row["samples"][: k // 2]), t_b)
            m2, _ = mrr_recall(rank_map(row["samples"][k // 2 :]), t_b)
            h1v.append(m1)
            h2v.append(m2)
        if len(h1v) >= 3 and np.std(h1v) > 1e-12 and np.std(h2v) > 1e-12:
            rho = sps.spearmanr(h1v, h2v)
            out[arm] = {"spearman": float(rho.statistic), "n": len(h1v)}
        else:
            out[arm] = {"spearman": None, "n": len(h1v)}
    return out


# ── patch-round analyze (plan v8 §3/§6: one-arm E(patch) + historical wedge) ──


def _hist_rows(swap_committed: dict) -> dict[str, dict[str, dict]]:
    """{arm: {pair_id: row}} from the COMMITTED swap_success per_cell rows —
    the historical contrast (criterion (b) + paired reads, recomputed CPU-side
    from committed per-cell values; plan §4)."""
    hist: dict[str, dict[str, dict]] = {}
    for c in swap_committed["per_cell"]:
        hist.setdefault(c["arm"], {})[c["pair_id"]] = c
    return hist


def cmd_analyze_patch(args) -> int:
    """Patch-round §4.2/§6 reduction: judge-free B-content acquisition of the
    patch arm vs its own recomputed shuffled-target nulls, E(patch) with the
    committed random-arm bound (criterion (b)), the registered conditional-
    judge trigger, the cross-round baseline-drift diagnostic, paired contrasts
    vs the committed operator arms, retention, representation companion,
    split-half, intrusion audit, dose covariate, figures."""
    assert args.out_dir.suffix != ".json", f"--out-dir must be a directory, got {args.out_dir}"
    assert args.patch_vectors and args.patch_build_report and args.swap_success, (
        "--round patch requires --patch-vectors + --patch-build-report + --swap-success"
    )
    pairs = [json.loads(ln) for ln in args.pairs.read_text().split("\n") if ln.strip()]
    inc = [r for r in pairs if r["included"]]
    targets = json.loads(args.targets.read_text())
    assert targets["eligibility"] == ELIGIBILITY, "eligibility drift vs build (screen identity)"
    pools = torch.load(args.pool, map_location="cpu", weights_only=False)
    dl = torch.load(args.deltas, map_location="cpu", weights_only=True)
    pv = torch.load(args.patch_vectors, map_location="cpu", weights_only=True)
    build_report = json.loads(args.build_report.read_text())  # committed SWAP build
    patch_report = json.loads(args.patch_build_report.read_text())
    swap_committed = json.loads(args.swap_success.read_text())
    pid_of_dl = {p: i for i, p in enumerate(dl["pair_ids"])}
    pid_of_pv = {p: i for i, p in enumerate(pv["pair_ids"])}
    merged = _load_merged(args.run_root, merged_subdir="steered_slotpatch", all_arms=PATCH_ALL_ARMS)
    cells = _load_cells(args.run_root, all_arms=PATCH_ALL_ARMS)

    # §3 row-coverage set-checks BEFORE any paired read: this round's arms AND
    # the committed historical arms must cover the registered pair set.
    reg = {r["pair_id"] for r in inc}
    for arm in PATCH_ALL_ARMS:
        got = set(merged[arm])
        assert got == reg, (
            f"row-coverage mismatch arm={arm}: missing={sorted(reg - got)[:5]} "
            f"extra={sorted(got - reg)[:5]}"
        )
    hist = _hist_rows(swap_committed)
    for arm in ALL_ARMS:
        missing = reg - set(hist.get(arm, {}))
        assert not missing, f"committed swap per_cell missing {arm} rows: {sorted(missing)[:5]}"

    per_cell: list[dict] = []
    memo_targets: dict[str, dict] = {leg: {} for leg in LEGS}
    base_rank: dict[str, dict[str, int]] = {}
    base_ra: dict[str, float] = {}
    for r in inc:
        base_rank[r["pair_id"]] = rank_map(merged[PATCH_BASELINE_ARM][r["pair_id"]]["samples"])
        m_a, _ = mrr_recall(base_rank[r["pair_id"]], targets["per_pair"][r["pair_id"]]["t_a"])
        base_ra[r["pair_id"]] = m_a

    for r in inc:
        pid, leg = r["pair_id"], r["leg"]
        tg = targets["per_pair"][pid]
        pool = pools[leg]
        for arm in PATCH_STEER_ARMS:
            row = merged[arm][pid]
            rk = rank_map(row["samples"])
            mrr_b, rec_b = mrr_recall(rk, tg["t_b"])
            rng = np.random.default_rng(derive_seed("null", arm, pid))
            nvals, redraws, n_bp = null_draw_scores(
                pool,
                r["a_idx"],
                r["b_idx"],
                rk,
                args.null_draws,
                rng,
                screen=pair_eligible,
                memo=memo_targets[leg],
            )
            null_mean = float(np.mean(nvals))
            mrr_a, _ = mrr_recall(rk, tg["t_a"])
            flags = coherence_check(row["samples"])
            per_cell.append(
                {
                    "pair_id": pid,
                    "leg": leg,
                    "arm": arm,
                    "mrr_b": mrr_b,
                    "recall50_b": rec_b,
                    "null_mean": null_mean,
                    "null_p975": float(np.percentile(nvals, 97.5)),
                    "null_n_redraws": redraws,
                    "null_n_distinct_bprime": n_bp,
                    "delta_mrr": mrr_b - null_mean,
                    "mrr_a_steered": mrr_a,
                    "mrr_a_baseline": base_ra[pid],
                    "retention_ratio": (mrr_a / base_ra[pid]) if base_ra[pid] > 0 else None,
                    "n_coherent": int(sum(flags)),
                    "n_samples": len(flags),
                    "coherence_pass": bool(condition_passes(flags)),
                    "cjk_intruded": bool(any(CJK_RE.search(s) for s in row["samples"])),
                    "patch_delta_norm": float(pv["delta_norm"][pid_of_pv[pid]]),
                }
            )
        rk0 = base_rank[pid]
        m0, rec0 = mrr_recall(rk0, tg["t_b"])
        per_cell.append(
            {
                "pair_id": pid,
                "leg": leg,
                "arm": PATCH_BASELINE_ARM,
                "mrr_b": m0,
                "recall50_b": rec0,
                "cjk_intruded": bool(
                    any(CJK_RE.search(s) for s in merged[PATCH_BASELINE_ARM][pid]["samples"])
                ),
            }
        )
        if len(per_cell) % 50 < 2:
            print(f"[patch-analyze] cells {len(per_cell)} done", flush=True)

    def cells_by(arm: str, leg: str | None = None) -> list[dict]:
        return [c for c in per_cell if c["arm"] == arm and (leg is None or c["leg"] == leg)]

    def hist_by(arm: str, leg: str | None = None) -> list[dict]:
        rows = [hist[arm][pid] for pid in sorted(reg)]
        return [c for c in rows if leg is None or c["leg"] == leg]

    # per-group aggregates: the patch arm (this round) + historical display
    # blocks recomputed from committed per-cell values (bootstrap reseeded).
    def _agg_patch(leg: str | None, seed_tag: str) -> dict:
        cs = cells_by("swap_patch", leg)
        dv = np.array([c["delta_mrr"] for c in cs])
        mv = np.array([c["mrr_b"] for c in cs])
        return {
            "n_pairs": len(cs),
            "mrr_mean": float(mv.mean()) if mv.size else None,
            "mrr_ci95": list(_boot_mean_ci(mv, args.n_boot, derive_seed(seed_tag, "m"))),
            "recall50_mean": float(np.mean([c["recall50_b"] for c in cs])) if cs else None,
            "delta_mrr_mean": float(dv.mean()) if dv.size else None,
            "delta_mrr_ci95": list(_boot_mean_ci(dv, args.n_boot, derive_seed(seed_tag, "d"))),
            "null_mean_mean": float(np.mean([c["null_mean"] for c in cs])) if cs else None,
            "null_p975_mean": float(np.mean([c["null_p975"] for c in cs])) if cs else None,
        }

    def _agg_hist(arm: str, leg: str | None, seed_tag: str) -> dict:
        cs = hist_by(arm, leg)
        mv = np.array([c["mrr_b"] for c in cs])
        out = {
            "n_pairs": len(cs),
            "mrr_mean": float(mv.mean()) if mv.size else None,
            "mrr_ci95": list(_boot_mean_ci(mv, args.n_boot, derive_seed(seed_tag, "m"))),
            "recall50_mean": float(np.mean([c["recall50_b"] for c in cs])) if cs else None,
            "source": "committed swap_success per_cell (recomputed CPU-side)",
        }
        if arm in STEER_ARMS:
            dv = np.array([c["delta_mrr"] for c in cs])
            out["delta_mrr_mean"] = float(dv.mean()) if dv.size else None
            out["delta_mrr_ci95"] = list(_boot_mean_ci(dv, args.n_boot, derive_seed(seed_tag, "d")))
            out["null_p975_mean"] = float(np.mean([c["null_p975"] for c in cs])) if cs else None
        return out

    per_arm = {
        "swap_patch": {
            "pooled": _agg_patch(None, "patch:pooled"),
            **{leg: _agg_patch(leg, f"patch:{leg}") for leg in LEGS},
        },
        **{
            arm: {
                "pooled": _agg_hist(arm, None, f"patch:hist:{arm}:pooled"),
                **{leg: _agg_hist(arm, leg, f"patch:hist:{arm}:{leg}") for leg in LEGS},
            }
            for arm in STEER_ARMS
        },
    }

    # §3 E(patch): (a) pooled dMRR CI strictly above 0 AND (b) mean MRR above
    # the committed random arm's recomputed 97.5% upper bound (same pairs).
    dv_pool = np.array([c["delta_mrr"] for c in cells_by("swap_patch")])
    mv_pool = np.array([c["mrr_b"] for c in cells_by("swap_patch")])
    rv_pool = np.array([c["mrr_b"] for c in hist_by("swap_random")])
    e_patch = _e_op(dv_pool, mv_pool, rv_pool, args.n_boot, derive_seed("patch:e"))
    verdict = "patch-executes" if e_patch["execute"] else "patch-null"

    # §7 registered conditional-judge trigger: pooled OR either-leg dMRR CI
    # lower bound > 0 (pooled read == E criterion (a)'s own CI).
    trigger_reads = {
        "pooled_delta_ci95": e_patch["delta_ci95"],
        **{f"{leg}_delta_ci95": per_arm["swap_patch"][leg]["delta_mrr_ci95"] for leg in LEGS},
    }
    judge_triggered = bool(
        e_patch["delta_ci95"][0] > 0
        or any(per_arm["swap_patch"][leg]["delta_mrr_ci95"][0] > 0 for leg in LEGS)
    )

    # §6 cross-round comparability diagnostic: patch_a0 vs committed swap_a0.
    def _drift(leg: str | None, tag: str) -> dict:
        pa = np.array([c["mrr_b"] for c in cells_by(PATCH_BASELINE_ARM, leg)])
        sa = np.array([c["mrr_b"] for c in hist_by(BASELINE_ARM, leg)])
        pa_ci = _boot_mean_ci(pa, args.n_boot, derive_seed(tag, "p"))
        sa_ci = _boot_mean_ci(sa, args.n_boot, derive_seed(tag, "s"))
        sep = bool(pa_ci[0] > sa_ci[1] or sa_ci[0] > pa_ci[1])
        return {
            "patch_a0_mrr_mean": float(pa.mean()) if pa.size else None,
            "patch_a0_ci95": list(pa_ci),
            "swap_a0_mrr_mean": float(sa.mean()) if sa.size else None,
            "swap_a0_ci95": list(sa_ci),
            "ci_separated": sep,
        }

    drift = {
        "pooled": _drift(None, "patch:drift:pooled"),
        **{leg: _drift(leg, f"patch:drift:{leg}") for leg in LEGS},
    }
    hist_demoted = bool(drift["pooled"]["ci_separated"])

    # §3 registered paired cross-round contrasts (pair-level, committed rows).
    dpatch = {c["pair_id"]: c["delta_mrr"] for c in cells_by("swap_patch")}
    paired_contrasts = {}
    for arm in OP_ARMS:
        dop = {c["pair_id"]: c["delta_mrr"] for c in hist_by(arm)}
        vals = np.array([dpatch[p] - dop[p] for p in sorted(dpatch) if p in dop])
        paired_contrasts[f"patch_minus_{arm}"] = {
            "mean": float(vals.mean()) if vals.size else None,
            "ci95": list(_boot_mean_ci(vals, args.n_boot, derive_seed("patch:paired", arm))),
            "n_pairs": int(vals.size),
            "note": "cross-round paired read; demoted to descriptive if the baseline-drift "
            "diagnostic separates (see baseline_drift_diagnostic)",
        }

    # representation companion: the patch "claims" the FULL dv_target.
    def _full_dv_claim(_arm: str, i: int) -> float:
        return float(dl["dv_target"][i].to(torch.float64).norm())

    rep_rows, pseudo_floors = _representation_reads(
        inc,
        cells,
        dl,
        pid_of_dl,
        targets,
        baseline_arm=PATCH_BASELINE_ARM,
        steer_arms=PATCH_STEER_ARMS,
        claimed_norm_of=_full_dv_claim,
    )
    metric_split = _metric_split_half(inc, merged, targets, steer_arms=PATCH_STEER_ARMS)

    # intrusion-excluded recounts (dose-round convention)
    intruded_pairs = {c["pair_id"] for c in per_cell if c.get("cjk_intruded")}
    keep = [c["delta_mrr"] for c in cells_by("swap_patch") if c["pair_id"] not in intruded_pairs]
    recounts = {
        "swap_patch": {
            "delta_mrr_mean_excluded": float(np.mean(keep)) if keep else None,
            "n_kept": len(keep),
            "n_excluded": len(cells_by("swap_patch")) - len(keep),
        }
    }

    # acquisition vs the dose covariate ||v14(B) - v14(A)|| (§3 caveat)
    cs = cells_by("swap_patch")
    x = np.array([c["patch_delta_norm"] for c in cs])
    y = np.array([c["delta_mrr"] for c in cs])
    if x.size >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
        from scipy import stats as sps

        rho = sps.spearmanr(x, y)
        acq_vs_norm = {"spearman": float(rho.statistic), "pvalue": float(rho.pvalue), "n": x.size}
    else:
        acq_vs_norm = {"spearman": None, "n": int(x.size)}

    result = {
        "label": LABEL_PATCH,
        "verdict": verdict,
        "E_patch": e_patch,
        "judge_triggered": judge_triggered,
        "judge_trigger_reads": trigger_reads,
        "judge_trigger_rule": "p7 judge RUNS iff pooled OR either-leg pair-clustered 95% CI "
        "lower bound of dMRR(patch) > 0 (plan v8 §7)",
        "per_arm": per_arm,
        "paired_contrasts": paired_contrasts,
        "baseline_drift_diagnostic": {
            **drift,
            "historical_contrast_demoted": hist_demoted,
        },
        "baseline_b_token_base_rate": {
            "mrr_mean": float(np.mean([c["mrr_b"] for c in cells_by(PATCH_BASELINE_ARM)])),
            "recall50_mean": float(
                np.mean([c["recall50_b"] for c in cells_by(PATCH_BASELINE_ARM)])
            ),
        },
        "metric_ceiling": build_report["gates"]["G-METRIC-SANITY"],
        "attainable_mrr_anchor_lmsys": build_report["attainable_mrr_anchor_lmsys"],
        "patch_delta_norm": patch_report["patch_delta_norm"],
        "acquisition_vs_patch_delta_norm": acq_vs_norm,
        "consistency_gate": patch_report["consistency_gate"],
        "intrusion_audit": {
            "regex": "CJK/Kana/Hangul ranges (runtime-built)",
            "n_intruded_pairs": len(intruded_pairs),
            "excluded_recounts": recounts,
        },
        "metric_split_half": metric_split,
        "eligibility": ELIGIBILITY,
        "null_draws_per_cell": args.null_draws,
        "n_boot": args.n_boot,
        "per_cell": per_cell,
        "operator_shas": swap_committed.get("operator_shas"),
        "historical_source": {
            "path": str(args.swap_success),
            "sha": _sha256_file(args.swap_success),
        },
        "repro": C76.repro_meta(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    C76.atomic_write_json(args.out_dir / "patch_success.json", result)
    shift = {
        "label": LABEL_PATCH,
        "per_cell": rep_rows,
        "alpha0_pseudo_shift_norm_median": pseudo_floors,
        "note": (
            "cos(dv_bar, dv_target) + achieved fraction (claim = FULL ||dv_target||) per patch "
            "cell; split-half of dv_bar gates narration (dose-round convention)"
        ),
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out_dir / "patch_shift_summaries.json", shift)
    _figures_patch(result, rep_rows, hist, args.fig_dir)
    print(
        f"[patch-analyze] [phase=analyze_done] verdict={verdict} "
        f"E_patch={e_patch['execute']} judge_triggered={judge_triggered} "
        f"-> {args.out_dir / 'patch_success.json'}",
        flush=True,
    )
    return 0


# ── figures ───────────────────────────────────────────────────────────────────


def _figures(result: dict, rep_rows: list[dict], fig_dir: Path) -> None:
    """HERO 2-panel + exploratory 6-panel dump (plan §6 Figures), paper style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    colors = paper_palette(5)
    arm_color = dict(zip(STEER_ARMS, colors[:3], strict=False))
    per_cell = result["per_cell"]

    # HERO: (left) per-arm MRR w/ CI vs null band + ceiling, per leg;
    # (right) paired per-pair scatter MRR(M') vs MRR(J), colored by leg.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    ax = axes[0]
    groups = ["pooled", *LEGS]
    xs = np.arange(len(groups))
    width = 0.8 / len(STEER_ARMS)
    for k, arm in enumerate(STEER_ARMS):
        vals, err_lo, err_hi, nulls = [], [], [], []
        for g in groups:
            blk = result["per_arm"][arm][g]
            v = blk["mrr_mean"] or 0.0
            lo, hi = blk["mrr_ci95"]
            vals.append(v)
            # non-negative offsets, element-wise clamped (xerr/yerr gotcha)
            err_lo.append(max(0.0, v - lo) if np.isfinite(lo) else 0.0)
            err_hi.append(max(0.0, hi - v) if np.isfinite(hi) else 0.0)
            nulls.append(blk["null_p975_mean"] or 0.0)
        ax.bar(
            xs + k * width,
            vals,
            width,
            yerr=[err_lo, err_hi],
            label=arm,
            color=arm_color[arm],
        )
        ax.scatter(xs + k * width, nulls, marker="_", s=120, color="0.2", zorder=3)
    ceil = result["metric_ceiling"].get("ceiling_mrr_median")
    if ceil is not None:
        ax.axhline(ceil, color="0.4", ls="--", lw=1, label="ceiling (B's own text)")
    ax.set_xticks(xs + 0.4 - width / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel("B-content MRR")
    ax.set_title("acquisition vs shuffled-target null (dashes)")
    ax.legend(fontsize=6)
    ax = axes[1]
    m = {c["pair_id"]: c["mrr_b"] for c in per_cell if c["arm"] == "swap_mprime"}
    j = {c["pair_id"]: c["mrr_b"] for c in per_cell if c["arm"] == "swap_jlast"}
    leg_of = {c["pair_id"]: c["leg"] for c in per_cell if c["arm"] == "swap_mprime"}
    for k, leg in enumerate(LEGS):
        pts = [(m[p], j[p]) for p in m if p in j and leg_of[p] == leg]
        if pts:
            ax.scatter(*zip(*pts, strict=False), s=14, color=colors[k], label=leg)
    lim = max(0.05, *(list(m.values()) + list(j.values()) + [0.01]))
    ax.plot([0, lim], [0, lim], color="0.6", lw=0.8)
    ax.set_xlabel("MRR (fitted map M')")
    ax.set_ylabel("MRR (Jacobian J_last)")
    ax.set_title("paired per-pair acquisition")
    ax.legend(fontsize=7)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "swap_success_hero.png", dpi=200)
    plt.close(fig)

    # EXPLORATORY dump: recall bars / claimed-frac hist / acquisition-vs-claim
    # / retention-vs-acquisition / rep-cos hist / split-half + norms.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    ax = axes[0, 0]
    for k, arm in enumerate(STEER_ARMS):
        vals = [result["per_arm"][arm][g]["recall50_mean"] or 0.0 for g in groups]
        ax.bar(xs + k * width, vals, width, label=arm, color=arm_color[arm])
    ax.set_xticks(xs + 0.4 - width / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel("recall@50")
    ax.set_title("recall@50 companion")
    ax.legend(fontsize=6)
    ax = axes[0, 1]
    for k, arm in enumerate(OP_ARMS):
        fr = [c["claimed_frac"] for c in per_cell if c["arm"] == arm and c["claimed_frac"]]
        if fr:
            ax.hist(fr, bins=20, alpha=0.6, label=arm, color=arm_color[arm])
    ax.set_xlabel("claimed fraction ||Op d||/||dv||")
    ax.set_title("operator-claimed shift fraction")
    ax.legend(fontsize=6)
    ax = axes[0, 2]
    for arm in OP_ARMS:
        cs = [c for c in per_cell if c["arm"] == arm and c["claimed_frac"] is not None]
        ax.scatter(
            [c["claimed_frac"] for c in cs],
            [c["delta_mrr"] for c in cs],
            s=10,
            color=arm_color[arm],
            label=arm,
        )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("claimed fraction")
    ax.set_ylabel("delta MRR vs null")
    ax.set_title("acquisition vs operator claim")
    ax.legend(fontsize=6)
    ax = axes[1, 0]
    for arm in STEER_ARMS:
        cs = [c for c in per_cell if c["arm"] == arm and c.get("retention_ratio") is not None]
        ax.scatter(
            [c["delta_mrr"] for c in cs],
            [c["retention_ratio"] for c in cs],
            s=10,
            color=arm_color[arm],
            label=arm,
        )
    ax.axhline(1.0, color="0.4", lw=0.8)
    ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("delta MRR (B acquisition)")
    ax.set_ylabel("A-retention ratio")
    ax.set_title("erasure cost vs acquisition")
    ax.legend(fontsize=6)
    ax = axes[1, 1]
    for arm in STEER_ARMS:
        vals = [r["cos_dvbar_dvtarget"] for r in rep_rows if r["arm"] == arm]
        if vals:
            ax.hist(vals, bins=20, alpha=0.6, label=arm, color=arm_color[arm])
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("cos(dv_bar, dv_target)")
    ax.set_title("representation acquisition")
    ax.legend(fontsize=6)
    ax = axes[1, 2]
    sh = [r.get("split_half_cos_dvbar") for r in rep_rows]
    sh = [x for x in sh if x is not None]
    if sh:
        ax.hist(sh, bins=20, alpha=0.7, color=colors[3])
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("split-half cos of dv_bar")
    ax.set_title("per-cell shift reliability")
    fig.savefig(fig_dir / "swap_success_exploratory.png", dpi=200)
    plt.close(fig)
    print(f"[swap-analyze] figures -> {fig_dir}", flush=True)


def _figures_patch(
    result: dict, rep_rows: list[dict], hist: dict[str, dict[str, dict]], fig_dir: Path
) -> None:
    """Patch-round HERO (patch beside the committed operator/random arms vs
    null band + anchor + ceiling, per leg + pooled) + exploratory dump.
    ``hist`` = the committed swap per_cell rows keyed {arm: {pair_id: row}}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    colors = paper_palette(5)
    arms = ("swap_patch", *STEER_ARMS)  # this round + the historical contrast
    arm_color = dict(zip(arms, colors[:4], strict=False))
    per_cell = result["per_cell"]
    groups = ["pooled", *LEGS]
    xs = np.arange(len(groups))
    width = 0.8 / len(arms)

    # HERO: per-arm MRR w/ CI vs null band (dashes) + anchor + ceiling.
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for k, arm in enumerate(arms):
        vals, err_lo, err_hi, nulls = [], [], [], []
        for g in groups:
            blk = result["per_arm"][arm][g]
            v = blk["mrr_mean"] or 0.0
            lo, hi = blk["mrr_ci95"]
            vals.append(v)
            # non-negative offsets, element-wise clamped (xerr/yerr gotcha)
            err_lo.append(max(0.0, v - lo) if np.isfinite(lo) else 0.0)
            err_hi.append(max(0.0, hi - v) if np.isfinite(hi) else 0.0)
            nulls.append(blk.get("null_p975_mean") or 0.0)
        ax.bar(
            xs + k * width,
            vals,
            width,
            yerr=[err_lo, err_hi],
            label=arm if arm == "swap_patch" else f"{arm} (committed)",
            color=arm_color[arm],
        )
        ax.scatter(xs + k * width, nulls, marker="_", s=110, color="0.2", zorder=3)
    ceil = result["metric_ceiling"].get("ceiling_mrr_median")
    if ceil is not None:
        ax.axhline(ceil, color="0.4", ls="--", lw=1, label="ceiling (B's own text)")
    anchor = (result.get("attainable_mrr_anchor_lmsys") or {}).get("mrr_median")
    if anchor is not None:
        ax.axhline(anchor, color="0.6", ls=":", lw=1, label="attainable anchor (lmsys)")
    ax.set_xticks(xs + 0.4 - width / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel("B-content MRR")
    ax.set_title("full-state patch vs committed swap arms (null band = dashes)")
    ax.legend(fontsize=6)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "patch_success_hero.png", dpi=200)
    plt.close(fig)

    # EXPLORATORY dump: recall bars / paired scatters / retention / dose hist
    # / acquisition-vs-dose / rep-cos + split-half / baseline-drift dumbbell.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), layout="constrained")
    ax = axes[0, 0]
    for k, arm in enumerate(arms):
        vals = [result["per_arm"][arm][g]["recall50_mean"] or 0.0 for g in groups]
        ax.bar(xs + k * width, vals, width, label=arm, color=arm_color[arm])
    ax.set_xticks(xs + 0.4 - width / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel("recall@50")
    ax.set_title("recall@50 companion")
    ax.legend(fontsize=5)
    p_mrr = {c["pair_id"]: c["mrr_b"] for c in per_cell if c["arm"] == "swap_patch"}
    leg_of = {c["pair_id"]: c["leg"] for c in per_cell if c["arm"] == "swap_patch"}
    leg_color = dict(zip(LEGS, colors[:2], strict=True))
    for col, hist_arm in enumerate(OP_ARMS):
        ax = axes[0, 1 + col]
        h_mrr = {pid: hist[hist_arm][pid]["mrr_b"] for pid in p_mrr if pid in hist[hist_arm]}
        for leg in LEGS:
            pts = [(h_mrr[p], p_mrr[p]) for p in h_mrr if leg_of[p] == leg]
            if pts:
                ax.scatter(*zip(*pts, strict=False), s=12, color=leg_color[leg], label=leg)
        lim = max(0.05, *(list(p_mrr.values()) + list(h_mrr.values()) + [0.01]))
        ax.plot([0, lim], [0, lim], color="0.6", lw=0.8)
        ax.set_xlabel(f"MRR ({hist_arm}, committed)")
        ax.set_ylabel("MRR (patch)")
        blk = result["paired_contrasts"][f"patch_minus_{hist_arm}"]
        ax.set_title(f"paired: dMRR(patch)-dMRR({hist_arm}) mean={blk['mean']:.4f}")
        ax.legend(fontsize=6)
    ax = axes[0, 3]
    cs = [c for c in per_cell if c["arm"] == "swap_patch" and c.get("retention_ratio") is not None]
    ax.scatter(
        [c["delta_mrr"] for c in cs],
        [c["retention_ratio"] for c in cs],
        s=10,
        color=arm_color["swap_patch"],
    )
    ax.axhline(1.0, color="0.4", lw=0.8)
    ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("delta MRR (B acquisition)")
    ax.set_ylabel("A-retention ratio")
    ax.set_title("erasure cost vs acquisition")
    ax = axes[1, 0]
    dn = [c["patch_delta_norm"] for c in per_cell if c["arm"] == "swap_patch"]
    if dn:
        ax.hist(dn, bins=20, color=arm_color["swap_patch"])
    ax.axvline(NORM_CAP_PLAN, color="0.3", ls="--", lw=1, label="swap cap 47.36")
    ax.set_xlabel("||v14(B) - v14(A)||")
    ax.set_title("patch dose covariate")
    ax.legend(fontsize=6)
    ax = axes[1, 1]
    cs = [c for c in per_cell if c["arm"] == "swap_patch"]
    ax.scatter(
        [c["patch_delta_norm"] for c in cs],
        [c["delta_mrr"] for c in cs],
        s=10,
        color=arm_color["swap_patch"],
    )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("||v14(B) - v14(A)||")
    ax.set_ylabel("delta MRR vs null")
    sp = result["acquisition_vs_patch_delta_norm"].get("spearman")
    ax.set_title(f"acquisition vs dose (rho={sp if sp is None else round(sp, 3)})")
    ax = axes[1, 2]
    vals = [r["cos_dvbar_dvtarget"] for r in rep_rows if r["arm"] == "swap_patch"]
    if vals:
        ax.hist(vals, bins=20, alpha=0.8, color=arm_color["swap_patch"])
    sh = [r.get("split_half_cos_dvbar") for r in rep_rows]
    sh = [v for v in sh if v is not None]
    if sh:
        ax.hist(sh, bins=20, alpha=0.5, color=colors[4], label="split-half")
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("cos(dv_bar, dv_target) / split-half")
    ax.set_title("representation acquisition + reliability")
    ax.legend(fontsize=6)
    ax = axes[1, 3]
    dd = result["baseline_drift_diagnostic"]
    ys = np.arange(len(groups))
    for j, g in enumerate(groups):
        blk = dd[g]
        for off, (key, ckey, lbl) in enumerate(
            (
                ("patch_a0_mrr_mean", "patch_a0_ci95", "patch_a0 (this round)"),
                ("swap_a0_mrr_mean", "swap_a0_ci95", "swap_a0 (committed)"),
            )
        ):
            v = blk[key]
            lo, hi = blk[ckey]
            if v is None:
                continue
            xerr = [
                [max(0.0, v - lo) if np.isfinite(lo) else 0.0],
                [max(0.0, hi - v) if np.isfinite(hi) else 0.0],
            ]
            ax.errorbar(
                [v],
                [ys[j] + 0.15 * off],
                xerr=xerr,
                fmt="o",
                ms=4,
                color=colors[off],
                label=lbl if j == 0 else None,
            )
    ax.set_yticks(ys + 0.075)
    ax.set_yticklabels(groups)
    ax.set_xlabel("baseline B-token MRR")
    ax.set_title(f"cross-round drift (separated={dd['pooled']['ci_separated']})")
    ax.legend(fontsize=6)
    fig.savefig(fig_dir / "patch_success_exploratory.png", dpi=200)
    plt.close(fig)
    print(f"[patch-analyze] figures -> {fig_dir}", flush=True)


# ── smoke fixtures (tiny-dim mirrors of the REAL artifacts' key sets) ────────


def cmd_smoke_fixtures(args) -> int:
    """Tiny-H fixture artifacts at the staged layout, mirroring the REAL
    payloads' realized key sets (fixture-from-real rule, #1073): operator +
    Jacobian + jpair capture + dose baseline JSON + one wildchat chunk pair.
    Texts are synthetic (distinct content words per context + shared stop
    words to exercise the DF filter + one CJK draw for the intrusion audit)."""
    h = args.hidden
    root = args.dest / C76.HF_PREFIX
    g = torch.Generator().manual_seed(7)
    # realized key sets read from the REAL local artifacts at plan/impl time
    m_keys = (
        "W",
        "input_layer",
        "kind",
        "output_layer",
        "selected_lambda",
        "tag",
        "xmu",
        "xsd",
        "ymu",
    )
    (root / "analysis_tensors/comparator").mkdir(parents=True, exist_ok=True)
    m_payload = {
        "W": torch.randn(h, h, generator=g) / math.sqrt(h),
        "xmu": torch.zeros(h),
        "xsd": torch.rand(h, generator=g) * 0.5 + 0.75,
        "ymu": torch.zeros(h),
        "selected_lambda": 1.0,
        "kind": "smoke",
        "tag": "m_ridge_x50k_smoke",
        "input_layer": args.source_layer,
        "output_layer": args.readout_layer,
    }
    assert set(m_payload) == set(m_keys)
    torch.save(m_payload, root / "analysis_tensors/comparator/m_ridge_x50k.pt")
    (root / "analysis_tensors/jac_full").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "J": torch.randn(h, h, generator=g) / math.sqrt(h),
            "c_bar_half": torch.zeros(2, h),
            "half_counts": [1, 1],
            "half_sums": torch.zeros(2, h),
            "manifests": ["smoke"],
            "n_pair_half": [1, 1],
            "v_bar_half": torch.zeros(2, h),
        },
        root / "analysis_tensors/jac_full/J_last.pt",
    )
    (root / "analysis_tensors/jpairs").mkdir(parents=True, exist_ok=True)
    n_cap = max(args.n_pcs + 8, 40)
    torch.save(
        {
            "c14": torch.randn(n_cap, h, generator=g),
            "c19": torch.randn(n_cap, h, generator=g),
            "v19": torch.randn(n_cap, h, generator=g),
            "layers": [args.source_layer, args.readout_layer],
            "pair_id": [f"jp{i}" for i in range(n_cap)],
        },
        root / "analysis_tensors/jpairs/jpair_capture.pt",
    )

    def _text(i: int, d: int) -> str:
        core = " ".join(f"tok{i}w{k} item{i}n{k}" for k in range(10))
        shared = "the quick common filler and with"
        cjk = chr(0x4E2D) + chr(0x6587) if (i == 1 and d == 0) else ""
        return f"{shared} {core} draw{d}marker{i} {cjk}".strip()

    n_lm = args.n_contexts
    (root / "analysis_tensors/contexts").mkdir(parents=True, exist_ok=True)
    ctx_rows = [
        {
            "context_id": f"lm{i:03d}",
            "user": f"question {i} about topic{i} alpha beta",
            "system": None,
            "source": LMSYS_SOURCE,
        }
        for i in range(n_lm)
    ]
    (root / "analysis_tensors/contexts/contexts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ctx_rows) + "\n"
    )
    (root / "raw_completions/steered_dose").mkdir(parents=True, exist_ok=True)
    C76.atomic_write_json(
        root / "raw_completions/steered_dose/baseline_a0.json",
        {
            "stratum": "baseline_a0",
            "direction": "baseline",
            "alpha": 0.0,
            "mode": "prefill",
            "model": "smoke",
            "contexts": [
                {**r, "samples": [_text(i, d) for d in range(5)]} for i, r in enumerate(ctx_rows)
            ],
        },
    )
    n_wc = args.n_contexts
    (root / "wildchat_fresh/final_token_capture").mkdir(parents=True, exist_ok=True)
    (root / "wildchat_fresh/raw_completions").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "chunk": 0,
            "shard_index": 0,
            "ci": list(range(n_wc)),
            "cx_last": torch.randn(n_wc, 2, h, generator=g),
            "v_x": torch.randn(n_wc, 2, h, generator=g),
            "layers": [args.source_layer, args.readout_layer],
            "prompts": [f"wc question {i} gamma delta" for i in range(n_wc)],
        },
        root / "wildchat_fresh/final_token_capture/shard00_chunk0000.pt",
    )
    C76.atomic_write_json(
        root / "wildchat_fresh/raw_completions/shard00_chunk0000.json",
        {
            "chunk": 0,
            "shard_index": 0,
            "rows": [
                {
                    "ci": i,
                    "prompt": f"wc question {i} gamma delta",
                    "response": _text(100 + i, 0) + f" wcword{i}x wcword{i}y",
                }
                for i in range(n_wc)
            ],
        },
    )
    # stage-report stand-in so build's fingerprint path works against fixtures
    C76.atomic_write_json(
        args.dest / "swap_stage_report.json",
        {
            "revision": "smoke-fixtures",
            "schema_probes": {
                "m_ridge_sha": "smoke",
                "jlast_sha": "smoke",
                "wc_layers": [args.source_layer, args.readout_layer],
            },
        },
    )
    print(f"[swap-fixtures] tiny fixtures (H={h}) -> {args.dest}", flush=True)
    return 0


# ── sentinels (pod-side-reporting contract; committed writers, no heredocs) ──


def cmd_progress(args) -> int:
    """Non-blocking tick sentinel (poll_pipeline schema)."""
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:progress",
        "version": 1,
        "task_id": C76.ISSUE,
        "gate": args.gate,
        "blocks_pipeline": False,
        "by": "issue1776_swap_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": {"msg": args.msg, "mode": args.mode, "label": LABEL},
    }
    slug = args.gate.replace(":", "_").replace("/", "_")
    path = log_dir / f"issue-{C76.ISSUE}-{slug}-{int(time.time() * 1000)}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)
    print(f"[swap-dispatch] progress sentinel: {path.name}")
    return 0


def cmd_final_sentinel(args) -> int:
    """Terminal results sentinel (epm:results / epm:smoke-result)."""
    import subprocess

    smoke_like = args.dry or args.mode == "smoke"
    kind = "epm:smoke-result" if smoke_like else "epm:results"
    rcfg = ROUNDS[args.round]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=args.repo_root,
            check=True,
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    eval_dir = Path(args.eval_dir)
    eval_paths = sorted(
        str(p.relative_to(args.repo_root)) if p.is_relative_to(args.repo_root) else str(p)
        for p in eval_dir.rglob("*.json")
    )
    note = {
        "followup_label": rcfg["label"],
        "mode": args.mode,
        "dry_run": args.dry,
        "ngpu": args.ngpu,
        "git_commit": sha,
        "eval_json_paths": eval_paths,
        "hf_prefixes": {
            "rollout_text": f"{args.hf_prefix}/raw_completions/{rcfg['merged_subdir']}",
            "analysis_tensors": f"{args.hf_prefix}/analysis_tensors/{rcfg['fu']}",
        },
        "offpod_handoffs": {
            "p7_judge_offpod": _judge_handoff(args.round, args.hf_prefix),
        },
        "wandb": "n/a (no training this round)",
    }
    if args.round == "patch":
        # the §7 trigger verdict rides the terminal sentinel (plan §6.5 note)
        success_path = eval_dir / "patch_success.json"
        if success_path.is_file():
            try:
                res = json.loads(success_path.read_text())
                note["judge_triggered"] = res.get("judge_triggered")
                note["judge_trigger_reads"] = res.get("judge_trigger_reads")
            except (json.JSONDecodeError, OSError) as exc:
                note["judge_triggered"] = f"unreadable ({exc})"
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "task_id": C76.ISSUE,
        "gate": "smoke" if smoke_like else "results",
        "blocks_pipeline": not smoke_like,
        "by": "issue1776_swap_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note,
    }
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"issue-{C76.ISSUE}-{kind.replace(':', '_')}-{int(time.time())}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)
    print(f"[swap-dispatch] final sentinel: {path.name} kind={kind}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _model(p):
        p.add_argument("--model", default=C.DEFAULT_MODEL)
        p.add_argument("--source-layer", type=int, default=C76.SOURCE_LAYER)
        p.add_argument("--readout-layer", type=int, default=C76.READOUT_LAYER)
        p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--tiny", action="store_true", help="from-config tiny Qwen2 (CPU smoke)")

    def _gen(p):
        p.add_argument("--k-samples", type=int, default=5)
        p.add_argument("--k-baseline", type=int, default=5)
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--max-new-tokens", type=int, default=1024)
        p.add_argument("--gen-batch", type=int, default=16)
        p.add_argument("--capture-batch", type=int, default=8)

    def _round(p):
        p.add_argument(
            "--round",
            choices=sorted(ROUNDS),
            default="swap",
            help="swap (default, byte-identical) | patch (slot_patch_sufficiency, plan v8)",
        )

    s = sub.add_parser("stage", help="stage all swap inputs at one fresh pin + schema probes")
    _round(s)
    s.add_argument("--dest", type=Path, default=C76.DATA_DIR / "hf_dl")
    s.add_argument("--pin-file", type=Path, default=C76.DATA_DIR / "data_repo_pin_swap.json")
    s.add_argument("--refresh-pin", action="store_true")
    s.add_argument("--max-wc-chunks", type=int, default=0, help="smoke cap (0 = all)")
    s.add_argument("--report", type=Path, required=True)
    s.add_argument(
        "--stage-source-layer",
        type=int,
        default=C76.SOURCE_LAYER,
        help="patch round: record whether the stored WildChat capture carries this layer",
    )
    s.set_defaults(fn=cmd_stage)

    b = sub.add_parser("build", help="pairs + targets + deltas + build gates")
    _round(b)
    _model(b)
    b.add_argument("--dest", type=Path, default=C76.DATA_DIR / "hf_dl")
    b.add_argument("--stage-report", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--pairs-per-leg", type=int, default=75)
    b.add_argument("--strata", type=int, default=4)
    b.add_argument("--n-pcs", type=int, default=256)
    b.add_argument("--rcond", type=float, default=1e-6)
    b.add_argument("--rcond-sensitivity", type=float, default=1e-4)
    b.add_argument("--ladder-json", type=Path, default=LADDER_JSON)
    b.add_argument("--capture-batch", type=int, default=8)
    b.add_argument("--gates-informational", action="store_true")
    b.add_argument("--force", action="store_true")
    b.add_argument("--swap-artifacts", type=Path, help="patch round: staged swap build artifacts")
    b.add_argument(
        "--swap-build-report", type=Path, help="patch round: COMMITTED swap build_report.json"
    )
    b.add_argument(
        "--limit-pairs",
        type=int,
        default=0,
        help="patch round smoke: first N included pairs per leg (0 = all, verbatim copy)",
    )
    b.set_defaults(fn=cmd_build)

    pi = sub.add_parser("pilot", help="parity probe (rc=8) + measured wall gate (rc=7)")
    _round(pi)
    _model(pi)
    _gen(pi)
    pi.add_argument("--pairs", type=Path, required=True)
    pi.add_argument("--deltas", type=Path, help="swap round: deltas.pt (required for --round swap)")
    pi.add_argument("--patch-vectors", type=Path, help="patch round: patch_vectors.pt")
    pi.add_argument("--pool", type=Path, required=True)
    pi.add_argument("--targets", type=Path, required=True)
    pi.add_argument("--null-draws", type=int, default=200)
    pi.add_argument("--budget-gpu-h", type=float, default=2.6)
    pi.add_argument("--ngpu", type=int, default=8)
    pi.add_argument("--out", type=Path, required=True)
    pi.set_defaults(fn=cmd_pilot)

    r = sub.add_parser("run", help="one shard of gen or capture units")
    _round(r)
    _model(r)
    _gen(r)
    r.add_argument("--phase", choices=["gen", "capture"], required=True)
    r.add_argument("--pairs", type=Path, required=True)
    r.add_argument(
        "--deltas",
        type=Path,
        required=True,
        help="per-pair edit store: deltas.pt (swap) | patch_vectors.pt (patch)",
    )
    r.add_argument("--out-root", type=Path, required=True)
    r.add_argument("--shard-index", type=int, default=0)
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--limit", type=int, default=0)
    r.set_defaults(fn=cmd_run)

    mt = sub.add_parser("merge-text", help="canonical per-arm rollout JSONs + judge manifest")
    _round(mt)
    mt.add_argument("--out-root", type=Path, required=True)
    mt.add_argument("--eval-out", type=Path, required=True)
    mt.add_argument("--hf-prefix", required=True)
    mt.set_defaults(fn=cmd_merge_text)

    an = sub.add_parser("analyze", help="metric + nulls + verdict + figures")
    _round(an)
    an.add_argument("--pairs", type=Path, required=True)
    an.add_argument("--targets", type=Path, required=True)
    an.add_argument("--pool", type=Path, required=True)
    an.add_argument("--deltas", type=Path, required=True)
    an.add_argument("--build-report", type=Path, required=True)
    an.add_argument("--run-root", type=Path, required=True)
    an.add_argument("--null-draws", type=int, default=200)
    an.add_argument("--n-boot", type=int, default=1000)
    an.add_argument("--out-dir", type=Path, required=True, help="eval-results DIRECTORY")
    an.add_argument("--fig-dir", type=Path, required=True)
    an.add_argument("--patch-vectors", type=Path, help="patch round: patch_vectors.pt")
    an.add_argument("--patch-build-report", type=Path, help="patch round: patch_build_report.json")
    an.add_argument("--swap-success", type=Path, help="patch round: COMMITTED swap_success.json")
    an.set_defaults(fn=cmd_analyze)

    sf = sub.add_parser("smoke-fixtures", help="tiny-H fixture artifacts (local CPU smoke)")
    sf.add_argument("--dest", type=Path, required=True)
    sf.add_argument("--hidden", type=int, default=64)
    sf.add_argument("--n-contexts", type=int, default=12)
    sf.add_argument("--n-pcs", type=int, default=16)
    sf.add_argument("--source-layer", type=int, default=1)
    sf.add_argument("--readout-layer", type=int, default=3)
    sf.set_defaults(fn=cmd_smoke_fixtures)

    pr = sub.add_parser("progress", help="tick sentinel writer")
    pr.add_argument("--log-dir", required=True)
    pr.add_argument("--gate", required=True)
    pr.add_argument("--msg", required=True)
    pr.add_argument("--mode", default="?")
    pr.set_defaults(fn=cmd_progress)

    fs = sub.add_parser("final-sentinel", help="terminal results sentinel writer")
    _round(fs)
    fs.add_argument("--log-dir", required=True)
    fs.add_argument("--mode", required=True)
    fs.add_argument("--dry", action="store_true")
    fs.add_argument("--eval-dir", required=True)
    fs.add_argument("--repo-root", type=Path, required=True)
    fs.add_argument("--hf-prefix", required=True)
    fs.add_argument("--ngpu", default="?")
    fs.set_defaults(fn=cmd_final_sentinel)

    ic = sub.add_parser("import-check", help="resolve deferred imports (Axis-1 smoke leg)")
    ic.set_defaults(fn=None)

    args = ap.parse_args(argv)
    if args.cmd == "import-check":
        # Execute every deferred/function-body import this module reaches on
        # its real code paths (the #1689 rounds-2/3/4 false-pass class).
        import fnmatch  # noqa: F401
        import subprocess  # noqa: F401

        from scipy import stats as sps

        import issue779_ffc_n1m_generate_capture as N1G

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        from explore_persona_space.analysis.paper_plots import (
            paper_palette,
            set_paper_style,
        )
        from explore_persona_space.orchestrate import hub

        for sym in (
            hub.stage_hub_file,
            hub.retry_transient,
            N1G.N50._remote_index,
            sps.spearmanr,
            paper_palette,
            set_paper_style,
            P3.load_model,
            P3._nonempty_idx,
            *_STEERING_SURFACE,
        ):
            assert callable(sym), sym
        print("[swap] import-check OK")
        return 0
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
