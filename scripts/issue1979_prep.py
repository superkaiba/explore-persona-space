"""#1979 F0 VM data-prep — prefix panel + shared query draw + config manifests.

Plan v2 §4 F0 (task #1979, unit 1a). Builds the experiment's INPUT MANIFESTS on
the VM (CPU-only; the only network calls are scoped HF single-file staging +
the ~30 sync Sonnet near-twin datagen calls):

- ``config/prefix_panel.json``     50-member prefix panel (8 families), every
                                   member rendered through the ONE construction
                                   path (``_build_generation_prompts``), recipe-
                                   sha-distinct, trained+negative families
                                   byte-asserted vs their sha-pinned training
                                   mixes (``_assert_mix_row_matches_context``).
- ``config/queries.json``          fixed 60-item query draw from the PINNED
                                   #1768 val+test block (seed 1979, stratified
                                   by length quintile; the pinned block is
                                   LMSYS-only by construction, so the plan's
                                   corpus axis is degenerate — recorded).
- ``config/wmap_selection.json``   which #1900 ``wmap_*_L19.pt`` generation is
                                   selected (judge-row-excluded split marker +
                                   n_val==800; fail-loud on ambiguity) + the
                                   counted overlap between the 60 queries and
                                   each map's fit rows.
- ``config/arms.json``             verbatim copy of the pinned #1900 arm config.
- ``config/panel_render_report.json``  realized lengths, screens, budgets,
                                   redraw log, datagen yield, file sha256s.

Everything is fail-loud (no silent defaults); staged inputs are sha-pinned at
the plan §10 revisions. Smoke: ``--panel-limit 4 --query-limit 8
--skip-datagen --scan-cap 1500 --out-root /tmp/...`` through the SAME code
paths (`--import-check` additionally resolves every deferred import).
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before any tokenizer/torch import: thread caps + HF credentials

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import issue1768_capture as CAP  # noqa: E402
import issue1768_cells as X  # noqa: E402
import issue810_common as I810  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1979_prep")

# ── pins + constants (plan v2 §10 Reused artifacts; §4 panel/query tables) ────

SEED = 1979
HF_PREFIX_1979 = "issue1979_prefixrace"
CORPUS_PIN = "c07267285d2cdbf3e0401ddc3e3accae50e496a7"  # issue1768_mapshift round-1 inputs
I1900_PIN = "3bb20debe2e68392897d6144b9180c8748c7afcb"  # issue1900_leakrace config + maps
R4_PIN = "ea39da308e80ccaaefb30e25d563fadc9b0e4dee"  # on_target_r4 prefix_ladder.json
R5_PIN = "5fab2a6f6a26754733da2c061b9027cc4926583a"  # on_target_r5 prefix_ladder_r5.json

N_PANEL = 50
N_QUERIES = 60
N_QUINTILES = 5

# conv-fresh: the r4 lad_build recipe generalized to 8 log-spaced length targets
CF_N = 8
CF_T_MIN = 25.0
CF_T_MAX = 730.0

# near-twins of persona_software_engineer (plan §4 panel table examples)
NEAR_TWIN_OCCUPATIONS = (
    ("nt_backend_engineer", "backend engineer"),
    ("nt_data_engineer", "data engineer"),
    ("nt_devops_engineer", "DevOps engineer"),
    ("nt_technical_writer", "technical writer"),
    ("nt_math_teacher", "math teacher"),
    ("nt_chef", "chef"),
)
NT_MAX_ROUNDS = 5  # 6 twins x 5 rounds = the plan's ~30 sync calls ceiling
NT_LEN_TOL = 0.20
DATAGEN_MODEL = "claude-sonnet-4-5-20250929"

ICL_N_PREFIXES = 2
ICL_DEMOS_PER_PREFIX = 3
ICL_TURN_CHAR_CAP = CAP.LAD_TURN_CONTENT_CAP  # 2000-char parity with the conv screens

BYSTANDER_IDS = (
    "persona_villain",
    "qwen_default_system",
    "qt_rephrase_curious",
    "qt_rephrase_formal",
    "prefix_cooking_smalltalk",
    "prefix_coding_help",
    "adv_roleplay_stage",
    "adv_hypothetical_wrap",
    "bare_wildchat_random",
)
N_BATTERY = 10

# The binding per-render prompt budget: content decode needs prompt+1024 <= 4096
# and mk decode needs prompt+2048 <= 6144 (the recorded #1768 deviation), so
# the binding cap is min(4096-1024, 6144-2048) = 3072 rendered prompt tokens.
BINDING_PROMPT_BUDGET = min(
    X.MAX_MODEL_LEN - X.MAX_NEW_CONTENT, X.PFX_MAX_MODEL_LEN_RAISED - X.MAX_NEW_MARKER
)

WMAP_EXPECTED_SPLIT_PREFIX = "judge rows excluded from tr/val"
WMAP_EXPECTED_N_VAL = 800
# Plan §10 quoted n_train=11,600 for the MAIN generation; the realized pinned
# sidecars carry n_tr=36,400 (3 siblings x 12,400). Recorded as a measured
# plan-figure deviation, NOT hard-asserted (the sidecar split marker + n_val
# are the generation identity).
WMAP_PLAN_FIGURE_N_TRAIN = 11_600


@dataclasses.dataclass
class Cfg:
    """F0 run configuration (CLI-resolved; smoke = limits + scratch out_root)."""

    out_root: Path
    work: Path
    panel_limit: int | None = None
    query_limit: int | None = None
    skip_datagen: bool = False
    skip_upload: bool = False
    scan_cap: int = CAP.LAD_SCAN_ROWS
    force: bool = False

    @property
    def limited(self) -> bool:
        return self.panel_limit is not None or self.query_limit is not None

    @property
    def config_dir(self) -> Path:
        return self.out_root / "config"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _shim(context_id: str, system, prefix_turns, user_wrap) -> types.SimpleNamespace:
    """A Context-shaped shim carrying exactly the 4 render/recipe fields the
    reused #1768 helpers read (`_pfx_prefix_sha`/`_lad_content_sha`/`_pfx_budget`)."""
    return types.SimpleNamespace(
        context_id=context_id,
        system=system,
        prefix_turns=tuple(dict(t) for t in prefix_turns),
        user_wrap=user_wrap,
    )


def _member(prefix_id: str, family: str, tier: str, source: str, ctx, provenance: dict) -> dict:
    return {
        "prefix_id": prefix_id,
        "family": family,
        "tier": tier,
        "source": source,
        "system": ctx.system,
        "prefix_turns": [dict(t) for t in ctx.prefix_turns],
        "user_wrap": ctx.user_wrap,
        "recipe_sha256": CAP._pfx_prefix_sha(ctx),
        "content_sha256": CAP._lad_content_sha(ctx),
        "provenance": provenance,
    }


def _member_ctx(m: dict) -> types.SimpleNamespace:
    return _shim(m["prefix_id"], m["system"], m["prefix_turns"], m["user_wrap"])


def _content_token_len(tok, m: dict) -> int:
    parts = [m["system"] or ""] + [t["content"] for t in m["prefix_turns"]]
    if m["user_wrap"]:
        parts.append(m["user_wrap"].replace("{q}", ""))
    return sum(len(tok(p, add_special_tokens=False)["input_ids"]) for p in parts if p)


# ── input staging (scoped single-file downloads at the plan §10 pins) ─────────


def stage_inputs(cfg: Cfg) -> dict[str, Path]:
    """Stage every F0 input at its pin; returns local paths (fail-loud)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    inputs = cfg.work / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    plan_pins = [
        ("corpus_sample", f"{X.HF_PREFIX}/inputs/corpus_sample.json", CORPUS_PIN),
        ("arm_registry", f"{X.HF_PREFIX}/arm_registry.json", CORPUS_PIN),
        (
            "raw_shard0",
            f"{X.HF_PREFIX}/corpus_capture/base_content/raw_rows_0000.jsonl",
            CORPUS_PIN,
        ),
        ("pfx_sample", f"{X.HF_PREFIX}/on_target/inputs/corpus_sample_pfx.json", None),
        ("r4_ladder", f"{X.HF_PREFIX}/on_target_r4/inputs/prefix_ladder.json", R4_PIN),
        ("r5_panel", f"{X.HF_PREFIX}/on_target_r5/inputs/prefix_ladder_r5.json", R5_PIN),
        ("arms", "issue1900_leakrace/config/arms.json", I1900_PIN),
        ("subset", "issue1900_leakrace/config/subset.json", I1900_PIN),
    ]
    for key, path, rev in plan_pins:
        staged[key] = hub.stage_hub_file(
            X.HF_DATA_REPO, path, inputs / Path(path).name, repo_type="dataset", revision=rev
        )
    # wmap json sidecars (12 pairs at the pin; .pt payloads are F1 inputs, not F0)
    api = HfApi()
    tree = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient at this call site (scoped listing)
            api.list_repo_tree(
                X.HF_DATA_REPO,
                path_in_repo="issue1900_leakrace/maps",
                repo_type="dataset",
                recursive=True,
                revision=I1900_PIN,
            )
        ),
        what="issue1900 maps listing",
    )
    map_files = [e.path for e in tree]
    wmap_dir = cfg.work / "wmap_json"
    for p in sorted(map_files):
        name = Path(p).name
        if name.startswith("wmap_") and name.endswith(".json"):
            staged[f"wmap_json/{name}"] = hub.stage_hub_file(
                X.HF_DATA_REPO, p, wmap_dir / name, repo_type="dataset", revision=I1900_PIN
            )
    staged["_map_files"] = map_files  # type: ignore[assignment]
    logger.info("[stage] %d inputs staged under %s", len(staged), cfg.work)
    return staged


# ── panel families ────────────────────────────────────────────────────────────


def trained_members() -> list[dict]:
    out = []
    for cid in ("persona_software_engineer", "wildchat_prefix_real545", "icl_prefix_sycophancy"):
        ctx = X.pfx_resolve_context(cid)
        out.append(
            _member(
                cid,
                "trained",
                "trained artifact",
                "registry render (pfx_resolve_context)",
                ctx,
                {"context_id": cid},
            )
        )
    bare = _shim("bare", None, (), None)
    out.append(_member("bare", "trained", "trained artifact", "empty prefix (bare arms)", bare, {}))
    return out


def negative_members() -> list[dict]:
    from explore_persona_space.artifacts import negatives as neg_mod

    out = []
    for neg in neg_mod.get_panel(neg_mod.DEFAULT_PANEL_NAME):
        ctx = neg.to_context()
        out.append(
            _member(
                ctx.context_id,
                "negatives",
                "trained artifact",
                "artifacts.negatives default_v1 panel",
                ctx,
                {"identity": neg.identity},
            )
        )
    assert len(out) == 5, [m["prefix_id"] for m in out]
    return out


def bystander_members() -> list[dict]:
    from explore_persona_space.artifacts.context import CONTEXTS

    out = []
    for cid in BYSTANDER_IDS:
        ctx = CONTEXTS[cid]  # KeyError = fail loud (registry drift)
        out.append(
            _member(
                cid,
                "bystander",
                "established instrument",
                "artifacts.context CONTEXTS",
                ctx,
                {"kind": ctx.kind, "registry_family": ctx.family},
            )
        )
    return out


def battery_members(prior: list[dict]) -> tuple[list[dict], dict]:
    """Stratified 10-member draw across the battery's 7 families (seed 1979),
    content-sha-deduped against the rows above (replacement from the same
    family pool; fail-loud on pool exhaustion)."""
    import numpy as np

    blob = I810.load_battery50()
    fam_of = I810.battery_family_map()  # 50 ids / 7 families, fail-loud
    by_fam: dict[str, list[dict]] = {}
    for inst in blob["instances"]:
        by_fam.setdefault(fam_of[str(inst["id"])], []).append(inst)
    sizes = {f: len(v) for f, v in by_fam.items()}
    # quota: 1 per family + the remaining (10-7) to the largest families
    quotas = dict.fromkeys(sorted(by_fam), 1)
    for f in sorted(by_fam, key=lambda f: (-sizes[f], f))[: N_BATTERY - len(by_fam)]:
        quotas[f] += 1
    assert sum(quotas.values()) == N_BATTERY, quotas
    prior_content = {m["content_sha256"] for m in prior}
    rng = np.random.default_rng(SEED)
    out: list[dict] = []
    skipped_dup: list[str] = []
    for fam in sorted(by_fam):
        pool = by_fam[fam]
        order = rng.permutation(len(pool))
        kept = 0
        for j in order:
            if kept >= quotas[fam]:
                break
            inst = pool[int(j)]
            ctx = _shim(
                str(inst["id"]), inst.get("system_prompt"), inst.get("prefix_messages") or (), None
            )
            m = _member(
                str(inst["id"]),
                "battery",
                "established instrument",
                "issue810_common.load_battery50 (sha-pinned)",
                ctx,
                {"battery_family": fam, "label": inst.get("label")},
            )
            if m["content_sha256"] in prior_content:
                skipped_dup.append(m["prefix_id"])
                continue
            prior_content.add(m["content_sha256"])
            out.append(m)
            kept += 1
        if kept < quotas[fam]:
            raise RuntimeError(
                f"battery family {fam!r}: only {kept}/{quotas[fam]} content-distinct "
                f"members (pool {sizes[fam]}, dup-skipped {skipped_dup})"
            )
    return out, {"quotas": quotas, "dedup_skipped": skipped_dup}


def conv_ladder_members(staged: dict[str, Path]) -> list[dict]:
    """r4 rungs (3, verbatim @ pin) + r5 prefixes (3, verbatim @ pin), with the
    ladder files' own sha/shape re-asserts."""
    out = []
    r4 = json.loads(staged["r4_ladder"].read_text())
    for cond in X.R4_CONDS:
        rec = r4["rungs"][cond]
        ctx = _shim(rec["context_id"], None, rec["prefix_turns"], None)
        assert CAP._pfx_prefix_sha(ctx) == rec["recipe_sha256"], (cond, "r4 recipe sha drift")
        assert CAP.lad_turns_sha(rec["prefix_turns"]) == rec["turns_sha256"], (cond, "r4 turns sha")
        out.append(
            _member(
                rec["context_id"],
                "conv-neutral",
                "tier 1 (real WildChat)",
                f"on_target_r4 prefix_ladder.json @ {R4_PIN[:10]}",
                ctx,
                {
                    "cond": cond,
                    "conversation_hash": rec["conversation_hash"],
                    "realized_tokens": rec["realized_tokens"],
                },
            )
        )
    r5 = json.loads(staged["r5_panel"].read_text())
    prefixes = r5["prefixes"]
    assert sorted(prefixes) == sorted(X.R5_CONDS), sorted(prefixes)
    for cond in X.R5_CONDS:
        rec = prefixes[cond]
        ctx = _shim(rec["context_id"], None, rec["prefix_turns"], None)
        assert CAP._pfx_prefix_sha(ctx) == rec["recipe_sha256"], (cond, "r5 recipe sha drift")
        assert CAP.lad_turns_sha(rec["prefix_turns"]) == rec["turns_sha256"], (cond, "r5 turns sha")
        out.append(
            _member(
                rec["context_id"],
                "conv-behavior",
                "tier 1 (real datagen pool, never-trained)",
                f"on_target_r5 prefix_ladder_r5.json @ {R5_PIN[:10]}",
                ctx,
                {
                    "cond": cond,
                    "request_ids": rec["request_ids"],
                    "realized_tokens": rec["realized_tokens"],
                },
            )
        )
    return out


# ── conv-fresh: the lad_build recipe at 8 log-spaced length targets ──────────


def cf_band_specs() -> dict[str, dict]:
    """8 log-spaced targets in [25, 730] content tokens; band = target x/÷ the
    half log-gap multiplier, so bands tile without overlap (recorded)."""
    ratio = (CF_T_MAX / CF_T_MIN) ** (1.0 / (CF_N - 1))
    m = math.sqrt(ratio)
    out = {}
    for i in range(CF_N):
        t = CF_T_MIN * ratio**i
        out[f"cf{i + 1}"] = {"target": t, "lo": t / m, "hi": t * m}
    return out


def _cf_bands_for(t_tokens: int, specs: dict[str, dict]) -> list[str]:
    return [c for c in specs if specs[c]["lo"] <= t_tokens <= specs[c]["hi"]]


def cf_scan(cfg: Cfg, tok, specs, excl, sha_set, banned_hashes: set[str]):
    """Deterministic WildChat-1M stream scan for the 8 fresh bands — the
    `_lad_scan` recipe generalized (same screens, cursor checkpointing,
    fingerprint-gated resume; candidate texts never hit logs)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    revision = hub.retry_transient(
        lambda: HfApi().dataset_info(CAP.WILDCHAT_DATASET).sha,
        what="WildChat-1M revision resolve",
    )
    fp = hashlib.sha256(
        json.dumps(
            {
                "dataset": CAP.WILDCHAT_DATASET,
                "revision": revision,
                "bands": specs,
                "cap": CAP.LAD_TURN_CONTENT_CAP,
                "screens": (
                    "english+toxicF+redactedF+2turn+ua+nonempty"
                    f"+latin{CAP.LAD_CONTENT_LATIN_MIN_RATIO}+excl123+bannedhash"
                ),
                "trained_shas": excl["trained_shas"],
                "banned_hashes": sorted(banned_hashes),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    cursor_path = cfg.work / "cf_scan" / "cursor.json"
    pools: dict[str, list[dict]] = {c: [] for c in specs}
    counters: dict[str, int] = {}
    start = 0
    if cursor_path.exists():
        prev = json.loads(cursor_path.read_text())
        if (
            prev.get("fingerprint") == fp
            and prev.get("rows_scanned", 0) <= CAP.LAD_SCAN_ROWS_WIDENED
        ):
            pools = {c: list(prev["pools"].get(c, [])) for c in specs}
            counters = dict(prev["counters"])
            start = int(prev["rows_scanned"])
            logger.info("[cf_scan] cursor resume at row %d (fp %s)", start, fp)
        else:
            logger.info("[cf_scan] cursor fingerprint mismatch — fresh scan")
    if start >= cfg.scan_cap:
        return pools, counters, start, revision

    from datasets import load_dataset

    ds = load_dataset(CAP.WILDCHAT_DATASET, split="train", streaming=True, revision=revision)
    if start:
        ds = ds.skip(start)
    n = start
    t0 = time.time()

    def _checkpoint() -> None:
        CAP._atomic_json(
            cursor_path,
            {
                "fingerprint": fp,
                "revision": revision,
                "rows_scanned": n,
                "pools": pools,
                "counters": counters,
                **CAP._meta(),
            },
        )

    row = None
    for row in ds:
        n += 1
        reason = CAP.lad_screen_reject(row)
        if reason is None:
            conv = row["conversation"]
            c0 = (conv[0].get("content") or "")[: CAP.LAD_TURN_CONTENT_CAP]
            c1 = (conv[1].get("content") or "")[: CAP.LAD_TURN_CONTENT_CAP]
            t_tokens = len(tok(c0, add_special_tokens=False)["input_ids"]) + len(
                tok(c1, add_special_tokens=False)["input_ids"]
            )
            bands = _cf_bands_for(t_tokens, specs)
            if not bands:
                reason = "out_of_band"
            elif str(row["conversation_hash"]) in banned_hashes:
                reason = "banned_source_conversation"
            else:
                reason = CAP.lad_exclusion_reject(
                    c0, c1, conv[0].get("content") or "", excl, sha_set
                )
            if reason is None:
                cand = {
                    "index": n - 1,
                    "conversation_hash": str(row["conversation_hash"]),
                    "T": t_tokens,
                    "turns": [
                        {"role": "user", "content": c0},
                        {"role": "assistant", "content": c1},
                    ],
                }
                for band in bands:
                    counters[f"band_{band}"] = counters.get(f"band_{band}", 0) + 1
                    pool = pools[band]
                    pool.append(
                        {**cand, "dist": abs(math.log(t_tokens) - math.log(specs[band]["target"]))}
                    )
                    pool.sort(key=lambda c: (c["dist"], c["index"]))
                    del pool[CAP.LAD_BAND_TOPK :]
        if reason is not None:
            counters[reason] = counters.get(reason, 0) + 1
        if n % CAP.LAD_SCAN_CHECKPOINT_EVERY == 0:
            _checkpoint()
        if n % 10_000 == 0:
            kept = sum(counters.get(f"band_{c}", 0) for c in specs)
            print(
                f"[cf_scan] scanned {n}/{cfg.scan_cap} kept={kept} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        if n >= cfg.scan_cap:
            break
    del row, ds  # #952: release the streaming dataset pre-shutdown
    gc.collect()
    _checkpoint()
    print(
        f"[cf_scan] done: scanned={n} rejects="
        f"{json.dumps({k: v for k, v in sorted(counters.items()) if not k.startswith('band_')})}",
        flush=True,
    )
    return pools, counters, n, revision


def conv_fresh_members(cfg: Cfg, tok, staged, prior: list[dict]) -> tuple[list[dict], dict]:
    """8 fresh WildChat conversation prefixes via the generalized lad_build
    recipe (all five never-trained/content-novel screens inherited, plus
    source-conversation + content-sha disjointness vs the r4/r5 members and
    every prior panel member)."""
    specs = cf_band_specs()
    excl = CAP._lad_trained_exclusion_material()
    r1 = json.loads(staged["corpus_sample"].read_text())
    assert r1["n_train"] == X.N_TRAIN, (r1["n_train"], "full-grain r1 required")
    sha_set = {r["sha"] for r in r1["rows"]}
    pfx = json.loads(staged["pfx_sample"].read_text())
    assert (
        pfx["n_train"] == X.PFX_N_TRAIN and len(pfx["rows"]) == X.PFX_N_TRAIN + X.N_VAL + X.N_TEST
    )
    query_texts = [r["prompt"] for r in pfx["rows"]]
    r4 = json.loads(staged["r4_ladder"].read_text())
    banned_hashes = {r4["rungs"][c]["conversation_hash"] for c in X.R4_CONDS}
    banned_content = set(excl["trained_content_shas"].values())
    banned_content |= {m["content_sha256"] for m in prior}

    pools, counters, n_scanned, revision = cf_scan(cfg, tok, specs, excl, sha_set, banned_hashes)

    out: list[dict] = []
    used_hashes: set[str] = set(banned_hashes)
    spares: dict[str, list[dict]] = {}
    shortage: list[str] = []
    for cond in specs:
        picked, extra = None, []
        for cand in sorted(pools[cond], key=lambda c: (c["dist"], c["index"])):
            if cand["conversation_hash"] in used_hashes:
                continue
            c0, c1 = (t["content"] for t in cand["turns"])
            if CAP.lad_substring_belt_hit(c0, c1, query_texts):
                counters["belt_query_text_substring"] = (
                    counters.get("belt_query_text_substring", 0) + 1
                )
                continue
            shim = _shim(f"conv_fresh_{cond}", None, cand["turns"], None)
            if CAP._lad_content_sha(shim) in banned_content:
                counters["content_sha_collision"] = counters.get("content_sha_collision", 0) + 1
                continue
            if picked is None:
                picked = cand
                used_hashes.add(cand["conversation_hash"])
                banned_content.add(CAP._lad_content_sha(shim))
            elif len(extra) < 3:
                extra.append(
                    {
                        "conversation_hash": cand["conversation_hash"],
                        "index": cand["index"],
                        "T": cand["T"],
                    }
                )
        if picked is None:
            shortage.append(cond)
            continue
        spares[cond] = extra
        ctx = _shim(f"conv_fresh_{cond}", None, picked["turns"], None)
        out.append(
            _member(
                f"conv_fresh_{cond}",
                "conv-fresh",
                "tier 1 (real WildChat)",
                f"WildChat-1M @ {revision[:10]} (lad_build recipe, 8 log-spaced targets)",
                ctx,
                {
                    "cond": cond,
                    "conversation_hash": picked["conversation_hash"],
                    "dataset_index": picked["index"],
                    "realized_tokens": picked["T"],
                    "target_tokens": specs[cond]["target"],
                    "band": [specs[cond]["lo"], specs[cond]["hi"]],
                },
            )
        )
    if shortage and cfg.scan_cap >= CAP.LAD_SCAN_ROWS:
        raise RuntimeError(
            f"conv-fresh shortage at production scan cap {cfg.scan_cap}: {shortage} "
            f"(widen --scan-cap toward {CAP.LAD_SCAN_ROWS_WIDENED}, kill-criterion class)"
        )
    if shortage:
        logger.warning(
            "[cf] shortage under reduced scan cap %d (smoke-tolerated): %s", cfg.scan_cap, shortage
        )
    report = {
        "specs": specs,
        "revision": revision,
        "rows_scanned": n_scanned,
        "counters": counters,
        "spares": spares,
        "shortage": shortage,
    }
    return out, report


# ── near-twin persona datagen (~30 sync Sonnet calls; --skip-datagen skips) ──


def near_twin_members(cfg: Cfg, tok, prior: list[dict]) -> tuple[list[dict], dict]:
    """6 Sonnet-written near-twin persona prompts of persona_software_engineer:
    template-matched, token length within ±20% of the anchor, byte-distinct
    from every trained/negative prefix. Sync via api_dispatch (plan §4)."""
    import asyncio

    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    anchor = X.pfx_resolve_context("persona_software_engineer")
    assert anchor.system, "trained persona lost its system string"
    anchor_len = len(tok(anchor.system, add_special_tokens=False)["input_ids"])
    lo, hi = anchor_len * (1 - NT_LEN_TOL), anchor_len * (1 + NT_LEN_TOL)
    banned = {m["content_sha256"] for m in prior}
    system_msg = (
        "You write assistant persona system prompts. Given an example persona system "
        "prompt and a target occupation, write a NEW system prompt for the target "
        "occupation that matches the example's structure, register, and length as "
        "closely as possible. Output ONLY the new system prompt text — no preamble, "
        "no quotes, no commentary."
    )

    def build_request(item: DispatchItem) -> dict:
        return {
            "model": DATAGEN_MODEL,
            "max_tokens": 1000,
            "temperature": 1.0,
            "system": system_msg,
            "messages": [{"role": "user", "content": item.payload["user_msg"]}],
        }

    twins: dict[str, dict] = {}
    n_calls = 0
    attempts_log: list[dict] = []
    for rnd in range(1, NT_MAX_ROUNDS + 1):
        pending = [(slug, occ) for slug, occ in NEAR_TWIN_OCCUPATIONS if slug not in twins]
        if not pending:
            break
        items = []
        for slug, occ in pending:
            user_msg = (
                f"Example persona system prompt:\n---\n{anchor.system}\n---\n\n"
                f"Target occupation: {occ}.\n"
                f"Match the example's template and aim for about {anchor_len} tokens "
                f"(roughly {len(anchor.system)} characters). Output only the prompt text."
                + (
                    f"\n(Attempt {rnd}: the previous draft missed the length band — "
                    f"stay within ±20% of the example's length.)"
                    if rnd > 1
                    else ""
                )
            )
            items.append(DispatchItem(item_id=f"nt-{slug}-r{rnd}", payload={"user_msg": user_msg}))
        n_calls += len(items)
        results = asyncio.run(
            dispatch_calls(
                items,
                model=DATAGEN_MODEL,
                build_request=build_request,
                parse_response=lambda text: text.strip(),
                force_path="sync",
                cache_dir=cfg.work / "nt_cache",
            )
        )
        for slug, _occ in pending:
            res = results[f"nt-{slug}-r{rnd}"]
            if res.error:
                attempts_log.append({"slug": slug, "round": rnd, "outcome": f"error: {res.reason}"})
                continue
            text = str(res.result).strip()
            t_len = len(tok(text, add_special_tokens=False)["input_ids"])
            shim = _shim(slug, text, (), None)
            csha = CAP._lad_content_sha(shim)
            if not text:
                attempts_log.append({"slug": slug, "round": rnd, "outcome": "empty"})
            elif not (lo <= t_len <= hi):
                attempts_log.append(
                    {
                        "slug": slug,
                        "round": rnd,
                        "outcome": f"len {t_len} outside [{lo:.0f}, {hi:.0f}]",
                    }
                )
            elif csha in banned:
                attempts_log.append(
                    {"slug": slug, "round": rnd, "outcome": "content sha collision"}
                )
            else:
                banned.add(csha)
                twins[slug] = {"system": text, "token_len": t_len, "round": rnd}
                attempts_log.append(
                    {"slug": slug, "round": rnd, "outcome": "accepted", "token_len": t_len}
                )
    missing = [slug for slug, _ in NEAR_TWIN_OCCUPATIONS if slug not in twins]
    if missing:
        raise RuntimeError(
            f"near-twin datagen: {missing} unfilled after {NT_MAX_ROUNDS} rounds "
            f"({n_calls} calls; log: {attempts_log})"
        )
    out = []
    for slug, occ in NEAR_TWIN_OCCUPATIONS:
        rec = twins[slug]
        ctx = _shim(slug, rec["system"], (), None)
        out.append(
            _member(
                slug,
                "near-twin",
                "tier 3 (justified in plan §4: graded-similarity probes of the "
                "trained persona exist in no corpus)",
                f"Sonnet datagen ({DATAGEN_MODEL}, sync via api_dispatch)",
                ctx,
                {
                    "occupation": occ,
                    "token_len": rec["token_len"],
                    "anchor_token_len": anchor_len,
                    "accepted_round": rec["round"],
                },
            )
        )
    return out, {
        "n_calls": n_calls,
        "anchor_token_len": anchor_len,
        "band": [lo, hi],
        "attempts": attempts_log,
    }


# ── icl-fresh: few-shot demos from train rows + banked base greedy completions ─


def icl_fresh_members(staged: dict[str, Path], sample: dict) -> tuple[list[dict], dict]:
    """2 few-shot prefixes: 3 Q/A pairs each; Q from corpus TRAIN rows (sha-
    disjoint from the val/test query pool by the sampler's own invariant),
    A = the base model's banked greedy completion (on-policy demos, from the
    pinned #1768 base_content raw shard — no new GPU generation at F0)."""
    import numpy as np

    n_train = sample["n_train"]
    rows = sample["rows"]
    eligible: list[dict] = []
    seen: set[str] = set()
    with staged["raw_shard0"].open(encoding="utf-8") as fh:  # text-mode, never splitlines
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            missing = [
                k
                for k in ("question_idx", "prompt_sha", "response_text", "finish_reason")
                if k not in r
            ]
            assert not missing, f"raw shard row schema drift — missing {missing}"
            qi = int(r["question_idx"])
            if qi >= n_train or r["finish_reason"] != "stop":
                continue
            srow = rows[qi]
            assert (
                srow["sha"].startswith(r["prompt_sha"])
                or r["prompt_sha"].startswith(srow["sha"])
                or r["prompt_sha"] == srow["sha"]
            ), (qi, "shard/sample sha mismatch")
            q, a = srow["prompt"], (r["response_text"] or "").strip()
            if not a or len(q) > ICL_TURN_CHAR_CAP or len(a) > ICL_TURN_CHAR_CAP:
                continue
            if srow["sha"] in seen:
                continue
            seen.add(srow["sha"])
            eligible.append({"sha": srow["sha"], "question_idx": qi, "q": q, "a": a})
    need = ICL_N_PREFIXES * ICL_DEMOS_PER_PREFIX
    assert len(eligible) >= need, (len(eligible), f"raw shard 0 yields < {need} eligible demos")
    rng = np.random.default_rng(SEED)
    chosen = [eligible[int(j)] for j in rng.permutation(len(eligible))[:need]]
    out = []
    for i in range(ICL_N_PREFIXES):
        demos = chosen[i * ICL_DEMOS_PER_PREFIX : (i + 1) * ICL_DEMOS_PER_PREFIX]
        turns = []
        for d in demos:
            turns.append({"role": "user", "content": d["q"]})
            turns.append({"role": "assistant", "content": d["a"]})
        pid = f"icl_fresh_{i + 1}"
        ctx = _shim(pid, None, turns, None)
        out.append(
            _member(
                pid,
                "icl-fresh",
                "tier 1 prompts + on-policy text",
                "corpus train rows + banked base_content greedy completions "
                f"(raw_rows_0000.jsonl @ {CORPUS_PIN[:10]})",
                ctx,
                {
                    "demo_shas": [d["sha"] for d in demos],
                    "question_idx": [d["question_idx"] for d in demos],
                },
            )
        )
    return out, {"pool_size": len(eligible), "chosen_shas": [d["sha"] for d in chosen]}


# ── trained/negative byte-asserts vs the sha-pinned training mixes ────────────


def _row_matches(ctx, msgs: list[dict]) -> bool:
    """Search predicate over `_assert_mix_row_matches_context` (the aggregate
    result stays fail-loud: zero matches raises at the call site)."""
    try:
        CAP._assert_mix_row_matches_context(ctx, msgs, "probe")
    except AssertionError:
        return False
    return True


def _iter_mix_rows(path: Path):
    with path.open(encoding="utf-8") as fh:  # text-mode iteration, never splitlines
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _row_msgs(r: dict) -> list[dict]:
    p = r["prompt"]
    return p if isinstance(p, list) else [{"role": "user", "content": p}]


def mix_byte_asserts(cfg: Cfg, staged: dict[str, Path]) -> dict:
    """Kill-criterion probe (plan §7.1 / §12 A2): trained prefixes byte-match
    their pinned mix POSITIVE rows; every default_v1 negative byte-matches at
    least one NEGATIVE row of at least one family train mix. Full grain (runs
    identically under smoke — the asserts read the pinned mix files)."""
    from explore_persona_space.artifacts import negatives as neg_mod
    from explore_persona_space.orchestrate import hub

    reg = json.loads(staged["arm_registry"].read_text())
    report: dict[str, dict] = {}
    negatives = [(n.slug, n.to_context()) for n in neg_mod.get_panel(neg_mod.DEFAULT_PANEL_NAME)]
    neg_matched: dict[str, list[str]] = {slug: [] for slug, _ in negatives}
    for tag, fam_arm in sorted(CAP.PFX_MIX_FAMILY_ARM.items()):
        src = reg["mix_pos_sources"][fam_arm]
        cid = X.pfx_unit_context_id(f"base_content@{tag}")
        ctx = X.pfx_resolve_context(cid)
        pos_local = cfg.work / "mix_assert" / tag / Path(src["pos_path"]).name
        hub.stage_hub_file(X.HF_DATA_REPO, src["pos_path"], pos_local, repo_type="dataset")
        checked = 0
        for r in _iter_mix_rows(pos_local):
            CAP._assert_mix_row_matches_context(ctx, _row_msgs(r), f"{tag}/{fam_arm}/pos")
            checked += 1
            if checked >= 5:
                break
        assert checked > 0, (tag, src["pos_path"], "no positive rows checked")
        # negatives: scan the family's full train mix for panel-member renders
        mix_prefix = src["mix_prefix"]
        cands = [f"{mix_prefix}/train_mix.jsonl", f"{mix_prefix}/mix/train_mix.jsonl"]
        from huggingface_hub import HfApi

        api = HfApi()
        mix_path = next(
            (
                c
                for c in cands
                if hub.retry_transient(
                    # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient at this call site (probe loop)
                    lambda c=c: api.file_exists(X.HF_DATA_REPO, c, repo_type="dataset"),
                    what=f"train_mix probe {c}",
                )
            ),
            None,
        )
        assert mix_path is not None, (tag, f"no train_mix.jsonl under {mix_prefix}")
        mix_local = cfg.work / "mix_assert" / tag / "train_mix.jsonl"
        hub.stage_hub_file(X.HF_DATA_REPO, mix_path, mix_local, repo_type="dataset")
        fam_neg_hits = 0
        pending = {slug for slug, _ in negatives}
        for r in _iter_mix_rows(mix_local):
            if not pending:
                break
            msgs = _row_msgs(r)
            for slug, nctx in negatives:
                if slug in pending and _row_matches(nctx, msgs):
                    neg_matched[slug].append(tag)
                    pending.discard(slug)
                    fam_neg_hits += 1
        assert fam_neg_hits > 0, (tag, mix_path, "no negative-panel row matched in the train mix")
        report[tag] = {
            "fam_arm": fam_arm,
            "pos_path": src["pos_path"],
            "pos_rows_checked": checked,
            "train_mix": mix_path,
            "negatives_matched_here": fam_neg_hits,
        }
    unmatched = [slug for slug, tags in neg_matched.items() if not tags]
    assert not unmatched, (unmatched, "negative panel members matched in NO family train mix")
    report["negatives_matched"] = neg_matched
    return report


# ── query draw + per-(prefix x query) budget with same-stratum redraw ─────────


def draw_queries(staged: dict[str, Path], n_queries: int) -> tuple[list[dict], dict, dict]:
    """Seed-1979 stratified draw from the pinned val+test block. The pinned
    block is LMSYS-only by construction (#779 round1 = LMSYS first-turns), so
    the plan's (corpus x quintile) stratification realizes as length quintiles
    only — recorded, not silently absorbed."""
    import numpy as np

    sample = json.loads(staged["corpus_sample"].read_text())
    n_train = sample["n_train"]
    vt = sample["rows"][n_train:]
    assert all(r["corpus"] == "valtest" for r in vt), "val/test block corpus labels drifted"
    pool: list[dict] = []
    seen: set[str] = set()
    for i, r in enumerate(vt):
        if r["sha"] in seen:  # frozen #779 property: 1,400 -> 1,318 unique
            continue
        seen.add(r["sha"])
        pool.append({"prompt": r["prompt"], "sha": r["sha"], "vt_index": i})
    lens = np.array([len(p["prompt"]) for p in pool])  # char-length quintiles (recorded rule)
    edges = np.quantile(lens, [0.2, 0.4, 0.6, 0.8])
    for p, ln in zip(pool, lens):
        p["quintile"] = int(np.searchsorted(edges, ln, side="right"))
    strata: dict[int, list[dict]] = {
        q: [p for p in pool if p["quintile"] == q] for q in range(N_QUINTILES)
    }
    base, extra = divmod(n_queries, N_QUINTILES)
    rng = np.random.default_rng(SEED)
    drawn: list[dict] = []
    remaining: dict[int, list[dict]] = {}
    for q in range(N_QUINTILES):
        quota = base + (1 if q < extra else 0)
        order = rng.permutation(len(strata[q]))
        picks = [strata[q][int(j)] for j in order]
        assert len(picks) >= quota, (q, len(picks), quota)
        drawn.extend(picks[:quota])
        remaining[q] = picks[quota:]
    meta = {
        "n_pool_unique": len(pool),
        "quintile_edges_chars": [float(e) for e in edges],
        "length_rule": "raw prompt char length quintiles over the deduped 1,318-sha pool",
        "corpus_axis_note": (
            "degenerate: the pinned #779 val/test block is LMSYS-only by "
            "construction (round1 = LMSYS first-turns), so the plan's "
            "(corpus x quintile) stratification realizes as quintiles only"
        ),
        "seed": SEED,
    }
    return drawn, remaining, {"sample": sample, "meta": meta}


def _rendered_lens(tok, ctx, prompts: list[str]) -> list[int]:
    """Per-row rendered-prompt token lengths through the ONE construction path
    (byte-identical to the `_pfx_budget` render)."""
    from explore_persona_space.analysis.representation_shift import _build_generation_prompts

    rendered, _keys = _build_generation_prompts(
        tok,
        {ctx.context_id: ctx.system},
        prompts,
        user_wraps={ctx.context_id: ctx.user_wrap},
        prior_turns={ctx.context_id: tuple(ctx.prefix_turns)},
    )
    return [len(ids) for ids in tok(rendered, add_special_tokens=False)["input_ids"]]


def budget_and_redraw(
    tok,
    members: list[dict],
    queries: list[dict],
    remaining: dict[int, list[dict]],
    *,
    prompt_budget: int = BINDING_PROMPT_BUDGET,
) -> tuple[list[dict], dict, list[dict]]:
    """Per-(prefix x query) token-budget check on the REAL renders; an
    over-budget query is replaced from the same stratum (recorded). After the
    redraw loop, `_pfx_budget` re-runs per member as the recorded (asserting)
    budget table. `prompt_budget` is parameterized so the redraw branch is
    unit-probeable at a degenerate cap (data-dependent-gates duty)."""
    queries = list(queries)
    redraw_log: list[dict] = []
    for _pass in range(10):  # bounded; each pass replaces every over-budget query once
        over: dict[int, dict] = {}
        for m in members:
            ctx = _member_ctx(m)
            lens = _rendered_lens(tok, ctx, [q["prompt"] for q in queries])
            for qi, ln in enumerate(lens):
                if ln > prompt_budget and qi not in over:
                    over[qi] = {"member": m["prefix_id"], "rendered_tokens": ln}
        if not over:
            break
        for qi, why in sorted(over.items()):
            old = queries[qi]
            stratum = old["quintile"]
            if not remaining[stratum]:
                raise RuntimeError(f"stratum {stratum} exhausted during budget redraw ({why})")
            new = remaining[stratum].pop(0)
            redraw_log.append(
                {
                    "replaced_sha": old["sha"],
                    "replacement_sha": new["sha"],
                    "stratum": stratum,
                    "reason": why,
                }
            )
            queries[qi] = new
    else:
        raise RuntimeError(
            f"budget redraw did not stabilize in 10 passes ({len(redraw_log)} swaps)"
        )
    budgets = {}
    rows = [{"prompt": q["prompt"]} for q in queries]
    for k, m in enumerate(members):
        budgets[m["prefix_id"]] = CAP._pfx_budget(tok, _member_ctx(m), rows)
        print(f"[budget] member {k + 1}/{len(members)} {m['prefix_id']} ok", flush=True)
    return queries, budgets, redraw_log


# ── wmap generation selection + query/fit-row overlap ─────────────────────────


def wmap_selection(cfg: Cfg, staged: dict, queries: list[dict]) -> dict:
    """Select the #1900 wmap generation at the pin and count query/fit-row
    overlap. Selection keys on the sidecar identity (judge-row-excluded split
    marker + n_val==800) with exactly one pt+json pair per content arm —
    fail-loud on ambiguity. The plan §10 figure n_train=11,600 does NOT match
    the realized sidecars (n_tr=36,400 = 3 sibling cells x 12,400 non-judge
    rows); recorded as a measured plan-figure deviation."""
    arms_blob = json.loads(staged["arms"].read_text())
    arm_rows = (
        arms_blob["arms"] if isinstance(arms_blob, dict) and "arms" in arms_blob else arms_blob
    )
    assert isinstance(arm_rows, list) and arm_rows, "arms.json shape drift"
    arm_ids = [a["arm_id"] for a in arm_rows]
    content_arms = sorted(a for a in arm_ids if not a.startswith("mk-"))
    marker_arms = sorted(a for a in arm_ids if a.startswith("mk-"))
    assert len(content_arms) == 12 and len(marker_arms) == 6, (len(content_arms), len(marker_arms))

    map_files: list[str] = staged["_map_files"]  # type: ignore[assignment]
    by_arm: dict[str, dict] = {}
    for arm in content_arms:
        pts = [
            p for p in map_files if Path(p).name.startswith(f"wmap_{arm}_L") and p.endswith(".pt")
        ]
        jss = [
            p for p in map_files if Path(p).name.startswith(f"wmap_{arm}_L") and p.endswith(".json")
        ]
        if len(pts) != 1 or len(jss) != 1:
            raise RuntimeError(
                f"wmap generation ambiguous/missing for {arm} at pin {I1900_PIN[:10]}: "
                f"pt={pts} json={jss} — refusing (plan §10 fail-loud rule)"
            )
        side = json.loads(staged[f"wmap_json/{Path(jss[0]).name}"].read_text())
        split = side.get("split", "")
        assert split.startswith(WMAP_EXPECTED_SPLIT_PREFIX), (arm, split)
        assert side["n_val"] == WMAP_EXPECTED_N_VAL, (arm, side["n_val"])
        by_arm[arm] = {
            "pt": pts[0],
            "json": jss[0],
            "n_tr": side["n_tr"],
            "n_val": side["n_val"],
            "n_te": side["n_te"],
            "split": split,
        }

    sample = json.loads(staged["corpus_sample"].read_text())
    sample_shas = {r["sha"] for r in sample["rows"]}
    subset = json.loads(staged["subset"].read_text())
    judge = set(subset["shas"])
    assert len(judge) == subset["n"], (len(judge), subset["n"])
    assert judge <= sample_shas, "judge subset carries shas outside the corpus sample"
    q_shas = {q["sha"] for q in queries}
    overlap_te = len(q_shas & judge)
    overlap_trval = len(q_shas & (sample_shas - judge))
    for arm, rec in by_arm.items():
        assert rec["n_te"] == len(judge), (arm, rec["n_te"], len(judge))
        rec["overlap_queries_fit_trval"] = overlap_trval
        rec["overlap_queries_te_judge"] = overlap_te
    return {
        "pin": I1900_PIN,
        "selection_rule": (
            "exactly one wmap_<arm>_L19.{pt,json} pair per content arm at the pin; sidecar "
            f"identity: split startswith {WMAP_EXPECTED_SPLIT_PREFIX!r} and n_val == "
            f"{WMAP_EXPECTED_N_VAL} (the MAIN judge-row-excluded generation)"
        ),
        "plan_figure_note": (
            f"plan §10 quoted n_train={WMAP_PLAN_FIGURE_N_TRAIN}; realized sidecars carry "
            f"n_tr={sorted({r['n_tr'] for r in by_arm.values()})} (3 sibling corpus cells x "
            "12,400 non-judge rows) — measured deviation recorded, selection keys on the "
            "split marker, not the plan figure"
        ),
        "fit_row_universe": (
            "each wmap fits on its sibling arms' corpus cells over the SAME 16,400-row "
            "corpus sample minus the 4,000 judge rows (te = the target arm's judge rows); "
            "overlap is counted at sha-set level against that universe"
        ),
        "n_queries": len(q_shas),
        "overlap_queries_fit_trval": overlap_trval,
        "overlap_queries_te_judge": overlap_te,
        "arms": by_arm,
        "marker_arms_no_wmap": marker_arms,
        **CAP._meta(),
    }


# ── assembly + outputs ────────────────────────────────────────────────────────


def assemble_panel(cfg: Cfg, tok, staged: dict) -> tuple[list[dict], dict]:
    members: list[dict] = []
    report: dict = {}
    members += trained_members()
    members += negative_members()
    members += bystander_members()
    battery, report["battery"] = battery_members(members)
    members += battery
    members += conv_ladder_members(staged)
    cf, report["conv_fresh"] = conv_fresh_members(cfg, tok, staged, members)
    members += cf
    if cfg.skip_datagen:
        report["near_twin"] = {
            "skipped": True,
            "note": "--skip-datagen: near-twin family OMITTED (never "
            "placeholder data); production F0 must run datagen",
        }
    else:
        nt, report["near_twin"] = near_twin_members(cfg, tok, members)
        members += nt
    sample = json.loads(staged["corpus_sample"].read_text())
    icl, report["icl_fresh"] = icl_fresh_members(staged, sample)
    members += icl

    for m in members:
        m["content_token_len"] = _content_token_len(tok, m)

    # panel invariants (plan §4): recipe-sha distinctness is HARD; content-sha
    # duplicate groups are REPORTED (bare vs neg_default_assistant is by design)
    recipe = {}
    for m in members:
        assert m["recipe_sha256"] not in recipe, (
            m["prefix_id"],
            recipe[m["recipe_sha256"]],
            "duplicate recipe sha",
        )
        recipe[m["recipe_sha256"]] = m["prefix_id"]
    by_content: dict[str, list[str]] = {}
    for m in members:
        by_content.setdefault(m["content_sha256"], []).append(m["prefix_id"])
    report["content_dup_groups"] = sorted(v for v in by_content.values() if len(v) > 1)
    fam_counts: dict[str, int] = {}
    for m in members:
        fam_counts[m["family"]] = fam_counts.get(m["family"], 0) + 1
    report["family_counts"] = fam_counts
    n_expected = N_PANEL - (len(NEAR_TWIN_OCCUPATIONS) if cfg.skip_datagen else 0)
    shortage = len(report["conv_fresh"].get("shortage", []))
    if not cfg.limited and cfg.scan_cap >= CAP.LAD_SCAN_ROWS and not cfg.skip_datagen:
        assert len(members) == N_PANEL, (len(members), fam_counts)
    else:
        assert len(members) == n_expected - shortage, (len(members), n_expected, shortage)
    return members, report


def write_outputs(
    cfg: Cfg, members, queries, budgets, redraw_log, q_meta, wmap, mix_report, panel_report, staged
) -> dict[str, Path]:
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    limits = {
        "panel_limit": cfg.panel_limit,
        "query_limit": cfg.query_limit,
        "skip_datagen": cfg.skip_datagen,
        "scan_cap": cfg.scan_cap,
        "smoke": cfg.limited or cfg.skip_datagen or cfg.scan_cap < CAP.LAD_SCAN_ROWS,
    }
    panel_path = cfg.config_dir / "prefix_panel.json"
    CAP._atomic_json(
        panel_path,
        {
            "members": members,
            "n_members": len(members),
            "seed": SEED,
            "limits": limits,
            "pins": {"corpus": CORPUS_PIN, "i1900": I1900_PIN, "r4": R4_PIN, "r5": R5_PIN},
            **CAP._meta(),
        },
    )
    queries_path = cfg.config_dir / "queries.json"
    CAP._atomic_json(
        queries_path,
        {
            "queries": queries,
            "n": len(queries),
            "seed": SEED,
            "limits": limits,
            "source": f"{X.HF_PREFIX}/inputs/corpus_sample.json @ {CORPUS_PIN} val+test block",
            "redraw_log": redraw_log,
            **q_meta["meta"],
            **CAP._meta(),
        },
    )
    wmap_path = cfg.config_dir / "wmap_selection.json"
    CAP._atomic_json(wmap_path, wmap)
    arms_path = cfg.config_dir / "arms.json"
    arms_path.write_bytes(staged["arms"].read_bytes())  # verbatim copy of the pinned config
    report_path = cfg.config_dir / "panel_render_report.json"
    CAP._atomic_json(
        report_path,
        {
            "panel": panel_report,
            "mix_byte_asserts": mix_report,
            "budgets": budgets,
            "redraw_log": redraw_log,
            "limits": limits,
            "binding_prompt_budget_tokens": BINDING_PROMPT_BUDGET,
            "file_sha256": {
                p.name: _sha256_file(p) for p in (panel_path, queries_path, wmap_path, arms_path)
            },
            **CAP._meta(),
        },
    )
    for p in (panel_path, queries_path, wmap_path, arms_path, report_path):
        print(f"[out] {p} sha256={_sha256_file(p)[:16]} bytes={p.stat().st_size}", flush=True)
    return {
        "panel": panel_path,
        "queries": queries_path,
        "wmap": wmap_path,
        "arms": arms_path,
        "report": report_path,
    }


# UPLOAD_PREFIX_EXEMPT: run complete 2026-08-01; single-issue prep, dest pinned in #1979 footer
def upload_config(cfg: Cfg, outs: dict[str, Path], dest: str = f"{HF_PREFIX_1979}/config") -> None:
    """One upload_folder commit to the canonical lane-safe read path + exact-set
    verify (the #1900 MF1 lesson: every lane stages config from the Hub).
    `dest` is parameterized so the branch is live-probeable at a scratch prefix
    (never the canonical path from a smoke)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    url = hub._upload(cfg.config_dir, X.HF_DATA_REPO, "dataset", dest)
    if not url:
        raise RuntimeError(f"config upload returned no path ({dest}) — refusing")
    expected = [f"{dest}/{p.name}" for p in outs.values()]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), X.HF_DATA_REPO, expected, path_in_repo=dest, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(f"config upload verify: missing on Hub: {missing}")
    logger.info("[upload] %d config files verified at %s", len(expected), dest)


# ── entrypoint ────────────────────────────────────────────────────────────────


def _import_check() -> None:
    """Resolve every deferred import on the real code path (Axis-1 smoke leg)."""
    import asyncio  # noqa: F401

    import numpy  # noqa: F401
    from datasets import load_dataset  # noqa: F401
    from huggingface_hub import HfApi, hf_hub_download  # noqa: F401

    from explore_persona_space.analysis.representation_shift import (  # noqa: F401
        _build_generation_prompts,
    )
    from explore_persona_space.artifacts import negatives  # noqa: F401
    from explore_persona_space.artifacts.context import CONTEXTS, Context  # noqa: F401
    from explore_persona_space.llm.api_dispatch import (  # noqa: F401
        DispatchItem,
        dispatch_calls,
    )
    from explore_persona_space.orchestrate import hub  # noqa: F401

    assert callable(hub.stage_hub_file) and callable(hub.verify_repo_paths_uploaded)
    print("[import-check] ok", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "eval_results" / "issue_1979")
    ap.add_argument(
        "--work-dir", type=Path, default=REPO_ROOT / "data" / "issue_1979" / "hf_dl" / "prep"
    )
    ap.add_argument(
        "--panel-limit",
        type=int,
        default=None,
        help="smoke: budget-phase member slice (production: unset)",
    )
    ap.add_argument(
        "--query-limit", type=int, default=None, help="smoke: query draw size (production: 60)"
    )
    ap.add_argument(
        "--skip-datagen",
        action="store_true",
        help="smoke: omit the near-twin Sonnet datagen family entirely",
    )
    ap.add_argument(
        "--skip-upload", action="store_true", help="skip the HF config mirror upload (smoke)"
    )
    ap.add_argument(
        "--scan-cap",
        type=int,
        default=CAP.LAD_SCAN_ROWS,
        help="conv-fresh WildChat scan row cap (smoke: ~1500)",
    )
    ap.add_argument("--force", action="store_true", help="rebuild over frozen outputs")
    ap.add_argument(
        "--import-check", action="store_true", help="resolve every deferred import and exit 0"
    )
    args = ap.parse_args(argv)
    if args.import_check:
        _import_check()
        sys.stdout.flush()
        sys.exit(0)
    cfg = Cfg(
        out_root=args.out_root,
        work=args.work_dir,
        panel_limit=args.panel_limit,
        query_limit=args.query_limit,
        skip_datagen=args.skip_datagen,
        skip_upload=args.skip_upload,
        scan_cap=args.scan_cap,
        force=args.force,
    )
    panel_path = cfg.config_dir / "prefix_panel.json"
    if panel_path.exists() and not cfg.force:
        prior = json.loads(panel_path.read_text())
        print(
            f"[f0] panel already frozen at {panel_path} (n={prior['n_members']}) — "
            "refusing rebuild without --force",
            flush=True,
        )
        sys.stdout.flush()
        sys.exit(0)

    from transformers import AutoTokenizer

    print("[phase=f0_stage]", flush=True)
    staged = stage_inputs(cfg)
    tok = AutoTokenizer.from_pretrained(X.BASE_MODEL)

    print("[phase=f0_mix_asserts]", flush=True)
    mix_report = mix_byte_asserts(cfg, staged)

    print("[phase=f0_panel]", flush=True)
    members, panel_report = assemble_panel(cfg, tok, staged)
    budget_members = members[: cfg.panel_limit] if cfg.panel_limit else members

    print("[phase=f0_queries]", flush=True)
    n_q = cfg.query_limit or N_QUERIES
    queries, remaining, q_meta = draw_queries(staged, n_q)
    queries, budgets, redraw_log = budget_and_redraw(tok, budget_members, queries, remaining)
    queries_out = [
        {
            "prompt": q["prompt"],
            "sha": q["sha"],
            "quintile": q["quintile"],
            "vt_index": q["vt_index"],
        }
        for q in queries
    ]

    print("[phase=f0_wmap]", flush=True)
    wmap = wmap_selection(cfg, staged, queries)

    print("[phase=f0_write]", flush=True)
    outs = write_outputs(
        cfg,
        members,
        queries_out,
        budgets,
        redraw_log,
        q_meta,
        wmap,
        mix_report,
        panel_report,
        staged,
    )
    if cfg.skip_upload:
        print("[phase=f0_upload] SKIPPED (--skip-upload)", flush=True)
    else:
        print("[phase=f0_upload]", flush=True)
        upload_config(cfg, outs)
    print(
        f"[phase=done] f0 members={len(members)} queries={len(queries)} redraws={len(redraw_log)}",
        flush=True,
    )
    # #1689/#952 class: with ~200 C-extension modules loaded (datasets/pyarrow
    # streaming + torch via the reused capture module), interpreter finalization
    # aborts SIGABRT (PyGILState_Release) AFTER all work completed — measured on
    # this script's own smoke (rc=134 after [phase=done]). Every output above is
    # atomically written (tmp + os.replace) and the upload verify raises in-band,
    # so skipping the finalize path is safe; error paths still raise normally.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
