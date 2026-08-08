"""#1979 F2 VM judge driver — graded 0-100 Batch-API judging in waves (plan v2 §4 F2/§7).

Judges the 13 content states (12 content arms x own rubric + base_content x 3
rubrics = 15 judged units) over the 50-prefix x 60-query grid, N=3 draws at
temperature 1.0, ``max_tokens=400`` passed explicitly, rubric texts pinned
verbatim from #1900 (``issue1900_judge.rubric_for`` — the same
``issue779_common``/``artifacts/behavior.py`` sources; prompt hashes recorded
per llm-judging rule 18). Dispatch rides the #1019 resumable Batch machinery
(``eval/batch_judge.py::judge_completions_batch`` via
``eval/graded_judge.judge_graded``); the per-arm drop report splits CONTENT
drops from TRANSPORT losses (rules 9/24).

Wave structure (plan §7): wave 1 is the Gate-1 pilot — base_content under the
impoliteness rubric + the two impoliteness-persona arms (po + con, s42) over
the full grid (~27k calls); waves 2+ are refused (rc=8) until the persisted
Gate-1 verdict is in the proceed set. Criteria A/B/C per plan §7; a
``marker-lead-rescope`` verdict exits rc=7 (designed artifact-routed halt) and
a criterion-A failure exits rc=7 with the resize branch named (re-drive wave 1
with ``--max-tokens 600`` — a fresh per-(unit, max_tokens) work dir means no
truncation-era checkpoint replay, rule 23).

VM-side phase (plan §10 ``off_pod_phases``): reads F1f judge-input shards from
the HF data repo, writes ``eval_results/issue_1979/judge/arm_scores_*.json``
(git-committed by the orchestrator) + ``judge_done.json`` sentinel.

Smoke (``--smoke-banked N``): pulls N banked #1900 raw completions (real model
text at the 3bb20deb pin), routes them through the PRODUCTION judge path
(``judge_unit`` -> ``judge_graded`` -> batch client) as one micro-batch with
real API calls, and writes to a scratch dir — never the canonical judge dir.
Content hygiene: completions are never printed; digests only.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before numpy/hub: shared-VM thread caps + ANTHROPIC/HF credentials

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1900_judge as J  # noqa: E402  (rubrics + judge_unit + loaders reused verbatim)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1979.judge")

ISSUE = 1979
HF_PREFIX_1979 = "issue1979_prefixrace"
I1900_PIN = "3bb20debe2e68392897d6144b9180c8748c7afcb"
CONFIG_DIR = REPO_ROOT / "eval_results" / "issue_1979" / "config"
JUDGE_DIR = REPO_ROOT / "eval_results" / "issue_1979" / "judge"
WORK_DIR = REPO_ROOT / "data" / "issue_1979" / "judge_work"
N_DRAWS = J.N_DRAWS  # 3 (plan §11)
JUDGE_MAX_TOKENS = J.JUDGE_MAX_TOKENS  # 400, passed explicitly (rule 23)
MAX_TOKENS_RESIZE = J.MAX_TOKENS_RESIZE  # 600 (§7 criterion A branch)

# Gate-1 criteria (plan §7 — thresholds verbatim)
CRIT_A_MAX_DROP_RATE = 0.10
CRIT_B_MIN_SD_LEVEL = 3.0
CRIT_B_MIN_PREFIXES_GE10 = 10
CRIT_B_CHANGE_MIN_SD = 2.0
CRIT_C_CEILING_SCORE = 90.0
CRIT_C_MAX_CEILING_SHARE = 0.40
RC_GATE_REFUSED = 7  # designed halt: gate verdict is not proceed
RC_PILOT_MISSING = 8  # wave >= 2 without a passing persisted Gate-1 verdict

PROCEED_VERDICTS = {"proceed", "proceed-ceiling-flagged", "proceed-change-dv"}


def _meta() -> dict:
    return {
        "script": "scripts/issue1979_judge.py",
        "issue": ISSUE,
        "git_commit": J._git_commit(),
        "ts": J._meta().get("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    }


# ── config manifests (F0 outputs; VM-resident by construction, HF fallback) ──


def _stage_config(name: str) -> Path:
    p = CONFIG_DIR / name
    if not p.exists():
        from explore_persona_space.orchestrate import hub

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        hub.stage_hub_file(
            J._data_repo(), f"{HF_PREFIX_1979}/config/{name}", p, repo_type="dataset"
        )
    return p


def load_config() -> dict:
    """Panel + queries + arms manifests; builds the (prefix, query) join maps."""
    panel = json.loads(_stage_config("prefix_panel.json").read_text())
    queries = json.loads(_stage_config("queries.json").read_text())["queries"]
    arms = json.loads(_stage_config("arms.json").read_text())["arms"]
    assert panel["members"] and queries and arms, "empty F0 manifests"
    q_text = {q["sha"]: q["prompt"] for q in queries}
    q_ix = {q["sha"]: i for i, q in enumerate(queries)}
    prefix_ids = [m["prefix_id"] for m in panel["members"]]
    return {
        "prefix_ids": prefix_ids,
        "queries": queries,
        "q_text": q_text,
        "q_ix": q_ix,
        "arms": arms,
        "content_arms": [a for a in arms if a["kind"] == "content"],
    }


# ── unit specs + waves ────────────────────────────────────────────────────────


def unit_specs(cfg: dict) -> list[dict]:
    """The 15 judged (state x rubric) units (plan §4 F2)."""
    specs = [
        {"state": a["arm_id"], "beh_key": a["beh_key"], "tag": a["arm_id"]}
        for a in cfg["content_arms"]
    ]
    for beh in ("imp", "cas", "syc"):
        specs.append({"state": "base_content", "beh_key": beh, "tag": f"base_{beh}"})
    return specs


def _pilot_arm(cfg: dict, regime: str) -> str:
    """The §7 wave-1 impoliteness persona arm id for one regime (fail-loud)."""
    hits = [
        a["arm_id"]
        for a in cfg["content_arms"]
        if a["beh_key"] == "imp"
        and a["ctx_key"] == "pers"
        and a["regime"] == regime
        and a["seed"] == 42
        and a["method"] == "lora"
    ]
    assert len(hits) == 1, f"wave-1 pilot arm ambiguous for regime={regime}: {hits}"
    return hits[0]


def waves(cfg: dict) -> dict[int, list[str]]:
    """Deterministic wave partition over unit TAGS (wave 1 = §7 Gate-1 pilot).

    Wave 1 = base_content under the imp rubric + the two imp-pers arms (po +
    con, s42) — 3 units x 3,000 rows x 3 draws = 27k calls, matching the §7
    arithmetic (the §7 prose "base_content (3 rubrics)" is inconsistent with
    its own 27k figure and with the criteria, which only read the imp rubric;
    the remaining base rubrics ride wave 2 — stated deviation, see the
    implementation report).
    """
    po, con = _pilot_arm(cfg, "po"), _pilot_arm(cfg, "con")
    wave1 = ["base_imp", po, con]
    imp_rest = sorted(a["arm_id"] for a in cfg["content_arms"] if a["beh_key"] == "imp")
    imp_rest = [t for t in imp_rest if t not in wave1]
    wave2 = ["base_cas", "base_syc", *imp_rest]
    wave3 = sorted(a["arm_id"] for a in cfg["content_arms"] if a["beh_key"] == "cas")
    wave4 = sorted(a["arm_id"] for a in cfg["content_arms"] if a["beh_key"] == "syc")
    out = {1: wave1, 2: wave2, 3: wave3, 4: wave4}
    all_tags = [t for ts in out.values() for t in ts]
    assert len(all_tags) == len(set(all_tags)) == 15, sorted(all_tags)
    return out


# ── items ─────────────────────────────────────────────────────────────────────


def load_state_rows(inputs_dir: Path, state: str) -> list[dict]:
    """All judge-input rows for one state from the F1f shards (fail-loud)."""
    shards = sorted(inputs_dir.glob(f"judge_inputs_{state}.shard*.jsonl"))
    assert shards, f"no judge_inputs shards for {state} under {inputs_dir}"
    rows: list[dict] = []
    for p in shards:
        rows.extend(J._read_jsonl_rows(p))
    for r in rows:
        assert r["state"] == state, (r["state"], state)
        assert "query_sha" in r, "F1f rows must carry query_sha (unit-2 F1f schema)"
    assert rows, (state, "empty judge inputs")
    return rows


def stage_state_inputs(inputs_dir: Path, states: list[str]) -> None:
    """Stage missing per-state shards from the HF mirror (scoped listing, #833)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    missing = [s for s in states if not list(inputs_dir.glob(f"judge_inputs_{s}.shard*.jsonl"))]
    if not missing:
        return
    inputs_dir.mkdir(parents=True, exist_ok=True)
    listing = hub.retry_transient(
        lambda: hub.list_hf_files_under_path(
            HfApi(), J._data_repo(), f"{HF_PREFIX_1979}/judge_inputs", repo_type="dataset"
        ),
        what="judge_inputs scoped listing",
    )
    for state in missing:
        wanted = [
            p
            for p in listing
            if Path(p).name.startswith(f"judge_inputs_{state}.shard") and p.endswith(".jsonl")
        ]
        assert wanted, (state, "no shards on the HF mirror — did F1f run + upload?")
        for p in wanted:
            hub.stage_hub_file(J._data_repo(), p, inputs_dir / Path(p).name, repo_type="dataset")
        logger.info("[stage] %s: %d shard files", state, len(wanted))


def build_items(
    cfg: dict, rows: list[dict], limit: int | None = None
) -> tuple[list[tuple[str, str, str]], dict[str, str], dict[str, dict]]:
    """(item_id, question, answer) triples in deterministic (prefix, query) order.

    item_id = the F1a row sha (16 hex chars — within the 53-char Batch
    custom_id budget, no '__'; #1415/#1776). Returns the identity id map the
    reused ``judge_unit`` persist-join expects plus a sha -> {prefix_id,
    query_sha} join map for the 1979 row augmentation.
    """
    pfx_ix = {p: i for i, p in enumerate(cfg["prefix_ids"])}
    rows = sorted(rows, key=lambda r: (pfx_ix[r["prefix_id"]], cfg["q_ix"][r["query_sha"]]))
    if limit is not None:
        rows = rows[:limit]
    items: list[tuple[str, str, str]] = []
    id_map: dict[str, str] = {}
    join: dict[str, dict] = {}
    for r in rows:
        sha = r["sha"]
        assert sha not in id_map, f"duplicate row sha {sha}"
        assert len(sha) <= J.MAX_ITEM_ID_LEN and "__" not in sha
        id_map[sha] = sha
        join[sha] = {"prefix_id": r["prefix_id"], "query_sha": r["query_sha"]}
        items.append((sha, cfg["q_text"][r["query_sha"]], str(r["response_text"])))
    return items, id_map, join


# ── judging one 1979 unit (reuses issue1900_judge.judge_unit) ─────────────────


def judge_state(
    cfg: dict,
    spec: dict,
    inputs_dir: Path,
    out_dir: Path,
    *,
    wave: int,
    n_draws: int,
    max_tokens: int,
    limit: int | None = None,
    force_batch: bool = False,
) -> dict:
    """Judge one (state x rubric) unit; augment the persisted JSON with the
    (prefix, query) join + 1979 meta. Resume-skips a matching persisted unit."""
    tag = spec["tag"]
    out_path = out_dir / f"arm_scores_{tag}.json"
    rows = load_state_rows(inputs_dir, spec["state"])
    items, id_map, join = build_items(cfg, rows, limit)
    if out_path.exists():
        prior = json.loads(out_path.read_text())
        if (
            prior.get("n_items") == len(items)
            and prior.get("judge", {}).get("n_draws") == n_draws
            and prior.get("judge", {}).get("max_tokens") == max_tokens
        ):
            logger.info("[f2] %s resume-skip (n_items=%d)", tag, len(items))
            return prior
        logger.info("[f2] %s regime changed — re-judging", tag)
    payload = J.judge_unit(
        tag,
        spec["beh_key"],
        items,
        id_map,
        WORK_DIR,
        out_path,
        n_draws=n_draws,
        max_tokens=max_tokens,
        force_batch=force_batch,
    )
    # augment (atomic rewrite): per-row (prefix_id, query_sha) join + 1979 meta
    for r in payload["rows"]:
        r.update(join[r["sha"]])
    payload["meta"].update(_meta())
    payload["state"] = spec["state"]
    payload["wave"] = wave
    J._atomic_json(out_path, payload)
    return payload


# ── Gate 1 (plan §7) ──────────────────────────────────────────────────────────


def _per_prefix_level(payload: dict, prefix_ids: list[str]) -> dict[str, float]:
    """Per-prefix LEVEL mean (mean over queries of per-completion score_mean)."""
    acc: dict[str, list[float]] = {p: [] for p in prefix_ids}
    for r in payload["rows"]:
        if r["score_mean"] is not None:
            acc[r["prefix_id"]].append(float(r["score_mean"]))
    return {p: float(np.mean(v)) for p, v in acc.items() if v}


def gate1_verdict(cfg: dict, out_dir: Path) -> dict:
    """Criteria A/B/C (plan §7) over the persisted wave-1 unit payloads."""
    po, con = _pilot_arm(cfg, "po"), _pilot_arm(cfg, "con")
    payloads = {}
    for tag in ("base_imp", po, con):
        p = out_dir / f"arm_scores_{tag}.json"
        assert p.exists(), f"gate 1 needs wave-1 unit {tag}: {p} missing"
        payloads[tag] = json.loads(p.read_text())

    # Criterion A — instrument health: content-drop rate per judged state
    crit_a = {
        tag: {
            "content_drop_rate": pl["content_drop_rate"],
            "n_transport_lost_draws": pl["n_transport_lost_draws"],
            "pass": pl["content_drop_rate"] is not None
            and pl["content_drop_rate"] <= CRIT_A_MAX_DROP_RATE,
        }
        for tag, pl in payloads.items()
    }
    a_pass = all(v["pass"] for v in crit_a.values())

    # Criterion B — per-prefix DV range on the LEVEL DV (change fallback)
    base_lvl = _per_prefix_level(payloads["base_imp"], cfg["prefix_ids"])
    crit_b_arms = {}
    for tag in (po, con):
        lvl = _per_prefix_level(payloads[tag], cfg["prefix_ids"])
        vals = np.asarray(list(lvl.values()), dtype=np.float64)
        chg = np.asarray([lvl[p] - base_lvl[p] for p in lvl if p in base_lvl], dtype=np.float64)
        crit_b_arms[tag] = {
            "n_prefixes": int(vals.size),
            "sd_level": float(np.std(vals, ddof=1)) if vals.size > 1 else None,
            "n_prefixes_ge10": int((vals >= 10.0).sum()),
            "sd_change": float(np.std(chg, ddof=1)) if chg.size > 1 else None,
        }
    b_level_pass = any(
        (v["sd_level"] or 0.0) >= CRIT_B_MIN_SD_LEVEL
        and v["n_prefixes_ge10"] >= CRIT_B_MIN_PREFIXES_GE10
        for v in crit_b_arms.values()
    )
    b_change_pass = any(
        (v["sd_change"] or 0.0) >= CRIT_B_CHANGE_MIN_SD for v in crit_b_arms.values()
    )

    # Criterion C — base-side ceiling profile (reporting rule, no spend branch)
    base_vals = np.asarray(list(base_lvl.values()), dtype=np.float64)
    ceiling_share = float((base_vals >= CRIT_C_CEILING_SCORE).mean()) if base_vals.size else 0.0
    ceiling_flag = ceiling_share > CRIT_C_MAX_CEILING_SHARE

    if not a_pass:
        verdict = "resize-required"
        branch = (
            f"criterion A failed (content-drop > {CRIT_A_MAX_DROP_RATE:.0%}): re-drive wave 1 "
            f"with --max-tokens {MAX_TOKENS_RESIZE} (fresh per-(unit, max_tokens) work dir; "
            "rule 23 re-measure), then re-run the gate"
        )
    elif b_level_pass:
        verdict = "proceed-ceiling-flagged" if ceiling_flag else "proceed"
        branch = "LEVEL DV race-viable"
    elif b_change_pass:
        verdict = "proceed-change-dv"
        branch = "LEVEL flat; CHANGE DV carries range (>=2 pts SD) — proceed per §7 B branch"
    else:
        verdict = "marker-lead-rescope"
        branch = (
            "both arms flat on LEVEL and CHANGE: marker-lead re-scope + content waves "
            "descoped to 6 arms (plan §7 B terminal branch — never a whole-run kill)"
        )
    out = {
        "meta": _meta(),
        "verdict": verdict,
        "branch": branch,
        "criterion_a": crit_a,
        "criterion_b": {
            "arms": crit_b_arms,
            "level_pass": b_level_pass,
            "change_pass": b_change_pass,
        },
        "criterion_c": {
            "base_ceiling_share": ceiling_share,
            "ceiling_flag": ceiling_flag,
            "note": "flag registers the ceiling-excluded re-read as the PRIMARY level read "
            "for flagged arms (reporting rule fixed before the data are seen)",
        },
        "pilot_arms": [po, con],
    }
    J._atomic_json(out_dir / "gate1_verdict.json", out)
    return out


# ── drop report + done sentinel ───────────────────────────────────────────────


def write_drop_report(cfg: dict, out_dir: Path) -> dict:
    """Per-arm drop report SPLIT content vs transport (rules 9/24/18)."""
    report: dict = {"meta": _meta(), "units": {}}
    for spec in unit_specs(cfg):
        p = out_dir / f"arm_scores_{spec['tag']}.json"
        if not p.exists():
            continue
        pl = json.loads(p.read_text())
        report["units"][spec["tag"]] = {
            "n_items": pl["n_items"],
            "n_total_draws": pl["n_total_draws"],
            "n_content_dropped_draws": pl["n_content_dropped_draws"],
            "n_refusal_draws": pl["n_refusal_draws"],
            "n_transport_lost_draws": pl["n_transport_lost_draws"],
            "content_drop_rate": pl["content_drop_rate"],
            "max_tokens": pl["judge"]["max_tokens"],
            "rubric_sha256": pl["judge"]["rubric_sha256"],
        }
    J._atomic_json(out_dir / "drop_report.json", report)
    return report


def maybe_write_done(cfg: dict, out_dir: Path) -> bool:
    """judge_done.json sentinel once all 15 units are persisted (plan §10)."""
    tags = [s["tag"] for s in unit_specs(cfg)]
    present = [t for t in tags if (out_dir / f"arm_scores_{t}.json").exists()]
    if len(present) == len(tags):
        transports = sum(
            json.loads((out_dir / f"arm_scores_{t}.json").read_text())["n_transport_lost_draws"]
            for t in tags
        )
        J._atomic_json(
            out_dir / "judge_done.json",
            {
                "meta": _meta(),
                "n_units": len(tags),
                "units": tags,
                "n_transport_lost_draws_total": int(transports),
                "transport_note": "nonzero transport losses are re-judged before publication "
                "(rule 24(ii)); see drop_report.json",
            },
        )
        return True
    logger.info("[f2] %d/%d units persisted — no done sentinel yet", len(present), len(tags))
    return False


# ── banked-real smoke (unit-2 brief: real #1900 completions, production path) ─


def smoke_banked(n_rows: int, n_draws: int, force_batch: bool) -> int:
    """Judge N banked #1900 completions (real model text at the pin) through
    the production path into a scratch dir. Digest-only output — no raw text."""
    import tempfile

    from explore_persona_space.orchestrate import hub

    scratch = Path(tempfile.mkdtemp(prefix="i1979-judge-smoke-"))
    unit = "imp-pers-con-lr3e5-s42"
    shard_rel = f"issue1900_leakrace/judge_inputs/{unit}.shard00.jsonl"
    local = scratch / "banked.jsonl"
    hub.stage_hub_file(J._data_repo(), shard_rel, local, repo_type="dataset", revision=I1900_PIN)
    rows = J._read_jsonl_rows(local)[:n_rows]
    assert rows, "no banked rows staged"
    items = [(r["sha"][: J.SHA_ALIAS_LEN], str(r["prompt"]), str(r["response_text"])) for r in rows]
    id_map = {i[0]: r["sha"] for i, r in zip(items, rows, strict=True)}
    payload = J.judge_unit(
        "smokebanked_imp",
        "imp",
        items,
        id_map,
        scratch / "work",
        scratch / "arm_scores_smokebanked_imp.json",
        n_draws=n_draws,
        max_tokens=JUDGE_MAX_TOKENS,
        force_batch=force_batch,
    )
    scores = [r["score_mean"] for r in payload["rows"]]
    print(
        f"[smoke-banked] n_items={payload['n_items']} scored={payload['n_scored_items']} "
        f"score_means={scores} content_drops={payload['n_content_dropped_draws']} "
        f"refusals={payload['n_refusal_draws']} transport={payload['n_transport_lost_draws']} "
        f"rubric_sha={payload['judge']['rubric_sha256'][:12]} scratch={scratch}",
        flush=True,
    )
    assert payload["n_scored_items"] >= 1, "smoke judged nothing"
    return 0


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wave", type=int, default=None, help="wave 1..4 (1 = Gate-1 pilot)")
    ap.add_argument("--out-dir", type=Path, default=JUDGE_DIR)
    ap.add_argument(
        "--inputs-dir", type=Path, default=REPO_ROOT / "data" / "issue_1979" / "judge_inputs"
    )
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--max-tokens", type=int, default=JUDGE_MAX_TOKENS)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N rows per unit")
    ap.add_argument("--force-batch", action="store_true", help="force the Batch API path")
    ap.add_argument(
        "--force-past-gate",
        action="store_true",
        help="EXPLICIT deviation: run waves >= 2 without a proceed Gate-1 verdict",
    )
    ap.add_argument("--smoke-banked", type=int, default=None, metavar="N")
    ap.add_argument("--smoke-draws", type=int, default=1)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        import inspect

        from explore_persona_space.eval.batch_judge import judge_completions_batch  # noqa: F401
        from explore_persona_space.eval.graded_judge import judge_graded
        from explore_persona_space.eval.judge_dispatch import dispatch_judge_items  # noqa: F401
        from explore_persona_space.orchestrate import hub

        inspect.signature(judge_graded).bind(
            [],
            "rubric",
            n_draws=1,
            cache_dir=None,
            save_raw=Path("/tmp/x"),
            judge_model=J.JUDGE_MODEL,
            max_tokens=400,
            dry_run=True,
        )
        inspect.signature(J.judge_unit).bind(
            "t", "imp", [], {}, Path("/tmp"), Path("/tmp/o.json"), n_draws=1, max_tokens=400
        )
        inspect.signature(hub.stage_hub_file).bind("r", "p", Path("/tmp/x"))
        print("[import-check] OK — judge path deferred imports resolved + signature-bound")
        return 0

    if args.smoke_banked is not None:
        return smoke_banked(args.smoke_banked, args.smoke_draws, args.force_batch)

    assert args.wave is not None, "--wave is required (or --smoke-banked / --import-check)"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.wave >= 2 and not args.force_past_gate:  # Gate-1 refusal BEFORE any staging
        gp = args.out_dir / "gate1_verdict.json"
        if not gp.exists() or json.loads(gp.read_text())["verdict"] not in PROCEED_VERDICTS:
            print(
                f"[gate1] wave {args.wave} REFUSED: no proceed Gate-1 verdict at {gp} "
                "(run --wave 1 first; rc=8)",
                flush=True,
            )
            return RC_PILOT_MISSING
    cfg = load_config()
    wave_map = waves(cfg)
    assert args.wave in wave_map, f"unknown wave {args.wave}"

    specs_by_tag = {s["tag"]: s for s in unit_specs(cfg)}
    tags = wave_map[args.wave]
    stage_state_inputs(args.inputs_dir, sorted({specs_by_tag[t]["state"] for t in tags}))
    for k, tag in enumerate(tags):
        t0 = time.time()
        print(f"[phase=f2 wave={args.wave} unit={tag} {k + 1}/{len(tags)}]", flush=True)
        judge_state(
            cfg,
            specs_by_tag[tag],
            args.inputs_dir,
            args.out_dir,
            wave=args.wave,
            n_draws=args.draws,
            max_tokens=args.max_tokens,
            limit=args.limit,
            force_batch=args.force_batch,
        )
        print(f"[f2] unit {k + 1}/{len(tags)} {tag} elapsed={time.time() - t0:.0f}s", flush=True)

    write_drop_report(cfg, args.out_dir)
    if args.wave == 1:
        verdict = gate1_verdict(cfg, args.out_dir)
        print(f"[gate1] verdict={verdict['verdict']} branch={verdict['branch']}", flush=True)
        if verdict["verdict"] not in PROCEED_VERDICTS:
            return RC_GATE_REFUSED
    maybe_write_done(cfg, args.out_dir)
    print(f"[phase=done] f2 wave={args.wave}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
