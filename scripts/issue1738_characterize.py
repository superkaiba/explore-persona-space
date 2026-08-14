#!/usr/bin/env python3
"""Issue #1738 — per-arm error characterization (ports the #1482 stages).

Phases (each a separate ``--phase``; all read the #1738 artifacts — the fits
driver's retained per-context holdout predictions/nerr, the pinned split, the
manifest depth/corpus fields, and the K-resample fresh draws):

- ``judge``          P4b: one categorization call per holdout context via the
                     Batch API (``dispatch_judge_items``; #1482 rubric VERBATIM
                     — language/topic/refusal-adjacency/answer-is-refusal/format
                     — with a multi-turn judge-visible excerpt: last user turn
                     ≤1200 chars + history tail ≤800 + completion ≤1000, plan
                     §11) + a 200-item test-retest (κ demotion 0.6).
- ``kresample-subsample``  P4a input: 2,000 holdout contexts stratified by
                     depth-band × language-arm (en vs non-en, from the judge
                     labels) × corpus, RNG seed 173801.
- ``kresample-floor``  P4a reduce: the #1482 floor estimator — ddof-1 trace of
                     the within-context covariance over the K=4 fresh draws ÷
                     the stored per-context denominator; the own seed-42 draw
                     is READ from the fits driver's persisted holdout targets
                     (never recaptured — the #1482 93.1% convention lesson),
                     behind a join/identity gate (median rel dev < 1e-3).
- ``taxonomy``       P4c: pre-enumerated category/depth contrasts on nerr per
                     arm (batched 10k-draw bootstrap + 10k permutation p +
                     BH-FDR q=0.05 — the #1482 batched-GEMM implementations,
                     imported), floor-adjusted where the K-resample covers.
- ``h1``             The registered contrast Δ_prefix = prefix-arm ridge
                     holdout R² (layer 19) − 0.11, 10k-draw context bootstrap
                     CI (batched), verdict per plan §3.
- ``perdirection``   Per-direction top-256 answer-PCA linear-vs-nonlinear gap
                     + the 38-λ per-direction shrinkage control (#1482 stage-10
                     recipe; runs on the Phase-3 pod against the fits memmaps).
- ``figures``        Hero paired-bars + depth-stratified prefix R² curve.

Refusal-safety: LMSYS/WildChat text rides ONLY in judge API payloads and the
gitignored scratch cache — never printed, logged, or committed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue1482_analysis as A82  # noqa: E402  (judge rubric + batched boot/perm/BH-FDR)
import issue1738_multiturn_fits as FT  # noqa: E402
import issue1738_multiturn_generate_capture as GG  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1738_char")


def _upload_summary_jsons(args, paths: list[Path], *, best_effort: bool = False) -> None:
    """r4 persist-by-default belt (#1738 r4 incident): mirror phase summary
    outputs (KB-scale JSONs + the gitignored per-context floors npz) to the HF
    data repo under {hf_prefix}/analysis_tensors/summaries/characterize/ IN
    ADDITION to the git dest, so no phase output depends on a boot disk or a
    later VM harvest. Skipped under --no-upload (and when a hand-built
    namespace omits the flag — the smoke path). ``best_effort=True`` is for
    DESIGNED-halt paths (the rc-23 identity gate): the upload is attempted but
    a failure is logged loudly instead of masking the designed rc
    (artifact-first halt routing)."""
    if getattr(args, "no_upload", True):
        logger.info("[upload] characterize summaries skipped (no_upload)")
        return
    present = [p for p in paths if p.is_file()]
    if not present:
        return
    rel = sorted(str(p.relative_to(args.out_eval)) for p in present)
    # UPLOAD_PREFIX_EXEMPT: default = this issue's own --hf-prefix (issue1738_multiturn); a child issue reusing this driver must pass --upload-prefix explicitly (plan v6 §4.3)
    up = getattr(args, "upload_prefix", "") or args.hf_prefix
    dest = f"{up}/{FT.ANALYSIS_TENSORS_SUBDIR}/summaries/characterize"
    try:
        # UPLOAD_PREFIX_EXEMPT: dest defaults to this issue's own --hf-prefix (issue1738_multiturn); child reuse must pass --upload-prefix (plan v6 §4.3)
        url = hub._upload_folder_filtered(
            args.out_eval,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=dest,
            allow_patterns=rel,
            expected_repo_paths=[f"{dest}/{r}" for r in rel],
        )
        if not url:
            raise RuntimeError(f"characterize summaries upload returned no URL ({rel})")
    except Exception:
        if not best_effort:
            raise
        logger.exception("[upload] best-effort summaries upload failed (designed rc kept)")


def _arms_list(args) -> list[str]:
    """Parse + validate --arms (plan §4.3: default prefix,context for back-compat;
    the bare-query round passes 'bare')."""
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in FT.ARMS]
    assert arms and not bad, f"--arms invalid: {bad or 'empty'} (choices {sorted(FT.ARMS)})"
    return arms


def _labels_floors_eval(args) -> Path:
    """Root the EXISTING judge labels + kresample floors are read from — the
    PARENT out-eval for the bare round (plan §4.3: 0 new judge calls; the floor
    is target-side, arm-independent). Default: this run's own --out-eval."""
    return Path(args.parent_eval) if args.parent_eval else args.out_eval


JUDGE_MODEL = A82.JUDGE_MODEL  # claude-sonnet-4-5-20250929 (project pin)
JUDGE_MAX_TOKENS = A82.JUDGE_MAX_TOKENS  # 1024 (reason-then-label, rule 23; raised #2063)
KRESAMPLE_N = 2_000
KRESAMPLE_SEED = 173_801
N_BOOT = 10_000
N_PERM = 10_000
BH_Q = 0.05
STAT_SEED = 1738
H1_PREFIX_BAND_TOP = 0.11  # top of #1092's 0.05–0.11 prefix band (registered constant)
PDSHRINK_TOPK = 256
DEFAULT_OUT_EVAL = FT.DEFAULT_OUT_EVAL
KAPPA_DEMOTE = 0.6


def _judge_user_msg_mt(last_user: str, history_tail: str, response: str, corpus: str) -> str:
    """Multi-turn judge-visible excerpt (plan §11: 1200/800/1000-char caps)."""
    return (
        f"Corpus: {corpus}\n\n"
        f"=== CONVERSATION HISTORY (tail, truncated to 800 chars) ===\n{history_tail[-800:]}\n\n"
        f"=== FINAL USER MESSAGE (truncated to 1200 chars) ===\n{last_user[:1200]}\n\n"
        f"=== ASSISTANT ANSWER (truncated to 1000 chars) ===\n{response[:1000]}\n\n"
        "Categorize this exchange per the system instructions. Reason briefly, then output "
        "the JSON object."
    )


def _load_labels(out_eval: Path) -> dict:
    doc = json.loads((out_eval / "judge_labels" / "labels.json").read_text())
    return doc["labels"]


def _manifest_fields(manifest_dir: Path) -> dict[int, dict]:
    pool, _meta = GG.N1M.read_manifest_pool(Path(manifest_dir))
    return {int(r["i"]): {"depth": int(r["depth"]), "corpus": r["corpus"]} for r in pool}


def _holdout_ci(split_file: Path) -> list[int]:
    doc = FT.load_split(Path(split_file))
    return [int(c) for c in doc["sets"]["holdout"]["ci"]]


# ── P4b: judge labels (Batch API; #1482 instrument, multi-turn excerpt) ───────────


def _collect_holdout_texts(args, needed: set[int]) -> dict[int, dict]:
    """ci -> {last_user, history_tail, response, corpus} from the raw-completion
    chunk JSONs (HF or --local-raw-dir), per-chunk checkpointed to SCRATCH
    (gitignored; text never logged)."""
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    cache = scratch / "judge_texts.jsonl"
    done_chunks_p = scratch / "judge_texts.done.json"
    found: dict[int, dict] = {}
    done_chunks: set[str] = set()
    if cache.exists() and done_chunks_p.exists():
        for row in GG.N1M._read_jsonl(cache):
            if int(row["ci"]) in needed:
                found[int(row["ci"])] = row
        done_chunks = set(json.loads(done_chunks_p.read_text()))
        logger.info("[texts] resume: %d cached rows, %d chunks done", len(found), len(done_chunks))
    if args.local_raw_dir:
        names = sorted(p.name for p in Path(args.local_raw_dir).glob("shard*_chunk*.json"))
    else:
        names = sorted(
            n
            for n in GG.N50._remote_index(f"{args.hf_prefix}/{GG.RAW_SUBDIR}")
            if n.endswith(".json") and "_chunk" in n
        )
    import issue779_ffc_n1m_fits as PF

    for k, name in enumerate(names):
        if name in done_chunks:
            continue
        if len(found) >= len(needed):
            break
        if args.local_raw_dir:
            local = Path(args.local_raw_dir) / name
        else:
            local = Path(
                PF._download_chunk_with_retry(
                    C.HF_DATA_REPO, f"{args.hf_prefix}/{GG.RAW_SUBDIR}/{name}", scratch / "dl"
                )
            )
        doc = json.loads(local.read_text())
        new_rows = []
        for r in doc["rows"]:
            ci = int(r["ci"])
            if ci not in needed or ci in found:
                continue
            msgs = r["messages"]
            last_user = msgs[-1]["content"]
            history_tail = GG._plain_render(msgs[:-1])
            row = {
                "ci": ci,
                "last_user": last_user,
                "history_tail": history_tail,
                "response": r["response"],
                "corpus": r.get("corpus", "?"),
            }
            found[ci] = row
            new_rows.append(row)
        with open(cache, "a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done_chunks.add(name)
        GG.N1M._atomic_write_json(done_chunks_p, sorted(done_chunks))
        if not args.local_raw_dir:
            local.unlink(missing_ok=True)
        if (k + 1) % 25 == 0:
            logger.info(
                "[texts] chunk %d/%d: %d/%d rows found", k + 1, len(names), len(found), len(needed)
            )
    logger.info("[texts] collected %d/%d holdout rows", len(found), len(needed))
    return found


def phase_judge(args) -> None:
    from explore_persona_space.eval.batch_judge import is_transport_error_dict
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items, keep_raw_judge_text

    needed = set(_holdout_ci(Path(args.split_file)))
    if args.n_items > 0:
        needed = set(sorted(needed)[: args.n_items])
    texts = _collect_holdout_texts(args, needed)
    items = [
        (
            f"ci{ci}",
            t["last_user"][:1200],
            t["response"][:1000],
            _judge_user_msg_mt(t["last_user"], t["history_tail"], t["response"], t["corpus"]),
        )
        for ci, t in sorted(texts.items())
    ]
    jdir = args.out_eval / "judge_labels"
    (jdir / "raw").mkdir(parents=True, exist_ok=True)

    def _run(tag: str, its):
        with keep_raw_judge_text():
            return dispatch_judge_items(
                its,
                judge_model=JUDGE_MODEL,
                judge_system_prompt=A82.JUDGE_SYSTEM,
                max_tokens=JUDGE_MAX_TOKENS,
                threshold_base=1 if args.force_batch else 2000,
                checkpoint_dir=jdir / f"dispatch_{tag}",
                error_dict_factory=lambda reason: {"error": True, "reason": reason},
            )

    results = _run("main", items)
    labels: dict[str, dict] = {}
    # rule-24 split: post-retry transport losses (freely re-judgeable) vs
    # content-class drops vs other error dicts (parse_error / quarantined 400s).
    drops = {"content": 0, "transport_loss": 0, "error_other": 0}
    raw_rows = []
    for cid, res in results.items():
        if isinstance(res, dict) and res.get("error"):
            drops["transport_loss" if is_transport_error_dict(res) else "error_other"] += 1
            raw_rows.append(
                {"custom_id": cid, "error": True, "reason": str(res.get("reason"))[:300]}
            )
            continue
        lab = A82._validate_label(res)
        raw_rows.append(
            {
                "custom_id": cid,
                "raw": (res or {}).get("_raw_text", "") if isinstance(res, dict) else "",
            }
        )
        if lab is None:
            drops["content"] += 1
            continue
        labels[cid.removeprefix("ci")] = lab
    (jdir / "raw" / "main.jsonl").write_text("\n".join(json.dumps(r) for r in raw_rows) + "\n")

    rng = np.random.default_rng(STAT_SEED)
    n_rt = min(args.retest_n, len(items))
    rt_pick = rng.choice(len(items), size=n_rt, replace=False)
    rt_results = _run("retest", [(f"rt_{items[i][0]}", *items[i][1:]) for i in rt_pick])
    kappa = {}
    for field in A82.FIELDS:
        a, b = [], []
        for i in rt_pick:
            cid = items[i][0]
            l1 = labels.get(cid.removeprefix("ci"))
            l2 = A82._validate_label(rt_results.get(f"rt_{cid}"))
            if l1 and l2:
                a.append(l1[field])
                b.append(l2[field])
        kap = A82._cohens_kappa(a, b)
        kappa[field] = {
            "n": len(a),
            "kappa": kap,
            "kept": bool(np.isfinite(kap) and kap >= KAPPA_DEMOTE),
        }
    doc = {
        "n_items": len(items),
        "n_labeled": len(labels),
        "drops": drops,
        "judge_model": JUDGE_MODEL,
        "max_tokens": JUDGE_MAX_TOKENS,
        "temperature": "API default",
        "n_draws": 1,
        "rubric_sha256_system": __import__("hashlib")
        .sha256(A82.JUDGE_SYSTEM.encode())
        .hexdigest()[:16],
        "excerpt_caps": {"last_user": 1200, "history_tail": 800, "response": 1000},
        "test_retest_kappa": kappa,
        "kappa_demotion_threshold": KAPPA_DEMOTE,
        "labels": labels,
    }
    GG.N1M._atomic_write_json(jdir / "labels.json", doc)
    logger.info("[judge] labeled %d/%d (drops=%s)", len(labels), len(items), drops)
    _upload_summary_jsons(args, [jdir / "labels.json"])


# ── P4a: kresample subsample + floor estimator ────────────────────────────────────


def phase_kresample_subsample(args) -> None:
    """2,000 holdout contexts stratified by depth-band × language-arm × corpus
    (largest-remainder proportional allocation), RNG seed 173801."""
    ho = _holdout_ci(Path(args.split_file))
    fields = _manifest_fields(Path(args.manifest_dir))
    labels = {}
    if not args.no_labels:
        labels = _load_labels(args.out_eval)
    strata: dict[tuple, list[int]] = {}
    for ci in ho:
        lang = labels.get(str(ci), {}).get("language", "unlabeled")
        arm = "en" if lang == "en" else ("unlabeled" if lang == "unlabeled" else "non-en")
        key = (GG._depth_band(fields[ci]["depth"]), arm, fields[ci]["corpus"])
        strata.setdefault(key, []).append(ci)
    n_take = min(args.kresample_n, len(ho))
    rng = np.random.default_rng(KRESAMPLE_SEED)
    keys = sorted(strata.keys())
    quota = {k: n_take * len(strata[k]) / len(ho) for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    rem = n_take - sum(alloc.values())
    for k in sorted(keys, key=lambda k: quota[k] - int(quota[k]), reverse=True)[:rem]:
        alloc[k] += 1
    picked: list[int] = []
    for k in keys:
        pool = np.asarray(sorted(strata[k]), dtype=np.int64)
        take = min(alloc[k], len(pool))
        picked.extend(int(x) for x in rng.choice(pool, size=take, replace=False))
    picked.sort()
    doc = {
        "ci": picked,
        "n": len(picked),
        "seed": KRESAMPLE_SEED,
        "strata_counts": {" | ".join(k): int(alloc[k]) for k in keys},
        "sha256": GG._sha_int_list(picked),
    }
    out = args.out_eval / "kresample" / "kresample_subsample.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    GG.N1M._atomic_write_json(out, doc)
    logger.info(
        "[kresample-subsample] %d contexts across %d strata -> %s", len(picked), len(keys), out
    )
    _upload_summary_jsons(args, [out])


def _load_kresample_v(args, layers: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(ci, V (n, K, L, H) fp32) concatenated over kresample shards (local or HF)."""
    if args.local_kresample_dir:
        paths = sorted(Path(args.local_kresample_dir).glob("kresample_shard*.pt"))
    else:
        import issue779_ffc_n1m_fits as PF

        scratch = Path(args.scratch) / "kresample_dl"
        scratch.mkdir(parents=True, exist_ok=True)
        names = sorted(
            n
            for n in GG.N50._remote_index(f"{args.hf_prefix}/{GG.KRESAMPLE_SUBDIR}")
            if n.endswith(".pt")
        )
        paths = [
            Path(
                PF._download_chunk_with_retry(
                    C.HF_DATA_REPO, f"{args.hf_prefix}/{GG.KRESAMPLE_SUBDIR}/{n}", scratch
                )
            )
            for n in names
        ]
    assert paths, "no kresample shards found"
    cis, vs = [], []
    for p in paths:
        b = torch.load(p, map_location="cpu", weights_only=False)
        li_pos = [list(b["layers"]).index(li) for li in layers]
        cis.extend(int(c) for c in b["ci"])
        vs.append(b["V"][:, :, li_pos, :].to(torch.float32).numpy())
    return np.asarray(cis, dtype=np.int64), np.concatenate(vs, axis=0)


def phase_kresample_floor(args) -> None:
    """#1482 floor estimator per layer: floor_i = ddof-1 trace of the
    within-context covariance over K fresh draws; share = floor_i / den_i with
    den_i = ||v42_i − μ_holdout||² (the stored nerr denominator). Identity gate:
    the ridge nerr recomputed from the RETAINED pred16 + stored v42 must match
    the stored percontext nerr (median rel dev < 1e-3) — validating the ci join
    + conventions before any floor read."""
    layers = [int(x) for x in args.layers.split(",")]
    kci, V = _load_kresample_v(args, layers)  # (n, K, L, H)
    gates: dict = {"n_kresample": int(len(kci)), "k_draws": int(V.shape[1])}
    floor_doc: dict = {"layers": layers, "per_layer": {}}
    fields = _manifest_fields(Path(args.manifest_dir)) if args.manifest_dir else {}
    kdir = args.out_eval / "kresample"
    kdir.mkdir(parents=True, exist_ok=True)
    for lpos, li in enumerate(layers):
        yh = np.load(Path(args.y_holdout_dir) / f"L{li}.npz")
        y16, yci = yh["y16"].astype(np.float64), yh["ci"]
        pos_of = {int(c): p for p, c in enumerate(yci.tolist())}
        joined = [(i, pos_of[int(c)]) for i, c in enumerate(kci) if int(c) in pos_of]
        gates.setdefault("join", {})[str(li)] = {
            "n_joined": len(joined),
            "ok": len(joined) == len(kci),
        }
        ki = np.asarray([a for a, _ in joined])
        hp = np.asarray([b for _, b in joined])
        joined_ci = np.asarray([int(kci[a]) for a in ki], dtype=np.int64)
        mu = y16.mean(axis=0)
        den = ((y16[hp] - mu) ** 2).sum(axis=1)  # stored denominator, per context
        Vl = V[ki, :, lpos, :].astype(np.float64)  # (n, K, H)
        vbar = Vl.mean(axis=1, keepdims=True)
        k = Vl.shape[1]
        floor = ((Vl - vbar) ** 2).sum(axis=(1, 2)) / (k - 1)  # ddof-1 trace
        with np.errstate(divide="ignore", invalid="ignore"):
            share = floor / den
        # per-context floors — the P4c floor-adjusted joint-bootstrap input
        # (plan §4; consumed by phase_taxonomy on the VM).
        np.savez(
            kdir / f"floors_L{li}.npz",
            ci=joined_ci,
            floor=floor.astype(np.float64),
            den=den.astype(np.float64),
            share=share.astype(np.float64),
        )
        entry: dict = {
            "floor_median": float(np.nanmedian(floor)),
            "floor_share_median": float(np.nanmedian(share)),
            "floor_share_mean": float(np.nanmean(share)),
            "n": int(len(joined)),
        }
        # identity gate + floor-vs-error fraction per retained fitter
        for arm in ("prefix", "context"):
            for fitter in ("ridge", "mlp_w8192"):
                pz = Path(args.pred16_dir) / f"{arm}_L{li}_{fitter}.npz"
                nz = args.out_eval / "percontext" / f"{arm}_L{li}_{fitter}.npz"
                if not (pz.exists() and nz.exists()):
                    continue
                pd_ = np.load(pz)
                nerr_doc = np.load(nz)
                assert (pd_["ci"] == yci).all(), f"pred16/y_holdout ci misalign ({arm} L{li})"
                pred = pd_["pred16"].astype(np.float64)
                num = ((y16 - pred) ** 2).sum(axis=1)
                nerr_re = num / ((y16 - mu) ** 2).sum(axis=1)
                stored = nerr_doc["nerr"].astype(np.float64)
                rel = np.abs(nerr_re - stored) / np.maximum(np.abs(stored), 1e-12)
                med = float(np.median(rel))
                if fitter == "ridge":
                    gates.setdefault("identity", {})[f"{arm}_L{li}"] = {
                        "median_rel_dev": med,
                        "ok": med < 1e-3,
                    }
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = floor / num[hp]
                entry[f"floor_over_error_median__{arm}_{fitter}"] = float(np.nanmedian(frac))
        if fields:
            joined_ci = [int(kci[a]) for a, _ in joined]
            by_band: dict[str, list[float]] = {}
            for j, ci in enumerate(joined_ci):
                by_band.setdefault(GG._depth_band(fields[ci]["depth"]), []).append(float(share[j]))
            entry["floor_share_median_by_depth"] = {
                b: float(np.nanmedian(v)) for b, v in sorted(by_band.items())
            }
        floor_doc["per_layer"][str(li)] = entry
    gates["ok"] = all(g["ok"] for g in gates.get("identity", {}).values()) and all(
        g["ok"] for g in gates.get("join", {}).values()
    )
    GG.N1M._atomic_write_json(kdir / "gates.json", gates)
    GG.N1M._atomic_write_json(kdir / "floor_summary.json", floor_doc)
    # r4 dual-write: floors_L*.npz is gitignored (*.npz rule) — HF is its ONLY
    # durable home; gates/floor_summary ride along (belt on top of git).
    up = [kdir / "gates.json", kdir / "floor_summary.json"] + [
        kdir / f"floors_L{li}.npz" for li in layers
    ]
    if not gates["ok"]:
        logger.error("[kresample-floor] GATE FAILED: %s", gates)
        _upload_summary_jsons(args, up, best_effort=True)  # designed rc 23 must survive
        sys.exit(23)  # designed halt (gates.json written first — #1482 RC convention)
    _upload_summary_jsons(args, up)
    logger.info("[kresample-floor] OK: %s", {k: v for k, v in floor_doc["per_layer"].items()})


# ── P4c: taxonomy + depth contrasts (batched boot/perm + BH-FDR, imported) ────────


def _contrast_masks(
    ci_rows: np.ndarray, labels: dict, fields: dict
) -> list[tuple[str, np.ndarray]]:
    """Pre-enumerated group-vs-rest contrast masks over holdout rows (registered
    families: language en/non-en, topics with n≥50, refusal-adjacency,
    answer-is-refusal, format, depth bands, corpus)."""
    n = len(ci_rows)
    lab = [labels.get(str(int(c))) for c in ci_rows]
    masks: list[tuple[str, np.ndarray]] = []

    def _add(name: str, pred) -> None:
        m = np.asarray([bool(pred(i)) for i in range(n)])
        if 0 < m.sum() < n:
            masks.append((name, m))

    _add("language=en", lambda i: lab[i] and lab[i]["language"] == "en")
    topics = {}
    for entry in lab:
        if entry:
            topics[entry["topic"]] = topics.get(entry["topic"], 0) + 1
    for t, cnt in sorted(topics.items(), key=lambda kv: -kv[1]):
        if cnt >= 50:
            _add(f"topic={t}", lambda i, t=t: lab[i] and lab[i]["topic"] == t)
    _add("refusal_adjacent=yes", lambda i: lab[i] and lab[i]["request_refusal_adjacent"] == "yes")
    _add("answer_is_refusal=yes", lambda i: lab[i] and lab[i]["answer_is_refusal"] == "yes")
    for fm in ("code", "list", "prose"):
        _add(f"format={fm}", lambda i, fm=fm: lab[i] and lab[i]["format"] == fm)
    for band in ("2-2", "3-4", ">=5"):
        _add(
            f"depth={band}",
            lambda i, band=band: GG._depth_band(fields[int(ci_rows[i])]["depth"]) == band,
        )
    _add("corpus=wildchat", lambda i: fields[int(ci_rows[i])]["corpus"] == "wildchat")
    return masks


def phase_taxonomy(args) -> None:
    layers = [int(x) for x in args.layers.split(",")]
    fields = _manifest_fields(Path(args.manifest_dir))
    src_eval = _labels_floors_eval(args)  # EXISTING labels + floors (plan §4.3)
    labels = {} if args.no_labels else _load_labels(src_eval)
    out: dict = {
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
        "bh_q": BH_Q,
        "seed": STAT_SEED,
        "arms": {},
    }
    depth_doc: dict = {"arms": {}}
    li = 19 if 19 in layers else layers[0]
    # per-context floors from the K-resample phase (P4c floor-adjusted input).
    floors_p = src_eval / "kresample" / f"floors_L{li}.npz"
    floor_summary_p = src_eval / "kresample" / "floor_summary.json"
    if floor_summary_p.exists() and not floors_p.exists():
        raise SystemExit(
            f"{floor_summary_p} exists but per-context floors {floors_p} are missing — "
            "stale pre-fix kresample-floor artifact; re-run --phase kresample-floor"
        )
    floors = None
    if floors_p.exists():
        with np.load(floors_p) as fz:
            floors = {"ci": fz["ci"].copy(), "share": fz["share"].copy()}
    out["floor_adjusted_available"] = floors is not None
    for arm in _arms_list(args):
        nz = args.out_eval / "percontext" / f"{arm}_L{li}_ridge.npz"
        if not nz.exists():
            logger.warning("[taxonomy] missing %s — skipping arm", nz)
            continue
        z = np.load(nz)
        nerr, ci_rows = z["nerr"].astype(np.float64), z["ci"]
        masks = _contrast_masks(ci_rows, labels, fields)
        names = [n for n, _ in masks]
        pvals = A82._perm_pvals(nerr, [m for _, m in masks], N_PERM, STAT_SEED)
        sig = A82._bh_fdr(pvals, BH_Q)
        rows = []
        for (name, m), p, s in zip(masks, pvals, sig, strict=True):
            deltas = A82._boot_group_delta(nerr, m, ~m, N_BOOT, STAT_SEED)
            rows.append(
                {
                    "contrast": name,
                    "n_group": int(m.sum()),
                    "delta_mean_nerr": float(nerr[m].mean() - nerr[~m].mean()),
                    "boot_ci": [
                        float(np.quantile(deltas, 0.025)),
                        float(np.quantile(deltas, 0.975)),
                    ],
                    "perm_p": float(p),
                    "bh_significant": bool(s),
                }
            )
        out["arms"][f"{arm}_L{li}_ridge"] = {
            "n": int(len(nerr)),
            "contrasts": rows,
            "family": names,
        }
        # depth-stratified holdout R² per band (from pred16 + y_holdout); the
        # three artifacts MUST share row order — a capture-set-changed re-run
        # that mixed stale/fresh files fails loud here (Major-1 ci asserts).
        pz = Path(args.pred16_dir) / f"{arm}_L{li}_ridge.npz"
        yh = np.load(Path(args.y_holdout_dir) / f"L{li}.npz")
        pd_z = np.load(pz)
        assert (pd_z["ci"] == yh["ci"]).all() and (pd_z["ci"] == ci_rows).all(), (
            f"pred16/y_holdout/percontext ci misalign ({arm} L{li}) — stale artifact mix"
        )
        pred = pd_z["pred16"].astype(np.float64)
        y = yh["y16"].astype(np.float64)
        bands: dict[str, dict] = {}
        for band in ("2-2", "3-4", ">=5"):
            sel = np.asarray(
                [GG._depth_band(fields[int(c)]["depth"]) == band for c in ci_rows], dtype=bool
            )
            if sel.sum() < 10:
                continue
            r2, cos = F._recon_point(pred[sel], y[sel])
            bands[band] = {"n": int(sel.sum()), "r2": float(r2), "mean_cosine": float(cos)}
        depth_doc["arms"][f"{arm}_L{li}_ridge"] = bands
        # floor-adjusted contrasts (plan §4 P4c): the SAME contrast families on
        # adj_i = nerr_i − floor_i/den_i over the K-resample subset, with a
        # 10k-draw JOINT bootstrap — each draw resamples CONTEXTS, so floor and
        # error ride together (paired) — via the #1482 batched masked-GEMM
        # helpers (never a serial draw loop).
        if floors is not None:
            pos_of = {int(c): p for p, c in enumerate(ci_rows.tolist())}
            missing = [int(c) for c in floors["ci"] if int(c) not in pos_of]
            assert not missing, (
                f"floors ci absent from {arm} percontext rows (capture-set drift?): {missing[:5]}"
            )
            sub_pos = np.asarray([pos_of[int(c)] for c in floors["ci"]], dtype=np.int64)
            adj_all = nerr[sub_pos] - floors["share"]
            fin = np.isfinite(adj_all)
            if not fin.all():  # named non-fatal exclusion (degenerate den), never a coerce
                logger.warning(
                    "[taxonomy] floor-adjusted %s: dropping %d non-finite rows",
                    arm,
                    int((~fin).sum()),
                )
            adj = adj_all[fin]
            sub_ci = np.asarray(floors["ci"])[fin]
            fmasks = _contrast_masks(sub_ci, labels, fields)
            fpv = A82._perm_pvals(adj, [m for _, m in fmasks], N_PERM, STAT_SEED)
            fsig = A82._bh_fdr(fpv, BH_Q)
            frows = []
            for (name, m), p, s in zip(fmasks, fpv, fsig, strict=True):
                deltas = A82._boot_group_delta(adj, m, ~m, N_BOOT, STAT_SEED)
                frows.append(
                    {
                        "contrast": name,
                        "n_group": int(m.sum()),
                        "delta_mean_adj_nerr": float(adj[m].mean() - adj[~m].mean()),
                        "boot_ci": [
                            float(np.quantile(deltas, 0.025)),
                            float(np.quantile(deltas, 0.975)),
                        ],
                        "perm_p": float(p),
                        "bh_significant": bool(s),
                    }
                )
            out["arms"][f"{arm}_L{li}_ridge"]["floor_adjusted"] = {
                "n": int(len(adj)),
                "n_dropped_nonfinite": int((~fin).sum()),
                "n_boot": N_BOOT,
                "n_perm": N_PERM,
                "seed": STAT_SEED,
                "definition": (
                    "adj_i = nerr_i - floor_i/den_i (map error net of answer-sampling "
                    "variance); contexts resampled JOINTLY per draw"
                ),
                "contrasts": frows,
            }
    GG.N1M._atomic_write_json(args.out_eval / "taxonomy.json", out)
    GG.N1M._atomic_write_json(args.out_eval / "depth_contrasts.json", depth_doc)
    logger.info("[taxonomy] wrote %d arm tables", len(out["arms"]))
    _upload_summary_jsons(
        args, [args.out_eval / "taxonomy.json", args.out_eval / "depth_contrasts.json"]
    )


# ── H1: registered contrast Δ_prefix (batched context bootstrap) ──────────────────


def phase_h1(args) -> None:
    li = 19
    yh = np.load(Path(args.y_holdout_dir) / f"L{li}.npz")
    y = yh["y16"].astype(np.float64)
    doc: dict = {"registered_constant": H1_PREFIX_BAND_TOP, "layer": li, "n_boot": N_BOOT}
    for arm in ("prefix", "context"):
        pz = Path(args.pred16_dir) / f"{arm}_L{li}_ridge.npz"
        if not pz.exists():
            continue
        pd_z = np.load(pz)
        # Major-1 ci assert: the REGISTERED H1 contrast must never pair stale
        # y_holdout rows with fresh pred16 rows (capture-set-changed re-run).
        assert (pd_z["ci"] == yh["ci"]).all(), f"pred16/y_holdout ci misalign ({arm} L{li})"
        pred = pd_z["pred16"].astype(np.float64)
        ci_b = FT._boot_recon_ci_batched(pred, y, N_BOOT, STAT_SEED)
        doc[arm] = {"holdout_r2": ci_b["r2"], "mean_cosine": ci_b["mean_cosine"]}
    if "prefix" in doc:
        r2 = doc["prefix"]["holdout_r2"]
        delta = {
            "point": r2["point"] - H1_PREFIX_BAND_TOP,
            "lo": r2["lo"] - H1_PREFIX_BAND_TOP,
            "hi": r2["hi"] - H1_PREFIX_BAND_TOP,
        }
        verdict = (
            "Confirmed"
            if delta["point"] > 0 and delta["lo"] > 0
            else "Falsified"
            if delta["hi"] < 0
            else "Inconclusive"
        )
        doc["delta_prefix"] = delta
        doc["verdict"] = verdict
        logger.info(
            "[h1] Δ_prefix=%.4f CI=[%.4f, %.4f] -> %s",
            delta["point"],
            delta["lo"],
            delta["hi"],
            verdict,
        )
    GG.N1M._atomic_write_json(args.out_eval / "h1_contrast.json", doc)
    _upload_summary_jsons(args, [args.out_eval / "h1_contrast.json"])


# ── per-direction PCA + 38-λ shrinkage control (#1482 stage-10, per arm) ──────────


def _pdshrink_lambda_grid() -> np.ndarray:
    """Union λ grid for the per-direction shrinkage control. NOTE (review
    Minor 7): the plan's "38" assumes the two logspaces' shared values (1e-3,
    1e-2) dedup, which holds only when float-identical across the differently-
    parameterized calls — the realized grid may be 38–40 values (harmless:
    denser grid; recorded as ``lambda_grid_len`` in the summary)."""
    import issue779_ffc_n1m_fits as PF

    vals = {float(v) for v in np.logspace(-5, -2, 16)} | {float(v) for v in PF.LAMBDAS_N1M} | {1e-9}
    return np.asarray(sorted(vals), dtype=np.float64)


def phase_perdirection(args) -> None:
    """Runs ON the Phase-3 pod against the fits memmaps: top-256 answer-PCA basis
    from Y train (fp64 streaming covariance + eigh, cuSOLVER CPU fallback), then
    per-direction holdout R² for ridge (shared-λ), MLP (retained pred16), and
    the per-direction-λ tuned ridge over the union grid (val-selected per
    direction) — the shrinkage control. Per arm."""
    import issue779_ffc_n1m_fits as PF

    layers = [int(x) for x in args.layers.split(",")]
    li = 19 if 19 in layers else layers[0]
    arms = _arms_list(args)
    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    mm, ci, _meta = FT.assemble_streams(args, layers)
    if "bare" in arms:  # plan §4.3: bare X memmaps ride the fits bare assembly
        bare_mm, _bm = FT.assemble_bare_streams(args, layers, ci, _meta["fingerprint"])
        mm.update(bare_mm)
    split = FT.load_split(Path(args.split_file))
    sets = FT.split_positions(split, ci)
    tr, val, ho = sets["train"], sets["val"], sets["holdout"]
    Y = mm[("vx", li)]
    block = args.ridge_block

    # top-256 PCA basis of Y over train (fp64 streaming covariance; sign-invariant reads)
    h_dim = Y.shape[1]
    A = torch.zeros((h_dim, h_dim), dtype=torch.float64, device=dev)
    mu_acc = torch.zeros(h_dim, dtype=torch.float64, device=dev)
    n_acc = 0
    for s in range(0, len(tr), block):
        yb = torch.as_tensor(np.asarray(Y[tr[s : s + block]], dtype=np.float64), device=dev)
        A += yb.T @ yb
        mu_acc += yb.sum(0)
        n_acc += yb.shape[0]
    mu = mu_acc / n_acc
    A = A / n_acc - torch.outer(mu, mu)
    try:
        evals, evecs = torch.linalg.eigh(A)
    except torch.linalg.LinAlgError:  # cuSOLVER non-convergence -> CPU LAPACK (gotchas)
        logger.warning("[pdshrink] cuda eigh non-convergence; CPU LAPACK fallback")
        evals, evecs = torch.linalg.eigh(A.cpu())
    topk = min(PDSHRINK_TOPK, h_dim)
    top = torch.flip(evecs[:, -topk:], dims=[1]).cpu().numpy()
    eigvals = torch.flip(evals[-topk:], dims=[0]).cpu().numpy()

    lam_grid = _pdshrink_lambda_grid()
    doc: dict = {
        "layer": li,
        "topk": int(topk),
        "lambda_grid_len": int(len(lam_grid)),
        "eigvals_head": [float(x) for x in eigvals[:8]],
        "arms": {},
    }
    yv = np.asarray(Y[val], dtype=np.float64)
    yh = np.asarray(Y[ho], dtype=np.float64)
    yv_rot, yh_rot = yv @ top, yh @ top
    bands = {"1-16": (0, 16), "17-64": (16, 64), "65-128": (64, 128), "129-256": (128, 256)}
    for arm in [a for a in FT.ARM_ORDER if a in arms]:
        X = mm[(FT.ARM_MM_KEY[arm], li)]
        fac = PF._ridge_factorize(X, Y, tr, dev, block)
        proj_val = np.empty((len(lam_grid), len(val), topk), dtype=np.float32)
        proj_ho = np.empty((len(lam_grid), len(ho), topk), dtype=np.float32)
        for gi, lam in enumerate(lam_grid):
            pv = PF._ridge_predict_one(X, val, fac, float(lam), dev, block)
            ph = PF._ridge_predict_one(X, ho, fac, float(lam), dev, block)
            proj_val[gi] = (pv @ top).astype(np.float32)
            proj_ho[gi] = (ph @ top).astype(np.float32)
            if (gi + 1) % 10 == 0:
                logger.info("[pdshrink] %s λ %d/%d", arm, gi + 1, len(lam_grid))

        def _pd_r2(proj: np.ndarray, true_rot: np.ndarray) -> np.ndarray:
            mu_d = true_rot.mean(axis=0)
            num = ((true_rot - proj) ** 2).sum(axis=0)
            den = ((true_rot - mu_d) ** 2).sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 - num / den

        val_r2 = np.stack(
            [_pd_r2(proj_val[g].astype(np.float64), yv_rot) for g in range(len(lam_grid))]
        )
        sel = np.nanargmax(val_r2, axis=0)  # per-direction λ index
        tuned_proj = np.take_along_axis(
            proj_ho, np.broadcast_to(sel[None, None, :], (1,) + proj_ho.shape[1:]), axis=0
        )[0]
        tuned_r2 = _pd_r2(tuned_proj.astype(np.float64), yh_rot)
        # shared-λ ridge + retained MLP holdout predictions, projected. Major-1
        # ci assert: retained pred16 rows must align with THIS assembly's
        # holdout rows (fresh mm) — stale retained fits fail loud here.
        ridge_z = np.load(Path(args.pred16_dir) / f"{arm}_L{li}_ridge.npz")
        assert (ridge_z["ci"] == ci[ho]).all(), f"pred16 ridge ci != assembly ho ci ({arm} L{li})"
        ridge16 = ridge_z["pred16"].astype(np.float64)
        mlp16_p = Path(args.pred16_dir) / f"{arm}_L{li}_mlp_w8192.npz"
        ridge_r2 = _pd_r2(ridge16 @ top, yh_rot)
        arm_doc: dict = {
            "bands": {},
            "per_direction": {
                "ridge_shared": [float(x) for x in ridge_r2],
                "ridge_tuned": [float(x) for x in tuned_r2],
                "tuned_lambda_idx": [int(x) for x in sel],
            },
        }
        if mlp16_p.exists():
            mlp_z = np.load(mlp16_p)
            assert (mlp_z["ci"] == ci[ho]).all(), f"pred16 mlp ci != assembly ho ci ({arm} L{li})"
            mlp_r2 = _pd_r2(mlp_z["pred16"].astype(np.float64) @ top, yh_rot)
            arm_doc["per_direction"]["mlp_w8192"] = [float(x) for x in mlp_r2]
            for bname, (lo, hi) in bands.items():
                hi = min(hi, topk)
                if lo >= topk:
                    continue
                arm_doc["bands"][bname] = {
                    "ridge_shared_mean": float(np.nanmean(ridge_r2[lo:hi])),
                    "ridge_tuned_mean": float(np.nanmean(tuned_r2[lo:hi])),
                    "mlp_mean": float(np.nanmean(mlp_r2[lo:hi])),
                    "gap_shared": float(np.nanmean(mlp_r2[lo:hi] - ridge_r2[lo:hi])),
                    "gap_excess_over_tuned": float(np.nanmean(mlp_r2[lo:hi] - tuned_r2[lo:hi])),
                }
        doc["arms"][arm] = arm_doc
        logger.info(
            "[pdshrink] %s: bands=%s",
            arm,
            {k: round(v["gap_excess_over_tuned"], 4) for k, v in arm_doc["bands"].items()},
        )
    out = args.out_eval / "perdirection"
    out.mkdir(parents=True, exist_ok=True)
    GG.N1M._atomic_write_json(out / "pdshrink_summary.json", doc)
    _upload_summary_jsons(args, [out / "pdshrink_summary.json"])


# ── figures (hero paired bars + depth curve) ──────────────────────────────────────


def phase_figures(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    fits = json.loads((args.out_eval / "fits" / f"{FT.FIT_POINT}_fits.json").read_text())
    layers = fits["layers"]
    fitters = list(FT.PREDICTORS)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = paper_palette(len(fitters))

    arms_l = _arms_list(args)
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, li in zip(axes, layers, strict=True):
        xs = np.arange(len(arms_l))
        w = 0.8 / len(fitters)
        for fi, name in enumerate(fitters):
            vals = [
                fits["cells"].get(f"{arm}_L{li}_{name}", {}).get("holdout_r2", np.nan)
                for arm in arms_l
            ]
            los, his = [], []
            for arm in arms_l:
                cib = fits["cells"].get(f"{arm}_L{li}_{name}", {}).get("holdout_bootstrap_ci", {})
                r2 = cib.get("r2", {})
                los.append(r2.get("lo", np.nan))
                his.append(r2.get("hi", np.nan))
            # xerr/yerr take NON-NEGATIVE offsets (gotchas #547/#1335)
            yerr = np.stack(
                [
                    np.maximum(0, np.asarray(vals) - np.asarray(los)),
                    np.maximum(0, np.asarray(his) - np.asarray(vals)),
                ]
            )
            ax.bar(
                xs + fi * w,
                vals,
                w,
                yerr=yerr,
                color=colors[fi],
                label=name if li == layers[0] else None,
            )
        ax.axhspan(0.05, H1_PREFIX_BAND_TOP, color="gray", alpha=0.25)
        ax.set_xticks(xs + 0.4 - w / 2)
        ax.set_xticklabels(arms_l)
        ax.set_title(f"layer {li}")
    axes[0].set_ylabel("holdout R² (10k contexts)")
    fig.legend(loc="upper center", ncol=len(fitters), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    hero = fig_dir / "hero_prefix_vs_context_r2.png"
    fig.savefig(hero, dpi=200)
    fig.savefig(hero.with_suffix(".pdf"))
    plt.close(fig)

    depth_p = args.out_eval / "depth_contrasts.json"
    if depth_p.exists():
        depth = json.loads(depth_p.read_text())
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for k, (key, bands) in enumerate(sorted(depth["arms"].items())):
            names = [b for b in ("2-2", "3-4", ">=5") if b in bands]
            ax.plot(
                range(len(names)),
                [bands[b]["r2"] for b in names],
                marker="o",
                color=paper_palette(2)[k % 2],
                label=key,
            )
            for xi, b in enumerate(names):
                ax.annotate(f"n={bands[b]['n']}", (xi, bands[b]["r2"]), fontsize=6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["2", "3–4", "≥5"])
        ax.set_xlabel("conversation depth (user turns)")
        ax.set_ylabel("holdout R²")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(fig_dir / "depth_stratified_r2.png", dpi=200)
        fig.savefig(fig_dir / "depth_stratified_r2.pdf")
        plt.close(fig)
    meta = {
        "source": str(args.out_eval / "fits" / f"{FT.FIT_POINT}_fits.json"),
        "git_commit": __import__("subprocess")
        .run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        .stdout.strip(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    GG.N1M._atomic_write_json(fig_dir / "figures.meta.json", meta)
    logger.info("[figures] wrote %s", fig_dir)


# ── smoke: production entrypoints on the fits smoke store ─────────────────────────


def _smoke(args) -> int:
    """Runs judge-item construction (+ optional 2-item LIVE sync judge call),
    subsample/floor/taxonomy/h1/perdirection/figures phases end-to-end against
    the fits driver's tiny synthetic store outputs (which _smoke regenerates by
    invoking the fits smoke first). Text is synthetic — refusal-safe."""
    import subprocess

    root = Path(args.scratch) / "_smoke_char"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    # (1) regenerate the fits smoke store + outputs through the fits PRODUCTION path
    rc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "issue1738_multiturn_fits.py"),
            "--smoke",
            "--mm-dir",
            str(root / "mm"),
        ],
        cwd=PROJECT_ROOT,
        env={**__import__("os").environ},
    ).returncode
    assert rc == 0, f"fits smoke rc={rc}"
    froot = root / "_smoke_fits"
    man = froot / "manifest"
    out_eval = froot / "eval"
    pred16 = froot / "local" / "pred16"
    yhold = froot / "local" / "y_holdout"

    ns = argparse.Namespace(
        **{
            **vars(args),
            "out_eval": out_eval,
            "manifest_dir": str(man),
            "split_file": str(man / "split_1738.json"),
            "pred16_dir": str(pred16),
            "y_holdout_dir": str(yhold),
            "layers": "19",
            "no_labels": True,
            "kresample_n": 12,
            "no_upload": True,  # r4: the dual-write belt never fires in smoke
        }
    )
    # (2) subsample (no labels — depth × corpus strata only, flagged)
    phase_kresample_subsample(ns)
    sub = json.loads((out_eval / "kresample" / "kresample_subsample.json").read_text())
    assert sub["n"] == 12 and sub["sha256"] == GG._sha_int_list(sub["ci"])

    # (3) synthetic kresample V shards (schema of the driver's kresample mode)
    kdir = root / "kres"
    kdir.mkdir(parents=True, exist_ok=True)
    yh = np.load(yhold / "L19.npz")
    pos_of = {int(c): p for p, c in enumerate(yh["ci"].tolist())}
    rng = np.random.default_rng(0)
    kci = sub["ci"]
    base = yh["y16"].astype(np.float32)[[pos_of[c] for c in kci]]
    V = base[:, None, None, :] + 0.1 * rng.standard_normal((len(kci), 4, 1, base.shape[1])).astype(
        np.float32
    )
    torch.save(
        {
            "V": torch.from_numpy(V).to(torch.float16),
            "ci": kci,
            "seeds": [43, 44, 45, 46],
            "layers": [19],
        },
        kdir / "kresample_shard00.pt",
    )
    ns.local_kresample_dir = str(kdir)
    phase_kresample_floor(ns)
    gates = json.loads((out_eval / "kresample" / "gates.json").read_text())
    assert gates["ok"], gates
    floor = json.loads((out_eval / "kresample" / "floor_summary.json").read_text())
    assert "19" in floor["per_layer"]
    with np.load(out_eval / "kresample" / "floors_L19.npz") as fz:
        assert fz["ci"].shape == fz["share"].shape == fz["floor"].shape == fz["den"].shape
        assert len(fz["ci"]) == sub["n"], (len(fz["ci"]), sub["n"])

    # degenerate probe (data-dependent-gates duty): a tampered stored nerr must
    # trip the identity gate -> designed halt rc 23 (gates.json written first).
    tampered = out_eval / "percontext" / "prefix_L19_ridge.npz"
    orig_bytes = tampered.read_bytes()
    z0 = np.load(tampered)
    np.savez(tampered, nerr=z0["nerr"] * 1.5, ci=z0["ci"])
    try:
        phase_kresample_floor(ns)
        raise AssertionError("kresample identity gate did not halt on tampered nerr")
    except SystemExit as e:
        assert e.code == 23, e.code
    tampered.write_bytes(orig_bytes)
    phase_kresample_floor(ns)  # restore + re-verify clean

    # judge drop-never-coerce probe: a malformed label validates to None (drop).
    good = {
        "language": "en",
        "topic": "coding",
        "request_refusal_adjacent": "no",
        "answer_is_refusal": "no",
        "format": "prose",
    }
    assert A82._validate_label(good) is not None
    assert A82._validate_label({**good, "language": "english"}) is None
    assert A82._validate_label("REFUSAL") is None

    # (4) taxonomy + depth (synthetic labels for every holdout ci)
    hoc = _holdout_ci(Path(ns.split_file))
    labels = {
        str(c): {
            "language": "en" if c % 4 else "es",
            "topic": "coding" if c % 2 else "chitchat_social",
            "request_refusal_adjacent": "no",
            "answer_is_refusal": "no",
            "format": "prose",
        }
        for c in hoc
    }
    jdir = out_eval / "judge_labels"
    jdir.mkdir(parents=True, exist_ok=True)
    GG.N1M._atomic_write_json(jdir / "labels.json", {"labels": labels})
    ns.no_labels = False
    phase_taxonomy(ns)
    tax = json.loads((out_eval / "taxonomy.json").read_text())
    assert tax["arms"], "no taxonomy arms"
    assert tax["floor_adjusted_available"] is True
    for arm in tax["arms"].values():
        assert arm["contrasts"], "empty contrast table"
        fa = arm["floor_adjusted"]  # plan §4 P4c: joint bootstrap present per arm
        assert fa["contrasts"] and fa["n_boot"] == N_BOOT and fa["seed"] == STAT_SEED, fa
        for row in fa["contrasts"]:
            assert len(row["boot_ci"]) == 2 and "perm_p" in row, row

    # (5) h1 + perdirection + figures
    phase_h1(ns)
    h1 = json.loads((out_eval / "h1_contrast.json").read_text())
    assert h1["verdict"] in ("Confirmed", "Falsified", "Inconclusive"), h1
    pdns = argparse.Namespace(
        **{
            **vars(ns),
            "local_capture_dir": str(froot / "capture"),
            "mm_dir": str(root / "mm2"),
            "hf_prefix": GG.HF_PREFIX,
            "device": "cpu",
            "ridge_block": 50_000,
        }
    )
    phase_perdirection(pdns)
    pd_doc = json.loads((out_eval / "perdirection" / "pdshrink_summary.json").read_text())
    assert "context" in pd_doc["arms"] and pd_doc["arms"]["context"]["bands"], pd_doc["arms"].keys()

    ns.fig_dir = str(root / "figs")
    phase_figures(ns)
    assert (Path(ns.fig_dir) / "hero_prefix_vs_context_r2.png").exists()

    # (5b) bare-arm threading (plan §4.3): taxonomy + depth + floor-relative +
    # perdirection with --arms bare against the fits smoke's bare outputs;
    # labels + floors read from the PARENT eval (--parent-eval), 0 judge calls.
    beval = froot / "eval_bare"
    bns = argparse.Namespace(
        **{
            **vars(ns),
            "arms": "bare",
            "out_eval": beval,
            "parent_eval": str(out_eval),
            "pred16_dir": str(froot / "local_bare" / "pred16"),
            "y_holdout_dir": str(froot / "local_bare" / "y_holdout"),
        }
    )
    phase_taxonomy(bns)
    btax = json.loads((beval / "taxonomy.json").read_text())
    assert set(btax["arms"]) == {"bare_L19_ridge"}, sorted(btax["arms"])
    assert btax["arms"]["bare_L19_ridge"]["contrasts"]
    assert btax["floor_adjusted_available"] is True
    assert btax["arms"]["bare_L19_ridge"]["floor_adjusted"]["contrasts"]
    bdep = json.loads((beval / "depth_contrasts.json").read_text())
    assert "bare_L19_ridge" in bdep["arms"], sorted(bdep["arms"])
    bpd = argparse.Namespace(
        **{
            **vars(bns),
            "local_capture_dir": str(froot / "capture"),
            "local_bare_dir": str(froot / "bare_capture"),
            "mm_dir": str(root / "mm3"),
            "hf_prefix": GG.HF_PREFIX,
            "device": "cpu",
            "ridge_block": 50_000,
        }
    )
    phase_perdirection(bpd)
    bpdoc = json.loads((beval / "perdirection" / "pdshrink_summary.json").read_text())
    assert "bare" in bpdoc["arms"] and bpdoc["arms"]["bare"]["bands"], sorted(bpdoc["arms"])

    # (6) judge request-builder: items composed via the production builder;
    # optionally 2 LIVE sync calls (--smoke-live-judge) through dispatch_judge_items.
    items = [
        (
            "ci0",
            "synthetic final user message",
            "synthetic answer",
            _judge_user_msg_mt(
                "synthetic final user message",
                "user: earlier\nassistant: earlier a",
                "synthetic answer",
                "lmsys",
            ),
        ),
        (
            "ci1",
            "another synthetic message",
            "another synthetic answer",
            _judge_user_msg_mt(
                "another synthetic message",
                "user: h\nassistant: ha",
                "another synthetic answer",
                "wildchat",
            ),
        ),
    ]
    assert all("=== ASSISTANT ANSWER" in it[3] for it in items)
    if args.smoke_live_judge:
        from explore_persona_space.eval.judge_dispatch import (
            dispatch_judge_items,
            keep_raw_judge_text,
        )

        with keep_raw_judge_text():
            res = dispatch_judge_items(
                items,
                judge_model=JUDGE_MODEL,
                judge_system_prompt=A82.JUDGE_SYSTEM,
                max_tokens=JUDGE_MAX_TOKENS,
                threshold_base=2000,  # sync path at n=2
                checkpoint_dir=root / "judge_smoke",
                error_dict_factory=lambda reason: {"error": True, "reason": reason},
            )
        n_ok = sum(1 for r in res.values() if A82._validate_label(r) is not None)
        logger.info("[smoke] live judge: %d/%d validated labels", n_ok, len(items))
        assert n_ok >= 1, f"live judge smoke returned no valid labels: {res}"
    logger.info("[smoke] characterize OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1738 per-arm error characterization.")
    ap.add_argument(
        "--phase",
        choices=[
            "judge",
            "kresample-subsample",
            "kresample-floor",
            "taxonomy",
            "h1",
            "perdirection",
            "figures",
        ],
        default="taxonomy",
    )
    ap.add_argument("--out-eval", type=Path, default=DEFAULT_OUT_EVAL)
    # UPLOAD_PREFIX_EXEMPT: issue 1738's own summaries prefix; a child issue reusing this driver must pass --hf-prefix explicitly (artifact-reuse check (i))
    ap.add_argument("--hf-prefix", default=GG.HF_PREFIX)
    # ── bare-arm threading (follow-up `bare-query`, plan §4.3) ────────────────────
    ap.add_argument(
        "--arms",
        default="prefix,context",
        help="comma list of arms (back-compat default; the bare round passes 'bare')",
    )
    ap.add_argument(
        "--parent-eval",
        default="",
        help="out-eval root the EXISTING judge labels + kresample floors are read "
        "from (the bare round passes eval_results/issue_1738; default: --out-eval)",
    )
    # UPLOAD_PREFIX_EXEMPT: issue 1738's own bare-arm store prefix (plan §4.3); read-side default
    ap.add_argument("--bare-hf-prefix", default=f"{GG.HF_PREFIX}/bare_query")
    ap.add_argument("--local-bare-dir", default="", help="read bare chunks locally (smoke)")
    ap.add_argument(
        "--upload-prefix",
        default="",
        help="HF prefix for THIS run's summary dual-writes (default = --hf-prefix); "
        "the bare round passes issue1738_multiturn/bare_query",
    )
    ap.add_argument("--manifest-dir", default="")
    ap.add_argument("--split-file", default="")
    ap.add_argument("--pred16-dir", default=str(FT.DEFAULT_OUT_LOCAL / "pred16"))
    ap.add_argument("--y-holdout-dir", default=str(FT.DEFAULT_OUT_LOCAL / "y_holdout"))
    ap.add_argument("--mm-dir", default=str(FT.DEFAULT_OUT_LOCAL / "mm"))
    ap.add_argument("--local-capture-dir", default="")
    ap.add_argument("--local-raw-dir", default="")
    ap.add_argument("--local-kresample-dir", default="")
    ap.add_argument("--scratch", default=str(PROJECT_ROOT / "data" / "issue_1738" / "scratch"))
    ap.add_argument("--layers", default="14,19,26")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--ridge-block", type=int, default=50_000)
    ap.add_argument("--n-items", type=int, default=0, help="judge: cap items (0 = all holdout)")
    ap.add_argument("--retest-n", type=int, default=200)
    ap.add_argument("--force-batch", action="store_true")
    ap.add_argument("--kresample-n", type=int, default=KRESAMPLE_N)
    ap.add_argument("--no-labels", action="store_true", help="subsample without judge labels")
    ap.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures" / "issue_1738"))
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="skip the r4 HF dual-write of phase summary JSONs (git dest still written)",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-live-judge", action="store_true", help="smoke: 2 LIVE judge calls")
    args = ap.parse_args()

    if args.smoke:
        rc = _smoke(args)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(rc)
    if not args.split_file:
        if not args.manifest_dir:
            raise SystemExit("--split-file or --manifest-dir required")
        args.split_file = str(Path(args.manifest_dir) / "split_1738.json")
    {
        "judge": phase_judge,
        "kresample-subsample": phase_kresample_subsample,
        "kresample-floor": phase_kresample_floor,
        "taxonomy": phase_taxonomy,
        "h1": phase_h1,
        "perdirection": phase_perdirection,
        "figures": phase_figures,
    }[args.phase](args)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
