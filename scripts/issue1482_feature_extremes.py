#!/usr/bin/env python
"""Issue #1482 user-chat inline free-analysis round: feature-extremes.

Are the BEST-predicted answer-side SAE features qualitatively different from the
WORST-predicted ones? Extends the committed feature-correlates Q1 abstraction
read (`scripts/issue1482_feature_correlates.py`, a stratified 300-feature draw
that came back NULL) to the two tails of the per-feature held-out R2 distribution
(`eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz`, 16,384
answer-side features of the context-features -> answer-features ridge map).

Selection (deterministic; a pure function of the committed npz):
  Set A (global tails):     top-150 and bottom-150 features by per-feature R2.
                            CONFOUNDED with activity by construction — the
                            per-tail activity distribution is reported.
  Set B (activity-controlled): within each of the 10 activity deciles (same
                            `FC._decile_of` edges as the reference round), the
                            15 best and 15 worst by R2 (300 features). Equal
                            per-decile arm sizes make the pooled best-vs-worst
                            contrast activity-controlled BY DESIGN, and the
                            headline test is a decile-STRATIFIED permutation.
The UNION of both sets is judged once (358 features); set membership rides each
feature row.

Judge instrument: every instrument-defining element is IMPORTED from the
reference module (`FC.JUDGE_SYSTEM` rubric, `FC.JUDGE_MODEL`,
`FC.JUDGE_MAX_TOKENS`, `FC._judge_items` evidence builder — top-8 fit-row
answers by `ans_mean` + Neuronpedia auto-interp as labeled auxiliary evidence,
`FC._validate_level` drop-never-coerce validation, `FC.RETEST_N` test-retest),
so parity holds by construction rather than by copy, and a rubric-hash gate
asserts it against the reference round's recorded `rubric_sha256_system`. The
ONE deliberate difference: this round PERSISTS the judge's `reasoning` string
(the reference validator discarded it) so the dashboard can show why each
feature was labeled.

No model fits anywhere: rank statistics, exact 2x2 tests and a stratified
permutation test only. Vectorized: per-shard `np.bincount` accumulation over the
local pooled store; no per-feature Python loop on the scan.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402

import issue1482_feature_correlates as FC  # noqa: E402
import issue1482_sae as FC_SAE  # noqa: E402  (BatchTopKSAE loader, decoder weights)

WORK_DEFAULT = Path("/mnt/eps-data/thomasjiralerspong/issue1482_featext")
PRIOR_WORK = Path("/mnt/eps-data/thomasjiralerspong/issue1482_featcorr")
OUT_EVAL = PROJECT_ROOT / "eval_results/issue_1482/feature_extremes"
OUT_FIGS = PROJECT_ROOT / "figures/issue_1482/feature_extremes"
PRIOR_ABSTRACTION = PROJECT_ROOT / "eval_results/issue_1482/feature_correlates/abstraction.json"
CONSISTENCY_NPZ = (
    PROJECT_ROOT / "eval_results/issue_1482/feature_correlates/consistency_perfeature.npz"
)

N_TAIL = 150  # Set A: global best/worst by per-feature R2
N_DECILE_TAIL = 15  # Set B: per-activity-decile best/worst
N_PERM = 10_000
N_BOOT = 10_000
PERM_SEED = 14_822_027  # recorded seed for the stratified permutation + bootstrap

# Neuronpedia auto-interp export (the reference round's auxiliary judge evidence).
# The public /api/feature endpoint 500s and /api/explanation/export was retired in
# favour of a public S3 dataset mirror; both carry the same explanations.
NP_MODEL_ID = "qwen2.5-7b-it"
NP_SOURCE_ID = "19-resid-post-aa"  # hfFolderId resid_post_layer_19/trainer_1 (k=64 trainer)
NP_S3 = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
NP_PREFIX = f"v1/{NP_MODEL_ID}/{NP_SOURCE_ID}/explanations/"
NP_EXPL_MODEL = "gemini-2.0-flash"  # parity with the reference round's descriptions
NP_FEATURE_URL = "https://www.neuronpedia.org/{model}/{source}/{index}"

LEVELS = ("low", "high", "unclear")
PERSONA_LEVELS = ("yes", "no", "unclear")

# Rubric EXTENSION (stated deviation from the byte-identical-instrument claim).
# APPENDED to `FC.JUDGE_SYSTEM`, so the reference round's every character — the
# `level` field and its full definition text — survives as a byte-exact PREFIX
# (asserted in `_assert_rubric_parity`); only the persona field is new.
PERSONA_RUBRIC_SUFFIX = (
    ' Additionally include a fourth key "persona_related": "yes" | "no" | "unclear" in the '
    "SAME JSON object. persona_related=yes: the feature encodes WHO is speaking or a "
    "persistent manner of speaking — identity, persona, style, register, tone, language, "
    "disposition or trait. persona_related=no: the feature encodes topical content, task "
    "format, or local syntax. persona_related=unclear: neither applies clearly."
)
JUDGE_SYSTEM_EXT = FC.JUDGE_SYSTEM + PERSONA_RUBRIC_SUFFIX

# Mechanical persona-alignment read: |cos(SAE decoder column, r_B)| at layer 19.
# r_B = the #779 monitoring per-layer mean-difference persona directions (the same
# set #1092's monitoring figures consume, mirrored on HF at issue779_monitoring/r_b).
RB_DIR = PROJECT_ROOT / "data/issue_779/r_b"
RB_TRAITS = ("evil", "sycophancy", "hallucination")
SAE_LAYER = 19  # BatchTopKSAE.load asserts cfg["layer"] == 19; r_b row index 19
SAE_K = 64  # trainer_1 — the k the committed per-feature R2 was computed under
ALIGN_TOP_N = 50  # "R2 percentile of the top-N most-aligned features"


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [featext] {msg}", flush=True)


def _provenance(extra: dict | None = None) -> dict:
    """Reproducibility block embedded in every emitted JSON (CLAUDE.md)."""
    import scipy

    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        commit = out.stdout.strip() if out.returncode == 0 else "uncommitted"
    except (FileNotFoundError, subprocess.SubprocessError):
        commit = "uncommitted"
    meta = {
        "git_commit": commit or "uncommitted",
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "script": Path(__file__).name,
        "perm_seed": PERM_SEED,
        "n_perm": N_PERM,
        "n_boot": N_BOOT,
    }
    if extra:
        meta.update(extra)
    return meta


# ── selection ────────────────────────────────────────────────────────────────


def _select(com: dict[str, np.ndarray]) -> dict:
    """Deterministic Set A / Set B selection over the committed per-feature npz.

    Returns restricted-index arrays (positions into the 16,384-feature npz) plus
    a per-feature membership table for the union. Stable argsort throughout, so
    the selection is reproducible bit-for-bit from the committed inputs alone.
    """
    r2, act, fid = com["r2"], com["activity"], com["feat_ids"]
    dec = FC._decile_of(act)
    order = np.argsort(r2, kind="stable")
    a_worst, a_best = order[:N_TAIL], order[-N_TAIL:][::-1]
    b_best_parts, b_worst_parts = [], []
    for d in range(FC.N_DECILES):
        ind = np.where(dec == d)[0]
        assert len(ind) >= 2 * N_DECILE_TAIL, f"decile {d} has only {len(ind)} features"
        o = ind[np.argsort(r2[ind], kind="stable")]
        b_worst_parts.append(o[:N_DECILE_TAIL])
        b_best_parts.append(o[-N_DECILE_TAIL:][::-1])
    b_best = np.concatenate(b_best_parts)
    b_worst = np.concatenate(b_worst_parts)
    union = np.unique(np.concatenate([a_best, a_worst, b_best, b_worst]))
    members = {
        "a_best": set(a_best.tolist()),
        "a_worst": set(a_worst.tolist()),
        "b_best": set(b_best.tolist()),
        "b_worst": set(b_worst.tolist()),
    }
    rows = [
        {
            "feat_id": int(fid[i]),
            "restricted_idx": int(i),
            "r2": float(r2[i]),
            "activity": float(act[i]),
            "decile": int(dec[i]),
            "a_best": i in members["a_best"],
            "a_worst": i in members["a_worst"],
            "b_best": i in members["b_best"],
            "b_worst": i in members["b_worst"],
        }
        for i in union.tolist()
    ]
    return {
        "n_tail": N_TAIL,
        "n_decile_tail": N_DECILE_TAIL,
        "n_deciles": FC.N_DECILES,
        "n_union": int(len(union)),
        "n_set_a": int(len(a_best) + len(a_worst)),
        "n_set_b": int(len(b_best) + len(b_worst)),
        "n_a_and_b": int(
            len(set(a_best.tolist() + a_worst.tolist()) & set(b_best.tolist() + b_worst.tolist()))
        ),
        "idx": {
            "a_best": a_best.tolist(),
            "a_worst": a_worst.tolist(),
            "b_best": b_best.tolist(),
            "b_worst": b_worst.tolist(),
            "union": union.tolist(),
        },
        "features": rows,
    }


# ── Neuronpedia auto-interp (auxiliary judge evidence) ───────────────────────


def _http_get(url: str, *, timeout: int = 120, attempts: int = 4) -> bytes:
    """GET with bounded retry on transient HTTP/network errors; fail loud after."""
    last: Exception | None = None
    for k in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "eps-research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 403):
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"GET failed after {attempts} attempts: {url} ({last})") from last


def _np_batch_keys() -> list[str]:
    keys: list[str] = []
    token = ""
    while True:
        q = f"list-type=2&prefix={NP_PREFIX}&max-keys=1000"
        if token:
            q += f"&continuation-token={urllib.parse.quote(token)}"
        body = _http_get(f"{NP_S3}?{q}").decode()
        keys.extend(re.findall(r"<Key>([^<]+)</Key>", body))
        if "<IsTruncated>true" not in body:
            break
        tok = re.findall(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if not tok:
            break
        token = tok[0]
    return sorted(k for k in keys if k.endswith(".jsonl.gz"))


def phase_neuronpedia(args) -> None:
    """Stage the Neuronpedia explanation export and emit the union's descriptions.

    Downloads the source's explanation batches once into a resumable per-file
    cache, then writes `neuronpedia_explanations.json` in the shape
    `FC._judge_items` reads. Fail-loud on transport failure — a silently missing
    description set would change the judge instrument.
    """
    com = FC._load_committed()
    sel = _select(com)
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "selection.json").write_text(json.dumps(sel, indent=1))
    want = {int(r["feat_id"]) for r in sel["features"]}

    cache = args.work / "np_cache"
    cache.mkdir(parents=True, exist_ok=True)
    keys = _np_batch_keys()
    assert keys, f"no explanation batches under {NP_PREFIX}"
    _log(f"neuronpedia: {len(keys)} explanation batches under {NP_PREFIX}")

    def _stage(key: str) -> Path:
        dest = cache / key.rsplit("/", 1)[-1]
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        blob = _http_get(f"{NP_S3}/{urllib.parse.quote(key)}")
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(blob)
        tmp.replace(dest)
        return dest

    with ThreadPoolExecutor(max_workers=6) as pool:
        staged = list(pool.map(_stage, keys))
    _log(f"neuronpedia: staged {len(staged)} batches into {cache}")

    found: dict[str, dict] = {}
    n_rows = 0
    for k, path in enumerate(staged):
        text = gzip.decompress(path.read_bytes()).decode("utf-8")
        for line in text.split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            n_rows += 1
            idx = int(rec["index"])
            if idx not in want:
                continue
            desc = rec.get("description")
            found[str(idx)] = {
                "description": (desc or "").strip(),
                "explanationModel": rec.get("explanationModelName"),
                "typeName": rec.get("typeName"),
                "hfFolderId": f"resid_post_layer_{NP_SOURCE_ID.split('-')[0]}/trainer_1",
                "np_model_id": NP_MODEL_ID,
                "np_source_id": NP_SOURCE_ID,
            }
        if (k + 1) % 50 == 0:
            _log(f"neuronpedia: parsed {k + 1}/{len(staged)} batches, {len(found)} hits")

    with_desc = sum(1 for v in found.values() if v["description"])
    same_model = sum(1 for v in found.values() if v["explanationModel"] == NP_EXPL_MODEL)
    (args.work / "neuronpedia_explanations.json").write_text(json.dumps(found, indent=1))

    agree = mism = 0
    prior_path = PRIOR_WORK / "neuronpedia_explanations.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        for fs, v in found.items():
            if fs in prior:
                pd = (prior[fs].get("description") or "").strip()
                if pd == v["description"]:
                    agree += 1
                else:
                    mism += 1
    doc = {
        "provenance": _provenance(
            {
                "np_prefix": NP_PREFIX,
                "n_batches": len(staged),
                "n_export_rows_scanned": n_rows,
            }
        ),
        "n_union": len(want),
        "n_resolved": len(found),
        "n_with_description": with_desc,
        "n_expected_expl_model": same_model,
        "expected_expl_model": NP_EXPL_MODEL,
        "prior_round_overlap_agree": agree,
        "prior_round_overlap_mismatch": mism,
    }
    (args.work / "neuronpedia_summary.json").write_text(json.dumps(doc, indent=1))
    _log(
        f"neuronpedia done: {len(found)}/{len(want)} resolved, {with_desc} with description, "
        f"{same_model} from {NP_EXPL_MODEL}; prior-overlap agree={agree} mismatch={mism}"
    )


# ── scan (top-K firing answers per union feature) ─────────────────────────────


def phase_scan(args) -> None:
    """One streamed pass over the local pooled shards: top-K fit-row contexts per
    union feature by `ans_mean`, plus the reference round's activity wiring gate.

    Vectorized per shard (bincount + boolean gather); periodic top-K compaction.
    """
    com = FC._load_committed()
    sel = _select(com)
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "selection.json").write_text(json.dumps(sel, indent=1))
    samp_ids = np.asarray([r["feat_id"] for r in sel["features"]], dtype=np.int64)
    samp_pos = np.full(FC.DICT_SIZE, -1, dtype=np.int64)
    samp_pos[samp_ids] = np.arange(len(samp_ids))
    _log(f"scan: {len(samp_ids)} union features (Set A + Set B)")

    shards = sorted(args.store.glob("pooled_*.npz"))
    assert len(shards) == 1920, f"expected 1920 shards, found {len(shards)}"
    cnt = np.zeros(FC.DICT_SIZE, dtype=np.int64)
    n_fit = 0
    cand: list[np.ndarray] = []  # columns: samp_row, ci, ans_mean

    def _compact(rows: list[np.ndarray]) -> list[np.ndarray]:
        if not rows:
            return []
        m = np.concatenate(rows, axis=0)
        keep = []
        for s in np.unique(m[:, 0]).astype(np.int64):
            sub = m[m[:, 0] == s]
            if len(sub) > FC.TOP_K_CONTEXTS:
                sub = sub[np.argpartition(-sub[:, 2], FC.TOP_K_CONTEXTS - 1)[: FC.TOP_K_CONTEXTS]]
            keep.append(sub)
        return [np.concatenate(keep, axis=0)]

    for i, p in enumerate(shards):
        with np.load(p, allow_pickle=False) as z:
            fit = np.asarray(z["set_tag"]) == 1
            off = np.asarray(z["idx_off"], dtype=np.int64)
            n_fit += int(fit.sum())
            keep = np.repeat(fit, off)
            ik = np.asarray(z["ans_idx"], dtype=np.int64)[keep]
            cnt += np.bincount(ik, minlength=FC.DICT_SIZE)
            sp = samp_pos[ik]
            hit = sp >= 0
            if hit.any():
                ci_rep = np.repeat(np.asarray(z["ci"], dtype=np.int64), off)[keep][hit]
                val = np.asarray(z["ans_mean"], dtype=np.float64)[keep][hit]
                cand.append(np.column_stack([sp[hit].astype(np.float64), ci_rep, val]))
        if (i + 1) % 256 == 0:
            cand = _compact(cand)
            _log(f"scan {i + 1}/1920 shards; n_fit so far {n_fit}")
    cand = _compact(cand)

    # ── wiring gate: recomputed activity must match the committed covariate ──
    fid = com["feat_ids"]
    act_re = cnt[fid] / n_fit
    gate = float(np.abs(act_re - com["activity"]).max())
    _log(f"activity wiring gate: n_fit={n_fit} max|delta|={gate:.2e}")
    assert gate < 1e-3, f"activity mismatch vs committed npz (max|delta|={gate})"

    np.savez(
        args.work / "scan.npz",
        feat_ids=fid,
        r2=com["r2"],
        activity=com["activity"],
        activity_recomputed=act_re,
        n_fit=np.int64(n_fit),
    )
    top = cand[0] if cand else np.zeros((0, 3))
    top = top[np.lexsort((-top[:, 2], top[:, 0]))]
    top_by_feat: dict[str, list[list[float]]] = {}
    for s, ci, val in top:
        top_by_feat.setdefault(str(int(samp_ids[int(s)])), []).append([float(val), int(ci)])
    (args.work / "sample_top_contexts.json").write_text(json.dumps(top_by_feat))
    _log(f"scan done: n_fit={n_fit}, gate={gate:.2e}, top-context features={len(top_by_feat)}")


# ── texts (raw answers behind the top-K contexts) ────────────────────────────


def phase_texts(args) -> None:
    """Collect the raw answers for the union's top-K contexts.

    Reuses the parent chunk enumeration + bounded-retry download helpers, seeds
    from the reference round's already-fetched texts, and early-exits once every
    needed context is in hand. Text stays in the gitignored work dir
    (digest-only discipline).
    """
    import issue1482_error_analysis as D  # heavy import (torch) deferred to use

    top = json.loads((args.work / "sample_top_contexts.json").read_text())
    needed_ci = {int(ci): 0 for lst in top.values() for _v, ci in lst[: FC.TOP_K_CONTEXTS]}
    _log(f"texts: {len(needed_ci)} unique contexts needed")

    cache = args.work / "texts.jsonl"

    def _read(src: Path) -> dict[int, str]:
        got: dict[int, str] = {}
        if not src.exists():
            return got
        for ln in src.read_text(encoding="utf-8").split("\n"):
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except ValueError:
                continue  # truncated tail from a crash mid-append
            if rec.get("kind") == "chunk_done":
                continue
            ci = int(rec["ci"])
            if ci in needed_ci:
                got[ci] = rec["response"]
        return got

    # The LOCAL cache is the only file downstream phases (FC._judge_items) read,
    # so every needed row seeded from the reference round's cache is MATERIALIZED
    # here — an in-memory-only seed silently starves the judge of those features.
    out = _read(cache)
    _log(f"texts: {len(out)} needed rows already in the local cache")
    prior = _read(PRIOR_WORK / "texts.jsonl")
    carry = {ci: t for ci, t in prior.items() if ci not in out}
    if carry:
        with cache.open("a", encoding="utf-8") as fh:
            for ci, text in carry.items():
                fh.write(json.dumps({"ci": int(ci), "response": text}) + "\n")
        out.update(carry)
    _log(f"texts: carried {len(carry)} rows over from {PRIOR_WORK.name} into the local cache")

    missing = {ci for ci in needed_ci if ci not in out}
    _log(f"texts: {len(missing)} contexts still missing after seeding")
    if missing:
        dns = argparse.Namespace(scratch=args.work, max_chunks=0)
        names = D._raw_chunk_names(dns)
        with cache.open("a", encoding="utf-8") as fh:
            for k, name in enumerate(names):
                for _nm, keep in D._iter_needed_rows(dns, [name], dict.fromkeys(missing, 0)):
                    for _row, ci, _prompt, response in keep:
                        out[int(ci)] = response
                        missing.discard(int(ci))
                        fh.write(json.dumps({"ci": int(ci), "response": response}) + "\n")
                fh.flush()
                if (k + 1) % 100 == 0:
                    _log(f"texts: chunk {k + 1}/{len(names)}, {len(missing)} still missing")
                if not missing:
                    _log(f"texts: all contexts collected after {k + 1}/{len(names)} chunks")
                    break
    assert not missing, f"{len(missing)} needed contexts had no raw-chunk text"
    _log(f"texts done: {len(out)} rows available")


# ── judge (instrument imported from the reference module) ────────────────────


def _assert_rubric_parity() -> dict[str, str]:
    """Fail loud unless the reference rubric survives byte-exact as a PREFIX.

    The extended rubric adds the `persona_related` field, so its hash necessarily
    differs from the reference round's — a STATED deviation. What this gate
    enforces is that the deviation is ADDITIVE only: `FC.JUDGE_SYSTEM` still
    hashes to the reference round's recorded value AND is a byte-exact prefix of
    the extended rubric, so the `level` field's definition text is provably
    unchanged and prior-round `level` agreement stays comparable.
    """
    base = hashlib.sha256(FC.JUDGE_SYSTEM.encode()).hexdigest()[:16]
    prior = json.loads(PRIOR_ABSTRACTION.read_text())
    want = prior["rubric_sha256_system"]
    assert base == want, f"reference rubric drift: {base} != {want}"
    assert JUDGE_SYSTEM_EXT.startswith(FC.JUDGE_SYSTEM), (
        "extended rubric must keep the reference rubric as a byte-exact prefix"
    )
    assert FC.JUDGE_MODEL == prior["judge_model"], "judge model drift"
    assert FC.JUDGE_MAX_TOKENS == prior["max_tokens"], "max_tokens drift"
    ext = hashlib.sha256(JUDGE_SYSTEM_EXT.encode()).hexdigest()[:16]
    _log(
        f"rubric parity OK: reference-prefix sha16={base} (matches prior round), "
        f"extended sha16={ext} (+persona_related field) model={FC.JUDGE_MODEL} "
        f"max_tokens={FC.JUDGE_MAX_TOKENS}"
    )
    return {"reference_prefix_sha16": base, "extended_sha16": ext}


def _validate_labels(res: object) -> dict | None:
    """Validate the EXTENDED rubric's return: reference level/label + persona field.

    Delegates the level/label half to `FC._validate_level` (so drop-never-coerce
    semantics are the reference round's verbatim) and additionally REQUIRES a
    well-formed `persona_related`; a missing/out-of-range persona value is a
    CONTENT drop, never coerced.
    """
    base = FC._validate_level(res)
    if base is None:
        return None
    pr = res.get("persona_related") if isinstance(res, dict) else None
    if not (isinstance(pr, str) and pr.strip().lower() in PERSONA_LEVELS):
        return None
    return {**base, "persona_related": pr.strip().lower()}


def phase_judge(args) -> None:
    """One label call per union feature + a `FC.RETEST_N` test-retest replica.

    Instrument elements all come from `FC`; the only addition over the reference
    round is that the judge's `reasoning` string is persisted (the reference
    validator dropped it). Fresh dispatch dirs => cold judge cache by
    construction. Drop-never-coerce, with the rule-24 transport/content split.
    """
    from explore_persona_space.eval.batch_judge import is_transport_error_dict
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items

    import issue1482_analysis as A

    hashes = _assert_rubric_parity()
    items = FC._judge_items(args)
    _log(f"judge: {len(items)} feature items (extended rubric: level + persona_related)")

    def _run(tag: str, its):
        return dispatch_judge_items(
            its,
            judge_model=FC.JUDGE_MODEL,
            judge_system_prompt=JUDGE_SYSTEM_EXT,
            max_tokens=FC.JUDGE_MAX_TOKENS,
            checkpoint_dir=args.work / f"dispatch_{tag}",
            error_dict_factory=lambda reason: {"error": True, "reason": reason},
        )

    def _collect(results: dict[str, dict]) -> tuple[dict[str, dict], dict[str, int]]:
        labels: dict[str, dict] = {}
        drops = {"content": 0, "transport": 0}
        for cid, res in results.items():
            if isinstance(res, dict) and res.get("error"):
                drops["transport" if is_transport_error_dict(res) else "content"] += 1
                continue
            lab = _validate_labels(res)
            if lab is None:
                drops["content"] += 1
                continue
            reason = res.get("reasoning") if isinstance(res, dict) else None
            labels[cid] = {**lab, "reasoning": str(reason)[:400] if reason else ""}
        return labels, drops

    raw_main, drops = _collect(_run("main_v2", items))
    labels = {cid.removeprefix("feat"): v for cid, v in raw_main.items()}

    rng = np.random.default_rng(FC.SAMPLE_SEED)
    rt_pick = rng.choice(len(items), size=min(FC.RETEST_N, len(items)), replace=False)
    rt_items = [(f"rt_{items[i][0]}", *items[i][1:]) for i in rt_pick]
    rt_labels, rt_drops = _collect(_run("retest_v2", rt_items))
    pairs: dict[str, tuple[list[str], list[str]]] = {
        "level": ([], []),
        "persona_related": ([], []),
    }
    for i in rt_pick:
        cid = items[i][0]
        first = labels.get(cid.removeprefix("feat"))
        second = rt_labels.get(f"rt_{cid}")
        if first and second:
            for field, (aa, bb) in pairs.items():
                aa.append(first[field])
                bb.append(second[field])
    kappa = A._cohens_kappa(*pairs["level"])
    kappa_persona = A._cohens_kappa(*pairs["persona_related"])
    a = pairs["level"][0]

    doc = {
        "provenance": _provenance(),
        "n_items": len(items),
        "n_labeled": len(labels),
        "drops": drops,
        "retest_drops": rt_drops,
        "judge_model": FC.JUDGE_MODEL,
        "max_tokens": FC.JUDGE_MAX_TOKENS,
        "temperature": "API default",
        "n_draws": 1,
        "rubric_sha256_system": hashes["extended_sha16"],
        "rubric_sha256_reference_prefix": hashes["reference_prefix_sha16"],
        "instrument_note": (
            "model/max_tokens/evidence-builder/level-validator imported from "
            "scripts/issue1482_feature_correlates.py; STATED DEVIATION from the "
            "byte-identical-instrument claim: the rubric APPENDS a persona_related "
            "field, so the reference rubric survives as a byte-exact PREFIX (its own "
            "sha16 still matches the reference round) and the level definition text is "
            "unchanged, but the full rubric hash differs. Also persists the judge "
            "reasoning string, which the reference validator dropped."
        ),
        "test_retest": {
            "n": len(a),
            "kappa_level": kappa,
            "kappa_persona_related": kappa_persona,
        },
        "labels": labels,
    }
    (args.work / "feature_levels.json").write_text(json.dumps(doc, indent=1))
    _log(
        f"judge done: {len(labels)}/{len(items)} labeled, drops={drops} "
        f"(retest {rt_drops}), kappa_level={kappa:.3f} "
        f"kappa_persona={kappa_persona:.3f} (n={len(a)})"
    )


# ── mechanical persona-alignment read (judge-free, all 16,384 features) ──────


def _load_rb_layer() -> tuple[np.ndarray, list[str]]:
    """Return (n_traits, hidden) layer-`SAE_LAYER` persona directions + trait names.

    Reads the #779 monitoring per-layer r_B artifacts (`data/issue_779/r_b/<trait>.pt`,
    key `r_b`, shape (28, 3584)); asserts the file's own `layers` list actually maps
    index `SAE_LAYER` to layer `SAE_LAYER`, so a re-indexed artifact fails loud rather
    than silently reading a different layer.
    """
    import torch

    rows, names = [], []
    for trait in RB_TRAITS:
        path = RB_DIR / f"{trait}.pt"
        assert path.exists(), f"r_B artifact missing: {path}"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        arr = np.asarray(payload["r_b"].detach().cpu().numpy(), dtype=np.float64)
        assert arr.ndim == 2 and arr.shape[1] == 3584, arr.shape
        layers = [int(x) for x in payload["layers"]]
        assert layers[SAE_LAYER] == SAE_LAYER, (
            f"{trait}: layers[{SAE_LAYER}] == {layers[SAE_LAYER]}, expected {SAE_LAYER}"
        )
        assert payload.get("trait") == trait, (payload.get("trait"), trait)
        assert payload.get("smoke") is False, f"{trait}: r_B is a SMOKE artifact"
        rows.append(arr[SAE_LAYER])
        names.append(trait)
    return np.stack(rows, axis=0), names


def phase_align(args) -> None:
    """|cos(SAE decoder column, r_B)| for all 16,384 answer-side features.

    ONE matrix product over the whole restricted feature set (no per-feature loop):
    column-normalized decoder slice (16384, 3584) @ normalized r_B (3584, n_traits).
    Cosine is invariant to the SAE's weight-folded positive norm factor, so the
    published raw-activation convention needs no undoing here.
    """
    com = FC._load_committed()
    fid = com["feat_ids"]
    rb, trait_names = _load_rb_layer()  # (n_traits, hidden)
    sae = FC_SAE.BatchTopKSAE.load(k=SAE_K, device="cpu", cache_dir=args.work / "sae")
    w_dec = sae.w_dec.detach().cpu().numpy().astype(np.float64)  # (hidden, dict_size)
    assert w_dec.shape == (rb.shape[1], FC.DICT_SIZE), w_dec.shape

    d = w_dec[:, fid].T  # (n_feat, hidden) — the restricted answer-side features
    d_norm = np.linalg.norm(d, axis=1, keepdims=True)
    r_norm = np.linalg.norm(rb, axis=1, keepdims=True)
    assert np.all(d_norm > 0) and np.all(r_norm > 0), "zero-norm decoder column or r_B"
    cos = (d / d_norm) @ (rb / r_norm).T  # ONE GEMM -> (n_feat, n_traits)
    assert cos.shape == (len(fid), len(trait_names)), cos.shape
    assert np.abs(cos).max() <= 1.0 + 1e-9, float(np.abs(cos).max())

    abs_cos = np.abs(cos)
    args.work.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.work / "align.npz",
        feat_ids=fid,
        cos=cos,
        abs_cos=abs_cos,
        max_abs_cos=abs_cos.max(axis=1),
        top_trait_idx=abs_cos.argmax(axis=1),
        trait_names=np.asarray(trait_names, dtype="U16"),
        layer=np.int64(SAE_LAYER),
    )
    _log(
        f"align done: {len(fid)} features x {len(trait_names)} traits at layer {SAE_LAYER}; "
        f"max|cos| median {float(np.median(abs_cos.max(axis=1))):.4f} "
        f"max {float(abs_cos.max()):.4f}"
    )


# ── analysis ─────────────────────────────────────────────────────────────────


def _boot_median_ci(v: np.ndarray, rng: np.random.Generator) -> list[float]:
    if len(v) == 0:
        return [float("nan"), float("nan")]
    draws = np.median(v[rng.integers(0, len(v), size=(N_BOOT, len(v)))], axis=1)
    lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    return [lo, hi]


def _partial_spearman_multi(x: np.ndarray, y: np.ndarray, zs: list[np.ndarray]) -> float:
    """Rank-partial correlation of x and y given SEVERAL covariates.

    Multi-covariate generalization of `FC._partial_spearman` (which takes one z);
    Pearson correlation of the residuals after regressing the ranks of x and y on
    the ranks of every covariate. Used to check that the judged-level effect is
    not simply a proxy for within-answer consistency, the known strong R2
    correlate from the reference round.
    """
    rx, ry = FC._rank(x), FC._rank(y)
    zc = np.column_stack([np.ones_like(rx)] + [FC._rank(z) for z in zs])
    bx, *_ = np.linalg.lstsq(zc, rx, rcond=None)
    by, *_ = np.linalg.lstsq(zc, ry, rcond=None)
    ex, ey = rx - zc @ bx, ry - zc @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def _unclear_contrast(lev_best: np.ndarray, lev_worst: np.ndarray) -> dict:
    """Best-vs-worst contrast on judged-UNINTERPRETABLE rate (unclear vs low|high).

    `unclear` means the judge found no coherent shared property in the examples,
    so this is a legibility contrast rather than an abstraction one — reported
    separately because collapsing it into the low/high comparison would hide it.
    """
    from scipy.stats import fisher_exact

    ub, uw = int((lev_best == "unclear").sum()), int((lev_worst == "unclear").sum())
    nb, nw = int(len(lev_best)), int(len(lev_worst))
    odds, p = fisher_exact([[ub, nb - ub], [uw, nw - uw]])
    return {
        "n_unclear_best": ub,
        "n_unclear_worst": uw,
        "frac_unclear_best": ub / nb if nb else float("nan"),
        "frac_unclear_worst": uw / nw if nw else float("nan"),
        "fisher_exact_2x2": {"odds_ratio": float(odds), "p": float(p)},
    }


def _composition(levels: np.ndarray) -> dict:
    n = int(len(levels))
    out = {"n": n}
    for lev in LEVELS:
        c = int((levels == lev).sum())
        out[f"n_{lev}"] = c
        out[f"frac_{lev}"] = (c / n) if n else float("nan")
    lohi = int(((levels == "low") | (levels == "high")).sum())
    out["n_low_or_high"] = lohi
    out["frac_high_of_low_high"] = (int((levels == "high").sum()) / lohi) if lohi else float("nan")
    return out


def _composition_field(vals: np.ndarray, levels: tuple[str, ...]) -> dict:
    """Category composition of `vals` over `levels` (counts + fractions)."""
    n = int(len(vals))
    out: dict = {"n": n}
    for lev in levels:
        c = int((vals == lev).sum())
        out[f"n_{lev}"] = c
        out[f"frac_{lev}"] = (c / n) if n else float("nan")
    return out


def _binary_contrast(best: np.ndarray, worst: np.ndarray, positive: str, negative: str) -> dict:
    """Best-vs-worst contrast on a binary category (`positive` vs `negative`)."""
    from scipy.stats import fisher_exact

    hb = (best[np.isin(best, [positive, negative])] == positive).astype(float)
    hw = (worst[np.isin(worst, [positive, negative])] == positive).astype(float)
    out = {
        "n_best": int(len(hb)),
        "n_worst": int(len(hw)),
        f"frac_{positive}_best": float(hb.mean()) if len(hb) else float("nan"),
        f"frac_{positive}_worst": float(hw.mean()) if len(hw) else float("nan"),
    }
    if len(hb) and len(hw):
        odds, pv = fisher_exact(
            [
                [int(hb.sum()), int(len(hb) - hb.sum())],
                [int(hw.sum()), int(len(hw) - hw.sum())],
            ]
        )
        out["fisher_exact_2x2"] = {"odds_ratio": float(odds), "p": float(pv)}
        out["diff_best_minus_worst"] = out[f"frac_{positive}_best"] - out[f"frac_{positive}_worst"]
    return out


def _arm_contrast(lev_best: np.ndarray, lev_worst: np.ndarray) -> dict:
    """Best-vs-worst contrast on the binary level variable (high=1 over {low,high}).

    Mann-Whitney on a binary variable IS a proportion comparison; Fisher's exact
    test is the exact test for the same 2x2 and is reported alongside.
    """
    from scipy.stats import fisher_exact, mannwhitneyu

    hb = (lev_best[np.isin(lev_best, ["low", "high"])] == "high").astype(float)
    hw = (lev_worst[np.isin(lev_worst, ["low", "high"])] == "high").astype(float)
    out = {
        "n_best_low_high": int(len(hb)),
        "n_worst_low_high": int(len(hw)),
        "frac_high_best": float(hb.mean()) if len(hb) else float("nan"),
        "frac_high_worst": float(hw.mean()) if len(hw) else float("nan"),
    }
    if len(hb) and len(hw):
        mw = mannwhitneyu(hb, hw, alternative="two-sided")
        odds, p = fisher_exact(
            [
                [int(hb.sum()), int(len(hb) - hb.sum())],
                [int(hw.sum()), int(len(hw) - hw.sum())],
            ]
        )
        out["mannwhitney_high_best_vs_worst"] = {"U": float(mw.statistic), "p": float(mw.pvalue)}
        out["fisher_exact_2x2"] = {"odds_ratio": float(odds), "p": float(p)}
        out["frac_high_diff_best_minus_worst"] = out["frac_high_best"] - out["frac_high_worst"]
    return out


def _stratified_perm(
    is_high: np.ndarray, is_best: np.ndarray, decile: np.ndarray, rng: np.random.Generator
) -> dict:
    """Decile-stratified permutation test on the pooled frac-high difference.

    Arm labels are permuted WITHIN each activity decile (preserving each decile's
    per-arm counts), so activity is held fixed by construction. Vectorized: one
    (N_PERM, n_d) key matrix per decile, no per-draw Python loop.
    """
    obs = float(is_high[is_best].mean() - is_high[~is_best].mean())
    perm_best = np.zeros((N_PERM, len(is_high)), dtype=bool)
    for d in np.unique(decile):
        m = np.where(decile == d)[0]
        keys = rng.random((N_PERM, len(m)))
        order = np.argsort(keys, axis=1)
        perm_best[:, m] = is_best[m][order]
    hi = is_high[None, :]
    n_b = perm_best.sum(axis=1)
    n_w = (~perm_best).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = (hi * perm_best).sum(axis=1) / n_b - (hi * ~perm_best).sum(axis=1) / n_w
    ok = np.isfinite(stat)
    p = float((np.abs(stat[ok]) >= abs(obs) - 1e-12).mean())
    return {
        "statistic_frac_high_best_minus_worst": obs,
        "p_two_sided": p,
        "n_perm": int(ok.sum()),
        "n_rows": int(len(is_high)),
        "note": "arm labels permuted within activity decile; statistic over {low,high} rows",
    }


def _build_rows(args) -> list[dict]:
    """Join selection + judged level + R2/activity/consistency per union feature."""
    sel = json.loads((args.work / "selection.json").read_text())
    lv = json.loads((args.work / "feature_levels.json").read_text())
    npd = json.loads((args.work / "neuronpedia_explanations.json").read_text())
    z = np.load(CONSISTENCY_NPZ)
    cons = {int(f): float(c) for f, c in zip(z["feat_ids"], z["consistency"])}
    prior = json.loads(PRIOR_ABSTRACTION.read_text())
    prior_lv = {int(f["feat_id"]): f["level"] for f in prior["features"]}
    al = np.load(args.work / "align.npz")
    al_traits = [str(t) for t in al["trait_names"]]
    al_max = {int(f): float(v) for f, v in zip(al["feat_ids"], al["max_abs_cos"])}
    al_trait = {int(f): al_traits[int(i)] for f, i in zip(al["feat_ids"], al["top_trait_idx"])}
    al_per = {
        int(f): {t: float(v) for t, v in zip(al_traits, row)}
        for f, row in zip(al["feat_ids"], al["abs_cos"])
    }

    rows = []
    for r in sel["features"]:
        lab = lv["labels"].get(str(r["feat_id"]))
        if lab is None:
            continue  # dropped by the judge (drop-never-coerce)
        entry = npd.get(str(r["feat_id"])) or {}
        rows.append(
            {
                **r,
                "level": lab["level"],
                "persona_related": lab["persona_related"],
                "label": lab["label"],
                "reasoning": lab.get("reasoning", ""),
                "consistency": cons.get(r["feat_id"], float("nan")),
                "align_max_abs_cos": al_max.get(r["feat_id"], float("nan")),
                "align_top_trait": al_trait.get(r["feat_id"], ""),
                "align_abs_cos_per_trait": al_per.get(r["feat_id"], {}),
                "neuronpedia_description": entry.get("description", ""),
                "prior_round_level": prior_lv.get(r["feat_id"]),
            }
        )
    return rows


def phase_analyze(args) -> None:
    """Rank statistics + figures + the committed JSON summary. No fits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    rows = _build_rows(args)
    sel = json.loads((args.work / "selection.json").read_text())
    lv = json.loads((args.work / "feature_levels.json").read_text())
    rng = np.random.default_rng(PERM_SEED)

    lev = np.asarray([r["level"] for r in rows])
    r2 = np.asarray([r["r2"] for r in rows])
    act = np.asarray([r["activity"] for r in rows])
    cons = np.asarray([r["consistency"] for r in rows])
    dec = np.asarray([r["decile"] for r in rows])
    m_ab = np.asarray([r["a_best"] for r in rows])
    m_aw = np.asarray([r["a_worst"] for r in rows])
    m_bb = np.asarray([r["b_best"] for r in rows])
    m_bw = np.asarray([r["b_worst"] for r in rows])
    pers = np.asarray([r["persona_related"] for r in rows])
    amax = np.asarray([r["align_max_abs_cos"] for r in rows])

    # ── Set A: global tails (activity-confounded by construction) ──
    set_a = {
        "definition": (
            f"top-{N_TAIL} and bottom-{N_TAIL} of 16,384 answer-side features by "
            "per-feature held-out R2; NOT activity-controlled"
        ),
        "best": _composition(lev[m_ab]),
        "worst": _composition(lev[m_aw]),
        "contrast": _arm_contrast(lev[m_ab], lev[m_aw]),
        "unclear_contrast": _unclear_contrast(lev[m_ab], lev[m_aw]),
        "persona_best": _composition_field(pers[m_ab], PERSONA_LEVELS),
        "persona_worst": _composition_field(pers[m_aw], PERSONA_LEVELS),
        "persona_contrast": _binary_contrast(pers[m_ab], pers[m_aw], "yes", "no"),
        "r2_range_best": [float(r2[m_ab].min()), float(r2[m_ab].max())],
        "r2_range_worst": [float(r2[m_aw].min()), float(r2[m_aw].max())],
        "median_r2_by_level": {},
        "activity_confound": {
            "median_activity_best": float(np.median(act[m_ab])),
            "median_activity_worst": float(np.median(act[m_aw])),
            "decile_counts_best": np.bincount(dec[m_ab], minlength=FC.N_DECILES).tolist(),
            "decile_counts_worst": np.bincount(dec[m_aw], minlength=FC.N_DECILES).tolist(),
            "mannwhitney_activity_best_vs_worst": None,
        },
    }
    mw_act = mannwhitneyu(act[m_ab], act[m_aw], alternative="two-sided")
    set_a["activity_confound"]["mannwhitney_activity_best_vs_worst"] = {
        "U": float(mw_act.statistic),
        "p": float(mw_act.pvalue),
    }
    for tail, mask in (("best", m_ab), ("worst", m_aw)):
        per = {}
        for level in LEVELS:
            v = r2[mask & (lev == level)]
            per[level] = {
                "n": int(len(v)),
                "median_r2": float(np.median(v)) if len(v) else None,
                "median_r2_ci95": _boot_median_ci(v, rng),
            }
        set_a["median_r2_by_level"][tail] = per

    # ── Set B: activity-controlled (equal per-decile arms) ──
    per_decile = []
    for d in range(FC.N_DECILES):
        best_m, worst_m = m_bb & (dec == d), m_bw & (dec == d)
        per_decile.append(
            {
                "decile": d,
                "best": _composition(lev[best_m]),
                "worst": _composition(lev[worst_m]),
                "median_r2_best": float(np.median(r2[best_m])) if best_m.any() else None,
                "median_r2_worst": float(np.median(r2[worst_m])) if worst_m.any() else None,
                "persona_best": _composition_field(pers[best_m], PERSONA_LEVELS),
                "persona_worst": _composition_field(pers[worst_m], PERSONA_LEVELS),
            }
        )
    lohi_b = (m_bb | m_bw) & np.isin(lev, ["low", "high"])
    perm = _stratified_perm((lev[lohi_b] == "high").astype(float), m_bb[lohi_b], dec[lohi_b], rng)
    yn_b = (m_bb | m_bw) & np.isin(pers, ["yes", "no"])
    perm_pers = _stratified_perm((pers[yn_b] == "yes").astype(float), m_bb[yn_b], dec[yn_b], rng)
    set_b = {
        "definition": (
            f"within each of the {FC.N_DECILES} activity deciles, the {N_DECILE_TAIL} best "
            f"and {N_DECILE_TAIL} worst by per-feature R2; equal per-decile arm sizes make "
            "the pooled contrast activity-controlled by design"
        ),
        "pooled_best": _composition(lev[m_bb]),
        "pooled_worst": _composition(lev[m_bw]),
        "pooled_contrast": _arm_contrast(lev[m_bb], lev[m_bw]),
        "unclear_contrast": _unclear_contrast(lev[m_bb], lev[m_bw]),
        "persona_pooled_best": _composition_field(pers[m_bb], PERSONA_LEVELS),
        "persona_pooled_worst": _composition_field(pers[m_bw], PERSONA_LEVELS),
        "persona_pooled_contrast": _binary_contrast(pers[m_bb], pers[m_bw], "yes", "no"),
        "persona_stratified_permutation": perm_pers,
        "stratified_permutation": perm,
        "per_decile": per_decile,
    }

    # ── union-level rank reads ──
    lohi = np.isin(lev, ["low", "high"])
    is_high = (lev == "high").astype(float)
    union = {
        "n_judged": int(len(rows)),
        "n_low_high": int(lohi.sum()),
        "spearman_high_vs_r2": float(FC._spearman(is_high[lohi], r2[lohi])),
        "partial_spearman_high_r2_given_activity": float(
            FC._partial_spearman(is_high[lohi], r2[lohi], act[lohi])
        ),
        "spearman_high_vs_activity": float(FC._spearman(is_high[lohi], act[lohi])),
        "spearman_high_vs_consistency": float(FC._spearman(is_high[lohi], cons[lohi])),
        "spearman_consistency_vs_r2": float(FC._spearman(cons, r2)),
        # Within-answer consistency is the reference round's STRONG R2 correlate, so
        # the level effect is also reported net of it (and net of both covariates).
        "partial_spearman_high_r2_given_consistency": _partial_spearman_multi(
            is_high[lohi], r2[lohi], [cons[lohi]]
        ),
        "partial_spearman_high_r2_given_activity_and_consistency": _partial_spearman_multi(
            is_high[lohi], r2[lohi], [act[lohi], cons[lohi]]
        ),
        "composition": _composition(lev),
        "persona_composition": _composition_field(pers, PERSONA_LEVELS),
        "spearman_persona_yes_vs_r2": float(
            FC._spearman(
                (pers[np.isin(pers, ["yes", "no"])] == "yes").astype(float),
                r2[np.isin(pers, ["yes", "no"])],
            )
        ),
        "partial_spearman_persona_yes_r2_given_activity": _partial_spearman_multi(
            (pers[np.isin(pers, ["yes", "no"])] == "yes").astype(float),
            r2[np.isin(pers, ["yes", "no"])],
            [act[np.isin(pers, ["yes", "no"])]],
        ),
        "median_r2_by_persona": {
            lv_: {
                "n": int((pers == lv_).sum()),
                "median_r2": float(np.median(r2[pers == lv_])) if (pers == lv_).any() else None,
                "median_r2_ci95": _boot_median_ci(r2[pers == lv_], rng),
                "median_activity": float(np.median(act[pers == lv_]))
                if (pers == lv_).any()
                else None,
                "median_align_max_abs_cos": float(np.median(amax[pers == lv_]))
                if (pers == lv_).any()
                else None,
            }
            for lv_ in PERSONA_LEVELS
        },
        # Does the JUDGE's persona field agree with the judge-free decoder-alignment
        # read? Load-bearing given the persona field's low test-retest kappa.
        "spearman_persona_yes_vs_align_max_abs_cos": float(
            FC._spearman(
                (pers[np.isin(pers, ["yes", "no"])] == "yes").astype(float),
                amax[np.isin(pers, ["yes", "no"])],
            )
        ),
    }

    # ── prior-round label agreement on the overlap ──
    ov = [r for r in rows if r["prior_round_level"] is not None]
    agree = sum(1 for r in ov if r["level"] == r["prior_round_level"])
    prior_agreement = {
        "n_overlap": len(ov),
        "n_agree": agree,
        "frac_agree": (agree / len(ov)) if ov else float("nan"),
        "pairs": [
            {
                "feat_id": r["feat_id"],
                "this_round": r["level"],
                "prior_round": r["prior_round_level"],
            }
            for r in ov
        ],
        "note": (
            "the reference round's stratified draw rarely hits the R2 extremes, so the "
            "overlap is small; agreement is a consistency check, not a reliability estimate"
        ),
    }

    # ── mechanical alignment read over ALL 16,384 features (judge-free) ──
    com = FC._load_committed()
    al = np.load(args.work / "align.npz")
    al_traits = [str(t) for t in al["trait_names"]]
    r2_all, act_all = com["r2"], com["activity"]
    assert np.array_equal(al["feat_ids"], com["feat_ids"]), "align.npz feature order drift"
    amax_all = al["max_abs_cos"]
    abscos_all = al["abs_cos"]
    r2_pct = (FC._rank(r2_all) - 0.5) / len(r2_all) * 100.0
    top_idx = np.argsort(amax_all, kind="stable")[-ALIGN_TOP_N:][::-1]
    alignment = {
        "definition": (
            f"|cos(SAE decoder column, r_B)| at layer {SAE_LAYER} for every one of the "
            f"{len(r2_all)} answer-side features; r_B = the #779 monitoring per-layer "
            "mean-difference persona directions; ONE matrix product, judge-free"
        ),
        "layer": SAE_LAYER,
        "traits": al_traits,
        "n_features": int(len(r2_all)),
        "max_abs_cos_quantiles": {
            q: float(np.quantile(amax_all, v))
            for q, v in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))
        },
        "spearman_max_abs_cos_vs_r2": float(FC._spearman(amax_all, r2_all)),
        "partial_spearman_max_abs_cos_r2_given_activity": _partial_spearman_multi(
            amax_all, r2_all, [act_all]
        ),
        "spearman_max_abs_cos_vs_activity": float(FC._spearman(amax_all, act_all)),
        "per_trait": {
            t: {
                "spearman_abs_cos_vs_r2": float(FC._spearman(abscos_all[:, i], r2_all)),
                "partial_given_activity": _partial_spearman_multi(
                    abscos_all[:, i], r2_all, [act_all]
                ),
                "max_abs_cos": float(abscos_all[:, i].max()),
                "median_abs_cos": float(np.median(abscos_all[:, i])),
            }
            for i, t in enumerate(al_traits)
        },
        "top_aligned": {
            "n": int(ALIGN_TOP_N),
            "min_max_abs_cos": float(amax_all[top_idx].min()),
            "median_r2": float(np.median(r2_all[top_idx])),
            "median_r2_percentile": float(np.median(r2_pct[top_idx])),
            "mean_r2_percentile": float(np.mean(r2_pct[top_idx])),
            "median_activity": float(np.median(act_all[top_idx])),
            "trait_counts": {
                t: int((al["top_trait_idx"][top_idx] == i).sum()) for i, t in enumerate(al_traits)
            },
            "feat_ids": [int(x) for x in al["feat_ids"][top_idx]],
        },
        "population_reference": {
            "median_r2_all": float(np.median(r2_all)),
            "median_r2_percentile_by_construction": 50.0,
        },
    }

    # ── cross-dispatch replication of the level headline ──
    ref_path = args.work / "extremes_refrubric.json"
    replication = {"available": False}
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        ref_lv = {int(f["feat_id"]): f["level"] for f in ref["features"]}
        both = [r for r in rows if r["feat_id"] in ref_lv]
        agree = sum(1 for r in both if r["level"] == ref_lv[r["feat_id"]])
        rsa, rsb = ref["set_a_global_tails"], ref["set_b_activity_controlled"]
        replication = {
            "available": True,
            "note": (
                "an EARLIER dispatch of this same round used the reference rubric "
                "verbatim (no persona field); comparing the two gives an independent "
                "replication of the level headline and a level-label stability read"
            ),
            "n_common": len(both),
            "n_level_agree": agree,
            "frac_level_agree": (agree / len(both)) if both else float("nan"),
            "reference_rubric_dispatch": {
                "kappa_level": ref["judge"]["test_retest"]["kappa_level"],
                "set_b_frac_high_best": rsb["pooled_contrast"]["frac_high_best"],
                "set_b_frac_high_worst": rsb["pooled_contrast"]["frac_high_worst"],
                "set_b_perm_p": rsb["stratified_permutation"]["p_two_sided"],
                "set_a_fisher_p": rsa["contrast"]["fisher_exact_2x2"]["p"],
                "set_b_unclear_fisher_p": rsb["unclear_contrast"]["fisher_exact_2x2"]["p"],
            },
        }

    doc = {
        "provenance": _provenance(),
        "question": (
            "Are the best-predicted answer-side SAE features qualitatively different "
            "from the worst-predicted ones — in judged abstraction level, in judged "
            "persona-relatedness, and in mechanical alignment with persona-vector "
            "directions?"
        ),
        "selection": {k: v for k, v in sel.items() if k not in ("features", "idx")},
        "judge": {
            k: lv[k]
            for k in (
                "n_items",
                "n_labeled",
                "drops",
                "retest_drops",
                "judge_model",
                "max_tokens",
                "rubric_sha256_system",
                "rubric_sha256_reference_prefix",
                "test_retest",
                "instrument_note",
            )
        },
        "neuronpedia": json.loads((args.work / "neuronpedia_summary.json").read_text()),
        "set_a_global_tails": set_a,
        "set_b_activity_controlled": set_b,
        "union": union,
        "alignment_mechanical": alignment,
        "cross_dispatch_replication": replication,
        "prior_round_agreement": prior_agreement,
        "features": rows,
    }
    (OUT_EVAL / "extremes.json").write_text(json.dumps(doc, indent=1))
    np.savez(
        OUT_EVAL / "extremes_perfeature.npz",
        feat_ids=np.asarray([r["feat_id"] for r in rows], dtype=np.int64),
        r2=r2,
        activity=act,
        consistency=cons,
        decile=dec,
        level=lev.astype("U8"),
        persona_related=pers.astype("U8"),
        align_max_abs_cos=amax,
        a_best=m_ab,
        a_worst=m_aw,
        b_best=m_bb,
        b_worst=m_bw,
    )

    # ── figures ──
    # One colour = one meaning across BOTH figures: the first two palette slots are
    # reserved for the ARM factor (best/worst) and disjoint slots encode the LEVEL
    # factor, so no palette pair ever means two different things.
    set_paper_style()
    pal = paper_palette(6)
    c_best, c_worst = pal[0], pal[1]
    # "unclear" takes the neutral role colour — semantically apt (no legible property)
    # and far enough from the arm blue that adjacent panels cannot be confused.
    level_color = {
        "low": pal[2],
        "high": pal[3],
        "unclear": paper_palette_role("neutral"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    x = np.arange(FC.N_DECILES)
    fb = [w["best"]["frac_high_of_low_high"] for w in per_decile]
    fw = [w["worst"]["frac_high_of_low_high"] for w in per_decile]
    ax.bar(x - 0.19, fb, width=0.36, color=c_best, label="best 15 by $R^2$")
    ax.bar(x + 0.19, fw, width=0.36, color=c_worst, label="worst 15 by $R^2$")
    ax.axhline(set_b["pooled_contrast"]["frac_high_best"], color=c_best, lw=1, ls="--", alpha=0.7)
    ax.axhline(set_b["pooled_contrast"]["frac_high_worst"], color=c_worst, lw=1, ls="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xlabel("activity decile (low → high)")
    ax.set_ylabel('fraction judged "high-level"\n(of low+high, within decile)')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Set B — activity-controlled", fontsize=9)

    ax = axes[1]
    bottoms = np.zeros(2)
    for k, level in enumerate(LEVELS):
        vals = np.asarray(
            [set_b["pooled_best"][f"frac_{level}"], set_b["pooled_worst"][f"frac_{level}"]]
        )
        ax.bar([0, 1], vals, bottom=bottoms, width=0.55, color=level_color[level], label=level)
        for j, (v, b0) in enumerate(zip(vals, bottoms)):
            if v > 0.04:
                ax.text(j, b0 + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=8)
        bottoms = bottoms + vals
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            f"best\n(n={set_b['pooled_best']['n']})",
            f"worst\n(n={set_b['pooled_worst']['n']})",
        ]
    )
    ax.set_ylabel("fraction of judged features")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Set B — pooled level composition", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "extremes_level_composition_setB", dir=OUT_FIGS)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    r2c = np.clip(r2, -1, None)
    for level in LEVELS:
        m = (m_ab | m_aw) & (lev == level)
        ax.scatter(
            act[m],
            r2c[m],
            s=16,
            alpha=0.65,
            color=level_color[level],
            linewidths=0,
            label=f"{level} (n={int(m.sum())})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("answer-side activity (fraction of fit rows active, log)")
    ax.set_ylabel("per-feature held-out $R^2$ (clipped at $-1$)")
    ax.axhline(0, color="black", lw=0.7, alpha=0.4)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Set A — global $R^2$ tails, coloured by judged level", fontsize=9)

    ax = axes[1]
    for j, (tail, mask) in enumerate((("best", m_ab), ("worst", m_aw))):
        for k, level in enumerate(LEVELS):
            st = set_a["median_r2_by_level"][tail][level]  # k only positions the x offset
            if st["median_r2"] is None:
                continue
            med = st["median_r2"]
            lo, hi = st["median_r2_ci95"]
            xpos = j * 3 + k * 0.7
            ax.errorbar(
                xpos,
                med,
                yerr=[[max(0.0, med - lo)], [max(0.0, hi - med)]],
                fmt="o",
                ms=5,
                color=level_color[level],
                capsize=3,
                label=level if j == 0 else None,
            )
            ax.annotate(
                f"n={st['n']}",
                (xpos, med),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=7,
            )
    ax.set_xticks([0.7, 3.7])
    ax.set_xticklabels(
        [f"best {N_TAIL}", f"worst {N_TAIL}"],
    )
    ax.set_ylabel("median per-feature $R^2$ (95% bootstrap CI)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Set A — median $R^2$ by level within each tail", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "extremes_setA_tails", dir=OUT_FIGS)
    plt.close(fig)

    # ── figure 3: persona-relatedness (judged) + mechanical alignment ──
    # Deliberately NO third colour factor: the persona panel is coloured by ARM
    # (same blue/orange as figure 1's arm factor), so one colour keeps one meaning
    # across every figure; the persona category rides the bar HEIGHT instead.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    arms = [
        ("Set A\nbest", set_a["persona_best"], set_a["persona_contrast"]["frac_yes_best"], c_best),
        (
            "Set A\nworst",
            set_a["persona_worst"],
            set_a["persona_contrast"]["frac_yes_worst"],
            c_worst,
        ),
        (
            "Set B\nbest",
            set_b["persona_pooled_best"],
            set_b["persona_pooled_contrast"]["frac_yes_best"],
            c_best,
        ),
        (
            "Set B\nworst",
            set_b["persona_pooled_worst"],
            set_b["persona_pooled_contrast"]["frac_yes_worst"],
            c_worst,
        ),
    ]
    xs = np.arange(len(arms))
    ax.bar(xs, [f for _n, _b, f, _c in arms], width=0.6, color=[c for *_r, c in arms])
    for j, (_n, blk, frac, _c) in enumerate(arms):
        ax.text(j, frac + 0.012, f"{frac:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(
            j,
            0.012,
            f"{blk['n_yes']}/{blk['n_yes'] + blk['n_no']}\n({blk['n_unclear']} unclear)",
            ha="center",
            va="bottom",
            fontsize=7,
            color="white",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, *_r in arms], fontsize=8.5)
    ax.set_ylabel('fraction judged "persona-related"\n(of yes+no)')
    ax.set_ylim(0, max(0.32, max(f for _n, _b, f, _c in arms) * 1.35))
    ax.set_title(
        "Judged persona-relatedness — test-retest $\\kappa$ = "
        f"{lv['test_retest']['kappa_persona_related']:.2f}, UNRELIABLE per feature",
        fontsize=8.5,
    )

    ax = axes[1]
    ax.scatter(
        amax_all,
        np.clip(r2_all, -1, None),
        s=2,
        alpha=0.10,
        color=paper_palette_role("baseline"),
        rasterized=True,
        linewidths=0,
        label=f"all {len(r2_all)} features",
    )
    qs = np.quantile(amax_all, np.linspace(0, 1, 11))
    bins = [(amax_all >= qs[i]) & (amax_all <= qs[i + 1]) for i in range(10)]
    mx = [float(np.median(amax_all[b])) for b in bins]
    my = [float(np.median(np.clip(r2_all, -1, None)[b])) for b in bins]
    ax.plot(
        mx,
        my,
        color=paper_palette_role("accent"),
        lw=2,
        marker="o",
        ms=4,
        label="decile median",
    )
    ax.set_xlabel(f"persona alignment  max$_t$ |cos(decoder, $r_B^t$)|,  layer {SAE_LAYER}")
    ax.set_ylabel("per-feature held-out $R^2$ (clipped at $-1$)")
    ax.axhline(0, color="black", lw=0.7, alpha=0.4)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(
        "Judge-free mechanical read: Spearman = "
        f"{alignment['spearman_max_abs_cos_vs_r2']:+.3f} "
        f"({alignment['partial_spearman_max_abs_cos_r2_given_activity']:+.3f} | activity)",
        fontsize=8.5,
    )
    fig.tight_layout()
    savefig_paper(fig, "extremes_persona_alignment", dir=OUT_FIGS)
    plt.close(fig)
    _log(
        "analyze done: extremes.json + extremes_perfeature.npz + 2 figures; "
        f"Set B frac-high best {set_b['pooled_contrast']['frac_high_best']:.3f} vs worst "
        f"{set_b['pooled_contrast']['frac_high_worst']:.3f}, "
        f"stratified perm p={perm['p_two_sided']:.4f}; "
        f"persona-yes best {set_b['persona_pooled_contrast']['frac_yes_best']:.3f} vs worst "
        f"{set_b['persona_pooled_contrast']['frac_yes_worst']:.3f} "
        f"(perm p={perm_pers['p_two_sided']:.4f}); "
        f"mechanical Spearman(max|cos|, R2)={alignment['spearman_max_abs_cos_vs_r2']:+.3f}"
    )


# ── dashboard ────────────────────────────────────────────────────────────────

_CSS = """
:root { --fg:#16181d; --mut:#5b6270; --line:#e3e6ec; --bg:#fbfbfd; --card:#fff; }
* { box-sizing:border-box; }
body { margin:0; padding:28px 22px 60px; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif; }
.wrap { max-width:1080px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:17px; margin:38px 0 4px; padding-top:14px; border-top:1px solid var(--line); }
p, li { color:var(--mut); font-size:13.5px; }
.head { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; margin-bottom:10px; }
.head p { margin:6px 0; }
.head b { color:var(--fg); }
table.kv { border-collapse:collapse; font-size:12.5px; margin:8px 0 2px; }
table.kv td { padding:2px 14px 2px 0; color:var(--mut); vertical-align:top; }
table.kv td.k { color:var(--fg); white-space:nowrap; }
.card { background:var(--card); border:1px solid var(--line); border-radius:9px;
  padding:12px 14px; margin:9px 0; }
.card .top { display:flex; flex-wrap:wrap; align-items:baseline; gap:10px; }
.rank { color:var(--mut); font-size:12px; min-width:34px; }
.fid { font-weight:600; font-size:14.5px; }
.fid a { color:#1b4fd8; text-decoration:none; }
.fid a:hover { text-decoration:underline; }
.badge { font-size:11px; padding:1px 7px; border-radius:9px; border:1px solid var(--line);
  color:var(--mut); }
.badge.high { background:#eef4ff; border-color:#c9d9fb; color:#1b3f9b; }
.badge.low { background:#fff4ec; border-color:#f6d6bd; color:#8a4a12; }
.badge.unclear { background:#f2f3f6; }
.badge.pers-yes { background:#eafaf1; border-color:#bfe6d2; color:#0f6b43; }
.badge.pers-no { background:#f7f7f9; }
.badge.pers-unclear { background:#f2f3f6; }
.nums { color:var(--mut); font-size:12px; font-variant-numeric:tabular-nums; }
.lab { margin:6px 0 0; font-size:13.5px; color:var(--fg); }
.why, .np { margin:4px 0 0; font-size:12.5px; color:var(--mut); }
details { margin-top:8px; }
summary { cursor:pointer; font-size:12.5px; color:#1b4fd8; }
.snip { border-left:3px solid var(--line); padding:5px 0 5px 10px; margin:8px 0;
  white-space:pre-wrap; font-size:12px; color:#2c313b; }
.snip .m { color:var(--mut); font-size:11px; }
.trunc { color:#a2470f; font-size:11px; }
"""


def _fmt(v: float, nd: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def _card(rank: int, r: dict, texts: dict[int, str], top: dict) -> str:
    fid = r["feat_id"]
    url = NP_FEATURE_URL.format(model=NP_MODEL_ID, source=NP_SOURCE_ID, index=fid)
    sets = [k.replace("_", "-") for k in ("a_best", "a_worst", "b_best", "b_worst") if r[k]]
    parts = [
        '<div class="card"><div class="top">',
        f'<span class="rank">#{rank}</span>',
        f'<span class="fid">feature <a href="{url}" target="_blank" '
        f'rel="noopener">{fid}</a></span>',
        f'<span class="badge {html.escape(r["level"])}">{html.escape(r["level"])}</span>',
        f'<span class="badge pers-{html.escape(r["persona_related"])}">persona: '
        f"{html.escape(r['persona_related'])}</span>",
        f'<span class="nums">R² {r["r2"]:.4f} &nbsp;·&nbsp; activity '
        f"{r['activity']:.5f} (decile {r['decile']}) &nbsp;·&nbsp; within-answer "
        f"consistency {_fmt(r['consistency'])} &nbsp;·&nbsp; persona alignment "
        f"max|cos| {_fmt(r['align_max_abs_cos'], 4)}"
        f"{' (' + html.escape(r['align_top_trait']) + ')' if r['align_top_trait'] else ''}"
        "</span>",
        f'<span class="badge">{html.escape(", ".join(sets))}</span>',
        "</div>",
    ]
    if r["label"]:
        parts.append(f'<p class="lab"><b>Judged property:</b> {html.escape(r["label"])}</p>')
    if r["reasoning"]:
        parts.append(f'<p class="why"><b>Judge reasoning:</b> {html.escape(r["reasoning"])}</p>')
    per_trait = r.get("align_abs_cos_per_trait") or {}
    if per_trait:
        parts.append(
            '<p class="why"><b>Persona-vector alignment</b> |cos(decoder, r_B)| at layer '
            f"{SAE_LAYER}: "
            + " &nbsp;·&nbsp; ".join(
                f"{html.escape(t)} {v:.4f}" for t, v in sorted(per_trait.items())
            )
            + "</p>"
        )
    nd = r["neuronpedia_description"]
    parts.append(
        f'<p class="np"><b>Neuronpedia auto-interp ({html.escape(NP_EXPL_MODEL)}, '
        f"token-level dashboard on a different corpus):</b> "
        f"{html.escape(nd) if nd else '<em>none published for this feature</em>'}</p>"
    )
    entries = top.get(str(fid), [])[: FC.TOP_K_CONTEXTS]
    parts.append(
        f"<details><summary>top-{len(entries)} firing assistant answers "
        "(verbatim, truncated)</summary>"
    )
    for val, ci in entries:
        raw = texts.get(int(ci), "")
        snip = raw[: FC.SNIPPET_CHARS]
        tail = (
            f'<span class="trunc">… [truncated at {FC.SNIPPET_CHARS} chars; '
            f"full answer is {len(raw)} chars]</span>"
            if len(raw) > FC.SNIPPET_CHARS
            else ""
        )
        parts.append(
            f'<div class="snip"><span class="m">context {int(ci)} · mean activation '
            f"{val:.4f}</span><br>{html.escape(snip)}{tail}</div>"
        )
    parts.append("</details></div>")
    return "".join(parts)


def phase_dashboard(args) -> None:
    """Render the self-contained best/worst-predicted feature dashboard."""
    from explore_persona_space.task_workflow import find_task_path

    doc = json.loads((OUT_EVAL / "extremes.json").read_text())
    rows = {r["feat_id"]: r for r in doc["features"]}
    top = json.loads((args.work / "sample_top_contexts.json").read_text())
    texts: dict[int, str] = {}
    for src in (PRIOR_WORK / "texts.jsonl", args.work / "texts.jsonl"):
        if not src.exists():
            continue
        for ln in src.read_text(encoding="utf-8").split("\n"):
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if rec.get("kind") != "chunk_done":
                texts[int(rec["ci"])] = rec["response"]

    best = sorted((r for r in rows.values() if r["a_best"]), key=lambda r: -r["r2"])
    worst = sorted((r for r in rows.values() if r["a_worst"]), key=lambda r: r["r2"])
    sa, sb = doc["set_a_global_tails"], doc["set_b_activity_controlled"]
    jd = doc["judge"]

    head = f"""<div class="head">
<p><b>What is shown.</b> The {len(best)} best-predicted and {len(worst)}
worst-predicted answer-side SAE features (Set A, global tails) of the issue-1482
context-features &rarr; answer-features ridge map, each with the abstraction level a
Sonnet judge assigned from its top-{FC.TOP_K_CONTEXTS} strongest-firing assistant
answers.</p>
<p><b>Per-arm provenance.</b> Answers are on-policy Qwen-2.5-7B-Instruct seed-42
generations from the parent run (the SAME single generation pool for every feature —
no per-arm generation difference); SAE feature activations are teacher-forced
BatchTopK (k=64) encodings of layer-19 residual-post activations of those answers;
per-feature <b>R&sup2;</b> is the committed held-out R&sup2; of the
context-features&rarr;answer-features ridge
(<code>eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz</code>);
<b>activity</b> is the fraction of fit rows in which the feature is answer-active;
<b>within-answer consistency</b> is the mean fraction of an answer's tokens where the
feature fires, given it fires at all
(<code>eval_results/issue_1482/feature_correlates/consistency_perfeature.npz</code>).
Answer snippets are shown <b>verbatim</b> with no substitutions, truncated at
{FC.SNIPPET_CHARS} characters — every truncation is marked inline. Snippets are real
user-facing assistant text from the parent corpus and are not screened for content.</p>
<p><b>Judge instrument.</b> {html.escape(jd["judge_model"])}, reason-then-label,
max_tokens {jd["max_tokens"]}, one draw per feature, malformed/out-of-range returns
dropped never coerced (drops: {jd["drops"]}); rubric sha16
<code>{html.escape(jd["rubric_sha256_system"])}</code> — identical to the reference
feature-correlates round; test-retest &kappa; = {_fmt(jd["test_retest"]["kappa_level"])}
(n = {jd["test_retest"]["n"]}). Neuronpedia auto-interp descriptions ride each judge
item as labeled auxiliary evidence.</p>
<table class="kv">
<tr><td class="k">Set A (this dashboard, global tails)</td><td>frac judged high-level:
best {_fmt(sa["contrast"]["frac_high_best"])} vs worst
{_fmt(sa["contrast"]["frac_high_worst"])} (of low+high); Fisher p =
{_fmt(sa["contrast"]["fisher_exact_2x2"]["p"])}. <b>Activity-confounded by
construction</b> — median activity best {
        _fmt(sa["activity_confound"]["median_activity_best"], 5)
    } vs worst {_fmt(sa["activity_confound"]["median_activity_worst"], 5)}.</td></tr>
<tr><td class="k">Set B (activity-controlled read)</td><td>frac judged high-level:
best {_fmt(sb["pooled_contrast"]["frac_high_best"])} vs worst
{_fmt(sb["pooled_contrast"]["frac_high_worst"])}; decile-stratified permutation p =
{_fmt(sb["stratified_permutation"]["p_two_sided"])}.</td></tr>
<tr><td class="k">Provenance</td><td>commit
<code>{html.escape(doc["provenance"]["git_commit"][:12])}</code>,
{html.escape(doc["provenance"]["timestamp_utc"])}; full numbers in
<code>eval_results/issue_1482/feature_extremes/extremes.json</code>.</td></tr>
</table>
</div>"""

    body = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Issue 1482 — best vs worst predicted answer-side SAE features</title>",
        f"<style>{_CSS}</style></head><body><div class='wrap'>",
        "<h1>Best- vs worst-predicted answer-side SAE features (issue #1482)</h1>",
        head,
        f"<h2>Best-predicted — top {len(best)} by held-out R&sup2;</h2>",
        f"<p>Ranked by descending R&sup2; ({best[0]['r2']:.4f} down to {best[-1]['r2']:.4f}).</p>",
    ]
    body += [_card(i + 1, r, texts, top) for i, r in enumerate(best)]
    body += [
        f"<h2>Worst-predicted — bottom {len(worst)} by held-out R&sup2;</h2>",
        f"<p>Ranked by ascending R&sup2; ({worst[0]['r2']:.4f} up to {worst[-1]['r2']:.4f}).</p>",
    ]
    body += [_card(i + 1, r, texts, top) for i, r in enumerate(worst)]
    body.append("</div></body></html>")

    out = find_task_path(1482) / "artifacts" / "feature_extremes_dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(body), encoding="utf-8")
    _log(f"dashboard done: {out} ({out.stat().st_size / 1024:.0f} KiB, {len(rows)} features)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue 1482 feature-extremes inline round.")
    ap.add_argument(
        "--phase",
        required=True,
        choices=["neuronpedia", "scan", "texts", "judge", "align", "analyze", "dashboard"],
    )
    ap.add_argument("--store", type=Path, default=FC.STORE_DEFAULT)
    ap.add_argument("--work", type=Path, default=WORK_DEFAULT)
    args = ap.parse_args()
    {
        "neuronpedia": phase_neuronpedia,
        "scan": phase_scan,
        "texts": phase_texts,
        "judge": phase_judge,
        "align": phase_align,
        "analyze": phase_analyze,
        "dashboard": phase_dashboard,
    }[args.phase](args)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit before C-extension finalize teardown


if __name__ == "__main__":
    main()
