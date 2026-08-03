"""#1773's DEFERRED register-axis validator: zero-shot steering transfer.

#1773 shipped the `register_style` axis on a LEXICAL proxy (`register_informality`:
n=1,233 features, mean 0.00561 vs a 0.00439 baseline) and recorded its substantive
validator as a plan deviation -- "the intervention rig is out of pilot scope".

The lexical proxy asks a CORRELATIONAL question: does text this feature fires on
look informal? This script asks the CAUSAL one: add the feature's decoder direction
to the residual stream and does the generated text's register actually move, in the
direction the feature's own description predicts?

Scope: this validates the register AXIS. It does not reopen #1773's overall
search-index-only verdict.

Phases (`--phase`), split so the GPU is released before the API-bound judging --
the standing #664 rule:

  gen      GPU, on-pod. Generates a shared unsteered baseline pool plus, for every
           direction, `--n-prompts` x `--n-draws` steered continuations. Batches
           ACROSS DIRECTIONS: the forward hook carries a per-batch-row steering
           vector, so one generate() call advances many features at once.
  judge    API-only, off-pod. Judges register (0-100) + coherence for every text,
           and separately judges each feature's DESCRIPTION for the register
           direction it predicts. Routes through the project judge dispatcher, so
           large sets go to the Batch API automatically.
  analyze  CPU. Per-feature register shift vs its own paired baseline, coherence
           rate, and direction match; then the aggregate transfer rate against the
           matched nulls.

Three directions arms, because "register features move register" is uninterpretable
without knowing whether ANY direction moves register:

  register    the 1,233 `register_style`-labelled features
  null_other  features labelled something OTHER than `register_style` -- the
              strong null (a real, interpretable SAE feature that is not about
              register)
  null_random random unit directions -- the weak null (generic perturbation)

The DIRECTION claim gets its own matched null: a label-shuffle permutation of the
predicted directions across register features, which centres the match rate at
chance by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

REGISTER_RUBRIC = (
    "You rate the REGISTER of a piece of text on a 0-100 scale.\n"
    "0 = maximally formal (academic, legal, or technical prose; full sentences; "
    "no contractions, slang, or interjections).\n"
    "50 = neutral everyday prose.\n"
    "100 = maximally informal (casual chat, slang, contractions, interjections, "
    "emoji, fragments).\n\n"
    "Rate ONLY register. Ignore topic, correctness, and whether the text is any "
    "good. Judge the text as written, even if it is truncated or incoherent.\n\n"
    "First give one sentence of justification, then the integer.\n"
    'Reply as JSON: {"reasoning": "<one sentence>", "score": <integer 0-100>}'
)

COHERENCE_RUBRIC = (
    "You judge whether a piece of text is COHERENT English.\n"
    "Coherent = grammatical, on-topic, and readable, even if short or dull.\n"
    "Incoherent = degenerate repetition, word salad, broken syntax, or text that "
    "abandons the prompt entirely.\n\n"
    "First give one sentence of justification, then the verdict.\n"
    'Reply as JSON: {"reasoning": "<one sentence>", "coherent": true or false}'
)

# Judges the feature DESCRIPTION, not any generated text: what register would this
# feature push toward if it were amplified? `neither` is a first-class answer --
# plenty of register_style features are about tone or genre rather than the
# formal/informal axis the register rubric scores, and scoring those for direction
# would manufacture coin-flips.
DIRECTION_RUBRIC = (
    "You are given a description of a feature found inside a language model. The "
    "feature was labelled as being about SPEAKER REGISTER or STYLE.\n\n"
    "Question: if this feature were amplified so the model produced more of what "
    "the description describes, would the text become MORE FORMAL, MORE INFORMAL, "
    "or NEITHER?\n\n"
    "formal   = more academic, technical, legal, professional, restrained.\n"
    "informal = more casual, conversational, slangy, emotive, playful.\n"
    "neither  = the description is not about the formal/informal axis at all "
    "(e.g. it is about a topic, a language, a syntactic position, or a style "
    "dimension orthogonal to formality).\n\n"
    "Answer for the description as written. Do not guess if it is genuinely "
    "orthogonal -- answer neither.\n\n"
    "First give one sentence of justification, then the verdict.\n"
    'Reply as JSON: {"reasoning": "<one sentence>", '
    '"direction": "formal" or "informal" or "neither"}'
)

# Register-neutral prompts: each admits both a formal and an informal answer, so a
# register shift is expressible. Deliberately generic -- a topic-loaded prompt would
# confound register with content.
PROMPTS = [
    "Explain what a hash table is.",
    "Describe what happened in your last conversation about travel plans.",
    "Give me your opinion on whether remote work is a good idea.",
    "Tell me about a movie you would recommend.",
    "What should someone do if their laptop won't turn on?",
    "Summarise the argument for eating less meat.",
    "How would you describe the weather today to a friend?",
    "Walk me through how to make coffee.",
]


def _log(msg: str) -> None:
    print(f"[i1773-val] {msg}", flush=True)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    """Short HEAD sha for the reproducibility block; never fatal off a git tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _repro(extra: dict) -> dict:
    import torch
    import transformers

    return {
        "git_commit": _git_sha(),
        "generated_at": _utc(),
        "model": MODEL_ID,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        **extra,
    }


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    """Single O_APPEND write per chunk -- the per-unit persistence contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r) + "\n" for r in rows)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().split("\n") if x.strip()]


# ── direction table ──────────────────────────────────────────────────────────


def _load_labels(labels_path: Path) -> tuple[list[int], list[int]]:
    """Return (register_style feat_ids, non-register interpretable feat_ids)."""
    reg: set[int] = set()
    interpretable: set[int] = set()
    for line in labels_path.read_text().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        fid = int(r["feat_id"])
        if r["axis"] == "speaker_property" and r["label"] == "register_style":
            reg.add(fid)
        if r["axis"] == "interpretable" and r["label"] == "yes":
            interpretable.add(fid)
    others = sorted(interpretable - reg)
    return sorted(reg), others


def build_direction_table(args) -> list[dict]:
    """(kind, dir_id) rows in a stable order; `feat_id` is None for random dirs."""
    reg, others = _load_labels(args.labels)
    if not reg:
        raise SystemExit("no register_style-labelled features found")
    rng = random.Random(args.seed)

    reg_sel = (
        reg
        if args.n_register <= 0 or args.n_register >= len(reg)
        else rng.sample(reg, args.n_register)
    )
    reg_sel = sorted(reg_sel)
    other_sel = sorted(rng.sample(others, min(args.n_null_other, len(others))))

    table = [{"kind": "register", "dir_id": f"reg_{f}", "feat_id": f} for f in reg_sel]
    table += [{"kind": "null_other", "dir_id": f"oth_{f}", "feat_id": f} for f in other_sel]
    table += [
        {"kind": "null_random", "dir_id": f"rnd_{i}", "feat_id": None}
        for i in range(args.n_null_random)
    ]
    _log(
        f"directions: register={len(reg_sel)} null_other={len(other_sel)} "
        f"null_random={args.n_null_random} (pool: reg={len(reg)} other={len(others)})"
    )
    return table


# ── phase: gen ───────────────────────────────────────────────────────────────


def phase_gen(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import issue1482_sae as SAE  # noqa: N812  (module alias, matches sibling scripts)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "generations.jsonl"

    table = build_direction_table(args)
    prompts = PROMPTS[: args.n_prompts]

    # resume: a direction is done when it has all n_prompts x n_draws rows
    done_counts: dict[str, int] = {}
    for r in _read_jsonl(gen_path):
        done_counts[r["dir_id"]] = done_counts.get(r["dir_id"], 0) + 1
    expect = len(prompts) * args.n_draws
    todo = [d for d in table if done_counts.get(d["dir_id"], 0) < expect]
    partial = [d["dir_id"] for d in todo if done_counts.get(d["dir_id"], 0) > 0]
    if partial:
        raise SystemExit(
            f"{len(partial)} direction(s) have a PARTIAL row count "
            f"(e.g. {partial[:3]}); refusing to resume onto a half-written direction. "
            f"Quarantine {gen_path} and re-run."
        )
    _log(f"resume: {len(table) - len(todo)}/{len(table)} directions already complete")

    sae = SAE.BatchTopKSAE.load(k=args.k, layer=args.layer, device="cpu")
    w_dec = sae.w_dec  # (act_dim, dict_size)
    act_dim = w_dec.shape[0]
    _log(f"decoder {tuple(w_dec.shape)}")

    g = torch.Generator().manual_seed(args.seed)
    rand_dirs: dict[str, "torch.Tensor"] = {}
    for d in table:
        if d["kind"] == "null_random":
            v = torch.randn(act_dim, generator=g)
            rand_dirs[d["dir_id"]] = v / v.norm()

    def direction_vec(d: dict) -> "torch.Tensor":
        if d["feat_id"] is None:
            return rand_dirs[d["dir_id"]]
        col = w_dec[:, d["feat_id"]]
        return col / col.norm()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # Qwen2.5 defaults to right-padding; batched generate() then continues past the
    # pads on every row but the longest.
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # per-batch-row steering: V is (B, act_dim), aligned to the rows of the batch
    state: dict[str, object] = {"V": None}

    def hook(_module, _inp, out):
        v = state["V"]
        if v is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        scale = h.norm(dim=-1, keepdim=True) * args.alpha  # (B, T, 1)
        h = h + v.unsqueeze(1) * scale  # (B,1,d) * (B,T,1) -> (B,T,d)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    handle = model.model.layers[args.layer].register_forward_hook(hook)

    chat = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]

    def run_batch(rows: list[tuple[str, int]], V) -> list[str]:
        """rows = [(dir_id, prompt_idx)]; V = (B, act_dim) on device, or None."""
        texts = [chat[pi] for _, pi in rows]
        batch = tok(texts, return_tensors="pt", padding=True).to(model.device)
        state["V"] = V
        try:
            with torch.no_grad():
                g_out = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
        finally:
            state["V"] = None
        cut = batch["input_ids"].shape[1]
        return [tok.decode(g_out[i, cut:], skip_special_tokens=True) for i in range(g_out.shape[0])]

    t0 = time.time()

    # ---- shared unsteered baseline pool ----------------------------------
    base_path = out_dir / "baseline.jsonl"
    if len(_read_jsonl(base_path)) < len(prompts) * args.n_baseline_draws:
        base_path.unlink(missing_ok=True)
        rows = [
            (f"__baseline__", pi)
            for pi in range(len(prompts))
            for _ in range(args.n_baseline_draws)
        ]
        got: list[dict] = []
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i : i + args.batch_size]
            outs = run_batch(chunk, None)
            got += [
                {"dir_id": "__baseline__", "kind": "baseline", "prompt_idx": pi, "text": t}
                for (_, pi), t in zip(chunk, outs, strict=True)
            ]
            _log(f"baseline {min(i + args.batch_size, len(rows))}/{len(rows)}")
        _append_jsonl(base_path, got)
    else:
        _log("baseline pool already complete")

    # ---- steered arms, batched ACROSS directions --------------------------
    n_done = 0
    dirs_per_batch = max(1, args.batch_size // (len(prompts) * args.n_draws))
    for bi in range(0, len(todo), dirs_per_batch):
        group = todo[bi : bi + dirs_per_batch]
        rows: list[tuple[str, int]] = []
        vecs = []
        for d in group:
            v = direction_vec(d)
            for pi in range(len(prompts)):
                for _ in range(args.n_draws):
                    rows.append((d["dir_id"], pi))
                    vecs.append(v)
        V = torch.stack(vecs).to(model.device, dtype=torch.bfloat16)
        t_b = time.time()
        outs = run_batch(rows, V)
        kind_of = {d["dir_id"]: d["kind"] for d in group}
        feat_of = {d["dir_id"]: d["feat_id"] for d in group}
        _append_jsonl(
            gen_path,
            [
                {
                    "dir_id": did,
                    "kind": kind_of[did],
                    "feat_id": feat_of[did],
                    "prompt_idx": pi,
                    "text": t,
                }
                for (did, pi), t in zip(rows, outs, strict=True)
            ],
        )
        n_done += len(group)
        _log(
            f"unit {n_done}/{len(todo)} dirs (batch={len(rows)} rows) "
            f"elapsed={time.time() - t_b:.1f}s total={time.time() - t0:.0f}s"
        )

    handle.remove()
    gen_s = time.time() - t0
    meta = _repro(
        {
            "phase": "gen",
            "layer": args.layer,
            "alpha": args.alpha,
            "k": args.k,
            "n_prompts": len(prompts),
            "n_draws": args.n_draws,
            "n_baseline_draws": args.n_baseline_draws,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "n_directions": len(table),
            "counts": {
                k: sum(1 for d in table if d["kind"] == k)
                for k in ("register", "null_other", "null_random")
            },
            "generation_wall_s": round(gen_s, 1),
        }
    )
    (out_dir / "gen_meta.json").write_text(json.dumps(meta, indent=1))
    _log(f"[phase=gen_done] {len(todo)} directions in {gen_s:.0f}s -> {gen_path}")
    return 0


# ── phase: upload ────────────────────────────────────────────────────────────

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1773_register_steer"
SHARD_BYTES = 9_000_000  # keep every shard on the always-open non-LFS path


def phase_upload(args) -> int:
    """Shard the raw generations under the 10 MB LFS force-route and push them.

    Raw generations are the artifact the whole validator is derived from, so they
    land on the data repo BEFORE the pod is released -- one bulk commit, then an
    exact-set verify against a fresh listing.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    out_dir = args.out_dir
    stage = out_dir / "_hf_stage"
    if stage.exists():
        for p in stage.iterdir():
            p.unlink()
    stage.mkdir(parents=True, exist_ok=True)

    expected: list[str] = []
    for name in ("generations.jsonl", "baseline.jsonl"):
        src = out_dir / name
        if not src.exists():
            continue
        stem = name.removesuffix(".jsonl")
        shard, size, idx = [], 0, 0
        parts: list[tuple[str, int]] = []

        def _flush(shard, idx, stem, parts):
            if not shard:
                return
            fn = f"{stem}.shard{idx:03d}.jsonl"
            (stage / fn).write_text("".join(shard))
            parts.append((fn, len(shard)))

        with open(src, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                b = len(line.encode())
                if size + b > SHARD_BYTES and shard:
                    _flush(shard, idx, stem, parts)
                    shard, size, idx = [], 0, idx + 1
                shard.append(line)
                size += b
        _flush(shard, idx, stem, parts)
        (stage / f"{stem}.manifest.json").write_text(
            json.dumps(
                {"source": name, "shards": [{"file": f, "lines": n} for f, n in parts]}, indent=1
            )
        )
        expected += [f for f, _ in parts] + [f"{stem}.manifest.json"]

    for extra in ("gen_meta.json", "validator.json", "per_feature.jsonl"):
        if (out_dir / extra).exists():
            (stage / extra).write_text((out_dir / extra).read_text())
            expected.append(extra)
    if not expected:
        raise SystemExit(f"nothing to upload under {out_dir}")

    api = HfApi()
    # deterministic guard, outside the retry wrapper: a raise here is never transient
    hub.assert_hub_dir_filecounts(stage, HF_PREFIX)
    _log(f"uploading {len(expected)} files -> {HF_DATA_REPO}:{HF_PREFIX}/")
    hub.retry_transient(
        lambda: api.upload_folder(
            folder_path=str(stage),
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=HF_PREFIX,
            commit_message=f"issue #1773 register steering-transfer raw generations ({_utc()})",
        ),
        what=f"upload_folder {HF_PREFIX}",
    )
    missing = hub.verify_repo_paths_uploaded(
        api,
        HF_DATA_REPO,
        [f"{HF_PREFIX}/{f}" for f in expected],
        path_in_repo=HF_PREFIX,
        repo_type="dataset",
    )
    if missing:
        raise SystemExit(f"upload verify FAILED, {len(missing)} missing: {sorted(missing)[:5]}")
    _log(f"[phase=upload_done] verified {len(expected)} files on {HF_DATA_REPO}:{HF_PREFIX}/")
    return 0


# ── phase: judge ─────────────────────────────────────────────────────────────


def _judge(items: list[tuple[str, str, str, str]], rubric: str, tag: str, ckpt: Path) -> dict:
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items, graded_temperature

    _log(f"judging {len(items)} items ({tag})")
    with graded_temperature(0.0):
        return dispatch_judge_items(
            items,
            judge_system_prompt=rubric,
            max_tokens=400,
            checkpoint_dir=ckpt,
        )


def phase_judge(args) -> int:
    from explore_persona_space.eval.batch_judge import make_custom_id
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    out_dir = args.out_dir
    gens = _read_jsonl(out_dir / "generations.jsonl") + _read_jsonl(out_dir / "baseline.jsonl")
    if not gens:
        raise SystemExit(f"no generations under {out_dir}; run --phase gen first")
    for i, r in enumerate(gens):
        r["row_id"] = f"{r['dir_id']}|{r['prompt_idx']}|{i}"
    _log(f"{len(gens)} generated texts to judge")

    cid_map: dict[str, str] = {}
    for r in gens:
        cid = make_custom_id(r["row_id"])
        if cid in cid_map:
            raise SystemExit(f"custom_id collision on {r['row_id']} vs {cid_map[cid]}")
        cid_map[cid] = r["row_id"]

    for tag, rubric in (("register", REGISTER_RUBRIC), ("coherence", COHERENCE_RUBRIC)):
        dest = out_dir / f"judged_{tag}.json"
        if dest.exists():
            _log(f"{tag}: already judged, skipping")
            continue
        items = [
            (make_custom_id(r["row_id"]), f"i1773:{tag}", "", f"TEXT:\n{r['text']}") for r in gens
        ]
        res = _judge(items, rubric, tag, out_dir / f"ckpt_{tag}")
        dest.write_text(
            json.dumps({cid_map[c]: v for c, v in res.items() if c in cid_map}, indent=1)
        )
        _log(f"{tag}: wrote {dest}")

    # ---- direction prediction from each feature's own description ---------
    dest = out_dir / "judged_direction.json"
    if dest.exists():
        _log("direction: already judged, skipping")
    else:
        desc = {}
        for line in args.descriptions.read_text().split("\n"):
            if line.strip():
                d = json.loads(line)
                desc[int(d["feat_id"])] = d["description"]
        feat_ids = sorted({r["feat_id"] for r in gens if r.get("feat_id") is not None})
        dmap: dict[str, int] = {}
        items = []
        for f in feat_ids:
            if f not in desc:
                continue
            cid = make_custom_id(f"dir|{f}")
            dmap[cid] = f
            items.append((cid, "i1773:direction", "", f"FEATURE DESCRIPTION:\n{desc[f]}"))
        res = _judge(items, DIRECTION_RUBRIC, "direction", out_dir / "ckpt_direction")
        dest.write_text(
            json.dumps({str(dmap[c]): v for c, v in res.items() if c in dmap}, indent=1)
        )
        _log(f"direction: wrote {dest} ({len(items)} descriptions)")

    (out_dir / "judge_meta.json").write_text(
        json.dumps(
            _repro({"phase": "judge", "n_texts": len(gens), "judge_model": "claude-sonnet-4-5"}),
            indent=1,
        )
    )
    _log("[phase=judge_done]")
    return 0


# ── phase: analyze ───────────────────────────────────────────────────────────


def _score(rec) -> float | None:
    if isinstance(rec, dict) and isinstance(rec.get("score"), (int, float)):
        s = float(rec["score"])
        return s if 0.0 <= s <= 100.0 else None
    return None


def phase_analyze(args) -> int:
    import numpy as np

    out_dir = args.out_dir
    gens = _read_jsonl(out_dir / "generations.jsonl") + _read_jsonl(out_dir / "baseline.jsonl")
    for i, r in enumerate(gens):
        r["row_id"] = f"{r['dir_id']}|{r['prompt_idx']}|{i}"
    reg_j = json.loads((out_dir / "judged_register.json").read_text())
    coh_j = json.loads((out_dir / "judged_coherence.json").read_text())
    dir_j = json.loads((out_dir / "judged_direction.json").read_text())

    def _coherent(row_id: str) -> bool | None:
        c = coh_j.get(row_id)
        if isinstance(c, dict) and isinstance(c.get("coherent"), bool):
            return bool(c["coherent"])
        return None

    # per-prompt baseline mean over COHERENT baseline text: the paired contrast each
    # direction is scored against.
    base_by_prompt: dict[int, list[float]] = {}
    for r in gens:
        if r["kind"] != "baseline":
            continue
        s = _score(reg_j.get(r["row_id"]))
        if s is not None and _coherent(r["row_id"]):
            base_by_prompt.setdefault(r["prompt_idx"], []).append(s)
    base_mean = {p: float(np.mean(v)) for p, v in base_by_prompt.items() if v}
    _log(f"baseline per-prompt means (coherent only): {sorted(base_mean.items())}")
    if not base_mean:
        raise SystemExit("no coherent baseline text; cannot form a paired contrast")

    # The pilot (alpha>=0.5, feat 8) scored register +40 on text that was 0%
    # coherent: degenerate output reads as "informal" to the register rubric. So the
    # PRIMARY shift is computed over coherent draws only; the all-draw shift is kept
    # alongside it as the contaminated comparison, never as the headline.
    per_dir: dict[str, dict] = {}
    for r in gens:
        if r["kind"] == "baseline":
            continue
        d = per_dir.setdefault(
            r["dir_id"],
            {
                "kind": r["kind"],
                "feat_id": r["feat_id"],
                "shifts_all": [],
                "shifts_coh": [],
                "coh": [],
                "n": 0,
            },
        )
        d["n"] += 1
        s = _score(reg_j.get(r["row_id"]))
        ok = _coherent(r["row_id"])
        if ok is not None:
            d["coh"].append(ok)
        if s is not None and r["prompt_idx"] in base_mean:
            d["shifts_all"].append(s - base_mean[r["prompt_idx"]])
            if ok:
                d["shifts_coh"].append(s - base_mean[r["prompt_idx"]])

    pred = {}
    for fid, rec in dir_j.items():
        if isinstance(rec, dict) and rec.get("direction") in ("formal", "informal", "neither"):
            pred[int(fid)] = {"formal": -1, "informal": +1, "neither": 0}[rec["direction"]]

    feats = []
    for did, d in per_dir.items():
        p = pred.get(d["feat_id"]) if d["feat_id"] is not None else None
        n_coh = len(d["shifts_coh"])
        usable = n_coh >= args.min_coherent
        shift = float(np.mean(d["shifts_coh"])) if usable else None
        feats.append(
            {
                "dir_id": did,
                "kind": d["kind"],
                "feat_id": d["feat_id"],
                "n_draws": d["n"],
                "n_coherent": n_coh,
                "coherent_rate": round(float(np.mean(d["coh"])), 4) if d["coh"] else None,
                # PRIMARY: coherent draws only. `_all` is the contaminated read the
                # pilot showed is dominated by degeneration.
                "register_shift": None if shift is None else round(shift, 3),
                "register_shift_all": (
                    round(float(np.mean(d["shifts_all"])), 3) if d["shifts_all"] else None
                ),
                "usable": usable,
                "predicted_direction": p,
                "direction_match": (
                    None if (not p or shift is None) else bool(np.sign(shift) == p)
                ),
            }
        )
    kinds = ("register", "null_other", "null_random")
    by_kind = {k: [f for f in feats if f["kind"] == k] for k in kinds}
    usable_by_kind = {k: [f for f in v if f["usable"]] for k, v in by_kind.items()}
    for k in kinds:
        _log(
            f"{k}: {len(usable_by_kind[k])}/{len(by_kind[k])} directions usable "
            f"(>= {args.min_coherent} coherent draws)"
        )

    # magnitude claim: a "mover" clears the 95th pct of |shift| under random dirs.
    # Threshold is built from USABLE random dirs only, so it is a coherent-text
    # threshold compared against coherent-text shifts.
    rnd_abs = np.array([abs(f["register_shift"]) for f in usable_by_kind["null_random"]])
    thr = float(np.percentile(rnd_abs, 95)) if rnd_abs.size else float("nan")
    has_thr = thr == thr  # NaN check: no usable random dirs -> no calibrated threshold

    def _summary(all_rows: list[dict]) -> dict:
        rows = [r for r in all_rows if r["usable"]]
        coh_all = [r["coherent_rate"] for r in all_rows if r["coherent_rate"] is not None]
        out = {
            "n_directions": len(all_rows),
            "n_usable": len(rows),
            "usable_rate": round(len(rows) / len(all_rows), 4) if all_rows else None,
            "coherent_rate_mean": round(float(np.mean(coh_all)), 4) if coh_all else None,
        }
        if not rows:
            return out
        sh = np.array([r["register_shift"] for r in rows])
        movers = [r for r in rows if abs(r["register_shift"]) > thr] if has_thr else []
        out.update(
            {
                "shift_mean": round(float(sh.mean()), 3),
                "abs_shift_mean": round(float(np.abs(sh).mean()), 3),
                "abs_shift_median": round(float(np.median(np.abs(sh))), 3),
                "mover_rate": round(len(movers) / len(rows), 4) if has_thr else None,
            }
        )
        return out

    # direction claim: label-shuffle permutation null (centres match rate at chance)
    directional = [f for f in usable_by_kind["register"] if f["predicted_direction"]]
    match_rate = (
        float(np.mean([f["direction_match"] for f in directional])) if directional else None
    )
    perm = None
    if directional:
        signs = np.sign([f["register_shift"] for f in directional])
        labels = np.array([f["predicted_direction"] for f in directional])
        rng = np.random.default_rng(args.seed)
        draws = np.array(
            [float(np.mean(signs == rng.permutation(labels))) for _ in range(args.n_perm)]
        )
        perm = {
            "n_draws": args.n_perm,
            "null_mean": round(float(draws.mean()), 4),
            "null_p95": round(float(np.percentile(draws, 95)), 4),
            "p_value": round(float((draws >= match_rate).mean()), 5),
        }

    pred_counts = {
        d: sum(1 for f in usable_by_kind["register"] if f["predicted_direction"] == v)
        for d, v in (("formal", -1), ("informal", 1), ("neither", 0))
    }
    payload = _repro(
        {
            "what": "#1773 register-axis zero-shot steering-transfer validator",
            "scope": "validates the register AXIS only; #1773's search-index-only "
            "verdict is untouched",
            "primary_read": "register shift over COHERENT draws only, paired against "
            "the per-prompt coherent baseline mean; `register_shift_all` in "
            "per_feature.jsonl is the contaminated all-draw comparison",
            "min_coherent_draws": args.min_coherent,
            "alpha": args.alpha,
            "mover_threshold_abs_shift": round(thr, 3) if has_thr else None,
            "mover_threshold_basis": "95th percentile of |shift| under usable null_random",
            "by_arm": {k: _summary(v) for k, v in by_kind.items()},
            "direction_claim": {
                "n_directional_register_features": len(directional),
                "predicted_direction_counts": pred_counts,
                "match_rate": round(match_rate, 4) if match_rate is not None else None,
                "shuffle_null": perm,
            },
        }
    )
    (out_dir / "validator.json").write_text(json.dumps(payload, indent=1))
    (out_dir / "per_feature.jsonl").write_text(
        "".join(json.dumps(f) + "\n" for f in sorted(feats, key=lambda x: x["dir_id"]))
    )
    _log(json.dumps(payload["by_arm"], indent=1))
    _log(json.dumps(payload["direction_claim"], indent=1))
    _log(f"[phase=analyze_done] -> {out_dir / 'validator.json'}")
    return 0


# ── cli ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=("gen", "upload", "judge", "analyze"))
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--n-draws", type=int, default=2)
    ap.add_argument("--n-baseline-draws", type=int, default=32)
    ap.add_argument("--n-register", type=int, default=0, help="0 = all register_style features")
    ap.add_argument("--n-null-other", type=int, default=200)
    ap.add_argument("--n-null-random", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1773)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument(
        "--min-coherent",
        type=int,
        default=3,
        help="coherent draws a direction needs before its shift is read (below this "
        "the direction is reported as unusable, not silently averaged)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1773/register_steer"))
    ap.add_argument(
        "--labels", type=Path, default=Path("eval_results/issue_1773/labels/axis_labels.jsonl")
    )
    ap.add_argument(
        "--descriptions",
        type=Path,
        default=Path("eval_results/issue_1773/labels/descriptions.jsonl"),
    )
    args = ap.parse_args()
    return {
        "gen": phase_gen,
        "upload": phase_upload,
        "judge": phase_judge,
        "analyze": phase_analyze,
    }[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
