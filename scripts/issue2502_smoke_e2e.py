"""Issue #2502 u4 — CPU per-phase end-to-end smoke driver (plan v6 smoke leg).

Drives every VM-runnable pipeline phase at a tiny slice THROUGH THE PRODUCTION
ENTRYPOINTS (`issue2502_corpus.main`, `issue2502_gen_capture.phase_capture`,
`issue2502_reliability.run_subset/run_ceiling`), with exactly the external
boundaries mocked/redirected:

  (1) `corpus_staging._hf_stream` -> synthetic fixture rows with PLANTED
      near-duplicates / rejects (the network + gated-dataset boundary; the
      plan's trigger-dense discipline mandates synthetic fixtures for
      code-review rounds — the real-corpus `--probe` is a pre-launch
      orchestrator step).
  (2) `issue2502_gen_capture.load_model_ctx` -> a signature-conformant
      FakeCaptureModel emitting real-shape bf16 hidden states on CPU (the
      model-load + CUDA boundary). `forward_batch`, `capture_chunk`, the MF-F
      asserts, the bf16 codec, chunk staging, HF upload + exact-set verify,
      and the resume scans all run REAL.
  (3) `issue2502_gen_capture.enforce_model_env` no-op'd for the Model-B leg
      ONLY (the #2378 pod-venv pin boundary; Model A runs the real branch).
  (4) `issue2502_reliability.{SUBSET_PREFIX,RELIABILITY_ROOT}` redirected to
      the smoke namespace (scratch-prefix redirect, not a behavior mock).
  (5) vLLM generation (P1) is POD-ONLY: the `gen` leg writes gen-chunk rows in
      `phase_gen`'s exact record schema (real tokenizer + real
      `render_prompt_ids`/`assert_chat_template`/`ids_sha16`) with synthetic
      completion token ids sampled from each row's own prompt ids — and runs
      the REAL u2 accounting around them: `load_corpus_rows` (validation +
      the SB-1(i) full-corpus content sha16) + the 3-arg `gen_regime` +
      `ensure_remote_regime` (raw-prefix digest published BEFORE any chunk
      upload; SB-1(iii) first-publication seal) + `chunk_key` +
      `count_chunk_stats` + `write_gen_meta`, so the capture leg's consumer
      gates (regime verify + `require_gen_complete`) execute REAL against this
      leg's artifacts.

All HF writes land under ``issue2502_ctxmap_xgen/smoke_u4/`` — never the
canonical run prefixes. Content hygiene: fixture texts are benign synthetic
sentences; no corpus/real text is generated, printed, or uploaded.

Legs (one per invocation; each timeout-bounded by the caller):
  corpus       P0 probe + build + upload + fingerprint-resume re-run + planted
               dedup/split/LODO assertions
  gen          gen-chunk writer per model (--model-key A|B [--rep-seed S])
  capture      REAL phase_capture per model (mock forward); --rep-seed routes
               to the reliability replicate prefixes
  rel-subset   reliability subset draw on the smoke corpus
  rel-ceiling  reliability ceiling per model through the REAL HfChunkStore
               read path at the smoke prefixes (+ ledger-resume re-run)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# r3 subprefix: r2 artifacts carry the OLD 2-arg regime shape (no
# corpus_content_sha16, num_shards stripped) and pre-seeded chunk files would
# trip the SB-1(iii) first-publication seal; a fresh nested namespace keeps
# every regime / completeness gate clean while staying confined to smoke_u4/
# (hygiene rule).
SMOKE_PREFIX = "issue2502_ctxmap_xgen/smoke_u4/r3"
# jbb_behaviors added round 9: the multi-split staging class (configs x
# splits attempts; the live crash's own source) must be smoke-covered per
# arm class, not left to production (#1090 fu5).
SMOKE_SOURCES = ("lmsys_chat_1m", "itw_jailbreak", "writingprompts", "jbb_behaviors")
FIXTURE_SEED = 20260823
GEN_SEED_MAIN = 42
CHUNK_SIZE = 100

_WORDS = (
    "harbor lantern meadow crystal violet ember quartz willow saffron marble "
    "cobalt juniper falcon breeze summit tundra mosaic cedar prism canyon "
    "velvet aurora thicket garnet lagoon pebble orchard zephyr drift walnut"
).split()


def _sentence(rng: random.Random, n_words: int, tag: str) -> str:
    body = " ".join(rng.choice(_WORDS) for _ in range(n_words))
    return f"fixture {tag} please describe {body}"


def _gc():
    import issue2502_gen_capture as GC

    return GC


# ---------------------------------------------------------------------------
# corpus leg
# ---------------------------------------------------------------------------


def build_fixtures() -> tuple[dict, dict]:
    """{(dataset_id, config): [raw rows]} + the planted-outcome expectations."""
    rng = random.Random(FIXTURE_SEED)

    def lmsys_row(text: str, *, flagged: bool = False, language: str = "English") -> dict:
        return {
            "conversation": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "fixture assistant reply"},
            ],
            "language": language,
            "toxic": flagged,
            "redacted": False,
        }

    lmsys: list[dict] = []
    for i in range(95):
        lmsys.append(lmsys_row(_sentence(rng, rng.randint(14, 40), f"lmsys-ord-{i:03d}")))
    for i in range(30):
        lmsys.append(
            lmsys_row(_sentence(rng, rng.randint(14, 40), f"lmsys-flag-{i:03d}"), flagged=True)
        )
    # Planted WITHIN-source near-dup pair (exact 5-gram Jaccard >= 0.8): long
    # shared body, one trailing word changed -> the later row must be dropped.
    base_a = _sentence(rng, 60, "lmsys-neardup")
    near_a = base_a + " tail alpha"
    near_a2 = base_a + " tail omega"
    lmsys.append(lmsys_row(near_a))
    lmsys.append(lmsys_row(near_a2))
    # Planted EXACT duplicate (within-source normalized-text dedup counter).
    exact = _sentence(rng, 25, "lmsys-exactdup")
    lmsys.append(lmsys_row(exact))
    lmsys.append(lmsys_row(exact))
    # Planted rejects: non-English + too-short + too-long-tokens.
    lmsys.append(lmsys_row(_sentence(rng, 20, "lmsys-pt"), language="Portuguese"))
    lmsys.append(lmsys_row("tiny"))
    # > MAX_TEXT_CHARS (usable_text char bound, reject "too_long"):
    lmsys.append(lmsys_row("fixture toolong " + "palavra " * 6500))
    # <= MAX_TEXT_CHARS but > the ~5.6k context-token budget (token-dense
    # digits; TokenLengthFilter reject "too_long_tokens"):
    lmsys.append(lmsys_row("fixture tokendense " + "2 4 6 8 " * 1500))

    # Cross-source near-dup: itw_jailbreak carries a >=0.8 twin of base_b (in
    # lmsys) -> dropped at the ACROSS-source stage.
    base_b = _sentence(rng, 55, "cross-neardup")
    lmsys.append(lmsys_row(base_b + " ending north"))
    itw_1 = [{"prompt": _sentence(rng, rng.randint(12, 35), f"itw-a-{i:03d}")} for i in range(22)]
    itw_2 = [{"prompt": _sentence(rng, rng.randint(12, 35), f"itw-b-{i:03d}")} for i in range(22)]
    itw_2.append({"prompt": base_b + " ending south"})
    # Borderline pair: LSH-candidate-range overlap but exact Jaccard < 0.8 ->
    # BOTH must survive (the candidate stage alone must not delete).
    seg_shared = " ".join(rng.choice(_WORDS) for _ in range(30))
    seg_x = " ".join(rng.choice(_WORDS) for _ in range(12))
    seg_y = " ".join(rng.choice(_WORDS) for _ in range(12))
    border_1 = f"fixture borderline {seg_shared} {seg_x}"
    border_2 = f"fixture borderline {seg_shared} {seg_y}"
    itw_1.append({"prompt": border_1})
    itw_1.append({"prompt": border_2})

    wp = [{"prompt": _sentence(rng, rng.randint(12, 30), f"wp-{i:03d}")} for i in range(30)]

    # jbb multi-split fixtures (round 9): the SAME (dataset, config) staged
    # from TWO splits — keyed with an explicit split element so each split
    # serves distinct rows (field 'Goal', the real behaviors-config field).
    jbb_h = [{"Goal": _sentence(rng, rng.randint(12, 30), f"jbb-h-{i:03d}")} for i in range(12)]
    jbb_b = [{"Goal": _sentence(rng, rng.randint(12, 30), f"jbb-b-{i:03d}")} for i in range(12)]

    fixtures = {
        ("lmsys/lmsys-chat-1m", None): lmsys,
        ("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25"): itw_1,
        ("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_05_07"): itw_2,
        ("euclaise/writingprompts", None): wp,
        ("JailbreakBench/JBB-Behaviors", "behaviors", "harmful"): jbb_h,
        ("JailbreakBench/JBB-Behaviors", "behaviors", "benign"): jbb_b,
    }
    expect = {
        "confirmed_dropped": 2,  # within-lmsys near-dup + cross-source near-dup
        "border_texts": (border_1, border_2),
        "near_texts": (near_a, near_a2),
    }
    return fixtures, expect


def _find_counter(tree, name: str) -> int:
    """Recursive sum of every ``name`` counter in a nested report dict."""
    total = 0
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == name and isinstance(v, int):
                total += v
            else:
                total += _find_counter(v, name)
    elif isinstance(tree, list):
        for v in tree:
            total += _find_counter(v, name)
    return total


def leg_corpus(args) -> None:
    import issue2502_corpus as CP
    from explore_persona_space.experiments.issue_1739 import corpus_staging as CS

    fixtures, expect = build_fixtures()

    def fake_hf_stream(dataset_id, config=None, split="train", **kw):
        # Split-keyed fixtures first (multi-split sources, round 9), then the
        # split-agnostic (dataset, config) key for single-split sources.
        key3 = (dataset_id, config, split)
        if key3 in fixtures:
            yield from fixtures[key3]
            return
        key = (dataset_id, config)
        if key not in fixtures:
            raise RuntimeError(f"smoke fixture has no rows for {key3}")
        yield from fixtures[key]

    CS._hf_stream = fake_hf_stream  # network/gated-read boundary (mock 1)
    # u1's HfApi().dataset_info revision-pin seam (same network boundary as
    # mock 1; corpus.py resolves the module global at both call sites).
    # Signature mirrors the round-9 (dataset_id, revision_ref) shape.
    CP._resolve_dataset_revision = lambda dataset_id, revision_ref=None: "smokerev"

    out_dir = Path(args.work) / "corpus"
    common = [
        "--out-dir",
        str(out_dir),
        "--sources",
        *SMOKE_SOURCES,
        "--budget",
        "400",
        "--seed",
        "42",
        # The registry-wide split preflight calls the REAL
        # datasets.get_dataset_split_names (a network metadata read) — the
        # offline smoke skips it; the preflight itself is pytest-pinned with
        # injected seams (tests/test_issue2502_smoke_pins.py round 9) and
        # network-verified against the real registry pre-relaunch.
        "--skip-split-preflight",
    ]
    rc = CP.main([*common, "--probe"])
    assert rc == 0, f"probe rc={rc}"
    probe = json.loads((out_dir / "probe_report.json").read_text())
    assert probe["mode"] == "probe" and probe["dedup"]["n_in"] > 0

    rc = CP.main([*common, "--upload", "--upload-prefix", f"{SMOKE_PREFIX}/context_corpus"])
    assert rc == 0, f"build rc={rc}"
    report = json.loads((out_dir / "dedup_report.json").read_text())
    # split("\n"), never .splitlines(): raw U+2028/U+2029/NEL inside
    # ensure_ascii=False JSON strings shred records under splitlines (#950).
    corpus = [
        json.loads(ln) for ln in (out_dir / "corpus.jsonl").read_text().split("\n") if ln.strip()
    ]

    # --- planted-outcome assertions (MF-K two-stage dedup) -------------------
    dd = report["dedup"]
    assert dd["n_confirmed_dropped"] == expect["confirmed_dropped"], dd
    assert dd["n_had_lsh_candidate"] >= dd["n_confirmed_dropped"], dd
    b1, b2 = expect["border_texts"]
    j = CP._exact_jaccard(CP._char_ngrams(CS.norm_text(b1)), CP._char_ngrams(CS.norm_text(b2)))
    assert 0.50 <= j < 0.80, f"borderline fixture drifted: jaccard={j:.3f}"
    shas = {r["context_sha"] for r in corpus}
    assert CP._context_sha(b1) in shas and CP._context_sha(b2) in shas, (
        "borderline (<0.8) pair must BOTH survive the confirm stage"
    )
    kept_near = [t for t in expect["near_texts"] if CP._context_sha(t) in shas]
    assert len(kept_near) == 1, "planted >=0.8 near-dup pair: exactly one row must survive"
    assert _find_counter(report["stream_counters"], "non_english") >= 1
    assert _find_counter(report["stream_counters"], "too_long") >= 1  # char bound
    assert _find_counter(report["stream_counters"], "too_long_tokens") >= 1  # token budget
    assert _find_counter(report["stream_counters"], "dup_text_within_source") >= 1
    assert _find_counter(report["stream_counters"], "too_short") >= 1

    # --- split + LODO assertions ---------------------------------------------
    by_src: dict[str, dict[str, int]] = {}
    for r in corpus:
        assert r["lodo_group"] == r["source_tag"]
        by_src.setdefault(r["source_tag"], {}).setdefault(r["split"], 0)
        by_src[r["source_tag"]][r["split"]] += 1
    for tag, splits in by_src.items():
        assert set(splits) == {"train", "val", "test"}, (tag, splits)
    # Round-9 multi-split staging class: BOTH jbb splits staged to DISTINCT
    # checkpoint files (the non-'train' split lands in a __<split>-suffixed
    # file) and both fixture sets survive into the corpus (24 distinct rows).
    n_jbb = sum(v for v in by_src.get("jbb_behaviors", {}).values())
    assert n_jbb == 24, f"jbb_behaviors multi-split rows: {n_jbb} != 24"
    for split_file in (
        "jbb_behaviors__behaviors__harmful.jsonl",
        "jbb_behaviors__behaviors__benign.jsonl",
    ):
        assert (out_dir / "staged" / split_file).exists(), split_file
    classes = {r["regime_class"] for r in corpus}
    assert classes == {"ordinary", "weird", "near-distribution", "idiosyncratic"}, classes
    n_ord_test = sum(1 for r in corpus if r["regime_class"] == "ordinary" and r["split"] == "test")
    assert n_ord_test >= 12, f"decide-phase floor: ordinary test rows {n_ord_test} < 12"

    # --- fingerprint-gated resume re-run (staged files reused) ---------------
    rc = CP.main(common)
    assert rc == 0
    corpus2 = [
        json.loads(ln) for ln in (out_dir / "corpus.jsonl").read_text().split("\n") if ln.strip()
    ]
    assert [r["context_sha"] for r in corpus2] == [r["context_sha"] for r in corpus], (
        "fingerprint-resume rebuild must be byte-stable on context shas"
    )
    print(
        f"[smoke-corpus] OK: n_final={len(corpus)} sources={sorted(by_src)} "
        f"dedup={dd['n_confirmed_dropped']} dropped / {dd['n_had_lsh_candidate']} candidates; "
        f"ordinary test rows={n_ord_test}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# gen leg (phase_gen record schema; vLLM itself is pod-only)
# ---------------------------------------------------------------------------


def _model_flags(model_key: str) -> list[str]:
    if model_key == "A":
        return ["--model", "Qwen/Qwen2.5-7B-Instruct", "--env", "repo-standard"]
    return [
        "--model",
        "Qwen/Qwen3.5-9B",
        "--env",
        "pod2378-venv",
        "--disable-thinking",
        "--gdn-prefill",
        "triton",
    ]


def smoke_prefixes(model_key: str, rep_seed: int | None) -> dict:
    if rep_seed is None:
        return {
            "corpus": f"{SMOKE_PREFIX}/context_corpus",
            "raw": f"{SMOKE_PREFIX}/raw_completions/final/model{model_key}",
            "tensors": f"{SMOKE_PREFIX}/analysis_tensors/model{model_key}",
        }
    return {
        "corpus": f"{SMOKE_PREFIX}/reliability_subset",
        "raw": f"{SMOKE_PREFIX}/raw_completions/reliability/model{model_key}/rep{rep_seed}",
        "tensors": f"{SMOKE_PREFIX}/analysis_tensors/reliability/model{model_key}/rep{rep_seed}",
    }


def leg_gen(args) -> None:
    """Write + upload gen chunks in phase_gen's EXACT record schema.

    Real tokenizer, real template assert, real render/sha helpers; completion
    token ids are sampled from each row's own prompt ids (in-vocab by
    construction; replicate seeds vary the sample -> across-replicate answer
    variance for the reliability ICC). The REAL u2 accounting brackets the
    writes: the raw-prefix regime digest is published FIRST (capture's
    consumer-side `ensure_remote_regime(write_if_absent=False)` requires it)
    and `write_gen_meta` runs LAST (capture's `require_gen_complete` gate +
    the cap-hit fraction arithmetic run REAL — 2 planted cap hits over the
    full smoke corpus stay under the 2% halt threshold)."""
    GC = _gc()
    from transformers import AutoTokenizer

    model_key = args.model_key
    seed = args.rep_seed if args.rep_seed is not None else GEN_SEED_MAIN
    prefixes = smoke_prefixes(model_key, args.rep_seed)
    work = Path(args.work) / f"gen_model{model_key}_seed{seed}"
    work.mkdir(parents=True, exist_ok=True)
    a = GC.build_parser().parse_args(
        [
            "--phase",
            "gen",
            *_model_flags(model_key),
            "--seed",
            str(seed),
            "--corpus-prefix",
            prefixes["corpus"],
            "--raw-prefix",
            prefixes["raw"],
            "--out-prefix",
            prefixes["tensors"],
            "--chunk-size",
            str(CHUNK_SIZE),
            "--work-dir",
            str(work),
        ]
    )
    disable = a.disable_thinking
    tok = AutoTokenizer.from_pretrained(a.model)
    template_sha = GC.assert_chat_template(tok, disable_thinking=disable)
    # Production order (RV3-u1 SB-1(i)): load_corpus_rows FIRST — its
    # full-corpus content sha16 is a REQUIRED gen_regime input — then the
    # 3-arg regime, then the remote-regime seal (all REAL u2 accounting).
    rows, corpus_sha = GC.load_corpus_rows(a, work)
    regime = GC.gen_regime(a, template_sha, corpus_sha)
    GC.ensure_remote_regime(prefixes["raw"], regime, work, write_if_absent=True)
    chunks = [rows[i : i + CHUNK_SIZE] for i in range(0, len(rows), CHUNK_SIZE)]
    keys = [GC.chunk_key(a, ci) for ci in range(len(chunks))]
    rng = random.Random(seed * 1000003 + (0 if model_key == "A" else 1))
    plant_main = args.rep_seed is None and model_key == "A"
    # Deterministic plants on TRAIN rows of chunk 0 (never test/val, so the
    # decide-phase + reliability-subset floors are untouched): the FIRST train
    # row gets an empty completion (capture's empty_completion drop branch),
    # the next two get cap-hit rows (finish_reason=="length" accounting), the
    # FOURTH gets think_leak=True (capture's think_leak drop branch).
    plant_empty_rj = plant_leak_rj = None
    plant_cap_rjs: set[int] = set()
    if plant_main:
        train_rjs = [rj for rj, r in enumerate(chunks[0]) if r.get("split") == "train"]
        assert len(train_rjs) >= 4, "smoke corpus chunk0 needs >=4 train rows for plants"
        plant_empty_rj, plant_leak_rj = train_rjs[0], train_rjs[3]
        plant_cap_rjs = set(train_rjs[1:3])
    stats: dict[str, dict] = {}
    for ci, chunk_rows in enumerate(chunks):
        key = keys[ci]
        out = work / f"{key}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for rj, row in enumerate(chunk_rows):
                pids = GC.render_prompt_ids(tok, row["text"], disable_thinking=disable)
                cap_hit = bool(plant_main and ci == 0 and rj in plant_cap_rjs)
                if plant_main and ci == 0 and rj == plant_empty_rj:
                    comp_ids: list[int] = []  # planted empty-completion drop
                else:
                    comp_ids = [rng.choice(pids) for _ in range(rng.randint(5, 24))]
                rec = {
                    "context_id": row["context_id"],
                    "context_sha": row["context_sha"],
                    "source_tag": row.get("source_tag"),
                    "dataset_id": row.get("dataset_id"),
                    "config": row.get("config"),
                    "regime_class": row.get("regime_class"),
                    "realism_tier": row.get("realism_tier"),
                    "split": row.get("split"),
                    "lodo_group": row.get("lodo_group"),
                    "prompt_sha": GC.ids_sha16(pids),
                    "n_prompt_tokens": len(pids),
                    "completion": f"fixture answer {rj}",
                    "completion_token_ids": comp_ids,
                    "n_gen_tokens": len(comp_ids),
                    "finish_reason": "length" if cap_hit else "stop",
                    "cap_hit": cap_hit,
                    "think_leak": bool(plant_main and ci == 0 and rj == plant_leak_rj),
                    "eos_stripped": False,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[key] = GC.count_chunk_stats(out)  # REAL recount, pre-unlink
        GC.upload_single_file(out, f"{prefixes['raw']}/{key}.jsonl")
        out.unlink()
    GC.write_gen_meta(a, work, keys, stats, template_sha, regime)
    print(
        f"[smoke-gen] model {model_key} seed={seed}: {len(chunks)} chunks x <= {CHUNK_SIZE} rows "
        f"-> {prefixes['raw']} (regime + gen_meta published)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# capture leg (REAL phase_capture; model boundary mocked)
# ---------------------------------------------------------------------------


class FakeCaptureModel:
    """Signature-conformant stand-in for the HF model at the forward boundary.

    Mirrors the real call surface `model(input_ids=, attention_mask=,
    output_hidden_states=, use_cache=, **lk)` — including the explicit
    `logits_to_keep` param, asserted ==1 so the smoke proves `forward_batch`
    threads `mctx.lk` (#779) — and returns an object with `.hidden_states`
    = tuple of n_layers+1 (b, t, H) bf16 tensors. Values are a deterministic
    per-token embedding passed through a causal 16-token window mean
    (context-keyed signal; replicate variance enters via the sampled
    completion ids), layer-varied by a roll + scale."""

    VOCAB_BUCKETS = 4096
    WINDOW = 16

    def __init__(self, torch_mod, n_layers: int, hidden: int, model_name: str):
        self.torch = torch_mod
        self.n_layers = n_layers
        self.hidden = hidden
        seed = int(hashlib.sha256(model_name.encode()).hexdigest()[:8], 16)
        g = torch_mod.Generator().manual_seed(seed)
        self.emb = torch_mod.randn(self.VOCAB_BUCKETS, hidden, generator=g) * 0.7

    def __call__(
        self,
        *,
        input_ids=None,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=True,
        logits_to_keep=None,
    ):
        torch = self.torch
        assert output_hidden_states and not use_cache
        assert logits_to_keep == 1, "forward_batch must thread mctx.lk logits_to_keep=1 (#779)"
        e = self.emb[input_ids % self.VOCAB_BUCKETS]  # (b, t, H) fp32
        e = e * attention_mask[..., None].to(e.dtype)
        cs = e.cumsum(1)
        shifted = torch.zeros_like(cs)
        if cs.shape[1] > self.WINDOW:
            shifted[:, self.WINDOW :] = cs[:, : -self.WINDOW]
        t_idx = torch.arange(cs.shape[1])
        counts = torch.clamp(torch.minimum(t_idx + 1, torch.tensor(self.WINDOW)), min=1)
        wm = (cs - shifted) / counts[None, :, None].to(cs.dtype)
        hidden_states = []
        for k in range(self.n_layers + 1):
            h = torch.roll(wm, shifts=k % 7, dims=-1) * (1.0 + 0.03 * k)
            hidden_states.append(h.to(torch.bfloat16))

        class _Out:
            pass

        out = _Out()
        out.hidden_states = tuple(hidden_states)
        return out


def leg_capture(args) -> None:
    GC = _gc()
    model_key = args.model_key
    prefixes = smoke_prefixes(model_key, args.rep_seed)
    seed = args.rep_seed if args.rep_seed is not None else GEN_SEED_MAIN
    argv = [
        "--phase",
        "capture",
        *_model_flags(model_key),
        "--seed",
        str(seed),
        "--corpus-prefix",
        prefixes["corpus"],
        "--raw-prefix",
        prefixes["raw"],
        "--out-prefix",
        prefixes["tensors"],
        "--chunk-size",
        str(CHUNK_SIZE),
        "--max-batch-rows",
        "16",
        "--work-dir",
        str(Path(args.work) / f"cap_model{model_key}_seed{seed}"),
    ]
    a = GC.build_parser().parse_args(argv)
    spec = GC.MODEL_SPECS[a.model]
    GC.validate_model_flags(a, spec)  # REAL per-model flag contract

    import torch

    fake = FakeCaptureModel(torch, spec["n_layers"], spec["hidden"], a.model)

    def fake_load_model_ctx(args_, spec_, tok):
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        return GC.ModelCtx(
            torch=torch,
            model=fake,
            device="cpu",
            n_layers=spec_["n_layers"],
            hidden=spec_["hidden"],
            pad_id=int(pad_id),
            # Exercise forward_batch's **mctx.lk threading (#779): the fake
            # forward asserts it receives logits_to_keep=1.
            lk={"logits_to_keep": 1},
        )

    GC.load_model_ctx = fake_load_model_ctx  # model/GPU boundary (mock 2)
    if spec["requires_env"] == "pod2378-venv":
        # #2378 venv pin boundary (mock 3) — Model A keeps the REAL branch.
        GC.enforce_model_env = lambda args_: print(
            "[smoke-capture] enforce_model_env no-op (pod-venv pin boundary)", flush=True
        )
    GC.phase_capture(a, spec)
    if args.rep_seed is None and model_key == "A":
        # Planted-drop accounting (A main only — the leg that planted): read the
        # REAL capture_meta from work_root (= <work-dir>/model{K}) and assert
        # every planted drop/cap branch fired exactly once/twice.
        meta = json.loads(
            (Path(a.work_dir) / f"model{model_key}" / "capture_meta.json").read_text(
                encoding="utf-8"
            )
        )
        t = meta["totals"]
        assert t["n_empty_completion_drops"] == 1, t
        assert t["n_think_leak_drops"] == 1, t
        assert t["n_cap_hit"] == 2, t
        print(f"[smoke-capture] planted-drop accounting OK: {t}", flush=True)
    print(f"[smoke-capture] model {model_key} seed={seed} OK -> {prefixes['tensors']}", flush=True)


# ---------------------------------------------------------------------------
# reliability legs (prefix constants redirected to the smoke namespace)
# ---------------------------------------------------------------------------


def _patched_rl():
    import issue2502_reliability as RL

    RL.SUBSET_PREFIX = f"{SMOKE_PREFIX}/reliability_subset"
    RL.RELIABILITY_ROOT = SMOKE_PREFIX
    return RL


def leg_rel_subset(args) -> None:
    RL = _patched_rl()
    a = RL.build_parser().parse_args(
        [
            "--phase",
            "subset",
            "--corpus-prefix",
            f"{SMOKE_PREFIX}/context_corpus",
            "--subset-size",
            "24",
            "--min-per-class",
            "2",
            "--work-dir",
            str(Path(args.work) / "rel_subset"),
        ]
    )
    res = RL.run_subset(a)
    assert res.get("skipped") or res["n_rows"] >= 8, res
    print(f"[smoke-rel-subset] OK: {res}", flush=True)


def leg_rel_ceiling(args) -> None:
    RL = _patched_rl()
    # Smoke default is a SCRATCH out-root; the canonical committed
    # eval_results/issue_2502 tree is explicit opt-in (--out-root) so an
    # ordinary smoke can never overwrite committed artifacts.
    out_root = args.out_root or str(Path(args.work) / "eval_results")
    argv = [
        "--phase",
        "ceiling",
        "--model-key",
        args.model_key,
        "--rep-seeds",
        "43,44",
        "--publish",
        "none",
        "--work-dir",
        str(Path(args.work) / f"rel_ceiling_{args.model_key}"),
        "--out-root",
        out_root,
    ]
    a = RL.build_parser().parse_args(argv)
    doc = RL.run_ceiling(a)
    assert doc["coverage"] == 1.0, doc["coverage"]
    for cell in doc["per_layer"].values():
        c = cell["ceiling_pooled"]
        assert 0.02 < c < 0.9999, f"ceiling out of sane range: {c} at hs {cell['hs']}"
    doc2 = RL.run_ceiling(a)  # ledger-resume re-run (cells served from percell)
    assert doc2["per_layer"].keys() == doc["per_layer"].keys()
    print(
        f"[smoke-rel-ceiling] model {args.model_key} OK: {len(doc['per_layer'])} layers, "
        f"coverage={doc['coverage']}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--leg",
        default=None,
        choices=("corpus", "gen", "capture", "rel-subset", "rel-ceiling"),
        help="required unless --import-check",
    )
    ap.add_argument("--model-key", choices=("A", "B"), default=None)
    ap.add_argument("--rep-seed", type=int, default=None, help="reliability replicate seed")
    ap.add_argument("--work", default="/tmp/issue2502-smoke-u4-r3")
    ap.add_argument("--out-root", default=None, help="rel-ceiling: eval_results out-root")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("issue2502_smoke_e2e: import-check OK", flush=True)
        return 0
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    Path(args.work).mkdir(parents=True, exist_ok=True)
    if args.leg is None:
        raise SystemExit("--leg is required (unless --import-check)")
    if args.leg in ("gen", "capture", "rel-ceiling") and args.model_key is None:
        raise SystemExit(f"--model-key required for leg {args.leg}")
    dispatch = {
        "corpus": leg_corpus,
        "gen": leg_gen,
        "capture": leg_capture,
        "rel-subset": leg_rel_subset,
        "rel-ceiling": leg_rel_ceiling,
    }
    dispatch[args.leg](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
