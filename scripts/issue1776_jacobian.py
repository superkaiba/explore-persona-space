"""#1776 Phase 2: averaged causal Jacobian J_{C->A} estimator (plan v4 §4 formal estimator).

For each stored (context, answer) pair the model is teacher-forced (producer
convention: full chat render tokenized once, answer span = [prompt_len, T) with
prompt_len from the prompt-only render — EXACT parity with #779's
``capture_answer_vector`` / v_x mean pooling, guarded by a prefix-token-identity
seam check). One backward per cotangent-seed chunk yields
``g_t(e, x) = d(e^T v_19(x)) / d h_14,t(x)`` at EVERY prompt position t, where
the differentiation slot is the block-14 OUTPUT (the §4 slot-pinning convention:
``model.model.layers[14]`` forward output == HF ``hidden_states[15]``; the same
tensor ``cx_last(14)`` captures and ``DeltaHook(layer=14)`` edits) and the
readout ``v_19`` = fp32 mean over answer positions of the block-19 output.

Three position-subset arms accumulate from the SAME backward:
  - ``prefix``: sum_{t < prefix_len} g_t      (prefix = template preamble/system)
  - ``ctx``:    sum_{t < context_len} g_t     (prefix + user query)
  - ``last``:   g at the LAST PROMPT TOKEN (index ``prompt_len - 1`` of the
    generation-suffix render — the plan prose's "context_end-1" denotes this
    slot: it is the exact ``cx_last(14)`` capture position (``hs[-1]`` of the
    ``add_generation_prompt=True`` render, issue779_collect L119-137) and the
    ``DeltaHook`` prefill edit position T-1, the unit-matched M' slot).

Half-sum persistence: pairs split even/odd by pair-manifest index; each pair
feeds exactly one half; J = count-weighted mean of the halves at zero extra
backwards. Per-chunk checkpoint + the #952 gate-5 resume manifest
(code SHA, slot pins, seed-set sha, pair-file sha, pooling variant, mode knobs
— match-or-recompute). G-NONZERO gate (plan §7): an all-zero context-gradient
field across the gate pairs HALTs rc=8 (the degenerate same-layer convention).

Content hygiene: pairs carry real LMSYS/WildChat text — this rig NEVER prints
prompt/response text; logs carry pair ids, shapes, norms only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847 gate)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402
from explore_persona_space.analysis.representation_shift import compute_prompt_spans  # noqa: E402

ARMS = ("prefix", "ctx", "last")
POOLING = "answer_mean_fp32"  # #779 v_x parity: .float() then mean over answer positions
SPAN_CONVENTION = "compute_prompt_spans:first_user:snap"


class _EarlyExit(Exception):
    """Raised by the readout hook once block-``readout_layer`` output is captured."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.to(torch.float32).cpu().numpy().tobytes()).hexdigest()[:16]


def load_pairs(pairs_path: Path) -> list[dict]:
    """Load the pair manifest: JSONL rows {"pair_id", "prompt", "response"}."""
    rows = []
    for line in pairs_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            assert {"pair_id", "prompt", "response"} <= set(r), sorted(r)
            rows.append(r)
    assert rows, f"empty pairs file: {pairs_path}"
    return rows


SPAN_SENTINEL = "EPSQSPAN7SENTINEL"  # locates the user-content slot in the rendered template


def _suffix_q_span(tok, prompt_text: str, prompt: str) -> tuple[int, int]:
    """EXACT char span of the final user content, anchored from the template TAIL.

    The legacy ``str.find``-from-0 locator mis-anchors whenever a short
    real-user query substring-matches inside the rendered chat-template
    preamble (Qwen default-system boilerplate, 115 chars): measured on the
    staged corpora 2026-07-30, 4/999 WildChat rows CRASH the strict span
    assert (match inside token 0 -> prefix_len=0 — the p3_pcj pod crash,
    ``AssertionError: (0, 1, 30)``) and 11/999 more SILENTLY mis-anchor
    (garbage spans that pass it); 1/1536 lmsys jpairs rows mis-anchor the same
    way. The single-turn render is ``preamble + content + tail`` with a
    content-independent tail, so anchoring from the tail is exact.
    """
    sent_text = tok.apply_chat_template(
        [{"role": "user", "content": SPAN_SENTINEL}], tokenize=False, add_generation_prompt=True
    )
    i = sent_text.find(SPAN_SENTINEL)
    assert i >= 0 and sent_text.count(SPAN_SENTINEL) == 1, "sentinel template render failed"
    tail = sent_text[i + len(SPAN_SENTINEL) :]
    assert tail and prompt_text.endswith(tail), (
        "chat-template tail drift — the suffix anchor requires a content-independent tail"
    )
    q_end = len(prompt_text) - len(tail)
    q_start = q_end - len(prompt)
    assert q_start >= 0 and prompt_text[q_start:q_end] == prompt, (
        "user content is not verbatim before the template tail (template mutated the content)"
    )
    return q_start, q_end


def legacy_find_anchor_agrees(tok, prompt: str) -> bool:
    """True iff the legacy find-from-0 anchor lands on the exact tail-anchored span.

    Tokenizer-only (~sub-ms). Used by the p3_pcj resume predicate: a retained
    per-context unit computed under the legacy anchor is valid exactly when
    this holds (for every such row the two anchors produce identical spans);
    a disagreeing row's legacy unit carries mis-anchored spans -> recompute.
    """
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    q_start, _ = _suffix_q_span(tok, text, prompt)
    return text.find(prompt) == q_start


def render_pair(tok, prompt: str, response: str, *, anchor: str = "find") -> dict | None:
    """Producer-convention render + span boundaries for one pair.

    Returns {"full_ids", "prompt_len", "prefix_len", "context_len", "anchor"}
    or None on an empty response / a prompt-boundary BPE seam (designed skip —
    counted by the caller; the #779 producer convention has no seam on this
    corpus, the guard keeps a drifted row out of the estimator instead of
    crashing it).

    ``anchor`` selects how the question's char span is located in the render:

    - ``"find"`` (default): first occurrence from char 0 — the parent jac_full
      convention, byte-identical to every prior run of this module.
    - ``"suffix"`` (#1776 p3_pcj crash-fix r15): exact tail-anchored span via
      :func:`_suffix_q_span`; REQUIRED for bare real-user corpora whose short
      queries can substring-match inside the template preamble (see
      ``_suffix_q_span``). For every row where "find" anchors correctly the
      two produce IDENTICAL spans. "suffix" additionally legalizes
      ``prefix_len == 0`` (a template-less render) — ``pair_backward`` then
      refuses LOUDLY if the 3-arm prefix/context split is requested on such a
      row (unreachable under the Qwen chat template, whose preamble is the
      prefix).
    """
    assert anchor in ("find", "suffix"), anchor
    msgs = [{"role": "user", "content": prompt}]
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, return_tensors="pt", padding=False)["input_ids"][0]
    suffix = tok.decode(prompt_ids[-3:])
    assert suffix == C.GENERATION_SUFFIX, f"generation-suffix assert failed: {suffix!r}"
    full_text = tok.apply_chat_template(
        [*msgs, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tok(full_text, return_tensors="pt", padding=False)["input_ids"][0]
    prompt_len = int(prompt_ids.shape[0])
    if full_ids.shape[0] <= prompt_len:
        return None  # empty-response row (producer kept non-empty; guard anyway)
    if not torch.equal(full_ids[:prompt_len], prompt_ids):
        return None  # prompt-boundary BPE seam under the producer convention
    span_kw = (
        {"q_char_span": _suffix_q_span(tok, prompt_text, prompt)} if anchor == "suffix" else {}
    )
    prefix_len, context_len = compute_prompt_spans(
        tok, None, prompt, [int(x) for x in prompt_ids], on_seam="snap", **span_kw
    )
    min_prefix = 0 if anchor == "suffix" else 1
    assert min_prefix <= prefix_len < context_len <= prompt_len, (
        prefix_len,
        context_len,
        prompt_len,
    )
    return {
        "full_ids": full_ids,
        "prompt_len": prompt_len,
        "prefix_len": int(prefix_len),
        "context_len": int(context_len),
        "anchor": anchor,
    }


class JacobianEstimator:
    """Teacher-forced forward + seed-batched backwards at the pinned slots.

    The block-``source_layer`` forward output is REPLACED (forward hook) by a
    detached ``requires_grad_(True)`` leaf, so the graph spans only blocks
    source+1..readout (params stay grad-frozen); the readout hook captures the
    block-``readout_layer`` output and raises ``_EarlyExit`` to skip the
    remaining blocks + lm_head entirely.
    """

    def __init__(self, model, *, source_layer: int, readout_layer: int, seed_chunk: int = 32):
        assert source_layer < readout_layer, (source_layer, readout_layer)
        self.base = model.model if hasattr(model, "model") else model
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.device = next(self.base.parameters()).device
        self.k = seed_chunk
        self._h_src: torch.Tensor | None = None
        self._h_ro: torch.Tensor | None = None
        blocks = self.base.layers
        assert 0 <= source_layer < len(blocks) and readout_layer < len(blocks)
        blocks[source_layer].register_forward_hook(self._hook_src)
        blocks[readout_layer].register_forward_hook(self._hook_ro)

    def _hook_src(self, mod, inp, out):
        h = (out[0] if isinstance(out, tuple) else out).detach().requires_grad_(True)
        self._h_src = h
        return (h, *out[1:]) if isinstance(out, tuple) else h

    def _hook_ro(self, mod, inp, out):
        self._h_ro = out[0] if isinstance(out, tuple) else out
        raise _EarlyExit

    def forward_captured(self, full_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One teacher-forced forward; returns (h_src leaf (1,T,H), h_ro (1,T,H))."""
        self._h_src = self._h_ro = None
        try:
            with torch.enable_grad():
                self.base(input_ids=full_ids.unsqueeze(0).to(self.device), use_cache=False)
        except _EarlyExit:
            pass
        assert self._h_src is not None and self._h_ro is not None, "capture hooks did not fire"
        assert self._h_src.requires_grad, "source-slot leaf lost requires_grad"
        return self._h_src, self._h_ro

    def _grad_chunk(self, v: torch.Tensor, h_src: torch.Tensor, chunk: torch.Tensor, serial: bool):
        if serial:
            gs = [
                torch.autograd.grad(v, h_src, grad_outputs=e, retain_graph=True)[0] for e in chunk
            ]
            return torch.stack(gs, dim=0)  # (k, 1, T, H)
        return torch.autograd.grad(
            v, h_src, grad_outputs=chunk, is_grads_batched=True, retain_graph=True
        )[0]

    def pair_backward(
        self, rend: dict, seeds: torch.Tensor, *, serial: bool = False
    ) -> dict[str, torch.Tensor | float]:
        """All-seed rows for one pair: {arm: (S,H) fp32 cpu} + pair summaries.

        Seeds (S, H) are cotangents on v_19; chunked K per vmapped backward
        (``is_grads_batched``), K auto-halved on CUDA OOM (mem_get_info logged).
        """
        pl, pre, ctx = rend["prompt_len"], rend["prefix_len"], rend["context_len"]
        if pre <= 0:
            # A suffix-anchored render legalizes prefix_len == 0 (template-less
            # render); the 3-arm split consumer refuses it LOUDLY (crash-fix
            # r15 contract): a zero-width prefix arm would silently persist
            # all-zero rows. Unreachable under the Qwen chat template.
            raise ValueError(
                f"prefix/context split requested on a zero-prefix render (prefix_len={pre}): "
                "bare-context rows under a template-less render have no prefix arm; the "
                "3-arm jacobian split requires a nonzero chat-template preamble prefix"
            )
        h_src, h_ro = self.forward_captured(rend["full_ids"])
        v = h_ro[0, pl:, :].to(torch.float32).mean(dim=0)  # (H,) — POOLING parity
        out = {arm: [] for arm in ARMS}
        ctx_maxabs = 0.0
        i = 0
        while i < seeds.shape[0]:
            k = min(self.k, seeds.shape[0] - i)
            chunk = seeds[i : i + k].to(self.device, torch.float32)
            try:
                g = self._grad_chunk(v, h_src, chunk, serial)
            except torch.cuda.OutOfMemoryError:
                if self.k <= 1:
                    raise
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                self.k = max(1, self.k // 2)
                print(
                    f"[jacobian] OOM at K={k}; free={free / 2**30:.1f}/{total / 2**30:.1f} GiB"
                    f" -> K={self.k}",
                    flush=True,
                )
                continue
            g = g[:, 0].to(torch.float32)  # (k, T, H)
            out["prefix"].append(g[:, :pre, :].sum(dim=1).cpu())
            out["ctx"].append(g[:, :ctx, :].sum(dim=1).cpu())
            out["last"].append(g[:, pl - 1, :].cpu())
            ctx_maxabs = max(ctx_maxabs, float(g[:, :ctx, :].abs().max()))
            i += k
        res: dict = {arm: torch.cat(out[arm], dim=0) for arm in ARMS}
        with torch.no_grad():
            h = h_src[0].to(torch.float32)
            res["v"] = v.detach().cpu()
            res["c_last"] = h[pl - 1, :].cpu()
            res["c_prefix"] = h[:pre, :].mean(dim=0).cpu()
            res["c_ctx"] = h[:ctx, :].mean(dim=0).cpu()
            res["ctx_maxabs"] = ctx_maxabs
        del h_src, h_ro, v
        return res


# ── accumulation (even/odd half sums, plan §4 half-sum persistence contract) ──


class HalfSumAccumulator:
    """Per-arm per-half running sums over pairs; J = count-weighted half mean."""

    def __init__(self, n_seeds: int, hidden: int):
        self.sums = {a: [torch.zeros(n_seeds, hidden), torch.zeros(n_seeds, hidden)] for a in ARMS}
        self.counts = [torch.zeros(n_seeds, dtype=torch.long) for _ in range(2)]
        self.v_sum = [torch.zeros(hidden), torch.zeros(hidden)]
        self.c_sum = {a: [torch.zeros(hidden), torch.zeros(hidden)] for a in ARMS}
        self.n_pair = [0, 0]

    def add(self, pair_idx: int, seed_idx: np.ndarray, res: dict) -> None:
        half = pair_idx % 2
        idx = torch.as_tensor(seed_idx, dtype=torch.long)
        for a in ARMS:
            self.sums[a][half].index_add_(0, idx, res[a])
        self.counts[half].index_add_(0, idx, torch.ones(len(idx), dtype=torch.long))
        self.v_sum[half] += res["v"]
        for a, key in (("prefix", "c_prefix"), ("ctx", "c_ctx"), ("last", "c_last")):
            self.c_sum[a][half] += res[key]
        self.n_pair[half] += 1

    def state_dict(self) -> dict:
        return {
            "sums": self.sums,
            "counts": self.counts,
            "v_sum": self.v_sum,
            "c_sum": self.c_sum,
            "n_pair": self.n_pair,
        }

    def load_state_dict(self, st: dict) -> None:
        self.sums, self.counts = st["sums"], st["counts"]
        self.v_sum, self.c_sum, self.n_pair = st["v_sum"], st["c_sum"], st["n_pair"]

    def finalize(self, arm: str) -> dict:
        """Merged J + both persisted half sums/counts + intercepts (plan §9 p2_jacobian)."""
        cnt = (self.counts[0] + self.counts[1]).clamp(min=1).to(torch.float32)
        j = (self.sums[arm][0] + self.sums[arm][1]) / cnt[:, None]
        halves = []
        for h in range(2):
            c = self.counts[h].clamp(min=1).to(torch.float32)
            halves.append(self.sums[arm][h] / c[:, None])
        return {
            "J": j,
            "half_sums": [self.sums[arm][0], self.sums[arm][1]],
            "half_counts": [self.counts[0], self.counts[1]],
            "half_J": halves,
            "v_bar_half": [self.v_sum[h] / max(1, self.n_pair[h]) for h in range(2)],
            "c_bar_half": [self.c_sum[arm][h] / max(1, self.n_pair[h]) for h in range(2)],
            "n_pair_half": list(self.n_pair),
        }


def splithalf_report(acc: HalfSumAccumulator) -> dict:
    """Per-seed cross-half row cosine at the realized m (noise-floor read)."""
    rep = {}
    for arm in ARMS:
        fin = acc.finalize(arm)
        a, b = fin["half_J"][0].to(torch.float64), fin["half_J"][1].to(torch.float64)
        num = (a * b).sum(dim=1)
        den = a.norm(dim=1) * b.norm(dim=1)
        cos = (num / den.clamp(min=1e-30)).numpy()
        rep[arm] = {
            "median_row_cos": float(np.median(cos)),
            "q10": float(np.quantile(cos, 0.10)),
            "q90": float(np.quantile(cos, 0.90)),
            "n_rows": int(cos.shape[0]),
        }
    return rep


# ── seed sets ─────────────────────────────────────────────────────────────────


def seeds_for_pair_full(pair_idx: int, shard_seeds: np.ndarray, m: int, n_pairs: int):
    """Full-rank cyclic assignment: seed i serves pairs [(i*m + t) % n_pairs, t<m]."""
    rel = (pair_idx - shard_seeds.astype(np.int64) * m) % n_pairs
    return shard_seeds[rel < m]


def build_sketch_seeds(args) -> Path:
    """256 sketch seeds: top v-pool PCs + M' left-singular u_i + Gaussian checks."""
    vp = torch.load(args.v_pool, map_location="cpu", weights_only=True)
    v = (vp["v"] if isinstance(vp, dict) else vp).to(torch.float32)
    assert v.ndim == 2, v.shape
    comp = torch.load(args.comparator, map_location="cpu", weights_only=True)
    a_op = (comp["W"].to(torch.float64) / comp["xsd"].to(torch.float64)[:, None]).T
    u, s, _ = torch.linalg.svd(a_op, full_matrices=False)
    u_i = u[:, : args.topk_comparator].T.to(torch.float32)  # (k, H) L19 cotangent space
    n_pc = args.n_total - args.topk_comparator - args.n_gaussian
    assert n_pc > 0, (args.n_total, args.topk_comparator, args.n_gaussian)
    vc = v - v.mean(dim=0, keepdim=True)
    _, _, pcs = torch.pca_lowrank(vc, q=min(n_pc + 8, min(vc.shape) - 1))
    pcs = pcs.T[:n_pc]  # (n_pc, H)
    gen = torch.Generator().manual_seed(args.seed)
    gauss = torch.randn(args.n_gaussian, v.shape[1], generator=gen)
    seeds = torch.cat([pcs, u_i, gauss], dim=0)
    seeds = seeds / seeds.norm(dim=1, keepdim=True).clamp(min=1e-12)
    names = (
        [f"vpc{i}" for i in range(n_pc)]
        + [f"mprime_u{i}" for i in range(args.topk_comparator)]
        + [f"gauss{i}" for i in range(args.n_gaussian)]
    )
    out = {"seeds": seeds.to(torch.float32), "names": names, "sigma_head": s[:8].tolist()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[jacobian] [phase=build_seeds_done] {seeds.shape} -> {args.out}", flush=True)
    return args.out


# ── manifest / checkpoint (#952 gate-5: match-or-recompute) ───────────────────

MATCH_KEYS = (
    "code_sha",
    "model",
    "source_layer",
    "readout_layer",
    "dtype",
    "mode",
    "m",
    "n_pairs",
    "shard_index",
    "num_shards",
    "seeds_sha",
    "pairs_sha",
    "pooling",
    "spans",
)


def build_manifest(args, seeds_sha: str, pairs_sha: str, n_pairs: int) -> dict:
    return {
        "code_sha": C76.git_commit(),
        "model": args.model,
        "source_layer": args.source_layer,
        "readout_layer": args.readout_layer,
        "dtype": args.dtype,
        "mode": args.mode,
        "m": args.m,
        "n_pairs": n_pairs,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "seeds_sha": seeds_sha,
        "pairs_sha": pairs_sha,
        "pooling": POOLING,
        "spans": SPAN_CONVENTION,
        "seed_chunk_informational": args.seed_chunk,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def try_resume(out_dir: Path, manifest: dict) -> tuple[dict | None, int]:
    """Match-or-recompute: returns (ckpt_state, n_done); mismatch quarantines."""
    man_path, ckpt_path = out_dir / "manifest.json", out_dir / "ckpt.pt"
    if not (man_path.exists() and ckpt_path.exists()):
        return None, 0
    old = json.loads(man_path.read_text())
    mismatch = [k for k in MATCH_KEYS if old.get(k) != manifest.get(k)]
    if mismatch:
        stale = out_dir / f"stale-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        stale.mkdir(parents=True, exist_ok=True)
        for p in (man_path, ckpt_path):
            p.rename(stale / p.name)
        print(f"[jacobian] resume manifest MISMATCH on {mismatch} -> recompute", flush=True)
        return None, 0
    st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"[jacobian] resume manifest MATCH -> skipping {st['n_done']} done pairs", flush=True)
    return st, int(st["n_done"])


def save_ckpt(out_dir: Path, acc: HalfSumAccumulator, n_done: int, seam_skips: list) -> None:
    tmp = out_dir / "ckpt.pt.tmp"
    torch.save({"acc": acc.state_dict(), "n_done": n_done, "seam_skips": seam_skips}, tmp)
    tmp.replace(out_dir / "ckpt.pt")


# ── G-NONZERO gate (plan §7: HALT rc=8 on an all-zero context-gradient field) ─


def g_nonzero_gate(est: JacobianEstimator, tok, pairs: list[dict], out_dir: Path) -> bool:
    """True = PASS (some context-position gradient is nonzero on the gate pairs)."""
    hidden = est.base.config.hidden_size
    gen = torch.Generator().manual_seed(0)
    seeds = torch.randn(2, hidden, generator=gen)
    seeds = seeds / seeds.norm(dim=1, keepdim=True)
    maxabs = []
    for row in pairs[:2]:
        rend = render_pair(tok, row["prompt"], row["response"])
        assert rend is not None, f"gate pair {row['pair_id']} failed to render"
        res = est.pair_backward(rend, seeds)
        maxabs.append(res["ctx_maxabs"])
    ok = any(m > 0.0 for m in maxabs)
    C76.atomic_write_json(
        out_dir / "gate_gnonzero.json",
        {"gate": "G-NONZERO", "ctx_maxabs": maxabs, "pass": ok, "repro": C76.repro_meta()},
    )
    print(f"[jacobian] [phase=gate_gnonzero] {'PASS' if ok else 'HALT rc=8'} {maxabs}", flush=True)
    return ok


# ── run driver ────────────────────────────────────────────────────────────────


def load_model(args):
    if args.tiny:
        import issue1776_jlens_fit as JF

        _, model, tok = JF.load_lens_model(C.DEFAULT_MODEL, device="cpu", tiny=True)
        return model, tok
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map=args.device
    ).eval()
    return model, tok


def run(args) -> int:
    pairs = load_pairs(args.pairs)
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    pairs_sha = _sha256_file(args.pairs)
    model, tok = load_model(args)
    hidden = (model.model if hasattr(model, "model") else model).config.hidden_size
    if args.mode == "sketch":
        sd = torch.load(args.seeds_file, map_location="cpu", weights_only=True)
        seeds, seed_names = sd["seeds"].to(torch.float32), list(sd["names"])
        seeds_sha = _sha256_tensor(seeds)
    else:  # full-rank standard basis, sharded by contiguous seed block
        seeds, seed_names, seeds_sha = None, None, f"std_basis:H={hidden}"
    n_seed_total = hidden if args.mode == "full" else seeds.shape[0]
    lo = args.shard_index * n_seed_total // args.num_shards
    hi = (args.shard_index + 1) * n_seed_total // args.num_shards
    shard_seeds = np.arange(lo, hi)
    est = JacobianEstimator(
        model,
        source_layer=args.source_layer,
        readout_layer=args.readout_layer,
        seed_chunk=args.seed_chunk,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not g_nonzero_gate(est, tok, pairs, args.out_dir):
        return 8

    manifest = build_manifest(args, seeds_sha, pairs_sha, len(pairs))
    st, n_done = try_resume(args.out_dir, manifest)
    acc = HalfSumAccumulator(len(shard_seeds), hidden)
    seam_skips: list[str] = []
    if st is not None:
        acc.load_state_dict(st["acc"])
        seam_skips = list(st["seam_skips"])
    C76.atomic_write_json(args.out_dir / "manifest.json", manifest)

    if args.mode == "full":
        eye = torch.eye(hidden, dtype=torch.float32)
    t0 = time.time()
    for j, row in enumerate(pairs):
        if j < n_done:
            continue
        if args.mode == "full":
            use = seeds_for_pair_full(j, shard_seeds, args.m, len(pairs))
            if use.size == 0:
                n_done = j + 1
                continue
            seed_mat, local_idx = eye[torch.as_tensor(use)], use - lo
        else:
            # Slice the seed matrix to THIS shard's block: local_idx is shard-sized,
            # so passing the full matrix crashes index_add_ on any multi-shard run
            # (code-review v1 Critical 1) and would redo every seed's backward 8x.
            seed_mat, local_idx = seeds[lo:hi], shard_seeds - lo
        rend = render_pair(tok, row["prompt"], row["response"])
        if rend is None:
            seam_skips.append(str(row["pair_id"]))
            n_done = j + 1
            continue
        res = est.pair_backward(rend, seed_mat, serial=args.serial_grads)
        acc.add(j, np.asarray(local_idx), res)
        n_done = j + 1
        print(
            f"[jacobian] unit {j + 1}/{len(pairs)} pair={row['pair_id']} "
            f"seeds={len(local_idx)} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        if n_done % args.ckpt_every == 0:
            save_ckpt(args.out_dir, acc, n_done, seam_skips)
    save_ckpt(args.out_dir, acc, n_done, seam_skips)

    for arm in ARMS:
        fin = acc.finalize(arm)
        fin.update(
            {
                "seed_names": seed_names,
                "seed_index_range": [int(lo), int(hi)],
                "manifest": manifest,
            }
        )
        torch.save(fin, args.out_dir / f"J_{arm}.pt")
    C76.atomic_write_json(
        args.out_dir / "splithalf.json",
        {"splithalf": splithalf_report(acc), "seam_skips": seam_skips, "repro": C76.repro_meta()},
    )
    print(
        f"[jacobian] [phase=jacobian_done] pairs={n_done} skipped={len(seam_skips)} "
        f"-> {args.out_dir}/J_{{prefix,ctx,last}}.pt",
        flush=True,
    )
    return 0


def merge_shards(args) -> int:
    """Sum per-shard half sums/counts into the full (H, H) J per arm."""
    shard_dirs = sorted(args.shards_root.glob("shard*"))
    assert shard_dirs, f"no shard dirs under {args.shards_root}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        parts = [
            torch.load(d / f"J_{arm}.pt", map_location="cpu", weights_only=True) for d in shard_dirs
        ]
        hidden = parts[0]["half_sums"][0].shape[1]
        # Rows = seed count (max hi over shards): hidden in full mode, n_seeds in
        # sketch mode (a hidden-row sketch merge would wrongly look square and
        # defeat phase3's sketch-refusal check; review v1 Minor).
        n_rows = max(int(p["seed_index_range"][1]) for p in parts)
        sums = [torch.zeros(n_rows, hidden) for _ in range(2)]
        counts = [torch.zeros(n_rows, dtype=torch.long) for _ in range(2)]
        for p in parts:
            lo, hi = p["seed_index_range"]
            for h in range(2):
                sums[h][lo:hi] += p["half_sums"][h]
                counts[h][lo:hi] += p["half_counts"][h]
        tot = counts[0] + counts[1]
        # All-rows-covered assert: a missing shard would silently zero its rows
        # via clamp(min=1) (review v1 Minor — fail loud instead).
        uncovered = (tot == 0).nonzero().flatten()
        assert uncovered.numel() == 0, (
            f"merge_shards[{arm}]: {uncovered.numel()} seed rows have zero count "
            f"(first: {uncovered[:8].tolist()}) — a shard is missing/incomplete"
        )
        cnt = tot.clamp(min=1).to(torch.float32)
        merged = {
            "J": (sums[0] + sums[1]) / cnt[:, None],
            "half_sums": sums,
            "half_counts": counts,
            "v_bar_half": parts[0]["v_bar_half"],
            "c_bar_half": parts[0]["c_bar_half"],
            "n_pair_half": parts[0]["n_pair_half"],
            "manifests": [p["manifest"] for p in parts],
        }
        torch.save(merged, args.out_dir / f"J_{arm}.pt")
        print(f"[jacobian] merged {arm}: {merged['J'].shape} from {len(parts)} shards", flush=True)
    return 0


# ── tiny-real CPU smoke (G-NONZERO + batched-vs-serial + half round-trip + resume) ──


def smoke(args) -> int:
    """Full-body CPU smoke on a from-config tiny Qwen2 over the real tokenizer."""
    args.tiny, args.mode, args.m = True, "sketch", 0
    args.source_layer, args.readout_layer = 1, 3
    # Base legs (1-5) are deterministic single-shard; leg 6 exercises multi-shard
    # sharding + merge explicitly regardless of the CLI --num-shards value.
    args.shard_index, args.num_shards = 0, 1
    model, tok = load_model(args)
    hidden = model.model.config.hidden_size
    prompts = ["What is the capital of France?", "Name one prime number."]
    responses = ["The capital of France is Paris.", "Two is a prime number."]
    args.pairs = args.out_dir / "smoke_pairs.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.pairs.write_text(
        "\n".join(
            json.dumps({"pair_id": f"s{i}", "prompt": p, "response": r})
            for i, (p, r) in enumerate(zip(prompts, responses, strict=True))
        )
        + "\n"
    )
    gen = torch.Generator().manual_seed(1)
    seeds = torch.randn(5, hidden, generator=gen)
    seeds = seeds / seeds.norm(dim=1, keepdim=True)
    args.seeds_file = args.out_dir / "smoke_seeds.pt"
    torch.save({"seeds": seeds, "names": [f"g{i}" for i in range(5)]}, args.seeds_file)

    # 1) batched-vs-serial grad equality on one pair (fp32 CPU: near-exact).
    est = JacobianEstimator(model, source_layer=1, readout_layer=3, seed_chunk=3)
    rend = render_pair(tok, prompts[0], responses[0])
    assert rend is not None
    rb = est.pair_backward(rend, seeds, serial=False)
    rs = est.pair_backward(rend, seeds, serial=True)
    for arm in ARMS:
        diff = float((rb[arm] - rs[arm]).abs().max())
        scale = float(rs[arm].abs().max())
        assert diff <= 1e-6 * max(scale, 1e-12), (arm, diff, scale)
    print(f"[jacobian] [smoke] batched==serial (max rel diff <=1e-6, arms={list(ARMS)})")
    # G-NONZERO must read nonzero on the tiny model (validates the 1<3 slot pair e2e).
    assert rb["ctx_maxabs"] > 0.0, "tiny-model context gradient is zero — slot convention broken"

    # 2) full run driver (gate + accumulate + persist + splithalf).
    rc = run(args)
    assert rc == 0, rc
    fin = torch.load(args.out_dir / "J_last.pt", map_location="cpu", weights_only=True)
    # Half-sum round-trip: merged J == count-weighted mean of the persisted halves.
    cnt = (fin["half_counts"][0] + fin["half_counts"][1]).clamp(min=1).to(torch.float32)
    j_rt = (fin["half_sums"][0] + fin["half_sums"][1]) / cnt[:, None]
    assert torch.allclose(fin["J"], j_rt, atol=0, rtol=0), "half-sum round-trip failed"
    print("[jacobian] [smoke] half-sum round-trip exact")

    # 3) resume MATCH branch: re-run skips all pairs (n_done persists).
    rc = run(args)
    assert rc == 0
    st = torch.load(args.out_dir / "ckpt.pt", map_location="cpu", weights_only=True)
    assert st["n_done"] == 2, st["n_done"]
    print("[jacobian] [smoke] resume manifest MATCH branch: skipped completed pairs")

    # 4) resume MISMATCH branch: perturb a regime key -> quarantine + recompute.
    man = json.loads((args.out_dir / "manifest.json").read_text())
    man["seeds_sha"] = "deadbeef"
    C76.atomic_write_json(args.out_dir / "manifest.json", man)
    rc = run(args)
    assert rc == 0
    assert list(args.out_dir.glob("stale-*")), "mismatch did not quarantine the stale ckpt"
    print("[jacobian] [smoke] resume manifest MISMATCH branch: quarantined + recomputed")

    # 5) degenerate-gate probes. The same-layer (src == ro) convention is refused
    #    at CONSTRUCTION (JacobianEstimator asserts src < ro) — assert that raise
    #    fires; then exercise the G-NONZERO HALT branch itself on a zero-field
    #    stand-in estimator (ctx_maxabs == 0.0), since a real src<ro tiny model
    #    is structurally nonzero (checked in probe 1 above).
    try:
        JacobianEstimator(model, source_layer=3, readout_layer=3, seed_chunk=2)
        raise RuntimeError("same-layer estimator must be refused at construction")
    except AssertionError:
        print("[jacobian] [smoke] same-layer (src==ro) construction refused (assert fired)")
    deg_dir = args.out_dir / "gate_probe"
    deg_dir.mkdir(exist_ok=True)
    fake = [{"pair_id": "s0", "prompt": prompts[0], "response": responses[0]}]

    class _ZeroEst:
        base = model.model

        def pair_backward(self, rend, seeds, **kw):
            return {"ctx_maxabs": 0.0}

    ok = g_nonzero_gate(_ZeroEst(), tok, fake, deg_dir)
    assert not ok, "zero-field probe must FAIL the G-NONZERO gate"
    print("[jacobian] [smoke] G-NONZERO HALT branch fired on the zero-field probe")

    # 6) multi-shard sketch leg (code-review v1 Critical 1): run the SAME sketch
    #    as two shards, merge, and assert merged J == single-shard J per arm.
    shards_root = args.out_dir / "mshard"
    for si in range(2):
        sa = argparse.Namespace(**vars(args))
        sa.shard_index, sa.num_shards = si, 2
        sa.out_dir = shards_root / f"shard{si}"
        sa.out_dir.mkdir(parents=True, exist_ok=True)
        rc = run(sa)
        assert rc == 0, (si, rc)
    ma = argparse.Namespace(shards_root=shards_root, out_dir=shards_root / "merged")
    rc = merge_shards(ma)
    assert rc == 0, rc
    for arm in ARMS:
        single = torch.load(args.out_dir / f"J_{arm}.pt", map_location="cpu", weights_only=True)
        merged = torch.load(
            shards_root / "merged" / f"J_{arm}.pt", map_location="cpu", weights_only=True
        )
        assert merged["J"].shape == single["J"].shape, (arm, merged["J"].shape, single["J"].shape)
        diff = float((merged["J"] - single["J"]).abs().max())
        scale = float(single["J"].abs().max())
        assert diff <= 1e-6 * max(scale, 1e-12), (arm, diff, scale)
    print("[jacobian] [smoke] multi-shard (num_shards=2) merge == single-shard J (all arms)")
    print("[jacobian] [phase=smoke_done] PASS", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _common(p):
        p.add_argument("--model", default=C.DEFAULT_MODEL)
        p.add_argument("--source-layer", type=int, default=C76.SOURCE_LAYER)
        p.add_argument("--readout-layer", type=int, default=C76.READOUT_LAYER)
        p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--seed-chunk", type=int, default=32)
        p.add_argument("--serial-grads", action="store_true")
        p.add_argument("--ckpt-every", type=int, default=16)
        p.add_argument("--tiny", action="store_true", help="from-config tiny Qwen2 (CPU smoke)")

    r = sub.add_parser("run", help="Phase-2 sketch/full estimation")
    _common(r)
    r.add_argument("--mode", choices=["sketch", "full"], required=True)
    r.add_argument("--pairs", type=Path, required=True)
    r.add_argument("--seeds-file", type=Path, help="sketch mode: build-seeds output")
    r.add_argument("--m", type=int, default=150, help="full mode: pairs per seed (cyclic)")
    r.add_argument("--limit-pairs", type=int, default=0)
    r.add_argument("--shard-index", type=int, default=0)
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--out-dir", type=Path, required=True)

    b = sub.add_parser("build-seeds", help="sketch seed set (v-PCs + M' u_i + Gaussian)")
    b.add_argument("--v-pool", type=Path, required=True)
    b.add_argument("--comparator", type=Path, required=True)
    b.add_argument("--n-total", type=int, default=256)
    b.add_argument("--topk-comparator", type=int, default=20)
    b.add_argument("--n-gaussian", type=int, default=8)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", type=Path, required=True)

    m = sub.add_parser("merge-shards", help="sum shard half-sums into full J per arm")
    m.add_argument("--shards-root", type=Path, required=True)
    m.add_argument("--out-dir", type=Path, required=True)

    s = sub.add_parser("smoke", help="tiny-real CPU e2e smoke")
    _common(s)
    s.add_argument("--mode", default="sketch")
    s.add_argument("--pairs", type=Path)
    s.add_argument("--seeds-file", type=Path)
    s.add_argument("--m", type=int, default=0)
    s.add_argument("--limit-pairs", type=int, default=0)
    s.add_argument("--shard-index", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--out-dir", type=Path, required=True)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        if args.mode == "sketch":
            assert args.seeds_file, "--seeds-file required in sketch mode"
        return run(args)
    if args.cmd == "build-seeds":
        build_sketch_seeds(args)
        return 0
    if args.cmd == "merge-shards":
        return merge_shards(args)
    return smoke(args)


if __name__ == "__main__":
    sys.exit(main())
