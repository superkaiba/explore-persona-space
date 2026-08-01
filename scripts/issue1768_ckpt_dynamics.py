"""#1768 checkpoint-dynamics horse race — per-rung teacher-forced write direction.

Round-1 (`issue1768_directions.run_p9`) raced the panel write ŵ at each arm's
SELECTED (verdict) checkpoint against δ / r_B / (marker) the unembedding row.
This round walks the whole Hub checkpoint LADDER of every arm and recomputes the
matched-text write ŵ_tf(step), giving alignment-vs-training-step curves, the
‖ŵ_tf‖(step) dose curve, direction stability cos(ŵ_tf(step), ŵ_tf(verdict)), and
the per-rung install coupling.

Definitions are inherited VERBATIM from round 1 (`issue1768_directions`) so the
curves join its verdict-step reads:

- ŵ_tf(step) = mean over the arm's source-context panel rows of the rung
  checkpoint's teacher-forced response-span activation, MINUS the base panel's
  half-A (even question index) mean — the ŵ leg's disjoint-half baseline.
- δ = t̄_{C,B} (the p5 `delta_tf/<delta_arm>/tbar.pt` mix mean) minus the base
  panel's half-B (odd) mean — the δ leg's disjoint baseline.
- r_B = the fleet behavior direction at that layer; W_U[83399] = the base
  unembedding row (marker arms only).

NO new null draws: round-1's per-(arm, layer, candidate) bands are cos(candidate,
random direction) distributions, independent of which write vector they are
compared against, so they transfer to every rung unchanged.

Phases (`--phase`): stage | pilot | capture | analyze | upload. `capture` shards
by unit index (`--shard i/N`) so one worker runs per provisioned GPU, with
`CUDA_VISIBLE_DEVICES` pinned in the launcher env and the matching `--gpu-id`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:  # script-mode sibling imports (issue1768_cells)
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402

logger = logging.getLogger("issue1768.ckpt_dynamics")

LAYERS: tuple[int, ...] = X.LAYERS  # (14, 19, 25)
SPANS: tuple[str, ...] = ("prefix", "context", "response")
WRITE_SPAN = "response"  # round-1 `_panel_rows` default (the ŵ span)
HF_DYN_PREFIX = f"{X.HF_PREFIX}/ckpt_dynamics_tf"
TF_BATCH = X.TF_BATCH_SIZE  # 8 (round-1 parity)

# Parity gate vs round-1 `direction_reads.json` cos_w_tf at the SELECTED rung.
# Both sides are cos(ŵ_tf, candidate) over the same 20 panel rows and the same
# candidate; they differ only in bf16 batch COMPOSITION (this round forwards the
# 20 source-context rows alone; round 1 forwarded all 120 panel rows), so the
# expected gap is bf16 padded-batch jitter on span means — orders of magnitude
# below a real wiring bug (gotchas.md § bf16 GPU parity gate tolerance).
#
# CALIBRATED on the 2026-07-31 pod smoke (6 selected-rung comparisons, H100
# bf16), which reproduced the documented DEPTH amplification — mean |Δcos| by
# layer: L14 0.00121, L19 0.00413, L25 0.00359, worst single cell 0.00554 (L25
# δ, where both cosines are ~0.01 so a fixed vector-space jitter is a large
# RELATIVE move). 0.025 is ~4.5x that measured worst deviation and still ~20x
# below a real wiring bug's signature (a mis-shifted span or wrong baseline half
# displaces the cosine by 0.1-1.0, not 0.005). Never widen this without
# re-attributing the miss per that gotchas entry.
PARITY_COS_ABS_TOL = 0.025
PARITY_MIN_UNITS = 8  # below this the gate cannot certify; reported, never silent

RB_HUB_PATHS = {  # mirrors issue1768_directions.RB_HUB_PATHS
    "syc": "issue1112_geometry2x2/analysis_tensors/rb/rb_sycophancy.pt",
    "mk": "issue1112_geometry2x2/analysis_tensors/rb/rb_marker.pt",
    "imp": "issue1315_impolite_geometry/analysis_tensors/rb/rb_impolite.pt",
    "cas": "issue1434_writingstyle/analysis_tensors/rb_writing_style.pt",
}


# ── small utils ──────────────────────────────────────────────────────────────


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _meta() -> dict:
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issue": X.ISSUE,
        "git_commit": _git_sha(),
    }


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return ""


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(a @ b / (na * nb))


def _device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _dtype():
    import torch

    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _phase(tag: str) -> None:
    print(f"[phase={tag}]", flush=True)


# ── ladder enumeration (registry-resolved; never hand-built paths) ────────────


def _ladder_prefix(arm: X.Arm) -> tuple[str, str, str]:
    """(repo_id, ladder prefix, selected subfolder) for one arm."""
    if arm.method == "lora":
        sub = X.adapter_subfolder(arm)
        repo = X.HF_MODEL_REPO
    else:
        sub = X.ft_ckpt_subfolder(arm)
        repo = X.FT_OVERFLOW_REPO
    return repo, re.sub(r"/checkpoint-\d+$", "", sub), sub


def enumerate_ladders(out_root: Path, refresh: bool = False) -> dict:
    """Per-arm Hub checkpoint ladder + coverage report (cached under inputs/).

    A rung the Hub does not carry is simply absent from the ladder (recorded,
    never a run-kill); an arm whose SELECTED rung is missing is flagged.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    path = out_root / "inputs" / "ladders.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    api = HfApi()
    arms = X.all_arms()
    ladders: dict[str, dict] = {}
    listed: dict[str, list[int]] = {}
    for arm in arms:
        repo, prefix, sel_sub = _ladder_prefix(arm)
        key = f"{repo}|{prefix}"
        if key not in listed:

            def _list_ladder(r: str = repo, p: str = prefix) -> list[str]:
                """Prefix-scoped ladder listing, materialized inside the caller's
                retry envelope (a lazy generator would raise outside it, #1402)."""
                # HUB_VERIFY_RETRY_EXEMPT: scoped non-recursive model-repo listing, retried by the retry_transient envelope that wraps this callable
                tree = api.list_repo_tree(r, path_in_repo=p, repo_type="model", recursive=False)
                return [e.path for e in tree]

            entries = hub.retry_transient(_list_ladder, what=f"ladder listing {key}")
            listed[key] = sorted(
                {int(m.group(1)) for f in entries if (m := re.search(r"/checkpoint-(\d+)$", f))}
            )
        steps = listed[key]
        ladders[arm.arm_id] = {
            "arm_id": arm.arm_id,
            "method": arm.method,
            "kind": arm.kind,
            "beh_key": arm.beh_key,
            "ctx_key": arm.ctx_key,
            "regime": arm.regime,
            "seed": arm.seed,
            "lr": arm.lr,
            "selected_step": arm.step,
            "selection_read": arm.selection_read,
            "repo": repo,
            "prefix": prefix,
            "selected_subfolder": sel_sub,
            "steps": steps,
            "n_steps": len(steps),
            "selected_present": arm.step in steps,
        }
    _atomic_json(path, {"ladders": ladders, **_meta()})
    return json.loads(path.read_text())


def rung_priority(steps: list[int], selected: int) -> list[int]:
    """Rungs ordered so ANY PREFIX spans the ladder (farthest-point traversal).

    The selected (verdict) rung first — it anchors the parity gate and the
    cos-vs-verdict reference — then the endpoints, then repeatedly whichever
    remaining rung is farthest from everything already chosen. A prefix of
    length k is therefore a near-uniform k-point subsample of the ladder, so a
    run stopped early yields COARSER curves for every arm rather than complete
    curves for the alphabetically-early arms and nothing for the rest.
    """
    remaining = sorted(set(steps))
    out: list[int] = []
    if selected in remaining:
        out.append(selected)
        remaining.remove(selected)
    for end in [remaining[0], remaining[-1]] if remaining else []:
        if end in remaining:
            out.append(end)
            remaining.remove(end)
    while remaining:
        far = max(remaining, key=lambda s: (min(abs(s - o) for o in out), s))
        out.append(far)
        remaining.remove(far)
    return out


def capture_units(
    ladders: dict, arms_filter=(), limit: int = 0, max_per_arm: int = 0
) -> list[dict]:
    """The (arm, rung) units this round CAPTURES on GPU: LoRA rungs only.

    full-FT arms are EXCLUDED by design: the Hub carries only their selected
    checkpoint (no ladder), so their verdict-rung read is reused verbatim from
    round 1 at zero GPU cost — see `analyze`'s `ft_verdict_only` rows and the
    coverage report's `ft_skipped_by_design`.

    Units are ordered by LADDER-SPREAD RANK, then arm — `rung_priority` pass 0
    (every arm's verdict rung) before pass 1, and so on — so a run stopped early
    degrades to coarser curves across ALL arms instead of full curves for the
    alphabetically-early arms and nothing for the rest. Index-modulo sharding
    over this order keeps each shard's own prefix spread the same way.
    """
    wanted_arms = set(arms_filter or ())
    units: list[dict] = []
    for arm_id, lad in sorted(ladders.items()):
        if wanted_arms and arm_id not in wanted_arms:
            continue
        if lad["method"] != "lora":
            continue
        ordered = rung_priority(list(lad["steps"]), lad["selected_step"])
        if max_per_arm:
            # smoke / descope sizing: the first k spread-ranked rungs, which
            # always include the SELECTED rung (so the round-1 parity gate and
            # the cos-vs-verdict path both fire).
            # NB: a distinct name from `wanted_arms` — reusing that name here
            # silently skipped every arm after the first (caught by the two-arm
            # smoke; pinned by test_max_per_arm_keeps_selected_rung_*).
            ordered = ordered[:max_per_arm]
        for rank, step in enumerate(ordered):
            units.append(
                {
                    "arm_id": arm_id,
                    "step": step,
                    "spread_rank": rank,
                    "repo": lad["repo"],
                    "subfolder": f"{lad['prefix']}/checkpoint-{step}",
                    "beh_key": lad["beh_key"],
                    "kind": lad["kind"],
                    "selected_step": lad["selected_step"],
                }
            )
    units.sort(key=lambda u: (u["spread_rank"], u["arm_id"]))
    return units[:limit] if limit else units


# ── input staging (parent stages ONCE, before any fan-out) ────────────────────


def _base_behaviors(ladders: dict) -> list[str]:
    return sorted({lad["beh_key"] for lad in ladders.values()})


def _ctx_order_from_rows(rows: list[dict]) -> list[str]:
    """First-appearance context order — mirrors directions._panel_ctx_order."""
    seen: list[str] = []
    for r in rows:
        if r["persona"] not in seen:
            seen.append(r["persona"])
    return seen


def stage_inputs(out_root: Path) -> dict:
    """Stage every shared read-only input ONCE (fan-out shared-staging rule).

    Base panel stores (pooled + raw rows), the p5 δ means, the fleet r_B stacks
    and the base W_U marker row. Writes `inputs/src_ctx.json` — the arm →
    source-context map, asserted against the POOLED store's context order so a
    raw-rows/pooled ordering drift fails loud here rather than silently
    mis-selecting a context 1,200 units later.
    """
    _phase("dyn_stage")
    import torch

    from explore_persona_space.orchestrate import hub

    ladders = enumerate_ladders(out_root)["ladders"]
    inputs = out_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    behs = _base_behaviors(ladders)
    for beh in behs:
        for name in ("pooled.pt", "raw_rows.json"):
            dest = inputs / "panel_capture" / f"base_{beh}" / name
            if not dest.exists():
                hub.stage_hub_file(
                    X.HF_DATA_REPO,
                    f"{X.HF_PREFIX}/panel_capture/base_{beh}/{name}",
                    dest,
                    repo_type="dataset",
                )
        print(f"[stage] base_{beh} panel staged", flush=True)

    # arm -> source context id, cross-checked raw_rows order vs pooled order
    src_ctx: dict[str, str] = {}
    for beh in behs:
        raw = json.loads((inputs / "panel_capture" / f"base_{beh}" / "raw_rows.json").read_text())
        order_raw = _ctx_order_from_rows(raw["rows"])
        pooled = torch.load(
            inputs / "panel_capture" / f"base_{beh}" / "pooled.pt",
            map_location="cpu",
            weights_only=False,
        )
        order_pooled: list[str] = []
        for m in pooled["row_meta"]:
            if m["context_id"] not in order_pooled:
                order_pooled.append(m["context_id"])
        assert order_raw == order_pooled, (beh, order_raw, order_pooled)
        del pooled
        for arm_id, lad in ladders.items():
            if lad["beh_key"] != beh:
                continue
            pos = d_src_pos(lad["ctx_key"])
            assert pos < len(order_raw), (arm_id, order_raw)
            src_ctx[arm_id] = order_raw[pos]
    _atomic_json(inputs / "src_ctx.json", {"src_ctx": src_ctx, **_meta()})

    # δ means (one per distinct delta_arm) + r_B stacks + W_U row
    arm_by_id = {a.arm_id: a for a in X.all_arms()}
    delta_arms = sorted({X.delta_arm_for(arm_by_id[a]) for a in ladders})
    for da in delta_arms:
        dest = inputs / "delta_tf" / da / "tbar.pt"
        if not dest.exists():
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/delta_tf/{da}/tbar.pt",
                dest,
                repo_type="dataset",
            )
    print(f"[stage] {len(delta_arms)} delta tbar staged", flush=True)

    for beh, hub_path in RB_HUB_PATHS.items():
        dest = inputs / "rb" / Path(hub_path).name
        if not dest.exists():
            hub.stage_hub_file(X.HF_DATA_REPO, hub_path, dest, repo_type="dataset")
    print("[stage] r_B stacks staged", flush=True)

    wu_path = inputs / "wu_marker_row.npy"
    if not wu_path.exists():
        import issue1768_directions as d

        row = d.load_wu_row(X.BASE_MODEL)
        np.save(wu_path, row)
    print("[stage] W_U marker row staged", flush=True)

    report = {
        "behaviors": behs,
        "n_arms": len(ladders),
        "n_delta_arms": len(delta_arms),
        "n_lora_units": len(capture_units(ladders)),
        **_meta(),
    }
    _atomic_json(out_root / "stage_done.json", report)
    return report


def d_src_pos(ctx_key: str) -> int:
    """Panel position of a context key — mirrors directions.SRC_CTX_POS."""
    import issue1768_directions as d

    return d.SRC_CTX_POS[ctx_key]


# ── teacher-forced span means on a PERSISTENT model (round-1 parity) ──────────


def _tf_span_means_on_model(
    model,
    tokenizer,
    rows: list[dict],
    layers: tuple[int, ...],
    device: str,
    tf_batch_size: int,
) -> dict[str, dict[int, np.ndarray]]:
    """Mirror of `representation_shift._teacher_forced_span_means`'s inner loop,
    against an ALREADY-LOADED model (the adapter-swap path).

    Same right-padding (positions index naturally from 0, so no explicit
    position_ids — the left-pad trap does not apply), same per-layer forward
    hooks, same float32 span mean, same `logits_to_keep=1` unread-logits skip.
    Returns ``{span: {layer: (n_rows, hidden) float64}}`` in ROW order.
    """
    import inspect

    import torch

    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    for i, r in enumerate(rows):
        p_len = len(r["prompt_token_ids"])
        assert 0 < r["prefix_len"] < r["context_len"] <= p_len, (
            i,
            r["prefix_len"],
            r["context_len"],
            p_len,
        )
        assert len(r["response_token_ids"]) > 0, f"row {i} has an empty response span"

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    blocks = base.model.layers
    for li in layers:
        assert 0 <= li < len(blocks), (li, len(blocks))
    hooks = [blocks[li].register_forward_hook(make_hook(li)) for li in layers]

    hidden = base.config.hidden_size
    pooled: dict[str, dict[int, list[np.ndarray]]] = {
        span: {li: [] for li in layers} for span in SPANS
    }
    # unread-logits skip (gotchas.md #779): introspect the UNWRAPPED forward —
    # a PeftModel's own forward is (*args, **kwargs) and hides the parameter.
    fwd = getattr(base, "forward", base.__call__)
    use_ltk = "logits_to_keep" in inspect.signature(fwd).parameters
    try:
        for start in range(0, len(rows), tf_batch_size):
            batch = rows[start : start + tf_batch_size]
            seqs = [r["prompt_token_ids"] + r["response_token_ids"] for r in batch]
            max_len = max(len(s) for s in seqs)
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                attn[i, : len(s)] = 1
            input_ids, attn = input_ids.to(device), attn.to(device)
            with torch.no_grad():
                kwargs = {"logits_to_keep": 1} if use_ltk else {}
                try:
                    _ = model(input_ids=input_ids, attention_mask=attn, **kwargs)
                except TypeError:
                    if not kwargs:
                        raise
                    use_ltk = False
                    _ = model(input_ids=input_ids, attention_mask=attn)
            for li in layers:
                hs = captured[li]
                assert hs.shape[:2] == (len(batch), max_len), (hs.shape, len(batch), max_len)
                for i, r in enumerate(batch):
                    p_len = len(r["prompt_token_ids"])
                    bounds = {
                        "prefix": (0, r["prefix_len"]),
                        "context": (0, r["context_len"]),
                        "response": (p_len, p_len + len(r["response_token_ids"])),
                    }
                    for span in SPANS:
                        s, e = bounds[span]
                        vec = hs[i, s:e, :].float().mean(dim=0)
                        assert vec.shape == (hidden,), (vec.shape, hidden)
                        pooled[span][li].append(vec.double().cpu().numpy())
            captured.clear()
    finally:
        for h in hooks:
            h.remove()
        captured.clear()
    return {span: {li: np.stack(pooled[span][li]) for li in layers} for span in SPANS}


MIN_FREE_GIB = 30.0  # 7B bf16 weights (~15 GiB) + activations, with headroom


def _assert_gpu_free(device: str) -> None:
    """Fail loud on a GPU another tenant (or an orphaned worker) still holds.

    Reads the DEVICE-level free bytes, not the per-process compute-apps table:
    a holder this container's NVML view cannot resolve produces no process row
    at all while `mem_get_info` still reads the truth (gotchas.md #825).
    """
    import torch

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return
    free, total = torch.cuda.mem_get_info(torch.device(device))
    free_gib, total_gib = free / 2**30, total / 2**30
    print(f"[preflight] {device} free={free_gib:.1f} GiB / {total_gib:.1f} GiB", flush=True)
    assert free_gib >= MIN_FREE_GIB, (
        f"{device} has only {free_gib:.1f} GiB free (need {MIN_FREE_GIB}) — a foreign "
        "or orphaned holder is on this GPU; refusing to launch"
    )


class AdapterRunner:
    """Persistent base model + LoRA adapter hot-swap (one 7B load per worker)."""

    def __init__(self, base_model: str, device: str, dtype, model=None, tokenizer=None):
        """``model``/``tokenizer`` inject an ALREADY-BUILT pair — used by the
        tiny-real CPU smoke (a from-config 2-layer same-arch model stands in for
        the 7B weights); production passes neither and loads from the Hub."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        self.device = device
        if model is None:
            _assert_gpu_free(device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        self.model = model or AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
        self.model.eval()
        self._peft = None
        self._live: str | None = None
        print(f"[runner] base loaded in {time.time() - t0:.0f}s on {device}", flush=True)

    def _assert_gauge_free(self, adapter_dir: Path) -> dict:
        """LoRA must not touch the unembedding — the W_U race's gauge premise."""
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        tm = cfg.get("target_modules") or []
        bad = [m for m in tm if "lm_head" in str(m) or "embed_tokens" in str(m)]
        assert not bad, f"adapter targets the unembedding ({bad}) — W_U gauge broken"
        assert not cfg.get("modules_to_save"), cfg.get("modules_to_save")
        return cfg

    def apply_adapter(self, adapter_dir: Path) -> dict:
        """Load + activate one adapter, dropping the previous one."""
        from peft import PeftModel

        cfg = self._assert_gauge_free(adapter_dir)
        name = f"a{int(time.time() * 1e6) % 10_000_000}"
        if self._peft is None:
            self._peft = PeftModel.from_pretrained(
                self.model, str(adapter_dir), adapter_name=name, is_trainable=False
            )
            self._peft.eval()
        else:
            self._peft.load_adapter(str(adapter_dir), adapter_name=name, is_trainable=False)
        self._peft.set_adapter(name)
        if self._live is not None and self._live != name:
            self._peft.delete_adapter(self._live)
        self._live = name
        return cfg

    def span_means(
        self, rows: list[dict], layers: tuple[int, ...] = LAYERS
    ) -> dict[str, dict[int, np.ndarray]]:
        assert self._peft is not None, "apply_adapter() first"
        return _tf_span_means_on_model(
            self._peft, self.tokenizer, rows, layers, self.device, TF_BATCH
        )


# ── capture phase ────────────────────────────────────────────────────────────


def _unit_path(out_root: Path, arm_id: str, step: int) -> Path:
    return out_root / "ckpt_dynamics_tf" / arm_id / f"step-{step}.pt"


def _download_adapter(unit: dict, scratch_root: Path) -> Path:
    """Fetch ONLY the two adapter files into a per-unit dir inside the out-root
    (same filesystem as the destination — the #1335 EXDEV rule).

    `local_dir=` is load-bearing: without it every rung's 40 MB blob ALSO lands
    in the shared HF hub cache, so deleting the scratch dir frees nothing and
    ~50 GB accumulates across 1,236 rungs (the #1092 delete-to-free lesson).
    """
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    scratch_root.mkdir(parents=True, exist_ok=True)
    dest = Path(tempfile.mkdtemp(dir=scratch_root, prefix="adapter_"))
    for fname in ("adapter_config.json", "adapter_model.safetensors"):
        got = hub.retry_transient(
            lambda f=fname: hf_hub_download(
                unit["repo"],
                f"{unit['subfolder']}/{f}",
                token=os.environ.get("HF_TOKEN"),
                local_dir=str(dest),
            ),
            what=f"adapter fetch {unit['arm_id']}@{unit['step']} {fname}",
        )
        # local_dir mirrors the repo-relative path; flatten so PEFT finds them
        got_path = Path(got)
        if got_path != dest / fname:
            got_path.replace(dest / fname)
    return dest


def _rows_for(inputs: Path, beh_key: str, src_ctx: str) -> list[dict]:
    raw = json.loads((inputs / "panel_capture" / f"base_{beh_key}" / "raw_rows.json").read_text())
    rows = [r for r in raw["rows"] if r["persona"] == src_ctx]
    assert rows, (beh_key, src_ctx)
    return sorted(rows, key=lambda r: r["question_idx"])


def phase_capture(
    out_root: Path,
    shard: tuple[int, int],
    gpu_id: int,
    arms_filter=(),
    limit: int = 0,
    smoke: bool = False,
    max_per_arm: int = 0,
) -> None:
    """Walk this shard's (arm, rung) units on ONE GPU; persist per unit."""
    _phase("dyn_capture")
    import torch

    ladders = enumerate_ladders(out_root)["ladders"]
    inputs = out_root / "inputs"
    src_ctx_map = json.loads((inputs / "src_ctx.json").read_text())["src_ctx"]
    units = capture_units(ladders, arms_filter, limit, max_per_arm)
    i, n = shard
    mine = [u for k, u in enumerate(units) if k % n == i]
    print(
        f"[capture] shard {i}/{n} gpu={gpu_id} units={len(mine)}/{len(units)} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} smoke={smoke}",
        flush=True,
    )
    # PER-INVOCATION scratch, never a shard-index-keyed one: two concurrent legs
    # over DIFFERENT arm sets reuse the same shard indices (`0/2` and `1/2` each),
    # so a `shard{i}`-keyed dir is SHARED across legs — and this function rmtree's
    # its scratch on exit, which then deletes a still-live sibling's freshly
    # downloaded adapter. PEFT sees no local safetensors, falls back to treating
    # the path as a Hub repo id, and dies with HFValidationError. That is exactly
    # the fan-out shared-staging race in gotchas.md, and it cost 59 content units
    # of this round when the marker leg finished first.
    scratch_root = out_root / "adapter_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(dir=scratch_root, prefix=f"shard{i}of{n}-"))
    runner: AdapterRunner | None = None
    rows_cache: dict[tuple[str, str], list[dict]] = {}
    t_phase = time.time()
    done = 0
    for k, u in enumerate(mine, start=1):
        dest = _unit_path(out_root, u["arm_id"], u["step"])
        if dest.exists():
            try:
                prev = torch.load(dest, map_location="cpu", weights_only=False)
                if prev.get("smoke") == smoke:
                    done += 1
                    print(
                        f"[capture] unit {k}/{len(mine)} {u['arm_id']}@{u['step']} resumed",
                        flush=True,
                    )
                    continue
            except Exception as exc:  # noqa: BLE001 — corrupt partial: recapture
                logger.warning("[capture] re-doing unreadable %s (%s)", dest, exc)
        t0 = time.time()
        if runner is None:
            runner = AdapterRunner(X.BASE_MODEL, _device(), _dtype())
        key = (u["beh_key"], src_ctx_map[u["arm_id"]])
        if key not in rows_cache:
            rows_cache[key] = _rows_for(inputs, *key)
        rows = rows_cache[key]
        adapter_dir = _download_adapter(u, scratch)
        try:
            cfg = runner.apply_adapter(adapter_dir)
            pooled = runner.span_means(rows)
        finally:
            shutil.rmtree(adapter_dir, ignore_errors=True)
        store = {
            "schema_version": 1,
            "arm_id": u["arm_id"],
            "step": u["step"],
            "src_ctx": key[1],
            "beh_key": u["beh_key"],
            "kind": u["kind"],
            "selected_step": u["selected_step"],
            "subfolder": u["subfolder"],
            "repo": u["repo"],
            "question_idx": [int(r["question_idx"]) for r in rows],
            "layers": list(LAYERS),
            "smoke": smoke,
            "adapter_recipe": {
                "r": cfg.get("r"),
                "lora_alpha": cfg.get("lora_alpha"),
                "use_rslora": cfg.get("use_rslora"),
                "target_modules": sorted(cfg.get("target_modules") or []),
            },
            "arms": {
                span: {li: torch.from_numpy(pooled[span][li]).to(torch.float16) for li in LAYERS}
                for span in SPANS
            },
            **_meta(),
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".pt.tmp")
        torch.save(store, tmp)
        os.replace(tmp, dest)
        done += 1
        print(
            f"[capture] unit {k}/{len(mine)} {u['arm_id']}@{u['step']} "
            f"n_rows={len(rows)} elapsed={time.time() - t0:.1f}s "
            f"(shard {time.time() - t_phase:.0f}s)",
            flush=True,
        )
    shutil.rmtree(scratch, ignore_errors=True)
    _atomic_json(
        out_root / f"capture_done_shard{i}.json",
        {
            "shard": i,
            "n_shards": n,
            "n_units": len(mine),
            "n_done": done,
            "smoke": smoke,
            **_meta(),
        },
    )
    print(
        f"[capture] shard {i} done={done}/{len(mine)} wall={time.time() - t_phase:.0f}s", flush=True
    )


def phase_pilot(out_root: Path, gpu_id: int) -> None:
    """MEASURED 1-unit pilot at production shape (compute-sizing basis)."""
    _phase("dyn_pilot")
    ladders = enumerate_ladders(out_root)["ladders"]
    units = capture_units(ladders)
    assert units, "no LoRA units enumerated"
    u = units[0]
    t0 = time.time()
    phase_capture(out_root, (0, 1), gpu_id, arms_filter=(u["arm_id"],), limit=1)
    wall = time.time() - t0
    rep = {
        "pilot_unit": f"{u['arm_id']}@{u['step']}",
        "wall_s_including_base_load": round(wall, 1),
        "n_units_total": len(units),
        **_meta(),
    }
    _atomic_json(out_root / "pilot_report.json", rep)
    print(f"[pilot] {json.dumps(rep)}", flush=True)


# ── analysis ─────────────────────────────────────────────────────────────────


def _load_pt(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _panel_rows_from_pooled(
    store: dict, ctx_id: str, layer: int, span: str
) -> dict[int, np.ndarray]:
    """{question_idx: vector} — mirrors directions._panel_rows."""
    mat = np.asarray(store["arms"][span][layer].float().numpy(), dtype=np.float64)
    return {
        m["question_idx"]: mat[i]
        for i, m in enumerate(store["row_meta"])
        if m["context_id"] == ctx_id
    }


def _half_means(rows: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(all, even-qidx half A, odd-qidx half B) — mirrors directions._half_means."""
    qs = sorted(rows)
    even = [rows[q] for q in qs if q % 2 == 0]
    odd = [rows[q] for q in qs if q % 2 == 1]
    assert even and odd, f"need >=1 question per half, got {len(qs)}"
    return (
        np.mean([rows[q] for q in qs], axis=0),
        np.mean(even, axis=0),
        np.mean(odd, axis=0),
    )


def _round1_reads() -> dict:
    p = REPO_ROOT / "eval_results/issue_1768/direction_reads.json"
    return json.loads(p.read_text())["reads"]


def _verdict_manifest() -> dict:
    return json.loads(X.VERDICT_MANIFEST.read_text())


def _install_by_step(man: dict, lad: dict) -> dict[int, dict]:
    """Per-rung install trajectory, where #1481 recorded one.

    BOTH families carry a full per-step ladder, under DIFFERENT manifest
    shapes — read each from its own path:

    - marker: ``man["marker"]["arms"][arm_id]["reads_by_step"]`` — 40 steps of
      delta_logp_mean / delta_margin_mean / emission rates.
    - content: ``man["content"][beh][ctx]["arms"][arm_id]["rates_by_step"]`` —
      15 steps of the Tier-1 selection-pool judged rate (the selection
      instrument). NB the `arms` key sits BESIDE `seeds`; `issue1768_cells`
      enumerates arms through `seeds`, so a reader that follows only the
      enumeration path sees just the selected step and wrongly concludes no
      per-step content rates exist. They do, for all 144 content entries, and
      their steps coincide exactly with the Hub checkpoint ladder.
    """
    if lad["kind"] == "marker":
        entry = man["marker"]["arms"].get(lad["arm_id"]) or {}
        out = {}
        for s, v in (entry.get("reads_by_step") or {}).items():
            out[int(s)] = {
                "install_metric": "delta_logp_mean",
                "install": v.get("delta_logp_mean"),
                "delta_margin_mean": v.get("delta_margin_mean"),
                "source_emission_rate": v.get("source_emission_rate"),
                "gen_emission_rate": v.get("gen_emission_rate"),
            }
        return out
    entry = ((man["content"].get(lad["beh_key"]) or {}).get(lad["ctx_key"]) or {}).get("arms") or {}
    rates = (entry.get(lad["arm_id"]) or {}).get("rates_by_step") or {}
    if rates:
        return {
            int(s): {"install_metric": "judged_rate_tier1_selection_pool", "install": v}
            for s, v in rates.items()
        }
    # no per-step rates for this arm: fall back to its selected-step read, and
    # SAY the read is a single point rather than implying a curve
    return {
        int(lad["selected_step"]): {
            "install_metric": "judged_rate_selected_step_only",
            "install": lad["selection_read"],
        }
    }


def phase_analyze(out_root: Path, results_dir: Path, smoke: bool = False) -> None:
    """Race every captured rung; write per-arm curves + the summary."""
    _phase("dyn_analyze")
    out_root, results_dir = Path(out_root), Path(results_dir)
    inputs = out_root / "inputs"
    ladders = enumerate_ladders(out_root)["ladders"]
    src_ctx_map = json.loads((inputs / "src_ctx.json").read_text())["src_ctx"]
    r1 = _round1_reads()
    man = _verdict_manifest()
    arm_by_id = {a.arm_id: a for a in X.all_arms()}

    rb: dict[str, np.ndarray] = {}
    for beh, hub_path in RB_HUB_PATHS.items():
        obj = _load_pt(inputs / "rb" / Path(hub_path).name)
        arr = obj["rb"] if "rb" in obj else obj["r_b"]
        rb[beh] = np.asarray(arr, dtype=np.float64)
    wu = np.load(inputs / "wu_marker_row.npy")

    base_cache: dict[str, dict] = {}
    curves: dict[str, dict] = {}
    parity: list[dict] = []
    coverage: dict[str, dict] = {}

    for arm_id, lad in sorted(ladders.items()):
        arm = arm_by_id[arm_id]
        beh, src = lad["beh_key"], src_ctx_map[arm_id]
        hub_steps = list(lad["steps"])
        if lad["method"] == "ft":
            coverage[arm_id] = {
                "method": "ft",
                "n_hub_rungs": len(hub_steps),
                "n_captured": 0,
                "frac": 0.0,
                "disposition": "ft_skipped_by_design — Hub carries no ladder; "
                "verdict-rung read reused from round 1 (0 GPU)",
                "hub_steps": hub_steps,
            }
        present = sorted(
            s
            for s in hub_steps
            if lad["method"] == "lora" and _unit_path(out_root, arm_id, s).exists()
        )
        if lad["method"] == "lora":
            coverage[arm_id] = {
                "method": "lora",
                "n_hub_rungs": len(hub_steps),
                "n_captured": len(present),
                "frac": (len(present) / len(hub_steps)) if hub_steps else 0.0,
                "missing_steps": [s for s in hub_steps if s not in present],
                "hub_steps": hub_steps,
            }
        if beh not in base_cache:
            base_cache[beh] = _load_pt(inputs / "panel_capture" / f"base_{beh}" / "pooled.pt")
        base_store = base_cache[beh]
        tbar = _load_pt(inputs / "delta_tf" / X.delta_arm_for(arm) / "tbar.pt")
        install = _install_by_step(man, lad)

        for layer in LAYERS:
            key = f"{arm_id}_L{layer}"
            r1_read = r1.get(key, {})
            v0 = _panel_rows_from_pooled(base_store, src, layer, WRITE_SPAN)
            _v0_all, v0_A, v0_B = _half_means(v0)
            tb = np.asarray(tbar["tbar"][layer].float().numpy(), dtype=np.float64)
            cands: dict[str, np.ndarray] = {"delta": tb - v0_B}
            if layer < rb[beh].shape[0]:
                cands["r_B"] = rb[beh][layer]
            if lad["kind"] == "marker":
                cands["W_U_marker_row"] = wu
            # Candidate degeneracy check — the marker "three-way" race is a
            # TWO-way race by construction: rb_marker.pt is the unembedding row
            # tiled per layer (its own note: "W_U[83399] tiled per layer (#653
            # convention)"), so cos(ŵ, r_B) and cos(ŵ, W_U) are the SAME read
            # for marker arms. Round 1 published both columns too (all 60 of its
            # marker (arm,layer) reads agree exactly). Recorded per curve so the
            # duplication can never be mistaken for two agreeing candidates.
            degenerate = sorted(
                {a, b}
                for a, b in itertools.combinations(sorted(cands), 2)
                if cands[a].shape == cands[b].shape and np.allclose(cands[a], cands[b])
            )

            # ŵ_tf at the arm's VERDICT rung — the stability reference
            w_ref: np.ndarray | None = None
            sel = int(lad["selected_step"])
            if lad["method"] == "lora" and sel in present:
                sref = _load_pt(_unit_path(out_root, arm_id, sel))
                ref_all, _ref_A, _ref_B = _half_means(
                    dict(
                        zip(
                            sref["question_idx"],
                            np.asarray(
                                sref["arms"][WRITE_SPAN][layer].float().numpy(), dtype=np.float64
                            ),
                            strict=True,
                        )
                    )
                )
                w_ref = ref_all - v0_A

            pts = []
            for step in present:
                st = _load_pt(_unit_path(out_root, arm_id, step))
                vtf = dict(
                    zip(
                        st["question_idx"],
                        np.asarray(st["arms"][WRITE_SPAN][layer].float().numpy(), dtype=np.float64),
                        strict=True,
                    )
                )
                vtf_all, vtf_A, vtf_B = _half_means(vtf)
                w = vtf_all - v0_A
                row = {
                    "step": step,
                    "is_selected": step == sel,
                    "w_tf_norm": float(np.linalg.norm(w)),
                    "w_tf_split_half_cos": _cos(vtf_A - v0_A, vtf_B - v0_B),
                    "cos_vs_verdict": _cos(w, w_ref) if w_ref is not None else None,
                    "cos": {cn: _cos(w, cv) for cn, cv in cands.items()},
                    "install": install.get(step),
                }
                if step == sel and "delta" in cands:
                    for cn in cands:
                        got = row["cos"][cn]
                        want = (r1_read.get("races", {}).get(cn, {}) or {}).get("cos_w_tf")
                        if want is None or got is None or not np.isfinite(got):
                            continue
                        parity.append(
                            {
                                "key": key,
                                "candidate": cn,
                                "round1_cos_w_tf": want,
                                "this_round_cos": got,
                                "abs_diff": abs(got - want),
                            }
                        )
                pts.append(row)

            nulls = {
                cn: {
                    "primary_null_family": (r1_read.get("races", {}).get(cn, {}) or {}).get(
                        "primary_null_family"
                    ),
                    "nulls": (r1_read.get("races", {}).get(cn, {}) or {}).get("nulls"),
                    "shuffled_row_null": (r1_read.get("races", {}).get(cn, {}) or {}).get(
                        "shuffled_row_null"
                    ),
                    "source": "round-1 direction_reads.json (cosine bands are "
                    "candidate-keyed and write-independent — reused, no new draws)",
                }
                for cn in cands
            }
            curves[key] = {
                "arm_id": arm_id,
                "layer": layer,
                "method": lad["method"],
                "kind": lad["kind"],
                "beh_key": beh,
                "ctx_key": lad["ctx_key"],
                "regime": lad["regime"],
                "seed": lad["seed"],
                "lr": lad["lr"],
                "src_ctx": src,
                "selected_step": sel,
                "epochs_equiv_note": "step = optimizer step (the #1481 ladder cadence)",
                "points": pts,
                "null_bands": nulls,
                "round1_verdict": {
                    "cos_w_on_policy": {
                        cn: (r1_read.get("races", {}).get(cn, {}) or {}).get("cos_w")
                        for cn in cands
                    },
                    "cos_w_tf": {
                        cn: (r1_read.get("races", {}).get(cn, {}) or {}).get("cos_w_tf")
                        for cn in cands
                    },
                    "w_norm_on_policy": r1_read.get("w_norm"),
                },
                "candidate_degeneracy": [list(d) for d in degenerate],
                "install_coverage": (
                    "per-rung (#1481 marker reads_by_step: delta_logp_mean)"
                    if lad["kind"] == "marker"
                    else "per-rung (#1481 content rates_by_step: Tier-1 selection-pool judged rate)"
                ),
                "install_n_steps": len(install),
            }

    summary = _summarize(curves, coverage, parity, smoke)
    _atomic_json(results_dir / "summary.json", summary)
    _atomic_json(results_dir / "curves.json", {"curves": curves, "smoke": smoke, **_meta()})
    _atomic_json(results_dir / "coverage.json", {"coverage": coverage, "smoke": smoke, **_meta()})
    print(f"[analyze] {json.dumps(summary['headline'], indent=1)[:1200]}", flush=True)


def _trend(steps: list[float], vals: list[float]) -> dict:
    """Spearman ρ(step, value) + first/last/peak — the decay-vs-rise read."""
    from scipy.stats import spearmanr

    ok = [(s, v) for s, v in zip(steps, vals, strict=True) if v is not None and np.isfinite(v)]
    if len(ok) < 3:
        return {"n": len(ok), "rho": None, "p": None}
    s = np.array([o[0] for o in ok], dtype=float)
    v = np.array([o[1] for o in ok], dtype=float)
    rho, p = spearmanr(s, v)
    imax = int(np.argmax(v))
    return {
        "n": len(ok),
        "rho": float(rho),
        "p": float(p),
        "first": float(v[0]),
        "last": float(v[-1]),
        "peak": float(v[imax]),
        "peak_step": float(s[imax]),
        "peak_frac_through_ladder": float(imax / max(1, len(v) - 1)),
    }


def _summarize(curves: dict, coverage: dict, parity: list[dict], smoke: bool) -> dict:
    per_arm_layer: dict[str, dict] = {}
    for key, c in curves.items():
        pts = c["points"]
        if not pts:
            continue
        steps = [p["step"] for p in pts]
        entry = {
            "arm_id": c["arm_id"],
            "layer": c["layer"],
            "beh_key": c["beh_key"],
            "ctx_key": c["ctx_key"],
            "regime": c["regime"],
            "kind": c["kind"],
            "method": c["method"],
            "n_rungs": len(pts),
            "norm_trend": _trend(steps, [p["w_tf_norm"] for p in pts]),
            "stability_last": pts[-1]["cos_vs_verdict"],
            "split_half_min": min(
                (p["w_tf_split_half_cos"] for p in pts if p["w_tf_split_half_cos"] is not None),
                default=None,
            ),
        }
        for cn in ("delta", "r_B", "W_U_marker_row"):
            vals = [p["cos"].get(cn) for p in pts]
            if any(v is not None for v in vals):
                entry[f"trend_{cn}"] = _trend(steps, vals)
                band = (c["null_bands"].get(cn) or {}).get("nulls") or {}
                fam = (c["null_bands"].get(cn) or {}).get("primary_null_family")
                b = (band.get(fam) or {}) if fam else {}
                hi = b.get("p97_5")
                entry[f"n_rungs_above_null_{cn}"] = (
                    sum(1 for v in vals if v is not None and hi is not None and v > hi)
                    if hi is not None
                    else None
                )
                entry[f"null_p97_5_{cn}"] = hi
        # install coupling across rungs (marker arms have a real ladder)
        iv = [
            (p["step"], p["install"]["install"], p["cos"].get("delta"), p["w_tf_norm"])
            for p in pts
            if p.get("install") and p["install"].get("install") is not None
        ]
        if len(iv) >= 3:
            from scipy.stats import spearmanr

            inst = np.array([x[1] for x in iv], dtype=float)
            cd = np.array([x[2] if x[2] is not None else np.nan for x in iv], dtype=float)
            nm = np.array([x[3] for x in iv], dtype=float)
            m = np.isfinite(inst) & np.isfinite(cd)
            if m.sum() >= 3:
                r1_, p1_ = spearmanr(inst[m], cd[m])
                entry["rho_install_vs_cos_delta"] = {
                    "rho": float(r1_),
                    "p": float(p1_),
                    "n": int(m.sum()),
                }
            m2 = np.isfinite(inst) & np.isfinite(nm)
            if m2.sum() >= 3:
                r2_, p2_ = spearmanr(inst[m2], nm[m2])
                entry["rho_install_vs_norm"] = {
                    "rho": float(r2_),
                    "p": float(p2_),
                    "n": int(m2.sum()),
                }
        per_arm_layer[key] = entry

    def _agg(pred, field, sub="rho"):
        vals = [
            e[field][sub]
            for e in per_arm_layer.values()
            if pred(e) and e.get(field) and e[field].get(sub) is not None
        ]
        if not vals:
            return None
        a = np.array(vals, dtype=float)
        return {
            "n": len(vals),
            "median": float(np.median(a)),
            "mean": float(a.mean()),
            "frac_positive": float((a > 0).mean()),
            "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)),
        }

    abs_diffs = [p["abs_diff"] for p in parity]
    headline = {
        "n_arm_layer_curves": len(per_arm_layer),
        "n_rungs_total": int(sum(e["n_rungs"] for e in per_arm_layer.values())),
        "parity_vs_round1": {
            "n_compared": len(parity),
            "max_abs_diff": float(max(abs_diffs)) if abs_diffs else None,
            "median_abs_diff": float(np.median(abs_diffs)) if abs_diffs else None,
            "tolerance": PARITY_COS_ABS_TOL,
            "verdict": (
                "PASS"
                if abs_diffs
                and max(abs_diffs) <= PARITY_COS_ABS_TOL
                and len(parity) >= PARITY_MIN_UNITS
                else ("INCONCLUSIVE" if len(parity) < PARITY_MIN_UNITS else "FAIL")
            ),
        },
        "cos_delta_trend": {
            "all": _agg(lambda e: True, "trend_delta"),
            "content": _agg(lambda e: e["kind"] == "content", "trend_delta"),
            "marker": _agg(lambda e: e["kind"] == "marker", "trend_delta"),
        },
        "cos_rb_trend": {
            "all": _agg(lambda e: True, "trend_r_B"),
            "content": _agg(lambda e: e["kind"] == "content", "trend_r_B"),
            "marker": _agg(lambda e: e["kind"] == "marker", "trend_r_B"),
        },
        "cos_wu_trend_marker": _agg(lambda e: e["kind"] == "marker", "trend_W_U_marker_row"),
        "norm_trend": {
            "all": _agg(lambda e: True, "norm_trend"),
            "content": _agg(lambda e: e["kind"] == "content", "norm_trend"),
            "marker": _agg(lambda e: e["kind"] == "marker", "norm_trend"),
        },
        "install_coupling": {
            "rho_install_vs_cos_delta": {
                "all": _agg(lambda e: True, "rho_install_vs_cos_delta"),
                "content": _agg(lambda e: e["kind"] == "content", "rho_install_vs_cos_delta"),
                "marker": _agg(lambda e: e["kind"] == "marker", "rho_install_vs_cos_delta"),
            },
            "rho_install_vs_norm": {
                "all": _agg(lambda e: True, "rho_install_vs_norm"),
                "content": _agg(lambda e: e["kind"] == "content", "rho_install_vs_norm"),
                "marker": _agg(lambda e: e["kind"] == "marker", "rho_install_vs_norm"),
            },
            "install_metric_note": "the two families' install metrics are NOT "
            "comparable in level — marker = delta_logp_mean (nats, #1481 "
            "reads_by_step), content = Tier-1 selection-pool judged rate (#1481 "
            "rates_by_step). Only the WITHIN-arm rank correlations aggregated "
            "here are cross-family comparable.",
        },
        "coverage": {
            "n_arms": len(coverage),
            "n_lora_arms": sum(1 for c in coverage.values() if c["method"] == "lora"),
            "n_ft_arms_skipped_by_design": sum(1 for c in coverage.values() if c["method"] == "ft"),
            "lora_rungs_captured": int(
                sum(c["n_captured"] for c in coverage.values() if c["method"] == "lora")
            ),
            "lora_rungs_on_hub": int(
                sum(c["n_hub_rungs"] for c in coverage.values() if c["method"] == "lora")
            ),
            "arms_below_80pct": sorted(
                k for k, c in coverage.items() if c["method"] == "lora" and c["frac"] < 0.8
            ),
        },
    }
    return {"headline": headline, "per_arm_layer": per_arm_layer, "smoke": smoke, **_meta()}


# ── upload ───────────────────────────────────────────────────────────────────


def phase_upload(out_root: Path) -> dict:
    """Bulk-upload the per-rung stores (one commit per arm wave) + verify."""
    _phase("dyn_upload")
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    tree = out_root / "ckpt_dynamics_tf"
    assert tree.exists(), tree
    expected: list[str] = []
    for arm_dir in sorted(p for p in tree.iterdir() if p.is_dir()):
        dest = f"{HF_DYN_PREFIX}/{arm_dir.name}"
        url = hub.retry_transient(
            lambda a=arm_dir, d=dest: hub._upload(
                a, repo_id=X.HF_DATA_REPO, repo_type="dataset", path_in_repo=d
            ),
            what=f"ckpt-dynamics upload {arm_dir.name}",
        )
        if not url:
            raise RuntimeError(f"upload of {arm_dir} -> {dest} returned no path")
        expected += [f"{dest}/{f.name}" for f in sorted(arm_dir.glob("step-*.pt"))]
        print(f"[upload] {arm_dir.name}: {len(list(arm_dir.glob('step-*.pt')))} rungs", flush=True)
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), X.HF_DATA_REPO, expected, path_in_repo=HF_DYN_PREFIX, repo_type="dataset"
    )
    assert not missing, f"upload verify: {len(missing)} missing, e.g. {missing[:5]}"
    rep = {"n_files_verified": len(expected), "prefix": HF_DYN_PREFIX, **_meta()}
    _atomic_json(out_root / "upload_done.json", rep)
    print(f"[upload] verified {len(expected)} files under {HF_DYN_PREFIX}", flush=True)
    return rep


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=str(__doc__ or "").split("\n")[0])
    ap.add_argument("--phase", choices=("stage", "pilot", "capture", "analyze", "upload"))
    ap.add_argument("--out-root", default="/workspace/issue-1768-dyn")
    ap.add_argument(
        "--results-dir", default=str(REPO_ROOT / "eval_results/issue_1768/ckpt_dynamics")
    )
    ap.add_argument("--shard", default="0/1", help="i/N unit-index shard for --phase capture")
    ap.add_argument("--gpu-id", type=int, default=0, help="physical GPU (CVD-pinned by launcher)")
    ap.add_argument("--arms", default="", help="comma-separated arm_id filter")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--max-per-arm",
        type=int,
        default=0,
        help="cap rungs per arm (keeps the SELECTED rung first) — smoke sizing",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--refresh-ladders", action="store_true")
    ap.add_argument("--import-check", action="store_true", help="resolve deferred imports, exit 0")
    args = ap.parse_args()

    if args.import_check:  # Axis-1 import resolution on the REAL branch
        import peft  # noqa: F401
        import torch  # noqa: F401
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
        from scipy.stats import spearmanr  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        import issue1768_directions as _d  # noqa: F401
        from explore_persona_space.orchestrate import hub  # noqa: F401

        print("[import-check] ok:", ",".join(sorted({"peft", "torch", "hub", "spearmanr", "d"})))
        sys.stdout.flush()
        sys.exit(0)

    if not args.phase:
        ap.error("--phase is required (unless --import-check)")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    arms_filter = tuple(a for a in args.arms.split(",") if a)
    if args.refresh_ladders:
        enumerate_ladders(out_root, refresh=True)

    if args.phase == "stage":
        stage_inputs(out_root)
    elif args.phase == "pilot":
        phase_pilot(out_root, args.gpu_id)
    elif args.phase == "capture":
        i, n = (int(x) for x in args.shard.split("/"))
        phase_capture(
            out_root,
            (i, n),
            args.gpu_id,
            arms_filter,
            args.limit,
            args.smoke,
            args.max_per_arm,
        )
    elif args.phase == "analyze":
        phase_analyze(out_root, results_dir, args.smoke)
    elif args.phase == "upload":
        phase_upload(out_root)

    # explicit exit BEFORE C-extension finalization (gotchas.md PyGILState race)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
