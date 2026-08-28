"""#2587 FFR bank builder — manifest-sourced (floor-failed re-elicitation round).

The `ffr-9b-fire-gated` follow-up (plan v6 §4.1) runs the #2564 floor-failed
re-elicitation grid on Qwen3.5-9B. The banked parent manifest
(``bank2564_ffr_manifest.json``) carries the REALIZED context grid — 276
contexts / 672 pairs over 11 re-elicited wordings (persona p2/p3/p5/p6/p7,
stance s1b/s2a/s3a/s4a, hedging h1c/h2a) plus the 12 bare ``query::E::c*``
carriers — not a values dict, so ``bank2587.build_bank(tok, values=...)``
(which rebuilds the PARENT 9-axis grid and pins parent counts) structurally
cannot produce it. This module builds the battery-consumable bank dict
straight from the manifest, re-running the 9B-side gates:

- byte-level manifest sha pin + ``values_sha256`` content pin + selected-id
  set equality (fail loud on any drift from the banked round);
- grid completeness re-counted from the loaded artifact against the
  manifest's OWN recorded counts AND the plan literals (276/672/{132,204,
  204,132}); every pair endpoint resolves;
- per-pair slot byte-identity (user string identical within a pair; system
  identical across carriers for a fixed wording; install a-side system
  empty);
- no-"assistant"-substring in system strings + the q35 render gate
  (``bank2587.gate_render_q35``: one assistant header + closed-empty-think);
- ``changed_tokens`` recomputed under the q35 tokenizer and asserted >= 1
  per pair (the manifest's 7B/q25 values preserved as
  ``changed_tokens_q25``); per-wording realized q35 token counts recorded.

The returned bank's ``values_sha256`` is the MANIFEST's (sha of the
re-elicited 11-wording values blob), which differs from the main run's
pinned bank2564 values sha by construction — so every battery resume /
cell fingerprint (``Cfg.bank_values_sha`` -> ``_regime_fp``) is disjoint
from the main run's (plan §4.1 resume-disjointness clause).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from explore_persona_space.experiments.issue2587 import bank2587 as B

ISSUE = 2587
MODEL_ID = B.MODEL_ID  # Qwen/Qwen3.5-9B — the FFR round evaluates the same 9B
SOURCE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # the manifest's producing model (pin check)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
FFR_MANIFEST_REPO_PATH = (
    "issue2564_minpair/manifests/floor_failed_reelicitation/bank2564_ffr_manifest.json"
)
# Byte-level pin of the banked manifest (sha256 of the raw file; probed live
# 2026-08-27, 286,889 bytes) — ANY byte drift fails loud before parsing.
FFR_MANIFEST_SHA256 = "cf481c80f31fa3def29468dd4a1e53771518b43a858f2fff23ec516550d925ac"
# Content pin: the manifest's own values_sha256 (plan §4.1 literal).
FFR_VALUES_SHA256 = "f5118984bd8266dfe6d990504da40522e3febb66b70895e60930ba96703c0db4"

# Selected-id sets (plan §4.1; fail-loud content pin against manifest["selected"]).
FFR_SELECTED: dict[str, tuple[str, ...]] = {
    "persona": ("p2", "p3", "p5", "p6", "p7"),
    "stance": ("s1b", "s2a", "s3a", "s4a"),
    "hedging": ("h1c", "h2a"),
}
FFR_AXES: tuple[str, ...] = ("hedging", "persona", "stance")
BARE_CELL = "query"  # the 12 empty-system install anchors live in this cell

N_FFR_CONTEXTS = 276
N_FFR_PAIRS = 672
N_FFR_CARRIERS = 12
FFR_PAIR_CLASS_COUNTS = {
    "install": 132,
    "swap": 204,
    "famswap": 204,
    "instruction_paraphrase": 132,
}
FFR_KIND_COUNTS = {"value": 132, "para": 132, "E": 12}
# Parent widths (floors in the judge use these, never realized widths).
FFR_PARENT_WIDTH = {"persona": 5, "stance": 5, "hedging": 2}


class Bank2587FfrGateError(RuntimeError):
    """A failed FFR bank gate — a mis-shaped bank invalidates every read."""


# ── manifest fetch (sha-pinned; revision recorded) ─────────────────────────


def fetch_ffr_manifest(local_path: Path | str | None = None) -> tuple[dict, dict]:
    """Fetch + sha-gate the banked FFR manifest; returns (manifest, provenance).

    ``local_path`` (tests / pre-staged pod copies) bypasses the HF fetch but
    NEVER the sha gates. The HF path resolves the dataset revision ONCE and
    records it (plan §4.1: resolved repo revision recorded into the run
    manifest)."""
    if local_path is not None:
        raw = Path(local_path).read_bytes()
        prov: dict = {"source": "local", "path": str(local_path)}
    else:
        from huggingface_hub import HfApi, hf_hub_download

        revision = HfApi().dataset_info(HF_DATA_REPO).sha
        assert revision, f"could not resolve dataset revision for {HF_DATA_REPO}"
        got = hf_hub_download(
            HF_DATA_REPO,
            filename=FFR_MANIFEST_REPO_PATH,
            repo_type="dataset",
            revision=revision,
        )
        raw = Path(got).read_bytes()
        prov = {
            "source": "hf",
            "repo_id": HF_DATA_REPO,
            "path_in_repo": FFR_MANIFEST_REPO_PATH,
            "revision": revision,
        }
    sha = hashlib.sha256(raw).hexdigest()
    if sha != FFR_MANIFEST_SHA256:
        raise Bank2587FfrGateError(
            f"FFR manifest byte sha mismatch: {sha} != pinned {FFR_MANIFEST_SHA256} ({prov})"
        )
    manifest = json.loads(raw)
    if manifest.get("values_sha256") != FFR_VALUES_SHA256:
        raise Bank2587FfrGateError(
            f"FFR manifest values_sha256 {manifest.get('values_sha256')!r} != pinned "
            f"{FFR_VALUES_SHA256}"
        )
    prov["manifest_sha256"] = sha
    prov["n_bytes"] = len(raw)
    return manifest, prov


# ── gates (plan §4.1 / §7 "FFR bank gates") ────────────────────────────────


def gate_ffr_pin(manifest: dict) -> None:
    """Selected-id sets + producing-round identity (fail-loud content pin)."""
    sel = {k: tuple(v) for k, v in manifest["selected"].items()}
    exp = {k: tuple(v) for k, v in FFR_SELECTED.items()}
    if sel != exp:
        raise Bank2587FfrGateError(f"FFR selected-id sets drifted: {sel} != {exp}")
    if manifest.get("issue") != 2564 or manifest.get("round") != "floor-failed-reelicitation":
        raise Bank2587FfrGateError(
            f"manifest identity drifted: issue={manifest.get('issue')} "
            f"round={manifest.get('round')!r}"
        )
    if manifest.get("model_id") != SOURCE_MODEL_ID:
        raise Bank2587FfrGateError(f"manifest model_id {manifest.get('model_id')!r} unexpected")


def gate_ffr_grid(manifest: dict) -> None:
    """Grid completeness re-counted from the loaded artifact: the manifest's
    OWN counts, the plan literals, per-class counts, kind counts, endpoint
    resolution, and full (axis x wording x carrier) coverage."""
    contexts, pairs = manifest["contexts"], manifest["pairs"]
    if not (len(contexts) == manifest["n_contexts"] == N_FFR_CONTEXTS):
        raise Bank2587FfrGateError(
            f"context count: len={len(contexts)} manifest={manifest['n_contexts']} "
            f"expected={N_FFR_CONTEXTS}"
        )
    if not (len(pairs) == manifest["n_pairs"] == N_FFR_PAIRS):
        raise Bank2587FfrGateError(
            f"pair count: len={len(pairs)} manifest={manifest['n_pairs']} expected={N_FFR_PAIRS}"
        )
    cls_counts: dict[str, int] = {}
    for p in pairs:
        cls_counts[p["pair_class"]] = cls_counts.get(p["pair_class"], 0) + 1
        for side in ("a", "b"):
            if p[side] not in contexts:
                raise Bank2587FfrGateError(f"pair {p['pair_id']}: endpoint {p[side]} unresolved")
    if cls_counts != FFR_PAIR_CLASS_COUNTS or cls_counts != manifest["pair_class_counts"]:
        raise Bank2587FfrGateError(
            f"pair-class counts: recounted={cls_counts} manifest={manifest['pair_class_counts']} "
            f"expected={FFR_PAIR_CLASS_COUNTS}"
        )
    kind_counts: dict[str, int] = {}
    for ctx in contexts.values():
        kind_counts[ctx["kind"]] = kind_counts.get(ctx["kind"], 0) + 1
    if kind_counts != FFR_KIND_COUNTS:
        raise Bank2587FfrGateError(f"context kind counts {kind_counts} != {FFR_KIND_COUNTS}")
    carriers = sorted({ctx["carrier"] for ctx in contexts.values()})
    if len(carriers) != N_FFR_CARRIERS:
        raise Bank2587FfrGateError(f"{len(carriers)} carriers != {N_FFR_CARRIERS}")
    _gate_ffr_coverage(contexts, carriers)


def _gate_ffr_coverage(contexts: dict, carriers: list[str]) -> None:
    """Full coverage: every (axis, wording[, para]) x carrier + every bare carrier."""
    for car in carriers:
        if f"{BARE_CELL}::E::{car}" not in contexts:
            raise Bank2587FfrGateError(f"missing bare context {BARE_CELL}::E::{car}")
        for axis, vids in FFR_SELECTED.items():
            for vid in vids:
                for slot_vid in (vid, f"{vid}p"):
                    cid = f"{axis}::{slot_vid}::{car}"
                    if cid not in contexts:
                        raise Bank2587FfrGateError(f"missing context {cid}")


def gate_ffr_slot_identity(manifest: dict) -> None:
    """Byte-identity of non-varied slots (plan §4.1): the carrier user string
    is identical within a pair AND across all contexts of one carrier; a
    wording's system string is identical across carriers; install a-sides
    have the EMPTY system; the two sides of a pair never share a system."""
    contexts, pairs = manifest["contexts"], manifest["pairs"]
    user_by_carrier: dict[str, str] = {}
    system_by_slot: dict[tuple[str, str], str] = {}
    for ctx in contexts.values():
        u = user_by_carrier.setdefault(ctx["carrier"], ctx["user"])
        if ctx["user"] != u:
            raise Bank2587FfrGateError(f"{ctx['id']}: user string differs within carrier")
        key = (ctx["cell"], ctx["value_id"])
        s = system_by_slot.setdefault(key, ctx["system"])
        if ctx["system"] != s:
            raise Bank2587FfrGateError(f"{ctx['id']}: system string differs across carriers")
    if system_by_slot[(BARE_CELL, "E")] != "":
        raise Bank2587FfrGateError("bare install anchors must carry the EMPTY system string")
    for p in pairs:
        a, b = contexts[p["a"]], contexts[p["b"]]
        if p["a"] == p["b"]:
            raise Bank2587FfrGateError(f"pair {p['pair_id']}: degenerate (a == b)")
        if a["user"] != b["user"]:
            raise Bank2587FfrGateError(f"pair {p['pair_id']}: user strings differ within pair")
        if a["carrier"] != b["carrier"] or a["carrier"] != p["carrier"]:
            raise Bank2587FfrGateError(f"pair {p['pair_id']}: carrier mismatch")
        if a["system"] == b["system"]:
            raise Bank2587FfrGateError(f"pair {p['pair_id']}: system strings identical")


# ── bank build (strings then token gates; mirrors bank2587's split) ────────


def build_ffr_bank_strings(
    manifest: dict | None = None, local_manifest_path: Path | str | None = None
) -> dict:
    """F0 entry (VM, repo venv, CPU): fetch/gate the manifest and return the
    battery-consumable bank dict with STRING gates run. ``token_gates`` is
    ``None`` until :func:`run_ffr_token_gates` (pod, q35 tokenizer)."""
    if manifest is None:
        manifest, prov = fetch_ffr_manifest(local_manifest_path)
    else:
        prov = {"source": "caller-supplied"}
        if manifest.get("values_sha256") != FFR_VALUES_SHA256:
            raise Bank2587FfrGateError(
                f"caller-supplied manifest values_sha256 "
                f"{manifest.get('values_sha256')!r} != pinned {FFR_VALUES_SHA256}"
            )
    gate_ffr_pin(manifest)
    gate_ffr_grid(manifest)
    gate_ffr_slot_identity(manifest)
    contexts = {cid: dict(ctx) for cid, ctx in manifest["contexts"].items()}
    pairs = [dict(p) for p in manifest["pairs"]]
    B.gate_no_assistant_in_system_strings(contexts)
    return {
        "issue": ISSUE,
        "model_id": MODEL_ID,
        "bank_source": "ffr",
        "parent_pin": B.PIN,
        "ffr_manifest": prov,
        "selected": {k: list(v) for k, v in FFR_SELECTED.items()},
        "selection": manifest["selection"],
        "values_sha256": manifest["values_sha256"],
        "n_contexts": len(contexts),
        "n_pairs": len(pairs),
        "pair_class_counts": dict(FFR_PAIR_CLASS_COUNTS),
        "contexts": contexts,
        "pairs": pairs,
        "string_gates": {
            "verdict": "PASS",
            "gates_run": ["ffr_pin", "ffr_grid", "ffr_slot_identity", "no_assistant"],
        },
        "token_gates": None,
    }


def run_ffr_token_gates(tok, bank: dict) -> dict:
    """Pod entry (model venv, q35 tokenizer): render gate + q35
    ``changed_tokens`` (>= 1 per pair; the manifest's 7B values preserved as
    ``changed_tokens_q25``) + realized per-wording q35 token counts. Mutates
    ``bank`` (sets ``token_gates``); returns the record."""
    contexts, pairs = bank["contexts"], bank["pairs"]
    rendered = {cid: B.render_context_q35(tok, ctx) for cid, ctx in contexts.items()}
    B.gate_render_q35(rendered)
    ids_by_context: dict[str, list[int]] = {}
    for cid in contexts:
        ids = tok(rendered[cid], add_special_tokens=False)["input_ids"]
        assert len(ids) >= 4, (len(ids), cid)
        ids_by_context[cid] = ids
    for p in pairs:
        p["changed_tokens_q25"] = int(p["changed_tokens"])  # manifest 7B-tokenizer value
    B._bk().attach_changed_tokens(pairs, ids_by_context)  # q35 recompute; asserts >= 1
    for p in pairs:
        p["changed_tokens_q35"] = int(p["changed_tokens"])
    system_by_slot = {
        (ctx["cell"], ctx["value_id"]): ctx["system"]
        for ctx in contexts.values()
        if ctx["cell"] != BARE_CELL
    }
    token_counts = {
        f"{axis}::{vid}": B._n_tokens(tok, system_by_slot[(axis, vid)])
        for axis, vids in FFR_SELECTED.items()
        for base in vids
        for vid in (base, f"{base}p")
    }
    record = {
        "verdict": "PASS",
        "gates_run": ["render_q35", "changed_tokens_q35", "q35_token_counts"],
        "tokenizer_id": getattr(tok, "name_or_path", None),
        "q35_system_token_counts": token_counts,
        "changed_tokens_q35_min": min(p["changed_tokens_q35"] for p in pairs),
        "changed_tokens_q35_max": max(p["changed_tokens_q35"] for p in pairs),
        "changed_tokens_q25_min": min(p["changed_tokens_q25"] for p in pairs),
        "changed_tokens_q25_max": max(p["changed_tokens_q25"] for p in pairs),
    }
    bank["token_gates"] = record
    return record


def build_ffr_bank(tok, local_manifest_path: Path | str | None = None) -> dict:
    """Full FFR bank build: manifest fetch + string gates + q35 token gates.

    The seam ``issue2587_battery_run.py --bank-source ffr`` calls (plan §4.2)."""
    bank = build_ffr_bank_strings(local_manifest_path=local_manifest_path)
    run_ffr_token_gates(tok, bank)
    return bank


def main(argv: list[str] | None = None) -> int:
    """F0 (``--strings-only``: VM, repo venv) / pod (default: q35 tokenizer)
    FFR bank build; writes the bank manifest (bank2587's pinned writer)."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--strings-only", action="store_true", help="F0: string gates only")
    ap.add_argument("--tokenizer", default=MODEL_ID, help="HF tokenizer id (pod token gates)")
    ap.add_argument("--manifest", default=None, help="local manifest path (default: HF fetch)")
    ap.add_argument(
        "--out",
        type=Path,
        default=B._repo_root() / "eval_results" / "issue_2587" / "bank_manifest_ffr.json",
    )
    args = ap.parse_args(argv)

    # Shared-VM thread caps (#847): bind BEFORE any transformers->torch import.
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    if args.strings_only:
        bank = build_ffr_bank_strings(local_manifest_path=args.manifest)
    else:
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        except OSError:
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
        bank = build_ffr_bank(tok, local_manifest_path=args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    B.write_bank_manifest(bank, args.out)
    phase = "ffr-bank-strings" if args.strings_only else "ffr-bank"
    print(
        f"[phase={phase}] source=ffr contexts={bank['n_contexts']} pairs={bank['n_pairs']} "
        f"string_gates=PASS token_gates="
        f"{'PASS' if bank['token_gates'] else 'not-run'} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
