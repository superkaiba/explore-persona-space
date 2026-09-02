"""Issue #2658 unit 3 — item_id -> prompt-text resolution with frozen sha256 pins.

Units 1-2 froze a TEXT-FREE registry (``eval_results/issue_2658/frame_manifest.json``):
pilot items are keyed by ``item_id`` + ``source_ref`` only.  This module is the
single place prompt TEXT is materialized from those references:

- **bank items** (``{row}|{frame}|{bank}#{i}``) read the committed offline query
  banks via ``explore_persona_space.artifacts.banks`` — no network;
- **correctness items** (``{row}|{frame}|{context_id}``) rebuild the exact #2388
  prompt text through the parent's own loaders, vendored byte-exact at the
  parent pin under ``scripts/issue2658_vendored/`` (the unmerged issue-2388
  branch object is unreachable on fresh-pod partial clones and git-less SLURM
  trees, so the pinned bytes are committed, sha-verified, and loaded by path).

Every resolved text is verified against the frozen pin table
``eval_results/issue_2658/prompt_pins.json`` (item_id -> sha256 of the prompt
text, utf-8; TEXT-FREE, committed).  The FIRST resolution freezes the table
(``--freeze-pins``); every later resolution — generation, capture — verifies and
RAISES ``RowHashMismatchError`` on any drift (plan §5 immutable prompt
manifests).  An unresolvable item is a LOUD failure naming the item — never a
skipped cell, never a substituted prompt.

Evidence packets (sycophancy / hallucination judge inputs) have a frozen-file
contract here (``resolve_evidence_packet``): the packet store does not exist
yet, so any request fails loud naming the unit-4 dependency — generation never
consumes evidence (the model never sees it), so unit 3 is not blocked on it.

CONTENT HYGIENE: prompt text flows file -> memory -> caller; logs and persisted
artifacts carry only item_ids, counts, and sha256 digests — never text.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # #847 thread caps + HF token, before any heavy import

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_common as C  # noqa: E402
import issue2658_frames as F  # noqa: E402

REPO_ROOT = _SCRIPTS_DIR.parent
VENDOR_DIR = _SCRIPTS_DIR / "issue2658_vendored"
VENDOR_MANIFEST = VENDOR_DIR / "vendor_manifest.json"
PIN_PATH = F.OUT_DIR / "prompt_pins.json"
EVIDENCE_PATH = F.OUT_DIR / "evidence_packets.json"
SHA_DOMAIN = "prompt_text_utf8"  # sha256 over the raw prompt text (utf-8 bytes)

# context_id prefix -> loader attribute on the pinned #2388 gen module.
_BENCH_LOADERS = {
    "mathfull-": "load_math_full",
    "mmlupro-": "load_mmlu_pro_full",
    "humaneval-": "load_humaneval_full",
    "mbpp-": "load_mbpp_full",
    "bcb-": "load_bigcodebench_full",
    "lcb-": "load_lcb_v5",
    "leetcode-": "load_leetcode",
}


class TextResolutionError(C.Issue2658GuardError):
    """An item_id could not be resolved to prompt text (loud, never skipped)."""


class EvidencePacketMissingError(C.Issue2658GuardError):
    """A frozen evidence packet was requested but has not been built yet."""


class VendorPinError(C.Issue2658GuardError):
    """A vendored pinned module's bytes deviate from the recorded sha256."""


@dataclass
class ResolvedItem:
    """One resolved prompt. ``text`` is memory-only (repr=False; never logged)."""

    item_id: str
    prompt_sha256: str
    source_ref: str
    text: str = field(repr=False, default="")


# ---------------------------------------------------------------------------
# Staging (shared-VM only): keep multi-GB dataset pulls off the boot disk.
# ---------------------------------------------------------------------------
def apply_datasets_cache() -> Path | None:
    """On the shared VM, stage HF datasets/hub caches under the issue dl dir.

    ``data/issue_2658/hf_dl`` resolves to the data disk in this clone; pods and
    SLURM lanes keep their own redirected ``HF_HOME`` (no override there).
    Must run BEFORE the first ``huggingface_hub``/``datasets`` import in the
    process (both freeze env at import) — call at main() entry, not per-item.
    """
    if not Path("/mnt/eps-data").exists():
        return None
    if os.environ.get("HF_DATASETS_CACHE") and os.environ.get("HF_HUB_CACHE"):
        return None
    if "huggingface_hub.constants" in sys.modules or "datasets" in sys.modules:
        raise TextResolutionError(
            "apply_datasets_cache() called after huggingface_hub/datasets import — "
            "cache env is frozen at import; call it at process entry"
        )
    root = F.ISSUE_DL
    (root / "datasets_cache").mkdir(parents=True, exist_ok=True)
    (root / "hub_cache").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "datasets_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "hub_cache"))
    print(f"[resolver] staged HF caches under {root} (shared-VM only)", flush=True)
    return root


# ---------------------------------------------------------------------------
# Vendored pinned #2388 modules.
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_vendor_pins() -> dict[str, Path]:
    """Verify every vendored blob's bytes against the recorded sha256 pins.

    Returns {source_basename_without_suffix: vendored path}. Raises
    ``VendorPinError`` on any missing file or digest drift (domain: file bytes).
    """
    manifest = json.loads(VENDOR_MANIFEST.read_text())
    out: dict[str, Path] = {}
    for fname, meta in manifest["files"].items():
        path = VENDOR_DIR / fname
        if not path.exists():
            raise VendorPinError(f"vendored pinned file missing: {path}")
        got = _sha256_file(path)
        if got != meta["sha256"]:
            raise VendorPinError(
                f"vendored pin drift for {fname}: sha256 {got} != recorded {meta['sha256']} "
                f"(source {manifest['source_commit']}:{meta['source_path']})"
            )
        out[fname.removesuffix(".py.pinned")] = path
    return out


def _load_by_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VendorPinError(f"cannot load pinned module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PINNED_GEN: ModuleType | None = None


def load_pinned_gen_module() -> ModuleType:
    """Materialize + exec the vendored #2388 gen module (loaders only).

    The pinned gen asserts a ``pyproject.toml`` two levels above itself and
    imports ``scripts.issue2388_spread_pilot`` at module top, so the vendored
    bytes land under ``<tmp>/scripts/`` beside a stub pyproject, and the
    vendored spread_pilot is pre-registered under the dotted name so the
    pinned bytes — not main's drifted copy — satisfy the import.  The module's
    ``REPO_ROOT`` is repointed to the real checkout after exec.
    """
    global _PINNED_GEN
    if _PINNED_GEN is not None:
        return _PINNED_GEN
    vend = verify_vendor_pins()
    dotted = "scripts.issue2388_spread_pilot"
    if dotted in sys.modules:
        raise VendorPinError(
            f"{dotted} is already imported in this process; refusing to shadow a live "
            "module with the pinned copy (run resolution in a fresh process)"
        )
    d = Path(tempfile.mkdtemp(prefix="i2658-pinned-"))
    (d / "pyproject.toml").write_text("# stub satisfying the pinned parent's repo-root assert\n")
    sdir = d / "scripts"
    sdir.mkdir()
    shutil.copyfile(vend["issue2388_spread_pilot"], sdir / "issue2388_spread_pilot.py")
    shutil.copyfile(vend["issue2388_gen"], sdir / "issue2388_gen.py")
    spread = _load_by_path("issue2658_pinned_spread_pilot", sdir / "issue2388_spread_pilot.py")
    sys.modules[dotted] = spread  # gen's `from scripts.issue2388_spread_pilot import ...`
    gen = _load_by_path("issue2658_pinned_gen", sdir / "issue2388_gen.py")
    gen.REPO_ROOT = REPO_ROOT  # never leave the tmp stub root live
    for attr in _BENCH_LOADERS.values():
        if not hasattr(gen, attr):
            raise VendorPinError(f"pinned gen module lacks loader {attr!r} — wrong pin?")
    _PINNED_GEN = gen
    return gen


# ---------------------------------------------------------------------------
# item_id parsing + resolution.
# ---------------------------------------------------------------------------
def parse_item_id(item_id: str) -> tuple[str, str, str]:
    """Split ``{row}|{frame}|{ref}`` and validate row + frame against FRAMES."""
    parts = item_id.split("|")
    if len(parts) != 3:
        raise TextResolutionError(f"malformed item_id {item_id!r}; expected row|frame|ref")
    row, frame_name, ref = parts
    if row not in C.ROW_IDS:
        raise TextResolutionError(f"unknown row {row!r} in item_id {item_id!r}")
    frame_names = {fr.name for fr in F.FRAMES[row].frames}
    if frame_name not in frame_names:
        raise TextResolutionError(
            f"unknown frame {frame_name!r} for row {row!r} (registered: {sorted(frame_names)})"
        )
    if not ref:
        raise TextResolutionError(f"empty ref in item_id {item_id!r}")
    return row, frame_name, ref


def _frame_spec(row: str, frame_name: str) -> F.FrameSpec:
    for fr in F.FRAMES[row].frames:
        if fr.name == frame_name:
            return fr
    raise TextResolutionError(f"frame {frame_name!r} not registered for row {row!r}")


def _bench_loader_attr(context_id: str) -> str:
    for prefix, attr in _BENCH_LOADERS.items():
        if context_id.startswith(prefix):
            return attr
    raise TextResolutionError(
        f"context_id {context_id!r} matches no known benchmark prefix ({sorted(_BENCH_LOADERS)})"
    )


def resolve_items(item_ids: list[str], *, verify_pins: bool = True) -> dict[str, ResolvedItem]:
    """Resolve every item_id to its prompt text. Loud failure on ANY miss.

    Banks load once per bank (committed, offline); benchmarks load once per
    loader through the vendored pinned #2388 modules (network on first use;
    stage caches via ``apply_datasets_cache()`` on the shared VM).  With
    ``verify_pins`` (the default), every resolved sha is checked against the
    frozen pin table and any mismatch raises ``RowHashMismatchError``.
    """
    if len(set(item_ids)) != len(item_ids):
        raise TextResolutionError("duplicate item_ids in resolution request")
    parsed = {iid: parse_item_id(iid) for iid in item_ids}

    # Group the work: bank name -> [(item_id, index)], loader attr -> [item_id].
    bank_requests: dict[str, list[tuple[str, int]]] = {}
    bench_requests: dict[str, list[str]] = {}
    keyed_requests: dict[tuple[str, str], list[str]] = {}
    for iid, (row, frame_name, ref) in parsed.items():
        spec = _frame_spec(row, frame_name)
        if spec.source_kind == "bank":
            bank = spec.source_ref.split(":", 1)[1]
            head, sep, idx_s = ref.rpartition("#")
            if not sep or head != bank or not idx_s.isdigit():
                raise TextResolutionError(
                    f"bank ref {ref!r} of {iid!r} does not match '{bank}#<index>'"
                )
            bank_requests.setdefault(bank, []).append((iid, int(idx_s)))
        elif spec.source_kind == "benchmark":
            bench_requests.setdefault(_bench_loader_attr(ref), []).append(iid)
        elif spec.source_kind == "keyed":
            keyed_requests.setdefault((row, frame_name), []).append(iid)
        else:
            raise TextResolutionError(f"unknown source_kind {spec.source_kind!r} for {iid!r}")

    out: dict[str, ResolvedItem] = {}

    if keyed_requests:
        # Composed-text frames resolve through the SAME builder the frame pass
        # uses (F._load_keyed_frame), so the bytes are identical by construction
        # rather than by a second implementation that has to be kept in step —
        # a divergent composer here would surface only as a pin mismatch.
        for (row, frame_name), iids in sorted(keyed_requests.items()):
            spec = _frame_spec(row, frame_name)
            built = {it.item_id: it for it in F._load_keyed_frame(row, spec)}
            for iid in iids:
                item = built.get(iid)
                if item is None:
                    raise TextResolutionError(
                        f"{iid!r} not produced by the keyed builder for "
                        f"{row}|{frame_name} ({len(built)} items) — wrong key or a "
                        "changed selector"
                    )
                out[iid] = ResolvedItem(
                    item_id=iid,
                    prompt_sha256=item.prompt_sha256,
                    source_ref=spec.source_ref,
                    text=item.text,
                )
            print(f"[resolver] keyed {row}|{frame_name}: resolved {len(iids)} items", flush=True)

    if bank_requests:
        from explore_persona_space.artifacts import banks

        for bank, reqs in sorted(bank_requests.items()):
            items = banks.load_bank(bank)
            for iid, idx in reqs:
                if not (0 <= idx < len(items)):
                    raise TextResolutionError(
                        f"bank index {idx} of {iid!r} out of range (bank {bank!r} has "
                        f"{len(items)} items)"
                    )
                text = items[idx]
                out[iid] = ResolvedItem(
                    item_id=iid,
                    prompt_sha256=F._sha_text(text),
                    source_ref=f"query_banks:{bank}",
                    text=text,
                )
            print(f"[resolver] bank {bank}: resolved {len(reqs)} items", flush=True)

    if bench_requests:
        gen = load_pinned_gen_module()
        for attr, iids in sorted(bench_requests.items()):
            rows = getattr(gen, attr)()  # pinned loader; count-asserts internally
            by_id = {r["item_id"]: r["prompt"] for r in rows}
            if len(by_id) != len(rows):
                raise TextResolutionError(f"pinned loader {attr} returned duplicate item_ids")
            for iid in iids:
                context_id = parsed[iid][2]
                if context_id not in by_id:
                    raise TextResolutionError(
                        f"context_id {context_id!r} of {iid!r} not found in pinned loader "
                        f"{attr} pool ({len(by_id)} items) — wrong pin or wrong id"
                    )
                text = by_id[context_id]
                out[iid] = ResolvedItem(
                    item_id=iid,
                    prompt_sha256=F._sha_text(text),
                    source_ref=f"issue2388_pinned:{attr}",
                    text=text,
                )
            print(f"[resolver] loader {attr}: resolved {len(iids)} items", flush=True)

    missing = [iid for iid in item_ids if iid not in out]
    if missing:
        raise TextResolutionError(f"{len(missing)} items unresolved (e.g. {missing[:3]})")

    if verify_pins:
        verify_against_pins(out)
    return out


# ---------------------------------------------------------------------------
# Frozen pin table.
# ---------------------------------------------------------------------------
def load_pins() -> dict[str, Any]:
    if not PIN_PATH.exists():
        raise TextResolutionError(
            f"frozen pin table missing at {PIN_PATH}; run "
            "`issue2658_text_resolver.py --freeze-pins` once before any resolution"
        )
    body = json.loads(PIN_PATH.read_text())
    if body.get("sha_domain") != SHA_DOMAIN:
        raise C.RowHashMismatchError(
            f"pin table sha_domain {body.get('sha_domain')!r} != {SHA_DOMAIN!r}"
        )
    return body


def verify_against_pins(resolved: dict[str, ResolvedItem]) -> None:
    """Every resolved item must match its frozen pin; drift RAISES."""
    pins = load_pins()["items"]
    unpinned = [iid for iid in resolved if iid not in pins]
    if unpinned:
        raise TextResolutionError(
            f"{len(unpinned)} resolved items have no frozen pin (e.g. {unpinned[:3]}); "
            "the pin table covers exactly the registered pilot items"
        )
    for iid, item in resolved.items():
        C.assert_row_hash(item.text, pins[iid]["prompt_sha256"])


def pilot_item_ids() -> list[tuple[str, str, str]]:
    """Ordered (row, cell, item_id) triples from the committed frame manifest.

    Verifies manifest immutability first; duplicate item_ids across cells RAISE.
    """
    body = json.loads(F.FRAME_MANIFEST_PATH.read_text())
    F.assert_manifest_immutable(body)
    triples: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for rr in body["rows"]:
        sel = rr["pilot_selection"]["per_cell_item_ids"]
        for cell in sorted(sel):
            for iid in sel[cell]:
                if iid in seen:
                    raise TextResolutionError(f"item_id {iid!r} appears in two pilot cells")
                seen.add(iid)
                triples.append((rr["row"], cell, iid))
    if not triples:
        raise TextResolutionError("frame manifest yielded zero pilot items")
    return triples


def superfamily_of(body: dict, row: str, item_id: str) -> str:
    """Superfamily id for one pilot item from the frame manifest body."""
    for rr in body["rows"]:
        if rr["row"] == row:
            sf = rr["item_superfamily"].get(item_id)
            if not sf:
                raise TextResolutionError(f"no superfamily recorded for {item_id!r}")
            return sf
    raise TextResolutionError(f"row {row!r} not in frame manifest")


def freeze_pins(rows: list[str] | None = None) -> dict[str, Any]:
    """Resolve ALL pilot items and freeze the text-free pin table.

    Idempotent: an existing table must match the fresh resolution EXACTLY
    (any drift raises ``RowHashMismatchError``); partial-row freezes are
    refused — the table always covers the full registered pilot.
    """
    if rows is not None:
        raise TextResolutionError(
            "freeze_pins covers the FULL registered pilot by design; per-row freezes "
            "would fragment the immutability contract"
        )
    triples = pilot_item_ids()
    resolved = resolve_items([iid for _, _, iid in triples], verify_pins=False)
    items = {
        iid: {"prompt_sha256": r.prompt_sha256, "source_ref": r.source_ref}
        for iid, r in resolved.items()
    }
    body = {
        "issue": 2658,
        "sha_domain": SHA_DOMAIN,
        "note": (
            "Frozen prompt-text pins for the registered pilot items. sha256 over the "
            "raw prompt text (utf-8). TEXT-FREE by design; drift on any later "
            "resolution raises RowHashMismatchError."
        ),
        "n_items": len(items),
        "items": dict(sorted(items.items())),
    }
    if PIN_PATH.exists():
        existing = json.loads(PIN_PATH.read_text())
        if existing.get("items") != body["items"]:
            old = existing.get("items", {})
            drift = [k for k in body["items"] if old.get(k) != body["items"][k]]
            extra = [k for k in old if k not in body["items"]]
            raise C.RowHashMismatchError(
                f"pin table drift on re-freeze: {len(drift)} changed/new "
                f"(e.g. {drift[:3]}), {len(extra)} stale (e.g. {extra[:3]}); the frozen "
                "table is immutable"
            )
        print(f"[resolver] pin table already frozen ({len(items)} items) — verified", flush=True)
        return existing
    PIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PIN_PATH.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, PIN_PATH)
    print(f"[resolver] froze {len(items)} prompt pins -> {PIN_PATH}", flush=True)
    return body


# ---------------------------------------------------------------------------
# Rendering (shared by generation + capture — bit-identical inputs).
# ---------------------------------------------------------------------------
def chat_template_guard(tokenizer) -> None:
    """The live tokenizer's chat template must match the frozen pin."""
    tpl = tokenizer.chat_template
    if not isinstance(tpl, str) or not tpl:
        raise C.RowHashMismatchError("tokenizer exposes no chat_template string")
    got = hashlib.sha256(tpl.encode("utf-8")).hexdigest()
    if got != C.CHAT_TEMPLATE_SHA256:
        raise C.RowHashMismatchError(
            f"chat_template sha {got} != frozen pin {C.CHAT_TEMPLATE_SHA256}"
        )


def render_user_prompt(tokenizer, text: str) -> str:
    """The ONE rendering used by generation AND capture: a single user turn,
    no system message, with the generation prompt appended (plan §5)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Evidence packets (frozen-file contract; consumer is unit 4's judge phase).
# ---------------------------------------------------------------------------
def evidence_packet_sha256(packet: dict[str, Any]) -> str:
    """Canonical packet digest: sha256 over sorted-keys compact JSON (utf-8)."""
    payload = json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_evidence_packet(row: str, item_id: str) -> tuple[dict[str, Any], str]:
    """Frozen evidence packet for one evidence-conditioned item.

    Raises ``EvidencePacketMissingError`` (loud) while the frozen packet store
    has not been built — the judge phase (unit 4) depends on it; generation
    never does (the model never sees evidence).
    """
    construct = C.CONSTRUCTS[row]
    if not construct.uses_evidence_packet:
        raise ValueError(f"row {row!r} does not use evidence packets (caller bug)")
    if not EVIDENCE_PATH.exists():
        raise EvidencePacketMissingError(
            f"frozen evidence packets not built: {EVIDENCE_PATH} is absent. The "
            "evidence-conditioned rows (sycophancy, hallucination) need frozen packets "
            "BEFORE the unit-4 judge phase; see the persisted concern on task 2658."
        )
    store = json.loads(EVIDENCE_PATH.read_text())
    entry = store.get("items", {}).get(item_id)
    if entry is None:
        raise EvidencePacketMissingError(
            f"no frozen evidence packet for {item_id!r} in {EVIDENCE_PATH}"
        )
    packet, stored_sha = entry["packet"], entry["evidence_sha256"]
    got = evidence_packet_sha256(packet)
    if got != stored_sha:
        raise C.RowHashMismatchError(
            f"evidence packet drift for {item_id!r}: {got} != stored {stored_sha}"
        )
    return packet, stored_sha


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze-pins", action="store_true", help="resolve all pilot items, freeze")
    ap.add_argument("--verify", action="store_true", help="resolve all pilot items vs pins")
    ap.add_argument("--import-check", action="store_true", help="static arg/bind check only")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[resolver] import-check OK", flush=True)
        return 0
    if not (args.freeze_pins or args.verify):
        raise SystemExit("pick one of --freeze-pins / --verify / --import-check")
    apply_datasets_cache()
    if args.freeze_pins:
        body = freeze_pins()
        per_source: dict[str, int] = {}
        for meta in body["items"].values():
            per_source[meta["source_ref"]] = per_source.get(meta["source_ref"], 0) + 1
        print(f"[resolver] pins frozen: {body['n_items']} items", flush=True)
        for src in sorted(per_source):
            print(f"[resolver]   {src}: {per_source[src]}", flush=True)
        return 0
    triples = pilot_item_ids()
    resolved = resolve_items([iid for _, _, iid in triples], verify_pins=True)
    print(f"[resolver] verified {len(resolved)} pilot items against frozen pins", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
