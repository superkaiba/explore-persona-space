"""Issue #2658 P4 preparation: freeze the production prompt selection (plan v4
section 4, amended v5/v6).

Writes ``eval_results/issue_2658/production_selection.json`` ONCE: for each
split (dev, test), each of the 11 registered rows and each of the 12 registered
(frame x stratum) cells, the ordered list of exactly ``n_common`` item ids
(read from ``power/production_n.json``, never hard-coded) drawn from that
cell's split-eligible items. Selection is deterministic and content-addressed:
items are ordered by a SHA-256 over (issue, split, row, cell, item_id) whose
seed material includes the frozen frame/split/prompt-pin manifest shas.

This module lives OUTSIDE ``issue2658_frames.py`` deliberately: the freeze
needs ``issue2658_generate``'s ``write_immutable_json`` / ``canonical_json`` /
``load_cap_amendment``, and generate already imports frames, so an in-frames
mode would create an import cycle. The eligibility helpers themselves
(``load_frame_prompts``, ``has_intrinsic_band``, ``stratum_band_of``,
``partition_band_of``, ``cell_cause``) stay in frames and are called here.

Eligibility (plan v4 section 4, plans/v5 A1-A6, plans/v6 A7):
- an item's content superfamily must be assigned to the split by the frozen
  ``split_manifest.json`` (``superfamily_splits``), must not be
  extraction-barred (``barred_superfamilies`` is the extraction-overlap set),
  and for TEST the item must not be a pilot item nor share a superfamily with
  the pilot or dev selection (pilot labels are never test labels).
- cell membership is a PARTITION within (split, row): intrinsic-band items
  (correctness difficulty, keyed assertion strength) sit in their own band;
  wrapper-band bank prompts are partitioned across bands by the SAME sha rule
  the pilot selection used (``F.partition_band_of``). A cross-cell duplicate
  would collide the (prompt_id, response_index) capture keys and duplicate the
  SHA-derived response seeds, so reuse-across-bands is structurally excluded.
- DEV prefers items outside the pilot selection and falls back to pilot items
  only when a cell would otherwise fall short of ``n_common``, recording the
  count reused per cell.
- shortfalls never raise and are never topped up from another split or cell:
  a cell with fewer than ``n_common`` eligible items takes every eligible item
  and records ``shortfall`` with the frames-module cause tag
  (``F.cell_cause``); fewer than the production floor (15, plan v4 section 8)
  is recorded ``below_production_floor``; zero eligible items is ``empty``.

Response seeds are NOT re-derived here: the header records the frozen
``seed_scheme`` (``C.response_seed``, the exact helper the pilot used) plus a
per-split ``requests_sha256`` over the fully expanded (item_id,
response_index, seed) list in cell order, so the 30-seed schedule per prompt
is content-addressed without materializing half a million integers.

Per-item shas: each selected item carries the frame loader's ``prompt_sha256``
(``sha_kind`` "text" for bank/keyed frames where the loader sha IS the prompt
text sha, "group-key" for correctness benchmark frames whose loader sha is
id-addressed because prompt text is resolved lazily through the sha-pinned
vendored #2388 loaders). Generation verifies resolved text against the "text"
shas and against the frozen pilot pin table for every pilot-reused item.

Usage:
    uv run python scripts/issue2658_production_selection.py --freeze-production-selection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_common as C  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_generate as G  # noqa: E402
import issue2658_text_resolver as R  # noqa: E402

SELECTION_SCHEMA = "i2658-production-selection-v1"
SELECTION_REL = "production_selection.json"
ORDER_DOMAIN = "i2658-prodsel"
PRODUCTION_SPLITS = ("dev", "test")
PLAN_VERSION = "v6"


class ProductionSelectionError(C.Issue2658GuardError):
    """Selection freeze or invariant failure (always fail loud)."""


def file_sha256(path: Path) -> str:
    """Chunked sha256 over raw file bytes (frozen-input addressing)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def order_key(seed_material: str, split: str, row: str, cell: str, item_id: str) -> str:
    """The frozen deterministic ordering sha for one candidate item."""
    return hashlib.sha256(
        f"{ORDER_DOMAIN}|{seed_material}|2658|{split}|{row}|{cell}|{item_id}".encode()
    ).hexdigest()


def load_frozen_inputs(eval_root: Path) -> dict[str, Any]:
    """Read + integrity-check every frozen input the selection derives from."""
    frame_path = eval_root / "frame_manifest.json"
    split_path = eval_root / "split_manifest.json"
    pins_path = eval_root / "prompt_pins.json"
    prod_n_path = eval_root / "power" / "production_n.json"
    for p in (frame_path, split_path, pins_path, prod_n_path):
        if not p.is_file():
            raise ProductionSelectionError(f"frozen input missing: {p}")
    frame_body = json.loads(frame_path.read_text())
    split_body = json.loads(split_path.read_text())
    F.assert_manifest_immutable(frame_body)
    F.assert_manifest_immutable(split_body)
    prod_n = json.loads(prod_n_path.read_text())
    n_common = prod_n.get("n_common")
    if not isinstance(n_common, int) or n_common <= 0:
        raise ProductionSelectionError(
            f"power/production_n.json n_common is {n_common!r}, expected a positive int"
        )
    cap_record = G.load_cap_amendment(eval_root)
    if cap_record is None:
        raise ProductionSelectionError(
            f"frozen cap amendment record absent under {eval_root} (plan v6 A7); "
            "production selection is not freezable without the production cap"
        )
    return {
        "frame_body": frame_body,
        "split_body": split_body,
        "n_common": n_common,
        "production_n_sha256": file_sha256(prod_n_path),
        "frame_sha": frame_body["content_sha256"],
        "split_sha": split_body["content_sha256"],
        "pins_sha": file_sha256(pins_path),
        "cap_record": cap_record,
        "cap_amendment_sha256": file_sha256(eval_root / G.CAP_AMENDMENT_REL),
    }


def _registered_cells(row: str) -> list[str]:
    rf = F.FRAMES[row]
    return [f"{fr.name}|{s.name}" for fr in rf.frames for s in rf.strata]


def _sha_kind(row: str, frame_name: str) -> str:
    spec = next(fr for fr in F.FRAMES[row].frames if fr.name == frame_name)
    return "group-key" if spec.source_kind == "benchmark" else "text"


def _row_split_lookup(split_body: dict, row: str) -> tuple[dict[str, str], frozenset[str]]:
    for rr in split_body["rows"]:
        if rr["row"] == row:
            return dict(rr["superfamily_splits"]), frozenset(rr["barred_superfamilies"])
    raise ProductionSelectionError(f"row {row!r} not in split manifest")


def _pilot_ids_and_sfs(frame_rr: dict) -> tuple[frozenset[str], frozenset[str]]:
    ids = frozenset(
        iid for iids in frame_rr["pilot_selection"]["per_cell_item_ids"].values() for iid in iids
    )
    sfs = frozenset(frame_rr["item_superfamily"][iid] for iid in ids)
    return ids, sfs


def _cell_pools(row: str, frame_body: dict) -> dict[str, list[F.PromptItem]]:
    """Band-partitioned full cell pools (before any split filtering).

    Loads the frame items through the SAME loaders the manifest build used and
    fail-louds on any drift between the loaded id set and the frozen
    ``item_superfamily`` map, so the selection can never draw an item the
    manifest does not know.
    """
    rf = F.FRAMES[row]
    frame_rr = next(rr for rr in frame_body["rows"] if rr["row"] == row)
    items: list[F.PromptItem] = []
    for fr in rf.frames:
        items.extend(F.load_frame_prompts(row, fr))
    loaded = {it.item_id for it in items}
    frozen = set(frame_rr["item_superfamily"])
    if loaded != frozen:
        raise ProductionSelectionError(
            f"{row}: loader/manifest drift: {len(loaded - frozen)} loaded-only ids "
            f"(e.g. {sorted(loaded - frozen)[:3]}), {len(frozen - loaded)} frozen-only ids "
            f"(e.g. {sorted(frozen - loaded)[:3]})"
        )
    pools: dict[str, list[F.PromptItem]] = {cell: [] for cell in _registered_cells(row)}
    for it in items:
        if F.has_intrinsic_band(it, row):
            band = F.stratum_band_of(it, row)
        else:
            band = F.partition_band_of(row, it.prompt_sha256)
        pools[f"{it.frame}|{band}"].append(it)
    return pools


def _select_row_split(
    *,
    row: str,
    split: str,
    pools: dict[str, list[F.PromptItem]],
    frame_body: dict,
    split_body: dict,
    n_common: int,
    seed_material: str,
    dev_selected_sfs: frozenset[str],
) -> dict[str, Any]:
    """One (row, split) selection: per registered cell the ordered ids + record."""
    frame_rr = next(rr for rr in frame_body["rows"] if rr["row"] == row)
    sf_splits, barred = _row_split_lookup(split_body, row)
    pilot_ids, pilot_sfs = _pilot_ids_and_sfs(frame_rr)
    out: dict[str, Any] = {}
    for cell in _registered_cells(row):
        pool = pools[cell]
        contributing = {R.superfamily_of(frame_body, row, it.item_id) for it in pool}
        eligible: list[F.PromptItem] = []
        for it in pool:
            sf = R.superfamily_of(frame_body, row, it.item_id)
            assigned = sf_splits.get(sf)
            if assigned is None:
                raise ProductionSelectionError(
                    f"{row}/{cell}: superfamily {sf!r} of {it.item_id!r} has no split "
                    "assignment in the frozen split manifest"
                )
            if assigned != split or sf in barred:
                continue
            if split == "test" and (
                it.item_id in pilot_ids or sf in pilot_sfs or sf in dev_selected_sfs
            ):
                continue
            eligible.append(it)
        key = {
            it.item_id: order_key(seed_material, split, row, cell, it.item_id) for it in eligible
        }
        if split == "dev":
            non_pilot = sorted(
                (it for it in eligible if it.item_id not in pilot_ids), key=lambda i: key[i.item_id]
            )
            pilot_pool = sorted(
                (it for it in eligible if it.item_id in pilot_ids), key=lambda i: key[i.item_id]
            )
            chosen = (non_pilot + pilot_pool)[:n_common]
        else:
            chosen = sorted(eligible, key=lambda i: key[i.item_id])[:n_common]
        n_eligible = len(eligible)
        if n_eligible == 0:
            status = "empty"
        elif n_eligible < F.PRODUCTION_TEST_PROMPTS_PER_CELL_FLOOR:
            status = "below_production_floor"
        elif n_eligible < n_common:
            status = "below_common_n"
        else:
            status = "ok"
        shortfall = None
        if n_eligible < n_common:
            shortfall = {
                "eligible": n_eligible,
                "cause": F.cell_cause(n_eligible, contributing, barred),
            }
        out[cell] = {
            "item_ids": [it.item_id for it in chosen],
            "item_sha256": {it.item_id: it.prompt_sha256 for it in chosen},
            "sha_kind": _sha_kind(row, cell.partition("|")[0]),
            "n_eligible": n_eligible,
            "status": status,
            "shortfall": shortfall,
            "n_pilot_reused": sum(1 for it in chosen if it.item_id in pilot_ids),
        }
    return out


def assert_selection_invariants(body: dict, frame_body: dict) -> None:
    """Fail-loud invariants over a selection body (freeze- AND read-side).

    Per row: dev/test item-id sets disjoint, dev/test superfamily sets
    disjoint, no test item or test superfamily in the pilot selection, every
    selected id resolves in the frame manifest for its row, every registered
    cell key present for every row and split, no item in two cells of one
    split, and per-cell record consistency (count vs status vs sha map).
    """
    n_common = body["n_common"]
    frame_rows = {rr["row"]: rr for rr in frame_body["rows"]}
    for split in PRODUCTION_SPLITS:
        if split not in body["splits"]:
            raise ProductionSelectionError(f"selection missing split {split!r}")
        if set(body["splits"][split]) != set(C.ROW_IDS):
            raise ProductionSelectionError(
                f"{split}: selection rows != registered ROW_IDS "
                f"({sorted(set(body['splits'][split]) ^ set(C.ROW_IDS))})"
            )
    for row in C.ROW_IDS:
        frame_rr = frame_rows.get(row)
        if frame_rr is None:
            raise ProductionSelectionError(f"row {row!r} not in frame manifest")
        pilot_ids, pilot_sfs = _pilot_ids_and_sfs(frame_rr)
        sf_of = frame_rr["item_superfamily"]
        per_split_ids: dict[str, set[str]] = {}
        per_split_sfs: dict[str, set[str]] = {}
        for split in PRODUCTION_SPLITS:
            cells = body["splits"][split][row]
            if set(cells) != set(_registered_cells(row)):
                raise ProductionSelectionError(
                    f"{split}/{row}: cell keys != registered cells "
                    f"({sorted(set(cells) ^ set(_registered_cells(row)))})"
                )
            seen: set[str] = set()
            for cell, rec in cells.items():
                iids = rec["item_ids"]
                if len(iids) > n_common:
                    raise ProductionSelectionError(
                        f"{split}/{row}/{cell}: {len(iids)} ids exceed n_common {n_common}"
                    )
                if (rec["status"] == "empty") != (len(iids) == 0):
                    raise ProductionSelectionError(
                        f"{split}/{row}/{cell}: status {rec['status']!r} inconsistent with "
                        f"{len(iids)} selected ids"
                    )
                if rec["status"] == "ok" and len(iids) != n_common:
                    raise ProductionSelectionError(
                        f"{split}/{row}/{cell}: status ok with {len(iids)} != {n_common} ids"
                    )
                if set(rec["item_sha256"]) != set(iids):
                    raise ProductionSelectionError(
                        f"{split}/{row}/{cell}: item_sha256 keys != item_ids"
                    )
                for iid in iids:
                    if iid not in sf_of:
                        raise ProductionSelectionError(
                            f"{split}/{row}/{cell}: {iid!r} not in the frame manifest row"
                        )
                    if iid in seen:
                        raise ProductionSelectionError(
                            f"{split}/{row}: {iid!r} selected in two cells (capture keys and "
                            "response seeds are (item, draw)-keyed, so this is forbidden)"
                        )
                    seen.add(iid)
            per_split_ids[split] = seen
            per_split_sfs[split] = {sf_of[iid] for iid in seen}
        inter_ids = per_split_ids["dev"] & per_split_ids["test"]
        if inter_ids:
            raise ProductionSelectionError(
                f"{row}: dev/test item ids overlap ({len(inter_ids)}, e.g. {sorted(inter_ids)[:3]})"
            )
        inter_sfs = per_split_sfs["dev"] & per_split_sfs["test"]
        if inter_sfs:
            raise ProductionSelectionError(
                f"{row}: dev/test superfamilies overlap ({len(inter_sfs)}, "
                f"e.g. {sorted(inter_sfs)[:3]})"
            )
        test_pilot = per_split_ids["test"] & pilot_ids
        if test_pilot:
            raise ProductionSelectionError(
                f"{row}: pilot items in TEST selection ({sorted(test_pilot)[:3]})"
            )
        test_pilot_sfs = per_split_sfs["test"] & pilot_sfs
        if test_pilot_sfs:
            raise ProductionSelectionError(
                f"{row}: pilot superfamilies in TEST selection ({sorted(test_pilot_sfs)[:3]})"
            )


def _split_totals(body_splits: dict[str, Any], responses_per_prompt: int) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for split, rows in body_splits.items():
        n_ok = n_short = n_floor = n_empty = n_items = 0
        requests: list[tuple[str, int, int]] = []
        for row in sorted(rows):
            for cell in sorted(rows[row]):
                rec = rows[row][cell]
                st = rec["status"]
                n_ok += st == "ok"
                n_short += st == "below_common_n"
                n_floor += st == "below_production_floor"
                n_empty += st == "empty"
                n_items += len(rec["item_ids"])
                requests.extend(
                    (iid, k, C.response_seed(iid, k))
                    for iid in rec["item_ids"]
                    for k in range(responses_per_prompt)
                )
        req_sha = hashlib.sha256(
            json.dumps(requests, sort_keys=False, separators=(",", ":")).encode()
        ).hexdigest()
        totals[split] = {
            "cells_selected_at_n_common": n_ok,
            "cells_below_common_n": n_short,
            "cells_below_production_floor": n_floor,
            "cells_empty": n_empty,
            "n_items": n_items,
            "n_requests": n_items * responses_per_prompt,
            "requests_sha256": req_sha,
        }
    return totals


def build_selection(eval_root: Path | None = None) -> dict[str, Any]:
    """Deterministic selection body over the frozen inputs (no file writes)."""
    root = Path(eval_root) if eval_root is not None else F.OUT_DIR
    fin = load_frozen_inputs(root)
    frame_body, split_body = fin["frame_body"], fin["split_body"]
    n_common = fin["n_common"]
    seed_material = f"{fin['frame_sha']}|{fin['split_sha']}|{fin['pins_sha']}"
    responses = int(C.DECODER["n_responses_per_prompt_production"])
    splits_out: dict[str, Any] = {"dev": {}, "test": {}}
    for rr in frame_body["rows"]:
        row = rr["row"]
        pools = _cell_pools(row, frame_body)
        dev_sel = _select_row_split(
            row=row,
            split="dev",
            pools=pools,
            frame_body=frame_body,
            split_body=split_body,
            n_common=n_common,
            seed_material=seed_material,
            dev_selected_sfs=frozenset(),
        )
        dev_sfs = frozenset(
            R.superfamily_of(frame_body, row, iid)
            for rec in dev_sel.values()
            for iid in rec["item_ids"]
        )
        test_sel = _select_row_split(
            row=row,
            split="test",
            pools=pools,
            frame_body=frame_body,
            split_body=split_body,
            n_common=n_common,
            seed_material=seed_material,
            dev_selected_sfs=dev_sfs,
        )
        splits_out["dev"][row] = dev_sel
        splits_out["test"][row] = test_sel

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    body: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "issue": 2658,
        "plan_version": PLAN_VERSION,
        "n_common": n_common,
        "production_n_sha256": fin["production_n_sha256"],
        "frame_manifest_content_sha256": fin["frame_sha"],
        "split_manifest_content_sha256": fin["split_sha"],
        "prompt_pins_sha256": fin["pins_sha"],
        "production_max_new_tokens": int(fin["cap_record"]["production_max_new_tokens"]),
        "cap_amendment_sha256": fin["cap_amendment_sha256"],
        "responses_per_prompt": responses,
        "seed_scheme": C.DECODER["seed_scheme"],
        "production_floor": F.PRODUCTION_TEST_PROMPTS_PER_CELL_FLOOR,
        "order_rule": (
            f"per cell ascending sha256('{ORDER_DOMAIN}|<frame_sha>|<split_sha>|<pins_sha>"
            "|2658|<split>|<row>|<cell>|<item_id>') truncated to n_common; dev lists "
            "non-pilot items first with pilot items as recorded fallback"
        ),
        "band_rule": (
            "intrinsic bands via issue2658_frames.stratum_band_of; wrapper bands "
            "partitioned by issue2658_frames.partition_band_of (the pilot rule)"
        ),
        "totals": _split_totals(splits_out, responses),
        "splits": splits_out,
        "metadata": as_metadata_dict(git_provenance(), phase="production-selection-freeze"),
    }
    addressable = {k: v for k, v in body.items() if k not in ("metadata", "content_sha256")}
    body["content_sha256"] = F._canonical_sha(addressable)
    assert_selection_invariants(body, frame_body)
    return body


def freeze(eval_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Build + write-once persist the selection (byte-identical rewrite is a no-op)."""
    root = Path(eval_root) if eval_root is not None else F.OUT_DIR
    body = build_selection(root)
    out_path = root / SELECTION_REL
    G.write_immutable_json(out_path, body)
    return out_path, body


def _print_totals(body: dict[str, Any]) -> None:
    for split in PRODUCTION_SPLITS:
        t = body["totals"][split]
        print(
            f"[prodsel] {split}: ok={t['cells_selected_at_n_common']} "
            f"below_common_n={t['cells_below_common_n']} "
            f"below_floor={t['cells_below_production_floor']} empty={t['cells_empty']} "
            f"items={t['n_items']} requests={t['n_requests']}",
            flush=True,
        )
        for row in sorted(body["splits"][split]):
            cells = body["splits"][split][row]
            short = [
                f"{c}:{r['n_eligible']}" for c, r in sorted(cells.items()) if r["status"] != "ok"
            ]
            line = f"[prodsel]   {row}: " + (
                "all cells at n_common" if not short else "short " + ", ".join(short)
            )
            print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--freeze-production-selection",
        action="store_true",
        help="build + write-once persist the production selection",
    )
    ap.add_argument("--eval-root", default=None, help="override eval root (tests only)")
    args = ap.parse_args(argv)
    if not args.freeze_production_selection:
        ap.error("nothing to do: pass --freeze-production-selection")
    root = Path(args.eval_root) if args.eval_root else None
    out_path, body = freeze(root)
    _print_totals(body)
    print(
        f"[prodsel] frozen -> {out_path} n_common={body['n_common']} "
        f"content_sha={body['content_sha256'][:16]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
