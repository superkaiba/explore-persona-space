#!/usr/bin/env python3
"""Issue #617 Step 3 adapter: cluster sweep -> #594-shaped extraction battery.

Per plan §4 step 3 (M1 + M2). Converts the swept clusters into a battery JSON
that the #594 extractor consumes via ``--schema issue617`` (loose validation).

M2 (unique-prefix pool cap = 400): a ``conv_id`` recurs across configs, so we
DEDUP BY ``conv_id`` and extract each conversation ONCE; cluster membership is
a SEPARATE map (``conv_id -> {config: cluster_id}``) consumed at scoring time.
If the dedup'd union exceeds 400, subsample DOWN to 400 deterministically
(seed 42), stratified to keep >= 30 members per cluster that appears in any
config. A cluster that cannot retain >= 30 extracted members is dropped from
the pair pool (recorded).

M1 (mandatory synthetic default instance): the #594 extractor's FULL path
hard-codes ``f6_default_template`` lookups (lines 181, 383, 516-518) that crash
a #617 battery with ``StopIteration`` AFTER GPU spin-up. We inject exactly ONE
synthetic instance ``id="f6_default_template"`` (bare default-assistant prefix,
#594 default-template schema), extracted alongside the WildChat prefixes and
EXCLUDED from the cluster pair pool.

The battery ``meta.probe_pool_hash`` is the Betley pool hash so the extractor's
default-path probe assert (line ~464) passes unchanged.

Usage::

    uv run python scripts/issue617_build_extraction_battery.py
    # smoke: tiny cap + floor
    uv run python scripts/issue617_build_extraction_battery.py --pool-cap 12 --per-cluster-floor 3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from issue404_common import (  # noqa: E402
    fetch_betley_main_8,
    fetch_preregistered_probes,
    reproducibility_metadata,
)
from issue594_common import (  # noqa: E402
    BATTERY_SCHEMA_VERSION,
    probes_hash,
    validate_battery_loose,
)
from issue617_common import (  # noqa: E402
    CLUSTER_MEMBERSHIP_PATH,
    CLUSTER_PATH,
    DATA_DIR,
    EXTRACTION_BATTERY_PATH,
    EXTRACTION_POOL_CAP,
    PER_CLUSTER_FLOOR,
    SEED,
    SLICE_PATH,
    SYNTHETIC_DEFAULT_ID,
)

load_dotenv()

logger = logging.getLogger("issue617_battery")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _config_cluster_id(algo: str, label: int) -> str | None:
    """Stable per-(config, cluster) id; None for HDBSCAN noise (label -1)."""
    if label < 0:
        return None
    return f"{algo}_c{label:02d}"


def build_membership(
    cluster_payload: dict,
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """conv_id -> {config: cluster_id} map, and cluster_id -> [conv_id] roster.

    HDBSCAN noise (-1) is dropped from BOTH maps (excluded from the pair pool).
    """
    conv_to_clusters: dict[str, dict[str, str]] = {}
    cluster_members: dict[str, list[str]] = {}
    for algo, cfg in cluster_payload["configs"].items():
        for conv_id, label in cfg["labels"].items():
            cid = _config_cluster_id(algo, int(label))
            if cid is None:
                continue
            conv_to_clusters.setdefault(conv_id, {})[algo] = cid
            cluster_members.setdefault(cid, []).append(conv_id)
    return conv_to_clusters, cluster_members


def select_pool(
    conv_to_clusters: dict[str, dict[str, str]],
    cluster_members: dict[str, list[str]],
    pool_cap: int,
    per_cluster_floor: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Stratified <= pool_cap selection of conv_ids; returns (selected, dropped_clusters).

    Every conv_id that appears in ANY non-noise cluster is a candidate. If the
    candidate union <= pool_cap, take it whole. Otherwise stratified-subsample
    to pool_cap (seed 42), greedily ensuring each cluster keeps >= floor of its
    members; clusters that cannot retain >= floor extracted members are
    recorded as dropped (excluded from the pair pool at scoring time).
    """
    rng = random.Random(seed)
    candidates = sorted(conv_to_clusters.keys())
    if len(candidates) <= pool_cap:
        selected = candidates
    else:
        # Round-robin over clusters (sorted by ascending size so the rarest
        # clusters reach their floor first), drawing members until pool_cap.
        ordered_clusters = sorted(cluster_members, key=lambda c: len(cluster_members[c]))
        remaining = {
            c: rng.sample(cluster_members[c], len(cluster_members[c])) for c in ordered_clusters
        }
        selected_set: set[str] = set()
        # Phase 1: guarantee the floor per cluster where possible.
        for c in ordered_clusters:
            for conv_id in remaining[c]:
                if len([x for x in selected_set if c in conv_to_clusters[x]]) >= per_cluster_floor:
                    break
                if len(selected_set) >= pool_cap:
                    break
                selected_set.add(conv_id)
        # Phase 2: fill the rest round-robin.
        idx = 0
        pools = [remaining[c] for c in ordered_clusters]
        while len(selected_set) < pool_cap:
            progressed = False
            for pool in pools:
                if idx < len(pool):
                    selected_set.add(pool[idx])
                    progressed = True
                    if len(selected_set) >= pool_cap:
                        break
            idx += 1
            if not progressed:
                break
        selected = sorted(selected_set)

    # Identify clusters whose extracted membership falls below the floor.
    selected_set = set(selected)
    dropped: list[str] = []
    for cid, members in cluster_members.items():
        kept = sum(1 for m in members if m in selected_set)
        if kept < per_cluster_floor:
            dropped.append(cid)
    if dropped:
        logger.warning(
            "%d clusters drop below the per-cluster floor (%d) after the %d-cap and are "
            "excluded from the pair pool: %s",
            len(dropped),
            per_cluster_floor,
            pool_cap,
            sorted(dropped),
        )
    return selected, sorted(dropped)


def synthetic_default_instance() -> dict:
    """The M1 synthetic ``f6_default_template`` instance (bare default-assistant).

    Matches the #594 default-template schema: no system prompt, no prior turns.
    The extractor uses it as the no-context length reference; it is EXCLUDED
    from the cluster pair pool at scoring time.
    """
    return {
        "id": SYNTHETIC_DEFAULT_ID,
        "family": SYNTHETIC_DEFAULT_ID,
        "sub_label": "default",
        "label": "default (template)",
        "system_prompt": None,
        "prefix_messages": [],
        "source": "synthetic bare default-assistant (M1; satisfies the #594 extractor's "
        "f6_default_template lookups; excluded from the #617 cluster pair pool)",
        "meta": {"_synthetic": True, "note": "Qwen2.5 template injects its built-in default"},
    }


def build_battery(
    slice_payload: dict,
    cluster_payload: dict,
    pool_cap: int,
    per_cluster_floor: int,
    seed: int,
) -> tuple[dict, dict]:
    """Build the #617 extraction battery + the cluster-membership map.

    Returns (battery_payload, membership_payload). The battery's
    prefix_messages are the conversation's short_prefix_msgs (user+assistant).
    """
    convs_by_id = {c["conv_id"]: c for c in slice_payload["conversations"]}
    conv_to_clusters, cluster_members = build_membership(cluster_payload)
    selected, dropped = select_pool(
        conv_to_clusters, cluster_members, pool_cap, per_cluster_floor, seed
    )
    logger.info(
        "Selected %d unique conv_ids for extraction (cap %d) over %d candidate clusters; "
        "%d clusters dropped below floor",
        len(selected),
        pool_cap,
        len(cluster_members),
        len(dropped),
    )

    # Betley probe pool hash for the extractor's default-path assert.
    main8 = set(fetch_betley_main_8())
    probes = fetch_preregistered_probes(n=200, exclude=main8)
    pph = probes_hash(probes)

    dataset_name = slice_payload["meta"]["dataset"]
    instances: list[dict] = [synthetic_default_instance()]
    for conv_id in selected:
        conv = convs_by_id[conv_id]
        instances.append(
            {
                "id": conv_id,
                "family": conv_id,  # per-conversation tag; cluster mapped separately
                "sub_label": "wildchat_cluster",
                "label": conv["first_user"][:60],
                "system_prompt": None,
                "prefix_messages": conv["short_prefix_msgs"],
                "source": f"{dataset_name} (tier 1 real-world); #617 cluster pool",
                "meta": {
                    "conv_id": conv_id,
                    "content_tokens": conv["content_tokens"],
                    "n_exchanges": conv["n_exchanges"],
                    "cluster_membership": conv_to_clusters.get(conv_id, {}),
                },
            }
        )

    battery = {
        "schema_version": BATTERY_SCHEMA_VERSION,
        "meta": {
            "issue": 617,
            "probe_pool_hash": pph,
            "probe_pool_n": len(probes),
            "pool_cap": pool_cap,
            "per_cluster_floor": per_cluster_floor,
            "n_extracted": len(selected),
            "synthetic_default_id": SYNTHETIC_DEFAULT_ID,
            "seed": seed,
            "metadata": reproducibility_metadata({"script": "issue617_build_extraction_battery"}),
        },
        "instances": instances,
    }
    validate_battery_loose(battery)

    membership = {
        "meta": {
            "issue": 617,
            "n_extracted": len(selected),
            "dropped_clusters_below_floor": dropped,
            "per_cluster_floor": per_cluster_floor,
            "pool_cap": pool_cap,
            "pooling_mode": cluster_payload["meta"].get("pooling_mode"),
            "embedder": cluster_payload["meta"].get("embedder"),
            "metadata": reproducibility_metadata(
                {"script": "issue617_build_extraction_battery", "module": "membership"}
            ),
        },
        # conv_id -> {config: cluster_id}, restricted to extracted conv_ids.
        "conv_to_clusters": {c: conv_to_clusters[c] for c in selected},
        # cluster_id -> [extracted conv_ids], dropping below-floor clusters.
        "cluster_members": {
            cid: [m for m in members if m in set(selected)]
            for cid, members in cluster_members.items()
            if cid not in set(dropped)
        },
        "cluster_examples": {
            algo: cfg["examples"] for algo, cfg in cluster_payload["configs"].items()
        },
        "synthetic_default_id": SYNTHETIC_DEFAULT_ID,
    }
    return battery, membership


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #617 Step 3: build the extraction battery.")
    parser.add_argument("--slice", type=Path, default=SLICE_PATH)
    parser.add_argument("--clusters", type=Path, default=CLUSTER_PATH)
    parser.add_argument("--out-battery", type=Path, default=EXTRACTION_BATTERY_PATH)
    parser.add_argument("--out-membership", type=Path, default=CLUSTER_MEMBERSHIP_PATH)
    parser.add_argument("--pool-cap", type=int, default=EXTRACTION_POOL_CAP)
    parser.add_argument("--per-cluster-floor", type=int, default=PER_CLUSTER_FLOOR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    with open(args.slice) as f:
        slice_payload = json.load(f)
    with open(args.clusters) as f:
        cluster_payload = json.load(f)

    battery, membership = build_battery(
        slice_payload, cluster_payload, args.pool_cap, args.per_cluster_floor, args.seed
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    args.out_battery.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_battery, "w") as f:
        json.dump(battery, f, ensure_ascii=False)
    with open(args.out_membership, "w") as f:
        json.dump(membership, f, ensure_ascii=False)
    logger.info(
        "Wrote %s (%d instances incl. synthetic) + %s (%d clusters)",
        args.out_battery,
        len(battery["instances"]),
        args.out_membership,
        len(membership["cluster_members"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
