---
name: hub-repack-plan-review
description: "Review pattern for HF-repo repack/deletion infra plans (#2321): verification-chain TOCTOU, commit-unit arithmetic, tier-option costing"
metadata:
  type: feedback
---

Checks that mattered on the #2321 data-repo repack plan (verification lens on a `kind: infra` deletion task). Reusable for the named next domino (model-repo `superkaiba1/explore-persona-space` repack, 188k files).

**Why:** a repack plan's "measurement" is its byte-exactness chain + slots accounting; the recurring soft spots were all recomputable from the inventory JSON in minutes.

**How to apply:**
1. **Server-anchor TOCTOU:** download-time anchor check (git-blob sha1 == `blob_id` non-LFS; `lfs.sha256` LFS) + round-trip-vs-downloaded-copy leaves ONE window — local staged file between the download check and the pack read. Ask the pack phase to re-assert the census anchor on the bytes it actually packs (free: bytes already in memory + hashed). Not blocking when originals stay in git history (post-hoc detectable + recoverable), but always worth the concern.
2. **Commit-unit arithmetic:** recompute units from the inventory — with ops≤N caps, tiny-file prefixes bind at ONE max-member shard per unit (4,001 ops of 4,500), so unit count ≈ shard count there; group (leaf-dir) size distribution can pull it back down. Plan-stated commit counts are basis-fragile ±30%; check pacing/self-cap/wall bands absorb the top of the band.
3. **Net-negative assert edge:** a unit of 1-member shards (members near the shard byte cap) is net-0 and trips `len(dels) > len(adds)` — fail-loud, but the composer should mix shard sizes.
4. **Tier-option costing (user-facing approval-gate options):** recompute declined tiers' slot/byte figures from the inventory — #2321 overstated tier-C b64 net (+23.5k stated vs ~+19.9k at 3 members/shard), tier-E file count (30,300 stated vs 10,399 = remaining LFS), and a "≈6×" byte ratio that is ~4× like-for-like. Direction favored the (correct) recommendation, so concerns not Must-Fix — but these are exactly the numbers the user decides on.
5. **Renegotiated acceptance bars are OK when explicit + costed at the approval gate** (bar derived pre-measurement; the measured split makes the literal bar cost 4-6× bytes for +4.5% benefit) — that is honest renegotiation, not retrofitting.
6. **Shared-repo headline:** the global end file-count is a live quantity other sessions move; per-prefix before/after scoped counts are the defensible freed-slots attribution — require per-prefix reporting (plan had it).
