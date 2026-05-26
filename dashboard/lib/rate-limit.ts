/**
 * Tiny in-memory token bucket, per IP. Single-instance only (Next 16
 * default), enough for a mentor-facing chat surface where ~30 req/hr per
 * client is the expected ceiling.
 *
 * Closes critic P0-4 (no rate limit anywhere). Two buckets: chat-token
 * mints (cheap) and sidecar/chat (expensive — each tick spawns Claude).
 */

type Bucket = { tokens: number; lastRefillMs: number };

const buckets = new Map<string, Map<string, Bucket>>();

export type BucketName = "chat-token" | "sidecar-chat";

const CONFIG: Record<BucketName, { capacity: number; refillPerHour: number }> = {
  // 60 mints/hr is generous — every page load mints one.
  "chat-token": { capacity: 60, refillPerHour: 60 },
  // Each sidecar/chat call spawns a Claude session. 30/hr is a hard cap.
  "sidecar-chat": { capacity: 30, refillPerHour: 30 },
};

export function checkRateLimit(
  bucket: BucketName,
  key: string,
): { allowed: boolean; remaining: number; retryAfterS: number } {
  const cfg = CONFIG[bucket];
  let perBucket = buckets.get(bucket);
  if (!perBucket) {
    perBucket = new Map();
    buckets.set(bucket, perBucket);
  }
  const now = Date.now();
  const b = perBucket.get(key) ?? { tokens: cfg.capacity, lastRefillMs: now };
  const elapsedMs = now - b.lastRefillMs;
  const refill = (elapsedMs / 3_600_000) * cfg.refillPerHour;
  b.tokens = Math.min(cfg.capacity, b.tokens + refill);
  b.lastRefillMs = now;
  if (b.tokens < 1) {
    perBucket.set(key, b);
    // Seconds until 1 full token: how long to add (1 - tokens) at refill rate
    const need = 1 - b.tokens;
    const retryAfterS = Math.ceil((need / cfg.refillPerHour) * 3600);
    return { allowed: false, remaining: 0, retryAfterS };
  }
  b.tokens -= 1;
  perBucket.set(key, b);
  return { allowed: true, remaining: Math.floor(b.tokens), retryAfterS: 0 };
}

/** Extract a stable client identifier from a Request. Prefers
 * Cloudflare's connecting-IP, falls back to XFF, then "unknown". */
export function clientKey(request: Request): string {
  const cf = request.headers.get("cf-connecting-ip");
  if (cf) return cf;
  const xff = request.headers.get("x-forwarded-for");
  if (xff) return xff.split(",")[0].trim();
  return "unknown";
}
