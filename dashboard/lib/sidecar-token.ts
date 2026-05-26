/**
 * Server-side helpers for talking to the local Claude Code sidecar
 * (the long-running daemon at 127.0.0.1:7654 in dev, or wherever
 * `SIDECAR_INTERNAL_URL` / `SIDECAR_URL` / `NEXT_PUBLIC_SIDECAR_URL`
 * point in prod).
 *
 * Two things this module owns:
 *
 *   1. `mintSidecarToken()` — produces the same HMAC-signed bearer token
 *      that `/api/chat-token` hands to the browser, but callable from
 *      OTHER server routes that need to drive the sidecar without a user
 *      round-trip. Token format: `base64url(exp_ms).hmac_sha256(secret, exp_ms_str)`.
 *      Sidecar verifies by recomputing the HMAC + checking exp_ms.
 *
 *   2. `sidecarBaseUrl()` — single source for the sidecar URL precedence
 *      (`SIDECAR_INTERNAL_URL` → `SIDECAR_URL` → `NEXT_PUBLIC_SIDECAR_URL`
 *      → 127.0.0.1:7654 in dev). Previously copy-pasted across
 *      `/api/chat-token`, `/api/sidecar/chat`, and `/api/sidecar/end-session`.
 *
 * This module is server-only (uses `process.env`). It does NOT live in
 * `lib/sidecar-client.ts` (browser-facing fetch helper); keep them split.
 *
 * No `import "server-only"` because the dashboard doesn't depend on that
 * package; consumers should only import this from route handlers / server
 * actions, never from `"use client"` files.
 */

const TOKEN_TTL_MS = 5 * 60 * 1000; // 5 minutes — same as /api/chat-token
const DEFAULT_SIDECAR_INTERNAL_URL = "http://127.0.0.1:7654";

function b64url(bytes: ArrayBuffer | Uint8Array): string {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let s = "";
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function hmacSign(secret: string, message: string): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(message));
  return b64url(sig);
}

export type SidecarTokenResult =
  | { ok: true; token: string; expiresAt: number; baseUrl: string }
  | { ok: false; reason: "no_secret" | "no_url" };

/**
 * Mint a fresh sidecar bearer token. Returns `{ok:false}` (no throw)
 * when the sidecar isn't configured for this environment — callers can
 * skip the sidecar call quietly. Token TTL = 5 min; mint a new one for
 * each request rather than caching.
 */
export async function mintSidecarToken(): Promise<SidecarTokenResult> {
  const secret = process.env.SIDECAR_SHARED_SECRET;
  if (!secret) return { ok: false, reason: "no_secret" };
  const baseUrl = sidecarBaseUrl();
  if (!baseUrl) return { ok: false, reason: "no_url" };

  const expMs = Date.now() + TOKEN_TTL_MS;
  const payload = String(expMs);
  const payloadB64 = b64url(new TextEncoder().encode(payload));
  const sig = await hmacSign(secret, payload);
  return { ok: true, token: `${payloadB64}.${sig}`, expiresAt: expMs, baseUrl };
}

function absoluteUrl(value: string | undefined) {
  const trimmed = value?.trim();
  return trimmed?.startsWith("http://") || trimmed?.startsWith("https://")
    ? trimmed.replace(/\/+$/, "")
    : null;
}

/**
 * Resolve the sidecar base URL. Same precedence as
 * `/api/sidecar/chat/route.ts` and `/api/sidecar/end-session/route.ts`:
 *   SIDECAR_INTERNAL_URL → SIDECAR_URL → NEXT_PUBLIC_SIDECAR_URL
 *   → http://127.0.0.1:7654 (NODE_ENV=development only)
 *
 * Returns `null` outside of dev when nothing is configured.
 */
export function sidecarBaseUrl(): string | null {
  const internalUrl = absoluteUrl(process.env.SIDECAR_INTERNAL_URL);
  if (internalUrl) return internalUrl;

  const sidecarUrl = absoluteUrl(process.env.SIDECAR_URL);
  if (sidecarUrl) return sidecarUrl;

  const publicUrl = absoluteUrl(process.env.NEXT_PUBLIC_SIDECAR_URL);
  if (publicUrl) return publicUrl;

  if (process.env.NODE_ENV === "development") return DEFAULT_SIDECAR_INTERNAL_URL;
  return null;
}
