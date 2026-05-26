/**
 * Mint a short-lived HMAC-signed token the browser can use to call the
 * sidecar (either through /api/sidecar/chat or directly when
 * NEXT_PUBLIC_SIDECAR_URL is set).
 *
 *   POST /api/chat-token
 *     auth: session cookie (or dev-stub when DASHBOARD_AUTH_ENABLED!=true)
 *     -> 200 {token, sidecar_url, expires_at}
 *     -> 401 if not signed in
 *
 * Token format:  base64url(<exp_ms>) "." base64url(hmac_sha256(SECRET, exp_ms_str))
 * Sidecar verifies by recomputing the HMAC + checking the exp_ms.
 *
 * Runtime is `nodejs` (not edge) so this route shares the same auth
 * helpers as the rest of the dashboard. `requireSessionAuth` reads
 * `next/headers.cookies()` which works in both runtimes, but jose uses
 * Node crypto when available — keeping nodejs avoids surprises.
 */
import { requireSessionAuth } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TOKEN_TTL_MS = 5 * 60 * 1000; // 5 minutes

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

export async function POST() {
  const user = await requireSessionAuth();
  if (!user) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  const secret = process.env.SIDECAR_SHARED_SECRET;
  // For sidecar_url returned to the client: prefer a same-origin proxy
  // path (`/api/sidecar`) in dev so the browser doesn't have to reach
  // 127.0.0.1 directly. When NEXT_PUBLIC_SIDECAR_URL is set (eventual
  // chat.superkaiba.com), use that.
  const sidecarUrl =
    process.env.NEXT_PUBLIC_SIDECAR_URL && process.env.NEXT_PUBLIC_SIDECAR_URL.trim().length > 0
      ? process.env.NEXT_PUBLIC_SIDECAR_URL.replace(/\/+$/, "")
      : "/api/sidecar";
  if (!secret) {
    return Response.json({ error: "Sidecar not configured" }, { status: 503 });
  }

  const expMs = Date.now() + TOKEN_TTL_MS;
  const payload = String(expMs);
  const payloadB64 = b64url(new TextEncoder().encode(payload));
  const sig = await hmacSign(secret, payload);
  const token = `${payloadB64}.${sig}`;

  return Response.json({
    token,
    sidecar_url: sidecarUrl,
    expires_at: expMs,
  });
}
