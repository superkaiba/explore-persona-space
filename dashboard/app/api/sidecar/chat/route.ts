/**
 * Proxy to the local sidecar's /chat SSE stream.
 *
 * Browser POSTs here with a Bearer token minted by /api/chat-token; this
 * route auths the browser via session cookie, then forwards the same
 * Authorization header to the sidecar (the sidecar verifies the HMAC,
 * NOT the cookie). We keep the SSE headers (`text/event-stream`,
 * `cache-control: no-cache`) so the stream flows through.
 *
 * Sidecar URL precedence: SIDECAR_INTERNAL_URL → SIDECAR_URL →
 * NEXT_PUBLIC_SIDECAR_URL → 127.0.0.1:7654 (dev). Production deploys
 * MUST set one of the first three; the in-place 127.0.0.1 fallback only
 * fires in NODE_ENV=development.
 */
import { requireSessionAuth } from "@/lib/auth";
import { checkRateLimit, clientKey } from "@/lib/rate-limit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_SIDECAR_INTERNAL_URL = "http://127.0.0.1:7654";

function absoluteUrl(value: string | undefined) {
  const trimmed = value?.trim();
  return trimmed?.startsWith("http://") || trimmed?.startsWith("https://")
    ? trimmed.replace(/\/+$/, "")
    : null;
}

function sidecarBaseUrl(): string | null {
  const internalUrl = absoluteUrl(process.env.SIDECAR_INTERNAL_URL);
  if (internalUrl) return internalUrl;

  const sidecarUrl = absoluteUrl(process.env.SIDECAR_URL);
  if (sidecarUrl) return sidecarUrl;

  const publicUrl = absoluteUrl(process.env.NEXT_PUBLIC_SIDECAR_URL);
  if (publicUrl) return publicUrl;

  if (process.env.NODE_ENV === "development") return DEFAULT_SIDECAR_INTERNAL_URL;
  return null;
}

export async function POST(request: Request) {
  const user = await requireSessionAuth();
  if (!user) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }
  const rl = checkRateLimit("sidecar-chat", clientKey(request));
  if (!rl.allowed) {
    return Response.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: { "retry-after": String(rl.retryAfterS) } },
    );
  }

  const authorization = request.headers.get("authorization");
  if (!authorization?.toLowerCase().startsWith("bearer ")) {
    return Response.json({ error: "Missing sidecar token" }, { status: 401 });
  }

  const baseUrl = sidecarBaseUrl();
  if (!baseUrl) {
    return Response.json({ error: "Sidecar URL not configured" }, { status: 503 });
  }

  try {
    const upstream = await fetch(`${baseUrl}/chat`, {
      method: "POST",
      headers: {
        authorization,
        "content-type": request.headers.get("content-type") ?? "application/json",
        accept: "text/event-stream",
      },
      body: await request.text(),
      cache: "no-store",
    });

    const headers = new Headers();
    headers.set(
      "content-type",
      upstream.headers.get("content-type") ?? "text/event-stream; charset=utf-8",
    );
    headers.set("cache-control", "no-cache, no-transform");
    headers.set("x-accel-buffering", "no");

    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unable to reach Claude Code sidecar";
    return Response.json({ error: message }, { status: 502 });
  }
}
