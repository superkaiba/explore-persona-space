/**
 * Tell the sidecar to drop a chat session (it kills the corresponding
 * `claude` child process). Called when the user closes a chat tab so we
 * don't leak idle CLI processes on the VM.
 *
 * Browser POSTs here with `{session_id}` JSON; this route auths via
 * session cookie and forwards the request to the sidecar using the
 * shared secret (sidecar's lifecycle endpoint uses the secret, NOT the
 * HMAC token, since session teardown is host-side).
 */
import { requireSessionAuth } from "@/lib/auth";

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

  const secret = process.env.SIDECAR_SHARED_SECRET;
  if (!secret) {
    return Response.json({ error: "Sidecar not configured" }, { status: 503 });
  }
  const baseUrl = sidecarBaseUrl();
  if (!baseUrl) {
    return Response.json({ error: "Sidecar URL not configured" }, { status: 503 });
  }

  const body = await request.text();
  const upstream = await fetch(`${baseUrl}/end-session`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${secret}`,
      "content-type": request.headers.get("content-type") ?? "application/json",
    },
    body,
    cache: "no-store",
  });

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: {
      "content-type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}
