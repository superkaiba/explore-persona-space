/**
 * Optional auth gate.
 *
 * When DASHBOARD_AUTH_ENABLED=true, require the session cookie on:
 *   - /updates  + /updates/*
 *   - /log      + /log/*
 *   - /api/sidecar/*
 *   - /api/chat-token
 *   - /api/log/*
 *
 * Otherwise everything passes through. Next 16 runs proxy/middleware
 * on the nodejs runtime (per upgrade docs), so jose's full crypto API
 * is available here just like in the API routes that import lib/auth.
 *
 * Anything outside the gated routes (e.g. /tasks/[id]/edit) is still
 * protected by `isEditorAuthed()` in lib/auth.ts, which reads the same
 * site-password session cookie.
 */
import { NextResponse, type NextRequest } from "next/server";
import { jwtVerify } from "jose";

const SESSION_COOKIE = "eps_session";

function authEnabled(): boolean {
  return process.env.DASHBOARD_AUTH_ENABLED === "true";
}

function authSecretBytes(): Uint8Array | null {
  const s = process.env.AUTH_SECRET;
  if (!s || s.length < 32) return null;
  return new TextEncoder().encode(s);
}

export async function proxy(request: NextRequest) {
  if (!authEnabled()) return NextResponse.next();

  const secret = authSecretBytes();
  if (!secret) {
    // Misconfigured: auth is on but no secret. Fail closed.
    const url = new URL("/sign-in?error=auth_misconfigured", request.url);
    return NextResponse.redirect(url);
  }

  const cookie = request.cookies.get(SESSION_COOKIE)?.value;
  if (!cookie) {
    return redirectToSignIn(request);
  }
  try {
    const { payload } = await jwtVerify(cookie, secret);
    if (
      typeof payload.sub !== "string" ||
      (payload as { kind?: unknown }).kind !== "session"
    ) {
      return redirectToSignIn(request);
    }
  } catch {
    return redirectToSignIn(request);
  }
  return NextResponse.next();
}

function redirectToSignIn(request: NextRequest) {
  // For API routes, return 401 JSON instead of redirecting.
  if (request.nextUrl.pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const signIn = new URL("/sign-in", request.url);
  signIn.searchParams.set("next", request.nextUrl.pathname + request.nextUrl.search);
  return NextResponse.redirect(signIn);
}

export const config = {
  matcher: [
    "/updates",
    "/updates/:path*",
    "/log",
    "/log/:path*",
    "/api/sidecar/:path*",
    "/api/chat-token",
    "/api/log/:path*",
  ],
};
